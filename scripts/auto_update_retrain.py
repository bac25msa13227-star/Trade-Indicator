#!/usr/bin/env python3
"""
auto_update_retrain.py — Combo #133 automatic data update + model retrain.

Workflow:
  1. Fetch missing M5 bars from yfinance (GC=F × ratio 0.9952) and append to
     src/xauusd_ai/real_data/XAUUSDm_M5.csv.
  2. Resample updated M5 into M15 / H1 / H4 / D1 CSVs.
  3. Load retrain state from outputs/combo133_retrain_state.json.
  4. If new M5 bars since last retrain >= RETRAIN_EVERY_BARS (6 000 ≈ 20 trading days),
     train a fresh Combo-#133 ensemble on the latest TRAIN_BARS rows, evaluate on
     the next TEST_BARS rows, and save artifacts if the fold is profitable.
  5. Update state file and append a line to outputs/combo133_retrain_log.jsonl.

Usage:
  cd "/path/to/Trade Indicator"
  python scripts/auto_update_retrain.py          # full run
  python scripts/auto_update_retrain.py --data-only   # only fetch/append data
  python scripts/auto_update_retrain.py --train-only  # skip data fetch, just check retrain

Cron (macOS / Linux) — run every day at 06:00 UTC:
  0 6 * * * cd "/path/to/Trade Indicator" && .venv/bin/python scripts/auto_update_retrain.py

Notes:
  * yfinance M5 data is only available for the last ~60 days.  Running at least
    once a month keeps the CSV fully up to date.
  * The model artifacts saved here (acc1_combo133_202604_model.pkl etc.) are identical in
    format to the trainer.py pipeline — live bots reload them automatically.
  * A new model is saved ONLY when the validation fold shows net P&L > 0 (safety
    gate).  The previous model is kept as *.bak if a new one is accepted.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# ── project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.dataset import FEATURE_COLUMNS, prepare_training_dataset
from xauusd_ai.strategies.hybrid import HybridStrategy

# ── constants ────────────────────────────────────────────────────────────────
CONFIG            = ROOT / "configs" / "xauusd_combo133_best.yaml"
REAL_DATA_DIR     = ROOT / "src" / "xauusd_ai" / "real_data"
OUTPUTS_DIR       = ROOT / "outputs"
STATE_FILE        = OUTPUTS_DIR / "combo133_retrain_state.json"
RETRAIN_LOG       = OUTPUTS_DIR / "combo133_retrain_log.jsonl"

MODEL_PATH        = OUTPUTS_DIR / "acc1_combo133_202604_model.pkl"
SCALER_PATH       = OUTPUTS_DIR / "acc1_combo133_202604_scaler.pkl"
META_PATH         = OUTPUTS_DIR / "acc1_combo133_202604_meta.json"

# GC=F → XAUUSDm price conversion ratio
GCF_RATIO         = 0.9952

# Combo #133 hyper-parameters (mirror of show_combo133_daily.py)
TRAIN_BARS        = 30_000
TEST_BARS         = 6_000
RETRAIN_EVERY_BARS = TEST_BARS   # retrain every 6 000 new M5 bars (~20 trading days)

MIN_CONF          = 0.70
REQ_TREND         = False
BLOCKED           = [3, 15, 17, 22, 23]
D1_GATE           = False

THRESHOLD_MIN     = 0.40
THRESHOLD_MAX     = 0.78
THRESHOLD_STEP    = 0.01
PREC_FLOOR        = 0.45

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("auto_retrain")

# ─────────────────────────────────────────────────────────────────────────────
# 0. ALERT HELPERS  (Telegram + Email)
# ─────────────────────────────────────────────────────────────────────────────
# Environment variables read at runtime (not required — silently skipped if absent):
#   RETRAIN_TELEGRAM_TOKEN   — Telegram bot token  (fallback: TELEGRAM_BOT_TOKEN)
#   RETRAIN_TELEGRAM_CHAT_ID — Telegram chat ID    (fallback: TELEGRAM_CHAT_ID)
#   RETRAIN_EMAIL_FROM       — sender address
#   RETRAIN_EMAIL_TO         — recipient(s), comma-separated
#   RETRAIN_SMTP_HOST        — default: smtp.gmail.com
#   RETRAIN_SMTP_PORT        — default: 587  (STARTTLS)
#   RETRAIN_SMTP_USER        — SMTP login (usually same as FROM address)
#   RETRAIN_SMTP_PASSWORD    — SMTP app password / token

import os as _os


def _send_telegram_alert(text: str) -> bool:
    """Send a plain-text Telegram message.  Returns True on success."""
    import requests as _req
    token   = _os.getenv("RETRAIN_TELEGRAM_TOKEN") or _os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = _os.getenv("RETRAIN_TELEGRAM_CHAT_ID") or _os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        log.debug("Telegram alert skipped — no token/chat_id configured.")
        return False
    try:
        resp = _req.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
        if resp.ok:
            log.info("Telegram alert sent OK.")
            return True
        log.warning("Telegram alert failed (%s): %s", resp.status_code, resp.text[:120])
        return False
    except Exception as exc:
        log.warning("Telegram alert error: %s", exc)
        return False


def _send_email_alert(subject: str, body: str) -> bool:
    """Send an e-mail via SMTP STARTTLS.  Returns True on success."""
    import smtplib
    from email.mime.text import MIMEText
    sender    = _os.getenv("RETRAIN_EMAIL_FROM", "")
    recipient = _os.getenv("RETRAIN_EMAIL_TO", "")
    smtp_host = _os.getenv("RETRAIN_SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(_os.getenv("RETRAIN_SMTP_PORT", "587"))
    smtp_user = _os.getenv("RETRAIN_SMTP_USER", sender)
    smtp_pass = _os.getenv("RETRAIN_SMTP_PASSWORD", "")
    if not sender or not recipient or not smtp_pass:
        log.debug("Email alert skipped — RETRAIN_EMAIL_FROM/TO/PASSWORD not configured.")
        return False
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = recipient
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as srv:
            srv.ehlo()
            srv.starttls()
            srv.login(smtp_user, smtp_pass)
            srv.sendmail(sender, [r.strip() for r in recipient.split(",")], msg.as_string())
        log.info("Email alert sent → %s", recipient)
        return True
    except Exception as exc:
        log.warning("Email alert error: %s", exc)
        return False


def _notify_retrain(log_entry: dict) -> None:
    """Send retrain result alert via Telegram and Email."""
    fold      = log_entry.get("fold", "?")
    result    = log_entry.get("result", "?")
    pnl       = float(log_entry.get("net_pnl", 0))
    trades    = log_entry.get("n_trades", "?")
    wr        = float(log_entry.get("win_rate", 0))
    auc       = float(log_entry.get("roc_auc", 0))
    thr       = float(log_entry.get("threshold", 0))
    n_feat    = log_entry.get("n_features", "?")
    ts_start  = str(log_entry.get("test_start", ""))[:10]
    ts_end    = str(log_entry.get("test_end", ""))[:10]
    total_bars = log_entry.get("m5_bars_total", "?")
    icon      = "✅" if log_entry.get("accepted") else "❌"

    subject = f"[XAUUSD AI] Retrain Fold #{fold} — {result}"
    body = (
        f"{icon} <b>Combo #133 Retrain Fold #{fold}</b>\n"
        f"Kết quả: <b>{result}</b>\n\n"
        f"Test period : {ts_start} → {ts_end}\n"
        f"Net P&L     : ${pnl:+.2f}\n"
        f"Trades      : {trades}  |  WR: {wr:.1%}\n"
        f"ROC-AUC     : {auc:.4f}\n"
        f"Threshold   : {thr:.2f}  |  Features: {n_feat}\n"
        f"M5 bars     : {total_bars:,}\n\n"
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    plain_body = body.replace("<b>", "").replace("</b>", "")
    _send_telegram_alert(body)
    _send_email_alert(subject, plain_body)


def _notify_data_update(n_new: int, current_count: int, new_bars_since_retrain: int) -> None:
    """Send data-update progress alert (only when new bars are added)."""
    if n_new <= 0:
        return
    remaining = max(0, RETRAIN_EVERY_BARS - new_bars_since_retrain)
    bars_per_day = 276  # ~23h × 12 bars/h for M5
    days_left = remaining / bars_per_day
    from datetime import date, timedelta
    eta_date = (date.today() + timedelta(days=int(days_left))).strftime("%Y-%m-%d")
    body = (
        f"📊 <b>XAUUSD AI — Data Update</b>\n"
        f"Thêm {n_new:,} M5 bars mới\n"
        f"Tổng M5 bars: {current_count:,}\n"
        f"Bars tích lũy: {new_bars_since_retrain:,} / {RETRAIN_EVERY_BARS:,}\n"
        f"Còn thiếu: {remaining:,} bars (~{days_left:.1f} ngày)\n"
        f"Dự kiến retrain: {eta_date}\n"
        f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    _send_telegram_alert(body)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA UPDATE
# ─────────────────────────────────────────────────────────────────────────────

def _load_existing_m5() -> pd.DataFrame:
    """Read existing M5 CSV; return DataFrame with UTC time column."""
    path = REAL_DATA_DIR / "XAUUSDm_M5.csv"
    if not path.exists():
        return pd.DataFrame(columns=["time", "open", "high", "low", "close",
                                     "tick_volume", "spread_points"])
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _fetch_gcf_m5(last_ts: pd.Timestamp | None) -> pd.DataFrame:
    """
    Download GC=F M5 bars from yfinance (max 60-day window).
    Returns new-only bars (after last_ts), normalized, ratio applied.
    """
    log.info("Fetching GC=F M5 from yfinance (period=60d, interval=5m)…")
    try:
        raw = yf.Ticker("GC=F").history(period="60d", interval="5m", auto_adjust=False)
        if raw is None or raw.empty:
            log.warning("yfinance returned empty data for GC=F M5")
            return pd.DataFrame()
    except Exception as exc:
        log.error("yfinance download failed: %s", exc)
        return pd.DataFrame()

    df = _normalize_ohlcv(raw, "M5")
    if last_ts is not None:
        df = df[df["time"] > last_ts]
    log.info("  → %d new M5 bars after %s", len(df), last_ts or "beginning")
    return df.reset_index(drop=True)


def _append_and_save_m5(existing: pd.DataFrame, new_bars: pd.DataFrame) -> pd.DataFrame:
    """Merge new M5 bars into existing, save CSV, return merged DataFrame."""
    if new_bars.empty:
        log.info("No new M5 bars to append.")
        return existing

    merged = pd.concat([existing, new_bars], ignore_index=True)
    merged = merged.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    merged = _add_derived_cols(merged)
    if "real_volume" not in merged.columns:
        merged["real_volume"] = 0.0

    path = REAL_DATA_DIR / "XAUUSDm_M5.csv"
    REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    log.info("Saved %d M5 bars → %s", len(merged), path)
    return merged


def _normalize_ohlcv(raw: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    """Normalize a yfinance DataFrame to standard OHLCV format with UTC time."""
    df = raw.reset_index().rename(columns={
        "Datetime": "time", "Date": "time",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "tick_volume",
    })
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * GCF_RATIO
    df["tick_volume"] = df.get("tick_volume", 0).fillna(0).astype(float)
    df["spread_points"] = 0.0
    df = df[["time", "open", "high", "low", "close", "tick_volume", "spread_points"]]
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


def _add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tick_volume_delta"] = df["tick_volume"].diff().fillna(0)
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["volume_imbalance"] = ((df["close"] - df["open"]).abs() / hl).fillna(0)
    return df


def _load_existing_tf(tf_name: str) -> pd.DataFrame:
    path = REAL_DATA_DIR / f"XAUUSDm_{tf_name}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["time", "open", "high", "low", "close",
                                     "tick_volume", "spread_points"])
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _fetch_tf_direct(
    ticker: yf.Ticker,
    interval: str,
    period: str,
    tf_name: str,
    last_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    """
    Fetch a specific timeframe directly from yfinance.
    Returns new bars only (after last_ts), normalized with GCF_RATIO applied.
    """
    log.info("  Fetching %s (interval=%s period=%s)…", tf_name, interval, period)
    try:
        raw = ticker.history(period=period, interval=interval, auto_adjust=False)
        if raw is None or raw.empty:
            log.warning("  yfinance returned empty for %s", tf_name)
            return pd.DataFrame()
    except Exception as exc:
        log.error("  yfinance fetch failed for %s: %s", tf_name, exc)
        return pd.DataFrame()

    df = _normalize_ohlcv(raw, tf_name)
    if last_ts is not None:
        df = df[df["time"] > last_ts]
    log.info("  → %d new %s bars", len(df), tf_name)
    return df


def _append_and_save_tf(tf_name: str, new_bars: pd.DataFrame) -> int:
    """Append new bars to an existing TF CSV. Returns count of rows appended."""
    if new_bars.empty:
        return 0
    existing = _load_existing_tf(tf_name)
    merged = pd.concat([existing, new_bars], ignore_index=True)
    merged = merged.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    merged = _add_derived_cols(merged)
    if "real_volume" not in merged.columns:
        merged["real_volume"] = 0.0
    out = REAL_DATA_DIR / f"XAUUSDm_{tf_name}.csv"
    REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    log.info("  Saved %d %s bars → %s", len(merged), tf_name, out)
    return len(new_bars)


def _resample_h4_from_h1(new_h1_bars: pd.DataFrame) -> None:
    """
    H4 is not available directly from yfinance — resample from H1.
    Only the new portion is resampled and appended; full CSV is rebuilt
    to keep candle boundaries consistent.
    """
    if new_h1_bars.empty:
        return
    existing_h1 = _load_existing_tf("H1")
    all_h1 = pd.concat([existing_h1, new_h1_bars], ignore_index=True)
    all_h1 = all_h1.sort_values("time").drop_duplicates("time").set_index("time")

    agg = all_h1[["open", "high", "low", "close", "tick_volume"]].resample("4h").agg({
        "open":        "first",
        "high":        "max",
        "low":         "min",
        "close":       "last",
        "tick_volume": "sum",
    }).dropna(subset=["open"]).reset_index()

    agg["spread_points"] = 0.0
    agg = _add_derived_cols(agg)
    out = REAL_DATA_DIR / "XAUUSDm_H4.csv"
    agg.to_csv(out, index=False)
    log.info("  H4 rebuilt from H1 → %d bars → %s", len(agg), out)


def fetch_and_update_data() -> int:
    """
    Main data-update entry point.

    Fetches each timeframe DIRECTLY from yfinance (no resampling for M5/M15/H1/D1)
    to preserve real broker-like OHLC values. Only H4 is resampled from H1 because
    yfinance has no 4h interval.

    yfinance interval limits:
      5m  / 15m / 30m  → 60 days
      1h              → 730 days
      1d              → unlimited

    Returns the number of new M5 bars appended (0 if none).
    """
    log.info("=== DATA UPDATE ===")
    ticker = yf.Ticker("GC=F")

    # ── M5 ────────────────────────────────────────────────────────────────────
    existing_m5 = _load_existing_m5()
    last_m5_ts  = existing_m5["time"].max() if not existing_m5.empty else None
    log.info("  Existing M5: %d bars  last=%s", len(existing_m5), last_m5_ts)

    new_m5 = _fetch_tf_direct(ticker, "5m", "60d", "M5", last_m5_ts)
    n_new_m5 = 0
    if not new_m5.empty:
        merged_m5 = _append_and_save_m5(existing_m5, new_m5)
        n_new_m5 = len(new_m5)
    else:
        log.info("  M5: already up to date.")

    # ── M15 ───────────────────────────────────────────────────────────────────
    last_m15_ts = _load_existing_tf("M15")["time"].max() if (REAL_DATA_DIR / "XAUUSDm_M15.csv").exists() else None
    new_m15 = _fetch_tf_direct(ticker, "15m", "60d", "M15", last_m15_ts)
    _append_and_save_tf("M15", new_m15)

    # ── H1 ────────────────────────────────────────────────────────────────────
    last_h1_ts = _load_existing_tf("H1")["time"].max() if (REAL_DATA_DIR / "XAUUSDm_H1.csv").exists() else None
    new_h1 = _fetch_tf_direct(ticker, "1h", "730d", "H1", last_h1_ts)
    _append_and_save_tf("H1", new_h1)

    # ── H4 — no direct yfinance interval; resample from H1 ───────────────────
    _resample_h4_from_h1(new_h1)

    # ── D1 ────────────────────────────────────────────────────────────────────
    last_d1_ts = _load_existing_tf("D1")["time"].max() if (REAL_DATA_DIR / "XAUUSDm_D1.csv").exists() else None
    new_d1 = _fetch_tf_direct(ticker, "1d", "5y", "D1", last_d1_ts)
    _append_and_save_tf("D1", new_d1)

    if n_new_m5 == 0:
        log.info("All CSVs are already up to date.")
    return n_new_m5


# ─────────────────────────────────────────────────────────────────────────────
# 2. RETRAIN STATE
# ─────────────────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_retrain_m5_count": 0, "last_retrain_date": None, "last_fold": 0}


def _save_state(state: dict) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _log_retrain(entry: dict) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with RETRAIN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN ONE FOLD (Combo #133 pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _build_sample_weights(y: np.ndarray) -> np.ndarray | None:
    pos_c, neg_c = int(y.sum()), int(len(y) - y.sum())
    if pos_c <= 10 or neg_c <= 10:
        return None
    pw = 2.0 * neg_c / pos_c
    class_w = np.where(y == 1, pw, 1.0).astype(float)
    n = len(y)
    decay_half = n * 0.4
    time_w = np.exp(np.log(2) * np.arange(n) / decay_half)
    time_w /= time_w.mean()
    sw = (class_w * time_w).astype(float)
    sw /= sw.mean()
    return sw


def _search_threshold(
    X_tr: np.ndarray, y_tr: np.ndarray, sw_tr: np.ndarray | None
) -> float:
    """Run threshold search on a 70/30 split of training data."""
    n_tr = len(y_tr)
    split = int(n_tr * 0.70)
    hgb = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.02, max_depth=6, min_samples_leaf=25,
        l2_regularization=1.0, max_bins=128,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        random_state=42,
    )
    sw_sub = sw_tr[:split] if sw_tr is not None else None
    hgb.fit(X_tr[:split], y_tr[:split], sample_weight=sw_sub)
    v_proba = hgb.predict_proba(X_tr[split:])[:, 1]
    y_v = y_tr[split:]

    best_thr, best_score = THRESHOLD_MIN, -float("inf")
    safe_thr, safe_prec = THRESHOLD_MAX, -1.0
    for thr in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP):
        preds = (v_proba >= thr).astype(int)
        if preds.sum() < 3:
            continue
        prec = precision_score(y_v, preds, zero_division=0)
        rec  = recall_score(y_v, preds, zero_division=0)
        if rec < 0.05:
            continue
        if prec > safe_prec:
            safe_prec, safe_thr = prec, float(thr)
        if prec < PREC_FLOOR:
            continue
        score = prec * np.sqrt(rec)
        if score > best_score:
            best_score, best_thr = score, float(thr)

    return best_thr if best_score > -float("inf") else safe_thr


def _select_features(
    X_tr: np.ndarray, y_tr: np.ndarray, sw_tr: np.ndarray | None
) -> np.ndarray:
    """Return boolean feature mask via RandomForest importance (drop bottom 30%)."""
    scout = RandomForestClassifier(
        n_estimators=80, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    scout.fit(X_tr, y_tr, sample_weight=sw_tr)
    imp = scout.feature_importances_
    mask = imp >= np.percentile(imp, 30)
    if mask.sum() < 10:
        mask = np.ones(len(imp), dtype=bool)
    return mask


def _train_ensemble(
    X_tr_sel: np.ndarray, y_tr: np.ndarray, sw_tr: np.ndarray | None
) -> VotingClassifier:
    hgb = HistGradientBoostingClassifier(
        max_iter=1000, learning_rate=0.01, max_depth=7, min_samples_leaf=20,
        l2_regularization=1.0, max_bins=128,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=15,
        max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42,
    )
    et = ExtraTreesClassifier(
        n_estimators=200, max_depth=14, min_samples_leaf=10,
        max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42,
    )
    model = VotingClassifier(
        estimators=[("hgb", hgb), ("rf", rf), ("et", et)],
        voting="soft", weights=[3, 2, 1],
    )
    model.fit(X_tr_sel, y_tr, sample_weight=sw_tr)
    return model


def _atomic_write_pickle(obj: object, dest: Path) -> None:
    import os, tempfile
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(obj, fh)
        os.replace(tmp, str(dest))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _backup_model() -> None:
    """Rotate current model artifacts to *.bak before saving new ones."""
    for src in [MODEL_PATH, SCALER_PATH, META_PATH]:
        bak = src.with_suffix(src.suffix + ".bak")
        if src.exists():
            import shutil
            shutil.copy2(src, bak)


def _save_artifacts(
    model: VotingClassifier,
    scaler: StandardScaler,
    feat_mask: np.ndarray,
    threshold: float,
    metrics: dict,
) -> None:
    """Atomically save model / scaler / meta in trainer.py-compatible format."""
    _backup_model()
    _atomic_write_pickle(model, MODEL_PATH)
    _atomic_write_pickle(scaler, SCALER_PATH)
    meta = {
        "decision_threshold": threshold,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_mask": feat_mask.tolist(),
        "roc_auc": round(float(metrics.get("roc_auc", 0.0)), 6),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "train_rows": metrics.get("train_rows"),
        "test_rows": metrics.get("test_rows"),
        "retrain_date": datetime.now(timezone.utc).isoformat(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Artifacts saved → %s", MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# 4. BACKTEST (profit/loss gate)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_live_overrides(sim_settings) -> None:
    sim_settings.risk.min_confidence = MIN_CONF
    sim_settings.strategy.sideway_min_confidence = MIN_CONF
    sim_settings.strategy.volatile_min_confidence = MIN_CONF
    sim_settings.strategy.require_trend_alignment = REQ_TREND
    sim_settings.strategy.blocked_hours_utc = BLOCKED
    sim_settings.strategy.d1_trend_gate = D1_GATE


def _run_backtest_on_fold(
    fold_test: pd.DataFrame,
    model: VotingClassifier,
    scaler: StandardScaler,
    feat_mask: np.ndarray,
    threshold: float,
    starting_bal: float,
    settings,
) -> tuple[float, float, int]:
    """
    Run the backtest engine on the test fold.
    Returns (net_pnl, win_rate, n_trades).
    """
    X_te = scaler.transform(fold_test[FEATURE_COLUMNS])
    X_te_sel = X_te[:, feat_mask]
    test_proba = model.predict_proba(X_te_sel)[:, 1]
    test_preds = (test_proba >= threshold).astype(int)

    fold_sim_df = fold_test.copy()
    fold_sim_df["split"]       = "test"
    fold_sim_df["prediction"]  = test_preds
    fold_sim_df["probability"] = test_proba
    fold_sim_df["trade_side"]  = np.where(
        fold_sim_df["strategy_score"] >= 0, "buy", "sell"
    )

    _apply_live_overrides(settings)
    settings.training.backtest_initial_balance = starting_bal
    risk_mgr = RiskManager(settings)

    sim_result = simulate_dynamic_concurrent_backtest(
        fold_sim_df,
        settings=settings,
        risk_manager=risk_mgr,
        m1_df=None,
    )
    trades_df = sim_result.trades

    if trades_df is None or trades_df.empty:
        return 0.0, 0.0, 0

    n_trades = len(trades_df)
    net_pnl  = float(trades_df["pnl"].sum())
    wins     = int((trades_df["pnl"] > 0).sum())
    win_rate = wins / n_trades if n_trades else 0.0
    return net_pnl, win_rate, n_trades


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN RETRAIN LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def check_and_retrain(starting_bal: float = 200.0) -> bool:
    """
    Check if enough new M5 bars have accumulated; if so, retrain.
    Returns True if retrain was triggered and succeeded.
    """
    log.info("=== RETRAIN CHECK ===")
    state = _load_state()
    last_count = int(state.get("last_retrain_m5_count", 0))

    # Count current M5 rows
    m5_path = REAL_DATA_DIR / "XAUUSDm_M5.csv"
    if not m5_path.exists():
        log.warning("M5 CSV not found — run with --data-only first.")
        return False

    current_count = sum(1 for _ in open(m5_path, encoding="utf-8")) - 1  # subtract header
    new_bars = current_count - last_count

    log.info(
        "  M5 total: %d  |  last retrain at: %d  |  new bars: %d / %d needed",
        current_count, last_count, new_bars, RETRAIN_EVERY_BARS,
    )

    if new_bars < RETRAIN_EVERY_BARS:
        log.info("  Not enough new bars — skipping retrain.")
        return False

    log.info("  Enough new bars — starting retrain (this takes 5–15 min)…")

    # ── Load data ────────────────────────────────────────────────────────────
    log.info("[1/5] Loading multi-timeframe data from CSV folder…")
    settings = load_settings(CONFIG)
    data_service = MarketDataService(settings)
    frames = data_service.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)

    # ── Build dataset ────────────────────────────────────────────────────────
    log.info("[2/5] Building feature dataset…")
    settings_full = load_settings(CONFIG)
    settings_full.training.train_start_date = None
    settings_full.training.train_end_date   = None
    settings_full.training.test_start_date  = None
    settings_full.training.test_end_date    = None
    _strategy = HybridStrategy(settings_full)
    full_ds = prepare_training_dataset(settings_full, frames, _strategy)
    log.info("  Full dataset: %d rows", len(full_ds))

    if len(full_ds) < TRAIN_BARS + TEST_BARS:
        log.error("  Not enough rows in dataset (%d). Need %d.", len(full_ds), TRAIN_BARS + TEST_BARS)
        return False

    # Use the MOST RECENT fold (latest train+test window)
    train_start = len(full_ds) - TRAIN_BARS - TEST_BARS
    train_end   = train_start + TRAIN_BARS
    fold_train  = full_ds.iloc[train_start:train_end].copy()
    fold_test   = full_ds.iloc[train_end:].copy()

    log.info(
        "  Fold  train: %d rows  [%s … %s]",
        len(fold_train),
        fold_train["time"].min().date(),
        fold_train["time"].max().date(),
    )
    log.info(
        "        test:  %d rows  [%s … %s]",
        len(fold_test),
        fold_test["time"].min().date(),
        fold_test["time"].max().date(),
    )

    y_tr = fold_train["target"].values

    # ── Scale ────────────────────────────────────────────────────────────────
    log.info("[3/5] Scaling + feature selection…")
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(fold_train[FEATURE_COLUMNS])

    sw_tr  = _build_sample_weights(y_tr)
    threshold = _search_threshold(X_tr, y_tr, sw_tr)
    log.info("  Selected threshold: %.2f", threshold)

    feat_mask = _select_features(X_tr, y_tr, sw_tr)
    log.info("  Feature mask: %d / %d features kept", feat_mask.sum(), len(feat_mask))

    X_tr_sel = X_tr[:, feat_mask]

    # ── Train ensemble ───────────────────────────────────────────────────────
    log.info("[4/5] Training Combo #133 ensemble (VotingClassifier HGB×3+RF×2+ET×1)…")
    model = _train_ensemble(X_tr_sel, y_tr, sw_tr)

    # ── Evaluate on test fold ─────────────────────────────────────────────────
    X_te     = scaler.transform(fold_test[FEATURE_COLUMNS])
    X_te_sel = X_te[:, feat_mask]
    test_proba = model.predict_proba(X_te_sel)[:, 1]
    test_preds = (test_proba >= threshold).astype(int)

    y_te = fold_test["target"].values
    roc_auc  = float(roc_auc_score(y_te, test_proba)) if len(np.unique(y_te)) > 1 else 0.5
    precision = float(precision_score(y_te, test_preds, zero_division=0))
    recall    = float(recall_score(y_te, test_preds, zero_division=0))
    f1        = float(f1_score(y_te, test_preds, zero_division=0))

    metrics = {
        "roc_auc": roc_auc,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "train_rows": len(fold_train),
        "test_rows": len(fold_test),
    }
    log.info(
        "  ROC-AUC %.4f  Prec %.4f  Rec %.4f  F1 %.4f",
        roc_auc, precision, recall, f1,
    )

    # ── Backtest gate ────────────────────────────────────────────────────────
    log.info("[5/5] Running backtest on test fold (profit gate)…")
    _apply_live_overrides(settings_full)
    net_pnl, win_rate, n_trades = _run_backtest_on_fold(
        fold_test, model, scaler, feat_mask, threshold,
        starting_bal, settings_full,
    )
    log.info(
        "  Backtest → trades: %d  WR: %.1f%%  Net P&L: $%.2f",
        n_trades, win_rate * 100, net_pnl,
    )

    fold_profitable = net_pnl > 0
    if fold_profitable:
        _save_artifacts(model, scaler, feat_mask, threshold, metrics)
        result_tag = "ACCEPTED"
    else:
        result_tag = "REJECTED (P&L ≤ 0 — keeping previous model)"
        log.warning("  New fold is not profitable. Previous model retained.")

    # ── Update state + log ───────────────────────────────────────────────────
    new_state = {
        "last_retrain_m5_count": current_count,
        "last_retrain_date":     datetime.now(timezone.utc).isoformat(),
        "last_fold":             int(state.get("last_fold", 0)) + 1,
        "last_result":           result_tag,
    }
    _save_state(new_state)

    log_entry = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "fold":            new_state["last_fold"],
        "m5_bars_total":   current_count,
        "new_bars":        new_bars,
        "train_start":     str(fold_train["time"].min()),
        "train_end":       str(fold_train["time"].max()),
        "test_start":      str(fold_test["time"].min()),
        "test_end":        str(fold_test["time"].max()),
        "threshold":       round(threshold, 4),
        "n_features":      int(feat_mask.sum()),
        **metrics,
        "n_trades":        n_trades,
        "win_rate":        round(win_rate, 4),
        "net_pnl":         round(net_pnl, 2),
        "accepted":        fold_profitable,
        "result":          result_tag,
    }
    _log_retrain(log_entry)
    _notify_retrain(log_entry)

    log.info("=== RETRAIN %s ===", result_tag)
    return fold_profitable


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto data-update + Combo #133 model retrain"
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="Only fetch/append data; skip retrain check.",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Skip data fetch; only run retrain check.",
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Force retrain even if < 6000 new bars.",
    )
    parser.add_argument(
        "--starting-bal", type=float, default=200.0,
        help="Starting balance for backtest profit gate (default 200).",
    )
    args = parser.parse_args()

    if not args.train_only:
        n_new = fetch_and_update_data()
        print(f"\n[DATA]  +{n_new} new M5 bars appended.\n")
        # Send data-update alert (shows countdown to next retrain)
        if n_new > 0:
            state = _load_state()
            m5_path = REAL_DATA_DIR / "XAUUSDm_M5.csv"
            current_count = sum(1 for _ in open(m5_path, encoding="utf-8")) - 1 if m5_path.exists() else 0
            last_count = int(state.get("last_retrain_m5_count", 0))
            _notify_data_update(n_new, current_count, current_count - last_count)

    if not args.data_only:
        if args.force_retrain:
            # Temporarily zero out last_retrain_m5_count to force trigger
            state = _load_state()
            state["last_retrain_m5_count"] = 0
            _save_state(state)

        ok = check_and_retrain(starting_bal=args.starting_bal)
        status = "Model updated." if ok else "No model update."
        print(f"\n[RETRAIN]  {status}\n")


if __name__ == "__main__":
    main()
