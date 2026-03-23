"""XAUUSD AI Bot — Comprehensive Live Dashboard v3.0 ICT+Wyckoff

Model: HistGradientBoostingClassifier, 28 features
Features: D1(1) + H4(9: ICT) + H1(3: Wyckoff) + M15(15: execution)
Threshold: 0.55 | Walk-Forward: 19 folds, precision avg 54.7%, AUC std 0.0123

Tabs:
  1. Live Monitor        — Bot status, account overview, latest signal
  2. Phan tich Chi tiet  — 6-step decision breakdown (ICT→Wyckoff→Execution)
  3. P&L & Von           — Equity curve, drawdown, win/loss streaks
  4. Hoc Lien Tuc        — Learning cycle, ROC-AUC improvement, win/loss log
  5. Backtest            — Historical backtest results (ICT+Wyckoff model)
  6. Walk-Forward        — 19-fold walk-forward analysis
  7. Risk & Cai dat      — Lot calculator, position sizing
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    _HAS_WATCHDOG = True
except ImportError:
    _HAS_WATCHDOG = False

ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "outputs"

GREEN  = "#26a69a"
RED    = "#ef5350"
AMBER  = "#ffa726"
BLUE   = "#42a5f5"
GREY   = "#90a4ae"
PURPLE = "#ab47bc"

_LIVE_COLS = [
    "time", "should_trade", "side", "confidence", "reason",
    "entry_price", "stop_loss", "take_profit", "volume",
    "strategy_score", "volatility_regime",
    "account_balance", "open_positions", "max_positions",
]
_PAPER_COLS = [
    "time", "should_trade", "side", "confidence", "reason",
    "entry_price", "stop_loss", "take_profit",
    "strategy_score", "volatility_regime",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def load_signals(filename: str = "paper_trade_signals.csv") -> pd.DataFrame:
    """Load paper_trade_signals CSV — handle mixed 10/14-col schemas robustly."""
    path = OUTPUTS / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        # Need at least header + 1 data row
        data_lines = [l for l in lines[1:] if l.strip()]
        if not data_lines:
            return pd.DataFrame()
        # Detect max cols from data rows (naive split — may overcount if reason has comma)
        max_cols = max(len(l.split(",")) for l in data_lines)
        col_names = _LIVE_COLS if max_cols >= len(_LIVE_COLS) else _PAPER_COLS
        # Use header=None + skiprows=1 to avoid pandas header/names count mismatch
        df = pd.read_csv(
            path,
            header=None,
            names=col_names,
            skiprows=1,
            on_bad_lines="skip",
        )
        df = df[df["should_trade"].astype(str).str.lower().isin(["true", "false", "0", "1"])]
        df["should_trade"] = df["should_trade"].astype(str).str.lower().isin(["true", "1"])
        for col in ["confidence", "strategy_score", "entry_price", "stop_loss", "take_profit"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["volatility_regime", "open_positions", "max_positions"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(1).astype(int)
        if "account_balance" in df.columns:
            df["account_balance"] = pd.to_numeric(df["account_balance"], errors="coerce")
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"])
        # Keep the LAST written row per bar-timestamp (most up-to-date open_positions)
        df = df.drop_duplicates(subset=["time"], keep="last")
        return df.sort_values("time", ascending=False).reset_index(drop=True)
    except Exception as exc:
        # Surface the error as a Streamlit warning so we can debug
        try:
            import streamlit as _st
            _st.warning(f"load_signals error: {exc}")
        except Exception:
            pass
        return pd.DataFrame()


def _colour(val: float, good: bool = True) -> str:
    if good:
        return GREEN if val > 0 else (RED if val < 0 else GREY)
    return RED if val > 0 else (GREEN if val < 0 else GREY)


def _pct(val, dec: int = 1) -> str:
    if val is None:
        return "n/a"
    try:
        return f"{float(val):.{dec}%}"
    except Exception:
        return "n/a"


def _round(val, dec: int = 4):
    try:
        return round(float(val), dec)
    except Exception:
        return val


def _card(title: str, value: str, subtitle: str = "", colour: str = BLUE) -> str:
    return (
        f'<div style="background:linear-gradient(135deg,{colour}0a,{colour}14);'
        f'border:1px solid {colour}30;border-left:3px solid {colour};'
        f'padding:14px 18px;border-radius:12px;margin-bottom:6px;'
        f'box-shadow:0 2px 8px {colour}10;transition:all 0.2s ease">'
        f'<div style="font-size:0.68rem;color:{colour};text-transform:uppercase;'
        f'letter-spacing:.08em;font-weight:600;margin-bottom:6px">{title}</div>'
        f'<div style="font-size:1.7rem;font-weight:800;color:#f1f5f9;'
        f'line-height:1.1">{value}</div>'
        f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:4px">{subtitle}</div>'
        f'</div>'
    )


def _threshold_bar(label: str, value: float, threshold: float, reverse: bool = False) -> None:
    pct    = min(abs(value) / max(abs(threshold) * 2, 1e-9), 1.0)
    passed = (value >= threshold) if not reverse else (value <= threshold)
    colour = GREEN if passed else RED
    icon   = "✓" if passed else "✗"
    bg     = f"{colour}10"
    st.markdown(
        f'<div style="margin-bottom:10px;background:{bg};padding:10px 14px;'
        f'border-radius:10px;border:1px solid {colour}20">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;'
        f'margin-bottom:6px;align-items:center">'
        f'<span style="font-weight:600;color:#e2e8f0">{icon} {label}</span>'
        f'<span style="color:{colour};font-weight:700;font-family:monospace">{value:.4f}'
        f'<span style="color:#64748b;font-weight:400"> / {threshold:.4f}</span></span>'
        f'</div>'
        f'<div style="background:#0f172a;border-radius:6px;height:6px;overflow:hidden">'
        f'<div style="width:{pct*100:.1f}%;background:linear-gradient(90deg,{colour},{colour}cc);'
        f'height:6px;border-radius:6px;transition:width 0.5s ease"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _step_ok(step: int, label: str, passed: bool | None, detail: str = "") -> None:
    colour = GREEN if passed is True else (RED if passed is False else GREY)
    icon   = "✅" if passed is True else ("❌" if passed is False else "ℹ️")
    badge_bg = f"{colour}22"
    badge_border = f"{colour}44"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;'
        f'padding:12px 16px;background:linear-gradient(135deg,{colour}08,{colour}04);'
        f'border:1px solid {colour}18;border-left:3px solid {colour};border-radius:10px;'
        f'transition:all 0.2s ease">'
        f'<div style="min-width:36px;height:36px;border-radius:10px;background:{badge_bg};'
        f'border:1px solid {badge_border};display:flex;align-items:center;justify-content:center;'
        f'font-size:0.85rem;font-weight:800;color:{colour}">{step}</div>'
        f'<div style="flex:1">'
        f'<div style="font-size:0.92rem;color:#f1f5f9;font-weight:600">{label}</div>'
        f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:2px">{detail}</div>'
        f'</div>'
        f'<div style="font-size:1.1rem">{icon}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def compute_streaks(is_win_series: pd.Series) -> dict:
    wins = losses = cur_win = cur_loss = max_win = max_loss = 0
    for w in is_win_series:
        if w:
            cur_win += 1; cur_loss = 0; wins += 1
        else:
            cur_loss += 1; cur_win = 0; losses += 1
        max_win  = max(max_win,  cur_win)
        max_loss = max(max_loss, cur_loss)
    return {
        "wins": wins, "losses": losses,
        "current_streak": cur_win if cur_win > 0 else -cur_loss,
        "max_win_streak": max_win,
        "max_loss_streak": max_loss,
    }


def compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd   = (equity - peak) / peak.replace(0, pd.NA) * 100
    return dd.fillna(0)


def _ds(series: pd.Series, max_pts: int = 1000) -> pd.Series:
    """Downsample a series to at most max_pts points and round to avoid MemoryError."""
    step = max(1, len(series) // max_pts)
    return series.iloc[::step].round(4)


def load_learning_events(filename: str = "live_learning_log.jsonl") -> list[dict]:
    path = OUTPUTS / filename
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except Exception:
            pass
    return events


def load_wf_signals() -> pd.DataFrame:
    """Load walk-forward signals from walk-forward analysis."""
    path = OUTPUTS / "walkforward_signals_ict_wyckoff.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def load_live_closed_trades(filename: str = "live_closed_trades.csv") -> pd.DataFrame:
    """Load live closed trades logged by the live trading loop (all sessions)."""
    path = OUTPUTS / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        if "is_win" not in df.columns and "pnl" in df.columns:
            df["is_win"] = df["pnl"] > 0
        if "close_type" not in df.columns:
            df["close_type"] = "UNKNOWN"
        # Retroactively fix: SL trades can never be a win (even if PnL >= 0)
        sl_mask = df["close_type"].astype(str).str.upper() == "SL"
        df.loc[sl_mask, "is_win"] = False
        if "session_id" not in df.columns:
            df["session_id"] = "legacy"
        # drop header-only rows that may appear mid-file from old resets
        df = df.dropna(subset=["ticket"])
        df = df[df["ticket"].astype(str).str.match(r"^\d")]
        return df.sort_values("time").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def simulate_signal_outcomes(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Với mỗi tín hiệu (kể cả bị lọc), scan M15 price data về sau để xác định:
      - Nếu vào lệnh tại entry_price, TP hay SL hit trước?
      - P&L ước tính dựa trên volume (nếu có) hoặc 0.01 lot mặc định.
    Trả về signals DataFrame với thêm các cột: sim_outcome, sim_pnl, sim_bars.
    """
    M15_PATH = ROOT / "src" / "xauusd_ai" / "real_data" / "XAUUSDm_M15.csv"
    if not M15_PATH.exists() or signals.empty:
        return signals.copy()

    try:
        price = pd.read_csv(M15_PATH, parse_dates=["time"], on_bad_lines="skip")
        price["time"] = pd.to_datetime(price["time"], utc=True, errors="coerce")
        price = price.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    except Exception:
        return signals.copy()

    result = signals.copy()
    for col in ["sim_outcome", "sim_pnl_per_lot", "sim_pnl", "sim_bars"]:
        result[col] = None

    for idx, row in result.iterrows():
        try:
            entry = float(row.get("entry_price") or 0)
            sl    = float(row.get("stop_loss") or 0)
            tp    = float(row.get("take_profit") or 0)
            side  = str(row.get("side", "")).lower()
            vol   = float(row.get("volume") or 0.01)
            sig_time = pd.to_datetime(row.get("time"), utc=True, errors="coerce")
            if entry <= 0 or sl <= 0 or tp <= 0 or side not in ("buy", "sell") or pd.isna(sig_time):
                continue

            future = price[price["time"] > sig_time].head(200)  # max 200 bars ~50h
            if future.empty:
                continue

            outcome = None
            bars_taken = 0
            for _, bar in future.iterrows():
                bars_taken += 1
                h, l = float(bar["high"]), float(bar["low"])
                if side == "buy":
                    if l <= sl:
                        outcome = "loss"
                        break
                    if h >= tp:
                        outcome = "win"
                        break
                else:  # sell
                    if h >= sl:
                        outcome = "loss"
                        break
                    if l <= tp:
                        outcome = "win"
                        break

            if outcome is None:
                result.at[idx, "sim_outcome"] = "open"
                continue

            # P&L per lot: (TP/SL - entry) × contract_oz
            if side == "buy":
                pnl_per_lot = (tp - entry) * 100.0 if outcome == "win" else (sl - entry) * 100.0
            else:
                pnl_per_lot = (entry - tp) * 100.0 if outcome == "win" else (entry - sl) * 100.0

            lot = max(vol, 0.01)
            result.at[idx, "sim_outcome"]     = outcome
            result.at[idx, "sim_pnl_per_lot"] = round(pnl_per_lot, 2)
            result.at[idx, "sim_pnl"]         = round(pnl_per_lot * lot, 2)
            result.at[idx, "sim_bars"]        = bars_taken
        except Exception:
            continue

    return result


def load_win_loss_events() -> tuple[list[dict], list[dict]]:
    """Load win/loss JSONL from walk-forward analysis."""
    wins: list[dict] = []
    losses: list[dict] = []
    for path, store in [
        (OUTPUTS / "win_analysis_walkforward.jsonl", wins),
        (OUTPUTS / "loss_analysis_walkforward.jsonl", losses),
    ]:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    store.append(json.loads(line))
                except Exception:
                    pass
    return wins, losses


def load_wf_txt_log(log_path: Path) -> dict:
    """Parse a walkforward text log (wf_v5_acc2.txt style) in real-time.

    Returns a dict with keys:
      - folds: list of fold result dicts
      - total_folds: int
      - n_features: int
      - config: str
      - is_complete: bool
      - avg_auc, avg_prec, avg_recall, avg_f1: float (across completed folds)
    """
    if not log_path.exists():
        return {}
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
    except Exception:
        return {}

    folds: list[dict] = []
    total_folds = 0
    n_features = 0
    config_str = ""
    is_complete = False

    # Patterns
    _fold_pat = re.compile(
        r"Fold\s+(\d+)/(\d+)\s+Test:\s+(\S+)\s+->\s+(\S+)\s+\|"
        r"\s+AUC=([\d.]+)\s+Prec=([\d.]+)\s+Recall=([\d.]+)\s+F1=([\d.]+)"
        r"\s+Thr=([\d.]+)\s+Sigs=(\d+)/(\d+)\s+\(([\d.]+)s\)",
    )
    _total_pat = re.compile(r"Estimated folds:\s*(\d+)")
    _feat_pat = re.compile(r"Features:\s*(\d+)")
    _cfg_pat = re.compile(r"Config\s*:\s*(.+)")

    for line in lines:
        if m := _cfg_pat.search(line):
            config_str = m.group(1).strip()
        if m := _total_pat.search(line):
            total_folds = int(m.group(1))
        if m := _feat_pat.search(line):
            n_features = int(m.group(1))
        if m := _fold_pat.search(line):
            folds.append({
                "fold":       int(m.group(1)),
                "total":      int(m.group(2)),
                "test_start": m.group(3),
                "test_end":   m.group(4),
                "roc_auc":    float(m.group(5)),
                "precision":  float(m.group(6)),
                "recall":     float(m.group(7)),
                "f1":         float(m.group(8)),
                "threshold":  float(m.group(9)),
                "n_signals":  int(m.group(10)),
                "elapsed_s":  float(m.group(12)),
            })
        if "Report saved" in line or "Hoàn thành" in line or "[4/4]" in line:
            is_complete = True

    if not folds:
        return {}

    # Remove duplicate fold numbers (keep last occurrence)
    seen: dict[int, dict] = {}
    for f in folds:
        seen[f["fold"]] = f
    folds = list(seen.values())

    n = len(folds)
    avg_auc   = sum(f["roc_auc"]   for f in folds) / n
    avg_prec  = sum(f["precision"] for f in folds) / n
    avg_recall = sum(f["recall"]   for f in folds) / n
    avg_f1    = sum(f["f1"]        for f in folds) / n

    return {
        "folds":       folds,
        "total_folds": total_folds or (folds[-1]["total"] if folds else 0),
        "n_features":  n_features,
        "config":      config_str,
        "is_complete": is_complete,
        "avg_auc":     avg_auc,
        "avg_prec":    avg_prec,
        "avg_recall":  avg_recall,
        "avg_f1":      avg_f1,
        "completed":   n,
    }


def load_feature_importance() -> pd.DataFrame | None:
    # Try ICT+Wyckoff model first, fall back to legacy model
    mp = OUTPUTS / "model_ict_wyckoff.pkl"
    if not mp.exists():
        mp = OUTPUTS / "model.pkl"
    if not mp.exists():
        return None
    try:
        with open(mp, "rb") as f:
            model = pickle.load(f)
        feature_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
        # HistGradientBoosting / tree-based models
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            if feature_names is None:
                feature_names = [f"feat_{i}" for i in range(len(imp))]
            df = pd.DataFrame({"feature": feature_names, "importance": imp})
            df["abs_importance"] = df["importance"].abs()
            return df.sort_values("abs_importance", ascending=False).head(25)
        # Linear models with coef_
        if hasattr(model, "coef_"):
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            if feature_names is None:
                feature_names = [f"feat_{i}" for i in range(len(coef))]
            df = pd.DataFrame({"feature": feature_names, "importance": coef})
            df["abs_importance"] = df["importance"].abs()
            return df.sort_values("abs_importance", ascending=False).head(25)
    except Exception:
        pass
    return None


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty or "pnl" not in trades.columns:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "gross_profit": 0.0, "gross_loss": 0.0, "net_profit": 0.0}
    wins = trades[trades["pnl"] > 0]
    losn = trades[trades["pnl"] <= 0]
    return {
        "trades":       len(trades),
        "wins":         len(wins),
        "losses":       len(losn),
        "win_rate":     len(wins) / max(len(trades), 1),
        "gross_profit": float(wins["pnl"].sum()),
        "gross_loss":   float(losn["pnl"].sum()),
        "net_profit":   float(trades["pnl"].sum()),
    }


def summarize_daily(trades: pd.DataFrame):
    empty_s = dict(mean_daily_return=0.0, median_daily_return=0.0,
                   best_day_return=0.0, worst_day_return=0.0,
                   share_ge_2=0.0, share_ge_5=0.0, share_ge_10=0.0)
    if trades.empty or "pnl" not in trades.columns:
        return empty_s, pd.DataFrame()
    d = trades.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time")
    d["date"] = d["time"].dt.date
    g = d.groupby("date").agg(
        trades=("pnl", "size"),
        net_pnl=("pnl", "sum"),
        gross_profit=("pnl", lambda s: s[s > 0].sum()),
        gross_loss=("pnl", lambda s: -s[s < 0].sum()),
        start_balance=("balance_before", "first"),
        end_balance=("balance_after", "last"),
    )
    g["return_pct"] = (g["end_balance"] / g["start_balance"] - 1.0) * 100.0
    return {
        "mean_daily_return":   float(g["return_pct"].mean()),
        "median_daily_return": float(g["return_pct"].median()),
        "best_day_return":     float(g["return_pct"].max()),
        "worst_day_return":    float(g["return_pct"].min()),
        "share_ge_2":  float((g["return_pct"] >= 2.0).mean()),
        "share_ge_5":  float((g["return_pct"] >= 5.0).mean()),
        "share_ge_10": float((g["return_pct"] >= 10.0).mean()),
    }, g.reset_index()


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="XAUUSD AI — Trading Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Modern CSS theme ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-card: #111827;
    --bg-card-hover: #1a2332;
    --bg-surface: #151d2b;
    --border-subtle: rgba(255,255,255,0.06);
    --border-accent: rgba(99,102,241,0.3);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --green: #10b981;
    --green-bg: rgba(16,185,129,0.08);
    --red: #ef4444;
    --red-bg: rgba(239,68,68,0.08);
    --blue: #6366f1;
    --blue-bg: rgba(99,102,241,0.08);
    --amber: #f59e0b;
    --amber-bg: rgba(245,158,11,0.08);
    --purple: #a855f7;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --shadow-card: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2);
    --shadow-glow: 0 0 20px rgba(99,102,241,0.1);
    --transition: all 0.2s ease;
}

.block-container {
    padding-top: 1.2rem !important;
    max-width: 1400px;
}

/* ── Metric containers ── */
div[data-testid="metric-container"] {
    background: var(--bg-card);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--blue);
    box-shadow: var(--shadow-card);
    transition: var(--transition);
}
div[data-testid="metric-container"]:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-glow);
    transform: translateY(-1px);
}
div[data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-weight: 700 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: var(--bg-card);
    border-radius: var(--radius-md);
    padding: 4px;
    border: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm);
    padding: 10px 16px;
    font-weight: 600;
    font-size: 0.82rem;
    color: var(--text-secondary);
    transition: var(--transition);
}
.stTabs [aria-selected="true"] {
    background: var(--blue) !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3);
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary);
    background: var(--bg-card-hover);
}

/* ── DataFrames ── */
.stDataFrame {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-subtle) !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-subtle) !important;
    font-weight: 600 !important;
}

/* ── Progress bars ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--blue), var(--purple)) !important;
    border-radius: 6px;
}

/* ── Dividers ── */
hr {
    border-color: var(--border-subtle) !important;
    opacity: 0.5;
}

/* ── Glass card base ── */
.glass-card {
    background: linear-gradient(135deg, rgba(17,24,39,0.9), rgba(15,23,42,0.95));
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    box-shadow: var(--shadow-card);
    transition: var(--transition);
}
.glass-card:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-glow);
}

/* ── Pulse animation for live dot ── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.live-pulse {
    animation: pulse 2s ease-in-out infinite;
}

/* ── Glow animation ── */
@keyframes glow {
    0%, 100% { box-shadow: 0 0 5px rgba(99,102,241,0.2); }
    50% { box-shadow: 0 0 15px rgba(99,102,241,0.4); }
}

/* ── Buttons ── */
.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    border: 1px solid var(--border-accent) !important;
    transition: var(--transition) !important;
}
.stButton > button:hover {
    box-shadow: var(--shadow-glow) !important;
    transform: translateY(-1px) !important;
}

/* ── Selectbox / Radio ── */
.stSelectbox > div, .stRadio > div {
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header with branding ────────────────────────────────────────────────────
st.markdown(
    '<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">'
    '<div style="font-size:2.2rem;font-weight:900;background:linear-gradient(135deg,#6366f1,#a855f7);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent">'
    '⚡ XAUUSD AI</div>'
    '<div style="background:#6366f122;border:1px solid #6366f144;border-radius:6px;'
    'padding:3px 10px;font-size:0.72rem;font-weight:700;color:#a5b4fc;letter-spacing:0.06em">'
    'ICT + WYCKOFF v5.0</div>'
    '<div style="margin-left:auto;display:flex;align-items:center;gap:8px">'
    '<div class="live-pulse" style="width:8px;height:8px;border-radius:50%;background:#10b981"></div>'
    '<span style="color:#94a3b8;font-size:0.78rem;font-weight:500">LIVE</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

_tunnel = OUTPUTS / "tunnel_url.txt"
if _tunnel.exists():
    _pub = _tunnel.read_text(encoding="utf-8").strip()
    if _pub:
        st.markdown(
            f'<div class="glass-card" style="display:flex;align-items:center;gap:16px;'
            f'margin-bottom:12px;border-color:#6366f130;padding:12px 20px">'
            f'<div style="width:38px;height:38px;border-radius:10px;background:#6366f118;'
            f'display:flex;align-items:center;justify-content:center;font-size:1.2rem">🌐</div>'
            f'<div style="flex:1">'
            f'<div style="color:#94a3b8;font-size:0.72rem;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.06em">Public Access URL</div>'
            f'<a href="{_pub}" target="_blank" style="color:#818cf8;font-size:0.95rem;'
            f'font-weight:700;text-decoration:none">{_pub}</a></div>'
            f'<div style="background:linear-gradient(135deg,#6366f1,#a855f7);color:#fff;'
            f'padding:5px 14px;border-radius:8px;font-size:0.72rem;font-weight:700;'
            f'letter-spacing:0.06em">LIVE</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Watchdog: push-notify dashboard when live files change ──────────────────
_WATCHED_FILES = {
    "live_status_acc1.json",
    "live_status_acc2.json",
    "paper_trade_signals.csv",
    "paper_trade_signals_acc2.csv",
    "live_closed_trades.csv",
    "live_closed_trades_acc2.csv",
}

if _HAS_WATCHDOG:
    class _LiveFileHandler(FileSystemEventHandler):
        """Set a session-state flag whenever a watched live file is modified."""
        def on_modified(self, event):
            if not event.is_directory and Path(event.src_path).name in _WATCHED_FILES:
                try:
                    st.session_state["_live_changed"] = True
                except Exception:
                    pass

    def _start_watchdog(outputs_dir: Path) -> None:
        """Start watchdog observer in daemon thread (once per process)."""
        observer = Observer()
        observer.schedule(_LiveFileHandler(), str(outputs_dir), recursive=False)
        observer.daemon = True
        observer.start()

    # Guard: only start one observer per Streamlit worker process
    if "_watchdog_started" not in st.session_state:
        _wt = threading.Thread(target=_start_watchdog, args=(OUTPUTS,), daemon=True)
        _wt.start()
        st.session_state["_watchdog_started"] = True
        st.session_state["_live_changed"] = False

(tab_live, tab_analysis, tab_pnl, tab_learning,
 tab_backtest, tab_walkforward, tab_risk, tab_data) = st.tabs([
    "� Live",
    "🔬 Phân tích",
    "💰 P&L",
    "🧠 Tự học",
    "📊 Backtest",
    "🔄 Walk-Forward",
    "⚙️ Rủi ro",
    "🗄️ Dữ liệu",
])

# --- Shared data -------------------------------------------------------
model_meta      = load_json(OUTPUTS / "model_meta_ict_wyckoff.json")
if not model_meta:
    model_meta  = load_json(OUTPUTS / "model_meta.json")  # fallback
model_path      = OUTPUTS / "model_ict_wyckoff.pkl"
if not model_path.exists():
    model_path  = OUTPUTS / "model.pkl"  # fallback
live_signals    = load_signals()
live_signals_acc2 = load_signals("paper_trade_signals_acc2.csv")
learn_events    = load_learning_events()
learn_events_acc2 = load_learning_events("live_learning_log_acc2.jsonl")
wf_win_events, wf_loss_events = load_win_loss_events()
backtest_report = load_json(OUTPUTS / "backtest_report_acc1.json")  # ACC1 retrain output
if not backtest_report:
    backtest_report = load_json(OUTPUTS / "backtest_report_ict_wyckoff.json")  # fallback old
if not backtest_report:
    backtest_report = load_json(OUTPUTS / "backtest_report.json")
training_report = load_json(OUTPUTS / "training_report_ict_wyckoff.json")
if not training_report:
    training_report = load_json(OUTPUTS / "training_report.json")  # fallback
trades          = load_csv(OUTPUTS / "backtest_trades_acc1.csv")  # ACC1 retrain output
if trades.empty:
    trades      = load_csv(OUTPUTS / "backtest_trades_ict_wyckoff.csv")  # fallback old
if trades.empty:
    trades      = load_csv(OUTPUTS / "backtest_trades.csv")
if not trades.empty and "time" in trades.columns:
    trades["time"] = pd.to_datetime(trades["time"], utc=True, errors="coerce")
    if "is_win" not in trades.columns and "pnl" in trades.columns:
        trades["is_win"] = trades["pnl"] > 0
# ACC2 (Model2) backtest data
backtest_report_acc2 = load_json(OUTPUTS / "backtest_report_acc2.json")
trades_acc2          = load_csv(OUTPUTS / "backtest_trades_acc2.csv")
if not trades_acc2.empty and "time" in trades_acc2.columns:
    trades_acc2["time"] = pd.to_datetime(trades_acc2["time"], utc=True, errors="coerce")
    if "is_win" not in trades_acc2.columns and "pnl" in trades_acc2.columns:
        trades_acc2["is_win"] = trades_acc2["pnl"] > 0
live_trades     = load_live_closed_trades()
live_trades_acc2 = load_live_closed_trades("live_closed_trades_acc2.csv")

threshold_val = float(
    model_meta.get("decision_threshold")
    or model_meta.get("selected_threshold")
    or 0.5
)
model_meta_acc2  = load_json(OUTPUTS / "model_meta2_weekly500.json")
threshold_val_acc2 = float(
    model_meta_acc2.get("decision_threshold")
    or model_meta_acc2.get("selected_threshold")
    or 0.55
)

# =============================================================================
# TAB 1 — LIVE MONITOR  (watchdog push + @st.fragment 30s fallback)
# =============================================================================
@st.fragment(run_every=30)
def _render_live_tab() -> None:
    """Renders live-monitor content.

    Watchdog sets st.session_state['_live_changed'] = True whenever a live
    file is modified.  This fragment checks that flag first — if set it clears
    the flag and calls st.rerun() to propagate a full-page push.  The
    run_every=30 is a fallback in case watchdog isn't running.
    """
    # ── Push: rerun immediately when watchdog detected a file change ──────
    if st.session_state.get("_live_changed", False):
        st.session_state["_live_changed"] = False
        st.rerun()

    # ── Fresh data reads on every auto-refresh cycle ─────────────────────
    _live_sigs      = load_signals()
    _live_sigs_acc2 = load_signals("paper_trade_signals_acc2.csv")
    _ltrades        = load_live_closed_trades()
    _ltrades_acc2   = load_live_closed_trades("live_closed_trades_acc2.csv")
    _thr     = float(model_meta.get("decision_threshold") or model_meta.get("selected_threshold") or 0.5)
    _thr_acc2 = float(model_meta_acc2.get("decision_threshold") or model_meta_acc2.get("selected_threshold") or 0.55)

    st.markdown(
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">'
        '<div style="font-size:1.6rem;font-weight:800;color:#f1f5f9">📡 Live Monitor</div>'
        '<div class="live-pulse" style="width:8px;height:8px;border-radius:50%;background:#10b981"></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        if model_path.exists():
            mtime = dt.datetime.fromtimestamp(model_path.stat().st_mtime)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:8px 14px;'
                f'background:#10b98110;border:1px solid #10b98130;border-radius:10px;font-size:0.82rem">'
                f'<span style="color:#10b981;font-weight:700">●</span>'
                f'<span style="color:#d1fae5">Model hoạt động</span>'
                f'<span style="color:#6ee7b7;font-weight:600">HistGBC 28 features</span>'
                f'<span style="color:#64748b">|</span>'
                f'<span style="color:#94a3b8">Trained: {mtime.strftime("%Y-%m-%d %H:%M")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.error("❌ Model chưa được train")
    with hdr_r:
        _now_str = dt.datetime.now().strftime("%H:%M:%S")
        st.markdown(
            f'<div style="text-align:right;color:#64748b;font-size:0.75rem">'
            f'Auto-refresh 30s<br>⏱ {_now_str}</div>',
            unsafe_allow_html=True,
        )
        if st.button("🔄 Refresh", key="btn_manual_refresh", type="secondary"):
            st.rerun()

    # ── 2-account quick overview ─────────────────────────────────────────
    def _acc_summary_card(label: str, signals: pd.DataFrame, acc_id: str, status_file: str = "") -> None:
        # Prefer live_status json for open_positions (updated every scan ~10s)
        _st_json = load_json(OUTPUTS / status_file) if status_file else {}
        if signals.empty and not _st_json:
            st.markdown(
                f'<div style="background:#1e1e2e;border-left:4px solid #555;padding:12px 16px;border-radius:8px">'
                f'<div style="color:#aaa;font-size:0.8rem">{label} ({acc_id})</div>'
                f'<div style="color:#555;font-size:1.1rem">Chưa có dữ liệu</div>'
                f'</div>', unsafe_allow_html=True)
            return
        latest: dict = signals.iloc[0].to_dict() if not signals.empty else {}
        bal   = float(_st_json.get("account_balance") or latest.get("account_balance", 0) or 0)
        opn   = int(float(_st_json.get("open_positions", latest.get("open_positions", 0)) or 0))
        mx    = int(float(_st_json.get("max_positions",  latest.get("max_positions", 1))  or 1))
        conf  = float(_st_json.get("confidence", latest.get("confidence", 0)) or 0)
        side  = str(_st_json.get("side", latest.get("side", "flat")))
        traded = bool(_st_json.get("should_trade", latest.get("should_trade", False)))
        _ts   = str(_st_json.get("ts") or latest.get("time", ""))[:16]
        bal_c = GREEN if bal >= 200 else (AMBER if bal >= 100 else RED)
        side_emoji = {"buy": "📈", "sell": "📉"}.get(side, "➖")
        dec_c = GREEN if traded else "#888"
        st.markdown(
            f'<div style="background:#1e1e2e;border-left:4px solid {bal_c};padding:12px 16px;border-radius:8px">'
            f'<div style="color:#aaa;font-size:0.8rem">{label} <code>{acc_id}</code></div>'
            f'<div style="display:flex;gap:20px;align-items:center;margin-top:6px">'
            f'<span style="font-size:1.3rem;font-weight:700;color:{bal_c}">${bal:,.2f}</span>'
            f'<span style="color:#aaa">{opn}/{mx} lệnh</span>'
            f'<span style="color:{dec_c}">{side_emoji} {side.upper()} {conf:.0%}</span>'
            f'<span style="color:#666;font-size:0.75rem">{_ts}</span>'
            f'</div>'
            f'</div>', unsafe_allow_html=True)

    both_have_data = not _live_sigs.empty or not _live_sigs_acc2.empty
    if both_have_data:
        st.markdown(
            '<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:12px 0 8px">'
            '🏦 Tổng quan Tài khoản</div>',
            unsafe_allow_html=True,
        )
        ov1, ov2 = st.columns(2)
        with ov1:
            _acc_summary_card("Acc 1 — Exness-MT5Trial17", _live_sigs, "270832477", "live_status_acc1.json")
        with ov2:
            _acc_summary_card("Acc 2 — Exness-MT5Trial7", _live_sigs_acc2, "433326057", "live_status_acc2.json")
        st.divider()

    # ── Account selector ─────────────────────────────────────────────────
    _acc_options = ["Acc 1 — 270832477 (Exness-MT5Trial17)", "Acc 2 — 433326057 (Exness-MT5Trial7)"]
    _sel_acc = st.radio("Xem chi tiết tài khoản:", _acc_options, horizontal=True, key="acc_selector")
    _selected_signals = _live_sigs if "Acc 1" in _sel_acc else _live_sigs_acc2
    _sel_thr = _thr_acc2 if "Acc 2" in _sel_acc else _thr

    # Convert ONCE to plain dict — avoids all Series-truth-value errors
    _sel_row: dict = _selected_signals.iloc[0].to_dict() if not _selected_signals.empty else {}

    def _f(key, default=0.0):
        """NaN-safe float from _sel_row."""
        v = _sel_row.get(key, default)
        try:
            r = float(v)
            return default if r != r else r  # guard NaN
        except (TypeError, ValueError):
            return float(default)

    if not _selected_signals.empty:
        conf    = _f("confidence")
        side    = str(_sel_row.get("side", "flat"))
        score   = _f("strategy_score")
        regime  = int(_f("volatility_regime", 1))
        traded  = bool(_sel_row.get("should_trade", False))
        reason  = str(_sel_row.get("reason", ""))
        r_label = {0: "Sideways", 1: "Normal", 2: "Strong Vol"}.get(regime, "?")
        side_c  = {"buy": GREEN, "sell": RED}.get(side, GREY)
        conf_c  = GREEN if conf >= _sel_thr else (AMBER if conf >= _sel_thr * 0.7 else RED)
        score_c = GREEN if abs(score) >= 0.3 else (AMBER if abs(score) >= 0.1 else RED)
        reg_c   = {0: AMBER, 1: BLUE, 2: GREEN}.get(regime, GREY)
        dec_c   = GREEN if traded else RED

        st.markdown(
            '<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:4px 0 8px">'
            '📡 Tín hiệu Mới nhất</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(_card("Tín hiệu", {"buy": "BUY", "sell": "SELL"}.get(side, "FLAT"),
                          str(_sel_row.get("time", ""))[:16], side_c), unsafe_allow_html=True)
        c2.markdown(_card("ML Confidence", f"{conf:.1%}", f"ngưỡng {_sel_thr:.0%}", conf_c), unsafe_allow_html=True)
        c3.markdown(_card("Strategy Score", f"{score:+.4f}", ">= 0.30 để vào lệnh", score_c), unsafe_allow_html=True)
        c4.markdown(_card("Chế độ", r_label, "thị trường", reg_c), unsafe_allow_html=True)
        c5.markdown(_card("Quyết định", "VÀO LỆNH" if traded else "KHÔNG VÀO",
                          "" if traded else reason[:40], dec_c), unsafe_allow_html=True)

    if not _selected_signals.empty and "account_balance" in _selected_signals.columns:
        # ── Đọc live_status.json để lấy open_positions realtime ──────────
        _status_file = "live_status_acc2.json" if "Acc 2" in _sel_acc else "live_status_acc1.json"
        _live_status = load_json(OUTPUTS / _status_file)
        try:
            bal = float(_live_status.get("account_balance") or _f("account_balance") or 0)
            opn = int(_live_status.get("open_positions", _f("open_positions")))
            mx  = int(_live_status.get("max_positions",  _f("max_positions", 1)) or 1)
            vol = _f("volume") if "volume" in _selected_signals.columns else 0.0
        except (TypeError, ValueError):
            bal, opn, mx, vol = 0.0, 0, 1, 0.0

        st.markdown(
            '<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:4px 0 8px">'
            '💰 Trạng thái Tài khoản</div>',
            unsafe_allow_html=True,
        )
        a1, a2, a3, a4 = st.columns(4)
        bal_c = GREEN if bal >= 200 else (AMBER if bal >= 100 else RED)
        a1.markdown(_card("Số dư", f"${bal:,.2f}", "MT5 live balance", bal_c), unsafe_allow_html=True)
        pos_c = RED if opn >= mx else GREEN
        a2.markdown(_card("Lệnh Đang Mở", f"{opn} / {mx}", "open / max", pos_c), unsafe_allow_html=True)
        a3.markdown(_card("Lot Size", f"{vol:.3f}", "current order size", BLUE), unsafe_allow_html=True)
        bar_pct = int((opn / max(mx, 1)) * 100)
        bar_c = GREEN if bar_pct < 70 else (AMBER if bar_pct < 100 else RED)
        a4.markdown(
            f'<div style="background:linear-gradient(135deg,{bar_c}0a,{bar_c}14);'
            f'border:1px solid {bar_c}30;border-left:3px solid {bar_c};'
            f'padding:14px 18px;border-radius:12px;margin-bottom:6px">'
            f'<div style="font-size:0.68rem;color:{bar_c};text-transform:uppercase;'
            f'letter-spacing:.08em;font-weight:600;margin-bottom:8px">Position Utilization</div>'
            f'<div style="font-size:1.7rem;font-weight:800;color:#f1f5f9;margin-bottom:8px">{bar_pct}%</div>'
            f'<div style="background:#0f172a;border-radius:6px;height:6px;overflow:hidden">'
            f'<div style="width:{bar_pct}%;background:linear-gradient(90deg,{bar_c},{bar_c}cc);'
            f'height:6px;border-radius:6px;transition:width 0.5s ease"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if opn >= mx:
            st.warning(f"⚠️ Đầy lệnh: {opn}/{mx} — Bot sẽ không mở thêm")

    if model_meta:
        st.divider()
        st.markdown(
            '<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:4px 0 8px">'
            '🧠 Hiệu suất Model</div>',
            unsafe_allow_html=True,
        )
        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("Threshold", f"{_sel_thr:.2f}")
        mm2.metric("Precision", _pct(model_meta.get("precision")))
        mm3.metric("Recall",    _pct(model_meta.get("recall")))
        mm4.metric("F1 Score",  _pct(model_meta.get("f1")))
        # Walk-Forward aggregate stats
        _wf_agg = load_json(OUTPUTS / "walkforward_report_ict_wyckoff.json").get("aggregate", {})
        if _wf_agg:
            st.caption("📊 Walk-Forward (19 folds, 2022–2026)")
            wm1, wm2, wm3, wm4 = st.columns(4)
            wm1.metric("WF Avg AUC",      f"{_wf_agg.get('avg_roc_auc', 0):.4f}",
                       delta=f"std={_wf_agg.get('std_roc_auc', 0):.4f}")
            wm2.metric("WF Avg Precision", _pct(_wf_agg.get("avg_precision")),
                       delta=f"range {_wf_agg.get('min_precision', 0):.2f}–{_wf_agg.get('max_precision', 0):.2f}")
            wm3.metric("WF Signal Win Rate", _pct(_wf_agg.get("signal_win_rate")),
                       delta=f"{_wf_agg.get('correct_signals', 0)}/{_wf_agg.get('total_signals', 0)} signals")
            wm4.metric("WF Avg F1",        _pct(_wf_agg.get("avg_f1")))

    if not _selected_signals.empty and bool(_sel_row.get("should_trade", False)):
        entry = _f("entry_price")
        sl    = _f("stop_loss")
        tp    = _f("take_profit")
        rr    = abs((tp - entry) / (entry - sl)) if (sl and tp and entry and sl != entry) else 0
        st.divider()
        st.subheader("📍 Kế hoạch vào lệnh")
        ep1, ep2, ep3, ep4 = st.columns(4)
        ep1.metric("Entry",       f"{entry:.3f}")
        ep2.metric("Stop Loss",   f"{sl:.3f}", delta=f"{sl - entry:+.2f}")
        ep3.metric("Take Profit", f"{tp:.3f}", delta=f"{tp - entry:+.2f}")
        ep4.metric("Risk/Reward", f"{rr:.2f}R")

    st.divider()
    st.markdown(
        '<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:4px 0 8px">'
        '📋 Tín hiệu gần nhất (100 mục)</div>',
        unsafe_allow_html=True,
    )
    if not _selected_signals.empty:
        _sig_disp = _selected_signals.copy().head(100)

        # ── Mô phỏng kết quả nếu vào lệnh (kể cả lệnh bị lọc) ──────────
        with st.spinner("Đang tính kết quả mô phỏng..."):
            _sig_sim = simulate_signal_outcomes(_sig_disp)

        # Format cột hiển thị
        _sig_sim["Thời gian"]    = _sig_sim["time"].astype(str).str[:16]
        _sig_sim["Tín hiệu"]     = _sig_sim["side"].str.upper()
        _sig_sim["Vào lệnh?"]    = _sig_sim["should_trade"].map(lambda x: "✅ CÓ" if x else "❌ KHÔNG")
        _sig_sim["Confidence"]   = _sig_sim["confidence"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        _sig_sim["Strategy Sc."] = _sig_sim["strategy_score"].map(lambda x: f"{x:+.4f}" if pd.notna(x) else "")
        for _c in ["entry_price", "stop_loss", "take_profit"]:
            if _c in _sig_sim.columns:
                _sig_sim[_c] = pd.to_numeric(_sig_sim[_c], errors="coerce")
        if "entry_price" in _sig_sim.columns and "stop_loss" in _sig_sim.columns and "take_profit" in _sig_sim.columns:
            _sig_sim["Entry"] = _sig_sim["entry_price"].map(lambda x: f"{x:.3f}" if pd.notna(x) and x > 0 else "—")
            _sig_sim["SL"]    = _sig_sim["stop_loss"].map(lambda x: f"{x:.3f}" if pd.notna(x) and x > 0 else "—")
            _sig_sim["TP"]    = _sig_sim["take_profit"].map(lambda x: f"{x:.3f}" if pd.notna(x) and x > 0 else "—")
            _sig_sim["R/R"]   = _sig_sim.apply(lambda r: (
                f"{abs((r['take_profit'] - r['entry_price']) / (r['entry_price'] - r['stop_loss'])):.2f}R"
                if (pd.notna(r['entry_price']) and pd.notna(r['stop_loss']) and pd.notna(r['take_profit'])
                    and r['entry_price'] > 0 and r['stop_loss'] > 0 and r['entry_price'] != r['stop_loss'])
                else "—"
            ), axis=1)
        _sig_sim["Lý do"]    = _sig_sim["reason"].astype(str).str[:55] if "reason" in _sig_sim.columns else ""
        if "volatility_regime" in _sig_sim.columns:
            _sig_sim["Regime"] = _sig_sim["volatility_regime"].map(
                lambda x: {0: "Sideways", 1: "Normal", 2: "Volatile"}.get(int(float(x)) if pd.notna(x) else 1, "?")
            )
        if "open_positions" in _sig_sim.columns and "max_positions" in _sig_sim.columns:
            _sig_sim["Pos"] = _sig_sim.apply(
                lambda r: f"{int(float(r['open_positions']))}/{int(float(r['max_positions']))}", axis=1
            )

        # ── Cột mô phỏng ──────────────────────────────────────────────
        def _fmt_sim_outcome(v):
            if v == "win":    return "✅ WIN"
            if v == "loss":   return "❌ LOSS"
            if v == "open":   return "⏳ Chưa KQ"
            return "—"

        _sig_sim["KQ mô phỏng"] = _sig_sim["sim_outcome"].map(_fmt_sim_outcome)
        _sig_sim["P&L sim ($)"] = _sig_sim["sim_pnl"].map(
            lambda x: f"+{x:.2f}" if pd.notna(x) and x > 0 else (f"{x:.2f}" if pd.notna(x) else "—")
        )
        _sig_sim["Bars đến KQ"] = _sig_sim["sim_bars"].map(
            lambda x: str(int(x)) if pd.notna(x) else "—"
        )

        # ── Thống kê mô phỏng ─────────────────────────────────────────
        _has_outcome = _sig_sim[_sig_sim["sim_outcome"].isin(["win", "loss"])]
        if not _has_outcome.empty:
            _sw = (_has_outcome["sim_outcome"] == "win").sum()
            _sl_c = (_has_outcome["sim_outcome"] == "loss").sum()
            _swr = _sw / len(_has_outcome)
            _spnl = _has_outcome["sim_pnl"].sum()
            _st1, _st2, _st3, _st4 = st.columns(4)
            _st1.metric("Tổng tín hiệu sim", len(_has_outcome),
                        delta=f"/ {len(_sig_sim)} tổng")
            _st2.metric("Win Rate (mô phỏng)", f"{_swr:.1%}",
                        delta=f"{_sw}W / {_sl_c}L")
            _st3.metric("P&L tổng (mô phỏng)", f"${_spnl:+.2f}")
            # Lệnh bị lọc nhưng sẽ win
            _filtered_win = _has_outcome[
                (_has_outcome["should_trade"] == False) & (_has_outcome["sim_outcome"] == "win")
            ]
            _st4.metric("Lọc bỏ nhưng WIN", len(_filtered_win),
                        delta="bỏ lỡ" if len(_filtered_win) > 0 else "ok")

        _show_cols = [c for c in [
            "Thời gian", "Tín hiệu", "Vào lệnh?", "Confidence", "Strategy Sc.",
            "Entry", "SL", "TP", "R/R", "KQ mô phỏng", "P&L sim ($)", "Bars đến KQ",
            "Regime", "Pos", "Lý do"
        ] if c in _sig_sim.columns]

        def _highlight_signal(row):
            kq = row.get("KQ mô phỏng", "")
            entered = row.get("Vào lệnh?", "") == "✅ CÓ"
            if kq == "✅ WIN" and entered:
                return ["background-color: #10b98118"] * len(row)
            if kq == "✅ WIN" and not entered:
                return ["background-color: #10b98108"] * len(row)
            if kq == "❌ LOSS" and entered:
                return ["background-color: #ef444415"] * len(row)
            if kq == "❌ LOSS" and not entered:
                return ["background-color: #ef444408"] * len(row)
            return [""] * len(row)

        st.dataframe(_sig_sim[_show_cols].style.apply(_highlight_signal, axis=1),
                     use_container_width=True)
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:0.72rem;color:#94a3b8;'
            'padding:8px 14px;background:#0f172a;border-radius:8px;border:1px solid #1e293b">'
            '<span>🟩 Vào lệnh & WIN</span>'
            '<span>🟩 Lọc bỏ nhưng sẽ WIN</span>'
            '<span>🟥 Vào lệnh & LOSS</span>'
            '<span>🟥 Lọc bỏ & tránh được LOSS</span>'
            '<span>⏳ Chưa có kết quả</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Closed trades với kết quả win/loss ──────────────────────────
        _closed = _ltrades if "Acc 1" in _sel_acc else _ltrades_acc2
        _acc_label_closed = "ACC1 (270832477)" if "Acc 1" in _sel_acc else "ACC2 (433326057)"
        st.divider()

        # ── Filter controls (always render so both ACC1 & ACC2 show the section) ──
        _flt_col1, _flt_col2, _flt_col3 = st.columns([2, 2, 3])
        _date_options = ["Tất cả", "Hôm nay", "7 ngày", "30 ngày", "Theo session"]
        _date_filter = _flt_col1.selectbox("Khoảng thời gian", _date_options, key="closed_date_filter")
        _now_utc = pd.Timestamp.utcnow()
        if not _closed.empty:
            if _date_filter == "Hôm nay":
                _closed = _closed[_closed["time"] >= _now_utc.normalize()]
            elif _date_filter == "7 ngày":
                _closed = _closed[_closed["time"] >= _now_utc - pd.Timedelta(days=7)]
            elif _date_filter == "30 ngày":
                _closed = _closed[_closed["time"] >= _now_utc - pd.Timedelta(days=30)]
            elif _date_filter == "Theo session":
                _sessions = sorted(_closed["session_id"].dropna().unique(), reverse=True)
                _sel_sess = _flt_col2.selectbox("Session (khởi động)", _sessions, key="closed_sess_filter")
                _closed = _closed[_closed["session_id"] == _sel_sess]

        _side_filter = _flt_col3.radio("Side", ["Tất cả", "BUY", "SELL"], horizontal=True, key="closed_side_filter")
        if not _closed.empty and _side_filter != "Tất cả":
            _closed = _closed[_closed["side"].str.upper() == _side_filter]

        st.subheader(f"💰 Lệnh đã đóng — {_acc_label_closed} ({len(_closed)} lệnh, toàn bộ lịch sử)")

        if _closed.empty:
            st.info("Chưa có lệnh đóng nào được ghi lại. Dữ liệu sẽ xuất hiện khi bot đóng lệnh đầu tiên.")
        else:
            _closed_disp = _closed.copy()
            def _result_label(row: pd.Series) -> str:
                ct = str(row.get("close_type", "")).strip().upper()
                is_w = bool(row.get("is_win", False))
                if is_w:
                    ct_tag = f" [{ct}]" if ct in ("TP", "EA") else ""
                    return f"✅ THẮNG{ct_tag}"
                elif ct == "SL" and float(row.get("pnl", -1)) >= 0:
                    return "⚡ SL Hoà"
                else:
                    return "❌ THUA"
            _closed_disp["Kết quả"]   = _closed_disp.apply(_result_label, axis=1)
            _closed_disp["P&L ($)"]   = _closed_disp["pnl"].map(lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}")
            _closed_disp["Entry"]     = _closed_disp["open_price"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            _closed_disp["Exit"]      = _closed_disp["close_price"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            _closed_disp["Lot"]       = _closed_disp["volume"].map(lambda x: f"{x:.2f}")
            _closed_disp["Thời gian"] = _closed_disp["time"].astype(str).str[:16]
            _closed_disp["Side"]      = _closed_disp["side"].str.upper() if "side" in _closed_disp.columns else ""
            _closed_disp["Session"]   = _closed_disp.get("session_id", "—").fillna("—")
            _w = int(_closed_disp["is_win"].sum())
            _l = len(_closed_disp) - _w
            _total_pnl = _closed_disp["pnl"].sum()
            _wr = _w / len(_closed_disp) if len(_closed_disp) > 0 else 0
            _best = _closed_disp["pnl"].max()
            _worst = _closed_disp["pnl"].min()
            _c1, _c2, _c3, _c4, _c5, _c6 = st.columns(6)
            _c1.metric("Tổng lệnh", len(_closed_disp))
            _c2.metric("Win Rate", f"{_wr:.1%}", delta=f"{_w}W / {_l}L")
            _c3.metric("Tổng P&L", f"${_total_pnl:+.2f}")
            _c4.metric("Avg P&L/lệnh", f"${_total_pnl/len(_closed_disp):+.2f}" if len(_closed_disp) > 0 else "$0")
            _c5.metric("Lệnh tốt nhất", f"${_best:+.2f}")
            _c6.metric("Lệnh tệ nhất", f"${_worst:+.2f}")

            # Cumulative equity curve
            _eq_cum = _closed_disp["pnl"].cumsum()
            if len(_eq_cum) > 1:
                st.caption("📈 Equity curve tích lũy (tất cả sessions)")
                st.line_chart(_eq_cum.pipe(_ds).rename("Cumulative P&L ($)"), height=180)

            def _highlight_trade(row):
                if row.get("Kết quả", "") == "✅ THẮNG":
                    return ["background-color: #10b98115"] * len(row)
                return ["background-color: #ef444412"] * len(row)

            _cl_cols = [c for c in ["Thời gian", "Session", "Side", "Lot", "Entry", "Exit", "P&L ($)", "Kết quả"] if c in _closed_disp.columns]
            st.dataframe(
                _closed_disp[_cl_cols].sort_values("Thời gian", ascending=False).head(200).style.apply(_highlight_trade, axis=1),
                use_container_width=True,
            )
    else:
        st.info("Chưa có tín hiệu. Bot đang chạy...")


with tab_live:
    _render_live_tab()


# =============================================================================
# TAB 2 — SIGNAL ANALYSIS
# =============================================================================
with tab_analysis:
    st.markdown(
        '<div style="border-bottom:2px solid #1e293b;padding-bottom:12px;margin-bottom:18px">'
        '<span style="font-size:1.35rem;font-weight:800;color:#f1f5f9">Phân tích Chi tiết Tín hiệu</span>'
        '<span style="color:#64748b;font-size:0.82rem;margin-left:12px">'
        'Mỗi bước bot cần PASS để vào lệnh</span></div>',
        unsafe_allow_html=True,
    )

    _ana_options = ["Acc 1 — 270832477", "Acc 2 — 433326057"]
    _ana_sel = st.radio("Tài khoản:", _ana_options, horizontal=True, key="ana_acc_selector")
    _analysis_signals = live_signals if "Acc 1" in _ana_sel else live_signals_acc2
    _ana_threshold = threshold_val_acc2 if "Acc 2" in _ana_sel else float(
        model_meta.get("decision_threshold") or model_meta.get("selected_threshold") or 0.5
    )

    if _analysis_signals.empty:
        st.warning("Chưa có tín hiệu. Đợi bot chạy ít nhất 1 chu kỳ.")
    else:
        latest  = _analysis_signals.iloc[0]
        conf    = float(latest.get("confidence", 0) or 0)
        side    = str(latest.get("side", "flat"))
        score   = float(latest.get("strategy_score", 0) or 0)
        regime  = int(float(latest.get("volatility_regime", 1) or 1))
        traded  = bool(latest.get("should_trade", False))
        reason  = str(latest.get("reason", ""))
        r_label = {0: "Sideways", 1: "Normal", 2: "Strong Vol"}.get(regime, "?")
        reg_mult = {0: 0.5, 1: 1.0, 2: 1.2}.get(regime, 1.0)
        reg_thr  = {0: 0.05, 1: 0.30, 2: 0.30}.get(regime, 0.30)
        blocked_hours  = [7, 10, 11, 22]
        now_utc        = dt.datetime.utcnow()
        cur_hour       = now_utc.hour
        is_blocked_now = cur_hour in blocked_hours

        st.markdown(f"**Phân tích tại:** `{str(latest.get('time',''))[:19]}`")
        st.divider()

        # 6-step flow
        st.markdown(
            '<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin:16px 0 10px">'
            'Luồng quyết định — 6 bước <span style="color:#64748b;font-weight:400">'
            '(ICT → Wyckoff → Execution)</span></div>',
            unsafe_allow_html=True,
        )
        fl, fr = st.columns([1, 1])
        with fl:
            ml_pass = conf >= _ana_threshold
            _step_ok(1, "Tin cậy ML Model",
                     ml_pass,
                     f"confidence = {conf:.1%} {'>=  ' if ml_pass else '< '} ngưỡng {_ana_threshold:.0%}")
            time_pass = not is_blocked_now
            _step_ok(2, "Bộ lọc thời gian",
                     time_pass,
                     f"UTC {cur_hour:02d}:xx — {'Giờ được phép' if time_pass else f'Giờ bị chặn {blocked_hours}'}")
            _step_ok(3, "Chế độ thị trường",
                     None,
                     f"{r_label} — multiplier={reg_mult}x | ngưỡng score={reg_thr}")
            score_pass = abs(score) >= reg_thr
            _step_ok(4, "Strategy Score (ICT+Wyckoff+Momentum)",
                     score_pass,
                     f"score={score:+.4f} abs={abs(score):.4f} {'>=  ' if score_pass else '< '}{reg_thr}")
            trend_ok = "trend" not in reason.lower()
            _step_ok(5, "Xu hướng D1 == H1",
                     trend_ok if not traded else True,
                     "OK" if trend_ok else reason)
            if "account_balance" in _analysis_signals.columns:
                opn_n = int(float(latest.get("open_positions", 0) or 0))
                mx_n  = int(float(latest.get("max_positions", 1) or 1))
                pos_ok = opn_n < mx_n
            else:
                pos_ok = "position" not in reason.lower()
            _step_ok(6, "Giới hạn số lệnh",
                     pos_ok if not traded else True,
                     f"{opn_n}/{mx_n}" if "account_balance" in _analysis_signals.columns else "")

        with fr:
            dec_c = GREEN if traded else RED
            st.markdown(
                f'<div style="background:linear-gradient(135deg,{dec_c}0a,{dec_c}18);'
                f'border:1px solid {dec_c}40;border-radius:16px;'
                f'padding:28px;text-align:center;margin-bottom:20px;'
                f'box-shadow:0 0 30px {dec_c}08">'
                f'<div style="font-size:2.4rem;font-weight:900;color:{dec_c};'
                f'text-shadow:0 0 20px {dec_c}40">{"VÀO LỆNH" if traded else "KHÔNG VÀO"}</div>'
                f'<div style="font-size:0.85rem;color:#94a3b8;margin-top:10px">'
                f'{"Tất cả điều kiện thỏa mãn ✓" if traded else reason}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            _threshold_bar("ML Confidence", conf, _ana_threshold)
            _threshold_bar("Strategy Score (abs)", abs(score), reg_thr)
            prec = float(model_meta.get("precision", 0))
            rec  = float(model_meta.get("recall", 0))
            f1v  = float(model_meta.get("f1", 0))
            _threshold_bar("Model Precision", prec, 0.45)
            _threshold_bar("Model Recall",    rec,  0.50)
            _threshold_bar("Model F1",        f1v,  0.48)

        st.divider()
        st.markdown(
            '<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin:16px 0 10px">'
            'Thành phần Chiến lược</div>',
            unsafe_allow_html=True,
        )
        sc_l, sc_r = st.columns([1, 1])
        with sc_l:
            spct = min(abs(score) / 1.2, 1.0) * 100
            sc   = GREEN if abs(score) >= 0.3 else (AMBER if abs(score) >= 0.1 else RED)
            st.markdown(
                f'<div style="text-align:center;padding:22px;background:linear-gradient(135deg,#0f172a,#151d2b);'
                f'border-radius:14px;border:1px solid #1e293b">'
                f'<div style="font-size:3.2rem;font-weight:900;color:{sc};text-shadow:0 0 20px {sc}30">{score:+.4f}</div>'
                f'<div style="font-size:0.78rem;color:#64748b;margin:4px 0 12px;letter-spacing:0.05em">STRATEGY SCORE</div>'
                f'<div style="background:#0a0e17;border-radius:20px;height:14px;margin:0 20px;overflow:hidden;'
                f'border:1px solid #1e293b">'
                f'<div style="width:{spct:.0f}%;background:linear-gradient(90deg,{sc}40,{sc});height:14px;'
                f'border-radius:20px;transition:width 0.6s ease"></div>'
                f'</div>'
                f'<div style="font-size:0.82rem;font-weight:700;color:{sc};margin-top:10px">'
                f'{"✓ Đủ mạnh" if abs(score) >= reg_thr else "✗ Quá yếu"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with sc_r:
            st.code(
                f"score = (ICT x 0.40) + (Wyckoff x 0.30) + (Momentum x 0.30)\n"
                f"      x hệ số chế độ\n\n"
                f"hệ số chế độ = {reg_mult}  ({r_label})\n"
                f"ngưỡng hiện tại   = {reg_thr}\n\n"
                f"score hiện tại = {score:.4f}\n"
                f"|score| x mult = {abs(score) * reg_mult:.4f}",
                language=None,
            )

        for cname, cweight, cdesc, ccond in [
            ("H4 ICT Structure (BOS/ChoCH/FVG/OB/Confluence)", "60% trọng số",
             "H4: Break-of-Structure, Change-of-Character, Fair Value Gap, Order Block, "
             "Displacement, Equal-High/Low, Market Structure Bias, ICT Confluence, Premium/Discount",
             "h4_bos | h4_choch | h4_fvg | h4_order_block | h4_ict_confluence | h4_premium_discount"),
            ("H1 Wyckoff Phase Analysis", "20% trọng số",
             "Hourly bias xác nhận, VSA (Volume Spread Analysis), Wyckoff Spring/Upthrust detection. "
             "Spring(1)=Bullish, Upthrust(-1)=Bearish, 0=Trung lập",
             "wyckoff_spring_signal != 0 | vsa_signal != 0 | hourly_bias aligned"),
            ("M15 Tín hiệu Thực thi (RSI/MACD/ATR/Momentum)", "20% trọng số",
             "M15 thời điểm vào lệnh: RSI, MACD histogram, ATR ratio, range efficiency, liquidity sweep, "
             "order flow proxy, chế độ biến động, session return, tick volume zscore, spread, "
             "kill zone flag (London/NY open), Judas swing",
             "rsi>55+macd>0=Bull | rsi<45+macd<0=Bear | kill_zone_flag=1"),
        ]:
            with st.expander(f"{cname} — {cweight}"):
                st.markdown(f"**Mô tả:** {cdesc}")
                st.markdown(f"**Điều kiện:** `{ccond}`")

        st.divider()
        st.markdown(
            '<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin:16px 0 10px">'
            'Chế độ biến động</div>',
            unsafe_allow_html=True,
        )
        vr_l, vr_r = st.columns([1, 2])
        with vr_l:
            ri = {0: ("SIDEWAYS", AMBER, "Đi ngang — ngưỡng=0.05, hệ số=0.5"),
                  1: ("NORMAL",   BLUE,  "Bình thường — ngưỡng=0.30"),
                  2: ("STRONG",   GREEN, "Biến động mạnh — ngưỡng=0.30, hệ số=1.2")}
            rl, rc, rdesc = ri.get(regime, ("?", GREY, ""))
            st.markdown(
                f'<div style="background:linear-gradient(135deg,{rc}0a,{rc}18);'
                f'border:1px solid {rc}40;border-radius:14px;'
                f'padding:24px;text-align:center;box-shadow:0 0 20px {rc}08">'
                f'<div style="font-size:2.2rem;font-weight:900;color:{rc};text-shadow:0 0 15px {rc}30">{rl}</div>'
                f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:10px">{rdesc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with vr_r:
            if "volatility_regime" in _analysis_signals.columns:
                rc_counts = _analysis_signals["volatility_regime"].map(
                    {0: "Sideways", 1: "Normal", 2: "Strong"}
                ).value_counts()
                if not rc_counts.empty:
                    st.bar_chart(rc_counts, height=180)
            st.markdown(f"=> Hệ số hiện tại: **{r_label}** | hệ số nhân = {reg_mult}x")

        st.divider()
        st.markdown(
            '<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin:16px 0 10px">'
            'Bộ lọc thời gian (UTC)</div>',
            unsafe_allow_html=True,
        )
        tf_l, tf_r = st.columns([1, 2])
        with tf_l:
            tf_c = RED if is_blocked_now else GREEN
            st.markdown(
                f'<div style="background:linear-gradient(135deg,{tf_c}0a,{tf_c}18);'
                f'border:1px solid {tf_c}40;border-radius:14px;'
                f'padding:24px;text-align:center;box-shadow:0 0 20px {tf_c}08">'
                f'<div style="font-size:1.6rem;font-weight:900;color:{tf_c}">'
                f'{"⛔ Giờ bị chặn" if is_blocked_now else "✓ Giờ giao dịch"}</div>'
                f'<div style="font-size:1rem;color:#f1f5f9;margin-top:8px;font-weight:600">UTC {cur_hour:02d}:xx</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with tf_r:
            hours_df = pd.DataFrame([{
                "Giờ UTC": f"{h:02d}:00",
                "Trạng thái": "Bị chặn" if h in blocked_hours else "OK",
            } for h in range(24)])
            st.dataframe(hours_df, use_container_width=True, height=240)

        st.divider()
        st.markdown(
            '<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin:16px 0 10px">'
            'Luồng quyết định đầy đủ</div>',
            unsafe_allow_html=True,
        )
        st.code(
                f"Dữ liệu: D1=100, H4=200, H1=500, M15=300 nhịp\n"
                f"=> 28 Features: D1(1) + H4(9:ICT) + H1(3:Wyckoff) + M15(15:execution)\n"
            f"   [{('PASS' if conf >= _ana_threshold else 'FAIL')}] >= {_ana_threshold:.0%}?\n"
            f"=> Bộ lọc thời gian UTC {cur_hour:02d}\n"
            f"   [{('PASS' if not is_blocked_now else 'FAIL')}] not in {blocked_hours}?\n"
            f"=> Strategy score = {score:.4f}\n"
            f"   [{('PASS' if abs(score) >= reg_thr else 'FAIL')}] |score| >= {reg_thr}?\n"
            f"=> Kết quả: {'VÀO LỆNH' if traded else 'KHÔNG VÀO — ' + reason}",
            language=None,
        )

        st.divider()
        st.markdown(
            '<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin:16px 0 10px">'
            'So sánh tín hiệu vào vs không vào lệnh</div>',
            unsafe_allow_html=True,
        )
        if len(_analysis_signals) >= 5:
            trd_s  = _analysis_signals[_analysis_signals["should_trade"] == True]
            ntrd_s = _analysis_signals[_analysis_signals["should_trade"] == False]
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Tổng tín hiệu", len(_analysis_signals))
            cm2.metric("Đã vào lệnh",   len(trd_s), delta=f"{len(trd_s)/len(_analysis_signals):.1%}")
            cm3.metric("Không vào",     len(ntrd_s))
            cm4.metric("Conf TB không vào", _pct(ntrd_s["confidence"].mean()) if not ntrd_s.empty else "n/a")
            if not trd_s.empty and not ntrd_s.empty:
                st.dataframe(pd.DataFrame({
                    "Metric": ["Avg Confidence", "Avg |Score|"],
                    "Vào lệnh": [
                        f"{trd_s['confidence'].mean():.1%}",
                        f"{trd_s['strategy_score'].abs().mean():.4f}" if "strategy_score" in trd_s.columns else "n/a",
                    ],
                    "Không vào": [
                        f"{ntrd_s['confidence'].mean():.1%}",
                        f"{ntrd_s['strategy_score'].abs().mean():.4f}" if "strategy_score" in ntrd_s.columns else "n/a",
                    ],
                }), use_container_width=True)
            if "reason" in _analysis_signals.columns and not ntrd_s.empty:
                st.bar_chart(ntrd_s["reason"].value_counts(), height=200)

        # Feature importance
        st.divider()
        st.markdown(
            '<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin:16px 0 10px">'
            '28 Features — <span style="color:#6366f1">ICT+Wyckoff</span> Model Map</div>',
            unsafe_allow_html=True,
        )
        feat_map_l, feat_map_r = st.columns([1, 1])
        with feat_map_l:
            st.markdown("""
**D1 (1 feature — Daily Bias)**
- `daily_bias` — D1 trend direction

**H4 (9 features — ICT Structure)**
- `h4_bos` — Break of Structure
- `h4_choch` — Change of Character
- `h4_fvg` — Fair Value Gap
- `h4_order_block` — Order Block presence
- `h4_displacement` — Displacement candle
- `h4_ehl` — Equal High/Low detection
- `h4_market_structure_bias` — Bias score
- `h4_ict_confluence` — ICT multi-factor confluence
- `h4_premium_discount` — Premium/Discount zone

**H1 (3 features — Wyckoff)**
- `hourly_bias` — H1 directional bias
- `vsa_signal` — Volume Spread Analysis
- `wyckoff_spring_signal` — Spring/Upthrust detection
""")
        with feat_map_r:
            st.markdown("""
**M15 (15 features — Execution)**
- `trend_alignment` — M15 trend vs H1
- `rsi` — RSI(14)
- `macd_hist` — MACD histogram
- `atr_ratio` — ATR normalized
- `range_efficiency` — Bar efficiency
- `liquidity_sweep` — Liquidity level swept
- `order_flow_proxy` — Order flow direction
- `wyckoff_phase` — Wyckoff phase (accumulation/dist.)
- `volatility_regime` — Sideways/Normal/Strong
- `session_return` — Session price return
- `tick_volume_zscore` — Volume Z-score
- `spread_points` — Current spread
- `strategy_score` — Combined ICT+Wyckoff+Mom score
- `kill_zone_flag` — London/NY open active
- `judas_swing_signal` — Fake move detection
""")

        st.divider()
        st.markdown(
            '<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin:16px 0 10px">'
            'Feature Importance <span style="color:#64748b">(Top 25 — HistGBC)</span></div>',
            unsafe_allow_html=True,
        )
        fi_df = load_feature_importance()
        if fi_df is not None and not fi_df.empty:
            st.bar_chart(fi_df.set_index("feature")["importance"].sort_values(), height=400)
            st.dataframe(fi_df[["feature", "importance", "abs_importance"]].round(6),
                         use_container_width=True)
        else:
            st.info("Chưa load được model_ict_wyckoff.pkl — cần có model trained. Chạy: python -m xauusd_ai.main train --config configs/train_ict_wyckoff_2022_2026.yaml")


# =============================================================================
# TAB 3 — P&L & VON
# =============================================================================
with tab_pnl:
    st.markdown(
        '<div style="border-bottom:2px solid #1e293b;padding-bottom:12px;margin-bottom:18px">'
        '<span style="font-size:1.35rem;font-weight:800;color:#f1f5f9">P&L & Vốn</span>'
        '<span style="color:#64748b;font-size:0.82rem;margin-left:12px">'
        'Lợi nhuận và Thua lỗ</span></div>',
        unsafe_allow_html=True,
    )

    # ── Live Account P&L sub-tabs (ACC1 / ACC2 / So Sanh) ────────────────────
    _pnl_acc1, _pnl_acc2, _pnl_both = st.tabs([
        "🏦 Live ACC1 — 270832477",
        "🏦 Live ACC2 — 433326057",
        "⚖️ So sánh 2 Tài khoản",
    ])

    def _render_live_pnl(lt: pd.DataFrame, ls: pd.DataFrame, acc_label: str) -> None:
        """Render live P&L section for one account."""
        if not lt.empty:
            lts = summarize_trades(lt)
            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("Net P&L Live",   f"${lts['net_profit']:.2f}", delta=f"{lts['net_profit']:+.2f}")
            lc2.metric("Win Rate",       f"{lts['win_rate']:.1%}")
            lc3.metric("Tổng lệnh",      lts["trades"])
            _pf = lts["gross_profit"] / max(abs(lts["gross_loss"]), 1e-9)
            lc4.metric("Profit Factor",  f"{_pf:.2f}")

            # Per-trade PnL bar
            if "pnl" in lt.columns:
                st.markdown("**P&L từng lệnh live:**")
                _pnl_idx = lt.set_index("time")["pnl"] if "time" in lt.columns else lt["pnl"]
                st.bar_chart(_pnl_idx, height=200)

            # Per-timeframe breakdown using volatility_regime
            if "pnl" in lt.columns:
                st.subheader("📊 P&L theo Khung Thời Gian (Chế độ)")
                _reg_map = {0: "Sideways", 1: "Normal", 2: "Strong Vol"}
                if "volatility_regime" in lt.columns:
                    _tf_grp = lt.groupby("volatility_regime").agg(
                        trades=("pnl", "size"),
                        net_pnl=("pnl", "sum"),
                        avg_pnl=("pnl", "mean"),
                        win_rate=("is_win", "mean") if "is_win" in lt.columns else ("pnl", lambda s: (s>0).mean()),
                    ).reset_index()
                    _tf_grp["regime_label"] = _tf_grp["volatility_regime"].map(_reg_map).fillna("Unknown")
                    st.dataframe(_tf_grp[["regime_label", "trades", "net_pnl", "avg_pnl", "win_rate"]].round(4),
                                 use_container_width=True, hide_index=True)

                # Hour-of-day P&L
                if "time" in lt.columns:
                    st.markdown("**P&L theo giờ (UTC):**")
                    _h_grp = lt.assign(hour=lt["time"].dt.hour).groupby("hour").agg(
                        net_pnl=("pnl", "sum"),
                        win_rate=("is_win", "mean") if "is_win" in lt.columns else ("pnl", lambda s: (s>0).mean()),
                        trades=("pnl", "size"),
                    )
                    if not _h_grp.empty:
                        _ha, _hb = st.columns(2)
                        with _ha:
                            st.bar_chart(_h_grp["net_pnl"], height=200)
                        with _hb:
                            st.bar_chart(_h_grp["win_rate"], height=200)

            # Cumulative equity from live closed trades
            if "pnl" in lt.columns and "time" in lt.columns:
                st.markdown("**Đường vốn lũy kế (lệnh live đã đóng):**")
                _lt_eq = lt.sort_values("time").copy()
                if "profit" in _lt_eq.columns:
                    _lt_eq["cum_pnl"] = _lt_eq["profit"].cumsum()
                else:
                    _lt_eq["cum_pnl"] = _lt_eq["pnl"].cumsum()
                st.line_chart(_lt_eq.set_index("time")["cum_pnl"].pipe(_ds), height=220)

            # Detailed trade table
            st.markdown("**Danh sách tất cả lệnh live đã đóng:**")
            _lcols = [c for c in ["time", "ticket", "side", "volume", "open_price", "close_price",
                                   "profit", "swap", "commission", "pnl", "is_win"]
                      if c in lt.columns]
            st.dataframe(lt[_lcols], use_container_width=True)
        else:
            st.info(f"Chưa có lệnh live nào được đóng cho {acc_label}.")

        # Balance history from signal log
        if not ls.empty and "account_balance" in ls.columns:
            st.subheader("Lịch sử Số dư Live")
            _bh = ls[["time", "account_balance"]].dropna().sort_values("time").set_index("time")
            if not _bh.empty:
                st.area_chart(_bh["account_balance"].pipe(_ds), height=200)

    with _pnl_acc1:
        _render_live_pnl(live_trades, live_signals, "ACC1 (270832477)")

    with _pnl_acc2:
        _render_live_pnl(live_trades_acc2, live_signals_acc2, "ACC2 (433326057)")

    with _pnl_both:
        st.subheader("⚖️ So sánh P&L 2 tài khoản")
        _cmp_data = []
        for _lbl, _lt in [("ACC1", live_trades), ("ACC2", live_trades_acc2)]:
            if not _lt.empty and "pnl" in _lt.columns:
                _s = summarize_trades(_lt)
                _pf2 = _s["gross_profit"] / max(abs(_s["gross_loss"]), 1e-9)
                _cmp_data.append({
                    "Tài khoản": _lbl,
                    "Tổng lệnh": _s["trades"],
                    "Thắng": _s["wins"],
                    "Thua": _s["losses"],
                    "Win Rate": f"{_s['win_rate']:.1%}",
                    "Net P&L": f"${_s['net_profit']:.2f}",
                    "Gross Profit": f"${_s['gross_profit']:.2f}",
                    "Gross Loss": f"${_s['gross_loss']:.2f}",
                    "Profit Factor": f"{_pf2:.2f}",
                })
        if _cmp_data:
            st.dataframe(pd.DataFrame(_cmp_data), use_container_width=True, hide_index=True)
            # Side-by-side cumulative equity curves
            _eq_combined = {}
            for _lbl, _lt in [("ACC1", live_trades), ("ACC2", live_trades_acc2)]:
                if not _lt.empty and "pnl" in _lt.columns and "time" in _lt.columns:
                    _lt2 = _lt.sort_values("time").copy()
                    _col = "profit" if "profit" in _lt2.columns else "pnl"
                    _eq_combined[_lbl] = _lt2.set_index("time")[_col].cumsum()
            if _eq_combined:
                st.markdown("**Đường vốn lũy kế 2 tài khoản:**")
                _eq_df = pd.DataFrame(_eq_combined)
                _eq_df = _eq_df.apply(lambda c: c.pipe(_ds))
                st.line_chart(_eq_df, height=280)
        else:
            st.info("Chưa có lệnh live nào được đóng. Dữ liệu sẽ hiển thị khi bot đóng lệnh.")

    st.divider()
    st.subheader("📊 Backtest P&L (lịch sử)")

    if not trades.empty:
        ts = summarize_trades(trades)
        ds, df = summarize_daily(trades)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Lợi nhuận ròng",    f"${ts['net_profit']:.2f}", delta=f"{ts['net_profit']:+.2f}")
        k2.metric("Tỷ lệ thắng",      f"{ts['win_rate']:.1%}")
        k3.metric("Tổng lệnh",  ts["trades"])
        k4.metric("Profit Factor", backtest_report.get("profit_factor", "n/a"))

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Gross Profit",  f"${ts['gross_profit']:.2f}")
        k6.metric("Gross Loss",    f"${ts['gross_loss']:.2f}")
        k7.metric("Max Drawdown",  str(backtest_report.get("max_drawdown_pct", "n/a")) + "%")
        k8.metric("Sharpe Ratio",  str(backtest_report.get("sharpe_ratio", "n/a")))

        st.divider()
        st.subheader("Đường vốn & Drawdown")
        if "balance_after" in trades.columns:
            eq_df = trades.dropna(subset=["time", "balance_after"]).set_index("time").sort_index()
            if not eq_df.empty:
                eq_s = eq_df["balance_after"]
                dd_s = compute_drawdown(eq_s)
                # Cap to 1000 points to avoid MemoryError in Altair serialization
                _max_pts = 1000
                eq_plot = eq_s.iloc[:: max(1, len(eq_s) // _max_pts)].round(2)
                dd_plot = dd_s.iloc[:: max(1, len(dd_s) // _max_pts)].round(3)
                eq_c, dd_c = st.columns([2, 1])
                with eq_c:
                    st.markdown("**Đường vốn**")
                    st.line_chart(eq_plot, height=280)
                with dd_c:
                    st.markdown("**Drawdown (%)**")
                    st.area_chart(dd_plot, height=280)
                min_dd = float(dd_s.min())
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,{RED}08,{RED}14);'
                    f'border-left:4px solid {RED};border:1px solid {RED}30;'
                    f'padding:12px 18px;border-radius:10px">'
                    f'<span style="color:{RED};font-weight:700;font-size:0.9rem">Max Drawdown: {min_dd:.2f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.subheader("Chuỗi thắng/thua")
        if "is_win" in trades.columns:
            stk = compute_streaks(trades.sort_values("time")["is_win"])
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Tổng Thắng",      stk["wins"])
            s2.metric("Tổng Thua",    stk["losses"])
            cur = stk["current_streak"]
            cur_c = GREEN if cur > 0 else RED
            s3.markdown(_card("Chuỗi hiện tại", f"{abs(cur)}", "thắng" if cur > 0 else "thua", cur_c),
                        unsafe_allow_html=True)
            s4.metric("Chuỗi thắng cao nhất",  stk["max_win_streak"])
            s5.metric("Chuỗi thua cao nhất", stk["max_loss_streak"])

            recent = trades.sort_values("time").tail(60)["is_win"].tolist()
            html = '<div style="display:flex;flex-wrap:wrap;gap:3px;padding:8px">'
            for w in recent:
                c = GREEN if w else RED
                html += f'<div style="width:18px;height:18px;background:{c};border-radius:3px" title="{"Thắng" if w else "Thua"}"></div>'
            html += "</div>"
            st.markdown("**Chuỗi thắng/thua 60 lệnh gần nhất:**")
            st.markdown(html, unsafe_allow_html=True)

        st.divider()
        st.subheader("Phân phối P&L")
        if "pnl" in trades.columns:
            pd1, pd2 = st.columns(2)
            with pd1:
                st.markdown("**Biểu đồ P&L:**")
                st.bar_chart(trades["pnl"].value_counts(bins=20).sort_index(), height=220)
            with pd2:
                st.markdown("**P&L theo ngày:**")
                if not df.empty:
                    st.bar_chart(df.set_index("date")["net_pnl"], height=220)

        st.divider()
        st.subheader("Hiệu suất theo giờ vào lệnh (UTC)")
        if "time" in trades.columns:
            hourly = trades.assign(hour=trades["time"].dt.hour).groupby("hour").agg(
                trades=("pnl", "size"),
                total_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                win_rate=("is_win", "mean"),
            )
            if not hourly.empty:
                h1, h2 = st.columns(2)
                with h1:
                    st.markdown("**P&L theo giờ:**")
                    st.bar_chart(hourly["total_pnl"], height=220)
                with h2:
                    st.markdown("**Tỷ lệ thắng theo giờ:**")
                    st.bar_chart(hourly["win_rate"], height=220)
                st.dataframe(hourly.round(4), use_container_width=True)

        st.divider()
        st.subheader("Danh sách lệnh (200 gần nhất)")
        dcols = [c for c in ["time", "side", "entry_price", "exit_price",
                               "pnl", "is_win", "balance_after", "drawdown", "realized_rr"]
                  if c in trades.columns]
        st.dataframe(trades[dcols].tail(200), use_container_width=True)
    else:
        st.info("Chưa có dữ liệu backtest. Chạy: python scripts/backtest_ict_wyckoff.py")


# =============================================================================
# TAB 4 — HOC LIEN TUC
# =============================================================================
with tab_learning:
    st.markdown(
        '<div style="border-bottom:2px solid #1e293b;padding-bottom:12px;margin-bottom:18px">'
        '<span style="font-size:1.35rem;font-weight:800;color:#f1f5f9">Học Liên Tục</span>'
        '<span style="color:#64748b;font-size:0.82rem;margin-left:12px">'
        'Giám sát Tự học — Retrain HistGBC 28 features</span></div>',
        unsafe_allow_html=True,
    )

    # Per-account learning sub-tabs
    _learn_acc1_tab, _learn_acc2_tab, _learn_both_tab = st.tabs([
        "🤖 Học — ACC1 (270832477)",
        "🤖 Học — ACC2 (433326057)",
        "📊 So sánh & Tổng hợp",
    ])

    def _render_learning_tab(
        sl_events: list,
        acc_label: str,
        log_file: str = "live_bot_log.txt",
        interval_min: int = 30,
    ) -> None:
        """Comprehensive real-time learning status panel for one account."""
        import time as _time

        # ── Bot alive detection (status file modified < 15 min ago) ─────────
        status_filename = "live_status_acc2.json" if "ACC2" in acc_label else "live_status_acc1.json"
        status_path = OUTPUTS / status_filename
        bot_alive = False
        log_age_sec: float | None = None
        if status_path.exists():
            log_age_sec = _time.time() - status_path.stat().st_mtime
            bot_alive = log_age_sec < 900

        # ── Detect if currently training (scan last 120 lines of stderr log) ─
        currently_training = False
        actual_log_file = "live_acc2_stderr.txt" if "ACC2" in acc_label else "live_bot_err.txt"
        actual_log_path = OUTPUTS / actual_log_file
        if bot_alive and actual_log_path.exists():
            try:
                tail = actual_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:]
                n_start = sum(1 for l in tail if "LearnerThread: regular retrain started" in l
                              or "LearnerThread: loss-retrain triggered" in l)
                n_done  = sum(1 for l in tail if "retrain done" in l)
                currently_training = n_start > n_done
            except Exception:
                pass

        # ── Parse events ─────────────────────────────────────────────────────
        sl_e   = [e for e in sl_events if e.get("event") in ("self_learn", "live_retrain", "loss_retrain")]
        loss_e = [e for e in sl_events if e.get("event") == "loss_retrain"]
        last_event: dict = sl_e[-1] if sl_e else {}
        last_ts: str = last_event.get("timestamp", "")

        # ── Compute countdown to next learning session ────────────────────────
        now_utc = dt.datetime.now(dt.timezone.utc)
        minutes_elapsed:   float | None = None
        minutes_remaining: float | None = None
        pct_progress = 0.0
        if last_ts:
            try:
                last_dt = dt.datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=dt.timezone.utc)
                elapsed_sec       = (now_utc - last_dt).total_seconds()
                minutes_elapsed   = elapsed_sec / 60
                minutes_remaining = max(0.0, interval_min - minutes_elapsed)
                pct_progress      = min(1.0, minutes_elapsed / interval_min)
            except Exception:
                pass

        # ── Determine state ──────────────────────────────────────────────────
        if not bot_alive:
            state, state_emoji = "OFFLINE",    "🔴"
            state_detail       = "Bot không chạy — hãy khởi động lại"
            state_bg, state_border = "#2a1010", "#ef5350"
        elif currently_training:
            state, state_emoji = "TRAINING",   "⚙️"
            state_detail       = "Đang retrain model trong background (giao dịch không bị gián đoạn)"
            state_bg, state_border = "#0d1f0d", "#66bb6a"
        elif not sl_e:
            state, state_emoji = "WAIT_FIRST", "🔄"
            state_detail       = "Chờ lần học đầu tiên — bot đã start, sẽ học sau ~30 phút"
            state_bg, state_border = "#12102a", "#ab47bc"
        elif minutes_remaining is not None and minutes_remaining < 2:
            state, state_emoji = "IMMINENT",   "⚙️"
            state_detail       = "Sắp đến giờ học hoặc đang chuẩn bị retrain..."
            state_bg, state_border = "#0d1f0d", "#66bb6a"
        else:
            state, state_emoji = "WAITING",    "⏱️"
            rem_str = f"{minutes_remaining:.0f}" if minutes_remaining is not None else "?"
            state_detail = f"Chờ {rem_str} phút nữa đến chu kỳ tự học tiếp theo"
            state_bg, state_border = "#0a1525", "#42a5f5"

        # ── Format helper strings ────────────────────────────────────────────
        bot_dot    = "🟢" if bot_alive else "🔴"
        bot_lbl    = "ĐANG CHẠY" if bot_alive else "KHÔNG CHẠY"
        if log_age_sec is None:
            age_str = "Không có log"
        elif log_age_sec < 60:
            age_str = f"{log_age_sec:.0f} giây trước"
        elif log_age_sec < 3600:
            age_str = f"{log_age_sec/60:.0f} phút trước"
        else:
            age_str = f"{log_age_sec/3600:.1f} giờ trước"
        last_learn_str = last_ts[:16].replace("T", " ") if last_ts else "Chưa có"
        retrain_count  = len(sl_e)

        # ── Status banner ────────────────────────────────────────────────────
        st.markdown(
            f'<div style="background:linear-gradient(135deg,{state_bg},{state_bg}dd);'
            f'border:1px solid {state_border}40;border-left:4px solid {state_border};'
            f'border-radius:14px;padding:20px 24px;margin-bottom:14px;'
            f'box-shadow:0 4px 20px rgba(0,0,0,0.2)">'
            f'<div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap">'
            f'<div style="font-size:2rem;line-height:1;background:{state_border}15;'
            f'width:52px;height:52px;border-radius:14px;display:flex;align-items:center;'
            f'justify-content:center">{state_emoji}</div>'
            f'<div style="flex:1;min-width:220px">'
            f'<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin-bottom:4px">'
            f'{state} <span style="font-weight:400;color:#94a3b8;font-size:0.85rem">{state_detail}</span></div>'
            f'<div style="color:#64748b;font-size:0.78rem">'
            f'{bot_dot} Bot: <b style="color:#94a3b8">{bot_lbl}</b>&ensp;|&ensp;'
            f'Status: <b style="color:#94a3b8">{age_str}</b>&ensp;|&ensp;'
            f'Đã retrain: <b style="color:#94a3b8">{retrain_count} lần</b>'
            f'</div></div>'
            f'<div style="text-align:right;min-width:130px">'
            f'<div style="color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em">Học lần cuối</div>'
            f'<div style="color:#f1f5f9;font-weight:700;font-size:1rem">{last_learn_str}</div>'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

        # ── Progress bar + state message ─────────────────────────────────────
        if state == "WAITING" and minutes_remaining is not None:
            mins_done = interval_min - minutes_remaining
            st.markdown(
                f'<div style="color:#aaa;font-size:0.82rem;margin-bottom:4px">'
                f'⏰ Tiến độ chu kỳ: <b style="color:#fff">{mins_done:.0f}</b> / {interval_min} phút'
                f' &ensp;—&ensp; còn <b style="color:#42a5f5">{minutes_remaining:.0f} phút</b></div>',
                unsafe_allow_html=True,
            )
            st.progress(pct_progress)
            if minutes_remaining <= 5:
                st.success("🔔 Sắp đến giờ học! Bot sẽ retrain trong vài phút tới.")
        elif state in ("TRAINING", "IMMINENT"):
            st.progress(1.0)
            st.success("⚙️ **Bot đang retrain model** trong background — giao dịch vẫn tiếp tục bình thường.")
        elif state == "WAIT_FIRST":
            st.progress(0.0)
            st.warning("🔄 Bot mới start. Lần học đầu tiên sẽ diễn ra sau **~30 phút** kể từ khi khởi động.")
        elif state == "OFFLINE":
            st.error(
                "🔴 **Bot không chạy.** Khởi động bằng lệnh:\n\n"
                "```\n.venv\\Scripts\\python.exe scripts/live_runner.py live "
                f"--config configs/live_ict_wyckoff.yaml\n```"
            )

        # ── Checklist (khi chưa có event nào) ───────────────────────────────
        if not sl_e and bot_alive:
            st.markdown(
                '<div style="background:linear-gradient(135deg,#0f172a,#151d2b);'
                'border:1px solid #1e293b;border-radius:12px;'
                'padding:16px 20px;margin:10px 0">'
                '<b style="color:#f1f5f9;font-size:0.88rem">📋 Checklist kích hoạt tự học:</b><br/>'
                '<span style="color:#10b981">✓</span> <code>live_learning_enabled: true</code> trong config<br/>'
                '<span style="color:#10b981">✓</span> CSV lịch sử đã preload (3000 rows/timeframe)<br/>'
                '<span style="color:#f59e0b">⏳</span> Chờ đủ 30 phút kể từ khi bot start<br/>'
                '<span style="color:#f59e0b">⏳</span> Dataset > 500 rows (đã OK với preload)'
                '</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Metrics & charts (nếu đã có ít nhất 1 sự kiện học) ─────────────
        if sl_e:
            improved  = [e for e in sl_e if "improved" in e.get("status", "")]
            best_roc  = max((e.get("roc_auc", 0) for e in sl_e), default=0)
            avg_rows  = int(sum(e.get("dataset_rows", 0) for e in sl_e) / max(len(sl_e), 1))
            last_rows = last_event.get("dataset_rows", 0)

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Tổng retrain",        len(sl_e))
            mc2.metric("Model cải thiện",     len(improved),
                       delta=f"{len(improved)/max(len(sl_e),1):.0%} tỷ lệ")
            mc3.metric("Best ROC-AUC",        f"{best_roc:.4f}")
            mc4.metric("Loss retrain",        len(loss_e))
            mc5.metric("Avg dataset rows",    f"{avg_rows:,}")

            # Dataset size indicator
            row_ok_color = "#10b981" if last_rows >= 500 else "#ef4444"
            row_ok_label = "✓ Đủ" if last_rows >= 500 else "✗ Thiếu"
            st.markdown(
                f'<div style="display:inline-block;background:linear-gradient(135deg,{row_ok_color}08,{row_ok_color}12);'
                f'border:1px solid {row_ok_color}40;'
                f'border-radius:8px;padding:5px 14px;font-size:0.78rem;color:{row_ok_color};margin-bottom:10px">'
                f'Dataset lần cuối: <b>{last_rows:,} rows</b> {row_ok_label} (min: 500)</div>',
                unsafe_allow_html=True,
            )

            # Trend charts
            learn_df = pd.DataFrame({
                "ROC-AUC":   [e.get("roc_auc", 0)   for e in sl_e],
                "Precision": [e.get("precision", 0)  for e in sl_e],
                "Recall":    [e.get("recall", 0)     for e in sl_e],
                "F1":        [e.get("f1", 0)         for e in sl_e],
            })
            st.markdown(f"**📈 Tiến trình học {acc_label}: ROC-AUC / Precision / Recall**")
            st.line_chart(learn_df, height=240)

            c_l, c_r = st.columns(2)
            with c_l:
                row_data = [e.get("dataset_rows", 0) for e in sl_e]
                if any(r > 0 for r in row_data):
                    st.markdown("**Dataset size qua từng lần retrain:**")
                    st.bar_chart(pd.DataFrame({"dataset_rows": row_data}), height=160)
            with c_r:
                status_counts = pd.Series([e.get("status", "unknown") for e in sl_e]).value_counts()
                st.markdown("**Kết quả mỗi lần retrain:**")
                st.bar_chart(status_counts, height=160)

            # Detail table
            with st.expander(f"📋 Chi tiết {min(30, len(sl_e))} lần tự học gần nhất", expanded=len(sl_e) <= 5):
                ev_df = pd.DataFrame(sl_e[-30:])
                dcols = [c for c in [
                    "timestamp", "event", "status", "dataset_rows",
                    "roc_auc", "best_roc_auc", "precision", "recall", "f1",
                    "retrain_count", "triggered_by_losses",
                ] if c in ev_df.columns]
                st.dataframe(ev_df[dcols] if dcols else ev_df, use_container_width=True)

            # Loss retrain history
            if loss_e:
                st.divider()
                st.subheader("⚠️ Lịch sử Loss Retrain (retrain sau lệnh thua liên tiếp)")
                l_df  = pd.DataFrame(loss_e)
                l_cols = [c for c in [
                    "timestamp", "triggered_by_losses", "loss_patterns_used",
                    "dataset_rows", "roc_auc", "status",
                ] if c in l_df.columns]
                st.dataframe(l_df[l_cols] if l_cols else l_df, use_container_width=True)
        else:
            st.info(
                f"**Hệ thống tự học chưa có sự kiện nào cho {acc_label}.**\n\n"
                f"👉 Trạng thái: {'Bot đang chạy ✅ — chờ thêm ~30 phút' if bot_alive else 'Bot KHÔNG chạy ❌'}"
            )

    with _learn_acc1_tab:
        _render_learning_tab(learn_events, "ACC1", log_file="live_bot_log.txt")

    with _learn_acc2_tab:
        _render_learning_tab(learn_events_acc2, "ACC2", log_file="live_bot_log_acc2.txt")

    with _learn_both_tab:
        st.subheader("📊 So sánh tốc độ học 2 tài khoản")
        _lc_data = []
        for _acc_lbl, _evs in [("ACC1 (270832477)", learn_events), ("ACC2 (433326057)", learn_events_acc2)]:
            _sl = [e for e in _evs if e.get("event") in ("self_learn", "live_retrain", "loss_retrain")]
            if _sl:
                _imp  = sum(1 for e in _sl if "improved" in e.get("status", ""))
                _best = max((e.get("roc_auc", 0) for e in _sl), default=0)
                _lc_data.append({
                    "Tài khoản":      _acc_lbl,
                    "Tổng retrain":   len(_sl),
                    "Model cải thiện": _imp,
                    "Tỷ lệ cải thiện": f"{_imp/max(len(_sl),1):.1%}",
                    "Best AUC":       f"{_best:.4f}",
                    "Lần cuối":       _sl[-1].get("timestamp", "")[:16],
                })
        if _lc_data:
            st.dataframe(pd.DataFrame(_lc_data), use_container_width=True, hide_index=True)
            _auc_dfs = {}
            for _acc_lbl2, _evs2 in [("ACC1", learn_events), ("ACC2", learn_events_acc2)]:
                _sl2 = [e for e in _evs2 if e.get("event") in ("self_learn", "live_retrain")]
                if _sl2:
                    _auc_dfs[_acc_lbl2] = [e.get("roc_auc", 0) for e in _sl2]
            if _auc_dfs:
                _max_len = max(len(v) for v in _auc_dfs.values())
                _auc_df2 = pd.DataFrame({k: v + [None]*(_max_len - len(v)) for k, v in _auc_dfs.items()})
                st.markdown("**So sánh AUC qua từng lần retrain:**")
                st.line_chart(_auc_df2, height=240)
        else:
            st.info("Chưa có dữ liệu tự học (cả ACC1 và ACC2). Bot cần chạy ít nhất 30 phút.")

        # Learning cycle diagram
        st.divider()
        st.markdown(
            '<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin:16px 0 10px">'
            'Vòng lặp học liên tục</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="background:linear-gradient(135deg,#0a0e17,#111827);border:1px solid #1e293b;'
            'border-radius:14px;padding:22px;font-family:Inter,sans-serif">'
            '<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;justify-content:center">'
            '<div style="background:linear-gradient(135deg,#6366f108,#6366f118);border:1px solid #6366f140;'
            'border-radius:12px;padding:12px 16px;text-align:center;min-width:110px">'
            '<div style="font-size:1.3rem">📥</div>'
            '<div style="color:#6366f1;font-weight:700;font-size:0.72rem;letter-spacing:0.05em">BƯỚC 1</div>'
            '<div style="color:#f1f5f9;font-size:0.78rem;font-weight:600">Fetch Data</div>'
            '<div style="color:#64748b;font-size:0.65rem">MT5/CSV preload</div></div>'
            '<div style="color:#334155;font-size:1.4rem;padding:0 2px">→</div>'
            '<div style="background:linear-gradient(135deg,#10b98108,#10b98118);border:1px solid #10b98140;'
            'border-radius:12px;padding:12px 16px;text-align:center;min-width:110px">'
            '<div style="font-size:1.3rem">🔧</div>'
            '<div style="color:#10b981;font-weight:700;font-size:0.72rem;letter-spacing:0.05em">BƯỚC 2</div>'
            '<div style="color:#f1f5f9;font-size:0.78rem;font-weight:600">Feature Eng.</div>'
            '<div style="color:#64748b;font-size:0.65rem">RSI/MACD/ATR/ICT</div></div>'
            '<div style="color:#334155;font-size:1.4rem;padding:0 2px">→</div>'
            '<div style="background:linear-gradient(135deg,#a855f708,#a855f718);border:1px solid #a855f740;'
            'border-radius:12px;padding:12px 16px;text-align:center;min-width:110px">'
            '<div style="font-size:1.3rem">🤖</div>'
            '<div style="color:#a855f7;font-weight:700;font-size:0.72rem;letter-spacing:0.05em">BƯỚC 3</div>'
            '<div style="color:#f1f5f9;font-size:0.78rem;font-weight:600">Retrain Model</div>'
            '<div style="color:#64748b;font-size:0.65rem">HistGBC 28feat</div></div>'
            '<div style="color:#334155;font-size:1.4rem;padding:0 2px">→</div>'
            '<div style="background:linear-gradient(135deg,#f59e0b08,#f59e0b18);border:1px solid #f59e0b40;'
            'border-radius:12px;padding:12px 16px;text-align:center;min-width:110px">'
            '<div style="font-size:1.3rem">📊</div>'
            '<div style="color:#f59e0b;font-weight:700;font-size:0.72rem;letter-spacing:0.05em">BƯỚC 4</div>'
            '<div style="color:#f1f5f9;font-size:0.78rem;font-weight:600">Evaluate</div>'
            '<div style="color:#64748b;font-size:0.65rem">ROC-AUC / F1</div></div>'
            '<div style="color:#334155;font-size:1.4rem;padding:0 2px">→</div>'
            '<div style="background:linear-gradient(135deg,#10b98108,#10b98118);border:1px solid #10b98140;'
            'border-radius:12px;padding:12px 16px;text-align:center;min-width:110px">'
            '<div style="font-size:1.3rem">💾</div>'
            '<div style="color:#10b981;font-weight:700;font-size:0.72rem;letter-spacing:0.05em">BƯỚC 5</div>'
            '<div style="color:#f1f5f9;font-size:0.78rem;font-weight:600">Deploy/Skip</div>'
            '<div style="color:#64748b;font-size:0.65rem">Nếu AUC tốt hơn</div></div>'
            '</div>'
            '<div style="color:#475569;font-size:0.75rem;margin-top:14px;text-align:center">'
            'Lặp lại sau mỗi <b style="color:#94a3b8">30 phút</b> — Data bổ sung từ MT5 + CSV preload + yfinance'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    if training_report:
        st.subheader("Kết quả Training gần nhất")
        tr1, tr2, tr3, tr4 = st.columns(4)
        tr1.metric("ROC-AUC",   _round(training_report.get("roc_auc"), 4))
        tr2.metric("Precision", _pct(training_report.get("precision")))
        tr3.metric("Recall",    _pct(training_report.get("recall")))
        tr4.metric("F1",        _pct(training_report.get("f1")))
        tr5, tr6 = st.columns(2)
        tr5.metric("Train Rows", training_report.get("train_rows", "n/a"))
        tr6.metric("Test Rows",  training_report.get("test_rows",  "n/a"))

    st.divider()
    st.subheader("📊 Phân tích thắng/thua tín hiệu Walk-Forward")
    st.caption("Dữ liệu từ 19 folds walk-forward (2022–2026) — log thắng/thua theo từng tín hiệu")
    if wf_win_events or wf_loss_events:
        n_wins   = len(wf_win_events)
        n_losses = len(wf_loss_events)
        n_total  = n_wins + n_losses
        wl1, wl2, wl3, wl4 = st.columns(4)
        wl_c = GREEN if n_wins / max(n_total, 1) >= 0.50 else AMBER
        wl1.markdown(_card("Tổng tín hiệu", f"{n_total:,}", "Walk-Forward 19 folds", BLUE), unsafe_allow_html=True)
        wl2.markdown(_card("Đúng (Thắng)", f"{n_wins:,}", f"{n_wins/max(n_total,1):.1%}", GREEN), unsafe_allow_html=True)
        wl3.markdown(_card("Sai (Thua)", f"{n_losses:,}", f"{n_losses/max(n_total,1):.1%}", RED), unsafe_allow_html=True)
        wl4.markdown(_card("Tỷ lệ thắng WF", f"{n_wins/max(n_total,1):.1%}", "mục tiêu >= 50%", wl_c), unsafe_allow_html=True)

        if wf_win_events:
            win_df  = pd.DataFrame(wf_win_events)
            loss_df = pd.DataFrame(wf_loss_events) if wf_loss_events else pd.DataFrame()
            st.markdown("**Phan phoi tin hieu thang/thua theo fold:**")
            if "fold" in win_df.columns:
                fold_wins   = win_df.groupby("fold").size().rename("Wins")
                fold_losses = loss_df.groupby("fold").size().rename("Losses") if not loss_df.empty and "fold" in loss_df.columns else pd.Series(dtype=int, name="Losses")
                fold_chart  = pd.concat([fold_wins, fold_losses], axis=1).fillna(0)
                st.bar_chart(fold_chart, height=220)
            if "confidence" in win_df.columns:
                conf_cmp = pd.DataFrame({
                    "Confidence — Win":  win_df["confidence"].dropna(),
                    "Confidence — Loss": loss_df["confidence"].dropna() if not loss_df.empty and "confidence" in loss_df.columns else pd.Series(dtype=float),
                })
                w_l, w_r = st.columns(2)
                with w_l:
                    st.markdown("**Avg Confidence: Win vs Loss**")
                    st.dataframe(pd.DataFrame({
                        "Nhóm": ["Thắng (đúng)", "Thua (sai)"],
                        "Count": [n_wins, n_losses],
                        "Avg Confidence": [
                            f"{win_df['confidence'].mean():.3f}" if "confidence" in win_df.columns else "n/a",
                            f"{loss_df['confidence'].mean():.3f}" if not loss_df.empty and "confidence" in loss_df.columns else "n/a",
                        ],
                    }), use_container_width=True, hide_index=True)
                with w_r:
                    if "side" in win_df.columns:
                        side_wr = pd.DataFrame({
                            "Chiều": win_df["side"].dropna().value_counts().index.tolist(),
                            "Thắng": win_df["side"].dropna().value_counts().values.tolist(),
                        })
                        st.markdown("**Thắng theo chiều:**")
                        st.dataframe(side_wr, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có win/loss log từ walk-forward. Chạy `scripts/walkforward_ict_wyckoff.py` trước.")

    st.divider()
    st.subheader("Feature Importance — Điều model đang học")
    fi_df = load_feature_importance()
    if fi_df is not None and not fi_df.empty:
        # HistGBC uses feature_importances_, no negative coef → show top/bottom by abs
        top_n = fi_df.head(15)
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown(f"**Top 15 features quan trọng nhất:**")
            if not top_n.empty:
                st.bar_chart(top_n.set_index("feature")["abs_importance"], height=320)
        with fc2:
            has_coef = fi_df["importance"].lt(0).any()
            if has_coef:
                neg = fi_df[fi_df["importance"] < 0].sort_values("importance").head(12)
                st.markdown(f"**Top SELL signals (coef < 0) — {len(neg)} features:**")
                if not neg.empty:
                    st.bar_chart(neg.set_index("feature")["importance"], height=320)
            else:
                # HistGBC: show bottom 10 as least important
                bot = fi_df.tail(10)
                st.markdown("**10 features ít quan trọng nhất:**")
                if not bot.empty:
                    st.bar_chart(bot.set_index("feature")["abs_importance"], height=320)
        with st.expander("Bảng đầy đủ feature importance"):
            st.dataframe(fi_df[["feature", "importance", "abs_importance"]].round(6),
                         use_container_width=True)
    else:
        st.info("Chưa có model. Chạy: python -m xauusd_ai.main train --config configs/train_ict_wyckoff_2022_2026.yaml")

    st.divider()
    st.subheader("Cấu hình Self-Learning hiện tại")
    st.code("""
# configs/train_ict_wyckoff_2022_2026.yaml
training:
  live_learning_enabled: true          # Bat/tat tu hoc
  live_learning_min_new_bars: 12       # Toi thieu 12 nen M15 moi (3 gio)
  live_learning_min_rows: 1000         # Dataset toi thieu 1000 hang
  live_learning_interval_hours: 1      # Tan suat fetch yfinance

# Model: HistGradientBoostingClassifier (max_iter=500, lr=0.05, depth=6, balanced)
# 28 features: D1(1) + H4(9:ICT) + H1(3:Wyckoff) + M15(15:execution)
# Walk-Forward: 19 folds, AUC avg=0.6415 (std=0.0123), Precision avg=54.7%
# Signal threshold: 0.55 (saved in model_meta_ict_wyckoff.json)
""", language="yaml")


# =============================================================================
# TAB 5 — BACKTEST
# =============================================================================
def _render_backtest_panel(rpt: dict, trd: pd.DataFrame, training_rpt: dict | None = None, label: str = "") -> None:
    """Render backtest metrics + charts for one account."""
    _ret       = float(rpt.get("return_pct",      0) or 0)
    _bal_start = float(rpt.get("starting_balance", rpt.get("initial_balance", 200)) or 200)
    _bal_end   = float(rpt.get("ending_balance",   rpt.get("final_balance", 0)) or 0)
    _pf        = float(rpt.get("profit_factor",   0) or 0)
    _dd        = float(rpt.get("max_drawdown_pct",0) or 0)
    _sharpe    = float(rpt.get("sharpe_like",      rpt.get("sharpe_ratio", 0)) or 0)
    _gross_p   = float(rpt.get("gross_profit",    0) or 0)
    _gross_l   = float(rpt.get("gross_loss",      0) or 0)
    _avg_w     = float(rpt.get("avg_win",         0) or 0)
    _avg_l     = float(rpt.get("avg_loss",        0) or 0)
    _sig_filt  = int(rpt.get("signals_filtered_out", 0) or 0)
    _sig_nslot = int(rpt.get("signals_no_slot",   0) or 0)
    _n_trades  = int(rpt.get("trades",            0) or 0)

    # ── Core P&L ─────────────────────────────────────────────────────────────
    b1, b2, b3, b4 = st.columns(4)
    b1.markdown(_card("Lợi nhuận %",
                      f"{_ret:.1f}%",
                      f"${_bal_start:.0f} → ${_bal_end:,.0f} (+${_bal_end - _bal_start:,.0f})",
                      GREEN if _ret > 0 else RED), unsafe_allow_html=True)
    b2.markdown(_card("Profit Factor",
                      f"{_pf:.3f}",
                      "✅ PF ≥ 1.5 tốt | PF ≥ 2.0 rất tốt" if _pf >= 1.5 else "⚠️ PF < 1.5 cần cải thiện",
                      GREEN if _pf >= 1.5 else AMBER), unsafe_allow_html=True)
    b3.markdown(_card("Max Drawdown",
                      f"{_dd:.1f}%",
                      "✅ DD < 20% an toàn" if abs(_dd) < 20 else "⚠️ DD cao > 20%",
                      GREEN if abs(_dd) < 20 else AMBER), unsafe_allow_html=True)
    b4.markdown(_card("Sharpe Ratio",
                      f"{_sharpe:.2f}",
                      "✅ ≥ 1.0 tốt | ≥ 2.0 rất tốt" if _sharpe >= 1.0 else "⚠️ < 1.0",
                      GREEN if _sharpe >= 1.0 else AMBER), unsafe_allow_html=True)

    # ── Balance và thời gian ──────────────────────────────────────────────────
    b5, b6, b7, b8 = st.columns(4)
    b5.metric("Bắt đầu",  rpt.get("start",  rpt.get("test_start", "n/a")))
    b6.metric("Kết thúc", rpt.get("end",    rpt.get("trade_end",  "n/a")))
    b7.metric("Vốn đầu",  f"${_bal_start:.0f}")
    b8.metric("Vốn cuối", f"${_bal_end:,.2f}")

    # ── Chi tiết Lãi/Lỗ ──────────────────────────────────────────────────────
    if _gross_p or _gross_l:
        st.markdown("#### 💰 Chi tiết Lãi/Lỗ")
        g1, g2, g3, g4 = st.columns(4)
        g1.markdown(_card("Tổng lãi gộp",    f"${_gross_p:,.2f}",
                          "Tổng $ từ lệnh thắng", GREEN), unsafe_allow_html=True)
        g2.markdown(_card("Tổng lỗ gộp",     f"-${abs(_gross_l):,.2f}",
                          "Tổng $ từ lệnh thua", RED), unsafe_allow_html=True)
        g3.markdown(_card("Avg lệnh thắng",  f"${_avg_w:.2f}",
                          "Lãi TB/lệnh → chất lượng exit", GREEN), unsafe_allow_html=True)
        g4.markdown(_card("Avg lệnh thua",   f"-${abs(_avg_l):.2f}",
                          "Thua TB/lệnh → kiểm soát rủi ro", AMBER), unsafe_allow_html=True)

    # ── Signal filter stats ───────────────────────────────────────────────────
    if _sig_filt or _sig_nslot:
        _total_raw = _n_trades + _sig_filt + _sig_nslot
        g5, g6, g7, g8 = st.columns(4)
        g5.metric("Tổng tín hiệu gốc",  f"{_total_raw:,}")
        g6.metric("Lọc (threshold)",     f"{_sig_filt:,}",
                  help="Bị loại vì xác suất thấp hơn ngưỡng")
        g7.metric("Bỏ (no slot)",        f"{_sig_nslot:,}",
                  help="Bị bỏ vì đã đủ lệnh đang mở")
        g8.metric("Thực thi lệnh",       f"{_n_trades:,}",
                  help=f"Tỷ lệ thực thi: {_n_trades/_total_raw:.1%}" if _total_raw else "0%")

    if not trd.empty:
        ts2 = summarize_trades(trd)
        ds2, df2 = summarize_daily(trd)

        bt1, bt2, bt3, bt4 = st.columns(4)
        bt1.metric("Tổng lệnh",      ts2["trades"])
        bt2.metric("Thắng / Thua",   f"{ts2['wins']} / {ts2['losses']}")
        bt3.metric("Tỷ lệ thắng",    f"{ts2['win_rate']:.1%}")
        bt4.metric("Lợi nhuận ròng", f"${ts2['net_profit']:.2f}")

        st.divider()
        if "balance_after" in trd.columns:
            st.subheader("📈 Đường vốn Backtest")
            eq2 = trd.dropna(subset=["time", "balance_after"]).set_index("time").sort_index()
            st.line_chart(eq2["balance_after"].pipe(_ds), height=280)

        if not df2.empty:
            st.subheader("📅 Hiệu suất hàng ngày")
            dd1, dd2 = st.columns(2)
            with dd1:
                st.bar_chart(df2.set_index("date")["return_pct"], height=200)
            with dd2:
                st.line_chart(df2.set_index("date")["net_pnl"], height=200)
            kd1, kd2, kd3, kd4 = st.columns(4)
            kd1.metric("TB ngày %",       f"{ds2['mean_daily_return']:.3f}%")
            kd2.metric("Ngày tốt nhất %", f"{ds2['best_day_return']:.3f}%")
            kd3.metric("Ngày tệ nhất %",  f"{ds2['worst_day_return']:.3f}%")
            kd4.metric("Ngày >= 2%",      f"{ds2['share_ge_2'] * 100:.1f}%")

        if "pnl" in trd.columns:
            st.subheader("📊 Phân phối P&L")
            st.bar_chart(trd["pnl"].value_counts(bins=30).sort_index(), height=200)

        # Monthly breakdown from report
        _monthly = rpt.get("monthly", [])
        if _monthly:
            _mo_df = pd.DataFrame(_monthly)
            _mo_active = _mo_df[_mo_df["trades_n"] > 0] if "trades_n" in _mo_df.columns else _mo_df
            if not _mo_active.empty:
                st.subheader("📅 Monthly Breakdown")
                _mo1, _mo2 = st.columns(2)
                with _mo1:
                    st.markdown("**PnL theo tháng ($):**")
                    st.bar_chart(_mo_active.set_index("month")["pnl_sum"], height=200)
                with _mo2:
                    st.markdown("**Win Rate theo tháng:**")
                    _wr_mo = _mo_active.set_index("month")["win_rate"].copy()
                    _wr_mo_df = pd.DataFrame({"win_rate": _wr_mo, "target_60pct": 0.60})
                    st.line_chart(_wr_mo_df, height=200)
                _mo_disp = _mo_active[[c for c in ["month", "trades_n", "pnl_sum", "wins_n", "win_rate"] if c in _mo_active.columns]].copy()
                if "win_rate" in _mo_disp.columns:
                    _mo_disp["win_rate"] = (_mo_disp["win_rate"] * 100).round(1).astype(str) + "%"
                if "pnl_sum" in _mo_disp.columns:
                    _mo_disp["pnl_sum"] = _mo_disp["pnl_sum"].round(2)
                st.dataframe(_mo_disp, use_container_width=True, hide_index=True)

        with st.expander(f"📋 Backtest Report JSON {label}"):
            st.json(rpt)
        if training_rpt:
            with st.expander(f"📋 Training Report JSON {label}"):
                st.json(training_rpt)

        with st.expander(f"📖 Giải thích các chỉ số Backtest", expanded=False):
            st.markdown("""
| Chỉ số | Ý nghĩa | Ngưỡng tốt |
|--------|---------|------------|
| **Lợi nhuận %** | % tăng trưởng vốn sau toàn bộ backtest | > 100% |
| **Profit Factor (PF)** | Tổng lãi / tổng lỗ. PF=2 nghĩa là mỗi $1 thua bù được $2 lãi | ≥ 1.5 |
| **Max Drawdown** | Mức giảm vốn tối đa từ đỉnh → đáy | < 20% |
| **Sharpe Ratio** | Lợi nhuận điều chỉnh theo rủi ro. Sharpe=2 rất tốt | ≥ 1.0 |
| **Avg lệnh thắng** | Lãi trung bình mỗi lệnh thắng | Nên cao hơn Avg thua |
| **Avg lệnh thua** | Thua trung bình mỗi lệnh thua | Nên thấp hơn Avg thắng |
| **Tín hiệu gốc** | Tổng tín hiệu AI tạo ra trước khi lọc | — |
| **Lọc (threshold)** | Bị loại vì xác suất < ngưỡng (AI không đủ tự tin) | — |
| **Bỏ (no slot)** | Bị bỏ vì đang có quá nhiều lệnh mở cùng lúc | — |
            """)

        st.subheader(f"Danh sách tất cả lệnh ({len(trd):,})")
        dcols2 = [c for c in ["time", "side", "entry_price", "exit_price",
                               "pnl", "is_win", "balance_after", "drawdown",
                               "realized_rr", "probability", "volatility_regime"]
                  if c in trd.columns]
        st.dataframe(trd[dcols2], use_container_width=True)
    else:
        st.info(f"Chưa có dữ liệu backtest ({label}). Chạy: python scripts/backtest_ict_wyckoff.py")


with tab_backtest:
    st.markdown(
        '<div style="border-bottom:2px solid #1e293b;padding-bottom:12px;margin-bottom:18px">'
        '<span style="font-size:1.35rem;font-weight:800;color:#f1f5f9">Kết quả Backtest</span></div>',
        unsafe_allow_html=True,
    )
    # Build dynamic tab labels from actual report data
    _bt1_wr  = backtest_report.get("win_rate", 0) if backtest_report else 0
    _bt1_ret = backtest_report.get("return_pct", 0) if backtest_report else 0
    _bt2_wr  = backtest_report_acc2.get("win_rate", 0) if backtest_report_acc2 else 0
    _bt2_ret = backtest_report_acc2.get("return_pct", 0) if backtest_report_acc2 else 0
    _bt_acc1_tab, _bt_acc2_tab = st.tabs([
        f"🏦 ACC1 — ICT+Wyckoff | WR {_bt1_wr:.1%} | +{_bt1_ret:.0f}%",
        f"🏦 ACC2 — v5 P0-P3 | WR {_bt2_wr:.1%} | +{_bt2_ret:.0f}%",
    ])
    with _bt_acc1_tab:
        _bt1_trades = backtest_report.get("trades", 0) if backtest_report else 0
        _bt1_end    = backtest_report.get("ending_balance", 0) if backtest_report else 0
        _bt1_pf     = backtest_report.get("profit_factor", 0) if backtest_report else 0
        _bt1_dd     = backtest_report.get("max_drawdown_pct", 0) if backtest_report else 0
        st.subheader(
            f"ACC1 — ICT+Wyckoff: {_bt1_trades:,} lệnh | WR {_bt1_wr:.1%} | "
            f"$200→${_bt1_end:,.0f} | +{_bt1_ret:.0f}% | PF {_bt1_pf:.2f} | MaxDD {_bt1_dd:.1f}%"
        )
        _render_backtest_panel(backtest_report, trades, training_report, "ACC1")
    with _bt_acc2_tab:
        _bt2_trades = backtest_report_acc2.get("trades", 0) if backtest_report_acc2 else 0
        _bt2_end    = backtest_report_acc2.get("ending_balance", 0) if backtest_report_acc2 else 0
        _bt2_pf     = backtest_report_acc2.get("profit_factor", 0) if backtest_report_acc2 else 0
        _bt2_dd     = backtest_report_acc2.get("max_drawdown_pct", 0) if backtest_report_acc2 else 0
        _bt2_net    = backtest_report_acc2.get("net_profit", 0) if backtest_report_acc2 else 0
        st.subheader(
            f"ACC2 — v5 ICT+Wyckoff+P0-P3: {_bt2_trades:,} lệnh | WR {_bt2_wr:.1%} | "
            f"$200→${_bt2_end:,.0f} | +{_bt2_ret:.0f}% | PF {_bt2_pf:.2f} | MaxDD {_bt2_dd:.1f}%"
        )
        # Weekly breakdown table
        _wk2 = backtest_report_acc2.get("weekly", [])
        _active_wk2 = [w for w in _wk2 if w.get("trades_n", 0) > 0]
        if _active_wk2:
            st.subheader("📅 Weekly Breakdown")
            _wk_df = pd.DataFrame(_active_wk2)[["week", "trades_n", "pnl_sum", "wins_n", "win_rate"]]
            _wk_df.columns = ["Tuần", "Lệnh", "PnL ($)", "Thắng", "Win Rate"]
            _wk_df["PnL ($)"] = _wk_df["PnL ($)"].round(2)
            _wk_df["Win Rate"] = (_wk_df["Win Rate"] * 100).round(1).astype(str) + "%"
            st.dataframe(_wk_df, use_container_width=True)
            _wk_pnl = pd.DataFrame(_active_wk2).set_index("week")["pnl_sum"]
            st.subheader("📈 PnL theo tuần")
            st.bar_chart(_wk_pnl, height=250)
            st.divider()
        _render_backtest_panel(backtest_report_acc2, trades_acc2, None, "ACC2")


# =============================================================================
# TAB 6 — WALK-FORWARD
# =============================================================================
def _render_walkforward_panel(wf_report_path, signals_csv_path, label: str = "") -> None:
    """Render walk-forward analysis panel for one account."""
    _wf_hdr_r = load_json(wf_report_path)
    _wf_n = _wf_hdr_r.get("walk_forward", {}).get("n_folds", "?") if _wf_hdr_r else "?"
    st.markdown(
        f'<div style="border-bottom:2px solid #1e293b;padding-bottom:12px;margin-bottom:18px">'
        f'<span style="font-size:1.35rem;font-weight:800;color:#f1f5f9">Walk-Forward Analysis</span>'
        f'<span style="color:#64748b;font-size:0.82rem;margin-left:12px">'
        f'{label} — {_wf_n} Folds</span></div>',
        unsafe_allow_html=True,
    )

    # ── Live progress tracking ──────────────────────────────────────────
    _pdata = load_json(OUTPUTS / "walkforward_progress.json")
    if _pdata:
        _pstatus  = _pdata.get("status", "")
        _pdone    = _pdata.get("completed_combinations", 0)
        _ptotal   = _pdata.get("total_combinations", 0)
        _ppct     = _pdata.get("pct_done", 0.0)
        _pelapsed = _pdata.get("elapsed_seconds", 0)
        _pmins, _psecs = int(_pelapsed // 60), int(_pelapsed % 60)
        _plast    = _pdata.get("last_updated", "")
        if _pstatus == "running":
            if _HAS_AUTOREFRESH:
                _st_autorefresh(interval=15_000, key="wf_progress_autorefresh")
            st.warning(
                f"⏳ Walk-Forward đang chạy: **{_pdone}/{_ptotal}** combinations "
                f"({_ppct:.1f}%) — {_pmins}m {_psecs}s elapsed"
            )
            st.progress(min(_ppct / 100.0, 1.0))
            _pb = _pdata.get("best_so_far")
            if _pb:
                _bpc1, _bpc2, _bpc3, _bpc4 = st.columns(4)
                _bpc1.metric("Best Return",    f"{_pb.get('avg_return_pct', 0):.2f}%")
                _bpc2.metric("Best PF",        f"{_pb.get('avg_profit_factor', 0):.3f}")
                _bpc3.metric("Best Drawdown",  f"{_pb.get('avg_max_drawdown_pct', 0):.2f}%")
                _bpc4.metric("Best Precision", f"{_pb.get('avg_precision', 0):.2%}")
                with st.expander("📋 Best params so far"):
                    st.json(_pb.get("params", {}))
            st.caption(f"Cập nhật lần cuối: {_plast}")
            st.divider()
        elif _pstatus == "done":
            st.success(f"✅ Walk-Forward hoàn thành! {_ptotal} combinations | {_pmins}m {_psecs}s")
            st.divider()

    wf_r = load_json(wf_report_path)

    if wf_r:
        agg = wf_r.get("aggregate", {})
        wf_info = wf_r.get("walk_forward", {})

        # ── Top-level aggregate metrics ──────────────────────────────────────
        st.subheader("Tổng kết Walk-Forward")
        wf1, wf2, wf3, wf4 = st.columns(4)
        # avg_roc_auc may be missing in older reports — compute from folds if needed
        _agg_folds_roc = [f.get("roc_auc", 0) for f in wf_r.get("folds", [])]
        auc_avg = agg.get("avg_roc_auc") or (
            sum(_agg_folds_roc) / max(len(_agg_folds_roc), 1) if _agg_folds_roc else 0.0
        )
        auc_std = agg.get("std_roc_auc", 0)
        auc_c = GREEN if auc_std < 0.04 else AMBER
        wf1.markdown(_card("Avg ROC-AUC", f"{auc_avg:.4f}",
                           f"std={auc_std:.4f} {'✅ ổn định' if auc_std < 0.04 else '⚠️ cao'}",
                           auc_c), unsafe_allow_html=True)

        prec_avg = agg.get("avg_precision", 0)
        prec_c = GREEN if prec_avg >= 0.52 else AMBER
        wf2.markdown(_card("Avg Precision", f"{prec_avg:.1%}",
                           f"range {agg.get('min_precision', 0):.2f}–{agg.get('max_precision', 0):.2f}",
                           prec_c), unsafe_allow_html=True)

        sig_wr = agg.get("signal_win_rate", 0)
        sig_c = GREEN if sig_wr >= 0.50 else AMBER
        wf3.markdown(_card("Tỷ lệ thắng tín hiệu", f"{sig_wr:.1%}",
                           f"{agg.get('correct_signals', 0):,} / {agg.get('total_signals', 0):,}",
                           sig_c), unsafe_allow_html=True)

        wf4.markdown(_card("Folds", f"{wf_info.get('n_folds', 'n/a')}",
                           f"model: HistGBC 28 feat", BLUE), unsafe_allow_html=True)

        # ── Hàng 2: Model stats (non-duplicate) ─────────────────────────────
        wf5c, wf6c, wf7c, wf8c = st.columns(4)
        n_prec50 = sum(1 for f in wf_r.get("folds", []) if f.get("precision", 0) >= 0.50)
        wf5c.metric("Avg Recall",       _pct(agg.get("avg_recall")),
                    help="Recall: % số tín hiệu đúng được model bắt được (ra signal)")
        wf6c.metric("Avg F1",           _pct(agg.get("avg_f1")),
                    help="F1: cân bằng giữa Precision & Recall. F1=1 hoàn hảo")
        wf7c.metric("Avg Accuracy",     _pct(agg.get("avg_accuracy")),
                    help="Accuracy: % dự đoán đúng tổng thể (cả thắng lẫn thua)")
        wf8c.metric("Folds prec ≥ 50%", f"{n_prec50} / {wf_info.get('n_folds', 0)}",
                    help="Số fold đạt Precision ≥ 50% — nhiều fold tốt = model nhất quán")

        # ── Hàng 3: Signal & AUC stats ──────────────────────────────────────
        wf9c, wf10c, wf11c, wf12c = st.columns(4)
        wf9c.metric("AUC Min",           f"{agg.get('min_roc_auc', 0):.4f}",
                    help="AUC thấp nhất qua các fold — đo sự ổn định tệ nhất")
        wf10c.metric("AUC Max",          f"{agg.get('max_roc_auc', 0):.4f}",
                     help="AUC cao nhất — đo tiềm năng tốt nhất của model")
        wf11c.metric("Avg Signal Rate",  _pct(agg.get("avg_signal_rate")),
                     help="% bar có tín hiệu — thấp = thận trọng, cao = tích cực hơn")
        _cs_top = agg.get("concurrent_sim", {})
        wf12c.metric("Sim Return TB",    f"{_cs_top.get('avg_return_pct', 0):+.0f}%",
                     help="Return trung bình mỗi fold trong mô phỏng giao dịch thực tế")

        # ── Hàng 4: P&L từ Concurrent Simulation ────────────────────────────
        if _cs_top:
            st.markdown("#### 💰 Kết quả P&L — Walk-Forward Simulation")
            _cs_wr  = float(_cs_top.get("avg_win_rate", 0) or 0)
            _cs_pf  = float(_cs_top.get("avg_profit_factor", 0) or 0)
            _cs_ret = float(_cs_top.get("avg_return_pct", 0) or 0)
            _cs_dd  = float(_cs_top.get("avg_max_drawdown_pct", 0) or 0)
            _cs_sig = int(agg.get("total_signals", 0) or 0)
            _cs_cor = int(agg.get("correct_signals", 0) or 0)
            p1, p2, p3, p4 = st.columns(4)
            p1.markdown(_card("Win Rate (Sim)",
                              f"{_cs_wr:.1%}",
                              "✅ ≥ 60% xuất sắc" if _cs_wr >= 0.60 else "⚠️ < 60%",
                              GREEN if _cs_wr >= 0.60 else AMBER), unsafe_allow_html=True)
            p2.markdown(_card("Profit Factor",
                              f"{_cs_pf:.3f}",
                              "✅ PF ≥ 1.5 có lời ổn định" if _cs_pf >= 1.5 else "⚠️ PF < 1.5",
                              GREEN if _cs_pf >= 1.5 else AMBER), unsafe_allow_html=True)
            p3.markdown(_card("Return TB/Fold",
                              f"{_cs_ret:+.0f}%",
                              "Lợi nhuận trung bình mỗi fold ($200 vốn)",
                              GREEN if _cs_ret >= 0 else RED), unsafe_allow_html=True)
            p4.markdown(_card("Max DD TB",
                              f"{_cs_dd:.1f}%",
                              "✅ < 20% an toàn" if abs(_cs_dd) < 20 else "⚠️ DD cao > 20%",
                              GREEN if abs(_cs_dd) < 20 else AMBER), unsafe_allow_html=True)

            p5, p6 = st.columns(2)
            p5.metric("Tổng tín hiệu WF",   f"{_cs_sig:,}",
                      help="Tổng số tín hiệu qua tất cả các fold")
            p6.metric("Tín hiệu đúng",       f"{_cs_cor:,} ({agg.get('signal_win_rate', 0):.1%})",
                      help="Số tín hiệu model dự đoán đúng chiều thị trường")

        # ── Per-fold charts ──────────────────────────────────────────────────
        folds = wf_r.get("folds", [])
        if folds:
            st.divider()
            st.subheader("Kết quả từng fold")
            fd = pd.DataFrame(folds)

            # AUC + Precision line chart
            chart_l, chart_r = st.columns(2)
            with chart_l:
                st.markdown("**ROC-AUC theo fold:**")
                auc_df = fd.set_index("fold")[["roc_auc"]].copy()
                auc_df["threshold_0.62"] = 0.62
                st.line_chart(auc_df, height=220)
            with chart_r:
                st.markdown("**Precision theo fold:**")
                prec_df = fd.set_index("fold")[["precision"]].copy()
                prec_df["target_0.50"] = 0.50
                prec_df["target_0.52"] = 0.52
                st.line_chart(prec_df, height=220)

            # Signals per fold
            chart_l2, chart_r2 = st.columns(2)
            with chart_l2:
                st.markdown("**Số tín hiệu / fold:**")
                if "n_signals" in fd.columns:
                    st.bar_chart(fd.set_index("fold")["n_signals"], height=200)
            with chart_r2:
                st.markdown("**F1 Score theo fold:**")
                if "f1" in fd.columns:
                    st.bar_chart(fd.set_index("fold")["f1"], height=200)

            # Fold table
            st.divider()
            _n_folds_display = wf_info.get("n_folds", len(folds))
            st.subheader(f"Bảng chi tiết {_n_folds_display} folds")
            display_cols = [c for c in [
                "fold", "test_start", "test_end", "threshold",
                "roc_auc", "precision", "recall", "f1", "accuracy",
                "n_signals", "signal_rate", "elapsed_s"
            ] if c in fd.columns]

            def _highlight_fold(row):
                prec_ok = row.get("precision", 0) >= 0.50
                auc_ok  = row.get("roc_auc", 0) >= 0.62
                if prec_ok and auc_ok:
                    return ["background-color: #10b98115"] * len(row)
                elif prec_ok:
                    return ["background-color: #10b98108"] * len(row)
                else:
                    return ["background-color: #ef444412"] * len(row)

            styled = fd[display_cols].style.apply(_highlight_fold, axis=1).format({
                "roc_auc":     "{:.4f}",
                "precision":   "{:.4f}",
                "recall":      "{:.4f}",
                "f1":          "{:.4f}",
                "accuracy":    "{:.4f}",
                "signal_rate": "{:.3f}",
                "threshold":   "{:.2f}",
                "elapsed_s":   "{:.1f}",
            }, na_rep="n/a")
            st.dataframe(styled, use_container_width=True)
            st.caption("Xanh đậm: precision ≥ 50% & AUC ≥ 0.62 | Xanh nhạt: precision ≥ 50% | Đỏ: precision < 50%")

        # ── Dynamic Concurrent Position Simulation ───────────────────────────
        _agg_csim = agg.get("concurrent_sim", {})
        if _agg_csim or any("concurrent_sim" in f for f in folds):
            st.divider()
            st.subheader("📈 Mô phỏng Lệnh Song Song — Dynamic Position Scaling")
            st.caption("Mỗi fold bắt đầu với $200. Khi balance tăng → mở thêm lệnh cùng lúc (2→3→5→8→15 slots)")

            # Aggregate concurrent sim metrics
            if _agg_csim:
                cs1, cs2, cs3, cs4 = st.columns(4)
                _cs_wr = _agg_csim.get("avg_win_rate", 0)
                _cs_pf = _agg_csim.get("avg_profit_factor", 0)
                _cs_ret = _agg_csim.get("avg_return_pct", 0)
                _cs_dd = _agg_csim.get("avg_max_drawdown_pct", 0)
                cs1.markdown(_card("Win Rate TB", f"{_cs_wr:.1%}",
                                   "trung bình các fold",
                                   GREEN if _cs_wr >= 0.70 else AMBER), unsafe_allow_html=True)
                cs2.markdown(_card("Profit Factor TB", f"{_cs_pf:.3f}",
                                   "lợi nhuận / thua lỗ",
                                   GREEN if _cs_pf >= 1.5 else AMBER), unsafe_allow_html=True)
                cs3.markdown(_card("Return TB/Fold", f"{_cs_ret:+.2f}%",
                                   f"từ $200 mỗi fold",
                                   GREEN if _cs_ret >= 0 else "#8b0000"), unsafe_allow_html=True)
                cs4.markdown(_card("Max DD TB", f"{_cs_dd:.2f}%",
                                   "drawdown tối đa trung bình",
                                   GREEN if abs(_cs_dd) < 15 else AMBER), unsafe_allow_html=True)

                cs5, cs6 = st.columns(2)
                _cs_avgpos = _agg_csim.get("avg_concurrent_positions", 0)
                _cs_maxpos = _agg_csim.get("max_concurrent_positions", 0)
                cs5.metric("Avg Concurrent Positions", f"{_cs_avgpos:.2f}")
                cs6.metric("Max Concurrent Positions", str(_cs_maxpos))

            # Per-fold concurrent sim chart
            _fold_csim_rows = [
                {
                    "fold": f.get("fold"),
                    "return_pct": f.get("concurrent_sim", {}).get("return_pct", 0),
                    "win_rate": f.get("concurrent_sim", {}).get("win_rate", 0),
                    "profit_factor": f.get("concurrent_sim", {}).get("profit_factor", 0),
                    "max_drawdown_pct": f.get("concurrent_sim", {}).get("max_drawdown_pct", 0),
                    "trades": f.get("concurrent_sim", {}).get("trades", 0),
                    "avg_concurrent": f.get("concurrent_sim", {}).get("avg_concurrent_positions", 0),
                    "ending_balance": f.get("concurrent_sim", {}).get("ending_balance", 200),
                }
                for f in folds if "concurrent_sim" in f
            ]
            if _fold_csim_rows:
                _csdf = pd.DataFrame(_fold_csim_rows).set_index("fold")

                _cc1, _cc2 = st.columns(2)
                with _cc1:
                    st.markdown("**Return % mỗi fold (từ $200):**")
                    st.bar_chart(_csdf[["return_pct"]], height=200)
                with _cc2:
                    st.markdown("**Win Rate mỗi fold:**")
                    _wr_chart = _csdf[["win_rate"]].copy()
                    _wr_chart["target_80pct"] = 0.80
                    st.line_chart(_wr_chart, height=200)

                _cc3, _cc4 = st.columns(2)
                with _cc3:
                    st.markdown("**Balance cuối mỗi fold (từ $200):**")
                    st.line_chart(_csdf[["ending_balance"]], height=200)
                with _cc4:
                    st.markdown("**Avg Concurrent Positions / fold:**")
                    st.bar_chart(_csdf[["avg_concurrent"]], height=200)

                st.divider()
                st.markdown("**Bảng chi tiết Concurrent Simulation:**")
                _csdf_display = _csdf.reset_index()
                st.dataframe(
                    _csdf_display.style.format({
                        "return_pct": "{:+.2f}%",
                        "win_rate": "{:.1%}",
                        "profit_factor": "{:.3f}",
                        "max_drawdown_pct": "{:.2f}%",
                        "avg_concurrent": "{:.2f}",
                        "ending_balance": "${:.2f}",
                    }, na_rep="n/a"),
                    use_container_width=True,
                )
                st.caption(
                    "Balance-tier scaling: < $500 → 2 lệnh | $500-$2k → 3 lệnh | "
                    "$2k-$10k → 5 lệnh | $10k-$50k → 8 lệnh | > $50k → 15 lệnh"
                )

        # ── Model info ───────────────────────────────────────────────────────
        st.divider()
        st.subheader("Thông tin Model Walk-Forward")
        mi_l, mi_r = st.columns(2)
        with mi_l:
            st.json({
                "model":       wf_info.get("model", "HistGBC"),
                "train_bars":  wf_info.get("train_bars", 20000),
                "test_bars":   wf_info.get("test_bars", 4000),
                "step_bars":   wf_info.get("step_bars", 4000),
                "n_folds":     wf_info.get("n_folds", 19),
                "n_features":  len(wf_info.get("features", [])),
            })
        with mi_r:
            feats = wf_info.get("features", [])
            if feats:
                st.markdown(f"**{len(feats)} Features:**")
                grps = {
                    "D1 (1)":  [f for f in feats if f.startswith("daily_")],
                    "H4 (9)":  [f for f in feats if f.startswith("h4_")],
                    "H1 (3)":  [f for f in feats if f in ("hourly_bias", "vsa_signal", "wyckoff_spring_signal")],
                    "M15 (15)": [f for f in feats if f not in [x for g in [
                        [f for f in feats if f.startswith("daily_")],
                        [f for f in feats if f.startswith("h4_")],
                        [f for f in feats if f in ("hourly_bias", "vsa_signal", "wyckoff_spring_signal")],
                    ] for x in g]],
                }
                for grp, names in grps.items():
                    if names:
                        st.markdown(f"**{grp}:** `{'`, `'.join(names)}`")

        # ── Walk-Forward signals CSV ─────────────────────────────────────────
        try:
            _wf_sigs = (pd.read_csv(signals_csv_path, on_bad_lines="skip")
                        if signals_csv_path and signals_csv_path.exists() else pd.DataFrame())
            if "time" in _wf_sigs.columns:
                _wf_sigs["time"] = pd.to_datetime(_wf_sigs["time"], utc=True, errors="coerce")
        except Exception:
            _wf_sigs = pd.DataFrame()
        if not _wf_sigs.empty:
            st.divider()
            st.subheader(f"Tín hiệu Walk-Forward ({len(_wf_sigs):,} dòng)")
            st.caption(
                "⚡ Đây là các tín hiệu model **chấp nhận** (predicted=1, proba ≥ threshold) — "
                "trước khi lọc giới hạn lệnh đồng thời. "
                "Xem mục *P&L — Walk-Forward Simulation* bên trên để xem kết quả có tính position slot."
            )
            # Normalize column names: proba→confidence, target→actual
            if "proba" in _wf_sigs.columns and "confidence" not in _wf_sigs.columns:
                _wf_sigs = _wf_sigs.rename(columns={"proba": "confidence"})
            if "target" in _wf_sigs.columns and "actual" not in _wf_sigs.columns:
                _wf_sigs = _wf_sigs.rename(columns={"target": "actual"})
            disp_wf = [c for c in ["time", "fold", "side", "confidence", "predicted",
                                    "actual", "correct", "strategy_score"] if c in _wf_sigs.columns]
            st.dataframe(_wf_sigs[disp_wf] if disp_wf else _wf_sigs,
                         use_container_width=True)

        with st.expander("📖 Giải thích các chỉ số Walk-Forward", expanded=False):
            st.markdown("""
**Walk-Forward là gì?** Model được train trên dữ liệu quá khứ, rồi test trên dữ liệu tương lai chưa thấy, lặp nhiều lần (folds). Đây là cách đánh giá thực tế nhất cho trading AI.

| Chỉ số | Ý nghĩa | Ngưỡng tốt |
|--------|---------|------------|
| **ROC-AUC** | Khả năng phân biệt tín hiệu thắng vs thua. 0.5=random, 1.0=hoàn hảo | ≥ 0.58 |
| **Precision** | Trong số lệnh AI vào, % thực sự thắng. Cao = ít vào lệnh xấu | ≥ 50% |
| **Recall** | % tín hiệu thắng thực sự được AI bắt (độ nhạy). Thấp = thận trọng | Tùy chiến lược |
| **F1 Score** | Cân bằng Precision & Recall. F1=1 hoàn hảo | ≥ 0.20 với threshold cao |
| **Signal Win Rate** | % tín hiệu AI dự đoán đúng chiều (không tính lãi/lỗ $) | ≥ 55% |
| **Std AUC** | Độ lệch chuẩn AUC qua các fold — thấp = model ổn định | < 0.04 |
| **Win Rate (Sim)** | Tỷ lệ thắng trong mô phỏng giao dịch thực (có SL/TP) | ≥ 60% |
| **Profit Factor** | Tổng lãi / tổng lỗ trong simulation | ≥ 1.5 |
| **Return TB/Fold** | Lợi nhuận trung bình mỗi period test (từ $200 vốn) | > 0% |

**Tại sao cần nhiều fold?** Mỗi fold = 1 khoảng thời gian khác nhau. Nếu model tốt ở nhiều fold khác nhau → trustworthy. Nếu chỉ tốt ở 1-2 fold → overfitting.
            """)

    else:
        st.info(
            "Chưa có dữ liệu Walk-Forward. Chạy:\n"
            "```\npython scripts/walkforward_ict_wyckoff.py\n```"
        )


with tab_walkforward:
    _wf_acc1_tab, _wf_acc2_tab = st.tabs([
        "🏦 ACC1 — ICT+Wyckoff",
        "🏦 ACC2 — v5 (ICT+Wyckoff+P0-P3)",
    ])
    with _wf_acc1_tab:
        _render_walkforward_panel(
            OUTPUTS / "walkforward_report_ict_wyckoff.json",
            OUTPUTS / "walkforward_trades_ict_wyckoff.csv",
            "ACC1 — ICT+Wyckoff",
        )
    with _wf_acc2_tab:
        # ── Load v5 live text log (updated every fold, even before JSON done) ──
        _wf5 = load_wf_txt_log(OUTPUTS / "wf_v5_acc2.txt")

        if _wf5:
            _done   = _wf5.get("completed", 0)
            _total  = _wf5.get("total_folds", 17)
            _is_fin = _wf5.get("is_complete", False)
            _wf_pct = _done / max(_total, 1)

            # ── Status banner ─────────────────────────────────────────────
            if _is_fin:
                st.success(f"✅ Walk-Forward v5 hoàn thành — {_done}/{_total} folds | "
                           f"Avg AUC={_wf5['avg_auc']:.4f} | Avg Prec={_wf5['avg_prec']:.1%}")
            else:
                if _HAS_AUTOREFRESH:
                    _st_autorefresh(interval=20_000, key="wf5_autorefresh")
                st.warning(f"⏳ Walk-Forward v5 đang chạy: **{_done}/{_total} folds** ({_wf_pct:.0%})")
                st.progress(_wf_pct)

            # ── Aggregate metrics ─────────────────────────────────────────
            st.markdown(
                '<div style="border-bottom:2px solid #1e293b;padding-bottom:10px;margin-bottom:14px">'
                '<span style="font-size:1.25rem;font-weight:800;color:#f1f5f9">Walk-Forward v5 — ACC2</span>'
                '<span style="background:#6366f122;border:1px solid #6366f144;border-radius:6px;'
                'padding:2px 10px;font-size:0.72rem;font-weight:700;color:#a5b4fc;margin-left:12px">'
                'ICT+Wyckoff+P0-P3 | VotingClassifier HGB+RF+ET</span></div>',
                unsafe_allow_html=True,
            )
            _m1, _m2, _m3, _m4, _m5 = st.columns(5)
            _auc_c  = GREEN if _wf5["avg_auc"]  >= 0.60 else AMBER
            _prec_c = GREEN if _wf5["avg_prec"] >= 0.55 else AMBER
            _m1.markdown(_card("Folds Done", f"{_done}/{_total}",
                               "hoàn thành" if _is_fin else "đang chạy", BLUE), unsafe_allow_html=True)
            _m2.markdown(_card("Avg AUC", f"{_wf5['avg_auc']:.4f}",
                               "≥0.60 là tốt", _auc_c), unsafe_allow_html=True)
            _m3.markdown(_card("Avg Precision", f"{_wf5['avg_prec']:.1%}",
                               "mục tiêu ≥55%", _prec_c), unsafe_allow_html=True)
            _m4.markdown(_card("Avg Recall", f"{_wf5['avg_recall']:.1%}",
                               "", BLUE), unsafe_allow_html=True)
            _m5.markdown(_card("Avg F1", f"{_wf5['avg_f1']:.1%}",
                               "", PURPLE), unsafe_allow_html=True)

            # ── Per-fold table ─────────────────────────────────────────────
            _folds_df = pd.DataFrame(_wf5["folds"])
            if not _folds_df.empty:
                st.divider()
                st.subheader(f"Kết quả {_done} folds (v5 — ACC2)")

                # Charts
                _ca, _cb = st.columns(2)
                with _ca:
                    st.markdown("**AUC theo fold:**")
                    _auc_chart = _folds_df.set_index("fold")[["roc_auc"]].copy()
                    _auc_chart["target_0.60"] = 0.60
                    st.line_chart(_auc_chart, height=200)
                with _cb:
                    st.markdown("**Precision theo fold:**")
                    _prec_chart = _folds_df.set_index("fold")[["precision"]].copy()
                    _prec_chart["target_0.55"] = 0.55
                    st.line_chart(_prec_chart, height=200)

                _cc, _cd = st.columns(2)
                with _cc:
                    st.markdown("**Số tín hiệu / fold:**")
                    st.bar_chart(_folds_df.set_index("fold")["n_signals"], height=180)
                with _cd:
                    st.markdown("**Threshold được chọn / fold:**")
                    st.bar_chart(_folds_df.set_index("fold")["threshold"], height=180)

                # Styled table
                st.divider()

                def _hl_fold_v5(row):
                    prec_ok = row.get("precision", 0) >= 0.55
                    auc_ok  = row.get("roc_auc", 0) >= 0.58
                    if prec_ok and auc_ok:
                        return ["background-color: #10b98118"] * len(row)
                    if prec_ok:
                        return ["background-color: #10b98108"] * len(row)
                    return ["background-color: #ef444412"] * len(row)

                _disp = _folds_df[[c for c in [
                    "fold", "test_start", "test_end",
                    "roc_auc", "precision", "recall", "f1",
                    "threshold", "n_signals", "elapsed_s",
                ] if c in _folds_df.columns]].copy()
                _disp_styled = _disp.style.apply(_hl_fold_v5, axis=1).format({
                    "roc_auc":   "{:.4f}",
                    "precision": "{:.4f}",
                    "recall":    "{:.4f}",
                    "f1":        "{:.4f}",
                    "threshold": "{:.2f}",
                    "elapsed_s": "{:.1f}s",
                }, na_rep="—")
                st.dataframe(_disp_styled, use_container_width=True, hide_index=True)
                st.caption(
                    "🟢 AUC≥0.58 & Prec≥55% | 🟩 Prec≥55% saja | 🟥 Prec<55%  "
                    f"| Config: {_wf5.get('config', 'live_acc2.yaml')} | {_wf5.get('n_features', 0)} features"
                )

                # Summary stats
                _prec_ok_n = sum(1 for f in _wf5["folds"] if f["precision"] >= 0.55)
                _auc_ok_n  = sum(1 for f in _wf5["folds"] if f["roc_auc"]   >= 0.58)
                _s1, _s2, _s3, _s4 = st.columns(4)
                _s1.metric("Folds prec ≥55%", f"{_prec_ok_n}/{_done}")
                _s2.metric("Folds AUC ≥0.58", f"{_auc_ok_n}/{_done}")
                _s3.metric("Best Precision",  f"{max(f['precision'] for f in _wf5['folds']):.1%}")
                _s4.metric("Best AUC",        f"{max(f['roc_auc']   for f in _wf5['folds']):.4f}")

        # ── Also show full JSON report if already written ──────────────────
        if (OUTPUTS / "walkforward_report_acc2.json").exists():
            st.divider()
            st.markdown("**Kết quả chi tiết từ JSON report (ACC2 — v5):**")
            _render_walkforward_panel(
                OUTPUTS / "walkforward_report_acc2.json",
                OUTPUTS / "walkforward_trades_acc2.csv",
                "ACC2 — v5",
            )
        elif not _wf5:
            st.info(
                "Chưa có dữ liệu Walk-Forward v5. Chạy:\n"
                "```\npython scripts/walkforward_ict_wyckoff.py --config configs/live_acc2.yaml\n```"
            )


# =============================================================================
# TAB 7 — RISK & CAI DAT
# =============================================================================
with tab_risk:
    st.markdown(
        '<div style="border-bottom:2px solid #1e293b;padding-bottom:12px;margin-bottom:18px">'
        '<span style="font-size:1.35rem;font-weight:800;color:#f1f5f9">Quản lý Rủi ro & Cài đặt</span></div>',
        unsafe_allow_html=True,
    )

    st.subheader("Quản lý Vị thế Động")
    st.markdown("""
| Số dư            | Lệnh tối đa | Rủi ro/lệnh | Ghi chú          |
|--------------------|----------|-----------|------------------|
| < $200             | **1**    | 0.65%     | Tài khoản nhỏ    |
| $200 – $500        | **2**    | 0.65%     | Tài khoản nhỏ    |
| $500 – $2,000      | **3**    | 0.65%     | Vừa-thấp        |
| $2,000 – $10,000   | **5**    | 0.65%     | Trung bình       |
| $10,000 – $50,000  | **8**    | 0.65%     | Lớn               |
| > $50,000          | **15**   | 0.65%     | Rất lớn          |

**Điều chỉnh theo Chế độ:**
- Sideways (regime=0): số lệnh / 2
- Strong Volatile (regime=2): số lệnh x 0.7
- Normal (regime=1): giữ nguyên
""")

    st.divider()
    st.subheader("Công cụ tính Lot")
    rc_l, rc_r = st.columns([1, 1])
    with rc_l:
        cb = st.number_input("Số dư (USD)", min_value=10.0, max_value=1_000_000.0, value=200.0, step=50.0)
        cs = st.number_input("Khoảng cách SL (USD)", min_value=0.1, max_value=200.0, value=3.6, step=0.5)
        cp = st.slider("Rủi ro mỗi lệnh (%)", min_value=0.1, max_value=3.0, value=0.65, step=0.05)
    with rc_r:
        ra    = cb * (cp / 100)
        cl    = max(0.01, round((ra / (100.0 * cs)) / 0.01) * 0.01)
        ml    = cl * 100 * cs
        lc    = GREEN if cl <= 0.1 else AMBER
        st.markdown(_card("Lot Size", f"{cl:.2f}", "XAUUSD", lc), unsafe_allow_html=True)
        st.metric("Số tiền rủi ro",    f"${ra:.2f}")
        st.metric("Lỗ tối đa (SL)",  f"${ml:.2f}")
        st.metric("Rủi ro/Số dư",   f"{(ml/cb)*100:.2f}%")

    st.divider()
    _risk_sigs = live_signals if not live_signals.empty else live_signals_acc2
    if not _risk_sigs.empty and "account_balance" in _risk_sigs.columns:
        st.subheader("Live Balance History")
        bh2 = _risk_sigs[["time", "account_balance"]].dropna().sort_values("time").set_index("time")
        if not bh2.empty:
            st.area_chart(bh2["account_balance"].pipe(_ds), height=200)

    st.divider()
    sl_ev = [e for e in learn_events if e.get("event") in ("self_learn", "live_retrain")]
    st.subheader("Tóm tắt sự kiện Tự học")
    if sl_ev:
        sk1, sk2, sk3 = st.columns(3)
        sk1.metric("Tổng lần retrain",  len(sl_ev))
        imp2 = sum(1 for e in sl_ev if e.get("status") == "improved")
        sk2.metric("Model cải thiện",  imp2)
        brv  = max((e.get("roc_auc", 0) for e in sl_ev), default=0)
        sk3.metric("Best ROC-AUC",    f"{brv:.4f}")
        if len(sl_ev) >= 2:
            st.line_chart(pd.DataFrame({"ROC-AUC": [e.get("roc_auc", 0) for e in sl_ev]}), height=180)
        ev2 = pd.DataFrame(sl_ev[-20:])
        dc2 = [c for c in ["timestamp", "status", "dataset_rows", "roc_auc",
                             "precision", "recall", "f1"] if c in ev2.columns]
        st.dataframe(ev2[dc2] if dc2 else ev2, use_container_width=True)
    else:
        st.info("Chưa có sự kiện tự học. Bật live_learning_enabled=true và chạy bot live.")


# =============================================================================
# TAB 8 — DU LIEU MT5
# =============================================================================
with tab_data:
    st.markdown(
        '<div style="border-bottom:2px solid #1e293b;padding-bottom:12px;margin-bottom:18px">'
        '<span style="font-size:1.35rem;font-weight:800;color:#f1f5f9">Quản lý Dữ liệu MT5</span>'
        '<span style="color:#64748b;font-size:0.82rem;margin-left:12px">'
        'Trạng thái OHLCV, kiểm tra khoảng thiếu</span></div>',
        unsafe_allow_html=True,
    )

    # ── Data coverage per timeframe ──────────────────────────────────────────
    st.subheader("📂 Trạng thái Dữ liệu theo Khung Thời Gian")

    _TF_FILES = {
        "M1":  "XAUUSDm_M1.csv",
        "M5":  "XAUUSDm_M5.csv",
        "M15": "XAUUSDm_M15.csv",
        "M30": "XAUUSDm_M30.csv",
        "H1":  "XAUUSDm_H1.csv",
        "H4":  "XAUUSDm_H4.csv",
        "D1":  "XAUUSDm_D1.csv",
    }
    _DATA_ROOT = ROOT / "src" / "xauusd_ai" / "real_data"

    _tf_status = []
    for _tf, _fname in _TF_FILES.items():
        _fpath = _DATA_ROOT / _fname
        if _fpath.exists():
            try:
                _df_tf = pd.read_csv(_fpath, usecols=["time"], on_bad_lines="skip")
                _df_tf["time"] = pd.to_datetime(_df_tf["time"], errors="coerce")
                _df_tf = _df_tf.dropna()
                _rows = len(_df_tf)
                _first = str(_df_tf["time"].min())[:10] if _rows > 0 else "n/a"
                _last  = str(_df_tf["time"].max())[:10] if _rows > 0 else "n/a"
                _size_kb = round(_fpath.stat().st_size / 1024, 1)
                _tf_status.append({
                    "Timeframe": _tf,
                    "File": _fname,
                    "Rows": f"{_rows:,}",
                    "Tu ngay": _first,
                    "Den ngay": _last,
                    "Size (KB)": _size_kb,
                    "Trang thai": "✅ OK",
                })
            except Exception as _e:
                _tf_status.append({
                    "Timeframe": _tf, "File": _fname,
                    "Rows": "loi", "Tu ngay": "n/a", "Den ngay": "n/a",
                    "Size (KB)": 0, "Trang thai": f"❌ {_e}",
                })
        else:
            _tf_status.append({
                "Timeframe": _tf, "File": _fname,
                "Rows": "0", "Tu ngay": "n/a", "Den ngay": "n/a",
                "Size (KB)": 0, "Trang thai": "⚠️ Chua co file",
            })

    st.dataframe(pd.DataFrame(_tf_status), use_container_width=True, hide_index=True)

    # ── Append instructions ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🔄 Câu Lệnh Cập Nhật Dữ Liệu Từ MT5")
    st.markdown("""
Để tải dữ liệu mới từ MT5 và nối vào file CSV hiện có (không trùng lặp), dùng script sau
trong Python hoặc terminal:
""")
    st.code("""
# Chạy trong Python (hoặc copy vào scripts/fetch_mt5_data.py):
import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path
import datetime

DATA_DIR = Path("src/xauusd_ai/real_data")
SYMBOL = "XAUUSDm"

TIMEFRAMES = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

mt5.initialize()

for tf_name, tf_code in TIMEFRAMES.items():
    csv_path = DATA_DIR / f"{SYMBOL}_{tf_name}.csv"
    # Find last timestamp already saved
    last_ts = None
    if csv_path.exists():
        existing = pd.read_csv(csv_path, usecols=["time"], on_bad_lines="skip")
        existing["time"] = pd.to_datetime(existing["time"], errors="coerce")
        last_ts = existing["time"].max()

    # Fetch from MT5 from last_ts onwards (or last 5000 bars)
    if last_ts is not None and not pd.isna(last_ts):
        from_dt = last_ts.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
        rates = mt5.copy_rates_from(SYMBOL, tf_code, from_dt, 5000)
    else:
        rates = mt5.copy_rates_from_pos(SYMBOL, tf_code, 0, 50000)

    if rates is None or len(rates) == 0:
        print(f"  {tf_name}: khong lay duoc du lieu")
        continue

    new_df = pd.DataFrame(rates)
    new_df["time"] = pd.to_datetime(new_df["time"], unit="s", utc=True)
    new_df = new_df[["time", "open", "high", "low", "close", "tick_volume"]]
    new_df.columns = ["time", "open", "high", "low", "close", "volume"]

    if csv_path.exists():
        old_df = pd.read_csv(csv_path, on_bad_lines="skip")
        old_df["time"] = pd.to_datetime(old_df["time"], errors="coerce")
        combined = pd.concat([old_df, new_df]).drop_duplicates("time").sort_values("time")
    else:
        combined = new_df

    combined.to_csv(csv_path, index=False)
    print(f"  {tf_name}: {len(combined):,} rows -> {csv_path.name}")

mt5.shutdown()
print("Xong!")
""", language="python")

    # ── Output files coverage ────────────────────────────────────────────────
    st.divider()
    st.subheader("📁 Trạng thái File Đầu ra (outputs/)")

    _OUT_FILES = [
        ("paper_trade_signals.csv",             "Tín hiệu live ACC1"),
        ("paper_trade_signals_acc2.csv",        "Tín hiệu live ACC2"),
        ("live_closed_trades.csv",              "Lệnh đã đóng ACC1"),
        ("live_closed_trades_acc2.csv",         "Lệnh đã đóng ACC2"),
        ("backtest_report_acc1.json",           "Báo cáo backtest ACC1 (mới nhất)"),
        ("backtest_trades_acc1.csv",            "Lệnh backtest ACC1 (mới nhất)"),
        ("backtest_report_acc2.json",           "Báo cáo backtest ACC2 (mới nhất)"),
        ("backtest_trades_acc2.csv",            "Lệnh backtest ACC2 (mới nhất)"),
        ("walkforward_report_ict_wyckoff.json", "Báo cáo walk-forward ACC1"),
        ("walkforward_report_acc2.json",        "Báo cáo walk-forward ACC2"),
        ("live_learning_log.jsonl",             "Log học liên tục ACC1"),
        ("live_learning_log_acc2.jsonl",        "Log học liên tục ACC2"),
        ("model_ict_wyckoff.pkl",               "Model ACC1 đang dùng"),
        ("model2_weekly500.pkl",                "Model ACC2 đang dùng"),
        ("model_meta_ict_wyckoff.json",         "Metadata model ACC1"),
        ("model_meta2_weekly500.json",          "Metadata model ACC2"),
        ("training_report_ict_wyckoff.json",    "Kết quả training ACC1"),
    ]

    _out_status = []
    for _fname2, _desc in _OUT_FILES:
        _fp2 = ROOT / "outputs" / _fname2
        if _fp2.exists():
            _mtime = dt.datetime.fromtimestamp(_fp2.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            _sz = round(_fp2.stat().st_size / 1024, 1)
            _out_status.append({"File": _fname2, "Mô tả": _desc, "Size (KB)": _sz,
                                 "Cap nhat cuoi": _mtime, "Trang thai": "✅"})
        else:
            _out_status.append({"File": _fname2, "Mô tả": _desc, "Size (KB)": 0,
                                 "Cap nhat cuoi": "—", "Trang thai": "⚠️ Chua co"})

    st.dataframe(pd.DataFrame(_out_status), use_container_width=True, hide_index=True)

    # ── Bot restart commands ─────────────────────────────────────────────────
    st.divider()
    st.subheader("⚡ Lệnh PowerShell — Khởi động lại Bot & Dashboard")
    st.code("""
# Khởi động lại cả 2 bot (chạy trong PowerShell tại thư mục dự án):
Get-Process python -EA SilentlyContinue | Stop-Process -Force; Start-Sleep 2

$p1 = Start-Process python `
    -ArgumentList "scripts/live_runner.py","live","--config","configs/live_ict_wyckoff.yaml" `
    -RedirectStandardError "outputs\\live_err_acc1.txt" `
    -WorkingDirectory "$PWD" -WindowStyle Hidden -PassThru

$p2 = Start-Process python `
    -ArgumentList "scripts/live_runner.py","live","--config","configs/live_acc2.yaml" `
    -RedirectStandardError "outputs\\live_err_acc2.txt" `
    -WorkingDirectory "$PWD" -WindowStyle Hidden -PassThru

Write-Host "ACC1 PID=$($p1.Id)  ACC2 PID=$($p2.Id)"
""", language="powershell")

    st.code("""
# Khởi động lại Dashboard + Cloudflare tunnel:
$env:PYTHONPATH = "src"
Start-Process ".\.venv\Scripts\streamlit.exe" `
    -ArgumentList "run","src/xauusd_ai/dashboard/app.py","--server.port","8501",
                  "--server.headless","true","--browser.gatherUsageStats","false" `
    -WindowStyle Hidden

Start-Process ".\\cloudflared.exe" `
    -ArgumentList "tunnel","--url","http://localhost:8501","--protocol","http2" `
    -RedirectStandardError "outputs\\tunnel_err.txt" -WindowStyle Hidden

Start-Sleep 12
Get-Content outputs\\tunnel_err.txt | Select-String "trycloudflare.com"
""", language="powershell")
