#!/usr/bin/env python3
"""
ACC2 M1 realistic walk-forward search.

Goal:
  - Optimize scalp M1 parameters using a realistic simulator
  - Enforce user constraints:
      * Avg daily PnL per fold >= target_daily_pnl
      * Max daily DD per fold <= target_daily_dd_pct

Pipeline:
  1) Build M1 scalp dataset once
  2) Train per fold (BUY/SELL dual models) once
  3) Evaluate many threshold/risk candidates by reusing fold probabilities
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import load_settings
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.scalp_dataset import apply_dynamic_sltp_labels, build_scalp_dataset
from xauusd_ai.features.scalp_features import SCALP_FEATURE_COLUMNS
from xauusd_ai.model.scalp_model import CalibratedDirModel


@dataclass
class FoldCache:
    fold: int
    test_start: str
    test_end: str
    base_df: pd.DataFrame
    buy_probs: np.ndarray
    sell_probs: np.ndarray
    directions: np.ndarray
    day_quality_probs: np.ndarray


def _allowed_hours_from_profile(name: str | None) -> set[int] | None:
    if not name or name == "config":
        return None
    m = {
        "liquid_all": {7, 8, 9, 10, 11, 13, 14, 15, 16, 17},
        "london_only": {7, 8, 9, 10, 11},
        "ny_only": {13, 14, 15, 16, 17},
        "overlap_core": {13, 14, 15},
        "london_open": {7, 8, 9},
        "ny_open": {13, 14, 15, 16},
        "sniper_dual": {8, 14},
        "sniper_ny": {13, 14},
    }
    return m.get(name)


def _allowed_weekdays_from_profile(name: str | None) -> set[int] | None:
    """
    Python weekday convention: Monday=0 ... Sunday=6
    """
    if not name or name == "all":
        return None
    m = {
        "no_monday": {1, 2, 3, 4},
        "tue_thu": {1, 2, 3},
        "wed_only": {2},
        "thu_fri": {3, 4},
        "midweek": {1, 2, 3, 4},
    }
    return m.get(name)


def _apply_daily_signal_cap(preds: pd.DataFrame, cap: int) -> pd.DataFrame:
    """
    Realistic cap: keep first N signals per UTC day (time-order), not hindsight top-prob.
    cap <= 0 => no cap.
    """
    if cap <= 0 or "prediction" not in preds.columns or "time" not in preds.columns:
        return preds
    out = preds.copy()
    t = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out["_date"] = t.dt.date
    out["_time"] = t
    sig_idx = out.index[out["prediction"] == 1].tolist()
    if not sig_idx:
        out.drop(columns=["_date", "_time"], inplace=True, errors="ignore")
        return out
    keep = set()
    sig = out.loc[sig_idx].sort_values("_time")
    for _, g in sig.groupby("_date", sort=True):
        idx = g.index.tolist()
        keep.update(idx[:cap])
    drop_idx = [i for i in sig_idx if i not in keep]
    if drop_idx:
        out.loc[drop_idx, "prediction"] = 0
    out.drop(columns=["_date", "_time"], inplace=True, errors="ignore")
    return out


def _apply_quality_gate(preds: pd.DataFrame, gate_profile: str) -> pd.DataFrame:
    """
    Filter predictions by directional confluence.
    """
    if gate_profile in ("off", "", None):
        return preds
    out = preds.copy()
    if "prediction" not in out.columns or "trade_side" not in out.columns:
        return out

    side_sign = np.where(out["trade_side"].astype(str).str.lower() == "buy", 1.0, -1.0)
    m5_sign = np.sign(pd.to_numeric(out.get("m5_bias", 0.0), errors="coerce").fillna(0.0).values)
    of_sign = np.sign(pd.to_numeric(out.get("of_flow_score", 0.0), errors="coerce").fillna(0.0).values)
    bos_sign = np.sign(pd.to_numeric(out.get("m1_bos", 0.0), errors="coerce").fillna(0.0).values)
    strat_sign = np.sign(pd.to_numeric(out.get("strategy_setup_score", out.get("strategy_score", 0.0)), errors="coerce").fillna(0.0).values)
    trend_sign = np.sign(pd.to_numeric(out.get("trend_strength_score", 0.0), errors="coerce").fillna(0.0).values)
    pull_sign = np.sign(pd.to_numeric(out.get("pullback_quality", 0.0), errors="coerce").fillna(0.0).values)
    exec_sign = np.sign(pd.to_numeric(out.get("execution_quality", 0.0), errors="coerce").fillna(0.0).values)
    turtle_sign = np.sign(pd.to_numeric(out.get("sm_turtle_soup", 0.0), errors="coerce").fillna(0.0).values)
    ifvg_sign = np.sign(pd.to_numeric(out.get("sm_ifvg", 0.0), errors="coerce").fillna(0.0).values)
    unicorn_sign = np.sign(pd.to_numeric(out.get("sm_unicorn", 0.0), errors="coerce").fillna(0.0).values)
    po3_sign = np.sign(pd.to_numeric(out.get("sm_po3_bias", 0.0), errors="coerce").fillna(0.0).values)
    wyck_sign = np.sign(pd.to_numeric(out.get("wyck_sos_sow", 0.0), errors="coerce").fillna(0.0).values)
    lps_sign = np.sign(pd.to_numeric(out.get("wyck_lps_quality", 0.0), errors="coerce").fillna(0.0).values)
    strat_abs = np.abs(pd.to_numeric(out.get("strategy_setup_score", out.get("strategy_score", 0.0)), errors="coerce").fillna(0.0).values)

    keep = np.ones(len(out), dtype=bool)
    if gate_profile == "m5_bias":
        keep = (m5_sign == side_sign)
    elif gate_profile == "oflow":
        keep = (of_sign == side_sign)
    elif gate_profile == "m5_oflow":
        keep = (m5_sign == side_sign) & (of_sign == side_sign)
    elif gate_profile == "confluence3":
        score = (m5_sign == side_sign).astype(int) + (of_sign == side_sign).astype(int) + (bos_sign == side_sign).astype(int)
        keep = score >= 2
    elif gate_profile == "setup_align":
        keep = (strat_sign == side_sign) & (strat_abs >= 0.10)
    elif gate_profile == "trend_exec":
        keep = (trend_sign == side_sign) & (exec_sign == side_sign)
    elif gate_profile == "pull_exec":
        keep = (pull_sign == side_sign) & (exec_sign == side_sign)
    elif gate_profile == "smc_core":
        score = (
            (trend_sign == side_sign).astype(int)
            + (pull_sign == side_sign).astype(int)
            + (exec_sign == side_sign).astype(int)
            + (strat_sign == side_sign).astype(int)
        )
        keep = score >= 3
    elif gate_profile == "unicorn_turtle":
        score = (
            (turtle_sign == side_sign).astype(int)
            + (ifvg_sign == side_sign).astype(int)
            + (unicorn_sign == side_sign).astype(int)
        )
        keep = score >= 1
    elif gate_profile == "po3_wyckoff":
        score = (
            (po3_sign == side_sign).astype(int)
            + (wyck_sign == side_sign).astype(int)
            + (lps_sign == side_sign).astype(int)
        )
        keep = score >= 1

    out.loc[(out["prediction"] == 1) & (~keep), "prediction"] = 0
    return out


def _apply_probability_quantile_gate(preds: pd.DataFrame, q: float) -> pd.DataFrame:
    """
    Keep only predictions above probability quantile among predicted rows.
    q <= 0 => no-op.
    """
    if q <= 0 or "prediction" not in preds.columns or "probability" not in preds.columns:
        return preds
    out = preds.copy()
    mask = out["prediction"] == 1
    if not mask.any():
        return out
    thr = float(out.loc[mask, "probability"].quantile(q))
    out.loc[mask & (out["probability"] < thr), "prediction"] = 0
    return out


def _aligned_series(out: pd.DataFrame, column: str, side_sign: np.ndarray) -> np.ndarray:
    vals = pd.to_numeric(out.get(column, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.clip(vals * side_sign, -1.0, 1.0)


def _apply_setup_confidence_adjustment(preds: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Softly boost/reduce probability using strategy-aware setup scores instead of hard blocking.
    alpha <= 0 => no-op.
    """
    if alpha <= 0 or "probability" not in preds.columns or "trade_side" not in preds.columns:
        return preds
    out = preds.copy()
    side_sign = np.where(out["trade_side"].astype(str).str.lower() == "buy", 1.0, -1.0)
    aligned = (
        0.35 * _aligned_series(out, "strategy_setup_score", side_sign)
        + 0.20 * _aligned_series(out, "trend_strength_score", side_sign)
        + 0.15 * _aligned_series(out, "pullback_quality", side_sign)
        + 0.20 * _aligned_series(out, "execution_quality", side_sign)
        + 0.10 * _aligned_series(out, "sm_po3_bias", side_sign)
    )
    boost = float(alpha) * np.clip(aligned, -1.0, 1.0)
    p = pd.to_numeric(out["probability"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out["probability"] = np.clip(p + boost, 0.0, 0.999)
    return out


def _apply_day_scout_gate(preds: pd.DataFrame, profile: str) -> pd.DataFrame:
    """
    Day-level meta gate using the first N predicted signals of each UTC day.
    If day fails, disable all signals that day.
    If day passes, drop scout signals and keep the remainder (causal behavior).
    """
    if not profile or profile == "off":
        return preds
    if "time" not in preds.columns or "prediction" not in preds.columns:
        return preds

    cfg = {
        "scout2_loose": {"n": 2, "min_prob": 0.70, "min_side_ratio": 0.50},
        "scout2_cons": {"n": 2, "min_prob": 0.76, "min_side_ratio": 0.50},
        "scout3_strict": {"n": 3, "min_prob": 0.80, "min_side_ratio": 0.67},
        "scout4_ultra": {"n": 4, "min_prob": 0.84, "min_side_ratio": 0.75},
    }.get(profile)
    if cfg is None:
        return preds

    out = preds.copy()
    out["_time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out["_date"] = out["_time"].dt.date

    for d, g in out.groupby("_date", sort=True):
        if pd.isna(d):
            continue
        sig = g[g["prediction"] == 1].sort_values("_time")
        n = int(cfg["n"])
        if len(sig) < n:
            out.loc[g.index, "prediction"] = 0
            continue
        scout = sig.head(n)
        mean_prob = float(pd.to_numeric(scout["probability"], errors="coerce").fillna(0.0).mean())
        side_ratio = float(scout["trade_side"].astype(str).value_counts(normalize=True).max())
        ok = mean_prob >= float(cfg["min_prob"]) and side_ratio >= float(cfg["min_side_ratio"])
        if not ok:
            out.loc[g.index, "prediction"] = 0
        else:
            # Keep causal behavior: first N signals are used as scout probes, skip them.
            out.loc[scout.index, "prediction"] = 0

    out.drop(columns=["_time", "_date"], inplace=True, errors="ignore")
    return out


DAY_META_FEATURES = [
    "scout_rows",
    "scout_sig_rows",
    "scout_sig_ratio",
    "scout_mean_prob",
    "scout_max_prob",
    "scout_std_prob",
    "scout_buy_ratio",
    "scout_m5_align_ratio",
    "scout_of_align_ratio",
    "scout_bos_align_ratio",
    "scout_mean_abs_m5",
    "scout_mean_abs_of",
    "scout_mean_abs_bos",
    "scout_mean_atr",
    "scout_hour_start",
    "weekday",
]


def _build_day_meta_rows(
    df: pd.DataFrame,
    buy_probs: np.ndarray,
    sell_probs: np.ndarray,
    *,
    scout_bars: int = 90,
    signal_floor: float = 0.60,
    with_label: bool = True,
) -> pd.DataFrame:
    """
    Build one row per UTC day from the first scout_bars (causal features).
    Label is computed from full-day realized_rr of rows above signal_floor.
    """
    if df.empty or "time" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out["_time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    if int(out["_time"].notna().sum()) == 0:
        return pd.DataFrame()
    out = out.sort_values("_time", na_position="last").reset_index(drop=True)

    dirs = pd.to_numeric(out.get("expected_direction", 0), errors="coerce").fillna(0).to_numpy(dtype=int)
    bp = np.asarray(buy_probs, dtype=float)
    sp = np.asarray(sell_probs, dtype=float)
    if len(bp) != len(out) or len(sp) != len(out):
        m = min(len(out), len(bp), len(sp))
        if m <= 0:
            return pd.DataFrame()
        out = out.iloc[:m].copy().reset_index(drop=True)
        dirs = dirs[:m]
        bp = bp[:m]
        sp = sp[:m]
    prob = np.where(dirs == -1, sp, bp)
    out["_prob"] = np.clip(np.nan_to_num(np.asarray(prob, dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    out["_date"] = out["_time"].dt.date
    out["_rr"] = pd.to_numeric(out.get("realized_rr", 0.0), errors="coerce").fillna(0.0)
    out["_m5"] = pd.to_numeric(out.get("m5_bias", 0.0), errors="coerce").fillna(0.0)
    out["_of"] = pd.to_numeric(out.get("of_flow_score", 0.0), errors="coerce").fillna(0.0)
    out["_bos"] = pd.to_numeric(out.get("m1_bos", 0.0), errors="coerce").fillna(0.0)
    out["_atr"] = pd.to_numeric(out.get("ms_atr5_norm", 0.0), errors="coerce").fillna(0.0)
    out["_dir"] = np.where(dirs >= 0, 1.0, -1.0)

    rows: list[dict[str, float]] = []
    for d, g in out.groupby("_date", sort=True):
        if pd.isna(d):
            continue
        g = g.sort_values("_time")
        scout = g.head(max(10, int(scout_bars)))
        scout_sig = scout[scout["_prob"] >= float(signal_floor)]
        if scout_sig.empty:
            scout_sig = scout
        full_sig = g[g["_prob"] >= float(signal_floor)]
        if full_sig.empty:
            full_sig = g

        side = scout_sig["_dir"].to_numpy(dtype=float)
        m5 = np.sign(scout_sig["_m5"].to_numpy(dtype=float))
        of_ = np.sign(scout_sig["_of"].to_numpy(dtype=float))
        bos = np.sign(scout_sig["_bos"].to_numpy(dtype=float))

        rec = {
            "date": str(d),
            "scout_rows": float(len(scout)),
            "scout_sig_rows": float(len(scout_sig)),
            "scout_sig_ratio": float(len(scout_sig) / max(len(scout), 1)),
            "scout_mean_prob": float(scout_sig["_prob"].mean()),
            "scout_max_prob": float(scout_sig["_prob"].max()),
            "scout_std_prob": float(scout_sig["_prob"].std(ddof=0) if len(scout_sig) > 1 else 0.0),
            "scout_buy_ratio": float((side > 0).mean()),
            "scout_m5_align_ratio": float((m5 == side).mean()),
            "scout_of_align_ratio": float((of_ == side).mean()),
            "scout_bos_align_ratio": float((bos == side).mean()),
            "scout_mean_abs_m5": float(np.abs(scout_sig["_m5"]).mean()),
            "scout_mean_abs_of": float(np.abs(scout_sig["_of"]).mean()),
            "scout_mean_abs_bos": float(np.abs(scout_sig["_bos"]).mean()),
            "scout_mean_atr": float(scout_sig["_atr"].mean()),
            "scout_hour_start": float(int(scout["_time"].dt.hour.iloc[0]) if len(scout) else 0),
            "weekday": float(int(scout["_time"].dt.weekday.iloc[0]) if len(scout) else 0),
        }
        if with_label:
            day_edge = float(full_sig["_rr"].sum())
            rec["label"] = 1 if day_edge > 0 else 0
        rows.append(rec)

    return pd.DataFrame(rows)


def _fit_day_quality_model(
    train_df: pd.DataFrame,
    buy_probs: np.ndarray,
    sell_probs: np.ndarray,
    *,
    scout_bars: int = 90,
    signal_floor: float = 0.60,
) -> HistGradientBoostingClassifier | None:
    day_train = _build_day_meta_rows(
        train_df,
        buy_probs,
        sell_probs,
        scout_bars=scout_bars,
        signal_floor=signal_floor,
        with_label=True,
    )
    if day_train.empty or "label" not in day_train.columns:
        return None
    y = pd.to_numeric(day_train["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    if len(y) < 40:
        return None
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos < 12 or neg < 12:
        return None
    x = day_train[DAY_META_FEATURES].fillna(0.0).to_numpy(dtype=float)
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=8,
        l2_regularization=0.5,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )
    cw = np.where(y == 1, float(neg) / float(max(pos, 1)), 1.0).astype(float)
    if cw.mean() > 0:
        cw /= cw.mean()
    model.fit(x, y, sample_weight=cw)
    return model


def _predict_day_quality_probs(
    model: HistGradientBoostingClassifier | None,
    test_df: pd.DataFrame,
    buy_probs: np.ndarray,
    sell_probs: np.ndarray,
    *,
    scout_bars: int = 90,
    signal_floor: float = 0.60,
) -> np.ndarray:
    if model is None or test_df.empty or "time" not in test_df.columns:
        return np.ones(len(test_df), dtype=float)
    day_test = _build_day_meta_rows(
        test_df,
        buy_probs,
        sell_probs,
        scout_bars=scout_bars,
        signal_floor=signal_floor,
        with_label=False,
    )
    if day_test.empty:
        return np.ones(len(test_df), dtype=float)
    x = day_test[DAY_META_FEATURES].fillna(0.0).to_numpy(dtype=float)
    p = model.predict_proba(x)[:, 1]
    day_map = dict(zip(day_test["date"].astype(str), p))

    t = pd.to_datetime(test_df["time"], utc=True, errors="coerce")
    d = t.dt.date.astype(str)
    arr = np.array([float(day_map.get(k, 1.0)) for k in d], dtype=float)
    return np.clip(arr, 0.0, 1.0)


def _apply_day_model_gate(preds: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Disable signals on days with predicted low day-quality probability.
    threshold <= 0 => no-op.
    """
    if threshold <= 0:
        return preds
    if "prediction" not in preds.columns or "day_quality_prob" not in preds.columns:
        return preds
    out = preds.copy()
    p = pd.to_numeric(out["day_quality_prob"], errors="coerce").fillna(1.0)
    out.loc[(out["prediction"] == 1) & (p < float(threshold)), "prediction"] = 0
    return out


def _make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=50,
        l2_regularization=1.0,
        max_bins=63,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        random_state=42,
    )


def _train_dir_model(
    train_df: pd.DataFrame,
    direction: int,
    train_objective: str = "baseline",
    bad_hours_penalty: float = 0.85,
    monday_penalty: float = 0.90,
) -> tuple[CalibratedDirModel | None, StandardScaler | None]:
    sub = train_df[train_df["expected_direction"] == direction].copy()
    n = len(sub)
    if n < 3000:
        return None, None

    x = sub[SCALP_FEATURE_COLUMNS].fillna(0).values
    y = sub["target"].values
    pos = int(y.sum())
    neg = n - pos
    if pos < 300 or neg < 300:
        return None, None

    scaler = StandardScaler()
    x_s = scaler.fit_transform(x)

    cw = np.where(y == 1, float(neg) / float(pos), 1.0).astype(float)
    decay_half = n * 0.30
    tw = np.exp(np.log(2) * np.arange(n) / decay_half)
    tw /= tw.mean()

    sw = cw * tw
    if train_objective in {"day_stability", "day_stability_strict"}:
        t = pd.to_datetime(sub.get("time"), utc=True, errors="coerce")
        # 1) Equalize contribution per day (avoid a few hyper-active days dominating fit)
        day_w = np.ones(n, dtype=float)
        if t.notna().any():
            d = t.dt.date
            counts = d.value_counts()
            day_w = np.array(
                d.map(lambda v: 1.0 / float(counts.get(v, 1))).fillna(1.0).to_numpy(dtype=float),
                dtype=float,
                copy=True,
            )
            if day_w.mean() > 0:
                day_w /= day_w.mean()
        sw *= day_w

        # 2) Penalize historically noisy windows
        if t.notna().any():
            hh = t.dt.hour.fillna(-1).astype(int)
            bad = hh.between(2, 7) | hh.between(17, 21)
            hp = float(bad_hours_penalty if train_objective == "day_stability" else min(0.75, bad_hours_penalty))
            hour_w = np.where(bad.to_numpy(), hp, 1.0)
            sw *= hour_w

            dow = t.dt.weekday.fillna(-1).astype(int)
            mp = float(monday_penalty if train_objective == "day_stability" else min(0.8, monday_penalty))
            monday_w = np.where(dow.to_numpy() == 0, mp, 1.0)
            sw *= monday_w

        # 3) Mild penalty on extreme micro-vol spikes
        if "ms_atr5_norm" in sub.columns:
            vol = pd.to_numeric(sub["ms_atr5_norm"], errors="coerce").fillna(0.0)
            q = float(vol.quantile(0.90))
            if q > 0:
                vp = 0.92 if train_objective == "day_stability" else 0.85
                vol_w = np.where(vol.to_numpy() >= q, vp, 1.0)
                sw *= vol_w

    if sw.mean() > 0:
        sw /= sw.mean()

    model = _make_model()
    model.fit(x_s, y, sample_weight=sw)

    val_size = max(int(n * 0.15), 1200)
    if val_size >= n:
        return None, None
    x_val = x_s[-val_size:]
    y_val = y[-val_size:]
    raw_p = model.predict_proba(x_val)[:, 1]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(raw_p, y_val)
    wrapped = CalibratedDirModel(model, iso, scaler, direction)
    return wrapped, scaler


def _predict(model: CalibratedDirModel | None, scaler: StandardScaler | None, test_df: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.zeros(len(test_df), dtype=float)
    x = test_df[SCALP_FEATURE_COLUMNS].fillna(0).values
    if isinstance(model, CalibratedDirModel):
        return model.predict_proba(x)[:, 1]
    if scaler is None:
        return np.zeros(len(test_df), dtype=float)
    return model.predict_proba(scaler.transform(x))[:, 1]


def _daily_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "days": 0,
            "avg_daily_pnl": 0.0,
            "median_daily_pnl": 0.0,
            "min_daily_pnl": 0.0,
            "max_daily_pnl": 0.0,
            "max_daily_dd_pct": 0.0,
            "days_pnl_ge_100": 0.0,
            "days_pnl_ge_200": 0.0,
            "days_pnl_neg": 0.0,
        }

    df = trades.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if df.empty:
        return {
            "days": 0,
            "avg_daily_pnl": 0.0,
            "median_daily_pnl": 0.0,
            "min_daily_pnl": 0.0,
            "max_daily_pnl": 0.0,
            "max_daily_dd_pct": 0.0,
            "days_pnl_ge_100": 0.0,
            "days_pnl_ge_200": 0.0,
            "days_pnl_neg": 0.0,
        }

    df["date"] = df["time"].dt.date
    day_rows: list[dict[str, float]] = []
    for _, g in df.groupby("date", sort=True):
        g = g.sort_values("time")
        start_balance = float(g["balance_before"].iloc[0])
        end_balance = float(g["balance_after"].iloc[-1])
        pnl = end_balance - start_balance

        eq = [start_balance]
        eq.extend(float(v) for v in g["balance_after"].tolist())
        peak = eq[0] if eq else start_balance
        max_dd = 0.0
        for v in eq[1:]:
            peak = max(peak, v)
            if peak > 0:
                dd = (peak - v) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd
        day_rows.append({"pnl": pnl, "dd_pct": max_dd})

    day_df = pd.DataFrame(day_rows)
    return {
        "days": int(len(day_df)),
        "avg_daily_pnl": float(day_df["pnl"].mean()),
        "median_daily_pnl": float(day_df["pnl"].median()),
        "min_daily_pnl": float(day_df["pnl"].min()),
        "max_daily_pnl": float(day_df["pnl"].max()),
        "max_daily_dd_pct": float(day_df["dd_pct"].max()),
        "days_pnl_ge_100": float((day_df["pnl"] >= 100.0).mean()),
        "days_pnl_ge_200": float((day_df["pnl"] >= 200.0).mean()),
        "days_pnl_neg": float((day_df["pnl"] < 0.0).mean()),
    }


def _build_fold_cache(
    dataset: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int,
    n_folds: int,
    train_objective: str = "baseline",
    bad_hours_penalty: float = 0.85,
    monday_penalty: float = 0.90,
) -> list[FoldCache]:
    total = len(dataset)
    starts = list(
        range(
            max(0, total - train_size - test_size * n_folds),
            total - train_size - test_size + 1,
            step_size,
        )
    )[-n_folds:]

    caches: list[FoldCache] = []
    for idx, s in enumerate(starts, start=1):
        tr = dataset.iloc[s : s + train_size].copy().reset_index(drop=True)
        te = dataset.iloc[s + train_size : s + train_size + test_size].copy().reset_index(drop=True)
        if len(tr) < 5000 or len(te) < 2000:
            continue

        buy_model, buy_scaler = _train_dir_model(
            tr,
            1,
            train_objective=train_objective,
            bad_hours_penalty=bad_hours_penalty,
            monday_penalty=monday_penalty,
        )
        sell_model, sell_scaler = _train_dir_model(
            tr,
            -1,
            train_objective=train_objective,
            bad_hours_penalty=bad_hours_penalty,
            monday_penalty=monday_penalty,
        )
        buy_probs = _predict(buy_model, buy_scaler, te)
        sell_probs = _predict(sell_model, sell_scaler, te)
        tr_buy_probs = _predict(buy_model, buy_scaler, tr)
        tr_sell_probs = _predict(sell_model, sell_scaler, tr)

        day_model = _fit_day_quality_model(
            tr,
            tr_buy_probs,
            tr_sell_probs,
            scout_bars=90,
            signal_floor=0.60,
        )
        day_quality_probs = _predict_day_quality_probs(
            day_model,
            te,
            buy_probs,
            sell_probs,
            scout_bars=90,
            signal_floor=0.60,
        )

        dirs = te["expected_direction"].values.astype(int)
        base_cols = ["time", "open", "high", "low", "close", "realized_rr"]
        base_df = te[[c for c in base_cols if c in te.columns]].copy()

        # Optional fields (used by dynamic simulator + diagnostics when present).
        optional_numeric_defaults = {
            "future_return": 0.0,
            "directional_return": 0.0,
            "bars_held": 8,
            "session_spread_mult": 1.0,
            "volatility_regime": 1.0,
            "trend_alignment": 1.0,
            "adx": 25.0,
            "atr": 0.0,
            "ms_atr5_norm": 0.0,
            "atr_percentile": 0.5,
            "execution_quality": 0.0,
            "trend_strength_score": 0.0,
            "pullback_quality": 0.0,
            "strategy_setup_score": 0.0,
            "of_flow_score": 0.0,
            "m1_bos": 0.0,
            "m1_macd_hist": 0.0,
            "ms_close_in_rng": 0.5,
            "m5_bias": 0.0,
            "sm_turtle_soup": 0.0,
            "sm_ote_score": 0.0,
            "sm_ifvg": 0.0,
            "sm_unicorn": 0.0,
            "sm_po3_bias": 0.0,
            "wyck_spring_utad": 0.0,
            "wyck_sos_sow": 0.0,
            "wyck_lps_quality": 0.0,
        }
        for col, default in optional_numeric_defaults.items():
            if col in te.columns:
                base_df[col] = pd.to_numeric(te[col], errors="coerce").fillna(default)
            else:
                base_df[col] = default

        if "strategy_score" in te.columns:
            base_df["strategy_score"] = pd.to_numeric(te["strategy_score"], errors="coerce").fillna(1.0)
        else:
            base_df["strategy_score"] = 1.0
        base_df["day_quality_prob"] = np.clip(day_quality_probs, 0.0, 1.0)
        base_df["split"] = "test"

        test_start = str(pd.to_datetime(te["time"].iloc[0]))
        test_end = str(pd.to_datetime(te["time"].iloc[-1]))
        caches.append(
            FoldCache(
                fold=idx,
                test_start=test_start,
                test_end=test_end,
                base_df=base_df,
                buy_probs=buy_probs,
                sell_probs=sell_probs,
                directions=dirs,
                day_quality_probs=day_quality_probs,
            )
        )
        print(
            f"      cached fold {idx}/{len(starts)} test=[{test_start[:16]} -> {test_end[:16]}] rows={len(te):,}",
            flush=True,
        )
    return caches


def _sample_candidates(max_candidates: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    thr_vals = [0.52, 0.56, 0.58, 0.60, 0.62, 0.66, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    risk_vals = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.120, 0.150, 0.200, 0.250, 0.300, 0.400, 0.500, 0.800, 1.000, 1.250, 1.500, 2.000, 2.500]
    daily_vals = [0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.50]
    max_open_vals = [1, 2, 3, 4, 5, 6, 8]
    side_vals = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
    vol_vals = [0.8, 1.0, 1.2, 1.4, 1.6, 2.0]
    cool_vals = [0, 2, 4, 8, 12, 24]
    hour_profiles = [
        "liquid_all",      # 07-11 + 13-17
        "london_only",     # 07-11
        "ny_only",         # 13-17
        "overlap_core",    # 13-15
        "london_open",     # 07-09
        "ny_open",         # 13-16
        "sniper_dual",     # 08 + 14
        "sniper_ny",       # 13-14
    ]
    weekday_profiles = [
        "all",
        "no_monday",
        "tue_thu",
        "wed_only",
        "thu_fri",
        "midweek",
    ]
    side_profiles = ["both", "buy_only", "sell_only"]
    daily_signal_caps = [0, 1, 2, 3, 5]
    quality_gate_profiles = [
        "off",
        "m5_bias",
        "oflow",
        "m5_oflow",
        "confluence3",
        "setup_align",
        "trend_exec",
        "pull_exec",
        "smc_core",
        "unicorn_turtle",
        "po3_wyckoff",
    ]
    probability_quantiles = [0.0, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]
    day_scout_gate_profiles = ["off", "scout2_loose", "scout2_cons", "scout3_strict", "scout4_ultra"]
    day_quality_thresholds = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    setup_confidence_boosts = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15]

    total_space = (
        len(thr_vals)
        * len(thr_vals)
        * len(risk_vals)
        * len(daily_vals)
        * len(max_open_vals)
        * len(side_vals)
        * len(vol_vals)
        * len(cool_vals)
        * len(hour_profiles)
        * len(weekday_profiles)
        * len(side_profiles)
        * len(daily_signal_caps)
        * len(quality_gate_profiles)
        * len(probability_quantiles)
        * len(day_scout_gate_profiles)
        * len(day_quality_thresholds)
        * len(setup_confidence_boosts)
    )

    # Full enumeration only when search space is tiny and requested exhaustively.
    if max_candidates >= total_space and total_space <= 200000:
        grid = []
        for thr_buy in thr_vals:
            for thr_sell in thr_vals:
                for risk in risk_vals:
                    for daily_limit in daily_vals:
                        for max_open in max_open_vals:
                            for side_mult in side_vals:
                                for vol_mult in vol_vals:
                                    for cool in cool_vals:
                                        for hour_profile in hour_profiles:
                                            for weekday_profile in weekday_profiles:
                                                for side_profile in side_profiles:
                                                    for daily_signal_cap in daily_signal_caps:
                                                        for quality_gate_profile in quality_gate_profiles:
                                                            for probability_quantile in probability_quantiles:
                                                                for day_scout_gate_profile in day_scout_gate_profiles:
                                                                    for day_quality_threshold in day_quality_thresholds:
                                                                        for setup_confidence_boost in setup_confidence_boosts:
                                                                            grid.append(
                                                                                {
                                                                                    "thr_buy": thr_buy,
                                                                                    "thr_sell": thr_sell,
                                                                                    "risk_per_trade": risk,
                                                                                    "daily_loss_limit_pct": daily_limit,
                                                                                    "max_open_positions": max_open,
                                                                                    "sideway_risk_multiplier": side_mult,
                                                                                    "strong_volatility_risk_multiplier": vol_mult,
                                                                                    "cooldown_bars": cool,
                                                                                    "hour_profile": hour_profile,
                                                                                    "weekday_profile": weekday_profile,
                                                                                    "side_profile": side_profile,
                                                                                    "daily_signal_cap": daily_signal_cap,
                                                                                    "quality_gate_profile": quality_gate_profile,
                                                                                    "probability_quantile": probability_quantile,
                                                                                    "day_scout_gate_profile": day_scout_gate_profile,
                                                                                    "day_quality_threshold": day_quality_threshold,
                                                                                    "setup_confidence_boost": setup_confidence_boost,
                                                                                }
                                                                            )
        return grid
    # Keep a few anchors deterministically
    anchors = [
        {"thr_buy": 0.58, "thr_sell": 0.58, "risk_per_trade": 0.015, "daily_loss_limit_pct": 0.20, "max_open_positions": 2, "sideway_risk_multiplier": 0.7, "strong_volatility_risk_multiplier": 1.2, "cooldown_bars": 8},
        {"thr_buy": 0.52, "thr_sell": 0.52, "risk_per_trade": 0.020, "daily_loss_limit_pct": 0.16, "max_open_positions": 2, "sideway_risk_multiplier": 0.7, "strong_volatility_risk_multiplier": 1.2, "cooldown_bars": 4},
        {"thr_buy": 0.48, "thr_sell": 0.48, "risk_per_trade": 0.025, "daily_loss_limit_pct": 0.20, "max_open_positions": 3, "sideway_risk_multiplier": 0.7, "strong_volatility_risk_multiplier": 1.4, "cooldown_bars": 4},
        {"thr_buy": 0.80, "thr_sell": 0.80, "risk_per_trade": 0.10, "daily_loss_limit_pct": 0.10, "max_open_positions": 1, "sideway_risk_multiplier": 1.0, "strong_volatility_risk_multiplier": 1.2, "cooldown_bars": 8},
        {"thr_buy": 0.75, "thr_sell": 0.75, "risk_per_trade": 0.02, "daily_loss_limit_pct": 0.20, "max_open_positions": 1, "sideway_risk_multiplier": 1.2, "strong_volatility_risk_multiplier": 1.2, "cooldown_bars": 8, "hour_profile": "ny_only"},
        {"thr_buy": 0.80, "thr_sell": 0.80, "risk_per_trade": 0.06, "daily_loss_limit_pct": 0.20, "max_open_positions": 1, "sideway_risk_multiplier": 1.0, "strong_volatility_risk_multiplier": 1.2, "cooldown_bars": 8, "hour_profile": "ny_only", "weekday_profile": "tue_thu", "side_profile": "buy_only", "daily_signal_cap": 1},
        {"thr_buy": 0.80, "thr_sell": 0.80, "risk_per_trade": 0.04, "daily_loss_limit_pct": 0.20, "max_open_positions": 1, "sideway_risk_multiplier": 1.0, "strong_volatility_risk_multiplier": 1.2, "cooldown_bars": 8, "hour_profile": "ny_only", "weekday_profile": "tue_thu", "side_profile": "both", "daily_signal_cap": 2, "quality_gate_profile": "confluence3", "probability_quantile": 0.9},
        {"thr_buy": 0.75, "thr_sell": 0.75, "risk_per_trade": 0.06, "daily_loss_limit_pct": 0.20, "max_open_positions": 1, "sideway_risk_multiplier": 1.0, "strong_volatility_risk_multiplier": 1.2, "cooldown_bars": 4, "hour_profile": "ny_only", "weekday_profile": "tue_thu", "side_profile": "both", "daily_signal_cap": 3, "quality_gate_profile": "m5_oflow", "probability_quantile": 0.8, "day_scout_gate_profile": "scout2_cons"},
        {"thr_buy": 0.75, "thr_sell": 0.75, "risk_per_trade": 0.06, "daily_loss_limit_pct": 0.20, "max_open_positions": 1, "sideway_risk_multiplier": 1.0, "strong_volatility_risk_multiplier": 1.2, "cooldown_bars": 4, "hour_profile": "ny_open", "weekday_profile": "tue_thu", "side_profile": "both", "daily_signal_cap": 2, "quality_gate_profile": "m5_oflow", "probability_quantile": 0.8, "day_scout_gate_profile": "off", "day_quality_threshold": 0.65},
        {"thr_buy": 0.58, "thr_sell": 0.58, "risk_per_trade": 0.015, "daily_loss_limit_pct": 0.20, "max_open_positions": 2, "sideway_risk_multiplier": 0.7, "strong_volatility_risk_multiplier": 1.2, "cooldown_bars": 8, "hour_profile": "config", "weekday_profile": "all", "side_profile": "both", "daily_signal_cap": 0, "quality_gate_profile": "setup_align", "probability_quantile": 0.0, "day_scout_gate_profile": "off", "day_quality_threshold": 0.0, "setup_confidence_boost": 0.05},
        {"thr_buy": 0.58, "thr_sell": 0.58, "risk_per_trade": 0.015, "daily_loss_limit_pct": 0.20, "max_open_positions": 2, "sideway_risk_multiplier": 0.7, "strong_volatility_risk_multiplier": 1.2, "cooldown_bars": 8, "hour_profile": "config", "weekday_profile": "all", "side_profile": "both", "daily_signal_cap": 0, "quality_gate_profile": "smc_core", "probability_quantile": 0.0, "day_scout_gate_profile": "off", "day_quality_threshold": 0.0, "setup_confidence_boost": 0.05},
        {"thr_buy": 0.58, "thr_sell": 0.58, "risk_per_trade": 0.015, "daily_loss_limit_pct": 0.20, "max_open_positions": 2, "sideway_risk_multiplier": 0.7, "strong_volatility_risk_multiplier": 1.2, "cooldown_bars": 8, "hour_profile": "ny_open", "weekday_profile": "midweek", "side_profile": "both", "daily_signal_cap": 0, "quality_gate_profile": "unicorn_turtle", "probability_quantile": 0.0, "day_scout_gate_profile": "off", "day_quality_threshold": 0.0, "setup_confidence_boost": 0.08},
        {"thr_buy": 0.58, "thr_sell": 0.58, "risk_per_trade": 0.015, "daily_loss_limit_pct": 0.20, "max_open_positions": 2, "sideway_risk_multiplier": 0.7, "strong_volatility_risk_multiplier": 1.2, "cooldown_bars": 8, "hour_profile": "london_only", "weekday_profile": "all", "side_profile": "both", "daily_signal_cap": 0, "quality_gate_profile": "po3_wyckoff", "probability_quantile": 0.0, "day_scout_gate_profile": "off", "day_quality_threshold": 0.0, "setup_confidence_boost": 0.08},
    ]
    out = anchors.copy()
    seen = {tuple(v for v in c.values()) for c in out}
    while len(out) < max_candidates:
        c = {
            "thr_buy": rng.choice(thr_vals),
            "thr_sell": rng.choice(thr_vals),
            "risk_per_trade": rng.choice(risk_vals),
            "daily_loss_limit_pct": rng.choice(daily_vals),
            "max_open_positions": rng.choice(max_open_vals),
            "sideway_risk_multiplier": rng.choice(side_vals),
            "strong_volatility_risk_multiplier": rng.choice(vol_vals),
            "cooldown_bars": rng.choice(cool_vals),
            "hour_profile": rng.choice(hour_profiles),
            "weekday_profile": rng.choice(weekday_profiles),
            "side_profile": rng.choice(side_profiles),
            "daily_signal_cap": rng.choice(daily_signal_caps),
            "quality_gate_profile": rng.choice(quality_gate_profiles),
            "probability_quantile": rng.choice(probability_quantiles),
            "day_scout_gate_profile": rng.choice(day_scout_gate_profiles),
            "day_quality_threshold": rng.choice(day_quality_thresholds),
            "setup_confidence_boost": rng.choice(setup_confidence_boosts),
        }
        key = tuple(v for v in c.values())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACC2 M1 realistic WF search")
    p.add_argument("--config", type=Path, default=Path("configs/live_acc2_scalp_m1.yaml"))
    p.add_argument("--start-date", type=str, default="2023-01-01")
    p.add_argument("--label-sl-atr-mult", type=float, default=0.8)
    p.add_argument("--label-tp-rr", type=float, default=1.5)
    p.add_argument("--label-max-horizon", type=int, default=8)
    p.add_argument("--train-objective", type=str, default="baseline", choices=["baseline", "day_stability", "day_stability_strict"])
    p.add_argument("--train-bad-hours-penalty", type=float, default=0.85)
    p.add_argument("--train-monday-penalty", type=float, default=0.90)
    p.add_argument("--n-folds", type=int, default=8)
    p.add_argument("--train-size", type=int, default=250000)
    p.add_argument("--test-size", type=int, default=50000)
    p.add_argument("--step-size", type=int, default=50000)
    p.add_argument("--max-candidates", type=int, default=70)
    p.add_argument("--seed", type=int, default=20260409)
    p.add_argument("--target-daily-pnl", type=float, default=100.0)
    p.add_argument("--target-daily-dd", type=float, default=20.0)
    p.add_argument("--target-days-ge-100-pct", type=float, default=100.0)
    p.add_argument("--target-days-neg-max-pct", type=float, default=1.0)
    p.add_argument("--target-min-trades-per-fold", type=int, default=1)
    # Verify-only mode: evaluate one fixed candidate (no random search).
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--thr-buy", type=float, default=0.54)
    p.add_argument("--thr-sell", type=float, default=0.54)
    p.add_argument("--risk-per-trade", type=float, default=0.025)
    p.add_argument("--daily-loss-limit-pct", type=float, default=0.20)
    p.add_argument("--max-open-positions", type=int, default=3)
    p.add_argument("--sideway-risk-multiplier", type=float, default=0.70)
    p.add_argument("--strong-volatility-risk-multiplier", type=float, default=1.20)
    p.add_argument("--cooldown-bars", type=int, default=4)
    p.add_argument("--hour-profile", type=str, default="config")
    p.add_argument("--weekday-profile", type=str, default="all")
    p.add_argument("--side-profile", type=str, default="both")
    p.add_argument("--daily-signal-cap", type=int, default=0)
    p.add_argument("--quality-gate-profile", type=str, default="off")
    p.add_argument("--probability-quantile", type=float, default=0.0)
    p.add_argument("--day-scout-gate-profile", type=str, default="off")
    p.add_argument("--day-quality-threshold", type=float, default=0.0)
    p.add_argument("--setup-confidence-boost", type=float, default=0.0)
    p.add_argument("--out-prefix", type=str, default="acc2_m1_realistic_wf_search")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    settings = load_settings(args.config)

    print("=" * 96, flush=True)
    print("ACC2 M1 REALISTIC WF SEARCH", flush=True)
    print(f"Config         : {args.config}", flush=True)
    print(f"Dataset start  : {args.start_date}", flush=True)
    print(
        f"Label params   : sl_atr={args.label_sl_atr_mult:.2f} | tp_rr={args.label_tp_rr:.2f} | horizon={args.label_max_horizon}",
        flush=True,
    )
    print(
        f"Train objective: {args.train_objective} | bad_hours_penalty={args.train_bad_hours_penalty:.2f} | monday_penalty={args.train_monday_penalty:.2f}",
        flush=True,
    )
    print(f"Folds          : {args.n_folds} | train={args.train_size:,} | test={args.test_size:,} | step={args.step_size:,}", flush=True)
    if args.verify_only:
        print("Mode           : verify-only (single fixed candidate)", flush=True)
    else:
        print(f"Candidates     : {args.max_candidates} (seed={args.seed})", flush=True)
    print(
        f"Targets        : avg_daily_pnl_per_fold >= ${args.target_daily_pnl:,.2f} | "
        f"max_daily_dd_per_fold <= {args.target_daily_dd:.2f}% | "
        f"days>=100 per fold >= {args.target_days_ge_100_pct:.2f}% | "
        f"days<0 per fold <= {args.target_days_neg_max_pct:.2f}% | "
        f"trades/fold >= {args.target_min_trades_per_fold}",
        flush=True,
    )
    print("=" * 96, flush=True)

    print("[1/4] Building M1 scalp dataset...", flush=True)
    dataset = build_scalp_dataset(
        start_date=args.start_date,
        sl_atr_mult=float(args.label_sl_atr_mult),
        tp_rr=float(args.label_tp_rr),
        max_horizon=int(args.label_max_horizon),
    )
    if settings.training.dynamic_sltp_label_enabled:
        print("      applying dynamic SL/TP relabel...", flush=True)
        dataset = apply_dynamic_sltp_labels(
            dataset,
            settings=settings,
            max_horizon=int(settings.training.sltp_label_max_horizon or 8),
        )
    print(f"      dataset_rows={len(dataset):,}", flush=True)

    print("[2/4] Training fold models once + caching probabilities...", flush=True)
    fold_cache = _build_fold_cache(
        dataset=dataset,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        n_folds=args.n_folds,
        train_objective=str(args.train_objective),
        bad_hours_penalty=float(args.train_bad_hours_penalty),
        monday_penalty=float(args.train_monday_penalty),
    )
    if not fold_cache:
        raise RuntimeError("No valid folds were built.")

    print("[3/4] Evaluating candidates...", flush=True)
    if args.verify_only:
        fixed_cand = {
            "thr_buy": float(args.thr_buy),
            "thr_sell": float(args.thr_sell),
            "risk_per_trade": float(args.risk_per_trade),
            "daily_loss_limit_pct": float(args.daily_loss_limit_pct),
            "max_open_positions": int(args.max_open_positions),
            "sideway_risk_multiplier": float(args.sideway_risk_multiplier),
            "strong_volatility_risk_multiplier": float(args.strong_volatility_risk_multiplier),
            "cooldown_bars": int(args.cooldown_bars),
            "hour_profile": str(args.hour_profile),
            "weekday_profile": str(args.weekday_profile),
            "side_profile": str(args.side_profile),
            "daily_signal_cap": int(args.daily_signal_cap),
            "quality_gate_profile": str(args.quality_gate_profile),
            "probability_quantile": float(args.probability_quantile),
            "day_scout_gate_profile": str(args.day_scout_gate_profile),
            "day_quality_threshold": float(args.day_quality_threshold),
            "setup_confidence_boost": float(args.setup_confidence_boost),
        }
        candidates = [fixed_cand]
        print(f"      verify candidate: {fixed_cand}", flush=True)
    else:
        candidates = _sample_candidates(args.max_candidates, args.seed)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = -1e18

    for idx, cand in enumerate(candidates, start=1):
        fold_rows: list[dict[str, Any]] = []
        sum_net = 0.0
        sum_gp = 0.0
        sum_gl = 0.0
        sum_trades = 0
        sum_wins = 0
        sum_losses = 0

        for fold in fold_cache:
            s = settings.model_copy(deep=True)
            s.risk.risk_per_trade = float(cand["risk_per_trade"])
            # Search mode: allow broader effective risk (still bounded) for high-return objective.
            s.risk.max_risk_fraction = min(2.50, max(s.risk.max_risk_fraction, float(cand["risk_per_trade"]) * 1.8))
            s.risk.max_open_positions = int(cand["max_open_positions"])
            s.risk.max_total_exposure_pct = min(4.00, max(float(cand["risk_per_trade"]) * 2.5, 0.05))
            s.risk.daily_loss_limit_pct = float(cand["daily_loss_limit_pct"])
            s.risk.consecutive_loss_pause_count = 2
            s.risk.consecutive_loss_cooldown_bars = int(cand["cooldown_bars"])
            s.risk.sideway_risk_multiplier = float(cand["sideway_risk_multiplier"])
            s.risk.strong_volatility_risk_multiplier = float(cand["strong_volatility_risk_multiplier"])
            s.risk.kill_switch_enabled = True
            s.risk.min_confidence = min(float(cand["thr_buy"]), float(cand["thr_sell"]))
            s.training.backtest_initial_balance = 200.0
            s.training.label_horizon = 8
            s.strategy.adx_gate_enabled = False
            s.strategy.min_strategy_score = 0.0
            s.strategy.sideway_min_strategy_score = 0.0
            s.strategy.strong_volatility_min_strategy_score = 0.0
            s.strategy.require_trend_alignment = False

            dirs = fold.directions
            buy_sig = (dirs == 1) & (fold.buy_probs >= float(cand["thr_buy"]))
            sell_sig = (dirs == -1) & (fold.sell_probs >= float(cand["thr_sell"]))

            preds = fold.base_df.copy()
            preds["prediction"] = 0
            preds.loc[buy_sig, "prediction"] = 1
            preds.loc[sell_sig, "prediction"] = 1
            preds["probability"] = np.where(dirs == -1, fold.sell_probs, fold.buy_probs)
            preds["trade_side"] = np.where(dirs == -1, "sell", "buy")
            preds = _apply_setup_confidence_adjustment(preds, float(cand.get("setup_confidence_boost", 0.0)))
            preds["prediction"] = 0
            preds.loc[(dirs == 1) & (preds["probability"] >= float(cand["thr_buy"])), "prediction"] = 1
            preds.loc[(dirs == -1) & (preds["probability"] >= float(cand["thr_sell"])), "prediction"] = 1

            allowed_hours = _allowed_hours_from_profile(str(cand.get("hour_profile", "config")))
            if allowed_hours is not None and "time" in preds.columns:
                t = pd.to_datetime(preds["time"], utc=True, errors="coerce")
                hour_ok = t.dt.hour.isin(sorted(allowed_hours))
                preds.loc[~hour_ok.fillna(False), "prediction"] = 0

            allowed_weekdays = _allowed_weekdays_from_profile(str(cand.get("weekday_profile", "all")))
            if allowed_weekdays is not None and "time" in preds.columns:
                t = pd.to_datetime(preds["time"], utc=True, errors="coerce")
                wd_ok = t.dt.weekday.isin(sorted(allowed_weekdays))
                preds.loc[~wd_ok.fillna(False), "prediction"] = 0

            side_profile = str(cand.get("side_profile", "both"))
            if side_profile == "buy_only":
                preds.loc[preds["trade_side"] != "buy", "prediction"] = 0
            elif side_profile == "sell_only":
                preds.loc[preds["trade_side"] != "sell", "prediction"] = 0

            preds = _apply_quality_gate(preds, str(cand.get("quality_gate_profile", "off")))
            preds = _apply_probability_quantile_gate(preds, float(cand.get("probability_quantile", 0.0)))
            preds = _apply_day_scout_gate(preds, str(cand.get("day_scout_gate_profile", "off")))
            preds = _apply_day_model_gate(preds, float(cand.get("day_quality_threshold", 0.0)))
            preds = _apply_daily_signal_cap(preds, int(cand.get("daily_signal_cap", 0)))

            sim = simulate_dynamic_concurrent_backtest(preds, s, RiskManager(s), label="test", compound=True)
            rep = sim.report
            dmet = _daily_metrics(sim.trades)

            fold_rows.append(
                {
                    "fold": fold.fold,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "net_profit": float(rep.get("net_profit", 0.0)),
                    "profit_factor": float(rep.get("profit_factor", 0.0)),
                    "max_drawdown_pct_abs": abs(float(rep.get("max_drawdown_pct", 0.0))),
                    "trades": int(rep.get("trades", 0)),
                    "win_rate": float(rep.get("win_rate", 0.0)),
                    **dmet,
                }
            )
            sum_net += float(rep.get("net_profit", 0.0))
            sum_gp += float(rep.get("gross_profit", 0.0))
            sum_gl += float(rep.get("gross_loss", 0.0))
            sum_trades += int(rep.get("trades", 0))
            sum_wins += int(rep.get("wins", 0))
            sum_losses += int(rep.get("losses", 0))

        fold_df = pd.DataFrame(fold_rows)
        min_fold_avg_daily = float(fold_df["avg_daily_pnl"].min()) if not fold_df.empty else 0.0
        worst_fold_daily_dd = float(fold_df["max_daily_dd_pct"].max()) if not fold_df.empty else 0.0
        min_fold_daily_pnl = float(fold_df["min_daily_pnl"].min()) if not fold_df.empty else 0.0
        min_fold_days_ge_100_pct = float((fold_df["days_pnl_ge_100"] * 100.0).min()) if not fold_df.empty else 0.0
        max_fold_days_neg_pct = float((fold_df["days_pnl_neg"] * 100.0).max()) if not fold_df.empty else 100.0
        min_fold_trades = int(fold_df["trades"].min()) if not fold_df.empty else 0
        avg_pf = float(fold_df["profit_factor"].mean()) if not fold_df.empty else 0.0
        avg_fold_dd = float(fold_df["max_drawdown_pct_abs"].mean()) if not fold_df.empty else 0.0
        avg_daily_pnl_all = float(fold_df["avg_daily_pnl"].mean()) if not fold_df.empty else 0.0
        pf_global = (sum_gp / sum_gl) if sum_gl > 0 else 0.0
        wr_global = sum_wins / max(sum_wins + sum_losses, 1)

        feasible = bool(
            min_fold_avg_daily >= float(args.target_daily_pnl)
            and worst_fold_daily_dd <= float(args.target_daily_dd)
            and min_fold_days_ge_100_pct >= float(args.target_days_ge_100_pct)
            and max_fold_days_neg_pct <= float(args.target_days_neg_max_pct)
            and min_fold_trades >= int(args.target_min_trades_per_fold)
        )

        gap = (
            max(0.0, float(args.target_daily_pnl) - min_fold_avg_daily) * 50.0
            + max(0.0, worst_fold_daily_dd - float(args.target_daily_dd)) * 200.0
            + max(0.0, float(args.target_days_ge_100_pct) - min_fold_days_ge_100_pct) * 120.0
            + max(0.0, max_fold_days_neg_pct - float(args.target_days_neg_max_pct)) * 120.0
            + max(0.0, int(args.target_min_trades_per_fold) - min_fold_trades) * 300.0
            + max(0.0, 2.0 - pf_global) * 200.0
        )
        score = (1e9 + sum_net) if feasible else (-gap + sum_net * 0.01 - avg_fold_dd)

        row = {
            **cand,
            "folds": int(len(fold_df)),
            "sum_net_profit": round(sum_net, 2),
            "avg_profit_factor": round(avg_pf, 4),
            "global_profit_factor": round(pf_global, 4),
            "avg_fold_dd_pct": round(avg_fold_dd, 4),
            "min_fold_avg_daily_pnl": round(min_fold_avg_daily, 4),
            "avg_of_fold_avg_daily_pnl": round(avg_daily_pnl_all, 4),
            "min_fold_min_daily_pnl": round(min_fold_daily_pnl, 4),
            "min_fold_days_ge_100_pct": round(min_fold_days_ge_100_pct, 4),
            "max_fold_days_neg_pct": round(max_fold_days_neg_pct, 4),
            "min_fold_trades": int(min_fold_trades),
            "worst_fold_max_daily_dd_pct": round(worst_fold_daily_dd, 4),
            "global_trades": int(sum_trades),
            "global_win_rate": round(wr_global, 4),
            "feasible_target": feasible,
            "distance_score": round(gap, 4),
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best = row | {"fold_details": fold_rows}

        if idx % 5 == 0 or feasible:
            tag = "✅" if feasible else "…"
            print(
                f"      [{idx}/{len(candidates)}]{tag} net=${row['sum_net_profit']:,.2f} "
                f"minDaily=${row['min_fold_avg_daily_pnl']:,.2f} "
                f"min%>=100={row['min_fold_days_ge_100_pct']:.2f}% "
                f"max%neg={row['max_fold_days_neg_pct']:.2f}% "
                f"minTrades={row['min_fold_trades']} "
                f"worstDayDD={row['worst_fold_max_daily_dd_pct']:.2f}% "
                f"PFg={row['global_profit_factor']:.3f}",
                flush=True,
            )

    res = pd.DataFrame(rows).sort_values(
        ["feasible_target", "sum_net_profit", "global_profit_factor"],
        ascending=[False, False, False],
    )
    feasible_res = res[res["feasible_target"] == True]  # noqa: E712

    print("[4/4] Saving report...", flush=True)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{args.out_prefix}_{stamp}.csv"
    json_path = out_dir / f"{args.out_prefix}_{stamp}.json"
    res.to_csv(csv_path, index=False)
    payload = {
        "search": {
            "config": str(args.config),
            "mode": "verify_only" if args.verify_only else "search",
            "dataset_start": args.start_date,
            "label_sl_atr_mult": float(args.label_sl_atr_mult),
            "label_tp_rr": float(args.label_tp_rr),
            "label_max_horizon": int(args.label_max_horizon),
            "train_objective": str(args.train_objective),
            "train_bad_hours_penalty": float(args.train_bad_hours_penalty),
            "train_monday_penalty": float(args.train_monday_penalty),
            "folds": int(len(fold_cache)),
            "train_size": int(args.train_size),
            "test_size": int(args.test_size),
            "step_size": int(args.step_size),
            "candidates": int(len(res)),
            "feasible_count": int(len(feasible_res)),
            "target_daily_pnl": float(args.target_daily_pnl),
            "target_daily_dd_pct": float(args.target_daily_dd),
            "target_days_ge_100_pct": float(args.target_days_ge_100_pct),
            "target_days_neg_max_pct": float(args.target_days_neg_max_pct),
            "target_min_trades_per_fold": int(args.target_min_trades_per_fold),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "best": best,
        "top10": res.head(10).to_dict(orient="records"),
        "top10_feasible": feasible_res.head(10).to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("-" * 96, flush=True)
    if best:
        print(
            f"Best: net=${best['sum_net_profit']:,.2f} | minFoldAvgDaily=${best['min_fold_avg_daily_pnl']:,.2f} "
            f"| minFold%>=100={best['min_fold_days_ge_100_pct']:.2f}% "
            f"| maxFold%neg={best['max_fold_days_neg_pct']:.2f}% "
            f"| minFoldTrades={best['min_fold_trades']} "
            f"| worstFoldDayDD={best['worst_fold_max_daily_dd_pct']:.2f}% | PFg={best['global_profit_factor']:.3f} "
            f"| feasible={best['feasible_target']}",
            flush=True,
        )
    print(f"feasible_count: {len(feasible_res)}/{len(res)}", flush=True)
    print(f"saved: {csv_path}", flush=True)
    print(f"saved: {json_path}", flush=True)
    print("-" * 96, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
