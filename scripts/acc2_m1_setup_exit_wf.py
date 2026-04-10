#!/usr/bin/env python3
"""
ACC2 M1 Setup-Exit Walk-Forward Search
=======================================
Mục tiêu: Tối ưu label + exit theo từng setup tier (ICT/Wyckoff).

Điểm khác biệt so với acc2_m1_realistic_wf_search.py:
  1. Dataset được build với setup_label_mode="setup_aware":
       - Tier 0 (plain):   tp=1.5R, sl=0.80ATR  (đa số bars)
       - Tier 1 (basic):   tp=1.8R, sl=0.75ATR  (po3, turtle, OTE)
       - Tier 2 (adv):     tp=2.0R, sl=0.72ATR  (wyckoff spring/SOS, IFVG+BOS)
       - Tier 3 (premium): tp=2.5R, sl=0.65ATR  (unicorn + BOS confirm)
  2. Model học label phù hợp với từng setup thay vì 1 label cố định 1.5R cho tất cả.
  3. Backtest engine nhận `setup_tp_rr` / `setup_sl_mult` per-trade từ preds dataframe
     → kết quả aligned với label training.
  4. Search thêm tham số:
       - `setup_exit_scale`:   scale factor toàn bộ tier TP (0.8 / 1.0 / 1.2 / 1.5)
       - `thr_premium`:        ngưỡng riêng cho tier-3 premium signals  (0.52-0.70)
       - `setup_exit_enabled`: True = dùng per-tier TP; False = fixed TP (baseline)

So sánh baseline (fixed tp=1.5) vs setup-aware cùng config:
  → nếu setup-aware PF cao hơn đáng kể → confirm thesis
  → nếu tương đương → label alignment không chip thêm
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Import shared helpers từ existing search script ──────────────────────────
_scripts_dir = str(Path(__file__).parent)
sys.path.insert(0, _scripts_dir)
from acc2_m1_realistic_wf_search import (  # type: ignore[import]
    FoldCache,
    _allowed_hours_from_profile,
    _allowed_weekdays_from_profile,
    _apply_daily_signal_cap,
    _apply_day_model_gate,
    _apply_day_scout_gate,
    _apply_probability_quantile_gate,
    _apply_quality_gate,
    _daily_metrics,
    _fit_day_quality_model,
    _predict,
    _predict_day_quality_probs,
    _train_dir_model,
)

from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import load_settings
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.scalp_dataset import (
    apply_dynamic_sltp_labels,
    build_scalp_dataset,
)
from xauusd_ai.features.scalp_features import SCALP_FEATURE_COLUMNS


# ─── Fold cache ───────────────────────────────────────────────────────────────

def _build_fold_cache_setup(
    dataset: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int,
    n_folds: int,
    label_horizon: int,
    train_objective: str = "day_stability_strict",
    bad_hours_penalty: float = 0.75,
    monday_penalty: float = 0.80,
) -> list[FoldCache]:
    """Build fold cache with setup_tp_rr / setup_sl_mult columns in base_df."""
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
            tr, 1,
            train_objective=train_objective,
            bad_hours_penalty=bad_hours_penalty,
            monday_penalty=monday_penalty,
        )
        sell_model, sell_scaler = _train_dir_model(
            tr, -1,
            train_objective=train_objective,
            bad_hours_penalty=bad_hours_penalty,
            monday_penalty=monday_penalty,
        )
        buy_probs  = _predict(buy_model, buy_scaler, te)
        sell_probs = _predict(sell_model, sell_scaler, te)
        tr_buy_probs  = _predict(buy_model, buy_scaler, tr)
        tr_sell_probs = _predict(sell_model, sell_scaler, tr)

        day_model = _fit_day_quality_model(tr, tr_buy_probs, tr_sell_probs)
        day_quality_probs = _predict_day_quality_probs(day_model, te, buy_probs, sell_probs)

        dirs = te["expected_direction"].values.astype(int)

        base_cols = ["time", "open", "high", "low", "close", "realized_rr"]
        base_df = te[[c for c in base_cols if c in te.columns]].copy()

        # Core numeric defaults
        numeric_defaults: dict[str, float] = {
            "bars_held": 8.0,
            "volatility_regime": 1.0,
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
            # Setup-exit columns — the key new additions
            "setup_tp_rr": 1.5,
            "setup_sl_mult": 0.8,
            "setup_tier": 0.0,
        }
        for col, default in numeric_defaults.items():
            if col in te.columns:
                base_df[col] = pd.to_numeric(te[col], errors="coerce").fillna(default)
            else:
                base_df[col] = default
        if "bars_held" not in te.columns:
            base_df["bars_held"] = float(label_horizon)

        base_df["strategy_score"] = pd.to_numeric(
            te.get("strategy_score", pd.Series(1.0, index=te.index)),
            errors="coerce"
        ).fillna(1.0)
        base_df["day_quality_prob"] = np.clip(day_quality_probs, 0.0, 1.0)
        base_df["split"] = "test"

        test_start = str(pd.to_datetime(te["time"].iloc[0]))
        test_end   = str(pd.to_datetime(te["time"].iloc[-1]))

        # Setup tier summary for info
        t3 = int((base_df.get("setup_tier", pd.Series(0)) == 3).sum())
        t2 = int((base_df.get("setup_tier", pd.Series(0)) == 2).sum())
        t1 = int((base_df.get("setup_tier", pd.Series(0)) == 1).sum())

        caches.append(FoldCache(
            fold=idx,
            test_start=test_start,
            test_end=test_end,
            base_df=base_df,
            buy_probs=buy_probs,
            sell_probs=sell_probs,
            directions=dirs,
            day_quality_probs=day_quality_probs,
        ))
        print(
            f"      fold {idx}/{len(starts)} [{test_start[:16]} → {test_end[:16]}] "
            f"rows={len(te):,}  t1={t1} t2={t2} t3={t3}",
            flush=True,
        )
    return caches


# ─── Candidate sampler ────────────────────────────────────────────────────────

def _sample_candidates(max_candidates: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    thr_vals      = [0.52, 0.56, 0.58, 0.60, 0.62, 0.66, 0.70, 0.75, 0.80]
    risk_vals     = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100]
    daily_vals    = [0.10, 0.16, 0.20, 0.30, 0.50, 1.00]
    max_open_vals = [1, 2, 3]
    side_vals     = [0.5, 0.7, 1.0, 1.2]
    vol_vals      = [0.8, 1.0, 1.2, 1.4]
    cool_vals     = [0, 4, 8, 12]
    exit_scales   = [0.8, 1.0, 1.2, 1.5]   # scale whole tier TP table
    thr_premium   = [0.0, 0.52, 0.56, 0.60, 0.65, 0.70]  # 0 = same as thr_buy
    hour_profiles  = ["config", "liquid_all", "london_only", "ny_open", "ny_only"]
    weekday_profs  = ["all", "midweek", "no_monday", "tue_thu"]
    side_profiles  = ["both", "buy_only"]
    exit_enabled   = [True, False]         # False = baseline fixed-TP for comparison

    # Anchors: best from strategy-first + setup-exit variants
    anchors = [
        # Current best baseline (no setup exit) — for comparison
        {"thr_buy": 0.58, "thr_sell": 0.58, "risk_per_trade": 0.015,
         "daily_loss_limit_pct": 0.20, "max_open_positions": 2,
         "sideway_risk_multiplier": 0.7, "strong_volatility_risk_multiplier": 1.2,
         "cooldown_bars": 8, "setup_exit_enabled": False,
         "setup_exit_scale": 1.0, "thr_premium": 0.0,
         "hour_profile": "config", "weekday_profile": "all",
         "side_profile": "both", "daily_signal_cap": 0,
         "quality_gate_profile": "off"},
        # Setup-exit enabled, same otherwise
        {"thr_buy": 0.58, "thr_sell": 0.58, "risk_per_trade": 0.015,
         "daily_loss_limit_pct": 0.20, "max_open_positions": 2,
         "sideway_risk_multiplier": 0.7, "strong_volatility_risk_multiplier": 1.2,
         "cooldown_bars": 8, "setup_exit_enabled": True,
         "setup_exit_scale": 1.0, "thr_premium": 0.0,
         "hour_profile": "config", "weekday_profile": "all",
         "side_profile": "both", "daily_signal_cap": 0,
         "quality_gate_profile": "off"},
        # Lower risk, setup-exit, scale 1.2
        {"thr_buy": 0.58, "thr_sell": 0.58, "risk_per_trade": 0.010,
         "daily_loss_limit_pct": 0.20, "max_open_positions": 2,
         "sideway_risk_multiplier": 0.7, "strong_volatility_risk_multiplier": 1.2,
         "cooldown_bars": 8, "setup_exit_enabled": True,
         "setup_exit_scale": 1.2, "thr_premium": 0.52,
         "hour_profile": "config", "weekday_profile": "all",
         "side_profile": "both", "daily_signal_cap": 0,
         "quality_gate_profile": "off"},
        # Higher thr for premium setups only
        {"thr_buy": 0.56, "thr_sell": 0.56, "risk_per_trade": 0.020,
         "daily_loss_limit_pct": 0.20, "max_open_positions": 2,
         "sideway_risk_multiplier": 0.7, "strong_volatility_risk_multiplier": 1.2,
         "cooldown_bars": 8, "setup_exit_enabled": True,
         "setup_exit_scale": 1.0, "thr_premium": 0.60,
         "hour_profile": "config", "weekday_profile": "all",
         "side_profile": "both", "daily_signal_cap": 0,
         "quality_gate_profile": "off"},
        # Setup-exit + scale 0.8 (conservative)
        {"thr_buy": 0.60, "thr_sell": 0.60, "risk_per_trade": 0.020,
         "daily_loss_limit_pct": 0.20, "max_open_positions": 2,
         "sideway_risk_multiplier": 0.7, "strong_volatility_risk_multiplier": 1.2,
         "cooldown_bars": 4, "setup_exit_enabled": True,
         "setup_exit_scale": 0.8, "thr_premium": 0.0,
         "hour_profile": "config", "weekday_profile": "all",
         "side_profile": "both", "daily_signal_cap": 0,
         "quality_gate_profile": "off"},
    ]
    out = anchors.copy()
    seen = {tuple(str(v) for v in c.values()) for c in out}
    while len(out) < max_candidates:
        c: dict[str, Any] = {
            "thr_buy":                           rng.choice(thr_vals),
            "thr_sell":                          rng.choice(thr_vals),
            "risk_per_trade":                    rng.choice(risk_vals),
            "daily_loss_limit_pct":              rng.choice(daily_vals),
            "max_open_positions":                rng.choice(max_open_vals),
            "sideway_risk_multiplier":           rng.choice(side_vals),
            "strong_volatility_risk_multiplier": rng.choice(vol_vals),
            "cooldown_bars":                     rng.choice(cool_vals),
            "setup_exit_enabled":                rng.choice(exit_enabled),
            "setup_exit_scale":                  rng.choice(exit_scales),
            "thr_premium":                       rng.choice(thr_premium),
            "hour_profile":                      rng.choice(hour_profiles),
            "weekday_profile":                   rng.choice(weekday_profs),
            "side_profile":                      rng.choice(side_profiles),
            "daily_signal_cap":                  rng.choice([0, 2, 3, 5]),
            "quality_gate_profile":              rng.choice(["off", "setup_align", "smc_core", "unicorn_turtle", "po3_wyckoff"]),
        }
        key = tuple(str(v) for v in c.values())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# ─── Candidate evaluation ─────────────────────────────────────────────────────

def _evaluate_candidate(
    cand: dict[str, Any],
    fold_cache: list[FoldCache],
    settings: Any,
) -> tuple[list[dict], dict[str, float]]:
    """Evaluate a single candidate across all folds. Returns (fold_rows, agg)."""
    setup_exit_enabled = bool(cand.get("setup_exit_enabled", True))
    setup_exit_scale   = float(cand.get("setup_exit_scale", 1.0))
    thr_premium_extra  = float(cand.get("thr_premium", 0.0))  # extra thr for tier3

    fold_rows: list[dict[str, Any]] = []
    sum_net = sum_gp = sum_gl = 0.0
    sum_trades = sum_wins = sum_losses = 0

    for fold in fold_cache:
        s = settings.model_copy(deep=True)
        s.risk.risk_per_trade = float(cand["risk_per_trade"])
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
        s.risk.setup_exit_enabled = bool(cand.get("setup_exit_enabled", True))
        s.risk.setup_exit_scale = float(cand.get("setup_exit_scale", 1.0))
        s.training.backtest_initial_balance = 200.0
        s.training.label_horizon = int(getattr(settings.training, "label_horizon", 8) or 8)
        s.strategy.adx_gate_enabled = False
        s.strategy.min_strategy_score = 0.0
        s.strategy.sideway_min_strategy_score = 0.0
        s.strategy.strong_volatility_min_strategy_score = 0.0
        s.strategy.require_trend_alignment = False

        dirs      = fold.directions
        buy_probs = fold.buy_probs
        sell_probs= fold.sell_probs

        preds = fold.base_df.copy()
        preds["probability"] = np.where(dirs == -1, sell_probs, buy_probs)
        preds["trade_side"]  = np.where(dirs == -1, "sell", "buy")

        # Build initial signal mask using base threshold
        thr_b = float(cand["thr_buy"])
        thr_s = float(cand["thr_sell"])
        buy_sig  = (dirs == 1)  & (buy_probs  >= thr_b)
        sell_sig = (dirs == -1) & (sell_probs >= thr_s)

        # Premium tier: optionally apply a stricter threshold
        if setup_exit_enabled and thr_premium_extra > 0:
            tier_arr = pd.to_numeric(preds.get("setup_tier", 0), errors="coerce").fillna(0).values
            premium_mask = (tier_arr == 3)
            # Premium signals need max(thr_b, thr_premium_extra) probability
            buy_sig  = buy_sig  & (~premium_mask | (buy_probs  >= thr_premium_extra))
            sell_sig = sell_sig & (~premium_mask | (sell_probs >= thr_premium_extra))

        preds["prediction"] = 0
        preds.loc[buy_sig,  "prediction"] = 1
        preds.loc[sell_sig, "prediction"] = 1

        # Apply setup_exit: scale setup_tp_rr / setup_sl_mult columns
        if setup_exit_enabled and "setup_tp_rr" in preds.columns:
            preds["setup_tp_rr"]   = preds["setup_tp_rr"]   * setup_exit_scale
            preds["setup_sl_mult"] = preds["setup_sl_mult"]
            preds["setup_exit_scaled"] = 1
        else:
            # Disable per-trade override so engine uses settings-based TP=1.5
            preds["setup_tp_rr"]   = 0.0
            preds["setup_sl_mult"] = 0.0
            preds["setup_exit_scaled"] = 0

        # Hour / weekday filters
        allowed_hours = _allowed_hours_from_profile(str(cand.get("hour_profile", "config")))
        if allowed_hours is not None and "time" in preds.columns:
            t = pd.to_datetime(preds["time"], utc=True, errors="coerce")
            preds.loc[~t.dt.hour.isin(sorted(allowed_hours)).fillna(False), "prediction"] = 0

        allowed_wd = _allowed_weekdays_from_profile(str(cand.get("weekday_profile", "all")))
        if allowed_wd is not None and "time" in preds.columns:
            t = pd.to_datetime(preds["time"], utc=True, errors="coerce")
            preds.loc[~t.dt.weekday.isin(sorted(allowed_wd)).fillna(False), "prediction"] = 0

        sp = str(cand.get("side_profile", "both"))
        if sp == "buy_only":
            preds.loc[preds["trade_side"] != "buy", "prediction"] = 0
        elif sp == "sell_only":
            preds.loc[preds["trade_side"] != "sell", "prediction"] = 0

        preds = _apply_quality_gate(preds, str(cand.get("quality_gate_profile", "off")))
        preds = _apply_probability_quantile_gate(preds, float(cand.get("probability_quantile", 0.0)))
        preds = _apply_day_scout_gate(preds, str(cand.get("day_scout_gate_profile", "off")))
        preds = _apply_day_model_gate(preds, float(cand.get("day_quality_threshold", 0.0)))
        preds = _apply_daily_signal_cap(preds, int(cand.get("daily_signal_cap", 0)))

        sim = simulate_dynamic_concurrent_backtest(preds, s, RiskManager(s), label="test", compound=True)
        rep  = sim.report
        dmet = _daily_metrics(sim.trades)

        fold_rows.append({
            "fold":                 fold.fold,
            "test_start":           fold.test_start,
            "test_end":             fold.test_end,
            "net_profit":           float(rep.get("net_profit", 0.0)),
            "profit_factor":        float(rep.get("profit_factor", 0.0)),
            "max_drawdown_pct_abs": abs(float(rep.get("max_drawdown_pct", 0.0))),
            "trades":               int(rep.get("trades", 0)),
            "win_rate":             float(rep.get("win_rate", 0.0)),
            **dmet,
        })
        sum_net  += float(rep.get("net_profit", 0.0))
        sum_gp   += float(rep.get("gross_profit", 0.0))
        sum_gl   += float(rep.get("gross_loss", 0.0))
        sum_trades += int(rep.get("trades", 0))
        sum_wins   += int(rep.get("wins", 0))
        sum_losses += int(rep.get("losses", 0))

    fold_df = pd.DataFrame(fold_rows)
    agg = {
        "sum_net_profit":          round(sum_net, 2),
        "avg_profit_factor":       round(float(fold_df["profit_factor"].mean()), 4) if not fold_df.empty else 0.0,
        "global_profit_factor":    round(sum_gp / sum_gl if sum_gl > 0 else 0.0, 4),
        "avg_fold_dd_pct":         round(float(fold_df["max_drawdown_pct_abs"].mean()), 4) if not fold_df.empty else 0.0,
        "min_fold_avg_daily_pnl":  round(float(fold_df["avg_daily_pnl"].min()), 4) if not fold_df.empty else 0.0,
        "avg_of_fold_avg_daily":   round(float(fold_df["avg_daily_pnl"].mean()), 4) if not fold_df.empty else 0.0,
        "min_fold_min_daily_pnl":  round(float(fold_df["min_daily_pnl"].min()), 4) if not fold_df.empty else 0.0,
        "min_fold_days_ge_100_pct":round(float((fold_df["days_pnl_ge_100"] * 100).min()), 4) if not fold_df.empty else 0.0,
        "max_fold_days_neg_pct":   round(float((fold_df["days_pnl_neg"] * 100).max()), 4) if not fold_df.empty else 100.0,
        "min_fold_trades":         int(fold_df["trades"].min()) if not fold_df.empty else 0,
        "worst_fold_max_daily_dd":  round(float(fold_df["max_daily_dd_pct"].max()), 4) if not fold_df.empty else 0.0,
        "global_trades":           int(sum_trades),
        "global_win_rate":         round(sum_wins / max(sum_wins + sum_losses, 1), 4),
        "folds":                   int(len(fold_df)),
    }
    return fold_rows, agg


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACC2 M1 Setup-Exit WF Search")
    p.add_argument("--config",               type=Path,  default=Path("configs/live_acc2_scalp_m1.yaml"))
    p.add_argument("--start-date",           type=str,   default="2023-01-01")
    p.add_argument("--label-max-horizon",    type=int,   default=8)
    p.add_argument("--train-objective",      type=str,   default="day_stability_strict",
                   choices=["baseline", "day_stability", "day_stability_strict"])
    p.add_argument("--train-bad-hours-penalty", type=float, default=0.75)
    p.add_argument("--train-monday-penalty",    type=float, default=0.80)
    p.add_argument("--n-folds",              type=int,   default=8)
    p.add_argument("--train-size",           type=int,   default=250_000)
    p.add_argument("--test-size",            type=int,   default=50_000)
    p.add_argument("--step-size",            type=int,   default=50_000)
    p.add_argument("--max-candidates",       type=int,   default=60)
    p.add_argument("--seed",                 type=int,   default=20260410)
    p.add_argument("--target-daily-pnl",     type=float, default=20.0)
    p.add_argument("--target-daily-dd",      type=float, default=20.0)
    p.add_argument("--target-days-neg-max-pct", type=float, default=60.0)
    p.add_argument("--target-min-trades",    type=int,   default=120)
    # Fixed-label comparison baseline (skip setup-aware labels)
    p.add_argument("--fixed-label",          action="store_true",
                   help="Use fixed label (tp=1.5, sl=0.8) instead of setup-aware")
    p.add_argument("--out-prefix",           type=str,   default="acc2_m1_setup_exit")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    t0 = time.time()
    settings = load_settings(args.config)

    label_mode = "fixed" if args.fixed_label else "setup_aware"
    print("=" * 96, flush=True)
    print("ACC2 M1 SETUP-EXIT WF SEARCH", flush=True)
    print(f"  Config        : {args.config}", flush=True)
    print(f"  Label mode    : {label_mode}", flush=True)
    print(f"  Dataset start : {args.start_date}", flush=True)
    print(f"  Max horizon   : {args.label_max_horizon} M1 bars", flush=True)
    print(f"  Train obj     : {args.train_objective} | bad_h={args.train_bad_hours_penalty} | mon={args.train_monday_penalty}", flush=True)
    print(f"  Folds         : {args.n_folds}  train={args.train_size:,}  test={args.test_size:,}  step={args.step_size:,}", flush=True)
    print(f"  Candidates    : {args.max_candidates}  seed={args.seed}", flush=True)
    print("=" * 96, flush=True)

    # ── 1. Build dataset ──────────────────────────────────────────────────────
    print(f"[1/4] Building scalp dataset (mode={label_mode})...", flush=True)
    dataset = build_scalp_dataset(
        start_date=args.start_date,
        max_horizon=int(args.label_max_horizon),
        setup_label_mode=label_mode,
    )
    if label_mode == "fixed" and settings.training.dynamic_sltp_label_enabled:
        print("      applying dynamic SL/TP relabel...", flush=True)
        dataset = apply_dynamic_sltp_labels(
            dataset, settings=settings,
            max_horizon=int(settings.training.sltp_label_max_horizon or 8),
        )
    print(f"      rows={len(dataset):,}", flush=True)

    # ── 2. Train fold models ──────────────────────────────────────────────────
    print("[2/4] Training fold models + caching probabilities...", flush=True)
    fold_cache = _build_fold_cache_setup(
        dataset=dataset,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        n_folds=args.n_folds,
        label_horizon=int(args.label_max_horizon),
        train_objective=str(args.train_objective),
        bad_hours_penalty=float(args.train_bad_hours_penalty),
        monday_penalty=float(args.train_monday_penalty),
    )
    if not fold_cache:
        raise RuntimeError("No valid folds built.")

    # ── 3. Search candidates ──────────────────────────────────────────────────
    print(f"[3/4] Evaluating {args.max_candidates} candidates...", flush=True)
    candidates = _sample_candidates(args.max_candidates, args.seed)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = -1e18
    feasible_count = 0

    for idx, cand in enumerate(candidates, start=1):
        fold_rows, agg = _evaluate_candidate(cand, fold_cache, settings)
        fold_df = pd.DataFrame(fold_rows)

        # Scoring: same distance formula as main search
        pf_global = agg["global_profit_factor"]
        sum_net   = agg["sum_net_profit"]
        min_daily = agg["min_fold_avg_daily_pnl"]
        worst_dd  = agg["worst_fold_max_daily_dd"]
        max_neg   = agg["max_fold_days_neg_pct"]
        min_trade = agg["min_fold_trades"]
        avg_fold_dd = agg["avg_fold_dd_pct"]

        feasible = bool(
            min_daily >= float(args.target_daily_pnl)
            and worst_dd <= float(args.target_daily_dd)
            and max_neg  <= float(args.target_days_neg_max_pct)
            and min_trade >= int(args.target_min_trades)
        )
        if feasible:
            feasible_count += 1

        gap = (
            max(0.0, float(args.target_daily_pnl) - min_daily) * 50.0
            + max(0.0, worst_dd - float(args.target_daily_dd)) * 200.0
            + max(0.0, max_neg  - float(args.target_days_neg_max_pct)) * 100.0
            + max(0.0, int(args.target_min_trades) - min_trade) * 300.0
            + max(0.0, 2.0 - pf_global) * 200.0
        )
        score = (1e9 + sum_net) if feasible else (-gap + sum_net * 0.01 - avg_fold_dd)

        row: dict[str, Any] = {
            **cand,
            **agg,
            "feasible": feasible,
            "gap_score": round(gap, 4),
        }
        rows.append(row)

        if score > best_score:
            best_score = score
            best = row | {"fold_details": fold_rows}

        if idx % 5 == 0 or feasible:
            bmark = "✅" if feasible else "  "
            exit_tag = "SA" if bool(cand.get("setup_exit_enabled", True)) else "FX"
            print(
                f"  [{idx:>4}/{len(candidates)}]{bmark} [{exit_tag}×{cand.get('setup_exit_scale',1.0):.1f}] "
                f"net=${row['sum_net_profit']:>9,.2f}  PFg={row['global_profit_factor']:.3f}  "
                f"minDay=${row['min_fold_avg_daily_pnl']:>6.2f}  "
                f"ddW={row['worst_fold_max_daily_dd']:.2f}%  "
                f"neg={row['max_fold_days_neg_pct']:.1f}%  "
                f"t={row['min_fold_trades']}",
                flush=True,
            )

    # ── 4. Save ───────────────────────────────────────────────────────────────
    print("[4/4] Saving results...", flush=True)
    res = pd.DataFrame(rows).sort_values(
        ["feasible", "sum_net_profit", "global_profit_factor"],
        ascending=[False, False, False],
    )
    feasible_res = res[res["feasible"] == True]  # noqa: E712

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{args.out_prefix}_{stamp}.csv"
    json_path = out_dir / f"{args.out_prefix}_{stamp}.json"
    res.to_csv(csv_path, index=False)

    payload = {
        "search": {
            "script": "acc2_m1_setup_exit_wf.py",
            "config": str(args.config),
            "label_mode": label_mode,
            "dataset_start": args.start_date,
            "label_max_horizon": int(args.label_max_horizon),
            "train_objective": str(args.train_objective),
            "folds": int(len(fold_cache)),
            "train_size": int(args.train_size),
            "test_size": int(args.test_size),
            "step_size": int(args.step_size),
            "candidates": int(len(res)),
            "feasible_count": int(len(feasible_res)),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "setup_tier_tp_table": {
            "tier_0_plain":   {"tp_rr": 1.5, "sl_mult": 0.80},
            "tier_1_basic":   {"tp_rr": 1.8, "sl_mult": 0.75, "setups": "po3, turtle_soup, OTE"},
            "tier_2_adv":     {"tp_rr": 2.0, "sl_mult": 0.72, "setups": "wyckoff spring/SOS, IFVG+BOS"},
            "tier_3_premium": {"tp_rr": 2.5, "sl_mult": 0.65, "setups": "unicorn + BOS confirm"},
        },
        "best": best,
        "top10": res.head(10).to_dict(orient="records"),
        "top10_feasible": feasible_res.head(10).to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("-" * 96, flush=True)
    if best:
        setup_tag = "SETUP-EXIT" if bool(best.get("setup_exit_enabled", True)) else "FIXED-TP"
        print(
            f"BEST [{setup_tag} scale={best.get('setup_exit_scale',1.0):.1f}]: "
            f"net=${best['sum_net_profit']:,.2f}  PFg={best['global_profit_factor']:.4f}  "
            f"minFoldAvgDay=${best['min_fold_avg_daily_pnl']:.2f}  "
            f"worstDD={best['worst_fold_max_daily_dd']:.2f}%  "
            f"maxNeg={best['max_fold_days_neg_pct']:.1f}%  "
            f"minTrades={best['min_fold_trades']}  feasible={best['feasible']}",
            flush=True,
        )
    print(f"feasible: {len(feasible_res)}/{len(res)}", flush=True)
    print(f"  {csv_path}", flush=True)
    print(f"  {json_path}", flush=True)
    print("-" * 96, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
