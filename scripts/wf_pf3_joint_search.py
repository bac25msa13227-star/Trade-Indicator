#!/usr/bin/env python3
"""
PF>=3 Joint Constraint Search — Progressive Multi-Round Optimizer.

Goal:
  Find WF config meeting ALL three:
    (1) PF >= 3.0
    (2) sum_net_profit >= old_net  (e.g. 127_313.86 for acc1, 35_714.67 for acc2)
    (3) global_max_drawdown_pct < old_dd  (e.g. 32.15% for acc1, 22.81% for acc2)

Strategy:
  Round 1  — Focused explosive-compounding space:
     compound_cap=0 (unlimited), high TP ratios (5-10:1), tight circuit breakers.
  Round 2  — Wider risk range (risk_per_trade up to 0.15) + moderate compound.
  Round 3  — Ultra-aggressive search; if still not found, relax net floor by 20%
             and keep searching until a joint-feasible candidate is found or
             the net floor drops below 20% of original (at which point the
             script reports the Pareto-best and exits).

Output:
  outputs/<prefix>_roundN.csv and _roundN_report.json per round.
  outputs/<prefix>_best_feasible.json  — updated whenever any feasible found.
  outputs/<prefix>_pareto_best.json    — best on each axis when no joint feasible.
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

# ── bootstrap path so we can import xauusd_ai ────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import Settings, load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.dataset import build_merged_context, prepare_training_dataset
from xauusd_ai.model.trainer import ModelTrainer
from xauusd_ai.strategies.hybrid import HybridStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Fold cache
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldCache:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    predictions: pd.DataFrame


def _max_dd_from_curve(curve: list[float]) -> float:
    if not curve:
        return 0.0
    peak = curve[0]
    max_dd = 0.0
    for v in curve[1:]:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd * 100.0


def _build_fold_cache(settings: Settings) -> tuple[list[FoldCache], pd.DataFrame]:
    print("[cache] Loading market data + building dataset …", flush=True)
    service = MarketDataService(settings)
    frames = service.fetch_multi_timeframe_data(
        source=settings.market.training_data_source,
        all_bars=True,
    )
    merged = build_merged_context(settings, frames)
    dataset = prepare_training_dataset(
        settings, frames, HybridStrategy(settings), cached_merged=merged
    )
    print(f"[cache]  dataset_rows={len(dataset):,}", flush=True)

    train_size = settings.training.walkforward_train_size
    test_size  = settings.training.walkforward_test_size
    step_size  = settings.training.walkforward_step_size
    max_folds  = settings.training.walkforward_max_folds_per_combination or 8
    fold_indices = list(range(0, max(len(dataset) - train_size - test_size + 1, 0), step_size))
    if max_folds > 0:
        fold_indices = fold_indices[-max_folds:]

    print(f"[cache] Training {len(fold_indices)} fold models …", flush=True)
    cached: list[FoldCache] = []
    for idx, fold_start in enumerate(fold_indices, start=1):
        train_end = fold_start + train_size
        test_end  = train_end  + test_size
        fold_train = dataset.iloc[fold_start:train_end].copy()
        fold_test  = dataset.iloc[train_end:test_end].copy()
        if len(fold_train) < 200 or len(fold_test) < 50:
            continue

        fold_dataset = pd.concat([fold_train, fold_test], ignore_index=True)
        fold_dataset["split"] = "train"
        fold_dataset.loc[len(fold_train):, "split"] = "test"

        trainer = ModelTrainer(settings)
        trainer.train(fold_dataset, save_artifacts=False)
        preds = trainer.predict_dataset(fold_dataset)

        cached.append(FoldCache(
            fold=idx,
            train_start=str(fold_train["time"].min()),
            train_end=str(fold_train["time"].max()),
            test_start=str(fold_test["time"].min()),
            test_end=str(fold_test["time"].max()),
            predictions=preds.copy(),
        ))
        print(
            f"[cache]   fold {idx}/{len(fold_indices)} "
            f"  test=[{cached[-1].test_start[:10]} → {cached[-1].test_end[:10]}]  rows={len(preds)}",
            flush=True,
        )
    if not cached:
        raise RuntimeError("No valid folds — check dataset size / walkforward settings.")
    print(f"[cache] Done. {len(cached)} folds cached.\n", flush=True)
    return cached, dataset


# ─────────────────────────────────────────────────────────────────────────────
# Time templates
# ─────────────────────────────────────────────────────────────────────────────

def _time_templates() -> dict[str, dict[str, Any]]:
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    return {
        "minimal": {
            "blocked_hours_utc": [15, 22, 23],
            "blocked_weekday_hours_utc": {},
            "allowed_weekday_hours_utc": {},
        },
        "block_3_17": {
            "blocked_hours_utc": [3, 15, 17, 22, 23],
            "blocked_weekday_hours_utc": {},
            "allowed_weekday_hours_utc": {},
        },
        "focus_hours": {
            "blocked_hours_utc": [],
            "blocked_weekday_hours_utc": {},
            "allowed_weekday_hours_utc": {w: [0, 1, 5, 6, 7, 13, 14, 20, 21] for w in weekdays},
        },
        "killzones_only": {
            "blocked_hours_utc": [],
            "blocked_weekday_hours_utc": {},
            "allowed_weekday_hours_utc": {w: sorted(range(6, 17)) for w in weekdays},
        },
        "mon_tue_weak_cluster": {
            "blocked_hours_utc": [15, 22, 23],
            "blocked_weekday_hours_utc": {
                "Monday":  sorted(set(list(range(2, 8)) + list(range(17, 22)))),
                "Tuesday": sorted(set(list(range(2, 8)) + list(range(17, 22)))),
            },
            "allowed_weekday_hours_utc": {},
        },
        "london_ny": {
            "blocked_hours_utc": [],
            "blocked_weekday_hours_utc": {},
            "allowed_weekday_hours_utc": {w: list(range(7, 16)) for w in weekdays},
        },
        "asia_london": {
            "blocked_hours_utc": [],
            "blocked_weekday_hours_utc": {},
            "allowed_weekday_hours_utc": {w: list(range(0, 12)) for w in weekdays},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Candidate application
# ─────────────────────────────────────────────────────────────────────────────

def _apply_candidate(
    base: Settings,
    c: dict[str, Any],
    templates: dict[str, dict[str, Any]],
) -> Settings:
    s = base.model_copy(deep=True)

    # ── Strategy filters ────────────────────────────────────────────────────
    s.strategy.signal_threshold                          = float(c["threshold"])
    s.strategy.min_strategy_score                        = float(c["min_strategy_score"])
    s.strategy.sideway_min_strategy_score                = float(c["sideway_min_strategy_score"])
    s.strategy.strong_volatility_min_strategy_score      = float(c["strong_volatility_min_strategy_score"])
    s.strategy.require_trend_alignment                   = bool(c["require_trend_alignment"])
    s.strategy.sideway_min_confidence                    = float(c["sideway_min_conf"])
    s.strategy.volatile_min_confidence                   = float(c["volatile_min_conf"])
    s.strategy.silver_bullet_enabled                     = bool(c.get("silver_bullet_enabled", True))
    s.strategy.adx_gate_enabled                         = bool(c.get("adx_gate_enabled", True))
    s.strategy.adx_min_trend                            = float(c.get("adx_min_trend", 17.0))

    tpl = templates[str(c["template"])]
    s.strategy.blocked_hours_utc         = list(tpl["blocked_hours_utc"])
    s.strategy.blocked_weekday_hours_utc = dict(tpl["blocked_weekday_hours_utc"])
    s.strategy.allowed_weekday_hours_utc = dict(tpl["allowed_weekday_hours_utc"])

    # ── Risk sizing ─────────────────────────────────────────────────────────
    s.risk.min_confidence                       = float(c["risk_min_conf"])
    s.risk.risk_per_trade                       = float(c["risk_per_trade"])
    s.risk.risk_tier_floor                      = float(c["risk_tier_floor"])
    s.risk.max_risk_fraction                    = float(c.get("max_risk_fraction", min(0.50, c["risk_per_trade"] * 5)))
    s.risk.sideway_risk_multiplier              = float(c["sideway_risk_multiplier"])
    s.risk.strong_volatility_risk_multiplier    = float(c["strong_volatility_risk_multiplier"])
    s.risk.max_open_positions                   = int(c["max_open_positions"])
    s.risk.max_total_exposure_pct               = float(c.get("max_total_exposure_pct", 0.30))

    # ── SL / TP ratios ──────────────────────────────────────────────────────
    s.risk.stop_loss_atr_multiple       = float(c.get("stop_loss_atr_multiple", 1.5))
    s.risk.take_profit_rr               = float(c["take_profit_rr"])
    s.risk.sideway_take_profit_rr       = float(c.get("sideway_take_profit_rr", max(2.0, c["take_profit_rr"] - 1.0)))
    s.risk.volatile_take_profit_rr      = float(c.get("volatile_take_profit_rr", c["take_profit_rr"] + 1.0))
    s.risk.sideway_sl_atr_multiple      = float(c.get("sideway_sl_atr_multiple", 1.0))
    s.risk.volatile_sl_atr_multiple     = float(c.get("volatile_sl_atr_multiple", 2.0))

    # ── Partial TP ──────────────────────────────────────────────────────────
    s.risk.partial_tp_enabled           = bool(c.get("partial_tp_enabled", False))
    s.risk.partial_tp_rr                = float(c.get("partial_tp_rr", 1.0))
    s.risk.partial_tp_pct               = float(c.get("partial_tp_pct", 0.4))

    # ── Circuit breakers ────────────────────────────────────────────────────
    s.risk.anti_martingale_factor          = float(c["anti_martingale_factor"])
    s.risk.consecutive_loss_pause_count    = int(c["consecutive_loss_pause_count"])
    s.risk.consecutive_loss_cooldown_bars  = int(c["consecutive_loss_cooldown_bars"])
    s.risk.daily_loss_limit_pct            = float(c["daily_loss_limit_pct"])
    s.risk.kill_switch_enabled             = True

    # ── Compound cap ────────────────────────────────────────────────────────
    # 0 = unlimited (allows explosive fold-over-fold compounding)
    s.risk.compound_cap                    = float(c.get("compound_cap", 0.0))

    # ── Friction ───────────────────────────────────────────────────────────
    s.risk.spread_cost_rr  = float(c.get("spread_cost_rr", 0.01))
    s.risk.slippage_rr     = float(c.get("slippage_rr", 0.02))
    s.risk.commission_rr   = float(c.get("commission_rr", 0.0))

    return s


# ─────────────────────────────────────────────────────────────────────────────
# Candidate evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(
    base: Settings,
    fold_cache: list[FoldCache],
    c: dict[str, Any],
    templates: dict[str, dict[str, Any]],
    initial_balance: float,
    pf_min: float,
    net_min: float,
    dd_max: float,
) -> dict[str, Any]:
    tuned   = _apply_candidate(base, c, templates)
    balance = float(initial_balance)
    equity_curve: list[float] = [balance]
    global_peak  = balance
    global_max_dd = 0.0
    fold_rows: list[dict[str, Any]] = []

    for fold in fold_cache:
        fold_s = tuned.model_copy(deep=True)
        fold_s.training.backtest_initial_balance = balance

        preds = fold.predictions.copy()
        preds["prediction"] = (preds["probability"] >= float(c["threshold"])).astype(int)

        sim = simulate_dynamic_concurrent_backtest(
            preds, fold_s, RiskManager(fold_s), label="test", compound=True,
        )
        rep  = sim.report
        fold_start_bal = balance
        balance = float(rep["ending_balance"])
        equity_curve.append(balance)

        # Track global max-DD conservatively from fold-level data
        global_peak = max(global_peak, fold_start_bal)
        fold_dd_frac = abs(float(rep["max_drawdown_pct"])) / 100.0
        fold_low_est = max(0.0, fold_start_bal * (1.0 - fold_dd_frac))
        if global_peak > 0:
            global_max_dd = max(global_max_dd, (global_peak - fold_low_est) / global_peak)
        global_peak = max(global_peak, balance)
        if global_peak > 0:
            global_max_dd = max(global_max_dd, (global_peak - balance) / global_peak)

        fold_rows.append({
            "fold": fold.fold,
            "ending_balance": balance,
            "net_profit":     float(rep["net_profit"]),
            "gross_profit":   float(rep["gross_profit"]),
            "gross_loss":     float(rep["gross_loss"]),
            "profit_factor":  float(rep["profit_factor"]),
            "max_drawdown_pct": float(rep["max_drawdown_pct"]),
            "trades":         int(rep["trades"]),
            "return_pct":     float(rep["return_pct"]),
            "wins":           int(rep["wins"]),
            "losses":         int(rep["losses"]),
        })

    fold_df = pd.DataFrame(fold_rows)
    gross_profit   = float(fold_df["gross_profit"].sum())   if not fold_df.empty else 0.0
    gross_loss     = float(fold_df["gross_loss"].sum())     if not fold_df.empty else 0.0
    net_profit     = balance - initial_balance
    profit_factor  = gross_profit / gross_loss if gross_loss > 0 else 0.0
    exact_dd_pct   = _max_dd_from_curve(equity_curve)
    max_dd_pct     = max(global_max_dd * 100.0, exact_dd_pct)
    total_trades   = int(fold_df["trades"].sum())   if not fold_df.empty else 0
    total_wins     = int(fold_df["wins"].sum())     if not fold_df.empty else 0
    total_losses   = int(fold_df["losses"].sum())   if not fold_df.empty else 0
    win_rate       = total_wins / max(total_wins + total_losses, 1)

    feasible = bool(
        profit_factor >= pf_min and
        net_profit    >= net_min and
        max_dd_pct    <  dd_max
    )

    # Distance score: 0 = fully feasible; larger = farther from target
    pf_gap  = max(0.0, pf_min  - profit_factor)
    net_gap = max(0.0, net_min - net_profit)
    dd_gap  = max(0.0, max_dd_pct - dd_max)
    distance = pf_gap * 5000 + net_gap * 0.01 + dd_gap * 2000

    return {
        **c,
        "starting_balance":         round(initial_balance, 2),
        "ending_balance":           round(balance, 2),
        "sum_net_profit":           round(net_profit, 2),
        "sum_gross_profit":         round(gross_profit, 2),
        "sum_gross_loss":           round(gross_loss, 2),
        "global_max_drawdown_pct":  round(max_dd_pct, 4),
        "profit_factor_global":     round(profit_factor, 4),
        "total_trades":             total_trades,
        "win_rate":                 round(win_rate, 4),
        "avg_fold_return_pct":      round(float(fold_df["return_pct"].mean()) if not fold_df.empty else 0.0, 4),
        "folds":                    int(len(fold_df)),
        "feasible":                 feasible,
        "distance_score":           round(distance, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parameter spaces per round
# ─────────────────────────────────────────────────────────────────────────────

def _space_round1(templates: list[str]) -> dict[str, list[Any]]:
    """
    Round 1: explosive compounding (compound_cap=0), high TP ratios, tight circuit breakers.
    Targeting PF>=3 via high TP:SL ratio + selective filters.
    """
    return {
        # Entry selectivity
        "threshold":                          [0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96],
        "min_strategy_score":                 [0.20, 0.30, 0.40, 0.50, 0.60],
        "sideway_min_strategy_score":         [0.15, 0.25, 0.35, 0.45],
        "strong_volatility_min_strategy_score": [0.20, 0.30, 0.40, 0.50],
        "require_trend_alignment":            [True, False],
        "risk_min_conf":                      [0.65, 0.70, 0.75, 0.80, 0.85],
        "sideway_min_conf":                   [0.78, 0.82, 0.86, 0.90, 0.94],
        "volatile_min_conf":                  [0.72, 0.76, 0.80, 0.84, 0.88],
        "template":                           templates,
        "silver_bullet_enabled":              [True, False],
        "adx_gate_enabled":                   [True],
        "adx_min_trend":                      [15.0, 18.0, 22.0, 26.0],
        # Risk sizing — aggressive but controlled
        "risk_per_trade":                     [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12],
        "risk_tier_floor":                    [0.02, 0.03, 0.04, 0.05, 0.06],
        "sideway_risk_multiplier":            [0.20, 0.30, 0.40, 0.50],
        "strong_volatility_risk_multiplier":  [0.60, 0.80, 1.00, 1.20, 1.40],
        "max_open_positions":                 [1, 2, 3],
        "max_total_exposure_pct":             [0.15, 0.20, 0.25, 0.30],
        "max_risk_fraction":                  [0.20, 0.30, 0.40, 0.50],
        # TP/SL — high TP ratio boosts PF
        "stop_loss_atr_multiple":             [1.0, 1.2, 1.5, 1.8],
        "take_profit_rr":                     [4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        "sideway_sl_atr_multiple":            [0.8, 1.0, 1.2],
        "volatile_sl_atr_multiple":           [1.5, 2.0, 2.5, 3.0],
        # Partial TP
        "partial_tp_enabled":                 [True, False],
        "partial_tp_rr":                      [1.0, 1.5, 2.0],
        "partial_tp_pct":                     [0.30, 0.40, 0.50],
        # Circuit breakers — tight to protect DD
        "anti_martingale_factor":             [0.40, 0.50, 0.60, 0.70],
        "consecutive_loss_pause_count":       [2, 3],
        "consecutive_loss_cooldown_bars":     [12, 16, 20, 24],
        "daily_loss_limit_pct":              [0.015, 0.020, 0.025, 0.030],
        # Compounding — unlimited for explosive growth
        "compound_cap":                       [0.0],   # 0 = unlimited
        # Friction (realistic)
        "spread_cost_rr":                     [0.01, 0.02],
        "slippage_rr":                        [0.01, 0.02],
        "commission_rr":                      [0.0, 0.005],
    }


def _space_round2(templates: list[str]) -> dict[str, list[Any]]:
    """
    Round 2: push even harder — higher risk, moderate compound cap too.
    Also explore slightly lower thresholds for more trade density.
    """
    return {
        "threshold":                          [0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96],
        "min_strategy_score":                 [0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        "sideway_min_strategy_score":         [0.10, 0.20, 0.30, 0.40],
        "strong_volatility_min_strategy_score": [0.15, 0.25, 0.35, 0.45],
        "require_trend_alignment":            [True, False],
        "risk_min_conf":                      [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
        "sideway_min_conf":                   [0.74, 0.78, 0.82, 0.86, 0.90, 0.94],
        "volatile_min_conf":                  [0.68, 0.72, 0.76, 0.80, 0.84, 0.88],
        "template":                           templates,
        "silver_bullet_enabled":              [True, False],
        "adx_gate_enabled":                   [True, False],
        "adx_min_trend":                      [15.0, 20.0, 25.0],
        "risk_per_trade":                     [0.08, 0.10, 0.12, 0.15, 0.18, 0.20],
        "risk_tier_floor":                    [0.04, 0.06, 0.08, 0.10],
        "sideway_risk_multiplier":            [0.15, 0.20, 0.30, 0.40, 0.50],
        "strong_volatility_risk_multiplier":  [0.80, 1.00, 1.20, 1.40, 1.60],
        "max_open_positions":                 [1, 2, 3, 4],
        "max_total_exposure_pct":             [0.20, 0.30, 0.40, 0.50],
        "max_risk_fraction":                  [0.30, 0.40, 0.50, 0.60],
        "stop_loss_atr_multiple":             [0.8, 1.0, 1.2, 1.5],
        "take_profit_rr":                     [5.0, 6.0, 7.0, 8.0, 10.0, 12.0],
        "sideway_sl_atr_multiple":            [0.7, 0.8, 1.0, 1.2],
        "volatile_sl_atr_multiple":           [1.5, 2.0, 2.5, 3.0],
        "partial_tp_enabled":                 [True, False],
        "partial_tp_rr":                      [1.0, 1.5, 2.0, 2.5],
        "partial_tp_pct":                     [0.25, 0.35, 0.45, 0.50],
        "anti_martingale_factor":             [0.30, 0.40, 0.50, 0.60, 0.70],
        "consecutive_loss_pause_count":       [2, 3, 4],
        "consecutive_loss_cooldown_bars":     [12, 16, 20, 24, 32],
        "daily_loss_limit_pct":              [0.010, 0.015, 0.020, 0.025, 0.030],
        "compound_cap":                       [0.0, 1000.0],
        "spread_cost_rr":                     [0.0, 0.01, 0.02],
        "slippage_rr":                        [0.01, 0.02, 0.03],
        "commission_rr":                      [0.0, 0.005],
    }


def _space_round3(templates: list[str]) -> dict[str, list[Any]]:
    """
    Round 3: extreme risk / ultra-compound — last resort before relaxing net floor.
    """
    return {
        "threshold":                          [0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96],
        "min_strategy_score":                 [0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
        "sideway_min_strategy_score":         [0.05, 0.15, 0.25, 0.35],
        "strong_volatility_min_strategy_score": [0.10, 0.20, 0.30, 0.40],
        "require_trend_alignment":            [True, False],
        "risk_min_conf":                      [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        "sideway_min_conf":                   [0.70, 0.75, 0.80, 0.85, 0.90],
        "volatile_min_conf":                  [0.65, 0.70, 0.75, 0.80, 0.85],
        "template":                           templates,
        "silver_bullet_enabled":              [True, False],
        "adx_gate_enabled":                   [True, False],
        "adx_min_trend":                      [12.0, 15.0, 20.0, 25.0],
        "risk_per_trade":                     [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
        "risk_tier_floor":                    [0.06, 0.08, 0.10, 0.12],
        "sideway_risk_multiplier":            [0.10, 0.20, 0.30, 0.40],
        "strong_volatility_risk_multiplier":  [1.00, 1.20, 1.40, 1.60, 1.80, 2.00],
        "max_open_positions":                 [1, 2, 3, 4, 5],
        "max_total_exposure_pct":             [0.30, 0.40, 0.50, 0.60, 0.70],
        "max_risk_fraction":                  [0.40, 0.50, 0.60, 0.70],
        "stop_loss_atr_multiple":             [0.7, 0.8, 1.0, 1.2, 1.5],
        "take_profit_rr":                     [6.0, 7.0, 8.0, 10.0, 12.0, 15.0],
        "sideway_sl_atr_multiple":            [0.6, 0.8, 1.0],
        "volatile_sl_atr_multiple":           [1.5, 2.0, 2.5, 3.0, 3.5],
        "partial_tp_enabled":                 [True, False],
        "partial_tp_rr":                      [1.0, 1.5, 2.0, 3.0],
        "partial_tp_pct":                     [0.25, 0.35, 0.50],
        "anti_martingale_factor":             [0.25, 0.35, 0.45, 0.55],
        "consecutive_loss_pause_count":       [2, 3],
        "consecutive_loss_cooldown_bars":     [16, 20, 24, 32, 48],
        "daily_loss_limit_pct":              [0.008, 0.010, 0.015, 0.020],
        "compound_cap":                       [0.0],
        "spread_cost_rr":                     [0.0, 0.01],
        "slippage_rr":                        [0.01, 0.02],
        "commission_rr":                      [0.0],
    }


def _sample_candidates(
    space: dict[str, list[Any]],
    n: int,
    rng: random.Random,
    warm: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    dims = list(space.keys())
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    if warm:
        for c in warm:
            key = tuple(c.get(d, space[d][0]) for d in dims)
            if key in seen:
                continue
            seen.add(key)
            out.append({d: c.get(d, rng.choice(space[d])) for d in dims})
    while len(out) < n:
        c = {d: rng.choice(space[d]) for d in dims}
        key = tuple(c[d] for d in dims)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _mutate(
    base: dict[str, Any],
    space: dict[str, list[Any]],
    rng: random.Random,
    n_edits: int = 2,
) -> dict[str, Any]:
    c = dict(base)
    dims = list(space.keys())
    picks = rng.sample(dims, k=min(n_edits, len(dims)))
    for dim in picks:
        vals = space[dim]
        cur = c.get(dim, vals[0])
        idx = vals.index(cur) if cur in vals else 0
        neigh = [idx]
        if idx > 0:
            neigh.append(idx - 1)
        if idx < len(vals) - 1:
            neigh.append(idx + 1)
        c[dim] = vals[rng.choice(neigh)]
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Run one search round
# ─────────────────────────────────────────────────────────────────────────────

def _run_round(
    round_num: int,
    label: str,
    base_settings: Settings,
    fold_cache: list[FoldCache],
    templates: dict[str, dict[str, Any]],
    space: dict[str, list[Any]],
    n_explore: int,
    n_refine: int,
    initial_balance: float,
    pf_min: float,
    net_min: float,
    dd_max: float,
    out_prefix: str,
    seed: int,
    warm_seeds: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns: (all_rows, feasible_rows).
    """
    rng = random.Random(seed)
    tpl_names = list(templates.keys())

    print(f"\n{'='*80}", flush=True)
    print(f"  ROUND {round_num} — {label}", flush=True)
    print(f"  Constraints: PF>={pf_min:.1f}  net>=${net_min:,.0f}  DD<{dd_max:.2f}%", flush=True)
    print(f"  Explore={n_explore}  Refine={n_refine}  balance=${initial_balance}", flush=True)
    print(f"{'='*80}", flush=True)

    t0 = time.time()
    candidates = _sample_candidates(space, n_explore, rng, warm=warm_seeds)

    all_rows: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    best_distance = float("inf")
    best_pf = 0.0
    best_net = 0.0
    best_dd_feasible = float("inf")  # best DD among PF-feasible candidates

    def _ev(c: dict[str, Any]) -> dict[str, Any]:
        return _evaluate(base_settings, fold_cache, c, templates, initial_balance, pf_min, net_min, dd_max)

    print(f"[R{round_num}] Exploring {n_explore} candidates …", flush=True)
    for i, c in enumerate(candidates, start=1):
        row = _ev(c)
        all_rows.append(row)
        if row["feasible"]:
            feasible.append(row)
        if row["distance_score"] < best_distance:
            best_distance = row["distance_score"]
        best_pf  = max(best_pf,  row["profit_factor_global"])
        best_net = max(best_net, row["sum_net_profit"])
        if row["profit_factor_global"] >= pf_min:
            best_dd_feasible = min(best_dd_feasible, row["global_max_drawdown_pct"])

        if i % 100 == 0 or i == n_explore:
            elapsed = time.time() - t0
            tr = n_explore + n_refine
            eta = elapsed / i * (tr - i) if i > 0 else 0
            print(
                f"[R{round_num}] {i:>{len(str(n_explore))}}/{n_explore}  "
                f"feasible={len(feasible)}  best_dist={best_distance:,.0f}  "
                f"best_PF={best_pf:.3f}  best_net=${best_net:,.0f}  "
                f"elapsed={elapsed:.0f}s  ETA≈{eta:.0f}s",
                flush=True,
            )

    # ── Refine around top-k ────────────────────────────────────────────────
    if n_refine > 0 and all_rows:
        print(f"[R{round_num}] Refining around top-{min(20, len(all_rows))} candidates …", flush=True)
        sorted_rows = sorted(all_rows, key=lambda r: r["distance_score"])
        seeds_for_refine = sorted_rows[:20]
        refine_cands: list[dict[str, Any]] = []
        seen_keys: set[tuple] = {tuple(r.get(d, "") for d in space) for r in all_rows}
        while len(refine_cands) < n_refine:
            seed_row = rng.choice(seeds_for_refine)
            n_ed = 1 if rng.random() < 0.6 else 2
            mut = _mutate(seed_row, space, rng, n_edits=n_ed)
            key = tuple(mut.get(d, "") for d in space)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            refine_cands.append(mut)

        for i, c in enumerate(refine_cands, start=1):
            row = _ev(c)
            all_rows.append(row)
            if row["feasible"]:
                feasible.append(row)
            if row["distance_score"] < best_distance:
                best_distance = row["distance_score"]
            best_pf  = max(best_pf,  row["profit_factor_global"])
            best_net = max(best_net, row["sum_net_profit"])
            if row["profit_factor_global"] >= pf_min:
                best_dd_feasible = min(best_dd_feasible, row["global_max_drawdown_pct"])

            if i % 50 == 0 or i == n_refine:
                elapsed = time.time() - t0
                done = n_explore + i
                total = n_explore + n_refine
                print(
                    f"[R{round_num}] refine {i}/{n_refine}  "
                    f"feasible={len(feasible)}  best_dist={best_distance:,.0f}  "
                    f"best_PF={best_pf:.3f}  best_net=${best_net:,.0f}  "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

    elapsed_total = time.time() - t0
    print(
        f"\n[R{round_num}] DONE  "
        f"evaluated={len(all_rows)}  feasible={len(feasible)}  "
        f"elapsed={elapsed_total:.1f}s",
        flush=True,
    )
    print(
        f"[R{round_num}]  best_PF={best_pf:.4f}  best_net=${best_net:,.2f}  "
        f"best_DD_when_PF_ok={best_dd_feasible:.2f}%  best_dist={best_distance:,.0f}",
        flush=True,
    )

    # ── Save round outputs ─────────────────────────────────────────────────
    csv_path = Path(f"outputs/{out_prefix}_round{round_num}.csv")
    pd.DataFrame(all_rows).sort_values("distance_score").to_csv(csv_path, index=False)

    report = {
        "round": round_num,
        "label": label,
        "constraints": {"pf_min": pf_min, "net_min": net_min, "dd_max": dd_max},
        "evaluated": len(all_rows),
        "feasible_count": len(feasible),
        "elapsed_sec": round(elapsed_total, 1),
        "best_distance": round(best_distance, 2),
        "best_pf_seen": round(best_pf, 4),
        "best_net_seen": round(best_net, 2),
        "best_dd_when_pf_ok": round(best_dd_feasible, 2) if best_dd_feasible < float("inf") else None,
        "best_feasible": sorted(feasible, key=lambda r: r["sum_net_profit"], reverse=True)[0]
                         if feasible else None,
        "best_by_distance": sorted(all_rows, key=lambda r: r["distance_score"])[0],
        "best_by_pf":       sorted(all_rows, key=lambda r: -r["profit_factor_global"])[0],
        "best_by_net":      sorted(all_rows, key=lambda r: -r["sum_net_profit"])[0],
        "best_by_dd":       sorted(
            [r for r in all_rows if r["profit_factor_global"] >= pf_min],
            key=lambda r: r["global_max_drawdown_pct"],
        )[0] if any(r["profit_factor_global"] >= pf_min for r in all_rows) else None,
        "csv": str(csv_path),
    }
    rpt_path = Path(f"outputs/{out_prefix}_round{round_num}_report.json")
    rpt_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[R{round_num}]  Saved: {csv_path}  {rpt_path}", flush=True)

    return all_rows, feasible


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PF>=3 joint constraint search: auto-expands until feasible found."
    )
    p.add_argument("--config",          type=Path,  required=True,
                   help="Benchmark YAML config (e.g. configs/benchmarks/acc1_expand_net127313_dd3215.yaml)")
    p.add_argument("--old-net",         type=float, required=True,
                   help="Minimum sum_net_profit target (e.g. 127313.86)")
    p.add_argument("--old-dd",          type=float, required=True,
                   help="Maximum global DD pct, exclusive (e.g. 32.15)")
    p.add_argument("--pf-min",          type=float, default=3.0,
                   help="Minimum profit factor (default 3.0)")
    p.add_argument("--initial-balance", type=float, default=200.0)
    p.add_argument("--seed",            type=int,   default=20260405)
    p.add_argument("--explore",         type=int,   default=2000,
                   help="Explore candidates per round (default 2000)")
    p.add_argument("--refine",          type=int,   default=1000,
                   help="Refine candidates per round (default 1000)")
    p.add_argument("--out-prefix",      type=str,   default="wf_pf3_joint")
    p.add_argument("--max-rounds",      type=int,   default=5,
                   help="Maximum rounds before relaxing net floor.")
    p.add_argument("--net-relax-pct",   type=float, default=0.80,
                   help="Multiply net_min by this factor if no feasible after max-rounds (default 0.80).")
    p.add_argument("--net-floor-pct",   type=float, default=0.20,
                   help="Stop relaxing when net_min < original * this factor (default 0.20).")
    return p.parse_args()


def _print_banner(r: dict, label: str) -> None:
    print(
        f"\n  ★ {label}: PF={r['profit_factor_global']:.4f}  "
        f"net=${r['sum_net_profit']:,.2f}  "
        f"DD={r['global_max_drawdown_pct']:.2f}%  "
        f"trades={r['total_trades']}  WR={r['win_rate']:.2%}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    t_global = time.time()

    print("=" * 80, flush=True)
    print("  WF PF>=3 JOINT SEARCH — Progressive Multi-Round", flush=True)
    print(f"  config    : {args.config}", flush=True)
    print(f"  PF >= {args.pf_min:.1f}  |  net >= ${args.old_net:,.2f}  |  DD < {args.old_dd:.2f}%", flush=True)
    print(f"  balance   : ${args.initial_balance}  seed={args.seed}", flush=True)
    print(f"  explore   : {args.explore}/round  refine: {args.refine}/round  max_rounds={args.max_rounds}", flush=True)
    print("=" * 80, flush=True)

    settings  = load_settings(args.config)
    templates = _time_templates()
    tpl_names = list(templates.keys())

    # Build fold cache once
    fold_cache, _ = _build_fold_cache(settings)

    spaces = [
        ("Round-1 Explosive Compound / High TP",       _space_round1(tpl_names)),
        ("Round-2 Wider Risk + Moderate Compound",      _space_round2(tpl_names)),
        ("Round-3 Ultra-Aggressive / Extreme TP",       _space_round3(tpl_names)),
    ]
    # Pad with repeats of round-3 space if max_rounds > 3
    while len(spaces) < args.max_rounds:
        spaces.append((f"Round-{len(spaces)+1} Extended Ultra-Aggressive", _space_round3(tpl_names)))

    all_feasible: list[dict[str, Any]] = []
    all_best_rows: list[dict[str, Any]] = []
    net_min_current = float(args.old_net)
    net_min_floor   = float(args.old_net) * args.net_floor_pct
    relax_multiplier = float(args.net_relax_pct)
    round_idx = 0

    # Track global Pareto best across rounds
    pareto_best_pf  = {"profit_factor_global": 0.0}
    pareto_best_net = {"sum_net_profit": 0.0}
    pareto_best_dd  = {"global_max_drawdown_pct": float("inf")}  # among PF>=pf_min

    while True:
        if round_idx >= len(spaces):
            # Exhausted all planned spaces → relax net and retry from round 2
            if net_min_current * relax_multiplier < net_min_floor:
                print(
                    f"\n[STOP] No feasible found even after relaxing net floor to ${net_min_floor:,.0f}. "
                    f"Reporting Pareto best.",
                    flush=True,
                )
                break
            net_min_current = net_min_current * relax_multiplier
            print(
                f"\n[RELAX] net_min relaxed to ${net_min_current:,.0f} "
                f"(floor=${net_min_floor:,.0f})",
                flush=True,
            )
            round_idx = 2  # restart from round-2 space (already explored round-1)
            spaces = spaces[:3]  # trim back to 3 entries to avoid infinite growth
            # Add new round entry
            spaces.append((
                f"Relaxed-Round net>=${net_min_current/1000:.0f}k",
                _space_round3(tpl_names),
            ))

        space_label, space = spaces[round_idx]
        warm = all_best_rows[-20:] if all_best_rows else None

        rows, feasible = _run_round(
            round_num      = round_idx + 1,
            label          = space_label,
            base_settings  = settings,
            fold_cache     = fold_cache,
            templates      = templates,
            space          = space,
            n_explore      = args.explore,
            n_refine       = args.refine,
            initial_balance= args.initial_balance,
            pf_min         = args.pf_min,
            net_min        = net_min_current,
            dd_max         = args.old_dd,
            out_prefix     = args.out_prefix,
            seed           = args.seed + round_idx * 1000,
            warm_seeds     = warm,
        )

        all_feasible.extend(feasible)
        all_best_rows.extend(sorted(rows, key=lambda r: r["distance_score"])[:30])

        # Update Pareto trackers
        for r in rows:
            if r["profit_factor_global"] > pareto_best_pf["profit_factor_global"]:
                pareto_best_pf = r
            if r["sum_net_profit"] > pareto_best_net["sum_net_profit"]:
                pareto_best_net = r
            if (r["profit_factor_global"] >= args.pf_min and
                    r["global_max_drawdown_pct"] < pareto_best_dd["global_max_drawdown_pct"]):
                pareto_best_dd = r

        if all_feasible:
            best = max(all_feasible, key=lambda r: r["sum_net_profit"])
            print(f"\n{'='*80}", flush=True)
            print(f"  ✅  FEASIBLE FOUND — PF>={args.pf_min:.1f}, net>=${net_min_current:,.0f}, DD<{args.old_dd:.2f}%", flush=True)
            _print_banner(best, "Best feasible by net")
            bpath = Path(f"outputs/{args.out_prefix}_best_feasible.json")
            bpath.write_text(json.dumps({
                "constraints": {
                    "pf_min": args.pf_min,
                    "net_min_original": args.old_net,
                    "net_min_used": net_min_current,
                    "dd_max": args.old_dd,
                },
                "total_feasible": len(all_feasible),
                "best_by_net": best,
                "top5": sorted(all_feasible, key=lambda r: -r["sum_net_profit"])[:5],
            }, indent=2, default=str), encoding="utf-8")
            print(f"  Saved: {bpath}", flush=True)
            print(f"{'='*80}\n", flush=True)
            # Keep searching more rounds if more unfound space
            round_idx += 1
            # Only stop if all planned rounds done and using original constraints
            if round_idx >= len(spaces) and abs(net_min_current - args.old_net) < 1.0:
                break
            elif round_idx >= len(spaces):
                break
        else:
            round_idx += 1

    # ── Final summary ───────────────────────────────────────────────────────
    elapsed_total = time.time() - t_global
    print(f"\n{'='*80}", flush=True)
    print(f"  SEARCH COMPLETE  elapsed={elapsed_total/60:.1f}min", flush=True)
    print(f"  Total feasible (all rounds): {len(all_feasible)}", flush=True)
    if all_feasible:
        best_final = max(all_feasible, key=lambda r: r["sum_net_profit"])
        print(f"  Best feasible (max net):", flush=True)
        _print_banner(best_final, "  →")
    else:
        print(f"  ⚠  No joint-feasible found. Reporting Pareto best:", flush=True)
        _print_banner(pareto_best_pf,  "  Best PF")
        _print_banner(pareto_best_net, "  Best Net")
        if pareto_best_dd["profit_factor_global"] >= args.pf_min:
            _print_banner(pareto_best_dd, "  Best DD (when PF ok)")

    # Save final Pareto report
    pareto_path = Path(f"outputs/{args.out_prefix}_pareto_best.json")
    pareto_path.write_text(json.dumps({
        "search_config": {
            "config": str(args.config),
            "pf_min": args.pf_min,
            "net_min_original": args.old_net,
            "dd_max": args.old_dd,
            "initial_balance": args.initial_balance,
        },
        "total_feasible": len(all_feasible),
        "pareto_best_pf":  pareto_best_pf,
        "pareto_best_net": pareto_best_net,
        "pareto_best_dd_when_pf_ok": pareto_best_dd
                                     if pareto_best_dd["profit_factor_global"] >= args.pf_min
                                     else None,
        "feasible_top5": sorted(all_feasible, key=lambda r: -r["sum_net_profit"])[:5]
                         if all_feasible else [],
    }, indent=2, default=str), encoding="utf-8")
    print(f"  Pareto report: {pareto_path}", flush=True)
    print(f"{'='*80}\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
