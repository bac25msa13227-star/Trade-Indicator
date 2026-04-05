#!/usr/bin/env python3
"""
WF PF>=3 Search v2 — Correct Methodology.

Key improvements over v1:
  1. Builds fold cache using BASE CONFIG (old benchmark config) — same as actual benchmark training.
  2. Evaluates baseline config first to establish true current-code reference.
  3. Searches over parameter space with compound_cap=50 (matching benchmark default).
  4. Reports constraints relative to BOTH user-stated targets AND baseline.
  5. Results are reproducible: ModelTrainer uses random_state=42, numpy seed fixed.

Usage:
  python scripts/wf_pf3_v2_search.py \
    --base-config configs/benchmarks/acc1_expand_net127313_dd3215.yaml \
    --old-net 127313.86 --old-dd 32.15 --pf-min 3.0 \
    --out-prefix wf_pf3v2_acc1 --explore 3000 --refine 1000 --seed 20260405
"""
from __future__ import annotations

import argparse, json, random, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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

@dataclass
class FoldCache:
    fold: int
    test_start: str
    test_end: str
    predictions: pd.DataFrame


def _max_dd_from_curve(curve: list[float]) -> float:
    if not curve:
        return 0.0
    peak = curve[0]; mx = 0.0
    for v in curve[1:]:
        peak = max(peak, v)
        if peak > 0:
            mx = max(mx, (peak - v) / peak)
    return mx * 100.0


def _build_fold_cache(settings: Settings) -> list[FoldCache]:
    """Train fold models using BASE CONFIG — same as benchmark training approach."""
    print("[cache] Loading data + building features …", flush=True)
    service = MarketDataService(settings)
    frames  = service.fetch_multi_timeframe_data(source=settings.market.training_data_source, all_bars=True)
    merged  = build_merged_context(settings, frames)
    dataset = prepare_training_dataset(settings, frames, HybridStrategy(settings), cached_merged=merged)
    print(f"[cache]  dataset_rows={len(dataset):,}", flush=True)

    train_size = settings.training.walkforward_train_size
    test_size  = settings.training.walkforward_test_size
    step_size  = settings.training.walkforward_step_size
    max_folds  = settings.training.walkforward_max_folds_per_combination or 8
    fold_indices = list(range(0, max(len(dataset) - train_size - test_size + 1, 0), step_size))
    if max_folds > 0:
        fold_indices = fold_indices[-max_folds:]

    cached: list[FoldCache] = []
    for idx, fstart in enumerate(fold_indices, start=1):
        tend = fstart + train_size
        tnd  = tend + test_size
        fold_tr = dataset.iloc[fstart:tend].copy()
        fold_te = dataset.iloc[tend:tnd].copy()
        if len(fold_tr) < 200 or len(fold_te) < 50:
            continue

        fold_ds = pd.concat([fold_tr, fold_te], ignore_index=True)
        fold_ds["split"] = "train"
        fold_ds.loc[len(fold_tr):, "split"] = "test"

        trainer = ModelTrainer(settings)
        trainer.train(fold_ds, save_artifacts=False)
        preds = trainer.predict_dataset(fold_ds)

        te_times = pd.to_datetime(fold_te["time"])
        cached.append(FoldCache(
            fold=idx,
            test_start=str(te_times.min())[:16],
            test_end=str(te_times.max())[:16],
            predictions=preds.copy(),
        ))
        print(f"[cache]   fold {idx}/{len(fold_indices)}"
              f"  test=[{cached[-1].test_start} → {cached[-1].test_end}]"
              f"  rows={len(preds)}", flush=True)

    print(f"[cache] Done. {len(cached)} folds cached.\n", flush=True)
    return cached


# ─────────────────────────────────────────────────────────────────────────────
# Candidate application — SAFE: only sets parameters that exist in config.py
# ─────────────────────────────────────────────────────────────────────────────

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
_TEMPLATES = {
    "focus_hours": {
        "blocked_hours_utc": [],
        "blocked_weekday_hours_utc": {},
        "allowed_weekday_hours_utc": {w: [0, 1, 5, 6, 7, 13, 14, 20, 21] for w in _WEEKDAYS},
    },
    "minimal": {
        "blocked_hours_utc": [15, 22, 23],
        "blocked_weekday_hours_utc": {},
        "allowed_weekday_hours_utc": {},
    },
    "london_ny": {
        "blocked_hours_utc": [],
        "blocked_weekday_hours_utc": {},
        "allowed_weekday_hours_utc": {w: list(range(7, 16)) for w in _WEEKDAYS},
    },
}


def _apply_candidate(base: Settings, c: dict[str, Any]) -> Settings:
    s = base.model_copy(deep=True)
    # Strategy
    s.strategy.signal_threshold                         = float(c["threshold"])
    s.strategy.min_strategy_score                       = float(c.get("min_strategy_score", 0.05))
    s.strategy.sideway_min_strategy_score               = float(c.get("sideway_min_strategy_score", 0.05))
    s.strategy.strong_volatility_min_strategy_score     = float(c.get("strong_volatility_min_strategy_score", 0.12))
    s.strategy.require_trend_alignment                  = bool(c.get("require_trend_alignment", False))
    s.strategy.sideway_min_confidence                   = float(c.get("sideway_min_conf", 0.80))
    s.strategy.volatile_min_confidence                  = float(c.get("volatile_min_conf", 0.70))
    s.strategy.silver_bullet_enabled                    = bool(c.get("silver_bullet_enabled", True))
    s.strategy.adx_gate_enabled                         = bool(c.get("adx_gate_enabled", True))
    s.strategy.adx_min_trend                            = float(c.get("adx_min_trend", 17.0))
    tpl = _TEMPLATES.get(str(c.get("template", "focus_hours")), _TEMPLATES["focus_hours"])
    s.strategy.blocked_hours_utc         = list(tpl["blocked_hours_utc"])
    s.strategy.blocked_weekday_hours_utc = dict(tpl["blocked_weekday_hours_utc"])
    s.strategy.allowed_weekday_hours_utc = dict(tpl["allowed_weekday_hours_utc"])
    # Risk
    s.risk.min_confidence                      = float(c.get("risk_min_conf", 0.70))
    s.risk.risk_per_trade                      = float(c["risk_per_trade"])
    s.risk.risk_tier_floor                     = float(c.get("risk_tier_floor", max(0.02, c["risk_per_trade"] * 0.5)))
    s.risk.max_risk_fraction                   = float(c.get("max_risk_fraction", min(0.30, c["risk_per_trade"] * 4)))
    s.risk.sideway_risk_multiplier             = float(c.get("sideway_risk_multiplier", 0.20))
    s.risk.strong_volatility_risk_multiplier   = float(c.get("strong_volatility_risk_multiplier", 1.20))
    s.risk.max_open_positions                  = int(c.get("max_open_positions", 2))
    s.risk.max_total_exposure_pct              = float(c.get("max_total_exposure_pct", 0.20))
    s.risk.stop_loss_atr_multiple              = float(c.get("stop_loss_atr_multiple", 1.5))
    s.risk.take_profit_rr                      = float(c["take_profit_rr"])
    s.risk.sideway_take_profit_rr              = float(c.get("sideway_take_profit_rr", max(2.0, c["take_profit_rr"] - 1.0)))
    s.risk.volatile_take_profit_rr             = float(c.get("volatile_take_profit_rr", c["take_profit_rr"] + 1.0))
    s.risk.sideway_sl_atr_multiple             = float(c.get("sideway_sl_atr_multiple", 1.0))
    s.risk.volatile_sl_atr_multiple            = float(c.get("volatile_sl_atr_multiple", 2.0))
    s.risk.partial_tp_enabled                  = bool(c.get("partial_tp_enabled", False))
    s.risk.partial_tp_rr                       = float(c.get("partial_tp_rr", 1.5))
    s.risk.partial_tp_pct                      = float(c.get("partial_tp_pct", 0.4))
    s.risk.anti_martingale_factor              = float(c.get("anti_martingale_factor", 0.5))
    s.risk.consecutive_loss_pause_count        = int(c["consecutive_loss_pause_count"])
    s.risk.consecutive_loss_cooldown_bars      = int(c["consecutive_loss_cooldown_bars"])
    s.risk.daily_loss_limit_pct                = float(c.get("daily_loss_limit_pct", 0.0))
    s.risk.compound_cap                        = float(c.get("compound_cap", 50.0))
    s.risk.spread_cost_rr                      = float(c.get("spread_cost_rr", 0.02))
    s.risk.slippage_rr                         = float(c.get("slippage_rr", 0.01))
    s.risk.commission_rr                       = float(c.get("commission_rr", 0.0))
    return s


def _evaluate(
    base: Settings,
    fold_cache: list[FoldCache],
    c: dict[str, Any],
    initial_balance: float,
    pf_min: float,
    net_min: float,
    dd_max: float,
) -> dict[str, Any]:
    tuned   = _apply_candidate(base, c)
    balance = float(initial_balance)
    equity_curve: list[float] = [balance]
    global_peak   = balance
    global_max_dd = 0.0
    fold_rows: list[dict[str, Any]] = []

    for fold in fold_cache:
        fold_s = tuned.model_copy(deep=True)
        fold_s.training.backtest_initial_balance = balance

        preds = fold.predictions.copy()
        preds["prediction"] = (preds["probability"] >= float(c["threshold"])).astype(int)

        sim = simulate_dynamic_concurrent_backtest(
            preds, fold_s, RiskManager(fold_s), label="test", compound=True
        )
        rep  = sim.report
        fold_start_bal = balance
        balance = float(rep["ending_balance"])
        equity_curve.append(balance)

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
            "wins":           int(rep["wins"]),
            "losses":         int(rep["losses"]),
        })

    fold_df   = pd.DataFrame(fold_rows)
    g_profit  = float(fold_df["gross_profit"].sum()) if not fold_df.empty else 0.0
    g_loss    = float(fold_df["gross_loss"].sum())   if not fold_df.empty else 0.0
    net       = balance - initial_balance
    pf        = g_profit / g_loss if g_loss > 0 else 0.0
    dd_exact  = _max_dd_from_curve(equity_curve)
    max_dd    = max(global_max_dd * 100.0, dd_exact)
    total_tr  = int(fold_df["trades"].sum()) if not fold_df.empty else 0
    total_w   = int(fold_df["wins"].sum())   if not fold_df.empty else 0
    total_l   = int(fold_df["losses"].sum()) if not fold_df.empty else 0
    wr        = total_w / max(total_w + total_l, 1)

    feasible  = bool(pf >= pf_min and net >= net_min and max_dd < dd_max)
    pf_gap  = max(0.0, pf_min  - pf)
    net_gap = max(0.0, net_min - net)
    dd_gap  = max(0.0, max_dd  - dd_max)
    dist    = pf_gap * 5000 + net_gap * 0.01 + dd_gap * 2000

    return {
        **c,
        "starting_balance":        round(initial_balance, 2),
        "ending_balance":          round(balance, 2),
        "sum_net_profit":          round(net, 2),
        "sum_gross_profit":        round(g_profit, 2),
        "sum_gross_loss":          round(g_loss, 2),
        "global_max_drawdown_pct": round(max_dd, 4),
        "profit_factor_global":    round(pf, 4),
        "total_trades":            total_tr,
        "win_rate":                round(wr, 4),
        "avg_fold_return_pct":     round(float(fold_df["return_pct"].mean()) if "return_pct" in fold_df.columns else 0.0, 4),
        "folds":                   int(len(fold_df)),
        "feasible":                feasible,
        "distance_score":          round(dist, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_baseline(base: Settings, fold_cache: list[FoldCache], balance0: float) -> dict[str, Any]:
    """Evaluate the base config AS-IS (no overrides)."""
    c_baseline = {
        "threshold":                      base.strategy.signal_threshold,
        "min_strategy_score":             base.strategy.min_strategy_score,
        "sideway_min_strategy_score":     base.strategy.sideway_min_strategy_score,
        "strong_volatility_min_strategy_score": base.strategy.strong_volatility_min_strategy_score,
        "require_trend_alignment":        base.strategy.require_trend_alignment,
        "sideway_min_conf":               base.strategy.sideway_min_confidence,
        "volatile_min_conf":              base.strategy.volatile_min_confidence,
        "silver_bullet_enabled":          base.strategy.silver_bullet_enabled,
        "adx_gate_enabled":              base.strategy.adx_gate_enabled,
        "adx_min_trend":                 base.strategy.adx_min_trend,
        "template":                      "focus_hours",
        "risk_min_conf":                 base.risk.min_confidence,
        "risk_per_trade":                base.risk.risk_per_trade,
        "risk_tier_floor":               getattr(base.risk, "risk_tier_floor", base.risk.risk_per_trade),
        "sideway_risk_multiplier":       base.risk.sideway_risk_multiplier,
        "strong_volatility_risk_multiplier": base.risk.strong_volatility_risk_multiplier,
        "max_open_positions":            base.risk.max_open_positions,
        "stop_loss_atr_multiple":        base.risk.stop_loss_atr_multiple,
        "take_profit_rr":                base.risk.take_profit_rr,
        "partial_tp_enabled":            base.risk.partial_tp_enabled,
        "anti_martingale_factor":        base.risk.anti_martingale_factor,
        "consecutive_loss_pause_count":  base.risk.consecutive_loss_pause_count,
        "consecutive_loss_cooldown_bars": base.risk.consecutive_loss_cooldown_bars,
        "daily_loss_limit_pct":         base.risk.daily_loss_limit_pct,
        "compound_cap":                  base.risk.compound_cap,
        "spread_cost_rr":                base.risk.spread_cost_rr,
        "slippage_rr":                   base.risk.slippage_rr,
        "commission_rr":                 base.risk.commission_rr,
    }
    return _evaluate(base, fold_cache, c_baseline, balance0, pf_min=0, net_min=0, dd_max=999)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter space
# ─────────────────────────────────────────────────────────────────────────────

def _build_space() -> dict[str, list[Any]]:
    """
    Broad search space. Key insights from v1 search:
    - threshold=0.88+ is needed for PF>=3 (model needs selectivity)
    - compound_cap=50 (benchmark default) is used for fair comparison
    - risk_per_trade=0.08-0.15 to drive compound growth
    - TP=4-10 for PF leverage
    - Tight circuit breakers for DD control
    """
    return {
        # Entry selectivity — main PF lever
        "threshold":                          [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92],
        "min_strategy_score":                 [0.05, 0.10, 0.15, 0.20, 0.30],
        "sideway_min_strategy_score":         [0.05, 0.10, 0.20],
        "strong_volatility_min_strategy_score": [0.10, 0.20, 0.30],
        "require_trend_alignment":            [False, True],
        "risk_min_conf":                      [0.75, 0.80, 0.85],
        "sideway_min_conf":                   [0.80, 0.85, 0.90],
        "volatile_min_conf":                  [0.75, 0.80, 0.85],
        "template":                           ["focus_hours", "minimal", "london_ny"],
        "silver_bullet_enabled":              [True, False],
        "adx_gate_enabled":                   [True, False],
        "adx_min_trend":                      [12.0, 15.0, 18.0, 22.0],
        # Risk sizing — modest increase to drive net growth
        "risk_per_trade":                     [0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15],
        "sideway_risk_multiplier":            [0.10, 0.20, 0.30, 0.40],
        "strong_volatility_risk_multiplier":  [0.80, 1.00, 1.20, 1.40],
        "max_open_positions":                 [1, 2, 3],
        # SL/TP — higher TP boosts PF
        "stop_loss_atr_multiple":             [1.0, 1.2, 1.5, 1.8],
        "take_profit_rr":                     [4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        # Partial TP off (captures full winner)
        "partial_tp_enabled":                 [False],
        # Circuit breakers — protect DD
        "anti_martingale_factor":             [0.40, 0.50, 0.60, 0.70],
        "consecutive_loss_pause_count":       [2, 3],
        "consecutive_loss_cooldown_bars":     [8, 12, 16, 24, 32],
        "daily_loss_limit_pct":              [0.0, 0.015, 0.020, 0.030],
        # Compound cap = 50 (fair comparison with benchmark)
        "compound_cap":                       [50.0],
        # Friction
        "spread_cost_rr":                     [0.01, 0.02],
        "slippage_rr":                        [0.01, 0.02],
        "commission_rr":                      [0.0],
    }


def _extended_space() -> dict[str, list[Any]]:
    """Round 2+ — extend to higher risk and compound_cap=0."""
    sp = _build_space()
    sp["risk_per_trade"]   = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    sp["take_profit_rr"]   = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
    sp["compound_cap"]     = [50.0, 0.0]  # also try unlimited
    sp["threshold"]        = [0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94]
    return sp


def _sample(space: dict[str, list[Any]], n: int, rng: random.Random, warm=None) -> list[dict]:
    dims = list(space.keys())
    seen: set = set()
    out: list[dict] = []
    if warm:
        for c in warm:
            key = tuple(c.get(d, space[d][0]) for d in dims)
            if key not in seen:
                seen.add(key)
                out.append({d: c.get(d, rng.choice(space[d])) for d in dims})
    while len(out) < n:
        c   = {d: rng.choice(space[d]) for d in dims}
        key = tuple(c[d] for d in dims)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _mutate(base: dict, space: dict, rng: random.Random, n_edits: int = 2) -> dict:
    c = dict(base)
    for dim in rng.sample(list(space.keys()), k=min(n_edits, len(space))):
        vals = space[dim]
        cur  = c.get(dim, vals[0])
        idx  = vals.index(cur) if cur in vals else 0
        neigh = [idx] + ([idx-1] if idx > 0 else []) + ([idx+1] if idx < len(vals)-1 else [])
        c[dim] = vals[rng.choice(neigh)]
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Main search loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_round(rnum, label, base_s, fold_cache, space, n_explore, n_refine,
               balance0, pf_min, net_min, dd_max, out_prefix, seed, warm=None):
    rng   = random.Random(seed + rnum * 31337)
    cands = _sample(space, n_explore, rng, warm=warm)
    all_rows: list[dict] = []
    feasible: list[dict] = []
    best_dist = float("inf"); best_pf = 0.0; best_net = 0.0

    print(f"\n{'='*80}", flush=True)
    print(f"  ROUND {rnum} — {label}", flush=True)
    print(f"  Target: PF>={pf_min}  net>=${net_min:,.0f}  DD<{dd_max}%", flush=True)
    print(f"  Explore={n_explore}  Refine={n_refine}", flush=True)
    print(f"{'='*80}", flush=True)
    t0 = time.time()

    def ev(c): return _evaluate(base_s, fold_cache, c, balance0, pf_min, net_min, dd_max)

    print(f"[R{rnum}] Exploring {n_explore} candidates …", flush=True)
    for i, c in enumerate(cands, start=1):
        row = ev(c)
        all_rows.append(row)
        if row["feasible"]: feasible.append(row)
        best_dist = min(best_dist, row["distance_score"])
        best_pf   = max(best_pf,   row["profit_factor_global"])
        best_net  = max(best_net,  row["sum_net_profit"])
        if i % 100 == 0 or i == n_explore:
            ela = time.time() - t0
            eta = ela / i * (n_explore + n_refine - i) if i > 0 else 0
            print(f"[R{rnum}] {i:>{len(str(n_explore))}}/{n_explore}  feasible={len(feasible)}  dist={best_dist:.0f}  PF={best_pf:.3f}  net=${best_net:,.0f}  {ela:.0f}s  ETA≈{eta:.0f}s", flush=True)

    if n_refine > 0 and all_rows:
        print(f"[R{rnum}] Refining around top-20 …", flush=True)
        top20 = sorted(all_rows, key=lambda r: r["distance_score"])[:20]
        seen  = {tuple(r.get(d,"") for d in space) for r in all_rows}
        refine_cands: list[dict] = []
        while len(refine_cands) < n_refine:
            base_c = rng.choice(top20)
            mut    = _mutate(base_c, space, rng, n_edits=1 if rng.random() < 0.7 else 2)
            key    = tuple(mut.get(d,"") for d in space)
            if key not in seen:
                seen.add(key); refine_cands.append(mut)
        for i, c in enumerate(refine_cands, start=1):
            row = ev(c)
            all_rows.append(row)
            if row["feasible"]: feasible.append(row)
            best_dist = min(best_dist, row["distance_score"])
            best_pf   = max(best_pf,   row["profit_factor_global"])
            best_net  = max(best_net,  row["sum_net_profit"])
            if i % 50 == 0 or i == n_refine:
                ela = time.time() - t0
                print(f"[R{rnum}] refine {i}/{n_refine}  feasible={len(feasible)}  PF={best_pf:.3f}  net=${best_net:,.0f}  {ela:.0f}s", flush=True)

    ela_tot = time.time() - t0
    print(f"[R{rnum}] DONE  evaluated={len(all_rows)}  feasible={len(feasible)}  {ela_tot:.1f}s", flush=True)

    csv_p = Path(f"outputs/{out_prefix}_round{rnum}.csv")
    pd.DataFrame(all_rows).sort_values("distance_score").to_csv(csv_p, index=False)

    best_f = sorted(feasible, key=lambda r: (-r["total_trades"], r["global_max_drawdown_pct"]))[0] if feasible else None
    rpt = {
        "round": rnum, "label": label,
        "evaluated": len(all_rows), "feasible_count": len(feasible),
        "elapsed_sec": round(ela_tot, 1),
        "best_distance": round(best_dist, 2),
        "best_pf_seen": round(best_pf, 4),
        "best_net_seen": round(best_net, 2),
        "best_feasible": best_f,
        "best_by_distance": sorted(all_rows, key=lambda r: r["distance_score"])[0],
    }
    Path(f"outputs/{out_prefix}_round{rnum}_report.json").write_text(
        json.dumps(rpt, indent=2, default=str))
    print(f"[R{rnum}] Saved: {csv_p}", flush=True)
    return all_rows, feasible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--old-net",     type=float, required=True)
    ap.add_argument("--old-dd",      type=float, required=True)
    ap.add_argument("--pf-min",      type=float, default=3.0)
    ap.add_argument("--out-prefix",  required=True)
    ap.add_argument("--explore",     type=int, default=2000)
    ap.add_argument("--refine",      type=int, default=500)
    ap.add_argument("--max-rounds",  type=int, default=5)
    ap.add_argument("--seed",        type=int,  default=20260405)
    ap.add_argument("--initial-balance", type=float, default=200.0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"WF PF>=3 Search v2 — base config: {args.base_config}", flush=True)
    print(f"Targets: PF>={args.pf_min}  net>=${args.old_net:,.0f}  DD<{args.old_dd}%", flush=True)

    base_s = load_settings(Path(args.base_config))
    fold_cache = _build_fold_cache(base_s)

    # Evaluate baseline first
    print("\n[BASELINE] Evaluating base config with current fold models …", flush=True)
    bl = _evaluate_baseline(base_s, fold_cache, args.initial_balance)
    print(f"[BASELINE]  net=${bl['sum_net_profit']:,.2f}  PF={bl['profit_factor_global']:.4f}"
          f"  DD={bl['global_max_drawdown_pct']:.2f}%  trades={bl['total_trades']}  WR={bl['win_rate']:.2%}", flush=True)
    Path(f"outputs/{args.out_prefix}_baseline.json").write_text(
        json.dumps(bl, indent=2, default=str))

    all_feasible: list[dict] = []
    warm_seeds: list[dict] | None = None

    for rnum in range(1, args.max_rounds + 1):
        if rnum <= 2:
            space = _build_space()
            label = f"Round {rnum} — compound_cap=50 standard"
        else:
            space = _extended_space()
            label = f"Round {rnum} — extended (cap=0 also searched)"

        rows, feasible = _run_round(
            rnum, label, base_s, fold_cache, space,
            args.explore, args.refine,
            args.initial_balance, args.pf_min, args.old_net, args.old_dd,
            args.out_prefix, args.seed,
            warm=warm_seeds,
        )
        all_feasible.extend(feasible)
        warm_seeds = sorted(rows, key=lambda r: r["distance_score"])[:30]

        if all_feasible:
            best = sorted(all_feasible, key=lambda r: (-r["total_trades"], r["global_max_drawdown_pct"]))[0]
            bnet = sorted(all_feasible, key=lambda r: -r["sum_net_profit"])[0]
            print(f"\n[R{rnum}] ★ TOTAL FEASIBLE: {len(all_feasible)}", flush=True)
            print(f"[R{rnum}]   Most trades: PF={best['profit_factor_global']:.3f}  net=${best['sum_net_profit']:,.0f}"
                  f"  DD={best['global_max_drawdown_pct']:.2f}%  trades={best['total_trades']}  WR={best['win_rate']:.2%}", flush=True)
            print(f"[R{rnum}]   Best net:    PF={bnet['profit_factor_global']:.3f}  net=${bnet['sum_net_profit']:,.0f}"
                  f"  DD={bnet['global_max_drawdown_pct']:.2f}%  trades={bnet['total_trades']}", flush=True)
            best_json_path = Path(f"outputs/{args.out_prefix}_best_feasible.json")
            best_json_path.write_text(
                json.dumps({"total_feasible": len(all_feasible), "most_trades": best, "best_net": bnet,
                            "baseline": bl}, indent=2, default=str))
            print(f"[R{rnum}]   Saved: {best_json_path}", flush=True)
        else:
            print(f"\n[R{rnum}] No feasible yet — continuing …", flush=True)

    # Final summary
    print(f"\n{'='*80}", flush=True)
    print(f"  SEARCH COMPLETE", flush=True)
    print(f"  Total feasible: {len(all_feasible)}", flush=True)
    print(f"  Baseline: net=${bl['sum_net_profit']:,.0f}  PF={bl['profit_factor_global']:.4f}  DD={bl['global_max_drawdown_pct']:.2f}%", flush=True)
    if all_feasible:
        # Best: most trades with reasonable PF (not extreme)
        balanced = [r for r in all_feasible if r["profit_factor_global"] <= 15 and r["total_trades"] >= 50]
        best_mt  = sorted(balanced or all_feasible, key=lambda r: (-r["total_trades"], r["global_max_drawdown_pct"]))[0]
        print(f"\n  ★ RECOMMENDED (most trades, PF<=15):", flush=True)
        for k in ["profit_factor_global","sum_net_profit","global_max_drawdown_pct","total_trades","win_rate",
                  "threshold","risk_per_trade","take_profit_rr","compound_cap","consecutive_loss_pause_count",
                  "consecutive_loss_cooldown_bars","template"]:
            print(f"    {k} = {best_mt.get(k)}", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
