#!/usr/bin/env python3
"""
Friction-Optimized WF Search
=============================
Same fold cache + evaluator as acc2_m1_setup_exit_wf.py but with:
  1. session_spread_mult populated per bar (realistic per-session friction)
  2. Dense focused grid on friction-sensitive levers (~120 candidates)

Goal: match or exceed baseline $91k net WITH realistic Exness Pro friction.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from acc2_m1_setup_exit_wf import (  # type: ignore[import]
    _build_fold_cache_setup,
    _evaluate_candidate,
)
from xauusd_ai.config import load_settings

# ── Session-aware spread multiplier (UTC hour → mult relative to config spread) ─
# Based on Exness Pro XAUUSD typical spreads 2024-2026:
#   config spread_cost_rr = 0.035 represents ~$0.15 average spread
#   London active:    ~$0.10 spread → mult=0.67
#   London-NY overlap:~$0.08 spread → mult=0.53
#   Asian dead-zone:  ~$0.25 spread → mult=1.67
_SESSION_SPREAD_MULT: dict[int, float] = {
    0: 1.65, 1: 1.65, 2: 1.65, 3: 1.65, 4: 1.65, 5: 1.65,  # Asian dead-zone
    6: 1.20,                                                    # Pre-London
    7: 0.67, 8: 0.67, 9: 0.67, 10: 0.67, 11: 0.67,           # London active
    12: 0.87,                                                   # London lunch
    13: 0.53, 14: 0.53, 15: 0.53, 16: 0.53,                   # London-NY overlap
    17: 0.80,                                                   # NY solo
    18: 1.20, 19: 1.50,                                        # NY closing
    20: 1.65, 21: 1.65, 22: 1.65, 23: 1.65,                   # Asian opening
}

# ── Session-aware slippage multiplier (UTC hour → mult relative to config slippage_rr) ─
# Liquidity-driven; less extreme than spread since M1 scalp orders are small.
_SESSION_SLIPPAGE_MULT: dict[int, float] = {
    0: 1.80, 1: 1.80, 2: 1.80, 3: 1.80, 4: 1.80, 5: 1.80,  # Asian dead-zone
    6: 1.20,                                                    # Pre-London
    7: 0.70, 8: 0.70, 9: 0.70, 10: 0.70, 11: 0.70,           # London active
    12: 0.85,                                                   # London lunch
    13: 0.60, 14: 0.60, 15: 0.60, 16: 0.60,                   # London-NY overlap
    17: 0.80,                                                   # NY solo
    18: 1.20, 19: 1.50,                                        # NY closing
    20: 1.80, 21: 1.80, 22: 1.80, 23: 1.80,                   # Asian opening
}


def _add_session_spread_mult(fold_cache: list) -> None:
    """Populate session_spread_mult and session_slippage_mult in each fold's base_df (in-place)."""
    for fold in fold_cache:
        df = fold.base_df
        if "time" in df.columns:
            hours = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.hour
            df["session_spread_mult"] = hours.map(_SESSION_SPREAD_MULT).fillna(1.0)
            df["session_slippage_mult"] = hours.map(_SESSION_SLIPPAGE_MULT).fillna(1.0)
        else:
            df["session_spread_mult"] = 1.0
            df["session_slippage_mult"] = 1.0


# ── Focused candidate grid ────────────────────────────────────────────────────

def _build_focused_candidates(seed: int = 42) -> list[dict[str, Any]]:
    """
    Dense grid on friction-sensitive levers:
      - thr (signal quality)
      - setup_exit_scale (TP width → amortize friction)
      - hour_profile (session filter → lower friction)
      - risk_per_trade (per-trade edge vs friction)
      - cooldown_bars (trade spacing)
      - daily_signal_cap (prevent over-trading)
    """
    rng = random.Random(seed)

    # Fixed params (known good from previous search)
    FIXED = {
        "daily_loss_limit_pct": 0.20,
        "max_open_positions": 2,
        "sideway_risk_multiplier": 0.7,
        "strong_volatility_risk_multiplier": 1.2,
        "setup_exit_enabled": True,
        "thr_premium": 0.0,
        "weekday_profile": "all",
        "side_profile": "both",
        "quality_gate_profile": "off",
    }

    # Lever 1: threshold (fewer trades, higher edge per trade)
    thr_vals = [0.58, 0.60, 0.62, 0.64, 0.66, 0.70]

    # Lever 2: setup exit scale (wider TP → bigger wins, same friction cost)
    scale_vals = [0.8, 1.0, 1.2, 1.5]

    # Lever 3: hour profile (avoid high-spread sessions)
    hour_vals = ["config", "liquid_all", "london_only", "ny_open", "overlap_core"]

    # Lever 4: risk per trade
    risk_vals = [0.020, 0.025, 0.030, 0.040]

    # Lever 5: cooldown bars
    cool_vals = [4, 8, 12]

    # Lever 6: daily signal cap
    cap_vals = [0, 3, 5, 8]

    # ── Phase A: Full grid on top 3 levers (thr × scale × hour)
    # 6 × 4 × 5 = 120 combos — we pick ~60 from this (strategic sampling)
    grid_a = list(itertools.product(thr_vals, scale_vals, hour_vals))
    rng.shuffle(grid_a)

    candidates: list[dict[str, Any]] = []
    seen: set = set()

    # Add strategic anchors first
    anchors = [
        # Current best (for comparison)
        {"thr_buy": 0.60, "thr_sell": 0.60, "risk_per_trade": 0.020,
         "cooldown_bars": 4, "setup_exit_scale": 0.8,
         "hour_profile": "config", "daily_signal_cap": 0},
        # Higher threshold + session filter
        {"thr_buy": 0.64, "thr_sell": 0.64, "risk_per_trade": 0.030,
         "cooldown_bars": 8, "setup_exit_scale": 1.0,
         "hour_profile": "liquid_all", "daily_signal_cap": 5},
        # High selectivity + wider TP + overlap-only
        {"thr_buy": 0.66, "thr_sell": 0.66, "risk_per_trade": 0.040,
         "cooldown_bars": 8, "setup_exit_scale": 1.2,
         "hour_profile": "overlap_core", "daily_signal_cap": 3},
        # London only + moderate threshold
        {"thr_buy": 0.62, "thr_sell": 0.62, "risk_per_trade": 0.030,
         "cooldown_bars": 4, "setup_exit_scale": 1.0,
         "hour_profile": "london_only", "daily_signal_cap": 0},
        # NY open focus
        {"thr_buy": 0.62, "thr_sell": 0.62, "risk_per_trade": 0.025,
         "cooldown_bars": 4, "setup_exit_scale": 1.2,
         "hour_profile": "ny_open", "daily_signal_cap": 5},
        # Max selectivity + high risk
        {"thr_buy": 0.70, "thr_sell": 0.70, "risk_per_trade": 0.050,
         "cooldown_bars": 12, "setup_exit_scale": 1.5,
         "hour_profile": "liquid_all", "daily_signal_cap": 3},
        # Moderate all-round
        {"thr_buy": 0.62, "thr_sell": 0.62, "risk_per_trade": 0.025,
         "cooldown_bars": 8, "setup_exit_scale": 1.0,
         "hour_profile": "liquid_all", "daily_signal_cap": 0},
        # Low threshold but session-filtered + wide TP
        {"thr_buy": 0.58, "thr_sell": 0.58, "risk_per_trade": 0.020,
         "cooldown_bars": 4, "setup_exit_scale": 1.5,
         "hour_profile": "london_only", "daily_signal_cap": 5},
    ]
    for a in anchors:
        c = {**FIXED, **a}
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            candidates.append(c)

    # ── Phase B: Structured grid sampling on top 3 levers
    for thr, scale, hour in grid_a:
        rf = rng.choice(risk_vals)
        cool = rng.choice(cool_vals)
        cap = rng.choice(cap_vals)
        c = {
            **FIXED,
            "thr_buy": thr, "thr_sell": thr,
            "risk_per_trade": rf,
            "cooldown_bars": cool,
            "setup_exit_scale": scale,
            "hour_profile": hour,
            "daily_signal_cap": cap,
        }
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            candidates.append(c)
        if len(candidates) >= 120:
            break

    # ── Phase C: Random fill (if needed)
    while len(candidates) < 120:
        c = {
            **FIXED,
            "thr_buy": rng.choice(thr_vals),
            "thr_sell": rng.choice(thr_vals),
            "risk_per_trade": rng.choice(risk_vals),
            "cooldown_bars": rng.choice(cool_vals),
            "setup_exit_scale": rng.choice(scale_vals),
            "hour_profile": rng.choice(hour_vals),
            "daily_signal_cap": rng.choice(cap_vals),
        }
        c["thr_sell"] = c["thr_buy"]  # keep symmetric
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            candidates.append(c)

    return candidates


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Friction-Optimized WF Search")
    p.add_argument("--config", type=Path, default=Path("configs/live_acc2_scalp_m1.yaml"))
    p.add_argument("--n-folds", type=int, default=8)
    p.add_argument("--train-size", type=int, default=250_000)
    p.add_argument("--test-size", type=int, default=50_000)
    p.add_argument("--step-size", type=int, default=50_000)
    p.add_argument("--label-max-horizon", type=int, default=8)
    p.add_argument("--train-objective", type=str, default="day_stability_strict")
    p.add_argument("--seed", type=int, default=20260413)
    p.add_argument("--out-prefix", type=str, default="acc2_wf_friction_opt")
    p.add_argument("--target-daily-pnl", type=float, default=20.0)
    p.add_argument("--target-daily-dd", type=float, default=20.0)
    p.add_argument("--target-days-neg-max-pct", type=float, default=60.0)
    p.add_argument("--target-min-trades", type=int, default=80)  # Lower floor for session-filtered
    args = p.parse_args()

    t0 = time.time()
    settings = load_settings(args.config)

    print("=" * 96, flush=True)
    print("FRICTION-OPTIMIZED WF SEARCH (session-aware spread)", flush=True)
    print(f"  Config        : {args.config}", flush=True)
    print(f"  Friction      : spread={settings.risk.spread_cost_rr} + slip={settings.risk.slippage_rr} + comm={settings.risk.commission_rr}", flush=True)
    print(f"  Session mult  : London=0.67  Overlap=0.53  Asian=1.65", flush=True)
    print(f"  Folds         : {args.n_folds}", flush=True)
    print("=" * 96, flush=True)

    # ── 1. Build dataset ──────────────────────────────────────────────────────
    print("[1/4] Building scalp dataset...", flush=True)
    from xauusd_ai.features.scalp_dataset import build_scalp_dataset
    dataset = build_scalp_dataset(
        start_date="2023-01-01",
        max_horizon=args.label_max_horizon,
        setup_label_mode="setup_aware",
    )
    print(f"      rows={len(dataset):,}", flush=True)

    # ── 2. Build fold cache + add session_spread_mult ─────────────────────────
    print("[2/4] Training fold models + adding session spread multipliers...", flush=True)
    fold_cache = _build_fold_cache_setup(
        dataset=dataset,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        n_folds=args.n_folds,
        label_horizon=args.label_max_horizon,
        train_objective=args.train_objective,
    )
    if not fold_cache:
        raise RuntimeError("No valid folds built.")
    _add_session_spread_mult(fold_cache)
    print(f"      {len(fold_cache)} folds with session_spread_mult ✓", flush=True)

    # Verify: show avg session_spread_mult per fold
    for f in fold_cache:
        avg_m = f.base_df["session_spread_mult"].mean()
        std_m = f.base_df["session_spread_mult"].std()
        print(f"      fold {f.fold}: session_spread_mult avg={avg_m:.3f} std={std_m:.3f}")

    # ── 3. Evaluate candidates ────────────────────────────────────────────────
    candidates = _build_focused_candidates(seed=args.seed)
    print(f"\n[3/4] Evaluating {len(candidates)} focused candidates...", flush=True)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = -1e18
    feasible_count = 0

    for idx, cand in enumerate(candidates, start=1):
        fold_rows, agg = _evaluate_candidate(cand, fold_cache, settings)
        fold_df = pd.DataFrame(fold_rows)

        pf_global = agg["global_profit_factor"]
        sum_net   = agg["sum_net_profit"]
        min_daily = agg["min_fold_avg_daily_pnl"]
        worst_dd  = agg["worst_fold_max_daily_dd"]
        max_neg   = agg["max_fold_days_neg_pct"]
        min_trade = agg["min_fold_trades"]
        avg_fold_dd = agg["avg_fold_dd_pct"]

        feasible = bool(
            min_daily >= args.target_daily_pnl
            and worst_dd <= args.target_daily_dd
            and max_neg <= args.target_days_neg_max_pct
            and min_trade >= args.target_min_trades
        )
        if feasible:
            feasible_count += 1

        gap = (
            max(0.0, args.target_daily_pnl - min_daily) * 50.0
            + max(0.0, worst_dd - args.target_daily_dd) * 200.0
            + max(0.0, max_neg - args.target_days_neg_max_pct) * 100.0
            + max(0.0, args.target_min_trades - min_trade) * 300.0
            + max(0.0, 2.0 - pf_global) * 200.0
        )
        score = (1e9 + sum_net) if feasible else (-gap + sum_net * 0.01 - avg_fold_dd)

        row: dict[str, Any] = {
            **cand, **agg,
            "feasible": feasible,
            "gap_score": round(gap, 4),
        }
        rows.append(row)

        if score > best_score:
            best_score = score
            best = row | {"fold_details": fold_rows}

        if idx % 5 == 0 or feasible:
            bmark = "✅" if feasible else "  "
            print(
                f"  [{idx:>4}/{len(candidates)}]{bmark} thr={cand['thr_buy']:.2f} "
                f"rf={cand['risk_per_trade']:.3f} scale={cand['setup_exit_scale']:.1f} "
                f"hr={cand['hour_profile']:<14} cap={cand.get('daily_signal_cap',0)} "
                f"net=${sum_net:>9,.2f}  PFg={pf_global:.3f}  "
                f"minDay=${min_daily:>6.2f}  ddW={worst_dd:.2f}%  "
                f"neg={max_neg:.1f}%  t={min_trade}",
                flush=True,
            )

    # ── 4. Save results ───────────────────────────────────────────────────────
    print("\n[4/4] Saving results...", flush=True)
    res = pd.DataFrame(rows).sort_values(
        ["feasible", "sum_net_profit", "global_profit_factor"],
        ascending=[False, False, False],
    )
    feasible_res = res[res["feasible"] == True]  # noqa: E712

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out_prefix}_{stamp}.csv"
    json_path = out_dir / f"{args.out_prefix}_{stamp}.json"
    res.to_csv(csv_path, index=False)

    payload = {
        "search": {
            "script": "acc2_friction_opt_wf.py",
            "config": str(args.config),
            "session_spread_mult": _SESSION_SPREAD_MULT,
            "candidates": len(candidates),
            "feasible_count": int(len(feasible_res)),
            "folds": len(fold_cache),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "best": best,
        "top10": res.head(10).to_dict(orient="records"),
        "top10_feasible": feasible_res.head(10).to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("-" * 96, flush=True)
    if best:
        print(
            f"BEST: thr={best['thr_buy']:.2f} rf={best['risk_per_trade']:.3f} "
            f"scale={best['setup_exit_scale']:.1f} hr={best['hour_profile']} "
            f"cap={best.get('daily_signal_cap',0)} cool={best['cooldown_bars']}",
            flush=True,
        )
        print(
            f"      net=${best['sum_net_profit']:,.2f}  PFg={best['global_profit_factor']:.4f}  "
            f"minFoldAvgDay=${best['min_fold_avg_daily_pnl']:.2f}  "
            f"worstDD={best['worst_fold_max_daily_dd']:.2f}%  "
            f"maxNeg={best['max_fold_days_neg_pct']:.1f}%  "
            f"minTrades={best['min_fold_trades']}  feasible={best['feasible']}",
            flush=True,
        )
        # Per-fold breakdown
        print("\n  Per-fold breakdown:", flush=True)
        for fd in best.get("fold_details", []):
            print(
                f"    fold {fd['fold']}: net=${fd['net_profit']:>8,.0f}  "
                f"PF={fd['profit_factor']:.3f}  WR={fd['win_rate']:.1%}  "
                f"trades={fd['trades']:,}  DD={fd['max_drawdown_pct_abs']:.1f}%",
                flush=True,
            )

    print(f"\nfeasible: {len(feasible_res)}/{len(res)}", flush=True)

    # Compare to benchmarks
    import glob
    bench_files = sorted(glob.glob("outputs/acc2_wf_maxdd12_verify_*.json"))
    exness_files = sorted(glob.glob("outputs/acc2_wf_exness_pro_*.json"))
    if bench_files:
        b = json.load(open(bench_files[-1]))["best"]
        print(f"\nBENCHMARK (fr=0.02):  net=${b['sum_net_profit']:>9,.2f}  PFg={b['global_profit_factor']:.4f}  trades={b['global_trades']:,}", flush=True)
    if exness_files:
        e = json.load(open(exness_files[-1]))["best"]
        print(f"EXNESS FLAT (fr=0.045): net=${e['sum_net_profit']:>9,.2f}  PFg={e['global_profit_factor']:.4f}  trades={e['global_trades']:,}", flush=True)
    if best:
        print(f"THIS RUN (session):   net=${best['sum_net_profit']:>9,.2f}  PFg={best['global_profit_factor']:.4f}  trades={best['global_trades']:,}", flush=True)

    print(f"\n  {csv_path}", flush=True)
    print(f"  {json_path}", flush=True)
    print(f"  elapsed: {time.time()-t0:.0f}s", flush=True)
    print("-" * 96, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
