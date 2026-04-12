#!/usr/bin/env python3
"""
Silver Bullet A/B Test
======================
For each account, evaluate the WF-best candidate with:
  A) silver_bullet_boost = 0.0  (disabled — WF baseline)
  B) silver_bullet_boost = configured value (mirroring live)

Uses the same fold cache as friction_opt_wf scripts, so results are
directly comparable to the existing WF numbers.

Usage:
  PYTHONPATH=src python scripts/silver_bullet_ab_test.py
"""
from __future__ import annotations

import json
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

# ── Session spread/slippage multipliers (must match the friction_opt scripts) ──

_ACC2_SESSION_SPREAD: dict[int, float] = {
    0: 1.65, 1: 1.65, 2: 1.65, 3: 1.65, 4: 1.65, 5: 1.65,
    6: 1.20,
    7: 0.67, 8: 0.67, 9: 0.67, 10: 0.67, 11: 0.67,
    12: 0.87,
    13: 0.53, 14: 0.53, 15: 0.53, 16: 0.53,
    17: 0.80,
    18: 1.20, 19: 1.50,
    20: 1.65, 21: 1.65, 22: 1.65, 23: 1.65,
}

_ACC1_SESSION_SPREAD: dict[int, float] = {
    0: 1.65, 1: 1.65, 2: 1.65, 3: 1.65, 4: 1.65, 5: 1.65,
    6: 1.20,
    7: 0.67, 8: 0.67, 9: 0.67, 10: 0.67, 11: 0.67,
    12: 0.87,
    13: 0.53, 14: 0.53, 15: 0.53, 16: 0.53,
    17: 0.80,
    18: 1.20, 19: 1.50,
    20: 1.65, 21: 1.65, 22: 1.65, 23: 1.65,
}

_SESSION_SLIPPAGE: dict[int, float] = {
    0: 1.80, 1: 1.80, 2: 1.80, 3: 1.80, 4: 1.80, 5: 1.80,
    6: 1.20,
    7: 0.70, 8: 0.70, 9: 0.70, 10: 0.70, 11: 0.70,
    12: 0.85,
    13: 0.60, 14: 0.60, 15: 0.60, 16: 0.60,
    17: 0.80,
    18: 1.20, 19: 1.50,
    20: 1.80, 21: 1.80, 22: 1.80, 23: 1.80,
}


def _add_session_mults(fold_cache: list, spread_map: dict[int, float]) -> None:
    for fold in fold_cache:
        df = fold.base_df
        if "time" in df.columns:
            hours = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.hour
            df["session_spread_mult"] = hours.map(spread_map).fillna(1.0)
            df["session_slippage_mult"] = hours.map(_SESSION_SLIPPAGE).fillna(1.0)
        else:
            df["session_spread_mult"] = 1.0
            df["session_slippage_mult"] = 1.0


# ── Best candidates from previous friction WF ────────────────────────────────

_ACC2_BEST = {
    "thr_buy": 0.60, "thr_sell": 0.60,
    "risk_per_trade": 0.020,
    "cooldown_bars": 4,
    "setup_exit_scale": 0.8,
    "hour_profile": "config",
    "daily_signal_cap": 0,
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

_ACC1_BEST = {
    "thr_buy": 0.59, "thr_sell": 0.59,
    "risk_per_trade": 0.026,
    "cooldown_bars": 4,
    "setup_exit_scale": 0.75,
    "hour_profile": "config",
    "daily_signal_cap": 0,
    "daily_loss_limit_pct": 0.20,
    "max_open_positions": 2,
    "sideway_risk_multiplier": 0.75,
    "strong_volatility_risk_multiplier": 1.20,
    "setup_exit_enabled": True,
    "thr_premium": 0.0,
    "weekday_profile": "all",
    "side_profile": "both",
    "quality_gate_profile": "off",
}


def _run_ab(
    label: str,
    config_path: Path,
    base_cand: dict[str, Any],
    sb_boost: float,
    sb_windows: list[list[int]],
    spread_map: dict[int, float],
    n_folds: int = 8,
    train_size: int = 250_000,
    test_size: int = 50_000,
    step_size: int = 50_000,
    label_horizon: int = 8,
) -> dict[str, Any]:
    """Build fold cache once, then evaluate A (SB=off) and B (SB=on)."""
    settings = load_settings(config_path)

    print(f"\n{'='*80}", flush=True)
    print(f"  {label} — Silver Bullet A/B Test", flush=True)
    print(f"  Config   : {config_path}", flush=True)
    print(f"  Friction : spread={settings.risk.spread_cost_rr} slip={settings.risk.slippage_rr}", flush=True)
    print(f"  SB boost : {sb_boost}  windows={sb_windows}", flush=True)
    print(f"{'='*80}", flush=True)

    # Build dataset
    print("[1/3] Building scalp dataset...", flush=True)
    from xauusd_ai.features.scalp_dataset import build_scalp_dataset
    dataset = build_scalp_dataset(
        start_date="2023-01-01",
        max_horizon=label_horizon,
        setup_label_mode="setup_aware",
    )
    print(f"      rows={len(dataset):,}", flush=True)

    # Build fold cache
    print("[2/3] Training fold models...", flush=True)
    fold_cache = _build_fold_cache_setup(
        dataset=dataset,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        n_folds=n_folds,
        label_horizon=label_horizon,
        train_objective="day_stability_strict",
    )
    if not fold_cache:
        raise RuntimeError(f"No valid folds for {label}")
    _add_session_mults(fold_cache, spread_map)
    print(f"      {len(fold_cache)} folds ready ✓", flush=True)

    # Evaluate A (SB=off) and B (SB=on)
    print("[3/3] Evaluating A vs B...", flush=True)
    results = {}
    for variant, boost_val in [("SB_OFF", 0.0), ("SB_ON", sb_boost)]:
        cand = {
            **base_cand,
            "silver_bullet_boost": boost_val,
            "silver_bullet_windows": sb_windows,
        }
        fold_rows, agg = _evaluate_candidate(cand, fold_cache, settings)
        results[variant] = {
            "candidate": cand,
            "agg": agg,
            "fold_details": fold_rows,
        }
        print(
            f"  {variant:>6}: net=${agg['sum_net_profit']:>10,.2f}  "
            f"PFg={agg['global_profit_factor']:.4f}  "
            f"avgDD={agg['avg_fold_dd_pct']:.2f}%  "
            f"minDay=${agg['min_fold_avg_daily_pnl']:.2f}  "
            f"worstDD={agg['worst_fold_max_daily_dd']:.2f}%  "
            f"minTrades={agg['min_fold_trades']}",
            flush=True,
        )

    # Delta
    on = results["SB_ON"]["agg"]
    off = results["SB_OFF"]["agg"]
    delta_net = on["sum_net_profit"] - off["sum_net_profit"]
    delta_pf = on["global_profit_factor"] - off["global_profit_factor"]
    delta_dd = on["avg_fold_dd_pct"] - off["avg_fold_dd_pct"]
    winner = "SB_ON" if delta_net > 0 else "SB_OFF"
    print(
        f"\n  DELTA (ON-OFF): net=${delta_net:>+,.2f}  PFg={delta_pf:>+.4f}  "
        f"avgDD={delta_dd:>+.2f}%  →  {winner}",
        flush=True,
    )

    return {
        "label": label,
        "config": str(config_path),
        "sb_boost": sb_boost,
        "sb_windows": sb_windows,
        "SB_OFF": results["SB_OFF"],
        "SB_ON": results["SB_ON"],
        "delta": {
            "net_profit": round(delta_net, 2),
            "profit_factor": round(delta_pf, 4),
            "avg_dd_pct": round(delta_dd, 4),
        },
        "winner": winner,
    }


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Silver Bullet A/B Test")
    p.add_argument("--acc1-only", action="store_true", help="Only run ACC1 A/B (skip ACC2)")
    p.add_argument("--acc2-only", action="store_true", help="Only run ACC2 A/B (skip ACC1)")
    args = p.parse_args()

    t0 = time.time()

    all_results: list[dict[str, Any]] = []

    # ACC2 (Exness Pro): SB boost = 0.03
    if not args.acc1_only:
        r2 = _run_ab(
            label="ACC2 (Exness Pro)",
            config_path=Path("configs/live_acc2_scalp_m1.yaml"),
            base_cand=_ACC2_BEST,
            sb_boost=0.03,
            sb_windows=[[3, 4], [6, 7], [10, 11], [14, 15]],
            spread_map=_ACC2_SESSION_SPREAD,
        )
        all_results.append(r2)

    # ACC1 (Exness Pro): SB boost = 0.04
    if not args.acc2_only:
        r1 = _run_ab(
            label="ACC1 (Exness Pro)",
            config_path=Path("configs/live_acc1_scalp_m1.yaml"),
            base_cand=_ACC1_BEST,
            sb_boost=0.04,
            sb_windows=[[3, 4], [6, 7], [10, 11], [14, 15]],
            spread_map=_ACC1_SESSION_SPREAD,
        )
        all_results.append(r1)

    # Save results
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path("outputs") / f"silver_bullet_ab_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(all_results, indent=2, default=str),
        encoding="utf-8",
    )

    elapsed = time.time() - t0
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    for r in all_results:
        off_net = r["SB_OFF"]["agg"]["sum_net_profit"]
        on_net = r["SB_ON"]["agg"]["sum_net_profit"]
        off_pf = r["SB_OFF"]["agg"]["global_profit_factor"]
        on_pf = r["SB_ON"]["agg"]["global_profit_factor"]
        print(
            f"  {r['label']:>25}:  OFF=${off_net:>10,.2f} PF={off_pf:.4f}  |  "
            f"ON=${on_net:>10,.2f} PF={on_pf:.4f}  |  "
            f"Δ=${r['delta']['net_profit']:>+,.2f}  →  {r['winner']}",
            flush=True,
        )
    print(f"\nElapsed: {elapsed:.0f}s", flush=True)
    print(f"Results: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
