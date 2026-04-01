#!/usr/bin/env python3
"""
Phase-3 optimizer: enforce minimum executed trades per volatility regime.

Goal:
  - Keep DD in target band
  - Keep net profit high
  - Force strategy to prove itself across regimes (0/1/2)

Notes:
  - Uses cached fold model probabilities from phase-2 pipeline.
  - Re-maps volatility_regime at simulation time from atr_ratio using candidate
    thresholds (sideways/strong). This stress-tests execution/risk behavior
    across regimes without retraining model weights each candidate.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import Settings, load_settings
from xauusd_ai.execution.risk import RiskManager

# Reuse phase-2 helpers for cache + core candidate wiring
import importlib.util
import sys


def _load_phase2_module():
    mod_path = Path("scripts/wf_phase2_optimizer.py")
    spec = importlib.util.spec_from_file_location("wf_phase2_optimizer", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _max_drawdown_pct_from_curve(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve[1:]:
        if value > peak:
            peak = value
        if peak <= 0:
            continue
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100.0


def _candidate_space() -> dict[str, list[Any]]:
    # Focused neighborhood around acc2_phase2_dd19 winner + regime remap knobs
    return {
        "threshold": [0.62, 0.66, 0.70, 0.74],
        "min_strategy_score": [0.10, 0.20, 0.30, 0.40],
        "sideway_min_strategy_score": [0.05, 0.10, 0.15],
        "strong_volatility_min_strategy_score": [0.10, 0.20, 0.30],
        "risk_min_conf": [0.56, 0.58, 0.60, 0.62],
        "sideway_min_conf": [0.60, 0.64, 0.68],
        "volatile_min_conf": [0.56, 0.60, 0.64],
        "require_trend_alignment": [False, True],
        "template": ["minimal", "block_3_17", "block_2_7_17_21", "focus_hours", "mon_tue_weak_cluster"],
        "risk_per_trade": [0.024, 0.028, 0.032, 0.036],
        "risk_tier_floor": [0.010, 0.014, 0.018],
        "sideway_risk_multiplier": [0.20, 0.30, 0.40],
        "strong_volatility_risk_multiplier": [0.50, 0.60, 0.70],
        "anti_martingale_factor": [0.50, 0.60, 0.70],
        "consecutive_loss_pause_count": [2, 3, 4],
        "consecutive_loss_cooldown_bars": [4, 8, 12],
        "daily_loss_limit_pct": [0.03, 0.05],
        "max_total_exposure_pct": [0.08, 0.10],
        "max_open_positions": [2, 3],
        # regime remap thresholds (from atr_ratio)
        "sideways_volatility_threshold": [0.0008, 0.0010, 0.0012, 0.0014],
        "strong_volatility_threshold": [0.0022, 0.0028, 0.0034, 0.0042, 0.0052],
    }


def _seed_candidates() -> list[dict[str, Any]]:
    # Winner from phase2_dd19 + nearby variants
    base = {
        "threshold": 0.74,
        "min_strategy_score": 0.30,
        "sideway_min_strategy_score": 0.10,
        "strong_volatility_min_strategy_score": 0.20,
        "risk_min_conf": 0.58,
        "sideway_min_conf": 0.64,
        "volatile_min_conf": 0.60,
        "require_trend_alignment": False,
        "template": "block_3_17",
        "risk_per_trade": 0.032,
        "risk_tier_floor": 0.014,
        "sideway_risk_multiplier": 0.35,
        "strong_volatility_risk_multiplier": 0.55,
        "anti_martingale_factor": 0.50,
        "consecutive_loss_pause_count": 4,
        "consecutive_loss_cooldown_bars": 8,
        "daily_loss_limit_pct": 0.03,
        "max_total_exposure_pct": 0.10,
        "max_open_positions": 3,
        "sideways_volatility_threshold": 0.0010,
        "strong_volatility_threshold": 0.0028,
    }
    seeds: list[dict[str, Any]] = [base]
    for rt in [0.028, 0.032, 0.036]:
        for ms in [0.10, 0.20, 0.30]:
            for sthr, vthr in [(0.0008, 0.0022), (0.0010, 0.0028), (0.0012, 0.0034)]:
                c = dict(base)
                c["risk_per_trade"] = rt
                c["min_strategy_score"] = ms
                c["sideways_volatility_threshold"] = sthr
                c["strong_volatility_threshold"] = vthr
                seeds.append(c)
    for tpl in ["minimal", "focus_hours", "mon_tue_weak_cluster", "block_2_7_17_21"]:
            c = dict(base)
            c["template"] = tpl
            seeds.append(c)
    return seeds


def _sample_candidates(space: dict[str, list[Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    dims = list(space.keys())
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for c in _seed_candidates():
        key = tuple(c[d] for d in dims)
        if key in seen:
            continue
        if float(c["sideways_volatility_threshold"]) >= float(c["strong_volatility_threshold"]):
            continue
        seen.add(key)
        out.append(c)

    while len(out) < n:
        c = {d: rng.choice(space[d]) for d in dims}
        if float(c["sideways_volatility_threshold"]) >= float(c["strong_volatility_threshold"]):
            continue
        key = tuple(c[d] for d in dims)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _apply_candidate(settings: Settings, candidate: dict[str, Any], phase2_mod: Any) -> Settings:
    tuned = phase2_mod._apply_candidate(
        settings,  # noqa: SLF001 (intentional reuse)
        {
            k: candidate[k]
            for k in [
                "threshold",
                "min_strategy_score",
                "sideway_min_strategy_score",
                "strong_volatility_min_strategy_score",
                "risk_min_conf",
                "sideway_min_conf",
                "volatile_min_conf",
                "require_trend_alignment",
                "template",
                "risk_per_trade",
                "risk_tier_floor",
                "sideway_risk_multiplier",
                "strong_volatility_risk_multiplier",
                "anti_martingale_factor",
                "consecutive_loss_pause_count",
                "consecutive_loss_cooldown_bars",
                "daily_loss_limit_pct",
                "max_total_exposure_pct",
                "max_open_positions",
            ]
        },
        phase2_mod._time_templates(),  # noqa: SLF001
    )
    tuned.strategy.sideways_volatility_threshold = float(candidate["sideways_volatility_threshold"])
    tuned.strategy.strong_volatility_threshold = float(candidate["strong_volatility_threshold"])
    return tuned


def _evaluate_candidate(
    base_settings: Settings,
    fold_cache: list[Any],
    candidate: dict[str, Any],
    phase2_mod: Any,
    initial_balance: float,
    target_net: float,
    dd_min: float,
    dd_max: float,
    min_trades_sideway: int,
    min_trades_normal: int,
    min_trades_strong: int,
    keep_trades: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    tuned = _apply_candidate(base_settings, candidate, phase2_mod)
    balance = float(initial_balance)
    all_trades: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    equity_curve: list[float] = [balance]
    global_peak = balance
    global_max_dd = 0.0
    regime_counts = {0: 0, 1: 0, 2: 0}
    regime_pnl = {0: 0.0, 1: 0.0, 2: 0.0}

    side_thr = float(candidate["sideways_volatility_threshold"])
    strong_thr = float(candidate["strong_volatility_threshold"])
    threshold = float(candidate["threshold"])

    for fold in fold_cache:
        fold_settings = tuned.model_copy(deep=True)
        fold_settings.training.backtest_initial_balance = balance

        preds = fold.predictions.copy()
        # Re-map regimes from atr_ratio per candidate thresholds.
        preds["volatility_regime"] = np.select(
            [
                preds["atr_ratio"] <= side_thr,
                preds["atr_ratio"] >= strong_thr,
            ],
            [0, 2],
            default=1,
        )
        preds["prediction"] = (preds["probability"] >= threshold).astype(int)

        sim = simulate_dynamic_concurrent_backtest(
            preds,
            fold_settings,
            RiskManager(fold_settings),
            label="test",
            compound=True,
        )
        rep = sim.report

        fold_start_balance = balance
        balance = float(rep["ending_balance"])
        equity_curve.append(balance)

        global_peak = max(global_peak, fold_start_balance)
        fold_dd_frac = abs(float(rep["max_drawdown_pct"])) / 100.0
        fold_low_est = max(0.0, fold_start_balance * (1.0 - fold_dd_frac))
        if global_peak > 0:
            global_max_dd = max(global_max_dd, (global_peak - fold_low_est) / global_peak)
        global_peak = max(global_peak, balance)
        if global_peak > 0:
            global_max_dd = max(global_max_dd, (global_peak - balance) / global_peak)

        fold_rows.append(
            {
                "fold": fold.fold,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "ending_balance": balance,
                "net_profit": float(rep["net_profit"]),
                "gross_profit": float(rep["gross_profit"]),
                "gross_loss": float(rep["gross_loss"]),
                "profit_factor": float(rep["profit_factor"]),
                "max_drawdown_pct": float(rep["max_drawdown_pct"]),
                "trades": int(rep["trades"]),
                "return_pct": float(rep["return_pct"]),
                "wins": int(rep["wins"]),
                "losses": int(rep["losses"]),
            }
        )

        if not sim.trades.empty:
            fold_trades = sim.trades.copy()
            # Always aggregate regime stats, even in fast mode (keep_trades=False)
            cnt_map = fold_trades["volatility_regime"].value_counts().to_dict()
            pnl_map = fold_trades.groupby("volatility_regime")["pnl"].sum().to_dict()
            for regime in (0, 1, 2):
                regime_counts[regime] += int(cnt_map.get(regime, 0))
                regime_pnl[regime] += float(pnl_map.get(regime, 0.0))

            # Always extend exact equity curve to keep DD estimation accurate.
            bal_after = pd.to_numeric(fold_trades["balance_after"], errors="coerce").dropna().tolist()
            if bal_after:
                equity_curve.extend(float(v) for v in bal_after)

            if keep_trades:
                fold_trades["fold"] = fold.fold
                fold_trades["test_start"] = fold.test_start
                fold_trades["test_end"] = fold.test_end
                all_trades.append(fold_trades)

    fold_df = pd.DataFrame(fold_rows)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    gross_profit = float(fold_df["gross_profit"].sum()) if not fold_df.empty else 0.0
    gross_loss = float(fold_df["gross_loss"].sum()) if not fold_df.empty else 0.0
    net_profit = balance - initial_balance
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    exact_dd_pct = _max_drawdown_pct_from_curve(equity_curve)
    max_dd_global_pct = max(global_max_dd * 100.0, exact_dd_pct)
    avg_fold_dd_abs = float(fold_df["max_drawdown_pct"].abs().mean()) if not fold_df.empty else 0.0
    total_trades = int(fold_df["trades"].sum()) if not fold_df.empty else 0
    total_wins = int(fold_df["wins"].sum()) if not fold_df.empty else 0
    total_losses = int(fold_df["losses"].sum()) if not fold_df.empty else 0
    win_rate = total_wins / max(total_wins + total_losses, 1)

    # If detailed trades are requested, reconcile with exact grouped values.
    if keep_trades and not trades_df.empty:
        counts = trades_df["volatility_regime"].value_counts().to_dict()
        regime_counts = {0: int(counts.get(0, 0)), 1: int(counts.get(1, 0)), 2: int(counts.get(2, 0))}
        pnl_map = trades_df.groupby("volatility_regime")["pnl"].sum().to_dict()
        regime_pnl = {0: float(pnl_map.get(0, 0.0)), 1: float(pnl_map.get(1, 0.0)), 2: float(pnl_map.get(2, 0.0))}

    meets_regime = (
        regime_counts[0] >= min_trades_sideway
        and regime_counts[1] >= min_trades_normal
        and regime_counts[2] >= min_trades_strong
    )
    meets_dd = dd_min <= max_dd_global_pct <= dd_max
    meets_target = bool(net_profit >= target_net and meets_dd and meets_regime)

    net_short = max(0.0, target_net - net_profit)
    dd_dist = 0.0 if meets_dd else min(abs(max_dd_global_pct - dd_min), abs(max_dd_global_pct - dd_max))
    regime_short = (
        max(0, min_trades_sideway - regime_counts[0])
        + max(0, min_trades_normal - regime_counts[1])
        + max(0, min_trades_strong - regime_counts[2])
    )
    distance_score = net_short + dd_dist * 500.0 + regime_short * 25.0

    result = {
        **candidate,
        "starting_balance": round(initial_balance, 2),
        "ending_balance": round(balance, 2),
        "sum_net_profit": round(net_profit, 2),
        "sum_gross_profit": round(gross_profit, 2),
        "sum_gross_loss": round(gross_loss, 2),
        "global_max_drawdown_pct": round(max_dd_global_pct, 4),
        "avg_fold_drawdown_pct_abs": round(avg_fold_dd_abs, 4),
        "profit_factor_global": round(profit_factor, 4),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 4),
        "avg_fold_return_pct": round(float(fold_df["return_pct"].mean()) if not fold_df.empty else 0.0, 4),
        "folds": int(len(fold_df)),
        "regime0_trades": regime_counts[0],
        "regime1_trades": regime_counts[1],
        "regime2_trades": regime_counts[2],
        "regime0_pnl": round(regime_pnl[0], 2),
        "regime1_pnl": round(regime_pnl[1], 2),
        "regime2_pnl": round(regime_pnl[2], 2),
        "meets_regime_constraint": bool(meets_regime),
        "meets_dd_constraint": bool(meets_dd),
        "meets_target": meets_target,
        "distance_score": round(distance_score, 4),
    }
    return result, trades_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-3 WF optimizer with regime min-trades constraints.")
    p.add_argument("--config", type=Path, default=Path("configs/experiments/acc2_phase3_regime_mintrades.yaml"))
    p.add_argument("--target-net", type=float, default=6_000.0)
    p.add_argument("--dd-min", type=float, default=15.0)
    p.add_argument("--dd-max", type=float, default=30.0)
    p.add_argument("--min-trades-sideway", type=int, default=220)
    p.add_argument("--min-trades-normal", type=int, default=25)
    p.add_argument("--min-trades-strong", type=int, default=20)
    p.add_argument("--initial-balance", type=float, default=200.0)
    p.add_argument("--candidates", type=int, default=320)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", type=str, default="wf_phase3_regime_acc2")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    phase2_mod = _load_phase2_module()
    base = load_settings(args.config)
    base.strategy.ict_weight = 0.35
    base.strategy.wyckoff_weight = 0.35
    base.strategy.momentum_weight = 0.40
    base.training.label_horizon = 12
    base.training.min_return_threshold = 0.0008

    print("=" * 96, flush=True)
    print("PHASE-3 WF OPTIMIZER (regime min-trades constraints)", flush=True)
    print(f"Config   : {args.config}", flush=True)
    print(
        f"Target   : net >= ${args.target_net:,.2f} | DD in [{args.dd_min:.1f}%, {args.dd_max:.1f}%] "
        f"| min trades regime (0/1/2)=({args.min_trades_sideway}/{args.min_trades_normal}/{args.min_trades_strong})",
        flush=True,
    )
    print(f"Search   : {args.candidates} candidates", flush=True)
    print("=" * 96, flush=True)

    fold_cache, dataset = phase2_mod._build_fold_cache(base)  # noqa: SLF001
    print(
        f"      test_window=[{fold_cache[0].test_start} -> {fold_cache[-1].test_end}] "
        f"folds={len(fold_cache)} dataset={len(dataset):,}",
        flush=True,
    )

    space = _candidate_space()
    candidates = _sample_candidates(space, args.candidates, args.seed)

    rows: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None

    for idx, candidate in enumerate(candidates, start=1):
        res, trades = _evaluate_candidate(
            base_settings=base,
            fold_cache=fold_cache,
            candidate=candidate,
            phase2_mod=phase2_mod,
            initial_balance=args.initial_balance,
            target_net=args.target_net,
            dd_min=args.dd_min,
            dd_max=args.dd_max,
            min_trades_sideway=args.min_trades_sideway,
            min_trades_normal=args.min_trades_normal,
            min_trades_strong=args.min_trades_strong,
            keep_trades=False,
        )
        rows.append(res)
        if idx % 20 == 0 or res["meets_target"]:
            tag = "✅" if res["meets_target"] else "…"
            print(
                f"[{idx}/{len(candidates)}]{tag} net=${res['sum_net_profit']:,.2f} "
                f"dd={res['global_max_drawdown_pct']:.2f}% pf={res['profit_factor_global']:.3f} "
                f"reg0/1/2={res['regime0_trades']}/{res['regime1_trades']}/{res['regime2_trades']}",
                flush=True,
            )

    res_df = pd.DataFrame(rows)
    feasible = res_df[res_df["meets_target"] == True].copy()  # noqa: E712

    if not feasible.empty:
        # prioritize highest net, then lowest DD, then stronger PF
        best = feasible.sort_values(
            ["sum_net_profit", "global_max_drawdown_pct", "profit_factor_global"],
            ascending=[False, True, False],
        ).iloc[0].to_dict()
        mode = "feasible"
    else:
        best = res_df.sort_values("distance_score", ascending=True).iloc[0].to_dict()
        mode = "closest_possible"

    best_candidate = {
        k: best[k]
        for k in [
            "threshold",
            "min_strategy_score",
            "sideway_min_strategy_score",
            "strong_volatility_min_strategy_score",
            "risk_min_conf",
            "sideway_min_conf",
            "volatile_min_conf",
            "require_trend_alignment",
            "template",
            "risk_per_trade",
            "risk_tier_floor",
            "sideway_risk_multiplier",
            "strong_volatility_risk_multiplier",
            "anti_martingale_factor",
            "consecutive_loss_pause_count",
            "consecutive_loss_cooldown_bars",
            "daily_loss_limit_pct",
            "max_total_exposure_pct",
            "max_open_positions",
            "sideways_volatility_threshold",
            "strong_volatility_threshold",
        ]
    }

    # Re-run best with detailed trades output
    best_eval, best_trades = _evaluate_candidate(
        base_settings=base,
        fold_cache=fold_cache,
        candidate=best_candidate,
        phase2_mod=phase2_mod,
        initial_balance=args.initial_balance,
        target_net=args.target_net,
        dd_min=args.dd_min,
        dd_max=args.dd_max,
        min_trades_sideway=args.min_trades_sideway,
        min_trades_normal=args.min_trades_normal,
        min_trades_strong=args.min_trades_strong,
        keep_trades=True,
    )
    best = best_eval

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = Path(f"outputs/{args.out_prefix}_{stamp}.csv")
    out_json = Path(f"outputs/{args.out_prefix}_{stamp}.json")
    out_trades = Path(f"outputs/{args.out_prefix}_{stamp}_best_trades.csv")

    res_df.sort_values(
        ["meets_target", "sum_net_profit", "global_max_drawdown_pct"],
        ascending=[False, False, True],
    ).to_csv(out_csv, index=False)
    if not best_trades.empty:
        best_trades.to_csv(out_trades, index=False)

    payload = {
        "mode": mode,
        "search": {
            "candidates": int(len(res_df)),
            "dataset_rows": int(len(dataset)),
            "folds": int(len(fold_cache)),
            "test_start": fold_cache[0].test_start,
            "test_end": fold_cache[-1].test_end,
            "elapsed_seconds": round(time.time() - t0, 2),
            "constraints": {
                "target_net": args.target_net,
                "dd_min": args.dd_min,
                "dd_max": args.dd_max,
                "min_trades_sideway": args.min_trades_sideway,
                "min_trades_normal": args.min_trades_normal,
                "min_trades_strong": args.min_trades_strong,
            },
        },
        "feasible_count": int(len(feasible)),
        "best": best,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("-" * 96, flush=True)
    print(
        f"Best [{mode}] net=${best['sum_net_profit']:,.2f} dd={best['global_max_drawdown_pct']:.2f}% "
        f"pf={best['profit_factor_global']:.3f} "
        f"reg0/1/2={best['regime0_trades']}/{best['regime1_trades']}/{best['regime2_trades']}",
        flush=True,
    )
    print(f"saved {out_csv}", flush=True)
    print(f"saved {out_json}", flush=True)
    if out_trades.exists():
        print(f"saved {out_trades}", flush=True)
    print("=" * 96, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
