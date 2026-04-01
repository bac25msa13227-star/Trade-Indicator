#!/usr/bin/env python3
"""
Walk-forward constrained optimizer.

Goal:
  Maximize net profit over continuous 5-month walk-forward while enforcing
  a drawdown constraint band (default 15%–20%).

Design highlights:
  - Build dataset once, train fold models once.
  - Cache fold probabilities to avoid retraining per candidate.
  - Evaluate each candidate with dynamic concurrent simulator.
  - Carry fold ending balance -> next fold starting balance (continuous equity).
  - Apply regime-aware router knobs (confidence/risk multipliers/time templates).
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

from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import Settings, load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.dataset import build_merged_context, prepare_training_dataset
from xauusd_ai.model.trainer import ModelTrainer
from xauusd_ai.strategies.hybrid import HybridStrategy


@dataclass
class FoldCache:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    predictions: pd.DataFrame


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
        "block_2_7_17_21": {
            "blocked_hours_utc": sorted(set([15, 22, 23] + list(range(2, 8)) + list(range(17, 22)))),
            "blocked_weekday_hours_utc": {},
            "allowed_weekday_hours_utc": {},
        },
        "mon_weak_cluster": {
            "blocked_hours_utc": [15, 22, 23],
            "blocked_weekday_hours_utc": {"Monday": sorted(set(list(range(2, 8)) + list(range(17, 22))))},
            "allowed_weekday_hours_utc": {},
        },
        "mon_tue_weak_cluster": {
            "blocked_hours_utc": [15, 22, 23],
            "blocked_weekday_hours_utc": {
                "Monday": sorted(set(list(range(2, 8)) + list(range(17, 22)))),
                "Tuesday": sorted(set(list(range(2, 8)) + list(range(17, 22)))),
            },
            "allowed_weekday_hours_utc": {},
        },
        "killzones_only": {
            "blocked_hours_utc": [],
            "blocked_weekday_hours_utc": {},
            "allowed_weekday_hours_utc": {w: sorted(set(range(6, 17))) for w in weekdays},
        },
        "focus_hours": {
            "blocked_hours_utc": [],
            "blocked_weekday_hours_utc": {},
            "allowed_weekday_hours_utc": {w: [0, 1, 5, 6, 7, 13, 14, 20, 21] for w in weekdays},
        },
    }


def _build_fold_cache(base_settings: Settings) -> tuple[list[FoldCache], pd.DataFrame]:
    print("[1/4] Loading frames + building dataset once...", flush=True)
    service = MarketDataService(base_settings)
    frames = service.fetch_multi_timeframe_data(
        source=base_settings.market.training_data_source,
        all_bars=True,
    )
    merged = build_merged_context(base_settings, frames)
    dataset = prepare_training_dataset(
        base_settings,
        frames,
        HybridStrategy(base_settings),
        cached_merged=merged,
    )
    print(f"      dataset_rows={len(dataset):,}", flush=True)

    train_size = base_settings.training.walkforward_train_size
    test_size = base_settings.training.walkforward_test_size
    step_size = base_settings.training.walkforward_step_size
    max_folds = base_settings.training.walkforward_max_folds_per_combination or 8
    fold_indices = list(range(0, max(len(dataset) - train_size - test_size + 1, 0), step_size))
    if max_folds > 0:
        fold_indices = fold_indices[-max_folds:]

    print("[2/4] Training fold models once + caching probabilities...", flush=True)
    cached: list[FoldCache] = []
    for idx, fold_start in enumerate(fold_indices, start=1):
        train_end = fold_start + train_size
        test_end = train_end + test_size
        fold_train = dataset.iloc[fold_start:train_end].copy()
        fold_test = dataset.iloc[train_end:test_end].copy()
        if len(fold_train) < 200 or len(fold_test) < 50:
            continue

        fold_dataset = pd.concat([fold_train, fold_test], ignore_index=True)
        fold_dataset["split"] = "train"
        fold_dataset.loc[len(fold_train):, "split"] = "test"

        trainer = ModelTrainer(base_settings)
        trainer.train(fold_dataset, save_artifacts=False)
        preds = trainer.predict_dataset(fold_dataset)

        cached.append(
            FoldCache(
                fold=idx,
                train_start=str(fold_train["time"].min()),
                train_end=str(fold_train["time"].max()),
                test_start=str(fold_test["time"].min()),
                test_end=str(fold_test["time"].max()),
                predictions=preds.copy(),
            )
        )
        print(
            f"      cached fold {idx}/{len(fold_indices)} "
            f"test=[{cached[-1].test_start} -> {cached[-1].test_end}] rows={len(preds)}",
            flush=True,
        )

    if not cached:
        raise RuntimeError("No valid folds available for optimization.")

    return cached, dataset


def _apply_candidate(settings: Settings, candidate: dict[str, Any], templates: dict[str, dict[str, Any]]) -> Settings:
    s = settings.model_copy(deep=True)

    s.strategy.signal_threshold = float(candidate["threshold"])
    s.risk.risk_per_trade = float(candidate["risk_per_trade"])
    s.risk.max_risk_fraction = min(
        0.10,
        max(float(s.risk.max_risk_fraction), float(candidate["risk_per_trade"]) * 1.8),
    )
    s.risk.risk_tier_floor = float(candidate["risk_tier_floor"])

    s.strategy.min_strategy_score = float(candidate["min_strategy_score"])
    s.strategy.sideway_min_strategy_score = float(candidate["sideway_min_strategy_score"])
    s.strategy.strong_volatility_min_strategy_score = float(candidate["strong_volatility_min_strategy_score"])
    s.strategy.require_trend_alignment = bool(candidate["require_trend_alignment"])

    s.risk.min_confidence = float(candidate["risk_min_conf"])
    s.strategy.sideway_min_confidence = float(candidate["sideway_min_conf"])
    s.strategy.volatile_min_confidence = float(candidate["volatile_min_conf"])

    s.risk.sideway_risk_multiplier = float(candidate["sideway_risk_multiplier"])
    s.risk.strong_volatility_risk_multiplier = float(candidate["strong_volatility_risk_multiplier"])

    s.risk.anti_martingale_factor = float(candidate["anti_martingale_factor"])
    s.risk.consecutive_loss_pause_count = int(candidate["consecutive_loss_pause_count"])
    s.risk.consecutive_loss_cooldown_bars = int(candidate["consecutive_loss_cooldown_bars"])
    s.risk.daily_loss_limit_pct = float(candidate["daily_loss_limit_pct"])
    s.risk.kill_switch_enabled = True

    tpl = templates[str(candidate["template"])]
    s.strategy.blocked_hours_utc = list(tpl["blocked_hours_utc"])
    s.strategy.blocked_weekday_hours_utc = dict(tpl["blocked_weekday_hours_utc"])
    s.strategy.allowed_weekday_hours_utc = dict(tpl["allowed_weekday_hours_utc"])
    return s


def _evaluate_candidate(
    base_settings: Settings,
    fold_cache: list[FoldCache],
    candidate: dict[str, Any],
    templates: dict[str, dict[str, Any]],
    initial_balance: float,
    dd_min: float,
    dd_max: float,
    target_net: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    tuned = _apply_candidate(base_settings, candidate, templates)
    balance = float(initial_balance)

    all_trades: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    equity_curve: list[float] = [balance]

    for fold in fold_cache:
        fold_settings = tuned.model_copy(deep=True)
        fold_settings.training.backtest_initial_balance = balance

        preds = fold.predictions.copy()
        preds["prediction"] = (preds["probability"] >= float(candidate["threshold"])).astype(int)

        sim = simulate_dynamic_concurrent_backtest(
            preds,
            fold_settings,
            RiskManager(fold_settings),
            label="test",
            compound=True,
        )
        rep = sim.report
        balance = float(rep["ending_balance"])
        equity_curve.append(balance)

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
            t = sim.trades.copy()
            t["fold"] = fold.fold
            t["test_start"] = fold.test_start
            t["test_end"] = fold.test_end
            all_trades.append(t)
            bal_after = pd.to_numeric(t["balance_after"], errors="coerce").dropna().tolist()
            if bal_after:
                equity_curve.extend(float(v) for v in bal_after)

    fold_df = pd.DataFrame(fold_rows)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    gross_profit = float(fold_df["gross_profit"].sum()) if not fold_df.empty else 0.0
    gross_loss = float(fold_df["gross_loss"].sum()) if not fold_df.empty else 0.0
    net_profit = balance - initial_balance
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    max_dd_global_pct = _max_drawdown_pct_from_curve(equity_curve)
    avg_fold_dd_abs = float(fold_df["max_drawdown_pct"].abs().mean()) if not fold_df.empty else 0.0
    total_trades = int(fold_df["trades"].sum()) if not fold_df.empty else 0
    total_wins = int(fold_df["wins"].sum()) if not fold_df.empty else 0
    total_losses = int(fold_df["losses"].sum()) if not fold_df.empty else 0
    win_rate = total_wins / max(total_wins + total_losses, 1)

    meets = bool(net_profit >= target_net and dd_min <= max_dd_global_pct <= dd_max)
    net_short = max(0.0, target_net - net_profit)
    dd_dist = 0.0 if dd_min <= max_dd_global_pct <= dd_max else min(
        abs(max_dd_global_pct - dd_min),
        abs(max_dd_global_pct - dd_max),
    )
    distance_score = net_short + dd_dist * 500.0

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
        "meets_target": meets,
        "distance_score": round(distance_score, 4),
    }
    return result, trades_df


def _candidate_space(templates: dict[str, dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        "threshold": [0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82],
        "risk_per_trade": [0.010, 0.014, 0.018, 0.022, 0.026, 0.030, 0.034, 0.038, 0.042],
        "risk_tier_floor": [0.0, 0.006, 0.010, 0.014, 0.018],
        "min_strategy_score": [0.20, 0.25, 0.30, 0.35, 0.40],
        "sideway_min_strategy_score": [0.10, 0.15, 0.20, 0.25, 0.30],
        "strong_volatility_min_strategy_score": [0.15, 0.20, 0.25, 0.30, 0.35],
        "risk_min_conf": [0.56, 0.58, 0.60, 0.62, 0.64],
        "sideway_min_conf": [0.60, 0.64, 0.68, 0.72],
        "volatile_min_conf": [0.54, 0.56, 0.58, 0.60, 0.62],
        "require_trend_alignment": [True, False],
        "sideway_risk_multiplier": [0.20, 0.30, 0.40, 0.50, 0.60],
        "strong_volatility_risk_multiplier": [0.45, 0.55, 0.65, 0.75],
        "anti_martingale_factor": [0.50, 0.60, 0.70, 0.80, 1.00],
        "consecutive_loss_pause_count": [0, 2, 3, 4],
        "consecutive_loss_cooldown_bars": [0, 4, 8, 12],
        "daily_loss_limit_pct": [0.0, 0.03, 0.05, 0.08],
        "template": list(templates.keys()),
    }


def _build_candidates(space: dict[str, list[Any]], n_candidates: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    dims = list(space.keys())
    seen: set[tuple[Any, ...]] = set()
    candidates: list[dict[str, Any]] = []

    targeted = []
    for thr in [0.74, 0.76, 0.78, 0.80]:
        for risk in [0.018, 0.022, 0.026, 0.030, 0.034]:
            for tpl in ["minimal", "block_3_17", "focus_hours", "killzones_only", "mon_weak_cluster"]:
                targeted.append(
                    {
                        "threshold": thr,
                        "risk_per_trade": risk,
                        "risk_tier_floor": 0.010,
                        "min_strategy_score": 0.30,
                        "sideway_min_strategy_score": 0.20,
                        "strong_volatility_min_strategy_score": 0.25,
                        "risk_min_conf": 0.60,
                        "sideway_min_conf": 0.68,
                        "volatile_min_conf": 0.58,
                        "require_trend_alignment": True,
                        "sideway_risk_multiplier": 0.30,
                        "strong_volatility_risk_multiplier": 0.55,
                        "anti_martingale_factor": 0.60,
                        "consecutive_loss_pause_count": 3,
                        "consecutive_loss_cooldown_bars": 8,
                        "daily_loss_limit_pct": 0.05,
                        "template": tpl,
                    }
                )

    for c in targeted:
        key = tuple(c[d] for d in dims)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(c)

    while len(candidates) < n_candidates:
        c = {d: rng.choice(space[d]) for d in dims}
        key = tuple(c[d] for d in dims)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(c)
    return candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Constrained WF optimizer (continuous equity).")
    p.add_argument("--config", type=Path, default=Path("configs/_exp_acc2_2003.yaml"))
    p.add_argument("--target-net", type=float, default=10_000.0)
    p.add_argument("--dd-min", type=float, default=15.0)
    p.add_argument("--dd-max", type=float, default=20.0)
    p.add_argument("--initial-balance", type=float, default=200.0)
    p.add_argument("--n-candidates", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", type=str, default="wf_constrained_acc2")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    base = load_settings(args.config)
    # Keep the current architecture baseline fixed; optimize router/risk/gates around it.
    base.strategy.ict_weight = 0.35
    base.strategy.wyckoff_weight = 0.35
    base.strategy.momentum_weight = 0.40
    base.training.label_horizon = 12
    base.training.min_return_threshold = 0.0008

    print("=" * 88, flush=True)
    print("CONSTRAINED WF OPTIMIZER (continuous-equity + regime router)", flush=True)
    print(f"Config       : {args.config}", flush=True)
    print(f"Target       : net >= ${args.target_net:,.2f} | DD in [{args.dd_min:.1f}%, {args.dd_max:.1f}%]", flush=True)
    print(f"Initial bal  : ${args.initial_balance:,.2f}", flush=True)
    print(f"Candidates   : {args.n_candidates} (seed={args.seed})", flush=True)
    print("=" * 88, flush=True)

    templates = _time_templates()
    fold_cache, dataset = _build_fold_cache(base)
    print(
        f"      test_window=[{fold_cache[0].test_start} -> {fold_cache[-1].test_end}] "
        f"folds={len(fold_cache)}",
        flush=True,
    )

    print("[3/4] Building candidate set...", flush=True)
    space = _candidate_space(templates)
    candidates = _build_candidates(space, args.n_candidates, args.seed)
    print(f"      candidate_count={len(candidates)}", flush=True)

    print("[4/4] Evaluating candidates...", flush=True)
    rows: list[dict[str, Any]] = []
    best_trades = pd.DataFrame()
    best_key: tuple[str, float] | None = None
    last_log = time.time()

    for idx, candidate in enumerate(candidates, start=1):
        result, trades = _evaluate_candidate(
            base_settings=base,
            fold_cache=fold_cache,
            candidate=candidate,
            templates=templates,
            initial_balance=args.initial_balance,
            dd_min=args.dd_min,
            dd_max=args.dd_max,
            target_net=args.target_net,
        )
        rows.append(result)

        current_key = (str(result["meets_target"]), float(result["distance_score"]))
        if best_key is None or current_key < best_key:
            best_key = current_key
            best_trades = trades

        if idx % 25 == 0 or result["meets_target"] or (time.time() - last_log > 20):
            tag = "✅" if result["meets_target"] else "…"
            print(
                f"      [{idx}/{len(candidates)}]{tag} "
                f"tpl={result['template']} thr={result['threshold']:.2f} risk={result['risk_per_trade']:.3f} "
                f"net=${result['sum_net_profit']:,.2f} dd={result['global_max_drawdown_pct']:.2f}%",
                flush=True,
            )
            last_log = time.time()

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("No optimization results generated.")

    feasible = res[res["meets_target"] == True].copy()  # noqa: E712
    if not feasible.empty:
        chosen = feasible.sort_values(["sum_net_profit", "profit_factor_global"], ascending=False).iloc[0]
        mode = "target_met"
    else:
        chosen = res.sort_values(["distance_score", "sum_net_profit"], ascending=[True, False]).iloc[0]
        mode = "closest_possible"

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.out_prefix}_{stamp}.csv"
    out_json = out_dir / f"{args.out_prefix}_{stamp}.json"
    out_trades = out_dir / f"{args.out_prefix}_{stamp}_best_trades.csv"

    res_sorted = res.sort_values(["meets_target", "sum_net_profit"], ascending=[False, False])
    res_sorted.to_csv(out_csv, index=False)

    if not best_trades.empty:
        best_trades.to_csv(out_trades, index=False)

    payload = {
        "target": {
            "sum_net_profit_min": args.target_net,
            "global_dd_band_pct": [args.dd_min, args.dd_max],
        },
        "search": {
            "evaluated": int(len(res)),
            "feasible_count": int(len(feasible)),
            "seed": int(args.seed),
            "n_candidates": int(args.n_candidates),
            "dataset_rows": int(len(dataset)),
            "folds": int(len(fold_cache)),
            "test_start": fold_cache[0].test_start,
            "test_end": fold_cache[-1].test_end,
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "mode": mode,
        "chosen": {k: (float(v) if hasattr(v, "__float__") else v) for k, v in chosen.to_dict().items()},
        "top10": [
            {k: (float(v) if hasattr(v, "__float__") else v) for k, v in row.items()}
            for row in res_sorted.head(10).to_dict(orient="records")
        ],
        "top_dd_band": [
            {k: (float(v) if hasattr(v, "__float__") else v) for k, v in row.items()}
            for row in res[
                (res["global_max_drawdown_pct"] >= args.dd_min)
                & (res["global_max_drawdown_pct"] <= args.dd_max)
            ]
            .sort_values("sum_net_profit", ascending=False)
            .head(10)
            .to_dict(orient="records")
        ],
        "top_net_10k": [
            {k: (float(v) if hasattr(v, "__float__") else v) for k, v in row.items()}
            for row in res[res["sum_net_profit"] >= args.target_net]
            .sort_values("global_max_drawdown_pct", ascending=True)
            .head(10)
            .to_dict(orient="records")
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("-" * 88, flush=True)
    print(f"mode={mode} feasible={len(feasible)}/{len(res)}", flush=True)
    print(
        "chosen: "
        f"net=${float(chosen['sum_net_profit']):,.2f} "
        f"dd={float(chosen['global_max_drawdown_pct']):.2f}% "
        f"pf={float(chosen['profit_factor_global']):.3f} "
        f"trades={int(chosen['total_trades'])}",
        flush=True,
    )
    print(f"saved: {out_csv}", flush=True)
    print(f"saved: {out_json}", flush=True)
    if not best_trades.empty:
        print(f"saved: {out_trades}", flush=True)
    print("-" * 88, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
