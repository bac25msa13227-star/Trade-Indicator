#!/usr/bin/env python3
"""
Breakthrough WF optimizer:
  - broad random exploration
  - local mutation refinement around best regions
Goal:
  maximize net profit while enforcing DD <= cap (default 25%).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from xauusd_ai.config import load_settings


def _load_phase2_module() -> Any:
    mod_path = Path(__file__).with_name("wf_phase2_optimizer.py")
    spec = importlib.util.spec_from_file_location("wf_phase2_optimizer", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WF breakthrough optimizer (net target + DD cap).")
    p.add_argument("--config", type=Path, default=Path("configs/snapshots/acc2_phase2_dd19_compat_20260401.yaml"))
    p.add_argument("--target-net", type=float, default=20_000.0)
    p.add_argument("--dd-max", type=float, default=25.0)
    p.add_argument("--initial-balance", type=float, default=200.0)
    p.add_argument("--explore-candidates", type=int, default=3000)
    p.add_argument("--refine-candidates", type=int, default=2500)
    p.add_argument("--topk-seeds", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260401)
    p.add_argument("--out-prefix", type=str, default="wf_breakthrough_dd25_acc2")
    return p.parse_args()


def _search_space(templates: list[str]) -> dict[str, list[Any]]:
    return {
        "threshold": [0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80],
        "min_strategy_score": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        "sideway_min_strategy_score": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "strong_volatility_min_strategy_score": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        "risk_min_conf": [0.50, 0.54, 0.58, 0.62, 0.66, 0.70],
        "sideway_min_conf": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        "volatile_min_conf": [0.55, 0.60, 0.64, 0.68, 0.72, 0.76],
        "require_trend_alignment": [False, True],
        "template": templates,
        "risk_per_trade": [0.020, 0.024, 0.028, 0.032, 0.036, 0.040, 0.044, 0.048, 0.052, 0.056],
        "risk_tier_floor": [0.0, 0.008, 0.014, 0.020, 0.026, 0.030],
        "sideway_risk_multiplier": [0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00],
        "strong_volatility_risk_multiplier": [0.40, 0.50, 0.60, 0.70, 0.80, 1.00],
        "anti_martingale_factor": [0.20, 0.30, 0.50, 0.70, 0.80, 1.00],
        "consecutive_loss_pause_count": [0, 2, 3, 4, 5],
        "consecutive_loss_cooldown_bars": [0, 4, 8, 12, 16],
        "daily_loss_limit_pct": [0.0, 0.01, 0.02, 0.03, 0.05],
        "max_total_exposure_pct": [0.08, 0.10, 0.12, 0.15, 0.20, 0.30],
        "max_open_positions": [1, 2, 3, 4],
    }


def _candidate_key(c: dict[str, Any], dims: list[str]) -> tuple[Any, ...]:
    return tuple(c[d] for d in dims)


def _sample_unique_candidates(
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
            key = _candidate_key(c, dims)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)

    while len(out) < n:
        c = {d: rng.choice(space[d]) for d in dims}
        key = _candidate_key(c, dims)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _score_row(row: dict[str, Any], dd_cap: float) -> float:
    net = float(row["sum_net_profit"])
    dd = float(row["global_max_drawdown_pct"])
    pf = float(row["profit_factor_global"])
    trades = float(row["total_trades"])
    if dd <= dd_cap:
        return net + (pf - 1.0) * 900.0 + min(trades, 800.0) * 1.1
    # hard penalty outside DD cap
    return net - (dd - dd_cap) * 1200.0


def _adjacent_mutation(
    base: dict[str, Any],
    space: dict[str, list[Any]],
    rng: random.Random,
) -> dict[str, Any]:
    c = dict(base)
    dims = list(space.keys())
    n_edits = 1 if rng.random() < 0.70 else 2
    picks = rng.sample(dims, k=n_edits)
    for dim in picks:
        vals = space[dim]
        cur = c[dim]
        if cur not in vals:
            c[dim] = rng.choice(vals)
            continue
        idx = vals.index(cur)
        neigh = [idx]
        if idx > 0:
            neigh.append(idx - 1)
        if idx < len(vals) - 1:
            neigh.append(idx + 1)
        c[dim] = vals[rng.choice(neigh)]
    return c


def main() -> int:
    args = parse_args()
    t0 = time.time()
    rng = random.Random(args.seed)
    mod = _load_phase2_module()

    settings = load_settings(args.config)
    # Keep architecture baseline fixed; only optimize router + risk.
    settings.strategy.ict_weight = 0.35
    settings.strategy.wyckoff_weight = 0.35
    settings.strategy.momentum_weight = 0.40
    settings.training.label_horizon = 12
    settings.training.min_return_threshold = 0.0008

    templates_dict = mod._time_templates()  # noqa: SLF001
    templates = list(templates_dict.keys())
    space = _search_space(templates)
    dims = list(space.keys())

    print("=" * 96, flush=True)
    print("WF BREAKTHROUGH OPTIMIZER", flush=True)
    print(f"Config      : {args.config}", flush=True)
    print(
        f"Target      : net >= ${args.target_net:,.2f} | DD <= {args.dd_max:.2f}% "
        f"| init=${args.initial_balance:,.2f}",
        flush=True,
    )
    print(
        f"Search      : explore={args.explore_candidates} | refine={args.refine_candidates} | topk={args.topk_seeds}",
        flush=True,
    )
    print("=" * 96, flush=True)

    fold_cache, dataset = mod._build_fold_cache(settings)  # noqa: SLF001
    print(
        f"Dataset     : rows={len(dataset):,} | folds={len(fold_cache)} "
        f"| test=[{fold_cache[0].test_start} -> {fold_cache[-1].test_end}]",
        flush=True,
    )

    # Warm-start around current strong configs.
    warm = [
        {
            "threshold": 0.72,
            "min_strategy_score": 0.40,
            "sideway_min_strategy_score": 0.15,
            "strong_volatility_min_strategy_score": 0.20,
            "risk_min_conf": 0.62,
            "sideway_min_conf": 0.70,
            "volatile_min_conf": 0.64,
            "require_trend_alignment": True,
            "template": "block_3_17",
            "risk_per_trade": 0.036,
            "risk_tier_floor": 0.014,
            "sideway_risk_multiplier": 0.35,
            "strong_volatility_risk_multiplier": 0.60,
            "anti_martingale_factor": 0.50,
            "consecutive_loss_pause_count": 3,
            "consecutive_loss_cooldown_bars": 8,
            "daily_loss_limit_pct": 0.03,
            "max_total_exposure_pct": 0.08,
            "max_open_positions": 3,
        },
        {
            "threshold": 0.76,
            "min_strategy_score": 0.40,
            "sideway_min_strategy_score": 0.15,
            "strong_volatility_min_strategy_score": 0.20,
            "risk_min_conf": 0.62,
            "sideway_min_conf": 0.70,
            "volatile_min_conf": 0.64,
            "require_trend_alignment": True,
            "template": "block_3_17",
            "risk_per_trade": 0.056,
            "risk_tier_floor": 0.022,
            "sideway_risk_multiplier": 0.40,
            "strong_volatility_risk_multiplier": 0.65,
            "anti_martingale_factor": 0.50,
            "consecutive_loss_pause_count": 3,
            "consecutive_loss_cooldown_bars": 8,
            "daily_loss_limit_pct": 0.0,
            "max_total_exposure_pct": 0.10,
            "max_open_positions": 4,
        },
    ]

    print("[1/2] Broad exploration...", flush=True)
    explore = _sample_unique_candidates(space, args.explore_candidates, rng, warm=warm)

    rows: list[dict[str, Any]] = []
    best_feasible: dict[str, Any] | None = None
    best_any: dict[str, Any] | None = None
    seen = {_candidate_key(c, dims) for c in explore}

    def _maybe_update_best(result: dict[str, Any]) -> None:
        nonlocal best_any, best_feasible
        if best_any is None or float(result["sum_net_profit"]) > float(best_any["sum_net_profit"]):
            best_any = result
        if float(result["global_max_drawdown_pct"]) <= args.dd_max:
            if best_feasible is None:
                best_feasible = result
            else:
                if (
                    float(result["sum_net_profit"]) > float(best_feasible["sum_net_profit"])
                    or (
                        float(result["sum_net_profit"]) == float(best_feasible["sum_net_profit"])
                        and float(result["profit_factor_global"]) > float(best_feasible["profit_factor_global"])
                    )
                ):
                    best_feasible = result

    for idx, cand in enumerate(explore, start=1):
        res, _ = mod._evaluate_candidate(  # noqa: SLF001
            base_settings=settings,
            fold_cache=fold_cache,
            candidate=cand,
            templates=templates_dict,
            initial_balance=args.initial_balance,
            dd_min=0.0,
            dd_max=args.dd_max,
            target_net=args.target_net,
            keep_trades=False,
        )
        res["phase"] = "explore"
        res["bt_score"] = round(_score_row(res, args.dd_max), 4)
        rows.append(res)
        _maybe_update_best(res)

        if idx % 40 == 0:
            print(
                f"  explore {idx}/{len(explore)} | net=${res['sum_net_profit']:,.2f} "
                f"dd={res['global_max_drawdown_pct']:.2f}% pf={res['profit_factor_global']:.3f}",
                flush=True,
            )

    ranked = sorted(rows, key=lambda r: float(r["bt_score"]), reverse=True)[: args.topk_seeds]
    seeds = [dict(r) for r in ranked]

    print("[2/2] Local refinement...", flush=True)
    refined: list[dict[str, Any]] = []
    attempts = 0
    while len(refined) < args.refine_candidates:
        attempts += 1
        parent = rng.choice(seeds)
        child = _adjacent_mutation(parent, space, rng)
        key = _candidate_key(child, dims)
        if key in seen:
            if attempts % 500 == 0:
                # inject random to avoid stalling
                child = {d: rng.choice(space[d]) for d in dims}
                key = _candidate_key(child, dims)
                if key in seen:
                    continue
            else:
                continue
        seen.add(key)
        refined.append(child)

    for idx, cand in enumerate(refined, start=1):
        res, _ = mod._evaluate_candidate(  # noqa: SLF001
            base_settings=settings,
            fold_cache=fold_cache,
            candidate=cand,
            templates=templates_dict,
            initial_balance=args.initial_balance,
            dd_min=0.0,
            dd_max=args.dd_max,
            target_net=args.target_net,
            keep_trades=False,
        )
        res["phase"] = "refine"
        res["bt_score"] = round(_score_row(res, args.dd_max), 4)
        rows.append(res)
        _maybe_update_best(res)

        if idx % 50 == 0 or res["meets_target"]:
            tag = "✅" if res["meets_target"] else "…"
            print(
                f"  refine {idx}/{len(refined)}{tag} | net=${res['sum_net_profit']:,.2f} "
                f"dd={res['global_max_drawdown_pct']:.2f}% pf={res['profit_factor_global']:.3f} tpl={res['template']}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    feasible = df[(df["sum_net_profit"] >= args.target_net) & (df["global_max_drawdown_pct"] <= args.dd_max)].copy()
    dd_ok = df[df["global_max_drawdown_pct"] <= args.dd_max].copy()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.out_prefix}_{stamp}.csv"
    out_json = out_dir / f"{args.out_prefix}_{stamp}.json"

    df.sort_values(["sum_net_profit"], ascending=False).to_csv(out_csv, index=False)

    payload = {
        "target": {
            "sum_net_profit_min": args.target_net,
            "dd_max_pct": args.dd_max,
            "initial_balance": args.initial_balance,
        },
        "search": {
            "config": str(args.config),
            "seed": int(args.seed),
            "explore_candidates": int(args.explore_candidates),
            "refine_candidates": int(args.refine_candidates),
            "evaluated_total": int(len(df)),
            "dataset_rows": int(len(dataset)),
            "folds": int(len(fold_cache)),
            "test_start": fold_cache[0].test_start,
            "test_end": fold_cache[-1].test_end,
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "counts": {
            "feasible_count": int(len(feasible)),
            "dd_ok_count": int(len(dd_ok)),
            "net_ge_target_count": int((df["sum_net_profit"] >= args.target_net).sum()),
        },
        "best_any": best_any,
        "best_dd_ok": best_feasible,
        "top10_net": df.sort_values("sum_net_profit", ascending=False).head(10).to_dict(orient="records"),
        "top10_dd_ok_by_net": dd_ok.sort_values("sum_net_profit", ascending=False).head(10).to_dict(orient="records"),
        "top10_feasible": feasible.sort_values("sum_net_profit", ascending=False).head(10).to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("-" * 96, flush=True)
    print(
        f"DONE evaluated={len(df)} feasible={len(feasible)} "
        f"best_any=${float(best_any['sum_net_profit']) if best_any else 0:,.2f} "
        f"best_dd_ok=${float(best_feasible['sum_net_profit']) if best_feasible else 0:,.2f}",
        flush=True,
    )
    print(f"saved: {out_csv}", flush=True)
    print(f"saved: {out_json}", flush=True)
    print("-" * 96, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
