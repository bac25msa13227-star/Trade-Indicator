#!/usr/bin/env python3
"""
Local DD tuner around a known high-net candidate.

Use case:
  Start from a candidate with strong net but high DD, then search nearby
  risk/filter controls to reduce DD while preserving as much net as possible.
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
    p = argparse.ArgumentParser(description="Local DD tune around high-net candidate")
    p.add_argument("--config", type=Path, default=Path("configs/_exp_acc2_2003.yaml"))
    p.add_argument("--target-net", type=float, default=6000.0)
    p.add_argument("--initial-balance", type=float, default=200.0)
    p.add_argument("--n-candidates", type=int, default=1200)
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--out-prefix", type=str, default="wf_phase2_local_dd_acc2")
    return p.parse_args()


def _build_candidates(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    # Base candidate (from phase2 result)
    base = {
        "threshold": 0.72,
        "min_strategy_score": 0.40,
        "sideway_min_strategy_score": 0.10,
        "strong_volatility_min_strategy_score": 0.25,
        "risk_min_conf": 0.58,
        "sideway_min_conf": 0.64,
        "volatile_min_conf": 0.62,
        "require_trend_alignment": False,
        "template": "focus_hours",
        "risk_per_trade": 0.034,
        "risk_tier_floor": 0.014,
        "sideway_risk_multiplier": 0.40,
        "strong_volatility_risk_multiplier": 0.65,
        "anti_martingale_factor": 0.80,
        "consecutive_loss_pause_count": 0,
        "consecutive_loss_cooldown_bars": 0,
        "daily_loss_limit_pct": 0.0,
        "max_total_exposure_pct": 0.08,
        "max_open_positions": 3,
    }

    search = {
        "threshold": [0.70, 0.72, 0.74],
        "min_strategy_score": [0.35, 0.40, 0.45],
        "sideway_min_strategy_score": [0.10, 0.15, 0.20],
        "strong_volatility_min_strategy_score": [0.20, 0.25, 0.30],
        "risk_min_conf": [0.58, 0.60, 0.62],
        "sideway_min_conf": [0.64, 0.66, 0.68, 0.70],
        "volatile_min_conf": [0.60, 0.62, 0.64],
        "require_trend_alignment": [False, True],
        "template": ["focus_hours", "mon_weak_cluster", "mon_tue_weak_cluster", "block_3_17"],
        "risk_per_trade": [0.024, 0.026, 0.028, 0.030, 0.032, 0.034, 0.036],
        "risk_tier_floor": [0.010, 0.014, 0.018, 0.022],
        "sideway_risk_multiplier": [0.20, 0.25, 0.30, 0.35, 0.40],
        "strong_volatility_risk_multiplier": [0.50, 0.55, 0.60, 0.65],
        "anti_martingale_factor": [0.50, 0.60, 0.70, 0.80],
        "consecutive_loss_pause_count": [0, 2, 3, 4],
        "consecutive_loss_cooldown_bars": [0, 4, 8, 12],
        "daily_loss_limit_pct": [0.0, 0.03, 0.05],
        "max_total_exposure_pct": [0.05, 0.08, 0.10],
        "max_open_positions": [2, 3],
    }

    # seeded deterministic hand-crafted ladder
    candidates: list[dict[str, Any]] = []
    for rpt in [0.028, 0.030, 0.032, 0.034]:
        for smr in [0.25, 0.30, 0.35, 0.40]:
            for vmr in [0.55, 0.60, 0.65]:
                c = dict(base)
                c["risk_per_trade"] = rpt
                c["sideway_risk_multiplier"] = smr
                c["strong_volatility_risk_multiplier"] = vmr
                c["anti_martingale_factor"] = 0.60
                c["consecutive_loss_pause_count"] = 3
                c["consecutive_loss_cooldown_bars"] = 8
                c["daily_loss_limit_pct"] = 0.05
                c["max_total_exposure_pct"] = 0.08
                c["max_open_positions"] = 2
                candidates.append(c)

    dims = list(search.keys())
    seen = {tuple(c[d] for d in dims) for c in candidates}
    while len(candidates) < n:
        c = {k: rng.choice(v) for k, v in search.items()}
        key = tuple(c[d] for d in dims)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(c)
    return candidates


def main() -> int:
    args = parse_args()
    t0 = time.time()
    mod = _load_phase2_module()

    settings = load_settings(args.config)
    settings.strategy.ict_weight = 0.35
    settings.strategy.wyckoff_weight = 0.35
    settings.strategy.momentum_weight = 0.40
    settings.training.label_horizon = 12
    settings.training.min_return_threshold = 0.0008

    print("=" * 88, flush=True)
    print("LOCAL DD TUNE (around high-net candidate)", flush=True)
    print(f"Config      : {args.config}", flush=True)
    print(f"Target net  : ${args.target_net:,.2f}", flush=True)
    print(f"Candidates  : {args.n_candidates} (seed={args.seed})", flush=True)
    print("=" * 88, flush=True)

    templates = mod._time_templates()
    fold_cache, dataset = mod._build_fold_cache(settings)
    print(
        f"      test_window=[{fold_cache[0].test_start} -> {fold_cache[-1].test_end}] folds={len(fold_cache)}",
        flush=True,
    )
    print(f"      dataset_rows={len(dataset):,}", flush=True)

    candidates = _build_candidates(args.n_candidates, args.seed)
    print(f"[3/3] Evaluating {len(candidates)} local candidates...", flush=True)

    rows: list[dict[str, Any]] = []
    best_dd20 = None
    best_dd25 = None
    best_dd30 = None

    for i, c in enumerate(candidates, start=1):
        r, _ = mod._evaluate_candidate(
            base_settings=settings,
            fold_cache=fold_cache,
            candidate=c,
            templates=templates,
            initial_balance=args.initial_balance,
            dd_min=15.0,
            dd_max=20.0,
            target_net=args.target_net,
            keep_trades=False,
        )
        rows.append(r)
        net = float(r["sum_net_profit"])
        dd = float(r["global_max_drawdown_pct"])

        if net >= args.target_net and dd <= 20.0:
            if best_dd20 is None or net > float(best_dd20["sum_net_profit"]):
                best_dd20 = r
        if net >= args.target_net and dd <= 25.0:
            if best_dd25 is None or (dd < float(best_dd25["global_max_drawdown_pct"]) or (dd == float(best_dd25["global_max_drawdown_pct"]) and net > float(best_dd25["sum_net_profit"]))):
                best_dd25 = r
        if net >= args.target_net and dd <= 30.0:
            if best_dd30 is None or (dd < float(best_dd30["global_max_drawdown_pct"]) or (dd == float(best_dd30["global_max_drawdown_pct"]) and net > float(best_dd30["sum_net_profit"]))):
                best_dd30 = r

        if i % 25 == 0:
            print(
                f"      [{i}/{len(candidates)}] net=${net:,.2f} dd={dd:.2f}% pf={float(r['profit_factor_global']):.3f} tpl={r['template']}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"{args.out_prefix}_{stamp}.csv"
    out_json = out_dir / f"{args.out_prefix}_{stamp}.json"
    df.sort_values(["sum_net_profit"], ascending=False).to_csv(out_csv, index=False)

    dd20 = df[(df["sum_net_profit"] >= args.target_net) & (df["global_max_drawdown_pct"] <= 20.0)].copy()
    dd25 = df[(df["sum_net_profit"] >= args.target_net) & (df["global_max_drawdown_pct"] <= 25.0)].copy()
    dd30 = df[(df["sum_net_profit"] >= args.target_net) & (df["global_max_drawdown_pct"] <= 30.0)].copy()

    payload = {
        "search": {
            "candidates": int(len(df)),
            "dataset_rows": int(len(dataset)),
            "folds": int(len(fold_cache)),
            "test_start": fold_cache[0].test_start,
            "test_end": fold_cache[-1].test_end,
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "target_net": float(args.target_net),
        "counts": {
            "net_ge_target_dd20": int(len(dd20)),
            "net_ge_target_dd25": int(len(dd25)),
            "net_ge_target_dd30": int(len(dd30)),
        },
        "best_dd20_by_net": best_dd20,
        "best_dd25_lowest_dd": best_dd25,
        "best_dd30_lowest_dd": best_dd30,
        "top_dd_band_15_20": [
            {k: (float(v) if hasattr(v, "__float__") else v) for k, v in r.items()}
            for r in df[(df["global_max_drawdown_pct"] >= 15.0) & (df["global_max_drawdown_pct"] <= 20.0)]
            .sort_values("sum_net_profit", ascending=False)
            .head(10)
            .to_dict(orient="records")
        ],
        "top_net_ge_target_low_dd": [
            {k: (float(v) if hasattr(v, "__float__") else v) for k, v in r.items()}
            for r in df[df["sum_net_profit"] >= args.target_net]
            .sort_values("global_max_drawdown_pct", ascending=True)
            .head(10)
            .to_dict(orient="records")
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("-" * 88, flush=True)
    print(f"net>=target & dd<=20: {len(dd20)}", flush=True)
    print(f"net>=target & dd<=25: {len(dd25)}", flush=True)
    print(f"net>=target & dd<=30: {len(dd30)}", flush=True)
    if best_dd30 is not None:
        print(
            "best_dd30_lowest_dd: "
            f"net=${float(best_dd30['sum_net_profit']):,.2f} "
            f"dd={float(best_dd30['global_max_drawdown_pct']):.2f}% "
            f"pf={float(best_dd30['profit_factor_global']):.3f}",
            flush=True,
        )
    print(f"saved: {out_csv}", flush=True)
    print(f"saved: {out_json}", flush=True)
    print("-" * 88, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
