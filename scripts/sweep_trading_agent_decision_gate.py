from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_trading_agent_decision_gate import _baseline_fold, _prepare, replay_agent
from xauusd_ai.agents.trading_agent_decision_gate import TradingAgentDecisionGate


def _score(frame: pd.DataFrame, deposit: float) -> float:
    median_final = float(frame["final"].median())
    min_final = float(frame["final"].min())
    loss_folds = int((frame["final"] < deposit).sum())
    worst_dd = abs(float(frame["worst_dd_pct"].min()))
    return median_final + 0.20 * min_final - 300.0 * loss_folds - 8.0 * worst_dd


def _summary(frame: pd.DataFrame, deposit: float) -> dict[str, Any]:
    return {
        "median_final": round(float(frame["final"].median()), 2),
        "min_final": round(float(frame["final"].min()), 2),
        "mean_final": round(float(frame["final"].mean()), 2),
        "worst_dd_pct": round(float(frame["worst_dd_pct"].min()), 2),
        "loss_folds": int((frame["final"] < deposit).sum()),
        "total_trades": int(frame["trades"].sum()),
        "total_skip": int(frame.get("skip", pd.Series(dtype=float)).sum()) if "skip" in frame else 0,
        "total_reduce": int(frame.get("reduce", pd.Series(dtype=float)).sum()) if "reduce" in frame else 0,
        "total_boost": int(frame.get("boost", pd.Series(dtype=float)).sum()) if "boost" in frame else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep deterministic TradingAgentDecisionGate parameters.")
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=1200.0)
    parser.add_argument("--min-history-folds", type=int, default=12)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades = _prepare(pd.read_csv(args.trades))

    baseline_rows = []
    for fold in sorted(trades["fold"].unique()):
        item = _baseline_fold(trades[trades["fold"] == fold], args.deposit)
        item["fold"] = int(fold)
        baseline_rows.append(item)
    baseline = pd.DataFrame(baseline_rows)
    baseline_summary = _summary(baseline, args.deposit)

    grid = {
        "min_memory_trades": [25, 50, 80],
        "bad_expectancy_usd": [-5.0, -10.0],
        "good_expectancy_usd": [20.0, 40.0, 9999.0],
        "bad_win_rate": [0.42],
        "reduce_multiplier": [0.75, 1.0],
        "boost_multiplier": [1.0, 1.1],
        "dd_reduce_pct": [-12.0],
        "dd_skip_pct": [-20.0],
    }

    rows: list[dict[str, Any]] = []
    best_agent: pd.DataFrame | None = None
    best_decisions: pd.DataFrame | None = None
    best_score = float("-inf")
    best_params: dict[str, Any] = {}

    for values in product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        gate = TradingAgentDecisionGate(**params)
        agent, decisions = replay_agent(trades, args.deposit, gate, args.min_history_folds)
        summary = _summary(agent, args.deposit)
        score = _score(agent, args.deposit)
        row = {
            **params,
            "score": round(score, 4),
            **summary,
            "delta_median_final": round(summary["median_final"] - baseline_summary["median_final"], 2),
            "delta_min_final": round(summary["min_final"] - baseline_summary["min_final"], 2),
            "delta_worst_dd_pct": round(summary["worst_dd_pct"] - baseline_summary["worst_dd_pct"], 2),
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best_agent = agent
            best_decisions = decisions
            best_params = params

    sweep = pd.DataFrame(rows).sort_values(["score", "median_final"], ascending=[False, False])
    sweep.to_csv(args.out_dir / "trading_agent_gate_sweep.csv", index=False)
    if best_agent is not None:
        best_agent.to_csv(args.out_dir / "best_agent_by_fold.csv", index=False)
    if best_decisions is not None:
        best_decisions.to_csv(args.out_dir / "best_agent_decisions_all_trades.csv", index=False)

    report = {
        "folds": int(trades["fold"].nunique()),
        "min_history_folds": int(args.min_history_folds),
        "baseline": baseline_summary,
        "best_params": best_params,
        "best": sweep.iloc[0].to_dict() if not sweep.empty else {},
        "top_10": sweep.head(10).to_dict(orient="records"),
    }
    (args.out_dir / "trading_agent_gate_sweep_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["best"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
