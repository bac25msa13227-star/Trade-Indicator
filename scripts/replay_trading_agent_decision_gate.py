from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from xauusd_ai.agents.trading_agent_decision_gate import TradingAgentDecisionGate


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["fold"] = pd.to_numeric(out["fold"], errors="coerce").astype(int)
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce").fillna(0.0)
    out["matched_probability"] = pd.to_numeric(out["matched_probability"], errors="coerce")
    out["side_key"] = out["side"].astype(str).str.strip().str.lower()
    out["session_key"] = out["open_time"].map(TradingAgentDecisionGate.session_from_time)
    out["confidence_bucket"] = out["matched_probability"].map(TradingAgentDecisionGate.confidence_bucket)
    return out.sort_values(["fold", "open_time", "open_ticket"]).reset_index(drop=True)


def _baseline_fold(trades: pd.DataFrame, deposit: float) -> dict[str, Any]:
    balance = float(deposit)
    peak = float(deposit)
    worst_dd = 0.0
    for pnl in trades["pnl"].astype(float):
        balance += pnl
        peak = max(peak, balance)
        worst_dd = min(worst_dd, (balance - peak) / peak * 100.0 if peak > 0 else 0.0)
    return {"final": round(balance, 2), "worst_dd_pct": round(worst_dd, 2), "trades": int(len(trades))}


def _summarize(frame: pd.DataFrame, deposit: float, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_median_final": round(float(frame["final"].median()), 2),
        f"{prefix}_min_final": round(float(frame["final"].min()), 2),
        f"{prefix}_mean_final": round(float(frame["final"].mean()), 2),
        f"{prefix}_worst_dd_pct": round(float(frame["worst_dd_pct"].min()), 2),
        f"{prefix}_loss_folds": int((frame["final"] < deposit).sum()),
        f"{prefix}_total_trades": int(frame["trades"].sum()),
    }


def replay_agent(trades: pd.DataFrame, deposit: float, gate: TradingAgentDecisionGate, min_history_folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    completed_prior: list[pd.DataFrame] = []
    folds = sorted(trades["fold"].unique())

    for idx, fold in enumerate(folds):
        fold_trades = trades[trades["fold"] == fold].copy()
        prior = pd.concat(completed_prior, ignore_index=True) if completed_prior else pd.DataFrame()
        use_agent = idx >= min_history_folds and not prior.empty

        balance = float(deposit)
        peak = float(deposit)
        worst_dd = 0.0
        action_counts: dict[str, int] = {"TAKE": 0, "SKIP": 0, "REDUCE": 0, "BOOST": 0}
        wins = 0
        losses = 0
        active = 0

        for _, trade in fold_trades.iterrows():
            current_dd = (balance - peak) / peak * 100.0 if peak > 0 else 0.0
            if use_agent:
                decision = gate.decide(
                    side=str(trade["side"]),
                    probability=None if pd.isna(trade["matched_probability"]) else float(trade["matched_probability"]),
                    open_time=trade["open_time"],
                    prior_trades=prior,
                    current_dd_pct=current_dd,
                )
            else:
                decision = gate.decide(
                    side=str(trade["side"]),
                    probability=None if pd.isna(trade["matched_probability"]) else float(trade["matched_probability"]),
                    open_time=trade["open_time"],
                    prior_trades=pd.DataFrame(),
                    current_dd_pct=current_dd,
                )
                decision = type(decision)(
                    action="TAKE",
                    risk_multiplier=1.0,
                    thesis=decision.thesis,
                    memory_note="bootstrap_baseline",
                    risk_note=decision.risk_note,
                )

            pnl = float(trade["pnl"]) * float(decision.risk_multiplier)
            action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
            if decision.risk_multiplier > 0:
                active += 1
                wins += int(pnl > 0)
                losses += int(pnl < 0)
            balance += pnl
            peak = max(peak, balance)
            dd = (balance - peak) / peak * 100.0 if peak > 0 else 0.0
            worst_dd = min(worst_dd, dd)

            row = trade.to_dict()
            row.update(
                {
                    "agent_action": decision.action,
                    "agent_risk_multiplier": round(float(decision.risk_multiplier), 4),
                    "agent_scaled_pnl": round(pnl, 2),
                    "agent_balance": round(balance, 2),
                    "agent_dd_pct": round(dd, 2),
                    "agent_thesis": decision.thesis,
                    "agent_memory_note": decision.memory_note,
                    "agent_risk_note": decision.risk_note,
                    "agent_enabled": bool(use_agent),
                }
            )
            decision_rows.append(row)

        fold_rows.append(
            {
                "fold": int(fold),
                "agent_enabled": bool(use_agent),
                "final": round(balance, 2),
                "net_pct": round((balance / deposit - 1.0) * 100.0, 2),
                "worst_dd_pct": round(worst_dd, 2),
                "trades": int(active),
                "raw_trades": int(len(fold_trades)),
                "wr_pct": round(wins / active * 100.0, 2) if active else 0.0,
                "take": action_counts.get("TAKE", 0),
                "skip": action_counts.get("SKIP", 0),
                "reduce": action_counts.get("REDUCE", 0),
                "boost": action_counts.get("BOOST", 0),
            }
        )
        completed_prior.append(fold_trades)

    return pd.DataFrame(fold_rows), pd.DataFrame(decision_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay TradingAgentDecisionGate using only prior-fold feedback.")
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=1200.0)
    parser.add_argument("--min-history-folds", type=int, default=12)
    parser.add_argument("--min-memory-trades", type=int, default=25)
    parser.add_argument("--bad-expectancy-usd", type=float, default=-3.0)
    parser.add_argument("--good-expectancy-usd", type=float, default=7.5)
    parser.add_argument("--bad-win-rate", type=float, default=0.42)
    parser.add_argument("--reduce-multiplier", type=float, default=0.50)
    parser.add_argument("--boost-multiplier", type=float, default=1.20)
    parser.add_argument("--dd-reduce-pct", type=float, default=-8.0)
    parser.add_argument("--dd-skip-pct", type=float, default=-14.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades = _prepare(pd.read_csv(args.trades))
    gate = TradingAgentDecisionGate(
        min_memory_trades=args.min_memory_trades,
        bad_expectancy_usd=args.bad_expectancy_usd,
        good_expectancy_usd=args.good_expectancy_usd,
        bad_win_rate=args.bad_win_rate,
        reduce_multiplier=args.reduce_multiplier,
        boost_multiplier=args.boost_multiplier,
        dd_reduce_pct=args.dd_reduce_pct,
        dd_skip_pct=args.dd_skip_pct,
    )

    baseline_rows = []
    for fold in sorted(trades["fold"].unique()):
        item = _baseline_fold(trades[trades["fold"] == fold], args.deposit)
        item["fold"] = int(fold)
        baseline_rows.append(item)
    baseline = pd.DataFrame(baseline_rows)
    agent, decisions = replay_agent(trades, args.deposit, gate, args.min_history_folds)

    comparison = baseline.rename(
        columns={"final": "baseline_final", "worst_dd_pct": "baseline_worst_dd_pct", "trades": "baseline_trades"}
    ).merge(agent, on="fold", how="inner", suffixes=("", "_agent"))
    comparison["delta_final"] = (comparison["final"] - comparison["baseline_final"]).round(2)
    comparison["delta_dd_pct"] = (comparison["worst_dd_pct"] - comparison["baseline_worst_dd_pct"]).round(2)

    baseline.to_csv(args.out_dir / "baseline_by_fold.csv", index=False)
    agent.to_csv(args.out_dir / "agent_by_fold.csv", index=False)
    comparison.to_csv(args.out_dir / "agent_vs_baseline_by_fold.csv", index=False)
    decisions.to_csv(args.out_dir / "agent_decisions_all_trades.csv", index=False)

    report = {
        "folds": int(len(agent)),
        "first_fold": int(agent["fold"].min()),
        "last_fold": int(agent["fold"].max()),
        "min_history_folds": int(args.min_history_folds),
        "gate_params": {
            "min_memory_trades": args.min_memory_trades,
            "bad_expectancy_usd": args.bad_expectancy_usd,
            "good_expectancy_usd": args.good_expectancy_usd,
            "bad_win_rate": args.bad_win_rate,
            "reduce_multiplier": args.reduce_multiplier,
            "boost_multiplier": args.boost_multiplier,
            "dd_reduce_pct": args.dd_reduce_pct,
            "dd_skip_pct": args.dd_skip_pct,
        },
        **_summarize(baseline, args.deposit, "baseline"),
        **_summarize(agent, args.deposit, "agent"),
        "agent_better_final_folds": int((comparison["delta_final"] > 0).sum()),
        "agent_worse_final_folds": int((comparison["delta_final"] < 0).sum()),
        "agent_total_skip": int(agent["skip"].sum()),
        "agent_total_reduce": int(agent["reduce"].sum()),
        "agent_total_boost": int(agent["boost"].sum()),
        "agent_total_take": int(agent["take"].sum()),
        "decision_actions": decisions["agent_action"].value_counts().to_dict(),
    }
    (args.out_dir / "trading_agent_decision_gate_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
