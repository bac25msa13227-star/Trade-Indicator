from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def max_drawdown_pct(curve: list[float]) -> float:
    peak = curve[0]
    worst = 0.0
    for value in curve:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, (value - peak) / peak * 100.0)
    return worst


def simulate_fold(rows: pd.DataFrame, args: argparse.Namespace, dd_stop_pct: float, loss_streak_pause: int) -> dict[str, Any]:
    balance = float(args.deposit)
    curve = [balance]
    used = 0
    skipped = 0
    loss_streak = 0
    cooldown = 0
    stopped = False
    for _, row in rows.sort_values(["close_time", "open_time"]).iterrows():
        if cooldown > 0:
            cooldown -= 1
            skipped += 1
            continue
        if max_drawdown_pct(curve) <= -float(dd_stop_pct):
            stopped = True
            skipped += len(rows) - used - skipped
            break
        profit = float(row["profit"])
        balance += profit
        curve.append(balance)
        used += 1
        if profit < 0:
            loss_streak += 1
            if loss_streak_pause > 0 and loss_streak >= int(loss_streak_pause):
                cooldown = int(args.cooldown_trades)
                loss_streak = 0
        else:
            loss_streak = 0
        if balance >= float(args.target_balance):
            break
    return {
        "fold": int(rows["fold"].iloc[0]),
        "final_balance": round(balance, 2),
        "target_hit": bool(balance >= float(args.target_balance)),
        "max_dd_pct": round(max_drawdown_pct(curve), 2),
        "dd_pass": bool(max_drawdown_pct(curve) >= -float(args.max_dd_pct)),
        "loss_fold": bool(balance < float(args.deposit)),
        "trades_used": int(used),
        "trades_skipped": int(skipped),
        "dd_stopped": bool(stopped),
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    strict = frame["target_hit"] & frame["dd_pass"] & (~frame["loss_fold"])
    return {
        "folds": int(len(frame)),
        "all_pass_folds": int(strict.sum()),
        "target_pass_folds": int(frame["target_hit"].sum()),
        "dd_pass_folds": int(frame["dd_pass"].sum()),
        "loss_folds": int(frame["loss_fold"].sum()),
        "min_final_balance": round(float(frame["final_balance"].min()), 2),
        "median_final_balance": round(float(frame["final_balance"].median()), 2),
        "worst_dd_pct": round(float(frame["max_dd_pct"].min()), 2),
        "total_trades_used": int(frame["trades_used"].sum()),
        "total_trades_skipped": int(frame["trades_skipped"].sum()),
        "fail_folds": frame.loc[~strict, "fold"].astype(int).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep clean drawdown-aware throttles over MT5 trade feedback sequence.")
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--dd-stops", default="12,14,16,18,20")
    parser.add_argument("--loss-streak-pauses", default="0,2,3,4")
    parser.add_argument("--cooldown-trades", type=int, default=8)
    args = parser.parse_args()

    feedback = pd.read_csv(args.feedback)
    for col in ["fold", "profit"]:
        feedback[col] = pd.to_numeric(feedback[col], errors="coerce")
    for col in ["open_time", "close_time"]:
        feedback[col] = pd.to_datetime(feedback[col], errors="coerce")
    feedback = feedback.dropna(subset=["fold", "profit"]).copy()

    records: list[dict[str, Any]] = []
    detail: dict[str, list[dict[str, Any]]] = {}
    for dd_stop in [float(x) for x in args.dd_stops.split(",") if x.strip()]:
        for loss_pause in [int(float(x)) for x in args.loss_streak_pauses.split(",") if x.strip()]:
            rows = [simulate_fold(group, args, dd_stop, loss_pause) for _, group in feedback.groupby("fold", sort=True)]
            summary = summarize(rows, args)
            key = f"dd{dd_stop:g}_ls{loss_pause}"
            summary.update({"key": key, "dd_stop_pct": dd_stop, "loss_streak_pause": loss_pause})
            records.append(summary)
            detail[key] = rows

    records.sort(
        key=lambda item: (
            item["all_pass_folds"],
            item["target_pass_folds"],
            item["dd_pass_folds"],
            item["worst_dd_pct"],
            item["median_final_balance"],
        ),
        reverse=True,
    )
    report = {"best": records[:10], "details": {record["key"]: detail[record["key"]] for record in records[:10]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best": records[:10]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
