from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_feedback(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty feedback file: {path}")
    for column in [
        "fold",
        "volume",
        "entry_price",
        "sl_price",
        "tp_price",
        "close_price",
        "profit",
        "balance_after",
        "equity_after",
        "hour",
        "weekday",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["open_time", "close_time"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=False)
    required = {"fold", "volume", "profit"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"feedback missing columns: {sorted(missing)}")
    return frame.dropna(subset=["fold", "volume", "profit"]).copy()


def max_drawdown_pct(curve: list[float]) -> float:
    peak = curve[0] if curve else 0.0
    worst = 0.0
    for value in curve:
        if value > peak:
            peak = value
        if peak > 0:
            worst = min(worst, (value - peak) / peak * 100.0)
    return worst


def extra_points_for_trade(row: pd.Series, args: argparse.Namespace) -> float:
    points = float(args.extra_roundtrip_points)
    hour = int(row.get("hour")) if pd.notna(row.get("hour")) else -1
    weekday = int(row.get("weekday")) if pd.notna(row.get("weekday")) else -1
    if hour in {int(x) for x in str(args.thin_hours_utc).split(",") if str(x).strip()}:
        points += float(args.thin_hour_extra_points)
    if weekday == 4 and hour >= int(args.friday_cutoff_hour_utc):
        points += float(args.friday_extra_points)
    return max(0.0, points)


def simulate_fold(rows: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    ordered = rows.sort_values(["close_time", "open_time"], na_position="last").copy()
    balance = float(args.deposit)
    curve = [balance]
    total_extra_cost = 0.0
    hit_target = False
    hit_dd = False
    trades_used = 0

    dd_floor = float(args.deposit) * (1.0 - float(args.max_dd_pct) / 100.0)
    for _, row in ordered.iterrows():
        volume = float(row["volume"])
        points = extra_points_for_trade(row, args)
        extra_cost = points * float(args.point_size) * float(args.contract_size) * volume
        extra_cost += float(args.commission_per_lot_roundtrip) * volume
        adjusted_profit = float(row["profit"]) - extra_cost
        balance += adjusted_profit
        trades_used += 1
        total_extra_cost += extra_cost
        curve.append(balance)

        if balance <= dd_floor:
            hit_dd = True
            if args.stop_on_dd:
                break
        if float(args.target_balance) > 0 and balance >= float(args.target_balance):
            hit_target = True
            if args.stop_on_target:
                break

    worst_dd = max_drawdown_pct(curve)
    return {
        "fold": int(ordered["fold"].iloc[0]),
        "trades_available": int(len(ordered)),
        "trades_used": trades_used,
        "final_balance": round(balance, 2),
        "net_pct": round((balance - float(args.deposit)) / float(args.deposit) * 100.0, 2),
        "max_dd_pct": round(worst_dd, 2),
        "target_hit": bool(hit_target or balance >= float(args.target_balance)),
        "dd_pass": bool(worst_dd >= -float(args.max_dd_pct) and not hit_dd),
        "loss_fold": bool(balance < float(args.deposit)),
        "total_extra_cost": round(total_extra_cost, 2),
        "avg_extra_cost_per_trade": round(total_extra_cost / trades_used, 4) if trades_used else 0.0,
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no fold rows to summarize")
    target_pass = int(frame["target_hit"].sum())
    dd_pass = int(frame["dd_pass"].sum())
    loss_folds = int(frame["loss_fold"].sum())
    all_pass_mask = frame["target_hit"] & frame["dd_pass"] & (~frame["loss_fold"])
    return {
        "feedback": str(args.feedback),
        "scenario": {
            "deposit": float(args.deposit),
            "target_balance": float(args.target_balance),
            "max_dd_pct": float(args.max_dd_pct),
            "extra_roundtrip_points": float(args.extra_roundtrip_points),
            "thin_hours_utc": str(args.thin_hours_utc),
            "thin_hour_extra_points": float(args.thin_hour_extra_points),
            "friday_cutoff_hour_utc": int(args.friday_cutoff_hour_utc),
            "friday_extra_points": float(args.friday_extra_points),
            "point_size": float(args.point_size),
            "contract_size": float(args.contract_size),
            "commission_per_lot_roundtrip": float(args.commission_per_lot_roundtrip),
            "stop_on_target": bool(args.stop_on_target),
            "stop_on_dd": bool(args.stop_on_dd),
        },
        "summary": {
            "folds": int(len(frame)),
            "target_pass_folds": target_pass,
            "dd_pass_folds": dd_pass,
            "loss_folds": loss_folds,
            "all_pass_folds": int(all_pass_mask.sum()),
            "min_final_balance": round(float(frame["final_balance"].min()), 2),
            "median_final_balance": round(float(frame["final_balance"].median()), 2),
            "max_final_balance": round(float(frame["final_balance"].max()), 2),
            "worst_dd_pct": round(float(frame["max_dd_pct"].min()), 2),
            "total_trades_used": int(frame["trades_used"].sum()),
            "total_extra_cost": round(float(frame["total_extra_cost"].sum()), 2),
            "realistic_gate_pass": bool(
                target_pass == len(frame)
                and dd_pass == len(frame)
                and loss_folds <= int(args.max_loss_folds)
            ),
            "max_loss_folds": int(args.max_loss_folds),
        },
        "folds": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stress MT5 tester trade feedback with extra realistic spread/slippage/commission costs."
    )
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold-report-out", type=Path, default=None)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--max-loss-folds", type=int, default=1)
    parser.add_argument("--extra-roundtrip-points", type=float, default=100.0)
    parser.add_argument("--thin-hours-utc", default="0,1,2,3,4,5,6,22,23")
    parser.add_argument("--thin-hour-extra-points", type=float, default=100.0)
    parser.add_argument("--friday-cutoff-hour-utc", type=int, default=20)
    parser.add_argument("--friday-extra-points", type=float, default=150.0)
    parser.add_argument("--point-size", type=float, default=0.001)
    parser.add_argument("--contract-size", type=float, default=100.0)
    parser.add_argument("--commission-per-lot-roundtrip", type=float, default=0.0)
    parser.add_argument("--no-stop-on-target", dest="stop_on_target", action="store_false")
    parser.add_argument("--no-stop-on-dd", dest="stop_on_dd", action="store_false")
    parser.set_defaults(stop_on_target=True, stop_on_dd=True)
    args = parser.parse_args()

    feedback = load_feedback(args.feedback)
    rows = [simulate_fold(group, args) for _, group in feedback.groupby("fold", sort=True)]
    report = summarize(rows, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.fold_report_out:
        args.fold_report_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.fold_report_out, index=False)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
