from __future__ import annotations

import argparse
import json
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def load_rates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time").drop_duplicates("time", keep="last")
    for col in ["open", "high", "low", "close"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["open", "high", "low", "close"])


def utc_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def paper_orders(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        if event.get("execute"):
            continue
        if event.get("status") != "signal_ready":
            continue
        generated_at = event.get("generated_at_utc")
        for order in event.get("orders") or []:
            if order.get("status") != "dry_run":
                continue
            key = str(order.get("signal_key") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            body = order.get("order") or {}
            rows.append(
                {
                    "signal_key": key,
                    "generated_at_utc": generated_at,
                    "signal_open_time_utc": order.get("signal_open_time_utc"),
                    "side": body.get("side"),
                    "entry_tick": order.get("entry_tick"),
                    "stop_loss": body.get("stop_loss"),
                    "take_profit": body.get("take_profit"),
                    "volume": body.get("volume"),
                    "probability": order.get("probability"),
                }
            )
    return rows


def resolve_outcome(order: dict[str, Any], rates: pd.DataFrame) -> dict[str, Any]:
    try:
        start = utc_ts(order["signal_open_time_utc"])
        side = str(order["side"]).lower()
        sl = float(order["stop_loss"])
        tp = float(order["take_profit"])
        entry = float(order["entry_tick"])
    except Exception as exc:  # noqa: BLE001
        return {**order, "outcome": "invalid", "error": str(exc)}

    future = rates[rates["time"] > start].copy()
    if future.empty:
        return {**order, "outcome": "open"}
    for _, bar in future.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        hit_sl = low <= sl if side == "buy" else high >= sl
        hit_tp = high >= tp if side == "buy" else low <= tp
        if hit_sl or hit_tp:
            # Conservative for bars where both levels are touched.
            outcome = "loss" if hit_sl else "win"
            exit_price = sl if hit_sl else tp
            rr = -1.0 if hit_sl else abs(tp - entry) / max(abs(entry - sl), 1e-12)
            return {
                **order,
                "outcome": outcome,
                "exit_time_utc": utc_ts(bar["time"]).isoformat(),
                "exit_price": exit_price,
                "rr": rr,
            }
    return {**order, "outcome": "open"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper report from rolling_v3 dry-run events and live MT5 rates.")
    parser.add_argument("--events", type=Path, default=Path("outputs/target1200_live_rolling_v3_current_selected/paper_bridge_events.jsonl"))
    parser.add_argument("--rates", type=Path, default=Path("outputs/mt5_rates_export_202306_20260513_live_merged.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/target1200_live_rolling_v3_current_selected/paper_30day_report.json"))
    parser.add_argument("--trades-out", type=Path, default=Path("outputs/target1200_live_rolling_v3_current_selected/paper_trades.csv"))
    parser.add_argument("--min-days", type=float, default=30.0)
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--min-win-rate-pct", type=float, default=50.0)
    args = parser.parse_args()

    events = read_events(args.events)
    rates = load_rates(args.rates) if args.rates.exists() else pd.DataFrame()
    orders = paper_orders(events)
    resolved = [resolve_outcome(order, rates) for order in orders] if not rates.empty else orders
    frame = pd.DataFrame(resolved)
    if not frame.empty:
        args.trades_out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.trades_out, index=False)

    event_times = [utc_ts(event["generated_at_utc"]) for event in events if event.get("generated_at_utc")]
    days = 0.0
    if event_times:
        days = (max(event_times) - min(event_times)).total_seconds() / 86400.0
    resolved_trades = frame[frame.get("outcome", pd.Series(dtype=str)).isin(["win", "loss"])] if not frame.empty else frame
    wins = int((resolved_trades.get("outcome") == "win").sum()) if not resolved_trades.empty else 0
    losses = int((resolved_trades.get("outcome") == "loss").sum()) if not resolved_trades.empty else 0
    trades = int(len(resolved_trades))
    win_rate = wins / trades * 100.0 if trades else 0.0
    blockers = []
    if days < args.min_days:
        blockers.append(f"paper days {days:.2f} < {args.min_days:.2f}")
    if trades < args.min_trades:
        blockers.append(f"resolved paper trades {trades} < {args.min_trades}")
    if win_rate < args.min_win_rate_pct:
        blockers.append(f"paper win_rate {win_rate:.2f}% < {args.min_win_rate_pct:.2f}%")
    report = {
        "paper_allowed_for_execute_gate": len(blockers) == 0,
        "blockers": blockers,
        "events": len(events),
        "signals": len(orders),
        "trades": trades,
        "open_trades": int((frame.get("outcome") == "open").sum()) if not frame.empty else 0,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "days": days,
        "events_path": str(args.events),
        "rates_path": str(args.rates),
        "trades_path": str(args.trades_out),
        "generated_at_utc": pd.Timestamp.now(tz=timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
