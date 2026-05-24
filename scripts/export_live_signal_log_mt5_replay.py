from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


MT5_COLUMNS = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected list in {path}")
    return [row for row in payload if isinstance(row, dict)]


def _side_to_direction(side: str) -> int:
    side = side.strip().lower()
    if side == "buy":
        return 1
    if side == "sell":
        return -1
    raise ValueError(f"unsupported side: {side!r}")


def _infer_atr(row: dict[str, Any], direction: int, entry: float, sl: float) -> float:
    atr = row.get("atr")
    if atr not in (None, ""):
        try:
            value = float(atr)
            if value > 0:
                return value
        except ValueError:
            pass
    return abs(entry - sl)


def export(args: argparse.Namespace) -> dict[str, Any]:
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    rows = _read_rows(args.signal_log)
    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for row in rows:
        bar_raw = row.get("bar_time") or row.get("signal_time") or row.get("time")
        if not bar_raw:
            continue
        bar_time = pd.Timestamp(bar_raw)
        if bar_time.tzinfo is None:
            bar_time = bar_time.tz_localize("UTC")
        else:
            bar_time = bar_time.tz_convert("UTC")
        if not (start <= bar_time < end):
            continue
        if str(row.get("status") or "").lower() not in set(args.statuses):
            continue

        side = str(row.get("side") or "")
        direction = _side_to_direction(side)
        entry = float(row.get("entry_price") or 0.0)
        sl = float(row.get("stop_loss") or row.get("sl_price") or 0.0)
        tp = float(row.get("take_profit") or row.get("tp_price") or 0.0)
        probability = float(row.get("probability") or row.get("confidence") or 0.0)
        atr = _infer_atr(row, direction, entry, sl)
        if entry <= 0 or sl <= 0 or tp <= 0 or probability <= 0:
            continue

        mt5_row = {
            "open_time": (bar_time + pd.Timedelta(hours=args.broker_gmt)).strftime("%Y.%m.%d %H:%M"),
            "direction": direction,
            "entry_price": f"{entry:.5f}",
            "sl_price": f"{sl:.5f}",
            "tp_price": f"{tp:.5f}",
            "atr": f"{atr:.5f}",
            "probability": f"{probability:.5f}",
        }
        selected.append(mt5_row)
        audit.append(
            {
                "bar_time_utc": bar_time.isoformat(),
                "status": row.get("status"),
                "side": side,
                "probability": probability,
                "entry_price": entry,
                "sl_price": sl,
                "tp_price": tp,
                "lot": row.get("lot"),
                "balance": row.get("balance"),
                "signal_key": row.get("signal_key"),
                "reason": row.get("reason"),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    signal_path = args.out_dir / f"fold_{args.fold:03d}_signals.csv"
    with signal_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MT5_COLUMNS)
        writer.writeheader()
        writer.writerows(selected)

    manifest = {
        "all_signals": str(signal_path),
        "folds": [
            {
                "fold": int(args.fold),
                "train_start": str(args.train_start),
                "train_end": str(args.train_end),
                "test_start": str(args.start_date),
                "test_end": str(args.end_date),
                "signals": len(selected),
                "csv": str(signal_path),
                "risk_pct": float(args.risk_pct),
                "max_risk_pct": float(args.risk_pct),
                "max_exposure_pct": float(args.risk_pct),
                "max_positions": int(args.max_positions),
                "selected_tp_rr": None,
                "selected_sl_mult": None,
                "selected_horizon_bars": None,
                "selected_min_probability": None,
                "selected_top_k_per_fold": 0,
            }
        ],
        "total_signals": len(selected),
        "live_replay_manifest": True,
        "source_signal_log": str(args.signal_log),
        "source_statuses": list(args.statuses),
        "source_policy": "replay exact live sent/selected decisions from bridge signal log; no model re-score",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out_dir / "live_replay_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {"signals": len(selected), "manifest": str(args.out_dir / "manifest.json"), "signal_csv": str(signal_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export bridge signal log rows into an MT5 replay manifest.")
    parser.add_argument("--signal-log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", required=True, help="UTC ISO start, inclusive")
    parser.add_argument("--end", required=True, help="UTC ISO end, exclusive")
    parser.add_argument("--start-date", required=True, help="MT5 tester start date")
    parser.add_argument("--end-date", required=True, help="MT5 tester end date")
    parser.add_argument("--train-start", default="")
    parser.add_argument("--train-end", default="")
    parser.add_argument("--fold", type=int, default=183)
    parser.add_argument("--risk-pct", type=float, default=2.0)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--broker-gmt", type=int, default=0)
    parser.add_argument("--statuses", nargs="+", default=["sent"])
    args = parser.parse_args()
    print(json.dumps(export(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
