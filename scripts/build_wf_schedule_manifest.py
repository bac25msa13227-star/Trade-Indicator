from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_time_range(features: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    frame = pd.read_csv(features, usecols=["time"])
    times = pd.to_datetime(frame["time"], utc=True, errors="coerce").dropna()
    if times.empty:
        raise ValueError(f"No valid time column rows found in {features}")
    return times.min(), times.max()


def build_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_days: int,
    test_days: int,
    step_days: int,
    max_folds: int,
) -> list[dict[str, Any]]:
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("train-days, test-days and step-days must be positive")

    cursor = start + pd.Timedelta(days=train_days)
    folds: list[dict[str, Any]] = []
    while cursor + pd.Timedelta(days=test_days) <= end:
        fold_id = len(folds) + 1
        train_start = cursor - pd.Timedelta(days=train_days)
        train_end = cursor
        test_start = cursor
        test_end = cursor + pd.Timedelta(days=test_days)
        folds.append(
            {
                "fold": fold_id,
                "train_start": train_start.strftime("%Y-%m-%d %H:%M:%S"),
                "train_end": train_end.strftime("%Y-%m-%d %H:%M:%S"),
                "test_start": test_start.strftime("%Y-%m-%d %H:%M:%S"),
                "test_end": test_end.strftime("%Y-%m-%d %H:%M:%S"),
                "signals": 0,
                "csv": f"fold_{fold_id:03d}_signals.csv",
            }
        )
        cursor += pd.Timedelta(days=step_days)

    if max_folds > 0 and len(folds) > max_folds:
        folds = folds[-max_folds:]
        for new_id, fold in enumerate(folds, start=1):
            fold["fold"] = new_id
            fold["csv"] = f"fold_{new_id:03d}_signals.csv"
    return folds


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a clean rolling WF fold schedule manifest from feature timestamps.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=180)
    parser.add_argument("--start", default=None, help="Optional UTC start, YYYY-MM-DD. Defaults to first feature row.")
    parser.add_argument("--end", default=None, help="Optional UTC end, YYYY-MM-DD. Defaults to last feature row.")
    args = parser.parse_args()

    feature_start, feature_end = parse_time_range(args.features)
    start = pd.Timestamp(args.start, tz="UTC") if args.start else feature_start
    end = pd.Timestamp(args.end, tz="UTC") if args.end else feature_end
    if start < feature_start:
        raise ValueError(f"Requested start {start} is before feature start {feature_start}")
    if end > feature_end:
        raise ValueError(f"Requested end {end} is after feature end {feature_end}")
    step_days = int(args.step_days or args.test_days)
    folds = build_folds(start, end, int(args.train_days), int(args.test_days), step_days, int(args.max_folds))
    if not folds:
        raise ValueError("No folds generated. Reduce train/test window or extend data range.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "wf_schedule_manifest_v1",
        "features": str(args.features),
        "feature_start": feature_start.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_end": feature_end.strftime("%Y-%m-%d %H:%M:%S"),
        "train_days": int(args.train_days),
        "test_days": int(args.test_days),
        "step_days": step_days,
        "max_folds": int(args.max_folds),
        "folds": folds,
        "total_signals": 0,
        "notes": "Schedule only. Signal files are produced by downstream clean OOS exporters.",
    }
    path = args.out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(path), "folds": len(folds), "first": folds[0]["test_start"], "last": folds[-1]["test_end"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
