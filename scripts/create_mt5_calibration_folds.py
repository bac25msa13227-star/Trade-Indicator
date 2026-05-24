from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create pre-WF MT5 calibration folds with the same reset-$200 test semantics."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first-test-start", default="2021-08-18")
    parser.add_argument("--last-test-end", default="2023-11-07")
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--train-months", type=int, default=5)
    args = parser.parse_args()

    first_start = pd.Timestamp(args.first_test_start)
    last_end = pd.Timestamp(args.last_test_end)
    if first_start >= last_end:
        raise ValueError("--first-test-start must be before --last-test-end")

    folds: list[dict[str, object]] = []
    fold_id = 1
    test_start = first_start
    while test_start < last_end:
        test_end = min(test_start + pd.Timedelta(days=int(args.test_days)), last_end)
        train_start = test_start - pd.DateOffset(months=int(args.train_months))
        folds.append(
            {
                "fold": fold_id,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": test_start.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "signals": 0,
                "csv": "",
                "risk_pct": 0.0,
                "max_risk_pct": 0.0,
                "max_exposure_pct": 0.0,
                "max_positions": 1,
            }
        )
        fold_id += 1
        test_start = test_end

    manifest = {
        "folds": folds,
        "total_signals": 0,
        "calibration_pre_wf": True,
        "live_protocol": True,
        "selection_uses_current_fold_metrics": False,
        "notes": "Pre-2023-11-07 calibration folds for rolling selector cold-start, generated from MT5-exported bars.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest={args.out}")
    print(f"folds={len(folds)} first={folds[0]['test_start']} last={folds[-1]['test_end']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
