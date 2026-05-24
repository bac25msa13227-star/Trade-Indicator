from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeError:
            continue
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON manifest: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone one fold from an MT5 manifest with optional risk override.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--risk-pct", type=float, default=None, help="Risk percent, e.g. 2.4 for 2.4%.")
    parser.add_argument("--max-risk-pct", type=float, default=None)
    parser.add_argument("--max-exposure-pct", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    args = parser.parse_args()

    source_manifest = read_json(args.manifest)
    source_fold: dict[str, Any] | None = None
    for fold in source_manifest.get("folds", []):
        if int(fold["fold"]) == int(args.fold):
            source_fold = dict(fold)
            break
    if source_fold is None:
        raise ValueError(f"Fold {args.fold} not found in {args.manifest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_csv = Path(source_fold["csv"])
    if not source_csv.is_absolute():
        source_csv = args.manifest.resolve().parents[1] / source_csv
        if not source_csv.exists():
            source_csv = Path(source_fold["csv"])
    target_csv = args.out_dir / f"fold_{int(args.fold):02d}_signals.csv"
    shutil.copy2(source_csv, target_csv)

    if args.risk_pct is not None:
        source_fold["risk_pct"] = round(float(args.risk_pct), 4)
        source_fold["max_risk_pct"] = round(float(args.max_risk_pct if args.max_risk_pct is not None else args.risk_pct), 4)
        source_fold["max_exposure_pct"] = round(
            float(args.max_exposure_pct if args.max_exposure_pct is not None else args.risk_pct),
            4,
        )
    if args.max_positions is not None:
        source_fold["max_positions"] = int(args.max_positions)
    source_fold["csv"] = str(target_csv)
    source_fold["signals"] = sum(1 for _ in target_csv.open("r", encoding="utf-8")) - 1

    manifest = {
        "all_signals": str(target_csv),
        "folds": [source_fold],
        "total_signals": int(source_fold["signals"]),
        "cloned_from_manifest": str(args.manifest),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(source_fold, indent=2))
    print(f"manifest={args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
