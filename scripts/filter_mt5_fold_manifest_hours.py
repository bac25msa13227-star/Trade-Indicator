from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_hours(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    hours: set[int] = set()
    for part in value.split(","):
        hour = int(part.strip())
        if hour < 0 or hour > 23:
            raise ValueError(f"Hour out of range: {hour}")
        hours.add(hour)
    return hours


def resolve_csv(manifest_path: Path, csv_text: str) -> Path:
    csv_path = Path(csv_text)
    if csv_path.is_absolute():
        return csv_path
    candidates = [
        Path.cwd() / csv_path,
        manifest_path.parent / csv_path,
        manifest_path.resolve().parents[1] / csv_path if len(manifest_path.resolve().parents) > 1 else csv_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return csv_path


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeError:
            continue
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON manifest: {path}")


def parse_directions(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    aliases = {"buy": 1, "long": 1, "1": 1, "sell": -1, "short": -1, "-1": -1}
    directions: set[int] = set()
    for part in value.split(","):
        key = part.strip().lower()
        if key not in aliases:
            raise ValueError(f"Unsupported direction: {part}")
        directions.add(aliases[key])
    return directions


def _filter_fold(
    *,
    source_manifest_path: Path,
    source_fold: dict[str, Any],
    out_dir: Path,
    include_hours: set[int] | None,
    exclude_hours: set[int] | None,
    hour_shift_minutes: int,
    min_probability: float | None,
    max_probability: float | None,
    include_directions: set[int] | None,
    exclude_directions: set[int] | None,
    risk_pct: float | None,
    max_risk_pct: float | None,
    max_exposure_pct: float | None,
    max_positions: int | None,
) -> dict[str, Any]:
    fold_id = int(source_fold["fold"])
    filtered_fold = dict(source_fold)

    source_csv = resolve_csv(source_manifest_path, str(filtered_fold["csv"]))
    signals = pd.read_csv(source_csv)
    if "open_time" not in signals.columns:
        raise ValueError(f"{source_csv} lacks open_time column")
    filter_time = pd.to_datetime(signals["open_time"], format="%Y.%m.%d %H:%M", errors="raise")
    if int(hour_shift_minutes) != 0:
        filter_time = filter_time + pd.Timedelta(minutes=int(hour_shift_minutes))
    hours = filter_time.dt.hour
    if include_hours is not None:
        signals = signals[hours.isin(include_hours)].copy()
    if exclude_hours is not None:
        signals = signals[~hours.isin(exclude_hours)].copy()
    if min_probability is not None:
        signals = signals[pd.to_numeric(signals["probability"], errors="coerce") >= float(min_probability)].copy()
    if max_probability is not None:
        signals = signals[pd.to_numeric(signals["probability"], errors="coerce") <= float(max_probability)].copy()
    if include_directions is not None:
        signals = signals[pd.to_numeric(signals["direction"], errors="coerce").astype("Int64").isin(include_directions)].copy()
    if exclude_directions is not None:
        signals = signals[~pd.to_numeric(signals["direction"], errors="coerce").astype("Int64").isin(exclude_directions)].copy()

    target_csv = out_dir / f"fold_{fold_id:02d}_signals.csv"
    signals.to_csv(target_csv, index=False)

    if max_positions is not None:
        filtered_fold["max_positions"] = int(max_positions)
    effective_positions = int(filtered_fold.get("max_positions", filtered_fold.get("max_positions_hint", 1)) or 1)
    if risk_pct is not None:
        filtered_fold["risk_pct"] = round(float(risk_pct), 4)
        filtered_fold["max_risk_pct"] = round(float(max_risk_pct if max_risk_pct is not None else risk_pct), 4)
        default_exposure = float(risk_pct) * effective_positions
        filtered_fold["max_exposure_pct"] = round(
            float(max_exposure_pct if max_exposure_pct is not None else default_exposure),
            4,
        )
    elif max_positions is not None and "risk_pct" in filtered_fold and "max_exposure_pct" in filtered_fold:
        filtered_fold["max_exposure_pct"] = round(float(filtered_fold["risk_pct"]) * effective_positions, 4)

    filtered_fold["csv"] = str(target_csv)
    filtered_fold["signals"] = int(len(signals))
    return filtered_fold


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter an MT5 fold manifest by signal broker hour after selection.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=None, help="Filter one fold only. Omit to filter every fold.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-hours", type=str, default=None)
    parser.add_argument("--exclude-hours", type=str, default=None)
    parser.add_argument(
        "--hour-shift-minutes",
        type=int,
        default=0,
        help="Apply include/exclude hour filters to open_time plus this shift. Use 5 when EA enters next M5 bar.",
    )
    parser.add_argument("--min-probability", type=float, default=None)
    parser.add_argument("--max-probability", type=float, default=None)
    parser.add_argument("--include-directions", type=str, default=None, help="Comma list: buy,sell,1,-1.")
    parser.add_argument("--exclude-directions", type=str, default=None, help="Comma list: buy,sell,1,-1.")
    parser.add_argument("--risk-pct", type=float, default=None, help="Risk percent, e.g. 1.75 for 1.75 percent.")
    parser.add_argument("--max-risk-pct", type=float, default=None)
    parser.add_argument("--max-exposure-pct", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    args = parser.parse_args()

    include_hours = parse_hours(args.include_hours)
    exclude_hours = parse_hours(args.exclude_hours)
    if include_hours is not None and exclude_hours is not None:
        raise ValueError("Use either --include-hours or --exclude-hours, not both.")
    include_directions = parse_directions(args.include_directions)
    exclude_directions = parse_directions(args.exclude_directions)
    if include_directions is not None and exclude_directions is not None:
        raise ValueError("Use either --include-directions or --exclude-directions, not both.")

    source_manifest = read_json(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_folds = [dict(fold) for fold in source_manifest.get("folds", [])]
    if args.fold is not None:
        source_folds = [fold for fold in source_folds if int(fold["fold"]) == int(args.fold)]
        if not source_folds:
            raise ValueError(f"Fold {args.fold} not found in {args.manifest}")

    filtered_folds = [
        _filter_fold(
            source_manifest_path=args.manifest,
            source_fold=fold,
            out_dir=args.out_dir,
            include_hours=include_hours,
            exclude_hours=exclude_hours,
            hour_shift_minutes=int(args.hour_shift_minutes),
            min_probability=args.min_probability,
            max_probability=args.max_probability,
            include_directions=include_directions,
            exclude_directions=exclude_directions,
            risk_pct=args.risk_pct,
            max_risk_pct=args.max_risk_pct,
            max_exposure_pct=args.max_exposure_pct,
            max_positions=args.max_positions,
        )
        for fold in source_folds
    ]

    manifest = {
        "all_signals": None if len(filtered_folds) > 1 else str(args.out_dir / f"fold_{int(filtered_folds[0]['fold']):02d}_signals.csv"),
        "folds": filtered_folds,
        "total_signals": int(sum(int(fold["signals"]) for fold in filtered_folds)),
        "filtered_from_manifest": str(args.manifest),
        "include_hours": sorted(include_hours) if include_hours is not None else None,
        "exclude_hours": sorted(exclude_hours) if exclude_hours is not None else None,
        "hour_shift_minutes": int(args.hour_shift_minutes),
        "min_probability": args.min_probability,
        "max_probability": args.max_probability,
        "include_directions": sorted(include_directions) if include_directions is not None else None,
        "exclude_directions": sorted(exclude_directions) if exclude_directions is not None else None,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"folds": len(filtered_folds), "total_signals": manifest["total_signals"]}, indent=2))
    print(f"manifest={args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
