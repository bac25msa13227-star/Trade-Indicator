from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not parse JSON manifest: {path}")


def resolve_csv(manifest_path: Path, csv_text: str) -> Path:
    raw = Path(csv_text)
    if raw.is_absolute():
        return raw
    for candidate in (Path.cwd() / raw, manifest_path.parent / raw, manifest_path.parent / raw.name):
        if candidate.exists():
            return candidate
    return Path.cwd() / raw


def parse_hours(text: str | None) -> set[int]:
    if not text:
        return set()
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def parse_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y.%m.%d %H:%M", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], errors="coerce")
    return parsed


def numeric_column(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter MT5 manifest signal CSVs by live-known signal columns.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-probability", type=float, default=None)
    parser.add_argument("--max-probability", type=float, default=None)
    parser.add_argument("--min-atr", type=float, default=None)
    parser.add_argument("--max-atr", type=float, default=None)
    parser.add_argument("--include-hours", default="")
    parser.add_argument("--exclude-hours", default="")
    parser.add_argument("--risk-pct", type=float, default=None)
    parser.add_argument("--max-risk-pct", type=float, default=None)
    parser.add_argument("--max-exposure-pct", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    args = parser.parse_args()

    include_hours = parse_hours(args.include_hours)
    exclude_hours = parse_hours(args.exclude_hours)
    if include_hours and exclude_hours:
        raise ValueError("Use either --include-hours or --exclude-hours, not both.")

    source = read_json(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    folds: list[dict[str, Any]] = []
    total = 0
    for source_fold in source.get("folds", []) or []:
        fold = dict(source_fold)
        fold_id = int(fold["fold"])
        source_csv = resolve_csv(args.manifest, str(fold["csv"]))
        signals = pd.read_csv(source_csv)
        mask = pd.Series(True, index=signals.index)

        probability = numeric_column(signals, ("probability", "signal_probability", "confidence", "conf"))
        if args.min_probability is not None:
            if probability is None:
                mask &= False
            else:
                mask &= probability >= float(args.min_probability)
        if args.max_probability is not None:
            if probability is None:
                mask &= False
            else:
                mask &= probability <= float(args.max_probability)

        atr = numeric_column(signals, ("atr", "signal_atr"))
        if args.min_atr is not None:
            if atr is None:
                mask &= False
            else:
                mask &= atr >= float(args.min_atr)
        if args.max_atr is not None:
            if atr is None:
                mask &= False
            else:
                mask &= atr <= float(args.max_atr)

        if include_hours or exclude_hours:
            if "open_time" not in signals.columns:
                mask &= False
            else:
                hours = parse_time(signals["open_time"]).dt.hour
                if include_hours:
                    mask &= hours.isin(include_hours)
                if exclude_hours:
                    mask &= ~hours.isin(exclude_hours)

        filtered = signals.loc[mask].copy()
        target_csv = args.out_dir / f"fold_{fold_id:02d}_signals.csv"
        if len(filtered) == len(signals) and source_csv.resolve() == target_csv.resolve():
            shutil.copy2(source_csv, target_csv)
        else:
            filtered.to_csv(target_csv, index=False)

        fold["csv"] = str(target_csv)
        fold["signals"] = int(len(filtered))
        if args.max_positions is not None:
            fold["max_positions"] = int(args.max_positions)
        positions = int(fold.get("max_positions") or 1)
        if args.risk_pct is not None:
            risk = round(float(args.risk_pct), 4)
            fold["risk_pct"] = risk
            fold["max_risk_pct"] = round(float(args.max_risk_pct if args.max_risk_pct is not None else risk), 4)
            fold["max_exposure_pct"] = round(
                float(args.max_exposure_pct if args.max_exposure_pct is not None else risk * positions),
                4,
            )
        fold["signal_column_filter"] = {
            "min_probability": args.min_probability,
            "max_probability": args.max_probability,
            "min_atr": args.min_atr,
            "max_atr": args.max_atr,
            "include_hours": sorted(include_hours),
            "exclude_hours": sorted(exclude_hours),
        }
        folds.append(fold)
        total += int(len(filtered))

    manifest = dict(source)
    manifest["folds"] = folds
    manifest["total_signals"] = int(total)
    manifest["filtered_from_manifest"] = str(args.manifest)
    manifest["signal_column_filter"] = {
        "min_probability": args.min_probability,
        "max_probability": args.max_probability,
        "min_atr": args.min_atr,
        "max_atr": args.max_atr,
        "include_hours": sorted(include_hours),
        "exclude_hours": sorted(exclude_hours),
    }
    target_manifest = args.out_dir / "manifest.json"
    target_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(target_manifest), "folds": len(folds), "signals": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
