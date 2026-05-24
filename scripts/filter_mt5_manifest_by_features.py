from __future__ import annotations

import argparse
import json
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


def parse_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y.%m.%d %H:%M", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], errors="coerce")
    return parsed


def parse_hours(text: str | None) -> set[int]:
    if not text:
        return set()
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def parse_conditions(text: str | None) -> list[tuple[str, str, float]]:
    if not text:
        return []
    conditions: list[tuple[str, str, float]] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        for op in [">=", "<=", "==", "!=", ">", "<"]:
            if op in item:
                left, right = item.split(op, 1)
                conditions.append((left.strip(), op, float(right.strip())))
                break
        else:
            raise ValueError(f"Unsupported condition: {item}")
    return conditions


def resolve_csv(manifest_path: Path, csv_text: str) -> Path:
    raw = Path(csv_text)
    if raw.is_absolute():
        return raw
    for candidate in [Path.cwd() / raw, manifest_path.parent / raw, manifest_path.parent / raw.name]:
        if candidate.exists():
            return candidate
    return Path.cwd() / raw


def load_features(path: Path, condition_columns: set[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    base = ["time", *sorted(column for column in condition_columns if column in header)]
    features = pd.read_csv(path, usecols=base)
    features["join_time"] = pd.to_datetime(features["time"], errors="coerce")
    features = features.dropna(subset=["join_time"]).drop_duplicates("join_time", keep="last")
    for column in condition_columns:
        if column in features.columns:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features.drop(columns=["time"])


def apply_conditions(frame: pd.DataFrame, conditions: list[tuple[str, str, float]]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, op, value in conditions:
        if column not in frame.columns:
            mask &= False
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if op == ">=":
            mask &= series >= value
        elif op == "<=":
            mask &= series <= value
        elif op == "==":
            mask &= series == value
        elif op == "!=":
            mask &= series != value
        elif op == ">":
            mask &= series > value
        elif op == "<":
            mask &= series < value
    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter MT5 manifest signal CSVs by joined feature-regime conditions.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--conditions", default="", help="Comma conditions, e.g. h4_market_structure_bias==1,volatility_regime>=1")
    parser.add_argument("--exclude-hours", default="")
    parser.add_argument("--include-hours", default="")
    parser.add_argument("--risk-pct", type=float, default=None)
    parser.add_argument("--max-risk-pct", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    args = parser.parse_args()

    source = read_json(args.manifest)
    conditions = parse_conditions(args.conditions)
    condition_columns = {column for column, _, _ in conditions}
    features = load_features(args.features, condition_columns)
    exclude_hours = parse_hours(args.exclude_hours)
    include_hours = parse_hours(args.include_hours)
    if exclude_hours and include_hours:
        raise ValueError("Use either include-hours or exclude-hours, not both")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, Any]] = []
    for source_fold in source.get("folds", []):
        fold = dict(source_fold)
        fold_id = int(fold["fold"])
        signals = pd.read_csv(resolve_csv(args.manifest, str(fold["csv"])))
        signals["join_time"] = parse_time(signals["open_time"])
        joined = signals.merge(features, on="join_time", how="left")
        mask = apply_conditions(joined, conditions)
        hours = joined["join_time"].dt.hour
        if exclude_hours:
            mask &= ~hours.isin(exclude_hours)
        if include_hours:
            mask &= hours.isin(include_hours)
        filtered = signals.loc[mask, [column for column in signals.columns if column != "join_time"]].copy()
        target_csv = args.out_dir / f"fold_{fold_id:02d}_signals.csv"
        filtered.to_csv(target_csv, index=False)
        fold["csv"] = str(target_csv)
        fold["signals"] = int(len(filtered))
        fold["feature_regime_filter"] = {
            "conditions": args.conditions,
            "exclude_hours": sorted(exclude_hours),
            "include_hours": sorted(include_hours),
        }
        if args.max_positions is not None:
            fold["max_positions"] = int(args.max_positions)
        positions = int(fold.get("max_positions") or 1)
        if args.risk_pct is not None:
            fold["risk_pct"] = round(float(args.risk_pct), 4)
            fold["max_risk_pct"] = round(float(args.max_risk_pct if args.max_risk_pct is not None else args.risk_pct), 4)
            fold["max_exposure_pct"] = round(float(fold["risk_pct"]) * positions, 4)
        folds.append(fold)

    manifest = dict(source)
    manifest["folds"] = folds
    manifest["total_signals"] = int(sum(int(fold["signals"]) for fold in folds))
    manifest["filtered_from_manifest"] = str(args.manifest)
    manifest["feature_regime_filter"] = {
        "features": str(args.features),
        "conditions": args.conditions,
        "exclude_hours": sorted(exclude_hours),
        "include_hours": sorted(include_hours),
    }
    out_manifest = args.out_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(out_manifest), "folds": len(folds), "total_signals": manifest["total_signals"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
