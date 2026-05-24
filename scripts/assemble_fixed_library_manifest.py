from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


MT5_COLUMNS = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]


def load_manifest(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not parse manifest: {path}")


def resolve_path(manifest_path: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    candidates = [Path.cwd() / raw, manifest_path.parent / raw.name, manifest_path.parent / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.cwd() / raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble a fixed, non-adaptive signal-library MT5 manifest.")
    parser.add_argument("--manifest", action="append", type=Path, required=True, help="Source candidate manifest. Repeatable.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--risk-pct", type=float, required=True, help="Risk percent, e.g. 2.0 for 2%.")
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--top-per-source", type=int, default=150)
    parser.add_argument("--max-per-fold", type=int, default=1200)
    parser.add_argument("--exclude-hours", type=str, default=None)
    args = parser.parse_args()

    exclude_hours = None
    if args.exclude_hours:
        exclude_hours = {int(part.strip()) for part in args.exclude_hours.split(",") if part.strip()}
        bad = sorted(hour for hour in exclude_hours if hour < 0 or hour > 23)
        if bad:
            raise ValueError(f"Hours must be 0..23: {bad}")

    source_manifests = [(path, load_manifest(path)) for path in args.manifest]
    fold_ids = sorted(
        {
            int(fold["fold"])
            for _, manifest in source_manifests
            for fold in manifest.get("folds", []) or []
        }
    )
    if not fold_ids:
        raise ValueError("No folds found in source manifests.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, Any]] = []
    total = 0
    for fold_id in fold_ids:
        pieces: list[pd.DataFrame] = []
        fold_meta: dict[str, Any] | None = None
        for source_idx, (manifest_path, manifest) in enumerate(source_manifests):
            source_fold = next((dict(fold) for fold in manifest.get("folds", []) if int(fold["fold"]) == fold_id), None)
            if source_fold is None:
                continue
            if fold_meta is None:
                fold_meta = source_fold
            csv_path = resolve_path(manifest_path, str(source_fold["csv"]))
            signals = pd.read_csv(csv_path)
            if signals.empty:
                continue
            missing = sorted(set(MT5_COLUMNS).difference(signals.columns))
            if missing:
                raise ValueError(f"{csv_path} missing columns: {missing}")
            signals = signals[MT5_COLUMNS].copy()
            signals["probability"] = pd.to_numeric(signals["probability"], errors="coerce").fillna(0.0)
            signals["source_idx"] = source_idx
            signals["source_rank"] = signals["probability"].rank(method="first", ascending=False)
            signals = signals.sort_values(["probability", "open_time"], ascending=[False, True]).head(args.top_per_source)
            pieces.append(signals)
        if fold_meta is None:
            continue
        merged = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=MT5_COLUMNS)
        if not merged.empty:
            open_dt = pd.to_datetime(merged["open_time"], format="%Y.%m.%d %H:%M", errors="coerce")
            if exclude_hours is not None:
                merged = merged[~open_dt.dt.hour.isin(exclude_hours)].copy()
                open_dt = pd.to_datetime(merged["open_time"], format="%Y.%m.%d %H:%M", errors="coerce")
            merged = merged.sort_values(["open_time", "source_rank", "probability"], ascending=[True, True, False])
            merged = merged.drop_duplicates(subset=["open_time", "direction"], keep="first")
            if len(merged) > args.max_per_fold:
                merged = (
                    merged.sort_values(["source_rank", "probability", "open_time"], ascending=[True, False, True])
                    .head(args.max_per_fold)
                    .sort_values("open_time")
                )
            merged = merged[MT5_COLUMNS].copy()

        out_csv = args.out_dir / f"fold_{fold_id:02d}_signals.csv"
        merged.to_csv(out_csv, index=False)
        count = int(len(merged))
        total += count
        folds.append(
            {
                "fold": fold_id,
                "train_start": fold_meta.get("train_start"),
                "train_end": fold_meta.get("train_end"),
                "test_start": fold_meta.get("test_start"),
                "test_end": fold_meta.get("test_end"),
                "signals": count,
                "csv": str(out_csv),
                "risk_pct": round(float(args.risk_pct), 4),
                "max_risk_pct": round(float(args.risk_pct), 4),
                "max_exposure_pct": round(float(args.risk_pct) * int(args.max_positions), 4),
                "max_positions": int(args.max_positions),
            }
        )

    manifest = {
        "all_signals": None,
        "folds": folds,
        "total_signals": total,
        "live_protocol": True,
        "fixed_library_policy": True,
        "adaptive_per_fold": False,
        "selection_uses_current_fold_metrics": False,
        "source_manifests": [str(path) for path, _ in source_manifests],
        "top_per_source": int(args.top_per_source),
        "max_per_fold": int(args.max_per_fold),
        "exclude_hours": sorted(exclude_hours) if exclude_hours is not None else None,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest={args.out_dir / 'manifest.json'}")
    print(f"folds={len(folds)} total_signals={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
