from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


MT5_COLUMNS = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not parse JSON: {path}")


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble clean consensus MT5 signal manifest from fixed source manifests.")
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--risk-pct", type=float, required=True)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--time-bucket-minutes", type=int, default=5)
    parser.add_argument("--max-per-fold", type=int, default=500)
    parser.add_argument("--exclude-hours", default="")
    args = parser.parse_args()

    exclude_hours = parse_hours(args.exclude_hours)
    sources = [(path, read_json(path), path.parent.name) for path in args.manifest]
    fold_ids = sorted({int(f["fold"]) for _, payload, _ in sources for f in payload.get("folds", []) or []})
    args.out_dir.mkdir(parents=True, exist_ok=True)

    folds: list[dict[str, Any]] = []
    total = 0
    for fold_id in fold_ids:
        pieces: list[pd.DataFrame] = []
        fold_meta: dict[str, Any] | None = None
        for manifest_path, payload, label in sources:
            source_fold = next((dict(f) for f in payload.get("folds", []) or [] if int(f["fold"]) == fold_id), None)
            if source_fold is None:
                continue
            if fold_meta is None:
                fold_meta = source_fold
            csv_path = resolve_csv(manifest_path, str(source_fold["csv"]))
            signals = pd.read_csv(csv_path)
            if signals.empty:
                continue
            missing = sorted(set(MT5_COLUMNS).difference(signals.columns))
            if missing:
                raise ValueError(f"{csv_path} missing columns: {missing}")
            signals = signals[MT5_COLUMNS].copy()
            signals["open_dt"] = pd.to_datetime(signals["open_time"], format="%Y.%m.%d %H:%M", errors="coerce")
            signals = signals.dropna(subset=["open_dt"])
            if exclude_hours:
                signals = signals[~signals["open_dt"].dt.hour.isin(exclude_hours)].copy()
            signals["bucket"] = signals["open_dt"].dt.floor(f"{int(args.time_bucket_minutes)}min")
            signals["source_label"] = label
            signals["probability"] = pd.to_numeric(signals["probability"], errors="coerce").fillna(0.0)
            signals["atr"] = pd.to_numeric(signals["atr"], errors="coerce")
            pieces.append(signals)
        if fold_meta is None:
            continue
        if not pieces:
            out = pd.DataFrame(columns=MT5_COLUMNS)
        else:
            all_signals = pd.concat(pieces, ignore_index=True)
            rows: list[dict[str, Any]] = []
            for (_, direction), group in all_signals.groupby(["bucket", "direction"], sort=True):
                source_count = group["source_label"].nunique()
                if source_count < int(args.min_sources):
                    continue
                best = group.sort_values(["probability", "atr"], ascending=[False, False]).iloc[0].to_dict()
                best["consensus_sources"] = int(source_count)
                best["probability"] = float(group["probability"].mean())
                rows.append(best)
            out = pd.DataFrame(rows)
            if not out.empty:
                out = out.sort_values(["consensus_sources", "probability", "open_dt"], ascending=[False, False, True])
                if len(out) > int(args.max_per_fold):
                    out = out.head(int(args.max_per_fold))
                out = out.sort_values("open_dt")[MT5_COLUMNS].copy()
            else:
                out = pd.DataFrame(columns=MT5_COLUMNS)

        out_csv = args.out_dir / f"fold_{fold_id:02d}_signals.csv"
        out.to_csv(out_csv, index=False)
        count = int(len(out))
        total += count
        positions = int(args.max_positions)
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
                "max_exposure_pct": round(float(args.risk_pct) * positions, 4),
                "max_positions": positions,
            }
        )

    manifest = {
        "all_signals": None,
        "folds": folds,
        "total_signals": total,
        "live_protocol": True,
        "consensus_manifest": True,
        "adaptive_per_fold": False,
        "research_oracle_fold_selection": False,
        "selection_uses_current_fold_metrics": False,
        "source_manifests": [str(path) for path, _, _ in sources],
        "min_sources": int(args.min_sources),
        "time_bucket_minutes": int(args.time_bucket_minutes),
        "risk_pct": float(args.risk_pct),
    }
    out_manifest = args.out_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(out_manifest), "folds": len(folds), "signals": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
