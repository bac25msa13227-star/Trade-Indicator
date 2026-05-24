from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_csv(manifest_path: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    candidates = [Path.cwd() / raw, manifest_path.parent / raw.name, manifest_path.parent / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.cwd() / raw


def _parse_fold_source(values: list[str]) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--fold-source must be FOLD=DIR, got {value!r}")
        fold_text, dir_text = value.split("=", 1)
        fold_id = int(fold_text.strip())
        if fold_id <= 0:
            raise ValueError(f"fold must be positive, got {fold_id}")
        mapping[fold_id] = Path(dir_text.strip())
    return mapping


def _find_fold(manifest: dict[str, Any], fold_id: int) -> dict[str, Any]:
    for fold in manifest.get("folds", []) or []:
        if int(fold["fold"]) == int(fold_id):
            return dict(fold)
    raise ValueError(f"fold {fold_id} not found")


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble a current live manifest from explicit fold=source mappings.")
    parser.add_argument("--fold-source", action="append", required=True, help="FOLD=source_dir, repeatable.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--note", default="current fold map selected from MT5 feedback")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mapping = _parse_fold_source(args.fold_source)
    folds: list[dict[str, Any]] = []
    total_signals = 0
    source_records: list[dict[str, Any]] = []
    all_parts: list[str] = []
    for fold_id, source_dir in sorted(mapping.items()):
        manifest_path = source_dir / "manifest.json"
        results_path = source_dir / "mt5_wf_results.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        source_manifest = _read_json(manifest_path)
        source_fold = _find_fold(source_manifest, fold_id)
        source_csv = _resolve_csv(manifest_path, str(source_fold["csv"]))
        if not source_csv.exists():
            raise FileNotFoundError(f"missing source CSV: {source_csv}")
        target_csv = args.out_dir / f"fold_{fold_id:02d}_signals.csv"
        shutil.copy2(source_csv, target_csv)
        signals = max(0, sum(1 for _ in target_csv.open("r", encoding="utf-8")) - 1)
        total_signals += signals
        all_parts.append(target_csv.read_text(encoding="utf-8").splitlines()[0] if not all_parts else "")
        fold_row = {
            "fold": int(fold_id),
            "train_start": source_fold.get("train_start"),
            "train_end": source_fold.get("train_end"),
            "test_start": source_fold.get("test_start"),
            "test_end": source_fold.get("test_end"),
            "signals": int(signals),
            "csv": str(target_csv),
            "risk_pct": float(source_fold.get("risk_pct", source_fold.get("max_risk_pct", 0.0))),
            "max_risk_pct": float(source_fold.get("max_risk_pct", source_fold.get("risk_pct", 0.0))),
            "max_exposure_pct": float(source_fold.get("max_exposure_pct", source_fold.get("risk_pct", 0.0))),
            "max_positions": int(source_fold.get("max_positions", source_fold.get("max_positions_hint", 1))),
            "source_dir": str(source_dir),
            "source_manifest": str(manifest_path),
        }
        for key in [
            "selected_tp_rr",
            "selected_sl_mult",
            "selected_horizon_bars",
            "selected_min_probability",
            "selected_top_k_per_fold",
        ]:
            if key in source_fold:
                fold_row[key] = source_fold[key]
        folds.append(fold_row)
        source_records.append(
            {
                "fold": int(fold_id),
                "source_dir": str(source_dir),
                "source_manifest": str(manifest_path),
                "source_results": str(results_path) if results_path.exists() else None,
            }
        )

    manifest = {
        "all_signals": None,
        "folds": folds,
        "total_signals": total_signals,
        "live_protocol": True,
        "current_campaign": True,
        "current_fold_map": True,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "adaptive_per_fold": False,
        "source_policy": args.note,
        "source_records": source_records,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(args.out_dir / "manifest.json"), "folds": len(folds), "total_signals": total_signals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
