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
        except UnicodeError:
            continue
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON file: {path}")


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


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


def find_fold(manifest: dict[str, Any], fold_id: int) -> dict[str, Any]:
    for fold in manifest.get("folds", []) or []:
        if int(fold.get("fold", -1)) == int(fold_id):
            return dict(fold)
    raise ValueError(f"Fold {fold_id} not found in source manifest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble a research MT5 combo manifest from strict-pass fold rows.")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--min-dd-pct", type=float, default=-20.0)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=30)
    args = parser.parse_args()

    matrix = pd.read_csv(args.matrix)
    for column in ["fold", "final_balance", "max_dd_pct", "score", "loaded_signal_gap"]:
        if column in matrix.columns:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    strict = matrix[
        (matrix["final_balance"] >= args.target_balance)
        & (matrix["max_dd_pct"] >= args.min_dd_pct)
        & (matrix["final_balance"] >= args.deposit)
        & (matrix["loaded_signal_gap"].fillna(0) == 0)
    ].copy()
    if strict.empty:
        raise ValueError("No strict-pass rows found in matrix")

    selected = (
        strict.sort_values(["fold", "score", "max_dd_pct", "final_balance"], ascending=[True, False, False, False])
        .groupby("fold", as_index=False)
        .head(1)
        .sort_values("fold")
    )
    expected_folds = set(range(int(args.fold_start), int(args.fold_end) + 1))
    missing = sorted(expected_folds - {int(x) for x in selected["fold"].dropna().astype(int)})
    if missing:
        raise ValueError(f"Missing strict-pass folds: {missing}")
    selected = selected[selected["fold"].astype(int).isin(expected_folds)].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, Any]] = []
    total_signals = 0
    for _, row in selected.iterrows():
        fold_id = int(row["fold"])
        manifest_path = Path(str(row["manifest"]))
        source_manifest = read_json(manifest_path)
        source_fold = find_fold(source_manifest, fold_id)
        source_csv = resolve_csv(manifest_path, str(source_fold["csv"]))
        target_csv = args.out_dir / f"fold_{fold_id:02d}_signals.csv"
        shutil.copy2(source_csv, target_csv)
        signal_count = sum(1 for _ in target_csv.open("r", encoding="utf-8")) - 1
        total_signals += signal_count

        result_row = pd.read_csv(Path(str(row["results_csv"])))
        result_row = result_row[result_row["fold"].astype(int) == fold_id].iloc[0]
        risk_pct = to_float(result_row.get("risk_pct"), to_float(source_fold.get("risk_pct"), 0.0))
        max_positions = to_int(result_row.get("max_positions"), to_int(source_fold.get("max_positions"), 1))
        max_risk_pct = to_float(result_row.get("max_risk_pct"), risk_pct)
        max_exposure_pct = to_float(result_row.get("max_exposure_pct"), risk_pct)

        fold_meta = {
            "fold": fold_id,
            "train_start": source_fold.get("train_start"),
            "train_end": source_fold.get("train_end"),
            "test_start": str(result_row.get("test_start") or source_fold.get("test_start")),
            "test_end": str(result_row.get("test_end") or source_fold.get("test_end")),
            "signals": signal_count,
            "csv": str(target_csv),
            "risk_pct": risk_pct,
            "max_risk_pct": max_risk_pct,
            "max_exposure_pct": max_exposure_pct,
            "max_positions": max_positions,
            "source_dir": str(row["source_dir"]),
            "source_manifest": str(manifest_path),
            "source_results_csv": str(row["results_csv"]),
            "source_final_balance": to_float(row.get("final_balance")),
            "source_max_dd_pct": to_float(row.get("max_dd_pct")),
            "source_trades": to_int(row.get("trades")),
            "source_win_rate_pct": to_float(row.get("win_rate_pct")),
        }
        folds.append(fold_meta)

    manifest = {
        "all_signals": None,
        "folds": folds,
        "total_signals": total_signals,
        "deposit": args.deposit,
        "target_balance": args.target_balance,
        "min_dd_pct": args.min_dd_pct,
        "research_oracle_fold_selection": True,
        "live_protocol": False,
        "source_matrix": str(args.matrix),
        "selection_rule": "strict pass per fold, then highest score",
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = selected[
        [
            "fold",
            "source_dir",
            "final_balance",
            "max_dd_pct",
            "trades",
            "win_rate_pct",
            "signals",
            "risk_pct",
            "max_positions",
            "score",
        ]
    ]
    summary_path = args.out_dir / "selected_folds.csv"
    summary.to_csv(summary_path, index=False)
    print(f"manifest={manifest_path}")
    print(f"selected={summary_path}")
    print(f"folds={len(folds)} total_signals={total_signals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
