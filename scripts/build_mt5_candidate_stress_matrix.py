from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            text = path.read_text(encoding=encoding).replace("NaN", "null")
            return json.loads(text)
        except (UnicodeError, json.JSONDecodeError):
            continue
    return {}


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def label_for(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 2 and parts[-2] in {"fixed_library_candidates_20260514", "mt5_rule_library_full_20260513"}:
        return f"{parts[-2]}/{parts[-1]}"
    return path.name


def run_cmd(args: list[str]) -> None:
    completed = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(args)}\n{completed.stdout}")


def load_candidates(args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    for item in args.candidate_dir or []:
        candidates.append(resolve(Path(item)))

    if args.inventory is not None:
        inv = pd.read_csv(resolve(args.inventory))
        if args.only_non_oracle:
            inv = inv[
                ~inv.get("adaptive", pd.Series(False, index=inv.index)).astype(bool)
                & ~inv.get("oracle", pd.Series(False, index=inv.index)).astype(bool)
                & ~inv.get("current_metrics", pd.Series(False, index=inv.index)).astype(bool)
            ].copy()
        if args.min_raw_strict > 0 and "strict_raw" in inv.columns:
            inv = inv[pd.to_numeric(inv["strict_raw"], errors="coerce").fillna(0) >= int(args.min_raw_strict)]
        if args.min_raw_target > 0 and "target_raw" in inv.columns:
            inv = inv[pd.to_numeric(inv["target_raw"], errors="coerce").fillna(0) >= int(args.min_raw_target)]
        if args.max_candidates > 0:
            sort_cols = [col for col in ["strict_raw", "target_raw", "median_final"] if col in inv.columns]
            if sort_cols:
                inv = inv.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            inv = inv.head(int(args.max_candidates))
        candidates.extend(resolve(Path(raw)) for raw in inv["dir"].astype(str).tolist())

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Build stress summaries for MT5 candidate directories.")
    parser.add_argument("--candidate-dir", action="append", default=None)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--refresh-feedback", action="store_true")
    parser.add_argument("--only-non-oracle", action="store_true")
    parser.add_argument("--min-raw-strict", type=int, default=0)
    parser.add_argument("--min-raw-target", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--extra-roundtrip-points", type=float, default=30.0)
    parser.add_argument("--thin-hour-extra-points", type=float, default=50.0)
    parser.add_argument("--friday-extra-points", type=float, default=100.0)
    parser.add_argument("--thin-hours-utc", default="0,1,2,3,4,5,6,22,23")
    parser.add_argument("--friday-cutoff-hour-utc", type=int, default=20)
    parser.add_argument("--point-size", type=float, default=0.001)
    parser.add_argument("--contract-size", type=float, default=100.0)
    parser.add_argument("--commission-per-lot-roundtrip", type=float, default=0.0)
    args = parser.parse_args()

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args)
    if not candidates:
        raise ValueError("No candidate directories provided.")

    summary_rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for candidate in candidates:
        manifest = candidate / "manifest.json"
        results = candidate / "mt5_wf_results.csv"
        if not manifest.exists() or not results.exists():
            errors.append({"candidate": str(candidate), "error": "missing manifest or mt5_wf_results.csv"})
            continue

        candidate_label = label_for(candidate)
        feedback = candidate / "mt5_trade_feedback_detailed.csv"
        if args.refresh_feedback or not feedback.exists():
            try:
                run_cmd(
                    [
                        sys.executable,
                        "scripts/export_mt5_trade_feedback.py",
                        "--manifest",
                        str(manifest),
                        "--results",
                        str(results),
                        "--out",
                        str(feedback),
                    ]
                )
            except Exception as exc:  # noqa: BLE001
                errors.append({"candidate": str(candidate), "error": f"feedback_export_failed: {exc}"})
                continue

        safe_name = candidate_label.replace("\\", "_").replace("/", "_").replace(":", "_")
        stress_json = out_dir / f"{safe_name}_stress.json"
        stress_folds = out_dir / f"{safe_name}_stress_folds.csv"
        try:
            run_cmd(
                [
                    sys.executable,
                    "scripts/mt5_realistic_stress_gate.py",
                    "--feedback",
                    str(feedback),
                    "--out",
                    str(stress_json),
                    "--fold-report-out",
                    str(stress_folds),
                    "--deposit",
                    str(args.deposit),
                    "--target-balance",
                    str(args.target_balance),
                    "--max-dd-pct",
                    str(args.max_dd_pct),
                    "--extra-roundtrip-points",
                    str(args.extra_roundtrip_points),
                    "--thin-hour-extra-points",
                    str(args.thin_hour_extra_points),
                    "--friday-extra-points",
                    str(args.friday_extra_points),
                    "--thin-hours-utc",
                    str(args.thin_hours_utc),
                    "--friday-cutoff-hour-utc",
                    str(args.friday_cutoff_hour_utc),
                    "--point-size",
                    str(args.point_size),
                    "--contract-size",
                    str(args.contract_size),
                    "--commission-per-lot-roundtrip",
                    str(args.commission_per_lot_roundtrip),
                ]
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"candidate": str(candidate), "error": f"stress_failed: {exc}"})
            continue

        report = read_json(stress_json)
        source_manifest = read_json(manifest)
        summary = report.get("summary", {})
        folds = pd.read_csv(stress_folds)
        folds.insert(0, "candidate_label", candidate_label)
        folds.insert(1, "candidate_dir", str(candidate))
        fold_frames.append(folds)

        summary_rows.append(
            {
                "candidate_label": candidate_label,
                "candidate_dir": str(candidate),
                "folds": int(summary.get("folds", len(folds)) or 0),
                "all_pass_folds": int(summary.get("all_pass_folds", 0) or 0),
                "target_pass_folds": int(summary.get("target_pass_folds", 0) or 0),
                "dd_pass_folds": int(summary.get("dd_pass_folds", 0) or 0),
                "loss_folds": int(summary.get("loss_folds", 0) or 0),
                "min_final_balance": float(summary.get("min_final_balance", 0.0) or 0.0),
                "median_final_balance": float(summary.get("median_final_balance", 0.0) or 0.0),
                "worst_dd_pct": float(summary.get("worst_dd_pct", 0.0) or 0.0),
                "total_extra_cost": float(summary.get("total_extra_cost", 0.0) or 0.0),
                "total_trades_used": int(summary.get("total_trades_used", 0) or 0),
                "realistic_gate_pass": bool(summary.get("realistic_gate_pass", False)),
                "live_protocol": is_true(source_manifest.get("live_protocol")),
                "adaptive_per_fold": is_true(source_manifest.get("adaptive_per_fold")),
                "research_oracle_fold_selection": is_true(source_manifest.get("research_oracle_fold_selection")),
                "selection_uses_current_fold_metrics": is_true(source_manifest.get("selection_uses_current_fold_metrics")),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["all_pass_folds", "target_pass_folds", "dd_pass_folds", "median_final_balance"],
        ascending=[False, False, False, False],
    )
    summary_df.to_csv(out_dir / "candidate_stress_summary.csv", index=False)
    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(out_dir / "candidate_stress_folds.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "candidate_stress_errors.csv", index=False)

    result = {
        "out_dir": str(out_dir),
        "candidates_requested": len(candidates),
        "candidates_scored": int(len(summary_df)),
        "errors": int(len(errors)),
        "best": summary_df.head(10).to_dict(orient="records") if not summary_df.empty else [],
    }
    (out_dir / "candidate_stress_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
