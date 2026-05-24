from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARAM_COLUMNS = [
    "tp_rr",
    "sl_mult",
    "horizon_bars",
    "min_probability",
    "top_k_per_fold",
    "risk_pct",
    "max_positions",
    "min_signal_gap_bars",
    "select_lowest_probability",
    "search_row",
    "rank",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeError:
            continue
        except json.JSONDecodeError:
            continue
    return {}


def _first_fold(manifest: dict[str, Any], fold_id: int) -> dict[str, Any]:
    for fold in manifest.get("folds", []) or []:
        if int(fold.get("fold", -1)) == int(fold_id):
            return dict(fold)
    folds = manifest.get("folds", []) or []
    return dict(folds[0]) if folds else {}


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _candidate_key(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for col in PARAM_COLUMNS:
        value = row.get(col)
        if value is None or value == "":
            continue
        parts.append(f"{col}={value}")
    raw = "|".join(parts)
    if not raw:
        raw = row.get("source_dir", "")
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _collect(root: Path, deposit: float, target_balance: float, min_dd_pct: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for results_path in sorted(root.rglob("mt5_wf_results.csv")):
        source_dir = results_path.parent
        try:
            results = pd.read_csv(results_path)
        except Exception as exc:  # noqa: BLE001
            rows.append({"source_dir": str(source_dir), "error": f"read_results_failed: {exc}"})
            continue

        candidate = _read_json(source_dir / "candidate_selection.json")
        manifest = _read_json(source_dir / "manifest.json")
        for _, result in results.iterrows():
            fold_id = _as_int(result.get("fold")) or _as_int(candidate.get("fold")) or 0
            fold_meta = _first_fold(manifest, fold_id)
            row: dict[str, Any] = {
                "source_dir": str(source_dir),
                "results_csv": str(results_path),
                "manifest": str(source_dir / "manifest.json"),
                "fold": fold_id,
                "test_start": result.get("test_start") or fold_meta.get("test_start"),
                "test_end": result.get("test_end") or fold_meta.get("test_end"),
                "signals": _as_int(result.get("signals")),
                "loaded_signals": _as_int(result.get("loaded_signals")),
                "loaded_signal_gap": None,
                "final_balance": _as_float(result.get("final_balance")),
                "net_pct": _as_float(result.get("net_pct")),
                "max_dd_pct": _as_float(result.get("max_dd_pct")),
                "trades": _as_int(result.get("trades")),
                "win_rate_pct": _as_float(result.get("win_rate_pct")),
                "tp_hits": _as_int(result.get("tp_hits")),
                "sl_hits": _as_int(result.get("sl_hits")),
                "source_search_csv": candidate.get("source_search_csv"),
                "csv": fold_meta.get("csv"),
            }
            for col in PARAM_COLUMNS:
                value = None
                if col in {"risk_pct", "max_positions"} and col in result.index:
                    value = result.get(col)
                if value is None:
                    value = candidate.get(col)
                if value is None and col in result.index:
                    value = result.get(col)
                if value is None:
                    value = fold_meta.get(col)
                if col in {"horizon_bars", "top_k_per_fold", "max_positions", "min_signal_gap_bars", "search_row", "rank"}:
                    row[col] = _as_int(value)
                elif col == "select_lowest_probability":
                    row[col] = bool(value) if value is not None else None
                else:
                    row[col] = _as_float(value)

            if row["signals"] is not None and row["loaded_signals"] is not None:
                row["loaded_signal_gap"] = int(row["loaded_signals"] - row["signals"])
            final_balance = row["final_balance"]
            max_dd = row["max_dd_pct"]
            row["target_pass"] = bool(final_balance is not None and final_balance >= target_balance)
            row["dd_pass"] = bool(max_dd is not None and max_dd >= min_dd_pct)
            row["loss_fold"] = bool(final_balance is not None and final_balance < deposit)
            row["strict_pass"] = bool(row["target_pass"] and row["dd_pass"] and not row["loss_fold"])
            row["target_gap"] = None if final_balance is None else float(final_balance - target_balance)
            row["dd_buffer"] = None if max_dd is None else float(max_dd - min_dd_pct)
            row["score"] = (
                (-10_000.0 if not row["dd_pass"] else 0.0)
                + (-5_000.0 if not row["target_pass"] else 0.0)
                + (float(row["target_gap"] or 0.0))
                + (50.0 * float(row["dd_buffer"] or 0.0))
            )
            row["candidate_key"] = _candidate_key(row)
            rows.append(row)
    return pd.DataFrame(rows)


def _summary(matrix: pd.DataFrame, target_balance: float, min_dd_pct: float) -> dict[str, Any]:
    if matrix.empty:
        return {"rows": 0}
    clean = matrix[pd.to_numeric(matrix["fold"], errors="coerce").notna()].copy()
    if clean.empty:
        return {"rows": int(len(matrix))}
    clean["fold"] = clean["fold"].astype(int)
    best_by_fold = (
        clean.sort_values(["fold", "strict_pass", "score", "final_balance"], ascending=[True, False, False, False])
        .groupby("fold", as_index=False)
        .head(1)
    )
    return {
        "rows": int(len(clean)),
        "folds_observed": sorted(int(x) for x in clean["fold"].unique()),
        "strict_pass_rows": int(clean["strict_pass"].sum()),
        "target_balance": float(target_balance),
        "min_dd_pct": float(min_dd_pct),
        "best_by_fold_strict_pass": int(best_by_fold["strict_pass"].sum()),
        "best_by_fold_target_pass": int(best_by_fold["target_pass"].sum()),
        "best_by_fold_dd_pass": int(best_by_fold["dd_pass"].sum()),
        "best_by_fold_min_final": float(best_by_fold["final_balance"].min()),
        "best_by_fold_worst_dd": float(best_by_fold["max_dd_pct"].min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate single-fold MT5 candidate tests into a reusable matrix.")
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/mt5_candidate_tests")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/mt5_candidate_matrix.csv")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "outputs/mt5_candidate_matrix_summary.json")
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--min-dd-pct", type=float, default=-20.0)
    args = parser.parse_args()

    matrix = _collect(args.root, args.deposit, args.target_balance, args.min_dd_pct)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.out, index=False)
    summary = _summary(matrix, args.target_balance, args.min_dd_pct)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved_matrix={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
