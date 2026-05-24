from __future__ import annotations

import argparse
import json
import math
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


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def resolve_csv(manifest_path: Path, csv_text: str) -> Path:
    raw = Path(csv_text)
    if raw.is_absolute():
        return raw
    candidates = [
        Path.cwd() / raw,
        manifest_path.parent / raw.name,
        manifest_path.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.cwd() / raw


def candidate_label(candidate_dir: Path, manifest: dict[str, Any]) -> str:
    selection = candidate_dir / "candidate_selection.json"
    if selection.exists():
        data = read_json(selection)
        parts = []
        for key in [
            "search_row",
            "tp_rr",
            "sl_mult",
            "horizon_bars",
            "min_probability",
            "top_k_per_fold",
            "risk_pct",
            "max_positions",
            "min_signal_gap_bars",
            "select_lowest_probability",
        ]:
            value = data.get(key)
            if value is not None:
                parts.append(f"{key}={value}")
        if parts:
            return "|".join(parts)
    if manifest.get("fixed_library_policy"):
        return f"fixed_library:{candidate_dir.name}"
    if manifest.get("live_protocol"):
        return f"live_protocol:{candidate_dir.name}"
    return candidate_dir.name


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fold_map(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(fold["fold"]): dict(fold) for fold in manifest.get("folds", []) or []}


def load_candidate(candidate_dir: Path) -> dict[str, Any]:
    manifest_path = candidate_dir / "manifest.json"
    results_path = candidate_dir / "mt5_wf_results.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing MT5 results: {results_path}")
    manifest = read_json(manifest_path)
    if is_true(manifest.get("research_oracle_fold_selection")):
        raise ValueError(f"Refusing research_oracle candidate in live rolling selector: {candidate_dir}")
    results = pd.read_csv(results_path)
    for column in [
        "fold",
        "signals",
        "loaded_signals",
        "final_balance",
        "max_dd_pct",
        "trades",
        "risk_pct",
        "max_risk_pct",
        "max_exposure_pct",
        "max_positions",
    ]:
        if column in results.columns:
            results[column] = pd.to_numeric(results[column], errors="coerce")
    return {
        "dir": candidate_dir,
        "label": candidate_label(candidate_dir, manifest),
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_live_protocol": is_true(manifest.get("live_protocol")),
        "manifest_adaptive_per_fold": is_true(manifest.get("adaptive_per_fold")),
        "manifest_research_oracle_fold_selection": is_true(manifest.get("research_oracle_fold_selection")),
        "manifest_selection_uses_current_fold_metrics": is_true(manifest.get("selection_uses_current_fold_metrics")),
        "folds": fold_map(manifest),
        "results_path": results_path,
        "results": results,
    }


def parse_bootstrap_map(values: list[str] | None) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--bootstrap-map must be FOLD=DIR, got: {value}")
        fold_text, dir_text = value.split("=", 1)
        fold_id = int(fold_text.strip())
        if fold_id <= 0:
            raise ValueError(f"Bootstrap fold must be positive, got: {fold_id}")
        mapping[fold_id] = Path(dir_text.strip())
    return mapping


def history_score(history: pd.DataFrame, args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    if history.empty:
        return -1.0e18, {
            "hist_rows": 0,
            "hist_strict_pass": 0,
            "hist_target_pass": 0,
            "hist_dd_pass": 0,
            "hist_loss": 0,
            "hist_min_final": math.nan,
            "hist_worst_dd": math.nan,
            "hist_total_trades": 0,
        }
    trades_ok = history["trades"].fillna(0) >= int(args.min_history_trades)
    loaded_gap = (history["loaded_signals"] - history["signals"]).abs() if {"loaded_signals", "signals"}.issubset(history.columns) else 0
    loaded_ok = pd.Series(True, index=history.index) if isinstance(loaded_gap, int) else loaded_gap.fillna(0) <= float(args.max_loaded_signal_gap)
    # Only count target/dd/loss on rows with real trades; zero-trade warmup folds must not contribute positive score
    active = history[trades_ok]
    target = active["final_balance"] >= float(args.target_balance)
    dd = active["max_dd_pct"] >= float(args.min_dd_pct)
    loss = active["final_balance"] < float(args.deposit)
    strict = target & dd & (~loss) & loaded_ok.reindex(active.index, fill_value=True)

    hist_rows = int(len(history))
    strict_count = int(strict.sum())
    target_count = int(target.sum())
    dd_count = int(dd.sum())
    loss_count = int(loss.sum())
    dd_fail_count = int((~dd).sum())
    target_fail_count = int((~target).sum())
    min_final = float(active["final_balance"].min()) if not active.empty else float(history["final_balance"].min())
    median_final = float(active["final_balance"].median()) if not active.empty else float(history["final_balance"].median())
    worst_dd = float(active["max_dd_pct"].min()) if not active.empty else float(history["max_dd_pct"].min())
    total_trades = int(history["trades"].fillna(0).sum())
    # Recency: fold order is already chronological; recent strict failures should hurt.
    recent_all = history.tail(int(args.recent_folds)) if int(args.recent_folds) > 0 else history
    recent = recent_all[recent_all["trades"].fillna(0) >= int(args.min_history_trades)]
    recent_target = int((recent["final_balance"] >= float(args.target_balance)).sum())
    recent_dd = int((recent["max_dd_pct"] >= float(args.min_dd_pct)).sum())
    recent_loss = int((recent["final_balance"] < float(args.deposit)).sum())

    score = (
        strict_count * 100_000.0
        + target_count * 20_000.0
        + dd_count * 8_000.0
        + recent_target * 15_000.0
        + recent_dd * 5_000.0
        - loss_count * 80_000.0
        - dd_fail_count * 25_000.0
        - target_fail_count * 20_000.0
        - recent_loss * 100_000.0
        + min(median_final, float(args.target_balance))
        + worst_dd * 500.0
        + min(total_trades, 10_000) * 0.05
    )
    stats = {
        "hist_rows": hist_rows,
        "hist_strict_pass": strict_count,
        "hist_target_pass": target_count,
        "hist_dd_pass": dd_count,
        "hist_loss": loss_count,
        "hist_target_fail": target_fail_count,
        "hist_dd_fail": dd_fail_count,
        "hist_min_final": min_final,
        "hist_median_final": median_final,
        "hist_worst_dd": worst_dd,
        "hist_total_trades": total_trades,
        "hist_recent_target_pass": recent_target,
        "hist_recent_dd_pass": recent_dd,
        "hist_recent_loss": recent_loss,
    }
    return score, stats


def result_row_for_fold(results: pd.DataFrame, fold_id: int) -> pd.Series | None:
    rows = results[results["fold"].astype("Int64") == int(fold_id)]
    if rows.empty:
        return None
    return rows.iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a live manifest selected from past MT5 feedback only.")
    parser.add_argument("--candidate-dir", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-candidate-dir", type=Path, default=None)
    parser.add_argument(
        "--bootstrap-map",
        action="append",
        default=None,
        help="Per-fold declared bootstrap candidate as FOLD=DIR. Used before min-history-folds is met.",
    )
    parser.add_argument("--min-history-folds", type=int, default=3)
    parser.add_argument("--lookback-folds", type=int, default=0, help="0 means all prior folds.")
    parser.add_argument("--recent-folds", type=int, default=3)
    parser.add_argument("--min-history-trades", type=int, default=1)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--min-dd-pct", type=float, default=-20.0)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--max-loaded-signal-gap", type=float, default=0.0)
    args = parser.parse_args()

    candidates = [load_candidate(path) for path in args.candidate_dir]
    bootstrap_map_paths = parse_bootstrap_map(args.bootstrap_map)
    bootstrap_label: str | None = None
    if args.bootstrap_candidate_dir is not None:
        bootstrap = next((candidate for candidate in candidates if candidate["dir"].resolve() == args.bootstrap_candidate_dir.resolve()), None)
        if bootstrap is None:
            bootstrap = load_candidate(args.bootstrap_candidate_dir)
            candidates.append(bootstrap)
        bootstrap_label = bootstrap["label"]
    bootstrap_map_labels: dict[int, str] = {}
    for fold_id, candidate_dir in bootstrap_map_paths.items():
        bootstrap = next((candidate for candidate in candidates if candidate["dir"].resolve() == candidate_dir.resolve()), None)
        if bootstrap is None:
            bootstrap = load_candidate(candidate_dir)
            candidates.append(bootstrap)
        bootstrap_map_labels[int(fold_id)] = bootstrap["label"]

    fold_ids = sorted({fold_id for candidate in candidates for fold_id in candidate["folds"]})
    if not fold_ids:
        raise ValueError("No folds found in candidate manifests.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_folds: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for fold_id in fold_ids:
        best: tuple[float, dict[str, Any], pd.Series | None, pd.DataFrame, dict[str, Any]] | None = None
        for candidate in candidates:
            if fold_id not in candidate["folds"]:
                continue
            current_result = result_row_for_fold(candidate["results"], fold_id)
            if current_result is None:
                continue
            start_fold = 1 if int(args.lookback_folds) <= 0 else max(1, fold_id - int(args.lookback_folds))
            history = candidate["results"][
                (candidate["results"]["fold"] >= start_fold)
                & (candidate["results"]["fold"] < fold_id)
            ].copy()
            score, stats = history_score(history, args)
            audit_rows.append(
                {
                    "fold": fold_id,
                    "candidate_label": candidate["label"],
                    "candidate_dir": str(candidate["dir"]),
                    "score": score,
                    **stats,
                    "audit_current_final": as_float(current_result.get("final_balance")),
                    "audit_current_dd": as_float(current_result.get("max_dd_pct")),
                    "audit_current_trades": as_int(current_result.get("trades")),
                }
            )
            if len(history) < int(args.min_history_folds):
                continue
            if best is None or score > best[0]:
                best = (score, candidate, current_result, history, stats)

        selection_mode = "past_mt5_feedback"
        if best is None:
            fold_bootstrap_label = bootstrap_map_labels.get(int(fold_id), bootstrap_label)
            if fold_bootstrap_label is None:
                raise ValueError(
                    f"Fold {fold_id:02d} has no eligible candidate with {args.min_history_folds} history rows "
                    "and no --bootstrap-candidate-dir was supplied."
                )
            candidate = next(candidate for candidate in candidates if candidate["label"] == fold_bootstrap_label)
            current_result = result_row_for_fold(candidate["results"], fold_id)
            if current_result is None or fold_id not in candidate["folds"]:
                raise ValueError(f"Bootstrap candidate lacks fold {fold_id:02d}")
            history = candidate["results"][candidate["results"]["fold"] < fold_id].copy()
            score, stats = history_score(history, args)
            best = (score, candidate, current_result, history, stats)
            selection_mode = "bootstrap_declared_map" if int(fold_id) in bootstrap_map_labels else "bootstrap_declared"

        score, candidate, current_result, history, stats = best
        source_fold = dict(candidate["folds"][fold_id])
        source_csv = resolve_csv(candidate["manifest_path"], str(source_fold["csv"]))
        target_csv = args.out_dir / f"fold_{fold_id:02d}_signals.csv"
        shutil.copy2(source_csv, target_csv)
        signals = sum(1 for _ in target_csv.open("r", encoding="utf-8")) - 1

        risk_pct = as_float(current_result.get("risk_pct"), as_float(source_fold.get("risk_pct"), 0.0))
        max_risk_pct = as_float(current_result.get("max_risk_pct"), risk_pct)
        max_exposure_pct = as_float(current_result.get("max_exposure_pct"), max_risk_pct)
        max_positions = as_int(current_result.get("max_positions"), as_int(source_fold.get("max_positions", source_fold.get("max_positions_hint", 1)), 1))

        selected_fold = {
            "fold": fold_id,
            "train_start": source_fold.get("train_start"),
            "train_end": source_fold.get("train_end"),
            "test_start": source_fold.get("test_start"),
            "test_end": source_fold.get("test_end"),
            "signals": signals,
            "csv": str(target_csv),
            "risk_pct": round(risk_pct, 4),
            "max_risk_pct": round(max_risk_pct, 4),
            "max_exposure_pct": round(max_exposure_pct, 4),
            "max_positions": max_positions,
            "selection_mode": selection_mode,
            "selection_uses_current_fold_metrics": False,
            "selection_history_folds": ",".join(str(int(x)) for x in sorted(history["fold"].dropna().astype(int).tolist())),
            "selected_candidate_label": candidate["label"],
            "selected_candidate_dir": str(candidate["dir"]),
            "selected_candidate_live_protocol": bool(candidate["manifest_live_protocol"]),
            "selected_candidate_adaptive_per_fold": bool(candidate["manifest_adaptive_per_fold"]),
            "selected_candidate_research_oracle_fold_selection": bool(candidate["manifest_research_oracle_fold_selection"]),
            "selected_candidate_selection_uses_current_fold_metrics": bool(candidate["manifest_selection_uses_current_fold_metrics"]),
            "selected_score_from_past_mt5": round(float(score), 6),
            **stats,
        }
        selected_folds.append(selected_fold)
        choices.append(
            {
                **selected_fold,
                "source_results": str(candidate["results_path"]),
                "audit_current_final_not_used": as_float(current_result.get("final_balance")),
                "audit_current_dd_not_used": as_float(current_result.get("max_dd_pct")),
                "audit_current_trades_not_used": as_int(current_result.get("trades")),
            }
        )

    manifest = {
        "all_signals": None,
        "folds": selected_folds,
        "total_signals": int(sum(fold["signals"] for fold in selected_folds)),
        "live_protocol": True,
        "mt5_feedback_rolling_selector": True,
        "adaptive_per_fold": False,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "candidate_dirs": [str(candidate["dir"]) for candidate in candidates],
        "selection_rule": "candidate selected from prior MT5 folds only; current fold MT5 metrics are audit-only",
        "min_history_folds": int(args.min_history_folds),
        "lookback_folds": int(args.lookback_folds),
        "bootstrap_candidate_label": bootstrap_label,
        "bootstrap_map_labels": bootstrap_map_labels,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(choices).to_csv(args.out_dir / "rolling_selector_choices.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.out_dir / "rolling_selector_candidate_audit.csv", index=False)

    print(f"manifest={args.out_dir / 'manifest.json'}")
    print(f"choices={args.out_dir / 'rolling_selector_choices.csv'}")
    print(f"candidate_audit={args.out_dir / 'rolling_selector_candidate_audit.csv'}")
    print(f"folds={len(selected_folds)} total_signals={manifest['total_signals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
