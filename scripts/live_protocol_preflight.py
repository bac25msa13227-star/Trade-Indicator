from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_path(value: str | None, manifest_path: Path) -> Path | None:
    if not value:
        return None
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(ROOT / raw)
        candidates.append(manifest_path.parent / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / raw).resolve() if not raw.is_absolute() else raw


def _as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _check_manifest(manifest: dict[str, Any], manifest_path: Path, args: argparse.Namespace) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []

    if args.require_live_protocol and not _is_true(manifest.get("live_protocol")):
        blockers.append("manifest is not marked live_protocol=true")
    if _is_true(manifest.get("adaptive_per_fold")):
        blockers.append("manifest is adaptive_per_fold; this is research/hindsight, not live protocol")
    if _is_true(manifest.get("research_oracle_fold_selection")):
        blockers.append("manifest is research_oracle_fold_selection; this is research/hindsight, not live protocol")
    if _is_true(manifest.get("selection_uses_current_fold_metrics")):
        blockers.append("manifest selection uses current fold metrics")
    source_flags = manifest.get("source_manifest_flags") if isinstance(manifest.get("source_manifest_flags"), dict) else {}
    if _is_true(source_flags.get("adaptive_per_fold")):
        message = "source manifest is adaptive_per_fold; this is research/hindsight, not live protocol"
        if args.block_adaptive_source:
            blockers.append(message)
        else:
            warnings.append(message)
    if _is_true(source_flags.get("research_oracle_fold_selection")):
        blockers.append("source manifest is research_oracle_fold_selection; this is research/hindsight, not live protocol")
    if _is_true(source_flags.get("selection_uses_current_fold_metrics")):
        blockers.append("source manifest selection uses current fold metrics")

    source_validation = manifest.get("source_validation") if isinstance(manifest.get("source_validation"), dict) else {}
    source_manifest_path = _resolve_repo_path(source_validation.get("source_manifest"), manifest_path)
    if source_manifest_path is not None and source_manifest_path.exists():
        source_manifest = _load_json(source_manifest_path)
        if _is_true(source_manifest.get("adaptive_per_fold")):
            message = f"source manifest is adaptive_per_fold: {source_manifest_path}"
            if args.block_adaptive_source:
                blockers.append(message)
            else:
                warnings.append(message)
        if _is_true(source_manifest.get("research_oracle_fold_selection")):
            blockers.append(f"source manifest is research_oracle_fold_selection: {source_manifest_path}")
        if _is_true(source_manifest.get("selection_uses_current_fold_metrics")):
            blockers.append(f"source manifest selection uses current fold metrics: {source_manifest_path}")

    folds = manifest.get("folds") or []
    if not isinstance(folds, list) or not folds:
        blockers.append("manifest has no folds")
        return blockers, warnings

    for fold in folds:
        fold_id = _as_int(fold.get("fold"))
        if _is_true(fold.get("selection_uses_current_fold_metrics")):
            blockers.append(f"fold {fold_id:02d} selection uses current fold metrics")

        mode = str(fold.get("selection_mode", "") or "")
        if args.block_bootstrap and mode.startswith("bootstrap_declared"):
            blockers.append(f"fold {fold_id:02d} uses bootstrap_declared selection")
        if args.block_warmup_no_trade and mode == "warmup_no_trade":
            blockers.append(f"fold {fold_id:02d} is warmup_no_trade; target cannot be proven on that fold")
        if _is_true(fold.get("selected_candidate_research_oracle_fold_selection")):
            blockers.append(f"fold {fold_id:02d} selected source candidate is research_oracle")
        if _is_true(fold.get("selected_candidate_selection_uses_current_fold_metrics")):
            blockers.append(f"fold {fold_id:02d} selected source candidate uses current fold metrics")
        if _is_true(fold.get("selected_candidate_adaptive_per_fold")):
            message = f"fold {fold_id:02d} selected source candidate is adaptive_per_fold"
            if args.block_adaptive_source:
                blockers.append(message)
            else:
                warnings.append(message)

        risk_pct = _as_float(fold.get("risk_pct"), 0.0)
        max_risk_pct = _as_float(fold.get("max_risk_pct"), risk_pct)
        max_exposure_pct = _as_float(fold.get("max_exposure_pct"), max_risk_pct)
        max_positions = _as_int(fold.get("max_positions", fold.get("max_positions_hint", 1)), 1)
        signals = _as_int(fold.get("signals"), 0)

        if risk_pct > args.max_risk_pct or max_risk_pct > args.max_risk_pct:
            blockers.append(
                f"fold {fold_id:02d} risk {max(risk_pct, max_risk_pct):.2f}% exceeds cap {args.max_risk_pct:.2f}%"
            )
        if max_exposure_pct > args.max_exposure_pct:
            blockers.append(
                f"fold {fold_id:02d} exposure {max_exposure_pct:.2f}% exceeds cap {args.max_exposure_pct:.2f}%"
            )
        if max_positions > args.max_positions:
            blockers.append(f"fold {fold_id:02d} max_positions {max_positions} exceeds cap {args.max_positions}")
        if signals < 0:
            blockers.append(f"fold {fold_id:02d} has negative signal count")

        csv_path = _resolve_repo_path(fold.get("csv"), manifest_path)
        if signals > 0 and (csv_path is None or not csv_path.exists()):
            blockers.append(f"fold {fold_id:02d} signal CSV missing: {fold.get('csv')}")
        if signals == 0:
            warnings.append(f"fold {fold_id:02d} has zero signals")

    return blockers, warnings


def _check_results(manifest: dict[str, Any], manifest_path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    folds = manifest.get("folds") or []
    fold_count = len(folds)

    results_path = args.results
    if results_path is None:
        results_path = manifest_path.parent / "mt5_wf_results.csv"
    if not results_path.exists():
        if args.allow_missing_results:
            warnings.append(f"MT5 results missing: {results_path}")
        else:
            blockers.append(f"MT5 results missing: {results_path}")
        return {"results_path": str(results_path), "rows": 0}, blockers, warnings

    df = pd.read_csv(results_path)
    required = {"fold", "signals", "final_balance", "max_dd_pct", "trades", "loaded_signals"}
    missing = sorted(required.difference(df.columns))
    if missing:
        blockers.append(f"MT5 results missing columns: {missing}")
        return {"results_path": str(results_path), "rows": int(len(df))}, blockers, warnings

    for col in ["fold", "signals", "final_balance", "max_dd_pct", "trades", "loaded_signals"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    target_ok = df["final_balance"] >= float(args.target_balance)
    dd_ok = df["max_dd_pct"] >= float(args.min_dd_pct)
    loss = df["final_balance"] < float(args.deposit)
    loaded_gap = (df["loaded_signals"] - df["signals"]).abs()
    loaded_ok = loaded_gap.fillna(0.0) <= float(args.max_loaded_signal_gap)
    if args.require_loaded_signal_match and not bool(loaded_ok.all()):
        bad = df.loc[~loaded_ok, "fold"].dropna().astype(int).tolist()
        blockers.append(f"loaded signal mismatch on folds: {bad}")

    manifest_signal_counts = {
        _as_int(fold.get("fold")): _as_int(fold.get("signals"))
        for fold in folds
    }
    result_signal_mismatches: list[dict[str, int]] = []
    result_loaded_mismatches: list[dict[str, int]] = []
    for _, row in df.iterrows():
        fold_id = _as_int(row.get("fold"))
        if fold_id not in manifest_signal_counts:
            continue
        expected = manifest_signal_counts[fold_id]
        result_signals = _as_int(row.get("signals"))
        result_loaded = _as_int(row.get("loaded_signals"))
        if result_signals != expected:
            result_signal_mismatches.append(
                {"fold": fold_id, "manifest_signals": expected, "result_signals": result_signals}
            )
        if result_loaded != expected:
            result_loaded_mismatches.append(
                {"fold": fold_id, "manifest_signals": expected, "loaded_signals": result_loaded}
            )
    if result_signal_mismatches:
        blockers.append(f"MT5 result signal counts do not match manifest: {result_signal_mismatches}")
    if result_loaded_mismatches:
        blockers.append(f"MT5 loaded signals do not match manifest: {result_loaded_mismatches}")

    expected_folds = sorted(_as_int(fold.get("fold")) for fold in folds)
    result_folds = sorted(df["fold"].dropna().astype(int).tolist())
    missing_folds = sorted(set(expected_folds).difference(result_folds))
    extra_folds = sorted(set(result_folds).difference(expected_folds))
    if missing_folds:
        blockers.append(f"MT5 results missing folds: {missing_folds}")
    if extra_folds:
        warnings.append(f"MT5 results include folds outside manifest: {extra_folds}")
    if len(df) < fold_count:
        blockers.append(f"MT5 result row count {len(df)} is below manifest fold count {fold_count}")

    target_fails = df.loc[~target_ok, "fold"].dropna().astype(int).tolist()
    dd_fails = df.loc[~dd_ok, "fold"].dropna().astype(int).tolist()
    loss_folds = df.loc[loss, "fold"].dropna().astype(int).tolist()
    if len(target_fails) > args.max_target_fails:
        blockers.append(f"target failures {len(target_fails)} exceed cap {args.max_target_fails}: {target_fails}")
    if len(dd_fails) > args.max_dd_fails:
        blockers.append(f"DD failures {len(dd_fails)} exceed cap {args.max_dd_fails}: {dd_fails}")
    if len(loss_folds) > args.max_loss_folds:
        blockers.append(f"loss folds {len(loss_folds)} exceed cap {args.max_loss_folds}: {loss_folds}")

    summary = {
        "results_path": str(results_path),
        "rows": int(len(df)),
        "folds": int(fold_count),
        "target_balance": float(args.target_balance),
        "target_pass_folds": int(target_ok.sum()),
        "dd_pass_folds": int(dd_ok.sum()),
        "loss_folds": int(loss.sum()),
        "target_fail_folds": target_fails,
        "dd_fail_folds": dd_fails,
        "loss_fold_ids": loss_folds,
        "min_final_balance": float(df["final_balance"].min()) if len(df) else None,
        "median_final_balance": float(df["final_balance"].median()) if len(df) else None,
        "worst_dd_pct": float(df["max_dd_pct"].min()) if len(df) else None,
        "loaded_signal_max_abs_gap": float(loaded_gap.max()) if len(df) else None,
        "manifest_result_signal_mismatches": result_signal_mismatches,
        "manifest_loaded_signal_mismatches": result_loaded_mismatches,
        "total_trades": int(df["trades"].fillna(0).sum()) if len(df) else 0,
    }
    return summary, blockers, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed gate before using a WF signal manifest for live/canary.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--min-dd-pct", type=float, default=-20.0)
    parser.add_argument("--max-target-fails", type=int, default=0)
    parser.add_argument("--max-dd-fails", type=int, default=0)
    parser.add_argument("--max-loss-folds", type=int, default=1)
    parser.add_argument("--max-risk-pct", type=float, default=4.0)
    parser.add_argument("--max-exposure-pct", type=float, default=4.0)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--max-loaded-signal-gap", type=float, default=0.0)
    parser.add_argument("--allow-missing-results", action="store_true")
    parser.add_argument("--no-require-live-protocol", dest="require_live_protocol", action="store_false")
    parser.add_argument("--no-require-loaded-signal-match", dest="require_loaded_signal_match", action="store_false")
    parser.add_argument("--allow-bootstrap", dest="block_bootstrap", action="store_false")
    parser.add_argument("--allow-warmup-no-trade", dest="block_warmup_no_trade", action="store_false")
    parser.add_argument("--block-adaptive-source", action="store_true")
    parser.add_argument("--allow-adaptive-source", dest="block_adaptive_source", action="store_false")
    parser.add_argument("--no-fail-exit", action="store_true")
    parser.set_defaults(
        require_live_protocol=True,
        require_loaded_signal_match=True,
        block_bootstrap=True,
        block_warmup_no_trade=True,
        block_adaptive_source=True,
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = _load_json(manifest_path)
    blockers, warnings = _check_manifest(manifest, manifest_path, args)
    result_summary, result_blockers, result_warnings = _check_results(manifest, manifest_path, args)
    blockers.extend(result_blockers)
    warnings.extend(result_warnings)

    report = {
        "manifest": str(manifest_path),
        "live_allowed": len(blockers) == 0,
        "blockers": blockers,
        "warnings": warnings,
        "manifest_flags": {
            "live_protocol": _is_true(manifest.get("live_protocol")),
            "adaptive_per_fold": _is_true(manifest.get("adaptive_per_fold")),
            "research_oracle_fold_selection": _is_true(manifest.get("research_oracle_fold_selection")),
            "selection_uses_current_fold_metrics": _is_true(manifest.get("selection_uses_current_fold_metrics")),
        },
        "gate": {
            "deposit": float(args.deposit),
            "target_balance": float(args.target_balance),
            "min_dd_pct": float(args.min_dd_pct),
            "max_target_fails": int(args.max_target_fails),
            "max_dd_fails": int(args.max_dd_fails),
            "max_loss_folds": int(args.max_loss_folds),
            "max_risk_pct": float(args.max_risk_pct),
            "max_exposure_pct": float(args.max_exposure_pct),
            "max_positions": int(args.max_positions),
        },
        "results": result_summary,
    }

    out_path = args.out or (manifest_path.parent / "live_preflight_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["live_allowed"] or args.no_fail_exit:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
