from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_target1200_candidate_manifest import MT5_COLUMNS, _build_scored_frame  # noqa: E402
from scripts.train_target1200_signal_universe import (  # noqa: E402
    available_feature_columns,
    load_feature_frame,
    parse_hours,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _date_text(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%d")


def _parse_now(value: str | None) -> pd.Timestamp:
    if value:
        return _utc(value)
    return pd.Timestamp.now(tz="UTC")


def _generate_missing_folds(base_folds: list[dict[str, Any]], now_utc: pd.Timestamp) -> list[dict[str, Any]]:
    if not base_folds:
        raise ValueError("base manifest has no folds")
    last = dict(sorted(base_folds, key=lambda row: int(row["fold"]))[-1])
    generated: list[dict[str, Any]] = []
    while _utc(last["test_end"]) <= now_utc:
        fold_id = int(last["fold"]) + 1
        test_start = _utc(last["test_end"])
        test_end = test_start + pd.DateOffset(months=1)
        train_start = _utc(last["train_start"]) + pd.DateOffset(months=1)
        train_end = test_start - pd.Timedelta(days=1)
        next_fold = {
            "fold": fold_id,
            "train_start": _date_text(train_start),
            "train_end": _date_text(train_end),
            "test_start": _date_text(test_start),
            "test_end": _date_text(test_end),
        }
        generated.append(next_fold)
        last = next_fold
    return generated


def _latest_validated_recipe(source_manifest: dict[str, Any], max_fold: int, source_fold: int | None) -> dict[str, Any]:
    folds = sorted(source_manifest.get("folds", []) or [], key=lambda row: int(row["fold"]), reverse=True)
    if source_fold is not None:
        folds = [row for row in folds if int(row["fold"]) == int(source_fold)]
    else:
        folds = [row for row in folds if int(row["fold"]) <= int(max_fold)]
    required = ["selected_tp_rr", "selected_sl_mult", "selected_horizon_bars", "selected_top_k_per_fold"]
    for fold in folds:
        if all(key in fold and fold[key] is not None for key in required):
            max_positions = fold.get("max_positions", fold.get("max_positions_hint", 1))
            return {
                "source_fold": int(fold["fold"]),
                "tp_rr": float(fold["selected_tp_rr"]),
                "sl_mult": float(fold["selected_sl_mult"]),
                "horizon_bars": int(fold["selected_horizon_bars"]),
                "min_probability": float(fold.get("selected_min_probability", 0.0) or 0.0),
                "top_k_per_fold": int(fold["selected_top_k_per_fold"]),
                "risk_pct": float(fold.get("risk_pct", 0.0) or 0.0),
                "max_risk_pct": float(fold.get("max_risk_pct", fold.get("risk_pct", 0.0)) or 0.0),
                "max_exposure_pct": float(fold.get("max_exposure_pct", fold.get("risk_pct", 0.0)) or 0.0),
                "max_positions": int(max_positions or 1),
                "mt5_validated_final_balance": fold.get("mt5_validated_final_balance"),
                "mt5_validated_max_dd_pct": fold.get("mt5_validated_max_dd_pct"),
            }
    raise ValueError("No concrete selected_* recipe found in source manifest")


def _apply_recipe_overrides(recipe: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = dict(recipe)
    overrides = {
        "tp_rr": args.tp_rr,
        "sl_mult": args.sl_mult,
        "horizon_bars": args.horizon_bars,
        "min_probability": args.min_probability,
        "top_k_per_fold": args.top_k,
        "risk_pct": args.risk_pct,
        "max_positions": args.max_positions,
    }
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    if args.max_risk_pct is not None:
        result["max_risk_pct"] = args.max_risk_pct
    elif args.risk_pct is not None:
        result["max_risk_pct"] = args.risk_pct
    if args.max_exposure_pct is not None:
        result["max_exposure_pct"] = args.max_exposure_pct
    elif args.risk_pct is not None and args.max_positions is not None:
        result["max_exposure_pct"] = float(args.risk_pct) * int(args.max_positions)
    elif args.risk_pct is not None:
        result["max_exposure_pct"] = args.risk_pct
    result["tp_rr"] = float(result["tp_rr"])
    result["sl_mult"] = float(result["sl_mult"])
    result["horizon_bars"] = int(result["horizon_bars"])
    result["min_probability"] = float(result["min_probability"])
    result["top_k_per_fold"] = int(result["top_k_per_fold"])
    result["risk_pct"] = float(result["risk_pct"])
    result["max_risk_pct"] = float(result.get("max_risk_pct", result["risk_pct"]))
    result["max_exposure_pct"] = float(result.get("max_exposure_pct", result["risk_pct"]))
    result["max_positions"] = int(result["max_positions"])
    return result


def _select_indices(
    scored: pd.DataFrame,
    folds: list[dict[str, Any]],
    min_probability: float,
    top_k: int,
    min_signal_gap_bars: int,
) -> np.ndarray:
    selected_parts: list[np.ndarray] = []
    folds_array = scored["fold"].to_numpy(np.int64)
    proba = scored["probability"].to_numpy(np.float64)
    start_idx = scored["start_idx"].to_numpy(np.int64)
    finite_trade_plan = (
        np.isfinite(scored["entry_price"].to_numpy(np.float64))
        & np.isfinite(scored["sl_price"].to_numpy(np.float64))
        & np.isfinite(scored["tp_price"].to_numpy(np.float64))
        & np.isfinite(scored["atr"].to_numpy(np.float64))
    )
    for fold in folds:
        fold_id = int(fold["fold"])
        idx = np.flatnonzero((folds_array == fold_id) & (proba >= float(min_probability)) & finite_trade_plan)
        if idx.size == 0:
            continue
        order = np.lexsort((start_idx[idx], -proba[idx]))
        idx = idx[order]
        if min_signal_gap_bars > 0 and idx.size > 1:
            kept: list[int] = []
            kept_starts: list[int] = []
            for row in idx:
                row_start = int(start_idx[row])
                if all(abs(row_start - seen) >= min_signal_gap_bars for seen in kept_starts):
                    kept.append(int(row))
                    kept_starts.append(row_start)
                    if top_k > 0 and len(kept) >= top_k:
                        break
            idx = np.asarray(kept, dtype=np.int64)
        if top_k > 0:
            idx = idx[:top_k]
        selected_parts.append(idx)
    if not selected_parts:
        return np.empty(0, dtype=np.int64)
    selected = np.concatenate(selected_parts).astype(np.int64)
    return selected[np.lexsort((start_idx[selected], folds_array[selected]))]


def _write_current_manifest(
    scored: pd.DataFrame,
    selected_idx: np.ndarray,
    folds: list[dict[str, Any]],
    out_dir: Path,
    broker_gmt: int,
    recipe: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    export = scored.iloc[selected_idx].sort_values(["fold", "time"]).copy() if len(selected_idx) else scored.iloc[[]].copy()
    signal_time = export["feature_time"] if "feature_time" in export.columns else export.get("time", pd.Series(dtype="datetime64[ns, UTC]"))
    if export.empty:
        all_export = pd.DataFrame(columns=MT5_COLUMNS)
    else:
        export["open_time"] = (
            pd.to_datetime(signal_time, utc=True) + pd.Timedelta(hours=broker_gmt)
        ).dt.strftime("%Y.%m.%d %H:%M")
        all_export = export[MT5_COLUMNS]
    all_path = out_dir / "all_signals.csv"
    all_export.to_csv(all_path, index=False, float_format="%.5f")

    manifest: dict[str, Any] = {
        "all_signals": str(all_path),
        "folds": [],
        "total_signals": int(len(all_export)),
        "live_protocol": True,
        "current_campaign": True,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "adaptive_per_fold": False,
        "source_policy": "reuse_latest_mt5_validated_recipe_from_past_fold",
        "source_recipe": recipe,
        **metadata,
    }
    empty = pd.DataFrame(columns=MT5_COLUMNS)
    for fold in folds:
        fold_id = int(fold["fold"])
        fold_rows = export[export["fold"].astype(int) == fold_id].copy() if not export.empty else export
        fold_path = out_dir / f"fold_{fold_id:02d}_signals.csv"
        if fold_rows.empty:
            empty.to_csv(fold_path, index=False)
            max_feature_time = None
        else:
            fold_rows[MT5_COLUMNS].to_csv(fold_path, index=False, float_format="%.5f")
            max_feature_time = pd.to_datetime(fold_rows["feature_time"], utc=True).max().isoformat()
        manifest["folds"].append(
            {
                **fold,
                "signals": int(len(fold_rows)),
                "csv": str(fold_path),
                "risk_pct": float(recipe["risk_pct"]),
                "max_risk_pct": float(recipe["max_risk_pct"]),
                "max_exposure_pct": float(recipe["max_exposure_pct"]),
                "max_positions": int(recipe["max_positions"]),
                "selected_tp_rr": float(recipe["tp_rr"]),
                "selected_sl_mult": float(recipe["sl_mult"]),
                "selected_horizon_bars": int(recipe["horizon_bars"]),
                "selected_min_probability": float(recipe["min_probability"]),
                "selected_top_k_per_fold": int(recipe["top_k_per_fold"]),
                "max_selected_feature_time": max_feature_time,
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a rolling_v3 current campaign manifest from fresh MT5 features.")
    parser.add_argument("--features", type=Path, default=ROOT / "outputs/mt5_full_ict_wyckoff_features_202306_20260513.csv")
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "outputs/target1200_live_rolling_v3/manifest.json")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "outputs/target1200_mt5full_ict_adaptive_combo_mt5v2/manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/target1200_live_rolling_v3_current")
    parser.add_argument("--now", default=None)
    parser.add_argument("--source-fold", type=int, default=None)
    parser.add_argument("--tp-rr", type=float, default=None)
    parser.add_argument("--sl-mult", type=float, default=None)
    parser.add_argument("--horizon-bars", type=int, default=None)
    parser.add_argument("--min-probability", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--risk-pct", type=float, default=None, help="Risk percent, e.g. 4.0 for 4%.")
    parser.add_argument("--max-risk-pct", type=float, default=None)
    parser.add_argument("--max-exposure-pct", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--broker-gmt", type=int, default=0)
    parser.add_argument("--include-broker-hours", default=None)
    parser.add_argument("--exclude-broker-hours", default=None)
    parser.add_argument("--side", choices=["all", "buy", "sell"], default="all")
    parser.add_argument("--direction-mode", choices=["trade_side", "reverse_trade_side", "buy", "sell"], default="trade_side")
    parser.add_argument("--min-atr-percentile", type=float, default=None)
    parser.add_argument("--max-atr-percentile", type=float, default=None)
    parser.add_argument("--exclude-news-blackout", action="store_true")
    parser.add_argument("--candidate-stride", type=int, default=1)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--min-signal-gap-bars", type=int, default=0)
    parser.add_argument("--model-type", choices=["hgb", "extra_trees"], default="hgb")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    parser.add_argument("--positive-weight", type=float, default=1.5)
    parser.add_argument("--recency-half-life-days", type=float, default=0.0)
    parser.add_argument("--min-train-rows", type=int, default=1000)
    args = parser.parse_args()

    now_utc = _parse_now(args.now)
    base_manifest = _read_json(args.base_manifest)
    source_manifest = _read_json(args.source_manifest)
    base_folds = sorted(base_manifest.get("folds", []) or [], key=lambda row: int(row["fold"]))
    current_folds = _generate_missing_folds(base_folds, now_utc)
    if not current_folds:
        raise RuntimeError("base manifest already covers now; no current fold generated")
    max_base_fold = max(int(row["fold"]) for row in base_folds)
    recipe = _latest_validated_recipe(source_manifest, max_fold=max_base_fold, source_fold=args.source_fold)
    recipe = _apply_recipe_overrides(recipe, args)

    feature_cols = available_feature_columns(args.features)
    frame = load_feature_frame(
        features_path=args.features,
        feature_cols=feature_cols,
        folds=current_folds,
        broker_gmt=args.broker_gmt,
        include_hours=parse_hours(args.include_broker_hours),
        exclude_hours=parse_hours(args.exclude_broker_hours),
        side=args.side,
        direction_mode=args.direction_mode,
        min_atr_percentile=args.min_atr_percentile,
        max_atr_percentile=args.max_atr_percentile,
        exclude_news_blackout=args.exclude_news_blackout,
        candidate_stride=args.candidate_stride,
    )
    scored = _build_scored_frame(
        frame=frame,
        folds=current_folds,
        feature_cols=feature_cols,
        tp_rr=float(recipe["tp_rr"]),
        sl_mult=float(recipe["sl_mult"]),
        horizon_bars=int(recipe["horizon_bars"]),
        args=SimpleNamespace(**vars(args)),
    )
    selected_idx = _select_indices(
        scored=scored,
        folds=current_folds,
        min_probability=float(recipe["min_probability"]),
        top_k=int(recipe["top_k_per_fold"]),
        min_signal_gap_bars=int(args.min_signal_gap_bars),
    )
    current_fold = next(
        (
            fold
            for fold in current_folds
            if _utc(fold["test_start"]) <= now_utc < _utc(fold["test_end"])
        ),
        current_folds[-1],
    )
    manifest = _write_current_manifest(
        scored=scored,
        selected_idx=selected_idx,
        folds=current_folds,
        out_dir=args.out_dir,
        broker_gmt=args.broker_gmt,
        recipe=recipe,
        metadata={
            "now_utc": now_utc.isoformat(),
            "base_manifest": str(args.base_manifest),
            "source_manifest": str(args.source_manifest),
            "features": str(args.features),
            "current_fold_id": int(current_fold["fold"]),
            "online_update_required": True,
            "online_update_command": "python scripts/export_rolling_v3_current_campaign.py --features <fresh_features>",
        },
    )
    print(json.dumps({"manifest": str(args.out_dir / "manifest.json"), "current_fold": manifest["current_fold_id"], "total_signals": manifest["total_signals"], "recipe": recipe}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
