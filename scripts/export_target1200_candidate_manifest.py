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
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.train_target1200_signal_universe import (  # noqa: E402
    available_feature_columns,
    build_selected_indices,
    first_hit_outcomes,
    fit_predict_oos,
    fold_timestamps,
    load_feature_frame,
    load_folds,
    parse_hours,
)


MT5_COLUMNS = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]


def _pick_candidate(search: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.search_row is not None:
        if args.search_row < 0 or args.search_row >= len(search):
            raise ValueError(f"--search-row out of range: {args.search_row}")
        return search.iloc[int(args.search_row)].copy()

    if args.fold is None:
        required = [
            args.tp_rr,
            args.sl_mult,
            args.horizon_bars,
            args.min_probability,
            args.top_k,
            args.risk_fraction if args.risk_pct is None else args.risk_pct,
        ]
        if any(value is None for value in required):
            raise ValueError("Provide --fold with --search-csv ranking, --search-row, or explicit candidate params.")
        return pd.Series(
            {
                "tp_rr": args.tp_rr,
                "sl_mult": args.sl_mult,
                "horizon_bars": args.horizon_bars,
                "min_probability": args.min_probability,
                "top_k_per_fold": args.top_k,
                "risk_pct": (float(args.risk_pct) / 100.0) if args.risk_pct is not None else args.risk_fraction,
                "max_positions": args.max_positions,
            }
        )

    final_col = f"fold_{int(args.fold):02d}_final"
    dd_col = f"fold_{int(args.fold):02d}_dd"
    trades_col = f"fold_{int(args.fold):02d}_trades"
    missing = [col for col in [final_col, dd_col, trades_col] if col not in search.columns]
    if missing:
        raise ValueError(f"Search CSV lacks columns: {missing}")

    candidates = search.copy()
    candidates[final_col] = candidates[final_col].astype(float)
    candidates[dd_col] = candidates[dd_col].astype(float)
    candidates[trades_col] = candidates[trades_col].astype(float)

    candidates = candidates[candidates[trades_col] > 0].copy()
    if args.min_proxy_final is not None:
        candidates = candidates[candidates[final_col] >= args.min_proxy_final].copy()
    if args.min_proxy_dd is not None:
        candidates = candidates[candidates[dd_col] >= args.min_proxy_dd].copy()
    if args.max_risk_fraction is not None:
        candidates = candidates[candidates["risk_pct"].astype(float) <= args.max_risk_fraction].copy()
    if args.tp_rr is not None:
        candidates = candidates[np.isclose(candidates["tp_rr"].astype(float), float(args.tp_rr))].copy()
    if args.sl_mult is not None:
        candidates = candidates[np.isclose(candidates["sl_mult"].astype(float), float(args.sl_mult))].copy()
    if args.horizon_bars is not None:
        candidates = candidates[candidates["horizon_bars"].astype(int) == int(args.horizon_bars)].copy()
    if args.min_probability is not None:
        candidates = candidates[np.isclose(candidates["min_probability"].astype(float), float(args.min_probability))].copy()
    if args.top_k is not None:
        candidates = candidates[candidates["top_k_per_fold"].astype(int) == int(args.top_k)].copy()
    if args.risk_fraction is not None:
        candidates = candidates[np.isclose(candidates["risk_pct"].astype(float), float(args.risk_fraction))].copy()

    if candidates.empty:
        raise ValueError("No candidate remains after filters.")

    candidates["fold_final"] = candidates[final_col].astype(float)
    candidates["fold_dd"] = candidates[dd_col].astype(float)
    candidates["fold_trades"] = candidates[trades_col].astype(float)
    candidates["dd_buffer"] = candidates["fold_dd"] + 20.0
    candidates["target_buffer"] = candidates["fold_final"] - 1200.0
    candidates = candidates.sort_values(
        ["target_buffer", "dd_buffer", "fold_final", "risk_pct", "top_k_per_fold"],
        ascending=[False, False, False, True, True],
    )
    rank = int(args.rank)
    if rank < 0 or rank >= len(candidates):
        raise ValueError(f"--rank out of range: {rank}; candidates={len(candidates)}")
    return candidates.iloc[rank].copy()


def _build_scored_frame(
    frame: pd.DataFrame,
    folds: list[dict[str, Any]],
    feature_cols: list[str],
    tp_rr: float,
    sl_mult: float,
    horizon_bars: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    open_ = frame["open"].to_numpy(np.float64)
    high = frame["high"].to_numpy(np.float64)
    low = frame["low"].to_numpy(np.float64)
    close = frame["close"].to_numpy(np.float64)
    atr = frame["atr"].to_numpy(np.float64)
    direction = frame["direction"].to_numpy(np.int8)
    rr_all, close_idx_all = first_hit_outcomes(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr=atr,
        direction=direction,
        tp_rr=tp_rr,
        sl_atr_mult=sl_mult,
        horizon_bars=horizon_bars,
        entry_delay_bars=args.entry_delay_bars,
    )
    scored = fit_predict_oos(
        frame=frame,
        folds=folds,
        feature_cols=feature_cols,
        labels=rr_all,
        horizon_bars=horizon_bars,
        args=args,
    )

    source_idx = scored["source_idx"].to_numpy(np.int64)
    execution_idx = source_idx + args.entry_delay_bars
    valid_execution = execution_idx < len(frame)
    safe_execution_idx = np.where(valid_execution, execution_idx, source_idx)
    scored["feature_time"] = scored["time"]
    scored["time"] = frame["time"].iloc[safe_execution_idx].to_numpy()
    scored["outcome_rr"] = rr_all[source_idx]
    scored["close_idx"] = close_idx_all[source_idx]
    scored["start_idx"] = execution_idx
    scored["entry_price"] = open_[safe_execution_idx]
    scored.loc[~valid_execution, "entry_price"] = np.nan
    scored["sl_dist"] = scored["atr"].astype(float) * sl_mult
    is_buy = scored["direction"].astype(int) == 1
    scored["sl_price"] = np.where(
        is_buy,
        scored["entry_price"] - scored["sl_dist"],
        scored["entry_price"] + scored["sl_dist"],
    )
    scored["tp_price"] = np.where(
        is_buy,
        scored["entry_price"] + scored["sl_dist"] * tp_rr,
        scored["entry_price"] - scored["sl_dist"] * tp_rr,
    )

    time_values = frame["time"].astype("int64").to_numpy()
    fold_end_by_id = {
        int(fold["fold"]): int(
            np.searchsorted(
                time_values,
                pd.Timestamp(fold["test_end"], tz="UTC").value,
                side="left",
            )
        )
        for fold in folds
    }
    fold_end_idx = scored["fold"].map(fold_end_by_id).to_numpy(np.int64)
    invalid = (
        (scored["close_idx"].to_numpy(np.int64) < 0)
        | (scored["close_idx"].to_numpy(np.int64) >= fold_end_idx)
        | (scored["start_idx"].to_numpy(np.int64) >= fold_end_idx)
    )
    scored.loc[invalid, "outcome_rr"] = 0.0
    scored.loc[invalid, "close_idx"] = -1
    return scored


def _write_manifest(
    scored: pd.DataFrame,
    folds: list[dict[str, Any]],
    out_dir: Path,
    broker_gmt: int,
    min_probability: float,
    top_k: int,
    risk_fraction: float,
    max_positions: int,
    only_fold: int | None,
    min_signal_gap_bars: int,
    select_lowest_probability: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds_array = scored["fold"].to_numpy(np.int64)
    selection_scores = scored["probability"].to_numpy(np.float64)
    selection_threshold = min_probability
    if select_lowest_probability:
        selection_scores = -selection_scores
        selection_threshold = -1.0
    selected = build_selected_indices(
        folds_array=folds_array,
        proba=selection_scores,
        start_idx=scored["start_idx"].to_numpy(np.int64),
        min_probability=selection_threshold,
        top_k=top_k,
        fold_count=len(folds),
        min_signal_gap_bars=min_signal_gap_bars,
    )
    if only_fold is not None:
        selected = selected[folds_array[selected] == int(only_fold)]

    export = scored.iloc[selected].sort_values(["fold", "time"]).copy()
    signal_time = export["feature_time"] if "feature_time" in export.columns else export["time"]
    export["open_time"] = (
        pd.to_datetime(signal_time, utc=True) + pd.Timedelta(hours=broker_gmt)
    ).dt.strftime("%Y.%m.%d %H:%M")

    if export.empty:
        all_export = pd.DataFrame(columns=MT5_COLUMNS)
    else:
        all_export = export[MT5_COLUMNS]
    all_export.to_csv(out_dir / "all_signals.csv", index=False, float_format="%.5f")

    risk_percent = round(float(risk_fraction) * 100.0, 4)
    max_exposure_percent = round(risk_percent * int(max_positions), 4)
    manifest: dict[str, Any] = {
        "all_signals": str(out_dir / "all_signals.csv"),
        "folds": [],
        "total_signals": int(len(export)),
        "single_candidate_export": True,
    }
    fold_iter = [fold for fold in folds if only_fold is None or int(fold["fold"]) == int(only_fold)]
    for fold in fold_iter:
        fold_id = int(fold["fold"])
        fold_rows = export[export["fold"].astype(int) == fold_id].copy() if not export.empty else export
        fold_path = out_dir / f"fold_{fold_id:02d}_signals.csv"
        if fold_rows.empty:
            pd.DataFrame(columns=MT5_COLUMNS).to_csv(fold_path, index=False)
        else:
            fold_rows[MT5_COLUMNS].to_csv(fold_path, index=False, float_format="%.5f")
        manifest["folds"].append(
            {
                "fold": fold_id,
                "train_start": fold.get("train_start"),
                "train_end": fold.get("train_end"),
                "test_start": fold.get("test_start"),
                "test_end": fold.get("test_end"),
                "signals": int(len(fold_rows)),
                "csv": str(fold_path),
                "risk_pct": risk_percent,
                "max_risk_pct": risk_percent,
                "max_exposure_pct": max_exposure_percent,
                "max_positions_hint": int(max_positions),
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export one target1200 candidate manifest for MT5 validation.")
    parser.add_argument("--features", type=Path, default=ROOT / "outputs/mt5_full_ict_wyckoff_features_202306_202603.csv")
    parser.add_argument("--fold-manifest", type=Path, default=ROOT / "outputs/mt5_wf_r2_5_tradeside/manifest.json")
    parser.add_argument("--search-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--broker-gmt", type=int, default=0)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--only-fold", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--search-row", type=int, default=None)
    parser.add_argument("--min-proxy-final", type=float, default=None)
    parser.add_argument("--min-proxy-dd", type=float, default=None)
    parser.add_argument("--max-risk-fraction", type=float, default=None)
    parser.add_argument("--tp-rr", type=float, default=None)
    parser.add_argument("--sl-mult", type=float, default=None)
    parser.add_argument("--horizon-bars", type=int, default=None)
    parser.add_argument("--min-probability", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--risk-fraction", type=float, default=None)
    parser.add_argument("--risk-pct", type=float, default=None, help="Risk in percent, e.g. 7 means 7%.")
    parser.add_argument("--select-lowest-probability", action="store_true")
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--direction-mode", choices=["trade_side", "reverse_trade_side", "buy", "sell"], default="trade_side")
    parser.add_argument("--side", choices=["all", "buy", "sell"], default="all")
    parser.add_argument("--include-broker-hours", default=None)
    parser.add_argument("--exclude-broker-hours", default=None)
    parser.add_argument("--min-atr-percentile", type=float, default=None)
    parser.add_argument("--max-atr-percentile", type=float, default=None)
    parser.add_argument("--exclude-news-blackout", action="store_true")
    parser.add_argument("--candidate-stride", type=int, default=1)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
    parser.add_argument("--min-signal-gap-bars", type=int, default=None)
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

    if args.search_csv is None:
        row = _pick_candidate(pd.DataFrame(), args)
    else:
        row = _pick_candidate(pd.read_csv(args.search_csv), args)

    tp_rr = float(row["tp_rr"])
    sl_mult = float(row["sl_mult"])
    horizon_bars = int(row["horizon_bars"])
    min_probability = float(row["min_probability"])
    top_k = int(row["top_k_per_fold"])
    risk_fraction = float(row["risk_pct"])
    if args.risk_pct is not None and args.risk_fraction is not None:
        raise ValueError("Use either --risk-pct or --risk-fraction, not both.")
    if args.risk_pct is not None:
        risk_fraction = float(args.risk_pct) / 100.0
    elif args.risk_fraction is not None:
        risk_fraction = float(args.risk_fraction)
    if args.top_k is not None:
        top_k = int(args.top_k)
    if args.min_probability is not None:
        min_probability = float(args.min_probability)
    min_signal_gap_bars = (
        int(args.min_signal_gap_bars)
        if args.min_signal_gap_bars is not None
        else int(row.get("min_signal_gap_bars", 0) or 0)
    )

    folds = load_folds(args.fold_manifest)
    feature_cols = available_feature_columns(args.features)
    frame = load_feature_frame(
        features_path=args.features,
        feature_cols=feature_cols,
        folds=folds,
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
    model_args = SimpleNamespace(**vars(args))
    scored = _build_scored_frame(
        frame=frame,
        folds=folds,
        feature_cols=feature_cols,
        tp_rr=tp_rr,
        sl_mult=sl_mult,
        horizon_bars=horizon_bars,
        args=model_args,
    )
    manifest = _write_manifest(
        scored=scored,
        folds=folds,
        out_dir=args.out_dir,
        broker_gmt=args.broker_gmt,
        min_probability=min_probability,
        top_k=top_k,
        risk_fraction=risk_fraction,
        max_positions=args.max_positions,
        only_fold=args.fold if args.only_fold else None,
        min_signal_gap_bars=min_signal_gap_bars,
        select_lowest_probability=bool(args.select_lowest_probability),
    )
    selected = {
        "tp_rr": tp_rr,
        "sl_mult": sl_mult,
        "horizon_bars": horizon_bars,
        "min_probability": min_probability,
        "top_k_per_fold": top_k,
        "risk_pct": risk_fraction,
        "max_positions": int(args.max_positions),
        "min_signal_gap_bars": int(min_signal_gap_bars),
        "select_lowest_probability": bool(args.select_lowest_probability),
        "fold": args.fold,
        "rank": int(args.rank),
        "search_row": args.search_row,
        "source_search_csv": str(args.search_csv) if args.search_csv is not None else None,
    }
    if args.fold is not None:
        fold_id = int(args.fold)
        for suffix in ["final", "dd", "trades"]:
            col = f"fold_{fold_id:02d}_{suffix}"
            if col in row.index:
                selected[f"proxy_{suffix}"] = float(row[col])
    (args.out_dir / "candidate_selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2))
    print(f"manifest={args.out_dir / 'manifest.json'}")
    print(f"total_signals={manifest['total_signals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
