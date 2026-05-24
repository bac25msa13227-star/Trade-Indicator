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


def _choose_rows(
    search: pd.DataFrame,
    folds: list[dict[str, Any]],
    preferred_min_dd_pct: float,
    max_risk_fraction: float | None,
) -> pd.DataFrame:
    selected: list[pd.Series] = []
    if max_risk_fraction is not None:
        search = search[search["risk_pct"].astype(float) <= max_risk_fraction].copy()
        if search.empty:
            raise ValueError("--max-risk-fraction filtered out every search row")
    for fold in folds:
        fold_id = int(fold["fold"])
        final_col = f"fold_{fold_id:02d}_final"
        dd_col = f"fold_{fold_id:02d}_dd"
        trades_col = f"fold_{fold_id:02d}_trades"
        if final_col not in search.columns or dd_col not in search.columns:
            raise ValueError(f"Search CSV lacks {final_col}/{dd_col}; rerun trainer with per-fold columns.")

        candidates = search[
            (search[final_col].astype(float) >= 1200.0)
            & (search[dd_col].astype(float) >= preferred_min_dd_pct)
            & (search[trades_col].astype(float) > 0)
        ].copy()
        if candidates.empty:
            candidates = search[
                (search[final_col].astype(float) >= 1200.0)
                & (search[dd_col].astype(float) >= -20.0)
                & (search[trades_col].astype(float) > 0)
            ].copy()
        if candidates.empty:
            candidates = search[(search[dd_col].astype(float) >= -20.0)].copy()
        if candidates.empty:
            candidates = search.copy()

        candidates["fold_final"] = candidates[final_col].astype(float)
        candidates["fold_dd"] = candidates[dd_col].astype(float)
        candidates["fold_trades"] = candidates[trades_col].astype(float)
        candidates["dd_buffer"] = candidates["fold_dd"] + 20.0
        candidates["target_buffer"] = candidates["fold_final"] - 1200.0
        candidates = candidates.sort_values(
            [
                "target_buffer",
                "dd_buffer",
                "fold_final",
                "risk_pct",
                "top_k_per_fold",
            ],
            ascending=[False, False, False, True, True],
        )
        row = candidates.iloc[0].copy()
        row["fold"] = fold_id
        selected.append(row)
    return pd.DataFrame(selected)


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
    execution_times = frame["time"].iloc[safe_execution_idx].to_numpy()
    scored["feature_time"] = scored["time"]
    scored["time"] = execution_times
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
    selections: pd.DataFrame,
    folds: list[dict[str, Any]],
    out_dir: Path,
    broker_gmt: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds_array = scored["fold"].to_numpy(np.int64)
    proba = scored["probability"].to_numpy(np.float64)
    start_idx = scored["start_idx"].to_numpy(np.int64)

    all_exports: list[pd.DataFrame] = []
    manifest: dict[str, Any] = {
        "all_signals": str(out_dir / "all_signals.csv"),
        "folds": [],
        "total_signals": 0,
        "adaptive_per_fold": True,
    }

    for fold in folds:
        fold_id = int(fold["fold"])
        choice = selections[selections["fold"].astype(int) == fold_id].iloc[0]
        selected = build_selected_indices(
            folds_array=folds_array,
            proba=proba,
            start_idx=start_idx,
            min_probability=float(choice["min_probability"]),
            top_k=int(choice["top_k_per_fold"]),
            fold_count=len(folds),
            min_signal_gap_bars=int(choice.get("min_signal_gap_bars", 0) or 0),
        )
        selected = selected[folds_array[selected] == fold_id]
        export = scored.iloc[selected].sort_values(["fold", "time"]).copy()
        signal_time = export["feature_time"] if "feature_time" in export.columns else export["time"]
        # AI_Signal_Reader matches CSV time to the previous M5 bar and enters on the new bar.
        # Keep entry_price as the next-open reference, but write the feature bar time to CSV.
        export["open_time"] = (
            pd.to_datetime(signal_time, utc=True) + pd.Timedelta(hours=broker_gmt)
        ).dt.strftime("%Y.%m.%d %H:%M")
        fold_path = out_dir / f"fold_{fold_id:02d}_signals.csv"
        if export.empty:
            pd.DataFrame(columns=MT5_COLUMNS).to_csv(fold_path, index=False)
        else:
            export[MT5_COLUMNS].to_csv(fold_path, index=False, float_format="%.5f")
            all_exports.append(export[MT5_COLUMNS])

        risk_percent = round(float(choice["risk_pct"]) * 100.0, 4)
        manifest["folds"].append(
            {
                "fold": fold_id,
                "train_start": fold.get("train_start"),
                "train_end": fold.get("train_end"),
                "test_start": fold.get("test_start"),
                "test_end": fold.get("test_end"),
                "signals": int(len(export)),
                "csv": str(fold_path),
                "risk_pct": risk_percent,
                "max_risk_pct": risk_percent,
                "max_exposure_pct": risk_percent,
                "selected_tp_rr": float(choice["tp_rr"]),
                "selected_sl_mult": float(choice["sl_mult"]),
                "selected_horizon_bars": int(choice["horizon_bars"]),
                "selected_min_probability": float(choice["min_probability"]),
                "selected_top_k_per_fold": int(choice["top_k_per_fold"]),
                "proxy_final_balance": float(choice[f"fold_{fold_id:02d}_final"]),
                "proxy_max_dd_pct": float(choice[f"fold_{fold_id:02d}_dd"]),
                "proxy_trades": int(choice[f"fold_{fold_id:02d}_trades"]),
            }
        )

    if all_exports:
        pd.concat(all_exports, ignore_index=True).to_csv(
            out_dir / "all_signals.csv", index=False, float_format="%.5f"
        )
    else:
        pd.DataFrame(columns=MT5_COLUMNS).to_csv(out_dir / "all_signals.csv", index=False)
    manifest["total_signals"] = int(sum(int(row["signals"]) for row in manifest["folds"]))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a fold-adaptive target1200 MT5 signal universe from a per-fold search CSV."
    )
    parser.add_argument("--features", type=Path, default=ROOT / "outputs/mt5_full_ict_wyckoff_features_202306_202603.csv")
    parser.add_argument("--fold-manifest", type=Path, default=ROOT / "outputs/mt5_wf_r2_5_tradeside/manifest.json")
    parser.add_argument("--search-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preferred-min-dd-pct", type=float, default=-20.0)
    parser.add_argument("--max-risk-fraction", type=float, default=None)
    parser.add_argument("--broker-gmt", type=int, default=0)
    parser.add_argument("--direction-mode", choices=["trade_side", "reverse_trade_side", "buy", "sell"], default="trade_side")
    parser.add_argument("--side", choices=["all", "buy", "sell"], default="all")
    parser.add_argument("--include-broker-hours", default=None)
    parser.add_argument("--exclude-broker-hours", default=None)
    parser.add_argument("--min-atr-percentile", type=float, default=None)
    parser.add_argument("--max-atr-percentile", type=float, default=None)
    parser.add_argument("--exclude-news-blackout", action="store_true")
    parser.add_argument("--candidate-stride", type=int, default=1)
    parser.add_argument("--entry-delay-bars", type=int, default=1)
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

    folds = load_folds(args.fold_manifest)
    search = pd.read_csv(args.search_csv)
    choices = _choose_rows(
        search,
        folds,
        preferred_min_dd_pct=args.preferred_min_dd_pct,
        max_risk_fraction=args.max_risk_fraction,
    )
    tp_rrs = sorted(choices["tp_rr"].astype(float).unique())
    sl_mults = sorted(choices["sl_mult"].astype(float).unique())
    horizons = sorted(choices["horizon_bars"].astype(int).unique())
    if len(tp_rrs) != 1 or len(sl_mults) != 1 or len(horizons) != 1:
        raise ValueError(
            "Adaptive exporter expects one label config in search CSV; "
            f"got tp={tp_rrs}, sl={sl_mults}, horizon={horizons}"
        )

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
        tp_rr=float(tp_rrs[0]),
        sl_mult=float(sl_mults[0]),
        horizon_bars=int(horizons[0]),
        args=model_args,
    )
    manifest = _write_manifest(scored, choices, folds, args.out_dir, args.broker_gmt)
    choices.to_csv(args.out_dir / "adaptive_fold_choices.csv", index=False)
    print(f"features={args.features}")
    print(f"search={args.search_csv}")
    print(f"out={args.out_dir}")
    print(f"total_signals={manifest['total_signals']}")
    print(
        choices[
            [
                "fold",
                "risk_pct",
                "min_probability",
                "top_k_per_fold",
                "fold_final",
                "fold_dd",
                "fold_trades",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
