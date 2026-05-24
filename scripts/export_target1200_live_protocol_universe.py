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
    load_feature_frame,
    load_folds,
    parse_hours,
)


MT5_COLUMNS = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]
LABEL_COLUMNS = ["tp_rr", "sl_mult", "horizon_bars"]


def _fold_metric_columns(search: pd.DataFrame, folds: list[dict[str, Any]]) -> None:
    missing: list[str] = []
    for fold in folds:
        fold_id = int(fold["fold"])
        for suffix in ["final", "dd", "trades"]:
            col = f"fold_{fold_id:02d}_{suffix}"
            if col not in search.columns:
                missing.append(col)
    if missing:
        raise ValueError(f"Search CSV lacks per-fold metric columns: {missing[:8]}")


def _history_fold_ids(fold_id: int, lookback_folds: int) -> list[int]:
    start = 1 if lookback_folds <= 0 else max(1, fold_id - lookback_folds)
    return list(range(start, fold_id))


def _score_history(
    search: pd.DataFrame,
    history_ids: list[int],
    target_balance: float,
    min_dd_pct: float,
    deposit: float,
    min_trades: int,
) -> pd.DataFrame:
    scored = search.copy()
    finals = []
    dds = []
    trades = []
    for fold_id in history_ids:
        finals.append(scored[f"fold_{fold_id:02d}_final"].astype(float).to_numpy())
        dds.append(scored[f"fold_{fold_id:02d}_dd"].astype(float).to_numpy())
        trades.append(scored[f"fold_{fold_id:02d}_trades"].astype(float).to_numpy())
    final_arr = np.vstack(finals).T
    dd_arr = np.vstack(dds).T
    trade_arr = np.vstack(trades).T
    active = trade_arr >= float(min_trades)
    passed = (final_arr >= target_balance) & (dd_arr >= min_dd_pct) & active
    scored["hist_pass_folds"] = passed.sum(axis=1)
    scored["hist_target_fails"] = ((final_arr < target_balance) | ~active).sum(axis=1)
    scored["hist_dd_fails"] = (dd_arr < min_dd_pct).sum(axis=1)
    scored["hist_loss_folds"] = (final_arr < deposit).sum(axis=1)
    scored["hist_min_final"] = np.nanmin(final_arr, axis=1)
    scored["hist_median_final"] = np.nanmedian(final_arr, axis=1)
    scored["hist_mean_final"] = np.nanmean(final_arr, axis=1)
    scored["hist_worst_dd"] = np.nanmin(dd_arr, axis=1)
    scored["hist_min_trades"] = np.nanmin(trade_arr, axis=1)
    scored["hist_fold_count"] = len(history_ids)
    return scored


def _choose_live_rows(
    search: pd.DataFrame,
    folds: list[dict[str, Any]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    _fold_metric_columns(search, folds)
    search = search.copy()
    if args.max_risk_fraction is not None:
        search = search[search["risk_pct"].astype(float) <= float(args.max_risk_fraction)].copy()
    if args.max_positions is not None:
        search = search[search["max_positions"].astype(int) <= int(args.max_positions)].copy()
    if search.empty:
        raise ValueError("Candidate filters removed every search row.")

    selections: list[pd.Series] = []
    bootstrap_row: pd.Series | None = None
    if args.bootstrap_search_row is not None:
        if args.bootstrap_search_row < 0 or args.bootstrap_search_row >= len(search):
            raise ValueError("--bootstrap-search-row is out of range after filtering.")
        bootstrap_row = search.iloc[int(args.bootstrap_search_row)].copy()

    for fold in folds:
        fold_id = int(fold["fold"])
        history_ids = _history_fold_ids(fold_id, int(args.lookback_folds))
        if len(history_ids) < int(args.min_history_folds):
            if bootstrap_row is None:
                row = pd.Series(
                    {
                        "fold": fold_id,
                        "selected": False,
                        "selection_mode": "warmup_no_trade",
                        "selection_history_folds": ",".join(str(x) for x in history_ids),
                        "selection_uses_current_fold_metrics": False,
                    }
                )
                selections.append(row)
                continue
            row = bootstrap_row.copy()
            row["selection_mode"] = "bootstrap_declared"
        else:
            scored = _score_history(
                search=search,
                history_ids=history_ids,
                target_balance=float(args.target_balance),
                min_dd_pct=float(args.min_dd_pct),
                deposit=float(args.deposit),
                min_trades=int(args.min_history_trades),
            )
            if args.min_history_pass_rate > 0:
                min_pass = int(np.ceil(len(history_ids) * float(args.min_history_pass_rate)))
                filtered = scored[scored["hist_pass_folds"].astype(int) >= min_pass].copy()
                if not filtered.empty:
                    scored = filtered
            risk_ascending = str(getattr(args, "risk_sort", "low")) == "low"
            top_k_ascending = str(getattr(args, "top_k_sort", "low")) == "low"
            scored = scored.sort_values(
                [
                    "hist_pass_folds",
                    "hist_loss_folds",
                    "hist_dd_fails",
                    "hist_target_fails",
                    "hist_worst_dd",
                    "hist_min_final",
                    "hist_median_final",
                    "risk_pct",
                    "top_k_per_fold",
                ],
                ascending=[False, True, True, True, False, False, False, risk_ascending, top_k_ascending],
            )
            row = scored.iloc[0].copy()
            row["selection_mode"] = "past_oos_folds"
        row["fold"] = fold_id
        row["selected"] = True
        row["selection_history_folds"] = ",".join(str(x) for x in history_ids)
        row["selection_uses_current_fold_metrics"] = False
        for suffix in ["final", "dd", "trades"]:
            col = f"fold_{fold_id:02d}_{suffix}"
            row[f"current_fold_proxy_{suffix}"] = float(row[col]) if col in row.index else np.nan
        selections.append(row)
    return pd.DataFrame(selections)


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
    return scored


def _label_key(row: pd.Series) -> tuple[float, float, int]:
    return (float(row["tp_rr"]), float(row["sl_mult"]), int(row["horizon_bars"]))


def _write_manifest(
    scored_by_key: dict[tuple[float, float, int], pd.DataFrame],
    selections: pd.DataFrame,
    folds: list[dict[str, Any]],
    out_dir: Path,
    broker_gmt: int,
    manifest_max_positions: int | None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_exports: list[pd.DataFrame] = []
    manifest: dict[str, Any] = {
        "all_signals": str(out_dir / "all_signals.csv"),
        "folds": [],
        "total_signals": 0,
        "live_protocol": True,
        "selection_uses_current_fold_metrics": False,
        "notes": "Each fold is selected only from prior OOS fold metrics, or explicit bootstrap/warmup.",
    }

    for fold in folds:
        fold_id = int(fold["fold"])
        choice = selections[selections["fold"].astype(int) == fold_id].iloc[0]
        fold_path = out_dir / f"fold_{fold_id:02d}_signals.csv"
        export = pd.DataFrame(columns=MT5_COLUMNS)

        if bool(choice.get("selected", False)):
            key = _label_key(choice)
            scored = scored_by_key[key]
            folds_array = scored["fold"].to_numpy(np.int64)
            selected = build_selected_indices(
                folds_array=folds_array,
                proba=scored["probability"].to_numpy(np.float64),
                start_idx=scored["start_idx"].to_numpy(np.int64),
                min_probability=float(choice["min_probability"]),
                top_k=int(choice["top_k_per_fold"]),
                fold_count=len(folds),
                min_signal_gap_bars=int(choice.get("min_signal_gap_bars", 0) or 0),
            )
            selected = selected[folds_array[selected] == fold_id]
            export_full = scored.iloc[selected].sort_values(["fold", "time"]).copy()
            if not export_full.empty:
                signal_time = export_full["feature_time"] if "feature_time" in export_full.columns else export_full["time"]
                export_full["open_time"] = (
                    pd.to_datetime(signal_time, utc=True) + pd.Timedelta(hours=broker_gmt)
                ).dt.strftime("%Y.%m.%d %H:%M")
                export = export_full[MT5_COLUMNS]
                all_exports.append(export)

        export.to_csv(fold_path, index=False, float_format="%.5f")
        max_positions = int(choice.get("max_positions", 1)) if bool(choice.get("selected", False)) else 1
        if manifest_max_positions is not None and bool(choice.get("selected", False)):
            max_positions = int(manifest_max_positions)
        risk_percent = round(float(choice.get("risk_pct", 0.0)) * 100.0, 4) if bool(choice.get("selected", False)) else 0.0
        row: dict[str, Any] = {
            "fold": fold_id,
            "train_start": fold.get("train_start"),
            "train_end": fold.get("train_end"),
            "test_start": fold.get("test_start"),
            "test_end": fold.get("test_end"),
            "signals": int(len(export)),
            "csv": str(fold_path),
            "risk_pct": risk_percent,
            "max_risk_pct": risk_percent,
            "max_exposure_pct": round(risk_percent * max_positions, 4),
            "max_positions": max_positions,
            "selection_mode": str(choice.get("selection_mode", "")),
            "selection_history_folds": str(choice.get("selection_history_folds", "")),
            "selection_uses_current_fold_metrics": False,
        }
        if bool(choice.get("selected", False)):
            row.update(
                {
                    "selected_tp_rr": float(choice["tp_rr"]),
                    "selected_sl_mult": float(choice["sl_mult"]),
                    "selected_horizon_bars": int(choice["horizon_bars"]),
                    "selected_min_probability": float(choice["min_probability"]),
                    "selected_top_k_per_fold": int(choice["top_k_per_fold"]),
                    "proxy_final_balance_for_audit_only": float(choice.get("current_fold_proxy_final", np.nan)),
                    "proxy_max_dd_pct_for_audit_only": float(choice.get("current_fold_proxy_dd", np.nan)),
                    "proxy_trades_for_audit_only": float(choice.get("current_fold_proxy_trades", np.nan)),
                }
            )
            for col in [
                "hist_pass_folds",
                "hist_loss_folds",
                "hist_dd_fails",
                "hist_target_fails",
                "hist_min_final",
                "hist_worst_dd",
                "hist_fold_count",
            ]:
                if col in choice.index and pd.notna(choice[col]):
                    value = choice[col]
                    row[col] = int(value) if col.endswith("folds") or col.endswith("fails") or col == "hist_fold_count" else float(value)
        manifest["folds"].append(row)

    if all_exports:
        pd.concat(all_exports, ignore_index=True).to_csv(out_dir / "all_signals.csv", index=False, float_format="%.5f")
    else:
        pd.DataFrame(columns=MT5_COLUMNS).to_csv(out_dir / "all_signals.csv", index=False)
    manifest["total_signals"] = int(sum(int(row["signals"]) for row in manifest["folds"]))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a no-lookahead live-protocol signal universe.")
    parser.add_argument("--features", type=Path, default=ROOT / "outputs/mt5_full_ict_wyckoff_features_202306_202603.csv")
    parser.add_argument("--fold-manifest", type=Path, default=ROOT / "outputs/mt5_wf_r2_5_tradeside/manifest.json")
    parser.add_argument("--search-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--broker-gmt", type=int, default=0)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--min-dd-pct", type=float, default=-20.0)
    parser.add_argument("--min-history-folds", type=int, default=3)
    parser.add_argument("--lookback-folds", type=int, default=6)
    parser.add_argument("--min-history-pass-rate", type=float, default=0.0)
    parser.add_argument("--min-history-trades", type=int, default=1)
    parser.add_argument("--max-risk-fraction", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--manifest-max-positions", type=int, default=None)
    parser.add_argument("--risk-sort", choices=["low", "high"], default="low")
    parser.add_argument("--top-k-sort", choices=["low", "high"], default="low")
    parser.add_argument("--bootstrap-search-row", type=int, default=None)
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
    selections = _choose_live_rows(search, folds, args)
    if "selected" in selections.columns:
        selected_mask = selections["selected"].fillna(False).astype(bool)
    else:
        selected_mask = pd.Series(False, index=selections.index)
    selected_rows = selections[selected_mask].copy()

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
    scored_by_key: dict[tuple[float, float, int], pd.DataFrame] = {}
    for key in sorted({_label_key(row) for _, row in selected_rows.iterrows()}):
        print(f"building scored frame for tp={key[0]} sl={key[1]} horizon={key[2]}")
        scored_by_key[key] = _build_scored_frame(
            frame=frame,
            folds=folds,
            feature_cols=feature_cols,
            tp_rr=key[0],
            sl_mult=key[1],
            horizon_bars=key[2],
            args=model_args,
        )

    manifest = _write_manifest(
        scored_by_key,
        selections,
        folds,
        args.out_dir,
        args.broker_gmt,
        args.manifest_max_positions,
    )
    selections.to_csv(args.out_dir / "live_protocol_choices.csv", index=False)
    report = {
        "features": str(args.features),
        "search_csv": str(args.search_csv),
        "out_dir": str(args.out_dir),
        "folds": len(folds),
        "selected_folds": int(selected_rows.shape[0]),
        "warmup_no_trade_folds": int((selections["selection_mode"] == "warmup_no_trade").sum()),
        "total_signals": manifest["total_signals"],
        "no_future_leakage_by_construction": True,
        "min_history_folds": int(args.min_history_folds),
        "lookback_folds": int(args.lookback_folds),
    }
    (args.out_dir / "live_protocol_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
