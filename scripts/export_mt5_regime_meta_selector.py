from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_mt5_feedback_rolling_selector import (  # noqa: E402
    as_float,
    as_int,
    candidate_label,
    history_score,
    is_true,
    load_candidate,
    parse_bootstrap_map,
    read_json,
    resolve_csv,
    result_row_for_fold,
)


REGIME_COLUMNS = [
    "atr",
    "atr_percentile",
    "atr_ratio",
    "range_efficiency",
    "strategy_score",
    "trend_strength_score",
    "execution_quality",
    "pullback_quality",
    "regime_trending",
    "regime_sideway",
    "regime_volatile",
    "regime_score",
    "regime_favorable",
    "daily_bias",
    "hourly_bias",
    "h4_ict_confluence",
    "wyckoff_phase",
    "volatility_regime",
    "adx",
    "price_roc",
    "tick_volume_zscore",
]


def _parse_time(value: Any) -> pd.Timestamp:
    return pd.to_datetime(str(value), errors="coerce")


def _load_feature_regimes(features_path: Path, folds: list[dict[str, Any]], lookback_days: int) -> pd.DataFrame:
    header = pd.read_csv(features_path, nrows=0).columns
    usecols = [c for c in ["time", "close", "trade_side"] + REGIME_COLUMNS if c in header]
    frame = pd.read_csv(features_path, usecols=usecols)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time")
    for col in [c for c in frame.columns if c not in {"time", "trade_side"}]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold"])
        test_start = _parse_time(fold["test_start"])
        if pd.isna(test_start):
            continue
        start = test_start - pd.Timedelta(days=int(lookback_days))
        window = frame[(frame["time"] >= start) & (frame["time"] < test_start)].copy()
        if window.empty:
            continue
        row: dict[str, Any] = {"fold": fold_id, "regime_rows": int(len(window))}
        close = pd.to_numeric(window.get("close"), errors="coerce").dropna()
        if len(close) >= 2 and float(close.iloc[0]) != 0.0:
            ret = float(close.iloc[-1] / close.iloc[0] - 1.0)
            row["regime_close_return"] = ret
            row["regime_abs_close_return"] = abs(ret)
        if "trade_side" in window.columns:
            side = window["trade_side"].astype(str).str.lower()
            row["regime_buy_ratio"] = float((side == "buy").mean())
        for col in REGIME_COLUMNS:
            if col not in window.columns:
                continue
            values = pd.to_numeric(window[col], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"regime_{col}_mean"] = float(values.mean())
            row[f"regime_{col}_std"] = float(values.std(ddof=0))
        rows.append(row)
    regimes = pd.DataFrame(rows).set_index("fold") if rows else pd.DataFrame()
    return regimes


def _normalizers(regimes: pd.DataFrame) -> tuple[pd.Series, pd.Series, list[str]]:
    cols = [c for c in regimes.columns if c != "regime_rows" and pd.api.types.is_numeric_dtype(regimes[c])]
    if not cols:
        return pd.Series(dtype=float), pd.Series(dtype=float), []
    mean = regimes[cols].mean()
    std = regimes[cols].std(ddof=0).replace(0.0, np.nan).fillna(1.0)
    return mean, std, cols


def _regime_distance(
    regimes: pd.DataFrame,
    current_fold: int,
    history_fold: int,
    mean: pd.Series,
    std: pd.Series,
    cols: list[str],
) -> float:
    if current_fold not in regimes.index or history_fold not in regimes.index or not cols:
        return 1.0
    cur = (regimes.loc[current_fold, cols] - mean[cols]) / std[cols]
    hist = (regimes.loc[history_fold, cols] - mean[cols]) / std[cols]
    diff = (cur - hist).replace([np.inf, -np.inf], np.nan).dropna()
    if diff.empty:
        return 1.0
    return float(np.sqrt(np.mean(np.square(diff.to_numpy(dtype=float)))))


def _strict(row: pd.Series, args: argparse.Namespace) -> bool:
    return (
        as_float(row.get("final_balance"), -1.0) >= float(args.target_balance)
        and as_float(row.get("max_dd_pct"), -999.0) >= float(args.min_dd_pct)
        and as_float(row.get("final_balance"), -1.0) >= float(args.deposit)
    )


def _outcome_value(row: pd.Series, args: argparse.Namespace) -> float:
    final_balance = as_float(row.get("final_balance"), float(args.deposit))
    max_dd = as_float(row.get("max_dd_pct"), -100.0)
    target = final_balance >= float(args.target_balance)
    dd_ok = max_dd >= float(args.min_dd_pct)
    loss = final_balance < float(args.deposit)
    capped_final = min(final_balance, float(args.target_balance))
    final_part = (capped_final - float(args.deposit)) / max(1.0, float(args.target_balance) - float(args.deposit))
    dd_part = max(-2.0, min(1.0, (max_dd - float(args.min_dd_pct)) / 20.0))
    value = 2.0 * final_part + 0.5 * dd_part
    if target and dd_ok and not loss:
        value += 3.0
    if target:
        value += 1.0
    if dd_ok:
        value += 0.5
    if not target:
        value -= 1.5
    if not dd_ok:
        value -= 2.0
    if loss:
        value -= 3.0
    return float(value)


def _regime_meta_score(
    history: pd.DataFrame,
    fold_id: int,
    regimes: pd.DataFrame,
    normal_mean: pd.Series,
    normal_std: pd.Series,
    regime_cols: list[str],
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    base, stats = history_score(history, args)
    if history.empty:
        return base, {**stats, "regime_effective_rows": 0.0, "regime_weighted_score": math.nan}

    values: list[float] = []
    weights: list[float] = []
    strict_values: list[float] = []
    target_values: list[float] = []
    dd_values: list[float] = []
    for _, row in history.iterrows():
        hist_fold = as_int(row.get("fold"), 0)
        if hist_fold <= 0 or hist_fold >= fold_id:
            continue
        dist = _regime_distance(regimes, fold_id, hist_fold, normal_mean, normal_std, regime_cols)
        sim = math.exp(-((dist / max(float(args.regime_bandwidth), 1.0e-6)) ** 2) / 2.0)
        age = max(0, fold_id - hist_fold)
        recency = 1.0
        if float(args.recency_half_life_folds) > 0.0:
            recency = 0.5 ** (age / float(args.recency_half_life_folds))
        weight = sim * recency
        values.append(_outcome_value(row, args))
        weights.append(weight)
        strict_values.append(1.0 if _strict(row, args) else 0.0)
        target_values.append(1.0 if as_float(row.get("final_balance"), -1.0) >= float(args.target_balance) else 0.0)
        dd_values.append(1.0 if as_float(row.get("max_dd_pct"), -999.0) >= float(args.min_dd_pct) else 0.0)

    if not values or sum(weights) <= 1.0e-9:
        return base, {**stats, "regime_effective_rows": 0.0, "regime_weighted_score": math.nan}

    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    strict_arr = np.asarray(strict_values, dtype=float)
    target_arr = np.asarray(target_values, dtype=float)
    dd_arr = np.asarray(dd_values, dtype=float)
    weighted_score = float(np.average(v, weights=w))
    weighted_strict = float(np.average(strict_arr, weights=w))
    weighted_target = float(np.average(target_arr, weights=w))
    weighted_dd = float(np.average(dd_arr, weights=w))
    effective_rows = float((w.sum() ** 2) / max(float(np.square(w).sum()), 1.0e-9))

    base_norm = float(base) / max(1.0, float(len(history)))
    regime_component = (
        weighted_score * 80_000.0
        + weighted_strict * 90_000.0
        + weighted_target * 25_000.0
        + weighted_dd * 15_000.0
        + min(effective_rows, 10.0) * 2_000.0
    )
    score = (1.0 - float(args.regime_weight)) * base_norm + float(args.regime_weight) * regime_component
    meta = {
        **stats,
        "base_score": float(base),
        "base_score_per_row": base_norm,
        "regime_effective_rows": effective_rows,
        "regime_weighted_score": weighted_score,
        "regime_weighted_strict": weighted_strict,
        "regime_weighted_target": weighted_target,
        "regime_weighted_dd": weighted_dd,
    }
    return float(score), meta


def _copy_selected_fold(
    candidate: dict[str, Any],
    fold_id: int,
    out_dir: Path,
    current_result: pd.Series,
    score: float,
    history: pd.DataFrame,
    stats: dict[str, Any],
    selection_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_fold = dict(candidate["folds"][fold_id])
    source_csv = resolve_csv(candidate["manifest_path"], str(source_fold["csv"]))
    target_csv = out_dir / f"fold_{fold_id:02d}_signals.csv"
    shutil.copy2(source_csv, target_csv)
    signals = max(0, sum(1 for _ in target_csv.open("r", encoding="utf-8")) - 1)
    risk_pct = as_float(current_result.get("risk_pct"), as_float(source_fold.get("risk_pct"), 0.0))
    max_risk_pct = as_float(current_result.get("max_risk_pct"), risk_pct)
    max_exposure_pct = as_float(current_result.get("max_exposure_pct"), max_risk_pct)
    max_positions = as_int(
        current_result.get("max_positions"),
        as_int(source_fold.get("max_positions", source_fold.get("max_positions_hint", 1)), 1),
    )
    selected = {
        "fold": fold_id,
        "train_start": source_fold.get("train_start"),
        "train_end": source_fold.get("train_end"),
        "test_start": source_fold.get("test_start"),
        "test_end": source_fold.get("test_end"),
        "signals": int(signals),
        "csv": str(target_csv),
        "risk_pct": round(risk_pct, 4),
        "max_risk_pct": round(max_risk_pct, 4),
        "max_exposure_pct": round(max_exposure_pct, 4),
        "max_positions": max_positions,
        "selection_mode": selection_mode,
        "selection_uses_current_fold_metrics": False,
        "selection_history_folds": ",".join(str(int(x)) for x in sorted(history["fold"].dropna().astype(int).tolist())) if "fold" in history else "",
        "selected_candidate_label": candidate["label"],
        "selected_candidate_dir": str(candidate["dir"]),
        "selected_candidate_live_protocol": bool(candidate["manifest_live_protocol"]),
        "selected_candidate_adaptive_per_fold": bool(candidate["manifest_adaptive_per_fold"]),
        "selected_candidate_research_oracle_fold_selection": bool(candidate["manifest_research_oracle_fold_selection"]),
        "selected_candidate_selection_uses_current_fold_metrics": bool(candidate["manifest_selection_uses_current_fold_metrics"]),
        "selected_score_from_past_mt5_regime": round(float(score), 6),
        **stats,
    }
    choice = {
        **selected,
        "source_results": str(candidate["results_path"]),
        "audit_current_final_not_used": as_float(current_result.get("final_balance")),
        "audit_current_dd_not_used": as_float(current_result.get("max_dd_pct")),
        "audit_current_trades_not_used": as_int(current_result.get("trades")),
    }
    return selected, choice


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a rolling MT5 feedback selector with pre-fold regime similarity scoring.")
    parser.add_argument("--candidate-dir", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--bootstrap-map", action="append", default=None)
    parser.add_argument("--min-history-folds", type=int, default=5)
    parser.add_argument("--lookback-folds", type=int, default=0)
    parser.add_argument("--recent-folds", type=int, default=3)
    parser.add_argument("--min-history-trades", type=int, default=1)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--min-dd-pct", type=float, default=-20.0)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--max-loaded-signal-gap", type=float, default=0.0)
    parser.add_argument("--regime-lookback-days", type=int, default=20)
    parser.add_argument("--regime-bandwidth", type=float, default=2.0)
    parser.add_argument("--regime-weight", type=float, default=0.65)
    parser.add_argument("--recency-half-life-folds", type=float, default=8.0)
    args = parser.parse_args()

    candidates = [load_candidate(path) for path in args.candidate_dir]
    bootstrap_map_paths = parse_bootstrap_map(args.bootstrap_map)
    for fold_id, candidate_dir in bootstrap_map_paths.items():
        if not any(candidate["dir"].resolve() == candidate_dir.resolve() for candidate in candidates):
            candidates.append(load_candidate(candidate_dir))

    fold_ids = sorted({fold_id for candidate in candidates for fold_id in candidate["folds"]})
    fold_templates = []
    first_candidate = candidates[0]
    for fold_id in fold_ids:
        if fold_id in first_candidate["folds"]:
            fold_templates.append(first_candidate["folds"][fold_id])
        else:
            for candidate in candidates:
                if fold_id in candidate["folds"]:
                    fold_templates.append(candidate["folds"][fold_id])
                    break
    regimes = _load_feature_regimes(args.features, fold_templates, args.regime_lookback_days)
    normal_mean, normal_std, regime_cols = _normalizers(regimes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_folds: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []

    for fold_id in fold_ids:
        best: tuple[float, dict[str, Any], pd.Series, pd.DataFrame, dict[str, Any]] | None = None
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
            score, stats = _regime_meta_score(history, fold_id, regimes, normal_mean, normal_std, regime_cols, args)
            audit = {
                "fold": fold_id,
                "candidate_label": candidate["label"],
                "candidate_dir": str(candidate["dir"]),
                "score": score,
                **stats,
                "audit_current_final": as_float(current_result.get("final_balance")),
                "audit_current_dd": as_float(current_result.get("max_dd_pct")),
                "audit_current_trades": as_int(current_result.get("trades")),
            }
            if fold_id in regimes.index:
                for col in regime_cols:
                    audit[col] = regimes.loc[fold_id, col]
            audit_rows.append(audit)
            for _, hist_row in history.iterrows():
                training_rows.append(
                    {
                        "selection_fold": fold_id,
                        "candidate_dir": str(candidate["dir"]),
                        "history_fold": as_int(hist_row.get("fold")),
                        "history_final_balance": as_float(hist_row.get("final_balance")),
                        "history_max_dd_pct": as_float(hist_row.get("max_dd_pct")),
                        "history_strict": int(_strict(hist_row, args)),
                    }
                )
            if len(history) < int(args.min_history_folds):
                continue
            if best is None or score > best[0]:
                best = (score, candidate, current_result, history, stats)

        selection_mode = "past_mt5_feedback_regime_meta"
        if best is None:
            if int(fold_id) not in bootstrap_map_paths:
                raise ValueError(f"Fold {fold_id:02d} has no eligible candidate and no bootstrap map.")
            candidate_dir = bootstrap_map_paths[int(fold_id)]
            candidate = next(candidate for candidate in candidates if candidate["dir"].resolve() == candidate_dir.resolve())
            current_result = result_row_for_fold(candidate["results"], fold_id)
            if current_result is None:
                raise ValueError(f"Bootstrap candidate lacks fold {fold_id:02d}: {candidate_dir}")
            history = candidate["results"][candidate["results"]["fold"] < fold_id].copy()
            score, stats = _regime_meta_score(history, fold_id, regimes, normal_mean, normal_std, regime_cols, args)
            best = (score, candidate, current_result, history, stats)
            selection_mode = "bootstrap_declared_map"

        score, candidate, current_result, history, stats = best
        selected, choice = _copy_selected_fold(candidate, fold_id, args.out_dir, current_result, score, history, stats, selection_mode)
        selected_folds.append(selected)
        choices.append(choice)

    manifest = {
        "all_signals": None,
        "folds": selected_folds,
        "total_signals": int(sum(fold["signals"] for fold in selected_folds)),
        "live_protocol": True,
        "mt5_feedback_regime_meta_selector": True,
        "adaptive_per_fold": False,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "candidate_dirs": [str(candidate["dir"]) for candidate in candidates],
        "selection_rule": "candidate selected from prior MT5 folds weighted by pre-fold MT5 feature regime similarity; current fold MT5 metrics are audit-only",
        "features": str(args.features),
        "regime_lookback_days": int(args.regime_lookback_days),
        "regime_bandwidth": float(args.regime_bandwidth),
        "regime_weight": float(args.regime_weight),
        "recency_half_life_folds": float(args.recency_half_life_folds),
        "min_history_folds": int(args.min_history_folds),
        "lookback_folds": int(args.lookback_folds),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(choices).to_csv(args.out_dir / "regime_meta_selector_choices.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.out_dir / "regime_meta_candidate_audit.csv", index=False)
    pd.DataFrame(training_rows).to_csv(args.out_dir / "regime_meta_training_rows.csv", index=False)
    regimes.reset_index().to_csv(args.out_dir / "fold_regime_features.csv", index=False)

    print(f"manifest={args.out_dir / 'manifest.json'}")
    print(f"choices={args.out_dir / 'regime_meta_selector_choices.csv'}")
    print(f"candidate_audit={args.out_dir / 'regime_meta_candidate_audit.csv'}")
    print(f"training_rows={args.out_dir / 'regime_meta_training_rows.csv'}")
    print(f"folds={len(selected_folds)} total_signals={manifest['total_signals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
