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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_mt5_feedback_rolling_selector import (  # noqa: E402
    as_float,
    as_int,
    history_score,
    load_candidate,
    parse_bootstrap_map,
    resolve_csv,
    result_row_for_fold,
)
from scripts.export_mt5_regime_meta_selector import REGIME_COLUMNS, _load_feature_regimes  # noqa: E402


def _strict(row: pd.Series, args: argparse.Namespace) -> bool:
    return (
        as_float(row.get("final_balance"), -1.0) >= float(args.target_balance)
        and as_float(row.get("max_dd_pct"), -999.0) >= float(args.min_dd_pct)
        and as_float(row.get("final_balance"), -1.0) >= float(args.deposit)
    )


def _outcome(row: pd.Series, args: argparse.Namespace) -> float:
    final_balance = as_float(row.get("final_balance"), float(args.deposit))
    max_dd = as_float(row.get("max_dd_pct"), -100.0)
    final_score = (min(final_balance, float(args.target_balance)) - float(args.deposit)) / max(
        1.0, float(args.target_balance) - float(args.deposit)
    )
    dd_score = max(-2.0, min(1.0, (max_dd - float(args.min_dd_pct)) / 20.0))
    value = 2.0 * final_score + 0.5 * dd_score
    if final_balance >= float(args.target_balance):
        value += 2.0
    else:
        value -= 1.0
    if max_dd >= float(args.min_dd_pct):
        value += 1.0
    else:
        value -= 2.0
    if final_balance < float(args.deposit):
        value -= 3.0
    return float(value)


def _history_stats(results: pd.DataFrame, fold_id: int, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_fold = 1 if int(args.lookback_folds) <= 0 else max(1, fold_id - int(args.lookback_folds))
    history = results[(results["fold"] >= start_fold) & (results["fold"] < fold_id)].copy()
    _, stats = history_score(history, args)
    if history.empty:
        stats.update(
            {
                "hist_strict_rate": 0.0,
                "hist_target_rate": 0.0,
                "hist_dd_rate": 0.0,
                "hist_loss_rate": 0.0,
                "hist_final_trend": 0.0,
            }
        )
        return history, stats
    strict = history.apply(lambda row: _strict(row, args), axis=1).astype(float)
    target = (pd.to_numeric(history["final_balance"], errors="coerce") >= float(args.target_balance)).astype(float)
    dd_ok = (pd.to_numeric(history["max_dd_pct"], errors="coerce") >= float(args.min_dd_pct)).astype(float)
    loss = (pd.to_numeric(history["final_balance"], errors="coerce") < float(args.deposit)).astype(float)
    finals = pd.to_numeric(history["final_balance"], errors="coerce").fillna(float(args.deposit))
    trend = 0.0
    if len(finals) >= 2:
        x = np.arange(len(finals), dtype=float)
        trend = float(np.polyfit(x, finals.to_numpy(dtype=float), deg=1)[0] / max(1.0, float(args.target_balance)))
    stats.update(
        {
            "hist_strict_rate": float(strict.mean()),
            "hist_target_rate": float(target.mean()),
            "hist_dd_rate": float(dd_ok.mean()),
            "hist_loss_rate": float(loss.mean()),
            "hist_final_trend": trend,
        }
    )
    return history, stats


def _candidate_family(path: Path) -> str:
    text = str(path).replace("/", "\\")
    for token in ["feedback_model", "profitr_pluscombo", "prior_mt5_feedback", "fixed_library", "rule_f07", "rule_f10", "rule_f12", "rule_f15", "rule_f21", "rule_f26", "rule_f05", "rule_f04"]:
        if token in text:
            return token
    return path.name


def _feature_row(
    candidate: dict[str, Any],
    fold_id: int,
    regimes: pd.DataFrame,
    args: argparse.Namespace,
    include_label: bool,
) -> dict[str, Any] | None:
    if fold_id not in candidate["folds"]:
        return None
    current_result = result_row_for_fold(candidate["results"], fold_id)
    if current_result is None:
        return None
    source_fold = candidate["folds"][fold_id]
    history, stats = _history_stats(candidate["results"], fold_id, args)
    risk_pct = as_float(current_result.get("risk_pct"), as_float(source_fold.get("risk_pct"), 0.0))
    max_positions = as_int(current_result.get("max_positions"), as_int(source_fold.get("max_positions", source_fold.get("max_positions_hint", 1)), 1))
    signals = as_int(source_fold.get("signals"), as_int(current_result.get("signals"), 0))
    row: dict[str, Any] = {
        "fold": int(fold_id),
        "candidate_dir": str(candidate["dir"]),
        "candidate_family": _candidate_family(Path(candidate["dir"])),
        "signals": signals,
        "risk_pct": risk_pct,
        "max_positions": max_positions,
        "max_exposure_pct": risk_pct * max_positions,
        "signal_risk_density": signals * risk_pct,
        **stats,
    }
    if fold_id in regimes.index:
        for col, value in regimes.loc[fold_id].items():
            row[col] = value
    if include_label:
        row["label_strict"] = int(_strict(current_result, args))
        row["label_target"] = int(as_float(current_result.get("final_balance"), -1.0) >= float(args.target_balance))
        row["label_dd"] = int(as_float(current_result.get("max_dd_pct"), -999.0) >= float(args.min_dd_pct))
        row["label_outcome"] = _outcome(current_result, args)
        row["label_final_balance"] = as_float(current_result.get("final_balance"))
        row["label_max_dd_pct"] = as_float(current_result.get("max_dd_pct"))
        row["label_trades"] = as_int(current_result.get("trades"))
    return row


def _make_models(seed: int) -> tuple[Pipeline, Pipeline]:
    clf = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    reg = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=seed + 10_000,
        n_jobs=-1,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", clf)]), Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("model", reg)]
    )


def _score_current(train: pd.DataFrame, current: pd.DataFrame, fold_id: int, args: argparse.Namespace) -> pd.DataFrame:
    out = current.copy()
    if train.empty or train["label_strict"].nunique() < 2:
        out["ml_p_strict"] = 0.0
        out["ml_expected_outcome"] = 0.0
        out["ml_score"] = out["hist_strict_rate"] * 100_000.0 + out["hist_target_rate"] * 20_000.0
        return out
    y = train["label_strict"].astype(int)
    y_outcome = pd.to_numeric(train["label_outcome"], errors="coerce").fillna(0.0)
    drop_cols = {
        "label_strict",
        "label_target",
        "label_dd",
        "label_outcome",
        "label_final_balance",
        "label_max_dd_pct",
        "label_trades",
        "candidate_dir",
    }
    combined = pd.concat([train.drop(columns=[c for c in drop_cols if c in train], errors="ignore"), current.drop(columns=[c for c in drop_cols if c in current], errors="ignore")], ignore_index=True)
    combined = pd.get_dummies(combined, columns=["candidate_family"], dummy_na=True)
    x_train = combined.iloc[: len(train)].copy()
    x_current = combined.iloc[len(train) :].copy()
    for col in x_train.columns:
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce")
        x_current[col] = pd.to_numeric(x_current[col], errors="coerce")
    clf, reg = _make_models(seed=fold_id)
    clf.fit(x_train, y)
    reg.fit(x_train, y_outcome)
    out["ml_p_strict"] = clf.predict_proba(x_current)[:, 1]
    out["ml_expected_outcome"] = reg.predict(x_current)
    out["ml_score"] = (
        out["ml_p_strict"] * 140_000.0
        + out["ml_expected_outcome"] * 35_000.0
        + out["hist_strict_rate"] * 35_000.0
        + out["hist_target_rate"] * 12_000.0
        + out["hist_dd_rate"] * 8_000.0
        - out["hist_loss_rate"] * 80_000.0
        + np.minimum(pd.to_numeric(out["signal_risk_density"], errors="coerce").fillna(0.0), 2_200.0) * float(args.density_weight)
    )
    return out


def _copy_fold(candidate: dict[str, Any], fold_id: int, out_dir: Path, score_row: pd.Series, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    source_fold = dict(candidate["folds"][fold_id])
    current_result = result_row_for_fold(candidate["results"], fold_id)
    if current_result is None:
        raise ValueError(f"Missing current result for fold {fold_id} in {candidate['dir']}")
    source_csv = resolve_csv(candidate["manifest_path"], str(source_fold["csv"]))
    target_csv = out_dir / f"fold_{fold_id:02d}_signals.csv"
    shutil.copy2(source_csv, target_csv)
    signals = max(0, sum(1 for _ in target_csv.open("r", encoding="utf-8")) - 1)
    risk_pct = as_float(current_result.get("risk_pct"), as_float(source_fold.get("risk_pct"), 0.0))
    max_risk_pct = as_float(current_result.get("max_risk_pct"), risk_pct)
    max_exposure_pct = as_float(current_result.get("max_exposure_pct"), max_risk_pct)
    max_positions = as_int(current_result.get("max_positions"), as_int(source_fold.get("max_positions", source_fold.get("max_positions_hint", 1)), 1))
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
        "selection_mode": str(score_row.get("selection_mode", "past_mt5_feedback_ml_meta")),
        "selection_uses_current_fold_metrics": False,
        "selected_candidate_dir": str(candidate["dir"]),
        "selected_candidate_family": _candidate_family(Path(candidate["dir"])),
        "selected_ml_score_from_past": as_float(score_row.get("ml_score")),
        "selected_ml_p_strict_from_past": as_float(score_row.get("ml_p_strict")),
        "selected_ml_expected_outcome_from_past": as_float(score_row.get("ml_expected_outcome")),
        "hist_rows": as_int(score_row.get("hist_rows")),
        "hist_strict_rate": as_float(score_row.get("hist_strict_rate")),
        "hist_target_rate": as_float(score_row.get("hist_target_rate")),
        "hist_dd_rate": as_float(score_row.get("hist_dd_rate")),
        "hist_loss_rate": as_float(score_row.get("hist_loss_rate")),
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
    parser = argparse.ArgumentParser(description="Export rolling MT5 selector trained as a fold-by-fold ML meta-model.")
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
    parser.add_argument("--density-weight", type=float, default=12.0)
    args = parser.parse_args()

    candidates = [load_candidate(path) for path in args.candidate_dir]
    bootstrap_map_paths = parse_bootstrap_map(args.bootstrap_map)
    for _, candidate_dir in bootstrap_map_paths.items():
        if not any(candidate["dir"].resolve() == candidate_dir.resolve() for candidate in candidates):
            candidates.append(load_candidate(candidate_dir))
    fold_ids = sorted({fold_id for candidate in candidates for fold_id in candidate["folds"]})
    fold_templates = []
    for fold_id in fold_ids:
        for candidate in candidates:
            if fold_id in candidate["folds"]:
                fold_templates.append(candidate["folds"][fold_id])
                break
    regimes = _load_feature_regimes(args.features, fold_templates, args.regime_lookback_days)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_folds: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    audit_parts: list[pd.DataFrame] = []
    train_parts: list[pd.DataFrame] = []

    for fold_id in fold_ids:
        current_rows = []
        train_rows = []
        for candidate in candidates:
            row = _feature_row(candidate, fold_id, regimes, args, include_label=True)
            if row is not None:
                current_rows.append(row)
            for hist_fold in fold_ids:
                if hist_fold >= fold_id:
                    continue
                hist_row = _feature_row(candidate, hist_fold, regimes, args, include_label=True)
                if hist_row is not None and as_int(hist_row.get("hist_rows")) >= int(args.min_history_folds):
                    train_rows.append(hist_row)
        current = pd.DataFrame(current_rows)
        train = pd.DataFrame(train_rows)
        if current.empty:
            continue
        if fold_id in bootstrap_map_paths:
            candidate_dir = bootstrap_map_paths[fold_id]
            current["selection_mode"] = "bootstrap_declared_map"
            scored = current.copy()
            scored["ml_p_strict"] = np.nan
            scored["ml_expected_outcome"] = np.nan
            scored["ml_score"] = -1.0e18
            chosen = scored[scored["candidate_dir"].apply(lambda value: Path(value).resolve() == candidate_dir.resolve())].iloc[0]
        else:
            eligible = current[current["hist_rows"].astype(float) >= int(args.min_history_folds)].copy()
            if eligible.empty:
                raise ValueError(f"Fold {fold_id:02d} has no eligible candidate and no bootstrap map.")
            scored = _score_current(train, eligible, fold_id, args)
            scored["selection_mode"] = "past_mt5_feedback_ml_meta"
            chosen = scored.sort_values(["ml_score", "ml_p_strict", "hist_strict_rate"], ascending=False).iloc[0]
        chosen_dir = Path(str(chosen["candidate_dir"]))
        candidate = next(candidate for candidate in candidates if candidate["dir"].resolve() == chosen_dir.resolve())
        selected, choice = _copy_fold(candidate, fold_id, args.out_dir, chosen, args)
        selected_folds.append(selected)
        choices.append(choice)
        audit_parts.append(scored.assign(selection_fold=fold_id))
        if not train.empty:
            train_parts.append(train.assign(selection_fold=fold_id))

    manifest = {
        "all_signals": None,
        "folds": selected_folds,
        "total_signals": int(sum(fold["signals"] for fold in selected_folds)),
        "live_protocol": True,
        "mt5_feedback_ml_meta_selector": True,
        "adaptive_per_fold": False,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "selection_rule": "ML meta-model trained only on candidate MT5 folds before the selected fold; current fold metrics are audit-only",
        "features": str(args.features),
        "candidate_dirs": [str(candidate["dir"]) for candidate in candidates],
        "min_history_folds": int(args.min_history_folds),
        "lookback_folds": int(args.lookback_folds),
        "regime_lookback_days": int(args.regime_lookback_days),
        "density_weight": float(args.density_weight),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(choices).to_csv(args.out_dir / "ml_meta_selector_choices.csv", index=False)
    pd.concat(audit_parts, ignore_index=True).to_csv(args.out_dir / "ml_meta_candidate_audit.csv", index=False)
    if train_parts:
        pd.concat(train_parts, ignore_index=True).to_csv(args.out_dir / "ml_meta_training_rows.csv", index=False)
    regimes.reset_index().to_csv(args.out_dir / "fold_regime_features.csv", index=False)
    print(f"manifest={args.out_dir / 'manifest.json'}")
    print(f"choices={args.out_dir / 'ml_meta_selector_choices.csv'}")
    print(f"candidate_audit={args.out_dir / 'ml_meta_candidate_audit.csv'}")
    print(f"folds={len(selected_folds)} total_signals={manifest['total_signals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
