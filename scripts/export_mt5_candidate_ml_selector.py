from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_mt5_candidate_history_selector import (  # noqa: E402
    attach_regime_vectors,
    build_regime_vectors,
    copy_selected_fold,
    is_true,
    json_safe,
    load_results,
    parse_label_path,
    parse_regime_columns,
    resolve_path,
)


def candidate_args(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        label, path = parse_label_path(value)
        out[label] = resolve_path(path)
    return out


def fit_predict_prob(
    *,
    train: pd.DataFrame,
    choices: pd.DataFrame,
    regime_cols: list[str],
    candidate_labels: list[str],
    seed: int,
    calibration_weight: float,
    actual_weight: float,
) -> pd.Series:
    if train.empty:
        return pd.Series(0.0, index=choices.index)

    feature_cols = list(regime_cols)
    train_x = train[["candidate_label", *feature_cols]].copy()
    choices_x = choices[["candidate_label", *feature_cols]].copy()
    combined = pd.concat([train_x, choices_x], ignore_index=True)
    combined = pd.get_dummies(combined, columns=["candidate_label"], prefix="candidate")
    for label in candidate_labels:
        col = f"candidate_{label}"
        if col not in combined.columns:
            combined[col] = 0
    combined = combined.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_train = combined.iloc[: len(train)].copy()
    x_choices = combined.iloc[len(train) :].copy()

    y = train["strict_pass"].astype(bool).astype(int)
    if y.nunique() < 2:
        base = float(y.mean()) if len(y) else 0.0
        return pd.Series(base, index=choices.index)

    weights = pd.Series(float(calibration_weight), index=train.index)
    weights.loc[train["split"].eq("actual")] = float(actual_weight)
    clf = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=-1,
    )
    clf.fit(x_train, y, sample_weight=weights.to_numpy())
    proba = clf.predict_proba(x_choices)
    class_index = list(clf.classes_).index(1)
    return pd.Series(proba[:, class_index], index=choices.index)


def history_summary(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {
            "hist_rows": 0,
            "hist_strict_pass": 0,
            "hist_target_pass": 0,
            "hist_dd_pass": 0,
            "hist_loss": 0,
            "hist_median_final": math.nan,
            "hist_worst_dd": math.nan,
        }
    return {
        "hist_rows": int(len(history)),
        "hist_strict_pass": int(history["strict_pass"].sum()),
        "hist_target_pass": int(history["target_pass"].sum()),
        "hist_dd_pass": int(history["dd_pass"].sum()),
        "hist_loss": int(history["loss_fold"].sum()),
        "hist_median_final": float(pd.to_numeric(history["final_balance"], errors="coerce").median()),
        "hist_worst_dd": float(pd.to_numeric(history["max_dd_pct"], errors="coerce").min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a rolling ML candidate selector from prior MT5 feedback only.")
    parser.add_argument("--actual-candidate", action="append", required=True, help="LABEL=DIR. May be repeated.")
    parser.add_argument("--calibration-candidate", action="append", default=None, help="LABEL=DIR. May be repeated.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--regime-columns", type=str, default=None)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--min-dd-pct", type=float, default=-20.0)
    parser.add_argument("--calibration-fold-shift", type=int, default=-29)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=30)
    parser.add_argument("--history-lookback", type=int, default=0)
    parser.add_argument("--calibration-weight", type=float, default=0.75)
    parser.add_argument("--actual-weight", type=float, default=1.0)
    parser.add_argument("--min-train-rows", type=int, default=20)
    args = parser.parse_args()

    actual_dirs = candidate_args(args.actual_candidate)
    calibration_dirs = candidate_args(args.calibration_candidate or [])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    history_frames: list[pd.DataFrame] = []
    actual_frames: list[pd.DataFrame] = []
    manifests: dict[str, dict[str, Any]] = {}
    manifest_paths: dict[str, Path] = {}

    for label, source_dir in actual_dirs.items():
        frame, manifest = load_results(
            label=label,
            source_dir=source_dir,
            split="actual",
            fold_shift=0,
            deposit=args.deposit,
            target_balance=args.target_balance,
            min_dd_pct=args.min_dd_pct,
        )
        if frame.empty:
            raise ValueError(f"Actual candidate {label} has no mt5_wf_results.csv rows: {source_dir}")
        actual_frames.append(frame)
        history_frames.append(frame)
        manifests[label] = manifest
        manifest_paths[label] = source_dir / "manifest.json"

    for label, source_dir in calibration_dirs.items():
        frame, _manifest = load_results(
            label=label,
            source_dir=source_dir,
            split="calibration",
            fold_shift=int(args.calibration_fold_shift),
            deposit=args.deposit,
            target_balance=args.target_balance,
            min_dd_pct=args.min_dd_pct,
        )
        if not frame.empty:
            history_frames.append(frame)

    history = pd.concat(history_frames, ignore_index=True)
    actual = pd.concat(actual_frames, ignore_index=True)
    requested_columns = parse_regime_columns(args.regime_columns)
    combined_periods = pd.concat([history, actual], ignore_index=True)
    regime_vectors, regime_cols, _regime_scales = build_regime_vectors(
        features_csv=resolve_path(args.features_csv),
        rows=combined_periods,
        requested_columns=requested_columns,
    )
    history = attach_regime_vectors(history, regime_vectors)
    actual = attach_regime_vectors(actual, regime_vectors)

    selected_folds: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    candidate_labels = sorted(actual_dirs)

    for fold_id in range(int(args.fold_start), int(args.fold_end) + 1):
        train = history[pd.to_numeric(history["history_fold"], errors="coerce") < int(fold_id)].copy()
        if int(args.history_lookback) > 0:
            train = train[train["history_fold"] >= int(fold_id) - int(args.history_lookback)].copy()
        target_rows = actual[pd.to_numeric(actual["fold"], errors="coerce") == int(fold_id)].copy()
        if target_rows.empty:
            raise ValueError(f"No candidate has actual fold {fold_id}")
        choices = target_rows[["candidate_label", *regime_cols]].copy()
        if len(train) >= int(args.min_train_rows):
            choices["ml_prob_strict"] = fit_predict_prob(
                train=train,
                choices=choices,
                regime_cols=regime_cols,
                candidate_labels=candidate_labels,
                seed=fold_id,
                calibration_weight=args.calibration_weight,
                actual_weight=args.actual_weight,
            )
            mode = "rolling_ml_candidate_selector"
        else:
            choices["ml_prob_strict"] = 0.0
            mode = "rolling_ml_insufficient_history"

        merged = target_rows.merge(choices[["candidate_label", "ml_prob_strict"]], on="candidate_label", how="left")
        merged["history_strict_rate"] = 0.0
        merged["history_dd_rate"] = 0.0
        merged["history_median_final"] = 0.0
        per_choice_rows: list[dict[str, Any]] = []
        for idx, choice in merged.iterrows():
            label = str(choice["candidate_label"])
            candidate_history = train[train["candidate_label"].eq(label)].copy()
            summary = history_summary(candidate_history)
            strict_rate = float(summary["hist_strict_pass"]) / max(1, int(summary["hist_rows"]))
            dd_rate = float(summary["hist_dd_pass"]) / max(1, int(summary["hist_rows"]))
            merged.loc[idx, "history_strict_rate"] = strict_rate
            merged.loc[idx, "history_dd_rate"] = dd_rate
            merged.loc[idx, "history_median_final"] = float(summary["hist_median_final"] or 0.0)
            per_choice_rows.append(
                {
                    "fold": fold_id,
                    "candidate_label": label,
                    "ml_prob_strict": float(choice.get("ml_prob_strict", 0.0)),
                    **summary,
                }
            )
        score_rows.extend(per_choice_rows)
        merged = merged.sort_values(
            ["ml_prob_strict", "history_strict_rate", "history_dd_rate", "history_median_final", "candidate_label"],
            ascending=[False, False, False, False, True],
        )
        selected = merged.iloc[0].to_dict()
        label = str(selected["candidate_label"])
        selected_history = train[train["candidate_label"].eq(label)].copy()
        score = {
            "score": float(selected.get("ml_prob_strict", 0.0)),
            "ml_train_rows": int(len(train)),
            "ml_mode": mode,
            "ml_prob_strict": float(selected.get("ml_prob_strict", 0.0)),
            **history_summary(selected_history),
        }
        fold = copy_selected_fold(
            label=label,
            source_manifest=manifests[label],
            source_manifest_path=manifest_paths[label],
            source_dir=actual_dirs[label],
            fold_id=fold_id,
            out_dir=args.out_dir,
            score=score,
        )
        fold["selection_mode"] = mode
        selected_folds.append(fold)
        audit_rows.append(
            {
                "fold": fold_id,
                "selected_candidate_label": label,
                "selected_candidate_dir": str(actual_dirs[label]),
                "ml_train_rows": int(len(train)),
                "ml_prob_strict": float(selected.get("ml_prob_strict", 0.0)),
                "actual_final_balance": selected["final_balance"],
                "actual_max_dd_pct": selected["max_dd_pct"],
                "actual_trades": selected["trades"],
                "actual_target_pass": bool(selected["target_pass"]),
                "actual_dd_pass": bool(selected["dd_pass"]),
                "actual_loss_fold": bool(selected["loss_fold"]),
                "actual_strict_pass": bool(selected["strict_pass"]),
            }
        )

    manifest = {
        "all_signals": None,
        "folds": selected_folds,
        "total_signals": int(sum(int(fold["signals"]) for fold in selected_folds)),
        "live_protocol": True,
        "rolling_candidate_ml_selector": True,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "adaptive_per_fold": False,
        "actual_candidates": {label: str(path) for label, path in sorted(actual_dirs.items())},
        "calibration_candidates": {label: str(path) for label, path in sorted(calibration_dirs.items())},
        "selector_args": {
            key: json_safe(value)
            for key, value in vars(args).items()
            if key not in {"actual_candidate", "calibration_candidate", "out_dir"}
        },
        "source_manifests_live_protocol": {
            label: is_true(manifests[label].get("live_protocol")) for label in sorted(manifests)
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(audit_rows).to_csv(args.out_dir / "selection_audit.csv", index=False)
    pd.DataFrame(score_rows).to_csv(args.out_dir / "candidate_scores_by_fold.csv", index=False)
    history.to_csv(args.out_dir / "candidate_history_matrix.csv", index=False)

    audit = pd.DataFrame(audit_rows)
    summary = {
        "manifest": str(args.out_dir / "manifest.json"),
        "folds": int(len(audit)),
        "signals": int(sum(int(fold["signals"]) for fold in selected_folds)),
        "strict_pass": int(audit["actual_strict_pass"].sum()),
        "target_pass": int(audit["actual_target_pass"].sum()),
        "dd_pass": int(audit["actual_dd_pass"].sum()),
        "loss_folds": int(audit["actual_loss_fold"].sum()),
        "min_final": float(pd.to_numeric(audit["actual_final_balance"], errors="coerce").min()),
        "median_final": float(pd.to_numeric(audit["actual_final_balance"], errors="coerce").median()),
        "worst_dd": float(pd.to_numeric(audit["actual_max_dd_pct"], errors="coerce").min()),
        "total_trades": int(pd.to_numeric(audit["actual_trades"], errors="coerce").fillna(0).sum()),
        "fail_folds": audit.loc[~audit["actual_strict_pass"], "fold"].astype(int).tolist(),
    }
    (args.out_dir / "selector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
