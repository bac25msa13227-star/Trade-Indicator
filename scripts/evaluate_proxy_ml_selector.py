from __future__ import annotations

import argparse
import json
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

from scripts.evaluate_proxy_candidate_rolling_selector import load_candidates  # noqa: E402
from scripts.evaluate_proxy_regime_selector import fast_load_feature_regimes, load_folds, outcome_score  # noqa: E402


def history_features(group: pd.DataFrame, fold_id: int, lookback: int, target: float, deposit: float, max_dd: float) -> dict[str, float]:
    start = 1 if lookback <= 0 else max(1, fold_id - lookback)
    hist = group[(group["fold"] >= start) & (group["fold"] < fold_id)].copy()
    if hist.empty:
        return {
            "hist_rows": 0.0,
            "hist_strict_rate": 0.0,
            "hist_target_rate": 0.0,
            "hist_dd_rate": 0.0,
            "hist_loss_rate": 0.0,
            "hist_median_final": deposit,
            "hist_min_final": deposit,
            "hist_worst_dd": -max_dd,
            "hist_recent_strict_rate": 0.0,
            "hist_recent_loss_rate": 0.0,
            "hist_final_trend": 0.0,
        }
    strict = (hist["final_balance"] >= target) & (hist["max_dd_pct"] >= -max_dd) & (hist["final_balance"] >= deposit)
    target_hit = hist["final_balance"] >= target
    dd_pass = hist["max_dd_pct"] >= -max_dd
    loss = hist["final_balance"] < deposit
    recent = hist.sort_values("fold").tail(5)
    recent_strict = (recent["final_balance"] >= target) & (recent["max_dd_pct"] >= -max_dd) & (recent["final_balance"] >= deposit)
    recent_loss = recent["final_balance"] < deposit
    finals = pd.to_numeric(hist["final_balance"], errors="coerce").fillna(deposit).to_numpy(dtype=float)
    trend = 0.0
    if len(finals) >= 2:
        x = np.arange(len(finals), dtype=float)
        trend = float(np.polyfit(x, finals, deg=1)[0] / max(1.0, target))
    return {
        "hist_rows": float(len(hist)),
        "hist_strict_rate": float(strict.mean()),
        "hist_target_rate": float(target_hit.mean()),
        "hist_dd_rate": float(dd_pass.mean()),
        "hist_loss_rate": float(loss.mean()),
        "hist_median_final": float(hist["final_balance"].median()),
        "hist_min_final": float(hist["final_balance"].min()),
        "hist_worst_dd": float(hist["max_dd_pct"].min()),
        "hist_recent_strict_rate": float(recent_strict.mean()),
        "hist_recent_loss_rate": float(recent_loss.mean()),
        "hist_final_trend": trend,
    }


def build_rows(frame: pd.DataFrame, regimes: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = {label: group.sort_values("fold").copy() for label, group in frame.groupby("candidate_label")}
    for _, row in frame.sort_values(["fold", "candidate_label"]).iterrows():
        fold_id = int(row["fold"])
        label = str(row["candidate_label"])
        final_balance = float(row["final_balance"])
        dd = float(row["max_dd_pct"])
        target_hit = final_balance >= float(args.target_balance)
        dd_pass = dd >= -float(args.max_dd_pct)
        loss = final_balance < float(args.deposit)
        strict = target_hit and dd_pass and not loss
        rec: dict[str, Any] = {
            "fold": fold_id,
            "candidate_label": label,
            "candidate_dir": row.get("candidate_dir"),
            "candidate_target_folds": float(row.get("candidate_target_folds", 0.0)),
            "candidate_loss_folds": float(row.get("candidate_loss_folds", 0.0)),
            "candidate_dd_fail_folds": float(row.get("candidate_dd_fail_folds", 0.0)),
            "candidate_median_final": float(row.get("candidate_median_final", 0.0)),
            **history_features(grouped[label], fold_id, int(args.lookback_folds), float(args.target_balance), float(args.deposit), float(args.max_dd_pct)),
            "label_strict": int(strict),
            "label_target": int(target_hit),
            "label_dd": int(dd_pass),
            "label_loss": int(loss),
            "label_outcome": outcome_score(row, float(args.target_balance), float(args.deposit), float(args.max_dd_pct)),
            "final_balance": final_balance,
            "max_dd_pct": dd,
        }
        if fold_id in regimes.index:
            for col, value in regimes.loc[fold_id].items():
                rec[col] = value
        rows.append(rec)
    return pd.DataFrame(rows)


def make_models(seed: int) -> tuple[Pipeline, Pipeline]:
    clf = ExtraTreesClassifier(
        n_estimators=350,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    reg = ExtraTreesRegressor(
        n_estimators=350,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed + 10000,
        n_jobs=-1,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", clf)]), Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("model", reg)]
    )


def feature_matrix(train: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    drop = {"label_strict", "label_target", "label_dd", "label_loss", "label_outcome", "final_balance", "max_dd_pct", "candidate_dir"}
    combined = pd.concat(
        [
            train.drop(columns=[c for c in drop if c in train], errors="ignore"),
            current.drop(columns=[c for c in drop if c in current], errors="ignore"),
        ],
        ignore_index=True,
    )
    combined = pd.get_dummies(combined, columns=["candidate_label"], dummy_na=True)
    for col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined.iloc[: len(train)].copy(), combined.iloc[len(train) :].copy()


def choose_bootstrap(current: pd.DataFrame) -> pd.Series:
    return current.sort_values(
        ["candidate_target_folds", "candidate_loss_folds", "candidate_dd_fail_folds", "candidate_median_final"],
        ascending=[False, True, True, False],
    ).iloc[0]


def evaluate(rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    for fold_id in sorted(rows["fold"].dropna().astype(int).unique()):
        current = rows[rows["fold"] == fold_id].copy()
        start = 1 if int(args.train_lookback_folds) <= 0 else max(1, fold_id - int(args.train_lookback_folds))
        train = rows[(rows["fold"] >= start) & (rows["fold"] < fold_id)].copy()
        train = train[train["hist_rows"] >= float(args.min_candidate_history_rows)].copy()
        mode = "ml_prior"
        if len(train) < int(args.min_train_rows) or train["label_strict"].nunique() < 2:
            chosen = choose_bootstrap(current)
            chosen = chosen.copy()
            chosen["ml_p_strict"] = np.nan
            chosen["ml_expected_outcome"] = np.nan
            chosen["ml_score"] = np.nan
            mode = "bootstrap_global_proxy"
        else:
            x_train, x_current = feature_matrix(train, current)
            clf, reg = make_models(fold_id)
            clf.fit(x_train, train["label_strict"].astype(int))
            reg.fit(x_train, pd.to_numeric(train["label_outcome"], errors="coerce").fillna(0.0))
            scored = current.copy()
            scored["ml_p_strict"] = clf.predict_proba(x_current)[:, 1]
            scored["ml_expected_outcome"] = reg.predict(x_current)
            scored["ml_score"] = (
                scored["ml_p_strict"] * float(args.strict_weight)
                + scored["ml_expected_outcome"] * float(args.outcome_weight)
                + scored["hist_strict_rate"] * float(args.hist_strict_weight)
                + scored["hist_target_rate"] * float(args.hist_target_weight)
                + scored["hist_dd_rate"] * float(args.hist_dd_weight)
                - scored["hist_loss_rate"] * float(args.hist_loss_weight)
                + scored["hist_recent_strict_rate"] * float(args.recent_strict_weight)
                - scored["hist_recent_loss_rate"] * float(args.recent_loss_weight)
            )
            chosen = scored.sort_values(["ml_score", "ml_p_strict", "ml_expected_outcome"], ascending=[False, False, False]).iloc[0]
        rec = chosen.to_dict()
        rec["selection_mode"] = mode
        selected.append(rec)
    return pd.DataFrame(selected)


def summarize(selected: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    strict = (selected["final_balance"] >= float(args.target_balance)) & (selected["max_dd_pct"] >= -float(args.max_dd_pct)) & (selected["final_balance"] >= float(args.deposit))
    return {
        "folds": int(len(selected)),
        "target_folds": int((selected["final_balance"] >= float(args.target_balance)).sum()),
        "dd_pass_folds": int((selected["max_dd_pct"] >= -float(args.max_dd_pct)).sum()),
        "loss_folds": int((selected["final_balance"] < float(args.deposit)).sum()),
        "strict_pass_folds": int(strict.sum()),
        "min_final_balance": round(float(selected["final_balance"].min()), 2),
        "median_final_balance": round(float(selected["final_balance"].median()), 2),
        "worst_dd_pct": round(float(selected["max_dd_pct"].min()), 2),
        "bootstrap_folds": int((selected["selection_mode"] == "bootstrap_global_proxy").sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean proxy ML selector over candidate fold results.")
    parser.add_argument("--candidate-root", action="append", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--lookback-folds", type=int, default=80)
    parser.add_argument("--train-lookback-folds", type=int, default=0)
    parser.add_argument("--regime-lookback-days", type=int, default=60)
    parser.add_argument("--min-train-rows", type=int, default=40)
    parser.add_argument("--min-candidate-history-rows", type=int, default=5)
    parser.add_argument("--strict-weight", type=float, default=140000.0)
    parser.add_argument("--outcome-weight", type=float, default=35000.0)
    parser.add_argument("--hist-strict-weight", type=float, default=35000.0)
    parser.add_argument("--hist-target-weight", type=float, default=12000.0)
    parser.add_argument("--hist-dd-weight", type=float, default=8000.0)
    parser.add_argument("--hist-loss-weight", type=float, default=80000.0)
    parser.add_argument("--recent-strict-weight", type=float, default=10000.0)
    parser.add_argument("--recent-loss-weight", type=float, default=20000.0)
    args = parser.parse_args()

    folds = load_folds(args.fold_manifest)
    regimes = fast_load_feature_regimes(args.features, folds, int(args.regime_lookback_days))
    frame = load_candidates(args.candidate_root, args.target_balance, args.max_dd_pct, args.deposit)
    rows = build_rows(frame, regimes, args)
    selected = evaluate(rows, args)
    summary = summarize(selected, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_dir / "ml_proxy_training_matrix.csv", index=False)
    selected.to_csv(args.out_dir / "ml_proxy_selected_folds.csv", index=False)
    (args.out_dir / "ml_proxy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
