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
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xauusd_ai.features.dataset import FEATURE_COLUMNS  # noqa: E402

import scripts.export_mt5_profitr_model_manifest as profitr

MT5_COLUMNS = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]


def window_dd_pct(profits: list[float], start_balance: float) -> float:
    balance = float(start_balance)
    peak = balance
    worst = 0.0
    for profit in profits:
        balance += float(profit)
        peak = max(peak, balance)
        if peak > 0:
            worst = min(worst, (balance - peak) / peak * 100.0)
    return worst


def build_sequence_frame(feedback: pd.DataFrame, features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, group in feedback.groupby("fold", sort=True):
        ordered = group.sort_values(["close_time", "open_time"], na_position="last").reset_index(drop=True).copy()
        future_sum_r: list[float] = []
        future_dd: list[float] = []
        for idx, row in ordered.iterrows():
            future = ordered.iloc[idx : idx + int(args.window_trades)]
            profits = pd.to_numeric(future["profit"], errors="coerce").fillna(0.0).tolist()
            rs = pd.to_numeric(future["realized_r"], errors="coerce").fillna(0.0).tolist()
            future_sum_r.append(float(np.sum(rs)))
            future_dd.append(window_dd_pct(profits, float(args.deposit)))
        ordered["future_sum_r"] = future_sum_r
        ordered["future_window_dd_pct"] = future_dd
        rows.append(ordered)
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    frame["join_time"] = profitr.parse_time(frame["signal_open_time"])
    joined = frame.join(features, on="join_time", how="inner", rsuffix="_feature")
    joined["signal_probability"] = profitr.numeric(joined, "signal_probability")
    joined["signal_atr"] = profitr.numeric(joined, "signal_atr")
    joined["direction"] = profitr.numeric(joined, "direction")
    joined["hour"] = joined["join_time"].dt.hour
    joined["weekday"] = joined["join_time"].dt.weekday
    joined["label_sequence_bad"] = (
        (pd.to_numeric(joined["future_sum_r"], errors="coerce").fillna(0.0) < float(args.min_future_r_label))
        | (pd.to_numeric(joined["future_window_dd_pct"], errors="coerce").fillna(0.0) < -float(args.bad_window_dd_pct))
    ).astype(int)
    return joined


def feature_columns(frame: pd.DataFrame) -> list[str]:
    extras = ["signal_probability", "signal_atr", "direction", "hour", "weekday"]
    return [c for c in FEATURE_COLUMNS + extras if c in frame.columns]


def make_models(args: argparse.Namespace, seed: int) -> tuple[Pipeline, Pipeline]:
    reg = ExtraTreesRegressor(
        n_estimators=int(args.n_estimators),
        min_samples_leaf=int(args.min_samples_leaf),
        max_features="sqrt",
        random_state=seed,
        n_jobs=int(args.model_jobs),
    )
    clf = ExtraTreesClassifier(
        n_estimators=int(args.n_estimators),
        min_samples_leaf=int(args.min_samples_leaf),
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed + 7000,
        n_jobs=int(args.model_jobs),
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", reg)]), Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("model", clf)]
    )


def score_signals(signals: pd.DataFrame, features: pd.DataFrame, reg: Pipeline, clf: Pipeline, cols: list[str], args: argparse.Namespace) -> pd.DataFrame:
    scored = profitr.add_join_time_from_signal(signals)
    scored["signal_probability"] = profitr.numeric(scored, "probability")
    scored["signal_atr"] = profitr.numeric(scored, "atr")
    scored["direction"] = profitr.numeric(scored, "direction")
    scored["hour"] = scored["join_time"].dt.hour
    scored["weekday"] = scored["join_time"].dt.weekday
    joined = scored.join(features, on="join_time", how="inner", rsuffix="_feature")
    if joined.empty:
        return joined
    x = joined[cols]
    joined["seq_expected_future_r"] = reg.predict(x)
    joined["seq_bad_probability"] = clf.predict_proba(x)[:, 1]
    joined["seq_score"] = joined["seq_expected_future_r"] - float(args.bad_prob_weight) * joined["seq_bad_probability"]
    return joined


def diversify_rows(rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if rows.empty:
        return rows
    if int(args.max_per_day) <= 0 and int(args.min_spacing_minutes) <= 0:
        return rows
    selected: list[int] = []
    per_day: dict[str, int] = {}
    selected_times: list[pd.Timestamp] = []
    for idx, row in rows.iterrows():
        ts = pd.to_datetime(row.get("join_time"), errors="coerce")
        day = ts.strftime("%Y-%m-%d") if pd.notna(ts) else ""
        if int(args.max_per_day) > 0 and per_day.get(day, 0) >= int(args.max_per_day):
            continue
        if int(args.min_spacing_minutes) > 0 and pd.notna(ts):
            too_close = any(abs((ts - other).total_seconds()) < int(args.min_spacing_minutes) * 60 for other in selected_times)
            if too_close:
                continue
        selected.append(idx)
        per_day[day] = per_day.get(day, 0) + 1
        if pd.notna(ts):
            selected_times.append(ts)
        if len(selected) >= int(args.top_k):
            break
    return rows.loc[selected].copy()


def select_rows(scored: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scored.empty:
        return pd.DataFrame(columns=MT5_COLUMNS), scored
    kept = scored[
        (pd.to_numeric(scored["seq_expected_future_r"], errors="coerce") >= float(args.min_expected_future_r))
        & (pd.to_numeric(scored["seq_bad_probability"], errors="coerce") <= float(args.max_bad_probability))
        & (pd.to_numeric(scored["seq_score"], errors="coerce") >= float(args.min_score))
    ].copy()
    if len(kept) < int(args.min_keep):
        kept = scored.sort_values("seq_score", ascending=False).copy()
        kept = diversify_rows(kept, args).head(int(args.min_keep)).copy()
    else:
        kept = kept.sort_values("seq_score", ascending=False).copy()
        kept = diversify_rows(kept, args).head(int(args.top_k)).copy()
    if bool(args.dedupe_open_time) and not kept.empty:
        kept = kept.sort_values("seq_score", ascending=False).drop_duplicates(["open_time"], keep="first").copy()
    export = kept[[c for c in MT5_COLUMNS if c in kept.columns]].copy()
    for col in MT5_COLUMNS:
        if col not in export.columns:
            export[col] = np.nan
    return export[MT5_COLUMNS].copy(), kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MT5 manifest using prior sequence/path MT5 feedback labels.")
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--risk-pct", type=float, default=7.0)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--window-trades", type=int, default=8)
    parser.add_argument("--bad-window-dd-pct", type=float, default=12.0)
    parser.add_argument("--min-future-r-label", type=float, default=0.0)
    parser.add_argument("--min-train-rows", type=int, default=600)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--model-jobs", type=int, default=-1)
    parser.add_argument("--min-expected-future-r", type=float, default=0.0)
    parser.add_argument("--max-bad-probability", type=float, default=0.65)
    parser.add_argument("--bad-prob-weight", type=float, default=2.0)
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--top-k", type=int, default=220)
    parser.add_argument("--min-keep", type=int, default=30)
    parser.add_argument("--max-per-day", type=int, default=0)
    parser.add_argument("--min-spacing-minutes", type=int, default=0)
    parser.add_argument("--exclude-hours", default="")
    parser.add_argument("--feature-conditions", default="")
    parser.add_argument("--dedupe-open-time", action="store_true")
    parser.add_argument("--cold-start-mode", choices=["source_signals", "no_trade"], default="source_signals")
    args = parser.parse_args()

    manifest = profitr.read_json(args.base_manifest)
    feedback = pd.read_csv(args.feedback)
    feedback["fold"] = pd.to_numeric(feedback["fold"], errors="coerce")
    feedback = profitr.add_realized_r(feedback, 100.0, -5.0, 5.0)
    feedback["close_time"] = pd.to_datetime(feedback["close_time"], errors="coerce")
    feedback["open_time"] = pd.to_datetime(feedback["open_time"], errors="coerce")
    features = profitr.load_features(args.features)
    sequence = build_sequence_frame(feedback, features, args)
    cols = feature_columns(sequence)
    exclude_hours = profitr.parse_hours(args.exclude_hours)
    feature_conditions = profitr.parse_feature_conditions(args.feature_conditions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    total = 0
    for source_fold in manifest.get("folds", []) or []:
        fold_id = int(source_fold["fold"])
        train = sequence[sequence["fold"] < fold_id].copy()
        source_csv = profitr.resolve_csv(args.base_manifest, str(source_fold["csv"]))
        signals = pd.read_csv(source_csv)
        export = pd.DataFrame(columns=MT5_COLUMNS)
        audit = pd.DataFrame()
        mode = "no_trade_insufficient_sequence_feedback"
        if len(train) >= int(args.min_train_rows) and train["label_sequence_bad"].nunique() == 2 and cols:
            reg, clf = make_models(args, fold_id)
            reg.fit(train[cols], train["future_sum_r"])
            clf.fit(train[cols], train["label_sequence_bad"])
            scored = score_signals(signals, features, reg, clf, cols, args)
            if exclude_hours is not None and not scored.empty:
                scored = scored[~pd.to_datetime(scored["join_time"], errors="coerce").dt.hour.isin(exclude_hours)].copy()
            scored = profitr.apply_feature_conditions(scored, feature_conditions)
            export, audit = select_rows(scored, args)
            mode = "prior_mt5_sequence_path_model"
        elif args.cold_start_mode == "source_signals":
            export = signals[[c for c in MT5_COLUMNS if c in signals.columns]].copy()
            if exclude_hours is not None:
                jt = profitr.parse_time(export["open_time"])
                export = export[~jt.dt.hour.isin(exclude_hours)].copy()
            if feature_conditions:
                export = profitr.add_join_time_from_signal(export).join(features, on="join_time", how="inner", rsuffix="_feature")
                export = profitr.apply_feature_conditions(export, feature_conditions)
                export = export[[c for c in MT5_COLUMNS if c in export.columns]].copy()
            for col in MT5_COLUMNS:
                if col not in export.columns:
                    export[col] = np.nan
            export = export[MT5_COLUMNS].copy()
            mode = "cold_start_source_signals"
        target_csv = args.out_dir / f"fold_{fold_id:02d}_signals.csv"
        audit_csv = args.out_dir / f"fold_{fold_id:02d}_sequence_audit.csv"
        export.to_csv(target_csv, index=False, float_format="%.5f")
        audit.to_csv(audit_csv, index=False, float_format="%.5f")
        risk_pct = float(args.risk_pct)
        fold = dict(source_fold)
        fold.update(
            {
                "signals": int(len(export)),
                "csv": str(target_csv),
                "sequence_audit_csv": str(audit_csv),
                "risk_pct": risk_pct,
                "max_risk_pct": risk_pct,
                "max_exposure_pct": risk_pct * int(args.max_positions),
                "max_positions": int(args.max_positions),
                "selection_mode": mode,
                "selection_uses_current_fold_metrics": False,
                "research_oracle_fold_selection": False,
                "mt5_sequence_path_model_selector": True,
                "sequence_model_train_rows": int(len(train)),
                "sequence_bad_rate": float(train["label_sequence_bad"].mean()) if len(train) else None,
                "sequence_window_trades": int(args.window_trades),
                "sequence_bad_window_dd_pct": float(args.bad_window_dd_pct),
                "sequence_min_future_r_label": float(args.min_future_r_label),
                "sequence_min_expected_future_r": float(args.min_expected_future_r),
                "sequence_max_bad_probability": float(args.max_bad_probability),
                "sequence_bad_prob_weight": float(args.bad_prob_weight),
                "sequence_max_per_day": int(args.max_per_day),
                "sequence_min_spacing_minutes": int(args.min_spacing_minutes),
            }
        )
        folds.append(fold)
        total += int(len(export))
        rules.append({"fold": fold_id, "signals": int(len(export)), "mode": mode, "train_rows": int(len(train))})

    out_manifest = {
        "all_signals": None,
        "folds": folds,
        "total_signals": int(total),
        "live_protocol": True,
        "mt5_sequence_path_model_selector": True,
        "selection_uses_current_fold_metrics": False,
        "research_oracle_fold_selection": False,
        "base_manifest": str(args.base_manifest),
        "feedback": str(args.feedback),
        "features": str(args.features),
        "feature_conditions": str(args.feature_conditions or ""),
        "target_basis": "prior_mt5_sequence_path_feedback",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    pd.DataFrame(rules).to_csv(args.out_dir / "sequence_path_model_rules.csv", index=False)
    print(json.dumps({"manifest": str(args.out_dir / "manifest.json"), "folds": len(folds), "signals": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
