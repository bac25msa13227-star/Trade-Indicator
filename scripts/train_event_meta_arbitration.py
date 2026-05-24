from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DIRTY_KEYS = ("adaptive_per_fold", "research_oracle_fold_selection", "selection_uses_current_fold_metrics")


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except (UnicodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not parse JSON: {path}")


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def label_for(path: Path) -> str:
    return path.name


def load_feedback(candidate_dir: Path) -> pd.DataFrame:
    manifest = candidate_dir / "manifest.json"
    feedback = candidate_dir / "mt5_trade_feedback_detailed.csv"
    payload = read_json(manifest)
    dirty = [key for key in DIRTY_KEYS if is_true(payload.get(key))]
    if dirty:
        raise ValueError(f"refusing dirty candidate {candidate_dir}: {dirty}")
    frame = pd.read_csv(feedback)
    if frame.empty:
        return frame
    frame["candidate_label"] = label_for(candidate_dir)
    frame["candidate_dir"] = str(candidate_dir)
    for column in [
        "fold",
        "volume",
        "profit",
        "hour",
        "weekday",
        "signal_probability",
        "signal_atr",
        "signal_match_lag_min",
        "rr",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["open_time", "close_time"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame.dropna(subset=["fold", "open_time", "close_time", "profit", "volume"]).copy()


def max_drawdown_pct(curve: list[float]) -> float:
    peak = curve[0] if curve else 0.0
    worst = 0.0
    for value in curve:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, (value - peak) / peak * 100.0)
    return worst


def extra_cost(row: pd.Series, args: argparse.Namespace) -> float:
    volume = float(row["volume"])
    points = float(args.extra_roundtrip_points)
    hour = int(row.get("hour")) if pd.notna(row.get("hour")) else -1
    weekday = int(row.get("weekday")) if pd.notna(row.get("weekday")) else -1
    thin_hours = {int(x) for x in str(args.thin_hours_utc).split(",") if str(x).strip()}
    if hour in thin_hours:
        points += float(args.thin_hour_extra_points)
    if weekday == 4 and hour >= int(args.friday_cutoff_hour_utc):
        points += float(args.friday_extra_points)
    return points * float(args.point_size) * float(args.contract_size) * volume + (
        float(args.commission_per_lot_roundtrip) * volume
    )


def add_density_features(rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = rows.sort_values(["fold", "candidate_label", "open_time"]).copy()
    frames: list[pd.DataFrame] = []
    windows = [float(x) for x in str(args.density_windows_hours).split(",") if str(x).strip()]
    for _, group in rows.groupby(["fold", "candidate_label"], sort=False):
        group = group.sort_values("open_time").copy()
        times = group["open_time"].tolist()
        for window_hours in windows:
            window = pd.Timedelta(hours=window_hours)
            counts: list[int] = []
            left = 0
            for right, ts in enumerate(times):
                while left < right and times[left] < ts - window:
                    left += 1
                counts.append(right - left + 1)
            group[f"density_{int(window_hours)}h"] = counts
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def add_prior_features(rows: pd.DataFrame, stress_folds: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stress = stress_folds.copy()
    stress["strict"] = stress["target_hit"].astype(bool) & stress["dd_pass"].astype(bool) & ~stress["loss_fold"].astype(bool)
    out_parts: list[pd.DataFrame] = []
    for fold_id, group in rows.groupby("fold", sort=True):
        hist = stress[stress["fold"] < int(fold_id)].copy()
        if int(args.history_lookback) > 0:
            hist = hist[hist["fold"] >= int(fold_id) - int(args.history_lookback)].copy()
        stats: dict[str, dict[str, float]] = {}
        for label, h in hist.groupby("candidate_label"):
            stats[str(label)] = {
                "prior_rows": float(len(h)),
                "prior_strict_rate": float(h["strict"].mean()) if len(h) else 0.0,
                "prior_target_rate": float(h["target_hit"].astype(bool).mean()) if len(h) else 0.0,
                "prior_dd_rate": float(h["dd_pass"].astype(bool).mean()) if len(h) else 0.0,
                "prior_loss_rate": float(h["loss_fold"].astype(bool).mean()) if len(h) else 0.0,
                "prior_median_final": float(pd.to_numeric(h["final_balance"], errors="coerce").median()) if len(h) else 200.0,
                "prior_worst_dd": float(pd.to_numeric(h["max_dd_pct"], errors="coerce").min()) if len(h) else 0.0,
            }
        part = group.copy()
        for key in [
            "prior_rows",
            "prior_strict_rate",
            "prior_target_rate",
            "prior_dd_rate",
            "prior_loss_rate",
            "prior_median_final",
            "prior_worst_dd",
        ]:
            part[key] = part["candidate_label"].map(lambda label: stats.get(str(label), {}).get(key, np.nan))
        out_parts.append(part)
    return pd.concat(out_parts, ignore_index=True)


def prepare_rows(candidate_dirs: list[Path], stress_folds_path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.concat([load_feedback(path) for path in candidate_dirs], ignore_index=True)
    rows["extra_cost"] = rows.apply(lambda row: extra_cost(row, args), axis=1)
    rows["net_profit"] = rows["profit"].astype(float) - rows["extra_cost"].astype(float)
    rows["label_win"] = (rows["net_profit"] > 0).astype(int)
    rows["label_good"] = (
        (rows["net_profit"] > 0)
        & (pd.to_numeric(rows["signal_match_lag_min"], errors="coerce").fillna(999) <= float(args.max_label_lag_min))
    ).astype(int)
    rows["side_num"] = rows.get("direction", 0)
    rows["hour_sin"] = np.sin(pd.to_numeric(rows["hour"], errors="coerce").fillna(0) / 24.0 * 2.0 * np.pi)
    rows["hour_cos"] = np.cos(pd.to_numeric(rows["hour"], errors="coerce").fillna(0) / 24.0 * 2.0 * np.pi)
    rows["weekday_sin"] = np.sin(pd.to_numeric(rows["weekday"], errors="coerce").fillna(0) / 7.0 * 2.0 * np.pi)
    rows["weekday_cos"] = np.cos(pd.to_numeric(rows["weekday"], errors="coerce").fillna(0) / 7.0 * 2.0 * np.pi)
    rows = add_density_features(rows, args)
    stress_folds = pd.read_csv(stress_folds_path)
    for column in ["fold", "final_balance", "max_dd_pct", "trades_used"]:
        if column in stress_folds.columns:
            stress_folds[column] = pd.to_numeric(stress_folds[column], errors="coerce")
    rows = add_prior_features(rows, stress_folds, args)
    rows["candidate_code"] = pd.factorize(rows["candidate_label"])[0].astype(float)
    return rows, stress_folds


def feature_columns(rows: pd.DataFrame) -> list[str]:
    base = [
        "signal_probability",
        "signal_atr",
        "signal_match_lag_min",
        "rr",
        "volume",
        "side_num",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "candidate_code",
        "prior_rows",
        "prior_strict_rate",
        "prior_target_rate",
        "prior_dd_rate",
        "prior_loss_rate",
        "prior_median_final",
        "prior_worst_dd",
    ]
    density = [column for column in rows.columns if column.startswith("density_")]
    return [column for column in [*base, *density] if column in rows.columns]


def build_model(args: argparse.Namespace):
    if args.model_type == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.1),
        )
    if args.model_type == "regressor":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(n_estimators=300, min_samples_leaf=8, random_state=42, n_jobs=-1),
        )
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(n_estimators=400, min_samples_leaf=8, random_state=42, n_jobs=-1, class_weight="balanced"),
    )


def predict_score(model: Any, frame: pd.DataFrame, features: list[str], args: argparse.Namespace) -> np.ndarray:
    x = frame[features]
    if args.model_type == "regressor":
        return np.asarray(model.predict(x), dtype=float)
    proba = model.predict_proba(x)
    return np.asarray(proba[:, 1] if proba.shape[1] > 1 else proba[:, 0], dtype=float)


def simulate_fold(rows: pd.DataFrame, fold_id: int, features: list[str], args: argparse.Namespace) -> dict[str, Any]:
    train = rows[rows["fold"].astype(int) < int(fold_id)].copy()
    test = rows[rows["fold"].astype(int) == int(fold_id)].copy()
    if int(args.history_lookback) > 0:
        train = train[train["fold"].astype(int) >= int(fold_id) - int(args.history_lookback)].copy()
    if len(test) == 0:
        return empty_fold(fold_id, args)
    if len(train) < int(args.min_train_rows) or train[args.label_column].nunique() < 2:
        if args.bootstrap_labels:
            allowed = {x.strip() for x in str(args.bootstrap_labels).split(",") if x.strip()}
            test = test[test["candidate_label"].isin(allowed)].copy()
        if len(test) == 0:
            return empty_fold(fold_id, args)
        test["score"] = pd.to_numeric(test["signal_probability"], errors="coerce").fillna(0.0)
        auc = np.nan
    else:
        model = build_model(args)
        y = train[args.label_column] if args.model_type != "regressor" else train["net_profit"]
        model.fit(train[features], y)
        test["score"] = predict_score(model, test, features, args)
        try:
            auc = float(roc_auc_score(test[args.label_column], test["score"])) if test[args.label_column].nunique() > 1 else np.nan
        except ValueError:
            auc = np.nan
    if args.min_score is not None:
        test = test[test["score"] >= float(args.min_score)].copy()
    if len(test) == 0:
        return empty_fold(fold_id, args)

    test = test.sort_values(["open_time", "score", "signal_probability"], ascending=[True, False, False])
    balance = float(args.deposit)
    curve = [balance]
    peak = balance
    dd_floor = float(args.deposit) * (1.0 - float(args.max_dd_pct) / 100.0)
    open_until = pd.Timestamp.min
    selected: list[str] = []
    trades_by_date: dict[str, int] = {}
    trades_used = 0
    hit_target = False
    hit_dd = False
    for open_time, slot in test.groupby("open_time", sort=True):
        if pd.isna(open_time) or open_time < open_until:
            continue
        date_key = str(pd.Timestamp(open_time).date())
        if int(args.max_trades_per_day) > 0 and trades_by_date.get(date_key, 0) >= int(args.max_trades_per_day):
            continue
        eligible = slot.copy()
        if float(args.account_dd_score_penalty) != 0.0 and peak > 0:
            current_dd = max(0.0, (peak - balance) / peak * 100.0)
            eligible["score"] = eligible["score"] - current_dd * float(args.account_dd_score_penalty)
        if float(args.skip_when_peak_dd_pct) > 0 and peak > 0:
            current_dd = max(0.0, (peak - balance) / peak * 100.0)
            if current_dd >= float(args.skip_when_peak_dd_pct):
                continue
        best = eligible.sort_values(["score", "signal_probability"], ascending=[False, False]).iloc[0]
        balance += float(best["net_profit"])
        peak = max(peak, balance)
        curve.append(balance)
        trades_used += 1
        trades_by_date[date_key] = trades_by_date.get(date_key, 0) + 1
        selected.append(str(best["candidate_label"]))
        if pd.notna(best["close_time"]):
            open_until = max(open_until, best["close_time"])
        if balance <= dd_floor:
            hit_dd = True
            if args.stop_on_dd:
                break
        if float(args.stop_on_peak_dd_pct) > 0 and max_drawdown_pct(curve) <= -float(args.stop_on_peak_dd_pct):
            break
        if balance >= float(args.target_balance):
            hit_target = True
            if args.stop_on_target:
                break
    worst_dd = max_drawdown_pct(curve)
    return {
        "fold": int(fold_id),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "auc": round(float(auc), 4) if pd.notna(auc) else np.nan,
        "trades_used": int(trades_used),
        "final_balance": round(balance, 2),
        "net_pct": round((balance - float(args.deposit)) / float(args.deposit) * 100.0, 2),
        "max_dd_pct": round(worst_dd, 2),
        "target_hit": bool(hit_target or balance >= float(args.target_balance)),
        "dd_pass": bool(worst_dd >= -float(args.max_dd_pct) and not hit_dd),
        "loss_fold": bool(balance < float(args.deposit)),
        "selected_candidates": ",".join(pd.Series(selected).value_counts().head(5).index.astype(str).tolist()),
    }


def empty_fold(fold_id: int, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "fold": int(fold_id),
        "train_rows": 0,
        "test_rows": 0,
        "auc": np.nan,
        "trades_used": 0,
        "final_balance": float(args.deposit),
        "net_pct": 0.0,
        "max_dd_pct": 0.0,
        "target_hit": False,
        "dd_pass": True,
        "loss_fold": False,
        "selected_candidates": "",
    }


def summarize(folds: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(folds)
    strict = frame["target_hit"].astype(bool) & frame["dd_pass"].astype(bool) & ~frame["loss_fold"].astype(bool)
    return {
        "folds": int(len(frame)),
        "all_pass_folds": int(strict.sum()),
        "target_pass_folds": int(frame["target_hit"].sum()),
        "dd_pass_folds": int(frame["dd_pass"].sum()),
        "loss_folds": int(frame["loss_fold"].sum()),
        "min_final_balance": round(float(frame["final_balance"].min()), 2),
        "median_final_balance": round(float(frame["final_balance"].median()), 2),
        "worst_dd_pct": round(float(frame["max_dd_pct"].min()), 2),
        "total_trades_used": int(frame["trades_used"].sum()),
        "mean_auc": round(float(frame["auc"].dropna().mean()), 4) if frame["auc"].notna().any() else None,
        "gate_pass": bool(strict.all()),
        "fail_folds": frame.loc[~strict, "fold"].astype(int).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train chronological event-level meta arbitration policy.")
    parser.add_argument("--candidate-dir", action="append", type=Path, required=True)
    parser.add_argument("--stress-folds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold-report-out", type=Path, default=None)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=30)
    parser.add_argument("--model-type", choices=["extra_trees", "hgb", "regressor"], default="extra_trees")
    parser.add_argument("--label-column", choices=["label_win", "label_good"], default="label_good")
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--history-lookback", type=int, default=8)
    parser.add_argument("--bootstrap-labels", default="")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--density-windows-hours", default="24,72,168")
    parser.add_argument("--max-label-lag-min", type=float, default=5.0)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--stop-on-peak-dd-pct", type=float, default=0.0)
    parser.add_argument("--skip-when-peak-dd-pct", type=float, default=0.0)
    parser.add_argument("--account-dd-score-penalty", type=float, default=0.0)
    parser.add_argument("--max-trades-per-day", type=int, default=0)
    parser.add_argument("--extra-roundtrip-points", type=float, default=0.0)
    parser.add_argument("--thin-hours-utc", default="99")
    parser.add_argument("--thin-hour-extra-points", type=float, default=0.0)
    parser.add_argument("--friday-cutoff-hour-utc", type=int, default=20)
    parser.add_argument("--friday-extra-points", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=0.001)
    parser.add_argument("--contract-size", type=float, default=100.0)
    parser.add_argument("--commission-per-lot-roundtrip", type=float, default=0.10)
    parser.add_argument("--no-stop-on-target", dest="stop_on_target", action="store_false")
    parser.add_argument("--no-stop-on-dd", dest="stop_on_dd", action="store_false")
    parser.set_defaults(stop_on_target=True, stop_on_dd=True)
    args = parser.parse_args()

    rows, _ = prepare_rows(args.candidate_dir, args.stress_folds, args)
    features = feature_columns(rows)
    folds = [simulate_fold(rows, fold_id, features, args) for fold_id in range(args.fold_start, args.fold_end + 1)]
    report = {"summary": summarize(folds), "features": features, "folds": folds}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.fold_report_out:
        pd.DataFrame(folds).to_csv(args.fold_report_out, index=False)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
