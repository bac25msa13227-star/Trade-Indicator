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

from scripts.mt5_prior_sequence_regime_throttle_sweep import max_drawdown_pct, parse_floats, state_features


FEATURE_COLUMNS = [
    "hour",
    "weekday",
    "direction",
    "signal_probability",
    "signal_atr",
    "rr",
    "sl_distance",
    "spread_to_sl_r",
    "cost_r",
    "risk_usd",
    "pre_dd_pct",
    "loss_streak",
    "win_streak",
    "last_r",
    "rolling3_r",
    "rolling5_r",
    "rolling10_r",
    "rolling3_winrate",
    "rolling5_winrate",
    "rolling10_winrate",
    "trade_index",
]

EXTRA_FEATURE_COLUMNS: list[str] = []


def numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def add_cost_and_r(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = frame.copy()
    for column in [
        "fold",
        "profit",
        "volume",
        "entry_price",
        "sl_price",
        "tp_price",
        "direction",
        "signal_probability",
        "signal_atr",
        "hour",
        "weekday",
        "rr",
    ]:
        out[column] = pd.to_numeric(out.get(column), errors="coerce")
    for column in ["open_time", "close_time"]:
        out[column] = pd.to_datetime(out[column], errors="coerce")
    out = out.dropna(subset=["fold", "profit", "volume", "entry_price", "sl_price"]).copy()
    out["fold"] = out["fold"].astype(int)
    sl_distance = (out["entry_price"] - out["sl_price"]).abs()
    risk_usd = sl_distance * out["volume"] * float(args.contract_size)
    spread_price = float(args.extra_roundtrip_points) * float(args.point_size)
    cost = spread_price * float(args.contract_size) * out["volume"]
    cost += float(args.commission_per_lot_roundtrip) * out["volume"]
    out["raw_profit"] = out["profit"]
    out["extra_cost"] = cost
    out["profit"] = out["profit"] - cost
    out["sl_distance"] = sl_distance
    out["risk_usd"] = risk_usd
    out["spread_to_sl_r"] = spread_price / sl_distance.replace(0, np.nan)
    out["cost_r"] = cost / risk_usd.replace(0, np.nan)
    realized = out["profit"] / risk_usd.replace(0, np.nan)
    out["realized_r"] = realized.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 8.0)
    return out


def parse_signal_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y.%m.%d %H:%M", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], errors="coerce")
    return parsed


def join_bar_features(feedback: pd.DataFrame, features_path: Path | None) -> tuple[pd.DataFrame, list[str]]:
    if features_path is None:
        return feedback, []
    header = pd.read_csv(features_path, nrows=0).columns.tolist()
    usecols = [column for column in header if column != "trade_side"]
    features = pd.read_csv(features_path, usecols=usecols)
    features["join_time"] = pd.to_datetime(features["time"], errors="coerce")
    features = features.dropna(subset=["join_time"]).drop_duplicates("join_time", keep="last")
    blocked = {"time", "join_time", "open", "high", "low", "close"}
    feature_cols = []
    for column in features.columns:
        if column in blocked:
            continue
        if pd.api.types.is_numeric_dtype(features[column]):
            feature_cols.append(column)
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce")
            if features[column].notna().any():
                feature_cols.append(column)
    renamed = {column: f"bar_{column}" for column in feature_cols}
    features = features[["join_time", *feature_cols]].rename(columns=renamed)
    out = feedback.copy()
    if "signal_open_time" not in out.columns:
        out["signal_open_time"] = out["open_time"]
    out["join_time"] = parse_signal_time(out["signal_open_time"])
    joined = out.merge(features, on="join_time", how="left")
    return joined, list(renamed.values())


def build_training_rows(feedback: pd.DataFrame, deposit: float, dd_bad_pct: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fold_id, group in feedback.groupby("fold", sort=True):
        balance = float(deposit)
        peak = balance
        recent_r: list[float] = []
        loss_streak = 0
        win_streak = 0
        for trade_index, (_, row) in enumerate(group.sort_values(["close_time", "open_time"]).iterrows(), start=1):
            state = state_features(
                balance=balance,
                peak=peak,
                recent_r=recent_r,
                loss_streak=loss_streak,
                win_streak=win_streak,
                trade_index=trade_index,
            )
            post_balance = balance + float(row["profit"])
            post_peak = max(peak, post_balance)
            post_dd = (post_balance - post_peak) / post_peak * 100.0 if post_peak > 0 else 0.0
            rec = {
                "fold": int(fold_id),
                "profit": float(row["profit"]),
                "realized_r": float(row["realized_r"]),
                "label_positive": int(float(row["realized_r"]) > 0.05),
                "label_bad": int(float(row["profit"]) < 0 or post_dd <= -float(dd_bad_pct)),
                "hour": float(row.get("hour", 0.0)),
                "weekday": float(row.get("weekday", 0.0)),
                "direction": float(row.get("direction", 0.0)),
                "signal_probability": float(row.get("signal_probability", 0.5)),
                "signal_atr": float(row.get("signal_atr", 0.0)),
                "rr": float(row.get("rr", 0.0)),
                "sl_distance": float(row.get("sl_distance", 0.0)),
                "spread_to_sl_r": float(row.get("spread_to_sl_r", 0.0)),
                "cost_r": float(row.get("cost_r", 0.0)),
                "risk_usd": float(row.get("risk_usd", 0.0)),
                **state,
            }
            for column in EXTRA_FEATURE_COLUMNS:
                rec[column] = float(row.get(column, 0.0)) if pd.notna(row.get(column, np.nan)) else 0.0
            records.append(rec)
            balance = post_balance
            peak = post_peak
            recent_r.append(float(row["realized_r"]))
            if float(row["profit"]) < 0:
                loss_streak += 1
                win_streak = 0
            elif float(row["profit"]) > 0:
                win_streak += 1
                loss_streak = 0
    return pd.DataFrame(records)


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
        random_state=seed + 10000,
        n_jobs=int(args.model_jobs),
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", reg)]), Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("model", clf)]
    )


def fit_models(training_rows: pd.DataFrame, fold_id: int, args: argparse.Namespace) -> dict[str, Any]:
    train = training_rows[training_rows["fold"] < fold_id].copy()
    bundle: dict[str, Any] = {"model_used": False, "train_rows": int(len(train)), "reg": None, "clf": None}
    if len(train) < int(args.min_train_rows) or train["label_bad"].nunique() < 2:
        return bundle
    reg, clf = make_models(args, fold_id)
    columns = FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS
    reg.fit(train[columns], train["realized_r"])
    clf.fit(train[columns], train["label_bad"])
    bundle.update({"model_used": True, "reg": reg, "clf": clf})
    return bundle


def feature_row(row: pd.Series, state: dict[str, float]) -> dict[str, float]:
    out = {
        "hour": float(row.get("hour", 0.0)),
        "weekday": float(row.get("weekday", 0.0)),
        "direction": float(row.get("direction", 0.0)),
        "signal_probability": float(row.get("signal_probability", 0.5)),
        "signal_atr": float(row.get("signal_atr", 0.0)),
        "rr": float(row.get("rr", 0.0)),
        "sl_distance": float(row.get("sl_distance", 0.0)),
        "spread_to_sl_r": float(row.get("spread_to_sl_r", 0.0)),
        "cost_r": float(row.get("cost_r", 0.0)),
        "risk_usd": float(row.get("risk_usd", 0.0)),
        **state,
    }
    for column in EXTRA_FEATURE_COLUMNS:
        out[column] = float(row.get(column, 0.0)) if pd.notna(row.get(column, np.nan)) else 0.0
    return out


def simulate_fold(
    group: pd.DataFrame,
    model_bundle: dict[str, Any],
    args: argparse.Namespace,
    *,
    min_score: float,
    max_bad_prob: float,
    reduce_score: float,
    reduce_bad_prob: float,
    reduce_mult: float,
    pre_dd_skip_pct: float,
    loss_streak_skip: float,
) -> dict[str, Any]:
    balance = float(args.deposit)
    peak = balance
    curve = [balance]
    recent_r: list[float] = []
    loss_streak = 0
    win_streak = 0
    used = skipped = reduced = model_hits = 0
    score_sum = bad_sum = 0.0
    reg = model_bundle.get("reg")
    clf = model_bundle.get("clf")
    use_model = bool(model_bundle.get("model_used"))
    for trade_index, (_, row) in enumerate(group.sort_values(["close_time", "open_time"]).iterrows(), start=1):
        state = state_features(
            balance=balance,
            peak=peak,
            recent_r=recent_r,
            loss_streak=loss_streak,
            win_streak=win_streak,
            trade_index=trade_index,
        )
        if float(state["pre_dd_pct"]) <= -float(pre_dd_skip_pct) or loss_streak >= int(loss_streak_skip):
            skipped += 1
            continue
        if use_model and reg is not None and clf is not None:
            x = pd.DataFrame([feature_row(row, state)], columns=FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS)
            expected_r = float(reg.predict(x)[0])
            bad_prob = float(clf.predict_proba(x)[:, 1][0])
            score = expected_r - float(args.bad_weight) * bad_prob - float(args.cost_weight) * float(row.get("cost_r", 0.0))
            model_hits += 1
        else:
            expected_r = float(row.get("realized_r", 0.0)) if args.cold_start_oracle else 0.0
            bad_prob = 0.0
            score = expected_r
        score_sum += score
        bad_sum += bad_prob
        if score < float(min_score) or bad_prob > float(max_bad_prob):
            skipped += 1
            continue
        mult = float(reduce_mult) if (score < float(reduce_score) or bad_prob > float(reduce_bad_prob)) else 1.0
        profit = float(row["profit"]) * mult
        if mult < 0.999:
            reduced += 1
        balance += profit
        peak = max(peak, balance)
        curve.append(balance)
        used += 1
        realized_r = float(row.get("realized_r", 0.0)) * mult
        recent_r.append(realized_r)
        if profit < 0:
            loss_streak += 1
            win_streak = 0
        elif profit > 0:
            win_streak += 1
            loss_streak = 0
        if balance >= float(args.target_balance):
            break
    dd = max_drawdown_pct(curve)
    return {
        "fold": int(group["fold"].iloc[0]),
        "final_balance": round(balance, 2),
        "target_hit": bool(balance >= float(args.target_balance)),
        "max_dd_pct": round(dd, 2),
        "dd_pass": bool(dd >= -float(args.max_dd_pct)),
        "loss_fold": bool(balance < float(args.deposit)),
        "trades_used": int(used),
        "trades_skipped": int(skipped),
        "trades_reduced": int(reduced),
        "model_hits": int(model_hits),
        "avg_score": round(score_sum / max(used + skipped, 1), 4),
        "avg_bad_prob": round(bad_sum / max(used + skipped, 1), 4),
        "model_train_rows": int(model_bundle.get("train_rows") or 0),
        "model_used": bool(use_model),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    strict = frame["target_hit"] & frame["dd_pass"] & (~frame["loss_fold"])
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
        "total_trades_skipped": int(frame["trades_skipped"].sum()),
        "total_trades_reduced": int(frame["trades_reduced"].sum()),
        "fail_folds": frame.loc[~strict, "fold"].astype(int).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward DD-aware trade selector sweep on MT5 feedback.")
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--extra-roundtrip-points", type=float, default=308.0)
    parser.add_argument("--point-size", type=float, default=0.001)
    parser.add_argument("--contract-size", type=float, default=100.0)
    parser.add_argument("--commission-per-lot-roundtrip", type=float, default=0.0)
    parser.add_argument("--dd-bad-pct", type=float, default=14.0)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--min-samples-leaf", type=int, default=12)
    parser.add_argument("--model-jobs", type=int, default=1)
    parser.add_argument("--bad-weight", type=float, default=0.55)
    parser.add_argument("--cost-weight", type=float, default=0.15)
    parser.add_argument("--min-scores", default="-0.5,-0.35,-0.2,-0.1,0")
    parser.add_argument("--max-bad-probs", default="0.55,0.65,0.75,0.85")
    parser.add_argument("--reduce-scores", default="-0.1,0,0.1")
    parser.add_argument("--reduce-bad-probs", default="0.45,0.55,0.65")
    parser.add_argument("--reduce-mults", default="0.35,0.5,0.75,1.0")
    parser.add_argument("--pre-dd-skip-pcts", default="16,18,20")
    parser.add_argument("--loss-streak-skips", default="3,4,99")
    parser.add_argument("--cold-start-oracle", action="store_true")
    args = parser.parse_args()

    global EXTRA_FEATURE_COLUMNS
    feedback = add_cost_and_r(pd.read_csv(args.feedback), args)
    feedback, EXTRA_FEATURE_COLUMNS = join_bar_features(feedback, args.features)
    training_rows = build_training_rows(feedback, args.deposit, args.dd_bad_pct)
    folds = sorted(feedback["fold"].unique().tolist())
    model_cache = {fold_id: fit_models(training_rows, fold_id, args) for fold_id in folds}
    records: list[dict[str, Any]] = []
    details: dict[str, list[dict[str, Any]]] = {}
    for min_score in parse_floats(args.min_scores):
        for max_bad_prob in parse_floats(args.max_bad_probs):
            for reduce_score in parse_floats(args.reduce_scores):
                for reduce_bad_prob in parse_floats(args.reduce_bad_probs):
                    for reduce_mult in parse_floats(args.reduce_mults):
                        for pre_dd_skip_pct in parse_floats(args.pre_dd_skip_pcts):
                            for loss_streak_skip in parse_floats(args.loss_streak_skips):
                                rows = [
                                    simulate_fold(
                                        feedback[feedback["fold"] == fold_id].copy(),
                                        model_cache[fold_id],
                                        args,
                                        min_score=min_score,
                                        max_bad_prob=max_bad_prob,
                                        reduce_score=reduce_score,
                                        reduce_bad_prob=reduce_bad_prob,
                                        reduce_mult=reduce_mult,
                                        pre_dd_skip_pct=pre_dd_skip_pct,
                                        loss_streak_skip=loss_streak_skip,
                                    )
                                    for fold_id in folds
                                ]
                                summary = summarize(rows)
                                key = (
                                    f"ms{min_score:g}_mbp{max_bad_prob:g}_rs{reduce_score:g}_"
                                    f"rbp{reduce_bad_prob:g}_rm{reduce_mult:g}_pdd{pre_dd_skip_pct:g}_"
                                    f"ls{loss_streak_skip:g}"
                                )
                                summary.update(
                                    {
                                        "key": key,
                                        "min_score": float(min_score),
                                        "max_bad_prob": float(max_bad_prob),
                                        "reduce_score": float(reduce_score),
                                        "reduce_bad_prob": float(reduce_bad_prob),
                                        "reduce_mult": float(reduce_mult),
                                        "pre_dd_skip_pct": float(pre_dd_skip_pct),
                                        "loss_streak_skip": float(loss_streak_skip),
                                    }
                                )
                                records.append(summary)
                                details[key] = rows
    records.sort(
        key=lambda item: (
            item["all_pass_folds"],
            item["target_pass_folds"],
            item["dd_pass_folds"],
            -item["loss_folds"],
            item["worst_dd_pct"],
            item["median_final_balance"],
        ),
        reverse=True,
    )
    report = {
        "best": records[:30],
        "details": {record["key"]: details[record["key"]] for record in records[:10]},
        "features": FEATURE_COLUMNS,
        "extra_features": EXTRA_FEATURE_COLUMNS,
        "cost": {
            "extra_roundtrip_points": float(args.extra_roundtrip_points),
            "point_size": float(args.point_size),
            "contract_size": float(args.contract_size),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best": records[:10]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
