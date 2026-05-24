from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "hour",
    "weekday",
    "direction",
    "signal_probability",
    "signal_atr",
    "rr",
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


def numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def max_drawdown_pct(curve: list[float]) -> float:
    peak = curve[0]
    worst = 0.0
    for value in curve:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, (value - peak) / peak * 100.0)
    return worst


def add_realized_r(frame: pd.DataFrame, contract_value_per_lot: float) -> pd.DataFrame:
    out = frame.copy()
    entry = numeric(out, "entry_price", np.nan)
    sl = numeric(out, "sl_price", np.nan)
    volume = numeric(out, "volume", np.nan)
    profit = numeric(out, "profit", 0.0)
    risk_usd = (entry - sl).abs() * volume * float(contract_value_per_lot)
    realized_r = profit / risk_usd.where(risk_usd > 0)
    fallback = pd.Series(0.0, index=out.index, dtype=float)
    outcome = out.get("outcome", pd.Series("", index=out.index)).astype(str).str.lower()
    fallback.loc[outcome.str.contains("tp", na=False)] = numeric(out, "rr", 1.0).clip(lower=0.1)
    fallback.loc[outcome.str.contains("sl", na=False)] = -1.0
    fallback.loc[(fallback == 0.0) & (profit > 0)] = 1.0
    fallback.loc[(fallback == 0.0) & (profit < 0)] = -1.0
    out["realized_r"] = realized_r.replace([np.inf, -np.inf], np.nan).fillna(fallback).clip(-3.0, 5.0)
    return out


def state_features(
    *,
    balance: float,
    peak: float,
    recent_r: list[float],
    loss_streak: int,
    win_streak: int,
    trade_index: int,
) -> dict[str, float]:
    pre_dd = (balance - peak) / peak * 100.0 if peak > 0 else 0.0

    def sum_last(n: int) -> float:
        return float(np.sum(recent_r[-n:])) if recent_r else 0.0

    def winrate_last(n: int) -> float:
        sample = recent_r[-n:]
        return float(np.mean([1.0 if value > 0 else 0.0 for value in sample])) if sample else 0.5

    return {
        "pre_dd_pct": float(pre_dd),
        "loss_streak": float(loss_streak),
        "win_streak": float(win_streak),
        "last_r": float(recent_r[-1]) if recent_r else 0.0,
        "rolling3_r": sum_last(3),
        "rolling5_r": sum_last(5),
        "rolling10_r": sum_last(10),
        "rolling3_winrate": winrate_last(3),
        "rolling5_winrate": winrate_last(5),
        "rolling10_winrate": winrate_last(10),
        "trade_index": float(trade_index),
    }


def build_training_rows(feedback: pd.DataFrame, deposit: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fold_id, group in feedback.groupby("fold", sort=True):
        balance = float(deposit)
        peak = balance
        recent_r: list[float] = []
        loss_streak = 0
        win_streak = 0
        for trade_index, (_, row) in enumerate(group.sort_values(["close_time", "open_time"]).iterrows(), start=1):
            rec = {
                "fold": int(fold_id),
                "profit": float(row["profit"]),
                "realized_r": float(row["realized_r"]),
                "label_positive": int(float(row["realized_r"]) > 0.05),
                "hour": float(row.get("hour", 0)),
                "weekday": float(row.get("weekday", 0)),
                "direction": float(row.get("direction", 0)),
                "signal_probability": float(row.get("signal_probability", 0.5)),
                "signal_atr": float(row.get("signal_atr", 0.0)),
                "rr": float(row.get("rr", 0.0)),
            }
            rec.update(
                state_features(
                    balance=balance,
                    peak=peak,
                    recent_r=recent_r,
                    loss_streak=loss_streak,
                    win_streak=win_streak,
                    trade_index=trade_index,
                )
            )
            records.append(rec)
            balance += float(row["profit"])
            peak = max(peak, balance)
            recent_r.append(float(row["realized_r"]))
            if float(row["profit"]) < 0:
                loss_streak += 1
                win_streak = 0
            elif float(row["profit"]) > 0:
                win_streak += 1
                loss_streak = 0
    return pd.DataFrame(records)


def make_models(seed: int, n_estimators: int, min_samples_leaf: int, n_jobs: int) -> tuple[Pipeline, Pipeline]:
    reg = ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        max_features="sqrt",
        random_state=seed,
        n_jobs=int(n_jobs),
    )
    clf = ExtraTreesClassifier(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed + 10000,
        n_jobs=int(n_jobs),
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", reg)]), Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("model", clf)]
    )


def simulate_fold(
    group: pd.DataFrame,
    *,
    args: argparse.Namespace,
    model_bundle: dict[str, Any],
    min_score: float,
    reduce_score: float,
    reduce_mult: float,
    pre_dd_skip_pct: float,
) -> dict[str, Any]:
    fold_id = int(group["fold"].iloc[0])
    balance = float(args.deposit)
    peak = balance
    curve = [balance]
    recent_r: list[float] = []
    loss_streak = 0
    win_streak = 0
    used = 0
    skipped = 0
    reduced = 0
    score_sum = 0.0
    score_count = 0

    use_model = bool(model_bundle.get("model_used"))
    reg = model_bundle.get("reg")
    clf = model_bundle.get("clf")

    for trade_index, (_, row) in enumerate(group.sort_values(["close_time", "open_time"]).iterrows(), start=1):
        state = state_features(
            balance=balance,
            peak=peak,
            recent_r=recent_r,
            loss_streak=loss_streak,
            win_streak=win_streak,
            trade_index=trade_index,
        )
        features = {
            "hour": float(row.get("hour", 0)),
            "weekday": float(row.get("weekday", 0)),
            "direction": float(row.get("direction", 0)),
            "signal_probability": float(row.get("signal_probability", 0.5)),
            "signal_atr": float(row.get("signal_atr", 0.0)),
            "rr": float(row.get("rr", 0.0)),
            **state,
        }
        if float(state["pre_dd_pct"]) <= -float(pre_dd_skip_pct):
            skipped += 1
            continue
        if use_model and reg is not None and clf is not None:
            x = pd.DataFrame([features], columns=FEATURE_COLUMNS)
            expected_r = float(reg.predict(x)[0])
            positive_prob = float(clf.predict_proba(x)[:, 1][0])
            score = expected_r + float(args.prob_weight) * (positive_prob - 0.5)
        else:
            score = 999.0
        score_sum += float(score)
        score_count += 1
        if score < float(min_score):
            skipped += 1
            continue
        mult = float(reduce_mult) if score < float(reduce_score) else 1.0
        if mult < 0.999:
            reduced += 1
        profit = float(row["profit"]) * mult
        balance += profit
        peak = max(peak, balance)
        curve.append(balance)
        used += 1
        realized_r = float(row["realized_r"]) * mult
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
        "fold": fold_id,
        "final_balance": round(balance, 2),
        "target_hit": bool(balance >= float(args.target_balance)),
        "max_dd_pct": round(dd, 2),
        "dd_pass": bool(dd >= -float(args.max_dd_pct)),
        "loss_fold": bool(balance < float(args.deposit)),
        "trades_used": int(used),
        "trades_skipped": int(skipped),
        "trades_reduced": int(reduced),
        "avg_score": round(score_sum / score_count, 4) if score_count else None,
        "model_train_rows": int(model_bundle.get("train_rows") or 0),
        "model_used": bool(use_model),
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
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


def parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep prior-MT5-sequence pre-entry regime throttles.")
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--contract-value-per-lot", type=float, default=100.0)
    parser.add_argument("--min-train-rows", type=int, default=800)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--model-jobs", type=int, default=-1)
    parser.add_argument("--prob-weight", type=float, default=0.45)
    parser.add_argument("--min-scores", default="-0.2,-0.1,0,0.05,0.1,0.15,0.2")
    parser.add_argument("--reduce-scores", default="0.05,0.1,0.15,0.2,0.3")
    parser.add_argument("--reduce-mults", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--pre-dd-skip-pcts", default="8,10,12,14,16,18,20")
    args = parser.parse_args()

    feedback = pd.read_csv(args.feedback)
    for col in ["fold", "profit", "hour", "weekday", "direction", "signal_probability", "signal_atr", "rr"]:
        feedback[col] = pd.to_numeric(feedback.get(col), errors="coerce")
    for col in ["open_time", "close_time"]:
        feedback[col] = pd.to_datetime(feedback[col], errors="coerce")
    feedback = feedback.dropna(subset=["fold", "profit"]).copy()
    feedback["fold"] = feedback["fold"].astype(int)
    feedback = add_realized_r(feedback, args.contract_value_per_lot)
    training_rows = build_training_rows(feedback, args.deposit)
    model_cache: dict[int, dict[str, Any]] = {}
    for fold_id in sorted(feedback["fold"].dropna().astype(int).unique().tolist()):
        train = training_rows[training_rows["fold"] < fold_id].copy()
        bundle: dict[str, Any] = {
            "train_rows": int(len(train)),
            "model_used": False,
            "reg": None,
            "clf": None,
        }
        if len(train) >= int(args.min_train_rows) and train["label_positive"].nunique() == 2:
            reg, clf = make_models(
                seed=fold_id,
                n_estimators=args.n_estimators,
                min_samples_leaf=args.min_samples_leaf,
                n_jobs=args.model_jobs,
            )
            reg.fit(train[FEATURE_COLUMNS], train["realized_r"])
            clf.fit(train[FEATURE_COLUMNS], train["label_positive"])
            bundle.update({"model_used": True, "reg": reg, "clf": clf})
        model_cache[fold_id] = bundle

    records: list[dict[str, Any]] = []
    details: dict[str, list[dict[str, Any]]] = {}
    folds = sorted(feedback["fold"].dropna().astype(int).unique().tolist())
    for min_score in parse_floats(args.min_scores):
        for reduce_score in parse_floats(args.reduce_scores):
            if reduce_score < min_score:
                continue
            for reduce_mult in parse_floats(args.reduce_mults):
                for pre_dd_skip_pct in parse_floats(args.pre_dd_skip_pcts):
                    rows: list[dict[str, Any]] = []
                    for fold_id in folds:
                        group = feedback[feedback["fold"] == fold_id].copy()
                        rows.append(
                            simulate_fold(
                                group,
                                args=args,
                                model_bundle=model_cache[fold_id],
                                min_score=min_score,
                                reduce_score=reduce_score,
                                reduce_mult=reduce_mult,
                                pre_dd_skip_pct=pre_dd_skip_pct,
                            )
                        )
                    summary = summarize(rows, args)
                    key = f"ms{min_score:g}_rs{reduce_score:g}_rm{reduce_mult:g}_pdd{pre_dd_skip_pct:g}"
                    summary.update(
                        {
                            "key": key,
                            "min_score": float(min_score),
                            "reduce_score": float(reduce_score),
                            "reduce_mult": float(reduce_mult),
                            "pre_dd_skip_pct": float(pre_dd_skip_pct),
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
        "best": records[:20],
        "details": {record["key"]: details[record["key"]] for record in records[:10]},
        "feature_columns": FEATURE_COLUMNS,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best": records[:10]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
