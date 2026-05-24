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

import scripts.mt5_dd_aware_trade_selector_sweep as ddsel
from scripts.mt5_dd_aware_trade_selector_sweep import FEATURE_COLUMNS
from scripts.mt5_prior_sequence_regime_throttle_sweep import max_drawdown_pct, parse_floats, state_features


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


def build_sequence_training_rows(
    feedback: pd.DataFrame,
    *,
    deposit: float,
    window_trades: int,
    bad_window_dd_pct: float,
    min_window_r: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fold_id, group in feedback.groupby("fold", sort=True):
        ordered = group.sort_values(["close_time", "open_time"]).reset_index(drop=True)
        balance = float(deposit)
        peak = balance
        recent_r: list[float] = []
        loss_streak = 0
        win_streak = 0
        for idx, row in ordered.iterrows():
            trade_index = idx + 1
            state = state_features(
                balance=balance,
                peak=peak,
                recent_r=recent_r,
                loss_streak=loss_streak,
                win_streak=win_streak,
                trade_index=trade_index,
            )
            future = ordered.iloc[idx : idx + int(window_trades)]
            future_profit = pd.to_numeric(future["profit"], errors="coerce").fillna(0.0)
            future_r = pd.to_numeric(future["realized_r"], errors="coerce").fillna(0.0)
            sum_r = float(future_r.sum())
            sum_profit = float(future_profit.sum())
            fwd_dd = window_dd_pct(future_profit.tolist(), balance)
            rec = ddsel.feature_row(row, state)
            rec.update(
                {
                    "fold": int(fold_id),
                    "profit": float(row["profit"]),
                    "realized_r": float(row["realized_r"]),
                    "future_sum_r": sum_r,
                    "future_sum_profit": sum_profit,
                    "future_window_dd_pct": fwd_dd,
                    "label_sequence_good": int(sum_r >= float(min_window_r) and fwd_dd >= -float(bad_window_dd_pct)),
                    "label_sequence_bad": int(sum_r < float(min_window_r) or fwd_dd < -float(bad_window_dd_pct)),
                }
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
        random_state=seed + 5000,
        n_jobs=int(args.model_jobs),
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", reg)]), Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("model", clf)]
    )


def fit_model_cache(training_rows: pd.DataFrame, folds: list[int], columns: list[str], args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    cache: dict[int, dict[str, Any]] = {}
    for fold_id in folds:
        train = training_rows[training_rows["fold"] < fold_id].copy()
        bundle: dict[str, Any] = {"model_used": False, "train_rows": int(len(train)), "reg": None, "clf": None}
        if len(train) >= int(args.min_train_rows) and train["label_sequence_bad"].nunique() == 2:
            reg, clf = make_models(args, fold_id)
            reg.fit(train[columns], train["future_sum_r"])
            clf.fit(train[columns], train["label_sequence_bad"])
            bundle.update({"model_used": True, "reg": reg, "clf": clf})
        cache[fold_id] = bundle
    return cache


def add_predictions(feedback: pd.DataFrame, model_cache: dict[int, dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold_id, group in feedback.groupby("fold", sort=True):
        bundle = model_cache[int(fold_id)]
        ordered = group.sort_values(["close_time", "open_time"]).copy()
        balance = 200.0
        peak = balance
        recent_r: list[float] = []
        loss_streak = 0
        win_streak = 0
        feature_rows: list[dict[str, float]] = []
        for trade_index, (_, row) in enumerate(ordered.iterrows(), start=1):
            state = state_features(
                balance=balance,
                peak=peak,
                recent_r=recent_r,
                loss_streak=loss_streak,
                win_streak=win_streak,
                trade_index=trade_index,
            )
            feature_rows.append(ddsel.feature_row(row, state))
            balance += float(row["profit"])
            peak = max(peak, balance)
            recent_r.append(float(row["realized_r"]))
            if float(row["profit"]) < 0:
                loss_streak += 1
                win_streak = 0
            elif float(row["profit"]) > 0:
                win_streak += 1
                loss_streak = 0
        if bundle.get("model_used"):
            x = pd.DataFrame(feature_rows, columns=columns)
            ordered["pred_future_r"] = bundle["reg"].predict(x)
            ordered["pred_bad_prob"] = bundle["clf"].predict_proba(x)[:, 1]
        else:
            ordered["pred_future_r"] = 0.0
            ordered["pred_bad_prob"] = 0.5
        ordered["model_used"] = bool(bundle.get("model_used"))
        ordered["model_train_rows"] = int(bundle.get("train_rows") or 0)
        frames.append(ordered)
    return pd.concat(frames, ignore_index=True)


def simulate_fold(
    group: pd.DataFrame,
    args: argparse.Namespace,
    *,
    min_future_r: float,
    max_bad_prob: float,
    reduce_future_r: float,
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
    used = skipped = reduced = 0
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
        pred_r = float(row.get("pred_future_r", 0.0))
        bad_prob = float(row.get("pred_bad_prob", 0.5))
        if pred_r < float(min_future_r) or bad_prob > float(max_bad_prob):
            skipped += 1
            continue
        mult = float(reduce_mult) if pred_r < float(reduce_future_r) or bad_prob > float(reduce_bad_prob) else 1.0
        profit = float(row["profit"]) * mult
        if mult < 0.999:
            reduced += 1
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
            loss_streak = 0
            win_streak += 1
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
        "model_used": bool(group["model_used"].any()),
        "model_train_rows": int(group["model_train_rows"].max()),
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
    parser = argparse.ArgumentParser(description="Sweep sequence-outcome selector using future N-trade labels.")
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
    parser.add_argument("--window-trades", type=int, default=6)
    parser.add_argument("--bad-window-dd-pct", type=float, default=10.0)
    parser.add_argument("--min-window-r", type=float, default=0.0)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--min-samples-leaf", type=int, default=12)
    parser.add_argument("--model-jobs", type=int, default=1)
    parser.add_argument("--min-future-rs", default="-1.0,-0.5,0,0.5")
    parser.add_argument("--max-bad-probs", default="0.65,0.75,0.85,1.0")
    parser.add_argument("--reduce-future-rs", default="0,0.5,1.0")
    parser.add_argument("--reduce-bad-probs", default="0.55,0.65,0.75")
    parser.add_argument("--reduce-mults", default="0.5,0.75,1.0")
    parser.add_argument("--pre-dd-skip-pcts", default="18,20")
    parser.add_argument("--loss-streak-skips", default="4,99")
    args = parser.parse_args()

    feedback = ddsel.add_cost_and_r(pd.read_csv(args.feedback), args)
    feedback, extras = ddsel.join_bar_features(feedback, args.features)
    ddsel.EXTRA_FEATURE_COLUMNS = extras
    columns = FEATURE_COLUMNS + extras
    training_rows = build_sequence_training_rows(
        feedback,
        deposit=args.deposit,
        window_trades=args.window_trades,
        bad_window_dd_pct=args.bad_window_dd_pct,
        min_window_r=args.min_window_r,
    )
    folds = sorted(feedback["fold"].unique().tolist())
    model_cache = fit_model_cache(training_rows, folds, columns, args)
    predicted = add_predictions(feedback, model_cache, columns)

    records: list[dict[str, Any]] = []
    details: dict[str, list[dict[str, Any]]] = {}
    for min_future_r in parse_floats(args.min_future_rs):
        for max_bad_prob in parse_floats(args.max_bad_probs):
            for reduce_future_r in parse_floats(args.reduce_future_rs):
                for reduce_bad_prob in parse_floats(args.reduce_bad_probs):
                    for reduce_mult in parse_floats(args.reduce_mults):
                        for pre_dd_skip_pct in parse_floats(args.pre_dd_skip_pcts):
                            for loss_streak_skip in parse_floats(args.loss_streak_skips):
                                rows = [
                                    simulate_fold(
                                        group.copy(),
                                        args,
                                        min_future_r=min_future_r,
                                        max_bad_prob=max_bad_prob,
                                        reduce_future_r=reduce_future_r,
                                        reduce_bad_prob=reduce_bad_prob,
                                        reduce_mult=reduce_mult,
                                        pre_dd_skip_pct=pre_dd_skip_pct,
                                        loss_streak_skip=loss_streak_skip,
                                    )
                                    for _, group in predicted.groupby("fold", sort=True)
                                ]
                                summary = summarize(rows)
                                key = (
                                    f"mfr{min_future_r:g}_mbp{max_bad_prob:g}_rfr{reduce_future_r:g}_"
                                    f"rbp{reduce_bad_prob:g}_rm{reduce_mult:g}_pdd{pre_dd_skip_pct:g}_"
                                    f"ls{loss_streak_skip:g}"
                                )
                                summary.update(
                                    {
                                        "key": key,
                                        "min_future_r": float(min_future_r),
                                        "max_bad_prob": float(max_bad_prob),
                                        "reduce_future_r": float(reduce_future_r),
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
        "extra_features": extras,
        "window_trades": int(args.window_trades),
        "bad_window_dd_pct": float(args.bad_window_dd_pct),
        "min_window_r": float(args.min_window_r),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best": records[:10]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
