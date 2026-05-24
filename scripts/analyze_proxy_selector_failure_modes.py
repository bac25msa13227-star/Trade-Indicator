from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_proxy_candidate_rolling_selector import load_candidates


def summarize(rows: pd.DataFrame, deposit: float, target: float, max_dd: float) -> dict[str, Any]:
    strict = (rows["final_balance"] >= target) & (rows["max_dd_pct"] >= -max_dd) & (rows["final_balance"] >= deposit)
    return {
        "folds": int(len(rows)),
        "target_folds": int((rows["final_balance"] >= target).sum()),
        "dd_pass_folds": int((rows["max_dd_pct"] >= -max_dd).sum()),
        "loss_folds": int((rows["final_balance"] < deposit).sum()),
        "strict_pass_folds": int(strict.sum()),
        "min_final_balance": round(float(rows["final_balance"].min()), 2),
        "median_final_balance": round(float(rows["final_balance"].median()), 2),
        "mean_final_balance": round(float(rows["final_balance"].mean()), 2),
        "best_final_balance": round(float(rows["final_balance"].max()), 2),
        "worst_dd_pct": round(float(rows["max_dd_pct"].min()), 2),
    }


def strict_mask(df: pd.DataFrame, deposit: float, target: float, max_dd: float) -> pd.Series:
    return (df["final_balance"] >= target) & (df["max_dd_pct"] >= -max_dd) & (df["final_balance"] >= deposit)


def random_baseline(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(int(args.seed))
    labels = sorted(frame["candidate_label"].dropna().unique().tolist())
    fold_ids = sorted(frame["fold"].dropna().astype(int).unique().tolist())
    rows: list[dict[str, Any]] = []
    for run in range(int(args.random_runs)):
        picks = []
        for fold_id in fold_ids:
            current = frame[frame["fold"] == fold_id]
            label = labels[int(rng.integers(0, len(labels)))]
            row = current[current["candidate_label"] == label]
            if row.empty:
                row = current.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
            picks.append(row.iloc[0])
        summary = summarize(pd.DataFrame(picks), args.deposit, args.target_balance, args.max_dd_pct)
        rows.append({"run": run, **summary})
    return pd.DataFrame(rows)


def oracle_baseline(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    picks = []
    for _, group in frame.groupby("fold"):
        strict = group[strict_mask(group, args.deposit, args.target_balance, args.max_dd_pct)].copy()
        pool = strict if not strict.empty else group
        picks.append(pool.sort_values(["final_balance", "max_dd_pct"], ascending=[False, False]).iloc[0])
    rows = pd.DataFrame(picks)
    return rows, summarize(rows, args.deposit, args.target_balance, args.max_dd_pct)


def train_test_global_best(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    train = frame[(frame["fold"] >= int(args.train_start)) & (frame["fold"] <= int(args.train_end))].copy()
    test = frame[(frame["fold"] >= int(args.test_start)) & (frame["fold"] <= int(args.test_end))].copy()
    scored = []
    for label, group in train.groupby("candidate_label"):
        if len(group) < int(args.min_train_folds):
            continue
        s = summarize(group, args.deposit, args.target_balance, args.max_dd_pct)
        score = (
            s["dd_pass_folds"] * float(args.survival_dd_weight)
            - s["loss_folds"] * float(args.survival_loss_weight)
            + s["strict_pass_folds"] * float(args.survival_strict_weight)
            + min(float(s["median_final_balance"]), args.target_balance) / args.target_balance * float(args.survival_median_weight)
            + float(s["worst_dd_pct"]) * float(args.survival_dd_buffer_weight)
        )
        scored.append({"candidate_label": label, "score": score, **{f"train_{k}": v for k, v in s.items()}})
    score_frame = pd.DataFrame(scored).sort_values("score", ascending=False)
    if score_frame.empty:
        raise ValueError("No train candidates.")
    selected = score_frame.iloc[0]["candidate_label"]
    test_rows = test[test["candidate_label"] == selected].copy()
    result = summarize(test_rows, args.deposit, args.target_balance, args.max_dd_pct)
    return pd.DataFrame([{**score_frame.iloc[0].to_dict(), **{f"test_{k}": v for k, v in result.items()}}])


def rolling_survival_selector(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    picks = []
    for fold_id in sorted(frame["fold"].dropna().astype(int).unique()):
        current = frame[frame["fold"] == fold_id].copy()
        hist = frame[frame["fold"] < fold_id].copy()
        start = 1 if int(args.lookback_folds) <= 0 else max(1, fold_id - int(args.lookback_folds))
        hist = hist[hist["fold"] >= start].copy()
        scored = []
        for label, group in hist.groupby("candidate_label"):
            if len(group) < int(args.min_history_folds):
                continue
            s = summarize(group, args.deposit, args.target_balance, args.max_dd_pct)
            score = (
                s["dd_pass_folds"] * float(args.survival_dd_weight)
                - s["loss_folds"] * float(args.survival_loss_weight)
                + s["strict_pass_folds"] * float(args.survival_strict_weight)
                + min(float(s["median_final_balance"]), args.target_balance) / args.target_balance * float(args.survival_median_weight)
                + float(s["worst_dd_pct"]) * float(args.survival_dd_buffer_weight)
            )
            scored.append({"candidate_label": label, "score": score})
        if not scored:
            row = current.sort_values(["candidate_loss_folds", "candidate_dd_fail_folds", "candidate_target_folds"], ascending=[True, True, False]).iloc[0]
            mode = "bootstrap_global_survival"
        else:
            best = pd.DataFrame(scored).sort_values("score", ascending=False).iloc[0]
            row = current[current["candidate_label"] == best["candidate_label"]]
            if row.empty:
                row = current.sort_values(["candidate_loss_folds", "candidate_dd_fail_folds", "candidate_target_folds"], ascending=[True, True, False]).iloc[0]
                mode = "fallback_missing"
            else:
                row = row.iloc[0]
                mode = "rolling_survival"
        rec = row.to_dict()
        rec["selection_mode"] = mode
        picks.append(rec)
    selected = pd.DataFrame(picks)
    return selected, summarize(selected, args.deposit, args.target_balance, args.max_dd_pct)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze proxy selector failure modes and multiple-comparisons baselines.")
    parser.add_argument("--candidate-root", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--random-runs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--train-start", type=int, default=1)
    parser.add_argument("--train-end", type=int, default=120)
    parser.add_argument("--test-start", type=int, default=121)
    parser.add_argument("--test-end", type=int, default=180)
    parser.add_argument("--min-train-folds", type=int, default=100)
    parser.add_argument("--min-history-folds", type=int, default=8)
    parser.add_argument("--lookback-folds", type=int, default=80)
    parser.add_argument("--survival-dd-weight", type=float, default=2500.0)
    parser.add_argument("--survival-loss-weight", type=float, default=3500.0)
    parser.add_argument("--survival-strict-weight", type=float, default=400.0)
    parser.add_argument("--survival-median-weight", type=float, default=100.0)
    parser.add_argument("--survival-dd-buffer-weight", type=float, default=25.0)
    args = parser.parse_args()

    frame = load_candidates(args.candidate_root, args.target_balance, args.max_dd_pct, args.deposit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "candidate_matrix.csv", index=False)

    random_runs = random_baseline(frame, args)
    oracle_rows, oracle_summary = oracle_baseline(frame, args)
    split = train_test_global_best(frame, args)
    rolling_rows, rolling_summary = rolling_survival_selector(frame, args)

    random_summary = {
        "runs": int(len(random_runs)),
        "strict_mean": round(float(random_runs["strict_pass_folds"].mean()), 2),
        "strict_p05": round(float(random_runs["strict_pass_folds"].quantile(0.05)), 2),
        "strict_p50": round(float(random_runs["strict_pass_folds"].quantile(0.50)), 2),
        "strict_p95": round(float(random_runs["strict_pass_folds"].quantile(0.95)), 2),
        "target_mean": round(float(random_runs["target_folds"].mean()), 2),
        "dd_pass_mean": round(float(random_runs["dd_pass_folds"].mean()), 2),
        "loss_mean": round(float(random_runs["loss_folds"].mean()), 2),
    }
    report = {
        "candidate_rows": int(len(frame)),
        "candidate_count": int(frame["candidate_label"].nunique()),
        "fold_count": int(frame["fold"].nunique()),
        "random_baseline": random_summary,
        "oracle_baseline": oracle_summary,
        "train_1_120_global_best_test_121_180": split.iloc[0].to_dict(),
        "rolling_survival_selector": rolling_summary,
    }

    random_runs.to_csv(args.out_dir / "random_selector_runs.csv", index=False)
    oracle_rows.to_csv(args.out_dir / "oracle_selected_folds.csv", index=False)
    split.to_csv(args.out_dir / "train120_test60_global_best.csv", index=False)
    rolling_rows.to_csv(args.out_dir / "rolling_survival_selected_folds.csv", index=False)
    (args.out_dir / "failure_mode_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
