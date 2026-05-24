from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_label(path: Path) -> str:
    return str(path).replace("\\", "/").replace("outputs/", "")


def load_candidates(paths: list[Path], target: float, max_dd: float, deposit: float) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for root in paths:
        for result_path in root.rglob("best_fold_results.csv"):
            if result_path.parent.name == "best_mt5":
                continue
            summary_path = result_path.parent / "best_summary.json"
            if not summary_path.exists():
                continue
            df = pd.read_csv(result_path)
            if df.empty or "fold" not in df:
                continue
            summary = read_json(summary_path)
            if "summary" in summary:
                summary = summary["summary"]
            df["candidate_dir"] = str(result_path.parent)
            df["candidate_label"] = candidate_label(result_path.parent)
            df["candidate_target_folds"] = int(summary.get("target_folds", 0))
            df["candidate_loss_folds"] = int(summary.get("loss_folds", 0))
            df["candidate_dd_fail_folds"] = int(summary.get("dd_fail_folds", 0))
            df["candidate_median_final"] = float(summary.get("median_final_balance", 0.0) or 0.0)
            parts.append(df)
    if not parts:
        raise ValueError("No candidate best_fold_results.csv files found.")
    frame = pd.concat(parts, ignore_index=True)
    for col in ["fold", "final_balance", "max_dd_pct", "trades", "wins", "losses"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["target_hit"] = frame["final_balance"] >= float(target)
    frame["dd_pass"] = frame["max_dd_pct"] >= -float(max_dd)
    frame["loss_fold"] = frame["final_balance"] < float(deposit)
    frame["strict_pass"] = frame["target_hit"] & frame["dd_pass"] & ~frame["loss_fold"]
    return frame.dropna(subset=["fold", "final_balance", "max_dd_pct"]).copy()


def score_history(hist: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, group in hist.groupby("candidate_label"):
        recent = group.sort_values("fold").tail(int(args.lookback_folds)) if int(args.lookback_folds) > 0 else group
        if len(recent) < int(args.min_history_folds):
            continue
        strict_rate = float(recent["strict_pass"].mean())
        target_rate = float(recent["target_hit"].mean())
        dd_rate = float(recent["dd_pass"].mean())
        loss_rate = float(recent["loss_fold"].mean())
        median_final = float(recent["final_balance"].median())
        worst_dd = float(recent["max_dd_pct"].min())
        score = (
            strict_rate * float(args.strict_weight)
            + target_rate * float(args.target_weight)
            + dd_rate * float(args.dd_weight)
            - loss_rate * float(args.loss_weight)
            + min(median_final, float(args.target_balance)) / float(args.target_balance) * float(args.median_weight)
            + worst_dd * float(args.dd_buffer_weight)
        )
        last = group.iloc[-1]
        rows.append(
            {
                "candidate_label": label,
                "score": score,
                "hist_folds": int(len(recent)),
                "hist_strict_rate": strict_rate,
                "hist_target_rate": target_rate,
                "hist_dd_rate": dd_rate,
                "hist_loss_rate": loss_rate,
                "hist_median_final": median_final,
                "hist_worst_dd": worst_dd,
                "candidate_dir": last["candidate_dir"],
            }
        )
    return pd.DataFrame(rows)


def choose_bootstrap(frame: pd.DataFrame, fold_id: int, args: argparse.Namespace) -> pd.Series:
    rows = frame[frame["fold"] == fold_id].copy()
    rows = rows[
        (rows["candidate_loss_folds"] <= int(args.max_candidate_loss_folds))
        & (rows["candidate_dd_fail_folds"] <= int(args.max_candidate_dd_fail_folds))
    ].copy()
    if rows.empty:
        rows = frame[frame["fold"] == fold_id].copy()
    rows = rows.sort_values(
        ["candidate_target_folds", "candidate_loss_folds", "candidate_dd_fail_folds", "candidate_median_final"],
        ascending=[False, True, True, False],
    )
    return rows.iloc[0]


def evaluate(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    max_fold = int(frame["fold"].max())
    for fold_id in range(int(args.fold_start), min(max_fold, int(args.fold_end)) + 1):
        current = frame[frame["fold"] == fold_id].copy()
        if current.empty:
            continue
        hist = frame[frame["fold"] < fold_id].copy()
        scores = score_history(hist, args)
        mode = "rolling_prior"
        if scores.empty:
            row = choose_bootstrap(frame, fold_id, args)
            mode = "bootstrap_global_proxy"
        else:
            scores = scores.sort_values(
                ["score", "hist_strict_rate", "hist_target_rate", "hist_dd_rate", "hist_loss_rate", "hist_median_final"],
                ascending=[False, False, False, False, True, False],
            )
            label = str(scores.iloc[0]["candidate_label"])
            row = current[current["candidate_label"] == label]
            if row.empty:
                row = choose_bootstrap(frame, fold_id, args)
                mode = "fallback_missing_current"
            else:
                row = row.iloc[0]
        rec = row.to_dict()
        rec["selection_mode"] = mode
        selected.append(rec)
    return pd.DataFrame(selected)


def summarize(rows: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    strict = (rows["final_balance"] >= float(args.target_balance)) & (rows["max_dd_pct"] >= -float(args.max_dd_pct)) & (rows["final_balance"] >= float(args.deposit))
    return {
        "folds": int(len(rows)),
        "target_folds": int((rows["final_balance"] >= float(args.target_balance)).sum()),
        "dd_pass_folds": int((rows["max_dd_pct"] >= -float(args.max_dd_pct)).sum()),
        "loss_folds": int((rows["final_balance"] < float(args.deposit)).sum()),
        "strict_pass_folds": int(strict.sum()),
        "min_final_balance": round(float(rows["final_balance"].min()), 2),
        "median_final_balance": round(float(rows["final_balance"].median()), 2),
        "worst_dd_pct": round(float(rows["max_dd_pct"].min()), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a clean rolling selector over proxy candidate fold results.")
    parser.add_argument("--candidate-root", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=999)
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--min-history-folds", type=int, default=8)
    parser.add_argument("--lookback-folds", type=int, default=30)
    parser.add_argument("--max-candidate-loss-folds", type=int, default=999)
    parser.add_argument("--max-candidate-dd-fail-folds", type=int, default=999)
    parser.add_argument("--strict-weight", type=float, default=2000.0)
    parser.add_argument("--target-weight", type=float, default=700.0)
    parser.add_argument("--dd-weight", type=float, default=500.0)
    parser.add_argument("--loss-weight", type=float, default=900.0)
    parser.add_argument("--median-weight", type=float, default=250.0)
    parser.add_argument("--dd-buffer-weight", type=float, default=5.0)
    args = parser.parse_args()

    frame = load_candidates(args.candidate_root, args.target_balance, args.max_dd_pct, args.deposit)
    selected = evaluate(frame, args)
    summary = summarize(selected, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out_dir / "rolling_proxy_selected_folds.csv", index=False)
    frame.to_csv(args.out_dir / "rolling_proxy_candidate_matrix.csv", index=False)
    (args.out_dir / "rolling_proxy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
