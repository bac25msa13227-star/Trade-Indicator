from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd


def _replay_fold(trades: pd.DataFrame, deposit: float, low_mult: float, mid_mult: float, high_mult: float) -> dict[str, Any]:
    balance = float(deposit)
    peak = float(deposit)
    worst_dd = 0.0
    active = 0
    wins = 0
    losses = 0
    for _, row in trades.iterrows():
        prob = row.get("matched_probability")
        if pd.isna(prob):
            mult = 1.0
        elif float(prob) < 0.55:
            mult = low_mult
        elif float(prob) < 0.70:
            mult = mid_mult
        else:
            mult = high_mult
        if mult <= 0:
            continue
        pnl = float(row["pnl"]) * float(mult)
        active += 1
        wins += int(pnl > 0)
        losses += int(pnl < 0)
        balance += pnl
        peak = max(peak, balance)
        dd = (balance - peak) / peak * 100.0 if peak > 0 else 0.0
        worst_dd = min(worst_dd, dd)
    return {
        "final": round(balance, 2),
        "net_pct": round((balance / deposit - 1.0) * 100.0, 2),
        "worst_dd_pct": round(worst_dd, 2),
        "active_trades": active,
        "wr_pct": round(wins / active * 100.0, 2) if active else 0.0,
        "loss_fold": balance < deposit,
    }


def _score(frame: pd.DataFrame, deposit: float) -> float:
    median_final = float(frame["final"].median())
    loss_folds = int((frame["final"] < deposit).sum())
    worst_dd = abs(float(frame["worst_dd_pct"].min()))
    # Prefer robust profit; penalize blow-up behavior.
    return median_final - 250.0 * loss_folds - 8.0 * worst_dd


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep confidence-bucket risk multipliers on joined MT5 deal replay.")
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=1200.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(args.trades)
    trades["fold"] = pd.to_numeric(trades["fold"], errors="coerce").astype(int)
    folds = sorted(trades["fold"].unique())

    low_grid = [0.0, 0.5, 0.75, 1.0, 1.15, 1.3]
    mid_grid = [0.6, 0.8, 1.0, 1.1, 1.25, 1.4]
    high_grid = [0.8, 1.0, 1.15, 1.3, 1.5]

    all_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for low, mid, high in product(low_grid, mid_grid, high_grid):
        per_fold = []
        for fold in folds:
            result = _replay_fold(trades[trades["fold"] == fold], args.deposit, low, mid, high)
            result.update({"fold": int(fold), "low_mult": low, "mid_mult": mid, "high_mult": high})
            per_fold.append(result)
            fold_rows.append(result)
        frame = pd.DataFrame(per_fold)
        all_rows.append(
            {
                "low_mult": low,
                "mid_mult": mid,
                "high_mult": high,
                "score": round(_score(frame, args.deposit), 4),
                "median_final": round(float(frame["final"].median()), 2),
                "min_final": round(float(frame["final"].min()), 2),
                "mean_final": round(float(frame["final"].mean()), 2),
                "worst_dd_pct": round(float(frame["worst_dd_pct"].min()), 2),
                "loss_folds": int((frame["final"] < args.deposit).sum()),
                "total_active_trades": int(frame["active_trades"].sum()),
                "median_wr_pct": round(float(frame["wr_pct"].median()), 2),
            }
        )

    sweep = pd.DataFrame(all_rows).sort_values(["score", "median_final"], ascending=[False, False])
    sweep.to_csv(args.out_dir / "rating_multiplier_sweep.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out_dir / "rating_multiplier_sweep_by_fold.csv", index=False)

    # Walk-forward selector: tune only on previous folds, apply selected multipliers to next fold.
    wf_rows: list[dict[str, Any]] = []
    min_history = 12
    for idx, fold in enumerate(folds):
        if idx < min_history:
            selected = {"low_mult": 1.0, "mid_mult": 1.0, "high_mult": 1.0, "source": "bootstrap_baseline"}
        else:
            hist_folds = set(folds[:idx])
            candidates = []
            for row in all_rows:
                hist = pd.DataFrame(
                    [
                        r
                        for r in fold_rows
                        if r["fold"] in hist_folds
                        and r["low_mult"] == row["low_mult"]
                        and r["mid_mult"] == row["mid_mult"]
                        and r["high_mult"] == row["high_mult"]
                    ]
                )
                candidates.append({**row, "hist_score": _score(hist, args.deposit)})
            best = sorted(candidates, key=lambda item: (item["hist_score"], item["median_final"]), reverse=True)[0]
            selected = {
                "low_mult": best["low_mult"],
                "mid_mult": best["mid_mult"],
                "high_mult": best["high_mult"],
                "source": "prior_fold_selector",
            }
        result = _replay_fold(
            trades[trades["fold"] == fold],
            args.deposit,
            float(selected["low_mult"]),
            float(selected["mid_mult"]),
            float(selected["high_mult"]),
        )
        wf_rows.append({"fold": int(fold), **selected, **result})

    wf = pd.DataFrame(wf_rows)
    wf.to_csv(args.out_dir / "rating_multiplier_rolling_selector.csv", index=False)
    report = {
        "folds": int(len(folds)),
        "best_oracle": sweep.iloc[0].to_dict() if not sweep.empty else {},
        "baseline": sweep[
            (sweep["low_mult"] == 1.0) & (sweep["mid_mult"] == 1.0) & (sweep["high_mult"] == 1.0)
        ].iloc[0].to_dict(),
        "implemented_rating": sweep[
            (sweep["low_mult"] == 0.0) & (sweep["mid_mult"] == 0.6) & (sweep["high_mult"] == 1.0)
        ].iloc[0].to_dict(),
        "rolling_selector": {
            "median_final": round(float(wf["final"].median()), 2),
            "min_final": round(float(wf["final"].min()), 2),
            "mean_final": round(float(wf["final"].mean()), 2),
            "worst_dd_pct": round(float(wf["worst_dd_pct"].min()), 2),
            "loss_folds": int((wf["final"] < args.deposit).sum()),
            "total_active_trades": int(wf["active_trades"].sum()),
            "median_wr_pct": round(float(wf["wr_pct"].median()), 2),
        },
    }
    (args.out_dir / "rating_multiplier_sweep_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
