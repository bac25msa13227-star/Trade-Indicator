#!/usr/bin/env python3
"""
Quick diagnostic: analyze what's limiting avg_daily_net.
Answers:
1. What is realized_rr distribution in fold data?
2. What is actual win rate at different thresholds?
3. How many signals/day at different confidence levels?
4. What is theoretical max daily PnL with $200 balance?
"""
from __future__ import annotations
import os, sys, pickle, warnings
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

def main():
    cache_path = ROOT / "outputs/.wf_cache/acc1_v14pp_profit_dailyreset_foldpreds_exact_w30000_4000_4000_20240814_all.pkl"
    print(f"Loading fold cache: {cache_path.name}")
    with open(cache_path, "rb") as f:
        folds = pickle.load(f)

    # Concat all fold test frames
    frames = [fi["frame"] for fi in folds]
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    print(f"Total test bars: {len(df):,} | folds: {len(folds)}")
    print(f"Date range: {df['time'].min().date()} → {df['time'].max().date()}")
    print()

    # 1. Signal counts per fold threshold
    print("=== 1. SIGNAL FREQUENCY vs PROBABILITY THRESHOLD ===")
    base_thrs = [fi["threshold"] for fi in folds]
    print(f"Fold thresholds: min={min(base_thrs):.3f} max={max(base_thrs):.3f} median={np.median(base_thrs):.3f}")
    print()
    print(f"{'Threshold':>10} {'N signals':>10} {'Pct bars':>10} {'Precision':>11} {'Recall':>8}")
    print("-" * 55)
    for thr in [0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.83, 0.85, 0.88, 0.90]:
        sigs = df[df["probability"] >= thr]
        n = len(sigs)
        pct = 100*n/len(df)
        if "target" in df.columns and n > 0:
            prec = sigs["target"].mean()
            rec  = n / df["target"].sum() if df["target"].sum() > 0 else 0
        else:
            prec = rec = float("nan")
        print(f"{thr:>10.2f} {n:>10,} {pct:>9.1f}% {prec:>10.1%} {rec:>8.1%}")
    print()

    # 2. realized_rr distribution
    print("=== 2. REALIZED RR DISTRIBUTION (all predicted signals at various thresholds) ===")
    if "realized_rr" not in df.columns:
        print("  realized_rr column not found in fold frames")
    else:
        for thr in [0.70, 0.76, 0.80, 0.83, 0.85, 0.88]:
            sigs = df[df["probability"] >= thr]
            if len(sigs) == 0:
                continue
            wins = sigs[sigs["realized_rr"] > 0]
            losses = sigs[sigs["realized_rr"] <= 0]
            avg_rr = sigs["realized_rr"].mean()
            win_rr = wins["realized_rr"].mean() if len(wins) > 0 else 0
            loss_rr = losses["realized_rr"].mean() if len(losses) > 0 else 0
            wr = len(wins)/len(sigs) if len(sigs)>0 else 0
            print(f"  thr={thr:.2f}: N={len(sigs):5,} WR={wr:.1%} avg_RR={avg_rr:.3f} win_RR={win_rr:.3f} loss_RR={loss_rr:.3f}")
    print()

    # 3. Daily signal frequency
    print("=== 3. DAILY SIGNALS at different thresholds ===")
    df["date"] = df["time"].dt.floor("D")
    for thr in [0.70, 0.74, 0.76, 0.78, 0.80, 0.83, 0.85]:
        sig = df[df["probability"] >= thr]
        daily_counts = sig.groupby("date").size()
        all_days = df.groupby("date").size().index
        full_daily = daily_counts.reindex(all_days, fill_value=0)
        pct_nonzero = (full_daily > 0).mean()
        print(f"  thr={thr:.2f}: {pct_nonzero:.0%} days have signal | avg={full_daily[full_daily>0].mean():.1f} sigs/day (on signal days) | total median={full_daily.median():.0f}")
    print()

    # 4. Theoretical max with $200 balance
    print("=== 4. THEORETICAL MAX DAILY PnL @ $200 balance ===")
    print("  Formula: avg_sigs_per_day × risk × realized_rr × $200")
    print("  (assuming all signals taken, 2 positions allowed)")
    if "realized_rr" in df.columns:
        for thr in [0.70, 0.74, 0.76, 0.78, 0.80, 0.83, 0.85]:
            sig = df[df["probability"] >= thr]
            if len(sig) == 0:
                continue
            daily_counts = sig.groupby(df.loc[sig.index, "date"]).size()
            all_days = df.groupby("date").size().index
            full_daily = daily_counts.reindex(all_days, fill_value=0)
            avg_sigs = full_daily.mean()
            avg_rr_val = sig["realized_rr"].mean()
            for risk in [0.10, 0.12, 0.15]:
                for max_pos in [2, 3]:
                    eff_sigs = min(avg_sigs, max_pos)
                    theo = eff_sigs * risk * avg_rr_val * 200
                    print(f"  thr={thr:.2f} risk={risk:.0%} maxpos={max_pos}: {avg_sigs:.2f} sigs/day → eff={eff_sigs:.2f} → theo=${theo:.2f}/day (avg_rr={avg_rr_val:.3f})")
        print()

    # 5. What does the filter chain reduce signals to?
    print("=== 5. SIGNALS AFTER PREDICTION FILTER (prediction==1) ===")
    pred_signals = df[df["prediction"] == 1]
    print(f"  Total predicted positive: {len(pred_signals):,} / {len(df):,} ({100*len(pred_signals)/len(df):.1f}%)")
    if "realized_rr" in df.columns:
        wins_p = pred_signals[pred_signals["realized_rr"] > 0]
        print(f"  WR of predicted positives: {len(wins_p)/max(1,len(pred_signals)):.1%}")
        print(f"  Avg realized_rr of predicted positives: {pred_signals['realized_rr'].mean():.4f}")
    daily_preds = pred_signals.groupby(df.loc[pred_signals.index, "date"]).size()
    all_days_idx = df.groupby("date").size().index
    full_preds = daily_preds.reindex(all_days_idx, fill_value=0)
    print(f"  Avg predicted sigs/day: {full_preds.mean():.2f}")
    print(f"  Pct days with any predicted signal: {(full_preds > 0).mean():.1%}")
    print(f"  Median predicted sigs/day: {full_preds.median():.1f}")

if __name__ == "__main__":
    main()
