#!/usr/bin/env python3
"""
Export model signals to CSV for MT5 Strategy Tester (Option A).

Output CSV format (MT5-readable):
  open_time,direction,entry_price,sl_price,tp_price,atr,probability

where:
  open_time   = bar open time  "YYYY.MM.DD HH:MM"  (UTC, M15 bar)
  direction   = 1 (BUY) or -1 (SELL)
  entry_price = bar close price (signal fires at bar close → next bar open)
  sl_price    = computed from ATR × SL_MULT
  tp_price    = computed from ATR × SL_MULT × TP_RR
  atr         = ATR at signal bar
  probability = model output probability

Usage:
  python scripts/export_signals_mt5.py --from 2024-12-01 --to 2026-05-01

Config defaults match grid_0014:
  --threshold 0.70   signal threshold
  --sl-mult   1.5    SL = ATR × sl_mult
  --tp-rr     5.5    TP = SL × tp_rr
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Export LightGBM signals to MT5 CSV")
    p.add_argument("--from",   dest="date_from", default="2024-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--to",     dest="date_to",   default="2026-05-01", help="End date YYYY-MM-DD")
    p.add_argument("--threshold", type=float, default=0.70,  help="Signal probability threshold (grid_0014=0.70)")
    p.add_argument("--sl-mult",   type=float, default=1.5,   help="SL = ATR × sl_mult")
    p.add_argument("--tp-rr",     type=float, default=5.5,   help="TP = SL × tp_rr")
    p.add_argument("--broker-gmt", type=int,  default=3,     help="Broker GMT offset (EXNESS=3, default=3)")
    p.add_argument("--out",  default="outputs/signals_for_mt5.csv", help="Output CSV path")
    p.add_argument("--model",   default="outputs/acc1_combo133_202604_model.pkl")
    p.add_argument("--scaler",  default="outputs/acc1_combo133_202604_scaler.pkl")
    p.add_argument("--meta",    default="outputs/acc1_combo133_202604_meta.json")
    p.add_argument("--dataset", default=None, help="Feature dataset CSV (auto-detect if not set)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  Export LightGBM signals → MT5 CSV")
    print(f"  Period   : {args.date_from} → {args.date_to}")
    print(f"  Threshold: {args.threshold}")
    print(f"  SL mult  : {args.sl_mult}xATR   |  TP R:R : {args.tp_rr}")
    print(f"  Broker GMT: +{args.broker_gmt} (timestamps shifted from UTC)")
    print("=" * 70)

    # ── Auto-detect dataset ────────────────────────────────────────────
    dataset_path = args.dataset
    if dataset_path is None:
        candidates = [
            ROOT / "outputs" / "historical_features_2022_2026.csv",
            ROOT / "outputs" / "task4_training_dataset_acc1.csv",
            ROOT / "outputs" / "training_dataset.csv",
        ]
        for c in candidates:
            if c.exists():
                dataset_path = str(c)
                print(f"  Auto-selected dataset: {c.name}")
                break
        if dataset_path is None:
            print("ERROR: No feature dataset found. Run build_historical_features.py first.")
            sys.exit(1)

    # ── Load model artifacts ───────────────────────────────────────────────
    print("[1/4] Loading model artifacts...")
    with open(ROOT / args.model, "rb") as f:
        model = pickle.load(f)
    with open(ROOT / args.scaler, "rb") as f:
        scaler = pickle.load(f)
    with open(ROOT / args.meta) as f:
        meta = json.load(f)

    feature_columns = meta["feature_columns"]   # 65 names
    feature_mask    = meta["feature_mask"]       # 65 bools

    # Active features (model was trained on these)
    active_features = [col for col, keep in zip(feature_columns, feature_mask) if keep]
    print(f"   Active features: {len(active_features)}/{len(feature_columns)}")

    # ── Load dataset ───────────────────────────────────────────────────────
    print("[2/4] Loading dataset...")
    df = pd.read_csv(dataset_path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # Filter date range
    date_from = pd.Timestamp(args.date_from, tz="UTC")
    date_to   = pd.Timestamp(args.date_to,   tz="UTC")
    df = df[(df["time"] >= date_from) & (df["time"] < date_to)].copy()
    print(f"   Rows in range: {len(df):,} bars")

    if len(df) == 0:
        print("ERROR: No data in range. Check --from/--to dates.")
        sys.exit(1)

    # ── Build feature matrix ───────────────────────────────────────────────
    print("[3/4] Building features & running inference...")

    # Scaler expects ALL 65 features (in scaler.feature_names_in_ order)
    all_features = list(scaler.feature_names_in_)  # 65 features

    # Fill missing features with 0 (mostly inactive ones: h4_breaker_block etc.)
    missing_all = [c for c in all_features if c not in df.columns]
    if missing_all:
        print(f"   Missing features (will use 0): {missing_all}")
        for c in missing_all:
            df[c] = 0.0

    X_all = df[all_features].fillna(0.0)

    # Step 1: Scale all 65 features
    X_scaled_all = scaler.transform(X_all)  # shape (N, 65)

    # Step 2: Apply feature_mask to select active features for model
    mask = np.array(feature_mask, dtype=bool)  # 65 bools
    X_scaled = X_scaled_all[:, mask]           # shape (N, 45)

    # Predict probabilities (class 1 = profitable signal)
    probs = model.predict_proba(X_scaled)[:, 1]
    df["probability"] = probs

    # ── Apply threshold filter ─────────────────────────────────────────────
    signals = df[df["probability"] >= args.threshold].copy()
    print(f"   Total signals ≥ {args.threshold}: {len(signals):,} out of {len(df):,} bars ({100*len(signals)/len(df):.1f}%)")

    # ── Calculate SL / TP ─────────────────────────────────────────────────
    # trade_side: "buy" or "sell" (from dataset — direction the model expects)
    # SL distance = ATR × sl_mult
    # For BUY:  SL = close - sl_dist,  TP = close + sl_dist × tp_rr
    # For SELL: SL = close + sl_dist,  TP = close - sl_dist × tp_rr

    signals = signals.copy()
    signals["sl_dist"] = signals["atr"] * args.sl_mult

    # Map direction
    signals["direction"] = signals["trade_side"].map({"buy": 1, "sell": -1}).fillna(1).astype(int)

    # entry price = bar close (signal fires at bar close, EA enters at next bar open — close ≈ open)
    signals["entry_price"] = signals["close"]

    signals["sl_price"] = np.where(
        signals["direction"] == 1,
        signals["close"] - signals["sl_dist"],
        signals["close"] + signals["sl_dist"],
    )
    signals["tp_price"] = np.where(
        signals["direction"] == 1,
        signals["close"] + signals["sl_dist"] * args.tp_rr,
        signals["close"] - signals["sl_dist"] * args.tp_rr,
    )

    # Format time for MT5: broker time = UTC + broker_gmt hours
    # MT5 EXNESS uses GMT+3 (Moscow time) — must match iTime() in EA
    from datetime import timedelta
    signals["open_time"] = (signals["time"] + timedelta(hours=args.broker_gmt)).dt.strftime("%Y.%m.%d %H:%M")

    # ── Export CSV ─────────────────────────────────────────────────────────
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_cols = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]
    signals[out_cols].to_csv(out_path, index=False, float_format="%.5f")

    print(f"[4/4] Exported {len(signals):,} signals → {out_path}")
    print()

    # ── Summary by month ───────────────────────────────────────────────────
    signals["month"] = signals["time"].dt.to_period("M")
    summary = signals.groupby("month").agg(
        signals_n   = ("probability", "count"),
        buy_n       = ("direction",   lambda x: (x == 1).sum()),
        sell_n      = ("direction",   lambda x: (x == -1).sum()),
        avg_prob    = ("probability", "mean"),
    )
    print("  Monthly signal breakdown:")
    print(f"  {'Month':<10} {'Signals':>8} {'BUY':>6} {'SELL':>6} {'AvgProb':>9}")
    print("  " + "-" * 45)
    for month, row in summary.iterrows():
        print(f"  {str(month):<10} {row['signals_n']:>8} {row['buy_n']:>6} {row['sell_n']:>6} {row['avg_prob']:>8.3f}")
    print()
    print("  ✅ CSV ready for MT5 Strategy Tester.")
    print(f"     Copy to MT5: MQL5\\Files\\signals_for_mt5.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
