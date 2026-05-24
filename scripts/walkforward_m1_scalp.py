#!/usr/bin/env python3
"""
M1 Scalp Model — Fast signals for intraday trading
==================================================
- Input: M1 bars (1-minute candles)
- Features: same ICT/Wyckoff but on M1 timeframe
- Labels: 10-15 bar lookahead (TP/SL 2-3x ATR)
- Output: M1 signals with faster frequency

Requirements:
  - MT5 with M1 data loaded
  - Extended features dataset with M1 bars

Scalp parameters:
  TP_RR = 2.0  (2:1 risk-reward vs 3.5:1 for swing)
  SL_ATR_MULT = 1.5
  MIN_CONF = 0.68  (lower threshold for faster feedback)
  LOOKHEAD_BARS = 10  (10 M1 bars ≈ 10 minutes vs 30 M5 = 2.5 hours)
"""
import os, sys, warnings
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np, pandas as pd
from datetime import datetime, timezone

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import load_settings
from xauusd_ai.features.dataset import FEATURE_COLUMNS, get_label_lookahead_bars

print("=" * 80)
print("  M1 SCALP WALK-FORWARD VALIDATION")
print("=" * 80)
print()

# ─ Config ───────────────────────────────────────────────────────────────────
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--config", default="configs/xauusd_combo133_best.yaml")
_args = _parser.parse_args()

CFG = load_settings(str(_args.config))
print(f"Base config: {_args.config}")
print()

# M1 Scalp parameters (override base M5 config)
SCALP_TP_RR = 2.0  # 2:1 vs 3.5:1 for swing
SCALP_SL_ATR = 1.5  # Same SL distance
SCALP_MIN_CONF = 0.68  # Lower threshold for faster signals
SCALP_LOOKAHEAD_BARS = 10  # 10 M1 bars = 10 minutes

# WF parameters
TRAIN_BARS = 50_000  # ~35 days M1 data per fold (less than M5 due to more bars)
TEST_BARS = 5_000   # ~3.5 days test
STEP_BARS = 2_500   # Slide every 1.75 days

IN_CSV = Path("outputs/historical_features_2003_2026_M1.csv")  # Will need to create M1 features
if not IN_CSV.exists():
    print(f"ERROR: M1 features not found: {IN_CSV}")
    print(f"Run first: python scripts/build_historical_features_extended_m1.py")
    sys.exit(1)

print(f"Loading M1 features: {IN_CSV}")
print(f"  TRAIN_BARS={TRAIN_BARS:,} (~{TRAIN_BARS//1440:.0f} days M1)")
print(f"  Scalp params: TP={SCALP_TP_RR}x, SL={SCALP_SL_ATR}x ATR, MinConf={SCALP_MIN_CONF}")
print(f"  Lookahead: {SCALP_LOOKAHEAD_BARS} bars (={SCALP_LOOKAHEAD_BARS} minutes)")
print()

try:
    full_ds = pd.read_csv(IN_CSV)
    full_ds["time"] = pd.to_datetime(full_ds["time"])
    full_ds = full_ds.sort_values("time").reset_index(drop=True)
    print(f"Loaded: {len(full_ds):,} rows (M1)")
    print(f"  Date range: {full_ds['time'].min()} to {full_ds['time'].max()}")
except Exception as e:
    print(f"ERROR loading data: {e}")
    sys.exit(1)

# ─ Walk-forward ─────────────────────────────────────────────────────────────
print()
print("=" * 80)
print("  WALK-FORWARD VALIDATION (M1 SCALP)")
print("=" * 80)

all_folds = []
fold_idx = 0
train_start_idx = 0

while True:
    train_end_idx = train_start_idx + TRAIN_BARS
    test_end_idx = train_end_idx + TEST_BARS
    
    if test_end_idx > len(full_ds):
        break
    
    fold_idx += 1
    train_start = full_ds.iloc[train_start_idx]["time"]
    train_end = full_ds.iloc[train_end_idx]["time"]
    test_start = full_ds.iloc[train_end_idx]["time"]
    test_end = full_ds.iloc[test_end_idx]["time"]
    
    print(f"\n[Fold {fold_idx}] M1 Scalp")
    print(f"  Train: {train_start} → {train_end} ({TRAIN_BARS:,} M1 bars)")
    print(f"  Test:  {test_start} → {test_end} ({TEST_BARS:,} M1 bars)")
    
    # Data prep
    train_ds = full_ds.iloc[train_start_idx:train_end_idx].copy()
    test_ds = full_ds.iloc[train_end_idx:test_end_idx].copy()
    
    # Labels with SCALP parameters
    train_labels = get_label_lookahead_bars(
        train_ds["close"].values,
        train_ds.get("atr", pd.Series([1.0]*len(train_ds))).values,
        lookahead_bars=SCALP_LOOKAHEAD_BARS,
        tp_rr=SCALP_TP_RR,
        sl_atr_mult=SCALP_SL_ATR,
    )
    test_labels = get_label_lookahead_bars(
        test_ds["close"].values,
        test_ds.get("atr", pd.Series([1.0]*len(test_ds))).values,
        lookahead_bars=SCALP_LOOKAHEAD_BARS,
        tp_rr=SCALP_TP_RR,
        sl_atr_mult=SCALP_SL_ATR,
    )
    
    # Features & training
    feature_cols = FEATURE_COLUMNS
    X_train = train_ds[feature_cols].fillna(0).values
    X_test = test_ds[feature_cols].fillna(0).values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Ensemble
    clf_hgb = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08, random_state=42)
    clf_rf = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
    clf_et = ExtraTreesClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
    
    ensemble = VotingClassifier(
        estimators=[("hgb", clf_hgb), ("rf", clf_rf), ("et", clf_et)],
        voting="soft",
    )
    ensemble.fit(X_train, train_labels)
    
    # Eval
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]
    acc = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred, zero_division=0)
    auc = roc_auc_score(test_labels, y_prob)
    
    print(f"  Eval: Acc={acc:.1%} F1={f1:.1%} AUC={auc:.3f}")
    
    all_folds.append({
        "fold": fold_idx,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "acc": float(acc),
        "f1": float(f1),
        "auc": float(auc),
    })
    
    train_start_idx += STEP_BARS

# Summary
print()
print("=" * 80)
print("  M1 SCALP SUMMARY")
print("=" * 80)

if all_folds:
    avg_acc = np.mean([f["acc"] for f in all_folds])
    avg_f1 = np.mean([f["f1"] for f in all_folds])
    avg_auc = np.mean([f["auc"] for f in all_folds])
    
    print(f"Total folds: {len(all_folds)}")
    print(f"Avg Accuracy: {avg_acc:.1%}")
    print(f"Avg F1: {avg_f1:.1%}")
    print(f"Avg AUC: {avg_auc:.3f}")

# Save
out_json = Path("outputs/walkforward_m1_scalp_results.json")
with open(out_json, "w") as f:
    json.dump(all_folds, f, indent=2, default=str)
print(f"\nResults saved: {out_json}")
print("Done!")
