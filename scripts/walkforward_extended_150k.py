#!/usr/bin/env python3
"""
Extended Walk-Forward Validation — using 150k TRAIN_BARS (500+ days history)
=====================================================================
Retrain model with much longer training window for better generalization.

Key parameters:
  TRAIN_BARS = 150,000  (~520 days, ~2 years of training per fold)
  TEST_BARS = 6,000     (~20 days test)
  STEP_BARS = 6,000     (slide 20 days at a time)

This requires extended_features_2003_2026.csv (20+ years of data)
Run first: python scripts/build_historical_features_extended.py

Usage:
  python scripts/walkforward_extended_150k.py --config configs/xauusd_combo133_best.yaml
"""
import os, sys, warnings
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json, time
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
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.dataset import FEATURE_COLUMNS, get_label_lookahead_bars, prepare_training_dataset
from xauusd_ai.infra.advanced_metrics import calculate_all_metrics

# Logging setup
_cfg_stem = Path(sys.argv[1]).stem if len(sys.argv) > 1 else "extended_150k"
_log_path = Path(f"outputs/walkforward_extended_150k_{_cfg_stem}.txt")
_log_path.parent.mkdir(parents=True, exist_ok=True)
_log_file = _log_path.open("w", encoding="utf-8")

_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)
    _orig_print(*args, file=_log_file, **{k: v for k, v in kwargs.items() if k != "file"})

print("=" * 80)
print("  EXTENDED WALK-FORWARD — 150k TRAIN_BARS (~500+ days/2 years)")
print("=" * 80)
print()

# ─ Config ───────────────────────────────────────────────────────────────────
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--config", default="configs/xauusd_combo133_best.yaml")
_parser.add_argument("--max-folds", type=int, default=None)
_parser.add_argument("--fast", action="store_true")
_parser.add_argument("--no-rr-sweep", action="store_true")
_parser.add_argument("--no-compound", action="store_true")
_args = _parser.parse_args()

if not _args.config:
    print("ERROR: --config required")
    sys.exit(1)

_cfg_path = Path(_args.config)
if not _cfg_path.exists():
    print(f"ERROR: Config not found: {_cfg_path}")
    sys.exit(1)

CFG = load_settings(str(_cfg_path))
print(f"Config: {_cfg_path.name}")
print(f"  TakeProfit RR: {CFG.models.trading.take_profit_rr}")
print(f"  StopLoss ATR: {CFG.models.trading.stop_loss_atr_multiple}x")
print(f"  Min confidence: {CFG.models.trading.min_confidence}")
print()

# ─ Data parameters ──────────────────────────────────────────────────────────
TRAIN_BARS = 150_000  # ~520 days M5 (vs 30k previously = 104 days)
TEST_BARS = 6_000     # ~20 days M5
STEP_BARS = 6_000     # Slide every 20 days (same as TEST_BARS = no overlap)

IN_CSV = Path("outputs/historical_features_2003_2026.csv")
if not IN_CSV.exists():
    print(f"ERROR: Extended data not found: {IN_CSV}")
    print(f"Run first: python scripts/build_historical_features_extended.py")
    sys.exit(1)

print(f"Loading extended features: {IN_CSV}")
print(f"  TRAIN_BARS={TRAIN_BARS:,} (~{TRAIN_BARS//288} days)")
print(f"  TEST_BARS={TEST_BARS:,} (~{TEST_BARS//288} days)")
print(f"  STEP_BARS={STEP_BARS:,} (~{STEP_BARS//288} days)")
print()

try:
    full_ds = pd.read_csv(IN_CSV)
    full_ds["time"] = pd.to_datetime(full_ds["time"])
    full_ds = full_ds.sort_values("time").reset_index(drop=True)
    print(f"Loaded: {len(full_ds):,} rows")
    print(f"  Date range: {full_ds['time'].min()} to {full_ds['time'].max()}")
except Exception as e:
    print(f"ERROR loading data: {e}")
    sys.exit(1)

if len(full_ds) < TRAIN_BARS + TEST_BARS:
    print(f"ERROR: Not enough data. Need {TRAIN_BARS + TEST_BARS:,}, have {len(full_ds):,}")
    sys.exit(1)

# ─ Walk-forward logic ───────────────────────────────────────────────────────
print()
print("=" * 80)
print("  WALK-FORWARD VALIDATION")
print("=" * 80)

all_folds = []
fold_idx = 0

# Calculate folds
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
    
    print(f"\n[Fold {fold_idx}]")
    print(f"  Train: {train_start.date()} → {train_end.date()} ({TRAIN_BARS:,} bars)")
    print(f"  Test:  {test_start.date()} → {test_end.date()} ({TEST_BARS:,} bars)")
    
    # Prepare data
    train_ds = full_ds.iloc[train_start_idx:train_end_idx].copy()
    test_ds = full_ds.iloc[train_end_idx:test_end_idx].copy()
    
    # Labels (lookahead)
    label_bars = 30  # TP/SL horizon
    train_labels = get_label_lookahead_bars(
        train_ds["close"].values,
        train_ds.get("atr", pd.Series([1.0]*len(train_ds))).values,
        lookahead_bars=label_bars,
        tp_rr=CFG.models.trading.take_profit_rr,
        sl_atr_mult=CFG.models.trading.stop_loss_atr_multiple,
    )
    test_labels = get_label_lookahead_bars(
        test_ds["close"].values,
        test_ds.get("atr", pd.Series([1.0]*len(test_ds))).values,
        lookahead_bars=label_bars,
        tp_rr=CFG.models.trading.take_profit_rr,
        sl_atr_mult=CFG.models.trading.stop_loss_atr_multiple,
    )
    
    # Features
    feature_cols = FEATURE_COLUMNS
    X_train = train_ds[feature_cols].fillna(0).values
    X_test = test_ds[feature_cols].fillna(0).values
    y_train = train_labels
    y_test = test_labels
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train ensemble (HistGB + RF + ExtraTrees, voting)
    print(f"  Training ensemble (HGB+RF+ET)...")
    clf_hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=8, learning_rate=0.05, random_state=42)
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    clf_et = ExtraTreesClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    
    ensemble = VotingClassifier(
        estimators=[("hgb", clf_hgb), ("rf", clf_rf), ("et", clf_et)],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"  Eval: Acc={acc:.1%} F1={f1:.1%} AUC={auc:.3f}")
    
    all_folds.append({
        "fold": fold_idx,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_bars": len(train_ds),
        "test_bars": len(test_ds),
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "test_y_true": y_test.tolist(),
        "test_y_pred": y_pred.tolist(),
        "test_y_prob": y_prob.tolist(),
    })
    
    train_start_idx += STEP_BARS
    
    if _args.max_folds and fold_idx >= _args.max_folds:
        break

print()
print("=" * 80)
print("  SUMMARY")
print("=" * 80)

avg_acc = np.mean([f["acc"] for f in all_folds])
avg_f1 = np.mean([f["f1"] for f in all_folds])
avg_auc = np.mean([f["auc"] for f in all_folds])

print(f"Total folds: {len(all_folds)}")
print(f"Avg Accuracy: {avg_acc:.1%}")
print(f"Avg F1: {avg_f1:.1%}")
print(f"Avg AUC: {avg_auc:.3f}")
print()

# Save results
out_json = Path(f"outputs/walkforward_extended_150k_{_cfg_stem}_results.json")
with open(out_json, "w") as f:
    json.dump(all_folds, f, indent=2, default=str)
print(f"Results saved: {out_json}")

print("\nDone!")
