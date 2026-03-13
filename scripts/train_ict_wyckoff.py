"""
Standalone training script — ICT + Wyckoff Enhanced Model
Train: 2022-01-01 - 2025-12-31
Test:  2026-01-01 - 2026-03-09
"""
import os
import sys
import warnings
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import json
import sys
import time
from pathlib import Path

# Thêm src vào path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.features.dataset import prepare_training_dataset, FEATURE_COLUMNS
from xauusd_ai.strategies.hybrid import HybridStrategy

print("=" * 70)
print("  ICT + WYCKOFF ENHANCED MODEL — TRAINING")
print(f"  Features: {len(FEATURE_COLUMNS)} total")
print("=" * 70)
print()

# Load config
settings = load_settings(Path("configs/train_ict_wyckoff_2022_2026.yaml"))

print(f"[1/6] Loading multi-timeframe data from CSV...")
t0 = time.time()
data_service = MarketDataService(settings)
strategy = HybridStrategy(settings)
frames = data_service.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)
print(f"      M15:{len(frames['M15'])} H4:{len(frames['H4'])} H1:{len(frames['H1'])} D1:{len(frames['D1'])} rows  ({time.time()-t0:.1f}s)")

print(f"\n[2/6] Building dataset with ICT+Wyckoff features...")
t0 = time.time()
dataset = prepare_training_dataset(settings, frames, strategy)
print(f"      Total rows: {len(dataset)}")
print(f"      Features:   {FEATURE_COLUMNS}")
print(f"      Time:       {time.time()-t0:.1f}s")

train_df = dataset[dataset["split"] == "train"]
test_df  = dataset[dataset["split"] == "test"]
print(f"\n      TRAIN: {len(train_df)} rows | {train_df['time'].min()} -> {train_df['time'].max()}")
print(f"      TEST:  {len(test_df)} rows | {test_df['time'].min()} -> {test_df['time'].max()}")
print(f"      Train label rate: {train_df['target'].mean():.1%}")
print(f"      Test  label rate: {test_df['target'].mean():.1%}")

print(f"\n[3/6] Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS])
X_test  = scaler.transform(test_df[FEATURE_COLUMNS])
y_train = train_df["target"].values
y_test  = test_df["target"].values

print(f"\n[4/7] Training threshold-search model on 80% of train set...")
t0 = time.time()
val_cut   = int(len(train_df) * 0.80)
X_opt     = scaler.transform(train_df.iloc[:val_cut][FEATURE_COLUMNS])
y_opt     = train_df.iloc[:val_cut]["target"].values
X_val_opt = scaler.transform(train_df.iloc[val_cut:][FEATURE_COLUMNS])
y_val_opt = train_df.iloc[val_cut:]["target"].values

opt_model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    class_weight="balanced",
    early_stopping=False,
    random_state=42,
)
opt_model.fit(X_opt, y_opt)
print(f"      Opt model time: {time.time()-t0:.1f}s")

print(f"\n[5/7] Optimizing threshold (floor={settings.training.min_precision_floor:.0%}, range={settings.training.threshold_min:.2f}-{settings.training.threshold_max:.2f})...")
val_proba_opt = opt_model.predict_proba(X_val_opt)[:, 1]
best_thr, best_prec, best_f1 = settings.training.threshold_min, 0.0, 0.0
for thr in np.arange(settings.training.threshold_min,
                     settings.training.threshold_max + settings.training.threshold_step,
                     settings.training.threshold_step):
    preds = (val_proba_opt >= thr).astype(int)
    prec  = precision_score(y_val_opt, preds, zero_division=0)
    f1_s  = f1_score(y_val_opt, preds, zero_division=0)
    if prec >= settings.training.min_precision_floor and f1_s > best_f1:
        best_f1, best_prec, best_thr = f1_s, prec, float(thr)
print(f"      Best threshold: {best_thr:.2f}  (precision={best_prec:.4f}, f1={best_f1:.4f})")

print(f"\n[6/7] Retraining FINAL model on 100% of train set...")
t0 = time.time()
model = HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    class_weight="balanced",
    early_stopping=False,
    random_state=42,
)
model.fit(X_train, y_train)
print(f"      Final model training time: {time.time()-t0:.1f}s")

# Evaluate on TEST set
print(f"\n[7/7] Evaluating on TEST set (2026)...")
test_proba = model.predict_proba(X_test)[:, 1]
test_preds = (test_proba >= best_thr).astype(int)

acc   = accuracy_score(y_test, test_preds)
prec  = precision_score(y_test, test_preds, zero_division=0)
rec   = recall_score(y_test, test_preds, zero_division=0)
f1    = f1_score(y_test, test_preds, zero_division=0)
auc   = roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0.5

# Feature importances — HistGBC uses permutation importance (no built-in impurity method)
from sklearn.inspection import permutation_importance as _perm_imp
print("  Computing permutation importances (n_repeats=8) on test set...")
_perm = _perm_imp(model, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1)
importances = _perm.importances_mean
feat_imp = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1])

print()
print("=" * 70)
print("  RESULTS — TEST SET 2026")
print("=" * 70)
print(f"  Dataset")
print(f"    Train: {len(train_df):>8,} rows  ({train_df['time'].min().date()} -> {train_df['time'].max().date()})")
print(f"    Test : {len(test_df):>8,} rows  ({test_df['time'].min().date()} -> {test_df['time'].max().date()})")
print()
print(f"  Metrics")
print(f"    ROC-AUC   : {auc:.4f}")
print(f"    Precision  : {prec:.4f}  ({'✅ GOOD' if prec >= 0.50 else '⚠️ LOW'})")
print(f"    Recall     : {rec:.4f}")
print(f"    F1 Score   : {f1:.4f}")
print(f"    Accuracy   : {acc:.4f}")
print(f"    Threshold  : {best_thr:.2f}")
print()
print(f"  Predictions")
n_signals = int(test_preds.sum())
n_total   = len(test_preds)
print(f"    Signals   : {n_signals:>6} / {n_total} ({n_signals/n_total:.1%} signal rate)")
print()
print(f"  Confusion Matrix")
cm = confusion_matrix(y_test, test_preds)
print(f"    TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
print(f"    FN={cm[1][0]:>5}  TP={cm[1][1]:>5}")
print()
print(f"  Top 15 Feature Importances")
for i, (feat, imp) in enumerate(feat_imp[:15]):
    bar = "█" * int(imp * 200)
    tag = ""
    if feat.startswith("h4_"):
        tag = " [H4-ICT]"
    elif feat in ["vsa_signal", "wyckoff_spring_signal", "wyckoff_phase"]:
        tag = " [WYCKOFF]"
    elif feat in ["kill_zone_flag", "judas_swing_signal"]:
        tag = " [M15-ICT]"
    print(f"    {i+1:2}. {feat:<30}{tag:<12} {imp:.4f}  {bar}")

print()
print(f"  Classification Report")
print(classification_report(y_test, test_preds, target_names=["No-Trade","Trade"]))
print("=" * 70)

# Save results
results = {
    "model": "HistGradientBoostingClassifier (max_iter=500, lr=0.05, depth=6, balanced)",
    "features_count": len(FEATURE_COLUMNS),
    "features": FEATURE_COLUMNS,
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "train_period": f"{train_df['time'].min().date()} to {train_df['time'].max().date()}",
    "test_period": f"{test_df['time'].min().date()} to {test_df['time'].max().date()}",
    "decision_threshold": round(best_thr, 3),
    "roc_auc": round(auc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1": round(f1, 4),
    "accuracy": round(acc, 4),
    "n_signals": n_signals,
    "signal_rate": round(n_signals / n_total, 4),
    "feature_importances": {f: round(float(imp), 5) for f, imp in feat_imp},
}
Path("outputs/training_report_ict_wyckoff.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"\nResults saved to outputs/training_report_ict_wyckoff.json")
print("Done.")
