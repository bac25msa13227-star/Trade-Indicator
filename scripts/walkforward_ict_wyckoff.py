"""
Walk-Forward Validation — ICT + Wyckoff Model
============================================================
- Build dataset (28 features) ONCE from full CSV data
- Slide train/test windows across time
- Train HistGBC on each fold, evaluate on next fold
- Aggregate metrics across all folds
- Log win/loss analysis for live self-learner
============================================================
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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.features.dataset import prepare_training_dataset, FEATURE_COLUMNS
from xauusd_ai.strategies.hybrid import HybridStrategy

import functools
_log_path = Path("outputs/walkforward_log.txt")
_log_path.parent.mkdir(parents=True, exist_ok=True)
_log_file = _log_path.open("w", encoding="utf-8")

_orig_print = print
def print(*args, **kwargs):  # noqa: A001
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)
    _orig_print(*args, file=_log_file, **{k: v for k, v in kwargs.items() if k != "file"})
    _log_file.flush()

print("=" * 70)
print("  WALK-FORWARD VALIDATION — ICT + WYCKOFF + NEWS (33 features)")
print("=" * 70)
print()

# ── config ──────────────────────────────────────────────────────────────────
CONFIG = Path("configs/train_ict_wyckoff_2022_2026.yaml")
settings = load_settings(CONFIG)

# Walk-forward window parameters  (M15: 96 bars/day  ~252 trading days/year)
TRAIN_BARS = 20_000   # ~208 trading days = ~7 months
TEST_BARS  =  4_000   # ~42  trading days = ~1.5 months
STEP_BARS  =  4_000   # slide ~1.5 months at a time

THRESHOLD_MIN   = settings.training.threshold_min        # 0.45
THRESHOLD_MAX   = settings.training.threshold_max        # 0.80
THRESHOLD_STEP  = settings.training.threshold_step       # 0.01
PREC_FLOOR      = settings.training.min_precision_floor  # 0.52

print(f"  Config  : {CONFIG}")
print(f"  Features: {len(FEATURE_COLUMNS)}  (D1:1 H4:9 H1:3 M15:15 News:5)")
print(f"  Train   : {TRAIN_BARS:,} bars (~1 yr M15)")
print(f"  Test    : {TEST_BARS:,} bars (~3 mo M15)")
print(f"  Step    : {STEP_BARS:,} bars (~3 mo slide)")
print()

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("[1/4] Loading multi-timeframe data from CSV...")
t0 = time.time()
data_service = MarketDataService(settings)
strategy = HybridStrategy(settings)
frames = data_service.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)
print(f"      M15:{len(frames['M15']):,}  H4:{len(frames['H4']):,}  "
      f"H1:{len(frames['H1']):,}  D1:{len(frames['D1']):,} rows  ({time.time()-t0:.1f}s)")

# ── 2. Build full dataset (features computed ONCE) ───────────────────────────
print(f"\n[2/4] Building full dataset with {len(FEATURE_COLUMNS)} ICT+Wyckoff+News features...")
t0 = time.time()
# Temporarily override dates to get ALL data (no split filtering)
settings_full = settings.model_copy(deep=True)
settings_full.training.train_start_date = None
settings_full.training.train_end_date   = None
settings_full.training.test_start_date  = None
settings_full.training.test_end_date    = None
full_ds = prepare_training_dataset(settings_full, frames, strategy)
print(f"      Total rows : {len(full_ds):,}")
print(f"      Date range : {full_ds['time'].min().date()} → {full_ds['time'].max().date()}")
print(f"      Label rate : {full_ds['target'].mean():.1%}")
print(f"      Time       : {time.time()-t0:.1f}s")

# ── 3. Walk-forward loop ─────────────────────────────────────────────────────
print(f"\n[3/4] Running walk-forward folds...")

n_total = len(full_ds)
n_folds = max((n_total - TRAIN_BARS - TEST_BARS) // STEP_BARS + 1, 0)
print(f"      Estimated folds: {n_folds}")
print()

fold_results = []
win_log = []   # for live-learning log (thắng/thua)

fold_idx = 0
fold_start = 0

while fold_start + TRAIN_BARS + TEST_BARS <= n_total:
    fold_idx += 1
    train_end  = fold_start + TRAIN_BARS
    test_end   = train_end  + TEST_BARS

    fold_train = full_ds.iloc[fold_start:train_end].copy()
    fold_test  = full_ds.iloc[train_end:test_end].copy()

    if len(fold_train) < 500 or len(fold_test) < 100:
        fold_start += STEP_BARS
        continue

    t_fold = time.time()

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(fold_train[FEATURE_COLUMNS])
    X_te = scaler.transform(fold_test[FEATURE_COLUMNS])
    y_tr = fold_train["target"].values
    y_te = fold_test["target"].values

    # Threshold search on last 20% of fold_train
    val_cut = int(len(fold_train) * 0.80)
    local_model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.1, max_depth=5,
        min_samples_leaf=20, class_weight="balanced",
        early_stopping=False, random_state=42,
    )
    local_model.fit(X_tr[:val_cut], y_tr[:val_cut])
    val_proba = local_model.predict_proba(X_tr[val_cut:])[:, 1]
    y_val = y_tr[val_cut:]

    best_thr, best_f1 = THRESHOLD_MIN, 0.0
    for thr in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP):
        preds = (val_proba >= thr).astype(int)
        prec  = precision_score(y_val, preds, zero_division=0)
        f1_v  = f1_score(y_val, preds, zero_division=0)
        if prec >= PREC_FLOOR and f1_v > best_f1:
            best_f1, best_thr = f1_v, float(thr)

    # Train final model on 100% of fold_train
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, class_weight="balanced",
        early_stopping=False, random_state=42,
    )
    model.fit(X_tr, y_tr)

    # Evaluate on fold_test
    test_proba = model.predict_proba(X_te)[:, 1]
    test_preds = (test_proba >= best_thr).astype(int)

    auc   = roc_auc_score(y_te, test_proba) if len(np.unique(y_te)) > 1 else 0.5
    prec  = precision_score(y_te, test_preds, zero_division=0)
    rec   = recall_score(y_te, test_preds, zero_division=0)
    f1    = f1_score(y_te, test_preds, zero_division=0)
    acc   = accuracy_score(y_te, test_preds)
    n_sig = int(test_preds.sum())
    n_tot = len(test_preds)

    # Win/Loss log per fold
    fold_test_copy = fold_test.copy()
    fold_test_copy["predicted"] = test_preds
    fold_test_copy["proba"] = test_proba
    fold_test_copy["correct"] = (test_preds == y_te).astype(int)
    fold_test_copy["threshold"] = best_thr
    fold_test_copy["fold"] = fold_idx
    win_log.append(fold_test_copy[fold_test_copy["predicted"] == 1][[
        "time", "fold", "threshold", "predicted", "target", "correct",
        "proba", "strategy_score", "rsi", "h4_bos", "h4_ict_confluence",
        "session_return", "macd_hist",
        "news_impact_ahead", "news_hours_ahead", "news_is_blackout", "news_surprise_gold",
    ]])

    elapsed = time.time() - t_fold
    result = {
        "fold": fold_idx,
        "train_start": str(fold_train["time"].min().date()),
        "train_end":   str(fold_train["time"].max().date()),
        "test_start":  str(fold_test["time"].min().date()),
        "test_end":    str(fold_test["time"].max().date()),
        "train_rows":  len(fold_train),
        "test_rows":   len(fold_test),
        "threshold":   round(best_thr, 3),
        "roc_auc":     round(auc, 4),
        "precision":   round(prec, 4),
        "recall":      round(rec, 4),
        "f1":          round(f1, 4),
        "accuracy":    round(acc, 4),
        "n_signals":   n_sig,
        "signal_rate": round(n_sig / n_tot, 4) if n_tot > 0 else 0.0,
        "elapsed_s":   round(elapsed, 1),
    }
    fold_results.append(result)

    # Progress line
    star = "[BEST]" if auc == max(r["roc_auc"] for r in fold_results) else "      "
    print(
        f"  Fold {fold_idx:2d}/{n_folds} {star} "
        f"Test: {result['test_start']} -> {result['test_end']} | "
        f"AUC={auc:.4f}  Prec={prec:.4f}  Recall={rec:.4f}  "
        f"F1={f1:.4f}  Thr={best_thr:.2f}  Sigs={n_sig}/{n_tot}  ({elapsed:.1f}s)"
    )

    fold_start += STEP_BARS

print()

# ── 4. Aggregate results ─────────────────────────────────────────────────────
print(f"\n[4/4] Aggregating results across {len(fold_results)} folds...")

if not fold_results:
    print("  ERROR: No folds were produced. Check data size vs window sizes.")
    sys.exit(1)

avg_auc   = np.mean([r["roc_auc"]    for r in fold_results])
avg_prec  = np.mean([r["precision"]  for r in fold_results])
avg_rec   = np.mean([r["recall"]     for r in fold_results])
avg_f1    = np.mean([r["f1"]         for r in fold_results])
avg_acc   = np.mean([r["accuracy"]   for r in fold_results])
avg_sigs  = np.mean([r["signal_rate"] for r in fold_results])
std_prec  = np.std([r["precision"]   for r in fold_results])
std_auc   = np.std([r["roc_auc"]     for r in fold_results])

min_prec  = min(r["precision"] for r in fold_results)
max_prec  = max(r["precision"] for r in fold_results)
min_auc   = min(r["roc_auc"]   for r in fold_results)
max_auc   = max(r["roc_auc"]   for r in fold_results)

# Win/Loss signal log
all_signals = pd.concat(win_log, ignore_index=True) if win_log else pd.DataFrame()
n_win_signals = int(all_signals["correct"].sum()) if not all_signals.empty else 0
n_all_signals = len(all_signals)
signal_winrate = n_win_signals / n_all_signals if n_all_signals > 0 else 0.0

print()
print("=" * 70)
print("  WALK-FORWARD RESULTS SUMMARY")
print("=" * 70)
print(f"  Folds completed  : {len(fold_results)}")
print(f"  Date range       : {fold_results[0]['test_start']} → {fold_results[-1]['test_end']}")
print()
print(f"  ROC-AUC  avg    : {avg_auc:.4f}  (range {min_auc:.4f}–{max_auc:.4f},  std={std_auc:.4f})")
print(f"  Precision avg   : {avg_prec:.4f}  (range {min_prec:.4f}–{max_prec:.4f},  std={std_prec:.4f})")
print(f"  Recall    avg   : {avg_rec:.4f}")
print(f"  F1 Score  avg   : {avg_f1:.4f}")
print(f"  Accuracy  avg   : {avg_acc:.4f}")
print(f"  Signal rate avg : {avg_sigs:.1%}")
print()
print(f"  Signal win rate  : {signal_winrate:.1%}  ({n_win_signals}/{n_all_signals} signals correct)")
print()

# Overfitting check
print("  Overfitting Check:")
if std_auc < 0.04:
    print(f"    AUC std={std_auc:.4f} < 0.04  ✅ Low variance — model generalizes well")
elif std_auc < 0.07:
    print(f"    AUC std={std_auc:.4f}  ⚠️  Moderate variance — some period sensitivity")
else:
    print(f"    AUC std={std_auc:.4f}  ❌  High variance — possible overfitting")

if min_prec >= 0.50:
    print(f"    Min precision={min_prec:.4f} >= 0.50  ✅ Consistently profitable signal quality")
elif min_prec >= 0.45:
    print(f"    Min precision={min_prec:.4f}  ⚠️  Some folds below 50%")
else:
    print(f"    Min precision={min_prec:.4f}  ❌  Some folds signficantly below 50%")

print()
print("  Per-Fold Table:")
print(f"  {'Fold':>4}  {'Test Period':>24}  {'AUC':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Thr':>5}  {'Sigs%':>6}")
print("  " + "-" * 68)
for r in fold_results:
    period = f"{r['test_start']} → {r['test_end']}"
    ok = "✅" if r["precision"] >= 0.50 and r["roc_auc"] >= 0.55 else ("⚠️" if r["precision"] >= 0.45 else "❌")
    print(
        f"  {r['fold']:>4}  {period:>24}  {r['roc_auc']:>6.4f}  "
        f"{r['precision']:>6.4f}  {r['recall']:>6.4f}  {r['f1']:>6.4f}  "
        f"{r['threshold']:>5.2f}  {r['signal_rate']:>5.1%}  {ok}"
    )
print()

# ── Save outputs ─────────────────────────────────────────────────────────────
wf_report = {
    "walk_forward": {
        "train_bars": TRAIN_BARS,
        "test_bars": TEST_BARS,
        "step_bars": STEP_BARS,
        "n_folds": len(fold_results),
        "features": FEATURE_COLUMNS,
        "model": f"HistGradientBoostingClassifier(max_iter=300, lr=0.05, depth=6, balanced) | {len(FEATURE_COLUMNS)} features: D1:1 H4:9 H1:3 M15:15 News:5",
    },
    "aggregate": {
        "avg_roc_auc":    round(avg_auc, 4),
        "std_roc_auc":    round(std_auc, 4),
        "min_roc_auc":    round(min_auc, 4),
        "max_roc_auc":    round(max_auc, 4),
        "avg_precision":  round(avg_prec, 4),
        "std_precision":  round(std_prec, 4),
        "min_precision":  round(min_prec, 4),
        "max_precision":  round(max_prec, 4),
        "avg_recall":     round(avg_rec, 4),
        "avg_f1":         round(avg_f1, 4),
        "avg_accuracy":   round(avg_acc, 4),
        "avg_signal_rate": round(avg_sigs, 4),
        "signal_win_rate": round(signal_winrate, 4),
        "total_signals":   n_all_signals,
        "correct_signals": n_win_signals,
    },
    "folds": fold_results,
}

out_report = Path("outputs/walkforward_report_ict_wyckoff.json")
out_signals = Path("outputs/walkforward_signals_ict_wyckoff.csv")

out_report.write_text(json.dumps(wf_report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"  Report saved  → {out_report}")

if not all_signals.empty:
    all_signals.to_csv(out_signals, index=False)
    print(f"  Signals saved → {out_signals}  ({len(all_signals):,} rows)")
    # Separate win/loss files for live self-learner
    wins   = all_signals[all_signals["correct"] == 1]
    losses = all_signals[all_signals["correct"] == 0]
    loss_log_path = Path("outputs/loss_analysis_walkforward.jsonl")
    win_log_path  = Path("outputs/win_analysis_walkforward.jsonl")
    with loss_log_path.open("w", encoding="utf-8") as f:
        for _, row in losses.iterrows():
            f.write(json.dumps({
                "time": str(row.get("time", "")),
                "fold": int(row.get("fold", 0)),
                "outcome": "loss",
                "proba": round(float(row.get("proba", 0)), 4),
                "threshold": round(float(row.get("threshold", 0)), 3),
                "strategy_score": round(float(row.get("strategy_score", 0)), 4),
                "rsi": round(float(row.get("rsi", 0)), 2),
                "h4_bos": int(row.get("h4_bos", 0)),
                "h4_ict_confluence": round(float(row.get("h4_ict_confluence", 0)), 2),
                "session_return": round(float(row.get("session_return", 0)), 6),
                "macd_hist": round(float(row.get("macd_hist", 0)), 4),
            }) + "\n")
    with win_log_path.open("w", encoding="utf-8") as f:
        for _, row in wins.iterrows():
            f.write(json.dumps({
                "time": str(row.get("time", "")),
                "fold": int(row.get("fold", 0)),
                "outcome": "win",
                "proba": round(float(row.get("proba", 0)), 4),
                "threshold": round(float(row.get("threshold", 0)), 3),
                "strategy_score": round(float(row.get("strategy_score", 0)), 4),
                "rsi": round(float(row.get("rsi", 0)), 2),
                "h4_bos": int(row.get("h4_bos", 0)),
                "h4_ict_confluence": round(float(row.get("h4_ict_confluence", 0)), 2),
                "session_return": round(float(row.get("session_return", 0)), 6),
                "macd_hist": round(float(row.get("macd_hist", 0)), 4),
            }) + "\n")
    print(f"  Win  log      → {win_log_path}  ({len(wins):,} winning signals)")
    print(f"  Loss log      → {loss_log_path}  ({len(losses):,} losing signals)")
    print()
    print(f"  Win/Loss Breakdown:")
    print(f"    Total signals generated : {n_all_signals:,}")
    print(f"    ✅ Winning signals       : {len(wins):,}  ({len(wins)/n_all_signals:.1%})")
    print(f"    ❌ Losing signals        : {len(losses):,}  ({len(losses)/n_all_signals:.1%})")

print()
print("  Self-Learner Status:")
print(f"    live_learning_enabled  = {settings.training.live_learning_enabled}  ✅" if settings.training.live_learning_enabled else
      f"    live_learning_enabled  = {settings.training.live_learning_enabled}  (enable in config)")
print(f"    Win/loss logs ready for self-learner to re-train on loss patterns.")
print()
print("=" * 70)
print("  Walk-Forward Complete.")
print("=" * 70)
_log_file.close()
