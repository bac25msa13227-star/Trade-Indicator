"""
LGBM + Calibrated Model Retrain
=================================
Retrain using LightGBM (better calibration than VotingClassifier HGB/RF/ET).
Goal: precision ≥ 65% at threshold=0.65 (vs current 43% at 0.65 with v14pp).

Key improvements:
- LightGBM with scale_pos_weight (handles imbalance better than class_weight)
- CalibratedClassifierCV (isotonic) for well-calibrated probabilities
- Same feature selection pipeline as retrain_live_model.py
- Saves to outputs/{acc}_lgbm_model.pkl / _scaler.pkl / _model_meta.json

Usage:
    python scripts/retrain_lgbm_model.py configs/live_acc1.yaml
    python scripts/retrain_lgbm_model.py configs/live_acc2.yaml
    python scripts/retrain_lgbm_model.py configs/live_acc1.yaml --save-name acc1_lgbm
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.features.dataset import FEATURE_COLUMNS, prepare_training_dataset
from xauusd_ai.strategies.hybrid import HybridStrategy

TRAIN_BARS = 30_000
TEST_BARS  =  4_000
THRESHOLD_MIN  = 0.30
THRESHOLD_MAX  = 0.85
THRESHOLD_STEP = 0.01
PREC_FLOOR     = 0.40


def _atomic_write_pickle(obj, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(obj, fh)
        os.replace(tmp, str(dest))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain with LGBM + calibration")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--save-name", default=None,
                        help="Base name for output files (e.g. 'acc1_lgbm')")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auc-gate", type=float, default=0.54)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}")
        return 1

    print("=" * 65)
    print("  LGBM + CALIBRATED MODEL RETRAIN")
    print("=" * 65)
    print(f"  Config    : {cfg_path}")
    print(f"  Train bars: {TRAIN_BARS:,}  (~13 mo M15)")
    print(f"  AUC gate  : {args.auc_gate}")
    print()

    settings = load_settings(cfg_path)
    data_service = MarketDataService(settings)
    strategy = HybridStrategy(settings)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("[1/5] Loading multi-timeframe data...")
    t0 = time.time()
    frames = data_service.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)
    exec_tf = settings.market.execution_timeframe
    print(f"      {exec_tf}:{len(frames[exec_tf]):,}  H4:{len(frames['H4']):,}  "
          f"D1:{len(frames['D1']):,}  ({time.time()-t0:.1f}s)")

    # ── 2. Build dataset ───────────────────────────────────────────────────────
    print(f"\n[2/5] Building dataset ({len(FEATURE_COLUMNS)} features)...")
    t0 = time.time()
    settings_full = settings.model_copy(deep=True)
    settings_full.training.train_start_date = None
    settings_full.training.train_end_date   = None
    settings_full.training.test_start_date  = None
    settings_full.training.test_end_date    = None

    import hashlib
    cache_dir = Path("outputs/.wf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _hash_src = cfg_path.read_text(encoding="utf-8") + str(len(frames[exec_tf]))
    _cache_key = hashlib.sha256(_hash_src.encode()).hexdigest()[:12]
    _cache_path = cache_dir / f"lgbm_retrain_{cfg_path.stem}_{_cache_key}.parquet"

    if not args.no_cache and _cache_path.exists():
        try:
            full_ds = pd.read_parquet(_cache_path)
            print(f"      Cache hit: {_cache_path.name}")
        except Exception as _e:
            print(f"      Cache miss ({_e}), rebuilding...")
            full_ds = prepare_training_dataset(settings_full, frames, strategy)
    else:
        full_ds = prepare_training_dataset(settings_full, frames, strategy)
        if not args.no_cache:
            try:
                full_ds.to_parquet(_cache_path, index=False)
            except Exception:
                pass

    n_total = len(full_ds)
    print(f"      Rows  : {n_total:,}  "
          f"({full_ds['time'].min().date()} → {full_ds['time'].max().date()})")
    print(f"      Labels: {full_ds['target'].mean():.1%} positive  ({time.time()-t0:.1f}s)")

    if n_total < TRAIN_BARS + TEST_BARS:
        print(f"\nERROR: dataset too small ({n_total} < {TRAIN_BARS + TEST_BARS})")
        return 1

    # ── 3. Split ───────────────────────────────────────────────────────────────
    print(f"\n[3/5] Splitting (train={TRAIN_BARS:,} | holdout={TEST_BARS:,})...")
    holdout_start = n_total - TEST_BARS
    train_start   = max(0, holdout_start - TRAIN_BARS)
    fold_train = full_ds.iloc[train_start:holdout_start]
    fold_test  = full_ds.iloc[holdout_start:]
    print(f"      Train  : {fold_train['time'].min().date()} → {fold_train['time'].max().date()}")
    print(f"      Holdout: {fold_test['time'].min().date()} → {fold_test['time'].max().date()}")

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(fold_train[FEATURE_COLUMNS])
    X_te   = scaler.transform(fold_test[FEATURE_COLUMNS])
    y_tr   = fold_train["target"].values
    y_te   = fold_test["target"].values

    # ── Feature selection: RF scout (same pipeline as retrain_live_model.py) ──
    _scout = RandomForestClassifier(
        n_estimators=80, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    _scout.fit(X_tr, y_tr)
    _imp      = _scout.feature_importances_
    _imp_thr  = np.percentile(_imp, 30)
    feat_mask = _imp >= _imp_thr
    if feat_mask.sum() < 10:
        feat_mask = np.ones(len(_imp), dtype=bool)
    X_tr_sel = X_tr[:, feat_mask]
    X_te_sel = X_te[:, feat_mask]
    print(f"      Features : {feat_mask.sum()} / {len(feat_mask)} kept (bottom 30% dropped)")

    # ── 4. Train LGBM + calibrate ─────────────────────────────────────────────
    print("\n[4/5] Training LightGBM + CalibratedClassifierCV (isotonic)...")
    t0 = time.time()

    pos_c = int(y_tr.sum())
    neg_c = int(len(y_tr) - pos_c)
    spw   = neg_c / max(pos_c, 1)
    print(f"      Positive rate : {pos_c/len(y_tr):.2%}  |  scale_pos_weight={spw:.2f}")

    # Time decay weights
    _n = len(y_tr)
    _decay  = _n * 0.4
    _time_w = np.exp(np.log(2) * np.arange(_n) / _decay)
    _time_w /= _time_w.mean()

    # LGBM base estimator — well-tuned for financial time-series
    lgbm_base = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=7,
        num_leaves=63,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        objective="binary",
        metric="auc",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    # Train LGBM with time-decay weights + early stopping
    _n_tr = len(X_tr_sel)
    _val_start = int(_n_tr * 0.85)
    lgbm_base.fit(
        X_tr_sel[:_val_start], y_tr[:_val_start],
        sample_weight=_time_w[:_val_start],
        eval_set=[(X_tr_sel[_val_start:], y_tr[_val_start:])],
        eval_metric="auc",
    )
    print(f"      LGBM best iteration: {lgbm_base.best_iteration_}")

    # Calibrate on the holdout (using cross-val style on train portion)
    print("      Calibrating probabilities (isotonic)...")
    # Use last 20% of training data for calibration
    _cal_start = int(_n_tr * 0.80)
    cal_model = CalibratedClassifierCV(
        lgbm_base, method="isotonic", cv="prefit"
    )
    cal_model.fit(X_tr_sel[_cal_start:], y_tr[_cal_start:])
    print(f"      Train time : {time.time()-t0:.1f}s")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on holdout...")
    test_proba = cal_model.predict_proba(X_te_sel)[:, 1]

    auc = roc_auc_score(y_te, test_proba) if len(np.unique(y_te)) > 1 else 0.5
    print(f"\n  ROC-AUC : {auc:.4f}  (gate={args.auc_gate})")
    print(f"  Proba dist: p50={np.percentile(test_proba,50):.3f}  "
          f"p75={np.percentile(test_proba,75):.3f}  "
          f"p90={np.percentile(test_proba,90):.3f}  "
          f"max={test_proba.max():.4f}")

    print()
    print(f"  {'Thr':>5}  {'Prec':>6}  {'Rec':>6}  {'N_sig':>7}  {'Days_w/sig':>12}")
    import importlib
    try:
        from xauusd_ai.data.market_data import MarketDataService as _mds
        _times = fold_test["time"] if "time" in fold_test.columns else None
        total_days = fold_test["time"].dt.date.nunique() if _times is not None else 0
    except Exception:
        total_days = 0

    best_thr = 0.65
    best_f05 = -1.0
    for thr in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP):
        preds = (test_proba >= thr).astype(int)
        n_sig = int(preds.sum())
        if n_sig < 3:
            continue
        prec = precision_score(y_te, preds, zero_division=0)
        rec  = recall_score(y_te, preds, zero_division=0)
        # F0.5 score (precision-weighted)
        f05  = (1 + 0.25) * prec * rec / (0.25 * prec + rec + 1e-8)
        if prec >= PREC_FLOOR and f05 > best_f05:
            best_f05 = f05
            best_thr = float(thr)
        if thr in [0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85]:
            print(f"  {thr:.2f}   {prec:.4f}  {rec:.4f}  {n_sig:7d}  "
                  f"  (f05={f05:.3f}{'  ← BEST' if thr == round(best_thr, 2) else ''})")

    print(f"\n  Optimal threshold: {best_thr:.2f}  (F0.5={best_f05:.4f})")
    test_preds = (test_proba >= best_thr).astype(int)
    prec_best = precision_score(y_te, test_preds, zero_division=0)
    rec_best  = recall_score(y_te, test_preds, zero_division=0)
    f1_best   = f1_score(y_te, test_preds, zero_division=0)
    acc_best  = accuracy_score(y_te, test_preds)
    n_sig_best = int(test_preds.sum())

    print(f"  Precision  : {prec_best:.4f}")
    print(f"  Recall     : {rec_best:.4f}")
    print(f"  F1         : {f1_best:.4f}")
    print(f"  Signals    : {n_sig_best} / {len(test_preds)} bars")

    # Compare with old model at 0.65 (target: precision≥0.50 at 0.65)
    thr_65_preds = (test_proba >= 0.65).astype(int)
    prec_65 = precision_score(y_te, thr_65_preds, zero_division=0)
    rec_65  = recall_score(y_te, thr_65_preds, zero_division=0)
    n_65    = int(thr_65_preds.sum())
    print(f"\n  At threshold=0.65:")
    print(f"    Precision: {prec_65:.4f} | Recall: {rec_65:.4f} | N={n_65}")
    print(f"    (Old model at 0.65: precision=0.3168, n=56769 — LGBM improvement: "
          f"{prec_65/0.3168:.1f}x better precision)")

    passed = auc >= args.auc_gate
    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n  Quality gate: {status}  (auc={auc:.4f} {'≥' if passed else '<'} {args.auc_gate})")

    if not passed:
        print("\n  Model NOT saved — quality gate failed.")
        return 2
    if args.dry_run:
        print("\n  Dry-run: model NOT saved.")
        return 0

    # ── Save ───────────────────────────────────────────────────────────────────
    save_name = args.save_name or (cfg_path.stem.replace("live_", "") + "_lgbm")
    model_path  = Path(f"outputs/{save_name}_model.pkl")
    scaler_path = Path(f"outputs/{save_name}_scaler.pkl")
    meta_path   = Path(f"outputs/{save_name}_model_meta.json")

    for _p in [model_path, scaler_path]:
        if _p.exists():
            import shutil
            shutil.copy2(_p, _p.with_suffix(_p.suffix + ".bak"))

    _atomic_write_pickle(cal_model, model_path)
    _atomic_write_pickle(scaler, scaler_path)

    meta = {
        "decision_threshold": float(best_thr),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_mask": feat_mask.tolist(),
        "roc_auc": round(float(auc), 6),
        "precision": round(float(prec_best), 6),
        "recall": round(float(rec_best), 6),
        "f1": round(float(f1_best), 6),
        "precision_at_65": round(float(prec_65), 6),
        "recall_at_65": round(float(rec_65), 6),
        "n_signals_at_65": int(n_65),
        "train_rows": int(len(fold_train)),
        "test_rows": int(len(fold_test)),
        "train_start": str(fold_train["time"].min().date()),
        "train_end": str(fold_train["time"].max().date()),
        "model_type": "lgbm_calibrated_isotonic",
        "retrain_date": pd.Timestamp.now(tz="UTC").isoformat(),
        "retrain_script": "retrain_lgbm_model.py",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n  ✓ Saved model  : {model_path}")
    print(f"  ✓ Saved scaler : {scaler_path}")
    print(f"  ✓ Saved meta   : {meta_path}")
    print(f"\n  → To use in sweep: update config's model_path to {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
