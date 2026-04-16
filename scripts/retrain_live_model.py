"""
Live Model Retrain — 1-Fold Schedule
======================================
Retrain the ICT+Wyckoff VotingClassifier ensemble using the most recent
WF-equivalent training window (30K bars train, 4K bars holdout), then
atomically replace the live model files if quality gate passes.

Usage:
    python scripts/retrain_live_model.py configs/live_acc1.yaml
    python scripts/retrain_live_model.py configs/live_acc2.yaml --dry-run
    python scripts/retrain_live_model.py configs/live_acc1.yaml --auc-gate 0.55

Schedule (cron — every 6 weeks, Sunday 02:00 UTC):
    0 2 * * 0  [ $(( $(date +%s) / 86400 / 42 )) != $(cat /tmp/.retrain_epoch_acc1 2>/dev/null) ] && \
        python /path/to/scripts/retrain_live_model.py /path/to/configs/live_acc1.yaml && \
        echo $(( $(date +%s) / 86400 / 42 )) > /tmp/.retrain_epoch_acc1

After retrain, restart the live bot to load the new model:
    docker restart trade-indicator-live-acc1-1
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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.features.dataset import FEATURE_COLUMNS, prepare_training_dataset
from xauusd_ai.strategies.hybrid import HybridStrategy

# ── WF-identical window sizes (M15 bars) ──────────────────────────────────────
TRAIN_BARS = 30_000   # ~13 months of M15 data
TEST_BARS  =  4_000   # ~1.5 months (= 1 WF fold holdout)

THRESHOLD_MIN  = 0.35
THRESHOLD_MAX  = 0.80
THRESHOLD_STEP = 0.01
PREC_FLOOR     = 0.40   # minimum precision required for threshold to count


def _atomic_write_pickle(obj: object, dest: Path) -> None:
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
    parser = argparse.ArgumentParser(description="Retrain live model on latest 1-fold window")
    parser.add_argument("config", help="Path to live config YAML (e.g. configs/live_acc1.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Train and evaluate but do NOT save model files")
    parser.add_argument("--auc-gate", type=float, default=0.54, help="Min ROC-AUC to accept new model (default: 0.54)")
    parser.add_argument("--no-cache", action="store_true", help="Skip parquet dataset cache")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}")
        return 1

    print("=" * 65)
    print("  LIVE MODEL RETRAIN — 1-FOLD SCHEDULE")
    print("=" * 65)
    print(f"  Config    : {cfg_path}")
    print(f"  Train bars: {TRAIN_BARS:,}  (~13 mo M15)")
    print(f"  Hold bars : {TEST_BARS:,}   (~1.5 mo M15 = 1 WF fold)")
    print(f"  AUC gate  : {args.auc_gate}")
    print(f"  Dry run   : {args.dry_run}")
    print()

    settings = load_settings(cfg_path)
    data_service = MarketDataService(settings)
    strategy = HybridStrategy(settings)

    # ── 1. Load multi-TF data ─────────────────────────────────────────────────
    print("[1/5] Loading multi-timeframe data from CSV...")
    t0 = time.time()
    frames = data_service.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)
    exec_tf = settings.market.execution_timeframe
    print(f"      {exec_tf}:{len(frames[exec_tf]):,}  H4:{len(frames['H4']):,}  "
          f"H1:{len(frames['H1']):,}  D1:{len(frames['D1']):,}  ({time.time()-t0:.1f}s)")

    # ── 2. Build full dataset ─────────────────────────────────────────────────
    print(f"\n[2/5] Building full dataset ({len(FEATURE_COLUMNS)} features)...")
    t0 = time.time()

    settings_full = settings.model_copy(deep=True)
    settings_full.training.train_start_date = None
    settings_full.training.train_end_date   = None
    settings_full.training.test_start_date  = None
    settings_full.training.test_end_date    = None

    cache_dir  = Path("outputs/.wf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    _hash_src  = cfg_path.read_text(encoding="utf-8") + str(len(frames[exec_tf]))
    _cache_key = hashlib.sha256(_hash_src.encode()).hexdigest()[:12]
    _cache_path = cache_dir / f"retrain_{cfg_path.stem}_{_cache_key}.parquet"

    full_ds: pd.DataFrame
    if not args.no_cache and _cache_path.exists():
        try:
            full_ds = pd.read_parquet(_cache_path)
            print(f"      Cache hit: {_cache_path.name}")
        except Exception as _e:
            print(f"      Cache read failed ({_e}), rebuilding...")
            full_ds = prepare_training_dataset(settings_full, frames, strategy)
    else:
        full_ds = prepare_training_dataset(settings_full, frames, strategy)
        if not args.no_cache:
            try:
                full_ds.to_parquet(_cache_path, index=False)
                print(f"      Cached to: {_cache_path.name}")
            except Exception:
                pass

    n_total = len(full_ds)
    print(f"      Rows  : {n_total:,}  "
          f"({full_ds['time'].min().date()} → {full_ds['time'].max().date()})")
    print(f"      Labels: {full_ds['target'].mean():.1%} positive  ({time.time()-t0:.1f}s)")

    if n_total < TRAIN_BARS + TEST_BARS:
        print(f"\nERROR: dataset too small ({n_total} < {TRAIN_BARS + TEST_BARS} required)")
        return 1

    # ── 3. Split: last 4K = holdout, prior 30K = train ───────────────────────
    print(f"\n[3/5] Splitting data (train={TRAIN_BARS:,} | holdout={TEST_BARS:,})...")
    holdout_start = n_total - TEST_BARS
    train_start   = max(0, holdout_start - TRAIN_BARS)
    fold_train = full_ds.iloc[train_start:holdout_start]
    fold_test  = full_ds.iloc[holdout_start:]
    print(f"      Train : {fold_train['time'].min().date()} → {fold_train['time'].max().date()}  ({len(fold_train):,} rows)")
    print(f"      Holdout: {fold_test['time'].min().date()} → {fold_test['time'].max().date()}  ({len(fold_test):,} rows)")

    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(fold_train[FEATURE_COLUMNS])
    X_te    = scaler.transform(fold_test[FEATURE_COLUMNS])
    y_tr    = fold_train["target"].values
    y_te    = fold_test["target"].values

    # ── Sample weights: 2× positive boost + time decay ────────────────────────
    _pos_c = int(y_tr.sum())
    _neg_c = int(len(y_tr) - _pos_c)
    if _pos_c > 10 and _neg_c > 10:
        _pw     = 2.0 * _neg_c / _pos_c
        _cls_w  = np.where(y_tr == 1, _pw, 1.0).astype(float)
        _n      = len(y_tr)
        _decay  = _n * 0.4
        _time_w = np.exp(np.log(2) * np.arange(_n) / _decay)
        _time_w /= _time_w.mean()
        _sw_tr  = (_cls_w * _time_w).astype(float)
        _sw_tr  /= _sw_tr.mean()
    else:
        _sw_tr = None

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("\n[4/5] Training VotingClassifier ensemble (WF-identical hyperparams)...")
    t0 = time.time()

    # Threshold search (fast mode: single temporal split)
    _n_tr = len(y_tr)
    _thr_hgb = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.02, max_depth=6,
        min_samples_leaf=25, l2_regularization=1.0,
        max_bins=128, class_weight=None,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=42,
    )
    _ts, _vs = int(_n_tr * 0.70), int(_n_tr * 0.70)
    _sw_sub = _sw_tr[:_ts] if _sw_tr is not None else None
    _thr_hgb.fit(X_tr[:_ts], y_tr[:_ts], sample_weight=_sw_sub)
    _v_proba = _thr_hgb.predict_proba(X_tr[_vs:])[:, 1]
    _y_v = y_tr[_vs:]
    best_thr, best_score = float(THRESHOLD_MAX), -float("inf")
    safe_thr, safe_prec  = float(THRESHOLD_MAX), -1.0
    for thr in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP):
        preds  = (_v_proba >= thr).astype(int)
        n_pred = int(preds.sum())
        if n_pred < 3:
            continue
        prec = precision_score(_y_v, preds, zero_division=0)
        rec  = recall_score(_y_v, preds, zero_division=0)
        if rec < 0.05:
            continue
        if prec > safe_prec:
            safe_prec = prec
            safe_thr  = float(thr)
        if prec < PREC_FLOOR:
            continue
        score = prec * np.sqrt(rec)
        if score > best_score:
            best_score, best_thr = score, float(thr)
    if best_score == -float("inf"):
        best_thr = safe_thr
    print(f"      Optimal threshold: {best_thr:.2f}")

    # Feature selection: drop bottom 30% by importance
    _scout = RandomForestClassifier(
        n_estimators=80, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    _scout.fit(X_tr, y_tr, sample_weight=_sw_tr)
    _imp      = _scout.feature_importances_
    _imp_thr  = np.percentile(_imp, 30)
    _feat_mask = _imp >= _imp_thr
    if _feat_mask.sum() < 10:
        _feat_mask = np.ones(len(_imp), dtype=bool)
    X_tr_sel = X_tr[:, _feat_mask]
    X_te_sel = X_te[:, _feat_mask]
    print(f"      Features kept    : {_feat_mask.sum()} / {len(_feat_mask)} (dropped bottom 30%)")

    # VotingClassifier (same weights as WF: HGB×3 + RF×2 + ET×1)
    model = VotingClassifier(
        estimators=[
            ("hgb", HistGradientBoostingClassifier(
                max_iter=1000, learning_rate=0.01, max_depth=7,
                min_samples_leaf=20, l2_regularization=1.0,
                max_bins=128, class_weight=None,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=40, random_state=42,
            )),
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=15,
                max_features="sqrt", class_weight="balanced",
                n_jobs=-1, random_state=42,
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=200, max_depth=14, min_samples_leaf=10,
                max_features="sqrt", class_weight="balanced",
                n_jobs=-1, random_state=42,
            )),
        ],
        voting="soft",
        weights=[3, 2, 1],
    )
    model.fit(X_tr_sel, y_tr, sample_weight=_sw_tr)
    print(f"      Train time       : {time.time()-t0:.1f}s")

    # ── 5. Evaluate on holdout ────────────────────────────────────────────────
    print("\n[5/5] Evaluating on holdout...")
    test_proba = model.predict_proba(X_te_sel)[:, 1]
    test_preds = (test_proba >= best_thr).astype(int)

    auc  = roc_auc_score(y_te, test_proba) if len(np.unique(y_te)) > 1 else 0.5
    prec = precision_score(y_te, test_preds, zero_division=0)
    rec  = recall_score(y_te, test_preds, zero_division=0)
    f1   = f1_score(y_te, test_preds, zero_division=0)
    acc  = accuracy_score(y_te, test_preds)
    n_sig = int(test_preds.sum())

    print()
    print(f"  ROC-AUC   : {auc:.4f}  (gate={args.auc_gate})")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Signals   : {n_sig} / {len(test_preds)} bars")
    print(f"  Threshold : {best_thr:.2f}")
    print()

    passed = auc >= args.auc_gate
    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"  Quality gate: {status}  (auc={auc:.4f} {'≥' if passed else '<'} {args.auc_gate})")

    if not passed:
        print("\n  Model NOT saved — quality gate failed.")
        print("  Tip: lower --auc-gate or investigate data quality.")
        return 2

    if args.dry_run:
        print("\n  Dry-run: model NOT saved (remove --dry-run to save).")
        return 0

    # ── Save model artifacts ──────────────────────────────────────────────────
    model_path  = Path(settings.app.model_path)
    scaler_path = Path(settings.app.scaler_path)
    meta_path   = Path(settings.app.model_meta_path)

    # Backup existing model before overwrite
    for _p in [model_path, scaler_path]:
        if _p.exists():
            _bak = _p.with_suffix(_p.suffix + ".bak")
            import shutil
            shutil.copy2(_p, _bak)

    _atomic_write_pickle(model, model_path)
    _atomic_write_pickle(scaler, scaler_path)

    meta = {
        "decision_threshold": float(best_thr),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_mask": _feat_mask.tolist(),
        "roc_auc": round(float(auc), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1": round(float(f1), 6),
        "train_rows": int(len(fold_train)),
        "test_rows": int(len(fold_test)),
        "train_start": str(fold_train["time"].min().date()),
        "train_end": str(fold_train["time"].max().date()),
        "retrain_date": pd.Timestamp.now(tz="UTC").isoformat(),
        "retrain_script": "retrain_live_model.py",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n  Saved model  : {model_path}")
    print(f"  Saved scaler : {scaler_path}")
    print(f"  Saved meta   : {meta_path}")
    print()
    print("  IMPORTANT: restart the live bot to load the new model:")
    print(f"    docker restart trade-indicator-live-acc1-1  # (or acc2)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
