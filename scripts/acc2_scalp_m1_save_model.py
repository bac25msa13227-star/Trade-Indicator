#!/usr/bin/env python3
"""
acc2_scalp_m1_save_model.py
============================
Train the M1 scalp dual model on the latest data and save artifacts
for live deployment, fully driven by `configs/live_acc2_scalp_m1.yaml`.

Usage:
  cd "Trade Indicator"
  source .venv2/bin/activate
  PYTHONPATH=src python scripts/acc2_scalp_m1_save_model.py

Training window: last TRAIN_SIZE M1 bars (~6 months).
Same hyperparameters and sample weighting as acc2_scalp_m1_wf.py.
"""
from __future__ import annotations
import json, os, pickle, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from xauusd_ai.config import load_settings
from xauusd_ai.features.scalp_dataset import apply_dynamic_sltp_labels, build_scalp_dataset
from xauusd_ai.features.scalp_features import SCALP_FEATURE_COLUMNS
from xauusd_ai.model.scalp_model import CalibratedDirModel, DualScalpModel

REPO     = Path(__file__).parent.parent
OUT_DIR  = REPO / "outputs"
CONFIG   = REPO / "configs" / "live_acc2_scalp_m1.yaml"

# ── Training defaults (config can override) ────────────────────────────────
TRAIN_SIZE   = 250_000
DEFAULT_DS_START = "2019-01-01"


def _make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=5,
        min_samples_leaf=50, l2_regularization=1.0, max_bins=63,
        early_stopping=True, validation_fraction=0.10,
        n_iter_no_change=20, random_state=42,
    )


def _train_dir(df, direction: int) -> CalibratedDirModel | None:
    """Train a calibrated direction model. Returns None if insufficient data."""
    label = "BUY" if direction == 1 else "SELL"
    sub   = df[df["expected_direction"] == direction].copy()
    n     = len(sub)
    if n < 2000:
        print(f"  [{label}] skip: only {n} rows", flush=True)
        return None

    X = sub[SCALP_FEATURE_COLUMNS].fillna(0).values
    y = sub["target"].values
    pos = int(y.sum()); neg = n - pos
    if pos < 200 or neg < 200:
        print(f"  [{label}] skip: pos={pos} neg={neg}", flush=True)
        return None

    print(f"  [{label}] n={n:,}  tp_rate={pos/n*100:.1f}%", flush=True)

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    # Class weight — balance imbalanced label
    cw = np.where(y == 1, float(neg) / float(pos), 1.0).astype(float)
    # Time decay — recent bars more important (half-life = 30% of window)
    decay_half = n * 0.30
    tw = np.exp(np.log(2) * np.arange(n) / decay_half)
    tw /= tw.mean()
    sw = cw * tw; sw /= sw.mean()

    model = _make_model()
    model.fit(X_s, y, sample_weight=sw)

    # Isotonic calibration on last 15%
    val_size = max(int(n * 0.15), 1000)
    try:
        X_val  = X_s[-val_size:]; y_val = y[-val_size:]
        raw_p  = model.predict_proba(X_val)[:, 1]
        iso    = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw_p, y_val)
        print(f"  [{label}] ✓ calibrated (val_size={val_size:,})", flush=True)
        return CalibratedDirModel(
            base_model = model,
            isotonic   = iso,
            scaler     = scaler,
            direction  = direction,
        )
    except Exception as e:
        print(f"  [{label}] calibration failed ({e}), using raw model", flush=True)
        # Build a trivial identity isotonic (no-op calibration)
        raw_all = model.predict_proba(X_s[-500:])[:, 1]
        iso_id  = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso_id.fit(raw_all, y[-500:])
        return CalibratedDirModel(model, iso_id, scaler, direction)


def _atomic_pickle(obj, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(obj, fh, protocol=4)
        os.replace(tmp, str(dest))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main() -> None:
    settings = load_settings(CONFIG)
    train_size = int(getattr(settings.training, "walkforward_train_size", TRAIN_SIZE) or TRAIN_SIZE)
    thr_buy = float(settings.strategy.signal_threshold)
    thr_sell = float(settings.strategy.signal_threshold)
    label_horizon = int(getattr(settings.training, "sltp_label_max_horizon", settings.training.label_horizon) or 8)
    setup_label_mode = str(getattr(settings.training, "setup_label_mode", "fixed") or "fixed").strip().lower()
    dataset_start = str(getattr(settings.training, "dataset_start_date", DEFAULT_DS_START) or DEFAULT_DS_START)
    base_sl_atr = float(getattr(settings.risk, "stop_loss_atr_multiple", 0.8))
    base_tp_rr = float(getattr(settings.risk, "take_profit_rr", 1.5))

    print("=" * 80)
    print("  ACC2 M1 Scalp — Train & Save Model Artifacts")
    print(f"  Train window: last {train_size:,} M1 bars (~6 months)")
    print(f"  Label mode: {setup_label_mode}")
    print(f"  Label: TP={base_tp_rr}R within {label_horizon} M1 bars  |  SL={base_sl_atr}×ATR5")
    print(f"  Thresholds: BUY≥{thr_buy}  SELL≥{thr_sell}")
    print(f"  Dynamic label: {'ON' if settings.training.dynamic_sltp_label_enabled else 'OFF'}")
    print("=" * 80)

    # ── 1. Build dataset ──────────────────────────────────────────────
    print(f"\n[1] Loading M1 scalp dataset from {dataset_start}…", flush=True)
    ds = build_scalp_dataset(
        start_date  = dataset_start,
        sl_atr_mult = base_sl_atr,
        tp_rr       = base_tp_rr,
        max_horizon = label_horizon,
        setup_label_mode = setup_label_mode,
    )
    total = len(ds)
    print(f"    {total:,} M1 rows  "
          f"[{str(ds['time'].iloc[0])[:10]} → {str(ds['time'].iloc[-1])[:10]}]")

    if total < train_size:
        print(f"ERROR: need {train_size:,} rows, have {total:,}. Aborting.")
        sys.exit(1)

    # ── 2. Take last TRAIN_SIZE bars as training window ────────────────
    train_df   = ds.iloc[-train_size:].copy().reset_index(drop=True)
    if settings.training.dynamic_sltp_label_enabled and setup_label_mode == "fixed":
        print("\n[2a] Re-labeling with dynamic SL/TP policy...", flush=True)
        train_df = apply_dynamic_sltp_labels(
            train_df,
            settings=settings,
            max_horizon=label_horizon,
        )
    train_end  = str(train_df["time"].iloc[-1])[:10]
    train_start= str(train_df["time"].iloc[0])[:10]
    print(f"\n[2] Training on {len(train_df):,} rows "
          f"[{train_start} → {train_end}]", flush=True)
    print(f"    Label positive rate (TP hit): {train_df['target'].mean()*100:.1f}%")
    print(f"    BUY rows:  {(train_df['expected_direction']==1).sum():,}")
    print(f"    SELL rows: {(train_df['expected_direction']==-1).sum():,}")

    # ── 3. Train direction models ─────────────────────────────────────
    print(f"\n[3] Training BUY/SELL models…", flush=True)
    buy_model  = _train_dir(train_df,  1)
    sell_model = _train_dir(train_df, -1)

    if buy_model is None and sell_model is None:
        print("ERROR: both models failed. Aborting.")
        sys.exit(1)

    # ── 4. Build DualScalpModel wrapper ───────────────────────────────
    dual = DualScalpModel(
        buy_model       = buy_model,
        sell_model      = sell_model,
        feature_columns = SCALP_FEATURE_COLUMNS,
        thr_buy         = thr_buy,
        thr_sell        = thr_sell,
        train_end       = train_end,
    )
    print(f"\n[4] Built: {dual}")

    # Quick sanity: score 100 random test rows
    sample = train_df.sample(min(200, len(train_df)), random_state=0)
    X_samp = sample[SCALP_FEATURE_COLUMNS].fillna(0).values
    dirs   = sample["expected_direction"].values
    proba  = dual.score_batch(X_samp, dirs)
    print(f"    Sanity check — mean proba={proba.mean():.3f}  "
          f"std={proba.std():.3f}  "
          f">=0.55: {(proba>=0.55).mean()*100:.0f}%")

    # ── 5. Save artifacts ─────────────────────────────────────────────
    print(f"\n[5] Saving artifacts to {OUT_DIR}/…", flush=True)

    model_path = REPO / settings.app.model_path
    scaler_path = REPO / settings.app.scaler_path
    meta_path = REPO / settings.app.model_meta_path
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # model.pkl — the DualScalpModel
    _atomic_pickle(dual, model_path)
    print(f"    ✓ {model_path.name}  ({model_path.stat().st_size/1024:.0f} KB)")

    # scaler.pkl — identity transformer (scaling is internal to each CalibratedDirModel)
    identity_scaler = FunctionTransformer(func=None, validate=False)
    _atomic_pickle(identity_scaler, scaler_path)
    print(f"    ✓ {scaler_path.name}  (identity — scaling is internal)")

    # model_meta.json
    meta = {
        "model_type":       "DualScalpM1",
        "feature_columns":  SCALP_FEATURE_COLUMNS,
        "decision_threshold": thr_sell,   # lower gate = SELL threshold
        "thr_buy":          thr_buy,
        "thr_sell":         thr_sell,
        "sl_atr_mult":      base_sl_atr,
        "tp_rr":            base_tp_rr,
        "max_horizon":      label_horizon,
        "train_size":       train_size,
        "train_start":      train_start,
        "train_end":        train_end,
        "buy_model_ok":     buy_model  is not None,
        "sell_model_ok":    sell_model is not None,
        "feature_count":    len(SCALP_FEATURE_COLUMNS),
        "setup_label_mode": setup_label_mode,
        "setup_exit_enabled": bool(settings.risk.setup_exit_enabled),
        "setup_exit_scale": float(getattr(settings.risk, "setup_exit_scale", 1.0)),
        "dynamic_sltp_label_enabled": bool(settings.training.dynamic_sltp_label_enabled),
        "dynamic_sltp_enabled": bool(settings.risk.dynamic_sltp_enabled),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"    ✓ {meta_path.name}")

    print(f"\n{'='*80}")
    print(f"  DONE — model ready for paper trade and live inference.")
    print(f"  Train period: {train_start} → {train_end}")
    print(f"  Next step: run acc2_scalp_m1_paper.py to validate on out-of-sample bars")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
