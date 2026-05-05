#!/usr/bin/env python3
"""
decision_parity_test.py — Layer 2 Pre-deploy Gate
===================================================
Kiểm tra: cùng 1 input bar → live engine và replay/WF path ra CÙNG 1 quyết định.

Nếu test này FAIL → live sẽ đưa ra quyết định khác WF trên cùng data → GAP.

Hai path được so sánh:
  PATH A (Live engine path):
      ModelTrainer.load_artifacts() → trainer.score_live_row(row)
      Đây chính xác là code live bot gọi trong orchestrator.py

  PATH B (Replay/WF path):
      joblib.load(model.pkl) + joblib.load(scaler.pkl) + feat_mask từ meta
      model.predict_proba(scaler.transform(row[FEATURE_COLUMNS])[:, feat_mask])
      Đây là code replay_last_30days.py và WF script dùng

Nếu confidence từ A và B khác nhau > TOL → BUG → deploy block.

Exit codes:
  0 = PASS
  1 = FAIL (parity broken)
  2 = ERROR (model/data không load được)

Usage:
  python scripts/decision_parity_test.py
  python scripts/decision_parity_test.py --config configs/live_acc1.yaml --n-rows 200
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.features.dataset import FEATURE_COLUMNS, prepare_training_dataset
from xauusd_ai.model.trainer import ModelTrainer
from xauusd_ai.strategies.hybrid import HybridStrategy

CONFIDENCE_TOL = 1e-6   # max delta allowed between path A and B
SIDE_TOL       = 0      # side must be identical (0 tolerance)


def run(config_path: Path, n_rows: int) -> int:
    print("=" * 78)
    print("  DECISION PARITY TEST  (Layer 2)")
    print(f"  Config  : {config_path}")
    print(f"  N rows  : {n_rows} (from most recent data)")
    print("=" * 78)

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}", file=sys.stderr)
        return 2

    settings = load_settings(config_path)

    # ── Check model artifacts ──
    model_path  = Path(settings.app.model_path)
    scaler_path = Path(settings.app.scaler_path)
    meta_path   = Path(settings.app.model_meta_path)
    for p in (model_path, scaler_path, meta_path):
        if not p.exists():
            print(f"❌ Missing artifact: {p}", file=sys.stderr)
            return 2
    print(f"\n[1/4] Model artifacts OK: {model_path.name}")

    # ── PATH A: live engine path ──
    print("[2/4] Loading PATH A (live engine: ModelTrainer.load_artifacts)...")
    trainer_a = ModelTrainer(settings)
    trainer_a.load_artifacts()
    print(f"      feature_columns : {len(trainer_a.feature_columns)}")
    print(f"      feature_mask    : {trainer_a._feature_mask.sum() if trainer_a._feature_mask is not None else 'None (use all)'}")
    print(f"      decision_thresh : {trainer_a.decision_threshold:.4f}")

    # ── PATH B: replay/WF path (raw joblib) ──
    print("[3/4] Loading PATH B (replay/WF: raw joblib + feature_mask from meta)...")
    model_b  = joblib.load(model_path)
    scaler_b = joblib.load(scaler_path)
    meta_b   = json.loads(meta_path.read_text(encoding="utf-8"))
    feat_mask_b = None
    if isinstance(meta_b.get("feature_mask"), list) and meta_b["feature_mask"]:
        feat_mask_b = np.asarray(meta_b["feature_mask"], dtype=bool)
    thr_b = float(
        getattr(settings.strategy, "signal_threshold", None)
        or meta_b.get("decision_threshold")
        or getattr(settings.risk, "min_confidence", 0.70)
    )
    n_active = feat_mask_b.sum() if feat_mask_b is not None else len(FEATURE_COLUMNS)
    print(f"      FEATURE_COLUMNS : {len(FEATURE_COLUMNS)}")
    print(f"      feature_mask    : {n_active if feat_mask_b is not None else 'None (use all)'}")
    print(f"      decision_thresh : {thr_b:.4f}")

    # ── Assert artifact IDs match (same threshold → same model generation) ──
    if abs(trainer_a.decision_threshold - thr_b) > 1e-4:
        print(f"⚠️  threshold mismatch: path_A={trainer_a.decision_threshold:.4f}  path_B={thr_b:.4f}")
        print("   This is OK if trainer uses runtime override. Continuing...")

    # ── Build test window ──
    print("[4/4] Building feature dataset...")
    data_service = MarketDataService(settings)
    frames = data_service.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)
    strategy = HybridStrategy(settings)
    full_ds = prepare_training_dataset(settings, frames, strategy)

    # Use most recent n_rows bars
    window = full_ds.tail(n_rows).copy().reset_index(drop=True)
    print(f"      Window: {len(window)} rows  ({window['time'].min()} → {window['time'].max()})")

    # ── Run PATH A: score_live_row row-by-row ──
    proba_a = np.empty(len(window), dtype=float)
    pred_a  = np.empty(len(window), dtype=int)
    for i in range(len(window)):
        row_df = window.iloc[[i]].copy()
        result = trainer_a.score_live_row(row_df)
        proba_a[i] = result["probability"]
        pred_a[i]  = result["prediction"]

    # ── Run PATH B: vectorized predict_proba ──
    X_b = scaler_b.transform(window[FEATURE_COLUMNS].values)
    if feat_mask_b is not None and len(feat_mask_b) == X_b.shape[1]:
        X_b = X_b[:, feat_mask_b]
    proba_b = model_b.predict_proba(X_b)[:, 1]
    pred_b  = (proba_b >= thr_b).astype(int)

    # ── Compare ──
    delta      = np.abs(proba_a - proba_b)
    max_delta  = delta.max()
    mean_delta = delta.mean()
    n_conf_fail = (delta > CONFIDENCE_TOL).sum()
    n_pred_fail = (pred_a != pred_b).sum()

    print()
    print("─" * 78)
    print("  PARITY RESULTS")
    print("─" * 78)
    print(f"  Rows tested          : {len(window)}")
    print(f"  Max |confidence_A - confidence_B| : {max_delta:.2e}  (tol={CONFIDENCE_TOL:.0e})")
    print(f"  Mean delta           : {mean_delta:.2e}")
    print(f"  Rows with conf delta > tol         : {n_conf_fail}")
    print(f"  Rows with pred mismatch            : {n_pred_fail}")
    print()

    # Sample mismatches if any
    if n_conf_fail > 0 or n_pred_fail > 0:
        mismatch_idx = np.where((delta > CONFIDENCE_TOL) | (pred_a != pred_b))[0]
        print(f"  First {min(10, len(mismatch_idx))} mismatching rows:")
        for idx in mismatch_idx[:10]:
            print(
                f"    row {idx:4d}  time={window['time'].iloc[idx]}  "
                f"A={proba_a[idx]:.8f}/pred={pred_a[idx]}  "
                f"B={proba_b[idx]:.8f}/pred={pred_b[idx]}  "
                f"delta={delta[idx]:.2e}"
            )
        print()

    # ── Verdict ──
    fail_reasons: list[str] = []
    if n_conf_fail > 0:
        fail_reasons.append(
            f"confidence divergence: {n_conf_fail}/{len(window)} rows exceed tol {CONFIDENCE_TOL:.0e} "
            f"(max delta {max_delta:.2e})"
        )
    if n_pred_fail > 0:
        fail_reasons.append(
            f"signal mismatch: {n_pred_fail}/{len(window)} rows predict differently (A vs B)"
        )

    # Feature alignment check: assert both paths use same feature list order
    feat_cols_a = trainer_a.feature_columns
    if feat_cols_a != list(FEATURE_COLUMNS):
        fail_reasons.append(
            f"feature column order mismatch: trainer has {len(feat_cols_a)} cols, "
            f"FEATURE_COLUMNS has {len(FEATURE_COLUMNS)} cols"
        )

    if fail_reasons:
        print("  RESULT: ❌ FAIL — Live and WF/replay paths diverge!")
        for r in fail_reasons:
            print(f"    • {r}")
        print()
        print("  Root cause hints:")
        print("    1. Check if _aligned_feature_frame pads missing cols differently")
        print("    2. Check if feature_mask in meta matches _feature_mask in trainer")
        print("    3. Check if scaler was saved AFTER being fit to fold_train (WF)")
        print("       vs being fit to full training data (live)")
        print("=" * 78)
        return 1

    print("  RESULT: ✅ PASS — Both paths produce identical decisions on same data.")
    print(f"           confidence delta ≤ {max_delta:.2e}  |  0 prediction mismatches")
    print("=" * 78)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/live_acc1.yaml"))
    ap.add_argument("--n-rows", type=int, default=500,
                    help="Number of most-recent rows to test (default: 500)")
    args = ap.parse_args()
    return run(args.config, args.n_rows)


if __name__ == "__main__":
    sys.exit(main())
