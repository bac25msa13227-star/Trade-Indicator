#!/usr/bin/env python3
"""
Diagnose rrlabel fold predictions after build.
Compares old model (target=1 at 16.8%) vs new model (realized_rr>0 at 37.6%).
"""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from wf_daily_reset_optimizer import load_or_build_dataset, build_fold_predictions

ROOT = Path('.')
config = ROOT / 'configs/acc1_v14pp_profit.yaml'
full_ds = load_or_build_dataset(config, cache=True)

print("=" * 70)
print("OLD MODEL (target = SL/TP race at 3.5×ATR, pos_rate=16.8%)")
print("=" * 70)
old_folds = build_fold_predictions(config, full_ds, use_cache=True, test_start='2024-08-14', use_realized_rr_label=False)
old_thrs = [f['threshold'] for f in old_folds]
old_aucs = [f['roc_auc'] for f in old_folds]
old_precs = [f['precision'] for f in old_folds]
print(f"Folds: {len(old_folds)}")
print(f"Threshold: min={min(old_thrs):.3f}, max={max(old_thrs):.3f}, median={np.median(old_thrs):.3f}")
print(f"AUC: avg={np.mean(old_aucs):.4f} ± {np.std(old_aucs):.4f}")
print(f"Precision: avg={np.mean(old_precs):.4f} ± {np.std(old_precs):.4f}")

# All probabilities from old model
all_probs_old = pd.concat([f['frame']['probability'] for f in old_folds])
print(f"\nProbability distribution (old model, min_conf=0.83 = p83 gate):")
for thr in [0.50, 0.60, 0.65, 0.70, 0.74, 0.78, 0.83]:
    cnt = (all_probs_old >= thr).sum()
    pct = cnt / len(all_probs_old) * 100
    print(f"  prob>={thr:.2f}: {cnt:,} rows ({pct:.2f}%)")

print()
print("=" * 70)
print("NEW MODEL (target = realized_rr>0, pos_rate=37.6%)")
print("=" * 70)
new_folds = build_fold_predictions(config, full_ds, use_cache=True, test_start='2024-08-14', use_realized_rr_label=True)
new_thrs = [f['threshold'] for f in new_folds]
new_aucs = [f['roc_auc'] for f in new_folds]
new_precs = [f['precision'] for f in new_folds]
print(f"Folds: {len(new_folds)}")
print(f"Threshold: min={min(new_thrs):.3f}, max={max(new_thrs):.3f}, median={np.median(new_thrs):.3f}")
print(f"AUC: avg={np.mean(new_aucs):.4f} ± {np.std(new_aucs):.4f}")
print(f"Precision: avg={np.mean(new_precs):.4f} ± {np.std(new_precs):.4f}")

all_probs_new = pd.concat([f['frame']['probability'] for f in new_folds])
print(f"\nProbability distribution (new model):")
for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.74, 0.78, 0.83]:
    cnt = (all_probs_new >= thr).sum()
    pct = cnt / len(all_probs_new) * 100
    print(f"  prob>={thr:.2f}: {cnt:,} rows ({pct:.2f}%)")

# Analyze signal quality at different thresholds for new model
print("\nSignal quality analysis (new model):")
all_frames_new = pd.concat([f['frame'] for f in new_folds], ignore_index=True)
print(f"  Total test rows: {len(all_frames_new):,}")
print(f"  Columns: {list(all_frames_new.columns[:10])} ...")

for thr in [0.55, 0.60, 0.65, 0.70, 0.74, 0.78, 0.83]:
    sub = all_frames_new[all_frames_new['probability'] >= thr]
    if len(sub) == 0:
        print(f"  thr={thr:.2f}: NO SIGNALS")
        continue
    wr = sub['realized_rr'].gt(0).mean() if 'realized_rr' in sub.columns else float('nan')
    avg_rr = sub['realized_rr'].mean() if 'realized_rr' in sub.columns else float('nan')
    theo_per_day = wr * avg_rr * 0.15 * 200 * (len(sub) / all_frames_new['time'].nunique()) if 'time' in all_frames_new.columns else float('nan')
    print(f"  thr={thr:.2f}: n={len(sub):,} ({len(sub)/len(all_frames_new)*100:.1f}%), WR={wr*100:.1f}%, avg_rr={avg_rr:.4f}, theo/day={theo_per_day:.2f}")

print("\nDone!")
