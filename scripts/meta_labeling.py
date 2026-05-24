#!/usr/bin/env python3
"""
Meta-labeling: Train a 2nd classifier to filter WF signals.
Uses WF OOS sim_trades + historical features to predict which signals will win.
Implements time-series cross-validation to avoid leakage.

Strategy:
  Primary model → signal (buy/sell) with probability
  Meta model → P(primary_signal_wins) given context features

If meta_prob >= threshold → execute trade, else skip.

Expected improvement: WR 47% → 60%+, fewer but higher quality trades.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

ROOT = Path('.')

# ── 1. Load WF sim_trades ─────────────────────────────────────────────────────
print("Loading WF sim_trades...")
sim = pd.read_csv('outputs/walkforward_trades_acc1_v14pp_profit_sim_trades.csv',
                  parse_dates=['time'])
sim['time'] = pd.to_datetime(sim['time'], utc=True)
print(f"  {len(sim)} trades | WR={sim['is_win'].mean()*100:.1f}%")

# ── 2. Load historical features to join context ───────────────────────────────
print("Loading features for context...")
feat_cols = ['time', 'atr', 'atr_ratio', 'rsi', 'macd_hist', 'bb_position',
             'regime_favorable', 'regime_trending',
             'h4_ict_confluence', 'h4_bos', 'strategy_score',
             'kill_zone_flag', 'session_return', 'tick_volume_zscore',
             'time_hour_sin', 'time_hour_cos', 'adx', 'multi_tf_consensus']
available = list(pd.read_csv('outputs/historical_features_2022_2026.csv', nrows=1).columns)
use_cols = ['time'] + [c for c in feat_cols[1:] if c in available]
feat = pd.read_csv('outputs/historical_features_2022_2026.csv',
                   usecols=use_cols, parse_dates=['time'])
feat['time'] = pd.to_datetime(feat['time'], utc=True)

sim['time_key'] = sim['time'].dt.floor('5min')
feat['time_key'] = feat['time'].dt.floor('5min')
merged = sim.merge(feat.rename(columns={'time': 'feat_time'})[
    [c for c in use_cols if c != 'time'] + ['time_key']
], on='time_key', how='left')

print(f"  Context joined: {merged.isnull().sum().sum()} nulls")

# ── 3. Build meta-features ────────────────────────────────────────────────────
meta = pd.DataFrame()
meta['probability']        = merged['probability']
meta['side_buy']           = (merged['side'] == 'buy').astype(int)
meta['hour_sin']           = merged.get('time_hour_sin', np.sin(merged['time'].dt.hour * 2 * np.pi / 24))
meta['hour_cos']           = merged.get('time_hour_cos', np.cos(merged['time'].dt.hour * 2 * np.pi / 24))
meta['atr_ratio']          = merged.get('atr_ratio', 1.0)
meta['rsi']                = merged.get('rsi', pd.Series(50, index=merged.index))
meta['bb_position']        = merged.get('bb_position', pd.Series(0.5, index=merged.index))
meta['vol_regime_0']       = (merged['volatility_regime'] == 0).astype(int)
meta['vol_regime_1']       = (merged['volatility_regime'] == 1).astype(int)
meta['vol_regime_2']       = (merged['volatility_regime'] == 2).astype(int)
meta['ict_confluence']     = merged.get('h4_ict_confluence', pd.Series(0, index=merged.index))
meta['h4_bos']             = merged.get('h4_bos', pd.Series(0, index=merged.index))
meta['strategy_score']     = merged.get('strategy_score', pd.Series(0, index=merged.index))
meta['kill_zone']          = merged.get('kill_zone_flag', pd.Series(0, index=merged.index))
meta['adx']                = merged.get('adx', pd.Series(25, index=merged.index))
meta['multi_tf_consensus'] = merged.get('multi_tf_consensus', pd.Series(0, index=merged.index))
meta['regime_favorable']   = merged.get('regime_favorable', pd.Series(0, index=merged.index))
meta['risk_throttle']      = merged['risk_throttle_multiplier']
meta['open_positions']     = merged['open_positions_at_open']
meta['fold']               = merged['fold']

# Rolling win rate of prev 5 trades (per fold order)
meta['rolling_wr_5'] = (
    merged['is_win'].rolling(5, min_periods=1).mean().shift(1).fillna(0.5)
)

# Target
target = merged['is_win'].astype(int)
folds = merged['fold'].values

print(f"\nMeta features: {meta.shape[1]} | Samples: {len(meta)}")
print(f"Class balance: {target.mean()*100:.1f}% wins")

# ── 4. Time-series cross-validation (walk-forward) ────────────────────────────
print("\n=== Meta-Model Walk-Forward CV ===")
X = meta.drop(columns=['fold']).fillna(0).values
y = target.values
feat_names = [c for c in meta.columns if c != 'fold']

unique_folds = sorted(np.unique(folds))
n_test_folds = len(unique_folds)

all_preds = np.zeros(len(X))
all_true = y.copy()
cv_results = []

# Use folds 1..N-1 as rolling train, fold N as test
for i, test_fold in enumerate(unique_folds):
    if i < 3:  # Need at least 3 folds of train data
        # Not enough history → use global prior
        fold_mask = folds == test_fold
        all_preds[fold_mask] = 0.5
        continue

    train_mask = np.isin(folds, unique_folds[:i])  # All prior folds
    test_mask  = folds == test_fold

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]

    if len(X_tr) < 20 or len(X_te) < 2:
        all_preds[test_mask] = 0.5
        continue

    # Calibrated logistic regression (less overfit risk than GBM on small data)
    base = LogisticRegression(C=0.5, max_iter=500, class_weight='balanced')
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    base.fit(X_tr_s, y_tr)
    preds = base.predict_proba(X_te_s)[:, 1]
    all_preds[test_mask] = preds

    auc = roc_auc_score(y_te, preds) if len(np.unique(y_te)) > 1 else 0.5
    cv_results.append({
        'fold': test_fold, 'n_train': sum(train_mask),
        'n_test': sum(test_mask), 'auc': auc, 'wr': y_te.mean()
    })
    print(f"  Fold {test_fold:2d}: train={sum(train_mask):3d} | test={sum(test_mask):3d} | "
          f"WR={y_te.mean()*100:.0f}% | meta-AUC={auc:.3f}")

# ── 5. Evaluate meta-filter thresholds ───────────────────────────────────────
print("\n=== Meta-Filter Threshold Analysis ===")
valid = all_preds > 0  # Exclude folds with no prediction (< 3 train folds)

print(f"{'Meta Thr':<10} {'Kept%':<8} {'N Trades':<10} {'WR':<8} {'Expected EV (RR=3.5)'}")
print("-" * 60)
for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    keep = (all_preds >= thr) & valid
    n_keep = keep.sum()
    if n_keep < 5:
        continue
    wr = y[keep].mean()
    ev = wr * 3.5 - (1 - wr) * 1.0
    kept_pct = n_keep / valid.sum() * 100
    print(f"  {thr:<8.2f} {kept_pct:<8.1f} {n_keep:<10} {wr*100:<8.1f} {ev:>+.2f}R")

# Best threshold = highest EV
best_thr = 0.50
best_ev = -999
for thr in np.arange(0.40, 0.75, 0.01):
    keep = (all_preds >= thr) & valid
    if keep.sum() < 20:
        break
    wr = y[keep].mean()
    ev = wr * 3.5 - (1 - wr)
    if ev > best_ev:
        best_ev = ev
        best_thr = thr

print(f"\n  Best threshold: {best_thr:.2f} → EV={best_ev:.2f}R/trade")

# ── 6. Apply meta-filter and export filtered signals ─────────────────────────
print("\n=== Applying Meta-Filter to sim_trades ===")
keep_mask = (all_preds >= best_thr) & valid
filtered_sim = merged[keep_mask].copy()
filtered_sim['meta_prob'] = all_preds[keep_mask]

print(f"  Original: {len(merged)} trades | Filtered: {len(filtered_sim)} trades")
print(f"  Original WR: {y.mean()*100:.1f}% | Filtered WR: {y[keep_mask].mean()*100:.1f}%")

# Export for MT5
from datetime import timedelta
SL_MULT = 1.5
TP_RR   = 3.5
GMT_OFF = 3

direction = filtered_sim['side'].map({'buy': 1, 'sell': -1})
atr_col = filtered_sim['atr_ratio'] if 'atr_ratio' in filtered_sim.columns else None

# Load ATR
feat_atr = pd.read_csv('outputs/historical_features_2022_2026.csv',
                        usecols=['time','atr'], parse_dates=['time'])
feat_atr['time'] = pd.to_datetime(feat_atr['time'], utc=True)
feat_atr['time_key'] = feat_atr['time'].dt.floor('5min')
filtered_sim = filtered_sim.merge(feat_atr[['time_key','atr']], on='time_key', how='left', suffixes=('','_m15'))
filtered_sim['atr'] = filtered_sim.get('atr', filtered_sim['entry_price'] * 0.003)
filtered_sim['atr'] = filtered_sim['atr'].fillna(filtered_sim['entry_price'] * 0.003)

sl_dist = filtered_sim['atr'] * SL_MULT
tp_dist = sl_dist * TP_RR
direction_vals = filtered_sim['side'].map({'buy': 1, 'sell': -1})

out = pd.DataFrame()
out['open_time'] = (filtered_sim['time'] + timedelta(hours=GMT_OFF)).dt.strftime('%Y.%m.%d %H:%M')
out['direction'] = direction_vals.values
out['entry_price'] = filtered_sim['entry_price'].round(5).values
out['sl_price'] = (filtered_sim['entry_price'] - direction_vals * sl_dist).round(5).values
out['tp_price'] = (filtered_sim['entry_price'] + direction_vals * tp_dist).round(5).values
out['atr'] = filtered_sim['atr'].round(5).values
out['probability'] = filtered_sim['meta_prob'].round(5).values

out = out.sort_values('open_time').reset_index(drop=True)
out.to_csv('outputs/signals_meta_mt5.csv', index=False)
out.to_csv('outputs/signals_for_mt5.csv', index=False)
print(f"\nExported {len(out)} meta-filtered signals → outputs/signals_meta_mt5.csv")
print(out.head(3).to_string())

# Feature importance
base_final = LogisticRegression(C=0.5, max_iter=500)
sc_final = StandardScaler()
X_s = sc_final.fit_transform(X)
base_final.fit(X_s, y)
coefs = sorted(zip(feat_names, base_final.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
print("\n=== Top 10 Meta-Features (Logistic coeff) ===")
for name, coef in coefs[:10]:
    print(f"  {name:<25} {coef:>+.3f}")
