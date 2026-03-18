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

from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.dataset import prepare_training_dataset, FEATURE_COLUMNS
from xauusd_ai.strategies.hybrid import HybridStrategy

import functools
# Log path derived from first argv (config) so ACC1 and ACC2 don't overwrite each other
_cfg_stem = Path(sys.argv[1]).stem if len(sys.argv) > 1 else "ict_wyckoff"
_log_path = Path(f"outputs/walkforward_log_{_cfg_stem}.txt")
_log_path.parent.mkdir(parents=True, exist_ok=True)
_log_file = _log_path.open("w", encoding="utf-8")

_orig_print = print
def print(*args, **kwargs):  # noqa: A001
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)
    _orig_print(*args, file=_log_file, **{k: v for k, v in kwargs.items() if k != "file"})
    _log_file.flush()


def _compute_max_drawdown(equity_curve: list) -> float:
    """Return max drawdown fraction (0.0–1.0) from a list of equity values."""
    peak, max_dd = equity_curve[0], 0.0
    for v in equity_curve[1:]:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


print("=" * 70)
print("  WALK-FORWARD VALIDATION — ICT + WYCKOFF + NEWS (33 features)")
print("=" * 70)
print()

# ── config ──────────────────────────────────────────────────────────────────
CONFIG    = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/train_ict_wyckoff_2022_2026.yaml")
MAX_FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else None    # None = no limit
settings = load_settings(CONFIG)

# Walk-forward window parameters  (M15: 96 bars/day  ~252 trading days/year)
TRAIN_BARS = 20_000   # ~208 trading days = ~7 months
TEST_BARS  =  4_000   # ~42  trading days = ~1.5 months
STEP_BARS  =  4_000   # slide ~1.5 months at a time

THRESHOLD_MIN   = settings.training.threshold_min        # 0.45
THRESHOLD_MAX   = settings.training.threshold_max        # 0.80
THRESHOLD_STEP  = settings.training.threshold_step       # 0.01
PREC_FLOOR      = settings.training.min_precision_floor  # now 0.60 (from config)

# ── Balance & RR sweep settings ─────────────────────────────────────────────
STARTING_BALANCE = 200.0                           # USD khởi đầu
RISK_PCT         = settings.risk.risk_per_trade    # rủi ro/lệnh (0.0065 = 0.65%)
RR_SWEEP         = [1.8, 2.0, 2.2, 2.5, 3.0, 3.5] # TP:SL ratios cần đánh giá

print(f"  Config  : {CONFIG}")
print(f"  Features: {len(FEATURE_COLUMNS)}  (D1:1 H4:9 H1:3 M15:15 News:5)")
print(f"  Train   : {TRAIN_BARS:,} bars (~1 yr M15)")
print(f"  Test    : {TEST_BARS:,} bars (~3 mo M15)")
print(f"  Step    : {STEP_BARS:,} bars (~3 mo slide)")
print(f"  Balance : ${STARTING_BALANCE:.0f} khởi đầu | Rủi ro {RISK_PCT:.2%}/lệnh")
print(f"  RR Sweep: {RR_SWEEP}")
print(f"  PrecFloor:{PREC_FLOOR:.0%}  (win rate tối thiểu yêu cầu)")
print()

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("[1/4] Loading multi-timeframe data from CSV...")
t0 = time.time()
data_service = MarketDataService(settings)
strategy = HybridStrategy(settings)
risk_mgr = RiskManager(settings)
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
rr_equity_curves  = {rr: [STARTING_BALANCE] for rr in RR_SWEEP}  # cumulative equity per RR

fold_idx = 0
fold_start = 0

while fold_start + TRAIN_BARS + TEST_BARS <= n_total:
    if MAX_FOLDS is not None and fold_idx >= MAX_FOLDS:
        print(f"      [max_folds={MAX_FOLDS} reached — stopping early]")
        break
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
        max_iter=150, learning_rate=0.1, max_depth=4,
        min_samples_leaf=30, class_weight="balanced",
        early_stopping=False, random_state=42,
    )
    local_model.fit(X_tr[:val_cut], y_tr[:val_cut])
    val_proba = local_model.predict_proba(X_tr[val_cut:])[:, 1]
    y_val = y_tr[val_cut:]

    # Objective: maximize precision^2 × recall — heavily penalise low precision,
    # lightly reward recall so we keep enough signals but never sacrifice win rate.
    # prec ≥ PREC_FLOOR (0.78) is a hard gate; at least 3 signals required.
    best_thr, best_score = THRESHOLD_MIN, -float("inf")
    for thr in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP):
        preds  = (val_proba >= thr).astype(int)
        n_pred = int(preds.sum())
        if n_pred < 3:
            continue   # too few signals → skip
        prec = precision_score(y_val, preds, zero_division=0)
        rec  = recall_score(y_val, preds, zero_division=0)
        if prec < PREC_FLOOR:
            continue
        if rec < 0.01:
            continue  # bỏ qua threshold cho quá ít lệnh
        # F-beta với beta=1.5: balance WR và số lệnh/fold
        beta = 1.5
        score = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec)
        if score > best_score:
            best_score, best_thr = score, float(thr)

    # Train final model on 100% of fold_train
    # class_weight neutral → avoids over-predicting positives at expense of precision
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=5,
        min_samples_leaf=30, class_weight={0: 1, 1: 1},
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

    # ── RR sweep P&L simulation (continuous equity, per fold) ────────────────
    fold_rr_stats: dict = {}
    for _rr in RR_SWEEP:
        _equity = rr_equity_curves[_rr][-1]   # continue from previous fold's end balance
        _wins   = 0
        for _p, _t in zip(test_preds, y_te):
            if _p == 1:
                _risk = _equity * RISK_PCT
                if _t == 1:
                    _equity += _risk * _rr
                    _wins   += 1
                else:
                    _equity -= _risk
        _tot  = int(test_preds.sum())
        _wr   = _wins / _tot if _tot > 0 else prec   # fallback to model precision
        _ev   = _wr * _rr - (1.0 - _wr)
        rr_equity_curves[_rr].append(round(_equity, 2))
        fold_rr_stats[_rr] = {
            "final_balance": round(_equity, 2),
            "win_rate":      round(_wr, 4),
            "ev_per_trade":  round(_ev, 4),
            "trades":        _tot,
        }

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
        "rr_sweep":    fold_rr_stats,
    }

    # ── per-fold dynamic concurrent P&L simulation ───────────────────────
    fold_sim_df = fold_test.copy()
    fold_sim_df["split"]       = "test"
    fold_sim_df["prediction"]  = test_preds
    fold_sim_df["probability"] = test_proba
    fold_sim_df["trade_side"]  = fold_sim_df.get("trade_side", pd.Series("buy", index=fold_sim_df.index))
    _sim_settings = settings_full.model_copy(deep=True)
    _sim_settings.training.backtest_initial_balance = STARTING_BALANCE
    fold_sim = simulate_dynamic_concurrent_backtest(fold_sim_df, _sim_settings, risk_mgr)
    sim_r = fold_sim.report
    result["concurrent_sim"] = {
        "starting_balance":       STARTING_BALANCE,
        "ending_balance":         sim_r["ending_balance"],
        "return_pct":             sim_r["return_pct"],
        "trades":                 sim_r["trades"],
        "wins":                   sim_r["wins"],
        "losses":                 sim_r["losses"],
        "win_rate":               sim_r["win_rate"],
        "profit_factor":          sim_r["profit_factor"],
        "max_drawdown_pct":       sim_r["max_drawdown_pct"],
        "max_concurrent_positions": sim_r["max_concurrent_positions"],
        "avg_concurrent_positions": sim_r["avg_concurrent_positions"],
        "position_tier_breakdown": sim_r["position_tier_breakdown"],
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

# Aggregate dynamic concurrent sim metrics
_csims = [r["concurrent_sim"] for r in fold_results if "concurrent_sim" in r]
avg_sim_wr  = float(np.mean([c["win_rate"]    for c in _csims])) if _csims else 0.0
avg_sim_pf  = float(np.mean([c["profit_factor"] for c in _csims])) if _csims else 0.0
avg_sim_ret = float(np.mean([c["return_pct"]  for c in _csims])) if _csims else 0.0
avg_sim_dd  = float(np.mean([c["max_drawdown_pct"] for c in _csims])) if _csims else 0.0
avg_sim_pos = float(np.mean([c["avg_concurrent_positions"] for c in _csims])) if _csims else 0.0
max_sim_pos = int(max([c["max_concurrent_positions"] for c in _csims], default=0))

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

# Dynamic concurrent simulation summary
if _csims:
    print("  📊 Dynamic Concurrent Position Simulation (per-fold avg):")
    print(f"     Win Rate avg   : {avg_sim_wr:.1%}")
    print(f"     Profit Factor  : {avg_sim_pf:.3f}")
    print(f"     Return/fold    : {avg_sim_ret:+.2f}%  (${STARTING_BALANCE:.0f} start per fold)")
    print(f"     Max Drawdown   : {avg_sim_dd:.2f}%")
    print(f"     Avg concurrent : {avg_sim_pos:.1f} positions | Max concurrent: {max_sim_pos}")
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

# ── RR Sweep Analysis ────────────────────────────────────────────────────────
print("=" * 70)
print(f"  RR SWEEP — khởi đầu ${STARTING_BALANCE:.0f}  |  rủi ro {RISK_PCT:.2%}/lệnh")
print("=" * 70)

best_rr   = None
best_avg  = 0.0
rr_summary: dict = {}
for _rr in RR_SWEEP:
    _balances = rr_equity_curves[_rr][1:]   # 1 entry per fold (skip init)
    if not _balances:
        continue
    _wr_list = [r["rr_sweep"][_rr]["win_rate"]     for r in fold_results if "rr_sweep" in r and _rr in r["rr_sweep"]]
    _ev_list = [r["rr_sweep"][_rr]["ev_per_trade"] for r in fold_results if "rr_sweep" in r and _rr in r["rr_sweep"]]
    _tr_list = [r["rr_sweep"][_rr]["trades"]       for r in fold_results if "rr_sweep" in r and _rr in r["rr_sweep"]]
    _avg_bal = float(np.mean(_balances))
    _fin_bal = _balances[-1]
    _avg_wr  = float(np.mean(_wr_list)) if _wr_list else 0.0
    _avg_ev  = float(np.mean(_ev_list)) if _ev_list else 0.0
    _tot_tr  = int(sum(_tr_list))        if _tr_list else 0
    _max_dd  = _compute_max_drawdown(rr_equity_curves[_rr])
    rr_summary[_rr] = {
        "avg_balance":      round(_avg_bal, 2),
        "final_balance":    round(_fin_bal, 2),
        "avg_win_rate":     round(_avg_wr,  4),
        "avg_ev_per_trade": round(_avg_ev,  4),
        "total_trades":     _tot_tr,
        "max_drawdown":     round(_max_dd,  4),
    }
    if _avg_bal > best_avg:
        best_avg = _avg_bal
        best_rr  = _rr

print(f"  {'RR':>5}  {'Avg Balance':>12}  {'Final Bal':>10}  {'Avg Win%':>9}  {'Avg EV/T':>9}  {'Trades':>7}  {'MaxDD':>7}")
print("  " + "-" * 72)
for _rr in RR_SWEEP:
    if _rr not in rr_summary:
        continue
    _s  = rr_summary[_rr]
    _mk = "  ← TỐI ƯU" if _rr == best_rr else ""
    print(
        f"  {_rr:>5.1f}  ${_s['avg_balance']:>11.2f}  ${_s['final_balance']:>9.2f}  "
        f"{_s['avg_win_rate']:>9.1%}  {_s['avg_ev_per_trade']:>+9.4f}  "
        f"{_s['total_trades']:>7d}  {_s['max_drawdown']:>6.1%}{_mk}"
    )

print()
if best_rr:
    print(f"  ✅  Khuyến nghị TP RR tối ưu : {best_rr}x")
    print(f"      Balance cuối             : ${rr_summary[best_rr]['final_balance']:.2f}")
    print(f"      Win rate trung bình      : {rr_summary[best_rr]['avg_win_rate']:.1%}")
    print(f"      EV mỗi lệnh              : {rr_summary[best_rr]['avg_ev_per_trade']:+.4f}R")
    print(f"      Max Drawdown             : {rr_summary[best_rr]['max_drawdown']:.1%}")
    print()
    print(f"  >> Cập nhật take_profit_rr: {best_rr} trong configs/live_ict_wyckoff.yaml")
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
        "concurrent_sim": {
            "avg_win_rate":          round(avg_sim_wr, 4),
            "avg_profit_factor":     round(avg_sim_pf, 4),
            "avg_return_pct":        round(avg_sim_ret, 2),
            "avg_max_drawdown_pct":  round(avg_sim_dd, 2),
            "avg_concurrent_positions": round(avg_sim_pos, 2),
            "max_concurrent_positions": max_sim_pos,
        },
    },
    "folds": fold_results,
    "rr_optimal": best_rr,
    "rr_analysis": {str(k): v for k, v in rr_summary.items()},
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
