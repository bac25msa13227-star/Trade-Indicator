#!/usr/bin/env python3
"""
Show per-day P&L breakdown for combo #133 (grid search best result).
Combo #133: min_conf=0.70, require_trend=False, min_strat=0.00, blocked=[3,15,17,22,23]
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.dataset import (
    FEATURE_COLUMNS,
    get_label_lookahead_bars,
    prepare_training_dataset,
)
from xauusd_ai.strategies.hybrid import HybridStrategy

CONFIG        = Path("configs/acc1_v14pp_profit.yaml")
STARTING_BAL  = 200.0
TRAIN_BARS    = 30_000
TEST_BARS     = 6_000
STEP_BARS     = 6_000
TEST_START    = "2024-01-01"

# Combo #133 params
MIN_CONF      = 0.70
REQ_TREND     = False
MIN_STRAT     = 0.00
BLOCKED       = [3, 15, 17, 22, 23]
D1_GATE       = False  # If True: only trade WITH D1 daily_bias (buy when +1, sell when -1)

print("=" * 80)
print("  COMBO #133 — Per-Day Breakdown")
print(f"  min_conf={MIN_CONF}  trend={REQ_TREND}  strat={MIN_STRAT}  blocked={BLOCKED}  d1_gate={D1_GATE}")
print("=" * 80)

settings = load_settings(CONFIG)
data_service = MarketDataService(settings)
print("[1/3] Loading data...")
frames = data_service.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)

settings_full = load_settings(CONFIG)
settings_full.training.train_start_date = None
settings_full.training.train_end_date   = None
settings_full.training.test_start_date  = None
settings_full.training.test_end_date    = None
risk_mgr  = RiskManager(settings_full)
_strategy = HybridStrategy(settings_full)

print("[2/3] Building dataset...")
t0 = time.time()
full_ds = prepare_training_dataset(settings_full, frames, _strategy)
print(f"      {len(full_ds):,} rows  ({time.time()-t0:.1f}s)")
LABEL_LOOKAHEAD_BARS = get_label_lookahead_bars(settings_full)
print(f"      label lookahead purge: {LABEL_LOOKAHEAD_BARS} bars")

_m1_path = Path("src/xauusd_ai/real_data/XAUUSDm_M1.csv")
if _m1_path.exists():
    _m1_df = pd.read_csv(_m1_path, index_col=0, parse_dates=True)
    if _m1_df.index.tz is None:
        _m1_df.index = _m1_df.index.tz_localize("UTC")
    else:
        _m1_df.index = _m1_df.index.tz_convert("UTC")
    _m1_df = _m1_df.sort_index()
else:
    _m1_df = None

THRESHOLD_MIN  = settings.training.threshold_min
THRESHOLD_MAX  = settings.training.threshold_max
THRESHOLD_STEP = settings.training.threshold_step
PREC_FLOOR     = settings.training.min_precision_floor

n_total = len(full_ds)
all_trades_df = []   # tập hợp tất cả trades để tính per-day

print("\n[3/3] Running combo #133 per fold...")

fold_idx   = 0
fold_start = 0
grand_trades = 0
grand_pnl    = 0.0

print()
print(f"  {'Fold':>5}  {'Kỳ test':>24}  {'Trades':>7}  {'WR':>6}  {'P&L':>9}  {'DD':>7}  {'WF EndBal':>10}")
print("  " + "-" * 80)

fold_summary = []

while fold_start + TRAIN_BARS + TEST_BARS <= n_total:
    fold_idx += 1
    train_end  = fold_start + TRAIN_BARS
    test_end   = train_end  + TEST_BARS
    fold_train = full_ds.iloc[fold_start:train_end]
    fold_test  = full_ds.iloc[train_end:test_end]

    # Prevent boundary leakage: drop train tail whose labels need future bars.
    if LABEL_LOOKAHEAD_BARS > 0:
        if len(fold_train) <= LABEL_LOOKAHEAD_BARS + 100:
            fold_start += STEP_BARS
            continue
        fold_train = fold_train.iloc[:-LABEL_LOOKAHEAD_BARS]

    if len(fold_train) < 500 or len(fold_test) < 100:
        fold_start += STEP_BARS
        continue
    if str(fold_test["time"].max().date()) < TEST_START:
        fold_start += STEP_BARS
        fold_idx -= 1
        continue
    if str(fold_test["time"].min().date()) > "2026-04-30":
        break

    y_tr = fold_train["target"].values
    y_te = fold_test["target"].values

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(fold_train[FEATURE_COLUMNS])
    X_te = scaler.transform(fold_test[FEATURE_COLUMNS])

    _pos_c, _neg_c = int(y_tr.sum()), int(len(y_tr) - y_tr.sum())
    if _pos_c > 10 and _neg_c > 10:
        _pw = 2.0 * _neg_c / _pos_c
        _class_w = np.where(y_tr == 1, _pw, 1.0).astype(float)
        _n = len(y_tr)
        _decay_half = _n * 0.4
        _time_w = np.exp(np.log(2) * np.arange(_n) / _decay_half)
        _time_w /= _time_w.mean()
        _sw_tr = (_class_w * _time_w).astype(float)
        _sw_tr /= _sw_tr.mean()
    else:
        _sw_tr = None

    # Threshold search
    _thr_hgb = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.02, max_depth=6, min_samples_leaf=25,
        l2_regularization=1.0, max_bins=128,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=42,
    )
    _n_tr = len(y_tr)
    _vs_s = int(_n_tr * 0.70)
    _sw_sub = _sw_tr[:_vs_s] if _sw_tr is not None else None
    _thr_hgb.fit(X_tr[:_vs_s], y_tr[:_vs_s], sample_weight=_sw_sub)
    _v_proba = _thr_hgb.predict_proba(X_tr[_vs_s:])[:, 1]
    _y_v = y_tr[_vs_s:]
    from sklearn.metrics import precision_score, recall_score
    best_thr, best_score = THRESHOLD_MIN, -float("inf")
    safe_thr, safe_prec = THRESHOLD_MAX, -1.0
    for thr in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP):
        preds = (_v_proba >= thr).astype(int)
        if preds.sum() < 3: continue
        prec = precision_score(_y_v, preds, zero_division=0)
        rec  = recall_score(_y_v, preds, zero_division=0)
        if rec < 0.05: continue
        if prec > safe_prec:
            safe_prec, safe_thr = prec, float(thr)
        if prec < PREC_FLOOR: continue
        score = prec * np.sqrt(rec)
        if score > best_score:
            best_score, best_thr = score, float(thr)
    if best_score == -float("inf"):
        best_thr = safe_thr

    # Feature selection
    _scout = RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_leaf=20,
                                     class_weight="balanced", n_jobs=-1, random_state=42)
    _scout.fit(X_tr, y_tr, sample_weight=_sw_tr)
    _imp = _scout.feature_importances_
    _feat_mask = _imp >= np.percentile(_imp, 30)
    if _feat_mask.sum() < 10:
        _feat_mask = np.ones(len(_imp), dtype=bool)
    X_tr_sel = X_tr[:, _feat_mask]
    X_te_sel = X_te[:, _feat_mask]

    # Train ensemble
    _hgb = HistGradientBoostingClassifier(
        max_iter=1000, learning_rate=0.01, max_depth=7, min_samples_leaf=20,
        l2_regularization=1.0, max_bins=128,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=40, random_state=42,
    )
    _rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=15,
                                  max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42)
    _et = ExtraTreesClassifier(n_estimators=200, max_depth=14, min_samples_leaf=10,
                                max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42)
    model = VotingClassifier(estimators=[("hgb", _hgb), ("rf", _rf), ("et", _et)],
                              voting="soft", weights=[3, 2, 1])
    model.fit(X_tr_sel, y_tr, sample_weight=_sw_tr)

    test_proba = model.predict_proba(X_te_sel)[:, 1]
    test_preds = (test_proba >= best_thr).astype(int)

    fold_sim_df = fold_test.copy()
    fold_sim_df["split"]       = "test"
    fold_sim_df["prediction"]  = test_preds
    fold_sim_df["probability"] = test_proba
    # trade_side: buy when strategy_score>=0, sell when <0
    fold_sim_df["trade_side"] = np.where(fold_sim_df["strategy_score"] >= 0, "buy", "sell")
    # D1 gate: force prediction=0 for counter-trend direction
    if D1_GATE and "daily_bias" in fold_sim_df.columns:
        counter_trend_mask = (
            ((fold_sim_df["daily_bias"] > 0) & (fold_sim_df["trade_side"] == "sell")) |
            ((fold_sim_df["daily_bias"] < 0) & (fold_sim_df["trade_side"] == "buy"))
        )
        fold_sim_df.loc[counter_trend_mask, "prediction"] = 0

    _sim_settings = settings_full.model_copy(deep=True)
    _sim_settings.training.backtest_initial_balance        = STARTING_BAL
    _sim_settings.risk.min_confidence                      = MIN_CONF
    _sim_settings.strategy.sideway_min_confidence          = MIN_CONF
    _sim_settings.strategy.volatile_min_confidence         = MIN_CONF
    _sim_settings.strategy.require_trend_alignment         = REQ_TREND
    _sim_settings.strategy.min_strategy_score              = MIN_STRAT
    _sim_settings.strategy.sideway_min_strategy_score      = MIN_STRAT
    _sim_settings.strategy.strong_volatility_min_strategy_score = MIN_STRAT
    _sim_settings.strategy.blocked_hours_utc               = BLOCKED
    _sim_settings.strategy.d1_trend_gate                   = False  # already handled above via prediction mask

    fold_sim = simulate_dynamic_concurrent_backtest(fold_sim_df, _sim_settings, risk_mgr, m1_df=_m1_df)
    sim_r = fold_sim.report

    pnl  = sim_r["ending_balance"] - STARTING_BAL
    flag = "✅" if pnl >= 0 else "❌"
    grand_trades += sim_r["trades"]
    grand_pnl    += pnl

    # Collect trades for per-day breakdown — fold_sim.trades is a DataFrame
    if hasattr(fold_sim, "trades") and isinstance(fold_sim.trades, pd.DataFrame) and not fold_sim.trades.empty:
        tdf = fold_sim.trades.copy()
        tdf["fold"] = fold_idx
        all_trades_df.append(tdf)

    fold_summary.append({
        "fold": fold_idx,
        "test_start": str(fold_test["time"].min().date()),
        "test_end": str(fold_test["time"].max().date()),
        "trades": sim_r["trades"],
        "wr": sim_r["win_rate"],
        "pnl": pnl,
        "dd": sim_r["max_drawdown_pct"],
        "end_bal": sim_r["ending_balance"],
    })

    print(f"  {fold_idx:>5}  {str(fold_test['time'].min().date())} → {str(fold_test['time'].max().date())}  "
          f"{sim_r['trades']:>7}  {sim_r['win_rate']:>5.1%}  "
          f"{'+' if pnl>=0 else ''}{pnl:>8.2f}  {sim_r['max_drawdown_pct']:>6.2f}%  "
          f"${sim_r['ending_balance']:>8.2f}  {flag}")

    fold_start += STEP_BARS

# ── Partial last fold: run whatever bars remain after the last full fold ───────
_remaining_train_end = fold_start + TRAIN_BARS
_remaining_test_end  = n_total  # use ALL remaining bars as test
if (_remaining_train_end < n_total and
        n_total - _remaining_train_end >= 200 and  # need at least 200 test bars
        str(full_ds.iloc[_remaining_train_end]["time"].date()) <= "2026-04-30"):
    fold_idx += 1
    fold_train = full_ds.iloc[fold_start:_remaining_train_end]
    fold_test  = full_ds.iloc[_remaining_train_end:_remaining_test_end]
    if LABEL_LOOKAHEAD_BARS > 0 and len(fold_train) > LABEL_LOOKAHEAD_BARS + 100:
        fold_train = fold_train.iloc[:-LABEL_LOOKAHEAD_BARS]
    if len(fold_train) >= 500 and len(fold_test) >= 50:
        y_tr = fold_train["target"].values
        y_te = fold_test["target"].values
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(fold_train[FEATURE_COLUMNS])
        X_te = scaler.transform(fold_test[FEATURE_COLUMNS])
        _pos_c, _neg_c = int(y_tr.sum()), int(len(y_tr) - y_tr.sum())
        if _pos_c > 10 and _neg_c > 10:
            _pw = 2.0 * _neg_c / _pos_c
            _class_w = np.where(y_tr == 1, _pw, 1.0).astype(float)
            _n = len(y_tr)
            _decay_half = _n * 0.4
            _time_w = np.exp(np.log(2) * np.arange(_n) / _decay_half)
            _time_w /= _time_w.mean()
            _sw_tr = (_class_w * _time_w).astype(float)
            _sw_tr /= _sw_tr.mean()
        else:
            _sw_tr = None
        # Full ensemble — identical to main fold pipeline
        # Threshold search
        _thr_hgb = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.02, max_depth=6, min_samples_leaf=25,
            l2_regularization=1.0, max_bins=128,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=42,
        )
        _n_tr_p = len(y_tr)
        _vs_s_p = int(_n_tr_p * 0.70)
        _sw_sub_p = _sw_tr[:_vs_s_p] if _sw_tr is not None else None
        _thr_hgb.fit(X_tr[:_vs_s_p], y_tr[:_vs_s_p], sample_weight=_sw_sub_p)
        _v_proba_p = _thr_hgb.predict_proba(X_tr[_vs_s_p:])[:, 1]
        _y_v_p = y_tr[_vs_s_p:]
        from sklearn.metrics import precision_score as _prec_s, recall_score as _rec_s
        _best_thr, _best_score = THRESHOLD_MIN, -float("inf")
        _safe_thr, _safe_prec = THRESHOLD_MAX, -1.0
        for _thr in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP):
            _preds_t = (_v_proba_p >= _thr).astype(int)
            if _preds_t.sum() < 3: continue
            _prec_t = _prec_s(_y_v_p, _preds_t, zero_division=0)
            _rec_t  = _rec_s(_y_v_p, _preds_t, zero_division=0)
            if _rec_t < 0.05: continue
            if _prec_t > _safe_prec:
                _safe_prec, _safe_thr = _prec_t, float(_thr)
            if _prec_t < PREC_FLOOR: continue
            _sc = _prec_t * np.sqrt(_rec_t)
            if _sc > _best_score:
                _best_score, _best_thr = _sc, float(_thr)
        if _best_score == -float("inf"):
            _best_thr = _safe_thr
        # Feature selection
        _scout_p = RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_leaf=20,
                                          class_weight="balanced", n_jobs=-1, random_state=42)
        _scout_p.fit(X_tr, y_tr, sample_weight=_sw_tr)
        _imp_p = _scout_p.feature_importances_
        _feat_mask_p = _imp_p >= np.percentile(_imp_p, 30)
        if _feat_mask_p.sum() < 10:
            _feat_mask_p = np.ones(len(_imp_p), dtype=bool)
        X_tr_sel_p = X_tr[:, _feat_mask_p]
        X_te_sel_p = X_te[:, _feat_mask_p]
        # Train full VotingClassifier
        _hgb_p = HistGradientBoostingClassifier(
            max_iter=1000, learning_rate=0.01, max_depth=7, min_samples_leaf=20,
            l2_regularization=1.0, max_bins=128,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=40, random_state=42,
        )
        _rf_p = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=15,
                                        max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42)
        _et_p = ExtraTreesClassifier(n_estimators=200, max_depth=14, min_samples_leaf=10,
                                      max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42)
        _p_model = VotingClassifier(estimators=[("hgb", _hgb_p), ("rf", _rf_p), ("et", _et_p)],
                                    voting="soft", weights=[3, 2, 1])
        _p_model.fit(X_tr_sel_p, y_tr, sample_weight=_sw_tr)
        test_proba = _p_model.predict_proba(X_te_sel_p)[:, 1]
        test_preds = (test_proba >= _best_thr).astype(int)

        fold_sim_df = fold_test.copy()
        fold_sim_df["split"]       = "test"
        fold_sim_df["prediction"]  = test_preds
        fold_sim_df["probability"] = test_proba
        fold_sim_df["trade_side"]  = np.where(fold_sim_df["strategy_score"] >= 0, "buy", "sell")
        if D1_GATE and "daily_bias" in fold_sim_df.columns:
            counter_trend_mask = (
                ((fold_sim_df["daily_bias"] > 0) & (fold_sim_df["trade_side"] == "sell")) |
                ((fold_sim_df["daily_bias"] < 0) & (fold_sim_df["trade_side"] == "buy"))
            )
            fold_sim_df.loc[counter_trend_mask, "prediction"] = 0

        _sim_settings = settings_full.model_copy(deep=True)
        _sim_settings.training.backtest_initial_balance        = STARTING_BAL
        _sim_settings.risk.min_confidence                      = MIN_CONF
        _sim_settings.strategy.sideway_min_confidence          = MIN_CONF
        _sim_settings.strategy.volatile_min_confidence         = MIN_CONF
        _sim_settings.strategy.require_trend_alignment         = REQ_TREND
        _sim_settings.strategy.min_strategy_score              = MIN_STRAT
        _sim_settings.strategy.sideway_min_strategy_score      = MIN_STRAT
        _sim_settings.strategy.strong_volatility_min_strategy_score = MIN_STRAT
        _sim_settings.strategy.blocked_hours_utc               = BLOCKED
        _sim_settings.strategy.d1_trend_gate                   = False

        fold_sim = simulate_dynamic_concurrent_backtest(fold_sim_df, _sim_settings, risk_mgr, m1_df=_m1_df)
        sim_r = fold_sim.report
        pnl  = sim_r["ending_balance"] - STARTING_BAL
        flag = "✅" if pnl >= 0 else "❌"
        grand_trades += sim_r["trades"]
        grand_pnl    += pnl

        if hasattr(fold_sim, "trades") and isinstance(fold_sim.trades, pd.DataFrame) and not fold_sim.trades.empty:
            tdf = fold_sim.trades.copy()
            tdf["fold"] = fold_idx
            all_trades_df.append(tdf)

        print(f"  {fold_idx:>4}* {str(fold_test['time'].min().date())} → {str(fold_test['time'].max().date())}  "
              f"{sim_r['trades']:>7}  {sim_r['win_rate']:>5.1%}  "
              f"{'+' if pnl>=0 else ''}{pnl:>8.2f}  {sim_r['max_drawdown_pct']:>6.2f}%  "
              f"${sim_r['ending_balance']:>8.2f}  {flag}  [partial]")

print("  " + "-" * 80)
print(f"  {'TOTAL':>5}  {'':>24}  {grand_trades:>7}  {'':>6}  +{grand_pnl:>8.2f}")

# Save trades to CSV for fast re-use
trades_cache = Path("outputs/combo133_trades.csv")
if all_trades_df:
    pd.concat(all_trades_df, ignore_index=True).to_csv(trades_cache, index=False)
    print(f"\n  [Trades saved → {trades_cache}]")

# ── Per-day breakdown ──────────────────────────────────────────────────────────
if all_trades_df:
    trades_combined = pd.concat(all_trades_df, ignore_index=True)

    # Determine pnl and date columns
    pnl_col  = next((c for c in ["pnl", "net_pnl", "profit", "realized_pnl"] if c in trades_combined.columns), None)
    time_col = next((c for c in ["time", "exit_time", "entry_time", "close_time", "open_time"] if c in trades_combined.columns), None)

    if pnl_col and time_col:
        trades_combined["_date"] = pd.to_datetime(trades_combined[time_col], utc=True, errors="coerce").dt.date
        daily = (trades_combined
                 .groupby("_date")
                 .agg(
                     trades   = (pnl_col, "count"),
                     day_pnl  = (pnl_col, "sum"),
                     win_cnt  = (pnl_col, lambda x: (x > 0).sum()),
                 )
                 .reset_index()
                 .sort_values("_date"))
        daily["cum_pnl"]  = daily["day_pnl"].cumsum()
        daily["wr_day"]   = daily["win_cnt"] / daily["trades"]
        daily["dd_from_peak"] = (daily["cum_pnl"] - daily["cum_pnl"].cummax())

        print()
        print("=" * 95)
        print("  PER-DAY BREAKDOWN — Combo #133")
        print("=" * 95)
        print(f"  {'Date':>12}  {'Trades':>7}  {'WR':>6}  {'Day P&L':>10}  {'Cum P&L':>10}  {'DD frm Peak':>12}  {'Note'}")
        print("  " + "-" * 90)

        peak_cum = 0.0
        for _, row in daily.iterrows():
            cum = row["cum_pnl"]
            dd  = row["dd_from_peak"]
            flag = ""
            if row["day_pnl"] < -20:
                flag = "⚠️ heavy loss"
            elif row["day_pnl"] > 1000:
                flag = "🚀 big win"
            elif row["day_pnl"] < 0:
                flag = "❌"
            else:
                flag = "✅"
            pnl_str = f"{'+' if row['day_pnl']>=0 else ''}{row['day_pnl']:.2f}"
            cum_str = f"{'+' if cum>=0 else ''}{cum:.2f}"
            dd_str  = f"{dd:.2f}" if dd < 0 else "0.00"
            print(f"  {str(row['_date']):>12}  {int(row['trades']):>7}  {row['wr_day']:>5.0%}  "
                  f"${pnl_str:>10}  ${cum_str:>10}  {dd_str:>12}  {flag}")

        print()
        print(f"  Total days traded : {len(daily)}")
        print(f"  Profitable days   : {(daily['day_pnl'] > 0).sum()}")
        print(f"  Losing days       : {(daily['day_pnl'] < 0).sum()}")
        print(f"  Best day          : +${daily['day_pnl'].max():.2f}")
        print(f"  Worst day         : ${daily['day_pnl'].min():.2f}")
        print(f"  Avg day P&L       : +${daily['day_pnl'].mean():.2f}")
        print(f"  Max DD from peak  : {daily['dd_from_peak'].min():.2f}")
        print(f"  Total net P&L     : +${daily['day_pnl'].sum():.2f}")

        # Save daily results as markdown for quick sharing/review.
        md_path = Path("outputs/combo133_daily_report.md")
        lines = [
            "# Combo #133 Daily P&L Report",
            "",
            f"- Config: {CONFIG}",
            f"- Label lookahead purge: {LABEL_LOOKAHEAD_BARS} bars",
            f"- Train/Test bars: {TRAIN_BARS}/{TEST_BARS}",
            f"- Step bars: {STEP_BARS}",
            f"- Test start: {TEST_START}",
            f"- Total trades: {grand_trades}",
            f"- Total net P&L: {daily['day_pnl'].sum():+.2f}",
            f"- Days traded: {len(daily)}",
            f"- Profitable days: {(daily['day_pnl'] > 0).sum()}",
            f"- Losing days: {(daily['day_pnl'] < 0).sum()}",
            f"- Best day: {daily['day_pnl'].max():+.2f}",
            f"- Worst day: {daily['day_pnl'].min():+.2f}",
            f"- Avg day P&L: {daily['day_pnl'].mean():+.2f}",
            f"- Max DD from peak: {daily['dd_from_peak'].min():.2f}",
            "",
            "## Daily Breakdown",
            "",
            "| Date | Trades | WR | Day P&L | Cum P&L | DD from Peak |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _, row in daily.iterrows():
            lines.append(
                f"| {row['_date']} | {int(row['trades'])} | {row['wr_day']:.0%} | {row['day_pnl']:+.2f} | {row['cum_pnl']:+.2f} | {row['dd_from_peak']:.2f} |"
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  [Daily markdown saved → {md_path}]")
    else:
        print(f"\n  [INFO] Trades collected but columns not found. Available: {list(trades_combined.columns)}")
        print(f"  Total trades collected: {len(trades_combined)}")
else:
    print("\n  [INFO] No per-trade data available from simulation (per-day not possible).")
    print("  Fold-level summary shown above is the detailed breakdown available.")

print("=" * 95)
