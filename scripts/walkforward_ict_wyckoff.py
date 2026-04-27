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
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

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
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--config", default=None)
_parser.add_argument("--max-folds", type=int, default=None)
_parser.add_argument("--test-start", default=None, help="Skip folds with test end < this date (YYYY-MM-DD)")
_parser.add_argument("--fast", action="store_true", help="Faster training with ≥96%% accuracy match (reduce trees, single threshold fold)")
_parser.add_argument("--cache", action="store_true", help="Cache full dataset to parquet for faster reruns")
_parser.add_argument("--no-rr-sweep", action="store_true", help="Skip RR sweep (faster, only run concurrent sim)")
_parser.add_argument("--no-compound", action="store_true", help="Reset balance to $200 each fold (no compounding)")
_parser.add_argument("--test-bars", type=int, default=None, help="Override TEST_BARS (bars per fold test window)")
_parser.add_argument("--step-bars", type=int, default=None, help="Override STEP_BARS (bars to slide per fold)")
_known, _rest = _parser.parse_known_args()
FAST_MODE = _known.fast
NO_COMPOUND = _known.no_compound

if _known.config:
    CONFIG = Path(_known.config)
elif _rest:
    CONFIG = Path(_rest[0])
else:
    CONFIG = Path("configs/train_ict_wyckoff_2022_2026.yaml")

if _known.max_folds is not None:
    MAX_FOLDS = _known.max_folds
elif len(_rest) > 1:
    MAX_FOLDS = int(_rest[1])
else:
    MAX_FOLDS = None   # None = no limit
settings = load_settings(CONFIG)

# Walk-forward window parameters  (M15: 96 bars/day  ~252 trading days/year)
TRAIN_BARS = 30_000   # ~312 trading days = ~13 months (v2: more data for stronger model)
TEST_BARS  = _known.test_bars if _known.test_bars is not None else  4_000   # ~42 trading days = ~1.5 months (override with --test-bars)
STEP_BARS  = _known.step_bars if _known.step_bars is not None else  4_000   # slide ~1.5 months at a time (override with --step-bars)

THRESHOLD_MIN   = settings.training.threshold_min        # 0.45
THRESHOLD_MAX   = settings.training.threshold_max        # 0.80
THRESHOLD_STEP  = settings.training.threshold_step       # 0.01
PREC_FLOOR      = settings.training.min_precision_floor  # now 0.60 (from config)

# ── Balance & RR sweep settings ─────────────────────────────────────────────
STARTING_BALANCE = 200.0                           # USD khởi đầu
RISK_PCT         = settings.risk.risk_per_trade    # rủi ro/lệnh (0.0065 = 0.65%)
RR_SWEEP         = [1.8, 2.0, 2.2, 2.5, 3.0, 3.5] # TP:SL ratios cần đánh giá

print(f"  Config  : {CONFIG}")
print(f"  Mode    : {'⚡ FAST' if FAST_MODE else '🎯 EXACT'}")
print(f"  Features: {len(FEATURE_COLUMNS)}  (D1:1 H4:9 H1:3 M15:15 News:5 StructMomentum:11 Adv:9)")
print(f"  Train   : {TRAIN_BARS:,} bars (~1 yr M15)")
print(f"  Test    : {TEST_BARS:,} bars (~3 mo M15)")
print(f"  Step    : {STEP_BARS:,} bars (~3 mo slide)")
print(f"  Balance : ${STARTING_BALANCE:.0f} khởi đầu | Rủi ro {RISK_PCT:.2%}/lệnh{' | NO-COMPOUND (reset $200/fold)' if NO_COMPOUND else ''}")
print(f"  RR Sweep: {'SKIP' if _known.no_rr_sweep else RR_SWEEP}")
print(f"  PrecFloor:{PREC_FLOOR:.0%}  (win rate tối thiểu yêu cầu)")
print()

LABEL_LOOKAHEAD_BARS = get_label_lookahead_bars(settings)
print(f"  Label lookahead purge: {LABEL_LOOKAHEAD_BARS} bars")

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("[1/4] Loading multi-timeframe data from CSV...")
t0 = time.time()
data_service = MarketDataService(settings)
strategy = HybridStrategy(settings)
risk_mgr = RiskManager(settings)
frames = data_service.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)
_exec_tf = settings.market.execution_timeframe
print(f"      {_exec_tf}:{len(frames[_exec_tf]):,}  H4:{len(frames['H4']):,}  "
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

# Dataset caching: avoid recomputing features + SL/TP labels on reruns
import hashlib as _hashlib
_cache_dir = Path("outputs/.wf_cache")
_cache_dir.mkdir(parents=True, exist_ok=True)
_cache_key_parts = [
    CONFIG.read_text(encoding="utf-8"),
    str(len(frames[_exec_tf])),
    str(len(FEATURE_COLUMNS)),
    str(getattr(settings_full.training, "sltp_label_max_horizon", 32)),
    str(settings_full.risk.take_profit_rr),
    str(settings_full.risk.stop_loss_atr_multiple),
]
_cache_hash = _hashlib.sha256("||".join(_cache_key_parts).encode()).hexdigest()[:12]
_cache_path = _cache_dir / f"{_cfg_stem}_{_cache_hash}.parquet"
_cache_hit = False

if _known.cache and _cache_path.exists():
    try:
        full_ds = pd.read_parquet(_cache_path)
        _cache_hit = True
        print(f"      ✅ Cache hit: {_cache_path.name}")
    except Exception as _e:
        print(f"      ⚠️  Cache read failed ({_e}), rebuilding...")

if not _cache_hit:
    full_ds = prepare_training_dataset(settings_full, frames, strategy)
    if _known.cache:
        try:
            full_ds.to_parquet(_cache_path, index=False)
            print(f"      💾 Cached to: {_cache_path.name}")
        except Exception:
            pass  # non-critical
print(f"      Total rows : {len(full_ds):,}")
print(f"      Date range : {full_ds['time'].min().date()} → {full_ds['time'].max().date()}")
print(f"      Label rate : {full_ds['target'].mean():.1%}")
print(f"      Time       : {time.time()-t0:.1f}s")

# ── 2b. Load M1 data for accurate trailing SL / partial TP simulation ───────
_m1_df: "pd.DataFrame | None" = None
_m1_csv = Path("src/xauusd_ai/real_data/XAUUSDm_M1.csv")
if _m1_csv.exists():
    print("\n[2b] Loading M1 data for bar-by-bar trailing SL simulation...")
    t0 = time.time()
    try:
        _m1_df = pd.read_csv(_m1_csv, index_col=0, parse_dates=True)
        # Ensure UTC-aware index
        if _m1_df.index.tz is None:
            _m1_df.index = _m1_df.index.tz_localize("UTC")
        else:
            _m1_df.index = _m1_df.index.tz_convert("UTC")
        _m1_df = _m1_df.sort_index()
        print(f"      M1 rows    : {len(_m1_df):,}  ({_m1_df.index[0].date()} → {_m1_df.index[-1].date()})  ({time.time()-t0:.1f}s)")
        print(f"      M1 sim     : ENABLED — more accurate trailing SL + partial TP")
    except Exception as _e:
        print(f"      ⚠️  M1 load failed ({_e}) — using peak_rr heuristic fallback")
        _m1_df = None
else:
    print("\n[2b] M1 CSV not found — using peak_rr heuristic fallback")

# ── 3. Walk-forward loop ─────────────────────────────────────────────────────
print(f"\n[3/4] Running walk-forward folds...")

n_total = len(full_ds)
n_folds = max((n_total - TRAIN_BARS - TEST_BARS) // STEP_BARS + 1, 0)
print(f"      Estimated folds: {n_folds}")
print()

fold_results = []
win_log = []   # for live-learning log (thắng/thua)
sim_trade_log: list[pd.DataFrame] = []  # per-trade records from concurrent sim
rr_equity_curves  = {rr: [STARTING_BALANCE] for rr in RR_SWEEP}  # cumulative equity per RR
_compound_balance = STARTING_BALANCE  # running balance carried across folds

fold_idx = 0
fold_start = 0

while fold_start + TRAIN_BARS + TEST_BARS <= n_total:
    if MAX_FOLDS is not None and fold_idx >= MAX_FOLDS:
        print(f"      [max_folds={MAX_FOLDS} reached — stopping early]")
        break
    fold_idx += 1
    train_end  = fold_start + TRAIN_BARS
    test_end   = train_end  + TEST_BARS

    fold_train = full_ds.iloc[fold_start:train_end]
    fold_test  = full_ds.iloc[train_end:test_end]

    # Prevent boundary leakage: drop train tail whose labels depend on future bars.
    if LABEL_LOOKAHEAD_BARS > 0:
        if len(fold_train) <= LABEL_LOOKAHEAD_BARS + 100:
            fold_start += STEP_BARS
            continue
        fold_train = fold_train.iloc[:-LABEL_LOOKAHEAD_BARS]

    if len(fold_train) < 500 or len(fold_test) < 100:
        fold_start += STEP_BARS
        continue

    # Skip folds before --test-start date
    if _known.test_start and str(fold_test["time"].max().date()) < _known.test_start:
        fold_start += STEP_BARS
        fold_idx -= 1  # don't count skipped folds
        continue

    t_fold = time.time()

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(fold_train[FEATURE_COLUMNS])
    X_te = scaler.transform(fold_test[FEATURE_COLUMNS])
    y_tr = fold_train["target"].values
    y_te = fold_test["target"].values

    # ── Sample weights: 2× positive boost + time decay ────────────────
    _pos_c = int(y_tr.sum())
    _neg_c = int(len(y_tr) - _pos_c)
    if _pos_c > 10 and _neg_c > 10:
        _pw = 2.0 * _neg_c / _pos_c
        _class_w = np.where(y_tr == 1, _pw, 1.0).astype(float)
        # Time-decay: recent bars get up to 2× weight (exponential decay)
        _n = len(y_tr)
        _decay_half = _n * 0.4  # half-life at 40% of window
        _time_w = np.exp(np.log(2) * np.arange(_n) / _decay_half)
        _time_w /= _time_w.mean()  # normalize so mean=1
        _sw_tr = (_class_w * _time_w).astype(float)
        _sw_tr /= _sw_tr.mean()
    else:
        _sw_tr = None

    # ── Threshold search: temporal CV for robustness ─────────────────
    # FAST mode: single fold (last split only) — EXACT mode: 3 folds
    _thr_candidates = []
    _n_tr = len(y_tr)
    if FAST_MODE:
        _thr_splits = [
            (0, int(_n_tr * 0.70), int(_n_tr * 0.70), _n_tr),
        ]
    else:
        _thr_splits = [
            (0, int(_n_tr * 0.50), int(_n_tr * 0.50), int(_n_tr * 0.70)),
            (0, int(_n_tr * 0.60), int(_n_tr * 0.60), int(_n_tr * 0.80)),
            (0, int(_n_tr * 0.70), int(_n_tr * 0.70), _n_tr),
        ]
    _thr_hgb_iter = 200 if FAST_MODE else 400
    for _ts_start, _ts_end, _vs_start, _vs_end in _thr_splits:
        _thr_hgb = HistGradientBoostingClassifier(
            max_iter=_thr_hgb_iter, learning_rate=0.02, max_depth=6,
            min_samples_leaf=25, l2_regularization=1.0,
            max_bins=128, class_weight=None,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=42,
        )
        _sw_sub = _sw_tr[_ts_start:_ts_end] if _sw_tr is not None else None
        _thr_hgb.fit(X_tr[_ts_start:_ts_end], y_tr[_ts_start:_ts_end], sample_weight=_sw_sub)
        _v_proba = _thr_hgb.predict_proba(X_tr[_vs_start:_vs_end])[:, 1]
        _y_v = y_tr[_vs_start:_vs_end]
        _fold_best_thr, _fold_best_score = THRESHOLD_MIN, -float("inf")
        _fold_safe_thr, _fold_safe_prec = THRESHOLD_MAX, -1.0
        for thr in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP):
            preds  = (_v_proba >= thr).astype(int)
            n_pred = int(preds.sum())
            if n_pred < 3:
                continue
            prec = precision_score(_y_v, preds, zero_division=0)
            rec  = recall_score(_y_v, preds, zero_division=0)
            if rec < 0.05:
                continue
            if prec > _fold_safe_prec:
                _fold_safe_prec = prec
                _fold_safe_thr = float(thr)
            if prec < PREC_FLOOR:
                continue
            score = prec * np.sqrt(rec)
            if score > _fold_best_score:
                _fold_best_score, _fold_best_thr = score, float(thr)
        if _fold_best_score == -float("inf"):
            _fold_best_thr = _fold_safe_thr
        _thr_candidates.append(_fold_best_thr)
    # Max of 3 temporal folds → most conservative (highest precision)
    best_thr = float(np.max(_thr_candidates))

    # ── Feature selection: drop bottom 30% by importance ─────────────
    _scout = RandomForestClassifier(
        n_estimators=80 if FAST_MODE else 200, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    _scout.fit(X_tr, y_tr, sample_weight=_sw_tr)
    _imp = _scout.feature_importances_
    _imp_thr = np.percentile(_imp, 30)  # drop bottom 30%
    _feat_mask = _imp >= _imp_thr
    if _feat_mask.sum() < 10:  # safety: keep at least 10 features
        _feat_mask = np.ones(len(_imp), dtype=bool)
    X_tr_sel = X_tr[:, _feat_mask]
    X_te_sel = X_te[:, _feat_mask]

    # ── Train ENSEMBLE on 100% of fold_train ─────────────────────────
    _hgb = HistGradientBoostingClassifier(
        max_iter=1000 if FAST_MODE else 2000,
        learning_rate=0.01, max_depth=7,
        min_samples_leaf=20, l2_regularization=1.0,
        max_bins=128, class_weight=None,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=40 if FAST_MODE else 80, random_state=42,
    )
    _rf = RandomForestClassifier(
        n_estimators=200 if FAST_MODE else 400, max_depth=12, min_samples_leaf=15,
        max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=42,
    )
    _et = ExtraTreesClassifier(
        n_estimators=200 if FAST_MODE else 400, max_depth=14, min_samples_leaf=10,
        max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=42,
    )
    model = VotingClassifier(
        estimators=[("hgb", _hgb), ("rf", _rf), ("et", _et)],
        voting="soft",
        weights=[3, 2, 1],  # HGB gets most weight (best single model)
    )
    model.fit(X_tr_sel, y_tr, sample_weight=_sw_tr)

    # Evaluate on fold_test
    test_proba = model.predict_proba(X_te_sel)[:, 1]
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
    if not _known.no_rr_sweep:
        _spread_rr_base = float(getattr(settings_full.risk, "spread_cost_rr", 0.10))
        _slippage_rr = float(getattr(settings_full.risk, "slippage_rr", 0.05))
        _commission_rr = float(getattr(settings_full.risk, "commission_rr", 0.02))
        _friction_rr = _spread_rr_base + _slippage_rr + _commission_rr
        _compound_cap = float(getattr(settings_full.risk, "compound_cap", 50.0))
        _max_rr_equity = STARTING_BALANCE * _compound_cap if _compound_cap > 0 else float("inf")
        # P1a: per-bar session spread multiplier
        _sess_mults = fold_test_copy["session_spread_mult"].values if "session_spread_mult" in fold_test_copy.columns else None
        # Pre-filter: only iterate over signal bars (skip non-signal bars)
        _signal_mask = test_preds == 1
        _signal_indices = np.where(_signal_mask)[0]
        _signal_targets = y_te[_signal_indices]
        _signal_sess = _sess_mults[_signal_indices] if _sess_mults is not None else None
        for _rr in RR_SWEEP:
            _equity = rr_equity_curves[_rr][-1]   # continue from previous fold's end balance
            _wins   = 0
            for _j in range(len(_signal_indices)):
                _eff = min(_equity, _max_rr_equity)  # cap compound growth
                _risk = _eff * RISK_PCT
                _sm = float(_signal_sess[_j]) if _signal_sess is not None else 1.0
                _fr = _spread_rr_base * _sm + _slippage_rr + _commission_rr
                if _signal_targets[_j] == 1:
                    _equity += _risk * (_rr - _fr)
                    _wins   += 1
                else:
                    _equity -= _risk * (1.0 + _fr)
            _tot  = len(_signal_indices)
            _wr   = _wins / _tot if _tot > 0 else prec   # fallback to model precision
            _ev   = _wr * (_rr - _friction_rr) - (1.0 - _wr) * (1.0 + _friction_rr)
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
    _sim_settings.training.backtest_initial_balance = _compound_balance  # compound across folds
    fold_sim = simulate_dynamic_concurrent_backtest(fold_sim_df, _sim_settings, risk_mgr, m1_df=_m1_df)
    sim_r = fold_sim.report
    _compound_balance = STARTING_BALANCE if NO_COMPOUND else sim_r["ending_balance"]  # carry forward or reset
    # Collect per-trade records for daily analysis
    if not fold_sim.trades.empty:
        _ft = fold_sim.trades.copy()
        _ft["fold"] = fold_idx
        _ft["test_start"] = result.get("test_start", "")
        _ft["test_end"]   = result.get("test_end", "")
        sim_trade_log.append(_ft)
    result["concurrent_sim"] = {
        "starting_balance":       _sim_settings.training.backtest_initial_balance,
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
        "model": f"Ensemble(HGB+RF+ET, weights=3:2:1, feature_sel=top70%) | {len(FEATURE_COLUMNS)} features",
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

out_report = Path(settings.app.walkforward_report_path)
out_signals = Path(settings.app.walkforward_trades_path)

out_report.write_text(json.dumps(wf_report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"  Report saved  → {out_report}")

# Save per-trade sim records
_out_sim_trades = out_signals.parent / (out_signals.stem + "_sim_trades.csv")
if sim_trade_log:
    all_sim_trades = pd.concat(sim_trade_log, ignore_index=True)
    all_sim_trades.to_csv(_out_sim_trades, index=False)
    print(f"  Sim trades    → {_out_sim_trades}  ({len(all_sim_trades):,} executed trades)")

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
