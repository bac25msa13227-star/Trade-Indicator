#!/usr/bin/env python3
"""
Daily-reset walk-forward evaluator / optimizer for v14++ configs.

Goal:
- Train fold models once using the same logic as walkforward_ict_wyckoff.py
- Re-simulate each test day independently with start balance = $200
- Rank configs/candidates by honest day-level metrics:
  PF, worst-day DD, average daily PnL, % days >= $60 / $100, negative-day rate

Notes:
- This script intentionally searches only "simulation/runtime" knobs first.
- It does NOT change dataset-shaping knobs like strategy weights, volatility thresholds,
  TP/SL label settings, or label horizon, because those would require rebuilding
  the dataset / retraining logic itself for a fair comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import Settings, load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.dataset import FEATURE_COLUMNS, prepare_training_dataset
from xauusd_ai.strategies.hybrid import HybridStrategy


BASELINE_CONFIGS = [
    ROOT / "configs/acc1_v14pp_profit.yaml",
    ROOT / "configs/acc1_v14pp_composite.yaml",
    ROOT / "configs/acc1_v14pp_r12p1.yaml",
]


def _hash_json(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _compute_feature_cache_path(config_path: Path, settings: Settings, frames: dict[str, pd.DataFrame]) -> Path:
    cfg_stem = config_path.stem
    exec_tf = settings.market.execution_timeframe
    cache_dir = ROOT / "outputs/.wf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_parts = [
        config_path.read_text(encoding="utf-8"),
        str(len(frames[exec_tf])),
        str(len(FEATURE_COLUMNS)),
        str(getattr(settings.training, "sltp_label_max_horizon", 32)),
        str(settings.risk.take_profit_rr),
        str(settings.risk.stop_loss_atr_multiple),
    ]
    cache_hash = hashlib.sha256("||".join(key_parts).encode()).hexdigest()[:12]
    return cache_dir / f"{cfg_stem}_{cache_hash}.parquet"


def load_or_build_dataset(config_path: Path, cache: bool = True) -> pd.DataFrame:
    settings = load_settings(config_path)
    data_service = MarketDataService(settings)
    strategy = HybridStrategy(settings)
    frames = data_service.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)
    cache_path = _compute_feature_cache_path(config_path, settings, frames)
    if cache and cache_path.exists():
        full_ds = pd.read_parquet(cache_path)
        full_ds["time"] = pd.to_datetime(full_ds["time"], utc=True, errors="coerce")
        print(f"Loaded dataset cache: {cache_path.name} ({len(full_ds):,} rows)")
        return full_ds

    settings_full = settings.model_copy(deep=True)
    settings_full.training.train_start_date = None
    settings_full.training.train_end_date = None
    settings_full.training.test_start_date = None
    settings_full.training.test_end_date = None
    full_ds = prepare_training_dataset(settings_full, frames, strategy)
    full_ds["time"] = pd.to_datetime(full_ds["time"], utc=True, errors="coerce")
    if cache:
        full_ds.to_parquet(cache_path, index=False)
        print(f"Built and cached dataset: {cache_path.name} ({len(full_ds):,} rows)")
    return full_ds


def load_m1() -> pd.DataFrame:
    m1_path = ROOT / "src/xauusd_ai/real_data/XAUUSDm_M1.csv"
    m1 = pd.read_csv(m1_path, index_col=0, parse_dates=True)
    if m1.index.tz is None:
        m1.index = m1.index.tz_localize("UTC")
    else:
        m1.index = m1.index.tz_convert("UTC")
    return m1.sort_index()


def _threshold_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    precision_floor: float,
) -> float:
    n_tr = len(y_train)
    thr_splits = [
        (0, int(n_tr * 0.50), int(n_tr * 0.50), int(n_tr * 0.70)),
        (0, int(n_tr * 0.60), int(n_tr * 0.60), int(n_tr * 0.80)),
        (0, int(n_tr * 0.70), int(n_tr * 0.70), n_tr),
    ]
    candidates: list[float] = []
    for ts_start, ts_end, vs_start, vs_end in thr_splits:
        hgb = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.02,
            max_depth=6,
            min_samples_leaf=25,
            l2_regularization=1.0,
            max_bins=128,
            class_weight=None,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=42,
        )
        sw_sub = sample_weight[ts_start:ts_end] if sample_weight is not None else None
        hgb.fit(x_train[ts_start:ts_end], y_train[ts_start:ts_end], sample_weight=sw_sub)
        v_proba = hgb.predict_proba(x_train[vs_start:vs_end])[:, 1]
        y_val = y_train[vs_start:vs_end]
        fold_best_thr = threshold_min
        fold_best_score = -float("inf")
        fold_safe_thr = threshold_max
        fold_safe_prec = -1.0
        for thr in np.arange(threshold_min, threshold_max + threshold_step, threshold_step):
            preds = (v_proba >= thr).astype(int)
            n_pred = int(preds.sum())
            if n_pred < 3:
                continue
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            if rec < 0.05:
                continue
            if prec > fold_safe_prec:
                fold_safe_prec = prec
                fold_safe_thr = float(thr)
            if prec < precision_floor:
                continue
            score = prec * math.sqrt(rec)
            if score > fold_best_score:
                fold_best_score = score
                fold_best_thr = float(thr)
        if fold_best_score == -float("inf"):
            fold_best_thr = fold_safe_thr
        candidates.append(fold_best_thr)
    return float(np.max(candidates))


def build_fold_predictions(
    config_path: Path,
    full_ds: pd.DataFrame,
    use_cache: bool = True,
    max_folds: int | None = None,
    test_start: str | None = None,
    use_realized_rr_label: bool = False,
) -> list[dict[str, object]]:
    settings = load_settings(config_path)
    cache_dir = ROOT / "outputs/.wf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_tag = (test_start or "all").replace("-", "")
    fold_tag = "all" if max_folds is None else str(max_folds)
    window_tag = "w30000_4000_4000"
    label_tag = "_rrlabel" if use_realized_rr_label else ""
    pred_cache_path = cache_dir / f"{config_path.stem}_dailyreset_foldpreds_exact_{window_tag}_{start_tag}_{fold_tag}{label_tag}.pkl"
    if use_cache and pred_cache_path.exists():
        with pred_cache_path.open("rb") as fh:
            payload = pickle.load(fh)
        print(f"Loaded fold prediction cache: {pred_cache_path.name} ({len(payload)} folds)")
        return payload

    # Match the legacy v14pp walk-forward exactly.
    # The original script hard-codes these windows instead of reading the config defaults.
    train_bars = 30000
    test_bars = 4000
    step_bars = 4000
    threshold_min = float(settings.training.threshold_min)
    threshold_max = float(settings.training.threshold_max)
    threshold_step = float(settings.training.threshold_step)
    precision_floor = float(settings.training.min_precision_floor)

    fold_results: list[dict[str, object]] = []
    fold_idx = 0
    fold_start = 0
    n_total = len(full_ds)
    while fold_start + train_bars + test_bars <= n_total:
        if max_folds is not None and fold_idx >= max_folds:
            break
        fold_idx += 1
        train_end = fold_start + train_bars
        test_end = train_end + test_bars
        fold_train = full_ds.iloc[fold_start:train_end].copy()
        fold_test = full_ds.iloc[train_end:test_end].copy()
        if len(fold_train) < 500 or len(fold_test) < 100:
            fold_start += step_bars
            continue
        if test_start and str(fold_test["time"].max().date()) < test_start:
            fold_start += step_bars
            fold_idx -= 1
            continue

        scaler = StandardScaler()
        x_tr = scaler.fit_transform(fold_train[FEATURE_COLUMNS])
        x_te = scaler.transform(fold_test[FEATURE_COLUMNS])
        # Use realized_rr > 0 as label if requested (more aligned with trade outcome)
        if use_realized_rr_label and "realized_rr" in fold_train.columns:
            y_tr = (fold_train["realized_rr"] > 0).astype(int).values
            y_te = (fold_test["realized_rr"] > 0).astype(int).values
        else:
            y_tr = fold_train["target"].values
            y_te = fold_test["target"].values

        pos_count = int(y_tr.sum())
        neg_count = int(len(y_tr) - pos_count)
        if pos_count > 10 and neg_count > 10:
            pos_weight = 2.0 * neg_count / pos_count
            class_w = np.where(y_tr == 1, pos_weight, 1.0).astype(float)
            n = len(y_tr)
            decay_half = n * 0.4
            time_w = np.exp(np.log(2) * np.arange(n) / decay_half)
            time_w /= time_w.mean()
            sw = (class_w * time_w).astype(float)
            sw /= sw.mean()
        else:
            sw = None

        best_thr = _threshold_search(
            x_train=x_tr,
            y_train=y_tr,
            sample_weight=sw,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
            precision_floor=precision_floor,
        )

        scout = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        scout.fit(x_tr, y_tr, sample_weight=sw)
        importances = scout.feature_importances_
        imp_thr = np.percentile(importances, 30)
        feat_mask = importances >= imp_thr
        if feat_mask.sum() < 10:
            feat_mask = np.ones(len(importances), dtype=bool)
        x_tr_sel = x_tr[:, feat_mask]
        x_te_sel = x_te[:, feat_mask]

        hgb = HistGradientBoostingClassifier(
            max_iter=2000,
            learning_rate=0.01,
            max_depth=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            max_bins=128,
            class_weight=None,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=80,
            random_state=42,
        )
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=15,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        et = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        model = VotingClassifier(
            estimators=[("hgb", hgb), ("rf", rf), ("et", et)],
            voting="soft",
            weights=[3, 2, 1],
        )
        model.fit(x_tr_sel, y_tr, sample_weight=sw)
        proba = model.predict_proba(x_te_sel)[:, 1]
        preds = (proba >= best_thr).astype(int)

        auc = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else 0.5
        fold_frame = fold_test.copy()
        fold_frame["probability"] = proba
        fold_frame["prediction"] = preds
        fold_frame["base_threshold"] = best_thr
        fold_frame["fold"] = fold_idx
        fold_results.append(
            {
                "fold": fold_idx,
                "train_start": str(fold_train["time"].min().date()),
                "train_end": str(fold_train["time"].max().date()),
                "test_start": str(fold_test["time"].min().date()),
                "test_end": str(fold_test["time"].max().date()),
                "threshold": best_thr,
                "roc_auc": round(float(auc), 4),
                "precision": round(float(precision_score(y_te, preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_te, preds, zero_division=0)), 4),
                "f1": round(float(f1_score(y_te, preds, zero_division=0)), 4),
                "accuracy": round(float(accuracy_score(y_te, preds)), 4),
                "frame": fold_frame,
            }
        )
        print(
            f"Fold {fold_idx:02d} | {fold_results[-1]['test_start']} -> {fold_results[-1]['test_end']} | "
            f"AUC={fold_results[-1]['roc_auc']:.4f} Prec={fold_results[-1]['precision']:.4f} "
            f"Recall={fold_results[-1]['recall']:.4f} Thr={best_thr:.2f}"
        )
        fold_start += step_bars

    with pred_cache_path.open("wb") as fh:
        pickle.dump(fold_results, fh)
    print(f"Saved fold prediction cache: {pred_cache_path.name}")
    return fold_results


def _apply_candidate(settings: Settings, candidate: dict[str, object]) -> Settings:
    out = settings.model_copy(deep=True)
    strategy_cfg = candidate.get("strategy", {})
    risk_cfg = candidate.get("risk", {})
    execution_cfg = candidate.get("execution", {})
    for key, value in strategy_cfg.items():
        setattr(out.strategy, key, value)
    for key, value in risk_cfg.items():
        setattr(out.risk, key, value)
    if execution_cfg:
        for key, value in execution_cfg.items():
            if key == "trailing_sl" and isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    setattr(out.execution.trailing_sl, sub_key, sub_val)
            else:
                setattr(out.execution, key, value)
    return out


def _candidate_fingerprint(candidate: dict[str, object]) -> str:
    return _hash_json(candidate)


def simulate_candidate_daily_reset(
    name: str,
    base_settings: Settings,
    candidate: dict[str, object],
    fold_payloads: list[dict[str, object]],
    m1_df: pd.DataFrame,
    starting_balance: float,
    eval_start: str | None = None,
) -> dict[str, object]:
    settings = _apply_candidate(base_settings, candidate)
    ml_threshold_offset = float(candidate.get("ml_threshold_offset", 0.0))
    all_days: list[dict[str, object]] = []
    all_trades: list[pd.DataFrame] = []
    eval_start_ts = pd.Timestamp(eval_start, tz="UTC") if eval_start else None

    for fold_info in fold_payloads:
        fold_idx = int(fold_info["fold"])
        fold_df = fold_info["frame"].copy()
        base_thr = float(fold_info["threshold"])
        effective_thr = max(0.0, min(0.999, base_thr + ml_threshold_offset))
        fold_df["prediction"] = (fold_df["probability"] >= effective_thr).astype(int)
        fold_df["split"] = "test"
        fold_df["time"] = pd.to_datetime(fold_df["time"], utc=True, errors="coerce")
        if eval_start_ts is not None:
            fold_df = fold_df[fold_df["time"] >= eval_start_ts].copy()
        if fold_df.empty:
            continue
        for day_ts, day_df in fold_df.groupby(fold_df["time"].dt.floor("D"), sort=True):
            sim_settings = settings.model_copy(deep=True)
            sim_settings.training.backtest_initial_balance = starting_balance
            risk_mgr = RiskManager(sim_settings)
            sim = simulate_dynamic_concurrent_backtest(day_df.copy(), sim_settings, risk_mgr, m1_df=m1_df)
            rep = sim.report
            all_days.append(
                {
                    "candidate": name,
                    "fold": fold_idx,
                    "date": str(pd.Timestamp(day_ts).date()),
                    "test_start": fold_info["test_start"],
                    "test_end": fold_info["test_end"],
                    "ml_threshold": round(effective_thr, 4),
                    "net_profit": float(rep["net_profit"]),
                    "gross_profit": float(rep["gross_profit"]),
                    "gross_loss": float(rep["gross_loss"]),
                    "profit_factor": float(rep["profit_factor"]),
                    "max_drawdown_pct": abs(float(rep["max_drawdown_pct"])),
                    "trades": int(rep["trades"]),
                    "wins": int(rep["wins"]),
                    "losses": int(rep["losses"]),
                    "win_rate": float(rep["win_rate"]),
                    "signals_filtered_out": int(rep["signals_filtered_out"]),
                    "signals_no_slot": int(rep["signals_no_slot"]),
                    "signals_circuit_breaker": int(rep["signals_circuit_breaker"]),
                    "signals_reentry_guard": int(rep["signals_reentry_guard"]),
                    "signals_market_closed": int(rep["signals_market_closed"]),
                }
            )
            if not sim.trades.empty:
                t = sim.trades.copy()
                t["candidate"] = name
                t["fold"] = fold_idx
                t["trade_date"] = str(pd.Timestamp(day_ts).date())
                all_trades.append(t)

    day_df = pd.DataFrame(all_days).sort_values(["date", "fold"]).reset_index(drop=True)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    total_gp = float(day_df["gross_profit"].sum()) if not day_df.empty else 0.0
    total_gl = float(day_df["gross_loss"].sum()) if not day_df.empty else 0.0
    overall_pf = total_gp / total_gl if total_gl > 0 else 0.0
    worst_day_dd = float(day_df["max_drawdown_pct"].max()) if not day_df.empty else 0.0
    avg_day_dd = float(day_df["max_drawdown_pct"].mean()) if not day_df.empty else 0.0
    avg_daily_net = float(day_df["net_profit"].mean()) if not day_df.empty else 0.0
    med_daily_net = float(day_df["net_profit"].median()) if not day_df.empty else 0.0
    avg_trades_day = float(day_df["trades"].mean()) if not day_df.empty else 0.0
    trading_days = int((day_df["trades"] > 0).sum()) if not day_df.empty else 0
    total_days = int(len(day_df))

    fold_agg = (
        day_df.groupby("fold", as_index=False)
        .agg(
            days=("date", "count"),
            trading_days=("trades", lambda s: int((s > 0).sum())),
            net_profit=("net_profit", "sum"),
            avg_daily_net=("net_profit", "mean"),
            gross_profit=("gross_profit", "sum"),
            gross_loss=("gross_loss", "sum"),
            worst_day_dd=("max_drawdown_pct", "max"),
            avg_day_dd=("max_drawdown_pct", "mean"),
            trades=("trades", "sum"),
            neg_days=("net_profit", lambda s: int((s < 0).sum())),
            ge60_days=("net_profit", lambda s: int((s >= 60).sum())),
            ge100_days=("net_profit", lambda s: int((s >= 100).sum())),
        )
        if not day_df.empty
        else pd.DataFrame()
    )
    if not fold_agg.empty:
        fold_agg["profit_factor"] = np.where(
            fold_agg["gross_loss"] > 0,
            fold_agg["gross_profit"] / fold_agg["gross_loss"],
            0.0,
        )

    summary = {
        "candidate": name,
        "fingerprint": _candidate_fingerprint(candidate),
        "starting_balance_per_day": starting_balance,
        "days": total_days,
        "trading_days": trading_days,
        "total_net_profit": round(float(day_df["net_profit"].sum()) if not day_df.empty else 0.0, 2),
        "gross_profit": round(total_gp, 2),
        "gross_loss": round(total_gl, 2),
        "profit_factor": round(overall_pf, 4),
        "avg_daily_net": round(avg_daily_net, 2),
        "median_daily_net": round(med_daily_net, 2),
        "best_day": round(float(day_df["net_profit"].max()) if not day_df.empty else 0.0, 2),
        "worst_day": round(float(day_df["net_profit"].min()) if not day_df.empty else 0.0, 2),
        "avg_day_dd_pct": round(avg_day_dd, 2),
        "worst_day_dd_pct": round(worst_day_dd, 2),
        "avg_trades_per_day": round(avg_trades_day, 2),
        "total_trades": int(day_df["trades"].sum()) if not day_df.empty else 0,
        "pct_days_ge_60": round(float((day_df["net_profit"] >= 60).mean() * 100) if not day_df.empty else 0.0, 2),
        "pct_days_ge_100": round(float((day_df["net_profit"] >= 100).mean() * 100) if not day_df.empty else 0.0, 2),
        "pct_negative_days": round(float((day_df["net_profit"] < 0).mean() * 100) if not day_df.empty else 0.0, 2),
        "pct_positive_days": round(float((day_df["net_profit"] > 0).mean() * 100) if not day_df.empty else 0.0, 2),
        "pct_zero_days": round(float((day_df["net_profit"] == 0).mean() * 100) if not day_df.empty else 0.0, 2),
        "min_fold_avg_daily_net": round(float(fold_agg["avg_daily_net"].min()) if not fold_agg.empty else 0.0, 2),
        "avg_fold_avg_daily_net": round(float(fold_agg["avg_daily_net"].mean()) if not fold_agg.empty else 0.0, 2),
        "max_fold_worst_day_dd_pct": round(float(fold_agg["worst_day_dd"].max()) if not fold_agg.empty else 0.0, 2),
        "avg_fold_profit_factor": round(float(fold_agg["profit_factor"].mean()) if not fold_agg.empty else 0.0, 4),
        "config": candidate,
    }
    feasible = overall_pf >= 2.0 and worst_day_dd < 18.0
    summary["feasible"] = feasible
    score = (
        summary["avg_daily_net"]
        + 0.20 * summary["median_daily_net"]
        + 0.10 * summary["pct_days_ge_60"]
        + 0.05 * summary["pct_days_ge_100"]
        - 0.15 * summary["pct_negative_days"]
        - 0.25 * summary["worst_day_dd_pct"]
    )
    if not feasible:
        score -= 200.0
        if overall_pf < 2.0:
            score -= (2.0 - overall_pf) * 50.0
        if worst_day_dd >= 18.0:
            score -= (worst_day_dd - 18.0) * 10.0
    summary["objective_score"] = round(score, 4)
    return {"summary": summary, "days": day_df, "folds": fold_agg, "trades": trades_df}


def build_fold_predictions_lgbm(
    config_path: Path,
    full_ds: pd.DataFrame,
    use_cache: bool = True,
    max_folds: int | None = None,
    test_start: str | None = None,
) -> list[dict[str, object]]:
    """
    Honest walk-forward fold predictions using LightGBM per fold.
    
    KEY INTEGRITY RULES (no data leakage):
    - Each fold trains LGBM ONLY on fold_train[fold_start:train_end]
    - Calibration uses a HELD-OUT portion of fold_train NOT seen during LGBM fit
    - Test fold data [train_end:test_end] is NEVER touched during training
    - Early stopping uses a validation split from WITHIN fold_train only
    
    Exactly mirrors build_fold_predictions() structure but replaces VotingClassifier
    with LGBM + CalibratedClassifierCV(isotonic, cv='prefit').
    """
    try:
        import lightgbm as lgb
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        raise ImportError("LightGBM required: pip install lightgbm")

    settings = load_settings(config_path)
    cache_dir = ROOT / "outputs/.wf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_tag = (test_start or "all").replace("-", "")
    fold_tag = "all" if max_folds is None else str(max_folds)
    window_tag = "w30000_4000_4000"
    pred_cache_path = cache_dir / f"{config_path.stem}_lgbm_foldpreds_{window_tag}_{start_tag}_{fold_tag}.pkl"

    if use_cache and pred_cache_path.exists():
        with pred_cache_path.open("rb") as fh:
            payload = pickle.load(fh)
        print(f"Loaded LGBM fold prediction cache: {pred_cache_path.name} ({len(payload)} folds)")
        return payload

    train_bars = 30000
    test_bars  =  4000
    step_bars  =  4000
    # Use 80% of train for LGBM fit, 20% for calibration (no overlap)
    fit_ratio  = 0.80
    threshold_min  = float(settings.training.threshold_min)
    threshold_max  = float(settings.training.threshold_max)
    threshold_step = float(settings.training.threshold_step)
    precision_floor = float(settings.training.min_precision_floor)

    fold_results: list[dict[str, object]] = []
    fold_idx = 0
    fold_start = 0
    n_total = len(full_ds)

    while fold_start + train_bars + test_bars <= n_total:
        if max_folds is not None and fold_idx >= max_folds:
            break
        fold_idx += 1
        train_end = fold_start + train_bars
        test_end  = train_end + test_bars
        fold_train = full_ds.iloc[fold_start:train_end].copy()
        fold_test  = full_ds.iloc[train_end:test_end].copy()

        if len(fold_train) < 500 or len(fold_test) < 100:
            fold_start += step_bars
            continue
        if test_start and str(fold_test["time"].max().date()) < test_start:
            fold_start += step_bars
            fold_idx -= 1
            continue

        scaler  = StandardScaler()
        x_tr    = scaler.fit_transform(fold_train[FEATURE_COLUMNS])
        x_te    = scaler.transform(fold_test[FEATURE_COLUMNS])
        y_tr    = fold_train["target"].values
        y_te    = fold_test["target"].values

        # Time-decay + class-imbalance weights (same as VotingClassifier fold)
        pos_c = int(y_tr.sum())
        neg_c = int(len(y_tr) - pos_c)
        spw   = neg_c / max(pos_c, 1)  # scale_pos_weight
        if pos_c > 10 and neg_c > 10:
            n = len(y_tr)
            decay = n * 0.4
            time_w = np.exp(np.log(2) * np.arange(n) / decay)
            time_w /= time_w.mean()
            sw = time_w.astype(float)
        else:
            sw = None

        # Feature selection (RF scout, same as VotingClassifier)
        scout = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        scout.fit(x_tr, y_tr)
        imp = scout.feature_importances_
        feat_mask = imp >= np.percentile(imp, 30)
        if feat_mask.sum() < 10:
            feat_mask = np.ones(len(imp), dtype=bool)
        x_tr_sel = x_tr[:, feat_mask]
        x_te_sel = x_te[:, feat_mask]

        # Split fold_train: 80% for LGBM fit, 20% for calibration
        n_fit = int(len(x_tr_sel) * fit_ratio)
        x_fit, x_cal = x_tr_sel[:n_fit], x_tr_sel[n_fit:]
        y_fit, y_cal = y_tr[:n_fit], y_tr[n_fit:]
        sw_fit = sw[:n_fit] if sw is not None else None

        # Early stopping uses LAST 15% of fit portion (within x_fit only)
        n_val = int(len(x_fit) * 0.85)

        lgbm_base = lgb.LGBMClassifier(
            n_estimators=800,
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
            early_stopping_rounds=40,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        # LGBM fit: no data from x_cal or x_te_sel
        lgbm_base.fit(
            x_fit[:n_val], y_fit[:n_val],
            sample_weight=sw_fit[:n_val] if sw_fit is not None else None,
            eval_set=[(x_fit[n_val:], y_fit[n_val:])],
        )

        # Calibration: uses x_cal/y_cal (20% of fold_train NOT seen by LGBM fit)
        cal_model = CalibratedClassifierCV(lgbm_base, method="isotonic", cv="prefit")
        cal_model.fit(x_cal, y_cal)

        # Threshold search on calibration set (honest: same data cal model fit on, but
        # this is fine because threshold search is a post-hoc probability cutoff, not training)
        proba_cal = cal_model.predict_proba(x_cal)[:, 1]
        best_thr = threshold_max
        best_score = -float("inf")
        safe_thr, safe_prec = float(threshold_max), -1.0
        for thr in np.arange(threshold_min, threshold_max + threshold_step, threshold_step):
            preds = (proba_cal >= thr).astype(int)
            n_p   = int(preds.sum())
            if n_p < 3:
                continue
            prec = precision_score(y_cal, preds, zero_division=0)
            rec  = recall_score(y_cal, preds, zero_division=0)
            if rec < 0.05:
                continue
            if prec > safe_prec:
                safe_prec = prec
                safe_thr  = float(thr)
            if prec < precision_floor:
                continue
            score = prec * math.sqrt(rec)
            if score > best_score:
                best_score, best_thr = score, float(thr)
        if best_score == -float("inf"):
            best_thr = safe_thr

        # Predict on test fold (no leakage: test data never seen during training)
        proba_te = cal_model.predict_proba(x_te_sel)[:, 1]
        preds_te = (proba_te >= best_thr).astype(int)

        auc = roc_auc_score(y_te, proba_te) if len(np.unique(y_te)) > 1 else 0.5
        fold_frame = fold_test.copy()
        fold_frame["probability"]    = proba_te
        fold_frame["prediction"]     = preds_te
        fold_frame["base_threshold"] = best_thr
        fold_frame["fold"]           = fold_idx

        fold_results.append({
            "fold": fold_idx,
            "train_start": str(fold_train["time"].min().date()),
            "train_end":   str(fold_train["time"].max().date()),
            "test_start":  str(fold_test["time"].min().date()),
            "test_end":    str(fold_test["time"].max().date()),
            "threshold":   best_thr,
            "roc_auc":     round(float(auc), 4),
            "precision":   round(float(precision_score(y_te, preds_te, zero_division=0)), 4),
            "recall":      round(float(recall_score(y_te, preds_te, zero_division=0)), 4),
            "f1":          round(float(f1_score(y_te, preds_te, zero_division=0)), 4),
            "accuracy":    round(float(accuracy_score(y_te, preds_te)), 4),
            "frame":       fold_frame,
            "model_type":  "lgbm_calibrated_isotonic",
        })
        print(
            f"Fold {fold_idx:02d} [LGBM] | {fold_results[-1]['test_start']} -> {fold_results[-1]['test_end']} | "
            f"AUC={fold_results[-1]['roc_auc']:.4f} Prec={fold_results[-1]['precision']:.4f} "
            f"Recall={fold_results[-1]['recall']:.4f} Thr={best_thr:.2f}"
        )
        fold_start += step_bars

    with pred_cache_path.open("wb") as fh:
        pickle.dump(fold_results, fh)
    print(f"Saved LGBM fold prediction cache: {pred_cache_path.name}")
    return fold_results


def build_fold_predictions_lgbm(
    config_path: Path,
    full_ds: pd.DataFrame,
    use_cache: bool = True,
    max_folds: int | None = None,
    test_start: str | None = None,
) -> list[dict[str, object]]:
    """
    Honest walk-forward fold predictions using LightGBM per fold.
    
    KEY INTEGRITY RULES (no data leakage):
    - Each fold trains LGBM ONLY on fold_train[fold_start:train_end]
    - Calibration uses a HELD-OUT portion of fold_train NOT seen during LGBM fit
    - Test fold data [train_end:test_end] is NEVER touched during training
    - Early stopping uses a validation split from WITHIN fold_train only
    
    Exactly mirrors build_fold_predictions() structure but replaces VotingClassifier
    with LGBM + CalibratedClassifierCV(isotonic, cv='prefit').
    """
    try:
        import lightgbm as lgb
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        raise ImportError("LightGBM required: pip install lightgbm")

    settings = load_settings(config_path)
    cache_dir = ROOT / "outputs/.wf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_tag = (test_start or "all").replace("-", "")
    fold_tag = "all" if max_folds is None else str(max_folds)
    window_tag = "w30000_4000_4000"
    pred_cache_path = cache_dir / f"{config_path.stem}_lgbm_foldpreds_{window_tag}_{start_tag}_{fold_tag}.pkl"

    if use_cache and pred_cache_path.exists():
        with pred_cache_path.open("rb") as fh:
            payload = pickle.load(fh)
        print(f"Loaded LGBM fold prediction cache: {pred_cache_path.name} ({len(payload)} folds)")
        return payload

    train_bars = 30000
    test_bars  =  4000
    step_bars  =  4000
    # Use 80% of train for LGBM fit, 20% for calibration (no overlap)
    fit_ratio  = 0.80
    threshold_min  = float(settings.training.threshold_min)
    threshold_max  = float(settings.training.threshold_max)
    threshold_step = float(settings.training.threshold_step)
    precision_floor = float(settings.training.min_precision_floor)

    fold_results: list[dict[str, object]] = []
    fold_idx = 0
    fold_start = 0
    n_total = len(full_ds)

    while fold_start + train_bars + test_bars <= n_total:
        if max_folds is not None and fold_idx >= max_folds:
            break
        fold_idx += 1
        train_end = fold_start + train_bars
        test_end  = train_end + test_bars
        fold_train = full_ds.iloc[fold_start:train_end].copy()
        fold_test  = full_ds.iloc[train_end:test_end].copy()

        if len(fold_train) < 500 or len(fold_test) < 100:
            fold_start += step_bars
            continue
        if test_start and str(fold_test["time"].max().date()) < test_start:
            fold_start += step_bars
            fold_idx -= 1
            continue

        scaler  = StandardScaler()
        x_tr    = scaler.fit_transform(fold_train[FEATURE_COLUMNS])
        x_te    = scaler.transform(fold_test[FEATURE_COLUMNS])
        y_tr    = fold_train["target"].values
        y_te    = fold_test["target"].values

        # Time-decay + class-imbalance weights (same as VotingClassifier fold)
        pos_c = int(y_tr.sum())
        neg_c = int(len(y_tr) - pos_c)
        spw   = neg_c / max(pos_c, 1)  # scale_pos_weight
        if pos_c > 10 and neg_c > 10:
            n = len(y_tr)
            decay = n * 0.4
            time_w = np.exp(np.log(2) * np.arange(n) / decay)
            time_w /= time_w.mean()
            sw = time_w.astype(float)
        else:
            sw = None

        # Feature selection (RF scout, same as VotingClassifier)
        scout = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        scout.fit(x_tr, y_tr)
        imp = scout.feature_importances_
        feat_mask = imp >= np.percentile(imp, 30)
        if feat_mask.sum() < 10:
            feat_mask = np.ones(len(imp), dtype=bool)
        x_tr_sel = x_tr[:, feat_mask]
        x_te_sel = x_te[:, feat_mask]

        # Split fold_train: 80% for LGBM fit, 20% for calibration
        n_fit = int(len(x_tr_sel) * fit_ratio)
        x_fit, x_cal = x_tr_sel[:n_fit], x_tr_sel[n_fit:]
        y_fit, y_cal = y_tr[:n_fit], y_tr[n_fit:]
        sw_fit = sw[:n_fit] if sw is not None else None

        # Early stopping uses LAST 15% of fit portion (within x_fit only)
        n_val = int(len(x_fit) * 0.85)

        lgbm_base = lgb.LGBMClassifier(
            n_estimators=800,
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
            early_stopping_rounds=40,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        # LGBM fit: no data from x_cal or x_te_sel
        lgbm_base.fit(
            x_fit[:n_val], y_fit[:n_val],
            sample_weight=sw_fit[:n_val] if sw_fit is not None else None,
            eval_set=[(x_fit[n_val:], y_fit[n_val:])],
        )

        # Calibration: uses x_cal/y_cal (20% of fold_train NOT seen by LGBM fit)
        cal_model = CalibratedClassifierCV(lgbm_base, method="isotonic", cv="prefit")
        cal_model.fit(x_cal, y_cal)

        # Threshold search on calibration set (honest: same data cal model fit on, but
        # this is fine because threshold search is a post-hoc probability cutoff, not training)
        proba_cal = cal_model.predict_proba(x_cal)[:, 1]
        best_thr = threshold_max
        best_score = -float("inf")
        safe_thr, safe_prec = float(threshold_max), -1.0
        for thr in np.arange(threshold_min, threshold_max + threshold_step, threshold_step):
            preds = (proba_cal >= thr).astype(int)
            n_p   = int(preds.sum())
            if n_p < 3:
                continue
            prec = precision_score(y_cal, preds, zero_division=0)
            rec  = recall_score(y_cal, preds, zero_division=0)
            if rec < 0.05:
                continue
            if prec > safe_prec:
                safe_prec = prec
                safe_thr  = float(thr)
            if prec < precision_floor:
                continue
            score = prec * math.sqrt(rec)
            if score > best_score:
                best_score, best_thr = score, float(thr)
        if best_score == -float("inf"):
            best_thr = safe_thr

        # Predict on test fold (no leakage: test data never seen during training)
        proba_te = cal_model.predict_proba(x_te_sel)[:, 1]
        preds_te = (proba_te >= best_thr).astype(int)

        auc = roc_auc_score(y_te, proba_te) if len(np.unique(y_te)) > 1 else 0.5
        fold_frame = fold_test.copy()
        fold_frame["probability"]    = proba_te
        fold_frame["prediction"]     = preds_te
        fold_frame["base_threshold"] = best_thr
        fold_frame["fold"]           = fold_idx

        fold_results.append({
            "fold": fold_idx,
            "train_start": str(fold_train["time"].min().date()),
            "train_end":   str(fold_train["time"].max().date()),
            "test_start":  str(fold_test["time"].min().date()),
            "test_end":    str(fold_test["time"].max().date()),
            "threshold":   best_thr,
            "roc_auc":     round(float(auc), 4),
            "precision":   round(float(precision_score(y_te, preds_te, zero_division=0)), 4),
            "recall":      round(float(recall_score(y_te, preds_te, zero_division=0)), 4),
            "f1":          round(float(f1_score(y_te, preds_te, zero_division=0)), 4),
            "accuracy":    round(float(accuracy_score(y_te, preds_te)), 4),
            "frame":       fold_frame,
            "model_type":  "lgbm_calibrated_isotonic",
        })
        print(
            f"Fold {fold_idx:02d} [LGBM] | {fold_results[-1]['test_start']} -> {fold_results[-1]['test_end']} | "
            f"AUC={fold_results[-1]['roc_auc']:.4f} Prec={fold_results[-1]['precision']:.4f} "
            f"Recall={fold_results[-1]['recall']:.4f} Thr={best_thr:.2f}"
        )
        fold_start += step_bars

    with pred_cache_path.open("wb") as fh:
        pickle.dump(fold_results, fh)
    print(f"Saved LGBM fold prediction cache: {pred_cache_path.name}")
    return fold_results


def settings_to_candidate(settings: Settings, name: str) -> dict[str, object]:
    return {
        "name": name,
        "strategy": {
            "sideway_min_confidence": float(settings.strategy.sideway_min_confidence),
            "volatile_min_confidence": float(settings.strategy.volatile_min_confidence),
            "min_strategy_score": float(settings.strategy.min_strategy_score),
            "sideway_min_strategy_score": float(settings.strategy.sideway_min_strategy_score),
            "strong_volatility_min_strategy_score": float(settings.strategy.strong_volatility_min_strategy_score),
            "blocked_hours_utc": list(settings.strategy.blocked_hours_utc),
            "silver_bullet_confidence_boost": float(settings.strategy.silver_bullet_confidence_boost),
            "adx_min_trend": float(settings.strategy.adx_min_trend),
        },
        "risk": {
            "risk_per_trade": float(settings.risk.risk_per_trade),
            "risk_tier_floor": float(settings.risk.risk_tier_floor),
            "max_open_positions": int(settings.risk.max_open_positions),
            "min_confidence": float(settings.risk.min_confidence),
            "daily_loss_limit_pct": float(settings.risk.daily_loss_limit_pct),
            "max_drawdown_kill_pct": float(settings.risk.max_drawdown_kill_pct),
            "consecutive_loss_pause_count": int(settings.risk.consecutive_loss_pause_count),
            "consecutive_loss_cooldown_bars": int(settings.risk.consecutive_loss_cooldown_bars),
            "anti_martingale_factor": float(settings.risk.anti_martingale_factor),
            "anti_martingale_max_reductions": int(settings.risk.anti_martingale_max_reductions),
            "sideway_risk_multiplier": float(settings.risk.sideway_risk_multiplier),
            "normal_risk_multiplier": float(settings.risk.normal_risk_multiplier),
            "strong_volatility_risk_multiplier": float(settings.risk.strong_volatility_risk_multiplier),
            "reentry_cooldown_bars_after_sl": int(settings.risk.reentry_cooldown_bars_after_sl),
            "reentry_min_distance_atr": float(settings.risk.reentry_min_distance_atr),
            "partial_tp_enabled": bool(settings.risk.partial_tp_enabled),
            "partial_tp_rr": float(settings.risk.partial_tp_rr),
            "partial_tp_pct": float(settings.risk.partial_tp_pct),
        },
        "execution": {
            "close_opposite_on_signal": bool(settings.execution.close_opposite_on_signal),
            "trailing_sl": {
                "enabled": bool(settings.execution.trailing_sl.enabled),
                "breakeven_at_rr": float(settings.execution.trailing_sl.breakeven_at_rr),
                "activation_rr": float(settings.execution.trailing_sl.activation_rr),
                "trail_atr_multiple": float(settings.execution.trailing_sl.trail_atr_multiple),
            },
        },
        "ml_threshold_offset": 0.0,
    }


def _sample_from_space(rng: random.Random, base: dict[str, object]) -> dict[str, object]:
    blocked_templates = [
        [3, 15, 17, 22, 23],
        [22, 23],
        [3, 17, 22, 23],
        [2, 3, 17, 22, 23],
        [15, 17, 22, 23],
        [],
    ]
    cand = json.loads(json.dumps(base))
    cand["strategy"]["sideway_min_confidence"] = rng.choice([0.76, 0.80, 0.82, 0.83, 0.85, 0.88])
    cand["strategy"]["volatile_min_confidence"] = rng.choice([0.76, 0.80, 0.82, 0.83, 0.85, 0.88])
    cand["strategy"]["min_strategy_score"] = rng.choice([0.00, 0.03, 0.05, 0.07])
    cand["strategy"]["sideway_min_strategy_score"] = rng.choice([0.05, 0.08, 0.10, 0.12, 0.15])
    cand["strategy"]["strong_volatility_min_strategy_score"] = rng.choice([0.25, 0.30, 0.35])
    cand["strategy"]["blocked_hours_utc"] = rng.choice(blocked_templates)
    cand["strategy"]["silver_bullet_confidence_boost"] = rng.choice([0.03, 0.05, 0.08])
    cand["strategy"]["adx_min_trend"] = rng.choice([10.0, 12.0, 15.0, 18.0, 20.0])

    cand["risk"]["risk_per_trade"] = rng.choice([0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
    cand["risk"]["risk_tier_floor"] = rng.choice([0.02, 0.03, 0.04, 0.05, 0.06])
    cand["risk"]["max_open_positions"] = rng.choice([1, 2, 3, 4])
    cand["risk"]["min_confidence"] = rng.choice([0.76, 0.80, 0.82, 0.83, 0.85, 0.88])
    cand["risk"]["daily_loss_limit_pct"] = rng.choice([0.08, 0.10, 0.12, 0.15, 0.18])
    cand["risk"]["max_drawdown_kill_pct"] = rng.choice([0.10, 0.12, 0.14, 0.16, 0.18])
    cand["risk"]["consecutive_loss_pause_count"] = rng.choice([0, 2, 3])
    cand["risk"]["consecutive_loss_cooldown_bars"] = rng.choice([0, 6, 12])
    cand["risk"]["anti_martingale_factor"] = rng.choice([0.5, 0.7, 0.85, 1.0])
    cand["risk"]["anti_martingale_max_reductions"] = rng.choice([1, 2, 3])
    cand["risk"]["sideway_risk_multiplier"] = rng.choice([0.50, 0.75, 1.00])
    cand["risk"]["normal_risk_multiplier"] = rng.choice([1.00, 1.10, 1.20, 1.40])
    cand["risk"]["strong_volatility_risk_multiplier"] = rng.choice([0.60, 0.80, 1.00, 1.20])
    cand["risk"]["reentry_cooldown_bars_after_sl"] = rng.choice([3, 5, 8])
    cand["risk"]["reentry_min_distance_atr"] = rng.choice([0.25, 0.35, 0.50])
    cand["risk"]["partial_tp_enabled"] = rng.choice([True, False])
    cand["risk"]["partial_tp_rr"] = rng.choice([0.8, 1.0, 1.2, 1.5])
    cand["risk"]["partial_tp_pct"] = rng.choice([0.25, 0.5, 0.75])
    cand["execution"]["close_opposite_on_signal"] = rng.choice([True, False])
    cand["execution"]["trailing_sl"]["enabled"] = rng.choice([True, False])
    cand["execution"]["trailing_sl"]["breakeven_at_rr"] = rng.choice([0.3, 0.5, 0.8])
    cand["execution"]["trailing_sl"]["activation_rr"] = rng.choice([0.8, 1.0, 1.5])
    cand["execution"]["trailing_sl"]["trail_atr_multiple"] = rng.choice([0.6, 1.0, 1.4])
    cand["ml_threshold_offset"] = rng.choice([-0.05, -0.03, -0.01, 0.0, 0.02])
    return cand


def run_search(
    base_settings: Settings,
    fold_payloads: list[dict[str, object]],
    m1_df: pd.DataFrame,
    starting_balance: float,
    random_samples: int,
    seed: int,
    eval_start: str | None = None,
    extra_bases: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    baselines: list[dict[str, object]] = []
    for cfg_path in BASELINE_CONFIGS:
        if cfg_path.exists():
            st = load_settings(cfg_path)
            base_name = cfg_path.stem
            baseline = settings_to_candidate(st, base_name)
            baseline["name"] = base_name
            baselines.append(baseline)
    if extra_bases:
        for idx, base in enumerate(extra_bases, start=1):
            seeded = json.loads(json.dumps(base))
            seeded.setdefault("name", f"seed_{idx}")
            baselines.append(seeded)

    sampled: list[dict[str, object]] = []
    seen: set[str] = set()
    for base in baselines:
        fp = _candidate_fingerprint(base)
        if fp not in seen:
            seen.add(fp)
            sampled.append(base)
    base_pool = baselines[:] if baselines else [settings_to_candidate(base_settings, "base")]
    while len(sampled) < len(baselines) + random_samples:
        source = rng.choice(base_pool)
        cand = _sample_from_space(rng, source)
        fp = _candidate_fingerprint(cand)
        if fp in seen:
            continue
        seen.add(fp)
        sampled.append(cand)

    results: list[dict[str, object]] = []
    for idx, candidate in enumerate(sampled, start=1):
        cand_name = candidate.get("name") or f"cand_{idx:03d}"
        cand_name = f"{cand_name}_{idx:03d}" if idx > len(baselines) else cand_name
        result = simulate_candidate_daily_reset(
            name=cand_name,
            base_settings=base_settings,
            candidate=candidate,
            fold_payloads=fold_payloads,
            m1_df=m1_df,
            starting_balance=starting_balance,
            eval_start=eval_start,
        )
        results.append(result)
        s = result["summary"]
        mark = "PASS" if s["feasible"] else "FAIL"
        print(
            f"[{idx:03d}/{len(sampled):03d}] {mark} {cand_name} | "
            f"avg_day=${s['avg_daily_net']:.2f} med_day=${s['median_daily_net']:.2f} "
            f"PF={s['profit_factor']:.3f} worstDD={s['worst_day_dd_pct']:.2f}% "
            f">=60={s['pct_days_ge_60']:.1f}% neg={s['pct_negative_days']:.1f}%"
        )
    return results


def save_outputs(results: list[dict[str, object]], out_prefix: str) -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = pd.DataFrame([r["summary"] for r in results]).sort_values(
        ["feasible", "avg_daily_net", "profit_factor", "median_daily_net"],
        ascending=[False, False, False, False],
    )
    summaries.to_csv(out_dir / f"{out_prefix}_summary.csv", index=False)
    (out_dir / f"{out_prefix}_summary.json").write_text(
        json.dumps(summaries.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if results:
        best = summaries.iloc[0].to_dict()
        print(f"\nBest candidate saved under prefix: {out_prefix}")
        print(json.dumps(best, indent=2, ensure_ascii=False, default=str))


def load_seed_candidates(path: str | None, top_n: int) -> list[dict[str, object]]:
    if not path:
        return []
    seed_path = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed summary not found: {seed_path}")
    data = json.loads(seed_path.read_text(encoding="utf-8"))
    seeds: list[dict[str, object]] = []
    for idx, row in enumerate(data[:top_n], start=1):
        cfg = row.get("config")
        if not isinstance(cfg, dict):
            continue
        seeded = json.loads(json.dumps(cfg))
        seeded["name"] = row.get("candidate", f"seed_{idx}")
        seeds.append(seeded)
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acc1_v14pp_profit.yaml")
    parser.add_argument("--starting-balance", type=float, default=200.0)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--test-start", default="2024-08-14")
    parser.add_argument("--random-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-summary", default=None)
    parser.add_argument("--seed-top-n", type=int, default=3)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out-prefix", default="wf_dailyreset_v14pp")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    base_settings = load_settings(config_path)
    full_ds = load_or_build_dataset(config_path, cache=not args.no_cache)
    m1_df = load_m1()
    fold_payloads = build_fold_predictions(
        config_path=config_path,
        full_ds=full_ds,
        use_cache=not args.no_cache,
        max_folds=args.max_folds,
        test_start=args.test_start,
    )
    print(
        f"\nDaily-reset search on {len(fold_payloads)} folds | "
        f"test span {fold_payloads[0]['test_start']} -> {fold_payloads[-1]['test_end']}"
    )
    seed_candidates = load_seed_candidates(args.seed_summary, args.seed_top_n)
    results = run_search(
        base_settings=base_settings,
        fold_payloads=fold_payloads,
        m1_df=m1_df,
        starting_balance=args.starting_balance,
        random_samples=args.random_samples,
        seed=args.seed,
        eval_start=args.test_start,
        extra_bases=seed_candidates,
    )
    save_outputs(results, args.out_prefix)

    ranked = pd.DataFrame([r["summary"] for r in results]).sort_values(
        ["feasible", "avg_daily_net", "profit_factor", "median_daily_net"],
        ascending=[False, False, False, False],
    )
    print("\nTop 10 candidates:")
    cols = [
        "candidate",
        "feasible",
        "avg_daily_net",
        "median_daily_net",
        "profit_factor",
        "worst_day_dd_pct",
        "pct_days_ge_60",
        "pct_days_ge_100",
        "pct_negative_days",
        "avg_trades_per_day",
        "total_trades",
    ]
    print(ranked[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
