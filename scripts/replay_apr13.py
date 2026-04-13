#!/usr/bin/env python3
"""
Replay WF backtest specifically on April 13, 2026 and compare with live trades.
Uses fold 8 model (train on ~250k bars ending Feb 2026, test Apr 2026).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from acc1_friction_opt_wf import _add_session_spread_mult, _SESSION_SPREAD_MULT, _SESSION_SLIPPAGE_MULT  # type: ignore
from acc2_m1_setup_exit_wf import _build_fold_cache_setup, _evaluate_candidate  # type: ignore
from xauusd_ai.config import load_settings
from xauusd_ai.features.scalp_dataset import build_scalp_dataset

TARGET_START = "2026-04-12"
TARGET_END   = "2026-04-13"

# ── ACC1 best params from WF run today ────────────────────────────────────────
ACC1_BEST = {
    "threshold": 0.59, "risk_per_trade": 0.026,
    "sideway_risk_multiplier": 0.75, "strong_volatility_risk_multiplier": 1.2,
    "max_open_positions": 2, "daily_loss_limit_pct": 0.20,
    "setup_exit_enabled": True, "thr_premium": 0.0,
    "hour_profile": "config", "cooldown_bars": 4, "setup_exit_scale": 0.75,
    "weekday_profile": "all", "side_profile": "both", "quality_gate_profile": "off",
    "session_spread_mult": _SESSION_SPREAD_MULT,
    "session_slippage_mult": _SESSION_SLIPPAGE_MULT,
}

ACC2_BEST = {
    "threshold": 0.60, "risk_per_trade": 0.020,
    "sideway_risk_multiplier": 0.75, "strong_volatility_risk_multiplier": 1.2,
    "max_open_positions": 2, "daily_loss_limit_pct": 0.20,
    "setup_exit_enabled": True, "thr_premium": 0.0,
    "hour_profile": "config", "cooldown_bars": 4, "setup_exit_scale": 0.80,
    "weekday_profile": "all", "side_profile": "both", "quality_gate_profile": "off",
    "session_spread_mult": _SESSION_SPREAD_MULT,
    "session_slippage_mult": _SESSION_SLIPPAGE_MULT,
}

def run_replay(acc: str, config_path: Path, best_params: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  {acc} REPLAY — {TARGET_START} to {TARGET_END}")
    print(f"{'='*70}")

    settings = load_settings(config_path)

    print("[1] Building dataset (2023-01-01 → latest)...")
    df = build_scalp_dataset(start_date="2023-01-01", max_horizon=8, setup_label_mode="setup_aware")
    print(f"    {len(df):,} rows")

    # Build fold 8 only (largest start = last 250k train + 50k test ending at Apr 13)
    total = len(df)
    train_size, test_size = 250_000, 50_000
    s = total - train_size - test_size
    if s < 0:
        print(f"[error] Not enough data: {total} rows, need {train_size + test_size}")
        return

    print(f"[2] Building fold 8 only (train start row={s:,})...")
    fold_cache = _build_fold_cache_setup(
        dataset=df,
        train_size=train_size,
        test_size=test_size,
        step_size=test_size,
        n_folds=1,
        label_horizon=8,
        train_objective="day_stability_strict",
    )
    if not fold_cache:
        print("[error] No folds built")
        return

    _add_session_spread_mult(fold_cache)
    fold = fold_cache[0]
    print(f"    Fold test period: {fold.test_start[:16]} → {fold.test_end[:16]}")

    # Use simulate_dynamic_concurrent_backtest directly for trade-level output
    from xauusd_ai.backtesting import engine as eng

    # Reconstruct what _evaluate_candidate does internally for this fold
    thr = float(best_params["threshold"])
    rf  = float(best_params["risk_per_trade"])
    max_pos = int(best_params["max_open_positions"])
    daily_limit = float(best_params["daily_loss_limit_pct"])
    cooldown = int(best_params["cooldown_bars"])
    scale = float(best_params.get("setup_exit_scale", 0.75))

    # ── Patch _evaluate_candidate to capture trades per fold ──────────────
    from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
    from xauusd_ai.execution.risk import RiskManager
    from acc2_m1_setup_exit_wf import _evaluate_candidate as _eval  # type: ignore

    # Build a minimal candidate dict matching what acc1_friction_opt_wf expects
    cand = {
        "thr_buy":                            thr,
        "thr_sell":                           thr,
        "risk_per_trade":                     rf,
        "max_open_positions":                 max_pos,
        "daily_loss_limit_pct":               daily_limit,
        "sideway_risk_multiplier":            float(best_params.get("sideway_risk_multiplier", 0.75)),
        "strong_volatility_risk_multiplier":  float(best_params.get("strong_volatility_risk_multiplier", 1.2)),
        "cooldown_bars":                      cooldown,
        "setup_exit_enabled":                 bool(best_params.get("setup_exit_enabled", True)),
        "setup_exit_scale":                   scale,
        "thr_premium":                        0.0,
        "hour_profile":                       "config",
        "weekday_profile":                    "all",
        "side_profile":                       "both",
        "quality_gate_profile":               "off",
        "daily_signal_cap":                   0,
        "session_spread_mult":                _SESSION_SPREAD_MULT,
        "session_slippage_mult":              _SESSION_SLIPPAGE_MULT,
    }

    # Monkeypatch simulate to capture trade DFs
    captured_trades = []
    _orig_sim = simulate_dynamic_concurrent_backtest

    def _capture_sim(preds, s, rm, label="test", compound=True):
        res = _orig_sim(preds, s, rm, label=label, compound=compound)
        captured_trades.append(res.trades)
        return res

    import acc2_m1_setup_exit_wf as _wf_mod
    _wf_mod.simulate_dynamic_concurrent_backtest = _capture_sim

    _eval(cand, fold_cache, settings)

    _wf_mod.simulate_dynamic_concurrent_backtest = _orig_sim  # restore

    if not captured_trades or captured_trades[0] is None or len(captured_trades[0]) == 0:
        print("[warn] No trades captured in fold replay")
        return

    all_trades = pd.concat(captured_trades, ignore_index=True)
    result_trades = all_trades

    trades = result_trades
    if trades is None or len(trades) == 0:
        print("[warn] No trades generated in fold 8")
        return

    # Filter April 12-13
    trades["time"] = pd.to_datetime(trades["time"], utc=True, errors="coerce")
    apr_filter = trades["time"].dt.date.astype(str).between(TARGET_START, TARGET_END)
    window = trades[apr_filter].copy().sort_values("time")

    total_pnl = window["pnl"].sum()
    wins = (window["pnl"] > 0).sum()
    losses = (window["pnl"] < 0).sum()

    print(f"\n  {acc} WF SIMULATION — {TARGET_START} to {TARGET_END} | balance=$200")
    print(f"  {len(window)} trades | PnL={total_pnl:+.2f}$ | W={wins} L={losses}")
    print()
    print(f"  {'Date-Time(UTC)':<16} {'Side':<5} {'Entry':>9} {'Exit':>9} {'PnL':>7}  Result")
    print(f"  {'-'*66}")
    for _, r in window.iterrows():
        t = str(r["time"])[:16].replace("T", " ")
        side = str(r["side"]).upper()
        entry = float(r.get("entry_price", 0))
        exit_ = float(r.get("exit_price", 0))
        pnl   = float(r["pnl"])
        mark  = "WIN" if pnl > 0 else "LOSS"
        print(f"  {t:<16} {side:<5} {entry:>9.3f} {exit_:>9.3f} {pnl:>+7.2f}  {mark}")

    print()
    print(f"  Full fold 8 → trades={len(trades)} PnL={trades['pnl'].sum():+.2f}$")
    print(f"  WR={trades['is_win'].mean()*100:.1f}%" if "is_win" in trades.columns else "")


if __name__ == "__main__":
    run_replay("ACC1", Path("configs/live_acc1_scalp_m1.yaml"), ACC1_BEST)
    run_replay("ACC2", Path("configs/live_acc2_scalp_m1.yaml"), ACC2_BEST)
