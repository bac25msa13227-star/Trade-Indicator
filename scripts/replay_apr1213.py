#!/usr/bin/env python3
"""
Replay backtest cho ngày 12-13/04/2026 với vốn khởi đầu $200.
- Train: 250k bars kết thúc trước 12/04
- Test: chỉ Apr 12-13/2026 (bắt đầu từ $200)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from acc1_friction_opt_wf import _add_session_spread_mult, _SESSION_SPREAD_MULT, _SESSION_SLIPPAGE_MULT  # type: ignore
from acc2_m1_setup_exit_wf import (  # type: ignore
    _build_fold_cache_setup,
    _allowed_hours_from_profile,
    _allowed_weekdays_from_profile,
    _apply_daily_signal_cap,
    _apply_quality_gate,
)
from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import load_settings
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.scalp_dataset import build_scalp_dataset

TARGET_START = pd.Timestamp("2026-04-12 00:00:00", tz="UTC")
TARGET_END   = pd.Timestamp("2026-04-13 23:59:59", tz="UTC")
INITIAL_BALANCE = 200.0

ACC1_BEST = dict(
    thr_buy=0.59, thr_sell=0.59, risk_per_trade=0.026,
    max_open_positions=2, daily_loss_limit_pct=0.20,
    sideway_risk_multiplier=0.75, strong_volatility_risk_multiplier=1.2,
    cooldown_bars=4, setup_exit_enabled=True, setup_exit_scale=0.75,
    thr_premium=0.0, hour_profile="config", weekday_profile="all",
    side_profile="both", quality_gate_profile="off", daily_signal_cap=0,
)
ACC2_BEST = dict(
    thr_buy=0.60, thr_sell=0.60, risk_per_trade=0.020,
    max_open_positions=2, daily_loss_limit_pct=0.20,
    sideway_risk_multiplier=0.75, strong_volatility_risk_multiplier=1.2,
    cooldown_bars=4, setup_exit_enabled=True, setup_exit_scale=0.80,
    thr_premium=0.0, hour_profile="config", weekday_profile="all",
    side_profile="both", quality_gate_profile="off", daily_signal_cap=0,
)


def _build_preds(fold, cand: dict) -> pd.DataFrame:
    """Reproduce signal building from _evaluate_candidate for one fold."""
    dirs       = fold.directions
    buy_probs  = fold.buy_probs.copy()
    sell_probs = fold.sell_probs.copy()
    preds = fold.base_df.copy()

    preds["probability"] = np.where(dirs == -1, sell_probs, buy_probs)
    preds["trade_side"]  = np.where(dirs == -1, "sell", "buy")

    thr_b = float(cand["thr_buy"])
    thr_s = float(cand["thr_sell"])
    buy_sig  = (dirs == 1)  & (buy_probs  >= thr_b)
    sell_sig = (dirs == -1) & (sell_probs >= thr_s)

    preds["prediction"] = 0
    preds.loc[buy_sig,  "prediction"] = 1
    preds.loc[sell_sig, "prediction"] = 1

    # Setup-exit scaling
    setup_exit_enabled = bool(cand.get("setup_exit_enabled", True))
    setup_exit_scale   = float(cand.get("setup_exit_scale", 1.0))
    if setup_exit_enabled and "setup_tp_rr" in preds.columns:
        preds["setup_tp_rr"]        = preds["setup_tp_rr"] * setup_exit_scale
        preds["setup_exit_scaled"]  = 1
    else:
        preds["setup_tp_rr"]        = 0.0
        preds["setup_sl_mult"]      = 0.0
        preds["setup_exit_scaled"]  = 0

    # Hour filter
    allowed_hours = _allowed_hours_from_profile(str(cand.get("hour_profile", "config")))
    if allowed_hours is not None and "time" in preds.columns:
        t = pd.to_datetime(preds["time"], utc=True, errors="coerce")
        preds.loc[~t.dt.hour.isin(sorted(allowed_hours)).fillna(False), "prediction"] = 0

    # Weekday filter
    allowed_wd = _allowed_weekdays_from_profile(str(cand.get("weekday_profile", "all")))
    if allowed_wd is not None and "time" in preds.columns:
        t = pd.to_datetime(preds["time"], utc=True, errors="coerce")
        preds.loc[~t.dt.weekday.isin(sorted(allowed_wd)).fillna(False), "prediction"] = 0

    # Side filter
    sp = str(cand.get("side_profile", "both"))
    if sp == "buy_only":
        preds.loc[preds["trade_side"] != "buy", "prediction"] = 0
    elif sp == "sell_only":
        preds.loc[preds["trade_side"] != "sell", "prediction"] = 0

    preds = _apply_quality_gate(preds, str(cand.get("quality_gate_profile", "off")))
    preds = _apply_daily_signal_cap(preds, int(cand.get("daily_signal_cap", 0)))

    return preds


def run_replay(acc: str, config_path: Path, best_params: dict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {acc}  |  {TARGET_START.date()} → {TARGET_END.date()}  |  Vốn: ${INITIAL_BALANCE:.0f}")
    print(sep)

    settings = load_settings(config_path)

    print("[1] Loading dataset (2023-01-01 → latest)…")
    df = build_scalp_dataset(start_date="2023-01-01", max_horizon=8, setup_label_mode="setup_aware")
    print(f"    {len(df):,} rows")

    total = len(df)
    train_size, test_size = 250_000, 50_000
    if total < train_size + test_size:
        print(f"[lỗi] Không đủ dữ liệu: {total} rows"); return

    print("[2] Training model (250k bars → test Feb-Apr 2026)…")
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
        print("[lỗi] Không build được fold"); return

    _add_session_spread_mult(fold_cache)
    fold = fold_cache[0]
    print(f"    Giai đoạn test: {fold.test_start[:10]} → {fold.test_end[:10]}")

    # Build full predictions then filter to Apr 12-13
    print("[3] Filtering predictions to Apr 12-13 only…")
    preds = _build_preds(fold, best_params)
    preds["time"] = pd.to_datetime(preds["time"], utc=True, errors="coerce")
    mask = (preds["time"] >= TARGET_START) & (preds["time"] <= TARGET_END)
    preds_window = preds[mask].copy().reset_index(drop=True)

    total_signals = int((preds_window["prediction"] == 1).sum())
    print(f"    {len(preds_window):,} bars  |  {total_signals} signals")
    if total_signals == 0:
        print("    ⚠  Không có tín hiệu nào trong khoảng thời gian này")
        return

    # Configure settings for the window
    s = settings.model_copy(deep=True)
    s.risk.risk_per_trade                   = float(best_params["risk_per_trade"])
    s.risk.max_open_positions               = int(best_params["max_open_positions"])
    s.risk.daily_loss_limit_pct             = float(best_params["daily_loss_limit_pct"])
    s.risk.sideway_risk_multiplier          = float(best_params.get("sideway_risk_multiplier", 0.75))
    s.risk.strong_volatility_risk_multiplier= float(best_params.get("strong_volatility_risk_multiplier", 1.2))
    s.risk.consecutive_loss_pause_count     = 2
    s.risk.consecutive_loss_cooldown_bars   = int(best_params["cooldown_bars"])
    s.risk.kill_switch_enabled              = True
    s.risk.setup_exit_enabled               = bool(best_params.get("setup_exit_enabled", True))
    s.risk.setup_exit_scale                 = float(best_params.get("setup_exit_scale", 1.0))
    s.risk.max_risk_fraction                = min(2.50, float(best_params["risk_per_trade"]) * 1.8)
    s.risk.max_total_exposure_pct           = min(4.00, float(best_params["risk_per_trade"]) * 2.5)
    # ── Reentry guard: mirror live config exactly ─────────────────────────
    s.risk.reentry_guard_enabled          = bool(getattr(settings.risk, "reentry_guard_enabled", True))
    s.risk.reentry_cooldown_bars_after_sl = int(getattr(settings.risk, "reentry_cooldown_bars_after_sl", 3))
    s.risk.reentry_min_distance_atr       = float(getattr(settings.risk, "reentry_min_distance_atr", 0.35))
    # ── Sync regime thresholds with candidate threshold ───────────────────
    _cand_thr = min(float(best_params["thr_buy"]), float(best_params["thr_sell"]))
    s.risk.min_confidence                 = _cand_thr
    s.strategy.sideway_min_confidence     = _cand_thr
    s.strategy.volatile_min_confidence    = _cand_thr
    s.strategy.adx_gate_enabled             = False
    s.strategy.min_strategy_score           = 0.0
    s.strategy.require_trend_alignment      = False
    s.training.backtest_initial_balance     = INITIAL_BALANCE

    print("[4] Running simulation with $200 balance…")
    preds_window["split"] = "apr1213"   # required by engine filter
    sim = simulate_dynamic_concurrent_backtest(
        preds_window, s, RiskManager(s), label="apr1213", compound=True
    )
    trades = sim.trades
    rep   = sim.report

    # Debug skip counters
    print(f"    Traded: {rep.get('trades',0)}  "
          f"| filtered: {rep.get('signals_filtered_out',0)}  "
          f"| circuit_breaker: {rep.get('skipped_circuit_breaker',0)}  "
          f"| no_slot: {rep.get('skipped_no_slot',0)}  "
          f"| no_trade_side: {rep.get('skipped_no_trade_side',0)}")
    if trades is None or len(trades) == 0:
        print("    ⚠  Không có trade nào được thực thi")
        print(f"    Full report: {rep}")
        return

    trades["time"] = pd.to_datetime(trades["time"], utc=True, errors="coerce")
    trades = trades.sort_values("time").reset_index(drop=True)

    # ── Summary ─────────────────────────────────────────────────────────────
    total_pnl  = float(trades["pnl"].sum())
    wins       = int((trades["pnl"] > 0).sum())
    losses     = int((trades["pnl"] <= 0).sum())
    win_rate   = wins / max(wins + losses, 1) * 100
    rep        = sim.report
    final_bal  = INITIAL_BALANCE + total_pnl
    pf         = float(rep.get("profit_factor", 0.0))

    print()
    print(f"  {'─'*70}")
    print(f"  Kết quả: {len(trades)} trades  |  W={wins} L={losses}  |  WR={win_rate:.0f}%  |  PF={pf:.2f}")
    print(f"  Tổng PnL: {total_pnl:+.2f}$   |  Số dư cuối: {final_bal:.2f}$")
    print(f"  {'─'*70}")
    print()
    print(f"  {'Thời gian (UTC)':<19} {'Side':<5} {'Entry':>9} {'SL mức':>9} {'RR net':>7} {'PnL':>8}  Kết quả")
    print(f"  {'─'*70}")

    for _, r in trades.iterrows():
        t      = str(r["time"])[:16].replace("T", " ")
        side   = str(r.get("side", r.get("trade_side", "?"))).upper()
        entry  = float(r.get("entry_price", 0))
        sl     = float(r.get("sl_marker_price", 0))
        net_rr = float(r.get("net_rr", 0))
        pnl    = float(r["pnl"])
        mark   = "WIN  ✅" if pnl > 0 else "LOSS ❌"
        print(f"  {t:<19} {side:<5} {entry:>9.3f} {sl:>9.3f} {net_rr:>+7.2f} {pnl:>+8.2f}  {mark}")

    print()
    print(f"  Ghi chú: Entry=giá đóng cửa bar. SL mức=giá stop loss. RR net=R:R sau friction.")
    print(f"  Số dư tại Apr 12: ${INITIAL_BALANCE:.2f}  →  Cuối Apr 13: ${final_bal:.2f}")
    print()

    # Apr 13 only summary
    apr13 = trades[trades["time"].dt.date.astype(str) == "2026-04-13"]
    if len(apr13) > 0:
        p13   = float(apr13["pnl"].sum())
        w13   = int((apr13["pnl"] > 0).sum())
        l13   = int((apr13["pnl"] <= 0).sum())
        print(f"  [Chỉ Apr 13] {len(apr13)} trades | W={w13} L={l13} | PnL={p13:+.2f}$")
    print()


if __name__ == "__main__":
    run_replay("ACC1", Path("configs/live_acc1_scalp_m1.yaml"), ACC1_BEST)
    run_replay("ACC2", Path("configs/live_acc2_scalp_m1.yaml"), ACC2_BEST)
