from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.execution.sltp import (
    base_sltp_by_regime,
    compute_dynamic_sltp,
    resolve_setup_exit_targets,
)
from xauusd_ai.strategies.hybrid import HybridStrategy
from xauusd_ai.visualization.reports import save_backtest_plots


@dataclass
class SimulationResult:
    report: dict[str, object]
    trades: pd.DataFrame


def simulate_prediction_backtest(
    predictions: pd.DataFrame,
    settings: Settings,
    risk_manager: RiskManager,
    compound: bool = True,
) -> SimulationResult:
    """
    compound=True  → balance compounds with each trade (for short windows like walkforward folds).
    compound=False → PnL always uses start_bal (linear/non-compounding — for long-window backtest
                     display to avoid astronomical numbers from high-WR strategies).
    """
    test_rows = predictions[predictions["split"] == "test"].copy()
    strategy = HybridStrategy(settings)
    start_bal = settings.training.backtest_initial_balance
    balance = start_bal
    peak_balance = balance
    wins = 0
    losses = 0
    trades: list[dict[str, object]] = []
    skipped_by_filters = 0

    # Friction components (fraction of 1R deducted per trade)
    _spread_rr = float(getattr(settings.risk, "spread_cost_rr", 0.10))
    _slippage_rr = float(getattr(settings.risk, "slippage_rr", 0.05))
    _commission_rr = float(getattr(settings.risk, "commission_rr", 0.02))
    friction_rr = _spread_rr + _slippage_rr + _commission_rr
    compound_cap = float(getattr(settings.risk, "compound_cap", 50.0))
    max_balance = start_bal * compound_cap if compound_cap > 0 else float("inf")

    for row in test_rows.itertuples(index=False):
        if row.prediction == 0:
            continue
        allowed, reason = strategy.should_allow_row(row, float(row.probability))
        if not allowed:
            skipped_by_filters += 1
            continue
        risk_fraction, throttle_mult, throttle_reason = risk_manager.apply_risk_throttle(
            settings.risk.risk_per_trade,
            row,
            side=str(getattr(row, "trade_side", "")),
            probability=float(row.probability),
        )
        effective_bal = balance if compound else start_bal
        if compound and compound_cap > 0 and effective_bal > max_balance:
            effective_bal = max_balance
        # P1a: Session-aware friction
        _sess_mult = float(getattr(row, 'session_spread_mult', 1.0))
        _slip_mult = float(getattr(row, 'session_slippage_mult', 1.0))
        _row_friction = _spread_rr * _sess_mult + _slippage_rr * _slip_mult + _commission_rr
        net_rr = row.realized_rr - _row_friction
        pnl = effective_bal * risk_fraction * net_rr
        balance_before = balance
        balance += pnl
        peak_balance = max(peak_balance, balance)
        drawdown = (balance - peak_balance) / peak_balance if peak_balance else 0.0
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        trades.append(
            {
                "time": row.time.isoformat() if hasattr(row.time, "isoformat") else str(row.time),
                "side": row.trade_side,
                "entry_price": float(row.close),
                "exit_price": float(row.close * (1 + row.future_return)),
                "future_return": float(row.future_return),
                "directional_return": float(row.directional_return),
                "realized_rr": float(row.realized_rr),
                "probability": float(row.probability),
                "risk_fraction": float(risk_fraction),
                "pnl": float(pnl),
                "balance_before": float(balance_before),
                "balance_after": float(balance),
                "drawdown": float(drawdown),
                "is_win": bool(pnl > 0),
                "is_loss": bool(pnl < 0),
                "is_draw": bool(pnl == 0),
                "skip_filter_reason": reason,
                "risk_throttle_multiplier": float(throttle_mult),
                "risk_throttle_reason": throttle_reason,
            }
        )

    trades_df = pd.DataFrame(trades)
    max_drawdown = 0.0 if trades_df.empty else float(trades_df["drawdown"].min())
    profit_factor = 0.0
    avg_holding_bars = 0.0
    sharpe = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    if not trades_df.empty:
        gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        avg_holding_bars = float(predictions.get("bars_held", pd.Series([settings.training.label_horizon])).mean())
        pnl_std = float(trades_df["pnl"].std()) if len(trades_df) > 1 else 0.0
        sharpe = float((trades_df["pnl"].mean() / pnl_std) * (len(trades_df) ** 0.5)) if pnl_std > 0 else 0.0
        avg_win = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()) if wins > 0 else 0.0
        avg_loss = float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()) if losses > 0 else 0.0

    start_bal = settings.training.backtest_initial_balance
    draws = len(trades) - wins - losses
    report = {
        "starting_balance": start_bal,
        "ending_balance": round(balance, 2),
        "net_profit": round(balance - start_bal, 2),
        "return_pct": round((balance / start_bal - 1) * 100, 2),
        "trades": len(trades),
        "signals_filtered_out": skipped_by_filters,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(wins / max(wins + losses, 1), 4),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "best_trade": round(float(trades_df["pnl"].max()) if not trades_df.empty else 0.0, 2),
        "worst_trade": round(float(trades_df["pnl"].min()) if not trades_df.empty else 0.0, 2),
        "sharpe_like": round(sharpe, 4),
        "avg_holding_bars": round(avg_holding_bars, 2),
        "friction_rr": round(friction_rr, 4),
        "compound_cap": compound_cap,
    }
    return SimulationResult(report=report, trades=trades_df)


def simulate_dynamic_concurrent_backtest(
    predictions: pd.DataFrame,
    settings: Settings,
    risk_manager: RiskManager,
    label: str = "test",
    compound: bool = True,
) -> SimulationResult:
    """
    Concurrent position simulation with dynamic slot scaling and realistic friction.

    Realism layers (configurable via RiskSettings):
      - spread_cost_rr:  spread deducted per trade as fraction of 1R
      - slippage_rr:     slippage deducted per trade as fraction of 1R
      - commission_rr:   commission deducted per trade as fraction of 1R
      - compound_cap:    max balance = start_bal × compound_cap (prevents explosion)

    compound=True  → balance compounds per trade (default — use for short windows / walkforward).
    compound=False → PnL always uses start_bal (linear / non-compounding — use for long-window
                     backtest display to avoid astronomical numbers from high-WR strategies).
    """
    test_rows = (
        predictions[predictions["split"] == label]
        .copy()
        .sort_values("time")
        .reset_index(drop=True)
    )
    strategy = HybridStrategy(settings)
    start_bal = settings.training.backtest_initial_balance
    balance = start_bal
    peak_balance = balance
    label_horizon = settings.training.label_horizon  # bars to hold each position

    # Friction components (fraction of 1R deducted per trade)
    _spread_rr = float(getattr(settings.risk, "spread_cost_rr", 0.10))
    _slippage_rr = float(getattr(settings.risk, "slippage_rr", 0.05))
    _commission_rr = float(getattr(settings.risk, "commission_rr", 0.02))
    friction_rr = _spread_rr + _slippage_rr + _commission_rr
    compound_cap = float(getattr(settings.risk, "compound_cap", 50.0))
    max_balance = start_bal * compound_cap if compound_cap > 0 else float("inf")
    dynamic_sltp_eval_enabled = bool(getattr(settings.risk, "dynamic_sltp_backtest_enabled", False))
    has_ohlc_for_dynamic = {"close", "high", "low"}.issubset(test_rows.columns)
    close_arr = (
        pd.to_numeric(test_rows["close"], errors="coerce").to_numpy(dtype=float)
        if "close" in test_rows.columns
        else np.array([], dtype=float)
    )
    high_arr = (
        pd.to_numeric(test_rows["high"], errors="coerce").to_numpy(dtype=float)
        if "high" in test_rows.columns
        else None
    )
    low_arr = (
        pd.to_numeric(test_rows["low"], errors="coerce").to_numpy(dtype=float)
        if "low" in test_rows.columns
        else None
    )
    ms_atr_norm_arr = (
        pd.to_numeric(test_rows["ms_atr5_norm"], errors="coerce").to_numpy(dtype=float)
        if "ms_atr5_norm" in test_rows.columns
        else None
    )
    bars_held_arr = (
        pd.to_numeric(test_rows["bars_held"], errors="coerce").fillna(label_horizon).to_numpy(dtype=int)
        if "bars_held" in test_rows.columns
        else np.full(len(test_rows), int(label_horizon), dtype=int)
    )
    dynamic_sltp_eval_active = dynamic_sltp_eval_enabled and has_ohlc_for_dynamic

    # pending: list of {"close_bar_idx": int, "absolute_pnl": float, "meta": dict}
    pending: list[dict] = []
    trades: list[dict] = []
    wins = 0
    losses = 0
    skipped_by_filters = 0
    skipped_no_slot = 0
    skipped_circuit_breaker = 0  # NEW: track circuit breaker blocks
    skipped_reentry_guard = 0
    dynamic_rr_recomputed = 0
    dynamic_rr_fallback = 0
    # Same-side re-entry guard after a loss (simulate live guard behavior).
    _last_loss_marker_by_side: dict[str, dict[str, float | int]] = {}

    balance_history: list[float] = [balance]
    max_concurrent_seen = 0

    # Circuit breaker state for backtest realism
    _consecutive_losses = 0
    _cooldown_remaining = 0
    _daily_loss = 0.0
    _current_date = ""
    _anti_mart_factor = float(getattr(settings.risk, "anti_martingale_factor", 0.6))
    _anti_mart_max = int(getattr(settings.risk, "anti_martingale_max_reductions", 3))
    _pause_count = int(getattr(settings.risk, "consecutive_loss_pause_count", 3))
    _cooldown_bars = int(getattr(settings.risk, "consecutive_loss_cooldown_bars", 8))
    _daily_limit = float(getattr(settings.risk, "daily_loss_limit_pct", 0.0))
    _max_dd_kill_pct = float(getattr(settings.risk, "max_drawdown_kill_pct", 0.0))
    _killed_by_dd = False  # once True, no new trades for rest of fold

    for i, row in enumerate(test_rows.itertuples(index=False)):
        # ── close matured positions ────────────────────────────────────────
        still_pending: list[dict] = []
        for entry in pending:
            if i >= entry["close_bar_idx"]:
                pnl = entry["absolute_pnl"]
                balance += pnl
                peak_balance = max(peak_balance, balance)
                dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
                if _max_dd_kill_pct > 0 and dd >= _max_dd_kill_pct:
                    _killed_by_dd = True
                entry["meta"]["balance_after"] = round(balance, 4)
                entry["meta"]["drawdown"] = round(-dd, 4)
                entry["meta"]["is_win"] = bool(pnl > 0)
                entry["meta"]["is_loss"] = bool(pnl < 0)
                entry["meta"]["is_draw"] = bool(pnl == 0)
                trades.append(entry["meta"])
                if pnl > 0:
                    wins += 1
                    _consecutive_losses = 0  # Reset on win
                elif pnl < 0:
                    losses += 1
                    _consecutive_losses += 1
                    _daily_loss += abs(pnl)
                    _side = str(entry["meta"].get("side", "")).strip().lower()
                    if _side in {"buy", "sell"}:
                        _last_loss_marker_by_side[_side] = {
                            "bar_idx": i,
                            "close_price": float(
                                entry["meta"].get("sl_marker_price")
                                or entry["meta"].get("exit_price")
                                or entry["meta"].get("entry_price")
                                or 0.0
                            ),
                            "atr": float(entry["meta"].get("atr") or 0.0),
                        }
                    # Trigger cooldown after N consecutive losses
                    if _pause_count > 0 and _consecutive_losses >= _pause_count:
                        _cooldown_remaining = _cooldown_bars
            else:
                still_pending.append(entry)
        pending = still_pending

        # ── Cooldown tick ──────────────────────────────────────────────────
        if _cooldown_remaining > 0:
            _cooldown_remaining -= 1

        # ── Daily loss reset ───────────────────────────────────────────────
        _row_date = str(getattr(row, "time", ""))[:10]
        if _row_date != _current_date:
            _current_date = _row_date
            _daily_loss = 0.0

        if row.prediction == 0:
            balance_history.append(balance)
            continue

        # ── circuit breaker gates ──────────────────────────────────────────
        if _cooldown_remaining > 0:
            skipped_circuit_breaker += 1
            balance_history.append(balance)
            continue
        if _daily_limit > 0 and peak_balance > 0 and _daily_loss >= peak_balance * _daily_limit:
            skipped_circuit_breaker += 1
            balance_history.append(balance)
            continue
        if _killed_by_dd:
            skipped_circuit_breaker += 1
            balance_history.append(balance)
            continue

        # ── strategy filter gate ───────────────────────────────────────────
        allowed, reason = strategy.should_allow_row(row, float(row.probability))
        if not allowed:
            skipped_by_filters += 1
            balance_history.append(balance)
            continue

        # ── anti re-entry guard (same side, near last SL/loss) ─────────────
        if bool(getattr(settings.risk, "reentry_guard_enabled", False)):
            _side = str(getattr(row, "trade_side", "")).strip().lower()
            _marker = _last_loss_marker_by_side.get(_side)
            if _side in {"buy", "sell"} and _marker:
                _cool = max(int(getattr(settings.risk, "reentry_cooldown_bars_after_sl", 0)), 0)
                _bars_since = i - int(_marker.get("bar_idx", -10_000_000))
                if _cool > 0 and _bars_since < _cool:
                    skipped_reentry_guard += 1
                    balance_history.append(balance)
                    continue
                _min_dist_atr = max(float(getattr(settings.risk, "reentry_min_distance_atr", 0.0)), 0.0)
                if _min_dist_atr > 0:
                    _entry_px = float(getattr(row, "close", 0.0) or 0.0)
                    _marker_px = float(_marker.get("close_price", 0.0) or 0.0)
                    _atr_ref = float(_marker.get("atr", 0.0) or 0.0)
                    if _atr_ref <= 0:
                        _atr_ref = float(getattr(row, "atr", 0.0) or 0.0)
                    if _entry_px > 0 and _marker_px > 0 and _atr_ref > 0:
                        if abs(_entry_px - _marker_px) < _atr_ref * _min_dist_atr:
                            skipped_reentry_guard += 1
                            balance_history.append(balance)
                            continue

        # ── dynamic slot check ─────────────────────────────────────────────
        regime = int(getattr(row, "volatility_regime", 1))
        max_pos = risk_manager.get_dynamic_max_positions(balance, regime)
        if len(pending) >= max_pos:
            skipped_no_slot += 1
            balance_history.append(balance)
            continue

        # ── open new position ──────────────────────────────────────────────
        # Dynamic risk tier: scale between floor and ceiling based on drawdown from peak.
        # risk_tier_floor = 3% = minimum during drawdown; risk_per_trade = 5% = max at peak.
        rf_base  = float(settings.risk.risk_per_trade)
        rf_floor = float(getattr(settings.risk, 'risk_tier_floor', 0.0))
        if rf_floor > 0 and peak_balance > start_bal:
            dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
            if dd >= 0.10:              # >= 10% drawdown: use floor (protection mode)
                rf = rf_floor
            elif dd >= 0.03:            # 3-10% drawdown: linear interpolation
                ratio = (dd - 0.03) / 0.07
                rf = rf_base - ratio * (rf_base - rf_floor)
            else:
                rf = rf_base            # < 3% drawdown: full risk
        else:
            rf = rf_base

        # Anti-martingale: reduce risk after consecutive losses
        if _anti_mart_factor < 1.0 and _consecutive_losses > 0:
            _n_reductions = min(_consecutive_losses, _anti_mart_max)
            rf *= _anti_mart_factor ** _n_reductions
        # compound=False: always use starting balance -> linear expectancy (no explosion).
        effective_bal = balance if compound else start_bal
        # Apply compound cap: prevent unrealistic exponential growth
        if compound and compound_cap > 0 and effective_bal > max_balance:
            effective_bal = max_balance
        side = str(getattr(row, "trade_side", "")).strip().lower()
        probability = float(getattr(row, "probability", 0.0) or 0.0)

        # Regime base SL/TP, then optional model-driven dynamic tuning.
        base_sl_mult, base_tp_rr = base_sltp_by_regime(settings, regime)
        sl_mult, tp_rr, _dyn_tags, _ = compute_dynamic_sltp(
            settings,
            row,
            probability,
            base_sl_mult,
            base_tp_rr,
        )

        # Setup-aware exit override: prefer row-provided setup exits when present.
        # If missing, derive them from the row features using the configured setup_exit profile.
        if bool(getattr(settings.risk, "setup_exit_enabled", False)):
            _setup_tp = float(getattr(row, "setup_tp_rr", 0.0) or 0.0)
            _setup_sl = float(getattr(row, "setup_sl_mult", 0.0) or 0.0)
            _setup_scaled = bool(int(getattr(row, "setup_exit_scaled", 0) or 0))
            if _setup_tp <= 0.0 or _setup_sl <= 0.0:
                _setup_tp, _setup_sl, _setup_tier, _setup_tags, _setup_metrics = resolve_setup_exit_targets(
                    row,
                    enabled=True,
                    tp_scale=float(getattr(settings.risk, "setup_exit_scale", 1.0)),
                )
                _setup_scaled = True
            elif not _setup_scaled:
                _setup_tp *= float(getattr(settings.risk, "setup_exit_scale", 1.0))
            if _setup_tp > 0:
                tp_rr = _setup_tp
            if _setup_sl > 0:
                sl_mult = _setup_sl

        entry_price = float(getattr(row, "close", 0.0) or 0.0)
        atr_value = float(getattr(row, "atr", 0.0) or 0.0)
        if atr_value <= 0 and i < len(close_arr):
            _ms_atr_norm = None if ms_atr_norm_arr is None else float(ms_atr_norm_arr[i])
            if _ms_atr_norm is not None and np.isfinite(_ms_atr_norm):
                atr_value = float(close_arr[i] * _ms_atr_norm / 1000.0)

        # Optional volatility boost for SL multiplier (same as live strategy).
        if (
            atr_value > 0
            and bool(getattr(settings.risk, "sl_volatility_boost_enabled", False))
        ):
            atr_percentile = float(getattr(row, "atr_percentile", np.nan))
            trigger = max(
                0.0,
                min(1.0, float(getattr(settings.risk, "sl_volatility_boost_trigger_percentile", 0.85))),
            )
            max_multiplier = max(
                1.0, float(getattr(settings.risk, "sl_volatility_boost_max_multiplier", 1.0))
            )
            if np.isfinite(atr_percentile):
                atr_percentile = float(np.clip(atr_percentile, 0.0, 1.0))
                if atr_percentile >= trigger and max_multiplier > 1.0:
                    scale = (atr_percentile - trigger) / max(1e-6, 1.0 - trigger)
                    boost = 1.0 + scale * (max_multiplier - 1.0)
                    sl_mult *= boost

        sl_dist = max(0.0, atr_value * sl_mult)
        min_stop_points = max(0.0, float(getattr(settings.risk, "min_stop_loss_points", 0.0) or 0.0))
        if sl_dist < min_stop_points:
            sl_dist = min_stop_points
        min_stop_atr_mult = max(0.0, float(getattr(settings.risk, "min_stop_loss_atr_multiple", 0.0) or 0.0))
        if atr_value > 0 and min_stop_atr_mult > 0:
            sl_dist = max(sl_dist, atr_value * min_stop_atr_mult)

        # Realized RR:
        # 1) default: precomputed row.realized_rr (old behavior)
        # 2) dynamic_sltp_backtest_enabled: recompute SL/TP race from OHLC path.
        raw_rr = float(getattr(row, "realized_rr", 0.0) or 0.0)
        _hold = int(bars_held_arr[i]) if i < len(bars_held_arr) else int(label_horizon)
        _hold = max(1, _hold)
        if dynamic_sltp_eval_active and side in {"buy", "sell"} and atr_value > 0 and high_arr is not None and low_arr is not None:
            sl_level = entry_price - sl_dist if side == "buy" else entry_price + sl_dist
            tp_level = entry_price + sl_dist * tp_rr if side == "buy" else entry_price - sl_dist * tp_rr
            end_i = min(len(close_arr) - 1, i + _hold)
            hit_rr = 0.0
            for j in range(i + 1, end_i + 1):
                hi = float(high_arr[j])
                lo = float(low_arr[j])
                if side == "buy":
                    sl_hit = bool(lo <= sl_level)
                    tp_hit = bool(hi >= tp_level)
                else:
                    sl_hit = bool(hi >= sl_level)
                    tp_hit = bool(lo <= tp_level)
                # tie in same bar -> conservative SL first
                if sl_hit:
                    hit_rr = -1.0
                    break
                if tp_hit:
                    hit_rr = float(tp_rr)
                    break
            raw_rr = float(hit_rr)
            dynamic_rr_recomputed += 1
        elif dynamic_sltp_eval_enabled:
            dynamic_rr_fallback += 1

        # P1a: Session-aware friction
        _sess_mult = float(getattr(row, "session_spread_mult", 1.0))
        _slip_mult = float(getattr(row, "session_slippage_mult", 1.0))
        _row_friction = _spread_rr * _sess_mult + _slippage_rr * _slip_mult + _commission_rr
        net_rr = raw_rr - _row_friction
        rf, throttle_mult, throttle_reason = risk_manager.apply_risk_throttle(
            rf,
            row,
            side=side,
            probability=probability,
        )
        absolute_pnl = effective_bal * rf * net_rr
        open_count = len(pending)
        max_concurrent_seen = max(max_concurrent_seen, open_count + 1)
        if side == "sell":
            sl_marker_price = entry_price + sl_dist
        else:
            sl_marker_price = entry_price - sl_dist
        directional_return = float(getattr(row, "directional_return", getattr(row, "future_return", 0.0)) or 0.0)
        exit_price = entry_price * (1.0 + directional_return)

        meta: dict = {
            "time": row.time.isoformat() if hasattr(row.time, "isoformat") else str(row.time),
            "side": row.trade_side,
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "atr": atr_value,
            "sl_marker_price": float(sl_marker_price),
            "realized_rr": float(raw_rr),
            "net_rr": float(net_rr),
            "friction_rr": float(friction_rr),
            "probability": probability,
            "risk_fraction": float(rf),
            "pnl": round(absolute_pnl, 4),
            "balance_before": round(balance, 4),
            "balance_after": None,  # filled at close
            "drawdown": None,       # filled at close
            "is_win": None,         # filled at close (pnl > 0)
            "is_loss": None,        # filled at close (pnl < 0)
            "is_draw": None,        # filled at close (pnl == 0)
            "max_positions_allowed": max_pos,
            "open_positions_at_open": open_count,
            "volatility_regime": regime,
            "sl_atr_multiple": float(sl_mult),
            "tp_rr": float(tp_rr),
            "risk_throttle_multiplier": float(throttle_mult),
            "risk_throttle_reason": throttle_reason,
        }
        # P0: Use per-trade bars_held from SL/TP race (fallback to label_horizon)
        pending.append(
            {"close_bar_idx": i + _hold, "absolute_pnl": absolute_pnl, "meta": meta}
        )
        balance_history.append(balance)

    # ── close any remaining open positions at period end ───────────────────
    for entry in pending:
        pnl = entry["absolute_pnl"]
        balance += pnl
        peak_balance = max(peak_balance, balance)
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
        entry["meta"]["balance_after"] = round(balance, 4)
        entry["meta"]["drawdown"] = round(-dd, 4)
        entry["meta"]["is_win"] = bool(pnl > 0)
        entry["meta"]["is_loss"] = bool(pnl < 0)
        entry["meta"]["is_draw"] = bool(pnl == 0)
        trades.append(entry["meta"])
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

    # ── build report ───────────────────────────────────────────────────────
    trades_df = pd.DataFrame(trades)
    start_bal = settings.training.backtest_initial_balance

    gross_profit = gross_loss = 0.0
    avg_win = avg_loss = 0.0
    if not trades_df.empty:
        gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
        gross_loss   = float(-trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())
        avg_win  = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()) if wins > 0 else 0.0
        avg_loss = float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()) if losses > 0 else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    max_drawdown = 0.0 if trades_df.empty else float(trades_df["drawdown"].min())
    pnl_std = float(trades_df["pnl"].std()) if not trades_df.empty and len(trades_df) > 1 else 0.0
    sharpe = (
        float((trades_df["pnl"].mean() / pnl_std) * (len(trades_df) ** 0.5))
        if pnl_std > 0 else 0.0
    )

    # Position tier breakdown (how many trades were opened per balance tier)
    tier_breakdown: dict[str, int] = {}
    if not trades_df.empty and "balance_before" in trades_df.columns:
        for _, tr in trades_df.iterrows():
            b = float(tr["balance_before"])
            if b < 500:
                tier = "<$500 (3 slots)"
            elif b < 2_000:
                tier = "$500-$2k (5 slots)"
            elif b < 10_000:
                tier = "$2k-$10k (8 slots)"
            elif b < 50_000:
                tier = "$10k-$50k (10 slots)"
            else:
                tier = ">$50k (15 slots)"
            tier_breakdown[tier] = tier_breakdown.get(tier, 0) + 1

    avg_concurrent = (
        float(trades_df["open_positions_at_open"].mean() + 1)
        if not trades_df.empty and "open_positions_at_open" in trades_df.columns
        else 0.0
    )

    report = {
        "simulation_mode": "dynamic_concurrent",
        "friction_rr": round(friction_rr, 4),
        "compound_cap": compound_cap,
        "starting_balance": start_bal,
        "ending_balance": round(balance, 2),
        "net_profit": round(balance - start_bal, 2),
        "return_pct": round((balance / start_bal - 1) * 100, 2),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "draws": len(trades) - wins - losses,
        "win_rate": round(wins / max(wins + losses, 1), 4),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "best_trade": round(float(trades_df["pnl"].max()) if not trades_df.empty else 0.0, 2),
        "worst_trade": round(float(trades_df["pnl"].min()) if not trades_df.empty else 0.0, 2),
        "sharpe_like": round(sharpe, 4),
        "signals_filtered_out": skipped_by_filters,
        "signals_no_slot": skipped_no_slot,
        "signals_circuit_breaker": skipped_circuit_breaker,
        "killed_by_max_dd": _killed_by_dd,
        "signals_reentry_guard": skipped_reentry_guard,
        "dynamic_sltp_eval_enabled": dynamic_sltp_eval_enabled,
        "dynamic_sltp_eval_active": dynamic_sltp_eval_active,
        "dynamic_rr_recomputed": dynamic_rr_recomputed,
        "dynamic_rr_fallback": dynamic_rr_fallback,
        "max_concurrent_positions": max_concurrent_seen,
        "avg_concurrent_positions": round(avg_concurrent, 2),
        "position_tier_breakdown": tier_breakdown,
    }
    return SimulationResult(report=report, trades=trades_df)


def write_backtest_outputs(
    report: dict[str, object],
    trades_df: pd.DataFrame,
    test_rows: pd.DataFrame,
    report_path: Path,
    trades_path: Path,
    equity_plot_path: Path,
    trades_plot_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)
    save_backtest_plots(test_rows, trades_df, equity_plot_path, trades_plot_path)
