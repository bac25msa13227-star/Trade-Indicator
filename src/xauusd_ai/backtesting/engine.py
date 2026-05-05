from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import time as time_obj
from pathlib import Path

import pandas as pd

from xauusd_ai.backtesting.slippage import calculate_slippage_rr, get_session_multiplier
from xauusd_ai.config import Settings
from xauusd_ai.execution.risk import RiskManager
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
    _slippage_rr = float(getattr(settings.risk, "slippage_rr", 0.05))  # Static fallback
    _commission_rr = float(getattr(settings.risk, "commission_rr", 0.02))
    _use_dynamic_slippage = bool(getattr(settings.risk, "use_dynamic_slippage", False))
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
        
        # Calculate friction: spread + slippage + commission
        _sess_mult = float(getattr(row, 'session_spread_mult', 1.0))
        
        # Dynamic slippage calculation (if enabled)
        if _use_dynamic_slippage:
            # Extract market conditions from row
            _atr = float(getattr(row, 'atr', 0.5))  # Actual ATR value
            _atr_mean = float(getattr(row, 'atr_mean', 0.5))  # ATR rolling mean
            _spread_pips = float(getattr(row, 'spread_points', 0.3))  # Spread in pips
            _volume_zscore = float(getattr(row, 'tick_volume_zscore', 0.0))
            
            # Convert volume z-score to volume ratio (z-score 0 = normal, +1 = high, -1 = low)
            # volume_ratio: 1.0 = normal, >1 = high volume (less slippage), <1 = low volume (more slippage)
            _volume_ratio = max(0.3, 1.0 + (_volume_zscore * 0.3))  # Scale z-score to ratio
            
            # Get session multiplier from row time
            if hasattr(row, 'time') and hasattr(row.time, 'time'):
                _session_mult = get_session_multiplier(row.time.time())
            else:
                _session_mult = 1.0  # Fallback
            
            # Calculate slippage using dynamic model
            from xauusd_ai.backtesting.slippage import calculate_slippage_pips
            _slippage_pips = calculate_slippage_pips(
                atr=_atr,
                atr_mean=_atr_mean,
                spread_pips=_spread_pips,
                volume_ratio=_volume_ratio,
                session_multiplier=_session_mult,
            )
            
            # Convert slippage pips to RR
            _entry_price = float(row.close)
            _sl_distance = abs(_entry_price * settings.risk.risk_per_trade * 0.01)  # Estimate SL distance
            _sl_price = _entry_price - _sl_distance if row.trade_side == 1 else _entry_price + _sl_distance
            _row_slippage_rr = calculate_slippage_rr(_slippage_pips, _entry_price, _sl_price)
            
            # Total friction with dynamic slippage
            _row_friction = _spread_rr * _sess_mult + _row_slippage_rr + _commission_rr
        else:
            # Static slippage (original behavior)
            _row_friction = _spread_rr * _sess_mult + _slippage_rr + _commission_rr
        
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


# ── M1 bar-by-bar trailing SL path simulator (for LOSING trades only) ────────
def _simulate_trade_m1_trailing(
    entry_time: "pd.Timestamp",
    entry_price: float,
    direction: int,       # +1=buy, -1=sell
    atr: float,
    bars_held: int,       # M5 bars
    sl_mult: float,       # stop_loss_atr_multiple (e.g. 1.5)
    m1_df: "pd.DataFrame",
    m1_sorted_index: "pd.DatetimeIndex",
    trailing_cfg,
    friction_rr: float,
) -> "float | None":
    """
    For LOSING TRADES ONLY: simulate M1 price path bar-by-bar to determine
    whether trailing SL would have improved exit before the M5-confirmed SL hit.

    The M5 dataset determines win/loss (realized_rr). This function only
    improves the trailing SL EXIT PRICE for losing trades — it does NOT
    override whether a trade was a win or loss (no re-running SL/TP race).

    Returns improved net_rr if trailing SL triggered above original SL,
    or None if no improvement (caller keeps M5 heuristic result).
    """
    sl_distance = atr * sl_mult
    if sl_distance <= 0 or entry_time is None:
        return None

    # Trailing SL config
    if not trailing_cfg:
        return None
    trail_enabled = bool(getattr(trailing_cfg, "enabled", False))
    if not trail_enabled:
        return None
    trail_be_rr  = float(getattr(trailing_cfg, "breakeven_at_rr", 0.5))
    trail_act_rr = float(getattr(trailing_cfg, "activation_rr", 1.0))
    trail_mult   = float(getattr(trailing_cfg, "trail_atr_multiple", 1.0))

    original_sl = entry_price - direction * sl_distance
    be_price        = entry_price + direction * sl_distance * trail_be_rr
    trail_act_price = entry_price + direction * sl_distance * trail_act_rr

    # Locate M1 bars in the hold window using pre-sorted index
    end_time  = entry_time + pd.Timedelta(minutes=(bars_held + 1) * 5)
    start_pos = m1_sorted_index.searchsorted(entry_time, side="right")
    end_pos   = m1_sorted_index.searchsorted(end_time,   side="left")

    if start_pos >= end_pos:
        return None  # no M1 data for this window

    m1_highs = m1_df["high"].values
    m1_lows  = m1_df["low"].values

    current_sl     = original_sl
    best_favorable = entry_price  # track peak price seen so far

    for k in range(start_pos, end_pos):
        h = m1_highs[k]
        l = m1_lows[k]
        favorable = h if direction > 0 else l
        adverse   = l if direction > 0 else h

        # Update best favorable price seen so far
        if direction > 0:
            best_favorable = max(best_favorable, favorable)
        else:
            best_favorable = min(best_favorable, favorable)

        # Update trailing SL based on best price reached so far
        act_cond = (direction > 0 and best_favorable >= trail_act_price) or \
                   (direction < 0 and best_favorable <= trail_act_price)
        be_cond  = (direction > 0 and best_favorable >= be_price) or \
                   (direction < 0 and best_favorable <= be_price)

        if act_cond:
            # Full trailing: SL trails trail_mult×1R behind best price
            if direction > 0:
                current_sl = max(current_sl, best_favorable - trail_mult * sl_distance)
            else:
                current_sl = min(current_sl, best_favorable + trail_mult * sl_distance)
        elif be_cond:
            # Breakeven only
            if direction > 0:
                current_sl = max(current_sl, entry_price)
            else:
                current_sl = min(current_sl, entry_price)

        # SL hit check — always stop as soon as current_sl is hit
        if (direction > 0 and adverse <= current_sl) or \
           (direction < 0 and adverse >= current_sl):
            # Trade exits here
            exit_rr = (current_sl - entry_price) / sl_distance * direction
            improved = (direction > 0 and current_sl > original_sl) or \
                       (direction < 0 and current_sl < original_sl)
            return (exit_rr - friction_rr) if improved else None

    # Reached end of window without SL hit — this shouldn't happen for a losing
    # M5 trade (original SL should fire within bars_held bars), but handle anyway.
    # If current_sl was never improved beyond original, no change needed.
    improved = (direction > 0 and current_sl > original_sl) or \
               (direction < 0 and current_sl < original_sl)
    if not improved:
        return None
    exit_rr = (current_sl - entry_price) / sl_distance * direction
    return exit_rr - friction_rr


def simulate_dynamic_concurrent_backtest(
    predictions: pd.DataFrame,
    settings: Settings,
    risk_manager: RiskManager,
    label: str = "test",
    compound: bool = True,
    m1_df: "pd.DataFrame | None" = None,
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

    # Hold-duration realism: overnight swap cost + weekend gap penalty
    # swap_per_night_rr: negative value → XAUUSD buy swap ~-$0.50/0.01lot/night = ~-0.005R/night
    # weekend_gap_penalty_rr: negative → expected loss from adverse weekend gap
    _swap_per_night_rr = float(getattr(settings.risk, "swap_per_night_rr", 0.0))
    _weekend_gap_rr = float(getattr(settings.risk, "weekend_gap_penalty_rr", 0.0))
    pending: list[dict] = []
    trades: list[dict] = []
    wins = 0
    losses = 0
    skipped_by_filters = 0
    skipped_no_slot = 0
    skipped_circuit_breaker = 0  # NEW: track circuit breaker blocks

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
    _max_dd_kill = float(getattr(settings.risk, "max_drawdown_kill_pct", 0.0))

    # G10: Regime & score multipliers (matching live risk_fraction path)
    _sideway_mult = float(getattr(settings.risk, 'sideway_risk_multiplier', 1.0))
    _normal_mult  = float(getattr(settings.risk, 'normal_risk_multiplier', 1.0))
    _volatile_mult = float(getattr(settings.risk, 'strong_volatility_risk_multiplier', 1.0))

    # Volatility ATR scaling — reduce risk during macro spikes
    _vol_scaling_enabled = bool(getattr(settings.risk, 'volatility_risk_scaling_enabled', False))
    _vol_atr_lookback = int(getattr(settings.risk, 'vol_atr_lookback_bars', 96))
    _vol_spike_ratio = float(getattr(settings.risk, 'vol_atr_spike_ratio', 2.0))
    _vol_spike_mult = float(getattr(settings.risk, 'vol_atr_spike_risk_mult', 0.5))
    _vol_extreme_ratio = float(getattr(settings.risk, 'vol_atr_extreme_ratio', 3.5))
    _vol_extreme_mult = float(getattr(settings.risk, 'vol_atr_extreme_risk_mult', 0.25))
    # Rolling ATR window — filled as we iterate
    from collections import deque as _deque
    _atr_window: "_deque[float]" = _deque(maxlen=_vol_atr_lookback)

    # G11: Trailing SL config
    _trailing_cfg = getattr(settings.execution, 'trailing_sl', None)
    _trail_enabled = bool(getattr(_trailing_cfg, 'enabled', False)) if _trailing_cfg else False
    _trail_be_rr   = float(getattr(_trailing_cfg, 'breakeven_at_rr', 0.5)) if _trailing_cfg else 0.5
    _trail_act_rr  = float(getattr(_trailing_cfg, 'activation_rr', 1.0)) if _trailing_cfg else 1.0
    _trail_atr_mult = float(getattr(_trailing_cfg, 'trail_atr_multiple', 1.0)) if _trailing_cfg else 1.0

    # G12: Partial TP config
    _partial_tp_enabled = bool(getattr(settings.risk, 'partial_tp_enabled', False))
    _partial_tp_rr  = float(getattr(settings.risk, 'partial_tp_rr', 1.2))
    _partial_tp_pct = float(getattr(settings.risk, 'partial_tp_pct', 0.5))

    # M1 simulation setup — pre-extract config values and sort M1 index once
    _sl_mult = float(getattr(settings.risk, 'stop_loss_atr_multiple', 1.5))
    _entry_slip_frac = float(getattr(settings.risk, 'entry_slippage_atr_frac', 0.0))
    _m1_sorted_index: "pd.DatetimeIndex | None" = None
    if m1_df is not None and not m1_df.empty:
        # Ensure UTC-aware index for consistent comparison with predictions timestamps
        if m1_df.index.tz is None:
            _m1_idx = m1_df.index.tz_localize("UTC")
        else:
            _m1_idx = m1_df.index.tz_convert("UTC")
        _m1_df_utc = m1_df.copy()
        _m1_df_utc.index = _m1_idx
        _m1_sorted_index = _m1_df_utc.index
    else:
        _m1_df_utc = None

    # G13: Close opposite on signal
    _close_opposite = bool(getattr(settings.execution, 'close_opposite_on_signal', False))

    # G15: Market gate — skip bars during closed hours (Sat, Sun, Fri 22:00+ UTC)
    _market_gate = bool(getattr(settings.market, 'enforce_market_open_gate', False))
    skipped_market_closed = 0

    # G15b: Re-entry guard — block same-side entry for N bars after SL hit
    _reentry_enabled = bool(getattr(settings.risk, 'reentry_guard_enabled', True))
    _reentry_cooldown = int(getattr(settings.risk, 'reentry_cooldown_bars_after_sl', 1))
    _reentry_min_atr = float(getattr(settings.risk, 'reentry_min_distance_atr', 0.35))
    # Track per-side: {"buy": (bar_idx_of_last_sl, close_price_at_sl), "sell": ...}
    _reentry_last_sl: dict[str, tuple[int, float]] = {}
    skipped_reentry_guard = 0

    for i, row in enumerate(test_rows.itertuples(index=False)):
        # ── close matured positions ────────────────────────────────────────
        still_pending: list[dict] = []
        for entry in pending:
            if i >= entry["close_bar_idx"]:
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
                    _consecutive_losses = 0  # Reset on win
                elif pnl < 0:
                    losses += 1
                    _consecutive_losses += 1
                    _daily_loss += abs(pnl)
                    # G15b: Record SL hit for reentry guard
                    if _reentry_enabled:
                        _sl_side = entry["meta"].get("side", "")
                        _sl_price = float(entry["meta"].get("entry_price", 0.0))
                        _reentry_last_sl[_sl_side] = (i, _sl_price)
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
            # Still update ATR window on every bar (including non-trade bars)
            if _vol_scaling_enabled:
                _bar_atr = float(getattr(row, 'atr', 0.0))
                if _bar_atr > 0:
                    _atr_window.append(_bar_atr)
            balance_history.append(balance)
            continue

        # ── G15: Market gate — skip if market closed ──────────────────────
        if _market_gate:
            _t = getattr(row, 'time', None)
            if _t is not None and hasattr(_t, 'weekday'):
                _wd = _t.weekday()  # Mon=0 .. Sun=6
                _hr = _t.hour
                # Skip: Saturday all day, Sunday before 22:00 UTC, Friday after 22:00 UTC
                if _wd == 5 or _wd == 6 or (_wd == 4 and _hr >= 22):
                    skipped_market_closed += 1
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
        if _max_dd_kill > 0 and peak_balance > 0:
            _cur_dd = (peak_balance - balance) / peak_balance
            if _cur_dd >= _max_dd_kill:
                skipped_circuit_breaker += 1
                balance_history.append(balance)
                continue

        # ── strategy filter gate ───────────────────────────────────────────
        allowed, reason = strategy.should_allow_row(row, float(row.probability))
        if not allowed:
            skipped_by_filters += 1
            balance_history.append(balance)
            continue

        # ── G15b: Re-entry guard — block same-side if too soon after SL ───
        if _reentry_enabled:
            _cur_side = str(getattr(row, 'trade_side', ''))
            if _cur_side in _reentry_last_sl:
                _sl_bar, _sl_price = _reentry_last_sl[_cur_side]
                _bars_since_sl = i - _sl_bar
                if _bars_since_sl < _reentry_cooldown:
                    skipped_reentry_guard += 1
                    balance_history.append(balance)
                    continue
                # Also check ATR distance from SL entry price
                _cur_close = float(getattr(row, 'close', 0.0))
                _cur_atr = float(getattr(row, 'atr', 0.0))
                if _cur_atr > 0 and abs(_cur_close - _sl_price) < _reentry_min_atr * _cur_atr:
                    skipped_reentry_guard += 1
                    balance_history.append(balance)
                    continue

        # ── G13: Close opposite on signal ───────────────────────────────
        if _close_opposite:
            _cur_side = str(getattr(row, 'trade_side', ''))
            _new_pending: list[dict] = []
            for entry in pending:
                _e_side = entry["meta"].get("side", "")
                _e_pnl  = entry["absolute_pnl"]
                # Close profitable opposite-side position
                if _e_side != _cur_side and _e_pnl > 0:
                    balance += _e_pnl
                    peak_balance = max(peak_balance, balance)
                    dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
                    entry["meta"]["balance_after"] = round(balance, 4)
                    entry["meta"]["drawdown"] = round(-dd, 4)
                    entry["meta"]["is_win"] = True
                    entry["meta"]["is_loss"] = False
                    entry["meta"]["is_draw"] = False
                    trades.append(entry["meta"])
                    wins += 1
                    _consecutive_losses = 0
                else:
                    _new_pending.append(entry)
            pending = _new_pending

        # ── dynamic slot check ─────────────────────────────────────────────
        regime = int(getattr(row, "volatility_regime", 1))
        max_pos = risk_manager.get_dynamic_max_positions(balance, regime)
        if len(pending) >= max_pos:
            skipped_no_slot += 1
            balance_history.append(balance)
            continue

        # ── open new position ──────────────────────────────────────────────
        # Dynamic risk tier: scale between floor and ceiling based on drawdown from peak.
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

        # G10: Apply regime multiplier (matching live risk_fraction path)
        if regime == 0:
            rf *= _sideway_mult
        elif regime == 2:
            rf *= _volatile_mult
        else:
            rf *= _normal_mult

        # G10: Apply score multiplier
        _strat_score = float(getattr(row, 'strategy_score', 0.5))
        _abs_score = abs(_strat_score)
        if bool(getattr(settings.risk, 'score_multiplier_enabled', True)):
            if _abs_score >= 0.5:
                _score_mult = 1.0
            else:
                _score_mult = 0.7 + 0.6 * _abs_score
        else:
            _score_mult = 1.0  # score_multiplier_enabled=False → always full size
        rf *= _score_mult

        # Anti-martingale: reduce risk after consecutive losses
        if _anti_mart_factor < 1.0 and _consecutive_losses > 0:
            _n_reductions = min(_consecutive_losses, _anti_mart_max)
            rf *= _anti_mart_factor ** _n_reductions

        # Volatility ATR spike scaling — reduce size during macro events
        _cur_atr = float(getattr(row, 'atr', 0.0))
        if _vol_scaling_enabled and _cur_atr > 0:
            _atr_window.append(_cur_atr)
            if len(_atr_window) >= 20:  # need at least 20 bars to form a baseline
                _mean_atr = sum(_atr_window) / len(_atr_window)
                if _mean_atr > 0:
                    _atr_ratio = _cur_atr / _mean_atr
                    if _atr_ratio >= _vol_extreme_ratio:
                        rf *= _vol_extreme_mult   # extreme spike → 25% size
                    elif _atr_ratio >= _vol_spike_ratio:
                        rf *= _vol_spike_mult      # moderate spike → 50% size
        # compound=False: always use starting balance -> linear expectancy (no explosion).
        effective_bal = balance if compound else start_bal
        # Apply compound cap: prevent unrealistic exponential growth
        if compound and compound_cap > 0 and effective_bal > max_balance:
            effective_bal = max_balance
        # P1a: Session-aware friction
        _sess_mult = float(getattr(row, 'session_spread_mult', 1.0))
        _row_friction = _spread_rr * _sess_mult + _slippage_rr + _commission_rr
        raw_rr = float(row.realized_rr)
        net_rr = raw_rr - _row_friction

        # G12+G11: M1 bar-by-bar simulation (when M1 data available) OR
        #          fall back to peak_rr heuristic for trailing SL + partial TP.
        if _m1_df_utc is not None and _m1_sorted_index is not None:
            _direction = 1 if str(getattr(row, 'trade_side', 'buy')) == 'buy' else -1
            _entry_atr = float(getattr(row, 'atr', 0.0))
            _entry_time = getattr(row, 'time', None)
            # Normalize to UTC for M1 index lookup
            if _entry_time is not None and hasattr(_entry_time, 'tz_localize'):
                _entry_time_utc = _entry_time.tz_localize("UTC") if _entry_time.tzinfo is None else _entry_time.tz_convert("UTC")
            else:
                _entry_time_utc = _entry_time
            _entry_price_slipped = float(row.close) + _direction * _entry_atr * _entry_slip_frac
            _bars = int(getattr(row, 'bars_held', label_horizon))

            # G12: Partial TP for winning trades — same heuristic (M5 is sufficient)
            if _partial_tp_enabled and raw_rr > 0 and raw_rr >= _partial_tp_rr:
                _pt_rr  = _partial_tp_rr - _row_friction
                _full_rr = net_rr
                net_rr = _partial_tp_pct * _pt_rr + (1.0 - _partial_tp_pct) * _full_rr

            # G11: For LOSING trades — use M1 path to find accurate trailing SL exit
            if raw_rr < 0 and _trail_enabled:
                _m1_net_rr = _simulate_trade_m1_trailing(
                    entry_time=_entry_time_utc,
                    entry_price=_entry_price_slipped,
                    direction=_direction,
                    atr=_entry_atr,
                    bars_held=_bars,
                    sl_mult=_sl_mult,
                    m1_df=_m1_df_utc,
                    m1_sorted_index=_m1_sorted_index,
                    trailing_cfg=_trailing_cfg,
                    friction_rr=_row_friction,
                )
                if _m1_net_rr is not None:
                    net_rr = _m1_net_rr
                else:
                    # M1 data unavailable for window — fall back to peak_rr heuristic
                    _peak_rr = float(getattr(row, 'peak_rr', 0.0))
                    if _peak_rr >= _trail_act_rr:
                        net_rr = max(_trail_act_rr - _trail_atr_mult, 0.0) - _row_friction
                    elif _peak_rr >= _trail_be_rr:
                        net_rr = -_row_friction
        else:
            # G12: Partial TP — if realized_rr >= threshold, simulate closing
            #      partial_tp_pct at partial_tp_rr and the rest at full TP.
            if _partial_tp_enabled and raw_rr > 0 and raw_rr >= _partial_tp_rr:
                # Part 1: closed at partial_tp_rr (e.g. 50% at 1.2R)
                _pt_rr = _partial_tp_rr - _row_friction
                # Part 2: remaining runs to full TP (e.g. 50% at 3.5R)
                _full_rr = net_rr
                net_rr = _partial_tp_pct * _pt_rr + (1.0 - _partial_tp_pct) * _full_rr

            # G11: Trailing SL — adjust net_rr for trades that would have been
            #      saved by breakeven or trailing stop.
            if _trail_enabled and raw_rr < 0:
                _peak_rr = float(getattr(row, 'peak_rr', 0.0))
                if _peak_rr >= _trail_act_rr:
                    # Price reached trailing activation → trail would hold at
                    # activation_rr - trail_atr_mult * 1R, minimum = breakeven
                    _trailed_rr = max(_trail_act_rr - _trail_atr_mult, 0.0)
                    net_rr = _trailed_rr - _row_friction
                elif _peak_rr >= _trail_be_rr:
                    # Price reached breakeven but not trailing activation → BE stop
                    net_rr = -_row_friction  # breakeven minus friction

        rf, throttle_mult, throttle_reason = risk_manager.apply_risk_throttle(
            rf,
            row,
            side=str(getattr(row, "trade_side", "")),
            probability=float(row.probability),
        )
        # ── Realistic lot-snapping (min_lot=0.01, XAUUSD $100/lot/pip) ──────
        # On small accounts ($200), ideal lot is e.g. 0.005 but MT5 requires 0.01.
        # Snap up to 0.01 and recompute actual risk fraction so PnL matches live.
        _atr_lot = float(getattr(row, 'atr', 0.0))
        if regime == 0:   # sideway
            _sl_mult_lot = float(getattr(settings.risk, 'sideway_sl_atr_multiple',
                                         settings.risk.stop_loss_atr_multiple))
        elif regime == 2:  # volatile
            _sl_mult_lot = float(getattr(settings.risk, 'volatile_sl_atr_multiple',
                                         settings.risk.stop_loss_atr_multiple))
        else:
            _sl_mult_lot = float(settings.risk.stop_loss_atr_multiple)
        _sl_dist = _atr_lot * _sl_mult_lot
        _XAUUSD_OZ = 100.0
        _MIN_LOT_SNAP = 0.01
        _MAX_LOT_CAP = float(getattr(settings.risk, 'max_lot', 0.0))
        if _sl_dist > 0 and effective_bal > 0:
            _ideal_lot = (effective_bal * rf) / (_XAUUSD_OZ * _sl_dist)
            _actual_lot = max(_MIN_LOT_SNAP, round(_ideal_lot / _MIN_LOT_SNAP) * _MIN_LOT_SNAP)
            if _MAX_LOT_CAP > 0:
                _actual_lot = min(_actual_lot, _MAX_LOT_CAP)
            rf = (_actual_lot * _XAUUSD_OZ * _sl_dist) / effective_bal
        # ── Hold-duration costs: overnight swap + weekend gap ────────────────
        # Applied BEFORE absolute_pnl so dollar amounts are accurate.
        _hold = int(getattr(row, 'bars_held', label_horizon))
        _entry_time_obj = getattr(row, 'time', None)
        _side_str = str(getattr(row, 'trade_side', ''))
        _swap_rr = 0.0
        _gap_rr = 0.0
        if _swap_per_night_rr != 0.0 and _hold > 0:
            _nights = _hold * 5.0 / (60.0 * 24.0)  # M5 bars → nights
            if _side_str == 'buy':
                _swap_rr = _swap_per_night_rr * _nights         # negative → cost
            else:
                _swap_rr = -abs(_swap_per_night_rr) * 0.3 * _nights  # sell: smaller
        if _weekend_gap_rr != 0.0 and _entry_time_obj is not None and hasattr(_entry_time_obj, 'weekday'):
            _wd = _entry_time_obj.weekday()  # Mon=0 .. Sun=6
            _hr = _entry_time_obj.hour
            # Friday trade that runs into the weekend (Fri 22:00 UTC close)
            if _wd == 4 and _hr < 22:
                _mins_to_close = (22 - _hr) * 60
                if _hold * 5 > _mins_to_close:
                    _gap_rr = _weekend_gap_rr  # negative → penalty
        net_rr += _swap_rr + _gap_rr
        absolute_pnl = effective_bal * rf * net_rr
        open_count = len(pending)
        max_concurrent_seen = max(max_concurrent_seen, open_count + 1)

        meta: dict = {
            "time": row.time.isoformat() if hasattr(row.time, "isoformat") else str(row.time),
            "side": row.trade_side,
            "entry_price": float(row.close),
            "realized_rr": float(row.realized_rr),
            "net_rr": float(net_rr),
            "friction_rr": float(friction_rr),
            "swap_rr": round(_swap_rr, 6),
            "gap_rr": round(_gap_rr, 6),
            "bars_held": _hold,
            "probability": float(row.probability),
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
        "signals_reentry_guard": skipped_reentry_guard,
        "signals_market_closed": skipped_market_closed,
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
