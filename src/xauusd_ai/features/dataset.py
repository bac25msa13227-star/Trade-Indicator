from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.data.news_features import attach_news_features
from xauusd_ai.features.indicators import (
    atr, macd, rsi, zscore,
    bos_choch, fair_value_gap, order_block, kill_zone, judas_swing,
    displacement, equal_highs_lows, vsa_signal, wyckoff_spring_upthrust,
    market_structure_bias, premium_discount_zone,
    bollinger_bands, stochastic, rsi_slope,
    adx, obv_slope, candle_body_ratio, price_roc,
    pullback_depth, atr_expansion, wick_rejection,
    volume_surge, close_position_in_range, macd_hist_acceleration,
    volume_delta_momentum, institutional_candle_score, swing_failure_pattern,
)


# Layout: D1(1) | H4 ICT(9) | H1 Wyckoff(3) | M15 execution(15) | News(5) | Price Structure(5) | Advanced(8)
# Total: 46 features — multi-timeframe ICT + Wyckoff + News + Price Structure + v2 Advanced
FEATURE_COLUMNS = [
    # --- D1 context (1) ---
    "daily_bias",
    # --- H4 ICT structure (9) ---
    "h4_bos",
    "h4_choch",
    "h4_fvg",
    "h4_order_block",
    "h4_displacement",
    "h4_ehl",
    "h4_market_structure_bias",
    "h4_ict_confluence",       # net agreement of H4 signals (-4 to +4)
    "h4_premium_discount",    # ICT PDZ: +1=discount(buy), -1=premium(sell)
    # --- H1 Wyckoff context (3) ---
    "hourly_bias",
    "vsa_signal",
    "wyckoff_spring_signal",
    # --- M15 execution (15) ---
    "trend_alignment",
    "rsi",
    "macd_hist",
    "atr_ratio",
    "range_efficiency",
    "liquidity_sweep",
    "order_flow_proxy",
    "wyckoff_phase",
    "volatility_regime",
    "session_return",
    "tick_volume_zscore",
    "spread_points",
    "strategy_score",
    "kill_zone_flag",
    "judas_swing_signal",
    # --- News awareness (5) ---
    "news_impact_ahead",    # 0=none, 1=medium, 2=high in next 4h
    "news_hours_ahead",     # hours to next high-impact news (0–48)
    "news_hours_since",     # hours since last high-impact news (0–48)
    "news_surprise_gold",   # +1 bullish gold / -1 bearish / 0 neutral
    "news_is_blackout",     # 1 = within ±1h of High-impact news
    # --- Price structure & momentum (5) ---
    "bb_position",          # Bollinger Band position relative to bands (-1..+1)
    "stoch_k",              # Stochastic K normalized: -50..+50 (50=neutral)
    "stoch_kd_diff",        # Stochastic K minus D: crossover momentum
    "rsi_slope",            # RSI slope over 5 bars (divergence proxy)
    "price_vs_h4_sma",      # Price distance from H4 SMA50 (trend alignment)
    # --- v2: Advanced features for model improvement (8) ---
    "adx",                  # Average Directional Index — trend strength 0-100
    "candle_body_ratio",    # Body/range ratio — conviction measure 0-1
    "obv_slope",            # OBV slope zscore — volume-confirmed momentum
    "price_roc",            # Rate of change 12-bar — price momentum
    "multi_tf_consensus",   # D1+H4+H1 direction consensus (-3..+3)
    "h4_h1_bias_agree",     # H4 and H1 same direction binary 0/1
    "atr_percentile",       # ATR rank in 100-bar rolling window 0-1
    "time_hour_sin",        # sin(2π × hour/24) — cyclical time of day
    "time_hour_cos",        # cos(2π × hour/24) — cyclical time of day
    # --- v3: Microstructure & momentum quality (6) ---
    "pullback_depth",       # Fibonacci retracement depth 0-1.5 (entry timing)
    "atr_expansion",        # ATR(7)/ATR(28) volatility trend >1=expanding
    "wick_rejection",       # Net wick pressure: +ve=buying, -ve=selling
    "volume_surge",         # Volume / 20-bar avg — confirms moves
    "close_in_range",       # Close position in bar range 0=low 1=high
    "macd_accel",           # MACD histogram acceleration — turning points
    # --- v4: Institutional Order Flow & Smart Money (3) ---
    "vol_delta_momentum",   # Bookmap-inspired volume delta momentum
    "inst_candle_score",    # Institutional candle detection score -1..+1
    "swing_failure",        # ICT Swing Failure Pattern +1=bullish -1=bearish
    # --- v5: Meta-features for cleaner execution timing (3) ---
    "trend_strength_score", # Trend quality from consensus + ADX + H4 premium/discount
    "pullback_quality",     # Pullback entry quality around 50% retracement + rejection
    "execution_quality",    # Breakout/continuation quality from candle structure + flow
]


def _build_date_mask(series: pd.Series, start: str | None, end: str | None) -> pd.Series:
    mask = pd.Series(True, index=series.index)
    series_tz = series.dt.tz  # timezone of the data (UTC or None)
    if start:
        ts = pd.Timestamp(start, tz="UTC") if series_tz is not None else pd.Timestamp(start)
        mask &= series >= ts
    if end:
        ts = pd.Timestamp(end, tz="UTC") if series_tz is not None else pd.Timestamp(end)
        mask &= series <= ts
    return mask


def _enrich_execution_frame(settings: Settings, frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["returns"] = enriched["close"].pct_change().fillna(0)
    enriched["rsi"] = rsi(enriched["close"], settings.strategy.rsi_period).fillna(50)
    _, _, macd_hist = macd(
        enriched["close"],
        settings.strategy.macd_fast,
        settings.strategy.macd_slow,
        settings.strategy.macd_signal,
    )
    enriched["macd_hist"] = macd_hist.fillna(0)
    enriched["atr"] = atr(enriched).bfill().fillna(0)
    enriched["atr_ratio"] = (enriched["atr"] / enriched["close"]).fillna(0)
    enriched["range_efficiency"] = (
        (enriched["close"] - enriched["open"]).abs() / (enriched["high"] - enriched["low"]).replace(0, np.nan)
    ).fillna(0)
    enriched["tick_volume_zscore"] = zscore(enriched["tick_volume"], 20).fillna(0)
    enriched["session_return"] = enriched["close"].pct_change(12).fillna(0)
    # NEW: Bollinger Band position
    bb_upper, bb_mid, bb_lower = bollinger_bands(enriched["close"], 20)
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    enriched["bb_position"] = ((enriched["close"] - bb_mid) / (bb_range / 2)).fillna(0).clip(-2.0, 2.0)
    # NEW: Stochastic K and K-D crossover
    stoch_k_raw, stoch_d_raw = stochastic(enriched, 14, 3)
    enriched["stoch_k"] = (stoch_k_raw - 50.0).fillna(0)
    enriched["stoch_kd_diff"] = (stoch_k_raw - stoch_d_raw).fillna(0)
    # NEW: RSI slope (momentum divergence proxy)
    enriched["rsi_slope"] = rsi_slope(enriched["close"], settings.strategy.rsi_period, 5)
    # v2: Advanced features
    enriched["adx"] = adx(enriched, period=14)
    enriched["candle_body_ratio"] = candle_body_ratio(enriched)
    enriched["obv_slope"] = obv_slope(enriched, slope_period=10)
    enriched["price_roc"] = price_roc(enriched["close"], period=12)
    # Cyclical time-of-day encoding (gold has strong session patterns)
    hour = enriched["time"].dt.hour + enriched["time"].dt.minute / 60.0
    enriched["time_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    enriched["time_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    # ATR percentile: current volatility rank in rolling window (vectorized using pandas)
    _atr_s = enriched["atr"]
    _window = 100
    enriched["atr_percentile"] = _atr_s.rolling(_window, min_periods=20).rank(pct=True).fillna(0.5)
    # v3: Microstructure & momentum quality features
    enriched["pullback_depth"] = pullback_depth(enriched, trend_lookback=50)
    enriched["atr_expansion"] = atr_expansion(enriched, fast=7, slow=28)
    enriched["wick_rejection"] = wick_rejection(enriched)
    enriched["volume_surge"] = volume_surge(enriched, lookback=20)
    enriched["close_in_range"] = close_position_in_range(enriched)
    enriched["macd_accel"] = macd_hist_acceleration(enriched["close"])
    # v4: Institutional Order Flow & Smart Money features
    enriched["vol_delta_momentum"] = volume_delta_momentum(enriched, fast=5, slow=20)
    enriched["inst_candle_score"] = institutional_candle_score(enriched, lookback=14)
    enriched["swing_failure"] = swing_failure_pattern(enriched, lookback=20)
    return enriched


def _frame_bias(frame: pd.DataFrame, fast_window: int, slow_window: int) -> pd.Series:
    fast = frame["close"].rolling(fast_window).mean()
    slow = frame["close"].rolling(slow_window).mean()
    return np.sign((fast - slow).fillna(0))


def _merge_context(settings: Settings, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    execution = _enrich_execution_frame(settings, frames[settings.market.execution_timeframe])
    daily = frames[settings.market.higher_timeframe][["time", "close"]].copy()
    hourly = frames[settings.market.mid_timeframe][["time", "close"]].copy()

    daily["daily_bias"] = _frame_bias(daily, 5, 20)
    hourly["hourly_bias"] = _frame_bias(hourly, 10, 50)

    merged = pd.merge_asof(execution.sort_values("time"), daily[["time", "daily_bias"]].sort_values("time"), on="time")
    merged = pd.merge_asof(merged.sort_values("time"), hourly[["time", "hourly_bias"]].sort_values("time"), on="time")
    merged[["daily_bias", "hourly_bias"]] = merged[["daily_bias", "hourly_bias"]].fillna(0)
    merged["trend_alignment"] = (merged["daily_bias"] == merged["hourly_bias"]).astype(int)
    merged["liquidity_sweep"] = (
        (merged["high"] > merged["high"].rolling(settings.strategy.liquidity_lookback).max().shift(1)) |
        (merged["low"] < merged["low"].rolling(settings.strategy.liquidity_lookback).min().shift(1))
    ).astype(int)
    merged["order_flow_proxy"] = (merged["volume_imbalance"] * merged["tick_volume_zscore"]).fillna(0)
    merged["wyckoff_phase"] = np.select(
        [
            (merged["session_return"] < -0.01) & (merged["tick_volume_zscore"] > 1.0),
            (merged["session_return"] > 0.01) & (merged["tick_volume_zscore"] > 1.0),
        ],
        [1, -1],
        default=0,
    )
    merged["volatility_regime"] = np.select(
        [
            merged["atr_ratio"] <= settings.strategy.sideways_volatility_threshold,
            merged["atr_ratio"] >= settings.strategy.strong_volatility_threshold,
        ],
        [0, 2],
        default=1,
    )

    # ── H4 ICT structure features (true multi-TF: structure from H4) ──
    h4_frame = frames[settings.market.structure_timeframe].copy()
    h4_bos_s, h4_choch_s = bos_choch(h4_frame, swing_lookback=settings.strategy.swing_lookback)
    h4_frame["h4_bos"]                   = h4_bos_s
    h4_frame["h4_choch"]                 = h4_choch_s
    h4_frame["h4_fvg"]                   = fair_value_gap(h4_frame)
    h4_frame["h4_order_block"]           = order_block(h4_frame)
    h4_frame["h4_displacement"]          = displacement(h4_frame)
    h4_frame["h4_ehl"]                   = equal_highs_lows(h4_frame)
    h4_frame["h4_market_structure_bias"] = market_structure_bias(h4_frame["h4_bos"])
    # H4 ICT confluence: net directional agreement of 4 H4 signals (-4 to +4)
    _h4_long  = ((h4_frame["h4_bos"] > 0).astype(int) +
                 (h4_frame["h4_fvg"] > 0).astype(int) +
                 (h4_frame["h4_order_block"] > 0).astype(int) +
                 (h4_frame["h4_displacement"] > 0).astype(int))
    _h4_short = ((h4_frame["h4_bos"] < 0).astype(int) +
                 (h4_frame["h4_fvg"] < 0).astype(int) +
                 (h4_frame["h4_order_block"] < 0).astype(int) +
                 (h4_frame["h4_displacement"] < 0).astype(int))
    h4_frame["h4_ict_confluence"]   = _h4_long - _h4_short
    h4_frame["h4_premium_discount"] = premium_discount_zone(h4_frame, lookback=50)
    h4_frame["h4_sma50"] = h4_frame["close"].rolling(50).mean()
    _h4_cols = [
        "time", "h4_bos", "h4_choch", "h4_fvg", "h4_order_block",
        "h4_displacement", "h4_ehl", "h4_market_structure_bias",
        "h4_ict_confluence", "h4_premium_discount", "h4_sma50",
    ]
    merged = pd.merge_asof(
        merged.sort_values("time"),
        h4_frame[_h4_cols].sort_values("time"),
        on="time",
    )
    # Compute price_vs_h4_sma before fillna(0) to avoid corrupting SMA price values
    merged["price_vs_h4_sma"] = (
        (merged["close"] - merged["h4_sma50"]) / merged["close"].replace(0, np.nan)
    ).fillna(0).clip(-0.05, 0.05)
    merged.drop(columns=["h4_sma50"], inplace=True)
    _h4_signal_cols = [c for c in _h4_cols if c not in ("time", "h4_sma50")]
    merged[_h4_signal_cols] = merged[_h4_signal_cols].fillna(0)

    # ── H1 Wyckoff context features (swing patterns on H1) ────────────
    h1_frame = frames[settings.market.mid_timeframe].copy()
    h1_frame["vsa_signal"]            = vsa_signal(h1_frame)
    h1_frame["wyckoff_spring_signal"] = wyckoff_spring_upthrust(h1_frame)
    _h1_cols = ["time", "vsa_signal", "wyckoff_spring_signal"]
    merged = pd.merge_asof(
        merged.sort_values("time"),
        h1_frame[_h1_cols].sort_values("time"),
        on="time",
    )
    merged[["vsa_signal", "wyckoff_spring_signal"]] = (
        merged[["vsa_signal", "wyckoff_spring_signal"]].fillna(0)
    )

    # ── M15 time-based ICT entry features ─────────────────────────────
    merged["kill_zone_flag"]     = kill_zone(merged)
    merged["judas_swing_signal"] = judas_swing(merged)

    # ── News awareness features (5 features, rule-based + ForexFactory) ──
    merged = attach_news_features(
        merged,
        start_date=merged["time"].min() - pd.Timedelta(days=1),
        end_date=merged["time"].max()   + pd.Timedelta(days=1),
    )

    # ── v2: Multi-TF consensus features ────────────────────────────────
    # H4 direction from market_structure_bias; H1 from hourly_bias; D1 from daily_bias
    _h4_dir = np.sign(merged["h4_market_structure_bias"])
    _h1_dir = merged["hourly_bias"]
    _d1_dir = merged["daily_bias"]
    merged["multi_tf_consensus"] = (_d1_dir + _h4_dir + _h1_dir).fillna(0)  # -3 to +3
    merged["h4_h1_bias_agree"] = ((_h4_dir == _h1_dir) & (_h4_dir != 0)).astype(int)

    # Meta-features that summarize whether the current bar is worth trading.
    direction_hint = np.sign(
        merged["multi_tf_consensus"]
        + np.sign(merged["macd_hist"])
        + np.sign(merged["rsi"] - 50.0)
        + np.sign(merged["h4_market_structure_bias"])
    )
    direction_hint = direction_hint.where(
        direction_hint != 0,
        np.sign(merged["hourly_bias"] + merged["daily_bias"]),
    ).fillna(0)
    direction_hint = direction_hint.replace(0, 1)

    trend_component = np.tanh((merged["adx"] - 18.0) / 10.0).clip(-1.0, 1.0)
    consensus_component = (merged["multi_tf_consensus"] / 3.0).clip(-1.0, 1.0)
    pd_component = (-merged["h4_premium_discount"] * direction_hint).clip(-1.0, 1.0)
    merged["trend_strength_score"] = (
        0.55 * consensus_component
        + 0.35 * trend_component * direction_hint
        + 0.10 * pd_component
    ).clip(-1.0, 1.0)

    ideal_pullback = 1.0 - (merged["pullback_depth"] - 0.5).abs().clip(0, 0.75) / 0.75
    rejection_component = (merged["wick_rejection"] * direction_hint).clip(-1.0, 1.0)
    structure_component = (merged["swing_failure"] * direction_hint).clip(-1.0, 1.0)
    merged["pullback_quality"] = (
        ideal_pullback * 0.55
        + rejection_component * 0.30
        + structure_component * 0.15
    ).clip(-1.0, 1.0)

    close_pressure = ((merged["close_in_range"] - 0.5) * 2.0 * direction_hint).clip(-1.0, 1.0)
    flow_component = np.tanh(merged["vol_delta_momentum"] / 1.5).clip(-1.0, 1.0) * direction_hint
    inst_component = merged["inst_candle_score"].clip(-1.0, 1.0) * direction_hint
    volume_component = ((merged["volume_surge"] - 1.0) / 1.5).clip(-1.0, 1.0)
    merged["execution_quality"] = (
        0.30 * close_pressure
        + 0.25 * flow_component
        + 0.25 * inst_component
        + 0.20 * volume_component
    ).clip(-1.0, 1.0)

    return merged


def _expected_direction_from_context(dataset: pd.DataFrame) -> pd.Series:
    """Infer trade direction from current market state without using future data.

    The previous EMA50/EMA200-only rule was too coarse for XAUUSD intraday trading.
    This version blends higher-timeframe consensus with execution momentum so the
    model sees labels and trade_side closer to the setups we actually want to trade.
    """
    ema_fast = dataset["close"].ewm(span=50, adjust=False).mean()
    ema_slow = dataset["close"].ewm(span=200, adjust=False).mean()
    ema_bias = np.sign(ema_fast - ema_slow).replace(0, 1)

    context_votes = (
        dataset["daily_bias"]
        + dataset["hourly_bias"]
        + np.sign(dataset["h4_market_structure_bias"])
        + np.sign(dataset["h4_ict_confluence"])
    )
    execution_votes = (
        np.sign(dataset["strategy_score"])
        + np.sign(dataset["macd_hist"])
        + np.sign(dataset["rsi"] - 50.0)
        + np.sign(dataset["rsi_slope"])
        + np.sign(dataset["execution_quality"])
    )

    raw_bias = 0.65 * context_votes + 0.35 * execution_votes
    expected = np.sign(raw_bias)
    expected = expected.where(expected != 0, ema_bias).fillna(ema_bias).astype(int)
    return expected.replace(0, 1)


def _build_sltp_label(dataset: pd.DataFrame, settings: "Settings") -> pd.Series:
    """
    SL/TP Race label: for each row, simulate forward price action and check
    whether TP gets hit before SL within sltp_label_max_horizon bars.
    Returns: 1 if TP hit first, 0 if SL hit first or timeout.
    Produces cleaner binary labels than n-bar directional return.
    """
    max_horizon = int(getattr(settings.training, "sltp_label_max_horizon", 32))
    # label_tp_rr: if set (>0), use a lower RR for labeling to improve label quality
    _label_rr = float(getattr(settings.training, "label_tp_rr", 0.0))
    tp_rr = _label_rr if _label_rr > 0 else float(settings.risk.take_profit_rr)
    sl_mult = float(settings.risk.stop_loss_atr_multiple)

    closes = dataset["close"].values.astype(float)
    highs = dataset["high"].values.astype(float)
    lows = dataset["low"].values.astype(float)
    atrs = dataset["atr"].values.astype(float)
    directions = dataset["expected_direction"].values.astype(float)

    tp_levels = closes + directions * atrs * tp_rr
    sl_levels = closes - directions * atrs * sl_mult

    n = len(dataset)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n - max_horizon):
        atr_val = atrs[i]
        if not np.isfinite(atr_val) or atr_val <= 0:
            continue
        direction = directions[i]
        tp = tp_levels[i]
        sl = sl_levels[i]
        future_h = highs[i + 1 : i + max_horizon + 1]
        future_l = lows[i + 1 : i + max_horizon + 1]
        if direction > 0:  # BUY: TP when high >= tp, SL when low <= sl
            tp_idx = np.where(future_h >= tp)[0]
            sl_idx = np.where(future_l <= sl)[0]
        else:              # SELL: TP when low <= tp, SL when high >= sl
            tp_idx = np.where(future_l <= tp)[0]
            sl_idx = np.where(future_h >= sl)[0]
        tp_first = tp_idx[0] if len(tp_idx) > 0 else max_horizon + 1
        sl_first = sl_idx[0] if len(sl_idx) > 0 else max_horizon + 1
        if tp_first < sl_first:
            labels[i] = 1

    return pd.Series(labels, index=dataset.index)


def _compute_sltp_realized_rr(dataset: pd.DataFrame, settings: "Settings") -> tuple[pd.Series, pd.Series]:
    """
    SL/TP Race — compute realistic realized RR and holding bars for each row.
    Uses identical TP/SL levels as _build_sltp_label().

    Returns: (realized_rr, bars_held)
      - realized_rr: +actual_rr if TP hit first, -1.0 if SL hit first,
                     or partial RR at timeout
      - bars_held:   number of bars until exit (1-based, max=max_horizon)
    """
    max_horizon = int(getattr(settings.training, "sltp_label_max_horizon", 32))
    _label_rr = float(getattr(settings.training, "label_tp_rr", 0.0))
    tp_rr_mult = _label_rr if _label_rr > 0 else float(settings.risk.take_profit_rr)
    sl_mult = float(settings.risk.stop_loss_atr_multiple)

    closes = dataset["close"].values.astype(float)
    highs = dataset["high"].values.astype(float)
    lows = dataset["low"].values.astype(float)
    atrs = dataset["atr"].values.astype(float)
    directions = dataset["expected_direction"].values.astype(float)

    # Same levels as _build_sltp_label
    tp_levels = closes + directions * atrs * tp_rr_mult
    sl_levels = closes - directions * atrs * sl_mult

    # Actual RR when TP is hit (in R-multiples where 1R = SL distance)
    actual_tp_rr = tp_rr_mult / sl_mult if sl_mult > 0 else tp_rr_mult

    n = len(dataset)
    rr_out = np.zeros(n, dtype=np.float64)
    bars_out = np.full(n, max_horizon, dtype=np.int32)

    for i in range(n - 1):
        atr_val = atrs[i]
        if not np.isfinite(atr_val) or atr_val <= 0:
            continue
        direction = directions[i]
        if direction == 0:
            continue
        tp = tp_levels[i]
        sl = sl_levels[i]
        sl_distance = atr_val * sl_mult

        end = min(i + max_horizon + 1, n)
        future_h = highs[i + 1 : end]
        future_l = lows[i + 1 : end]

        if direction > 0:  # BUY
            tp_idx = np.where(future_h >= tp)[0]
            sl_idx = np.where(future_l <= sl)[0]
        else:              # SELL
            tp_idx = np.where(future_l <= tp)[0]
            sl_idx = np.where(future_h >= sl)[0]

        tp_first = tp_idx[0] if len(tp_idx) > 0 else max_horizon + 1
        sl_first = sl_idx[0] if len(sl_idx) > 0 else max_horizon + 1

        if tp_first < sl_first:
            rr_out[i] = actual_tp_rr
            bars_out[i] = int(tp_first + 1)
        elif sl_first <= max_horizon:
            rr_out[i] = -1.0
            bars_out[i] = int(sl_first + 1)
        else:
            # Timeout: partial RR from close at horizon
            last_idx = min(i + max_horizon, n - 1)
            price_change = (closes[last_idx] - closes[i]) * direction
            rr_out[i] = float(np.clip(price_change / sl_distance, -1.0, actual_tp_rr)) if sl_distance > 0 else 0.0
            bars_out[i] = min(max_horizon, n - 1 - i)

    return pd.Series(rr_out, index=dataset.index), pd.Series(bars_out, index=dataset.index)


def _session_spread_multiplier(hours: pd.Series) -> pd.Series:
    """Session-aware spread multiplier for XAUUSD.
    Asian (0-6 UTC): wider spread → 1.5x
    London (7-14 UTC): tightest spread → 1.0x
    NY afternoon (15-21 UTC): moderate → 1.2x
    Rollover (22-23 UTC): widest → 2.0x
    """
    mult = np.ones(len(hours), dtype=np.float64)
    h = hours.values
    mult[(h >= 0) & (h <= 6)] = 1.5
    mult[(h >= 15) & (h <= 21)] = 1.2
    mult[(h >= 22) & (h <= 23)] = 2.0
    return pd.Series(mult, index=hours.index)


def build_merged_context(settings: Settings, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the expensive merged context frame once and cache for reuse.
    The result is independent of strategy weights and label settings."""
    return _merge_context(settings, frames)


def prepare_training_dataset(settings: Settings, frames: dict[str, pd.DataFrame], strategy,
                              cached_merged: pd.DataFrame | None = None) -> pd.DataFrame:
    merged = cached_merged if cached_merged is not None else _merge_context(settings, frames)
    strategy_output = strategy.annotate_dataset(merged)
    dataset = merged.join(strategy_output)
    future_close = dataset["close"].shift(-settings.training.label_horizon)
    future_return = (future_close - dataset["close"]) / dataset["close"]

    dataset["expected_direction"] = _expected_direction_from_context(dataset)

    directional_return = future_return * dataset["expected_direction"]
    stop_loss_return = (dataset["atr_ratio"] * settings.risk.stop_loss_atr_multiple).clip(lower=1e-6)
    dataset["future_return"] = future_return
    dataset["directional_return"] = directional_return
    if getattr(settings.training, "use_sltp_label", True):
        # SL/TP race: much cleaner label than n-bar directional return → higher AUC
        dataset["target"] = _build_sltp_label(dataset, settings)
    else:
        dataset["target"] = np.where(directional_return > settings.training.min_return_threshold, 1, 0)

    # P0: Realistic realized_rr via SL/TP bar-by-bar race (replaces fixed-horizon)
    sltp_rr, sltp_bars = _compute_sltp_realized_rr(dataset, settings)
    dataset["realized_rr"] = sltp_rr
    dataset["bars_held"] = sltp_bars

    # P1a: Session-aware spread multiplier
    _hours = pd.to_datetime(dataset["time"], utc=True).dt.hour
    dataset["session_spread_mult"] = _session_spread_multiplier(_hours)

    dataset["trade_side"] = np.where(dataset["expected_direction"] > 0, "buy", "sell")
    dataset = dataset.dropna().reset_index(drop=True)

    use_date_split = any(
        [
            settings.training.train_start_date,
            settings.training.train_end_date,
            settings.training.test_start_date,
            settings.training.test_end_date,
        ]
    )
    if use_date_split:
        train_mask = _build_date_mask(
            dataset["time"],
            settings.training.train_start_date,
            settings.training.train_end_date,
        )
        test_mask = _build_date_mask(
            dataset["time"],
            settings.training.test_start_date,
            settings.training.test_end_date,
        )
        filtered = dataset.loc[train_mask | test_mask].copy()
        filtered["split"] = "excluded"
        filtered.loc[train_mask.loc[filtered.index], "split"] = "train"
        filtered.loc[test_mask.loc[filtered.index], "split"] = "test"
        filtered = filtered[filtered["split"].isin(["train", "test"])].reset_index(drop=True)
        train_count = (filtered["split"] == "train").sum()
        test_count = (filtered["split"] == "test").sum()
        if train_count == 0 or test_count == 0:
            raise RuntimeError(
                f"Date-based split produced empty train or test set "
                f"(train={train_count}, test={test_count}, "
                f"dataset_rows={len(dataset)}, "
                f"time_min={dataset['time'].min()}, time_max={dataset['time'].max()}, "
                f"time_tz={dataset['time'].dt.tz}, "
                f"train_mask_true={int(train_mask.sum())}, test_mask_true={int(test_mask.sum())})"
            )
        return filtered

    split_index = int(len(dataset) * settings.training.train_split)
    dataset["split"] = "train"
    dataset.loc[split_index:, "split"] = "test"
    return dataset


def build_live_feature_frame(settings: Settings, frames: dict[str, pd.DataFrame], strategy) -> pd.DataFrame:
    # ── DualScalpM1 fast path: build scalp features when model_type matches ──
    import json
    from pathlib import Path as _Path
    try:
        _meta_path = _Path(settings.app.model_meta_path)
        if _meta_path.exists():
            _meta = json.loads(_meta_path.read_text())
            if _meta.get("model_type") == "DualScalpM1":
                return _build_live_scalp_feature_frame(settings, frames)
    except Exception:
        pass
    # ── Standard multi-timeframe path ────────────────────────────────────────
    merged = _merge_context(settings, frames)
    strategy_output = strategy.annotate_dataset(merged)
    dataset = merged.join(strategy_output)
    return dataset.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def _build_live_scalp_feature_frame(settings: Settings, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a live feature frame with SCALP_FEATURE_COLUMNS for DualScalpM1 model.

    Pulls the execution (M1) frame, computes all scalp features, merges M5 context,
    and fills required columns expected by HybridStrategy.build_trade_decision().
    """
    from xauusd_ai.execution.sltp import resolve_setup_exit_targets, setup_label_horizon
    from xauusd_ai.features.scalp_features import build_all_scalp_features, SCALP_FEATURE_COLUMNS

    exec_tf = settings.market.execution_timeframe  # "M1"
    m1 = frames.get(exec_tf, frames.get("M1"))
    if m1 is None or m1.empty:
        raise ValueError(f"_build_live_scalp_feature_frame: '{exec_tf}' frame missing from fetch.")

    m1 = m1.copy().reset_index(drop=True)

    # Ensure required tick volume cols exist (default 0 if MT5 doesn't provide)
    for col in ("tick_volume", "tick_volume_delta", "volume_imbalance"):
        if col not in m1.columns:
            m1[col] = 0.0

    featured = build_all_scalp_features(m1)

    # ── Merge M5 context (m5_bias, m5_rsi_14, m5_atr_norm) ────────────────
    m5 = frames.get("M5")
    if m5 is not None and not m5.empty:
        m5 = m5.copy().sort_values("time").reset_index(drop=True)
        from xauusd_ai.features.indicators import rsi as _rsi_ind, atr as _atr_ind
        m5_rsi = _rsi_ind(m5["close"], 14).iloc[-1]
        m5_atr_series = _atr_ind(m5, 14)
        m5_atr = m5_atr_series.iloc[-1]
        m5_atr_mean = m5_atr_series.rolling(50).mean().iloc[-1]
        m5_bias = float(1 if m5["close"].iloc[-1] > m5["close"].rolling(20).mean().iloc[-1] else -1)
        m5_atr_norm = float(m5_atr / m5_atr_mean) if m5_atr_mean and m5_atr_mean > 0 else 1.0
        featured["m5_bias"]    = m5_bias
        featured["m5_rsi_14"]  = float(m5_rsi)
        featured["m5_atr_norm"] = m5_atr_norm
    else:
        featured["m5_bias"]    = 0.0
        featured["m5_rsi_14"]  = 50.0
        featured["m5_atr_norm"] = 1.0

    # ── Compute ATR14 for SL/TP (used by RiskManager) ─────────────────────
    from xauusd_ai.features.indicators import atr as _atr_ind2
    featured["atr"] = _atr_ind2(featured, 14)

    # ── Compute volatility_regime from ATR expansion ───────────────────────
    atr5_norm = featured.get("atr5_norm_pct", featured["atr"] / featured["atr"].rolling(50).mean().fillna(1))
    latest_atr_norm = float(atr5_norm.ffill().iloc[-1]) if not atr5_norm.empty else 1.0
    if latest_atr_norm < float(settings.strategy.sideways_volatility_threshold) * 100:
        vol_regime = 0  # sideways
    elif latest_atr_norm > float(settings.strategy.strong_volatility_threshold) * 100:
        vol_regime = 2  # volatile
    else:
        vol_regime = 1  # normal
    featured["volatility_regime"] = vol_regime

    featured["trend_strength_score"] = featured.get("trend_strength_score", 0.0)
    featured["pullback_quality"] = featured.get("pullback_quality", 0.0)
    featured["execution_quality"] = featured.get("execution_quality", 0.0)
    featured["strategy_score"] = featured.get("strategy_setup_score", 0.0)

    # ── Required columns defaulted for compatibility ─────────────────────
    for _col, _default in [
        ("news_is_blackout", 0),
        ("news_impact_ahead", 0),
        ("news_hours_ahead", 48.0),
    ]:
        if _col not in featured.columns:
            featured[_col] = _default

    if "trend_alignment" not in featured.columns:
        featured["trend_alignment"] = (
            np.sign(pd.to_numeric(featured.get("m1_bos", 0.0), errors="coerce").fillna(0.0))
            == np.sign(pd.to_numeric(featured.get("m5_bias", 0.0), errors="coerce").fillna(0.0))
        ).astype(int)

    if settings.risk.setup_exit_enabled:
        setup_rows = []
        base_horizon = int(getattr(settings.training, "sltp_label_max_horizon", settings.training.label_horizon) or 8)
        for _, row in featured.iterrows():
            trade_side = "buy" if float(row.get("strategy_setup_score", row.get("m1_bos", 0.0)) or 0.0) >= 0 else "sell"
            row_payload = row.to_dict()
            row_payload["trade_side"] = trade_side
            row_payload["expected_direction"] = 1 if trade_side == "buy" else -1
            tp_rr, sl_mult, tier, _, _ = resolve_setup_exit_targets(
                row_payload,
                enabled=True,
                tp_scale=float(settings.risk.setup_exit_scale),
            )
            hold_bars = setup_label_horizon(base_horizon, row_payload)
            setup_rows.append(
                {
                    "setup_tp_rr": tp_rr,
                    "setup_sl_mult": sl_mult,
                    "setup_tier": tier,
                    "bars_held": hold_bars,
                    "setup_exit_scaled": 1,
                }
            )
        setup_df = pd.DataFrame(setup_rows, index=featured.index)
        featured[["setup_tp_rr", "setup_sl_mult", "setup_tier", "bars_held", "setup_exit_scaled"]] = setup_df
    else:
        featured["setup_tp_rr"] = 0.0
        featured["setup_sl_mult"] = 0.0
        featured["setup_tier"] = 0
        featured["bars_held"] = int(getattr(settings.training, "label_horizon", 8) or 8)
        featured["setup_exit_scaled"] = 0

    return featured.dropna(subset=SCALP_FEATURE_COLUMNS[:5]).reset_index(drop=True)
