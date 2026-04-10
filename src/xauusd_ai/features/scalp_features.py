"""
Scalping feature engineering — M1 native indicators.

Designed for XAUUSD M1 scalping: predict SL/TP hit within 8 M1 bars (8 min).
All features are computable purely from M1 OHLCV + tick_volume_delta + volume_imbalance.
M5 context is merged externally (just bias direction + RSI).

Feature groups:
  1. Fast momentum   — EMA crossovers, RSI(5), MACD(3,8,3), ROC
  2. Order flow      — tick_vol delta cumulative, pressure, surge, OBV
  3. Microstructure  — wick rejection, body ratio, inside/outside bar, ATR regime
  4. Structure       — FVG(3-bar), BOS on M1, liquidity sweeps, swing distance
  5. Session context — VWAP deviation, time encoding (hour+minute cyclical), session ID
  6. Bollinger + Stochastics (short period, M1-tuned)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ─── Fast helpers ─────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 5) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return (100 - (100 / (1 + gain / loss.replace(0, np.nan)))).fillna(50.0)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mu  = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return ((series - mu) / std).fillna(0)


def _atr(frame: pd.DataFrame, period: int = 5) -> pd.Series:
    hl  = frame["high"] - frame["low"]
    hpc = (frame["high"]  - frame["close"].shift()).abs()
    lpc = (frame["low"]   - frame["close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(period).mean().bfill()


def _series_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


# ─── Feature functions ────────────────────────────────────────────────────────

def m1_momentum_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Fast momentum: EMA crossovers, RSI(5), MACD(3,8,3), price ROC.
    All lookbacks ≤21 bars → ≤21 minutes for M1 scalping.
    """
    c = frame["close"]
    e3  = _ema(c, 3);  e8  = _ema(c, 8)
    e13 = _ema(c, 13); e21 = _ema(c, 21)

    # Normalised crossover values (positive = bullish momentum)
    atr5 = _atr(frame, 5).replace(0, np.nan)
    out = pd.DataFrame(index=frame.index)
    out["m1_ema3_8_cross"]  = ((e3 - e8)  / atr5).fillna(0).clip(-3.0, 3.0)
    out["m1_ema8_21_cross"] = ((e8 - e21) / atr5).fillna(0).clip(-3.0, 3.0)
    out["m1_rsi5"]          = (_rsi(c, 5) - 50.0)               # -50..+50
    out["m1_rsi9"]          = (_rsi(c, 9) - 50.0)
    # MACD fast (3,8,3) — tuned for M1
    macd_line   = e3 - e8
    macd_signal = _ema(macd_line, 3)
    out["m1_macd_hist"]     = ((macd_line - macd_signal) / atr5).fillna(0).clip(-3.0, 3.0)
    out["m1_macd_accel"]    = out["m1_macd_hist"].diff(2).fillna(0)  # turning point detector
    # Price rate of change (%)
    out["m1_roc3"]          = c.pct_change(3).fillna(0).clip(-0.01, 0.01) * 1000  # in 0.1%
    out["m1_roc8"]          = c.pct_change(8).fillna(0).clip(-0.02, 0.02) * 1000
    return out


def m1_orderflow_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Order flow pressure from tick_volume_delta and volume_imbalance.
    These are the core "order flow" signals for scalping.

    tick_volume_delta:  positive = buyers increasing (volume accelerating up)
    volume_imbalance:   close pressure = body/range, 0=wick-heavy, 1=full body
    """
    tv  = frame["tick_volume"]
    tvd = frame["tick_volume_delta"]   # MT5 raw tick delta
    vi  = frame["volume_imbalance"]    # body/range ratio

    out = pd.DataFrame(index=frame.index)

    # Cumulative delta momentum: sum of delta over last N bars (buy pressure)
    out["of_delta_cum5"]   = tvd.rolling(5).sum().fillna(0)
    out["of_delta_cum13"]  = tvd.rolling(13).sum().fillna(0)
    # Normalise by ATR-like tick volume std
    tv_std = tv.rolling(20).std().replace(0, np.nan)
    out["of_delta_zscore"] = _zscore(tvd, 20)

    # Delta acceleration: is buy pressure increasing or decreasing?
    out["of_delta_accel"]  = (out["of_delta_cum5"] - out["of_delta_cum5"].shift(5)).fillna(0)

    # Volume imbalance momentum: sustained close pressure
    out["of_vi_ema5"]      = _ema(vi, 5).fillna(0.5) - 0.5   # centred at 0
    out["of_vi_ema13"]     = _ema(vi, 13).fillna(0.5) - 0.5

    # Volume surge: is this bar unusual volume?
    tv_avg = tv.rolling(20).mean().replace(0, np.nan)
    out["of_vol_surge"]    = (tv / tv_avg).fillna(1.0).clip(0.0, 5.0)

    # OBV slope (very short — 5 bars for M1)
    obv = (np.sign(frame["close"].diff()) * tv).fillna(0).cumsum()
    obv_norm = obv / (tv.rolling(20).sum().replace(0, np.nan))
    out["of_obv_slope5"]   = obv_norm.diff(5).fillna(0)

    # Combined order flow score (-1..+1, positive = buy pressure)
    sign_vi = (vi - 0.5) * 2  # centred wick pressure
    sign_body = np.sign(frame["close"] - frame["open"])
    out["of_flow_score"]   = (sign_vi * sign_body).fillna(0)   # body direction × pressure

    return out


def m1_microstructure_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Candlestick microstructure: wick rejection, body shape,
    inside/outside bar, ATR volatility regime.
    """
    o, h, l, c = frame["open"], frame["high"], frame["low"], frame["close"]
    rng = (h - l).replace(0, np.nan)

    out = pd.DataFrame(index=frame.index)

    # Body ratio — 1.0 = no wicks (full conviction), 0 = all wicks (indecision)
    out["ms_body_ratio"]    = ((c - o).abs() / rng).fillna(0)

    # Wick rejection — net wick pressure (+= buying rejection at bottom)
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    out["ms_wick_net"]      = ((lower_wick - upper_wick) / rng).fillna(0).clip(-1.0, 1.0)
    out["ms_upper_wick"]    = (upper_wick / rng).fillna(0)
    out["ms_lower_wick"]    = (lower_wick / rng).fillna(0)

    # Inside bar: range < previous range (compression → awaiting breakout)
    prev_rng = rng.shift(1)
    out["ms_inside_bar"]    = (rng < prev_rng).astype(int)
    # Outside bar: current range > prev range (momentum/absorption)
    out["ms_outside_bar"]   = (rng > prev_rng * 1.5).astype(int)

    # ATR regime: expanding or contracting volatility
    atr5  = _atr(frame, 5)
    atr20 = _atr(frame, 20)
    out["ms_atr5_norm"]     = (atr5 / c.replace(0, np.nan) * 1000).fillna(0)   # ATR in 0.1%
    out["ms_atr_expansion"] = (atr5 / atr20.replace(0, np.nan)).fillna(1.0).clip(0.3, 3.0)

    # ATR percentile (rolling 50)
    out["ms_atr_pct"]       = atr5.rolling(50, min_periods=20).rank(pct=True).fillna(0.5)

    # Close position in range (where does price close? 0=bottom, 1=top)
    out["ms_close_in_rng"]  = ((c - l) / rng).fillna(0.5)

    return out


def m1_structure_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Micro-structure: FVG (3-bar), BOS/sweep on M1 swing windows,
    proximity to recent highs/lows, round number distance.
    """
    h, l, c = frame["high"], frame["low"], frame["close"]
    atr5  = _atr(frame, 5)

    out = pd.DataFrame(index=frame.index)

    # ── 3-bar Fair Value Gap (M1-scale) ──────────────────────────────
    # Bullish FVG: bar[i-2].high < bar[i].low (gap nobody traded through)
    # Bearish FVG: bar[i-2].low > bar[i].high
    gap_bull = (l - h.shift(2)).clip(lower=0) / atr5.replace(0, np.nan)
    gap_bear = (l.shift(2) - h).clip(lower=0) / atr5.replace(0, np.nan)
    out["m1_fvg"]           = (gap_bull - gap_bear).fillna(0).clip(-2.0, 2.0)

    # ── Liquidity sweeps at rolling windows ──────────────────────────
    # Did price just sweep the 8-bar / 20-bar high or low?
    swing_h8  = h.rolling(8).max().shift(1)
    swing_l8  = l.rolling(8).min().shift(1)
    swing_h20 = h.rolling(20).max().shift(1)
    swing_l20 = l.rolling(20).min().shift(1)
    out["m1_sweep_h8"]   = (h > swing_h8).astype(int)
    out["m1_sweep_l8"]   = (l < swing_l8).astype(int)
    out["m1_sweep_h20"]  = (h > swing_h20).astype(int)
    out["m1_sweep_l20"]  = (l < swing_l20).astype(int)

    # ── BOS on M1 (8-bar swing) ───────────────────────────────────────
    # Bullish BOS = close > 8-bar high; Bearish BOS = close < 8-bar low
    bos_bull = (c > swing_h8).astype(int)
    bos_bear = (c < swing_l8).astype(int)
    out["m1_bos"]            = (bos_bull - bos_bear).clip(-1, 1)

    # Distance to swing highs/lows (in ATR units)
    out["m1_dist_swing_h"]  = ((swing_h20 - c) / atr5.replace(0, np.nan)).fillna(3.0).clip(0.0, 5.0)
    out["m1_dist_swing_l"]  = ((c - swing_l20) / atr5.replace(0, np.nan)).fillna(3.0).clip(0.0, 5.0)

    # ── Round number proximity (gold: $1, $5, $10, $50 levels) ───────
    # Close distance to nearest $10 level (key psychological level for XAUUSD)
    round10 = (c / 10.0).round() * 10.0
    round50 = (c / 50.0).round() * 50.0
    out["m1_round10_dist"]  = ((c - round10).abs() / atr5.replace(0, np.nan)).fillna(2.0).clip(0.0, 3.0)
    out["m1_round50_dist"]  = ((c - round50).abs() / atr5.replace(0, np.nan)).fillna(5.0).clip(0.0, 5.0)

    return out


def m1_session_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Session-aware features: VWAP, time encoding, session ID.
    VWAP approximated from session-open tracking within each calendar day.
    """
    c = frame["close"]
    h = frame["high"]
    l = frame["low"]
    tv = frame["tick_volume"]
    t  = frame["time"]

    out = pd.DataFrame(index=frame.index)

    # ── Cyclical time encoding ─────────────────────────────────────────
    # Hour (0-23) and minute (0-59) as sin/cos
    if hasattr(t.iloc[0], 'hour'):
        hour = t.dt.hour + t.dt.minute / 60.0
        minute_of_day = t.dt.hour * 60 + t.dt.minute
    else:
        ts = pd.to_datetime(t, utc=True)
        hour = ts.dt.hour + ts.dt.minute / 60.0
        minute_of_day = ts.dt.hour * 60 + ts.dt.minute

    out["sess_hour_sin"]  = np.sin(2 * np.pi * hour / 24.0)
    out["sess_hour_cos"]  = np.cos(2 * np.pi * hour / 24.0)
    out["sess_min_sin"]   = np.sin(2 * np.pi * (minute_of_day % 60) / 60.0)
    out["sess_min_cos"]   = np.cos(2 * np.pi * (minute_of_day % 60) / 60.0)

    # ── Session ID ─────────────────────────────────────────────────────
    # 0=Asian(0-7 UTC), 1=London(7-12 UTC), 2=NY-London overlap(12-16 UTC), 3=NY(16-22 UTC)
    h_utc = hour.astype(int)
    out["sess_id"] = np.select(
        [h_utc < 7, (h_utc >= 7) & (h_utc < 12),
         (h_utc >= 12) & (h_utc < 16)],
        [0, 1, 2],
        default=3
    )

    # ── Approximate intra-day VWAP deviation ──────────────────────────
    # Group by calendar date, compute expanding VWAP, measure deviation
    typical = (h + l + c) / 3.0
    tv_safe = tv.replace(0, 1e-9)
    if hasattr(t.iloc[0], 'date'):
        date_key = pd.to_datetime(t).dt.date
    else:
        date_key = pd.to_datetime(t, utc=True).dt.date

    # vectorised intraday VWAP: cumsum(tp×vol) / cumsum(vol) per day
    frame2 = pd.DataFrame({"typ": typical, "tv": tv_safe, "date": date_key}, index=frame.index)
    frame2["tvwap_num"] = frame2.groupby("date")["typ"].transform(
        lambda x: (x * tv_safe.reindex(x.index)).cumsum()
    )
    # We can't easily do this grouped in one pass; use a simpler rolling proxy
    # VWAP ~ rolling(240) typical * vol / vol sum  (4-hour context)
    tp_x_v = typical * tv_safe
    rolling_n = 240  # 4h in M1 bars
    vwap_approx = (
        tp_x_v.rolling(rolling_n, min_periods=20).sum() /
        tv_safe.rolling(rolling_n, min_periods=20).sum()
    )
    atr20 = _atr(frame, 20).replace(0, np.nan)
    out["sess_vwap_dev"] = ((c - vwap_approx) / atr20).fillna(0).clip(-3.0, 3.0)

    # Intraday return from session approximate open (240-bar rolling)
    out["sess_return"]    = c.pct_change(60).fillna(0).clip(-0.01, 0.01) * 1000  # 1h return

    return out


def m1_bb_stoch_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Bollinger Bands (20,2) and Stochastic (14,3) on M1.
    """
    c = frame["close"]
    h = frame["high"]
    l = frame["low"]

    out = pd.DataFrame(index=frame.index)

    # Bollinger Bands (20,2)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    out["bb_pos"]    = ((c - bb_mid) / (bb_width / 2)).fillna(0).clip(-2.0, 2.0)
    out["bb_width"]  = (bb_width / c.replace(0, np.nan) * 1000).fillna(1.0)  # width as ‰

    # Stochastic (14,3)
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    k_raw  = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    stoch_k = k_raw.fillna(50.0)
    stoch_d = stoch_k.rolling(3).mean()
    out["stoch_k"]   = (stoch_k - 50.0)   # centred at 0, -50..+50
    out["stoch_kd"]  = (stoch_k - stoch_d).fillna(0)  # crossover momentum

    return out


def m1_strategy_setup_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy-aware setup features inspired by ICT/Wyckoff ideas that are
    realistically derivable from M1 OHLCV + existing scalp signals.
    """
    o = pd.to_numeric(frame["open"], errors="coerce").ffill()
    h = pd.to_numeric(frame["high"], errors="coerce").ffill()
    l = pd.to_numeric(frame["low"], errors="coerce").ffill()
    c = pd.to_numeric(frame["close"], errors="coerce").ffill()
    ts = pd.to_datetime(frame["time"], utc=True, errors="coerce")

    body_sign = np.sign(c - o)
    close_in_rng = _series_or_default(frame, "ms_close_in_rng", 0.5).clip(0.0, 1.0)
    wick_net = _series_or_default(frame, "ms_wick_net", 0.0).clip(-1.0, 1.0)
    flow = _series_or_default(frame, "of_flow_score", 0.0).clip(-1.0, 1.0)
    bos = _series_or_default(frame, "m1_bos", 0.0).clip(-1.0, 1.0)
    fvg = _series_or_default(frame, "m1_fvg", 0.0).clip(-2.0, 2.0)
    vol_surge = _series_or_default(frame, "of_vol_surge", 1.0).clip(0.0, 5.0)
    body_ratio = _series_or_default(frame, "ms_body_ratio", 0.0).clip(0.0, 1.0)
    sweep_h20 = _series_or_default(frame, "m1_sweep_h20", 0.0)
    sweep_l20 = _series_or_default(frame, "m1_sweep_l20", 0.0)
    m5_bias = _series_or_default(frame, "m5_bias", 0.0).clip(-1.0, 1.0)

    out = pd.DataFrame(index=frame.index)

    swing_low20 = l.rolling(20).min().shift(1)
    swing_high20 = h.rolling(20).max().shift(1)
    turtle_bull = (l < swing_low20) & (c > swing_low20) & (close_in_rng > 0.55)
    turtle_bear = (h > swing_high20) & (c < swing_high20) & (close_in_rng < 0.45)
    out["sm_turtle_soup"] = (turtle_bull.astype(int) - turtle_bear.astype(int)).clip(-1, 1)

    swing_low24 = l.rolling(24).min()
    swing_high24 = h.rolling(24).max()
    swing_range24 = (swing_high24 - swing_low24).replace(0, np.nan)
    pct_in_swing = ((c - swing_low24) / swing_range24).fillna(0.5).clip(0.0, 1.0)
    trend = np.sign(_ema(c, 8) - _ema(c, 21)).fillna(0.0)
    bull_ote = (1.0 - (pct_in_swing.sub(0.30).abs() / 0.18)).clip(0.0, 1.0)
    bear_ote = (1.0 - (pct_in_swing.sub(0.70).abs() / 0.18)).clip(0.0, 1.0)
    out["sm_ote_score"] = np.where(trend > 0, bull_ote, np.where(trend < 0, -bear_ote, 0.0))

    bear_fvg_recent = fvg.shift(1).rolling(4).min().fillna(0.0) < -0.10
    bull_fvg_recent = fvg.shift(1).rolling(4).max().fillna(0.0) > 0.10
    bull_ifvg = bear_fvg_recent & (c > h.shift(1)) & (body_sign > 0) & (bos > 0)
    bear_ifvg = bull_fvg_recent & (c < l.shift(1)) & (body_sign < 0) & (bos < 0)
    out["sm_ifvg"] = (bull_ifvg.astype(int) - bear_ifvg.astype(int)).clip(-1, 1)

    recent_sweep_low = sweep_l20.rolling(3).max().fillna(0.0) > 0
    recent_sweep_high = sweep_h20.rolling(3).max().fillna(0.0) > 0
    bull_unicorn = recent_sweep_low & (fvg > 0.10) & (bos > 0) & (flow > 0)
    bear_unicorn = recent_sweep_high & (fvg < -0.10) & (bos < 0) & (flow < 0)
    out["sm_unicorn"] = (bull_unicorn.astype(int) - bear_unicorn.astype(int)).clip(-1, 1)

    hour = ts.dt.hour.fillna(-1).astype(int)
    date_key = ts.dt.date.astype(str)
    asia_mask = hour < 7
    asia_ref = pd.DataFrame(
        {
            "date": date_key,
            "asia_high": h.where(asia_mask),
            "asia_low": l.where(asia_mask),
        },
        index=frame.index,
    )
    asia_levels = asia_ref.groupby("date").agg({"asia_high": "max", "asia_low": "min"})
    asia_high = date_key.map(asia_levels["asia_high"]).astype(float)
    asia_low = date_key.map(asia_levels["asia_low"]).astype(float)
    asia_mid = ((asia_high + asia_low) / 2.0).astype(float)
    london_ny = hour.between(7, 16)
    bull_po3 = london_ny & (l < asia_low) & (c > asia_low) & (c > asia_mid) & (flow > -0.15)
    bear_po3 = london_ny & (h > asia_high) & (c < asia_high) & (c < asia_mid) & (flow < 0.15)
    out["sm_po3_bias"] = (bull_po3.astype(int) - bear_po3.astype(int)).clip(-1, 1)

    support = l.rolling(30).min().shift(1)
    resistance = h.rolling(30).max().shift(1)
    spring = (l < support) & (c > support) & (close_in_rng > 0.55) & (vol_surge > 1.05)
    utad = (h > resistance) & (c < resistance) & (close_in_rng < 0.45) & (vol_surge > 1.05)
    out["wyck_spring_utad"] = (spring.astype(int) - utad.astype(int)).clip(-1, 1)

    spring_recent = (out["wyck_spring_utad"].rolling(6).max().shift(1).fillna(0.0) > 0)
    utad_recent = (out["wyck_spring_utad"].rolling(6).min().shift(1).fillna(0.0) < 0)
    sos = spring_recent & (c > h.shift(1)) & (vol_surge > 1.05) & (body_ratio > 0.45) & (flow > 0)
    sow = utad_recent & (c < l.shift(1)) & (vol_surge > 1.05) & (body_ratio > 0.45) & (flow < 0)
    out["wyck_sos_sow"] = (sos.astype(int) - sow.astype(int)).clip(-1, 1)

    sos_recent = (out["wyck_sos_sow"].rolling(6).max().shift(1).fillna(0.0) > 0)
    sow_recent = (out["wyck_sos_sow"].rolling(6).min().shift(1).fillna(0.0) < 0)
    bull_lps = sos_recent & (wick_net > 0) & (flow > 0) & (vol_surge < 1.40) & (close_in_rng > 0.45)
    bear_lpsy = sow_recent & (wick_net < 0) & (flow < 0) & (vol_surge < 1.40) & (close_in_rng < 0.55)
    out["wyck_lps_quality"] = (bull_lps.astype(int) - bear_lpsy.astype(int)).clip(-1, 1)

    trend_strength = (
        0.30 * bos
        + 0.20 * np.sign(flow)
        + 0.15 * out["sm_po3_bias"]
        + 0.20 * out["wyck_sos_sow"]
        + 0.15 * np.sign(m5_bias)
    )
    pullback_quality = (
        0.40 * out["sm_ote_score"]
        + 0.20 * out["sm_turtle_soup"]
        + 0.20 * out["wyck_lps_quality"]
        + 0.20 * wick_net
    )
    execution_quality = (
        0.25 * out["sm_unicorn"]
        + 0.20 * out["sm_ifvg"]
        + 0.20 * out["sm_turtle_soup"]
        + 0.20 * np.sign(flow)
        + 0.15 * body_sign
    )
    out["trend_strength_score"] = np.tanh(trend_strength).clip(-1.0, 1.0)
    out["pullback_quality"] = np.tanh(pullback_quality).clip(-1.0, 1.0)
    out["execution_quality"] = np.tanh(execution_quality).clip(-1.0, 1.0)
    out["strategy_setup_score"] = np.tanh(
        0.35 * out["trend_strength_score"]
        + 0.30 * out["pullback_quality"]
        + 0.35 * out["execution_quality"]
    ).clip(-1.0, 1.0)

    return out.fillna(0.0)


# ─── Feature column registry ──────────────────────────────────────────────────

SCALP_FEATURE_COLUMNS = [
    # 1. Fast momentum (8 features)
    "m1_ema3_8_cross", "m1_ema8_21_cross",
    "m1_rsi5", "m1_rsi9",
    "m1_macd_hist", "m1_macd_accel",
    "m1_roc3", "m1_roc8",
    # 2. Order flow (8 features)
    "of_delta_cum5", "of_delta_cum13",
    "of_delta_zscore", "of_delta_accel",
    "of_vi_ema5", "of_vi_ema13",
    "of_vol_surge", "of_flow_score",
    # 3. Microstructure (9 features)
    "ms_body_ratio", "ms_wick_net",
    "ms_upper_wick", "ms_lower_wick",
    "ms_inside_bar", "ms_outside_bar",
    "ms_atr5_norm", "ms_atr_expansion", "ms_atr_pct",
    "ms_close_in_rng",
    # 4. Structure (12 features)
    "m1_fvg",
    "m1_sweep_h8", "m1_sweep_l8",
    "m1_sweep_h20", "m1_sweep_l20",
    "m1_bos",
    "m1_dist_swing_h", "m1_dist_swing_l",
    "m1_round10_dist", "m1_round50_dist",
    # 5. Session context (8 features)
    "sess_hour_sin", "sess_hour_cos",
    "sess_min_sin",  "sess_min_cos",
    "sess_id",
    "sess_vwap_dev", "sess_return",
    # 6. BB + Stochastics (4 features)
    "bb_pos", "bb_width",
    "stoch_k", "stoch_kd",
    # 7. M5 context — filled in by scalp_dataset.py (3 features)
    "m5_bias",      # sign of M5 EMA8 - EMA21 (-1 / 0 / +1)
    "m5_rsi_14",    # M5 RSI(14) - 50 (-50..+50)
    "m5_atr_norm",  # M5 ATR / price * 1000
    # 8. ICT/Wyckoff setup features (8 features)
    "sm_turtle_soup",
    "sm_ote_score",
    "sm_ifvg",
    "sm_unicorn",
    "sm_po3_bias",
    "wyck_spring_utad",
    "wyck_sos_sow",
    "wyck_lps_quality",
    # 9. Meta quality scores (4 features)
    "trend_strength_score",
    "pullback_quality",
    "execution_quality",
    "strategy_setup_score",
]
# Total: 66 scalp features


def build_all_scalp_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Build all M1 scalp features. Input frame must have:
        time, open, high, low, close, tick_volume,
        tick_volume_delta, volume_imbalance

    Returns the original frame with all scalp feature columns appended.
    Does NOT include M5 context columns (m5_*) — those are merged externally.
    """
    result = frame.copy()
    for part in (
        m1_momentum_features(result),
        m1_orderflow_features(result),
        m1_microstructure_features(result),
        m1_structure_features(result),
        m1_session_features(result),
        m1_bb_stoch_features(result),
    ):
        for col in part.columns:
            result[col] = part[col].values
    setup_part = m1_strategy_setup_features(result)
    for col in setup_part.columns:
        result[col] = setup_part[col].values
    return result
