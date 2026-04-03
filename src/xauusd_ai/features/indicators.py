from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    line = fast_ema - slow_ema
    signal_line = ema(line, signal)
    histogram = line - signal_line
    return line, signal_line, histogram


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift()).abs()
    low_close = (frame["low"] - frame["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std().replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


# ─────────────────────────────────────────────────────────────────────────────
# ICT (Inner Circle Trader) Indicators
# ─────────────────────────────────────────────────────────────────────────────

def bos_choch(frame: pd.DataFrame, swing_lookback: int = 20) -> tuple[pd.Series, pd.Series]:
    """
    BOS  — Break of Structure: giá phá vỡ đỉnh/đáy gần nhất xác nhận xu hướng.
    CHoCH — Change of Character: BOS ngược chiều xu hướng hiện tại → đảo chiều.

    Returns: (bos_signal, choch_signal)  +1=bullish, -1=bearish, 0=none
    """
    swing_high = frame["high"].rolling(swing_lookback).max().shift(1)
    swing_low  = frame["low"].rolling(swing_lookback).min().shift(1)
    close = frame["close"]

    trend = np.sign(ema(close, 20) - ema(close, 50))

    bullish_bos = (close > swing_high).astype(int)
    bearish_bos = (close < swing_low).astype(int)
    bos = (bullish_bos - bearish_bos).clip(-1, 1)

    choch = pd.Series(
        np.where(
            (bullish_bos == 1) & (trend.shift(1) < 0), 1,
            np.where((bearish_bos == 1) & (trend.shift(1) > 0), -1, 0),
        ),
        index=frame.index,
    )
    return bos.fillna(0).astype(int), choch.fillna(0).astype(int)


def fair_value_gap(frame: pd.DataFrame, decay_bars: int = 10) -> pd.Series:
    """
    Fair Value Gap with fill tracking.
    FVG signal persists for up to *decay_bars* (linearly decaying weight)
    and immediately drops to zero once price revisits the gap zone.
    Returns: float in [-1, +1].  Positive=bullish unfilled gap nearby.
    """
    n = len(frame)
    high = frame["high"].values.astype(float)
    low = frame["low"].values.astype(float)

    result = np.zeros(n, dtype=np.float64)
    # Active FVGs: (type +1/-1, gap_low, gap_high, birth_bar)
    active: list[tuple[int, float, float, int]] = []

    for i in range(2, n):
        # Detect new FVGs
        if low[i] > high[i - 2]:      # Bullish FVG
            active.append((1, high[i - 2], low[i], i))
        elif high[i] < low[i - 2]:    # Bearish FVG
            active.append((-1, high[i], low[i - 2], i))

        # Evaluate active FVGs: accumulate signal, remove filled/expired
        remaining: list[tuple[int, float, float, int]] = []
        sig = 0.0
        for fvg_type, gap_lo, gap_hi, birth in active:
            age = i - birth
            if age >= decay_bars:
                continue  # expired
            # Fill check: price entered gap zone → gap filled
            if low[i] <= gap_hi and high[i] >= gap_lo:
                continue  # filled
            remaining.append((fvg_type, gap_lo, gap_hi, birth))
            weight = 1.0 - age / decay_bars
            sig += fvg_type * weight
        active = remaining
        result[i] = max(-1.0, min(1.0, sig))

    return pd.Series(result, index=frame.index)


def order_block(frame: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """
    Order Block (vectorized):
      Bullish OB: nến trước bearish + nến hiện tại bullish impulse mạnh
                  → institutional buy order đặt tại nến bearish trước đó
      Bearish OB: nến trước bullish + nến hiện tại bearish impulse mạnh

    Returns: +1=bullish OB signal, -1=bearish OB signal, 0=none
    """
    body     = (frame["close"] - frame["open"]).abs()
    avg_body = body.rolling(lookback).mean().shift(1)
    is_impulse   = body > (avg_body * 1.3)
    prev_bearish = (frame["close"].shift(1) < frame["open"].shift(1))
    prev_bullish = (frame["close"].shift(1) > frame["open"].shift(1))
    curr_bullish = (frame["close"] > frame["open"])
    curr_bearish = (frame["close"] < frame["open"])
    bullish_ob = (is_impulse & curr_bullish & prev_bearish).astype(int)
    bearish_ob = (is_impulse & curr_bearish & prev_bullish).astype(int)
    return (bullish_ob - bearish_ob).fillna(0).astype(int)


def kill_zone(frame: pd.DataFrame) -> pd.Series:
    """
    Kill Zones (UTC) — thời điểm thanh khoản cao nhất:
      Asian  :  00-03 UTC
      London :  07-10 UTC  (London Open Kill Zone)
      NY     :  12-15 UTC  (NY Open Kill Zone)
    Returns: 1 nếu trong kill zone, 0 nếu không
    """
    hours = pd.to_datetime(frame["time"], utc=True).dt.hour
    in_kz = (hours.between(7, 10) | hours.between(12, 15) | hours.between(0, 3))
    return in_kz.astype(int).fillna(0)


def judas_swing(frame: pd.DataFrame, session_open_hours: tuple = (7, 12)) -> pd.Series:
    """
    Judas Swing: pha quét thanh khoản giả đầu phiên trước khi đảo chiều.
      Bullish Judas: nến tại giờ open có wick dưới dài (quét lows) + close trên open
      Bearish Judas: nến tại giờ open có wick trên dài (quét highs) + close dưới open
    Returns: +1=bullish Judas (bullish reversal signal),
             -1=bearish Judas (bearish reversal signal), 0=none
    """
    hours = pd.to_datetime(frame["time"], utc=True).dt.hour
    at_open = hours.isin(session_open_hours)
    upper_wick = frame["high"] - frame[["open", "close"]].max(axis=1)
    lower_wick = frame[["open", "close"]].min(axis=1) - frame["low"]
    candle_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    bullish_judas = (at_open & (lower_wick > upper_wick * 1.5) & (frame["close"] > frame["open"])).astype(int)
    bearish_judas = (at_open & (upper_wick > lower_wick * 1.5) & (frame["close"] < frame["open"])).astype(int)
    return (bullish_judas - bearish_judas).fillna(0).astype(int)


def displacement(frame: pd.DataFrame, atr_multiple: float = 1.5) -> pd.Series:
    """
    Displacement: cú đẩy mạnh xác nhận dòng tiền tổ chức.
    Body của nến > atr_multiple × ATR trung bình 14 nến.
    Returns: +1=bullish displacement, -1=bearish, 0=none
    """
    body     = (frame["close"] - frame["open"]).abs()
    atr_avg  = (frame["high"] - frame["low"]).rolling(14).mean()
    is_disp  = body > (atr_avg * atr_multiple)
    direction = np.sign(frame["close"] - frame["open"])
    return (is_disp.astype(int) * direction).fillna(0).astype(int)


def equal_highs_lows(frame: pd.DataFrame, atr_factor: float = 0.10, lookback: int = 20) -> pd.Series:
    """
    Equal Highs / Equal Lows — vùng thanh khoản tập trung.
    ATR-relative tolerance: range < atr_factor × ATR / close.
    Returns: +1=equal highs (watch for bearish sweep),
             -1=equal lows  (watch for bullish sweep), 0=none
    """
    close = frame["close"].replace(0, np.nan)
    atr_val = atr(frame)
    tolerance = (atr_factor * atr_val / close).fillna(0.0003)
    recent_high_range = (
        frame["high"].rolling(lookback).max() - frame["high"].rolling(lookback).min()
    ) / close
    recent_low_range = (
        frame["low"].rolling(lookback).max() - frame["low"].rolling(lookback).min()
    ) / close
    equal_highs = (recent_high_range < tolerance).astype(int)
    equal_lows  = (recent_low_range  < tolerance).astype(int)
    return (equal_highs - equal_lows).fillna(0).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Wyckoff Method Indicators
# ─────────────────────────────────────────────────────────────────────────────

def vsa_signal(frame: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Volume Spread Analysis (VSA):
      Effort vs Result:
        - High volume + small body  = institutional absorption (potential reversal)
        - High volume + large body  = institutional momentum (trend continuation)
      Accumulation: high vol, narrow spread, close near high → bullish
      Distribution: high vol, narrow spread, close near low  → bearish
    Returns: +1=accumulation, -1=distribution, 0=neutral
    """
    spread    = (frame["high"] - frame["low"]).replace(0, np.nan)
    body      = (frame["close"] - frame["open"]).abs()
    avg_vol   = frame["tick_volume"].rolling(lookback).mean()
    avg_spread= spread.rolling(lookback).mean()
    avg_body  = body.rolling(lookback).mean()

    high_vol     = frame["tick_volume"] > avg_vol * 1.3
    narrow_spread= spread < avg_spread * 0.8
    close_pos    = ((frame["close"] - frame["low"]) / spread).fillna(0.5)

    # Accumulation: high vol + narrow spread + closes near top
    accumulation = (high_vol & narrow_spread & (close_pos > 0.6))
    # Distribution: high vol + narrow spread + closes near bottom
    distribution = (high_vol & narrow_spread & (close_pos < 0.4))
    # No effort = hidden absorption (body < avg but vol high)
    no_result    = high_vol & (body < avg_body * 0.5)
    bullish_close= frame["close"] > frame["open"]
    bearish_close= frame["close"] < frame["open"]

    signal = np.select(
        [
            (accumulation | (no_result & bullish_close)),
            (distribution | (no_result & bearish_close)),
        ],
        [1, -1],
        default=0,
    )
    return pd.Series(signal, index=frame.index).fillna(0).astype(int)


def wyckoff_spring_upthrust(frame: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Spring  (Phase C Wyckoff): giá tạm thời phá hỗ trợ rồi đóng lại bên trên → tín hiệu mua.
    Upthrust(Phase C Wyckoff): giá tạm thời phá kháng cự rồi đóng lại bên dưới → tín hiệu bán.
    Returns: +1=spring (bullish), -1=upthrust (bearish), 0=none
    """
    support    = frame["low"].rolling(lookback).min().shift(1)
    resistance = frame["high"].rolling(lookback).max().shift(1)

    spring = (
        (frame["low"] < support) &
        (frame["close"] > support) &
        (frame["close"] > frame["open"])
    ).astype(int)

    upthrust = (
        (frame["high"] > resistance) &
        (frame["close"] < resistance) &
        (frame["close"] < frame["open"])
    ).astype(int) * -1

    return (spring + upthrust).fillna(0).astype(int)


def market_structure_bias(bos_series: pd.Series, lookback: int = 8) -> pd.Series:
    """
    Tính xu hướng cấu trúc thị trường tổng hợp từ chuỗi BOS signals.
    BOS tích lũy trong lookback nến → xác định bias hiện tại.
    Returns: +1=bullish structure, -1=bearish structure, 0=neutral
    """
    return bos_series.rolling(lookback).sum().apply(np.sign).fillna(0).astype(int)


def premium_discount_zone(frame: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    ICT Premium / Discount Zone (PDZ) — vùng giá trị theo Fibonacci 50%:
      Discount (giá dưới 40% của swing range): +1  → mua rẻ, buy bias
      Premium  (giá trên 60% của swing range): -1  → mua đắt, sell bias
      Equilibrium (40–60%):                     0  → neutral
    Tính trên H4 → xác định liệu giá đang ở vùng có lợi để vào lệnh.
    Returns: +1=discount (bullish), -1=premium (bearish), 0=equilibrium
    """
    swing_high = frame["high"].rolling(lookback).max()
    swing_low  = frame["low"].rolling(lookback).min()
    range_ = (swing_high - swing_low).replace(0, np.nan)
    pct_in_range = (frame["close"] - swing_low) / range_
    signal = np.select(
        [pct_in_range > 0.60, pct_in_range < 0.40],
        [-1, 1],
        default=0,
    )
    return pd.Series(signal, index=frame.index).fillna(0).astype(int)


def bollinger_bands(
    series: pd.Series, period: int = 20, std_mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands. Returns (upper, middle, lower)."""
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    return middle + std_mult * std, middle, middle - std_mult * std


def stochastic(
    frame: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator. Returns (K%, D%) where K is 0–100."""
    low_min = frame["low"].rolling(k_period).min()
    high_max = frame["high"].rolling(k_period).max()
    k = 100.0 * (frame["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = k.fillna(50.0)
    d = k.rolling(d_period).mean()
    return k, d


def rsi_slope(series: pd.Series, rsi_period: int = 14, slope_period: int = 5) -> pd.Series:
    """RSI momentum slope: positive = RSI rising, negative = falling."""
    rsi_vals = rsi(series, rsi_period)
    return rsi_vals.diff(slope_period).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# Trend Strength & Volume Indicators (v2 — model improvement)
# ─────────────────────────────────────────────────────────────────────────────

def adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength 0–100.
    High ADX (>25) = strong trend; Low ADX (<20) = ranging/choppy."""
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, np.nan))
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(span=period, adjust=False).mean().fillna(0)


def obv_slope(frame: pd.DataFrame, slope_period: int = 10) -> pd.Series:
    """On-Balance Volume slope — volume-confirmed momentum.
    Positive = volume following price up; Negative = volume following price down."""
    direction = np.sign(frame["close"].diff())
    vol = frame.get("tick_volume", frame.get("volume", pd.Series(0, index=frame.index)))
    obv = (direction * vol).cumsum()
    slope = obv.diff(slope_period) / slope_period
    return zscore(slope.fillna(0), 50)


def candle_body_ratio(frame: pd.DataFrame) -> pd.Series:
    """Candle body / total range — conviction measure.
    Values near 1.0 = strong directional candle; near 0.0 = doji/indecision."""
    body = (frame["close"] - frame["open"]).abs()
    total_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    return (body / total_range).fillna(0)


def price_roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change normalized — momentum oscillator."""
    return series.pct_change(period).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# v3 Indicators — Microstructure & Momentum Quality
# ─────────────────────────────────────────────────────────────────────────────

def pullback_depth(frame: pd.DataFrame, trend_lookback: int = 50) -> pd.Series:
    """Fibonacci-like retracement depth within the current trend.
    0 = at the trend extreme (no pullback); ~0.5 = 50% retracement;
    >1 = price reversed beyond the trend start.
    Good entries typically occur at 0.38–0.62 retracement."""
    swing_high = frame["high"].rolling(trend_lookback).max()
    swing_low = frame["low"].rolling(trend_lookback).min()
    swing_range = (swing_high - swing_low).replace(0, np.nan)
    # Measure pullback from upper extreme (how far price has dropped from high)
    depth_from_high = (swing_high - frame["close"]) / swing_range
    # Measure pullback from lower extreme (how far price has risen from low)
    depth_from_low = (frame["close"] - swing_low) / swing_range
    # Use the SMALLER depth — it represents the pullback in the dominant direction
    return pd.Series(
        np.minimum(depth_from_high.values, depth_from_low.values),
        index=frame.index,
    ).fillna(0.5).clip(0, 1.5)


def atr_expansion(frame: pd.DataFrame, fast: int = 7, slow: int = 28) -> pd.Series:
    """ATR(fast) / ATR(slow) — volatility trend.
    >1 = expanding volatility (good for directional trades);
    <1 = contracting volatility (choppy/ranging)."""
    tr = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - frame["close"].shift()).abs(),
        (frame["low"] - frame["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_fast = tr.rolling(fast).mean()
    atr_slow = tr.rolling(slow).mean().replace(0, np.nan)
    return (atr_fast / atr_slow).fillna(1.0).clip(0.3, 3.0)


def wick_rejection(frame: pd.DataFrame) -> pd.Series:
    """Net wick directional pressure: (lower_wick - upper_wick) / range.
    Positive = buying pressure (long lower wick rejects lower prices);
    Negative = selling pressure (long upper wick rejects higher prices)."""
    upper_wick = frame["high"] - frame[["open", "close"]].max(axis=1)
    lower_wick = frame[["open", "close"]].min(axis=1) - frame["low"]
    total_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    return ((lower_wick - upper_wick) / total_range).fillna(0).clip(-1, 1)


def volume_surge(frame: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Current tick volume relative to recent average.
    >1.5 = volume surge (confirms moves); <0.5 = low activity."""
    vol = frame.get("tick_volume", frame.get("volume", pd.Series(0, index=frame.index)))
    avg_vol = vol.rolling(lookback).mean().replace(0, np.nan)
    return (vol / avg_vol).fillna(1.0).clip(0, 5.0)


def close_position_in_range(frame: pd.DataFrame) -> pd.Series:
    """Where close sits within the high-low range: 0=at low, 1=at high.
    Strong bullish candles close near 1.0; strong bearish near 0.0."""
    total_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    return ((frame["close"] - frame["low"]) / total_range).fillna(0.5).clip(0, 1)


def macd_hist_acceleration(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """Rate of change of MACD histogram — momentum turning points.
    Positive = momentum accelerating upward; Negative = decelerating/reversing."""
    _, _, hist = macd(series, fast, slow, signal)
    return hist.diff().fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# v4 Indicators — Institutional Order Flow & Smart Money
# ─────────────────────────────────────────────────────────────────────────────

def volume_delta_momentum(frame: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
    """Bookmap-inspired: volume delta smoothed momentum.
    Estimates buy/sell pressure from candle structure × volume.
    Positive = buying pressure accelerating; Negative = selling pressure."""
    vol = frame.get("tick_volume", frame.get("volume", pd.Series(0, index=frame.index)))
    bar_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    # Estimate buy volume fraction from close position in range
    buy_pct = ((frame["close"] - frame["low"]) / bar_range).fillna(0.5)
    sell_pct = 1.0 - buy_pct
    delta = (buy_pct - sell_pct) * vol
    fast_ma = delta.rolling(fast).mean()
    slow_ma = delta.rolling(slow).mean().replace(0, np.nan)
    return ((fast_ma - slow_ma) / slow_ma.abs().clip(lower=1)).fillna(0).clip(-3, 3)


def institutional_candle_score(frame: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """Detects institutional activity from candle patterns:
    - Large body + above-average volume + closing near extreme = institutional move
    - Score: -1 to +1 (positive = bullish institutional, negative = bearish)"""
    body = (frame["close"] - frame["open"])
    body_abs = body.abs()
    avg_body = body_abs.rolling(lookback).mean()
    vol = frame.get("tick_volume", frame.get("volume", pd.Series(0, index=frame.index)))
    avg_vol = vol.rolling(lookback).mean().replace(0, np.nan)
    bar_range = (frame["high"] - frame["low"]).replace(0, np.nan)

    # Body strength (how much of the candle is body)
    body_ratio = (body_abs / bar_range).fillna(0)
    # Volume conviction (is this move backed by volume)
    vol_ratio = (vol / avg_vol).fillna(1).clip(0, 5)
    # Size relative to average (is this an outsized move)
    size_ratio = (body_abs / avg_body.replace(0, np.nan)).fillna(1).clip(0, 5)

    # Combine: direction × body_quality × volume_conviction × size
    direction = np.sign(body)
    raw_score = direction * body_ratio * np.minimum(vol_ratio, 2.0) * np.minimum(size_ratio, 2.0)
    return raw_score.fillna(0).clip(-1, 1)


def swing_failure_pattern(frame: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """ICT Swing Failure Pattern (SFP): price sweeps a high/low then reverses.
    More reliable than simple liquidity sweep — requires close back inside range.
    +1 = bullish SFP (swept lows, reversed up)
    -1 = bearish SFP (swept highs, reversed down)"""
    swing_high = frame["high"].rolling(lookback).max().shift(1)
    swing_low = frame["low"].rolling(lookback).min().shift(1)

    # Bullish SFP: low penetrates swing low but close is ABOVE swing low AND above open
    bullish_sfp = (
        (frame["low"] < swing_low) &
        (frame["close"] > swing_low) &
        (frame["close"] > frame["open"]) &
        (frame["close"] > frame["low"] + (frame["high"] - frame["low"]) * 0.5)  # closes in upper half
    ).astype(int)

    # Bearish SFP: high penetrates swing high but close is BELOW swing high AND below open
    bearish_sfp = (
        (frame["high"] > swing_high) &
        (frame["close"] < swing_high) &
        (frame["close"] < frame["open"]) &
        (frame["close"] < frame["low"] + (frame["high"] - frame["low"]) * 0.5)  # closes in lower half
    ).astype(int)

    return (bullish_sfp - bearish_sfp).fillna(0).astype(int)

