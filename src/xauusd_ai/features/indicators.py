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


def fair_value_gap(frame: pd.DataFrame) -> pd.Series:
    """
    Fair Value Gap (FVG / Imbalance):
      Bullish FVG: gap giữa high của nến 2 trước và low của nến hiện tại
                   low[i] > high[i-2]  →  vùng imbalance chưa lấp
      Bearish FVG: high[i] < low[i-2]
    Returns: +1=bullish FVG, -1=bearish FVG, 0=none
    """
    bullish_fvg = (frame["low"] > frame["high"].shift(2)).astype(int)
    bearish_fvg = (frame["high"] < frame["low"].shift(2)).astype(int)
    return (bullish_fvg - bearish_fvg).fillna(0).astype(int)


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


def equal_highs_lows(frame: pd.DataFrame, tolerance: float = 0.0003, lookback: int = 20) -> pd.Series:
    """
    Equal Highs / Equal Lows — vùng thanh khoản tập trung:
      Equal Highs: các đỉnh gần nhau → sell-side liquidity bên trên
      Equal Lows : các đáy gần nhau  → buy-side  liquidity bên dưới
    Returns: +1=equal highs (watch for bearish sweep),
             -1=equal lows  (watch for bullish sweep), 0=none
    """
    close = frame["close"].replace(0, np.nan)
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

