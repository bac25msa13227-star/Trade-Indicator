from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    plus_dm = (high.diff()).where((high.diff() > -low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0.0)
    true_range = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr_s = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_s
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_s
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100.0
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def last_rank(values: np.ndarray) -> float:
    if len(values) == 0 or not np.isfinite(values[-1]):
        return np.nan
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan
    return float((finite <= values[-1]).mean())


def build_features(rates: pd.DataFrame) -> pd.DataFrame:
    df = rates.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "tick_volume", "spread"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["tick_volume"].fillna(0.0)

    df["returns"] = close.pct_change().fillna(0.0)
    df["rsi"] = rsi(close)
    macd_line = ema(close, 12) - ema(close, 26)
    macd_signal = ema(macd_line, 9)
    df["macd_hist"] = macd_line - macd_signal
    df["atr"] = atr(df)
    df["atr_mean"] = df["atr"].rolling(100, min_periods=20).mean()
    df["atr_ratio"] = df["atr"] / df["atr_mean"].replace(0.0, np.nan)
    bar_range = (high - low).replace(0.0, np.nan)
    df["range_efficiency"] = (close - open_).abs() / bar_range
    df["tick_volume_zscore"] = (
        (volume - volume.rolling(50, min_periods=10).mean())
        / volume.rolling(50, min_periods=10).std().replace(0.0, np.nan)
    )
    day_open = open_.groupby(df["time"].dt.date).transform("first")
    df["session_return"] = close / day_open.replace(0.0, np.nan) - 1.0

    bb_mid = close.rolling(20, min_periods=10).mean()
    bb_std = close.rolling(20, min_periods=10).std()
    df["bb_position"] = (close - bb_mid) / (2.0 * bb_std).replace(0.0, np.nan)
    low_14 = low.rolling(14, min_periods=7).min()
    high_14 = high.rolling(14, min_periods=7).max()
    stoch = (close - low_14) / (high_14 - low_14).replace(0.0, np.nan) * 100.0
    df["stoch_k"] = stoch - 50.0
    stoch_d = stoch.rolling(3, min_periods=2).mean()
    df["stoch_kd_diff"] = stoch - stoch_d
    df["rsi_slope"] = df["rsi"].diff(5)
    df["adx"] = adx(df)
    df["candle_body_ratio"] = (close - open_).abs() / bar_range
    df["price_roc"] = close.pct_change(12)
    df["atr_percentile"] = df["atr"].rolling(200, min_periods=50).apply(last_rank, raw=True)

    hour = df["time"].dt.hour + df["time"].dt.minute / 60.0
    radians = 2.0 * np.pi * hour / 24.0
    df["time_hour_sin"] = np.sin(radians)
    df["time_hour_cos"] = np.cos(radians)

    ema_20 = ema(close, 20)
    ema_50 = ema(close, 50)
    ema_200 = ema(close, 200)
    df["trend_alignment"] = np.sign(ema_20 - ema_50).fillna(0.0)
    df["price_vs_h4_sma"] = (close - ema_200) / df["atr"].replace(0.0, np.nan)
    df["price_momentum"] = close.diff(12) / df["atr"].replace(0.0, np.nan)
    df["volatility_regime"] = np.select(
        [df["atr_percentile"] < 0.33, df["atr_percentile"] > 0.66],
        [0, 2],
        default=1,
    )
    df["regime_trending"] = (df["adx"] >= 22).astype(float)
    df["regime_sideway"] = (df["adx"] < 18).astype(float)
    df["regime_volatile"] = (df["atr_percentile"] >= 0.66).astype(float)
    df["regime_score"] = (
        0.4 * df["regime_trending"]
        - 0.3 * df["regime_sideway"]
        + 0.2 * (df["atr_percentile"].fillna(0.5) - 0.5)
    )
    df["regime_favorable"] = (df["regime_score"] > 0).astype(float)

    # Directional placeholders based only on MT5 broker bars. Direction-mode
    # in the trainer can override this to buy-only/sell-only.
    trend_score = (
        np.sign(ema_20 - ema_50).fillna(0.0)
        + np.sign(close - ema_200).fillna(0.0)
        + np.sign(df["macd_hist"]).fillna(0.0)
    )
    df["expected_direction"] = np.where(trend_score >= 0, 1, -1)
    df["trade_side"] = np.where(df["expected_direction"] >= 0, "buy", "sell")
    df["news_is_blackout"] = 0.0

    usable = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "returns",
        "rsi",
        "macd_hist",
        "atr",
        "atr_mean",
        "atr_ratio",
        "range_efficiency",
        "tick_volume_zscore",
        "session_return",
        "bb_position",
        "stoch_k",
        "stoch_kd_diff",
        "rsi_slope",
        "adx",
        "candle_body_ratio",
        "price_roc",
        "time_hour_sin",
        "time_hour_cos",
        "atr_percentile",
        "trend_alignment",
        "price_vs_h4_sma",
        "price_momentum",
        "volatility_regime",
        "regime_trending",
        "regime_sideway",
        "regime_volatile",
        "regime_score",
        "regime_favorable",
        "expected_direction",
        "trade_side",
        "news_is_blackout",
    ]
    out = df[usable].replace([np.inf, -np.inf], np.nan).dropna(subset=["atr", "rsi", "adx"]).copy()
    return out.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact model features from MT5 Strategy Tester M5 rates.")
    parser.add_argument("--rates", type=Path, default=ROOT / "outputs/mt5_rates_export_202306_202603.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/mt5_rate_features_202306_202603.csv")
    args = parser.parse_args()

    rates = pd.read_csv(args.rates)
    features = build_features(rates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False)
    print({"rows": len(features), "cols": len(features.columns), "out": str(args.out)})
    print(features[["time", "open", "high", "low", "close", "atr", "trade_side"]].head().to_string(index=False))
    print(features[["time", "open", "high", "low", "close", "atr", "trade_side"]].tail().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
