from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.features.indicators import atr, macd, rsi, zscore


FEATURE_COLUMNS = [
    "daily_bias",
    "hourly_bias",
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
]


def _build_date_mask(series: pd.Series, start: str | None, end: str | None) -> pd.Series:
    mask = pd.Series(True, index=series.index)
    if start:
        mask &= series >= pd.to_datetime(start, utc=True)
    if end:
        mask &= series <= pd.to_datetime(end, utc=True)
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
    return merged


def prepare_training_dataset(settings: Settings, frames: dict[str, pd.DataFrame], strategy) -> pd.DataFrame:
    merged = _merge_context(settings, frames)
    strategy_output = strategy.annotate_dataset(merged)
    dataset = merged.join(strategy_output)
    future_close = dataset["close"].shift(-settings.training.label_horizon)
    future_return = (future_close - dataset["close"]) / dataset["close"]
    dataset["expected_direction"] = np.sign(dataset["strategy_score"]).replace(0, 1)
    directional_return = future_return * dataset["expected_direction"]
    stop_loss_return = (dataset["atr_ratio"] * settings.risk.stop_loss_atr_multiple).clip(lower=1e-6)
    dataset["future_return"] = future_return
    dataset["directional_return"] = directional_return
    dataset["target"] = np.where(directional_return > settings.training.min_return_threshold, 1, 0)
    dataset["realized_rr"] = (directional_return / stop_loss_return).clip(lower=-1.0, upper=settings.risk.take_profit_rr)
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
        if filtered[filtered["split"] == "train"].empty or filtered[filtered["split"] == "test"].empty:
            raise RuntimeError("Date-based split produced empty train or test set")
        return filtered

    split_index = int(len(dataset) * settings.training.train_split)
    dataset["split"] = "train"
    dataset.loc[split_index:, "split"] = "test"
    return dataset


def build_live_feature_frame(settings: Settings, frames: dict[str, pd.DataFrame], strategy) -> pd.DataFrame:
    merged = _merge_context(settings, frames)
    strategy_output = strategy.annotate_dataset(merged)
    dataset = merged.join(strategy_output)
    return dataset.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
