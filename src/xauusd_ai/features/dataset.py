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
)


# Layout: D1(1) | H4 ICT(9) | H1 Wyckoff(3) | M15 execution(15) | News(5)
# Total: 33 features — multi-timeframe ICT + Wyckoff + News Awareness
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
    _h4_cols = [
        "time", "h4_bos", "h4_choch", "h4_fvg", "h4_order_block",
        "h4_displacement", "h4_ehl", "h4_market_structure_bias",
        "h4_ict_confluence", "h4_premium_discount",
    ]
    merged = pd.merge_asof(
        merged.sort_values("time"),
        h4_frame[_h4_cols].sort_values("time"),
        on="time",
    )
    merged[[c for c in _h4_cols if c != "time"]] = (
        merged[[c for c in _h4_cols if c != "time"]].fillna(0)
    )

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
