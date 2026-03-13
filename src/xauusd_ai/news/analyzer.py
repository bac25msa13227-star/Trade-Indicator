"""News deviation analyzer and market-feature builder for XAUUSD.

Computes the Deviation = Actual – Forecast for each news event and translates
the deviation into a directional trading bias for gold (XAUUSD):

  - USD positive-surprise events  →  strong USD  →  SELL gold  (bias = -1)
  - USD negative-surprise events  →  weak   USD  →  BUY  gold  (bias = +1)
  - Major EUR/GBP/AUD/CHF/CAD positive-surprise  →  USD weakens  →  BUY gold

Five new features are added to the market DataFrame:

  news_impact_score    : 0–3  (None / Low / Medium / High)
  news_deviation_norm  : signed, clipped to [-3, +3]  (actual–forecast / scale)
  news_gold_bias       : -1 / 0 / +1
  news_in_window       : 1 inside [event−pre_min, event+post_min], else 0
  news_upcoming_impact : max impact of events in the next ``upcoming_hours`` hours
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain knowledge: how each event type relates to USD / gold
# ---------------------------------------------------------------------------

# Events where a POSITIVE deviation means USD STRENGTHENS → sell gold
_USD_BULLISH_KEYWORDS = frozenset(
    [
        "nonfarm payroll", "non-farm payroll", "employment change",
        "gdp", "retail sales", "cpi", "core cpi", "ppi", "core ppi",
        "ism manufacturing", "ism services", "ism non-manufacturing",
        "consumer confidence", "building permits", "housing starts",
        "adp non-farm", "jolts", "federal reserve", "fomc",
        "fed interest rate", "interest rate decision",
        "philly fed", "empire state", "chicago pmi", "dallas fed",
        "trade balance",  # larger surplus = USD positive
        "current account",
        "durable goods", "core durable goods",
        "existing home sales", "new home sales", "pending home sales",
        "industrial production", "capacity utilization",
        "average hourly earnings",
    ]
)

# Events where a POSITIVE deviation means USD WEAKENS → buy gold
_USD_BEARISH_KEYWORDS = frozenset(
    [
        "unemployment rate",
        "initial jobless claims", "jobless claims", "continuing claims",
        "unemployment claims",
        "inflation expectations",  # high inflation → FOMC may act but immediate = gold up
    ]
)

# Weights: impact level → numeric score (used as ML feature)
IMPACT_WEIGHTS: dict[str, int] = {
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "Bank Holiday": 0,
}

# Currencies relevant for XAUUSD (as gold often moves inversely to DXY)
RELEVANT_CURRENCIES = frozenset(["USD", "EUR", "GBP", "AUD", "CHF", "CAD", "XAU"])

# Normalisation scale for deviation → news_deviation_norm
# We divide the raw deviation by this factor then clip to [-3, +3]
# This is event-type agnostic; the sign encodes the direction
_DEVIATION_SCALE = 10.0


def determine_gold_bias(currency: str, event: str, deviation: Optional[float]) -> int:
    """Return the expected gold trading bias given a single news event.

    Returns
    -------
    +1  Buy gold expected
    -1  Sell gold expected
     0  Neutral / unknown
    """
    if deviation is None or deviation == 0.0:
        return 0

    event_lower = event.lower()
    direction = 1 if deviation > 0 else -1  # sign of the deviation

    if currency == "USD":
        if any(kw in event_lower for kw in _USD_BEARISH_KEYWORDS):
            # Higher unemployment / claims → weak USD → gold UP
            return direction  # positive deviation → bias +1 (buy gold)
        # Default USD: positive surprise → strong USD → gold DOWN
        return -direction

    # For other major currencies: if the foreign currency strengthens,
    # DXY weakens, which is typically bullish for gold.
    if currency in ("EUR", "GBP", "AUD", "CHF", "CAD"):
        return direction  # positive surprise → their currency up → USD down → gold up

    if currency == "XAU":
        return direction  # direct gold news

    return 0


# ---------------------------------------------------------------------------
# Main feature-builder
# ---------------------------------------------------------------------------


class NewsAnalyzer:
    """Merges cached news events into a market OHLCV DataFrame as model features.

    Parameters
    ----------
    pre_window_minutes:
        Bars that fall inside [event_time − pre_window, event_time] are tagged as
        "in the news window" (traders are already positioning).
    post_window_minutes:
        Bars up to this many minutes after the event time are also tagged as
        "in the news window" (price reaction period).
    upcoming_hours:
        How far ahead the scheduler looks when computing ``news_upcoming_impact``.
    min_impact:
        Minimum impact level to include in feature computation.
    """

    def __init__(
        self,
        pre_window_minutes: int = 30,
        post_window_minutes: int = 60,
        upcoming_hours: int = 4,
        min_impact: str = "Medium",
    ) -> None:
        self.pre_window_minutes = pre_window_minutes
        self.post_window_minutes = post_window_minutes
        self.upcoming_hours = upcoming_hours
        self.min_impact_weight = IMPACT_WEIGHTS.get(min_impact, 2)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_features(
        self, market_df: pd.DataFrame, news_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add five news-derived feature columns to *market_df*.

        The original DataFrame is not mutated; a copy is returned.

        Parameters
        ----------
        market_df:
            Must contain a ``time`` column (UTC-aware or naive).
        news_df:
            Output of :meth:`~ForexFactoryCrawler.load_as_dataframe`.

        Returns
        -------
        pd.DataFrame
            *market_df* with extra columns:
            ``news_impact_score``, ``news_deviation_norm``, ``news_gold_bias``,
            ``news_in_window``, ``news_upcoming_impact``.
        """
        market_df = market_df.copy()
        _init_news_columns(market_df)

        if news_df.empty or "datetime_utc" not in news_df.columns:
            LOGGER.debug("No news data available; news features set to zero.")
            return market_df

        relevant = self._prepare_news(news_df)
        if relevant.empty:
            LOGGER.debug("No relevant news events after filtering.")
            return market_df

        market_times = _to_utc_series(market_df["time"])
        self._merge_window_features(market_df, market_times, relevant)
        self._merge_upcoming_features(market_df, market_times, relevant)

        return market_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter, enrich and return the relevant subset of *news_df*."""
        df = news_df.copy()

        # Ensure UTC datetimes
        df["datetime_utc"] = _to_utc_series(df["datetime_utc"])
        df = df.dropna(subset=["datetime_utc"])

        # Parse deviation as float
        df["deviation_num"] = pd.to_numeric(df.get("deviation", pd.Series(dtype=float)), errors="coerce")

        # Compute gold bias per event
        df["gold_bias"] = df.apply(
            lambda r: determine_gold_bias(
                str(r.get("currency", "")),
                str(r.get("event", "")),
                r.get("deviation_num"),
            ),
            axis=1,
        )

        # Numeric impact weight
        df["impact_weight"] = df["impact"].map(IMPACT_WEIGHTS).fillna(0).astype(int)

        # Filter to relevant currencies and minimum impact
        relevant = df[
            df["currency"].isin(RELEVANT_CURRENCIES)
            & (df["impact_weight"] >= self.min_impact_weight)
        ].copy()

        return relevant.sort_values("datetime_utc").reset_index(drop=True)

    def _merge_window_features(
        self,
        market_df: pd.DataFrame,
        market_times: pd.Series,
        relevant: pd.DataFrame,
    ) -> None:
        """Vectorised merge of impact_score, deviation_norm, gold_bias, in_window."""
        pre_td = pd.Timedelta(minutes=self.pre_window_minutes)
        post_td = pd.Timedelta(minutes=self.post_window_minutes)

        impact_arr = np.zeros(len(market_df), dtype=np.int8)
        deviation_arr = np.zeros(len(market_df), dtype=np.float32)
        bias_arr = np.zeros(len(market_df), dtype=np.int8)
        window_arr = np.zeros(len(market_df), dtype=np.int8)

        times_np = market_times.values  # numpy datetime64[ns, UTC]

        for _, event in relevant.iterrows():
            evt_time = event["datetime_utc"]
            t_start = (evt_time - pre_td).to_datetime64()
            t_end = (evt_time + post_td).to_datetime64()

            mask = (times_np >= t_start) & (times_np <= t_end)
            if not mask.any():
                continue

            impact = int(event["impact_weight"])
            bias = int(event["gold_bias"])
            dev_num = event.get("deviation_num", np.nan)
            dev_norm = float(np.clip(dev_num / _DEVIATION_SCALE, -3.0, 3.0)) if not np.isnan(dev_num) else 0.0

            # Only overwrite where this event has higher impact than existing
            higher_mask = mask & (impact > impact_arr)
            impact_arr[higher_mask] = impact
            bias_arr[higher_mask] = bias
            deviation_arr[higher_mask] = dev_norm
            window_arr[mask] = 1  # always mark as in-window

        market_df["news_impact_score"] = impact_arr
        market_df["news_deviation_norm"] = deviation_arr.astype(float)
        market_df["news_gold_bias"] = bias_arr.astype(int)
        market_df["news_in_window"] = window_arr.astype(int)

    def _merge_upcoming_features(
        self,
        market_df: pd.DataFrame,
        market_times: pd.Series,
        relevant: pd.DataFrame,
    ) -> None:
        """For each bar, find the max impact of events in the next ``upcoming_hours``."""
        upcoming_td = pd.Timedelta(hours=self.upcoming_hours)
        upcoming_arr = np.zeros(len(market_df), dtype=np.int8)
        times_np = market_times.values

        for _, event in relevant.iterrows():
            evt_time = event["datetime_utc"].to_datetime64()
            impact = int(event["impact_weight"])
            # Bars where: bar_time < event_time <= bar_time + upcoming_td
            mask = (times_np < evt_time) & (evt_time <= times_np + upcoming_td.to_timedelta64())
            higher_mask = mask & (impact > upcoming_arr)
            upcoming_arr[higher_mask] = impact

        market_df["news_upcoming_impact"] = upcoming_arr.astype(int)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _init_news_columns(df: pd.DataFrame) -> None:
    """Set default zero values for all five news feature columns in-place."""
    df["news_impact_score"] = 0
    df["news_deviation_norm"] = 0.0
    df["news_gold_bias"] = 0
    df["news_in_window"] = 0
    df["news_upcoming_impact"] = 0


def _to_utc_series(series: pd.Series) -> pd.Series:
    """Convert a datetime series to UTC-aware, returning a fresh Series."""
    if pd.api.types.is_datetime64_any_dtype(series):
        if series.dt.tz is None:
            return series.dt.tz_localize("UTC", ambiguous="infer", nonexistent="shift_forward")
        return series.dt.tz_convert("UTC")
    return pd.to_datetime(series, utc=True, errors="coerce")
