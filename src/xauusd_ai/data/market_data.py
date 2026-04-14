from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import urllib.parse as _urlparse
import urllib.request as _urllib
import urllib.error as _urlerr

import logging as _logging

import pandas as pd

from xauusd_ai.config import Settings

_log = _logging.getLogger(__name__)

# yfinance is imported lazily — only when training/backtest actually needs it.
# This prevents accidental usage in live/paper trading where it would introduce
# 15-30s delay on GC=F proxy data, fatal for M1 scalping.


def _import_yfinance():
    """Lazy import yfinance — raises clear error if not installed."""
    try:
        import yfinance as yf
        return yf
    except ImportError:
        raise ImportError(
            "yfinance is not installed. It is only needed for training/backtest. "
            "Install with: pip install yfinance"
        )

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


TIMEFRAME_MAP = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "60m",
    "H4": "60m",
    "D1": "1d",
}

YFINANCE_PERIOD_MAP = {
    "M1": "7d",
    "M5": "30d",
    "M15": "60d",
    "M30": "60d",
    "H1": "730d",
    "H4": "730d",
    "D1": "10y",
}

YFINANCE_FALLBACK_TICKERS = ["GC=F", "GLD"]

PANDAS_RESAMPLE_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
}


@dataclass
class NewsEvent:
    title: str
    timestamp: datetime
    impact: str


class NewsFilter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def upcoming_events(self) -> list[NewsEvent]:
        try:
            from xauusd_ai.data.news_features import build_news_calendar
            now = pd.Timestamp.now(tz="UTC")
            cal = build_news_calendar(now, now + pd.Timedelta(hours=24))
            if cal.empty:
                return []
            hi = cal[cal["impact"] == "High"]
            return [
                NewsEvent(
                    title=str(row["event"]),
                    timestamp=row["datetime_utc"].to_pydatetime(),
                    impact=str(row["impact"]),
                )
                for _, row in hi.iterrows()
            ]
        except Exception:
            return []


class MarketDataService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.news_filter = NewsFilter(settings)

    def _bridge_url(self) -> str:
        return os.getenv("MT5_BRIDGE_URL", "").rstrip("/")

    def _bridge_call_json(self, path: str):
        bridge_url = self._bridge_url()
        if not bridge_url:
            raise RuntimeError("MT5_BRIDGE_URL is not set")
        url = bridge_url + path
        req = _urllib.Request(url, method="GET")
        try:
            with _urllib.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read())
                if isinstance(payload, dict) and "error" in payload:
                    raise RuntimeError(str(payload["error"]))
                return payload
        except _urlerr.URLError as exc:
            raise RuntimeError(f"MT5 bridge unreachable at {url}: {exc}") from exc

    def _initialize_mt5(self) -> None:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package is not installed in this environment")

        if self.settings.integrations.mt5.enabled:
            mt5cfg = self.settings.integrations.mt5
            # Prefer direct config values (login, server) over env vars
            # to avoid session conflicts when running 2 accounts simultaneously
            login    = str(mt5cfg.login)    if mt5cfg.login    else os.getenv(mt5cfg.login_env)
            password = mt5cfg.password      or  os.getenv(mt5cfg.password_env)
            server   = mt5cfg.server        or  os.getenv(mt5cfg.server_env)
            path     = mt5cfg.terminal_path or None
            if login and password and server:
                kwargs: dict = dict(login=int(login), password=password, server=server)
                if path:
                    kwargs["path"] = path
                if not mt5.initialize(**kwargs):
                    raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
                return

        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    def _fetch_rates_mt5(self, timeframe_name: str, bars: int) -> pd.DataFrame:
        if mt5 is None and self._bridge_url():
            return self._fetch_rates_bridge(timeframe_name, bars)
        self._initialize_mt5()
        mt5_timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        timeframe = mt5_timeframe_map[timeframe_name]
        mt5.symbol_select(self.settings.market.symbol, True)
        # Retry with exponential backoff — MT5 needs time to download history on first start
        import time as _time
        rates = None
        for _attempt in range(10):
            rates = mt5.copy_rates_from_pos(self.settings.market.symbol, timeframe, 0, bars)
            if rates is not None and len(rates) > 0:
                break
            _sleep = min(3 * (1.5 ** _attempt), 15)  # 3s, 4.5s, 6.75s, ... max 15s
            _time.sleep(_sleep)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned for {self.settings.market.symbol} {timeframe_name}")

        frame = pd.DataFrame(rates)
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame["spread_points"] = frame["spread"].fillna(0)
        frame["tick_volume_delta"] = frame["tick_volume"].diff().fillna(0)
        frame["volume_imbalance"] = (
            (frame["close"] - frame["open"]).abs() / (frame["high"] - frame["low"]).replace(0, pd.NA)
        ).fillna(0)
        return frame

    def _fetch_rates_bridge(self, timeframe_name: str, bars: int) -> pd.DataFrame:
        symbol = self.settings.market.symbol
        query = _urlparse.urlencode(
            {
                "symbol": symbol,
                "timeframe": timeframe_name,
                "count": int(bars),
                "start_pos": 0,
            }
        )
        payload = self._bridge_call_json(f"/bars?{query}")
        if not payload:
            raise RuntimeError(f"No bridge bars returned for {symbol} {timeframe_name}")

        frame = pd.DataFrame(payload).rename(columns={"spread": "spread_points"})
        required_columns = {"time", "open", "high", "low", "close"}
        missing = required_columns.difference(frame.columns)
        if missing:
            raise RuntimeError(f"Bridge bars missing columns: {sorted(missing)}")

        if "tick_volume" not in frame.columns:
            frame["tick_volume"] = 0.0
        if "spread_points" not in frame.columns:
            frame["spread_points"] = 0.0

        if pd.api.types.is_numeric_dtype(frame["time"]):
            frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        else:
            frame["time"] = pd.to_datetime(frame["time"], utc=True)

        frame = (
            frame.sort_values("time")
            .drop_duplicates(subset=["time"])
            .tail(int(bars))
            .reset_index(drop=True)
        )
        frame["tick_volume_delta"] = pd.to_numeric(frame["tick_volume"], errors="coerce").fillna(0.0).diff().fillna(0.0)
        frame["volume_imbalance"] = (
            (pd.to_numeric(frame["close"], errors="coerce") - pd.to_numeric(frame["open"], errors="coerce")).abs()
            / (pd.to_numeric(frame["high"], errors="coerce") - pd.to_numeric(frame["low"], errors="coerce")).replace(0, pd.NA)
        ).fillna(0.0)
        return frame

    def _fetch_rates_yfinance(self, timeframe_name: str, bars: int) -> pd.DataFrame:
        yf = _import_yfinance()
        interval = TIMEFRAME_MAP[timeframe_name]
        period = YFINANCE_PERIOD_MAP[timeframe_name]
        requested_ticker = self.settings.market.training_symbol or "GC=F"
        tickers = [requested_ticker, *[ticker for ticker in YFINANCE_FALLBACK_TICKERS if ticker != requested_ticker]]
        history = None
        ticker = requested_ticker
        for candidate in tickers:
            history = yf.Ticker(candidate).history(period=period, interval=interval, auto_adjust=False)
            if history is not None and not history.empty:
                ticker = candidate
                break
        if history is None or history.empty:
            raise RuntimeError(f"No Yahoo Finance data returned for {requested_ticker} {timeframe_name}")

        frame = history.reset_index().rename(
            columns={
                "Datetime": "time",
                "Date": "time",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "tick_volume",
            }
        )
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        if timeframe_name == "H4":
            frame = (
                frame.set_index("time")
                .resample("4h")
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "tick_volume": "sum",
                })
                .dropna()
                .reset_index()
            )
        frame = frame.tail(bars).reset_index(drop=True)
        frame["spread_points"] = 0.0
        frame["tick_volume_delta"] = frame["tick_volume"].diff().fillna(0)
        frame["volume_imbalance"] = (
            (frame["close"] - frame["open"]).abs() / (frame["high"] - frame["low"]).replace(0, pd.NA)
        ).fillna(0)
        return frame

    # Sources that are safe for live/paper trading (real-time, no delay).
    _LIVE_SAFE_SOURCES = frozenset({"bridge", "mt5_bridge", "mt5", "csv_folder", "csv"})

    def _fetch_rates(self, timeframe_name: str, bars: int, source: str) -> pd.DataFrame:
        if source == "yfinance":
            return self._fetch_rates_yfinance(timeframe_name, bars)
        if source in {"bridge", "mt5_bridge"}:
            return self._fetch_rates_bridge(timeframe_name, bars)
        if source == "csv_folder":
            return self._fetch_rates_csv_folder(timeframe_name, bars)
        if source == "csv":
            return self._fetch_rates_csv(timeframe_name, bars)
        if source == "mt5":
            return self._fetch_rates_mt5(timeframe_name, bars)
        raise ValueError(f"Unsupported data source: {source}")

    def _fetch_rates_csv_folder(self, timeframe_name: str, bars: int) -> pd.DataFrame:
        folder_path = Path(self.settings.market.csv_folder_path)
        if not folder_path.exists():
            raise RuntimeError(f"CSV folder not found: {folder_path}")

        candidates = sorted(folder_path.glob(f"*_{timeframe_name}.csv"))
        if not candidates:
            raise RuntimeError(f"No CSV found for timeframe {timeframe_name} in {folder_path}")

        csv_path = candidates[0]
        frame = pd.read_csv(csv_path)
        normalized = frame.rename(
            columns={
                "Datetime": "time",
                "Date": "time",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "tick_volume",
                "TickVolume": "tick_volume",
                "Spread": "spread_points",
                "spread": "spread_points",
            }
        )
        required_columns = {"time", "open", "high", "low", "close"}
        missing = required_columns.difference(normalized.columns)
        if missing:
            raise RuntimeError(f"CSV folder data missing required columns: {sorted(missing)} in {csv_path}")

        if "tick_volume" not in normalized.columns:
            normalized["tick_volume"] = 0.0
        if "spread_points" not in normalized.columns:
            normalized["spread_points"] = 0.0

        normalized["time"] = pd.to_datetime(normalized["time"], utc=True)
        normalized = normalized.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
        normalized = normalized[["time", "open", "high", "low", "close", "tick_volume", "spread_points"]].tail(bars).copy()
        normalized["tick_volume_delta"] = normalized["tick_volume"].diff().fillna(0)
        normalized["volume_imbalance"] = (
            (normalized["close"] - normalized["open"]).abs() / (normalized["high"] - normalized["low"]).replace(0, pd.NA)
        ).fillna(0)
        return normalized

    def _fetch_rates_csv(self, timeframe_name: str, bars: int) -> pd.DataFrame:
        csv_path = Path(self.settings.market.csv_data_path)
        if not csv_path.exists():
            raise RuntimeError(f"CSV data file not found: {csv_path}")

        frame = pd.read_csv(csv_path)
        normalized = frame.rename(
            columns={
                "Datetime": "time",
                "Date": "time",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "tick_volume",
                "TickVolume": "tick_volume",
                "Spread": "spread_points",
            }
        )
        required_columns = {"time", "open", "high", "low", "close"}
        missing = required_columns.difference(normalized.columns)
        if missing:
            raise RuntimeError(f"CSV data missing required columns: {sorted(missing)}")

        if "tick_volume" not in normalized.columns:
            normalized["tick_volume"] = 0.0
        if "spread_points" not in normalized.columns:
            normalized["spread_points"] = 0.0

        normalized["time"] = pd.to_datetime(normalized["time"], utc=True)
        normalized = normalized.sort_values("time")

        base_timeframe = self.settings.market.csv_timeframe
        if timeframe_name == base_timeframe:
            result = normalized.copy()
        else:
            result = (
                normalized.set_index("time")
                .resample(PANDAS_RESAMPLE_MAP[timeframe_name])
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "tick_volume": "sum",
                    "spread_points": "mean",
                })
                .dropna()
                .reset_index()
            )

        result = result.tail(bars).reset_index(drop=True)
        result["tick_volume_delta"] = result["tick_volume"].diff().fillna(0)
        result["volume_imbalance"] = (
            (result["close"] - result["open"]).abs() / (result["high"] - result["low"]).replace(0, pd.NA)
        ).fillna(0)
        return result

    def fetch_multi_timeframe_data(self, source: str | None = None, all_bars: bool = False) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        required = {
            self.settings.market.higher_timeframe,
            self.settings.market.mid_timeframe,
            self.settings.market.structure_timeframe,
            self.settings.market.execution_timeframe,
        }
        # Live/scalp profiles often need the full configured TF stack available.
        # Prefer the bars-config universe as the fetch set so D1/H4/H1/M30/M15/M5/M1
        # are all hydrated when present in config, especially for bridge-backed live mode.
        required.update({str(tf).upper() for tf in self.settings.market.bars.keys()})
        resolved_source = source or self.settings.market.live_data_source

        # ── GUARD: block yfinance in live/paper trading ──────────────────────
        live_src = self.settings.market.live_data_source
        if resolved_source == "yfinance" and live_src in self._LIVE_SAFE_SOURCES:
            _log.critical(
                "BLOCKED: yfinance requested but live_data_source=%s. "
                "yfinance has 15-30s delay — fatal for M1 scalping. "
                "Using %s instead.",
                live_src, live_src,
            )
            resolved_source = live_src

        for timeframe_name in required:
            # all_bars=True dùng cho training — load toàn bộ CSV không giới hạn
            bars = 999_999_999 if all_bars else self.settings.market.bars[timeframe_name]
            frames[timeframe_name] = self._fetch_rates(timeframe_name, bars, resolved_source)
        return frames

    def market_state(self) -> dict[str, object]:
        return {
            "symbol": self.settings.market.symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "upcoming_news": [event.title for event in self.news_filter.upcoming_events()],
        }
