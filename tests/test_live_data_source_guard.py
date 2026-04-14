"""
Tests for live data source guard — ensure yfinance is NEVER used in live/paper mode.

These tests verify:
1. MarketDataService blocks yfinance during live/paper trading
2. SelfLearner never falls back to yfinance when bridge is configured
3. Bridge is the only data source for real-time M1 scalping
4. yfinance lazy import does not pollinate live code path
"""
from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService


def _settings():
    return load_settings(Path("configs/live_acc2_scalp_m1.yaml"))


def _dummy_frame(n: int = 5) -> pd.DataFrame:
    """Return a minimal valid OHLCV DataFrame."""
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-04-14T00:00:00Z", periods=n, freq="min"),
            "open": [3200.0] * n,
            "high": [3201.0] * n,
            "low": [3199.0] * n,
            "close": [3200.5] * n,
            "tick_volume": [100.0] * n,
            "spread_points": [0.25] * n,
            "tick_volume_delta": [0.0] * n,
            "volume_imbalance": [0.5] * n,
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. MarketDataService guards
# ──────────────────────────────────────────────────────────────────────────────


class TestMarketDataYfinanceGuard:
    """Verify yfinance is blocked/redirected during live trading."""

    def test_config_live_data_source_is_bridge(self):
        """Both live configs must have live_data_source=bridge."""
        for cfg in ("configs/live_acc1_scalp_m1.yaml", "configs/live_acc2_scalp_m1.yaml"):
            if Path(cfg).exists():
                s = load_settings(Path(cfg))
                assert s.market.live_data_source == "bridge", (
                    f"{cfg} has live_data_source={s.market.live_data_source} — must be 'bridge'"
                )

    def test_fetch_multi_timeframe_redirects_yfinance_to_bridge(self, monkeypatch):
        """If someone passes source='yfinance' but live_data_source=bridge,
        it must be redirected to bridge — NOT call yfinance."""
        svc = MarketDataService(_settings())
        calls: list[tuple[str, str]] = []

        def _fake_fetch_rates(timeframe_name: str, bars: int, source: str):
            calls.append((timeframe_name, source))
            return _dummy_frame()

        monkeypatch.setattr(svc, "_fetch_rates", _fake_fetch_rates)

        svc.fetch_multi_timeframe_data(source="yfinance")

        # All calls must have been redirected to 'bridge', NOT 'yfinance'
        for tf, src in calls:
            assert src == "bridge", (
                f"TF {tf} was fetched with source={src} — should be 'bridge'"
            )

    def test_fetch_multi_timeframe_allows_bridge(self, monkeypatch):
        """Normal bridge call passes through as-is."""
        svc = MarketDataService(_settings())
        calls: list[str] = []

        def _fake_fetch_rates(timeframe_name: str, bars: int, source: str):
            calls.append(source)
            return _dummy_frame()

        monkeypatch.setattr(svc, "_fetch_rates", _fake_fetch_rates)

        svc.fetch_multi_timeframe_data(source="bridge")

        assert all(s == "bridge" for s in calls)

    def test_fetch_multi_timeframe_default_uses_bridge(self, monkeypatch):
        """When no source is passed, defaults to live_data_source config."""
        svc = MarketDataService(_settings())
        calls: list[str] = []

        def _fake_fetch_rates(timeframe_name: str, bars: int, source: str):
            calls.append(source)
            return _dummy_frame()

        monkeypatch.setattr(svc, "_fetch_rates", _fake_fetch_rates)

        svc.fetch_multi_timeframe_data()

        assert all(s == "bridge" for s in calls)

    def test_live_safe_sources_constant(self):
        """_LIVE_SAFE_SOURCES must NOT include 'yfinance'."""
        assert "yfinance" not in MarketDataService._LIVE_SAFE_SOURCES
        assert "bridge" in MarketDataService._LIVE_SAFE_SOURCES
        assert "mt5" in MarketDataService._LIVE_SAFE_SOURCES

    def test_yfinance_not_imported_at_module_level(self):
        """yfinance should NOT be imported at module level in market_data.py."""
        import xauusd_ai.data.market_data as mod
        # yf should not be a module-level name anymore
        assert not hasattr(mod, "yf"), (
            "yfinance is imported at module level in market_data.py — should be lazy"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. SelfLearner guards
# ──────────────────────────────────────────────────────────────────────────────


class TestSelfLearnerYfinanceGuard:
    """Verify SelfLearner never falls back to yfinance during live."""

    def test_fetch_fresh_data_uses_bridge(self, monkeypatch):
        """When live_data_source=bridge, fetch_fresh_data must call bridge — not yfinance."""
        from xauusd_ai.learning.self_learner import SelfLearner
        from xauusd_ai.model.trainer import ModelTrainer
        from xauusd_ai.strategies.hybrid import HybridStrategy

        settings = _settings()
        trainer = ModelTrainer(settings)
        strategy = HybridStrategy(settings)
        learner = SelfLearner(settings, trainer, strategy)

        bridge_called = []

        def _fake_bridge(timeframe):
            bridge_called.append(timeframe)
            return _dummy_frame()

        monkeypatch.setattr(learner, "fetch_fresh_bridge", _fake_bridge)

        result = learner.fetch_fresh_data("M1")

        assert len(bridge_called) == 1
        assert bridge_called[0] == "M1"
        assert not result.empty

    def test_fetch_fresh_data_no_yfinance_fallback(self, monkeypatch):
        """When live_data_source is unknown, must NOT fall back to yfinance."""
        from xauusd_ai.learning.self_learner import SelfLearner
        from xauusd_ai.model.trainer import ModelTrainer
        from xauusd_ai.strategies.hybrid import HybridStrategy

        settings = _settings()
        # Force an unusual source
        settings.market.live_data_source = "custom_feed"
        trainer = ModelTrainer(settings)
        strategy = HybridStrategy(settings)
        learner = SelfLearner(settings, trainer, strategy)

        yfinance_called = []

        def _fake_yfinance(timeframe):
            yfinance_called.append(timeframe)
            return _dummy_frame()

        monkeypatch.setattr(learner, "fetch_fresh_yfinance", _fake_yfinance)

        result = learner.fetch_fresh_data("M1")

        assert len(yfinance_called) == 0, "yfinance was called as fallback — must never happen"
        assert result.empty

    def test_yfinance_not_imported_at_module_level_self_learner(self):
        """yfinance should NOT be imported at module level in self_learner.py."""
        import xauusd_ai.learning.self_learner as mod
        assert not hasattr(mod, "yf"), (
            "yfinance is imported at module level in self_learner.py — should be lazy"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Bridge data path
# ──────────────────────────────────────────────────────────────────────────────


class TestBridgeDataPath:
    """Verify bridge produces correct real-time data for M1 scalping."""

    def test_bridge_bars_have_required_columns(self, monkeypatch):
        """Bridge bars must have all columns needed for feature engineering."""
        svc = MarketDataService(_settings())
        payload = [
            {
                "time": 1712707200,
                "open": 2300.1,
                "high": 2301.5,
                "low": 2299.9,
                "close": 2301.0,
                "tick_volume": 150,
                "spread": 25,
            },
            {
                "time": 1712707260,
                "open": 2301.0,
                "high": 2302.0,
                "low": 2300.8,
                "close": 2301.7,
                "tick_volume": 175,
                "spread": 30,
            },
        ]

        monkeypatch.setenv("MT5_BRIDGE_URL", "http://bridge.test")
        monkeypatch.setattr(svc, "_bridge_call_json", lambda _path: payload)

        frame = svc._fetch_rates_bridge("M1", 2)

        required = {"time", "open", "high", "low", "close", "tick_volume",
                     "spread_points", "tick_volume_delta", "volume_imbalance"}
        assert required.issubset(frame.columns), f"Missing: {required - set(frame.columns)}"
        assert isinstance(frame["time"].dtype, pd.DatetimeTZDtype)
        assert len(frame) == 2

    def test_bridge_bars_sorted_and_deduped(self, monkeypatch):
        """Bridge bars must be sorted by time and deduplicated."""
        svc = MarketDataService(_settings())
        # Send duplicate + out-of-order bars
        payload = [
            {"time": 1712707260, "open": 2301.0, "high": 2302.0, "low": 2300.8, "close": 2301.7, "tick_volume": 175, "spread": 30},
            {"time": 1712707200, "open": 2300.1, "high": 2301.5, "low": 2299.9, "close": 2301.0, "tick_volume": 150, "spread": 25},
            {"time": 1712707200, "open": 2300.1, "high": 2301.5, "low": 2299.9, "close": 2301.0, "tick_volume": 150, "spread": 25},
        ]

        monkeypatch.setenv("MT5_BRIDGE_URL", "http://bridge.test")
        monkeypatch.setattr(svc, "_bridge_call_json", lambda _path: payload)

        frame = svc._fetch_rates_bridge("M1", 10)

        assert len(frame) == 2, f"Expected 2 unique bars, got {len(frame)}"
        assert frame["time"].is_monotonic_increasing

    def test_bridge_raises_on_empty_response(self, monkeypatch):
        """Bridge must raise error if no data returned."""
        svc = MarketDataService(_settings())
        monkeypatch.setenv("MT5_BRIDGE_URL", "http://bridge.test")
        monkeypatch.setattr(svc, "_bridge_call_json", lambda _path: [])

        with pytest.raises(RuntimeError, match="No bridge bars"):
            svc._fetch_rates_bridge("M1", 5)

    def test_bridge_raises_on_missing_url(self, monkeypatch):
        """Bridge must raise error if MT5_BRIDGE_URL is not set."""
        svc = MarketDataService(_settings())
        monkeypatch.delenv("MT5_BRIDGE_URL", raising=False)

        with pytest.raises(RuntimeError, match="MT5_BRIDGE_URL"):
            svc._fetch_rates_bridge("M1", 5)
