from __future__ import annotations

from pathlib import Path

import pandas as pd

from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService


def _settings():
    return load_settings(Path("configs/live_acc2_scalp_m1.yaml"))


def test_fetch_rates_bridge_normalizes_payload(monkeypatch):
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
            "real_volume": 0,
        },
        {
            "time": 1712707260,
            "open": 2301.0,
            "high": 2302.0,
            "low": 2300.8,
            "close": 2301.7,
            "tick_volume": 175,
            "spread": 30,
            "real_volume": 0,
        },
    ]

    monkeypatch.setenv("MT5_BRIDGE_URL", "http://bridge.test")
    monkeypatch.setattr(svc, "_bridge_call_json", lambda _path: payload)

    frame = svc._fetch_rates_bridge("M1", 2)

    assert list(frame["spread_points"]) == [25, 30]
    assert list(frame["tick_volume_delta"]) == [0.0, 25.0]
    assert isinstance(frame["time"].dtype, pd.DatetimeTZDtype)
    assert {"open", "high", "low", "close", "tick_volume", "volume_imbalance"}.issubset(frame.columns)


def test_fetch_multi_timeframe_data_adds_m5_for_m1_live(monkeypatch):
    svc = MarketDataService(_settings())

    calls: list[str] = []

    def _fake_fetch_rates(timeframe_name: str, bars: int, source: str):
        calls.append(timeframe_name)
        return pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-04-10T00:00:00Z"]),
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "tick_volume": [0.0],
                "spread_points": [0.0],
                "tick_volume_delta": [0.0],
                "volume_imbalance": [0.0],
            }
        )

    monkeypatch.setattr(svc, "_fetch_rates", _fake_fetch_rates)

    frames = svc.fetch_multi_timeframe_data(source="bridge")

    assert {"D1", "H4", "H1", "M30", "M15", "M5", "M1"}.issubset(frames.keys())
    assert set(calls) == {"D1", "H1", "H4", "M30", "M15", "M5", "M1"}
