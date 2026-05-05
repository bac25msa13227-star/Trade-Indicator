from __future__ import annotations

from unittest.mock import patch

from xauusd_ai.config import Settings
from xauusd_ai.execution import mt5_executor as mt5_mod
from xauusd_ai.execution.mt5_executor import MT5Executor


def test_get_market_state_uses_bridge_when_available() -> None:
    settings = Settings()
    payload = {
        "symbol": "XAUUSD",
        "is_open": False,
        "reason": "OUTSIDE_SESSION",
        "next_open_utc": "2026-04-06T00:05:00+00:00",
        "minutes_to_next_open": 25,
    }
    with patch.object(mt5_mod, "_BRIDGE_URL", "http://bridge.local"), patch.object(
        mt5_mod, "_bridge_call", return_value=payload
    ):
        ex = MT5Executor(settings)
        state = ex.get_market_state("XAUUSD", stale_seconds=300)
    assert state["reason"] == "OUTSIDE_SESSION"
    assert state["is_open"] is False
    assert state["minutes_to_next_open"] == 25


def test_get_market_state_bridge_error_is_safe_closed() -> None:
    settings = Settings()
    with patch.object(mt5_mod, "_BRIDGE_URL", "http://bridge.local"), patch.object(
        mt5_mod, "_bridge_call", side_effect=RuntimeError("bridge down")
    ):
        ex = MT5Executor(settings)
        state = ex.get_market_state("XAUUSD", stale_seconds=120)
    assert state["is_open"] is False
    assert str(state["reason"]).startswith("BRIDGE_ERROR:")


def test_get_market_state_unavailable_without_mt5_or_bridge() -> None:
    settings = Settings()
    with patch.object(mt5_mod, "_BRIDGE_URL", ""), patch.object(MT5Executor, "_ensure_connection", return_value=None):
        ex = MT5Executor(settings)
        state = ex.get_market_state("XAUUSD", stale_seconds=120)
    assert state["is_open"] is False
    assert state["reason"] == "MT5_UNAVAILABLE"
