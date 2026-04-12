from __future__ import annotations

from pathlib import Path

import pandas as pd

from xauusd_ai.config import load_settings
from xauusd_ai.model.scalp_runtime import adjust_scalp_thresholds


def _settings():
    settings = load_settings(
        Path("/Users/dodoannang/Documents/Thạc sĩ MSE/Trade Indicator/configs/live_acc1_scalp_m1.yaml")
    )
    settings.strategy.scalp_dynamic_threshold_enabled = True
    settings.strategy.scalp_threshold_offset_strong_regime = -0.04
    settings.strategy.scalp_threshold_offset_sideway_regime = 0.02
    settings.strategy.scalp_threshold_offset_off_session = 0.03
    return settings


def test_adjust_scalp_thresholds_lowers_threshold_in_strong_regime():
    settings = _settings()
    frame = pd.DataFrame(
        {
            "time": ["2026-03-20T13:00:00Z"],
            "volatility_regime": [2],
        }
    )
    adjusted, delta = adjust_scalp_thresholds(settings, frame, [0.62])
    assert round(float(adjusted.iloc[0]), 4) == 0.58
    assert round(float(delta.iloc[0]), 4) == -0.04


def test_adjust_scalp_thresholds_adds_sideway_and_off_session_offsets():
    settings = _settings()
    frame = pd.DataFrame(
        {
            "time": ["2026-03-20T22:00:00Z"],
            "volatility_regime": [0],
        }
    )
    adjusted, delta = adjust_scalp_thresholds(settings, frame, [0.60])
    assert round(float(adjusted.iloc[0]), 4) == 0.65
    assert round(float(delta.iloc[0]), 4) == 0.05


def test_adjust_scalp_thresholds_noop_when_disabled():
    settings = _settings()
    settings.strategy.scalp_dynamic_threshold_enabled = False
    frame = pd.DataFrame(
        {
            "time": ["2026-03-20T13:00:00Z"],
            "volatility_regime": [2],
        }
    )
    adjusted, delta = adjust_scalp_thresholds(settings, frame, [0.62])
    assert float(adjusted.iloc[0]) == 0.62
    assert float(delta.iloc[0]) == 0.0
