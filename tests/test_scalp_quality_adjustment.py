from __future__ import annotations

import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.model.scalp_runtime import adjust_scalp_probabilities, compute_scalp_quality_score


def _make_settings(enabled: bool, weight: float = 0.10) -> Settings:
    settings = Settings()
    settings.strategy.scalp_quality_adjustment_enabled = enabled
    settings.strategy.scalp_quality_adjustment_weight = weight
    settings.strategy.scalp_quality_adjustment_max_delta = 0.12
    return settings


def test_quality_score_rewards_aligned_buy_setup() -> None:
    frame = pd.DataFrame(
        {
            "trade_side": ["buy"],
            "strategy_setup_score": [0.8],
            "trend_strength_score": [0.7],
            "pullback_quality": [0.5],
            "execution_quality": [0.6],
            "m1_bos": [1.0],
            "of_flow_score": [0.4],
            "trend_alignment": [1],
            "volatility_regime": [1],
        }
    )
    score = compute_scalp_quality_score(frame).iloc[0]
    assert score > 0.35


def test_quality_score_penalizes_misaligned_sell_setup() -> None:
    frame = pd.DataFrame(
        {
            "trade_side": ["sell"],
            "strategy_setup_score": [0.7],
            "trend_strength_score": [0.5],
            "pullback_quality": [0.4],
            "execution_quality": [0.3],
            "m1_bos": [1.0],
            "of_flow_score": [0.5],
            "trend_alignment": [0],
            "volatility_regime": [0],
        }
    )
    score = compute_scalp_quality_score(frame).iloc[0]
    assert score < -0.20


def test_probability_adjustment_increases_good_signal() -> None:
    frame = pd.DataFrame(
        {
            "trade_side": ["buy"],
            "strategy_setup_score": [0.9],
            "trend_strength_score": [0.8],
            "pullback_quality": [0.6],
            "execution_quality": [0.7],
            "m1_bos": [1.0],
            "of_flow_score": [0.5],
            "trend_alignment": [1],
            "volatility_regime": [2],
        }
    )
    adjusted, quality, delta = adjust_scalp_probabilities(_make_settings(True, 0.10), frame, [0.61])
    assert adjusted.iloc[0] > 0.61
    assert quality.iloc[0] > 0
    assert delta.iloc[0] > 0


def test_probability_adjustment_decreases_bad_signal() -> None:
    frame = pd.DataFrame(
        {
            "trade_side": ["buy"],
            "strategy_setup_score": [-0.8],
            "trend_strength_score": [-0.7],
            "pullback_quality": [-0.5],
            "execution_quality": [-0.6],
            "m1_bos": [-1.0],
            "of_flow_score": [-0.5],
            "trend_alignment": [0],
            "volatility_regime": [0],
        }
    )
    adjusted, quality, delta = adjust_scalp_probabilities(_make_settings(True, 0.10), frame, [0.61])
    assert adjusted.iloc[0] < 0.61
    assert quality.iloc[0] < 0
    assert delta.iloc[0] < 0


def test_probability_adjustment_is_noop_when_disabled() -> None:
    frame = pd.DataFrame({"trade_side": ["buy"], "strategy_setup_score": [0.9]})
    adjusted, quality, delta = adjust_scalp_probabilities(_make_settings(False), frame, [0.61])
    assert adjusted.iloc[0] == 0.61
    assert quality.iloc[0] == 0.0
    assert delta.iloc[0] == 0.0
