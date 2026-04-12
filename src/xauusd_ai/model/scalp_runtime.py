from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_ai.config import Settings


def _series_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def compute_scalp_quality_score(frame: pd.DataFrame, side_col: str = "trade_side") -> pd.Series:
    """
    Direction-aware quality score for scalp trades.

    Positive values mean the current row is well aligned for its intended side
    (buy/sell). Negative values mean structure/quality are fighting the signal.
    """
    if frame.empty:
        return pd.Series(dtype=float)

    sides = frame.get(side_col)
    if sides is None:
        exp_dir = _series_or_default(frame, "expected_direction", 1.0)
        side_sign = np.where(exp_dir < 0, -1.0, 1.0)
    else:
        side_sign = np.where(
            pd.Series(sides, index=frame.index).astype(str).str.lower().eq("sell"),
            -1.0,
            1.0,
        )

    def _dir_quality(column: str, default: float = 0.0) -> np.ndarray:
        values = _series_or_default(frame, column, default).clip(-1.0, 1.0).to_numpy(dtype=float)
        return np.clip(values * side_sign, -1.0, 1.0)

    setup_q = _dir_quality("strategy_setup_score", 0.0)
    trend_q = _dir_quality("trend_strength_score", 0.0)
    pull_q = _dir_quality("pullback_quality", 0.0)
    exec_q = _dir_quality("execution_quality", 0.0)
    bos_q = _dir_quality("m1_bos", 0.0)
    flow_q = _dir_quality("of_flow_score", 0.0)

    trend_alignment = _series_or_default(frame, "trend_alignment", 1.0).to_numpy(dtype=float)
    align_q = np.where(trend_alignment == 1, 1.0, -0.6)

    regime = _series_or_default(frame, "volatility_regime", 1.0).astype(int).to_numpy()
    regime_q = np.select(
        [regime == 0, regime == 2],
        [-0.25, 0.08],
        default=0.0,
    )

    quality = (
        0.30 * setup_q
        + 0.20 * trend_q
        + 0.15 * pull_q
        + 0.15 * exec_q
        + 0.10 * bos_q
        + 0.10 * flow_q
        + 0.10 * align_q
        + regime_q
    )
    return pd.Series(np.clip(quality, -1.0, 1.0), index=frame.index, dtype=float)


def scalp_session_bucket(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or "time" not in frame.columns:
        return pd.Series("off", index=frame.index, dtype=object)
    ts = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    hours = ts.dt.hour.fillna(-1).astype(int)
    session = np.select(
        [
            hours.between(0, 5),
            hours.between(6, 11),
            hours.between(12, 17),
        ],
        ["asia", "london", "ny"],
        default="off",
    )
    return pd.Series(session, index=frame.index, dtype=object)


def adjust_scalp_thresholds(
    settings: Settings,
    frame: pd.DataFrame,
    base_thresholds: pd.Series | np.ndarray | list[float],
) -> tuple[pd.Series, pd.Series]:
    """
    Return effective scalp threshold and additive delta.

    This keeps ACC2/backward-compatible behavior unchanged unless explicitly
    enabled in config. ACC1 can use it to lower the bar in profitable market
    regimes (for example strong volatility) without retraining the model.
    """
    base = pd.Series(np.asarray(base_thresholds, dtype=float), index=frame.index, dtype=float)
    enabled = bool(getattr(settings.strategy, "scalp_dynamic_threshold_enabled", False))
    if not enabled or frame.empty:
        zero = pd.Series(0.0, index=frame.index, dtype=float)
        return base.clip(0.01, 0.99), zero

    regime = _series_or_default(frame, "volatility_regime", 1.0).astype(int).to_numpy()
    strong_offset = float(getattr(settings.strategy, "scalp_threshold_offset_strong_regime", 0.0) or 0.0)
    sideway_offset = float(getattr(settings.strategy, "scalp_threshold_offset_sideway_regime", 0.0) or 0.0)
    off_session_offset = float(getattr(settings.strategy, "scalp_threshold_offset_off_session", 0.0) or 0.0)

    session = scalp_session_bucket(frame).astype(str)
    delta = np.where(regime == 2, strong_offset, 0.0)
    delta = delta + np.where(regime == 0, sideway_offset, 0.0)
    delta = delta + np.where(session.eq("off"), off_session_offset, 0.0)

    delta_series = pd.Series(delta, index=frame.index, dtype=float)
    adjusted = (base + delta_series).clip(0.01, 0.99)
    return adjusted, delta_series


def adjust_scalp_probabilities(
    settings: Settings,
    frame: pd.DataFrame,
    raw_probabilities: pd.Series | np.ndarray | list[float],
    side_col: str = "trade_side",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Return adjusted probability, quality score, and additive delta.

    The adjustment is intentionally small and bounded. It should improve runtime
    calibration without replacing the underlying model.
    """
    raw = pd.Series(np.asarray(raw_probabilities, dtype=float), index=frame.index, dtype=float)
    enabled = bool(getattr(settings.strategy, "scalp_quality_adjustment_enabled", False))
    weight = float(getattr(settings.strategy, "scalp_quality_adjustment_weight", 0.0) or 0.0)
    max_delta = max(0.0, float(getattr(settings.strategy, "scalp_quality_adjustment_max_delta", 0.12) or 0.0))

    if not enabled or weight <= 0.0 or frame.empty:
        zero = pd.Series(0.0, index=frame.index, dtype=float)
        return raw.clip(0.01, 0.99), zero, zero

    quality = compute_scalp_quality_score(frame, side_col=side_col)
    delta = (quality * weight).clip(-max_delta, max_delta)
    adjusted = (raw + delta).clip(0.01, 0.99)
    return adjusted, quality, delta
