from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_ai.config import load_settings
from xauusd_ai.features.dataset import _build_live_scalp_feature_frame
from xauusd_ai.features.scalp_dataset import (
    compute_m5_context_features,
    infer_scalp_volatility_regime,
)
from xauusd_ai.features.scalp_features import build_all_scalp_features


def _settings():
    return load_settings(Path("configs/live_acc2_scalp_m1.yaml"))


def _make_m1_frame() -> pd.DataFrame:
    times = pd.date_range("2026-04-10 06:00", periods=180, freq="min", tz="UTC")
    base = 3200.0 + np.linspace(0.0, 7.5, len(times)) + np.sin(np.arange(len(times)) / 9.0) * 0.8
    open_ = base + np.sin(np.arange(len(times)) / 5.0) * 0.15
    close = base + np.cos(np.arange(len(times)) / 7.0) * 0.18
    high = np.maximum(open_, close) + 0.35 + (np.arange(len(times)) % 4) * 0.03
    low = np.minimum(open_, close) - 0.35 - (np.arange(len(times)) % 3) * 0.02
    tick_volume = 90 + (np.arange(len(times)) % 17) * 4
    tick_volume_delta = np.diff(np.r_[tick_volume[0], tick_volume]).astype(float)
    volume_imbalance = np.clip((np.abs(close - open_) / np.maximum(high - low, 1e-6)), 0.0, 1.0)

    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": tick_volume.astype(float),
            "tick_volume_delta": tick_volume_delta,
            "volume_imbalance": volume_imbalance.astype(float),
        }
    )


def _make_m5_frame(m1: pd.DataFrame) -> pd.DataFrame:
    return (
        m1.set_index("time")
        .resample("5min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "tick_volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )


def test_live_scalp_frame_matches_training_m5_context_and_regime() -> None:
    settings = _settings()
    m1 = _make_m1_frame()
    m5 = _make_m5_frame(m1)

    train_like = build_all_scalp_features(m1.copy())
    m5_ctx = compute_m5_context_features(m5)
    train_like = pd.merge_asof(
        train_like.sort_values("time"),
        m5_ctx.sort_values("time"),
        on="time",
        direction="backward",
    )
    train_like["m5_bias"] = pd.to_numeric(train_like["m5_bias"], errors="coerce").fillna(0).astype(int)
    train_like["m5_rsi_14"] = pd.to_numeric(train_like["m5_rsi_14"], errors="coerce").fillna(0.0)
    train_like["m5_atr_norm"] = pd.to_numeric(train_like["m5_atr_norm"], errors="coerce").fillna(train_like["ms_atr5_norm"])
    train_like["volatility_regime"] = infer_scalp_volatility_regime(train_like).astype(int)

    live_frame = _build_live_scalp_feature_frame(settings, {"M1": m1, "M5": m5})

    assert np.array_equal(live_frame["m5_bias"].to_numpy(), train_like["m5_bias"].to_numpy())
    assert np.allclose(live_frame["m5_rsi_14"].to_numpy(), train_like["m5_rsi_14"].to_numpy())
    assert np.allclose(live_frame["m5_atr_norm"].to_numpy(), train_like["m5_atr_norm"].to_numpy())
    assert np.array_equal(
        live_frame["volatility_regime"].astype(int).to_numpy(),
        train_like["volatility_regime"].astype(int).to_numpy(),
    )


def test_live_scalp_frame_uses_training_fallback_without_m5() -> None:
    settings = _settings()
    m1 = _make_m1_frame()

    live_frame = _build_live_scalp_feature_frame(settings, {"M1": m1})

    assert np.array_equal(live_frame["m5_bias"].astype(int).to_numpy(), np.zeros(len(live_frame), dtype=int))
    assert np.allclose(live_frame["m5_rsi_14"].to_numpy(), 0.0)
    assert np.allclose(live_frame["m5_atr_norm"].to_numpy(), live_frame["ms_atr5_norm"].to_numpy())
    assert np.array_equal(
        live_frame["volatility_regime"].astype(int).to_numpy(),
        infer_scalp_volatility_regime(live_frame).astype(int).to_numpy(),
    )
