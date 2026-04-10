from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xauusd_ai.config import Settings


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_value(row: object | None, field: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(field, default)
    if isinstance(row, pd.Series):
        return row.get(field, default)
    return getattr(row, field, default)


def _infer_side_sign(row: object | None) -> float:
    trade_side = str(_row_value(row, "trade_side", "") or "").strip().lower()
    if trade_side == "buy":
        return 1.0
    if trade_side == "sell":
        return -1.0

    exp_dir = _as_float(_row_value(row, "expected_direction", None))
    if exp_dir is not None and abs(exp_dir) > 0:
        return float(np.sign(exp_dir))

    for field in ("strategy_setup_score", "strategy_score", "trend_strength_score", "execution_quality", "pullback_quality"):
        v = _as_float(_row_value(row, field, None))
        if v is not None and abs(v) > 1e-9:
            return float(np.sign(v))
    return 0.0


def base_sltp_by_regime(settings: Settings, volatility_regime: int) -> tuple[float, float]:
    """Return (sl_atr_multiple, tp_rr) for regime before dynamic adjustment."""
    r = settings.risk
    if int(volatility_regime) == 0:
        return float(r.sideway_sl_atr_multiple), float(r.sideway_take_profit_rr)
    if int(volatility_regime) == 2:
        return float(r.volatile_sl_atr_multiple), float(r.volatile_take_profit_rr)
    return float(r.stop_loss_atr_multiple), float(r.take_profit_rr)


def _quality_from_row(row: object | None) -> float:
    """
    Build a normalized quality score in [-1, 1] from available row features.
    Works for both standard MTF and scalp-M1 feature frames.
    """
    values: list[float] = []
    side_sign = _infer_side_sign(row)

    def _align(v: float) -> float:
        if side_sign == 0:
            return float(np.clip(abs(v), 0.0, 1.0))
        return float(np.clip(v * side_sign, -1.0, 1.0))

    # Standard pipeline meta-features
    for name in ("execution_quality", "trend_strength_score", "pullback_quality", "strategy_setup_score"):
        v = _as_float(_row_value(row, name, None))
        if v is not None:
            values.append(_align(v))

    # Scalp pipeline proxies
    _flow = _as_float(_row_value(row, "of_flow_score", None))
    if _flow is not None:
        values.append(_align(_flow))
    _bos = _as_float(_row_value(row, "m1_bos", None))
    if _bos is not None:
        values.append(_align(_bos))
    _macd = _as_float(_row_value(row, "m1_macd_hist", _row_value(row, "macd_hist", None)))
    if _macd is not None:
        values.append(_align(float(np.tanh(_macd))))
    _close_in_rng = _as_float(_row_value(row, "ms_close_in_rng", _row_value(row, "close_in_range", None)))
    if _close_in_rng is not None:
        values.append(float(np.clip(abs((_close_in_rng - 0.5) * 2.0), 0.0, 1.0)))

    if not values:
        return 0.0
    return float(np.mean(values))


def _setup_profile_from_row(row: object | None) -> dict[str, float | str]:
    side_sign = _infer_side_sign(row)

    def _aligned(field: str) -> float:
        v = _as_float(_row_value(row, field, 0.0)) or 0.0
        if side_sign == 0:
            return float(np.clip(abs(v), 0.0, 1.0))
        return float(np.clip(v * side_sign, -1.0, 1.0))

    continuation_parts = [
        max(0.0, _aligned("sm_unicorn")),
        max(0.0, _aligned("sm_ifvg")),
        max(0.0, _aligned("sm_po3_bias")),
        max(0.0, _aligned("wyck_sos_sow")),
        max(0.0, _aligned("sm_ote_score")),
    ]
    reversal_parts = [
        max(0.0, _aligned("sm_turtle_soup")),
        max(0.0, _aligned("wyck_spring_utad")),
        max(0.0, _aligned("wyck_lps_quality")),
    ]
    continuation = float(np.mean(continuation_parts)) if continuation_parts else 0.0
    reversal = float(np.mean(reversal_parts)) if reversal_parts else 0.0
    setup_bias = continuation - reversal

    if continuation >= reversal + 0.10 and continuation >= 0.12:
        setup_type = "continuation"
    elif reversal >= continuation + 0.10 and reversal >= 0.12:
        setup_type = "reversal"
    elif max(continuation, reversal) >= 0.12:
        setup_type = "hybrid"
    else:
        setup_type = "neutral"

    return {
        "setup_type": setup_type,
        "continuation_strength": continuation,
        "reversal_strength": reversal,
        "setup_bias": setup_bias,
        "side_sign": side_sign,
    }


def _setup_sltp_adjustment(row: object | None) -> tuple[float, float, list[str], dict[str, float | str]]:
    """
    Setup-aware overlay on top of quality/confidence tuning.
    Reversal setups get a bit more stop room and slightly closer targets.
    Continuation setups get tighter stops and farther targets.
    """
    profile = _setup_profile_from_row(row)
    setup_type = str(profile["setup_type"])
    cont = float(profile["continuation_strength"])
    rev = float(profile["reversal_strength"])

    sl_ratio = 1.0
    tp_ratio = 1.0
    tags: list[str] = []

    if setup_type == "continuation":
        sl_ratio *= 1.0 - min(0.12, 0.10 * cont + 0.02)
        tp_ratio *= 1.0 + min(0.22, 0.18 * cont + 0.04)
        tags.append(f"setupCONT{cont:.2f}")
    elif setup_type == "reversal":
        sl_ratio *= 1.0 + min(0.18, 0.12 * rev + 0.03)
        tp_ratio *= 1.0 - min(0.10, 0.06 * rev + 0.02)
        tags.append(f"setupREV{rev:.2f}")
    elif setup_type == "hybrid":
        strength = max(cont, rev)
        sl_ratio *= 1.0 + 0.03 * rev - 0.04 * cont
        tp_ratio *= 1.0 + min(0.10, 0.08 * strength)
        tags.append(f"setupHYB{strength:.2f}")

    profile["sl_ratio"] = float(sl_ratio)
    profile["tp_ratio"] = float(tp_ratio)
    return float(sl_ratio), float(tp_ratio), tags, profile


def resolve_setup_exit_targets(
    row: object | None,
    *,
    enabled: bool = False,
    tp_scale: float = 1.0,
) -> tuple[float, float, int, list[str], dict[str, float | int]]:
    """
    Return setup-tier exit overrides derived from ICT/Wyckoff setup signals.

    Output:
      tp_rr, sl_mult, tier, tags, metrics

    Tier mapping:
      0 plain    -> tp=1.5R, sl=0.80
      1 basic    -> tp=1.8R, sl=0.75
      2 advanced -> tp=2.0R, sl=0.72
      3 premium  -> tp=2.5R, sl=0.65

    tp_scale only scales TP. SL tier table is preserved.
    """
    if not enabled:
        return 0.0, 0.0, 0, [], {"setup_tier": 0, "setup_exit_enabled": 0}

    side_sign = _infer_side_sign(row)

    def _aligned(field: str) -> float:
        v = _as_float(_row_value(row, field, 0.0)) or 0.0
        if side_sign == 0:
            return float(np.clip(abs(v), 0.0, 1.0))
        return float(np.clip(v * side_sign, -1.0, 1.0))

    bos = max(0.0, _aligned("m1_bos"))
    unicorn = max(0.0, _aligned("sm_unicorn"))
    ifvg = max(0.0, _aligned("sm_ifvg"))
    spring = max(0.0, _aligned("wyck_spring_utad"))
    sos = max(0.0, _aligned("wyck_sos_sow"))
    po3 = max(0.0, _aligned("sm_po3_bias"))
    turtle = max(0.0, _aligned("sm_turtle_soup"))
    ote = max(0.0, _aligned("sm_ote_score"))

    tier = 0
    if unicorn >= 1.0 and bos >= 1.0:
        tier = 3
    elif spring >= 1.0 or sos >= 1.0 or (ifvg >= 1.0 and bos >= 1.0):
        tier = 2
    elif po3 >= 1.0 or turtle >= 1.0 or ote >= 0.4:
        tier = 1

    base_table = {
        0: (1.5, 0.80),
        1: (1.8, 0.75),
        2: (2.0, 0.72),
        3: (2.5, 0.65),
    }
    base_tp_rr, sl_mult = base_table[tier]
    tp_rr = float(base_tp_rr * max(0.05, float(tp_scale)))
    tags = [f"setupT{tier}"]
    if abs(tp_scale - 1.0) > 1e-9:
        tags.append(f"setupTPx{tp_scale:.2f}")

    metrics: dict[str, float | int] = {
        "setup_tier": int(tier),
        "setup_exit_enabled": 1,
        "setup_exit_tp_rr": float(tp_rr),
        "setup_exit_sl_mult": float(sl_mult),
        "setup_exit_tp_scale": float(tp_scale),
    }
    return tp_rr, float(sl_mult), int(tier), tags, metrics


def setup_label_horizon(base_horizon: int, row: object | None) -> int:
    """
    Setup-aware labeling horizon:
      continuation setups get more room to develop,
      reversal setups get a modest extension,
      weak/neutral setups keep or slightly shorten horizon.
    """
    base_horizon = max(1, int(base_horizon))
    profile = _setup_profile_from_row(row)
    setup_type = str(profile["setup_type"])
    cont = float(profile["continuation_strength"])
    rev = float(profile["reversal_strength"])

    mult = 1.0
    if setup_type == "continuation":
        mult = 1.0 + min(0.60, 0.30 * cont + 0.10)
    elif setup_type == "reversal":
        mult = 1.0 + min(0.35, 0.18 * rev + 0.05)
    elif setup_type == "neutral":
        mult = 0.90

    lo = max(2, int(round(base_horizon * 0.75)))
    hi = max(lo, int(round(base_horizon * 2.0)))
    return int(np.clip(round(base_horizon * mult), lo, hi))


def compute_dynamic_sltp(
    settings: Settings,
    row: object | None,
    confidence: float,
    base_sl_mult: float,
    base_tp_rr: float,
) -> tuple[float, float, list[str], dict[str, float]]:
    """
    Compute dynamic SL/TP multipliers from model confidence + market quality.

    Returns:
      (sl_mult, tp_rr, tags, debug_metrics)
    """
    base_sl_mult = max(1e-6, float(base_sl_mult))
    base_tp_rr = max(1e-6, float(base_tp_rr))

    if not bool(getattr(settings.risk, "dynamic_sltp_enabled", False)):
        return base_sl_mult, base_tp_rr, [], {
            "combined_score": 0.5,
            "confidence_norm": 0.5,
            "quality_norm": 0.5,
        }

    conf_floor = max(1e-6, float(settings.risk.min_confidence))
    conf_cap = 0.95
    if conf_cap <= conf_floor:
        conf_cap = conf_floor + 1e-3
    conf_norm = (float(confidence) - conf_floor) / (conf_cap - conf_floor)
    conf_norm = float(np.clip(conf_norm, 0.0, 1.0))

    quality_raw = _quality_from_row(row)  # [-1, 1]
    quality_norm = float(np.clip((quality_raw + 1.0) * 0.5, 0.0, 1.0))

    w_conf = max(0.0, float(getattr(settings.risk, "dynamic_sltp_confidence_weight", 0.7)))
    w_quality = max(0.0, float(getattr(settings.risk, "dynamic_sltp_quality_weight", 0.3)))
    w_total = w_conf + w_quality
    combined = conf_norm if w_total <= 1e-9 else (conf_norm * w_conf + quality_norm * w_quality) / w_total
    combined = float(np.clip(combined, 0.0, 1.0))

    # combined>0.5 => stronger setup: can target farther TP, use tighter SL.
    # combined<0.5 => weaker setup: bring TP closer, widen SL.
    sl_bias = (0.5 - combined) * 2.0
    tp_bias = (combined - 0.5) * 2.0

    sl_scale = max(0.0, float(getattr(settings.risk, "dynamic_sltp_sl_scale", 0.25)))
    tp_scale = max(0.0, float(getattr(settings.risk, "dynamic_sltp_tp_scale", 0.40)))

    sl_ratio = 1.0 + sl_scale * sl_bias
    tp_ratio = 1.0 + tp_scale * tp_bias

    sl_ratio_min = max(0.1, float(getattr(settings.risk, "dynamic_sltp_sl_mult_min", 0.85)))
    sl_ratio_max = max(sl_ratio_min, float(getattr(settings.risk, "dynamic_sltp_sl_mult_max", 1.60)))
    tp_rr_min = max(0.1, float(getattr(settings.risk, "dynamic_sltp_tp_rr_min", 1.0)))
    tp_rr_max = max(tp_rr_min, float(getattr(settings.risk, "dynamic_sltp_tp_rr_max", 4.0)))

    setup_sl_ratio, setup_tp_ratio, setup_tags, setup_metrics = _setup_sltp_adjustment(row)
    sl_mult = float(np.clip(base_sl_mult * sl_ratio * setup_sl_ratio, base_sl_mult * sl_ratio_min, base_sl_mult * sl_ratio_max))
    tp_rr = float(np.clip(base_tp_rr * tp_ratio * setup_tp_ratio, tp_rr_min, tp_rr_max))

    tags: list[str] = []
    if bool(getattr(settings.risk, "dynamic_sltp_debug_in_reason", True)):
        tags.append(f"dynC{combined:.2f}")
        tags.append(f"dynSL×{(sl_mult / base_sl_mult):.2f}")
        tags.append(f"dynRR{tp_rr:.2f}")
        tags.extend(setup_tags)

    return sl_mult, tp_rr, tags, {
        "combined_score": combined,
        "confidence_norm": conf_norm,
        "quality_norm": quality_norm,
        "quality_raw": quality_raw,
        **setup_metrics,
    }
