"""
Scalp dataset builder — M1 base + M5 context.

Loads:
  - XAUUSDm_M1.csv (execution timeframe)
  - XAUUSDm_M5.csv (context: bias direction, RSI, ATR)

Computes:
  - All M1 scalp features (scalp_features.py)
  - M5 context features (merged_asof)
  - expected_direction using fast M1 + M5 bias voting
  - SL/TP binary label: hits TP within `max_horizon` M1 bars before SL?

Returns a DataFrame ready for dual BUY/SELL model training.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.execution.sltp import (
    base_sltp_by_regime,
    compute_dynamic_sltp,
    setup_label_horizon,
)
from xauusd_ai.features.scalp_features import build_all_scalp_features, SCALP_FEATURE_COLUMNS


REPO_ROOT = Path(__file__).parent.parent.parent.parent    # → Trade Indicator/
DATA_DIR  = REPO_ROOT / "src" / "xauusd_ai" / "real_data"

M1_CSV = DATA_DIR / "XAUUSDm_M1.csv"
M5_CSV = DATA_DIR / "XAUUSDm_M5.csv"


# ─── M1 / M5 loaders ─────────────────────────────────────────────────────────

def load_m1(start_date: str | None = None,
            end_date:   str | None = None) -> pd.DataFrame:
    """Load M1 CSV, parse timestamps, apply date filter."""
    df = pd.read_csv(M1_CSV, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    if start_date:
        df = df[df["time"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df["time"] <= pd.Timestamp(end_date, tz="UTC")]
    return df.reset_index(drop=True)


def _load_m5(start_date: str | None = None,
             end_date:   str | None = None) -> pd.DataFrame:
    """Load M5 CSV and compute M5-level context features."""
    csv = M5_CSV
    if not csv.exists():
        # Fallback: check alternate filename
        alt = DATA_DIR / "XAUUSD_M5.csv"
        if alt.exists():
            csv = alt
        else:
            return pd.DataFrame()     # caller handles missing M5

    df = pd.read_csv(csv, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC") - pd.Timedelta(days=2)
        df = df[df["time"] >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        df = df[df["time"] <= end_ts]

    return compute_m5_context_features(df)


def compute_m5_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the exact M5 context columns used by scalp training.

    This helper is shared by both dataset build and live inference so M5 context
    stays numerically identical across train/WF/live paths.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["time", "m5_bias", "m5_rsi_14", "m5_atr_norm"])

    df = frame.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    c  = df["close"]
    h  = df["high"]
    l  = df["low"]

    # M5 bias: EMA8 vs EMA21
    e8  = c.ewm(span=8,  adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    df["m5_bias"] = np.sign(e8 - e21).astype(int)

    # M5 RSI(14) centred
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi14 = (100 - (100 / (1 + gain / loss.replace(0, np.nan)))).fillna(50.0)
    df["m5_rsi_14"] = (rsi14 - 50.0)

    # M5 ATR(14) normalised
    hl  = h - l
    hpc = (h - c.shift()).abs()
    lpc = (l - c.shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().bfill()
    df["m5_atr_norm"] = (atr14 / c.replace(0, np.nan) * 1000).fillna(0)

    return df[["time", "m5_bias", "m5_rsi_14", "m5_atr_norm"]]


# ─── Expected direction (M1-based voting) ────────────────────────────────────

def _compute_direction(df: pd.DataFrame) -> pd.Series:
    """
    Direction for each M1 bar, derived from local momentum + M5 context.

    Votes (+1 or -1 each):
      m5_bias          ×2 weight (directional context)
      m1_ema8_21_cross  direction
      m1_macd_hist      direction
      m1_rsi5           direction (>0 = bullish)

    Returns: +1 (BUY setup) or -1 (SELL setup) never 0.
    """
    m5  = df.get("m5_bias", pd.Series(0, index=df.index)).fillna(0)
    ec  = np.sign(df.get("m1_ema8_21_cross", pd.Series(0, index=df.index)).fillna(0))
    mh  = np.sign(df.get("m1_macd_hist", pd.Series(0, index=df.index)).fillna(0))
    rsi = np.sign(df.get("m1_rsi5", pd.Series(0, index=df.index)).fillna(0))
    of  = np.sign(df.get("of_delta_cum5", pd.Series(0, index=df.index)).fillna(0))

    raw = m5 * 2 + ec + mh + rsi + of    # -6..+6
    direction = np.sign(raw).astype(int)

    # Break ties: fall back to M5 bias, then ema cross
    tie_mask = (direction == 0)
    direction = np.where(tie_mask, np.sign(m5 + ec), direction)
    direction = np.where(direction == 0, 1, direction)  # last resort: BUY
    return pd.Series(direction.astype(int), index=df.index)


# ─── SL/TP race label (M1-scale, short horizon) ──────────────────────────────

def _build_sltp_label(
    df: pd.DataFrame,
    sl_atr_mult: float = 0.8,
    tp_rr:       float = 1.5,
    max_horizon: int   = 8,
    atr_col:     str   = "ms_atr5_norm",   # ATR already in ‰ of price
) -> tuple[pd.Series, pd.Series]:
    """
    For each M1 bar, simulate a trade in expected_direction:
      SL = close - direction × ATR5 × sl_atr_mult
      TP = close + direction × ATR5 × sl_atr_mult × tp_rr

    Returns:
      label: 1 if TP hit before SL within max_horizon bars, else 0
      realized_rr: +tp_rr if TP hit, -1.0 if SL hit, or 0 at timeout
    """
    close = df["close"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows  = df["low"].values.astype(float)
    dirs  = df["expected_direction"].values.astype(float)

    # ATR in price units (ms_atr5_norm is in ‰ of close)
    atr5_pct = df["ms_atr5_norm"].values.astype(float)
    atr_price = atr5_pct / 1000.0 * close   # convert ‰ back to price

    n      = len(df)
    labels = np.zeros(n, dtype=np.int8)
    real_rr = np.zeros(n, dtype=np.float32)

    for i in range(n - max_horizon):
        atr = atr_price[i]
        if atr <= 0 or not np.isfinite(atr):
            continue
        d  = dirs[i]
        sl = close[i] - d * atr * sl_atr_mult
        tp = close[i] + d * atr * sl_atr_mult * tp_rr
        fh = highs[i + 1 : i + max_horizon + 1]
        fl = lows[ i + 1 : i + max_horizon + 1]
        if d > 0:
            tp_hits = np.where(fh >= tp)[0]
            sl_hits = np.where(fl <= sl)[0]
        else:
            tp_hits = np.where(fl <= tp)[0]
            sl_hits = np.where(fh >= sl)[0]
        tp_first = tp_hits[0] if len(tp_hits) > 0 else max_horizon + 1
        sl_first = sl_hits[0] if len(sl_hits) > 0 else max_horizon + 1
        if tp_first < sl_first:
            labels[i]  = 1
            real_rr[i] = float(tp_rr)
        elif sl_first < max_horizon + 1:
            real_rr[i] = -1.0
        # else: timeout → label=0, rr=0

    return pd.Series(labels, index=df.index), pd.Series(real_rr, index=df.index)


# ─── Setup-aware SL/TP arrays ─────────────────────────────────────────────────

def _build_setup_sltp_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-bar setup-aware TP_RR and SL_mult based on detected ICT/Wyckoff setup tier.

    Tiers (highest priority wins per bar):
      Tier 3 — Premium  (unicorn + BOS confirm)      → tp=2.5, sl=0.65
      Tier 2 — Advanced (Wyckoff spring/SOS or IFVG+BOS) → tp=2.0, sl=0.72
      Tier 1 — Basic    (PO3, turtle soup, OTE)       → tp=1.8, sl=0.75
      Tier 0 — Plain    (no special setup)             → tp=1.5, sl=0.80

    Returns: (tp_rr_arr, sl_mult_arr, tier_arr) each shape (n,)
    """
    n = len(df)

    def _get(col: str, default: float = 0.0) -> np.ndarray:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=float)
        return np.full(n, default, dtype=float)

    dir_   = _get("expected_direction", 1.0)
    unicorn= _get("sm_unicorn")
    bos    = _get("m1_bos")
    ifvg   = _get("sm_ifvg")
    wyck_spring = _get("wyck_spring_utad")
    wyck_sos    = _get("wyck_sos_sow")
    po3    = _get("sm_po3_bias")
    turtle = _get("sm_turtle_soup")
    ote    = _get("sm_ote_score")

    # Directional alignment: +1 if setup confirms direction
    dir_sign = np.sign(dir_)

    tier3 = (unicorn * dir_sign >= 1) & (bos * dir_sign >= 1)
    tier2 = ~tier3 & (
        (wyck_spring * dir_sign >= 1)
        | (wyck_sos   * dir_sign >= 1)
        | ((ifvg * dir_sign >= 1) & (bos * dir_sign >= 1))
    )
    tier1 = ~tier3 & ~tier2 & (
        (po3    * dir_sign >= 1)
        | (turtle * dir_sign >= 1)
        | (ote    * dir_sign >= 0.4)
    )

    tp_arr   = np.full(n, 1.5, dtype=float)
    sl_arr   = np.full(n, 0.8, dtype=float)
    tier_arr = np.zeros(n, dtype=np.int8)

    tp_arr[tier1]  = 1.8;  sl_arr[tier1]  = 0.75;  tier_arr[tier1]  = 1
    tp_arr[tier2]  = 2.0;  sl_arr[tier2]  = 0.72;  tier_arr[tier2]  = 2
    tp_arr[tier3]  = 2.5;  sl_arr[tier3]  = 0.65;  tier_arr[tier3]  = 3

    return tp_arr, sl_arr, tier_arr


def _build_sltp_label_setup_aware(
    df: pd.DataFrame,
    tp_rr_arr:    np.ndarray,
    sl_mult_arr:  np.ndarray,
    max_horizon:  int = 8,
) -> tuple[pd.Series, pd.Series]:
    """
    Per-bar SL/TP label using setup-specific TP_RR and SL_mult for each bar.
    Same race logic as _build_sltp_label but vectorised per setup tier.

    Returns:
      label:       1 if setup-TP hit before SL within max_horizon bars, else 0
      realized_rr: +tp_rr[i] if TP hit, -1.0 if SL hit, 0 at timeout
    """
    close    = df["close"].values.astype(float)
    highs    = df["high"].values.astype(float)
    lows     = df["low"].values.astype(float)
    dirs     = df["expected_direction"].values.astype(float)
    atr5_pct = df["ms_atr5_norm"].values.astype(float)
    atr_price = atr5_pct / 1000.0 * close

    n        = len(df)
    labels   = np.zeros(n, dtype=np.int8)
    real_rr  = np.zeros(n, dtype=np.float32)

    for i in range(n - max_horizon):
        atr = atr_price[i]
        if atr <= 0 or not np.isfinite(atr):
            continue
        d      = dirs[i]
        sl_m   = float(sl_mult_arr[i])
        tp_r   = float(tp_rr_arr[i])
        sl     = close[i] - d * atr * sl_m
        tp     = close[i] + d * atr * sl_m * tp_r
        fh = highs[i + 1 : i + max_horizon + 1]
        fl = lows[ i + 1 : i + max_horizon + 1]
        if d > 0:
            tp_hits = np.where(fh >= tp)[0]
            sl_hits = np.where(fl <= sl)[0]
        else:
            tp_hits = np.where(fl <= tp)[0]
            sl_hits = np.where(fh >= sl)[0]
        tp_first = tp_hits[0] if len(tp_hits) > 0 else max_horizon + 1
        sl_first = sl_hits[0] if len(sl_hits) > 0 else max_horizon + 1
        if tp_first < sl_first:
            labels[i]  = 1
            real_rr[i] = tp_r
        elif sl_first < max_horizon + 1:
            real_rr[i] = -1.0

    return pd.Series(labels, index=df.index), pd.Series(real_rr, index=df.index)


def _scalp_confidence_proxy(df: pd.DataFrame, settings: Settings) -> np.ndarray:
    """Build a pseudo-confidence [min_conf..0.95] from scalp quality features."""
    min_conf = float(settings.risk.min_confidence)
    max_conf = 0.95
    if max_conf <= min_conf:
        max_conf = min_conf + 1e-3

    comps: list[np.ndarray] = []
    if "of_flow_score" in df.columns:
        comps.append(np.clip(np.abs(pd.to_numeric(df["of_flow_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)), 0.0, 1.0))
    if "m1_bos" in df.columns:
        comps.append(np.clip(np.abs(pd.to_numeric(df["m1_bos"], errors="coerce").fillna(0.0).to_numpy(dtype=float)), 0.0, 1.0))
    if "m1_macd_hist" in df.columns:
        comps.append(np.clip(np.abs(np.tanh(pd.to_numeric(df["m1_macd_hist"], errors="coerce").fillna(0.0).to_numpy(dtype=float))), 0.0, 1.0))
    if "ms_wick_net" in df.columns:
        comps.append(np.clip(np.abs(pd.to_numeric(df["ms_wick_net"], errors="coerce").fillna(0.0).to_numpy(dtype=float)), 0.0, 1.0))
    if "ms_body_ratio" in df.columns:
        comps.append(np.clip(pd.to_numeric(df["ms_body_ratio"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0))
    if "m5_rsi_14" in df.columns:
        comps.append(np.clip(np.abs(pd.to_numeric(df["m5_rsi_14"], errors="coerce").fillna(0.0).to_numpy(dtype=float)) / 50.0, 0.0, 1.0))

    if not comps:
        proxy = np.full(len(df), 0.5, dtype=float)
    else:
        proxy = np.mean(np.vstack(comps), axis=0)

    return np.clip(min_conf + (max_conf - min_conf) * proxy, min_conf, max_conf)


def _scalp_regime_proxy(df: pd.DataFrame) -> np.ndarray:
    """
    Infer scalp volatility regime from M1 volatility expansion.
      0 sideway / 1 normal / 2 strong
    """
    return infer_scalp_volatility_regime(df).to_numpy(dtype=int)


def infer_scalp_volatility_regime(df: pd.DataFrame) -> pd.Series:
    """
    Shared scalp volatility regime inference for training and live runtime.

    Priority:
      1. M1 ATR expansion (same as label generation)
      2. M5 ATR norm fallback
    """
    if "ms_atr_expansion" in df.columns:
        exp = pd.to_numeric(df["ms_atr_expansion"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        regime = np.where(exp <= 0.95, 0, np.where(exp >= 1.30, 2, 1))
        return pd.Series(regime.astype(int), index=df.index, dtype=int)
    if "m5_atr_norm" in df.columns:
        exp = pd.to_numeric(df["m5_atr_norm"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        regime = np.where(exp <= 0.95, 0, np.where(exp >= 1.25, 2, 1))
        return pd.Series(regime.astype(int), index=df.index, dtype=int)
    return pd.Series(np.ones(len(df), dtype=int), index=df.index, dtype=int)


def apply_dynamic_sltp_labels(
    df: pd.DataFrame,
    settings: Settings,
    max_horizon: int | None = None,
) -> pd.DataFrame:
    """
    Recompute target/realized_rr/bars_held using hybrid dynamic SL/TP rules.
    Used for retraining so model learns the same SL/TP policy as live execution.
    """
    out = df.copy()
    horizon = int(max_horizon or settings.training.sltp_label_max_horizon or 8)
    horizon = max(1, horizon)

    close = pd.to_numeric(out["close"], errors="coerce").ffill().to_numpy(dtype=float)
    high = pd.to_numeric(out["high"], errors="coerce").ffill().to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").ffill().to_numpy(dtype=float)
    dirs = pd.to_numeric(out["expected_direction"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    if "atr" in out.columns:
        atr = pd.to_numeric(out["atr"], errors="coerce").to_numpy(dtype=float)
    elif "ms_atr5_norm" in out.columns:
        atr = close * pd.to_numeric(out["ms_atr5_norm"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 1000.0
    else:
        atr = np.full(len(out), np.nan, dtype=float)

    conf_proxy = _scalp_confidence_proxy(out, settings)
    regimes = _scalp_regime_proxy(out)

    labels = np.zeros(len(out), dtype=np.int8)
    rr_out = np.zeros(len(out), dtype=np.float32)
    bars_out = np.full(len(out), horizon, dtype=np.int32)

    min_stop_points = max(0.0, float(getattr(settings.risk, "min_stop_loss_points", 0.0) or 0.0))
    min_stop_atr_mult = max(0.0, float(getattr(settings.risk, "min_stop_loss_atr_multiple", 0.0) or 0.0))

    for i in range(len(out) - 1):
        direction = dirs[i]
        if direction == 0:
            continue
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue
        side = "buy" if direction > 0 else "sell"

        base_sl, base_rr = base_sltp_by_regime(settings, int(regimes[i]))
        row_payload = {
            "trade_side": side,
            "expected_direction": float(direction),
            "strategy_score": float(out["strategy_score"].iloc[i]) if "strategy_score" in out.columns else 0.0,
            "strategy_setup_score": float(out["strategy_setup_score"].iloc[i]) if "strategy_setup_score" in out.columns else 0.0,
            "of_flow_score": float(out["of_flow_score"].iloc[i]) if "of_flow_score" in out.columns else 0.0,
            "m1_bos": float(out["m1_bos"].iloc[i]) if "m1_bos" in out.columns else 0.0,
            "m1_macd_hist": float(out["m1_macd_hist"].iloc[i]) if "m1_macd_hist" in out.columns else 0.0,
            "ms_close_in_rng": float(out["ms_close_in_rng"].iloc[i]) if "ms_close_in_rng" in out.columns else 0.5,
            "execution_quality": float(out["execution_quality"].iloc[i]) if "execution_quality" in out.columns else (float(out["of_flow_score"].iloc[i]) if "of_flow_score" in out.columns else 0.0),
            "trend_strength_score": float(out["trend_strength_score"].iloc[i]) if "trend_strength_score" in out.columns else (float(out["m1_bos"].iloc[i]) if "m1_bos" in out.columns else 0.0),
            "pullback_quality": float(out["pullback_quality"].iloc[i]) if "pullback_quality" in out.columns else (float(out["ms_wick_net"].iloc[i]) if "ms_wick_net" in out.columns else 0.0),
            "sm_turtle_soup": float(out["sm_turtle_soup"].iloc[i]) if "sm_turtle_soup" in out.columns else 0.0,
            "sm_ote_score": float(out["sm_ote_score"].iloc[i]) if "sm_ote_score" in out.columns else 0.0,
            "sm_ifvg": float(out["sm_ifvg"].iloc[i]) if "sm_ifvg" in out.columns else 0.0,
            "sm_unicorn": float(out["sm_unicorn"].iloc[i]) if "sm_unicorn" in out.columns else 0.0,
            "sm_po3_bias": float(out["sm_po3_bias"].iloc[i]) if "sm_po3_bias" in out.columns else 0.0,
            "wyck_spring_utad": float(out["wyck_spring_utad"].iloc[i]) if "wyck_spring_utad" in out.columns else 0.0,
            "wyck_sos_sow": float(out["wyck_sos_sow"].iloc[i]) if "wyck_sos_sow" in out.columns else 0.0,
            "wyck_lps_quality": float(out["wyck_lps_quality"].iloc[i]) if "wyck_lps_quality" in out.columns else 0.0,
        }
        sl_mult, tp_rr, _, _ = compute_dynamic_sltp(
            settings=settings,
            row=row_payload,
            confidence=float(conf_proxy[i]),
            base_sl_mult=float(base_sl),
            base_tp_rr=float(base_rr),
        )

        sl_dist = max(0.0, atr_i * sl_mult)
        if sl_dist < min_stop_points:
            sl_dist = min_stop_points
        if min_stop_atr_mult > 0:
            sl_dist = max(sl_dist, atr_i * min_stop_atr_mult)
        if sl_dist <= 0:
            continue

        entry = close[i]
        if side == "buy":
            sl_level = entry - sl_dist
            tp_level = entry + sl_dist * tp_rr
        else:
            sl_level = entry + sl_dist
            tp_level = entry - sl_dist * tp_rr

        horizon_i = setup_label_horizon(horizon, row_payload)
        end_i = min(len(out) - 1, i + horizon_i)
        hit_rr = 0.0
        hit_bar = horizon_i
        hit_tp = False
        hit_sl = False
        for j in range(i + 1, end_i + 1):
            hi = high[j]
            lo = low[j]
            if side == "buy":
                sl_hit = lo <= sl_level
                tp_hit = hi >= tp_level
            else:
                sl_hit = hi >= sl_level
                tp_hit = lo <= tp_level

            # Conservative tie-break on same bar: SL first.
            if sl_hit:
                hit_sl = True
                hit_rr = -1.0
                hit_bar = j - i
                break
            if tp_hit:
                hit_tp = True
                hit_rr = float(tp_rr)
                hit_bar = j - i
                break

        if hit_tp:
            labels[i] = 1
            rr_out[i] = hit_rr
            bars_out[i] = hit_bar
            continue
        if hit_sl:
            rr_out[i] = hit_rr
            bars_out[i] = hit_bar
            continue

        # Timeout: partial RR from horizon close
        horizon_close = close[end_i]
        price_change = (horizon_close - entry) * direction
        rr_partial = price_change / sl_dist
        rr_out[i] = float(np.clip(rr_partial, -1.0, float(tp_rr)))
        bars_out[i] = max(1, end_i - i)

    out["target"] = labels
    out["realized_rr"] = rr_out
    out["bars_held"] = bars_out
    return out


# ─── Main builder ────────────────────────────────────────────────────────────

def build_scalp_dataset(
    start_date:       str        = "2023-01-01",
    end_date:         str | None = None,
    sl_atr_mult:      float      = 0.8,
    tp_rr:            float      = 1.5,
    max_horizon:      int        = 8,
    setup_label_mode: str        = "fixed",  # "fixed" | "setup_aware"
) -> pd.DataFrame:
    """
    Full scalp dataset pipeline:
      1. Load M1 + M5
      2. Build all M1 features
      3. Merge M5 context
      4. Compute expected_direction
      5. SL/TP label
      6. Drop warm-up rows (first 250 M1 bars for indicator stability)

    Returns DataFrame with columns: time, OHLCV, all scalp features,
      expected_direction, target, realized_rr
    """
    print(f"  [ScalpDS] Loading M1 data ({start_date} → {end_date or 'latest'})…", flush=True)
    m1 = load_m1(start_date=start_date, end_date=end_date)
    print(f"  [ScalpDS] {len(m1):,} M1 bars", flush=True)

    print("  [ScalpDS] Building M1 features…", flush=True)
    m1 = build_all_scalp_features(m1)

    # ── M5 context ────────────────────────────────────────────────────
    print("  [ScalpDS] Loading M5 context…", flush=True)
    m5 = _load_m5(start_date=start_date, end_date=end_date)
    if not m5.empty:
        m1 = pd.merge_asof(
            m1.sort_values("time"),
            m5.sort_values("time"),
            on="time",
            direction="backward",
        )
        m1["m5_bias"]    = m1["m5_bias"].fillna(0).astype(int)
        m1["m5_rsi_14"]  = m1["m5_rsi_14"].fillna(0)
        m1["m5_atr_norm"] = m1["m5_atr_norm"].fillna(m1["ms_atr5_norm"])
        print(f"  [ScalpDS] M5 context merged: {(~m1['m5_bias'].isna()).sum():,} rows", flush=True)
    else:
        m1["m5_bias"]     = 0
        m1["m5_rsi_14"]   = 0.0
        m1["m5_atr_norm"] = m1["ms_atr5_norm"]
        print("  [ScalpDS] M5 not available, using zeros", flush=True)

    # ── Expected direction ────────────────────────────────────────────
    m1["expected_direction"] = _compute_direction(m1)

    # ── SL/TP label ───────────────────────────────────────────────────
    if setup_label_mode == "setup_aware":
        print(f"  [ScalpDS] Building setup-aware SL/TP labels (horizon={max_horizon})…", flush=True)
        tp_arr, sl_arr, tier_arr = _build_setup_sltp_arrays(m1)
        m1["setup_tp_rr"]   = tp_arr
        m1["setup_sl_mult"] = sl_arr
        m1["setup_tier"]    = tier_arr
        tier_counts = {t: int((tier_arr == t).sum()) for t in range(4)}
        print(f"  [ScalpDS] Setup tiers: plain={tier_counts[0]:,}  "
              f"basic={tier_counts[1]:,}  adv={tier_counts[2]:,}  "
              f"premium={tier_counts[3]:,}", flush=True)
        labels, rr = _build_sltp_label_setup_aware(
            m1, tp_rr_arr=tp_arr, sl_mult_arr=sl_arr, max_horizon=max_horizon
        )
    else:
        print(f"  [ScalpDS] Building SL/TP label (horizon={max_horizon}, sl×{sl_atr_mult}ATR, tp={tp_rr}R)…", flush=True)
        labels, rr = _build_sltp_label(m1, sl_atr_mult=sl_atr_mult, tp_rr=tp_rr, max_horizon=max_horizon)
        m1["setup_tp_rr"]   = float(tp_rr)
        m1["setup_sl_mult"] = float(sl_atr_mult)
        m1["setup_tier"]    = 0

    m1["target"]       = labels.values
    m1["realized_rr"]  = rr.values

    # ── Drop warm-up rows ─────────────────────────────────────────────
    m1 = m1.iloc[250:].reset_index(drop=True)

    tp_rate = m1["target"].mean()
    t3 = int((m1.get("setup_tier", pd.Series(0)) == 3).sum())
    t2 = int((m1.get("setup_tier", pd.Series(0)) == 2).sum())
    t1 = int((m1.get("setup_tier", pd.Series(0)) == 1).sum())
    print(f"  [ScalpDS] Final: {len(m1):,} rows  "
          f"TP-rate={tp_rate*100:.1f}%  "
          f"BUY={( m1['expected_direction']>0).mean()*100:.0f}%  "
          f"SELL={(m1['expected_direction']<0).mean()*100:.0f}%", flush=True)
    if setup_label_mode == "setup_aware":
        print(f"  [ScalpDS] After warmup — tier1={t1:,}  tier2={t2:,}  tier3={t3:,}", flush=True)

    # Validate all feature columns present
    missing = [c for c in SCALP_FEATURE_COLUMNS if c not in m1.columns]
    if missing:
        print(f"  [ScalpDS] WARNING: missing columns: {missing}", flush=True)

    return m1
