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

    c  = df["close"]
    h  = df["high"]
    l  = df["low"]
    tv = df["tick_volume"]

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


# ─── Main builder ────────────────────────────────────────────────────────────

def build_scalp_dataset(
    start_date:  str        = "2023-01-01",
    end_date:    str | None = None,
    sl_atr_mult: float      = 0.8,
    tp_rr:       float      = 1.5,
    max_horizon: int        = 8,
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
    print(f"  [ScalpDS] Building SL/TP label (horizon={max_horizon}, sl×{sl_atr_mult}ATR, tp={tp_rr}R)…", flush=True)
    labels, rr = _build_sltp_label(m1, sl_atr_mult=sl_atr_mult, tp_rr=tp_rr, max_horizon=max_horizon)
    m1["target"]       = labels.values
    m1["realized_rr"]  = rr.values

    # ── Drop warm-up rows ─────────────────────────────────────────────
    m1 = m1.iloc[250:].reset_index(drop=True)

    tp_rate = m1["target"].mean()
    print(f"  [ScalpDS] Final: {len(m1):,} rows  "
          f"TP-rate={tp_rate*100:.1f}%  "
          f"BUY={( m1['expected_direction']>0).mean()*100:.0f}%  "
          f"SELL={(m1['expected_direction']<0).mean()*100:.0f}%", flush=True)

    # Validate all feature columns present
    missing = [c for c in SCALP_FEATURE_COLUMNS if c not in m1.columns]
    if missing:
        print(f"  [ScalpDS] WARNING: missing columns: {missing}", flush=True)

    return m1
