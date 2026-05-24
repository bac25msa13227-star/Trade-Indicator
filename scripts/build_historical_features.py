#!/usr/bin/env python3
"""
Build historical feature dataset from MT5 for 2022-2026.
Fetches M5, H1, H4, D1 OHLCV from MT5, computes all features,
saves to outputs/historical_features_2022_2026.csv

Usage:
  python scripts/build_historical_features.py --from 2022-01-01 --to 2026-05-09

Requirements:
  MT5 terminal must be running and logged in.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False

from xauusd_ai.config import load_settings
from xauusd_ai.features.dataset import _merge_context

CONFIG = ROOT / "configs" / "grid_search" / "grid_0014_th70_r5_mp4_notrail_slip15.yaml"
OUT_CSV = ROOT / "outputs" / "historical_features_2022_2026.csv"

TF_MAP = {
    "M5":  5,
    "M15": 15,
    "H1":  60,
    "H4":  240,
    "D1":  1440,
}

MT5_TF_MAP = {
    "M5":  mt5.TIMEFRAME_M5  if HAS_MT5 else 5,
    "M15": mt5.TIMEFRAME_M15 if HAS_MT5 else 15,
    "H1":  mt5.TIMEFRAME_H1  if HAS_MT5 else 60,
    "H4":  mt5.TIMEFRAME_H4  if HAS_MT5 else 240,
    "D1":  mt5.TIMEFRAME_D1  if HAS_MT5 else 1440,
}


def fetch_ohlcv(symbol: str, tf_name: str, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    """Fetch OHLCV bars from MT5 for given symbol and timeframe.
    Retries up to 10 times with 3s delay to handle async download."""
    import time
    tf_code = MT5_TF_MAP[tf_name]
    # MT5 downloads historical data asynchronously.
    # First request triggers download; subsequent requests return the cached data.
    rates = None
    for attempt in range(10):
        rates = mt5.copy_rates_range(symbol, tf_code, date_from, date_to)
        n = len(rates) if rates is not None else 0
        if n > 10:  # got meaningful data
            break
        if attempt == 0:
            # First attempt: also trigger loading via copy_rates_from_pos
            mt5.copy_rates_from_pos(symbol, tf_code, 0, 5000)
        print(f"  {tf_name} attempt {attempt+1}: {n} bars, waiting 3s for download...")
        time.sleep(3)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        print(f"  WARNING: {tf_name} fetch failed: {err}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "tick_volume"})

    # Standardize columns
    base_cols = ["time", "open", "high", "low", "close", "tick_volume"]
    df = df[[c for c in base_cols if c in df.columns]].copy()

    # Add derived columns used by feature pipeline
    df["spread_points"] = 0.0
    df["tick_volume_delta"] = df["tick_volume"].diff().fillna(0)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["volume_imbalance"] = ((df["close"] - df["open"]).abs() / rng).fillna(0).clip(0, 1)

    df = df.sort_values("time").reset_index(drop=True)
    print(f"  {tf_name}: {len(df):,} bars  {df['time'].min()} → {df['time'].max()}")
    return df


def build_features(settings, frames: dict) -> pd.DataFrame:
    """Compute all model features from multi-TF OHLCV frames."""
    print("  Computing features...")
    df = _merge_context(settings, frames)
    df = df.dropna(subset=["close", "atr"]).reset_index(drop=True)
    print(f"  Feature dataset: {len(df):,} rows")
    return df


def load_from_csv(csv_dir: Path, tf_name: str, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    """Load OHLCV data from pre-exported CSV files (e.g. XAUUSDm_M5.csv)."""
    # Find matching file: XAUUSDm_M5.csv or XAUUSD_M5.csv
    candidates = list(csv_dir.glob(f"*_{tf_name}.csv"))
    if not candidates:
        print(f"  WARNING: No CSV file found for {tf_name} in {csv_dir}")
        return pd.DataFrame()
    csv_path = candidates[0]
    print(f"  Reading {csv_path.name} ...", end=" ", flush=True)
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Filter by date range
    if date_from:
        from_utc = pd.Timestamp(date_from).tz_localize("UTC") if date_from.tzinfo is None else pd.Timestamp(date_from)
        df = df[df["time"] >= from_utc]
    if date_to:
        to_utc = pd.Timestamp(date_to).tz_localize("UTC") if date_to.tzinfo is None else pd.Timestamp(date_to)
        df = df[df["time"] < to_utc]

    df = df.sort_values("time").reset_index(drop=True)
    print(f"{len(df):,} bars  {df['time'].min()} to {df['time'].max()}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="date_from", default="2022-01-01")
    p.add_argument("--to",   dest="date_to",   default="2026-05-09")
    p.add_argument("--symbol", default="XAUUSDm")
    p.add_argument("--csv-dir", dest="csv_dir", default=None,
                   help="Directory with pre-exported CSV files (e.g. XAUUSDm_M5.csv). "
                        "If provided, skips MT5 connection.")
    p.add_argument("--out", default=str(OUT_CSV))
    args = p.parse_args()

    print("=" * 70)
    print("  Build Historical Features")
    mode = f"CSV: {args.csv_dir}" if args.csv_dir else "MT5"
    print(f"  Mode   : {mode}")
    print(f"  Period : {args.date_from} to {args.date_to}")
    print(f"  Symbol : {args.symbol}")
    print("=" * 70)

    settings = load_settings(CONFIG)

    date_from = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    date_to   = datetime.strptime(args.date_to,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    tf_names = [
        settings.market.execution_timeframe,  # M5
        settings.market.mid_timeframe,         # H1
        settings.market.structure_timeframe,   # H4
        settings.market.higher_timeframe,      # D1
    ]
    tf_names = list(dict.fromkeys(tf_names))  # deduplicate

    frames = {}

    if args.csv_dir:
        # ── Load from CSV files ─────────────────────────────────────────
        csv_dir = Path(args.csv_dir)
        print("\n[2/4] Loading OHLCV data from CSV files...")
        for tf in tf_names:
            df = load_from_csv(csv_dir, tf, date_from, date_to)
            if df.empty:
                print(f"  ERROR: No data for {tf}, cannot proceed.")
                sys.exit(1)
            frames[tf] = df
    else:
        # ── Fetch from MT5 ──────────────────────────────────────────────
        if not HAS_MT5:
            print("ERROR: MetaTrader5 package not found. Run: pip install MetaTrader5")
            sys.exit(1)

        print("\n[1/4] Connecting to MT5...")
        if not mt5.initialize():
            print(f"MT5 init failed: {mt5.last_error()}")
            print("Make sure MT5 is running and logged in.")
            sys.exit(1)

        info = mt5.terminal_info()
        print(f"  Connected: {info.name}  build={info.build}")

        symbol = args.symbol
        if not mt5.symbol_select(symbol, True):
            symbol = symbol.rstrip("m")
            if not mt5.symbol_select(symbol, True):
                print(f"  Cannot select symbol {symbol}")
                mt5.shutdown()
                sys.exit(1)
        print(f"  Symbol selected: {symbol}")

        print("\n[2/4] Fetching OHLCV data from MT5...")
        for tf in tf_names:
            df = fetch_ohlcv(symbol, tf, date_from, date_to)
            if df.empty:
                print(f"  ERROR: No data for {tf}, cannot proceed.")
                mt5.shutdown()
                sys.exit(1)
            frames[tf] = df

        mt5.shutdown()
        print("  MT5 disconnected.")

    # ── Compute features ───────────────────────────────────────────────
    print("\n[3/4] Computing features (this may take 2-5 minutes)...")
    try:
        df = build_features(settings, frames)
    except Exception as e:
        print(f"  ERROR during feature computation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Save dataset ───────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Add trade_side column (direction from expected_direction)
    if "expected_direction" not in df.columns:
        # Simple EMA bias if expected_direction not computed
        ema_f = df["close"].ewm(span=50, adjust=False).mean()
        ema_s = df["close"].ewm(span=200, adjust=False).mean()
        df["expected_direction"] = np.sign(ema_f - ema_s).replace(0, 1)

    df["trade_side"] = df["expected_direction"].map({1: "buy", -1: "sell"}).fillna("buy")

    # Keep ATR (needed for SL/TP in export script)
    if "atr" not in df.columns:
        print("  WARNING: ATR column missing")

    print(f"\n[4/4] Saving → {out_path}")
    df.to_csv(out_path, index=False)
    rows = len(df)
    print(f"  Saved {rows:,} rows")

    # Coverage summary
    print()
    print("  Coverage by year:")
    df["year"] = df["time"].dt.year
    for yr, cnt in df.groupby("year").size().items():
        print(f"    {yr}: {cnt:,} bars")

    print()
    print(f"  Dataset saved: {out_path}")
    print("  Next: python scripts/export_signals_mt5.py --dataset outputs/historical_features_2022_2026.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
