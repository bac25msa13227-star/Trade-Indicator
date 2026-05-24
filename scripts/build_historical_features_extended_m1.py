#!/usr/bin/env python3
"""
Build M1 (1-minute) scalp feature dataset from 2003-2026.
Same features as M5 but on 1-minute timeframe for faster signals.

Usage:
  python scripts/build_historical_features_extended_m1.py

Output:
  outputs/historical_features_2003_2026_M1.csv
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 not installed. pip install MetaTrader5")
    sys.exit(1)

from xauusd_ai.features.dataset import _merge_context

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

OUT_CSV = ROOT / "outputs" / "historical_features_2003_2026_M1.csv"

MT5_TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
}

def fetch_ohlcv(symbol: str, tf_name: str, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    """Fetch OHLCV from MT5."""
    import time
    tf_code = MT5_TF_MAP[tf_name]
    rates = None
    for attempt in range(10):
        rates = mt5.copy_rates_range(symbol, tf_code, date_from, date_to)
        n = len(rates) if rates is not None else 0
        if n > 100:
            break
        if attempt == 0:
            mt5.copy_rates_from_pos(symbol, tf_code, 0, 5000)
        log.info(f"  {tf_name} attempt {attempt+1}: {n} bars, waiting...")
        time.sleep(3)
    
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        log.warning(f"  {tf_name} fetch failed: {err}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["time", "open", "high", "low", "close", "tick_volume", "spread"]].rename(
        columns={"tick_volume": "volume", "spread": "spread_points"}
    )

def compute_m1_features(df_m1: pd.DataFrame, df_m5, df_m15, df_h1) -> pd.DataFrame:
    """Compute features for M1 bars."""
    from xauusd_ai.features import ict_wyckoff
    
    result = df_m1.copy()
    
    # M1 features
    result["returns"] = result["close"].pct_change()
    result["rsi"] = ict_wyckoff._rsi(result["close"], 14)
    result["macd_hist"] = ict_wyckoff._macd_histogram(result["close"], 12, 26, 9)[2]
    result["atr"] = ict_wyckoff._atr(result["high"], result["low"], result["close"], 14)
    result["atr_mean"] = result["atr"].rolling(100).mean()
    result["atr_ratio"] = result["atr"] / result["atr_mean"]
    
    # Merge higher TF context
    result = _merge_context(result, df_m5, df_m15, df_h1)
    
    return result.dropna()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", default="2003-01-01")
    parser.add_argument("--to", dest="date_to", default="2026-05-10")
    args = parser.parse_args()
    
    date_from = datetime.fromisoformat(args.date_from).replace(tzinfo=timezone.utc)
    date_to = datetime.fromisoformat(args.date_to).replace(tzinfo=timezone.utc)
    
    log.info(f"Building M1 features: {date_from.date()} to {date_to.date()}")
    
    if not mt5.initialize():
        log.error("MT5 init failed")
        sys.exit(1)
    
    # Fetch all timeframes
    log.info("Fetching M1 data...")
    df_m1 = fetch_ohlcv("XAUUSD", "M1", date_from, date_to)
    log.info(f"  M1: {len(df_m1)} bars ({len(df_m1)/1440:.0f} days)")
    
    log.info("Fetching context TFs...")
    df_m5 = fetch_ohlcv("XAUUSD", "M5", date_from, date_to)
    df_m15 = fetch_ohlcv("XAUUSD", "M15", date_from, date_to)
    df_h1 = fetch_ohlcv("XAUUSD", "H1", date_from, date_to)
    
    # Compute
    log.info("Computing M1 features...")
    df = compute_m1_features(df_m1, df_m5, df_m15, df_h1)
    log.info(f"  Result: {len(df)} rows, {df.shape[1]} columns")
    
    # Save
    log.info(f"Saving to {OUT_CSV}")
    df.to_csv(OUT_CSV, index=False)
    log.info("Done!")
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
