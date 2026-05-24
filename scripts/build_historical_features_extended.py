#!/usr/bin/env python3
"""
Build extended historical features from 2003-2026 for longer training.
Pulls from MT5 (which should have all historical data).
Requires: MT5 terminal running with XAUUSD history loaded.

Usage:
  python scripts/build_historical_features_extended.py

Output:
  outputs/historical_features_2003_2026.csv (1M+ rows, 20+ years)
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone
import logging

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    print("ERROR: MetaTrader5 not installed. Install via: pip install MetaTrader5")
    sys.exit(1)

from xauusd_ai.config import load_settings
from xauusd_ai.features.dataset import _merge_context

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

OUT_CSV = ROOT / "outputs" / "historical_features_2003_2026.csv"

MT5_TF_MAP = {
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

def fetch_ohlcv(symbol: str, tf_name: str, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    """Fetch OHLCV from MT5 with retry logic."""
    import time
    tf_code = MT5_TF_MAP[tf_name]
    rates = None
    for attempt in range(10):
        rates = mt5.copy_rates_range(symbol, tf_code, date_from, date_to)
        n = len(rates) if rates is not None else 0
        if n > 100:  # got data
            break
        if attempt == 0:
            mt5.copy_rates_from_pos(symbol, tf_code, 0, 5000)  # trigger load
        log.info(f"  {tf_name} attempt {attempt+1}: {n} bars, waiting 3s...")
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

def compute_features(df_m5: pd.DataFrame, df_m15, df_h1, df_h4, df_d1) -> pd.DataFrame:
    """Compute all technical features."""
    from xauusd_ai.features import ict_wyckoff
    
    result = df_m5.copy()
    
    # M5 features
    result["returns"] = result["close"].pct_change()
    result["rsi"] = ict_wyckoff._rsi(result["close"], 14)
    result["macd_hist"] = ict_wyckoff._macd_histogram(result["close"], 12, 26, 9)[2]
    result["atr"] = ict_wyckoff._atr(result["high"], result["low"], result["close"], 14)
    result["atr_mean"] = result["atr"].rolling(100).mean()
    result["atr_ratio"] = result["atr"] / result["atr_mean"]
    
    # Merge context from higher timeframes
    result = _merge_context(result, df_m15, df_h1, df_h4, df_d1)
    
    return result.dropna()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", default="2003-01-01")
    parser.add_argument("--to", dest="date_to", default="2026-05-10")
    args = parser.parse_args()
    
    date_from = datetime.fromisoformat(args.date_from).replace(tzinfo=timezone.utc)
    date_to = datetime.fromisoformat(args.date_to).replace(tzinfo=timezone.utc)
    
    log.info(f"Building extended features: {date_from.date()} to {date_to.date()}")
    
    if not mt5.initialize():
        log.error("MT5 init failed")
        sys.exit(1)
    
    # Fetch all timeframes
    log.info("Fetching M5 data...")
    df_m5 = fetch_ohlcv("XAUUSD", "M5", date_from, date_to)
    log.info(f"  M5: {len(df_m5)} bars")
    
    log.info("Fetching M15 data...")
    df_m15 = fetch_ohlcv("XAUUSD", "M15", date_from, date_to)
    log.info("Fetching H1 data...")
    df_h1 = fetch_ohlcv("XAUUSD", "H1", date_from, date_to)
    log.info("Fetching H4 data...")
    df_h4 = fetch_ohlcv("XAUUSD", "H4", date_from, date_to)
    log.info("Fetching D1 data...")
    df_d1 = fetch_ohlcv("XAUUSD", "D1", date_from, date_to)
    
    # Compute features
    log.info("Computing features...")
    df = compute_features(df_m5, df_m15, df_h1, df_h4, df_d1)
    log.info(f"  Result: {len(df)} rows, {df.shape[1]} columns")
    
    # Save
    log.info(f"Saving to {OUT_CSV}")
    df.to_csv(OUT_CSV, index=False)
    log.info("Done!")
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
