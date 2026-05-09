#!/usr/bin/env python3
"""
MT5 Data Provider - Hybrid approach with cache + live data
"""
import os
import pandas as pd
import MetaTrader5 as mt5
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

class MT5DataProvider:
    """Provide historical data from MT5 with intelligent caching"""
    
    def __init__(self, cache_dir: str = "outputs/.data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mt5_initialized = False
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if self.mt5_initialized:
            return True
        
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False
        
        print(f"✅ MT5 initialized: {mt5.version()}")
        self.mt5_initialized = True
        return True
    
    def fetch_from_mt5(
        self, 
        symbol: str, 
        timeframe: int,
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data from MT5"""
        if not self.initialize_mt5():
            return None
        
        print(f"📊 Fetching {symbol} {timeframe} from MT5...")
        print(f"   Range: {start_date} → {end_date}")
        
        # Map timeframe
        tf_map = {
            1: mt5.TIMEFRAME_M1,
            5: mt5.TIMEFRAME_M5,
            15: mt5.TIMEFRAME_M15,
            60: mt5.TIMEFRAME_H1,
            240: mt5.TIMEFRAME_H4,
            1440: mt5.TIMEFRAME_D1
        }
        
        mt5_tf = tf_map.get(timeframe)
        if mt5_tf is None:
            print(f"❌ Unsupported timeframe: {timeframe}")
            return None
        
        # Fetch data
        rates = mt5.copy_rates_range(symbol, mt5_tf, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            print(f"❌ No data received: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        print(f"✅ Fetched {len(df):,} bars from MT5")
        return df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']]
    
    def get_cache_path(self, symbol: str, timeframe: int, start_date: datetime, end_date: datetime) -> Path:
        """Generate cache file path"""
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        return self.cache_dir / f"{symbol}_{timeframe}min_{start_str}_{end_str}.parquet"
    
    def load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        if not cache_path.exists():
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            print(f"✅ Loaded {len(df):,} bars from cache")
            return df
        except Exception as e:
            print(f"⚠️  Cache load failed: {e}")
            return None
    
    def save_to_cache(self, df: pd.DataFrame, cache_path: Path):
        """Save data to cache"""
        try:
            df.to_parquet(cache_path, compression='snappy')
            print(f"✅ Saved {len(df):,} bars to cache")
        except Exception as e:
            print(f"⚠️  Cache save failed: {e}")
    
    def get_data(
        self,
        symbol: str,
        timeframe: int,
        start_date: datetime,
        end_date: datetime,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get historical data with intelligent caching
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            timeframe: Timeframe in minutes (1, 5, 15, 60, 240, 1440)
            start_date: Start date
            end_date: End date
            force_refresh: Force fetch from MT5 even if cache exists
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_path = self.get_cache_path(symbol, timeframe, start_date, end_date)
        
        # Try load from cache first
        if not force_refresh:
            cached_df = self.load_from_cache(cache_path)
            
            if cached_df is not None:
                # Check if cache is up-to-date
                cache_end = cached_df['time'].max()
                now = pd.Timestamp.now()
                
                # If cache is older than 1 day, fetch new data
                if cache_end < now - pd.Timedelta(days=1):
                    print("⚠️  Cache outdated, fetching new data...")
                    new_start = cache_end + pd.Timedelta(minutes=timeframe)
                    new_data = self.fetch_from_mt5(symbol, timeframe, new_start, end_date)
                    
                    if new_data is not None and len(new_data) > 0:
                        # Merge old + new
                        df = pd.concat([cached_df, new_data]).drop_duplicates(subset='time')
                        df = df.sort_values('time').reset_index(drop=True)
                        self.save_to_cache(df, cache_path)
                        return df
                
                # Cache is fresh, use it
                return cached_df
        
        # Fetch from MT5
        df = self.fetch_from_mt5(symbol, timeframe, start_date, end_date)
        
        if df is not None:
            self.save_to_cache(df, cache_path)
            return df
        
        # Fallback to cache if MT5 fetch failed
        cached_df = self.load_from_cache(cache_path)
        if cached_df is not None:
            print("⚠️  Using cached data (MT5 fetch failed)")
            return cached_df
        
        raise Exception("Failed to fetch data from MT5 and no cache available")
    
    def cleanup(self):
        """Clean up MT5 connection"""
        if self.mt5_initialized:
            mt5.shutdown()
            self.mt5_initialized = False


# Example usage
if __name__ == '__main__':
    provider = MT5DataProvider()
    
    # Fetch XAUUSD M1 data for last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    df = provider.get_data(
        symbol='XAUUSD',
        timeframe=1,  # M1
        start_date=start_date,
        end_date=end_date,
        force_refresh=False  # Use cache if available
    )
    
    print(f"\n📊 Data Summary:")
    print(f"   Rows: {len(df):,}")
    print(f"   Date range: {df['time'].min()} → {df['time'].max()}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n{df.head()}")
    
    provider.cleanup()
