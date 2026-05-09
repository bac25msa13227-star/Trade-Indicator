#!/usr/bin/env python3
"""
Example: Run WF with MT5 live data instead of CSV
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xauusd_ai.data.mt5_data_provider import MT5DataProvider
from xauusd_ai.config import load_settings

def run_wf_with_mt5_data():
    """Run walk-forward validation with MT5 live data"""
    
    # 1. Load config
    config_path = "configs/grid_search/grid_0014_th70_r5_mp4_notrail_slip15.yaml"
    settings = load_settings(config_path)
    
    # 2. Initialize MT5 data provider
    provider = MT5DataProvider()
    
    try:
        # 3. Define date range for WF
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)  # 3 years
        
        print("="*70)
        print("🔌 WALK-FORWARD with MT5 LIVE DATA")
        print("="*70)
        print(f"Symbol: XAUUSD")
        print(f"Range: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
        print()
        
        # 4. Fetch data for each timeframe
        timeframes = {
            'M1': 1,
            'M5': 5,
            'M15': 15,
            'H1': 60,
            'H4': 240,
            'D1': 1440
        }
        
        data_dict = {}
        for tf_name, tf_minutes in timeframes.items():
            print(f"[{tf_name}] Fetching data...")
            df = provider.get_data(
                symbol='XAUUSD',
                timeframe=tf_minutes,
                start_date=start_date,
                end_date=end_date,
                force_refresh=False  # Use cache if available
            )
            data_dict[tf_name] = df
            print(f"[{tf_name}] ✅ {len(df):,} bars loaded")
            print()
        
        # 5. Now you can use data_dict with your WF script
        # Instead of loading from CSV, use these DataFrames directly
        
        print("✅ All data loaded from MT5!")
        print()
        print("📊 Summary:")
        for tf_name, df in data_dict.items():
            print(f"   {tf_name}: {len(df):,} bars | {df['time'].min()} → {df['time'].max()}")
        
        # 6. TODO: Integrate with existing WF logic
        print()
        print("⚠️  Next step: Modify walkforward_ict_wyckoff.py to accept data_dict")
        print("   instead of loading from CSV files")
        
        return data_dict
        
    finally:
        # Clean up MT5 connection
        provider.cleanup()

if __name__ == '__main__':
    data = run_wf_with_mt5_data()
