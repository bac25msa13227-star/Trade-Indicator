"""Test MT5 copy_rates_from vs copy_rates_range."""
import MetaTrader5 as mt5
from datetime import datetime
import time
import pandas as pd

mt5.initialize()
sym = "XAUUSD"
mt5.symbol_select(sym, True)

# Try copy_rates_from (starting date + count)
print("Test copy_rates_from:")
r = mt5.copy_rates_from(sym, mt5.TIMEFRAME_M5, datetime(2024, 10, 1), 500)
n = len(r) if r is not None else 0
print(f"  Oct 2024 from: {n} bars, error={mt5.last_error()}")

# Retry after wait
time.sleep(5)
r2 = mt5.copy_rates_from(sym, mt5.TIMEFRAME_M5, datetime(2024, 10, 1), 500)
n2 = len(r2) if r2 is not None else 0
print(f"  Oct 2024 retry: {n2} bars")

if r2 is not None and n2 > 0:
    df = pd.DataFrame(r2)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    print("  First bar:", df.iloc[0]["time"])
    print("  Last bar: ", df.iloc[-1]["time"])

# Test range
r3 = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M5, datetime(2024, 10, 1), datetime(2024, 11, 1))
n3 = len(r3) if r3 is not None else 0
print(f"  Oct range: {n3} bars")

# Test 2022
r4 = mt5.copy_rates_from(sym, mt5.TIMEFRAME_M5, datetime(2022, 1, 1), 500)
n4 = len(r4) if r4 is not None else 0
print(f"  2022 from: {n4} bars, error={mt5.last_error()}")

mt5.shutdown()
print("Done.")
