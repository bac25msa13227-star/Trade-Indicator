"""Check MT5 historical data availability - how far back does M5 data go."""
import MetaTrader5 as mt5
from datetime import datetime
import time
import pandas as pd

mt5.initialize()
sym = "XAUUSD"
mt5.symbol_select(sym, True)

print("Triggering async download from 2020...")
for attempt in range(5):
    r = mt5.copy_rates_from(sym, mt5.TIMEFRAME_M5, datetime(2020, 1, 1), 5000)
    n = len(r) if r is not None else 0
    print(f"  Attempt {attempt+1}: {n} bars, err={mt5.last_error()}")
    if n > 100:
        df = pd.DataFrame(r)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        print(f"  Range: {df['time'].min()} to {df['time'].max()}")
        break
    time.sleep(5)

# Also try H1 for 2022
r2 = mt5.copy_rates_from(sym, mt5.TIMEFRAME_H1, datetime(2022, 1, 1), 1000)
n2 = len(r2) if r2 is not None else 0
print(f"H1 from 2022: {n2} bars")
if r2 is not None and n2 > 0:
    df2 = pd.DataFrame(r2)
    df2["time"] = pd.to_datetime(df2["time"], unit="s")
    print(f"  Range: {df2['time'].min()} to {df2['time'].max()}")

# Try D1 for 2020
r3 = mt5.copy_rates_from(sym, mt5.TIMEFRAME_D1, datetime(2020, 1, 1), 500)
n3 = len(r3) if r3 is not None else 0
print(f"D1 from 2020: {n3} bars")
if r3 is not None and n3 > 0:
    df3 = pd.DataFrame(r3)
    df3["time"] = pd.to_datetime(df3["time"], unit="s")
    print(f"  Range: {df3['time'].min()} to {df3['time'].max()}")

mt5.shutdown()
print("Done.")
