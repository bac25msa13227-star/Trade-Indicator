"""Quick test to check MT5 Python API data availability."""
import MetaTrader5 as mt5
from datetime import datetime, timezone

mt5.initialize()
info = mt5.terminal_info()
print(f"Connected: {info.name}  build={info.build}")
print(f"Datapath: {info.data_path}")

symbol = "XAUUSD"
mt5.symbol_select(symbol, True)
sym_info = mt5.symbol_info(symbol)
print(f"Symbol info: spread={sym_info.spread if sym_info else 'N/A'}")

# Test 1: Recent data (naive datetime)
rates1 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, datetime(2024, 10, 1), datetime(2024, 10, 10))
print(f"Recent M5 naive: {'OK ' + str(len(rates1)) + ' bars' if rates1 is not None and len(rates1) > 0 else 'FAIL ' + str(mt5.last_error())}")

# Test 2: Recent data (UTC aware)
rates2 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, datetime(2024, 10, 1, tzinfo=timezone.utc), datetime(2024, 10, 10, tzinfo=timezone.utc))
print(f"Recent M5 UTC:   {'OK ' + str(len(rates2)) + ' bars' if rates2 is not None and len(rates2) > 0 else 'FAIL ' + str(mt5.last_error())}")

# Test 3: Old data 2022
rates3 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, datetime(2022, 1, 1), datetime(2022, 2, 1))
print(f"Old M5 2022:     {'OK ' + str(len(rates3)) + ' bars' if rates3 is not None and len(rates3) > 0 else 'FAIL ' + str(mt5.last_error())}")

# Test 4: Copy from pos
rates4 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 5)
print(f"Last 5 M5 bars:  {'OK' if rates4 is not None and len(rates4) > 0 else 'FAIL ' + str(mt5.last_error())}")
if rates4 is not None and len(rates4) > 0:
    import pandas as pd
    df = pd.DataFrame(rates4)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    print(df[["time", "open", "high", "low", "close"]].to_string())

# Test 5: H1 data
rates5 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, datetime(2022, 1, 1), datetime(2022, 2, 1))
print(f"Old H1 2022:     {'OK ' + str(len(rates5)) + ' bars' if rates5 is not None and len(rates5) > 0 else 'FAIL ' + str(mt5.last_error())}")

mt5.shutdown()
print("Done.")
