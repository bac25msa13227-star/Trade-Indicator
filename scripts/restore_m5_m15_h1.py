"""Restore M5, M15, H1 CSV data from MT5 with 99999 bars."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import MetaTrader5 as mt5
import pandas as pd
from xauusd_ai.config import load_settings

s = load_settings(Path("configs/live_acc2.yaml"))
ok = mt5.initialize(
    login=int(s.integrations.mt5.login),
    password=s.integrations.mt5.password,
    server=s.integrations.mt5.server,
)
if not ok:
    print("MT5 init failed:", mt5.last_error())
    sys.exit(1)
mt5.symbol_select(s.market.symbol, True)
time.sleep(1)

data_dir = Path("src/xauusd_ai/real_data")

targets = [
    ("M5",  mt5.TIMEFRAME_M5,  99999, "XAUUSD_M5.csv"),
    ("M15", mt5.TIMEFRAME_M15, 99999, "XAUUSD_M15.csv"),
    ("H1",  mt5.TIMEFRAME_H1,  99999, "XAUUSD_H1.csv"),
]

for tf_name, tf_code, nb, fname in targets:
    print(f"Fetching {tf_name} ({nb} bars)...", flush=True)
    r = mt5.copy_rates_from_pos(s.market.symbol, tf_code, 0, nb)
    if r is None:
        print(f"  ERROR: {mt5.last_error()}")
        continue

    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "tick_volume": "Volume", "real_volume": "RealVolume", "spread": "Spread",
    }, inplace=True)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "RealVolume", "Spread"] if c in df.columns]
    df = df[keep]

    out_path = data_dir / fname
    df.to_csv(out_path)
    first = df.index[0].strftime("%Y-%m-%d")
    last = df.index[-1].strftime("%Y-%m-%d")
    print(f"  OK: {len(df)} rows | {first} -> {last} -> {fname}")

mt5.shutdown()
print("Done.")
