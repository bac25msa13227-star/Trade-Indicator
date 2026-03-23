"""
Re-fetch full history cho M5, M15, H1, H4, D1 tu MT5 voi so bars lon nhat.
Chay: python scripts/fetch_full_history_mt5.py
"""
import sys
import time
import pandas as pd
from pathlib import Path

sys.path.insert(0, "src")

from xauusd_ai.config import load_settings
import MetaTrader5 as mt5

DATA = Path("src/xauusd_ai/real_data")

s = load_settings(Path("configs/live_acc2.yaml"))
cfg = s.integrations.mt5

print("Connecting to MT5...")
ok = mt5.initialize(login=int(cfg.login), password=cfg.password, server=cfg.server)
if not ok:
    ok = mt5.initialize()
if not ok:
    print(f"MT5 init failed: {mt5.last_error()}")
    sys.exit(1)

print(f"Connected: {mt5.terminal_info().name}")
symbol = s.market.symbol
mt5.symbol_select(symbol, True)
time.sleep(2)

# Fetch tu pos=0 voi so bars toi da
TARGETS = [
    ("M5",  mt5.TIMEFRAME_M5,  200_000),
    ("M15", mt5.TIMEFRAME_M15, 200_000),
    ("H1",  mt5.TIMEFRAME_H1,  100_000),
    ("H4",  mt5.TIMEFRAME_H4,   50_000),
    ("D1",  mt5.TIMEFRAME_D1,    5_000),
]

for tf_name, tf_code, max_bars in TARGETS:
    csv_path = DATA / f"XAUUSDm_{tf_name}.csv"
    print(f"\nFetching {tf_name} ({max_bars:,} bars)...")
    rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, max_bars)
    if rates is None or len(rates) == 0:
        print(f"  ERROR: {mt5.last_error()} - keeping existing CSV")
        if csv_path.exists():
            df_ex = pd.read_csv(csv_path, usecols=["time"])
            print(f"  Existing: {len(df_ex):,} rows")
        continue

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    keep = ["time", "open", "high", "low", "close", "tick_volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["spread_points"] = 0.0
    df["tick_volume_delta"] = df["tick_volume"].diff().fillna(0)
    rng = (df["high"] - df["low"]).replace(0, float("nan"))
    df["volume_imbalance"] = (df["close"] - df["open"]).abs() / rng
    df["volume_imbalance"] = df["volume_imbalance"].fillna(0)

    # Merge voi CSV cu neu co data cu hon
    if csv_path.exists():
        old = pd.read_csv(csv_path, on_bad_lines="skip")
        old["time"] = pd.to_datetime(old["time"], utc=True, errors="coerce")
        old = old.dropna(subset=["time"])
        # Chi merge neu CSV cu co data cu hon
        old_start = old["time"].min()
        new_start = df["time"].min()
        if old_start < new_start:
            print(f"  Old CSV starts {str(old_start)[:10]}, new starts {str(new_start)[:10]} - merging")
            # Lay phan cu hon tu old
            old_part = old[old["time"] < new_start].copy()
            for col in df.columns:
                if col not in old_part.columns:
                    old_part[col] = 0
            df = pd.concat([old_part[df.columns], df], ignore_index=True)
            df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

    tmp = csv_path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(csv_path)
    print(f"  Saved: {len(df):,} rows | {str(df['time'].min())[:10]} -> {str(df['time'].max())[:10]}")

mt5.shutdown()

print("\n=== Final CSV Summary ===")
for f in ["M5", "M15", "H1", "H4", "D1"]:
    p = DATA / f"XAUUSDm_{f}.csv"
    df = pd.read_csv(p, usecols=["time"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    print(f"  {f:4s}: {len(df):8,} rows | {str(df['time'].min())[:10]} -> {str(df['time'].max())[:10]}")

print("\nDone!")
