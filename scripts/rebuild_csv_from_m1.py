"""
Resample M1 data -> M5, M15, M30, H1, H4
Va update D1 tu MT5 neu can.
Chay: python scripts/rebuild_csv_from_m1.py
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, "src")

DATA = Path("src/xauusd_ai/real_data")

print("=== Rebuild CSV from M1 resampling ===")

# Doc M1
m1_path = DATA / "XAUUSDm_M1.csv"
print(f"Reading {m1_path}...")
m1 = pd.read_csv(m1_path, on_bad_lines="skip")
m1["time"] = pd.to_datetime(m1["time"], utc=True, errors="coerce")
m1 = m1.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
if "volume" in m1.columns and "tick_volume" not in m1.columns:
    m1 = m1.rename(columns={"volume": "tick_volume"})
if "tick_volume" not in m1.columns:
    m1["tick_volume"] = 0
print(f"M1: {len(m1):,} rows | {str(m1['time'].min())[:10]} -> {str(m1['time'].max())[:10]}")


def resample_to_tf(df: pd.DataFrame, rule: str, tf_name: str) -> pd.DataFrame:
    agg = (
        df.set_index("time")
        .resample(rule)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            tick_volume=("tick_volume", "sum"),
        )
        .dropna(subset=["open"])
        .reset_index()
    )
    agg["spread_points"] = 0.0
    agg["tick_volume_delta"] = agg["tick_volume"].diff().fillna(0)
    rng = (agg["high"] - agg["low"]).replace(0, float("nan"))
    agg["volume_imbalance"] = (agg["close"] - agg["open"]).abs() / rng
    agg["volume_imbalance"] = agg["volume_imbalance"].fillna(0)
    print(f"{tf_name}: {len(agg):,} rows | {str(agg['time'].min())[:10]} -> {str(agg['time'].max())[:10]}")
    return agg


timeframes = [
    ("M5",  "5min"),
    ("M15", "15min"),
    ("M30", "30min"),
    ("H1",  "1h"),
    ("H4",  "4h"),
]

for tf_name, rule in timeframes:
    print(f"\nResampling -> {tf_name}...")
    out = resample_to_tf(m1, rule, tf_name)
    csv_path = DATA / f"XAUUSDm_{tf_name}.csv"
    # Write to temp then rename (atomic, safe)
    tmp = csv_path.with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(csv_path)
    print(f"  Saved: {csv_path} ({len(out):,} rows)")

# Update D1 from MT5 (D1 only has ~14 new bars, quick)
print("\nUpdating D1 from MT5...")
try:
    import MetaTrader5 as mt5
    from xauusd_ai.config import load_settings
    s = load_settings(Path("configs/live_acc2.yaml"))
    cfg = s.integrations.mt5
    ok = mt5.initialize(login=int(cfg.login), password=cfg.password, server=cfg.server)
    if not ok:
        ok = mt5.initialize()
    if ok:
        symbol = s.market.symbol
        mt5.symbol_select(symbol, True)
        import time; time.sleep(1)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 500)
        mt5.shutdown()
        if rates is not None and len(rates) > 0:
            new_d1 = pd.DataFrame(rates)
            new_d1["time"] = pd.to_datetime(new_d1["time"], unit="s", utc=True)
            new_d1 = new_d1[["time", "open", "high", "low", "close", "tick_volume"]]
            old_d1 = pd.read_csv(DATA / "XAUUSDm_D1.csv", on_bad_lines="skip")
            old_d1["time"] = pd.to_datetime(old_d1["time"], utc=True, errors="coerce")
            combined = pd.concat([old_d1, new_d1], ignore_index=True)
            combined = combined.drop_duplicates("time").sort_values("time").reset_index(drop=True)
            tmp = (DATA / "XAUUSDm_D1.csv").with_suffix(".tmp")
            combined.to_csv(tmp, index=False)
            tmp.replace(DATA / "XAUUSDm_D1.csv")
            print(f"D1 updated: {len(combined):,} rows | last: {str(combined['time'].max())[:10]}")
        else:
            print(f"D1: MT5 returned no data: {mt5.last_error()}")
    else:
        print(f"D1: MT5 init failed, keeping existing")
except Exception as e:
    print(f"D1: MT5 update failed ({e}), keeping existing")

print("\n=== Summary ===")
for f in ["M5", "M15", "M30", "H1", "H4", "D1"]:
    p = DATA / f"XAUUSDm_{f}.csv"
    df = pd.read_csv(p, usecols=["time"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    print(f"  {f:4s}: {len(df):8,} rows | {str(df['time'].min())[:10]} -> {str(df['time'].max())[:10]}")

print("\nDone! CSV files updated.")
