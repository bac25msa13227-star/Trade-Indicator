"""
Script cập nhật CSV data từ MT5 (incremental append).
Chỉ tải các bar mới hơn timestamp cuối đã có trong CSV.

Usage:
  python scripts/update_csv_from_mt5.py
  python scripts/update_csv_from_mt5.py --config configs/live_acc2.yaml
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd

TIMEFRAME_MT5_MAP = {
    "M1":  1,
    "M5":  5,
    "M15": 15,
    "M30": 30,
    "H1":  16385,
    "H4":  16388,
    "D1":  16408,
}


def update_csv(symbol: str, tf_name: str, mt5_tf: int, csv_path: Path, max_bars: int = 200_000) -> int:
    """Fetch incremental bars từ MT5 và append vào CSV. Trả về số rows mới."""
    import MetaTrader5 as mt5

    existing_last_ts = None
    existing_df = None
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path, usecols=["time"], on_bad_lines="skip")
            existing_df["time"] = pd.to_datetime(existing_df["time"], errors="coerce", utc=True)
            existing_df = existing_df.dropna(subset=["time"])
            if not existing_df.empty:
                existing_last_ts = existing_df["time"].max()
        except Exception as e:
            print(f"  [{tf_name}] Không đọc được CSV hiện tại: {e}")

    # Luôn dùng copy_rates_from_pos (đáng tin hơn copy_rates_from khi dùng UTC datetime)
    # Lấy đủ bars để cover 30 ngày: M1=30*24*60=43200, M5=8640, M15=2880, H1=720, etc.
    bars_per_tf = {1: 60000, 5: 20000, 15: 8000, 30: 4000, 16385: 2000, 16388: 800, 16408: 200}
    fetch_bars = bars_per_tf.get(mt5_tf, max_bars)
    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, fetch_bars)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        print(f"  [{tf_name}] MT5 error: {err}")
        return 0

    new_df = pd.DataFrame(rates)
    new_df["time"] = pd.to_datetime(new_df["time"], unit="s", utc=True)
    cols = ["time", "open", "high", "low", "close", "tick_volume"]
    available = [c for c in cols if c in new_df.columns]
    new_df = new_df[available].copy()
    if "tick_volume" not in new_df.columns:
        new_df["tick_volume"] = 0
    if "spread_points" not in new_df.columns:
        new_df["spread_points"] = new_df.get("spread", 0)

    if csv_path.exists():
        old_df = pd.read_csv(csv_path, on_bad_lines="skip")
        old_df["time"] = pd.to_datetime(old_df["time"], errors="coerce", utc=True)
        # Align columns
        for col in new_df.columns:
            if col not in old_df.columns:
                old_df[col] = 0
        for col in old_df.columns:
            if col not in new_df.columns:
                new_df[col] = 0
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined["time"] = pd.to_datetime(combined["time"], utc=True, errors="coerce")
    before = len(combined)
    combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    new_rows = len(combined) - (len(old_df) if csv_path.exists() else 0)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    return max(new_rows, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_acc2.yaml")
    parser.add_argument("--symbol", default=None, help="Override symbol (default: từ config)")
    args = parser.parse_args()

    from xauusd_ai.config import load_settings
    settings = load_settings(Path(args.config))

    symbol = args.symbol or settings.market.symbol
    data_dir = Path(settings.market.csv_folder_path)

    print(f"=== Update CSV từ MT5 ===")
    print(f"Symbol : {symbol}")
    print(f"DataDir: {data_dir}")
    print()

    # Kết nối MT5
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 package không được cài. Dùng: pip install MetaTrader5")
        sys.exit(1)

    # Khởi tạo MT5 với credentials từ config
    mt5_cfg = settings.integrations.mt5
    init_ok = mt5.initialize(
        login=int(mt5_cfg.login),
        password=mt5_cfg.password,
        server=mt5_cfg.server,
    )
    if not init_ok:
        # Thử khởi tạo không cần credentials (nếu MT5 đã mở sẵn)
        init_ok = mt5.initialize()
    if not init_ok:
        print(f"ERROR: Không kết nối được MT5: {mt5.last_error()}")
        sys.exit(1)

    print(f"MT5 connected: {mt5.terminal_info().name if mt5.terminal_info() else 'OK'}")
    mt5.symbol_select(symbol, True)
    time.sleep(1)

    total_new = 0
    for tf_name, mt5_tf in TIMEFRAME_MT5_MAP.items():
        csv_path = data_dir / f"XAUUSDm_{tf_name}.csv"
        max_bars = 200_000 if tf_name in ("M1", "M5") else 100_000
        try:
            n = update_csv(symbol, tf_name, mt5_tf, csv_path, max_bars=max_bars)
            # Đọc lại để kiểm tra
            df = pd.read_csv(csv_path, usecols=["time"])
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            last = str(df["time"].max())[:16]
            print(f"  {tf_name:4s}: +{n:5d} rows mới | tổng {len(df):7,} rows | đến {last}")
            total_new += n
        except Exception as e:
            print(f"  {tf_name}: ERROR: {e}")

    mt5.shutdown()
    print(f"\nHoàn thành. Tổng rows mới: {total_new:,}")


if __name__ == "__main__":
    main()
