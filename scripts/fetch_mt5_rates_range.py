from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import MetaTrader5 as mt5  # noqa: E402
from xauusd_ai.config import load_settings  # noqa: E402


TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def parse_dt(text: str) -> datetime:
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def normalise_rates(rates) -> pd.DataFrame:
    frame = pd.DataFrame(rates)
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    keep = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    frame = frame[[column for column in keep if column in frame.columns]].copy()
    if "spread" in frame.columns:
        frame = frame.rename(columns={"spread": "spread_points"})
    if "spread_points" not in frame.columns:
        frame["spread_points"] = 0.0
    if "tick_volume" not in frame.columns:
        frame["tick_volume"] = 0.0
    frame["tick_volume_delta"] = pd.to_numeric(frame["tick_volume"], errors="coerce").diff().fillna(0.0)
    candle_range = (frame["high"] - frame["low"]).replace(0, pd.NA)
    frame["volume_imbalance"] = ((frame["close"] - frame["open"]).abs() / candle_range).fillna(0.0).clip(0.0, 1.0)
    return frame.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch MT5 rates by UTC date range with copy_rates_range.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/live_acc2.yaml")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", choices=sorted(TIMEFRAMES), default="M5")
    parser.add_argument("--from-date", default="2003-01-01")
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    settings = load_settings(args.config)
    cfg = settings.integrations.mt5
    symbol = args.symbol or settings.market.symbol
    date_from = parse_dt(args.from_date)
    date_to = parse_dt(args.to_date) if args.to_date else datetime.now(timezone.utc)

    ok = False
    if cfg.login and cfg.password and cfg.server:
        ok = mt5.initialize(login=int(cfg.login), password=cfg.password, server=cfg.server)
    if not ok:
        ok = mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select failed for {symbol}: {mt5.last_error()}")
        rates = mt5.copy_rates_range(symbol, TIMEFRAMES[args.timeframe], date_from, date_to)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_range returned no bars: {mt5.last_error()}")
        frame = normalise_rates(rates)
    finally:
        mt5.shutdown()

    out = args.out if args.out.is_absolute() else ROOT / args.out
    if args.merge_existing and out.exists():
        old = pd.read_csv(out)
        old["time"] = pd.to_datetime(old["time"], utc=True, errors="coerce")
        frame = pd.concat([old, frame], ignore_index=True, sort=False)
        frame = frame.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    export = frame.copy()
    export["time"] = pd.to_datetime(export["time"], utc=True).dt.strftime("%Y.%m.%d %H:%M")
    export.to_csv(out, index=False)
    result = {
        "out": str(out),
        "symbol": symbol,
        "timeframe": args.timeframe,
        "rows": int(len(frame)),
        "start": str(frame["time"].min()),
        "end": str(frame["time"].max()),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
