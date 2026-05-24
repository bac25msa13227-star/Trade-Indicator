from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


TIMEFRAMES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
}


def _resolve_symbol(mt5, symbol: str) -> str:
    requested = str(symbol)
    mt5.symbol_select(requested, True)
    if mt5.symbol_info(requested) is not None:
        return requested

    candidates = list(mt5.symbols_get(f"{requested}*") or [])
    if not candidates:
        candidates = list(mt5.symbols_get(f"*{requested}*") or [])
    if not candidates:
        raise ValueError(f"symbol not found: {requested}")

    def _score(item) -> tuple:
        name = str(getattr(item, "name", ""))
        trade_mode = int(getattr(item, "trade_mode", 0) or 0)
        visible = bool(getattr(item, "visible", False))
        return (
            1 if name.upper().startswith(requested.upper()) else 0,
            1 if trade_mode == 4 else 0,
            1 if visible else 0,
            -len(name),
        )

    resolved = str(getattr(sorted(candidates, key=_score, reverse=True)[0], "name", requested))
    mt5.symbol_select(resolved, True)
    if mt5.symbol_info(resolved) is None:
        raise ValueError(f"resolved symbol unavailable: requested={requested} resolved={resolved}")
    print(f"resolved_symbol={requested}->{resolved}")
    return resolved


def _normalise_rates(frame: pd.DataFrame) -> pd.DataFrame:
    lower = {column: column.strip().lower() for column in frame.columns}
    frame = frame.rename(columns=lower).copy()
    if "time" not in frame.columns:
        raise ValueError("rates frame has no time column")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "tick_volume" not in frame.columns:
        frame["tick_volume"] = 0
    if "spread" not in frame.columns:
        frame["spread"] = 0
    if "real_volume" not in frame.columns:
        frame["real_volume"] = 0
    required = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    return (
        frame[required]
        .dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )


def _rates_from_bridge(bridge_url: str, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    query = urllib.parse.urlencode({"symbol": symbol, "timeframe": timeframe, "bars": int(bars)})
    url = bridge_url.rstrip("/") + "/rates?" + query
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"bridge returned no rates from {url}")
    frame = pd.DataFrame(payload)
    if "time" in frame.columns and pd.api.types.is_numeric_dtype(frame["time"]):
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    return _normalise_rates(frame)


def _synthesise_missing_m5_from_ticks(mt5, symbol: str, after_time: pd.Timestamp) -> pd.DataFrame:
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or int(getattr(tick, "time", 0) or 0) <= 0:
        return pd.DataFrame()
    point = float(getattr(info, "point", 0.001) or 0.001)
    start = after_time.to_pydatetime() - dt.timedelta(minutes=1)
    ticks = mt5.copy_ticks_from(symbol, start, 200000, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return pd.DataFrame()
    tf = pd.DataFrame(ticks)
    tf["time"] = pd.to_datetime(tf["time"], unit="s", utc=True)
    price = np.where(
        (tf.get("bid", 0).astype(float) > 0) & (tf.get("ask", 0).astype(float) > 0),
        (tf["bid"].astype(float) + tf["ask"].astype(float)) / 2.0,
        np.where(tf.get("bid", 0).astype(float) > 0, tf["bid"].astype(float), tf.get("ask", 0).astype(float)),
    )
    tf["price"] = price
    tf = tf[np.isfinite(tf["price"]) & (tf["price"] > 0)].copy()
    if tf.empty:
        return pd.DataFrame()
    tf["bar_time"] = tf["time"].dt.floor("5min")
    tf = tf[tf["bar_time"] > after_time].copy()
    if tf.empty:
        return pd.DataFrame()
    grouped = tf.groupby("bar_time", sort=True)
    out = pd.DataFrame(
        {
            "time": grouped["bar_time"].first(),
            "open": grouped["price"].first(),
            "high": grouped["price"].max(),
            "low": grouped["price"].min(),
            "close": grouped["price"].last(),
            "tick_volume": grouped.size().astype(int),
            "spread": grouped.apply(
                lambda rows: int(round(float(((rows["ask"].astype(float) - rows["bid"].astype(float)) / point).replace([np.inf, -np.inf], np.nan).dropna().median() or 0)))
            ),
            "real_volume": 0,
        }
    ).reset_index(drop=True)
    return _normalise_rates(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Append fresh M5 bars from the live MT5 terminal into an MT5 rates CSV.")
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", choices=sorted(TIMEFRAMES), default="M5")
    parser.add_argument("--bars", type=int, default=20000)
    parser.add_argument("--bridge-url", default="", help="Prefer HTTP MT5 bridge rates instead of direct MetaTrader5 initialize.")
    args = parser.parse_args()

    old = _normalise_rates(pd.read_csv(args.old)) if args.old.exists() else pd.DataFrame()
    if args.bridge_url:
        try:
            fresh = _rates_from_bridge(args.bridge_url, args.symbol, args.timeframe, int(args.bars))
        except Exception as exc:  # noqa: BLE001
            print(f"bridge rates fetch failed: {exc}", file=sys.stderr)
            return 5
        combined = fresh if old.empty else pd.concat([old, fresh], ignore_index=True)
        combined = _normalise_rates(combined)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        export = combined.copy()
        export["time"] = export["time"].dt.strftime("%Y.%m.%d %H:%M")
        export.to_csv(args.out, index=False)
        print(f"old_rows={len(old)} bridge_rows={len(fresh)} out_rows={len(combined)}")
        print(f"range={combined['time'].min().isoformat()} -> {combined['time'].max().isoformat()}")
        print(f"bridge_url={args.bridge_url}")
        print(f"out={args.out}")
        return 0

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 package is not installed", file=sys.stderr)
        return 2

    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}", file=sys.stderr)
        return 3
    try:
        symbol = _resolve_symbol(mt5, args.symbol)
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[args.timeframe], 0, int(args.bars))
        if rates is None or len(rates) == 0:
            print(f"MT5 returned no rates: {mt5.last_error()}", file=sys.stderr)
            return 4
        fresh = pd.DataFrame(rates)
        fresh["time"] = pd.to_datetime(fresh["time"], unit="s", utc=True)
        fresh = _normalise_rates(fresh)
        if args.timeframe == "M5" and not fresh.empty:
            synthetic = _synthesise_missing_m5_from_ticks(mt5, symbol, fresh["time"].max())
            if not synthetic.empty:
                fresh = _normalise_rates(pd.concat([fresh, synthetic], ignore_index=True))
                print(
                    "synthetic_tick_bars="
                    f"{len(synthetic)} range={synthetic['time'].min().isoformat()} -> {synthetic['time'].max().isoformat()}"
                )
    finally:
        mt5.shutdown()

    combined = fresh if old.empty else pd.concat([old, fresh], ignore_index=True)
    combined = _normalise_rates(combined)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export = combined.copy()
    export["time"] = export["time"].dt.strftime("%Y.%m.%d %H:%M")
    export.to_csv(args.out, index=False)
    print(f"old_rows={len(old)} fresh_rows={len(fresh)} out_rows={len(combined)}")
    print(f"range={combined['time'].min().isoformat()} -> {combined['time'].max().isoformat()}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
