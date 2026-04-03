#!/usr/bin/env python3
"""
Download XAU/USD M1 data from Dukascopy and rebuild all project CSV timeframes.

Usage:
  python scripts/fetch_xauusd_dukascopy.py --from 2018-01-01 --to 2026-03-30
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from time import sleep

import pandas as pd


TF_RULES: dict[str, str] = {
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
}


@dataclass(frozen=True)
class Chunk:
    start: date
    end: date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch XAUUSD data from Dukascopy and rebuild CSV folder.")
    parser.add_argument("--from", dest="from_date", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", default=date.today().isoformat(), help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--instrument",
        default="xauusd",
        help="Dukascopy instrument id (default: xauusd)",
    )
    parser.add_argument(
        "--out-dir",
        default="src/xauusd_ai/real_data",
        help="Output folder for CSV files",
    )
    parser.add_argument(
        "--chunk",
        choices=["year", "quarter"],
        default="year",
        help="Download chunking strategy",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="dukascopy-node batch size")
    parser.add_argument("--batch-pause", type=int, default=1000, help="dukascopy-node batch pause (ms)")
    parser.add_argument("--retries", type=int, default=2, help="dukascopy-node retries per artifact")
    parser.add_argument("--chunk-retries", type=int, default=4, help="Script-level retries per chunk")
    return parser.parse_args()


def to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def build_chunks(start: date, end: date, mode: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur = start
    while cur <= end:
        if mode == "quarter":
            q = (cur.month - 1) // 3
            q_end_month = 3 * (q + 1)
            if q_end_month == 12:
                next_boundary = date(cur.year + 1, 1, 1)
            else:
                next_boundary = date(cur.year, q_end_month + 1, 1)
            chunk_end = min(end, next_boundary - timedelta(days=1))
        else:
            next_boundary = date(cur.year + 1, 1, 1)
            chunk_end = min(end, next_boundary - timedelta(days=1))
        chunks.append(Chunk(start=cur, end=chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def run_download(
    instrument: str,
    chunk: Chunk,
    out_dir: Path,
    cache_dir: Path,
    chunk_store_dir: Path,
    batch_size: int,
    batch_pause: int,
    retries: int,
    chunk_retries: int,
) -> Path:
    safe_instrument = instrument.replace("/", "_").lower()
    cached_chunk = chunk_store_dir / f"{safe_instrument}_m1_{chunk.start.isoformat()}_{chunk.end.isoformat()}.csv"
    if cached_chunk.exists() and cached_chunk.stat().st_size > 0:
        print(f"\n[download] {chunk.start} -> {chunk.end} (cached)")
        return cached_chunk

    # dukascopy-node 'to' is effectively exclusive for daily boundaries.
    to_exclusive = chunk.end + timedelta(days=1)
    cmd = [
        "npx",
        "dukascopy-node",
        "-i",
        instrument,
        "-from",
        chunk.start.isoformat(),
        "-to",
        to_exclusive.isoformat(),
        "-t",
        "m1",
        "-p",
        "bid",
        "-v",
        "-f",
        "csv",
        "-dir",
        str(out_dir),
        "-bs",
        str(batch_size),
        "-bp",
        str(batch_pause),
        "-r",
        str(retries),
        "-re",
        "-ch",
        "-chpath",
        str(cache_dir),
    ]
    last_error: Exception | None = None
    for attempt in range(1, max(chunk_retries, 1) + 1):
        print(f"\n[download] {chunk.start} -> {chunk.end} (attempt {attempt}/{chunk_retries})")
        before_files = {p.resolve() for p in out_dir.glob("*.csv")}
        try:
            subprocess.run(cmd, check=True)
            after_files = [p.resolve() for p in out_dir.glob("*.csv")]
            new_files = [Path(p) for p in after_files if p not in before_files]
            candidates = sorted(new_files, key=lambda p: p.stat().st_mtime) if new_files else []
            if not candidates:
                fallback = sorted(out_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
                if not fallback:
                    raise RuntimeError(f"No CSV output found for {chunk.start} -> {chunk.end}")
                picked = fallback[-1]
            else:
                picked = candidates[-1]

            chunk_store_dir.mkdir(parents=True, exist_ok=True)
            picked.replace(cached_chunk)
            return cached_chunk
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[warn] chunk failed {chunk.start}->{chunk.end}: {exc}")
            if attempt < chunk_retries:
                backoff = min(15 * attempt, 60)
                print(f"[retry] sleeping {backoff}s before retry...")
                sleep(backoff)

    raise RuntimeError(f"Chunk failed after retries: {chunk.start} -> {chunk.end}") from last_error


def load_chunk_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, on_bad_lines="skip")
    if frame.empty or "timestamp" not in frame.columns:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])
    out = frame.rename(columns={"volume": "tick_volume"}).copy()
    out["time"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True, errors="coerce")
    out = out.dropna(subset=["time"])
    keep = ["time", "open", "high", "low", "close", "tick_volume"]
    out = out[[c for c in keep if c in out.columns]].copy()
    if "tick_volume" not in out.columns:
        out["tick_volume"] = 0.0
    out["tick_volume"] = pd.to_numeric(out["tick_volume"], errors="coerce").fillna(0.0)
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def enrich_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["spread_points"] = 0.0
    enriched["tick_volume_delta"] = enriched["tick_volume"].diff().fillna(0.0)
    rng = (enriched["high"] - enriched["low"]).replace(0.0, float("nan"))
    enriched["volume_imbalance"] = ((enriched["close"] - enriched["open"]).abs() / rng).fillna(0.0)
    return enriched


def save_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def resample_from_m1(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = (
        m1.set_index("time")
        .resample(rule)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            tick_volume=("tick_volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return enrich_columns(out)


def print_summary(tf: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        print(f"  {tf:4s}: empty")
        return
    print(f"  {tf:4s}: {len(frame):10,} rows | {frame['time'].min()} -> {frame['time'].max()}")


def main() -> None:
    args = parse_args()
    start = to_date(args.from_date)
    end = to_date(args.to_date)
    if start > end:
        raise ValueError("--from must be <= --to")

    out_dir = Path(args.out_dir)
    cache_dir = Path(".dukascopy-cache")
    chunk_store_dir = cache_dir / "chunks"
    chunks = build_chunks(start, end, args.chunk)
    print(f"Downloading {len(chunks)} chunk(s): {start} -> {end}")

    frames: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="dukascopy_xauusd_") as tmp:
        tmp_dir = Path(tmp)
        for chunk in chunks:
            csv_path = run_download(
                instrument=args.instrument,
                chunk=chunk,
                out_dir=tmp_dir,
                cache_dir=cache_dir,
                chunk_store_dir=chunk_store_dir,
                batch_size=args.batch_size,
                batch_pause=args.batch_pause,
                retries=args.retries,
                chunk_retries=args.chunk_retries,
            )
            chunk_df = load_chunk_csv(csv_path)
            print(f"  -> loaded {len(chunk_df):,} rows from {csv_path.name}")
            frames.append(chunk_df)

    if not frames:
        raise RuntimeError("No data downloaded")

    m1 = pd.concat(frames, ignore_index=True)
    m1 = m1.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts_exclusive = pd.Timestamp(end + timedelta(days=1), tz="UTC")
    m1 = m1[(m1["time"] >= start_ts) & (m1["time"] < end_ts_exclusive)].copy()
    m1 = enrich_columns(m1)

    m1_path = out_dir / "XAUUSDm_M1.csv"
    save_csv_atomic(m1, m1_path)
    print(f"\nSaved M1: {m1_path} ({len(m1):,} rows)")

    print("\nResampling from M1...")
    outputs: dict[str, pd.DataFrame] = {"M1": m1}
    for tf, rule in TF_RULES.items():
        tf_df = resample_from_m1(m1, rule)
        tf_path = out_dir / f"XAUUSDm_{tf}.csv"
        save_csv_atomic(tf_df, tf_path)
        outputs[tf] = tf_df
        print(f"  saved {tf_path.name} ({len(tf_df):,} rows)")

    print("\n=== Summary ===")
    for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
        print_summary(tf, outputs[tf])


if __name__ == "__main__":
    main()
