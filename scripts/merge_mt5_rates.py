"""
Merge two MT5 M5 rate CSV files (e.g. pre-2023 + existing 2023-2026) into one sorted file.

Usage:
    python scripts/merge_mt5_rates.py \
        --old outputs/mt5_rates_export_pre2023.csv \
        --new outputs/mt5_rates_export_202306_202603.csv \
        --out outputs/mt5_rates_merged.csv

The output can then be used with build_mt5_full_features.py:
    python scripts/build_mt5_full_features.py \
        --rates outputs/mt5_rates_merged.csv \
        --from-date 2019-01-01 --to-date 2026-04-01 \
        --out outputs/mt5_full_ict_wyckoff_features_merged.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLS = {"time", "open", "high", "low", "close"}


def _load_rates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = sorted(REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if "tick_volume" not in df.columns:
        df["tick_volume"] = df.get("volume", 0)
    if "spread" not in df.columns:
        df["spread"] = 0
    if "real_volume" not in df.columns:
        df["real_volume"] = 0
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge two MT5 M5 rate CSV files into one sorted file.")
    parser.add_argument("--old", type=Path, required=True, help="Older (earlier) rate CSV, e.g. pre-2023 export")
    parser.add_argument("--new", type=Path, required=True, help="Newer rate CSV, e.g. 202306_202603 export")
    parser.add_argument("--out", type=Path, required=True, help="Output merged CSV path")
    args = parser.parse_args()

    old = _load_rates(args.old)
    new = _load_rates(args.new)

    combined = pd.concat([old, new], ignore_index=True)
    combined = (
        combined
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )

    overlap_start = max(old["time"].min(), new["time"].min())
    overlap_end = min(old["time"].max(), new["time"].max())
    overlap_rows = int((combined["time"] >= overlap_start) & (combined["time"] <= overlap_end)).sum() if overlap_end > overlap_start else 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined["time"] = combined["time"].dt.strftime("%Y.%m.%d %H:%M")
    out_cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    out_cols = [c for c in out_cols if c in combined.columns]
    combined[out_cols].to_csv(args.out, index=False)

    print(f"old:      {args.old}  rows={len(old)}  range={old['time'].min()} -> {old['time'].max()}")
    print(f"new:      {args.new}  rows={len(new)}  range={new['time'].min()} -> {new['time'].max()}")
    print(f"overlap:  {overlap_rows} rows between {overlap_start} and {overlap_end}")
    print(f"merged:   {args.out}  rows={len(combined)}  range={combined['time'].min()} -> {combined['time'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
