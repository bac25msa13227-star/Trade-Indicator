from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xauusd_ai.config import load_settings
from xauusd_ai.features.dataset import (  # noqa: E402
    FEATURE_COLUMNS,
    _expected_direction_from_context,
    _merge_context,
)
from xauusd_ai.strategies.hybrid import HybridStrategy  # noqa: E402


RESAMPLE_RULES = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
}


def _parse_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _normalise_mt5_rates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    lower_map = {column: column.strip().lower() for column in frame.columns}
    frame = frame.rename(columns=lower_map)

    required = {"time", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    if "tick_volume" not in frame.columns:
        if "volume" in frame.columns:
            frame["tick_volume"] = frame["volume"]
        else:
            frame["tick_volume"] = 0.0
    if "spread_points" not in frame.columns:
        if "spread" in frame.columns:
            frame["spread_points"] = frame["spread"]
        else:
            frame["spread_points"] = 0.0

    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "tick_volume", "spread_points"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = (
        frame.dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )
    frame = frame[frame["high"] >= frame["low"]].copy()
    if frame.empty:
        raise ValueError(f"No usable MT5 bars loaded from {path}")
    return _add_flow_columns(frame)


def _add_flow_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["tick_volume"] = result["tick_volume"].fillna(0.0)
    result["spread_points"] = result["spread_points"].fillna(0.0)
    result["tick_volume_delta"] = result["tick_volume"].diff().fillna(0.0)
    candle_range = (result["high"] - result["low"]).replace(0, np.nan)
    result["volume_imbalance"] = ((result["close"] - result["open"]).abs() / candle_range).fillna(0.0).clip(0.0, 1.0)
    return result


def _resample_from_base(base: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "M5":
        return base.copy().reset_index(drop=True)
    if timeframe not in RESAMPLE_RULES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; known={sorted(RESAMPLE_RULES)}")

    resampled = (
        base.set_index("time")
        .resample(RESAMPLE_RULES[timeframe], label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "tick_volume": "sum",
                "spread_points": "mean",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return _add_flow_columns(resampled)


def _build_frames(base: pd.DataFrame, timeframes: set[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for timeframe in sorted(timeframes, key=lambda tf: list(RESAMPLE_RULES).index(tf)):
        frames[timeframe] = _resample_from_base(base, timeframe)
    return frames


def _safe_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    export = frame.copy()
    export["time"] = pd.to_datetime(export["time"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    export.to_csv(path, index=False, float_format="%.10g")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build full ICT/Wyckoff multi-timeframe features from MT5 exported M5 bars only."
    )
    parser.add_argument("--rates", type=Path, default=ROOT / "outputs/mt5_rates_export_202306_202603.csv")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/grid_search/grid_0014_th70_r5_mp4_notrail_slip15.yaml",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/mt5_full_ict_wyckoff_features_202306_202603.csv")
    parser.add_argument("--from-date", default="2023-06-01")
    parser.add_argument("--to-date", default="2026-04-01")
    args = parser.parse_args()

    settings = load_settings(args.config)
    required_timeframes = {
        settings.market.execution_timeframe,
        settings.market.mid_timeframe,
        settings.market.structure_timeframe,
        settings.market.higher_timeframe,
    }
    base = _normalise_mt5_rates(args.rates)
    frames = _build_frames(base, required_timeframes)

    strategy = HybridStrategy(settings)
    merged = _merge_context(settings, frames)
    strategy_output = strategy.annotate_dataset(merged)
    dataset = merged.join(strategy_output)
    dataset["expected_direction"] = _expected_direction_from_context(dataset)
    dataset["trade_side"] = np.where(dataset["expected_direction"].astype(int) >= 0, "buy", "sell")

    start = _parse_timestamp(args.from_date)
    end = _parse_timestamp(args.to_date)
    dataset["time"] = pd.to_datetime(dataset["time"], utc=True)
    if start is not None:
        dataset = dataset[dataset["time"] >= start]
    if end is not None:
        dataset = dataset[dataset["time"] < end]

    required_export_columns = ["time", "open", "high", "low", "close", "tick_volume", "spread_points"]
    missing_feature_columns = [column for column in FEATURE_COLUMNS if column not in dataset.columns]
    if missing_feature_columns:
        raise RuntimeError(f"Feature build missed required columns: {missing_feature_columns}")

    dataset = dataset.dropna(subset=FEATURE_COLUMNS).copy()
    ordered_columns = [
        column
        for column in dict.fromkeys(required_export_columns + FEATURE_COLUMNS + ["expected_direction", "trade_side"])
        if column in dataset.columns
    ]
    dataset = dataset[ordered_columns].reset_index(drop=True)
    _safe_to_csv(dataset, args.out)

    print(f"input_rates={args.rates}")
    print(f"config={args.config}")
    for timeframe, frame in frames.items():
        print(
            f"{timeframe}: rows={len(frame)} "
            f"range={frame['time'].min()} -> {frame['time'].max()}"
        )
    print(f"output={args.out}")
    print(
        f"feature_rows={len(dataset)} feature_cols={len(dataset.columns)} "
        f"range={dataset['time'].min()} -> {dataset['time'].max()}"
    )
    print(f"missing_feature_columns={missing_feature_columns}")
    print(dataset["trade_side"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
