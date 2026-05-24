from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_folds(path: Path) -> list[dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    folds = report.get("folds", [])
    if not folds:
        raise ValueError(f"No folds found in {path}")
    return folds


def _empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "open_time",
            "direction",
            "entry_price",
            "sl_price",
            "tp_price",
            "atr",
            "probability",
        ]
    )


def build_signals(
    wf_signals_path: Path,
    features_path: Path,
    min_profit_r: float,
    broker_gmt: int,
    sl_mult: float,
    tp_rr: float,
    volatile_tp_rr: float,
    sideway_tp_rr: float,
    min_probability: float | None = None,
    include_broker_hours: set[int] | None = None,
    exclude_broker_hours: set[int] | None = None,
    side: str = "all",
    min_atr_percentile: float | None = None,
    max_atr_percentile: float | None = None,
) -> pd.DataFrame:
    wf = pd.read_csv(wf_signals_path)
    wf["time"] = pd.to_datetime(wf["time"], utc=True, errors="coerce")
    wf = wf.dropna(subset=["time"]).copy()
    if "proba" not in wf.columns:
        raise ValueError(f"{wf_signals_path} must contain a proba column")
    if "fold" not in wf.columns:
        raise ValueError(f"{wf_signals_path} must contain a fold column")

    usecols = ["time", "close", "atr", "trade_side", "volatility_regime", "atr_percentile"]
    feat = pd.read_csv(features_path, usecols=usecols)
    feat["time"] = pd.to_datetime(feat["time"], utc=True, errors="coerce")
    feat = feat.dropna(subset=["time"]).copy()

    merged = wf.merge(feat, on="time", how="left")
    merged = merged.dropna(subset=["close", "atr"]).copy()

    # Same expected-R estimator used by walkforward_ict_wyckoff.py and live loop.
    data_driven_rr = 3.67
    merged["estimated_profit_r"] = (
        merged["proba"].astype(float) * data_driven_rr
        - (1.0 - merged["proba"].astype(float))
    )
    merged = merged[merged["estimated_profit_r"] > min_profit_r].copy()
    if min_probability is not None:
        merged = merged[merged["proba"].astype(float) >= min_probability].copy()
    if side in {"buy", "sell"}:
        merged = merged[merged["trade_side"] == side].copy()
    if min_atr_percentile is not None:
        merged = merged[merged["atr_percentile"].astype(float) >= min_atr_percentile].copy()
    if max_atr_percentile is not None:
        merged = merged[merged["atr_percentile"].astype(float) <= max_atr_percentile].copy()

    # Python WF/live simulation trades the model's probability in the
    # feature-engineered expected direction. The raw strategy_score sign is
    # only one component and can disagree with trade_side; using it here makes
    # MT5 test a different strategy.
    merged["direction"] = (
        merged["trade_side"].map({"buy": 1, "sell": -1}).fillna(1).astype(int)
    )

    regime = merged["volatility_regime"].fillna(1).astype(int)
    merged["row_tp_rr"] = np.select(
        [regime == 0, regime == 2],
        [sideway_tp_rr, volatile_tp_rr],
        default=tp_rr,
    )
    merged["entry_price"] = merged["close"].astype(float)
    merged["sl_dist"] = merged["atr"].astype(float) * sl_mult

    is_buy = merged["direction"] == 1
    merged["sl_price"] = np.where(
        is_buy,
        merged["entry_price"] - merged["sl_dist"],
        merged["entry_price"] + merged["sl_dist"],
    )
    merged["tp_price"] = np.where(
        is_buy,
        merged["entry_price"] + merged["sl_dist"] * merged["row_tp_rr"],
        merged["entry_price"] - merged["sl_dist"] * merged["row_tp_rr"],
    )
    merged["open_time"] = (
        merged["time"] + pd.Timedelta(hours=broker_gmt)
    ).dt.strftime("%Y.%m.%d %H:%M")
    merged["broker_hour"] = (
        merged["time"] + pd.Timedelta(hours=broker_gmt)
    ).dt.hour
    if include_broker_hours is not None:
        merged = merged[merged["broker_hour"].isin(include_broker_hours)].copy()
    if exclude_broker_hours is not None:
        merged = merged[~merged["broker_hour"].isin(exclude_broker_hours)].copy()
    merged["probability"] = merged["proba"].astype(float)

    out = merged[
        [
            "fold",
            "time",
            "open_time",
            "direction",
            "entry_price",
            "sl_price",
            "tp_price",
            "atr",
            "probability",
            "estimated_profit_r",
        ]
    ].copy()
    return out.sort_values(["fold", "time", "direction"]).reset_index(drop=True)


def write_fold_files(signals: pd.DataFrame, folds: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "all_signals.csv"
    mt5_cols = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]
    signals[mt5_cols].to_csv(all_path, index=False, float_format="%.5f")

    manifest: dict[str, Any] = {
        "all_signals": str(all_path),
        "folds": [],
        "total_signals": int(len(signals)),
    }
    for fold in folds:
        fold_id = int(fold["fold"])
        fold_signals = signals[signals["fold"].astype(int) == fold_id].copy()
        fold_path = out_dir / f"fold_{fold_id:02d}_signals.csv"
        if fold_signals.empty:
            _empty_signal_frame().to_csv(fold_path, index=False)
        else:
            fold_signals[mt5_cols].to_csv(fold_path, index=False, float_format="%.5f")
        manifest["folds"].append(
            {
                "fold": fold_id,
                "train_start": fold.get("train_start"),
                "train_end": fold.get("train_end"),
                "test_start": fold.get("test_start"),
                "test_end": fold.get("test_end"),
                "signals": int(len(fold_signals)),
                "csv": str(fold_path),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MT5 Strategy Tester CSV files per WF fold.")
    parser.add_argument("--wf-report", type=Path, default=ROOT / "outputs/walkforward_report_acc1_sweep_r2_5.json")
    parser.add_argument("--wf-signals", type=Path, default=ROOT / "outputs/walkforward_trades_acc1_sweep_r2_5.csv")
    parser.add_argument("--features", type=Path, default=ROOT / "outputs/historical_features_2022_2026.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/mt5_wf_r2_5")
    parser.add_argument("--min-profit-r", type=float, default=2.5)
    parser.add_argument("--broker-gmt", type=int, default=3)
    parser.add_argument("--sl-mult", type=float, default=1.5)
    parser.add_argument("--tp-rr", type=float, default=5.5)
    parser.add_argument("--volatile-tp-rr", type=float, default=5.0)
    parser.add_argument("--sideway-tp-rr", type=float, default=2.5)
    parser.add_argument("--min-probability", type=float, default=None)
    parser.add_argument("--include-broker-hours", type=str, default=None, help="Comma-separated broker hours to keep, e.g. 0,1,12,13")
    parser.add_argument("--exclude-broker-hours", type=str, default=None, help="Comma-separated broker hours to drop, e.g. 2,3,16,17,20,21")
    parser.add_argument("--side", choices=["all", "buy", "sell"], default="all")
    parser.add_argument("--min-atr-percentile", type=float, default=None)
    parser.add_argument("--max-atr-percentile", type=float, default=None)
    args = parser.parse_args()

    def parse_hours(text: str | None) -> set[int] | None:
        if not text:
            return None
        hours = {int(part.strip()) for part in text.split(",") if part.strip()}
        bad = [hour for hour in hours if hour < 0 or hour > 23]
        if bad:
            raise ValueError(f"Broker hours must be 0..23, got {bad}")
        return hours

    folds = _load_folds(args.wf_report)
    signals = build_signals(
        wf_signals_path=args.wf_signals,
        features_path=args.features,
        min_profit_r=args.min_profit_r,
        broker_gmt=args.broker_gmt,
        sl_mult=args.sl_mult,
        tp_rr=args.tp_rr,
        volatile_tp_rr=args.volatile_tp_rr,
        sideway_tp_rr=args.sideway_tp_rr,
        min_probability=args.min_probability,
        include_broker_hours=parse_hours(args.include_broker_hours),
        exclude_broker_hours=parse_hours(args.exclude_broker_hours),
        side=args.side,
        min_atr_percentile=args.min_atr_percentile,
        max_atr_percentile=args.max_atr_percentile,
    )
    manifest = write_fold_files(signals, folds, args.out_dir)
    print(json.dumps({"out_dir": str(args.out_dir), "total_signals": manifest["total_signals"], "folds": len(folds)}, indent=2))
    for row in manifest["folds"]:
        print(f"fold {row['fold']:02d}: {row['test_start']} -> {row['test_end']} signals={row['signals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
