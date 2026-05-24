#!/usr/bin/env python3
"""
Lightweight ACC2 feature drift check.

This is intentionally independent of Evidently/Airflow dependencies. It compares
the latest live feature window against the recent historical reference window
from the same MT5-exported feature stream and writes a JSON report.

Exit codes:
  0 = OK
  1 = drift warning/block threshold exceeded
  2 = data/input error
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "spread_points",
    "rsi",
    "macd_hist",
    "atr",
    "atr_ratio",
    "atr_percentile",
    "bb_position",
    "adx",
    "price_roc",
    "range_efficiency",
    "tick_volume_zscore",
    "strategy_score",
    "volatility_regime",
    "regime_score",
    "regime_favorable",
    "trade_side",
]


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().astype(float)
    cur = pd.to_numeric(current, errors="coerce").dropna().astype(float)
    if len(ref) < 20 or len(cur) < 20:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(ref, quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1.0), 1e-6)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1.0), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _categorical_drift(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.astype(str).fillna("__nan__")
    cur = current.astype(str).fillna("__nan__")
    keys = sorted(set(ref.unique()).union(set(cur.unique())))
    if not keys:
        return 0.0
    ref_pct = ref.value_counts(normalize=True).reindex(keys, fill_value=0.0)
    cur_pct = cur.value_counts(normalize=True).reindex(keys, fill_value=0.0)
    return float((cur_pct - ref_pct).abs().sum() / 2.0)


def _feature_report(name: str, reference: pd.Series, current: pd.Series) -> dict:
    if name == "trade_side" or not pd.api.types.is_numeric_dtype(reference):
        score = _categorical_drift(reference, current)
        method = "categorical_tvd"
        ref_mean = None
        cur_mean = None
        z_delta = None
    else:
        score = _psi(reference, current)
        method = "psi"
        ref_num = pd.to_numeric(reference, errors="coerce")
        cur_num = pd.to_numeric(current, errors="coerce")
        ref_mean = float(ref_num.mean()) if len(ref_num.dropna()) else None
        cur_mean = float(cur_num.mean()) if len(cur_num.dropna()) else None
        ref_std = float(ref_num.std()) if len(ref_num.dropna()) else 0.0
        z_delta = None
        if ref_mean is not None and cur_mean is not None and ref_std > 1e-12:
            z_delta = float(abs(cur_mean - ref_mean) / ref_std)

    return {
        "feature": name,
        "method": method,
        "score": round(float(score), 6),
        "reference_missing_pct": round(float(reference.isna().mean() * 100.0), 4),
        "current_missing_pct": round(float(current.isna().mean() * 100.0), 4),
        "reference_mean": ref_mean,
        "current_mean": cur_mean,
        "z_delta": z_delta,
    }


def run(args: argparse.Namespace) -> int:
    if not args.features.exists():
        print(f"features file not found: {args.features}")
        return 2

    usecols = None
    header = pd.read_csv(args.features, nrows=0)
    columns = list(header.columns)
    monitored = [c for c in DEFAULT_FEATURES if c in columns]
    if "time" in columns:
        usecols = ["time", *monitored]
    else:
        usecols = monitored
    if not monitored:
        print("no monitored feature columns found")
        return 2

    df = pd.read_csv(args.features, usecols=usecols)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if len(df) < args.reference_rows + args.current_rows:
        print(f"not enough rows: have={len(df)} need={args.reference_rows + args.current_rows}")
        return 2

    current = df.tail(args.current_rows).copy()
    reference = df.iloc[-(args.reference_rows + args.current_rows):-args.current_rows].copy()

    features = [_feature_report(c, reference[c], current[c]) for c in monitored]
    drifted = [
        f["feature"]
        for f in features
        if (
            (f["method"] == "psi" and f["score"] >= args.psi_warn)
            or (f["method"] == "categorical_tvd" and f["score"] >= args.cat_warn)
            or (f.get("z_delta") is not None and f["z_delta"] >= args.z_warn)
        )
    ]
    blocked = [
        f["feature"]
        for f in features
        if (
            (f["method"] == "psi" and f["score"] >= args.psi_block)
            or (f["method"] == "categorical_tvd" and f["score"] >= args.cat_block)
            or (f.get("z_delta") is not None and f["z_delta"] >= args.z_block)
        )
    ]

    report = {
        "generated_at_epoch": time.time(),
        "mode": "acc2_feature_drift",
        "features_file": str(args.features),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "reference_time_min": str(reference["time"].min()) if "time" in reference.columns else "",
        "reference_time_max": str(reference["time"].max()) if "time" in reference.columns else "",
        "current_time_min": str(current["time"].min()) if "time" in current.columns else "",
        "current_time_max": str(current["time"].max()) if "time" in current.columns else "",
        "drift_detected": bool(drifted),
        "blocked": bool(blocked),
        "drifted_features": drifted,
        "blocked_features": blocked,
        "thresholds": {
            "psi_warn": args.psi_warn,
            "psi_block": args.psi_block,
            "cat_warn": args.cat_warn,
            "cat_block": args.cat_block,
            "z_warn": args.z_warn,
            "z_block": args.z_block,
        },
        "features": features,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 78)
    print("ACC2 FEATURE DRIFT CHECK")
    print(f"reference: {report['reference_time_min']} -> {report['reference_time_max']} ({len(reference)} rows)")
    print(f"current  : {report['current_time_min']} -> {report['current_time_max']} ({len(current)} rows)")
    print(f"drifted  : {drifted}")
    print(f"blocked  : {blocked}")
    print(f"report   : {args.out}")
    print("=" * 78)
    return 1 if blocked else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/mt5_full_ict_wyckoff_features_live_acc2_risk2_check_current.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/acc2_demo_1200_risk2_live_check_20260520/drift_report.json"),
    )
    parser.add_argument("--reference-rows", type=int, default=5000)
    parser.add_argument("--current-rows", type=int, default=288)
    parser.add_argument("--psi-warn", type=float, default=0.20)
    parser.add_argument("--psi-block", type=float, default=0.35)
    parser.add_argument("--cat-warn", type=float, default=0.25)
    parser.add_argument("--cat-block", type=float, default=0.40)
    parser.add_argument("--z-warn", type=float, default=2.0)
    parser.add_argument("--z-block", type=float, default=3.0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
