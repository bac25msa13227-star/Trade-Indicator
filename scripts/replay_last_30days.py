#!/usr/bin/env python3
"""
replay_last_30days.py
=====================
Lớp 3 — Pre-deploy replay test.

Mục đích:
  - Trước mỗi lần deploy, chạy lại N ngày gần nhất qua engine WF với LIVE config
    (configs/live_acc1.yaml), dùng MODEL FROZEN hiện tại.
  - So với ngưỡng tối thiểu (return %, max_dd %).
  - Exit code != 0 → block deploy.

Khác show_combo133_daily.py:
  - KHÔNG retrain (load .pkl frozen).
  - KHÔNG đụng vào WF baseline 27 folds.
  - Chỉ replay window N ngày để ước lượng "bot SẼ chạy ra sao trong 30 ngày qua
    nếu config hiện tại được áp dụng".

Exit codes:
  0 = PASS (return >= min_return_pct AND max_dd <= max_dd_pct)
  1 = FAIL (vi phạm threshold)
  2 = ERROR (model/data thiếu, không chạy được)

Usage:
  python scripts/replay_last_30days.py
  python scripts/replay_last_30days.py --days 30 --min-return-pct -25 --max-dd-pct 30
  python scripts/replay_last_30days.py --config configs/live_acc1.yaml
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest  # noqa: E402
from xauusd_ai.config import load_settings  # noqa: E402
from xauusd_ai.data.market_data import MarketDataService  # noqa: E402
from xauusd_ai.execution.risk import RiskManager  # noqa: E402
from xauusd_ai.features.dataset import (  # noqa: E402
    FEATURE_COLUMNS,
    prepare_training_dataset,
)
from xauusd_ai.strategies.hybrid import HybridStrategy  # noqa: E402


def replay(
    config_path: Path,
    days: int,
    min_return_pct: float,
    max_dd_pct: float,
    starting_bal: float,
) -> int:
    print("=" * 78)
    print("  PRE-DEPLOY REPLAY TEST")
    print(f"  Config        : {config_path}")
    print(f"  Window        : last {days} days")
    print(f"  Pass criteria : return >= {min_return_pct:+.1f}%  AND  max_dd <= {max_dd_pct:.1f}%")
    print("=" * 78)

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}", file=sys.stderr)
        return 2

    settings = load_settings(config_path)

    # ── Resolve model & scaler ──
    model_path = Path(settings.app.model_path)
    scaler_path = Path(settings.app.scaler_path)
    if not model_path.exists() or not scaler_path.exists():
        print(f"❌ Model/scaler missing:\n    {model_path}\n    {scaler_path}", file=sys.stderr)
        return 2

    print(f"\n[1/5] Loading model: {model_path.name}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feat_mask = None
    meta_path = Path(settings.app.model_meta_path)
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(meta.get("feature_mask"), list) and meta["feature_mask"]:
            feat_mask = np.asarray(meta["feature_mask"], dtype=bool)
            print(f"      feature_mask: {feat_mask.sum()}/{len(feat_mask)} active")

    # ── Load market data (CSV folder) ──
    print("[2/5] Loading market data (CSV folder)...")
    data_service = MarketDataService(settings)
    frames = data_service.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)

    print("[3/5] Building feature dataset...")
    strategy = HybridStrategy(settings)
    full_ds = prepare_training_dataset(settings, frames, strategy)
    print(f"      {len(full_ds):,} rows ({full_ds['time'].min().date()} → {full_ds['time'].max().date()})")

    # ── Window = last N days ──
    end_ts = full_ds["time"].max()
    start_ts = end_ts - pd.Timedelta(days=days)
    window = full_ds[full_ds["time"] >= start_ts].copy().reset_index(drop=True)
    if len(window) < 200:
        print(f"❌ Window too small: {len(window)} rows (need ≥200)", file=sys.stderr)
        return 2
    print(f"      Window rows: {len(window):,} ({window['time'].min()} → {window['time'].max()})")

    # ── Predict ──
    print("[4/5] Running model predictions...")
    X = scaler.transform(window[FEATURE_COLUMNS])
    if feat_mask is not None and len(feat_mask) == X.shape[1]:
        X = X[:, feat_mask]
        print(f"      applied feat_mask → {X.shape[1]} features")
    proba = model.predict_proba(X)[:, 1]

    # Threshold from model meta if available, else min_confidence
    threshold = float(getattr(settings.risk, "min_confidence", 0.70))
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "decision_threshold" in meta:
            threshold = float(meta["decision_threshold"])
    print(f"      decision threshold: {threshold:.3f}")

    window["split"] = "test"
    window["probability"] = proba
    window["prediction"] = (proba >= threshold).astype(int)
    # trade_side: same logic as live (buy when strategy_score >= 0)
    if "expected_direction" in window.columns:
        window["trade_side"] = np.where(window["expected_direction"] > 0, "buy", "sell")
    else:
        window["trade_side"] = np.where(window["strategy_score"] >= 0, "buy", "sell")

    # ── Simulate ──
    print("[5/5] Simulating with live_acc1.yaml settings...")
    sim_settings = settings.model_copy(deep=True)
    sim_settings.training.backtest_initial_balance = starting_bal
    risk_mgr = RiskManager(sim_settings)

    result = simulate_dynamic_concurrent_backtest(window, sim_settings, risk_mgr)
    rep = result.report

    trades = int(rep.get("trades", 0))
    win_rate = float(rep.get("win_rate", 0.0))
    end_bal = float(rep.get("ending_balance", starting_bal))
    pnl = end_bal - starting_bal
    return_pct = (pnl / starting_bal) * 100.0
    max_dd_pct_actual = float(rep.get("max_drawdown_pct", 0.0))

    print()
    print("─" * 78)
    print(f"  Trades        : {trades}")
    print(f"  Win rate      : {win_rate:.1%}")
    print(f"  Starting bal  : ${starting_bal:.2f}")
    print(f"  Ending bal    : ${end_bal:.2f}")
    print(f"  P&L           : {'+' if pnl >= 0 else ''}{pnl:.2f}  ({return_pct:+.2f}%)")
    print(f"  Max drawdown  : {max_dd_pct_actual:.2f}%")
    print("─" * 78)

    # ── Verdict ──
    fail_reasons: list[str] = []
    if return_pct < min_return_pct:
        fail_reasons.append(f"return {return_pct:+.2f}% < threshold {min_return_pct:+.2f}%")
    if abs(max_dd_pct_actual) > max_dd_pct:
        fail_reasons.append(f"max_dd {abs(max_dd_pct_actual):.2f}% > threshold {max_dd_pct:.2f}%")
    if trades < 5:
        fail_reasons.append(f"too few trades ({trades}) — config may be too restrictive")

    if fail_reasons:
        print("  RESULT: ❌ FAIL — Do NOT deploy:")
        for r in fail_reasons:
            print(f"            • {r}")
        print("=" * 78)
        return 1

    print("  RESULT: ✅ PASS — Replay results within acceptable thresholds.")
    print("=" * 78)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/live_acc1.yaml"))
    ap.add_argument("--days", type=int, default=30, help="Replay window in days (default: 30)")
    ap.add_argument("--min-return-pct", type=float, default=-25.0,
                    help="Minimum acceptable return (default: -25%%, baseline WF worst fold = -29%%)")
    ap.add_argument("--max-dd-pct", type=float, default=30.0,
                    help="Maximum acceptable drawdown (default: 30%%, baseline WF avg = 19%%)")
    ap.add_argument("--starting-balance", type=float, default=200.0)
    args = ap.parse_args()

    return replay(
        args.config,
        args.days,
        args.min_return_pct,
        args.max_dd_pct,
        args.starting_balance,
    )


if __name__ == "__main__":
    sys.exit(main())
