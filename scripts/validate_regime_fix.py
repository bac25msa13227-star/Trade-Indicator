#!/usr/bin/env python3
"""
Validate Regime Fix: compare OLD vs NEW volatility_regime thresholds.

Shows:
 1. Regime distribution  (old 0.95 vs new 0.80 threshold)
 2. SL/TP impact per regime (from config defaults)
 3. Per-fold WF simulation: old (hardcoded=1) vs new (real regime)
 4. Live-parity check: regime distribution over recent bars (last 2 weeks)

Usage:
  python scripts/validate_regime_fix.py [--full-wf]

  --full-wf   Run full WF backtest comparison (slow, ~5-10 min).
              Without this flag, only distribution + SL/TP analysis is shown (fast).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import pandas as pd

from xauusd_ai.features.scalp_dataset import build_scalp_dataset, infer_scalp_volatility_regime
from xauusd_ai.features.scalp_features import build_all_scalp_features

# ─── Constants ────────────────────────────────────────────────────────────────
OLD_SIDEWAY_THR = 0.95
NEW_SIDEWAY_THR = 0.80
STRONG_THR = 1.30

REGIME_LABELS = {0: "sideway", 1: "normal", 2: "strong"}


def _regime_old(expansion: np.ndarray) -> np.ndarray:
    """Old threshold: sideway <= 0.95."""
    return np.where(expansion <= OLD_SIDEWAY_THR, 0, np.where(expansion >= STRONG_THR, 2, 1))


def _regime_new(expansion: np.ndarray) -> np.ndarray:
    """New threshold: sideway <= 0.80."""
    return np.where(expansion <= NEW_SIDEWAY_THR, 0, np.where(expansion >= STRONG_THR, 2, 1))


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze_distribution(ds: pd.DataFrame):
    """Compare old vs new regime distribution."""
    print_section("1. REGIME DISTRIBUTION: OLD (thr=0.95) vs NEW (thr=0.80)")

    exp = ds["ms_atr_expansion"].values

    old_r = _regime_old(exp)
    new_r = _regime_new(exp)

    n = len(ds)
    for label_name, code in [("sideway (0)", 0), ("normal  (1)", 1), ("strong  (2)", 2)]:
        old_pct = (old_r == code).sum() / n * 100
        new_pct = (new_r == code).sum() / n * 100
        delta = new_pct - old_pct
        print(f"  {label_name}:  OLD={old_pct:5.1f}%   NEW={new_pct:5.1f}%   Δ={delta:+5.1f}%")

    # Also show what the hardcoded=1 case looked like
    print(f"\n  WF (hardcoded=1): sideway=0.0%  normal=100.0%  strong=0.0%")
    print(f"  → WF always used normal SL/TP, never sideway or strong adjustments")

    return old_r, new_r


def analyze_sltp_impact():
    """Show SL/TP multipliers per regime."""
    print_section("2. SL/TP IMPACT PER REGIME (from config defaults)")

    from xauusd_ai.config import load_settings
    configs = [
        ("ACC1", Path("configs/live_acc1_scalp_m1.yaml")),
        ("ACC2", Path("configs/live_acc2_scalp_m1.yaml")),
    ]
    for name, cfg_path in configs:
        if not cfg_path.exists():
            print(f"  {name}: config not found at {cfg_path}")
            continue
        s = load_settings(cfg_path)
        r = s.risk
        print(f"\n  {name} ({cfg_path.name}):")
        print(f"    {'regime':>10}  {'SL mult':>10}  {'TP RR':>10}  {'risk_mult':>10}")
        print(f"    {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
        print(f"    {'sideway':>10}  {r.sideway_sl_atr_multiple:>10.2f}  {r.sideway_take_profit_rr:>10.2f}  {r.sideway_risk_multiplier:>10.2f}")
        print(f"    {'normal':>10}  {r.stop_loss_atr_multiple:>10.2f}  {r.take_profit_rr:>10.2f}  {r.normal_risk_multiplier:>10.2f}")
        print(f"    {'strong':>10}  {r.volatile_sl_atr_multiple:>10.2f}  {r.volatile_take_profit_rr:>10.2f}  {r.strong_volatility_risk_multiplier:>10.2f}")


def analyze_recent_bars(ds: pd.DataFrame, old_r: np.ndarray, new_r: np.ndarray):
    """Show regime breakdown for most recent period (simulates 'what live would see')."""
    print_section("3. RECENT 2-WEEK WINDOW (simulating live conditions)")

    if "time" in ds.columns:
        ds_t = ds.copy()
        ds_t["time"] = pd.to_datetime(ds_t["time"])
        cutoff = ds_t["time"].max() - pd.Timedelta(days=14)
        mask = ds_t["time"] >= cutoff
    else:
        # Fallback: last 20,000 M1 bars ≈ 2 weeks
        mask = np.zeros(len(ds), dtype=bool)
        mask[-min(20_000, len(ds)):] = True

    n_recent = mask.sum()
    print(f"  Bars in window: {n_recent:,}")

    for label_name, code in [("sideway (0)", 0), ("normal  (1)", 1), ("strong  (2)", 2)]:
        old_pct = (old_r[mask] == code).sum() / n_recent * 100
        new_pct = (new_r[mask] == code).sum() / n_recent * 100
        print(f"  {label_name}:  OLD={old_pct:5.1f}%   NEW={new_pct:5.1f}%")

    # Show ms_atr_expansion stats for recent window
    exp = ds.loc[mask, "ms_atr_expansion"]
    print(f"\n  ms_atr_expansion stats (recent window):")
    print(f"    mean={exp.mean():.3f}  median={exp.median():.3f}  "
          f"p25={exp.quantile(0.25):.3f}  p75={exp.quantile(0.75):.3f}")
    print(f"    min={exp.min():.3f}  max={exp.max():.3f}")
    print(f"    % <= 0.80: {(exp <= 0.80).mean()*100:.1f}%")
    print(f"    % <= 0.95: {(exp <= 0.95).mean()*100:.1f}%")


def run_full_wf_comparison(ds: pd.DataFrame):
    """Run WF with old (hardcoded=1) vs new (real regime) and compare results."""
    print_section("4. FULL WF COMPARISON: HARDCODED vs REAL REGIME")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    from xauusd_ai.model.scalp_model import CalibratedDirModel
    from xauusd_ai.features.scalp_features import SCALP_FEATURE_COLUMNS
    from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
    from xauusd_ai.execution.risk import RiskManager
    from xauusd_ai.config import load_settings

    cfg_path = Path("configs/live_acc2_scalp_m1.yaml")
    if not cfg_path.exists():
        cfg_path = Path("configs/live_acc1_scalp_m1.yaml")
    settings = load_settings(cfg_path)

    TRAIN_SIZE = 250_000
    TEST_SIZE  = 50_000
    STEP       = 50_000
    THR_BUY    = 0.60
    THR_SELL   = 0.58
    INITIAL_BAL = 200.0

    n = len(ds)
    folds = []
    s = 0
    while s + TRAIN_SIZE + TEST_SIZE <= n:
        folds.append((s, s + TRAIN_SIZE, s + TRAIN_SIZE + TEST_SIZE))
        s += STEP

    if not folds:
        print("  Not enough data for WF folds. Need at least 300K M1 bars.")
        return

    print(f"  Folds: {len(folds)}, train={TRAIN_SIZE:,}, test={TEST_SIZE:,}")

    results = []

    for i, (start, train_end, test_end) in enumerate(folds):
        tr = ds.iloc[start:train_end]
        te = ds.iloc[train_end:test_end].copy().reset_index(drop=True)

        # Train dual model (simplified — same as WF scripts)
        def _train_dir(data, direction):
            mask = data["expected_direction"] == direction
            sub = data[mask]
            if len(sub) < 500:
                return None, None
            X = sub[SCALP_FEATURE_COLUMNS].fillna(0).values
            y = sub["target"].values.astype(int)
            sc = StandardScaler()
            X_s = sc.fit_transform(X)
            mdl = HistGradientBoostingClassifier(
                max_iter=500, max_leaf_nodes=31, learning_rate=0.05,
                min_samples_leaf=200, max_depth=6, l2_regularization=2.0,
                random_state=42,
            )
            mdl.fit(X_s, y)
            val_size = max(int(len(X_s) * 0.15), 500)
            raw_p = mdl.predict_proba(X_s[-val_size:])[:, 1]
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(raw_p, y[-val_size:])
            return CalibratedDirModel(mdl, iso, sc, direction), sc

        buy_m, _ = _train_dir(tr, 1)
        sell_m, _ = _train_dir(tr, -1)

        if buy_m is None and sell_m is None:
            continue

        # Generate predictions
        X_te = te[SCALP_FEATURE_COLUMNS].fillna(0).values
        dirs = te["expected_direction"].values
        buy_p = np.zeros(len(te))
        sell_p = np.zeros(len(te))
        if buy_m is not None:
            buy_mask = dirs == 1
            if buy_mask.any():
                buy_p[buy_mask] = buy_m.predict_win_proba(X_te[buy_mask])
        if sell_m is not None:
            sell_mask = dirs == -1
            if sell_mask.any():
                sell_p[sell_mask] = sell_m.predict_win_proba(X_te[sell_mask])

        for scenario, regime_override in [("hardcoded_1", True), ("real_regime", False)]:
            preds = te.copy()
            preds["split"] = "test"
            preds["prediction"] = 0
            buy_sig = (dirs == 1) & (buy_p >= THR_BUY)
            sell_sig = (dirs == -1) & (sell_p >= THR_SELL)
            preds.loc[buy_sig, "prediction"] = 1
            preds.loc[sell_sig, "prediction"] = 1
            preds["probability"] = np.where(dirs == -1, sell_p, buy_p)
            preds["trade_side"] = np.where(dirs == -1, "sell", "buy")
            preds["strategy_score"] = 1.0
            preds["trend_alignment"] = 1
            preds["adx"] = 25.0

            if regime_override:
                preds["volatility_regime"] = 1
            # else: keep the real volatility_regime from build_scalp_dataset

            sim_s = settings.model_copy(deep=True)
            sim_s.training.backtest_initial_balance = INITIAL_BAL
            sim_s.strategy.adx_gate_enabled = False
            sim_s.strategy.min_strategy_score = 0.0
            sim_s.strategy.require_trend_alignment = False
            sim_s.training.label_horizon = 8
            sim_s.risk.compound_cap = 25.0
            sim_s.risk.consecutive_loss_pause_count = 2
            sim_s.risk.consecutive_loss_cooldown_bars = 4

            try:
                sim = simulate_dynamic_concurrent_backtest(
                    preds, sim_s, RiskManager(sim_s), label="test", compound=True)
                rep = sim.report
                results.append({
                    "fold": i + 1,
                    "scenario": scenario,
                    "trades": int(rep.get("trades", 0) or 0),
                    "wins": int(rep.get("wins", 0) or 0),
                    "pf": float(rep.get("profit_factor", 0) or 0),
                    "net": float(rep.get("net_profit", 0) or 0),
                    "max_dd_pct": float(rep.get("max_drawdown_pct", 0) or 0),
                    "final_bal": float(rep.get("final_balance", INITIAL_BAL) or INITIAL_BAL),
                })
            except Exception as e:
                print(f"  Fold {i+1} {scenario}: sim error: {e}")

    if not results:
        print("  No results produced.")
        return

    df = pd.DataFrame(results)
    print(f"\n  {'Fold':>4} | {'Scenario':>12} | {'Trades':>6} | {'Wins':>4} | {'PF':>6} | {'Net$':>10} | {'DD%':>6} | {'Final$':>10}")
    print(f"  {'─'*4} | {'─'*12} | {'─'*6} | {'─'*4} | {'─'*6} | {'─'*10} | {'─'*6} | {'─'*10}")
    for _, r in df.iterrows():
        print(f"  {r['fold']:>4} | {r['scenario']:>12} | {r['trades']:>6} | {r['wins']:>4} | "
              f"{r['pf']:>6.2f} | {r['net']:>10.2f} | {r['max_dd_pct']:>6.1f} | {r['final_bal']:>10.2f}")

    # Summary comparison
    print_section("5. SUMMARY: HARDCODED vs REAL REGIME")
    for scenario in ["hardcoded_1", "real_regime"]:
        sub = df[df["scenario"] == scenario]
        total_trades = sub["trades"].sum()
        avg_pf = sub.loc[sub["pf"] > 0, "pf"].mean() if (sub["pf"] > 0).any() else 0
        total_net = sub["net"].sum()
        avg_dd = sub["max_dd_pct"].mean()
        profitable_folds = (sub["net"] > 0).sum()
        print(f"  {scenario:>12}: trades={total_trades:>5}  avg_PF={avg_pf:.2f}  "
              f"total_net=${total_net:>10,.2f}  avg_DD={avg_dd:.1f}%  "
              f"profitable_folds={profitable_folds}/{len(sub)}")


def main():
    parser = argparse.ArgumentParser(description="Validate volatility regime fix")
    parser.add_argument("--full-wf", action="store_true", help="Run full WF comparison (slow)")
    parser.add_argument("--start-date", default="2023-06-01", help="Dataset start date")
    args = parser.parse_args()

    print("Loading dataset…")
    ds = build_scalp_dataset(start_date=args.start_date)
    print(f"Dataset: {len(ds):,} rows")

    if "ms_atr_expansion" not in ds.columns:
        print("ERROR: ms_atr_expansion not in dataset. Check build_all_scalp_features().")
        sys.exit(1)

    old_r, new_r = analyze_distribution(ds)
    analyze_sltp_impact()
    analyze_recent_bars(ds, old_r, new_r)

    if args.full_wf:
        run_full_wf_comparison(ds)
    else:
        print(f"\n{'='*70}")
        print("  Tip: Run with --full-wf to see fold-by-fold WF comparison")
        print(f"  python scripts/validate_regime_fix.py --full-wf")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
