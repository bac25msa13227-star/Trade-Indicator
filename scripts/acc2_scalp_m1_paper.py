#!/usr/bin/env python3
"""
acc2_scalp_m1_paper.py
=======================
Paper trade simulation for M1 scalp model.

Strategy: WF-style holdout — train on bars BEFORE the test window,
test on the last TEST_SIZE bars (genuine out-of-sample).

The saved DualScalpModel pkl is trained on the full latest dataset
(for live use). The paper trade trains a fresh model on data *excluding*
the test window and evaluates on those withheld bars.

Flow:
  1. Build scalp dataset (full history)
  2. Holdout split: train=last (TRAIN_SIZE+TEST_SIZE) excl last TEST_SIZE
                    test =last TEST_SIZE bars
  3. Train a fresh paper-trade model on the train split
  4. Build signals via DualScalpModel.build_signal_df()
  5. Run simulate_dynamic_concurrent_backtest() with 0.3% risk
  6. Print paper trade report + save signals to CSV

Usage:
  cd "Trade Indicator"
  source .venv2/bin/activate
  PYTHONPATH=src python scripts/acc2_scalp_m1_paper.py

  # Override test window size:
  PYTHONPATH=src python scripts/acc2_scalp_m1_paper.py --test-bars 20000
"""
from __future__ import annotations

import argparse, json, pickle, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from xauusd_ai.config import load_settings
from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.scalp_dataset import build_scalp_dataset
from xauusd_ai.features.scalp_features import SCALP_FEATURE_COLUMNS
from xauusd_ai.model.scalp_model import CalibratedDirModel, DualScalpModel

REPO    = Path(__file__).parent.parent
OUT_DIR = REPO / "outputs"
# Use live_acc2_scalp.yaml as settings base for the backtest engine.
# live_acc2_scalp_m1.yaml is for live bot configuration (MT5 execution).
# The simulation engine only needs risk/strategy scalar parameters, not
# execution_timeframe or csv paths — _run_sim overrides all critical fields.
CONFIG  = REPO / "configs" / "live_acc2_scalp.yaml"

# ── Paper trade parameters ─────────────────────────────────────────────────
DS_START     = "2019-01-01"
SL_ATR_MULT  = 0.8
TP_RR        = 1.5
MAX_HORIZON  = 8
TRAIN_SIZE   = 250_000    # same as save_model / WF
TEST_SIZE    = 50_000     # ~8 weeks — same as WF fold
INITIAL_BAL  = 200.0
RISK         = 0.003      # 0.3% per trade


def _make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=5,
        min_samples_leaf=50, l2_regularization=1.0, max_bins=63,
        early_stopping=True, validation_fraction=0.10,
        n_iter_no_change=20, random_state=42,
    )


def _train_dir(df, direction: int) -> CalibratedDirModel | None:
    label = "BUY" if direction == 1 else "SELL"
    sub   = df[df["expected_direction"] == direction].copy()
    n     = len(sub)
    if n < 2000:
        print(f"  [{label}] skip: only {n} rows", flush=True)
        return None
    X = sub[SCALP_FEATURE_COLUMNS].fillna(0).values
    y = sub["target"].values
    pos = int(y.sum()); neg = n - pos
    if pos < 200 or neg < 200:
        return None
    print(f"  [{label}] n={n:,}  tp_rate={pos/n*100:.1f}%", flush=True)
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    cw = np.where(y == 1, float(neg) / float(pos), 1.0).astype(float)
    decay_half = n * 0.30
    tw = np.exp(np.log(2) * np.arange(n) / decay_half)
    tw /= tw.mean()
    sw = cw * tw; sw /= sw.mean()
    model = _make_model()
    model.fit(X_s, y, sample_weight=sw)
    val_size = max(int(n * 0.15), 1000)
    try:
        X_val = X_s[-val_size:]; y_val = y[-val_size:]
        raw_p = model.predict_proba(X_val)[:, 1]
        iso   = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw_p, y_val)
        print(f"  [{label}] ✓ calibrated", flush=True)
        return CalibratedDirModel(model, iso, scaler, direction)
    except Exception as e:
        print(f"  [{label}] calibration failed ({e}), using raw", flush=True)
        iso_id = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        raw_all = model.predict_proba(X_s[-500:])[:, 1]
        iso_id.fit(raw_all, y[-500:])
        return CalibratedDirModel(model, iso_id, scaler, direction)


def _run_sim(preds, settings, risk: float, initial_bal: float) -> dict:
    """Run backtest — identical parameters as acc2_scalp_m1_wf.py._run_sim."""
    s = settings.model_copy(deep=True)
    s.training.backtest_initial_balance             = initial_bal
    s.strategy.adx_gate_enabled                     = False
    s.strategy.min_strategy_score                   = 0.0
    s.strategy.sideway_min_strategy_score           = 0.0
    s.strategy.strong_volatility_min_strategy_score = 0.0
    s.strategy.require_trend_alignment               = False
    s.strategy.silver_bullet_enabled                 = False
    s.training.label_horizon                         = MAX_HORIZON
    s.risk.risk_per_trade                  = risk
    s.risk.max_risk_fraction               = risk + 0.02
    s.risk.take_profit_rr                  = TP_RR
    s.risk.sideway_take_profit_rr          = TP_RR * 0.8
    s.risk.volatile_take_profit_rr         = TP_RR * 1.3
    s.risk.stop_loss_atr_multiple          = SL_ATR_MULT
    s.risk.min_confidence                  = 0.55
    s.risk.compound_cap                    = 25.0
    s.risk.consecutive_loss_pause_count    = 2
    s.risk.consecutive_loss_cooldown_bars  = 4
    s.risk.daily_loss_limit_pct            = 0.0
    s.risk.anti_martingale_factor          = 0.3
    s.risk.max_open_positions              = 3
    s.risk.partial_tp_enabled             = True
    s.risk.partial_tp_rr                  = TP_RR * 0.5
    s.risk.partial_tp_pct                 = 0.60
    s.execution.close_opposite_on_signal  = True
    sim = simulate_dynamic_concurrent_backtest(
        preds, s, RiskManager(s), label="test", compound=True,
    )
    rep   = sim.report
    n_tr  = int(rep.get("trades", 0) or 0)
    n_win = int(rep.get("wins", 0) or 0)
    n_los = int(rep.get("losses", 0) or 0)
    pf_   = float(rep.get("profit_factor", 0) or 0)
    if pf_ == 0 and n_win > 0 and n_los == 0:
        pf_ = float("inf")
    return {
        "trades":    n_tr,
        "wins":      n_win,
        "losses":    n_los,
        "wr":        float((rep.get("win_rate", 0) or 0) * 100),
        "pf":        pf_,
        "ending_balance": float(rep.get("ending_balance", initial_bal)),
        "max_drawdown_pct": abs(float(rep.get("max_drawdown_pct", 0) or 0)),
    }


def main(test_bars: int = TEST_SIZE) -> None:
    meta_path  = OUT_DIR / "acc2_scalp_m1_model_meta.json"
    train_end_saved = ""
    if meta_path.exists():
        train_end_saved = json.loads(meta_path.read_text()).get("train_end", "")

    print("=" * 80)
    print("  ACC2 M1 Scalp — PAPER TRADE SIMULATION (WF holdout)")
    print(f"  Holdout: train on last {TRAIN_SIZE:,}+{test_bars:,} bars excl. last {test_bars:,}")
    print(f"  Test on last {test_bars:,} M1 bars (~{test_bars//7200:.0f} weeks)  — GENUINE out-of-sample")
    print(f"  Risk: {RISK*100:.1f}%  |  TP={TP_RR}R  SL={SL_ATR_MULT}×ATR  |  Max positions: 2")
    if train_end_saved:
        print(f"  Live model (saved pkl) trained to: {train_end_saved}")
    print("=" * 80)

    # ── 1. Build dataset ──────────────────────────────────────────────────
    print(f"\n[1] Building scalp dataset from {DS_START}…", flush=True)
    ds = build_scalp_dataset(
        start_date=DS_START,
        sl_atr_mult=SL_ATR_MULT,
        tp_rr=TP_RR,
        max_horizon=MAX_HORIZON,
    )
    total = len(ds)
    print(f"    {total:,} M1 rows  "
          f"[{str(ds['time'].iloc[0])[:10]} → {str(ds['time'].iloc[-1])[:10]}]")

    need = TRAIN_SIZE + test_bars
    if total < need:
        print(f"ERROR: need {need:,} rows, have {total:,}. Aborting.")
        sys.exit(1)

    # ── 2. WF holdout split ───────────────────────────────────────────────
    # test  = last test_bars rows (withheld from training)
    # train = TRAIN_SIZE rows immediately before test
    test_df  = ds.iloc[-test_bars:].copy().reset_index(drop=True)
    train_df = ds.iloc[-(TRAIN_SIZE + test_bars):-test_bars].copy().reset_index(drop=True)

    t_start = str(test_df["time"].iloc[0])[:10]
    t_end   = str(test_df["time"].iloc[-1])[:10]
    tr_start= str(train_df["time"].iloc[0])[:10]
    tr_end  = str(train_df["time"].iloc[-1])[:10]
    weeks   = len(test_df) / 7200

    print(f"\n[2] Holdout split:")
    print(f"    Train: {len(train_df):,} bars  [{tr_start} → {tr_end}]")
    print(f"    Test:  {len(test_df):,} bars  [{t_start} → {t_end}]  ({weeks:.1f} weeks)")
    print(f"    Test TP hit rate: {test_df['target'].mean()*100:.1f}%")

    # ── 3. Train paper-trade model ────────────────────────────────────────
    print(f"\n[3] Training paper-trade model on holdout train split…", flush=True)
    buy_model  = _train_dir(train_df,  1)
    sell_model = _train_dir(train_df, -1)
    if buy_model is None and sell_model is None:
        print("ERROR: both models failed. Aborting.")
        sys.exit(1)

    paper_model = DualScalpModel(
        buy_model       = buy_model,
        sell_model      = sell_model,
        feature_columns = SCALP_FEATURE_COLUMNS,
        thr_buy         = 0.58,
        thr_sell        = 0.55,
        train_end       = tr_end,
    )
    print(f"    {paper_model}")

    # ── 4. Build signals on test window ───────────────────────────────────
    print(f"\n[4] Building signals on test window…", flush=True)
    preds  = paper_model.build_signal_df(test_df)
    n_total= int(preds["prediction"].sum())
    n_buy  = int(((preds["prediction"]==1) & (test_df["expected_direction"]==1)).sum())
    n_sell = int(((preds["prediction"]==1) & (test_df["expected_direction"]==-1)).sum())
    print(f"    Total signals: {n_total:,}  (BUY: {n_buy:,}  SELL: {n_sell:,})")
    print(f"    Signal rate: {n_total/len(test_df)*100:.1f}% of bars")
    if n_total == 0:
        print("ERROR: zero signals — check model thresholds.")
        sys.exit(1)

    # ── 5. Run simulation ─────────────────────────────────────────────────
    print(f"\n[5] Running paper trade simulation (risk={RISK*100:.1f}%)…", flush=True)
    settings = load_settings(CONFIG)
    report   = _run_sim(preds, settings, RISK, INITIAL_BAL)

    # ── 6. Print results ──────────────────────────────────────────────────
    trades   = report["trades"]
    wins     = report["wins"]
    losses   = report["losses"]
    wr       = report["wr"]
    pf       = report["pf"]
    end_bal  = report["ending_balance"]
    ret_pct  = (end_bal - INITIAL_BAL) / INITIAL_BAL * 100
    dd_pct   = report["max_drawdown_pct"]

    yrs      = max(weeks / 52, 1 / 52)
    f        = 1 + ret_pct / 100
    ann_pct  = (f ** (1 / yrs) - 1) * 100 if f > 0 else ret_pct / yrs
    edge_pct = (wins * TP_RR - losses * 1.0) * RISK * 100
    edge_ann = edge_pct / yrs

    print(f"\n{'='*80}")
    print(f"  PAPER TRADE RESULTS — {len(test_df):,} M1 bars  [{t_start} → {t_end}]")
    print(f"{'='*80}")
    print(f"  Period:          {weeks:.1f} weeks ({yrs:.3f} yr)")
    print(f"  Total signals:   {n_total:,}  (BUY {n_buy:,}, SELL {n_sell:,})")
    print(f"  Executed trades: {trades:,}  (~{trades/max(weeks,0.01):.0f} trades/week)")
    print(f"  Wins / Losses:   {wins} / {losses}")
    print(f"  Win Rate:        {wr:.1f}%")
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
    print(f"  Profit Factor:   {pf_str}")
    print(f"  Return:          {ret_pct:+.1f}%  (compound @ {RISK*100:.1f}% risk)")
    print(f"  Ann. Return:     {ann_pct:+.1f}%/yr  (compound)")
    print(f"  Max Drawdown:    {dd_pct:.1f}%")
    print(f"  ─────────────────────────────────────")
    print(f"  Non-compound edge: {edge_pct:+.1f}%  /  {edge_ann:+.0f}%/yr  "
          f"(1% sizing: {edge_pct/RISK*0.01:+.0f}%  /  {edge_ann/RISK*0.01:+.0f}%/yr)")
    print(f"{'='*80}")

    ok = (wr >= 51.0 and (pf >= 1.4 or pf == float("inf")) and ret_pct > 0)
    verdict = "✅  PASS — model edge confirmed on out-of-sample holdout" if ok else \
              "⚠️  CHECK — review WR/PF before live deployment"
    print(f"\n  {verdict}")
    if train_end_saved:
        print(f"  Live model (saved pkl) trained to: {train_end_saved}  → ready for go-live")
    print()

    # ── 7. Save signals to CSV ────────────────────────────────────────────
    sig_csv = OUT_DIR / "paper_trade_signals_acc2_scalp_m1.csv"
    active  = preds[preds["prediction"] == 1].copy()
    active["paper_trade_time"] = pd.Timestamp.utcnow().isoformat()
    active.to_csv(str(sig_csv), index=False, mode="a",
                  header=not sig_csv.exists())
    print(f"  Signals saved to {sig_csv.name}  ({len(active):,} rows)")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ACC2 M1 Scalp paper trade simulation")
    ap.add_argument("--test-bars", type=int, default=TEST_SIZE,
                    help=f"Number of M1 bars to test on (default: {TEST_SIZE})")
    args = ap.parse_args()
    main(test_bars=args.test_bars)
