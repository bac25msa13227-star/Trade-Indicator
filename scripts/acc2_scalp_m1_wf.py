#!/usr/bin/env python3
"""
ACC2 Scalping Walk-Forward — M1 native architecture (EXTENDED)
==============================================================
Kiến trúc scalping thực sự:
  ▶ M1 data (XAUUSDm_M1.csv) — execution timeframe
  ▶ M5 context (direction bias, RSI) — next higher TF
  ▶ 54 scalp-specific features: order flow, microstructure, structure, session
  ▶ Label: SL/TP hit within 8 M1 bars (8 phút) — horizon ≤10 bars
  ▶ Dual BUY/SELL model (HistGBDT + isotonic calibration)
  ▶ WF: 18 folds, train=250K M1 bars (~6 tháng), test=50K (~8 tuần)

  Mục tiêu CHÍNH: kiểm trầ fold 6 của M5-model (Aug 2023-Jul 2024 gold ATH):
    - M5 model: WR=0%, PF=0.00, -13% return — hoàn toàn thất bại
    - M1 scalp: liệu kiến trúc 8-bar horizon có survive không?

  Phạm vi: Sep 2023 → Mar 2026 (18 folds, 8-week test windows)
  Include: ATH gold period, post-ATH correction, 2025-2026 recent
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

from xauusd_ai.config import load_settings
from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.scalp_dataset import build_scalp_dataset
from xauusd_ai.features.scalp_features import SCALP_FEATURE_COLUMNS
from xauusd_ai.model.scalp_model import CalibratedDirModel

REPO = Path(__file__).parent.parent
CONFIG = REPO / "configs" / "live_acc2_scalp.yaml"

# ── WF parameters (M1 bars) ──────────────────────────────────────────────────
TRAIN_SIZE   = 250_000    # ~6 months M1 bars  (prev: 120K)
TEST_SIZE    =  50_000    # ~8 weeks M1 bars   (prev: 20K)
STEP_SIZE    =  50_000    # step 8 weeks
N_FOLDS      = 18         # covers Sep 2023 → Mar 2026  (prev: 8)
INITIAL_BAL  = 200.0

# ── Scalp model parameters ───────────────────────────────────────────────────
SL_ATR_MULT  = 0.8
TP_RR        = 1.5
MAX_HORIZON  = 8          # M1 bars = 8 minutes
THR_BUY      = 0.58       # BUY signal threshold
THR_SELL     = 0.55       # SELL signal threshold (slightly lower — more signals)

# ── Risk levels to test ───────────────────────────────────────────────────────
RISK_LEVELS  = [0.01, 0.005, 0.003]   # realistic for high-freq M1 scalp
RISK_LABELS  = ["1%", "0.5%", "0.3%"]


# ─── Model training ───────────────────────────────────────────────────────────

def _make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=5,
        min_samples_leaf=50, l2_regularization=1.0, max_bins=63,
        early_stopping=True, validation_fraction=0.10,
        n_iter_no_change=20, random_state=42,
    )


def _train_dir_model(train_df: pd.DataFrame, direction: int) -> tuple:
    """Train a calibrated HistGBDT for a single direction (BUY or SELL)."""
    label = "BUY" if direction == 1 else "SELL"
    sub = train_df[train_df["expected_direction"] == direction].copy()
    n   = len(sub)
    if n < 2000:
        print(f"    [{label}] skip: only {n} rows", flush=True)
        return None, None
    X = sub[SCALP_FEATURE_COLUMNS].fillna(0).values
    y = sub["target"].values
    pos = int(y.sum()); neg = n - pos
    if pos < 200 or neg < 200:
        return None, None

    print(f"    [{label}] n={n:,}  tp_rate={pos/n*100:.1f}%", flush=True)
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    # Class weight balancing
    cw = np.where(y == 1, float(neg) / float(pos), 1.0).astype(float)
    # Time decay — recent bars more important for scalping (half-life = 30% of training)
    decay_half = n * 0.30
    tw = np.exp(np.log(2) * np.arange(n) / decay_half)
    tw /= tw.mean()
    sw = cw * tw; sw /= sw.mean()

    model = _make_model()
    model.fit(X_s, y, sample_weight=sw)

    # Isotonic calibration on last 15% of training data
    val_size = max(int(n * 0.15), 1000)
    try:
        X_val = X_s[-val_size:]; y_val = y[-val_size:]
        raw_p = model.predict_proba(X_val)[:, 1]
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw_p, y_val)
        print(f"    [{label}] ✓ calibrated", flush=True)
        return CalibratedDirModel(model, iso, scaler, direction), scaler
    except Exception as e:
        print(f"    [{label}] calibration failed ({e}), using raw", flush=True)
        return model, scaler


def _predict(model, scaler, test_df: pd.DataFrame) -> np.ndarray:
    """Get calibrated probabilities for test rows."""
    if model is None:
        return np.zeros(len(test_df))
    X = test_df[SCALP_FEATURE_COLUMNS].fillna(0).values
    # CalibratedDirModel handles scaling internally (scaler bundled inside)
    if isinstance(model, CalibratedDirModel) or hasattr(model, "_sc"):
        return model.predict_proba(X)[:, 1]
    X_s = scaler.transform(X)
    return model.predict_proba(X_s)[:, 1]


def _build_signals(fold_df: pd.DataFrame,
                   buy_probs: np.ndarray,
                   sell_probs: np.ndarray) -> pd.DataFrame:
    """Apply thresholds and direction filter to produce signal column."""
    n      = len(fold_df)
    dirs   = fold_df["expected_direction"].values
    bp     = buy_probs[:n];  sp = sell_probs[:n]

    buy_sig  = (dirs ==  1) & (bp >= THR_BUY)
    sell_sig = (dirs == -1) & (sp >= THR_SELL)

    p = fold_df.copy()
    p["split"]       = "test"
    p["prediction"]  = 0
    p.loc[buy_sig,  "prediction"] = 1
    p.loc[sell_sig, "prediction"] = 1
    # probability = model output for that direction
    p["probability"] = np.where(dirs == -1, sp, bp)
    # Required by engine / HybridStrategy
    p["trade_side"]       = np.where(dirs == -1, "sell", "buy")
    p["strategy_score"]   = 1.0   # M1 scalp: use own probability gate only
    p["volatility_regime"] = 1
    p["trend_alignment"]  = 1
    p["adx"]              = 25.0  # pass ADX gate (our M1 momentum features replace ADX)
    return p


# ─── Simulation ───────────────────────────────────────────────────────────────

def _run_sim(preds: pd.DataFrame, settings, risk: float) -> dict:
    """Run backtest engine with scalp-tuned risk settings."""
    s = settings.model_copy(deep=True)
    s.training.backtest_initial_balance    = INITIAL_BAL
    # Disable M5-era strategy filters — M1 scalp uses its own confidence gate
    s.strategy.adx_gate_enabled            = False
    s.strategy.min_strategy_score          = 0.0
    s.strategy.sideway_min_strategy_score  = 0.0
    s.strategy.strong_volatility_min_strategy_score = 0.0
    s.strategy.require_trend_alignment     = False
    s.strategy.silver_bullet_enabled       = False  # skip silver_bullet for M1 speed
    # CRITICAL: match engine hold time to M1 label horizon (8 bars = 8 min)
    # Default label_horizon=24 from M5 config would hold each position for 24 M1 bars,
    # blocking 3 slots × 24 bars = 72-bar dead zone — would suppress 70%+ of valid signals.
    s.training.label_horizon               = MAX_HORIZON   # 8 M1 bars = 8 minutes
    s.risk.risk_per_trade                  = risk
    s.risk.max_risk_fraction               = risk + 0.02
    s.risk.take_profit_rr                  = TP_RR
    s.risk.sideway_take_profit_rr          = TP_RR * 0.8
    s.risk.volatile_take_profit_rr         = TP_RR * 1.3
    s.risk.stop_loss_atr_multiple          = SL_ATR_MULT
    s.risk.min_confidence                  = THR_SELL      # lower bound
    s.risk.compound_cap                    = 25.0          # 25× cap per fold
    s.risk.consecutive_loss_pause_count    = 2
    s.risk.consecutive_loss_cooldown_bars  = 4             # 4 M1 bars = 4 min
    s.risk.daily_loss_limit_pct            = 0.0
    s.risk.anti_martingale_factor          = 0.3
    s.risk.max_open_positions              = 3
    s.risk.partial_tp_enabled             = True
    s.risk.partial_tp_rr                  = TP_RR * 0.5   # partial at 0.75R
    s.risk.partial_tp_pct                 = 0.60
    s.execution.close_opposite_on_signal  = True
    try:
        sim = simulate_dynamic_concurrent_backtest(
            preds, s, RiskManager(s), label="test", compound=True)
        rep   = sim.report
        n_tr  = int(rep.get("trades", 0) or 0)
        n_win = int(rep.get("wins",   0) or 0)
        n_los = int(rep.get("losses", 0) or 0)
        pf    = float(rep.get("profit_factor", 0) or 0)
        if pf == 0 and n_win > 0 and n_los == 0:
            pf = float("inf")
        return {
            "trades": n_tr,
            "wins":   n_win,
            "losses": n_los,
            "wr":     float((rep.get("win_rate", 0) or 0) * 100),
            "pf":     pf,
            "ret":    (float(rep.get("ending_balance", INITIAL_BAL)) - INITIAL_BAL) / INITIAL_BAL * 100,
            "dd":     abs(float(rep.get("max_drawdown_pct", 0) or 0)),
            "end_bal": float(rep.get("ending_balance", INITIAL_BAL)),
        }
    except ZeroDivisionError:
        return {"trades":0,"wins":0,"losses":0,"wr":0.0,"pf":0.0,"ret":0.0,"dd":0.0,"end_bal":INITIAL_BAL}


def _fold_years(fold_df: pd.DataFrame) -> float:
    t0 = pd.to_datetime(fold_df["time"].iloc[0])
    t1 = pd.to_datetime(fold_df["time"].iloc[-1])
    return max((t1 - t0).days / 365.25, 1 / 52)


def _ann(ret_pct: float, years: float) -> float:
    f = 1 + ret_pct / 100
    return (f ** (1 / years) - 1) * 100 if f > 0 else ret_pct / years


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    settings = load_settings(CONFIG)

    print("=" * 110)
    print("  ACC2 — SCALPING Walk-Forward (M1 architecture) — EXTENDED: covers gold ATH 2023-2024")
    print("  ▶ Execution: M1 bars  |  Context: M5 bias")
    print("  ▶ Features: 54 scalp-native (order flow, microstructure, structure, session)")
    print(f"  ▶ Label: TP={TP_RR}R within {MAX_HORIZON} M1 bars  |  SL={SL_ATR_MULT}×ATR5")
    print(f"  ▶ WF: {N_FOLDS} folds  train={TRAIN_SIZE//1000}K  test={TEST_SIZE//1000}K M1 bars")
    print(f"  ▶ Signal: BUY≥{THR_BUY}  SELL≥{THR_SELL}  |  m/position=3  partial_tp={TP_RR*0.5:.1f}R")
    print(f"  ▶ Risk grid: {RISK_LABELS}  |  KEY TEST: does scalp edge survive ATH gold (Aug2023-Jul2024)?")
    print("=" * 110)

    # ── Build dataset (extended back to 2019 to cover ATH folds) ──────────
    DS_START = "2019-01-01"    # 7+ years; folds reach back to Sep 2023
    print(f"\n[1] Building extended scalp dataset {DS_START} → latest…", flush=True)
    ds = build_scalp_dataset(
        start_date=DS_START,
        sl_atr_mult=SL_ATR_MULT,
        tp_rr=TP_RR,
        max_horizon=MAX_HORIZON,
    )
    print(f"    Total: {len(ds):,} M1 rows  "
          f"[{str(ds['time'].iloc[0])[:10]} → {str(ds['time'].iloc[-1])[:10]}]", flush=True)

    # ── Fold boundaries ───────────────────────────────────────────────
    total = len(ds)
    needed = TRAIN_SIZE + TEST_SIZE * N_FOLDS
    if total < needed:
        print(f"WARNING: dataset has {total:,} rows, need {needed:,} for {N_FOLDS} folds. "
              f"Reducing to {(total - TRAIN_SIZE) // TEST_SIZE} folds.", flush=True)

    fold_starts = list(range(
        max(0, total - TRAIN_SIZE - TEST_SIZE * N_FOLDS),
        total - TRAIN_SIZE - TEST_SIZE + 1,
        STEP_SIZE,
    ))[-N_FOLDS:]

    folds = [(s, s + TRAIN_SIZE, s + TRAIN_SIZE + TEST_SIZE) for s in fold_starts]
    fold_periods = []
    for ts, te, tnd in folds:
        fold_te = ds.iloc[te:tnd]
        try:
            p = f"{str(fold_te['time'].iloc[0])[:10]} → {str(fold_te['time'].iloc[-1])[:10]}"
        except IndexError:
            p = "?"
        fold_periods.append(p)

    print(f"\n  Fold test periods:")
    for i, (period, (ts, te, tnd)) in enumerate(zip(fold_periods, folds), 1):
        print(f"    Fold {i}: {period}  train={te-ts:,}  test={tnd-te:,}")

    # ── Walk-forward evaluation ───────────────────────────────────────
    print(f"\n[2] Walk-forward training + evaluation ({N_FOLDS} folds)…", flush=True)

    all_results = {r: [] for r in RISK_LEVELS}
    fold_meta   = []

    for fold_idx, (ts, te, tnd) in enumerate(folds):
        fold_num  = fold_idx + 1
        train_df  = ds.iloc[ts:te].copy().reset_index(drop=True)
        test_df   = ds.iloc[te:tnd].copy().reset_index(drop=True)
        years     = _fold_years(test_df)

        print(f"\n  ── Fold {fold_num}/{N_FOLDS} ({fold_periods[fold_idx]}, {years:.2f} yr) ──", flush=True)
        print(f"     Training: {len(train_df):,} M1 bars  "
              f"BUY={( train_df['expected_direction']>0).mean()*100:.0f}%  "
              f"SELL={(train_df['expected_direction']<0).mean()*100:.0f}%", flush=True)

        # Train
        buy_model,  buy_sc  = _train_dir_model(train_df,  1)
        sell_model, sell_sc = _train_dir_model(train_df, -1)

        # Predict
        buy_p  = _predict(buy_model,  buy_sc,  test_df)
        sell_p = _predict(sell_model, sell_sc, test_df)
        preds  = _build_signals(test_df, buy_p, sell_p)

        n_buy_sig  = (preds["prediction"] == 1) & (test_df["expected_direction"] == 1)
        n_sell_sig = (preds["prediction"] == 1) & (test_df["expected_direction"] == -1)
        print(f"     Signals: {n_buy_sig.sum()} BUY + {n_sell_sig.sum()} SELL "
              f"= {preds['prediction'].sum()} total", flush=True)

        # Run simulation at each risk level
        base_res = None
        for risk in RISK_LEVELS:
            r = _run_sim(preds, settings, risk)
            all_results[risk].append(r)
            if base_res is None:
                base_res = r
            ann = _ann(r["ret"], years)
            pf_str = "inf" if r["pf"] == float("inf") else f"{r['pf']:.2f}"
            print(f"     risk={risk*100:.0f}%:  "
                  f"trades={r['trades']:>4}  WR={r['wr']:>5.1f}%  PF={pf_str:>5}  "
                  f"ret={r['ret']:>+7.1f}%  ann={ann:>+7.1f}%/yr  DD={r['dd']:>5.1f}%", flush=True)

        # Signal-level diagnostics (at 1% risk fold results)
        r1 = all_results[0.01][-1]
        fold_meta.append({
            "fold": fold_num, "period": fold_periods[fold_idx], "years": years,
            "n_buy": int(n_buy_sig.sum()), "n_sell": int(n_sell_sig.sum()),
            "tp_rate_test": float(test_df["target"].mean() * 100),
        })

    # ─── Final report ─────────────────────────────────────────────────────────
    # Identify ATH period folds (Aug 2023 - Jul 2024) for comparison
    ATH_START = "2023-08"
    ATH_END   = "2024-07"
    def _is_ath(period: str) -> bool:
        try:
            return ATH_START <= period[:7] <= ATH_END
        except Exception:
            return False

    print(f"\n\n{'='*110}")
    print(f"  WALK-FORWARD RESULTS — M1 Scalping (per-fold fresh model, {N_FOLDS} folds)")
    print(f"  M5 comparison: Aug2023-Jul2024 ATH folds = WR=0%, PF=0, -13% loss — can M1 scalp survive?")
    print(f"{'='*110}")

    for risk, rlabel in zip(RISK_LEVELS, RISK_LABELS):
        rows = all_results[risk]
        print(f"\n  ── Risk={rlabel} ────────────────────────────────────────────────────────")
        print(f"  {'Fold':>4}  {'Period':>24}  {'Yrs':>4}  {'Trades':>6}  "
              f"{'WR%':>5}  {'PF':>5}  {'Ret%':>7}  {'Ann%':>7}  {'DD%':>5}  ✓")
        print("  " + "─" * 85)
        ann_rets, dds = [], []
        for i, (r, meta) in enumerate(zip(rows, fold_meta)):
            years  = meta["years"]
            ann    = _ann(r["ret"], years)
            pf_str = "inf" if r["pf"] == float("inf") else f"{r['pf']:.2f}"
            flag   = ("✓✓✓" if (r["pf"] >= 2.0 and ann >= 50) else
                      ("✓✓ " if  r["pf"] >= 2.0 else
                      ("✓  " if  r["ret"] > 0 else  "✗  ")))
            ath_mark = " ◀ATH" if _is_ath(meta["period"]) else "     "
            ann_rets.append(ann); dds.append(r["dd"])
            print(f"  {i+1:>4}  {meta['period']:>24}  {years:>4.2f}  "
                  f"{r['trades']:>6}  {r['wr']:>5.1f}  {pf_str:>5}  "
                  f"{r['ret']:>+6.1f}%  {ann:>+6.1f}%  {r['dd']:>5.1f}%  {flag}{ath_mark}")
        print("  " + "─" * 85)
        pos   = sum(1 for r in rows if r["ret"] > 0)
        ann50 = sum(1 for a in ann_rets if a >= 50)
        med   = float(np.median(ann_rets))
        print(f"  SUMMARY  Pos.folds={pos}/{N_FOLDS}  Ann≥50%={ann50}/{N_FOLDS}  "
              f"Med.ann={med:>+6.1f}%/yr  MaxDD={max(dds):>5.1f}%")

    # Cross-risk summary
    print(f"\n\n{'='*110}")
    print(f"  CROSS-RISK SUMMARY")
    print(f"{'='*110}")
    print(f"  {'Risk':>6}  {'Med.Ann%':>9}  {'Avg.Ann%':>9}  {'T.Trades':>9}  "
          f"{'Avg/fold':>8}  {'PF≥2':>5}  {'Pos':>4}  {'MaxDD':>6}")
    print(f"  {'─'*75}")
    for risk, rlabel in zip(RISK_LEVELS, RISK_LABELS):
        rows = all_results[risk]
        ann_rets = [_ann(r["ret"], m["years"]) for r, m in zip(rows, fold_meta)]
        dds = [r["dd"] for r in rows]
        med  = float(np.median(ann_rets))
        avg  = float(np.mean(ann_rets))
        tot  = sum(r["trades"] for r in rows)
        pf2  = sum(1 for r in rows if r["pf"] >= 2.0 or r["pf"] == float("inf"))
        pos  = sum(1 for r in rows if r["ret"] > 0)
        print(f"  {rlabel:>6}  {med:>+8.1f}%  {avg:>+8.1f}%  "
              f"{tot:>9}  {tot//N_FOLDS:>8}  {pf2:>5}/{N_FOLDS}  {pos:>3}/{N_FOLDS}  {max(dds):>5.1f}%")

    # Non-compound edge
    print(f"\n\n{'='*110}")
    print(f"  MODEL EDGE — Non-compound (1% fixed risk, {TP_RR}R TP, {SL_ATR_MULT}×ATR SL)")
    print(f"{'='*110}")
    print(f"  {'Fold':>4}  {'Period':>24}  {'Trades':>7}  {'WR%':>5}  {'PF':>5}  "
          f"{'P&L(1%fix)':>11}  {'Ann%':>6}  Assessment")
    print("  " + "─" * 85)
    actual_tp = TP_RR   # realized at TP (no sl_mult correction needed for edge calc)
    total_pnl = 0.0
    total_yrs = 0.0
    for i, (r, meta) in enumerate(zip(all_results[0.01], fold_meta)):
        pnl  = (r["wins"] * actual_tp - r["losses"] * 1.0) * 0.01 * 100
        ann  = pnl / meta["years"]
        total_pnl += pnl; total_yrs += meta["years"]
        pf_str = "inf" if r["pf"] == float("inf") else f"{r['pf']:.2f}"
        ath_tag = " ◀ATH" if _is_ath(meta["period"]) else ""
        rating = ("\u2713 Profitable" if ann >= 20 else
                  ("~ Breakeven" if ann >= -5  else "\u2717 Loss"))
        print(f"  {i+1:>4}  {meta['period']:>24}  {r['trades']:>7}  "
              f"{r['wr']:>5.1f}  {pf_str:>5}  {pnl:>+10.1f}%  {ann:>+5.1f}%  {rating}{ath_tag}")
    print("  " + "─" * 85)
    print(f"  TOTAL  {total_pnl:>+10.1f}%  {total_pnl/total_yrs:>+5.1f}%/yr  "
          f"(non-compound, reflects pure model edge at 1% sizing)")

    # ATH-specific summary
    ath_rows_1pct = [(r, m) for r, m in zip(all_results[0.01], fold_meta) if _is_ath(m["period"])]
    non_ath_rows  = [(r, m) for r, m in zip(all_results[0.01], fold_meta) if not _is_ath(m["period"])]
    print(f"\n\n  ATH COMPARISON (M5 model: WR=0%, PF=0 for all Aug2023-Jul2024 folds)")
    print(f"  {'Period':>24}  {'WR%':>5}  {'PF':>5}  {'Ann%':>7}  M1 vs M5")
    print("  " + "─" * 60)
    for r, m in ath_rows_1pct:
        pf_s = "inf" if r["pf"] == float("inf") else f"{r['pf']:.2f}"
        ann  = _ann(r["ret"], m["years"])
        verdict = "M1 WINS \u2713" if r["ret"] > 0 else "M1 FAIL \u2717"
        print(f"  {m['period']:>24}  {r['wr']:>5.1f}  {pf_s:>5}  {ann:>+7.1f}%  {verdict}")
    if not ath_rows_1pct:
        print("  (no folds in Aug2023-Jul2024 range)")

    # Signal diagnostics
    print(f"\n\n  SIGNAL DIAGNOSTICS")
    print(f"  {'Fold':>4}  {'BUY sigs':>8}  {'SELL sigs':>9}  {'TP rate%':>8}  {'Period':>24}")
    print("  " + "─" * 60)
    for meta in fold_meta:
        print(f"  {meta['fold']:>4}  {meta['n_buy']:>8}  {meta['n_sell']:>9}  "
              f"{meta['tp_rate_test']:>7.1f}%  {meta['period']:>24}")
    tot_buy  = sum(m["n_buy"]  for m in fold_meta)
    tot_sell = sum(m["n_sell"] for m in fold_meta)
    print(f"  {'TOT':>4}  {tot_buy:>8}  {tot_sell:>9}")

    print(f"\n{'='*110}")
    print(f"  ▶ Kiến trúc: M1 native  |  54 scalp features  |  per-fold fresh dual model")
    print(f"  ▶ Horizon: {MAX_HORIZON} M1 bars = {MAX_HORIZON} phút  |  TP={TP_RR}R  SL={SL_ATR_MULT}×ATR")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()
