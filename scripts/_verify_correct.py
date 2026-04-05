"""
CORRECT cross-verification: build fold cache using OLD BENCHMARK config,
then evaluate BOTH old config AND new recommended config against same fold models.
This matches the search's methodology (search also builds folds WITH old config).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np, random, pandas as pd

rng_seed = 20260405
random.seed(rng_seed)
np.random.seed(rng_seed)

from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import load_settings
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.dataset import build_merged_context, prepare_training_dataset
from xauusd_ai.model.trainer import ModelTrainer
from xauusd_ai.strategies.hybrid import HybridStrategy
from xauusd_ai.data.market_data import MarketDataService

BASE_CONFIG = "configs/benchmarks/acc1_expand_net127313_dd3215.yaml"


@dataclass
class FoldCache:
    fold: int
    predictions: pd.DataFrame


def build_fold_cache(config_path: str):
    """Train fold models with base config and cache predictions."""
    s = load_settings(Path(config_path))
    service = MarketDataService(s)
    frames = service.fetch_multi_timeframe_data(source=s.market.training_data_source, all_bars=True)
    merged = build_merged_context(s, frames)
    dataset = prepare_training_dataset(s, frames, HybridStrategy(s), cached_merged=merged)
    print(f"  Dataset rows: {len(dataset):,}")

    train_size = s.training.walkforward_train_size
    test_size  = s.training.walkforward_test_size
    step_size  = s.training.walkforward_step_size
    max_folds  = s.training.walkforward_max_folds_per_combination or 8
    fold_indices = list(range(0, max(len(dataset) - train_size - test_size + 1, 0), step_size))
    if max_folds > 0:
        fold_indices = fold_indices[-max_folds:]
    print(f"  Using last {len(fold_indices)} folds")

    cached = []
    for idx, fstart in enumerate(fold_indices, start=1):
        tend = fstart + train_size
        tnd  = tend + test_size
        fold_tr = dataset.iloc[fstart:tend].copy()
        fold_te = dataset.iloc[tend:tnd].copy()
        if len(fold_tr) < 200 or len(fold_te) < 50:
            continue
        fold_ds = pd.concat([fold_tr, fold_te], ignore_index=True)
        fold_ds["split"] = "train"
        fold_ds.loc[len(fold_tr):, "split"] = "test"
        trainer = ModelTrainer(s)
        trainer.train(fold_ds, save_artifacts=False)
        preds = trainer.predict_dataset(fold_ds)
        cached.append(FoldCache(fold=idx, predictions=preds.copy()))
        te_start = str(pd.to_datetime(fold_te["time"].min()))[:16]
        te_end   = str(pd.to_datetime(fold_te["time"].max()))[:16]
        print(f"    Fold {idx}: test=[{te_start} → {te_end}]  rows={len(preds)}")
    print(f"  Fold cache built: {len(cached)} folds\n")
    return s, cached


def _max_dd_from_curve(curve):
    peak = curve[0]; mx = 0.0
    for v in curve[1:]:
        peak = max(peak, v)
        if peak > 0: mx = max(mx, (peak - v) / peak)
    return mx * 100.0


def evaluate_config(base_s, fold_cache, override: dict[str, Any], label: str, balance0: float = 200.0):
    """Evaluate a set of parameter overrides against the cached fold predictions."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)

    # Apply overrides to base settings
    s = base_s.model_copy(deep=True)
    threshold = override.get("threshold", s.strategy.signal_threshold)
    if "threshold" in override:
        s.strategy.signal_threshold = float(override["threshold"])
    if "risk_per_trade" in override:
        s.risk.risk_per_trade = float(override["risk_per_trade"])
    if "take_profit_rr" in override:
        s.risk.take_profit_rr = float(override["take_profit_rr"])
        s.risk.sideway_take_profit_rr  = float(override.get("sideway_take_profit_rr",  max(override["take_profit_rr"] - 1, 2.0)))
        s.risk.volatile_take_profit_rr = float(override.get("volatile_take_profit_rr", override["take_profit_rr"] + 1))
    if "compound_cap" in override:
        s.risk.compound_cap = float(override["compound_cap"])
    if "consecutive_loss_pause_count" in override:
        s.risk.consecutive_loss_pause_count = int(override["consecutive_loss_pause_count"])
    if "consecutive_loss_cooldown_bars" in override:
        s.risk.consecutive_loss_cooldown_bars = int(override["consecutive_loss_cooldown_bars"])
    if "risk_tier_floor" in override:
        s.risk.risk_tier_floor = float(override["risk_tier_floor"])
    if "min_strategy_score" in override:
        s.strategy.min_strategy_score = float(override["min_strategy_score"])
    if "stop_loss_atr_multiple" in override:
        s.risk.stop_loss_atr_multiple = float(override["stop_loss_atr_multiple"])
    if "partial_tp_enabled" in override:
        s.risk.partial_tp_enabled = bool(override["partial_tp_enabled"])

    print(f"  threshold={s.strategy.signal_threshold}  risk={s.risk.risk_per_trade}  TP={s.risk.take_profit_rr}  cap={s.risk.compound_cap}  pause={s.risk.consecutive_loss_pause_count}")

    balance = float(balance0)
    equity_curve = [balance]
    global_peak = balance
    global_max_dd = 0.0
    ag_gross_profit = 0.0
    ag_gross_loss   = 0.0
    ag_trades = 0; ag_wins = 0; ag_losses = 0

    for fold in fold_cache:
        fold_s = s.model_copy(deep=True)
        fold_s.training.backtest_initial_balance = balance

        preds = fold.predictions.copy()
        preds["prediction"] = (preds["probability"] >= s.strategy.signal_threshold).astype(int)

        sim = simulate_dynamic_concurrent_backtest(preds, fold_s, RiskManager(fold_s), label="test", compound=True)
        rep = sim.report
        fold_start_bal = balance
        balance = float(rep["ending_balance"])
        equity_curve.append(balance)

        global_peak = max(global_peak, fold_start_bal)
        fold_dd = abs(float(rep["max_drawdown_pct"])) / 100.0
        fold_low = max(0.0, fold_start_bal * (1.0 - fold_dd))
        if global_peak > 0:
            global_max_dd = max(global_max_dd, (global_peak - fold_low) / global_peak)
        global_peak = max(global_peak, balance)
        if global_peak > 0:
            global_max_dd = max(global_max_dd, (global_peak - balance) / global_peak)

        ag_gross_profit += float(rep["gross_profit"])
        ag_gross_loss   += float(rep["gross_loss"])
        ag_trades += int(rep["trades"])
        ag_wins   += int(rep["wins"])
        ag_losses += int(rep["losses"])
        print(f"    Fold {fold.fold}: bal=${balance:>12,.0f}  trades={rep['trades']:>4}  WR={rep['win_rate']:.1%}  PF={rep['profit_factor']:.3f}  DD={rep['max_drawdown_pct']:.1f}%")

    net = balance - balance0
    pf  = ag_gross_profit / ag_gross_loss if ag_gross_loss > 0 else 0.0
    wr  = ag_wins / max(ag_wins + ag_losses, 1)
    dd  = max(global_max_dd * 100, _max_dd_from_curve(equity_curve))

    PASS = []
    if pf  >= 3.0:     PASS.append("PF>=3 ✅")
    else:             PASS.append(f"PF={pf:.3f} ❌ (need>=3)")
    if net >= 127313:  PASS.append("Net>=127k ✅")
    else:             PASS.append(f"Net=${net:,.0f} ❌ (need>=127k)")
    if dd  < 32.15:   PASS.append("DD<32.15% ✅")
    else:             PASS.append(f"DD={dd:.2f}% ❌ (need<32.15%)")

    feasible = pf >= 3.0 and net >= 127313 and dd < 32.15
    print(f"\n  RESULT: net=${net:,.2f}  PF={pf:.4f}  DD={dd:.2f}%  trades={ag_trades}  WR={wr:.2%}")
    print(f"  STATUS: {' | '.join(PASS)}")
    print(f"  FEASIBLE: {'YES 🎉' if feasible else 'NO'}")
    return dict(net=net, pf=pf, dd=dd, trades=ag_trades, wr=wr, feasible=feasible)


if __name__ == "__main__":
    print("Building fold cache with OLD BENCHMARK config …")
    base_s, fold_cache = build_fold_cache(BASE_CONFIG)

    # 1. Evaluate old benchmark (should match its own published metrics roughly)
    r0 = evaluate_config(base_s, fold_cache, {}, "ACC1 OLD BENCHMARK (baseline)")

    # 2. Evaluate previous search's recommended config
    r1 = evaluate_config(base_s, fold_cache, {
        "threshold": 0.88,
        "risk_per_trade": 0.12,
        "take_profit_rr": 4.0,
        "compound_cap": 0.0,
        "consecutive_loss_pause_count": 2,
        "consecutive_loss_cooldown_bars": 16,
        "risk_tier_floor": 0.03,
        "min_strategy_score": 0.20,
        "partial_tp_enabled": False,
        "stop_loss_atr_multiple": 1.2,
    }, "ACC1 PREV RECOMMENDED (threshold=0.88, risk=0.12, TP=4, cap=0)")

    # 3. Try moderate threshold + higher TP (TP=5.0, threshold=0.72)
    r2 = evaluate_config(base_s, fold_cache, {
        "threshold": 0.72,
        "risk_per_trade": 0.04,
        "take_profit_rr": 5.0,
        "compound_cap": 50.0,
        "consecutive_loss_pause_count": 2,
        "consecutive_loss_cooldown_bars": 8,
        "partial_tp_enabled": False,
    }, "CANDIDATE: threshold=0.72, risk=0.04, TP=5.0, cap=50")

    # 4. Try threshold=0.75, TP=6.0, risk=0.05
    r3 = evaluate_config(base_s, fold_cache, {
        "threshold": 0.75,
        "risk_per_trade": 0.05,
        "take_profit_rr": 6.0,
        "compound_cap": 50.0,
        "consecutive_loss_pause_count": 2,
        "consecutive_loss_cooldown_bars": 8,
        "partial_tp_enabled": False,
    }, "CANDIDATE: threshold=0.75, risk=0.05, TP=6.0, cap=50")

    # 5. More moderate: threshold=0.70, TP=7.0, risk=0.04
    r4 = evaluate_config(base_s, fold_cache, {
        "threshold": 0.70,
        "risk_per_trade": 0.04,
        "take_profit_rr": 7.0,
        "compound_cap": 50.0,
        "consecutive_loss_pause_count": 2,
        "consecutive_loss_cooldown_bars": 8,
        "partial_tp_enabled": False,
    }, "CANDIDATE: threshold=0.70, risk=0.04, TP=7.0, cap=50")

    print("\n\n=== SUMMARY ===")
    rows = [("baseline",r0), ("prev_recommended",r1), ("thr72_tp5",r2), ("thr75_tp6",r3), ("thr70_tp7",r4)]
    for name, r in rows:
        status = "FEASIBLE 🎉" if r["feasible"] else "not feasible"
        print(f"  {name:25s}: net=${r['net']:>12,.0f}  PF={r['pf']:.3f}  DD={r['dd']:.2f}%  trades={r['trades']:>4}  {status}")
