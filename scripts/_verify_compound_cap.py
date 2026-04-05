"""
Verify the recommended ACC1 config with compound_cap=0 vs compound_cap=50
to see if results are comparable to the current benchmark.
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from xauusd_ai.backtesting.engine import simulate_dynamic_concurrent_backtest
from xauusd_ai.config import load_settings
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.dataset import build_merged_context, prepare_training_dataset
from xauusd_ai.model.trainer import ModelTrainer
from xauusd_ai.strategies.hybrid import HybridStrategy
from xauusd_ai.data.market_data import MarketDataService
import pandas as pd

CONFIGS = [
    ("ACC1 old benchmark (compound_cap=50 default)", "configs/benchmarks/acc1_expand_net127313_dd3215.yaml", None, None, None),
    ("ACC1 new (compound_cap=0)",  "configs/benchmarks/acc1_pf3_net297k_dd2629.yaml", None, None, None),
    ("ACC1 new (compound_cap=50)", "configs/benchmarks/acc1_pf3_net297k_dd2629.yaml", 50.0, None, None),
]

def run_wf(config_path: str, override_cap=None, override_threshold=None, override_risk=None):
    s = load_settings(Path(config_path))
    if override_cap is not None:
        s.risk.compound_cap = float(override_cap)
    if override_threshold is not None:
        s.strategy.signal_threshold = float(override_threshold)
    if override_risk is not None:
        s.risk.risk_per_trade = float(override_risk)

    service = MarketDataService(s)
    frames = service.fetch_multi_timeframe_data(source=s.market.training_data_source, all_bars=True)
    from xauusd_ai.features.dataset import build_merged_context, prepare_training_dataset
    merged = build_merged_context(s, frames)
    dataset = prepare_training_dataset(s, frames, HybridStrategy(s), cached_merged=merged)

    train_size = s.training.walkforward_train_size
    test_size  = s.training.walkforward_test_size
    step_size  = s.training.walkforward_step_size
    max_folds  = s.training.walkforward_max_folds_per_combination or 8
    fold_indices = list(range(0, max(len(dataset) - train_size - test_size + 1, 0), step_size))
    if max_folds > 0:
        fold_indices = fold_indices[-max_folds:]

    balance = float(s.training.backtest_initial_balance)
    print(f"  Starting balance: ${balance:.0f}, compound_cap={s.risk.compound_cap}, threshold={s.strategy.signal_threshold}, risk={s.risk.risk_per_trade}")
    print(f"  Folds: {len(fold_indices)}", flush=True)

    all_gross_profit = 0.0
    all_gross_loss = 0.0
    all_trades = 0
    all_wins = 0
    all_losses = 0
    equity_curve = [balance]
    global_peak = balance
    global_max_dd = 0.0

    for idx, fold_start in enumerate(fold_indices, start=1):
        train_end = fold_start + train_size
        test_end  = train_end  + test_size
        fold_train = dataset.iloc[fold_start:train_end].copy()
        fold_test  = dataset.iloc[train_end:test_end].copy()
        if len(fold_train) < 200 or len(fold_test) < 50:
            continue

        fold_dataset = pd.concat([fold_train, fold_test], ignore_index=True)
        fold_dataset["split"] = "train"
        fold_dataset.loc[len(fold_train):, "split"] = "test"

        fold_s = s.model_copy(deep=True)
        fold_s.training.backtest_initial_balance = balance

        trainer = ModelTrainer(fold_s)
        trainer.train(fold_dataset, save_artifacts=False)
        preds = trainer.predict_dataset(fold_dataset)
        preds["prediction"] = (preds["probability"] >= fold_s.strategy.signal_threshold).astype(int)

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

        all_gross_profit += float(rep["gross_profit"])
        all_gross_loss   += float(rep["gross_loss"])
        all_trades += int(rep["trades"])
        all_wins   += int(rep["wins"])
        all_losses += int(rep["losses"])
        print(f"    Fold {idx}: bal=${balance:,.0f}  trades={rep['trades']}  WR={rep['win_rate']:.1%}  PF={rep['profit_factor']:.2f}  DD={rep['max_drawdown_pct']:.1f}%")

    net = balance - float(s.training.backtest_initial_balance)
    pf = all_gross_profit / all_gross_loss if all_gross_loss > 0 else 0
    wr = all_wins / max(all_wins + all_losses, 1)
    # also compute exact dd from equity curve
    pk = equity_curve[0]
    max_dd_exact = 0.0
    for v in equity_curve:
        pk = max(pk, v)
        if pk > 0:
            max_dd_exact = max(max_dd_exact, (pk - v) / pk)

    max_dd = max(global_max_dd * 100, max_dd_exact * 100)
    print(f"  TOTAL: net=${net:,.2f}  PF={pf:.4f}  DD={max_dd:.2f}%  trades={all_trades}  WR={wr:.2%}")
    return net, pf, max_dd, all_trades, wr


for label, cfg, cap, thr, risk in CONFIGS:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)
    try:
        net, pf, dd, trades, wr = run_wf(cfg, override_cap=cap, override_threshold=thr, override_risk=risk)
    except Exception as e:
        print(f"  ERROR: {e}")
