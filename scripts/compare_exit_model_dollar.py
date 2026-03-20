"""
So sánh dollar impact của exit model trên backtest trades thực tế của ACC2.
"""
import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from pathlib import Path
from xauusd_ai.config import load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.features.dataset import prepare_training_dataset
from xauusd_ai.strategies.hybrid import HybridStrategy

START_BAL = 200.0
RISK      = 0.05
THR       = 0.83

bt  = pd.read_csv("outputs/backtest_trades_acc2.csv")
em  = pd.read_csv("outputs/exit_model_impact_acc2.csv")

print(f"Backtest trades  : {len(bt):,}  |  win rate: {bt['is_win'].mean():.1%}")
print(f"Exit model impact: {len(em):,} simulated TP-entry trades")

# Build dataset để lấy time mapping cho test set
print("Loading market data + building features (de map entry_bar -> time)...")
settings = load_settings(Path("configs/live_acc2.yaml"))
svc      = MarketDataService(settings)
frames   = svc.fetch_multi_timeframe_data(source="csv_folder", all_bars=True)
strategy = HybridStrategy(settings)
ds       = prepare_training_dataset(settings, frames, strategy)

print(f"Dataset rows     : {len(ds):,}")
print()

# Map entry_bar index -> time trong test set
test_ds = ds[ds["split"] == "test"].reset_index(drop=True)
test_ds["entry_bar"] = test_ds.index
em2 = em.merge(test_ds[["entry_bar", "time"]], on="entry_bar", how="left")

# Parse time thanh format chung yyyy-mm-dd HH:MM
bt["tc"]  = pd.to_datetime(bt["time"],  utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
em2["tc"] = pd.to_datetime(em2["time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

merged = bt.merge(
    em2[["tc", "final_rr_base", "final_rr_exit", "rr_delta", "outcome", "exit_prob"]],
    on="tc", how="left"
)

matched = merged["outcome"].notna().sum()
print(f"Matched trades   : {matched:,} / {len(merged):,}")
print()

has_fire = merged["exit_prob"].notna() & (merged["exit_prob"] >= THR)

rr_base  = float(merged["realized_rr"].sum())
rr_adj   = float((merged.loc[has_fire, "final_rr_exit"] - merged.loc[has_fire, "realized_rr"]).sum())
rr_exit  = rr_base + rr_adj

pnl_base = START_BAL * RISK * rr_base
pnl_exit = START_BAL * RISK * rr_exit

print("=" * 60)
print("  EXIT MODEL IMPACT TREN BACKTEST THUC TE  (non-compound)")
print("=" * 60)
print()
print(f"  Threshold                : {THR}")
print(f"  Trades bi anh huong      : {has_fire.sum():,} / {len(merged):,} ({has_fire.mean():.1%})")
print()
print(f"  Tong R:R baseline        : {rr_base:+.1f}R")
print(f"  Tong R:R voi exit model  : {rr_exit:+.1f}R")
print(f"  Delta R:R                : {rr_adj:+.1f}R")
print()
print(f"  P&L baseline             : ${pnl_base:+,.2f}")
print(f"  P&L voi exit model       : ${pnl_exit:+,.2f}")
print(f"  Delta P&L                : ${pnl_exit - pnl_base:+,.2f}")
print()

affected = merged.loc[has_fire, ["realized_rr", "final_rr_exit", "rr_delta", "outcome", "exit_prob"]].copy()
if len(affected) > 0:
    print(f"  Breakdown {len(affected)} trades bi exit model fire (thr={THR}):")
    for outcome, grp in affected.groupby("outcome"):
        avg_d = grp["rr_delta"].mean()
        tot_d = grp["rr_delta"].sum()
        usd   = tot_d * START_BAL * RISK
        print(f"    {outcome:<22}: {len(grp):>4} trades  avg {avg_d:+.3f}R  total {tot_d:+.2f}R  (${usd:+,.2f})")
    print()

# Threshold sweep tren backtest thuc te
print("  Threshold sweep tren backtest thuc te:")
print("  thr  | fires | delta_R  | delta_$")
print("  " + "-" * 40)
best_usd = 0.0
best_thr_real = THR
has_any_match = merged["exit_prob"].notna()
for thr_val in np.arange(0.50, 0.92, 0.03):
    fire = has_any_match & (merged["exit_prob"] >= thr_val)
    adj = float((merged.loc[fire, "final_rr_exit"] - merged.loc[fire, "realized_rr"]).sum())
    usd = adj * START_BAL * RISK
    marker = " <--" if usd > best_usd else ""
    if usd > best_usd:
        best_usd = usd
        best_thr_real = thr_val
    print(f"  {thr_val:.2f} | {fire.sum():>5} | {adj:+7.1f}R | ${usd:+,.2f}{marker}")

print()
print(f"  >> Best threshold tren backtest thuc te: {best_thr_real:.2f}  (${best_usd:+,.2f})")
print()
print("=" * 60)
