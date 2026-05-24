#!/usr/bin/env python3
"""
Walk-Forward Monthly Breakdown — grid_0014 config
Starting balance: $500/fold (reset each month)
Period: Dec 2024 → May 2026
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
report_path = ROOT / "outputs" / "walkforward_report_acc1_v14pp_profit.json"

with open(report_path) as f:
    d = json.load(f)

folds = d["folds"]

START_BAL   = 500.0
TARGET_FROM = "2024-12-01"

print("=" * 80)
print("  WALK-FORWARD MONTHLY RESULTS")
print("  Config: grid_0014  (threshold=0.70 | risk=5% | max_pos=4 | no trailing)")
print(f"  Starting balance per fold: ${START_BAL:.0f}  (reset each month)")
print(f"  Period: Dec 2024 → May 2026")
print("=" * 80)
print(f"  {'Month':<8} {'Start':>7}  {'End Bal':>9}  {'Net PnL':>10}  {'Ret%':>8}  {'Trades':>7}  {'WR%':>6}  {'Status'}")
print("  " + "-" * 76)

total_pnl    = 0.0
months_shown = 0
wins_months  = 0
rows = []

for fold in folds:
    ts = fold["test_start"]
    te = fold["test_end"]
    if ts < TARGET_FROM:
        continue
    cs     = fold["concurrent_sim"]
    ret    = cs["return_pct"]
    trades = cs["trades"]
    wr     = cs["win_rate"] * 100
    dd     = cs.get("max_drawdown_pct", 0)

    end_bal = START_BAL * (1 + ret / 100.0)
    net_pnl = end_bal - START_BAL
    total_pnl   += net_pnl
    months_shown += 1
    if net_pnl > 0:
        wins_months += 1

    status = "✅ PROFIT" if net_pnl > 0 else "❌ LOSS"
    sign   = "+" if net_pnl >= 0 else ""
    print(
        f"  {ts[:7]:<8} ${START_BAL:>6.0f}  ${end_bal:>9.2f}  "
        f"{sign}${abs(net_pnl):>8.2f}  {ret:>7.1f}%  {trades:>7}  {wr:>5.1f}%  {status}"
    )
    rows.append(dict(month=ts[:7], ret=ret, net_pnl=net_pnl, trades=trades, wr=wr, dd=dd))

print("  " + "-" * 76)
avg_pnl = total_pnl / months_shown if months_shown > 0 else 0
sign_t  = "+" if total_pnl >= 0 else ""
print(
    f"  {'TOTAL':<8} {'':>7}  {'':>9}  {sign_t}${abs(total_pnl):>8.2f}  "
    f"  (avg {avg_pnl:+.2f}/month)"
)
print()
print(f"  📊 Months profitable  : {wins_months}/{months_shown}  ({100*wins_months/months_shown:.0f}%)")
print(f"  💰 Total net PnL      : {sign_t}${abs(total_pnl):.2f}")
print(f"  📅 Avg PnL/month      : {avg_pnl:+.2f}")
print(f"  📅 Avg PnL/day        : {avg_pnl/30:+.2f}")
print(f"  📈 Best month         : +${max(r['net_pnl'] for r in rows):.2f}")
print(f"  📉 Worst month        : ${min(r['net_pnl'] for r in rows):.2f}")
print(f"  📊 Avg win rate       : {sum(r['wr'] for r in rows)/len(rows):.1f}%")
print(f"  📊 Avg trades/month   : {sum(r['trades'] for r in rows)/len(rows):.0f}")
print()
print("  ⚠️  NOTE: Return% uses concurrent position simulation from model predictions.")
print("     These are the SAME signals as live, just scaled to $500 base/fold.")
print("     Not MT5 Strategy Tester — uses Python WF engine with real OHLCV data.")
print("=" * 80)
