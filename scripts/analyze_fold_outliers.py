#!/usr/bin/env python3
"""
Analyze why Fold 5 and Fold 7 had explosive returns (+54k%, +86k%)
while other folds had normal returns (hundreds of %).
"""
import pandas as pd
import json
import numpy as np

# Read simulation trades
df = pd.read_csv("outputs/walkforward_trades_acc1_v14pp_profit_sim_trades.csv")

# Fold 5 and Fold 7
fold5 = df[df['fold'] == 4].copy()  # Fold 5 is index 4 (0-based)
fold7 = df[df['fold'] == 6].copy()  # Fold 7 is index 6 (0-based)
other_folds = df[~df['fold'].isin([4, 6])].copy()

df['time'] = pd.to_datetime(df['time'])
fold5['time'] = pd.to_datetime(fold5['time'])
fold7['time'] = pd.to_datetime(fold7['time'])

print("=" * 100)
print("FOLD 5 Analysis (2024-12-12 → 2025-03-17): +54,261% return ($200 → $108,723)")
print("=" * 100)
print(f"Total trades executed: {len(fold5)}")
print(f"Winners: {(fold5['pnl'] > 0).sum()} ({(fold5['pnl'] > 0).sum()/len(fold5)*100:.1f}%)")
print(f"Losers: {(fold5['pnl'] <= 0).sum()} ({(fold5['pnl'] <= 0).sum()/len(fold5)*100:.1f}%)")

print(f"\n💰 P&L Statistics:")
print(f"  Total P&L: ${fold5['pnl'].sum():,.2f}")
print(f"  Avg win: ${fold5[fold5['pnl'] > 0]['pnl'].mean():.2f}")
print(f"  Avg loss: ${fold5[fold5['pnl'] <= 0]['pnl'].mean():.2f}")
print(f"  Max single win: ${fold5['pnl'].max():,.2f}")
print(f"  Max single loss: ${fold5['pnl'].min():,.2f}")
win_sum = fold5[fold5['pnl'] > 0]['pnl'].sum()
loss_sum = abs(fold5[fold5['pnl'] <= 0]['pnl'].sum())
print(f"  Profit Factor: {win_sum / loss_sum:.3f}")

print(f"\n📊 RR Statistics:")
print(f"  Avg RR (all trades): {fold5['realized_rr'].mean():.2f}")
print(f"  Avg RR (winners only): {fold5[fold5['pnl'] > 0]['realized_rr'].mean():.2f}")
print(f"  Max RR achieved: {fold5['realized_rr'].max():.2f}")
print(f"  Min RR: {fold5['realized_rr'].min():.2f}")

print(f"\n🏆 Top 10 Winning Trades:")
top_wins_5 = fold5.nlargest(10, 'pnl')[['time', 'pnl', 'realized_rr', 'side', 'entry_price']]
for idx, row in top_wins_5.iterrows():
    print(f"  {row['time']:%Y-%m-%d %H:%M}  ${row['pnl']:>9,.2f}  RR={row['realized_rr']:.2f}  {row['side']:<4}  Entry={row['entry_price']:.2f}")

# Check for compounding effect
fold5_sorted = fold5.sort_values('time')
fold5_sorted['cumulative_pnl'] = fold5_sorted['pnl'].cumsum()
print(f"\n📈 Compounding Analysis:")
print(f"  First 25% trades PnL: ${fold5_sorted.iloc[:len(fold5_sorted)//4]['pnl'].sum():,.2f}")
print(f"  Middle 50% trades PnL: ${fold5_sorted.iloc[len(fold5_sorted)//4:3*len(fold5_sorted)//4]['pnl'].sum():,.2f}")
print(f"  Last 25% trades PnL: ${fold5_sorted.iloc[3*len(fold5_sorted)//4:]['pnl'].sum():,.2f}")

print("\n" + "=" * 100)
print("FOLD 7 Analysis (2025-06-18 → 2025-09-17): +86,151% return ($200 → $172,503)")
print("=" * 100)
print(f"Total trades executed: {len(fold7)}")
print(f"Winners: {(fold7['pnl'] > 0).sum()} ({(fold7['pnl'] > 0).sum()/len(fold7)*100:.1f}%)")
print(f"Losers: {(fold7['pnl'] <= 0).sum()} ({(fold7['pnl'] <= 0).sum()/len(fold7)*100:.1f}%)")

print(f"\n💰 P&L Statistics:")
print(f"  Total P&L: ${fold7['pnl'].sum():,.2f}")
print(f"  Avg win: ${fold7[fold7['pnl'] > 0]['pnl'].mean():.2f}")
print(f"  Avg loss: ${fold7[fold7['pnl'] <= 0]['pnl'].mean():.2f}")
print(f"  Max single win: ${fold7['pnl'].max():,.2f}")
print(f"  Max single loss: ${fold7['pnl'].min():,.2f}")
win_sum_7 = fold7[fold7['pnl'] > 0]['pnl'].sum()
loss_sum_7 = abs(fold7[fold7['pnl'] <= 0]['pnl'].sum())
print(f"  Profit Factor: {win_sum_7 / loss_sum_7:.3f}")

print(f"\n📊 RR Statistics:")
print(f"  Avg RR (all trades): {fold7['realized_rr'].mean():.2f}")
print(f"  Avg RR (winners only): {fold7[fold7['pnl'] > 0]['realized_rr'].mean():.2f}")
print(f"  Max RR achieved: {fold7['realized_rr'].max():.2f}")
print(f"  Min RR: {fold7['realized_rr'].min():.2f}")

print(f"\n🏆 Top 10 Winning Trades:")
top_wins_7 = fold7.nlargest(10, 'pnl')[['time', 'pnl', 'realized_rr', 'side', 'entry_price']]
for idx, row in top_wins_7.iterrows():
    print(f"  {row['time']:%Y-%m-%d %H:%M}  ${row['pnl']:>9,.2f}  RR={row['realized_rr']:.2f}  {row['side']:<4}  Entry={row['entry_price']:.2f}")

# Check for compounding effect
fold7_sorted = fold7.sort_values('time')
fold7_sorted['cumulative_pnl'] = fold7_sorted['pnl'].cumsum()
print(f"\n📈 Compounding Analysis:")
print(f"  First 25% trades PnL: ${fold7_sorted.iloc[:len(fold7_sorted)//4]['pnl'].sum():,.2f}")
print(f"  Middle 50% trades PnL: ${fold7_sorted.iloc[len(fold7_sorted)//4:3*len(fold7_sorted)//4]['pnl'].sum():,.2f}")
print(f"  Last 25% trades PnL: ${fold7_sorted.iloc[3*len(fold7_sorted)//4:]['pnl'].sum():,.2f}")

print("\n" + "=" * 100)
print("COMPARISON WITH OTHER FOLDS (Fold 1-4, 6, 8-10)")
print("=" * 100)
print(f"Total trades: {len(other_folds)}")
print(f"Win rate: {(other_folds['pnl'] > 0).sum()/len(other_folds)*100:.1f}%")
print(f"Avg win: ${other_folds[other_folds['pnl'] > 0]['pnl'].mean():.2f}")
print(f"Avg loss: ${other_folds[other_folds['pnl'] <= 0]['pnl'].mean():.2f}")
print(f"Avg RR (winners): {other_folds[other_folds['pnl'] > 0]['realized_rr'].mean():.2f}")
print(f"Max single win: ${other_folds['pnl'].max():,.2f}")
print(f"Profit Factor: {other_folds[other_folds['pnl'] > 0]['pnl'].sum() / abs(other_folds[other_folds['pnl'] <= 0]['pnl'].sum()):.3f}")

print("\n" + "=" * 100)
print("ROOT CAUSE ANALYSIS")
print("=" * 100)

# Hypothesis 1: Compounding effect (NO-COMPOUND mode, so shouldn't be this)
print("\n1. Compounding Effect Check:")
print(f"   WF Mode: NO-COMPOUND (each fold resets to $200)")
print(f"   ❌ This is NOT the cause - balance resets each fold")

# Hypothesis 2: Unrealistic RR values
print("\n2. Unrealistic RR Values:")
print(f"   Fold 5 max RR: {fold5['realized_rr'].max():.2f}")
print(f"   Fold 7 max RR: {fold7['realized_rr'].max():.2f}")
print(f"   Other folds max RR: {other_folds['realized_rr'].max():.2f}")
print(f"   🚨 POTENTIAL ISSUE: Check if these RR values are realistic")

# Hypothesis 3: Large winning streaks
print("\n3. Winning Streak Analysis:")
fold5_wins = fold5[fold5['pnl'] > 0]
fold7_wins = fold7[fold7['pnl'] > 0]
other_wins = other_folds[other_folds['pnl'] > 0]
print(f"   Fold 5 - Total winning trades: {len(fold5_wins)}, Total: ${fold5_wins['pnl'].sum():,.2f}")
print(f"   Fold 7 - Total winning trades: {len(fold7_wins)}, Total: ${fold7_wins['pnl'].sum():,.2f}")
print(f"   Other folds - Total winning trades: {len(other_wins)}, Total: ${other_wins['pnl'].sum():,.2f}")
print(f"   📊 Fold 5/7 have significantly higher total P&L from wins")

# Hypothesis 4: Simulation error (M1 bar-by-bar vs real execution)
print("\n4. Simulation Accuracy:")
print(f"   M1 bar-by-bar simulation: ENABLED")
print(f"   Trailing SL: ENABLED")
print(f"   ⚠️  CONCERN: Real execution may not achieve these high RR values")

# Hypothesis 5: Market conditions
print("\n5. Market Regime Analysis:")
print(f"   Fold 5 period: Dec 2024 - Mar 2025")
print(f"   Fold 7 period: Jun 2025 - Sep 2025")
print(f"   💡 HYPOTHESIS: These periods may have strong trending conditions")
print(f"   💡 ICT/Wyckoff patterns excel in trends, may have caught large moves")

print("\n" + "=" * 100)
print("RECOMMENDATIONS")
print("=" * 100)
print("1. ⚠️  BACKTEST-LIVE GAP: These explosive returns (50k-80k%) are suspicious")
print("2. 🔍 Validate with REAL demo account data from these periods")
print("3. 📉 Check if max RR > 20 is realistic with real slippage/spread")
print("4. 🎯 Paper trading will show if orchestrator can achieve similar RR")
print("5. 📊 If live results show 5-10× lower returns, WF is overfitting on simulation")
print("6. 🚨 DO NOT deploy to LIVE account based on these WF results alone")
print("7. ✅ Week 1 paper validation (May 6-13) will be critical to trust WF")
print("\n")
