#!/usr/bin/env python3
"""
Demo: Turnover-Adjusted Return Impact Analysis

Shows how spread costs dramatically reduce real profitability
compared to raw backtest P&L.
"""

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, 'src')
from xauusd_ai.infra.advanced_metrics import calculate_turnover_adjusted_return


def demo_scenario(name: str, trades_df: pd.DataFrame, spread_pips: float = 0.5):
    """Run turnover analysis for a trading scenario."""
    print(f"\n{'='*70}")
    print(f"📊 {name}")
    print(f"{'='*70}")
    
    result = calculate_turnover_adjusted_return(
        trades_df,
        spread_pips=spread_pips,
        swap_per_lot_per_day=0.15,
        pip_value=10.0,
    )
    
    print(f"\n📈 P&L Breakdown:")
    print(f"  Gross P&L:      ${result['gross_pnl']:>10,.2f}  (raw backtest)")
    print(f"  Spread Cost:    ${result['spread_cost']:>10,.2f}  ({len(trades_df)} trades × 2× × {spread_pips} pips × $10)")
    print(f"  Swap Cost:      ${result['swap_cost']:>10,.2f}  (overnight fees)")
    print(f"  ─────────────────────────────────────")
    print(f"  Net P&L:        ${result['net_pnl']:>10,.2f}  ✅ ACTUAL profit")
    
    print(f"\n💰 Performance Metrics:")
    print(f"  Turnover Drag:  {result['turnover_drag']:>10.1%}  (profit lost to costs)")
    print(f"  Net Return:     {result['net_return_pct']:>10.1%}  (on ${trades_df['balance'].iloc[0]:,.0f} balance)")
    
    # Calculate win rate for context
    wins = (trades_df['pnl'] > 0).sum()
    losses = (trades_df['pnl'] < 0).sum()
    win_rate = wins / len(trades_df)
    
    print(f"\n📊 Trade Statistics:")
    print(f"  Total Trades:   {len(trades_df):>10,}")
    print(f"  Wins:           {wins:>10,}  ({win_rate:.1%})")
    print(f"  Losses:         {losses:>10,}")
    print(f"  Avg Win:        ${trades_df[trades_df['pnl'] > 0]['pnl'].mean():>10.2f}" if wins > 0 else "  Avg Win:        $      0.00")
    print(f"  Avg Loss:       ${trades_df[trades_df['pnl'] < 0]['pnl'].mean():>10.2f}" if losses > 0 else "  Avg Loss:       $      0.00")
    
    # Warning if net is negative
    if result['net_pnl'] < 0:
        print(f"\n⚠️  WARNING: Strategy loses money after costs!")
        print(f"    Gross profit ${result['gross_pnl']:,.0f} → Net loss ${result['net_pnl']:,.0f}")
        print(f"    Need to reduce trade frequency or increase profit per trade.")
    
    return result


def main():
    """Run turnover impact demonstrations."""
    print("\n" + "="*70)
    print("🎯 TURNOVER-ADJUSTED RETURN IMPACT ANALYSIS")
    print("="*70)
    print("\nDemonstrates how spread costs reduce real profitability")
    print("compared to raw backtest P&L.\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 1: High-frequency scalping (100 trades, small wins)
    # ─────────────────────────────────────────────────────────────────────────
    np.random.seed(42)
    trades_1 = pd.DataFrame({
        'pnl': np.random.choice([8, -5], size=100, p=[0.60, 0.40]),  # 60% win rate, small gains
        'lot_size': [1.0] * 100,
        'holding_bars': [0] * 100,
        'balance': [200.0] * 100,
    })
    
    result_1 = demo_scenario(
        "Scenario 1: High-Frequency Scalping (100 trades, 60% win rate)",
        trades_1,
        spread_pips=0.5,
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 2: Swing trading (20 trades, larger wins)
    # ─────────────────────────────────────────────────────────────────────────
    trades_2 = pd.DataFrame({
        'pnl': [50, -30, 40, -25, 60, -20, 45, -30, 55, -25,
                50, -30, 40, -25, 60, -20, 45, -30, 55, -25],  # Same 60% win rate
        'lot_size': [1.0] * 20,
        'holding_bars': [2880] * 20,  # 2 days average hold
        'balance': [200.0] * 20,
    })
    
    result_2 = demo_scenario(
        "Scenario 2: Swing Trading (20 trades, 60% win rate, 2-day hold)",
        trades_2,
        spread_pips=0.5,
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 3: Real XAUUSD WF results (simulated 10,825 trades)
    # ─────────────────────────────────────────────────────────────────────────
    # Simulate realistic WF results: 40% win rate, profit factor 1.3
    np.random.seed(123)
    n_trades = 10825
    n_wins = int(n_trades * 0.40)
    n_losses = n_trades - n_wins
    
    # Generate realistic P&L distribution
    wins = np.random.lognormal(mean=3.0, sigma=0.5, size=n_wins)  # Avg ~$25
    losses = -np.random.lognormal(mean=2.5, sigma=0.5, size=n_losses)  # Avg -$15
    pnl_array = np.concatenate([wins, losses])
    np.random.shuffle(pnl_array)
    
    trades_3 = pd.DataFrame({
        'pnl': pnl_array,
        'lot_size': [1.0] * n_trades,
        'holding_bars': np.random.choice([0, 1440, 2880], size=n_trades, p=[0.7, 0.2, 0.1]),  # 70% intraday
        'balance': [200.0] * n_trades,
    })
    
    result_3 = demo_scenario(
        f"Scenario 3: XAUUSD WF Results (10,825 trades, realistic distribution)",
        trades_3,
        spread_pips=0.5,
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Summary Comparison
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("📊 SUMMARY COMPARISON")
    print(f"{'='*70}\n")
    
    comparison = pd.DataFrame({
        'Scenario': ['Scalping (100T)', 'Swing (20T)', f'WF ({n_trades:,}T)'],
        'Gross P&L': [result_1['gross_pnl'], result_2['gross_pnl'], result_3['gross_pnl']],
        'Spread Cost': [result_1['spread_cost'], result_2['spread_cost'], result_3['spread_cost']],
        'Net P&L': [result_1['net_pnl'], result_2['net_pnl'], result_3['net_pnl']],
        'Drag %': [result_1['turnover_drag'], result_2['turnover_drag'], result_3['turnover_drag']],
    })
    
    print(comparison.to_string(index=False, float_format=lambda x: f"${x:,.0f}" if abs(x) > 1 else f"{x:.1%}"))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Key Insights
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("💡 KEY INSIGHTS")
    print(f"{'='*70}\n")
    
    print("1. High-frequency strategies pay MORE in spread costs:")
    print(f"   - Scalping (100 trades):  ${result_1['spread_cost']:,.0f} spread cost")
    print(f"   - Swing (20 trades):      ${result_2['spread_cost']:,.0f} spread cost")
    print(f"   - 5× fewer trades = 5× lower cost\n")
    
    print("2. Spread cost is FIXED per trade:")
    print(f"   - Each trade costs: 2× × 0.5 pips × $10 = $10")
    print(f"   - Regardless of profit size!")
    print(f"   - Small wins (<$10) → negative after spread\n")
    
    print("3. Backtest overstates profitability:")
    scalping_overstate = (result_1['gross_pnl'] / result_1['net_pnl'] - 1) * 100 if result_1['net_pnl'] > 0 else float('inf')
    print(f"   - Scalping: {scalping_overstate:.0f}% overestimate")
    print(f"   - Ignoring spread → wrong strategy selection\n")
    
    print("4. Optimize for profit-per-trade, not just win rate:")
    print(f"   - 60% win rate with $8 avg win → LOSES money after spread")
    print(f"   - 60% win rate with $50 avg win → PROFITABLE after spread")
    print(f"   - Need: Avg Win > $10 + (Avg Loss × Loss Rate / Win Rate)\n")
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run full WF validation to calculate actual turnover drag")
    print("  2. Compare backtest results with/without turnover adjustment")
    print("  3. Set minimum profit-per-trade filter: avg_win >= $15")
    print("  4. Consider reducing trade frequency if turnover drag > 30%\n")


if __name__ == "__main__":
    main()
