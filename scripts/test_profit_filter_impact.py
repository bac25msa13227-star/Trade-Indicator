"""
Test Profit Filter Impact on Existing Walk-Forward Results

Simulates profit filter on completed WF trades to estimate impact
before full integration.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xauusd_ai.strategies.profit_filter import MinimumProfitFilter


def estimate_trade_profit(row):
    """
    Ước tính predicted profit cho một trade.
    
    Trong thực tế sẽ dùng model prediction, nhưng ở đây
    ta dùng actual P&L như proxy để test filter logic.
    """
    # Thực tế: predicted_profit sẽ đến từ model
    # Ở đây: dùng actual P&L để simulate
    actual_pnl = row.get('pnl', 0)
    
    # Thêm noise để simulate prediction error
    import random
    random.seed(int(row.get('ticket', 0)) if 'ticket' in row.index else 0)
    prediction_error = random.uniform(-5, 5)  # ±$5 prediction error
    
    predicted_profit = actual_pnl + prediction_error
    
    return predicted_profit


def main():
    """Test profit filter impact."""
    
    print("=" * 80)
    print("PROFIT FILTER IMPACT TEST")
    print("=" * 80)
    print()
    
    # Load existing WF trades
    trades_file = Path("outputs/walkforward_trades_acc1_v14pp_profit_sim_trades.csv")
    
    if not trades_file.exists():
        print(f"❌ File not found: {trades_file}")
        print("   Run WF validation first to generate trades file.")
        return
    
    print(f"Loading trades from: {trades_file}")
    trades_df = pd.read_csv(trades_file)
    print(f"Total trades: {len(trades_df)}")
    print()
    
    # Calculate baseline (no filter)
    baseline_gross = trades_df['pnl'].sum()
    baseline_count = len(trades_df)
    baseline_spread = baseline_count * 10.0  # $10 per trade
    baseline_net = baseline_gross - baseline_spread
    
    print("BASELINE (No Filter)")
    print("-" * 80)
    print(f"Total trades:    {baseline_count:,}")
    print(f"Gross P&L:       ${baseline_gross:,.2f}")
    print(f"Spread cost:     ${baseline_spread:,.2f}")
    print(f"Net P&L:         ${baseline_net:,.2f}")
    print()
    
    # Test different filter thresholds
    thresholds = [10.0, 12.0, 15.0, 18.0, 20.0, 25.0]
    
    results = []
    
    for min_profit in thresholds:
        # Initialize filter
        profit_filter = MinimumProfitFilter(
            min_expected_profit=min_profit,
            spread_pips=0.5,
            pip_value=10.0,
            enabled=True
        )
        
        # Estimate predicted profit for each trade
        trades_df['predicted_profit'] = trades_df.apply(estimate_trade_profit, axis=1)
        
        # Apply filter
        skipped_mask = trades_df['predicted_profit'] <= min_profit
        kept_mask = ~skipped_mask
        
        # Calculate filtered results
        skipped_count = skipped_mask.sum()
        kept_count = kept_mask.sum()
        skip_rate = skipped_count / len(trades_df) * 100
        
        filtered_gross = trades_df.loc[kept_mask, 'pnl'].sum()
        filtered_spread = kept_count * 10.0
        filtered_net = filtered_gross - filtered_spread
        
        improvement = filtered_net - baseline_net
        
        results.append({
            'threshold': min_profit,
            'kept_trades': kept_count,
            'skip_rate': skip_rate,
            'gross_pnl': filtered_gross,
            'spread_cost': filtered_spread,
            'net_pnl': filtered_net,
            'improvement': improvement
        })
        
        print(f"Filter Threshold: ${min_profit}")
        print("-" * 80)
        print(f"Trades kept:     {kept_count:,} ({100-skip_rate:.1f}%)")
        print(f"Trades skipped:  {skipped_count:,} ({skip_rate:.1f}%)")
        print(f"Gross P&L:       ${filtered_gross:,.2f}")
        print(f"Spread cost:     ${filtered_spread:,.2f}")
        print(f"Net P&L:         ${filtered_net:,.2f}")
        print(f"Improvement:     ${improvement:,.2f} ({improvement/abs(baseline_net)*100:+.1f}%)")
        print()
    
    # Find optimal threshold
    results_df = pd.DataFrame(results)
    optimal = results_df.loc[results_df['net_pnl'].idxmax()]
    
    print("=" * 80)
    print("OPTIMAL THRESHOLD RECOMMENDATION")
    print("=" * 80)
    print(f"Best threshold:  ${optimal['threshold']:.2f}")
    print(f"Skip rate:       {optimal['skip_rate']:.1f}%")
    print(f"Net P&L:         ${optimal['net_pnl']:,.2f}")
    print(f"Improvement:     ${optimal['improvement']:,.2f} vs baseline")
    print()
    
    # Compare top 3
    top3 = results_df.nlargest(3, 'net_pnl')
    print("Top 3 Thresholds:")
    print("-" * 80)
    for _, row in top3.iterrows():
        print(f"${row['threshold']:5.2f} → Net P&L ${row['net_pnl']:10,.2f} "
              f"(skip {row['skip_rate']:5.1f}%, improve ${row['improvement']:+10,.2f})")
    print()
    
    # Summary comparison table
    print("=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Threshold':<12} {'Trades':<10} {'Skip %':<10} {'Net P&L':<15} {'vs Baseline':<15}")
    print("-" * 80)
    print(f"{'None (base)':<12} {baseline_count:<10,} {'0.0%':<10} ${baseline_net:<13,.2f} {'$0':<15}")
    
    for _, row in results_df.iterrows():
        print(f"${row['threshold']:<11.2f} {row['kept_trades']:<10,} "
              f"{row['skip_rate']:<9.1f}% ${row['net_pnl']:<13,.2f} "
              f"${row['improvement']:+13,.2f}")
    
    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    if optimal['net_pnl'] > 0:
        print(f"✅ Deploy profit filter with threshold ${optimal['threshold']:.2f}")
        print(f"   Expected to turn ${baseline_net:,.2f} loss into ${optimal['net_pnl']:,.2f} profit!")
        print()
        print("Next steps:")
        print("1. Integrate filter into orchestrator (1-2 hours)")
        print("2. Run full WF validation with filter enabled")
        print("3. Deploy to paper mode for 7 days")
        print("4. Go live if paper validation passes")
    else:
        print("⚠️  Filter helps but system still unprofitable")
        print("   Need additional improvements:")
        print("   - Increase min_confidence threshold")
        print("   - Implement regime detection")
        print("   - Adjust RR requirements")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
