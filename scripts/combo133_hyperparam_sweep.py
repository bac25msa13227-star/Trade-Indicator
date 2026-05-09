#!/usr/bin/env python3
"""
Hyperparameter sweep for combo133 realistic replay.
Tests: threshold [0.70-0.80], risk [2-6%], max_positions [3-7]
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.combo133_realistic_replay import apply_trailing_stop, replay_month_realistic

def run_hyperparam_sweep():
    """Run comprehensive hyperparameter sweep."""
    
    # Load trades
    trades_path = PROJECT_ROOT / 'outputs' / 'combo133_trades.csv'
    if not trades_path.exists():
        print(f"❌ Trades file not found: {trades_path}")
        return
    
    df = pd.read_csv(trades_path)
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.to_period('M')
    
    print(f"📊 Loaded {len(df):,} trades from {df['time'].min()} to {df['time'].max()}")
    print(f"   Win rate: {(df['net_rr'] > 0).mean()*100:.1f}%")
    print(f"   Mean net_rr: {df['net_rr'].mean():.3f}")
    print()
    
    # Hyperparameter grid
    thresholds = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
    risk_pcts = [2, 3, 4, 5, 6]
    max_positions_list = [3, 4, 5, 6, 7]
    cut_rate = 0.30  # Fixed from calibration
    starting_balance = 500.0
    
    print("🔍 HYPERPARAMETER SWEEP")
    print(f"   Thresholds: {thresholds}")
    print(f"   Risk %: {risk_pcts}")
    print(f"   Max positions: {max_positions_list}")
    print(f"   Cut rate: {cut_rate} (keep {(1-cut_rate)*100:.0f}% profit above BE)")
    print(f"   Starting balance: ${starting_balance}")
    print()
    
    # Run sweep
    results = []
    total_configs = len(thresholds) * len(risk_pcts) * len(max_positions_list)
    config_num = 0
    
    for threshold in thresholds:
        for risk_pct in risk_pcts:
            for max_positions in max_positions_list:
                config_num += 1
                
                # Filter trades by threshold
                trades_filtered = df[df['probability'] >= threshold].copy()
                
                if len(trades_filtered) == 0:
                    print(f"⚠️  [{config_num}/{total_configs}] th={threshold} r={risk_pct}% mp={max_positions}: NO TRADES")
                    continue
                
                # Run realistic WF
                monthly_results = []
                total_trail_cuts = 0
                
                for month_period, month_df in trades_filtered.groupby('month'):
                    month_result = replay_month_realistic(
                        trades=month_df,
                        capital=starting_balance,
                        risk_pct=risk_pct,  # Already in percentage (2, 3, 4, 5, 6)
                        max_positions=max_positions,
                        prob_min=threshold,
                        cut_rate=cut_rate
                    )
                    if month_result is not None:
                        # Compute daily P&L (assume 22 trading days/month)
                        month_result['daily_pnl'] = month_result['total_pnl'] / 22.0
                        month_result['month'] = str(month_period)
                        monthly_results.append(month_result)
                        total_trail_cuts += month_result['trail_cuts']
                
                # Skip if no valid monthly results
                if len(monthly_results) == 0:
                    print(f"⚠️  [{config_num}/{total_configs}] th={threshold} r={risk_pct}% mp={max_positions}: NO VALID MONTHS")
                    continue
                
                # Aggregate results
                monthly_df = pd.DataFrame(monthly_results)
                
                avg_daily = monthly_df['daily_pnl'].mean()
                median_daily = monthly_df['daily_pnl'].median()
                std_daily = monthly_df['daily_pnl'].std()
                min_daily = monthly_df['daily_pnl'].min()
                max_daily = monthly_df['daily_pnl'].max()
                
                # Success = months with daily P&L > $10
                success_months = (monthly_df['daily_pnl'] > 10).sum()
                total_months = len(monthly_df)
                success_rate = success_months / total_months if total_months > 0 else 0
                
                avg_wr = monthly_df['win_rate'].mean()
                avg_pf = monthly_df['profit_factor'].mean()
                avg_dd = monthly_df['max_dd'].mean()
                avg_trades = monthly_df['num_trades'].mean()
                avg_trail_cuts = total_trail_cuts / total_months if total_months > 0 else 0
                
                # Store result
                result = {
                    'threshold': threshold,
                    'risk_pct': risk_pct,
                    'max_positions': max_positions,
                    'cut_rate': cut_rate,
                    'avg_daily_pnl': avg_daily,
                    'median_daily_pnl': median_daily,
                    'std_daily_pnl': std_daily,
                    'min_daily_pnl': min_daily,
                    'max_daily_pnl': max_daily,
                    'success_months': success_months,
                    'total_months': total_months,
                    'success_rate': success_rate,
                    'avg_wr': avg_wr,
                    'avg_pf': avg_pf,
                    'avg_dd': avg_dd,
                    'avg_trades': avg_trades,
                    'avg_trail_cuts': avg_trail_cuts,
                    'total_trades': len(trades_filtered)
                }
                results.append(result)
                
                # Print progress
                print(f"[{config_num:3d}/{total_configs}] "
                      f"th={threshold:.2f} r={risk_pct:d}% mp={max_positions:d} | "
                      f"avg=${avg_daily:7.2f} med=${median_daily:6.2f} succ={success_rate*100:4.1f}% "
                      f"wr={avg_wr:5.1f}% pf={avg_pf:4.2f} trades={len(trades_filtered):5d}")
    
    print()
    print("=" * 120)
    print("✅ SWEEP COMPLETE")
    print("=" * 120)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Sort by avg_daily_pnl descending
    results_df = results_df.sort_values('avg_daily_pnl', ascending=False)
    
    # Save full results
    output_path = PROJECT_ROOT / 'outputs' / 'combo133_hyperparam_sweep.csv'
    results_df.to_csv(output_path, index=False)
    print(f"📁 Saved full results: {output_path}")
    
    # Also save JSON
    json_path = PROJECT_ROOT / 'outputs' / 'combo133_hyperparam_sweep.json'
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_configs': len(results_df),
            'results': results_df.to_dict('records')
        }, f, indent=2)
    print(f"📁 Saved JSON: {json_path}")
    print()
    
    # Show TOP 10 configs
    print("🏆 TOP 10 CONFIGS (by avg daily P&L):")
    print("=" * 120)
    print(f"{'Rank':<5} {'Threshold':<10} {'Risk%':<7} {'MaxPos':<8} {'Avg$/day':<12} "
          f"{'Med$/day':<12} {'Std$/day':<12} {'Succ%':<8} {'WR%':<7} {'PF':<7} {'Trades':<8}")
    print("-" * 120)
    
    for idx, row in results_df.head(10).iterrows():
        print(f"{idx+1:<5} "
              f"{row['threshold']:<10.2f} "
              f"{row['risk_pct']:<7.0f} "
              f"{row['max_positions']:<8.0f} "
              f"${row['avg_daily_pnl']:<11.2f} "
              f"${row['median_daily_pnl']:<11.2f} "
              f"${row['std_daily_pnl']:<11.2f} "
              f"{row['success_rate']*100:<7.1f}% "
              f"{row['avg_wr']:<6.1f}% "
              f"{row['avg_pf']:<6.2f} "
              f"{row['total_trades']:<8.0f}")
    
    print("=" * 120)
    print()
    
    # Show BOTTOM 10 configs
    print("💀 BOTTOM 10 CONFIGS (worst performers):")
    print("=" * 120)
    print(f"{'Rank':<5} {'Threshold':<10} {'Risk%':<7} {'MaxPos':<8} {'Avg$/day':<12} "
          f"{'Med$/day':<12} {'Std$/day':<12} {'Succ%':<8} {'WR%':<7} {'PF':<7} {'Trades':<8}")
    print("-" * 120)
    
    for idx, row in results_df.tail(10).iterrows():
        rank = len(results_df) - (len(results_df) - idx) + 1
        print(f"{rank:<5} "
              f"{row['threshold']:<10.2f} "
              f"{row['risk_pct']:<7.0f} "
              f"{row['max_positions']:<8.0f} "
              f"${row['avg_daily_pnl']:<11.2f} "
              f"${row['median_daily_pnl']:<11.2f} "
              f"${row['std_daily_pnl']:<11.2f} "
              f"{row['success_rate']*100:<7.1f}% "
              f"{row['avg_wr']:<6.1f}% "
              f"{row['avg_pf']:<6.2f} "
              f"{row['total_trades']:<8.0f}")
    
    print("=" * 120)
    print()
    
    # Parameter analysis
    print("📊 PARAMETER ANALYSIS:")
    print("=" * 120)
    
    # Best by threshold
    print("\n🎯 By Threshold:")
    threshold_stats = results_df.groupby('threshold').agg({
        'avg_daily_pnl': 'mean',
        'success_rate': 'mean',
        'avg_wr': 'mean',
        'avg_pf': 'mean',
        'total_trades': 'mean'
    }).round(2)
    print(threshold_stats)
    
    # Best by risk
    print("\n💰 By Risk %:")
    risk_stats = results_df.groupby('risk_pct').agg({
        'avg_daily_pnl': 'mean',
        'success_rate': 'mean',
        'avg_wr': 'mean',
        'avg_pf': 'mean'
    }).round(2)
    print(risk_stats)
    
    # Best by max_positions
    print("\n📈 By Max Positions:")
    pos_stats = results_df.groupby('max_positions').agg({
        'avg_daily_pnl': 'mean',
        'success_rate': 'mean',
        'avg_wr': 'mean',
        'avg_pf': 'mean'
    }).round(2)
    print(pos_stats)
    
    print("=" * 120)
    
    # Recommendation
    best = results_df.iloc[0]
    print()
    print("🎯 RECOMMENDED CONFIG:")
    print(f"   Threshold: {best['threshold']:.2f}")
    print(f"   Risk per trade: {best['risk_pct']:.0f}%")
    print(f"   Max positions: {best['max_positions']:.0f}")
    print(f"   Cut rate: {best['cut_rate']:.2f}")
    print()
    print(f"   Expected avg daily: ${best['avg_daily_pnl']:.2f} ± ${best['std_daily_pnl']:.2f}")
    print(f"   Success rate: {best['success_rate']*100:.1f}% ({best['success_months']:.0f}/{best['total_months']:.0f} months)")
    print(f"   Win rate: {best['avg_wr']:.1f}%")
    print(f"   Profit factor: {best['avg_pf']:.2f}")
    print(f"   Max DD: {best['avg_dd']:.1f}%")
    print(f"   Avg trades/month: {best['avg_trades']:.0f}")
    print(f"   Trailing cuts/month: {best['avg_trail_cuts']:.0f}")
    print()

if __name__ == '__main__':
    run_hyperparam_sweep()
