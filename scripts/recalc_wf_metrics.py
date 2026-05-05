#!/usr/bin/env python3
"""
Post-process Walk-Forward Results to Add Metrics
================================================
Recalculate Sharpe/Calmar/Sortino metrics from existing WF trade data
without re-running full validation.

Usage:
    python scripts/recalc_wf_metrics.py outputs/walkforward_trades_acc1_v14pp_profit_sim_trades.csv
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xauusd_ai.infra.advanced_metrics import calculate_all_metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/recalc_wf_metrics.py <sim_trades.csv>")
        sys.exit(1)
    
    trades_file = Path(sys.argv[1])
    if not trades_file.exists():
        print(f"❌ File not found: {trades_file}")
        sys.exit(1)
    
    print("=" * 70)
    print("  Recalculating WF Metrics from Trade Data")
    print("=" * 70)
    print(f"  Input: {trades_file}")
    print()
    
    # Load trades
    trades = pd.read_csv(trades_file)
    print(f"  Total trades: {len(trades)}")
    
    if "fold" not in trades.columns:
        print("❌ No 'fold' column found in trades CSV")
        sys.exit(1)
    
    folds = sorted(trades["fold"].unique())
    print(f"  Total folds: {len(folds)}")
    print()
    
    # Calculate metrics per fold
    fold_metrics = []
    
    for fold_idx in folds:
        fold_trades = trades[trades["fold"] == fold_idx].copy()
        
        if len(fold_trades) < 2:
            # Not enough trades for metrics
            fold_metrics.append({
                "fold": fold_idx,
                "trades": len(fold_trades),
                "sharpe": None,
                "sortino": None,
                "calmar": None,
            })
            continue
        
        # Build balance series from balance_after column
        balance_series = fold_trades["balance_after"].tolist()
        
        # Add starting balance (assume $200 if not available)
        starting_balance = fold_trades.iloc[0].get("balance_before", 200.0)
        if pd.isna(starting_balance):
            starting_balance = 200.0
        balance_series.insert(0, starting_balance)
        
        # Calculate metrics
        metrics = calculate_all_metrics(
            balance_series=balance_series,
            periods_per_year=252,  # Daily returns
            risk_free_rate=0.03,   # 3% risk-free rate
        )
        
        fold_metrics.append({
            "fold": fold_idx,
            "trades": len(fold_trades),
            "starting_balance": starting_balance,
            "ending_balance": fold_trades.iloc[-1]["balance_after"],
            "return_pct": (fold_trades.iloc[-1]["balance_after"] / starting_balance - 1) * 100,
            "sharpe": metrics.get("sharpe"),
            "sortino": metrics.get("sortino"),
            "calmar": metrics.get("calmar"),
            "max_dd": metrics.get("max_dd"),
        })
    
    # Calculate averages
    valid_sharpe = [m["sharpe"] for m in fold_metrics if m["sharpe"] is not None]
    valid_sortino = [m["sortino"] for m in fold_metrics if m["sortino"] is not None]
    valid_calmar = [m["calmar"] for m in fold_metrics if m["calmar"] is not None]
    
    print("  📈 Risk-Adjusted Performance Metrics:")
    if valid_sharpe:
        print(f"     Sharpe Ratio   : {np.mean(valid_sharpe):.3f}  (>1.0 good, >2.0 excellent)")
        print(f"     Sortino Ratio  : {np.mean(valid_sortino):.3f}  (only penalizes downside risk)")
        print(f"     Calmar Ratio   : {np.mean(valid_calmar):.3f}  (return/max DD, >1.0 good)")
        print(f"     Metrics folds  : {len(valid_sharpe)}/{len(folds)}")
    else:
        print("     ⚠️  No valid metrics (need ≥2 trades per fold)")
    print()
    
    # Show per-fold breakdown
    print("  Per-Fold Metrics:")
    print("  " + "-" * 68)
    print("  Fold   Trades   Return%   Sharpe   Sortino   Calmar")
    print("  " + "-" * 68)
    
    for m in fold_metrics[:10]:  # Show first 10 folds
        sharpe_str = f"{m['sharpe']:.3f}" if m['sharpe'] is not None else "N/A"
        sortino_str = f"{m['sortino']:.3f}" if m['sortino'] is not None else "N/A"
        calmar_str = f"{m['calmar']:.3f}" if m['calmar'] is not None else "N/A"
        print(f"  {m['fold']:4d}   {m['trades']:6d}   {m['return_pct']:+7.1f}%   {sharpe_str:>7}  {sortino_str:>8}  {calmar_str:>7}")
    
    if len(fold_metrics) > 10:
        print(f"  ... ({len(fold_metrics) - 10} more folds)")
    
    print("  " + "-" * 68)
    print()
    print("✅ Metrics recalculated successfully")
    print()


if __name__ == "__main__":
    main()
