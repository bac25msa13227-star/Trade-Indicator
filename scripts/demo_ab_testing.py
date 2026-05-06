#!/usr/bin/env python3
"""
A/B Testing Framework Demo

Demonstrates:
1. Treatment assignment (deterministic, 50/50 split)
2. Logging outcomes to JSONL
3. Statistical analysis (p-value, Cohen's d, confidence intervals)

Usage:
    python scripts/demo_ab_testing.py
"""

import sys
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xauusd_ai.infra.ab_testing import ABTestManager


def simulate_trading_experiment(n_trades: int = 100) -> None:
    """
    Simulate A/B test comparing static vs dynamic slippage.
    
    Scenario:
    - Control (static slippage): Win rate 40%, avg PnL +5.0 RR
    - Treatment (dynamic slippage): Win rate 42%, avg PnL +6.5 RR
    
    Dynamic slippage should show:
    - Higher win rate (more realistic friction modeling → better trade selection)
    - Higher avg PnL (lower slippage cost on winning trades)
    """
    print("=" * 80)
    print("A/B TESTING FRAMEWORK DEMO")
    print("=" * 80)
    print(f"\nSimulating {n_trades} trades...")
    print("  Control: Static slippage (0.5 pips fixed)")
    print("  Treatment: Dynamic slippage (0.5-6 pips, ATR-based)\n")
    
    ab_manager = ABTestManager(log_file="outputs/demo_ab_test.jsonl", seed=42)
    
    # Simulate trades
    control_wins = 0
    treatment_wins = 0
    
    for i in range(n_trades):
        signal_id = f"trade_{i:04d}"
        treatment = ab_manager.assign_treatment(signal_id)
        
        # Simulate outcome based on treatment
        if treatment == "control":
            # Static slippage: 40% win rate, avg +5.0 RR
            is_win = random.random() < 0.40
            pnl = random.gauss(5.0, 2.0) if is_win else random.gauss(-1.0, 0.5)
            control_wins += is_win
        else:
            # Dynamic slippage: 42% win rate, avg +6.5 RR
            is_win = random.random() < 0.42
            pnl = random.gauss(6.5, 2.5) if is_win else random.gauss(-1.0, 0.5)
            treatment_wins += is_win
        
        # Log outcome
        ab_manager.log_result(
            signal_id=signal_id,
            treatment=treatment,
            outcome={
                "pnl": pnl,
                "entry_price": 2650.0 + random.uniform(-10, 10),
                "exit_price": 2650.0 + pnl * 0.5 + random.uniform(-5, 5),
                "slippage_rr": 0.05 if treatment == "control" else 0.03,
            }
        )
    
    # Show summary
    print("\n" + "-" * 80)
    print("SIMULATION RESULTS")
    print("-" * 80)
    
    summary = ab_manager.get_summary()
    print(f"\nTotal signals: {summary['total_signals']}")
    print(f"  Control: {summary['control_count']} trades")
    print(f"  Treatment: {summary['treatment_count']} trades\n")
    
    print(f"Win rates:")
    print(f"  Control: {control_wins}/{summary['control_count']} = {control_wins/summary['control_count']:.1%}")
    print(f"  Treatment: {treatment_wins}/{summary['treatment_count']} = {treatment_wins/summary['treatment_count']:.1%}\n")
    
    print(f"Total PnL:")
    print(f"  Control: {summary['control_total_pnl']:+.2f} RR (avg {summary['control_mean_pnl']:+.2f})")
    print(f"  Treatment: {summary['treatment_total_pnl']:+.2f} RR (avg {summary['treatment_mean_pnl']:+.2f})\n")
    
    # Statistical analysis
    print("\n" + "-" * 80)
    print("STATISTICAL ANALYSIS")
    print("-" * 80)
    
    analysis = ab_manager.analyze()
    
    if "error" in analysis:
        print(f"\n⚠️  {analysis['error']}")
        print(f"  Control: {analysis.get('control_count', 0)} samples")
        print(f"  Treatment: {analysis.get('treatment_count', 0)} samples")
        print("\n  Need >= 30 samples per group for valid analysis")
    else:
        print(f"\nMean PnL:")
        print(f"  Control: {analysis['control_mean']:.3f} RR")
        print(f"  Treatment: {analysis['treatment_mean']:.3f} RR")
        print(f"  Difference: {analysis['mean_difference']:+.3f} RR")
        print(f"  95% CI: [{analysis['confidence_interval']['lower']:.3f}, {analysis['confidence_interval']['upper']:.3f}]")
        
        print(f"\nStatistical significance:")
        print(f"  p-value: {analysis['p_value']:.4f}")
        print(f"  t-statistic: {analysis['t_statistic']:.3f}")
        
        print(f"\nEffect size:")
        print(f"  Cohen's d: {analysis['effect_size']:.3f}")
        
        effect_interpretation = (
            "negligible" if abs(analysis['effect_size']) < 0.2 else
            "small" if abs(analysis['effect_size']) < 0.5 else
            "medium" if abs(analysis['effect_size']) < 0.8 else
            "large"
        )
        print(f"  Interpretation: {effect_interpretation} effect")
        
        print(f"\n{'=' * 80}")
        print(f"CONCLUSION")
        print(f"{'=' * 80}")
        print(f"\n{analysis['interpretation']}\n")
        
        # Decision recommendation
        if analysis['p_value'] < 0.05 and analysis['mean_difference'] > 0:
            print("✅ RECOMMENDATION: Deploy dynamic slippage to production")
            print("   Treatment significantly outperforms control (p < 0.05)")
        elif analysis['p_value'] < 0.05 and analysis['mean_difference'] < 0:
            print("❌ RECOMMENDATION: Keep static slippage")
            print("   Treatment significantly underperforms control (p < 0.05)")
        else:
            print("⏸️  RECOMMENDATION: Continue A/B test, need more data")
            print("   No significant difference detected yet (p >= 0.05)")
    
    print(f"\n📁 Full results saved to: outputs/demo_ab_test.jsonl")
    print(f"   View with: tail -20 outputs/demo_ab_test.jsonl | jq .\n")


if __name__ == "__main__":
    random.seed(42)  # Reproducible results
    simulate_trading_experiment(n_trades=100)
