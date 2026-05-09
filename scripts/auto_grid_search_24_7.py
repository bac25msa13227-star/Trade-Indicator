#!/usr/bin/env python3
"""
Automatic 24/7 WF Grid Search System
Continuously searches for optimal XAUUSD trading configs with realistic simulation.
Addresses all known gaps: trailing SL, spread, slippage, live vs WF discrepancies.
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import itertools

PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# GRID SEARCH PARAMETERS
# ============================================================================

GRID_PARAMS = {
    # Signal thresholds to test
    'threshold': [0.60, 0.65, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80],
    
    # Risk per trade (%)
    'risk_pct': [2, 3, 4, 5, 6],
    
    # Max open positions
    'max_positions': [3, 4, 5, 6, 7],
    
    # Trailing SL settings (breakeven_rr, activation_rr, trail_mult)
    'trailing': [
        (0.5, 1.0, 1.0),   # Standard (current)
        (0.3, 0.8, 0.8),   # Aggressive trail
        (0.7, 1.2, 1.2),   # Conservative trail
        (0.0, 0.0, 0.0),   # NO TRAILING (baseline)
    ],
    
    # Slippage models (ATR fraction)
    'slippage': [0.005, 0.010, 0.015, 0.020, 0.025],  # 0.5% to 2.5% ATR
    
    # Spread (friction_r)
    'friction': [0.15, 0.17, 0.20],  # Conservative to aggressive
}

# Total configs = 8 × 5 × 5 × 4 × 5 × 3 = 24,000 configs!
# At 90 min/config = 36,000 hours = 4.1 YEARS for full grid!
# Need smart sampling strategy

# ============================================================================
# SMART SAMPLING STRATEGIES
# ============================================================================

def generate_smart_grid(strategy='coarse'):
    """
    Generate grid with smart sampling to reduce total configs.
    
    Strategies:
    - 'coarse': Test major variations only (~100 configs, ~6 days)
    - 'balanced': Medium granularity (~500 configs, ~30 days)
    - 'fine': High granularity (~2000 configs, ~4 months)
    - 'full': Everything (~24000 configs, ~4 years)
    """
    
    if strategy == 'coarse':
        # Test key thresholds, 2 risk levels, 2 positions, 2 trailing, 1 slippage
        return {
            'threshold': [0.60, 0.70, 0.76, 0.80],
            'risk_pct': [3, 5],
            'max_positions': [4, 6],
            'trailing': [
                (0.5, 1.0, 1.0),   # Standard
                (0.0, 0.0, 0.0),   # No trailing
            ],
            'slippage': [0.015],  # 1.5% ATR (middle)
            'friction': [0.17],   # Standard
        }
        # Total: 4 × 2 × 2 × 2 × 1 × 1 = 64 configs (~4 days)
    
    elif strategy == 'balanced':
        # More granular
        return {
            'threshold': [0.60, 0.65, 0.70, 0.74, 0.76, 0.78, 0.80],
            'risk_pct': [2, 3, 4, 5, 6],
            'max_positions': [3, 5, 7],
            'trailing': [
                (0.5, 1.0, 1.0),
                (0.3, 0.8, 0.8),
                (0.0, 0.0, 0.0),
            ],
            'slippage': [0.010, 0.015, 0.020],
            'friction': [0.15, 0.17, 0.20],
        }
        # Total: 7 × 5 × 3 × 3 × 3 × 3 = 2835 configs (~6 weeks)
    
    elif strategy == 'fine':
        # High granularity
        return GRID_PARAMS
    
    else:  # full
        return GRID_PARAMS

# ============================================================================
# CONFIG GENERATOR
# ============================================================================

def create_wf_config(params: Dict, config_id: int) -> Tuple[Path, str]:
    """Create WF config file from parameters."""
    
    threshold = params['threshold']
    risk_pct = params['risk_pct']
    max_pos = params['max_positions']
    be_rr, act_rr, trail_mult = params['trailing']
    slippage = params['slippage']
    friction = params['friction']
    
    # Generate config name
    trail_str = f"trail{int(act_rr*10)}" if act_rr > 0 else "notrail"
    config_name = (f"grid_{config_id:04d}_"
                   f"th{int(threshold*100)}_"
                   f"r{risk_pct}_"
                   f"mp{max_pos}_"
                   f"{trail_str}_"
                   f"slip{int(slippage*1000)}")
    
    config_content = f"""# Auto-generated WF config #{config_id}
# Threshold: {threshold}, Risk: {risk_pct}%, MaxPos: {max_pos}
# Trailing: BE={be_rr}, Act={act_rr}, Mult={trail_mult}
# Slippage: {slippage}, Friction: {friction}

app:
  model_path: outputs/acc1_combo133_202604_model.pkl
  scaler_path: outputs/acc1_combo133_202604_scaler.pkl
  model_meta_path: outputs/acc1_combo133_202604_meta.json
  log_level: INFO

market:
  symbol: XAUUSDm
  training_symbol: GC=F
  training_data_source: csv_folder
  csv_folder_path: src/xauusd_ai/real_data
  csv_timeframe: M15
  higher_timeframe: D1
  mid_timeframe: H1
  structure_timeframe: H4
  execution_timeframe: M5
  bars:
    D1: 100
    H1: 500
    H4: 200
    M30: 300
    M15: 300
    M5: 500
    M1: 200
  timezone: UTC

strategy:
  signal_threshold: {threshold}
  sideways_volatility_threshold: 0.20
  strong_volatility_threshold: 0.80
  sideway_min_confidence: {threshold}
  volatile_min_confidence: {threshold}

risk:
  mode: fixed_fractional
  risk_per_trade: {risk_pct / 100.0}
  max_open_positions: {max_pos}
  stop_loss_atr_multiple: 1.5
  take_profit_rr: 5.5
  entry_slippage_atr_frac: {slippage}
  min_confidence: {threshold}
  kill_switch_enabled: true
  daily_loss_limit_pct: 0.15
  max_drawdown_kill_pct: 0.15
  spread_cost_rr: {friction * 0.59}
  slippage_rr: {friction * 0.29}
  commission_rr: {friction * 0.12}
  use_dynamic_slippage: {str(slippage > 0).lower()}

execution:
  auto_trade: true
  trailing_sl:
    enabled: {str(act_rr > 0).lower()}
    breakeven_at_rr: {be_rr}
    activation_rr: {act_rr}
    trail_atr_multiple: {trail_mult}

training:
  use_ensemble: true
  retrain_on_startup: false
  live_learning_enabled: false
  max_train_bars: 30000
  backtest_initial_balance: 500.0
"""
    
    config_path = PROJECT_ROOT / 'configs' / 'grid_search' / f'{config_name}.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content)
    
    return config_path, config_name

# ============================================================================
# WF RUNNER
# ============================================================================

def run_wf_single(config_path: Path, config_name: str) -> Dict:
    """Run single WF and return results."""
    
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING WF: {config_name}")
    print(f"   Started: {datetime.now()}")
    print(f"{'='*80}\n")
    
    cmd = [
        'python',
        str(PROJECT_ROOT / 'scripts' / 'walkforward_ict_wyckoff.py'),
        str(config_path),
        '--test-start', '2023-01-01',
        '--monthly-reset',
        '--test-bars', '6000',
        '--step-bars', '6000',
        '--cache'
    ]
    
    start_time = datetime.now()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Parse results from output
    output_text = result.stdout + result.stderr
    
    # Extract metrics from WF output
    metrics = {
        'config_name': config_name,
        'duration_seconds': duration,
        'exit_code': result.returncode,
        'avg_daily_pnl': None,
        'success_rate': None,
        'win_rate': None,
        'profit_factor': None,
        'max_dd': None,
        'sharpe': None,
        'sortino': None,
        'calmar': None,
    }
    
    # Parse metrics from WF summary section
    for line in output_text.split('\n'):
        line_lower = line.lower()
        
        # Win Rate avg : 24.3%
        if 'win rate avg' in line_lower and ':' in line:
            try:
                val_str = line.split(':')[-1].strip().replace('%', '')
                metrics['win_rate'] = float(val_str) / 100
            except:
                pass
        
        # Profit Factor : 0.731
        elif 'profit factor' in line_lower and ':' in line:
            try:
                val_str = line.split(':')[-1].strip()
                metrics['profit_factor'] = float(val_str)
            except:
                pass
        
        # Sharpe Ratio : -3.029
        elif 'sharpe ratio' in line_lower and ':' in line:
            try:
                val_str = line.split(':')[-1].strip().split()[0]
                metrics['sharpe'] = float(val_str)
            except:
                pass
        
        # Sortino Ratio : -3.831
        elif 'sortino ratio' in line_lower and ':' in line:
            try:
                val_str = line.split(':')[-1].strip().split()[0]
                metrics['sortino'] = float(val_str)
            except:
                pass
        
        # Calmar Ratio : -1.269
        elif 'calmar ratio' in line_lower and ':' in line:
            try:
                val_str = line.split(':')[-1].strip().split()[0]
                metrics['calmar'] = float(val_str)
            except:
                pass
        
        # Max Drawdown : -15.20%
        elif 'max drawdown' in line_lower and ':' in line:
            try:
                val_str = line.split(':')[-1].strip().replace('%', '').split()[0]
                metrics['max_dd'] = float(val_str) / 100
            except:
                pass
        
        # Return/fold : -5.01% ($500 start per fold) → calculate avg daily PNL
        elif 'return/fold' in line_lower and '$500' in line:
            try:
                val_str = line.split(':')[-1].strip().replace('%', '').split()[0]
                return_pct = float(val_str) / 100
                # Assume ~40 folds, each ~30 days, $500 start
                # avg_daily_pnl = $500 * return_pct / 30 days
                metrics['avg_daily_pnl'] = 500 * return_pct / 30
            except:
                pass
    
    # Save detailed log
    log_path = PROJECT_ROOT / 'outputs' / 'grid_search' / f'{config_name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output_text)
    
    print(f"\n✅ COMPLETED: {config_name}")
    print(f"   Duration: {duration/60:.1f} min")
    print(f"   Exit code: {result.returncode}")
    print(f"   Avg daily: ${metrics['avg_daily_pnl']}")
    
    return metrics

# ============================================================================
# GRID SEARCH ORCHESTRATOR
# ============================================================================

def run_grid_search(strategy='coarse', resume_from=None):
    """
    Run automatic grid search.
    
    Args:
        strategy: 'coarse', 'balanced', 'fine', or 'full'
        resume_from: Resume from config ID (for interrupted runs)
    """
    
    print("="*80)
    print("🔬 AUTOMATIC 24/7 WF GRID SEARCH")
    print("="*80)
    print(f"\nStrategy: {strategy}")
    print(f"Started: {datetime.now()}")
    
    # Generate grid
    grid = generate_smart_grid(strategy)
    total_configs = (len(grid['threshold']) * 
                    len(grid['risk_pct']) * 
                    len(grid['max_positions']) * 
                    len(grid['trailing']) * 
                    len(grid['slippage']) * 
                    len(grid['friction']))
    
    print(f"\nGrid dimensions:")
    for key, values in grid.items():
        print(f"  {key}: {len(values)} values")
    print(f"\nTotal configs: {total_configs}")
    print(f"Estimated time: {total_configs * 90 / 60:.1f} hours ({total_configs * 90 / 60 / 24:.1f} days)")
    
    # Generate all config combinations
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    
    all_results = []
    config_id = 0
    
    # Load existing results if resuming
    results_file = PROJECT_ROOT / 'outputs' / 'grid_search' / 'results.jsonl'
    if resume_from and results_file.exists():
        with open(results_file, 'r') as f:
            all_results = [json.loads(line) for line in f]
        config_id = len(all_results)
        print(f"\n📂 Resuming from config #{config_id}")
    
    # Run grid search
    for combo in itertools.product(*values):
        config_id += 1
        
        # Skip if resuming and already done
        if resume_from and config_id <= resume_from:
            continue
        
        # Create param dict
        params = dict(zip(keys, combo))
        
        print(f"\n{'='*80}")
        print(f"CONFIG {config_id}/{total_configs} ({config_id/total_configs*100:.1f}%)")
        print(f"{'='*80}")
        print(f"Parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        # Create config file
        config_path, config_name = create_wf_config(params, config_id)
        
        # Run WF
        try:
            result = run_wf_single(config_path, config_name)
            result['params'] = params
            result['config_id'] = config_id
            all_results.append(result)
            
            # Save result incrementally
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted at config {config_id}. Can resume with --resume-from {config_id}")
            break
        except Exception as e:
            print(f"\n❌ ERROR in config {config_id}: {e}")
            result = {
                'config_id': config_id,
                'config_name': config_name,
                'params': params,
                'error': str(e),
                'exit_code': -1
            }
            all_results.append(result)
            with open(results_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
    
    # Final summary
    print("\n" + "="*80)
    print("✅ GRID SEARCH COMPLETE")
    print("="*80)
    print(f"\nTotal configs tested: {len(all_results)}")
    print(f"Results saved: {results_file}")
    
    # Analyze results
    analyze_results(all_results)

# ============================================================================
# RESULTS ANALYZER
# ============================================================================

def analyze_results(results: List[Dict]):
    """Analyze and rank grid search results."""
    
    # Filter successful runs
    valid = [r for r in results if r.get('exit_code') == 0 and r.get('avg_daily_pnl') is not None]
    
    if not valid:
        print("\n⚠️  No valid results to analyze")
        return
    
    print(f"\n📊 ANALYZING {len(valid)} VALID RESULTS")
    print("="*80)
    
    # Sort by different objectives
    
    # 1. Max daily P&L
    by_pnl = sorted(valid, key=lambda x: x.get('avg_daily_pnl', 0), reverse=True)
    print("\n🏆 TOP 10 BY AVG DAILY P&L:")
    print("-"*80)
    for i, r in enumerate(by_pnl[:10], 1):
        p = r['params']
        print(f"{i:2d}. ${r['avg_daily_pnl']:7.2f}/day | "
              f"th={p['threshold']:.2f} r={p['risk_pct']}% mp={p['max_positions']} | "
              f"PF={r.get('profit_factor', 0):.2f} WR={r.get('win_rate', 0)*100:.1f}%")
    
    # 2. Best risk-adjusted (Sharpe ratio)
    by_sharpe = sorted([r for r in valid if r.get('sharpe')], 
                       key=lambda x: x.get('sharpe', 0), reverse=True)
    if by_sharpe:
        print("\n📈 TOP 10 BY SHARPE RATIO:")
        print("-"*80)
        for i, r in enumerate(by_sharpe[:10], 1):
            p = r['params']
            print(f"{i:2d}. Sharpe={r['sharpe']:5.2f} | ${r['avg_daily_pnl']:7.2f}/day | "
                  f"th={p['threshold']:.2f} r={p['risk_pct']}% mp={p['max_positions']}")
    
    # 3. Best capital preservation (low drawdown + positive returns)
    by_safety = sorted([r for r in valid if r.get('max_dd') and r.get('avg_daily_pnl', 0) > 0],
                       key=lambda x: x.get('max_dd', 100))
    if by_safety:
        print("\n🛡️  TOP 10 BY CAPITAL PRESERVATION (Low DD + Positive):")
        print("-"*80)
        for i, r in enumerate(by_safety[:10], 1):
            p = r['params']
            print(f"{i:2d}. DD={r['max_dd']:5.1f}% | ${r['avg_daily_pnl']:7.2f}/day | "
                  f"th={p['threshold']:.2f} r={p['risk_pct']}% mp={p['max_positions']}")
    
    # Save top results
    summary = {
        'total_tested': len(results),
        'valid_results': len(valid),
        'top_by_pnl': by_pnl[:10],
        'top_by_sharpe': by_sharpe[:10] if by_sharpe else [],
        'top_by_safety': by_safety[:10] if by_safety else [],
    }
    
    summary_file = PROJECT_ROOT / 'outputs' / 'grid_search' / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📁 Summary saved: {summary_file}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='24/7 Automatic WF Grid Search')
    parser.add_argument('--strategy', default='coarse', 
                       choices=['coarse', 'balanced', 'fine', 'full'],
                       help='Grid sampling strategy')
    parser.add_argument('--resume-from', type=int, default=None,
                       help='Resume from config ID')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze existing results')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Load and analyze existing results
        results_file = PROJECT_ROOT / 'outputs' / 'grid_search' / 'results.jsonl'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = [json.loads(line) for line in f]
            analyze_results(results)
        else:
            print(f"❌ No results file found: {results_file}")
    else:
        # Run grid search
        run_grid_search(strategy=args.strategy, resume_from=args.resume_from)
