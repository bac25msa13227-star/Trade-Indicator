#!/usr/bin/env python3
"""
Monitor 24/7 automatic grid search progress.
Shows: configs tested, time elapsed, ETA, current best results.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

PROJECT_ROOT = Path(__file__).parent.parent

def get_process_info():
    """Check if grid search is running."""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        
        for line in result.stdout.split('\n'):
            if 'auto_grid_search_24_7.py' in line and 'python' in line:
                parts = line.split()
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                time_running = parts[9]
                return {
                    'running': True,
                    'pid': pid,
                    'cpu': cpu,
                    'mem': mem,
                    'time': time_running
                }
        
        return {'running': False}
    except:
        return {'running': False}

def load_results():
    """Load current results."""
    results_file = PROJECT_ROOT / 'outputs' / 'grid_search' / 'results.jsonl'
    
    if not results_file.exists():
        return []
    
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                pass
    
    return results

def calculate_eta(results, total_configs):
    """Calculate estimated time remaining."""
    if len(results) == 0:
        return None
    
    # Average duration per config
    durations = [r.get('duration_seconds', 0) for r in results if r.get('duration_seconds')]
    if not durations:
        return None
    
    avg_duration = sum(durations) / len(durations)
    remaining_configs = total_configs - len(results)
    remaining_seconds = remaining_configs * avg_duration
    
    return timedelta(seconds=int(remaining_seconds))

def format_timedelta(td):
    """Format timedelta nicely."""
    if td is None:
        return "Unknown"
    
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    
    return " ".join(parts) if parts else "<1m"

def main():
    """Monitor grid search."""
    
    print("="*80)
    print("📊 GRID SEARCH MONITOR")
    print("="*80)
    print(f"Time: {datetime.now()}\n")
    
    # Check if running
    proc_info = get_process_info()
    if proc_info['running']:
        print(f"✅ Grid search RUNNING")
        print(f"   PID: {proc_info['pid']}")
        print(f"   CPU: {proc_info['cpu']}%")
        print(f"   Memory: {proc_info['mem']}%")
        print(f"   Running for: {proc_info['time']}")
    else:
        print("❌ Grid search NOT running")
        print("   Start with: bash scripts/run_grid_search_background.sh")
    
    print()
    
    # Load results
    results = load_results()
    
    if not results:
        print("📁 No results yet")
        return
    
    # Get strategy from latest result
    # Estimate total configs based on strategy
    # For now, assume 'coarse' with 64 configs
    total_configs = 64  # This should be read from config or args
    
    # Progress
    print("📈 PROGRESS")
    print("-"*80)
    print(f"Configs tested: {len(results)}/{total_configs} ({len(results)/total_configs*100:.1f}%)")
    
    # Calculate ETA
    eta = calculate_eta(results, total_configs)
    if eta:
        print(f"ETA: {format_timedelta(eta)}")
    
    # Success rate
    successful = [r for r in results if r.get('exit_code') == 0]
    print(f"Successful: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    
    # Average duration
    durations = [r.get('duration_seconds', 0) for r in results if r.get('duration_seconds')]
    if durations:
        avg_minutes = sum(durations) / len(durations) / 60
        print(f"Avg time/config: {avg_minutes:.1f} min")
    
    print()
    
    # Current best results
    valid = [r for r in results if r.get('exit_code') == 0 and r.get('avg_daily_pnl') is not None]
    
    if valid:
        print("🏆 CURRENT TOP 5 (by avg daily P&L)")
        print("-"*80)
        
        by_pnl = sorted(valid, key=lambda x: x.get('avg_daily_pnl', 0), reverse=True)
        
        for i, r in enumerate(by_pnl[:5], 1):
            p = r.get('params', {})
            th = p.get('threshold', 0)
            risk = p.get('risk_pct', 0)
            mp = p.get('max_positions', 0)
            trail = p.get('trailing', (0, 0, 0))
            
            trail_str = f"trail{int(trail[1]*10)}" if trail[1] > 0 else "notrail"
            
            pnl = r.get('avg_daily_pnl', 0)
            pf = r.get('profit_factor', 0)
            wr = r.get('win_rate', 0) * 100
            
            print(f"{i}. ${pnl:7.2f}/day | "
                  f"th={th:.2f} r={risk}% mp={mp} {trail_str} | "
                  f"PF={pf:.2f} WR={wr:.1f}%")
    else:
        print("⚠️  No valid results yet")
    
    print()
    print("="*80)
    print("Refresh: python scripts/monitor_grid_search.py")
    print("Logs: tail -f outputs/grid_search/logs/*.log")
    print("Stop: kill $(cat outputs/grid_search/pid.txt)")
    print("="*80)

if __name__ == '__main__':
    main()
