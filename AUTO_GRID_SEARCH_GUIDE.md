# 24/7 AUTOMATIC WF GRID SEARCH SYSTEM

**Generated:** 2026-05-08  
**Purpose:** Continuously search for optimal XAUUSD trading configs with REALISTIC simulation

---

## 🎯 WHAT THIS SOLVES

### Known GAPS (Critical Issues):
1. ✅ **Trailing SL bug:** combo133 has `gap_rr=0.0` for ALL trades → 30-50% overstatement
2. ✅ **Spread:** Static friction may not match Live reality
3. ✅ **Slippage:** Dynamic slippage (2% ATR) may be too aggressive or too conservative
4. ✅ **Live vs WF:** "Live cắn BE rất nhiều" but WF doesn't reflect this

### Solution:
**Automatic 24/7 grid search** testing:
- **Thresholds:** 0.60 → 0.80 (find optimal signal quality)
- **Risk:** 2-6% (find best sizing)
- **Positions:** 3-7 (find optimal diversification)
- **Trailing:** Standard, aggressive, conservative, NONE (find best exit)
- **Slippage:** 0.5%-2.5% ATR (find realistic model)
- **Friction:** 0.15-0.20 (find accurate spread cost)

**Full WF from 2023 to present** with **$500 monthly reset** to find configs that:
1. ✅ Maximize profit
2. ✅ Preserve capital (low drawdown)
3. ✅ Match Live behavior (realistic simulation)

---

## 🚀 QUICK START

### Option 1: COARSE Grid (Recommended First) ⭐
**64 configs × 90 min = ~4 days**

```bash
# Start in background
chmod +x scripts/run_grid_search_background.sh
bash scripts/run_grid_search_background.sh coarse

# Monitor progress
python scripts/monitor_grid_search.py

# View logs
tail -f outputs/grid_search/logs/*.log
```

### Option 2: BALANCED Grid (More thorough)
**~2,835 configs × 90 min = ~6 weeks**

```bash
bash scripts/run_grid_search_background.sh balanced
```

### Option 3: FINE Grid (Comprehensive)
**~24,000 configs × 90 min = ~4 years (!)**

```bash
# Don't actually run this unless you have a cluster!
bash scripts/run_grid_search_background.sh fine
```

---

## 📊 GRID PARAMETERS

### COARSE Strategy (64 configs, ~4 days):
```python
{
    'threshold': [0.60, 0.70, 0.76, 0.80],           # 4 values
    'risk_pct': [3, 5],                              # 2 values
    'max_positions': [4, 6],                         # 2 values
    'trailing': [                                    # 2 values
        (0.5, 1.0, 1.0),   # Standard
        (0.0, 0.0, 0.0),   # NO trailing (baseline)
    ],
    'slippage': [0.015],                             # 1 value (1.5% ATR)
    'friction': [0.17],                              # 1 value (standard)
}
# Total: 4 × 2 × 2 × 2 × 1 × 1 = 64 configs
```

### BALANCED Strategy (~2,835 configs, ~6 weeks):
```python
{
    'threshold': [0.60, 0.65, 0.70, 0.74, 0.76, 0.78, 0.80],  # 7
    'risk_pct': [2, 3, 4, 5, 6],                              # 5
    'max_positions': [3, 5, 7],                               # 3
    'trailing': [                                             # 3
        (0.5, 1.0, 1.0),   # Standard
        (0.3, 0.8, 0.8),   # Aggressive
        (0.0, 0.0, 0.0),   # NO trailing
    ],
    'slippage': [0.010, 0.015, 0.020],                       # 3
    'friction': [0.15, 0.17, 0.20],                          # 3
}
# Total: 7 × 5 × 3 × 3 × 3 × 3 = 2,835 configs
```

### FULL Strategy (~24,000 configs, ~4 years):
```python
{
    'threshold': [0.60, 0.65, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80],  # 8
    'risk_pct': [2, 3, 4, 5, 6],                                    # 5
    'max_positions': [3, 4, 5, 6, 7],                               # 5
    'trailing': [                                                   # 4
        (0.5, 1.0, 1.0),   # Standard
        (0.3, 0.8, 0.8),   # Aggressive
        (0.7, 1.2, 1.2),   # Conservative
        (0.0, 0.0, 0.0),   # NO trailing
    ],
    'slippage': [0.005, 0.010, 0.015, 0.020, 0.025],              # 5
    'friction': [0.15, 0.17, 0.20],                                # 3
}
# Total: 8 × 5 × 5 × 4 × 5 × 3 = 24,000 configs
```

---

## 🎯 OBJECTIVES & RANKING

Results are ranked by 3 objectives:

### 1. Max Profit
**Sort by:** Avg daily P&L (descending)  
**Goal:** Highest absolute returns  
**Use case:** Aggressive trading, high risk tolerance

### 2. Max Risk-Adjusted Returns
**Sort by:** Sharpe ratio (descending)  
**Goal:** Best returns per unit of risk  
**Use case:** Balanced approach, institutional-style

### 3. Max Capital Preservation
**Sort by:** Max drawdown (ascending) + positive returns  
**Goal:** Lowest drawdown while staying profitable  
**Use case:** Conservative, capital protection priority

---

## 📈 MONITORING

### Check Progress:
```bash
python scripts/monitor_grid_search.py
```

**Output:**
```
================================================================================
📊 GRID SEARCH MONITOR
================================================================================
Time: 2026-05-08 23:45:00

✅ Grid search RUNNING
   PID: 12345
   CPU: 95.2%
   Memory: 8.3%
   Running for: 2:34:15

📈 PROGRESS
--------------------------------------------------------------------------------
Configs tested: 12/64 (18.8%)
ETA: 3d 14h 32m
Successful: 11/12 (91.7%)
Avg time/config: 87.3 min

🏆 CURRENT TOP 5 (by avg daily P&L)
--------------------------------------------------------------------------------
1. $ 52.34/day | th=0.76 r=5% mp=6 trail10 | PF=1.38 WR=51.2%
2. $ 48.21/day | th=0.70 r=6% mp=6 notrail | PF=1.25 WR=49.8%
3. $ 45.67/day | th=0.80 r=5% mp=4 trail10 | PF=1.52 WR=53.1%
4. $ 43.89/day | th=0.76 r=3% mp=6 trail10 | PF=1.41 WR=50.5%
5. $ 42.15/day | th=0.70 r=5% mp=6 trail10 | PF=1.33 WR=48.9%
```

### View Live Logs:
```bash
tail -f outputs/grid_search/logs/*.log
```

### Stop Grid Search:
```bash
# Get PID
cat outputs/grid_search/pid.txt

# Kill process
kill <PID>

# Or directly
kill $(cat outputs/grid_search/pid.txt)
```

### Resume Interrupted Run:
```bash
# Find last completed config
tail outputs/grid_search/results.jsonl | grep config_id

# Resume from next config
python scripts/auto_grid_search_24_7.py --strategy coarse --resume-from 12
```

---

## 📁 OUTPUT FILES

### Results Database:
```
outputs/grid_search/results.jsonl
```
**Format:** One JSON object per line, one line per config  
**Fields:**
- `config_id`: Unique ID
- `config_name`: Descriptive name
- `params`: Dict of hyperparameters
- `avg_daily_pnl`: Average daily P&L
- `success_rate`: % of months profitable
- `win_rate`: Trade win rate
- `profit_factor`: Gross profit / gross loss
- `max_dd`: Maximum drawdown %
- `sharpe`: Sharpe ratio (if available)
- `sortino`: Sortino ratio (if available)
- `calmar`: Calmar ratio (if available)
- `duration_seconds`: Time to run
- `exit_code`: 0=success, non-zero=error

### Summary Report:
```
outputs/grid_search/summary.json
```
**Contains:**
- Top 10 by profit
- Top 10 by Sharpe ratio
- Top 10 by capital preservation

### Individual Config Logs:
```
outputs/grid_search/grid_<id>_<params>.log
```

### Generated Configs:
```
configs/grid_search/grid_<id>_<params>.yaml
```

---

## 🔬 ANALYSIS

### After Grid Search Completes:

```bash
# Analyze all results
python scripts/auto_grid_search_24_7.py --analyze-only
```

**Output:**
```
📊 ANALYZING 64 VALID RESULTS
================================================================================

🏆 TOP 10 BY AVG DAILY P&L:
--------------------------------------------------------------------------------
 1. $ 52.34/day | th=0.76 r=5% mp=6 | PF=1.38 WR=51.2%
 2. $ 48.21/day | th=0.70 r=6% mp=6 | PF=1.25 WR=49.8%
...

📈 TOP 10 BY SHARPE RATIO:
--------------------------------------------------------------------------------
 1. Sharpe= 2.45 | $ 45.67/day | th=0.80 r=5% mp=4
 2. Sharpe= 2.31 | $ 43.89/day | th=0.76 r=3% mp=6
...

🛡️  TOP 10 BY CAPITAL PRESERVATION (Low DD + Positive):
--------------------------------------------------------------------------------
 1. DD=18.3% | $ 38.92/day | th=0.80 r=3% mp=4
 2. DD=21.7% | $ 41.25/day | th=0.76 r=3% mp=5
...
```

### Compare with Realistic Replay:

Load results and check if WF matches replay predictions:

```python
import json
import pandas as pd

# Load WF results
with open('outputs/grid_search/results.jsonl') as f:
    wf_results = [json.loads(line) for line in f]

# Load replay predictions
with open('outputs/combo133_hyperparam_sweep.json') as f:
    replay = json.load(f)

# Compare (example for threshold=0.80, risk=5%, mp=6)
wf_config = [r for r in wf_results 
             if r['params']['threshold']==0.80 
             and r['params']['risk_pct']==5 
             and r['params']['max_positions']==6][0]

replay_config = [r for r in replay['results'] 
                 if r['threshold']==0.80 
                 and r['risk_pct']==5 
                 and r['max_positions']==6][0]

print(f"Replay predicted: ${replay_config['avg_daily_pnl']:.2f}/day")
print(f"WF actual:        ${wf_config['avg_daily_pnl']:.2f}/day")
print(f"Gap:              {(wf_config['avg_daily_pnl'] - replay_config['avg_daily_pnl']) / replay_config['avg_daily_pnl'] * 100:.1f}%")
```

---

## ⚠️ KNOWN LIMITATIONS

### 1. Time Requirement
- **Coarse:** ~4 days
- **Balanced:** ~6 weeks  
- **Full:** ~4 years (IMPRACTICAL!)

**Mitigation:** Start with COARSE, then refine around best configs.

### 2. Compute Resources
- **CPU:** 95-100% usage during WF runs
- **Memory:** 6-10 GB per config
- **Disk:** ~100 MB per config (logs + results)

**Mitigation:** Run on dedicated machine or cloud instance.

### 3. Interruption Risk
- Long-running process may be interrupted
- Network issues, power outages, system restarts

**Mitigation:** 
- Use `nohup` for background execution
- Results saved incrementally (can resume)
- Use `--resume-from` flag to continue

### 4. Model Retraining
- Each WF retrains model → NOT using optimized combo133
- May find worse configs than realistic replay

**Mitigation:** Grid includes NO TRAILING baseline to compare.

---

## 🎯 RECOMMENDED WORKFLOW

### Phase 1: COARSE Grid (~4 days)
```bash
bash scripts/run_grid_search_background.sh coarse
```
**Goal:** Find best regions of hyperparameter space

### Phase 2: Analyze Top Configs
```bash
python scripts/auto_grid_search_24_7.py --analyze-only
```
**Goal:** Identify top 10 configs by each objective

### Phase 3: Refine Around Best
Create custom grid focusing on best threshold/risk/positions ranges:

```python
# Example: If best configs all use th=0.76-0.80, risk=4-6%, mp=5-7
# Create refined grid:
REFINED_GRID = {
    'threshold': [0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80],
    'risk_pct': [4, 4.5, 5, 5.5, 6],
    'max_positions': [5, 6, 7],
    # ... other params
}
```

### Phase 4: Validate Top 3
- Run paper trading with top 3 configs for 1 week
- Compare with WF predictions
- Choose final config for Live

---

## 🚨 EMERGENCY STOPS

### If Grid Search Crashes:
```bash
# Check error logs
tail -100 outputs/grid_search/logs/*.log

# Resume from last successful config
python scripts/auto_grid_search_24_7.py --strategy coarse --resume-from <last_id>
```

### If System Overload:
```bash
# Kill grid search
kill $(cat outputs/grid_search/pid.txt)

# Reduce to lighter strategy
bash scripts/run_grid_search_background.sh coarse
```

### If Results Look Wrong:
```bash
# Analyze partial results
python scripts/auto_grid_search_24_7.py --analyze-only

# Check individual config logs
cat outputs/grid_search/<config_name>.log
```

---

## 📞 SUPPORT

If issues arise:
1. Check monitor output: `python scripts/monitor_grid_search.py`
2. Check logs: `tail -f outputs/grid_search/logs/*.log`
3. Verify process running: `ps aux | grep auto_grid_search`
4. Check disk space: `df -h`
5. Check memory: `free -h` (Linux) or `vm_stat` (macOS)

---

**Generated:** 2026-05-08  
**Status:** Ready to run  
**Estimated completion (COARSE):** 2026-05-12 (4 days)
