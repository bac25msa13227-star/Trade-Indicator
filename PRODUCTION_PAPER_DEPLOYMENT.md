# Production Deployment Guide — Paper Mode Validation

**Date:** May 6, 2026  
**Version:** 1.0 (Monthly Reset Fixed)  
**Status:** Ready for Deployment

---

## ✅ Prerequisites Completed

1. ✅ **Monthly reset fixed** — Resets balance to $200 every ~30 days
2. ✅ **Dynamic slippage enabled** — ATR-based slippage model active
3. ✅ **Profit filter validated** — Skip rate 60-82%, $15 threshold optimal
4. ✅ **WF validation complete** — 15 folds with monthly reset, realistic returns
5. ✅ **Code pushed to main** — Commit 8922d97

---

## 📋 Deployment Steps (Windows Production)

### **Step 1: Pull Latest Code**

```powershell
cd "<REPO_PATH>"  # e.g., C:\Users\User\Documents\Trade Indicator
git fetch origin
git checkout main
git pull origin main
git log --oneline -5  # Verify latest commit is 8922d97
```

**Expected output:**
```
8922d97 fix: monthly reset logic - reset every 30 days properly
a54e78f feat: add monthly reset and improve WF realism
78e16fd Merge pull request #X from feature/turnover-profit-filter
...
```

---

### **Step 2: Verify Configuration**

**Check ACC1 config** (configs/live_acc1.yaml):
```yaml
execution:
  mode: paper  # ✅ PAPER MODE for Week 1 validation

risk:
  profit_filter_enabled: true  # ✅ Filter enabled
  min_expected_profit: 15.0    # ✅ $15 threshold
  profit_filter_spread_pips: 0.5
  use_dynamic_slippage: true   # ✅ Dynamic slippage enabled
  risk_per_trade: 0.03         # 3% risk
```

**Check ACC2 config** (configs/live_acc2.yaml):
```yaml
execution:
  mode: live  # Real orders on demo account

risk:
  profit_filter_enabled: true
  min_expected_profit: 15.0
  use_dynamic_slippage: true
  risk_per_trade: 0.03
```

---

### **Step 3: Stop Old Containers/Processes**

```powershell
# If using Docker
docker ps
docker stop live-acc1 live-acc2 2>$null

# If using native processes
Get-Process | Where-Object { $_.ProcessName -match "python" -and $_.CommandLine -match "orchestrator" }
# Kill PIDs if found: Stop-Process -Id <PID> -Force
```

---

### **Step 4: Start Paper Trading (ACC1)**

**Option A: Docker (recommended):**
```powershell
docker compose up live-acc1 -d
docker logs -f live-acc1
```

**Option B: Native Python:**
```powershell
cd "<REPO_PATH>"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD"

# Start ACC1 paper mode
python -m xauusd_ai.orchestrator configs/live_acc1.yaml 2>&1 | Tee-Object -FilePath outputs\paper_mode_week1.log
```

---

### **Step 5: Start Live Demo Trading (ACC2)**

**Docker:**
```powershell
docker compose up live-acc2 -d
docker logs -f live-acc2
```

**Native:**
```powershell
# In a NEW terminal
cd "<REPO_PATH>"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD"

python -m xauusd_ai.orchestrator configs/live_acc2.yaml 2>&1 | Tee-Object -FilePath outputs\live_acc2.log
```

---

### **Step 6: Verify Deployment**

**Check ACC1 Paper Mode:**
```powershell
# Check last 50 lines of log
Get-Content outputs\paper_mode_week1.log -Tail 50

# Look for:
# ✅ "Mode: PAPER"
# ✅ "Profit filter enabled: $15.00"
# ✅ "Dynamic slippage: True"
# ✅ "Connected to MT5 bridge port 5600"
```

**Check ACC2 Live Mode:**
```powershell
Get-Content outputs\live_acc2.log -Tail 50

# Look for:
# ✅ "Mode: LIVE"
# ✅ "Profit filter enabled: $15.00"
# ✅ "Connected to MT5 bridge port 5601"
```

---

## 📊 Week 1 Monitoring (May 6-13, 2026)

### **Daily Checks (Every 24h)**

**1. Check Skip Rate:**
```powershell
# Count total signals vs skipped
Get-Content outputs\paper_mode_week1.log | Select-String "SKIP.*profit_too_low" | Measure-Object
# Target: 60-82% skip rate
```

**2. Check Real RR Achieved:**
```powershell
# Check paper trade results
python -c "import pandas as pd; df = pd.read_csv('outputs/paper_mode_trades.csv'); print(f'Avg RR: {df[\"realized_rr\"].mean():.2f}')"
# Target: RR > 2.5 avg (vs WF assumption 3.67)
```

**3. Check Slippage Observed:**
```powershell
# Grep paper logs for slippage
Get-Content outputs\paper_mode_week1.log | Select-String "slippage.*pips" | Select -First 20
# Expected: 1-2 pips normal, 3-5 pips Asian session
```

**4. Check Balance Trend:**
```powershell
# Check latest balance
Get-Content outputs\paper_mode_week1.log | Select-String "Balance:" | Select -Last 10
# Target: Positive trend, no explosive gains (realistic)
```

---

### **Decision Points (May 13)**

**✅ PASS Criteria (Deploy to Live):**
- Skip rate: 60-82% ✅
- Real RR achieved: > 2.5 avg ✅
- Slippage observed: 1-5 pips (matches dynamic model) ✅
- P&L trend: Positive or flat (no major losses) ✅
- No critical errors in logs ✅

**⚠️ FAIL Criteria (Pause & Retrain):**
- Skip rate < 50% or > 90% (filter miscalibrated)
- Real RR < 2.0 avg (WF overestimates)
- Major losses (> -$100 in Week 1)
- Frequent errors/crashes

---

## 🎯 Post-Validation Actions

### **If Validation PASSES (May 13):**

1. **Switch ACC1 to Live Mode:**
   ```yaml
   # configs/live_acc1.yaml
   execution:
     mode: live  # Change from paper to live
   ```

2. **Restart ACC1:**
   ```powershell
   docker restart live-acc1
   # Or restart native process
   ```

3. **Monitor Daily:**
   - Real P&L vs WF projections
   - Monthly withdrawals (reset to $200 each month)
   - Risk metrics (DD, win rate, RR)

---

### **If Validation FAILS:**

1. **Analyze gap:**
   ```bash
   python scripts/compare_paper_vs_wf.py  # TODO: create this script
   ```

2. **Identify issues:**
   - RR assumption wrong? (3.67 too high)
   - Slippage model inaccurate? (need tuning)
   - Profit filter threshold wrong? ($15 too low/high)

3. **Retrain model:**
   - Use paper trades as training data
   - Adjust feature weights
   - Re-run WF validation
   - Repeat Week 1 validation

---

## 📝 Key Files & Logs

| File | Purpose |
|------|---------|
| `outputs/paper_mode_week1.log` | ACC1 paper mode full log |
| `outputs/live_acc2.log` | ACC2 live demo log |
| `outputs/paper_mode_trades.csv` | Paper trade results |
| `outputs/live_closed_trades_acc2.csv` | ACC2 live trade results |
| `configs/live_acc1.yaml` | ACC1 configuration |
| `configs/live_acc2.yaml` | ACC2 configuration |

---

## 🔧 Troubleshooting

**Problem:** ACC1 not skipping trades (filter not working)  
**Solution:** Check `profit_filter_enabled: true` in config, restart process

**Problem:** MT5 bridge connection failed  
**Solution:** Check MT5 running, check bridge port (5600/5601), restart bridge

**Problem:** Slippage too high (> 10 pips)  
**Solution:** Check `use_dynamic_slippage: true`, verify ATR calculation

**Problem:** Process crashes  
**Solution:** Check logs for errors, verify Python 3.11+, check dependencies

---

## 📞 Support & Next Steps

**Current Status:** ✅ Ready for deployment  
**Validation Period:** May 6-13, 2026 (7 days)  
**Decision Date:** May 13, 2026  
**Target:** Live deployment Week 2 if validation passes

**Documentation:**
- Full roadmap: [WF_IMPROVEMENTS_AND_ROADMAP.md](WF_IMPROVEMENTS_AND_ROADMAP.md)
- Deployment scripts: `deploy_windows.ps1`
- Production quickstart: `PRODUCTION_QUICKSTART.md`

---

**Last Updated:** May 6, 2026  
**Version:** 1.0  
**Status:** ✅ Ready for Paper Validation
