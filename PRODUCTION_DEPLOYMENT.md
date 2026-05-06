# 🚀 Production Deployment Guide — Dual-Mode Trading

## 📊 Deployment Configuration

### **ACC1 — Paper Mode (Shadow Execution)**
- **Purpose:** Validation without real orders
- **Mode:** `paper` (logs signals, no MT5 execution)
- **Config:** `configs/live_acc1.yaml`
- **Container:** `live-acc1`
- **MT5 Bridge:** Port 5600 (http://host.docker.internal:5600)
- **Profit Filter:** $15 threshold (82.3% skip rate)
- **Output:** `outputs/paper_trade_signals_acc1.csv`

### **ACC2 — Live Demo (Real Trading)**
- **Purpose:** Real trading on demo account
- **Mode:** `live` (executes orders on MT5)
- **Config:** `configs/live_acc2.yaml`
- **Container:** `live`
- **MT5 Bridge:** Port 5601 (http://host.docker.internal:5601)
- **Profit Filter:** $15 threshold (82.3% skip rate)
- **Output:** `outputs/live_closed_trades_acc2.csv`

---

## 🎯 Quick Start (Copy-Paste Commands)

### **Step 1: Start Docker Desktop**
```bash
# macOS
open -a Docker

# Wait 30 seconds for Docker to start
sleep 30
```

### **Step 2: Deploy Both Accounts**
```bash
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"

# Run deployment script
./deploy_dual_mode.sh
```

**What it does:**
1. ✅ Checks Docker running
2. 🛑 Stops old containers
3. 🏗️  Builds images (if needed)
4. 🔧 Starts infrastructure (Postgres, MLflow, Prometheus)
5. 📝 Starts ACC1 (paper mode)
6. 💰 Starts ACC2 (live demo)
7. 📊 Shows status + logs

### **Step 3: Monitor Performance**
```bash
# Run monitoring script (safe to run anytime)
./monitor_dual_mode.sh
```

**Output:**
```
📊 ACC1 (Paper Mode) — Shadow Execution
────────────────────────────────────────────────────────────────────────────
  Status: ✅ Running
  Total signals: 150
  Skipped: 123 (82%)
  Kept: 27
  Skip rate status: ✅ Within target (60-82%)
  Paper trades logged: 27

💰 ACC2 (Live Demo) — Real Trading on Demo Account
────────────────────────────────────────────────────────────────────────────
  Status: ✅ Running
  Total signals: 148
  Skipped: 121 (82%)
  Kept: 27
  Skip rate status: ✅ Within target (60-82%)
  Closed trades: 12
  Total P&L: $145.30
  Win rate: 58% (7/12)
```

---

## 📋 Daily Monitoring Workflow

### **Option A: Automated (Recommended)**

**Create cron job (runs every 6 hours):**
```bash
# Edit crontab
crontab -e

# Add this line:
0 */6 * * * cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator" && ./monitor_dual_mode.sh >> outputs/monitor_log.txt 2>&1
```

### **Option B: Manual Check (30 seconds/day)**

```bash
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"
./monitor_dual_mode.sh
```

**Key metrics to check:**
- ✅ **Skip rate:** 60-82% (both accounts)
- ✅ **ACC2 P&L:** Positive trend
- ✅ **Win rate:** 40%+ (expected 41% from WF)
- ⚠️  **Large difference:** <5pp between ACC1/ACC2 skip rates

---

## 🔍 Detailed Commands

### **View Live Logs**
```bash
# ACC1 (paper) — last 50 lines
docker logs live-acc1 --tail 50

# ACC2 (demo) — last 50 lines
docker logs live --tail 50

# Follow logs (real-time)
docker logs -f live-acc1  # Press Ctrl+C to exit
docker logs -f live
```

### **Restart Specific Account**
```bash
# Restart ACC1 (paper)
docker compose restart live-acc1

# Restart ACC2 (demo)
docker compose restart live

# Restart both
docker compose restart live-acc1 live
```

### **Stop Accounts**
```bash
# Stop both
docker compose down live live-acc1

# Stop specific
docker compose stop live-acc1  # ACC1
docker compose stop live       # ACC2
```

### **Check Container Status**
```bash
docker ps --filter "name=live" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### **Check Configuration**
```bash
# Verify ACC1 is paper mode
grep "mode:" configs/live_acc1.yaml
# Expected: mode: paper

# Verify ACC2 is live mode
grep "mode:" configs/live_acc2.yaml
# Expected: mode: live

# Check profit filter enabled
grep "profit_filter_enabled" configs/live_acc1.yaml configs/live_acc2.yaml
# Expected: true for both
```

---

## 📊 Performance Analysis

### **Daily P&L Check (ACC2 Demo)**
```bash
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"

# Calculate total P&L
tail -n +2 outputs/live_closed_trades_acc2.csv | \
  awk -F',' '{sum+=$8} END {printf "Total P&L: $%.2f\n", sum}'

# Calculate win rate
WIN=$(tail -n +2 outputs/live_closed_trades_acc2.csv | awk -F',' '$8 > 0 {count++} END {print count}')
TOTAL=$(tail -n +2 outputs/live_closed_trades_acc2.csv | wc -l)
echo "Win rate: $((WIN * 100 / TOTAL))% ($WIN/$TOTAL)"

# Show recent trades
tail -10 outputs/live_closed_trades_acc2.csv | column -t -s','
```

### **Skip Rate Trend (Both Accounts)**
```bash
# ACC1 skip rate
TOTAL_ACC1=$(docker logs live-acc1 | grep -c "should_trade")
SKIPPED_ACC1=$(docker logs live-acc1 | grep -c "PROFIT_FILTER BLOCKED")
echo "ACC1 skip: $((SKIPPED_ACC1 * 100 / TOTAL_ACC1))%"

# ACC2 skip rate
TOTAL_ACC2=$(docker logs live | grep -c "should_trade")
SKIPPED_ACC2=$(docker logs live | grep -c "PROFIT_FILTER BLOCKED")
echo "ACC2 skip: $((SKIPPED_ACC2 * 100 / TOTAL_ACC2))%"
```

### **Export Logs for Analysis**
```bash
# Export last 1000 lines (ACC1 paper)
docker logs live-acc1 --tail 1000 > outputs/acc1_logs_$(date +%Y%m%d).txt

# Export last 1000 lines (ACC2 demo)
docker logs live --tail 1000 > outputs/acc2_logs_$(date +%Y%m%d).txt
```

---

## 🔧 Troubleshooting

### **Problem: Skip rate too low (<60%)**

**Diagnosis:**
```bash
# Check current threshold
grep "min_expected_profit" configs/live_acc1.yaml configs/live_acc2.yaml
```

**Solution:**
1. Increase threshold to $18-20:
```bash
vim configs/live_acc1.yaml  # Change min_expected_profit: 18.0
vim configs/live_acc2.yaml  # Change min_expected_profit: 18.0
```

2. Restart containers:
```bash
docker compose restart live-acc1 live
```

3. Monitor for 24h to verify new skip rate.

### **Problem: Skip rate too high (>85%)**

**Impact:** Missing profitable trades.

**Solution:**
1. Lower threshold to $12:
```bash
vim configs/live_acc1.yaml  # Change min_expected_profit: 12.0
vim configs/live_acc2.yaml  # Change min_expected_profit: 12.0
```

2. Restart containers:
```bash
docker compose restart live-acc1 live
```

### **Problem: Large difference between ACC1/ACC2 skip rates (>5pp)**

**Diagnosis:**
```bash
# Compare configs
diff configs/live_acc1.yaml configs/live_acc2.yaml | grep "profit_filter"
```

**Solution:**
- Ensure both configs have identical profit filter settings
- Restart both containers
- Check if MT5 bridge latency differs (port 5600 vs 5601)

### **Problem: Container keeps restarting**

**Check logs:**
```bash
docker logs live-acc1 --tail 100
docker logs live --tail 100
```

**Common issues:**
- ❌ MT5 bridge not running (check ports 5600, 5601)
- ❌ Model file missing (`outputs/acc1_combo133_202604_model.pkl`)
- ❌ Database connection failed (check postgres container)

**Fix:**
```bash
# Restart infrastructure
docker compose restart postgres mlflow

# Wait 10 seconds
sleep 10

# Restart trading containers
docker compose restart live-acc1 live
```

### **Problem: No signals generated**

**Check:**
```bash
# Verify market hours (XAUUSD 22:00 Sun - 21:00 Fri GMT)
date -u

# Check if blocked hours active
docker logs live-acc1 | grep "BLOCKED HOUR"

# Verify MT5 bridge responsive
curl http://localhost:5600/health  # ACC1
curl http://localhost:5601/health  # ACC2
```

---

## 📈 Success Metrics (Week 1 Validation)

**Target Performance (from WF validation):**
- ✅ Skip rate: 60-82% (both accounts)
- ✅ Win rate: 40%+ (expected 41%)
- ✅ ACC2 P&L: Positive after 50+ trades
- ✅ Skip rate consistency: <5pp difference ACC1/ACC2
- ✅ Container uptime: >99% (max 1 restart/day)

**Decision Framework (Day 7):**

| Condition | Action |
|-----------|--------|
| Skip 60-82% + P&L positive | ✅ Deploy to LIVE |
| Skip 40-60% | ⚠️  Adjust threshold $18-20 |
| Skip <40% OR P&L negative | ❌ Disable filter, investigate |
| ACC1/ACC2 diff >10pp | 🔍 Check config sync |

---

## 🎯 Production Prompt (Copy-Paste Ready)

```bash
#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# XAUUSD AI — Production Dual-Mode Deployment
# Paper mode (ACC1) + Live demo (ACC2) with $15 profit filter
# ═══════════════════════════════════════════════════════════════════════════

cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"

# 1. Start Docker Desktop (if not running)
open -a Docker && sleep 30

# 2. Deploy both accounts
./deploy_dual_mode.sh

# 3. Wait 60 seconds for startup
sleep 60

# 4. Verify deployment
./monitor_dual_mode.sh

# 5. Check logs
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📋 Recent Logs (ACC1 Paper):"
echo "════════════════════════════════════════════════════════════════════════════"
docker logs live-acc1 --tail 20 | grep -E "(Starting|Profit filter|BLOCKED|signal)"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📋 Recent Logs (ACC2 Demo):"
echo "════════════════════════════════════════════════════════════════════════════"
docker logs live --tail 20 | grep -E "(Starting|Profit filter|BLOCKED|signal|order)"

echo ""
echo "✅ Deployment complete! Monitor with: ./monitor_dual_mode.sh"
```

---

## 📝 Daily Checklist (30 seconds)

**Morning routine (9 AM local time):**
```bash
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"

# 1. Quick status check
./monitor_dual_mode.sh

# 2. Check ACC2 P&L
tail -n +2 outputs/live_closed_trades_acc2.csv | \
  awk -F',' '{sum+=$8} END {printf "ACC2 P&L: $%.2f\n", sum}'

# 3. Verify containers running
docker ps | grep "live"
```

**Expected time:** 15-30 seconds  
**Frequency:** Once daily (weekdays)  
**Red flags:** Skip rate <40%, containers stopped, P&L declining 3+ days

---

## 🔐 Security Notes

**Credentials:**
- ❌ Never commit `.env` file
- ✅ Use environment variables for MT5 credentials
- ✅ Demo account only (no real money)

**Monitoring:**
- ✅ Paper mode (ACC1) has NO access to real trading
- ✅ Demo account (ACC2) isolated from live funds
- ✅ Profit filter prevents >82% of trades (cost protection)

**Before going live:**
1. ✅ Complete 7-day paper validation
2. ✅ Verify skip rate 60-82% stable
3. ✅ ACC2 demo P&L positive
4. ✅ Review all BLOCKED trades manually
5. ✅ Update main branch with PR approval

---

**Last Updated:** 2026-05-06  
**Version:** 1.0.0 (Profit Filter with WF Estimation Fix)  
**Status:** ✅ Ready for production deployment
