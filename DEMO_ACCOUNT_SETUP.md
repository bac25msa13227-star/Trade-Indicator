# Demo Account Setup Guide

**Purpose:** Test profit filter với real order execution (no risk - demo money)  
**Duration:** 3-7 days (parallel với paper mode hoặc after)  
**Risk:** 🟢 ZERO (demo account = fake money)

---

## 📋 **Prerequisites**

### **1. Có Demo Account từ Broker**

**Cách tạo demo account (ví dụ với MetaTrader 5):**

```
1. Mở MT5 Terminal
2. File → Open an Account → Demo Account
3. Fill form:
   - Name: Your Name
   - Email: your@email.com
   - Phone: +84...
   - Leverage: 1:100 hoặc 1:500
   - Deposit: $10,000 USD (demo money)
4. Server: Chọn demo server của broker (e.g., "MetaQuotes-Demo")
5. Submit → Nhận login/password/server
```

**Save credentials:**
```
Demo Login:    12345678
Demo Password: abc123xyz
Demo Server:   MetaQuotes-Demo
```

### **2. Verify Demo Account Hoạt Động**

```bash
# Test login với MT5 Bridge
# (On production server where Docker is running)

# Check if demo account can connect
docker exec live-acc1 python -c "
import MetaTrader5 as mt5
mt5.initialize()
login = mt5.login(12345678, password='abc123xyz', server='MetaQuotes-Demo')
if login:
    print('✅ Demo account connected!')
    info = mt5.account_info()
    print(f'Balance: \${info.balance}')
    print(f'Equity: \${info.equity}')
else:
    print('❌ Failed to connect demo account')
    print(mt5.last_error())
mt5.shutdown()
"
```

---

## 🚀 **Setup Steps**

### **Step 1: Create Demo Config**

```bash
# SSH to production server
ssh your-production-server

# Navigate to repo
cd ~/Trade-Indicator

# Copy live config as template
cp configs/live_acc1.yaml configs/demo_acc1.yaml
```

### **Step 2: Edit Demo Config**

**Edit `configs/demo_acc1.yaml`:**

```yaml
# ============================================================================
# DEMO ACCOUNT CONFIG (Profit Filter Testing)
# ============================================================================

execution:
  mode: live              # ✅ CHANGE: paper → live (for real orders on demo)
  auto_trade: true        # ✅ Enable auto execution
  mt5_login: 12345678     # ✅ CHANGE: Your demo login
  mt5_password: "abc123xyz"  # ✅ CHANGE: Your demo password
  mt5_server: "MetaQuotes-Demo"  # ✅ CHANGE: Your demo server
  symbol: "XAUUSD"
  timeframe: "M15"
  check_interval: 60

risk:
  risk_pct: 0.03            # 3% risk per trade
  min_rr: 1.2
  max_holding_bars: 960
  
  # ✅ KEEP FILTER ENABLED
  profit_filter_enabled: true
  min_expected_profit: 12.0      # $12 threshold
  profit_filter_spread_pips: 0.5

strategy:
  force_trade: false
  trend_filter_enabled: false
  min_strat_score: 0.0

model:
  model_path: "outputs/acc1_combo133_202604_model.pkl"
  scaler_path: "outputs/acc1_combo133_202604_scaler.pkl"
  meta_path: "outputs/acc1_combo133_202604_meta.json"
  min_confidence: 0.70

# ... rest of config same as live_acc1.yaml ...
```

**Key changes:**
- ✅ `mode: paper` → `mode: live`
- ✅ `mt5_login`, `mt5_password`, `mt5_server` → Your demo credentials
- ✅ Keep `profit_filter_enabled: true`

### **Step 3: Add Demo Service to docker-compose.yml**

**Edit `docker-compose.yml`:**

```yaml
services:
  # ... existing services ...

  # ============================================================================
  # DEMO ACCOUNT (Profit Filter Testing)
  # ============================================================================
  demo-acc1:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: demo-acc1
    volumes:
      - ./configs/demo_acc1.yaml:/app/configs/live_acc1.yaml:ro
      - ./outputs:/app/outputs
      - ./data:/app/data:ro
    environment:
      - CONFIG_FILE=configs/live_acc1.yaml
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    networks:
      - trading-network
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    depends_on:
      - db
      - mlflow
```

### **Step 4: Start Demo Container**

```bash
# Start demo bot
docker compose up demo-acc1 -d --no-build

# Check status
docker ps | grep demo-acc1

# Expected output:
# demo-acc1    Up 5 seconds    ...
```

### **Step 5: Verify Deployment**

```bash
# Check logs
docker logs demo-acc1 | tail -50

# Should see:
# ✅ MT5 connected successfully
# ✅ Profit filter initialized: enabled=True, min_profit=$12.00
# ✅ Model loaded: acc1_combo133_202604_model.pkl
# ✅ Orchestrator started
```

**Expected logs:**
```
INFO: MT5 login successful: 12345678 on MetaQuotes-Demo
INFO: Account balance: $10000.00
INFO: Profit filter initialized: enabled=True, min_profit=$12.00
INFO: Model loaded with 67 features
INFO: Orchestrator loop started - checking every 60s
```

---

## 📊 **Monitoring Demo Account**

### **Daily Checklist (5 min/day)**

```bash
# 1. Check if bot is running
docker ps | grep demo-acc1
# Should be "Up X hours"

# 2. Check skip rate
TOTAL=$(docker logs demo-acc1 | grep "should_trade" | wc -l | tr -d ' ')
SKIPPED=$(docker logs demo-acc1 | grep "PROFIT_FILTER BLOCKED" | wc -l | tr -d ' ')
if [ "$TOTAL" -gt 0 ]; then
    echo "Total signals: $TOTAL"
    echo "Skipped: $SKIPPED ($((SKIPPED * 100 / TOTAL))%)"
fi

# Expected: 60-75% skip rate

# 3. Check P&L
docker exec demo-acc1 cat /app/outputs/live_closed_trades_acc1.csv | \
    awk -F',' 'NR>1 {sum+=$10} END {print "Total P&L: $"sum}'

# Expected: Positive trend

# 4. Check recent trades
docker exec demo-acc1 cat /app/outputs/live_closed_trades_acc1.csv | tail -10

# 5. Watch real-time logs (optional)
docker logs -f demo-acc1 | grep -E "PROFIT_FILTER|Order"
```

### **Watch Real-Time (First 30 Minutes)**

```bash
# Monitor first signals
docker logs -f demo-acc1 | grep -E "PROFIT_FILTER|Order|should_trade"

# Expected output:
# INFO: Signal detected: BUY @ 2345.50, conf=0.75
# INFO: Predicted profit: $15.20 > min $12.00
# INFO: Signal PASSED filter, should_trade=True
# INFO: Order placed: BUY 0.06 lot @ 2345.50, SL=2335.50, TP=2365.50
# INFO: Order ID: 123456789

# Or:
# INFO: Signal detected: SELL @ 2340.20, conf=0.68
# INFO: Predicted profit: $8.50 < min $12.00
# INFO: PROFIT_FILTER BLOCKED: profit_too_low
```

---

## 🎯 **Success Criteria (Day 7)**

### **✅ Deploy to LIVE If:**

| Metric | Target | Check |
|--------|--------|-------|
| **Skip rate** | 60-75% | `docker logs demo-acc1 \| grep "PROFIT_FILTER BLOCKED" \| wc -l` |
| **Total P&L** | > $0 | Check `live_closed_trades_acc1.csv` |
| **Win rate** | > 35% | Count wins/losses in CSV |
| **Prediction accuracy** | > 60% | Kept trades mostly win, skipped trades mostly lose |
| **No crashes** | 0 crashes in 7 days | `docker logs demo-acc1 \| grep -i error` |
| **No bias** | Balanced BUY/SELL filtering | Count BUY vs SELL skipped |

### **⚠️ Adjust Threshold If:**

- Skip rate 40-60% → Test $10 or $15
- P&L breakeven → Need 3 more days
- Win rate 30-35% → Marginal, needs review

### **❌ Disable Filter If:**

- Skip rate < 30% → Filter broken
- P&L < -$50 → Filter harmful
- Systematic bias → Skipping winners

---

## 🔧 **Troubleshooting**

### **Issue 1: Demo bot không connect được MT5**

**Symptoms:**
```
ERROR: Failed to connect MT5: unauthorized
```

**Solutions:**
1. Check credentials in `configs/demo_acc1.yaml`
2. Verify demo account still active (some brokers expire after 30 days)
3. Check server name (must match exactly, e.g., "MetaQuotes-Demo")
4. Try login manually in MT5 Terminal first

### **Issue 2: Orders không được placed**

**Symptoms:**
```
INFO: should_trade=True
INFO: Building order plan...
(No order logs)
```

**Solutions:**
1. Check `execution.mode: live` (not `paper`)
2. Check `execution.auto_trade: true`
3. Check MT5 connection: `docker logs demo-acc1 | grep "MT5"`
4. Check risk limits: Balance > minimum order size

### **Issue 3: Skip rate quá thấp (<30%)**

**Symptoms:**
```
Total signals: 100
Skipped: 20 (20%)
```

**Solutions:**
1. Check profit estimation in logs
2. Increase threshold to $15 or $18
3. Check if using correct config (demo_acc1.yaml)
4. Review `_estimate_profit()` logic in orchestrator

### **Issue 4: Container keeps crashing**

**Symptoms:**
```
docker ps | grep demo-acc1
(No output - container stopped)
```

**Solutions:**
```bash
# Check crash logs
docker logs demo-acc1 | tail -100

# Common issues:
# - MT5 license error → Check broker allows API trading
# - Config syntax error → Validate YAML
# - Missing files → Check model/scaler paths

# Restart with logs
docker compose up demo-acc1 --no-build
```

---

## 📈 **Comparison: Paper vs Demo**

| Aspect | Paper Mode | Demo Account |
|--------|------------|--------------|
| **Risk** | 🟢 Zero | 🟢 Zero |
| **Order execution** | ❌ Shadow only | ✅ Real orders (demo) |
| **Spread/slippage** | ❌ Simulated | ✅ Real from broker |
| **Broker response** | ❌ Not tested | ✅ Tested |
| **Network latency** | ❌ Not tested | ✅ Tested |
| **Partial fills** | ❌ Not tested | ✅ Can happen |
| **Confidence level** | Medium | High |
| **When to use** | Initial validation | Pre-live final test |

**Recommendation:** 
- Paper mode (7 days) → Demo (3-7 days) → Live
- OR: Paper mode (7 days) → Live (if confident)

---

## 📝 **Demo Trade Log Template**

Create: `outputs/demo_profit_filter_log.md`

```markdown
# Demo Account Testing - Profit Filter

**Start Date:** May 6, 2026  
**Demo Account:** 12345678  
**Initial Balance:** $10,000  

## Daily Log

### Day 1 (May 6, 2026)
- **Signals total:** X
- **Signals skipped:** Y (Z%)
- **Signals executed:** W
- **Orders placed:** N
- **P&L today:** $XXX
- **Balance:** $XXX
- **Notes:** 

### Day 2 (May 7, 2026)
...

## Summary (Day 7)
- **Total signals:** 
- **Average skip rate:** 
- **Total trades:** 
- **Win rate:** 
- **Total P&L:** 
- **Final balance:** 
- **Decision:** [Deploy to live / Adjust threshold / Disable]
```

---

## 🚀 **Quick Start (Copy-Paste Commands)**

```bash
# ============================================================================
# DEMO ACCOUNT QUICK SETUP
# ============================================================================

# 1. Create demo config
cp configs/live_acc1.yaml configs/demo_acc1.yaml

# 2. Edit demo config (use your favorite editor)
vim configs/demo_acc1.yaml
# - Change mode: paper → live
# - Update mt5_login, mt5_password, mt5_server
# - Keep profit_filter_enabled: true

# 3. Add demo service to docker-compose.yml
vim docker-compose.yml
# - Add demo-acc1 service (see Step 3 above)

# 4. Start demo bot
docker compose up demo-acc1 -d --no-build

# 5. Verify deployment
docker logs demo-acc1 | tail -50

# 6. Monitor first signals (60 seconds)
timeout 60 docker logs -f demo-acc1 | grep -E "PROFIT_FILTER|Order"

# 7. Daily monitoring (run daily)
TOTAL=$(docker logs demo-acc1 | grep "should_trade" | wc -l | tr -d ' ')
SKIPPED=$(docker logs demo-acc1 | grep "PROFIT_FILTER BLOCKED" | wc -l | tr -d ' ')
if [ "$TOTAL" -gt 0 ]; then
    echo "Skip rate: $((SKIPPED * 100 / TOTAL))%"
fi

docker exec demo-acc1 cat /app/outputs/live_closed_trades_acc1.csv | \
    awk -F',' 'NR>1 {sum+=$10} END {print "Total P&L: $"sum}'

# 8. Stop demo bot (when done)
docker compose stop demo-acc1
```

---

## 🎯 **Decision Tree (Day 7)**

```
Demo Results After 7 Days
  │
  ├─ Skip rate 60-75% + P&L positive
  │  └─> ✅ Deploy to LIVE (high confidence)
  │
  ├─ Skip rate 60-75% + P&L breakeven
  │  └─> ⚠️  Test 3 more days OR deploy with caution
  │
  ├─ Skip rate 40-60% + P&L positive
  │  └─> ⚠️  Adjust threshold ($10/$15), test again
  │
  ├─ Skip rate 40-60% + P&L negative
  │  └─> ❌ Investigate profit estimation, don't deploy
  │
  └─ Skip rate <30%
     └─> ❌ Filter broken, fix estimation before deploy
```

---

**Last Updated:** May 6, 2026  
**Related Docs:**
- `PAPER_MODE_DEPLOYMENT.md` - Paper mode setup
- `PROFIT_FILTER_INTEGRATION.md` - Technical details
- `deploy_profit_filter_full.sh` - Automated deployment script
