# 🚀 QUICK START - Deploy Profit Filter (Production)

**Copy-paste commands này vào production server để deploy ngay!**

---

## ⚡ **DEPLOY NGAY (3 PHÚT)**

### **Bước 1: Pull Code Mới**

```bash
cd ~/Trade-Indicator
git fetch origin
git checkout feature/turnover-profit-filter
git pull origin feature/turnover-profit-filter
```

### **Bước 2: Verify Config**

```bash
# Check filter đã enabled chưa
cat configs/live_acc1.yaml | grep -A3 "profit_filter"
```

**Expected output:**
```yaml
profit_filter_enabled: true
min_expected_profit: 12.0
profit_filter_spread_pips: 0.5
```

**Nếu thấy `false`:** Edit `configs/live_acc1.yaml` đổi thành `true`

### **Bước 3: Restart Bot**

```bash
docker compose restart live-acc1
sleep 5
```

### **Bước 4: Verify Deploy**

```bash
docker logs live-acc1 | grep "Profit filter" | tail -3
```

**Expected output:**
```
Profit filter initialized: enabled=True, min_profit=$12.00
```

✅ **DONE! Bot đã chạy với profit filter!**

---

## 📊 **CHECK DAILY (30 GIÂY/NGÀY)**

### **Check Skip Rate**

```bash
TOTAL=$(docker logs live-acc1 | grep "should_trade" | wc -l | tr -d ' ')
SKIPPED=$(docker logs live-acc1 | grep "PROFIT_FILTER BLOCKED" | wc -l | tr -d ' ')
if [ "$TOTAL" -gt 0 ]; then
    KEPT=$((TOTAL - SKIPPED))
    echo "Signals: $TOTAL | Skipped: $SKIPPED ($((SKIPPED * 100 / TOTAL))%) | Kept: $KEPT"
fi
```

**Target:** 60-75% skip rate

### **Watch Real-Time (Optional)**

```bash
docker logs -f live-acc1 | grep -E "PROFIT_FILTER|should_trade"
```

Press `Ctrl+C` để stop.

---

## 🎯 **AFTER 7 DAYS (May 13)**

### **If Skip Rate 60-75%:**
✅ **Deploy to live mode**

```bash
# Edit config
vim configs/live_acc1.yaml
# Change: mode: paper → mode: live

# Restart
docker compose restart live-acc1
```

### **If Skip Rate 40-60%:**
⚠️ **Adjust threshold**

```bash
# Edit config
vim configs/live_acc1.yaml
# Change: min_expected_profit: 12.0 → 15.0

# Restart
docker compose restart live-acc1
```

### **If Skip Rate <40%:**
❌ **Disable filter**

```bash
# Edit config
vim configs/live_acc1.yaml
# Change: profit_filter_enabled: true → false

# Restart
docker compose restart live-acc1
```

---

## 🔧 **EMERGENCY COMMANDS**

### **Disable Filter Immediately**
```bash
vim configs/live_acc1.yaml
# Set: profit_filter_enabled: false
docker compose restart live-acc1
```

### **Check Bot Status**
```bash
docker ps | grep live-acc1
```

### **Check Bot Logs**
```bash
docker logs live-acc1 | tail -100
```

### **Stop Bot**
```bash
docker compose stop live-acc1
```

### **Start Bot**
```bash
docker compose up live-acc1 -d
```

---

## 📝 **MONITORING TEMPLATE**

**Copy vào notes và track daily:**

```
DAY 1 (May 6): Signals: ___ | Skipped: ___ (___%)
DAY 2 (May 7): Signals: ___ | Skipped: ___ (___%)
DAY 3 (May 8): Signals: ___ | Skipped: ___ (___%)
DAY 4 (May 9): Signals: ___ | Skipped: ___ (___%)
DAY 5 (May 10): Signals: ___ | Skipped: ___ (___%)
DAY 6 (May 11): Signals: ___ | Skipped: ___ (___%)
DAY 7 (May 12): Signals: ___ | Skipped: ___ (___%)

DECISION (May 13): [Deploy / Adjust / Disable]
```

---

## ✅ **CHECKLIST**

**Day 1:**
- [ ] Pull code ✅
- [ ] Restart bot ✅
- [ ] See "Profit filter initialized" in logs ✅
- [ ] Monitor first 30 minutes

**Day 2-6:**
- [ ] Check skip rate daily
- [ ] Bot still running (no crashes)

**Day 7:**
- [ ] Calculate average skip rate
- [ ] Make decision (deploy/adjust/disable)

---

## 💡 **QUICK REFERENCE**

| Command | Purpose |
|---------|---------|
| `docker logs live-acc1 \| tail -50` | Check recent logs |
| `docker ps \| grep live-acc1` | Check if running |
| `docker compose restart live-acc1` | Restart bot |
| `docker compose stop live-acc1` | Stop bot |
| `docker compose up live-acc1 -d` | Start bot |

---

## 📞 **NEED HELP?**

**Read full docs:**
- `DEPLOYMENT_READY.md` - Complete overview
- `PAPER_MODE_DEPLOYMENT.md` - Detailed paper mode guide
- `deploy_profit_filter_full.sh` - Automated script (alternative)

**Or just run automated script:**
```bash
./deploy_profit_filter_full.sh
```

---

**🎯 TÓM TẮT: 4 lệnh để deploy, check daily 30 giây, quyết định Day 7!**
