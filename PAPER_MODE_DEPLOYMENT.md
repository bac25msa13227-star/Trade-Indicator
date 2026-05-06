# Profit Filter - Paper Mode Deployment Guide

**Status:** ✅ DEPLOYED (May 6, 2026)  
**Duration:** 7 days paper testing  
**Threshold:** $12.00 (optimal from simulation)

---

## ✅ **Deployment Steps Completed**

### 1. Config Updated
```yaml
# configs/live_acc1.yaml
risk:
  profit_filter_enabled: true        # ✅ ENABLED
  min_expected_profit: 12.0          # $12 optimal threshold
  profit_filter_spread_pips: 0.5     # XAUUSD spread
```

**Changes:**
- `profit_filter_enabled: false → true`
- `min_expected_profit: 15.0 → 12.0` (based on simulation results)

### 2. Code Integration
- `src/xauusd_ai/config.py`: Added profit filter config params
- `src/xauusd_ai/orchestrator.py`: Integrated filter logic
- Hot-reload support: Can toggle without restart

### 3. Git Commits
```bash
Commit: 7277abb - feat: integrate profit filter into orchestrator
Commit: 542d6a0 - feat: enable profit filter for 7-day paper testing
Branch: feature/turnover-profit-filter
```

---

## 🚀 **Server Deployment (Manual Steps Required)**

### Step 1: Pull Latest Code on Production Server
```bash
# SSH to production server
ssh your-server

# Navigate to repo
cd ~/Trade-Indicator  # Or your path

# Pull latest changes
git fetch origin
git checkout feature/turnover-profit-filter
git pull origin feature/turnover-profit-filter
```

### Step 2: Restart Live Bot Container
```bash
# Restart acc1 bot to load new config
docker compose restart live-acc1

# Verify container started
docker ps | grep live-acc1

# Check logs for filter confirmation
docker logs live-acc1 | grep "Profit filter"
```

**Expected log output:**
```
Profit filter initialized: enabled=True, min_profit=$12.00
```

### Step 3: Monitor First Hour
```bash
# Watch live logs
docker logs -f live-acc1 | grep -E "PROFIT_FILTER|should_trade"

# Check if filter is working
docker logs live-acc1 | grep "PROFIT_FILTER BLOCKED" | wc -l
```

**What to look for:**
- ✅ Filter blocks some trades: "PROFIT_FILTER BLOCKED: profit_too_low"
- ✅ Some trades still pass: "should_trade=True"
- ✅ Skip rate ~70-75% of signals

---

## 📊 **Monitoring Checklist (Daily for 7 Days)**

### Day 1 (May 6, 2026)
- [ ] Container restarted successfully
- [ ] Filter logs appear in output
- [ ] First signal filtered or passed
- [ ] No errors/crashes

### Days 2-6
- [ ] Skip rate stable (70-75%)
- [ ] Net P&L trending positive
- [ ] No systematic bias (not skipping all BUYs or all SELLs)
- [ ] Bot continues running (no crashes)

### Day 7 (May 13, 2026) - Decision Day
- [ ] Calculate final net P&L
- [ ] Calculate actual skip rate
- [ ] Compare predicted profit vs actual
- [ ] Decide: deploy to live or adjust threshold

---

## 📈 **Success Criteria (Day 7 Review)**

### ✅ **Deploy to Live If:**
1. **Net P&L > 0** (positive after 7 days)
2. **Skip rate 70-75%** (filter working as expected)
3. **No systematic bias** (balanced BUY/SELL filtering)
4. **Prediction accuracy > 60%** (correct filter decisions)
5. **Bot stable** (no crashes or errors)

### ⚠️ **Adjust Threshold If:**
- Skip rate too high (>80%) → Lower threshold to $10
- Skip rate too low (<60%) → Raise threshold to $15
- Net P&L negative but close to 0 → Need more days data

### ❌ **Disable Filter If:**
- Net P&L significantly negative (< -$50)
- Systematic bias detected (skipping only profitable trades)
- Bot crashes/errors related to filter

---

## 🔧 **Quick Toggle Commands**

### Disable Filter (Emergency)
```yaml
# Edit configs/live_acc1.yaml
risk:
  profit_filter_enabled: false  # Disable filter
```

```bash
# Restart to apply
docker compose restart live-acc1
```

### Change Threshold (On-the-Fly)
```yaml
# Edit configs/live_acc1.yaml
risk:
  min_expected_profit: 15.0  # Increase to $15
```

```bash
# Hot-reload (no restart needed)
# Config will reload automatically within 60s
# Or force restart: docker compose restart live-acc1
```

---

## 📊 **WF Validation Status**

### Full Validation (268 folds, 2023-2026)
**Status:** ⏳ RUNNING (started May 6, 2026)

**Command:**
```bash
python scripts/walkforward_ict_wyckoff.py configs/acc1_v14pp_profit.yaml \
  --test-start 2023-01-01 --test-bars 6000 --step-bars 6000 \
  --no-compound --combo133 --cache --risk-pct 0.030 --no-rr-sweep \
  --profit-filter --min-profit 12.0 \
  2>&1 | tee outputs/wf_with_profit_filter_full.log
```

**Monitor progress:**
```bash
tail -f outputs/wf_with_profit_filter_full.log
```

**ETA:** 2-3 hours (268 folds)

**Expected results:**
- Net P&L should be positive (vs -$25k baseline)
- Skip rate ~74%
- Confirm simulation results in real WF

---

## 📞 **Contact & Escalation**

**If issues arise:**
1. Check logs: `docker logs live-acc1`
2. Disable filter: Set `profit_filter_enabled: false`
3. Restart container: `docker compose restart live-acc1`
4. Review this guide: `PAPER_MODE_DEPLOYMENT.md`

**Critical issues (immediate disable):**
- Bot crashes with filter-related errors
- All trades skipped (100% skip rate)
- Systematic bias detected in first 24h

---

## 📝 **Paper Mode Log Template**

Create: `outputs/profit_filter_paper_log.md`

```markdown
# Profit Filter Paper Mode - Daily Log

## Day 1 (May 6, 2026)
- **Deployment time:** HH:MM UTC
- **Signals total:** X
- **Signals skipped:** Y (Z%)
- **Signals executed:** W
- **Net P&L:** $XXX
- **Notes:** 

## Day 2 (May 7, 2026)
...
```

---

## 🎯 **Next Steps After Day 7**

### If Successful (Net P&L > 0)
1. **Merge to main:**
   ```bash
   git checkout main
   git merge feature/turnover-profit-filter
   git push origin main
   ```

2. **Deploy to live:**
   ```yaml
   execution:
     mode: live  # Switch from paper to live
   ```

3. **Monitor for 2 weeks:**
   - Daily P&L tracking
   - Weekly threshold adjustment if needed

### If Needs Adjustment
1. **Analyze paper results**
2. **Adjust threshold** (try $10, $15, or $18)
3. **Run another 3-day paper test**
4. **Re-evaluate**

### If Failed (Net P&L < -$50)
1. **Disable filter:**
   ```yaml
   profit_filter_enabled: false
   ```

2. **Root cause analysis:**
   - Why predictions were wrong?
   - Systematic bias?
   - Need better profit estimator?

3. **Consider Option B:**
   - Train profit predictor model
   - Use historical performance data
   - More sophisticated estimation

---

## 📚 **Reference Documents**

- **Integration Guide:** `PROFIT_FILTER_INTEGRATION.md`
- **Test Results:** Simulation showed -$25k → +$98k with $12 threshold
- **Spread Explanation:** `SPREAD_EXPLAINED_VN.md`
- **Turnover Results:** `TURNOVER_VALIDATION_RESULTS.md`
- **Git Branch:** `feature/turnover-profit-filter`

---

**Last Updated:** May 6, 2026  
**Deployment Phase:** Paper Mode (7 days)  
**Next Review:** May 13, 2026
