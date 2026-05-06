# WF Improvements & System Completion Roadmap

**Date:** May 6, 2026  
**Version:** 1.0  
**Status:** Implementation Complete — Validation Pending

---

## ✅ Completed Improvements

### 1. **Dynamic Slippage Model Integration**

**Problem:** WF simulation assumed perfect execution (no slippage), leading to overestimated returns.

**Solution:** Integrated ATR-based dynamic slippage model into backtesting engine.

**Implementation:**
- File: `src/xauusd_ai/backtesting/slippage.py` (already existed)
- Engine: `src/xauusd_ai/backtesting/engine.py` (already integrated)
- Config: `configs/acc1_v14pp_profit.yaml` — **`use_dynamic_slippage: true`** ✅

**Slippage Formula:**
```python
slippage = (base + atr_component + spread_component + volume_component) × session_multiplier

base = 1.0 pips (minimum)
atr_component = (atr/atr_mean - 1.0) × 0.5 × base  # Higher volatility = more slippage
spread_component = spread_pips × 0.3               # Wider spread = more slippage
volume_component = (1.0 - volume_ratio) × 0.5 × base  # Low volume = more slippage

Session multipliers:
- Asian: 1.5× (low liquidity)
- London: 1.0× (normal)
- NY: 0.9× (high liquidity)
```

**Expected Impact:**
- **Realistic slippage:** 1-2 pips normal, 3-6 pips Asian low-volume
- **Returns adjustment:** WF returns will be **5-15% lower** after slippage
- **Better live correlation:** Slippage-adjusted WF should match live execution better

**How to Enable/Disable:**
```yaml
risk:
  use_dynamic_slippage: true   # Enable (default: false)
  slippage_rr: 0.05            # Fallback static slippage (5% of 1R)
```

---

### 2. **Monthly Capital Reset (Withdrawal Simulation)**

**Problem:** WF assumed infinite compounding, but user withdraws profits monthly → balance resets to $200. WF didn't reflect this.

**Solution:** Added `--monthly-reset` flag to simulate monthly withdrawals.

**Implementation:**
- File: `scripts/walkforward_ict_wyckoff.py` (lines 112, 299-305, 675-695)
- Flag: `--monthly-reset`
- Logic: Reset balance to $200 every 30 days (tracks last reset date)

**How It Works:**
1. Track last reset date (`_monthly_reset_last_date`)
2. After each fold, check if 30 days passed (`days_since_reset >= 30`)
3. If yes: Reset balance to $200, log withdrawn amount
4. Continue trading with fresh $200

**Example Output:**
```
Fold 5: Test 2024-12-12 → 2025-01-15 | Balance: $200 → $1,500
💰 MONTHLY RESET: Withdrew $1,300, reset balance to $200 (after 34 days)
Fold 6: Test 2025-01-15 → 2025-02-15 | Balance: $200 → $800
```

**Command:**
```bash
python scripts/walkforward_ict_wyckoff.py configs/acc1_v14pp_profit.yaml \
  --test-start 2024-01-01 --test-bars 18000 --step-bars 18000 \
  --combo133 --cache --profit-filter --min-profit 15.0 \
  --monthly-reset  # ← NEW FLAG
```

**Expected Impact:**
- **Realistic compounding:** Matches user's actual withdrawal behavior
- **Lower total returns:** Each reset prevents exponential compounding
- **Better risk assessment:** Shows sustainable monthly returns, not explosive one-time gains

---

## 🎯 Next Steps Roadmap

### **Phase 1: Validation (Week 1 — May 6-13, 2026)**

#### **1.1 Run Improved WF Validation**
```bash
# Quick test (10 folds, 2024-present)
python scripts/walkforward_ict_wyckoff.py configs/acc1_v14pp_profit.yaml \
  --test-start 2024-01-01 --test-bars 18000 --step-bars 18000 \
  --combo133 --cache --profit-filter --min-profit 15.0 \
  --monthly-reset 2>&1 | tee outputs/wf_monthly_reset_test.log

# Expected results:
# - Slippage reduces avg returns by 10-20%
# - Monthly reset prevents explosive folds (no +54k%, +86k%)
# - More realistic: 500-3000% per month instead of 50k%
```

**Success Criteria:**
- ✅ Skip rate: 60-82% (profit filter working)
- ✅ Returns per month: 500-5000% (down from 50k% outliers)
- ✅ Slippage impact: 0.5-2 pips avg (realistic)
- ✅ Monthly reset working: Balance resets every 30 days

#### **1.2 Paper Trading Validation**
**Status:** ACC1 already running paper mode (PID 428)

**Monitor:**
- Real RR achieved vs WF estimated (target: RR > 3.0 avg)
- Skip rate matches WF (60-82%)
- Slippage observed in paper logs vs dynamic model

**Daily Check:**
```powershell
# Check paper logs
Get-Content "outputs\paper_mode_week1.err.log" -Tail 50
```

**Decision Point (May 13):**
- ✅ If paper RR ≥ 3.0 avg + skip rate 60-82% → **Trust WF, deploy to live**
- ⚠️ If paper RR < 2.5 avg → **WF overestimates, need correction**

---

### **Phase 2: System Completion (Week 2-3 — May 13-27, 2026)**

#### **2.1 Live Trading Deployment**
**Prerequisite:** Paper validation successful (Week 1)

**Deploy:**
```powershell
# Windows production
cd "<REPO_PATH>"
git fetch origin
git checkout main
git pull origin main
.\deploy_windows.ps1
```

**Monitoring:**
- Daily P&L comparison: Live vs WF projections
- Skip rate verification: 60-82% target
- Real RR tracking: Should match WF (3.0-3.67 avg)

#### **2.2 Model Improvement (Data-Driven)**

**After 2-4 weeks live data:**
1. **Analyze backtest-live gap:**
   ```bash
   python scripts/analyze_fold_outliers.py
   python scripts/compare_live_vs_wf.py  # TODO: create this
   ```

2. **Identify patterns:**
   - Which market conditions cause gap? (sideway, news, Asian session)
   - Which features underperform live?
   - Which RR assumptions were wrong?

3. **Retrain model:**
   - Add live losing trades to training data
   - Adjust feature weights based on live performance
   - Re-run WF with updated model

#### **2.3 Risk Management Tuning**

**Based on live experience:**
- Adjust risk% per trade (currently 3%)
- Tune profit filter threshold ($15 → $12 or $18?)
- Add session-specific risk multipliers (reduce Asian session)
- Implement dynamic position sizing based on realized RR

---

### **Phase 3: Automation & Scaling (Month 2+ — June 2026+)**

#### **3.1 Automated Retraining Pipeline**

**Goal:** Model learns from live trades weekly.

**Implementation:**
```python
# scripts/auto_retrain_weekly.py
# 1. Load live trades from past 7 days
# 2. Append to training data
# 3. Retrain model with updated data
# 4. Run WF validation
# 5. If WF metrics improve → deploy new model
# 6. If WF metrics degrade → keep old model, alert user
```

**Schedule:** Every Sunday 00:00 UTC (cron job)

#### **3.2 Advanced Monitoring Dashboard**

**Components:**
- **Real-time P&L:** Live balance, daily returns, monthly ROI
- **WF comparison:** Live vs WF projected returns (gap tracking)
- **Signal quality:** Skip rate, RR distribution, win rate
- **Risk metrics:** Current DD, max positions, circuit breaker status
- **Performance attribution:** Which patterns work best live?

**Tech Stack:**
- Backend: ChartWF (already exists, port 8800)
- Frontend: React dashboard (already exists)
- Alerts: Telegram bot for critical events

#### **3.3 Multi-Account Scaling**

**After 2-3 months profitable:**
- **ACC1:** $200 → $5,000 (monthly compounding)
- **ACC2:** $200 → $5,000
- **ACC3:** New account with refined model
- **Risk allocation:** 3% per account, max 9% total exposure

---

## 📊 Success Metrics (3-Month Goals)

| Metric | Target | Status |
|--------|--------|--------|
| **Monthly ROI** | 50-200% | 🔄 Validating |
| **Sharpe Ratio** | > 2.0 | ✅ WF: 4.37 |
| **Win Rate** | 40-50% | ✅ WF: 47% |
| **Skip Rate** | 60-82% | 🔄 Paper validation |
| **Backtest-Live Gap** | < 30% | ⚠️ TBD Week 1 |
| **Max Drawdown** | < 20% | ✅ WF: 13.9% |
| **Profit Factor** | > 2.0 | ✅ WF: 2.63 |
| **Real RR (avg)** | > 2.5 | 🔄 Paper validation |

---

## ⚠️ Critical Risks & Mitigation

### **Risk 1: WF Overestimation**
**Symptom:** Live returns 50% lower than WF projections  
**Mitigation:** 
- Paper validation (Week 1) catches this before live deployment
- Monthly reset reduces compounding bias
- Dynamic slippage adds realism

**Action:** If gap > 30% → pause live, retrain model with realistic slippage multiplier

---

### **Risk 2: Market Regime Change**
**Symptom:** Model trained on trending markets, live encounters sideways/choppy  
**Mitigation:**
- Regime detection (ADX gate, volatility filters)
- Sideway min confidence = 0.85 (strict filter)
- Circuit breaker: max DD 15%, daily loss 8%

**Action:** If 3 consecutive losing days → pause, analyze regime, adjust filters

---

### **Risk 3: Over-Optimization (Curve Fitting)**
**Symptom:** WF looks perfect, live fails immediately  
**Mitigation:**
- Profit filter prevents low-quality trades
- 10-fold WF validation (not just 1-2 folds)
- Paper mode validation before live

**Action:** If paper shows < 60% skip rate → model predictions unreliable, retrain

---

## 🎓 Lessons Learned

### **From Fold 5/7 Analysis:**
1. **Explosive returns (50k-80k%) are suspicious** → likely simulation artifacts
2. **Within-fold compounding is powerful** → but unrealistic with monthly withdrawals
3. **RR=3.67 assumption may not hold live** → slippage, execution quality vary
4. **Multi-position risks** → concurrent trades amplify wins AND losses

### **From WF Estimation Fix:**
1. **Data-driven > Assumptions** → RR 3.67 (from winners) vs 1.5 (guessed)
2. **Dynamic risk > Fixed pips** → balance-based risk matches real execution
3. **Estimation quality matters** → 7% skip → 82% skip (11× improvement)

---

## 📝 Next Immediate Actions

**For User:**
1. ✅ **Pull latest code** from main branch (includes monthly reset + slippage)
2. 🔄 **Run improved WF validation** (10 folds, with monthly reset)
3. 🔄 **Monitor paper trading** (ACC1) for 7 days
4. 📊 **Compare paper vs WF** results on May 13
5. 🚀 **Deploy to live** if validation successful

**Command to run:**
```bash
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"
source .venv/bin/activate
python scripts/walkforward_ict_wyckoff.py configs/acc1_v14pp_profit.yaml \
  --test-start 2024-01-01 --test-bars 18000 --step-bars 18000 \
  --combo133 --cache --profit-filter --min-profit 15.0 \
  --monthly-reset --no-rr-sweep \
  2>&1 | tee outputs/wf_monthly_reset_slippage_test.log
```

**Expected runtime:** ~3-5 minutes (10 folds, cached dataset)

---

## 📚 Additional Resources

- **Slippage Model Details:** `src/xauusd_ai/backtesting/slippage.py`
- **WF Script:** `scripts/walkforward_ict_wyckoff.py`
- **Fold Analysis:** `scripts/analyze_fold_outliers.py`
- **Production Deployment:** `PRODUCTION_DEPLOYMENT.md`
- **Week 1 Validation:** `PRODUCTION_QUICKSTART.md`

---

**Status:** ✅ **Ready for Validation**  
**Next Review:** May 13, 2026 (after Week 1 paper validation)
