# XAUUSD AI Dev - Core Focus

## Đã Ứng Dụng ✅

### Simulation & Testing:
1. **Dynamic Slippage** - ATR-based, session-aware (0.5-6 pips)
2. **Paper Trading** - Shadow execution, signal logging
3. **A/B Testing** - Statistical comparison framework

### Metrics:
4. **Sharpe/Calmar Ratio** - Risk-adjusted returns
5. **Turnover Analysis** - Spread cost tracking
6. **Profit Filter** - Skip unprofitable trades ($15 threshold)

---

## Chưa Có ❌ (Quan Trọng Để Improve)

### Model Nâng Cao:
1. **Regime Detection** - HMM/ADX để tránh sideway
2. **Ensemble** - LightGBM + LSTM + XGBoost
3. **RL Fine-tuning** - PPO/SAC optimize sizing & exit

### Simulation Chính Xác:
4. **Tick-by-tick** - Replay M1 data tick-level
5. **Partial Fill** - Simulate khớp lệnh một phần
6. **Latency** - 50-200ms delay injection

### Monitoring:
7. **Feature Stability** - Track importance drift per fold
8. **Stress Testing** - Crash 2020, news spike scenarios

---

## Priority Roadmap (Improve $200/Fold)

### **Phase 1: Regime Detection (Week 1-2) 🔥**
**Goal:** Skip sideway markets, improve win rate +5-10%

**Files created:**
- `src/xauusd_ai/features/regime_detection.py` ✅
- Integration: Add to `features/dataset.py`
- Testing: Run WF with regime filter

**Expected:**
- Win rate: 42.6% → 50%+
- Skip rate: +10-15% (avoid bad trades)
- Profit/fold: +$100-200 improvement

### **Phase 2: Feature Stability (Week 3)** 
**Goal:** Monitor feature drift, retrain when needed

**Files created:**
- `src/xauusd_ai/monitoring/feature_stability.py` ✅
- Integration: Add to WF script fold loop
- Output: `outputs/feature_stability.jsonl`

**Expected:**
- Detect drift early (before model degrades)
- Auto-retrain trigger when top 10 features drift >30%

### **Phase 3: Better Entry/Exit (Week 4)**
**Goal:** Improve RR from 3.67 → 4.5+

**Changes needed:**
- Entry: Pullback confirmation, volume surge
- Exit: Trailing SL (activate at 1.5R, trail 0.8R)
- Target RR: +0.5-1.0 improvement

**Expected:**
- Avg RR: 3.67 → 4.5 (+23%)
- Profit/fold: +$150-300

---

## Current Baseline (No-Compound, $200/Fold)

**Running now:** 3-fold WF validation with profit filter

**Expected metrics:**
- Win rate: ~43%
- Profit factor: ~2.7
- Return/fold: +300-500% ($600-$1,000 profit)
- Max DD: -13%

**After improvements:**
- Win rate: ~50% (+7%)
- Profit factor: ~3.5 (+30%)
- Return/fold: +500-800% ($1,000-$1,600 profit)
- Max DD: <12% (better)

---

## WF vs Live Gap Analysis

### **Causes of Gap:**
1. **Slippage** - WF uses estimate, live has real slippage ✅ FIXED
2. **Lookahead bias** - Features use future data ⚠️ CHECK
3. **Execution timing** - WF instant, live has latency ❌ NOT SIMULATED
4. **Partial fills** - WF assumes full fill ❌ NOT SIMULATED

### **Validation Plan:**
1. Run paper mode 7 days (configs/live_acc1.yaml, mode: paper)
2. Compare: Paper P&L vs WF projection (same period)
3. Calculate gap: (Live - WF) / WF × 100%
4. Target: Gap < 25% acceptable

**If gap > 25%:**
- Add latency simulation (50-200ms)
- Add partial fill logic
- Tighten profit filter ($20 threshold)

---

## Implementation Status

**Created files:**
- ✅ `regime_detection.py` (155 lines, regime scoring + HMM)
- ✅ `feature_stability.py` (224 lines, drift tracking)

**Next immediate actions:**
1. Wait for 3-fold WF baseline results (running now)
2. Integrate regime detection into dataset.py
3. Test WF with regime filter enabled
4. Compare: Baseline vs Regime-filtered performance

---

## Questions to Answer

1. **Baseline performance?** → Running 3-fold WF now
2. **Regime detection impact?** → Test after integration
3. **WF vs Live gap?** → Need 7-day paper validation
4. **Feature drift?** → Need to log importance per fold

**Timeline:** 2 weeks to validate Phase 1 (regime detection)

---

**Last Updated:** May 7, 2026  
**Status:** Baseline WF running, regime detection ready to integrate
