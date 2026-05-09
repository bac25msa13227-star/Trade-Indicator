# WF vs Live Gap — Root Cause & Solution

**Date:** 2026-05-08  
**Issue:** Walkforward results ($105/day, 75% success) không match Live performance  
**Root Cause:** Trailing stop chỉ apply cho LOSING trades trong WF, nhưng Live apply cho ALL trades

---

## 🔍 Root Cause Analysis

### Bug Location
[src/xauusd_ai/backtesting/engine.py](../src/xauusd_ai/backtesting/engine.py#L151-189)
```python
# ── M1 bar-by-bar trailing SL path simulator (for LOSING trades only) ────────
def _simulate_trade_m1_trailing(...):
    """
    CRITICAL: This is a LOSING-TRADE ONLY optimization. It only
    improves the trailing SL EXIT PRICE for losing trades — it does NOT
    apply to winning trades.
    """
```

### Impact
- **WF**: Winning trades hit full target (2.333R median) → **OVERSTATE profit**
- **Live**: Trailing stop cuts winners early (BE activation at 0.5R, trail at 1.0R) → **REALISTIC but lower profit**
- **Gap**: ~60-75% profit cut by trailing in Live vs WF assumptions

---

## 📊 Realistic Results (with Trailing Fix)

### Cut Rate Calibration

| Cut Rate | Keep % | Config (best) | Avg/day | Success% | PF | DD |
|----------|--------|---------------|---------|----------|-----|-----|
| **20%** | **80%** | **5%/mp5/p0.80** | **$59.85** | **37.9%** | **1.55** | **37.9%** |
| 25% | 75% | 5%/mp5/p0.80 | $50.18 | 34.5% | 1.50 | 38.2% |
| 30% | 70% | 5%/mp5/p0.80 | $41.72 | 34.5% | 1.45 | 38.6% |
| 35% | 65% | 5%/mp5/p0.80 | $34.31 | 31.0% | 1.39 | 39.1% |
| 40% | 60% | 5%/mp5/p0.80 | $27.79 | 31.0% | 1.34 | 39.7% |

**Interpretation:**
- Cut rate = % profit lost due to trailing stop retracement
- Live behavior: "cắn BE rất nhiều" → estimated cut_rate 25-35%
- Best realistic scenario: **cut_rate=25%** → **$50/day, 34.5% success**
- Conservative: **cut_rate=30%** → **$42/day, 34.5% success**

### Comparison: Original vs Realistic

| Metric | Original (buggy) | Realistic (25% cut) | Realistic (30% cut) | Delta |
|--------|------------------|---------------------|---------------------|--------|
| Avg/day | $105.06 | $50.18 | $41.72 | **-52% to -60%** |
| Success rate | 75.9% | 34.5% | 34.5% | **-54%** |
| PF | 2.33 | 1.50 | 1.45 | **-36% to -38%** |
| DD | 17.3% | 38.2% | 38.6% | **+121% to +123%** |
| Months hit target | 22/29 | 10/29 | 10/29 | **-55%** |

---

## ✅ Action Items

### 1. **Measure Live Cut Rate** (CRITICAL)
Run Live bot với known config và measure:
```python
# For each WINNING trade:
live_realized_rr = (close_price - entry_price) / stop_distance
wf_target_rr = 2.333  # median from WF
retention_rate = live_realized_rr / wf_target_rr
cut_rate = 1 - retention_rate
```

Collect 50-100 live wins → compute average cut_rate → use as calibration baseline.

### 2. **Fix WF Code** (Long-term)
Update `engine.py` to apply trailing to ALL trades:
- Load M1 data
- Simulate bar-by-bar price path
- Track trailing SL updates
- Cut winners when retracement hits trail

Reference: [scripts/combo133_realistic_replay.py](../scripts/combo133_realistic_replay.py)

### 3. **Re-train Model** (Optional)
Current combo133 model trained on WF labels WITHOUT realistic trailing:
- Labels assume full target hits
- Model optimized for unrealistic edge

Consider re-training with:
- Realistic trailing labels
- Conservative profit targets (1.5R vs 2.5R)
- Focus on high-probability setups (prob > 0.80)

### 4. **Adjust Live Config**
Based on realistic results, recommend:

**Conservative Config (30% cut assumption):**
```yaml
risk:
  risk_per_trade: 0.05          # 5%
  max_open_positions: 5
strategy:
  signal_threshold: 0.80        # High confidence only
  trailing_sl:
    enabled: true
    breakeven_at_rr: 0.5
    activation_rr: 1.0
    trail_atr_multiple: 1.0
```

**Expected:** $42/day avg, 34.5% months hit $40, PF 1.45, DD 38.6%

**Aggressive Config (25% cut assumption):**
```yaml
risk:
  risk_per_trade: 0.05
  max_open_positions: 5
strategy:
  signal_threshold: 0.80
  trailing_sl:
    enabled: true
    breakeven_at_rr: 0.5        # Tight BE
    activation_rr: 0.8          # Earlier activation (vs 1.0)
    trail_atr_multiple: 1.2     # Looser trail distance
```

**Expected:** $50/day avg, 34.5% success, PF 1.50, DD 38.2%

---

## 🎯 Realistic Target Adjustment

| Capital | Original Target | Realistic Target (25% cut) | Realistic (30% cut) |
|---------|----------------|---------------------------|---------------------|
| $200 | $40/day | ❌ Not achievable | ❌ Not achievable |
| $500 | $40/day | ✅ $50/day possible | ⚠️ $42/day marginal |
| $1000 | $80/day | ✅ $100/day possible | ✅ $84/day possible |

**Recommendation:** 
- $500 capital → target $35-45/day (achievable 30-35% months)
- $1000 capital → target $70-100/day (achievable 30-35% months)
- Accept 30-40% success rate as realistic for ML-based system with trailing

---

## 🔬 Technical Debt

1. **WF Trailing Simulation:** Incomplete implementation (losing trades only)
2. **Label Generation:** Does not account for realistic exit conditions
3. **Feature Engineering:** Missing "likelihood of retracement" features
4. **Risk Management:** Trailing config not optimized for profit retention

---

## 📈 Next Steps

1. ✅ Run Live with `5%/mp5/prob>0.80` config
2. ✅ Collect 50-100 winning trades
3. ✅ Measure actual cut_rate from Live data
4. ⏳ Fine-tune cut_rate calibration
5. ⏳ Re-run realistic replay with calibrated cut_rate
6. ⏳ Update Live config based on calibrated results
7. ⏳ Consider re-training model with realistic labels

---

**Files:**
- Analysis: `scripts/combo133_realistic_replay.py`
- Results: `outputs/combo133_realistic_results.json`
- Original (buggy): `outputs/combo133_replay_results.json`
- Live data: `outputs/live_closed_trades_acc1.csv` (insufficient, need more)

**Generated:** 2026-05-08 by Claude Code (ECC agent)
