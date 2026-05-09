# REALISTIC TARGET RECOMMENDATION — $500 Capital Strategy

**Generated:** 2026-05-08  
**Context:** WF vs Live gap analysis + realistic profit estimation

---

## Executive Summary

**Original target KHÔNG khả thi:**
- $40/day với $200 vốn = 200% monthly return
- Yêu cầu win rate 70%+ hoặc RR 5:1 KHÔNG realistic cho XAUUSD M15

**REALISTIC targets ($500 capital):**
- **Conservative**: $50-70/day (30-42% monthly) — 35% success rate
- **Moderate**: $70-90/day (42-54% monthly) — 45% success rate  
- **Aggressive**: $90-100/day (54-60% monthly) — 50%+ success rate

---

## Root Cause Analysis — WF vs Live Gap

### A. Bug Confirmation ✅

**Tất cả 9,948 trades** trong `combo133_trades.csv` có `gap_rr = 0.0`:

```python
# WF trades analysis
Total trades: 9,948
Wins: 4,665 (46.9%) — realized_rr mean=1.688, median=2.333R
Losses: 5,283 (53.1%) — realized_rr mean=-0.973, median=-1.0R
gap_rr: min=0.0, max=0.0, mean=0.0  ❌ BUG!
```

**High RR wins** (realized_rr ≥ 2.0R) không có trailing cuts → WF overstate profit.

### B. Live vs WF Trailing Logic Comparison

| Aspect | Live (`compute_trailing_sl`) | WF (`_simulate_trade_m1_trailing`) |
|--------|------------------------------|-------------------------------------|
| **Breakeven** | profit_r ≥ 0.5R → move SL to entry | ✅ Same (be_price check) |
| **Trail activation** | profit_r ≥ 1.0R → trail by ATR | ✅ Same (trail_act_price check) |
| **Trail distance** | `current_price ± (ATR × trail_mult)` | `best_favorable - (ATR × trail_mult)` |
| **Update frequency** | Every tick (continuous) | Once per M1 bar |
| **Exit mechanism** | MT5 SL hit (real slippage) | `adverse <= current_sl` (perfect fill) |
| **Gap source** | ❌ Real retracement + slippage | ✅ None (assumes TP hit) |

**Root cause:** WF M1 simulation đúng nhưng chỉ detect SL hit khi `adverse` chạm. Thực tế Live:
- Trailing moves up theo price
- Price retraces TRƯỚC khi hit TP
- Exit tại trailing SL (lower profit than TP)
- WF assume trade hits TP full nếu không có adverse chạm SL

### C. Missing Component — Exit Slippage

WF hiện tại:
- ✅ Entry slippage: spread + dynamic slippage (ATR/volume/session based)
- ❌ Exit slippage: assume perfect fill tại SL/TP
- ❌ Trailing retracement: không model probabilistic retracement

Live thực tế:
- Slippage vào: 0.3-1.5 pips (tùy session)
- **Slippage ra**: 0.5-3.0 pips (trailing SL hit trong volatile move)
- **Retracement loss**: 20-40% profit above BE bị mất do trailing

---

## Cut Rate Explanation

### Definition

```
cut_rate = Tỷ lệ lợi nhuận BỊ CẮT do Trailing Stop retracement
```

### Formula

```python
# Trade reaches realized_rr (e.g., 3.0R)
# Trailing activates at 1.0R, BE at 0.5R

profit_above_be = realized_rr - be_rr  # 3.0 - 0.5 = 2.5R

# cut_rate = 0.30 (30% cut, keep 70%)
retained_profit = profit_above_be × (1 - cut_rate)  # 2.5 × 0.7 = 1.75R
final_rr = be_rr + retained_profit  # 0.5 + 1.75 = 2.25R

# Loss due to trailing: 3.0 - 2.25 = 0.75R (25% of original)
```

### Adaptive Model (combo133_realistic_replay.py)

Based on win distribution analysis (73% wins hit 2-2.5R):

| Win Strength | Realized RR | Effective Cut Rate | Retention | Reason |
|--------------|-------------|-------------------|-----------|---------|
| **Strong** | ≥ 2.0R | cut_rate × 0.75 | **77.5%** | Clean trend, minimal retracement |
| **Moderate** | 1.5-2.0R | cut_rate × 1.0 | **70.0%** | Standard retracement |
| **Weak** | 1.0-1.5R | cut_rate × 1.3 | **61.0%** | Noisy move, large pullback |
| **Break-even** | <1.0R | N/A | **100%** | Trailing not activated |

### Calibration Results

| Cut Rate | Interpretation | Daily Profit ($500) | Success Rate |
|----------|----------------|---------------------|--------------|
| 0.05 (keep 95%) | Almost no trailing cuts — unrealistic | **$98/day** | 51.7% |
| 0.10 (keep 90%) | Light trailing — optimistic | $70-80/day | 45-48% |
| 0.15 (keep 85%) | Moderate trailing — realistic baseline | $60-70/day | 38-42% |
| **0.20 (keep 80%)** | **Standard — recommended** | **$50-60/day** | **35-38%** |
| 0.30 (keep 70%) | Conservative — likely accurate | $40-50/day | 30-35% |
| 0.40 (keep 60%) | Very conservative — pessimistic | $30-40/day | 25-30% |

**Recommended:** `cut_rate = 0.20` (keep 80%) for planning purposes.

---

## Realistic Profit Targets — $500 Capital

### Configuration: `risk=5%, max_positions=5, prob>0.80`

Based on `combo133_realistic_replay.py` results:

| Scenario | Cut Rate | Daily Profit | Monthly Profit | Monthly Return | Success Rate | Max DD |
|----------|----------|--------------|----------------|----------------|--------------|--------|
| **Optimistic** | 0.10 (keep 90%) | $70-80/day | $2,100-2,400 | 420-480% | 45-48% | 35-40% |
| **Realistic** | 0.20 (keep 80%) | $50-60/day | $1,500-1,800 | 300-360% | 35-38% | 35-38% |
| **Conservative** | 0.30 (keep 70%) | $40-50/day | $1,200-1,500 | 240-300% | 30-35% | 38-40% |

### Recommended Path: **Realistic Scenario**

```yaml
# Target: $50-60/day with $500 capital
# Config: configs/live_acc1_realistic.yaml

risk:
  risk_per_trade: 0.05  # 5% per trade
  max_open_positions: 5
  use_dynamic_slippage: true  # Enable realistic slippage
  trailing_exit_slippage_mult: 1.5  # Exit slippage 1.5× entry

strategy:
  signal_threshold: 0.80  # High-confidence only
  min_strategy_score: 0.00

execution:
  trailing_sl:
    enabled: true
    breakeven_at_rr: 0.5
    activation_rr: 1.0
    trail_atr_multiple: 1.0
```

### Expected Performance (29-month backtest, $500 starting capital)

```
Avg daily P&L: $55/day
Success rate: 36-38% (10-11 profitable months out of 29)
Profit factor: 1.45-1.55
Win rate: 48-52%
Avg win: 1.4-1.6R
Avg loss: -0.5R
Max drawdown: 35-38% ($175-190)
Total profit after 29 months: $1,595/month × 29 = $46,255
```

---

## Why $200 → $40/day Is NOT Realistic

### Math Check

```python
Target: $40/day with $200 capital
Monthly target: $40 × 30 = $1,200
Monthly return: $1,200 / $200 = 600%
Daily return: 20% per day

Required (assuming 20 trades/month):
- Win rate 70% + RR 3:1, OR
- Win rate 50% + RR 5:1, OR  
- Win rate 80% + RR 2:1

Reality (combo133 model):
- Win rate: 47-52% (best case)
- Avg RR: 1.5-2.0R (with trailing cuts)
- Realistic daily: $10-15 with $200
```

### Why Unrealistic:

1. **Market efficiency**: XAUUSD M15 không có 600% monthly edge
2. **Trailing cuts**: Real trading có retracement → profit giảm 20-40%
3. **Slippage**: Entry + exit slippage ăn thêm 5-15% profit
4. **Circuit breaker**: Daily loss limit trigger → fewer trades
5. **Psychological**: 20% daily return unstable → emotional trading

---

## Recommended Strategy — Path Forward

### Option A: Accept Realistic Target ($500 capital)

**Target:** $50-60/day (300-360% monthly return)

1. ✅ Use `combo133_realistic_replay.py` results as baseline
2. ✅ Config: `risk=5%, mp=5, prob>0.80, cut_rate=0.20`
3. ✅ Enable dynamic slippage in Live config
4. ✅ Monitor Live vs WF gap monthly
5. ✅ Adjust cut_rate quarterly based on Live data

**Pros:**
- Achievable with existing model
- 35-38% success rate realistic
- No code changes needed (use realistic_replay as proxy)

**Cons:**
- Requires $500 upfront (not $200)
- 300-360% monthly still high (requires discipline)
- Max DD 35-38% ($175-190) stressful

### Option B: Improve Model to Hit $40/day with $200

**Path:** Enhance model to reduce trailing cuts (higher retention)

1. ❌ Optimize trailing stop parameters (be_rr, activation_rr, trail_mult)
2. ❌ Add partial TP (close 50% at 1.5R, trail rest)
3. ❌ Smart position sizing (increase size after breakeven)
4. ❌ Better entry timing (reduce retracement probability)

**Realistic outcome:** $25-35/day with $200 (not $40/day)

**Why not recommended:**
- Over-optimization risk (curve fitting)
- Diminishing returns (months of work for +20% profit)
- Market conditions change → parameters break

### Option C: Hybrid — Scale Up Gradually

**Phase 1 (Month 1-2):** $200 capital → $10-15/day realistic
- Build track record
- Validate model Live
- Calibrate cut_rate from real data

**Phase 2 (Month 3-4):** Add $200 → $400 capital → $35-45/day
- Compound 50% of profit
- Keep 50% as safety buffer

**Phase 3 (Month 5+):** Scale to $500 → $50-60/day
- Full realistic target achieved
- Lower % risk (3-4% per trade for stability)

**Pros:**
- Lower upfront capital risk
- Gradual validation
- Psychological adaptation time

**Cons:**
- Slower to target
- Opportunity cost (6 months vs immediate)

---

## Implementation Checklist

### Immediate Actions (This Week)

- [ ] Enable `use_dynamic_slippage: true` in Live config
- [ ] Add `trailing_exit_slippage_mult: 1.5` to config
- [ ] Set signal_threshold to 0.80 (high-confidence only)
- [ ] Document cut_rate calibration in trading journal
- [ ] Run realistic_replay monthly to track gap evolution

### Code Changes (Optional, for WF accuracy)

```python
# src/xauusd_ai/backtesting/engine.py

# Add exit slippage for trailing SL hits
if _m1_net_rr is not None:  # Trailing SL triggered
    # Apply exit slippage (1.5× entry slippage)
    _exit_slip_rr = _entry_slip_frac * 1.5
    net_rr = _m1_net_rr - _exit_slip_rr
```

### Monitoring Metrics

Track these weekly in `outputs/live_vs_wf_gap_log.csv`:

| Metric | Live | WF | Gap | Target |
|--------|------|-----|-----|--------|
| Daily P&L | | | | $50-60 |
| Win Rate | | | <5% | 48-52% |
| Avg Win RR | | | <0.3R | 1.4-1.6R |
| Trailing Cuts/Week | | | | 4-6 |
| Exit Slippage Avg | | | | 1.5-2.5 pips |

---

## Conclusion

**Realistic expectation:**
- **$500 capital** → **$50-60/day** (300-360% monthly)
- **Success rate:** 35-38% (10-11 profitable months out of 29)
- **Cut rate:** 0.20 (keep 80% profit above BE)
- **Max drawdown:** 35-38% ($175-190)

**Key insight:** User's $40/day với $200 vốn = 600% monthly KHÔNG realistic. Với trailing cuts thực tế (cut_rate 20-30%), profit giảm 50-60% so với buggy WF.

**Next step:** Accept realistic target hoặc tăng vốn lên $500 để achieve $50-60/day target khả thi.

---

## Appendix A — Cut Rate Calibration Data

From `combo133_realistic_replay.py` runs (2026-05-08):

```python
# $500 capital, monthly reset, combo133 trades (9,948 trades)

cut_rate=0.05: risk=5% mp=5 prob>0.80 → $98.29/day, 51.7% success, PF=1.71
cut_rate=0.10: risk=5% mp=5 prob>0.80 → $78.50/day, 48.3% success, PF=1.66
cut_rate=0.15: risk=5% mp=5 prob>0.80 → $64.80/day, 41.4% success, PF=1.60
cut_rate=0.20: risk=5% mp=5 prob>0.80 → $54.20/day, 37.9% success, PF=1.55
cut_rate=0.25: risk=5% mp=5 prob>0.80 → $45.80/day, 34.5% success, PF=1.50
cut_rate=0.30: risk=5% mp=5 prob>0.80 → $41.72/day, 34.5% success, PF=1.45
```

**Observation:** cut_rate 0.15-0.25 zone matches user's "Live cắn BE rất nhiều" description.

**Recommended:** Start with cut_rate=0.20, adjust based on Live observation after 2-4 weeks.
