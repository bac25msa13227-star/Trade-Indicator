# HYPERPARAMETER SWEEP RESULTS — OPTIMAL CONFIG

**Generated:** 2026-05-08
**Total configs tested:** 150 (6 thresholds × 5 risk levels × 5 max_positions)
**Data:** 9,948 trades from 2023-12 to 2026-04 with cut_rate=0.30

---

## 🏆 OPTIMAL CONFIG (BEST OVERALL)

| Parameter | Value | Reason |
|-----------|-------|--------|
| **Signal threshold** | **0.80** | Highest quality signals, best PF=1.46 |
| **Risk per trade** | **6%** | Maximizes daily P&L ($76.39/day) |
| **Max positions** | **7** | Best balance of opportunity & diversification |
| **Cut rate** | **0.30** | Keep 70% profit above BE (calibrated) |
| **Starting capital** | **$500** | Reset monthly |

### Expected Performance:
- **Avg daily P&L:** $76.39 ± $187.20 (HIGH VARIANCE!)
- **Median daily:** $14.14 (more realistic expectation)
- **Success rate:** 55.2% months hit >$10/day target
- **Win rate:** 52.2%
- **Profit factor:** 1.46
- **Max DD:** 44.7%
- **Trades/month:** ~69 (~3.2 trades/day)
- **Trailing cuts/month:** ~35 (50% of trades)

---

## 📊 PARAMETER ANALYSIS

### 1. Signal Threshold (MOST IMPORTANT!)

| Threshold | Avg$/day | Median | Success% | WR% | PF | Total Trades |
|-----------|----------|--------|----------|-----|----|--------------|
| 0.70 | **-$9.65** ❌ | -$18 | 8% | 38.9% | 0.82 | 9,016 |
| 0.72 | -$4.38 | -$12 | 18% | 40.2% | 0.87 | 7,133 |
| 0.74 | $6.91 | $1 | 19% | 42.8% | 0.97 | 5,518 |
| 0.76 | $11.95 | $2 | 28% | 45.3% | 1.10 | 4,071 |
| 0.78 | $18.44 | $5 | 41% | 47.6% | 1.23 | 2,949 |
| **0.80** | **$26.77** ✅ | **$9** | **46%** | **50.8%** | **1.46** | **2,102** |

**Key insights:**
- Threshold 0.70 → LOSS -$9.65/day (38.9% WR, PF 0.82)
- Threshold 0.80 → WIN +$26.77/day (50.8% WR, PF 1.46)
- **Fewer, higher-quality signals = better profits**
- Trade count drops 77% (9,016 → 2,102) but profitability 3.7x better

**Recommendation:** **Use 0.80** — even 0.78 is significantly worse ($18.44 vs $26.77)

---

### 2. Risk Per Trade (LINEAR RELATIONSHIP)

| Risk% | Avg$/day | Median | Success% | WR% | PF | Effect on PF |
|-------|----------|--------|----------|-----|----|--------------|
| 2% | $2.70 | $1.39 | 21% | 44.3% | **1.13** | Best PF |
| 3% | $4.98 | $2.18 | 26% | 44.3% | 1.10 | Good PF |
| 4% | $7.86 | $2.61 | 28% | 44.3% | 1.07 | OK PF |
| 5% | $11.24 | $4.14 | 28% | 44.3% | 1.05 | Lower PF |
| **6%** | **$14.92** | **$5.07** | **28%** | **44.2%** | **1.02** | Lowest PF |

**Key insights:**
- P&L scales LINEARLY with risk (2% → 6% = 2.7x more P&L)
- Win rate stays CONSTANT (~44.3%)
- Profit factor DECREASES as risk increases (friction/slippage overhead)
- Success rate plateaus at 28% for risk ≥4%

**Trade-off:**
- **Conservative (2-3%):** Lower P&L but better PF (1.10-1.13)
- **Moderate (4-5%):** Balanced approach
- **Aggressive (6%):** Max P&L but PF drops to 1.02 (barely profitable)

**Recommendation:** **Use 5-6%** for max P&L, accept lower PF. If prefer safety → use 3-4%.

---

### 3. Max Positions (OPPORTUNITY SCALING)

| MaxPos | Avg$/day | Median | Success% | WR% | PF | Trade Capacity |
|--------|----------|--------|----------|-----|----|----------------|
| 3 | -$0.03 | -$1.33 | 21% | 42.5% | 1.01 | Limited |
| 4 | $4.12 | $0.48 | 24% | 43.8% | 1.06 | OK |
| 5 | $7.94 | $2.20 | 27% | 44.5% | 1.08 | Good |
| 6 | $13.13 | $5.16 | 29% | 45.0% | 1.10 | Better |
| **7** | **$16.53** | **$6.80** | **31%** | **45.5%** | **1.13** | **Best** |

**Key insights:**
- More positions → more opportunities → better P&L
- Win rate INCREASES with more positions (42.5% → 45.5%)
- Profit factor IMPROVES with more positions (1.01 → 1.13)
- Success rate increases steadily (21% → 31%)

**Why more is better:**
- Portfolio diversification effect
- Capture more high-quality signals simultaneously
- Better risk-adjusted returns

**Recommendation:** **Use 7** — clear winner across all metrics.

---

## 🎯 TOP 10 CONFIGS (by avg daily P&L)

| Rank | Threshold | Risk% | MaxPos | Avg$/day | Med$/day | Std$/day | Succ% | WR% | PF | Trades |
|------|-----------|-------|--------|----------|----------|----------|-------|-----|----|----|
| **1** | **0.80** | **6** | **7** | **$76.39** | **$14.14** | **$187.20** | **55.2%** | **52.2%** | **1.46** | **2,102** |
| 2 | 0.80 | 6 | 6 | $64.91 | $13.50 | $160.78 | 55.2% | 51.5% | 1.42 | 2,102 |
| 3 | 0.80 | 5 | 7 | $54.10 | $12.26 | $117.56 | 55.2% | 52.2% | 1.50 | 2,102 |
| 4 | 0.78 | 6 | 7 | $52.74 | $12.87 | $153.63 | 51.7% | 48.7% | 1.22 | 2,949 |
| 5 | 0.80 | 5 | 6 | $46.42 | $11.73 | $102.29 | 55.2% | 51.5% | 1.46 | 2,102 |
| 6 | 0.78 | 6 | 6 | $46.10 | $10.75 | $139.61 | 51.7% | 48.3% | 1.19 | 2,949 |
| 7 | 0.80 | 6 | 5 | $45.66 | $9.32 | $96.77 | 48.3% | 50.8% | 1.39 | 2,102 |
| 8 | 0.78 | 5 | 7 | $39.09 | $11.09 | $102.95 | 51.7% | 48.7% | 1.25 | 2,949 |
| 9 | 0.80 | 4 | 7 | $36.66 | $10.41 | $69.73 | 51.7% | 52.2% | 1.54 | 2,102 |
| 10 | 0.76 | 6 | 7 | $35.35 | -$6.66 | $114.44 | 37.9% | 46.6% | 1.08 | 4,071 |

**Pattern:** ALL top 10 use threshold 0.78-0.80, high risk (5-6%), max positions (6-7).

---

## ⚠️ RISK WARNINGS

### High Variance Alert!
- **Best config:** Avg=$76.39, Std=$187.20 → **Coefficient of variation = 2.45**
- **1-sigma range:** -$111 to +$264/day (HUGE swing!)
- **Median=$14.14** much lower than mean → expect frequent small wins, rare huge wins
- **Max DD:** 44.7% (can lose nearly half capital in one bad month)

### Realistic Expectations:
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Median daily** | $14.14 | Typical day result |
| **Mean daily** | $76.39 | Inflated by outlier big-win months |
| **Success rate** | 55.2% | ~16/29 months hit target |
| **Best month** | +$7,306 | Rare event (2024-09) |
| **Worst month** | -$385 | Can happen (2025-01) |

**Recommendation:** Use **median=$14.14/day** as realistic expectation, not mean=$76.39.

---

## 💡 ALTERNATIVE CONFIGS

### Option A: BALANCED (Recommended for most users)
- **Threshold:** 0.80
- **Risk:** 5%
- **Max positions:** 6
- **Expected:** $46.42/day avg, $11.73 median, 55.2% success
- **Pros:** Lower variance ($102 std vs $187), good success rate
- **Cons:** 39% lower P&L than optimal

### Option B: CONSERVATIVE (Risk-averse)
- **Threshold:** 0.80
- **Risk:** 3%
- **Max positions:** 5
- **Expected:** $22.52/day avg, $7.39 median, 48.3% success
- **Pros:** Lowest std ($61), best PF=1.59, safe
- **Cons:** 71% lower P&L than optimal

### Option C: AGGRESSIVE (Max P&L)
- **Threshold:** 0.80
- **Risk:** 6%
- **Max positions:** 7
- **Expected:** $76.39/day avg, $14.14 median, 55.2% success
- **Pros:** Highest P&L, good success rate
- **Cons:** Extreme variance ($187 std), 44.7% max DD

---

## 📝 DEPLOYMENT CHECKLIST

### Before Live Trading:
1. ✅ Update `configs/live_acc1.yaml`:
   ```yaml
   risk_per_trade: 0.06  # 6%
   max_open_positions: 7
   signal_threshold: 0.80  # Critical!
   trailing_sl:
     enabled: true
     breakeven_at_rr: 0.5
     activation_rr: 1.0
     trail_atr_multiple: 1.0
   ```

2. ✅ Enable dynamic slippage (already in config):
   ```yaml
   use_dynamic_slippage: true
   entry_slippage_atr_frac: 0.020  # 2% ATR
   ```

3. ✅ Set monthly reset:
   ```yaml
   monthly_reset: true
   starting_balance: 500.0
   ```

4. ✅ Monitor key metrics:
   - Daily P&L vs expected median ($14.14)
   - Trailing cuts per month (expect ~35)
   - Win rate (expect 52%)
   - Max DD stay <50%

5. ✅ Kill switches:
   ```yaml
   max_daily_loss_pct: 15  # Stop if -15% in one day
   max_monthly_loss_pct: 30  # Stop if -30% in one month
   ```

---

## 🔍 COMPARISON: ORIGINAL vs OPTIMAL

| Metric | Original (risk=5%, mp=5, th=0.80) | Optimal (risk=6%, mp=7, th=0.80) | Change |
|--------|-----------------------------------|----------------------------------|--------|
| Avg daily P&L | $41.72 | $76.39 | **+83%** |
| Median daily | $22.09 | $14.14 | -36% |
| Std daily | $73.10 | $187.20 | **+156%** |
| Success rate | 34.5% | 55.2% | **+60%** |
| Win rate | 50.9% | 52.2% | +2.6% |
| Profit factor | 1.45 | 1.46 | +0.7% |
| Max DD | 38.6% | 44.7% | +16% |
| Trades/month | 63 | 69 | +10% |

**Key takeaway:** Optimal config has **83% higher mean** but **156% higher variance**. Success rate improves dramatically (34.5% → 55.2%) due to more opportunities (7 positions vs 5).

---

**Final Recommendation:** Deploy with **Threshold=0.80, Risk=6%, MaxPos=7** for maximum P&L. Expect realistic **$14-20/day median** (not $76 mean). Monitor closely for first month and adjust if actual trailing cuts differ significantly from predicted 35/month.
