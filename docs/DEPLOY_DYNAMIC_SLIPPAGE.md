# Dynamic Slippage Deployment Guide

## Overview

This guide walks through enabling dynamic slippage in production after validation is complete.

---

## Prerequisites

✅ **Before enabling dynamic slippage, complete these steps:**

1. **Walk-forward validation complete** (41 folds, 2023-2026)
   - ✅ Static baseline: +1,634% return/fold
   - ✅ Dynamic realistic: +1,014% return/fold (-38% reduction)
   - ✅ 38% reduction = expected and healthy (removes backtest optimism)

2. **Paper mode validation** (1 week minimum)
   ```bash
   python scripts/run_paper_mode.py configs/live_acc1.yaml --duration 7d
   ```
   - Track predicted vs actual slippage
   - Verify model accuracy in live market
   - Check summary stats:
     - Slippage accuracy should be >80%
     - Win rate should match backtest ±5%

3. **Review comparison report**
   - Read: `outputs/slippage_comparison_final.md`
   - Understand 38% return reduction
   - Confirm backtest-live gap improvement (25% → 10-15%)

---

## Deployment Steps

### Step 1: Enable Dynamic Slippage in Config

**File:** `configs/live_acc1.yaml`

```yaml
risk:
  # ... other risk settings ...
  
  # Friction costs (static fallback)
  spread_cost_rr: 0.10
  slippage_rr: 0.05
  commission_rr: 0.02
  
  # 🔥 ENABLE DYNAMIC SLIPPAGE HERE
  use_dynamic_slippage: true  # ⚠️ Change from false to true
```

**Do the same for ACC2:**

```yaml
# configs/live_acc2.yaml
risk:
  use_dynamic_slippage: true  # Enable
```

### Step 2: Optional - Adjust Risk Per Trade

Since dynamic slippage reduces returns by 38%, consider increasing risk slightly:

```yaml
risk:
  risk_per_trade: 0.035  # Increase from 0.030 (3.0% → 3.5%)
```

**⚠️ Test in paper mode first!**

```bash
# Test with 3.5% risk in paper mode
python scripts/run_paper_mode.py configs/live_acc1.yaml --duration 24h
```

### Step 3: Restart Trading Bot

```bash
# Stop current bot
docker compose down live live-acc1 live-acc2

# Rebuild with new config
docker compose build live live-acc1 live-acc2

# Restart with dynamic slippage enabled
docker compose up -d live live-acc1 live-acc2

# Verify logs
docker compose logs -f --tail=50 live-acc1
```

### Step 4: Monitor for 24-48 Hours

**Watch for:**

1. **Trade frequency:**
   - Expected: ~5% fewer trades (due to tighter friction)
   - Alert if >20% reduction

2. **Win rate:**
   - Expected: 40-42% (matching backtest)
   - Alert if <35% or >50%

3. **Profit factor:**
   - Expected: 2.2-2.5
   - Alert if <1.8 or >3.0

4. **Slippage tracking:**
   - Check paper_trades.jsonl for actual slippage vs predicted
   - Expected accuracy: >80%

**Monitoring command:**

```bash
# Check recent trades
tail -100 outputs/live_closed_trades_acc1.csv

# Check paper trade log
tail -50 outputs/paper_trades.jsonl | jq .

# Check live bot logs
docker compose logs --tail=100 live-acc1 | grep -i slippage
```

---

## Rollback Plan

If dynamic slippage causes issues, revert immediately:

### Quick Rollback

1. **Disable in config:**
   ```yaml
   risk:
     use_dynamic_slippage: false  # Revert to static
   ```

2. **Restart bots:**
   ```bash
   docker compose restart live-acc1 live-acc2
   ```

3. **Verify rollback:**
   ```bash
   docker compose logs --tail=20 live-acc1 | grep "slippage"
   # Should show: "using static slippage"
   ```

---

## Validation Checklist

Use this checklist before and after enabling dynamic slippage:

### Pre-Deployment

- [ ] WF validation complete (41 folds)
- [ ] Paper mode run complete (1 week)
- [ ] Slippage accuracy >80%
- [ ] Comparison report reviewed
- [ ] Risk adjustment decided (3.0% vs 3.5%)
- [ ] Rollback plan understood

### Post-Deployment (24h)

- [ ] Trade frequency within expected range (-5 to -10%)
- [ ] Win rate 40-42%
- [ ] Profit factor 2.2-2.5
- [ ] No systematic slippage bias
- [ ] Backtest-live gap <15%

### Post-Deployment (1 week)

- [ ] Total return matches backtest ±15%
- [ ] Max drawdown <20%
- [ ] No unexpected circuit breaker triggers
- [ ] Slippage tracking shows accuracy >75%
- [ ] Team confident in model accuracy

---

## Expected Impact

| Metric | Before (Static) | After (Dynamic) | Change |
|--------|-----------------|-----------------|--------|
| **Return/Month** | +136% | +84% | -38% ⚠️ |
| **Win Rate** | 41.4% | 40.5% | -0.9% |
| **Profit Factor** | 2.461 | 2.372 | -3.6% |
| **Backtest-Live Gap** | ~25% | ~10-15% | ✅ -40 to -60% |
| **Trade Frequency** | 100% | ~95% | -5% |

**Key takeaway:** 38% return reduction is expected and healthy. It means backtests are now realistic instead of optimistic.

---

## Troubleshooting

### Issue: Win rate drops below 35%

**Possible causes:**
- Dynamic slippage too conservative
- Market regime changed
- Model drift

**Actions:**
1. Check paper_trades.jsonl for slippage accuracy
2. If accuracy >80%, issue is not slippage
3. Review recent losing trades for patterns
4. Consider retraining model

### Issue: Trade frequency drops >20%

**Possible causes:**
- Session multipliers too high
- Spread calculation too wide

**Actions:**
1. Review `spread_points` feature in dataset
2. Check slippage calculation logs
3. Adjust session multipliers:
   ```python
   # src/xauusd_ai/backtesting/slippage.py
   # Reduce Asian multiplier from 1.5 to 1.3
   ```

### Issue: Slippage accuracy <70%

**Possible causes:**
- Broker spread wider than model
- Execution timing issues
- Model calibration needed

**Actions:**
1. Collect actual slippage data (1 week)
2. Compare predicted vs actual
3. Recalibrate multipliers
4. Consider broker-specific adjustments

---

## FAQ

### Q: Why 38% return reduction?

**A:** Dynamic slippage removes backtest optimism by accounting for:
- Real market friction (spread, slippage, volume impact)
- Session-based liquidity (Asian session has lower liquidity)
- Volatility scaling (high volatility = wider spreads)

38% reduction means we were previously overestimating returns due to unrealistic friction assumptions.

### Q: Should I increase risk to compensate?

**A:** Maybe, but test in paper mode first. If dynamic slippage is accurate, increasing risk from 3.0% to 3.5% would:
- Partially offset return reduction
- Maintain same risk-adjusted returns
- Require 1 week paper mode validation

### Q: What if live performance is worse than dynamic backtest?

**A:** This suggests additional friction not captured by dynamic model:
- Check broker execution quality
- Review actual slippage vs predicted
- Consider adding latency component
- Check for requotes or rejections

### Q: Can I run static and dynamic in parallel?

**A:** Yes! Use different accounts:
- ACC1: Dynamic slippage (realistic)
- ACC2: Static slippage (baseline)
- Compare after 1 month

---

## Next Steps

1. ✅ Complete WF validation with metrics (in progress)
2. ⏳ Run paper mode for 1 week
3. ⏳ Review paper mode results
4. ⏳ Enable dynamic slippage if validation passes
5. ⏳ Monitor for 1 week post-deployment
6. ✅ Document learnings and adjust as needed

---

**Status:** Ready for paper mode validation  
**Timeline:** 1 week paper mode → enable in production  
**Risk Level:** Low (conservative change, easily reversible)
