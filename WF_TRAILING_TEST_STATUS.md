# WF Trailing Comparison Test — $500 Capital

**Start Time:** 2026-05-08 11:32:00  
**Status:** 🔄 RUNNING...

---

## Test Setup

### Period
- **Start:** 2024-01-01
- **End:** 2026-05-08 (present)
- **Duration:** ~28 months

### Configuration
- **Capital:** $500 per month (monthly reset)
- **Risk per trade:** 5% ($25/trade)
- **Max positions:** 5 concurrent
- **Test bars:** 6,000 M15 bars (~1 month)
- **Step bars:** 6,000 (no overlap, sequential months)
- **Dynamic slippage:** ENABLED (session/volume/volatility based)
- **Kill switch:** ENABLED (daily loss limit, circuit breaker)
- **Signal threshold:** 0.76 (COMBO133 model)

### Two Tests

#### Test 1: WITH Trailing SL (Realistic)
```yaml
Config: configs/wf_test_trailing.yaml
Output: outputs/wf_test_trailing_trades.csv
Trailing SL:
  - Breakeven at: 0.5R profit
  - Activation at: 1.0R profit  
  - Trail distance: 1.0× ATR
  - Enabled: YES ✅
```

#### Test 2: WITHOUT Trailing SL (Baseline)
```yaml
Config: configs/wf_test_notrail.yaml
Output: outputs/wf_test_notrail_trades.csv
Trailing SL:
  - Disabled via --no-trail flag
  - Fixed TP at 3.5R
  - No breakeven, no trailing
  - Enabled: NO ❌
```

---

## Purpose

**Compare trailing SL impact on:**
1. **Daily/monthly profit** — How much trailing reduces profit?
2. **Win rate** — Does trailing protect more losses or cut more wins?
3. **Avg win RR** — How much do winning trades get cut?
4. **Gap_rr trigger rate** — Does trailing actually trigger in WF?
5. **Drawdown** — Does trailing reduce max DD?

**Expected results:**
- Trailing reduces profit by 10-30% (matches Live behavior)
- Trailing triggers on 20-40% of winning trades (if working)
- Avg win RR drops from 2.2-2.5R → 1.6-1.9R
- gap_rr != 0 for high RR wins (evidence of trailing)

---

## Execution Details

### Command Line

**Test 1: WITH Trailing**
```bash
python scripts/walkforward_ict_wyckoff.py \
  configs/wf_test_trailing.yaml \
  --test-start 2024-01-01 \
  --monthly-reset \
  --test-bars 6000 \
  --step-bars 6000 \
  --cache
```

**Test 2: WITHOUT Trailing**
```bash
python scripts/walkforward_ict_wyckoff.py \
  configs/wf_test_notrail.yaml \
  --test-start 2024-01-01 \
  --monthly-reset \
  --test-bars 6000 \
  --step-bars 6000 \
  --no-trail \  # ← Key flag
  --cache
```

### Flags Explained

| Flag | Purpose |
|------|---------|
| `--test-start 2024-01-01` | Skip folds before 2024-01-01 |
| `--monthly-reset` | Reset balance to $500 every 30 days |
| `--test-bars 6000` | ~1 month per fold (30 days × ~200 M15 bars/day) |
| `--step-bars 6000` | No overlap (sequential months) |
| `--no-trail` | Disable trailing SL for Test 2 |
| `--cache` | Use cached dataset (faster) |

---

## Expected Folds

| Fold | Period | Test Days |
|------|--------|-----------|
| 1 | 2024-01 | ~30 |
| 2 | 2024-02 | ~30 |
| 3 | 2024-03 | ~30 |
| ... | ... | ... |
| 28 | 2026-04 | ~30 |
| 29 | 2026-05 | ~8 (partial) |

**Total:** ~29 folds, ~840 test days

---

## Key Metrics to Compare

### Profit Metrics
- **Daily P&L:** Target $50-60/day with trailing
- **Monthly P&L:** Target $1,500-1,800/month
- **Success rate:** % months profitable
- **Total profit:** 29 months cumulative

### Trading Metrics
- **Total trades:** Count per config
- **Win rate:** % profitable trades
- **Avg win RR:** Mean realized_rr for wins
- **Avg loss RR:** Mean realized_rr for losses
- **Profit factor:** Gross win / Gross loss

### Trailing Evidence
- **gap_rr != 0 count:** How many trades hit trailing
- **gap_rr % of trades:** Trailing trigger rate
- **Avg gap_rr for wins:** Profit lost to trailing
- **High RR wins (≥2R) with gap:** Evidence trailing works

---

## Progress Tracking

### Test 1: WITH Trailing SL
- [x] Config created
- [x] Started at 11:32:00
- [ ] Feature building (~5 min)
- [ ] Fold 1-29 training & testing (~30-45 min)
- [ ] Report generation
- [ ] COMPLETED

### Test 2: WITHOUT Trailing SL
- [x] Config created
- [ ] Started (after Test 1 completes)
- [ ] Feature building (~5 min)
- [ ] Fold 1-29 training & testing (~30-45 min)
- [ ] Report generation
- [ ] COMPLETED

### Comparison
- [ ] Load both reports
- [ ] Calculate monthly stats
- [ ] Compare key metrics
- [ ] Generate summary

**Estimated total time:** 60-90 minutes

---

## Files Generated

### Configs
- ✅ `configs/wf_test_trailing.yaml`
- ✅ `configs/wf_test_notrail.yaml`

### Scripts
- ✅ `scripts/run_wf_trailing_comparison.py`

### Outputs (will be created)
- `outputs/wf_test_trailing_model.pkl`
- `outputs/wf_test_trailing_report.json`
- `outputs/wf_test_trailing_trades.csv`
- `outputs/wf_test_notrail_model.pkl`
- `outputs/wf_test_notrail_report.json`
- `outputs/wf_test_notrail_trades.csv`
- `outputs/wf_trailing_comparison_YYYYMMDD_HHMMSS.log`

### Logs
- `outputs/walkforward_log_wf_test_trailing.txt`
- `outputs/walkforward_log_wf_test_notrail.txt`

---

## Expected Results

### Baseline Scenario (NO Trailing)

```
Period: 2024-01 to 2026-05
Capital: $500, monthly reset
Friction: 0.17-0.30 (dynamic slippage)

Expected:
  - Daily: $55-75/day
  - Monthly: $1,650-2,250
  - Win months: 20-24/29 (69-83%)
  - Win rate: 48-52%
  - Avg win RR: 2.2-2.5R (full TP)
  - Trailing triggered: 0% (disabled)
```

### Realistic Scenario (WITH Trailing)

```
Period: 2024-01 to 2026-05
Capital: $500, monthly reset
Friction: 0.17-0.30 (dynamic slippage)
Trailing: BE at 0.5R, activate at 1.0R, trail 1.0×ATR

Expected:
  - Daily: $40-55/day
  - Monthly: $1,200-1,650
  - Win months: 16-20/29 (55-69%)
  - Win rate: 48-52% (same)
  - Avg win RR: 1.6-1.9R (trailing cuts)
  - Trailing triggered: 20-40% wins

Profit reduction: -25-35% vs baseline
```

### Gap Analysis

```
Difference (Trailing - Baseline):
  - Daily P&L: -$15-20/day (-27-36%)
  - Avg win RR: -0.3-0.6R (-14-24%)
  - Trailing evidence: gap_rr > 0 for 20-40% wins

If gap_rr = 0 for all trades:
  → Trailing NOT working (M1 data insufficient)
  → Need probabilistic cut (see WF_ENGINE_FIX_EXPLANATION.md)
```

---

## Success Criteria

### ✅ Test PASSES if:
1. Both WF runs complete without errors
2. Test 1 (trailing) shows gap_rr != 0 for some trades (>5%)
3. Test 1 profit is 20-40% lower than Test 2
4. Avg win RR drops by 0.3-0.6R with trailing
5. Both results close to Live behavior

### ❌ Test FAILS if:
1. gap_rr = 0 for ALL trades in Test 1 (trailing not triggering)
2. Profit difference <10% (trailing not impacting)
3. Avg win RR same in both tests (trailing not working)
4. Either WF crashes or produces invalid results

### ⚠️ Need Follow-up if:
- Trailing triggers but impact >50% (too aggressive)
- Trailing never triggers (need probabilistic fix)
- Results still 30%+ above Live (need more conservatism)

---

## Next Steps After Completion

### If Trailing Works (gap_rr > 0 for 20%+ trades):
1. ✅ Trailing SL working correctly in WF
2. Use Test 1 results as realistic baseline
3. Compare with Live data after 2-4 weeks
4. Adjust trail_atr_multiple if needed (0.8-1.2 range)

### If Trailing Doesn't Trigger (gap_rr = 0):
1. Implement probabilistic trailing cut (WF_ENGINE_FIX_EXPLANATION.md)
2. Use realistic_replay script with cut_rate 0.20-0.30
3. Add exit slippage multiplier
4. Re-test with combined fixes

### If Results Still Too Optimistic:
1. Increase dynamic slippage multiplier (1.3× instead of 1.2×)
2. Add exit slippage penalty (1.5-2.0× entry slip)
3. Consider lower risk (4% instead of 5%)
4. Tighten signal threshold (0.80 instead of 0.76)

---

## Monitoring Commands

### Check Progress (while running)

```bash
# Test 1 log
tail -f outputs/walkforward_log_wf_test_trailing.txt

# Test 2 log  
tail -f outputs/walkforward_log_wf_test_notrail.txt

# Main comparison log
tail -f outputs/wf_trailing_comparison_*.log
```

### Quick Status Check

```bash
# Check if reports generated
ls -lh outputs/wf_test_*_report.json

# Check if trades generated
wc -l outputs/wf_test_*_trades.csv

# Check process
ps aux | grep walkforward
```

### Intermediate Results

```python
# Check Test 1 progress
python3 << EOF
import json
from pathlib import Path

report = Path("outputs/wf_test_trailing_report.json")
if report.exists():
    data = json.loads(report.read_text())
    print(f"Completed folds: {len(data.get('results', []))}")
    print(f"Last fold: {data.get('results', [{}])[-1].get('test_end', 'N/A')}")
EOF
```

---

## Contacts & References

- **Main doc:** `REALISTIC_TARGET_RECOMMENDATION.md`
- **Fix guide:** `WF_ENGINE_FIX_EXPLANATION.md`
- **Dynamic slip:** `DYNAMIC_SLIPPAGE_SUMMARY.md`
- **Realistic replay:** `scripts/combo133_realistic_replay.py`
- **WF script:** `scripts/walkforward_ict_wyckoff.py`

---

**Status:** Test running, check back in 60-90 minutes for results...
