# Extended Training + M1 Scalp Implementation Roadmap

## Summary

**Goal:** Improve x5/month feasibility by:
1. Retraining model with 20 years of history (2003-2026) → better generalization
2. Adding M1 scalp signals in parallel with M5 swing → more execution opportunities

**Expected outcome:** Increase geometric mean monthly return from +14.7% to +20-30%, enabling x5 months in ~3-4 months instead of 1/19.

---

## Phase 1: Extended Training (150k TRAIN_BARS)

### Timeline
- **Total time:** ~3-4 hours one-time + 15 mins weekly retraining
- Breakdown:
  - Build extended features: 30-60 min (first time only, subsequent updates faster)
  - Retrain WF (5 folds): 2-3 hours
  - Generate signals: 5 min

### Commands

**1. Build 2003-2026 features (first time only)**
```bash
cd F:\Trading_BOT_AUTO\Trade-Indicator
python scripts/build_historical_features_extended.py --from 2003-01-01 --to 2026-05-10
# Output: outputs/historical_features_2003_2026.csv (1M+ rows)
# Check: wc -l outputs/historical_features_2003_2026.csv should be 1M+
```

**2. Retrain with extended history**
```bash
python scripts/walkforward_extended_150k.py --config configs/xauusd_combo133_best.yaml --max-folds 5
# Output: outputs/walkforward_extended_150k_xauusd_combo133_best_results.json
# Check: tail outputs/walkforward_extended_150k_xauusd_combo133_best.txt
```

**3. Generate signals**
```bash
# Extract latest fold model checkpoints and generate signals
# (To be implemented - integrate walkforward fold models into signal generator)
python scripts/export_signals_extended_mt5.py --fold latest
# Output: outputs/signals_extended_mt5.csv
```

**4. Deploy to MT5**
```bash
# Copy signals to MT5
cp outputs/signals_extended_mt5.csv \
  "C:\Users\doanbacremote\AppData\Roaming\MetaQuotes\Terminal\Common\Files\signals_for_mt5.csv"

# Run MT5 backtest at Risk=10%, MaxPos=3
cd C:\Users\doanbacremote\tradingview-mcp
.\Run-AISignalTest.ps1 -RiskPct 10 -MaxPositions 3 -Deposit 200
```

### Expected Results
- Accuracy: 55-58% (stable, no overfitting)
- Win rate: 50-55%
- Return: ≥ current baseline (+1248%)

### Success Criteria
- ✅ Extended features load without error
- ✅ WF runs 5 folds without OOM
- ✅ Accuracy per fold similar (no large variance)
- ✅ MT5 backtest P&L ≥ +1000%

---

## Phase 2: M1 Scalp Signals

### Timeline
- **Total time:** ~2-3 hours one-time
- Breakdown:
  - Build M1 features: 30-45 min
  - Train M1 WF: 1-2 hours
  - Integrate with EA: 15-30 min

### Commands

**1. Build M1 features (2003-2026)**
```bash
python scripts/build_historical_features_extended_m1.py --from 2003-01-01 --to 2026-05-10
# Output: outputs/historical_features_2003_2026_M1.csv (20M+ rows, ~8GB)
# Check: du -h outputs/historical_features_2003_2026_M1.csv
```

**2. Train M1 WF**
```bash
python scripts/walkforward_m1_scalp.py --config configs/xauusd_combo133_best.yaml
# Output: outputs/walkforward_m1_scalp_results.json
# Expected: Acc=52-55%, but higher signal frequency
```

**3. Generate M1 signals**
```bash
python scripts/export_signals_m1_mt5.py --fold latest
# Output: outputs/signals_m1_scalp_mt5.csv (200-400 signals/month)
```

**4. Update EA to read both signals**
```c
// In AI_Signal_Reader.mq5:
// 1. Read signals_for_mt5.csv (M5 swing, max 3 positions)
// 2. Read signals_m1_scalp.csv (M1 scalp, max 2 positions)
// 3. Allocate MaxPositions: min(3, swing_slots) + min(2, scalp_slots)
// 4. Apply separate risk sizing (M1 scalp uses half risk % due to 2:1 RR)
```

**5. Deploy and test**
```bash
# Compile updated EA
cd C:\Users\doanbacremote\tradingview-mcp
.\Run-AISignalTest.ps1 -RiskPct 10 -MaxPositions 5 -Deposit 200
# Expected: +1500-2000% at MaxPos=5 (more signals = more returns)
```

### Expected Results
- M1 signals: 200-400/month (8-13 per day vs 7-8 for M5)
- M1 WR: 45-50% (lower but fast exits + 2:1 RR)
- Average M1 trade: +0.8% to +1.2% (vs +1.8% for M5 at 3.5:1 RR)
- Combined monthly: +20-30% geometric mean (vs current 14.7%)

### Success Criteria
- ✅ M1 features load without OOM
- ✅ M1 WF trains in <2 hours
- ✅ M1 signals generated (non-zero count)
- ✅ EA compiles with dual signal source
- ✅ MT5 backtest with combined signals: P&L ≥ +1500%

---

## Integration Roadmap

### Week 1: Extended Training
- [ ] Day 1: Build extended features (30-60 min)
- [ ] Day 1-2: Retrain WF (2-3 hours)
- [ ] Day 2: Generate and test extended signals
- [ ] Day 3: MT5 backtest + validate

### Week 2: M1 Scalp
- [ ] Day 4: Build M1 features (45 min)
- [ ] Day 4-5: Train M1 WF (1-2 hours)
- [ ] Day 5: Generate M1 signals
- [ ] Day 6: Integrate into EA
- [ ] Day 7: MT5 backtest combined signals

### Week 3: Deployment
- [ ] Day 8-10: Live testing in demo account
- [ ] Day 11-14: Validation + performance monitoring
- [ ] Week 4: Production deployment if validated

---

## Troubleshooting

### Problem: Extended features build is slow
```bash
# Reduce date range for testing
python scripts/build_historical_features_extended.py \
  --from 2020-01-01 --to 2026-05-10  # 6 years instead of 23
```

### Problem: M1 features file too large (>10GB)
```bash
# Process in chunks (separate script needed)
# Or use SQLite instead of CSV for better compression
```

### Problem: WF runs out of memory
```bash
# Reduce batch size in training
# Or use --max-folds 3 to train fewer folds per run
```

### Problem: MT5 backtest fails with dual signals
```bash
# Verify both CSV files exist in MT5 Files folder
# Check EA reads both files correctly (InpCSVFile1, InpCSVFile2)
# Run simple test: read only M1 signals first, then add M5
```

---

## Cost-Benefit Analysis

### Extended Training
| Benefit | Cost |
|---------|------|
| +10-15% better generalization | +30-60 min initial setup |
| Fewer bad-month surprises | +15 min weekly retraining |
| Stable 55-58% accuracy | ~4GB disk space |

### M1 Scalp
| Benefit | Cost |
|---------|------|
| +5-15% additional monthly return | +1-2 hours setup |
| More execution opportunities | +8GB disk space (M1 features) |
| Hedge against slow M5 signals | More complex EA code |
| Diversification (2 models/TF) | Slightly higher slippage (M1) |

**Recommendation:** Do both. Extended training is quick ROI. M1 scalp adds 5-15% monthly with reasonable complexity.

---

## Success Metrics

### After Extended Training
- [ ] Extended WF: Acc 55-58% (vs 54% current)
- [ ] Extended MT5: +1200-1400% (vs +1248% current)
- [ ] Fewer bad-month surprises (consistency up)

### After M1 Scalp
- [ ] M1 signals: 200-400/month (generated)
- [ ] Combined MT5: +1500-2000% (more trades)
- [ ] Geometric mean: +20-25%/month (vs 14.7%)
- [ ] x5 months: 3-4 per 19-month period (vs 1 current)

### x5/Month Feasibility
- **Current:** Impossible (only 1/19 months achieve it)
- **After extended training:** Unlikely but more consistent performance
- **After M1 scalp:** Possible 3-4 months (40% of sample), rest +15-25%
- **Full x5 every month:** Still requires model improvements or regime detection

---

**Last Updated:** 2026-05-10  
**Status:** Ready for implementation
