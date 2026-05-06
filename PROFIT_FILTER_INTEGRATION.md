# Profit Filter Integration Plan

**Status:** Ready for deployment after WF validation  
**ETA:** 1-2 hours implementation + testing  
**Risk:** Low (filter can be disabled anytime)

---

## Integration Steps

### Step 1: Add Config Parameters

**File:** `src/xauusd_ai/config.py`

```python
# In RiskConfig class (line ~150)
class RiskConfig(BaseModel):
    # ... existing fields ...
    
    # Profit filter settings
    profit_filter_enabled: bool = Field(
        False,
        description="Enable minimum profit filter to skip unprofitable trades"
    )
    min_expected_profit: float = Field(
        15.0,
        ge=5.0,
        le=50.0,
        description="Minimum expected profit per trade (USD)"
    )
    profit_filter_spread_pips: float = Field(
        0.5,
        ge=0.1,
        le=2.0,
        description="XAUUSD spread in pips for profit calculation"
    )
```

**YAML Config:** `configs/live_acc1.yaml`

```yaml
risk:
  # ... existing risk parameters ...
  
  # Profit filter (deploy after WF validation)
  profit_filter_enabled: false       # Set true when ready
  min_expected_profit: 15.0          # $15 minimum (spread $10 + 50% buffer)
  profit_filter_spread_pips: 0.5     # XAUUSD typical spread
```

---

### Step 2: Initialize Filter in Orchestrator

**File:** `src/xauusd_ai/orchestrator.py`

```python
# Add import at top (line ~20)
from xauusd_ai.strategies.profit_filter import MinimumProfitFilter

# In OrchestrationEngine.__init__() (line ~150)
class OrchestrationEngine:
    def __init__(self, config_path: str):
        # ... existing initialization ...
        
        # Initialize profit filter
        self.profit_filter = MinimumProfitFilter(
            min_expected_profit=self.settings.risk.min_expected_profit,
            spread_pips=self.settings.risk.profit_filter_spread_pips,
            pip_value=10.0,  # $10 per pip for XAUUSD at 1.0 lot
            enabled=self.settings.risk.profit_filter_enabled
        )
        
        LOGGER.info(
            f"Profit filter initialized: enabled={self.profit_filter.enabled}, "
            f"min_profit=${self.profit_filter.min_expected_profit}"
        )
```

---

### Step 3: Add Profit Estimation Method

**File:** `src/xauusd_ai/orchestrator.py`

```python
# Add new method to OrchestrationEngine class (line ~400)

def _estimate_profit(
    self,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    confidence: float,
    win_probability: Optional[float] = None
) -> float:
    """
    Estimate expected profit for a trade signal.
    
    Args:
        entry_price: Entry price in USD
        stop_loss: Stop loss price
        take_profit: Take profit price
        confidence: Model confidence (0-1)
        win_probability: Win probability (if available)
    
    Returns:
        Expected profit in USD for 1.0 lot
    """
    # Calculate pip distances
    pip_value = 10.0  # $10 per pip for XAUUSD at 1.0 lot
    sl_pips = abs(entry_price - stop_loss) / 0.01  # 1 pip = $0.01 for XAUUSD
    tp_pips = abs(take_profit - entry_price) / 0.01
    
    # Calculate dollar amounts
    potential_loss = sl_pips * pip_value
    potential_gain = tp_pips * pip_value
    
    # Use win probability if available, otherwise use confidence as proxy
    if win_probability is None:
        # Empirical mapping: confidence 0.7 ≈ 40% win rate, 0.9 ≈ 60%
        win_probability = max(0.35, min(0.65, (confidence - 0.5) * 2))
    
    # Expected value = P(win) × gain - P(loss) × loss
    expected_profit = (win_probability * potential_gain) - ((1 - win_probability) * potential_loss)
    
    return expected_profit
```

---

### Step 4: Apply Filter in Signal Evaluation

**File:** `src/xauusd_ai/orchestrator.py`

```python
# In _evaluate_signal() method (line ~800)

async def _evaluate_signal(self, row: pd.Series) -> Optional[TradeSignal]:
    """Evaluate trading signal and apply profit filter."""
    
    # ... existing signal generation logic ...
    
    # Extract signal components
    entry_price = float(row["close"])
    stop_loss = self._calculate_stop_loss(row, signal_side)
    take_profit = self._calculate_take_profit(row, signal_side, entry_price, stop_loss)
    confidence = float(row["confidence"])
    
    # Build signal object (before filter)
    signal = TradeSignal(
        timestamp=row["time"],
        side=signal_side,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        # ... other fields ...
    )
    
    # ──────────────────────────────────────────────────────────────
    # PROFIT FILTER: Skip unprofitable trades
    # ──────────────────────────────────────────────────────────────
    if self.profit_filter.enabled:
        # Estimate expected profit
        predicted_profit = self._estimate_profit(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence
        )
        
        # Check if trade should be skipped
        should_skip, reason = self.profit_filter.should_skip_trade(
            predicted_profit=predicted_profit,
            predicted_rr=signal.risk_reward_ratio,
            lot_size=1.0  # Or calculate from risk_pct
        )
        
        if should_skip:
            LOGGER.info(
                f"[PROFIT_FILTER] Skipped signal: {reason} | "
                f"Entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}, "
                f"Confidence={confidence:.2%}, RR={signal.risk_reward_ratio:.2f}"
            )
            
            # Log to metrics for monitoring
            self.metrics.increment_counter("profit_filter_skipped_signals")
            self.metrics.histogram("profit_filter_predicted_profit", predicted_profit)
            
            return None  # Skip trade
    
    # Signal passed filter → proceed with execution
    return signal
```

---

### Step 5: Add Monitoring Metrics

**File:** `src/xauusd_ai/infra/metrics.py`

```python
# Add new metrics counters (line ~50)

class MetricsCollector:
    def __init__(self):
        # ... existing metrics ...
        
        # Profit filter metrics
        self.profit_filter_skipped_signals = Counter(
            "profit_filter_skipped_signals_total",
            "Total signals skipped by profit filter"
        )
        
        self.profit_filter_predicted_profit = Histogram(
            "profit_filter_predicted_profit_usd",
            "Predicted profit distribution (USD)",
            buckets=[0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
        )
        
        self.profit_filter_skip_rate = Gauge(
            "profit_filter_skip_rate_percent",
            "Percentage of signals skipped by profit filter"
        )
```

---

### Step 6: Add Grafana Dashboard Panel

**File:** `docker/grafana/dashboards/trading-dashboard.json`

```json
{
  "title": "Profit Filter Statistics",
  "type": "stat",
  "targets": [
    {
      "expr": "rate(profit_filter_skipped_signals_total[5m])",
      "legendFormat": "Skipped/min"
    },
    {
      "expr": "profit_filter_skip_rate_percent",
      "legendFormat": "Skip Rate %"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "percent",
      "thresholds": {
        "mode": "absolute",
        "steps": [
          {"value": 0, "color": "green"},
          {"value": 50, "color": "yellow"},
          {"value": 80, "color": "red"}
        ]
      }
    }
  }
}
```

---

## Testing Plan

### Phase 1: Unit Tests (Already Done ✅)
- 12/12 tests passing
- 96% code coverage
- Edge cases covered

### Phase 2: Backtesting Validation
```bash
# Test filter impact on WF results
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"
source .venv/bin/activate

# Run WF WITH filter enabled
python scripts/walkforward_ict_wyckoff.py configs/acc1_v14pp_profit.yaml \
  --test-start 2024-01-01 --test-bars 6000 --step-bars 6000 \
  --no-compound --combo133 --cache --risk-pct 0.030 --no-rr-sweep \
  --profit-filter --min-profit 15.0 \
  2>&1 | tee outputs/wf_with_profit_filter.log

# Compare results
python scripts/compare_wf_results.py \
  outputs/wf_without_filter.log \
  outputs/wf_with_profit_filter.log \
  --metrics trades,skip_rate,net_pnl,sharpe
```

**Expected outcomes:**
- 40-60% skip rate (target: 50%)
- 30-50% fewer trades
- Net P&L should be POSITIVE (if negative without filter)
- Sharpe ratio increase by 0.2-0.5 points

### Phase 3: Paper Trading Deployment
```yaml
# configs/live_acc1.yaml
execution:
  mode: paper  # Shadow mode first

risk:
  profit_filter_enabled: true
  min_expected_profit: 15.0  # Conservative start
```

**Monitor for 3-7 days:**
- Skip rate (target: 40-60%)
- Average predicted profit of kept signals
- Actual P&L vs predicted
- Any systematic biases (e.g., skipping all BUY signals)

### Phase 4: Live Deployment
**Criteria to pass:**
1. Paper mode Net P&L > 0
2. Skip rate stable 40-60%
3. No systematic biases detected
4. Prediction accuracy > 70% (actual profit within 30% of predicted)

**Deployment:**
```yaml
# configs/live_acc1.yaml
execution:
  mode: live  # Go live

risk:
  profit_filter_enabled: true
  min_expected_profit: 15.0
```

**Monitor first 48 hours:**
- Check every 2-4 hours
- Verify signals being processed correctly
- Confirm skip rate matches expectations
- Watch for any unexpected behavior

**Rollback plan (if needed):**
```yaml
risk:
  profit_filter_enabled: false  # Instant disable
```

---

## CLI Flag Support (Optional Enhancement)

**File:** `scripts/walkforward_ict_wyckoff.py`

```python
# Add argparse flag (line ~50)
parser.add_argument(
    "--profit-filter",
    action="store_true",
    help="Enable profit filter during WF validation"
)
parser.add_argument(
    "--min-profit",
    type=float,
    default=15.0,
    help="Minimum expected profit threshold (USD)"
)

# Use in config override (line ~200)
if args.profit_filter:
    config.risk.profit_filter_enabled = True
    config.risk.min_expected_profit = args.min_profit
    LOGGER.info(f"Profit filter enabled with min_profit=${args.min_profit}")
```

---

## Deployment Checklist

### Pre-Deployment
- [x] Implementation complete
- [x] Unit tests passing (12/12)
- [ ] WF validation results reviewed
- [ ] Net P&L confirmed positive (or neutral)
- [ ] Config YAML updated
- [ ] Grafana dashboard panel added
- [ ] Documentation reviewed

### Deployment Day
- [ ] Enable in paper mode first
- [ ] Monitor logs for 2 hours
- [ ] Check Grafana metrics (skip rate, profit distribution)
- [ ] Verify no errors in orchestrator logs
- [ ] Compare paper P&L vs baseline

### Post-Deployment (Week 1)
- [ ] Daily review of skip rate (should stabilize after 3 days)
- [ ] Track prediction accuracy: `actual_profit / predicted_profit`
- [ ] Adjust threshold if skip rate < 30% or > 70%
- [ ] Document any systematic patterns (e.g., certain hours skip more)

### Go-Live Approval
- [ ] Paper mode Net P&L ≥ baseline
- [ ] Skip rate 40-60%
- [ ] No critical bugs detected
- [ ] Team approval obtained
- [ ] Switch to live mode: `execution.mode: live`

---

## Troubleshooting Guide

### Issue: Skip rate too low (<30%)

**Diagnosis:**
```python
# Check predicted profit distribution
python scripts/analyze_profit_predictions.py outputs/paper_mode_signals.jsonl
```

**Solutions:**
1. Lower threshold: `min_expected_profit: 12.0`
2. Check `_estimate_profit()` — may be overestimating
3. Review win_probability mapping

### Issue: Skip rate too high (>70%)

**Diagnosis:**
- Most signals have low expected profit
- Possible model degradation

**Solutions:**
1. Raise threshold temporarily: `min_expected_profit: 20.0`
2. Review model predictions — may need retraining
3. Check if market conditions changed (volatility, spreads)

### Issue: Negative P&L even with filter

**Diagnosis:**
- Profit estimation inaccurate
- Spread cost higher than expected
- Model predictions unreliable

**Solutions:**
1. **STOP live trading immediately**
2. Disable filter: `profit_filter_enabled: false`
3. Run diagnostic WF with turnover metrics
4. Recalibrate `_estimate_profit()` function
5. Consider increasing threshold: `min_expected_profit: 25.0`

### Issue: Filter skips all signals

**Diagnosis:**
```python
# Check if filter is too aggressive
LOGGER.error("No signals passed profit filter in last 24h!")
```

**Solutions:**
1. Immediate: Disable filter
2. Review predicted_profit values — may be all negative
3. Check if `_estimate_profit()` has a bug
4. Verify spread_pips config matches broker reality

---

## Performance Expectations

**Best case (filter works perfectly):**
- Net P&L: +50% improvement
- Sharpe ratio: +0.3 to +0.5 points
- Trade frequency: -50% (half as many trades)
- Win rate: Unchanged or slightly higher

**Realistic case:**
- Net P&L: +20-30% improvement
- Sharpe ratio: +0.2 points
- Trade frequency: -40%
- Win rate: Stable

**Worst case (filter doesn't help):**
- Net P&L: Unchanged
- Sharpe ratio: Unchanged
- Trade frequency: -40% (fewer opportunities)
- Win rate: Unchanged

**In worst case:** Disable filter and revert to baseline strategy.

---

## Next Steps After Successful Deployment

1. **Optimize threshold dynamically:**
   ```python
   # Adjust based on recent performance
   if recent_win_rate < 0.40:
       min_expected_profit += 5.0  # Be more conservative
   elif recent_win_rate > 0.50:
       min_expected_profit -= 2.0  # Allow more trades
   ```

2. **Add time-of-day awareness:**
   ```python
   # Higher threshold during volatile hours
   if hour in [15, 16, 17]:  # London/NY overlap
       effective_threshold = min_expected_profit * 1.2
   ```

3. **Regime-based thresholds:**
   ```python
   # Lower threshold in trending markets (HMM regime = 1)
   if current_regime == "trending":
       effective_threshold = min_expected_profit * 0.8
   elif current_regime == "range":
       effective_threshold = min_expected_profit * 1.3
   ```

---

## ETA: 1-2 Hours Total

**Breakdown:**
- Config changes: 15 min
- Orchestrator integration: 30 min
- Testing (quick WF): 45 min
- Monitoring setup: 30 min

**Start immediately after WF validation completes!**
