# Lộ Trình Phát Triển (May 2026)

**Ngày cập nhật:** May 2026  
**Tình trạng hiện tại:** Paper trading đang chạy validation 7 ngày trên production

---

## 📊 Tình Trạng Tính Năng

**Hoàn thành (6/18):**
- ✅ Dynamic Slippage Model (ATR + session + volume)
- ✅ Paper Trading Mode (shadow execution)
- ✅ A/B Testing Framework (statistical analysis)
- ✅ Sharpe/Sortino/Calmar Metrics
- ✅ Advanced Metrics Integration (WF script)
- ✅ Feature Engineering (spread_points, atr_ratio, atr_mean)

**Đang chạy:**
- 🔄 Paper Mode Validation (7 days) — Ngày 1-6 hiện tại
  - Check daily: logs, trades count, no crashes
  - Ngày 7: Review slippage accuracy >80%, win rate 35-45%, min 50 trades

**Chưa làm (12/18):**
- 🎯 **P0 (Quick Wins, 2-3h):** Turnover-Adjusted Return
- 🎯 **P1 (1-2 days):** Regime Detection HMM
- 🎯 **P2 (3-5 days):** Ensemble (LightGBM + LSTM)
- 🎯 **P3 (1 week):** RL Fine-tuning (PPO/SAC)
- **P4 (Optional):** Market Microstructure, Partial Fill, Latency, Tick Replay

---

## 📅 Roadmap 4 Tuần

### **Week 1 (Current)** — Paper Validation
**Ngày 1-6:** Monitor paper trading
- Check containers: `docker ps` — acc1, acc2 running
- Check logs: `docker logs live-acc1 | tail -50`
- Check trades: `wc -l outputs/paper_trades_acc1.jsonl` (expect 5-15 trades/day)

**Ngày 7 (Decision Gate):** Review Results
```python
# Check slippage accuracy
python scripts/analyze_paper_results.py

# Criteria:
# - Slippage accuracy >= 80%
# - Win rate: 35-45%
# - No systematic bias (control group < 5% P&L deviation from WF)
# - Min 50 trades total
```

**Decision:**
- ✅ **PASS** → Week 2 Option A (deploy dynamic slippage)
- ❌ **FAIL** → Week 2 Option B (recalibrate + re-run)

---

### **Week 2** — Deploy or Recalibrate + Quick Win

#### **Option A: Paper Passed**
1. **Deploy Dynamic Slippage (30 min)**
   ```yaml
   # configs/live_acc1.yaml
   risk:
     use_dynamic_slippage: true  # ← ENABLE
     ab_test_enabled: false
   ```
   - Push to production: `git add . && git commit -m "feat: enable dynamic slippage" && git push`
   - Restart containers: `docker compose restart live-acc1 live-acc2`

2. **Monitor 48h**
   - Check backtest-live gap: expect 25% → 10-15%
   - Review `outputs/live_closed_trades_acc1.csv`

3. **Implement Turnover-Adjusted Return (2-3h)** ← Start parallel

#### **Option B: Paper Failed**
1. **Recalibrate Slippage (2h)**
   - Adjust session multipliers in `src/xauusd_ai/backtesting/slippage.py`
   - Example: Asian 1.5× → 1.3×, London 1.0× → 0.8×
   - Re-run paper 7 days

2. **Implement Turnover-Adjusted Return (2-3h)** ← Start parallel anyway

---

### **Week 3** — Regime Detection HMM

**Objective:** Adaptive strategy — reduce losses in choppy markets, increase sizing in trending

**Implementation (1-2 days):**

```python
# src/xauusd_ai/features/regime.py
from hmmlearn.hmm import GaussianHMM
import numpy as np

class RegimeDetector:
    def __init__(self, n_states=3):
        """
        n_states = 3:
          0 = Low Volatility (range-bound, choppy)
          1 = Medium Volatility (trending)
          2 = High Volatility (news spike, crash)
        """
        self.model = GaussianHMM(n_components=n_states, covariance_type="full")
        self.fitted = False
    
    def fit(self, returns: np.ndarray, volatility: np.ndarray):
        """
        Fit HMM on historical data.
        
        Args:
            returns: 1D array of log returns
            volatility: 1D array of rolling ATR or std
        """
        X = np.column_stack([returns, volatility])
        self.model.fit(X)
        self.fitted = True
    
    def predict_regime(self, current_returns, current_vol):
        """
        Predict current regime.
        
        Returns:
            regime: int 0, 1, or 2
        """
        X = np.array([[current_returns, current_vol]])
        regime = self.model.predict(X)[0]
        return regime
    
    def get_strategy_params(self, regime):
        """
        Return adaptive parameters per regime.
        
        Returns:
            dict: {risk_pct, min_rr, max_holding_bars}
        """
        if regime == 0:  # Low Vol (range)
            return {"risk_pct": 0.02, "min_rr": 1.5, "max_holding_bars": 100}
        elif regime == 1:  # Medium Vol (trending)
            return {"risk_pct": 0.04, "min_rr": 1.2, "max_holding_bars": 200}
        else:  # High Vol (volatile)
            return {"risk_pct": 0.01, "min_rr": 2.0, "max_holding_bars": 50}
```

**Integration (1h):**
1. Add `regime` column to `src/xauusd_ai/features/dataset.py`
2. Fit detector in `scripts/walkforward_ict_wyckoff.py` before each fold
3. Use `get_strategy_params(regime)` in `orchestrator.py` to adjust risk_pct

**Value:**
- Reduce drawdown 10-20% in sideways markets
- Increase Sharpe by ~0.3-0.5 points

---

### **Week 4** — A/B Testing in Production

**Enable A/B Testing:**
```yaml
# configs/live_acc1.yaml (after paper validation passes)
risk:
  use_dynamic_slippage: true   # Already enabled in Week 2
  ab_test_enabled: true        # ← ENABLE NOW
  ab_test_log_file: "outputs/ab_test_results_live.jsonl"
```

**Run 2-4 weeks:**
- Control: static slippage 0.5 pips
- Treatment: dynamic slippage (ATR + session + volume)
- Collect >= 50 trades per group (10-15 trades/day × 7 days = ~100 total)

**Analysis (after 2-4 weeks):**
```python
from xauusd_ai.infra.ab_testing import ABTestManager

ab = ABTestManager("outputs/ab_test_results_live.jsonl", seed=42)
analysis = ab.analyze()

print(f"Control mean P&L: {analysis['control_mean_pnl']:.2f}")
print(f"Treatment mean P&L: {analysis['treatment_mean_pnl']:.2f}")
print(f"p-value: {analysis['p_value']:.4f}")
print(f"Effect size: {analysis['effect_size']:.2f}")
print(f"Interpretation: {analysis['interpretation']}")

# Decision:
# p-value < 0.05 AND mean_diff > 0 → Deploy treatment winner
# p-value >= 0.05 → No significant difference, keep control (simpler)
```

---

## 🎯 Next Priority: Turnover-Adjusted Return (P0)

**Effort:** 2-3 hours  
**Impact:** High — quantifies actual costs ignored in current P&L  
**Can start now:** Independent of paper validation

### **Why This Matters:**
Current P&L ignores:
- Spread cost: 0.5 pips × 10,825 trades = **5,412 pips lost** (~$54k on 1.0 lot)
- Swap/rollover: holding overnight incurs interest
- True profitability = gross P&L - spread - swap

### **Implementation:**

```python
# src/xauusd_ai/infra/advanced_metrics.py
def calculate_turnover_adjusted_return(
    trades: pd.DataFrame,
    spread_pips: float = 0.5,
    swap_per_lot_per_day: float = 0.15,  # XAUUSD typical swap
    pip_value: float = 10.0  # $10 per pip for 1.0 lot
) -> dict:
    """
    Calculate net return after spread and swap costs.
    
    Args:
        trades: DataFrame with columns [pnl, lot_size, holding_bars]
        spread_pips: Average bid-ask spread in pips
        swap_per_lot_per_day: Overnight financing cost per lot
        pip_value: $ per pip (10 for XAUUSD at 1.0 lot)
    
    Returns:
        dict with gross_pnl, spread_cost, swap_cost, net_pnl, turnover_drag
    """
    gross_pnl = trades['pnl'].sum()
    
    # Spread cost (entry + exit = 2× spread per trade)
    spread_cost = sum(
        2 * spread_pips * pip_value * row['lot_size'] 
        for _, row in trades.iterrows()
    )
    
    # Swap cost (holding overnight)
    swap_cost = sum(
        swap_per_lot_per_day * row['lot_size'] * (row['holding_bars'] // 1440)  # 1440 mins = 1 day
        for _, row in trades.iterrows()
    )
    
    net_pnl = gross_pnl - spread_cost - swap_cost
    turnover_drag = (spread_cost + swap_cost) / gross_pnl if gross_pnl > 0 else 0.0
    
    return {
        "gross_pnl": gross_pnl,
        "spread_cost": spread_cost,
        "swap_cost": swap_cost,
        "net_pnl": net_pnl,
        "turnover_drag": turnover_drag,
        "net_return_pct": net_pnl / trades['balance'].iloc[0] if len(trades) > 0 else 0.0
    }
```

### **Integration (30 min):**

```python
# scripts/walkforward_ict_wyckoff.py (add to concurrent simulation)
from xauusd_ai.infra.advanced_metrics import calculate_turnover_adjusted_return

# After backtest finishes
turnover_metrics = calculate_turnover_adjusted_return(
    trades_df,
    spread_pips=0.5,
    swap_per_lot_per_day=0.15
)

concurrent_sim.update({
    "gross_pnl": turnover_metrics["gross_pnl"],
    "net_pnl": turnover_metrics["net_pnl"],
    "spread_cost": turnover_metrics["spread_cost"],
    "swap_cost": turnover_metrics["swap_cost"],
    "turnover_drag": turnover_metrics["turnover_drag"]
})

# Log to MLflow
mlflow.log_metric("net_pnl", turnover_metrics["net_pnl"], step=fold_id)
mlflow.log_metric("turnover_drag", turnover_metrics["turnover_drag"], step=fold_id)
```

### **Testing (30 min):**

```python
# tests/infra/test_advanced_metrics.py
def test_turnover_adjusted_return():
    trades = pd.DataFrame({
        'pnl': [50, -20, 30],  # Gross $60
        'lot_size': [1.0, 1.0, 1.0],
        'holding_bars': [100, 50, 1440],  # Last trade holds 1 day
        'balance': [10000] * 3
    })
    
    result = calculate_turnover_adjusted_return(
        trades,
        spread_pips=0.5,
        swap_per_lot_per_day=0.15
    )
    
    # Spread cost: 3 trades × 2× × 0.5 pips × $10 = $30
    # Swap cost: 1 trade × 1 day × 1.0 lot × $0.15 = $0.15
    # Net: $60 - $30 - $0.15 = $29.85
    
    assert result["gross_pnl"] == 60
    assert abs(result["spread_cost"] - 30) < 0.01
    assert abs(result["swap_cost"] - 0.15) < 0.01
    assert abs(result["net_pnl"] - 29.85) < 0.01
    assert abs(result["turnover_drag"] - 0.5025) < 0.01  # 50.25%
```

**Value:**
- Realistic P&L: Sharpe 3.7 may drop to ~2.5-3.0 after costs (still excellent!)
- Better decision-making: Shows true profitability after friction
- MLflow tracking: Monitor turnover drag over time

---

## 🧪 Workflow TDD cho Turnover Metrics

1. **[@tdd-guide] Write tests first (RED)**
   ```bash
   pytest tests/infra/test_advanced_metrics.py::test_turnover_adjusted_return -v
   # Should FAIL initially
   ```

2. **[@XAUUSD AI Dev] Implement function (GREEN)**
   - Add `calculate_turnover_adjusted_return()` to `src/xauusd_ai/infra/advanced_metrics.py`
   - Run test again — should PASS

3. **[@code-reviewer] Review code quality**
   - Check naming, docstrings, edge cases (zero trades, negative P&L)

4. **[@XAUUSD AI Dev] Integrate with WF script**
   - Add to `scripts/walkforward_ict_wyckoff.py`
   - Run 1-fold WF to verify metrics log to MLflow

5. **[@XAUUSD AI Dev] Deploy**
   - Commit: `git add -A && git commit -m "feat: turnover-adjusted return metrics"`
   - Push: `git push origin main`

**Timeline:** 2-3 hours total

---

## 🔮 Future Priorities (After Week 4)

### **P2: Ensemble Model (3-5 days)**
- Combine LightGBM + LSTM + statistical features
- Voting or meta-learner approach
- Expected Sharpe improvement: +0.5-1.0

### **P3: RL Fine-tuning (1 week)**
- PPO/SAC for dynamic sizing and exit timing
- Reward shaping: Sharpe ratio + drawdown penalty
- Expected improvement: +15-20% return, -5% drawdown

### **P4: Optional Enhancements**
- Market microstructure (tick speed, order book depth)
- Partial fill simulation
- Latency injection (50-200ms)
- Tick replay for sub-M1 backtesting

---

## 📈 Success Metrics

**Paper Validation (Week 1):**
- ✅ Slippage accuracy >= 80%
- ✅ Win rate: 35-45%
- ✅ No systematic bias
- ✅ Minimum 50 trades

**Dynamic Slippage Deployment (Week 2):**
- ✅ Backtest-live gap: 25% → 10-15%
- ✅ No crashes or errors
- ✅ Sharpe >= 1.5 after costs

**Regime Detection (Week 3):**
- ✅ Drawdown reduction: 10-20%
- ✅ Sharpe improvement: +0.3-0.5
- ✅ Adaptive risk_pct working correctly

**A/B Testing (Week 4):**
- ✅ >= 50 trades per group
- ✅ p-value < 0.05 for winner
- ✅ Effect size (Cohen's d) > 0.2

---

## ❓ FAQ

**Q: Tại sao chưa enable dynamic slippage trong production?**
A: Đang chạy paper validation 7 ngày để verify slippage accuracy trước. Safety first!

**Q: Tại sao Turnover-Adjusted Return là P0?**
A: Quick win (2-3h), high impact (shows true profitability), independent (không cần đợi paper validation).

**Q: Regime Detection vs Ensemble — cái nào trước?**
A: Regime Detection (1-2 days) trước vì đơn giản hơn và impact nhanh hơn. Ensemble cần 3-5 days.

**Q: Khi nào deploy RL Fine-tuning?**
A: Sau khi có Ensemble + Regime working stably. RL là high-risk, high-reward — cần foundation solid trước.

**Q: A/B testing chạy bao lâu?**
A: Minimum 2 weeks (>50 trades per group). Ideal 4 weeks (>100 trades per group) để có statistical power cao.

---

## 🚀 Bắt đầu ngay

**Immediate Action (hôm nay):**
```bash
# 1. Implement Turnover-Adjusted Return (2-3h)
cd ~/Documents/Thạc\ sĩ\ MSE/Trade\ Indicator
source .venv/bin/activate

# 2. Write test first (TDD)
touch tests/infra/test_turnover_adjusted_return.py
# Add test case

# 3. Run test (should FAIL)
pytest tests/infra/test_turnover_adjusted_return.py -v

# 4. Implement function
# Edit src/xauusd_ai/infra/advanced_metrics.py

# 5. Test again (should PASS)
pytest tests/infra/test_turnover_adjusted_return.py -v

# 6. Integrate with WF script
# Edit scripts/walkforward_ict_wyckoff.py

# 7. Commit & push
git add -A
git commit -m "feat: add turnover-adjusted return metrics"
git push origin main
```

**Daily Check (5 min/day):**
```bash
# Check paper trading status
docker ps | grep live
docker logs live-acc1 | tail -20
wc -l outputs/paper_trades_acc1.jsonl
```

**Week 1 Day 7 Review:**
```bash
# Analyze paper results
python scripts/analyze_paper_results.py

# Decision: Enable dynamic slippage OR recalibrate
```
