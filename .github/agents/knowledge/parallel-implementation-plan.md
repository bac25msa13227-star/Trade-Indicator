# Parallel Implementation Plan — Week 1 Sprint

**Date**: 2026-05-05  
**Goal**: Deploy slippage model + Paper trading + Sharpe/Calmar metrics  
**Total Estimate**: 10 hours upfront work

---

## Feature 1: Deploy Slippage Model (2h)

### Objectives
- Merge `feature/slippage-model` branch to main
- Create comprehensive PR with comparison data
- Enable dynamic slippage in production configs
- Monitor backtest-live gap reduction (25% → 10-15%)

### Implementation Steps

#### 1.1 Pre-merge Validation ✅ DONE
- [x] Unit tests passing (16/16)
- [x] WF validation complete (41 folds, 2023-2026)
- [x] Comparison report generated
- [x] Trade HTML viewer created

#### 1.2 Merge & Deploy
```bash
# Switch to main and merge
git checkout main
git merge feature/slippage-model

# Push to remote
git push origin main

# Tag release
git tag -a v1.1.0-slippage -m "feat: Add dynamic slippage model"
git push origin v1.1.0-slippage
```

#### 1.3 Update Production Configs
**Files to modify**:
- `configs/live_acc1.yaml` → set `use_dynamic_slippage: true`
- `configs/live_acc2.yaml` → set `use_dynamic_slippage: true`

**Config changes**:
```yaml
risk:
  # ... existing config ...
  spread_cost_rr: 0.10
  slippage_rr: 0.05       # Static fallback
  commission_rr: 0.02
  use_dynamic_slippage: true  # ← ENABLE THIS
```

#### 1.4 Docker Rebuild & Deploy
```bash
# Rebuild containers with new code
docker compose build live live-acc1 live-acc2

# Restart services
docker compose up -d live live-acc1 live-acc2

# Verify logs
docker logs live-acc1 --tail 50
```

**Success Criteria**:
- Containers restart without errors
- Logs show "Dynamic slippage enabled"
- Live trading continues normally
- Friction_rr values vary by market conditions (not constant 0.05)

---

## Feature 2: Paper Trading Mode (4-6h)

### Objectives
- Shadow mode: generate signals, log decisions, NO real execution
- Compare paper P&L vs live P&L
- Validate slippage model accuracy (predicted vs actual)
- Enable zero-risk testing of new strategies

### Architecture

#### 2.1 New Execution Mode Enum
```python
# src/xauusd_ai/execution/mode.py (NEW FILE)
from enum import Enum

class ExecutionMode(Enum):
    LIVE = "live"           # Real execution
    PAPER = "paper"         # Shadow logging only
    BACKTEST = "backtest"   # Historical simulation
```

#### 2.2 Orchestrator Changes
**File**: `src/xauusd_ai/orchestrator.py`

```python
class XauUsdOrchestrator:
    def __init__(self, config, mode: ExecutionMode = ExecutionMode.LIVE):
        self.mode = mode
        # ... existing init ...
    
    def execute_signal(self, signal):
        """Execute or log signal based on mode."""
        if self.mode == ExecutionMode.PAPER:
            self._log_paper_trade(signal)
            return None  # No real execution
        elif self.mode == ExecutionMode.LIVE:
            return self.executor.place_order(signal)
```

#### 2.3 Paper Trading Logger
**File**: `src/xauusd_ai/execution/paper_logger.py` (NEW FILE)

```python
import json
from datetime import datetime
from pathlib import Path

class PaperTradeLogger:
    """Logs paper trades for comparison with live."""
    
    def __init__(self, output_path: str = "outputs/paper_trades.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(exist_ok=True)
    
    def log_signal(self, signal, predicted_slippage, market_data):
        """Log paper trade signal."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "side": signal.side,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "predicted_slippage_pips": predicted_slippage,
            "predicted_slippage_rr": self._calc_slippage_rr(signal, predicted_slippage),
            "atr": market_data.get("atr"),
            "spread": market_data.get("spread"),
            "volume_ratio": market_data.get("volume_ratio"),
            "session": market_data.get("session"),
            "probability": signal.probability,
            "risk_fraction": signal.risk_fraction,
        }
        
        with open(self.output_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def log_exit(self, trade_id, exit_price, actual_slippage_pips):
        """Log paper trade exit and actual slippage."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "trade_id": trade_id,
            "exit_price": exit_price,
            "actual_slippage_pips": actual_slippage_pips,
        }
        
        with open(self.output_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

#### 2.4 Comparison Dashboard
**File**: `scripts/compare_paper_vs_live.py` (NEW FILE)

Generates HTML report comparing:
- Paper P&L vs Live P&L
- Predicted slippage vs Actual slippage
- Signal accuracy (win rate, profit factor)
- Gap analysis (why paper != live)

#### 2.5 Config Integration
Add to `configs/acc1_v14pp_profit.yaml`:
```yaml
execution:
  mode: "paper"  # Options: live, paper, backtest
```

**Tests**:
- `tests/unit/test_paper_logger.py` — log formatting, file writes
- `tests/integration/test_paper_mode.py` — orchestrator runs without errors
- `tests/unit/test_execution_mode.py` — enum behavior

**Success Criteria**:
- Bot runs in paper mode without crashes
- `outputs/paper_trades.jsonl` accumulates entries
- No MT5 orders placed
- Can switch back to live mode seamlessly

---

## Feature 3: Sharpe/Calmar Metrics (3h)

### Objectives
- Add Sharpe ratio per fold → log to MLflow
- Add Calmar ratio (Return / Max DD)
- Track feature stability across folds
- Enable better WF validation decisions

### Implementation

#### 3.1 Metrics Module Enhancement
**File**: `src/xauusd_ai/infra/metrics.py`

```python
import numpy as np
from typing import List

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: List of daily returns (as fractions, e.g., 0.02 for 2%)
        risk_free_rate: Annual risk-free rate (default 0%)
    
    Returns:
        Sharpe ratio (annualized)
    """
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    # Annualize: sqrt(252 trading days)
    sharpe = (mean_excess / std_excess) * np.sqrt(252)
    return sharpe


def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """
    Calculate Calmar ratio (Return / Max Drawdown).
    
    Args:
        total_return: Total return as fraction (e.g., 1.5 for 150%)
        max_drawdown: Max drawdown as fraction (e.g., 0.15 for 15%)
    
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return 0.0
    
    return total_return / max_drawdown


def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio (only penalizes downside volatility).
    
    Args:
        returns: List of daily returns
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Sortino ratio (annualized)
    """
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / 252)
    
    mean_excess = np.mean(excess_returns)
    
    # Only downside returns
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 0.0
    
    downside_std = np.std(downside_returns, ddof=1)
    if downside_std == 0:
        return 0.0
    
    sortino = (mean_excess / downside_std) * np.sqrt(252)
    return sortino
```

#### 3.2 WF Script Integration
**File**: `scripts/walkforward_ict_wyckoff.py`

Add metrics calculation after each fold:
```python
# After fold backtest completes
fold_metrics = {
    "sharpe": calculate_sharpe_ratio(daily_returns),
    "calmar": calculate_calmar_ratio(total_return, max_dd),
    "sortino": calculate_sortino_ratio(daily_returns),
    "win_rate": wins / total_trades,
    "profit_factor": profit_factor,
}

# Log to MLflow
mlflow.log_metrics({
    f"fold_{fold_id}_sharpe": fold_metrics["sharpe"],
    f"fold_{fold_id}_calmar": fold_metrics["calmar"],
    f"fold_{fold_id}_sortino": fold_metrics["sortino"],
}, step=fold_id)
```

#### 3.3 Feature Stability Tracking
**File**: `src/xauusd_ai/monitoring/feature_stability.py` (NEW FILE)

```python
import pandas as pd
from typing import Dict, List

class FeatureStabilityTracker:
    """Tracks feature importance drift across WF folds."""
    
    def __init__(self):
        self.fold_importances: List[Dict[str, float]] = []
    
    def add_fold(self, fold_id: int, importances: Dict[str, float]):
        """Add feature importances from one fold."""
        self.fold_importances.append({
            "fold_id": fold_id,
            **importances
        })
    
    def get_stability_report(self) -> pd.DataFrame:
        """Generate stability report."""
        df = pd.DataFrame(self.fold_importances)
        df = df.set_index("fold_id")
        
        stability = {
            "mean_importance": df.mean(),
            "std_importance": df.std(),
            "cv": df.std() / df.mean(),  # Coefficient of variation
            "rank_correlation": self._rank_correlation(df),
        }
        
        return pd.DataFrame(stability)
    
    def _rank_correlation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate rank correlation across folds."""
        ranks = df.rank(axis=1, ascending=False)
        return ranks.std(axis=0)
```

**Tests**:
- `tests/unit/test_metrics.py` — Sharpe/Calmar/Sortino calculations
- `tests/unit/test_feature_stability.py` — Stability tracking logic
- `tests/integration/test_wf_metrics.py` — MLflow logging integration

**Success Criteria**:
- WF logs Sharpe/Calmar to MLflow
- Dashboard shows metrics per fold
- Feature stability report generated
- Metrics align with manual calculations

---

## Integration & Testing (1h)

### Integration Test Suite
**File**: `tests/integration/test_parallel_features.py` (NEW FILE)

```python
def test_paper_mode_with_metrics():
    """Test paper trading mode logs metrics correctly."""
    config = load_config("configs/acc1_v14pp_profit.yaml")
    config.execution.mode = "paper"
    
    orchestrator = XauUsdOrchestrator(config)
    orchestrator.run_one_cycle()
    
    # Verify paper trades logged
    assert Path("outputs/paper_trades.jsonl").exists()
    
    # Verify metrics calculated
    with open("outputs/paper_trades.jsonl") as f:
        trades = [json.loads(line) for line in f]
        assert len(trades) > 0
        assert "predicted_slippage_pips" in trades[0]


def test_dynamic_slippage_in_production():
    """Test dynamic slippage enabled in production config."""
    config = load_config("configs/live_acc1.yaml")
    assert config.risk.use_dynamic_slippage == True
```

### Deployment Checklist
- [ ] All unit tests passing (pytest)
- [ ] Integration tests passing
- [ ] Coverage ≥ 70% for new code
- [ ] Docker build successful
- [ ] Containers restart without errors
- [ ] Paper mode runs for 1 day without crashes
- [ ] Metrics appear in MLflow dashboard
- [ ] Live bot continues trading normally

---

## Timeline & Milestones

| Day | Task | Owner | Status |
|-----|------|-------|--------|
| Day 1 (Mon) | Merge slippage model, update configs | Agent | ⏳ In Progress |
| Day 1 (Mon) | Implement paper trading mode | Agent | ⏳ Next |
| Day 1 (Mon) | Add Sharpe/Calmar metrics | Agent | ⏳ Next |
| Day 2 (Tue) | Integration testing | Agent | 🔜 Pending |
| Day 2 (Tue) | Docker rebuild & deploy | Agent | 🔜 Pending |
| Day 3-4 | Monitor paper trading | User | 🔜 Pending |
| Week 2 | Analyze paper vs live gap | User | 🔜 Pending |
| Week 2 | Decision: enable dynamic slippage | User | 🔜 Pending |

---

## Risk Management

### Risks
1. **Dynamic slippage increases friction too much** → Live P&L drops
   - Mitigation: Paper trading validates first
   
2. **Paper mode has bugs** → Silent failures
   - Mitigation: Comprehensive logging, alerts on exceptions
   
3. **Metrics calculation errors** → Wrong MLflow data
   - Mitigation: Unit tests with known inputs/outputs

### Rollback Plan
If live bot fails after deployment:
```bash
# Revert to previous version
git revert HEAD
docker compose build live live-acc1 live-acc2
docker compose up -d live live-acc1 live-acc2

# Disable dynamic slippage
# Edit configs/live_acc*.yaml → use_dynamic_slippage: false
```

---

## Post-Deployment Monitoring

### Week 1 Metrics to Watch
1. **Backtest-live gap** (expect 25% → 10-15% reduction)
2. **Paper vs Live P&L** (should align within 10%)
3. **Sharpe ratio** (expect ≥ 1.5)
4. **Slippage accuracy** (predicted vs actual within 1 pip)

### Dashboard URLs
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Paper trading report: `outputs/paper_vs_live_report.html`

---

## Success Criteria (End of Week 1)

✅ **Deploy slippage model**:
- [ ] Merged to main
- [ ] Production configs updated
- [ ] Docker deployed
- [ ] Live bot running with dynamic slippage

✅ **Paper trading**:
- [ ] Mode implemented
- [ ] Logging working
- [ ] 1+ days of paper trades collected
- [ ] Comparison report generated

✅ **Metrics**:
- [ ] Sharpe/Calmar calculated per fold
- [ ] Feature stability tracked
- [ ] MLflow dashboard showing metrics
- [ ] Metrics validate model quality

**Overall Success**: All 3 features deployed, tested, and monitoring active by end of Week 1.
