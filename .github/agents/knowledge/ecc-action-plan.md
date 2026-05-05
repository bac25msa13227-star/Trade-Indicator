# ECC Integration Action Plan — XAUUSD AI Trading
**Version**: 1.0 — Optimized for ML Research + Production Deployment
**Timeline**: 2 weeks (10 working days)
**Focus**: Advanced ML techniques + 70% test coverage + Security hardening

---

## 🔍 Audit Results

### Current State
- **Test files**: 24 (68.5% file ratio vs 35 source files)
- **Credentials**: ⚠️ HARDCODED in `configs/live_acc1.yaml` and `live_acc2.yaml`
  - MT5 passwords: `07032001bB@` (both accounts)
  - Logins: 270832477 (acc1), 433326057 (acc2)
- **Orchestration**: ✅ Docker Compose already configured (postgres, minio, mlflow, prometheus, grafana, airflow)
- **Infrastructure**: ✅ Ready for production (MLflow tracking, Prometheus metrics, Grafana dashboards)

### Security Risk Assessment
🔴 **CRITICAL**: Production credentials exposed in version control
- MT5 live trading accounts accessible to anyone with repo access
- Risk: Account takeover, unauthorized trading, capital loss
- **ACTION REQUIRED**: Immediate credential rotation + refactor to env vars

---

## 🎯 Implementation Plan

### Phase 0: EMERGENCY Security Patch (Day 0 — 2 hours)

**MUST DO FIRST** before any other work:

```bash
# 1. Rotate credentials IMMEDIATELY
# Login to MT5 broker portal → change passwords for both accounts

# 2. Remove from git history
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch configs/live_acc1.yaml configs/live_acc2.yaml" \
  --prune-empty --tag-name-filter cat -- --all

# 3. Create .env template
cat > .env.example << 'EOF'
# MT5 Credentials
MT5_LOGIN_ACC1=
MT5_PASSWORD_ACC1=
MT5_SERVER_ACC1=
MT5_LOGIN_ACC2=
MT5_PASSWORD_ACC2=
MT5_SERVER_ACC2=

# Telegram
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID_ACC1=
TELEGRAM_CHAT_ID_ACC2=

# Database
POSTGRES_USER=trader
POSTGRES_PASSWORD=
POSTGRES_DB=tradedb

# MLflow
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=

# Grafana
GRAFANA_USER=admin
GRAFANA_PASSWORD=
EOF

# 4. Create actual .env (gitignored)
cp .env.example .env
# Fill in real values

# 5. Update .gitignore
echo ".env" >> .gitignore
echo "configs/live_*.yaml" >> .gitignore  # Don't track live configs

# 6. Commit cleanup
git add .gitignore .env.example
git commit -m "security: remove hardcoded credentials, add .env pattern"
```

**Deliverable**: Credentials rotated, removed from git, moved to .env

---

### Phase 1: ECC Foundation + Test Infrastructure (Days 1-2)

#### 1.1 Install ECC Core (Manual Selective)
```bash
cd "$HOME/Documents/Thạc sĩ MSE"

# Already cloned: everything-claude-code/
cd everything-claude-code/everything-claude-code

# Manual copy to Trade Indicator
DEST="$HOME/Documents/Thạc sĩ MSE/Trade Indicator"

# Copy 5 essential agents
mkdir -p "$DEST/.github/agents"
cp -r .agents/planner.agent.md "$DEST/.github/agents/"
cp -r .agents/tdd-guide.agent.md "$DEST/.github/agents/"
cp -r .agents/code-reviewer.agent.md "$DEST/.github/agents/"
cp -r .agents/security-reviewer.agent.md "$DEST/.github/agents/"
cp -r .agents/build-error-resolver.agent.md "$DEST/.github/agents/"

# Copy 5 essential skills
mkdir -p "$DEST/.github/agents/skills"
cp -r .agents/skills/tdd-workflow "$DEST/.github/agents/skills/"
cp -r .agents/skills/verification-loop "$DEST/.github/agents/skills/"
cp -r .agents/skills/eval-harness "$DEST/.github/agents/skills/"
cp -r .agents/skills/security-review "$DEST/.github/agents/skills/"
cp -r .agents/skills/strategic-compact "$DEST/.github/agents/skills/"

# Copy Python rules
mkdir -p "$DEST/.github/rules"
cp -r rules/python "$DEST/.github/rules/"
cp -r rules/common "$DEST/.github/rules/"

echo "ECC core installed to Trade Indicator"
```

#### 1.2 Setup Test Infrastructure
```bash
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"

# Install test dependencies
cat > requirements-test.txt << 'EOF'
# Testing
pytest>=8.0.0
pytest-cov>=4.1.0
pytest-asyncio>=0.23.0
pytest-mock>=3.12.0
pytest-xdist>=3.5.0  # Parallel testing

# Type checking
pyright>=1.1.350

# Linting
ruff>=0.2.0

# Test utilities
httpx>=0.26.0  # Async HTTP for API tests
faker>=22.0.0  # Fake data generation
freezegun>=1.4.0  # Time mocking
EOF

source .venv/bin/activate
pip install -r requirements-test.txt

# Create pytest.ini
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --cov=src/xauusd_ai
    --cov-report=term-missing:skip-covered
    --cov-report=html:coverage_html
    --cov-report=json:coverage.json
    --cov-fail-under=70
    -n auto  # Parallel testing
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow-running tests
EOF

# Create test structure
mkdir -p tests/{unit,integration,e2e}
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
touch tests/e2e/__init__.py

# Move existing tests
mv tests/test_*.py tests/unit/ 2>/dev/null || true
```

#### 1.3 Create Verification Script
```bash
cat > scripts/verify.sh << 'EOF'
#!/bin/bash
set -e

echo "════════════════════════════════════════════════════════════"
echo "  XAUUSD AI — VERIFICATION LOOP"
echo "════════════════════════════════════════════════════════════"
echo ""

# Phase 1: Lint
echo "▶ Phase 1: Lint Check"
ruff check src/ scripts/ 2>&1 | head -30 || echo "⚠️ Lint warnings found"
echo ""

# Phase 2: Type Check
echo "▶ Phase 2: Type Check"
pyright src/ 2>&1 | head -30 || echo "⚠️ Type errors found"
echo ""

# Phase 3: Security Scan
echo "▶ Phase 3: Security Scan"
echo "Checking for exposed credentials..."
SECRETS=$(grep -rn "password.*=.*['\"]" configs/ 2>/dev/null | grep -v "_env" | wc -l)
if [ "$SECRETS" -gt 0 ]; then
  echo "❌ FAIL: Found $SECRETS hardcoded credentials"
  grep -rn "password.*=.*['\"]" configs/ | grep -v "_env" | head -5
  exit 1
else
  echo "✅ PASS: No hardcoded credentials"
fi
echo ""

# Phase 4: Tests
echo "▶ Phase 4: Test Suite"
pytest tests/ \
  --cov=src/xauusd_ai \
  --cov-report=term-missing:skip-covered \
  --cov-fail-under=70 \
  -n auto \
  2>&1 | tail -50
echo ""

# Phase 5: Diff Review
echo "▶ Phase 5: Changed Files"
git diff --stat HEAD 2>/dev/null || echo "No git repo"
echo ""

echo "════════════════════════════════════════════════════════════"
echo "  VERIFICATION COMPLETE"
echo "════════════════════════════════════════════════════════════"
EOF

chmod +x scripts/verify.sh
```

**Deliverable**: ECC installed, pytest configured, verification script ready

---

### Phase 2: Test Coverage to 70% (Days 3-5)

**Strategy**: AI-assisted test generation focusing on critical paths

#### 2.1 Priority Test Targets (35 source files)

**Critical files** (90%+ coverage required):
1. `src/xauusd_ai/orchestrator.py` — Trading loop
2. `src/xauusd_ai/execution/mt5_executor.py` — Live execution
3. `src/xauusd_ai/strategies/hybrid.py` — Strategy logic
4. `src/xauusd_ai/backtesting/engine.py` — Backtest engine

**Core logic** (70%+ coverage):
5. `src/xauusd_ai/features/dataset.py` — Feature engineering
6. `src/xauusd_ai/model/trainer.py` — Model training
7. `src/xauusd_ai/model/ensemble.py` — Combo133
8. `src/xauusd_ai/monitoring/drift.py` — Drift detection

**Infrastructure** (50%+ coverage):
9. `src/xauusd_ai/infra/db.py` — Database ops
10. `src/xauusd_ai/infra/storage.py` — S3/MinIO
11. `src/xauusd_ai/infra/metrics.py` — Prometheus

#### 2.2 Test Generation Workflow

**For each file, use this prompt with custom AI agent**:
```
@XAUUSD AI Dev Create comprehensive pytest tests for [file_path]:

Requirements:
- Unit tests for all public functions
- Mock external dependencies (MT5, DB, MLflow)
- Test edge cases and error handling
- Use pytest fixtures for setup
- Target 80% coverage for this file

Example test structure:
```python
import pytest
from unittest.mock import Mock, patch
from [module] import [Class/Function]

@pytest.fixture
def mock_mt5():
    with patch('MetaTrader5.initialize') as mock:
        mock.return_value = True
        yield mock

def test_function_success(mock_mt5):
    # Test happy path
    pass

def test_function_error_handling(mock_mt5):
    # Test error scenarios
    pass
```

#### 2.3 Test Checklist

**Unit Tests** (15 new files):
- [ ] `tests/unit/test_orchestrator.py` (10 tests)
- [ ] `tests/unit/test_mt5_executor.py` (8 tests)
- [ ] `tests/unit/test_hybrid_strategy.py` (12 tests)
- [ ] `tests/unit/test_features.py` (15 tests)
- [ ] `tests/unit/test_model_trainer.py` (10 tests)
- [ ] `tests/unit/test_ensemble.py` (8 tests)
- [ ] `tests/unit/test_slippage.py` (6 tests) — NEW for roadmap
- [ ] `tests/unit/test_regime_detection.py` (8 tests) — NEW for roadmap
- [ ] `tests/unit/test_drift.py` (6 tests)
- [ ] `tests/unit/test_db.py` (5 tests)
- [ ] `tests/unit/test_storage.py` (5 tests)
- [ ] `tests/unit/test_metrics.py` (4 tests)
- [ ] `tests/unit/test_config_loader.py` (3 tests)
- [ ] `tests/unit/test_risk_manager.py` (8 tests)
- [ ] `tests/unit/test_feature_importance.py` (4 tests)

**Integration Tests** (8 new files):
- [ ] `tests/integration/test_wf_pipeline.py` (5 tests)
- [ ] `tests/integration/test_mt5_bridge.py` (6 tests)
- [ ] `tests/integration/test_mlflow_tracking.py` (4 tests)
- [ ] `tests/integration/test_db_persistence.py` (5 tests)
- [ ] `tests/integration/test_paper_trading.py` (4 tests) — NEW
- [ ] `tests/integration/test_ab_testing.py` (3 tests) — NEW
- [ ] `tests/integration/test_stress_scenarios.py` (4 tests) — NEW
- [ ] `tests/integration/test_slippage_impact.py` (3 tests) — NEW

**E2E Tests** (3 new files):
- [ ] `tests/e2e/test_backtest_flow.py` (2 tests)
- [ ] `tests/e2e/test_paper_trade_flow.py` (2 tests)
- [ ] `tests/e2e/test_live_signal_flow.py` (2 tests)

**Total**: 26 new test files, ~140 tests → **Target: 70%+ coverage**

**Time estimate**: 
- AI generates test skeleton: 5 min/file
- Human review + fix: 15 min/file
- Total: 20 min/file × 26 files = **~9 hours** (spread across 3 days)

**Deliverable**: `pytest --cov` shows ≥70% coverage

---

### Phase 3: Advanced ML Techniques (Days 6-8)

**Priority order** based on roadmap impact:

#### 3.1 Slippage Model (Day 6 — CRITICAL)
**Impact**: Reduce backtest-live gap from 30% to <15%

```bash
# Create slippage module
cat > src/xauusd_ai/backtesting/slippage.py << 'EOF'
"""
Slippage model for realistic backtesting
"""
import numpy as np

def apply_slippage(price, direction, atr, config):
    """
    Calculate realistic fill price with slippage
    
    Components:
    1. Fixed spread (0.2-0.5 pips)
    2. Volatility-based slippage (atr × factor)
    3. Market impact (volume-dependent)
    """
    spread = config.get('spread', 0.2)  # pips
    slippage_factor = config.get('slippage_factor', 0.5)
    
    # Dynamic slippage based on ATR
    dynamic_slip = atr * slippage_factor
    total_slip = spread + dynamic_slip
    
    if direction == 'long':
        return price + total_slip  # Buy at higher price
    else:
        return price - total_slip  # Sell at lower price
EOF

# Update backtest engine to use slippage
# (AI agent will implement this)
```

**Config update**:
```yaml
# Add to configs/acc1_v14pp_profit.yaml
slippage:
  enabled: true
  spread: 0.2
  slippage_factor: 0.5
```

**Eval template**:
```bash
cat > evals/slippage-model.eval.md << 'EOF'
[CAPABILITY EVAL: slippage-model]

Task: Implement slippage model reducing backtest-live gap

Success Criteria:
- [ ] Slippage model in backtesting/slippage.py
- [ ] --slippage flag works
- [ ] WF with slippage shows 10-15% lower P&L
- [ ] Backtest-live gap < 20%

Expected:
Without slippage: PF=1.8, DD=8%
With slippage: PF=1.5, DD=10%
Live: PF=1.3, DD=12%
Gap: 13% (PASS)
EOF
```

**Test**:
```bash
# Run WF with slippage
.venv/bin/python scripts/walkforward_ict_wyckoff.py \
  configs/acc1_v14pp_profit.yaml \
  --slippage \
  --test-start 2026-04-01 \
  --test-bars 500 \
  --cache
```

---

#### 3.2 Regime Detection (Day 6-7 — HIGH)
**Impact**: Reduce DD by 20%, improve Sharpe by 0.3

```bash
# Create regime module
cat > src/xauusd_ai/strategies/regime_detection.py << 'EOF'
"""
Market regime detection using HMM and rule-based methods
"""
from hmmlearn import hmm
import numpy as np

def detect_regime_hmm(df, n_regimes=3):
    """
    HMM-based regime detection
    Returns: 0=ranging, 1=trending, 2=volatile
    """
    features = np.column_stack([
        df['atr'].values,
        df['close'].pct_change().rolling(20).std().values,
        df['close'].rolling(20).mean().values
    ])
    
    model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full")
    model.fit(features)
    return model.predict(features)

def detect_regime_simple(df, lookback=100):
    """
    Simple rule-based regime detection
    """
    returns = df['close'].pct_change()
    volatility = returns.rolling(lookback).std()
    trend = abs(returns.rolling(lookback).mean())
    
    if trend > 0.002 and volatility < 0.01:
        return 'trending'
    elif volatility > 0.02:
        return 'volatile'
    else:
        return 'ranging'
EOF
```

**Integration**:
```python
# In strategies/hybrid.py
from .regime_detection import detect_regime_simple

def generate_signal(df):
    regime = detect_regime_simple(df)
    
    if regime == 'ranging':
        # Reduce risk, pause trading
        max_positions = 1
        risk_pct = 0.015
    elif regime == 'trending':
        # Aggressive
        max_positions = 3
        risk_pct = 0.03
    # ...
```

---

#### 3.3 Sharpe & Calmar Metrics (Day 7 — HIGH)
**Impact**: Better risk-adjusted performance measurement

```bash
# Update backtesting/engine.py
cat >> src/xauusd_ai/backtesting/metrics.py << 'EOF'
"""
Advanced trading metrics
"""
import numpy as np

def calculate_sharpe(trades_df, risk_free_rate=0.02):
    """
    Annualized Sharpe ratio
    """
    daily_returns = trades_df.groupby('date')['profit'].sum()
    daily_returns_pct = daily_returns / initial_balance
    
    excess_returns = daily_returns_pct - risk_free_rate / 252
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe

def calculate_calmar(total_return, max_dd, years):
    """
    Calmar ratio = Annual Return / Max Drawdown
    """
    annual_return = total_return / years
    return annual_return / abs(max_dd)

def calculate_turnover_adjusted_return(trades_df, config):
    """
    Return adjusted for transaction costs
    """
    gross_pnl = trades_df['profit'].sum()
    
    # Costs
    spread_cost = len(trades_df) * config['spread_per_trade']
    swap_cost = (trades_df['bars_held'] / 1440).sum() * config['swap_per_day']
    
    return gross_pnl - spread_cost - swap_cost
EOF
```

**Log to MLflow**:
```python
import mlflow

mlflow.log_metric("sharpe_ratio", calculate_sharpe(trades_df))
mlflow.log_metric("calmar_ratio", calculate_calmar(total_return, max_dd, years))
mlflow.log_metric("turnover_adj_return", calculate_turnover_adjusted_return(trades_df, config))
```

---

#### 3.4 Paper Trading Framework (Day 8 — CRITICAL)
**Impact**: Validate model before live deploy

```bash
# Create paper trading mode
cat > src/xauusd_ai/paper_trading.py << 'EOF'
"""
Paper trading shadow mode
"""
import logging
from datetime import datetime

class PaperTradingMode:
    def __init__(self, output_path='outputs/paper_trades.csv'):
        self.signals = []
        self.output_path = output_path
        
    def on_signal(self, signal):
        """Log signal without executing"""
        self.signals.append({
            'time': datetime.now(),
            'direction': signal['direction'],
            'entry_price': signal['price'],
            'sl': signal['sl'],
            'tp': signal['tp'],
            'size': signal['size'],
            'model_score': signal['score']
        })
        logging.info(f"[PAPER] Signal: {signal['direction']} @ {signal['price']}")
        
    def simulate_exit(self, signal_id, current_price):
        """Calculate hypothetical P&L"""
        signal = self.signals[signal_id]
        if signal['direction'] == 'long':
            pnl = current_price - signal['entry_price']
        else:
            pnl = signal['entry_price'] - current_price
        
        logging.info(f"[PAPER] Exit {signal_id}: P&L = {pnl:.2f}")
        return pnl
EOF
```

**Config**:
```yaml
# Add to configs/live_acc1.yaml
paper_trading: true  # Enable shadow mode
```

**Docker service**:
```yaml
# Add to docker-compose.yml
  paper-bot:
    build: .
    command: python src/xauusd_ai/orchestrator.py --paper-mode
    environment:
      - PAPER_TRADING=true
    volumes:
      - ./outputs:/app/outputs
    depends_on:
      - postgres
      - mlflow
```

---

#### 3.5 Feature Importance Stability Check (Day 8)
**Impact**: Detect overfitting, improve model robustness

```bash
cat > scripts/check_feature_stability.py << 'EOF'
"""
Check feature importance stability across folds
"""
import json
import pandas as pd
import numpy as np

def check_stability(fold_results):
    """
    Calculate correlation of feature importance across folds
    Target: ≥0.75
    """
    importances = []
    for fold in fold_results:
        meta_file = f'outputs/fold_{fold}_model_meta.json'
        with open(meta_file) as f:
            meta = json.load(f)
        importances.append(meta['feature_importances'])
    
    df = pd.DataFrame(importances)
    corr_matrix = df.T.corr()
    avg_corr = corr_matrix.mean().mean()
    
    print(f"Average feature importance correlation: {avg_corr:.2f}")
    
    if avg_corr < 0.75:
        print("⚠️ WARNING: Feature importance not stable across folds")
        print("Possible lookahead bias or unstable features")
    else:
        print("✅ PASS: Feature importance stable")
    
    return avg_corr

if __name__ == '__main__':
    # Load fold results
    check_stability(range(1, 10))
EOF
```

**Deliverable**: Slippage, regime detection, metrics, paper trading, stability check implemented

---

### Phase 4: Advanced Testing & Orchestration (Days 9-10)

#### 4.1 A/B Testing Framework
```yaml
# Add to docker-compose.yml
  live-bot-model-a:
    build: .
    command: python src/xauusd_ai/orchestrator.py --model=outputs/acc1_v14pp_model.pkl --account=acc1_a
    environment:
      - AB_TEST_GROUP=A
      - CAPITAL_SPLIT=0.5  # 50% of capital
    
  live-bot-model-b:
    build: .
    command: python src/xauusd_ai/orchestrator.py --model=outputs/acc1_v15_model.pkl --account=acc1_b
    environment:
      - AB_TEST_GROUP=B
      - CAPITAL_SPLIT=0.5
```

#### 4.2 Stress Testing
```bash
cat > tests/integration/test_stress_scenarios.py << 'EOF'
"""
Stress testing for extreme scenarios
"""
import pytest
import pandas as pd
from src.xauusd_ai.backtesting.engine import backtest

def generate_flash_crash(df, crash_bar=500):
    """Inject 5% flash crash"""
    df_copy = df.copy()
    df_copy.loc[crash_bar:crash_bar+5, 'close'] *= 0.95
    return df_copy

@pytest.mark.slow
def test_flash_crash_scenario():
    """Model survives flash crash with DD < 25%"""
    df = load_test_data()
    df_crash = generate_flash_crash(df)
    
    result = backtest(model, df_crash)
    
    assert result['max_dd'] < 0.25, "DD too high during flash crash"

@pytest.mark.slow
def test_consecutive_losses():
    """Model recovers from 10 consecutive losses"""
    # Simulate worst-case scenario
    pass
EOF
```

#### 4.3 Orchestration Recommendations

**Recommendation: Use Docker Compose (NOT PM2)**

**Rationale**:
1. ✅ Already configured with all infrastructure
2. ✅ Better isolation (containers vs processes)
3. ✅ Easier scaling (docker-compose scale)
4. ✅ Health checks built-in
5. ✅ Production-ready (Kubernetes path later)
6. ❌ PM2 adds Node.js dependency (unnecessary for Python project)

**Enhanced docker-compose.yml**:
```yaml
# Services for parallel execution
services:
  # Paper trading (shadow mode)
  paper-bot:
    build: .
    command: python src/xauusd_ai/orchestrator.py --paper-mode
    environment:
      - PAPER_TRADING=true
    depends_on:
      - postgres
      - mlflow
    restart: unless-stopped
    
  # A/B testing — Model A
  live-bot-a:
    build: .
    command: python src/xauusd_ai/orchestrator.py --model-a
    environment:
      - AB_GROUP=A
      - CAPITAL_SPLIT=0.5
    depends_on:
      - postgres
      - mlflow
    restart: unless-stopped
    
  # A/B testing — Model B
  live-bot-b:
    build: .
    command: python src/xauusd_ai/orchestrator.py --model-b
    environment:
      - AB_GROUP=B
      - CAPITAL_SPLIT=0.5
    depends_on:
      - postgres
      - mlflow
    restart: unless-stopped
```

**Usage**:
```bash
# Start all services
docker-compose up -d

# Start only paper trading
docker-compose up -d paper-bot

# Start A/B test
docker-compose up -d live-bot-a live-bot-b

# Monitor logs
docker-compose logs -f paper-bot
docker-compose logs -f live-bot-a live-bot-b

# Check status
docker-compose ps

# Stop specific service
docker-compose stop paper-bot
```

**Deliverable**: A/B testing, stress tests, Docker orchestration configured

---

## 📊 Success Metrics

### Week 1 End (Day 5)
- [ ] ✅ Credentials secured (rotated + moved to .env)
- [ ] ✅ ECC installed (5 agents, 5 skills)
- [ ] ✅ Test coverage ≥ 70%
- [ ] ✅ Verification loop green

### Week 2 End (Day 10)
- [ ] ✅ Slippage model implemented + tested
- [ ] ✅ Regime detection working
- [ ] ✅ Sharpe/Calmar metrics logged
- [ ] ✅ Paper trading framework ready
- [ ] ✅ Feature stability checked
- [ ] ✅ A/B testing infrastructure ready
- [ ] ✅ Stress tests passing

### Post-Implementation
- [ ] Backtest-live gap < 20% (from 30%)
- [ ] Sharpe ratio ≥ 1.5 (from ~1.2)
- [ ] Max DD ≤ 12% (from 15%)
- [ ] Paper trading 2 weeks pass (P&L match WF ±20%)

---

## 🚀 Next Steps (After Week 2)

### Month 2: Advanced ML (If time allows)
1. **Ensemble Models**: LightGBM + LSTM + LogReg
2. **Reinforcement Learning**: PPO/SAC for exit timing
3. **Synthetic Data**: GARCH-based data augmentation
4. **Market Microstructure**: L2 orderbook features (if broker supports)

### Month 3: Production Hardening
1. **Continuous Learning**: Model retraining automation (Airflow DAG)
2. **Drift Monitoring**: Real-time feature drift alerts
3. **Cost Optimization**: MLflow artifact lifecycle management
4. **Multi-Account Scaling**: 2 accounts → 10 accounts

---

## 📋 Daily Checklist Template

```bash
# Start of day
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"
source .venv/bin/activate
git pull

# Before commit
./scripts/verify.sh

# End of day
git add -A
git commit -m "feat: [describe work]"
git push

# Check metrics
docker-compose logs -f mlflow
open http://localhost:5000  # MLflow UI
open http://localhost:3000  # Grafana
```

---

## 🎯 Final Recommendation

**Week 1 priorities**:
1. **Day 0**: Rotate credentials (EMERGENCY)
2. **Days 1-2**: Install ECC + test infrastructure
3. **Days 3-5**: Test coverage to 70% (AI-assisted)

**Week 2 priorities**:
4. **Day 6**: Slippage model + regime detection
5. **Day 7**: Sharpe/Calmar metrics + feature stability
6. **Day 8**: Paper trading framework
7. **Days 9-10**: A/B testing + stress tests + Docker orchestration

**Post-Week-2**: Focus on model improvements (ensemble, RL) using solid foundation

**Total investment**: 10 days (2 weeks)
**Expected ROI**: 5-10x (prevented losses + faster research cycles + production-ready infrastructure)

---

## ⚠️ Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Test coverage takes longer | Focus on critical files first (orchestrator, execution) |
| Slippage model complex | Start simple (spread + volatility), iterate later |
| Regime detection false signals | A/B test in paper mode first |
| Docker resource limits | Set memory limits, monitor with Prometheus |
| Credential rotation downtime | Rotate 1 account at a time, test immediately |

---

## 📚 References

- ECC Integration Plan: `.github/agents/knowledge/ecc-integration-plan.md`
- Implementation Roadmap: `.github/agents/knowledge/implementation-roadmap.md`
- Backtest-Live Gap Analysis: `.github/agents/knowledge/backtest-live-gap.md`
- Model Training Guide: `.github/agents/knowledge/model-training.md`
- Config Reference: `.github/agents/knowledge/config-reference.md`
