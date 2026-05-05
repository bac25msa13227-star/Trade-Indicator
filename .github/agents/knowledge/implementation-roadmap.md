# Roadmap Implementation — XAUUSD AI Trading

## Tổng quan 4 nhóm tính năng

| Nhóm | Mục đích | Timeline | Priority |
|------|----------|----------|----------|
| **Model nâng cao** | Tăng accuracy, robust | Tháng 2-3 | Medium-High |
| **Simulation thực tế** | Giảm backtest-live gap | Tuần 1-2 | **Critical** |
| **Quy trình validation** | Rút ngắn thời gian test | Tuần 3-4 | **Critical** |
| **Metrics nâng cao** | Đo lường chính xác hơn | Tuần 1-2 | High |

---

## NHÓM 1: Model Nâng Cao

### 1.1 Ensemble Models
**Mục đích**: Kết hợp nhiều model giảm variance, tăng robustness.

**Architecture**:
```python
# model/ensemble.py
class EnsembleModel:
    def __init__(self):
        self.lgbm = LightGBM()
        self.lstm = LSTM(input_dim=50, hidden=128)
        self.stat_model = LogisticRegression()  # Baseline
        
    def predict(self, X):
        lgbm_pred = self.lgbm.predict_proba(X)[:, 1]
        lstm_pred = self.lstm.predict(X)
        stat_pred = self.stat_model.predict_proba(X)[:, 1]
        
        # Weighted average
        return 0.5 * lgbm_pred + 0.3 * lstm_pred + 0.2 * stat_pred
```

**Integration**:
- Thêm `--ensemble` flag vào `walkforward_ict_wyckoff.py`
- Train 3 models song song mỗi fold
- Log weights vào MLflow

**Challenges**:
- LSTM cần nhiều data → có thể overfit với 6000 bars
- Training time tăng 3x
- Hyperparams cho 3 models → search space lớn

**Priority**: Medium (sau khi slippage/gap đã fix)

**Files cần sửa**:
- `model/trainer.py` — thêm EnsembleTrainer class
- `scripts/walkforward_ict_wyckoff.py` — thêm --ensemble flag
- `backtesting/engine.py` — support ensemble prediction

---

### 1.2 Regime Detection
**Mục đích**: Phát hiện thị trường trending/ranging/volatile → adjust strategy.

**Method 1: HMM (Hidden Markov Model)**
```python
# strategies/regime_detection.py
from hmmlearn import hmm

def detect_regime_hmm(df, n_regimes=3):
    """
    Returns: regime_id (0=ranging, 1=trending, 2=volatile)
    """
    features = np.column_stack([
        df['atr'].values,
        df['close'].pct_change().rolling(20).std().values,
        df['close'].rolling(20).mean().values
    ])
    
    model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full")
    model.fit(features)
    regimes = model.predict(features)
    return regimes
```

**Method 2: Rule-based (simpler)**
```python
def detect_regime_simple(df, lookback=100):
    returns = df['close'].pct_change()
    volatility = returns.rolling(lookback).std()
    trend = abs(returns.rolling(lookback).mean())
    
    if trend > 0.002 and volatility < 0.01:
        return 'trending'
    elif volatility > 0.02:
        return 'volatile'
    else:
        return 'ranging'
```

**Usage in strategy**:
```python
# strategies/hybrid.py
regime = detect_regime(df)

if regime == 'ranging':
    # Giảm risk, pause trading
    max_positions = 1
    risk_pct = 0.015
elif regime == 'trending':
    # Aggressive
    max_positions = 3
    risk_pct = 0.03
elif regime == 'volatile':
    # Defensive
    max_positions = 2
    risk_pct = 0.02
```

**Priority**: High — có thể giảm DD ngay lập tức

**Files cần sửa**:
- `strategies/regime_detection.py` — new file
- `strategies/hybrid.py` — integrate regime logic
- `backtesting/engine.py` — add regime column to results

---

### 1.3 Reinforcement Learning (RL) Fine-tuning
**Mục đích**: Model tự học từ P&L thực, không cần label.

**Architecture**:
```python
# model/rl_agent.py
from stable_baselines3 import PPO

class RLTradingAgent:
    def __init__(self, base_model):
        self.base_model = base_model  # LightGBM signal generator
        self.rl_model = PPO("MlpPolicy", env, verbose=1)
        
    def decide_action(self, state):
        # LightGBM suggest signal
        signal = self.base_model.predict(state)
        
        # RL adjust: should we trade? what size? when exit?
        action = self.rl_model.predict(state)
        return action
```

**Reward function**:
```python
def calculate_reward(trade_result):
    pnl = trade_result['profit']
    hold_time = trade_result['bars_held']
    
    # Reward: profit/time (Sharpe-like)
    reward = pnl / max(hold_time, 1)
    
    # Penalty for drawdown
    if pnl < 0:
        reward *= 2  # Punish losses harder
    
    return reward
```

**Priority**: Low — chỉ sau khi base model đã stable

**Challenges**:
- RL cần rất nhiều episodes (1000+) → compute intensive
- Reward shaping rất khó → dễ học policy sai
- Debugging RL rất khó

**Files cần tạo**:
- `model/rl_agent.py`
- `model/rl_env.py` — trading environment
- `scripts/train_rl.py` — training script

---

### 1.4 Market Microstructure Features
**Mục đích**: Thêm features từ order book, spread, volume imbalance.

**Nếu có access L2 data (order book)**:
```python
# features/microstructure.py
def extract_microstructure_features(orderbook):
    bid_volume = sum(orderbook['bids']['volume'])
    ask_volume = sum(orderbook['asks']['volume'])
    
    return {
        'volume_imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume),
        'bid_ask_spread': orderbook['asks'][0]['price'] - orderbook['bids'][0]['price'],
        'depth_imbalance': len(orderbook['bids']) - len(orderbook['asks'])
    }
```

**Nếu KHÔNG có L2 (chỉ có OHLCV)**:
```python
def proxy_microstructure(df):
    # Proxy spread bằng high-low
    df['proxy_spread'] = df['high'] - df['low']
    
    # Volume surge = unusual volume
    df['volume_surge'] = df['volume'] / df['volume'].rolling(20).mean()
    
    return df
```

**Priority**: Low — cần broker support L2 data

**Files cần sửa**:
- `features/dataset.py` — add microstructure columns

---

## NHÓM 2: Simulation Thực Tế (CRITICAL)

### 2.1 Slippage Model
**Mục đích**: Simulate realistic execution trong backtest.

**Implementation**:
```python
# backtesting/engine.py
def apply_slippage(price, direction, config, market_state):
    """
    Slippage = spread + volatility-based component
    """
    spread = config['spread']  # Fixed part (0.2 pips)
    atr = market_state['atr']
    slippage_factor = config['slippage_factor']  # 0.3-0.8
    
    # Dynamic slippage
    dynamic_slip = atr * slippage_factor
    total_slip = spread + dynamic_slip
    
    if direction == 'long':
        return price + total_slip  # Buy cao hơn
    else:
        return price - total_slip  # Sell thấp hơn
```

**Config**:
```yaml
# configs/acc1_v14pp_profit.yaml
slippage:
  enabled: true
  spread: 0.2       # pips
  slippage_factor: 0.5
```

**CLI**:
```bash
python scripts/walkforward_ict_wyckoff.py config.yaml --slippage
```

**Priority**: **CRITICAL** — triển khai tuần 1

**Files cần sửa**:
- `backtesting/engine.py` — thêm `apply_slippage()` function
- `scripts/walkforward_ict_wyckoff.py` — thêm `--slippage` flag
- `configs/*.yaml` — thêm slippage section

**Expected impact**: Backtest P&L giảm 10-15%, gần với live hơn.

---

### 2.2 Partial Fill Simulation
**Mục đích**: Simulate lệnh không fill hoàn toàn ngay lập tức.

**Scenario**: Đặt 1.0 lot, nhưng chỉ fill 0.7 lot trong 1 bar.

**Implementation**:
```python
def simulate_partial_fill(order_size, market_liquidity):
    """
    Returns: (filled_size, remaining_size)
    """
    # High liquidity (normal) → fill 100%
    if market_liquidity > 0.8:
        return order_size, 0
    
    # Low liquidity → random partial fill
    fill_ratio = np.random.uniform(0.5, 1.0)
    filled = order_size * fill_ratio
    remaining = order_size - filled
    
    return filled, remaining
```

**Priority**: Medium — XAUUSD thường liquid, ít xảy ra

**Files cần sửa**:
- `backtesting/engine.py` — thêm partial fill logic

---

### 2.3 Latency Injection
**Mục đích**: Simulate delay giữa signal → order sent → fill.

**Implementation**:
```python
# backtesting/engine.py
def inject_latency(df, signal_idx, latency_ms=100):
    """
    Signal at bar N, but order sent at bar N+delay
    """
    # 1 bar M1 = 60,000 ms
    delay_bars = latency_ms / 60000
    
    # Nếu delay < 1 bar, assume fill cùng bar nhưng giá worse
    if delay_bars < 1:
        slippage += 0.1  # Extra slippage
        return signal_idx
    else:
        return signal_idx + int(delay_bars)
```

**Config**:
```yaml
latency:
  enabled: true
  delay_ms: 100  # 100ms network latency
```

**Priority**: Low — tác động nhỏ với timeframe M1+

**Files cần sửa**:
- `backtesting/engine.py`

---

### 2.4 Tick-by-Tick Replay
**Mục đích**: Backtest ở độ chi tiết tick thay vì OHLC bars.

**Requirement**: Cần tick data (không có sẵn trong M1 CSV).

**Implementation sketch**:
```python
# backtesting/tick_engine.py
def replay_ticks(df_m1):
    """
    Generate ticks from M1 OHLC
    Assume: open → high → low → close
    """
    ticks = []
    for idx, row in df_m1.iterrows():
        ticks.append({'time': row['time'], 'price': row['open']})
        ticks.append({'time': row['time'] + 20s, 'price': row['high']})
        ticks.append({'time': row['time'] + 40s, 'price': row['low']})
        ticks.append({'time': row['time'] + 60s, 'price': row['close']})
    return ticks
```

**Priority**: Low — cần tick data thật, không phải synthetic

---

## NHÓM 3: Quy Trình Validation (CRITICAL)

### 3.1 Paper Trading Shadow Mode
**Mục đích**: Chạy model live nhưng không vào lệnh thật — chỉ log signals.

**Implementation**:
```python
# orchestrator.py
class PaperTradingMode:
    def __init__(self):
        self.signals = []
        
    def on_signal(self, signal):
        # Log signal nhưng KHÔNG gọi mt5_executor
        self.signals.append({
            'time': datetime.now(),
            'direction': signal['direction'],
            'entry_price': signal['price'],
            'sl': signal['sl'],
            'tp': signal['tp'],
            'size': signal['size']
        })
        logger.info(f"[PAPER] Signal: {signal['direction']} @ {signal['price']}")
        
    def simulate_exit(self, signal_id, current_price):
        # Tính P&L giả định
        signal = self.signals[signal_id]
        if signal['direction'] == 'long':
            pnl = current_price - signal['entry_price']
        else:
            pnl = signal['entry_price'] - current_price
        
        logger.info(f"[PAPER] Exit {signal_id}: P&L = {pnl}")
```

**Config**:
```yaml
# configs/live_acc1.yaml
paper_trading: true  # Enable shadow mode
```

**Priority**: **CRITICAL** — tuần 3-4

**Files cần sửa**:
- `orchestrator.py` — thêm PaperTradingMode class
- `main.py` — flag để enable paper mode
- `configs/live_acc1.yaml` — thêm paper_trading option

---

### 3.2 A/B Testing Framework
**Mục đích**: Chạy model mới và cũ song song, so sánh P&L.

**Architecture**:
```python
# orchestrator.py
class ABTestOrchestrator:
    def __init__(self):
        self.model_a = load_model('outputs/acc1_v14pp_model.pkl')  # Current
        self.model_b = load_model('outputs/acc1_v15_model.pkl')    # New
        self.accounts = {
            'A': Account(id='acc1', model=self.model_a, size=0.5),  # 50% capital
            'B': Account(id='acc1_test', model=self.model_b, size=0.5)
        }
        
    def run(self):
        while True:
            data = fetch_live_data()
            
            signal_a = self.model_a.predict(data)
            signal_b = self.model_b.predict(data)
            
            if signal_a:
                self.accounts['A'].place_order(signal_a)
            if signal_b:
                self.accounts['B'].place_order(signal_b)
            
            # Log metrics
            log_ab_metrics(self.accounts)
```

**Dashboard metric**:
| Model | Trades | Win Rate | P&L | Sharpe | Max DD |
|-------|--------|----------|-----|--------|--------|
| A (current) | 45 | 52% | +$320 | 1.2 | -8% |
| B (new) | 38 | 58% | +$410 | 1.5 | -6% |

→ Sau 2 tuần, nếu B tốt hơn rõ ràng → deploy full.

**Priority**: High — tuần 3-4

**Files cần tạo**:
- `orchestrator_ab.py` — new file
- `scripts/compare_ab_results.py` — analysis script

---

### 3.3 Stress Testing
**Mục đích**: Test model trên extreme scenarios không có trong training data.

**Scenarios**:
1. **Flash Crash** — giá giảm 5% trong 5 phút
2. **News Spike** — giá tăng 3% đột ngột
3. **Low Liquidity** — spread tăng 10x
4. **Consecutive Losses** — 10 losses liên tiếp

**Implementation**:
```python
# scripts/stress_test.py
def generate_flash_crash(df, crash_bar=500):
    """
    Inject flash crash vào backtest data
    """
    df_copy = df.copy()
    
    # Bar 500: giá giảm 5%
    df_copy.loc[crash_bar:crash_bar+5, 'close'] *= 0.95
    df_copy.loc[crash_bar:crash_bar+5, 'low'] *= 0.93
    
    return df_copy

# Test model
df_crash = generate_flash_crash(df)
result = backtest(model, df_crash)
print(f"Max DD during crash: {result['max_dd']}")
```

**Acceptance**: Max DD < 25% trong stress scenarios.

**Priority**: Medium — tháng 2

**Files cần tạo**:
- `scripts/stress_test.py`

---

### 3.4 Synthetic Data Generation (GARCH)
**Mục đích**: Tạo thêm data để train model, test robustness.

**Method**: GARCH model sinh data có tính chất thống kê giống data thật.

**Implementation**:
```python
# scripts/generate_synthetic_data.py
from arch import arch_model

def generate_garch_data(historical_returns, n_samples=10000):
    """
    Fit GARCH model trên returns thật, generate synthetic returns
    """
    model = arch_model(historical_returns, vol='Garch', p=1, q=1)
    model_fit = model.fit()
    
    # Simulate
    simulated = model_fit.forecast(horizon=n_samples, method='simulation')
    synthetic_returns = simulated.simulations.values[-1, :, 0]
    
    # Convert returns → prices
    synthetic_prices = (1 + synthetic_returns).cumprod() * historical_prices[0]
    return synthetic_prices
```

**Usage**:
- Generate 10,000 bars synthetic data
- Augment training set: real data + synthetic data
- Test if model generalizes better

**Priority**: Low — tháng 3

**Files cần tạo**:
- `scripts/generate_synthetic_data.py`
- `model/trainer.py` — thêm augmentation option

---

## NHÓM 4: Metrics Nâng Cao (HIGH PRIORITY)

### 4.1 Sharpe Ratio
**Mục đích**: Đo risk-adjusted return.

**Formula**:
```
Sharpe = (Return - Risk_Free_Rate) / Std(Returns)
```

**Implementation**:
```python
# backtesting/engine.py
def calculate_sharpe(trades_df, risk_free_rate=0.02):
    """
    Annualized Sharpe ratio
    """
    daily_returns = trades_df.groupby('date')['profit'].sum()
    daily_returns_pct = daily_returns / initial_balance
    
    excess_returns = daily_returns_pct - risk_free_rate / 252
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    return sharpe
```

**Log to MLflow**:
```python
mlflow.log_metric("sharpe_ratio", sharpe, step=fold_id)
```

**Priority**: **CRITICAL** — tuần 1

**Files cần sửa**:
- `backtesting/engine.py` — thêm calculate_sharpe()
- `scripts/walkforward_ict_wyckoff.py` — log mỗi fold
- `visualization/reports.py` — thêm Sharpe vào report

---

### 4.2 Calmar Ratio
**Mục đích**: Return / Max Drawdown.

**Formula**:
```
Calmar = Annual_Return / Max_Drawdown
```

**Implementation**:
```python
def calculate_calmar(trades_df, initial_balance):
    total_return = (final_balance - initial_balance) / initial_balance
    annual_return = total_return * (252 / trading_days)
    
    max_dd = calculate_max_drawdown(balance_curve)
    
    calmar = annual_return / max_dd
    return calmar
```

**Priority**: High — tuần 1

**Files cần sửa**:
- `backtesting/engine.py`

---

### 4.3 Turnover-Adjusted Return
**Mục đích**: Trừ hết chi phí giao dịch vào P&L.

**Formula**:
```
Adjusted_Return = Gross_PnL - (Num_Trades × Spread) - (Holding_Days × Swap)
```

**Implementation**:
```python
def calculate_adjusted_return(trades_df, config):
    gross_pnl = trades_df['profit'].sum()
    
    # Spread cost
    spread_cost = len(trades_df) * config['spread_per_trade']
    
    # Swap cost (holding overnight)
    swap_cost = (trades_df['bars_held'] / 1440).sum() * config['swap_per_day']
    
    adjusted_pnl = gross_pnl - spread_cost - swap_cost
    return adjusted_pnl
```

**Priority**: High — tuần 1-2

**Files cần sửa**:
- `backtesting/engine.py`

---

### 4.4 Feature Importance Stability
**Mục đích**: Đảm bảo model không overfit bằng cách check features quan trọng stable qua các fold.

**Implementation**:
```python
# scripts/check_feature_stability.py
def check_stability(fold_results):
    importances = []
    for fold in fold_results:
        importances.append(fold['model'].feature_importances_)
    
    # Tính correlation
    df = pd.DataFrame(importances)
    corr_matrix = df.T.corr()
    avg_corr = corr_matrix.mean().mean()
    
    print(f"Avg feature importance correlation: {avg_corr:.2f}")
    
    # Nếu < 0.7 → không stable
    if avg_corr < 0.7:
        print("⚠️ WARNING: Feature importance not stable across folds")
    
    return avg_corr
```

**Threshold**: ≥ 0.75

**Priority**: High — tuần 2

**Files cần tạo**:
- `scripts/check_feature_stability.py`
- `scripts/walkforward_ict_wyckoff.py` — log importances mỗi fold

---

## Implementation Order (Recommended)

### Week 1-2 (Foundation)
1. ✅ Slippage model + flag
2. ✅ Sharpe/Calmar calculation
3. ✅ Feature stability check
4. ✅ Turnover-adjusted return

→ **Goal**: Backtest-live gap giảm từ 30% xuống 15-20%

### Week 3-4 (Validation)
5. ✅ Paper trading shadow mode
6. ✅ A/B test framework
7. ✅ Compare WF vs Live script
8. ✅ Regime detection (simple rule-based)

→ **Goal**: Có tool để validate model mới nhanh (2 tuần thay vì 3 tháng)

### Month 2 (Improvement)
9. ✅ Regime detection (HMM)
10. ✅ Stress testing
11. ✅ Adaptive sizing by regime
12. ✅ Ensemble model (LightGBM + stat)

→ **Goal**: Model robust hơn, DD giảm

### Month 3 (Advanced)
13. ✅ RL fine-tuning (PPO)
14. ✅ Synthetic data (GARCH)
15. ✅ Tick simulation (nếu có data)
16. ✅ Market microstructure features (nếu có L2)

→ **Goal**: State-of-the-art performance

---

## Success Metrics

| Milestone | Current | Target | Deadline |
|-----------|---------|--------|----------|
| Backtest-Live Gap | 30% | ≤ 15% | Week 2 |
| Validation time | 3 months | 2 weeks | Week 4 |
| Sharpe (annualized) | 1.0-1.2 | ≥ 1.5 | Month 2 |
| Max DD | 12-15% | ≤ 10% | Month 2 |
| Feature stability | Unknown | ≥ 0.75 | Week 2 |
