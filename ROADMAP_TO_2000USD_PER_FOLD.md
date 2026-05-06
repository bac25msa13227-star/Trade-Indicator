# Roadmap: $200 → $2,000-3,000/Fold (1 Month)

**Ngày:** 7 May 2026  
**Mục tiêu:** Tăng profit từ $378/fold → $2,000-3,000/fold (gấp 5-8×)  
**Ràng buộc:** Max DD < 15%, có cơ chế bảo toàn vốn

---

## 📊 Hiện Trạng (Baseline)

| Chỉ số | Giá trị hiện tại |
|--------|------------------|
| **Avg profit/fold** | $378 (+89%) |
| **Best fold** | $784 (Fold 2, +292%) |
| **Worst fold** | $163 (Fold 13, -18%) |
| **Win rate** | 41.6% |
| **Avg RR** | ~3.67 |
| **Max DD** | -9.15% |
| **Sharpe** | 4.557 |
| **Trades/fold** | 60 avg |

**Gap cần đạt:** $2,000 / $378 = **5.3× improvement**

---

## 🎯 Strategy: 3-Phase Approach

### **Phase 1: Optimize Existing (Target: $378 → $800, 2× improvement)**
- Better exit timing
- Dynamic position sizing
- Entry quality improvement

### **Phase 2: Advanced Models (Target: $800 → $1,500, 1.9× improvement)**
- Ensemble models
- Regime-based strategy switching
- Feature engineering

### **Phase 3: RL & Microstructure (Target: $1,500 → $2,500, 1.7× improvement)**
- Reinforcement learning
- Market microstructure
- Alternative data

---

## 📋 Detailed Roadmap (Prioritized)

### **P0 - CRITICAL (Implement First)**

#### **1. Better Exit Strategy - Trailing Stop Improvement**
**Current:** Fixed SL/TP based on ATR  
**Problem:** Leaving money on table (avg RR 3.67, but could be 5-6)  
**Solution:** Adaptive trailing SL based on price action

**Expected Impact:**
- RR: 3.67 → 5.0 (+36% per trade)
- Profit: $378 → $514/fold (+$136)
- **Total improvement: 1.36×**

**Implementation:**
```python
# src/xauusd_ai/execution/trailing_sl.py
class AdaptiveTrailingSL:
    def __init__(self):
        self.activation_rr = 1.5  # Start trailing at 1.5R
        self.trail_distance = 0.8  # Trail 0.8R below high
        self.acceleration = True   # Tighten when momentum weakens
    
    def update_sl(self, current_rr, momentum, volatility):
        if current_rr < self.activation_rr:
            return None  # Don't trail yet
        
        if momentum < 0.3:  # Momentum weakening
            trail_distance = 0.5  # Tighten
        else:
            trail_distance = self.trail_distance
        
        return current_high - (trail_distance * initial_sl_distance)
```

**Testing:**
- Backtest on 2024-2025 data
- Compare: Fixed SL vs Trailing vs Adaptive
- Target: Improve worst folds (Fold 13: -18% → +20%)

**Effort:** Medium (2-3 days)  
**Risk:** Low (can A/B test)

---

#### **2. Dynamic Risk Based on Confidence**
**Current:** Fixed 3% risk per trade  
**Problem:** Same risk for 60% vs 90% confidence signals  
**Solution:** Scale risk 2-5% based on model confidence

**Expected Impact:**
- High confidence trades (>0.80): 5% risk → 1.67× profit
- Low confidence trades (<0.65): 2% risk → 0.67× risk
- Net effect: +40% profit on good trades, -30% loss on bad trades
- **Total improvement: 1.25×**

**Implementation:**
```python
# src/xauusd_ai/execution/risk.py
def calculate_dynamic_risk(base_risk, confidence, regime_score):
    """
    Base risk: 3%
    Confidence: 0.60-1.00
    Regime: -1 to +1
    """
    # Confidence scaling
    if confidence > 0.85:
        conf_mult = 1.5  # 3% → 4.5%
    elif confidence > 0.75:
        conf_mult = 1.2  # 3% → 3.6%
    elif confidence < 0.65:
        conf_mult = 0.7  # 3% → 2.1%
    else:
        conf_mult = 1.0
    
    # Regime scaling
    if regime_score > 0.7:  # Strong trending
        regime_mult = 1.1
    elif regime_score < 0.3:  # Weak/sideway
        regime_mult = 0.8
    else:
        regime_mult = 1.0
    
    # Final risk (capped at 5%)
    risk = min(base_risk * conf_mult * regime_mult, 0.05)
    
    return risk
```

**Testing:**
- Backtest with dynamic risk on 2025 data
- Compare: 15% higher profit on high-conf trades
- Validate: DD doesn't exceed 15%

**Effort:** Low (1 day)  
**Risk:** Medium (need to validate DD)

---

#### **3. Entry Quality Improvement - Multiple Confirmations**
**Current:** Single model prediction  
**Problem:** False breakouts, noise  
**Solution:** Require 2-3 confirmations before entry

**Expected Impact:**
- Win rate: 41.6% → 48% (+15%)
- Signal rate: 40% → 25% (-37%, more selective)
- Net profit: +10% (quality > quantity)
- **Total improvement: 1.10×**

**Confirmations:**
1. Model confidence > 0.75
2. Regime favorable (regime_score > 0.5)
3. Volume confirmation (tick_volume_zscore > 0.5)
4. Momentum alignment (RSI not overbought/oversold)
5. Multi-TF alignment (H4 + H1 same direction)

**Implementation:**
```python
# src/xauusd_ai/strategies/hybrid.py
def should_enter_trade(signal, features, regime):
    confirmations = 0
    
    # 1. Model confidence
    if signal.probability > 0.75:
        confirmations += 1
    
    # 2. Regime favorable
    if regime.regime_favorable == 1:
        confirmations += 1
    
    # 3. Volume
    if features.tick_volume_zscore > 0.5:
        confirmations += 1
    
    # 4. Momentum
    rsi = features.rsi_h1
    if 30 < rsi < 70:  # Not extreme
        confirmations += 1
    
    # 5. Multi-TF alignment
    if features.multi_tf_consensus >= 2:
        confirmations += 1
    
    # Require 3/5 confirmations
    return confirmations >= 3
```

**Testing:**
- Backtest with confirmation filter
- Target: Fold 13 (worst) improves -18% → +10%

**Effort:** Medium (2 days)  
**Risk:** Low

---

**P0 Combined Impact:**
- Trailing SL: 1.36×
- Dynamic risk: 1.25×
- Entry quality: 1.10×
- **Total: 1.36 × 1.25 × 1.10 = 1.87×**
- **Profit: $378 → $707/fold**

---

### **P1 - HIGH PRIORITY (Implement Next)**

#### **4. Ensemble Models - Multiple Model Voting**
**Current:** Single LightGBM model (COMBO133)  
**Problem:** Single point of failure  
**Solution:** Ensemble of LightGBM + XGBoost + CatBoost

**Expected Impact:**
- Win rate: 48% → 53% (+10%)
- AUC: 0.686 → 0.720 (+5%)
- Robustness: Lower variance across folds
- **Total improvement: 1.15×**

**Implementation:**
```python
# src/xauusd_ai/model/ensemble.py
class EnsembleModel:
    def __init__(self):
        self.models = {
            'lgbm': LightGBM(),      # Fast, current model
            'xgboost': XGBoost(),    # Robust to outliers
            'catboost': CatBoost(),  # Good with categorical
        }
        self.weights = {
            'lgbm': 0.5,
            'xgboost': 0.3,
            'catboost': 0.2,
        }
    
    def predict(self, X):
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X)
        
        # Weighted average
        ensemble_pred = sum([
            pred * self.weights[name]
            for name, pred in predictions.items()
        ])
        
        return ensemble_pred
    
    def get_confidence(self, predictions):
        # Agreement between models
        agreement = std([p for p in predictions.values()])
        
        # High agreement = high confidence
        if agreement < 0.1:
            return 0.9
        elif agreement < 0.2:
            return 0.7
        else:
            return 0.5
```

**Testing:**
- Train 3 models on same data
- Backtest ensemble on 2025
- Compare: Single vs Ensemble
- Target: +10% win rate

**Effort:** Medium (3-4 days)  
**Risk:** Low (fallback to single model)

---

#### **5. Regime-Based Strategy Switching**
**Current:** Same strategy for all regimes  
**Problem:** Trending strategy loses in sideway  
**Solution:** Switch strategy based on regime

**Expected Impact:**
- Fold 13 (sideway): -18% → +15% (+$66)
- Fold 5 (sideway): +7% → +30% (+$46)
- Other folds: Maintain or slight improvement
- **Total improvement: 1.20×**

**Strategies:**
1. **Trending:** Breakout + trend following (current)
2. **Sideway:** Mean reversion + support/resistance
3. **Volatile:** Reduce risk, wait for calm

**Implementation:**
```python
# src/xauusd_ai/strategies/regime_strategy.py
class RegimeBasedStrategy:
    def __init__(self):
        self.trending_strategy = TrendFollowingStrategy()
        self.sideway_strategy = MeanReversionStrategy()
        self.volatile_strategy = ConservativeStrategy()
    
    def select_strategy(self, regime):
        if regime.regime_trending == 1 and regime.regime_volatile == 0:
            return self.trending_strategy
        
        elif regime.regime_sideway == 1:
            return self.sideway_strategy
        
        elif regime.regime_volatile == 1:
            return self.volatile_strategy
        
        else:
            return self.trending_strategy  # Default
    
    def generate_signal(self, features, regime):
        strategy = self.select_strategy(regime)
        return strategy.generate_signal(features)
```

**Mean Reversion Strategy:**
```python
class MeanReversionStrategy:
    def generate_signal(self, features):
        # Look for price deviation from mean
        zscore = (features.close - features.sma50) / features.atr_h4
        
        # Buy when oversold
        if zscore < -2.0 and features.rsi_h1 < 30:
            return Signal(direction='BUY', confidence=0.75)
        
        # Sell when overbought
        elif zscore > 2.0 and features.rsi_h1 > 70:
            return Signal(direction='SELL', confidence=0.75)
        
        return None
```

**Testing:**
- Classify each fold as trending/sideway/volatile
- Test mean reversion on sideway folds
- Target: Improve Fold 5, 13

**Effort:** Medium (3 days)  
**Risk:** Medium (need validation)

---

#### **6. Feature Engineering v2 - Market Microstructure**
**Current:** 72 features (ICT + Wyckoff + Regime)  
**Add:** Market microstructure features  
**Expected Impact:** +5-8% win rate

**New Features:**
1. **Spread dynamics** - Widen = volatile, tighten = liquid
2. **Tick speed** - High frequency = momentum
3. **Order flow imbalance** - Buy vs sell pressure
4. **Volume profile** - Support/resistance levels
5. **Session liquidity** - Asian (low) vs London (high)

**Implementation:**
```python
# src/xauusd_ai/features/microstructure.py
def add_microstructure_features(df):
    # 1. Spread dynamics
    df['spread'] = df['ask'] - df['bid']
    df['spread_zscore'] = (df['spread'] - df['spread'].rolling(100).mean()) / df['spread'].rolling(100).std()
    
    # 2. Tick speed (ticks per minute)
    df['tick_speed'] = df.index.to_series().diff().dt.total_seconds()
    df['tick_speed_ma'] = df['tick_speed'].rolling(10).mean()
    
    # 3. Order flow imbalance
    df['buy_volume'] = df['volume'] * (df['close'] > df['open']).astype(int)
    df['sell_volume'] = df['volume'] * (df['close'] < df['open']).astype(int)
    df['order_flow_imbalance'] = (df['buy_volume'] - df['sell_volume']) / df['volume']
    
    # 4. Volume profile (accumulation zones)
    df['volume_ma50'] = df['volume'].rolling(50).mean()
    df['volume_surge'] = (df['volume'] > df['volume_ma50'] * 2).astype(int)
    
    # 5. Session liquidity
    hour = df.index.hour
    df['session_liquidity'] = np.select([
        (hour >= 0) & (hour < 8),   # Asian
        (hour >= 8) & (hour < 16),  # London
        (hour >= 16) & (hour < 24), # NY
    ], [0.5, 1.0, 0.8], default=0.5)
    
    return df
```

**Testing:**
- Add 10 microstructure features
- Retrain model with 82 features
- Backtest on 2025
- Target: +5% win rate

**Effort:** Medium (3 days)  
**Risk:** Low

---

**P1 Combined Impact:**
- Ensemble: 1.15×
- Regime switching: 1.20×
- Microstructure: 1.08×
- **Total: 1.15 × 1.20 × 1.08 = 1.49×**
- **Profit: $707 → $1,053/fold**

**Cumulative (P0 + P1): $378 → $1,053 = 2.79×**

---

### **P2 - MEDIUM PRIORITY (After P0+P1)**

#### **7. Reinforcement Learning - Dynamic Position Sizing**
**Current:** Fixed position size based on balance  
**Problem:** Not adaptive to market conditions  
**Solution:** RL agent learns optimal sizing

**Expected Impact:**
- High confidence: Size up to 7%
- Low confidence: Size down to 1%
- Adaptive to recent performance
- **Total improvement: 1.25×**

**Implementation:**
```python
# src/xauusd_ai/model/rl_agent.py
import gym
from stable_baselines3 import PPO

class PositionSizingEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=0.01, high=0.07, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
    
    def step(self, action):
        # Action: position size (1%-7%)
        position_size = action[0]
        
        # Execute trade with this size
        pnl = self.execute_trade(position_size)
        
        # Reward: profit - penalty for large DD
        reward = pnl - 0.5 * max(0, self.current_dd - 0.10)
        
        return self.get_state(), reward, False, {}
    
    def get_state(self):
        # 20 features describing current state
        return np.array([
            self.signal_confidence,
            self.regime_score,
            self.recent_win_rate,
            self.current_dd,
            self.consecutive_wins,
            self.consecutive_losses,
            # ... 14 more features
        ])

# Training
env = PositionSizingEnv()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Usage
state = env.get_state()
action = model.predict(state, deterministic=True)
position_size = action[0]
```

**Testing:**
- Train RL agent on 2023-2024 data
- Test on 2025 data
- Compare: Fixed 3% vs RL dynamic
- Target: +20% profit, same DD

**Effort:** High (5-7 days)  
**Risk:** Medium (complex system)

---

#### **8. Alternative Data - Sentiment & Positioning**
**Current:** Only price/volume data  
**Add:** Market sentiment, COT positioning  
**Expected Impact:** +3-5% win rate

**Data Sources:**
1. **COT (Commitment of Traders)** - Large specs positioning
2. **News sentiment** - ForexFactory, Twitter, Reddit
3. **DXY (Dollar Index)** - Gold inverse correlation
4. **VIX** - Market fear gauge
5. **Fed Watch Tool** - Interest rate expectations

**Implementation:**
```python
# src/xauusd_ai/data/alternative_data.py
import yfinance as yf

def fetch_alternative_data(start_date, end_date):
    # 1. DXY (Dollar Index)
    dxy = yf.download('DX-Y.NYB', start=start_date, end=end_date)
    
    # 2. VIX (Volatility Index)
    vix = yf.download('^VIX', start=start_date, end=end_date)
    
    # 3. COT positioning (from CFTC)
    cot = fetch_cot_data('GOLD', start_date, end_date)
    
    # 4. News sentiment (from ForexFactory API)
    sentiment = fetch_news_sentiment(start_date, end_date)
    
    return {
        'dxy': dxy,
        'vix': vix,
        'cot': cot,
        'sentiment': sentiment,
    }

def add_alternative_features(df, alt_data):
    # Merge with main dataframe
    df = df.merge(alt_data['dxy'][['Close']], left_index=True, right_index=True, how='left')
    df.rename(columns={'Close': 'dxy'}, inplace=True)
    
    # DXY inverse correlation
    df['gold_dxy_spread'] = df['close'] / df['dxy']
    
    # VIX risk-on/risk-off
    df = df.merge(alt_data['vix'][['Close']], left_index=True, right_index=True, how='left')
    df.rename(columns={'Close': 'vix'}, inplace=True)
    df['risk_sentiment'] = np.where(df['vix'] < 20, 1, -1)  # Low VIX = risk-on
    
    return df
```

**Testing:**
- Fetch historical data
- Add 5 alt features
- Retrain with 87 features
- Target: +3% win rate

**Effort:** Medium (3-4 days)  
**Risk:** Low

---

**P2 Combined Impact:**
- RL sizing: 1.25×
- Alt data: 1.05×
- **Total: 1.25 × 1.05 = 1.31×**
- **Profit: $1,053 → $1,380/fold**

**Cumulative (P0 + P1 + P2): $378 → $1,380 = 3.65×**

---

### **P3 - LOWER PRIORITY (Polish & Optimize)**

#### **9. Advanced Testing - Tick-by-Tick + Stress Test**
**Current:** Bar-by-bar simulation  
**Improve:** Tick-level precision + extreme scenarios

**Components:**
1. **Tick replay:** True L1 data simulation
2. **Partial fills:** Order not fully filled at once
3. **Latency:** 50-200ms delay injection
4. **Market impact:** Large orders move price
5. **Stress test:** 2020 COVID crash, 2022 inflation spike

**Implementation:**
```python
# src/xauusd_ai/backtesting/tick_simulator.py
class TickSimulator:
    def __init__(self, tick_data):
        self.ticks = tick_data
        self.latency_ms = 100  # 100ms delay
        self.market_impact = 0.2  # 0.2 pips per lot
    
    def execute_order(self, order, current_time):
        # 1. Latency delay
        execution_time = current_time + timedelta(milliseconds=self.latency_ms)
        
        # 2. Get tick at execution time
        tick = self.ticks[self.ticks.index == execution_time].iloc[0]
        
        # 3. Market impact
        if order.lots > 1.0:
            slippage = order.lots * self.market_impact
            execution_price = tick.ask + slippage if order.side == 'BUY' else tick.bid - slippage
        else:
            execution_price = tick.ask if order.side == 'BUY' else tick.bid
        
        # 4. Partial fill (if spread too wide)
        if tick.spread > 2.0:  # Wide spread
            fill_rate = 0.5  # Only 50% filled
            filled_lots = order.lots * fill_rate
        else:
            filled_lots = order.lots
        
        return Fill(
            price=execution_price,
            lots=filled_lots,
            time=execution_time,
            slippage=slippage if order.lots > 1.0 else 0,
        )
```

**Stress Testing:**
```python
# Test on extreme periods
stress_periods = [
    ('2020-03-09', '2020-03-23'),  # COVID crash
    ('2022-02-24', '2022-03-08'),  # Ukraine war
    ('2023-03-10', '2023-03-17'),  # SVB collapse
]

for start, end in stress_periods:
    results = backtest_strategy(
        strategy=current_strategy,
        data=get_data(start, end),
        initial_balance=200,
    )
    
    print(f"Period: {start} → {end}")
    print(f"  Max DD: {results.max_dd:.2%}")
    print(f"  Final balance: ${results.final_balance:.0f}")
    print(f"  Survived: {'✅' if results.final_balance > 100 else '❌'}")
```

**Expected Impact:**
- More realistic P&L (-10% adjustment)
- Better risk management
- Confidence in live deployment

**Effort:** High (5 days)  
**Risk:** Low

---

#### **10. Capital Protection - Kelly Criterion + Stop Loss**
**Current:** No explicit capital protection  
**Add:** Kelly sizing + hard stop loss at -30%

**Kelly Criterion:**
```python
def kelly_fraction(win_rate, avg_win, avg_loss):
    """
    Optimal position size based on edge
    """
    if avg_loss == 0:
        return 0
    
    q = 1 - win_rate  # Probability of loss
    b = avg_win / avg_loss  # Payoff ratio
    
    kelly = (win_rate * b - q) / b
    
    # Use half-Kelly for safety
    return max(0, min(kelly * 0.5, 0.05))

# Usage
win_rate = 0.45
avg_win = 500
avg_loss = 200

optimal_risk = kelly_fraction(win_rate, avg_win, avg_loss)
# → 0.0375 (3.75% risk per trade)
```

**Hard Stop Loss:**
```python
# In orchestrator.py
if current_balance < initial_balance * 0.70:
    # Lost 30% of capital
    print("⚠️ HARD STOP TRIGGERED - Shutting down")
    save_state()
    send_alert("Bot stopped: -30% DD reached")
    sys.exit(1)
```

**Expected Impact:**
- Protect from catastrophic loss
- Optimize position size dynamically

**Effort:** Low (1 day)  
**Risk:** Low

---

**P3 Combined Impact:**
- Tick simulation: 0.90× (more realistic, lower profit estimate)
- Capital protection: 1.05× (slightly better sizing)
- **Total: 0.90 × 1.05 = 0.95× (more realistic, not profit boost)**

**Final estimate (P0+P1+P2+P3): $378 → $1,310/fold (3.47×)**

---

## 🎯 Path to $2,000-3,000/Fold

### **Realistic Scenario (Conservative):**

| Phase | Techniques | Cumulative Multiplier | Profit/Fold |
|-------|-----------|----------------------|-------------|
| **Baseline** | Current system | 1.00× | $378 |
| **P0** | Exit + Dynamic risk + Entry | 1.87× | $707 |
| **P1** | Ensemble + Regime + Micro | 2.79× | $1,053 |
| **P2** | RL + Alt data | 3.65× | $1,380 |
| **P3** | Testing + Protection | 3.47× | **$1,310** |

**Conservative estimate: $1,300/fold** ⚠️ Below target

---

### **Aggressive Scenario (Optimistic + Compound):**

**Option 1: Intra-Fold Compounding**
- Start: $200
- After P0 improvements, best folds can do +300% → $800
- Compound within fold: $800 → $2,400 by end (3× again)
- Risk: High DD (need careful management)

**Option 2: Higher Base Risk (5-7%)**
- Current: 3% risk
- Increase to 5%: 1.67× profit → $1,310 × 1.67 = **$2,187/fold**
- Risk: DD could reach 20%+

**Option 3: Leverage (2×)**
- Use 2× leverage on high confidence trades
- Effective capital: $400 instead of $200
- Profit: $1,310 × 2 = **$2,620/fold**
- Risk: DD × 2 = 18-20%

**Recommended Approach:**
```
Phase 1-2 (1-2 months): Implement P0 + P1
  → Target: $1,000/fold
  → Validate: 2-3 months live paper

Phase 3 (month 3-4): Add P2 + optimize
  → Target: $1,300-1,500/fold
  → Validate: 1-2 months live paper

Phase 4 (month 5+): Aggressive options
  → Intra-fold compound OR
  → Higher risk (5%) OR
  → Leverage (1.5-2×)
  → Target: $2,000-3,000/fold
  → Start small, scale gradually
```

---

## 📊 Risk Management for $2,000+ Target

### **Capital Protection Mechanism:**

```python
class CapitalProtection:
    def __init__(self, initial_capital=200):
        self.initial = initial_capital
        self.current = initial_capital
        self.peak = initial_capital
        self.trailing_stop = 0.70  # Stop at -30% from peak
        
    def update(self, current_balance):
        self.current = current_balance
        
        # Update peak
        if current_balance > self.peak:
            self.peak = current_balance
        
        # Check trailing stop
        if current_balance < self.peak * self.trailing_stop:
            return 'STOP_TRADING'  # Lost 30% from peak
        
        # Check absolute stop
        if current_balance < self.initial * 0.50:
            return 'STOP_TRADING'  # Lost 50% of initial
        
        return 'CONTINUE'
    
    def get_max_risk(self):
        """
        Scale down risk as we approach stops
        """
        dd_from_peak = (self.peak - self.current) / self.peak
        
        if dd_from_peak > 0.20:  # 20% DD
            return 0.02  # Reduce to 2%
        elif dd_from_peak > 0.15:  # 15% DD
            return 0.025  # Reduce to 2.5%
        else:
            return 0.05  # Normal 5%
```

---

## 🗓️ Implementation Timeline

### **Month 1-2: Quick Wins (P0)**
- Week 1: Trailing SL implementation + testing
- Week 2: Dynamic risk based on confidence
- Week 3: Entry quality improvements
- Week 4: Integration + WF validation
- **Target: $700/fold**

### **Month 3-4: Advanced Models (P1)**
- Week 5-6: Ensemble models (LightGBM + XGBoost + CatBoost)
- Week 7: Regime-based strategy switching
- Week 8: Market microstructure features
- Week 9: Integration + WF validation
- **Target: $1,000-1,300/fold**

### **Month 5-6: RL & Optimization (P2)**
- Week 10-11: RL agent training + testing
- Week 12: Alternative data integration
- Week 13-14: Live paper testing
- **Target: $1,500-1,800/fold**

### **Month 7+: Scale to Target (P3 + Aggressive)**
- Week 15-16: Tick simulation + stress testing
- Week 17: Select aggressive option (compound/risk/leverage)
- Week 18-20: Live validation at small scale
- Week 21+: Scale to full deployment
- **Target: $2,000-3,000/fold**

---

## ⚠️ Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Over-optimization** | Backtest great, live fails | Out-of-sample validation, 3-month paper |
| **High DD** | -30% drawdown | Hard stop at -30%, reduce risk in DD |
| **Model drift** | Performance degrades | Monitor feature stability, retrain monthly |
| **Market regime change** | Strategy stops working | Regime detection, multiple strategies |
| **Leverage risk** | Large losses | Start 1.5×, gradually to 2×, monitor daily |
| **Psychological** | Panic during DD | Automated system, no manual intervention |

---

## 📈 Success Metrics

### **Phase Gates:**

**P0 Gate (After Month 2):**
- ✅ Avg profit > $700/fold
- ✅ Max DD < 12%
- ✅ Win rate > 45%
- ✅ Sharpe > 4.5

**P1 Gate (After Month 4):**
- ✅ Avg profit > $1,000/fold
- ✅ Max DD < 13%
- ✅ Win rate > 50%
- ✅ Ensemble agreement > 70%

**P2 Gate (After Month 6):**
- ✅ Avg profit > $1,500/fold
- ✅ Max DD < 15%
- ✅ RL agent > baseline
- ✅ Paper 3 months successful

**Production Gate (Month 7+):**
- ✅ Avg profit > $2,000/fold for 2 consecutive months
- ✅ Max DD < 18%
- ✅ Live-WF gap < 20%
- ✅ No catastrophic losses

---

## 📋 Technology Checklist

### **Models:**
- [x] LightGBM (current)
- [ ] Ensemble (LightGBM + XGBoost + CatBoost) - P1
- [ ] RL (PPO/SAC) - P2
- [ ] HMM (regime classification) - Optional

### **Simulation:**
- [x] Dynamic slippage (current)
- [x] Trailing SL (M1 bar-by-bar, current)
- [ ] Tick-by-tick replay - P3
- [ ] Partial fills - P3
- [ ] Market impact (Almgren-Chriss) - P3
- [ ] Latency injection - P3

### **Testing:**
- [x] Paper trading (current)
- [x] A/B testing framework (current)
- [ ] Stress testing (COVID, war, etc.) - P3
- [ ] Synthetic data (GARCH/GAN) - Optional

### **Metrics:**
- [x] Sharpe ratio (current)
- [x] Calmar ratio (current)
- [x] Turnover (current)
- [ ] Feature stability tracking - P1
- [ ] Kelly criterion - P3

### **Features:**
- [x] ICT + Wyckoff (67 features, current)
- [x] Regime detection (5 features, current)
- [ ] Market microstructure (10 features) - P1
- [ ] Alternative data (5 features) - P2

---

## 🎯 Final Recommendation

**Conservative Path (Lower Risk):**
1. Implement P0 + P1 (4 months)
2. Target: $1,000-1,300/fold
3. Use 4-5% risk on high confidence
4. **Achievable within 6 months**

**Aggressive Path (Higher Risk, Higher Reward):**
1. Implement P0 + P1 + P2 (6 months)
2. Target: $1,500/fold base
3. Add intra-fold compounding OR 2× leverage
4. Final target: $2,000-3,000/fold
5. **Achievable within 9-12 months**

**My Recommendation: Start Conservative**
- Implement P0 first (2 months)
- Validate on live paper (1 month)
- If successful ($700+ consistent), proceed to P1
- Only consider aggressive options after 6+ months of stable performance
- **Goal: $2,000/fold by Month 10-12**

**Reasoning:**
- Going from $378 → $2,000 (5.3×) is very aggressive
- Market conditions change, need time to validate
- Better to grow slowly and sustainably than chase unrealistic targets
- Once you hit $1,300/fold consistently, can explore leverage/compounding

---

**Status:** Roadmap created  
**Next Step:** Choose path (Conservative vs Aggressive) and start with P0 implementation  
**ETA to Target:** 10-12 months (conservative) or 6-8 months (aggressive with higher risk)
