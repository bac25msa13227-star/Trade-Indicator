# Phân tích Gap giữa Backtest và Live Trade

## Tổng quan vấn đề

**Hiện tượng**: Walk-forward backtest cho kết quả tốt hơn live trade ~20-30%.

| Metric | WF Backtest | Live Trade | Gap |
|--------|-------------|------------|-----|
| Profit Factor | 1.8 | 1.2-1.3 | -33% |
| Max Drawdown | 8% | 12-15% | +50% |
| Win Rate | 55% | 48-52% | -7% |
| Avg RR | 2.1 | 1.6 | -24% |

**Impact**: Model được deploy dựa trên backtest expectations, nhưng live không đạt được → risk management sai, sizing sai, P&L không như kỳ vọng.

---

## Nguyên nhân chính

### 1. Execution Assumptions (Quan trọng nhất)

**Backtest hiện tại**:
```python
# backtesting/engine.py
entry_price = df.loc[signal_bar, 'close']  # ❌
exit_price = df.loc[exit_bar, 'close']     # ❌
```

**Thực tế live**:
- Signal bar close = 3200.0
- Market order thực tế fill ở = 3200.3 (slippage +0.3)
- Spread = 0.2
- Latency = 50-200ms → giá đã di chuyển
- → Effective entry = 3200.5

**Gap**: 0.5 / 3200 = 0.015% per trade × 200 trades/month = **3% P&L gap**

#### Slippage sources
| Source | Magnitude | Frequency |
|--------|-----------|-----------|
| Spread | 0.2-0.5 pips | 100% |
| Market impact | 0.1-0.3 pips | High volatility |
| Latency (signal → fill) | 0.2-1.0 pips | Network delay |
| Partial fills | N/A | Rare (XAUUSD liquid) |

#### Solution đang implement
```python
# Thêm vào backtesting/engine.py
def apply_slippage(price, direction, atr, config):
    spread = config.get('spread', 0.2)  # pips
    slippage_factor = config.get('slippage_factor', 0.5)
    
    # Slippage theo volatility
    slippage = spread + atr * slippage_factor
    
    if direction == 'long':
        return price + slippage  # Buy cao hơn
    else:
        return price - slippage  # Sell thấp hơn
```

Flag: `--slippage` để enable/disable so sánh.

---

### 2. Lookahead Bias

**Định nghĩa**: Feature hoặc logic "nhìn trước" — dùng info chỉ có sau khi signal xảy ra.

#### Ví dụ phổ biến

**❌ Tính indicator trên toàn bộ period**:
```python
# features/dataset.py
df['rsi'] = talib.RSI(df['close'], 14)  # ❌ Nhìn hết data
```
Vấn đề: Khi backtest bar 1000, RSI đã được tính với bar 1001, 1002... → leak future.

**✅ Tính indicator rolling**:
```python
df['rsi'] = df['close'].rolling(14).apply(lambda x: talib.RSI(x, 14).iloc[-1])
```

**❌ Filter trades bằng future P&L**:
```python
# Trong WF fold, nếu có logic:
if trade_result > 0:  # ❌ Chỉ biết sau khi exit
    include_in_training = True
```

**✅ Chỉ dùng info tại thời điểm signal**:
```python
if model_score > threshold:  # ✅ Biết trước khi vào lệnh
    enter_trade = True
```

#### Audit checklist
- [ ] Mọi feature trong `dataset.py` được tính rolling window?
- [ ] Indicator không dùng `shift(-1)` (future bar)?
- [ ] Training data không chứa test period info?
- [ ] Model không train trên toàn bộ data rồi mới split?

#### Solution
```python
# Kiểm tra feature stability qua các fold
for fold in wf_folds:
    feature_importance_fold = model.feature_importances_
    log_to_mlflow(feature_importance_fold, fold_id)
    
# Nếu top features thay đổi nhiều giữa các fold → có lookahead bias
```

---

### 3. Overfitting

**Triệu chứng**:
- Train Sharpe = 3.0, Test Sharpe = 1.5 → gap 50%
- Model có 100+ features nhưng chỉ 500 trades training → ratio quá thấp
- Combo133 có 133 configs → quá nhiều hyperparams

**Giải pháp**:

**Feature selection**:
```python
# Chỉ giữ top 20 features quan trọng nhất
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X_train, y_train)
```

**Regularization**:
```python
# LightGBM
params = {
    'min_data_in_leaf': 100,  # Tăng từ 20 → giảm overfit
    'lambda_l1': 0.1,         # L1 regularization
    'lambda_l2': 0.1,         # L2 regularization
}
```

**Cross-validation**:
```python
# Thay vì train 1 model, train 5 models với CV
cv_scores = []
for fold in cv_folds:
    model.fit(fold.train)
    score = model.score(fold.test)
    cv_scores.append(score)
    
# Nếu std(cv_scores) > 0.5 → unstable, overfit
```

---

### 4. Regime Change

**Vấn đề**: Model train trên 2022-2024 (trending market) → live 2025-2026 (ranging market) → performance drop.

**Solution**:

**Regime detection**:
```python
# strategies/hybrid.py
def detect_regime(df, lookback=100):
    """
    Phân loại: trending, ranging, volatile
    """
    returns = df['close'].pct_change()
    volatility = returns.rolling(lookback).std()
    trend_strength = abs(returns.rolling(lookback).mean())
    
    if trend_strength > 0.002 and volatility < 0.01:
        return 'trending'
    elif volatility > 0.02:
        return 'volatile'
    else:
        return 'ranging'
```

**Adaptive strategy**:
```python
regime = detect_regime(df)

if regime == 'ranging':
    # Không trade hoặc giảm size 50%
    max_positions = 1
    risk_pct = 0.015  # Giảm từ 0.03
    
elif regime == 'trending':
    # Aggressive
    max_positions = 3
    risk_pct = 0.03
```

---

## Công cụ phân tích gap

### Script 1: Compare WF vs Live same period
```python
# scripts/compare_wf_live.py
import pandas as pd

# Load WF results fold 2024-01 ~ 2024-03
wf_trades = pd.read_csv('outputs/wf_fold_202401_trades.csv')
wf_pnl = wf_trades['profit'].sum()

# Load live trades same period
live_trades = pd.read_csv('outputs/live_closed_trades_acc1.csv')
live_trades['close_time'] = pd.to_datetime(live_trades['close_time'])
live_period = live_trades[
    (live_trades['close_time'] >= '2024-01-01') & 
    (live_trades['close_time'] <= '2024-03-31')
]
live_pnl = live_period['profit'].sum()

# Calculate gap
gap_pct = (wf_pnl - live_pnl) / wf_pnl * 100
print(f"WF P&L: ${wf_pnl:.2f}")
print(f"Live P&L: ${live_pnl:.2f}")
print(f"Gap: {gap_pct:.1f}%")

# Nếu gap > 25% → investigate deeper
if abs(gap_pct) > 25:
    print("⚠️ Gap quá lớn — check slippage, lookahead bias")
```

### Script 2: Slippage calculator từ live trades
```python
# scripts/calculate_slippage.py
import pandas as pd

live = pd.read_csv('outputs/live_closed_trades_acc1.csv')

# Giả định WF entry = signal bar close
# Live entry = actual fill price (có trong CSV)
# Slippage = live_entry - expected_entry

slippages = []
for idx, trade in live.iterrows():
    signal_price = trade['signal_price']  # Lưu trong CSV
    actual_entry = trade['open_price']
    
    slip = actual_entry - signal_price
    slip_pips = slip / 0.01  # XAUUSD 1 pip = 0.01
    slippages.append(slip_pips)

avg_slippage = np.mean(slippages)
print(f"Avg slippage: {avg_slippage:.2f} pips")
print(f"Std slippage: {np.std(slippages):.2f} pips")

# Dùng avg này làm config cho --slippage flag
```

### Script 3: Feature importance stability
```python
# scripts/check_feature_stability.py
import json
import pandas as pd

# Load feature importance từ MLflow mỗi fold
fold_importances = []
for fold_id in range(1, 10):
    meta_file = f'outputs/fold_{fold_id}_model_meta.json'
    with open(meta_file) as f:
        meta = json.load(f)
    fold_importances.append(meta['feature_importances'])

# Convert to DataFrame
df = pd.DataFrame(fold_importances)

# Tính correlation giữa các fold
corr = df.T.corr()
print(f"Avg correlation: {corr.mean().mean():.2f}")

# Nếu corr < 0.7 → features không stable, có thể lookahead
if corr.mean().mean() < 0.7:
    print("⚠️ Feature importance không stable giữa các fold")
```

---

## Target Metrics sau khi fix

| Metric | Current | Target (sau fix) |
|--------|---------|------------------|
| Backtest-Live Gap | 30% | ≤ 15% |
| Slippage impact | Unknown | Modeled, measured |
| Feature stability corr | Unknown | ≥ 0.75 |
| Regime-aware trading | No | Yes (pause khi ranging) |

---

## Action Items (Priority Order)

### Week 1-2: Quick wins
1. [ ] Implement slippage model: `--slippage` flag
2. [ ] Measure live slippage: `scripts/calculate_slippage.py`
3. [ ] Apply slippage to WF: re-run với estimated slippage
4. [ ] Calculate Sharpe/Calmar mỗi fold: so sánh với live

### Week 3-4: Validation
5. [ ] Audit features: check lookahead bias trong `dataset.py`
6. [ ] Feature stability check: correlation ≥ 0.75
7. [ ] Compare WF vs Live: same period analysis
8. [ ] Paper trading mode: shadow lệnh không vào thật

### Month 2: Strategic improvements
9. [ ] Regime detection: HMM hoặc changepoint
10. [ ] Adaptive sizing: giảm risk khi ranging
11. [ ] A/B test: chạy model mới song song với cũ
12. [ ] Stress test: flash crash scenarios

### Month 3: Advanced
13. [ ] Ensemble models: LightGBM + LSTM
14. [ ] RL fine-tuning: PPO/SAC adjust exit
15. [ ] Tick simulation: replay M1 tick-by-tick
16. [ ] Market microstructure features: spread, volume imbalance

---

## Acceptance Criteria để deploy model mới

✅ **Backtest validation**:
- [ ] WF Sharpe ≥ 1.5 (annualized)
- [ ] Max DD ≤ 15%
- [ ] Profit Factor ≥ 1.3
- [ ] At least 3 consecutive positive folds

✅ **Gap analysis**:
- [ ] Backtest-live gap ≤ 25% trên period có live data
- [ ] Slippage modeled và measured
- [ ] Feature importance stable (corr ≥ 0.75)

✅ **Paper trading**:
- [ ] 2 weeks shadow mode pass
- [ ] P&L match within 20% of WF expectations
- [ ] No unexpected errors or crashes

✅ **Live micro-test**:
- [ ] 2 weeks live với $500 risk
- [ ] P&L positive hoặc match paper trade
- [ ] Max DD không vượt 10% (của $500)

→ Only then: scale up to full size.
