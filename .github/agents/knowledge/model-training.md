# Model Training Guide — LightGBM & Combo133

## Tổng quan

**Base model**: LightGBM (GBDT - Gradient Boosted Decision Trees)

**Training method**: Walk-forward rolling window với cross-validation

**Ensemble**: Combo133 = 133 configs với hyperparams khác nhau

---

## LightGBM Basics

### Tại sao chọn LightGBM?

| Tiêu chí | LightGBM | XGBoost | RandomForest |
|----------|----------|---------|--------------|
| Speed | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Memory | Low | Medium | High |
| Accuracy | High | High | Medium |
| Overfitting resistance | Good | Good | Excellent |

**Ưu điểm cho XAUUSD**:
- Nhanh: train 6000 bars trong ~10s
- Hiệu quả với tabular features (OHLCV + indicators)
- Hỗ trợ categorical features (regime, session)
- Feature importance rõ ràng → dễ debug

**Nhược điểm**:
- Cần tuning cẩn thận → dễ overfit
- Không capture temporal patterns tốt như LSTM
- Không tự tạo features (phải engineer bằng tay)

---

## Training Pipeline

### Step 1: Feature Engineering
```python
# features/dataset.py
def create_features(df):
    """
    Tạo ~150 features từ OHLCV
    """
    # Technical indicators (80 features)
    df['rsi_14'] = talib.RSI(df['close'], 14)
    df['rsi_28'] = talib.RSI(df['close'], 28)
    df['ema_10'] = talib.EMA(df['close'], 10)
    df['ema_50'] = talib.EMA(df['close'], 50)
    df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
    # ... 70+ more indicators
    
    # ICT concepts (20 features)
    df['fvg_bullish'] = detect_fair_value_gap(df, 'bullish')
    df['fvg_bearish'] = detect_fair_value_gap(df, 'bearish')
    df['order_block_bull'] = detect_order_block(df, 'bull')
    df['bisi'] = calculate_bisi(df)
    df['liquidity_sweep'] = detect_liquidity_sweep(df)
    # ...
    
    # Wyckoff phases (10 features)
    df['wyckoff_phase'] = classify_wyckoff_phase(df)
    df['volume_climax'] = detect_volume_climax(df)
    df['spring_detected'] = detect_spring(df)
    # ...
    
    # Price action (15 features)
    df['swing_high'] = identify_swing_highs(df)
    df['swing_low'] = identify_swing_lows(df)
    df['resistance_near'] = distance_to_resistance(df)
    df['support_near'] = distance_to_support(df)
    # ...
    
    # Time-based (10 features)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_london_session'] = (df['hour'] >= 8) & (df['hour'] <= 16)
    df['is_ny_session'] = (df['hour'] >= 13) & (df['hour'] <= 21)
    # ...
    
    # Lag features (20 features)
    for i in [1, 2, 5, 10]:
        df[f'close_lag_{i}'] = df['close'].shift(i)
        df[f'volume_lag_{i}'] = df['volume'].shift(i)
    # ...
    
    return df
```

**⚠️ Lookahead Bias Check**:
- Mọi feature phải tính từ past → present
- KHÔNG dùng `shift(-1)` (future bar)
- KHÔNG dùng info chỉ biết sau khi exit

---

### Step 2: Label Creation
```python
# features/dataset.py
def create_labels(df, forward_bars=10, reward_risk=2.0):
    """
    Label: 1 = long, -1 = short, 0 = no trade
    
    Logic:
    - Tìm peak trong next 10 bars
    - Nếu peak > entry + (atr × RR) → label = 1
    - Nếu trough < entry - (atr × RR) → label = -1
    - Else → label = 0
    """
    labels = []
    
    for i in range(len(df) - forward_bars):
        future_highs = df['high'].iloc[i+1:i+forward_bars+1]
        future_lows = df['low'].iloc[i+1:i+forward_bars+1]
        entry = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        tp_long = entry + atr * reward_risk
        tp_short = entry - atr * reward_risk
        
        if future_highs.max() >= tp_long:
            labels.append(1)
        elif future_lows.min() <= tp_short:
            labels.append(-1)
        else:
            labels.append(0)
    
    return labels
```

**Variants**:
- `forward_bars=10` → short-term trades
- `forward_bars=50` → swing trades
- `reward_risk=1.5` vs `2.0` vs `3.0` → affects label distribution

---

### Step 3: Train-Test Split (Walk-forward)
```python
# scripts/walkforward_ict_wyckoff.py
def walk_forward_split(df, test_bars=6000, step_bars=6000):
    """
    Rolling window:
    Fold 1: train [0:18000], test [18000:24000]
    Fold 2: train [6000:24000], test [24000:30000]
    Fold 3: train [12000:30000], test [30000:36000]
    ...
    """
    folds = []
    n = len(df)
    
    for test_start in range(18000, n, step_bars):
        test_end = min(test_start + test_bars, n)
        train_start = max(0, test_start - 18000)  # 18k bars = ~3 months
        
        fold = {
            'train': df.iloc[train_start:test_start],
            'test': df.iloc[test_start:test_end]
        }
        folds.append(fold)
    
    return folds
```

**Trade-offs**:
- `test_bars` lớn → ít folds, test lâu hơn, stable hơn
- `test_bars` nhỏ → nhiều folds, test nhanh, noisy hơn
- `train_window=18000` bars → ~12.5 days M1 → đủ data không quá xa

---

### Step 4: Train Model
```python
# model/trainer.py
def train_lgbm(X_train, y_train, params):
    """
    Train single LightGBM model
    """
    # Convert to LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Params
    lgbm_params = {
        'objective': 'multiclass',
        'num_class': 3,  # long, short, neutral
        'metric': 'multi_logloss',
        'num_leaves': params['num_leaves'],
        'learning_rate': params['learning_rate'],
        'feature_fraction': params['feature_fraction'],
        'bagging_fraction': params['bagging_fraction'],
        'bagging_freq': params['bagging_freq'],
        'min_data_in_leaf': params['min_data_in_leaf'],
        'lambda_l1': params['lambda_l1'],
        'lambda_l2': params['lambda_l2'],
        'verbosity': -1
    }
    
    # Train
    model = lgb.train(
        lgbm_params,
        train_data,
        num_boost_round=params['n_estimators'],
        valid_sets=[train_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    return model
```

**Hyperparameters quan trọng**:
| Param | Range | Effect |
|-------|-------|--------|
| `num_leaves` | 15-63 | Complexity: cao → overfit |
| `learning_rate` | 0.01-0.1 | Speed vs accuracy |
| `min_data_in_leaf` | 20-200 | Regularization: cao → underfit |
| `lambda_l1` | 0-1 | L1 regularization |
| `lambda_l2` | 0-1 | L2 regularization |

---

### Step 5: Hyperparameter Tuning
**Method 1: Grid Search (slow)**
```python
param_grid = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_data_in_leaf': [20, 50, 100]
}

for params in itertools.product(*param_grid.values()):
    model = train_lgbm(X_train, y_train, dict(zip(param_grid.keys(), params)))
    score = evaluate(model, X_val, y_val)
    # Track best
```

**Method 2: Random Search (faster)**
```python
from scipy.stats import randint, uniform

param_dist = {
    'num_leaves': randint(15, 64),
    'learning_rate': uniform(0.01, 0.09),
    'min_data_in_leaf': randint(20, 200)
}

for i in range(100):  # 100 random trials
    params = {k: v.rvs() for k, v in param_dist.items()}
    model = train_lgbm(X_train, y_train, params)
    score = evaluate(model, X_val, y_val)
```

**Method 3: Bayesian Optimization (smartest)**
```python
from skopt import BayesSearchCV

search_space = {
    'num_leaves': (15, 63),
    'learning_rate': (0.01, 0.1, 'log-uniform'),
    'min_data_in_leaf': (20, 200)
}

opt = BayesSearchCV(
    lgb.LGBMClassifier(),
    search_space,
    n_iter=50,
    cv=3
)
opt.fit(X_train, y_train)
best_params = opt.best_params_
```

→ **Combo133 dùng method nào?** Hybrid: grid search trên core params + random variations.

---

## Combo133 Ensemble

### What is Combo133?

**Definition**: Ensemble of 133 LightGBM models with different:
1. Hyperparameters (num_leaves, learning_rate, etc.)
2. Feature subsets (dropout random 10-20% features)
3. Label methods (forward_bars = 5, 10, 20, reward_risk = 1.5, 2.0, 3.0)

**Why ensemble?**
- Giảm variance → robust hơn
- Mỗi model catch patterns khác nhau
- Voting → reduce false positives

**Structure**:
```python
# 133 configs generated như thế này
configs = []

for num_leaves in [15, 31, 63]:
    for learning_rate in [0.01, 0.05, 0.1]:
        for min_data_in_leaf in [20, 50, 100]:
            for feature_fraction in [0.8, 0.9, 1.0]:
                for forward_bars in [5, 10, 20]:
                    config = {
                        'num_leaves': num_leaves,
                        'learning_rate': learning_rate,
                        'min_data_in_leaf': min_data_in_leaf,
                        'feature_fraction': feature_fraction,
                        'forward_bars': forward_bars
                    }
                    configs.append(config)

# Total: 3 × 3 × 3 × 3 × 3 = 243 → giảm xuống 133 bằng filtering
```

**Filtering logic**:
- Loại configs quá extreme (num_leaves=15 + learning_rate=0.1 → unstable)
- Loại configs duplicate performance
- Giữ lại 133 configs diverse nhất

---

### Training Combo133

**File**: `configs/xauusd_combo133_best.yaml`

```yaml
combo133:
  enabled: true
  n_models: 133
  voting: 'soft'  # soft = average probabilities
  threshold: 0.6  # At least 60% models vote cùng direction
  
  base_params:
    objective: 'multiclass'
    num_class: 3
    metric: 'multi_logloss'
    n_estimators: 100
    
  param_variations:
    num_leaves: [15, 31, 47, 63]
    learning_rate: [0.01, 0.03, 0.05, 0.07, 0.1]
    min_data_in_leaf: [20, 50, 100, 150, 200]
    feature_fraction: [0.7, 0.8, 0.9, 1.0]
    lambda_l1: [0, 0.1, 0.5, 1.0]
    lambda_l2: [0, 0.1, 0.5, 1.0]
```

**Training script**:
```python
# scripts/walkforward_ict_wyckoff.py --combo133
models = []

for i, config in enumerate(combo133_configs):
    print(f"Training model {i+1}/133...")
    
    # Train
    model = train_lgbm(X_train, y_train, config)
    
    # Validate
    score = model.score(X_val, y_val)
    
    # Save
    models.append({
        'model': model,
        'config': config,
        'score': score
    })
    
    # Log to MLflow
    mlflow.log_param(f"combo133_model_{i}_num_leaves", config['num_leaves'])
    mlflow.log_metric(f"combo133_model_{i}_score", score)

# Save ensemble
with open('outputs/combo133_models.pkl', 'wb') as f:
    pickle.dump(models, f)
```

**Time cost**: 133 models × 10s/model = **~22 phút per fold**.

---

### Prediction with Combo133

```python
# model/ensemble.py
def predict_combo133(models, X):
    """
    Voting scheme: soft voting
    """
    predictions = []
    
    for model_dict in models:
        model = model_dict['model']
        proba = model.predict_proba(X)  # [n_samples, 3]
        predictions.append(proba)
    
    # Average probabilities
    avg_proba = np.mean(predictions, axis=0)  # [n_samples, 3]
    
    # Apply threshold
    max_proba = avg_proba.max(axis=1)
    threshold = 0.6
    
    final_pred = np.where(
        max_proba >= threshold,
        avg_proba.argmax(axis=1),  # 0=neutral, 1=long, 2=short
        0  # Neutral nếu confidence < 60%
    )
    
    return final_pred, avg_proba
```

**Threshold tuning**:
- `threshold=0.5` → nhiều signals, win rate thấp hơn
- `threshold=0.7` → ít signals, win rate cao hơn
- `threshold=0.6` → balance

---

## Feature Importance Analysis

### Top 20 Features (từ combo133)
```
1. atr (14.2%) — Volatility quan trọng nhất
2. rsi_14 (8.5%)
3. ema_10_distance (7.3%) — Distance from price to EMA10
4. bisi (6.8%) — ICT concept
5. fvg_bullish (5.9%)
6. macd_signal (5.2%)
7. hour (4.8%) — Session timing
8. volume_surge (4.3%)
9. swing_high (3.7%)
10. order_block_bull (3.5%)
... (10 more)
```

**Insights**:
- Volatility features (ATR, BB width) dominant
- ICT concepts (BISI, FVG, order blocks) strong
- Time-based features (hour, session) important
- Lag features ít quan trọng → model không dựa quá nhiều vào past bars

**Stability check**:
```python
# Compare top 10 features across 5 folds
fold_1_top10 = ['atr', 'rsi_14', 'ema_10_distance', ...]
fold_2_top10 = ['atr', 'rsi_14', 'bisi', ...]
fold_3_top10 = ['atr', 'ema_10_distance', 'rsi_14', ...]

# Intersection
common = set(fold_1_top10) & set(fold_2_top10) & set(fold_3_top10)
print(f"Common top features: {common}")

# Nếu < 7/10 common → not stable, overfitting
```

---

## Model Persistence

### Save model
```python
# After training
model_meta = {
    'model': model,
    'feature_names': X_train.columns.tolist(),
    'feature_importances': dict(zip(X_train.columns, model.feature_importances_)),
    'params': params,
    'train_score': train_score,
    'test_score': test_score,
    'fold_id': fold_id,
    'train_period': {'start': train_start, 'end': train_end},
    'test_period': {'start': test_start, 'end': test_end},
    'timestamp': datetime.now().isoformat()
}

# Save model
with open('outputs/acc1_combo133_202604_model.pkl', 'wb') as f:
    pickle.dump(model_meta, f)

# Save metadata
with open('outputs/acc1_combo133_202604_meta.json', 'w') as f:
    json.dump({k: v for k, v in model_meta.items() if k != 'model'}, f, indent=2)
```

### Load model
```python
# In production
with open('outputs/acc1_combo133_202604_model.pkl', 'rb') as f:
    model_meta = pickle.load(f)

model = model_meta['model']
feature_names = model_meta['feature_names']

# Predict
X_live = prepare_features(live_data)[feature_names]  # Must match train order
prediction = model.predict(X_live)
```

---

## MLflow Tracking

### Log training run
```python
import mlflow

with mlflow.start_run(run_name=f"combo133_fold_{fold_id}"):
    # Log params
    mlflow.log_param("n_models", 133)
    mlflow.log_param("fold_id", fold_id)
    mlflow.log_param("train_bars", len(X_train))
    mlflow.log_param("test_bars", len(X_test))
    
    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("profit_factor", pf)
    mlflow.log_metric("max_drawdown", max_dd)
    mlflow.log_metric("sharpe_ratio", sharpe)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("outputs/fold_1_trades.csv")
    mlflow.log_artifact("outputs/fold_1_report.html")
```

### View runs
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```

Navigate to http://localhost:5000 → compare runs.

---

## Training Best Practices

### ✅ DO
1. **Always use walk-forward** — không train trên toàn bộ data
2. **Log everything to MLflow** — params, metrics, artifacts
3. **Check feature importance stability** — corr ≥ 0.75
4. **Calculate Sharpe/Calmar mỗi fold** — không chỉ P&L
5. **Save model + metadata** — để reproduce
6. **Version control configs** — git commit mỗi experiment

### ❌ DON'T
1. **Don't train/test split randomly** — temporal leak
2. **Don't tune on test set** — overfit to test
3. **Don't ignore feature lookahead** — invalidates results
4. **Don't deploy without validation** — paper trade first
5. **Don't over-optimize hyperparams** — diminishing returns
6. **Don't use too many features** — curse of dimensionality

---

## Debugging Poor Performance

### Checklist khi model perform kém

**Symptom: Train accuracy cao, test accuracy thấp**
- → Overfitting
- Fix: tăng `min_data_in_leaf`, giảm `num_leaves`, thêm regularization

**Symptom: Both train/test accuracy thấp**
- → Underfitting hoặc bad features
- Fix: thêm features, giảm regularization, increase `num_leaves`

**Symptom: Backtest tốt, live kém**
- → Lookahead bias hoặc execution assumptions sai
- Fix: audit features, add slippage model, paper trade

**Symptom: Win rate cao nhưng P&L thấp**
- → Risk/reward không cân bằng
- Fix: adjust TP/SL, check RR ratio

**Symptom: P&L tốt nhưng DD lớn**
- → Position sizing quá aggressive
- Fix: giảm `risk_pct`, add DD-based sizing

**Symptom: Feature importance thay đổi nhiều giữa các fold**
- → Unstable features, có thể lookahead
- Fix: remove unstable features, check calculation

---

## Performance Benchmarks

**Hardware**: MacBook Air M1, 8GB RAM

| Operation | Time (approx) |
|-----------|---------------|
| Load 100k bars M1 data | 2s |
| Feature engineering (150 features) | 15s |
| Train single LightGBM | 10s |
| Train Combo133 (133 models) | 22 min |
| Backtest 6000 bars | 3s |
| Full WF (10 folds, combo133) | ~4 hours |

**Optimization tips**:
- Cache features: `--cache` flag → save computed features to disk
- Parallel training: `n_jobs=-1` trong LightGBM
- Reduce features: top 50 features → 2x speed
- GPU: LightGBM có GPU support nhưng gain nhỏ với dataset này

---

## Files Reference

| File | Purpose |
|------|---------|
| `model/trainer.py` | Core training logic |
| `model/ensemble.py` | Combo133 implementation |
| `features/dataset.py` | Feature engineering |
| `scripts/walkforward_ict_wyckoff.py` | WF orchestration |
| `configs/xauusd_combo133_best.yaml` | Combo133 config |
| `outputs/acc1_combo133_202604_model.pkl` | Trained model |
| `outputs/acc1_combo133_202604_meta.json` | Model metadata |
