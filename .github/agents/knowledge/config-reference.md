# Config File Reference — YAML Structure

## Tổng quan

Tất cả configs nằm trong `configs/` directory:

```
configs/
├── acc1_v14pp_profit.yaml       # Main config acc1
├── acc1_v14pp_composite.yaml    # Composite strategy
├── acc1_r147tight.yaml          # Tight SL variant
├── live_acc1.yaml               # Live trading config
├── live_acc2.yaml               # Live trading config account 2
├── xauusd_combo133_best.yaml    # Combo133 ensemble
└── benchmarks/                  # Old configs for comparison
```

**Naming convention**:
- `acc1` / `acc2` → account ID
- `v14pp` → version 14 post-processing
- `profit` / `composite` / `r147tight` → variant name
- `live_` prefix → production live config
- `xauusd_` prefix → symbol-specific config

---

## Config Structure

### Section 1: Metadata
```yaml
name: "acc1_v14pp_profit"
version: "1.0"
description: "Account 1 — V14++ profit-focused with tight RR"
symbol: "XAUUSD"
timeframe: "M1"
created: "2026-04-01"
```

**Purpose**: Identify config, track versions.

---

### Section 2: Data
```yaml
data:
  source: "csv"  # csv | mt5 | api
  path: "data/XAUUSD_M1_2022-2026.csv"
  
  # Date range for backtest
  start_date: "2024-01-01"
  end_date: "2026-05-01"
  
  # Normalization
  normalize: true
  normalize_method: "standard"  # standard | minmax | robust
```

**Options**:
- `source="mt5"` → fetch live from MT5 terminal
- `source="api"` → fetch from external API
- `normalize=false` → raw prices (not recommended for ML)

---

### Section 3: Features
```yaml
features:
  # Technical indicators
  indicators:
    - rsi_14
    - rsi_28
    - ema_10
    - ema_50
    - macd
    - atr
    - bollinger_bands
    - stochastic
    - adx
    
  # ICT concepts
  ict:
    enabled: true
    fvg: true         # Fair Value Gap
    bisi: true        # Buy/Sell Side Imbalance
    order_blocks: true
    liquidity_sweeps: true
    
  # Wyckoff
  wyckoff:
    enabled: true
    phase_detection: true
    volume_analysis: true
    spring_detection: true
    
  # Price action
  price_action:
    swing_points: true
    support_resistance: true
    candlestick_patterns: true
    
  # Time-based
  time_features:
    hour: true
    day_of_week: true
    session: true  # London, NY, Asia
    
  # Lag features
  lags:
    enabled: true
    periods: [1, 2, 5, 10]
    columns: ["close", "volume", "atr"]
```

**Disable features**: Set `enabled: false` để tắt nhóm features.

---

### Section 4: Labels
```yaml
labels:
  method: "fixed_rr"  # fixed_rr | adaptive | swing
  
  # Fixed RR params
  reward_risk: 2.0    # TP = entry ± (SL × 2.0)
  sl_multiplier: 1.5  # SL = ATR × 1.5
  
  # Forward bars để tìm TP
  forward_bars: 10
  
  # Class balance
  balance_classes: true
  balance_method: "smote"  # smote | oversample | undersample
```

**Label methods**:
- `fixed_rr` → TP/SL based on ATR × RR
- `adaptive` → TP/SL adapt theo volatility regime
- `swing` → TP = next swing high/low

---

### Section 5: Model
```yaml
model:
  type: "lightgbm"  # lightgbm | xgboost | randomforest | lstm | ensemble
  
  # LightGBM params
  params:
    objective: "multiclass"
    num_class: 3
    metric: "multi_logloss"
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 100
    min_data_in_leaf: 50
    feature_fraction: 0.9
    bagging_fraction: 0.8
    bagging_freq: 5
    lambda_l1: 0.1
    lambda_l2: 0.1
    verbosity: -1
    
  # Feature selection
  feature_selection:
    enabled: true
    method: "importance"  # importance | correlation | recursive
    top_k: 50  # Giữ top 50 features
    
  # Early stopping
  early_stopping:
    enabled: true
    rounds: 50
```

**Model types**:
- `lightgbm` — Default, fastest
- `xgboost` — Alternative GBDT
- `randomforest` — Baseline
- `lstm` — Deep learning (cần nhiều data)
- `ensemble` — Combo133

---

### Section 6: Combo133 (Ensemble)
```yaml
combo133:
  enabled: false  # true = train ensemble
  n_models: 133
  voting: "soft"  # soft = avg probabilities | hard = majority vote
  threshold: 0.6  # Min confidence to enter trade
  
  # Param grid
  param_variations:
    num_leaves: [15, 31, 47, 63]
    learning_rate: [0.01, 0.03, 0.05, 0.07, 0.1]
    min_data_in_leaf: [20, 50, 100, 150]
    feature_fraction: [0.7, 0.8, 0.9, 1.0]
    lambda_l1: [0, 0.1, 0.5]
    lambda_l2: [0, 0.1, 0.5]
```

**Enable**: Set `enabled: true` + CLI flag `--combo133`.

---

### Section 7: Walk-forward
```yaml
walkforward:
  enabled: true
  
  # Window sizes
  train_bars: 18000  # ~12.5 days M1
  test_bars: 6000    # ~4 days M1
  step_bars: 6000    # Step = test_bars (không overlap)
  
  # Cache
  cache_features: true
  cache_path: ".cache/features/"
  
  # Retrain threshold
  retrain_on_metric: "sharpe"
  retrain_threshold: 1.0  # Nếu Sharpe < 1.0 → retrain
```

**Window tuning**:
- `train_bars` lớn → model có nhiều data, nhưng quá xa test period
- `test_bars` lớn → ít folds, test stable hơn
- `step_bars < test_bars` → overlap giữa folds (không khuyến khích)

---

### Section 8: Risk Management
```yaml
risk:
  # Position sizing
  risk_per_trade: 0.030  # 3% balance per trade
  max_positions: 3
  max_daily_trades: 10
  
  # Stop loss / Take profit
  sl_atr_multiplier: 1.5  # SL = ATR × 1.5
  tp_rr_ratio: 2.0        # TP = SL × 2.0
  
  # Trailing stop
  trailing_stop:
    enabled: true
    activation: 1.0  # Activate khi profit >= 1.0 × ATR
    step: 0.5        # Trail 0.5 × ATR
    
  # Drawdown protection
  max_daily_dd: 0.10     # Stop nếu DD > 10% trong 1 ngày
  max_overall_dd: 0.15   # Stop nếu DD > 15% overall
  
  # Adaptive sizing
  adaptive_sizing:
    enabled: true
    method: "kelly"  # kelly | fixed | volatility
    kelly_fraction: 0.5  # Half-Kelly (conservative)
```

**Adaptive sizing methods**:
- `kelly` → Kelly Criterion: `size = (p × b - q) / b` where p=win_rate, q=1-p, b=avg_win/avg_loss
- `fixed` → Always `risk_per_trade`
- `volatility` → Scale size inverse với ATR

---

### Section 9: Slippage & Costs
```yaml
slippage:
  enabled: false  # Enable với --slippage flag
  
  # Fixed component
  spread: 0.2  # pips
  
  # Dynamic component
  slippage_factor: 0.5  # Extra slippage = ATR × 0.5
  
  # Costs
  commission_per_lot: 0  # $0 (spread-only broker)
  swap_long: -0.5        # Swap per day (long)
  swap_short: 0.2        # Swap per day (short)
```

**Enable slippage**: CLI `--slippage` hoặc set `enabled: true`.

---

### Section 10: Execution (Live Trading)
```yaml
execution:
  # MT5 connection
  mt5:
    enabled: true
    terminal_path: "/Applications/MetaTrader 5.app"
    login: 123456789
    password: "secret"
    server: "Broker-Server"
    
  # Order params
  magic_number: 20260401
  comment: "acc1_v14pp_profit"
  deviation: 10  # Max price deviation (points)
  
  # Retry logic
  max_retries: 3
  retry_delay: 1  # seconds
```

**MT5 setup**: Cần install MT5 terminal + login trước.

---

### Section 11: Logging & Monitoring
```yaml
logging:
  level: "INFO"  # DEBUG | INFO | WARNING | ERROR
  
  # File logging
  file:
    enabled: true
    path: "logs/acc1_v14pp_profit.log"
    max_size: "10MB"
    backup_count: 5
    
  # Console
  console:
    enabled: true
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
  # MLflow
  mlflow:
    enabled: true
    tracking_uri: "http://localhost:5000"
    experiment_name: "acc1_v14pp"
```

---

### Section 12: Alerts
```yaml
alerts:
  # Telegram bot
  telegram:
    enabled: false
    bot_token: "your_bot_token"
    chat_id: "your_chat_id"
    
  # Email
  email:
    enabled: false
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    sender: "trader@example.com"
    password: "app_password"
    recipients: ["you@example.com"]
    
  # Triggers
  triggers:
    - event: "trade_opened"
      notify: ["telegram", "email"]
    - event: "daily_dd_exceeded"
      notify: ["telegram"]
    - event: "error"
      notify: ["telegram", "email"]
```

---

## Live Config Example: `live_acc1.yaml`

```yaml
name: "live_acc1"
version: "1.0"
description: "Live trading — Account 1"
symbol: "XAUUSD"
timeframe: "M1"

data:
  source: "mt5"
  lookback_bars: 500  # Tính features từ 500 bars gần nhất

model:
  load_from: "outputs/acc1_combo133_202604_model.pkl"
  threshold: 0.65  # Higher threshold cho live (more conservative)

risk:
  risk_per_trade: 0.020  # 2% (giảm từ 3% backtest)
  max_positions: 2       # Giảm từ 3
  max_daily_trades: 5
  max_daily_dd: 0.08
  max_overall_dd: 0.12
  
  trailing_stop:
    enabled: true
    activation: 1.2
    step: 0.3

execution:
  mt5:
    enabled: true
    login: 123456789
    password: "secret"
    server: "ICMarkets-Live"
  magic_number: 20260401
  comment: "live_acc1_v14pp"

logging:
  level: "INFO"
  file:
    enabled: true
    path: "logs/live_acc1.log"
  mlflow:
    enabled: true
    tracking_uri: "http://vps-ip:5000"

alerts:
  telegram:
    enabled: true
    bot_token: "real_bot_token"
    chat_id: "real_chat_id"
  triggers:
    - event: "trade_opened"
      notify: ["telegram"]
    - event: "daily_dd_exceeded"
      notify: ["telegram"]

# Paper trading mode
paper_trading: false  # Set true để test trước
```

---

## Config Validation

**CLI command**:
```bash
python scripts/validate_config.py configs/acc1_v14pp_profit.yaml
```

**Checks**:
- [ ] YAML syntax valid
- [ ] All required fields present
- [ ] Value ranges valid (e.g., `risk_per_trade` ∈ [0, 1])
- [ ] Model file exists (nếu `load_from`)
- [ ] MT5 credentials valid (nếu `mt5.enabled`)
- [ ] Paths exist (data, logs, cache)

---

## Override via CLI

**Example**: Override risk_per_trade via command line
```bash
python scripts/walkforward_ict_wyckoff.py \
  configs/acc1_v14pp_profit.yaml \
  --risk-pct 0.025 \
  --max-positions 2 \
  --combo133
```

**Priority**: CLI args > config file > defaults

---

## Best Practices

### ✅ DO
1. **Version control configs** — git commit after every change
2. **One config per experiment** — don't reuse same config
3. **Document changes** — update `description` field
4. **Validate before run** — use `validate_config.py`
5. **Keep backups** — save old configs to `configs/benchmarks/`

### ❌ DON'T
1. **Don't hardcode credentials** — use env vars hoặc separate secrets file
2. **Don't commit secrets** — add `*.secret.yaml` to `.gitignore`
3. **Don't change config mid-run** — restart nếu cần thay đổi
4. **Don't copy-paste entire configs** — dùng inheritance (nếu có)

---

## Config Inheritance (Advanced)

**Pattern**: Base config + override
```yaml
# configs/base.yaml
name: "base"
symbol: "XAUUSD"
timeframe: "M1"

model:
  type: "lightgbm"
  params:
    num_leaves: 31
    learning_rate: 0.05

risk:
  risk_per_trade: 0.03
  max_positions: 3
```

```yaml
# configs/acc1_variant.yaml
extends: "base.yaml"  # Inherit from base

name: "acc1_variant"  # Override name

risk:
  risk_per_trade: 0.025  # Override risk
  # max_positions: 3 inherited from base
```

**Implementation**: Load base config, merge với variant config.

---

## Config Templates

### Template 1: Conservative Live
```yaml
risk:
  risk_per_trade: 0.015  # 1.5%
  max_positions: 1
  max_daily_trades: 3
  trailing_stop:
    enabled: true
    activation: 1.5
    step: 0.5
```

### Template 2: Aggressive Backtest
```yaml
risk:
  risk_per_trade: 0.05  # 5%
  max_positions: 5
  max_daily_trades: 20
  trailing_stop:
    enabled: false
```

### Template 3: Paper Trading
```yaml
paper_trading: true
risk:
  risk_per_trade: 0.01  # Small size để test
  max_positions: 1
alerts:
  telegram:
    enabled: true
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `configs/acc1_v14pp_profit.yaml` | Main backtest config |
| `configs/live_acc1.yaml` | Production live config |
| `configs/xauusd_combo133_best.yaml` | Ensemble config |
| `scripts/validate_config.py` | Config validator |
| `src/xauusd_ai/config_loader.py` | YAML loader utility |
