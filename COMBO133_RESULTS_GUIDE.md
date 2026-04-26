# XAUUSD AI — Combo #133: Kết Quả, Cấu Hình & Hướng Dẫn Tái Tạo

> **Ngày đóng băng:** 26 tháng 4, 2026  
> **Kết quả:** Walk-Forward Jan 2024 → Apr 2026 | **28 folds | 27/28 profitable | Tổng P&L = +$78,204.48**  
> **Vốn mỗi fold:** $200 (không compound giữa các fold)

---

## 1. Tóm Tắt Kết Quả

| Chỉ số | Giá trị |
|--------|---------|
| Tổng số lệnh | 11,668 |
| Số fold profitable | 27/28 (96.4%) |
| Tổng P&L (tất cả fold) | **+$78,204.48** |
| WR trung bình | ~46% |
| Max DD trung bình mỗi fold | ~-17% |
| Profit Factor | >1.5 (thắng nhờ RR=3.5, không cần WR cao) |
| Thời gian test | Jan 2024 → Apr 2026 |

### Kết quả từng fold (thực tế từ `outputs/combo133_trades.csv`)

| Fold | Kỳ test | Lệnh | WR | P&L | Bal kết thúc | Max DD |
|------|---------|------|----|-----|--------------|--------|
| 1 | 2023-12-20 → 2024-01-22 | 569 | 50% | +$13,622 | $13,822 | -12.3% |
| 2 | 2024-01-23 → 2024-02-21 | 581 | 45% | +$2,414 | $2,614 | -19.3% |
| 3 | 2024-02-21 → 2024-03-22 | 517 | 44% | +$715 | $916 | -16.8% |
| 4 | 2024-03-22 → 2024-04-24 | 452 | 49% | +$3,233 | $3,433 | -16.0% |
| 5 | 2024-04-24 → 2024-05-08 | 190 | 43% | +$179 | $379 | -20.1% |
| 6 | 2024-05-24 → 2024-06-14 | 327 | 33% | +$64 | $264 | -20.1% |
| 7 | 2024-06-25 → 2024-07-01 | 90 | 43% | +$101 | $301 | -22.8% |
| 8 | 2024-07-25 → 2024-08-23 | 450 | 46% | +$2,503 | $2,703 | -17.1% |
| 9 | 2024-08-23 → 2024-09-24 | 573 | 51% | +$4,305 | $4,505 | -16.3% |
| 10 | 2024-09-24 → 2024-10-24 | 590 | 47% | +$7,149 | $7,349 | -16.8% |
| 11 | 2024-10-24 → 2024-11-22 | 480 | 49% | +$1,970 | $2,170 | -14.8% |
| 12 | 2024-11-25 → 2024-12-26 | 500 | 43% | +$697 | $897 | -14.7% |
| 13 | 2024-12-26 → 2025-01-27 | 557 | 45% | +$1,411 | $1,611 | -17.8% |
| 14 | 2025-01-27 → 2025-02-18 | 349 | 49% | +$1,281 | $1,481 | -21.0% |
| 15 | 2025-02-26 → 2025-03-28 | 591 | 51% | +$14,348 | $14,548 | -16.3% |
| 16 | 2025-03-28 → 2025-04-30 | 239 | 47% | +$765 | $965 | -13.1% |
| 17 | 2025-04-30 → 2025-05-30 | 433 | 51% | +$1,076 | $1,276 | -18.0% |
| 18 | 2025-05-30 → 2025-06-30 | 592 | 47% | +$1,441 | $1,641 | -14.2% |
| 19 | 2025-06-30 → 2025-07-17 | 323 | 39% | +$36 | $236 | -20.1% |
| 20 | 2025-07-31 → 2025-08-29 | 563 | 42% | +$971 | $1,171 | -19.1% |
| 21 | 2025-08-29 → 2025-09-11 | 201 | 49% | +$439 | $639 | -20.3% |
| 22 | 2025-09-30 → 2025-10-29 | 330 | 55% | +$4,926 | $5,126 | -18.7% |
| 23 | 2025-10-30 → 2025-12-01 | 534 | 49% | +$1,436 | $1,636 | -13.6% |
| 24 | 2025-12-01 → 2025-12-04 | 86 | 29% | **-$2** | $198 | -20.8% |
| 25 | 2026-01-02 → 2026-01-16 | 246 | 39% | +$35 | $235 | -20.1% |
| 26 | 2026-02-02 → 2026-03-04 | 499 | 46% | +$995 | $1,195 | -16.6% |
| 27 | 2026-03-04 → 2026-04-06 | 507 | 52% | +$11,907 | $12,107 | -15.8% |
| 28 | 2026-04-06 → 2026-04-24 | 299 | 36% | +$185 | $385 | -17.0% |

> Fold 24 là fold duy nhất thua (-$1.61, gần như hòa vốn).

---

## 2. Combo #133 Params

Các tham số này hardcoded trong `scripts/show_combo133_daily.py`:

```python
# ─── Walk-Forward ─────────────────────────────────────────────────────
CONFIG        = Path("configs/xauusd_combo133_best.yaml")
STARTING_BAL  = 200.0          # Vốn ban đầu mỗi fold — KHÔNG compound
TRAIN_BARS    = 30_000          # ~104 ngày M5 data để train
TEST_BARS     = 6_000           # ~20 ngày M5 data để test
STEP_BARS     = 6_000           # Bước nhảy = TEST_BARS (không overlap)
TEST_START    = "2024-01-01"    # Chỉ tính fold có test_end >= ngày này

# ─── Signal Filter ────────────────────────────────────────────────────
MIN_CONF      = 0.70            # Ngưỡng xác suất tối thiểu để vào lệnh
REQ_TREND     = False           # KHÔNG yêu cầu trend alignment
MIN_STRAT     = 0.00            # KHÔNG lọc theo strategy_score
BLOCKED       = [3, 15, 17, 22, 23]   # Block giờ UTC này
D1_GATE       = False           # D1 gate TẮT (đã test: bật thì -87% P&L)
```

---

## 3. Model Ensemble

### Kiến trúc

```
VotingClassifier (soft voting)
├── HistGradientBoostingClassifier  [weight=3]
│     max_iter=1000, lr=0.01, max_depth=7, min_samples_leaf=20
│     l2_reg=1.0, max_bins=128
│     early_stopping=True, val_frac=0.1, n_iter_no_change=40
├── RandomForestClassifier          [weight=2]
│     n_estimators=200, max_depth=12, min_samples_leaf=15
│     max_features="sqrt", class_weight="balanced"
└── ExtraTreesClassifier            [weight=1]
      n_estimators=200, max_depth=14, min_samples_leaf=10
      max_features="sqrt", class_weight="balanced"
```

### Pipeline mỗi fold

```
1. StandardScaler.fit_transform(X_train)
2. Feature Selection: RF scout 80 trees → giữ top 70% importance
3. Sample Weights:
   - class_weight: 2× cho minority class (positive)
   - time_decay: exp với half-life = 40% len(train)
   - combined: class_w × time_w (normalized mean=1)
4. Threshold Search (trên validation split 30% cuối train):
   - HGB nhanh train → tìm threshold max precision×√recall
   - Điều kiện: recall >= 5%, precision >= PREC_FLOOR (từ config)
5. VotingClassifier.fit(X_train_selected, y_train, sample_weight)
6. Predict: proba = model.predict_proba(X_test)[:,1]
   → vào lệnh khi proba >= best_threshold
```

---

## 4. Config File

**File:** `configs/xauusd_combo133_best.yaml` (bản gốc: `configs/acc1_v14pp_profit.yaml`)

### Các thông số quan trọng nhất

```yaml
strategy:
  blocked_hours_utc: [3, 15, 17, 22, 23]   # ← Combo #133
  sideway_min_confidence: 0.85
  volatile_min_confidence: 0.85
  d1_trend_gate: false                       # ← TẮT

risk:
  risk_per_trade: 0.04             # 4%/lệnh
  stop_loss_atr_multiple: 1.5      # SL = 1.5× ATR
  take_profit_rr: 3.5              # TP = 3.5R
  partial_tp_rr: 1.2               # Chốt 50% tại 1.2R
  partial_tp_pct: 0.5
  daily_loss_limit_pct: 0.12       # Kill switch 12%/ngày
  consecutive_loss_cooldown_bars: 12
  anti_martingale_factor: 0.7
  volatility_risk_scaling_enabled: false    # ← TẮT

execution:
  trailing_sl:
    enabled: true
    breakeven_at_rr: 0.5           # Dời SL về BE khi đạt 0.5R
    activation_rr: 1.0
    trail_atr_multiple: 1.0

training:
  max_train_bars: 30000
  feature_selection_drop_pct: 30   # Giữ top 70% features
```

---

## 5. Code Tham Chiếu — Toàn Bộ Script (Phòng Khi Code Bị Sửa)

### 5.1 Đoạn override settings trong `show_combo133_daily.py`

Đây là block quan trọng nhất — nó override settings YAML trước khi chạy backtest mỗi fold:

```python
_sim_settings = settings_full.model_copy(deep=True)
_sim_settings.training.backtest_initial_balance        = STARTING_BAL   # 200.0
_sim_settings.risk.min_confidence                      = MIN_CONF        # 0.70
_sim_settings.strategy.sideway_min_confidence          = MIN_CONF        # 0.70
_sim_settings.strategy.volatile_min_confidence         = MIN_CONF        # 0.70
_sim_settings.strategy.require_trend_alignment         = REQ_TREND       # False
_sim_settings.strategy.min_strategy_score              = MIN_STRAT       # 0.00
_sim_settings.strategy.sideway_min_strategy_score      = MIN_STRAT       # 0.00
_sim_settings.strategy.strong_volatility_min_strategy_score = MIN_STRAT  # 0.00
_sim_settings.strategy.blocked_hours_utc               = BLOCKED         # [3,15,17,22,23]
_sim_settings.strategy.d1_trend_gate                   = False
```

> **Quan trọng:** MIN_CONF=0.70 override `min_confidence: 0.85` trong YAML. Đây là điểm khác biệt lớn nhất giữa WF config và YAML gốc.

### 5.2 Trade side logic

```python
# trade_side: buy khi strategy_score >= 0, sell khi < 0
fold_sim_df["trade_side"] = np.where(fold_sim_df["strategy_score"] >= 0, "buy", "sell")
# D1_GATE = False → không filter gì thêm
```

### 5.3 Sample weights (mỗi fold)

```python
_pos_c, _neg_c = int(y_tr.sum()), int(len(y_tr) - y_tr.sum())
if _pos_c > 10 and _neg_c > 10:
    _pw = 2.0 * _neg_c / _pos_c                    # class imbalance weight
    _class_w = np.where(y_tr == 1, _pw, 1.0)
    _n = len(y_tr)
    _decay_half = _n * 0.4                          # half-life = 40% train len
    _time_w = np.exp(np.log(2) * np.arange(_n) / _decay_half)
    _time_w /= _time_w.mean()
    _sw_tr = (_class_w * _time_w)
    _sw_tr /= _sw_tr.mean()                         # normalize mean=1
```

### 5.4 Threshold search (mỗi fold)

```python
# Train nhanh HGB trên 70% đầu train set
_thr_hgb = HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.02, max_depth=6, min_samples_leaf=25,
    l2_regularization=1.0, max_bins=128,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=42,
)
_thr_hgb.fit(X_tr[:int(n*0.70)], y_tr[:int(n*0.70)], sample_weight=...)

# Sweep threshold tìm max precision × √recall
for thr in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP):
    # THRESHOLD_MIN/MAX/STEP/PREC_FLOOR lấy từ settings.training (yaml)
    preds = (val_proba >= thr).astype(int)
    if preds.sum() < 3: continue
    if recall < 0.05: continue
    if precision < PREC_FLOOR: continue
    score = precision * sqrt(recall)   # maximize this
```

### 5.5 Feature selection (mỗi fold)

```python
_scout = RandomForestClassifier(
    n_estimators=80, max_depth=8, min_samples_leaf=20,
    class_weight="balanced", n_jobs=-1, random_state=42
)
_scout.fit(X_tr, y_tr, sample_weight=_sw_tr)
_feat_mask = importances >= np.percentile(importances, 30)  # giữ top 70%
if _feat_mask.sum() < 10: _feat_mask = all_true             # fallback
```

### 5.6 Final ensemble (mỗi fold)

```python
_hgb = HistGradientBoostingClassifier(
    max_iter=1000, learning_rate=0.01, max_depth=7, min_samples_leaf=20,
    l2_regularization=1.0, max_bins=128,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=40, random_state=42,
)
_rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=15,
    max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42
)
_et = ExtraTreesClassifier(
    n_estimators=200, max_depth=14, min_samples_leaf=10,
    max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42
)
model = VotingClassifier(
    estimators=[("hgb", _hgb), ("rf", _rf), ("et", _et)],
    voting="soft", weights=[3, 2, 1]
)
model.fit(X_tr_selected, y_tr, sample_weight=_sw_tr)
```

### 5.7 Feature columns (56 features — `src/xauusd_ai/features/dataset.py`)

```python
FEATURE_COLUMNS = [
    # D1 context (1)
    "daily_bias",
    # H4 ICT structure (9)
    "h4_bos", "h4_choch", "h4_fvg", "h4_order_block", "h4_displacement",
    "h4_ehl", "h4_market_structure_bias", "h4_ict_confluence", "h4_premium_discount",
    # H1 Wyckoff context (3)
    "hourly_bias", "vsa_signal", "wyckoff_spring_signal",
    # M15 execution (15)
    "trend_alignment", "rsi", "macd_hist", "atr_ratio", "range_efficiency",
    "liquidity_sweep", "order_flow_proxy", "wyckoff_phase", "volatility_regime",
    "session_return", "tick_volume_zscore", "spread_points",
    "strategy_score", "kill_zone_flag", "judas_swing_signal",
    # News (5)
    "news_impact_ahead", "news_hours_ahead", "news_hours_since",
    "news_surprise_gold", "news_is_blackout",
    # Price structure & momentum (5)
    "bb_position", "stoch_k", "stoch_kd_diff", "rsi_slope", "price_vs_h4_sma",
    # v2 advanced (8)
    "adx", "candle_body_ratio", "obv_slope", "price_roc",
    "multi_tf_consensus", "h4_h1_bias_agree", "atr_percentile",
    "time_hour_sin", "time_hour_cos",
    # v3 microstructure (6)
    "pullback_depth", "atr_expansion", "wick_rejection",
    "volume_surge", "close_in_range", "macd_accel",
    # v4 institutional (3)
    "vol_delta_momentum", "inst_candle_score", "swing_failure",
    # v5 ICT (2+)
    "h4_breaker_block", "silver_bullet_setup",
]
```

---

## 6. Model Có Cần Train Lại Không? Bao Lâu Train Lại?

### Câu trả lời ngắn

**Có — model PHẢI train lại định kỳ.** Combo #133 dùng Walk-Forward tức là model được re-train mỗi khi window dịch chuyển. Không có một model cố định nào tồn tại lâu dài.

### Tại sao phải retrain?

| Lý do | Giải thích |
|-------|------------|
| **Market regime shift** | Gold thay đổi vol regime theo macro (Fed, geopolitics) — model cũ predict sai context mới |
| **Feature drift** | ATR, RSI, ADX distribution thay đổi theo thời gian |
| **Data mới** | Mỗi thanh nến M5 mới là thêm signal cho model học |
| **Fold window** | WF dùng 30,000 bars (~104 ngày) train → sau 6,000 bars (~20 ngày) phải shift và retrain |

### Lịch retrain đề xuất

| Tần suất | Khi nào | Hành động |
|----------|---------|-----------|
| **Mỗi ~20 ngày** (WF cadence) | Tương đương 1 fold TEST_BARS=6,000 M5 | Chạy lại `show_combo133_daily.py` với data mới |
| **Khi thị trường đổi regime** | VIX vàng spike >40%, Fed pivot, war event | Retrain ngay, không chờ lịch |
| **Khi WR live < 35% trong 2 tuần** | Model có thể đã stale | Retrain + review config |
| **Khi max DD live > 15% trong tháng** | Kill switch + retrain | Tạm dừng trade, retrain, verify |

### Quy trình retrain chuẩn

```bash
# 1. Cập nhật data mới (nếu chưa có đến hôm nay)
# Thêm data M5 mới vào src/xauusd_ai/real_data/XAUUSDm_M5.csv

# 2. Chạy lại WF để kiểm tra model mới có còn profitable không
cd "/Users/dodoannang/Documents/Thạc sĩ MSE/Trade Indicator"
source .venv/bin/activate
python scripts/show_combo133_daily.py 2>&1 | tee /tmp/retrain_$(date +%Y%m%d).txt

# 3. Kiểm tra kết quả 3 fold gần nhất (fold 26-28)
grep "Fold 2[6-9]\|Fold [3-9]" /tmp/retrain_$(date +%Y%m%d).txt

# 4. Nếu 3 fold gần nhất đều profitable → model vẫn tốt
# Nếu 2+ fold gần nhất thua → cần review config
```

---

## 7. Đảm Bảo WF Sát Với Live Trading

### Trạng thái sync WF ↔ Live (cập nhật 26/04/2026)

> ✅ **`configs/live_acc1.yaml` đã được sync hoàn toàn với Combo #133.**

| Param | WF (Combo #133) | Live (`live_acc1.yaml`) | Trạng thái |
|-------|----------------|------------------------|------------|
| `min_confidence` | 0.70 | **0.70** | ✅ Khớp |
| `sideway_min_confidence` | 0.70 | **0.70** | ✅ Khớp |
| `volatile_min_confidence` | 0.70 | **0.70** | ✅ Khớp |
| `risk_per_trade` | 4% | **4%** | ✅ Khớp |
| `max_open_positions` | 3 | **3** | ✅ Khớp |
| `d1_trend_gate` | false | **false** | ✅ Khớp |
| `volatility_risk_scaling_enabled` | false | **false** | ✅ Khớp |
| `require_trend_alignment` | false | false | ✅ Khớp |
| `blocked_hours_utc` | [3,15,17,22,23] | [3,15,17,22,23] | ✅ Khớp |
| `partial_tp_rr` | 1.2 | 1.2 | ✅ Khớp |
| `take_profit_rr` | 3.5 | 3.5 | ✅ Khớp |
| `anti_martingale_factor` | 0.7 | 0.7 | ✅ Khớp |
| `daily_loss_limit_pct` | 0.12 | 0.12 | ✅ Khớp |
| `trailing_sl.breakeven_at_rr` | 0.5 | 0.5 | ✅ Khớp |

### Những thay đổi đã áp dụng vào `live_acc1.yaml`

```yaml
# ĐÃ SỬA (26/04/2026) để match Combo #133:

strategy:
  sideway_min_confidence: 0.70    # ← đổi từ 0.78
  volatile_min_confidence: 0.70   # ← đổi từ 0.78
  d1_trend_gate: false            # ← thêm mới

risk:
  min_confidence: 0.70            # ← đổi từ 0.78
  risk_per_trade: 0.04            # ← đổi từ 0.05
  max_open_positions: 3           # ← đổi từ 2
  volatility_risk_scaling_enabled: false  # ← thêm mới
```

> **Lưu ý khi restart bot:** Confidence thấp hơn + max_pos=3 sẽ tăng số lệnh live ~30-40% so với trước. Kiểm tra margin MT5 đủ cho 3 lệnh đồng thời trước khi restart.

### Checklist trước khi restart bot sau mỗi lần retrain

- [x] `min_confidence = 0.70` ← **ĐÃ SET**
- [x] `risk_per_trade = 0.04` ← **ĐÃ SET**
- [x] `max_open_positions = 3` ← **ĐÃ SET**
- [x] `d1_trend_gate: false` ← **ĐÃ SET**
- [x] `volatility_risk_scaling_enabled: false` ← **ĐÃ SET**
- [x] `blocked_hours_utc: [3, 15, 17, 22, 23]` ← **ĐÃ SET**
- [x] `partial_tp_rr: 1.2`, `partial_tp_pct: 0.5` ← **ĐÃ SET**
- [x] `take_profit_rr: 3.5` ← **ĐÃ SET**
- [x] `trailing_sl.enabled: true`, `breakeven_at_rr: 0.5` ← **ĐÃ SET**
- [ ] Data M5 đã cập nhật đến ngày hôm nay ← kiểm tra trước mỗi lần chạy
- [ ] Chạy WF test 1 fold cuối và profitable ← kiểm tra trước mỗi lần deploy

---

## 8. Hướng Dẫn Tái Tạo Kết Quả

### Yêu cầu

- Python ≥ 3.10, scikit-learn ≥ 1.3
- Data: `src/xauusd_ai/real_data/XAUUSDm_M5.csv` (cần có đến ít nhất 2026-04-24)
- venv đã cài đầy đủ requirements

### Bước 1 — Chuẩn bị môi trường

```bash
cd "/path/to/Trade Indicator"
source .venv/bin/activate

# Kiểm tra
python -c "import sklearn; print(sklearn.__version__)"   # >= 1.3
wc -l src/xauusd_ai/real_data/XAUUSDm_M5.csv            # ~477,970 rows
```

### Bước 2 — Chỉnh script (nếu cần)

Mở `scripts/show_combo133_daily.py`, đảm bảo các dòng đầu như sau:

```python
CONFIG        = Path("configs/xauusd_combo133_best.yaml")   # ← file đóng băng
STARTING_BAL  = 200.0
TRAIN_BARS    = 30_000
TEST_BARS     = 6_000
STEP_BARS     = 6_000
TEST_START    = "2024-01-01"
MIN_CONF      = 0.70
REQ_TREND     = False
MIN_STRAT     = 0.00
BLOCKED       = [3, 15, 17, 22, 23]
D1_GATE       = False
```

### Bước 3 — Chạy Walk-Forward

```bash
cd "/Users/dodoannang/Documents/Thạc sĩ MSE/Trade Indicator"
source .venv/bin/activate
python scripts/show_combo133_daily.py 2>&1 | tee /tmp/combo133_result.txt
```

**Thời gian chạy:** ~15–30 phút (28 folds × train ensemble)

**Output:**
- `/tmp/combo133_result.txt` — log terminal
- `outputs/combo133_trades.csv` — 11,668 lệnh raw

### Bước 4 — Tạo báo cáo

```bash
python scripts/gen_trade_report_md.py
```

**Output:** `outputs/combo133_trade_report.md` (chia theo ngày)

### Bước 5 — Export HTML / PDF

```bash
pandoc outputs/combo133_trade_report.md \
  -o outputs/combo133_trade_report.html \
  --standalone \
  --metadata title="Báo Cáo Giao Dịch Combo #133" \
  --css - <<'EOF'
body { font-family: Arial, sans-serif; font-size: 11px; margin: 20px; }
table { border-collapse: collapse; width: 100%; font-size: 10px; margin-bottom: 8px; }
th, td { border: 1px solid #ccc; padding: 3px 6px; }
th { background: #2c3e50; color: white; }
tr:nth-child(even) { background: #f9f9f9; }
h2 { font-size: 15px; color: #34495e; border-bottom: 2px solid #3498db; margin-top: 30px; }
h3 { font-size: 12px; color: #555; background: #ecf0f1; padding: 5px 8px; border-left: 4px solid #27ae60; }
EOF

open outputs/combo133_trade_report.html
# Sau đó: Cmd+P → Save as PDF
```

---

## 9. Auto Data Update + Retrain — `scripts/auto_update_retrain.py`

Script tự động hóa toàn bộ vòng lặp dữ liệu → model:

### 9.1 Workflow

```
1. Tải M5 mới từ yfinance (GC=F × 0.9952) → append vào XAUUSDm_M5.csv
2. Tải M15, H1, D1 trực tiếp từ yfinance (không resample)
   Tính toán lại H4 bằng cách resample từ H1 (yfinance không có interval 4h)
3. Đọc state file: outputs/combo133_retrain_state.json
4. Nếu new_bars >= 6000 → retrain fold mới (30k train + 6k test)
5. Backtest profit gate: chỉ lưu model mới nếu P&L > 0
6. Cập nhật state + ghi log vào outputs/combo133_retrain_log.jsonl
7. Gửi cảnh báo kết quả qua Telegram + Email
```

### 9.2 Cách Dùng

```bash
# Full run (fetch data + check retrain)
python scripts/auto_update_retrain.py

# Chỉ tải data, không retrain
python scripts/auto_update_retrain.py --data-only

# Chỉ check retrain, không tải data
python scripts/auto_update_retrain.py --train-only

# Ép retrain ngay (bỏ qua ngưỡng 6000 bars)
python scripts/auto_update_retrain.py --force-retrain

# Đặt balance ban đầu cho profit gate
python scripts/auto_update_retrain.py --starting-bal 500
```

### 9.3 Cron Job (macOS/Linux)

```cron
# Chạy mỗi ngày lúc 06:00 UTC
0 6 * * * cd "/path/to/Trade Indicator" && .venv/bin/python scripts/auto_update_retrain.py >> logs/auto_retrain.log 2>&1
```

### 9.4 Cấu Hình Cảnh Báo (Telegram + Email)

Set environment variables trước khi chạy script:

```bash
# ── Telegram ────────────────────────────────────────────────────
# Uu tiên RETRAIN_TELEGRAM_TOKEN; nếu không có sẽ fallback sang TELEGRAM_BOT_TOKEN
export RETRAIN_TELEGRAM_TOKEN="123456789:ABCDEF..."
export RETRAIN_TELEGRAM_CHAT_ID="-100123456789"

# ── Email (Gmail) ────────────────────────────────────────────
# Biến này chỉ cần thiết nếu muốn nhận email (Telegram là đủ)
export RETRAIN_EMAIL_FROM="your@gmail.com"
export RETRAIN_EMAIL_TO="your@gmail.com"
export RETRAIN_SMTP_HOST="smtp.gmail.com"  # mặc định
export RETRAIN_SMTP_PORT="587"             # mặc định
export RETRAIN_SMTP_USER="your@gmail.com"
export RETRAIN_SMTP_PASSWORD="app-password-here"  # Google App Password
```

Gmail App Password: [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)

**Mẫu tin nhắn Telegram sau mỗi retrain:**
```
✅ Combo #133 Retrain Fold #29
Kết quả: ACCEPTED

Test period : 2026-03-10 → 2026-05-14
Net P&L     : +$3,421.50
Trades      : 312  |  WR: 47.0%
ROC-AUC     : 0.5823
Threshold   : 0.71  |  Features: 39
M5 bars     : 484,000

⏰ 2026-05-15 06:00 UTC
```

### 9.5 State File — `outputs/combo133_retrain_state.json`

```json
{
  "last_retrain_m5_count": 477970,
  "last_retrain_date": "2026-04-26T06:00:00+00:00",
  "last_fold": 28,
  "last_result": "ACCEPTED"
}
```

### 9.5 Retrain Log — `outputs/combo133_retrain_log.jsonl`

Mỗi lần retrain ghi 1 dòng JSON:
```json
{
  "timestamp": "2026-05-15T06:00:00+00:00",
  "fold": 29,
  "m5_bars_total": 484000,
  "new_bars": 6030,
  "train_start": "2024-11-01", "train_end": "2026-03-10",
  "test_start": "2026-03-10", "test_end": "2026-05-14",
  "threshold": 0.71, "n_features": 39,
  "roc_auc": 0.5823, "precision": 0.51, "recall": 0.47, "f1": 0.49,
  "n_trades": 312, "win_rate": 0.47, "net_pnl": 3421.50,
  "accepted": true, "result": "ACCEPTED"
}
```

### 9.7 Model Artifacts (trainer.py-compatible format)

| File | Nội dung |
|------|----------|
| `outputs/acc1_v14pp_model.pkl` | VotingClassifier trained on selected features |
| `outputs/acc1_v14pp_model.pkl.bak` | Backup của model trước |
| `outputs/acc1_v14pp_scaler.pkl` | StandardScaler (fit on ALL features) |
| `outputs/acc1_v14pp_model_meta.json` | `decision_threshold`, `feature_mask`, metrics |

> Live bot tự load lại artifacts sau mỗi lần retrain (nếu `live_learning_enabled: true`).

### 9.8 Lưu Ý Kỹ Thuật

- **yfinance M5 giới hạn 60 ngày** → Chạy ít nhất mỗi tháng để không bị gap.  
- **Ratio GC=F → XAUUSDm = 0.9952** (giá CME Futures thấp hơn MT5 spot ~0.48%).  
- **M15 / H1 / D1 được lấy trực tiếp** từ yfinance (không resample từ M5 — giá chính xác hơn).  
- **H4 là ngoại lệ** — resample từ H1 (vì yfinance không có interval 4h; H4 = 4×H1 nên vẫn chính xác).  
- Script dùng **đúng pipeline Combo #133**: sample weights, threshold search, RF feature selection (drop bottom 30%), VotingClassifier HGB×3+RF×2+ET×1.  
- `feature_mask` được lưu trong meta JSON — live bot đọc và apply khi inference.  
- **Cảnh báo tự động**: kết quả mỗi fold gửi qua Telegram và Email ngay sau khi hoàn thành.  
- **Dashboard**: phần "🔁 Combo #133 — Trạng Thái Retrain" hiển thị tiến trình và ETA fold tiếp theo.

---

## 10. Files Liên Quan

| File | Vai trò |
|------|---------|
| `configs/xauusd_combo133_best.yaml` | **Config đóng băng** — YAML đầy đủ |
| `scripts/show_combo133_daily.py` | Script chạy WF + sinh trades CSV |
| `scripts/auto_update_retrain.py` | **Auto data fetch + retrain** |
| `scripts/gen_trade_report_md.py` | Script sinh báo cáo MD theo ngày |
| `outputs/combo133_trades.csv` | **Raw trades** — 11,668 lệnh (28 folds) |
| `outputs/combo133_trade_report.md` | Báo cáo chi tiết theo ngày |
| `outputs/combo133_trade_report.html` | Bản HTML để in PDF |
| `outputs/combo133_retrain_state.json` | State file: last retrain bar count + date |
| `outputs/combo133_retrain_log.jsonl` | Log mỗi lần retrain (JSONL) |
| `src/xauusd_ai/real_data/XAUUSDm_M5.csv` | Dữ liệu M5 (đến 2026-04-24) |

---

## 10. Lưu Ý Quan Trọng

1. **Không compound giữa các fold** — mỗi fold bắt đầu lại với $200. Trong thực tế live trading, balance sẽ tích lũy.

2. **D1 Gate = False** — Đã test D1_GATE=True: P&L giảm từ +$78,204 xuống ~+$9,000 (-87%). Lý do: XAUUSD có xu hướng tăng bền vững 2024-2026, model BUY hoạt động tốt ngay cả trong ngày D1 downtrend.

3. **Volatility Risk Scaling = False** — Model hoạt động tốt nhất khi ATR spike (Fold 1, 15, 27 đều là các kỳ biến động cao). Tắt scaling giúp giữ nguyên size lệnh.

4. **MIN_CONF = 0.70** (không phải 0.85 trong yaml) — Ngưỡng 0.70 áp dụng trong script WF override ngưỡng min_confidence của config. Đây là điểm khác biệt chính so với live config.

5. **WR ~46% vẫn có lãi** — Do TP/SL ratio = 3.5:1. Chỉ cần WR > 22% để hòa vốn. WR 46% cho Profit Factor > 2.0.

6. **Retrain cadence = WF cadence** — Mỗi khi thêm ~20 ngày data mới (6,000 bars M5) là đủ để trigger retrain 1 fold mới. Không cần retrain hàng ngày.

7. **Config live đã sync** — `live_acc1.yaml` đã được cập nhật ngày 26/04/2026: `min_confidence=0.70`, `risk_per_trade=0.04`, `max_open_positions=3`, `d1_trend_gate=false`, `volatility_risk_scaling_enabled=false`. Tất cả 13 params quan trọng đều khớp WF.
