# XAUUSD AI Trading System — Hướng dẫn đầy đủ

> Dành cho AI session mới hoặc developer mới clone project về.  
> Branch chính đang hoạt động: `feature/modelv5-cai-tien`

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Cấu trúc project](#2-cấu-trúc-project)
3. [Cài đặt môi trường](#3-cài-đặt-môi-trường)
4. [Cấu hình .env](#4-cấu-hình-env)
5. [Các lệnh chính](#5-các-lệnh-chính)
6. [Config files](#6-config-files)
7. [Models & Outputs](#7-models--outputs)
8. [Risk Management (Circuit Breaker)](#8-risk-management-circuit-breaker)
9. [Tính năng Strategy](#9-tính-năng-strategy)
10. [Features (56 cột)](#10-features-56-cột)
11. [Kết quả backtest mới nhất](#11-kết-quả-backtest-mới-nhất)
12. [Live trading](#12-live-trading)
13. [Các account đang chạy](#13-các-account-đang-chạy)
14. [Bugs đã fix](#14-bugs-đã-fix)
15. [Workflow chuẩn khi thay đổi code](#15-workflow-chuẩn-khi-thay-đổi-code)

---

## 1. Tổng quan hệ thống

- **Asset**: XAUUSD (Gold) — symbol live: `XAUUSDm` trên Exness MT5
- **Timeframe chính**: M15 (execution), H1/H4/D1 (context)
- **Model**: VotingClassifier (HistGradientBoosting + RandomForest + ExtraTree) + CalibratedClassifierCV
- **Chiến lược**: ICT + Wyckoff + Momentum hybrid
- **Risk**: Dynamic sizing 3–5%, Circuit breaker, Anti-martingale
- **Python**: 3.11+, PYTHONPATH=`src`

---

## 2. Cấu trúc project

```
Trade-Indicator/
├── configs/
│   ├── live_acc2.yaml              # Config chính đang dùng (acc2)
│   ├── live_ict_wyckoff.yaml       # Config acc1 (ICT+Wyckoff model)
│   ├── model2_weekly500.yaml       # Config train model2
│   ├── train_ict_wyckoff_2022_2026.yaml
│   └── settings.example.yaml      # Template có full documentation
│
├── src/xauusd_ai/
│   ├── main.py                     # Entry point: train/backtest/live/...
│   ├── app.py                      # CLI parser
│   ├── config.py                   # Pydantic settings models
│   ├── orchestrator.py             # Điều phối train/backtest/live
│   ├── features/
│   │   ├── dataset.py              # Build feature dataset (56 cols)
│   │   └── indicators.py           # ICT/Wyckoff/technical indicators
│   ├── strategies/
│   │   └── hybrid.py               # HybridStrategy: ICT+Wyckoff+Momentum
│   ├── model/
│   │   ├── trainer.py              # ModelTrainer + calibration
│   │   └── exit_model.py           # ExitModel (khi đóng lệnh sớm)
│   ├── execution/
│   │   ├── risk.py                 # RiskManager + Circuit breaker
│   │   └── mt5_executor.py         # MT5 order placement
│   ├── backtesting/
│   │   └── engine.py               # Backtest engine v5
│   ├── data/
│   │   ├── news_features.py        # Economic calendar (Finnhub/ForexFactory)
│   │   └── mt5_data.py             # MT5 data download
│   └── learning/
│       └── self_learner.py         # Background learning thread
│
├── outputs/                        # Models, reports, logs (gitignored trừ models)
│   ├── model2_weekly500.pkl        # ⭐ Model chính acc2 (106MB, Git LFS)
│   ├── scaler2_weekly500.pkl
│   ├── model_meta2_weekly500.json
│   ├── exit_model_acc2.pkl         # Exit model acc2
│   ├── exit_scaler_acc2.pkl
│   ├── model_ict_wyckoff.pkl       # Model acc1
│   ├── scaler_ict_wyckoff.pkl
│   └── *.json                      # Reports: backtest, walkforward, training
│
├── scripts/
│   ├── retrain_full.py             # Script retrain toàn bộ
│   ├── train_exit_model.py         # Train exit model riêng
│   └── live_runner.py              # Runner script
│
└── .env                            # Credentials (KHÔNG commit)
```

---

## 3. Cài đặt môi trường

### Windows (khuyến nghị)

```powershell
# Clone project
git clone https://github.com/bac25msa13227-star/Trade-Indicator.git
cd Trade-Indicator
git checkout feature/modelv5-cai-tien

# Download models (Git LFS)
git lfs pull

# Tạo virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies (Windows với MT5)
pip install -r requirements-windows.txt

# Hoặc chỉ core (không MT5)
pip install -r requirements-core.txt
```

### Linux/Mac (không có MT5, chỉ train/backtest)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

### Dependencies chính

| Package | Version | Dùng cho |
|---|---|---|
| pandas | >=2.2.3 | Data processing |
| numpy | >=1.26.4 | Numerical |
| scikit-learn | >=1.5.2 | ML model |
| PyYAML | >=6.0.2 | Config |
| pydantic | >=2.9.2 | Settings validation |
| yfinance | >=0.2.54 | Download training data |
| MetaTrader5 | latest | Live trading (Windows only) |
| matplotlib | >=3.9.2 | Plots |

---

## 4. Cấu hình .env

Tạo file `.env` ở root (copy từ `.env.example` nếu có):

```env
# Account 2 (đang chạy live)
MT5_LOGIN_ACC2=433326057
MT5_PASSWORD_ACC2=your_password
MT5_SERVER_ACC2=Exness-MT5Trial7

# Account 1 (ICT Wyckoff)
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server

# Telegram notifications
TELEGRAM_BOT_TOKEN_ACC2=your_bot_token
TELEGRAM_CHAT_ID_ACC2=your_chat_id

# Finnhub (optional - fallback ForexFactory nếu 403)
FINNHUB_API_KEY=your_key
```

> **Quan trọng**: Nếu không có Finnhub key, hệ thống tự fallback sang ForexFactory → Rule-based. Hoàn toàn bình thường.

---

## 5. Các lệnh chính

Tất cả lệnh chạy từ root folder với `PYTHONPATH=src`:

```powershell
$env:PYTHONPATH="src"
```

### Train model

```powershell
python -m xauusd_ai.main train --config configs/live_acc2.yaml
```

- Download data từ Yahoo Finance (XAUUSD=X) + CSV local
- Build 56 features
- Train VotingClassifier + CalibratedClassifierCV
- Optimize threshold trên test set
- Lưu: `outputs/model2_weekly500.pkl`, `outputs/scaler2_weekly500.pkl`
- Thời gian: ~15–20 phút

### Backtest

```powershell
python -m xauusd_ai.main backtest --config configs/live_acc2.yaml
```

- Walk-forward concurrent backtest
- Lưu: `outputs/backtest_report_acc2.json`, `outputs/backtest_trades_acc2.csv`
- Thời gian: ~10 phút

### Walk-forward validation

```powershell
python -m xauusd_ai.main walkforward --config configs/live_acc2.yaml
```

### Paper trading (không vào tiền thật)

```powershell
python -m xauusd_ai.main paper --config configs/live_acc2.yaml
```

### Live trading (kết nối MT5 thật)

```powershell
python -m xauusd_ai.main live --config configs/live_acc2.yaml
```

- Poll MT5 mỗi 120 giây
- Circuit breaker tự động
- Self-learning background thread
- Gửi Telegram alert

### Kiểm tra kết nối MT5

```powershell
python -m xauusd_ai.main mt5-check --config configs/live_acc2.yaml
```

### Xem log live đang chạy

```powershell
Get-Content outputs/live_acc2_stdout.txt -Wait -Tail 30
# hoặc nếu redirect to file:
Get-Content outputs/live_run2.txt -Wait -Tail 30
```

---

## 6. Config files

### `configs/live_acc2.yaml` — Config chính (đang dùng)

| Section | Key param | Giá trị | Ý nghĩa |
|---|---|---|---|
| app | model_path | `outputs/model2_weekly500.pkl` | Model chính |
| app | poll_seconds | 120 | Poll MT5 mỗi 2 phút |
| market | symbol | XAUUSDm | Symbol MT5 |
| strategy | signal_threshold | 0.55 | Ngưỡng confidence tối thiểu |
| strategy | silver_bullet_enabled | true | ICT Silver Bullet windows |
| strategy | adx_gate_enabled | true | ADX trend filter |
| risk | risk_per_trade | 0.03 | 3% base risk |
| circuit_breaker | daily_loss_limit_pct | 0.06 | Dừng nếu -6% trong ngày |
| circuit_breaker | max_drawdown_kill_pct | 0.15 | Kill switch -15% |

### `configs/live_ict_wyckoff.yaml` — Config acc1

- Model: `outputs/model_ict_wyckoff.pkl`
- MT5 login: env var `MT5_LOGIN`

### `configs/settings.example.yaml` — Template đầy đủ

Có documentation cho tất cả params. Đọc file này khi cần biết option nào có sẵn.

---

## 7. Models & Outputs

### Models hiện có

| File | Train date | Dùng bởi | Size |
|---|---|---|---|
| `model2_weekly500.pkl` | 21/03/2026 01:00 | live_acc2.yaml ⭐ | 106MB |
| `scaler2_weekly500.pkl` | 21/03/2026 | acc2 | ~3KB |
| `model_ict_wyckoff.pkl` | trước | live_ict_wyckoff.yaml | 2.6MB |
| `model_acc2.pkl` | 18/03/2026 | không dùng (cũ) | 1MB |
| `exit_model_acc2.pkl` | - | acc2 exit | 1.2MB |
| `exit_scaler_acc2.pkl` | - | acc2 exit | ~2KB |

### Model meta

```json
// outputs/model_meta2_weekly500.json
{
  "decision_threshold": 0.71,   // threshold tối ưu từ backtest
  "feature_columns": [...]       // 56 features
}
```

### Reports

| File | Nội dung |
|---|---|
| `training_report_acc2.json` | Accuracy, precision, recall, roc_auc |
| `backtest_report_acc2.json` | Full backtest metrics |
| `walkforward_report_acc2.json` | Walk-forward fold results |
| `risk_peak_balance.json` | Peak balance cho drawdown calculation |
| `live_status_acc2.json` | Trạng thái live hiện tại |

---

## 8. Risk Management (Circuit Breaker)

Tất cả tính năng này live trong `src/xauusd_ai/execution/risk.py`, cấu hình trong section `circuit_breaker` của YAML:

```yaml
circuit_breaker:
  daily_loss_limit_pct: 0.06        # Thua >6% vốn trong ngày → dừng cả ngày
  max_drawdown_kill_pct: 0.15       # Drawdown >15% → kill switch (restart thủ công)
  consecutive_loss_pause_count: 3   # Thua 3 lần liên tiếp → pause
  consecutive_loss_cooldown_bars: 8 # Nghỉ 8 nến M15 (= 2 giờ)
  anti_martingale_factor: 0.6       # Mỗi lần thua: risk × 0.6
  anti_martingale_max_reductions: 3 # Tối đa giảm 3 lần (~21% risk gốc)
  max_total_exposure_pct: 0.09      # Tổng tất cả lệnh mở ≤ 9% vốn

partial_tp:
  partial_tp_enabled: true
  partial_tp_rr: 1.0                # Đóng 50% lệnh tại 1R profit
  partial_tp_pct: 0.5               # Đóng 50%
```

### State persistence

Circuit breaker state được lưu vào JSON và reset mỗi ngày tự động. Nếu kill switch triggered → cần restart process thủ công.

---

## 9. Tính năng Strategy

### Silver Bullet ICT

```yaml
silver_bullet_enabled: true
silver_bullet_windows_utc: [[3, 4], [10, 11], [14, 15]]
silver_bullet_confidence_boost: 0.03  # +0.03 confidence trong giờ này
```

Các giờ tốt nhất để giao dịch theo ICT: 03–04 UTC (London open), 10–11 UTC (NY open overlap), 14–15 UTC (NY afternoon).

### ADX Gate

```yaml
adx_gate_enabled: true
adx_min_trend: 18.0   # Chỉ trade khi ADX >= 18 (có xu hướng đủ rõ)
```

Block giao dịch khi thị trường đang sideway (ADX thấp).

### Regime-specific confidence

```yaml
sideway_min_confidence: 0.65      # Cần confidence cao hơn khi sideway
volatile_min_confidence: 0.60     # Cần confidence cao hơn khi biến động mạnh
```

### Blocked hours

```yaml
blocked_hours_utc: [22, 23]             # Block rollover (spread rộng)
blocked_weekday_hours_utc:
  Friday: [20, 21, 22, 23]             # Block cuối tuần
```

---

## 10. Features (56 cột)

Model dùng 56 features chia thành các nhóm:

### ICT/Wyckoff features (15+)
- `h4_bos`, `h4_choch` — Break of Structure, Change of Character
- `h4_fvg` — Fair Value Gap (float [-1, +1], decay 10 bars)
- `h4_order_block` — Order Block
- `h4_displacement` — Displacement candle
- `h4_market_structure_bias` — Bias H4
- `h4_ict_confluence` — ICT confluence score
- `h4_premium_discount` — Premium/Discount zones
- `daily_bias`, `hourly_bias` — Bias theo timeframe
- `wyckoff_spring_signal` — Wyckoff Spring pattern
- `vsa_signal` — Volume Spread Analysis
- `swing_failure_pattern` — ICT Sweep + reverse

### Momentum features (10+)
- `rsi`, `macd_hist`, `atr_ratio`
- `volume_delta_momentum` — Buy/sell pressure từ candle structure × volume
- `institutional_candle_score` — Body ratio × vol ratio × size ratio
- `trend_alignment` — EMA 50/200 alignment

### News features (5)
- `news_impact_score` — Impact của tin tức gần nhất
- `news_direction` — Hướng kỳ vọng từ tin
- `hours_since_news` — Thời gian từ tin gần nhất
- `news_blackout` — Đang trong vùng blackout hay không
- `upcoming_news_score` — Tin sắp tới

### Session features (5+)
- `session_spread_mult` — Asian=1.5x, London=1.0x, NY=1.2x, Rollover=2.0x
- `is_london`, `is_new_york`, `is_asian` — Session flags
- `silver_bullet_session` — Đang trong Silver Bullet window

---

## 11. Kết quả backtest mới nhất

> Chạy ngày 21/03/2026, config `live_acc2.yaml`, model `model2_weekly500.pkl`

| Metric | Giá trị |
|---|---|
| Starting balance | $200 |
| Ending balance | $77,684 |
| Return | +38,742% |
| Trades | 1,112 |
| Win rate | 65.11% |
| Profit factor | 1.87 |
| Max drawdown | -40.18% |
| Sharpe-like | 8.78 |
| Period | 2024-12-04 → 2026-03-02 (453 ngày) |

> **Lưu ý**: Backtest kết quả rất tốt — trong live trading, performance thực tế sẽ thấp hơn do slippage, spread thực, và điều kiện thị trường thay đổi.

---

## 12. Live trading

### Quy trình khởi động

1. Mở MetaTrader 5 → đăng nhập account
2. Đảm bảo `.env` có đủ credentials
3. Chạy lệnh live:

```powershell
$env:PYTHONPATH="src"
python -m xauusd_ai.main live --config configs/live_acc2.yaml
```

### Startup sequence (log bình thường)

```
INFO | SelfLearner preload: XAUUSDm_M15.csv -> 8000 rows
INFO | ExitModel loaded (thr=0.65 min_rr=0.80)
INFO | LearnerThread started (background learning)
WARNING | Finnhub key invalid/quota exceeded (403) – disabling for 3600s  ← bình thường
INFO | build_news_calendar: 13 events [ForexFactory(12) + RuleBased(1)]
INFO | attach_news_features: 300 bars | blackout=31                       ← số hợp lý
INFO | No trade: Model confidence below threshold | bal=221.06 open=0/3
```

### Telegram alerts

Live bot gửi alert qua Telegram khi:
- Mở lệnh mới
- Đóng lệnh (TP/SL/Exit model)
- Circuit breaker triggered
- Lỗi nghiêm trọng

### Monitoring

```powershell
# Xem log realtime
Get-Content outputs/live_run2.txt -Wait -Tail 20

# Xem trạng thái hiện tại
Get-Content outputs/live_status_acc2.json

# Xem lịch sử giao dịch
Get-Content outputs/live_closed_trades_acc2.csv
```

---

## 13. Các account đang chạy

### Account 2 — Đang live (chính)

| Item | Giá trị |
|---|---|
| Account | 433326057 |
| Broker | Exness-MT5Trial7 |
| Symbol | XAUUSDm |
| Config | `configs/live_acc2.yaml` |
| Model | `outputs/model2_weekly500.pkl` |
| Magic number | 20260316 |
| Max positions | 3 |
| Base risk | 3% per trade |

### Account 1 — ICT Wyckoff

| Item | Giá trị |
|---|---|
| Config | `configs/live_ict_wyckoff.yaml` |
| Model | `outputs/model_ict_wyckoff.pkl` |
| Magic number | 20260313 |
| MT5 login | env `MT5_LOGIN` |

---

## 14. Bugs đã fix

### fix: pandas 2.x blackout=100000 (news_features.py)

**Triệu chứng**: 100% bars bị đánh dấu news blackout → bot không bao giờ trade.

**Root cause**: pandas 2.x lưu tz-aware datetime dưới dạng `datetime64[us]`, khi `.values.astype("int64")` trả về microseconds nhưng `ns_bo` là nanoseconds → mọi bar đều nằm trong blackout.

**Fix** (`src/xauusd_ai/data/news_features.py`):
```python
def _to_epoch_ns(s: pd.Series) -> np.ndarray:
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.values.astype("datetime64[ns]").view("int64")
```

**Kết quả**: blackout=1799 (hợp lý) thay vì 100000.

### fix: Finnhub 403 spam loop (news_features.py)

**Triệu chứng**: Sau khi nhận 403, vẫn tiếp tục gọi API cho các quý còn lại.

**Fix**: Thêm `if _finnhub_disabled: break` sau mỗi lần gọi `_fetch_finnhub_year()`.

---

## 15. Workflow chuẩn khi thay đổi code

### Khi sửa features/model

```powershell
# 1. Sửa code
# 2. Kiểm tra syntax
python -c "import py_compile; py_compile.compile('src/xauusd_ai/features/dataset.py')"

# 3. Kiểm tra import
$env:PYTHONPATH="src"; python -c "from xauusd_ai.features.dataset import FEATURE_COLUMNS; print(len(FEATURE_COLUMNS))"

# 4. Train lại model
$env:PYTHONPATH="src"; python -m xauusd_ai.main train --config configs/live_acc2.yaml

# 5. Backtest
$env:PYTHONPATH="src"; python -m xauusd_ai.main backtest --config configs/live_acc2.yaml

# 6. Commit
git add -A
git commit -m "feat: mô tả thay đổi"
git push origin feature/modelv5-cai-tien
```

### Khi thêm feature mới vào FEATURE_COLUMNS

1. Thêm function indicator vào `indicators.py`
2. Add vào `FEATURE_COLUMNS` list trong `dataset.py`
3. Đảm bảo indicator được compute trong `build_features()` của `dataset.py`
4. **Không cần** sửa gì thêm — `EXIT_FEATURE_COLUMNS = FEATURE_COLUMNS + EXIT_EXTRA_FEATURES` tự động pickup

### Khi deploy lên máy mới

```powershell
git clone https://github.com/bac25msa13227-star/Trade-Indicator.git
cd Trade-Indicator
git checkout feature/modelv5-cai-tien
git lfs pull                    # Download models (106MB pkl)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-windows.txt
# Tạo .env với credentials
# Mở MT5, đăng nhập
$env:PYTHONPATH="src"
python -m xauusd_ai.main live --config configs/live_acc2.yaml
```

---

## Version history

| Version | Ngày | Thay đổi chính |
|---|---|---|
| v6 (hiện tại) | 21/03/2026 | Circuit breaker, Silver Bullet, ADX gate, Partial TP, 56 features, fix pandas 2.x bugs |
| v5 | 18/03/2026 | Session-aware spread, FVG tracking, ATR-relative EHL, EMA direction |
| v4 | trước | ICT/Wyckoff indicators, walk-forward validation, calibrated probabilities |
| v3 | trước | Parallel live learning, ICT/Wyckoff strategy |
