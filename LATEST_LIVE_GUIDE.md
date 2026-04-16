# Latest Live Guide

Last updated: `2026-04-16` · Branch: `feature/dashboard-controls-v14pp`

---

## TL;DR — Triển khai nhanh trên máy Windows mới

1. Clone repo + vào thư mục
2. Copy `.env.example` → `.env`, điền credentials
3. Chạy **một lệnh duy nhất**:

```powershell
Set-ExecutionPolicy -Scope Process Bypass; .\scripts\windows\deploy_new_machine.ps1
```

Script sẽ tự động: cài dependencies → verify models → khởi động bridges → khởi động bots → khởi dashboard → in checklist.

---

## 1. Cấu hình đang dùng (V14++ PROFIT / COMPOSITE)

### Model artifacts (BẮT BUỘC phải có đúng file này)

| Account | Model | Scaler | Meta |
|---|---|---|---|
| ACC1 | `outputs/acc1_v14pp_model.pkl` | `outputs/acc1_v14pp_scaler.pkl` | `outputs/acc1_v14pp_model_meta.json` |
| ACC2 | `outputs/acc2_v14pp_model.pkl` | `outputs/acc2_v14pp_scaler.pkl` | `outputs/acc2_v14pp_model_meta.json` |

### Live config files

- ACC1: `configs/live_acc1.yaml` — V14++ PROFIT (`sc=0.85`)
- ACC2: `configs/live_acc2.yaml` — V14++ COMPOSITE (`sc=0.88`)

### Threshold snapshot (quan trọng khi debug)

| Account | `signal_threshold` | `decision_threshold` (meta) | `min_confidence` |
|---|---|---|---|
| ACC1 | `0.60` | `0.62` | `0.85` |
| ACC2 | `0.60` | `0.72` | `0.85` |

> **Note:** `signal_threshold` là gate ML đầu vào.  
> `sideway_min_confidence` / `volatile_min_confidence` đều = `0.85` → gate thực tế nghiêm hơn.

### WF Benchmark (V14++ đã validate 29 folds, 2024-09→2026-04)

| Account | Strategy | PF avg | DD avg | Return avg | Positive folds |
|---|---|---|---|---|---|
| ACC1 | V14++ PROFIT | **4.07** | -8.63% | +47.96%/fold | 24/29 (83%) |
| ACC2 | V14++ COMPOSITE | **3.75** | -6.69% | +33.38%/fold | 24/29 (83%) |

### Model performance (train/test split)

- ROC-AUC: `0.659`
- Precision: `0.783`
- Feature columns: `84`
- Train rows: `1,616,062`

---

## 2. Risk settings (key params)

| Param | ACC1 | ACC2 |
|---|---|---|
| `risk_per_trade` | `4%` | `4%` |
| `take_profit_rr` | `3.5R` | `3.5R` |
| `partial_tp_rr` | `1.2R` (50%) | `1.2R` (50%) |
| `consecutive_loss_pause_count` | `3` | `3` |
| `consecutive_loss_cooldown_bars` | `12` (1h) | `12` (1h) |
| `daily_loss_limit_pct` | `12%` | `12%` |
| `max_drawdown_kill_pct` | `20%` | `20%` |
| `kill_switch_enabled` | `true` | `true` |
| `auto_trade` | `true` | `true` |

---

## 3. Environment variables (.env)

Tạo file `.env` từ `.env.example`:

```powershell
Copy-Item .env.example .env
notepad .env
```

Các biến BẮT BUỘC:

```env
# ── MT5 Bridge ACC1 (Exness account 1) ──────────────
MT5_LOGIN_ACC1=<your_acc1_login>
MT5_PASSWORD_ACC1=<your_acc1_password>
MT5_SERVER_ACC1=Exness-MT5Real8

# ── MT5 Bridge ACC2 (Exness account 2) ──────────────
MT5_LOGIN_ACC2=<your_acc2_login>
MT5_PASSWORD_ACC2=<your_acc2_password>
MT5_SERVER_ACC2=Exness-MT5Real8

# ── Telegram notifications ──────────────────────────
TELEGRAM_BOT_TOKEN_ACC1=<bot_token_acc1>
TELEGRAM_CHAT_ID_ACC1=<chat_id_acc1>
TELEGRAM_BOT_TOKEN_ACC2=<bot_token_acc2>
TELEGRAM_CHAT_ID_ACC2=<chat_id_acc2>

# ── Optional ────────────────────────────────────────
FINNHUB_API_KEY=<your_finnhub_key>
```

---

## 4. Khởi động thủ công (nếu không dùng deploy script)

### Bước 1: Cài môi trường (lần đầu)

```powershell
# Trong PowerShell (chạy với quyền admin lần đầu)
Set-ExecutionPolicy -Scope Process Bypass
cd C:\path\to\Trade-Indicator
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
```

### Bước 2: Verify model binding

```powershell
$env:PYTHONPATH="src"
python -c "
from pathlib import Path
from xauusd_ai.config import load_settings
for cfg in ['configs/live_acc1.yaml', 'configs/live_acc2.yaml']:
    s = load_settings(Path(cfg))
    print(cfg, '->', s.app.model_path)
    print('  threshold:', s.strategy.signal_threshold)
    print('  auto_trade:', s.execution.auto_trade)
"
```

Expected output:
```
configs/live_acc1.yaml -> outputs/acc1_v14pp_model.pkl
  threshold: 0.6
  auto_trade: True
configs/live_acc2.yaml -> outputs/acc2_v14pp_model.pkl
  threshold: 0.6
  auto_trade: True
```

### Bước 3: Khởi động MT5 Bridges

Mở 2 cửa sổ PowerShell riêng:

```powershell
# Window 1 — Bridge ACC1 (port 5600)
cd C:\path\to\Trade-Indicator
$env:MT5_LOGIN = $env:MT5_LOGIN_ACC1
$env:MT5_PASSWORD = $env:MT5_PASSWORD_ACC1
$env:MT5_SERVER = $env:MT5_SERVER_ACC1
python scripts/windows/mt5_bridge.py --port 5600

# Window 2 — Bridge ACC2 (port 5601)
$env:MT5_LOGIN = $env:MT5_LOGIN_ACC2
$env:MT5_PASSWORD = $env:MT5_PASSWORD_ACC2
$env:MT5_SERVER = $env:MT5_SERVER_ACC2
python scripts/windows/mt5_bridge.py --port 5601
```

Verify bridges:

```powershell
Invoke-RestMethod http://localhost:5600/ping | ConvertTo-Json
Invoke-RestMethod http://localhost:5601/ping | ConvertTo-Json
```

### Bước 4: Khởi động Live Bots

```powershell
# Cửa sổ 3 — Bot ACC1
powershell -File scripts\windows\launch_bot_acc1.ps1

# Cửa sổ 4 — Bot ACC2
powershell -File scripts\windows\launch_bot_acc2.ps1
```

### Bước 5: Khởi động Dashboard API

```powershell
# Cửa sổ 5 — FastAPI
powershell -File scripts\windows\launch_api.ps1
```

Dashboard: **http://localhost:8000/dashboard**

---

## 5. Dashboard features

| Feature | Mô tả |
|---|---|
| Auto Trade toggle | Bật/tắt `auto_trade` cho từng acc mà không cần restart |
| Kill Switch Reset | Manual reset kill switch từ dashboard |
| Market Regime | Hiển thị Sideways / Normal / Volatile / Extreme |
| Signal Feed | Xem tối đa 500 tín hiệu gần nhất |
| Tunnel URL → Telegram | Detect Cloudflare URL + gửi về Telegram |

### API endpoints quan trọng

```
POST /api/v1/auto-trade/acc1?enabled=true   # Bật auto trade ACC1
POST /api/v1/auto-trade/acc2?enabled=false  # Tắt auto trade ACC2
POST /api/v1/reset-kill-switch/acc1         # Reset kill switch ACC1
GET  /api/v1/tunnel-url                     # Lấy Cloudflare tunnel URL
POST /api/v1/send-telegram                  # Gửi Telegram thủ công
GET  /api/v1/dashboard                      # Dashboard JSON payload
GET  /docs                                  # Swagger UI
```

---

## 6. Tunnel public dashboard (tùy chọn)

Tải `cloudflared.exe` từ https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

```powershell
# Tạo tunnel
.\cloudflared.exe tunnel --url http://localhost:8000 > outputs\tunnel_err.txt 2>&1 &

# URL sẽ xuất hiện trong outputs\tunnel_err.txt
# Nhấn nút "Detect & Send Tunnel URL" trên dashboard để gửi về Telegram
```

---

## 7. Dừng toàn bộ

```powershell
powershell -File scripts\windows\stop_all.ps1
```

Hoặc thủ công:

```powershell
Get-Process -Name python, cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
```

---

## 8. Checklist verify sau khi khởi động

Sau khi chạy `deploy_new_machine.ps1`, script tự in checklist. Kiểm tra thủ công:

```powershell
# 1. Model files có đủ không?
Test-Path outputs\acc1_v14pp_model.pkl
Test-Path outputs\acc2_v14pp_model.pkl

# 2. Bridge ACC1 sống không?
(Invoke-RestMethod http://localhost:5600/ping -ErrorAction SilentlyContinue).status

# 3. Bridge ACC2 sống không?
(Invoke-RestMethod http://localhost:5601/ping -ErrorAction SilentlyContinue).status

# 4. API sống không?
(Invoke-RestMethod http://localhost:8000/health -ErrorAction SilentlyContinue).status

# 5. Dashboard payload ACC1 có model đúng không?
(Invoke-RestMethod http://localhost:8000/api/v1/dashboard).accounts.acc1.status.model_path

# 6. Kill switch trạng thái?
Get-Content outputs\risk_daily_state_acc1.json | ConvertFrom-Json | Select killed, consecutive_losses
Get-Content outputs\risk_daily_state_acc2.json | ConvertFrom-Json | Select killed, consecutive_losses

# 7. Auto trade đang bật không?
(Invoke-RestMethod http://localhost:8000/api/v1/dashboard).accounts.acc1.auto_trade_enabled
(Invoke-RestMethod http://localhost:8000/api/v1/dashboard).accounts.acc2.auto_trade_enabled
```

---

## 9. Troubleshooting

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| `model_binding_ok: false` | Model file không khớp config | Copy đúng `acc1_v14pp_model.pkl` vào `outputs/` |
| Bridge 5600 không phản hồi | MT5 terminal chưa mở / sai credentials | Mở MT5 → đăng nhập → restart bridge |
| `killed: true` trong daily state | Kill switch bị kích hoạt do loss | Nhấn "Reset Kill Switch" trên dashboard |
| `auto_trade: false` sau restart | Config bị ghi đè | Dùng POST `/api/v1/auto-trade/acc1?enabled=true` |
| Dashboard không load | uvicorn chưa chạy hoặc port 8000 bị chiếm | `netstat -ano \| findstr :8000` rồi kill |

---

## 10. Market open/close gate

Bot tự block lệnh khi sàn đóng. Cảnh báo Telegram:

- Trước mở cửa: 1440 phút (1 ngày) và 30 phút
- Trước đóng cửa: 1440 phút (1 ngày) và 30 phút

Config:
```yaml
market:
  enforce_market_open_gate: true
  market_tick_stale_seconds: 300
  market_preopen_alert_minutes_list: [1440, 30]
  market_preclose_alert_minutes_list: [1440, 30]
```

---

## 11. Data source

Training data: `src/xauusd_ai/real_data/` (CSV, từ `2003-05-05`)

- `XAUUSDm_D1.csv`, `H4.csv`, `H1.csv`, `M30.csv`, `M15.csv`, `M5.csv`, `M1.csv`

Live data: MT5 bridge (real-time feed)


