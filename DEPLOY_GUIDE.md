# DEPLOY GUIDE — V14++ (ACC1 + ACC2)

> **Dành cho:** Máy đã clone repo, chưa ở branch `feature/dashboard-controls-v14pp`  
> **Ngày cập nhật:** 17/04/2026  
> **Version:** V14++ PROFIT (ACC1, sc=0.85) + COMPOSITE (ACC2, sc=0.88)

---

## BƯỚC 0 — Chuyển sang đúng branch + kéo model về

```powershell
# Mở PowerShell as Administrator, vào thư mục repo
cd C:\Users\Administrator\Documents\Trade-Indicator

# Chuyển branch và pull code + model LFS
git fetch origin
git checkout feature/dashboard-controls-v14pp
git pull origin feature/dashboard-controls-v14pp

# Kéo model pkl về qua Git LFS (~70 MB)
git lfs install
git lfs pull

# Kiểm tra model đã có chưa (phải > 1 MB mới đúng, không phải LFS pointer)
Get-Item outputs\acc1_v14pp_model.pkl | Select-Object Name, Length
Get-Item outputs\acc2_v14pp_model.pkl | Select-Object Name, Length
```

**Kết quả mong đợi:**
```
Name                    Length
----                    ------
acc1_v14pp_model.pkl    ~25000000   (≈25 MB)
acc2_v14pp_model.pkl    ~25000000   (≈25 MB)
```

---

## BƯỚC 1 — Chuẩn bị data CSV (~1.2 GB, không có trong git)

Data CSV cần thiết để retrain model hoặc chạy WF verify.  
**Nếu chỉ chạy live bot thì BỎ QUA bước này** — bot dùng MT5 live data.

```powershell
# Kiểm tra xem đã có data chưa
Test-Path src\xauusd_ai\real_data\XAUUSDm_M15.csv

# Nếu chưa có → copy từ máy cũ qua mạng LAN:
#   robocopy \\<IP-may-cu>\Trade-Indicator\src\xauusd_ai\real_data src\xauusd_ai\real_data *.csv /Z /MT:4

# Hoặc copy thủ công qua USB vào thư mục:
#   src\xauusd_ai\real_data\
# File cần: XAUUSDm_M1.csv (896 MB), M5.csv (193 MB), M15.csv (65 MB),
#           M30.csv (33 MB), H1.csv (16 MB), H4.csv (4 MB), D1.csv (1 MB)
```

---

## BƯỚC 2 — Tạo file .env

```powershell
Copy-Item .env.example .env
notepad .env
```

Điền các giá trị bắt buộc:

```env
# MT5 Bridge ACC1
MT5_LOGIN_ACC1=<số tài khoản MT5 acc1>
MT5_PASSWORD_ACC1=<mật khẩu acc1>
MT5_SERVER_ACC1=Exness-MT5Real8

# MT5 Bridge ACC2
MT5_LOGIN_ACC2=<số tài khoản MT5 acc2>
MT5_PASSWORD_ACC2=<mật khẩu acc2>
MT5_SERVER_ACC2=Exness-MT5Real8

# Telegram (để nhận thông báo)
TELEGRAM_BOT_TOKEN_ACC1=<bot_token>
TELEGRAM_CHAT_ID_ACC1=<chat_id>
TELEGRAM_BOT_TOKEN_ACC2=<bot_token>
TELEGRAM_CHAT_ID_ACC2=<chat_id>

# Database (giữ mặc định nếu dùng Docker)
POSTGRES_USER=trader
POSTGRES_PASSWORD=trader_secret
POSTGRES_DB=tradedb
```

---

## BƯỚC 3 — Chạy deploy script (một lệnh duy nhất)

```powershell
# Từ thư mục repo:
Set-ExecutionPolicy -Scope Process Bypass
.\deploy_v14pp.ps1
```

Script tự làm:
1. Verify model files từ LFS
2. Đọc WF benchmark đã verify (JSON trong git — không cần data CSV)
3. Build Docker image
4. Khởi động postgres + live-acc1 + live-acc2
5. In báo cáo triển khai ra `outputs\deploy_report_YYYYMMDD_HHmmss.txt`

**Tuỳ chọn nâng cao:**
```powershell
# Chỉ verify, không start bot
.\deploy_v14pp.ps1 -SkipBotStart

# Retrain model (cần data CSV từ Bước 1)
.\deploy_v14pp.ps1 -RetrainModels

# Chạy WF verify 30 folds (cần data CSV, ~20 phút)
.\deploy_v14pp.ps1 -RunWF

# Kết hợp: retrain + WF + deploy
.\deploy_v14pp.ps1 -RetrainModels -RunWF
```

---

## BƯỚC 4 — Kiểm tra sau deploy

```powershell
# Trạng thái container
docker ps --format "table {{.Names}}`t{{.Status}}"

# Log 15 dòng cuối ACC1
docker logs trade-indicator-live-acc1-1 --tail 15

# Log 15 dòng cuối ACC2
docker logs trade-indicator-live-1 --tail 15

# Kiểm tra model meta (threshold phải là 0.44 / 0.46)
Get-Content outputs\acc1_v14pp_model_meta.json | ConvertFrom-Json | Select decision_threshold, roc_auc, retrain_date
Get-Content outputs\acc2_v14pp_model_meta.json | ConvertFrom-Json | Select decision_threshold, roc_auc, retrain_date

# Kill switch status (phải là false)
Get-Content outputs\risk_daily_state_acc1.json | ConvertFrom-Json | Select killed, consecutive_losses
Get-Content outputs\risk_daily_state_acc2.json | ConvertFrom-Json | Select killed, consecutive_losses
```

**Kết quả mong đợi ACC1 meta:**
```
decision_threshold : 0.44
roc_auc            : 0.5498
retrain_date       : 2026-04-16T17:46:03
```

---

## BƯỚC 5 — Khởi động MT5 Bridges (nếu không dùng Docker cho bridges)

Mở **2 cửa sổ PowerShell riêng biệt**:

```powershell
# Cửa sổ 1 — Bridge ACC1 (port 5600)
cd C:\Users\Administrator\Documents\Trade-Indicator
.\.venv\Scripts\Activate.ps1
$env:MT5_LOGIN = "<acc1_login>"
$env:MT5_PASSWORD = "<acc1_password>"
$env:MT5_SERVER = "Exness-MT5Real8"
python scripts/windows/mt5_bridge.py --port 5600
```

```powershell
# Cửa sổ 2 — Bridge ACC2 (port 5601)
cd C:\Users\Administrator\Documents\Trade-Indicator
.\.venv\Scripts\Activate.ps1
$env:MT5_LOGIN = "<acc2_login>"
$env:MT5_PASSWORD = "<acc2_password>"
$env:MT5_SERVER = "Exness-MT5Real8"
python scripts/windows/mt5_bridge.py --port 5601
```

Kiểm tra bridges hoạt động:
```powershell
Invoke-RestMethod http://localhost:5600/ping   # {"status":"ok","account":"ACC1"}
Invoke-RestMethod http://localhost:5601/ping   # {"status":"ok","account":"ACC2"}
```

---

## Thông số V14++ quan trọng

| Tham số | ACC1 (PROFIT) | ACC2 (COMPOSITE) |
|---|---|---|
| Config file | `configs/live_acc1.yaml` | `configs/live_acc2.yaml` |
| Model pkl | `outputs/acc1_v14pp_model.pkl` | `outputs/acc2_v14pp_model.pkl` |
| `min_confidence` | `0.85` | `0.88` |
| `signal_threshold` | `0.60` | `0.60` |
| `decision_threshold` | `0.44` | `0.46` |
| Features | 65 | 65 |
| `risk_per_trade` | `4%` | `4%` |
| `take_profit_rr` | `3.5R` | `3.5R` |
| `daily_loss_limit_pct` | `12%` | `12%` |
| WF Profit Factor | **3.931** | **3.364** |
| WF Max Drawdown | -8.63% | -6.69% |
| WF Positive folds | 24/29 (83%) | 25/29 (86%) |

---

## Restart / Update nhanh

```powershell
# Restart cả 2 bot (sau khi thay đổi config hoặc model)
docker restart trade-indicator-live-acc1-1
docker restart trade-indicator-live-1

# Pull code mới nhất (không đổi branch, không mất data)
git pull origin feature/dashboard-controls-v14pp
git lfs pull

# Retrain model hàng tuần (cần data CSV)
python scripts/retrain_live_model.py configs/live_acc1.yaml
python scripts/retrain_live_model.py configs/live_acc2.yaml
docker restart trade-indicator-live-acc1-1
docker restart trade-indicator-live-1
```

---

## Troubleshooting nhanh

| Lỗi | Cách fix |
|---|---|
| `git lfs pull` lỗi authentication | `git lfs install` rồi thử lại, hoặc `git credential-manager` |
| Model file chỉ 134B (LFS pointer) | Chạy lại `git lfs pull` khi có internet tốt |
| Container exit ngay | `docker logs trade-indicator-live-acc1-1 --tail 30` để xem lỗi |
| Bridge timeout | Kiểm tra MT5 terminal đã mở và đăng nhập chưa |
| `killed: true` | Dashboard → Reset Kill Switch, hoặc: `Invoke-RestMethod -Method POST http://localhost:8000/api/v1/reset-kill-switch/acc1` |
| Missing `.env` | `Copy-Item .env.example .env` rồi điền credentials |

---

> Chi tiết đầy đủ xem [LATEST_LIVE_GUIDE.md](LATEST_LIVE_GUIDE.md)
