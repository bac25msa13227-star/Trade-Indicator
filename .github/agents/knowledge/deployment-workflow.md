# Deployment & Testing Workflow

## Quy trình phát triển chuẩn

```
Develop code → Test local → Commit → Push GitHub → Deploy VPS
     ↓             ↓            ↓          ↓            ↓
Trade Indicator   WF/Live    Git repos   Pull code   Restart
    (local)       service                 on VPS      services
```

---

## 1. Local Development (Laptop)

### Setup venv lần đầu
```bash
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
pip install -r requirements-infra.txt
```

### Activate venv
```bash
source "$HOME/Documents/Thạc sĩ MSE/Trade Indicator/.venv/bin/activate"
```

### Run tests

**Test WF Engine (port 8801)**:
```bash
export BASE="$HOME/Documents/Thạc sĩ MSE"
export UVICORN="$BASE/Trade Indicator/.venv/bin/uvicorn"
cd "$BASE/WFService"
"$UVICORN" app.main:app --host 0.0.0.0 --port 8801 --reload
```

Verify: `curl http://localhost:8801/health`

**Test Bot Engine (port 8802)**:
```bash
export BASE="$HOME/Documents/Thạc sĩ MSE"
export UVICORN="$BASE/Trade Indicator/.venv/bin/uvicorn"
cd "$BASE/LiveBotService"
PYTHONPATH="src:../WFService/src" "$UVICORN" app.main:app --host 0.0.0.0 --port 8802 --reload
```

Verify: `curl http://localhost:8802/health`

**Test Live Dashboard (port 8000)**:
```bash
cd "$BASE/LiveBotService"
PYTHONPATH="src:../WFService/src" "$UVICORN" xauusd_ai.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Verify: `curl http://localhost:8000/health`

---

## 2. Run Walk-Forward Tests

### Quick test (1 fold, cache enabled)
```bash
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"
.venv/bin/python scripts/walkforward_ict_wyckoff.py \
  configs/acc1_v14pp_profit.yaml \
  --no-rr-sweep --cache \
  --test-start 2026-05-01 \
  --test-bars 500 --step-bars 500 \
  --no-compound --combo133 \
  --risk-pct 0.030
```

### Full validation (multiple folds)
```bash
.venv/bin/python scripts/walkforward_ict_wyckoff.py \
  configs/acc1_v14pp_profit.yaml \
  --no-rr-sweep --cache \
  --test-start 2024-01-01 \
  --test-bars 6000 --step-bars 6000 \
  --no-compound --combo133 \
  --risk-pct 0.030 \
  2>&1 | tee outputs/wf_verify_$(date +%Y%m%d).txt
```

### Check results
```bash
tail -30 outputs/wf_verify_*.txt
grep "Profit Factor" outputs/wf_verify_*.txt
grep "Max Drawdown" outputs/wf_verify_*.txt
```

**Acceptance criteria**:
- Profit Factor ≥ 1.3
- Max Drawdown ≤ 15%
- At least 3 consecutive folds positive

---

## 3. Commit & Push to GitHub

### WFService changes
```bash
cd "$HOME/Documents/Thạc sĩ MSE/WFService"
git status
git add -A
git commit -m "feat: implement slippage model with --slippage flag"
eval "$(/opt/homebrew/bin/brew shellenv)"
git push
```

### LiveBotService changes
```bash
cd "$HOME/Documents/Thạc sĩ MSE/LiveBotService"
git status
git add -A
git commit -m "fix: add Sharpe ratio calculation to orchestrator"
eval "$(/opt/homebrew/bin/brew shellenv)"
git push
```

### Verify on GitHub
- WFService: https://github.com/bac25msa13227-star/WFService
- LiveBotService: https://github.com/bac25msa13227-star/LiveBotService

---

## 4. Deploy to VPS (Production)

### SSH vào VPS
```bash
ssh user@vps-ip
```

### Pull latest code
```bash
cd /root/trading/WFService
git pull origin main

cd /root/trading/LiveBotService
git pull origin main
```

### Rebuild Docker images (nếu cần)
```bash
cd /root/trading/WFService
docker compose build wf-engine

cd /root/trading/LiveBotService
docker compose build bot-engine live-api
```

### Restart services
```bash
# Restart WF services
cd /root/trading/WFService
docker compose restart wf-engine

# Restart live bots (CẨN THẬN!)
cd /root/trading/LiveBotService
docker compose restart bot-engine
docker compose restart live-api
```

**CẢNH BÁO**: Restart live bots sẽ đóng tất cả lệnh đang mở. Chỉ restart khi:
- Không có lệnh active (check trước)
- Hoặc đã có backup plan cho positions đang mở

### Check logs
```bash
docker logs -f wf-engine
docker logs -f bot-engine
docker logs -f live-api
```

### Verify services
```bash
curl http://localhost:8801/health  # WF Engine
curl http://localhost:8802/health  # Bot Engine
curl http://localhost:8000/health  # Live API
```

---

## 5. Monitor Production

### Check live trades
```bash
cd /root/trading/Trade\ Indicator
tail -f outputs/live_closed_trades_acc1.csv
```

### Check metrics
- Prometheus: http://vps-ip:9090
- Grafana: http://vps-ip:3000

### Check MLflow
http://vps-ip:5000

---

## 6. Rollback nếu có lỗi

### Revert git commit
```bash
cd WFService  # hoặc LiveBotService
git log --oneline -5
git revert <commit-hash>
git push
```

### Hoặc reset về commit trước
```bash
git reset --hard HEAD~1
git push --force  # NGUY HIỂM: chỉ dùng khi emergency
```

### Pull và restart trên VPS
```bash
ssh user@vps-ip
cd /root/trading/WFService
git pull origin main
docker compose restart wf-engine
```

---

## 7. Testing Checklist trước khi deploy

- [ ] WF Engine health check pass (port 8801)
- [ ] Bot Engine health check pass (port 8802)
- [ ] Live API health check pass (port 8000)
- [ ] ChartWF dashboard load được (port 5173)
- [ ] Walk-forward test pass với Profit Factor ≥ 1.3
- [ ] Max Drawdown ≤ 15%
- [ ] Không có error trong logs local
- [ ] Git commit message rõ ràng
- [ ] Code đã push lên GitHub
- [ ] Backup livebot state nếu restart production

---

## 8. Emergency Recovery

### Service không start
```bash
# Check logs
docker logs wf-engine
docker logs bot-engine

# Rebuild từ đầu
docker compose down
docker compose build --no-cache
docker compose up -d
```

### DB connection fail
```bash
# Check PostgreSQL
docker ps | grep postgres
docker logs postgres

# Restart postgres
docker restart postgres
```

### MLflow tracking fail
```bash
# Check MLflow
docker ps | grep mlflow
docker logs mlflow

# Restart mlflow
docker restart mlflow
```

### Live bots không vào lệnh
1. Check MT5 terminal running: `ps aux | grep terminal`
2. Check MT5 bridge logs: `docker logs bot-engine | grep MT5`
3. Verify account credentials trong config
4. Check broker connection: login MT5 terminal manually

---

## 9. Performance Tuning

### WF quá chậm
- Enable cache: `--cache`
- Giảm số bars: `--test-bars 3000` thay vì 6000
- Run ngoài Docker: local venv nhanh hơn
- Dùng `--fast` flag (nếu có)

### Docker memory issue
```yaml
# docker-compose.yml
services:
  wf-engine:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

### Logs quá nhiều
```python
# Giảm log level
import logging
logging.getLogger("xauusd_ai").setLevel(logging.WARNING)
```

---

## 10. Code sync workflow giữa 3 repos

### Trade Indicator → WFService
```bash
BASE="$HOME/Documents/Thạc sĩ MSE"
rsync -av --exclude='.git' --exclude='.venv' \
  "$BASE/Trade Indicator/src/xauusd_ai/" \
  "$BASE/WFService/src/xauusd_ai/"

cd "$BASE/WFService"
git add -A
git commit -m "sync: update from Trade Indicator"
git push
```

### Trade Indicator → LiveBotService
```bash
rsync -av --exclude='.git' --exclude='.venv' \
  "$BASE/Trade Indicator/src/xauusd_ai/orchestrator.py" \
  "$BASE/LiveBotService/src/xauusd_ai/"

rsync -av --exclude='.git' --exclude='.venv' \
  "$BASE/Trade Indicator/src/xauusd_ai/execution/" \
  "$BASE/LiveBotService/src/xauusd_ai/execution/"

cd "$BASE/LiveBotService"
git add -A
git commit -m "sync: update from Trade Indicator"
git push
```

**LƯU Ý**: Không sync toàn bộ — chỉ sync files thực sự cần vì mỗi repo có mục đích riêng.
