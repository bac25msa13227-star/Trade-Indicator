# Lịch sử Debugging & Giải pháp

## 1. Namespace Package Conflict (5/5/2026)

### Triệu chứng
```
ModuleNotFoundError: No module named 'xauusd_ai.api.main'
```
Khi chạy LiveBotService, Python load `Trade Indicator/src/xauusd_ai/__init__.py` (regular package) thay vì LiveBotService modules.

### Root cause
Python's import system tìm **regular package** (có `__init__.py`) trước **namespace package** (không có `__init__.py`). `Trade Indicator` có `__init__.py` → Python dừng search ngay đó.

### Giải pháp
Tạo `LiveBotService/src/xauusd_ai/__init__.py` với `pkgutil.extend_path`:

```python
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
__all__ = ["__version__"]
__version__ = "0.1.0"
```

**Cơ chế**:
1. Python tìm thấy `LiveBotService/src/xauusd_ai/__init__.py` trước (via PYTHONPATH order)
2. `extend_path` merge tất cả `xauusd_ai` paths từ PYTHONPATH
3. Kết quả: `xauusd_ai.__path__` bao gồm cả LiveBotService + WFService + Trade Indicator

**PYTHONPATH pattern**:
```bash
PYTHONPATH="src:../WFService/src"
```
→ LiveBotService trước, WFService sau.

### Bài học
- Namespace packages phải **đồng nhất**: hoặc tất cả đều có `__init__.py`, hoặc không có cái nào
- Nếu cần merge, dùng `pkgutil.extend_path` trong package "sở hữu" namespace

---

## 2. /health Endpoint HTTP 500 (5/5/2026)

### Triệu chứng
```
GET /health → 500 Internal Server Error
ModuleNotFoundError: No module named 'psycopg2'
```

### Root cause
FastAPI dependency injection:
```python
@app.get("/health")
def health_check(engine=Depends(get_engine)):  # ❌
    ...
```

`Depends(get_engine)` được resolve **trước khi handler chạy** → `get_engine()` raise exception → 500 error.

`get_engine()` cần import `infra.db` → cần `psycopg2` → không có trong LiveBotService (chỉ có trong WFService).

### Giải pháp
Bỏ dependency injection, wrap `get_engine()` trong try/except:

```python
@app.get("/health")
def health_check():
    try:
        engine = get_engine()
        db_status = "ok" if engine else "degraded"
    except Exception as e:
        db_status = "degraded"
    return {"status": "ok", "db": db_status}
```

Sửa `get_engine()` trả về `None` thay vì raise:
```python
def get_engine():
    try:
        return _ge()
    except Exception:
        return None
```

### Bài học
- `/health` endpoint **không được phụ thuộc vào external services** (DB, broker...)
- Nếu service unavailable → return `degraded` status, không raise exception
- Dependency injection chỉ dùng khi service **phải có** — nếu optional thì call trực tiếp + try/except

---

## 3. Failed to initialize DB tables: 'NoneType' (5/5/2026)

### Triệu chứng
```python
File "src/xauusd_ai/api/main.py", line 2280, in on_startup
    init_tables(engine)
AttributeError: 'NoneType' object has no attribute '_run_ddl_visitor'
```

### Root cause
`on_startup` event handler gọi `init_tables(engine)` mà không check `engine is None`.

Khi `get_engine()` fail (vì không có psycopg2), `engine = None` → `init_tables(None)` crash.

### Giải pháp
Thêm guard clause:
```python
@app.on_event("startup")
async def on_startup():
    engine = get_engine()
    if engine is None:
        logger.warning("DB engine unavailable, skipping table init")
        return
    init_tables(engine)
```

### Bài học
- Mọi external dependency call phải có **None check** trước khi dùng
- Startup handlers nên graceful degradation — service vẫn start được dù một số features unavailable

---

## 4. __pycache__ và .pyc bị track trong git (5/5/2026)

### Triệu chứng
`git status` thấy hàng trăm file `__pycache__/`, `*.pyc`, `*.nbc`, `*.nbi` đã committed.

### Root cause
`.gitignore` có `__pycache__/` nhưng các file đã được `git add` trước khi `.gitignore` được tạo → vẫn bị track.

### Giải pháp
```bash
# Xóa khỏi git cache (không xóa file trên disk)
cd WFService
git rm -r --cached .
git add -A

# Xóa luôn khỏi disk
find . -not -path '*/.git/*' -name '__pycache__' -type d -exec rm -rf {} +
find . -not -path '*/.git/*' \( -name '*.pyc' -o -name '*.pyo' -o -name '*.nbc' -o -name '*.nbi' \) -delete
```

Commit:
```bash
git commit -m "chore: remove __pycache__, .pyc, build artifacts"
git push
```

### Bài học
- Luôn tạo `.gitignore` **trước** khi `git add` lần đầu
- Nếu file đã tracked, `.gitignore` không có tác dụng → phải `git rm --cached`
- Dùng `find` để xóa recursive thay vì xóa từng thư mục

---

## 5. WF Backtest vs Live Trade Gap (Hiện tại)

### Triệu chứng
- WF backtest: Profit Factor 1.8, Max DD 8%
- Live trade: Profit Factor 1.2, Max DD 12%
- Gap ~30%

### Root causes (hypothesis)

**1. Execution assumption**
- WF dùng `close` price để fill lệnh → không realistic
- Live có slippage, spread, latency

**2. Lookahead bias**
- Một số features trong `dataset.py` có thể leak future info
- Ví dụ: tính indicator trên toàn bộ period thay vì rolling window

**3. Overfitting**
- Model train trên 2022-2024, nhưng live 2025-2026 thị trường thay đổi
- Combo133 config quá nhiều hyperparams → fit noise

**4. Risk management khác biệt**
- WF: fixed risk_pct
- Live: dynamic risk adjustment + max DD protection → size nhỏ hơn

### Giải pháp đang implement

**Short-term (tuần 1-2)**:
- [ ] Thêm slippage model: `entry_price += spread + ATR * slippage_factor`
- [ ] Tính Sharpe/Calmar sau mỗi WF fold → compare với live
- [ ] Log feature importance → check stability qua các folds

**Mid-term (tháng 1)**:
- [ ] Regime detection: không trade khi thị trường sideway/low volatility
- [ ] Paper trading: shadow mode với live data nhưng không vào lệnh thật
- [ ] A/B test framework: chạy model mới song song với model cũ

**Long-term (tháng 2-3)**:
- [ ] Ensemble models: combine LightGBM + LSTM
- [ ] RL fine-tuning: PPO/SAC adjust sizing + exit
- [ ] Tick-by-tick simulation: replay M1 data tick-level

### Metric hiện tại để đánh giá gap
```python
# Sau mỗi WF fold
wf_pnl = fold_result['final_balance'] - fold_result['initial_balance']
live_pnl = sum(live_trades_csv['profit'] for same period)
gap = (wf_pnl - live_pnl) / wf_pnl * 100

# Threshold: gap < 25% là acceptable
```

---

## 6. Docker Desktop Memory Limit (Intermittent)

### Triệu chứng
WF Engine hoặc MLflow bị OOM killed trong Docker.

### Giải pháp
Tăng memory limit:
```yaml
services:
  wf-engine:
    deploy:
      resources:
        limits:
          memory: 4G
```

Hoặc chạy WF ngoài Docker (local venv) cho training jobs lớn.

---

## Best Practices đã học

1. **Health endpoints không phụ thuộc external services** — return degraded status thay vì crash
2. **Namespace packages cần strategy rõ ràng** — pkgutil hoặc không dùng `__init__.py`
3. **Dependency injection = hard dependency** — nếu optional thì try/except
4. **Mọi production metrics phải có backtest equivalent** — đo gap liên tục
5. **Slippage không phải detail nhỏ** — có thể làm gap 20-30%
6. **Feature engineering cần rolling window** — không leak future info
7. **Model validation = backtest + paper trade + live micro-size** — không skip bước nào
