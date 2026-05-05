# Kiến trúc hệ thống XAUUSD AI Trading

## Tổng quan 3 repos

### 1. Trade Indicator (Repo gốc)
**Path**: `~/Documents/Thạc sĩ MSE/Trade Indicator/`  
**Mục đích**: Repo gốc đang chạy live trading trên VPS, không được động đến  
**Trạng thái**: Production — đang chạy live bots acc1 và acc2

**Nội dung**:
- `src/xauusd_ai/` — full package (35+ modules)
- `scripts/` — 30+ scripts training, WF, backtest
- `configs/` — 13 YAML configs
- `outputs/` — models, trades CSV, reports
- `.venv/` — Python 3.12 virtual environment

### 2. WFService (Research/Walkforward)
**Path**: `~/Documents/Thạc sĩ MSE/WFService/`  
**GitHub**: https://github.com/bac25msa13227-star/WFService  
**Mục đích**: Walk-forward research, model training, backtesting

**Services**:
- **WF Engine API** — port 8801 (`app/main.py`)
  - Endpoints: `/configs`, `/runs`, `/health`
  - Quản lý WF jobs, config, results
- **ChartWF Backend** — port 8800 (`chartwf/backend/app/main.py`)
  - API cho dashboard: OHLCV, trades, metrics
- **ChartWF Frontend** — port 5173 (React + Vite)
  - Dashboard trực quan WF results

**Cấu trúc**:
```
WFService/
├── src/xauusd_ai/          # 35 .py files, NO __init__.py (namespace pkg)
│   ├── features/           # dataset.py, indicators.py
│   ├── model/              # trainer.py
│   ├── backtesting/        # engine.py
│   ├── strategies/         # hybrid.py
│   ├── infra/              # db.py, mlflow_client.py, metrics.py
│   └── monitoring/         # drift.py
├── app/main.py             # WF Engine API port 8801
├── chartwf/                # Dashboard
│   ├── backend/            # FastAPI port 8800
│   └── frontend/           # React + Vite
├── scripts/                # 20 WF scripts
├── configs/                # 13 YAML configs
└── pyproject.toml
```

### 3. LiveBotService (Live Trading)
**Path**: `~/Documents/Thạc sĩ MSE/LiveBotService/`  
**GitHub**: https://github.com/bac25msa13227-star/LiveBotService  
**Mục đích**: Live trading bots, live monitoring, execution

**Services**:
- **Bot Engine API** — port 8802 (`app/main.py`)
  - Endpoints: `/status`, `/accounts`, `/health`
  - Bot control, real-time status
- **Live Dashboard API** — port 8000 (`src/xauusd_ai/api/main.py`)
  - 18 endpoints: trades, metrics, override, signals
  - Real-time live trading dashboard

**Cấu trúc**:
```
LiveBotService/
├── src/xauusd_ai/
│   ├── __init__.py         # pkgutil.extend_path namespace package
│   ├── orchestrator.py     # Main trading loop
│   ├── main.py             # Bot entry point
│   ├── api/main.py         # Live Dashboard port 8000
│   ├── execution/          # mt5_executor.py, risk.py
│   └── config.py
├── app/main.py             # Bot Engine port 8802
├── scripts/                # 10 live scripts
├── configs/                # live_acc1.yaml, live_acc2.yaml
└── pyproject.toml
```

---

## Namespace Package Pattern

### Vấn đề ban đầu
Python tìm thấy `Trade Indicator/src/xauusd_ai/__init__.py` (regular package) trước và load nó thay vì LiveBotService modules.

### Giải pháp
**LiveBotService** sở hữu namespace `xauusd_ai` bằng cách tạo `__init__.py` với `pkgutil.extend_path`:

```python
# LiveBotService/src/xauusd_ai/__init__.py
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
__all__ = ["__version__"]
__version__ = "0.1.0"
```

**WFService** KHÔNG có `__init__.py` → pure namespace package.

**PYTHONPATH pattern**:
```bash
# Từ LiveBotService directory
PYTHONPATH="src:../WFService/src" python -m xauusd_ai.api.main
```
LiveBotService's `__init__.py` được load trước, extend path để merge WFService modules.

---

## Port Mapping

| Service | Port | Module | Mục đích |
|---------|------|--------|----------|
| WF Engine API | 8801 | `WFService/app/main.py` | WF job management |
| ChartWF Backend | 8800 | `WFService/chartwf/backend/app/main.py` | Dashboard data API |
| ChartWF Frontend | 5173 | `WFService/chartwf/frontend/` | React dashboard |
| Bot Engine API | 8802 | `LiveBotService/app/main.py` | Bot control API |
| Live Dashboard API | 8000 | `LiveBotService/src/xauusd_ai/api/main.py` | Live monitoring |

---

## Environment Variables

```bash
BASE="$HOME/Documents/Thạc sĩ MSE"
WFSERVICE_ROOT="$BASE/WFService"
LIVEBOTSERVICE_ROOT="$BASE/LiveBotService"
UVICORN="$BASE/Trade Indicator/.venv/bin/uvicorn"
```

---

## Docker Architecture

### WFService
```yaml
services:
  wf-engine:
    build: .
    ports: [8801:8801]
    
  chartwf-backend:
    build: chartwf/
    ports: [8800:8800]
    
  chartwf-frontend:
    build: chartwf/
    ports: [5173:5173]
```

### LiveBotService
```yaml
services:
  bot-engine:
    build: .
    ports: [8802:8802]
    
  live-api:
    build: .
    dockerfile: Dockerfile.backend
    ports: [8000:8000]
```

---

## Database & Infrastructure

### Shared services (Trade Indicator)
- **PostgreSQL** — trade history, positions
- **MLflow** — model tracking, versioning
- **Prometheus** — metrics collection
- **Grafana** — monitoring dashboard

### MT5 Bridge
- Python-MT5 connector qua MetaTrader 5 terminal
- Chạy trên Windows/Wine
- Execution: market orders, pending orders, SL/TP management

---

## Testing venv

Tất cả local tests đều dùng: `/Users/dodoannang/Documents/Thạc sĩ MSE/Trade Indicator/.venv`

Python 3.12, packages:
- lightgbm, scikit-learn, pandas, numpy
- fastapi, uvicorn, sqlalchemy
- MetaTrader5, prometheus_client
- plotly, jinja2
