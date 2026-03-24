"""FastAPI application — REST API for the trading system.

Endpoints:
  GET  /health                         – health check + DB/MinIO/MLflow ping
  GET  /api/v1/status/{account}        – live trading status from PostgreSQL
  GET  /api/v1/trades/{account}        – closed trade history
  GET  /api/v1/signals/{account}       – signal log
  GET  /api/v1/models                  – MLflow model registry list
  GET  /api/v1/models/{name}/metrics   – production model metrics
  POST /api/v1/predict                 – single-row inference (JSON features)
  GET  /api/v1/reports/{account}/{name}– presigned URL to Evidently report in MinIO
  GET  /metrics                        – Prometheus metrics (mounted separately)

Start:
    uvicorn xauusd_ai.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="XAUUSD AI Trading API",
    description="REST API for live trading status, trade history, model management, and inference.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics endpoint
try:
    from prometheus_client import make_asgi_app
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    logger.info("Prometheus /metrics endpoint mounted.")
except ImportError:
    logger.warning("prometheus_client not installed — /metrics endpoint unavailable.")


# ── Dependency injection ──────────────────────────────────────────────────────

_engine = None
_storage = None
_tracker = None


def get_engine():
    global _engine
    if _engine is None:
        from xauusd_ai.infra.db import get_engine as _ge
        _engine = _ge()
    return _engine


def get_storage():
    global _storage
    if _storage is None:
        from xauusd_ai.infra.storage import StorageClient
        _storage = StorageClient()
    return _storage


def get_tracker():
    global _tracker
    if _tracker is None:
        from xauusd_ai.infra.mlflow_client import MLflowTracker
        _tracker = MLflowTracker()
    return _tracker


def get_trade_store(engine=Depends(get_engine)):
    from xauusd_ai.infra.db import TradeStore
    return TradeStore(engine)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    account: str
    features: dict[str, float]


class PredictResponse(BaseModel):
    account: str
    confidence: float
    should_trade: bool
    side: str
    latency_ms: float


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health_check(engine=Depends(get_engine)) -> dict[str, Any]:
    status: dict[str, Any] = {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

    # PostgreSQL ping
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["postgres"] = "ok"
    except Exception as exc:
        status["postgres"] = f"error: {exc}"
        status["status"] = "degraded"

    # MinIO ping
    try:
        storage = get_storage()
        status["minio"] = "ok" if storage._client is not None else "unavailable"
    except Exception as exc:
        status["minio"] = f"error: {exc}"

    # MLflow ping
    try:
        import mlflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        status["mlflow"] = "ok"
    except Exception as exc:
        status["mlflow"] = f"error: {exc}"

    return status


# ── Live status ───────────────────────────────────────────────────────────────

@app.get("/api/v1/status/{account}", tags=["trading"])
def get_live_status(account: str, store=Depends(get_trade_store)) -> dict[str, Any]:
    status = store.get_live_status(account)
    if status is None:
        raise HTTPException(status_code=404, detail=f"No status found for account '{account}'")
    # Serialize datetime fields
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in status.items()}


# ── Trades ────────────────────────────────────────────────────────────────────

@app.get("/api/v1/trades/{account}", tags=["trading"])
def get_trades(
    account: str,
    since: str | None = Query(None, description="ISO datetime filter (e.g. 2026-01-01T00:00:00)"),
    limit: int = Query(200, ge=1, le=5000),
    store=Depends(get_trade_store),
) -> list[dict[str, Any]]:
    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid 'since' datetime format.")

    df = store.get_trades(account, since=since_dt, limit=limit)
    if df.empty:
        return []
    # Convert datetime columns to ISO strings for JSON serialization
    for col in df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df.to_dict("records")


# ── Trade summary (equity curve, win rate) ────────────────────────────────────

@app.get("/api/v1/trades/{account}/summary", tags=["trading"])
def get_trade_summary(account: str, store=Depends(get_trade_store)) -> dict[str, Any]:
    df = store.get_trades(account, limit=5000)
    if df.empty:
        return {"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0}

    total = len(df)
    wins = int(df["is_win"].sum()) if "is_win" in df.columns else 0
    total_pnl = float(df["pnl"].sum()) if "pnl" in df.columns else 0.0
    avg_pnl = float(df["pnl"].mean()) if "pnl" in df.columns else 0.0

    return {
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total, 4) if total > 0 else 0.0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "max_profit": round(float(df["pnl"].max()), 2) if "pnl" in df.columns else 0.0,
        "max_loss": round(float(df["pnl"].min()), 2) if "pnl" in df.columns else 0.0,
    }


# ── Signals ───────────────────────────────────────────────────────────────────

@app.get("/api/v1/signals/{account}", tags=["trading"])
def get_signals(
    account: str,
    limit: int = Query(200, ge=1, le=2000),
    store=Depends(get_trade_store),
) -> list[dict[str, Any]]:
    df = store.get_signals(account, limit=limit)
    if df.empty:
        return []
    for col in df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df.to_dict("records")


# ── MLflow model registry ─────────────────────────────────────────────────────

@app.get("/api/v1/models", tags=["models"])
def list_models(tracker=Depends(get_tracker)) -> list[dict[str, Any]]:
    return tracker.search_runs(max_results=50)


@app.get("/api/v1/models/{model_name}/metrics", tags=["models"])
def get_model_metrics(model_name: str, tracker=Depends(get_tracker)) -> dict[str, float]:
    metrics = tracker.get_latest_metrics(model_name)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No production model found: '{model_name}'")
    return metrics


# ── Inference endpoint ────────────────────────────────────────────────────────

@app.post("/api/v1/predict", tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    """Run model inference on a feature vector. Loads model lazily on first call."""
    t0 = time.perf_counter()
    try:
        from xauusd_ai.infra._model_cache import get_predictor
        predictor = get_predictor(req.account)
        confidence, should_trade, side = predictor.score(req.features)
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        account=req.account,
        confidence=confidence,
        should_trade=should_trade,
        side=side,
        latency_ms=round(latency_ms, 2),
    )


# ── Evidently reports ─────────────────────────────────────────────────────────

@app.get("/api/v1/reports/{account}/{report_name}", tags=["monitoring"])
def get_report_url(
    account: str,
    report_name: str,
    storage=Depends(get_storage),
) -> dict[str, str]:
    url = storage.get_report_url(account, report_name)
    if not url:
        raise HTTPException(status_code=404, detail="Report not found or MinIO unavailable.")
    return {"url": url, "account": account, "report": report_name}


# ── Startup event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    engine = get_engine()
    from xauusd_ai.infra.db import init_tables
    try:
        init_tables(engine)
        logger.info("Database tables verified/created on startup.")
    except Exception as exc:
        logger.error("Failed to initialize DB tables: %s", exc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
