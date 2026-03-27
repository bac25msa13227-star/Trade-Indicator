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

import asyncio
import json as _json
import logging
import os
import threading as _threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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

# ── Dashboard constants ───────────────────────────────────────────────────────

_OUTPUTS = Path(os.getenv("OUTPUTS_PATH", "outputs"))
_STATIC_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "static"

_ACCOUNT_CFG: dict[str, dict[str, str]] = {
    "acc1": {
        "label": "ACC1 · Model v3 (Aggressive)",
        "status": "live_status_acc1.json",
        "trades": "live_closed_trades_acc1.csv",
        "trades_fallback": "live_closed_trades.csv",
        "signals": "paper_trade_signals_acc1.csv",
        "meta": "model_meta_acc1.json",
        "wf": "walkforward_report_acc1.json",
        "bt": "backtest_report_acc1.json",
        "bt_trades": "backtest_trades_acc1.csv",
        "peak": "risk_peak_balance_acc1.json",
        "daily": "risk_daily_state_acc1.json",
    },
    "acc2": {
        "label": "ACC2 · Model v2",
        "status": "live_status_acc2.json",
        "trades": "live_closed_trades_acc2.csv",
        "trades_fallback": "live_closed_trades_acc2.csv",
        "signals": "paper_trade_signals_acc2.csv",
        "meta": "model_meta_acc2.json",
        "wf": "walkforward_report_acc2.json",
        "bt": "backtest_report_acc2.json",
        "bt_trades": "backtest_trades_acc2.csv",
        "peak": "risk_peak_balance_acc2.json",
        "daily": "risk_daily_state_acc2.json",
    },
}


# ── WebSocket connection manager ──────────────────────────────────────────────

class _WSManager:
    def __init__(self) -> None:
        self._conns: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._conns.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        try:
            self._conns.remove(ws)
        except ValueError:
            pass

    async def broadcast(self, text: str) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._conns):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


_ws_manager = _WSManager()


# ── Dashboard data builder ────────────────────────────────────────────────────

def _safe_json(path: Path) -> dict:
    if not path.exists():
        return {}
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return _json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    return {}


def _safe_csv_tail(path: Path, n: int = 30) -> list[dict]:
    """Read last n rows of a CSV without loading the entire file into memory."""
    try:
        if not path.exists():
            return []
        import csv as _csv
        from collections import deque as _deque
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = _csv.DictReader(f)
            rows = list(_deque(reader, maxlen=n))
        # Replace empty strings with None to match previous pandas NaN->None behavior
        return [{k: (v if v != "" else None) for k, v in row.items()} for row in rows]
    except Exception:
        return []


def _build_pnl_series(trades: list[dict]) -> dict:
    """Build equity / PnL series for Chart.js. Uses balance_after if present, else cumulative PnL."""
    if not trades:
        return {"labels": [], "values": [], "individual": []}
    labels, values, individual = [], [], []
    running = 0.0
    use_balance = "balance_after" in trades[0]
    for t in trades:
        try:
            pnl = float(t.get("pnl") or t.get("profit") or 0)
        except (TypeError, ValueError):
            pnl = 0.0
        ts = str(t.get("time") or t.get("close_time") or t.get("ts") or "")[:16]
        if use_balance:
            try:
                val = float(t.get("balance_after") or 0)
            except (TypeError, ValueError):
                val = running
        else:
            running = round(running + pnl, 2)
            val = running
        labels.append(ts)
        values.append(val)
        individual.append(round(pnl, 2))
    return {"labels": labels, "values": values, "individual": individual}


_dashboard_cache: dict = {}
_dashboard_cache_ts: float = 0.0
_dashboard_cache_lock = _threading.Lock()
_DASHBOARD_CACHE_TTL = 30.0  # seconds — longer than build time to avoid thrashing

# ── News constants & helpers ──────────────────────────────────────────────────

_FF_NEWS_WEEK_URL  = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_FF_NEWS_MONTH_URL = "https://nfs.faireconomy.media/ff_calendar_thismonth.json"
_NEWS_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; XAUBot/1.0)"}
_NEWS_CACHE_TTL = 300.0   # 5 min in-memory cache for news

_NEWS_GOLD_DIR: dict[str, int] = {
    "Non-Farm Payroll": -1, "Nonfarm Payrolls": -1, "NF Payrolls": -1,
    "ADP Employment": -1, "ADP Nonfarm": -1,
    "Jobless Claims": +1, "Initial Claims": +1, "Unemployment Claims": +1,
    "Unemployment Rate": +1,
    "CPI": +1, "Consumer Price Index": +1, "Core CPI": +1,
    "PCE": +1, "Core PCE": +1, "Personal Consumption": +1,
    "PPI": +1, "Producer Price": +1, "Inflation": +1,
    "GDP": -1, "Gross Domestic Product": -1,
    "Retail Sales": -1, "ISM": -1, "PMI": -1, "Purchasing Managers": -1,
    "Durable Goods": -1, "Housing": -1, "Building Permits": -1,
    "New Home Sales": -1, "Consumer Confidence": -1, "Consumer Sentiment": -1,
    "Michigan": -1, "FOMC": -1, "Federal Reserve": -1, "Fed Funds": -1,
    "Interest Rate": -1, "Powell": -1, "Trade Balance": -1, "Current Account": -1,
}


def _news_base_dir(title: str) -> int:
    t = title.lower()
    for kw, d in _NEWS_GOLD_DIR.items():
        if kw.lower() in t:
            return d
    return 0


def _parse_news_num(s: str) -> float | None:
    """Parse news value strings like '155K', '8.3%', '-2.1B'."""
    if not s:
        return None
    sv = str(s).strip()
    mul = 1.0
    if sv.endswith(("B", "b")):
        mul, sv = 1e9, sv[:-1]
    elif sv.endswith(("M", "m")):
        mul, sv = 1e6, sv[:-1]
    elif sv.endswith(("K", "k")):
        mul, sv = 1e3, sv[:-1]
    sv = sv.rstrip("%").replace(",", "")
    try:
        return float(sv) * mul
    except ValueError:
        return None


def _compute_gold_impact(title: str, actual: str, forecast: str, previous: str, currency: str = "USD") -> dict[str, Any]:
    """Compute gold impact label and direction for a news event."""
    base = _news_base_dir(title)
    is_usd = currency.upper() in ("USD", "US", "USA", "")
    if not is_usd and base != 0:
        # Non-USD: strong foreign economy → weaker USD → gold UP
        # Invert direction for economic-strength events.
        # Exception: global inflation events (CPI/PPI/PCE) affect gold the same way.
        tl = title.lower()
        is_inflation = any(kw in tl for kw in ("cpi", "pce", "ppi", "inflation", "price index"))
        if not is_inflation:
            base = -base
    act_f  = _parse_news_num(actual)
    for_f  = _parse_news_num(forecast)
    prev_f = _parse_news_num(previous)
    released = act_f is not None

    if not released:
        if base > 0:
            label, sentiment = "📈 Bullish vàng nếu vượt dự báo", "bullish_potential"
        elif base < 0:
            label, sentiment = "📉 Bearish vàng nếu vượt dự báo", "bearish_potential"
        else:
            label, sentiment = "⚪ Trung tính", "neutral"
        return {"direction": 0, "label": label, "sentiment": sentiment, "released": False}

    # Released — compute from surprise
    surprise = 0
    if for_f is not None:
        if act_f > for_f:
            surprise = base
        elif act_f < for_f:
            surprise = -base
    elif prev_f is not None:
        if act_f > prev_f:
            surprise = base
        elif act_f < prev_f:
            surprise = -base

    if surprise > 0:
        label, sentiment = "🟢 Tốt cho vàng (Bullish Gold)", "bullish"
    elif surprise < 0:
        label, sentiment = "🔴 Xấu cho vàng (Bearish Gold)", "bearish"
    else:
        label, sentiment = "⚪ Trung tính (đúng dự báo)", "neutral"
    return {"direction": surprise, "label": label, "sentiment": sentiment, "released": True}


def _parse_ff_event_dt(ev: dict) -> "datetime | None":
    """Parse datetime from a raw ForexFactory event dict."""
    from datetime import datetime as _dt, timezone as _tz
    for key in ("date", "datetime", "time"):
        raw = ev.get(key)
        if not raw:
            continue
        try:
            ts = _dt.fromisoformat(str(raw))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_tz.utc)
            return ts.astimezone(_tz.utc)
        except Exception:
            pass
    return None


_news_cache: dict = {}
_news_cache_ts: float = 0.0
_news_cache_lock = _threading.Lock()
# Track sent Telegram alerts: "{event_id}_{milestone}"
_news_alerted: set[str] = set()


def _build_news_payload() -> dict[str, Any]:
    """Return news calendar (5-min in-memory cache)."""
    global _news_cache, _news_cache_ts
    now = time.monotonic()
    with _news_cache_lock:
        if _news_cache and now - _news_cache_ts < _NEWS_CACHE_TTL:
            return _news_cache
        result = _fetch_news_uncached()
        _news_cache = result
        _news_cache_ts = now
        return result


def _fetch_news_uncached() -> dict[str, Any]:
    """Directly fetch ForexFactory JSON and process events."""
    import requests as _req
    from datetime import datetime as _dt, timezone as _tz

    now_utc = _dt.now(_tz.utc)
    today_date = now_utc.date()

    raw_events: list[dict] = []
    for url in [_FF_NEWS_WEEK_URL, _FF_NEWS_MONTH_URL]:
        try:
            r = _req.get(url, headers=_NEWS_HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                raw_events = data
                break
        except Exception as exc:
            logger.debug("News fetch from %s failed: %s", url, exc)

    # Currencies that directly influence gold (skip exotic/irrelevant ones)
    _GOLD_CURRENCIES = {
        "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "CNY", "CNH",
        "US", "USA",
    }

    events: list[dict] = []
    for ev in raw_events:
        title    = str(ev.get("title", "") or ev.get("name", "")).strip()
        country  = str(ev.get("country", "") or ev.get("currency", "") or "").strip().upper()
        impact   = str(ev.get("impact", "") or "").capitalize().strip()
        # Skip non-tradable entries
        if impact in ("", "Holiday", "Non-Economic", "Non-trading"):
            continue
        if not title:
            continue
        # Only include currencies that affect gold
        if country and country not in _GOLD_CURRENCIES:
            continue

        ev_dt = _parse_ff_event_dt(ev)
        if ev_dt is None:
            continue

        actual   = str(ev.get("actual",   "") or "").strip()
        forecast = str(ev.get("forecast", "") or "").strip()
        previous = str(ev.get("previous", "") or "").strip()

        gold_impact = _compute_gold_impact(title, actual, forecast, previous, country)
        diff_min = (ev_dt - now_utc).total_seconds() / 60
        ev_date = ev_dt.date()

        events.append({
            "id":           f"{ev_dt.strftime('%Y%m%d_%H%M')}_{title[:20].replace(' ', '_')}",
            "date":         ev_dt.strftime("%Y-%m-%d"),
            "time_utc":     ev_dt.strftime("%H:%M"),
            "datetime_iso": ev_dt.isoformat(),
            "event":        title,
            "currency":     country,
            "impact":       impact,
            "forecast":     forecast or None,
            "previous":     previous or None,
            "actual":       actual   or None,
            "released":     gold_impact["released"],
            "gold_direction": gold_impact["direction"],
            "gold_label":   gold_impact["label"],
            "gold_sentiment": gold_impact["sentiment"],
            "minutes_until": round(diff_min, 1),
            "is_today":     ev_date == today_date,
        })

    events.sort(key=lambda x: x["datetime_iso"])
    today_events    = [e for e in events if e["is_today"]]
    upcoming_events = [e for e in events if e["minutes_until"] > 0]
    next_high = next((e for e in upcoming_events if e["impact"] == "High"), None)

    return {
        "ts":               now_utc.isoformat(),
        "today":            today_events,
        "week":             events,
        "next_high_impact": next_high,
    }


def _build_dashboard_payload() -> dict:
    global _dashboard_cache, _dashboard_cache_ts
    now = time.monotonic()
    # Fast path: cache is fresh — return immediately
    if _dashboard_cache and now - _dashboard_cache_ts < _DASHBOARD_CACHE_TTL:
        return _dashboard_cache
    # Try to acquire the lock without waiting (non-blocking)
    acquired = _dashboard_cache_lock.acquire(blocking=False)
    if not acquired:
        # Another thread is already rebuilding — return stale cache if available
        if _dashboard_cache:
            return _dashboard_cache
        # No cache at all — wait for the lock
        _dashboard_cache_lock.acquire(blocking=True)
    try:
        # Re-check inside lock (another thread may have just finished)
        now2 = time.monotonic()
        if _dashboard_cache and now2 - _dashboard_cache_ts < _DASHBOARD_CACHE_TTL:
            return _dashboard_cache
        result = _build_dashboard_payload_uncached()
        _dashboard_cache = result
        _dashboard_cache_ts = now2
        return result
    finally:
        _dashboard_cache_lock.release()


def _build_dashboard_payload_uncached() -> dict:
    accounts: dict[str, Any] = {}
    for acct, cfg in _ACCOUNT_CFG.items():
        daily    = _safe_json(_OUTPUTS / cfg.get("daily", "risk_daily_state.json"))
        peak_bal = float(_safe_json(_OUTPUTS / cfg.get("peak", "risk_peak_balance.json")).get("peak_balance") or 0)
        status = _safe_json(_OUTPUTS / cfg["status"])
        meta   = _safe_json(_OUTPUTS / cfg["meta"])
        wf     = _safe_json(_OUTPUTS / cfg["wf"])
        bt     = _safe_json(_OUTPUTS / cfg["bt"])

        trades_path = _OUTPUTS / cfg["trades"]
        if not trades_path.exists():
            trades_path = _OUTPUTS / cfg["trades_fallback"]
        trades   = _safe_csv_tail(trades_path, 200)
        signals  = _safe_csv_tail(_OUTPUTS / cfg["signals"], 20)
        bt_trades = _safe_csv_tail(_OUTPUTS / cfg.get("bt_trades", ""), 500) if cfg.get("bt_trades") else []

        bal = float(status.get("account_balance") or 0)
        dd  = round((peak_bal - bal) / peak_bal * 100, 2) if peak_bal > 0 else 0.0

        wins = 0
        total_pnl = 0.0
        if trades:
            try:
                for t in trades:
                    pnl_val = float(t.get("pnl") or t.get("profit") or 0)
                    is_win_field = str(t.get("is_win", "")).lower() in ("true", "1") or t.get("is_win") is True
                    if is_win_field or pnl_val > 0:
                        wins += 1
                    total_pnl += pnl_val
            except Exception:
                pass
        win_rate  = round(wins / len(trades), 4) if trades else 0.0
        total_pnl = round(total_pnl, 2)

        # Walkforward folds
        wf_folds: list[dict] = []
        for fold in wf.get("folds", []):
            sim = fold.get("concurrent_sim", {})
            wf_folds.append({
                "fold":        fold.get("fold"),
                "train_start": fold.get("train_start"),
                "train_end":   fold.get("train_end"),
                "test_start":  fold.get("test_start"),
                "test_end":    fold.get("test_end"),
                "threshold":   fold.get("threshold"),
                "precision":   fold.get("precision"),
                "roc_auc":     fold.get("roc_auc"),
                "n_signals":   fold.get("n_signals"),
                "win_rate":    sim.get("win_rate"),
                "profit_factor": sim.get("profit_factor"),
                "return_pct":  sim.get("return_pct"),
                "max_dd":      sim.get("max_drawdown_pct"),
                "trades":      sim.get("trades"),
            })

        wf_agg = wf.get("aggregate", {})
        wf_cfg = wf.get("walk_forward", {})

        accounts[acct] = {
            "label":   cfg["label"],
            "status":  status,
            "model": {
                "precision":         meta.get("precision"),
                "threshold":         meta.get("threshold") or meta.get("decision_threshold"),
                "roc_auc":           meta.get("roc_auc"),
                "recall":            meta.get("recall"),
                "f1":                meta.get("f1"),
                "accuracy":          meta.get("accuracy"),
                "train_rows":        meta.get("train_rows"),
                "test_rows":         meta.get("test_rows"),
                "wf_precision_avg":  wf_agg.get("avg_precision"),
                "wf_win_rate_avg":   wf_agg.get("signal_win_rate"),
                "wf_n_folds":        wf_cfg.get("n_folds") or len(wf.get("folds", [])),
            },
            "backtest": bt,
            "backtest_equity": _build_pnl_series(bt_trades),
            "walkforward": {
                "aggregate": wf_agg,
                "folds":     wf_folds,
                "config": {
                    "n_folds":    wf_cfg.get("n_folds"),
                    "train_bars": wf_cfg.get("train_bars"),
                    "test_bars":  wf_cfg.get("test_bars"),
                    "model":      wf_cfg.get("model"),
                },
            },
            "daily": {
                "daily_loss":         daily.get("daily_loss", 0),
                "consecutive_losses": daily.get("consecutive_losses", 0),
                "cooldown_bars":      daily.get("cooldown_bars", 0),
                "killed":             daily.get("killed", False),
            },
            "peak_balance":   peak_bal or float(status.get("account_balance") or 0),
            "drawdown_pct":   dd,
            "win_rate":       win_rate,
            "total_pnl":      total_pnl,
            "recent_trades":  trades[-50:],
            "recent_signals": signals,
            "pnl_series":     _build_pnl_series(trades),
        }

    return {"ts": datetime.utcnow().isoformat(), "accounts": accounts}


# ── Background broadcast loop ─────────────────────────────────────────────────

async def _broadcast_loop() -> None:
    """Keep dashboard cache warm every 30s; push to WS clients when connected."""
    while True:
        await asyncio.sleep(30)  # matches cache TTL
        try:
            loop = asyncio.get_running_loop()
            payload = await loop.run_in_executor(None, _build_dashboard_payload)
            if _ws_manager._conns:
                await _ws_manager.broadcast(_json.dumps(payload, default=str))
        except Exception as exc:
            logger.error("Dashboard broadcast error: %s", exc)


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


@app.get("/api/v1/news", tags=["trading"])
async def get_news_calendar() -> dict[str, Any]:
    """Today's USD economic news calendar with gold impact interpretation."""
    loop = asyncio.get_running_loop()
    payload = await loop.run_in_executor(None, _build_news_payload)
    return payload


# ── REST dashboard snapshot (HTTP polling fallback) ─────────────────────────

@app.get("/api/v1/dashboard", tags=["dashboard"])
async def get_dashboard_snapshot() -> dict[str, Any]:
    """Full dashboard payload — used as HTTP polling fallback when WS unavailable."""
    loop = asyncio.get_running_loop()
    payload = await loop.run_in_executor(None, _build_dashboard_payload)
    return payload


# ── WebSocket dashboard endpoint ─────────────────────────────────────────────

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """Real-time dashboard data stream."""
    await ws.accept()
    try:
        loop = asyncio.get_running_loop()
        # Send immediately from cache (instant if cache is warm)
        first_payload = await loop.run_in_executor(None, _build_dashboard_payload)
        await ws.send_text(_json.dumps(first_payload, default=str))
        while True:
            # Wait for next broadcast cycle — the broadcast loop keeps cache fresh
            await asyncio.sleep(30)
            payload = await loop.run_in_executor(None, _build_dashboard_payload)
            await ws.send_text(_json.dumps(payload, default=str))
    except Exception as _exc:
        logger.debug("ws_live closed: %s", _exc)


@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard() -> FileResponse:
    """Serve the WebSocket-powered HTML dashboard."""
    html = _STATIC_DIR / "index.html"
    if html.exists():
        return FileResponse(str(html), media_type="text/html")
    raise HTTPException(status_code=404, detail="Dashboard HTML not found at " + str(html))


# ── Telegram news alerter ─────────────────────────────────────────────────────

def _tg_news_send(text: str) -> None:
    """Send Telegram message for news alerts (tolerates missing credentials)."""
    import requests as _req
    token   = os.getenv("TELEGRAM_BOT_TOKEN_ACC2") or os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID_ACC2")   or os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        resp = _req.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            # Retry without HTML parse mode
            _req.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=10,
            )
    except Exception as exc:
        logger.debug("Telegram news alert send error: %s", exc)


async def _news_alert_loop() -> None:
    """Background task: check every 60s and alert upcoming / just-released news."""
    await asyncio.sleep(30)   # short delay after startup
    while True:
        try:
            loop = asyncio.get_running_loop()
            payload = await loop.run_in_executor(None, _build_news_payload)
            for ev in payload.get("week", []):
                if ev["impact"] not in ("High",):
                    continue
                eid   = ev["id"]
                mins  = ev["minutes_until"]
                rel   = ev["released"]

                if rel:
                    key = f"{eid}_released"
                    if key not in _news_alerted:
                        _news_alerted.add(key)
                        actual   = ev.get("actual")   or "—"
                        forecast = ev.get("forecast") or "—"
                        previous = ev.get("previous") or "—"
                        label    = ev.get("gold_label", "")
                        msg = (
                            f"📊 <b>{ev['event']}</b> — Đã phát hành!\n"
                            f"🕐 {ev['time_utc']} UTC ({ev['date']})\n"
                            f"⬅️ Trước: <b>{previous}</b> &nbsp; 🎯 Dự báo: <b>{forecast}</b> &nbsp; ✅ Thực tế: <b>{actual}</b>\n"
                            f"🏅 Vàng: {label}"
                        )
                        await loop.run_in_executor(None, _tg_news_send, msg)

                elif 0 < mins <= 32:
                    for milestone, milestone_lbl in [(30, "30 phút"), (5, "5 phút")]:
                        if mins <= (milestone + 2):
                            key = f"{eid}_{milestone}min"
                            if key not in _news_alerted:
                                _news_alerted.add(key)
                                icon    = "🔴" if ev["impact"] == "High" else "🟡"
                                forecast = ev.get("forecast") or "—"
                                previous = ev.get("previous") or "—"
                                label   = ev.get("gold_label", "")
                                msg = (
                                    f"{icon} <b>Tin {ev['impact']} sắp ra — {milestone_lbl} nữa!</b>\n"
                                    f"📰 <b>{ev['event']}</b> ({ev['currency']})\n"
                                    f"🕐 {ev['time_utc']} UTC\n"
                                    f"🎯 Dự báo: <b>{forecast}</b> &nbsp; ⬅️ Trước: <b>{previous}</b>\n"
                                    f"💡 {label}"
                                )
                                await loop.run_in_executor(None, _tg_news_send, msg)
        except Exception as exc:
            logger.error("News alert loop error: %s", exc)
        await asyncio.sleep(60)


# ── Startup event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup() -> None:
    # Run DB init in a thread pool so it never blocks the event loop
    loop = asyncio.get_running_loop()
    try:
        engine = get_engine()
        from xauusd_ai.infra.db import init_tables
        await asyncio.wait_for(
            loop.run_in_executor(None, lambda: init_tables(engine)),
            timeout=20.0,
        )
        logger.info("Database tables verified/created on startup.")
    except asyncio.TimeoutError:
        logger.warning("DB init timed out (>20s) — continuing without DB tables.")
    except Exception as exc:
        logger.error("Failed to initialize DB tables: %s", exc)
    asyncio.create_task(_broadcast_loop())
    logger.info("WebSocket dashboard broadcast loop started.")
    asyncio.create_task(_news_alert_loop())
    logger.info("News Telegram alert loop started.")
    # Pre-warm the dashboard cache in the background (don't block startup)
    async def _warmup():
        try:
            await loop.run_in_executor(None, _build_dashboard_payload)
            logger.info("Dashboard cache pre-warmed.")
        except Exception as e:
            logger.warning("Dashboard cache warmup failed: %s", e)
    asyncio.create_task(_warmup())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
