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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from xauusd_ai.config import load_settings

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
_LIVE_CFG_MAP: dict[str, Path] = {
    "acc1": Path(os.getenv("ACC1_SETTINGS_PATH", "configs/live_acc1.yaml")),
    "acc2": Path(os.getenv("ACC2_SETTINGS_PATH", "configs/live_acc2.yaml")),
}
_REQUIRED_MODEL_BINDINGS: dict[str, dict[str, str]] = {
    "acc1": {
        "model": "outputs/acc1_breakthrough_net66590_dd3953_model.pkl",
        "scaler": "outputs/acc1_breakthrough_net66590_dd3953_scaler.pkl",
        "meta": "outputs/acc1_breakthrough_net66590_dd3953_model_meta.json",
    },
    "acc2": {
        "model": "outputs/acc2_breakthrough_net21k_dd2333_model.pkl",
        "scaler": "outputs/acc2_breakthrough_net21k_dd2333_scaler.pkl",
        "meta": "outputs/acc2_breakthrough_net21k_dd2333_model_meta.json",
    },
}
_BENCHMARK_VERIFY_FILE = Path(os.getenv("BENCHMARK_VERIFY_FILE", "outputs/wf_exact_recovery_verify_2016501.json"))
_ACCOUNT_RUNTIME_CFG: dict[str, dict[str, Any]] = {}
_PROFILE_MODES = {"conservative", "balanced", "aggressive"}
_RUNTIME_PROFILE_OVERRIDE_TEMPLATE = "runtime_profile_override_{account}.json"

_ACCOUNT_CFG: dict[str, dict[str, str]] = {
    "acc1": {
        "label": "ACC1 · Live Profile",
        "status": "live_status_acc1.json",
        "trades": "live_closed_trades_acc1.csv",
        "trades_fallback": "live_closed_trades.csv",
        "journal": "trade_journal_acc1.jsonl",
        "signals": "paper_trade_signals_acc1.csv",
        "meta": "acc1_live_model_meta.json",
        "wf": "walkforward_report_acc1.json",
        "bt": "backtest_report_acc1.json",
        "bt_trades": "backtest_trades_acc1.csv",
        "peak": "risk_peak_balance_acc1.json",
        "daily": "risk_daily_state_acc1.json",
    },
    "acc2": {
        "label": "ACC2 · Live Profile",
        "status": "live_status_acc2.json",
        "trades": "live_closed_trades_acc2.csv",
        "trades_fallback": "live_closed_trades_acc2.csv",
        "journal": "trade_journal_acc2.jsonl",
        "signals": "paper_trade_signals_acc2.csv",
        "meta": "acc2_live_model_meta.json",
        "wf": "walkforward_report_acc2.json",
        "bt": "backtest_report_acc2.json",
        "bt_trades": "backtest_trades_acc2.csv",
        "peak": "risk_peak_balance_acc2.json",
        "daily": "risk_daily_state_acc2.json",
    },
}

_SIGNAL_COLUMNS_V1 = [
    "time",
    "should_trade",
    "side",
    "confidence",
    "reason",
    "entry_price",
    "stop_loss",
    "take_profit",
    "volume",
    "strategy_score",
    "volatility_regime",
    "account_balance",
    "open_positions",
    "max_positions",
]

_SIGNAL_COLUMNS_V2 = [
    "time",
    "should_trade",
    "side",
    "confidence",
    "reason",
    "entry_price",
    "stop_loss",
    "take_profit",
    "volume",
    "strategy_score",
    "strategy_raw_score",
    "regime_bias",
    "strategy_required_min",
    "strategy_gate_pass",
    "ict_score",
    "wyckoff_score",
    "momentum_score",
    "ict_weight",
    "wyckoff_weight",
    "momentum_weight",
    "strategy_total_weight",
    "volatility_regime",
    "account_balance",
    "open_positions",
    "max_positions",
]


def _output_path(path_like: str | Path) -> Path:
    """Resolve output file path robustly for absolute/relative/basename inputs."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    # Keep "outputs/..." relative paths as-is; they are already anchored.
    if p.parts and p.parts[0] == "outputs":
        return p
    return _OUTPUTS / p


def _runtime_profile_override_path(account: str) -> Path:
    name = _RUNTIME_PROFILE_OVERRIDE_TEMPLATE.format(account=account.strip().lower())
    return _output_path(name)


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_runtime_profile_override(account: str) -> dict[str, Any]:
    path = _runtime_profile_override_path(account)
    raw = _safe_json(path)
    if not isinstance(raw, dict):
        return {}
    profile = str(raw.get("profile", "")).strip().lower()
    if profile not in _PROFILE_MODES:
        return {}
    raw["profile"] = profile
    raw["path"] = str(path)
    return raw


def _path_matches_required(actual: str | Path, required: str | Path) -> bool:
    """Allow both exact relative paths and absolute paths ending with the required suffix."""
    actual_posix = Path(str(actual)).as_posix()
    required_posix = Path(str(required)).as_posix()
    return actual_posix == required_posix or actual_posix.endswith(f"/{required_posix}")


def _load_benchmark_targets() -> dict[str, Any]:
    """Load locked benchmark values used for ACC1/ACC2 profile verification."""
    payload = _safe_json(_output_path(_BENCHMARK_VERIFY_FILE))
    result: dict[str, Any] = {
        "source": str(_output_path(_BENCHMARK_VERIFY_FILE)),
        "acc1": {},
        "acc2": {},
    }
    for acct, key in (("acc1", "acc1_expected"), ("acc2", "acc2_expected")):
        raw = payload.get(key, {}) if isinstance(payload, dict) else {}
        if not isinstance(raw, dict):
            continue
        result[acct] = {
            "net_profit": raw.get("net"),
            "max_drawdown_pct": raw.get("dd"),
            "profit_factor": raw.get("pf"),
            "trades": raw.get("trades"),
            "win_rate": raw.get("wr"),
        }
    return result


def _apply_live_config_overrides() -> None:
    """Use live YAML configs as source of truth for dashboard artifact filenames."""
    _ACCOUNT_RUNTIME_CFG.clear()
    for acct, cfg_path in _LIVE_CFG_MAP.items():
        if acct not in _ACCOUNT_CFG:
            continue
        try:
            settings = load_settings(cfg_path)
            app_cfg = settings.app
            _ACCOUNT_CFG[acct]["label"] = f"{acct.upper()} · {Path(app_cfg.model_path).name}"
            _ACCOUNT_CFG[acct]["meta"] = Path(app_cfg.model_meta_path).name
            _ACCOUNT_CFG[acct]["wf"] = Path(app_cfg.walkforward_report_path).name
            _ACCOUNT_CFG[acct]["bt"] = Path(app_cfg.backtest_report_path).name
            _ACCOUNT_CFG[acct]["bt_trades"] = Path(app_cfg.backtest_trades_path).name
            _ACCOUNT_CFG[acct]["signals"] = Path(app_cfg.paper_trade_log_path).name
            _ACCOUNT_CFG[acct]["trades"] = Path(app_cfg.live_closed_trades_path).name
            _account_suffix = Path(app_cfg.live_closed_trades_path).stem.replace("live_closed_trades_", "").strip("_") or acct
            _ACCOUNT_CFG[acct]["journal"] = str(
                Path(app_cfg.live_closed_trades_path).with_name(f"trade_journal_{_account_suffix}.jsonl")
            )

            required_binding = _REQUIRED_MODEL_BINDINGS.get(acct, {})
            actual_binding = {
                "model": str(app_cfg.model_path),
                "scaler": str(app_cfg.scaler_path),
                "meta": str(app_cfg.model_meta_path),
            }
            mismatches = [
                key
                for key, required_path in required_binding.items()
                if not _path_matches_required(actual_binding.get(key, ""), required_path)
            ]
            model_threshold_cfg = float(settings.strategy.signal_threshold)
            risk_threshold = float(settings.risk.min_confidence)
            _ACCOUNT_RUNTIME_CFG[acct] = {
                "config_path": str(cfg_path),
                "model_binding_required": required_binding,
                "model_binding_actual": actual_binding,
                "model_binding_ok": len(mismatches) == 0,
                "model_binding_mismatch": mismatches,
                "threshold_config": model_threshold_cfg,
                "risk_min_confidence": risk_threshold,
                # threshold_effective = strategy.signal_threshold (the ML gate).
                # risk.min_confidence is a separate execution guard, not the signal threshold.
                "threshold_effective_config": model_threshold_cfg,
                "sideway_min_confidence": float(settings.strategy.sideway_min_confidence),
                "volatile_min_confidence": float(settings.strategy.volatile_min_confidence),
                "min_strategy_score": float(settings.strategy.min_strategy_score),
                "sideway_min_strategy_score": float(settings.strategy.sideway_min_strategy_score),
                "strong_volatility_min_strategy_score": float(settings.strategy.strong_volatility_min_strategy_score),
                "require_trend_alignment": bool(settings.strategy.require_trend_alignment),
                "adx_gate_enabled": bool(settings.strategy.adx_gate_enabled),
                "adx_min_trend": float(settings.strategy.adx_min_trend),
                "max_open_positions": int(settings.risk.max_open_positions),
                "kill_switch_enabled": bool(settings.risk.kill_switch_enabled),
                "daily_loss_limit_pct": float(settings.risk.daily_loss_limit_pct),
                "max_drawdown_kill_pct": float(settings.risk.max_drawdown_kill_pct),
                "consecutive_loss_pause_count": int(settings.risk.consecutive_loss_pause_count),
                "consecutive_loss_cooldown_bars": int(settings.risk.consecutive_loss_cooldown_bars),
                "admin_matrix_windows_days": [int(v) for v in settings.admin.regime_session_matrix_windows_days],
                "canary_enabled": bool(settings.admin.canary.enabled),
                "canary_volume_fraction": float(settings.admin.canary.volume_fraction),
                "auto_rollback_enabled": bool(settings.admin.auto_rollback.enabled),
                "auto_rollback_profile": str(settings.admin.auto_rollback.rollback_profile),
                "auto_rollback_daily_dd_trigger_pct": float(settings.admin.auto_rollback.daily_dd_trigger_pct),
                "auto_rollback_consecutive_losses_trigger": int(settings.admin.auto_rollback.consecutive_losses_trigger),
                "auto_rollback_cooldown_seconds": int(settings.admin.auto_rollback.cooldown_seconds),
                "data_health_enabled": bool(settings.admin.data_health.enabled),
                "data_health_alert_cooldown_seconds": int(settings.admin.data_health.alert_cooldown_seconds),
                "data_health_missing_bar_gap_factor": float(settings.admin.data_health.missing_bar_gap_factor),
                "data_health_stale_tick_alert_seconds": int(settings.admin.data_health.stale_tick_alert_seconds),
                "data_health_spread_spike_multiplier": float(settings.admin.data_health.spread_spike_multiplier),
                "data_health_spread_spike_abs_points": float(settings.admin.data_health.spread_spike_abs_points),
                "data_health_spread_lookback_bars": int(settings.admin.data_health.spread_lookback_bars),
                "data_health_bridge_feed_divergence_points": float(settings.admin.data_health.bridge_feed_divergence_points),
                "auto_trade_enabled": bool(settings.execution.auto_trade),
            }
        except Exception as exc:
            logger.warning("Could not apply live config override for %s from %s: %s", acct, cfg_path, exc)


_apply_live_config_overrides()


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


def _as_float(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> int | None:
    try:
        if v is None or v == "":
            return None
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _row_to_dict(header: list[str], row: list[str]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for idx, col in enumerate(header):
        mapped[col] = row[idx] if idx < len(row) else None
    return mapped


def _normalize_signal_row(row: dict[str, Any], account: str | None = None) -> dict[str, Any]:
    out = dict(row)
    for col in _SIGNAL_COLUMNS_V2:
        out.setdefault(col, None)

    # Common aliases used by dashboard JS.
    out["direction"] = out.get("direction") or out.get("side")
    out["signal"] = out.get("signal") or out.get("side")
    out["sl"] = out.get("sl") or out.get("stop_loss")
    out["tp"] = out.get("tp") or out.get("take_profit")

    strategy_score = _as_float(out.get("strategy_score"))
    vol_regime = _as_int(out.get("volatility_regime"))
    if out.get("regime_bias") in (None, "") and vol_regime is not None:
        out["regime_bias"] = 0.5 if vol_regime == 0 else (1.2 if vol_regime == 2 else 1.0)

    if out.get("strategy_required_min") in (None, ""):
        runtime_cfg = _ACCOUNT_RUNTIME_CFG.get(account or "", {})
        base_min = _as_float(runtime_cfg.get("min_strategy_score")) or 0.05
        side_min = _as_float(runtime_cfg.get("sideway_min_strategy_score")) or 0.10
        strong_min = _as_float(runtime_cfg.get("strong_volatility_min_strategy_score")) or 0.35
        if vol_regime == 0:
            out["strategy_required_min"] = side_min
        elif vol_regime == 2:
            out["strategy_required_min"] = strong_min
        else:
            out["strategy_required_min"] = base_min

    # Fallback for legacy logs (14 columns): keep dashboard columns populated.
    if out.get("ict_score") in (None, "") and strategy_score is not None:
        out["ict_score"] = strategy_score
    if out.get("wyckoff_score") in (None, "") and strategy_score is not None:
        out["wyckoff_score"] = strategy_score
    if out.get("momentum_score") in (None, "") and strategy_score is not None:
        out["momentum_score"] = strategy_score

    if out.get("strategy_gate_pass") in (None, ""):
        req = _as_float(out.get("strategy_required_min"))
        if strategy_score is not None and req is not None:
            out["strategy_gate_pass"] = int(abs(strategy_score) >= req)

    return out


def _decode_signal_row(header: list[str], row: list[str], account: str | None = None) -> dict[str, Any]:
    clean_header = [str(h).strip() for h in header]
    is_v1_header = clean_header == _SIGNAL_COLUMNS_V1

    # New signal rows appended to old 14-col header file.
    if is_v1_header and len(row) >= len(_SIGNAL_COLUMNS_V2):
        mapped = _row_to_dict(_SIGNAL_COLUMNS_V2, row[: len(_SIGNAL_COLUMNS_V2)])
        return _normalize_signal_row(mapped, account=account)

    # Legacy 14-col row.
    if is_v1_header and len(row) == len(_SIGNAL_COLUMNS_V1):
        mapped = _row_to_dict(_SIGNAL_COLUMNS_V1, row)
        return _normalize_signal_row(mapped, account=account)

    # Proper V2 header file.
    if len(clean_header) >= len(_SIGNAL_COLUMNS_V2):
        mapped = _row_to_dict(clean_header, row)
        return _normalize_signal_row(mapped, account=account)

    # Fallback best-effort.
    mapped = _row_to_dict(clean_header, row)
    return _normalize_signal_row(mapped, account=account)


def _normalize_trade_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["entry_price"] = out.get("entry_price") or out.get("open_price")
    out["stop_loss"] = out.get("stop_loss") or out.get("sl")
    out["take_profit"] = out.get("take_profit") or out.get("tp")
    out["sl"] = out.get("sl") or out.get("stop_loss")
    out["tp"] = out.get("tp") or out.get("take_profit")
    return out


def _safe_csv_tail(path: Path, n: int = 30, *, account: str | None = None) -> list[dict]:
    """Read last n rows of a CSV and normalize legacy/new schemas for dashboard."""
    try:
        if not path.exists():
            return []
        import csv as _csv
        from collections import deque as _deque

        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = _csv.reader(f)
            header = next(reader, None)
            if not header:
                return []
            rows = list(_deque(reader, maxlen=n))

        is_signal_file = "paper_trade_signals" in path.name
        is_live_trade_file = "live_closed_trades" in path.name
        result: list[dict] = []
        for row in rows:
            if is_signal_file:
                mapped = _decode_signal_row(header, row, account=account)
            else:
                mapped = _row_to_dict(header, row)
                if is_live_trade_file:
                    mapped = _normalize_trade_row(mapped)
            result.append({k: (v if v != "" else None) for k, v in mapped.items()})
        return result
    except Exception:
        return []


def _safe_jsonl_tail(path: Path, n: int = 30) -> list[dict]:
    try:
        if not path.exists():
            return []
        from collections import deque as _deque

        rows: _deque[dict] = _deque(maxlen=n)
        with open(path, "r", encoding="utf-8", errors="replace") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = _json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        return list(rows)
    except Exception:
        return []


def _first_existing(paths: list[Path]) -> Path | None:
    for candidate in paths:
        try:
            if candidate.exists():
                return candidate
        except Exception:
            continue
    return None


def _avg(values: list[float]) -> float | None:
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _normalize_walkforward_report(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw.get("aggregate"), dict) and isinstance(raw.get("folds"), list):
        return raw

    # Fallback for wf_verify_acc{1,2}_report.json
    folds_raw = raw.get("folds")
    if not isinstance(folds_raw, list):
        return raw

    params = raw.get("params", {}) if isinstance(raw.get("params"), dict) else {}
    norm_folds: list[dict[str, Any]] = []
    roc_vals: list[float] = []
    win_vals: list[float] = []
    trades_total = 0
    for f in folds_raw:
        if not isinstance(f, dict):
            continue
        tm = f.get("train_metrics", {}) if isinstance(f.get("train_metrics"), dict) else {}
        roc = _as_float(tm.get("roc_auc"))
        if roc is not None:
            roc_vals.append(roc)
        wr = _as_float(f.get("win_rate"))
        if wr is not None:
            win_vals.append(wr)
        trades_val = _as_int(f.get("trades")) or 0
        trades_total += max(trades_val, 0)
        max_dd_val = _as_float(f.get("max_drawdown_pct"))
        if max_dd_val is not None:
            max_dd_val = abs(max_dd_val)
        norm_folds.append(
            {
                "fold": f.get("fold"),
                "train_start": f.get("train_start"),
                "train_end": f.get("train_end"),
                "test_start": f.get("test_start"),
                "test_end": f.get("test_end"),
                "threshold": tm.get("selected_threshold") or params.get("min_confidence"),
                "precision": tm.get("precision") or raw.get("avg_precision"),
                "roc_auc": tm.get("roc_auc"),
                "n_signals": f.get("signals_filtered_out"),
                "concurrent_sim": {
                    "win_rate": f.get("win_rate"),
                    "profit_factor": f.get("profit_factor"),
                    "return_pct": f.get("return_pct"),
                    "max_drawdown_pct": max_dd_val,
                    "trades": f.get("trades"),
                },
            }
        )

    aggregate = {
        "avg_roc_auc": raw.get("avg_roc_auc") or _avg(roc_vals),
        "avg_precision": raw.get("avg_precision"),
        "signal_win_rate": _avg(win_vals),
        "total_signals": trades_total or None,
        "concurrent_sim": {
            "avg_win_rate": _avg(win_vals),
            "avg_profit_factor": raw.get("avg_profit_factor"),
            "avg_return_pct": raw.get("avg_return_pct"),
            "avg_max_drawdown_pct": abs(raw.get("avg_max_drawdown_pct")) if isinstance(raw.get("avg_max_drawdown_pct"), (int, float)) else raw.get("avg_max_drawdown_pct"),
        },
    }
    return {
        "aggregate": aggregate,
        "folds": norm_folds,
        "walk_forward": {
            "n_folds": len(norm_folds),
            "train_bars": None,
            "test_bars": None,
            "model": params.get("model") or "wf_verify_fallback",
        },
    }


def _derive_backtest_from_wf_verify(wf_verify_report: dict[str, Any]) -> dict[str, Any]:
    folds = wf_verify_report.get("folds")
    if not isinstance(folds, list) or not folds:
        return {}
    if not isinstance(folds[0], dict) or "net_profit" not in folds[0]:
        return {}

    trades = 0
    wins = 0
    losses = 0
    draws = 0
    gross_profit = 0.0
    gross_loss = 0.0
    net_profit = 0.0
    max_dd = 0.0
    avg_holds: list[float] = []
    friction_rr_vals: list[float] = []
    sharpes: list[float] = []
    best_trade: float | None = None
    worst_trade: float | None = None
    starts: list[str] = []
    ends: list[str] = []
    start_bal: float | None = None

    for f in folds:
        if not isinstance(f, dict):
            continue
        trades += _as_int(f.get("trades")) or 0
        wins += _as_int(f.get("wins")) or 0
        losses += _as_int(f.get("losses")) or 0
        draws += _as_int(f.get("draws")) or 0
        net_profit += _as_float(f.get("net_profit")) or 0.0
        gross_profit += _as_float(f.get("gross_profit")) or 0.0
        gross_loss += _as_float(f.get("gross_loss")) or 0.0
        fold_dd = _as_float(f.get("max_drawdown_pct"))
        if fold_dd is not None:
            max_dd = max(max_dd, abs(fold_dd))
        hold = _as_float(f.get("avg_holding_bars"))
        if hold is not None:
            avg_holds.append(hold)
        fr = _as_float(f.get("friction_rr"))
        if fr is not None:
            friction_rr_vals.append(fr)
        sh = _as_float(f.get("sharpe_like"))
        if sh is not None:
            sharpes.append(sh)
        bt = _as_float(f.get("best_trade"))
        wt = _as_float(f.get("worst_trade"))
        if bt is not None:
            best_trade = bt if best_trade is None else max(best_trade, bt)
        if wt is not None:
            worst_trade = wt if worst_trade is None else min(worst_trade, wt)
        if f.get("test_start"):
            starts.append(str(f.get("test_start")))
        if f.get("test_end"):
            ends.append(str(f.get("test_end")))
        if start_bal is None:
            start_bal = _as_float(f.get("starting_balance"))

    trade_start = min(starts) if starts else None
    trade_end = max(ends) if ends else None
    trade_days: int | None = None
    if trade_start and trade_end:
        try:
            start_dt = datetime.fromisoformat(str(trade_start).replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(str(trade_end).replace("Z", "+00:00"))
            trade_days = max((end_dt.date() - start_dt.date()).days + 1, 1)
        except Exception:
            trade_days = None

    starting_balance = start_bal or 200.0
    ending_balance = starting_balance + net_profit
    win_rate = (wins / trades) if trades > 0 else None
    pf = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None
    avg_win = (gross_profit / wins) if wins > 0 else None
    avg_loss = (gross_loss / losses) if losses > 0 else None
    return_pct = ((ending_balance - starting_balance) / starting_balance * 100.0) if starting_balance > 0 else None

    return {
        "trade_start": trade_start,
        "trade_end": trade_end,
        "trade_days": trade_days,
        "starting_balance": starting_balance,
        "ending_balance": ending_balance,
        "net_profit": net_profit,
        "return_pct": return_pct,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": pf,
        "max_drawdown_pct": max_dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "sharpe_like": _avg(sharpes),
        "avg_holding_bars": _avg(avg_holds),
        "friction_rr": _avg(friction_rr_vals),
        "source": "derived_from_wf_verify",
    }


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


def _parse_trade_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        elif ts < 1e9:
            return None
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.replace(".", "", 1).isdigit():
        try:
            return _parse_trade_timestamp(float(text))
        except Exception:
            return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _normalize_trade_side(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"buy", "long"}:
        return "buy"
    if raw in {"sell", "short"}:
        return "sell"
    return "unknown"


def _session_bucket(ts: datetime | None) -> str:
    if ts is None:
        return "Unknown"
    hour = int(ts.hour)
    if 0 <= hour < 7:
        return "Asian"
    if 7 <= hour < 13:
        return "London"
    if 13 <= hour < 22:
        return "New York"
    return "Off-hours"


def _regime_bucket(value: Any) -> str:
    regime = _as_int(value)
    if regime == 0:
        return "sideway"
    if regime == 1:
        return "normal"
    if regime == 2:
        return "strong"
    return "unknown"


def _summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    trades = len(items)
    net_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    for item in items:
        pnl = _as_float(item.get("pnl")) or 0.0
        net_pnl += pnl
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        elif pnl < 0:
            losses += 1
            gross_loss += pnl
    win_rate = (wins / trades) if trades > 0 else 0.0
    pf = (gross_profit / abs(gross_loss)) if gross_loss < 0 else (None if gross_profit == 0 else 999.0)
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "net_pnl": round(net_pnl, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": None if pf is None else round(pf, 4),
    }


def _build_pnl_explain(trades: list[dict], signals: list[dict]) -> dict[str, list[dict[str, Any]]]:
    if not trades:
        empty_row = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": None,
        }
        return {
            "session": [{"bucket": key, **empty_row} for key in ("Asian", "London", "New York")],
            "regime": [{"bucket": key, **empty_row} for key in ("sideway", "normal", "strong")],
            "side": [{"bucket": key, **empty_row} for key in ("buy", "sell")],
        }

    signal_index: list[dict[str, Any]] = []
    for sg in signals:
        sg_ts = _parse_trade_timestamp(sg.get("time") or sg.get("ts") or sg.get("bar_time"))
        if sg_ts is None:
            continue
        signal_index.append(
            {
                "ts": sg_ts,
                "side": _normalize_trade_side(sg.get("direction") or sg.get("signal") or sg.get("side")),
                "regime": _regime_bucket(sg.get("volatility_regime")),
            }
        )
    signal_index.sort(key=lambda item: item["ts"])

    max_link_age = timedelta(days=7)
    enriched: list[dict[str, Any]] = []
    for trade in trades:
        ts = _parse_trade_timestamp(trade.get("time") or trade.get("close_time") or trade.get("ts") or trade.get("open_time"))
        side = _normalize_trade_side(trade.get("side") or trade.get("direction"))
        pnl = _as_float(trade.get("pnl") or trade.get("profit")) or 0.0
        regime = "unknown"
        if ts is not None and side != "unknown":
            for sg in reversed(signal_index):
                if sg["side"] not in {side, "unknown"}:
                    continue
                if sg["ts"] > ts:
                    continue
                if (ts - sg["ts"]) > max_link_age:
                    break
                regime = str(sg["regime"] or "unknown")
                break
        enriched.append(
            {
                "ts": ts,
                "side": side,
                "session": _session_bucket(ts),
                "regime": regime,
                "pnl": pnl,
            }
        )

    def _build_dimension(
        items: list[dict[str, Any]],
        key: str,
        order: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for bucket in order:
            grouped = [item for item in items if item.get(key) == bucket]
            rows.append({"bucket": bucket, **_summarize_group(grouped)})
        return rows

    return {
        "session": _build_dimension(enriched, "session", ("Asian", "London", "New York")),
        "regime": _build_dimension(enriched, "regime", ("sideway", "normal", "strong")),
        "side": _build_dimension(enriched, "side", ("buy", "sell")),
    }


def _estimate_profile_presets(
    trades: list[dict],
    runtime_cfg: dict[str, Any],
    drawdown_pct: float | None,
) -> dict[str, dict[str, Any]]:
    parsed_times = [
        _parse_trade_timestamp(item.get("time") or item.get("close_time") or item.get("ts") or item.get("open_time"))
        for item in trades
    ]
    parsed_times = [ts for ts in parsed_times if ts is not None]
    if parsed_times:
        span_days = max((max(parsed_times) - min(parsed_times)).total_seconds() / 86400.0, 1.0)
        base_trades_per_day = len(parsed_times) / span_days
    else:
        threshold_hint = _as_float(runtime_cfg.get("threshold_effective_config")) or 0.7
        base_trades_per_day = max(0.8, (1.0 - threshold_hint) * 12.0)

    kill_pct = _as_float(runtime_cfg.get("max_drawdown_kill_pct"))
    if drawdown_pct is not None:
        base_dd = float(drawdown_pct)
    elif kill_pct is not None and kill_pct > 0:
        base_dd = float(kill_pct * 100.0 * 0.75)
    else:
        base_dd = 18.0
    base_dd = max(3.0, min(base_dd, 60.0))

    threshold_eff = _as_float(runtime_cfg.get("threshold_effective_config"))
    if threshold_eff is None:
        threshold_eff = max(
            _as_float(runtime_cfg.get("threshold_config")) or 0.0,
            _as_float(runtime_cfg.get("risk_min_confidence")) or 0.0,
        )
    if threshold_eff <= 0:
        threshold_eff = 0.7

    def _profile(mult_trade: float, mult_dd: float, thr_delta: float, label: str, desc: str) -> dict[str, Any]:
        return {
            "label": label,
            "description": desc,
            "estimated_trades_per_day": round(base_trades_per_day * mult_trade, 2),
            "estimated_dd_pct": round(max(1.0, min(80.0, base_dd * mult_dd)), 2),
            "effective_threshold_est": round(min(0.99, max(0.5, threshold_eff + thr_delta)), 4),
        }

    return {
        "conservative": _profile(
            mult_trade=0.65,
            mult_dd=0.72,
            thr_delta=0.05,
            label="Conservative",
            desc="Lower frequency, tighter gate, prioritize capital defense.",
        ),
        "balanced": _profile(
            mult_trade=1.00,
            mult_dd=1.00,
            thr_delta=0.00,
            label="Balanced",
            desc="Current baseline profile with balanced risk/reward.",
        ),
        "aggressive": _profile(
            mult_trade=1.35,
            mult_dd=1.28,
            thr_delta=-0.05,
            label="Aggressive",
            desc="Higher activity and return potential with larger DD tolerance.",
        ),
    }


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _max_drawdown_pct_from_pnls(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    ordered = sorted(items, key=lambda row: row.get("ts") or datetime.min.replace(tzinfo=timezone.utc))
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for row in ordered:
        pnl = _as_float(row.get("pnl")) or 0.0
        equity += pnl
        if equity > peak:
            peak = equity
        drawdown = max(0.0, peak - equity)
        if drawdown > max_dd:
            max_dd = drawdown
    if peak <= 0:
        return 0.0
    return round((max_dd / peak) * 100.0, 4)


def _build_regime_session_matrix(
    trades: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    windows_days: list[int] | None = None,
) -> dict[str, Any]:
    sessions = ("Asian", "London", "New York")
    regimes = ("sideway", "normal", "strong")
    windows = [int(w) for w in (windows_days or [7, 30]) if int(w) > 0]
    if not windows:
        windows = [7, 30]

    signal_rows: list[dict[str, Any]] = []
    for sg in signals:
        sg_ts = _parse_trade_timestamp(sg.get("time") or sg.get("ts") or sg.get("bar_time"))
        if sg_ts is None:
            continue
        signal_rows.append(
            {
                "ts": sg_ts,
                "side": _normalize_trade_side(sg.get("direction") or sg.get("signal") or sg.get("side")),
                "regime": _regime_bucket(sg.get("volatility_regime")),
                "session": _session_bucket(sg_ts),
                "should_trade": _is_truthy(sg.get("should_trade")),
            }
        )
    signal_rows.sort(key=lambda item: item["ts"])

    max_link_age = timedelta(days=7)
    trade_rows: list[dict[str, Any]] = []
    for tr in trades:
        tr_ts = _parse_trade_timestamp(tr.get("time") or tr.get("close_time") or tr.get("ts") or tr.get("open_time"))
        if tr_ts is None:
            continue
        tr_side = _normalize_trade_side(tr.get("side") or tr.get("direction"))
        tr_pnl = _as_float(tr.get("pnl") or tr.get("profit")) or 0.0
        regime = "unknown"
        if tr_side != "unknown":
            for sg in reversed(signal_rows):
                if sg["side"] not in {tr_side, "unknown"}:
                    continue
                if sg["ts"] > tr_ts:
                    continue
                if (tr_ts - sg["ts"]) > max_link_age:
                    break
                regime = str(sg.get("regime") or "unknown")
                break
        trade_rows.append(
            {
                "ts": tr_ts,
                "side": tr_side,
                "pnl": tr_pnl,
                "session": _session_bucket(tr_ts),
                "regime": regime,
            }
        )

    now_ref = datetime.now(timezone.utc)
    ts_candidates = [row["ts"] for row in trade_rows if row.get("ts")] + [row["ts"] for row in signal_rows if row.get("ts")]
    if ts_candidates:
        now_ref = max(ts_candidates)

    payload: dict[str, Any] = {"generated_at": now_ref.isoformat()}
    for window in windows:
        cutoff = now_ref - timedelta(days=int(window))
        signals_cut = [sg for sg in signal_rows if sg["ts"] >= cutoff and bool(sg.get("should_trade"))]
        trades_cut = [tr for tr in trade_rows if tr["ts"] >= cutoff]

        signal_counts: dict[tuple[str, str], int] = {}
        for sg in signals_cut:
            key = (str(sg.get("session") or "Unknown"), str(sg.get("regime") or "unknown"))
            signal_counts[key] = signal_counts.get(key, 0) + 1

        rows: list[dict[str, Any]] = []
        for session in sessions:
            for regime in regimes:
                grouped = [tr for tr in trades_cut if tr.get("session") == session and tr.get("regime") == regime]
                summary = _summarize_group(grouped)
                signal_count = signal_counts.get((session, regime), 0)
                recall = (summary["trades"] / signal_count) if signal_count > 0 else None
                rows.append(
                    {
                        "session": session,
                        "regime": regime,
                        "trades": summary["trades"],
                        "wins": summary["wins"],
                        "losses": summary["losses"],
                        "win_rate": summary["win_rate"],
                        "profit_factor": summary["profit_factor"],
                        "net_pnl": summary["net_pnl"],
                        "max_drawdown_pct": _max_drawdown_pct_from_pnls(grouped),
                        "signals_passed": signal_count,
                        "recall": None if recall is None else round(recall, 4),
                    }
                )

        leak_rows = sorted(
            [row for row in rows if (row.get("net_pnl") or 0.0) < 0 or ((row.get("profit_factor") or 0.0) < 1.0 and row.get("trades", 0) > 0)],
            key=lambda row: (row.get("net_pnl") or 0.0),
        )
        payload[f"rolling_{window}d"] = {
            "window_days": int(window),
            "rows": rows,
            "leakage_hotspots": leak_rows[:6],
            "total_trades": len(trades_cut),
            "total_signals_passed": len(signals_cut),
        }
    return payload


_dashboard_cache: dict = {}
_dashboard_cache_ts: float = 0.0
_dashboard_cache_lock = _threading.Lock()
_DASHBOARD_CACHE_TTL = 30.0  # seconds — longer than build time to avoid thrashing

# ── News constants & helpers ──────────────────────────────────────────────────

_FF_NEWS_WEEK_URL  = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_FF_NEWS_MONTH_URL = "https://nfs.faireconomy.media/ff_calendar_thismonth.json"
_NEWS_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; XAUBot/1.0)"}
_NEWS_CACHE_TTL = 300.0   # 5 min default — short enough to catch actuals, avoids FF 429
_NEWS_CACHE_TTL_POST_EVENT = 60.0  # reduce to 1 min for 10 min after an event fires

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


def _is_news_alert_impact(impact: str) -> bool:
    return str(impact or "").strip().capitalize() in {"High", "Medium"}


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
    # released = any non-empty actual string (not just numeric — e.g. "RBNZ Rate Statement" has text actual)
    released = bool(actual.strip())

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
_news_last_event_ts: float = 0.0  # monotonic time when last _past fired (shorter TTL window)


def _build_news_payload() -> dict[str, Any]:
    """Return news calendar with adaptive-TTL cache (shorter for 10 min after event fires)."""
    global _news_cache, _news_cache_ts
    now = time.monotonic()
    # Use short TTL for 10 min after any event just-fired (_past branch)
    effective_ttl = (
        _NEWS_CACHE_TTL_POST_EVENT if now - _news_last_event_ts < 600
        else _NEWS_CACHE_TTL
    )
    with _news_cache_lock:
        if _news_cache and now - _news_cache_ts < effective_ttl:
            return _news_cache
        result = _fetch_news_uncached()
        # Only replace cache if we got real data — don't overwrite with empty (FF rate-limit / timeout)
        if result.get("week"):
            _news_cache = result
            _news_cache_ts = now
        elif not _news_cache:
            _news_cache = result  # no prior cache at all — store even if empty
        return _news_cache


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
            if r.status_code == 429:
                logger.warning("ForexFactory rate-limited (429) — keeping stale cache")
                break  # don't try fallback URL, just keep existing cache
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
    benchmarks = _load_benchmark_targets()
    for acct, cfg in _ACCOUNT_CFG.items():
        runtime_cfg = _ACCOUNT_RUNTIME_CFG.get(acct, {})
        daily    = _safe_json(_output_path(cfg.get("daily", "risk_daily_state.json")))
        peak_bal = float(_safe_json(_output_path(cfg.get("peak", "risk_peak_balance.json"))).get("peak_balance") or 0)
        status = _safe_json(_output_path(cfg["status"]))
        meta   = _safe_json(_output_path(cfg["meta"]))
        wf_path_candidates = [
            _output_path(cfg["wf"]),
            _output_path(f"walkforward_report_{acct}.json"),
            _output_path(f"exp_{acct}_2003_walkforward_report.json"),
            _output_path(f"wf_verify_{acct}_report.json"),
        ]
        wf_path = _first_existing(wf_path_candidates) or _output_path(cfg["wf"])
        wf_raw = _safe_json(wf_path)
        wf = _normalize_walkforward_report(wf_raw)

        bt_path_candidates = [
            _output_path(cfg["bt"]),
            _output_path(f"backtest_report_{acct}.json"),
            _output_path(f"exp_{acct}_2003_backtest_report.json"),
        ]
        bt_path = _first_existing(bt_path_candidates) or _output_path(cfg["bt"])
        bt = _safe_json(bt_path)
        if not bt:
            bt = _derive_backtest_from_wf_verify(_safe_json(_output_path(f"wf_verify_{acct}_report.json")))
        if bt and "source_path" not in bt:
            bt["source_path"] = str(bt_path)

        trades_path = _output_path(cfg["trades"])
        if not trades_path.exists():
            trades_path = _output_path(cfg["trades_fallback"])
        trades = _safe_csv_tail(trades_path, 200, account=acct)
        journal = _safe_jsonl_tail(_output_path(cfg["journal"]), 200) if cfg.get("journal") else []
        signals = _safe_csv_tail(_output_path(cfg["signals"]), 600, account=acct)

        bt_trades: list[dict[str, Any]] = []
        if cfg.get("bt_trades"):
            bt_trades_candidates = [
                _output_path(cfg.get("bt_trades", "")),
                _output_path(f"backtest_trades_{acct}.csv"),
                _output_path(f"exp_{acct}_2003_backtest_trades.csv"),
                _output_path(f"wf_verify_{acct}_trades.csv"),
            ]
            bt_trades_path = _first_existing(bt_trades_candidates)
            if bt_trades_path is not None:
                bt_trades = _safe_csv_tail(bt_trades_path, 1000, account=acct)

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
        pnl_explain = _build_pnl_explain(trades, signals)
        profile_presets = _estimate_profile_presets(trades, runtime_cfg, dd)
        matrix_windows = runtime_cfg.get("admin_matrix_windows_days") if isinstance(runtime_cfg, dict) else [7, 30]
        matrix_payload = _build_regime_session_matrix(trades, signals, windows_days=matrix_windows)
        profile_override = _read_runtime_profile_override(acct)
        profile_active = str(profile_override.get("profile") or "balanced").strip().lower()
        if profile_active not in _PROFILE_MODES:
            profile_active = "balanced"

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
        model_threshold = meta.get("threshold") or meta.get("decision_threshold")
        config_threshold = runtime_cfg.get("threshold_config")
        risk_min_conf = runtime_cfg.get("risk_min_confidence")
        # threshold_effective_config = strategy.signal_threshold (set in _apply_live_config_overrides).
        # This is the ML signal gate: ACC1=0.62, ACC2=0.76 per LATEST_LIVE_GUIDE.md.
        threshold_eff_cfg = runtime_cfg.get("threshold_effective_config")
        threshold_candidates = [v for v in (model_threshold, config_threshold, risk_min_conf) if isinstance(v, (int, float))]
        threshold_effective = (
            float(threshold_eff_cfg)
            if isinstance(threshold_eff_cfg, (int, float))
            else (max(threshold_candidates) if threshold_candidates else None)
        )

        accounts[acct] = {
            "label":   cfg["label"],
            "status":  status,
            "model": {
                "precision":         meta.get("precision"),
                "threshold":         model_threshold,
                "threshold_model":   model_threshold,
                "threshold_config":  config_threshold,
                "threshold_risk":    risk_min_conf,
                "threshold_effective": threshold_effective,
                "roc_auc":           meta.get("roc_auc"),
                "recall":            meta.get("recall"),
                "f1":                meta.get("f1"),
                "accuracy":          meta.get("accuracy"),
                "train_rows":        meta.get("train_rows"),
                "test_rows":         meta.get("test_rows"),
                "wf_precision_avg":  wf_agg.get("avg_precision"),
                "wf_win_rate_avg":   wf_agg.get("signal_win_rate"),
                "wf_n_folds":        wf_cfg.get("n_folds") or len(wf.get("folds", [])),
                "binding_ok":        runtime_cfg.get("model_binding_ok", True),
                "binding_mismatch":  runtime_cfg.get("model_binding_mismatch", []),
                "binding_required":  runtime_cfg.get("model_binding_required", {}),
                "binding_actual":    runtime_cfg.get("model_binding_actual", {}),
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
            "recent_journal": journal[-50:],
            "recent_signals": signals,
            "pnl_series":     _build_pnl_series(trades),
            "pnl_explain":    pnl_explain,
            "regime_session_matrix": matrix_payload,
            "profile_presets": profile_presets,
            "profile_active": profile_active,
            "runtime": runtime_cfg,
            "benchmark": benchmarks.get(acct, {}),
            "auto_trade_enabled": runtime_cfg.get("auto_trade_enabled"),
        }

    return {
        "ts": datetime.utcnow().isoformat(),
        "benchmark_source": benchmarks.get("source"),
        "accounts": accounts,
    }


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


class ApplyProfileRequest(BaseModel):
    account: str
    profile: str
    source: str = "dashboard"
    requested_by: str | None = None


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


@app.post("/api/v1/profile/apply", tags=["dashboard"])
async def apply_runtime_profile(req: ApplyProfileRequest) -> dict[str, Any]:
    account = req.account.strip().lower()
    profile = req.profile.strip().lower()
    if account not in _ACCOUNT_CFG:
        raise HTTPException(status_code=404, detail=f"Unknown account: {account}")
    if profile not in _PROFILE_MODES:
        raise HTTPException(status_code=422, detail=f"Invalid profile: {profile}")

    now_utc = datetime.now(timezone.utc).isoformat()
    payload = {
        "account": account,
        "profile": profile,
        "source": (req.source or "dashboard").strip()[:64],
        "requested_by": (req.requested_by or "").strip()[:64] or None,
        "requested_at_utc": now_utc,
    }
    override_path = _runtime_profile_override_path(account)
    try:
        _safe_write_json(override_path, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot write runtime profile override: {exc}") from exc

    tg_message = (
        f"🧭 <b>Profile Apply Request</b>\n"
        f"Account: <b>{account.upper()}</b>\n"
        f"Profile: <b>{profile.capitalize()}</b>\n"
        f"Source: {payload['source']}\n"
        f"Time (UTC): {now_utc}\n"
        f"Bot will hot-reload this profile automatically."
    )
    telegram_sent = _tg_send_account(account, tg_message)

    # Bust cache so dashboard reflects the requested profile quickly.
    global _dashboard_cache_ts
    _dashboard_cache_ts = 0.0

    return {
        "status": "accepted",
        "account": account,
        "profile": profile,
        "override_path": str(override_path),
        "telegram_sent": telegram_sent,
        "requested_at_utc": now_utc,
    }


@app.post("/api/v1/reset-kill-switch/{acct_tag}", tags=["dashboard"])
async def reset_kill_switch(acct_tag: str) -> dict[str, Any]:
    """Manually reset the daily kill switch / circuit breaker for an account."""
    acct = acct_tag.strip().lower()
    if acct not in _ACCOUNT_CFG:
        raise HTTPException(status_code=404, detail=f"Unknown account: {acct}")
    cfg = _ACCOUNT_CFG[acct]
    state_path = _OUTPUTS / cfg.get("daily", f"risk_daily_state_{acct}.json")

    current_state: dict[str, Any] = {}
    try:
        if state_path.exists():
            current_state = _json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        pass

    current_state["killed"] = False
    current_state["consecutive_losses"] = 0
    current_state["cooldown_bars"] = 0
    current_state["daily_loss"] = 0.0
    current_state["reset_by"] = "dashboard_manual"
    now_utc = datetime.now(timezone.utc).isoformat()
    current_state["reset_at_utc"] = now_utc

    try:
        _safe_write_json(state_path, current_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot write kill switch state: {exc}") from exc

    global _dashboard_cache_ts
    _dashboard_cache_ts = 0.0

    tg_msg = (
        f"🔓 <b>Kill Switch Reset</b>\n"
        f"Account: <b>{acct.upper()}</b>\n"
        f"Action: Manual reset via dashboard\n"
        f"Time (UTC): {now_utc}\n"
        f"Consecutive losses → 0, Cooldown → 0"
    )
    telegram_sent = _tg_send_account(acct, tg_msg)

    return {
        "status": "reset",
        "account": acct,
        "telegram_sent": telegram_sent,
        "reset_at_utc": now_utc,
    }


@app.post("/api/v1/auto-trade/{acct_tag}", tags=["dashboard"])
async def set_auto_trade(acct_tag: str, enabled: bool = Query(...)) -> dict[str, Any]:
    """Toggle auto_trade on/off for an account by patching the live YAML config."""
    import re as _re
    acct = acct_tag.strip().lower()
    if acct not in _LIVE_CFG_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown account: {acct}")
    cfg_path = _LIVE_CFG_MAP[acct]
    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {cfg_path}")

    try:
        text = cfg_path.read_text(encoding="utf-8")
        new_val = "true" if enabled else "false"
        # Use multiline anchor so commented-out lines (starting with #) are skipped
        new_text, n = _re.subn(r'(?m)^([ \t]*auto_trade\s*:)\s*\S+', lambda m: m.group(1) + f" {new_val}", text)
        if n == 0:
            raise HTTPException(status_code=422, detail="auto_trade key not found in config YAML")
        cfg_path.write_text(new_text, encoding="utf-8")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot update config: {exc}") from exc

    global _dashboard_cache_ts
    _dashboard_cache_ts = 0.0
    _apply_live_config_overrides()  # Refresh runtime config to pick up YAML change

    now_utc = datetime.now(timezone.utc).isoformat()
    tg_msg = (
        f"{'🟢' if enabled else '🔴'} <b>Auto Trade {'Enabled' if enabled else 'Disabled'}</b>\n"
        f"Account: <b>{acct.upper()}</b>\n"
        f"Changed via: dashboard UI\n"
        f"Time (UTC): {now_utc}"
    )
    telegram_sent = _tg_send_account(acct, tg_msg)

    return {
        "status": "updated",
        "account": acct,
        "auto_trade": enabled,
        "telegram_sent": telegram_sent,
        "updated_at_utc": now_utc,
    }


@app.get("/api/v1/tunnel-url", tags=["dashboard"])
async def get_tunnel_url() -> dict[str, Any]:
    """Return the current Cloudflare tunnel URL if available."""
    import re as _re
    candidates = [
        _OUTPUTS / "tunnel_err.txt",
        _OUTPUTS / "cloudflared.log",
        _OUTPUTS / "tunnel.log",
    ]
    pattern = _re.compile(r'https://[a-z0-9\-]+\.trycloudflare\.com')
    for f in candidates:
        try:
            if f.exists():
                text = f.read_text(encoding="utf-8", errors="ignore")
                m = pattern.search(text)
                if m:
                    return {"url": m.group(0), "source": f.name}
        except Exception:
            continue
    return {"url": None, "source": None}


class SendTelegramRequest(BaseModel):
    text: str
    account: str | None = None


@app.post("/api/v1/send-telegram", tags=["dashboard"])
async def send_telegram(req: SendTelegramRequest) -> dict[str, Any]:
    """Send a Telegram message to one or all configured accounts."""
    text = req.text.strip()[:4000]
    if not text:
        raise HTTPException(status_code=422, detail="text is empty")
    if req.account:
        acct = req.account.strip().lower()
        if acct not in _LIVE_CFG_MAP:
            raise HTTPException(status_code=404, detail=f"Unknown account: {acct}")
        sent = _tg_send_account(acct, text)
        return {"sent": sent, "accounts": [acct]}
    else:
        sent_list = []
        for acct in _LIVE_CFG_MAP:
            if _tg_send_account(acct, text):
                sent_list.append(acct)
        return {"sent": bool(sent_list), "accounts": sent_list}


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
    """Send Telegram news alert to ALL configured accounts."""
    import requests as _req
    # Collect (token, chat_id) pairs for every account in _LIVE_CFG_MAP
    sent_tokens: set[str] = set()
    for acct, cfg_path in _LIVE_CFG_MAP.items():
        try:
            s = load_settings(cfg_path)
            token   = os.getenv(s.integrations.telegram.token_env, "")
            chat_id = os.getenv(s.integrations.telegram.chat_id_env, "")
        except Exception:
            continue
        if not token or not chat_id:
            continue
        # Avoid duplicate sends when two accounts share the same bot token
        if token in sent_tokens:
            continue
        sent_tokens.add(token)
        try:
            resp = _req.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            if not resp.ok:
                _req.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": text},
                    timeout=10,
                )
        except Exception as exc:
            logger.debug("Telegram news alert send error (%s): %s", acct, exc)


def _tg_send_account(account: str, text: str) -> bool:
    """Send Telegram message for a specific account using that account's config env mapping."""
    import requests as _req

    acct = account.strip().lower()
    cfg_path = _LIVE_CFG_MAP.get(acct)
    if cfg_path is None:
        return False
    try:
        settings = load_settings(cfg_path)
    except Exception as exc:
        logger.warning("Could not load settings for Telegram account send (%s): %s", acct, exc)
        return False

    token = os.getenv(settings.integrations.telegram.token_env, "")
    chat_id = os.getenv(settings.integrations.telegram.chat_id_env, "")
    if not token or not chat_id:
        return False

    try:
        resp = _req.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if resp.ok:
            return True
        _req.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        return True
    except Exception as exc:
        logger.debug("Telegram account alert send error (%s): %s", acct, exc)
        return False


async def _news_alert_loop() -> None:
    """Background task: check every 60s and alert upcoming / just-released news."""
    global _news_last_event_ts
    await asyncio.sleep(30)   # short delay after startup
    while True:
        try:
            loop = asyncio.get_running_loop()
            payload = await loop.run_in_executor(None, _build_news_payload)
            for ev in payload.get("week", []):
                if not _is_news_alert_impact(str(ev.get("impact", ""))):
                    continue
                eid   = ev["id"]
                mins  = ev["minutes_until"]
                rel   = ev["released"]

                # ── Released with actual data ────────────────────────────────
                if rel:
                    key = f"{eid}_released"
                    if key not in _news_alerted:
                        _news_alerted.add(key)
                        _news_alerted.discard(f"{eid}_past")  # drop preliminary key
                        actual   = ev.get("actual")   or "—"
                        forecast = ev.get("forecast") or "—"
                        previous = ev.get("previous") or "—"
                        label    = ev.get("gold_label", "")
                        icon     = "🔴" if ev["impact"] == "High" else "🟡"
                        msg = (
                            f"📊 {icon} <b>{ev['event']}</b> — Đã phát hành!\n"
                            f"📰 ({ev['currency']}) 🕐 {ev['time_utc']} UTC\n"
                            f"⬅️ Trước: <b>{previous}</b>  🎯 Dự báo: <b>{forecast}</b>  ✅ Thực tế: <b>{actual}</b>\n"
                            f"🏅 Vàng: {label}"
                        )
                        await loop.run_in_executor(None, _tg_news_send, msg)

                # ── Just passed — fire immediately even without actual ────────
                elif -120 < mins <= 0:
                    key = f"{eid}_past"
                    if key not in _news_alerted:
                        _news_alerted.add(key)
                        _news_last_event_ts = time.monotonic()  # shorten cache TTL for next 10 min
                        forecast = ev.get("forecast") or "—"
                        previous = ev.get("previous") or "—"
                        label    = ev.get("gold_label", "")
                        icon     = "🔴" if ev["impact"] == "High" else "🟡"
                        msg = (
                            f"{icon} <b>{ev['event']}</b> — Vừa phát hành!\n"
                            f"📰 ({ev['currency']}) 🕐 {ev['time_utc']} UTC\n"
                            f"🎯 Dự báo: <b>{forecast}</b>  ⬅️ Trước: <b>{previous}</b>\n"
                            f"⏳ Đang chờ số liệu thực tế từ ForexFactory...\n"
                            f"💡 {label}"
                        )
                        await loop.run_in_executor(None, _tg_news_send, msg)

                # ── Upcoming — T-30, T-20, T-10, T-5 ────────────────────────
                elif 0 < mins <= 32:
                    for milestone, milestone_lbl in [(30, "30 phút"), (20, "20 phút"), (10, "10 phút"), (5, "5 phút")]:
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
                                    f"🎯 Dự báo: <b>{forecast}</b>  ⬅️ Trước: <b>{previous}</b>\n"
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
