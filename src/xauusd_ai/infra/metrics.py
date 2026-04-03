"""Prometheus metrics registry for the trading system.

Exposes custom business metrics alongside standard process/HTTP metrics.

Metric categories:
  - Trading: open_positions, pnl_total, win_rate, drawdown
  - Model: prediction_confidence, model_version, drift_score
  - System: retrain_count, signal_count, error_count

Usage (in orchestrator / FastAPI):
    from xauusd_ai.infra.metrics import TradingMetrics
    metrics = TradingMetrics(account="acc2")
    metrics.record_signal(confidence=0.72, should_trade=True, side="buy")
    metrics.record_trade(pnl=45.0, is_win=True)
    metrics.update_account(balance=10500.0, open_positions=1, drawdown_pct=0.02)

FastAPI integration (in api/main.py):
    from prometheus_client import make_asgi_app
    app.mount("/metrics", make_asgi_app())
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        CollectorRegistry,
        REGISTRY,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — metrics will be no-ops.")


# ── Global metric definitions (registered once) ───────────────────────────────

if _PROMETHEUS_AVAILABLE:

    # Trading metrics
    TRADE_COUNTER = Counter(
        "trading_trades_total",
        "Total number of closed trades",
        ["account", "side", "result"],          # result: win / loss
    )
    SIGNAL_COUNTER = Counter(
        "trading_signals_total",
        "Total number of signals evaluated",
        ["account", "side", "decision"],         # decision: trade / skip
    )
    PNL_GAUGE = Gauge(
        "trading_pnl_total",
        "Cumulative realized PnL in account currency",
        ["account"],
    )
    WIN_RATE_GAUGE = Gauge(
        "trading_win_rate",
        "Rolling win rate (last 50 trades)",
        ["account"],
    )
    DRAWDOWN_GAUGE = Gauge(
        "trading_drawdown_pct",
        "Current drawdown from peak balance as percentage",
        ["account"],
    )
    BALANCE_GAUGE = Gauge(
        "trading_account_balance",
        "Current account balance",
        ["account"],
    )
    OPEN_POSITIONS_GAUGE = Gauge(
        "trading_open_positions",
        "Number of currently open positions",
        ["account"],
    )

    # Model metrics
    CONFIDENCE_HISTOGRAM = Histogram(
        "model_prediction_confidence",
        "Distribution of model prediction confidence scores",
        ["account"],
        buckets=[0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0],
    )
    MODEL_VERSION_GAUGE = Gauge(
        "model_version_info",
        "Current active model version (as label)",
        ["account", "version", "roc_auc"],
    )
    DRIFT_SCORE_GAUGE = Gauge(
        "model_drift_score",
        "Evidently data drift score (0=no drift, 1=full drift)",
        ["account", "feature_set"],
    )
    RETRAIN_COUNTER = Counter(
        "model_retrain_total",
        "Number of model retrains triggered",
        ["account", "trigger"],                  # trigger: scheduled / loss_threshold / manual
    )

    # Latency
    SIGNAL_LATENCY = Histogram(
        "trading_signal_latency_seconds",
        "Time to evaluate a signal (feature compute + model inference)",
        ["account"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )
    MT5_REQUEST_LATENCY = Histogram(
        "mt5_request_latency_seconds",
        "MT5 API request latency",
        ["operation"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 3.0],
    )

    # Error tracking
    ERROR_COUNTER = Counter(
        "trading_errors_total",
        "Total number of errors by component",
        ["account", "component"],
    )


class TradingMetrics:
    """Per-account metrics recorder. Safe to call when prometheus is unavailable."""

    def __init__(self, account: str) -> None:
        self.account = account
        self._enabled = _PROMETHEUS_AVAILABLE

    def record_signal(
        self,
        confidence: float,
        should_trade: bool,
        side: str = "none",
    ) -> None:
        if not self._enabled:
            return
        decision = "trade" if should_trade else "skip"
        SIGNAL_COUNTER.labels(account=self.account, side=side, decision=decision).inc()
        CONFIDENCE_HISTOGRAM.labels(account=self.account).observe(confidence)

    def record_trade(self, pnl: float, is_win: bool, side: str = "none") -> None:
        if not self._enabled:
            return
        result = "win" if is_win else "loss"
        TRADE_COUNTER.labels(account=self.account, side=side, result=result).inc()
        # Increment cumulative PnL
        PNL_GAUGE.labels(account=self.account).inc(pnl)

    def update_account(
        self,
        balance: float,
        open_positions: int,
        drawdown_pct: float = 0.0,
        win_rate: float | None = None,
    ) -> None:
        if not self._enabled:
            return
        BALANCE_GAUGE.labels(account=self.account).set(balance)
        OPEN_POSITIONS_GAUGE.labels(account=self.account).set(open_positions)
        DRAWDOWN_GAUGE.labels(account=self.account).set(drawdown_pct)
        if win_rate is not None:
            WIN_RATE_GAUGE.labels(account=self.account).set(win_rate)

    def record_model_version(self, version: str, roc_auc: float) -> None:
        if not self._enabled:
            return
        # Reset any previous label set for this account
        try:
            MODEL_VERSION_GAUGE.labels(
                account=self.account, version=version, roc_auc=f"{roc_auc:.4f}"
            ).set(1)
        except Exception:  # noqa: BLE001
            pass

    def record_drift(self, feature_set: str, drift_score: float) -> None:
        if not self._enabled:
            return
        DRIFT_SCORE_GAUGE.labels(account=self.account, feature_set=feature_set).set(drift_score)

    def record_retrain(self, trigger: str = "scheduled") -> None:
        if not self._enabled:
            return
        RETRAIN_COUNTER.labels(account=self.account, trigger=trigger).inc()

    def record_error(self, component: str) -> None:
        if not self._enabled:
            return
        ERROR_COUNTER.labels(account=self.account, component=component).inc()

    def signal_latency_timer(self):
        """Use as context manager: `with metrics.signal_latency_timer(): ...`"""
        if not self._enabled:
            return _NoOpTimer()
        return SIGNAL_LATENCY.labels(account=self.account).time()

    def mt5_latency_timer(self, operation: str):
        if not self._enabled:
            return _NoOpTimer()
        return MT5_REQUEST_LATENCY.labels(operation=operation).time()


class _NoOpTimer:
    """Dummy context manager when prometheus is not available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
