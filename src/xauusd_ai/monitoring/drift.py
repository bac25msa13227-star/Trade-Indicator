"""Evidently ML monitoring — data drift & model performance reports.

Integrates with:
  - PostgreSQL: pulls reference/current feature data
  - MinIO: uploads generated HTML/JSON reports
  - Prometheus: pushes drift scores as gauges
  - MLflow: links drift reports to model runs

Usage:
    from xauusd_ai.monitoring.drift import DriftMonitor
    monitor = DriftMonitor(account="acc2")
    monitor.run_feature_drift(reference_df, current_df)
    monitor.run_model_performance(reference_df, current_df, y_true, y_pred)
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import (
        DataDriftPreset,
        ClassificationPreset,
        DataQualityPreset,
    )
    from evidently.metrics import (
        DatasetDriftMetric,
        DatasetMissingValuesSummaryMetric,
        ColumnDriftMetric,
    )
    _EVIDENTLY_AVAILABLE = True
except ImportError:
    _EVIDENTLY_AVAILABLE = False
    logger.warning("evidently not installed — DriftMonitor will generate no reports.")


# Feature columns to monitor (subset of trading features)
_MONITORED_FEATURES = [
    "rsi",
    "macd",
    "bb_position",
    "atr_norm",
    "volume_ratio",
    "strategy_score",
    "confidence",
    "volatility_regime",
    "ict_score",
    "wyckoff_score",
]


class DriftMonitor:
    """Generates Evidently drift/performance reports and pushes them to MinIO."""

    def __init__(
        self,
        account: str,
        storage=None,
        metrics_recorder=None,
    ) -> None:
        self.account = account
        self._storage = storage  # StorageClient or None
        self._metrics = metrics_recorder  # TradingMetrics or None

    # ── Feature Drift ─────────────────────────────────────────────────────────

    def run_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        column_mapping: Any | None = None,
    ) -> dict[str, Any]:
        """Compare current feature distribution to reference (training data)."""
        if not _EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available; skipping drift check.")
            return {"drift_detected": False, "drift_score": 0.0}

        # Use only feature columns present in both DataFrames
        available = [c for c in _MONITORED_FEATURES if c in reference_df.columns and c in current_df.columns]
        if not available:
            logger.warning("No monitored features found in DataFrames; skipping drift.")
            return {"drift_detected": False, "drift_score": 0.0}

        report = Report(metrics=[
            DatasetDriftMetric(),
            DataQualityPreset(),
            *[ColumnDriftMetric(column_name=col) for col in available[:10]],
        ])
        report.run(
            reference_data=reference_df[available],
            current_data=current_df[available],
            column_mapping=column_mapping,
        )

        result = report.as_dict()
        drift_score = self._extract_drift_score(result)
        drift_detected = drift_score > 0.3  # Alert threshold

        # Push to Prometheus
        if self._metrics:
            self._metrics.record_drift("features", drift_score)

        # Upload HTML report to MinIO
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        report_name = f"drift_features_{timestamp}.html"
        self._save_and_upload(report, report_name)

        logger.info(
            "Feature drift report: score=%.3f drift_detected=%s account=%s",
            drift_score, drift_detected, self.account,
        )
        return {
            "drift_detected": drift_detected,
            "drift_score": round(drift_score, 4),
            "report_name": report_name,
            "features_drifted": self._extract_drifted_columns(result),
        }

    # ── Model Performance ─────────────────────────────────────────────────────

    def run_model_performance(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_col: str = "target",
        prediction_col: str = "prediction",
        prediction_proba_col: str = "confidence",
    ) -> dict[str, Any]:
        """Compare model accuracy/precision/recall on current vs reference period."""
        if not _EVIDENTLY_AVAILABLE:
            return {}

        column_mapping = ColumnMapping(
            target=target_col,
            prediction=prediction_col,
            pos_label=1,
        )
        if prediction_proba_col in reference_df.columns:
            column_mapping.prediction = prediction_proba_col

        report = Report(metrics=[ClassificationPreset()])
        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping,
        )

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        report_name = f"model_perf_{timestamp}.html"
        self._save_and_upload(report, report_name)

        result = report.as_dict()
        return {"report_name": report_name, "metrics": self._extract_classification_metrics(result)}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save_and_upload(self, report: Any, report_name: str) -> None:
        """Save report to a temp file and upload to MinIO."""
        if self._storage is None:
            return
        try:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            report.save_html(str(tmp_path))
            self._storage.upload_html_report(self.account, report_name, tmp_path)
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.error("Failed to upload Evidently report: %s", exc)

    @staticmethod
    def _extract_drift_score(result: dict) -> float:
        """Extract share_of_drifted_columns from Evidently result dict."""
        try:
            for metric in result.get("metrics", []):
                if metric.get("metric") == "DatasetDriftMetric":
                    return float(metric["result"].get("share_of_drifted_columns", 0.0))
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _extract_drifted_columns(result: dict) -> list[str]:
        """Return list of column names that are drifting."""
        drifted = []
        try:
            for metric in result.get("metrics", []):
                if metric.get("metric") == "ColumnDriftMetric":
                    if metric["result"].get("drift_detected"):
                        drifted.append(metric["result"].get("column_name", ""))
        except Exception:
            pass
        return drifted

    @staticmethod
    def _extract_classification_metrics(result: dict) -> dict[str, float]:
        metrics: dict[str, float] = {}
        try:
            for metric in result.get("metrics", []):
                if "ClassificationQualityMetric" in metric.get("metric", ""):
                    current = metric.get("result", {}).get("current", {})
                    metrics.update({
                        "accuracy": current.get("accuracy", 0.0),
                        "precision": current.get("precision", 0.0),
                        "recall": current.get("recall", 0.0),
                        "f1": current.get("f1", 0.0),
                        "roc_auc": current.get("roc_auc", 0.0),
                    })
        except Exception:
            pass
        return metrics
