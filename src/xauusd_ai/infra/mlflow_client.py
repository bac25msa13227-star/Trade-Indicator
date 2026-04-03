"""MLflow experiment tracking integration.

Replaces manual JSON model metadata with MLflow experiments + Model Registry.
MinIO is used as the artifact store backend (configured via env vars).

Usage:
    from xauusd_ai.infra.mlflow_client import MLflowTracker
    tracker = MLflowTracker(experiment_name="xauusd_training")
    with tracker.start_run(run_name="v7_acc2") as run_id:
        tracker.log_params({"max_iter": 300, "learning_rate": 0.05})
        tracker.log_metrics({"roc_auc": 0.72, "precision": 0.64})
        tracker.log_artifact(Path("outputs/model.pkl"))
        tracker.register_model(run_id, "xauusd-acc2", stage="Production")
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.warning("mlflow package not installed — MLflowTracker will be a no-op.")


def _get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


class MLflowTracker:
    """Wrapper around MLflow tracking API for the trading project."""

    def __init__(self, experiment_name: str = "xauusd_trading") -> None:
        self.experiment_name = experiment_name
        self._active_run_id: str | None = None

        if not _MLFLOW_AVAILABLE:
            return

        mlflow.set_tracking_uri(_get_tracking_uri())
        # Configure MinIO as artifact store via env (MLFLOW_S3_ENDPOINT_URL etc.)
        _configure_s3_artifact_store()

        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator[str, None, None]:
        """Context manager that starts an MLflow run and yields the run_id."""
        if not _MLFLOW_AVAILABLE:
            yield "no-mlflow"
            return

        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            self._active_run_id = run.info.run_id
            logger.info("MLflow run started: %s (id=%s)", run_name, self._active_run_id)
            try:
                yield self._active_run_id
            finally:
                self._active_run_id = None

    def log_params(self, params: dict[str, Any]) -> None:
        if not _MLFLOW_AVAILABLE:
            return
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not _MLFLOW_AVAILABLE:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: Path, artifact_path: str | None = None) -> None:
        if not _MLFLOW_AVAILABLE:
            return
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_artifacts_dir(self, local_dir: Path, artifact_path: str | None = None) -> None:
        if not _MLFLOW_AVAILABLE:
            return
        mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)

    def log_sklearn_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_name: str | None = None,
    ) -> str:
        """Log a scikit-learn model and optionally register it. Returns model URI."""
        if not _MLFLOW_AVAILABLE:
            return ""
        result = mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_name,
        )
        return result.model_uri

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        stage: str = "Staging",
        description: str = "",
    ) -> None:
        """Register a model in the MLflow Model Registry and transition its stage."""
        if not _MLFLOW_AVAILABLE:
            return
        client = MlflowClient()
        model_uri = f"runs:/{run_id}/{artifact_path}"
        try:
            mv = mlflow.register_model(model_uri, model_name)
            client.update_registered_model(model_name, description=description)
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage=stage,
                archive_existing_versions=(stage == "Production"),
            )
            logger.info("Registered model %s v%s → %s", model_name, mv.version, stage)
        except Exception as exc:  # noqa: BLE001
            logger.error("MLflow model registration failed: %s", exc)

    def get_production_run_id(self, model_name: str) -> str | None:
        """Return the run_id of the current Production model version."""
        if not _MLFLOW_AVAILABLE:
            return None
        client = MlflowClient()
        try:
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                return versions[0].run_id
        except Exception as exc:  # noqa: BLE001
            logger.error("MLflow get_production_run_id failed: %s", exc)
        return None

    def get_latest_metrics(self, model_name: str, stage: str = "Production") -> dict[str, float]:
        """Fetch metrics logged during the latest run for the given model stage."""
        if not _MLFLOW_AVAILABLE:
            return {}
        run_id = self.get_production_run_id(model_name)
        if not run_id:
            return {}
        client = MlflowClient()
        run = client.get_run(run_id)
        return dict(run.data.metrics)

    def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """Return recent runs as list of dicts with run_id, metrics, params, tags."""
        if not _MLFLOW_AVAILABLE:
            return []
        runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            filter_string=filter_string,
            max_results=max_results,
            output_format="list",
        )
        results = []
        for run in runs:
            results.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
                "tags": dict(run.data.tags),
            })
        return results


# ── S3/MinIO artifact store configuration ────────────────────────────────────

def _configure_s3_artifact_store() -> None:
    """Set boto3 env vars so MLflow can write artifacts to MinIO."""
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    scheme = "https" if secure else "http"

    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", f"{scheme}://{minio_endpoint}")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.getenv("MINIO_SECRET_KEY", "minioadmin"))
