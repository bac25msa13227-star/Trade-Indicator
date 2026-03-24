"""MinIO object storage client.

Replaces local file system storage for:
  - Model artifacts: model.pkl, scaler.pkl, model.pkl.cal.pkl
  - Training datasets: training_dataset.csv (as parquet)
  - Reports: backtest_report.json, walkforward_report.json
  - Evidently HTML reports

Buckets:
  models/     – model artifacts, version-namespaced
  datasets/   – training/backtest datasets as parquet
  reports/    – JSON / HTML reports
  mlflow/     – MLflow artifact store (configured separately in MLflow)

Usage:
    from xauusd_ai.infra.storage import StorageClient
    client = StorageClient()
    client.upload_model("acc2", "v7", Path("outputs/model.pkl"))
    path = client.download_model("acc2", "v7", dest_dir=Path("/tmp"))
"""
from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from minio import Minio
    from minio.error import S3Error
    _MINIO_AVAILABLE = True
except ImportError:
    _MINIO_AVAILABLE = False
    logger.warning("minio package not installed — StorageClient will operate in local-only mode.")


_BUCKET_MODELS = "models"
_BUCKET_DATASETS = "datasets"
_BUCKET_REPORTS = "reports"
_BUCKET_MLFLOW = "mlflow"

_ALL_BUCKETS = [_BUCKET_MODELS, _BUCKET_DATASETS, _BUCKET_REPORTS, _BUCKET_MLFLOW]


class StorageClient:
    """Thin wrapper around MinIO for trading model artifact management."""

    def __init__(self) -> None:
        endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

        if not _MINIO_AVAILABLE:
            self._client = None
            logger.warning("StorageClient: minio not available, all ops are no-ops.")
            return

        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self._ensure_buckets()

    def _ensure_buckets(self) -> None:
        if self._client is None:
            return
        for bucket in _ALL_BUCKETS:
            try:
                if not self._client.bucket_exists(bucket):
                    self._client.make_bucket(bucket)
                    logger.info("Created MinIO bucket: %s", bucket)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not verify/create bucket %s: %s", bucket, exc)

    # ── Model artifacts ───────────────────────────────────────────────────────

    def upload_model(self, account: str, version: str, local_path: Path) -> str:
        """Upload a model artifact. Returns the object path in MinIO."""
        object_name = f"{account}/{version}/{local_path.name}"
        return self._upload_file(_BUCKET_MODELS, object_name, local_path)

    def download_model(self, account: str, version: str, filename: str, dest_dir: Path) -> Path:
        """Download a model artifact to dest_dir. Returns local Path."""
        object_name = f"{account}/{version}/{filename}"
        dest_path = dest_dir / filename
        self._download_file(_BUCKET_MODELS, object_name, dest_path)
        return dest_path

    def list_model_versions(self, account: str) -> list[str]:
        """List all version tags for a given account."""
        if self._client is None:
            return []
        prefix = f"{account}/"
        objects = self._client.list_objects(_BUCKET_MODELS, prefix=prefix, recursive=False)
        versions: set[str] = set()
        for obj in objects:
            # obj.object_name looks like "acc2/v7/model.pkl"
            parts = obj.object_name.split("/")
            if len(parts) >= 2:
                versions.add(parts[1])
        return sorted(versions)

    # ── Datasets ─────────────────────────────────────────────────────────────

    def upload_dataset(self, account: str, name: str, local_path: Path) -> str:
        object_name = f"{account}/{name}"
        return self._upload_file(_BUCKET_DATASETS, object_name, local_path)

    def download_dataset(self, account: str, name: str, dest_path: Path) -> Path:
        self._download_file(_BUCKET_DATASETS, f"{account}/{name}", dest_path)
        return dest_path

    # ── Reports ───────────────────────────────────────────────────────────────

    def upload_report(self, account: str, report_name: str, data: dict[str, Any]) -> str:
        """Upload a JSON report dict. Returns object path."""
        object_name = f"{account}/{report_name}"
        content = json.dumps(data, indent=2, default=str).encode()
        return self._upload_bytes(_BUCKET_REPORTS, object_name, content, "application/json")

    def upload_html_report(self, account: str, report_name: str, local_path: Path) -> str:
        object_name = f"{account}/{report_name}"
        return self._upload_file(_BUCKET_REPORTS, object_name, local_path, "text/html")

    def get_report_url(self, account: str, report_name: str, expires_seconds: int = 3600) -> str:
        """Return a presigned download URL for a report."""
        if self._client is None:
            return ""
        from datetime import timedelta
        try:
            return self._client.presigned_get_object(
                _BUCKET_REPORTS,
                f"{account}/{report_name}",
                expires=timedelta(seconds=expires_seconds),
            )
        except S3Error as exc:
            logger.error("presigned URL error: %s", exc)
            return ""

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _upload_file(
        self, bucket: str, object_name: str, local_path: Path, content_type: str = "application/octet-stream"
    ) -> str:
        if self._client is None:
            logger.debug("StorageClient no-op upload: %s", local_path)
            return object_name
        try:
            self._client.fput_object(bucket, object_name, str(local_path), content_type=content_type)
            logger.info("Uploaded %s → minio://%s/%s", local_path, bucket, object_name)
        except S3Error as exc:
            logger.error("MinIO upload failed %s: %s", object_name, exc)
        return object_name

    def _download_file(self, bucket: str, object_name: str, dest_path: Path) -> None:
        if self._client is None:
            return
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.fget_object(bucket, object_name, str(dest_path))
            logger.info("Downloaded minio://%s/%s → %s", bucket, object_name, dest_path)
        except S3Error as exc:
            logger.error("MinIO download failed %s: %s", object_name, exc)

    def _upload_bytes(
        self, bucket: str, object_name: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        if self._client is None:
            return object_name
        try:
            self._client.put_object(
                bucket, object_name, io.BytesIO(data), length=len(data), content_type=content_type
            )
            logger.info("Uploaded bytes → minio://%s/%s (%d bytes)", bucket, object_name, len(data))
        except S3Error as exc:
            logger.error("MinIO bytes upload failed %s: %s", object_name, exc)
        return object_name
