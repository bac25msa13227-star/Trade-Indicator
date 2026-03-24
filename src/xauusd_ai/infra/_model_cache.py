"""Lazy model cache for the FastAPI inference endpoint.

Loads model artifacts from local disk (or downloads from MinIO if not present).
Thread-safe singleton per account.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_predictors: dict[str, "_Predictor"] = {}


def get_predictor(account: str) -> "_Predictor":
    with _lock:
        if account not in _predictors:
            _predictors[account] = _Predictor(account)
        return _predictors[account]


def invalidate(account: str) -> None:
    """Force reload on next inference call (called after retraining)."""
    with _lock:
        _predictors.pop(account, None)


class _Predictor:
    """Wraps ModelTrainer.score_live_row for single-dict inference."""

    def __init__(self, account: str) -> None:
        self.account = account
        self._trainer = self._load(account)

    def _load(self, account: str):
        try:
            import os
            # Determine model path from environment or default
            suffix = f"_{account}" if account not in ("default", "") else ""
            model_path = os.getenv(
                f"MODEL_PATH_{account.upper()}",
                f"outputs/model{suffix}.pkl",
            )
            meta_path = model_path.replace(".pkl", "_meta.json").replace("model", "model_meta")
            scaler_path = model_path.replace("model", "scaler")

            from xauusd_ai.config import Settings
            import yaml

            config_path = os.getenv("SETTINGS_PATH", "configs/settings.yaml")
            if not Path(config_path).exists():
                config_path = "configs/settings.example.yaml"

            with open(config_path) as f:
                raw = yaml.safe_load(f)
            settings = Settings(**raw)

            from xauusd_ai.model.trainer import ModelTrainer
            trainer = ModelTrainer(settings)
            trainer.load_artifacts()
            return trainer
        except Exception as exc:
            logger.error("Failed to load predictor for account %s: %s", account, exc)
            raise

    def score(self, features: dict[str, Any]) -> tuple[float, bool, str]:
        """Returns (confidence, should_trade, side)."""
        import pandas as pd
        row = pd.Series(features)
        confidence, should_trade, side = self._trainer.score_live_row(row)
        return float(confidence), bool(should_trade), str(side)
