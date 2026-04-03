"""
Exit Model — learns WHEN to close a position early.

Architecture: HistGradientBoostingClassifier trained on position-lifecycle
rows. Each row represents one bar of a live position with:
  - 38 market context features (FEATURE_COLUMNS)
  - 6 position-state features (xm_*)

Output: P(should_exit | current market + position state)

Usage:
  # Training (see scripts/train_exit_model.py)
  model = ExitModel(settings)
  model.train(exit_dataset)   # exit_dataset from build_exit_dataset()
  model.save()

  # Live prediction
  model = ExitModel(settings)
  model.load()
  prob = model.predict(live_feature_row, pos_state_dict)
  if prob > settings.execution.exit_model.exit_threshold:
      executor.close_position(ticket, volume)
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from xauusd_ai.config import Settings
from xauusd_ai.features.exit_dataset import EXIT_FEATURE_COLUMNS, EXIT_EXTRA_FEATURES
from xauusd_ai.features.dataset import FEATURE_COLUMNS

LOGGER = logging.getLogger(__name__)


class ExitModel:
    """Supervised exit classifier: predicts whether to close a position now."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        exit_cfg = settings.execution.exit_model
        self.model = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            max_depth=5,
            min_samples_leaf=20,
            l2_regularization=0.3,
            class_weight="balanced",
            early_stopping=False,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.threshold = float(exit_cfg.exit_threshold)
        self.feature_columns: list[str] = list(EXIT_FEATURE_COLUMNS)

    # ------------------------------------------------------------------
    def train(self, exit_dataset: pd.DataFrame) -> dict[str, float]:
        """Train the exit model on rows from ``build_exit_dataset()``.

        Parameters
        ----------
        exit_dataset :
            DataFrame with EXIT_FEATURE_COLUMNS + ``should_exit`` label.

        Returns
        -------
        dict with precision, recall, roc_auc, positive_rate on hold-out val.
        """
        if exit_dataset.empty:
            raise ValueError("exit_dataset is empty — cannot train ExitModel")

        avail = [c for c in self.feature_columns if c in exit_dataset.columns]
        missing = set(self.feature_columns) - set(avail)
        if missing:
            LOGGER.warning("ExitModel: missing features (will be 0): %s", missing)
        self.feature_columns = avail

        y = exit_dataset["should_exit"].values.astype(int)
        # Time-ordered 80/20 split for validation
        split = int(len(exit_dataset) * 0.80)
        X_train = self.scaler.fit_transform(exit_dataset[avail].values[:split])
        X_val   = self.scaler.transform(exit_dataset[avail].values[split:])
        y_train = y[:split]
        y_val   = y[split:]

        LOGGER.info(
            "ExitModel training: %d rows | positive_rate=%.1f%%",
            split, float(y_train.mean()) * 100,
        )
        self.model.fit(X_train, y_train)

        # Optimise threshold on validation set (maximise F1)
        val_proba = self.model.predict_proba(X_val)[:, 1]
        best_thr, best_f1 = 0.50, -1.0
        for thr in np.arange(0.35, 0.80, 0.01):
            preds = (val_proba >= thr).astype(int)
            if preds.sum() < 3:
                continue
            prec = precision_score(y_val, preds, zero_division=0)
            rec  = recall_score(y_val, preds, zero_division=0)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1 and prec >= 0.40:   # precision floor: 40%
                best_f1, best_thr = f1, float(thr)
        self.threshold = best_thr

        val_preds = (val_proba >= self.threshold).astype(int)
        metrics = {
            "train_rows": int(split),
            "val_rows": int(len(y_val)),
            "positive_rate": float(y_train.mean()),
            "threshold": float(self.threshold),
            "val_precision": float(precision_score(y_val, val_preds, zero_division=0)),
            "val_recall":    float(recall_score(y_val, val_preds, zero_division=0)),
            "val_roc_auc":   float(roc_auc_score(y_val, val_proba)) if len(set(y_val)) > 1 else 0.5,
            "feature_columns": self.feature_columns,
        }
        LOGGER.info(
            "ExitModel trained | thr=%.2f prec=%.2f rec=%.2f auc=%.3f",
            self.threshold,
            metrics["val_precision"],
            metrics["val_recall"],
            metrics["val_roc_auc"],
        )
        return metrics

    # ------------------------------------------------------------------
    def predict(
        self,
        market_row: pd.Series,
        pos_state: dict[str, float],
    ) -> float:
        """Return P(should_exit) given current market features + pos state.

        Parameters
        ----------
        market_row :
            Series with FEATURE_COLUMNS (e.g. live_frame.iloc[-1]).
        pos_state :
            Dict with EXIT_EXTRA_FEATURES keys:
              xm_bars_held, xm_unrealized_rr, xm_pos_side,
              xm_rsi_at_entry, xm_rsi_delta, xm_price_vs_entry_atr

        Returns
        -------
        float in [0, 1] — probability that the position should be exited.
        """
        row_dict: dict[str, float] = {}
        for col in FEATURE_COLUMNS:
            v = market_row.get(col, 0.0) if hasattr(market_row, "get") else getattr(market_row, col, 0.0)
            row_dict[col] = float(v) if v is not None else 0.0
        for col in EXIT_EXTRA_FEATURES:
            row_dict[col] = float(pos_state.get(col, 0.0))

        # Use only the columns the model was trained on, in order
        x = np.array([[row_dict.get(c, 0.0) for c in self.feature_columns]], dtype=float)
        x_scaled = self.scaler.transform(x)
        return float(self.model.predict_proba(x_scaled)[0, 1])

    # ------------------------------------------------------------------
    def save(self) -> None:
        exit_cfg = self.settings.execution.exit_model
        model_path  = Path(exit_cfg.exit_model_path)
        scaler_path = Path(exit_cfg.exit_scaler_path)
        meta_path   = Path(exit_cfg.exit_model_meta_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as fh:
            pickle.dump(self.model, fh)
        with scaler_path.open("wb") as fh:
            pickle.dump(self.scaler, fh)
        meta_path.write_text(json.dumps({
            "threshold": self.threshold,
            "feature_columns": self.feature_columns,
        }, indent=2), encoding="utf-8")
        LOGGER.info("ExitModel saved → %s (thr=%.2f)", model_path, self.threshold)

    def load(self) -> bool:
        """Load saved artifacts. Returns True on success, False if not found."""
        exit_cfg = self.settings.execution.exit_model
        model_path  = Path(exit_cfg.exit_model_path)
        scaler_path = Path(exit_cfg.exit_scaler_path)
        meta_path   = Path(exit_cfg.exit_model_meta_path)
        if not model_path.exists() or not scaler_path.exists():
            return False
        with model_path.open("rb") as fh:
            self.model = pickle.load(fh)
        with scaler_path.open("rb") as fh:
            self.scaler = pickle.load(fh)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # NOTE: threshold is intentionally NOT loaded from meta —
            # the config exit_threshold is the optimised override (e.g. 0.65).
            # Meta threshold (training-time F1 optimum) is informational only.
            if "feature_columns" in meta:
                self.feature_columns = list(meta["feature_columns"])
        LOGGER.info("ExitModel loaded from %s (thr=%.2f)", model_path, self.threshold)
        return True
