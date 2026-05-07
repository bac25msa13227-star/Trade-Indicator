"""
Ensemble Model - Combines LightGBM/HGB + LSTM

P1 Ensemble Technique:
- Tree-based model (HGB/LightGBM): Captures feature interactions, non-linear patterns
- LSTM: Captures temporal sequences and momentum patterns
- Weighted voting: Combine predictions with configurable weights
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble wrapper combining tree-based and LSTM models."""
    
    def __init__(
        self,
        tree_model: Any,  # HGB or LightGBM sklearn-compatible model
        lstm_model: Any | None = None,  # LSTM wrapper or None
        tree_weight: float = 0.7,
        lstm_weight: float = 0.3,
    ):
        """
        Args:
            tree_model: Primary tree-based model (HGB, LightGBM, etc.)
            lstm_model: Optional LSTM model for temporal patterns
            tree_weight: Weight for tree model predictions (0-1)
            lstm_weight: Weight for LSTM predictions (0-1)
        """
        self.tree_model = tree_model
        self.lstm_model = lstm_model
        self.tree_weight = tree_weight
        self.lstm_weight = lstm_weight
        
        # Normalize weights
        total = tree_weight + (lstm_weight if lstm_model else 0)
        if total > 0:
            self.tree_weight = tree_weight / total
            self.lstm_weight = lstm_weight / total if lstm_model else 0
            
        self.classes_ = np.array([0, 1])  # Binary classification
        self._is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train both models."""
        logger.info(f"Training ensemble: tree_weight={self.tree_weight:.2f}, lstm_weight={self.lstm_weight:.2f}")
        
        # Train tree model
        logger.info("Training tree-based model...")
        self.tree_model.fit(X, y)
        
        # Train LSTM if available
        if self.lstm_model is not None:
            logger.info("Training LSTM model...")
            try:
                self.lstm_model.fit(X, y)
                logger.info("LSTM training completed")
            except Exception as e:
                logger.error(f"LSTM training failed: {e}")
                logger.warning("Falling back to tree-only mode")
                self.lstm_model = None
                self.tree_weight = 1.0
                self.lstm_weight = 0.0
                
        self._is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using ensemble."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using weighted ensemble."""
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted yet")
            
        # Get tree predictions
        tree_proba = self.tree_model.predict_proba(X)
        
        # If no LSTM, return tree predictions
        if self.lstm_model is None or self.lstm_weight == 0:
            return tree_proba
            
        # Get LSTM predictions
        try:
            lstm_proba = self.lstm_model.predict_proba(X)
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}, using tree-only")
            return tree_proba
            
        # Weighted ensemble
        ensemble_proba = (
            self.tree_weight * tree_proba +
            self.lstm_weight * lstm_proba
        )
        
        # Ensure probabilities sum to 1
        ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)
        
        return ensemble_proba
        
    def get_feature_importance(self) -> np.ndarray | None:
        """Get feature importance from tree model (LSTM doesn't have explicit importance)."""
        if hasattr(self.tree_model, 'feature_importances_'):
            return self.tree_model.feature_importances_
        return None


def create_ensemble_model(tree_model, lstm_model=None, settings=None) -> EnsembleModel:
    """Factory function to create ensemble from settings."""
    if settings is None or lstm_model is None:
        # No ensemble config or no LSTM - return tree-only
        return EnsembleModel(tree_model, None, tree_weight=1.0, lstm_weight=0.0)
        
    ensemble_config = getattr(settings.training, 'ensemble', None)
    if ensemble_config is None:
        # Default weights: 70% tree, 30% LSTM
        tree_weight = 0.7
        lstm_weight = 0.3
    else:
        tree_weight = getattr(ensemble_config, 'tree_weight', 0.7)
        lstm_weight = getattr(ensemble_config, 'lstm_weight', 0.3)
        
    return EnsembleModel(
        tree_model=tree_model,
        lstm_model=lstm_model,
        tree_weight=tree_weight,
        lstm_weight=lstm_weight,
    )
