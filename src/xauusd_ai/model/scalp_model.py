"""
Dual M1 Scalp Model — picklable, direction-aware.

Architecture:
  ▶ Two separate HistGBDT models: one for BUY direction, one for SELL direction
  ▶ Each model has its own scaler + isotonic calibration bundled inside
  ▶ DualScalpModel is the top-level picklable artifact saved to the configured
    `outputs/*_model.pkl` path (for example ACC2 freeze H10 or ACC1 scalp M1)

Pickle safety: all classes defined at module level → fully importable during unpickling.
"""
from __future__ import annotations

import numpy as np


class CalibratedDirModel:
    """
    Isotonic-calibrated HistGBDT for a single trade direction (BUY or SELL).

    Scaling is handled internally — the model accepts RAW (unscaled) features.
    This avoids double-scaling when loaded alongside a separate scaler artifact.
    """

    def __init__(
        self,
        base_model,
        isotonic,
        scaler,
        direction: int,  # +1 = BUY, -1 = SELL
    ) -> None:
        self._b        = base_model   # HistGradientBoostingClassifier
        self._iso      = isotonic     # IsotonicRegression
        self._sc       = scaler       # StandardScaler (fitted on training data)
        self.direction = direction

    # ------------------------------------------------------------------
    # sklearn-compatible API (accepts raw X, handles scaling internally)
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return [[P(loss), P(win)]] for each row.  Input: RAW unscaled features."""
        X_s = self._sc.transform(X)
        raw = self._b.predict_proba(X_s)[:, 1]
        cal = self._iso.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict_win_proba(self, X: np.ndarray) -> np.ndarray:
        """Convenience: P(win) only.  Input: RAW unscaled features."""
        return self.predict_proba(X)[:, 1]


class DualScalpModel:
    """
    Wraps a BUY model and a SELL model for M1 scalping inference.

    Saved to the configured `outputs/*_model.pkl` path.

    Usage:
        model = pickle.load(open("outputs/<your_model>.pkl", "rb"))

        # Score a batch of M1 bars (pre-filtered to one direction):
        buy_proba  = model.score_buy(X_buy_rows)    # rows where expected_direction == +1
        sell_proba = model.score_sell(X_sell_rows)  # rows where expected_direction == -1

        # Or score a mixed batch with direction column:
        proba = model.score_batch(X, directions)    # directions: array of +1/-1
    """

    def __init__(
        self,
        buy_model:  CalibratedDirModel | None,
        sell_model: CalibratedDirModel | None,
        feature_columns: list[str],
        thr_buy:  float = 0.58,
        thr_sell: float = 0.55,
        train_end: str  = "",   # ISO date string of last training bar
    ) -> None:
        self.buy_model       = buy_model
        self.sell_model      = sell_model
        self.feature_columns = list(feature_columns)
        self.thr_buy         = thr_buy
        self.thr_sell        = thr_sell
        self.train_end       = train_end   # for provenance tracking

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def score_buy(self, X: np.ndarray) -> np.ndarray:
        """P(win) for BUY candidates.  Input: RAW features."""
        if self.buy_model is None:
            return np.zeros(len(X))
        return self.buy_model.predict_win_proba(X)

    def score_sell(self, X: np.ndarray) -> np.ndarray:
        """P(win) for SELL candidates.  Input: RAW features."""
        if self.sell_model is None:
            return np.zeros(len(X))
        return self.sell_model.predict_win_proba(X)

    def score_batch(
        self,
        X: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        """
        Score a mixed batch by routing each row to the correct direction model.

        directions: int array, +1 = BUY direction, -1 = SELL direction.
        Returns: float array of win probabilities, shape (n,).
        """
        n   = len(X)
        out = np.zeros(n, dtype=float)
        buy_idx  = np.where(directions ==  1)[0]
        sell_idx = np.where(directions == -1)[0]
        if len(buy_idx):
            out[buy_idx]  = self.score_buy(X[buy_idx])
        if len(sell_idx):
            out[sell_idx] = self.score_sell(X[sell_idx])
        return out

    def build_signal_df(self, test_df) -> "pandas.DataFrame":  # noqa: F821
        """
        Given a test DataFrame (from build_scalp_dataset), returns a signal
        DataFrame suitable for simulate_dynamic_concurrent_backtest.

        Adds columns: prediction, probability, trade_side, strategy_score,
        volatility_regime, trend_alignment, adx.
        """
        import pandas as pd

        n    = len(test_df)
        dirs = test_df["expected_direction"].values
        X    = test_df[self.feature_columns].fillna(0).values

        buy_p  = np.zeros(n)
        sell_p = np.zeros(n)

        buy_mask  = dirs ==  1
        sell_mask = dirs == -1

        if buy_mask.any():
            buy_p[buy_mask]  = self.score_buy(X[buy_mask])
        if sell_mask.any():
            sell_p[sell_mask] = self.score_sell(X[sell_mask])

        buy_sig  = buy_mask  & (buy_p  >= self.thr_buy)
        sell_sig = sell_mask & (sell_p >= self.thr_sell)

        out = test_df.copy()
        out["split"]            = "test"
        out["prediction"]       = 0
        out.loc[buy_sig,  "prediction"] = 1
        out.loc[sell_sig, "prediction"] = 1
        out["probability"]       = np.where(sell_mask, sell_p, buy_p)
        out["trade_side"]        = np.where(sell_mask, "sell", "buy")
        out["strategy_score"]    = 1.0
        # Use real volatility_regime from dataset (computed by infer_scalp_volatility_regime)
        # instead of hardcoding 1 — matches live regime inference.
        if "volatility_regime" not in out.columns:
            out["volatility_regime"] = 1
        out["trend_alignment"]   = 1
        out["adx"]               = 25.0
        return out

    # ------------------------------------------------------------------
    # Magic
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        buy_ok  = self.buy_model  is not None
        sell_ok = self.sell_model is not None
        return (
            f"DualScalpModel(buy={'✓' if buy_ok else '✗'}, "
            f"sell={'✓' if sell_ok else '✗'}, "
            f"features={len(self.feature_columns)}, "
            f"thr_buy={self.thr_buy}, thr_sell={self.thr_sell}, "
            f"train_end='{self.train_end}')"
        )
