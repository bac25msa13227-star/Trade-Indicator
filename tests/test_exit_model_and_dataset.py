from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.features.dataset import FEATURE_COLUMNS
from xauusd_ai.features.exit_dataset import EXIT_EXTRA_FEATURES, build_exit_dataset
from xauusd_ai.model.exit_model import ExitModel


class _DummyScaler:
    def transform(self, x):
        return x


class _DummyModel:
    def predict_proba(self, x):
        return np.array([[0.2, 0.8]])


class ExitModelAndDatasetTests(unittest.TestCase):
    def _base_dataset(self, n: int = 100) -> pd.DataFrame:
        time = pd.date_range("2026-03-01 00:00:00+00:00", periods=n, freq="15min")
        base = {
            "time": time,
            "close": np.linspace(2000.0, 2003.0, n),
            "high": np.linspace(2000.2, 2003.2, n),
            "low": np.linspace(1999.8, 2002.8, n),
            "open": np.linspace(1999.9, 2002.9, n),
            "atr": np.full(n, 1.0),
            "rsi": np.full(n, 55.0),
            "trade_side": ["buy"] * n,
            "split": ["train"] * n,
            "target": [0] * n,
        }
        base["target"][10] = 1
        for col in FEATURE_COLUMNS:
            if col not in base:
                base[col] = np.zeros(n)
        return pd.DataFrame(base)

    def test_build_exit_dataset_produces_rows_and_columns(self) -> None:
        settings = Settings()
        settings.execution.exit_model.include_loss_entries = False
        settings.execution.exit_model.max_entry_samples = 50
        ds = self._base_dataset(110)
        exit_df = build_exit_dataset(ds, settings)
        self.assertFalse(exit_df.empty)
        self.assertIn("should_exit", exit_df.columns)
        for col in EXIT_EXTRA_FEATURES:
            self.assertIn(col, exit_df.columns)

    def test_exit_model_predict_and_save_load(self) -> None:
        settings = Settings()
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            settings.execution.exit_model.exit_model_path = str(td_path / "exit_model.pkl")
            settings.execution.exit_model.exit_scaler_path = str(td_path / "exit_scaler.pkl")
            settings.execution.exit_model.exit_model_meta_path = str(td_path / "exit_meta.json")

            model = ExitModel(settings)
            model.feature_columns = ["rsi", "xm_bars_held"]
            model.scaler = _DummyScaler()
            model.model = _DummyModel()
            prob = model.predict(pd.Series({"rsi": 56.0}), {"xm_bars_held": 0.25})
            self.assertAlmostEqual(prob, 0.8, places=6)

            model.save()
            self.assertTrue(Path(settings.execution.exit_model.exit_model_path).exists())
            self.assertTrue(Path(settings.execution.exit_model.exit_scaler_path).exists())
            self.assertTrue(Path(settings.execution.exit_model.exit_model_meta_path).exists())

            loaded = ExitModel(settings)
            self.assertTrue(loaded.load())


if __name__ == "__main__":
    unittest.main()
