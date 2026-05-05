from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.orchestrator import run_backtest


class BacktestArtifactSafetyTests(unittest.TestCase):
    @patch("xauusd_ai.orchestrator.write_backtest_outputs")
    @patch("xauusd_ai.orchestrator.simulate_prediction_backtest")
    @patch("xauusd_ai.orchestrator.prepare_training_dataset")
    @patch("xauusd_ai.orchestrator._bootstrap")
    @patch("builtins.print")
    def test_backtest_training_never_saves_live_artifacts(
        self,
        _print_mock: MagicMock,
        bootstrap_mock: MagicMock,
        prepare_dataset_mock: MagicMock,
        simulate_mock: MagicMock,
        write_outputs_mock: MagicMock,
    ) -> None:
        settings = Settings()

        data_service = MagicMock()
        trainer = MagicMock()
        strategy = MagicMock()
        notifier = MagicMock()
        executor = MagicMock()
        risk_manager = MagicMock()

        bootstrap_mock.return_value = (
            data_service,
            trainer,
            strategy,
            notifier,
            executor,
            risk_manager,
        )

        data_service.fetch_multi_timeframe_data.return_value = {
            "M5": pd.DataFrame({"time": [pd.Timestamp("2026-03-20T00:00:00Z")]})
        }

        dataset = pd.DataFrame(
            {
                "split": ["train", "test"],
                "time": [
                    pd.Timestamp("2026-03-20T00:00:00Z"),
                    pd.Timestamp("2026-03-21T00:00:00Z"),
                ],
                "target": [1, 0],
            }
        )
        predictions = pd.DataFrame(
            {
                "split": ["train", "test"],
                "time": [
                    pd.Timestamp("2026-03-20T00:00:00Z"),
                    pd.Timestamp("2026-03-21T00:00:00Z"),
                ],
                "target": [1, 0],
                "probability": [0.62, 0.41],
                "prediction": [1, 0],
            }
        )
        prepare_dataset_mock.return_value = dataset
        trainer.train.return_value = {"roc_auc": 0.61}
        trainer.predict_dataset.return_value = predictions

        simulate_mock.return_value = SimpleNamespace(
            trades=pd.DataFrame({"time": ["2026-03-21T00:00:00Z"]}),
            report={"net_profit": 12.5},
        )

        run_backtest(settings)

        trainer.train.assert_called_once()
        _, kwargs = trainer.train.call_args
        self.assertIn("save_artifacts", kwargs)
        self.assertFalse(kwargs["save_artifacts"])
        write_outputs_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
