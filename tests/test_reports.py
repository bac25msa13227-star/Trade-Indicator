from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from xauusd_ai.visualization.reports import save_backtest_plots, save_training_plot


class ReportsTests(unittest.TestCase):
    def test_save_training_plot_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "training.png"
            save_training_plot(
                {"accuracy": 0.6, "precision": 0.5, "recall": 0.4, "f1": 0.45, "roc_auc": 0.7},
                path,
            )
            self.assertTrue(path.exists())

    def test_save_backtest_plots_creates_trade_marker_plot_even_without_trades(self) -> None:
        test_rows = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-03-30 09:00:00+00:00", "2026-03-30 09:15:00+00:00"]),
                "close": [2000.0, 2001.0],
            }
        )
        with tempfile.TemporaryDirectory() as td:
            equity = Path(td) / "equity.png"
            markers = Path(td) / "markers.png"
            save_backtest_plots(test_rows, pd.DataFrame(), equity, markers)
            self.assertTrue(markers.exists())


if __name__ == "__main__":
    unittest.main()
