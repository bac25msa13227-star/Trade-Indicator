from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.data.market_data import MarketDataService


class MarketDataCsvTests(unittest.TestCase):
    def _write_sample_csv(self, path: Path) -> None:
        times = pd.date_range("2026-03-30 00:00:00+00:00", periods=8, freq="15min")
        df = pd.DataFrame(
            {
                "Datetime": times,
                "Open": [2000 + i for i in range(8)],
                "High": [2001 + i for i in range(8)],
                "Low": [1999 + i for i in range(8)],
                "Close": [2000.5 + i for i in range(8)],
                "Volume": [100 + i for i in range(8)],
                "Spread": [30 for _ in range(8)],
            }
        )
        df.to_csv(path, index=False)

    def test_fetch_rates_csv_and_resample(self) -> None:
        settings = Settings()
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "xau.csv"
            self._write_sample_csv(csv_path)
            settings.market.csv_data_path = str(csv_path)
            settings.market.csv_timeframe = "M15"

            svc = MarketDataService(settings)
            m15 = svc._fetch_rates_csv("M15", bars=100)
            h1 = svc._fetch_rates_csv("H1", bars=100)

            self.assertEqual(len(m15), 8)
            self.assertEqual(len(h1), 2)
            self.assertIn("volume_imbalance", h1.columns)
            self.assertIn("tick_volume_delta", h1.columns)

    def test_fetch_rates_csv_folder_selects_timeframe_file(self) -> None:
        settings = Settings()
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            file_path = folder / "XAUUSD_M15.csv"
            self._write_sample_csv(file_path)
            settings.market.csv_folder_path = str(folder)

            svc = MarketDataService(settings)
            out = svc._fetch_rates_csv_folder("M15", bars=50)
            self.assertEqual(len(out), 8)
            self.assertIn("spread_points", out.columns)

    def test_fetch_rates_unknown_source_raises(self) -> None:
        settings = Settings()
        svc = MarketDataService(settings)
        with self.assertRaises(ValueError):
            svc._fetch_rates("M15", bars=10, source="unknown")


if __name__ == "__main__":
    unittest.main()
