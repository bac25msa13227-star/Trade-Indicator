from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from xauusd_ai.monitoring.drift import DriftMonitor


class DriftHelpersTests(unittest.TestCase):
    def test_extract_drift_score_and_columns(self) -> None:
        result = {
            "metrics": [
                {"metric": "DatasetDriftMetric", "result": {"share_of_drifted_columns": 0.42}},
                {"metric": "ColumnDriftMetric", "result": {"drift_detected": True, "column_name": "rsi"}},
                {"metric": "ColumnDriftMetric", "result": {"drift_detected": False, "column_name": "macd"}},
            ]
        }
        self.assertAlmostEqual(DriftMonitor._extract_drift_score(result), 0.42, places=6)
        self.assertEqual(DriftMonitor._extract_drifted_columns(result), ["rsi"])

    def test_run_feature_drift_fallback_when_evidently_unavailable(self) -> None:
        monitor = DriftMonitor(account="acc1")
        with patch("xauusd_ai.monitoring.drift._EVIDENTLY_AVAILABLE", False):
            out = monitor.run_feature_drift(pd.DataFrame({"rsi": [1, 2]}), pd.DataFrame({"rsi": [2, 3]}))
        self.assertEqual(out["drift_detected"], False)
        self.assertEqual(out["drift_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
