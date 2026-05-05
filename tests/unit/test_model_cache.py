from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from xauusd_ai.infra import _model_cache


class ModelCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        _model_cache._predictors.clear()

    def test_get_predictor_caches_per_account_and_invalidate_reloads(self) -> None:
        fake_trainer = Mock()
        fake_trainer.score_live_row.return_value = (0.77, True, "buy")
        with patch.object(_model_cache._Predictor, "_load", return_value=fake_trainer):
            p1 = _model_cache.get_predictor("acc1")
            p2 = _model_cache.get_predictor("acc1")
            self.assertIs(p1, p2)
            self.assertEqual(p1.score({"rsi": 55})[0], 0.77)

            _model_cache.invalidate("acc1")
            p3 = _model_cache.get_predictor("acc1")
            self.assertIsNot(p1, p3)


if __name__ == "__main__":
    unittest.main()
