from __future__ import annotations

import unittest
from datetime import datetime, timezone

import pandas as pd

from xauusd_ai.data.news_features import (
    _direction_from_surprise,
    _direction_from_title,
    _first_friday,
    _generate_rule_based_calendar,
    _last_friday,
    _weekday_skip,
    attach_news_features,
)


class NewsFeaturesTests(unittest.TestCase):
    def test_direction_from_title_and_surprise(self) -> None:
        self.assertEqual(_direction_from_title("US CPI m/m"), 1)
        self.assertEqual(_direction_from_title("Nonfarm Payrolls"), -1)
        self.assertEqual(_direction_from_surprise("US CPI m/m", 3.5, 3.2), 1)
        self.assertEqual(_direction_from_surprise("US CPI m/m", 3.0, 3.2), -1)

    def test_first_and_last_friday(self) -> None:
        first = _first_friday(2026, 3)
        last = _last_friday(2026, 3)
        self.assertEqual(first.weekday(), 4)
        self.assertEqual(last.weekday(), 4)
        self.assertLessEqual(first.day, 7)

    def test_weekday_skip_moves_weekend_to_monday(self) -> None:
        sat = datetime(2026, 3, 28, 13, 30, tzinfo=timezone.utc)  # Saturday
        sun = datetime(2026, 3, 29, 13, 30, tzinfo=timezone.utc)  # Sunday
        self.assertEqual(_weekday_skip(sat).weekday(), 0)
        self.assertEqual(_weekday_skip(sun).weekday(), 0)

    def test_rule_based_calendar_has_expected_columns(self) -> None:
        cal = _generate_rule_based_calendar(
            pd.Timestamp("2026-03-01 00:00:00+00:00"),
            pd.Timestamp("2026-03-31 23:59:00+00:00"),
        )
        self.assertFalse(cal.empty)
        self.assertIn("datetime_utc", cal.columns)
        self.assertIn("gold_direction", cal.columns)

    def test_attach_news_features_vectorized(self) -> None:
        bars = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2026-03-30 09:30:00+00:00",
                        "2026-03-30 10:15:00+00:00",
                        "2026-03-30 12:00:00+00:00",
                    ]
                ),
                "open": [1, 1, 1],
                "high": [1, 1, 1],
                "low": [1, 1, 1],
                "close": [1, 1, 1],
                "tick_volume": [1, 1, 1],
            }
        )
        cal = pd.DataFrame(
            {
                "datetime_utc": [pd.Timestamp("2026-03-30 10:00:00+00:00")],
                "event": ["CPI"],
                "impact": ["High"],
                "gold_direction": [1],
                "source": ["rule_based"],
            }
        )
        out = attach_news_features(bars, news_calendar=cal, blackout_hours=1.0, lookahead_hours=4.0)
        self.assertEqual(out.loc[0, "news_impact_ahead"], 2)
        self.assertEqual(out.loc[0, "news_is_blackout"], 1)
        self.assertEqual(out.loc[2, "news_surprise_gold"], 1)


if __name__ == "__main__":
    unittest.main()
