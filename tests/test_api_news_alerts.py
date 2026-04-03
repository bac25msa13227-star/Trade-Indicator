from __future__ import annotations

import unittest

from xauusd_ai.api.main import _is_news_alert_impact


class ApiNewsAlertImpactTests(unittest.TestCase):
    def test_high_and_medium_are_alertable(self) -> None:
        self.assertTrue(_is_news_alert_impact("High"))
        self.assertTrue(_is_news_alert_impact("Medium"))
        self.assertTrue(_is_news_alert_impact("high"))
        self.assertTrue(_is_news_alert_impact(" medium "))

    def test_low_or_unknown_are_not_alertable(self) -> None:
        self.assertFalse(_is_news_alert_impact("Low"))
        self.assertFalse(_is_news_alert_impact("Holiday"))
        self.assertFalse(_is_news_alert_impact(""))
        self.assertFalse(_is_news_alert_impact("Non-Economic"))


if __name__ == "__main__":
    unittest.main()

