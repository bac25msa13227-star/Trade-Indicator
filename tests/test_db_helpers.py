from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from xauusd_ai.infra.db import _parse_dt, get_db_url


class DbHelpersTests(unittest.TestCase):
    def test_get_db_url_builds_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "POSTGRES_HOST": "localhost",
                "POSTGRES_PORT": "5433",
                "POSTGRES_USER": "u",
                "POSTGRES_PASSWORD": "p",
                "POSTGRES_DB": "d",
            },
            clear=False,
        ):
            url = get_db_url()
        self.assertEqual(url, "postgresql+psycopg2://u:p@localhost:5433/d")

    def test_parse_dt_handles_valid_and_invalid(self) -> None:
        now = datetime.now(timezone.utc)
        self.assertIs(_parse_dt(now), now)
        parsed = _parse_dt("2026-03-30T09:00:00")
        self.assertIsNotNone(parsed)
        self.assertIsNone(_parse_dt("not-a-date"))
        self.assertIsNone(_parse_dt(None))


if __name__ == "__main__":
    unittest.main()
