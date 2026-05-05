from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from xauusd_ai.config import load_settings


class ConfigValidationTests(unittest.TestCase):
    def _write_yaml(self, content: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "settings.yaml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_rejects_unknown_top_level_key(self) -> None:
        path = self._write_yaml(
            """
            app:
              poll_seconds: 60
            unexpected_root_key: true
            """
        )
        with self.assertRaises(ValidationError):
            load_settings(path)

    def test_rejects_unknown_nested_key(self) -> None:
        path = self._write_yaml(
            """
            risk:
              mode: fixed_fractional
              unknown_risk_toggle: 1
            """
        )
        with self.assertRaises(ValidationError):
            load_settings(path)

    def test_accepts_new_reentry_guard_settings(self) -> None:
        path = self._write_yaml(
            """
            risk:
              reentry_guard_enabled: false
              reentry_cooldown_bars_after_sl: 3
              reentry_min_distance_atr: 0.75
            """
        )
        settings = load_settings(path)
        self.assertFalse(settings.risk.reentry_guard_enabled)
        self.assertEqual(settings.risk.reentry_cooldown_bars_after_sl, 3)
        self.assertAlmostEqual(settings.risk.reentry_min_distance_atr, 0.75, places=6)

    def test_repo_configs_are_valid_under_strict_mode(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        config_dir = repo_root / "configs"
        yaml_files = sorted(config_dir.glob("*.yaml"))
        self.assertGreater(len(yaml_files), 0)
        for cfg in yaml_files:
            with self.subTest(config=str(cfg.name)):
                loaded = load_settings(cfg)
                self.assertIsNotNone(loaded)


if __name__ == "__main__":
    unittest.main()
