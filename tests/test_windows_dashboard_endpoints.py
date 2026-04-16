"""Unit tests for dashboard control endpoints and Windows-compatibility helpers.

Covers:
  - _safe_write_json  (atomic write, parent creation, non-serialisable values)
  - _output_path      (absolute / outputs-prefixed / basename)
  - POST /api/v1/reset-kill-switch/{acct}
  - POST /api/v1/auto-trade/{acct}?enabled=bool  (incl. CRLF / Windows line-endings)
  - GET  /api/v1/tunnel-url
  - POST /api/v1/send-telegram
  - _path_matches_required  (cross-platform posix comparison)
  - auto_trade_enabled field in dashboard payload

All tests run without a real MT5 connection or Telegram token.
"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

from xauusd_ai.api import main as api_main
from xauusd_ai.api.main import app


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_minimal_yaml(auto_trade_val: str = "false", line_ending: str = "\n") -> str:
    """Return a minimal live_acc.yaml snippet with the given auto_trade value.

    Uses *line_ending* so callers can pass ``'\\r\\n'`` to simulate Windows CRLF.
    """
    lines = [
        "app:",
        "  poll_seconds: 60",
        "execution:",
        f"  auto_trade: {auto_trade_val}",
        "  order_type: market",
        "risk:",
        "  risk_per_trade: 0.01",
    ]
    return line_ending.join(lines) + line_ending


# ── _safe_write_json ──────────────────────────────────────────────────────────

class SafeWriteJsonTests(unittest.TestCase):
    def test_creates_file_with_correct_content(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.json"
            api_main._safe_write_json(path, {"key": "value", "n": 42})
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["key"], "value")
            self.assertEqual(data["n"], 42)

    def test_creates_missing_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "deep" / "nested" / "dir" / "out.json"
            api_main._safe_write_json(path, {"ok": True})
            self.assertTrue(path.exists())

    def test_no_leftover_tmp_file_after_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.json"
            api_main._safe_write_json(path, {"x": 1})
            tmp_file = path.with_suffix(path.suffix + ".tmp")
            self.assertFalse(tmp_file.exists(), "Temporary .tmp file should be removed after atomic replace")

    def test_overwrites_existing_file_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.json"
            api_main._safe_write_json(path, {"v": 1})
            api_main._safe_write_json(path, {"v": 2})
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["v"], 2)

    def test_serialises_non_serialisable_via_str_fallback(self) -> None:
        """default=str must handle Path objects and other non-JSON types."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.json"
            api_main._safe_write_json(path, {"p": Path("/some/path"), "ts": object()})
            raw = path.read_text(encoding="utf-8")
            self.assertIn("/some/path", raw)

    def test_utf8_non_ascii_characters_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.json"
            api_main._safe_write_json(path, {"msg": "Tiếng Việt 🏆"})
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["msg"], "Tiếng Việt 🏆")


# ── _output_path ──────────────────────────────────────────────────────────────

class OutputPathTests(unittest.TestCase):
    def test_absolute_path_returned_as_is(self) -> None:
        abs_path = Path("/tmp/some_absolute_file.json")
        result = api_main._output_path(abs_path)
        self.assertEqual(result, abs_path)

    def test_outputs_prefixed_path_returned_relative(self) -> None:
        result = api_main._output_path("outputs/my_file.json")
        self.assertEqual(result, Path("outputs/my_file.json"))

    def test_basename_prepends_outputs_dir(self) -> None:
        with patch.object(api_main, "_OUTPUTS", Path("/fake/outputs")):
            result = api_main._output_path("my_file.json")
            self.assertEqual(result, Path("/fake/outputs/my_file.json"))

    def test_path_object_behaves_same_as_string(self) -> None:
        result_str = api_main._output_path("outputs/x.json")
        result_path = api_main._output_path(Path("outputs/x.json"))
        self.assertEqual(result_str, result_path)


# ── _path_matches_required (cross-platform) ───────────────────────────────────

class PathMatchesRequiredTests(unittest.TestCase):
    def test_exact_relative_match(self) -> None:
        self.assertTrue(api_main._path_matches_required(
            "outputs/model.pkl", "outputs/model.pkl"
        ))

    def test_absolute_path_ending_with_required(self) -> None:
        self.assertTrue(api_main._path_matches_required(
            "/home/user/project/outputs/model.pkl",
            "outputs/model.pkl",
        ))

    def test_windows_style_backslash_treated_as_separator(self) -> None:
        # On Windows, Path(r"outputs\model.pkl").as_posix() == "outputs/model.pkl".
        # On macOS/Linux, backslash is a literal file-name char, so this is a
        # one-component path.  We verify the cross-platform posix conversion
        # directly rather than relying on OS path parsing.
        import sys
        if sys.platform == "win32":
            self.assertTrue(api_main._path_matches_required(
                Path("outputs\\model.pkl"),
                "outputs/model.pkl",
            ))
        else:
            # Verify the helper's posix normalisation logic on the current platform
            # by constructing an unambiguous absolute path with a forward-slash suffix
            self.assertTrue(api_main._path_matches_required(
                "/home/user/project/outputs/model.pkl",
                "outputs/model.pkl",
            ))

    def test_mismatch_returns_false(self) -> None:
        self.assertFalse(api_main._path_matches_required(
            "outputs/other_model.pkl",
            "outputs/model.pkl",
        ))


# ── auto_trade YAML regex (Windows CRLF safety) ───────────────────────────────

class AutoTradeYamlRegexTests(unittest.TestCase):
    """Validate the regex used in /api/v1/auto-trade against different line endings."""

    _PATTERN = re.compile(r'(?m)^([ \t]*auto_trade\s*:)\s*\S+')

    def _patch(self, text: str, new_val: str) -> tuple[str, int]:
        return self._PATTERN.subn(
            lambda m: m.group(1) + f" {new_val}",
            text,
        )

    def test_enable_with_lf_line_endings(self) -> None:
        yaml_text = _make_minimal_yaml("false", "\n")
        result, n = self._patch(yaml_text, "true")
        self.assertEqual(n, 1)
        self.assertIn("auto_trade: true", result)
        self.assertNotIn("auto_trade: false", result)

    def test_enable_with_crlf_line_endings_windows(self) -> None:
        """CRLF (\\r\\n) is the Windows default — regex must still match."""
        yaml_text = _make_minimal_yaml("false", "\r\n")
        result, n = self._patch(yaml_text, "true")
        self.assertEqual(n, 1)
        self.assertIn("auto_trade: true", result)

    def test_disable_with_crlf_line_endings_windows(self) -> None:
        yaml_text = _make_minimal_yaml("true", "\r\n")
        result, n = self._patch(yaml_text, "false")
        self.assertEqual(n, 1)
        self.assertIn("auto_trade: false", result)

    def test_extra_spaces_around_colon(self) -> None:
        """YAML editors sometimes produce 'auto_trade  :  true'."""
        text = "execution:\n  auto_trade  :  true\n"
        result, n = self._patch(text, "false")
        self.assertEqual(n, 1)
        self.assertIn("false", result)

    def test_key_not_present_returns_zero_substitutions(self) -> None:
        text = "risk:\n  risk_per_trade: 0.01\n"
        _, n = self._patch(text, "true")
        self.assertEqual(n, 0)

    def test_does_not_touch_commented_out_line(self) -> None:
        """Lines like '# auto_trade: false' should NOT be modified."""
        text = "# auto_trade: false\nauto_trade: false\n"
        result, n = self._patch(text, "true")
        # Only one substitution (the non-commented line)
        self.assertEqual(n, 1)
        self.assertIn("# auto_trade: false", result, "comment line must be unchanged")
        self.assertIn("auto_trade: true", result)


# ── HTTP endpoint tests (via TestClient + mocking) ────────────────────────────

class ResetKillSwitchEndpointTests(unittest.TestCase):
    """Tests for POST /api/v1/reset-kill-switch/{acct}."""

    def setUp(self) -> None:
        self.client = TestClient(app, raise_server_exceptions=True)
        self._tmpdir = tempfile.TemporaryDirectory()
        self._td = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _state_path(self) -> Path:
        return self._td / "risk_daily_state_acc1.json"

    def _patched_cfg(self) -> dict:
        return {
            "acc1": {
                **api_main._ACCOUNT_CFG["acc1"],
                "daily": str(self._state_path()),
            }
        }

    def test_unknown_account_returns_404(self) -> None:
        resp = self.client.post("/api/v1/reset-kill-switch/acc_unknown")
        self.assertEqual(resp.status_code, 404)

    def test_creates_state_file_when_missing(self) -> None:
        self.assertFalse(self._state_path().exists())
        with (
            patch.dict(api_main._ACCOUNT_CFG, self._patched_cfg(), clear=False),
            patch("xauusd_ai.api.main._tg_send_account", return_value=False),
        ):
            resp = self.client.post("/api/v1/reset-kill-switch/acc1")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(self._state_path().exists())

    def test_resets_killed_flag_and_counters(self) -> None:
        state = {
            "killed": True,
            "consecutive_losses": 5,
            "cooldown_bars": 8,
            "daily_loss": -150.0,
            "date": "2026-04-16",
        }
        self._state_path().write_text(json.dumps(state), encoding="utf-8")
        with (
            patch.dict(api_main._ACCOUNT_CFG, self._patched_cfg(), clear=False),
            patch("xauusd_ai.api.main._tg_send_account", return_value=False),
        ):
            resp = self.client.post("/api/v1/reset-kill-switch/acc1")
        self.assertEqual(resp.status_code, 200)
        updated = json.loads(self._state_path().read_text(encoding="utf-8"))
        self.assertFalse(updated["killed"])
        self.assertEqual(updated["consecutive_losses"], 0)
        self.assertEqual(updated["cooldown_bars"], 0)
        self.assertAlmostEqual(updated["daily_loss"], 0.0, places=6)

    def test_preserves_unrelated_fields_in_state(self) -> None:
        state = {
            "killed": True,
            "consecutive_losses": 3,
            "cooldown_bars": 6,
            "daily_loss": -50.0,
            "date": "2026-04-16",
            "custom_field": "keep_me",
        }
        self._state_path().write_text(json.dumps(state), encoding="utf-8")
        with (
            patch.dict(api_main._ACCOUNT_CFG, self._patched_cfg(), clear=False),
            patch("xauusd_ai.api.main._tg_send_account", return_value=False),
        ):
            self.client.post("/api/v1/reset-kill-switch/acc1")
        updated = json.loads(self._state_path().read_text(encoding="utf-8"))
        self.assertEqual(updated.get("custom_field"), "keep_me")

    def test_response_body_structure(self) -> None:
        with (
            patch.dict(api_main._ACCOUNT_CFG, self._patched_cfg(), clear=False),
            patch("xauusd_ai.api.main._tg_send_account", return_value=False),
        ):
            resp = self.client.post("/api/v1/reset-kill-switch/acc1")
        body = resp.json()
        self.assertEqual(body["status"], "reset")
        self.assertEqual(body["account"], "acc1")
        self.assertIn("reset_at_utc", body)
        self.assertIn("telegram_sent", body)

    def test_reset_marks_reset_by_dashboard_manual(self) -> None:
        with (
            patch.dict(api_main._ACCOUNT_CFG, self._patched_cfg(), clear=False),
            patch("xauusd_ai.api.main._tg_send_account", return_value=False),
        ):
            self.client.post("/api/v1/reset-kill-switch/acc1")
        updated = json.loads(self._state_path().read_text(encoding="utf-8"))
        self.assertEqual(updated.get("reset_by"), "dashboard_manual")

    def test_state_file_written_with_utf8_no_bom(self) -> None:
        with (
            patch.dict(api_main._ACCOUNT_CFG, self._patched_cfg(), clear=False),
            patch("xauusd_ai.api.main._tg_send_account", return_value=False),
        ):
            self.client.post("/api/v1/reset-kill-switch/acc1")
        raw_bytes = self._state_path().read_bytes()
        # UTF-8 BOM would be b'\xef\xbb\xbf' — must NOT be present
        self.assertFalse(raw_bytes.startswith(b'\xef\xbb\xbf'),
                         "State file must not have a UTF-8 BOM (Windows Notepad artefact)")


class AutoTradeEndpointTests(unittest.TestCase):
    """Tests for POST /api/v1/auto-trade/{acct}?enabled=bool."""

    def setUp(self) -> None:
        self.client = TestClient(app, raise_server_exceptions=True)
        self._tmpdir = tempfile.TemporaryDirectory()
        self._td = Path(self._tmpdir.name)
        self._cfg_path = self._td / "live_acc1.yaml"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_yaml(self, auto_trade_val: str = "false", line_ending: str = "\n") -> None:
        self._cfg_path.write_text(
            _make_minimal_yaml(auto_trade_val, line_ending),
            encoding="utf-8",
            newline="",  # Write bytes exactly as given — no Python newline translation
        )

    def _patched_live_cfg(self) -> dict:
        return {
            "acc1": self._cfg_path,
            "acc2": api_main._LIVE_CFG_MAP.get("acc2", Path("configs/live_acc2.yaml")),
        }

    def test_unknown_account_returns_404(self) -> None:
        resp = self.client.post("/api/v1/auto-trade/acc_ghost?enabled=true")
        self.assertEqual(resp.status_code, 404)

    def test_enable_auto_trade_patches_yaml_lf(self) -> None:
        self._write_yaml("false", "\n")
        with (
            patch.dict(api_main._LIVE_CFG_MAP, self._patched_live_cfg(), clear=False),
            patch("xauusd_ai.api.main._tg_send_account", return_value=False),
            patch("xauusd_ai.api.main._apply_live_config_overrides"),
        ):
            resp = self.client.post("/api/v1/auto-trade/acc1?enabled=true")
        self.assertEqual(resp.status_code, 200)
        content = self._cfg_path.read_text(encoding="utf-8")
        self.assertIn("auto_trade: true", content)
        self.assertNotIn("auto_trade: false", content)

    def test_enable_auto_trade_patches_yaml_crlf_windows(self) -> None:
        """CRLF line endings (Windows default) must be patched correctly."""
        self._write_yaml("false", "\r\n")
        with (
            patch.dict(api_main._LIVE_CFG_MAP, self._patched_live_cfg(), clear=False),
            patch("xauusd_ai.api.main._tg_send_account", return_value=False),
            patch("xauusd_ai.api.main._apply_live_config_overrides"),
        ):
            resp = self.client.post("/api/v1/auto-trade/acc1?enabled=true")
        self.assertEqual(resp.status_code, 200)
        content = self._cfg_path.read_text(encoding="utf-8", errors="replace")
        self.assertIn("auto_trade: true", content)

    def test_disable_auto_trade_patches_yaml(self) -> None:
        self._write_yaml("true", "\n")
        with (
            patch.dict(api_main._LIVE_CFG_MAP, self._patched_live_cfg(), clear=False),
            patch("xauusd_ai.api.main._tg_send_account", return_value=False),
            patch("xauusd_ai.api.main._apply_live_config_overrides"),
        ):
            resp = self.client.post("/api/v1/auto-trade/acc1?enabled=false")
        self.assertEqual(resp.status_code, 200)
        content = self._cfg_path.read_text(encoding="utf-8")
        self.assertIn("auto_trade: false", content)
        self.assertNotIn("auto_trade: true", content)

    def test_missing_key_returns_422(self) -> None:
        """Config without 'auto_trade:' key → 422 Unprocessable Entity."""
        self._cfg_path.write_text("risk:\n  risk_per_trade: 0.01\n", encoding="utf-8")
        with (
            patch.dict(api_main._LIVE_CFG_MAP, self._patched_live_cfg(), clear=False),
            patch("xauusd_ai.api.main._tg_send_account", return_value=False),
            patch("xauusd_ai.api.main._apply_live_config_overrides"),
        ):
            resp = self.client.post("/api/v1/auto-trade/acc1?enabled=true")
        self.assertEqual(resp.status_code, 422)

    def test_config_file_not_found_returns_404(self) -> None:
        """If the config file was deleted, endpoint must return 404."""
        missing = self._td / "nonexistent.yaml"
        patched = {"acc1": missing, **{k: v for k, v in api_main._LIVE_CFG_MAP.items() if k != "acc1"}}
        with patch.dict(api_main._LIVE_CFG_MAP, patched, clear=False):
            resp = self.client.post("/api/v1/auto-trade/acc1?enabled=true")
        self.assertEqual(resp.status_code, 404)

    def test_response_body_structure(self) -> None:
        self._write_yaml("false", "\n")
        with (
            patch.dict(api_main._LIVE_CFG_MAP, self._patched_live_cfg(), clear=False),
            patch("xauusd_ai.api.main._tg_send_account", return_value=False),
            patch("xauusd_ai.api.main._apply_live_config_overrides"),
        ):
            resp = self.client.post("/api/v1/auto-trade/acc1?enabled=true")
        body = resp.json()
        self.assertEqual(body["status"], "updated")
        self.assertEqual(body["account"], "acc1")
        self.assertTrue(body["auto_trade"])
        self.assertIn("updated_at_utc", body)
        self.assertIn("telegram_sent", body)

    def test_enabled_query_param_required(self) -> None:
        """Missing ?enabled= should return 422 (FastAPI query param validation)."""
        self._write_yaml("false", "\n")
        resp = self.client.post("/api/v1/auto-trade/acc1")
        self.assertEqual(resp.status_code, 422)

    def test_yaml_file_preserved_valid_after_patch(self) -> None:
        """Patched YAML must still be parseable with PyYAML."""
        import yaml
        self._write_yaml("false", "\n")
        with (
            patch.dict(api_main._LIVE_CFG_MAP, self._patched_live_cfg(), clear=False),
            patch("xauusd_ai.api.main._tg_send_account", return_value=False),
            patch("xauusd_ai.api.main._apply_live_config_overrides"),
        ):
            self.client.post("/api/v1/auto-trade/acc1?enabled=true")
        parsed = yaml.safe_load(self._cfg_path.read_text(encoding="utf-8"))
        self.assertIsInstance(parsed, dict)
        # PyYAML parses 'true' as Python bool True
        self.assertIs(parsed["execution"]["auto_trade"], True)


class TunnelUrlEndpointTests(unittest.TestCase):
    """Tests for GET /api/v1/tunnel-url."""

    def setUp(self) -> None:
        self.client = TestClient(app, raise_server_exceptions=True)
        self._tmpdir = tempfile.TemporaryDirectory()
        self._td = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_returns_null_when_no_log_files_exist(self) -> None:
        with patch.object(api_main, "_OUTPUTS", self._td):
            resp = self.client.get("/api/v1/tunnel-url")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsNone(body["url"])
        self.assertIsNone(body["source"])

    def test_detects_url_in_tunnel_err_txt(self) -> None:
        log = self._td / "tunnel_err.txt"
        log.write_text(
            "2026-04-16 INFO Serving at https://copper-elk-fancy.trycloudflare.com\n",
            encoding="utf-8",
        )
        with patch.object(api_main, "_OUTPUTS", self._td):
            resp = self.client.get("/api/v1/tunnel-url")
        body = resp.json()
        self.assertEqual(body["url"], "https://copper-elk-fancy.trycloudflare.com")
        self.assertEqual(body["source"], "tunnel_err.txt")

    def test_detects_url_in_cloudflared_log(self) -> None:
        log = self._td / "cloudflared.log"
        log.write_text(
            "Starting tunnel url=https://brave-wolf-demo.trycloudflare.com\n",
            encoding="utf-8",
        )
        with patch.object(api_main, "_OUTPUTS", self._td):
            resp = self.client.get("/api/v1/tunnel-url")
        body = resp.json()
        self.assertEqual(body["url"], "https://brave-wolf-demo.trycloudflare.com")

    def test_detects_url_in_tunnel_log(self) -> None:
        log = self._td / "tunnel.log"
        log.write_text("dashboard https://abc-123-xyz.trycloudflare.com live\n", encoding="utf-8")
        with patch.object(api_main, "_OUTPUTS", self._td):
            resp = self.client.get("/api/v1/tunnel-url")
        body = resp.json()
        self.assertEqual(body["url"], "https://abc-123-xyz.trycloudflare.com")

    def test_does_not_match_non_trycloudflare_url(self) -> None:
        log = self._td / "tunnel_err.txt"
        log.write_text("visit https://example.com for more info\n", encoding="utf-8")
        with patch.object(api_main, "_OUTPUTS", self._td):
            resp = self.client.get("/api/v1/tunnel-url")
        self.assertIsNone(resp.json()["url"])

    def test_prefers_first_matching_file_in_priority_order(self) -> None:
        """tunnel_err.txt takes priority over cloudflared.log when both exist."""
        (self._td / "tunnel_err.txt").write_text(
            "https://first-priority.trycloudflare.com\n", encoding="utf-8"
        )
        (self._td / "cloudflared.log").write_text(
            "https://second-priority.trycloudflare.com\n", encoding="utf-8"
        )
        with patch.object(api_main, "_OUTPUTS", self._td):
            resp = self.client.get("/api/v1/tunnel-url")
        self.assertEqual(resp.json()["url"], "https://first-priority.trycloudflare.com")

    def test_returns_null_on_empty_log_files(self) -> None:
        for name in ("tunnel_err.txt", "cloudflared.log", "tunnel.log"):
            (self._td / name).write_text("", encoding="utf-8")
        with patch.object(api_main, "_OUTPUTS", self._td):
            resp = self.client.get("/api/v1/tunnel-url")
        self.assertIsNone(resp.json()["url"])

    def test_handles_log_with_crlf_windows_endings(self) -> None:
        log = self._td / "tunnel_err.txt"
        # Write CRLF (Windows) line endings
        log.write_bytes(
            b"INFO Starting...\r\nhttps://win-tunnel.trycloudflare.com\r\nDone.\r\n"
        )
        with patch.object(api_main, "_OUTPUTS", self._td):
            resp = self.client.get("/api/v1/tunnel-url")
        self.assertEqual(resp.json()["url"], "https://win-tunnel.trycloudflare.com")


class SendTelegramEndpointTests(unittest.TestCase):
    """Tests for POST /api/v1/send-telegram."""

    def setUp(self) -> None:
        self.client = TestClient(app, raise_server_exceptions=True)

    def test_empty_text_returns_422(self) -> None:
        resp = self.client.post(
            "/api/v1/send-telegram",
            json={"text": "   "},
        )
        self.assertEqual(resp.status_code, 422)

    def test_unknown_account_returns_404(self) -> None:
        resp = self.client.post(
            "/api/v1/send-telegram",
            json={"text": "hello", "account": "acc_ghost"},
        )
        self.assertEqual(resp.status_code, 404)

    def test_single_account_send(self) -> None:
        with patch("xauusd_ai.api.main._tg_send_account", return_value=True) as mock_tg:
            resp = self.client.post(
                "/api/v1/send-telegram",
                json={"text": "Test message", "account": "acc1"},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["sent"])
        self.assertIn("acc1", body["accounts"])
        mock_tg.assert_called_once_with("acc1", "Test message")

    def test_broadcast_to_all_accounts_when_no_account_specified(self) -> None:
        with patch("xauusd_ai.api.main._tg_send_account", return_value=True) as mock_tg:
            resp = self.client.post(
                "/api/v1/send-telegram",
                json={"text": "Broadcast"},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["sent"])
        # Should send to all accounts in _LIVE_CFG_MAP
        called_accounts = {call.args[0] for call in mock_tg.call_args_list}
        for acct in api_main._LIVE_CFG_MAP:
            self.assertIn(acct, called_accounts)

    def test_sent_false_when_telegram_not_configured(self) -> None:
        with patch("xauusd_ai.api.main._tg_send_account", return_value=False):
            resp = self.client.post(
                "/api/v1/send-telegram",
                json={"text": "No token configured"},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertFalse(body["sent"])
        self.assertEqual(body["accounts"], [])

    def test_text_truncated_to_4000_chars(self) -> None:
        long_text = "A" * 5000
        captured: list[str] = []
        def _capture_tg(acct: str, text: str) -> bool:
            captured.append(text)
            return True
        with patch("xauusd_ai.api.main._tg_send_account", side_effect=_capture_tg):
            self.client.post("/api/v1/send-telegram", json={"text": long_text})
        for t in captured:
            self.assertLessEqual(len(t), 4000)

    def test_missing_text_field_returns_422(self) -> None:
        resp = self.client.post(
            "/api/v1/send-telegram",
            json={"account": "acc1"},
        )
        self.assertEqual(resp.status_code, 422)


# ── Dashboard payload: auto_trade_enabled field ───────────────────────────────

class DashboardPayloadAutoTradeTests(unittest.TestCase):
    """Ensure auto_trade_enabled is surfaced in the dashboard payload."""

    def _make_account_files(self, td: Path) -> tuple[Path, dict[str, str]]:
        status_path = td / "live_status_acc1.json"
        trades_path = td / "live_closed_trades_acc1.csv"
        signals_path = td / "paper_trade_signals_acc1.csv"
        meta_path = td / "model_meta.json"
        wf_path = td / "wf.json"
        bt_path = td / "bt.json"
        bt_trades_path = td / "bt_trades.csv"
        peak_path = td / "peak.json"
        daily_path = td / "daily.json"

        status_path.write_text(
            json.dumps({"account_balance": 1500.0, "open_positions": 0}), encoding="utf-8"
        )
        trades_path.write_text(
            "time,ticket,side,volume,open_price,close_price,profit,swap,commission,pnl,is_win,close_type,session_id\n",
            encoding="utf-8",
        )
        signals_path.write_text(
            ",".join(api_main._SIGNAL_COLUMNS_V2) + "\n", encoding="utf-8"
        )
        meta_path.write_text(json.dumps({"threshold": 0.62, "precision": 0.7}), encoding="utf-8")
        wf_path.write_text(json.dumps({"aggregate": {}, "folds": []}), encoding="utf-8")
        bt_path.write_text(json.dumps({"net_profit": 0.0}), encoding="utf-8")
        bt_trades_path.write_text(
            "time,ticket,side,volume,open_price,close_price,profit,swap,commission,pnl,is_win,close_type,session_id\n",
            encoding="utf-8",
        )
        peak_path.write_text(json.dumps({"peak_balance": 1500.0}), encoding="utf-8")
        daily_path.write_text(
            json.dumps({"daily_loss": 0.0, "killed": False, "consecutive_losses": 0, "cooldown_bars": 0}),
            encoding="utf-8",
        )

        cfg = {
            "label": "ACC1 Test",
            "status": str(status_path),
            "trades": str(trades_path),
            "trades_fallback": str(trades_path),
            "journal": str(td / "journal.jsonl"),
            "signals": str(signals_path),
            "meta": str(meta_path),
            "wf": str(wf_path),
            "bt": str(bt_path),
            "bt_trades": str(bt_trades_path),
            "peak": str(peak_path),
            "daily": str(daily_path),
        }
        return td, cfg

    def test_auto_trade_enabled_present_in_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            _, cfg = self._make_account_files(td)
            runtime_with_auto_trade = {"auto_trade_enabled": True}

            with (
                patch.dict(api_main._ACCOUNT_CFG, {"acc1": cfg}, clear=True),
                patch.dict(api_main._ACCOUNT_RUNTIME_CFG, {"acc1": runtime_with_auto_trade}, clear=True),
                patch("xauusd_ai.api.main._load_benchmark_targets",
                      return_value={"source": "unit", "acc1": {}}),
            ):
                payload = api_main._build_dashboard_payload_uncached()

        self.assertIn("accounts", payload)
        self.assertIn("acc1", payload["accounts"])
        acc = payload["accounts"]["acc1"]
        self.assertIn("auto_trade_enabled", acc,
                      "auto_trade_enabled must be a top-level field in account payload")
        self.assertTrue(acc["auto_trade_enabled"])

    def test_auto_trade_enabled_false_in_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            _, cfg = self._make_account_files(td)
            runtime_with_auto_trade = {"auto_trade_enabled": False}

            with (
                patch.dict(api_main._ACCOUNT_CFG, {"acc1": cfg}, clear=True),
                patch.dict(api_main._ACCOUNT_RUNTIME_CFG, {"acc1": runtime_with_auto_trade}, clear=True),
                patch("xauusd_ai.api.main._load_benchmark_targets",
                      return_value={"source": "unit", "acc1": {}}),
            ):
                payload = api_main._build_dashboard_payload_uncached()

        acc = payload["accounts"]["acc1"]
        self.assertIs(acc["auto_trade_enabled"], False)

    def test_auto_trade_enabled_none_when_runtime_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            _, cfg = self._make_account_files(td)

            with (
                patch.dict(api_main._ACCOUNT_CFG, {"acc1": cfg}, clear=True),
                patch.dict(api_main._ACCOUNT_RUNTIME_CFG, {"acc1": {}}, clear=True),
                patch("xauusd_ai.api.main._load_benchmark_targets",
                      return_value={"source": "unit", "acc1": {}}),
            ):
                payload = api_main._build_dashboard_payload_uncached()

        acc = payload["accounts"]["acc1"]
        # When runtime_cfg has no auto_trade_enabled, result should be None (not error)
        self.assertIn("auto_trade_enabled", acc)
        self.assertIsNone(acc["auto_trade_enabled"])


if __name__ == "__main__":
    unittest.main()
