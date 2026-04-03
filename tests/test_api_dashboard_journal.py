from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xauusd_ai.api import main as api_main


class ApiDashboardJournalTests(unittest.TestCase):
    def test_safe_jsonl_tail_reads_last_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "journal.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"ticket": 1, "result": "WIN"}),
                        "not-json",
                        json.dumps({"ticket": 2, "result": "LOSS"}),
                        json.dumps({"ticket": 3, "result": "BREAKEVEN"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = api_main._safe_jsonl_tail(path, n=2)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["ticket"], 2)
            self.assertEqual(rows[1]["ticket"], 3)

    def test_dashboard_payload_includes_recent_journal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            status_path = base / "status.json"
            trades_path = base / "live_closed_trades.csv"
            signals_path = base / "paper_trade_signals.csv"
            meta_path = base / "model_meta.json"
            wf_path = base / "wf.json"
            bt_path = base / "bt.json"
            bt_trades_path = base / "bt_trades.csv"
            peak_path = base / "peak.json"
            daily_path = base / "daily.json"
            journal_path = base / "trade_journal.jsonl"

            status_path.write_text(
                json.dumps(
                    {
                        "account_balance": 1000.0,
                        "open_positions": 0,
                        "max_positions": 1,
                        "confidence": 0.8,
                        "should_trade": False,
                        "side": "buy",
                    }
                ),
                encoding="utf-8",
            )
            meta_path.write_text(json.dumps({"threshold": 0.62, "precision": 0.7}), encoding="utf-8")
            wf_path.write_text(json.dumps({"aggregate": {}, "folds": []}), encoding="utf-8")
            bt_path.write_text(json.dumps({"net_profit": 0.0}), encoding="utf-8")
            peak_path.write_text(json.dumps({"peak_balance": 1000.0}), encoding="utf-8")
            daily_path.write_text(json.dumps({"daily_loss": 0.0}), encoding="utf-8")

            trades_path.write_text(
                "time,ticket,side,volume,open_price,close_price,profit,swap,commission,pnl,is_win,close_type,session_id\n"
                "2026-04-03T09:30:00+00:00,778899,buy,0.01,2300,2310,10,0,0,10,true,TP,test\n",
                encoding="utf-8",
            )
            signal_row = {col: "" for col in api_main._SIGNAL_COLUMNS_V2}
            signal_row.update(
                {
                    "time": "2026-04-03T09:20:00+00:00",
                    "should_trade": "true",
                    "side": "buy",
                    "direction": "buy",
                    "confidence": "0.81",
                    "volatility_regime": "1",
                    "strategy_score": "0.52",
                }
            )
            signals_path.write_text(
                ",".join(api_main._SIGNAL_COLUMNS_V2)
                + "\n"
                + ",".join(str(signal_row.get(col, "")) for col in api_main._SIGNAL_COLUMNS_V2)
                + "\n",
                encoding="utf-8",
            )
            bt_trades_path.write_text(
                "time,ticket,side,volume,open_price,close_price,profit,swap,commission,pnl,is_win,close_type,session_id\n",
                encoding="utf-8",
            )
            journal_path.write_text(
                json.dumps(
                    {
                        "time": "2026-04-03T02:00:00+00:00",
                        "ticket": 778899,
                        "side": "buy",
                        "result": "WIN",
                        "pnl": 12.5,
                        "lesson": "Keep this setup profile.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = {
                "acc1": {
                    "label": "ACC1 Test",
                    "status": str(status_path),
                    "trades": str(trades_path),
                    "trades_fallback": str(trades_path),
                    "journal": str(journal_path),
                    "signals": str(signals_path),
                    "meta": str(meta_path),
                    "wf": str(wf_path),
                    "bt": str(bt_path),
                    "bt_trades": str(bt_trades_path),
                    "peak": str(peak_path),
                    "daily": str(daily_path),
                }
            }

            with patch.dict(api_main._ACCOUNT_CFG, cfg, clear=True), patch.dict(
                api_main._ACCOUNT_RUNTIME_CFG,
                {"acc1": {}},
                clear=True,
            ), patch(
                "xauusd_ai.api.main._load_benchmark_targets",
                return_value={"source": "unit", "acc1": {}},
            ):
                payload = api_main._build_dashboard_payload_uncached()

            self.assertIn("accounts", payload)
            self.assertIn("acc1", payload["accounts"])
            recent = payload["accounts"]["acc1"].get("recent_journal", [])
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0]["ticket"], 778899)
            self.assertIn("pnl_explain", payload["accounts"]["acc1"])
            self.assertIn("profile_presets", payload["accounts"]["acc1"])
            self.assertEqual(
                payload["accounts"]["acc1"]["pnl_explain"]["side"][0]["bucket"],
                "buy",
            )

    def test_build_pnl_explain_groups_by_session_regime_and_side(self) -> None:
        trades = [
            {"time": "2026-04-01T01:15:00+00:00", "side": "buy", "pnl": 15.0},
            {"time": "2026-04-01T08:10:00+00:00", "side": "sell", "pnl": -8.0},
            {"time": "2026-04-01T14:40:00+00:00", "side": "buy", "pnl": 12.0},
        ]
        signals = [
            {"time": "2026-04-01T01:00:00+00:00", "side": "buy", "volatility_regime": 0},
            {"time": "2026-04-01T08:00:00+00:00", "side": "sell", "volatility_regime": 1},
            {"time": "2026-04-01T14:20:00+00:00", "side": "buy", "volatility_regime": 2},
        ]
        explain = api_main._build_pnl_explain(trades, signals)
        self.assertEqual(len(explain["session"]), 3)
        self.assertEqual(len(explain["regime"]), 3)
        self.assertEqual(len(explain["side"]), 2)
        side_map = {row["bucket"]: row for row in explain["side"]}
        self.assertEqual(side_map["buy"]["trades"], 2)
        self.assertAlmostEqual(side_map["buy"]["net_pnl"], 27.0, places=2)
        self.assertEqual(side_map["sell"]["trades"], 1)

    def test_apply_runtime_profile_writes_override_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            override_path = Path(td) / "runtime_profile_override_acc1.json"
            req = api_main.ApplyProfileRequest(
                account="acc1",
                profile="aggressive",
                source="unit_test",
                requested_by="tester",
            )
            with patch(
                "xauusd_ai.api.main._runtime_profile_override_path",
                return_value=override_path,
            ), patch(
                "xauusd_ai.api.main._tg_send_account",
                return_value=True,
            ):
                result = asyncio.run(api_main.apply_runtime_profile(req))

            self.assertEqual(result["status"], "accepted")
            self.assertEqual(result["account"], "acc1")
            self.assertEqual(result["profile"], "aggressive")
            self.assertTrue(result["telegram_sent"])
            self.assertTrue(override_path.exists())
            payload = json.loads(override_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["account"], "acc1")
            self.assertEqual(payload["profile"], "aggressive")
            self.assertEqual(payload["source"], "unit_test")


if __name__ == "__main__":
    unittest.main()
