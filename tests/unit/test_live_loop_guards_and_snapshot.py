from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.execution.risk import OrderPlan
from xauusd_ai.orchestrator import run_live_loop
from xauusd_ai.strategies.hybrid import TradeDecision


class _StubRiskManager:
    def __init__(self) -> None:
        self._peak_balance = 0.0
        self.recorded_results: list[tuple[float, float]] = []

    def tick_cooldown(self) -> None:
        return None

    def can_open_position(
        self,
        balance: float,
        current_open_positions: int,
        volatility_regime: int = 1,
    ) -> tuple[bool, str]:
        return True, "ok"

    def get_dynamic_max_positions(self, balance: float, volatility_regime: int = 1) -> int:
        return 3

    def build_order_plan(
        self,
        decision: TradeDecision,
        latest_bar: pd.Series,
        account_balance: float | None = None,
        current_open_positions: int = 0,
        volatility_regime: int = 1,
    ) -> OrderPlan:
        return OrderPlan(
            symbol="XAUUSD",
            side=decision.side,
            volume=0.01,
            entry_price=float(decision.entry_price),
            stop_loss=float(decision.stop_loss),
            take_profit=float(decision.take_profit),
            confidence=float(decision.confidence),
            reason=str(decision.reason),
        )

    def record_trade_result(self, pnl: float, balance: float) -> None:
        self.recorded_results.append((float(pnl), float(balance)))

    @staticmethod
    def order_risk_amount(volume: float, entry_price: float, stop_loss: float) -> float:
        return abs(float(entry_price) - float(stop_loss)) * 100.0 * float(volume)

    def check_total_exposure(
        self, balance: float, current_risk_amount: float, existing_risk_total: float
    ) -> tuple[bool, str]:
        return True, "ok"

    def is_circuit_breaker_active(self, balance: float) -> tuple[bool, str]:
        return False, "ok"


class LiveLoopGuardAndSnapshotTests(unittest.TestCase):
    def _make_settings(self, temp_dir: Path, account_tag: str) -> Settings:
        settings = Settings()
        settings.app.poll_seconds = 1
        settings.app.log_level = "CRITICAL"
        settings.app.paper_trade_log_path = str(temp_dir / f"paper_trade_signals_{account_tag}.csv")
        settings.app.live_closed_trades_path = str(temp_dir / f"live_closed_trades_{account_tag}.csv")
        settings.app.live_learning_log_path = str(temp_dir / f"live_learning_{account_tag}.jsonl")
        settings.market.execution_timeframe = "M5"
        settings.training.retrain_on_startup = False
        settings.training.live_learning_enabled = False
        settings.execution.auto_trade = True
        settings.execution.close_opposite_on_signal = False
        settings.execution.dca.enabled = False
        settings.execution.trailing_sl.enabled = False
        settings.risk.partial_tp_enabled = False
        settings.integrations.news.enabled = False
        settings.strategy.force_trade = False
        return settings

    @staticmethod
    def _make_frames(now_ts: pd.Timestamp) -> dict[str, pd.DataFrame]:
        execution_frame = pd.DataFrame(
            {
                "time": [now_ts],
                "open": [2300.0],
                "high": [2302.0],
                "low": [2298.0],
                "close": [2301.0],
            }
        )
        return {"M5": execution_frame}

    @staticmethod
    def _make_live_frame(now_ts: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": [now_ts],
                "open": [2300.0],
                "high": [2302.0],
                "low": [2298.0],
                "close": [2301.0],
                "atr": [10.0],
                "rsi": [56.0],
                "macd_hist": [0.2],
                "strategy_score": [0.8],
                "volatility_regime": [1],
                "trend_alignment": [1],
                "hourly_bias": [0.2],
                "daily_bias": [0.3],
                "liquidity_sweep": [0],
                "wyckoff_phase": [0],
                "order_flow_proxy": [0.1],
                "adx": [24.0],
                "trend_strength_score": [0.4],
                "pullback_quality": [0.2],
                "execution_quality": [0.3],
            }
        )

    @staticmethod
    def _cleanup_output_state(account_tag: str) -> None:
        output_dir = Path("outputs")
        for prefix in (
            "entry_rsi_tracker_",
            "entry_snapshot_tracker_",
            "reentry_guard_",
            "risk_peak_balance_",
            "risk_daily_state_",
        ):
            path = output_dir / f"{prefix}{account_tag}.json"
            if path.exists():
                path.unlink()

    def test_reentry_guard_blocks_same_side_trade_after_sl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            account_tag = "guard_case_unit"
            self._cleanup_output_state(account_tag)

            settings = self._make_settings(temp_dir, account_tag)
            settings.risk.reentry_guard_enabled = True
            settings.risk.reentry_cooldown_bars_after_sl = 1
            settings.risk.reentry_min_distance_atr = 0.35

            now_ts = pd.Timestamp("2026-03-30T00:00:00Z")
            frames = self._make_frames(now_ts)
            live_frame = self._make_live_frame(now_ts)

            data_service = MagicMock()
            data_service.fetch_multi_timeframe_data.return_value = frames

            trainer = MagicMock()
            trainer.load_artifacts.return_value = True
            trainer.score_live_row.return_value = {"probability": 0.91, "side": "buy"}

            strategy = MagicMock()
            strategy.build_trade_decision.return_value = TradeDecision(
                should_trade=True,
                side="buy",
                confidence=0.91,
                reason="unit_test_decision",
                entry_price=2301.0,
                stop_loss=2295.0,
                take_profit=2313.0,
            )

            notifier = MagicMock()

            closed_sl = {
                "ticket": 900001,
                "side": "buy",
                "volume": 0.01,
                "open_price": 2305.0,
                "close_price": 2301.0,
                "profit": 0.0,
                "swap": 0.0,
                "commission": 0.0,
                "close_time": float(time.time()),
                "reason": 4,
            }

            executor = MagicMock()
            executor.get_account_info.return_value = {"balance": 1000.0}
            executor.get_open_positions_count.return_value = 0
            executor.get_recently_closed_positions.return_value = [closed_sl]
            executor.get_open_positions.return_value = []
            executor.get_current_price.return_value = 2301.0
            executor.get_market_state.return_value = {
                "is_open": True,
                "reason": "OPEN",
                "next_open_utc": None,
                "next_close_utc": None,
                "minutes_to_next_open": None,
                "minutes_to_next_close": None,
                "tick_age_sec": 1.0,
            }
            executor.place_order.return_value = {"position": 123456}

            risk_manager = _StubRiskManager()
            self_learner = MagicMock()
            self_learner.LOSS_RETRAIN_THRESHOLD = 3
            self_learner._accumulated_losses = 0

            with patch(
                "xauusd_ai.orchestrator._bootstrap_with_learner",
                return_value=(
                    data_service,
                    trainer,
                    strategy,
                    notifier,
                    executor,
                    risk_manager,
                    self_learner,
                ),
            ), patch(
                "xauusd_ai.orchestrator.build_live_feature_frame",
                return_value=live_frame,
            ), patch(
                "xauusd_ai.orchestrator.time.sleep",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_live_loop(settings)

            executor.place_order.assert_not_called()
            notifier.send_signal.assert_not_called()

            signals = pd.read_csv(settings.app.paper_trade_log_path)
            self.assertEqual(len(signals), 1)
            self.assertFalse(bool(signals.iloc[0]["should_trade"]))
            self.assertIn("REENTRY_GUARD", str(signals.iloc[0]["reason"]))

            self._cleanup_output_state(account_tag)

    def test_entry_snapshot_is_saved_when_order_is_placed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            account_tag = "snapshot_case_unit"
            self._cleanup_output_state(account_tag)

            settings = self._make_settings(temp_dir, account_tag)
            settings.risk.reentry_guard_enabled = False

            now_ts = pd.Timestamp("2026-03-30T01:00:00Z")
            frames = self._make_frames(now_ts)
            live_frame = self._make_live_frame(now_ts)

            data_service = MagicMock()
            data_service.fetch_multi_timeframe_data.return_value = frames

            trainer = MagicMock()
            trainer.load_artifacts.return_value = True
            trainer.score_live_row.return_value = {"probability": 0.88, "side": "buy"}

            strategy = MagicMock()
            strategy.build_trade_decision.return_value = TradeDecision(
                should_trade=True,
                side="buy",
                confidence=0.88,
                reason="unit_test_place_order",
                entry_price=2301.0,
                stop_loss=2294.0,
                take_profit=2315.0,
            )

            notifier = MagicMock()
            executor = MagicMock()
            executor.get_account_info.return_value = {"balance": 1000.0}
            executor.get_open_positions_count.return_value = 0
            executor.get_recently_closed_positions.return_value = []
            executor.get_open_positions.return_value = []
            executor.get_current_price.return_value = None
            executor.get_market_state.return_value = {
                "is_open": True,
                "reason": "OPEN",
                "next_open_utc": None,
                "next_close_utc": None,
                "minutes_to_next_open": None,
                "minutes_to_next_close": None,
                "tick_age_sec": 1.0,
            }
            executor.place_order.return_value = {"position": 556677}

            risk_manager = _StubRiskManager()
            self_learner = MagicMock()
            self_learner.LOSS_RETRAIN_THRESHOLD = 3
            self_learner._accumulated_losses = 0

            with patch(
                "xauusd_ai.orchestrator._bootstrap_with_learner",
                return_value=(
                    data_service,
                    trainer,
                    strategy,
                    notifier,
                    executor,
                    risk_manager,
                    self_learner,
                ),
            ), patch(
                "xauusd_ai.orchestrator.build_live_feature_frame",
                return_value=live_frame,
            ), patch(
                "xauusd_ai.orchestrator.time.sleep",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_live_loop(settings)

            executor.place_order.assert_called_once()

            snapshot_file = Path("outputs") / f"entry_snapshot_tracker_{account_tag}.json"
            self.assertTrue(snapshot_file.exists())
            payload = json.loads(snapshot_file.read_text(encoding="utf-8"))
            self.assertIn("556677", payload)
            self.assertEqual(payload["556677"]["side"], "buy")
            self.assertAlmostEqual(float(payload["556677"]["rsi"]), 56.0, places=6)

            self._cleanup_output_state(account_tag)

    def test_auto_journal_is_written_and_notified_for_closed_loss(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            account_tag = "journal_case_unit"
            self._cleanup_output_state(account_tag)

            settings = self._make_settings(temp_dir, account_tag)
            settings.risk.reentry_guard_enabled = False

            now_ts = pd.Timestamp("2026-03-30T02:00:00Z")
            frames = self._make_frames(now_ts)
            live_frame = self._make_live_frame(now_ts)

            data_service = MagicMock()
            data_service.fetch_multi_timeframe_data.return_value = frames

            trainer = MagicMock()
            trainer.load_artifacts.return_value = True
            trainer.score_live_row.return_value = {"probability": 0.34, "side": "sell"}

            strategy = MagicMock()
            strategy.build_trade_decision.return_value = TradeDecision(
                should_trade=False,
                side="sell",
                confidence=0.34,
                reason="unit_test_no_entry",
                entry_price=2301.0,
                stop_loss=2308.0,
                take_profit=2290.0,
            )

            notifier = MagicMock()

            closed_loss = {
                "ticket": 900777,
                "side": "buy",
                "volume": 0.01,
                "open_price": 2305.0,
                "close_price": 2296.0,
                "profit": -9.0,
                "swap": 0.0,
                "commission": 0.0,
                "close_time": float(time.time()),
                "reason": 4,
            }

            executor = MagicMock()
            executor.get_account_info.return_value = {"balance": 1000.0}
            executor.get_open_positions_count.return_value = 0
            _closed_seq = iter([[], [closed_loss], []])
            executor.get_recently_closed_positions.side_effect = lambda *args, **kwargs: next(_closed_seq, [])
            executor.get_open_positions.return_value = []
            executor.get_current_price.return_value = None
            executor.get_market_state.return_value = {
                "is_open": True,
                "reason": "OPEN",
                "next_open_utc": None,
                "next_close_utc": None,
                "minutes_to_next_open": None,
                "minutes_to_next_close": None,
                "tick_age_sec": 1.0,
            }
            executor.place_order.return_value = {"position": 111222}

            risk_manager = _StubRiskManager()
            self_learner = MagicMock()
            self_learner.LOSS_RETRAIN_THRESHOLD = 3
            self_learner._accumulated_losses = 0
            self_learner.log_loss_analysis.return_value = {
                "features": {
                    "atr": 10.1,
                    "rsi": 61.0,
                    "strategy_score": -0.22,
                    "trend_alignment": 0,
                    "volatility_regime": 0,
                    "ict_score": -0.4,
                    "wyckoff_score": -0.2,
                    "momentum_score": -0.3,
                },
                "reasons": ["TREND_MISALIGNED", "REGIME_SIDEWAY"],
            }

            snapshot_file = Path("outputs") / f"entry_snapshot_tracker_{account_tag}.json"
            snapshot_file.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file.write_text(
                json.dumps(
                    {
                        "900777": {
                            "time": "2026-03-30T01:30:00+00:00",
                            "side": "buy",
                            "confidence": 0.79,
                            "reason": "Trend follow buy near pullback",
                            "entry_price": 2305.0,
                            "stop_loss": 2296.0,
                            "take_profit": 2323.0,
                            "strategy_score": 0.41,
                            "ict_score": 0.5,
                            "wyckoff_score": 0.2,
                            "momentum_score": 0.4,
                            "trend_alignment": 1,
                            "volatility_regime": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "xauusd_ai.orchestrator._bootstrap_with_learner",
                return_value=(
                    data_service,
                    trainer,
                    strategy,
                    notifier,
                    executor,
                    risk_manager,
                    self_learner,
                ),
            ), patch(
                "xauusd_ai.orchestrator.build_live_feature_frame",
                return_value=live_frame,
            ), patch(
                "xauusd_ai.orchestrator.time.sleep",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_live_loop(settings)

            journal_path = temp_dir / f"trade_journal_{account_tag}.jsonl"
            self.assertTrue(journal_path.exists())
            journal_lines = [line for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(journal_lines), 1)
            payload = json.loads(journal_lines[0])
            self.assertEqual(payload.get("ticket"), 900777)
            self.assertEqual(payload.get("result"), "LOSS")
            self.assertEqual(payload.get("close_type"), "SL")
            self.assertTrue(str(payload.get("lesson", "")).strip())

            self.assertTrue(
                any(
                    "Auto Journal" in str(call.args[0])
                    for call in notifier.send_message.call_args_list
                    if call.args
                )
            )

            self._cleanup_output_state(account_tag)


if __name__ == "__main__":
    unittest.main()
