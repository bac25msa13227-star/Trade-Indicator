from __future__ import annotations

import tempfile
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
        return None


class LiveLoopNewsOverrideTests(unittest.TestCase):
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
        settings.strategy.force_trade = False
        settings.risk.reentry_guard_enabled = False
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

    def test_news_override_forces_medium_plus_high_filter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            account_tag = "news_override_unit"
            self._cleanup_output_state(account_tag)

            settings = self._make_settings(temp_dir, account_tag)
            settings.integrations.news.enabled = True
            settings.integrations.news.high_impact_only = True
            settings.integrations.news.news_trade_override = True
            settings.integrations.news.minutes_before = 2
            settings.integrations.news.minutes_after = 5

            now_ts = pd.Timestamp("2026-03-30T02:00:00Z")
            frames = self._make_frames(now_ts)
            live_frame = self._make_live_frame(now_ts)

            data_service = MagicMock()
            data_service.fetch_multi_timeframe_data.return_value = frames

            trainer = MagicMock()
            trainer.load_artifacts.return_value = True
            trainer.score_live_row.return_value = {"probability": 0.89, "side": "buy"}

            strategy = MagicMock()
            strategy.build_trade_decision.return_value = TradeDecision(
                should_trade=True,
                side="buy",
                confidence=0.89,
                reason="unit_test_news_override",
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
            executor.place_order.return_value = {"position": 778899}

            risk_manager = _StubRiskManager()
            self_learner = MagicMock()
            self_learner.LOSS_RETRAIN_THRESHOLD = 3
            self_learner._accumulated_losses = 0

            news_crawler = MagicMock()
            news_crawler.is_near_news.return_value = (False, "clear", False)
            news_crawler.get_news_direction.return_value = (0, "", "")
            news_crawler._load_or_fetch.return_value = None

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
                "xauusd_ai.data.news_crawler.NewsCrawler",
                return_value=news_crawler,
            ), patch(
                "xauusd_ai.orchestrator.time.sleep",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_live_loop(settings)

            self.assertTrue(news_crawler.is_near_news.called)
            self.assertTrue(news_crawler.get_news_direction.called)
            self.assertFalse(news_crawler.is_near_news.call_args.kwargs["high_impact_only"])
            self.assertFalse(news_crawler.get_news_direction.call_args.kwargs["high_impact_only"])

            self._cleanup_output_state(account_tag)


if __name__ == "__main__":
    unittest.main()
