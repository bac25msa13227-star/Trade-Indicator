"""
SelfLearner — Hệ thống tự học liên tục.

Vòng lặp:
  1. Mỗi N nến mới → fetch data bổ sung từ yfinance (hoặc CSV/MT5)
  2. Gộp với data lịch sử → retrain toàn bộ model
  3. Đánh giá model mới vs model cũ (precision, recall, roc_auc)
  4. Nếu model tốt hơn (hoặc chưa có model cũ) → lưu
  5. Ghi log sự kiện → dashboard đọc được
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

if TYPE_CHECKING:
    from xauusd_ai.config import Settings
    from xauusd_ai.features.dataset import prepare_training_dataset  # noqa: F401
    from xauusd_ai.model.trainer import ModelTrainer
    from xauusd_ai.strategies.hybrid import HybridStrategy

LOGGER = logging.getLogger(__name__)


class SelfLearner:
    """
    Quản lý vòng học liên tục:
    - Fetch data mới từ yfinance / CSV / MT5
    - Gộp với data lịch sử trong bộ nhớ
    - Retrain model
    - So sánh kết quả và chỉ lưu khi tốt hơn
    """

    def __init__(
        self,
        settings: "Settings",
        trainer: "ModelTrainer",
        strategy: "HybridStrategy",
    ) -> None:
        self.settings = settings
        self.trainer = trainer
        self.strategy = strategy
        self._best_roc_auc: float = 0.0
        self._retrain_count: int = 0
        self._log_path = Path(settings.app.live_learning_log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Cache data đã tải
        self._cached_frames: dict[str, pd.DataFrame] = {}
        self._last_fetch_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def maybe_retrain(self, frames: dict[str, pd.DataFrame]) -> dict | None:
        """
        Gọi từ live loop sau mỗi batch nến mới.
        Trả về metrics dict hoặc None nếu không cần retrain.
        """
        from xauusd_ai.features.dataset import prepare_training_dataset

        try:
            # Gộp frames mới vào cache
            frames = self._merge_with_cache(frames)
            dataset = prepare_training_dataset(self.settings, frames, self.strategy)
            if len(dataset) < self.settings.training.live_learning_min_rows:
                LOGGER.info(
                    "SelfLearner: dataset quá nhỏ (%d rows), bỏ qua", len(dataset)
                )
                return None

            # Train và đánh giá
            metrics = self.trainer.train(dataset)
            self._retrain_count += 1

            roc_auc = float(metrics.get("roc_auc", 0.0))
            improved = roc_auc >= self._best_roc_auc

            if improved:
                self._best_roc_auc = roc_auc
                # model đã được lưu bên trong trainer.train() → trainer._save_artifacts()
                status = "improved"
            else:
                status = "no_improvement"

            event = {
                "event": "self_learn",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "retrain_count": self._retrain_count,
                "dataset_rows": int(len(dataset)),
                "roc_auc": round(roc_auc, 4),
                "best_roc_auc": round(self._best_roc_auc, 4),
                "precision": round(float(metrics.get("precision", 0)), 4),
                "recall": round(float(metrics.get("recall", 0)), 4),
                "f1": round(float(metrics.get("f1", 0)), 4),
                "status": status,
            }
            self._log_event(event)
            LOGGER.info("SelfLearner: %s | roc_auc=%.4f | rows=%d", status, roc_auc, len(dataset))
            return event

        except Exception as exc:
            LOGGER.error("SelfLearner.maybe_retrain error: %s", exc, exc_info=True)
            return None

    def fetch_fresh_yfinance(self, timeframe: str = "M15") -> pd.DataFrame:
        """
        Cào data mới nhất từ yfinance (XAUUSD=X).
        Dùng làm bổ sung khi data CSV/MT5 chưa cập nhật.
        """
        from xauusd_ai.data.market_data import TIMEFRAME_MAP, YFINANCE_PERIOD_MAP

        interval = TIMEFRAME_MAP.get(timeframe, "15m")
        period = YFINANCE_PERIOD_MAP.get(timeframe, "60d")
        tickers = [self.settings.market.training_symbol or "GC=F", "GC=F", "GLD"]

        for ticker in tickers:
            try:
                history = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
                if history is not None and not history.empty:
                    frame = history.reset_index().rename(
                        columns={
                            "Datetime": "time",
                            "Date": "time",
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "tick_volume",
                        }
                    )
                    frame["time"] = pd.to_datetime(frame["time"], utc=True)
                    frame["spread_points"] = 0.0
                    frame["tick_volume_delta"] = frame["tick_volume"].diff().fillna(0)
                    frame["volume_imbalance"] = (
                        (frame["close"] - frame["open"]).abs()
                        / (frame["high"] - frame["low"]).replace(0, pd.NA)
                    ).fillna(0)
                    frame = frame.sort_values("time").reset_index(drop=True)
                    LOGGER.info(
                        "SelfLearner: fetched %d rows from yfinance (%s %s)",
                        len(frame), ticker, timeframe,
                    )
                    return frame
            except Exception as e:
                LOGGER.warning("yfinance fetch failed (%s): %s", ticker, e)

        LOGGER.warning("SelfLearner: không lấy được data từ yfinance")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _merge_with_cache(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Gộp frames hiện tại với cache.
        Nếu đã quá 1 giờ kể từ lần fetch cuối → cào thêm từ yfinance.
        """
        now = time.time()
        merged: dict[str, pd.DataFrame] = {}

        for tf, frame in frames.items():
            if tf in self._cached_frames and not self._cached_frames[tf].empty:
                combined = pd.concat(
                    [self._cached_frames[tf], frame], ignore_index=True
                )
                combined = (
                    combined.drop_duplicates("time")
                    .sort_values("time")
                    .reset_index(drop=True)
                )
                merged[tf] = combined
            else:
                merged[tf] = frame
            self._cached_frames[tf] = merged[tf]

        # Mỗi 1 giờ cào thêm từ yfinance để bổ sung data mới nhất
        exec_tf = self.settings.market.execution_timeframe
        if now - self._last_fetch_ts > 3600:
            yf_frame = self.fetch_fresh_yfinance(exec_tf)
            if not yf_frame.empty and exec_tf in merged:
                combined = pd.concat([merged[exec_tf], yf_frame], ignore_index=True)
                combined = (
                    combined.drop_duplicates("time")
                    .sort_values("time")
                    .reset_index(drop=True)
                )
                merged[exec_tf] = combined
                self._cached_frames[exec_tf] = combined
            self._last_fetch_ts = now

        return merged

    def _log_event(self, event: dict) -> None:
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")

    # ------------------------------------------------------------------
    # Summary for dashboard
    # ------------------------------------------------------------------
    def get_summary(self) -> dict:
        """Trả về tóm tắt trạng thái tự học để dashboard hiển thị."""
        events: list[dict] = []
        if self._log_path.exists():
            for line in self._log_path.read_text(encoding="utf-8").splitlines():
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass

        learn_events = [e for e in events if e.get("event") == "self_learn"]

        return {
            "retrain_count": self._retrain_count,
            "best_roc_auc": round(self._best_roc_auc, 4),
            "last_retrain": learn_events[-1]["timestamp"] if learn_events else None,
            "total_events": len(events),
            "recent_events": learn_events[-5:],
        }
