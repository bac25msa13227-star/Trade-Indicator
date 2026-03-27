"""
SelfLearner — Hệ thống tự học liên tục.

Vòng lặp:
  1. Mỗi N nến mới → fetch data bổ sung từ yfinance (hoặc CSV/MT5)
  2. Gộp với data lịch sử → retrain toàn bộ model
  3. Đánh giá model mới vs model cũ (precision, recall, roc_auc)
  4. Nếu model tốt hơn (hoặc chưa có model cũ) → lưu
  5. Ghi log sự kiện → dashboard đọc được

Loss Learning (Phase 2):
  - Mỗi khi phát hiện lệnh thua → phân tích nguyên nhân
  - Log chi tiết features lúc vào lệnh vào outputs/loss_analysis.jsonl
  - Sau mỗi K lệnh thua → trigger retrain với sample_weight tăng
    cho các pattern tương tự → model học cẩn thận hơn
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
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
        # Reload best_roc_auc from saved model meta (survives restarts)
        _meta_path = Path(settings.app.model_meta_path)
        if _meta_path.exists():
            try:
                _meta = json.loads(_meta_path.read_text(encoding="utf-8"))
                self._best_roc_auc = float(_meta.get("roc_auc", 0.0))
                LOGGER.info("SelfLearner: loaded best_roc_auc=%.4f from meta", self._best_roc_auc)
            except Exception:
                pass
        self._retrain_count: int = 0
        self._log_path = Path(settings.app.live_learning_log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._loss_log_path = Path("outputs/loss_analysis.jsonl")
        self._loss_log_path.parent.mkdir(parents=True, exist_ok=True)
        # Cache data đã tải
        self._cached_frames: dict[str, pd.DataFrame] = {}
        self._last_fetch_ts: float = 0.0
        # Trạng thái loss learning
        self._accumulated_losses: int = 0
        self._loss_patterns: list[dict] = []  # features của các lệnh thua gần đây
        # Pre-load historical CSV để live-learning có đủ dataset ngay từ đầu
        self._preload_csv_cache()

    @staticmethod
    def _optional_float(value: object) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _load_saved_metrics(self) -> dict[str, float | None]:
        meta_path = Path(self.settings.app.model_meta_path)
        if not meta_path.exists():
            return {}
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return {
            "roc_auc": self._optional_float(meta.get("roc_auc")),
            "precision": self._optional_float(meta.get("precision")),
            "recall": self._optional_float(meta.get("recall")),
            "f1": self._optional_float(meta.get("f1")),
            "decision_threshold": self._optional_float(meta.get("decision_threshold")),
        }

    @staticmethod
    def _metric_delta(current: float, baseline: float | None) -> float | None:
        if baseline is None:
            return None
        return round(float(current) - float(baseline), 4)

    def _recent_loss_focus(self, limit: int = 3) -> list[str]:
        if not self._loss_log_path.exists():
            return []
        try:
            lines = self._loss_log_path.read_text(encoding="utf-8").splitlines()[-50:]
        except Exception:
            return []
        prefixes: list[str] = []
        for line in lines:
            try:
                event = json.loads(line)
            except Exception:
                continue
            for reason in event.get("reasons", []):
                prefix = str(reason).split(":", 1)[0].strip()
                if prefix:
                    prefixes.append(prefix)
        return [name for name, _ in Counter(prefixes).most_common(limit)]

    def _describe_learning(
        self,
        *,
        event_name: str,
        accepted: bool,
        roc_auc_delta: float | None,
        precision_delta: float | None,
        recall_delta: float | None,
        focus_reasons: list[str] | None = None,
    ) -> str:
        notes: list[str] = []
        if event_name == "loss_retrain" and focus_reasons:
            notes.append("siết lại các mẫu lỗ: " + ", ".join(focus_reasons))
        if roc_auc_delta is not None:
            if roc_auc_delta >= 0.005:
                notes.append("khả năng tách tín hiệu tốt/xấu tăng")
            elif roc_auc_delta <= -0.005:
                notes.append("ứng viên tổng quát hóa kém hơn model đang chạy")
        if precision_delta is not None and precision_delta >= 0.01:
            notes.append("lọc tín hiệu nhiễu tốt hơn")
        if recall_delta is not None and recall_delta >= 0.01:
            notes.append("bắt thêm được setup hợp lệ")
        if not notes:
            if accepted:
                notes.append("cải thiện nhỏ nhưng đủ vượt ngưỡng chấp nhận")
            else:
                notes.append("ứng viên chưa vượt model hiện tại nên bị từ chối")
        return "; ".join(notes)

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
            # Dùng ratio-based split (80/20) thay vì date-based khi live-learning
            # để tránh lỗi khi test_end_date đã qua
            live_settings = self.settings.model_copy(deep=True)
            live_settings.training.train_start_date = None
            live_settings.training.train_end_date = None
            live_settings.training.test_start_date = None
            live_settings.training.test_end_date = None
            dataset = prepare_training_dataset(live_settings, frames, self.strategy)
            if len(dataset) < self.settings.training.live_learning_min_rows:
                LOGGER.info(
                    "SelfLearner: dataset too small (%d rows), skipping", len(dataset)
                )
                return None

            # Train (do NOT save inside trainer — we decide here based on improvement)
            baseline_metrics = self._load_saved_metrics()
            best_before = float(self._best_roc_auc)
            metrics = self.trainer.train(dataset, save_artifacts=False)
            self._retrain_count += 1

            roc_auc = float(metrics.get("roc_auc", 0.0))
            _min_floor = self.settings.training.min_self_learning_roc_auc
            acceptance_floor = max(best_before, _min_floor)
            improved = roc_auc >= acceptance_floor

            if improved:
                self._best_roc_auc = roc_auc
                self.trainer._last_roc_auc = roc_auc
                self.trainer._save_artifacts()  # Only save when truly better
                status = "improved"
            else:
                status = "no_improvement"

            precision = float(metrics.get("precision", 0.0))
            recall = float(metrics.get("recall", 0.0))
            f1 = float(metrics.get("f1", 0.0))
            selected_threshold = float(metrics.get("selected_threshold", self.trainer.decision_threshold))
            roc_auc_delta = self._metric_delta(roc_auc, baseline_metrics.get("roc_auc"))
            precision_delta = self._metric_delta(precision, baseline_metrics.get("precision"))
            recall_delta = self._metric_delta(recall, baseline_metrics.get("recall"))
            f1_delta = self._metric_delta(f1, baseline_metrics.get("f1"))

            event = {
                "event": "self_learn",
                "learning_kind": "scheduled",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "retrain_count": self._retrain_count,
                "dataset_rows": int(len(dataset)),
                "roc_auc": round(roc_auc, 4),
                "best_roc_auc_before": round(best_before, 4),
                "best_roc_auc_after": round(self._best_roc_auc, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "selected_threshold": round(selected_threshold, 4),
                "baseline_roc_auc": round(baseline_metrics["roc_auc"], 4) if baseline_metrics.get("roc_auc") is not None else None,
                "baseline_precision": round(baseline_metrics["precision"], 4) if baseline_metrics.get("precision") is not None else None,
                "baseline_recall": round(baseline_metrics["recall"], 4) if baseline_metrics.get("recall") is not None else None,
                "baseline_f1": round(baseline_metrics["f1"], 4) if baseline_metrics.get("f1") is not None else None,
                "roc_auc_delta": roc_auc_delta,
                "precision_delta": precision_delta,
                "recall_delta": recall_delta,
                "f1_delta": f1_delta,
                "acceptance_floor": round(acceptance_floor, 4),
                "accepted": improved,
                "status": status,
                "learning_summary": self._describe_learning(
                    event_name="self_learn",
                    accepted=improved,
                    roc_auc_delta=roc_auc_delta,
                    precision_delta=precision_delta,
                    recall_delta=recall_delta,
                ),
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
        _sym = (self.settings.market.training_symbol or "GC=F").strip()
        tickers = list(dict.fromkeys([_sym, "GC=F", "GLD"]))  # dedup, keep order

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

        LOGGER.warning("SelfLearner: could not fetch data from yfinance")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    # Maximum bars to keep per timeframe to prevent unbounded memory growth
    _MAX_CACHE_BARS = {"M1": 5_000, "M5": 12_000, "M15": 10_000, "H1": 5_000, "H4": 3_000, "D1": 2_000}

    def _merge_with_cache(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Gộp frames hiện tại với cache.
        Nếu đã quá 1 giờ kể từ lần fetch cuối → cào thêm từ yfinance.
        Tự động trim cache để tránh memory leak.
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
                # Trim to prevent unbounded growth
                max_bars = self._MAX_CACHE_BARS.get(tf, 5_000)
                if len(combined) > max_bars:
                    combined = combined.tail(max_bars).reset_index(drop=True)
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
                max_bars = self._MAX_CACHE_BARS.get(exec_tf, 10_000)
                if len(combined) > max_bars:
                    combined = combined.tail(max_bars).reset_index(drop=True)
                merged[exec_tf] = combined
                self._cached_frames[exec_tf] = combined
            self._last_fetch_ts = now

        return merged

    def _preload_csv_cache(self) -> None:
        """Pre-load historical CSV data vào cache để live-learning luôn có đủ dataset."""
        csv_path = Path(self.settings.market.csv_folder_path)
        if not csv_path.exists():
            LOGGER.warning("SelfLearner: csv_folder_path not found, skipping preload")
            return
        tf_file_map = {
            "M1": "XAUUSDm_M1.csv",
            "M5": "XAUUSDm_M5.csv",
            "M15": "XAUUSDm_M15.csv",
            "M30": "XAUUSDm_M30.csv",
            "H1": "XAUUSDm_H1.csv",
            "H4": "XAUUSDm_H4.csv",
            "D1": "XAUUSDm_D1.csv",
        }
        for tf, fname in tf_file_map.items():
            fpath = csv_path / fname
            if not fpath.exists():
                continue
            try:
                df = pd.read_csv(fpath)
                # Normalize column names (MT5 CSVs use Title case: Open/High/Low/Close)
                df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low", "Close": "close",
                    "Volume": "tick_volume", "TickVolume": "tick_volume",
                    "RealVolume": "real_volume", "Spread": "spread_points",
                }, inplace=True)
                if "time" not in df.columns and df.index.name:
                    df = df.rename_axis("time").reset_index()
                df["time"] = pd.to_datetime(df["time"], utc=True)
                # Chỉ lấy 8000 hàng gần nhất để cân bằng data đủ/retrain vừa phải
                df = df.sort_values("time").tail(8000).reset_index(drop=True)
                if "tick_volume_delta" not in df.columns:
                    tv = df["tick_volume"] if "tick_volume" in df.columns else pd.Series(0, index=df.index)
                    df["tick_volume_delta"] = tv.diff().fillna(0)
                if "volume_imbalance" not in df.columns:
                    rng = (df["high"] - df["low"]).replace(0, float("nan"))
                    df["volume_imbalance"] = (df["close"] - df["open"]).abs() / rng
                    df["volume_imbalance"] = df["volume_imbalance"].fillna(0)
                if "spread_points" not in df.columns:
                    df["spread_points"] = df["spread"] if "spread" in df.columns else 0
                self._cached_frames[tf] = df
                LOGGER.info("SelfLearner preload: %s -> %d rows", fname, len(df))
            except Exception as exc:
                LOGGER.warning("SelfLearner preload failed %s: %s", fname, exc)
        self._last_fetch_ts = time.time()  # skip immediate yfinance fetch

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

    # ------------------------------------------------------------------
    # Loss Learning — phân tích lệnh thua và học lại
    # ------------------------------------------------------------------

    # Số lệnh thua tích lũy trước khi trigger retrain (có thể ghi vào config sau)
    LOSS_RETRAIN_THRESHOLD = 3

    def analyze_loss(self, position: dict, live_row: "pd.Series") -> dict:
        """
        Phân tích tại sao lệnh bị thua.
        Trả về dict mô tả nguyên nhân và các features lúc vào lệnh.
        """
        reasons: list[str] = []

        side = position.get("side", "unknown")
        profit = float(position.get("profit", 0))
        swap = float(position.get("swap", 0))
        commission = float(position.get("commission", 0))
        net_pnl = profit + swap + commission

        # --- Feature snapshot lúc vào lệnh (lấy từ live_row hiện tại) ---
        strategy_score = float(getattr(live_row, "strategy_score", 0))
        volatility_regime = int(getattr(live_row, "volatility_regime", 1))
        rsi_val = float(getattr(live_row, "rsi", 50))
        macd_hist = float(getattr(live_row, "macd_hist", 0))
        atr_val = float(getattr(live_row, "atr", 0))
        trend_alignment = int(getattr(live_row, "trend_alignment", 0))
        hourly_bias = float(getattr(live_row, "hourly_bias", 0))
        daily_bias = float(getattr(live_row, "daily_bias", 0))
        liquidity_sweep = int(getattr(live_row, "liquidity_sweep", 0))
        wyckoff_phase = int(getattr(live_row, "wyckoff_phase", 0))
        order_flow_proxy = float(getattr(live_row, "order_flow_proxy", 0))

        # --- Phân tích nguyên nhân ---
        settings = self.settings

        # 1. Momentum ngược chiều
        if side == "buy" and macd_hist < 0:
            reasons.append("MACD_HIST_NEGATIVE: momentum bán dù vào BUY")
        if side == "sell" and macd_hist > 0:
            reasons.append("MACD_HIST_POSITIVE: momentum mua dù vào SELL")

        # 2. RSI vùng không thuận lợi
        if side == "buy" and rsi_val > 70:
            reasons.append(f"RSI_OVERBOUGHT: RSI={rsi_val:.1f} — mua ở đỉnh")
        if side == "sell" and rsi_val < 30:
            reasons.append(f"RSI_OVERSOLD: RSI={rsi_val:.1f} — bán ở đáy")

        # 3. Trend không hỗ trợ
        if side == "buy" and daily_bias < 0:
            reasons.append("DAILY_BIAS_BEARISH: daily bias âm dù vào BUY")
        if side == "sell" and daily_bias > 0:
            reasons.append("DAILY_BIAS_BULLISH: daily bias dương dù vào SELL")
        if trend_alignment != 1:
            reasons.append(f"TREND_MISALIGNED: trend_alignment={trend_alignment}")

        # 4. Volatility quá cao / quá thấp
        if volatility_regime == 0:
            reasons.append("REGIME_SIDEWAY: thị trường sideway — rủi ro whipsaw cao")
        if volatility_regime == 2:
            reasons.append("REGIME_STRONG_VOLATILE: volatility rất cao — spread rộng, SL dễ bị quét")

        # 5. Wyckoff không hỗ trợ
        if side == "buy" and wyckoff_phase == -1:
            reasons.append("WYCKOFF_DISTRIBUTION: Wyckoff phase phân phối dù vào BUY")
        if side == "sell" and wyckoff_phase == 1:
            reasons.append("WYCKOFF_ACCUMULATION: Wyckoff phase tích lũy dù vào SELL")

        # 6. Order flow không hỗ trợ
        if side == "buy" and order_flow_proxy < -0.3:
            reasons.append(f"ORDER_FLOW_SELL: order_flow_proxy={order_flow_proxy:.2f} nghiêng bán")
        if side == "sell" and order_flow_proxy > 0.3:
            reasons.append(f"ORDER_FLOW_BUY: order_flow_proxy={order_flow_proxy:.2f} nghiêng mua")

        # 7. Strategy score yếu
        req_score = settings.strategy.sideway_min_strategy_score if volatility_regime == 0 else settings.strategy.min_strategy_score
        if abs(strategy_score) < req_score * 1.5:
            reasons.append(f"WEAK_STRATEGY_SCORE: score={strategy_score:.3f} (ngưỡng={req_score:.2f}) — tín hiệu yếu")

        if not reasons:
            reasons.append("UNKNOWN: không xác định rõ nguyên nhân — có thể do thị trường đảo chiều đột ngột")

        feature_snapshot = {
            "strategy_score": round(strategy_score, 4),
            "volatility_regime": volatility_regime,
            "rsi": round(rsi_val, 2),
            "macd_hist": round(macd_hist, 6),
            "atr": round(atr_val, 2),
            "trend_alignment": trend_alignment,
            "hourly_bias": round(hourly_bias, 4),
            "daily_bias": round(daily_bias, 4),
            "liquidity_sweep": liquidity_sweep,
            "wyckoff_phase": wyckoff_phase,
            "order_flow_proxy": round(order_flow_proxy, 4),
        }

        return {
            "reasons": reasons,
            "features": feature_snapshot,
            "net_pnl": round(net_pnl, 4),
        }

    def log_loss_analysis(self, position: dict, live_row: "pd.Series") -> dict:
        """
        Gọi analyze_loss() → ghi log vào outputs/loss_analysis.jsonl.
        Trả về dict phân tích để orchestrator dùng tiếp.
        """
        analysis = self.analyze_loss(position, live_row)
        event = {
            "event": "loss_analysis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticket": position.get("ticket"),
            "side": position.get("side"),
            "volume": position.get("volume"),
            "profit": position.get("profit"),
            "swap": position.get("swap"),
            "commission": position.get("commission"),
            "net_pnl": analysis["net_pnl"],
            "reasons": analysis["reasons"],
            "features": analysis["features"],
        }
        with self._loss_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")

        reason_summary = " | ".join(analysis["reasons"])
        LOGGER.warning(
            "LOSS ANALYSIS ticket=%s side=%s net_pnl=%.2f | %s",
            position.get("ticket"), position.get("side"),
            analysis["net_pnl"], reason_summary,
        )

        # Lưu feature pattern để dùng cho retrain
        self._loss_patterns.append(analysis["features"])
        if len(self._loss_patterns) > 50:
            self._loss_patterns = self._loss_patterns[-50:]

        return analysis

    def maybe_retrain_on_loss(self, frames: dict, loss_count: int) -> dict | None:
        """
        Trigger retrain có trọng số nếu tích lũy đủ LOSS_RETRAIN_THRESHOLD lệnh thua.
        Upweight các sample tương tự pattern lệnh thua → model cẩn thận hơn.
        """
        if loss_count < self.LOSS_RETRAIN_THRESHOLD or not self._loss_patterns:
            return None

        from xauusd_ai.features.dataset import prepare_training_dataset

        try:
            frames = self._merge_with_cache(frames)
            dataset = prepare_training_dataset(self.settings, frames, self.strategy)
            if len(dataset) < self.settings.training.live_learning_min_rows:
                return None

            metrics = self.trainer.train_with_loss_weights(
                dataset, self._loss_patterns, save_artifacts=False
            )
            self._retrain_count += 1

            baseline_metrics = self._load_saved_metrics()
            best_before = float(self._best_roc_auc)
            roc_auc = float(metrics.get("roc_auc", 0.0))
            if roc_auc >= best_before:
                self._best_roc_auc = roc_auc
                self.trainer._last_roc_auc = roc_auc
                self.trainer._save_artifacts()  # Only save when truly better
                status = "loss_retrain_improved"
            else:
                status = "loss_retrain_no_improvement"

            precision = float(metrics.get("precision", 0.0))
            recall = float(metrics.get("recall", 0.0))
            f1 = float(metrics.get("f1", 0.0))
            selected_threshold = float(metrics.get("selected_threshold", self.trainer.decision_threshold))
            roc_auc_delta = self._metric_delta(roc_auc, baseline_metrics.get("roc_auc"))
            precision_delta = self._metric_delta(precision, baseline_metrics.get("precision"))
            recall_delta = self._metric_delta(recall, baseline_metrics.get("recall"))
            f1_delta = self._metric_delta(f1, baseline_metrics.get("f1"))
            focus_reasons = self._recent_loss_focus()
            accepted = status == "loss_retrain_improved"

            event = {
                "event": "loss_retrain",
                "learning_kind": "loss_driven",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "retrain_count": self._retrain_count,
                "triggered_by_losses": loss_count,
                "loss_patterns_used": len(self._loss_patterns),
                "dataset_rows": int(len(dataset)),
                "roc_auc": round(roc_auc, 4),
                "best_roc_auc_before": round(best_before, 4),
                "best_roc_auc_after": round(self._best_roc_auc, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "selected_threshold": round(selected_threshold, 4),
                "baseline_roc_auc": round(baseline_metrics["roc_auc"], 4) if baseline_metrics.get("roc_auc") is not None else None,
                "baseline_precision": round(baseline_metrics["precision"], 4) if baseline_metrics.get("precision") is not None else None,
                "baseline_recall": round(baseline_metrics["recall"], 4) if baseline_metrics.get("recall") is not None else None,
                "baseline_f1": round(baseline_metrics["f1"], 4) if baseline_metrics.get("f1") is not None else None,
                "roc_auc_delta": roc_auc_delta,
                "precision_delta": precision_delta,
                "recall_delta": recall_delta,
                "f1_delta": f1_delta,
                "accepted": accepted,
                "focus_reasons": focus_reasons,
                "status": status,
                "learning_summary": self._describe_learning(
                    event_name="loss_retrain",
                    accepted=accepted,
                    roc_auc_delta=roc_auc_delta,
                    precision_delta=precision_delta,
                    recall_delta=recall_delta,
                    focus_reasons=focus_reasons,
                ),
            }
            self._log_event(event)
            LOGGER.info(
                "LossRetrain: %s | roc_auc=%.4f | losses=%d | patterns=%d",
                status, roc_auc, loss_count, len(self._loss_patterns),
            )
            return event

        except Exception as exc:
            LOGGER.error("SelfLearner.maybe_retrain_on_loss error: %s", exc, exc_info=True)
            return None

    def get_loss_summary(self) -> dict:
        """Tóm tắt thống kê lệnh thua để dashboard hiển thị."""
        events: list[dict] = []
        if self._loss_log_path.exists():
            for line in self._loss_log_path.read_text(encoding="utf-8").splitlines():
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass

        if not events:
            return {"total_losses": 0, "recent": []}

        # Thống kê nguyên nhân phổ biến nhất
        from collections import Counter
        all_reasons: list[str] = []
        for e in events:
            all_reasons.extend(e.get("reasons", []))

        reason_prefix = [r.split(":")[0] for r in all_reasons]
        top_reasons = Counter(reason_prefix).most_common(5)

        return {
            "total_losses": len(events),
            "top_reasons": top_reasons,
            "recent": events[-5:],
            "accumulated_pending": self._accumulated_losses,
        }
