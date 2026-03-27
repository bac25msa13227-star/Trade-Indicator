from __future__ import annotations

import datetime
import html
import itertools
import json
import logging
import os
import queue
import signal as _signal
import threading
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from xauusd_ai.backtesting.engine import simulate_prediction_backtest, write_backtest_outputs
from xauusd_ai.config import Settings, load_settings
from xauusd_ai.data.market_data import MarketDataService
from xauusd_ai.execution.mt5_executor import MT5Executor
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.features.dataset import build_live_feature_frame, prepare_training_dataset, build_merged_context
from xauusd_ai.learning.self_learner import SelfLearner
from xauusd_ai.model.trainer import ModelTrainer
from xauusd_ai.notifications.telegram import TelegramNotifier
from xauusd_ai.strategies.hybrid import HybridStrategy
from xauusd_ai.visualization.reports import save_backtest_plots, save_training_plot


LOGGER = logging.getLogger(__name__)


class _LearnerThread(threading.Thread):
    """
    Background thread cháº¡y self-learning song song vá»›i live trading.

    - Main thread gá»­i frames qua submit_frames() / submit_loss() â†’ khÃ´ng bao giá» block.
    - Khi retrain xong, set _reload_event Ä‘á»ƒ main thread biáº¿t load láº¡i model.
    - DÃ¹ng trainer_learn riÃªng (tÃ¡ch biá»‡t trainer_live cá»§a main thread) Ä‘á»ƒ trÃ¡nh race.
    """

    def __init__(
        self,
        self_learner: SelfLearner,
        reload_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="LearnerThread")
        self._self_learner = self_learner
        self._reload_event = reload_event
        # maxsize=1: chá»‰ giá»¯ frame má»›i nháº¥t, drop frame cÅ© chÆ°a ká»‹p xá»­ lÃ½
        self._frames_q: queue.Queue[dict[str, pd.DataFrame]] = queue.Queue(maxsize=1)
        # HÃ ng Ä‘á»£i loss-retrain (loss_count, frames)
        self._loss_q: queue.Queue[tuple[int, dict[str, pd.DataFrame]]] = queue.Queue(maxsize=5)
        # Hàng đợi kết quả học gần nhất để main thread gửi Telegram / reload model.
        self._results_q: queue.Queue[dict[str, object]] = queue.Queue(maxsize=10)
        self._stop_event = threading.Event()  # renamed to avoid conflict with Thread._stop() internal method

    # ------------------------------------------------------------------
    # Public API (gá»i tá»« main thread)
    # ------------------------------------------------------------------
    def submit_frames(self, frames: dict[str, pd.DataFrame]) -> None:
        """Gá»­i frames Ä‘á»ƒ trigger self-learning. KhÃ´ng block, drop frame cÅ© náº¿u queue Ä‘áº§y."""
        try:
            self._frames_q.put_nowait(frames)
        except queue.Full:
            # LÃ m rá»—ng slot cÅ© rá»“i Ä‘áº·t frame má»›i nháº¥t vÃ o
            try:
                self._frames_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames_q.put_nowait(frames)
            except queue.Full:
                pass

    def submit_loss(self, loss_count: int, frames: dict[str, pd.DataFrame]) -> None:
        """Gá»­i yÃªu cáº§u loss-retrain. KhÃ´ng block."""
        try:
            self._loss_q.put_nowait((loss_count, frames))
        except queue.Full:
            LOGGER.debug("LearnerThread: loss queue Ä‘áº§y, bá» qua láº§n nÃ y")

    def stop(self) -> None:
        self._stop_event.set()

    def _publish_result(self, result: dict[str, object]) -> None:
        try:
            self._results_q.put_nowait(result)
        except queue.Full:
            try:
                self._results_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._results_q.put_nowait(result)
            except queue.Full:
                pass

    def drain_results(self) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        while True:
            try:
                results.append(self._results_q.get_nowait())
            except queue.Empty:
                break
        return results

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------
    def run(self) -> None:
        LOGGER.info("LearnerThread: started (background learning)")
        _last_loss_retrain_ts: float = 0.0
        _LOSS_RETRAIN_COOLDOWN = 600  # min 10 minutes between loss retrains
        while not self._stop_event.is_set():
            # Æ¯u tiÃªn loss-retrain trÆ°á»›c (ngáº¯n hÆ¡n regular retrain)
            try:
                loss_count, frames = self._loss_q.get_nowait()
                now = time.time()
                if now - _last_loss_retrain_ts < _LOSS_RETRAIN_COOLDOWN:
                    LOGGER.info("LearnerThread: loss-retrain skipped (cooldown %ds)",
                                int(_LOSS_RETRAIN_COOLDOWN - (now - _last_loss_retrain_ts)))
                    continue
                LOGGER.info("LearnerThread: loss-retrain triggered (losses=%d)", loss_count)
                result = self._self_learner.maybe_retrain_on_loss(frames, loss_count)
                _last_loss_retrain_ts = time.time()
                if result:
                    self._publish_result(result)
                    if bool(result.get("accepted")):
                        self._reload_event.set()
                        LOGGER.info("LearnerThread: loss-retrain done â†’ signal reload | %s", result.get("status"))
                    else:
                        LOGGER.info("LearnerThread: loss-retrain rejected | %s", result.get("status"))
                continue  # kiá»ƒm tra loss_q láº¡i ngay
            except queue.Empty:
                pass

            # Regular self-learning (block tá»‘i Ä‘a 2s Ä‘á»ƒ khÃ´ng spin)
            try:
                frames = self._frames_q.get(timeout=2.0)
                LOGGER.info("LearnerThread: regular retrain started")
                result = self._self_learner.maybe_retrain(frames)
                if result:
                    self._publish_result(result)
                    if bool(result.get("accepted")):
                        self._reload_event.set()
                        LOGGER.info("LearnerThread: retrain done â†’ signal reload | %s", result.get("status"))
                    else:
                        LOGGER.info("LearnerThread: retrain rejected | %s", result.get("status"))
            except queue.Empty:
                pass

        LOGGER.info("LearnerThread: stopped")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _make_mlflow_tracker(settings: Settings):
    """Create an MLflowTracker for the given settings, or None if unavailable."""
    try:
        from xauusd_ai.infra.mlflow_client import MLflowTracker
        from pathlib import Path as _Path
        _stem = _Path(settings.app.model_path).stem  # e.g. model_ict_wyckoff / model2_weekly500
        _experiment = f"xauusd_{_stem}"
        return MLflowTracker(experiment_name=_experiment)
    except Exception as _exc:  # noqa: BLE001
        LOGGER.warning("MLflow tracker unavailable: %s", _exc)
        return None


def _bootstrap(settings: Settings) -> tuple[MarketDataService, ModelTrainer, HybridStrategy, TelegramNotifier, MT5Executor, RiskManager]:
    load_dotenv()
    _configure_logging(settings.app.log_level)
    data_service = MarketDataService(settings)
    mlflow_tracker = _make_mlflow_tracker(settings)
    trainer = ModelTrainer(settings, mlflow_tracker=mlflow_tracker)
    strategy = HybridStrategy(settings)
    notifier = TelegramNotifier(settings)
    executor = MT5Executor(settings)
    risk_manager = RiskManager(settings)
    return data_service, trainer, strategy, notifier, executor, risk_manager


def _bootstrap_with_learner(
    settings: Settings,
) -> tuple[MarketDataService, ModelTrainer, HybridStrategy, TelegramNotifier, MT5Executor, RiskManager, SelfLearner]:
    data_service, trainer_live, strategy, notifier, executor, risk_manager = _bootstrap(settings)
    # trainer_learn lÃ  instance riÃªng dÃ nh cho background thread â€” trÃ¡nh race condition
    # vá»›i trainer_live Ä‘ang Ä‘Æ°á»£c main thread dÃ¹ng Ä‘á»ƒ score tÃ­n hiá»‡u.
    trainer_learn = ModelTrainer(settings)
    self_learner = SelfLearner(settings, trainer_learn, strategy)
    return data_service, trainer_live, strategy, notifier, executor, risk_manager, self_learner


def _write_training_outputs(settings: Settings, dataset: pd.DataFrame, metrics: dict[str, float]) -> None:
    output_dir = Path(settings.app.training_report_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(settings.app.training_report_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_training_plot(metrics, Path(settings.app.training_plot_path))

    if settings.training.save_dataset:
        Path(settings.training.dataset_path).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(settings.training.dataset_path, index=False)


def _train_on_frames(settings: Settings, trainer: ModelTrainer, strategy: HybridStrategy, frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, float]]:
    dataset = prepare_training_dataset(settings, frames, strategy)
    metrics = trainer.train(dataset)
    _write_training_outputs(settings, dataset, metrics)
    return dataset, metrics


def _log_live_learning_event(settings: Settings, event: dict[str, object]) -> None:
    log_path = Path(settings.app.live_learning_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(event, default=str) + "\n")


def _format_learning_metric(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_learning_delta(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_learning_message(event: dict[str, object]) -> str:
    accepted = bool(event.get("accepted"))
    event_name = str(event.get("event", "self_learn"))
    is_loss_driven = event_name == "loss_retrain"
    title = "Học lại từ lệnh thua" if is_loss_driven else "Tự học định kỳ"
    verdict = "ĐẠT — đã áp dụng model mới" if accepted else "KHÔNG ĐẠT — giữ model cũ"
    icon = "🧠" if accepted else "📝"
    focus_reasons = event.get("focus_reasons") or []
    if not isinstance(focus_reasons, list):
        focus_reasons = []
    focus_text = ", ".join(html.escape(str(reason)) for reason in focus_reasons[:3])

    lines = [
        f"{icon} <b>{title}</b>",
        f"├ Kết luận: <b>{verdict}</b>",
    ]
    if is_loss_driven:
        lines.append(
            f"├ Trigger: {int(event.get('triggered_by_losses') or 0)} lệnh thua | "
            f"patterns={int(event.get('loss_patterns_used') or 0)}"
        )
    if focus_text:
        lines.append(f"├ Trọng tâm học: {focus_text}")
    lines.append(
        f"├ Rows={int(event.get('dataset_rows') or 0)} | threshold={_format_learning_metric(event.get('selected_threshold'), 2)}"
    )
    lines.append(
        f"├ ROC AUC={_format_learning_metric(event.get('roc_auc'))} "
        f"({_format_learning_delta(event.get('roc_auc_delta'))}) | "
        f"floor={_format_learning_metric(event.get('acceptance_floor') or event.get('best_roc_auc_before'))}"
    )
    lines.append(
        f"├ P/R/F1={_format_learning_metric(event.get('precision'))} / "
        f"{_format_learning_metric(event.get('recall'))} / "
        f"{_format_learning_metric(event.get('f1'))}"
    )
    lines.append(
        f"└ Bài học: {html.escape(str(event.get('learning_summary') or 'Khong co tom tat'))}"
    )
    return "\n".join(lines)


def run_training(settings: Settings) -> None:
    data_service, trainer, strategy, _, _, _ = _bootstrap(settings)
    frames = data_service.fetch_multi_timeframe_data(source=settings.market.training_data_source, all_bars=True)
    max_bars = settings.training.max_train_bars
    if max_bars and max_bars > 0:
        for tf in list(frames.keys()):
            if len(frames[tf]) > max_bars:
                frames[tf] = frames[tf].tail(max_bars).reset_index(drop=True)
        LOGGER.info("Training capped to last %d bars per TF (max_train_bars)", max_bars)
    _, metrics = _train_on_frames(settings, trainer, strategy, frames)

    LOGGER.info("Training complete: %s", metrics)


def run_backtest(settings: Settings) -> None:
    data_service, trainer, strategy, _, _, risk_manager = _bootstrap(settings)
    frames = data_service.fetch_multi_timeframe_data(source=settings.market.training_data_source, all_bars=True)
    dataset = prepare_training_dataset(settings, frames, strategy)
    metrics = trainer.train(dataset)
    predictions = trainer.predict_dataset(dataset)
    result = simulate_prediction_backtest(predictions, settings, risk_manager)
    train_rows = predictions[predictions["split"] == "train"].copy()
    test_rows = predictions[predictions["split"] == "test"].copy()
    train_start = str(train_rows["time"].min()) if not train_rows.empty else None
    train_end = str(train_rows["time"].max()) if not train_rows.empty else None
    test_start = str(test_rows["time"].min()) if not test_rows.empty else None
    test_end = str(test_rows["time"].max()) if not test_rows.empty else None
    trade_start = str(result.trades["time"].min()) if not result.trades.empty else None
    trade_end = str(result.trades["time"].max()) if not result.trades.empty else None
    backtest_report = {
        "train_metrics": metrics,
        "train_start": train_start,
        "train_end": train_end,
        "train_days": int((train_rows["time"].max() - train_rows["time"].min()).days) if not train_rows.empty else 0,
        "test_start": test_start,
        "test_end": test_end,
        "test_days": int((test_rows["time"].max() - test_rows["time"].min()).days) if not test_rows.empty else 0,
        "trade_start": trade_start,
        "trade_end": trade_end,
        "trade_days": int((pd.to_datetime(result.trades["time"]).max() - pd.to_datetime(result.trades["time"]).min()).days)
        if not result.trades.empty
        else 0,
        **result.report,
    }
    write_backtest_outputs(
        backtest_report,
        result.trades,
        test_rows,
        Path(settings.app.backtest_report_path),
        Path(settings.app.backtest_trades_path),
        Path(settings.app.backtest_equity_plot_path),
        Path(settings.app.backtest_trades_plot_path),
    )
    LOGGER.info("Backtest report: %s", backtest_report)
    print(json.dumps(backtest_report, indent=2))


def run_walkforward(settings: Settings) -> None:
    data_service, _, _, _, _, risk_manager = _bootstrap(settings)
    # Walk-forward must use the full historical window, otherwise it silently
    # evaluates only the small live bar cache from market.bars and produces
    # invalid folds / date splits.
    frames = data_service.fetch_multi_timeframe_data(
        source=settings.market.training_data_source,
        all_bars=True,
    )

    # Build the expensive base merged frame once (includes news features fetch).
    # Only strategy_score and labels need to be recomputed per combination.
    LOGGER.info("WalkForward: building base merged context (once for all combinations)...")
    base_merged = build_merged_context(settings, frames)
    LOGGER.info("WalkForward: base merged context ready (%d rows)", len(base_merged))

    full_candidate_grid = list(
        itertools.product(
        settings.training.walkforward_ict_weights,
        settings.training.walkforward_wyckoff_weights,
        settings.training.walkforward_momentum_weights,
        settings.training.walkforward_min_strategy_scores,
        settings.training.walkforward_sideway_min_strategy_scores,
        settings.training.walkforward_strong_volatility_min_strategy_scores,
        settings.training.walkforward_min_confidences,
        settings.training.walkforward_require_trend_alignment,
        settings.training.walkforward_label_horizons,
        settings.training.walkforward_return_thresholds,
        )
    )

    best_summary: dict[str, object] | None = None
    best_score: float | None = None
    best_fallback_summary: dict[str, object] | None = None
    best_fallback_score: float | None = None
    all_trade_frames: list[pd.DataFrame] = []
    fallback_trade_frames: list[pd.DataFrame] = []
    max_combinations = settings.training.walkforward_max_combinations

    if len(full_candidate_grid) > max_combinations:
        sampled_indices = {
            round(index * (len(full_candidate_grid) - 1) / (max_combinations - 1))
            for index in range(max_combinations)
        }
        candidate_grid = [candidate for index, candidate in enumerate(full_candidate_grid) if index in sampled_indices]
    else:
        candidate_grid = full_candidate_grid

    n_combos = len(candidate_grid)
    progress_path = Path(settings.app.walkforward_report_path).parent / "walkforward_progress.json"
    _wf_started_at = datetime.datetime.now(datetime.timezone.utc)

    def _write_progress(
        completed: int,
        status: str = "running",
        current_params: dict | None = None,
        best_so_far: dict | None = None,
    ) -> None:
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - _wf_started_at).total_seconds()
        condensed: dict | None = None
        if best_so_far:
            condensed = {
                "params": best_so_far.get("params", {}),
                "avg_return_pct": best_so_far.get("avg_return_pct"),
                "avg_profit_factor": best_so_far.get("avg_profit_factor"),
                "avg_max_drawdown_pct": best_so_far.get("avg_max_drawdown_pct"),
                "avg_precision": best_so_far.get("avg_precision"),
                "avg_recall": best_so_far.get("avg_recall"),
                "avg_trades": best_so_far.get("avg_trades"),
            }
        try:
            progress_path.write_text(json.dumps({
                "status": status,
                "total_combinations": n_combos,
                "completed_combinations": completed,
                "pct_done": round(completed / n_combos * 100, 1) if n_combos > 0 else 0,
                "elapsed_seconds": round(elapsed, 1),
                "started_at": _wf_started_at.isoformat(),
                "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "current_params": current_params,
                "best_so_far": condensed,
            }, indent=2), encoding="utf-8")
        except Exception:
            pass

    _write_progress(0)
    for combo_idx, (
        ict_weight,
        wyckoff_weight,
        momentum_weight,
        min_score,
        sideway_min_score,
        strong_volatility_min_score,
        min_confidence,
        require_trend_alignment,
        label_horizon,
        return_threshold,
    ) in enumerate(candidate_grid):
        candidate_settings = settings.model_copy(deep=True)
        candidate_settings.strategy.ict_weight = ict_weight
        candidate_settings.strategy.wyckoff_weight = wyckoff_weight
        candidate_settings.strategy.momentum_weight = momentum_weight
        candidate_settings.strategy.min_strategy_score = min_score
        candidate_settings.strategy.sideway_min_strategy_score = sideway_min_score
        candidate_settings.strategy.strong_volatility_min_strategy_score = strong_volatility_min_score
        candidate_settings.strategy.require_trend_alignment = require_trend_alignment
        candidate_settings.risk.min_confidence = min_confidence
        candidate_settings.training.label_horizon = label_horizon
        candidate_settings.training.min_return_threshold = return_threshold

        trainer = ModelTrainer(candidate_settings)
        strategy = HybridStrategy(candidate_settings)
        dataset = prepare_training_dataset(candidate_settings, frames, strategy, cached_merged=base_merged)

        fold_reports: list[dict[str, object]] = []
        fold_trade_frames: list[pd.DataFrame] = []
        train_size = candidate_settings.training.walkforward_train_size
        test_size = candidate_settings.training.walkforward_test_size
        step_size = candidate_settings.training.walkforward_step_size

        max_folds = candidate_settings.training.walkforward_max_folds_per_combination
        fold_indices = range(0, max(len(dataset) - train_size - test_size + 1, 0), step_size)
        if max_folds > 0:
            fold_indices = list(fold_indices)[-max_folds:]
        for fold_start in fold_indices:
            train_end = fold_start + train_size
            test_end = train_end + test_size
            fold_train = dataset.iloc[fold_start:train_end].copy()
            fold_test = dataset.iloc[train_end:test_end].copy()
            if len(fold_train) < 200 or len(fold_test) < 50:
                continue

            fold_dataset = pd.concat([fold_train, fold_test], ignore_index=True)
            fold_dataset["split"] = "train"
            fold_dataset.loc[len(fold_train):, "split"] = "test"
            metrics = trainer.train(fold_dataset, save_artifacts=False)
            predictions = trainer.predict_dataset(fold_dataset)
            simulation = simulate_prediction_backtest(predictions, candidate_settings, risk_manager)
            fold_report = {
                "fold": len(fold_reports) + 1,
                "train_start": str(fold_train["time"].min()),
                "train_end": str(fold_train["time"].max()),
                "test_start": str(fold_test["time"].min()),
                "test_end": str(fold_test["time"].max()),
                "train_metrics": metrics,
                **simulation.report,
            }
            fold_reports.append(fold_report)
            if not simulation.trades.empty:
                trades = simulation.trades.copy()
                trades["fold"] = fold_report["fold"]
                trades["ict_weight"] = ict_weight
                trades["wyckoff_weight"] = wyckoff_weight
                trades["momentum_weight"] = momentum_weight
                trades["min_strategy_score"] = min_score
                trades["sideway_min_strategy_score"] = sideway_min_score
                trades["strong_volatility_min_strategy_score"] = strong_volatility_min_score
                trades["min_confidence"] = min_confidence
                trades["require_trend_alignment"] = require_trend_alignment
                trades["label_horizon"] = label_horizon
                trades["return_threshold"] = return_threshold
                fold_trade_frames.append(trades)

        if not fold_reports:
            _write_progress(combo_idx + 1, current_params={
                "ict_weight": ict_weight, "wyckoff_weight": wyckoff_weight,
                "momentum_weight": momentum_weight,
            }, best_so_far=best_summary or best_fallback_summary)
            LOGGER.info("WalkForward: combo %d/%d | no valid folds, skipping", combo_idx + 1, n_combos)
            continue

        avg_recall = sum(report["train_metrics"]["recall"] for report in fold_reports) / len(fold_reports)
        avg_precision = sum(report["train_metrics"]["precision"] for report in fold_reports) / len(fold_reports)
        avg_f1 = sum(report["train_metrics"]["f1"] for report in fold_reports) / len(fold_reports)
        avg_return = sum(report["return_pct"] for report in fold_reports) / len(fold_reports)
        avg_profit_factor = sum(report["profit_factor"] for report in fold_reports) / len(fold_reports)
        avg_drawdown = sum(report["max_drawdown_pct"] for report in fold_reports) / len(fold_reports)
        avg_trades = sum(report["trades"] for report in fold_reports) / len(fold_reports)

        summary = {
            "params": {
                "ict_weight": ict_weight,
                "wyckoff_weight": wyckoff_weight,
                "momentum_weight": momentum_weight,
                "min_strategy_score": min_score,
                "sideway_min_strategy_score": sideway_min_score,
                "strong_volatility_min_strategy_score": strong_volatility_min_score,
                "min_confidence": min_confidence,
                "require_trend_alignment": require_trend_alignment,
                "label_horizon": label_horizon,
                "min_return_threshold": return_threshold,
            },
            "folds": fold_reports,
            "avg_recall": round(avg_recall, 4),
            "avg_precision": round(avg_precision, 4),
            "avg_f1": round(avg_f1, 4),
            "avg_return_pct": round(avg_return, 4),
            "avg_profit_factor": round(avg_profit_factor, 4),
            "avg_max_drawdown_pct": round(avg_drawdown, 4),
            "avg_trades": round(avg_trades, 2),
        }

        composite_score = (
            summary["avg_recall"] * 1.2
            + max(summary["avg_profit_factor"] - 1.0, -1.0) * 0.8
            + summary["avg_return_pct"] / 100.0
            - abs(summary["avg_max_drawdown_pct"]) / 25.0
            + summary["avg_precision"] * 0.6
            + min(summary["avg_trades"], 100.0) / 200.0
        )
        passes_guardrails = (
            summary["avg_profit_factor"] >= settings.training.walkforward_min_avg_profit_factor
            and summary["avg_precision"] >= settings.training.walkforward_min_avg_precision
            and abs(summary["avg_max_drawdown_pct"]) <= settings.training.walkforward_max_avg_drawdown_pct
            and summary["avg_return_pct"] >= settings.training.walkforward_min_avg_return_pct
            and summary["avg_trades"] >= settings.training.walkforward_min_avg_trades
        )

        if passes_guardrails and (best_score is None or composite_score > best_score):
            best_summary = summary
            best_score = composite_score
            all_trade_frames = fold_trade_frames

        if best_fallback_score is None or composite_score > best_fallback_score:
            best_fallback_summary = summary
            best_fallback_score = composite_score
            fallback_trade_frames = fold_trade_frames

        _write_progress(combo_idx + 1, current_params={
            "ict_weight": ict_weight, "wyckoff_weight": wyckoff_weight,
            "momentum_weight": momentum_weight,
        }, best_so_far=best_summary or best_fallback_summary)
        LOGGER.info(
            "WalkForward: combo %d/%d done | score=%.4f | return=%.2f%% | pf=%.3f | dd=%.2f%% | %s",
            combo_idx + 1, n_combos, composite_score,
            avg_return, avg_profit_factor, avg_drawdown,
            "qualified" if passes_guardrails else "filtered",
        )

    if best_summary is None:
        if best_fallback_summary is None:
            raise RuntimeError("Walk-forward could not produce any valid folds")
        best_summary = best_fallback_summary
        all_trade_frames = fallback_trade_frames
        best_summary["fallback_used"] = True
    else:
        best_summary["fallback_used"] = False

    best_summary["selection_guardrails"] = {
        "min_avg_profit_factor": settings.training.walkforward_min_avg_profit_factor,
        "min_avg_precision": settings.training.walkforward_min_avg_precision,
        "max_avg_drawdown_pct": settings.training.walkforward_max_avg_drawdown_pct,
        "min_avg_return_pct": settings.training.walkforward_min_avg_return_pct,
        "min_avg_trades": settings.training.walkforward_min_avg_trades,
    }

    walkforward_trades = pd.concat(all_trade_frames, ignore_index=True) if all_trade_frames else pd.DataFrame()
    Path(settings.app.walkforward_report_path).write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    if not walkforward_trades.empty:
        walkforward_trades.to_csv(settings.app.walkforward_trades_path, index=False)
    _write_progress(n_combos, status="done", best_so_far=best_summary)
    LOGGER.info("Walk-forward best summary: %s", best_summary)
    print(json.dumps(best_summary, indent=2))


def run_paper_trade_loop(settings: Settings) -> None:
    data_service, trainer, strategy, notifier, _, risk_manager = _bootstrap(settings)
    model_ready = trainer.load_artifacts()
    if settings.training.retrain_on_startup or not model_ready:
        run_training(settings)
        trainer.load_artifacts()

    log_path = Path(settings.app.paper_trade_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    loops = 0
    last_logged_time: str | None = None

    while True:
        frames = data_service.fetch_multi_timeframe_data(source=settings.execution.paper_data_source)
        live_frame = build_live_feature_frame(settings, frames, strategy)
        latest_row = live_frame.iloc[-1]
        latest_time = str(latest_row["time"])
        if latest_time != last_logged_time:
            signal = trainer.score_live_row(live_frame)
            decision = strategy.build_trade_decision(frames, latest_row, signal)
            order_plan = risk_manager.build_order_plan(decision, frames[settings.market.execution_timeframe].iloc[-1])
            signal_row = pd.DataFrame(
                [
                    {
                        "time": latest_time,
                        "should_trade": decision.should_trade,
                        "side": decision.side,
                        "confidence": decision.confidence,
                        "reason": decision.reason,
                        "entry_price": order_plan.entry_price,
                        "stop_loss": order_plan.stop_loss,
                        "take_profit": order_plan.take_profit,
                        "strategy_score": float(latest_row["strategy_score"]),
                        "volatility_regime": int(latest_row["volatility_regime"]),
                    }
                ]
            )
            if log_path.exists():
                signal_row.to_csv(log_path, mode="a", header=False, index=False)
            else:
                signal_row.to_csv(log_path, index=False)
            notifier.send_signal(decision, order_plan)
            last_logged_time = latest_time
            LOGGER.info("Paper trade signal logged: %s", signal_row.to_dict(orient="records")[0])

        loops += 1
        if settings.execution.paper_trade_max_loops > 0 and loops >= settings.execution.paper_trade_max_loops:
            break
        time.sleep(settings.app.poll_seconds)


def run_live_loop(settings: Settings) -> None:
    data_service, trainer, strategy, notifier, executor, risk_manager, self_learner = _bootstrap_with_learner(settings)

    # ── SIGTERM handler: Docker stop → send Telegram then exit cleanly ────────
    def _handle_sigterm(signum, frame):
        try:
            notifier.send_message(
                f"🔴 <b>Bot đã dừng</b> — Docker stop (SIGTERM)\n"
                f"└ Account: {settings.market.symbol}"
            )
        except Exception:
            pass
        raise SystemExit(0)
    _signal.signal(_signal.SIGTERM, _handle_sigterm)

    # ── Infrastructure: PostgreSQL + Prometheus ───────────────────────────────
    _account_tag = Path(settings.app.live_closed_trades_path).stem.replace("live_closed_trades_", "").strip("_") or "default"
    _trade_store = None
    _trading_metrics = None
    try:
        from xauusd_ai.infra.db import get_engine, init_tables, TradeStore
        _db_engine = get_engine()
        init_tables(_db_engine)
        _trade_store = TradeStore(_db_engine)
        LOGGER.info("PostgreSQL TradeStore initialized (account=%s)", _account_tag)
    except Exception as _db_exc:
        LOGGER.warning("PostgreSQL unavailable — falling back to CSV only: %s", _db_exc)
    try:
        from xauusd_ai.infra.metrics import TradingMetrics
        _trading_metrics = TradingMetrics(_account_tag)
        LOGGER.info("Prometheus metrics initialized (account=%s)", _account_tag)
        # Start HTTP metrics server so Prometheus (in Docker) can scrape via host.docker.internal
        try:
            from prometheus_client import start_http_server
            _metrics_port = int(os.environ.get("METRICS_PORT", 8002 if "acc2" in _account_tag else 8001))
            start_http_server(_metrics_port)
            LOGGER.info("Prometheus metrics HTTP server started on port %d (account=%s)", _metrics_port, _account_tag)
        except OSError as _port_exc:
            LOGGER.warning("Prometheus HTTP server port busy (already running?): %s", _port_exc)
        except Exception as _hs_exc:
            LOGGER.warning("Prometheus HTTP server failed to start: %s", _hs_exc)
    except Exception as _pm_exc:
        LOGGER.warning("Prometheus metrics unavailable: %s", _pm_exc)

    # ── Config hot-reload: detect YAML file changes ──────────────────────────
    _config_path = getattr(settings, "_config_path", None)
    _cp = Path(_config_path) if _config_path else None
    _config_mtime: float = _cp.stat().st_mtime if _cp and _cp.exists() else 0.0
    _tunnel_url_mtime: float = 0.0
    _tunnel_url_sent: str = ""  # track last URL sent to avoid duplicate notifications

    def _maybe_reload_config() -> None:
        """Reload toggleable settings from YAML without restart."""
        nonlocal settings, _config_mtime, _exit_model, news_crawler
        if not _config_path or not Path(_config_path).exists():
            return
        try:
            cur_mtime = Path(_config_path).stat().st_mtime
            if cur_mtime <= _config_mtime:
                return
            _config_mtime = cur_mtime
            new_settings = load_settings(Path(_config_path))
            # Hot-reload toggleable fields (keep heavy objects as-is)
            settings.execution.auto_trade = new_settings.execution.auto_trade
            settings.execution.trailing_sl.enabled = new_settings.execution.trailing_sl.enabled
            settings.execution.dca.enabled = new_settings.execution.dca.enabled
            settings.execution.close_opposite_on_signal = new_settings.execution.close_opposite_on_signal
            settings.risk.risk_per_trade = new_settings.risk.risk_per_trade
            settings.risk.max_open_positions = new_settings.risk.max_open_positions
            settings.risk.take_profit_rr = new_settings.risk.take_profit_rr
            settings.risk.stop_loss_atr_multiple = new_settings.risk.stop_loss_atr_multiple
            settings.strategy.signal_threshold = new_settings.strategy.signal_threshold
            settings.strategy.min_strategy_score = new_settings.strategy.min_strategy_score
            settings.strategy.force_trade = new_settings.strategy.force_trade
            settings.strategy.blocked_hours_utc = new_settings.strategy.blocked_hours_utc
            settings.strategy.blocked_weekdays_utc = new_settings.strategy.blocked_weekdays_utc
            # Exit model toggle
            if new_settings.execution.exit_model.enabled and _exit_model is None:
                try:
                    from xauusd_ai.model.exit_model import ExitModel
                    _exit_model = ExitModel(settings)
                    if not _exit_model.load():
                        _exit_model = None
                except Exception:
                    _exit_model = None
            elif not new_settings.execution.exit_model.enabled:
                _exit_model = None
            settings.execution.exit_model.enabled = new_settings.execution.exit_model.enabled
            settings.execution.exit_model.exit_threshold = new_settings.execution.exit_model.exit_threshold
            # News toggle
            if new_settings.integrations.news.enabled and news_crawler is None:
                try:
                    from xauusd_ai.data.news_crawler import NewsCrawler
                    news_crawler = NewsCrawler(settings)
                except Exception:
                    pass
            elif not new_settings.integrations.news.enabled:
                news_crawler = None
            settings.integrations.news.enabled = new_settings.integrations.news.enabled
            LOGGER.info("Config hot-reloaded from %s", _config_path)
            notifier.send_message("🔄 <b>Config reloaded</b> — thay đổi đã áp dụng (không cần restart)")
        except Exception as exc:
            LOGGER.warning("Config hot-reload failed: %s", exc)

    model_ready = trainer.load_artifacts()
    if settings.training.retrain_on_startup or not model_ready:
        LOGGER.info("Training artifacts missing or retrain enabled, starting training")
        run_training(settings)
        trainer.load_artifacts()

    # ── Exit Model (optional) ────────────────────────────────────────────────
    _exit_model = None
    if settings.execution.exit_model.enabled:
        try:
            from xauusd_ai.model.exit_model import ExitModel  # noqa: PLC0415
            _exit_model = ExitModel(settings)
            if _exit_model.load():
                LOGGER.info(
                    "ExitModel loaded (thr=%.2f min_rr=%.2f)",
                    _exit_model.threshold,
                    settings.execution.exit_model.min_unrealized_rr,
                )
            else:
                LOGGER.warning("ExitModel enabled but artifacts not found — disabled. Run: python scripts/train_exit_model.py")
                _exit_model = None
        except Exception as _em_err:
            LOGGER.error("ExitModel load error: %s", _em_err)
            _exit_model = None

    # Tracks RSI value at the moment each order was placed: {ticket: rsi_at_entry}
    _RSI_TRACKER_FILE = Path("outputs/entry_rsi_tracker.json")

    def _load_rsi_tracker() -> dict[int, float]:
        try:
            if _RSI_TRACKER_FILE.exists():
                data = json.loads(_RSI_TRACKER_FILE.read_text(encoding="utf-8"))
                return {int(k): float(v) for k, v in data.items()}
        except Exception as _e:
            LOGGER.warning("Failed to load entry_rsi_tracker: %s", _e)
        return {}

    def _save_rsi_tracker(tracker: dict[int, float]) -> None:
        try:
            _RSI_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _RSI_TRACKER_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps({str(k): v for k, v in tracker.items()}), encoding="utf-8")
            tmp.replace(_RSI_TRACKER_FILE)
        except Exception as _e:
            LOGGER.warning("Failed to save entry_rsi_tracker: %s", _e)

    _entry_rsi_tracker: dict[int, float] = _load_rsi_tracker()

    # â”€â”€ Khá»Ÿi táº¡o News Crawler (náº¿u báº­t) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    news_crawler = None
    if settings.integrations.news.enabled:
        try:
            from xauusd_ai.data.news_crawler import NewsCrawler
            news_crawler = NewsCrawler(settings)
            LOGGER.info("NewsCrawler initialized")
        except Exception as news_init_err:
            LOGGER.warning("NewsCrawler init failed: %s", news_init_err)

    # â”€â”€ Parallel learning setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # _model_reload_event: learner thread set khi retrain xong â†’ main thread reload
    _model_reload_event = threading.Event()
    # _model_lock: báº£o vá»‡ trainer.score_live_row() vÃ  trainer.load_artifacts()
    # khá»i cháº¡y Ä‘á»“ng thá»i (dÃ¹ load_artifacts nhanh, váº«n cáº§n an toÃ n)
    _model_lock = threading.RLock()

    learner_thread: _LearnerThread | None = None
    if settings.training.live_learning_enabled:
        learner_thread = _LearnerThread(self_learner, _model_reload_event)
        learner_thread.start()
        LOGGER.info("LearnerThread started â€” live trading & learning cháº¡y song song")

    # â”€â”€ DCA state: track sá»‘ láº§n DCA Ä‘Ã£ thá»±c hiá»‡n per ticket â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _dca_state_path = Path("outputs/dca_state.json")

    def _load_dca_state() -> dict[str, int]:
        if _dca_state_path.exists():
            try:
                return json.loads(_dca_state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_dca_state(state: dict) -> None:
        _dca_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = _dca_state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(_dca_state_path)

    # ── Live closed trades log ────────────────────────────────────────────────
    _live_trades_path = Path(settings.app.live_closed_trades_path)
    _live_trades_path.parent.mkdir(parents=True, exist_ok=True)
    _live_trades_header = "time,ticket,side,volume,open_price,close_price,profit,swap,commission,pnl,is_win,close_type,session_id\n"
    # Preserve history across restarts -- only write header for a brand-new file
    if not _live_trades_path.exists() or _live_trades_path.stat().st_size == 0:
        _live_trades_path.write_text(_live_trades_header, encoding="utf-8")
    else:
        # Migrate old files: ensure close_type and session_id columns exist
        try:
            _mig_df = pd.read_csv(_live_trades_path, on_bad_lines="skip")
            _mig_changed = False
            if "session_id" not in _mig_df.columns:
                _mig_df["session_id"] = "legacy"
                _mig_changed = True
            if "close_type" not in _mig_df.columns:
                # Retroactively fix: SL trades with pnl>=0 that were wrongly marked as win
                _mig_df["close_type"] = "UNKNOWN"
                if "is_win" in _mig_df.columns and "pnl" in _mig_df.columns:
                    # Any row marked is_win=True but pnl<=0 was a data anomaly, fix it
                    _mig_df.loc[
                        (_mig_df["is_win"].astype(str).isin(["True", "true", "1"])) &
                        (_mig_df["pnl"].astype(float) <= 0),
                        "is_win"
                    ] = False
                _mig_changed = True
            if _mig_changed:
                _col_order = [c for c in [
                    "time", "ticket", "side", "volume", "open_price", "close_price",
                    "profit", "swap", "commission", "pnl", "is_win", "close_type", "session_id"
                ] if c in _mig_df.columns]
                _mig_df[_col_order].to_csv(_live_trades_path, index=False)
                LOGGER.info("Migrated live_trades CSV: added close_type/session_id columns")
        except Exception as _mig_err:
            LOGGER.debug("Could not migrate live trades CSV: %s", _mig_err)
    _signals_path = Path(settings.app.paper_trade_log_path)
    _signals_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep signals CSV across restarts (dashboard de-duplicates by bar time)
    _session_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    def _append_live_trade(pos: dict, pnl: float, is_win: bool, close_type: str = "UNKNOWN") -> None:
        """Ghi lệnh đóng vào CSV + PostgreSQL để dashboard P&L đọc được."""
        import datetime as _dt
        _trade_row = {
            "time":        _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "ticket":      pos.get("ticket", 0),
            "side":        pos.get("side", ""),
            "volume":      pos.get("volume", 0),
            "open_price":  pos.get("open_price", 0),
            "close_price": pos.get("close_price", 0),
            "profit":      pos.get("profit", 0),
            "swap":        pos.get("swap", 0),
            "commission":  pos.get("commission", 0),
            "pnl":         pnl,
            "is_win":      is_win,
            "close_type":  close_type,
            "session_id":  _session_id,
        }
        row = pd.DataFrame([_trade_row])
        # CSV (backward compat)
        if _live_trades_path.exists():
            row.to_csv(_live_trades_path, mode="a", header=False, index=False)
        else:
            row.to_csv(_live_trades_path, index=False)
        # PostgreSQL
        if _trade_store is not None:
            try:
                _trade_store.insert_trade(_account_tag, _trade_row)
            except Exception as _pg_exc:
                LOGGER.debug("PostgreSQL insert_trade failed (non-fatal): %s", _pg_exc)
        # Prometheus
        if _trading_metrics is not None:
            try:
                _trading_metrics.record_trade(pnl=pnl, is_win=is_win, side=pos.get("side", "none"))
            except Exception:
                pass

    last_seen_bar_time: pd.Timestamp | None = None
    last_learning_time: float = time.time()
    last_learning_bar_time: pd.Timestamp | None = None
    # Pre-load last logged bar time so we skip re-logging same candle after restart
    _last_logged_bar_time: pd.Timestamp | None = None
    if _signals_path.exists() and _signals_path.stat().st_size > 0:
        try:
            import csv as _scsv
            with _signals_path.open("r", encoding="utf-8") as _sf:
                _last_csv_row = None
                for _last_csv_row in _scsv.DictReader(_sf):
                    pass
                if _last_csv_row:
                    _last_logged_bar_time = pd.to_datetime(_last_csv_row.get("time"), utc=True, errors="coerce")
        except Exception:
            pass
    # â”€â”€ Loss learning state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _last_closed_check_epoch: float = time.time() - 86400  # look back 24h on startup
    _known_loss_tickets: set[int] = set()
    _accumulated_losses: int = 0
    # Pre-load already-recorded tickets so we don't re-notify on restart
    if _live_trades_path.exists():
        try:
            import csv as _csv
            with _live_trades_path.open("r", encoding="utf-8") as _f:
                for _row in _csv.DictReader(_f):
                    _t = int(float(_row.get("ticket", 0) or 0))
                    if _t:
                        _known_loss_tickets.add(_t)
        except Exception as _le:
            LOGGER.debug("Could not pre-load known tickets: %s", _le)
    _last_row_ctx: pd.Series | None = None
    _last_frames_ctx: dict | None = None

    def _check_closed_positions(row_ctx: pd.Series | None, frames_ctx: dict | None) -> None:
        """Phát hiện lệnh vừa đóng và gửi Telegram ngay lập tức (30s sau khi đóng)."""
        nonlocal _last_closed_check_epoch, _accumulated_losses
        if not settings.execution.auto_trade:
            return
        try:
            now_epoch = time.time()
            closed = executor.get_recently_closed_positions(
                since_epoch=_last_closed_check_epoch - 60,
                magic_number=settings.execution.magic_number,
            )
            _last_closed_check_epoch = now_epoch
            for pos in closed:
                ticket = pos.get("ticket", 0)
                pnl = pos.get("profit", 0) + pos.get("swap", 0) + pos.get("commission", 0)
                if ticket in _known_loss_tickets:
                    continue
                _known_loss_tickets.add(ticket)
                # Determine close type from MT5 deal reason
                # DEAL_REASON_SL=4, DEAL_REASON_TP=5, DEAL_REASON_EXPERT=3
                _reason_code = int(pos.get("reason", 0))
                if _reason_code == 4:
                    _close_type = "SL"
                elif _reason_code == 5:
                    _close_type = "TP"
                elif _reason_code == 3:
                    _close_type = "EA"
                else:
                    _close_type = "MANUAL"

                # SL hit with non-negative PnL = trailing SL closed in profit/breakeven
                # This must NOT be counted as a win — it is an SL close, not a TP close
                if _reason_code == 4 and pnl >= 0:
                    LOGGER.info(
                        "SL-breakeven ticket=%d side=%s pnl=%.2f (SL moved to profit/BE — NOT a win)",
                        ticket, pos.get("side"), pnl,
                    )
                    notifier.send_message(
                        f"\u26a1 <b>L\u1ec7nh SL Ho\u00e0 #{ticket}</b> ({pos.get('side','?').upper()}) \u2014 SL ch\u1ea1m nh\u01b0ng PnL d\u01b0\u01a1ng\n"
                        f"\u251c P&amp;L: <b>{'+' if pnl > 0 else ''}{pnl:.2f}$</b> (kh\u00f4ng t\u00ednh l\u00e0 l\u1ec7nh th\u1eafng)\n"
                        f"\u251c Entry: {pos.get('open_price', 0):.5f} \u2192 Close: {pos.get('close_price', 0):.5f}\n"
                        f"\u251c Volume: {pos.get('volume', 0):.2f} lot\n"
                        f"\u2514 Profit: {pos.get('profit', 0):.2f}$ | Swap: {pos.get('swap', 0):.2f}$ | Comm: {pos.get('commission', 0):.2f}$"
                    )
                    _append_live_trade(pos, pnl, is_win=False, close_type="SL")
                    _entry_rsi_tracker.pop(ticket, None)
                    _save_rsi_tracker(_entry_rsi_tracker)
                elif pnl < 0:
                    _accumulated_losses += 1
                    self_learner._accumulated_losses = _accumulated_losses
                    LOGGER.warning("LOSS detected ticket=%d side=%s pnl=%.2f close_type=%s", ticket, pos.get("side"), pnl, _close_type)
                    _feat: dict = {}
                    _reasons: list[str] = []
                    try:
                        if row_ctx is not None:
                            analysis = self_learner.log_loss_analysis(pos, row_ctx)
                            _feat = analysis.get("features", {})
                            _reasons = analysis.get("reasons", [])
                            LOGGER.warning("LOSS reasons: %s", " | ".join(_reasons))
                    except Exception as _ae:
                        LOGGER.error("Loss analysis failed (notification still sent): %s", _ae)
                    _regime_label = {0: "sideway", 1: "normal", 2: "volatile"}.get(
                        int(_feat.get("volatility_regime", 1)), "?"
                    )
                    _reasons_text = (
                        "\n".join(f"  \u2022 {html.escape(str(r))}" for r in _reasons)
                        if _reasons else "  \u2022 Khong xac dinh ro nguyen nhan"
                    )
                    notifier.send_message(
                        f"\u26a0\ufe0f <b>Lenh THUA #{ticket}</b> ({pos.get('side','?').upper()}) \u2014 Loss #{_accumulated_losses} [{_close_type}]\n"
                        f"\u251c P&amp;L: <b>{pnl:.2f}$</b>\n"
                        f"\u251c Entry: {pos.get('open_price', 0):.5f} \u2192 Close: {pos.get('close_price', 0):.5f}\n"
                        f"\u251c Volume: {pos.get('volume', 0):.2f} lot\n"
                        f"\u251c ATR: {_feat.get('atr', 0):.2f} | RSI: {_feat.get('rsi', 0):.1f} | Score: {_feat.get('strategy_score', 0):.3f}\n"
                        f"\u251c Regime: {_regime_label} | Trend: {'OK' if int(_feat.get('trend_alignment', 0)) == 1 else 'MISS'}\n"
                        f"\u2514 Nguyen nhan:\n{_reasons_text}"
                    )
                    _append_live_trade(pos, pnl, is_win=False, close_type=_close_type)
                    # Record loss for circuit breaker
                    risk_manager.record_trade_result(pnl, account_info.get("balance", 0.0) if 'account_info' in dir() else 0.0)
                    _entry_rsi_tracker.pop(ticket, None)
                    _save_rsi_tracker(_entry_rsi_tracker)
                else:
                    LOGGER.info("Position closed in profit: ticket=%d pnl=%.2f close_type=%s", ticket, pnl, _close_type)
                    notifier.send_message(
                        f"\u2705 <b>Lenh THANG #{ticket}</b> ({pos.get('side','?').upper()}) [{_close_type}]\n"
                        f"\u251c P&amp;L: <b>+{pnl:.2f}$</b>\n"
                        f"\u251c Entry: {pos.get('open_price', 0):.5f} \u2192 Close: {pos.get('close_price', 0):.5f}\n"
                        f"\u251c Volume: {pos.get('volume', 0):.2f} lot\n"
                        f"\u2514 Profit: {pos.get('profit', 0):.2f}$ | Swap: {pos.get('swap', 0):.2f}$ | Comm: {pos.get('commission', 0):.2f}$"
                    )
                    _append_live_trade(pos, pnl, is_win=True, close_type=_close_type)
                    # Record win for circuit breaker (resets consecutive loss counter)
                    risk_manager.record_trade_result(pnl, account_info.get("balance", 0.0) if 'account_info' in dir() else 0.0)
                    _entry_rsi_tracker.pop(ticket, None)
                    _save_rsi_tracker(_entry_rsi_tracker)
            # Trigger loss-retrain sau LOSS_RETRAIN_THRESHOLD lệnh thua
            if _accumulated_losses >= self_learner.LOSS_RETRAIN_THRESHOLD and learner_thread is not None:
                LOGGER.info("LossRetrain queued for background thread (losses=%d)", _accumulated_losses)
                learner_thread.submit_loss(_accumulated_losses, frames_ctx or {})
                notifier.send_message(
                    f"🔄 <b>Đang học lại</b> sau {_accumulated_losses} lệnh thua\n"
                    f"(background \u2013 kh\u00f4ng d\u1eebng giao d\u1ecbch)"
                )
                _accumulated_losses = 0
                self_learner._accumulated_losses = 0
        except Exception as loss_err:
            LOGGER.error("Loss learning error: %s", loss_err)

    # timeframe seconds mapping for bars_held computation
    _tf_secs_map = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}
    _exec_tf_secs = _tf_secs_map.get(settings.market.execution_timeframe, 900)

    def _check_exit_model(row_ctx: pd.Series | None, frames_ctx: dict | None) -> None:
        """Exit open positions early when ExitModel probability exceeds threshold."""
        import numpy as np  # noqa: PLC0415
        if _exit_model is None or not settings.execution.auto_trade or row_ctx is None:
            return
        exit_cfg = settings.execution.exit_model
        atr_val = float(row_ctx.get("atr", 0))
        sl_dist = atr_val * settings.risk.stop_loss_atr_multiple
        if sl_dist <= 0:
            return
        try:
            open_pos_list = executor.get_open_positions(magic_number=settings.execution.magic_number)
        except Exception as _ep_err:
            LOGGER.debug("ExitModel: could not fetch open positions: %s", _ep_err)
            return
        for pos in open_pos_list:
            try:
                ticket = int(pos["ticket"])
                side_sign = 1.0 if pos["side"] == "buy" else -1.0
                entry_price = float(pos["open_price"])
                current_price = float(pos.get("current_price", entry_price))
                unrealized_rr = side_sign * (current_price - entry_price) / sl_dist

                # Gate: minimum profit threshold (unless also_cut_losses is enabled)
                if not exit_cfg.also_cut_losses and unrealized_rr < exit_cfg.min_unrealized_rr:
                    continue
                if exit_cfg.also_cut_losses and unrealized_rr < exit_cfg.min_loss_rr:
                    continue

                # Gate: minimum hold bars
                open_time_epoch = float(pos.get("open_time", time.time()))
                bars_held = int((time.time() - open_time_epoch) / _exec_tf_secs)
                if bars_held < exit_cfg.exit_min_hold_bars:
                    continue

                # Build position-state features
                rsi_at_entry = float(_entry_rsi_tracker.get(ticket, 50.0))
                rsi_now = float(row_ctx.get("rsi", 50.0))
                # rr_momentum: approximate from bar's open→close direction aligned with trade side
                _bar_open  = float(row_ctx.get("open",  current_price))
                _bar_close = float(row_ctx.get("close", current_price))
                pos_state = {
                    "xm_bars_held":           min(bars_held, 32) / 32.0,
                    "xm_unrealized_rr":       float(np.clip(unrealized_rr, -2.0, settings.risk.take_profit_rr + 0.5)),
                    "xm_pos_side":            side_sign,
                    "xm_rsi_at_entry":        rsi_at_entry,
                    "xm_rsi_delta":           rsi_now - rsi_at_entry,
                    "xm_price_vs_entry_atr":  float(np.clip(side_sign * (current_price - entry_price) / max(atr_val, 1e-6), -3.0, 3.0)),
                    "xm_progress_to_tp":      float(np.clip(unrealized_rr / max(settings.risk.take_profit_rr, 0.1), -0.5, 1.5)),
                    "xm_rr_momentum":         float(np.clip(side_sign * (_bar_close - _bar_open) / max(atr_val, 1e-6), -2.0, 2.0)),
                    "xm_bars_remaining":      float(max(0.0, (32 - bars_held) / 32.0)),
                }
                exit_prob = _exit_model.predict(row_ctx, pos_state)
                LOGGER.debug(
                    "ExitModel: ticket=%d side=%s rr=%.2f bars=%d prob=%.3f thr=%.2f",
                    ticket, pos["side"], unrealized_rr, bars_held, exit_prob, _exit_model.threshold,
                )
                if exit_prob >= _exit_model.threshold:
                    executor.close_position(ticket, pos["volume"])
                    _pnl_str = f"{'+' if unrealized_rr >= 0 else ''}{unrealized_rr:.2f}R"
                    notifier.send_message(
                        f"\U0001f3af <b>Exit S\u1edbm #{ticket}</b> ({pos['side'].upper()}) [ExitModel]\n"
                        f"\u251c X\u00e1c su\u1ea5t \u0111\u1ea3o chi\u1ec1u: <b>{exit_prob:.1%}</b>\n"
                        f"\u251c Unrealized R:R: <b>{_pnl_str}</b>\n"
                        f"\u2514 Bars held: {bars_held}"
                    )
                    LOGGER.info(
                        "ExitModel triggered early exit: ticket=%d prob=%.3f rr=%.2f bars=%d",
                        ticket, exit_prob, unrealized_rr, bars_held,
                    )
            except Exception as _pos_err:
                LOGGER.debug("ExitModel: error processing ticket %s: %s", pos.get("ticket"), _pos_err)

    try:
        # ── Đọc public dashboard URL từ tunnel (nếu có) ─────────────────
        _tunnel_url_file = Path("outputs/tunnel_url.txt")
        _dashboard_line = ""
        if _tunnel_url_file.exists():
            try:
                _tunnel_url = _tunnel_url_file.read_text(encoding="utf-8").strip()
                if _tunnel_url:
                    _dashboard_line = f"\n└ Dashboard: {_tunnel_url}"
            except Exception:
                pass
        notifier.send_message(
            f"🟢 <b>Bot khởi động</b> — {settings.market.symbol}\n"
            f"├ Risk: {settings.risk.risk_per_trade*100:.1f}%/lệnh | RR: {settings.risk.take_profit_rr}\n"
            f"├ Live learning: {'BẬT' if settings.training.live_learning_enabled else 'TẮT'}"
            f" ({settings.training.live_learning_interval_minutes}m / min {settings.training.live_learning_min_new_bars} bars)"
            f"{_dashboard_line}"
        )
        while True:
            try:
                # ── Hot-reload config if YAML file changed ─────────────────────
                _maybe_reload_config()

                # ── Tunnel URL watcher: notify Telegram when URL changes ───────
                _tuf = Path("outputs/tunnel_url.txt")
                if _tuf.exists():
                    try:
                        _tuf_mtime = _tuf.stat().st_mtime
                        if _tuf_mtime > _tunnel_url_mtime:
                            _tunnel_url_mtime = _tuf_mtime
                            _new_url = _tuf.read_text(encoding="utf-8").strip()
                            if _new_url and _new_url != _tunnel_url_sent:
                                _tunnel_url_sent = _new_url
                                notifier.send_message(
                                    f"🌐 <b>Dashboard URL mới</b>\n"
                                    f"└ {_new_url}"
                                )
                    except Exception as _tuf_err:
                        LOGGER.debug("Tunnel URL watcher error: %s", _tuf_err)

                # ── Reload model nếu background learner vừa train xong ────────
                if _model_reload_event.is_set():
                    with _model_lock:
                        trainer.load_artifacts()
                    _model_reload_event.clear()
                    LOGGER.info("Model reloaded from background LearnerThread")
                if learner_thread is not None:
                    for _learn_event in learner_thread.drain_results():
                        notifier.send_message(_format_learning_message(_learn_event))

                frames = data_service.fetch_multi_timeframe_data(source=settings.market.live_data_source)
                execution_frame = frames[settings.market.execution_timeframe]
                latest_bar_time = pd.to_datetime(execution_frame.iloc[-1]["time"], utc=True)
                if last_seen_bar_time is None:
                    last_seen_bar_time = latest_bar_time
                elif latest_bar_time > last_seen_bar_time:
                    last_seen_bar_time = latest_bar_time

                # â”€â”€ Láº¥y thÃ´ng tin tÃ i khoáº£n â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                account_info = executor.get_account_info()
                account_balance = account_info.get("balance", 0.0)
                # Fallback: MT5 khong chay trong Docker -> dung peak_balance da luu
                if account_balance == 0.0 and risk_manager._peak_balance > 0:
                    account_balance = risk_manager._peak_balance
                open_positions = executor.get_open_positions_count(
                    magic_number=settings.execution.magic_number
                )

                # ── Self-learning (theo thời gian, mỗi X phút) ──────────────
                if (
                    settings.training.live_learning_enabled
                    and learner_thread is not None
                ):
                    interval_sec = int(settings.training.live_learning_interval_minutes) * 60
                    min_new_bars = max(int(settings.training.live_learning_min_new_bars), 0)
                    now = time.time()
                    if last_learning_bar_time is None:
                        new_bars_since_last_learning = len(execution_frame)
                    else:
                        _exec_times = pd.to_datetime(execution_frame["time"], utc=True, errors="coerce")
                        new_bars_since_last_learning = int((_exec_times > last_learning_bar_time).sum())
                    if (
                        now - last_learning_time >= interval_sec
                        and new_bars_since_last_learning >= min_new_bars
                    ):
                        learner_thread.submit_frames(frames)
                        last_learning_time = now
                        last_learning_bar_time = latest_bar_time
                        LOGGER.debug(
                            "Self-learning: submitted frames to background LearnerThread "
                            "(interval=%d min, new_bars=%d)",
                            interval_sec // 60,
                            new_bars_since_last_learning,
                        )

                # â”€â”€ Build live feature frame (cáº§n ATR cho Trailing SL + DCA) â”€â”€
                live_frame = build_live_feature_frame(settings, frames, strategy)
                latest_row = live_frame.iloc[-1]
                volatility_regime = int(latest_row["volatility_regime"]) if "volatility_regime" in latest_row.index else 1
                atr_value = float(latest_row["atr"]) if "atr" in latest_row.index else 0.0
                # Save fresh context for between-poll close checks
                _last_row_ctx = latest_row
                _last_frames_ctx = frames

                # â”€â”€ Loss Learning â€” phÃ¡t hiá»‡n vÃ  phÃ¢n tÃ­ch lá»‡nh thua â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # -- Loss Learning: detect & notify closed trades immediately
                _check_closed_positions(latest_row, frames)

                # ── Circuit breaker cooldown tick ──────────────────────────────────
                risk_manager.tick_cooldown()

                # â”€â”€ Trailing SL â€” dá»‹ch SL cÃ¡c lá»‡nh Ä‘ang má»Ÿ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if settings.execution.trailing_sl.enabled and settings.execution.auto_trade and atr_value > 0:
                    try:
                        open_pos_list = executor.get_open_positions(
                            magic_number=settings.execution.magic_number
                        )
                        for pos in open_pos_list:
                            new_sl = risk_manager.compute_trailing_sl(pos, atr_value)
                            if new_sl is not None:
                                executor.modify_position_sl(pos["ticket"], new_sl)
                                LOGGER.info(
                                    "TrailingSL: ticket=%d %s old_sl=%.2f â†’ new_sl=%.2f | price=%.2f R=%.2f",
                                    pos["ticket"], pos["side"],
                                    pos["sl"], new_sl, pos["current_price"],
                                    (pos["current_price"] - pos["open_price"]) / (atr_value * settings.risk.stop_loss_atr_multiple)
                                    if pos["side"] == "buy" else
                                    (pos["open_price"] - pos["current_price"]) / (atr_value * settings.risk.stop_loss_atr_multiple),
                                )
                    except Exception as trail_err:
                        LOGGER.error("TrailingSL error: %s", trail_err)

                # ── Partial Take Profit — close partial position at 1R ────────────
                if settings.risk.partial_tp_enabled and settings.execution.auto_trade and atr_value > 0:
                    try:
                        _sl_dist = atr_value * settings.risk.stop_loss_atr_multiple
                        if _sl_dist > 0:
                            _ptp_positions = executor.get_open_positions(
                                magic_number=settings.execution.magic_number
                            )
                            for _ptp_pos in _ptp_positions:
                                _ptp_ticket = int(_ptp_pos["ticket"])
                                _ptp_side = 1.0 if _ptp_pos["side"] == "buy" else -1.0
                                _ptp_entry = float(_ptp_pos["open_price"])
                                _ptp_price = float(_ptp_pos.get("current_price", _ptp_entry))
                                _ptp_rr = _ptp_side * (_ptp_price - _ptp_entry) / _sl_dist
                                _ptp_vol = float(_ptp_pos["volume"])
                                # Check if position has reached partial TP level and has enough volume to split
                                if _ptp_rr >= settings.risk.partial_tp_rr and _ptp_vol >= 0.02:
                                    _close_vol = round(max(0.01, _ptp_vol * settings.risk.partial_tp_pct / 0.01) * 0.01, 2)
                                    _close_vol = min(_close_vol, _ptp_vol - 0.01)  # Keep at least 0.01 lot open
                                    if _close_vol >= 0.01:
                                        executor.close_position(_ptp_ticket, _close_vol)
                                        LOGGER.info(
                                            "PartialTP: ticket=%d closed %.2f lot at %.2fR (%.2f/%.2f)",
                                            _ptp_ticket, _close_vol, _ptp_rr, _close_vol, _ptp_vol,
                                        )
                                        notifier.send_message(
                                            f"\U0001f4b0 <b>Partial TP #{_ptp_ticket}</b> ({_ptp_pos['side'].upper()})\n"
                                            f"\u251c Closed {_close_vol:.2f}/{_ptp_vol:.2f} lot at {_ptp_rr:.1f}R\n"
                                            f"\u2514 Remaining {_ptp_vol - _close_vol:.2f} lot running to full TP"
                                        )
                    except Exception as _ptp_err:
                        LOGGER.error("PartialTP error: %s", _ptp_err)

                # â”€â”€ DCA â€” thÃªm lá»‡nh khi giÃ¡ Ä‘i ngÆ°á»£c â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # Exit Model: check whether open positions should exit early
                _check_exit_model(latest_row, frames)

                if settings.execution.dca.enabled and settings.execution.auto_trade and atr_value > 0:
                    try:
                        dca_state = _load_dca_state()
                        open_pos_list_dca = executor.get_open_positions(
                            magic_number=settings.execution.magic_number
                        )
                        total_lots = sum(p["volume"] for p in open_pos_list_dca)
                        dca_changed = False
                        for pos in open_pos_list_dca:
                            ticket_str = str(pos["ticket"])
                            dca_count = dca_state.get(ticket_str, 0)
                            if risk_manager.should_dca(pos, atr_value, dca_count, account_balance, total_lots):
                                dca_plan = risk_manager.build_dca_plan(pos, atr_value, dca_count, account_balance)
                                dca_result = executor.place_order(dca_plan)
                                dca_state[ticket_str] = dca_count + 1
                                total_lots += dca_plan.volume
                                dca_changed = True
                                LOGGER.info(
                                    "DCA#%d placed: ticket=%d lot=%.2f | %s",
                                    dca_count + 1, pos["ticket"], dca_plan.volume, dca_result,
                                )
                                notifier.send_signal(
                                    type("_D", (), {
                                        "should_trade": True, "side": dca_plan.side,
                                        "confidence": 0.0, "reason": dca_plan.reason,
                                        "entry_price": dca_plan.entry_price,
                                        "stop_loss": dca_plan.stop_loss,
                                        "take_profit": dca_plan.take_profit,
                                    })(),
                                    dca_plan,
                                )
                        open_tickets = {str(p["ticket"]) for p in open_pos_list_dca}
                        dca_state = {k: v for k, v in dca_state.items() if k in open_tickets}
                        if dca_changed:
                            _save_dca_state(dca_state)
                    except Exception as dca_err:
                        LOGGER.error("DCA error: %s", dca_err)

                # â”€â”€ Refresh news cache má»—i news.cache_hours giá» â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if news_crawler is not None:
                    try:
                        news_crawler._load_or_fetch()
                    except Exception:
                        pass

                # â”€â”€ TÃ­nh signal vÃ  decision (dÆ°á»›i lock Ä‘á»ƒ trÃ¡nh race vá»›i reload) â”€â”€
                with _model_lock:
                    signal = trainer.score_live_row(live_frame)
                decision = strategy.build_trade_decision(frames, latest_row, signal)

                # â”€â”€ News filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if decision.should_trade and news_crawler is not None:
                    import datetime as _dt
                    now_utc = _dt.datetime.now(_dt.timezone.utc)
                    news_cfg = settings.integrations.news
                    is_near, news_reason, is_before = news_crawler.is_near_news(
                        now=now_utc,
                        minutes_before=news_cfg.minutes_before,
                        minutes_after=news_cfg.minutes_after,
                        currencies=news_cfg.currencies,
                        high_impact_only=news_cfg.high_impact_only,
                    )
                    if is_near:
                        block = (is_before and not news_cfg.trade_before_news) or \
                                (not is_before and not news_cfg.trade_after_news)
                        if block:
                            LOGGER.info("NEWS BLOCK: %s", news_reason)
                            decision = decision.__class__(
                                should_trade=False,
                                side=decision.side,  # giữ hướng gốc để simulate
                                confidence=decision.confidence,
                                reason=f"NEWS BLOCK: {news_reason}",
                                entry_price=decision.entry_price,
                                stop_loss=decision.stop_loss,
                                take_profit=decision.take_profit,
                            )
                        else:
                            LOGGER.info("NEWS TRADE allowed: trade_before=%s trade_after=%s | %s",
                                        news_cfg.trade_before_news, news_cfg.trade_after_news, news_reason)

                # -- News Trade Override -------------------------------------------------
                # Khi tin vua ra: phan tich actual vs estimate -> ep BUY/SELL ngay.
                # Chay doc lap voi decision.should_trade (ke ca khi ML khong signal).
                _novr_cfg = settings.integrations.news
                if (
                    news_crawler is not None
                    and getattr(_novr_cfg, "news_trade_override", False)
                    and atr_value > 0
                ):
                    import datetime as _dt_novr
                    _novr_now = _dt_novr.datetime.now(_dt_novr.timezone.utc)
                    _novr_window = int(getattr(_novr_cfg, "news_trade_window_minutes", 3))
                    _novr_dir, _novr_title, _novr_reason = news_crawler.get_news_direction(
                        now=_novr_now,
                        window_minutes=_novr_window,
                        currencies=_novr_cfg.currencies,
                        high_impact_only=_novr_cfg.high_impact_only,
                    )
                    if _novr_dir != 0:
                        _novr_side = "buy" if _novr_dir > 0 else "sell"
                        _novr_entry = float(latest_row.get("close", 0.0))
                        if _novr_entry <= 0:
                            _novr_entry = decision.entry_price or 0.0
                        _novr_sl_mult = float(getattr(_novr_cfg, "news_trade_atr_sl_mult", 1.5))
                        _novr_tp_mult = float(getattr(_novr_cfg, "news_trade_atr_tp_mult", 3.5))
                        _novr_sl = round(_novr_entry - _novr_dir * atr_value * _novr_sl_mult, 2)
                        _novr_tp = round(_novr_entry + _novr_dir * atr_value * _novr_tp_mult, 2)
                        LOGGER.info(
                            "NEWS OVERRIDE: '%s' -> %s | entry=%.2f sl=%.2f tp=%.2f ATR=%.2f | %s",
                            _novr_title, _novr_side.upper(),
                            _novr_entry, _novr_sl, _novr_tp, atr_value, _novr_reason,
                        )
                        _novr_dir_label = "📈 BUY" if _novr_dir > 0 else "📉 SELL"
                        notifier.send_message(
                            f"📰 <b>NEWS TRADE: {_novr_title}</b>\n"
                            f"├ Hướng: <b>{_novr_dir_label} GOLD</b>\n"
                            f"├ {_novr_reason}\n"
                            f"└ Entry={_novr_entry:.2f} | SL={_novr_sl:.2f} | TP={_novr_tp:.2f}"
                        )
                        decision = decision.__class__(
                            should_trade=True,
                            side=_novr_side,
                            confidence=0.97,
                            reason=f"NEWS: {_novr_title} | {_novr_reason}",
                            entry_price=_novr_entry,
                            stop_loss=_novr_sl,
                            take_profit=_novr_tp,
                        )

                # â”€â”€ Position gate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                position_allowed, position_reason = risk_manager.can_open_position(
                    account_balance, open_positions, volatility_regime
                )
                max_allowed = risk_manager.get_dynamic_max_positions(account_balance, volatility_regime)

                if decision.should_trade and not position_allowed and not settings.strategy.force_trade:
                    LOGGER.warning(
                        "Position gate BLOCKED: %s | balance=%.2f open=%d max=%d",
                        position_reason, account_balance, open_positions, max_allowed,
                    )
                    decision = decision.__class__(
                        should_trade=False,
                        side=decision.side,  # giữ hướng gốc để simulate
                        confidence=decision.confidence,
                        reason=position_reason,
                        entry_price=decision.entry_price,
                        stop_loss=decision.stop_loss,
                        take_profit=decision.take_profit,
                    )

                # â”€â”€ Build order plan vá»›i dynamic lot size â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                order_plan = risk_manager.build_order_plan(
                    decision,
                    frames[settings.market.execution_timeframe].iloc[-1],
                    account_balance=account_balance if account_balance > 0 else None,
                    current_open_positions=open_positions,
                    volatility_regime=volatility_regime,
                )

                # â”€â”€ Log tÃ­n hiá»‡u cho dashboard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                log_path = Path(settings.app.paper_trade_log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                signal_row = pd.DataFrame([{
                    "time": str(latest_bar_time),
                    "should_trade": decision.should_trade,
                    "side": decision.side,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "entry_price": order_plan.entry_price,
                    "stop_loss": order_plan.stop_loss,
                    "take_profit": order_plan.take_profit,
                    "volume": order_plan.volume,
                    "strategy_score": float(latest_row["strategy_score"]) if "strategy_score" in latest_row.index else 0.0,
                    "volatility_regime": volatility_regime,
                    "account_balance": account_balance,
                    "open_positions": open_positions,
                    "max_positions": max_allowed,
                }])
                # Write with header if file is empty/new, append without header otherwise
                # Skip if same bar was already logged (prevents duplicate on restart)
                if latest_bar_time != _last_logged_bar_time:
                    _write_header = not log_path.exists() or log_path.stat().st_size == 0
                    signal_row.to_csv(log_path, mode="a", header=_write_header, index=False)
                    _last_logged_bar_time = latest_bar_time
                    # PostgreSQL signal insert
                    if _trade_store is not None:
                        try:
                            _trade_store.insert_signal(_account_tag, signal_row.iloc[0].to_dict())
                        except Exception as _pg_sig_exc:
                            LOGGER.debug("PostgreSQL insert_signal failed: %s", _pg_sig_exc)

                # ── Ghi live_status.json để dashboard đọc account state realtime ──
                import datetime as _dtnow
                _is_acc2 = "acc2" in log_path.stem
                _status_path = log_path.parent / ("live_status_acc2.json" if _is_acc2 else "live_status_acc1.json")
                _status_payload = {
                    "ts": _dtnow.datetime.now(_dtnow.timezone.utc).isoformat(),
                    "bar_time": str(latest_bar_time),
                    "account_balance": account_balance,
                    "open_positions": open_positions,
                    "max_positions": max_allowed,
                    "volatility_regime": volatility_regime,
                    "confidence": round(decision.confidence, 4),
                    "should_trade": decision.should_trade,
                    "side": decision.side,
                    "reason": decision.reason,
                }
                try:
                    _status_path.write_text(json.dumps(_status_payload, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
                # PostgreSQL upsert (non-blocking, best-effort)
                if _trade_store is not None:
                    try:
                        _trade_store.upsert_live_status(_account_tag, _status_payload)
                    except Exception as _pg_exc:
                        LOGGER.debug("PostgreSQL upsert_live_status failed: %s", _pg_exc)
                # Prometheus account metrics
                if _trading_metrics is not None:
                    try:
                        _trading_metrics.record_signal(
                            confidence=decision.confidence,
                            should_trade=decision.should_trade,
                            side=decision.side or "none",
                        )
                        _peak_balance = account_balance  # simplified; real drawdown tracked in risk module
                        _trading_metrics.update_account(
                            balance=account_balance,
                            open_positions=open_positions,
                        )
                    except Exception:
                        pass

                # ── Đặt lệnh thật ──────────────────────────────────────────────────────────
                if decision.should_trade:
                    # ── Rebase entry/SL/TP về giá real-time từ MT5 bridge ──────────────────
                    # Nguyên nhân lệch: live_data_source=yfinance có delay ~15 phút.
                    # live_row["close"] = bear M5 bar ~15 phút trước → lệch 10-20 USD.
                    # Fetch tick thật từ MT5 bridge TRƯỚC khi gửi Telegram để notification
                    # và lệnh thực tế khớp nhau.
                    try:
                        _rt_price = executor.get_current_price(order_plan.symbol, order_plan.side)
                        if _rt_price and _rt_price > 0 and order_plan.entry_price > 0:
                            _offset = abs(_rt_price - order_plan.entry_price)
                            if _offset > 0.05:  # chỉ rebase khi lệch đáng kể (>0.05 USD)
                                _sl_dist = abs(order_plan.entry_price - order_plan.stop_loss)
                                _tp_dist = abs(order_plan.take_profit - order_plan.entry_price)
                                if order_plan.side == "sell":
                                    _rebased_sl = round(_rt_price + _sl_dist, 2)
                                    _rebased_tp = round(_rt_price - _tp_dist, 2)
                                else:
                                    _rebased_sl = round(_rt_price - _sl_dist, 2)
                                    _rebased_tp = round(_rt_price + _tp_dist, 2)
                                LOGGER.info(
                                    "Price rebase: yfinance=%.2f → mt5_tick=%.2f (offset=%.2f) | "
                                    "SL %.2f→%.2f TP %.2f→%.2f",
                                    order_plan.entry_price, _rt_price, _offset,
                                    order_plan.stop_loss, _rebased_sl,
                                    order_plan.take_profit, _rebased_tp,
                                )
                                from xauusd_ai.execution.risk import OrderPlan as _OP
                                order_plan = _OP(
                                    symbol=order_plan.symbol,
                                    side=order_plan.side,
                                    volume=order_plan.volume,
                                    entry_price=_rt_price,
                                    stop_loss=_rebased_sl,
                                    take_profit=_rebased_tp,
                                    confidence=order_plan.confidence,
                                    reason=order_plan.reason,
                                )
                    except Exception as _rebase_err:
                        LOGGER.warning("Price rebase failed (using yfinance price): %s", _rebase_err)

                    notifier.send_signal(decision, order_plan)
                    if settings.execution.auto_trade:
                        # ── Chốt lệnh ngược chiều đang lời trước khi vào lệnh mới ──
                        if settings.execution.close_opposite_on_signal:
                            try:
                                _opposite = "sell" if decision.side == "buy" else "buy"
                                _current_positions = executor.get_open_positions(
                                    magic_number=settings.execution.magic_number
                                )
                                for _opos in _current_positions:
                                    if _opos["side"] != _opposite:
                                        continue
                                    _opos_pnl = _opos["profit"] + _opos.get("swap", 0)
                                    _min_profit = settings.execution.close_opposite_min_profit
                                    if _opos_pnl >= _min_profit:
                                        try:
                                            executor.close_position(_opos["ticket"], _opos["volume"])
                                            LOGGER.info(
                                                "CloseOpposite: closed ticket=%d %s profit=%.2f | new signal=%s",
                                                _opos["ticket"], _opos["side"], _opos_pnl, decision.side,
                                            )
                                            notifier.send_message(
                                                f"🔄 <b>Chốt lệnh ngược chiều #{_opos['ticket']}</b> ({_opos['side'].upper()})\n"
                                                f"├ P&L: +{_opos_pnl:.2f}$\n"
                                                f"├ Entry: {_opos['open_price']:.3f} | Lot: {_opos['volume']:.2f}\n"
                                                f"└ Tín hiệu mới: {decision.side.upper()} — chốt lời lệnh ngược chiều"
                                            )
                                        except Exception as _ce:
                                            LOGGER.error("CloseOpposite failed ticket=%d: %s", _opos["ticket"], _ce)
                            except Exception as _oe:
                                LOGGER.error("CloseOpposite scan error: %s", _oe)

                        try:
                            result = executor.place_order(order_plan)
                            LOGGER.info(
                                "MT5 order placed: side=%s lot=%.2f bal=%.2f open=%d/%d | %s",
                                order_plan.side, order_plan.volume,
                                account_balance, open_positions + 1, max_allowed,
                                result,
                            )
                            # Track RSI at entry for exit model position-state features
                            _new_ticket = int(result.get("order", 0) or result.get("deal", 0))
                            if _new_ticket and "rsi" in latest_row.index:
                                _entry_rsi_tracker[_new_ticket] = float(latest_row.get("rsi", 50.0))
                                _save_rsi_tracker(_entry_rsi_tracker)
                        except Exception as order_err:
                            LOGGER.error("MT5 order failed: %s", order_err)
                else:
                    LOGGER.info(
                        "No trade: %s | conf=%.3f | bal=%.2f open=%d/%d atr=%.2f",
                        decision.reason, decision.confidence, account_balance, open_positions, max_allowed, atr_value,
                    )

            except Exception as loop_err:
                LOGGER.error("Live loop error (will retry in %ss): %s", settings.app.poll_seconds, loop_err, exc_info=True)

            # -- Sleep in 30s chunks; check for closed trades between polls
            _close_poll_secs = 30
            _poll_waited = 0
            _poll_total = settings.app.poll_seconds
            while _poll_waited < _poll_total:
                _chunk = min(_close_poll_secs, _poll_total - _poll_waited)
                time.sleep(_chunk)
                _poll_waited += _chunk
                if _poll_waited < _poll_total:  # avoid double-check at loop start
                    _check_closed_positions(_last_row_ctx, _last_frames_ctx)
                    _check_exit_model(_last_row_ctx, _last_frames_ctx)

    finally:
        if learner_thread is not None:
            learner_thread.stop()
            learner_thread.join(timeout=10)
            LOGGER.info("LearnerThread stopped")
        try:
            notifier.send_message("🔴 <b>Bot đã dừng</b> (shutdown / Ctrl+C)")
        except Exception:
            pass


def run_mt5_check(settings: Settings) -> None:
    _, _, _, _, executor, _ = _bootstrap(settings)
    status = executor.connection_status()
    LOGGER.info("MT5 connection status: %s", status)
    print(json.dumps(status, indent=2, default=str))
