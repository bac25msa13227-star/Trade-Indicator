from __future__ import annotations

import datetime
import html
import itertools
import json
import logging
import queue
import threading
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from xauusd_ai.backtesting.engine import simulate_prediction_backtest, write_backtest_outputs
from xauusd_ai.config import Settings
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

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------
    def run(self) -> None:
        LOGGER.info("LearnerThread: started (background learning)")
        while not self._stop_event.is_set():
            # Æ¯u tiÃªn loss-retrain trÆ°á»›c (ngáº¯n hÆ¡n regular retrain)
            try:
                loss_count, frames = self._loss_q.get_nowait()
                LOGGER.info("LearnerThread: loss-retrain triggered (losses=%d)", loss_count)
                result = self._self_learner.maybe_retrain_on_loss(frames, loss_count)
                if result:
                    self._reload_event.set()
                    LOGGER.info("LearnerThread: loss-retrain done â†’ signal reload | %s", result.get("status"))
                continue  # kiá»ƒm tra loss_q láº¡i ngay
            except queue.Empty:
                pass

            # Regular self-learning (block tá»‘i Ä‘a 2s Ä‘á»ƒ khÃ´ng spin)
            try:
                frames = self._frames_q.get(timeout=2.0)
                LOGGER.info("LearnerThread: regular retrain started")
                result = self._self_learner.maybe_retrain(frames)
                if result:
                    self._reload_event.set()
                    LOGGER.info("LearnerThread: retrain done â†’ signal reload | %s", result.get("status"))
            except queue.Empty:
                pass

        LOGGER.info("LearnerThread: stopped")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _bootstrap(settings: Settings) -> tuple[MarketDataService, ModelTrainer, HybridStrategy, TelegramNotifier, MT5Executor, RiskManager]:
    load_dotenv()
    _configure_logging(settings.app.log_level)
    data_service = MarketDataService(settings)
    trainer = ModelTrainer(settings)
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


def run_training(settings: Settings) -> None:
    data_service, trainer, strategy, _, _, _ = _bootstrap(settings)
    frames = data_service.fetch_multi_timeframe_data(source=settings.market.training_data_source, all_bars=True)
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
    frames = data_service.fetch_multi_timeframe_data(source=settings.market.training_data_source)

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
    _wf_started_at = datetime.datetime.now()

    def _write_progress(
        completed: int,
        status: str = "running",
        current_params: dict | None = None,
        best_so_far: dict | None = None,
    ) -> None:
        elapsed = (datetime.datetime.now() - _wf_started_at).total_seconds()
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
                "last_updated": datetime.datetime.now().isoformat(),
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
    model_ready = trainer.load_artifacts()
    if settings.training.retrain_on_startup or not model_ready:
        LOGGER.info("Training artifacts missing or retrain enabled, starting training")
        run_training(settings)
        trainer.load_artifacts()

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
        _dca_state_path.write_text(json.dumps(state), encoding="utf-8")

    # ── Live closed trades log ────────────────────────────────────────────────
    _live_trades_path = Path(settings.app.live_closed_trades_path)
    _live_trades_path.parent.mkdir(parents=True, exist_ok=True)    # Reset closed trades + signals CSV at each session start — dashboard chỉ hiển thị session hiện tại
    _live_trades_header = "time,ticket,side,volume,open_price,close_price,profit,swap,commission,pnl,is_win\n"
    _live_trades_path.write_text(_live_trades_header, encoding="utf-8")
    _signals_path = Path(settings.app.paper_trade_log_path)
    _signals_path.parent.mkdir(parents=True, exist_ok=True)
    _signals_path.write_text("", encoding="utf-8")  # clear stale signals from previous sessions
    def _append_live_trade(pos: dict, pnl: float, is_win: bool) -> None:
        """Ghi lệnh đóng vào CSV để dashboard P&L đọc được."""
        import datetime as _dt
        row = pd.DataFrame([{
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
        }])
        if _live_trades_path.exists():
            row.to_csv(_live_trades_path, mode="a", header=False, index=False)
        else:
            row.to_csv(_live_trades_path, index=False)

    last_seen_bar_time: pd.Timestamp | None = None
    last_learning_time: float = time.time()
    # â”€â”€ Loss learning state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _last_closed_check_epoch: float = time.time() - 300  # look back 5 min on startup
    _known_loss_tickets: set[int] = set()
    _accumulated_losses: int = 0

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
            f"{_dashboard_line}"
        )
        while True:
            try:
                # ── Reload model nếu background learner vừa train xong ────────
                if _model_reload_event.is_set():
                    with _model_lock:
                        trainer.load_artifacts()
                    _model_reload_event.clear()
                    LOGGER.info("Model reloaded from background LearnerThread")
                    notifier.send_message(
                        f"🧠 <b>Model cập nhật</b> từ live learning\n"
                        f"Lần retrain #{self_learner._retrain_count}"
                    )

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
                open_positions = executor.get_open_positions_count(
                    magic_number=settings.execution.magic_number
                )

                # ── Self-learning (theo thời gian, mỗi X phút) ──────────────
                if (
                    settings.training.live_learning_enabled
                    and learner_thread is not None
                ):
                    interval_sec = int(getattr(settings.training, "live_learning_interval_minutes", 30)) * 60
                    now = time.time()
                    if now - last_learning_time >= interval_sec:
                        learner_thread.submit_frames(frames)
                        last_learning_time = now
                        LOGGER.debug(f"Self-learning: submitted frames to background LearnerThread (interval {interval_sec//60} min)")

                # â”€â”€ Build live feature frame (cáº§n ATR cho Trailing SL + DCA) â”€â”€
                live_frame = build_live_feature_frame(settings, frames, strategy)
                latest_row = live_frame.iloc[-1]
                volatility_regime = int(latest_row["volatility_regime"]) if "volatility_regime" in latest_row.index else 1
                atr_value = float(latest_row["atr"]) if "atr" in latest_row.index else 0.0

                # â”€â”€ Loss Learning â€” phÃ¡t hiá»‡n vÃ  phÃ¢n tÃ­ch lá»‡nh thua â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if settings.execution.auto_trade:
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
                            if pnl < 0:
                                _accumulated_losses += 1
                                self_learner._accumulated_losses = _accumulated_losses
                                LOGGER.warning(
                                    "LOSS detected ticket=%d side=%s pnl=%.2f",
                                    ticket, pos.get("side"), pnl,
                                )
                                # Run analysis separately — never let it block the notification
                                _feat: dict = {}
                                _reasons: list[str] = []
                                try:
                                    analysis = self_learner.log_loss_analysis(pos, latest_row)
                                    _feat = analysis.get("features", {})
                                    _reasons = analysis.get("reasons", [])
                                    LOGGER.warning("LOSS reasons: %s", " | ".join(_reasons))
                                except Exception as _ae:
                                    LOGGER.error("Loss analysis failed (notification still sent): %s", _ae)
                                _regime_label = {0: "sideway", 1: "normal", 2: "volatile"}.get(
                                    int(_feat.get("volatility_regime", 1)), "?"
                                )
                                _reasons_text = "\n".join(
                                    f"  \u2022 {html.escape(str(r))}" for r in _reasons
                                ) if _reasons else "  • Khong xac dinh ro nguyen nhan"
                                notifier.send_message(
                                    f"\u26a0\ufe0f <b>Lenh THUA #{ticket}</b> ({pos.get('side','?').upper()}) \u2014 Loss #{_accumulated_losses}\n"
                                    f"\u251c P&amp;L: <b>{pnl:.2f}$</b>\n"
                                    f"\u251c Entry: {pos.get('open_price', 0):.5f} \u2192 Close: {pos.get('close_price', 0):.5f}\n"
                                    f"\u251c Volume: {pos.get('volume', 0):.2f} lot\n"
                                    f"\u251c ATR: {_feat.get('atr', 0):.2f} | RSI: {_feat.get('rsi', 0):.1f} | Score: {_feat.get('strategy_score', 0):.3f}\n"
                                    f"\u251c Regime: {_regime_label} | Trend: {'OK' if int(_feat.get('trend_alignment', 0)) == 1 else 'MISS'}\n"
                                    f"\u2514 Nguyen nhan:\n{_reasons_text}"
                                )
                                _append_live_trade(pos, pnl, is_win=False)
                            else:
                                LOGGER.info("Position closed in profit: ticket=%d pnl=%.2f", ticket, pnl)
                                notifier.send_message(
                                    f"✅ <b>Lenh THANG #{ticket}</b> ({pos.get('side','?').upper()})\n"
                                    f"├ P&amp;L: <b>+{pnl:.2f}$</b>\n"
                                    f"├ Entry: {pos.get('open_price', 0):.5f} → Close: {pos.get('close_price', 0):.5f}\n"
                                    f"├ Volume: {pos.get('volume', 0):.2f} lot\n"
                                    f"└ Profit: {pos.get('profit', 0):.2f}$ | Swap: {pos.get('swap', 0):.2f}$ | Comm: {pos.get('commission', 0):.2f}$"
                                )
                                _append_live_trade(pos, pnl, is_win=True)

                        # Trigger loss-retrain (song song) sau LOSS_RETRAIN_THRESHOLD lá»‡nh thua
                        if (
                            _accumulated_losses >= self_learner.LOSS_RETRAIN_THRESHOLD
                            and learner_thread is not None
                        ):
                            LOGGER.info(
                                "LossRetrain queued for background thread (losses=%d)", _accumulated_losses
                            )
                            learner_thread.submit_loss(_accumulated_losses, frames)
                            notifier.send_message(
                                f"🔄 <b>Đang học lại</b> sau {_accumulated_losses} lệnh thua\n"
                                f"(background – không dừng giao dịch)"
                            )
                            _accumulated_losses = 0
                            self_learner._accumulated_losses = 0
                    except Exception as loss_err:
                        LOGGER.error("Loss learning error: %s", loss_err)

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

                # â”€â”€ DCA â€” thÃªm lá»‡nh khi giÃ¡ Ä‘i ngÆ°á»£c â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                _write_header = not log_path.exists() or log_path.stat().st_size == 0
                signal_row.to_csv(log_path, mode="a", header=_write_header, index=False)

                # ── Ghi live_status.json để dashboard đọc account state realtime ──
                import datetime as _dtnow
                _is_acc2 = "acc2" in log_path.stem
                _status_path = log_path.parent / ("live_status_acc2.json" if _is_acc2 else "live_status_acc1.json")
                try:
                    _status_path.write_text(json.dumps({
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
                    }, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass

                # ── Đặt lệnh thật ──────────────────────────────────────────────────────────
                if decision.should_trade:
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
                                    if _opos_pnl > 0:
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
                        except Exception as order_err:
                            LOGGER.error("MT5 order failed: %s", order_err)
                else:
                    LOGGER.info(
                        "No trade: %s | bal=%.2f open=%d/%d atr=%.2f",
                        decision.reason, account_balance, open_positions, max_allowed, atr_value,
                    )

            except Exception as loop_err:
                LOGGER.error("Live loop error (will retry in %ss): %s", settings.app.poll_seconds, loop_err, exc_info=True)

            time.sleep(settings.app.poll_seconds)

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
