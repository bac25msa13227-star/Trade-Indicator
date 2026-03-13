from __future__ import annotations

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
from xauusd_ai.features.dataset import build_live_feature_frame, prepare_training_dataset
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
        self._stop = threading.Event()

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
        self._stop.set()

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------
    def run(self) -> None:
        LOGGER.info("LearnerThread: started (background learning)")
        while not self._stop.is_set():
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

    for (
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
    ) in candidate_grid:
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
        dataset = prepare_training_dataset(candidate_settings, frames, strategy)

        fold_reports: list[dict[str, object]] = []
        fold_trade_frames: list[pd.DataFrame] = []
        train_size = candidate_settings.training.walkforward_train_size
        test_size = candidate_settings.training.walkforward_test_size
        step_size = candidate_settings.training.walkforward_step_size

        for fold_start in range(0, max(len(dataset) - train_size - test_size + 1, 0), step_size):
            train_end = fold_start + train_size
            test_end = train_end + test_size
            fold_train = dataset.iloc[fold_start:train_end].copy()
            fold_test = dataset.iloc[train_end:test_end].copy()
            if len(fold_train) < 200 or len(fold_test) < 50:
                continue

            fold_dataset = pd.concat([fold_train, fold_test], ignore_index=True)
            fold_dataset["split"] = "train"
            fold_dataset.loc[len(fold_train):, "split"] = "test"
            metrics = trainer.train(fold_dataset)
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

    last_seen_bar_time: pd.Timestamp | None = None
    accumulated_new_bars = 0
    # â”€â”€ Loss learning state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _last_closed_check_epoch: float = time.time()
    _known_loss_tickets: set[int] = set()
    _accumulated_losses: int = 0

    try:
        while True:
            try:
                # â”€â”€ Reload model náº¿u background learner vá»«a train xong â”€â”€â”€â”€â”€â”€â”€â”€
                if _model_reload_event.is_set():
                    with _model_lock:
                        trainer.load_artifacts()
                    _model_reload_event.clear()
                    LOGGER.info("Model reloaded from background LearnerThread")

                frames = data_service.fetch_multi_timeframe_data(source=settings.market.live_data_source)
                execution_frame = frames[settings.market.execution_timeframe]
                latest_bar_time = pd.to_datetime(execution_frame.iloc[-1]["time"], utc=True)
                if last_seen_bar_time is None:
                    last_seen_bar_time = latest_bar_time
                elif latest_bar_time > last_seen_bar_time:
                    accumulated_new_bars += int((execution_frame["time"] > last_seen_bar_time).sum())
                    last_seen_bar_time = latest_bar_time

                # â”€â”€ Láº¥y thÃ´ng tin tÃ i khoáº£n â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                account_info = executor.get_account_info()
                account_balance = account_info.get("balance", 0.0)
                open_positions = executor.get_open_positions_count(
                    magic_number=settings.execution.magic_number
                )

                # â”€â”€ Self-learning (song song) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # KhÃ´ng block main loop â€” gá»­i frames cho background thread xá»­ lÃ½
                if (
                    settings.training.live_learning_enabled
                    and learner_thread is not None
                    and accumulated_new_bars >= settings.training.live_learning_min_new_bars
                ):
                    learner_thread.submit_frames(frames)
                    accumulated_new_bars = 0
                    LOGGER.debug("Self-learning: submitted frames to background LearnerThread")

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
                                analysis = self_learner.log_loss_analysis(pos, latest_row)
                                reason_str = " | ".join(analysis.get("reasons", []))
                                LOGGER.warning(
                                    "LOSS detected ticket=%d side=%s pnl=%.2f | %s",
                                    ticket, pos.get("side"), pnl, reason_str,
                                )
                                notifier.send_message(
                                    f"âš ï¸ Lá»‡nh THUA #{ticket} ({pos.get('side','?').upper()}) "
                                    f"P&L={pnl:.2f}$\n"
                                    f"NguyÃªn nhÃ¢n:\n" +
                                    "\n".join(f"â€¢ {r}" for r in analysis.get("reasons", []))
                                )
                            else:
                                LOGGER.info("Position closed in profit: ticket=%d pnl=%.2f", ticket, pnl)

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
                                f"ðŸ”„ Äang há»c láº¡i sau {_accumulated_losses} lá»‡nh thua "
                                f"(background â€” khÃ´ng dá»«ng giao dá»‹ch)"
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
                                side="flat",
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

                if decision.should_trade and not position_allowed:
                    LOGGER.warning(
                        "Position gate BLOCKED: %s | balance=%.2f open=%d max=%d",
                        position_reason, account_balance, open_positions, max_allowed,
                    )
                    decision = decision.__class__(
                        should_trade=False,
                        side="flat",
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
                if log_path.exists():
                    signal_row.to_csv(log_path, mode="a", header=False, index=False)
                else:
                    signal_row.to_csv(log_path, index=False)

                # â”€â”€ Äáº·t lá»‡nh tháº­t â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if decision.should_trade:
                    notifier.send_signal(decision, order_plan)
                    if settings.execution.auto_trade:
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
        # Dá»«ng background learner thread khi main loop káº¿t thÃºc (vÃ­ dá»¥ Ctrl+C)
        if learner_thread is not None:
            learner_thread.stop()
            learner_thread.join(timeout=10)
            LOGGER.info("LearnerThread stopped")


def run_mt5_check(settings: Settings) -> None:
    _, _, _, _, executor, _ = _bootstrap(settings)
    status = executor.connection_status()
    LOGGER.info("MT5 connection status: %s", status)
    print(json.dumps(status, indent=2, default=str))
