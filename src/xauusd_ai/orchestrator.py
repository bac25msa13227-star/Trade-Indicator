from __future__ import annotations

import itertools
import json
import logging
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
from xauusd_ai.model.trainer import ModelTrainer
from xauusd_ai.notifications.telegram import TelegramNotifier
from xauusd_ai.strategies.hybrid import HybridStrategy
from xauusd_ai.visualization.reports import save_backtest_plots, save_training_plot


LOGGER = logging.getLogger(__name__)


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
    frames = data_service.fetch_multi_timeframe_data(source=settings.market.training_data_source)
    _, metrics = _train_on_frames(settings, trainer, strategy, frames)

    LOGGER.info("Training complete: %s", metrics)


def run_backtest(settings: Settings) -> None:
    data_service, trainer, strategy, _, _, risk_manager = _bootstrap(settings)
    frames = data_service.fetch_multi_timeframe_data(source=settings.market.training_data_source)
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
    data_service, trainer, strategy, notifier, executor, risk_manager = _bootstrap(settings)
    model_ready = trainer.load_artifacts()
    if settings.training.retrain_on_startup or not model_ready:
        LOGGER.info("Training artifacts missing or retrain enabled, starting training")
        run_training(settings)
        trainer.load_artifacts()

    last_seen_bar_time: pd.Timestamp | None = None
    accumulated_new_bars = 0

    while True:
        frames = data_service.fetch_multi_timeframe_data(source=settings.market.live_data_source)
        execution_frame = frames[settings.market.execution_timeframe]
        latest_bar_time = pd.to_datetime(execution_frame.iloc[-1]["time"], utc=True)
        if last_seen_bar_time is None:
            last_seen_bar_time = latest_bar_time
        elif latest_bar_time > last_seen_bar_time:
            accumulated_new_bars += int((execution_frame["time"] > last_seen_bar_time).sum())
            last_seen_bar_time = latest_bar_time

        if settings.training.live_learning_enabled and accumulated_new_bars >= settings.training.live_learning_min_new_bars:
            dataset, metrics = _train_on_frames(settings, trainer, strategy, frames)
            accumulated_new_bars = 0
            learning_event = {
                "event": "live_retrain",
                "timestamp": latest_bar_time.isoformat(),
                "rows": len(dataset),
                "train_rows": int(metrics.get("train_rows", 0)),
                "test_rows": int(metrics.get("test_rows", 0)),
                "selected_threshold": metrics.get("selected_threshold"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
            }
            if len(dataset) >= settings.training.live_learning_min_rows:
                _log_live_learning_event(settings, learning_event)
                LOGGER.info("Live learning retrain complete: %s", learning_event)
            else:
                LOGGER.info("Live learning skipped due to insufficient rows: %s", len(dataset))

        live_frame = build_live_feature_frame(settings, frames, strategy)
        signal = trainer.score_live_row(live_frame)
        decision = strategy.build_trade_decision(frames, live_frame.iloc[-1], signal)

        if decision.should_trade:
            order_plan = risk_manager.build_order_plan(decision, frames[settings.market.execution_timeframe].iloc[-1])
            notifier.send_signal(decision, order_plan)
            if settings.execution.auto_trade:
                executor.place_order(order_plan)
        else:
            LOGGER.info("No trade: %s", decision.reason)

        time.sleep(settings.app.poll_seconds)


def run_mt5_check(settings: Settings) -> None:
    _, _, _, _, executor, _ = _bootstrap(settings)
    status = executor.connection_status()
    LOGGER.info("MT5 connection status: %s", status)
    print(json.dumps(status, indent=2, default=str))
