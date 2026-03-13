from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    poll_seconds: int = 60
    model_path: str = "outputs/model.pkl"
    scaler_path: str = "outputs/scaler.pkl"
    model_meta_path: str = "outputs/model_meta.json"
    training_report_path: str = "outputs/training_report.json"
    training_plot_path: str = "outputs/training_metrics.png"
    backtest_report_path: str = "outputs/backtest_report.json"
    backtest_trades_path: str = "outputs/backtest_trades.csv"
    backtest_equity_plot_path: str = "outputs/backtest_equity.png"
    backtest_trades_plot_path: str = "outputs/backtest_trades.png"
    walkforward_report_path: str = "outputs/walkforward_report.json"
    walkforward_trades_path: str = "outputs/walkforward_trades.csv"
    paper_trade_log_path: str = "outputs/paper_trade_signals.csv"
    live_learning_log_path: str = "outputs/live_learning_log.jsonl"
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8501
    log_level: str = "INFO"


class MarketSettings(BaseModel):
    symbol: str = "XAUUSD"
    training_symbol: str = "XAUUSD=X"
    training_data_source: str = "yfinance"
    live_data_source: str = "mt5"
    csv_data_path: str = "data/xauusd_m15.csv"
    csv_folder_path: str = "src/xauusd_ai/real_data"
    csv_timeframe: str = "M15"
    higher_timeframe: str = "D1"
    mid_timeframe: str = "H1"
    execution_timeframe: str = "M15"
    bars: dict[str, int] = Field(default_factory=lambda: {"D1": 400, "H1": 1000, "M15": 3000})
    timezone: str = "UTC"


class StrategyEnabled(BaseModel):
    ict: bool = True
    wyckoff: bool = True
    order_flow_proxy: bool = True
    rsi: bool = True
    macd: bool = True
    news_filter: bool = True


class StrategySettings(BaseModel):
    enabled: StrategyEnabled = Field(default_factory=StrategyEnabled)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    liquidity_lookback: int = 20
    swing_lookback: int = 10
    volatility_window: int = 20
    signal_threshold: float = 0.58
    sideways_volatility_threshold: float = 0.004
    strong_volatility_threshold: float = 0.012
    news_block_minutes: int = 30
    rsi_long_threshold: float = 55.0
    rsi_short_threshold: float = 45.0
    ict_weight: float = 0.25
    wyckoff_weight: float = 0.15
    momentum_weight: float = 0.30
    min_strategy_score: float = 0.1
    sideway_min_strategy_score: float = 0.25
    strong_volatility_min_strategy_score: float = 0.2
    require_trend_alignment: bool = True
    blocked_hours_utc: list[int] = Field(default_factory=list)
    blocked_weekdays_utc: list[str] = Field(default_factory=list)
    blocked_weekday_hours_utc: dict[str, list[int]] = Field(default_factory=dict)
    allowed_weekday_hours_utc: dict[str, list[int]] = Field(default_factory=dict)


class RiskSettings(BaseModel):
    mode: str = "fixed_fractional"
    account_balance: float = 100.0  # reference account size in USD
    risk_per_trade: float = 0.0075  # fraction of balance risked per trade
    # Dynamic concurrent-position management (replaces fixed max_open_positions)
    max_portfolio_risk_fraction: float = 0.03   # max total open risk at any time (e.g. 3%)
    max_concurrent_positions_cap: int = 5       # hard safety cap regardless of budget
    stop_loss_atr_multiple: float = 1.8
    take_profit_rr: float = 2.2
    min_confidence: float = 0.60
    fixed_lot: float = 0.01
    min_lot_size: float = 0.01
    max_lot_size: float = 0.10
    max_risk_fraction: float = 0.015
    sideway_risk_multiplier: float = 0.45
    normal_risk_multiplier: float = 1.0
    strong_volatility_risk_multiplier: float = 0.75


class ExecutionSettings(BaseModel):
    auto_trade: bool = False
    deviation: int = 20
    magic_number: int = 20260309
    comment: str = "xauusd-ai"
    paper_trade_max_loops: int = 1
    paper_data_source: str = "csv_folder"


class TrainingSettings(BaseModel):
    train_split: float = 0.8
    train_start_date: str | datetime | None = None
    train_end_date: str | datetime | None = None
    test_start_date: str | datetime | None = None
    test_end_date: str | datetime | None = None
    label_horizon: int = 8
    min_return_threshold: float = 0.0008
    retrain_on_startup: bool = True
    live_learning_enabled: bool = False
    live_learning_min_new_bars: int = 12
    live_learning_min_rows: int = 500
    save_dataset: bool = True
    dataset_path: str = "outputs/training_dataset.csv"
    optimize_threshold: bool = True
    validation_split: float = 0.15
    threshold_min: float = 0.40
    threshold_max: float = 0.80
    threshold_step: float = 0.02
    min_precision_floor: float = 0.40
    walkforward_train_size: int = 1200
    walkforward_test_size: int = 300
    walkforward_step_size: int = 300
    walkforward_ict_weights: list[float] = Field(default_factory=lambda: [0.25, 0.35, 0.45])
    walkforward_wyckoff_weights: list[float] = Field(default_factory=lambda: [0.15, 0.25, 0.35])
    walkforward_momentum_weights: list[float] = Field(default_factory=lambda: [0.30, 0.40, 0.50])
    walkforward_min_strategy_scores: list[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3])
    walkforward_sideway_min_strategy_scores: list[float] = Field(default_factory=lambda: [0.15, 0.25, 0.35])
    walkforward_strong_volatility_min_strategy_scores: list[float] = Field(default_factory=lambda: [0.15, 0.2, 0.3])
    walkforward_min_confidences: list[float] = Field(default_factory=lambda: [0.55, 0.58, 0.60])
    walkforward_require_trend_alignment: list[bool] = Field(default_factory=lambda: [True, False])
    walkforward_label_horizons: list[int] = Field(default_factory=lambda: [6, 8, 12])
    walkforward_return_thresholds: list[float] = Field(default_factory=lambda: [0.0006, 0.0008, 0.001])
    walkforward_max_combinations: int = 30
    walkforward_max_avg_drawdown_pct: float = 18.0
    walkforward_min_avg_profit_factor: float = 1.0
    walkforward_min_avg_precision: float = 0.42
    walkforward_min_avg_return_pct: float = 2.0
    walkforward_min_avg_trades: float = 25.0


class NotificationSettings(BaseModel):
    telegram_enabled: bool = False
    verbose_message: bool = True


class Mt5IntegrationSettings(BaseModel):
    enabled: bool = True
    login_env: str = "MT5_LOGIN"
    password_env: str = "MT5_PASSWORD"
    server_env: str = "MT5_SERVER"


class TelegramIntegrationSettings(BaseModel):
    token_env: str = "TELEGRAM_BOT_TOKEN"
    chat_id_env: str = "TELEGRAM_CHAT_ID"


class NewsIntegrationSettings(BaseModel):
    enabled: bool = False
    provider: str = "stub"


class IntegrationSettings(BaseModel):
    mt5: Mt5IntegrationSettings = Field(default_factory=Mt5IntegrationSettings)
    telegram: TelegramIntegrationSettings = Field(default_factory=TelegramIntegrationSettings)
    news: NewsIntegrationSettings = Field(default_factory=NewsIntegrationSettings)


class NewsSettings(BaseModel):
    """Settings for the Forex Factory news crawler and scheduler."""

    enabled: bool = True
    # Path to store the cached news CSV
    cache_path: str = "outputs/news_data.csv"
    # Crawler behaviour
    request_delay_seconds: float = 3.0  # polite delay between historical week fetches
    # Scheduler timing
    refresh_interval_seconds: int = 300   # background poll every 5 minutes
    pre_crawl_minutes: int = 15           # crawl N min before a scheduled event
    post_crawl_seconds: int = 90          # crawl N sec after event for Actual value
    min_scheduled_impact: str = "Medium" # minimum impact to schedule event-specific crawls
    # Feature builder windows
    pre_window_minutes: int = 30          # bars within this window before event are tagged
    post_window_minutes: int = 60         # bars within this window after event are tagged
    upcoming_hours: int = 4              # look-ahead for news_upcoming_impact feature
    # Weight of news signal inside the hybrid strategy score
    news_weight: float = 0.20
    # Only include these currencies when building ML features
    relevant_currencies: list[str] = Field(
        default_factory=lambda: ["USD", "EUR", "GBP", "AUD", "CHF", "CAD", "XAU"]
    )
    # FRED (St. Louis Federal Reserve) API key — free at fred.stlouisfed.org/docs/api/api_key.html
    # Leave empty to use the public endpoint (rate-limited but sufficient for historical backfill)
    fred_api_key: str = ""


class Settings(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    market: MarketSettings = Field(default_factory=MarketSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    integrations: IntegrationSettings = Field(default_factory=IntegrationSettings)
    news: NewsSettings = Field(default_factory=NewsSettings)


def load_settings(path: Path) -> Settings:
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(raw)
