from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class StrictSettingsModel(BaseModel):
    model_config = {"extra": "forbid"}


class AppSettings(StrictSettingsModel):
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
    live_closed_trades_path: str = "outputs/live_closed_trades.csv"
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8501
    log_level: str = "INFO"


class MarketSettings(StrictSettingsModel):
    symbol: str = "XAUUSD"
    training_symbol: str = "XAUUSD=X"
    training_data_source: str = "yfinance"
    live_data_source: str = "mt5"
    csv_data_path: str = "data/xauusd_m15.csv"
    csv_folder_path: str = "src/xauusd_ai/real_data"
    csv_timeframe: str = "M15"
    higher_timeframe: str = "D1"
    mid_timeframe: str = "H1"
    structure_timeframe: str = "H4"      # ICT structure analysis timeframe
    execution_timeframe: str = "M15"
    bars: dict[str, int] = Field(default_factory=lambda: {"D1": 400, "H1": 1000, "H4": 2000, "M15": 3000})
    timezone: str = "UTC"
    # Live market session gating (broker-aware from MT5 tick + sessions)
    enforce_market_open_gate: bool = True
    market_tick_stale_seconds: int = 300
    market_preopen_alert_minutes: int = 30
    market_preclose_alert_minutes: int = 30
    # Multi-milestone alerts (minutes). If provided, takes priority over single-value fields above.
    # Example: [1440, 30] => alert 1 day before and 30 minutes before.
    market_preopen_alert_minutes_list: list[int] = Field(default_factory=lambda: [1440, 30])
    market_preclose_alert_minutes_list: list[int] = Field(default_factory=lambda: [1440, 30])


class StrategyEnabled(StrictSettingsModel):
    ict: bool = True
    wyckoff: bool = True
    order_flow_proxy: bool = True
    rsi: bool = True
    macd: bool = True
    news_filter: bool = True


class RegimeShutdownRuleSettings(StrictSettingsModel):
    enabled: bool = True
    name: str = ""
    weekdays_utc: list[str] = Field(default_factory=list)
    hours_utc: list[int] = Field(default_factory=list)
    sides: list[str] = Field(default_factory=list)
    volatility_regimes: list[int] = Field(default_factory=list)
    trend_alignment_values: list[int] = Field(default_factory=list)
    probability_min: float | None = None
    probability_max: float | None = None
    adx_min: float | None = None
    adx_max: float | None = None
    trend_strength_min: float | None = None
    trend_strength_max: float | None = None
    pullback_quality_min: float | None = None
    pullback_quality_max: float | None = None
    execution_quality_min: float | None = None
    execution_quality_max: float | None = None
    strategy_score_min: float | None = None
    strategy_score_max: float | None = None


class RiskThrottleRuleSettings(StrictSettingsModel):
    enabled: bool = True
    name: str = ""
    risk_multiplier: float = 1.0
    weekdays_utc: list[str] = Field(default_factory=list)
    hours_utc: list[int] = Field(default_factory=list)
    sides: list[str] = Field(default_factory=list)
    volatility_regimes: list[int] = Field(default_factory=list)
    trend_alignment_values: list[int] = Field(default_factory=list)
    probability_min: float | None = None
    probability_max: float | None = None
    adx_min: float | None = None
    adx_max: float | None = None
    trend_strength_min: float | None = None
    trend_strength_max: float | None = None
    pullback_quality_min: float | None = None
    pullback_quality_max: float | None = None
    execution_quality_min: float | None = None
    execution_quality_max: float | None = None
    strategy_score_min: float | None = None
    strategy_score_max: float | None = None


class StrategySettings(StrictSettingsModel):
    enabled: StrategyEnabled = Field(default_factory=StrategyEnabled)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    liquidity_lookback: int = 20
    swing_lookback: int = 10
    volatility_window: int = 20
    signal_threshold: float = 0.58
    sideways_volatility_threshold: float = 0.20   # atr_percentile cutoff (0-1): bottom N% = sideway
    strong_volatility_threshold: float = 0.80    # atr_percentile cutoff (0-1): top N% = strong trend
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
    trend_bypass_confidence: float | None = None  # If set, signals above this confidence bypass trend filter
    force_trade: bool = False  # Bypass ALL filters, trade every signal (for testing)
    blocked_hours_utc: list[int] = Field(default_factory=list)
    blocked_weekdays_utc: list[str] = Field(default_factory=list)
    blocked_weekday_hours_utc: dict[str, list[int]] = Field(default_factory=dict)
    allowed_weekday_hours_utc: dict[str, list[int]] = Field(default_factory=dict)
    # Silver Bullet session windows — high-probability ICT timing (UTC)
    silver_bullet_enabled: bool = False
    silver_bullet_windows_utc: list[list[int]] = Field(
        default_factory=lambda: [[3, 4], [10, 11], [14, 15]]  # Asian, London, NY
    )
    silver_bullet_confidence_boost: float = 0.05  # Boost confidence by 5% during SB windows
    # ADX gate — only trade when trend strength is sufficient
    adx_gate_enabled: bool = False
    adx_min_trend: float = 20.0  # Minimum ADX value to allow entry
    # Regime-specific confidence thresholds (override min_confidence per regime)
    sideway_min_confidence: float = 0.65    # Higher bar in choppy markets
    volatile_min_confidence: float = 0.60   # Moderate bar in volatile markets
    sell_min_confidence: float | None = None  # If set, SELL trades require this confidence (higher = fewer sells)
    buy_min_confidence: float | None = None   # If set, BUY trades require this confidence
    d1_trend_gate: bool = False  # If True, only trade WITH D1 daily_bias direction (block counter-trend)
    regime_shutdown_rules: list[RegimeShutdownRuleSettings] = Field(default_factory=list)


class RiskSettings(StrictSettingsModel):
    mode: str = "fixed_fractional"
    risk_per_trade: float = 0.0075
    max_open_positions: int = 1
    stop_loss_atr_multiple: float = 1.8
    take_profit_rr: float = 2.2
    min_confidence: float = 0.60
    fixed_lot: float = 0.01
    max_risk_fraction: float = 0.015
    sideway_risk_multiplier: float = 0.45
    normal_risk_multiplier: float = 1.0
    strong_volatility_risk_multiplier: float = 0.75
    # ── Regime-adaptive TP/SL multipliers ──────────────────────────────
    # Sideways: SL chặt hơn (1.0× ATR), TP dễ hơn (RR 2.0) — ít room biến động
    sideway_sl_atr_multiple: float = 1.0
    sideway_take_profit_rr: float = 2.0
    # Strong volatile: SL rộng hơn (2.5× ATR), TP lớn hơn (RR 4.0) — cho giá chạy xa
    volatile_sl_atr_multiple: float = 2.5
    volatile_take_profit_rr: float = 4.0
    # Dynamic risk tier: tự điều chỉnh risk theo drawdown từ peak
    # risk_tier_floor = 3% → dùng khi drawdown >= 10% từ peak balance
    # risk_per_trade   = 5% → dùng khi balance ở peak (không có drawdown)
    risk_tier_floor: float = 0.0  # 0 = disabled (flat risk). Set to e.g. 0.03 for 3% floor
    # ── Backtest realism friction ──────────────────────────────────────
    # spread_cost_rr:  spread as fraction of 1R (e.g. 0.10 = spread eats 10% of one R)
    #   XAUUSD typical spread ~30pts, SL ~300pts → 30/300 = 0.10
    spread_cost_rr: float = 0.10
    # slippage_rr:  slippage as fraction of 1R (entry+exit combined)
    slippage_rr: float = 0.05
    # use_dynamic_slippage: enable ATR+spread+volume+session-based slippage calculation
    #   True  → dynamic slippage per trade (realistic, varies 1-6 pips)
    #   False → static slippage_rr applied uniformly (legacy behavior)
    use_dynamic_slippage: bool = False
    # ab_test_enabled: enable A/B testing framework (compare control vs treatment)
    #   True  → randomly assign 50/50 to control (static) vs treatment (dynamic slippage)
    #   False → use use_dynamic_slippage flag directly (no A/B test)
    ab_test_enabled: bool = False
    # ab_test_log_file: path to JSONL log file for A/B test results
    ab_test_log_file: str = "outputs/ab_test_results.jsonl"
    # entry_slippage_atr_frac: shift SL/TP price LEVELS by ATR×frac against trade direction
    # Models MT5 tick fill differing from bar.close (live rebases preserving $ distances).
    # 0.07 ≈ $0.14 slippage for M5 XAUUSD ATR~$2. Reduces win rate ~1-3%.  0.0 = disabled.
    entry_slippage_atr_frac: float = 0.0
    # commission_rr:  broker commission as fraction of 1R per trade
    commission_rr: float = 0.02
    # max_lot:  absolute cap on lot size per trade (broker/account limit)
    #   5.0 → max 5 standard lots per XAUUSD trade (realistic for retail $200–$100k)
    #   0.0 → no cap (unlimited, unrealistic for large compound balances)
    max_lot: float = 0.0
    # compound_cap:  max balance multiplier per fold for sim (0=unlimited)
    #   e.g. 50.0 → balance capped at 50× starting balance per fold
    compound_cap: float = 50.0
    # max_spread_points:  max allowed spread in price points for live order (0=disabled)
    #   XAUUSD ~30 pts normal; reject if > 80 pts (news/off-hours)
    max_spread_points: float = 0.0
    # ── Circuit breaker & capital preservation ─────────────────────
    # Master switch: set to False to bypass ALL circuit breakers (kill switch, daily loss, cooldown)
    kill_switch_enabled: bool = True
    # Daily loss limit: stop trading when cumulative daily loss exceeds N% of starting balance
    daily_loss_limit_pct: float = 0.08    # 8% max daily loss → stop all trading today
    # Max drawdown kill switch: halt trading when drawdown from peak exceeds this
    max_drawdown_kill_pct: float = 0.20   # 20% drawdown → full stop (manual restart needed)
    # Consecutive loss cooldown: skip N bars after M consecutive losses
    consecutive_loss_pause_count: int = 3  # After 3 consecutive losses...
    consecutive_loss_cooldown_bars: int = 8  # ...wait 8 bars (2h on M15) before resuming
    # Anti-martingale: reduce risk_per_trade by this factor after each consecutive loss
    anti_martingale_factor: float = 0.6    # risk *= 0.6 per consecutive loss (compounds)
    anti_martingale_max_reductions: int = 3  # Max 3 reductions (0.6^3 = 21.6% of base)
    # Total exposure cap: max % of balance at risk across all open positions
    max_total_exposure_pct: float = 0.12   # 12% total risk across all open positions
    # ── Volatility-based risk scaling ──────────────────────────────
    # Scale down risk_per_trade when current ATR spikes vs rolling mean (macro event filter)
    volatility_risk_scaling_enabled: bool = False
    vol_atr_lookback_bars: int = 96        # Rolling window for mean ATR (96 M15 bars = 24h)
    vol_atr_spike_ratio: float = 2.0       # If current_atr > 2× mean_atr → scale down
    vol_atr_spike_risk_mult: float = 0.5   # Risk multiplier when spike detected (50%)
    vol_atr_extreme_ratio: float = 3.5     # If current_atr > 3.5× mean_atr → extreme (25%)
    vol_atr_extreme_risk_mult: float = 0.25  # Risk multiplier for extreme spike

    # Anti re-entry guard after SL (same side)
    reentry_guard_enabled: bool = True
    reentry_cooldown_bars_after_sl: int = 1
    reentry_min_distance_atr: float = 0.35
    # Cross-side cooldown: block opposite-side flip for N bars after any SL.
    # Prevents whipsaw BUY-SL → SELL-SL → BUY-SL pattern in sideway markets.
    # 0 = disabled (legacy behavior). Recommended 3-5 bars on M5 timeframe.
    cross_side_reentry_cooldown_bars: int = 0
    # ── Partial Take Profit ────────────────────────────────────────
    partial_tp_enabled: bool = False
    partial_tp_rr: float = 1.0             # Close partial_tp_pct at 1R profit
    partial_tp_pct: float = 0.5            # Close 50% of position at partial_tp_rr
    # Score multiplier: when False, all trades use full risk_per_trade (score_mult=1.0)
    score_multiplier_enabled: bool = True
    # ── Overnight / hold-duration costs ────────────────────────────
    # swap_per_night_rr: negative swap cost per night held as fraction of 1R.
    #   XAUUSD buy swap ≈ -$0.50–$1.50/0.01lot/night. At 1R=$7.50 → ≈ -0.003–0.007 per night.
    #   Sell swap is typically a smaller positive (use 30% of absolute value).
    #   0.0 = disabled (default for backward compat).
    swap_per_night_rr: float = 0.0
    # weekend_gap_penalty_rr: extra friction (negative) applied to trades that span
    #   Fri 22:00 UTC → Sun 22:00 UTC. Models the risk of an adverse gap opening.
    #   XAUUSD avg absolute gap ~$3–$10. In expectation (50% adverse): ~-0.05 to -0.10R.
    #   0.0 = disabled (default for backward compat).
    weekend_gap_penalty_rr: float = 0.0
    # ── Profit Filter ──────────────────────────────────────────────
    # Filter out signals with expected profit < threshold to reduce transaction costs
    profit_filter_enabled: bool = False
    min_expected_profit: float = 15.0      # Minimum expected profit per trade (USD)
    profit_filter_spread_pips: float = 0.5  # XAUUSD spread for profit calculation
    risk_throttle_rules: list[RiskThrottleRuleSettings] = Field(default_factory=list)


class TrailingSlSettings(StrictSettingsModel):
    """Diịch SL động theo giá để bảo vệ lợi nhuậnChức năng:
      1. Breakeven: chuyển SL về hoà vốn sau breakeven_at_rr R lợi nhuận
      2. Trail: diời SL theo giá sau activation_rr R lợi nhuận
    """
    enabled: bool = False
    breakeven_at_rr: float = 0.5       # Chuyển SL về entry sau 0.5R lợi nhuận
    activation_rr: float = 1.0         # Bắt đầu trail SL sau 1R lợi nhuận
    trail_atr_multiple: float = 1.0    # Trail distance = N × ATR


class DcaSettings(StrictSettingsModel):
    """Dollar Cost Averaging — mở thêm lệnh khi giá đi ngược chiều.
    Cảnh báo: DCA tăng exposure, dùng thận trọng."""
    enabled: bool = False
    max_dca_count: int = 2             # Tối đa 2 lần DCA mỗi lệnh
    trigger_atr_multiple: float = 1.5  # DCA khi giá đi ngược N × ATR
    lot_multiplier: float = 1.5        # Lot DCA = lot trước × multiplier
    max_total_risk_pct: float = 0.03   # Tổng risk tối đa 3% balance (an toàn)


class ExitModelSettings(StrictSettingsModel):
    """Exit model — model riêng học khi nào nên chốt lời sớm / cắt lỗ sớm.
    Hoạt động độc lập với entry model, chạy bar-by-bar trên lệnh đang mở.

    Label training:
      Tại mỗi bar j đang giữ lệnh, nhìn ahead exit_label_horizon bars.
      should_exit=1 nếu unrealized_rr[j] - min(rr[j+1..j+H]) > exit_label_threshold_rr
      Tức là: nếu ở lại sẽ bị rút lại > threshold R, nên chốt ngay.

    Features: 38 market features (FEATURE_COLUMNS) + 6 position-state features.
    """
    # ── Master switch ──────────────────────────────────────────────────────
    enabled: bool = False             # Bật sau khi đã train exit model

    # ── Model paths ─────────────────────────────────────────────────────────
    exit_model_path: str = "outputs/exit_model.pkl"
    exit_scaler_path: str = "outputs/exit_scaler.pkl"
    exit_model_meta_path: str = "outputs/exit_model_meta.json"

    # ── Live execution params ─────────────────────────────────────────────
    exit_threshold: float = 0.60       # Xác suất min để trigger early exit
    min_unrealized_rr: float = 0.30    # Chỉ xét chốt khi đã có >= 0.3R lợi nhuận
    exit_min_hold_bars: int = 2        # Không chốt trước N bars (cho trade thở)
    also_cut_losses: bool = False      # Nếu True: cũng cắt lỗ sớm (không chỉ chốt lời)
    min_loss_rr: float = -0.50         # Chỉ cắt lỗ khi loss > -0.5R (nếu also_cut_losses=True)

    # ── Training label params ──────────────────────────────────────────────
    exit_label_horizon: int = 4        # Nhìn N bars ahead để tính label
    exit_label_threshold_rr: float = 0.35  # Bị rút > 0.35R → should_exit=1
    max_entry_samples: int = 6000      # Số entry bars dùng để tạo exit dataset
    include_loss_entries: bool = True  # Thêm target=0 entries để model học cắt lỗ


class ExecutionSettings(StrictSettingsModel):
    # Execution mode: "live" (real trading), "paper" (shadow mode), "backtest" (historical)
    # Paper mode logs all signals and decisions without placing real orders
    mode: str = "live"  # Options: live, paper, backtest
    auto_trade: bool = False
    deviation: int = 20
    magic_number: int = 20260309
    comment: str = "xauusd-ai"
    paper_trade_max_loops: int = 1
    paper_data_source: str = "csv_folder"
    close_opposite_on_signal: bool = False  # Chốt lệnh ngược chiều đang lời khi có tín hiệu mới
    close_opposite_min_profit: float = 1.0  # Minimum profit ($) to close opposite position
    trailing_sl: TrailingSlSettings = Field(default_factory=TrailingSlSettings)
    dca: DcaSettings = Field(default_factory=DcaSettings)
    exit_model: ExitModelSettings = Field(default_factory=ExitModelSettings)


class CanaryDeploySettings(StrictSettingsModel):
    enabled: bool = False
    # 0.10-0.20 is typical canary volume range before full promote.
    volume_fraction: float = 0.20
    min_lot: float = 0.01


class AutoRollbackSettings(StrictSettingsModel):
    enabled: bool = False
    rollback_profile: str = "balanced"
    # Trigger rollback when daily loss exceeds N% of balance (0 = disabled)
    daily_dd_trigger_pct: float = 0.0
    # Trigger rollback when consecutive losses >= N (0 = disabled)
    consecutive_losses_trigger: int = 0
    # Prevent spam rollback notifications
    cooldown_seconds: int = 1800


class DataHealthMonitorSettings(StrictSettingsModel):
    enabled: bool = True
    alert_cooldown_seconds: int = 900
    # Missing bar if actual gap > expected_tf_seconds * factor
    missing_bar_gap_factor: float = 1.8
    # Alert stale tick beyond this age (seconds)
    stale_tick_alert_seconds: int = 300
    # Spread spike alert threshold:
    # current_spread > max(abs_points, median_spread * multiplier)
    spread_spike_multiplier: float = 2.5
    spread_spike_abs_points: float = 1.5
    spread_lookback_bars: int = 50
    # Bridge vs feed divergence alert in price points (USD)
    bridge_feed_divergence_points: float = 1.5
    # Skip divergence checks when market is closed.
    only_when_market_open: bool = True


class AdminSettings(StrictSettingsModel):
    canary: CanaryDeploySettings = Field(default_factory=CanaryDeploySettings)
    auto_rollback: AutoRollbackSettings = Field(default_factory=AutoRollbackSettings)
    data_health: DataHealthMonitorSettings = Field(default_factory=DataHealthMonitorSettings)
    regime_session_matrix_windows_days: list[int] = Field(default_factory=lambda: [7, 30])


class TrainingSettings(StrictSettingsModel):
    train_split: float = 0.8
    train_start_date: str | datetime | None = None
    train_end_date: str | datetime | None = None
    test_start_date: str | datetime | None = None
    test_end_date: str | datetime | None = None
    label_horizon: int = 8
    min_return_threshold: float = 0.0008
    use_sltp_label: bool = True           # SL/TP race label: cleaner targets vs n-bar return
    sltp_label_max_horizon: int = 32      # Max bars forward to scan for TP/SL hit
    label_tp_rr: float = 0.0              # TP RR for labeling (0 = use risk.take_profit_rr)
    retrain_on_startup: bool = True
    live_learning_enabled: bool = True
    live_learning_interval_minutes: int = 30
    live_learning_min_new_bars: int = 12
    live_learning_min_rows: int = 500
    min_self_learning_roc_auc: float = 0.53
    save_dataset: bool = True
    dataset_path: str = "outputs/training_dataset.csv"
    optimize_threshold: bool = True
    validation_split: float = 0.15
    threshold_min: float = 0.35
    threshold_max: float = 0.60
    threshold_step: float = 0.01
    min_precision_floor: float = 0.30
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
    walkforward_max_folds_per_combination: int = 0   # 0 = unlimited
    walkforward_max_avg_drawdown_pct: float = 18.0
    walkforward_min_avg_profit_factor: float = 1.0
    walkforward_min_avg_precision: float = 0.42
    walkforward_min_avg_return_pct: float = 2.0
    walkforward_min_avg_trades: float = 25.0
    backtest_initial_balance: float = 200.0
    # Cap compounding in backtest to prevent astronomical values.
    # Effective balance for PnL is limited to start_bal × this multiplier.
    # 0 = no cap (default). E.g. 200 caps at 200× initial = $40k for $200 start.
    backtest_max_balance_multiplier: float = 200.0
    # Limit bars per timeframe used for training (0 = no limit).
    # Use e.g. 30000 M5 bars (~104 days) to avoid OOM during live startup training.
    max_train_bars: int = 0
    # Use WF-identical ensemble (HGB+RF+ET) instead of single HGB.
    use_ensemble: bool = False
    # Feature selection: drop bottom N% by RF importance. 0 = disabled.
    feature_selection_drop_pct: int = 0
    # Offset applied to the optimized ML threshold at inference time.
    # Negative = accept more signals (lower threshold), positive = stricter.
    ml_threshold_offset: float = 0.0


class NotificationSettings(StrictSettingsModel):
    telegram_enabled: bool = False
    verbose_message: bool = True


class Mt5IntegrationSettings(StrictSettingsModel):
    enabled: bool = True
    login_env: str = "MT5_LOGIN"
    password_env: str = "MT5_PASSWORD"
    server_env: str = "MT5_SERVER"
    # Direct credentials (override env vars when set)
    login: int | None = None
    password: str | None = None
    server: str | None = None
    terminal_path: str | None = None  # Path to MT5 terminal64.exe folder


class TelegramIntegrationSettings(StrictSettingsModel):
    token_env: str = "TELEGRAM_BOT_TOKEN"
    chat_id_env: str = "TELEGRAM_CHAT_ID"


class NewsIntegrationSettings(StrictSettingsModel):
    enabled: bool = False
    provider: str = "forexfactory"   # "forexfactory" | "stub"
    cache_hours: int = 6             # Tự động refresh cache sau N giờ
    trade_before_news: bool = False  # True = cho phép trade trước tin (scalp)
    trade_after_news: bool = False   # True = cho phép trade sau tin (breakout)
    minutes_before: int = 30         # Block N phút trước tin
    minutes_after: int = 30          # Block N phút sau tin
    high_impact_only: bool = True    # Chỉ filter tin High Impact
    currencies: list[str] = Field(default_factory=lambda: ["USD"])  # Tiền tệ liên quan
    # ── News Trade Override ────────────────────────────────────────────────
    # Khi ra tin: phân tích actual vs estimate → BUY/SELL ngay lập tức
    news_trade_override: bool = False    # True = ép lệnh theo hướng tin
    news_trade_window_minutes: int = 3   # Cửa sổ giao dịch sau khi tin ra (phút)
    news_trade_atr_sl_mult: float = 1.5  # SL = N × ATR (xa hơn để tránh spike)
    news_trade_atr_tp_mult: float = 3.5  # TP = N × ATR (RR ~1:2.3)


class IntegrationSettings(StrictSettingsModel):
    mt5: Mt5IntegrationSettings = Field(default_factory=Mt5IntegrationSettings)
    telegram: TelegramIntegrationSettings = Field(default_factory=TelegramIntegrationSettings)
    news: NewsIntegrationSettings = Field(default_factory=NewsIntegrationSettings)


class Settings(StrictSettingsModel):
    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}
    app: AppSettings = Field(default_factory=AppSettings)
    market: MarketSettings = Field(default_factory=MarketSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    admin: AdminSettings = Field(default_factory=AdminSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    integrations: IntegrationSettings = Field(default_factory=IntegrationSettings)
    _config_path: str | None = None


def load_settings(path: Path) -> Settings:
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    s = Settings.model_validate(raw)
    s._config_path = str(path)
    return s
