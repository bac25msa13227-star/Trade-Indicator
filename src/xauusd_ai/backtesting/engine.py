from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from xauusd_ai.config import Settings
from xauusd_ai.execution.risk import RiskManager
from xauusd_ai.strategies.hybrid import HybridStrategy
from xauusd_ai.visualization.reports import save_backtest_plots


@dataclass
class SimulationResult:
    report: dict[str, object]
    trades: pd.DataFrame


def simulate_prediction_backtest(
    predictions: pd.DataFrame,
    settings: Settings,
    risk_manager: RiskManager,
    compound: bool = True,
) -> SimulationResult:
    """
    compound=True  → balance compounds with each trade (for short windows like walkforward folds).
    compound=False → PnL always uses start_bal (linear/non-compounding — for long-window backtest
                     display to avoid astronomical numbers from high-WR strategies).
    """
    test_rows = predictions[predictions["split"] == "test"].copy()
    strategy = HybridStrategy(settings)
    start_bal = settings.training.backtest_initial_balance
    balance = start_bal
    peak_balance = balance
    wins = 0
    losses = 0
    trades: list[dict[str, object]] = []
    skipped_by_filters = 0

    # Friction components (fraction of 1R deducted per trade)
    _spread_rr = float(getattr(settings.risk, "spread_cost_rr", 0.10))
    _slippage_rr = float(getattr(settings.risk, "slippage_rr", 0.05))
    _commission_rr = float(getattr(settings.risk, "commission_rr", 0.02))
    friction_rr = _spread_rr + _slippage_rr + _commission_rr
    compound_cap = float(getattr(settings.risk, "compound_cap", 50.0))
    max_balance = start_bal * compound_cap if compound_cap > 0 else float("inf")

    for row in test_rows.itertuples(index=False):
        if row.prediction == 0:
            continue
        allowed, reason = strategy.should_allow_row(row, float(row.probability))
        if not allowed:
            skipped_by_filters += 1
            continue
        # Fixed fractional: use exact risk_per_trade (no regime/score scaling in backtest).
        risk_fraction = settings.risk.risk_per_trade
        effective_bal = balance if compound else start_bal
        if compound and compound_cap > 0 and effective_bal > max_balance:
            effective_bal = max_balance
        # P1a: Session-aware friction
        _sess_mult = float(getattr(row, 'session_spread_mult', 1.0))
        _row_friction = _spread_rr * _sess_mult + _slippage_rr + _commission_rr
        net_rr = row.realized_rr - _row_friction
        pnl = effective_bal * risk_fraction * net_rr
        balance_before = balance
        balance += pnl
        peak_balance = max(peak_balance, balance)
        drawdown = (balance - peak_balance) / peak_balance if peak_balance else 0.0
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        trades.append(
            {
                "time": row.time.isoformat() if hasattr(row.time, "isoformat") else str(row.time),
                "side": row.trade_side,
                "entry_price": float(row.close),
                "exit_price": float(row.close * (1 + row.future_return)),
                "future_return": float(row.future_return),
                "directional_return": float(row.directional_return),
                "realized_rr": float(row.realized_rr),
                "probability": float(row.probability),
                "risk_fraction": float(risk_fraction),
                "pnl": float(pnl),
                "balance_before": float(balance_before),
                "balance_after": float(balance),
                "drawdown": float(drawdown),
                "is_win": bool(pnl > 0),
                "is_loss": bool(pnl < 0),
                "is_draw": bool(pnl == 0),
                "skip_filter_reason": reason,
            }
        )

    trades_df = pd.DataFrame(trades)
    max_drawdown = 0.0 if trades_df.empty else float(trades_df["drawdown"].min())
    profit_factor = 0.0
    avg_holding_bars = 0.0
    sharpe = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    if not trades_df.empty:
        gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        avg_holding_bars = float(predictions.get("bars_held", pd.Series([settings.training.label_horizon])).mean())
        pnl_std = float(trades_df["pnl"].std()) if len(trades_df) > 1 else 0.0
        sharpe = float((trades_df["pnl"].mean() / pnl_std) * (len(trades_df) ** 0.5)) if pnl_std > 0 else 0.0
        avg_win = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()) if wins > 0 else 0.0
        avg_loss = float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()) if losses > 0 else 0.0

    start_bal = settings.training.backtest_initial_balance
    draws = len(trades) - wins - losses
    report = {
        "starting_balance": start_bal,
        "ending_balance": round(balance, 2),
        "net_profit": round(balance - start_bal, 2),
        "return_pct": round((balance / start_bal - 1) * 100, 2),
        "trades": len(trades),
        "signals_filtered_out": skipped_by_filters,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(wins / max(wins + losses, 1), 4),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "best_trade": round(float(trades_df["pnl"].max()) if not trades_df.empty else 0.0, 2),
        "worst_trade": round(float(trades_df["pnl"].min()) if not trades_df.empty else 0.0, 2),
        "sharpe_like": round(sharpe, 4),
        "avg_holding_bars": round(avg_holding_bars, 2),
        "friction_rr": round(friction_rr, 4),
        "compound_cap": compound_cap,
    }
    return SimulationResult(report=report, trades=trades_df)


def simulate_dynamic_concurrent_backtest(
    predictions: pd.DataFrame,
    settings: Settings,
    risk_manager: RiskManager,
    label: str = "test",
    compound: bool = True,
) -> SimulationResult:
    """
    Concurrent position simulation with dynamic slot scaling and realistic friction.

    Realism layers (configurable via RiskSettings):
      - spread_cost_rr:  spread deducted per trade as fraction of 1R
      - slippage_rr:     slippage deducted per trade as fraction of 1R
      - commission_rr:   commission deducted per trade as fraction of 1R
      - compound_cap:    max balance = start_bal × compound_cap (prevents explosion)

    compound=True  → balance compounds per trade (default — use for short windows / walkforward).
    compound=False → PnL always uses start_bal (linear / non-compounding — use for long-window
                     backtest display to avoid astronomical numbers from high-WR strategies).
    """
    test_rows = (
        predictions[predictions["split"] == label]
        .copy()
        .sort_values("time")
        .reset_index(drop=True)
    )
    strategy = HybridStrategy(settings)
    start_bal = settings.training.backtest_initial_balance
    balance = start_bal
    peak_balance = balance
    label_horizon = settings.training.label_horizon  # bars to hold each position

    # Friction components (fraction of 1R deducted per trade)
    _spread_rr = float(getattr(settings.risk, "spread_cost_rr", 0.10))
    _slippage_rr = float(getattr(settings.risk, "slippage_rr", 0.05))
    _commission_rr = float(getattr(settings.risk, "commission_rr", 0.02))
    friction_rr = _spread_rr + _slippage_rr + _commission_rr
    compound_cap = float(getattr(settings.risk, "compound_cap", 50.0))
    max_balance = start_bal * compound_cap if compound_cap > 0 else float("inf")

    # pending: list of {"close_bar_idx": int, "absolute_pnl": float, "meta": dict}
    pending: list[dict] = []
    trades: list[dict] = []
    wins = 0
    losses = 0
    skipped_by_filters = 0
    skipped_no_slot = 0

    balance_history: list[float] = [balance]
    max_concurrent_seen = 0

    for i, row in enumerate(test_rows.itertuples(index=False)):
        # ── close matured positions ────────────────────────────────────────
        still_pending: list[dict] = []
        for entry in pending:
            if i >= entry["close_bar_idx"]:
                pnl = entry["absolute_pnl"]
                balance += pnl
                peak_balance = max(peak_balance, balance)
                dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
                entry["meta"]["balance_after"] = round(balance, 4)
                entry["meta"]["drawdown"] = round(-dd, 4)
                entry["meta"]["is_win"] = bool(pnl > 0)
                entry["meta"]["is_loss"] = bool(pnl < 0)
                entry["meta"]["is_draw"] = bool(pnl == 0)
                trades.append(entry["meta"])
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1
            else:
                still_pending.append(entry)
        pending = still_pending

        if row.prediction == 0:
            balance_history.append(balance)
            continue

        # ── strategy filter gate ───────────────────────────────────────────
        allowed, reason = strategy.should_allow_row(row, float(row.probability))
        if not allowed:
            skipped_by_filters += 1
            balance_history.append(balance)
            continue

        # ── dynamic slot check ─────────────────────────────────────────────
        regime = int(getattr(row, "volatility_regime", 1))
        max_pos = risk_manager.get_dynamic_max_positions(balance, regime)
        if len(pending) >= max_pos:
            skipped_no_slot += 1
            balance_history.append(balance)
            continue

        # ── open new position ──────────────────────────────────────────────
        # Dynamic risk tier: scale between floor and ceiling based on drawdown from peak.
        # risk_tier_floor = 3% = minimum during drawdown; risk_per_trade = 5% = max at peak.
        rf_base  = float(settings.risk.risk_per_trade)
        rf_floor = float(getattr(settings.risk, 'risk_tier_floor', 0.0))
        if rf_floor > 0 and peak_balance > start_bal:
            dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
            if dd >= 0.10:              # >= 10% drawdown: use floor (protection mode)
                rf = rf_floor
            elif dd >= 0.03:            # 3-10% drawdown: linear interpolation
                ratio = (dd - 0.03) / 0.07
                rf = rf_base - ratio * (rf_base - rf_floor)
            else:
                rf = rf_base            # < 3% drawdown: full risk
        else:
            rf = rf_base
        # compound=False: always use starting balance -> linear expectancy (no explosion).
        effective_bal = balance if compound else start_bal
        # Apply compound cap: prevent unrealistic exponential growth
        if compound and compound_cap > 0 and effective_bal > max_balance:
            effective_bal = max_balance
        # P1a: Session-aware friction
        _sess_mult = float(getattr(row, 'session_spread_mult', 1.0))
        _row_friction = _spread_rr * _sess_mult + _slippage_rr + _commission_rr
        raw_rr = float(row.realized_rr)
        net_rr = raw_rr - _row_friction
        absolute_pnl = effective_bal * rf * net_rr
        open_count = len(pending)
        max_concurrent_seen = max(max_concurrent_seen, open_count + 1)

        meta: dict = {
            "time": row.time.isoformat() if hasattr(row.time, "isoformat") else str(row.time),
            "side": row.trade_side,
            "entry_price": float(row.close),
            "realized_rr": float(row.realized_rr),
            "net_rr": float(net_rr),
            "friction_rr": float(friction_rr),
            "probability": float(row.probability),
            "risk_fraction": float(rf),
            "pnl": round(absolute_pnl, 4),
            "balance_before": round(balance, 4),
            "balance_after": None,  # filled at close
            "drawdown": None,       # filled at close
            "is_win": None,         # filled at close (pnl > 0)
            "is_loss": None,        # filled at close (pnl < 0)
            "is_draw": None,        # filled at close (pnl == 0)
            "max_positions_allowed": max_pos,
            "open_positions_at_open": open_count,
            "volatility_regime": regime,
        }
        # P0: Use per-trade bars_held from SL/TP race (fallback to label_horizon)
        _hold = int(getattr(row, 'bars_held', label_horizon))
        pending.append(
            {"close_bar_idx": i + _hold, "absolute_pnl": absolute_pnl, "meta": meta}
        )
        balance_history.append(balance)

    # ── close any remaining open positions at period end ───────────────────
    for entry in pending:
        pnl = entry["absolute_pnl"]
        balance += pnl
        peak_balance = max(peak_balance, balance)
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
        entry["meta"]["balance_after"] = round(balance, 4)
        entry["meta"]["drawdown"] = round(-dd, 4)
        entry["meta"]["is_win"] = bool(pnl > 0)
        entry["meta"]["is_loss"] = bool(pnl < 0)
        entry["meta"]["is_draw"] = bool(pnl == 0)
        trades.append(entry["meta"])
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

    # ── build report ───────────────────────────────────────────────────────
    trades_df = pd.DataFrame(trades)
    start_bal = settings.training.backtest_initial_balance

    gross_profit = gross_loss = 0.0
    avg_win = avg_loss = 0.0
    if not trades_df.empty:
        gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
        gross_loss   = float(-trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())
        avg_win  = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()) if wins > 0 else 0.0
        avg_loss = float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()) if losses > 0 else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    max_drawdown = 0.0 if trades_df.empty else float(trades_df["drawdown"].min())
    pnl_std = float(trades_df["pnl"].std()) if not trades_df.empty and len(trades_df) > 1 else 0.0
    sharpe = (
        float((trades_df["pnl"].mean() / pnl_std) * (len(trades_df) ** 0.5))
        if pnl_std > 0 else 0.0
    )

    # Position tier breakdown (how many trades were opened per balance tier)
    tier_breakdown: dict[str, int] = {}
    if not trades_df.empty and "balance_before" in trades_df.columns:
        for _, tr in trades_df.iterrows():
            b = float(tr["balance_before"])
            if b < 500:
                tier = "<$500 (3 slots)"
            elif b < 2_000:
                tier = "$500-$2k (5 slots)"
            elif b < 10_000:
                tier = "$2k-$10k (8 slots)"
            elif b < 50_000:
                tier = "$10k-$50k (10 slots)"
            else:
                tier = ">$50k (15 slots)"
            tier_breakdown[tier] = tier_breakdown.get(tier, 0) + 1

    avg_concurrent = (
        float(trades_df["open_positions_at_open"].mean() + 1)
        if not trades_df.empty and "open_positions_at_open" in trades_df.columns
        else 0.0
    )

    report = {
        "simulation_mode": "dynamic_concurrent",
        "friction_rr": round(friction_rr, 4),
        "compound_cap": compound_cap,
        "starting_balance": start_bal,
        "ending_balance": round(balance, 2),
        "net_profit": round(balance - start_bal, 2),
        "return_pct": round((balance / start_bal - 1) * 100, 2),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "draws": len(trades) - wins - losses,
        "win_rate": round(wins / max(wins + losses, 1), 4),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "best_trade": round(float(trades_df["pnl"].max()) if not trades_df.empty else 0.0, 2),
        "worst_trade": round(float(trades_df["pnl"].min()) if not trades_df.empty else 0.0, 2),
        "sharpe_like": round(sharpe, 4),
        "signals_filtered_out": skipped_by_filters,
        "signals_no_slot": skipped_no_slot,
        "max_concurrent_positions": max_concurrent_seen,
        "avg_concurrent_positions": round(avg_concurrent, 2),
        "position_tier_breakdown": tier_breakdown,
    }
    return SimulationResult(report=report, trades=trades_df)


def write_backtest_outputs(
    report: dict[str, object],
    trades_df: pd.DataFrame,
    test_rows: pd.DataFrame,
    report_path: Path,
    trades_path: Path,
    equity_plot_path: Path,
    trades_plot_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)
    save_backtest_plots(test_rows, trades_df, equity_plot_path, trades_plot_path)