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


def simulate_prediction_backtest(predictions: pd.DataFrame, settings: Settings, risk_manager: RiskManager) -> SimulationResult:
    test_rows = predictions[predictions["split"] == "test"].copy()
    strategy = HybridStrategy(settings)
    balance = 10000.0
    peak_balance = balance
    wins = 0
    losses = 0
    trades: list[dict[str, object]] = []
    skipped_by_filters = 0

    for row in test_rows.itertuples(index=False):
        if row.prediction == 0:
            continue
        allowed, reason = strategy.should_allow_row(row, float(row.probability))
        if not allowed:
            skipped_by_filters += 1
            continue
        risk_fraction = risk_manager.risk_fraction(
            float(row.probability),
            int(getattr(row, "volatility_regime", 1)),
            float(getattr(row, "strategy_score", 0.0)),
        )
        pnl = balance * risk_fraction * row.realized_rr
        balance_before = balance
        balance += pnl
        peak_balance = max(peak_balance, balance)
        drawdown = (balance - peak_balance) / peak_balance if peak_balance else 0.0
        if pnl >= 0:
            wins += 1
        else:
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
                "is_win": bool(pnl >= 0),
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
        avg_holding_bars = float(predictions.get("holding_bars", pd.Series([settings.training.label_horizon])).mean())
        pnl_std = float(trades_df["pnl"].std()) if len(trades_df) > 1 else 0.0
        sharpe = float((trades_df["pnl"].mean() / pnl_std) * (len(trades_df) ** 0.5)) if pnl_std > 0 else 0.0
        avg_win = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()) if wins > 0 else 0.0
        avg_loss = float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()) if losses > 0 else 0.0

    report = {
        "starting_balance": 10000.0,
        "ending_balance": round(balance, 2),
        "net_profit": round(balance - 10000.0, 2),
        "return_pct": round((balance / 10000.0 - 1) * 100, 2),
        "trades": len(trades),
        "signals_filtered_out": skipped_by_filters,
        "wins": wins,
        "losses": losses,
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