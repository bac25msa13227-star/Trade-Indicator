from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_training_plot(metrics: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [metrics.get(label, 0.0) for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color=["#c49a3a", "#4d7c8a", "#6b9a62", "#b55d3d", "#2f4858"])
    ax.set_ylim(0, 1)
    ax.set_title("Training Metrics")
    ax.set_ylabel("Score")
    for index, value in enumerate(values):
        ax.text(index, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_backtest_plots(
    test_rows: pd.DataFrame,
    trades_df: pd.DataFrame,
    equity_output_path: Path,
    trades_output_path: Path,
) -> None:
    equity_output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not trades_df.empty:
        equity_fig, equity_ax = plt.subplots(figsize=(12, 6))
        equity_ax.plot(pd.to_datetime(trades_df["time"]), trades_df["balance_after"], color="#2f4858", linewidth=2)
        equity_ax.fill_between(pd.to_datetime(trades_df["time"]), trades_df["balance_after"], color="#c49a3a", alpha=0.2)
        equity_ax.set_title("Backtest Equity Curve")
        equity_ax.set_ylabel("Balance")
        equity_ax.set_xlabel("Time")
        equity_fig.autofmt_xdate()
        equity_fig.tight_layout()
        equity_fig.savefig(equity_output_path)
        plt.close(equity_fig)

    price_fig, price_ax = plt.subplots(figsize=(14, 7))
    price_ax.plot(pd.to_datetime(test_rows["time"]), test_rows["close"], color="#444444", linewidth=1.2, label="Price")
    if not trades_df.empty:
        buy_trades = trades_df[trades_df["side"] == "buy"]
        sell_trades = trades_df[trades_df["side"] == "sell"]
        if not buy_trades.empty:
            price_ax.scatter(pd.to_datetime(buy_trades["time"]), buy_trades["entry_price"], color="#1f7a1f", marker="^", s=40, label="Buy")
        if not sell_trades.empty:
            price_ax.scatter(pd.to_datetime(sell_trades["time"]), sell_trades["entry_price"], color="#b22222", marker="v", s=40, label="Sell")
    price_ax.set_title("Backtest Trade Markers")
    price_ax.set_ylabel("Price")
    price_ax.set_xlabel("Time")
    price_ax.legend()
    price_fig.autofmt_xdate()
    price_fig.tight_layout()
    price_fig.savefig(trades_output_path)
    plt.close(price_fig)
