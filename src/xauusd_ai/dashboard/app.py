from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "outputs"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def summarize_trades(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty or "pnl" not in trades.columns:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_profit": 0.0,
        }

    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
    return {
        "trades": int(len(trades)),
        "wins": int((trades["pnl"] > 0).sum()),
        "losses": int((trades["pnl"] < 0).sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": float(trades["pnl"].sum()),
    }


def summarize_daily_performance(trades: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    if trades.empty:
        return {
            "mean_daily_return": 0.0,
            "median_daily_return": 0.0,
            "best_day_return": 0.0,
            "worst_day_return": 0.0,
            "share_ge_2": 0.0,
            "share_ge_5": 0.0,
            "share_ge_10": 0.0,
        }, pd.DataFrame()

    daily = trades.copy()
    daily["time"] = pd.to_datetime(daily["time"], utc=True, errors="coerce")
    daily = daily.dropna(subset=["time"]).sort_values("time")
    daily["date"] = daily["time"].dt.date
    daily_summary = daily.groupby("date").agg(
        trades=("pnl", "size"),
        net_pnl=("pnl", "sum"),
        gross_profit=("pnl", lambda series: series[series > 0].sum()),
        gross_loss=("pnl", lambda series: -series[series < 0].sum()),
        start_balance=("balance_before", "first"),
        end_balance=("balance_after", "last"),
    )
    daily_summary["return_pct"] = (daily_summary["end_balance"] / daily_summary["start_balance"] - 1.0) * 100.0

    summary = {
        "mean_daily_return": float(daily_summary["return_pct"].mean()),
        "median_daily_return": float(daily_summary["return_pct"].median()),
        "best_day_return": float(daily_summary["return_pct"].max()),
        "worst_day_return": float(daily_summary["return_pct"].min()),
        "share_ge_2": float((daily_summary["return_pct"] >= 2.0).mean()),
        "share_ge_5": float((daily_summary["return_pct"] >= 5.0).mean()),
        "share_ge_10": float((daily_summary["return_pct"] >= 10.0).mean()),
    }
    return summary, daily_summary.reset_index()


def enrich_trades_with_dataset_context(trades: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or dataset.empty or "time" not in trades.columns or "time" not in dataset.columns:
        return trades

    enriched_trades = trades.copy()
    enriched_trades["time"] = pd.to_datetime(enriched_trades["time"], utc=True, errors="coerce")

    context_columns = [column for column in ["time", "volatility_regime", "strategy_score", "trade_side"] if column in dataset.columns]
    if len(context_columns) <= 1:
        return enriched_trades

    context = dataset[context_columns].copy()
    context["time"] = pd.to_datetime(context["time"], utc=True, errors="coerce")
    context = context.dropna(subset=["time"]).sort_values("time")
    enriched_trades = pd.merge_asof(enriched_trades.sort_values("time"), context, on="time", direction="backward")

    if "volatility_regime" in enriched_trades.columns:
        regime_map = {0: "sideway", 1: "normal", 2: "strong_volatility"}
        enriched_trades["regime_label"] = enriched_trades["volatility_regime"].map(regime_map).fillna("unknown")

    return enriched_trades


st.set_page_config(page_title="XAUUSD AI Dashboard", layout="wide")
st.title("XAUUSD AI Dashboard")

backtest_report = load_json(OUTPUTS / "backtest_report.json")
walkforward_report = load_json(OUTPUTS / "walkforward_report.json")
training_report = load_json(OUTPUTS / "training_report.json")
trades = load_csv(OUTPUTS / "backtest_trades.csv")
walkforward_trades = load_csv(OUTPUTS / "walkforward_trades.csv")
paper_signals = load_csv(OUTPUTS / "paper_trade_signals.csv")
dataset = load_csv(OUTPUTS / "training_dataset.csv")

trades = enrich_trades_with_dataset_context(trades, dataset)
trade_summary = summarize_trades(trades)
daily_summary, daily_frame = summarize_daily_performance(trades)

left, middle, right = st.columns(3)
left.metric("Backtest Return %", backtest_report.get("return_pct", "n/a"))
middle.metric("Backtest Profit Factor", backtest_report.get("profit_factor", "n/a"))
right.metric("Walk-Forward Recall", walkforward_report.get("avg_recall", "n/a"))

sub_left, sub_middle, sub_right = st.columns(3)
sub_left.metric("Walk-Forward Avg Return %", walkforward_report.get("avg_return_pct", "n/a"))
sub_middle.metric("Walk-Forward Avg Profit Factor", walkforward_report.get("avg_profit_factor", "n/a"))
sub_right.metric("Paper Signals Logged", 0 if paper_signals.empty else len(paper_signals))

range_left, range_middle, range_right, range_fourth = st.columns(4)
range_left.metric("Test Start", backtest_report.get("test_start", "n/a"))
range_middle.metric("Trade End", backtest_report.get("trade_end", "n/a"))
range_right.metric("Test Days", backtest_report.get("test_days", "n/a"))
range_fourth.metric("Trade Days", backtest_report.get("trade_days", "n/a"))

trade_left, trade_middle, trade_right, trade_fourth = st.columns(4)
trade_left.metric("Trades", trade_summary["trades"])
trade_middle.metric("Wins / Losses", f"{trade_summary['wins']} / {trade_summary['losses']}")
trade_right.metric("Gross Profit", round(trade_summary["gross_profit"], 2))
trade_fourth.metric("Gross Loss", round(trade_summary["gross_loss"], 2))

net_left, net_middle, net_right = st.columns(3)
net_left.metric("Net Profit", round(trade_summary["net_profit"], 2))
net_middle.metric("Best Trade", backtest_report.get("best_trade", "n/a"))
net_right.metric("Worst Trade", backtest_report.get("worst_trade", "n/a"))

daily_left, daily_middle, daily_right, daily_fourth = st.columns(4)
daily_left.metric("Mean Daily Return %", round(daily_summary["mean_daily_return"], 3))
daily_middle.metric("Median Daily Return %", round(daily_summary["median_daily_return"], 3))
daily_right.metric("Best Day %", round(daily_summary["best_day_return"], 3))
daily_fourth.metric("Worst Day %", round(daily_summary["worst_day_return"], 3))

daily_share_left, daily_share_middle, daily_share_right = st.columns(3)
daily_share_left.metric("Share Days >= 2%", f"{daily_summary['share_ge_2'] * 100:.2f}%")
daily_share_middle.metric("Share Days >= 5%", f"{daily_summary['share_ge_5'] * 100:.2f}%")
daily_share_right.metric("Share Days >= 10%", f"{daily_summary['share_ge_10'] * 100:.2f}%")

metrics_col, threshold_col = st.columns(2)
with metrics_col:
    st.subheader("Backtest Report")
    st.json(backtest_report)
with threshold_col:
    st.subheader("Training Report")
    st.json(training_report)

if not trades.empty:
    st.subheader("Backtest Equity Curve")
    trades["time"] = pd.to_datetime(trades["time"], utc=True)
    st.line_chart(trades.set_index("time")["balance_after"])

    if not daily_frame.empty:
        st.subheader("Daily Return Profile")
        daily_frame["date"] = pd.to_datetime(daily_frame["date"])
        daily_chart_left, daily_chart_right = st.columns(2)
        with daily_chart_left:
            st.bar_chart(daily_frame.set_index("date")["return_pct"])
        with daily_chart_right:
            st.line_chart(daily_frame.set_index("date")[["net_pnl"]])
        st.dataframe(daily_frame.tail(120), width="stretch")

    st.subheader("Backtest PnL Distribution")
    st.bar_chart(trades["pnl"].value_counts(bins=30).sort_index())

    st.subheader("Backtest Trades")
    st.dataframe(trades.tail(200), width="stretch")

    hourly_stats = trades.assign(hour=trades["time"].dt.hour).groupby("hour").agg(
        trades=("pnl", "size"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        win_rate=("is_win", "mean"),
    )
    if not hourly_stats.empty:
        st.subheader("Performance by Entry Hour")
        hourly_col1, hourly_col2 = st.columns(2)
        with hourly_col1:
            st.bar_chart(hourly_stats["total_pnl"])
        with hourly_col2:
            st.bar_chart(hourly_stats["win_rate"])
        st.dataframe(hourly_stats.round(4), width="stretch")

    if "regime_label" in trades.columns:
        regime_stats = trades.groupby("regime_label").agg(
            trades=("pnl", "size"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            win_rate=("is_win", "mean"),
        )
        if not regime_stats.empty:
            st.subheader("Performance by Regime")
            regime_col1, regime_col2 = st.columns(2)
            with regime_col1:
                st.bar_chart(regime_stats["total_pnl"])
            with regime_col2:
                st.bar_chart(regime_stats["win_rate"])
            st.dataframe(regime_stats.round(4), width="stretch")

if not walkforward_trades.empty:
    st.subheader("Walk-Forward Trades")
    st.dataframe(walkforward_trades.tail(200), width="stretch")

if walkforward_report.get("folds"):
    folds = pd.DataFrame(walkforward_report["folds"])
    if "return_pct" in folds.columns:
        st.subheader("Walk-Forward Fold Returns")
        st.bar_chart(folds.set_index("fold")["return_pct"])
    st.dataframe(folds, width="stretch")

if not paper_signals.empty:
    st.subheader("Paper Trade Signals")
    st.dataframe(paper_signals.tail(200), width="stretch")
