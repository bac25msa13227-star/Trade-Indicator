from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sum_signals_risk_cap(report: dict[str, Any]) -> int:
    total = 0
    for fold in report.get("folds", []) or []:
        sim = fold.get("concurrent_sim", {}) if isinstance(fold, dict) else {}
        total += int(_safe_float(sim.get("signals_risk_cap"), 0.0))
    aggregate = report.get("aggregate", {}) or {}
    concurrent = aggregate.get("concurrent_sim", {}) if isinstance(aggregate, dict) else {}
    if total == 0:
        total = int(_safe_float(concurrent.get("signals_risk_cap"), 0.0))
    return total


def _max_drawdown_pct(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    if "drawdown" in frame.columns:
        drawdown = pd.to_numeric(frame["drawdown"], errors="coerce").dropna()
        if drawdown.empty:
            return 0.0
        return round(float(drawdown.min()) * 100.0, 2)
    if "balance_after" not in frame.columns:
        return 0.0
    equity = pd.to_numeric(frame["balance_after"], errors="coerce").dropna()
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return round(float(dd.min()) * 100.0, 2)


def _summarize_group(frame: pd.DataFrame) -> dict[str, Any]:
    pnl = pd.to_numeric(frame.get("pnl", 0.0), errors="coerce").fillna(0.0)
    net_rr = pd.to_numeric(frame.get("net_rr", 0.0), errors="coerce").fillna(0.0)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    return {
        "trades": int(len(frame)),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / max(wins + losses, 1), 4),
        "pnl": round(float(pnl.sum()), 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else 0.0,
        "expectancy_r": round(float(net_rr.mean()) if len(net_rr) else 0.0, 4),
        "max_dd_pct": _max_drawdown_pct(frame),
    }


def build_report(trades_path: Path, wf_report_path: Path | None = None) -> dict[str, Any]:
    wf_report = _load_report(wf_report_path)
    try:
        trades = pd.read_csv(trades_path)
    except pd.errors.EmptyDataError:
        trades = pd.DataFrame()

    if trades.empty:
        summary = _summarize_group(trades)
        summary["signals_risk_cap"] = _sum_signals_risk_cap(wf_report)
        summary["source_trades"] = str(trades_path)
        summary["source_walkforward_report"] = str(wf_report_path) if wf_report_path else None
        return {
            "summary": summary,
            "monthly": [],
            "pass_criteria": {
                "max_dd_pct_min": -15.0,
                "expectancy_r_min": 0.8,
                "profit_factor_min": 1.6,
                "risk_cap_signals_expected": 0,
            },
            "pass": {
                "max_dd": summary["max_dd_pct"] >= -15.0,
                "expectancy_r": summary["expectancy_r"] >= 0.8,
                "profit_factor": summary["profit_factor"] >= 1.6,
                "risk_cap": summary["signals_risk_cap"] == 0,
            },
        }

    trades["time"] = pd.to_datetime(trades["time"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    trades["month"] = trades["time"].dt.strftime("%Y-%m")

    monthly = []
    for month, group in trades.groupby("month", sort=True):
        row = {"month": month}
        row.update(_summarize_group(group))
        monthly.append(row)

    summary = _summarize_group(trades)
    summary["signals_risk_cap"] = _sum_signals_risk_cap(wf_report)
    summary["source_trades"] = str(trades_path)
    summary["source_walkforward_report"] = str(wf_report_path) if wf_report_path else None

    return {
        "summary": summary,
        "monthly": monthly,
        "pass_criteria": {
            "max_dd_pct_min": -15.0,
            "expectancy_r_min": 0.8,
            "profit_factor_min": 1.6,
            "risk_cap_signals_expected": 0,
        },
        "pass": {
            "max_dd": summary["max_dd_pct"] >= -15.0,
            "expectancy_r": summary["expectancy_r"] >= 0.8,
            "profit_factor": summary["profit_factor"] >= 1.6,
            "risk_cap": summary["signals_risk_cap"] == 0,
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    s = report["summary"]
    lines = [
        "# Walk-Forward Risk Report",
        "",
        "## Summary",
        "",
        f"- Trades: {s['trades']}",
        f"- P&L: ${s['pnl']:.2f}",
        f"- Max DD: {s['max_dd_pct']:.2f}%",
        f"- Expectancy: {s['expectancy_r']:.4f}R",
        f"- Profit factor: {s['profit_factor']:.4f}",
        f"- Signals blocked by risk cap: {s['signals_risk_cap']}",
        "",
        "## Monthly",
        "",
        "| Month | Trades | P&L | Max DD | Expectancy R | PF | WR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["monthly"]:
        lines.append(
            f"| {row['month']} | {row['trades']} | ${row['pnl']:.2f} | "
            f"{row['max_dd_pct']:.2f}% | {row['expectancy_r']:.4f} | "
            f"{row['profit_factor']:.4f} | {row['win_rate']:.2%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate monthly WF risk report from sim trades.")
    parser.add_argument("--trades", required=True, type=Path)
    parser.add_argument("--wf-report", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/wf_risk_report.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/wf_risk_report.md"))
    args = parser.parse_args()

    report = build_report(args.trades, args.wf_report)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_md)
    print(json.dumps(report["summary"], indent=2))
    print(f"JSON: {args.out_json}")
    print(f"MD:   {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
