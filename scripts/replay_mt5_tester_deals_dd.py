from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DEAL_RE = re.compile(
    r"(?P<time>\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2})\s+deal\s+#(?P<ticket>\d+)\s+"
    r"(?P<side>buy|sell)\s+(?P<volume>[0-9.]+)\s+(?P<symbol>\S+)\s+at\s+(?P<price>[0-9.]+)\s+done",
    re.IGNORECASE,
)


def read_log(path: Path) -> str:
    for encoding in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            return path.read_text(encoding=encoding, errors="strict")
        except UnicodeError:
            continue
    return path.read_text(errors="ignore")


def replay_log(path: Path, deposit: float, contract_size: float) -> tuple[dict, pd.DataFrame]:
    text = read_log(path)
    balance = float(deposit)
    peak = float(deposit)
    max_dd_pct = 0.0
    open_deal: dict | None = None
    rows: list[dict] = []

    for match in DEAL_RE.finditer(text):
        deal = {
            "time": match.group("time"),
            "ticket": int(match.group("ticket")),
            "side": match.group("side").lower(),
            "volume": float(match.group("volume")),
            "symbol": match.group("symbol"),
            "price": float(match.group("price")),
        }
        if open_deal is None:
            open_deal = deal
            continue

        # With MaxPositions=1, the next opposite-side deal is the closing deal.
        if deal["side"] == open_deal["side"]:
            # Defensive reset; this should not happen for these WF runs.
            open_deal = deal
            continue

        direction = 1.0 if open_deal["side"] == "buy" else -1.0
        volume = min(float(open_deal["volume"]), float(deal["volume"]))
        pnl = direction * (float(deal["price"]) - float(open_deal["price"])) * volume * float(contract_size)
        balance += pnl
        if balance > peak:
            peak = balance
        dd_pct = (balance - peak) / peak * 100.0 if peak > 0 else 0.0
        if dd_pct < max_dd_pct:
            max_dd_pct = dd_pct
        rows.append(
            {
                "open_time": open_deal["time"],
                "close_time": deal["time"],
                "side": open_deal["side"],
                "volume": volume,
                "open_price": open_deal["price"],
                "close_price": deal["price"],
                "pnl": round(pnl, 2),
                "balance": round(balance, 2),
                "peak": round(peak, 2),
                "dd_pct": round(dd_pct, 2),
                "open_ticket": open_deal["ticket"],
                "close_ticket": deal["ticket"],
            }
        )
        open_deal = None

    df = pd.DataFrame(rows)
    summary = {
        "log": str(path),
        "trades_replayed": int(len(df)),
        "final_balance_replayed": round(balance, 2),
        "max_dd_pct_replayed": round(max_dd_pct, 2),
        "worst_balance": round(float(df["balance"].min()), 2) if not df.empty else round(float(deposit), 2),
        "wins": int((df["pnl"] > 0).sum()) if not df.empty else 0,
        "losses": int((df["pnl"] < 0).sum()) if not df.empty else 0,
    }
    return summary, df


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay MT5 tester deal logs and compute realized balance drawdown.")
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deposit", type=float, default=1200.0)
    parser.add_argument("--contract-size", type=float, default=100.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for log_path in sorted(args.logs_dir.glob("fold_*_tester.log")):
        fold_match = re.search(r"fold_(\d+)_tester\.log", log_path.name)
        if not fold_match:
            continue
        fold = int(fold_match.group(1))
        summary, trades = replay_log(log_path, args.deposit, args.contract_size)
        summary["fold"] = fold
        trades.insert(0, "fold", fold)
        trades.to_csv(args.out_dir / f"fold_{fold:03d}_deals_replay.csv", index=False)
        summaries.append(summary)

    frame = pd.DataFrame(summaries).sort_values("fold") if summaries else pd.DataFrame()
    if args.results and args.results.exists() and not frame.empty:
        results = pd.read_csv(args.results)
        if "fold" in results.columns:
            keep = ["fold", "final_balance", "trades", "win_rate_pct"]
            keep = [col for col in keep if col in results.columns]
            frame = frame.merge(results[keep], on="fold", how="left", suffixes=("", "_mt5"))
            frame["final_balance_gap"] = (frame["final_balance_replayed"] - frame["final_balance"]).round(2)
            frame["trade_count_gap"] = (frame["trades_replayed"] - frame["trades"]).astype("Int64")

    frame.to_csv(args.out_dir / "dd_replay_by_fold.csv", index=False)
    report = {
        "folds": int(len(frame)),
        "worst_dd_pct": round(float(frame["max_dd_pct_replayed"].min()), 2) if not frame.empty else 0.0,
        "worst_dd_fold": int(frame.loc[frame["max_dd_pct_replayed"].idxmin(), "fold"]) if not frame.empty else None,
        "min_final_balance_replayed": round(float(frame["final_balance_replayed"].min()), 2) if not frame.empty else None,
        "loss_folds_replayed": int((frame["final_balance_replayed"] < args.deposit).sum()) if not frame.empty else 0,
        "max_abs_final_balance_gap": round(float(frame["final_balance_gap"].abs().max()), 2)
        if "final_balance_gap" in frame
        else None,
        "max_abs_trade_count_gap": int(frame["trade_count_gap"].abs().max()) if "trade_count_gap" in frame else None,
    }
    (args.out_dir / "dd_replay_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(args.out_dir / "dd_replay_by_fold.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
