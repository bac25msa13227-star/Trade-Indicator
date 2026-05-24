from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from statistics import median
from typing import Any


BALANCE_RE = re.compile(
    r"BALANCE_UPDATE\s+balance=(?P<balance>-?[0-9.]+)\s+equity=(?P<equity>-?[0-9.]+)\s+profit=(?P<profit>-?[0-9.]+)"
)


def read_text(path: Path) -> str:
    for encoding in ("utf-16", "utf-8", "utf-8-sig"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError:
            continue
    return path.read_text(errors="replace")


def parse_trade_returns(log_dir: Path) -> list[float]:
    returns: list[float] = []
    for path in sorted(log_dir.glob("fold_*_tester.log")):
        for match in BALANCE_RE.finditer(read_text(path)):
            balance = float(match.group("balance"))
            profit = float(match.group("profit"))
            prior_balance = balance - profit
            if prior_balance > 0:
                returns.append(profit / prior_balance)
    return returns


def max_drawdown_pct(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 0.0
    worst = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (equity - peak) / peak * 100.0
            worst = min(worst, dd)
    return worst


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    weight = idx - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def run_mc(
    returns: list[float],
    *,
    iterations: int,
    initial_balance: float,
    dd_floor: float,
    target_balance: float,
    bootstrap: bool,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    finals: list[float] = []
    max_dds: list[float] = []
    target_hits = 0
    dd_hits = 0
    for _ in range(iterations):
        if bootstrap:
            sequence = [rng.choice(returns) for _ in range(len(returns))]
        else:
            sequence = list(returns)
            rng.shuffle(sequence)
        balance = initial_balance
        curve = [balance]
        hit_target = False
        hit_dd = False
        for ret in sequence:
            balance *= 1.0 + ret
            curve.append(balance)
            if target_balance > 0 and balance >= target_balance:
                balance = target_balance
                curve[-1] = balance
                hit_target = True
                break
            if dd_floor > 0 and balance <= dd_floor:
                hit_dd = True
                break
        finals.append(balance)
        max_dds.append(max_drawdown_pct(curve))
        target_hits += int(hit_target)
        dd_hits += int(hit_dd)
    return {
        "iterations": iterations,
        "bootstrap": bootstrap,
        "initial_balance": initial_balance,
        "dd_floor": dd_floor,
        "target_balance": target_balance,
        "trades_per_iteration": len(returns),
        "target_hit_rate": target_hits / iterations if iterations else 0.0,
        "dd_hit_rate": dd_hits / iterations if iterations else 0.0,
        "final_balance": {
            "p5": percentile(finals, 5),
            "p50": median(finals) if finals else 0.0,
            "p95": percentile(finals, 95),
            "min": min(finals) if finals else 0.0,
            "max": max(finals) if finals else 0.0,
        },
        "max_dd_pct": {
            "p5": percentile(max_dds, 5),
            "p50": median(max_dds) if max_dds else 0.0,
            "p95": percentile(max_dds, 95),
            "min": min(max_dds) if max_dds else 0.0,
            "max": max(max_dds) if max_dds else 0.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Monte Carlo over MT5 tester BALANCE_UPDATE trade returns.")
    parser.add_argument("--tester-log-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--initial-balance", type=float, default=200.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle-only", action="store_true")
    args = parser.parse_args()

    returns = parse_trade_returns(args.tester_log_dir)
    if not returns:
        raise RuntimeError(f"no BALANCE_UPDATE returns found in {args.tester_log_dir}")
    dd_floor = args.initial_balance * (1.0 - args.max_dd_pct / 100.0)
    report = {
        "tester_log_dir": str(args.tester_log_dir),
        "trade_returns": {
            "count": len(returns),
            "win_rate": sum(1 for item in returns if item > 0) / len(returns),
            "avg_return": sum(returns) / len(returns),
            "p5_return": percentile(returns, 5),
            "p50_return": median(returns),
            "p95_return": percentile(returns, 95),
        },
        "monte_carlo": run_mc(
            returns,
            iterations=args.iterations,
            initial_balance=args.initial_balance,
            dd_floor=dd_floor,
            target_balance=args.target_balance,
            bootstrap=not args.shuffle_only,
            seed=args.seed,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
