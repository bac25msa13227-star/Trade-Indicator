import argparse
from pathlib import Path

from xauusd_ai.config import load_settings
from xauusd_ai.orchestrator import run_backtest, run_live_loop, run_mt5_check, run_paper_trade_loop, run_training, run_walkforward


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XAUUSD AI trading system")
    parser.add_argument("command", choices=["train", "backtest", "walkforward", "paper", "live", "mt5-check"])
    parser.add_argument("--config", default="configs/settings.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings(Path(args.config))

    if args.command == "train":
        run_training(settings)
    elif args.command == "backtest":
        run_backtest(settings)
    elif args.command == "walkforward":
        run_walkforward(settings)
    elif args.command == "paper":
        run_paper_trade_loop(settings)
    elif args.command == "live":
        run_live_loop(settings)
    elif args.command == "mt5-check":
        run_mt5_check(settings)

    return 0
