import argparse
from pathlib import Path

from xauusd_ai.config import load_settings
from xauusd_ai.orchestrator import (
    run_backtest,
    run_live_loop,
    run_mt5_check,
    run_news_backtest,
    run_news_fetch,
    run_paper_trade_loop,
    run_training,
    run_walkforward,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XAUUSD AI trading system")
    sub = parser.add_subparsers(dest="command")

    # Standard commands
    for cmd in ("train", "backtest", "walkforward", "paper", "live", "mt5-check"):
        sub.add_parser(cmd)

    # news-fetch: crawl Forex Factory calendar (or FRED)
    nf = sub.add_parser("news-fetch", help="Fetch economic calendar data and cache results")
    nf.add_argument("--from", dest="from_date", default=None,
                    help="Start date for historical crawl, e.g. '2023-01-01'")
    nf.add_argument("--to", dest="to_date", default=None,
                    help="End date for historical crawl (default: today)")
    nf.add_argument("--fred", dest="use_fred", action="store_true", default=False,
                    help="Use FRED (Federal Reserve) API instead of Forex Factory for historical USD macro data")

    # news-backtest: train + backtest with news features
    sub.add_parser("news-backtest", help="Train and backtest the model with Forex Factory news features")

    parser.add_argument("--config", default="configs/settings.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

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
    elif args.command == "news-fetch":
        run_news_fetch(settings, from_date=args.from_date, to_date=args.to_date, use_fred=args.use_fred)
    elif args.command == "news-backtest":
        run_news_backtest(settings)

    return 0
