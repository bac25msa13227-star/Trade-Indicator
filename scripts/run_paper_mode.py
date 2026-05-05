#!/usr/bin/env python3
"""
Paper Trading Mode Runner
=========================
Run paper trading loop để validate slippage model với live data.

Features:
- Shadow execution: log all signals without placing real orders
- Track predicted vs actual slippage
- Market condition logging (ATR, spread, volume, session)
- Summary stats: win rate, profit factor, slippage accuracy

Usage:
    python scripts/run_paper_mode.py configs/live_acc1.yaml --duration 7d
    python scripts/run_paper_mode.py configs/live_acc1.yaml --max-signals 100
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xauusd_ai.config import load_settings
from xauusd_ai.orchestrator import run_paper_trade_loop


def parse_duration(duration_str: str) -> int:
    """Parse duration string like '7d', '24h', '30m' to seconds."""
    if duration_str.endswith("d"):
        return int(duration_str[:-1]) * 86400
    elif duration_str.endswith("h"):
        return int(duration_str[:-1]) * 3600
    elif duration_str.endswith("m"):
        return int(duration_str[:-1]) * 60
    else:
        raise ValueError(f"Invalid duration format: {duration_str}. Use 7d, 24h, or 30m")


def main():
    parser = argparse.ArgumentParser(
        description="Run paper trading mode for slippage validation"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to config YAML (e.g., configs/live_acc1.yaml)",
    )
    parser.add_argument(
        "--duration",
        type=str,
        default=None,
        help="Run duration (e.g., 7d, 24h, 30m). Default: run until stopped",
    )
    parser.add_argument(
        "--max-signals",
        type=int,
        default=None,
        help="Stop after N signals logged. Default: unlimited",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper_trades.jsonl"),
        help="Output path for paper trades log",
    )
    
    args = parser.parse_args()
    
    # Load settings
    settings = load_settings(args.config)
    
    # Override execution mode to paper
    settings.execution.mode = "paper"
    
    # Set paper trade parameters
    if args.max_signals:
        settings.execution.paper_trade_max_loops = args.max_signals
    elif args.duration:
        # Calculate max loops from duration (assuming 15min polling)
        duration_seconds = parse_duration(args.duration)
        poll_seconds = getattr(settings.app, "poll_seconds", 900)  # 15min default
        settings.execution.paper_trade_max_loops = int(duration_seconds / poll_seconds)
    else:
        settings.execution.paper_trade_max_loops = 0  # Run forever
    
    # Print configuration
    print("=" * 70)
    print("  PAPER TRADING MODE")
    print("=" * 70)
    print(f"  Config      : {args.config}")
    print(f"  Mode        : {settings.execution.mode}")
    print(f"  Output      : {args.output}")
    print(f"  Data source : {settings.execution.paper_data_source}")
    print(f"  Poll rate   : {getattr(settings.app, 'poll_seconds', 900)}s")
    
    if args.duration:
        duration_seconds = parse_duration(args.duration)
        end_time = datetime.now() + timedelta(seconds=duration_seconds)
        print(f"  Duration    : {args.duration} (until {end_time.strftime('%Y-%m-%d %H:%M:%S')})")
    elif args.max_signals:
        print(f"  Max signals : {args.max_signals}")
    else:
        print(f"  Duration    : Unlimited (Ctrl+C to stop)")
    
    print()
    print("  Features:")
    print("    ✅ Dynamic slippage prediction")
    print("    ✅ Market condition tracking")
    print("    ✅ Signal skip logging")
    print("    ✅ Summary stats generation")
    print("=" * 70)
    print()
    
    # Set output path
    settings.app.paper_trade_log_path = str(args.output.parent / f"{args.output.stem}.csv")
    
    try:
        start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting paper trading loop...")
        print()
        
        # Run paper trading loop
        run_paper_trade_loop(settings)
        
        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print(f"✅ Paper trading complete ({elapsed/60:.1f} minutes)")
        print("=" * 70)
        print()
        print("Output files:")
        print(f"  - {args.output} (JSONL with slippage data)")
        print(f"  - {settings.app.paper_trade_log_path} (Legacy CSV)")
        print()
        
        # Print summary stats if paper_trades.jsonl exists
        if args.output.exists():
            from xauusd_ai.execution.paper_logger import PaperTradeLogger
            logger = PaperTradeLogger(output_path=str(args.output))
            stats = logger.get_summary_stats()
            
            if stats and stats.get("total_trades", 0) > 0:
                print("📊 Summary Statistics:")
                print(f"  Total trades     : {stats['total_trades']}")
                print(f"  Win rate         : {stats['win_rate']:.1%}")
                print(f"  Profit factor    : {stats['profit_factor']:.2f}")
                print(f"  Avg RR           : {stats['avg_rr']:+.2f}R")
                print(f"  Total RR         : {stats['total_rr']:+.2f}R")
                
                if stats.get("slippage_accuracy") is not None:
                    print(f"  Slippage accuracy: {stats['slippage_accuracy']:.1%}")
                print()
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print(f"⚠️  Stopped by user ({elapsed/60:.1f} minutes)")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Error: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
