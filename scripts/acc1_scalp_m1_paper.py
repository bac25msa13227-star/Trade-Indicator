#!/usr/bin/env python3
"""
ACC1 wrapper for the shared M1 scalp paper-trade pipeline.

ACC2 remains frozen; ACC1 can validate its own scalp config/artifacts independently.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import acc2_scalp_m1_paper as base  # type: ignore


base.CONFIG = REPO / "configs" / "live_acc1_scalp_m1.yaml"

ACC2_SIG_PATH = REPO / "outputs" / "paper_trade_signals_acc2_scalp_m1.csv"
ACC1_SIG_PATH = REPO / "outputs" / "paper_trade_signals_acc1_scalp_m1.csv"
ACC2_SIG_BACKUP = REPO / "outputs" / "paper_trade_signals_acc2_scalp_m1.csv.acc1_backup"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-bars", type=int, default=base.TEST_SIZE)
    args = parser.parse_args()

    had_acc2_sig = ACC2_SIG_PATH.exists()
    if had_acc2_sig:
        if ACC2_SIG_BACKUP.exists():
            ACC2_SIG_BACKUP.unlink()
        ACC2_SIG_PATH.replace(ACC2_SIG_BACKUP)

    base.main(test_bars=args.test_bars)

    if ACC1_SIG_PATH.exists():
        ACC1_SIG_PATH.unlink()
    if ACC2_SIG_PATH.exists():
        ACC2_SIG_PATH.replace(ACC1_SIG_PATH)

    if had_acc2_sig and ACC2_SIG_BACKUP.exists():
        ACC2_SIG_BACKUP.replace(ACC2_SIG_PATH)
    elif ACC2_SIG_BACKUP.exists():
        ACC2_SIG_BACKUP.unlink()

    print(f"ACC1 paper signals moved to: {ACC1_SIG_PATH}")
