#!/bin/bash
set -e
PROJ_DIR="/Users/dodoannang/Documents/Thạc sĩ MSE/Trade Indicator"
cd "$PROJ_DIR"
source .venv/bin/activate

echo "=== Step 1: Rebuild full data from 2003 to 2026-04-30 ==="
python scripts/fetch_xauusd_dukascopy.py --from 2003-01-01 --to 2026-04-30

echo "=== Step 2: Run WF comparison ==="
python scripts/walkforward_ict_wyckoff.py configs/acc1_v14pp_profit.yaml --no-rr-sweep --cache --test-start 2024-01-01 --test-bars 6000 --step-bars 6000 --no-compound --combo133 --risk-pct 0.05 2>&1 | tee outputs/wf_apr30_comparison.txt
