#!/bin/bash
# WF Comparison: Static vs Dynamic Slippage WITH Metrics

set -e
source .venv/bin/activate

echo "=========================================="
echo "WF Comparison: Static vs Dynamic Slippage"
echo "=========================================="
echo ""

# 1. Run with STATIC slippage (use_dynamic_slippage: false)
echo "[1/2] Running WF with STATIC slippage..."
python scripts/walkforward_ict_wyckoff.py configs/acc1_v14pp_profit.yaml \
  --test-start 2023-01-01 \
  --test-bars 6000 \
  --step-bars 6000 \
  --no-compound \
  --combo133 \
  --cache \
  --risk-pct 0.030 \
  --no-rr-sweep \
  2>&1 | tee outputs/wf_static_with_metrics.log

echo ""
echo "[1/2] ✅ Static slippage complete"
echo ""

# 2. Enable dynamic slippage in config
echo "[2/2] Enabling dynamic slippage in config..."
sed -i.bak 's/use_dynamic_slippage: false/use_dynamic_slippage: true/' configs/acc1_v14pp_profit.yaml

# 3. Run with DYNAMIC slippage
echo "[2/2] Running WF with DYNAMIC slippage..."
python scripts/walkforward_ict_wyckoff.py configs/acc1_v14pp_profit.yaml \
  --test-start 2023-01-01 \
  --test-bars 6000 \
  --step-bars 6000 \
  --no-compound \
  --combo133 \
  --risk-pct 0.030 \
  --no-rr-sweep \
  2>&1 | tee outputs/wf_dynamic_with_metrics.log

echo ""
echo "[2/2] ✅ Dynamic slippage complete"
echo ""

# 4. Restore config (disable dynamic for safety)
echo "Restoring config to static slippage (safe default)..."
mv configs/acc1_v14pp_profit.yaml.bak configs/acc1_v14pp_profit.yaml

echo ""
echo "=========================================="
echo "✅ WF Comparison Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - outputs/wf_static_with_metrics.log"
echo "  - outputs/wf_dynamic_with_metrics.log"
echo ""
echo "Compare Sharpe/Calmar metrics between runs."
