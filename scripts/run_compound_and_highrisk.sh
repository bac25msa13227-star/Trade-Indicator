#!/usr/bin/env bash
# Run compound WF + high-risk sweep
WF="scripts/walkforward_ict_wyckoff.py"
CFG="configs/acc1_v14pp_profit_candidate_ptp25_tp55.yaml"

echo "=== 1. COMPOUND (risk 5%, carry balance between folds) ==="
.venv/bin/python $WF $CFG \
  --no-rr-sweep --cache --test-start 2024-01-01 \
  --test-bars 6000 --step-bars 6000 --combo133 --risk-pct 0.05 \
  2>&1 | tee outputs/wf_compound_r5.txt
echo "=== COMPOUND DONE ==="

echo "=== 2. HIGH-RISK 8% no-compound ==="
.venv/bin/python $WF $CFG \
  --no-rr-sweep --cache --test-start 2024-01-01 \
  --test-bars 6000 --step-bars 6000 --combo133 --risk-pct 0.08 --no-compound \
  2>&1 | tee outputs/wf_risk8pct_nocompound.txt
echo "=== RISK-8% DONE ==="

echo "=== 3. HIGH-RISK 10% no-compound ==="
.venv/bin/python $WF $CFG \
  --no-rr-sweep --cache --test-start 2024-01-01 \
  --test-bars 6000 --step-bars 6000 --combo133 --risk-pct 0.10 --no-compound \
  2>&1 | tee outputs/wf_risk10pct_nocompound.txt
echo "=== RISK-10% DONE ==="

echo "=== 4. HIGH-RISK 10% + COMPOUND ==="
.venv/bin/python $WF $CFG \
  --no-rr-sweep --cache --test-start 2024-01-01 \
  --test-bars 6000 --step-bars 6000 --combo133 --risk-pct 0.10 \
  2>&1 | tee outputs/wf_risk10pct_compound.txt
echo "=== RISK-10% COMPOUND DONE ==="

echo "=== ALL RUNS COMPLETE ==="
