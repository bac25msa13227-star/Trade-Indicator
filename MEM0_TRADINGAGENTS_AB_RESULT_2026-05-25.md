# mem0/TradingAgents A/B Result - 2026-05-25

Scope: offline replay only. No live ACC2 executor was changed.

Input:

- MT5 deal replay directory: `outputs/mt5_1200_fixed_ea_rewf_20260523/dd_replay_test_current_logs`
- Signal manifest: `outputs/mt5_1200_fixed_ea_rewf_20260523/profit_risk_sweep_fixed_ea_20260523/source_threshold_050_manifest.json`
- Deposit: `$1200`
- Folds matched: `63`
- Median signal-match rate: `100%`

## Test 1 - Implemented 5-tier rating risk

Mapping tested:

- confidence `<0.55`: skip
- confidence `0.55-0.70`: risk x0.60
- confidence `>=0.70`: risk x1.00

Result:

| Metric | Baseline | Implemented rating |
|---|---:|---:|
| Median final | $3,032.58 | $1,967.74 |
| Min final | $1,506.50 | $1,165.85 |
| Worst DD | -29.89% | -35.94% |
| Loss folds | 0/63 | 1/63 |
| Active trades | 2,561 | 1,443 |

Conclusion: do not enable `risk.rating_risk_enabled` for this strategy. It reduces/blocks too many profitable 0.50-0.70 confidence trades.

## Test 2 - Confidence bucket multiplier sweep

Best in-sample/oracle bucket:

- `<0.55`: x1.30
- `0.55-0.70`: x1.40
- `>=0.70`: x0.80

Baseline vs best:

| Metric | Baseline | Best bucket |
|---|---:|---:|
| Median final | $3,032.58 | $3,702.39 |
| Min final | $1,506.50 | $1,625.81 |
| Worst DD | -29.89% | -35.10% |
| Loss folds | 0/63 | 0/63 |

Clean rolling selector using only prior folds:

- Median final: `$3,490.39`
- Min final: `$1,506.50`
- Worst DD: `-35.10%`
- Loss folds: `0/63`

Conclusion: confidence-bucket sizing can increase profit, but it raises worst DD. It should not go live until MT5 WF validates the selected multiplier policy directly.

## Code decision

Added `risk.confidence_bucket_risk_enabled` and configurable bucket multipliers, default off. This keeps live unchanged while allowing a clean WF-selected policy to be tested without hard-coding the bad 5-tier risk assumption.
