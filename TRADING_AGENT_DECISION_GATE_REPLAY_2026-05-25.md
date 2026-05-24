# TradingAgentDecisionGate Replay - 2026-05-25

Scope: offline replay only. ACC2 live was not changed.

Input:

- Joined MT5 trade/signal replay: `outputs/mem0_tradingagents_ab_20260525/rating_risk_replay/ab_rating_all_trades.csv`
- Available MT5 deal folds: `118-180` (`63` folds, 2021-2026 segment)
- Deposit: `$1200`
- Agent uses only prior completed folds after a 12-fold bootstrap.

Gate design:

- Market perspective: side, confidence bucket, session.
- Memory perspective: prior-fold outcomes for similar side/session/confidence buckets.
- Risk perspective: current fold drawdown path.
- Actions: `TAKE`, `SKIP`, `REDUCE`, `BOOST`.

## Direct gate replay

| Metric | Baseline | Agent gate |
|---|---:|---:|
| Median final | $3,032.58 | $2,884.18 |
| Min final | $1,506.50 | $1,160.89 |
| Mean final | $3,730.46 | $3,786.22 |
| Worst DD | -29.89% | -34.10% |
| Loss folds | 0/63 | 1/63 |
| Active trades | 2,561 | 2,442 |

The first gate improved mean only by boosting a few tails, but it hurt median, min final, DD, and loss-fold count.

## Parameter sweep

Best practical sweep result:

| Metric | Baseline | Best swept gate |
|---|---:|---:|
| Median final | $3,032.58 | $2,885.65 |
| Min final | $1,506.50 | $1,506.50 |
| Mean final | $3,730.46 | $3,656.63 |
| Worst DD | -29.89% | -29.89% |
| Loss folds | 0/63 | 0/63 |
| Active trades | 2,561 | 2,519 |

Best swept parameters:

- `min_memory_trades=25`
- `bad_expectancy_usd=-5.0`
- `good_expectancy_usd=20.0`
- `bad_win_rate=0.42`
- `reduce_multiplier=0.75`
- `boost_multiplier=1.0`
- `dd_reduce_pct=-12.0`
- `dd_skip_pct=-20.0`

## Conclusion

This deterministic TradingAgents-style decision gate does not beat the existing baseline on the available MT5 replay folds. It should stay research-only and must not be enabled for live.

The result is useful: it shows that a simple prior-memory/agent gate mostly removes or reduces trades that the current MT5-validated strategy still needs for median profit. A useful TradingAgents integration likely needs richer features at decision time, not just side/session/confidence buckets.
