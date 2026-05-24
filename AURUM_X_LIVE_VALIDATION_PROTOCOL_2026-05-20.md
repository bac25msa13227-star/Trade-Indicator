# AURUM-X Live Validation Protocol - 2026-05-20

## Current Decision

Use ACC2 demo with the same profile as the chosen live candidate:

- Account mode: Exness demo
- Starting capital model: 1200 USD
- Risk: 2.0% per trade
- Symbol: XAUUSDm
- Magic: 2505201202
- Profile source: `outputs/mt5_1200_deposit_sweep_20260520/risk_2_0/`
- Execution loop output: `outputs/acc2_demo_1200_risk2_live_check_20260520/`

ACC1 remains paper / monitoring only.

## Step 1 - Collect 30 Real Demo Closed Trades

ACC2 demo is allowed to run full risk because it is a demo account. The purpose is not to protect capital; the purpose is to measure whether live broker execution behaves like MT5 WF.

Do not validate after only a few trades. Minimum sample:

- Closed trades: at least 30
- Source: MT5 bridge `/history/closed`
- Symbol: XAUUSDm
- Magic: 2505201202

## Step 2 - Decision Parity Test After 30 Trades

Run:

```powershell
python scripts\decision_parity_test.py `
  --mode wr `
  --min-trades 30 `
  --tolerance-points 5 `
  --bridge-url http://localhost:5601 `
  --symbol XAUUSDm `
  --magic 2505201202
```

Output report:

```text
outputs/acc2_demo_1200_risk2_live_check_20260520/decision_parity_report.json
```

Pass rule:

- Live WR must be within 5 percentage points of MT5 WF WR.
- Current MT5 WF Risk2 weighted WR baseline: about 54.26%.
- Therefore live WR after 30+ closed trades should be roughly 49.26% to 59.26%.

If the script returns `WAIT`, there are not enough closed trades yet.

If it returns `FAIL`, live execution is not matching WF and the profile must not be moved to real money.

## Step 3 - Income Gap Decision

Use the realistic income path:

- Target from 1200 USD capital: 400-500 USD/month if the profile proves stable.
- Priority: consistency, live parity, and lower execution drift.
- Do not force 1200 USD/month immediately from 1200 USD capital, because that pushes the system back into fragile oversized risk.

## Step 4 - Real Account Canary Gate

Only after 30+ demo trades pass parity:

1. Open/use an Exness Real Raw or Pro account with lower spread than the current trial/demo feed.
2. Run canary with minimum lot first.
3. Compare real canary WR, spread, slippage, and rejected/modified orders against demo and MT5 WF.
4. Scale only if real canary stays within parity tolerance.

## Current Status

As of 2026-05-20:

- ACC2 bridge is connected.
- ACC2 demo Risk2 loop is active.
- Closed trades found by bridge history for magic 2505201202: 0.
- `decision_parity_test.py --mode wr` currently returns `WAIT`.

