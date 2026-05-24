# Rolling v3 Canary Runbook

## Verdict

`rolling_v3_current_selected` is the active live/canary package.

It passes the user's current gate on MT5 feedback:

- Each fold starts from 200 USD.
- Each validated fold final balance is at least 1200 USD.
- No loss folds.
- Worst DD is within 20%.
- Loaded signal gap is 0.
- Current live guard is green with fresh MT5 Python-feed features.

This is not the old 28-fold historical artifact by itself. The active package is a current rolling extension assembled from MT5-validated fold feedback.

## Active Artifacts

- Active manifest: `outputs/target1200_live_rolling_v3_current_selected/manifest.json`
- Active MT5 results: `outputs/target1200_live_rolling_v3_current_selected/mt5_wf_results.csv`
- Active preflight: `outputs/target1200_live_rolling_v3_current_selected/live_preflight_report.json`
- Active live guard: `outputs/target1200_live_rolling_v3_current_selected/live_canary_guard_report.json`
- Live features: `outputs/mt5_full_ict_wyckoff_features_live_current.csv`
- Live rates: `outputs/mt5_rates_export_202306_20260513_live_merged.csv`
- Live bridge executor: `scripts/rolling_v3_bridge_executor.py`
- Live one-shot wrapper: `scripts/windows/run_rolling_v3_bridge_once.ps1`
- rolling_v3 bridge startup: `scripts/windows/start_rolling_v3_bridge.ps1`
- WF-only EA source: `scripts/mt5/AI_Signal_Reader_Live.mq5`

## Current MT5 Gate

Latest selected current validation:

| Fold | Period | Signals | Loaded | Final | DD | Trades | Risk |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 29 | 2026-03-23 to 2026-04-23 | 250 | 250 | 1510.81 | -17.11% | 146 | 5.6% |
| 30 | 2026-04-23 to 2026-05-23 | 138 | 138 | 1254.04 | -14.62% | 96 | 6% |

Aggregate:

- Target pass: 2/2
- DD pass: 2/2
- Loss folds: 0
- Min final balance: 1254.04
- Worst DD: -17.11%
- Loaded signal max abs gap: 0
- Total trades: 242

These numbers are from MT5 Strategy Tester on the live broker symbol resolved from `XAUUSD` to `XAUUSDm`. Older `XAUUSD` results are not valid for live decisions on this terminal.

## Live Bridge Flow

Live orders are model/Python decisions sent through the MT5 HTTP bridge. EA/MetaEditor is only for WF/Strategy Tester tick-data validation.

Start the rolling_v3 bridge once:

```powershell
Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File scripts\windows\start_rolling_v3_bridge.ps1" -WindowStyle Hidden
```

Check bridge health:

```powershell
Invoke-WebRequest http://localhost:5601/health -UseBasicParsing
Invoke-WebRequest http://localhost:5601/account -UseBasicParsing
```

Dry-run the live protocol:

```powershell
.\scripts\windows\run_rolling_v3_bridge_once.ps1
```

Send a real order only with the explicit execute flag:

```powershell
.\scripts\windows\run_rolling_v3_bridge_once.ps1 -Execute
```

The executor still fail-closes if the real account balance is below the reset/deposit guard. Current strict blocker observed on 2026-05-13: balance 197.21 is below the 199.00 minimum tolerance for the 200 USD sleeve.

The wrapper:

1. Fetches fresh M5 bars directly from the live MT5 Python API.
2. Rebuilds ICT/Wyckoff features from MT5 bars.
3. Keeps the active MT5-validated signal package unchanged.
4. Runs the live guard with fresh features.
5. Runs the bridge executor against the latest closed M5 bar.

The executor will not replay historical signals. On first execute run it initializes state and skips any already-closed-bar signal unless `-AllowFirstRunTrade` is passed.

## Signal Update Only

Refresh the live signal/guard without placing orders:

```powershell
.\scripts\windows\update_rolling_v3_live_signal.ps1
```

By default this updates live bars/features and guard only. It does not regenerate the active signal package, because changing signal counts invalidates MT5 WF results.

Regenerate a new current-fold candidate only as a research/WF step:

```powershell
.\scripts\windows\update_rolling_v3_live_signal.ps1 -RegenerateCurrentFold
```

After `-RegenerateCurrentFold`, rerun MT5 WF and preflight before any live execution.

For WF/Strategy Tester only, copy the current signal to MT5 Files:

```powershell
.\scripts\windows\update_rolling_v3_live_signal.ps1 -CopySignalToMt5ForTester
```

## WF-Only EA Settings

Use this only in Strategy Tester / WF validation. Do not use this as the live order path.

Attach `AI_Signal_Reader_Live` to `XAUUSD` M5 and load:

```text
outputs/target1200_live_rolling_v3_current_selected/AI_Signal_Reader_Live_current.set
```

Important live inputs:

- `InpCSVFile=signals_for_mt5.csv`
- `InpReloadCsvOnNewBar=true`
- `InpSkipHistoricalSignalsOnInit=true`
- `InpRiskPct=5`
- `InpMaxPositions=1`
- `InpMaxRiskPct=5`
- `InpMaxExposurePct=5`
- `InpMaxDDKillPct=20`
- `InpTrailingEnabled=false`

`InpSkipHistoricalSignalsOnInit=true` is critical for tester/live-parity checks so old CSV signals are not replayed on attach.

## Guard Command

The prepare script defaults to the active current-selected artifact and should normally run without copying signals to MT5:

```powershell
.\scripts\windows\prepare_rolling_v3_canary.ps1 -Features outputs\mt5_full_ict_wyckoff_features_live_current.csv
```

No live/canary order should be placed if:

- `live_start_allowed=false`
- `live_allowed=false`
- loaded signal gap is nonzero
- features are stale
- current fold does not cover now
- risk/exposure/max positions exceed caps

## Cleanup Note

Old sweep directories and stale root output files were deleted to avoid confusion. The current active path is `target1200_live_rolling_v3_current_selected`; do not use old `target1200_live_rolling_v1/v2`, fixed-library, or proxy-only artifacts for live decisions.
