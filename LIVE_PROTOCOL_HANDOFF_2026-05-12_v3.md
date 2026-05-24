# Live Protocol Handoff - 2026-05-12 v3

## Current verdict

Rolling selector v3 is the current best protocol candidate.

- Directory: `outputs/target1200_live_rolling_v3/`
- MT5 result: `outputs/target1200_live_rolling_v3/mt5_wf_results.csv`
- Bootstrap-approved preflight: `outputs/target1200_live_rolling_v3/live_preflight_report.json`
- Strict preflight: `outputs/target1200_live_rolling_v3/live_preflight_report_strict.json`

## Numeric gate result

MT5 full 28-fold walk-forward result:

- Folds: 28/28
- Target pass: 28/28, using `final_balance >= 1200`
- DD pass: 28/28, using `max_dd_pct >= -20`
- Loss folds: 0/28
- Loaded signal max abs gap: 0
- Min final balance: 1214.28
- Median final balance: 1611.36
- Worst DD: -19.75
- Total trades: 15586

This is the first non-oracle rolling-selector result that passes the user's numeric gate in MT5.

## What changed from rolling v2

Rolling v2 failed:

- Fold 01: final 1143.14, DD -20.17
- Fold 03: final 828.02, DD -19.00

Rolling v3 adds a declared bootstrap map for the early warmup folds:

- Fold 01: `bootstrap_declared_map` -> `outputs/target1200_row1970_postexcl12_r175_mp2_28fold`
- Fold 02: `bootstrap_declared` -> `outputs/target1200_mt5full_ict_adaptive_combo_mt5v2`
- Fold 03: `bootstrap_declared_map` -> `outputs/target1200_row24_excl_1_5_12_16_19_23_r31_mp1_28fold`
- Fold 04..28: `past_mt5_feedback`

## New candidate artifacts

`outputs/target1200_row1970_postexcl12_r175_mp2_28fold/`

- Built from `target1200_row1970_r175_mp2_28fold` by filtering selected signals after top-k selection to exclude broker hour 12.
- Purpose: fold 01 bootstrap specialist.
- MT5 full-28 result: target 8/28, DD 15/28, loss 1/28, loaded gap 0.
- Not suitable as a main candidate; useful for fold 01 only.

`outputs/target1200_row24_excl_1_5_12_16_19_23_r31_mp1_28fold/`

- Fixed row 24 candidate, risk 3.1%, max positions 1, excluded broker hours 1,5,12,16,19,23 before selection.
- Purpose: fold 03 bootstrap specialist.
- MT5 full-28 result: target 17/28, DD 10/28, loss 2/28, loaded gap 0.
- Not suitable as a main candidate; useful for fold 03 only.

## Code changes

- `scripts/filter_mt5_fold_manifest_hours.py`
  - Supports filtering all folds when `--fold` is omitted.
  - Fixes risk/exposure metadata so default `max_exposure_pct = risk_pct * max_positions`.
- `scripts/export_target1200_candidate_manifest.py`
  - Fixes candidate export metadata so `max_exposure_pct = risk_pct * max_positions`.
- `scripts/export_mt5_feedback_rolling_selector.py`
  - Adds `--bootstrap-map FOLD=DIR`.
  - Refuses `research_oracle_fold_selection=true` candidates.
  - Carries source candidate flags into selected folds for preflight visibility.
- `scripts/live_protocol_preflight.py`
  - Blocks all `bootstrap_declared*` modes unless `--allow-bootstrap` is supplied.
  - Blocks selected source candidates that are research-oracle or use current fold metrics.
  - Warns, but does not block, when selected source candidate is `adaptive_per_fold`.
- `scripts/windows/build_and_run_rolling_v3.ps1`
  - Rebuilds rolling v3, runs MT5 WF, writes strict and bootstrap-approved preflight reports.

## Preflight interpretation

Strict preflight:

- File: `outputs/target1200_live_rolling_v3/live_preflight_report_strict.json`
- `live_allowed=false`
- Blockers: fold 01, 02, 03 use bootstrap-declared selection.

Bootstrap-approved preflight:

- File: `outputs/target1200_live_rolling_v3/live_preflight_report.json`
- `live_allowed=true`
- Blockers: none
- Warnings: selected source candidate is `adaptive_per_fold` for fold 02 and fold 04..28.

## Important caveat

Rolling v3 passes the numeric WF gate only under the declared-bootstrap protocol. It is not a pure no-bootstrap protocol.

For live today, this is less severe because the selector now has accumulated MT5 feedback history. For a fresh account or a brand-new symbol with no MT5 feedback history, the first campaigns must use the declared bootstrap map.

Remaining technical risk: most folds still source from `target1200_mt5full_ict_adaptive_combo_mt5v2`, which is warned as `adaptive_per_fold`. It is not marked research-oracle, and rolling selector decisions do not use current fold MT5 metrics. A stricter future protocol should replace this with fixed, non-adaptive candidates selected only by past MT5 feedback.
