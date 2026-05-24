param(
    [string]$RatesFile    = "outputs\mt5_rates_export_202306_202603.csv",  # or merged file after pre-2023 export
    [string]$FeaturesOut  = "outputs\mt5_full_ict_wyckoff_features_v7.csv",
    [string]$FromDate     = "2023-06-01",
    [string]$ToDate       = "2026-04-01"
)
# ============================================================
# Rebuild v7 feature dataset + re-run full grid search + WF
# Run this AFTER adding pre-2023 rates (optional) OR after any
# feature change (v7 weekly_bias_d1, weekly_range_tight, d1_run_length).
#
# PRE-2023 RATES (optional, fixes bootstrap for folds 1-3):
#   1. Open MetaTrader 5 → Tools → Strategy Tester
#   2. Load EA: Rate_Exporter.mq5  (scripts/mt5/Rate_Exporter.mq5)
#   3. Symbol: XAUUSD  Timeframe: M5
#   4. Date range: 2019-01-01 to 2023-05-30
#   5. Run → EA exports to MT5 Common\Files\mt5_rates_export.csv
#   6. Copy to: outputs\mt5_rates_export_pre2023.csv
#   7. Then merge:
#      python scripts/merge_mt5_rates.py `
#          --old outputs\mt5_rates_export_pre2023.csv `
#          --new outputs\mt5_rates_export_202306_202603.csv `
#          --out outputs\mt5_rates_merged.csv
#   8. Re-run this script with -RatesFile outputs\mt5_rates_merged.csv `
#          -FeaturesOut outputs\mt5_full_ict_wyckoff_features_merged_v7.csv `
#          -FromDate 2019-01-01
# ============================================================

$ErrorActionPreference = "Stop"
$root = "f:\Trading_BOT_AUTO\Trade-Indicator"
Set-Location $root

function Log { param([string]$m); Write-Output ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) }

# ── Step 1: Rebuild feature CSV ───────────────────────────────────────────────
Log "Step 1: Building v7 features from $RatesFile ..."
python scripts/build_mt5_full_features.py `
    --rates "$RatesFile" `
    --out "$FeaturesOut" `
    --from-date "$FromDate" `
    --to-date "$ToDate"
if ($LASTEXITCODE -ne 0) { Write-Error "Feature build failed"; exit 1 }
Log "Feature CSV: $FeaturesOut"

# ── Step 2: Grid search (uses the new feature CSV) ────────────────────────────
# The grid search script reads the features CSV to find best hyperparameter rows.
# Output: outputs/target1200_mt5full_ict_foldcols_multilabel_focus/target1200_model_search.csv
Log "Step 2: Running grid search on v7 features (~30-60 min) ..."
python scripts/run_mt5_full_grid_search.py `
    --features "$FeaturesOut"
if ($LASTEXITCODE -ne 0) { Write-Error "Grid search failed"; exit 1 }
Log "Grid search done."

# ── Step 3: Re-identify best candidates ───────────────────────────────────────
Log "Step 3: Printing top strict-pass candidates from new grid ..."
python -c "
import pandas as pd
df = pd.read_csv('outputs/target1200_mt5full_ict_foldcols_multilabel_focus/target1200_model_search.csv')
strict = df[df.get('strict_pass_count', pd.Series(0)) == df.get('strict_pass_count', pd.Series(0)).max()]
print(strict[['row','tp','sl','h','risk_pct','max_positions','strict_pass_count']].head(10).to_string(index=False))
"

Log ""
Log "NEXT STEPS:"
Log "  1. From top candidates above, pick new v2 candidate (high strict_pass, balanced DD)"
Log "  2. Export its 28-fold manifest:"
Log "     python scripts/export_target1200_candidate_manifest.py --search-row <ROW> ..."
Log "  3. Run MT5 WF on it (scripts/run_mt5_wf_folds.ps1 -TrailingEnabled 0 -MaxDailyLoss 100)"
Log "  4. Re-run rolling selector with new candidates:"
Log "     .\scripts\windows\build_and_run_rolling_v2.ps1"
