# Live Trading Protocol

Status ngay 2026-05-12: live/canary bi khoa cho den khi manifest live protocol vuot preflight va MT5 WF tick-by-tick vuot gate.

## Nguyen tac bat buoc

- Moi fold reset ve `Deposit=200`.
- Target moi fold: `final_balance >= 1200`.
- Drawdown moi fold: `max_dd_pct >= -20`.
- Fold lo: toi da 1, nhung target gate mac dinh van la 0 fold truot target.
- MT5 phai load dung signal: `loaded_signals == signals`.
- Khong dung manifest `adaptive_per_fold` hoac bat ky manifest nao chon tham so bang ket qua fold hien tai/tuong lai.

## Pipeline live dung

1. Build lai feature tu MT5 bars, khong dung CSV cu:

```powershell
python scripts\build_mt5_full_features.py
```

2. Train/search signal universe tren feature MT5:

```powershell
python scripts\train_target1200_signal_universe.py --features outputs\mt5_full_ict_wyckoff_features_202306_202603.csv --out-dir outputs\target1200_mt5full_ict_foldcols_multilabel_focus
```

3. Export manifest no-lookahead. Selector chi duoc dung fold OOS da xay ra truoc fold dang trade:

```powershell
python scripts\export_target1200_live_protocol_universe.py `
  --search-csv outputs\target1200_mt5full_ict_foldcols_multilabel_focus\target1200_model_search.csv `
  --out-dir outputs\target1200_live_protocol_trade_side_v1 `
  --direction-mode trade_side `
  --side all `
  --max-risk-fraction 0.04 `
  --max-positions 1 `
  --min-history-folds 3 `
  --lookback-folds 6 `
  --min-history-pass-rate 0.5
```

4. Chay MT5 WF tuan tu. Khong chay song song nhieu WF vi EA dung chung `signals_for_mt5.csv`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_mt5_wf_folds.ps1 `
  -Manifest outputs\target1200_live_protocol_trade_side_v1\manifest.json `
  -RiskPct 2 `
  -MaxPositions 1 `
  -MaxDDKillPct 20 `
  -TargetBalanceStop 1200 `
  -TrailingEnabled 0 `
  -MaxHoldBars -1 `
  -Deposit 200
```

5. Chay preflight. Chi khi `live_allowed=true` moi duoc canary/live:

```powershell
python scripts\live_protocol_preflight.py `
  --manifest outputs\target1200_live_protocol_trade_side_v1\manifest.json `
  --results outputs\target1200_live_protocol_trade_side_v1\mt5_wf_results.csv
```

## Research-only outputs

Thu muc nhu `outputs/target1200_mt5full_ict_adaptive_combo_mt5v2` chi duoc dung de phan tich. Ket qua 26/28 tot hon cac sweep cu, nhung van khong phai live protocol vi tham so duoc chon theo fold. Preflight se block loai manifest nay.

`outputs/target1200_mt5_strict_combo_v1` la research/oracle combo moi nhat. MT5 tick-by-tick da pass target/DD 28/28 fold voi `TargetBalanceStop=1200`, nhung van bi chan live vi fold selection duoc lap rap tu best result cua tung fold sau khi da biet OOS.

## Ket qua validation hien tai

- `outputs/target1200_mt5full_ict_adaptive_combo_mt5v2`: research-only, 26/28 fold dat target, DD pass 27/28, bi block vi adaptive/hindsight va fold 1/3 fail.
- `outputs/target1200_mt5_strict_combo_v1`: research-only, 28/28 fold dat `final_balance >= 1200`, DD pass 28/28, `worst_dd_pct=-19.75`, `loaded_signal_max_abs_gap=0`; bi block live vi khong phai no-lookahead protocol.
- `outputs/target1200_live_protocol_trade_side_v1`: live_protocol=true, no-lookahead, MaxPos1. MT5 pass DD 28/28 nhung chi 2/28 fold dat `1200`; bi block.
- `outputs/target1200_live_protocol_trade_side_v3_maxpos2`: live_protocol=true, no-lookahead, MaxPos2. MT5 dat target 19/28 nhung DD pass 19/28; bi block.

## Risk sleeve live

- Moi cycle live la mot sleeve `200 USD`.
- Khong compound trong sleeve neu protocol chua cho phep.
- Khi sleeve dat target, khoa loi nhuan/withdraw phan vuot target roi reset sleeve moi.
- Dung sleeve ngay neu DD cham `-20%`, MT5 load signal lech, spread/drift vuot cap, bridge mat ket noi, hoac lenh ngoai protocol xuat hien.
- Canary bat dau bang demo/paper truoc; real live chi mo sau khi preflight va canary deu pass.
