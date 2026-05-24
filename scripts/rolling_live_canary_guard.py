from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_now(value: str | None) -> pd.Timestamp:
    if value:
        ts = pd.Timestamp(value)
    else:
        ts = pd.Timestamp(datetime.now(timezone.utc))
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _resolve_path(raw_value: str | None, manifest_path: Path) -> Path | None:
    if not raw_value:
        return None
    raw = Path(raw_value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(
            [
                Path.cwd() / raw,
                manifest_path.parent / raw.name,
                manifest_path.parent / raw,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / raw).resolve() if not raw.is_absolute() else raw


def _feature_range(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    header = pd.read_csv(path, nrows=0)
    time_col = next((col for col in ["time", "open_time", "timestamp", "datetime"] if col in header.columns), None)
    if time_col is None:
        return {"path": str(path), "exists": True, "error": "no time column"}
    times = pd.read_csv(path, usecols=[time_col])[time_col]
    parsed = pd.to_datetime(times, utc=True, errors="coerce").dropna()
    if parsed.empty:
        return {"path": str(path), "exists": True, "error": "no parseable timestamps"}
    return {
        "path": str(path),
        "exists": True,
        "time_col": time_col,
        "rows": int(len(parsed)),
        "min_time": parsed.iloc[0].isoformat(),
        "max_time": parsed.iloc[-1].isoformat(),
    }


def _signal_range(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    header = pd.read_csv(path, nrows=0)
    if "open_time" not in header.columns:
        return {"path": str(path), "exists": True, "error": "no open_time column"}
    required = ["open_time", "direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]
    missing = sorted(set(required).difference(header.columns))
    if missing:
        return {"path": str(path), "exists": True, "error": f"missing columns: {missing}"}
    frame = pd.read_csv(path, usecols=required)
    times = frame["open_time"]
    parsed = pd.to_datetime(times, format="%Y.%m.%d %H:%M", utc=True, errors="coerce").dropna()
    numeric_columns = ["direction", "entry_price", "sl_price", "tp_price", "atr", "probability"]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    bad_numeric_rows = int(numeric.isna().any(axis=1).sum())
    nonpositive_plan_rows = int(
        (
            (numeric["atr"] <= 0.0)
            | (numeric["entry_price"] <= 0.0)
            | (numeric["sl_price"] <= 0.0)
            | (numeric["tp_price"] <= 0.0)
        ).sum()
    )
    if parsed.empty:
        return {"path": str(path), "exists": True, "rows": 0}
    return {
        "path": str(path),
        "exists": True,
        "rows": int(len(frame)),
        "min_open_time": parsed.min().isoformat(),
        "max_open_time": parsed.max().isoformat(),
        "bad_numeric_rows": bad_numeric_rows,
        "nonpositive_trade_plan_rows": nonpositive_plan_rows,
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def _find_current_fold(manifest: dict[str, Any], now_utc: pd.Timestamp) -> dict[str, Any] | None:
    for fold in manifest.get("folds", []) or []:
        start = pd.Timestamp(fold["test_start"], tz="UTC")
        end = pd.Timestamp(fold["test_end"], tz="UTC")
        if start <= now_utc < end:
            return dict(fold)
    return None


def _check_results(
    results_path: Path,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not results_path.exists():
        return {"path": str(results_path), "exists": False}, [f"missing MT5 WF results: {results_path}"]
    df = pd.read_csv(results_path)
    required = {"fold", "signals", "loaded_signals", "final_balance", "max_dd_pct", "trades"}
    missing = sorted(required.difference(df.columns))
    if missing:
        return {"path": str(results_path), "exists": True, "rows": int(len(df))}, [f"results missing columns: {missing}"]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    target_ok = df["final_balance"] >= float(args.target_balance)
    dd_ok = df["max_dd_pct"] >= float(args.min_dd_pct)
    loss = df["final_balance"] < float(args.deposit)
    loaded_gap = (df["loaded_signals"] - df["signals"]).abs()
    online_mode = bool(manifest.get("online_signal_stream"))
    manifest_signal_counts = {}
    if not online_mode:
        manifest_signal_counts = {
            _as_int(fold.get("fold")): _as_int(fold.get("signals"))
            for fold in manifest.get("folds", []) or []
        }
    result_signal_mismatches: list[dict[str, int]] = []
    result_loaded_mismatches: list[dict[str, int]] = []
    for _, row in df.iterrows():
        fold_id = _as_int(row.get("fold"))
        if fold_id not in manifest_signal_counts:
            continue
        expected = manifest_signal_counts[fold_id]
        result_signals = _as_int(row.get("signals"))
        result_loaded = _as_int(row.get("loaded_signals"))
        if result_signals != expected:
            result_signal_mismatches.append(
                {"fold": fold_id, "manifest_signals": expected, "result_signals": result_signals}
            )
        if result_loaded != expected:
            result_loaded_mismatches.append(
                {"fold": fold_id, "manifest_signals": expected, "loaded_signals": result_loaded}
            )
    if result_signal_mismatches:
        if not online_mode:
            blockers.append(f"MT5 result signal counts do not match manifest: {result_signal_mismatches}")
    if result_loaded_mismatches:
        if not online_mode:
            blockers.append(f"MT5 loaded signals do not match manifest: {result_loaded_mismatches}")
    target_fails = df.loc[~target_ok, "fold"].dropna().astype(int).tolist()
    dd_fails = df.loc[~dd_ok, "fold"].dropna().astype(int).tolist()
    loss_folds = df.loc[loss, "fold"].dropna().astype(int).tolist()
    if target_fails:
        blockers.append(f"target failures remain: {target_fails}")
    if dd_fails:
        blockers.append(f"DD failures remain: {dd_fails}")
    if loss_folds:
        blockers.append(f"loss folds remain: {loss_folds}")
    if float(loaded_gap.max()) > float(args.max_loaded_signal_gap):
        bad = df.loc[loaded_gap > float(args.max_loaded_signal_gap), "fold"].dropna().astype(int).tolist()
        blockers.append(f"loaded signal mismatch on folds: {bad}")
    return (
        {
            "path": str(results_path),
            "exists": True,
            "rows": int(len(df)),
            "target_pass_folds": int(target_ok.sum()),
            "dd_pass_folds": int(dd_ok.sum()),
            "loss_folds": int(loss.sum()),
            "min_final_balance": float(df["final_balance"].min()),
            "worst_dd_pct": float(df["max_dd_pct"].min()),
            "loaded_signal_max_abs_gap": float(loaded_gap.max()),
            "online_signal_stream": online_mode,
            "validation_signal_count_source": "mt5_results_csv" if online_mode else "manifest",
            "manifest_result_signal_mismatches": result_signal_mismatches,
            "manifest_loaded_signal_mismatches": result_loaded_mismatches,
            "total_trades": int(df["trades"].fillna(0).sum()),
        },
        blockers,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed guard before starting rolling MT5 canary/live.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--now", default=None, help="UTC timestamp override, e.g. 2026-05-12T15:00:00Z")
    parser.add_argument("--deposit", type=float, default=200.0)
    parser.add_argument("--target-balance", type=float, default=1200.0)
    parser.add_argument("--min-dd-pct", type=float, default=-20.0)
    parser.add_argument("--max-risk-pct", type=float, default=7.0)
    parser.add_argument("--max-exposure-pct", type=float, default=7.0)
    parser.add_argument("--max-positions", type=int, default=2)
    parser.add_argument("--max-loaded-signal-gap", type=float, default=0.0)
    parser.add_argument("--max-feature-staleness-days", type=float, default=3.0)
    parser.add_argument("--max-signal-file-age-minutes", type=float, default=30.0)
    parser.add_argument(
        "--max-signal-horizon-lag-minutes",
        type=float,
        default=15.0,
        help="Fail live if the latest signal open_time lags now by more than this many minutes.",
    )
    parser.add_argument("--max-online-decision-age-minutes", type=float, default=15.0)
    parser.add_argument("--max-online-target-bar-lag-minutes", type=float, default=15.0)
    parser.add_argument("--require-current-fold", action="store_true", default=True)
    parser.add_argument("--no-require-current-fold", dest="require_current_fold", action="store_false")
    parser.add_argument("--no-fail-exit", action="store_true")
    args = parser.parse_args()

    now_utc = _parse_now(args.now)
    blockers: list[str] = []
    warnings: list[str] = []

    manifest_path = args.manifest.resolve()
    manifest = _read_json(manifest_path)
    preflight = _read_json(args.preflight_report)
    online_mode = bool(manifest.get("online_signal_stream"))

    online_decision_summary: dict[str, Any] | None = None
    online_decision_fresh = False
    online_decision_should_trade: bool | None = None
    if online_mode:
        decision_path = _resolve_path(str(manifest.get("online_decision") or ""), manifest_path)
        if decision_path is None or not decision_path.exists():
            blockers.append(f"online decision missing: {manifest.get('online_decision')}")
        else:
            try:
                online_decision_summary = _read_json(decision_path)
                generated_at = pd.Timestamp(str(online_decision_summary.get("generated_at_utc")))
                if generated_at.tzinfo is None:
                    generated_at = generated_at.tz_localize("UTC")
                else:
                    generated_at = generated_at.tz_convert("UTC")
                age_minutes = (now_utc - generated_at).total_seconds() / 60.0
                online_decision_summary["age_minutes"] = round(age_minutes, 2)
                online_decision_should_trade = bool(online_decision_summary.get("should_trade"))
                online_decision_fresh = age_minutes <= float(args.max_online_decision_age_minutes)
                if not online_decision_fresh:
                    blockers.append(
                        f"online decision stale: {age_minutes:.1f} minutes exceeds "
                        f"cap {args.max_online_decision_age_minutes:.1f}"
                    )
                target_raw = online_decision_summary.get("target_bar_utc")
                if target_raw:
                    target_bar = pd.Timestamp(str(target_raw))
                    if target_bar.tzinfo is None:
                        target_bar = target_bar.tz_localize("UTC")
                    else:
                        target_bar = target_bar.tz_convert("UTC")
                    target_lag_minutes = (now_utc - target_bar).total_seconds() / 60.0
                    online_decision_summary["target_bar_lag_minutes"] = round(target_lag_minutes, 2)
                    if target_lag_minutes > float(args.max_online_target_bar_lag_minutes):
                        blockers.append(
                            f"online target bar stale: {target_bar.isoformat()} lags now by "
                            f"{target_lag_minutes:.1f} minutes; cap={args.max_online_target_bar_lag_minutes:.1f}"
                        )
                else:
                    blockers.append("online decision missing target_bar_utc")
            except Exception as exc:  # noqa: BLE001
                blockers.append(f"online decision invalid: {exc}")

    if not bool(preflight.get("live_allowed")):
        blockers.append(f"preflight live_allowed=false: {preflight.get('blockers', [])}")
    if preflight.get("blockers"):
        blockers.append(f"preflight blockers present: {preflight.get('blockers')}")
    source_flags = manifest.get("source_manifest_flags") if isinstance(manifest.get("source_manifest_flags"), dict) else {}
    if _is_true(source_flags.get("adaptive_per_fold")):
        blockers.append("source manifest is adaptive_per_fold; this is research/hindsight, not live protocol")
    if _is_true(source_flags.get("research_oracle_fold_selection")):
        blockers.append("source manifest is research_oracle_fold_selection")
    if _is_true(source_flags.get("selection_uses_current_fold_metrics")):
        blockers.append("source manifest selection uses current fold metrics")
    source_validation = manifest.get("source_validation") if isinstance(manifest.get("source_validation"), dict) else {}
    source_manifest_path = _resolve_path(str(source_validation.get("source_manifest") or ""), manifest_path)
    if source_manifest_path is not None and source_manifest_path.exists():
        source_manifest = _read_json(source_manifest_path)
        if _is_true(source_manifest.get("adaptive_per_fold")):
            blockers.append(f"source manifest is adaptive_per_fold: {source_manifest_path}")
        if _is_true(source_manifest.get("research_oracle_fold_selection")):
            blockers.append(f"source manifest is research_oracle_fold_selection: {source_manifest_path}")
        if _is_true(source_manifest.get("selection_uses_current_fold_metrics")):
            blockers.append(f"source manifest selection uses current fold metrics: {source_manifest_path}")

    result_summary, result_blockers = _check_results(args.results, manifest, args)
    blockers.extend(result_blockers)

    current_fold = _find_current_fold(manifest, now_utc)
    current_signal_summary: dict[str, Any] | None = None
    if current_fold is None:
        latest_end = max(
            (pd.Timestamp(fold["test_end"], tz="UTC") for fold in manifest.get("folds", []) or []),
            default=None,
        )
        message = "no manifest fold covers current UTC time"
        if latest_end is not None:
            message += f"; latest fold test_end={latest_end.isoformat()}"
        if args.require_current_fold:
            blockers.append(message)
        else:
            warnings.append(message)
    else:
        risk_pct = _as_float(current_fold.get("risk_pct"), 0.0)
        max_risk_pct = _as_float(current_fold.get("max_risk_pct"), risk_pct)
        max_exposure_pct = _as_float(current_fold.get("max_exposure_pct"), max_risk_pct)
        max_positions = _as_int(current_fold.get("max_positions"), 1)
        if risk_pct > args.max_risk_pct or max_risk_pct > args.max_risk_pct:
            blockers.append(f"current fold risk {max(risk_pct, max_risk_pct):.2f}% exceeds cap {args.max_risk_pct:.2f}%")
        if max_exposure_pct > args.max_exposure_pct:
            blockers.append(f"current fold exposure {max_exposure_pct:.2f}% exceeds cap {args.max_exposure_pct:.2f}%")
        if max_positions > args.max_positions:
            blockers.append(f"current fold max_positions {max_positions} exceeds cap {args.max_positions}")
        signal_path = _resolve_path(str(current_fold.get("csv")), manifest_path)
        if signal_path is None or not signal_path.exists():
            blockers.append(f"current fold signal CSV missing: {current_fold.get('csv')}")
        else:
            current_signal_summary = _signal_range(signal_path)
            if not current_signal_summary.get("exists"):
                blockers.append(f"current fold signal CSV missing: {signal_path}")
            elif current_signal_summary.get("rows", 0) <= 0:
                if online_mode and online_decision_fresh and online_decision_should_trade is False:
                    warnings.append("online decision is fresh no_trade; current signal CSV is intentionally empty")
                else:
                    blockers.append(f"current fold signal CSV has no signals: {signal_path}")
            elif current_signal_summary.get("error"):
                blockers.append(f"current fold signal CSV invalid: {current_signal_summary['error']}")
            else:
                if (
                    not online_mode
                    and int(current_signal_summary.get("rows", 0)) != _as_int(current_fold.get("signals"))
                ):
                    blockers.append(
                        "current fold signal CSV row count does not match manifest: "
                        f"csv_rows={current_signal_summary.get('rows')} manifest_signals={current_fold.get('signals')}"
                    )
                if int(current_signal_summary.get("bad_numeric_rows", 0)) > 0:
                    blockers.append(
                        f"current fold signal CSV has bad numeric rows: {current_signal_summary['bad_numeric_rows']}"
                    )
                if int(current_signal_summary.get("nonpositive_trade_plan_rows", 0)) > 0:
                    blockers.append(
                        "current fold signal CSV has nonpositive entry/sl/tp/atr rows: "
                        f"{current_signal_summary['nonpositive_trade_plan_rows']}"
                    )
                mtime = pd.Timestamp(str(current_signal_summary["mtime_utc"]))
                signal_age_minutes = (now_utc - mtime).total_seconds() / 60.0
                current_signal_summary["file_age_minutes"] = round(signal_age_minutes, 2)
                if signal_age_minutes > float(args.max_signal_file_age_minutes):
                    blockers.append(
                        f"current fold signal CSV file age {signal_age_minutes:.1f} minutes exceeds "
                        f"cap {args.max_signal_file_age_minutes:.1f}"
                    )
                max_open_time = pd.Timestamp(str(current_signal_summary["max_open_time"]))
                if max_open_time.tzinfo is None:
                    max_open_time = max_open_time.tz_localize("UTC")
                else:
                    max_open_time = max_open_time.tz_convert("UTC")
                signal_horizon_lag_minutes = (now_utc - max_open_time).total_seconds() / 60.0
                current_signal_summary["signal_horizon_lag_minutes"] = round(signal_horizon_lag_minutes, 2)
                if signal_horizon_lag_minutes > float(args.max_signal_horizon_lag_minutes):
                    blockers.append(
                        f"current fold signal horizon is stale: latest_signal={max_open_time.isoformat()} "
                        f"lags now by {signal_horizon_lag_minutes:.1f} minutes; "
                        f"cap={args.max_signal_horizon_lag_minutes:.1f}"
                    )
                if online_mode and online_decision_fresh and online_decision_should_trade is False:
                    blockers.append("online decision says no_trade but signal CSV contains trade rows")

    feature_summary: dict[str, Any] | None = None
    if args.features is not None:
        feature_summary = _feature_range(args.features)
        if not feature_summary.get("exists"):
            blockers.append(f"features file missing: {args.features}")
        elif feature_summary.get("error"):
            blockers.append(f"features invalid: {feature_summary['error']}")
        else:
            feature_max = pd.Timestamp(str(feature_summary["max_time"]))
            staleness_days = (now_utc - feature_max).total_seconds() / 86400.0
            feature_summary["staleness_days"] = round(staleness_days, 4)
            if staleness_days > float(args.max_feature_staleness_days):
                blockers.append(
                    f"features stale by {staleness_days:.2f} days; cap={args.max_feature_staleness_days:.2f}"
                )

    report = {
        "live_start_allowed": len(blockers) == 0,
        "now_utc": now_utc.isoformat(),
        "manifest": str(manifest_path),
        "preflight_report": str(args.preflight_report),
        "results": result_summary,
        "current_fold": current_fold,
        "current_signal": current_signal_summary,
        "online_decision": online_decision_summary,
        "features": feature_summary,
        "blockers": blockers,
        "warnings": warnings,
        "required_live_semantics": [
            "WF/preflight must pass",
            "a manifest fold must cover current UTC time",
            "features must be fresh",
            "current fold signal horizon must be fresh",
            "online signal stream must have a fresh online_decision when enabled",
            "current fold signal CSV must exist and have signals",
            "risk/exposure/max_positions must stay within caps",
        ],
    }

    out_path = args.out or (manifest_path.parent / "live_canary_guard_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["live_start_allowed"] or args.no_fail_exit:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
