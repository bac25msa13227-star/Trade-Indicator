#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

from xauusd_ai.config import load_settings
from xauusd_ai.features.dataset import FEATURE_COLUMNS


def _load_phase2_module() -> Any:
    mod_path = Path(__file__).with_name("wf_phase2_optimizer.py")
    spec = importlib.util.spec_from_file_location("wf_phase2_optimizer", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


def _is_metric_ok(actual: float, expected: float, tol: float) -> bool:
    return abs(float(actual) - float(expected)) <= float(tol)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify locked WF benchmarks (ACC1/ACC2) with exact expected metrics."
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/benchmarks/wf_locked_breakthrough_20260401.json"),
    )
    p.add_argument(
        "--report-out",
        type=Path,
        default=Path("outputs/wf_locked_verify_report.json"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base_config = Path(manifest["base_config"])
    initial_balance = float(manifest.get("initial_balance", 200.0))
    tol = manifest.get("metric_tolerances", {})
    profiles = manifest.get("profiles", {})
    overrides = manifest.get("base_overrides", {})

    mod = _load_phase2_module()
    settings = load_settings(base_config)
    for dotted_key, value in overrides.items():
        _set_nested_attr(settings, dotted_key, value)

    t0 = time.time()
    fold_cache, dataset = mod._build_fold_cache(settings)
    templates = mod._time_templates()

    report: dict[str, Any] = {
        "manifest": str(args.manifest),
        "base_config": str(base_config),
        "feature_count_runtime": len(FEATURE_COLUMNS),
        "feature_mode": "locked_benchmark_59_features",
        "dataset_rows": len(dataset),
        "folds": len(fold_cache),
        "test_window": [fold_cache[0].test_start, fold_cache[-1].test_end],
        "profiles": {},
        "all_passed": True,
        "elapsed_sec": None,
    }

    print("=" * 96, flush=True)
    print("LOCKED WF BENCHMARK VERIFY", flush=True)
    print(f"Manifest    : {args.manifest}", flush=True)
    print(f"Base config : {base_config}", flush=True)
    print(
        f"Dataset     : rows={len(dataset):,} | folds={len(fold_cache)} "
        f"| test=[{fold_cache[0].test_start} -> {fold_cache[-1].test_end}]",
        flush=True,
    )
    print("=" * 96, flush=True)

    for name, profile in profiles.items():
        expected = profile["expected"]
        candidate = profile["candidate"]
        result, _ = mod._evaluate_candidate(
            base_settings=settings,
            fold_cache=fold_cache,
            candidate=candidate,
            templates=templates,
            initial_balance=initial_balance,
            dd_min=0.0,
            dd_max=100.0,
            target_net=0.0,
            keep_trades=False,
        )
        actual = {
            "sum_net_profit": float(result["sum_net_profit"]),
            "global_max_drawdown_pct": float(result["global_max_drawdown_pct"]),
            "profit_factor_global": float(result["profit_factor_global"]),
            "total_trades": int(result["total_trades"]),
            "win_rate": float(result["win_rate"]),
        }
        checks = {
            "sum_net_profit": _is_metric_ok(
                actual["sum_net_profit"], expected["sum_net_profit"], tol.get("sum_net_profit", 0.0)
            ),
            "global_max_drawdown_pct": _is_metric_ok(
                actual["global_max_drawdown_pct"],
                expected["global_max_drawdown_pct"],
                tol.get("global_max_drawdown_pct", 0.0),
            ),
            "profit_factor_global": _is_metric_ok(
                actual["profit_factor_global"], expected["profit_factor_global"], tol.get("profit_factor_global", 0.0)
            ),
            "total_trades": _is_metric_ok(
                actual["total_trades"], expected["total_trades"], tol.get("total_trades", 0.0)
            ),
            "win_rate": _is_metric_ok(actual["win_rate"], expected["win_rate"], tol.get("win_rate", 0.0)),
        }
        passed = all(checks.values())
        report["all_passed"] = bool(report["all_passed"] and passed)
        report["profiles"][name] = {
            "expected": expected,
            "actual": actual,
            "checks": checks,
            "passed": passed,
            "candidate": candidate,
        }

        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] {name} | net={actual['sum_net_profit']:.2f} | dd={actual['global_max_drawdown_pct']:.2f}% "
            f"| pf={actual['profit_factor_global']:.4f} | trades={actual['total_trades']} | wr={actual['win_rate']:.4f}",
            flush=True,
        )

    report["elapsed_sec"] = round(time.time() - t0, 2)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 96, flush=True)
    print(f"Report      : {args.report_out}", flush=True)
    print(f"All passed  : {report['all_passed']}", flush=True)
    print("=" * 96, flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
