#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _round(v: float, d: int) -> float:
    return round(float(v), d)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify expanded benchmark profiles.")
    p.add_argument(
        "--mode",
        choices=["locked", "recompute"],
        default="locked",
        help=(
            "locked: verify exact locked benchmark artifacts and config/model bindings "
            "(recommended for 100%% reproducibility across machines). "
            "recompute: rerun WF evaluator from raw data."
        ),
    )
    return p.parse_args()


def _expected_norm(expected: dict[str, Any]) -> dict[str, Any]:
    return {
        "sum_net_profit": _round(expected["sum_net_profit"], 2),
        "global_max_drawdown_pct": _round(expected["global_max_drawdown_pct"], 2),
        "profit_factor_global": _round(expected["profit_factor_global"], 4),
        "total_trades": int(expected["total_trades"]),
        "win_rate": _round(expected["win_rate"], 4),
    }


def _check_locked(repo: Path, manifest: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    all_ok = True
    for profile_name, spec in manifest.get("profiles", {}).items():
        expected = _expected_norm(spec["expected_wf"])
        cfg = repo / spec["config_path"]
        model = repo / spec["model_path"]
        scaler = repo / spec["scaler_path"]
        meta = repo / spec["model_meta_path"]
        summary_path = (
            repo / "outputs" / "acc1_dd39_local_refine_n40_summary.json"
            if profile_name.startswith("acc1_")
            else repo / "outputs" / "acc2_dd23_probe_n80_summary.json"
        )

        exists = all(p.exists() for p in [cfg, model, scaler, meta, summary_path])
        actual = None
        if exists:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            key = "best_dd39" if profile_name.startswith("acc1_") else "best_dd23"
            got = summary[key]
            actual = {
                "sum_net_profit": _round(got["sum_net_profit"], 2),
                "global_max_drawdown_pct": _round(got["global_max_drawdown_pct"], 2),
                "profit_factor_global": _round(got["profit_factor_global"], 4),
                "total_trades": int(got["total_trades"]),
                "win_rate": _round(got["win_rate"], 4),
            }
        passed = bool(exists and actual == expected)
        all_ok = all_ok and passed
        rows.append(
            {
                "profile": profile_name,
                "mode": "locked",
                "config_path": spec["config_path"],
                "expected": expected,
                "actual": actual,
                "artifacts_exist": exists,
                "passed": passed,
            }
        )
        print(f"[{profile_name}] locked_pass={passed}")
    return all_ok, rows


def _check_recompute(repo: Path, manifest: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    # Reuse exact evaluator used during constrained optimization.
    sys.path.insert(0, str(repo / "scripts"))
    import wf_constrained_optimizer as wfo  # type: ignore

    from xauusd_ai.config import load_settings

    rows: list[dict[str, Any]] = []
    all_ok = True
    templates = wfo._time_templates()
    for profile_name, spec in manifest.get("profiles", {}).items():
        cfg_path = repo / spec["config_path"]
        settings = load_settings(cfg_path)
        fold_cache, dataset = wfo._build_fold_cache(settings)
        candidate = dict(spec["candidate"])
        result, _ = wfo._evaluate_candidate(
            base_settings=settings,
            fold_cache=fold_cache,
            candidate=candidate,
            templates=templates,
            initial_balance=200.0,
            dd_min=0.0,
            dd_max=100.0,
            target_net=0.0,
        )
        expected = _expected_norm(spec["expected_wf"])
        actual = {
            "sum_net_profit": _round(result["sum_net_profit"], 2),
            "global_max_drawdown_pct": _round(result["global_max_drawdown_pct"], 2),
            "profit_factor_global": _round(result["profit_factor_global"], 4),
            "total_trades": int(result["total_trades"]),
            "win_rate": _round(result["win_rate"], 4),
            "dataset_rows": int(len(dataset)),
            "folds": int(len(fold_cache)),
            "test_start": fold_cache[0].test_start if fold_cache else None,
            "test_end": fold_cache[-1].test_end if fold_cache else None,
        }
        passed = actual["sum_net_profit"] == expected["sum_net_profit"] and actual["global_max_drawdown_pct"] == expected["global_max_drawdown_pct"] and actual["profit_factor_global"] == expected["profit_factor_global"] and actual["total_trades"] == expected["total_trades"] and actual["win_rate"] == expected["win_rate"]
        all_ok = all_ok and passed
        rows.append(
            {
                "profile": profile_name,
                "mode": "recompute",
                "config_path": spec["config_path"],
                "expected": expected,
                "actual": actual,
                "passed": passed,
            }
        )
        print(
            f"[{profile_name}] recompute "
            f"net={actual['sum_net_profit']:.2f} dd={actual['global_max_drawdown_pct']:.2f}% "
            f"pf={actual['profit_factor_global']:.4f} trades={actual['total_trades']} "
            f"wr={actual['win_rate']:.4f} pass={passed}"
        )
    return all_ok, rows


def main() -> int:
    args = _parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest_path = repo / "configs" / "benchmarks" / "wf_expanded_profiles_20260405.json"
    out_path = repo / "outputs" / "wf_verify_expanded_profiles_20260405.json"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profiles = manifest.get("profiles", {})
    if not profiles:
        print("No profiles in manifest.", file=sys.stderr)
        return 2

    if args.mode == "locked":
        all_ok, rows = _check_locked(repo, manifest)
    else:
        all_ok, rows = _check_recompute(repo, manifest)

    payload = {
        "manifest": str(manifest_path.relative_to(repo)),
        "mode": args.mode,
        "all_passed": all_ok,
        "profiles": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
