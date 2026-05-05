#!/usr/bin/env python3
"""
verify_live_wf_parity.py
========================
Lớp 1 — Static parity check giữa LIVE config và WF (Combo #133) ground-truth.

Mục đích:
  - PHÁT HIỆN config drift trước khi deploy production.
  - Block deploy nếu lệch các tham số critical → tránh trường hợp WF thắng nhưng
    live thua do config khác.

Ground-truth của WF Combo #133 lấy từ:
  1. Constants override trong scripts/show_combo133_daily.py
     (MIN_CONF, REQ_TREND, MIN_STRAT, BLOCKED, D1_GATE, TRAIN_BARS, ...)
  2. Defaults trong src/xauusd_ai/backtesting/engine.py
     (spread/slippage/commission, partial_tp, trailing_sl)
  3. Một số yaml-driven params được override không-đổi mỗi run
     (risk_per_trade, max_open_positions, partial_tp_*, trailing_sl.*).

So sánh đối tượng: configs/live_acc1.yaml

Exit codes:
  0 = parity OK (an toàn để deploy)
  1 = critical mismatch (BLOCK deploy)
  2 = warning only (proceed with caution)

Usage:
  python scripts/verify_live_wf_parity.py
  python scripts/verify_live_wf_parity.py --live-config configs/live_acc1.yaml --strict

Tham khảo: LIVE_VS_WF_CHECKLIST.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# ──────────────────────────────────────────────────────────────────────────
# GROUND-TRUTH: Combo #133 (frozen 2026-04-26)
# Nguồn: scripts/show_combo133_daily.py (constants) + engine.py (defaults)
# Giá trị này KHÔNG được sửa trừ khi cùng lúc retrain WF + cập nhật baseline.
# ──────────────────────────────────────────────────────────────────────────
WF_COMBO133: dict[str, Any] = {
    # === Signal filters (override trong show_combo133_daily.py) ===
    "strategy.sideway_min_confidence":           0.70,
    "strategy.volatile_min_confidence":          0.70,
    "strategy.min_strategy_score":               0.00,
    "strategy.sideway_min_strategy_score":       0.00,
    "strategy.strong_volatility_min_strategy_score": 0.00,
    "strategy.require_trend_alignment":          False,
    "strategy.d1_trend_gate":                    False,
    "strategy.blocked_hours_utc":                [3, 15, 17, 22, 23],

    # === Risk sizing (WF dùng) ===
    "risk.min_confidence":                       0.70,
    "risk.risk_per_trade":                       0.030,
    "risk.max_drawdown_kill_pct":                0.15,
    "risk.max_open_positions":                   3,
    "risk.stop_loss_atr_multiple":               1.5,
    "risk.take_profit_rr":                       5.5,
    "risk.volatility_risk_scaling_enabled":      False,

    # === Partial TP (engine.py) ===
    "risk.partial_tp_enabled":                   True,
    "risk.partial_tp_rr":                        2.5,
    "risk.partial_tp_pct":                       0.5,

    # === Trailing SL (engine.py) ===
    "execution.trailing_sl.enabled":             True,
    "execution.trailing_sl.breakeven_at_rr":     0.5,
    "execution.trailing_sl.activation_rr":       1.0,
    "execution.trailing_sl.trail_atr_multiple":  1.0,

    # === Friction (engine.py defaults — WF assumes these) ===
    # Live không dùng các value này (live spread = MT5 thực) nhưng để verify
    # config không tự ý override sai.
    "risk.spread_cost_rr":                       0.10,  # default if absent
    "risk.slippage_rr":                          0.05,
    "risk.commission_rr":                        0.02,
}

# Params chỉ là cảnh báo (warning), không block deploy.
WF_WARN_ONLY = {
    "strategy.silver_bullet_enabled",          # WF không có — live có boost ~3-5%
    "strategy.silver_bullet_confidence_boost",
    "risk.spread_cost_rr",                     # Live spread thực từ MT5
    "risk.slippage_rr",
    "risk.commission_rr",
}


def get_nested(d: dict, dotted: str) -> Any:
    """Lấy value theo dotted-path. Return _MISSING nếu không có."""
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


_MISSING = object()


def values_equal(a: Any, b: Any) -> bool:
    if a is _MISSING or b is _MISSING:
        return False
    # Lists compared as sorted (blocked_hours order không quan trọng)
    if isinstance(a, list) and isinstance(b, list):
        return sorted(a) == sorted(b)
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    return a == b


def fmt(v: Any) -> str:
    if v is _MISSING:
        return "<missing>"
    return repr(v)


def verify(live_path: Path, strict: bool = False) -> int:
    if not live_path.exists():
        print(f"❌ Live config not found: {live_path}", file=sys.stderr)
        return 1

    with live_path.open("r", encoding="utf-8") as f:
        live_cfg = yaml.safe_load(f) or {}

    print("=" * 78)
    print("  LIVE ↔ WF (Combo #133) PARITY CHECK")
    print(f"  Live config: {live_path}")
    print(f"  Strict mode: {strict}")
    print("=" * 78)

    critical_mismatches: list[tuple[str, Any, Any]] = []
    warnings: list[tuple[str, Any, Any]] = []
    ok_count = 0

    for key, expected in WF_COMBO133.items():
        actual = get_nested(live_cfg, key)
        if values_equal(actual, expected):
            ok_count += 1
            continue

        is_warn = key in WF_WARN_ONLY
        # Nếu missing và là friction → live thực dùng MT5 spread → warn only
        if actual is _MISSING and key in WF_WARN_ONLY:
            warnings.append((key, expected, actual))
            continue

        if is_warn:
            warnings.append((key, expected, actual))
        else:
            critical_mismatches.append((key, expected, actual))

    # Report
    print(f"\n[OK]    {ok_count}/{len(WF_COMBO133)} params match WF baseline")
    if warnings:
        print(f"\n[WARN]  {len(warnings)} non-critical differences:")
        for key, exp, act in warnings:
            print(f"          {key}")
            print(f"            WF expects:  {fmt(exp)}")
            print(f"            Live has:    {fmt(act)}")
    if critical_mismatches:
        print(f"\n[FAIL]  {len(critical_mismatches)} CRITICAL mismatches:")
        for key, exp, act in critical_mismatches:
            print(f"          {key}")
            print(f"            WF expects:  {fmt(exp)}")
            print(f"            Live has:    {fmt(act)}")

    print("=" * 78)
    if critical_mismatches:
        print("  RESULT: ❌ FAIL — Do NOT deploy. Fix mismatches above first.")
        print("=" * 78)
        return 1
    if warnings and strict:
        print("  RESULT: ⚠️  WARN (strict mode) — Review warnings before deploy.")
        print("=" * 78)
        return 2
    print("  RESULT: ✅ PASS — Live config matches WF Combo #133 baseline.")
    print("=" * 78)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-config", type=Path, default=Path("configs/live_acc1.yaml"))
    ap.add_argument("--strict", action="store_true", help="Fail on warnings too")
    args = ap.parse_args()
    return verify(args.live_config, args.strict)


if __name__ == "__main__":
    sys.exit(main())
