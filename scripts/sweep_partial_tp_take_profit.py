#!/usr/bin/env python3
"""
sweep_partial_tp_take_profit.py
================================
Task B — WF sweep ô (partial_tp_rr, take_profit_rr) tìm cấu hình cải thiện
fold 24-29 nhưng KHÔNG làm giảm baseline.

Quy tắc apply (cam kết với user):
  - Total Return/fold avg ≥ baseline AND
  - |Max DD avg| ≤ |baseline DD|

Chạy 9 combo (3×3) + 1 baseline (=10 WF runs).
Mỗi run: --combo133 --risk-pct 0.05 --no-compound --test-start 2024-01-01
Cache reused giữa các run (dataset prep chung).

Output:
  outputs/sweep_ptp_tp_results.json
  outputs/sweep_ptp_tp_results.md
  outputs/sweep_ptp_tp/wf_<ptp>_<tp>.txt   (raw stdout per combo)
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml  # PyYAML

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "configs" / "acc1_v14pp_profit.yaml"
OUTPUT_DIR = ROOT / "outputs" / "sweep_ptp_tp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sweep grid
PARTIAL_TP_RR_GRID = [1.5, 2.0, 2.5]
TAKE_PROFIT_RR_GRID = [3.5, 4.5, 5.5]

# Common WF flags (parity with baseline)
# NOTE: scoped to fold 24-29 (weak/bull-regime) for tractable runtime.
# test-start 2025-11-01 yields ~6 folds of size 6000 bars each.
# Winners must then be verified on full range manually (or via deploy gate).
WF_FLAGS = [
    "--no-rr-sweep",
    "--cache",
    "--test-start", "2025-11-01",
    "--test-bars", "6000",
    "--step-bars", "6000",
    "--no-compound",
    "--combo133",
    "--risk-pct", "0.05",
]


@dataclass
class RunResult:
    partial_tp_rr: float
    take_profit_rr: float
    folds: int = 0
    return_per_fold_pct: float = 0.0
    max_dd_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    auc_avg: float = 0.0
    log_path: str = ""
    ok: bool = False


def make_config(ptp: float, tp: float) -> Path:
    """Create temp config copy with overridden partial_tp_rr + take_profit_rr."""
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg.setdefault("risk", {})["take_profit_rr"] = tp
    cfg["risk"]["partial_tp_rr"] = ptp
    f = tempfile.NamedTemporaryFile(
        prefix=f"sweep_ptp{ptp}_tp{tp}_",
        suffix=".yaml",
        delete=False,
        dir=str(OUTPUT_DIR),
        mode="w",
        encoding="utf-8",
    )
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    f.close()
    return Path(f.name)


# Parse key WF summary lines
RE_FOLDS = re.compile(r"Folds completed\s*:\s*(\d+)")
RE_RETURN_PER_FOLD = re.compile(r"Return/fold\s*:\s*\+?(-?[\d.]+)%")
RE_MAX_DD = re.compile(r"Max Drawdown\s*:\s*(-?[\d.]+)%")
RE_WIN_RATE = re.compile(r"Win Rate avg\s*:\s*(-?[\d.]+)%")
RE_PF = re.compile(r"Profit Factor\s*:\s*(-?[\d.]+)")
RE_AUC = re.compile(r"ROC-AUC\s*avg\s*:\s*(-?[\d.]+)")


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for key, rx in [
        ("folds", RE_FOLDS),
        ("return_per_fold_pct", RE_RETURN_PER_FOLD),
        ("max_dd_pct", RE_MAX_DD),
        ("win_rate", RE_WIN_RATE),
        ("profit_factor", RE_PF),
        ("auc_avg", RE_AUC),
    ]:
        m = rx.search(text)
        if m:
            out[key] = float(m.group(1))
    return out


def run_wf(config_path: Path, log_path: Path) -> bool:
    cmd = [sys.executable, str(ROOT / "scripts" / "walkforward_ict_wyckoff.py"),
           str(config_path)] + WF_FLAGS
    print(f"  → {' '.join(cmd[1:])}")
    print(f"  → log: {log_path.relative_to(ROOT)}")
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode == 0


def run_one(ptp: float, tp: float, label: str = "") -> RunResult:
    print()
    print(f"━━━ Run {label}  partial_tp_rr={ptp}  take_profit_rr={tp} ━━━")
    cfg = make_config(ptp, tp)
    log = OUTPUT_DIR / f"wf_ptp{ptp}_tp{tp}.txt"
    ok = run_wf(cfg, log)
    metrics = parse_log(log) if log.exists() else {}
    res = RunResult(
        partial_tp_rr=ptp,
        take_profit_rr=tp,
        folds=int(metrics.get("folds", 0)),
        return_per_fold_pct=metrics.get("return_per_fold_pct", 0.0),
        max_dd_pct=metrics.get("max_dd_pct", 0.0),
        win_rate=metrics.get("win_rate", 0.0),
        profit_factor=metrics.get("profit_factor", 0.0),
        auc_avg=metrics.get("auc_avg", 0.0),
        log_path=str(log.relative_to(ROOT)),
        ok=ok and bool(metrics),
    )
    print(f"     folds={res.folds}  ret/fold={res.return_per_fold_pct:+.1f}%"
          f"  DD={res.max_dd_pct:.1f}%  WR={res.win_rate:.1f}%"
          f"  PF={res.profit_factor:.3f}")
    return res


def write_markdown(baseline: RunResult, runs: list[RunResult], path: Path) -> None:
    lines = []
    lines.append("# WF Sweep: `partial_tp_rr` × `take_profit_rr`")
    lines.append("")
    lines.append(f"**Baseline** (current live config): "
                 f"`partial_tp_rr={baseline.partial_tp_rr}`, "
                 f"`take_profit_rr={baseline.take_profit_rr}`")
    lines.append(f"- Return/fold: **{baseline.return_per_fold_pct:+.1f}%**")
    lines.append(f"- Max DD: **{baseline.max_dd_pct:.1f}%**")
    lines.append(f"- WR: {baseline.win_rate:.1f}% | PF: {baseline.profit_factor:.3f} "
                 f"| AUC: {baseline.auc_avg:.3f} | folds: {baseline.folds}")
    lines.append("")
    lines.append("## Sweep Matrix (Return/fold %)")
    lines.append("")
    lines.append("| partial_tp_rr ↓ \\ take_profit_rr → | " +
                 " | ".join(f"{tp}" for tp in TAKE_PROFIT_RR_GRID) + " |")
    lines.append("|" + "---|" * (len(TAKE_PROFIT_RR_GRID) + 1))
    by_key = {(r.partial_tp_rr, r.take_profit_rr): r for r in runs}
    for ptp in PARTIAL_TP_RR_GRID:
        row = [f"**{ptp}**"]
        for tp in TAKE_PROFIT_RR_GRID:
            r = by_key.get((ptp, tp))
            if r is None or not r.ok:
                row.append("_n/a_")
            else:
                row.append(f"{r.return_per_fold_pct:+.1f}% (DD {r.max_dd_pct:.1f}%)")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Winners (P&L ≥ baseline AND |DD| ≤ |baseline|)")
    lines.append("")
    winners = [
        r for r in runs
        if r.ok
        and r.return_per_fold_pct >= baseline.return_per_fold_pct
        and abs(r.max_dd_pct) <= abs(baseline.max_dd_pct)
    ]
    if winners:
        lines.append("| ptp | tp | Return/fold | Max DD | WR | PF | AUC |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in sorted(winners, key=lambda x: -x.return_per_fold_pct):
            lines.append(f"| {r.partial_tp_rr} | {r.take_profit_rr} "
                         f"| {r.return_per_fold_pct:+.1f}% "
                         f"| {r.max_dd_pct:.1f}% "
                         f"| {r.win_rate:.1f}% "
                         f"| {r.profit_factor:.3f} "
                         f"| {r.auc_avg:.3f} |")
    else:
        lines.append("_No combo beats baseline on both metrics. Keep current config._")
    lines.append("")
    lines.append("## Full Results")
    lines.append("")
    lines.append("| ptp | tp | folds | Return/fold | Max DD | WR | PF | AUC | log |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in [baseline] + sorted(runs, key=lambda x: (x.partial_tp_rr, x.take_profit_rr)):
        tag = "**BASE**" if r is baseline else ""
        lines.append(f"| {r.partial_tp_rr}{tag} | {r.take_profit_rr} | {r.folds} "
                     f"| {r.return_per_fold_pct:+.1f}% "
                     f"| {r.max_dd_pct:.1f}% "
                     f"| {r.win_rate:.1f}% "
                     f"| {r.profit_factor:.3f} "
                     f"| {r.auc_avg:.3f} "
                     f"| `{r.log_path}` |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("=" * 78)
    print("  WF Sweep — partial_tp_rr × take_profit_rr")
    print(f"  Base config : {BASE_CONFIG.relative_to(ROOT)}")
    print(f"  Grid        : ptp ∈ {PARTIAL_TP_RR_GRID}, tp ∈ {TAKE_PROFIT_RR_GRID}")
    print(f"  WF flags    : {' '.join(WF_FLAGS)}")
    print("=" * 78)

    # Load baseline values from base config
    base_cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    base_ptp = float(base_cfg.get("risk", {}).get("partial_tp_rr", 1.2))
    base_tp = float(base_cfg.get("risk", {}).get("take_profit_rr", 3.5))

    # 1) Baseline run
    baseline = run_one(base_ptp, base_tp, label="(baseline)")
    if not baseline.ok:
        print("❌ Baseline run failed; aborting sweep.", file=sys.stderr)
        return 2

    # 2) Sweep runs (skip combo == baseline)
    runs: list[RunResult] = []
    total = len(PARTIAL_TP_RR_GRID) * len(TAKE_PROFIT_RR_GRID)
    i = 0
    for ptp in PARTIAL_TP_RR_GRID:
        for tp in TAKE_PROFIT_RR_GRID:
            i += 1
            if abs(ptp - base_ptp) < 1e-9 and abs(tp - base_tp) < 1e-9:
                runs.append(baseline)
                continue
            runs.append(run_one(ptp, tp, label=f"{i}/{total}"))

    # 3) Write reports
    json_path = ROOT / "outputs" / "sweep_ptp_tp_results.json"
    md_path = ROOT / "outputs" / "sweep_ptp_tp_results.md"
    json_path.write_text(json.dumps({
        "baseline": asdict(baseline),
        "runs": [asdict(r) for r in runs],
    }, indent=2), encoding="utf-8")
    write_markdown(baseline, runs, md_path)

    print()
    print("=" * 78)
    print(f"  Done. Reports:")
    print(f"    {md_path.relative_to(ROOT)}")
    print(f"    {json_path.relative_to(ROOT)}")
    print("=" * 78)
    print(md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
