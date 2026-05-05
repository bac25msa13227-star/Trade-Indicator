"""Launch compound + high-risk WF runs as background subprocesses."""
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
WF = os.path.join(ROOT, "scripts", "walkforward_ict_wyckoff.py")
CFG = os.path.join(ROOT, "configs", "acc1_v14pp_profit.yaml")
OUT = os.path.join(ROOT, "outputs")

RUNS = [
    ("compound_r5", [PYTHON, WF, CFG,
        "--no-rr-sweep", "--cache", "--test-start", "2025-01-01",
        "--test-bars", "6000", "--step-bars", "6000",
        "--combo133", "--risk-pct", "0.05"]),
    ("risk8pct_nocompound", [PYTHON, WF, CFG,
        "--no-rr-sweep", "--cache", "--test-start", "2025-01-01",
        "--test-bars", "6000", "--step-bars", "6000",
        "--combo133", "--risk-pct", "0.08", "--no-compound"]),
    ("risk10pct_nocompound", [PYTHON, WF, CFG,
        "--no-rr-sweep", "--cache", "--test-start", "2025-01-01",
        "--test-bars", "6000", "--step-bars", "6000",
        "--combo133", "--risk-pct", "0.10", "--no-compound"]),
    ("risk10pct_compound", [PYTHON, WF, CFG,
        "--no-rr-sweep", "--cache", "--test-start", "2025-01-01",
        "--test-bars", "6000", "--step-bars", "6000",
        "--combo133", "--risk-pct", "0.10"]),
]

procs = []
for name, cmd in RUNS:
    log = os.path.join(OUT, f"wf_{name}.txt")
    f = open(log, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    procs.append((name, p, f, log))
    print(f"Started {name} PID={p.pid} -> {log}")

print(f"\nAll {len(procs)} runs launched. Waiting...")
for name, p, f, log in procs:
    ret = p.wait()
    f.close()
    print(f"  {name}: exit={ret}")

print("\n=== ALL DONE ===")
