"""Kill the running walk-forward process (finds by cmdline, avoids live bot)."""
import subprocess
import sys

result = subprocess.run(
    ["wmic", "process", "get", "processid,commandline", "/format:csv"],
    capture_output=True, text=True
)

killed = []
for line in result.stdout.splitlines():
    if "walkforward" in line.lower() or ("xauusd_ai.main" in line and "backtest_2025_2026_200usd" in line):
        parts = line.split(",", 2)
        if len(parts) >= 3:
            try:
                pid = int(parts[2].strip())
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
                killed.append(pid)
                print(f"Killed PID {pid}")
            except (ValueError, IndexError):
                pass

if not killed:
    print("No walkforward process found.")
else:
    print(f"Killed {len(killed)} process(es).")
