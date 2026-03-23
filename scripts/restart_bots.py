"""
restart_bots.py — Kill old bots, send Telegram notification, restart acc1+acc2.
"""
import os, sys, time, pathlib, json, subprocess, requests, signal

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Load .env ──────────────────────────────────────────────────────────────
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ── Telegram helpers ────────────────────────────────────────────────────────
def send_telegram(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"  Telegram error: {e}")
        return False

def notify_all(text: str):
    for suffix in ["", "_ACC1", "_ACC2"]:
        token = os.environ.get(f"TELEGRAM_BOT_TOKEN{suffix}", "")
        chat  = os.environ.get(f"TELEGRAM_CHAT_ID{suffix}", "")
        if token and chat:
            ok = send_telegram(token, chat, text)
            print(f"  Telegram{suffix}: {'OK' if ok else 'FAIL'}")

# ── Step 1: Old bots already killed by PowerShell before this script ───────
print("\n[1/4] Old bots were killed by PowerShell. Waiting 2s...")
time.sleep(2)

# ── Step 2: Notify restart ──────────────────────────────────────────────────
print("\n[2/4] Sending Telegram restart notification...")
msg = (
    "🔄 <b>Bot Restarted</b> — 23/03/2026\n\n"
    "✅ <b>Bugs fixed:</b>\n"
    "  • <code>blackout=8878</code> — news_features.py datetime fix\n"
    "  • Unicode log crash (Windows cp1252)\n"
    "  • M5 execution enabled (was M15)\n"
    "  • <code>strong_volatility_threshold: 0.016</code> (ATR high)\n"
    "  • <code>volatile_min_confidence: 0.55</code> (lowered)\n"
    "  • <code>retrain_on_startup: true</code>\n\n"
    "📊 <b>Backtest Mar 9–23 (new model):</b>\n"
    "  WinRate: 74.2% | PF: 2.30 | +308% ROI\n"
    "  (89 trades, 501 days training data)\n\n"
    "⚡ Bots restarting with new configs..."
)
notify_all(msg)

# ── Step 3: Start ACC1 ──────────────────────────────────────────────────────
print("\n[3/4] Starting ACC1 (live_ict_wyckoff.yaml)...")
python_exe = sys.executable
log_acc1_out = str(ROOT / "outputs/live_out_acc1.txt")
log_acc1_err = str(ROOT / "outputs/live_err_acc1.txt")
proc1 = subprocess.Popen(
    [python_exe, "scripts/live_runner.py", "live", "--config", "configs/live_ict_wyckoff.yaml"],
    cwd=str(ROOT),
    stdout=open(log_acc1_out, "a", encoding="utf-8"),
    stderr=open(log_acc1_err, "a", encoding="utf-8"),
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
)
print(f"  ACC1 PID: {proc1.pid}")
time.sleep(3)
if proc1.poll() is None:
    print("  ACC1 running OK")
else:
    print(f"  ACC1 CRASHED (rc={proc1.returncode})")

# ── Step 4: Start ACC2 ──────────────────────────────────────────────────────
print("\n[4/4] Starting ACC2 (live_acc2.yaml)...")
log_acc2_out = str(ROOT / "outputs/live_out_acc2.txt")
log_acc2_err = str(ROOT / "outputs/live_err_acc2.txt")
proc2 = subprocess.Popen(
    [python_exe, "scripts/live_runner.py", "live", "--config", "configs/live_acc2.yaml"],
    cwd=str(ROOT),
    stdout=open(log_acc2_out, "a", encoding="utf-8"),
    stderr=open(log_acc2_err, "a", encoding="utf-8"),
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
)
print(f"  ACC2 PID: {proc2.pid}")
time.sleep(3)
if proc2.poll() is None:
    print("  ACC2 running OK")
else:
    print(f"  ACC2 CRASHED (rc={proc2.returncode})")

# Save PIDs
pids_file = ROOT / "outputs/pids.txt"
pids_file.write_text(f"{proc1.pid} ACC1\n{proc2.pid} ACC2\n", encoding="utf-8")

print(f"\n{'='*50}")
print(f"  ACC1 PID {proc1.pid} — configs/live_ict_wyckoff.yaml")
print(f"  ACC2 PID {proc2.pid} — configs/live_acc2.yaml")
print(f"  Logs: outputs/live_out_acc1.txt | outputs/live_out_acc2.txt")
print(f"{'='*50}")
