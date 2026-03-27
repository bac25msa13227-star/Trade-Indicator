import subprocess, sys
r = subprocess.run(['docker','logs','xauusd-cloudflared'], capture_output=True, text=True)
all_out = r.stdout + r.stderr
lines = [l.strip() for l in all_out.splitlines() if 'trycloudflare' in l.lower()]
if lines:
    for l in lines[-5:]:
        print(l)
else:
    print("NOT FOUND")
    print("--- last 5 lines ---")
    for l in all_out.splitlines()[-5:]:
        print(l)
