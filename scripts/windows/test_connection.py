"""Test MT5 + Telegram connection trước khi live trade."""
import sys, os
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

# ── Telegram test ──────────────────────────────────────────────────────────────
import urllib.request, urllib.parse, json

token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

if token and chat_id:
    try:
        msg = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": "🤖 Bot XAUUSD-AI đã kết nối!\n✅ Chuẩn bị vào live trading...",
            "parse_mode": "HTML"
        })
        r = urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage?{msg}", timeout=10)
        result = json.loads(r.read())
        print(f"[Telegram] OK - message_id={result['result']['message_id']}")
    except Exception as e:
        print(f"[Telegram] FAIL: {e}")
else:
    print("[Telegram] SKIP - token/chat_id chưa set")

# ── MT5 test ───────────────────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    login = int(os.environ["MT5_LOGIN"])
    password = os.environ["MT5_PASSWORD"]
    server = os.environ["MT5_SERVER"]

    print(f"\n[MT5] Connecting: login={login}, server={server}...")
    if mt5.initialize(login=login, password=password, server=server):
        info = mt5.account_info()
        print(f"[MT5] OK!")
        print(f"  Name     : {info.name}")
        print(f"  Balance  : ${info.balance:,.2f}")
        print(f"  Equity   : ${info.equity:,.2f}")
        print(f"  Server   : {info.server}")
        print(f"  Currency : {info.currency}")
        print(f"  Leverage : 1:{info.leverage}")

        # Check XAUUSD symbol
        sym = mt5.symbol_info("XAUUSDm")
        if sym is None:
            sym = mt5.symbol_info("XAUUSD")
        if sym:
            print(f"  Symbol   : {sym.name} bid={sym.bid:.2f} ask={sym.ask:.2f}")
        else:
            print("  Symbol   : XAUUSD not found - check broker symbol name!")

        mt5.shutdown()
        print("\n✅ MT5 connection SUCCESS - Ready for live trading!")
    else:
        err = mt5.last_error()
        print(f"[MT5] FAIL: {err}")
except ImportError:
    print("[MT5] MetaTrader5 package not installed - chỉ chạy được trên Windows với MT5")
except Exception as e:
    print(f"[MT5] Error: {e}")
