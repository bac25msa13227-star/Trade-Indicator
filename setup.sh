#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — Trade Indicator automated setup (macOS / Linux)
#
# Usage:
#   bash setup.sh --bot acc2        # M1 Scalp bot (ACC2) only
#   bash setup.sh --bot acc1        # M5 PF3v2 bot (ACC1) only
#   bash setup.sh --bot both        # both bots
#
# Prerequisites: Docker Desktop, git, git-lfs
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BOT="acc2"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bot) BOT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Trade Indicator — Automated Setup                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Check Docker ──────────────────────────────────────────────────────────
echo "[1/7] Checking Docker..."
docker info > /dev/null 2>&1 || err "Docker is not running. Start Docker Desktop first."
ok "Docker running"

# ── 2. git pull + lfs ────────────────────────────────────────────────────────
echo "[2/7] Pulling latest code..."
git pull 2>&1 || warn "git pull failed — continuing with current code"
git lfs pull 2>&1 || warn "git lfs pull failed — model files may be missing"
ok "Code up to date"

# ── 3. .env setup ────────────────────────────────────────────────────────────
echo "[3/7] Checking .env..."
if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
    warn ".env created from .env.example — EDIT it with your MT5 credentials before starting!"
    echo "      Required: MT5_LOGIN_ACC2, MT5_PASSWORD_ACC2, MT5_SERVER_ACC2"
    echo "      Required: TELEGRAM_BOT_TOKEN_ACC2, TELEGRAM_CHAT_ID_ACC2"
    read -p "      Press ENTER after editing .env to continue..." _
  else
    err ".env.example not found. Cannot create .env."
  fi
else
  ok ".env exists"
fi

# ── 4. outputs/ directory ────────────────────────────────────────────────────
echo "[4/7] Checking outputs/ directory..."
mkdir -p outputs logs
ok "outputs/ and logs/ directories ready"

# ── 5. Check model .pkl ──────────────────────────────────────────────────────
echo "[5/7] Checking model artifacts..."
check_pkl() {
  local f="$1"
  if [[ -f "$f" ]]; then
    # Check it's a real pkl (not an LFS pointer stub)
    local size
    size=$(wc -c < "$f")
    if [[ $size -gt 10000 ]]; then
      ok "$f ($(du -sh "$f" | cut -f1))"
    else
      warn "$f looks like an LFS pointer stub — run: git lfs pull"
    fi
  else
    warn "$f not found — will auto-train on first start (~4 min)"
  fi
}

if [[ "$BOT" == "acc2" || "$BOT" == "both" ]]; then
  check_pkl "outputs/acc2_scalp_m1_model.pkl"
fi
if [[ "$BOT" == "acc1" || "$BOT" == "both" ]]; then
  check_pkl "outputs/acc1_expand_net127313_dd3215_model.pkl"
fi

# ── 6. Build Docker image ────────────────────────────────────────────────────
echo "[6/7] Building Docker image(s)..."
case "$BOT" in
  acc2) docker compose build live-scalp-acc2 ;;
  acc1) docker compose build live-acc1 ;;
  both) docker compose build live-scalp-acc2 live-acc1 ;;
esac
ok "Docker image built"

# ── 7. Start bot ─────────────────────────────────────────────────────────────
echo "[7/7] Starting bot(s)..."
case "$BOT" in
  acc2) docker compose up -d live-scalp-acc2 ;;
  acc1) docker compose up -d live-acc1 ;;
  both) docker compose up -d live-scalp-acc2 live-acc1 ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
ok " Setup complete!"
echo "║"
echo "║  Logs:    docker compose logs live-scalp-acc2 -f"
echo "║  Status:  cat outputs/live_status_acc2.json"
echo "║  Stop:    docker compose stop live-scalp-acc2"
echo "║"
echo "║  ⚠️  Make sure MT5 Bridge is running on host port 5601"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
