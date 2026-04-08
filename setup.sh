#!/usr/bin/env bash
# =============================================================================
#  setup.sh — Trade Indicator: bootstrap mới hoàn toàn từ git clone
#  Chạy: bash setup.sh [--bot acc2|acc1|both] [--data-src user@host]
# =============================================================================
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✅  $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️   $*${NC}"; }
info() { echo -e "${CYAN}➤  $*${NC}"; }
die()  { echo -e "${RED}❌  $*${NC}"; exit 1; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ── Parse args ────────────────────────────────────────────────────────────────
BOT_TARGET="acc2"          # default: only run M1 scalp bot
DATA_SRC=""                # e.g. user@192.168.1.10

while [[ $# -gt 0 ]]; do
  case $1 in
    --bot)     BOT_TARGET="$2"; shift 2 ;;
    --data-src) DATA_SRC="$2"; shift 2 ;;
    *) warn "Unknown arg: $1"; shift ;;
  esac
done

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}   Trade Indicator — Setup Bootstrap                       ${NC}"
echo -e "${BOLD}   Bot target: ${CYAN}${BOT_TARGET}${NC}${BOLD}  |  Dir: ${CYAN}${REPO_DIR}${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Prerequisites
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 1 — Checking prerequisites"

command -v docker  >/dev/null 2>&1 || die "Docker not found. Install: https://docs.docker.com/get-docker/"
command -v git     >/dev/null 2>&1 || die "git not found."

DOCKER_COMPOSE_CMD=""
if docker compose version >/dev/null 2>&1; then
  DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  DOCKER_COMPOSE_CMD="docker-compose"
else
  die "docker compose / docker-compose not found."
fi

ok "Docker: $(docker --version | awk '{print $3}' | tr -d ',')"
ok "Compose: $($DOCKER_COMPOSE_CMD version --short 2>/dev/null || echo 'v1')"
ok "Git: $(git --version | awk '{print $3}')"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Git pull
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 2 — Git pull"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
info "Branch: ${BRANCH}"
git pull
COMMIT=$(git rev-parse --short HEAD)
ok "HEAD: ${COMMIT}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — .env file
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 3 — Environment file"

if [[ -f .env ]]; then
  ok ".env already exists — skipping"
else
  if [[ -f .env.example ]]; then
    cp .env.example .env
    ok ".env created from .env.example"
    warn "Kiểm tra .env và sửa nếu cần: nano .env"
  else
    die ".env.example not found — cannot create .env"
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Create output directories
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 4 — Directories"

mkdir -p outputs src/xauusd_ai/real_data
ok "outputs/ và real_data/ ready"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CSV market data
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 5 — Market data CSV"

# Files required per bot
NEED_M1_CSV=false
NEED_M5_CSV=false

[[ "$BOT_TARGET" == "acc2" || "$BOT_TARGET" == "both" ]] && NEED_M1_CSV=true
[[ "$BOT_TARGET" == "acc1" || "$BOT_TARGET" == "both" ]] && NEED_M5_CSV=true

COMMON_FILES=("XAUUSDm_D1.csv" "XAUUSDm_H4.csv" "XAUUSDm_H1.csv")
REQUIRED_FILES=("${COMMON_FILES[@]}")
$NEED_M1_CSV && REQUIRED_FILES+=("XAUUSDm_M1.csv")
$NEED_M5_CSV && REQUIRED_FILES+=("XAUUSDm_M5.csv")

CSV_MISSING=()
for f in "${REQUIRED_FILES[@]}"; do
  [[ -f "src/xauusd_ai/real_data/$f" ]] || CSV_MISSING+=("$f")
done

if [[ ${#CSV_MISSING[@]} -eq 0 ]]; then
  ok "Tất cả CSV data đã có"
else
  warn "Thiếu ${#CSV_MISSING[@]} file(s) CSV: ${CSV_MISSING[*]}"

  if [[ -n "$DATA_SRC" ]]; then
    info "Đang copy từ ${DATA_SRC} ..."
    for f in "${CSV_MISSING[@]}"; do
      info "  rsync: ${f}"
      rsync -az --progress \
        "${DATA_SRC}:\"$(ssh "$DATA_SRC" 'find ~/Documents -name Trade-Indicator -type d 2>/dev/null | head -1' 2>/dev/null)/src/xauusd_ai/real_data/${f}\"" \
        "src/xauusd_ai/real_data/" 2>/dev/null || \
      rsync -az --progress \
        "${DATA_SRC}:/root/Trade-Indicator/src/xauusd_ai/real_data/${f}" \
        "src/xauusd_ai/real_data/" 2>/dev/null || \
      warn "  ❌ Không copy được ${f} — copy thủ công sau"
    done
  else
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  Copy dữ liệu từ máy Mac (chạy trên Mac):${NC}"
    echo ""
    SOURCE_MAC_DIR="/Users/dodoannang/Documents/Thac si MSE/Trade Indicator"
    TARGET_DIR="$(pwd)/src/xauusd_ai/real_data/"
    for f in "${CSV_MISSING[@]}"; do
      echo -e "  ${CYAN}scp \"${SOURCE_MAC_DIR}/src/xauusd_ai/real_data/${f}\" \\${NC}"
      echo -e "      ${CYAN}user@TARGET_IP:${TARGET_DIR}${NC}"
    done
    echo ""
    echo -e "  Hoặc dùng rsync (nhanh hơn, có resume):"
    echo -e "  ${CYAN}bash setup.sh --bot ${BOT_TARGET} --data-src user@MAC_IP${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    read -r -p "  Tiếp tục mà không có CSV? (bot sẽ fail khi chạy) [y/N]: " CONT
    [[ "$CONT" =~ ^[Yy]$ ]] || { warn "Setup dừng — copy CSV rồi chạy lại"; exit 0; }
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Model artifacts (đã có trong git)
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 6 — Model artifacts"

MODELS_OK=true
for f in "outputs/acc2_scalp_m1_model.pkl" \
         "outputs/acc2_scalp_m1_scaler.pkl" \
         "outputs/acc2_scalp_m1_model_meta.json"; do
  if [[ -f "$f" ]]; then
    ok "  $f"
  else
    warn "  Thiếu: $f (sẽ tự train khi start bot)"
    MODELS_OK=false
  fi
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Build Docker image
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 7 — Build Docker image"

case "$BOT_TARGET" in
  acc2) BUILD_SERVICES="live-scalp-acc2" ;;
  acc1) BUILD_SERVICES="live-acc1" ;;
  both) BUILD_SERVICES="live-scalp-acc2 live-acc1" ;;
  *) die "Unknown --bot value: $BOT_TARGET (dùng: acc2 | acc1 | both)" ;;
esac

$DOCKER_COMPOSE_CMD build $BUILD_SERVICES
ok "Build xong"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Start bots
# ─────────────────────────────────────────────────────────────────────────────
info "STEP 8 — Start bots"

$DOCKER_COMPOSE_CMD up -d $BUILD_SERVICES
ok "Containers started"

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✅  SETUP HOÀN TẤT — commit: ${COMMIT}${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Commands hữu ích:"
echo ""
echo -e "  ${CYAN}# Xem live logs${NC}"
echo -e "  $DOCKER_COMPOSE_CMD logs live-scalp-acc2 -f"
echo ""
echo -e "  ${CYAN}# Xem trạng thái bot (cập nhật mỗi 60s)${NC}"
echo -e "  watch -n 5 cat outputs/live_status_acc2.json"
echo ""
echo -e "  ${CYAN}# Dừng bot${NC}"
echo -e "  $DOCKER_COMPOSE_CMD stop live-scalp-acc2"
echo ""
echo -e "  ${CYAN}# Train lại model với data mới${NC}"
echo -e "  $DOCKER_COMPOSE_CMD run --rm live-scalp-acc2 bash -c \\"
echo -e "    'rm -f outputs/acc2_scalp_m1_model.pkl && python scripts/acc2_scalp_m1_save_model.py'"
echo ""
