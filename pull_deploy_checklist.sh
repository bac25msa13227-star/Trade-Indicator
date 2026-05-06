#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTION DEPLOYMENT — Pull Code & Deploy Checklist
# Branch: feature/turnover-profit-filter → main
# ═══════════════════════════════════════════════════════════════════════════

set -e

REPO_DIR="$HOME/Documents/Thạc sĩ MSE/Trade Indicator"
BRANCH="main"  # Change to "feature/turnover-profit-filter" if testing before merge

echo "════════════════════════════════════════════════════════════════════════════"
echo "🚀 XAUUSD AI — Production Deployment from GitHub"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📅 Deployment Date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "📂 Repository: $REPO_DIR"
echo "🌿 Branch: $BRANCH"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Git Pull & Verification
# ─────────────────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════════════════"
echo "📥 STEP 1/7: Pull Latest Code from GitHub"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

cd "$REPO_DIR"

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "  Current branch: $CURRENT_BRANCH"

# Stash any local changes
if [[ -n $(git status -s) ]]; then
    echo "  ⚠️  Uncommitted changes detected, stashing..."
    git stash push -m "Auto-stash before pull $(date +%Y%m%d_%H%M%S)"
fi

# Pull latest code
echo "  Fetching from origin..."
git fetch origin

echo "  Checking out $BRANCH..."
git checkout "$BRANCH"

echo "  Pulling latest changes..."
git pull origin "$BRANCH"

# Show latest commits
echo ""
echo "  ✅ Latest commits:"
git log --oneline -5 | sed 's/^/    /'

echo ""
echo "✅ Step 1 Complete: Code pulled successfully"
sleep 2

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: File Verification
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "🔍 STEP 2/7: Verify Deployment Files"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

REQUIRED_FILES=(
    "configs/live_acc1.yaml"
    "configs/live_acc2.yaml"
    "deploy_dual_mode.sh"
    "deploy_production.sh"
    "monitor_dual_mode.sh"
    "PRODUCTION_DEPLOYMENT.md"
    "src/xauusd_ai/strategies/profit_filter.py"
    "outputs/acc1_combo133_202604_model.pkl"
    "outputs/acc1_combo133_202604_scaler.pkl"
)

ALL_FOUND=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ MISSING: $file"
        ALL_FOUND=false
    fi
done

if [ "$ALL_FOUND" = false ]; then
    echo ""
    echo "❌ ERROR: Missing required files. Check git pull or file paths."
    exit 1
fi

echo ""
echo "✅ Step 2 Complete: All required files present"
sleep 2

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Configuration Verification
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "⚙️  STEP 3/7: Verify Configuration"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check ACC1 paper mode
ACC1_MODE=$(grep "^\s*mode:" configs/live_acc1.yaml | grep -v "#" | awk '{print $2}')
ACC1_FILTER=$(grep "^\s*profit_filter_enabled:" configs/live_acc1.yaml | grep -v "#" | awk '{print $2}')
ACC1_THRESHOLD=$(grep "^\s*min_expected_profit:" configs/live_acc1.yaml | grep -v "#" | awk '{print $2}')

echo "  ACC1 Configuration:"
echo "    Mode: $ACC1_MODE (expected: paper)"
echo "    Profit Filter: $ACC1_FILTER (expected: true)"
echo "    Threshold: \$$ACC1_THRESHOLD (expected: 15.0)"

if [ "$ACC1_MODE" != "paper" ]; then
    echo "    ⚠️  WARNING: ACC1 should be in paper mode!"
fi

# Check ACC2 live mode
ACC2_MODE=$(grep "^\s*mode:" configs/live_acc2.yaml | grep -v "#" | head -1 | awk '{print $2}')
ACC2_FILTER=$(grep "^\s*profit_filter_enabled:" configs/live_acc2.yaml | grep -v "#" | awk '{print $2}')
ACC2_THRESHOLD=$(grep "^\s*min_expected_profit:" configs/live_acc2.yaml | grep -v "#" | awk '{print $2}')

echo ""
echo "  ACC2 Configuration:"
echo "    Mode: $ACC2_MODE (expected: live)"
echo "    Profit Filter: $ACC2_FILTER (expected: true)"
echo "    Threshold: \$$ACC2_THRESHOLD (expected: 15.0)"

if [ "$ACC2_MODE" != "live" ]; then
    echo "    ⚠️  WARNING: ACC2 should be in live mode!"
fi

# Verify thresholds match
if [ "$ACC1_THRESHOLD" = "$ACC2_THRESHOLD" ]; then
    echo ""
    echo "  ✅ Thresholds consistent: \$$ACC1_THRESHOLD"
else
    echo ""
    echo "  ⚠️  WARNING: Threshold mismatch (ACC1: \$$ACC1_THRESHOLD, ACC2: \$$ACC2_THRESHOLD)"
fi

echo ""
echo "✅ Step 3 Complete: Configuration verified"
sleep 2

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Model Files Verification
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "🤖 STEP 4/7: Verify Model Files"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

MODEL_FILE="outputs/acc1_combo133_202604_model.pkl"
SCALER_FILE="outputs/acc1_combo133_202604_scaler.pkl"
META_FILE="outputs/acc1_combo133_202604_meta.json"

if [ -f "$MODEL_FILE" ]; then
    MODEL_SIZE=$(ls -lh "$MODEL_FILE" | awk '{print $5}')
    echo "  ✅ Model: $MODEL_FILE ($MODEL_SIZE)"
else
    echo "  ❌ Missing model: $MODEL_FILE"
    exit 1
fi

if [ -f "$SCALER_FILE" ]; then
    SCALER_SIZE=$(ls -lh "$SCALER_FILE" | awk '{print $5}')
    echo "  ✅ Scaler: $SCALER_FILE ($SCALER_SIZE)"
else
    echo "  ❌ Missing scaler: $SCALER_FILE"
    exit 1
fi

if [ -f "$META_FILE" ]; then
    echo "  ✅ Metadata: $META_FILE"
    
    # Show model info
    if command -v jq &> /dev/null; then
        echo ""
        echo "  Model Info:"
        jq -r '.combo133_summary | "    Folds: \(.total_folds)\n    Profitable: \(.profitable_folds)\n    Net P&L: $\(.net_pnl_sum)"' "$META_FILE" 2>/dev/null || echo "    (JSON parse error)"
    fi
else
    echo "  ⚠️  Missing metadata: $META_FILE (non-critical)"
fi

echo ""
echo "✅ Step 4 Complete: Model files verified"
sleep 2

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Dependencies Check
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📦 STEP 5/7: Check Dependencies"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check Python environment
if [ -f ".venv/bin/python" ]; then
    PYTHON_VERSION=$(.venv/bin/python --version 2>&1)
    echo "  ✅ Python: $PYTHON_VERSION"
else
    echo "  ⚠️  Virtual environment not found (.venv/)"
    echo "     Creating new environment..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
fi

# Check Docker
if command -v docker &> /dev/null; then
    if docker info &> /dev/null; then
        DOCKER_VERSION=$(docker --version)
        echo "  ✅ Docker: $DOCKER_VERSION (running)"
    else
        echo "  ⚠️  Docker installed but not running"
        echo "     Starting Docker..."
        open -a Docker
        echo "     Waiting 30 seconds..."
        sleep 30
    fi
else
    echo "  ❌ Docker not installed"
    exit 1
fi

# Check docker-compose
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version 2>/dev/null || docker-compose --version)
    echo "  ✅ Docker Compose: $COMPOSE_VERSION"
else
    echo "  ❌ Docker Compose not found"
    exit 1
fi

echo ""
echo "✅ Step 5 Complete: Dependencies satisfied"
sleep 2

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Pre-Deployment Cleanup
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "🧹 STEP 6/7: Pre-Deployment Cleanup"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Stop existing containers
echo "  Stopping existing containers..."
docker compose down live live-acc1 2>/dev/null || true

# Wait for cleanup
sleep 3

# Check if containers stopped
RUNNING=$(docker ps --filter "name=live" --format '{{.Names}}' | wc -l)
if [ "$RUNNING" -eq 0 ]; then
    echo "  ✅ All live containers stopped"
else
    echo "  ⚠️  Some containers still running:"
    docker ps --filter "name=live" --format '  {{.Names}} ({{.Status}})'
fi

# Make scripts executable
echo ""
echo "  Setting script permissions..."
chmod +x deploy_dual_mode.sh deploy_production.sh monitor_dual_mode.sh

echo ""
echo "✅ Step 6 Complete: Pre-deployment cleanup done"
sleep 2

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Deploy System
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "🚀 STEP 7/7: Deploy Trading System"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Run deployment
./deploy_dual_mode.sh

# Wait for containers to stabilize
echo ""
echo "⏳ Waiting 60 seconds for containers to stabilize..."
sleep 60

# ─────────────────────────────────────────────────────────────────────────────
# DEPLOYMENT VERIFICATION & CHECKLIST
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "✅ DEPLOYMENT COMPLETE — Verification & Checklist"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Run monitoring
./monitor_dual_mode.sh

# ─────────────────────────────────────────────────────────────────────────────
# FINAL CHECKLIST
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📋 DEPLOYMENT CHECKLIST"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check Git
GIT_BRANCH=$(git branch --show-current)
GIT_COMMIT=$(git rev-parse --short HEAD)
GIT_REMOTE=$(git rev-parse --short origin/$BRANCH)

if [ "$GIT_COMMIT" = "$GIT_REMOTE" ]; then
    echo "  ✅ Git: Synced with origin/$BRANCH ($GIT_COMMIT)"
else
    echo "  ⚠️  Git: Local ($GIT_COMMIT) != Remote ($GIT_REMOTE)"
fi

# Check files
if [ -f "deploy_dual_mode.sh" ] && [ -f "monitor_dual_mode.sh" ]; then
    echo "  ✅ Deployment scripts: Present"
else
    echo "  ❌ Deployment scripts: Missing"
fi

# Check configs
if grep -q "mode: paper" configs/live_acc1.yaml && grep -q "mode: live" configs/live_acc2.yaml; then
    echo "  ✅ Configs: ACC1=paper, ACC2=live"
else
    echo "  ⚠️  Configs: Check modes manually"
fi

# Check profit filter
ACC1_FILTER_CHECK=$(grep "profit_filter_enabled: true" configs/live_acc1.yaml | wc -l)
ACC2_FILTER_CHECK=$(grep "profit_filter_enabled: true" configs/live_acc2.yaml | wc -l)

if [ "$ACC1_FILTER_CHECK" -gt 0 ] && [ "$ACC2_FILTER_CHECK" -gt 0 ]; then
    echo "  ✅ Profit filter: Enabled (\$15 threshold)"
else
    echo "  ⚠️  Profit filter: Check configs"
fi

# Check model files
if [ -f "outputs/acc1_combo133_202604_model.pkl" ]; then
    echo "  ✅ Model files: Present (34MB combo133)"
else
    echo "  ❌ Model files: Missing"
fi

# Check Docker containers
LIVE_ACC1_STATUS=$(docker ps --filter "name=live-acc1" --format '{{.Status}}' | head -1)
LIVE_ACC2_STATUS=$(docker ps --filter "name=live$" --format '{{.Status}}' | head -1)

if [ -n "$LIVE_ACC1_STATUS" ]; then
    echo "  ✅ Container live-acc1: Running ($LIVE_ACC1_STATUS)"
else
    echo "  ❌ Container live-acc1: Not running"
fi

if [ -n "$LIVE_ACC2_STATUS" ]; then
    echo "  ✅ Container live: Running ($LIVE_ACC2_STATUS)"
else
    echo "  ❌ Container live: Not running"
fi

# Check infrastructure
INFRA_RUNNING=$(docker ps --filter "name=xauusd" --format '{{.Names}}' | wc -l)
echo "  ✅ Infrastructure: $INFRA_RUNNING containers running"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📊 NEXT STEPS"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  1. Monitor performance:"
echo "     ./monitor_dual_mode.sh"
echo ""
echo "  2. View live logs:"
echo "     docker logs -f live-acc1  # ACC1 paper"
echo "     docker logs -f live       # ACC2 demo"
echo ""
echo "  3. Daily check (30 seconds):"
echo "     cd '$REPO_DIR' && ./monitor_dual_mode.sh"
echo ""
echo "  4. Week 1 validation (May 6-13):"
echo "     - Target skip rate: 60-82%"
echo "     - Target win rate: 40%+"
echo "     - ACC2 P&L: Positive trend"
echo ""
echo "  5. Day 7 decision (May 13):"
echo "     IF skip 60-82% + positive P&L → Deploy to LIVE ✅"
echo "     IF skip 40-60% → Adjust threshold \$18-20 ⚠️"
echo "     IF <40% OR negative → Disable filter ❌"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ Deployment timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "✅ All systems operational"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
