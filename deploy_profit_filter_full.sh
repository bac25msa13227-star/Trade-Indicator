#!/bin/bash
# ============================================================================
# PROFIT FILTER DEPLOYMENT - FULL SCRIPT (Paper + Demo)
# ============================================================================
# Version: 1.0
# Date: May 6, 2026
# Purpose: Deploy profit filter to production with paper mode and demo account
# ============================================================================

set -e  # Exit on error

echo "🚀 PROFIT FILTER DEPLOYMENT"
echo "========================================================================"
echo ""

# Configuration
REPO_PATH=~/Trade-Indicator  # Change to your path
FEATURE_BRANCH="feature/turnover-profit-filter"
PAPER_CONTAINER="live-acc1"
DEMO_CONTAINER="demo-acc1"

# ============================================================================
# PHASE 1: PAPER MODE DEPLOYMENT
# ============================================================================

echo "📋 PHASE 1: Paper Mode Deployment"
echo "------------------------------------------------------------------------"
echo ""

# Step 1: Pull latest code
echo "[1/5] Pulling latest code from GitHub..."
cd "$REPO_PATH" || { echo "❌ Error: Repository not found at $REPO_PATH"; exit 1; }

git fetch origin
git checkout "$FEATURE_BRANCH"
git pull origin "$FEATURE_BRANCH"

if [ $? -eq 0 ]; then
    echo "✅ Code updated successfully"
    git log --oneline -3
else
    echo "❌ Error: Failed to pull code"
    exit 1
fi
echo ""

# Step 2: Verify config
echo "[2/5] Verifying paper mode config..."
FILTER_ENABLED=$(grep "profit_filter_enabled:" configs/live_acc1.yaml | awk '{print $2}')
MIN_PROFIT=$(grep "min_expected_profit:" configs/live_acc1.yaml | awk '{print $2}')

echo "  Filter enabled: $FILTER_ENABLED"
echo "  Min profit: \$$MIN_PROFIT"

if [ "$FILTER_ENABLED" != "true" ]; then
    echo "⚠️  Warning: Filter not enabled in config!"
    echo "   Edit configs/live_acc1.yaml to enable profit_filter_enabled: true"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo "✅ Config verified"
echo ""

# Step 3: Check if container exists
echo "[3/5] Checking container status..."
if docker ps -a --format '{{.Names}}' | grep -q "^${PAPER_CONTAINER}$"; then
    echo "✅ Container $PAPER_CONTAINER exists"
    docker ps --filter "name=$PAPER_CONTAINER" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
    echo "❌ Error: Container $PAPER_CONTAINER not found"
    echo "   Run: docker compose up $PAPER_CONTAINER -d"
    exit 1
fi
echo ""

# Step 4: Restart container
echo "[4/5] Restarting paper mode container..."
docker compose restart "$PAPER_CONTAINER"
sleep 5
echo "✅ Container restarted"
echo ""

# Step 5: Verify deployment
echo "[5/5] Verifying paper mode deployment..."
echo ""
echo "Checking logs for profit filter initialization..."
docker logs "$PAPER_CONTAINER" 2>&1 | grep -i "profit filter" | tail -5

if docker logs "$PAPER_CONTAINER" 2>&1 | grep -q "Profit filter initialized"; then
    echo ""
    echo "✅ Paper mode deployment SUCCESS!"
else
    echo ""
    echo "⚠️  Warning: Profit filter logs not found. Check manually:"
    echo "   docker logs $PAPER_CONTAINER | grep -i filter"
fi
echo ""

# ============================================================================
# PHASE 2: DEMO ACCOUNT SETUP (OPTIONAL)
# ============================================================================

echo ""
echo "📋 PHASE 2: Demo Account Setup (Optional)"
echo "------------------------------------------------------------------------"
echo ""
echo "Do you want to setup demo account for additional testing?"
echo "  - Paper mode tests filter logic (shadow execution)"
echo "  - Demo tests real order execution on demo broker account"
echo ""
read -p "Setup demo account? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting demo account setup..."
    echo ""
    
    # Step 1: Check if demo config exists
    if [ ! -f "configs/demo_acc1.yaml" ]; then
        echo "[1/6] Creating demo config from template..."
        cp configs/live_acc1.yaml configs/demo_acc1.yaml
        echo "✅ Config created: configs/demo_acc1.yaml"
        echo ""
        echo "⚠️  IMPORTANT: Edit configs/demo_acc1.yaml with your demo credentials:"
        echo "   1. Change execution.mode from 'paper' to 'live'"
        echo "   2. Update mt5_login with demo account number"
        echo "   3. Update mt5_password with demo password"
        echo "   4. Update mt5_server with demo server (e.g., 'MetaQuotes-Demo')"
        echo "   5. Keep profit_filter_enabled: true"
        echo ""
        read -p "Press Enter after editing configs/demo_acc1.yaml..." -r
    else
        echo "[1/6] Demo config already exists: configs/demo_acc1.yaml"
    fi
    echo ""
    
    # Step 2: Verify demo config
    echo "[2/6] Verifying demo config..."
    DEMO_MODE=$(grep "mode:" configs/demo_acc1.yaml | head -1 | awk '{print $2}')
    DEMO_FILTER=$(grep "profit_filter_enabled:" configs/demo_acc1.yaml | awk '{print $2}')
    
    echo "  Execution mode: $DEMO_MODE"
    echo "  Filter enabled: $DEMO_FILTER"
    
    if [ "$DEMO_MODE" != "live" ]; then
        echo "⚠️  Warning: Demo mode should be 'live' for real order execution"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    echo "✅ Demo config verified"
    echo ""
    
    # Step 3: Add demo service to docker-compose.yml
    echo "[3/6] Checking docker-compose.yml for demo service..."
    if grep -q "demo-acc1:" docker-compose.yml; then
        echo "✅ Demo service already exists in docker-compose.yml"
    else
        echo "⚠️  Demo service not found in docker-compose.yml"
        echo ""
        echo "Add this to docker-compose.yml:"
        echo ""
        cat << 'EOF'
  demo-acc1:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: demo-acc1
    volumes:
      - ./configs/demo_acc1.yaml:/app/configs/live_acc1.yaml:ro
      - ./outputs:/app/outputs
      - ./data:/app/data:ro
    environment:
      - CONFIG_FILE=configs/live_acc1.yaml
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    networks:
      - trading-network
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
EOF
        echo ""
        read -p "Press Enter after adding demo service to docker-compose.yml..." -r
    fi
    echo ""
    
    # Step 4: Start demo container
    echo "[4/6] Starting demo account container..."
    docker compose up demo-acc1 -d --no-build
    
    if [ $? -eq 0 ]; then
        echo "✅ Demo container started"
        sleep 5
    else
        echo "❌ Error: Failed to start demo container"
        exit 1
    fi
    echo ""
    
    # Step 5: Check demo container status
    echo "[5/6] Checking demo container status..."
    docker ps --filter "name=$DEMO_CONTAINER" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    
    # Step 6: Verify demo deployment
    echo "[6/6] Verifying demo account deployment..."
    echo ""
    echo "Checking demo logs..."
    docker logs "$DEMO_CONTAINER" 2>&1 | tail -20
    
    if docker logs "$DEMO_CONTAINER" 2>&1 | grep -q "Profit filter initialized"; then
        echo ""
        echo "✅ Demo account deployment SUCCESS!"
    else
        echo ""
        echo "⚠️  Warning: Check demo logs manually:"
        echo "   docker logs $DEMO_CONTAINER"
    fi
    echo ""
    
else
    echo "⏭️  Skipping demo account setup"
    echo "   You can setup later by running this script again"
fi

# ============================================================================
# PHASE 3: MONITORING SETUP
# ============================================================================

echo ""
echo "📊 PHASE 3: Monitoring Commands"
echo "========================================================================"
echo ""

echo "PAPER MODE MONITORING:"
echo "------------------------------------------------------------------------"
echo ""
echo "1. Watch real-time logs:"
echo "   docker logs -f $PAPER_CONTAINER | grep -E 'PROFIT_FILTER|should_trade'"
echo ""
echo "2. Check skip rate:"
cat << 'EOF'
   TOTAL=$(docker logs live-acc1 | grep "should_trade" | wc -l | tr -d ' ')
   SKIPPED=$(docker logs live-acc1 | grep "PROFIT_FILTER BLOCKED" | wc -l | tr -d ' ')
   if [ "$TOTAL" -gt 0 ]; then
       echo "Skip rate: $((SKIPPED * 100 / TOTAL))%"
   fi
EOF
echo ""
echo "3. Check paper trade logs:"
echo "   docker exec $PAPER_CONTAINER cat /app/outputs/paper_trade_log.jsonl | tail -20"
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "DEMO ACCOUNT MONITORING:"
    echo "------------------------------------------------------------------------"
    echo ""
    echo "1. Watch demo real-time logs:"
    echo "   docker logs -f $DEMO_CONTAINER | grep -E 'PROFIT_FILTER|Order'"
    echo ""
    echo "2. Check demo trades:"
    echo "   docker exec $DEMO_CONTAINER cat /app/outputs/live_closed_trades_acc1.csv | tail -20"
    echo ""
    echo "3. Check demo skip rate:"
cat << 'EOF'
   TOTAL=$(docker logs demo-acc1 | grep "should_trade" | wc -l | tr -d ' ')
   SKIPPED=$(docker logs demo-acc1 | grep "PROFIT_FILTER BLOCKED" | wc -l | tr -d ' ')
   if [ "$TOTAL" -gt 0 ]; then
       echo "Skip rate: $((SKIPPED * 100 / TOTAL))%"
   fi
EOF
    echo ""
fi

# ============================================================================
# PHASE 4: MONITORING FIRST SIGNALS
# ============================================================================

echo ""
echo "🔍 PHASE 4: Monitoring First Signals (60 seconds)"
echo "========================================================================"
echo ""
echo "Watching for signals from paper mode..."
echo ""

# Monitor for 60 seconds
timeout 60 docker logs -f "$PAPER_CONTAINER" 2>&1 | grep -E "PROFIT_FILTER|should_trade" &
MONITOR_PID=$!

sleep 60
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "⏱️  Initial monitoring complete"
echo ""

# ============================================================================
# PHASE 5: SUMMARY & NEXT STEPS
# ============================================================================

echo ""
echo "✅ DEPLOYMENT COMPLETE!"
echo "========================================================================"
echo ""

echo "📋 DEPLOYMENT SUMMARY:"
echo "------------------------------------------------------------------------"
echo "✅ Paper mode:        $PAPER_CONTAINER (running)"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "✅ Demo account:      $DEMO_CONTAINER (running)"
else
    echo "⏭️  Demo account:      Not configured"
fi
echo "✅ Profit filter:     Enabled (\$$MIN_PROFIT threshold)"
echo "✅ Git branch:        $FEATURE_BRANCH"
echo ""

echo "📊 MONITORING SCHEDULE (Next 7 Days):"
echo "------------------------------------------------------------------------"
echo ""
echo "Daily checklist (5 minutes/day):"
echo "  [ ] Check paper mode skip rate (target: 60-75%)"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  [ ] Check demo account skip rate (target: 60-75%)"
    echo "  [ ] Check demo P&L (target: positive trend)"
fi
echo "  [ ] Check paper mode logs for errors"
echo "  [ ] Verify bot is running (no crashes)"
echo ""

echo "🎯 DAY 7 REVIEW CRITERIA (May 13, 2026):"
echo "------------------------------------------------------------------------"
echo ""
echo "✅ Deploy to LIVE if:"
echo "  1. Paper skip rate: 60-75%"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  2. Demo skip rate: 60-75%"
    echo "  3. Demo P&L: Positive"
    echo "  4. Demo prediction accuracy: >60%"
else
    echo "  2. Paper P&L trend: Positive"
    echo "  3. No systematic bias detected"
fi
echo "  $(if [[ $REPLY =~ ^[Yy]$ ]]; then echo 5; else echo 4; fi). No crashes or errors"
echo ""

echo "⚠️  ADJUST threshold if:"
echo "  - Skip rate 40-60%: Lower to \$10 or raise to \$15"
echo "  - P&L breakeven: Test 3 more days"
echo ""

echo "❌ DISABLE filter if:"
echo "  - Skip rate <30%: Filter not working"
echo "  - P&L significantly negative: Filter harmful"
echo "  - Systematic bias: Skipping winners, keeping losers"
echo ""

echo "🔧 QUICK COMMANDS:"
echo "------------------------------------------------------------------------"
echo ""
echo "Disable filter (emergency):"
echo "  1. Edit configs/live_acc1.yaml: profit_filter_enabled: false"
echo "  2. docker compose restart $PAPER_CONTAINER"
echo ""
echo "Change threshold:"
echo "  1. Edit configs/live_acc1.yaml: min_expected_profit: 15.0"
echo "  2. docker compose restart $PAPER_CONTAINER (or wait 60s for hot-reload)"
echo ""
echo "Stop everything:"
echo "  docker compose stop $PAPER_CONTAINER"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  docker compose stop $DEMO_CONTAINER"
fi
echo ""

echo "📚 DOCUMENTATION:"
echo "------------------------------------------------------------------------"
echo "  - Deployment guide:    PAPER_MODE_DEPLOYMENT.md"
echo "  - Filter integration:  PROFIT_FILTER_INTEGRATION.md"
echo "  - WF analysis:         WF_PROFIT_FILTER_ANALYSIS.md"
echo "  - Spread explanation:  SPREAD_EXPLAINED_VN.md"
echo ""

echo "========================================================================"
echo "🎉 All done! Monitor daily and review results on May 13, 2026"
echo "========================================================================"
echo ""
