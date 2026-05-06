#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# XAUUSD AI — ONE-COMMAND PRODUCTION DEPLOYMENT
# Paper mode (ACC1) + Live demo (ACC2) with $15 profit filter
# ═══════════════════════════════════════════════════════════════════════════

cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"

# 1. Start Docker Desktop (if not running)
echo "🚀 Starting Docker Desktop..."
open -a Docker && sleep 30

# 2. Deploy both accounts
echo ""
echo "📦 Deploying dual-mode trading system..."
./deploy_dual_mode.sh

# 3. Wait for startup
echo ""
echo "⏳ Waiting 60 seconds for containers to initialize..."
sleep 60

# 4. Verify deployment
echo ""
echo "📊 Checking deployment status..."
./monitor_dual_mode.sh

# 5. Show recent logs
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📋 Recent Logs (ACC1 Paper):"
echo "════════════════════════════════════════════════════════════════════════════"
docker logs live-acc1 --tail 20 | grep -E "(Starting|Profit filter|BLOCKED|signal)" || echo "No matching logs yet"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📋 Recent Logs (ACC2 Demo):"
echo "════════════════════════════════════════════════════════════════════════════"
docker logs live --tail 20 | grep -E "(Starting|Profit filter|BLOCKED|signal|order)" || echo "No matching logs yet"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "✅ Deployment Complete!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Next Steps:"
echo ""
echo "  1. Monitor daily (30 sec): ./monitor_dual_mode.sh"
echo "  2. View live logs: docker logs -f live-acc1  # or 'live'"
echo "  3. Check after 7 days for live deployment decision"
echo ""
echo "📈 Expected Results (from WF validation):"
echo "  • Skip rate: 60-82% (both accounts)"
echo "  • Win rate: 40%+"
echo "  • ACC2 P&L: Positive trend"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
