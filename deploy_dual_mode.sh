#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# DUAL-MODE DEPLOYMENT SCRIPT
# ACC1: Paper mode (shadow execution, no real orders)
# ACC2: Live demo account (real orders on demo)
# Profit filter: $15 threshold on both (82.3% skip rate validated)
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════════════════════════════"
echo "🚀 XAUUSD AI — Dual-Mode Deployment"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Deployment Configuration:"
echo ""
echo "  ACC1 (live-acc1):"
echo "    • Mode: PAPER (shadow execution)"
echo "    • Purpose: Validation without real orders"
echo "    • MT5 Bridge: http://host.docker.internal:5600"
echo "    • Profit Filter: \$15 (82.3% skip rate)"
echo "    • Logs: outputs/paper_trade_signals_acc1.csv"
echo ""
echo "  ACC2 (live):"
echo "    • Mode: LIVE DEMO"
echo "    • Purpose: Real trading on demo account"
echo "    • MT5 Bridge: http://host.docker.internal:5601"
echo "    • Profit Filter: \$15 (82.3% skip rate)"
echo "    • Logs: outputs/live_closed_trades_acc2.csv"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check Docker running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    echo "   Run: open -a Docker"
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker compose down live live-acc1 2>/dev/null || true
sleep 2

# Build images if needed
echo ""
echo "🏗️  Building images (this may take a few minutes)..."
docker compose build api mlflow 2>&1 | tail -5

# Start infrastructure services
echo ""
echo "🔧 Starting infrastructure services..."
docker compose up -d postgres minio mlflow prometheus grafana
sleep 5

# Wait for postgres
echo ""
echo "⏳ Waiting for PostgreSQL..."
timeout 30 bash -c 'until docker exec xauusd-postgres pg_isready -U trader >/dev/null 2>&1; do sleep 1; done' || {
    echo "❌ PostgreSQL failed to start"
    exit 1
}
echo "✅ PostgreSQL ready"

# Start ACC1 (Paper mode)
echo ""
echo "📝 Starting ACC1 (Paper mode)..."
docker compose up -d live-acc1
sleep 3

# Start ACC2 (Live demo)
echo ""
echo "💰 Starting ACC2 (Live demo)..."
docker compose up -d live
sleep 3

# Check container status
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📊 Container Status:"
echo "════════════════════════════════════════════════════════════════════════════"
docker ps --filter "name=xauusd" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check logs
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📋 Recent Logs (last 10 lines each):"
echo "════════════════════════════════════════════════════════════════════════════"

echo ""
echo "ACC1 (Paper):"
docker logs live-acc1 --tail 10 2>&1 | grep -E "(Starting|Profit filter|Config loaded|ERROR|WARNING)" || echo "  (No matching logs yet)"

echo ""
echo "ACC2 (Live Demo):"
docker logs live --tail 10 2>&1 | grep -E "(Starting|Profit filter|Config loaded|ERROR|WARNING)" || echo "  (No matching logs yet)"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "✅ Deployment Complete!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Monitoring Commands:"
echo ""
echo "  # View live logs (ACC1 paper):"
echo "  docker logs -f live-acc1"
echo ""
echo "  # View live logs (ACC2 demo):"
echo "  docker logs -f live"
echo ""
echo "  # Check profit filter stats:"
echo "  ./monitor_dual_mode.sh"
echo ""
echo "  # Stop both accounts:"
echo "  docker compose down live live-acc1"
echo ""
echo "  # Restart specific account:"
echo "  docker compose restart live-acc1  # ACC1 paper"
echo "  docker compose restart live       # ACC2 demo"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
