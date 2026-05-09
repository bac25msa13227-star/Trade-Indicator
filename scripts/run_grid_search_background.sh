#!/bin/bash
# Background runner for 24/7 automatic WF grid search
# Usage: ./run_grid_search_background.sh [strategy]
# Strategy: coarse (default), balanced, fine, full

STRATEGY="${1:-coarse}"
LOG_DIR="outputs/grid_search/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/grid_search_${STRATEGY}_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "========================================="
echo "🚀 STARTING 24/7 GRID SEARCH"
echo "========================================="
echo "Strategy: $STRATEGY"
echo "Log file: $LOG_FILE"
echo "Started: $(date)"
echo ""
echo "This will run in background with nohup."
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check status:"
echo "  python scripts/monitor_grid_search.py"
echo ""
echo "To stop:"
echo "  ps aux | grep auto_grid_search"
echo "  kill <PID>"
echo "========================================="
echo ""

# Ask for confirmation
read -p "Continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    echo "❌ Aborted"
    exit 1
fi

# Run in background
nohup python scripts/auto_grid_search_24_7.py --strategy "$STRATEGY" > "$LOG_FILE" 2>&1 &

PID=$!
echo ""
echo "✅ Started in background"
echo "   PID: $PID"
echo "   Log: $LOG_FILE"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo ""

# Save PID for later
echo $PID > outputs/grid_search/pid.txt
