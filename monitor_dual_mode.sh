#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# DUAL-MODE MONITORING SCRIPT
# Tracks skip rates, P&L, and signal counts for both accounts
# ═══════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════════════════════════════"
echo "📊 XAUUSD AI — Dual-Mode Monitoring"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "⏰ Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# ACC1 (PAPER MODE)
# ─────────────────────────────────────────────────────────────────────────────

echo "📝 ACC1 (Paper Mode) — Shadow Execution"
echo "────────────────────────────────────────────────────────────────────────────"

if docker ps --format '{{.Names}}' | grep -q "^live-acc1$"; then
    ACC1_RUNNING="✅ Running"
else
    ACC1_RUNNING="❌ Stopped"
fi
echo "  Status: $ACC1_RUNNING"

if [ "$ACC1_RUNNING" = "✅ Running" ]; then
    # Extract skip rate from logs
    TOTAL_ACC1=$(docker logs live-acc1 2>&1 | grep -c "should_trade" || echo "0")
    SKIPPED_ACC1=$(docker logs live-acc1 2>&1 | grep -c "PROFIT_FILTER BLOCKED" || echo "0")
    
    if [ "$TOTAL_ACC1" -gt 0 ]; then
        SKIP_RATE_ACC1=$((SKIPPED_ACC1 * 100 / TOTAL_ACC1))
        echo "  Total signals: $TOTAL_ACC1"
        echo "  Skipped: $SKIPPED_ACC1 ($SKIP_RATE_ACC1%)"
        echo "  Kept: $((TOTAL_ACC1 - SKIPPED_ACC1))"
        
        # Check if skip rate is in target range (60-82%)
        if [ "$SKIP_RATE_ACC1" -ge 60 ] && [ "$SKIP_RATE_ACC1" -le 82 ]; then
            echo "  Skip rate status: ✅ Within target (60-82%)"
        elif [ "$SKIP_RATE_ACC1" -lt 60 ]; then
            echo "  Skip rate status: ⚠️  Below target (need $15+ threshold)"
        else
            echo "  Skip rate status: ⚠️  Above target (consider lowering threshold)"
        fi
    else
        echo "  No signals processed yet"
    fi
    
    # Count paper trades
    if [ -f "outputs/paper_trade_signals_acc1.csv" ]; then
        PAPER_COUNT=$(wc -l < outputs/paper_trade_signals_acc1.csv)
        echo "  Paper trades logged: $((PAPER_COUNT - 1))"  # Subtract header
    fi
    
    # Check last activity
    LAST_LOG_ACC1=$(docker logs live-acc1 --tail 1 2>&1 | head -1)
    echo "  Last log: ${LAST_LOG_ACC1:0:80}..."
else
    echo "  Container not running. Start with: docker compose up -d live-acc1"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# ACC2 (LIVE DEMO)
# ─────────────────────────────────────────────────────────────────────────────

echo "💰 ACC2 (Live Demo) — Real Trading on Demo Account"
echo "────────────────────────────────────────────────────────────────────────────"

if docker ps --format '{{.Names}}' | grep -q "^live$"; then
    ACC2_RUNNING="✅ Running"
else
    ACC2_RUNNING="❌ Stopped"
fi
echo "  Status: $ACC2_RUNNING"

if [ "$ACC2_RUNNING" = "✅ Running" ]; then
    # Extract skip rate from logs
    TOTAL_ACC2=$(docker logs live 2>&1 | grep -c "should_trade" || echo "0")
    SKIPPED_ACC2=$(docker logs live 2>&1 | grep -c "PROFIT_FILTER BLOCKED" || echo "0")
    
    if [ "$TOTAL_ACC2" -gt 0 ]; then
        SKIP_RATE_ACC2=$((SKIPPED_ACC2 * 100 / TOTAL_ACC2))
        echo "  Total signals: $TOTAL_ACC2"
        echo "  Skipped: $SKIPPED_ACC2 ($SKIP_RATE_ACC2%)"
        echo "  Kept: $((TOTAL_ACC2 - SKIPPED_ACC2))"
        
        # Check if skip rate is in target range (60-82%)
        if [ "$SKIP_RATE_ACC2" -ge 60 ] && [ "$SKIP_RATE_ACC2" -le 82 ]; then
            echo "  Skip rate status: ✅ Within target (60-82%)"
        elif [ "$SKIP_RATE_ACC2" -lt 60 ]; then
            echo "  Skip rate status: ⚠️  Below target (need $15+ threshold)"
        else
            echo "  Skip rate status: ⚠️  Above target (consider lowering threshold)"
        fi
    else
        echo "  No signals processed yet"
    fi
    
    # Count live trades
    if [ -f "outputs/live_closed_trades_acc2.csv" ]; then
        TRADE_COUNT=$(wc -l < outputs/live_closed_trades_acc2.csv)
        TRADE_COUNT=$((TRADE_COUNT - 1))  # Subtract header
        echo "  Closed trades: $TRADE_COUNT"
        
        # Calculate P&L if trades exist
        if [ "$TRADE_COUNT" -gt 0 ]; then
            # Sum profit column (column 8, assuming CSV format)
            TOTAL_PNL=$(tail -n +2 outputs/live_closed_trades_acc2.csv | awk -F',' '{sum+=$8} END {printf "%.2f", sum}')
            echo "  Total P&L: \$$TOTAL_PNL"
            
            # Calculate win rate
            WIN_COUNT=$(tail -n +2 outputs/live_closed_trades_acc2.csv | awk -F',' '$8 > 0 {count++} END {print count}')
            WIN_RATE=$((WIN_COUNT * 100 / TRADE_COUNT))
            echo "  Win rate: $WIN_RATE% ($WIN_COUNT/$TRADE_COUNT)"
        fi
    fi
    
    # Check last activity
    LAST_LOG_ACC2=$(docker logs live --tail 1 2>&1 | head -1)
    echo "  Last log: ${LAST_LOG_ACC2:0:80}..."
else
    echo "  Container not running. Start with: docker compose up -d live"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON & SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

echo "📊 Comparison Summary"
echo "────────────────────────────────────────────────────────────────────────────"

if [ "$ACC1_RUNNING" = "✅ Running" ] && [ "$ACC2_RUNNING" = "✅ Running" ]; then
    echo "  Both accounts running ✅"
    
    if [ "$TOTAL_ACC1" -gt 0 ] && [ "$TOTAL_ACC2" -gt 0 ]; then
        echo ""
        echo "  Skip Rate Comparison:"
        echo "    ACC1 (paper): $SKIP_RATE_ACC1%"
        echo "    ACC2 (demo):  $SKIP_RATE_ACC2%"
        
        # Calculate difference
        DIFF=$((SKIP_RATE_ACC1 - SKIP_RATE_ACC2))
        if [ $DIFF -lt 0 ]; then
            DIFF=$((-DIFF))
            echo "    Difference: ${DIFF}pp (ACC2 higher)"
        else
            echo "    Difference: ${DIFF}pp (ACC1 higher)"
        fi
        
        # Expected difference should be small (<5pp)
        if [ $DIFF -le 5 ]; then
            echo "    ✅ Skip rates consistent between accounts"
        else
            echo "    ⚠️  Large difference - check if configs are synced"
        fi
    fi
else
    if [ "$ACC1_RUNNING" != "✅ Running" ]; then
        echo "  ⚠️  ACC1 (paper) not running"
    fi
    if [ "$ACC2_RUNNING" != "✅ Running" ]; then
        echo "  ⚠️  ACC2 (demo) not running"
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📝 Quick Actions:"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  # View live logs:"
echo "  docker logs -f live-acc1  # ACC1 paper"
echo "  docker logs -f live       # ACC2 demo"
echo ""
echo "  # Restart accounts:"
echo "  docker compose restart live-acc1"
echo "  docker compose restart live"
echo ""
echo "  # Stop all:"
echo "  docker compose down live live-acc1"
echo ""
echo "  # Check detailed logs (last 50 lines):"
echo "  docker logs live-acc1 --tail 50"
echo "  docker logs live --tail 50"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
