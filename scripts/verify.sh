#!/usr/bin/env bash
#
# Verification Loop — Pre-commit Quality Gate
# Based on ECC verification-loop skill
#
# Usage: ./scripts/verify.sh [--skip-tests]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse args
SKIP_TESTS=false
if [[ "$1" == "--skip-tests" ]]; then
    SKIP_TESTS=true
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  XAUUSD AI — VERIFICATION LOOP"
echo "════════════════════════════════════════════════════════════"
echo ""

# Activate venv
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓${NC} Virtual environment activated"
else
    echo -e "${RED}✗${NC} No .venv found, please run: python -m venv .venv"
    exit 1
fi

ERRORS=0

#
# Phase 1: Lint Check
#
echo ""
echo -e "${BLUE}▶ Phase 1: Lint Check${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v ruff &> /dev/null; then
    if ruff check src/ scripts/ --quiet 2>&1 | head -30; then
        echo -e "${GREEN}✓${NC} Lint check passed"
    else
        echo -e "${YELLOW}⚠${NC} Lint warnings found (non-blocking)"
    fi
else
    echo -e "${YELLOW}⚠${NC} ruff not installed, skipping lint check"
fi

#
# Phase 2: Type Check
#
echo ""
echo -e "${BLUE}▶ Phase 2: Type Check${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v pyright &> /dev/null; then
    if pyright src/ --level warning 2>&1 | tail -20; then
        echo -e "${GREEN}✓${NC} Type check passed"
    else
        echo -e "${YELLOW}⚠${NC} Type warnings found (non-blocking)"
    fi
else
    echo -e "${YELLOW}⚠${NC} pyright not installed, skipping type check"
fi

#
# Phase 3: Security Scan
#
echo ""
echo -e "${BLUE}▶ Phase 3: Security Scan${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Checking for exposed credentials..."

# Check for hardcoded passwords
SECRETS=$(grep -rn "password.*=.*['\"]" configs/ 2>/dev/null | grep -v "_env" | grep -v "example" | wc -l | tr -d ' ')

if [[ "$SECRETS" -gt 0 ]]; then
    echo -e "${RED}✗${NC} FAIL: Found $SECRETS potential hardcoded credentials"
    grep -rn "password.*=.*['\"]" configs/ 2>/dev/null | grep -v "_env" | grep -v "example" | head -5
    echo ""
    echo "Please move credentials to .env file:"
    echo "  1. Create .env from .env.example"
    echo "  2. Remove hardcoded values from configs/"
    echo "  3. Use environment variables (login_env, password_env)"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓${NC} No hardcoded credentials found"
fi

# Check for API keys
API_KEYS=$(grep -rn "api_key.*=.*['\"]" configs/ src/ 2>/dev/null | grep -v "_env" | grep -v "example" | wc -l | tr -d ' ')

if [[ "$API_KEYS" -gt 0 ]]; then
    echo -e "${YELLOW}⚠${NC} Warning: Found $API_KEYS potential API keys"
    grep -rn "api_key.*=.*['\"]" configs/ src/ 2>/dev/null | grep -v "_env" | grep -v "example" | head -3
else
    echo -e "${GREEN}✓${NC} No hardcoded API keys found"
fi

#
# Phase 4: Test Suite
#
if [[ "$SKIP_TESTS" == false ]]; then
    echo ""
    echo -e "${BLUE}▶ Phase 4: Test Suite${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if pytest tests/ \
        --cov=src/xauusd_ai \
        --cov-report=term-missing:skip-covered \
        --cov-fail-under=70 \
        -n auto \
        --tb=short \
        2>&1 | tee /tmp/pytest_output.txt | tail -50; then
        echo -e "${GREEN}✓${NC} All tests passed"
    else
        echo -e "${RED}✗${NC} Tests failed"
        ERRORS=$((ERRORS + 1))
    fi

    # Show coverage summary
    echo ""
    echo "Coverage Summary:"
    grep -A20 "^TOTAL" /tmp/pytest_output.txt 2>/dev/null || echo "No coverage data"
else
    echo ""
    echo -e "${YELLOW}⚠${NC} Tests skipped (--skip-tests flag)"
fi

#
# Phase 5: Changed Files
#
echo ""
echo -e "${BLUE}▶ Phase 5: Changed Files${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v git &> /dev/null && [[ -d .git ]]; then
    CHANGED=$(git diff --name-only HEAD 2>/dev/null | wc -l | tr -d ' ')
    STAGED=$(git diff --cached --name-only 2>/dev/null | wc -l | tr -d ' ')
    
    echo "Changed files: $CHANGED"
    echo "Staged files: $STAGED"
    
    if [[ "$CHANGED" -gt 0 ]]; then
        echo ""
        git diff --stat HEAD 2>/dev/null | head -20
    fi
    
    if [[ "$STAGED" -gt 0 ]]; then
        echo ""
        echo "Staged changes:"
        git diff --cached --stat 2>/dev/null | head -20
    fi
else
    echo "Not a git repository"
fi

#
# Final Summary
#
echo ""
echo "════════════════════════════════════════════════════════════"

if [[ $ERRORS -eq 0 ]]; then
    echo -e "  ${GREEN}✓ VERIFICATION PASSED${NC}"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "Ready to commit! ✨"
    echo ""
    exit 0
else
    echo -e "  ${RED}✗ VERIFICATION FAILED${NC}"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "Found $ERRORS critical issue(s). Please fix before committing."
    echo ""
    exit 1
fi
