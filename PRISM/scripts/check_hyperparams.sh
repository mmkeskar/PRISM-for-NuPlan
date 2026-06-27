#!/usr/bin/env bash
# check_hyperparams.sh — Validate hyperparams.json after running make hyperparams-mini.
# Checks that all required keys exist and that calibrated values are physically plausible.
#
# Usage:
#   source lab.env
#   conda activate prism
#   bash scripts/check_hyperparams.sh
#   # or via Makefile: make check-hyperparams

set -uo pipefail

PASS=0
FAIL=0

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}PASS${NC}  $1"; ((PASS++)); }
fail() { echo -e "  ${RED}FAIL${NC}  $1"; ((FAIL++)); }

echo ""
echo "PRISM Hyperparameter Check"
echo "════════════════════════════════════════════════════════"

# ── File existence ────────────────────────────────────────────────────────────
echo ""
echo "── File ──"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HP_FILE="$PRISM_ROOT/hyperparams.json"

if [ ! -f "$HP_FILE" ]; then
    fail "hyperparams.json not found at $HP_FILE"
    echo ""
    echo -e "${RED}Run: make hyperparams-mini${NC}"
    exit 1
fi

pass "hyperparams.json found: $HP_FILE"

# ── Value checks delegated to Python ─────────────────────────────────────────
# Python owns all remaining output so section headers and PASS/FAIL lines
# appear in the correct order (bash echo and python print are interleaved
# unpredictably when mixed in the same script).

python "$SCRIPT_DIR/check_hyperparams.py" "$HP_FILE"
exit $?
