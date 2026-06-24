#!/usr/bin/env bash
# check_env.sh — Validate lab.env paths before running make setup or make cache.
# No conda activation required — pure bash, no Python.
#
# Usage:
#   source lab.env
#   bash scripts/check_env.sh
#   # or via Makefile: make check-env

set -uo pipefail

PASS=0
WARN=0
FAIL=0

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}PASS${NC}  $1"; ((PASS++)); }
warn() { echo -e "  ${YELLOW}WARN${NC}  $1"; ((WARN++)); }
fail() { echo -e "  ${RED}FAIL${NC}  $1"; ((FAIL++)); }
fix()  { echo -e "        ${YELLOW}→ How to verify:${NC} $1"; }

echo ""
echo "PRISM lab.env Check"
echo "════════════════════════════════════════════════════════"

# ── NUPLAN_DATA_ROOT ──────────────────────────────────────────────────────────
echo ""
echo "── NUPLAN_DATA_ROOT ──"
if [ -z "${NUPLAN_DATA_ROOT:-}" ]; then
    fail "NUPLAN_DATA_ROOT is not set"
    fix "Open lab.env and set NUPLAN_DATA_ROOT to the nuPlan dataset root"
elif [ ! -d "${NUPLAN_DATA_ROOT}" ]; then
    fail "Directory not found: ${NUPLAN_DATA_ROOT}"
    fix "ls \$(dirname ${NUPLAN_DATA_ROOT})  — find where the dataset was placed"
else
    pass "Directory exists: ${NUPLAN_DATA_ROOT}"
    # Expect maps/ subdirectory
    if [ -d "${NUPLAN_DATA_ROOT}/maps" ]; then
        pass "  maps/ subdirectory found"
    else
        fail "  maps/ subdirectory missing"
        fix "ls ${NUPLAN_DATA_ROOT}  — confirm maps/ is present; update NUPLAN_MAP_ROOT if it lives elsewhere"
    fi
    # Expect nuplan-v1.1/ subdirectory
    if [ -d "${NUPLAN_DATA_ROOT}/nuplan-v1.1" ]; then
        pass "  nuplan-v1.1/ subdirectory found"
    else
        fail "  nuplan-v1.1/ subdirectory missing"
        fix "ls ${NUPLAN_DATA_ROOT}  — the dataset version directory may differ; update NUPLAN_DATA_ROOT accordingly"
    fi
fi

# ── NUPLAN_MAP_ROOT ───────────────────────────────────────────────────────────
echo ""
echo "── NUPLAN_MAP_ROOT ──"
if [ -z "${NUPLAN_MAP_ROOT:-}" ]; then
    fail "NUPLAN_MAP_ROOT is not set"
    fix "Open lab.env and set NUPLAN_MAP_ROOT (usually \$NUPLAN_DATA_ROOT/maps)"
elif [ ! -d "${NUPLAN_MAP_ROOT}" ]; then
    fail "Directory not found: ${NUPLAN_MAP_ROOT}"
    fix "ls ${NUPLAN_DATA_ROOT}  — look for a maps/ folder and update NUPLAN_MAP_ROOT"
else
    pass "Directory exists: ${NUPLAN_MAP_ROOT}"
    if [ -f "${NUPLAN_MAP_ROOT}/nuplan-maps-v1.0.json" ]; then
        pass "  nuplan-maps-v1.0.json found"
    else
        fail "  nuplan-maps-v1.0.json not found"
        fix "ls ${NUPLAN_MAP_ROOT}  — confirm the maps JSON is present; the file must be named exactly nuplan-maps-v1.0.json"
    fi
fi

# ── NUPLAN_MINI_ROOT ──────────────────────────────────────────────────────────
echo ""
echo "── NUPLAN_MINI_ROOT ──"
if [ -z "${NUPLAN_MINI_ROOT:-}" ]; then
    fail "NUPLAN_MINI_ROOT is not set"
    fix "Open lab.env and set NUPLAN_MINI_ROOT to the mini split directory (containing *.db files)"
elif [ ! -d "${NUPLAN_MINI_ROOT}" ]; then
    fail "Directory not found: ${NUPLAN_MINI_ROOT}"
    fix "ls ${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/  — locate the mini split folder"
else
    pass "Directory exists: ${NUPLAN_MINI_ROOT}"
    DB_COUNT=$(find "${NUPLAN_MINI_ROOT}" -name "*.db" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${DB_COUNT}" -gt 0 ]; then
        pass "  ${DB_COUNT} .db file(s) found"
    else
        fail "  No .db files found"
        fix "ls ${NUPLAN_MINI_ROOT}  — the mini split should contain .db scenario files"
    fi
fi

# ── NUPLAN_DEVKIT_PATH ────────────────────────────────────────────────────────
echo ""
echo "── NUPLAN_DEVKIT_PATH ──"
if [ -z "${NUPLAN_DEVKIT_PATH:-}" ]; then
    fail "NUPLAN_DEVKIT_PATH is not set"
    fix "Clone nuplan-devkit and set NUPLAN_DEVKIT_PATH to the cloned directory"
elif [ ! -d "${NUPLAN_DEVKIT_PATH}" ]; then
    fail "Directory not found: ${NUPLAN_DEVKIT_PATH}"
    fix "git clone https://github.com/motional/nuplan-devkit <path>  then update NUPLAN_DEVKIT_PATH"
else
    pass "Directory exists: ${NUPLAN_DEVKIT_PATH}"
    if [ -f "${NUPLAN_DEVKIT_PATH}/setup.py" ] || [ -f "${NUPLAN_DEVKIT_PATH}/pyproject.toml" ]; then
        pass "  setup.py / pyproject.toml found — looks like a valid Python package"
    else
        warn "  No setup.py or pyproject.toml found — may not be the devkit root"
        fix "ls ${NUPLAN_DEVKIT_PATH}  — confirm this is the nuplan-devkit repository root"
    fi
fi

# ── TORCH_CUDA_VERSION ────────────────────────────────────────────────────────
echo ""
echo "── TORCH_CUDA_VERSION ──"
CUDA_VER="${TORCH_CUDA_VERSION:-}"
if [ -z "${CUDA_VER}" ]; then
    warn "TORCH_CUDA_VERSION is not set — defaulting to cu121 during make setup"
    fix "Run: nvidia-smi | grep 'CUDA Version'  then set TORCH_CUDA_VERSION in lab.env (e.g. cu121, cu124)"
else
    case "${CUDA_VER}" in
        cu118|cu121|cu124|cu126)
            pass "TORCH_CUDA_VERSION=${CUDA_VER}"
            ;;
        cu*)
            warn "TORCH_CUDA_VERSION=${CUDA_VER} — uncommon value, double-check it matches your driver"
            fix "Run: nvidia-smi | grep 'CUDA Version'  and cross-reference https://pytorch.org/get-started/locally/"
            ;;
        *)
            fail "TORCH_CUDA_VERSION=${CUDA_VER} — does not look like a valid CUDA tag (expected e.g. cu124)"
            fix "Run: nvidia-smi | grep 'CUDA Version'  then set TORCH_CUDA_VERSION=cu<XYZ> in lab.env"
            ;;
    esac
fi

# ── CONDA_ENV ─────────────────────────────────────────────────────────────────
echo ""
echo "── CONDA_ENV ──"
CENV="${CONDA_ENV:-prism}"
pass "CONDA_ENV=${CENV}"
# Check if already exists
if conda env list 2>/dev/null | grep -qE "^${CENV}[[:space:]]"; then
    pass "  conda env '${CENV}' already exists"
else
    warn "  conda env '${CENV}' not yet created — run: make setup"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo -e "  ${GREEN}PASS${NC}: $PASS    ${YELLOW}WARN${NC}: $WARN    ${RED}FAIL${NC}: $FAIL"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}One or more paths in lab.env are incorrect. Fix FAILs before running make setup.${NC}"
    exit 1
elif [ "$WARN" -gt 0 ]; then
    echo -e "${YELLOW}lab.env looks mostly OK but has warnings — review above before continuing.${NC}"
    exit 0
else
    echo -e "${GREEN}lab.env looks good. Run: make setup${NC}"
    exit 0
fi
