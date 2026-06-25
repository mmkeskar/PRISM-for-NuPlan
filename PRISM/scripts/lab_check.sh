#!/usr/bin/env bash
# PRISM lab environment check
# ─────────────────────────────────────────────────────────────────────────────
# Run from the PRISM/PRISM/ directory after activating the conda env:
#   conda activate prism
#   source lab.env
#   bash scripts/lab_check.sh
#   # or via Makefile: make check
# ─────────────────────────────────────────────────────────────────────────────

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

echo ""
echo "PRISM Lab Environment Check"
echo "════════════════════════════════════════════════════════"

# ── 1. Python version ─────────────────────────────────────────────────────────
echo ""
echo "── Python ──"
PY_VER=$(python --version 2>&1 | awk '{print $2}')
MAJOR=$(echo "$PY_VER" | cut -d. -f1)
MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
    pass "Python $PY_VER"
else
    fail "Python $PY_VER — need 3.10+. Activate the prism conda env: conda activate prism"
fi

# ── 2. Core Python packages ───────────────────────────────────────────────────
echo ""
echo "── Core packages ──"

check_import() {
    local pkg=$1
    local label=${2:-$pkg}
    if python -c "import $pkg" 2>/dev/null; then
        VER=$(python -c "import $pkg; print(getattr($pkg, '__version__', 'unknown'))" 2>/dev/null)
        pass "$label $VER"
    else
        fail "$label not importable — run: make setup"
    fi
}

check_import torch
check_import numpy
check_import scipy
check_import matplotlib
check_import yaml pyyaml
check_import stable_baselines3 stable-baselines3

# ── 3. CUDA ───────────────────────────────────────────────────────────────────
echo ""
echo "── CUDA ──"
CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_OK" = "True" ]; then
    CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    N_GPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    pass "CUDA $CUDA_VER  ($N_GPU GPU(s))"
else
    warn "CUDA not available — training will fall back to CPU"
fi

# ── 4. nuplan-devkit ──────────────────────────────────────────────────────────
echo ""
echo "── nuplan-devkit ──"
if python -c "import nuplan" 2>/dev/null; then
    NUPLAN_VER=$(python -c "import importlib.metadata; print(importlib.metadata.version('nuplan-devkit'))" 2>/dev/null || echo "unknown")
    pass "nuplan-devkit $NUPLAN_VER"
    # Check key submodules used by PRISM
    for mod in \
        "nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder" \
        "nuplan.planning.scenario_builder.scenario_filter" \
        "nuplan.planning.utils.multithreading.worker_sequential"; do
        if python -c "import $mod" 2>/dev/null; then
            pass "  $mod"
        else
            fail "  $mod missing"
        fi
    done
else
    fail "nuplan not importable — run: pip install -e \$NUPLAN_DEVKIT_PATH"
fi

# ── 5. carl_nuplan ────────────────────────────────────────────────────────────
echo ""
echo "── carl_nuplan ──"
# train.py adds ../nuPlan to sys.path; replicate that here
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NUPLAN_DIR="$(cd "$PRISM_ROOT/../nuPlan" 2>/dev/null && pwd)" || NUPLAN_DIR=""

if [ -n "$NUPLAN_DIR" ] && python -c "import sys; sys.path.insert(0,'$NUPLAN_DIR'); import carl_nuplan" 2>/dev/null; then
    pass "carl_nuplan (found in $NUPLAN_DIR)"
    for mod in \
        "carl_nuplan.planning.gym.cache.gym_scenario_cache" \
        "carl_nuplan.planning.gym.environment.scenario_sampler.cache_scenario_sampler"; do
        if python -c "import sys; sys.path.insert(0,'$NUPLAN_DIR'); import $mod" 2>/dev/null; then
            pass "  $mod"
        else
            fail "  $mod missing"
        fi
    done
else
    fail "carl_nuplan not found in $NUPLAN_DIR — check that nuPlan/carl_nuplan/ exists in the repo root"
fi

# ── 6. PRISM package ──────────────────────────────────────────────────────────
echo ""
echo "── PRISM package ──"
if python -c "import prism" 2>/dev/null; then
    pass "prism package importable"
    for mod in \
        prism.morl.utility_functions \
        prism.morl.dpmorl_trainer \
        prism.morl.cvar_lagrangian \
        prism.curriculum.alpha_schedule \
        prism.utils.hyperparams \
        prism.utils.zt_normaliser; do
        if python -c "import $mod" 2>/dev/null; then
            pass "  $mod"
        else
            fail "  $mod"
        fi
    done
else
    fail "prism package not importable — run: pip install -e ."
fi

# Check for missing env/ modules (not yet implemented)
echo ""
echo "── PRISM env/ module (implementation status) ──"
for mod in \
    prism.env.nuplan_env \
    prism.env.rewards \
    prism.env.safety_cost \
    prism.env.regime_detector; do
    if python -c "import $mod" 2>/dev/null; then
        pass "  $mod"
    else
        warn "  $mod NOT YET IMPLEMENTED — training will fail without it"
    fi
done

# ── 7. nuPlan data paths ──────────────────────────────────────────────────────
echo ""
echo "── nuPlan data ──"

check_path() {
    local label=$1
    local path=$2
    if [ -z "$path" ]; then
        warn "$label path not set (check lab.env)"
    elif [ -d "$path" ]; then
        N=$(find "$path" -name "*.db" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$N" -gt 0 ]; then
            pass "$label: $path  ($N .db files)"
        else
            warn "$label exists but contains no .db files: $path"
        fi
    else
        fail "$label not found: $path"
    fi
}

check_path "NUPLAN_DATA_ROOT" "${NUPLAN_DATA_ROOT:-}"
check_path "NUPLAN_MINI_ROOT" "${NUPLAN_MINI_ROOT:-}"

# Maps
MAP_ROOT="${NUPLAN_MAP_ROOT:-}"
if [ -z "$MAP_ROOT" ]; then
    warn "NUPLAN_MAP_ROOT not set"
elif [ -d "$MAP_ROOT" ]; then
    MAP_JSON="$MAP_ROOT/nuplan-maps-v1.0.json"
    if [ -f "$MAP_JSON" ]; then
        pass "NUPLAN_MAP_ROOT: $MAP_ROOT"
    else
        warn "NUPLAN_MAP_ROOT exists but nuplan-maps-v1.0.json not found: $MAP_ROOT"
    fi
else
    fail "NUPLAN_MAP_ROOT not found: $MAP_ROOT"
fi

# ── 8. Scenario cache ─────────────────────────────────────────────────────────
echo ""
echo "── Scenario cache ──"

check_cache() {
    local label=$1
    local path=$2
    if [ -z "$path" ]; then
        warn "$label path not set"
    elif [ -d "$path" ]; then
        N=$(find "$path" -name "*.gz" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$N" -gt 0 ]; then
            pass "$label: $path  ($N .gz scenarios)"
        else
            warn "$label exists but is empty: $path — run make cache-mini"
        fi
    else
        warn "$label not built yet: $path — run make cache-mini"
    fi
}

check_cache "MINI_CACHE" "${MINI_CACHE_PATH:-}"
check_cache "FULL_CACHE"  "${CACHE_PATH:-}"

# ── 9. hyperparams.json ───────────────────────────────────────────────────────
echo ""
echo "── hyperparams.json ──"
HP_PATH="$PRISM_ROOT/hyperparams.json"
if [ -f "$HP_PATH" ]; then
    pass "hyperparams.json found"
    # Quick sanity check — key must exist
    python -c "
import json, sys
with open('$HP_PATH') as f:
    hp = json.load(f)
required = ['reward_scaling','epsilon_curve','lead_times','z_normalisation']
missing = [k for k in required if k not in hp]
if missing:
    print('WARN: missing keys: ' + ', '.join(missing)); sys.exit(1)
" 2>/dev/null && pass "  keys OK" || warn "  hyperparams.json exists but may be incomplete"
else
    warn "hyperparams.json not found — run: make hyperparams-mini"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo -e "  ${GREEN}PASS${NC}: $PASS    ${YELLOW}WARN${NC}: $WARN    ${RED}FAIL${NC}: $FAIL"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}Environment is NOT ready. Fix the FAILs above before training.${NC}"
    exit 1
elif [ "$WARN" -gt 0 ]; then
    echo -e "${YELLOW}Environment has warnings. Review WARNs above.${NC}"
    echo "If all WARNs are expected (e.g., prism/env/ not yet implemented),"
    echo "the environment is ready for the next setup step."
    exit 0
else
    echo -e "${GREEN}All checks passed. Ready to train.${NC}"
    exit 0
fi
