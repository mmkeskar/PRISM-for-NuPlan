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
    echo "════════════════════════════════════════════════════════"
    echo -e "  ${GREEN}PASS${NC}: $PASS    ${YELLOW}WARN${NC}: $WARN    ${RED}FAIL${NC}: $FAIL"
    echo ""
    echo -e "${RED}Run: make hyperparams-mini${NC}"
    exit 1
fi

pass "hyperparams.json found: $HP_FILE"

# ── All checks delegated to Python for floating-point math ───────────────────
echo ""
echo "── Required keys ──"
echo ""
echo "── Reward scaling ──"
echo ""
echo "── CVaR epsilon curve ──"
echo ""
echo "── z_t normalisation ──"
echo ""
echo "── Indicator weights ──"
echo ""
echo "── Metadata ──"

python - "$HP_FILE" <<'PYEOF'
import json, sys, math

hp_path = sys.argv[1]
with open(hp_path) as f:
    hp = json.load(f)

GREEN  = '\033[0;32m'
YELLOW = '\033[1;33m'
RED    = '\033[0;31m'
NC     = '\033[0m'

passes = 0
warns  = 0
fails  = 0


def _ok(val, lo, hi):
    return lo <= val <= hi


def pass_(msg):
    global passes
    print(f"  {GREEN}PASS{NC}  {msg}")
    passes += 1


def warn_(msg):
    global warns
    print(f"  {YELLOW}WARN{NC}  {msg}")
    warns += 1


def fail_(msg):
    global fails
    print(f"  {RED}FAIL{NC}  {msg}")
    fails += 1


def check_range(label, val, lo, hi, unit="", warn_only=False):
    """Emit PASS/WARN/FAIL for a scalar value against [lo, hi]."""
    range_str = f"[{lo} – {hi}]{(' ' + unit) if unit else ''}"
    val_str   = f"{val:.4f}{(' ' + unit) if unit else ''}"
    msg = f"{label:<20}  got {val_str:<14}  expect {range_str}"
    if _ok(val, lo, hi):
        pass_(msg)
    elif warn_only:
        warn_(msg)
    else:
        fail_(msg)


# ── Required top-level keys ───────────────────────────────────────────────────
REQUIRED = [
    "reward_scaling", "epsilon_curve", "lead_times",
    "indicator_weights", "indicator_caps", "outcome_weights",
    "z_normalisation", "alpha_curriculum", "safety_thresholds", "metadata",
]
for key in REQUIRED:
    if key in hp:
        pass_(f"key present: {key}")
    else:
        fail_(f"key missing:  {key}")

# ── Reward scaling ────────────────────────────────────────────────────────────
sc = hp.get("reward_scaling", {})

check_range("sigma_j_sq", sc.get("sigma_j_sq", -1),  0.1,  50.0, "(m/s³)²")
check_range("gamma_a",    sc.get("gamma_a",    -1),  0.3,   3.0, "m/s²",   warn_only=True)
check_range("phi",        sc.get("phi",        -1),  0.01,  0.3, "rad/s")
check_range("tau",        sc.get("tau",        -1),  1.0,   6.0, "s")
check_range("beta",       sc.get("beta",       -1),  0.4,   0.6)   # fixed at 0.5
check_range("sigma_d",    sc.get("sigma_d",    -1),  0.15,  0.25, "m")

# ── CVaR epsilon curve ────────────────────────────────────────────────────────
ec = hp.get("epsilon_curve", {})
alphas = sorted(ec.keys(), key=float)
vals   = [ec[a] for a in alphas]

if len(vals) >= 2:
    monotone = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    curve_str = "  ".join(f"{a}:{v:.2f}" for a, v in zip(alphas[::4], vals[::4]))
    msg = f"monotone increasing    {curve_str}"
    if monotone:
        pass_(msg)
    else:
        fail_(msg + "  ← CVaR must increase with alpha")
else:
    fail_("epsilon_curve has fewer than 2 entries")

# All epsilon values must be non-negative
neg = [(a, v) for a, v in zip(alphas, vals) if v < 0]
if neg:
    fail_(f"negative epsilon values: {neg}")
else:
    pass_("all epsilon values >= 0")

# At least some non-zero (catches the all-zeros bug from missing safety costs)
nonzero = sum(1 for v in vals if v > 0)
if nonzero == 0:
    fail_("all epsilon values are 0.0 — safety costs were not recorded (see _extract_episode)")
elif nonzero < len(vals) // 2:
    warn_(f"only {nonzero}/{len(vals)} epsilon values are non-zero — safety events may be under-counted")
else:
    pass_(f"non-zero epsilon values: {nonzero}/{len(vals)}")

# ── z_t normalisation ─────────────────────────────────────────────────────────
zn     = hp.get("z_normalisation", {})
z_mu   = zn.get("z_mu",    [])
z_sigma = zn.get("z_sigma", [])

if len(z_mu) == 4:
    pass_("z_mu has 4 entries (one per style objective)")
    for i, (name, val) in enumerate(zip(["comfort","progress","lateral","spacing"], z_mu)):
        if val > 0:
            pass_(f"  z_mu[{i}] {name:<10}  {val:.3f}  > 0")
        else:
            fail_(f"  z_mu[{i}] {name:<10}  {val:.3f}  must be positive")
else:
    fail_(f"z_mu length {len(z_mu)}, expected 4")

if len(z_sigma) == 4:
    pass_("z_sigma has 4 entries")
    for i, (name, val) in enumerate(zip(["comfort","progress","lateral","spacing"], z_sigma)):
        if val > 1e-4:
            pass_(f"  z_sigma[{i}] {name:<10}  {val:.3f}  > 0")
        else:
            warn_(f"  z_sigma[{i}] {name:<10}  {val:.4f}  near zero — reward may be degenerate")
else:
    fail_(f"z_sigma length {len(z_sigma)}, expected 4")

# ── Indicator weights ─────────────────────────────────────────────────────────
iw = hp.get("indicator_weights", {})
EXPECTED_INDICATORS = ["ttc", "thw", "speed", "blind_spot", "red_light"]
for ind in EXPECTED_INDICATORS:
    v = iw.get(ind, None)
    if v is None:
        fail_(f"indicator_weights missing: {ind}")
    elif v > 0:
        pass_(f"  {ind:<12}  w = {v:.4f}  > 0")
    else:
        fail_(f"  {ind:<12}  w = {v:.4f}  must be > 0")

# ── Metadata ──────────────────────────────────────────────────────────────────
meta = hp.get("metadata", {})
n = meta.get("n_warmup_rollouts", 0)
if n >= 100:
    pass_(f"n_warmup_rollouts = {n}  (>= 100)")
elif n >= 20:
    warn_(f"n_warmup_rollouts = {n}  (recommend >= 100 for stable tail statistics)")
else:
    fail_(f"n_warmup_rollouts = {n}  (too few for reliable CVaR estimates)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("")
print("════════════════════════════════════════════════════════")
print(f"  {GREEN}PASS{NC}: {passes}    {YELLOW}WARN{NC}: {warns}    {RED}FAIL{NC}: {fails}")
print("")
if fails > 0:
    print(f"{RED}hyperparams.json has errors. Delete it and re-run: make hyperparams-mini{NC}")
    sys.exit(1)
elif warns > 0:
    print(f"{YELLOW}hyperparams.json looks mostly OK. Review WARNs above before training.{NC}")
    sys.exit(0)
else:
    print(f"{GREEN}hyperparams.json looks good. Ready to train.{NC}")
    sys.exit(0)
PYEOF

PY_EXIT=$?

# Propagate Python's exit code
if [ $PY_EXIT -ne 0 ]; then
    exit 1
fi
