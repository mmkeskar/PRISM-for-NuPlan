#!/usr/bin/env python3
"""
check_hyperparams.py — Validate hyperparams.json values are physically plausible.

Called by scripts/check_hyperparams.sh (and make check-hyperparams).
Prints colour-coded PASS / WARN / FAIL lines with actual values and expected ranges.
Exit code: 0 = all pass or warn only, 1 = any fail.
"""

import json
import sys

GREEN  = '\033[0;32m'
YELLOW = '\033[1;33m'
RED    = '\033[0;31m'
NC     = '\033[0m'

_passes = 0
_warns  = 0
_fails  = 0


def section(title):
    print(f"\n── {title} ──")


def pass_(msg):
    global _passes
    print(f"  {GREEN}PASS{NC}  {msg}")
    _passes += 1


def warn_(msg):
    global _warns
    print(f"  {YELLOW}WARN{NC}  {msg}")
    _warns += 1


def fail_(msg):
    global _fails
    print(f"  {RED}FAIL{NC}  {msg}")
    _fails += 1


def check_range(label, val, lo, hi, unit="", warn_only=False):
    """
    Emit PASS / WARN / FAIL for a scalar value against [lo, hi].

    Justifications for each expected range:
      sigma_j_sq  0.1–50  (m/s³)²   — comfort reward calibrated so jerk=sqrt(σ²) gives r≈0.37;
                                        RMS jerk in comfortable driving is 0.5–2 m/s³ → σ²≈2–30;
                                        upper 50 is generous for edge cases.
      gamma_a     0.3–3.0  m/s²      — r_accel=0.63 when |a|=γ_a; empirical mean |a| in urban
                                        driving is 0.5–2 m/s²; warn (not fail) if low since
                                        smooth mini scenarios can be legitimately near 0.2 m/s².
      phi         0.01–0.3  rad/s    — r_heading calibrated so |Δψ|=φ gives r≈0.37; std of
                                        heading rate is ~0.02–0.10 rad/s on typical roads
                                        (straight: ~0, lane change: ~0.05, gentle turn: ~0.1).
      tau         1.0–6.0  s         — mean TTC when a lead vehicle is present and closing;
                                        safe following ≈ 2–4 s, aggressive ≈ 1–2 s, relaxed ≈ 5 s.
      beta        0.4–0.6            — fixed at 0.5 by design (speed shortfall scaling).
      sigma_d     0.15–0.25  m       — fixed at 0.2 m by design (lateral deviation tolerance).
    """
    range_str = f"[{lo} – {hi}]{(' ' + unit) if unit else ''}"
    val_str   = f"{val:.4f}{(' ' + unit) if unit else ''}"
    msg = f"{label:<20}  got {val_str:<16}  expect {range_str}"
    if lo <= val <= hi:
        pass_(msg)
    elif warn_only:
        warn_(msg)
    else:
        fail_(msg)


def main(hp_path: str):
    with open(hp_path) as f:
        hp = json.load(f)

    # ── Required top-level keys ───────────────────────────────────────────────
    section("Required keys")
    REQUIRED = [
        "reward_scaling", "lead_times",
        "indicator_weights", "indicator_caps", "outcome_weights",
        "z_normalisation", "alpha_curriculum", "safety_thresholds", "metadata",
    ]
    for key in REQUIRED:
        if key in hp:
            pass_(f"key present: {key}")
        else:
            fail_(f"key missing:  {key}")

    # ── Reward scaling ────────────────────────────────────────────────────────
    section("Reward scaling")
    sc = hp.get("reward_scaling", {})
    check_range("sigma_j_sq", sc.get("sigma_j_sq", -1),  0.1,  50.0, "(m/s³)²")
    check_range("gamma_a",    sc.get("gamma_a",    -1),  0.3,   3.0, "m/s²",   warn_only=True)
    check_range("phi",        sc.get("phi",        -1),  0.01,  0.3, "rad/s")
    check_range("tau",        sc.get("tau",        -1),  1.0,   6.0, "s")
    check_range("beta",       sc.get("beta",       -1),  0.4,   0.6)
    check_range("sigma_d",    sc.get("sigma_d",    -1),  0.15,  0.25, "m")

    # ── z_t normalisation ─────────────────────────────────────────────────────
    section("z_t normalisation")
    zn      = hp.get("z_normalisation", {})
    z_mu    = zn.get("z_mu",    [])
    z_sigma = zn.get("z_sigma", [])
    names   = ["comfort", "progress", "lateral", "spacing"]

    if len(z_mu) == 4:
        pass_("z_mu has 4 entries (one per style objective)")
        for i, (name, val) in enumerate(zip(names, z_mu)):
            if val > 0:
                pass_(f"  z_mu[{i}] {name:<10}  {val:.3f}  > 0")
            else:
                fail_(f"  z_mu[{i}] {name:<10}  {val:.3f}  must be positive")
    else:
        fail_(f"z_mu length {len(z_mu)}, expected 4")

    if len(z_sigma) == 4:
        pass_("z_sigma has 4 entries")
        for i, (name, val) in enumerate(zip(names, z_sigma)):
            if val > 1e-4:
                pass_(f"  z_sigma[{i}] {name:<10}  {val:.3f}  > 0")
            else:
                warn_(f"  z_sigma[{i}] {name:<10}  {val:.4f}  near zero — reward may be degenerate")
    else:
        fail_(f"z_sigma length {len(z_sigma)}, expected 4")

    # ── Indicator weights ─────────────────────────────────────────────────────
    section("Indicator weights")
    iw = hp.get("indicator_weights", {})
    for ind in ["ttc", "thw", "speed", "blind_spot", "red_light"]:
        v = iw.get(ind)
        if v is None:
            fail_(f"  {ind:<12}  missing")
        elif v > 0:
            pass_(f"  {ind:<12}  w = {v:.4f}  > 0")
        else:
            fail_(f"  {ind:<12}  w = {v:.4f}  must be > 0")

    # ── Metadata ──────────────────────────────────────────────────────────────
    section("Metadata")
    meta = hp.get("metadata", {})
    n    = meta.get("n_warmup_rollouts", 0)
    if n >= 100:
        pass_(f"n_warmup_rollouts = {n}  (>= 100)")
    elif n >= 20:
        warn_(f"n_warmup_rollouts = {n}  (recommend >= 100 for stable tail statistics)")
    else:
        fail_(f"n_warmup_rollouts = {n}  (too few — CVaR estimates will be unreliable)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("")
    print("════════════════════════════════════════════════════════")
    print(f"  {GREEN}PASS{NC}: {_passes}    {YELLOW}WARN{NC}: {_warns}    {RED}FAIL{NC}: {_fails}")
    print("")
    if _fails > 0:
        print(f"{RED}hyperparams.json has errors. Delete it and re-run: make hyperparams-mini{NC}")
        sys.exit(1)
    elif _warns > 0:
        print(f"{YELLOW}hyperparams.json looks mostly OK. Review WARNs above before training.{NC}")
        sys.exit(0)
    else:
        print(f"{GREEN}hyperparams.json looks good. Ready to train.{NC}")
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path/to/hyperparams.json>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
