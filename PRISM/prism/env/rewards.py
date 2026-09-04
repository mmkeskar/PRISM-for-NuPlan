"""
PRISM style reward vector (4-dimensional, all in (0, 1]).

Components:
    r_comfort  : jerk-based comfort
    r_progress : speed only (r_speed, floored) -- see compute_progress()
    r_lateral  : lateral discipline (deviation + heading)
    r_spacing  : TTC-based safe following distance

All values are in (0, 1].  Floor values (delta_d=0.3, delta_s=0.2) prevent
rewards from reaching zero, as required by the PRISM formulation.

Hyperparameters are always read from hyperparams.json (never hardcoded).
See compute_hyperparams.py for how they are calibrated from IDM rollouts.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


# ------------------------------------------------------------------
# Individual reward components
# ------------------------------------------------------------------

def compute_comfort(j_lon: float, j_lat: float, sigma_j_sq: float) -> float:
    """
    Eq. (comfort) in paper.
    r_comfort = exp(-(j_lon^2 + j_lat^2) / sigma_j_sq)
    """
    jerk_sq = j_lon ** 2 + j_lat ** 2
    return float(math.exp(-jerk_sq / max(sigma_j_sq, 1e-8)))


def compute_progress(
    v_ego: float,
    v_des: float,
    beta: float,
) -> float:
    """
    Eq. (progress) in paper.
    r_progress = r_speed = delta_v + (1 - delta_v) * exp(-max(0, v_des-v_ego) / (beta*v_des))
    delta_v = 0.1 (floor, matches the r_dev/r_spacing pattern used elsewhere
    in this vector -- see below).

    Simplified from r_speed * r_accel * (0.5 + 0.5*r_lane); both r_accel and
    r_lane removed (see CHANGES.md):
      - r_accel rewarded acceleration MAGNITUDE unconditionally. At the
        ideal progress behaviour (holding v_des, a_ego~=0), r_accel~=0, so
        the multiplicative r_progress~=0 -- the best possible behaviour
        scored worst on its own objective. It also directly fought
        r_comfort's jerk-minimisation goal. Assertiveness (closing a speed
        gap quickly) is already captured by r_speed accumulated over time
        via z_t/R_t, without a separate term.
      - r_lane prescribed an unconditional leftmost-lane preference --
        wrong at exits, mandatory turns, and in right-lane-norm
        jurisdictions (already a listed limitation). Lane position is not
        a robust style axis. PRISM's contribution is the Pareto front over
        style axes established in the literature (Yusof et al. 2016; Das &
        Won 2023), not the specific reward formulation -- forcing a lane
        preference was exactly the kind of unjustified prescriptiveness
        that formulation shouldn't carry.

    beta is regime-dependent (free_flow / car_following / congested each
    calibrated separately from real data, since a policy's typical
    shortfall differs a lot by regime) -- the CALLER selects which beta to
    pass in based on the active regime; this function stays a pure
    function of a single scalar beta, same as before. See
    compute_hyperparams.py's per-regime calibration and CHANGES.md.

    delta_v is a numerical floor (CLAUDE.md: rewards always in (0,1]), NOT
    a fix for vanishing gradient in the exponential's tail -- the
    derivative still shrinks the same way with or without an additive
    floor. The actual fix for a persistently-low-signal regime is
    calibrating beta (per-regime, above) against real data so the typical
    case lands in the responsive part of the curve, not deep in the tail.
    """
    delta_v = 0.1
    v_des_safe = max(v_des, 0.5)  # avoid division by zero
    shortfall = max(0.0, v_des_safe - v_ego)
    r_speed = math.exp(-shortfall / (max(beta, 1e-6) * v_des_safe))
    return float(delta_v + (1.0 - delta_v) * r_speed)


def _lateral_components(
    d_lat: float,
    delta_psi: float,
    sigma_d: float,
    phi: float,
) -> tuple:
    """Shared by compute_lateral_discipline() and
    compute_style_rewards_verbose() -- returns (r_dev, r_heading)."""
    sigma_d_safe = max(sigma_d, 1e-4)
    phi_safe = max(phi, 1e-4)

    r_dev = 0.3 + 0.7 * math.exp(-((d_lat / sigma_d_safe) ** 2))
    r_heading = math.exp(-(((abs(delta_psi)) / phi_safe) ** 2))
    return r_dev, r_heading


def compute_lateral_discipline(
    d_lat: float,
    delta_psi: float,
    sigma_d: float,
    phi: float,
) -> float:
    """
    Eq. (lateral) in paper.
    r_lateral = r_dev * r_heading
    r_dev     = 0.3 + 0.7 * exp(-(d_lat / sigma_d)^2)
    r_heading = exp(-(|delta_psi| / phi)^2)
    """
    r_dev, r_heading = _lateral_components(
        d_lat=d_lat, delta_psi=delta_psi, sigma_d=sigma_d, phi=phi
    )
    return float(r_dev * r_heading)


def compute_spacing(
    ttc: float,
    tau: float,
) -> float:
    """
    Eq. (spacing) in paper.
    r_spacing = 0.2 + 0.8 * (1 - exp(-(max(0, TTC) / tau)^2))
    """
    tau_safe = max(tau, 0.1)
    ttc_clamped = max(0.0, ttc)
    return float(0.2 + 0.8 * (1.0 - math.exp(-((ttc_clamped / tau_safe) ** 2))))


def compute_ttc(
    d_lead: float,
    v_ego: float,
    v_lead: float,
    has_lead: bool,
) -> float:
    """
    Return TTC in seconds.
    TTC = d_lead / (v_ego - v_lead)  when v_ego > v_lead and has_lead.
    TTC = inf otherwise.
    """
    if not has_lead:
        return float("inf")
    closing_speed = v_ego - v_lead
    if closing_speed <= 0.0:
        return float("inf")
    return float(d_lead / max(closing_speed, 1e-3))


# ------------------------------------------------------------------
# Vector interface
# ------------------------------------------------------------------

def compute_style_rewards(
    j_lon: float,
    j_lat: float,
    v_ego: float,
    v_des: float,
    beta: float,
    d_lat: float,
    delta_psi: float,
    d_lead: float,
    v_lead: float,
    has_lead: bool,
    hp: Dict,
) -> np.ndarray:
    """
    Compute the full 4D style reward vector from the current simulation state.

    beta: the regime-selected shortfall scale (see compute_progress()) --
    hyperparams.json's reward_scaling.beta is now a per-regime dict
    (free_flow/car_following/congested, each calibrated separately from
    real data -- see compute_hyperparams.py and CHANGES.md), so the caller
    looks up the value for whichever regime is currently active and passes
    it in explicitly. Keeps this function agnostic to the dict structure.

    Returns np.ndarray of shape (4,) with values in (0, 1]:
        [r_comfort, r_progress, r_lateral, r_spacing]
    """
    scaling = hp["reward_scaling"]

    r_comfort = compute_comfort(
        j_lon=j_lon,
        j_lat=j_lat,
        sigma_j_sq=scaling["sigma_j_sq"],
    )
    r_progress = compute_progress(
        v_ego=v_ego,
        v_des=v_des,
        beta=beta,
    )
    r_lateral = compute_lateral_discipline(
        d_lat=d_lat,
        delta_psi=delta_psi,
        # Was hp["floor_values"]["delta_d"] -- the wrong key. That's delta_d
        # (the ADDITIVE FLOOR, 0.3, hardcoded directly in
        # compute_lateral_discipline's formula below), not sigma_d (the
        # Gaussian WIDTH, a separate, properly-calibrated/cited value under
        # reward_scaling per compute_hyperparams.py and the paper). In
        # production this silently fed sigma_d=0.3 instead of the intended,
        # cited 0.2 -- the test fixture's floor_values.delta_d happened to
        # already equal 0.2, masking the wrong-key read. See CHANGES.md.
        sigma_d=scaling["sigma_d"],
        phi=scaling["phi"],
    )
    ttc = compute_ttc(d_lead=d_lead, v_ego=v_ego, v_lead=v_lead, has_lead=has_lead)
    r_spacing = compute_spacing(ttc=ttc, tau=scaling["tau"])

    return np.array([r_comfort, r_progress, r_lateral, r_spacing], dtype=np.float32)


def compute_style_rewards_verbose(
    j_lon: float,
    j_lat: float,
    v_ego: float,
    v_des: float,
    beta: float,
    d_lat: float,
    delta_psi: float,
    d_lead: float,
    v_lead: float,
    has_lead: bool,
    hp: Dict,
):
    """
    Same computation as compute_style_rewards(), but also returns the raw
    sub-components each dimension is built from -- for diagnosing whether
    an individual piece (not just the combined 4D vector) is plateauing /
    saturating near its ceiling with little room left to differentiate
    between actions, independent of anything happening in PPO training.
    Opt-in only (PRISMRewardBuilder's log_components flag) -- not used on
    the default hot training path. See scripts/analyze_reward_spread.py
    and CHANGES.md.

    r_progress no longer has a separate sub-component to report -- it IS
    r_speed now (see compute_progress()) -- so "r_speed" below is both the
    progress sub-component and r_vec[1] itself, not a distinct piece.

    Returns (r_vec, components) where r_vec is identical to
    compute_style_rewards()'s return, and components is a dict of raw
    sub-component values (ttc is None, not inf, when there's no lead
    vehicle or ego isn't closing -- json.dumps can't serialise inf/nan).
    """
    scaling = hp["reward_scaling"]

    r_comfort = compute_comfort(j_lon=j_lon, j_lat=j_lat, sigma_j_sq=scaling["sigma_j_sq"])
    r_progress = compute_progress(v_ego=v_ego, v_des=v_des, beta=beta)

    r_dev, r_heading = _lateral_components(
        d_lat=d_lat, delta_psi=delta_psi, sigma_d=scaling["sigma_d"], phi=scaling["phi"]
    )
    r_lateral = float(r_dev * r_heading)

    ttc = compute_ttc(d_lead=d_lead, v_ego=v_ego, v_lead=v_lead, has_lead=has_lead)
    r_spacing = compute_spacing(ttc=ttc, tau=scaling["tau"])

    r_vec = np.array([r_comfort, r_progress, r_lateral, r_spacing], dtype=np.float32)
    components = {
        "jerk_lon": float(j_lon),
        "jerk_lat": float(j_lat),
        "r_speed": r_progress,
        "r_dev": float(r_dev),
        "r_heading": float(r_heading),
        "ttc": float(ttc) if math.isfinite(ttc) else None,
    }
    return r_vec, components
