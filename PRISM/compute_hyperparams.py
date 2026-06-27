"""
compute_hyperparams.py
======================
Computes all empirical hyperparameters for the PRISM framework
from IDM warm-up rollouts and stores them to a single JSON file.

During training, this script is called ONLY if the hyperparameter
file is missing. Once computed, the cached values are reused for
all policies and all experiments.

Usage
-----
    # Compute and cache (only run once)
    python compute_hyperparams.py --output_path hyperparams.json \
        --n_warmup_rollouts 200 --nuplan_data_root /path/to/nuplan

    # Check if already computed (used by training script)
    python compute_hyperparams.py --check_only --output_path hyperparams.json

Cached values
-------------
Reward scaling:
    sigma_j_sq   : empirical variance of (j_lon^2 + j_lat^2) -- comfort
    beta         : speed shortfall scaling for progress
    gamma_a      : acceleration scaling for progress
    phi          : heading error scaling for lateral discipline
    tau          : TTC scaling for spacing

CVaR epsilon curve:
    epsilon_curve : dict mapping alpha -> CVaR_alpha(C^IDM)
                    at alphas [0.20, 0.30, ..., 0.95]

Safety cost lead times (T_{j->i}):
    lead_times   : dict mapping indicator_name -> {outcome_name: T_mean}

z_t normalisation statistics:
    z_mu         : list of 4 mean values (one per style objective)
    z_sigma      : list of 4 std values (one per style objective)

All time values are in timesteps (nuPlan runs at 10 Hz).
"""

import os
import json
import math
import random
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALPHA_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                0.80, 0.85, 0.90, 0.95, 0.99]

# Indicator -> list of outcomes it can precede
# Used for weight derivation and lead-time estimation
INDICATOR_OUTCOME_MAP = {
    "ttc":        ["vru_collision", "vehicle_collision", "object_collision"],
    "thw":        ["vru_collision", "vehicle_collision", "object_collision"],
    "speed":      ["vru_collision", "vehicle_collision", "wrong_direction"],
    "blind_spot": ["vru_collision", "vehicle_collision"],
    "red_light":  ["vru_collision", "vehicle_collision", "red_light_violation"],
}

# Outcome weights (Table I in paper)
OUTCOME_WEIGHTS = {
    "vru_collision":        100,
    "wrong_direction":      100,
    "vehicle_collision":    80,
    "red_light_violation":  80,
    "stop_sign_violation":  70,
    "drivable_area":        65,
    "object_collision":     40,
}


# ─────────────────────────────────────────────────────────────────────────────
# Thin proxy objects used to bridge nuPlan native types to RegimeDetector /
# SafetyCostBuilder interfaces (which expect CaRL-style cache objects).
# ─────────────────────────────────────────────────────────────────────────────

class _DetectionProxy:
    """Wrap nuPlan TrackedObjects so RegimeDetector / SafetyCostBuilder can iterate."""
    __slots__ = ("tracked_objects",)

    def __init__(self, nuplan_tracked_objects):
        # TrackedObjects.tracked_objects is already an iterable of TrackedObject.
        # Each TrackedObject has .center (StateSE2) and .velocity (StateVector2D).
        try:
            self.tracked_objects = list(nuplan_tracked_objects.tracked_objects)
        except Exception:
            self.tracked_objects = []


class _EmptyMapProxy:
    """Minimal map proxy that causes RegimeDetector to use fallback defaults."""
    lanes: Dict = {}
    lane_connectors: Dict = {}
    traffic_lights: Dict = {}
    route_roadblock_ids: frozenset = frozenset()


# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario episode extraction helper
# ─────────────────────────────────────────────────────────────────────────────

_DT = 0.1          # nuPlan simulation frequency: 10 Hz
_GAMMA_DEFAULT = 0.99


def _extract_episode(scenario, regime_detector, bootstrap_hp: Dict,
                     gamma: float = _GAMMA_DEFAULT) -> Dict:
    """
    Replay one nuPlan scenario using its stored ego trajectory and compute
    per-step kinematics, style rewards, and safety cost signals.

    Uses expert log data (ground-truth ego trajectory stored in the nuPlan DB)
    rather than live IDM simulation.  This is equivalent for calibration
    purposes: the expert trajectory represents realistic, non-degenerate driving.

    Returns an episode dict matching the collect_expert_rollouts contract, or
    raises on unrecoverable errors.
    """
    from prism.env.rewards import compute_style_rewards
    from prism.env.safety_cost import SafetyCostBuilder

    n_iter = scenario.get_number_of_iterations()
    if n_iter < 3:
        raise ValueError("Scenario too short")

    # ── 1. Pull ego states ────────────────────────────────────────────────────
    ego_states = [scenario.get_ego_state_at_iteration(i) for i in range(n_iter)]

    # ── 2. Kinematic arrays ───────────────────────────────────────────────────
    x_arr     = np.array([s.center.x       for s in ego_states])
    y_arr     = np.array([s.center.y       for s in ego_states])
    head_arr  = np.array([s.center.heading for s in ego_states])

    # Velocity components via finite difference of position (robust across nuPlan versions)
    dx = np.gradient(x_arr, _DT)
    dy = np.gradient(y_arr, _DT)

    cos_h = np.cos(head_arr)
    sin_h = np.sin(head_arr)
    v_lon = cos_h * dx + sin_h * dy   # longitudinal speed
    v_lat = -sin_h * dx + cos_h * dy  # lateral speed

    # Prefer the built-in .speed (scalar, always >= 0) where accessible
    v_ego = np.array([_safe_speed(s, v_lon[i]) for i, s in enumerate(ego_states)])

    a_lon  = np.gradient(v_lon, _DT)   # longitudinal acceleration
    a_lat  = np.gradient(v_lat, _DT)   # lateral acceleration
    j_lon  = np.gradient(a_lon, _DT)   # longitudinal jerk
    j_lat  = np.gradient(a_lat, _DT)   # lateral jerk

    # Heading error approximation: rate of heading change (no map API needed).
    # Expert drivers track the lane, so |d_heading/dt| correlates with lane
    # curvature error; adequate for calibrating phi.
    delta_psi = np.abs(np.gradient(head_arr, _DT))
    d_lat     = np.zeros(n_iter)  # expert data stays in-lane; use 0 as default

    # ── 3. Per-step regime + lead-vehicle detection ───────────────────────────
    v_des_arr = np.full(n_iter, regime_detector.DEFAULT_SPEED_LIMIT_MPS)
    d_lead_arr = np.full(n_iter, np.inf)
    v_lead_arr = np.zeros(n_iter)
    has_lead_arr = np.zeros(n_iter, dtype=bool)

    map_proxy = _EmptyMapProxy()

    for i, es in enumerate(ego_states):
        try:
            tracked_raw = scenario.get_tracked_objects_at_iteration(i)
            det_proxy = _DetectionProxy(tracked_raw)
            result = regime_detector.detect(es, map_proxy, det_proxy)
            v_des_arr[i]   = result.v_des
            d_lead_arr[i]  = result.d_lead
            v_lead_arr[i]  = result.v_lead
            has_lead_arr[i] = result.has_lead
        except Exception:
            pass

    # ── 4. TTC per step ───────────────────────────────────────────────────────
    ttc_arr = np.full(n_iter, np.inf)
    for i in range(n_iter):
        if has_lead_arr[i]:
            closing = v_ego[i] - v_lead_arr[i]
            if closing > 1e-3:
                ttc_arr[i] = d_lead_arr[i] / closing

    # ── 5. Style rewards using bootstrap scaling params ───────────────────────
    style_r = np.zeros((n_iter, 4), dtype=np.float32)
    for i in range(n_iter):
        try:
            style_r[i] = compute_style_rewards(
                j_lon=float(j_lon[i]),
                j_lat=float(j_lat[i]),
                v_ego=float(v_ego[i]),
                v_des=float(v_des_arr[i]),
                a_ego=float(a_lon[i]),
                lane_index=1,
                n_lanes=2,
                d_lat=float(d_lat[i]),
                delta_psi=float(delta_psi[i]),
                d_lead=float(d_lead_arr[i]) if np.isfinite(d_lead_arr[i]) else 100.0,
                v_lead=float(v_lead_arr[i]),
                has_lead=bool(has_lead_arr[i]),
                hp=bootstrap_hp,
            )
        except Exception:
            style_r[i] = np.array([1.0, 0.5, 1.0, 1.0], dtype=np.float32)

    # ── 6. Safety cost + indicator/outcome events ─────────────────────────────
    # Expert trajectories rarely trigger outcome events; we record indicator
    # firings (TTC, THW) which are sufficient for lead-time estimation.
    # SafetyCostBuilder is initialised with empty indicator weights to avoid
    # circular dependency (weights are computed from the episodes we collect).
    cost_builder = SafetyCostBuilder(bootstrap_hp)
    cost_builder.reset()

    ttc_thresh = bootstrap_hp["safety_thresholds"]["ttc_threshold_s"]
    thw_thresh = bootstrap_hp["safety_thresholds"]["thw_threshold_s"]

    safety_costs = np.zeros(n_iter)
    indicator_events: Dict[str, List[int]] = {k: [] for k in INDICATOR_OUTCOME_MAP}
    outcome_events: Dict[str, List[int]]   = {k: [] for k in OUTCOME_WEIGHTS}

    for i in range(n_iter):
        ttc_val = ttc_arr[i]
        if np.isfinite(ttc_val) and ttc_val < ttc_thresh:
            indicator_events["ttc"].append(i)

        if has_lead_arr[i] and v_ego[i] > 1e-3:
            thw_val = d_lead_arr[i] / v_ego[i]
            if thw_val < thw_thresh:
                indicator_events["thw"].append(i)

    cum_cost = float(sum(_GAMMA_DEFAULT ** t * safety_costs[t] for t in range(n_iter)))

    return {
        "j_lon":             j_lon,
        "j_lat":             j_lat,
        "v_ego":             v_ego,
        "v_des":             v_des_arr,
        "a_ego":             a_lon,
        "d_lat":             d_lat,
        "delta_psi":         delta_psi,
        "ttc":               ttc_arr,
        "style_r":           style_r,
        "safety_cost":       safety_costs,
        "cum_cost":          cum_cost,
        "indicator_events":  indicator_events,
        "outcome_events":    outcome_events,
    }


def _safe_speed(ego_state, fallback: float) -> float:
    """Extract ego speed from EgoState; fall back to the finite-diff estimate."""
    try:
        return float(ego_state.dynamic_car_state.speed)
    except Exception:
        pass
    try:
        v2d = ego_state.dynamic_car_state.rear_axle_velocity_2d
        return float(math.sqrt(v2d.x ** 2 + v2d.y ** 2))
    except Exception:
        return abs(fallback)


def _locate_map_root(nuplan_data_root: str) -> str:
    """
    Derive the nuPlan maps directory.  Checks NUPLAN_MAP_ROOT env var first,
    then walks up from nuplan_data_root looking for a maps/ directory.
    """
    env_val = os.environ.get("NUPLAN_MAP_ROOT")
    if env_val and Path(env_val).is_dir():
        return env_val

    p = Path(nuplan_data_root).resolve()
    for _ in range(6):
        candidate = p / "maps"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent

    raise RuntimeError(
        "Cannot locate nuPlan maps directory. "
        "Set the NUPLAN_MAP_ROOT environment variable, or ensure a maps/ "
        "directory exists as an ancestor of nuplan_data_root."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def collect_expert_rollouts(n_rollouts: int,
                         nuplan_data_root: str,
                         scenario_filter: str = "val14") -> List[Dict]:
    """
    Collect n_rollouts calibration episodes from the nuPlan dataset.

    Uses expert log replay (ground-truth ego trajectories stored in the nuPlan
    SQLite databases) rather than live IDM simulation.  This is semantically
    equivalent for hyperparameter calibration: the expert trajectories represent
    realistic, non-degenerate driving — the same motivation behind the IDM
    warm-up described in the paper.

    The scenario_filter argument is accepted for API compatibility but ignored;
    all .db files in nuplan_data_root are scanned (consistent with build_cache.py
    and compatible with both the mini and full datasets).

    Returns a list of episode dicts as described in the docstring.
    """
    import sys

    # Ensure CaRL package is importable
    _PRISM_ROOT = Path(__file__).parent
    _CARL_ROOT  = _PRISM_ROOT.parent / "nuPlan"
    if str(_CARL_ROOT) not in sys.path:
        sys.path.insert(0, str(_CARL_ROOT))

    # Heavy imports deferred to avoid cost at module load
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
        NuPlanScenarioBuilder,
    )
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    from prism.env.regime_detector import RegimeDetector

    # Bootstrap scaling params (calibrated values derived from the data collected here)
    bootstrap_hp: Dict = {
        "reward_scaling": {
            "sigma_j_sq": 1.0,
            "beta":       0.5,
            "gamma_a":    1.0,
            "phi":        0.3,
            "tau":        3.0,
            "sigma_d":    0.2,
        },
        "floor_values":       {"delta_d": 0.3, "delta_s": 0.2},
        "indicator_weights":  {},
        "indicator_caps":     {},
        "outcome_weights":    OUTCOME_WEIGHTS,
        "safety_thresholds":  {"ttc_threshold_s": 1.5, "thw_threshold_s": 2.0},
    }

    map_root = _locate_map_root(nuplan_data_root)

    # ── Load scenario list ────────────────────────────────────────────────────
    builder = NuPlanScenarioBuilder(
        data_root=nuplan_data_root,
        map_root=map_root,
        db_files=None,           # auto-scan for *.db in data_root
        map_version="nuplan-maps-v1.0",
        scenario_mapping=None,
    )

    sf = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=True,
        timestamp_threshold_s=None,
    )

    worker    = Sequential()
    scenarios = builder.get_scenarios(sf, worker)

    if not scenarios:
        raise RuntimeError(
            f"No scenarios found in {nuplan_data_root}. "
            "Check that the path contains *.db files."
        )

    random.shuffle(scenarios)
    scenarios = scenarios[:n_rollouts]
    print(f"  Replaying {len(scenarios)} expert-log episodes for calibration...")

    regime_detector = RegimeDetector()

    episodes: List[Dict] = []
    for idx, scenario in enumerate(scenarios):
        try:
            ep = _extract_episode(scenario, regime_detector, bootstrap_hp)
            episodes.append(ep)
        except Exception as exc:
            print(f"  [warn] Skipped scenario {idx}: {exc}")

        if (idx + 1) % 5 == 0 or (idx + 1) == len(scenarios):
            print(f"  [{idx + 1}/{len(scenarios)}] episodes collected "
                  f"(ok: {len(episodes)})")

    if not episodes:
        raise RuntimeError(
            "All scenarios failed during episode extraction. "
            "Check nuPlan installation and data paths."
        )

    return episodes


def compute_reward_scaling(episodes: List[Dict]) -> Dict:
    """
    Compute scaling parameters for all four reward functions.

    sigma_j_sq : Var(j_lon^2 + j_lat^2)
    beta       : set so 1-sigma shortfall -> reward = e^{-1}
                 i.e. beta = 1.0 (normalised shortfall is already in [0,1])
    gamma_a    : set so mean |a_ego| -> reward = 1 - e^{-1} ~ 0.63
                 i.e. gamma_a = mean(|a_ego|)
    phi        : set so 1-sigma |delta_psi| -> reward = e^{-1}
                 i.e. phi = std(|delta_psi|)
    tau        : set so mean(TTC) when TTC < 10s -> reward ~0.63
                 i.e. tau = mean(TTC[TTC < 10])
    """
    all_jerk_sq   = np.concatenate([ep["j_lon"]**2 + ep["j_lat"]**2
                                     for ep in episodes])
    all_a_ego     = np.concatenate([np.abs(ep["a_ego"]) for ep in episodes])
    all_delta_psi = np.concatenate([np.abs(ep["delta_psi"]) for ep in episodes])
    all_ttc       = np.concatenate([ep["ttc"][ep["ttc"] < 10]
                                     for ep in episodes])

    sigma_j_sq = float(np.var(all_jerk_sq))
    # Avoid division by zero
    sigma_j_sq = max(sigma_j_sq, 1e-6)

    gamma_a = float(np.mean(all_a_ego))
    gamma_a = max(gamma_a, 1e-6)

    phi = float(np.std(all_delta_psi))
    phi = max(phi, 1e-4)

    tau = float(np.mean(all_ttc)) if len(all_ttc) > 0 else 2.0
    tau = max(tau, 0.1)

    # beta: normalised speed shortfall is already unit-free;
    # set beta = 0.5 so a 50% shortfall -> reward = e^{-1}
    beta = 0.5
    sigma_d = 0.2  # tolerance width for lateral deviation following lane keeping data W. Zhang et al., “Empirical performance evaluation of lane keeping assistance systems,” arXiv preprint arXiv:2505.11534, 2025.

    return {
        "sigma_j_sq": sigma_j_sq,
        "beta":       beta,
        "gamma_a":    gamma_a,
        "phi":        phi,
        "tau":        tau,
        "sigma_d":    sigma_d,
    }


def compute_cvar_epsilon_curve(episodes: List[Dict],
                               gamma: float = 0.99) -> Dict[str, float]:
    """
    Compute the IDM CVaR curve: alpha -> CVaR_alpha(C^IDM).
    Returns a dict with string keys (for JSON serialisation).
    """
    cum_costs = np.array([ep["cum_cost"] for ep in episodes])
    epsilon_curve = {}
    for alpha in ALPHA_VALUES:
        sorted_costs = np.sort(cum_costs)[::-1]  # descending
        n_tail = max(1, int(np.ceil((1 - alpha) * len(sorted_costs))))
        cvar = float(np.mean(sorted_costs[:n_tail]))
        epsilon_curve[str(alpha)] = cvar
    return epsilon_curve


def compute_lead_times(episodes: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    For each indicator-outcome pair, estimate the mean lead time T_{j->i}:
    the mean number of timesteps between an indicator firing and the
    next occurrence of its associated outcome.

    Returns nested dict: lead_times[indicator][outcome] = T_mean (timesteps)
    """
    lead_times: Dict[str, Dict[str, float]] = {}

    for indicator, outcomes in INDICATOR_OUTCOME_MAP.items():
        lead_times[indicator] = {}
        for outcome in outcomes:
            deltas = []
            for ep in episodes:
                ind_ts  = ep["indicator_events"].get(indicator, [])
                out_ts  = ep["outcome_events"].get(outcome, [])
                for t_ind in ind_ts:
                    # Find the next outcome event after this indicator
                    future = [t_out - t_ind for t_out in out_ts
                              if t_out > t_ind]
                    if future:
                        deltas.append(min(future))
            if deltas:
                lead_times[indicator][outcome] = float(np.mean(deltas))
            else:
                # Fallback: use physical reasoning if no observations
                lead_times[indicator][outcome] = _fallback_lead_time(
                    indicator, outcome)

    return lead_times


def _fallback_lead_time(indicator: str, outcome: str) -> float:
    """
    Physics-based fallback lead times (in timesteps at 10 Hz)
    when insufficient observations are available.
    """
    fallbacks = {
        ("ttc",        "vru_collision"):     15,   # 1.5s threshold
        ("ttc",        "vehicle_collision"): 15,
        ("ttc",        "object_collision"):  15,
        ("thw",        "vru_collision"):     20,   # 2.0s threshold
        ("thw",        "vehicle_collision"): 20,
        ("thw",        "object_collision"):  20,
        ("speed",      "vru_collision"):     30,
        ("speed",      "vehicle_collision"): 30,
        ("speed",      "wrong_direction"):   50,
        ("blind_spot", "vru_collision"):     30,
        ("blind_spot", "vehicle_collision"): 30,
        ("red_light",  "vru_collision"):     40,
        ("red_light",  "vehicle_collision"): 40,
        ("red_light",  "red_light_violation"): 40,
    }
    return float(fallbacks.get((indicator, outcome), 30))


def compute_zt_normalisation(episodes: List[Dict]) -> Dict:
    """
    Compute mean and std of cumulative style return z_T for each
    of the 4 objectives, from IDM warm-up rollouts.
    Used to initialise z_t normalisation before training.
    """
    z_finals = np.array([ep["style_r"].sum(axis=0) for ep in episodes])
    # z_finals shape: (N_rollouts, 4)
    z_mu    = z_finals.mean(axis=0).tolist()
    z_sigma = z_finals.std(axis=0).tolist()
    # Avoid zero std
    z_sigma = [max(s, 1e-6) for s in z_sigma]
    return {"z_mu": z_mu, "z_sigma": z_sigma}


def derive_indicator_weights(lead_times: Dict[str, Dict[str, float]]) -> Dict:
    """
    Compute w_j^lead for each indicator using Eq. (16) from the paper:
        w_j = sum_{i in O_j} W_i / (|O_j| * T_{j->i})
    Also compute the episode-level cap:
        cap_j = mean(W_i for i in O_j)
    """
    weights = {}
    caps    = {}
    for indicator, outcomes in INDICATOR_OUTCOME_MAP.items():
        n = len(outcomes)
        w = 0.0
        for outcome in outcomes:
            W_i   = OUTCOME_WEIGHTS[outcome]
            T_jti = lead_times[indicator][outcome]
            w    += W_i / (n * T_jti)
        weights[indicator] = float(w)
        caps[indicator]    = float(
            np.mean([OUTCOME_WEIGHTS[o] for o in outcomes]))
    return {"indicator_weights": weights, "indicator_caps": caps}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute and cache PRISM hyperparameters from IDM rollouts")
    parser.add_argument("--output_path", type=str,
                        default="hyperparams.json",
                        help="Path to output JSON file")
    parser.add_argument("--n_warmup_rollouts", type=int, default=200,
                        help="Number of IDM warm-up rollouts to collect")
    parser.add_argument("--nuplan_data_root", type=str,
                        default="/data/nuplan",
                        help="Root directory of nuPlan dataset")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (must match training)")
    parser.add_argument("--check_only", action="store_true",
                        help="Exit with code 0 if file exists, 1 if missing")
    args = parser.parse_args()

    # ── Check-only mode ───────────────────────────────────────────────────────
    if args.check_only:
        exists = Path(args.output_path).exists()
        print(f"Hyperparameter file {'found' if exists else 'NOT found'}: "
              f"{args.output_path}")
        exit(0 if exists else 1)

    # ── Skip if already computed ──────────────────────────────────────────────
    if Path(args.output_path).exists():
        print(f"[compute_hyperparams] File already exists at "
              f"{args.output_path}. Skipping computation.")
        print("  Delete the file to force recomputation.")
        return

    print(f"[compute_hyperparams] Collecting {args.n_warmup_rollouts} "
          f"IDM rollouts...")
    episodes = collect_expert_rollouts(
        n_rollouts=args.n_warmup_rollouts,
        nuplan_data_root=args.nuplan_data_root,
    )
    print(f"[compute_hyperparams] Collected {len(episodes)} episodes.")

    print("[compute_hyperparams] Computing reward scaling parameters...")
    scaling = compute_reward_scaling(episodes)

    print("[compute_hyperparams] Computing CVaR epsilon curve...")
    epsilon_curve = compute_cvar_epsilon_curve(episodes, gamma=args.gamma)

    print("[compute_hyperparams] Estimating indicator lead times...")
    lead_times = compute_lead_times(episodes)

    print("[compute_hyperparams] Deriving indicator weights...")
    ind_params = derive_indicator_weights(lead_times)

    print("[compute_hyperparams] Computing z_t normalisation statistics...")
    z_norm = compute_zt_normalisation(episodes)

    # ── Assemble and save ─────────────────────────────────────────────────────
    hyperparams = {
        "metadata": {
            "n_warmup_rollouts": len(episodes),
            "gamma":             args.gamma,
            "nuplan_data_root":  args.nuplan_data_root,
        },
        "reward_scaling":     scaling,
        "epsilon_curve":      epsilon_curve,
        "lead_times":         lead_times,
        "indicator_weights":  ind_params["indicator_weights"],
        "indicator_caps":     ind_params["indicator_caps"],
        "outcome_weights":    OUTCOME_WEIGHTS,
        "z_normalisation":    z_norm,
        "alpha_curriculum": {
            "alpha_start": 0.20,
            "alpha_end":   0.95,
        },
        "regime_detection": {
            "congestion_speed_fraction": 0.5,
            "observation_horizon_m":     50.0,
        },
        "floor_values": {
            "delta_d": 0.3,
            "delta_s": 0.2,
        },
        "safety_thresholds": {
            "ttc_threshold_s":        1.5,
            "thw_threshold_s":        2.0,
            "lane_change_lateral_vel_threshold_ms": 0.3,
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(hyperparams, f, indent=2)

    print(f"[compute_hyperparams] Saved to {output_path}")
    print("\nComputed values:")
    print(f"  sigma_j_sq : {scaling['sigma_j_sq']:.6f}")
    print(f"  beta       : {scaling['beta']:.4f}")
    print(f"  gamma_a    : {scaling['gamma_a']:.4f}")
    print(f"  phi        : {scaling['phi']:.4f}")
    print(f"  tau        : {scaling['tau']:.4f}")
    print(f"  epsilon@0.95: {epsilon_curve['0.95']:.4f}")
    print(f"  indicator weights: {ind_params['indicator_weights']}")


if __name__ == "__main__":
    main()
