"""
PRISM training entry point.

Two-stage training:
    Stage 1 — Utility function initialisation (preference-based or DPMORL diversity).
    Stage 2 — Per-policy PPO training with CVaR Lagrangian constraint.

Usage
-----
    python scripts/train.py \\
        --config configs/prism_default.yaml \\
        --hyperparams hyperparams.json \\
        --nuplan_data_root /data/nuplan \\
        --n_policies 5 \\
        --total_timesteps 10000000 \\
        --output_dir runs/prism_run_001

Pre-requisite: hyperparams.json must exist (run compute_hyperparams.py first).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

# ── Path setup so scripts can import from the PRISM package ──────────────────
_PRISM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PRISM_ROOT))
sys.path.insert(0, str(_PRISM_ROOT.parent / "nuPlan"))

import numpy as np
import torch
import yaml

from prism.morl.utility_functions import (
    UtilityFunction,
    LinearProjectionUtility,
    init_utility_functions_from_preferences,
    train_utility_functions_stage1,
)
from prism.morl.dpmorl_trainer import DPMORLTrainer
from prism.utils.hyperparams import load_hyperparams
from prism.utils.zt_normaliser import ZtNormaliser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("prism.train")


# ─────────────────────────────────────────────────────────────────────────────
# Guard: hyperparams.json must exist
# ─────────────────────────────────────────────────────────────────────────────

def _require_hyperparams(path: str) -> None:
    result = subprocess.run(
        [sys.executable, str(_PRISM_ROOT / "compute_hyperparams.py"),
         "--check_only", "--output_path", path],
        capture_output=True,
    )
    if result.returncode != 0:
        print(
            f"\n[ERROR] hyperparams.json not found at '{path}'.\n"
            "Run compute_hyperparams.py first:\n\n"
            "  python compute_hyperparams.py \\\n"
            "      --output_path hyperparams.json \\\n"
            "      --n_warmup_rollouts 200 \\\n"
            "      --nuplan_data_root /path/to/nuplan\n"
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_env(cfg: dict, hp: dict, utility_fn, zt_normaliser):
    """
    Build a PRISMEnv by directly instantiating CaRL components.
    No Hydra / OmegaConf builders needed.

    Requires:
      cfg["cache_path"]  — path to the CaRL scenario cache directory
      cfg["log_names"]   — list of log names to sample (or [] for whole cache)

    Trajectory builder selection:
      carl_ppo  → ActionTrajectoryBuilder  (2D accel + steering)
      alpamayo  → UnicycleTrajectoryBuilder (128D flattened 64-waypoint trajectory)
    """
    from carl_nuplan.planning.gym.environment.scenario_sampler.cache_scenario_sampler import (
        CacheScenarioSampler,
    )
    from carl_nuplan.planning.gym.environment.simulation_builder.default_simulation_builder import (
        DefaultSimulationBuilder,
    )
    from carl_nuplan.planning.gym.environment.observation_builder.default.default_observation_builder import (
        DefaultObservationBuilder,
    )
    from carl_nuplan.planning.gym.environment.observation_builder.default.default_renderer import (
        DefaultRenderer,
    )
    from carl_nuplan.planning.gym.environment.helper.environment_area import (
        RectangleEnvironmentArea,
    )
    from prism.env.nuplan_env import make_prism_env

    cache_path = cfg["cache_path"]
    log_names = cfg.get("log_names", [])
    backend = cfg.get("model_backend", "carl_ppo")

    # ── Trajectory builder — backend-specific ─────────────────────────────────
    if backend == "alpamayo":
        from prism.env.unicycle_trajectory_builder import UnicycleTrajectoryBuilder
        alpamayo_cfg = cfg.get("alpamayo", {})
        trajectory_builder = UnicycleTrajectoryBuilder(
            n_waypoints=alpamayo_cfg.get("n_waypoints", 64),
            dt=0.1,
            accel_bounds=tuple(alpamayo_cfg.get("accel_bounds", [-9.8, 9.8])),
            curvature_bounds=tuple(alpamayo_cfg.get("curvature_bounds", [-0.33, 0.33])),
        )
        action_space_dim = cfg.get("action_space_dim", 128)
    else:
        from carl_nuplan.planning.gym.environment.trajectory_builder.action_trajectory_builder import (
            ActionTrajectoryBuilder,
        )
        trajectory_builder = ActionTrajectoryBuilder()
        action_space_dim = cfg.get("action_space_dim", 2)

    environment_area = RectangleEnvironmentArea(
        front=78.0, back=50.0, left=64.0, right=64.0
    )

    # ── Observation builder — backend-specific ────────────────────────────────
    camera_builder = None  # set below for alpamayo; set_env() called after env creation
    if backend == "alpamayo":
        from prism.observations.camera_observation_builder import CameraObservationBuilder
        obs_cfg = cfg.get("alpamayo_obs", {})
        camera_builder = CameraObservationBuilder(
            camera_channels=obs_cfg.get("camera_channels", ["CAM_F0", "CAM_L0", "CAM_R0"]),
            num_frames=obs_cfg.get("num_frames", 4),
            image_height=obs_cfg.get("image_height", 320),
            image_width=obs_cfg.get("image_width", 576),
            sensor_root_env_var=obs_cfg.get("sensor_root_env_var", "NUPLAN_SENSOR_ROOT"),
            action_space_dim=action_space_dim,
        )
        observation_builder = camera_builder
    else:
        renderer = DefaultRenderer(
            environment_area=environment_area,
            pixel_per_meter=2.0,
            max_vehicle_speed=30.0,
            max_pedestrian_speed=4.0,
            vehicle_scaling=1.0,
            pedestrian_scaling=1.0,
            static_scaling=1.0,
            include_speed_line=False,
            lane_connector_route=False,
        )
        observation_builder = DefaultObservationBuilder(
            environment_area=environment_area,
            renderer=renderer,
            obs_num_measurements=10,
            num_value_measurements=4,
            action_space_dim=action_space_dim,
            inference=False,
        )

    scenario_sampler = CacheScenarioSampler(
        log_names=log_names if log_names else [""],
        cache_path=cache_path,
        format="gz",
        ignore_log_names=(not log_names),
    )
    simulation_builder = DefaultSimulationBuilder(agent_type="tracks")

    env = make_prism_env(
        hp=hp,
        scenario_sampler=scenario_sampler,
        simulation_builder=simulation_builder,
        trajectory_builder=trajectory_builder,
        observation_builder=observation_builder,
        environment_area=environment_area,
        utility_fn=utility_fn.as_callable(),
        gamma=cfg.get("gamma", 0.99),
        zt_normaliser=zt_normaliser,
        terminate_on_failure=False,
        outcome_costs_enabled=cfg.get("outcome_costs_enabled", True),
        active_indicators=cfg.get("active_indicators", None),
        cost_scale=cfg.get("cost_scale", 1.0),
        log_components=cfg.get("log_reward_components", False),
    )

    # Inject env back-reference so the camera builder can resolve the log file path
    # from the current scenario at each step.
    if camera_builder is not None:
        camera_builder.set_env(env)

    return env


# ─────────────────────────────────────────────────────────────────────────────
# Preference vector helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_preference_vectors(n_policies: int, reward_dim: int) -> list:
    """
    Return the default preference vectors for K policies.

    Mirrors the defaults inside init_utility_functions_from_preferences so
    Stage 2 can retrieve w_k for each policy without re-running Stage 1.
    """
    if n_policies == 5 and reward_dim == 4:
        return [
            [0.55, 0.15, 0.15, 0.15],  # comfort
            [0.15, 0.55, 0.15, 0.15],  # progress
            [0.15, 0.15, 0.55, 0.15],  # lateral discipline
            [0.15, 0.15, 0.15, 0.55],  # spacing
            [0.25, 0.25, 0.25, 0.25],  # balanced
        ]
    # No separate n_policies==4 case here anymore -- the generic formula
    # below now reproduces the curated 4-policy list above exactly (all
    # four style extremes, same 0.55/0.15 concentration), so the two would
    # otherwise be duplicate logic. It's still a SPECIAL CASE relative to
    # n_policies==5, which the generic formula can't reproduce: policy
    # index 4 would wrap around (4 % reward_dim == 0) and duplicate
    # policy 0's vector instead of giving a genuinely "balanced" 5th
    # policy, so that case is kept above.

    # own:other concentration ratio matches the curated lists above
    # (0.55:0.15 = 3.67), not a weaker ratio -- this used to be
    # base=1/reward_dim with a +=0.4*(1-base) boost then renormalise,
    # which produces only a ~2.2 ratio (e.g. 0.4231/0.1923 for K=2). Ratio
    # is scale-invariant to renormalisation, so that weaker ratio directly
    # meant weaker COSINE SEPARATION between different policies' vectors
    # regardless of normalisation -- confirmed via
    # scripts/check_gradient_alignment.py against a real K=2 run: nabla
    # f_0(z) and nabla f_1(z) had cosine similarity ~0.81, essentially
    # constant across 2000 real z points sampled, matching the old
    # formula's vectors' own cosine similarity (0.8164) almost exactly.
    # This fix's vectors have cosine similarity 0.5676 for K=2 -- a real,
    # not cosmetic, improvement (not zero: two non-negative, fixed-sum,
    # non-sparse vectors can't be fully orthogonal; that would need
    # one-hot vectors, i.e. LinearProjectionUtility's ablation). See
    # CHANGES.md.
    #
    # Pre-existing, still-unfixed limitation kept as-is: for n_policies=3,
    # dimension 3 is never emphasized (k % reward_dim only reaches 0,1,2).
    low = 0.15
    high = 1.0 - (reward_dim - 1) * low
    assert high > low, (
        f"reward_dim={reward_dim} makes high={high:.3f} <= low={low} -- "
        f"the 0.55/0.15-style concentration only makes sense while "
        f"(reward_dim - 1) * {low} < {1.0 - low:.2f} (reward_dim <= 6 for "
        f"low=0.15)."
    )
    vecs = []
    for k in range(n_policies):
        vec = [low] * reward_dim
        vec[k % reward_dim] = high
        vecs.append(vec)
    return vecs


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory — returns a PRISMPolicyBase subclass for the chosen backend
# ─────────────────────────────────────────────────────────────────────────────

def _build_agent(
    env,
    cfg: dict,
    device: torch.device,
    policy_id: int = 0,
    backbone_phase: str = "a",
):
    """
    Instantiate the policy adapter for the backend named in cfg["model_backend"].

    backbone_phase — "a" (frozen backbone, default) or "b" (LoRA active).
    Passed through so the adapter knows which mode to operate in.
    Problem 4 will use this to toggle LoRA on/off inside AlpamayoAdapter.

    Returns a PRISMPolicyBase instance on the correct device.  Adding a new
    backbone means adding one branch here and one adapter under prism/models/.
    """
    backend = cfg.get("model_backend", "carl_ppo")

    if backend == "carl_ppo":
        from carl_nuplan.planning.gym.policy.ppo.ppo_config import GlobalConfig
        from carl_nuplan.planning.gym.policy.ppo.ppo_model import PPOPolicy
        from prism.models.carl_ppo.adapter import CaRLPPOAdapter

        config = GlobalConfig()
        config.initialize(**{k: v for k, v in cfg.items() if hasattr(config, k)})
        obs_space = (
            env.single_observation_space if hasattr(env, "single_observation_space")
            else env.observation_space
        )
        act_space = (
            env.single_action_space if hasattr(env, "single_action_space")
            else env.action_space
        )
        carl_policy = PPOPolicy(obs_space, act_space, config=config).to(device)

        reward_dim = cfg.get("reward_dim", 4)
        return CaRLPPOAdapter(
            carl_policy=carl_policy,
            reward_dim=reward_dim,
            features_dim=config.features_dim,
            policy_head_arch=tuple(getattr(config, "policy_head_arch", [256, 256])),
            action_dim=int(act_space.shape[0]),
            init_log_std=cfg.get("carl_init_log_std", -0.5),
        ).to(device)

    if backend == "alpamayo":
        from prism.models.alpamayo.adapter import AlpamayoAdapter
        from prism.models.common.qformer_critic import QFormerCritic

        alpamayo_cfg = cfg.get("alpamayo", {})
        reward_dim = cfg.get("reward_dim", 4)

        critic = QFormerCritic(
            backbone_hidden_dim=alpamayo_cfg.get("backbone_hidden_dim", 4096),
            n_queries=alpamayo_cfg.get("critic_n_queries", 8),
            query_dim=alpamayo_cfg.get("critic_query_dim", 256),
            n_heads=alpamayo_cfg.get("critic_n_heads", 8),
            style_dim=alpamayo_cfg.get("critic_style_dim", reward_dim * 2),
            value_hidden_dims=tuple(alpamayo_cfg.get("critic_value_hidden_dims", [256, 128])),
        ).to(device)

        # CVaR cost critic (v2) — separate weights, style_dim=1 (e_t alone,
        # not [w_k, z_t]) since safety is style-independent. Reads the same
        # backbone_hidden_states as `critic` above, no extra VLM forward pass.
        cost_critic = QFormerCritic(
            backbone_hidden_dim=alpamayo_cfg.get("backbone_hidden_dim", 4096),
            n_queries=alpamayo_cfg.get("cost_critic_n_queries", 8),
            query_dim=alpamayo_cfg.get("cost_critic_query_dim", 256),
            n_heads=alpamayo_cfg.get("cost_critic_n_heads", 8),
            style_dim=alpamayo_cfg.get("cost_critic_style_dim", 1),
            value_hidden_dims=tuple(
                alpamayo_cfg.get("cost_critic_value_hidden_dims", [256, 128])
            ),
        ).to(device)

        return AlpamayoAdapter(
            policy_id=policy_id,
            action_dim=cfg.get("action_space_dim", 128),
            reward_dim=reward_dim,
            init_log_std=alpamayo_cfg.get("init_log_std", -0.5),
            critic=critic,
            cost_critic=cost_critic,
            extract_layers=alpamayo_cfg.get("critic_extract_layers", [22, 29, 35]),
            backbone_model_name=alpamayo_cfg.get("backbone_model_name"),
            backbone_phase=backbone_phase,
            alpamayo_cfg=alpamayo_cfg,
            observation_mode=cfg.get("observation_mode", "camera"),
        ).to(device)

    raise ValueError(f"Unknown model_backend: {backend!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1
# ─────────────────────────────────────────────────────────────────────────────

def run_stage1(cfg: dict, hp: dict, device: torch.device):
    """
    Initialise K utility functions (Stage 1).

    Returns (utility_fns, preference_vectors).  preference_vectors is a list
    of K lists (one per policy), used by Stage 2 to set w_k on adapters that
    need explicit style conditioning (e.g. AlpamayoAdapter / QFormerCritic).

    cfg["utility_ablation"] == "linear_projection" bypasses everything below
    and returns pure f_k(z) = z[k] projections instead -- see
    LinearProjectionUtility's docstring and CHANGES.md for why (Test 2 of
    the K-policy-divergence investigation). These get saved to
    stage1/utility_fn_k.pth exactly like normal utility functions, but note
    inspect_utility_functions.py assumes the real UtilityFunction class's
    buffers (fc_in, _pref_weights, etc.) and won't produce a meaningful
    report against these -- not needed anyway, since this ablation's
    behavior is fully known by construction.
    """
    n_policies = cfg.get("n_policies", 5)
    reward_dim = cfg.get("reward_dim", 4)
    mode = cfg.get("stage1", {}).get("mode", "preferences")

    if cfg.get("utility_ablation", "none") == "linear_projection":
        assert n_policies <= reward_dim, (
            f"--utility_ablation linear_projection needs n_policies ({n_policies}) "
            f"<= reward_dim ({reward_dim}) -- one projection per style dimension, "
            f"no sensible K+1-th projection."
        )
        logger.info(
            f"[Stage 1] ABLATION: linear_projection -- bypassing Stage 1 entirely, "
            f"using f_k(z) = z[k] for k=0..{n_policies-1} (see CHANGES.md)."
        )
        utility_fns = [
            LinearProjectionUtility(reward_dim=reward_dim, dim=k).to(device)
            for k in range(n_policies)
        ]
        # One-hot preference vectors matching each projection's own
        # dimension, so w_k (FiLM actor conditioning) stays consistent with
        # what the utility function actually rewards -- avoids a mismatched
        # signal between the reward the policy trains on and the "which
        # style am I" signal the actor sees.
        preference_vectors = [
            [1.0 if i == k else 0.0 for i in range(reward_dim)]
            for k in range(n_policies)
        ]
        return utility_fns, preference_vectors

    logger.info(f"[Stage 1] mode={mode}  K={n_policies}  reward_dim={reward_dim}")

    preference_vectors = _get_preference_vectors(n_policies, reward_dim)

    if mode == "preferences":
        utility_fns = init_utility_functions_from_preferences(
            n_policies=n_policies,
            reward_dim=reward_dim,
            preference_vectors=preference_vectors,
            device=str(device),
        )
        logger.info("[Stage 1] Utility functions initialised from preference vectors.")
        return utility_fns, preference_vectors

    elif mode == "diversity":
        z_mu = hp.get("z_normalisation", {}).get("z_mu", [0.0] * reward_dim)
        z_sigma = hp.get("z_normalisation", {}).get("z_sigma", [1.0] * reward_dim)

        rng = np.random.default_rng(42)
        z_samples = rng.normal(
            loc=z_mu,
            scale=[max(s, 0.1) for s in z_sigma],
            size=(1000, reward_dim),
        ).astype(np.float32)

        utility_fns = init_utility_functions_from_preferences(
            n_policies=n_policies,
            reward_dim=reward_dim,
            preference_vectors=preference_vectors,
            device=str(device),
        )
        stage1_cfg = cfg.get("stage1", {})
        utility_fns = train_utility_functions_stage1(
            utility_fns=utility_fns,
            z_samples=z_samples,
            n_iters=stage1_cfg.get("n_iters", 2000),
            lr=stage1_cfg.get("lr", 1e-3),
            diversity_weight=stage1_cfg.get("diversity_weight", 1.0),
            device=str(device),
        )
        logger.info("[Stage 1] Utility functions trained with diversity loss.")
        return utility_fns, preference_vectors
    else:
        raise ValueError(f"Unknown stage1 mode: {mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2(
    utility_fns,
    cfg: dict,
    hp: dict,
    output_dir: Path,
    device: torch.device,
    preference_vectors: list = None,
    backbone_phase: str = "a",
    phase_a_dir: Path = None,
    policy_ids: list = None,
    warm_start_dir: Path = None,
) -> None:
    """
    Train K policies with their respective utility functions.

    policy_ids: if given, only these indices are trained IN THIS PROCESS
    (the rest of utility_fns/preference_vectors are simply skipped, not an
    error) -- lets multiple concurrent OS processes each own a disjoint
    subset for real wall-clock parallelism. None (default) trains all,
    sequentially, exactly as before.

    warm_start_dir: general, backend-agnostic checkpoint warm-start (see
    --warm_start_dir's help text) -- distinct from the Alpamayo-specific
    Phase A/B mechanism below.
    """
    n_policies = len(utility_fns)
    stage2_cfg = cfg.get("stage2", {})
    n_updates = stage2_cfg.get("n_updates", 10000)
    steps_per_update = stage2_cfg.get("steps_per_update", 512)
    lr = stage2_cfg.get("learning_rate", 3e-4)

    for k, utility_fn in enumerate(utility_fns):
        if policy_ids is not None and k not in policy_ids:
            continue
        logger.info(f"\n{'='*60}")
        logger.info(f"[Stage 2] Training policy k={k} / {n_policies - 1}  "
                    f"[backbone phase {backbone_phase.upper()}]")
        logger.info(f"{'='*60}")

        policy_dir = output_dir / f"policy_{k}"
        policy_dir.mkdir(parents=True, exist_ok=True)

        zt_normaliser = ZtNormaliser.from_hyperparams(hp, beta=0.01)

        # Build environment for this policy
        env = _build_env(
            cfg=cfg, hp=hp,
            utility_fn=utility_fn,
            zt_normaliser=zt_normaliser,
        )

        # Build agent and set preference vector w_k for this policy
        agent = _build_agent(env, cfg, device, policy_id=k,
                             backbone_phase=backbone_phase)
        if preference_vectors is not None and hasattr(agent, "set_w_k"):
            w_k = torch.tensor(preference_vectors[k], dtype=torch.float32)
            agent.set_w_k(w_k)
            logger.info(f"[Stage 2] Policy {k}: w_k = {preference_vectors[k]}")

        # Phase B: load Phase A agent checkpoint (weights only — Lagrangian
        # starts fresh so the CVaR buffer reflects Phase B experience only)
        if backbone_phase == "b" and phase_a_dir is not None:
            ckpt = phase_a_dir / f"policy_{k}" / f"policy_{k}_model_final.pth"
            if ckpt.exists():
                agent.load_state_dict(
                    torch.load(ckpt, map_location=device), strict=False
                )
                logger.info(f"[Phase B] Loaded Phase A checkpoint: {ckpt}")
            else:
                logger.warning(
                    f"[Phase B] Phase A checkpoint not found at {ckpt} — "
                    "starting Phase B from scratch for policy {k}"
                )

        # General warm-start: load a previous run's saved policy weights
        # (e.g. a DPMORL-only run, trained with beta=0) as this policy's
        # starting point. Independent of backbone_phase -- this is NOT the
        # Alpamayo Phase A/B mechanism above (that one is specific to LoRA
        # phase transitions); this is a general "start from a previously-
        # trained checkpoint" path usable for any backend/experiment. See
        # --warm_start_dir's help text and CHANGES.md.
        if warm_start_dir is not None:
            ckpt = Path(warm_start_dir) / f"policy_{k}" / f"policy_{k}_model_final.pth"
            if ckpt.exists():
                missing, unexpected = agent.load_state_dict(
                    torch.load(ckpt, map_location=device), strict=False
                )
                logger.info(f"[Warm start] Loaded checkpoint for policy {k}: {ckpt}")
                if missing or unexpected:
                    # Loud, not a debug-level note: strict=False means a
                    # genuine architecture mismatch between the warm-start
                    # source and this run's config would silently skip
                    # layers rather than error, making the warm start
                    # partially or fully inert with no other symptom.
                    logger.warning(
                        f"[Warm start] Policy {k}: state_dict mismatch -- "
                        f"missing={missing}  unexpected={unexpected}. If "
                        f"this is more than a handful of expected new keys "
                        f"(e.g. a newly-added module), the source and "
                        f"target configs' architectures likely don't match "
                        f"and the warm start may not be doing what you "
                        f"expect."
                    )
            else:
                logger.warning(
                    f"[Warm start] Checkpoint not found at {ckpt} — "
                    f"starting policy {k} from scratch instead."
                )

        optimizer = torch.optim.Adam(
            list(agent.trainable_parameters()), lr=lr, eps=1e-5
        )

        # Trainer
        trainer = DPMORLTrainer(
            policy_id=k,
            agent=agent,
            optimizer=optimizer,
            env=env,
            utility_fn=utility_fn,
            hp=hp,
            cfg={**cfg, **stage2_cfg},
            output_dir=policy_dir,
            device=device,
        )

        summary = trainer.train(
            n_updates=n_updates,
            steps_per_update=steps_per_update,
        )

        logger.info(
            f"[Stage 2] Policy {k} done.  "
            f"Final CVaR={summary['cvar_history'][-1]:.4f}  "
            f"Final var_nu={summary['var_nu_history'][-1]:.4f}  "
            f"Final total_loss={summary['total_loss_history'][-1]:.4f}"
        )

        # Save summary
        import json
        with open(policy_dir / "training_summary.json", "w") as f:
            json.dump(
                {
                    "policy_id": k,
                    "cvar_history": summary["cvar_history"],
                    "var_nu_history": summary["var_nu_history"],
                    "mu_c_history": summary["mu_c_history"],
                    "sigma_c_history": summary["sigma_c_history"],
                    "reward_loss_history": summary["reward_loss_history"],
                    "cost_penalty_history": summary["cost_penalty_history"],
                    "cost_critic_loss_history": summary["cost_critic_loss_history"],
                    "total_loss_history": summary["total_loss_history"],
                    "episode_zts": [z.tolist() for z in summary["episode_zts"]],
                },
                f,
            )

        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train PRISM K-policy MORL agent")
    parser.add_argument("--config", type=str, default="configs/prism_default.yaml")
    parser.add_argument("--hyperparams", type=str, default="hyperparams.json")
    parser.add_argument("--nuplan_data_root", type=str, default=None)
    parser.add_argument("--n_policies", type=int, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stage1_only", action="store_true",
                        help="Run Stage 1 only and exit")
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Load pre-saved utility functions and go straight to Stage 2")
    parser.add_argument("--utility_fn_dir", type=str, default=None,
                        help="Directory containing saved utility function checkpoints")
    parser.add_argument(
        "--warm_start_dir", type=str, default=None,
        help=(
            "Directory of a PREVIOUS run's output (e.g. a DPMORL-only run) to "
            "load each policy's agent weights from before Stage 2 training, "
            "instead of the from-scratch random init CLAUDE.md's Design "
            "Decision #1 otherwise specifies. Expects "
            "<warm_start_dir>/policy_{k}/policy_{k}_model_final.pth per "
            "policy k (the same layout run_stage2() already writes). "
            "Independent of "
            "--backbone-phase b (that mechanism is Alpamayo-specific, for "
            "LoRA phase transitions; this is a general, backend-agnostic "
            "warm-start path). Loaded with strict=False -- if the source and "
            "target configs' architecture hyperparameters don't match "
            "exactly, mismatched layers are silently skipped rather than "
            "erroring, which would make the warm start partially or fully "
            "inert without any error. Missing/unexpected keys are logged "
            "explicitly for this reason -- check that log line, don't just "
            "assume it worked. See CHANGES.md."
        ),
    )
    parser.add_argument(
        "--policy_ids", type=str, default=None,
        help=(
            "Comma-separated policy indices to train in THIS process (e.g. "
            "'0' or '2,3'). Omit to train all n_policies sequentially "
            "(default). Combine with --stage1_only once to save utility "
            "functions, then launch one process per policy_id concurrently "
            "(each with --skip_stage1 --utility_fn_dir pointing at that "
            "saved Stage 1 output) to train K policies at the same time on "
            "the same machine -- separate OS processes, not batched math: "
            "the actual per-update bottleneck is CPU-bound nuPlan rollout "
            "collection (the large majority of update time), not GPU compute, so this is "
            "real wall-clock parallelism where matrix-batching the network "
            "forward/backward wouldn't help. Each policy already writes to "
            "its own policy_{k}/ subdirectory, so concurrent processes "
            "never touch the same output files."
        ),
    )
    parser.add_argument("--cache_path", type=str, default=None,
                        help="Override cache_path from config (path to CaRL scenario cache)")
    parser.add_argument(
        "--log_reward_components", action="store_true",
        help=(
            "Also log raw reward sub-components (r_speed, r_dev, r_heading, "
            "jerk, ttc) plus raw kinematic/regime diagnostics (v_ego, v_des, "
            "shortfall, regime, n_surrounding_agents, beta) per update -- for "
            "scripts/analyze_reward_spread.py to check whether individual "
            "pieces are plateauing near their ceiling. Off by default (small "
            "extra per-step compute + wider metrics.jsonl records). See "
            "CHANGES.md."
        ),
    )
    parser.add_argument(
        "--utility_ablation", choices=["none", "linear_projection"], default="none",
        help=(
            "Instability-analysis ablation (Test 2 of the K-policy-divergence "
            "investigation, see CHANGES.md). 'linear_projection' bypasses Stage 1 "
            "entirely and uses f_k(z) = z[k] for policy k -- a pure one-dimension "
            "projection with maximally, unambiguously different gradients "
            "(one-hot, orthogonal everywhere, no saturation possible by "
            "construction) instead of the learned/preference-biased utility "
            "functions. Isolates whether Stage 1 utility-function construction "
            "is why policies aren't diverging more, or whether the problem is "
            "downstream (PPO, the environment, or the reward formulas "
            "themselves). Requires n_policies <= reward_dim (one projection per "
            "style dimension; there's no sensible K+1-th projection)."
        ),
    )
    # ── Backbone phase (Alpamayo experiment) ─────────────────────────────────
    parser.add_argument(
        "--model-backend", choices=["carl_ppo", "alpamayo"], default=None,
        help=(
            "Override the model_backend field in the config file. "
            "Useful for quick backend switches without swapping the whole config. "
            "Note: backend-specific hyperparams (alpamayo_obs, alpamayo.*) must "
            "still be present in the config file for the chosen backend."
        ),
    )
    parser.add_argument(
        "--backbone-phase", choices=["a", "b"], default=None,
        help=(
            "Backbone training phase for Alpamayo experiments. "
            "'a' = frozen backbone (only action head + critic train). "
            "'b' = LoRA active (requires --phase-a-dir). "
            "Omit for CaRL or non-phased experiments."
        ),
    )
    parser.add_argument(
        "--phase-a-dir", type=str, default=None,
        help=(
            "Path to a completed Phase A output directory. "
            "Required when --backbone-phase b. "
            "Example: runs/prism_alpamayo_run_001"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override config with CLI args
    if args.model_backend:
        cfg["model_backend"] = args.model_backend
    if args.nuplan_data_root:
        cfg["nuplan_data_root"] = args.nuplan_data_root
    if args.n_policies:
        cfg["n_policies"] = args.n_policies
    if args.total_timesteps:
        n_per_policy = args.total_timesteps // cfg.get("n_policies", 5)
        steps_per_update = cfg.get("stage2", {}).get("steps_per_update", 512)
        cfg.setdefault("stage2", {})["n_updates"] = n_per_policy // steps_per_update
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.cache_path:
        cfg["cache_path"] = args.cache_path
    if args.utility_ablation != "none":
        cfg["utility_ablation"] = args.utility_ablation
    if args.log_reward_components:
        cfg["log_reward_components"] = True

    # ── Backbone phase (Alpamayo only) ────────────────────────────────────────
    backbone_phase = args.backbone_phase  # "a", "b", or None
    phase_a_dir: Path = None

    if backbone_phase is not None:
        # Apply phase-specific CVaR buffer size from config
        phase_key = f"phase_{backbone_phase}"
        phase_cfg = cfg.get("backbone_phase", {}).get(phase_key, {})
        if "cvar_buffer_size" in phase_cfg:
            cfg["cvar_buffer_size"] = phase_cfg["cvar_buffer_size"]
            logger.info(
                f"[Backbone phase {backbone_phase.upper()}] "
                f"CVaR buffer size set to {cfg['cvar_buffer_size']}"
            )

        if backbone_phase == "b":
            if not args.phase_a_dir:
                print(
                    "\n[ERROR] --phase-a-dir is required when --backbone-phase b.\n"
                    "Point it at the output directory from your Phase A run.\n"
                    "Example: --phase-a-dir runs/prism_alpamayo_run_001\n"
                )
                sys.exit(1)
            phase_a_dir = Path(args.phase_a_dir)
            if not phase_a_dir.exists():
                print(f"\n[ERROR] Phase A directory not found: {phase_a_dir}\n")
                sys.exit(1)
            logger.info(f"[Phase B] Loading Phase A checkpoints from: {phase_a_dir}")

    # ── Guard: hyperparams must exist ─────────────────────────────────────────
    _require_hyperparams(args.hyperparams)

    # ── Load hyperparams ──────────────────────────────────────────────────────
    hp = load_hyperparams(args.hyperparams)

    # ── Seeding ───────────────────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Device ───────────────────────────────────────────────────────────────
    device_str = cfg.get("device", "cpu")
    if cfg.get("cuda", True) and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Device: {device}")

    output_dir = Path(cfg.get("output_dir", "runs")) / cfg.get("experiment_name", "prism")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    # Phase B skips Stage 1 automatically — utility functions are loaded from
    # the Phase A output directory so they are identical across both phases.
    n_policies = cfg.get("n_policies", 5)
    reward_dim = cfg.get("reward_dim", 4)

    if backbone_phase == "b":
        logger.info("[Stage 1] Skipped for Phase B — loading from Phase A.")
        utility_fn_source = phase_a_dir / "stage1"
        utility_fns = []
        for k in range(n_policies):
            uf = UtilityFunction(reward_dim=reward_dim)
            uf_path = utility_fn_source / f"utility_fn_{k}.pth"
            uf.load_state_dict(torch.load(uf_path, map_location=device))
            uf.to(device)
            utility_fns.append(uf)
        preference_vectors = _get_preference_vectors(n_policies, reward_dim)

    elif args.skip_stage1 and args.utility_fn_dir:
        if cfg.get("utility_ablation", "none") == "linear_projection":
            # LinearProjectionUtility has no learned state (fully determined
            # by reward_dim/dim alone) -- reconstruct directly rather than
            # trying to load its trivial checkpoint into a real
            # UtilityFunction, which would fail (missing keys) since the
            # ablation's checkpoints only ever contain the class's dummy
            # buffer. Matches run_stage1()'s own ablation branch exactly
            # (same one-hot preference_vectors), so a --stage1_only run
            # followed by concurrent --skip_stage1 --policy_ids processes
            # (see Makefile's *-parallel targets) works under this ablation
            # too. See CHANGES.md.
            assert n_policies <= reward_dim, (
                f"--utility_ablation linear_projection needs n_policies "
                f"({n_policies}) <= reward_dim ({reward_dim})."
            )
            logger.info(
                "[Stage 1] Skipped. Reconstructing linear_projection ablation "
                "utility functions directly (no learned state to load)."
            )
            utility_fns = [
                LinearProjectionUtility(reward_dim=reward_dim, dim=k).to(device)
                for k in range(n_policies)
            ]
            preference_vectors = [
                [1.0 if i == k else 0.0 for i in range(reward_dim)]
                for k in range(n_policies)
            ]
        else:
            logger.info("[Stage 1] Skipped. Loading pre-saved utility functions.")
            utility_fns = []
            for k in range(n_policies):
                uf = UtilityFunction(reward_dim=reward_dim)
                uf_path = Path(args.utility_fn_dir) / f"utility_fn_{k}.pth"
                uf.load_state_dict(torch.load(uf_path, map_location=device))
                uf.to(device)
                utility_fns.append(uf)
            preference_vectors = _get_preference_vectors(n_policies, reward_dim)

    else:
        utility_fns, preference_vectors = run_stage1(cfg, hp, device)

        # Save Stage 1 utility functions
        stage1_dir = output_dir / "stage1"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        for k, uf in enumerate(utility_fns):
            torch.save(uf.state_dict(), stage1_dir / f"utility_fn_{k}.pth")
        logger.info(f"[Stage 1] Utility functions saved to {stage1_dir}")

    if args.stage1_only:
        logger.info("[Stage 1] --stage1_only flag set. Exiting after Stage 1.")
        return

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    policy_ids = (
        [int(x) for x in args.policy_ids.split(",")] if args.policy_ids else None
    )
    run_stage2(
        utility_fns=utility_fns,
        cfg=cfg,
        hp=hp,
        output_dir=output_dir,
        device=device,
        preference_vectors=preference_vectors,
        backbone_phase=backbone_phase or "a",
        phase_a_dir=phase_a_dir,
        policy_ids=policy_ids,
        warm_start_dir=Path(args.warm_start_dir) if args.warm_start_dir else None,
    )

    logger.info(f"\nTraining complete.  Results in: {output_dir}")


if __name__ == "__main__":
    main()
