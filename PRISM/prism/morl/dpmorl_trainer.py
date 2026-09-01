"""
DPMORL Stage 2 — Per-policy PPO training with state-augmented CVaR (v2).

One DPMORLTrainer instance manages a single policy k.  The outer training
loop (scripts/train.py) creates K trainers and calls them sequentially.

Training loop (per policy):
    for update in range(n_updates):
        1. Collect rollouts with current PRISMEnv.  Reward = raw R_t (the
           DPMORL utility-difference reward).  Each step also carries the
           policy's cost-critic value V^C(s, e_t) and the raw per-step
           safety cost c_t.
        2. Compute per-episode cumulative safety costs; update the rolling
           EpisodeCostBuffer and the VaR threshold nu = quantile(costs, alpha)
        3. Compute a DENSE per-timestep cost signal c~_t via the fixed hinge
           g_nu (see cvar_penalty.py) -- nonzero at every step, not just
           episode boundaries, and well-defined even for a still-in-progress
           trailing episode (no need to mask incomplete episodes).
        4. Two parallel GAE passes: reward (unchanged) and cost (new, on
           c~_t against the cost critic).  Combine into a single advantage
           A_total = A_reward - beta * A_cost and run ONE PPO clipped-ratio
           update on it -- reward and cost share the same ratio/clip/epochs,
           so there is no REINFORCE term and no importance-ratio mismatch.
        5. Save checkpoint.

Sanity checks (see _compute_dense_costs / _ppo_update):
    A1 - per-completed-episode runtime telescoping check on c~_t (WARNING only).
    A2 - NaN/Inf checks on c~_t, A_t^C, and total_loss; training halts
         (RuntimeError) after _NAN_HALT_STREAK consecutive bad updates.

This replaces v1's unconstrained REINFORCE-style penalty (which reused the
Rockafellar-Uryasev VALUE decomposition as a policy-gradient weight -- an
envelope-theorem violation -- and had no importance-ratio correction across
PPO's epochs).  See PRISM/CHANGES.md for the full rationale and the
verification of the Muni et al. (arXiv:2602.03778) formula that motivated
switching to the g_nu telescoping construction instead.

Rollout buffer tracks:
    - obs (generic dict — whatever keys PRISMEnv returns; backend-agnostic;
      now includes "cumulative_cost" = e_t alongside "value_measurements" = z_t)
    - actions, log_probs, values, cost_values, rewards, dones
    - step_costs (raw per-step safety cost, for episode-cost/dense-signal calc)
    - episode_id per step (local index within this update's rollout, used to
      group steps by episode when reconstructing e_t sequences post-hoc)
    - dense_costs (c~_t, computed post-hoc once nu is known for this update)
    - episode_zt (for Pareto-front logging at episode end)
"""

from __future__ import annotations

import logging
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from prism.curriculum.alpha_schedule import AlphaSchedule
from prism.models.base import PRISMPolicyBase
from prism.morl.cvar_penalty import (
    EpisodeCostBuffer,
    compute_empirical_cvar,
    compute_episode_cost,
    dense_cost_signal,
    g_nu,
    update_var,
)
from prism.morl.utility_functions import UtilityFunction
from prism.utils.gpu_monitor import GPUMonitor
from prism.utils.metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)

# Style reward vector dimension names (comfort, progress, lateral, spacing --
# see prism/env/nuplan_env.py's _REWARD_DIM), used only to label the
# per-update mean-z_T fields below. Falls back to generic z_dim{i} names if
# a future config's reward_dim doesn't match len 4.
_STYLE_DIM_NAMES = ("comfort", "progress", "lateral", "spacing")

_ZERO_COST_WARN_STREAK = 10   # consecutive updates with all-zero episode cost
_TELESCOPING_REL_TOL = 1e-4   # per-episode runtime check tolerance
_NAN_HALT_STREAK = 3          # consecutive updates with NaN/Inf before halting


def _get_git_commit() -> str:
    """Best-effort short git commit hash of this checkout, for run
    provenance in the metrics config record. MetricsLogger appends rather
    than overwrites (see metrics_logger.py), so if an output dir isn't
    cleared between a `git pull` and a rerun, a single policy_k_metrics.jsonl
    can end up containing runs from two different code versions with no way
    to tell them apart after the fact -- this closes that gap."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except Exception:
        return "unknown"


_GIT_COMMIT = _get_git_commit()


@dataclass
class RolloutBuffer:
    """
    Lightweight buffer for one PPO update iteration.

    obs stores each step's full observation dict as a numpy dict,
    preserving whatever keys and dtypes PRISMEnv returns.  This makes
    the buffer backend-agnostic: CaRL obs (bev_semantics + measurements)
    and Alpamayo obs (cameras + ...) are stored identically.
    """
    obs: List[dict] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    cost_values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    # Per-step raw safety costs, for episode-cost / dense-signal computation
    step_costs: List[float] = field(default_factory=list)
    # Local episode index (within this update's buffer) for each step --
    # used to reconstruct each episode's e_t sequence when computing the
    # dense cost signal post-hoc (after nu is known for this update)
    episode_ids: List[int] = field(default_factory=list)
    # Dense per-timestep cost signal c~_t, populated post-hoc
    dense_costs: List[float] = field(default_factory=list)
    # Episode-level z_T vectors for logging
    episode_zts: List[np.ndarray] = field(default_factory=list)

    def clear(self) -> None:
        for lst in vars(self).values():
            if isinstance(lst, list):
                lst.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class DPMORLTrainer:
    """
    Manages one policy in the PRISM MORL system.

    Parameters
    ----------
    policy_id : int
        Index k ∈ {0, ..., K-1}.
    agent : PRISMPolicyBase
        Policy adapter (CaRLPPOAdapter, AlpamayoAdapter, etc.).
    optimizer : torch.optim.Optimizer
    env : PRISMEnv
        The gymnasium environment.  utility_fn is set here.
    utility_fn : UtilityFunction
        The utility function f_k for this policy.
    hp : Dict
        Loaded hyperparams.json.
    cfg : dict
        Training config (from prism_default.yaml).  Reads `beta` (fixed
        combined-advantage penalty weight), `tau` (cost-hinge smoothness),
        `cf_coef` (cost critic loss coefficient), and `cvar_buffer_size`
        (rolling episode-cost window for the VaR/nu estimate). Also reads
        `fixed_alpha` (instability-analysis ablation: constant alpha for
        the whole run, bypassing alpha_start/alpha_end/n_curriculum_iters
        entirely, when set) and `gpu_log_interval_s` (GPUMonitor sample
        interval, default 15.0s -- kept coarse deliberately: each sample
        spawns an `nvidia-smi` subprocess from a background thread, which
        at a too-short interval can noticeably load a machine used for both
        training and interactive desktop work). The env-level ablation toggles
        (`outcome_costs_enabled`, `active_indicators`, `cost_scale`) are
        consumed in scripts/train.py's _build_env(), not here -- they're
        only echoed into this trainer's metrics-log config record for
        self-describing output files.
    output_dir : Path
        Where to save checkpoints and verbose per-update metrics
        (policy_{id}_metrics.jsonl, policy_{id}_gpu.jsonl).
    device : torch.device
    """

    def __init__(
        self,
        policy_id: int,
        agent: PRISMPolicyBase,
        optimizer: torch.optim.Optimizer,
        env,                          # PRISMEnv
        utility_fn: UtilityFunction,
        hp: Dict,
        cfg: dict,
        output_dir: Path,
        device: torch.device,
    ) -> None:
        self.policy_id = policy_id
        self.agent = agent
        self.optimizer = optimizer
        # Linear LR decay to 0 over the run (standard PPO practice, e.g. the
        # original PPO paper / CleanRL's implementation) -- previously
        # absent: lr stayed at its initial value for the entire run,
        # including the final updates where large steps are least wanted.
        # A late, still-full-sized update is a plausible contributor to the
        # late-training instability seen in a real run (huge approx_kl spike
        # in the final bin). Captured here (not read fresh each update) so
        # it reflects whatever LR the optimizer was actually constructed
        # with, matching cfg["stage2"]["learning_rate"].
        self._initial_lr = optimizer.param_groups[0]["lr"]
        self._lr_decay = cfg.get("lr_decay", True)
        # Linear ent_coef decay to 0 over the run, same rationale as LR decay
        # above -- action_heads.py's own docstring already diagnoses why this
        # was needed: the entropy bonus (-ent_coef * entropy) is a constant,
        # unopposed pull toward higher log_std, while PPO's own gradient on
        # log_std is structurally close to zero (per-minibatch advantage
        # normalization + ratio~1 make it near-self-cancelling). A fixed,
        # non-zero ent_coef therefore never stops pulling entropy upward --
        # it only pulls more slowly -- which is consistent with the late-
        # training entropy upticks seen even after reducing ent_coef
        # 0.01 -> 0.001 (see CHANGES.md). Decaying it to 0 removes the
        # persistent pull entirely by the end of training instead of just
        # shrinking it.
        self._initial_ent_coef = cfg.get("ent_coef", 0.01)
        self._ent_coef_decay = cfg.get("ent_coef_decay", True)
        self.env = env
        self.utility_fn = utility_fn
        self.hp = hp
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.device = device

        self.alpha_schedule = AlphaSchedule(
            alpha_start=cfg.get("alpha_start", 0.20),
            alpha_end=cfg.get("alpha_end", 0.95),
            n_curriculum=cfg.get("n_curriculum_iters", 5000),
        )
        # Instability-analysis ablation: bypass the curriculum entirely and
        # use a single constant alpha for the whole run when set. None (the
        # default) preserves the original curriculum behavior.
        self._fixed_alpha = cfg.get("fixed_alpha", None)
        self._cost_buffer = EpisodeCostBuffer(
            buffer_size=cfg.get("cvar_buffer_size", 500)
        )
        self._beta = cfg.get("beta", 1.0)
        self._tau = cfg.get("tau", 20.0)
        # Auto-detected, not a separate config key: when beta=0 AND cf_coef=0
        # (e.g. DPMORL-only experiments), the entire cost/CVaR machinery is
        # mathematically inert -- still computed (harmless), but logging it
        # is pure noise for judging whether personalization is training.
        # Gates the cost-diagnostic fields in the per-update metrics record
        # below; the config record always logs beta/cf_coef themselves so
        # it's still visible from the log which mode a run was in.
        self._log_cost_diagnostics = self._beta != 0.0 or cfg.get("cf_coef", 0.5) != 0.0

        self._global_step = 0
        self._buffer = RolloutBuffer()

        # Sanity-check state: warn if the cost pipeline appears broken
        self._zero_cost_streak = 0
        self._last_valid_nu = 0.0
        # Consecutive updates containing a NaN/Inf in c~_t, A_t^C, or
        # total_loss. Training halts once this reaches _NAN_HALT_STREAK.
        self._nan_streak = 0

        # ── Verbose per-update metrics + GPU utilization logging ─────────
        # Written for every update (not just every 100th, unlike the INFO
        # log line below) so a full run can be analyzed offline in pandas
        # without depending on terminal scrollback.
        self._metrics = MetricsLogger(
            self.output_dir / f"policy_{policy_id}_metrics.jsonl"
        )
        gpu_index = device.index if device.type == "cuda" and device.index is not None else 0
        self._gpu_monitor = GPUMonitor(
            device_index=gpu_index,
            interval_s=cfg.get("gpu_log_interval_s", 15.0),
            log_path=self.output_dir / f"policy_{policy_id}_gpu.jsonl",
        )
        self._metrics.log_config({
            "policy_id": policy_id,
            "git_commit": _GIT_COMMIT,
            "beta": self._beta, "tau": self._tau,
            "cf_coef": cfg.get("cf_coef", 0.5), "vf_coef": cfg.get("vf_coef", 0.5),
            "ent_coef": cfg.get("ent_coef", 0.01), "ent_coef_decay": self._ent_coef_decay,
            "clip_coef": cfg.get("clip_coef", 0.2),
            "gamma": cfg.get("gamma", 0.99), "gae_lambda": cfg.get("gae_lambda", 0.95),
            "max_grad_norm": cfg.get("max_grad_norm", 0.5),
            "update_epochs": cfg.get("update_epochs", 4),
            "minibatch_size": cfg.get("minibatch_size", 64),
            "cvar_buffer_size": cfg.get("cvar_buffer_size", 500),
            "alpha_start": cfg.get("alpha_start", 0.20), "alpha_end": cfg.get("alpha_end", 0.95),
            "n_curriculum_iters": cfg.get("n_curriculum_iters", 5000),
            "fixed_alpha": self._fixed_alpha,
            # Ablation toggles live in the env/safety-cost stack, echoed here
            # (from cfg, where scripts/train.py also reads them) so this
            # file is self-describing without also needing the run's YAML.
            "outcome_costs_enabled": cfg.get("outcome_costs_enabled", True),
            "active_indicators": cfg.get("active_indicators", None),
            "cost_scale": cfg.get("cost_scale", 1.0),
        })

        # Set env utility function (no lambda -- PRISMEnv returns raw R_t)
        self.env.set_utility_fn(utility_fn.as_callable())

    def train(self, n_updates: int, steps_per_update: int = 512) -> Dict:
        """
        Run the full PPO training loop for this policy.

        Returns a summary dict with CVaR/cost diagnostics and episode z_Ts.
        """
        self.agent.train()
        summary = {
            "policy_id": self.policy_id,
            "cvar_history": [],
            "var_nu_history": [],
            "mu_c_history": [],
            "sigma_c_history": [],
            "reward_loss_history": [],
            "cost_penalty_history": [],
            "cost_critic_loss_history": [],
            "total_loss_history": [],
            "episode_zts": [],
        }

        obs, _ = self.env.reset()
        next_obs = {k: torch.from_numpy(np.array(v)).unsqueeze(0).to(self.device)
                    for k, v in obs.items()}

        self._gpu_monitor.start()
        try:
            self._train_loop(n_updates, steps_per_update, next_obs, summary)
        finally:
            self._gpu_monitor.stop()
            self._metrics.close()

        self._save_checkpoint(n_updates - 1, final=True)
        return summary

    def _train_loop(self, n_updates: int, steps_per_update: int, next_obs, summary: Dict) -> None:
        for update in range(n_updates):
            update_start = time.time()

            # Linear LR decay: 1.0 -> 0.0 fraction of initial_lr over the run.
            if self._lr_decay:
                frac = 1.0 - (update / max(n_updates, 1))
                current_lr = self._initial_lr * frac
                for pg in self.optimizer.param_groups:
                    pg["lr"] = current_lr
            else:
                current_lr = self._initial_lr

            # Linear ent_coef decay: 1.0 -> 0.0 fraction of initial_ent_coef
            # over the run (see __init__ for why).
            if self._ent_coef_decay:
                ent_frac = 1.0 - (update / max(n_updates, 1))
                current_ent_coef = self._initial_ent_coef * ent_frac
            else:
                current_ent_coef = self._initial_ent_coef

            # Guard: alpha must never reach 1.0 (defense-in-depth -- neither
            # update_var nor compute_empirical_cvar currently divides by
            # (1-alpha), but this matches the spec's explicit guard and
            # protects any future code that might).
            if self._fixed_alpha is not None:
                alpha = min(self._fixed_alpha, 0.999)
            else:
                alpha = min(self.alpha_schedule.get(update), 0.999)

            # ── Rollout collection ─────────────────────────────────────
            rollout_start = time.time()
            self._buffer.clear()
            episode_step_costs: List[float] = []
            local_episode_costs: List[float] = []
            # z_T and episode length, per completed episode THIS update --
            # the one thing needed to judge whether K policies are actually
            # learning distinguishable styles (DPMORL-only experiments) and
            # a cheap capability proxy (longer episodes ~ fewer early
            # terminations), neither of which was previously logged anywhere.
            local_episode_zts: List[np.ndarray] = []
            local_episode_lengths: List[int] = []
            # Why each episode ended THIS update -- "collision"/"off_road"/
            # "completed" (survived to scenario end / truncation). Direct
            # capability signal, distinct from length alone (a stationary
            # policy can also produce long episodes without driving well).
            local_episode_outcomes: List[str] = []
            current_ep_id = 0
            current_ep_len = 0

            for _ in range(steps_per_update):
                self._global_step += 1
                current_ep_len += 1

                with torch.no_grad():
                    out = self.agent.forward(next_obs)

                act_np = out.action.squeeze(0).cpu().numpy()
                next_obs_raw, reward, term, trunc, info = self.env.step(act_np)
                done = term or trunc

                c_t = float(info.get("safety_cost", 0.0))
                episode_step_costs.append(c_t)

                self._buffer.obs.append({
                    k: v.squeeze(0).cpu().numpy()
                    for k, v in next_obs.items()
                })
                self._buffer.actions.append(act_np)
                self._buffer.log_probs.append(float(out.log_prob.item()))
                self._buffer.values.append(float(out.value.item()))
                self._buffer.cost_values.append(float(out.cost_value.item()))
                self._buffer.rewards.append(float(reward))
                self._buffer.dones.append(done)
                self._buffer.step_costs.append(c_t)
                self._buffer.episode_ids.append(current_ep_id)

                if done:
                    # Record episode-level cumulative cost for the nu/CVaR estimate
                    ep_cost = compute_episode_cost(
                        episode_step_costs, gamma=self.cfg.get("gamma", 0.99)
                    )
                    local_episode_costs.append(ep_cost)
                    self._cost_buffer.add_episode(ep_cost)
                    episode_step_costs = []
                    current_ep_id += 1
                    local_episode_lengths.append(current_ep_len)
                    current_ep_len = 0

                    comp = info.get("safety_components")
                    if comp is not None and (comp.vru_collision or comp.vehicle_collision):
                        local_episode_outcomes.append("collision")
                    elif comp is not None and comp.off_road:
                        local_episode_outcomes.append("off_road")
                    else:
                        local_episode_outcomes.append("completed")

                    if "episode_zt" in info:
                        self._buffer.episode_zts.append(info["episode_zt"].copy())
                        summary["episode_zts"].append(info["episode_zt"].copy())
                        local_episode_zts.append(info["episode_zt"].copy())

                    obs, _ = self.env.reset()
                    next_obs = {
                        k: torch.from_numpy(np.array(v)).unsqueeze(0).to(self.device)
                        for k, v in obs.items()
                    }
                else:
                    next_obs = {
                        k: torch.from_numpy(np.array(v)).unsqueeze(0).to(self.device)
                        for k, v in next_obs_raw.items()
                    }

            rollout_end = time.time()

            # ── VaR / CVaR diagnostics + dense cost signal ──────────────
            buffer_costs = self._cost_buffer.costs
            mu_c = float(np.mean(buffer_costs)) if buffer_costs else 0.0
            sigma_c = float(np.std(buffer_costs)) if buffer_costs else 0.0
            cvar_hat = compute_empirical_cvar(buffer_costs, alpha)  # diagnostic only

            nu = update_var(buffer_costs, alpha)
            if not math.isfinite(nu):
                logger.error(
                    f"[Policy {self.policy_id}] update={update+1}: nu is "
                    f"NaN/Inf (buffer_costs may be degenerate) -- clamping "
                    f"to last valid value {self._last_valid_nu:.3f}"
                )
                nu = self._last_valid_nu
            else:
                self._last_valid_nu = nu

            # Backends that condition their actor on nu via text (Alpamayo)
            # opt in via set_nu(); CaRL has no such method and is unaffected.
            if hasattr(self.agent, "set_nu"):
                self.agent.set_nu(nu)

            if mu_c == 0.0 and buffer_costs:
                self._zero_cost_streak += 1
                if self._zero_cost_streak == _ZERO_COST_WARN_STREAK:
                    logger.warning(
                        f"[Policy {self.policy_id}] All episode costs have been "
                        f"zero for {_ZERO_COST_WARN_STREAK} consecutive updates -- "
                        f"the safety cost pipeline may be broken. Training is "
                        f"proceeding as an effectively unconstrained reward-only run."
                    )
            else:
                self._zero_cost_streak = 0

            dense_costs, dense_nan = self._compute_dense_costs(
                nu, n_local=len(local_episode_costs)
            )

            # ── PPO update ─────────────────────────────────────────────
            ppo_start = time.time()
            loss_info = self._ppo_update(next_obs, dense_costs, ent_coef=current_ent_coef)
            ppo_end = time.time()

            # A2 -- NaN/Inf halt check. c~_t (dense_nan, above), A_t^C, and
            # total_loss (both inside loss_info, from _ppo_update) each log
            # their own ERROR when they occur; here we just track whether
            # THIS update had any of them and halt after _NAN_HALT_STREAK
            # consecutive bad updates. This is deliberately more tolerant
            # than halting on the very first occurrence: a single transient
            # NaN (e.g. a rare degenerate minibatch) logs loudly and skips
            # its own optimizer step (see _ppo_update) without necessarily
            # ending a long training run, but persistent NaNs across
            # multiple consecutive updates are treated as a real failure.
            update_had_nan = dense_nan or loss_info["nan_detected"]
            if update_had_nan:
                self._nan_streak += 1
            else:
                self._nan_streak = 0

            # ── Verbose per-update metrics (every update, not just every 100th) ──
            gpu_stats = self._gpu_monitor.average_window(update_start) or {}

            # Mean z_T across episodes completed THIS update, per style
            # dimension -- the signal needed to judge whether K policies are
            # producing distinguishable driving styles (e.g. DPMORL-only
            # runs with beta=0), not previously logged anywhere.
            z_stats = {}
            if local_episode_zts:
                mean_zt = np.mean(np.stack(local_episode_zts), axis=0)
                names = (
                    _STYLE_DIM_NAMES if len(mean_zt) == len(_STYLE_DIM_NAMES)
                    else tuple(f"z_dim{i}" for i in range(len(mean_zt)))
                )
                z_stats = {f"z_{name}": float(v) for name, v in zip(names, mean_zt)}

            mean_reward_this_update = (
                float(np.mean(self._buffer.rewards)) if self._buffer.rewards else None
            )
            # Does the raw reward signal actually vary at all? A near-zero
            # std here regardless of what the mean is doing would mean
            # there's little for the policy to distinguish between actions
            # on -- separate question from whether the MEAN is trending up.
            std_reward_this_update = (
                float(np.std(self._buffer.rewards)) if self._buffer.rewards else None
            )

            # Episode outcome breakdown THIS update -- direct capability
            # signal (a stationary/timid policy can produce long episodes
            # without driving anywhere; this distinguishes "survived because
            # it drove well" from "survived because it barely moved").
            n_eps_outcome = len(local_episode_outcomes)
            outcome_stats = {
                "frac_collision": (
                    local_episode_outcomes.count("collision") / n_eps_outcome
                    if n_eps_outcome else None
                ),
                "frac_off_road": (
                    local_episode_outcomes.count("off_road") / n_eps_outcome
                    if n_eps_outcome else None
                ),
                "frac_completed": (
                    local_episode_outcomes.count("completed") / n_eps_outcome
                    if n_eps_outcome else None
                ),
            }

            record = {
                "update": update + 1,
                "policy_id": self.policy_id,
                "global_step": self._global_step,
                "n_episodes_this_update": len(local_episode_costs),
                "mean_episode_length": (
                    float(np.mean(local_episode_lengths)) if local_episode_lengths else None
                ),
                "min_episode_length": (
                    min(local_episode_lengths) if local_episode_lengths else None
                ),
                "max_episode_length": (
                    max(local_episode_lengths) if local_episode_lengths else None
                ),
                "mean_reward_this_update": mean_reward_this_update,
                "std_reward_this_update": std_reward_this_update,
                **z_stats,
                **outcome_stats,
                "total_loss": loss_info["total_loss"],
                "v_loss": loss_info["v_loss"],
                "entropy": loss_info["entropy"],
                "ppo_loss": loss_info["ppo_loss"],
                "approx_kl": loss_info["approx_kl"],
                "clip_fraction": loss_info["clip_fraction"],
                "reward_advantage_std": loss_info["reward_advantage_std"],
                "current_lr": current_lr,
                "current_ent_coef": current_ent_coef,
                "nan_detected": loss_info["nan_detected"],
                "nan_streak": self._nan_streak,
                "rollout_time_s": rollout_end - rollout_start,
                "ppo_update_time_s": ppo_end - ppo_start,
                "update_total_time_s": time.time() - update_start,
                **gpu_stats,
            }
            # Cost/CVaR diagnostics -- meaningless noise when beta=0 and
            # cf_coef=0 (see self._log_cost_diagnostics above), so omitted
            # entirely from the record rather than logged as inert zeros.
            if self._log_cost_diagnostics:
                record.update({
                    "alpha": alpha,
                    "var_nu": nu,
                    "cvar_hat": cvar_hat,
                    "mu_c": mu_c,
                    "sigma_c": sigma_c,
                    "mean_episode_cost_this_update": (
                        float(np.mean(local_episode_costs)) if local_episode_costs else None
                    ),
                    "std_episode_cost_this_update": (
                        float(np.std(local_episode_costs)) if local_episode_costs else None
                    ),
                    "reward_loss": loss_info["reward_loss"],
                    "cost_penalty": loss_info["cost_penalty"],
                    "cost_critic_loss": loss_info["cost_critic_loss"],
                    "grad_norm_total_preclip": loss_info["grad_norm_total_preclip"],
                    "grad_norm_cost_critic": loss_info["grad_norm_cost_critic"],
                    "grad_norm_other": loss_info["grad_norm_other"],
                    "dense_cost_nan": dense_nan,
                    "zero_cost_streak": self._zero_cost_streak,
                })
            self._metrics.log_update(record)

            if self._nan_streak >= _NAN_HALT_STREAK:
                logger.error(
                    f"[Policy {self.policy_id}] NaN/Inf detected in c~_t, A_t^C, "
                    f"or total_loss for {self._nan_streak} consecutive updates "
                    f"(update={update+1}/{n_updates}) -- halting training."
                )
                raise RuntimeError(
                    f"[Policy {self.policy_id}] Training halted: NaN/Inf persisted "
                    f"for {self._nan_streak} consecutive updates. See preceding "
                    f"ERROR log lines for which quantity (c~_t / A_t^C / total_loss) "
                    f"and which update."
                )

            summary["cvar_history"].append(cvar_hat)
            summary["var_nu_history"].append(nu)
            summary["mu_c_history"].append(mu_c)
            summary["sigma_c_history"].append(sigma_c)
            summary["reward_loss_history"].append(loss_info["reward_loss"])
            summary["cost_penalty_history"].append(loss_info["cost_penalty"])
            summary["cost_critic_loss_history"].append(loss_info["cost_critic_loss"])
            summary["total_loss_history"].append(loss_info["total_loss"])

            if (update + 1) % 100 == 0:
                logger.info(
                    f"[Policy {self.policy_id}] update={update+1}/{n_updates}  "
                    f"alpha={alpha:.3f}  beta={self._beta:.4f}  "
                    f"var_nu={nu:.3f}  CVaR={cvar_hat:.3f}  "
                    f"mu_c={mu_c:.3f}  sigma_c={sigma_c:.3f}  "
                    f"reward_loss={loss_info['reward_loss']:.3f}  "
                    f"cost_penalty={loss_info['cost_penalty']:.3f}  "
                    f"cost_critic_loss={loss_info['cost_critic_loss']:.3f}  "
                    f"total_loss={loss_info['total_loss']:.3f}"
                )

            if (update + 1) % self.cfg.get("save_every", 500) == 0:
                self._save_checkpoint(update)

    # ------------------------------------------------------------------
    # Dense cost signal
    # ------------------------------------------------------------------

    def _compute_dense_costs(self, nu: float, n_local: int) -> tuple:
        """
        Reconstruct each local episode's e_t sequence (e_0=0, matching
        PRISMEnv's own per-episode recursion) and compute the dense
        per-timestep cost signal c~_t for every step in the buffer --
        including a still-in-progress trailing episode, since
        dense_cost_signal only needs the LOCAL e_t/e_{t+1} pair, not the
        episode's final outcome (unlike v1's return-capping, which had to
        mask incomplete episodes because it needed the full episode cost).

        Also runs two sanity checks:
          A1 (WARNING only, never halts): for each COMPLETED local episode
             (ep_id < n_local; the trailing in-progress episode, if any, is
             skipped since its "final" cost isn't final yet), verify the
             telescoping identity
                 |sum_t gamma^t c~_t - (g_nu(C) - g_nu(0))| / max(|g_nu(C)|, 1) < 1e-4
             A violation indicates a bug in the dense-signal computation
             itself (indexing, episode-boundary bookkeeping, etc.) -- this
             is a correctness monitor, not a training-halt condition.
          A2 (contributes to the NaN/Inf halt streak in train()): checks
             c~_t for NaN/Inf across the whole buffer.

        Returns (dense_costs, nan_detected).
        """
        n = len(self._buffer)
        dense_costs = np.zeros(n, dtype=np.float32)
        if n == 0:
            self._buffer.dense_costs = []
            return dense_costs, False

        gamma = self.cfg.get("gamma", 0.99)
        episode_ids = np.asarray(self._buffer.episode_ids)
        step_costs = np.asarray(self._buffer.step_costs, dtype=np.float64)

        for ep_id in np.unique(episode_ids):
            idxs = np.where(episode_ids == ep_id)[0]
            e = 0.0
            disc_sum = 0.0
            for local_t, i in enumerate(idxs):
                e_next = e + (gamma ** local_t) * float(step_costs[i])
                c_tilde = dense_cost_signal(e, e_next, nu, self._tau, local_t, gamma)
                dense_costs[i] = c_tilde
                disc_sum += (gamma ** local_t) * c_tilde
                e = e_next

            # A1 -- runtime telescoping check, completed episodes only.
            if ep_id < n_local:
                g_nu_C = g_nu(e, nu, self._tau)
                g_nu_0 = g_nu(0.0, nu, self._tau)
                expected = g_nu_C - g_nu_0
                rel_err = abs(disc_sum - expected) / max(abs(g_nu_C), 1.0)
                if rel_err >= _TELESCOPING_REL_TOL:
                    logger.warning(
                        f"[Policy {self.policy_id}] Telescoping check failed for "
                        f"local episode {ep_id}: rel_err={rel_err:.2e} >= "
                        f"{_TELESCOPING_REL_TOL:.0e} (sum(gamma^t*c~_t)={disc_sum:.6f}, "
                        f"g_nu(C)-g_nu(0)={expected:.6f}, C={e:.4f}, nu={nu:.4f}, "
                        f"tau={self._tau:.4f})"
                    )

        # A2 -- NaN/Inf check on c~_t.
        nan_detected = not bool(np.isfinite(dense_costs).all())
        if nan_detected:
            n_bad = int((~np.isfinite(dense_costs)).sum())
            logger.error(
                f"[Policy {self.policy_id}] {n_bad}/{n} dense cost values (c~_t) "
                f"are NaN/Inf (nu={nu}, tau={self._tau})."
            )

        self._buffer.dense_costs = dense_costs.tolist()
        return dense_costs, nan_detected

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(
        self, last_obs: dict, dense_costs: np.ndarray, ent_coef: float
    ) -> Dict[str, float]:
        """
        Single PPO update combining reward and cost advantages through the
        SAME clipped-ratio objective:

            A_total = normalize(A_reward) - beta * normalize(A_cost)
            ppo_loss = clip(A_total, ratio)   -- one ratio, one clip

        reward_loss / cost_penalty returned below are UNCLIPPED surrogate
        magnitudes (-A_reward*ratio and beta*A_cost*ratio respectively),
        reported purely as diagnostics for beta calibration (per spec
        section 11) -- they do not sum exactly to the clipped total_loss,
        since clipping is nonlinear in the combined advantage.

        ent_coef is passed in (this update's decayed value, see __init__)
        rather than read from self.cfg -- self.cfg only holds the initial
        value.
        """
        gamma = self.cfg.get("gamma", 0.99)
        gae_lambda = self.cfg.get("gae_lambda", 0.95)
        clip_coef = self.cfg.get("clip_coef", 0.2)
        vf_coef = self.cfg.get("vf_coef", 0.5)
        cf_coef = self.cfg.get("cf_coef", 0.5)
        update_epochs = self.cfg.get("update_epochs", 4)
        minibatch_size = self.cfg.get("minibatch_size", 64)
        max_grad_norm = self.cfg.get("max_grad_norm", 0.5)

        n = len(self._buffer)
        if n == 0:
            return {
                "reward_loss": 0.0, "cost_penalty": 0.0,
                "cost_critic_loss": 0.0, "total_loss": 0.0,
                "v_loss": 0.0, "entropy": 0.0,
                "ppo_loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0,
                "reward_advantage_std": 0.0,
                "nan_detected": False,
                "grad_norm_total_preclip": 0.0,
                "grad_norm_cost_critic": 0.0,
                "grad_norm_other": 0.0,
            }

        rewards = np.array(self._buffer.rewards, dtype=np.float32)
        values = np.array(self._buffer.values, dtype=np.float32)
        cost_values = np.array(self._buffer.cost_values, dtype=np.float32)
        dones = np.array(self._buffer.dones, dtype=np.float32)
        log_probs = np.array(self._buffer.log_probs, dtype=np.float32)
        actions = np.array(self._buffer.actions, dtype=np.float32)
        dense_costs = np.asarray(dense_costs, dtype=np.float32)

        # Bootstrap values -- one forward call yields both reward and cost value
        with torch.no_grad():
            boot_out = self.agent.forward(last_obs)
            next_value = float(boot_out.value.item())
            next_cost_value = float(boot_out.cost_value.item())

        # GAE advantages -- reward (unchanged) and cost (new), parallel loops
        advantages = np.zeros(n, dtype=np.float32)
        cost_advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        last_cost_gae = 0.0
        for t in reversed(range(n)):
            non_term = 1.0 - dones[t]

            next_val = next_value if t == n - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * non_term - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * non_term * last_gae

            next_cval = next_cost_value if t == n - 1 else cost_values[t + 1]
            cost_delta = dense_costs[t] + gamma * next_cval * non_term - cost_values[t]
            cost_advantages[t] = last_cost_gae = (
                cost_delta + gamma * gae_lambda * non_term * last_cost_gae
            )
        returns = advantages + values
        cost_returns = cost_advantages + cost_values

        # Spread of the RAW reward advantage, before per-minibatch
        # normalization erases its scale. This is what actually drives the
        # actor's gradient -- if it's near-zero regardless of what mean
        # reward is doing, there's essentially no signal telling the actor
        # which actions are better, independent of raw reward's own spread
        # (the critic may already be absorbing most of the predictable
        # variation, leaving little in the residual).
        reward_advantage_std = float(np.std(advantages))

        # A2 -- NaN/Inf check on A_t^C (cost advantages).
        nan_detected = not bool(np.isfinite(cost_advantages).all())
        if nan_detected:
            n_bad = int((~np.isfinite(cost_advantages)).sum())
            logger.error(
                f"[Policy {self.policy_id}] {n_bad}/{n} cost advantage values "
                f"(A_t^C) are NaN/Inf."
            )

        # Convert to tensors — obs keys and dtypes are preserved from the buffer
        b_obs = {
            k: torch.from_numpy(
                np.stack([step[k] for step in self._buffer.obs])
            ).to(self.device)
            for k in self._buffer.obs[0].keys()
        }
        b_actions = torch.from_numpy(actions).to(self.device)
        b_log_probs = torch.from_numpy(log_probs).to(self.device)
        b_advantages = torch.from_numpy(advantages).to(self.device)
        b_cost_advantages = torch.from_numpy(cost_advantages).to(self.device)
        b_returns = torch.from_numpy(returns).to(self.device)
        b_cost_returns = torch.from_numpy(cost_returns).to(self.device)

        # Cost-critic parameter set, fixed for the whole update -- used to
        # split each minibatch's gradient norm into cost-critic vs. rest
        # (see below). Empty for backbones without a cost critic.
        cost_critic_params = list(self.agent.cost_critic_parameters())
        cost_critic_param_ids = {id(p) for p in cost_critic_params}

        epoch_reward_loss = 0.0
        epoch_cost_penalty = 0.0
        epoch_cost_critic_loss = 0.0
        epoch_total_loss = 0.0
        epoch_v_loss = 0.0
        epoch_entropy = 0.0
        epoch_ppo_loss = 0.0
        epoch_approx_kl = 0.0
        epoch_clip_fraction = 0.0
        epoch_grad_norm_total = 0.0
        epoch_grad_norm_cost_critic = 0.0
        epoch_grad_norm_other = 0.0
        n_minibatches = 0

        for _ in range(update_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, minibatch_size):
                mb = perm[start : start + minibatch_size]
                mb_obs = {k: v[mb] for k, v in b_obs.items()}
                mb_out = self.agent.forward(mb_obs, actions=b_actions[mb])

                log_ratio = mb_out.log_prob - b_log_probs[mb]
                ratio = log_ratio.exp()

                # Actor-health diagnostics -- standard PPO metrics (matches
                # what e.g. stable-baselines3 tracks), previously entirely
                # absent: approx_kl estimates how much the policy moved this
                # minibatch (Schulman's low-variance estimator: an unclipped
                # surrogate ratio of 1 / zero log-ratio means no movement);
                # clip_fraction is how often updates are hitting the trust-
                # region boundary. Both actor-specific, computed regardless
                # of whether the cost side is active.
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > clip_coef
                    ).float().mean()

                mb_adv_r = b_advantages[mb]
                mb_adv_r = (mb_adv_r - mb_adv_r.mean()) / (mb_adv_r.std() + 1e-8)
                mb_adv_c = b_cost_advantages[mb]
                mb_adv_c = (mb_adv_c - mb_adv_c.mean()) / (mb_adv_c.std() + 1e-8)

                # A_total = A_reward - beta * A_cost, combined BEFORE clipping --
                # reward and cost share one ratio and one clip, unlike v1's
                # separate REINFORCE term.
                mb_adv_total = mb_adv_r - self._beta * mb_adv_c

                ppo_loss = torch.max(
                    -mb_adv_total * ratio,
                    -mb_adv_total * ratio.clamp(1 - clip_coef, 1 + clip_coef),
                ).mean()

                v_loss = 0.5 * ((mb_out.value - b_returns[mb]) ** 2).mean()
                cost_critic_loss = 0.5 * (
                    (mb_out.cost_value - b_cost_returns[mb]) ** 2
                ).mean()
                ent_loss = mb_out.entropy.mean()

                total_loss = (
                    ppo_loss
                    + vf_coef * v_loss
                    + cf_coef * cost_critic_loss
                    - ent_coef * ent_loss
                )

                # A2 -- NaN/Inf check on total_loss. Skip the optimizer step
                # for this minibatch specifically (a NaN/Inf gradient would
                # corrupt every weight in the network) but keep processing
                # the rest of this update -- the halt decision is made at
                # update granularity in train(), not on the first occurrence.
                mb_grad_norm_total = 0.0
                mb_grad_norm_cost_critic = 0.0
                mb_grad_norm_other = 0.0
                if not torch.isfinite(total_loss):
                    nan_detected = True
                    logger.error(
                        f"[Policy {self.policy_id}] total_loss is NaN/Inf this "
                        f"minibatch (ppo_loss={ppo_loss.item():.4g}, "
                        f"v_loss={v_loss.item():.4g}, "
                        f"cost_critic_loss={cost_critic_loss.item():.4g}, "
                        f"ent_loss={ent_loss.item():.4g}) -- skipping optimizer "
                        f"step for this minibatch."
                    )
                else:
                    self.optimizer.zero_grad()
                    total_loss.backward()

                    # Diagnostic: gradient norm split by parameter group,
                    # computed BEFORE clipping (clip_grad_norm_ below rescales
                    # .grad in place) -- directly measures whether the cost
                    # critic's (potentially spiky) gradients dominate the
                    # shared clip, one of the hypothesized instability drivers.
                    all_params = list(self.agent.trainable_parameters())
                    cc_sq = other_sq = 0.0
                    for p in all_params:
                        if p.grad is None:
                            continue
                        sq = float(p.grad.detach().pow(2).sum().item())
                        if id(p) in cost_critic_param_ids:
                            cc_sq += sq
                        else:
                            other_sq += sq
                    mb_grad_norm_cost_critic = cc_sq ** 0.5
                    mb_grad_norm_other = other_sq ** 0.5

                    grad_norm_total_t = torch.nn.utils.clip_grad_norm_(
                        all_params, max_grad_norm
                    )
                    mb_grad_norm_total = float(grad_norm_total_t.item())
                    self.optimizer.step()

                # Diagnostics only (unclipped surrogate magnitudes -- see docstring)
                with torch.no_grad():
                    reward_loss_diag = (-mb_adv_r * ratio).mean()
                    cost_penalty_diag = (self._beta * mb_adv_c * ratio).mean()

                epoch_reward_loss += float(reward_loss_diag.item())
                epoch_cost_penalty += float(cost_penalty_diag.item())
                epoch_cost_critic_loss += float(cost_critic_loss.item())
                epoch_total_loss += float(total_loss.item())
                epoch_v_loss += float(v_loss.item())
                epoch_entropy += float(ent_loss.item())
                epoch_ppo_loss += float(ppo_loss.item())
                epoch_approx_kl += float(approx_kl.item())
                epoch_clip_fraction += float(clip_fraction.item())
                epoch_grad_norm_total += mb_grad_norm_total
                epoch_grad_norm_cost_critic += mb_grad_norm_cost_critic
                epoch_grad_norm_other += mb_grad_norm_other
                n_minibatches += 1

        return {
            "reward_loss": epoch_reward_loss / max(n_minibatches, 1),
            "cost_penalty": epoch_cost_penalty / max(n_minibatches, 1),
            "cost_critic_loss": epoch_cost_critic_loss / max(n_minibatches, 1),
            "total_loss": epoch_total_loss / max(n_minibatches, 1),
            # v_loss: reward critic's own MSE -- should trend down as it
            # learns to predict returns. entropy: should trend down as the
            # policy commits to more confident actions (but collapsing to
            # ~0 immediately would suggest premature convergence, not
            # learning). Neither was previously surfaced past this method.
            "v_loss": epoch_v_loss / max(n_minibatches, 1),
            "entropy": epoch_entropy / max(n_minibatches, 1),
            # Actor-specific health diagnostics (see the approx_kl/
            # clip_fraction comment above where they're computed).
            # ppo_loss: the actor's own clipped-surrogate objective, split
            # out from total_loss (which also includes vf_coef*v_loss,
            # cf_coef*cost_critic_loss, -ent_coef*ent_loss).
            "ppo_loss": epoch_ppo_loss / max(n_minibatches, 1),
            "approx_kl": epoch_approx_kl / max(n_minibatches, 1),
            "clip_fraction": epoch_clip_fraction / max(n_minibatches, 1),
            "reward_advantage_std": reward_advantage_std,
            "nan_detected": nan_detected,
            "grad_norm_total_preclip": epoch_grad_norm_total / max(n_minibatches, 1),
            "grad_norm_cost_critic": epoch_grad_norm_cost_critic / max(n_minibatches, 1),
            "grad_norm_other": epoch_grad_norm_other / max(n_minibatches, 1),
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, update: int, final: bool = False) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tag = "final" if final else f"{update:09d}"
        model_path = self.output_dir / f"policy_{self.policy_id}_model_{tag}.pth"
        util_path = self.output_dir / f"policy_{self.policy_id}_utility_{tag}.pth"
        costbuf_path = self.output_dir / f"policy_{self.policy_id}_costbuffer_{tag}.json"

        torch.save(self.agent.state_dict(), model_path)
        torch.save(self.utility_fn.state_dict(), util_path)

        import json
        with open(costbuf_path, "w") as f:
            json.dump(self._cost_buffer.state_dict(), f)

        logger.info(f"[Policy {self.policy_id}] Saved checkpoint: {tag}")
