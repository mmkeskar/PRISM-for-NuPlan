"""
DPMORL Stage 2 — Per-policy PPO training with an unconstrained CVaR penalty.

One DPMORLTrainer instance manages a single policy k.  The outer training
loop (scripts/train.py) creates K trainers and calls them sequentially.

Training loop (per policy):
    for update in range(n_updates):
        1. Collect rollouts with current PRISMEnv (scalar reward = raw R_t,
           the DPMORL utility-difference reward -- no per-step penalty)
        2. Compute per-episode cumulative safety costs from the rollout
        3. Estimate CVaR_alpha via return capping over a rolling episode
           cost buffer, and cap each *local* episode's cost at the
           buffer's VaR_alpha so every collected step contributes signal
        4. PPO update: actor_loss = reward_loss + beta * cost_penalty
           (fixed beta -- no Lagrange multiplier, no dual update)
        5. Save checkpoint

This replaces the earlier Lagrangian-constrained formulation (lambda_k,
dual ascent, epsilon threshold from IDM baseline).  See PRISM/CHANGES.md
for the full rationale.

Rollout buffer tracks:
    - obs (generic dict — whatever keys PRISMEnv returns; backend-agnostic)
    - actions, log_probs, values, rewards, dones
    - episode_id per step (local index within this update's rollout, used
      to broadcast each episode's capped cost to its own timesteps)
    - episode_zt (for Pareto-front logging at episode end)
    - step_costs (step-level safety costs, for discounted per-episode sum)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from prism.curriculum.alpha_schedule import AlphaSchedule
from prism.models.base import PRISMPolicyBase
from prism.morl.cvar_penalty import (
    EpisodeCostBuffer,
    compute_cvar_with_return_capping,
    compute_episode_cost,
)
from prism.morl.utility_functions import UtilityFunction

logger = logging.getLogger(__name__)


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
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    # Per-step safety costs for episode cost tracking
    step_costs: List[float] = field(default_factory=list)
    # Local episode index (within this update's buffer) for each step --
    # used to broadcast each episode's capped CVaR cost to its timesteps
    episode_ids: List[int] = field(default_factory=list)
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
        CVaR penalty weight) and `cvar_buffer_size` (rolling episode-cost
        window for the VaR/CVaR estimate).
    output_dir : Path
        Where to save checkpoints.
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
        self._cost_buffer = EpisodeCostBuffer(
            buffer_size=cfg.get("cvar_buffer_size", 500)
        )
        self._beta = cfg.get("beta", 1.0)

        self._global_step = 0
        self._buffer = RolloutBuffer()

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
            "mu_c_history": [],
            "sigma_c_history": [],
            "reward_loss_history": [],
            "cost_penalty_history": [],
            "actor_loss_history": [],
            "episode_zts": [],
        }

        obs, _ = self.env.reset()
        next_obs = {k: torch.from_numpy(np.array(v)).unsqueeze(0).to(self.device)
                    for k, v in obs.items()}

        for update in range(n_updates):
            alpha = self.alpha_schedule.get(update)

            # ── Rollout collection ─────────────────────────────────────
            self._buffer.clear()
            episode_step_costs: List[float] = []
            local_episode_costs: List[float] = []
            current_ep_id = 0

            for _ in range(steps_per_update):
                self._global_step += 1

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
                self._buffer.rewards.append(float(reward))
                self._buffer.dones.append(done)
                self._buffer.step_costs.append(c_t)
                self._buffer.episode_ids.append(current_ep_id)

                if done:
                    # Record episode-level cumulative cost for CVaR
                    ep_cost = compute_episode_cost(
                        episode_step_costs, gamma=self.cfg.get("gamma", 0.99)
                    )
                    local_episode_costs.append(ep_cost)
                    self._cost_buffer.add_episode(ep_cost)
                    episode_step_costs = []
                    current_ep_id += 1

                    if "episode_zt" in info:
                        self._buffer.episode_zts.append(info["episode_zt"].copy())
                        summary["episode_zts"].append(info["episode_zt"].copy())

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

            # ── Empirical CVaR + return capping ─────────────────────────
            # VaR_alpha is estimated from the rolling cost buffer (many
            # more episodes than a single update collects, so the tail
            # quantile is stable); capped costs are then computed for
            # THIS update's local episodes so every step in the current
            # batch gets a gradient signal (see cvar_penalty.py).
            buffer_costs = self._cost_buffer.costs
            cvar_hat, _ = compute_cvar_with_return_capping(buffer_costs, alpha)
            mu_c = float(np.mean(buffer_costs)) if buffer_costs else 0.0
            sigma_c = float(np.std(buffer_costs)) if buffer_costs else 0.0

            capped_cost_per_step = np.zeros(len(self._buffer), dtype=np.float32)
            if local_episode_costs:
                var_alpha = self._estimate_var(buffer_costs, alpha)
                n_local = len(local_episode_costs)
                local_excess = np.clip(
                    np.array(local_episode_costs) - var_alpha, a_min=0.0, a_max=None
                )
                local_capped = (
                    var_alpha / n_local + local_excess / ((1.0 - alpha) * n_local)
                )
                episode_ids = np.array(self._buffer.episode_ids)
                # Steps belonging to a trailing episode that had not yet
                # terminated when the update ended get weight 0 -- its
                # outcome (and therefore its cost) is still unknown.
                completed_mask = episode_ids < n_local
                capped_cost_per_step[completed_mask] = local_capped[
                    episode_ids[completed_mask]
                ]

            # ── PPO update ─────────────────────────────────────────────
            loss_info = self._ppo_update(next_obs, capped_cost_per_step)

            summary["cvar_history"].append(cvar_hat)
            summary["mu_c_history"].append(mu_c)
            summary["sigma_c_history"].append(sigma_c)
            summary["reward_loss_history"].append(loss_info["reward_loss"])
            summary["cost_penalty_history"].append(loss_info["cost_penalty"])
            summary["actor_loss_history"].append(loss_info["actor_loss"])

            if (update + 1) % 100 == 0:
                logger.info(
                    f"[Policy {self.policy_id}] update={update+1}/{n_updates}  "
                    f"alpha={alpha:.2f}  beta={self._beta:.4f}  "
                    f"CVaR={cvar_hat:.3f}  mu_c={mu_c:.3f}  sigma_c={sigma_c:.3f}  "
                    f"reward_loss={loss_info['reward_loss']:.3f}  "
                    f"cost_penalty={loss_info['cost_penalty']:.3f}  "
                    f"actor_loss={loss_info['actor_loss']:.3f}"
                )

            if (update + 1) % self.cfg.get("save_every", 500) == 0:
                self._save_checkpoint(update)

        self._save_checkpoint(n_updates - 1, final=True)
        return summary

    # ------------------------------------------------------------------
    # VaR estimation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_var(episode_costs: List[float], alpha: float) -> float:
        """Alpha-quantile of a batch of episode costs (VaR_alpha)."""
        if not episode_costs:
            return 0.0
        import math
        costs = np.sort(np.asarray(episode_costs, dtype=np.float64))[::-1]
        k = max(1, int(math.ceil((1.0 - alpha) * len(costs))))
        return float(costs[k - 1])

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, last_obs: dict, capped_cost_per_step: np.ndarray) -> Dict[str, float]:
        """
        Single PPO update using the collected rollout buffer.

        actor_loss (for logging) = reward_loss + beta * cost_penalty, per
        the unconstrained-penalty formulation.  The tensor actually
        back-propagated additionally includes the (unchanged) PPO value
        loss and entropy bonus, since policy and value heads share one
        optimizer -- that bookkeeping is orthogonal to the CVaR refactor.
        """
        gamma = self.cfg.get("gamma", 0.99)
        gae_lambda = self.cfg.get("gae_lambda", 0.95)
        clip_coef = self.cfg.get("clip_coef", 0.2)
        vf_coef = self.cfg.get("vf_coef", 0.5)
        ent_coef = self.cfg.get("ent_coef", 0.01)
        update_epochs = self.cfg.get("update_epochs", 4)
        minibatch_size = self.cfg.get("minibatch_size", 64)
        max_grad_norm = self.cfg.get("max_grad_norm", 0.5)

        n = len(self._buffer)
        if n == 0:
            return {"reward_loss": 0.0, "cost_penalty": 0.0, "actor_loss": 0.0}

        rewards = np.array(self._buffer.rewards, dtype=np.float32)
        values = np.array(self._buffer.values, dtype=np.float32)
        dones = np.array(self._buffer.dones, dtype=np.float32)
        log_probs = np.array(self._buffer.log_probs, dtype=np.float32)
        actions = np.array(self._buffer.actions, dtype=np.float32)

        # Bootstrap value
        with torch.no_grad():
            next_value = float(self.agent.forward(last_obs).value.item())

        # GAE advantages (reward-side objective, unchanged by the CVaR refactor)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_val = next_value if t == n - 1 else values[t + 1]
            non_term = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * non_term - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * non_term * last_gae
        returns = advantages + values

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
        b_returns = torch.from_numpy(returns).to(self.device)
        b_values = torch.from_numpy(values).to(self.device)
        b_capped_costs = torch.from_numpy(capped_cost_per_step).to(self.device)

        epoch_reward_loss, epoch_cost_term, epoch_actor_loss = 0.0, 0.0, 0.0
        n_minibatches = 0

        for _ in range(update_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, minibatch_size):
                mb = perm[start : start + minibatch_size]
                mb_obs = {k: v[mb] for k, v in b_obs.items()}
                mb_out = self.agent.forward(mb_obs, actions=b_actions[mb])

                new_value = mb_out.value.view(-1)
                log_ratio = mb_out.log_prob - b_log_probs[mb]
                ratio = log_ratio.exp()

                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # reward_loss: -E_w[w . Z(pi)], via the clipped PPO surrogate (unchanged)
                reward_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - clip_coef, 1 + clip_coef),
                ).mean()

                # cost_penalty: beta * CVaR_alpha(C^pi), via REINFORCE on the
                # return-capped per-episode cost (broadcast to its timesteps)
                mb_capped = b_capped_costs[mb]
                cost_term = (mb_out.log_prob * mb_capped).mean()
                cost_penalty = self._beta * cost_term

                v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()
                ent_loss = mb_out.entropy.mean()

                actor_loss = reward_loss + cost_penalty
                loss = actor_loss + vf_coef * v_loss - ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.agent.trainable_parameters()), max_grad_norm
                )
                self.optimizer.step()

                epoch_reward_loss += float(reward_loss.item())
                epoch_cost_term += float(cost_penalty.item())
                epoch_actor_loss += float(actor_loss.item())
                n_minibatches += 1

        return {
            "reward_loss": epoch_reward_loss / max(n_minibatches, 1),
            "cost_penalty": epoch_cost_term / max(n_minibatches, 1),
            "actor_loss": epoch_actor_loss / max(n_minibatches, 1),
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
