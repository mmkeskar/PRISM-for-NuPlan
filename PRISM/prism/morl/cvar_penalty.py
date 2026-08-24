"""
CVaR safety objective: state-augmented cost with a dense per-timestep signal.

v2 of PRISM's CVaR safety machinery.  Replaces v1's unconstrained penalty
via a REINFORCE-style term on raw episode costs (which reused the
Rockafellar-Uryasev VALUE decomposition as a policy-gradient WEIGHT --
an envelope-theorem violation -- and applied it with no importance-ratio
correction across PPO's epochs, unlike the reward term).

v2 instead extends DPMORL's own state-augmentation trick (already used for
the reward side -- see prism/env/nuplan_env.py's R_t construction) to the
cost side:

    e_t = cumulative discounted safety cost (tracked by PRISMEnv, exposed
          as obs["cumulative_cost"], identical recursion to z_t)
    nu  = VaR_alpha(episode costs) -- the alpha-quantile, re-estimated once
          per training update from a rolling window of recent episodes
    g_nu(e) = tau * softplus((e - nu) / tau)  -- a FIXED (not learned, not
          per-policy) smooth hinge at the VaR threshold
    c~_t = gamma^{-t} * [g_nu(e_{t+1}) - g_nu(e_t)]  -- dense per-timestep
          cost signal, fed to a separate cost critic V^C(s, e_t) via GAE,
          exactly mirroring how R_t is fed to the reward critic

Why g_nu instead of the Muni et al. (arXiv:2602.03778) formula originally
proposed: that paper's actual recursion (z' = (r+z)/gamma) does not match
PRISM's e_t convention (e_{t+1} = e_t + gamma^t * c_t), and a literal
transcription of their formula was verified (by direct summation) to
telescope to min(C^pi, nu) -- which stops discriminating once an episode
exceeds VaR, backwards for a CVaR objective.  g_nu telescopes correctly by
construction (identical proof to R_t's telescoping: the gamma^{-t} prefactor
exactly cancels GAE's gamma^l discounting), is dense everywhere
(g_nu'(e) = sigmoid((e-nu)/tau) > 0), and reuses machinery already proven
correct in this codebase instead of an unverified external formula.  See
PRISM/CHANGES.md for the full derivation and paper-verification notes.

The resulting PPO loss combines reward and cost through the SAME clipped
advantage objective:

    A_total = A_reward - beta * A_cost
    actor_loss = ppo_clip(A_total)   # single ratio, single clip, both terms

No lambda, no dual update, no threshold d, no REINFORCE, no raw log_prob
weighting -- see prism/morl/dpmorl_trainer.py.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List

import numpy as np
import torch


def compute_episode_cost(costs_per_step: List[float], gamma: float = 0.99) -> float:
    """
    Discounted cumulative safety cost of one episode: C^i = sum_t(gamma^t * c_t).
    """
    total = 0.0
    for t, c in enumerate(costs_per_step):
        total += (gamma ** t) * c
    return total


def compute_empirical_cvar(episode_costs, alpha: float) -> float:
    """
    Empirical CVaR_alpha from a batch of episode cumulative costs.

    Diagnostic only in v2 -- CVaR is no longer part of the loss (the dense
    cost signal / cost critic / GAE machinery is).  No distributional
    assumption: sorts the batch descending and averages the worst
    ceil((1-alpha) * N) episodes.  Returns 0.0 for an empty batch.
    """
    costs = np.asarray(episode_costs, dtype=np.float64)
    n = costs.shape[0]
    if n == 0:
        return 0.0

    k = max(1, int(math.ceil((1.0 - alpha) * n)))
    sorted_costs = np.sort(costs)[::-1]  # descending
    return float(sorted_costs[:k].mean())


def update_var(episode_costs, alpha: float) -> float:
    """
    Estimate VaR_alpha (the nu threshold) as the alpha-quantile of a batch
    of episode cumulative costs.

    This is a plain scalar statistic -- not a gradient step, not a neural
    network output.  alpha is clamped to <= 0.999 so downstream 1/(1-alpha)
    uses (e.g. in the CVaR diagnostic) never divide by zero, even if a
    config sets alpha_end=1.0.  Returns 0.0 for an empty batch.
    """
    if len(episode_costs) == 0:
        return 0.0
    alpha_clamped = min(float(alpha), 0.999)
    costs_t = torch.as_tensor(list(episode_costs), dtype=torch.float64)
    return float(torch.quantile(costs_t, alpha_clamped).item())


def _softplus(x: float) -> float:
    """Numerically stable softplus for a python float: log(1 + exp(x))."""
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def g_nu(e: float, nu: float, tau: float) -> float:
    """
    Fixed smooth hinge at the VaR threshold: tau * softplus((e - nu) / tau).

    Non-decreasing, everywhere-differentiable approximation of (e-nu)^+.
    tau controls sharpness: tau -> 0 recovers the exact (sparse) hinge;
    larger tau spreads density further from nu.  NOT learned, NOT
    per-policy -- a single fixed function of the current nu, shared across
    all K policies (safety is style-independent).
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    return tau * _softplus((e - nu) / tau)


def dense_cost_signal(
    e_t: float, e_tp1: float, nu: float, tau: float, t: int, gamma: float
) -> float:
    """
    Dense per-timestep cost signal: c~_t = gamma^{-t} * [g_nu(e_{t+1}) - g_nu(e_t)].

    Telescopes exactly (same proof as DPMORL's R_t): for an episode of
    length T with e_0 = 0,
        sum_t gamma^t * c~_t = g_nu(e_T) - g_nu(e_0) = g_nu(C^pi) - g_nu(0)
    Dense because g_nu' > 0 everywhere, unlike the naive sparse hinge
    difference (e_{t+1}-nu)^+ - (e_t-nu)^+, which is exactly zero whenever
    both e_t and e_{t+1} are below nu.
    """
    return (gamma ** (-t)) * (g_nu(e_tp1, nu, tau) - g_nu(e_t, nu, tau))


class EpisodeCostBuffer:
    """
    Rolling window of recent episode cumulative costs.

    Purely a variance-reduction device for the VaR/CVaR estimate -- a
    single training update collects too few complete episodes
    (~steps_per_update / episode_length) for a stable quantile at high
    alpha, so costs are pooled across the last `buffer_size` episodes.

    Holds NO dual variable and performs no update rule; it is a plain
    statistics buffer, reused unmodified from v1.
    """

    def __init__(self, buffer_size: int = 500) -> None:
        self._costs: Deque[float] = deque(maxlen=buffer_size)

    def add_episode(self, cumulative_cost: float) -> None:
        self._costs.append(float(cumulative_cost))

    def add_episodes(self, costs: List[float]) -> None:
        for c in costs:
            self.add_episode(c)

    @property
    def costs(self) -> List[float]:
        return list(self._costs)

    def __len__(self) -> int:
        return len(self._costs)

    def state_dict(self) -> dict:
        return {"costs": list(self._costs)}

    @classmethod
    def from_state_dict(cls, d: dict, buffer_size: int = 500) -> "EpisodeCostBuffer":
        obj = cls(buffer_size=buffer_size)
        obj._costs.extend(d.get("costs", []))
        return obj
