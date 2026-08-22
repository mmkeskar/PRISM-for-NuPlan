"""
Unconstrained CVaR safety penalty with empirical estimation and return capping.

Replaces the old Lagrangian-dual formulation (prism/morl/cvar_lagrangian.py,
removed).  There is no lambda, no dual update, and no threshold epsilon.
Instead the actor loss is penalised directly by a fixed weight:

    actor_loss = reward_loss + beta * CVaR_alpha(C^pi)

CVaR is estimated empirically (sorted episode costs), never via a Gaussian
closed form -- the two-tier safety cost distribution is right-skewed and a
Gaussian assumption on it is not valid (see PRISM/CHANGES.md).

Return capping (Rockafellar-Uryasev) lets every collected episode
contribute to the CVaR gradient instead of only the top (1-alpha) tail,
which would otherwise waste most of a rollout batch.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Tuple

import numpy as np


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

    No distributional assumption: sorts the batch descending and averages
    the worst ceil((1-alpha) * N) episodes.  alpha=0.95 -> average of the
    worst 5% of episodes.  Returns 0.0 for an empty batch.
    """
    costs = np.asarray(episode_costs, dtype=np.float64)
    n = costs.shape[0]
    if n == 0:
        return 0.0

    k = max(1, int(math.ceil((1.0 - alpha) * n)))
    sorted_costs = np.sort(costs)[::-1]  # descending
    return float(sorted_costs[:k].mean())


def compute_cvar_with_return_capping(
    episode_costs, alpha: float
) -> Tuple[float, np.ndarray]:
    """
    Rockafellar-Uryasev CVaR estimate with per-episode capped costs.

        CVaR_alpha(X) = min_nu [ nu + 1/(1-alpha) * E[(X - nu)^+] ]

    VaR_alpha (the optimal nu*) is estimated as the alpha-quantile of the
    batch.  capped_costs sums to the same CVaR value but spreads the
    signal across every episode in the batch (not just the tail), so all
    collected rollouts contribute to the policy-gradient estimate.

    Args:
        episode_costs: episode cumulative costs, shape (N,)
        alpha: CVaR confidence level in (0, 1)

    Returns:
        cvar: scalar empirical CVaR (Rockafellar-Uryasev estimate)
        capped_costs: shape (N,) array of per-episode gradient weights;
            capped_costs.sum() == cvar
    """
    costs = np.asarray(episode_costs, dtype=np.float64)
    n = costs.shape[0]
    if n == 0:
        return 0.0, np.zeros(0, dtype=np.float64)

    k = max(1, int(math.ceil((1.0 - alpha) * n)))
    sorted_costs = np.sort(costs)[::-1]  # descending
    var_alpha = float(sorted_costs[k - 1])  # k-th worst cost ~= VaR_alpha

    excess = np.clip(costs - var_alpha, a_min=0.0, a_max=None)
    cvar = var_alpha + float(excess.mean()) / (1.0 - alpha)

    capped_costs = var_alpha / n + excess / ((1.0 - alpha) * n)
    return cvar, capped_costs


class EpisodeCostBuffer:
    """
    Rolling window of recent episode cumulative costs.

    Purely a variance-reduction device for the empirical CVaR/VaR estimate
    -- a single training update collects too few complete episodes
    (~steps_per_update / episode_length) for a stable tail quantile at
    high alpha, so costs are pooled across the last `buffer_size` episodes.

    Unlike the old CVaRLagrangian this holds NO dual variable and performs
    no update rule; it is a plain statistics buffer.
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
