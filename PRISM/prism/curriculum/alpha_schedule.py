"""
Alpha curriculum schedule for PRISM.

alpha(n) controls how conservative the CVaR penalty is:
    - Small alpha (0.20)  -> only the worst 20% of episodes matter (lenient)
    - Large alpha (0.95)  -> almost all episodes must satisfy the constraint (strict)

The schedule starts lenient and tightens over training so the agent first
learns to drive at all, then gradually becomes safer.

    alpha(n) = alpha_start + (alpha_end - alpha_start) * min(1, n / N_curriculum)

There is no epsilon/threshold anymore -- CVaR_alpha(C^pi) is penalised
directly in the actor loss via a fixed weight beta (see cvar_penalty.py),
not constrained against a threshold read from hyperparams.json.
"""

import numpy as np


def compute_alpha(n: int, alpha_start: float, alpha_end: float, n_curriculum: int) -> float:
    """Return the current CVaR confidence level at training iteration n."""
    frac = min(1.0, n / max(n_curriculum, 1))
    return alpha_start + (alpha_end - alpha_start) * frac


class AlphaSchedule:
    """Stateful schedule tracking alpha over training iterations."""

    def __init__(
        self,
        alpha_start: float = 0.20,
        alpha_end: float = 0.95,
        n_curriculum: int = 5000,
    ) -> None:
        self._alpha_start = alpha_start
        self._alpha_end = alpha_end
        self._n_curriculum = n_curriculum

    def get(self, n: int) -> float:
        """Return alpha for training iteration n."""
        return compute_alpha(n, self._alpha_start, self._alpha_end, self._n_curriculum)

    def alpha_grid(self, n_steps: int = 200):
        """Return the full alpha trajectory for plotting."""
        ns = np.linspace(0, self._n_curriculum * 1.5, n_steps)
        alphas = [
            compute_alpha(int(n), self._alpha_start, self._alpha_end, self._n_curriculum)
            for n in ns
        ]
        return ns, alphas
