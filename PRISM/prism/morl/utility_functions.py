"""
Non-decreasing neural network utility functions for PRISM.

Architecture follows DPMORL (Cai et al., NeurIPS 2023).  Monotonicity is
enforced by clamping all layer weights to [0, max_weight] after every
gradient step via make_monotone().

The network maps a cumulative style-return vector z ∈ R^4 to a scalar
utility value f(z) ∈ R.  Because weights are non-negative, f is
non-decreasing in each z_i — meaning "more cumulative reward in any
objective is always at least as good" from the perspective of this utility.

Stage 1 of PRISM trains K utility functions jointly to maximise their
pairwise diversity on the space of achievable return vectors.

Usage
-----
    uf = UtilityFunction(reward_dim=4, n_hidden=64)
    uf.make_monotone()          # call after every optimizer step
    scalar = uf(z_batch)        # (B, 4) -> (B,)  or  (4,) -> ()
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UtilityFunction(nn.Module):
    """
    A non-decreasing neural network mapping z ∈ R^{reward_dim} → R.

    Monotonicity constraint: all weight matrices have non-negative entries,
    enforced by clamping after each gradient update.

    The "tripling" activation trick from DPMORL replaces ReLU with three
    clamped versions of the pre-activation, tripling the hidden dimension:
        out = cat(clamp(x, min=-0.5), clamp(x, -0.5, 0.5), clamp(x, max=0.5))
    This produces a piecewise-linear activation that is monotone-compatible.
    """

    def __init__(
        self,
        reward_dim: int = 4,
        n_hidden: int = 24,
        n_layers: int = 3,
        max_weight: float = 1.0,
        lamda: float = 0.05,
        normalization_warmup_calls: int = 50_000,
    ) -> None:
        super().__init__()
        self._reward_dim = reward_dim
        self._n_hidden = n_hidden
        self._max_weight = max_weight
        self._lamda = lamda
        # _min_val/_max_val (registered below) stop updating once this many
        # forward() calls have been seen -- previously updated on EVERY
        # call, forever, for the entire run (not just early on): the same
        # raw z value gets normalised differently depending on WHEN during
        # training it's evaluated, since the range it's divided by keeps
        # growing. A moving target for the network to fit against the whole
        # run, not just a warmup-period effect. ~50k calls is roughly
        # 165-330 episodes (at ~150-300 calls/episode) -- enough to get a
        # representative sense of the achievable range without waiting so
        # long that most of a run still has it drifting. See CHANGES.md.
        self._normalization_warmup_calls = normalization_warmup_calls

        # Input layer: reward_dim -> n_hidden
        self.input_bn = nn.BatchNorm1d(reward_dim, affine=False)
        self.fc_in = nn.Linear(reward_dim, n_hidden)

        # Hidden layers: (n_hidden * 3) -> n_hidden, n_layers times
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        for _ in range(n_layers):
            self.hidden_bns.append(nn.BatchNorm1d(n_hidden * 3, affine=True))
            self.hidden_layers.append(nn.Linear(n_hidden * 3, n_hidden))

        # Output layer: n_hidden * 3 -> 1
        self.fc_out = nn.Linear(n_hidden * 3, 1)

        # Running min/max for normalisation (not backpropd). Frozen after
        # _normalization_warmup_calls (see _update_running_stats) -- part of
        # state_dict() like the other buffers here, so the freeze point
        # (and whatever range was captured) survives checkpoint save/load.
        self.register_buffer("_min_val", torch.full((reward_dim,), float("inf")))
        self.register_buffer("_max_val", torch.full((reward_dim,), float("-inf")))
        self.register_buffer("_calls_seen", torch.tensor(0, dtype=torch.long))

        # Preference weights for the linear regularisation term (see
        # forward()). Defaults to uniform (equivalent to the old z.mean()
        # behavior) until set_preference() is called with this policy's
        # actual w_k -- part of state_dict(), so saved/restored automatically
        # by checkpoint load/save, no separate plumbing needed there.
        self.register_buffer("_pref_weights", torch.ones(reward_dim) / reward_dim)

        self._init_positive()
        self.make_monotone()
        self._init_params = copy.deepcopy(self.state_dict())

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, reward_dim) or (reward_dim,)
        Returns scalar(s) of shape (B,) or () respectively.
        """
        squeeze = z.dim() == 1
        if squeeze:
            z = z.unsqueeze(0)

        z = z.float()
        self._update_running_stats(z.detach())

        # Normalise input to [0, 1] with running min/max
        min_v = self._min_val.clamp(max=self._max_val - 1e-5)
        x = (z - min_v.unsqueeze(0)) / (self._max_val - min_v + 1e-5).unsqueeze(0)
        x = self.input_bn(x)
        x = self.fc_in(x)

        for bn, fc in zip(self.hidden_bns, self.hidden_layers):
            x = self._triple_clamp(x)
            x = bn(x)
            x = fc(x)

        x = self._triple_clamp(x)
        x = self.fc_out(x)  # (B, 1)

        # Linear regularisation: add lamda * (pref_weights . z) to keep
        # gradient non-zero -- preference-WEIGHTED (not a plain mean), so
        # this fallback term differentiates between policies instead of
        # applying the same generic "grow every objective equally"
        # incentive to all of them. In practice this term dominates the
        # observed reward signal (>99% in an offline check) because the
        # neural pathway above is structurally desensitized: it operates on
        # z after min-max normalisation by the running min/max seen so far,
        # which spans the full cumulative episode range (~0 to ~80) -- a
        # single timestep's change in z is tiny by comparison, so the
        # network barely reacts to it regardless of what happened that
        # step. Getting this term right therefore matters far more than
        # its "regularisation" name implies. See CHANGES.md.
        linear_term = self._lamda * (z * self._pref_weights).sum(dim=-1, keepdim=True)
        out = x + linear_term  # (B, 1)

        if squeeze:
            return out.squeeze(0).squeeze(-1)  # scalar
        return out.squeeze(-1)  # (B,)

    # ------------------------------------------------------------------
    # NumPy convenience wrapper (used in env step)
    # ------------------------------------------------------------------

    def __call_numpy(self, z_np: np.ndarray) -> float:
        """Evaluate utility on a single z vector (numpy), return float."""
        with torch.no_grad():
            device = next(self.parameters()).device
            z_t = torch.from_numpy(z_np.astype(np.float32)).to(device)
            val = super().__call__(z_t)
            return float(val.cpu().item())

    def as_callable(self):
        """Return a plain callable (np.ndarray -> float) for use in PRISMEnv."""
        def _fn(z: np.ndarray) -> float:
            self.eval()
            return self.__call_numpy(z)
        return _fn

    def set_preference(self, pref: Sequence[float]) -> None:
        """
        Set this policy's preference vector, used to weight the linear
        regularisation term in forward() (see the note there). Does NOT
        affect monotonicity or the neural pathway -- only which combination
        of z components the (in practice dominant) fallback linear term
        rewards. `pref` need not sum to 1 (not enforced, matching how w_k
        is used elsewhere in this codebase).
        """
        pref_t = torch.as_tensor(pref, dtype=torch.float32, device=self._pref_weights.device)
        assert pref_t.shape == self._pref_weights.shape, (
            f"preference vector must have shape ({self._reward_dim},), got {tuple(pref_t.shape)}"
        )
        self._pref_weights.copy_(pref_t)

    # ------------------------------------------------------------------
    # Monotonicity enforcement
    # ------------------------------------------------------------------

    def make_monotone(self) -> None:
        """Clamp all weights to [0, max_weight]. Call after every optimizer step."""
        for m in self._monotone_layers():
            m.weight.data.clamp_(0.0, self._max_weight)
            if hasattr(m, "bias") and m.bias is not None:
                pass  # biases are unconstrained

    def _init_positive(self) -> None:
        """Initialise all weights positive (required for monotonicity at init)."""
        for m in self._monotone_layers():
            m.weight.data = m.weight.data.abs()

    def _monotone_layers(self):
        layers = [self.fc_in] + list(self.hidden_layers) + [self.fc_out]
        bns = list(self.hidden_bns)
        return layers + bns

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def random_reinit(self) -> None:
        """Reinitialise to stored initial params (for Stage 1 diversity search)."""
        self.load_state_dict(self._init_params)
        self.make_monotone()

    def _update_running_stats(self, z: torch.Tensor) -> None:
        if self._calls_seen.item() >= self._normalization_warmup_calls:
            return  # frozen -- normalization range stays stable for the rest of the run
        self._min_val = torch.minimum(self._min_val, z.min(0).values)
        self._max_val = torch.maximum(self._max_val, z.max(0).values)
        self._calls_seen += z.shape[0]

    @staticmethod
    def _triple_clamp(x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [x.clamp(min=-0.5), x.clamp(-0.5, 0.5), x.clamp(max=0.5)],
            dim=-1,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Diversity training
# ─────────────────────────────────────────────────────────────────────────────

def train_utility_functions_stage1(
    utility_fns: List[UtilityFunction],
    z_samples: np.ndarray,
    n_iters: int = 2000,
    lr: float = 1e-3,
    diversity_weight: float = 1.0,
    device: str = "cpu",
) -> List[UtilityFunction]:
    """
    Stage 1: jointly train K utility functions to be diverse.

    Diversity loss = -sum_{i != j} ||f_i(z) - f_j(z)||_2^2  (maximise spread)
    Each f_i also has a regularisation loss to avoid degenerate solutions.

    Parameters
    ----------
    utility_fns : list of K UtilityFunction instances
    z_samples   : np.ndarray of shape (N, 4) — return vectors from IDM rollouts
    n_iters     : training iterations
    lr          : learning rate
    diversity_weight : weight on diversity vs regularisation
    device      : "cpu" or "cuda:X"
    """
    K = len(utility_fns)
    for uf in utility_fns:
        uf.to(device)
        uf.train()

    optimizers = [
        torch.optim.Adam(uf.parameters(), lr=lr) for uf in utility_fns
    ]

    z_tensor = torch.from_numpy(z_samples.astype(np.float32)).to(device)  # (N, 4)
    N = z_tensor.shape[0]

    for iteration in range(n_iters):
        # Sample mini-batch
        idx = torch.randint(0, N, (min(256, N),))
        z_batch = z_tensor[idx]  # (B, 4)

        # Evaluate all K utilities on this batch: (K, B)
        utilities = torch.stack([uf(z_batch) for uf in utility_fns], dim=0)

        # Diversity loss: maximise pairwise distances
        div_loss = torch.tensor(0.0, device=device)
        for i in range(K):
            for j in range(i + 1, K):
                diff = utilities[i] - utilities[j]
                div_loss = div_loss - (diff ** 2).mean()

        total_loss = diversity_weight * div_loss

        for opt in optimizers:
            opt.zero_grad()
        total_loss.backward()
        for opt in optimizers:
            opt.step()
        for uf in utility_fns:
            uf.make_monotone()

        if (iteration + 1) % 500 == 0:
            print(
                f"[Stage1] iter {iteration + 1}/{n_iters}  "
                f"div_loss={div_loss.item():.4f}"
            )

    for uf in utility_fns:
        uf.eval()

    return utility_fns


def init_utility_functions_from_preferences(
    n_policies: int = 5,
    reward_dim: int = 4,
    preference_vectors: Optional[Sequence[Sequence[float]]] = None,
    device: str = "cpu",
) -> List[UtilityFunction]:
    """
    Initialise K utility functions from explicit preference vectors.

    Each preference vector defines which objectives the policy should
    prioritise (used when Stage 1 diversity training is skipped).

    Default vectors for K=5, reward_dim=4:
        1. Comfort-focused
        2. Progress-focused
        3. Lateral discipline-focused
        4. Spacing-focused
        5. Balanced
    """
    if preference_vectors is None:
        if n_policies == 5 and reward_dim == 4:
            preference_vectors = [
                [0.55, 0.15, 0.15, 0.15],  # comfort
                [0.15, 0.55, 0.15, 0.15],  # progress
                [0.15, 0.15, 0.55, 0.15],  # lateral
                [0.15, 0.15, 0.15, 0.55],  # spacing
                [0.25, 0.25, 0.25, 0.25],  # balanced
            ]
        else:
            # Evenly distribute emphasis
            base = [1.0 / reward_dim] * reward_dim
            preference_vectors = []
            for k in range(n_policies):
                vec = base.copy()
                obj_idx = k % reward_dim
                vec[obj_idx] += 0.4 * (1.0 - 1.0 / reward_dim)
                total = sum(vec)
                vec = [v / total for v in vec]
                preference_vectors.append(vec)

    ufs = []
    for pref in preference_vectors:
        uf = UtilityFunction(reward_dim=reward_dim)
        uf.to(device)
        uf.set_preference(pref)
        # Bias fc_in weights toward the preference direction
        with torch.no_grad():
            w = uf.fc_in.weight.data  # (n_hidden, reward_dim)
            pref_t = torch.tensor(pref, dtype=torch.float32, device=device)
            w[:] = w * pref_t.unsqueeze(0)
            w.clamp_(0.0, uf._max_weight)
        uf.make_monotone()
        ufs.append(uf)

    return ufs
