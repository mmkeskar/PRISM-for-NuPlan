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
        output_scale: float = 16.0,
    ) -> None:
        super().__init__()
        self._reward_dim = reward_dim
        self._n_hidden = n_hidden
        self._max_weight = max_weight
        self._lamda = lamda
        # Output self-normalisation (see forward()) correctly bounds f(z) to
        # ~[0,1], but that also removed an UNINTENTIONAL scale inflation the
        # old, unbounded output had (an unconstrained output bias could push
        # raw output arbitrarily large). ent_coef/vf_coef/learning_rate were
        # all tuned against that old, larger scale. Measured directly (real
        # training run + offline simulated trajectory, same order of
        # magnitude in both): per-step utility deltas are ~16-17x smaller
        # after the corner-normalisation fix. Restoring that scale here
        # (applied AFTER normalisation + the lamda blend, so it preserves
        # the now-correct relative shape/differentiation and only rescales
        # magnitude) avoids re-tuning every downstream hyperparameter. 16.0
        # is a reasoned starting point from that measurement, not a
        # precisely calibrated value -- may need further tuning, same as
        # ent_coef's own history in this project. See CHANGES.md.
        self._output_scale = output_scale
        # _min_val/_max_val (registered below) stop updating once this many
        # forward() calls have been seen -- previously updated on EVERY
        # call, forever, for the entire run (not just early on): the same
        # raw z value gets normalised differently depending on WHEN during
        # training it's evaluated, since the range it's divided by keeps
        # growing. A moving target for the network to fit against the whole
        # run, not just a warmup-period effect. ~50k calls is roughly
        # 165-330 episodes (at ~150-300 calls/episode, i.e. ~1 forward()
        # call per env step -- PRISMEnv.step() previously called this twice
        # per step (once for f(z_t), once for f(z_{t+1}), not caching the
        # overlap between consecutive steps), which silently halved this to
        # ~82-165 episodes; fixed, see CHANGES.md) -- enough to get a
        # representative sense of the achievable range without waiting so
        # long that most of a run still has it drifting.
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

        # Normalise input to [0, 1] with running min/max (per-dimension --
        # deliberately NOT the reference DPMORL repo's "keep_scale" shared-
        # range option, since our 4 style dimensions have genuinely
        # different natural scales, e.g. z_progress ~13-14 vs z_comfort
        # ~70-80 in a real run -- forcing a shared window would compress the
        # already-smaller dimension further).
        min_v = self._min_val.clamp(max=self._max_val - 1e-5)
        x = (z - min_v.unsqueeze(0)) / (self._max_val - min_v + 1e-5).unsqueeze(0)

        # Output self-normalisation, ported from DPMORL's own reference
        # implementation (MI3/DPMORL/.../utility_function_parameterized.py):
        # evaluate this SAME network, in the SAME forward pass, at the
        # normalised domain's corners (all-0s, all-1s), then rescale this
        # batch's raw output into [0,1] using the network's OWN achievable
        # range at those corners. Monotonicity (every weight clamped >= 0 by
        # make_monotone(), including through the triple-clamp activation)
        # guarantees all-0s is the network's global min and all-1s its
        # global max over the normalised domain, so this is an exact
        # rescaling, not an approximation. Without it, the network's raw
        # output -- in particular its UNCONSTRAINED output bias
        # (make_monotone() only clamps weights, never bias) -- could drift
        # to any magnitude with nothing to correct it; this is very likely
        # why the linear term below was previously found to dominate >99%
        # of the observed signal (the un-normalised neural pathway's own
        # contribution was comparatively tiny and structurally
        # desensitized), and why the separate, currently-unused diversity
        # loss grew unboundedly in an offline test. See CHANGES.md.
        corners = torch.stack(
            [
                torch.zeros(self._reward_dim, device=x.device, dtype=x.dtype),
                torch.ones(self._reward_dim, device=x.device, dtype=x.dtype),
            ],
            dim=0,
        )  # (2, reward_dim)
        h = torch.cat([corners, x], dim=0)  # (2+B, reward_dim)
        h = self.input_bn(h)
        h = self.fc_in(h)

        for bn, fc in zip(self.hidden_bns, self.hidden_layers):
            h = self._triple_clamp(h)
            h = bn(h)
            h = fc(h)

        h = self._triple_clamp(h)
        h = self.fc_out(h)  # (2+B, 1)

        min_util, max_util, raw_util = h[0, 0], h[1, 0], h[2:, 0]  # (), (), (B,)
        net_out = (raw_util - min_util) / (max_util - min_util + 1e-6)  # (B,), in [0, 1]

        # Linear regularisation term -- preference-WEIGHTED (not a plain
        # mean), so it differentiates between policies instead of applying
        # the same generic "grow every objective equally" incentive to all
        # of them. Now blended as a CONVEX combination with net_out (both in
        # ~[0,1]) via lamda, rather than added on top of an un-normalised
        # net_out -- keeps the two terms on a comparable scale, so lamda
        # actually controls their relative weight instead of the linear
        # term dominating purely because it happened to be numerically
        # larger (the >99% figure above). See CHANGES.md.
        linear_term = (x * self._pref_weights).sum(dim=-1)  # (B,)
        out = (1.0 - self._lamda) * net_out + self._lamda * linear_term  # (B,), in ~[0,1]

        # Restore the scale downstream hyperparameters were tuned against
        # (see __init__ for why this is needed, not just cosmetic).
        out = out * self._output_scale

        if squeeze:
            return out.squeeze(0)  # scalar
        return out

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


# ─────────────────────────────────────────────────────────────────────────────
# Ablation: pure linear projection utility (Test 2 of the K-policy-
# divergence investigation)
# ─────────────────────────────────────────────────────────────────────────────

class LinearProjectionUtility(nn.Module):
    """
    Ablation utility function: f(z) = z[dim] exactly -- a pure projection
    onto one style dimension, bypassing ALL of UtilityFunction's learned/
    preference-biased machinery (no monotone network, no input/output
    normalisation, no linear-term blending).

    Purpose: isolate whether Stage 1 utility-function CONSTRUCTION is why K
    policies aren't diverging more, or whether the problem is downstream
    (PPO, the environment, or the reward formulas themselves). This
    function's gradient is maximally, unambiguously different from any
    other dimension's projection -- nabla f_k is a one-hot vector at index
    k, orthogonal to every other dimension's projection EVERYWHERE in R^4,
    with no saturation and no shared-direction confound possible by
    construction (contrast with check_gradient_alignment.py's finding for
    the learned/preference-biased functions). If two policies still don't
    diverge under this, Stage 1 isn't the bottleneck.

    Side effect worth knowing: this also collapses R_t back to the raw
    per-step reward component exactly. Since PRISMEnv updates
    z_{t+1} = z_t + gamma^t * r_vec, z_{t+1}[dim] - z_t[dim] =
    gamma^t * r_vec[dim]_t, so R_t = gamma^{-t} * (f(z_{t+1}) - f(z_t)) =
    r_vec[dim]_t -- i.e. this ablation is simultaneously a test of "does PPO
    on the RAW per-step reward component alone, with no DPMORL wrapping at
    all, produce divergence." See CHANGES.md.
    """

    def __init__(self, reward_dim: int, dim: int) -> None:
        super().__init__()
        assert 0 <= dim < reward_dim, f"dim={dim} out of range for reward_dim={reward_dim}"
        self._reward_dim = reward_dim
        self._dim = dim
        # No real parameters -- a dummy buffer so state_dict()/
        # load_state_dict() (used by checkpointing, same path as
        # UtilityFunction) aren't operating on an empty module, and so
        # .to(device) has something to anchor placement to.
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            return z[self._dim]
        return z[:, self._dim]

    def as_callable(self):
        def _fn(z: np.ndarray) -> float:
            return float(z[self._dim])
        return _fn

    def set_preference(self, pref) -> None:
        pass  # no-op -- no preference-weighted term exists to set here
