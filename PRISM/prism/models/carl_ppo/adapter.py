"""
Adapter that wraps CaRL's PPOPolicy into PRISMPolicyBase, with style
conditioning added to the actor (Problem 1) and a separate CVaR cost
critic (v2 CVaR refactor).

CaRL's PPOPolicy architecture (from ppo_model.py):
    get_features(obs)          BEV CNN + measurements MLP → (batch, features_dim)
    policy_head(features)      MLP → latent_pi            [ACTOR]
    dist_mu/sigma(latent_pi)   → action distribution      [ACTOR]
    value_head(cat([features, value_measurements]))        [REWARD CRITIC]

Problem 1 fix — style conditioning on the actor:
    The original actor sees only `features` and has no knowledge of the
    style preference w_k or the cumulative return z_t.  We replace the
    actor path entirely, and (v2) additionally state-augment it with e_t
    (cumulative safety cost) alongside z_t -- per the CVaR v2 spec
    s~_t = (s_t, z_t, e_t), so the actor itself can act more conservatively
    as it approaches the VaR threshold nu, not just have its cost judged
    after the fact by a critic it cannot see:

    features = policy.get_features(obs)             (batch, features_dim)
    features_styled = FiLM(w_k, features)           (batch, features_dim)   ← w_k
    features_aug = cat([features_styled, z_t, e_t]) (batch, features_dim+5) ← z_t, e_t
    latent_pi = style_policy_head(features_aug)     (batch, 256)
    action_mean = action_proj(latent_pi)             (batch, action_dim)
    act, log_prob, entropy = StochasticActionHead(action_mean, actions)

    FiLM initialisation: scale bias=1, shift bias=0, all weights=0.
    This makes FiLM identity at the start of training (no modulation when
    w_k ≈ 0) and lets it learn gradually.

    The CaRL critic (value_head) is kept as-is.  z_t reaches it through
    obs["value_measurements"], which is the CaRL slot PRISM repurposes for
    z_t_normalised.  w_k conditioning on the critic is addressed separately
    (Problem 2). e_t deliberately does NOT reach this reward critic --
    "Reward critic and its training" is explicitly out of scope for the
    CVaR refactor.

CVaR cost critic (v2):
    A separate, style-INDEPENDENT head reads the same raw `features` (not
    FiLM'd -- safety is not conditioned on w_k, matching how CaRL's own
    reward value_head is also unconditioned on w_k) concatenated with
    e_t = obs["cumulative_cost"] (the cumulative discounted safety cost).
    This is a SEPARATE consumer of e_t from the actor's state augmentation
    above -- the critic judges the cost of what already happened; the actor
    (via features_aug) is what actually needs e_t to change its behavior.
    Trained via GAE on the dense cost signal from prism/morl/cvar_penalty.py,
    combined with the reward advantage in the trainer as
    A_total = A_reward - beta * A_cost -- see dpmorl_trainer.py.  This
    replaces v1's REINFORCE-style penalty entirely.

Trainable parameters:
    features_extractor (shared backbone CNN + measurements MLP)
    film_scale, film_shift
    style_policy_head
    action_proj, action_head (StochasticActionHead log_std)
    value_head (CaRL's original reward critic)
    cost_value_head (new CVaR cost critic, v2)

Frozen / bypassed:
    policy_head, dist_mu, dist_sigma (CaRL's original actor — replaced)
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn

from prism.models.base import PolicyOutput, PRISMPolicyBase
from prism.models.common.action_heads import StochasticActionHead

logger = logging.getLogger(__name__)


class CaRLPPOAdapter(PRISMPolicyBase):
    """
    PRISMPolicyBase adapter for CaRL's PPOPolicy with FiLM(w_k) + z_t actor
    conditioning (Problem 1).

    Args:
        carl_policy:      Instantiated CaRL PPOPolicy (already on device).
        reward_dim:       Style reward dimensions — 4 in PRISM.
        features_dim:     Output dimension of CaRL's features_extractor.
                          Matches config.features_dim (typically 512).
        policy_head_arch: Hidden layer sizes for the new style_policy_head.
                          Should match CaRL's original policy_head_arch so
                          the actor capacity is unchanged.
        action_dim:       Actor output dimension (2 for CaRL's 2D action space).
        init_log_std:     Initial log-std for StochasticActionHead.
    """

    def __init__(
        self,
        carl_policy: nn.Module,
        reward_dim: int = 4,
        features_dim: int = 512,
        policy_head_arch: Tuple[int, ...] = (256, 256),
        action_dim: int = 2,
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.policy = carl_policy
        self.reward_dim = reward_dim
        self._features_dim = features_dim

        # ── FiLM conditioning for w_k ─────────────────────────────────────────
        # gamma(w_k) scales features, beta(w_k) shifts them.
        # Initialised to identity: gamma≈1, beta≈0 when w_k≈0.
        self.film_scale = nn.Linear(reward_dim, features_dim)
        self.film_shift = nn.Linear(reward_dim, features_dim)
        nn.init.zeros_(self.film_scale.weight)
        nn.init.ones_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

        # ── Style-conditioned policy head — replaces CaRL's policy_head ───────
        # Input: features_dim (FiLM output) + reward_dim (z_t concat) + 1 (e_t concat)
        # e_t (v2): the actor is state-augmented with cumulative safety cost,
        # not just the cost critic -- it needs to be able to act more
        # conservatively as e_t approaches the VaR threshold nu, exactly the
        # motivation for state augmentation in the first place.
        in_dim = features_dim + reward_dim + 1
        layers: list[nn.Module] = []
        for out_dim in policy_head_arch:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.style_policy_head = nn.Sequential(*layers)

        # Linear projection from policy head output to action mean
        self.action_proj = nn.Linear(in_dim, action_dim)

        # PRISM stochastic action head (diagonal Gaussian)
        self.action_head = StochasticActionHead(action_dim, init_log_std)

        # ── CVaR cost critic (v2) — reads RAW features (no FiLM/w_k) + e_t ─────
        # Style-independent: safety is not a function of driving preference.
        cost_in_dim = features_dim + 1
        cost_layers: list[nn.Module] = []
        for out_dim in policy_head_arch:
            cost_layers += [nn.Linear(cost_in_dim, out_dim), nn.ReLU()]
            cost_in_dim = out_dim
        cost_layers.append(nn.Linear(cost_in_dim, 1))
        self.cost_value_head = nn.Sequential(*cost_layers)

        # Preference vector w_k — set per policy via set_w_k()
        self.register_buffer("_w_k", torch.zeros(reward_dim))

        # Freeze CaRL's original actor components (replaced above)
        for name, param in carl_policy.named_parameters():
            if any(name.startswith(p) for p in ("policy_head", "dist_mu", "dist_sigma")):
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Per-policy setup
    # ------------------------------------------------------------------

    def set_w_k(self, w_k: torch.Tensor) -> None:
        """Set the preference vector for this policy.  Called by run_stage2."""
        self._w_k.data.copy_(w_k.to(self._w_k.device))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: dict,
        actions: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """
        Args:
            obs:     observation dict from PRISMEnv.  Must contain
                     "bev_semantics", "measurements", "value_measurements",
                     "cumulative_cost".
            actions: if provided, evaluate log_prob of these actions (PPO update).
                     If None, sample from the policy (rollout collection).
        Returns:
            PolicyOutput(action, log_prob, entropy, value, cost_value)
        """
        device = self.film_scale.weight.device

        # ── Shared feature extraction (BEV CNN + measurements MLP) ───────────
        # Computed ONCE and reused for the actor, reward critic, and cost
        # critic below -- CaRL's own get_value(obs) recomputes this from
        # scratch internally, so we call value_head directly instead (see
        # the reward-value line further down) to avoid a redundant CNN pass.
        features = self.policy.get_features(obs)        # (batch, features_dim)
        batch = features.shape[0]

        # ── z_t from obs["value_measurements"] ───────────────────────────────
        z_t = obs.get("value_measurements")
        if z_t is None:
            z_t = torch.zeros(batch, self.reward_dim, device=device)
        else:
            z_t = z_t.float().to(device)

        # ── e_t from obs["cumulative_cost"] (v2 state augmentation) ──────────
        # Fed to BOTH the actor (below, alongside z_t) and the cost critic
        # (further down) -- the actor needs to be able to act more
        # conservatively as e_t approaches nu, not just have its cost judged
        # after the fact by a critic it can't see.
        e_t = obs.get("cumulative_cost")
        if e_t is None:
            e_t = torch.zeros(batch, 1, device=device)
        else:
            e_t = e_t.float().to(device)
            if e_t.dim() == 1:
                e_t = e_t.unsqueeze(-1)

        # ── FiLM conditioning: scale and shift features with w_k ─────────────
        w_k = self._w_k.unsqueeze(0).expand(batch, -1)  # (batch, reward_dim)
        gamma = self.film_scale(w_k)                      # (batch, features_dim)
        beta = self.film_shift(w_k)                       # (batch, features_dim)
        features_styled = gamma * features + beta         # (batch, features_dim)

        # ── Concatenate z_t AND e_t, pass through new style policy head ──────
        # State augmentation per the CVaR v2 spec: s~_t = (s_t, z_t, e_t).
        features_aug = torch.cat([features_styled, z_t, e_t], dim=-1)  # (batch, features_dim+5)
        latent_pi = self.style_policy_head(features_aug)            # (batch, 256)
        action_mean = self.action_proj(latent_pi)                   # (batch, action_dim)
        act, log_prob, entropy = self.action_head(action_mean, actions)

        # ── Reward value from CaRL's unmodified critic weights ────────────────
        # (z_t already in value_measurements). Calls value_head directly with
        # the already-computed `features` instead of get_value(obs), which
        # would recompute get_features(obs) a second time.
        value_measurements = obs.get("value_measurements")
        if value_measurements is None:
            value_measurements = torch.zeros(batch, self.reward_dim, device=device)
        else:
            value_measurements = value_measurements.float().to(device)
        value = self.policy.value_head(
            torch.cat([features, value_measurements], dim=1)
        ).squeeze(-1)                                                # (batch,)

        # ── CVaR cost value: raw (unstyled) features + e_t, no w_k ────────────
        cost_value = self.cost_value_head(
            torch.cat([features, e_t], dim=-1)
        ).squeeze(-1)                                                 # (batch,)

        return PolicyOutput(
            action=act, log_prob=log_prob, entropy=entropy,
            value=value, cost_value=cost_value,
        )

    # ------------------------------------------------------------------
    # Trainable parameters
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """
        Shared backbone + new style actor components + CaRL critic.
        Excludes CaRL's original policy_head/dist_mu/dist_sigma (frozen above).
        """
        params = []
        # CaRL's shared backbone (features_extractor)
        params += list(self.policy.features_extractor.parameters())
        if hasattr(self.policy, "lstm"):
            params += list(self.policy.lstm.parameters())
        # CaRL's critic (value_head) — unchanged, keep training
        params += list(self.policy.value_head.parameters())
        # New style conditioning
        params += list(self.film_scale.parameters())
        params += list(self.film_shift.parameters())
        params += list(self.style_policy_head.parameters())
        params += list(self.action_proj.parameters())
        params += list(self.action_head.parameters())
        # CVaR cost critic (v2) — separate weights, style-independent
        params += list(self.cost_value_head.parameters())
        return iter(params)

    def cost_critic_parameters(self) -> Iterator[nn.Parameter]:
        return iter(list(self.cost_value_head.parameters()))
