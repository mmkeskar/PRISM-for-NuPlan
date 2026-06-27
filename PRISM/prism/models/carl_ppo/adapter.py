"""
Adapter that wraps CaRL's PPOPolicy into PRISMPolicyBase, with style
conditioning added to the actor (Problem 1).

CaRL's PPOPolicy architecture (from ppo_model.py):
    get_features(obs)          BEV CNN + measurements MLP → (batch, features_dim)
    policy_head(features)      MLP → latent_pi            [ACTOR]
    dist_mu/sigma(latent_pi)   → action distribution      [ACTOR]
    value_head(cat([features, value_measurements]))        [CRITIC]

Problem 1 fix — style conditioning on the actor:
    The original actor sees only `features` and has no knowledge of the
    style preference w_k or the cumulative return z_t.  We replace the
    actor path entirely:

    features = policy.get_features(obs)          (batch, features_dim)
    features_styled = FiLM(w_k, features)        (batch, features_dim)   ← w_k
    features_aug = cat([features_styled, z_t])   (batch, features_dim+4) ← z_t
    latent_pi = style_policy_head(features_aug)  (batch, 256)
    action_mean = action_proj(latent_pi)          (batch, action_dim)
    act, log_prob, entropy = StochasticActionHead(action_mean, actions)

    FiLM initialisation: scale bias=1, shift bias=0, all weights=0.
    This makes FiLM identity at the start of training (no modulation when
    w_k ≈ 0) and lets it learn gradually.

    The CaRL critic (value_head) is kept as-is.  z_t reaches it through
    obs["value_measurements"], which is the CaRL slot PRISM repurposes for
    z_t_normalised.  w_k conditioning on the critic is addressed separately
    (Problem 2).

Trainable parameters:
    features_extractor (shared backbone CNN + measurements MLP)
    film_scale, film_shift
    style_policy_head
    action_proj, action_head (StochasticActionHead log_std)
    value_head (CaRL's original critic)

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
        # Input: features_dim (FiLM output) + reward_dim (z_t concat)
        in_dim = features_dim + reward_dim
        layers: list[nn.Module] = []
        for out_dim in policy_head_arch:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.style_policy_head = nn.Sequential(*layers)

        # Linear projection from policy head output to action mean
        self.action_proj = nn.Linear(in_dim, action_dim)

        # PRISM stochastic action head (diagonal Gaussian)
        self.action_head = StochasticActionHead(action_dim, init_log_std)

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
                     "bev_semantics", "measurements", "value_measurements".
            actions: if provided, evaluate log_prob of these actions (PPO update).
                     If None, sample from the policy (rollout collection).
        Returns:
            PolicyOutput(action, log_prob, entropy, value)
        """
        device = self.film_scale.weight.device

        # ── Shared feature extraction (BEV CNN + measurements MLP) ───────────
        features = self.policy.get_features(obs)        # (batch, features_dim)
        batch = features.shape[0]

        # ── z_t from obs["value_measurements"] ───────────────────────────────
        z_t = obs.get("value_measurements")
        if z_t is None:
            z_t = torch.zeros(batch, self.reward_dim, device=device)
        else:
            z_t = z_t.float().to(device)

        # ── FiLM conditioning: scale and shift features with w_k ─────────────
        w_k = self._w_k.unsqueeze(0).expand(batch, -1)  # (batch, reward_dim)
        gamma = self.film_scale(w_k)                      # (batch, features_dim)
        beta = self.film_shift(w_k)                       # (batch, features_dim)
        features_styled = gamma * features + beta         # (batch, features_dim)

        # ── Concatenate z_t and pass through new style policy head ───────────
        features_aug = torch.cat([features_styled, z_t], dim=-1)   # (batch, features_dim+4)
        latent_pi = self.style_policy_head(features_aug)            # (batch, 256)
        action_mean = self.action_proj(latent_pi)                   # (batch, action_dim)
        act, log_prob, entropy = self.action_head(action_mean, actions)

        # ── Value from CaRL's unmodified critic (z_t already in value_meas.) ─
        value = self.policy.get_value(obs).squeeze(-1)              # (batch,)

        return PolicyOutput(action=act, log_prob=log_prob, entropy=entropy, value=value)

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
        return iter(params)
