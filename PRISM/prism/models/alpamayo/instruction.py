"""
Instruction string builder for Alpamayo z_t / e_t injection.

Encodes PRISM's per-policy state into natural language at every timestep so
the Alpamayo transformer can condition its actions on the driving style
objective (w_k, static), the current episode progress (z_t, dynamic), and
(v2 CVaR refactor) the current accumulated safety cost relative to the VaR
threshold (e_t / nu, dynamic).

Design follows π₀.5 (arXiv:2504.16054): proprioceptive / task state is
serialized as text and prepended to the VLM input.  Text injection exploits
Alpamayo's pretrained number and driving-vocabulary representations rather
than requiring a new projector trained from scratch.

The instruction has two parts:
    Preamble  — built once from w_k at policy construction time.
                Describes which driving objectives this policy prioritises.
                Re-built only if w_k changes (never in normal training).

    Dynamic   — rebuilt every step from obs["value_measurements"] (z_t) and
                obs["cumulative_cost"] (e_t).  z_t is EMA-normalised so the
                model sees direction/relative magnitude without needing the
                absolute scale; e_t is raw (same units as nu -- see
                prism/env/nuplan_env.py). Three decimal places used
                throughout (sufficient to distinguish 0.023 from 0.031 after
                EMA normalisation).

Source for z_t: obs["value_measurements"] (shape: (4,)).
Source for e_t: obs["cumulative_cost"] (shape: (1,)).
Both are written every step by PRISMEnv; no changes to PRISMEnv's obs
assembly are required beyond what it already does.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


_OBJ_NAMES = ["comfort", "progress", "lateral discipline", "spacing"]

_OBJ_DESCS = [
    "smooth ride, minimal jerk and harsh acceleration",
    "forward advancement, maintaining target speed",
    "lane centering, minimal unnecessary lane changes",
    "safe following distance from lead vehicle",
]


class AlpamayoInstructionBuilder:
    """
    Builds Alpamayo's per-step instruction string.

    Args:
        policy_id: index k of this policy (0 … K-1).
        w_k:       preference vector for this policy, shape (reward_dim,).
                   Used to determine priority order for the preamble.
    """

    def __init__(self, policy_id: int, w_k: np.ndarray) -> None:
        self._policy_id = policy_id
        self._preamble = self._build_preamble(policy_id, w_k)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(
        self,
        z_t_normalised: np.ndarray,
        e_t: Optional[float] = None,
        nu: Optional[float] = None,
        spatial_desc: str = "",
    ) -> str:
        """
        Return the full instruction string for one timestep.

        Args:
            z_t_normalised: (reward_dim,) EMA-normalised cumulative style
                            returns from obs["value_measurements"].
            e_t:            raw cumulative safety cost so far this episode,
                            from obs["cumulative_cost"] (v2). Pass None to
                            omit the safety-margin sentence (backward-compatible).
            nu:             current VaR threshold (v2), set via
                            AlpamayoAdapter.set_nu() once per training
                            update. Only used if e_t is also provided.
            spatial_desc:   compact natural-language spatial state string
                            produced by spatial_description.build_spatial_description().
                            Inserted between the style preamble and the z_t string.
                            Pass "" (default) to omit (backward-compatible).
        """
        zt_string = self._build_dynamic(z_t_normalised)
        parts = [self._preamble]
        if spatial_desc:
            parts.append(spatial_desc)
        parts.append(zt_string)
        if e_t is not None:
            parts.append(self._build_safety_margin(e_t, nu))
        return "\n".join(parts)

    def update(self, policy_id: int, w_k: np.ndarray) -> None:
        """Rebuild the preamble — call if w_k ever changes."""
        self._policy_id = policy_id
        self._preamble = self._build_preamble(policy_id, w_k)

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_preamble(k: int, w_k: np.ndarray) -> str:
        priority = np.argsort(w_k)[::-1]
        names = _OBJ_NAMES
        return (
            f"You are driving style k={k}, which prioritizes "
            f"{names[priority[0]]} above all else, followed by "
            f"{names[priority[1]]}, {names[priority[2]]}, "
            f"and {names[priority[3]]}."
        )

    @staticmethod
    def _build_dynamic(z: np.ndarray) -> str:
        return (
            f"Cumulative style returns this episode (normalised):\n"
            f"- Comfort ({_OBJ_DESCS[0]}): {z[0]:.3f}\n"
            f"- Progress ({_OBJ_DESCS[1]}): {z[1]:.3f}\n"
            f"- Lateral discipline ({_OBJ_DESCS[2]}): {z[2]:.3f}\n"
            f"- Spacing ({_OBJ_DESCS[3]}): {z[3]:.3f}"
        )

    @staticmethod
    def _build_safety_margin(e_t: float, nu: Optional[float]) -> str:
        """Cumulative safety cost so far, and margin to the VaR threshold (v2)."""
        if nu is None:
            return f"Cumulative safety cost this episode: {e_t:.3f}."
        margin = nu - e_t
        return (
            f"Cumulative safety cost this episode: {e_t:.3f} "
            f"(safety threshold: {nu:.3f}, margin remaining: {margin:+.3f})."
        )
