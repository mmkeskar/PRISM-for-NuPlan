"""
Load and validate the cached PRISM hyperparameter file.
"""
import json
from pathlib import Path
from typing import Any, Dict


def load_hyperparams(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"hyperparams.json not found at {path}.\n"
            "Run: python compute_hyperparams.py --output_path hyperparams.json ..."
        )
    with open(p) as f:
        hp = json.load(f)
    _validate(hp)
    return hp


def _validate(hp: Dict[str, Any]) -> None:
    required_top = [
        "reward_scaling", "lead_times",
        "indicator_weights", "indicator_caps", "outcome_weights",
        "z_normalisation", "alpha_curriculum", "safety_thresholds",
    ]
    for key in required_top:
        if key not in hp:
            raise KeyError(f"hyperparams.json is missing required key: '{key}'")

    required_scaling = ["sigma_j_sq", "beta", "phi", "tau"]
    for key in required_scaling:
        if key not in hp["reward_scaling"]:
            raise KeyError(f"reward_scaling is missing key: '{key}'")

    # beta is now a per-regime dict (see compute_hyperparams.py's
    # compute_reward_scaling) rather than a single scalar -- required so a
    # caller reading hp["reward_scaling"]["beta"]["free_flow"] (etc.) fails
    # loudly on a stale, pre-per-regime hyperparams.json instead of raising
    # a confusing TypeError deep inside rewards.py.
    required_regimes = ["free_flow", "car_following", "congested"]
    beta = hp["reward_scaling"]["beta"]
    if not isinstance(beta, dict):
        raise TypeError(
            "reward_scaling.beta must be a per-regime dict "
            f"({required_regimes}) -- got {type(beta).__name__}. "
            "Delete hyperparams.json and rerun compute_hyperparams.py."
        )
    for regime in required_regimes:
        if regime not in beta:
            raise KeyError(f"reward_scaling.beta is missing regime key: '{regime}'")
