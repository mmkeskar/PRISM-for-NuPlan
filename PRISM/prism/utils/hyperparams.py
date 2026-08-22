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

    required_scaling = ["sigma_j_sq", "beta", "gamma_a", "phi", "tau"]
    for key in required_scaling:
        if key not in hp["reward_scaling"]:
            raise KeyError(f"reward_scaling is missing key: '{key}'")
