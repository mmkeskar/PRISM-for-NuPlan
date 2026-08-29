#!/usr/bin/env python3
"""
Inspect saved UtilityFunction checkpoints to check whether K policies'
utility functions are actually behaviorally true to their preference
vectors -- a DIRECT check on the weights/behavior themselves, not an
indirect inference from z_T trajectories (which are confounded by
training stability, exploration noise, episode randomness, etc.).

In "preferences" mode (the default -- see run_stage1() in scripts/train.py),
the utility function's neural network weights are NEVER updated during
Stage 2: they're set once at construction by init_utility_functions_from_preferences()
and only ever called inside torch.no_grad() afterward. So this script is
checking whether that one-time construction actually produced meaningfully
different, preference-correct functions -- independent of anything that
happened (or went wrong) during RL training.

Usage (position-based preferences, standard K<=4 convention: 0=comfort,
1=progress, 2=lateral, 3=spacing, from scripts/train.py's
_get_preference_vectors):
    python scripts/inspect_utility_functions.py \
        runs/dpmorl_only/prism_dpmorl_only_001/stage1/utility_fn_0.pth \
        runs/dpmorl_only/prism_dpmorl_only_001/stage1/utility_fn_1.pth

Usage (explicit preferences, if yours don't follow the standard convention):
    python scripts/inspect_utility_functions.py \
        path/to/utility_fn_0.pth:0.55,0.15,0.15,0.15 \
        path/to/utility_fn_1.pth:0.15,0.55,0.15,0.15

Needs the prism package + torch (same environment as training), unlike
analyze_metrics.py which is stdlib-only -- run this from the PRISM repo
root, in the same conda env used for training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from prism.morl.utility_functions import UtilityFunction

_STANDARD_K4_PREFS = [
    [0.55, 0.15, 0.15, 0.15],  # comfort
    [0.15, 0.55, 0.15, 0.15],  # progress
    [0.15, 0.15, 0.55, 0.15],  # lateral
    [0.15, 0.15, 0.15, 0.55],  # spacing
]
_DIM_NAMES = ["comfort", "progress", "lateral", "spacing"]


def load_checkpoint(spec: str, position: int, reward_dim: int = 4):
    """spec is either 'path' (preference inferred from position, standard
    K<=4 convention) or 'path:w1,w2,w3,w4' (explicit preference)."""
    if ":" in spec:
        path_str, pref_str = spec.rsplit(":", 1)
        pref = [float(x) for x in pref_str.split(",")]
    else:
        path_str = spec
        if position >= len(_STANDARD_K4_PREFS):
            raise ValueError(
                f"No standard preference for position {position} (only defined for "
                f"K<=4) -- pass explicit 'path:w1,w2,w3,w4' for this checkpoint."
            )
        pref = _STANDARD_K4_PREFS[position]

    path = Path(path_str)
    uf = UtilityFunction(reward_dim=reward_dim)
    # strict=False: checkpoints saved before the normalization-freeze fix
    # (_calls_seen buffer) won't have that key -- load everything else and
    # let _calls_seen fall back to its freshly-constructed default (0)
    # rather than crashing on a missing key.
    missing, unexpected = uf.load_state_dict(
        torch.load(path, map_location="cpu"), strict=False
    )
    if missing:
        print(f"  (note: {path.name} checkpoint predates these buffers, using defaults: {missing})")
    uf.eval()

    # Preserve the AS-LOADED stats for reporting before overriding them below.
    as_loaded = {
        "calls_seen": uf._calls_seen.item(),
        "min_val": uf._min_val.tolist(),
        "max_val": uf._max_val.tolist(),
    }

    # Freeze the normalization range for inspection purposes regardless of
    # what _calls_seen says -- otherwise the sensitivity test below would
    # itself perturb _min_val/_max_val between its own sequential calls
    # (the exact moving-target problem this whole investigation is about),
    # corrupting the comparison it's trying to make.
    if torch.isfinite(uf._min_val).all() and torch.isfinite(uf._max_val).all():
        uf._calls_seen.fill_(uf._normalization_warmup_calls)
    else:
        print(f"  !! {path.name}: _min_val/_max_val are not finite (this checkpoint never "
              f"saw a real z value) -- falling back to a fixed [0, 50] range for the "
              f"sensitivity test below, since the real range is unknown.")
        uf._min_val.fill_(0.0)
        uf._max_val.fill_(50.0)
        uf._calls_seen.fill_(uf._normalization_warmup_calls)

    return path, pref, uf, as_loaded


def report(path: Path, declared_pref: list, uf: UtilityFunction, as_loaded: dict):
    print(f"\n{'='*78}\n{path}\n{'='*78}")
    print(f"declared preference (from CLI/convention): {declared_pref}")
    print(f"_pref_weights actually stored in the checkpoint: {uf._pref_weights.tolist()}")
    match = torch.allclose(uf._pref_weights, torch.tensor(declared_pref, dtype=torch.float32))
    print(f"  {'OK -- matches' if match else '!! MISMATCH -- does not match declared preference'}")

    print(f"\nnormalization range AS SAVED in the checkpoint (before this script pins it "
          f"for the sensitivity test below):")
    print(f"  _calls_seen={as_loaded['calls_seen']}  _min_val={as_loaded['min_val']}  "
          f"_max_val={as_loaded['max_val']}")

    # fc_in.weight column-wise mean |weight| -- how much each of the 4 input
    # dimensions is emphasized by the network's FIRST layer (biased at
    # construction by init_utility_functions_from_preferences() multiplying
    # each column by the preference weight, before clamping to max_weight).
    col_mag = uf.fc_in.weight.detach().abs().mean(dim=0)  # (reward_dim,)
    print(f"\nfc_in.weight mean |weight| per input dimension (higher = more emphasized):")
    for name, val in zip(_DIM_NAMES[:len(col_mag)], col_mag.tolist()):
        print(f"  {name:<10} {val:.5f}")
    top_dim = _DIM_NAMES[int(col_mag.argmax().item())]
    print(f"  fc_in most emphasizes: {top_dim}")


def sensitivity_test(path: Path, declared_pref: list, uf: UtilityFunction, base_z=None):
    """The direct behavioral check: hold z fixed, boost ONE dimension by
    +1.0 at a time, measure the utility gain. A preference-faithful
    function should show its OWN preferred dimension producing the largest
    gain. Uses the checkpoint's own normalization range, PINNED (by
    load_checkpoint(), before this runs) so this test's own sequential
    calls don't shift it mid-comparison."""
    if base_z is None:
        base_z = np.array([20.0, 20.0, 20.0, 20.0], dtype=np.float32)
    fn = uf.as_callable()
    base_val = fn(base_z)
    gains = {}
    for i, name in enumerate(_DIM_NAMES[:len(base_z)]):
        z_boost = base_z.copy()
        z_boost[i] += 1.0
        gains[name] = fn(z_boost) - base_val
    return gains


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help="one or more utility_fn_*.pth paths (see usage above)")
    parser.add_argument("--reward_dim", type=int, default=4)
    args = parser.parse_args()

    loaded = []
    for i, spec in enumerate(args.checkpoints):
        path, pref, uf, as_loaded = load_checkpoint(spec, position=i, reward_dim=args.reward_dim)
        report(path, pref, uf, as_loaded)
        loaded.append((path, pref, uf))

    if len(loaded) < 2:
        print("\n(only one checkpoint given -- skipping cross-policy comparison)")
        return

    print(f"\n{'='*78}\nSensitivity test: utility gain from +1.0 to each dimension "
          f"(base z=[20,20,20,20])\n{'='*78}")
    print(f"{'checkpoint':<20} | " + " | ".join(f"{d:>10}" for d in _DIM_NAMES) + " | preferred dim highest?")
    all_gains = []
    for path, pref, uf in loaded:
        gains = sensitivity_test(path, pref, uf)
        all_gains.append((path, pref, gains))
        own_dim = _DIM_NAMES[int(np.argmax(pref))]
        highest_dim = max(gains, key=gains.get)
        flag = "YES" if highest_dim == own_dim else f"no (highest was {highest_dim})"
        row = f"{path.stem:<20} | " + " | ".join(f"{gains[d]:>10.5f}" for d in _DIM_NAMES)
        print(f"{row} | {flag}")

    print(f"\nRead this as: for each checkpoint, is the LARGEST utility gain on the "
          f"dimension that checkpoint's own preference weights most heavily? If yes for "
          f"all checkpoints, the utility functions are behaviorally faithful to their "
          f"preferences -- any remaining lack of z_T divergence in real training is NOT "
          f"because the utility functions are the same, it's downstream (e.g. the actor "
          f"not exploiting the signal, or training instability). If it's the same "
          f"dimension winning for multiple checkpoints (or a dimension none of them "
          f"prefer), that's a real problem worth investigating at the utility-function "
          f"level specifically.")


if __name__ == "__main__":
    main()
