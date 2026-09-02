#!/usr/bin/env python3
"""
Test 1 of the K-policy-divergence investigation: do f_0 and f_1's GRADIENTS
point in nearly the same direction across the actually-REACHABLE z region,
even though they're different functions by construction (different
preference weights)?

Why this matters: PPO's actor gradient ultimately follows the gradient of
the scalar reward it's given. If nabla f_0(z) and nabla f_1(z) are nearly
parallel everywhere the policy can actually reach (not just "different
somewhere in the full R^4 space"), PPO would pull both policies in nearly
the same direction regardless of which utility function is driving
training -- which would fully explain weak/absent divergence WITHOUT
needing to blame PPO itself, the environment, or the reward formulas. A low
cosine similarity would point the other way: the utility functions ARE
locally distinguishing directions where it matters, so a divergence problem
is downstream of Stage 1 (see check_linear_projection ablation / CHANGES.md
for the complementary test).

Uses REAL z points already logged during training -- each update's mean
z_T, straight from policy_k_metrics.jsonl -- not synthetic samples, and
needs no new training run.

Needs the prism package + torch (same environment as training), unlike
analyze_metrics.py which is stdlib-only -- run this from the PRISM repo
root, in the same conda env used for training.

Usage:
    python scripts/check_gradient_alignment.py \
        runs/dpmorl_only/prism_dpmorl_only_001/stage1/utility_fn_0.pth \
        runs/dpmorl_only/prism_dpmorl_only_001/stage1/utility_fn_1.pth \
        --metrics runs/dpmorl_only/prism_dpmorl_only_001/policy_0_metrics.jsonl \
                  runs/dpmorl_only/prism_dpmorl_only_001/policy_1_metrics.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from inspect_utility_functions import load_checkpoint, _DIM_NAMES
from analyze_metrics import load_runs


def extract_z_points(metrics_paths):
    """Pull every update's mean z_T (z_comfort/z_progress/z_lateral/
    z_spacing) out of one or more metrics.jsonl files -- these are REAL,
    reachable-region samples (each is a mean over the ~2-3 episodes that
    update collected), not synthetic. Uses the LAST run in each file if it
    contains multiple concatenated runs (matches analyze_metrics.py's own
    multi-run handling -- see its docstring)."""
    points = []
    for p in metrics_paths:
        runs = load_runs(Path(p))
        if not runs:
            print(f"  (no parseable records in {p}, skipping)", file=sys.stderr)
            continue
        updates = runs[-1]["updates"]
        for u in updates:
            if all(f"z_{d}" in u and u[f"z_{d}"] is not None for d in _DIM_NAMES):
                points.append([u[f"z_{d}"] for d in _DIM_NAMES])
    return np.array(points, dtype=np.float32)


def gradient_at(uf, z_np) -> np.ndarray:
    """nabla f(z) via autograd, in RAW z units -- the gradient that
    actually determines which direction a change in each style dimension
    moves this policy's utility, which is what an advantage signal has to
    pick up on downstream."""
    z = torch.tensor(z_np, dtype=torch.float32, requires_grad=True)
    val = uf(z)
    val.backward()
    return z.grad.detach().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs=2, help="exactly 2 utility_fn_*.pth paths (same convention as inspect_utility_functions.py)")
    parser.add_argument("--metrics", nargs="+", required=True, help="policy_*_metrics.jsonl paths to source real z points from (pooled across all given files)")
    parser.add_argument("--reward_dim", type=int, default=4)
    parser.add_argument("--max_points", type=int, default=2000, help="subsample if more points than this")
    args = parser.parse_args()

    loaded = []
    for i, spec in enumerate(args.checkpoints):
        path, pref, uf, as_loaded = load_checkpoint(spec, position=i, reward_dim=args.reward_dim)
        uf.eval()
        loaded.append((path, pref, uf))

    z_points = extract_z_points(args.metrics)
    if len(z_points) == 0:
        print("No z_T points found in the given metrics.jsonl files -- check paths / field names.", file=sys.stderr)
        sys.exit(1)
    if len(z_points) > args.max_points:
        idx = np.random.default_rng(0).choice(len(z_points), args.max_points, replace=False)
        z_points = z_points[idx]

    print(f"Pooled {len(z_points)} real z_T points from {len(args.metrics)} metrics file(s).")
    print(f"z range in this sample: min={z_points.min(0)}  max={z_points.max(0)}")

    (path0, _pref0, uf0), (path1, _pref1, uf1) = loaded
    print(f"f_0 = {path0.name}   f_1 = {path1.name}")

    cosines = []
    n_degenerate = 0
    for z in z_points:
        g0 = gradient_at(uf0, z)
        g1 = gradient_at(uf1, z)
        n0, n1 = np.linalg.norm(g0), np.linalg.norm(g1)
        if n0 < 1e-10 or n1 < 1e-10:
            n_degenerate += 1
            continue  # zero gradient at this point (saturated) -- not informative for direction
        cosines.append(float(np.dot(g0, g1) / (n0 * n1)))
    cosines = np.array(cosines)

    print(f"\n{'='*70}\nGradient cosine similarity: nabla f_0(z) vs nabla f_1(z), "
          f"over {len(cosines)} real z points ({n_degenerate} skipped, zero gradient)\n{'='*70}")
    if len(cosines) == 0:
        print("  All sampled points had a zero gradient on at least one function -- "
              "can't compute directions. That itself is a finding: the utility function(s) "
              "may be saturated across the entire reachable region sampled here.")
        return
    print(f"  mean={cosines.mean():.4f}  median={np.median(cosines):.4f}  std={cosines.std():.4f}")
    print(f"  min={cosines.min():.4f}  max={cosines.max():.4f}")
    print(f"  fraction > 0.9 (nearly parallel): {(cosines > 0.9).mean()*100:.1f}%")
    print(f"  fraction > 0.5: {(cosines > 0.5).mean()*100:.1f}%")
    print(f"  fraction < 0.0 (opposing): {(cosines < 0.0).mean()*100:.1f}%")

    print(f"\nRead this as: 1.0 = gradients point in EXACTLY the same direction at that z "
          f"(PPO would pull both policies the same way regardless of which utility function "
          f"drives training there -- explains weak divergence without needing to blame PPO, "
          f"the environment, or the reward formulas). 0.0 = orthogonal (independent, as "
          f"intended for two genuinely different preferences). Negative = actively opposing "
          f"(unusual for two supposedly different-preference functions -- worth a closer look "
          f"if common). If the mean is high (roughly >0.7-0.8) across most of the reachable "
          f"region sampled here, that's strong evidence the utility-function layer itself is "
          f"why the two policies aren't diverging more, independent of anything found by the "
          f"linear-projection ablation (see CHANGES.md) -- the two tests are complementary, "
          f"not redundant.")


if __name__ == "__main__":
    main()
