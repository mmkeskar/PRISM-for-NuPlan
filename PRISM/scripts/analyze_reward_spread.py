#!/usr/bin/env python3
"""
Reward-spread diagnostic: are individual reward sub-components plateauing
near their ceiling (little room left to differentiate one action from
another), independent of anything PPO training is doing?

Motivation: the same reward FORMULA can be a bug (something computed wrong)
or a genuine DESIGN property (the formula is correct, but the environment
is such that a reasonably competent policy sits near a component's ceiling
most of the time, leaving little variance for the advantage signal to pick
up on). This script can't tell those apart automatically, but it makes the
distinction visible per-component instead of only seeing it indirectly
through PPO's aggregate reward_advantage_std.

Requires a run with --log_reward_components (see scripts/train.py) --
without it, rc_*_mean/rc_*_std fields won't be in the metrics.jsonl records
and this script has nothing to report. Stdlib only, same as
analyze_metrics.py -- run this anywhere against the raw log file.

Usage:
    python scripts/analyze_reward_spread.py runs/dpmorl_only/policy_0_metrics.jsonl
    python scripts/analyze_reward_spread.py runs/dpmorl_only/policy_*_metrics.jsonl
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_metrics import load_runs

# (component, ceiling, floor) -- ceiling/floor are None for components with
# no natural fixed bound (jerk, ttc), where "headroom" isn't a meaningful
# concept and we just report raw statistics instead.
_COMPONENTS = [
    ("r_speed", 1.0, 0.0),
    ("r_dev", 1.0, 0.3),
    ("r_heading", 1.0, 0.0),
    ("jerk_lon", None, None),
    ("jerk_lat", None, None),
    ("ttc", None, None),
    # Raw kinematic/regime diagnostics (not normalised rewards -- no fixed
    # ceiling/floor, reported as raw stats only). Added alongside the
    # traffic-aware v_des / per-regime beta calibration so a persistently
    # low r_speed can be traced to "v_des target got harder" vs "beta wasn't
    # recalibrated for it" vs "policy is genuinely not keeping up". See
    # CHANGES.md (Part 3.1). Per-regime frac_regime_* breakdown is reported
    # separately by dpmorl_trainer.py (regime is categorical, not a mean).
    ("v_ego", None, None),
    ("v_des", None, None),
    ("shortfall", None, None),
    ("beta", None, None),
    ("n_surrounding_agents", None, None),
]


def report_file(path: Path) -> None:
    runs = load_runs(path)
    if not runs:
        print(f"  {path}: no parseable records, skipping")
        return
    updates = runs[-1]["updates"]
    if not updates:
        print(f"  {path}: no update records, skipping")
        return

    print(f"\n{'='*78}\n{path}\n{'='*78}")

    n_early = len(updates) // 2
    early, late = updates[:n_early], updates[n_early:]

    found_any = False
    for name, ceiling, floor in _COMPONENTS:
        mean_key, std_key = f"rc_{name}_mean", f"rc_{name}_std"
        vals_mean = [u[mean_key] for u in updates if u.get(mean_key) is not None]
        vals_std = [u[std_key] for u in updates if u.get(std_key) is not None]
        if not vals_mean:
            continue
        found_any = True

        overall_mean = statistics.mean(vals_mean)
        overall_std = statistics.mean(vals_std) if vals_std else float("nan")

        early_means = [u[mean_key] for u in early if u.get(mean_key) is not None]
        late_means = [u[mean_key] for u in late if u.get(mean_key) is not None]
        early_m = statistics.mean(early_means) if early_means else float("nan")
        late_m = statistics.mean(late_means) if late_means else float("nan")

        print(f"\n-- {name} --")
        print(f"  mean over run: {overall_mean:.4f}   typical per-step std: {overall_std:.4f}")
        print(f"  early-half mean: {early_m:.4f}   late-half mean: {late_m:.4f}")

        if ceiling is not None:
            headroom = ceiling - overall_mean
            floor_room = overall_mean - floor
            print(f"  headroom to ceiling ({ceiling}): {headroom:.4f}   "
                  f"room above floor ({floor}): {floor_room:.4f}")
            if headroom < 0.05 and overall_std < 0.05:
                print(f"  >> FLAG: sitting within 0.05 of ceiling with std < 0.05 -- "
                      f"plateaued/saturated near its MAXIMUM. Little room left for "
                      f"this component to differentiate one action from another; "
                      f"PPO's advantage signal can only reflect what's left in the "
                      f"remaining {headroom:.4f} of headroom.")
            elif floor_room < 0.05 and overall_std < 0.05:
                print(f"  >> FLAG: sitting within 0.05 of its floor ({floor}) with "
                      f"std < 0.05 -- plateaued/saturated near its MINIMUM. This is "
                      f"the more concerning direction: it usually means the policy is "
                      f"persistently far from what this component wants (e.g. "
                      f"consistently well below desired speed), not that the reward "
                      f"is well-tuned and maxed out.")
            elif overall_std < 0.02:
                print(f"  >> FLAG: std < 0.02 regardless of mean level -- very little "
                      f"per-step spread, whatever the typical value is.")
        else:
            print(f"  (no fixed ceiling for this component -- reporting raw stats only)")

    # Regime fraction breakdown (categorical -- see dpmorl_trainer.py's
    # regime_stats, computed separately from the rc_*_mean/std loop above
    # since a regime label isn't a number to average).
    regime_keys = [f"frac_regime_{r}" for r in ("free_flow", "car_following", "congested")]
    if any(u.get(k) is not None for u in updates for k in regime_keys):
        print("\n-- regime fractions (mean over run) --")
        for key in regime_keys:
            vals = [u[key] for u in updates if u.get(key) is not None]
            if vals:
                print(f"  {key}: {statistics.mean(vals):.4f}")

    if not found_any:
        print("  No rc_*_mean/rc_*_std fields found -- this run wasn't started with "
              "--log_reward_components. Nothing to report.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        report_file(Path(arg))


if __name__ == "__main__":
    main()
