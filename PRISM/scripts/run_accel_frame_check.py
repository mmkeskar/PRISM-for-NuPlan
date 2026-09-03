#!/usr/bin/env python3
"""
Loads a handful of real nuPlan scenarios -- same scenario-building path as
compute_hyperparams.py's collect_expert_rollouts() (NuPlanScenarioBuilder +
ScenarioFilter + Sequential worker, identical kwargs) -- and runs
check_accel_frame.report_frame_convention() against real ego states.

Settles whether DynamicCarState.rear_axle_velocity_2d/acceleration_2d are
local (vehicle body frame, x=longitudinal/y=lateral, no rotation needed) or
global (map frame, needs rotation by heading) -- see CHANGES.md and
scripts/check_accel_frame.py's docstring for why this matters (the jerk_lat
fix is intentionally blocked on this).

Usage (on the lab machine, in the training conda env, from the PRISM repo
root):
    python scripts/run_accel_frame_check.py \
        --nuplan_data_root /path/to/nuplan/dataset

Use the mini dataset root for a quick check (matches lab.env's
NUPLAN_MINI_ROOT), or the full dataset root for more scenario variety.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
_CARL_ROOT = Path(__file__).parent.parent.parent / "nuPlan"
if str(_CARL_ROOT) not in sys.path:
    sys.path.insert(0, str(_CARL_ROOT))

from compute_hyperparams import _locate_map_root
from check_accel_frame import report_frame_convention


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nuplan_data_root", type=str, required=True)
    parser.add_argument("--n_scenarios", type=int, default=3,
                         help="how many real scenarios to sample and check")
    parser.add_argument("--n_states_per_scenario", type=int, default=20,
                         help="how many ego states per scenario to print/average")
    args = parser.parse_args()

    # Same imports/construction as compute_hyperparams.py's
    # collect_expert_rollouts() -- deliberately not reinventing scenario
    # loading here, to avoid a second, different bug confounding this check.
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
        NuPlanScenarioBuilder,
    )
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    map_root = _locate_map_root(args.nuplan_data_root)
    builder = NuPlanScenarioBuilder(
        data_root=args.nuplan_data_root,
        map_root=map_root,
        sensor_root=args.nuplan_data_root,
        db_files=None,
        map_version="nuplan-maps-v1.0",
        include_cameras=False,
        max_workers=1,
        verbose=False,
    )
    sf = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
        expand_scenarios=False,
        remove_invalid_goals=True,
        shuffle=False,
    )
    worker = Sequential()
    scenarios = builder.get_scenarios(sf, worker)
    if not scenarios:
        raise RuntimeError(
            f"No scenarios found in {args.nuplan_data_root} -- "
            "check the path contains *.db files."
        )

    random.shuffle(scenarios)
    scenarios = scenarios[: args.n_scenarios]
    print(f"Checking {len(scenarios)} real scenario(s), "
          f"{args.n_states_per_scenario} ego states each.\n")

    for i, scenario in enumerate(scenarios):
        name = getattr(scenario, "scenario_name", None) or f"scenario_{i}"
        n_iter = min(args.n_states_per_scenario, scenario.get_number_of_iterations())
        ego_states = [scenario.get_ego_state_at_iteration(j) for j in range(n_iter)]
        print(f"\n{'=' * 78}\nScenario {i + 1}/{len(scenarios)}: {name}  ({n_iter} states)\n{'=' * 78}")
        report_frame_convention(ego_states, n_check=n_iter)


if __name__ == "__main__":
    main()
