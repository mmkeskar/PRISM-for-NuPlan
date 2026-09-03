#!/usr/bin/env python3
"""
One-off diagnostic: is DynamicCarState.rear_axle_acceleration_2d (and
rear_axle_velocity_2d) reported in the ego vehicle's LOCAL/BODY frame
(x=longitudinal, y=lateral -- no rotation needed) or in the GLOBAL/MAP
frame (needs rotation by heading to get longitudinal/lateral)?

This matters because prism/env/nuplan_env.py's PRISMRewardBuilder reads
rear_axle_acceleration_2d.x/.y DIRECTLY as (a_lon, a_lat), with no
rotation -- while compute_hyperparams.py's _safe_kinematics() rotates by
heading first. The two disagree, and getting it wrong either way is worse
than leaving it alone (see CHANGES.md for the full account) -- this
script settles it empirically instead of guessing.

Ground truth check: `speed` is an unambiguous scalar (forward speed
magnitude). If rear_axle_velocity_2d.x alone already matches `speed`
closely, the vectors are LOCAL frame (no rotation needed -- nuplan_env.py
is right). If instead the HEADING-ROTATED v_lon = v2d.x*cos(h)+v2d.y*sin(h)
matches `speed` while the raw v2d.x does not, the vectors are GLOBAL frame
(compute_hyperparams.py's rotation is right, nuplan_env.py needs fixing).

Usage (run on the lab machine, in the training conda env, from a script
or notebook with a live `scenario`/`ego_state` sequence available -- e.g.
adapt the loop below to however you already iterate ego states, such as
via PRISMEnv.reset()/step() during a short rollout, or
scenario.get_ego_state_at_iteration(i) as compute_hyperparams.py does):

    from scripts.check_accel_frame import report_frame_convention
    report_frame_convention(ego_states)  # list of EgoState, in order
"""

from __future__ import annotations

import math
from typing import List


def report_frame_convention(ego_states: List, n_check: int = 20) -> None:
    print(f"{'i':>3}  {'speed':>8}  {'v2d.x (raw)':>12}  {'v2d.y (raw)':>12}  "
          f"{'v_lon (rotated)':>16}  {'|raw.x - speed|':>16}  {'|rot - speed|':>14}")
    raw_err_sum = 0.0
    rot_err_sum = 0.0
    n = 0
    for i, s in enumerate(ego_states[:n_check]):
        try:
            dcs = s.dynamic_car_state
            speed = float(dcs.speed)
            v2d = dcs.rear_axle_velocity_2d
            heading = float(s.center.heading)
            cos_h, sin_h = math.cos(heading), math.sin(heading)
            v_lon_rotated = v2d.x * cos_h + v2d.y * sin_h
            raw_err = abs(abs(v2d.x) - speed)
            rot_err = abs(abs(v_lon_rotated) - speed)
            raw_err_sum += raw_err
            rot_err_sum += rot_err
            n += 1
            print(f"{i:>3}  {speed:>8.3f}  {v2d.x:>12.4f}  {v2d.y:>12.4f}  "
                  f"{v_lon_rotated:>16.4f}  {raw_err:>16.4f}  {rot_err:>14.4f}")
        except Exception as e:
            print(f"{i:>3}  (failed: {e})")

    if n == 0:
        print("\nNo usable ego states -- nothing to report.")
        return

    print(f"\nmean |raw v2d.x - speed|      = {raw_err_sum / n:.4f}")
    print(f"mean |rotated v_lon - speed|  = {rot_err_sum / n:.4f}")
    print(
        "\nRead this as: whichever row is closer to 0 tells you the real "
        "convention. If 'raw' is near 0, rear_axle_velocity_2d/acceleration_2d "
        "are already LOCAL/BODY frame -- nuplan_env.py's direct .x/.y read is "
        "correct as-is, no rotation needed (the jerk_lat=0 mystery needs a "
        "different explanation, e.g. the original angular_velocity-propagation "
        "hypothesis). If 'rotated' is near 0 instead, the fields are GLOBAL "
        "frame -- nuplan_env.py needs the same cos_h/sin_h rotation "
        "compute_hyperparams.py's _safe_kinematics() already does, and every "
        "run's r_comfort/j_lon this session has been computing a frame-mixed "
        "quantity, not true longitudinal jerk."
    )


if __name__ == "__main__":
    print(__doc__)
