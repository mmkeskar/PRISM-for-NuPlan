"""
PRISM two-tier safety cost c_t.

    c_t = c_outcome + sum_j(c_lead_j)

Tier 1 — Outcome events (large, infrequent):
    Fired when an outcome actually occurs (collision, red light, off-road, …).
    Weight W_i comes from outcome_weights in hyperparams.json.

Tier 2 — Indicator signals (smaller, continuous warning):
    Fired at every timestep where a safety indicator is active (TTC < 1.5s,
    THW < 2.0s, speed violation, blind spot occupancy, must-stop-ahead).
    Per-step cost = w_j * s_j(t) where s_j ∈ [0,1] is a graded severity signal.
    Weight w_j and cap cap_j come from hyperparams.json indicator_weights /
    indicator_caps, precomputed from expert rollouts.

Episode-level cap for persistent indicators:
    sum_t(c_lead_j) <= cap_j  (see Design Decision #3 in CLAUDE.md)
    Prevents cumulative indicator cost from exceeding outcome event cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ------------------------------------------------------------------
# Blind spot geometry constants (CLAUDE.md conventions)
# ------------------------------------------------------------------
_BLIND_SPOT_ANGLE_INNER = math.radians(45)    # ±45° from heading
_BLIND_SPOT_ANGLE_OUTER = math.radians(135)   # ±135° from heading
_BLIND_SPOT_LONG_M = 10.0                     # longitudinal depth [m]
_BLIND_SPOT_LAT_M = 4.0                       # half-width [m]

# Proximity shortcut for must-stop-ahead: skip distance computation for
# targets further than this. At 4s horizon × 20 m/s = 80m max look-ahead.
_MAX_LOOKAHEAD_M = 80.0


@dataclass
class SafetyCostComponents:
    c_outcome: float = 0.0
    c_lead: float = 0.0

    # Individual outcome flags (for logging)
    vru_collision: bool = False
    vehicle_collision: bool = False
    object_collision: bool = False
    wrong_direction: bool = False
    red_light: bool = False
    off_road: bool = False
    stop_sign: bool = False

    # Individual indicator flags (for logging)
    ttc_active: bool = False
    thw_active: bool = False
    speed_active: bool = False
    blind_spot_active: bool = False
    must_stop_ahead_active: bool = False   # red light OR stop sign within arrival horizon

    @property
    def total(self) -> float:
        return self.c_outcome + self.c_lead


class SafetyCostBuilder:
    """
    Stateful (per-episode) builder for the PRISM safety cost.

    Call reset() at the start of each episode, then compute() at each step.
    """

    def __init__(self, hp: Dict) -> None:
        self._hp = hp
        self._ind_weights: Dict[str, float] = hp.get("indicator_weights", {})
        self._ind_caps: Dict[str, float] = hp.get("indicator_caps", {})
        self._out_weights: Dict[str, float] = hp.get("outcome_weights", {})
        self._thresholds: Dict = hp.get("safety_thresholds", {
            "ttc_threshold_s": 1.5,
            "thw_threshold_s": 2.0,
        })
        # Episode accumulators for indicator cap enforcement
        self._acc: Dict[str, float] = {}
        # Per-sign state machine for stop sign violation detection
        # token -> {"stopped": bool}  (populated when sign enters approach window)
        self._stop_sign_state: Dict[str, Dict] = {}
        self.reset()

    def reset(self) -> None:
        self._acc = {ind: 0.0 for ind in self._ind_weights}
        self._stop_sign_state = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        ego_state,
        detection_cache,
        map_cache,
        had_vru_collision: bool,
        had_vehicle_collision: bool,
        had_object_collision: bool,
        is_off_road: bool,
        ran_red_light: bool,
        had_wrong_direction: bool,
        v_lead: float,
        d_lead: float,
        has_lead: bool,
        v_limit: float,
        v_lat_ego: float = 0.0,
    ) -> SafetyCostComponents:
        comp = SafetyCostComponents()

        # ── Tier 1: outcome events ──────────────────────────────────────
        if had_vru_collision:
            comp.c_outcome += self._out_weights.get("vru_collision", 100)
            comp.vru_collision = True
        elif had_vehicle_collision:
            comp.c_outcome += self._out_weights.get("vehicle_collision", 80)
            comp.vehicle_collision = True
        elif had_object_collision:
            comp.c_outcome += self._out_weights.get("object_collision", 40)
            comp.object_collision = True

        if had_wrong_direction:
            comp.c_outcome += self._out_weights.get("wrong_direction", 100)
            comp.wrong_direction = True

        if ran_red_light:
            comp.c_outcome += self._out_weights.get("red_light_violation", 80)
            comp.red_light = True

        if is_off_road:
            comp.c_outcome += self._out_weights.get("drivable_area", 65)
            comp.off_road = True

        # Stop sign violation — detected internally via state machine
        if self._check_stop_sign_violation(ego_state, map_cache):
            comp.c_outcome += self._out_weights.get("stop_sign_violation", 70)
            comp.stop_sign = True

        # ── Tier 2: indicator signals ──────────────────────────────────
        v_ego = ego_state.dynamic_car_state.speed

        # TTC indicator — graded severity: s = (threshold - TTC) / threshold
        ttc = self._ttc(v_ego, v_lead, d_lead, has_lead)
        ttc_thresh = self._thresholds.get("ttc_threshold_s", 1.5)
        if ttc < ttc_thresh:
            comp.ttc_active = True
            s_ttc = max(0.0, (ttc_thresh - ttc) / ttc_thresh)
            comp.c_lead += self._cap_indicator("ttc", s_ttc)

        # THW indicator — graded severity: s = (threshold - THW) / threshold
        thw = self._thw(v_ego, d_lead, has_lead)
        thw_thresh = self._thresholds.get("thw_threshold_s", 2.0)
        if thw < thw_thresh:
            comp.thw_active = True
            s_thw = max(0.0, (thw_thresh - thw) / thw_thresh)
            comp.c_lead += self._cap_indicator("thw", s_thw)

        # Speed violation indicator — graded severity: s = (v_ego - v_limit) / v_limit
        if v_limit is not None and v_limit > 0 and v_ego > v_limit:
            comp.speed_active = True
            s_spd = max(0.0, (v_ego - v_limit) / v_limit)
            comp.c_lead += self._cap_indicator("speed", s_spd)

        # Blind spot indicator — fires only when ego is converging with the occupied side.
        # Directional gate: an object in the LEFT blind spot only matters when ego is
        # moving LEFT (v_lat_ego > lc_thresh), and vice versa.  Moving away is neutral.
        lc_thresh = self._thresholds.get("lane_change_lateral_vel_threshold_ms", 0.3)
        left_occ, right_occ = self._blind_spot_sides_occupied(ego_state, detection_cache)
        converging = (
            (left_occ and v_lat_ego > lc_thresh)
            or (right_occ and v_lat_ego < -lc_thresh)
        )
        if converging:
            comp.blind_spot_active = True
            comp.c_lead += self._cap_indicator("blind_spot", 1.0)

        # Must-stop-ahead indicator — fires if ego will arrive at a red traffic light
        # or a stop sign within the arrival horizon under constant acceleration.
        # Both controls require a full stop; the same temporal check applies to both.
        arrival_horizon_s = self._thresholds.get("red_light_arrival_horizon_s", 4.0)
        if self._must_stop_ahead(ego_state, map_cache, arrival_horizon_s):
            comp.must_stop_ahead_active = True
            comp.c_lead += self._cap_indicator("red_light", 1.0)

        return comp

    # ------------------------------------------------------------------
    # Stop sign violation — state-machine positive check
    # ------------------------------------------------------------------

    def _check_stop_sign_violation(
        self,
        ego_state,
        map_cache,
        approach_m: float = 5.0,
        stop_threshold_ms: float = 0.5,
    ) -> bool:
        """
        State-machine stop sign violation detector.

        Inspired by CARLA's RunStopSign criterion (Leaderboard 2.0).

        For each stop sign:
          Encounter  — sign centroid enters approach window (0 < d_lon < approach_m)
          Track      — record whether ego speed drops below stop_threshold_ms
          Verdict    — when ego passes the sign (d_lon ≤ 0):
                         if min speed during approach ≥ threshold → violation

        Stopping anywhere before passing the line satisfies the check; stopping after
        does not (the approach window tracking ends at the crossing event).

        Returns True on the step where a violation is detected.  Each sign fires at
        most once per episode (state cleaned up after verdict).
        """
        from nuplan.common.maps.maps_datatypes import StopLineType

        try:
            stop_lines = getattr(map_cache, "stop_lines", {})
            if not stop_lines:
                return False

            v_ego = ego_state.dynamic_car_state.speed
            cos_h = math.cos(ego_state.center.heading)
            sin_h = math.sin(ego_state.center.heading)
            ex, ey = ego_state.center.x, ego_state.center.y
            violated = False

            for token, sl in stop_lines.items():
                if sl.stop_line_type != StopLineType.STOP_SIGN:
                    continue

                cx, cy = sl.polygon.centroid.x, sl.polygon.centroid.y
                dx, dy = cx - ex, cy - ey
                d_lon = cos_h * dx + sin_h * dy   # positive = sign is ahead

                state = self._stop_sign_state.get(token)

                if state is None:
                    # Not yet tracking — start when sign enters approach window ahead
                    if 0 < d_lon < approach_m:
                        self._stop_sign_state[token] = {
                            "stopped": v_ego < stop_threshold_ms
                        }
                elif d_lon > 0:
                    # Still in approach zone — update stopped flag
                    if v_ego < stop_threshold_ms:
                        state["stopped"] = True
                else:
                    # Ego has passed the sign — issue verdict and clean up
                    if not state["stopped"]:
                        violated = True
                    del self._stop_sign_state[token]

            return violated
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Must-stop-ahead indicator (red light + stop sign, temporal check)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_time_to_arrival(
        distance: float,
        v_ego: float,
        a_ego: float,
    ) -> float:
        """
        Estimate time to travel `distance` metres under constant acceleration.

        Solves: d = v*t + 0.5*a*t².  Returns the first (smaller) positive root,
        which corresponds to the first time ego reaches the target.  Returns inf
        if unreachable (vehicle stops before the target when a < 0) or if distance
        is already zero.
        """
        if distance <= 0.0:
            return 0.0
        if abs(a_ego) < 1e-6:
            return distance / v_ego if v_ego > 1e-6 else float("inf")
        discriminant = v_ego ** 2 + 2.0 * a_ego * distance
        if discriminant < 0.0:
            return float("inf")   # vehicle decelerates to a stop before reaching target
        t = (-v_ego + math.sqrt(discriminant)) / a_ego
        return t if t > 0.0 else float("inf")

    @staticmethod
    def _find_red_connectors_on_route(map_cache) -> List:
        """Return list of red LaneConnectors on the ego route."""
        try:
            from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
            route_ids = set(map_cache.route_roadblock_ids)
            return [
                conn
                for conn_id, conn in map_cache.lane_connectors.items()
                if (
                    map_cache.traffic_lights.get(conn_id) == TrafficLightStatusType.RED
                    and conn.get_roadblock_id() in route_ids
                )
            ]
        except Exception:
            return []

    @staticmethod
    def _must_stop_ahead(
        ego_state,
        map_cache,
        arrival_horizon_s: float = 4.0,
    ) -> bool:
        """
        Fires if ego is estimated to arrive at a red traffic light OR a stop sign
        within `arrival_horizon_s` seconds, assuming constant acceleration.

        Both controls require a full stop; the same temporal arrival check applies.
        `_MAX_LOOKAHEAD_M` is used as a computation shortcut before computing the
        exact polygon distance.

        # Proximity-only fallback (commented out; may be restored for ablation):
        # return any(conn.polygon.distance(ego_pt) < 30.0
        #            for conn in red_connectors)
        """
        from shapely.geometry import Point
        try:
            v_ego = ego_state.dynamic_car_state.speed
            try:
                a_ego = float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x)
            except Exception:
                a_ego = 0.0
            ego_pt = Point(ego_state.center.x, ego_state.center.y)

            # Check red traffic light connectors
            for conn in SafetyCostBuilder._find_red_connectors_on_route(map_cache):
                d = conn.polygon.distance(ego_pt)
                if d > _MAX_LOOKAHEAD_M:
                    continue
                if SafetyCostBuilder._estimate_time_to_arrival(d, v_ego, a_ego) <= arrival_horizon_s:
                    return True

            # Check stop signs
            from nuplan.common.maps.maps_datatypes import StopLineType
            for sl in getattr(map_cache, "stop_lines", {}).values():
                if sl.stop_line_type != StopLineType.STOP_SIGN:
                    continue
                d = sl.polygon.distance(ego_pt)
                if d > _MAX_LOOKAHEAD_M:
                    continue
                if SafetyCostBuilder._estimate_time_to_arrival(d, v_ego, a_ego) <= arrival_horizon_s:
                    return True

        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cap_indicator(self, name: str, severity: float) -> float:
        """
        Apply graded indicator weight subject to episode cap.

        Per-step contribution = w_j * severity, capped so that the episode total
        does not exceed cap_j.  Returns the capped contribution for this step.
        """
        if name not in self._ind_weights:
            return 0.0
        cap = self._ind_caps.get(name, float("inf"))
        acc = self._acc.get(name, 0.0)
        remaining = max(0.0, cap - acc)
        contrib = min(self._ind_weights[name] * severity, remaining)
        self._acc[name] = acc + contrib
        return contrib

    @staticmethod
    def _ttc(v_ego: float, v_lead: float, d_lead: float, has_lead: bool) -> float:
        if not has_lead:
            return float("inf")
        closing = v_ego - v_lead
        if closing <= 0.0:
            return float("inf")
        return d_lead / max(closing, 1e-3)

    @staticmethod
    def _thw(v_ego: float, d_lead: float, has_lead: bool) -> float:
        if not has_lead or v_ego <= 0.0:
            return float("inf")
        return d_lead / max(v_ego, 1e-3)

    @staticmethod
    def _blind_spot_sides_occupied(
        ego_state,
        detection_cache,
    ) -> Tuple[bool, bool]:
        """
        Return (left_occupied, right_occupied): whether any tracked object is in
        the left or right blind spot zone respectively.

        Blind spot geometry (CLAUDE.md): ±45°–135° from ego heading, within
        10m longitudinal and 4m lateral of ego centre.

        The signed lateral offset (`d_lat_signed`) determines which side:
            d_lat_signed > 0  →  object is to the LEFT
            d_lat_signed < 0  →  object is to the RIGHT
        """
        ex, ey = ego_state.center.x, ego_state.center.y
        heading = ego_state.center.heading
        cos_h, sin_h = math.cos(heading), math.sin(heading)

        left_occ = False
        right_occ = False

        try:
            for obj in detection_cache.tracked_objects:
                ox, oy = obj.center.x, obj.center.y
                dx, dy = ox - ex, oy - ey
                d_lon = cos_h * dx + sin_h * dy          # positive = ahead
                d_lat_signed = -sin_h * dx + cos_h * dy  # positive = left

                # Must be behind ego and within longitudinal depth
                if d_lon > 0 or d_lon < -_BLIND_SPOT_LONG_M:
                    continue
                if abs(d_lat_signed) > _BLIND_SPOT_LAT_M:
                    continue

                # Angular check
                angle = math.atan2(abs(d_lat_signed), abs(d_lon))
                if not (_BLIND_SPOT_ANGLE_INNER <= angle <= _BLIND_SPOT_ANGLE_OUTER):
                    continue

                if d_lat_signed > 0:
                    left_occ = True
                else:
                    right_occ = True

                if left_occ and right_occ:
                    break   # no need to check further
        except Exception:
            pass

        return left_occ, right_occ
