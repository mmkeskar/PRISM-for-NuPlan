"""
Driving regime detector.

Detects one of three regimes and returns the desired speed v_des:
    1. Congested     : v_lane_avg < 0.5 * v_limit  →  v_des = v_lane_avg
    2. Car-following : lead vehicle within horizon   →  v_des = v_lead
    3. Free-flow     : otherwise                     →  v_des = v_limit

Check order is congested → car-following → free-flow.
This prevents a slow lead vehicle from triggering car-following when the
broader lane is already congested (Key Design Decision #6 in CLAUDE.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np


class Regime(Enum):
    CONGESTED = auto()
    CAR_FOLLOWING = auto()
    FREE_FLOW = auto()


@dataclass
class RegimeResult:
    regime: Regime
    v_des: float       # desired speed [m/s]
    v_lead: float      # lead vehicle speed [m/s], 0 if no lead
    d_lead: float      # longitudinal distance to lead vehicle [m], inf if no lead
    has_lead: bool     # whether a lead vehicle was detected
    v_lane_avg: float  # average speed of nearby vehicles [m/s]
    v_limit: float     # speed limit of current lane [m/s]
    n_surrounding_agents: int = 0  # vehicles used for v_lane_avg / the
                                    # free-flow percentile target -- exposed
                                    # so callers can log how often the
                                    # percentile estimate is running on a
                                    # thin sample (see CHANGES.md)


class RegimeDetector:
    """
    Extracts regime and desired speed from the nuPlan simulation state.

    Parameters
    ----------
    congestion_speed_fraction : float
        Threshold fraction of v_limit below which lane is considered congested.
    observation_horizon_m : float
        Longitudinal look-ahead and look-behind distance for lane traffic [m].
    lead_lateral_tolerance_m : float
        Maximum lateral offset for an object to count as in the ego lane [m].
    flow_percentile : float
        Free-flow v_des tracks this percentile of surrounding-traffic speed
        (not the posted limit alone) -- progress captures swiftness/
        efficiency of travel, not "match the median" (a comfort-style
        target). 80th percentile: keep pace with the faster portion of
        traffic, not just the typical car. Reasoned starting point, not
        precisely tuned -- see CHANGES.md.
    min_agents_for_percentile : int
        Minimum surrounding-vehicle count before trusting the percentile
        estimate; below this, falls back to v_limit (same fallback the
        zero-agent case already used). A percentile over 2-3 vehicles is
        essentially "the fastest one" -- too noisy/outlier-sensitive to
        use as a moving target. Reasoned starting point, not precisely
        tuned.
    """

    DEFAULT_SPEED_LIMIT_MPS: float = 13.89  # 50 km/h fallback

    def __init__(
        self,
        congestion_speed_fraction: float = 0.5,
        observation_horizon_m: float = 50.0,
        lead_lateral_tolerance_m: float = 2.5,
        flow_percentile: float = 80.0,
        min_agents_for_percentile: int = 4,
    ) -> None:
        self._cong_frac = congestion_speed_fraction
        self._horizon = observation_horizon_m
        self._lat_tol = lead_lateral_tolerance_m
        self._flow_percentile = flow_percentile
        self._min_agents = min_agents_for_percentile

    def detect(
        self,
        ego_state,        # nuPlan EgoState
        map_cache,        # MapCache from CaRL
        detection_cache,  # DetectionCache from CaRL
    ) -> RegimeResult:
        v_ego = ego_state.dynamic_car_state.speed
        v_limit = self._get_speed_limit(ego_state, map_cache)

        v_lead, d_lead, has_lead = self._find_lead(ego_state, detection_cache)
        surrounding_speeds = self._surrounding_speeds(ego_state, detection_cache)
        n_surrounding = len(surrounding_speeds)
        v_lane_avg = float(np.mean(surrounding_speeds)) if surrounding_speeds else self.DEFAULT_SPEED_LIMIT_MPS

        # 1. Congested?
        if v_lane_avg < self._cong_frac * v_limit:
            return RegimeResult(
                regime=Regime.CONGESTED,
                v_des=max(v_lane_avg, 0.3),  # keep positive to avoid division by zero
                v_lead=v_lead,
                d_lead=d_lead,
                has_lead=has_lead,
                v_lane_avg=v_lane_avg,
                v_limit=v_limit,
                n_surrounding_agents=n_surrounding,
            )

        # 2. Car-following?
        if has_lead and d_lead < self._horizon:
            return RegimeResult(
                regime=Regime.CAR_FOLLOWING,
                v_des=max(v_lead, 0.0),
                v_lead=v_lead,
                d_lead=d_lead,
                has_lead=has_lead,
                v_lane_avg=v_lane_avg,
                v_limit=v_limit,
                n_surrounding_agents=n_surrounding,
            )

        # 3. Free-flow -- traffic-aware: track the fast portion of
        # surrounding traffic, not just the posted limit, since progress
        # is about swiftness/efficiency of travel, not "match the median"
        # (a comfort-style target already captured elsewhere). Falls back
        # to v_limit when too few vehicles are nearby to trust a
        # percentile estimate. max() with v_limit means this never pulls
        # v_des BELOW the posted limit, only ever raises it. See CHANGES.md.
        if n_surrounding >= self._min_agents:
            v_flow = float(np.percentile(surrounding_speeds, self._flow_percentile))
            v_des_free_flow = max(v_limit, v_flow)
        else:
            v_des_free_flow = v_limit
        return RegimeResult(
            regime=Regime.FREE_FLOW,
            v_des=v_des_free_flow,
            v_lead=v_lead,
            d_lead=d_lead,
            has_lead=has_lead,
            v_lane_avg=v_lane_avg,
            v_limit=v_limit,
            n_surrounding_agents=n_surrounding,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_speed_limit(self, ego_state, map_cache) -> float:
        """Return speed limit of current lane, or default if unavailable."""
        try:
            from nuplan.common.maps.abstract_map import Lane
            ego_center = ego_state.center
            lanes = map_cache.lanes
            lane_connectors = map_cache.lane_connectors
            all_lanes = {**lanes, **lane_connectors}

            from shapely.geometry import Point
            ego_pt = Point(ego_center.x, ego_center.y)

            best_lane = None
            best_dist = float("inf")
            for lane in all_lanes.values():
                try:
                    dist = lane.polygon.distance(ego_pt)
                    if dist < best_dist:
                        best_dist = dist
                        best_lane = lane
                except Exception:
                    continue

            if best_lane is not None and isinstance(best_lane, Lane):
                if best_lane.speed_limit_mps is not None:
                    return float(best_lane.speed_limit_mps)
        except Exception:
            pass
        return self.DEFAULT_SPEED_LIMIT_MPS

    def _find_lead(
        self, ego_state, detection_cache
    ) -> Tuple[float, float, bool]:
        """Find the closest vehicle directly ahead in the ego lane."""
        cos_h, sin_h = self._ego_basis(ego_state)
        ex, ey = ego_state.center.x, ego_state.center.y

        best_d = float("inf")
        v_lead = 0.0
        has_lead = False

        try:
            for obj in detection_cache.tracked_objects:
                ox, oy = obj.center.x, obj.center.y
                dx, dy = ox - ex, oy - ey
                d_lon = cos_h * dx + sin_h * dy
                d_lat = abs(-sin_h * dx + cos_h * dy)

                if d_lon <= 0.5 or d_lat > self._lat_tol:
                    continue
                if d_lon < best_d:
                    best_d = d_lon
                    try:
                        v_lead = float(obj.velocity.x * cos_h + obj.velocity.y * sin_h)
                    except Exception:
                        v_lead = 0.0
                    has_lead = True
        except Exception:
            pass

        return v_lead, best_d, has_lead

    def _surrounding_speeds(self, ego_state, detection_cache) -> list:
        """
        Raw speeds of vehicles within +-horizon ahead/behind in ego frame,
        same lateral tolerance as the lead-vehicle check -- i.e. same
        direction of travel, in-lane-ish. Shared by v_lane_avg (mean, used
        for congestion detection) and the free-flow percentile target
        (detect()) so both read from one scan of tracked_objects, not two.
        """
        cos_h, sin_h = self._ego_basis(ego_state)
        ex, ey = ego_state.center.x, ego_state.center.y

        speeds = []
        try:
            for obj in detection_cache.tracked_objects:
                ox, oy = obj.center.x, obj.center.y
                dx, dy = ox - ex, oy - ey
                d_lon = abs(cos_h * dx + sin_h * dy)
                d_lat = abs(-sin_h * dx + cos_h * dy)
                if d_lon <= self._horizon and d_lat <= self._lat_tol:
                    try:
                        spd = math.sqrt(obj.velocity.x ** 2 + obj.velocity.y ** 2)
                        speeds.append(spd)
                    except Exception:
                        pass
        except Exception:
            pass
        return speeds

    @staticmethod
    def _ego_basis(ego_state) -> Tuple[float, float]:
        h = ego_state.center.heading
        return math.cos(h), math.sin(h)
