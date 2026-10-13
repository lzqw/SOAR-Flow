"""
Control-point-aligned MPC environment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np

from gymnasium import Env, Wrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import FlattenObservation

from relax.control.bspline_utils import (
    bspline_eval,
    bspline_eval_with_derivs_scaled,
    make_clamped_knots,
)
from relax.control.drive_mgf import (
    DriveMGFField,
    DriveMGFGeometryConfig,
    compute_drive_mgf_field,
    target_reference_from_action,
)
from relax.env.drive.lane_change import ThreeLaneStraightEnv, ThreeLaneStraightMultiObsEnv
from relax.env.drive.observation_features import (
    encode_explicit_obstacle_features as _encode_explicit_obstacle_features,
    inject_explicit_obstacle_features,
)
from relax.env.registration_utils import safe_register
from relax.utils.action_semantics import (
    FIXED_SPEED_ACTION_SEMANTICS,
    SPEED_COUPLED_ACTION_SEMANTICS,
    SUPPORTED_ACTION_SEMANTICS,
    TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
)


N_WAYPOINTS = 8
WAYPOINT_DIM = 2
DEFAULT_N_CTRL_POINTS = 5
MPC_HORIZON = 5
ROUTE_LOOKAHEAD_METERS = 15.0
DEFAULT_ROUTE_SAMPLES = 16
ROUTE_ASSIGNMENT_SCALE = 2.0

# Historical target-point semantics keep speed as an internal MPC continuation
# variable.  V2.2-Simple instead couples longitudinal target progress and
# target speed to the same two-dimensional Actor action.
# In the historical mode, a coarse speed bank can skip a narrow feasible
# interval where a routed bend needs slightly more braking but a following
# vehicle makes heavy braking unsafe.  In speed-coupled mode every bank entry
# is first encoded back into action[0], so speed authority is never hidden.
# A coarse {2/3, 1/3, 0} bank can skip a narrow feasible interval where a
# routed bend needs slightly more braking but a following vehicle makes heavy
# braking unsafe.  Search downward from cruise with extra resolution around
# the empirically sensitive 2.4--3.6 m/s band (for the 4.8 m/s default).
ADAPTIVE_VELOCITY_FRACTIONS = (
    5.0 / 6.0,
    3.0 / 4.0,
    2.0 / 3.0,
    5.0 / 8.0,
    7.0 / 12.0,
    1.0 / 2.0,
    5.0 / 12.0,
    1.0 / 3.0,
    1.0 / 4.0,
    1.0 / 6.0,
    0.0,
)


def reachable_speed_interval(
    current_speed: float,
    *,
    horizon_time: float,
    acceleration_min: float,
    acceleration_max: float,
    command_speed_min: float = 1.0,
    command_speed_max: float = 4.8,
) -> Tuple[float, float]:
    """State-dependent speed-command interval used by V2.3 actions."""

    values = (
        current_speed,
        horizon_time,
        acceleration_min,
        acceleration_max,
        command_speed_min,
        command_speed_max,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("reachable speed interval inputs must be finite")
    if horizon_time <= 0.0:
        raise ValueError("horizon_time must be positive")
    if acceleration_min >= 0.0 or acceleration_max <= 0.0:
        raise ValueError("require acceleration_min < 0 < acceleration_max")
    if command_speed_min < 0.0 or command_speed_max <= command_speed_min:
        raise ValueError("invalid command speed limits")
    lower = max(
        float(command_speed_min),
        float(current_speed) + float(acceleration_min) * float(horizon_time),
    )
    upper = min(
        float(command_speed_max),
        float(current_speed) + float(acceleration_max) * float(horizon_time),
    )
    # Normal operation satisfies lower <= upper.  A state far outside the
    # declared command envelope can make the mathematical intersection empty;
    # collapse to the nearest command bound instead of silently swapping it.
    if lower > upper:
        nearest = float(
            np.clip(current_speed, command_speed_min, command_speed_max)
        )
        lower = nearest
        upper = nearest
    return float(lower), float(upper)


def decode_target_speed_lateral_action(
    action: np.ndarray,
    *,
    current_speed: float,
    horizon_time: float,
    acceleration_min: float,
    acceleration_max: float,
    command_speed_min: float = 1.0,
    command_speed_max: float = 4.8,
    lateral_scale: float,
    target_x_min: float = 0.0,
    target_x_max: float,
) -> Dict[str, float]:
    """Decode ``[speed intent, lateral target]`` under V2.3 semantics."""

    action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
    lower, upper = reachable_speed_interval(
        current_speed,
        horizon_time=horizon_time,
        acceleration_min=acceleration_min,
        acceleration_max=acceleration_max,
        command_speed_min=command_speed_min,
        command_speed_max=command_speed_max,
    )
    if not np.isfinite(lateral_scale) or lateral_scale <= 0.0:
        raise ValueError("lateral_scale must be positive and finite")
    if not np.isfinite(target_x_min) or not np.isfinite(target_x_max):
        raise ValueError("target x bounds must be finite")
    if target_x_max <= target_x_min:
        raise ValueError("target_x_max must exceed target_x_min")
    fraction = 0.5 * (float(action[0]) + 1.0)
    target_speed = lower + fraction * (upper - lower)
    unconstrained_x = 0.5 * (float(current_speed) + target_speed) * float(
        horizon_time
    )
    target_x = float(np.clip(unconstrained_x, target_x_min, target_x_max))
    return {
        "progress": fraction,
        "target_x": target_x,
        "target_y": float(lateral_scale) * float(action[1]),
        "target_speed": float(target_speed),
        "reachable_speed_lower": lower,
        "reachable_speed_upper": upper,
        "actual_vehicle_speed": float(current_speed),
    }


def encode_target_speed_lateral_action(
    action: np.ndarray,
    *,
    target_speed: float,
    reachable_speed_lower: float,
    reachable_speed_upper: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Encode a Guardian-corrected reachable speed back into action[0]."""

    action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
    values = (target_speed, reachable_speed_lower, reachable_speed_upper, epsilon)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("speed encoding inputs must be finite")
    if epsilon <= 0.0 or reachable_speed_upper < reachable_speed_lower:
        raise ValueError("invalid reachable speed interval")
    encoded = action.copy()
    fraction = (float(target_speed) - float(reachable_speed_lower)) / (
        float(reachable_speed_upper) - float(reachable_speed_lower) + float(epsilon)
    )
    encoded[0] = np.float32(np.clip(2.0 * fraction - 1.0, -1.0, 1.0))
    return encoded


def decode_target_point_speed_action(
    action: np.ndarray,
    *,
    lookahead_distance: float,
    lateral_scale: float,
    max_command_speed: float,
) -> Dict[str, float]:
    """Decode the V2.2 two-dimensional progress/speed-coupled action."""

    action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
    lookahead_distance = float(lookahead_distance)
    lateral_scale = float(lateral_scale)
    max_command_speed = float(max_command_speed)
    if not np.isfinite(lookahead_distance) or lookahead_distance <= 0.0:
        raise ValueError("lookahead_distance must be positive and finite")
    if not np.isfinite(lateral_scale) or lateral_scale <= 0.0:
        raise ValueError("lateral_scale must be positive and finite")
    if not np.isfinite(max_command_speed) or max_command_speed <= 0.0:
        raise ValueError("max_command_speed must be positive and finite")
    progress = 0.5 * (float(action[0]) + 1.0)
    return {
        "progress": progress,
        "target_x": lookahead_distance * progress,
        "target_y": lateral_scale * float(action[1]),
        "target_speed": max_command_speed * progress,
    }


def encode_target_speed_in_action(
    action: np.ndarray,
    *,
    target_speed: float,
    max_command_speed: float,
) -> np.ndarray:
    """Return a two-dimensional action whose progress encodes target speed."""

    action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
    target_speed = float(target_speed)
    max_command_speed = float(max_command_speed)
    if not np.isfinite(target_speed):
        raise ValueError("target_speed must be finite")
    if not np.isfinite(max_command_speed) or max_command_speed <= 0.0:
        raise ValueError("max_command_speed must be positive and finite")
    progress = float(np.clip(target_speed / max_command_speed, 0.0, 1.0))
    encoded = action.copy()
    encoded[0] = np.float32(2.0 * progress - 1.0)
    return encoded


def target_speed_intervention_audit(
    raw_action: np.ndarray,
    executed_action: np.ndarray,
    *,
    raw_target_speed: float,
    executed_target_speed: float,
    action_semantics_version: str,
    action_tolerance: float = 1e-4,
    emergency_stop: bool = False,
) -> Dict[str, object]:
    """Audit that V2.2 speed authority is visible in executed action."""

    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(2)
    executed_action = np.asarray(executed_action, dtype=np.float32).reshape(2)
    raw_target_speed = float(raw_target_speed)
    executed_target_speed = float(executed_target_speed)
    if not all(
        np.isfinite(value)
        for value in (raw_target_speed, executed_target_speed, action_tolerance)
    ):
        raise ValueError("speed intervention audit values must be finite")
    action_delta = executed_action - raw_action
    speed_command_delta = executed_target_speed - raw_target_speed
    actual_control_modified = bool(
        np.linalg.norm(action_delta) > float(action_tolerance)
    )
    speed_only_intervention = bool(
        action_semantics_version
        in (
            SPEED_COUPLED_ACTION_SEMANTICS,
            TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        )
        and abs(speed_command_delta) > 1e-6
        and abs(float(action_delta[1])) <= float(action_tolerance)
    )
    if (
        action_semantics_version
        in (
            SPEED_COUPLED_ACTION_SEMANTICS,
            TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        )
        and abs(speed_command_delta) > 1e-6
        # ``action_tolerance`` is the threshold for counting a material
        # intervention, not for deciding whether the encoded action changed at
        # all.  A small but real float32 longitudinal change can encode a
        # correspondingly small, fully consistent speed change.  Fail closed
        # only when speed changed while the encoded value is exactly unchanged.
        and float(executed_action[0]) == float(raw_action[0])
        and not bool(emergency_stop)
    ):
        raise RuntimeError(
            "speed-coupled execution changed target speed without changing "
            "executed action[0]"
        )
    return {
        "raw_target_speed": raw_target_speed,
        "executed_target_speed": executed_target_speed,
        "speed_command_delta": speed_command_delta,
        "speed_only_intervention": speed_only_intervention,
        "actual_control_modified": actual_control_modified,
        "emergency_stop": bool(emergency_stop),
    }


def apply_terminal_throttle_boost(
    throttle: float,
    target_velocity: Optional[float],
    current_velocity: float,
) -> float:
    if target_velocity is None:
        return float(np.clip(throttle, -1.0, 1.0))
    velocity_error = float(target_velocity) - float(current_velocity)
    if velocity_error > 0.0:
        denom = max(float(target_velocity), 1e-3)
        throttle += min(0.3, velocity_error / denom * 0.5)
    return float(np.clip(throttle, -1.0, 1.0))


def resolve_flat_boost_target_velocity(
    target_velocity_exec: Optional[float],
    nominal_target_velocity: float,
) -> float:
    if target_velocity_exec is None:
        return float(nominal_target_velocity)
    return float(max(float(target_velocity_exec), float(nominal_target_velocity)))


def sanitize_metric_np(
    metric: np.ndarray,
    *,
    act_dim: int,
    fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    metric_np = np.asarray(metric, dtype=np.float32)
    if metric_np.shape != (act_dim, act_dim):
        metric_np = metric_np.reshape(act_dim, act_dim)

    if np.all(np.isfinite(metric_np)):
        return metric_np

    if fallback is not None:
        fallback_np = np.asarray(fallback, dtype=np.float32)
        if fallback_np.shape == (act_dim, act_dim) and np.all(np.isfinite(fallback_np)):
            return fallback_np.copy()

    return np.eye(act_dim, dtype=np.float32)


def _sigmoid_np(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=np.float32)
    out = 1.0 / (1.0 + np.exp(-np.clip(x_arr, -60.0, 60.0)))
    if np.isscalar(x):
        return float(out)
    return out


def _road_penalty_from_clearance_np(
    clearance: float,
    *,
    road_buffer: float,
    road_temperature: float,
) -> float:
    if not np.isfinite(clearance):
        return 0.0
    temp = max(float(road_temperature), 1e-6)
    return float(_sigmoid_np((float(road_buffer) - float(clearance)) / temp) ** 2)


def oriented_box_sat_margin_np(
    center_a: np.ndarray,
    heading_a: float,
    length_a: float,
    width_a: float,
    center_b: np.ndarray,
    heading_b: float,
    length_b: float,
    width_b: float,
) -> float:
    """Signed separating-axis margin between two oriented vehicle bodies.

    Positive values prove that the rectangles are separated, zero denotes
    contact, and negative values denote overlap on every separating axis.
    This live-body diagnostic is deliberately independent of the MPC circle
    cover and its extra clearance, so it can reveal a conservative CBF
    envelope without treating MetaDrive's collision flag as the only signal.
    """

    center_a = np.asarray(center_a, dtype=np.float64).reshape(-1)[:2]
    center_b = np.asarray(center_b, dtype=np.float64).reshape(-1)[:2]
    dimensions = np.asarray(
        [length_a, width_a, length_b, width_b], dtype=np.float64
    )
    if (
        center_a.shape != (2,)
        or center_b.shape != (2,)
        or not np.all(np.isfinite(center_a))
        or not np.all(np.isfinite(center_b))
        or not np.all(np.isfinite(dimensions))
        or np.any(dimensions <= 0.0)
        or not np.isfinite(heading_a)
        or not np.isfinite(heading_b)
    ):
        return float("nan")

    def body_axes(heading: float) -> tuple[np.ndarray, np.ndarray]:
        longitudinal = np.asarray(
            [np.cos(heading), np.sin(heading)], dtype=np.float64
        )
        lateral = np.asarray(
            [-np.sin(heading), np.cos(heading)], dtype=np.float64
        )
        return longitudinal, lateral

    axes_a = body_axes(float(heading_a))
    axes_b = body_axes(float(heading_b))
    delta = center_b - center_a
    half_length_a, half_width_a = 0.5 * dimensions[:2]
    half_length_b, half_width_b = 0.5 * dimensions[2:]
    margins = []
    for axis in (*axes_a, *axes_b):
        support_a = (
            half_length_a * abs(float(np.dot(axes_a[0], axis)))
            + half_width_a * abs(float(np.dot(axes_a[1], axis)))
        )
        support_b = (
            half_length_b * abs(float(np.dot(axes_b[0], axis)))
            + half_width_b * abs(float(np.dot(axes_b[1], axis)))
        )
        margins.append(abs(float(np.dot(delta, axis))) - support_a - support_b)
    return float(max(margins))


def _vehicle_body_geometry_np(vehicle) -> tuple[np.ndarray, float, float, float]:
    """Read the geometry actually used by the MetaDrive vehicle object."""

    position = np.asarray(getattr(vehicle, "position"), dtype=np.float64).reshape(-1)[:2]
    heading = getattr(vehicle, "heading_theta", None)
    if heading is None:
        heading_vector = np.asarray(
            getattr(vehicle, "heading", [1.0, 0.0]), dtype=np.float64
        ).reshape(-1)
        heading = float(np.arctan2(heading_vector[1], heading_vector[0]))
    config = getattr(vehicle, "config", {})
    config = config if isinstance(config, dict) else {}

    def dimension(name: str, fallback: float) -> float:
        value = config.get(name.lower(), None)
        if value is None:
            value = getattr(vehicle, name.upper(), fallback)
        value = float(value)
        return value if np.isfinite(value) and value > 0.0 else float(fallback)

    return position, float(heading), dimension("length", 4.5), dimension("width", 2.0)


def _choose_recovery_bypass_sign(
    active_obstacles: np.ndarray,
    *,
    target_lateral_scale: float,
    position_scale: float,
    default_sign: float = -1.0,
) -> float:
    """Choose the less occupied bypass side from the current obstacle field.

    A fixed left/right preference gives contradictory demonstrations in
    mirrored scenes.  This deterministic risk score favors the side whose
    lane-center target has less nearby obstacle occupancy.  Center-lane
    obstacles contribute equally, so an exact tie retains ``default_sign``.
    """

    obstacles = np.asarray(active_obstacles, dtype=np.float32)
    default = -1.0 if float(default_sign) < 0.0 else 1.0
    if obstacles.size == 0:
        return default
    obstacles = obstacles.reshape(-1, obstacles.shape[-1])
    if obstacles.shape[1] < 2:
        return default
    xy = obstacles[:, :2]
    finite = np.all(np.isfinite(xy), axis=1)
    xy = xy[finite]
    if xy.size == 0:
        return default

    lateral_scale = max(abs(float(target_lateral_scale)), 1e-3)
    longitudinal_scale = max(0.5 * abs(float(position_scale)), 2.0)
    lateral_bandwidth = max(0.40 * lateral_scale, 0.8)
    # Obstacles just behind the ego can still block an immediate lane change;
    # retain them briefly, then discount distance smoothly.
    longitudinal_weight = np.exp(
        -np.maximum(xy[:, 0], 0.0) / longitudinal_scale
    ) * np.where(xy[:, 0] >= -2.0, 1.0, 0.0)

    risks = {}
    for sign in (-1.0, 1.0):
        target_y = sign * lateral_scale
        lateral_weight = np.exp(
            -0.5 * np.square((xy[:, 1] - target_y) / lateral_bandwidth)
        )
        risks[sign] = float(np.sum(longitudinal_weight * lateral_weight))
    if abs(risks[-1.0] - risks[1.0]) <= 1e-6:
        return default
    return -1.0 if risks[-1.0] < risks[1.0] else 1.0


def _safe_lane_width_at(lane, longitudinal: float) -> float:
    if lane is None:
        return 3.7
    try:
        if hasattr(lane, "width_at"):
            return max(float(lane.width_at(longitudinal)), 1e-3)
    except Exception:
        pass
    try:
        if hasattr(lane, "width"):
            return max(float(lane.width), 1e-3)
    except Exception:
        pass
    try:
        if hasattr(lane, "get_width"):
            return max(float(lane.get_width()), 1e-3)
    except Exception:
        pass
    return 3.7


def _safe_lane_heading_theta_at(lane, longitudinal: float) -> float:
    try:
        if hasattr(lane, "heading_theta_at"):
            return float(lane.heading_theta_at(longitudinal))
    except Exception:
        pass
    try:
        heading = getattr(lane, "heading", None)
        if heading is not None and np.isscalar(heading):
            return float(heading)
    except Exception:
        pass
    try:
        if hasattr(lane, "heading_at"):
            heading_vec = np.asarray(lane.heading_at(longitudinal), dtype=np.float32).reshape(-1)
            if heading_vec.shape[0] >= 2:
                return float(np.arctan2(heading_vec[1], heading_vec[0]))
    except Exception:
        pass
    try:
        p0 = np.asarray(lane.position(longitudinal, 0.0), dtype=np.float32).reshape(-1)[:2]
        p1 = np.asarray(
            lane.position(min(float(longitudinal) + 0.5, float(getattr(lane, "length", longitudinal + 0.5))), 0.0),
            dtype=np.float32,
        ).reshape(-1)[:2]
        d = p1 - p0
        return float(np.arctan2(d[1], d[0]))
    except Exception:
        return 0.0


def _sample_route_corridor_from_lane_sets(
    current_ref_lanes,
    next_ref_lanes,
    current_position: np.ndarray,
    *,
    lookahead: float = ROUTE_LOOKAHEAD_METERS,
    max_samples: int = DEFAULT_ROUTE_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    route_xy = np.zeros((max_samples, 2), dtype=np.float32)
    route_normal = np.zeros((max_samples, 2), dtype=np.float32)
    route_half_width = np.zeros((max_samples,), dtype=np.float32)
    route_mask = np.zeros((max_samples,), dtype=bool)

    current_lanes = list(current_ref_lanes or [])
    next_lanes = list(next_ref_lanes or [])
    if len(current_lanes) == 0 or max_samples <= 0:
        return route_xy, route_normal, route_half_width, route_mask

    lane0 = current_lanes[0]
    try:
        current_s, _ = lane0.local_coordinates(np.asarray(current_position, dtype=np.float32))
    except Exception:
        current_s = 0.0
    current_length = max(float(getattr(lane0, "length", 0.0)), 0.0)
    current_s = float(np.clip(current_s, 0.0, current_length))

    next_lane0 = next_lanes[0] if len(next_lanes) > 0 else None
    next_length = max(float(getattr(next_lane0, "length", 0.0)), 0.0) if next_lane0 is not None else 0.0

    sample_ds = np.linspace(0.0, float(lookahead), int(max_samples), dtype=np.float32)
    for idx, ds in enumerate(sample_ds):
        s_total = current_s + float(ds)
        lane_group = current_lanes
        lane = lane0
        if s_total <= current_length + 1e-6:
            longitudinal = float(np.clip(s_total, 0.0, current_length))
        elif next_lane0 is not None:
            s_next = s_total - current_length
            if s_next > next_length + 1e-6:
                break
            lane_group = next_lanes
            lane = next_lane0
            longitudinal = float(np.clip(s_next, 0.0, next_length))
        else:
            break

        lane_width = _safe_lane_width_at(lane, longitudinal)
        lane_num = max(len(lane_group), 1)
        road_half_width = 0.5 * lane_width * lane_num
        road_center_lateral = (0.5 * lane_num - 0.5) * lane_width
        try:
            pos = np.asarray(lane.position(longitudinal, road_center_lateral), dtype=np.float32).reshape(-1)[:2]
        except Exception:
            break

        heading = _safe_lane_heading_theta_at(lane, longitudinal)
        normal = np.array([np.sin(heading), -np.cos(heading)], dtype=np.float32)
        normal /= max(float(np.linalg.norm(normal)), 1e-6)

        route_xy[idx] = pos
        route_normal[idx] = normal
        route_half_width[idx] = road_half_width
        route_mask[idx] = True

    return route_xy, route_normal, route_half_width, route_mask


def _transform_route_descriptor_to_ego(
    route_xy_world: np.ndarray,
    route_normal_world: np.ndarray,
    route_half_width: np.ndarray,
    route_mask: np.ndarray,
    ego_pos: np.ndarray,
    ego_heading: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    route_xy_ego = np.zeros_like(route_xy_world, dtype=np.float32)
    route_normal_ego = np.zeros_like(route_normal_world, dtype=np.float32)
    route_half_width = np.asarray(route_half_width, dtype=np.float32).copy()
    route_mask = np.asarray(route_mask, dtype=bool).copy()
    if not np.any(route_mask):
        return route_xy_ego, route_normal_ego, route_half_width, route_mask

    c = np.cos(-float(ego_heading))
    s = np.sin(-float(ego_heading))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    valid_xy = np.asarray(route_xy_world, dtype=np.float32)[route_mask]
    valid_normal = np.asarray(route_normal_world, dtype=np.float32)[route_mask]
    route_xy_ego[route_mask] = (valid_xy - np.asarray(ego_pos, dtype=np.float32)[None, :]) @ rot.T
    route_normal_ego[route_mask] = valid_normal @ rot.T
    normal_norm = np.linalg.norm(route_normal_ego[route_mask], axis=1, keepdims=True)
    route_normal_ego[route_mask] /= np.maximum(normal_norm, 1e-6)
    return route_xy_ego, route_normal_ego, route_half_width, route_mask


def _compute_route_edge_clearance_np(
    query_xy: np.ndarray,
    route_xy: np.ndarray,
    route_normal: np.ndarray,
    route_half_width: np.ndarray,
    route_mask: np.ndarray,
    *,
    assignment_scale: float = ROUTE_ASSIGNMENT_SCALE,
) -> np.ndarray:
    query = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    valid = np.asarray(route_mask, dtype=bool).reshape(-1)
    if query.shape[0] == 0 or not np.any(valid):
        return np.zeros((query.shape[0],), dtype=np.float32)

    route_xy_valid = np.asarray(route_xy, dtype=np.float32).reshape(-1, 2)[valid]
    del route_normal, assignment_scale
    route_half_width_valid = np.asarray(route_half_width, dtype=np.float32).reshape(-1)[valid]

    if route_xy_valid.shape[0] == 1:
        return route_half_width_valid[0] - np.linalg.norm(
            query - route_xy_valid[0][None, :], axis=1
        )
    p0 = route_xy_valid[:-1]
    p1 = route_xy_valid[1:]
    segment = p1 - p0
    segment_len_sq = np.maximum(np.sum(segment * segment, axis=-1), 1e-8)
    rel = query[:, None, :] - p0[None, :, :]
    fraction = np.clip(
        np.sum(rel * segment[None, :, :], axis=-1) / segment_len_sq[None, :],
        0.0,
        1.0,
    )
    closest = p0[None, :, :] + fraction[:, :, None] * segment[None, :, :]
    distance = np.linalg.norm(query[:, None, :] - closest, axis=-1)
    width = (
        route_half_width_valid[:-1][None, :] * (1.0 - fraction)
        + route_half_width_valid[1:][None, :] * fraction
    )
    return np.max(width - distance, axis=1)


def _compute_route_road_cost_np(
    query_xy: np.ndarray,
    route_xy: np.ndarray,
    route_normal: np.ndarray,
    route_half_width: np.ndarray,
    route_mask: np.ndarray,
    *,
    road_buffer: float,
    road_temperature: float,
    exclude_initial: bool = True,
) -> Tuple[float, float]:
    query = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    if exclude_initial and query.shape[0] > 0:
        query = query[1:]
    if query.shape[0] == 0 or not np.any(np.asarray(route_mask, dtype=bool)):
        return 0.0, 0.0

    clearance = _compute_route_edge_clearance_np(
        query,
        route_xy,
        route_normal,
        route_half_width,
        route_mask,
        assignment_scale=ROUTE_ASSIGNMENT_SCALE,
    )
    temp = max(float(road_temperature), 1e-6)
    road_cost = _sigmoid_np((float(road_buffer) - clearance) / temp) ** 2
    return float(np.mean(road_cost)), float(np.min(clearance))


safe_register(
    id="FlatThreeLaneStraightMPCCtrlPts",
    entry_point="relax.env.drive.lane_change_mpc:make_flat_metadrive_mpc_env",
    max_episode_steps=1000,
)

safe_register(
    id="FlatThreeLaneStraightMPC",
    entry_point="relax.env.drive.lane_change_mpc:make_flat_metadrive_mpc_env",
    max_episode_steps=1000,
)

safe_register(
    id="FlatThreeLaneStraightMultiObsMPCCtrlPts",
    entry_point="relax.env.drive.lane_change_mpc:make_flat_metadrive_multi_obs_mpc_env",
    max_episode_steps=1000,
)

safe_register(
    id="FlatThreeLaneStraightMultiObsMPC",
    entry_point="relax.env.drive.lane_change_mpc:make_flat_metadrive_multi_obs_mpc_env",
    max_episode_steps=1000,
)


@dataclass
class VehicleState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: float = 0.0
    heading: float = 0.0
    lateral_offset: float = 0.0
    heading_error: float = 0.0


@dataclass
class SimpleMPCResult:
    steering: float = 0.0
    throttle: float = 0.0
    trajectory_x: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    trajectory_y: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))


class FastMPCControlWrapper(Wrapper):
    def __init__(
        self,
        env: Env,
        dt: float = 0.1,
        position_scale: float = 50.0,
        heading_scale: float = np.pi,
        velocity_scale: float = 30.0,
        wheelbase: float = 2.5,
        max_velocity: float = 25.0,
        max_steering: float = 0.6,
        max_steer_rate: float = 0.1,
        target_velocity: float = 15.0,
        w_tracking: float = 10.0,
        w_velocity: float = 0.0,
        w_risk: float = 0.0,
        w_cbf: float = 0.0,
        cbf_solve_weight_scale: float = 0.0,
        w_road: float = 0.0,
        road_solve_weight_scale: float = 0.0,
        risk_buffer: float = 1.0,
        risk_temperature: float = 0.3,
        road_buffer: float = 0.5,
        road_temperature: float = 0.5,
        metric_risk_aggregation: str = "smoothmax",
        cbf_enable: bool = False,
        d_clearance: float = 0.1,
        cbf_gamma: float = 1.0,
        max_obstacles: int = 4,
        road_penalty_weight: float = 0.0,
        planned_road_penalty_weight: float = 0.0,
        n_sqp_steps: int = 5,
        n_admm_steps: int = 50,
        lane_penalty_weight: float = 0.05,
        planned_centerline_penalty_weight: float = 0.5,
        enable_mpc_optimization: bool = True,
        verbose: bool = False,
        n_ctrl_points: int = DEFAULT_N_CTRL_POINTS,
        spline_degree: int = 3,
        horizon_time: float = 0.5,
        max_route_samples: int = DEFAULT_ROUTE_SAMPLES,
        route_lookahead: float = ROUTE_LOOKAHEAD_METERS,
        planner_action_mode: str = "free_ctrl_pts",
        target_lateral_scale: Optional[float] = None,
        action_semantics_version: str = FIXED_SPEED_ACTION_SEMANTICS,
        max_command_speed: Optional[float] = None,
        min_command_speed: float = 1.0,
        mgf_terminal_projection: bool = False,
        mgf_projection_steps: int = 6,
        mgf_projection_tolerance: float = 1e-4,
        mgf_projection_reserve: float = 2.5e-1,
        mgf_projection_kinematic_reserve: float = 2.0e-2,
        mgf_margin_softmin_kappa: float = 5000.0,
        mgf_road_tightening: float = 1.10,
        mgf_chance_beta: float = 0.0,
        mgf_conformal_delta: float = 0.0,
        explicit_obstacle_observation: bool = True,
    ):
        super().__init__(env)

        self.waypoint_dim = WAYPOINT_DIM
        self.dt = float(dt)
        self.verbose = bool(verbose)
        self.enable_mpc_optimization = bool(enable_mpc_optimization)
        self.explicit_obstacle_observation = bool(
            explicit_obstacle_observation
        )

        self.position_scale = float(position_scale)
        self.heading_scale = float(heading_scale)
        self.velocity_scale = float(velocity_scale)
        self.wheelbase = float(wheelbase)
        self.max_velocity = float(max_velocity)
        self.max_steer_rate = float(max_steer_rate)
        self.target_velocity = float(target_velocity)
        if planner_action_mode not in ("free_ctrl_pts", "target_point"):
            raise ValueError("planner_action_mode must be 'free_ctrl_pts' or 'target_point'")
        self.planner_action_mode = planner_action_mode
        self.action_semantics_version = str(action_semantics_version)
        if (
            self.planner_action_mode == "target_point"
            and self.action_semantics_version
            not in SUPPORTED_ACTION_SEMANTICS
        ):
            raise ValueError(
                "unsupported target-point action semantics: "
                f"{self.action_semantics_version}"
            )
        if (
            self.action_semantics_version
            in (
                SPEED_COUPLED_ACTION_SEMANTICS,
                TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
            )
            and self.planner_action_mode != "target_point"
        ):
            raise ValueError(
                "explicit-speed action semantics require target-point mode"
            )
        self.max_command_speed = float(
            self.target_velocity
            if max_command_speed is None
            else max_command_speed
        )
        if (
            not np.isfinite(self.max_command_speed)
            or self.max_command_speed <= 0.0
        ):
            raise ValueError("max_command_speed must be positive and finite")
        self.min_command_speed = float(min_command_speed)
        if (
            not np.isfinite(self.min_command_speed)
            or self.min_command_speed < 0.0
            or self.min_command_speed >= self.max_command_speed
        ):
            raise ValueError(
                "min_command_speed must be finite, non-negative and below max"
            )
        self.target_lateral_scale = float(
            self.position_scale if target_lateral_scale is None else target_lateral_scale
        )
        self.mgf_terminal_projection = bool(mgf_terminal_projection)
        self.mgf_projection_steps = max(int(mgf_projection_steps), 0)
        self.mgf_projection_tolerance = float(mgf_projection_tolerance)
        self.mgf_projection_reserve = max(
            float(mgf_projection_reserve), self.mgf_projection_tolerance
        )
        self.mgf_projection_kinematic_reserve = max(
            float(mgf_projection_kinematic_reserve),
            self.mgf_projection_tolerance,
        )
        self.mgf_geometry = DriveMGFGeometryConfig(
            margin_softmin_kappa=mgf_margin_softmin_kappa,
            road_tightening=mgf_road_tightening,
            chance_beta=mgf_chance_beta,
            conformal_delta=mgf_conformal_delta,
        )

        self.lane_penalty_weight = float(lane_penalty_weight)
        self.planned_centerline_penalty_weight = float(planned_centerline_penalty_weight)
        self.road_penalty_weight = float(road_penalty_weight)
        self.planned_road_penalty_weight = float(planned_road_penalty_weight)
        self.road_buffer = float(road_buffer)
        self.road_temperature = float(road_temperature)
        self.route_lookahead = float(route_lookahead)

        self.n_ctrl_points = int(n_ctrl_points)
        self.spline_degree = int(spline_degree)
        if self.spline_degree < 1:
            raise ValueError("spline_degree must be >= 1")
        if self.n_ctrl_points < self.spline_degree + 1:
            raise ValueError("n_ctrl_points must be >= spline_degree + 1")
        if self.n_ctrl_points < 3:
            raise ValueError("n_ctrl_points must be >= 3 for Flat NMPC")
        self.n_free_ctrl_pts = max(self.n_ctrl_points - 2, 1)
        self.planner_point_count = self.n_free_ctrl_pts
        self.traj_dim = (
            WAYPOINT_DIM
            if self.planner_action_mode == "target_point"
            else self.planner_point_count * WAYPOINT_DIM
        )

        self._prev_free_ctrl_pts = None
        self._prev_target_action: Optional[np.ndarray] = None
        self._flat_knots = None
        self._flat_degree = self.spline_degree
        self._flat_horizon_time = float(horizon_time)
        self._flat_t_eval = np.linspace(0.0, 1.0, MPC_HORIZON, dtype=np.float32)
        self.mpc = None

        from relax.control.flat_nmpc import FlatNMPCConfig, FlatSpaceNMPC

        kappa_max = np.tan(max_steering) / wheelbase
        if (
            self.mgf_terminal_projection
            and self.mgf_projection_kinematic_reserve >= kappa_max
        ):
            raise ValueError(
                "mgf_projection_kinematic_reserve must be below the maximum "
                f"kinematic margin kappa_max={kappa_max:.6g}; otherwise the "
                "projection target is structurally unreachable"
            )
        self._flat_config = FlatNMPCConfig(
            n_ctrl_points=self.n_ctrl_points,
            spline_degree=self.spline_degree,
            horizon_time=horizon_time,
            n_eval_points=MPC_HORIZON,
            wheelbase=wheelbase,
            max_steering=max_steering,
            v_min=1.0,
            target_velocity=target_velocity,
            kappa_max=kappa_max,
            a_min=-5.0,
            a_max=3.0,
            w_tracking=w_tracking,
            w_velocity=w_velocity,
            w_risk=w_risk,
            w_cbf=w_cbf,
            cbf_solve_weight_scale=cbf_solve_weight_scale,
            w_road=w_road,
            road_solve_weight_scale=road_solve_weight_scale,
            risk_buffer=risk_buffer,
            risk_temperature=risk_temperature,
            road_buffer=road_buffer,
            road_temperature=road_temperature,
            metric_risk_aggregation=metric_risk_aggregation,
            cbf_enable=cbf_enable,
            d_clearance=d_clearance,
            cbf_gamma=cbf_gamma,
            max_obstacles=max_obstacles,
            max_route_samples=max_route_samples,
            n_sqp_steps=n_sqp_steps,
            n_admm_steps=n_admm_steps,
            ego_half_width=self.mgf_geometry.ego_half_width,
            ego_half_length=self.mgf_geometry.ego_half_length,
        )
        self._flat_knots = np.asarray(
            make_clamped_knots(self.n_ctrl_points, self._flat_config.spline_degree),
            dtype=np.float32,
        )
        self._flat_degree = self._flat_config.spline_degree
        self._flat_horizon_time = self._flat_config.horizon_time
        self._flat_basis_matrix = np.asarray(
            bspline_eval(
                self._flat_t_eval,
                np.eye(self.n_ctrl_points, dtype=np.float32),
                self._flat_knots,
                self._flat_degree,
            ),
            dtype=np.float32,
        )
        _, self._flat_basis_d1, self._flat_basis_d2 = (
            bspline_eval_with_derivs_scaled(
                self._flat_t_eval,
                np.eye(self.n_ctrl_points, dtype=np.float32),
                self._flat_knots,
                self._flat_horizon_time,
                self._flat_degree,
            )
        )
        self._flat_basis_d1 = np.asarray(self._flat_basis_d1, dtype=np.float32)
        self._flat_basis_d2 = np.asarray(self._flat_basis_d2, dtype=np.float32)
        if self.enable_mpc_optimization:
            self.mpc = FlatSpaceNMPC(self._flat_config)

        if self.mpc is not None:
            self.mpc.warmup()

        self.vehicle_state = VehicleState()
        self.prev_steering = 0.0
        self.prev_result: Optional[SimpleMPCResult] = None
        self._cached_flat_ref_traj: Optional[np.ndarray] = None
        self._cached_flat_ref_velocity: Optional[float] = None
        self._cached_flat_ref_free_ctrl_pts: Optional[np.ndarray] = None
        self._cached_flat_ref_path_xy: Optional[np.ndarray] = None

        self._action_space = Box(low=-1.0, high=1.0, shape=(self.traj_dim,), dtype=np.float32)
        self._low_level_action_space = env.action_space
        self._step_count = 0
        self._current_hj = np.eye(self.traj_dim, dtype=np.float32)
        self._last_metric_fallback_hit = False
        self._cached_mgf_field_step = -1
        self._cached_mgf_field_action: Optional[np.ndarray] = None
        self._cached_mgf_field: Optional[DriveMGFField] = None
        self._recovery_bypass_sign: Optional[float] = None
        # Keep the recovery teacher's temporal reference independent from the
        # behaviour policy.  During ordinary-recovery collection both happen
        # to be the same action, but during DAgger the previous executed
        # target belongs to the Actor.  Reusing it here makes the teacher move
        # with the student's error and can reintroduce adjacent-frame label
        # jumps even though the bypass side is fixed.
        self._recovery_target_action: Optional[np.ndarray] = None

    @property
    def action_space(self) -> Box:
        return self._action_space

    @property
    def projection_reserves(self) -> Dict[str, float]:
        """Return the mutable *soft* projection reserves.

        These values are authorization/projection buffers outside the hard
        obstacle, road, and kinematic constraints.  Exposing them separately
        lets an online curriculum become more permissive without changing the
        CBF gain, collision geometry, actuator limits, or feasibility
        tolerance.
        """

        return {
            "spatial": float(self.mgf_projection_reserve),
            "kinematic": float(self.mgf_projection_kinematic_reserve),
        }

    def set_projection_reserves(
        self,
        *,
        spatial: float,
        kinematic: float,
    ) -> Dict[str, float]:
        """Set only the curriculum-controlled soft projection buffers."""

        spatial = max(float(spatial), self.mgf_projection_tolerance)
        kinematic = max(float(kinematic), self.mgf_projection_tolerance)
        kappa_max = float(self._flat_config.kappa_max)
        if not np.isfinite(spatial) or not np.isfinite(kinematic):
            raise ValueError("projection reserves must be finite")
        if kinematic >= kappa_max:
            raise ValueError(
                "kinematic projection reserve must remain below "
                f"kappa_max={kappa_max:.6g}"
            )
        self.mgf_projection_reserve = spatial
        self.mgf_projection_kinematic_reserve = kinematic
        return self.projection_reserves

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        self.vehicle_state = VehicleState()
        self._update_vehicle_state(obs, info)
        self.prev_steering = 0.0
        self.prev_result = None
        self._prev_free_ctrl_pts = None
        self._prev_target_action = None
        self._cached_flat_ref_traj = None
        self._cached_flat_ref_velocity = None
        self._cached_flat_ref_free_ctrl_pts = None
        self._cached_flat_ref_path_xy = None
        if self.mpc is not None:
            self.mpc.reset()
        self._step_count = 0
        self._last_metric_fallback_hit = False
        self._cached_mgf_field_step = -1
        self._cached_mgf_field_action = None
        self._cached_mgf_field = None
        self._recovery_bypass_sign = None
        self._recovery_target_action = None
        self._refresh_current_hj()
        obs, obstacle_features = self._inject_explicit_obstacle_observation(obs)
        info = dict(info)
        info["explicit_obstacle_features"] = obstacle_features.tolist()
        info["explicit_obstacle_observation_enabled"] = bool(
            self.explicit_obstacle_observation
        )
        info.update(self.action_semantics_metadata)
        return obs, info

    @property
    def speed_coupled_action(self) -> bool:
        return bool(
            self.action_semantics_version == SPEED_COUPLED_ACTION_SEMANTICS
        )

    @property
    def target_speed_lateral_action(self) -> bool:
        return bool(
            self.action_semantics_version
            == TARGET_SPEED_LATERAL_ACTION_SEMANTICS
        )

    @property
    def explicit_speed_action(self) -> bool:
        return bool(self.speed_coupled_action or self.target_speed_lateral_action)

    def reachable_speed_interval(self) -> Tuple[float, float]:
        return reachable_speed_interval(
            self.vehicle_state.velocity,
            horizon_time=self._flat_horizon_time,
            acceleration_min=self._flat_config.a_min,
            acceleration_max=self._flat_config.a_max,
            command_speed_min=self.min_command_speed,
            command_speed_max=self.max_command_speed,
        )

    @property
    def action_semantics_metadata(self) -> Dict[str, object]:
        return {
            "action_semantics_version": self.action_semantics_version,
            "lookahead_distance": float(self.position_scale),
            "max_command_speed": float(self.max_command_speed),
            "speed_mapping": (
                "state_reachable_speed_linear_v1"
                if self.target_speed_lateral_action
                else "v=max_command_speed*(action_x+1)/2"
                if self.speed_coupled_action
                else "fixed_internal_target_velocity"
            ),
            "mpc_horizon": float(self._flat_horizon_time),
            "min_command_speed": float(self.min_command_speed),
            "reachable_speed_lower": float(self.reachable_speed_interval()[0]),
            "reachable_speed_upper": float(self.reachable_speed_interval()[1]),
        }

    def decode_target_action(self, action: np.ndarray) -> Dict[str, float]:
        action = self._prepare_high_level_action(action)
        if self.planner_action_mode != "target_point":
            raise RuntimeError("target action decoding requires target-point mode")
        if self.speed_coupled_action:
            return decode_target_point_speed_action(
                action,
                lookahead_distance=self.position_scale,
                lateral_scale=self.target_lateral_scale,
                max_command_speed=self.max_command_speed,
            )
        if self.target_speed_lateral_action:
            return decode_target_speed_lateral_action(
                action,
                current_speed=self.vehicle_state.velocity,
                horizon_time=self._flat_horizon_time,
                acceleration_min=self._flat_config.a_min,
                acceleration_max=self._flat_config.a_max,
                command_speed_min=self.min_command_speed,
                command_speed_max=self.max_command_speed,
                lateral_scale=self.target_lateral_scale,
                target_x_min=0.0,
                target_x_max=self.position_scale,
            )
        return {
            "progress": 0.5 * (float(action[0]) + 1.0),
            "target_x": self.position_scale
            * 0.5
            * (float(action[0]) + 1.0),
            "target_y": self.target_lateral_scale * float(action[1]),
            "target_speed": float(self.target_velocity),
        }

    def _target_velocity_for_action(self, action: np.ndarray) -> float:
        return float(self.decode_target_action(action)["target_speed"])

    def _action_with_target_velocity(
        self,
        action: np.ndarray,
        target_velocity: Optional[float],
    ) -> np.ndarray:
        action = self._prepare_high_level_action(action)
        if not self.explicit_speed_action or target_velocity is None:
            return action
        if self.target_speed_lateral_action:
            lower, upper = self.reachable_speed_interval()
            return encode_target_speed_lateral_action(
                action,
                target_speed=float(target_velocity),
                reachable_speed_lower=lower,
                reachable_speed_upper=upper,
            )
        return encode_target_speed_in_action(
            action,
            target_speed=float(target_velocity),
            max_command_speed=self.max_command_speed,
        )

    def _geometry_action_for_target_action(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Return legacy geometric coordinates and d(geometry_x)/d(action_x)."""

        action = self._prepare_high_level_action(action)
        if not self.target_speed_lateral_action:
            return action, 1.0
        decoded = self.decode_target_action(action)
        geometry = action.copy()
        geometry[0] = np.float32(
            np.clip(
                2.0 * float(decoded["target_x"]) / max(self.position_scale, 1e-6)
                - 1.0,
                -1.0,
                1.0,
            )
        )
        span = float(decoded["reachable_speed_upper"]) - float(
            decoded["reachable_speed_lower"]
        )
        unconstrained_derivative = (
            self._flat_horizon_time * span / (2.0 * max(self.position_scale, 1e-6))
        )
        unconstrained_x = (
            0.5
            * (self.vehicle_state.velocity + float(decoded["target_speed"]))
            * self._flat_horizon_time
        )
        derivative = (
            unconstrained_derivative
            if 0.0 < unconstrained_x < self.position_scale
            else 0.0
        )
        return geometry, float(derivative)

    def _inject_explicit_obstacle_observation(
        self, observation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        obs_xy, _obs_radius, obs_mask = self._get_obstacles_ego_frame()
        features = _encode_explicit_obstacle_features(
            np.asarray(obs_xy, dtype=np.float32)[
                np.asarray(obs_mask, dtype=bool)
            ],
            target_lateral_scale=self.target_lateral_scale,
            position_scale=self.position_scale,
        )
        if self.explicit_obstacle_observation:
            observation = inject_explicit_obstacle_features(
                observation, features
            )
        else:
            observation = np.asarray(
                observation, dtype=np.float32
            ).reshape(-1).copy()
        return observation, features

    def _prepare_high_level_action(self, trajectory: np.ndarray) -> np.ndarray:
        trajectory = np.asarray(trajectory, dtype=np.float32).flatten()
        trajectory = np.clip(trajectory, -1.0, 1.0)
        if len(trajectory) < self.traj_dim:
            trajectory = np.pad(trajectory, (0, self.traj_dim - len(trajectory)))
        elif len(trajectory) > self.traj_dim:
            trajectory = trajectory[: self.traj_dim]
        return trajectory

    def _decode_free_ctrl_pts(self, trajectory: np.ndarray) -> np.ndarray:
        if self.planner_action_mode == "target_point":
            x0 = np.array([0.0, 0.0, 0.0, self.vehicle_state.velocity], dtype=np.float32)
            _, c1 = self._compute_initial_ctrl_pts_np(x0)
            geometry_action, _ = self._geometry_action_for_target_action(trajectory)
            ref, _ = target_reference_from_action(
                geometry_action,
                c1=c1,
                n_free=self.n_free_ctrl_pts,
                position_scale=self.position_scale,
                lateral_scale=self.target_lateral_scale,
            )
            return ref
        traj_points = trajectory.reshape(self.n_free_ctrl_pts, self.waypoint_dim)
        free_ctrl_pts = np.zeros((self.n_free_ctrl_pts, 2), dtype=np.float32)
        for i in range(self.n_free_ctrl_pts):
            # Scale each free control point by its knot position so that
            # the implied velocity is consistent across different n_ctrl_points.
            knot_frac = float(self._flat_knots[self._flat_degree + 1 + i])
            knot_frac = max(knot_frac, 0.1)
            scale_i = self.position_scale * knot_frac
            free_ctrl_pts[i, 0] = (traj_points[i, 0] + 1.0) / 2.0 * scale_i
            free_ctrl_pts[i, 1] = traj_points[i, 1] * scale_i
        return free_ctrl_pts

    def _compute_initial_ctrl_pts_np(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x0_state = np.asarray(x0, dtype=np.float32)
        x_pos, y_pos, theta0, v0 = [float(v) for v in x0_state[:4]]
        c0 = np.array([x_pos, y_pos], dtype=np.float32)
        vel_dir = np.array([np.cos(theta0), np.sin(theta0)], dtype=np.float32)
        delta_knot = float(max(self._flat_knots[self._flat_degree + 1] - self._flat_knots[1], 1e-6))
        alpha = delta_knot * self._flat_horizon_time / max(self._flat_degree, 1)
        c1 = c0 + v0 * vel_dir * alpha
        return c0, c1

    def _target_reference_with_jacobian(
        self,
        target_action: np.ndarray,
        x0: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        c0, c1 = self._compute_initial_ctrl_pts_np(x0)
        geometry_action, geometry_x_derivative = (
            self._geometry_action_for_target_action(target_action)
        )
        ref_free, ref_jacobian = target_reference_from_action(
            geometry_action,
            c1=c1,
            n_free=self.n_free_ctrl_pts,
            position_scale=self.position_scale,
            lateral_scale=self.target_lateral_scale,
        )
        if self.target_speed_lateral_action:
            ref_jacobian = np.asarray(ref_jacobian, dtype=np.float64).copy()
            ref_jacobian[:, 0] *= float(geometry_x_derivative)
        return ref_free, ref_jacobian, c0, c1

    def _sample_flat_reference_path_np(self, ref_free_ctrl_pts: np.ndarray, x0: np.ndarray) -> np.ndarray:
        c0, c1 = self._compute_initial_ctrl_pts_np(x0)
        full_ctrl_pts = np.vstack([c0[None, :], c1[None, :], np.asarray(ref_free_ctrl_pts, dtype=np.float32)])
        return np.asarray(self._flat_basis_matrix @ full_ctrl_pts, dtype=np.float32)

    def _flat_cache_matches(self, trajectory: np.ndarray, x0: np.ndarray, *, require_path: bool) -> bool:
        if self._cached_flat_ref_traj is None or self._cached_flat_ref_velocity is None:
            return False
        if not np.array_equal(self._cached_flat_ref_traj, trajectory):
            return False
        if not np.isclose(self._cached_flat_ref_velocity, float(x0[3]), atol=1e-6):
            return False
        if self._cached_flat_ref_free_ctrl_pts is None:
            return False
        if require_path and self._cached_flat_ref_path_xy is None:
            return False
        return True

    def _cache_flat_reference(
        self,
        trajectory: np.ndarray,
        x0: np.ndarray,
        ref_free_ctrl_pts: np.ndarray,
        ref_path_xy: Optional[np.ndarray],
    ) -> None:
        self._cached_flat_ref_traj = np.asarray(trajectory, dtype=np.float32).copy()
        self._cached_flat_ref_velocity = float(x0[3])
        self._cached_flat_ref_free_ctrl_pts = np.asarray(ref_free_ctrl_pts, dtype=np.float32).copy()
        self._cached_flat_ref_path_xy = (
            None if ref_path_xy is None else np.asarray(ref_path_xy, dtype=np.float32).copy()
        )

    def _get_flat_reference(
        self,
        trajectory: np.ndarray,
        x0: np.ndarray,
        *,
        require_path: bool,
        prefer_cache: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if prefer_cache and self._flat_cache_matches(trajectory, x0, require_path=require_path):
            ref_free = self._cached_flat_ref_free_ctrl_pts.copy()
            ref_path = None if self._cached_flat_ref_path_xy is None else self._cached_flat_ref_path_xy.copy()
            return ref_free, ref_path

        ref_free_ctrl_pts = self._decode_free_ctrl_pts(trajectory)
        ref_path_xy = None
        if require_path:
            ref_path_xy = self._sample_flat_reference_path_np(ref_free_ctrl_pts, x0)

        self._cache_flat_reference(trajectory, x0, ref_free_ctrl_pts, ref_path_xy)
        ref_free = np.asarray(ref_free_ctrl_pts, dtype=np.float32)
        return ref_free, None if ref_path_xy is None else ref_path_xy.copy()

    def _get_ego_world_state(self) -> Tuple[np.ndarray, float]:
        try:
            vehicle = self.env.unwrapped.vehicle
            position = np.asarray(vehicle.position, dtype=np.float32)
            if hasattr(vehicle, "heading_theta"):
                heading = float(vehicle.heading_theta)
            else:
                heading_vec = vehicle.heading
                heading = float(np.arctan2(heading_vec[1], heading_vec[0]))
            return position, heading
        except Exception:
            return np.zeros(2, dtype=np.float32), 0.0

    def _get_route_corridor_ego_frame(self):
        max_route_samples = max(int(getattr(self._flat_config, "max_route_samples", 0)), 0)
        route_xy = np.zeros((max_route_samples, 2), dtype=np.float32)
        route_normal = np.zeros((max_route_samples, 2), dtype=np.float32)
        route_half_width = np.zeros((max_route_samples,), dtype=np.float32)
        route_mask = np.zeros((max_route_samples,), dtype=bool)
        if max_route_samples <= 0:
            return route_xy, route_normal, route_half_width, route_mask

        try:
            vehicle = self.env.unwrapped.vehicle
            navigation = getattr(vehicle, "navigation", None)
            current_ref_lanes = getattr(navigation, "current_ref_lanes", None)
            next_ref_lanes = getattr(navigation, "next_ref_lanes", None)
            current_position = np.asarray(vehicle.position, dtype=np.float32)
        except Exception:
            return route_xy, route_normal, route_half_width, route_mask

        route_xy_world, route_normal_world, route_half_width, route_mask = _sample_route_corridor_from_lane_sets(
            current_ref_lanes,
            next_ref_lanes,
            current_position,
            lookahead=self.route_lookahead,
            max_samples=max_route_samples,
        )
        ego_pos, ego_heading = self._get_ego_world_state()
        return _transform_route_descriptor_to_ego(
            route_xy_world,
            route_normal_world,
            route_half_width,
            route_mask,
            ego_pos,
            ego_heading,
        )

    def _get_current_road_clearance(self) -> float:
        try:
            vehicle = self.env.unwrapped.vehicle
            left = getattr(vehicle, "dist_to_left_side", None)
            right = getattr(vehicle, "dist_to_right_side", None)
            if left is not None and right is not None and np.isfinite(left) and np.isfinite(right):
                return float(min(float(left), float(right)))
        except Exception:
            pass

        try:
            vehicle = self.env.unwrapped.vehicle
            navigation = getattr(vehicle, "navigation", None)
            current_ref_lanes = getattr(navigation, "current_ref_lanes", None)
            if navigation is None or current_ref_lanes is None or len(current_ref_lanes) == 0:
                return 0.0
            current_reference_lane = current_ref_lanes[0]
            _, lateral_to_reference = current_reference_lane.local_coordinates(vehicle.position)
            lateral_to_left = float(lateral_to_reference) + float(navigation.get_current_lane_width()) / 2.0
            lateral_range = float(navigation.get_current_lateral_range(vehicle.position, self.env.unwrapped.engine))
            lateral_to_right = lateral_range - lateral_to_left
            return float(min(lateral_to_left, lateral_to_right))
        except Exception:
            return 0.0

    def get_physical_body_safety(self) -> Dict[str, float | int]:
        """Measure the live MetaDrive bodies without using the CBF model.

        ``sat_margin`` is a signed rectangle separation test in metres.  It
        excludes the configurable CBF clearance and the obstacle circle
        approximation.  Consequently, ``sat_margin > 0`` together with a
        negative CBF clearance is direct evidence that the safety envelope,
        rather than the rendered vehicle bodies, caused the intervention.
        """

        try:
            core = self.env.unwrapped
            ego = core.vehicle
            ego_geometry = _vehicle_body_geometry_np(ego)
        except Exception:
            return {
                "sat_margin": float("nan"),
                "nearest_center_distance": float("nan"),
                "obstacle_count": 0,
            }

        obstacles = []
        for obstacle in list(getattr(core, "_parked_objs", [])):
            if obstacle is not None:
                obstacles.append(obstacle)
        parked = getattr(core, "_parked_obj", None)
        if parked is not None:
            obstacles.append(parked)
        # The official SafeMetaDriveEnv creates accident cones, barriers and
        # broken-down vehicles through object_manager, while pending traffic
        # lives in traffic_manager.spawned_objects before it is triggered.
        # Include both manager-owned sets so the live-body diagnostic covers
        # the same world objects as the MetaSafe CBF obstacle adapter.
        for manager_name in ("object_manager", "traffic_manager"):
            try:
                manager = getattr(core.engine, manager_name)
                obstacles.extend(manager.spawned_objects.values())
            except Exception:
                pass
        try:
            obstacles.extend(
                list(core.engine.traffic_manager.traffic_vehicles)
            )
        except Exception:
            pass

        unique = []
        seen = set()
        for obstacle in obstacles:
            identity = id(obstacle)
            if identity == id(ego) or identity in seen:
                continue
            seen.add(identity)
            unique.append(obstacle)

        margins = []
        center_distances = []
        for obstacle in unique:
            try:
                obstacle_geometry = _vehicle_body_geometry_np(obstacle)
                center_distances.append(
                    float(
                        np.linalg.norm(
                            obstacle_geometry[0] - ego_geometry[0]
                        )
                    )
                )
                margins.append(
                    oriented_box_sat_margin_np(
                        ego_geometry[0],
                        ego_geometry[1],
                        ego_geometry[2],
                        ego_geometry[3],
                        obstacle_geometry[0],
                        obstacle_geometry[1],
                        obstacle_geometry[2],
                        obstacle_geometry[3],
                    )
                )
            except Exception:
                continue
        finite_margins = [value for value in margins if np.isfinite(value)]
        finite_distances = [
            value for value in center_distances if np.isfinite(value)
        ]
        return {
            "sat_margin": (
                float(min(finite_margins))
                if finite_margins
                else float("nan")
            ),
            "nearest_center_distance": (
                float(min(finite_distances))
                if finite_distances
                else float("nan")
            ),
            "obstacle_count": len(finite_margins),
        }

    def _get_obstacles_ego_frame(self):
        max_obstacles = max(int(getattr(self._flat_config, "max_obstacles", 0)), 0)
        # [current x, current y, horizon dx, horizon dy].
        obs_xy = np.zeros((max_obstacles, 4), dtype=np.float32)
        obs_radius = np.zeros((max_obstacles,), dtype=np.float32)
        obs_mask = np.zeros((max_obstacles,), dtype=bool)
        if max_obstacles <= 0:
            return obs_xy, obs_radius, obs_mask

        try:
            obstacle_info = self.env.unwrapped.get_obstacle_info()
            positions_world, radii = obstacle_info[:2]
            velocities_world = (
                obstacle_info[2]
                if len(obstacle_info) >= 3
                else np.zeros_like(positions_world)
            )
        except Exception:
            return obs_xy, obs_radius, obs_mask

        positions_world = np.asarray(positions_world, dtype=np.float32).reshape(-1, 2)
        radii = np.asarray(radii, dtype=np.float32).reshape(-1)
        velocities_world = np.asarray(velocities_world, dtype=np.float32).reshape(-1, 2)
        total_count = min(
            positions_world.shape[0],
            radii.shape[0],
            velocities_world.shape[0],
        )
        if total_count <= 0:
            return obs_xy, obs_radius, obs_mask
        selected_count = min(total_count, max_obstacles)

        ego_pos, ego_heading = self._get_ego_world_state()
        rel = positions_world[:total_count] - ego_pos[None, :]
        c = np.cos(-ego_heading)
        s = np.sin(-ego_heading)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        rel_ego = rel @ rot.T
        velocity_ego = velocities_world[:total_count] @ rot.T

        # Certify a one-control-period swept obstacle tube, not only a
        # point-time constant-velocity prediction.  At the next controller
        # call the one-second MPC horizon has shifted forward by one MetaDrive
        # decision interval.  Without this terminal-time reserve, a fast
        # crossing vehicle can have a positive certificate now and make the
        # complete target set infeasible on the very next frame.  A circle at
        # the midpoint of its one-step displacement, enlarged by half that
        # displacement, conservatively covers the whole translated circle
        # segment while preserving the fixed-size MPC descriptor.
        control_interval = 0.0
        try:
            env_config = self.env.unwrapped.config
            control_interval = float(env_config.get("decision_repeat", 0)) * float(
                env_config.get("physics_world_step_size", 0.0)
            )
        except Exception:
            control_interval = 0.0
        if not np.isfinite(control_interval) or control_interval < 0.0:
            control_interval = 0.0
        one_step_displacement = velocity_ego * control_interval
        rel_ego = rel_ego + 0.5 * one_step_displacement
        radii = radii.copy()
        radii[:total_count] += 0.5 * np.linalg.norm(
            one_step_displacement, axis=1
        )

        # Rank by closest approach to the complete reachable target box, at
        # both ends of the moving-obstacle horizon.  Current-distance-only
        # ranking can hide a farther but fast-approaching vehicle until it
        # abruptly displaces the eighth obstacle on the next control step.
        predicted_ego = rel_ego + velocity_ego * float(self._flat_horizon_time)
        box_low = np.array(
            [-self.mgf_geometry.ego_half_length, -self.target_lateral_scale],
            dtype=np.float32,
        )
        box_high = np.array(
            [self.position_scale + self.mgf_geometry.ego_half_length, self.target_lateral_scale],
            dtype=np.float32,
        )

        def distance_to_reachable_box(points):
            return np.linalg.norm(points - np.clip(points, box_low, box_high), axis=1)

        approach_score = np.minimum(
            distance_to_reachable_box(rel_ego),
            distance_to_reachable_box(predicted_ego),
        )
        order = np.argsort(approach_score, kind="stable")
        keep = order[:selected_count]
        obs_xy[:selected_count, :2] = rel_ego[keep]
        obs_xy[:selected_count, 2:4] = (
            velocity_ego[keep] * float(self._flat_horizon_time)
        )
        obs_radius[:selected_count] = radii[:total_count][keep]
        obs_mask[:selected_count] = True
        return obs_xy, obs_radius, obs_mask

    def _solve_target_candidate(
        self,
        target_action: np.ndarray,
        *,
        warm_start_mode: str = "previous",
        target_velocity: Optional[float] = None,
        obstacle_cbf_enabled: bool = True,
    ):
        if self.planner_action_mode != "target_point":
            raise RuntimeError("MGF target fields require planner_action_mode='target_point'")
        if self.mpc is None:
            raise RuntimeError("MGF target fields require MPC optimization")
        requested_target_velocity = target_velocity
        emergency_stop = bool(
            self.target_speed_lateral_action
            and requested_target_velocity is not None
            and float(requested_target_velocity) <= 1e-6
        )
        target_action = self._action_with_target_velocity(
            target_action,
            target_velocity,
        )
        if self.speed_coupled_action:
            # The longitudinal Actor coordinate is the sole authority for
            # both target progress and target speed.  Explicit Guardian speed
            # requests have already been encoded back into target_action[0].
            target_velocity = self._target_velocity_for_action(target_action)
        elif self.target_speed_lateral_action and not emergency_stop:
            # Ordinary Guardian corrections are encoded into action[0] and
            # decoded again before a complete solve.  Only the explicitly
            # flagged emergency-stop branch may command below the Actor's
            # reachable interval.
            target_velocity = self._target_velocity_for_action(target_action)
        elif self.target_speed_lateral_action and emergency_stop:
            target_velocity = max(float(requested_target_velocity), 0.0)
        x0 = np.array([0.0, 0.0, 0.0, self.vehicle_state.velocity], dtype=np.float32)
        ref_free, ref_jacobian, c0, c1 = self._target_reference_with_jacobian(target_action, x0)
        ref_path = self._sample_flat_reference_path_np(ref_free, x0)
        obs_xy, obs_radius, obs_mask = self._get_obstacles_ego_frame()
        route_xy, route_normal, route_half_width, route_mask = self._get_route_corridor_ego_frame()
        if warm_start_mode == "previous":
            warm_start = self._prev_free_ctrl_pts
        elif warm_start_mode == "reference":
            warm_start = ref_free
        elif warm_start_mode == "zero":
            warm_start = np.zeros_like(ref_free)
        else:
            raise ValueError(f"unknown target MPC warm-start mode: {warm_start_mode}")
        result = self.mpc.step(
            x0,
            ref_free,
            prev_free_ctrl_pts=warm_start,
            target_velocity=target_velocity,
            obs_xy=obs_xy,
            obs_radius=obs_radius,
            obs_mask=obs_mask,
            route_xy=route_xy,
            route_normal=route_normal,
            route_half_width=route_half_width,
            route_mask=route_mask,
            obstacle_cbf_enabled=obstacle_cbf_enabled,
        )
        hard_certificate = self.mpc.shadow_safety_certificate(
            result.opt_free_ctrl_pts,
            x0,
            obs_xy=obs_xy,
            obs_radius=obs_radius,
            obs_mask=obs_mask,
            route_xy=route_xy,
            route_normal=route_normal,
            route_half_width=route_half_width,
            route_mask=route_mask,
        )
        field = compute_drive_mgf_field(
            action=target_action,
            ref_free_ctrl_pts=ref_free,
            ref_jacobian=ref_jacobian,
            optimized_free_ctrl_pts=result.opt_free_ctrl_pts,
            objective_hessian=result.hj,
            objective_value=result.cost_total,
            tracking_weight=self._flat_config.w_tracking,
            c0=c0,
            c1=c1,
            basis_matrix=self._flat_basis_matrix,
            obs_xy=obs_xy,
            obs_radius=obs_radius,
            obs_mask=obs_mask,
            d_clearance=self._flat_config.d_clearance,
            route_xy=route_xy,
            route_normal=route_normal,
            route_half_width=route_half_width,
            route_mask=route_mask,
            geometry=self.mgf_geometry,
            kinematic_margin=result.min_kappa_slack,
            basis_d1=self._flat_basis_d1,
            basis_d2=self._flat_basis_d2,
            kappa_max=self._flat_config.kappa_max,
            v_min=self._flat_config.v_min,
            flat_eps=self._flat_config.eps,
        )
        return {
            "action": target_action,
            "x0": x0,
            "ref_free": ref_free,
            "ref_path": ref_path,
            "result": result,
            "field": field,
            "route_xy": route_xy,
            "route_normal": route_normal,
            "route_half_width": route_half_width,
            "route_mask": route_mask,
            "obs_xy": obs_xy,
            "obs_radius": obs_radius,
            "obs_mask": obs_mask,
            "warm_start_mode": warm_start_mode,
            "target_velocity": float(result.target_velocity_exec),
            "emergency_stop": emergency_stop,
            "action_semantics_version": self.action_semantics_version,
            "obstacle_cbf_enabled": bool(obstacle_cbf_enabled),
            "hard_certificate": hard_certificate,
            "solver_finite": bool(
                hard_certificate.finite
                and all(
                    np.isfinite(value)
                    for value in (
                        result.steering,
                        result.throttle,
                        result.target_velocity_exec,
                        result.cost_total,
                        result.min_kappa_slack,
                        result.min_road_clearance,
                    )
                )
                and np.all(np.isfinite(result.opt_free_ctrl_pts))
                and np.all(np.isfinite(result.predicted_x))
                and np.all(np.isfinite(result.predicted_y))
            ),
        }

    @staticmethod
    def _hard_certificate_summary(certificate) -> Dict[str, object]:
        return {
            "hard_cbf_margin": float(certificate.min_hard_cbf_margin),
            "obstacle_clearance": float(certificate.min_obstacle_clearance),
            "road_clearance": float(certificate.min_road_clearance),
            "kinematic_slack": float(certificate.min_kinematic_slack),
            "has_obstacle": bool(certificate.has_obstacle),
            "has_route": bool(certificate.has_route),
            "finite": bool(certificate.finite),
        }

    @staticmethod
    def _trajectory_difference(first_result, second_result) -> Dict[str, float]:
        first = np.stack(
            [first_result.predicted_x, first_result.predicted_y], axis=1
        ).astype(np.float32)
        second = np.stack(
            [second_result.predicted_x, second_result.predicted_y], axis=1
        ).astype(np.float32)
        count = min(first.shape[0], second.shape[0])
        error = np.linalg.norm(first[:count] - second[:count], axis=1)
        return {
            "trajectory_rmse": float(np.sqrt(np.mean(error**2))),
            "trajectory_max_error": float(np.max(error)),
            "control_delta": float(
                np.linalg.norm(
                    np.asarray(
                        [first_result.steering, first_result.throttle],
                        dtype=np.float32,
                    )
                    - np.asarray(
                        [second_result.steering, second_result.throttle],
                        dtype=np.float32,
                    )
                )
            ),
        }

    def compare_obstacle_cbf_authority(
        self,
        target_action: np.ndarray,
        *,
        target_velocity: Optional[float] = None,
    ) -> Dict[str, object]:
        """Compare hard-CBF MPC with an independently solved ordinary MPC."""

        hard_solved = self._solve_target_candidate(
            target_action,
            warm_start_mode="reference",
            target_velocity=target_velocity,
            obstacle_cbf_enabled=True,
        )
        ordinary_solved = self._solve_target_candidate(
            target_action,
            warm_start_mode="reference",
            target_velocity=target_velocity,
            obstacle_cbf_enabled=False,
        )
        difference = self._trajectory_difference(
            hard_solved["result"], ordinary_solved["result"]
        )
        return {
            "action": np.asarray(target_action, dtype=np.float32).tolist(),
            "hard_mpc": self._hard_certificate_summary(
                hard_solved["hard_certificate"]
            ),
            "ordinary_mpc_shadow": self._hard_certificate_summary(
                ordinary_solved["hard_certificate"]
            ),
            "hard_solver_finite": bool(hard_solved["solver_finite"]),
            "ordinary_solver_finite": bool(ordinary_solved["solver_finite"]),
            **difference,
            "obstacle_cbf_independent": bool(
                ordinary_solved["solver_finite"]
                and (
                    not ordinary_solved["hard_certificate"].has_obstacle
                    or ordinary_solved[
                        "hard_certificate"
                    ].min_hard_cbf_margin
                    >= 0.0
                    and ordinary_solved[
                        "hard_certificate"
                    ].min_obstacle_clearance
                    >= 0.0
                )
                and (
                    not ordinary_solved["hard_certificate"].has_route
                    or ordinary_solved[
                        "hard_certificate"
                    ].min_road_clearance
                    >= 0.0
                )
                and ordinary_solved[
                    "hard_certificate"
                ].min_kinematic_slack
                >= 0.0
            ),
        }

    def certify_independent_actor_target(
        self,
        target_action: np.ndarray,
        *,
        reference_solved: Optional[Dict[str, object]] = None,
        hard_margin: float = 0.0,
        spatial_margin: float = 0.0,
        road_margin: float = 0.0,
        kinematic_margin: float = 0.0,
        trajectory_tolerance: float = 0.08,
        target_velocity: Optional[float] = None,
    ) -> Dict[str, object]:
        """Prove that an Actor target is executable without obstacle CBF.

        The ordinary trajectory is independently optimized and checked by the
        shadow hard certificate.  When a CBF-on reference solve is supplied,
        the two trajectories must also agree within ``trajectory_tolerance``.
        This is stricter than checking that the high-level target was unchanged:
        it detects hidden low-level B-spline/control authority from CBF-MPC.
        """

        ordinary = self._solve_target_candidate(
            target_action,
            warm_start_mode="reference",
            target_velocity=target_velocity,
            obstacle_cbf_enabled=False,
        )
        certificate = ordinary["hard_certificate"]
        difference = (
            self._trajectory_difference(
                reference_solved["result"], ordinary["result"]
            )
            if reference_solved is not None
            else {
                "trajectory_rmse": 0.0,
                "trajectory_max_error": 0.0,
                "control_delta": 0.0,
            }
        )
        valid = bool(
            ordinary["solver_finite"]
            and (
                not certificate.has_obstacle
                or (
                    certificate.min_hard_cbf_margin >= hard_margin
                    and certificate.min_obstacle_clearance >= spatial_margin
                )
            )
            and (
                not certificate.has_route
                or certificate.min_road_clearance >= road_margin
            )
            and certificate.min_kinematic_slack >= kinematic_margin
            and difference["trajectory_rmse"] <= trajectory_tolerance
        )
        execution_certificate = None
        if valid:
            execution_certificate = {
                "step_count": int(self._step_count),
                "raw_action": np.asarray(
                    ordinary["action"], dtype=np.float32
                ).copy(),
                "raw_solved": ordinary,
                "solved": ordinary,
                "projection_used": False,
                "projection_failed": False,
                "certificate_solves": 1,
                "teacher_ordinary_mpc": True,
                "teacher_trajectory_rmse": float(
                    difference["trajectory_rmse"]
                ),
                "raw_hard_certificate": self._hard_certificate_summary(
                    certificate
                ),
                "executed_hard_certificate": self._hard_certificate_summary(
                    certificate
                ),
            }
        return {
            "valid": valid,
            "action": np.asarray(ordinary["action"], dtype=np.float32).copy(),
            "execution_certificate": execution_certificate,
            "ordinary_solved": ordinary,
            **difference,
            **self._hard_certificate_summary(certificate),
        }

    def certify_ordinary_target_action(
        self,
        target_action: np.ndarray,
        *,
        warm_start_mode: str = "previous",
        target_velocity: Optional[float] = None,
    ) -> Dict[str, object]:
        """Certify an ordinary-MPC solve without granting obstacle-CBF authority."""

        ordinary = self._solve_target_candidate(
            target_action,
            warm_start_mode=warm_start_mode,
            target_velocity=target_velocity,
            obstacle_cbf_enabled=False,
        )
        return {
            "step_count": int(self._step_count),
            "raw_action": np.asarray(ordinary["action"], dtype=np.float32).copy(),
            "raw_solved": ordinary,
            "solved": ordinary,
            "projection_used": False,
            "projection_failed": False,
            "certificate_solves": 1,
            "teacher_ordinary_mpc": True,
            "raw_hard_certificate": self._hard_certificate_summary(
                ordinary["hard_certificate"]
            ),
            "executed_hard_certificate": self._hard_certificate_summary(
                ordinary["hard_certificate"]
            ),
        }

    def search_independent_target(
        self,
        anchor_action: np.ndarray,
        *,
        hard_margin: float = 0.0,
        spatial_margin: float = 0.0,
        road_margin: float = 0.0,
        kinematic_margin: float = 0.0,
        search_radius: float = 0.45,
        grid_size: int = 5,
    ) -> Dict[str, object]:
        """Maximize shadow hard-CBF margin over ordinary-MPC target actions."""

        anchor = self._prepare_high_level_action(anchor_action)
        grid_size = max(int(grid_size), 1)
        offsets = np.linspace(-search_radius, search_radius, grid_size)
        actions = [anchor]
        actions.extend(
            np.clip(
                anchor + np.asarray([dx, dy], dtype=np.float32),
                -1.0,
                1.0,
            )
            for dx in offsets
            for dy in offsets
            if not (np.isclose(dx, 0.0) and np.isclose(dy, 0.0))
        )
        rows = []
        best = None
        seen = []
        for action in actions:
            if any(
                np.allclose(action, previous, atol=1e-6, rtol=0.0)
                for previous in seen
            ):
                continue
            seen.append(action)
            ordinary = self._solve_target_candidate(
                action,
                warm_start_mode="reference",
                obstacle_cbf_enabled=False,
            )
            certificate = ordinary["hard_certificate"]
            valid = bool(
                ordinary["solver_finite"]
                and (
                    not certificate.has_obstacle
                    or (
                        certificate.min_hard_cbf_margin >= hard_margin
                        and certificate.min_obstacle_clearance >= spatial_margin
                    )
                )
                and (
                    not certificate.has_route
                    or certificate.min_road_clearance >= road_margin
                )
                and certificate.min_kinematic_slack >= kinematic_margin
            )
            psi = (
                float(certificate.min_hard_cbf_margin)
                if certificate.has_obstacle
                else 10.0
            )
            # Hard CBF is primary.  Progress and proximity to the actor are
            # deterministic tie-breakers, not substitutes for safety.
            score = (
                psi,
                float(np.asarray(action)[0]),
                -float(np.linalg.norm(np.asarray(action) - anchor)),
            )
            rows.append(
                {
                    "action": np.asarray(action, dtype=np.float32).tolist(),
                    "valid": valid,
                    "psi": psi,
                    **self._hard_certificate_summary(certificate),
                }
            )
            if valid and (best is None or score > best[0]):
                best = (score, ordinary)
        selected = best[1] if best is not None else None
        execution_certificate = None
        if selected is not None:
            execution_certificate = {
                "step_count": int(self._step_count),
                "raw_action": np.asarray(
                    selected["action"], dtype=np.float32
                ).copy(),
                "raw_solved": selected,
                "solved": selected,
                "projection_used": False,
                "projection_failed": False,
                "certificate_solves": len(seen),
                "teacher_ordinary_mpc": True,
            }
        return {
            "valid": bool(selected is not None),
            "action": (
                np.asarray(selected["action"], dtype=np.float32).copy()
                if selected is not None
                else anchor.copy()
            ),
            "hard_cbf_margin": (
                float(selected["hard_certificate"].min_hard_cbf_margin)
                if selected is not None
                else float("nan")
            ),
            "execution_certificate": execution_certificate,
            "candidates": rows,
        }

    def plan_safe_recovery_trajectory(
        self,
        *,
        hard_margin: float = 0.0,
        spatial_margin: float = 0.0,
        road_margin: float = 0.0,
        kinematic_margin: float = 0.0,
        min_progress_m: float = 5.0,
        pass_distance: float = 1.5,
        preferred_bypass_sign: Optional[float] = None,
        use_action_history: bool = True,
    ) -> Dict[str, object]:
        """Choose a progressive safe CBF-MPC bypass/recenter trajectory.

        Recovery no longer starts from a permanent ``[1, 0]`` target.  While
        an obstacle is ahead, both legal bypass directions are solved.  A
        deterministic side preference is applied before progress so tiny SQP
        differences cannot make the teacher alternate left/right at adjacent
        states.  Route-centre actions enter the bank only after all observed
        obstacles are behind the pass distance.
        """

        obs_xy, _obs_radius, obs_mask = self._get_obstacles_ego_frame()
        active_obstacles = np.asarray(obs_xy, dtype=np.float32)[
            np.asarray(obs_mask, dtype=bool)
        ]
        route_xy, _route_normal, _route_half_width, route_mask = (
            self._get_route_corridor_ego_frame()
        )
        route_centerline = np.asarray(route_xy, dtype=np.float32)[
            np.asarray(route_mask, dtype=bool)
        ]
        route_target_y = 0.0
        route_action = np.asarray([1.0, 0.0], dtype=np.float32)
        if route_centerline.size:
            route_index = int(
                np.argmin(
                    np.abs(route_centerline[:, 0] - self.position_scale)
                )
            )
            route_target = route_centerline[route_index]
            route_target_y = float(route_target[1])
            route_action = np.clip(
                np.asarray(
                    [
                        2.0
                        * float(route_target[0])
                        / max(self.position_scale, 1e-6)
                        - 1.0,
                        route_target_y
                        / max(self.target_lateral_scale, 1e-6),
                    ],
                    dtype=np.float32,
                ),
                -1.0,
                1.0,
            )
        obstacle_ahead = bool(
            active_obstacles.size
            and np.max(active_obstacles[:, 0]) > -float(pass_distance)
        )
        if obstacle_ahead:
            candidate_actions = (
                (1.0, -1.0),
                (1.0, 1.0),
                (1.0, -0.80),
                (1.0, 0.80),
                (0.75, -1.0),
                (0.75, 1.0),
                (0.75, -0.75),
                (0.75, 0.75),
            )
        else:
            candidate_actions = tuple(
                np.clip(action, -1.0, 1.0)
                for action in (
                    route_action,
                    route_action + np.asarray([0.0, -0.25], dtype=np.float32),
                    route_action + np.asarray([0.0, 0.25], dtype=np.float32),
                    np.asarray([1.0, 0.0], dtype=np.float32),
                    np.asarray([0.75, 0.0], dtype=np.float32),
                )
            )

        rows = []
        best = None
        previous_target_action = (
            np.asarray(self._recovery_target_action, dtype=np.float32).reshape(2)
            if use_action_history and self._recovery_target_action is not None
            # The recovery teacher must be a function of its own history and
            # the current state, not of the student's previous executed
            # target.  On the first obstacle-local query the route-centre
            # action is the deterministic continuity anchor.  In ordinary
            # expert rollouts this is equivalent to the preceding nominal
            # straight action, while DAgger remains independent of Actor drift.
            else route_action.copy()
        )
        if obstacle_ahead:
            if preferred_bypass_sign is not None and np.isfinite(
                preferred_bypass_sign
            ):
                selected_sign = float(np.sign(preferred_bypass_sign))
                if selected_sign == 0.0:
                    selected_sign = -1.0
                self._recovery_bypass_sign = selected_sign
            elif self._recovery_bypass_sign is None:
                self._recovery_bypass_sign = _choose_recovery_bypass_sign(
                    active_obstacles,
                    target_lateral_scale=self.target_lateral_scale,
                    position_scale=self.position_scale,
                )
            preferred_bypass_sign = float(self._recovery_bypass_sign)
        else:
            preferred_bypass_sign = float(
                self._recovery_bypass_sign
                if self._recovery_bypass_sign is not None
                else -1.0
            )
            self._recovery_bypass_sign = None
        for candidate_action in candidate_actions:
            solved = self._solve_target_candidate(
                np.asarray(candidate_action, dtype=np.float32),
                warm_start_mode="reference",
                obstacle_cbf_enabled=True,
            )
            certificate = solved["hard_certificate"]
            progress = float(solved["result"].predicted_x[-1])
            final_lateral = float(solved["result"].predicted_y[-1])
            valid = bool(
                solved["solver_finite"]
                and progress >= min_progress_m
                and (
                    not certificate.has_obstacle
                    or (
                        certificate.min_hard_cbf_margin >= hard_margin
                        and certificate.min_obstacle_clearance >= spatial_margin
                    )
                )
                and (
                    not certificate.has_route
                    or certificate.min_road_clearance >= road_margin
                )
                and certificate.min_kinematic_slack >= kinematic_margin
            )
            if obstacle_ahead and active_obstacles.size:
                obstacle_lateral = float(
                    active_obstacles[
                        int(np.argmin(np.abs(active_obstacles[:, 0]))), 1
                    ]
                )
                signed_lateral_separation = (
                    final_lateral - obstacle_lateral
                )
                lateral_separation = abs(signed_lateral_separation)
                preferred_side = float(
                    preferred_bypass_sign * signed_lateral_separation > 0.0
                )
            else:
                lateral_separation = -abs(final_lateral - route_target_y)
                preferred_side = 1.0
            psi = (
                float(certificate.min_hard_cbf_margin)
                if certificate.has_obstacle
                else 10.0
            )
            action_change = float(
                np.linalg.norm(
                    np.asarray(candidate_action, dtype=np.float32)
                    - previous_target_action
                )
            )
            score = (
                (
                    preferred_side,
                    -action_change,
                    progress,
                    lateral_separation,
                    float(np.clip(psi, -1.0, 2.0)),
                    float(np.asarray(candidate_action)[0]),
                )
                if obstacle_ahead
                else (
                    lateral_separation,
                    progress,
                    float(np.clip(psi, -1.0, 2.0)),
                    float(np.asarray(candidate_action)[0]),
                )
            )
            rows.append(
                {
                    "action": np.asarray(
                        candidate_action, dtype=np.float32
                    ).tolist(),
                    "valid": valid,
                    "progress_m": progress,
                    "final_lateral_m": final_lateral,
                    "lateral_separation_m": lateral_separation,
                    "preferred_side": bool(preferred_side),
                    "action_change": action_change,
                    **self._hard_certificate_summary(certificate),
                }
            )
            if valid and (best is None or score > best[0]):
                best = (score, solved)

        selected = best[1] if best is not None else None
        source_certificate = None
        if selected is not None:
            action = np.asarray(selected["action"], dtype=np.float32)
            if use_action_history:
                self._recovery_target_action = action.copy()
            source_certificate = {
                "step_count": int(self._step_count),
                "raw_action": action.copy(),
                "raw_solved": selected,
                "solved": selected,
                "projection_used": False,
                "projection_failed": False,
                "certificate_solves": len(candidate_actions),
                "recovery_trajectory_plan": True,
                "preferred_bypass_sign": preferred_bypass_sign,
            }
        return {
            "valid": bool(selected is not None),
            "obstacle_ahead": obstacle_ahead,
            "preferred_bypass_sign": preferred_bypass_sign,
            "action": (
                np.asarray(selected["action"], dtype=np.float32).copy()
                if selected is not None
                else self._prepare_high_level_action((1.0, 0.0))
            ),
            "safe_solved": selected,
            "source_certificate": source_certificate,
            "candidates": rows,
        }

    def reset_recovery_bypass_preference(self) -> None:
        """Forget the obstacle-local recovery side without resetting the env.

        The recovery planner intentionally keeps one bypass side while an
        obstacle is active so adjacent same-state teacher queries cannot flip
        between symmetric MPC solutions.  Collectors call this method after a
        far-field frame, which makes the next obstacle choose its own side
        from geometry rather than inheriting the previous obstacle's choice.
        """

        self._recovery_bypass_sign = None
        self._recovery_target_action = None

    def derive_inverse_mpc_teacher(
        self,
        safe_solved: Dict[str, object],
        *,
        hard_margin: float = 0.0,
        spatial_margin: float = 0.0,
        road_margin: float = 0.0,
        kinematic_margin: float = 0.0,
        lookahead_fraction: float = 1.0,
        min_progress_m: float = 5.0,
    ) -> Dict[str, object]:
        """Infer a 2-D target that ordinary MPC can execute independently.

        The input trajectory is produced by the safe MPC.  Candidate target
        points are solved again with obstacle-CBF authority disabled and are
        ranked by shadow hard-CBF margin plus trajectory agreement.  A label
        is valid only when the independent ordinary trajectory passes every
        physical hard check.
        """

        safe_result = safe_solved["result"]
        safe_path = np.stack(
            [safe_result.predicted_x, safe_result.predicted_y], axis=1
        ).astype(np.float32)
        index = min(
            max(
                int(round((safe_path.shape[0] - 1) * lookahead_fraction)),
                int(safe_path.shape[0] > 1),
            ),
            safe_path.shape[0] - 1,
        )
        target_xy = safe_path[index]
        derived_action = np.clip(
            np.asarray(
                [
                    2.0 * float(target_xy[0]) / max(self.position_scale, 1e-6)
                    - 1.0,
                    float(target_xy[1])
                    / max(self.target_lateral_scale, 1e-6),
                ],
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        )
        safe_action = np.asarray(safe_solved["action"], dtype=np.float32)
        offsets = (
            (0.0, 0.0),
            (-0.15, 0.0),
            (0.15, 0.0),
            (0.0, -0.15),
            (0.0, 0.15),
            (-0.15, -0.15),
            (-0.15, 0.15),
            (0.15, -0.15),
            (0.15, 0.15),
            (0.0, -0.30),
            (0.0, 0.30),
        )
        actions = [derived_action, safe_action]
        actions.extend(
            np.clip(
                derived_action + np.asarray(offset, dtype=np.float32),
                -1.0,
                1.0,
            )
            for offset in offsets[1:]
        )
        unique_actions = []
        for action in actions:
            if not any(
                np.allclose(action, existing, atol=1e-6, rtol=0.0)
                for existing in unique_actions
            ):
                unique_actions.append(action)

        rows = []
        best = None
        for action in unique_actions:
            ordinary = self._solve_target_candidate(
                action,
                warm_start_mode="reference",
                target_velocity=float(safe_solved["target_velocity"]),
                obstacle_cbf_enabled=False,
            )
            certificate = ordinary["hard_certificate"]
            difference = self._trajectory_difference(
                safe_result, ordinary["result"]
            )
            hard_ok = bool(
                ordinary["solver_finite"]
                and float(ordinary["result"].predicted_x[-1])
                >= min_progress_m
                and (
                    not certificate.has_obstacle
                    or (
                        certificate.min_hard_cbf_margin >= hard_margin
                        and certificate.min_obstacle_clearance >= spatial_margin
                    )
                )
                and (
                    not certificate.has_route
                    or certificate.min_road_clearance >= road_margin
                )
                and certificate.min_kinematic_slack >= kinematic_margin
            )
            finite_hard_margin = (
                float(certificate.min_hard_cbf_margin)
                if certificate.has_obstacle
                else 10.0
            )
            # Once physical thresholds pass, trajectory reproduction is the
            # teacher objective.  Cap the extra psi reward so a short/braking
            # path cannot beat the safe trajectory merely by moving farther
            # away from an already satisfied barrier.
            capped_psi = float(np.clip(finite_hard_margin, -1.0, 2.0))
            score = float(
                -5.0 * difference["trajectory_rmse"]
                - 0.05 * np.linalg.norm(action - derived_action)
                + 0.20 * float(ordinary["result"].predicted_x[-1])
                + 0.05 * capped_psi
            )
            row = {
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "valid": hard_ok,
                "score": score,
                "progress_m": float(ordinary["result"].predicted_x[-1]),
                **difference,
                **self._hard_certificate_summary(certificate),
            }
            rows.append(row)
            if hard_ok and (best is None or score > best[0]):
                best = (score, ordinary, difference)

        selected = best[1] if best is not None else None
        selected_difference = best[2] if best is not None else None
        action = (
            np.asarray(selected["action"], dtype=np.float32)
            if selected is not None
            else derived_action
        )
        execution_certificate = None
        if selected is not None:
            execution_certificate = {
                "step_count": int(self._step_count),
                "raw_action": action.copy(),
                "raw_solved": selected,
                "solved": selected,
                "projection_used": False,
                "projection_failed": False,
                "certificate_solves": len(unique_actions),
                "teacher_ordinary_mpc": True,
            }
        return {
            "valid": bool(selected is not None),
            "action": action.copy(),
            "lookahead_action": derived_action.copy(),
            "safe_mpc_action": safe_action.copy(),
            "trajectory_rmse": (
                float(selected_difference["trajectory_rmse"])
                if selected_difference is not None
                else float("nan")
            ),
            "trajectory_max_error": (
                float(selected_difference["trajectory_max_error"])
                if selected_difference is not None
                else float("nan")
            ),
            "hard_cbf_margin": (
                float(selected["hard_certificate"].min_hard_cbf_margin)
                if selected is not None
                else float("nan")
            ),
            "candidate_count": len(unique_actions),
            "ordinary_solved": selected,
            "execution_certificate": execution_certificate,
            "candidates": rows,
        }

    def get_current_mgf_field(self, anchor_action: Optional[np.ndarray] = None) -> DriveMGFField:
        if self.planner_action_mode != "target_point":
            raise RuntimeError("get_current_mgf_field is only available in target-point mode")
        if anchor_action is None:
            anchor_action = (
                np.zeros((2,), dtype=np.float32)
                if self._prev_target_action is None
                else self._prev_target_action
            )
        anchor_action = self._prepare_high_level_action(anchor_action)
        if (
            self._cached_mgf_field is not None
            and self._cached_mgf_field_step == self._step_count
            and self._cached_mgf_field_action is not None
            and np.array_equal(anchor_action, self._cached_mgf_field_action)
        ):
            return self._cached_mgf_field
        solved = self._solve_target_candidate(anchor_action)

        def field_is_finite(candidate_field: DriveMGFField) -> bool:
            arrays = (
                candidate_field.anchor,
                candidate_field.grad_margin,
                candidate_field.grad_value,
                candidate_field.state_metric,
            )
            scalars = (
                candidate_field.margin,
                candidate_field.margin_delta,
                candidate_field.value,
                candidate_field.state_metric_logdet,
                candidate_field.sensitivity_condition,
            )
            return bool(
                candidate_field.sensitivity_valid
                and all(np.all(np.isfinite(value)) for value in arrays)
                and all(np.isfinite(value) for value in scalars)
            )

        if not field_is_finite(solved["field"]):
            # Degenerate stop references can occasionally make the SQP
            # Hessian/sensitivity invalid even though its scalar certificate
            # is finite.  Re-anchor the affine guide at independent forward
            # targets; the executed action still receives its own exact MPC
            # solve and terminal certificate.
            retry_solved = []
            for retry_action in (
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([-0.5, 0.0], dtype=np.float32),
                np.array([0.5, 0.0], dtype=np.float32),
            ):
                if np.array_equal(retry_action, anchor_action):
                    continue
                candidate = self._solve_target_candidate(retry_action)
                if field_is_finite(candidate["field"]):
                    retry_solved.append(candidate)
            if retry_solved:
                solved = max(
                    retry_solved,
                    key=lambda candidate: candidate["field"].margin_delta,
                )
            else:
                # Last-resort finite guide: no analytic correction is applied,
                # while the exact terminal projection remains authoritative.
                invalid = solved["field"]
                metric = np.eye(2, dtype=np.float32) * float(
                    self.mgf_geometry.metric_damping
                )
                solved["field"] = replace(
                    invalid,
                    margin=-1.0,
                    margin_delta=-1.0,
                    grad_margin=np.zeros((2,), dtype=np.float32),
                    value=0.0,
                    grad_value=np.zeros((2,), dtype=np.float32),
                    state_metric=metric,
                    state_metric_logdet=float(np.linalg.slogdet(metric)[1]),
                    sensitivity_condition=1.0,
                    sensitivity_valid=False,
                    min_constraint=-1.0,
                    min_obstacle_constraint=-1.0,
                    min_road_constraint=-1.0,
                    min_kinematic_constraint=-1.0,
                    softmin_gap=0.0,
                )
        self._cached_mgf_field_step = self._step_count
        self._cached_mgf_field_action = anchor_action.copy()
        self._cached_mgf_field = solved["field"]
        return solved["field"]

    def _project_target_candidate(self, raw_solved):
        field = raw_solved["field"]
        nominal_target_velocity = float(
            raw_solved.get(
                "target_velocity",
                self._target_velocity_for_action(raw_solved["action"]),
            )
        )
        reserve_only = bool(field.margin_delta >= self.mgf_projection_tolerance)
        route_candidates = []
        projection_target = max(
            self.mgf_projection_tolerance,
            float(vars(self).get("mgf_projection_reserve", self.mgf_projection_tolerance)),
        )
        # Curvature/actuation slack has different units from spatial
        # clearance.  Keep a small operational buffer so the projection
        # line-search does not land at the 1e-4 mathematical boundary, where
        # receding-horizon tracking becomes brittle, without demanding the
        # much larger 0.25 m obstacle/road reserve from kinematics.
        kinematic_projection_target = max(
            self.mgf_projection_tolerance,
            float(vars(self).get("mgf_projection_kinematic_reserve", 2.0e-2)),
        )

        def projection_spatial_margin(solved_field):
            """Return the uncertainty-adjusted obstacle/road reserve.

            ``margin_delta`` is the composite feasibility certificate and
            therefore includes curvature/actuation slack.  Those kinematic
            inequalities must be feasible, but asking them to carry the same
            metre-valued reserve as obstacle and road clearance biases the
            exact projection toward a stopped straight trajectory on bends.
            Keep their strict feasibility in ``margin_delta`` and apply the
            larger recursive reserve only to spatial safety constraints.

            Older test doubles expose only ``margin_delta``; falling back to
            it preserves the legacy scalar-certificate contract for them.
            The subtraction conservatively transfers any softmin/chance/
            conformal correction already present in the composite margin to
            the spatial minimum.
            """

            spatial = []
            for name in ("min_obstacle_constraint", "min_road_constraint"):
                value = float(getattr(solved_field, name, np.nan))
                if np.isfinite(value):
                    spatial.append(value)
            margin_delta = float(solved_field.margin_delta)
            if not spatial:
                return margin_delta
            min_constraint = float(
                getattr(solved_field, "min_constraint", margin_delta)
            )
            correction = (
                max(min_constraint - margin_delta, 0.0)
                if np.isfinite(min_constraint)
                else 0.0
            )
            return float(min(spatial) - correction)

        def projection_reserve_met(solved_field):
            kinematic_margin = float(
                getattr(solved_field, "min_kinematic_constraint", np.nan)
            )
            kinematic_reserve_met = bool(
                not np.isfinite(kinematic_margin)
                or kinematic_margin >= kinematic_projection_target
            )
            return bool(
                float(solved_field.margin_delta) >= self.mgf_projection_tolerance
                and kinematic_reserve_met
                and projection_spatial_margin(solved_field) >= projection_target
            )

        def diagnostic_row(solved):
            solved_field = solved["field"]
            solved_action = np.asarray(solved["action"], dtype=np.float32)
            return {
                "action": solved_action.tolist(),
                "source": (
                    "route"
                    if any(
                        np.allclose(solved_action, action, atol=1e-6, rtol=0.0)
                        for action in route_candidates
                    )
                    else "search"
                ),
                "warm_start_mode": solved.get("warm_start_mode", "previous"),
                "target_velocity": float(
                    solved.get("target_velocity", nominal_target_velocity)
                ),
                "m_delta": float(solved_field.margin_delta),
                "min_constraint": float(getattr(solved_field, "min_constraint", np.nan)),
                "min_obstacle_constraint": float(
                    getattr(solved_field, "min_obstacle_constraint", np.nan)
                ),
                "min_road_constraint": float(
                    getattr(solved_field, "min_road_constraint", np.nan)
                ),
                "min_kinematic_constraint": float(
                    getattr(solved_field, "min_kinematic_constraint", np.nan)
                ),
                "spatial_reserve_margin": projection_spatial_margin(solved_field),
                "spatial_reserve_met": bool(
                    projection_spatial_margin(solved_field) >= projection_target
                ),
                "projection_reserve_met": projection_reserve_met(solved_field),
                "kinematic_projection_target": float(
                    kinematic_projection_target
                ),
                "softmin_gap": float(getattr(solved_field, "softmin_gap", np.nan)),
            }

        diagnostics = {
            "raw": diagnostic_row(raw_solved),
            "candidates": [],
            "projection_target": float(projection_target),
        }
        if not self.mgf_terminal_projection or projection_reserve_met(field):
            diagnostics["selected"] = diagnostics["raw"]
            diagnostics["projection_reserve_met"] = projection_reserve_met(field)
            raw_solved["projection_diagnostics"] = diagnostics
            return raw_solved, False, False, 1

        # A previous/zero target is often safe, but it is not a mathematical
        # safe set invariant (notably at reset or after a large state change).
        # Search a small deterministic bank ending in an emergency stop target
        # before declaring that exact projection has no certified anchor.
        candidates = []
        if self._prev_target_action is not None:
            candidates.append(self._prev_target_action.copy())

        # Route-derived anchors follow the actual curved centerline instead of
        # assuming that a fixed lateral target is safe on every PG block.
        route_xy = np.asarray(raw_solved.get("route_xy", np.zeros((0, 2))), dtype=np.float32)
        route_mask = np.asarray(
            raw_solved.get("route_mask", np.zeros((route_xy.shape[0],), dtype=bool)),
            dtype=bool,
        )
        if route_xy.ndim == 2 and route_xy.shape[1] == 2 and np.any(route_mask):
            centerline = route_xy[route_mask]
            if centerline.shape[0] > 1:
                cumulative = np.concatenate(
                    [
                        np.zeros((1,), dtype=np.float32),
                        np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1)),
                    ]
                )
                route_anchor_distances = (
                    (2.5, 4.0, 6.0)
                    if reserve_only
                    else (1.25, 2.5, 4.0)
                )
                for distance in route_anchor_distances:
                    point = centerline[int(np.argmin(np.abs(cumulative - distance)))]
                    lateral = float(point[1]) / max(
                        self.target_lateral_scale, 1e-6
                    )
                    if self.target_speed_lateral_action:
                        route_speed = (
                            2.0 * float(point[0]) / self._flat_horizon_time
                            - self.vehicle_state.velocity
                        )
                        lower, upper = self.reachable_speed_interval()
                        route_action = self._action_with_target_velocity(
                            np.array([0.0, lateral], dtype=np.float32),
                            float(np.clip(route_speed, lower, upper)),
                        )
                    else:
                        route_action = np.clip(
                            np.array(
                                [
                                    2.0
                                    * float(point[0])
                                    / max(self.position_scale, 1e-6)
                                    - 1.0,
                                    lateral,
                                ],
                                dtype=np.float32,
                            ),
                            -1.0,
                            1.0,
                        )
                    candidates.append(route_action)
                    route_candidates.append(route_action)
        standard_anchors = (
            (
                (0.0, 0.0),
                (-0.5, 0.0),
                (0.5, 0.0),
                (1.0, 0.0),
            )
            if reserve_only
            else (
                (0.0, 0.0),
                (-1.0, 0.0),
                (-1.0, -0.25),
                (-1.0, 0.25),
                (-0.5, 0.0),
                (-0.5, -0.25),
                (-0.5, 0.25),
                (-0.5, -0.5),
                (-0.5, 0.5),
                (0.0, -0.25),
                (0.0, 0.25),
                (0.0, -0.5),
                (0.0, 0.5),
                # Forward lane-change anchors prevent the terminal projection
                # from becoming absorbed by a previously safe stop target.
                # They remain a small fixed bank; the exact MPC certificate,
                # not the anchor heuristic, decides whether either side is
                # executable in the current routed corridor.
                (0.5, -0.75),
                (0.5, 0.75),
                (0.75, -1.0),
                (0.75, 1.0),
                (1.0, -1.0),
                (1.0, 1.0),
            )
        )
        candidates.extend(
            np.asarray(action, dtype=np.float32) for action in standard_anchors
        )
        safe_solved = None
        reserve_solutions = []
        best = raw_solved
        solves = 1

        def is_route_action(action):
            action = np.asarray(action, dtype=np.float32)
            return any(
                np.allclose(action, route_action, atol=1e-6, rtol=0.0)
                for route_action in route_candidates
            )

        def candidate_key(action, target_velocity):
            keyed_action = self._action_with_target_velocity(
                action,
                target_velocity,
            )
            if self.speed_coupled_action:
                keyed_velocity = self._target_velocity_for_action(keyed_action)
            elif self.target_speed_lateral_action:
                keyed_velocity = (
                    self._target_velocity_for_action(keyed_action)
                    if target_velocity is None
                    else float(target_velocity)
                )
            else:
                keyed_velocity = float(
                    nominal_target_velocity
                    if target_velocity is None
                    else target_velocity
                )
            return (
                round(float(keyed_velocity), 6),
                *tuple(np.asarray(keyed_action, dtype=np.float32).tolist()),
            )

        def solve_with_candidate_velocity(action, source_solved, *, warm_start_mode="previous"):
            velocity_aware = (
                "target_velocity" in source_solved or nominal_target_velocity > 0.0
            )
            source_velocity = float(
                source_solved.get("target_velocity", nominal_target_velocity)
            )
            if not velocity_aware and warm_start_mode == "previous":
                return self._solve_target_candidate(action)
            if not velocity_aware:
                return self._solve_target_candidate(
                    action,
                    warm_start_mode=warm_start_mode,
                )
            if warm_start_mode == "previous":
                return self._solve_target_candidate(
                    action,
                    target_velocity=source_velocity,
                )
            return self._solve_target_candidate(
                action,
                warm_start_mode=warm_start_mode,
                target_velocity=source_velocity,
            )

        seen = {
            candidate_key(
                raw_solved["action"],
                raw_solved.get("target_velocity", nominal_target_velocity),
            )
        }
        for safe_action in candidates:
            key = candidate_key(safe_action, None)
            if key in seen:
                continue
            seen.add(key)
            candidate_solved = self._solve_target_candidate(safe_action)
            solves += 1
            diagnostics["candidates"].append(diagnostic_row(candidate_solved))
            if candidate_solved["field"].margin_delta > best["field"].margin_delta:
                best = candidate_solved
            if projection_reserve_met(candidate_solved["field"]):
                reserve_solutions.append(candidate_solved)

        if safe_solved is None and reserve_solutions:
            # Once several anchors meet the complete spatial and operational
            # kinematic reserve, preserve task progress: prefer a map-derived
            # route anchor and then the farthest certified point along it.
            # Nearest-to-policy remains the tie-breaker, so obstacle/road
            # filtering is still authoritative while a historical short/stop
            # target cannot absorb the receding-horizon controller.
            raw_action = np.asarray(raw_solved["action"], dtype=np.float32)
            safe_solved = max(
                reserve_solutions,
                key=lambda solved: (
                    is_route_action(solved["action"]),
                    float(np.asarray(solved["action"], dtype=np.float32)[0]),
                    -float(
                        np.linalg.norm(
                            np.asarray(solved["action"], dtype=np.float32)
                            - raw_action
                        )
                    ),
                ),
            )

        best_field = best["field"]
        best_obstacle = float(
            getattr(best_field, "min_obstacle_constraint", np.nan)
        )
        obstacle_requires_velocity_reserve = bool(
            np.isfinite(best_obstacle)
            and best_obstacle < projection_target
        )
        best_action = np.asarray(best["action"], dtype=np.float32)
        best_target_distance = float(
            self.decode_target_action(best_action)["target_x"]
        )
        # If the policy request is unsafe and the nominal-speed search can
        # certify only a near-stop target, the safe point must not suppress
        # the internal velocity search.  On a routed bend, a forward anchor
        # can miss the curvature certificate by numerical tolerance at cruise
        # yet become comfortably feasible after a small speed reduction.
        # This is distinct from an already-safe raw action, where the reserve
        # remains optional and must not reduce nominal speed.
        progress_requires_velocity_search = bool(
            not reserve_only and best_target_distance < 2.0
        )
        spatial_requires_velocity_reserve = bool(
            projection_spatial_margin(best_field) < projection_target
        )
        best_kinematic = float(
            getattr(best_field, "min_kinematic_constraint", np.nan)
        )
        kinematic_requires_velocity_reserve = bool(
            np.isfinite(best_kinematic)
            and best_kinematic < kinematic_projection_target
        )
        if (
            safe_solved is None
            and nominal_target_velocity > 0.0
            and (
                best_field.margin_delta < self.mgf_projection_tolerance
                or obstacle_requires_velocity_reserve
                or progress_requires_velocity_search
                or spatial_requires_velocity_reserve
                or kinematic_requires_velocity_reserve
            )
        ):
            # A fixed cruise-speed objective can make a geometrically viable
            # braking target look infeasible at a sharp routed turn.  Search
            # a small state-dependent velocity bank before spending the much
            # larger global action-grid budget.  The lower-level target speed
            # is not an RL action dimension; every returned trajectory still
            # receives the same exact spatial/curvature certificate.
            #
            # The reserve is an optional recursive-feasibility buffer, not a
            # second safety threshold.  If a nominal-speed candidate is
            # strictly certified and road/curvature is its active constraint,
            # keep that velocity: reducing speed merely to gain spatial reserve
            # can turn a routed bend into an absorbing stop.  When the moving
            # obstacle margin itself is below the reserve, however, it can
            # become active on the next frame even if curvature is the current
            # minimum.  Search lower speeds proactively; only an exactly
            # certified trajectory can replace the nominal one.
            adaptive_actions = [
                np.asarray(best["action"], dtype=np.float32),
                *[np.asarray(action, dtype=np.float32) for action in candidates[:6]],
                np.asarray([-1.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 0.0], dtype=np.float32),
            ]
            # At each candidate speed, evaluate forward route anchors before
            # a certified stop.  The former first-feasible ordering made a
            # stopped target absorb the search even when an equally certified
            # forward continuation existed at that same lower speed.
            adaptive_actions = sorted(
                adaptive_actions,
                key=lambda action: (
                    is_route_action(action),
                    float(np.asarray(action)[0]),
                ),
                reverse=True,
            )
            nominal_best = best
            nominal_best_is_safe = bool(
                nominal_best["field"].margin_delta >= self.mgf_projection_tolerance
            )
            adaptive_seen = set()
            for velocity_fraction in ADAPTIVE_VELOCITY_FRACTIONS:
                adaptive_velocity = float(nominal_target_velocity * velocity_fraction)
                velocity_reserve_solutions = []
                for adaptive_action in adaptive_actions:
                    key = (
                        round(adaptive_velocity, 6),
                        *tuple(adaptive_action.tolist()),
                    )
                    if key in adaptive_seen:
                        continue
                    adaptive_seen.add(key)
                    seen.add(candidate_key(adaptive_action, adaptive_velocity))
                    candidate_solved = self._solve_target_candidate(
                        adaptive_action,
                        target_velocity=adaptive_velocity,
                    )
                    solves += 1
                    diagnostics["candidates"].append(diagnostic_row(candidate_solved))
                    if (
                        candidate_solved["field"].margin_delta
                        > best["field"].margin_delta
                        and (
                            not nominal_best_is_safe
                            or projection_reserve_met(candidate_solved["field"])
                        )
                    ):
                        best = candidate_solved
                    if projection_reserve_met(candidate_solved["field"]):
                        velocity_reserve_solutions.append(candidate_solved)
                if velocity_reserve_solutions:
                    raw_action = np.asarray(raw_solved["action"], dtype=np.float32)
                    safe_solved = max(
                        velocity_reserve_solutions,
                        key=lambda solved: (
                            is_route_action(solved["action"]),
                            float(np.asarray(solved["action"])[0]),
                            float(solved["field"].margin_delta),
                            -float(
                                np.linalg.norm(
                                    np.asarray(solved["action"], dtype=np.float32)
                                    - raw_action
                                )
                            ),
                        ),
                    )
                    break
            if safe_solved is None and nominal_best_is_safe:
                # The lower-speed sweep is diagnostic unless it actually
                # finds the requested reserve.  Do not replace a certified
                # cruise-speed continuation with a slower high-margin stop
                # merely because its scalar margin is larger.
                best = nominal_best

        # The reserve is deliberately larger than the strict execution
        # tolerance.  Once an unsafe raw action has a strictly certified
        # replacement, reserve maximization must not turn into an exhaustive
        # scan of the whole target domain.  If the fast bank has not recovered
        # safety, search the coarse global grid only until its first certified
        # point; local exact solves below then improve that point with a fixed
        # budget.  The exhaustive refinements and alternate warm starts remain
        # available when no certified point exists at all.
        if (
            safe_solved is None
            and not reserve_only
            and best["field"].margin_delta < self.mgf_projection_tolerance
        ):
            coarse_candidates = (
                np.array([longitudinal, lateral], dtype=np.float32)
                for longitudinal in np.linspace(-1.0, 1.0, 9)
                for lateral in (
                    0.0,
                    -0.25,
                    0.25,
                    -0.5,
                    0.5,
                    -0.75,
                    0.75,
                    -1.0,
                    1.0,
                )
            )
            for coarse_action in coarse_candidates:
                key = candidate_key(coarse_action, None)
                if key in seen:
                    continue
                seen.add(key)
                candidate_solved = self._solve_target_candidate(coarse_action)
                solves += 1
                diagnostics["candidates"].append(diagnostic_row(candidate_solved))
                if candidate_solved["field"].margin_delta > best["field"].margin_delta:
                    best = candidate_solved
                if projection_reserve_met(candidate_solved["field"]):
                    safe_solved = candidate_solved
                    break
                if candidate_solved["field"].margin_delta >= self.mgf_projection_tolerance:
                    break

        # A coarse target grid can straddle the intersection of the road and
        # curvature feasible sets on a tight bend.  Refine around the exact
        # best candidate without relaxing either certificate component.
        if safe_solved is None:
            already_certified = bool(
                best["field"].margin_delta >= self.mgf_projection_tolerance
            )
            directions = (
                (
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (0.0, -1.0),
                    (0.0, 1.0),
                )
                if already_certified
                else (
                    (-1.0, 0.0),
                    (1.0, 0.0),
                    (0.0, -1.0),
                    (0.0, 1.0),
                    (-1.0, -1.0),
                    (-1.0, 1.0),
                    (1.0, -1.0),
                    (1.0, 1.0),
                )
            )
            refinement_radii = (
                (0.125, 0.0625, 0.03125)
                if already_certified
                else (
                    0.125,
                    0.0625,
                    0.03125,
                    0.015625,
                    0.0078125,
                    0.00390625,
                    0.001953125,
                    0.0009765625,
                    0.00048828125,
                    0.000244140625,
                    0.0001220703125,
                )
            )
            for radius in refinement_radii:
                center = np.asarray(best["action"], dtype=np.float32)
                center_velocity = float(
                    best.get("target_velocity", nominal_target_velocity)
                )
                radius_best = best
                for dx, dy in directions:
                    local_action = np.clip(
                        center + radius * np.array([dx, dy], dtype=np.float32),
                        -1.0,
                        1.0,
                    )
                    key = candidate_key(local_action, center_velocity)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate_solved = solve_with_candidate_velocity(
                        local_action,
                        best,
                    )
                    solves += 1
                    diagnostics["candidates"].append(diagnostic_row(candidate_solved))
                    if (
                        candidate_solved["field"].margin_delta
                        > radius_best["field"].margin_delta
                    ):
                        radius_best = candidate_solved
                    if projection_reserve_met(candidate_solved["field"]):
                        safe_solved = candidate_solved
                        break
                if radius_best["field"].margin_delta > best["field"].margin_delta:
                    best = radius_best
                if safe_solved is not None:
                    break
                if (
                    not already_certified
                    and best["field"].margin_delta >= self.mgf_projection_tolerance
                ):
                    # Safety has been recovered.  Continue reserve improvement
                    # through the bounded gradient phase, not the remaining
                    # sub-millimetre exhaustive stencil.
                    break

        # Follow the smooth composite certificate gradient in continuous
        # target space.  This closes the narrow, oblique obstacle/curvature
        # intersections that an axis/diagonal grid can straddle.  Every trial
        # is still an exact MPC solve and is accepted only by its true margin.
        if safe_solved is None:
            gradient_is_reserve_only = bool(
                best["field"].margin_delta >= self.mgf_projection_tolerance
            )
            for _ in range(2 if gradient_is_reserve_only else 10):
                center = np.asarray(best["action"], dtype=np.float32)
                center_velocity = float(
                    best.get("target_velocity", nominal_target_velocity)
                )
                gradient = np.asarray(
                    getattr(best["field"], "grad_margin", np.zeros(2)),
                    dtype=np.float64,
                ).reshape(2)
                gradient_norm = float(np.linalg.norm(gradient))
                if not np.all(np.isfinite(gradient)) or gradient_norm <= 1e-8:
                    break
                direction = (gradient / gradient_norm).astype(np.float32)
                required_step = max(
                    (projection_target - float(best["field"].margin_delta))
                    / gradient_norm,
                    0.0,
                )
                trial_steps = (
                    [
                        float(
                            np.clip(1.25 * required_step, 4.8828125e-4, 0.125)
                        ),
                        0.0625,
                        0.03125,
                    ]
                    if gradient_is_reserve_only
                    else [
                        float(
                            np.clip(1.25 * required_step, 4.8828125e-4, 0.125)
                        ),
                        0.0625,
                        0.03125,
                        0.015625,
                        0.0078125,
                        0.00390625,
                        0.001953125,
                        0.0009765625,
                        0.00048828125,
                    ]
                )
                iteration_best = best
                for trial_step in dict.fromkeys(trial_steps):
                    local_action = np.clip(
                        center + float(trial_step) * direction,
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                    key = candidate_key(local_action, center_velocity)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate_solved = solve_with_candidate_velocity(
                        local_action,
                        best,
                    )
                    solves += 1
                    diagnostics["candidates"].append(diagnostic_row(candidate_solved))
                    if (
                        candidate_solved["field"].margin_delta
                        > iteration_best["field"].margin_delta
                    ):
                        iteration_best = candidate_solved
                    if projection_reserve_met(candidate_solved["field"]):
                        safe_solved = candidate_solved
                        break
                if safe_solved is not None:
                    break
                if (
                    iteration_best["field"].margin_delta
                    <= best["field"].margin_delta + 1e-8
                ):
                    break
                best = iteration_best

        # Coarse/local target refinement can recover a strictly safe action
        # after the earlier velocity-reserve phase has already run.  Recheck
        # the moving-obstacle reserve here: otherwise a low-margin recovery
        # can be executed at cruise speed and become infeasible one frame
        # later.  This late pass is narrow and shares ``seen`` with the first
        # pass, so already-evaluated action/velocity pairs are not repeated.
        late_best_field = best["field"]
        late_obstacle = float(
            getattr(late_best_field, "min_obstacle_constraint", np.nan)
        )
        if (
            safe_solved is None
            and nominal_target_velocity > 0.0
            and late_best_field.margin_delta >= self.mgf_projection_tolerance
            and late_best_field.margin_delta < projection_target
            and np.isfinite(late_obstacle)
            and late_obstacle < projection_target
        ):
            late_adaptive_actions = [
                np.asarray(best["action"], dtype=np.float32),
                np.asarray(raw_solved["action"], dtype=np.float32),
                np.asarray([-1.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 0.0], dtype=np.float32),
            ]
            if self._prev_target_action is not None:
                late_adaptive_actions.insert(
                    1, np.asarray(self._prev_target_action, dtype=np.float32)
                )
            for velocity_fraction in ADAPTIVE_VELOCITY_FRACTIONS:
                adaptive_velocity = float(
                    nominal_target_velocity * velocity_fraction
                )
                for adaptive_action in late_adaptive_actions:
                    key = candidate_key(adaptive_action, adaptive_velocity)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate_solved = self._solve_target_candidate(
                        adaptive_action,
                        target_velocity=adaptive_velocity,
                    )
                    solves += 1
                    diagnostics["candidates"].append(
                        diagnostic_row(candidate_solved)
                    )
                    if (
                        candidate_solved["field"].margin_delta
                        > best["field"].margin_delta
                    ):
                        best = candidate_solved
                    if projection_reserve_met(candidate_solved["field"]):
                        safe_solved = candidate_solved
                        break
                if safe_solved is not None:
                    break

        # The non-convex SQP can inherit a locally infeasible curvature/road
        # basin from the previous receding-horizon trajectory.  Before
        # declaring the certified set empty, solve the most promising targets
        # from independent reference and zero trajectory initializations.
        # These are full exact MPC solves; only a positively certified result
        # can be selected.
        if (
            safe_solved is None
            and best["field"].margin_delta < self.mgf_projection_tolerance
        ):
            alternate_candidates = [
                (np.asarray(best["action"], dtype=np.float32), best),
                (np.asarray(raw_solved["action"], dtype=np.float32), raw_solved),
            ]
            if self._prev_target_action is not None:
                alternate_candidates.append(
                    (
                        np.asarray(self._prev_target_action, dtype=np.float32),
                        raw_solved,
                    )
                )
            alternate_candidates.extend(
                [
                    (
                        np.array([-1.0, 0.0], dtype=np.float32),
                        raw_solved,
                    ),
                    (
                        np.array([0.0, 0.0], dtype=np.float32),
                        raw_solved,
                    ),
                    (
                        np.array([1.0, 0.0], dtype=np.float32),
                        raw_solved,
                    ),
                ]
            )
            # Low target speeds alter the non-convex SQP basin substantially.
            # The fast adaptive bank above intentionally uses only the warm
            # receding-horizon trajectory; if all normal searches fail, retry
            # the same small action bank from independent trajectories at each
            # adaptive velocity before claiming the exact feasible set empty.
            if nominal_target_velocity > 0.0:
                velocity_bank = (
                    nominal_target_velocity,
                    *(
                        nominal_target_velocity * fraction
                        for fraction in ADAPTIVE_VELOCITY_FRACTIONS
                    ),
                )
                velocity_candidates = []
                for alternate_action, source_solved in alternate_candidates:
                    source_velocity = float(
                        source_solved.get(
                            "target_velocity", nominal_target_velocity
                        )
                    )
                    for alternate_velocity in (
                        source_velocity,
                        *velocity_bank,
                    ):
                        velocity_candidates.append(
                            (
                                alternate_action,
                                {"target_velocity": alternate_velocity},
                            )
                        )
                alternate_candidates = velocity_candidates
            alternate_seen = set()
            for alternate_action, source_solved in alternate_candidates:
                alternate_velocity = float(
                    source_solved.get("target_velocity", nominal_target_velocity)
                )
                for warm_start_mode in ("reference", "zero"):
                    key = (
                        warm_start_mode,
                        round(alternate_velocity, 6),
                        *tuple(np.asarray(alternate_action, dtype=np.float32).tolist()),
                    )
                    if key in alternate_seen:
                        continue
                    alternate_seen.add(key)
                    candidate_solved = solve_with_candidate_velocity(
                        alternate_action,
                        source_solved,
                        warm_start_mode=warm_start_mode,
                    )
                    solves += 1
                    diagnostics["candidates"].append(diagnostic_row(candidate_solved))
                    if candidate_solved["field"].margin_delta > best["field"].margin_delta:
                        best = candidate_solved
                    if projection_reserve_met(candidate_solved["field"]):
                        safe_solved = candidate_solved
                        break
                if safe_solved is not None:
                    break

        if safe_solved is None and best["field"].margin_delta >= self.mgf_projection_tolerance:
            # The requested recursive-feasibility reserve is unavailable, but
            # a certified point still exists.  Execute the maximum-margin
            # point instead of moving it back to the certificate boundary.
            diagnostics["selected"] = diagnostic_row(best)
            diagnostics["projection_reserve_met"] = False
            best["projection_diagnostics"] = diagnostics
            return best, True, False, solves

        if safe_solved is None:
            # Preserve the least unsafe exact MPC solution and report failure;
            # the trainer's replay assertion stops instead of silently storing
            # an uncertified transition.
            diagnostics["selected"] = diagnostic_row(best)
            best["projection_diagnostics"] = diagnostics
            return best, True, True, solves

        low_action = safe_solved["action"].copy()
        low_solved = safe_solved
        high_action = raw_solved["action"].copy()
        warm_start_mode = safe_solved.get("warm_start_mode", "previous")
        safe_target_velocity = float(
            safe_solved.get("target_velocity", nominal_target_velocity)
        )
        for _ in range(self.mgf_projection_steps):
            midpoint = 0.5 * (low_action + high_action)
            velocity_aware = (
                "target_velocity" in safe_solved or nominal_target_velocity > 0.0
            )
            if warm_start_mode == "previous" and not velocity_aware:
                midpoint_solved = self._solve_target_candidate(midpoint)
            elif warm_start_mode == "previous":
                midpoint_solved = self._solve_target_candidate(
                    midpoint,
                    target_velocity=safe_target_velocity,
                )
            elif not velocity_aware:
                midpoint_solved = self._solve_target_candidate(
                    midpoint,
                    warm_start_mode=warm_start_mode,
                )
            else:
                midpoint_solved = self._solve_target_candidate(
                    midpoint,
                    warm_start_mode=warm_start_mode,
                    target_velocity=safe_target_velocity,
                )
            solves += 1
            if projection_reserve_met(midpoint_solved["field"]):
                low_action = midpoint
                low_solved = midpoint_solved
            else:
                high_action = midpoint
        diagnostics["selected"] = diagnostic_row(low_solved)
        diagnostics["projection_reserve_met"] = True
        low_solved["projection_diagnostics"] = diagnostics
        return low_solved, True, False, solves

    def certify_target_action(self, trajectory: np.ndarray) -> Dict:
        """Solve and project a target without advancing MetaDrive.

        The returned certificate is tied to the current wrapper step and can
        be passed to :meth:`execute_certified_target`.  This public split lets
        supervisory controllers inspect the actor's *raw* margin and reuse the
        exact same solve for execution instead of solving the NMPC twice.
        """

        if (
            self.planner_action_mode != "target_point"
            or not self.enable_mpc_optimization
            or self.mpc is None
        ):
            raise RuntimeError(
                "certify_target_action requires optimized target-point mode"
            )
        raw_action = self._prepare_high_level_action(trajectory)
        raw_solved = self._solve_target_candidate(raw_action)
        solved, projection_used, projection_failed, certificate_solves = (
            self._project_target_candidate(raw_solved)
        )
        return {
            "step_count": int(self._step_count),
            "raw_action": raw_action.copy(),
            "raw_solved": raw_solved,
            "solved": solved,
            "projection_used": bool(projection_used),
            "projection_failed": bool(projection_failed),
            "certificate_solves": int(certificate_solves),
            "raw_hard_certificate": self._hard_certificate_summary(
                raw_solved["hard_certificate"]
            ),
            "executed_hard_certificate": self._hard_certificate_summary(
                solved["hard_certificate"]
            ),
        }

    def certify_unprotected_target_action(self, trajectory: np.ndarray) -> Dict:
        """Solve a target with ordinary MPC and no terminal projection.

        This is intentionally narrow: it supports a fixed post-reset sensor
        warm-up frame that must be identical in protected and ordinary runs.
        Actor decisions continue to use :meth:`certify_target_action`.
        """

        if (
            self.planner_action_mode != "target_point"
            or not self.enable_mpc_optimization
            or self.mpc is None
        ):
            raise RuntimeError(
                "certify_unprotected_target_action requires optimized "
                "target-point mode"
            )
        raw_action = self._prepare_high_level_action(trajectory)
        solved = self._solve_target_candidate(
            raw_action,
            obstacle_cbf_enabled=False,
        )
        summary = self._hard_certificate_summary(solved["hard_certificate"])
        return {
            "step_count": int(self._step_count),
            "raw_action": raw_action.copy(),
            "raw_solved": solved,
            "solved": solved,
            "projection_used": False,
            "projection_failed": False,
            "certificate_solves": 1,
            "raw_hard_certificate": summary,
            "executed_hard_certificate": summary,
        }

    def execute_certified_target(
        self, certificate: Dict
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute a same-step certificate produced by ``certify_target_action``."""

        if int(certificate.get("step_count", -1)) != int(self._step_count):
            raise ValueError("stale target certificate: environment state has advanced")
        raw_action = np.asarray(certificate["raw_action"], dtype=np.float32).copy()
        solved = certificate["solved"]
        projection_used = bool(certificate["projection_used"])
        projection_failed = bool(certificate["projection_failed"])
        certificate_solves = int(certificate["certificate_solves"])
        if self.target_speed_lateral_action and projection_failed:
            raise RuntimeError(
                "target_speed_lateral_v1 refuses to execute an "
                "uncertified terminal projection"
            )
        projection_diagnostics = solved.get("projection_diagnostics", {})
        executed_action = solved["action"].copy()
        reachable_lower, reachable_upper = self.reachable_speed_interval()
        actual_vehicle_speed = float(solved["x0"][3])
        raw_target_speed = self._target_velocity_for_action(raw_action)
        executed_target_speed = float(
            solved.get(
                "target_velocity",
                self._target_velocity_for_action(executed_action),
            )
        )
        decoded_executed_speed = self._target_velocity_for_action(
            executed_action
        )
        emergency_stop = bool(solved.get("emergency_stop", False))
        if self.explicit_speed_action and not emergency_stop and not np.isclose(
            executed_target_speed,
            decoded_executed_speed,
            atol=1e-6,
            rtol=0.0,
        ):
            raise RuntimeError(
                "explicit-speed certificate target speed does not match "
                "executed action[0]"
            )
        speed_audit = target_speed_intervention_audit(
            raw_action,
            executed_action,
            raw_target_speed=raw_target_speed,
            executed_target_speed=executed_target_speed,
            action_semantics_version=self.action_semantics_version,
            action_tolerance=self.mgf_projection_tolerance,
            emergency_stop=emergency_stop,
        )
        result = solved["result"]
        field = solved["field"]
        self._cache_flat_reference(
            executed_action,
            solved["x0"],
            solved["ref_free"],
            solved["ref_path"],
        )
        self._prev_free_ctrl_pts = np.asarray(
            result.opt_free_ctrl_pts, dtype=np.float32
        )
        self._prev_target_action = executed_action.copy()
        mpc_traj_xy = np.stack([result.predicted_x, result.predicted_y], axis=1)
        obs, reward, terminated, truncated, info = self._apply_control_step(
            executed_action,
            result.steering,
            result.throttle,
            solved["ref_path"],
            planner_action_kind="target_point",
            # Preserve the certificate/model contract: do not apply the
            # legacy throttle boost or steering slew heuristic after MPC.
            boost_target_velocity=None,
            info_target_velocity=result.target_velocity_exec,
            apply_steer_rate_limit=False,
            mpc_result=result,
            executed_path_xy=mpc_traj_xy,
            route_xy=solved["route_xy"],
            route_normal=solved["route_normal"],
            route_half_width=solved["route_half_width"],
            route_mask=solved["route_mask"],
        )
        mgf_info = field.as_info()
        obstacle_mask = np.asarray(
            solved.get("obs_mask", np.zeros((0,), dtype=bool)), dtype=bool
        )
        obstacle_xy = np.asarray(
            solved.get("obs_xy", np.zeros((0, 4))), dtype=np.float32
        )
        active_obstacle_x = (
            obstacle_xy[obstacle_mask, 0]
            if obstacle_xy.ndim == 2
            and obstacle_xy.shape[0] == obstacle_mask.shape[0]
            else np.zeros((0,), dtype=np.float32)
        )
        mgf_info.update(
            {
                "raw_target": raw_action.tolist(),
                "executed_target": executed_action.tolist(),
                "projection_used": projection_used,
                "projection_failed": projection_failed,
                "projection_norm": float(
                    np.linalg.norm(executed_action - raw_action)
                ),
                **speed_audit,
                "reachable_speed_lower": float(reachable_lower),
                "reachable_speed_upper": float(reachable_upper),
                "actual_vehicle_speed": actual_vehicle_speed,
                "action_semantics_version": self.action_semantics_version,
                "certificate_mpc_solves": certificate_solves,
                "projection_diagnostics": projection_diagnostics,
                "obstacle_cbf_enabled": bool(
                    solved.get("obstacle_cbf_enabled", True)
                ),
                "solver_finite": bool(solved.get("solver_finite", True)),
                "hard_safety": self._hard_certificate_summary(
                    solved["hard_certificate"]
                ),
                "raw_hard_safety": self._hard_certificate_summary(
                    certificate["raw_solved"]["hard_certificate"]
                ),
                "teacher_ordinary_mpc": bool(
                    certificate.get("teacher_ordinary_mpc", False)
                ),
                "cbf_trajectory_delta": float(
                    certificate.get("teacher_trajectory_rmse", np.nan)
                ),
                "teacher_action": np.asarray(
                    certificate.get("teacher_action", executed_action),
                    dtype=np.float32,
                ).tolist(),
                "obstacle_count": int(active_obstacle_x.size),
                "min_obstacle_longitudinal": (
                    float(np.min(active_obstacle_x))
                    if active_obstacle_x.size
                    else float("nan")
                ),
                "max_obstacle_longitudinal": (
                    float(np.max(active_obstacle_x))
                    if active_obstacle_x.size
                    else float("nan")
                ),
            }
        )
        if projection_failed:
            obstacle_radius = np.asarray(
                solved.get("obs_radius", np.zeros((0,))), dtype=np.float32
            )
            active = np.flatnonzero(obstacle_mask)
            mgf_info["failure_obstacles_ego"] = [
                {
                    "position": obstacle_xy[index, :2].tolist(),
                    "horizon_displacement": obstacle_xy[index, 2:4].tolist(),
                    "radius": float(obstacle_radius[index]),
                }
                for index in active
            ]
        info["mgf"] = mgf_info
        info["m_delta"] = float(field.margin_delta)
        info["raw_target"] = raw_action.copy()
        info["executed_target"] = executed_action.copy()
        info.update(
            {
                **self.action_semantics_metadata,
                "reachable_speed_lower": float(reachable_lower),
                "reachable_speed_upper": float(reachable_upper),
                "actual_vehicle_speed": actual_vehicle_speed,
                **speed_audit,
            }
        )
        self._cached_mgf_field_step = -1
        self._cached_mgf_field_action = None
        self._cached_mgf_field = None
        return obs, reward, terminated, truncated, info

    def step(self, trajectory: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        trajectory = self._prepare_high_level_action(trajectory)
        if self.planner_action_mode == "target_point" and self.enable_mpc_optimization and self.mpc is not None:
            return self.execute_certified_target(
                self.certify_target_action(trajectory)
            )

        x0 = np.array([0.0, 0.0, 0.0, self.vehicle_state.velocity], dtype=np.float32)

        ref_free_ctrl_pts, ref_path_xy = self._get_flat_reference(trajectory, x0, require_path=True)
        route_xy, route_normal, route_half_width, route_mask = self._get_route_corridor_ego_frame()
        if self.enable_mpc_optimization and self.mpc is not None:
            obs_xy, obs_radius, obs_mask = self._get_obstacles_ego_frame()
            mpc_result = self.mpc.step(
                x0,
                ref_free_ctrl_pts,
                prev_free_ctrl_pts=self._prev_free_ctrl_pts,
                obs_xy=obs_xy,
                obs_radius=obs_radius,
                obs_mask=obs_mask,
                route_xy=route_xy,
                route_normal=route_normal,
                route_half_width=route_half_width,
                route_mask=route_mask,
            )
            self._prev_free_ctrl_pts = np.asarray(mpc_result.opt_free_ctrl_pts, dtype=np.float32)
            mpc_traj_xy = np.stack([mpc_result.predicted_x, mpc_result.predicted_y], axis=1)
            return self._apply_control_step(
                trajectory,
                mpc_result.steering,
                mpc_result.throttle,
                ref_path_xy,
                planner_action_kind="free_ctrl_pts",
                boost_target_velocity=resolve_flat_boost_target_velocity(
                    mpc_result.target_velocity_exec,
                    self.target_velocity,
                ),
                info_target_velocity=mpc_result.target_velocity_exec,
                mpc_result=mpc_result,
                executed_path_xy=mpc_traj_xy,
                route_xy=route_xy,
                route_normal=route_normal,
                route_half_width=route_half_width,
                route_mask=route_mask,
            )
        return self._apply_control_step(
            trajectory,
            0.0,
            0.3,
            ref_path_xy,
            planner_action_kind="free_ctrl_pts",
            boost_target_velocity=None,
            info_target_velocity=None,
            route_xy=route_xy,
            route_normal=route_normal,
            route_half_width=route_half_width,
            route_mask=route_mask,
        )

    def get_mpc_inputs(self, trajectory: np.ndarray):
        trajectory = self._prepare_high_level_action(trajectory)
        x0 = np.array([0.0, 0.0, 0.0, self.vehicle_state.velocity], dtype=np.float32)

        ref_free_ctrl_pts, _ = self._get_flat_reference(trajectory, x0, require_path=False)
        obs_xy, obs_radius, obs_mask = self._get_obstacles_ego_frame()
        route_xy, route_normal, route_half_width, route_mask = self._get_route_corridor_ego_frame()
        return x0, ref_free_ctrl_pts, obs_xy, obs_radius, obs_mask, route_xy, route_normal, route_half_width, route_mask

    def get_current_hj_inputs(self):
        x0 = np.array([0.0, 0.0, 0.0, self.vehicle_state.velocity], dtype=np.float32)
        obs_xy, obs_radius, obs_mask = self._get_obstacles_ego_frame()
        route_xy, route_normal, route_half_width, route_mask = self._get_route_corridor_ego_frame()
        return x0, obs_xy, obs_radius, obs_mask, route_xy, route_normal, route_half_width, route_mask

    def _refresh_current_hj(self):
        if self.planner_action_mode == "target_point":
            # AE-FLAC's control-point Hessian is intentionally not exposed as a
            # target-space metric. MGF uses get_current_mgf_field instead.
            self._current_hj = np.eye(self.traj_dim, dtype=np.float32)
            self._last_metric_fallback_hit = False
            return
        if self.mpc is None:
            self._current_hj = np.eye(self.traj_dim, dtype=np.float32)
            return
        prev_metric = None if self._current_hj is None else np.asarray(self._current_hj, dtype=np.float32)
        x0, obs_xy, obs_radius, obs_mask, route_xy, route_normal, route_half_width, route_mask = self.get_current_hj_inputs()
        metric = self.mpc.get_current_hj(
            x0,
            prev_free_ctrl_pts=self._prev_free_ctrl_pts,
            obs_xy=obs_xy,
            obs_radius=obs_radius,
            obs_mask=obs_mask,
            route_xy=route_xy,
            route_normal=route_normal,
            route_half_width=route_half_width,
            route_mask=route_mask,
        )
        self._last_metric_fallback_hit = not np.all(np.isfinite(np.asarray(metric, dtype=np.float32)))
        self._current_hj = sanitize_metric_np(
            metric,
            act_dim=self.traj_dim,
            fallback=prev_metric,
        )

    def get_current_hj(self):
        if self._current_hj is None:
            self._refresh_current_hj()
        return np.asarray(self._current_hj, dtype=np.float32).copy()

    def _apply_control_step(
        self,
        trajectory_raw: np.ndarray,
        steering: float,
        throttle: float,
        ref_path_xy: np.ndarray,
        planner_action_kind: str,
        boost_target_velocity: Optional[float],
        info_target_velocity: Optional[float],
        apply_steer_rate_limit: bool = True,
        mpc_result=None,
        executed_path_xy: Optional[np.ndarray] = None,
        route_xy: Optional[np.ndarray] = None,
        route_normal: Optional[np.ndarray] = None,
        route_half_width: Optional[np.ndarray] = None,
        route_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        steering = float(np.clip(steering, -1.0, 1.0))
        if apply_steer_rate_limit:
            steer_delta = steering - self.prev_steering
            steering = self.prev_steering + float(
                np.clip(steer_delta, -self.max_steer_rate, self.max_steer_rate)
            )
        throttle = apply_terminal_throttle_boost(throttle, boost_target_velocity, self.vehicle_state.velocity)

        low_level_action = np.clip(np.array([steering, throttle], dtype=np.float32), -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(low_level_action)

        lateral_pen = 0.0
        if self.lane_penalty_weight > 0.0:
            lateral_pen = float(self.lane_penalty_weight * abs(self.vehicle_state.lateral_offset))
            reward -= lateral_pen

        planned_centerline_pen = 0.0
        if self.planned_centerline_penalty_weight > 0.0:
            try:
                if planner_action_kind == "free_ctrl_pts":
                    ys = np.asarray(ref_path_xy, dtype=np.float32)[:, 1]
                else:
                    traj_points = np.asarray(trajectory_raw, dtype=np.float32).reshape(-1, WAYPOINT_DIM)
                    ys = traj_points[:, 1] * self.position_scale
                lane_width = 3.7
                try:
                    _lane = getattr(self.env.unwrapped.vehicle, "lane", None)
                    if _lane is not None and hasattr(_lane, "width"):
                        lane_width = float(_lane.width)
                except Exception:
                    pass
                half_w = max(lane_width * 0.5, 1e-3)
                planned_centerline_pen = float(
                    self.planned_centerline_penalty_weight * np.mean((ys / half_w) ** 2)
                )
                reward -= planned_centerline_pen
            except Exception:
                planned_centerline_pen = 0.0

        self._update_vehicle_state(obs, info)
        current_road_clearance = self._get_current_road_clearance()
        physical_body_safety = self.get_physical_body_safety()

        road_penalty = 0.0
        if self.road_penalty_weight > 0.0:
            road_penalty = float(
                self.road_penalty_weight
                * _road_penalty_from_clearance_np(
                    current_road_clearance,
                    road_buffer=self.road_buffer,
                    road_temperature=self.road_temperature,
                )
            )
            reward -= road_penalty

        planned_road_penalty = 0.0
        if (
            self.planned_road_penalty_weight > 0.0
            and route_xy is not None
            and route_normal is not None
            and route_half_width is not None
            and route_mask is not None
        ):
            planned_road_cost, _planned_road_clearance = _compute_route_road_cost_np(
                ref_path_xy,
                route_xy,
                route_normal,
                route_half_width,
                route_mask,
                road_buffer=self.road_buffer,
                road_temperature=self.road_temperature,
                exclude_initial=True,
            )
            planned_road_penalty = float(self.planned_road_penalty_weight * planned_road_cost)
            reward -= planned_road_penalty

        self.prev_steering = float(steering)

        ref_path_xy = np.asarray(ref_path_xy, dtype=np.float32)
        exec_path_xy = ref_path_xy
        if executed_path_xy is not None:
            exec_path_xy = np.asarray(executed_path_xy, dtype=np.float32)
        self.prev_result = SimpleMPCResult(
            steering=steering,
            throttle=throttle,
            trajectory_x=exec_path_xy[:, 0],
            trajectory_y=exec_path_xy[:, 1],
        )
        self._refresh_current_hj()

        info["mpc_control"] = {
            "trajectory": np.asarray(trajectory_raw, dtype=np.float32).tolist(),
            "planner_action_kind": planner_action_kind,
            "low_level_action": low_level_action.tolist(),
            "current_velocity": float(self.vehicle_state.velocity),
            "mpc_cost_total": float(mpc_result.cost_total) if mpc_result is not None else 0.0,
            "mpc_cost_tracking": float(mpc_result.cost_tracking) if mpc_result is not None else 0.0,
            "mpc_cost_cbf": float(mpc_result.cost_cbf) if mpc_result is not None else 0.0,
            "mpc_cost_road": float(mpc_result.cost_road) if mpc_result is not None else 0.0,
            "mpc_max_kappa_violation": float(mpc_result.max_kappa_violation) if mpc_result is not None else 0.0,
            "mpc_min_kappa_slack": float(mpc_result.min_kappa_slack) if mpc_result is not None else 0.0,
            "mpc_min_obstacle_dist": float(mpc_result.min_obstacle_dist) if mpc_result is not None else 0.0,
            "mpc_min_cbf_value": float(mpc_result.min_cbf_value) if mpc_result is not None else 0.0,
            "mpc_min_road_clearance": float(mpc_result.min_road_clearance) if mpc_result is not None else 0.0,
            "mpc_traj_x": self.prev_result.trajectory_x.tolist(),
            "mpc_traj_y": self.prev_result.trajectory_y.tolist(),
            "mpc_ref_traj_x": ref_path_xy[:, 0].tolist(),
            "mpc_ref_traj_y": ref_path_xy[:, 1].tolist(),
            "lateral_penalty": lateral_pen,
            "planned_centerline_penalty": planned_centerline_pen,
            "road_penalty": road_penalty,
            "planned_road_penalty": planned_road_penalty,
            "current_road_clearance": float(current_road_clearance),
            "physical_body_sat_margin": float(
                physical_body_safety["sat_margin"]
            ),
            "physical_nearest_center_distance": float(
                physical_body_safety["nearest_center_distance"]
            ),
            "physical_obstacle_count": int(
                physical_body_safety["obstacle_count"]
            ),
        }
        try:
            vehicle = self.env.unwrapped.vehicle
            navigation = getattr(vehicle, "navigation", None)
            lane = getattr(vehicle, "lane", None)
            lane_longitudinal = float("nan")
            lane_lateral = float("nan")
            if lane is not None and hasattr(lane, "local_coordinates"):
                lane_longitudinal, lane_lateral = map(
                    float, lane.local_coordinates(vehicle.position)
                )
            world_position = np.asarray(vehicle.position, dtype=np.float64).reshape(-1)
            info["mpc_control"].update(
                {
                    "vehicle_on_lane": bool(getattr(vehicle, "on_lane", False)),
                    "vehicle_out_of_route": bool(getattr(vehicle, "out_of_route", False)),
                    "on_yellow_continuous_line": bool(
                        getattr(vehicle, "on_yellow_continuous_line", False)
                    ),
                    "on_white_continuous_line": bool(
                        getattr(vehicle, "on_white_continuous_line", False)
                    ),
                    "crash_sidewalk": bool(getattr(vehicle, "crash_sidewalk", False)),
                    "route_completion": float(getattr(navigation, "route_completion", float("nan"))),
                    "world_x": float(world_position[0]),
                    "world_y": float(world_position[1]),
                    "lane_longitudinal": lane_longitudinal,
                    "lane_lateral": lane_lateral,
                    "lane_length": float(getattr(lane, "length", float("nan"))),
                }
            )
        except Exception:
            pass
        if info_target_velocity is not None:
            info["mpc_control"]["target_velocity_exec"] = float(info_target_velocity)
        info["mpc_control"]["vehicle_state"] = {
            "lateral_offset": float(self.vehicle_state.lateral_offset),
            "heading_error": float(self.vehicle_state.heading_error),
            "velocity": float(self.vehicle_state.velocity),
        }
        info["ae_flac"] = {
            "next_hj": self.get_current_hj().tolist(),
            "metric_fallback_hit": bool(self._last_metric_fallback_hit),
        }

        obs, obstacle_features = self._inject_explicit_obstacle_observation(obs)
        info["mpc_control"]["explicit_obstacle_features"] = (
            obstacle_features.tolist()
        )
        info["mpc_control"]["explicit_obstacle_observation_enabled"] = bool(
            self.explicit_obstacle_observation
        )

        self._step_count += 1
        return obs, reward, terminated, truncated, info

    def apply_mpc_step(
        self,
        trajectory_raw: np.ndarray,
        steering: float,
        throttle: float,
        target_velocity_exec: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        trajectory = self._prepare_high_level_action(trajectory_raw)
        x0 = np.array([0.0, 0.0, 0.0, self.vehicle_state.velocity], dtype=np.float32)

        _, ref_path_xy = self._get_flat_reference(trajectory, x0, require_path=True)
        route_xy, route_normal, route_half_width, route_mask = self._get_route_corridor_ego_frame()
        return self._apply_control_step(
            trajectory,
            steering,
            throttle,
            ref_path_xy,
            planner_action_kind=self.planner_action_mode,
            boost_target_velocity=None,
            info_target_velocity=target_velocity_exec,
            route_xy=route_xy,
            route_normal=route_normal,
            route_half_width=route_half_width,
            route_mask=route_mask,
        )

    def _update_vehicle_state(self, obs: np.ndarray, info: Dict):
        del obs
        if "velocity" in info:
            self.vehicle_state.velocity = float(info["velocity"])
        elif "speed" in info:
            self.vehicle_state.velocity = float(info["speed"])
        else:
            try:
                vehicle = self.env.unwrapped.vehicle
                if hasattr(vehicle, "speed_km_h"):
                    self.vehicle_state.velocity = float(vehicle.speed_km_h / 3.6)
            except Exception:
                pass

        if "lateral_offset" in info:
            self.vehicle_state.lateral_offset = float(info["lateral_offset"])
        else:
            try:
                vehicle = self.env.unwrapped.vehicle
                lane = getattr(vehicle, "lane", None)
                if lane is not None and hasattr(lane, "local_coordinates"):
                    _, lateral = lane.local_coordinates(vehicle.position)
                    self.vehicle_state.lateral_offset = float(lateral)
            except Exception:
                pass

        if "heading_error" in info:
            self.vehicle_state.heading_error = float(info["heading_error"])
        else:
            try:
                vehicle = self.env.unwrapped.vehicle
                lane = getattr(vehicle, "lane", None)
                if lane is not None and hasattr(lane, "local_coordinates") and hasattr(lane, "heading_theta_at"):
                    s, _ = lane.local_coordinates(vehicle.position)
                    lane_heading = float(lane.heading_theta_at(s))
                    if hasattr(vehicle, "heading_theta"):
                        ego_heading = float(vehicle.heading_theta)
                    else:
                        h_vec = vehicle.heading
                        ego_heading = float(np.arctan2(h_vec[1], h_vec[0]))
                    err = ego_heading - lane_heading
                    self.vehicle_state.heading_error = float(np.arctan2(np.sin(err), np.cos(err)))
            except Exception:
                pass


MPCControlWrapper = FastMPCControlWrapper


def _make_flat_metadrive_mpc_env_with_env_cls(
    env_cls,
    use_render: bool = False,
    **mpc_kwargs,
) -> Env:
    render_mode = mpc_kwargs.pop("render_mode", None)
    if render_mode is not None:
        use_render = use_render or (render_mode != "rgb_array")
    env = env_cls(
        {
            "use_render": use_render,
            "log_level": logging.ERROR,
        }
    )
    env = FlattenObservation(env)
    env = FastMPCControlWrapper(env, **mpc_kwargs)
    return env


def make_flat_metadrive_mpc_env(
    use_render: bool = False,
    **mpc_kwargs,
) -> Env:
    return _make_flat_metadrive_mpc_env_with_env_cls(
        ThreeLaneStraightEnv,
        use_render=use_render,
        **mpc_kwargs,
    )


def make_flat_metadrive_multi_obs_mpc_env(
    use_render: bool = False,
    **mpc_kwargs,
) -> Env:
    return _make_flat_metadrive_mpc_env_with_env_cls(
        ThreeLaneStraightMultiObsEnv,
        use_render=use_render,
        **mpc_kwargs,
    )


def make_flat_metadrive_mpc_env_ctrlpts(
    use_render: bool = False,
    **mpc_kwargs,
) -> Env:
    return make_flat_metadrive_mpc_env(use_render=use_render, **mpc_kwargs)
