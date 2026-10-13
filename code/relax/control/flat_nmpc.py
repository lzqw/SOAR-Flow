"""
Unified Flat NMPC module for the control-point MPC path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Literal, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from relax.control.bspline_utils import (
    apply_initial_constraint,
    bspline_eval_with_derivs_scaled,
    make_clamped_knots,
)


N_EVAL_POINTS = 5
EPS = 1e-6
NO_OBSTACLE_METRIC = 1e6
ADMM_RHO = 1.0
MAX_SQP_DX = 4.0
METRIC_EIGEN_FLOOR = 1e-4
ROUTE_ASSIGNMENT_SCALE = 2.0
METRIC_RISK_SMOOTHMAX = "smoothmax"
MetricRiskAggregation = Literal["smoothmax"]

CFG_HORIZON_TIME = 0
CFG_WHEELBASE = 1
CFG_MAX_STEERING = 2
CFG_EPS = 3
CFG_V_MIN = 4
CFG_KAPPA_MAX = 5
CFG_A_MIN = 6
CFG_A_MAX = 7
CFG_W_TRACKING = 8
CFG_W_RISK = 9
CFG_W_CBF = 10
CFG_RISK_BUFFER = 11
CFG_RISK_TEMPERATURE = 12
CFG_D_CLEARANCE = 13
CFG_CBF_GAMMA = 14
CFG_CBF_SOLVE_WEIGHT_SCALE = 15
CFG_W_ROAD = 16
CFG_ROAD_BUFFER = 17
CFG_ROAD_TEMPERATURE = 18
CFG_ROAD_SOLVE_WEIGHT_SCALE = 19
CFG_EGO_HALF_WIDTH = 20
CFG_EGO_HALF_LENGTH = 21
CFG_TARGET_VELOCITY = 22
CFG_W_VELOCITY = 23


@dataclass(frozen=True)
class FlatNMPCConfig:
    n_ctrl_points: int = 5
    spline_degree: int = 3
    horizon_time: float = 0.5
    n_eval_points: int = N_EVAL_POINTS

    wheelbase: float = 2.5
    max_steering: float = 0.6
    v_min: float = 1.0
    kappa_max: float = 0.24
    a_min: float = -5.0
    a_max: float = 3.0

    w_tracking: float = 10.0
    # Keep zero as the compatibility default.  The real-drive target-point
    # entry point enables this term so its advertised target velocity is part
    # of the MPC value landscape instead of being metadata only.
    w_velocity: float = 0.0
    w_risk: float = 0.0
    w_cbf: float = 0.0
    cbf_solve_weight_scale: float = 0.0
    w_road: float = 0.0
    road_solve_weight_scale: float = 0.0
    risk_buffer: float = 1.0
    risk_temperature: float = 0.3
    road_buffer: float = 0.5
    road_temperature: float = 0.5
    metric_risk_aggregation: MetricRiskAggregation = METRIC_RISK_SMOOTHMAX

    d_clearance: float = 2.0
    cbf_gamma: float = 1.0
    max_obstacles: int = 4
    max_route_samples: int = 16
    cbf_enable: bool = False
    n_sqp_steps: int = 5
    n_admm_steps: int = 50

    # Zero preserves the legacy point-mass solver unless the real drive
    # wrapper explicitly enables its physical footprint.
    ego_half_width: float = 0.0
    ego_half_length: float = 0.0

    eps: float = EPS
    target_velocity: float = 15.0


def _normalize_metric_risk_aggregation(metric_risk_aggregation: str | None) -> MetricRiskAggregation:
    del metric_risk_aggregation
    return METRIC_RISK_SMOOTHMAX


class FlatNMPCResult(NamedTuple):
    steering: float
    throttle: float
    target_velocity_exec: float
    cost_total: float
    cost_tracking: float
    cost_cbf: float
    cost_road: float
    max_kappa_violation: float
    min_kappa_slack: float
    min_obstacle_dist: float
    min_cbf_value: float
    min_road_clearance: float
    opt_free_ctrl_pts: np.ndarray
    full_ctrl_pts: np.ndarray
    predicted_x: np.ndarray
    predicted_y: np.ndarray
    hj: np.ndarray
    solver_time: float


class FlatNMPCSafetyCertificate(NamedTuple):
    """Hard safety diagnostics evaluated independently of solver authority."""

    min_hard_cbf_margin: float
    min_obstacle_clearance: float
    min_road_clearance: float
    min_kinematic_slack: float
    hard_cbf_margin: np.ndarray
    obstacle_clearance: np.ndarray
    road_clearance: np.ndarray
    kinematic_slack: np.ndarray
    has_obstacle: bool
    has_route: bool
    finite: bool


def _pack_config(cfg: FlatNMPCConfig) -> jnp.ndarray:
    return jnp.array(
        [
            cfg.horizon_time,
            cfg.wheelbase,
            cfg.max_steering,
            cfg.eps,
            cfg.v_min,
            cfg.kappa_max,
            cfg.a_min,
            cfg.a_max,
            cfg.w_tracking,
            cfg.w_risk,
            cfg.w_cbf,
            cfg.risk_buffer,
            cfg.risk_temperature,
            cfg.d_clearance,
            cfg.cbf_gamma,
            cfg.cbf_solve_weight_scale,
            cfg.w_road,
            cfg.road_buffer,
            cfg.road_temperature,
            cfg.road_solve_weight_scale,
            cfg.ego_half_width,
            cfg.ego_half_length,
            cfg.target_velocity,
            cfg.w_velocity,
        ],
        dtype=jnp.float32,
    )


def _stabilize_metric(hessian: jnp.ndarray) -> jnp.ndarray:
    sym = 0.5 * (hessian + jnp.swapaxes(hessian, -1, -2))
    diag = jnp.diagonal(sym, axis1=-2, axis2=-1)
    off_diag_abs_sum = jnp.sum(jnp.abs(sym), axis=-1) - jnp.abs(diag)
    gershgorin_lower = diag - off_diag_abs_sum
    min_lower = jnp.min(gershgorin_lower, axis=-1)
    shift = jnp.maximum(METRIC_EIGEN_FLOOR - min_lower, 0.0)
    eye = jnp.eye(sym.shape[-1], dtype=sym.dtype)
    return sym + shift[..., None, None] * eye


def _safe_pivot_value(x: jnp.ndarray) -> jnp.ndarray:
    sign = jnp.where(x < 0.0, -1.0, 1.0).astype(x.dtype)
    return jnp.where(jnp.abs(x) < METRIC_EIGEN_FLOOR, sign * METRIC_EIGEN_FLOOR, x)


def _swap_rows(mat: jnp.ndarray, i: int, j: jnp.ndarray) -> jnp.ndarray:
    row_i = mat[i]
    row_j = mat[j]
    mat = mat.at[i].set(row_j)
    return mat.at[j].set(row_i)


def _gauss_jordan_inverse(mat: jnp.ndarray) -> jnp.ndarray:
    n = mat.shape[0]
    eye = jnp.eye(n, dtype=mat.dtype)
    aug = jnp.concatenate([mat, eye], axis=1)

    def eliminate_column(k: int, aug_mat: jnp.ndarray) -> jnp.ndarray:
        col_abs = jnp.abs(aug_mat[:, k])
        valid_rows = jnp.arange(n) >= k
        pivot_row = jnp.argmax(jnp.where(valid_rows, col_abs, -jnp.ones_like(col_abs)))
        aug_mat = _swap_rows(aug_mat, k, pivot_row)

        pivot = _safe_pivot_value(aug_mat[k, k])
        aug_mat = aug_mat.at[k].set(aug_mat[k] / pivot)
        pivot_row_values = aug_mat[k]

        def eliminate_row(i: int, current: jnp.ndarray) -> jnp.ndarray:
            factor = jnp.where(i == k, 0.0, current[i, k]).astype(current.dtype)
            return current.at[i].set(current[i] - factor * pivot_row_values)

        return jax.lax.fori_loop(0, n, eliminate_row, aug_mat)

    aug = jax.lax.fori_loop(0, n, eliminate_column, aug)
    return aug[:, n:]


def _flatness_map(
    dxy: jnp.ndarray,
    ddxy: jnp.ndarray,
    wheelbase: float,
    eps: float,
    v_min: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dx, dy = dxy[:, 0], dxy[:, 1]
    ddx, ddy = ddxy[:, 0], ddxy[:, 1]

    v_sq = dx * dx + dy * dy + eps
    v = jnp.maximum(jnp.sqrt(v_sq), v_min)
    theta = jnp.arctan2(dy, dx)
    kappa = (dx * ddy - dy * ddx) / (v**3 + eps)
    delta = jnp.arctan(wheelbase * kappa)
    accel = (dx * ddx + dy * ddy) / (v + eps)
    return v, theta, kappa, delta, accel


def _build_eval_terms(
    free_pts: jnp.ndarray,
    x0_state: jnp.ndarray,
    config_arr: jnp.ndarray,
    knots: jnp.ndarray,
    t_eval: jnp.ndarray,
    degree: int,
):
    horizon_time = config_arr[0]
    wheelbase = config_arr[1]
    eps = config_arr[3]
    v_min = config_arr[4]

    full_pts = apply_initial_constraint(free_pts, x0_state, knots, horizon_time, degree)
    xy_eval, dxy_eval, ddxy_eval = bspline_eval_with_derivs_scaled(
        t_eval,
        full_pts,
        knots,
        horizon_time,
        degree,
    )
    v_eval, _, kappa_eval, delta_eval, accel_eval = _flatness_map(
        dxy_eval,
        ddxy_eval,
        wheelbase,
        eps,
        v_min,
    )
    return full_pts, xy_eval, dxy_eval, ddxy_eval, v_eval, kappa_eval, delta_eval, accel_eval


def _compute_distance_risk_cost(
    dist_sq: jnp.ndarray,
    d_safe: jnp.ndarray,
    obs_mask: jnp.ndarray,
    risk_buffer: float,
    risk_temperature: float,
) -> jnp.ndarray:
    d_warn = d_safe + risk_buffer
    warn_margin = d_warn**2 - dist_sq
    masked_warn_margin = jnp.where(obs_mask[None, :], warn_margin, -NO_OBSTACLE_METRIC)
    temp = jnp.maximum(risk_temperature, EPS)
    nearest_warn_margin = temp * jsp.special.logsumexp(masked_warn_margin / temp, axis=1)
    risk_per_step = jax.nn.softplus(nearest_warn_margin / temp) ** 2
    return jnp.where(jnp.any(obs_mask), jnp.mean(risk_per_step), 0.0)


def _compute_obstacle_geometry_terms(
    xy_eval: jnp.ndarray,
    dxy_eval: jnp.ndarray,
    obs_xy: jnp.ndarray,
    obs_radius: jnp.ndarray,
    d_clearance: jnp.ndarray,
    cbf_gamma: jnp.ndarray,
    ego_half_width: jnp.ndarray,
    ego_half_length: jnp.ndarray,
    horizon_time: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    xy_future = xy_eval[1:]
    dxy_future = dxy_eval[1:]
    obstacle_position = obs_xy[:, :2]
    if obs_xy.shape[-1] >= 4:
        # Columns 2:4 encode the obstacle displacement over the complete MPC
        # horizon.  Interpolation keeps the JIT shape fixed while making the
        # hard distance constraints time aligned with moving traffic.
        fractions = jnp.linspace(
            1.0 / max(xy_future.shape[0], 1),
            1.0,
            xy_future.shape[0],
            dtype=xy_future.dtype,
        )
        obstacle_position = (
            obstacle_position[None, :, :]
            + fractions[:, None, None] * obs_xy[None, :, 2:4]
        )
    else:
        obstacle_position = obstacle_position[None, :, :]
    diff = xy_future[:, None, :] - obstacle_position
    dist_sq = jnp.sum(diff**2, axis=-1)
    dist = jnp.sqrt(jnp.maximum(dist_sq, EPS))
    # Clamp the squared norm before sqrt.  Clamping only after sqrt leaves an
    # infinite derivative at an exactly stationary spline point; even when an
    # obstacle is masked (or the footprint weight is zero), second-order JAX
    # differentiation can then produce 0 * inf = NaN in the MPC Hessian.
    tangent_norm = jnp.sqrt(
        jnp.maximum(
            jnp.sum(dxy_future**2, axis=-1, keepdims=True),
            EPS**2,
        )
    )
    tangent = dxy_future / tangent_norm
    radial = diff / jnp.maximum(dist[:, :, None], EPS)
    longitudinal_projection = jnp.abs(
        jnp.sum(tangent[:, None, :] * radial, axis=-1)
    )
    lateral_projection = jnp.abs(
        tangent[:, None, 0] * radial[:, :, 1]
        - tangent[:, None, 1] * radial[:, :, 0]
    )
    ego_support = (
        ego_half_length * longitudinal_projection
        + ego_half_width * lateral_projection
    )
    d_safe = obs_radius[None, :] + ego_support + d_clearance
    h = dist - d_safe
    # ``obs_xy[:, 2:4]`` is displacement across the full prediction horizon,
    # so its derivative is the obstacle velocity.  CBF h-dot must use relative
    # velocity; using ego velocity alone is only correct for parked obstacles.
    obstacle_velocity = (
        obs_xy[:, 2:4] / jnp.maximum(horizon_time, EPS)
        if obs_xy.shape[-1] >= 4
        else jnp.zeros_like(obstacle_position[0])
    )
    relative_velocity = dxy_future[:, None, :] - obstacle_velocity[None, :, :]
    dh_dt = jnp.sum(diff * relative_velocity, axis=-1) / jnp.maximum(dist, EPS)
    cbf_margin = dh_dt + cbf_gamma * h
    return dist_sq, d_safe, cbf_margin


def _compute_cbf_soft_cost(
    cbf_margin: jnp.ndarray,
    obs_mask: jnp.ndarray,
    temperature: float,
    dist_sq: jnp.ndarray | None = None,
    d_safe: jnp.ndarray | None = None,
    risk_buffer: float = 1.0,
) -> jnp.ndarray:
    temp = jnp.maximum(temperature, EPS)
    violation = -cbf_margin
    masked_violation = jnp.where(obs_mask[None, :], violation, -NO_OBSTACLE_METRIC)
    per_step_violation = temp * jsp.special.logsumexp(masked_violation / temp, axis=1)
    cbf_violation_prob = jax.nn.sigmoid(per_step_violation / temp)
    cbf_penalty = cbf_violation_prob**2
    if dist_sq is not None and d_safe is not None:
        dist = jnp.sqrt(jnp.maximum(dist_sq, EPS))
        clearance = dist - d_safe
        near_margin = risk_buffer - clearance
        masked_near_margin = jnp.where(obs_mask[None, :], near_margin, -NO_OBSTACLE_METRIC)
        per_step_near_margin = temp * jsp.special.logsumexp(masked_near_margin / temp, axis=1)
        cbf_penalty = cbf_penalty * jax.nn.sigmoid(per_step_near_margin / temp)
    return jnp.where(jnp.any(obs_mask), jnp.mean(cbf_penalty), 0.0)


def _smooth_distance_from_squared(squared_distance: jnp.ndarray) -> jnp.ndarray:
    """Euclidean distance with an exact zero and finite derivatives.

    ``sqrt(max(d2, EPS))`` creates a fictitious 1 mm route offset at the
    centreline.  A small C1 quadratic patch matches ``sqrt`` in value and
    first derivative at ``EPS`` while remaining exactly zero at ``d2 == 0``.
    """

    squared_distance = jnp.maximum(squared_distance, 0.0)
    eps = jnp.asarray(EPS, dtype=squared_distance.dtype)
    normalized = jnp.clip(squared_distance / eps, 0.0, 1.0)
    near_zero = jnp.sqrt(eps) * (
        1.5 * normalized - 0.5 * normalized**2
    )
    regular = jnp.sqrt(jnp.maximum(squared_distance, eps))
    return jnp.where(squared_distance < eps, near_zero, regular)


def _compute_route_edge_geometry(
    xy_eval: jnp.ndarray,
    route_xy: jnp.ndarray,
    route_normal: jnp.ndarray,
    route_half_width: jnp.ndarray,
    route_mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    xy_future = xy_eval[1:]
    if xy_future.shape[0] == 0:
        return (
            jnp.zeros((0,), dtype=jnp.float32),
            jnp.zeros((0, 2), dtype=jnp.float32),
        )

    # Distance to the finite piecewise-linear centerline.  The prior
    # normal-weighted formulation described an infinite strip and could claim
    # positive clearance after a curved road segment had already ended.
    p0 = route_xy[:-1]
    p1 = route_xy[1:]
    segment = p1 - p0
    segment_len_sq = jnp.maximum(jnp.sum(segment * segment, axis=-1), EPS)
    rel = xy_future[:, None, :] - p0[None, :, :]
    fraction = jnp.clip(
        jnp.sum(rel * segment[None, :, :], axis=-1) / segment_len_sq[None, :],
        0.0,
        1.0,
    )
    closest = p0[None, :, :] + fraction[:, :, None] * segment[None, :, :]
    distance = _smooth_distance_from_squared(
        jnp.sum((xy_future[:, None, :] - closest) ** 2, axis=-1)
    )
    width = (
        route_half_width[:-1][None, :] * (1.0 - fraction)
        + route_half_width[1:][None, :] * fraction
    )
    segment_clearance = width - distance
    segment_mask = route_mask[:-1] & route_mask[1:]
    masked_segment_clearance = jnp.where(
        segment_mask[None, :], segment_clearance, -NO_OBSTACLE_METRIC
    )
    best_segment = jnp.argmax(masked_segment_clearance, axis=1)
    clearance = jnp.take_along_axis(
        masked_segment_clearance, best_segment[:, None], axis=1
    )[:, 0]
    segment_tangent = segment / jnp.sqrt(
        jnp.maximum(jnp.sum(segment * segment, axis=-1, keepdims=True), EPS)
    )
    tangent = segment_tangent[best_segment]
    point_distance = _smooth_distance_from_squared(
        jnp.sum((xy_future[:, None, :] - route_xy[None, :, :]) ** 2, axis=-1)
    )
    masked_point_clearance = jnp.where(
        route_mask[None, :],
        route_half_width[None, :] - point_distance,
        -NO_OBSTACLE_METRIC,
    )
    best_point = jnp.argmax(masked_point_clearance, axis=1)
    point_clearance = jnp.take_along_axis(
        masked_point_clearance, best_point[:, None], axis=1
    )[:, 0]
    # Route normals are supplied by the map descriptor.  Either perpendicular
    # orientation gives the same rectangular support because only absolute
    # dot/cross products are used below.
    point_normal = route_normal[best_point]
    point_tangent = jnp.stack([point_normal[:, 1], -point_normal[:, 0]], axis=1)
    point_tangent = point_tangent / jnp.sqrt(
        jnp.maximum(jnp.sum(point_tangent * point_tangent, axis=1, keepdims=True), EPS)
    )
    clearance = jnp.where(jnp.any(segment_mask), clearance, point_clearance)
    tangent = jnp.where(jnp.any(segment_mask), tangent, point_tangent)
    has_route = jnp.any(route_mask)
    return (
        jnp.where(has_route, clearance, jnp.zeros_like(clearance)),
        jnp.where(has_route, tangent, jnp.zeros_like(tangent)),
    )


def _compute_route_edge_clearance(
    xy_eval: jnp.ndarray,
    route_xy: jnp.ndarray,
    route_normal: jnp.ndarray,
    route_half_width: jnp.ndarray,
    route_mask: jnp.ndarray,
) -> jnp.ndarray:
    return _compute_route_edge_geometry(
        xy_eval, route_xy, route_normal, route_half_width, route_mask
    )[0]


def _compute_body_road_clearance(
    xy_eval: jnp.ndarray,
    dxy_eval: jnp.ndarray,
    route_xy: jnp.ndarray,
    route_normal: jnp.ndarray,
    route_half_width: jnp.ndarray,
    route_mask: jnp.ndarray,
    ego_half_width: jnp.ndarray,
    ego_half_length: jnp.ndarray,
) -> jnp.ndarray:
    """Road-edge clearance of the oriented rectangular ego footprint."""
    center_clearance, route_tangent = _compute_route_edge_geometry(
        xy_eval, route_xy, route_normal, route_half_width, route_mask
    )
    path_tangent = dxy_eval[1:]
    path_tangent = path_tangent / jnp.sqrt(
        jnp.maximum(jnp.sum(path_tangent * path_tangent, axis=1, keepdims=True), EPS)
    )
    aligned = jnp.abs(jnp.sum(path_tangent * route_tangent, axis=1))
    crossed = jnp.abs(
        path_tangent[:, 0] * route_tangent[:, 1]
        - path_tangent[:, 1] * route_tangent[:, 0]
    )
    body_support = ego_half_width * aligned + ego_half_length * crossed
    return center_clearance - body_support


def _compute_road_soft_cost(
    road_clearance: jnp.ndarray,
    route_mask: jnp.ndarray,
    road_buffer: float,
    road_temperature: float,
) -> jnp.ndarray:
    temp = jnp.maximum(road_temperature, EPS)
    road_penalty = jax.nn.sigmoid((road_buffer - road_clearance) / temp) ** 2
    return jnp.where(jnp.any(route_mask), jnp.mean(road_penalty), 0.0)


def _objective_from_free(
    free_pts: jnp.ndarray,
    x0_state: jnp.ndarray,
    ref_free_ctrl_pts: jnp.ndarray,
    config_arr: jnp.ndarray,
    knots: jnp.ndarray,
    t_eval: jnp.ndarray,
    obs_xy: jnp.ndarray,
    obs_radius: jnp.ndarray,
    obs_mask: jnp.ndarray,
    route_xy: jnp.ndarray,
    route_normal: jnp.ndarray,
    route_half_width: jnp.ndarray,
    route_mask: jnp.ndarray,
    degree: int,
    cbf_weight_scale: float | jnp.ndarray = 1.0,
    road_weight_scale: float | jnp.ndarray = 1.0,
    target_velocity: jnp.ndarray | None = None,
) -> jnp.ndarray:
    w_tracking = config_arr[CFG_W_TRACKING]
    w_velocity = config_arr[CFG_W_VELOCITY]
    w_risk = config_arr[CFG_W_RISK]
    w_cbf = config_arr[CFG_W_CBF]
    risk_buffer = config_arr[CFG_RISK_BUFFER]
    risk_temperature = config_arr[CFG_RISK_TEMPERATURE]
    d_clearance = config_arr[CFG_D_CLEARANCE]
    cbf_gamma = config_arr[CFG_CBF_GAMMA]
    w_road = config_arr[CFG_W_ROAD]
    road_buffer = config_arr[CFG_ROAD_BUFFER]
    road_temperature = config_arr[CFG_ROAD_TEMPERATURE]

    _, xy_eval, dxy_eval, _ddxy_eval, _v_eval, _kappa_eval, _delta_eval, _accel_eval = _build_eval_terms(
        free_pts,
        x0_state,
        config_arr,
        knots,
        t_eval,
        degree,
    )
    dist_sq, d_safe, cbf_margin = _compute_obstacle_geometry_terms(
        xy_eval,
        dxy_eval,
        obs_xy,
        obs_radius,
        d_clearance,
        cbf_gamma,
        config_arr[CFG_EGO_HALF_WIDTH],
        config_arr[CFG_EGO_HALF_LENGTH],
        config_arr[CFG_HORIZON_TIME],
    )

    cost_tracking = jnp.sum((free_pts - ref_free_ctrl_pts) ** 2)
    # Exclude the fixed initial state: penalizing it adds a constant and can
    # dominate diagnostics without changing the optimizer.  The remaining
    # horizon speed makes short/stopping point references costly whenever the
    # task calls for cruising, preventing value guidance from collapsing to
    # the nearest target solely because it is easiest to track.
    # The horizon lives in the current ego frame, so positive d(x)/dt is the
    # task's forward velocity.  Tracking its signed value (and damping lateral
    # velocity) is smooth with a useful gradient even at a stationary spline;
    # using norm(dxy) would have a zero-gradient cusp exactly at the stopping
    # collapse this term is intended to prevent.
    velocity_target = (
        config_arr[CFG_TARGET_VELOCITY]
        if target_velocity is None
        else target_velocity
    )
    velocity_error = dxy_eval[1:, 0] - velocity_target
    cost_velocity = jnp.mean(velocity_error**2 + 0.25 * dxy_eval[1:, 1] ** 2)
    cost_risk = _compute_distance_risk_cost(
        dist_sq,
        d_safe,
        obs_mask,
        risk_buffer=risk_buffer,
        risk_temperature=risk_temperature,
    )
    cost_cbf = _compute_cbf_soft_cost(
        cbf_margin,
        obs_mask,
        risk_temperature,
        dist_sq=dist_sq,
        d_safe=d_safe,
        risk_buffer=risk_buffer,
    )
    road_clearance = _compute_body_road_clearance(
        xy_eval,
        dxy_eval,
        route_xy,
        route_normal,
        route_half_width,
        route_mask,
        config_arr[CFG_EGO_HALF_WIDTH],
        config_arr[CFG_EGO_HALF_LENGTH],
    )
    cost_road = _compute_road_soft_cost(
        road_clearance,
        route_mask,
        road_buffer=road_buffer,
        road_temperature=road_temperature,
    )
    return (
        w_tracking * cost_tracking
        + w_velocity * cost_velocity
        + w_risk * cost_risk
        + (w_cbf * cbf_weight_scale) * cost_cbf
        + (w_road * road_weight_scale) * cost_road
    )


def _compute_constraint_terms(
    free_pts: jnp.ndarray,
    x0_state: jnp.ndarray,
    config_arr: jnp.ndarray,
    knots: jnp.ndarray,
    t_eval: jnp.ndarray,
    obs_xy: jnp.ndarray,
    obs_radius: jnp.ndarray,
    obs_mask: jnp.ndarray,
    route_xy: jnp.ndarray,
    route_normal: jnp.ndarray,
    route_half_width: jnp.ndarray,
    route_mask: jnp.ndarray,
    degree: int,
):
    kappa_max = config_arr[5]
    d_clearance = config_arr[13]
    cbf_gamma = config_arr[14]

    full_pts, xy_eval, dxy_eval, _ddxy_eval, _v_eval, kappa_eval, delta_eval, accel_eval = _build_eval_terms(
        free_pts,
        x0_state,
        config_arr,
        knots,
        t_eval,
        degree,
    )

    dist_sq, d_safe, cbf_margin = _compute_obstacle_geometry_terms(
        xy_eval,
        dxy_eval,
        obs_xy,
        obs_radius,
        d_clearance,
        cbf_gamma,
        config_arr[CFG_EGO_HALF_WIDTH],
        config_arr[CFG_EGO_HALF_LENGTH],
        config_arr[CFG_HORIZON_TIME],
    )

    g_kappa_pos = kappa_eval - kappa_max
    g_kappa_neg = -kappa_eval - kappa_max
    # The QP convention is g(x) <= 0.  For the safe-positive barrier
    # h = distance - d_safe, exponential CBF feasibility is
    # h_dot + gamma*h >= 0, hence g_cbf = -(h_dot + gamma*h) <= 0.
    # Distance remains a diagnostic/soft-cost input; it is intentionally not
    # duplicated as a second hard constraint.  Forward invariance is defined
    # solely by this CBF inequality.
    g_cbf = jnp.where(obs_mask[None, :], -cbf_margin, -NO_OBSTACLE_METRIC)
    road_clearance = _compute_body_road_clearance(
        xy_eval,
        dxy_eval,
        route_xy,
        route_normal,
        route_half_width,
        route_mask,
        config_arr[CFG_EGO_HALF_WIDTH],
        config_arr[CFG_EGO_HALF_LENGTH],
    )
    # Keep the complete oriented ego rectangle inside the routed road.  A
    # fixed half-width misses a front/rear corner whenever the vehicle has a
    # heading error relative to the road (the exact failure reported by
    # MetaDrive's continuous-line contact detector).
    g_road = -road_clearance
    g_road = jnp.where(jnp.any(route_mask), g_road, -NO_OBSTACLE_METRIC)

    constraints = jnp.concatenate(
        [g_kappa_pos, g_kappa_neg, g_cbf.reshape(-1), g_road.reshape(-1)]
    )
    masked_dist = jnp.where(obs_mask[None, :], jnp.sqrt(jnp.maximum(dist_sq, 0.0)), NO_OBSTACLE_METRIC)
    masked_cbf_margin = jnp.where(obs_mask[None, :], cbf_margin, NO_OBSTACLE_METRIC)

    return (
        full_pts,
        xy_eval,
        delta_eval,
        accel_eval,
        constraints,
        jnp.max(jax.nn.relu(jnp.abs(kappa_eval) - kappa_max)),
        kappa_max - jnp.max(jnp.abs(kappa_eval)),
        jnp.min(masked_dist),
        jnp.min(masked_cbf_margin),
    )


def _compute_hard_safety_certificate_impl(
    free_pts: jnp.ndarray,
    x0_state: jnp.ndarray,
    config_arr: jnp.ndarray,
    knots: jnp.ndarray,
    t_eval: jnp.ndarray,
    obs_xy: jnp.ndarray,
    obs_radius: jnp.ndarray,
    obs_mask: jnp.ndarray,
    route_xy: jnp.ndarray,
    route_normal: jnp.ndarray,
    route_half_width: jnp.ndarray,
    route_mask: jnp.ndarray,
    degree: int,
):
    """Evaluate physical margins without changing the optimized trajectory.

    This is intentionally separate from ``_solve_flat_nmpc_sqp_impl``.  It
    can therefore verify a trajectory produced by an ordinary MPC solve with
    obstacle CBF constraints disabled, which is the measurement needed to
    distinguish physical safety from dependence on the safety filter.
    """

    (
        _full_pts,
        xy_eval,
        dxy_eval,
        _ddxy_eval,
        _v_eval,
        kappa_eval,
        _delta_eval,
        _accel_eval,
    ) = _build_eval_terms(
        free_pts,
        x0_state,
        config_arr,
        knots,
        t_eval,
        degree,
    )
    dist_sq, d_safe, hard_cbf_margin = _compute_obstacle_geometry_terms(
        xy_eval,
        dxy_eval,
        obs_xy,
        obs_radius,
        config_arr[CFG_D_CLEARANCE],
        config_arr[CFG_CBF_GAMMA],
        config_arr[CFG_EGO_HALF_WIDTH],
        config_arr[CFG_EGO_HALF_LENGTH],
        config_arr[CFG_HORIZON_TIME],
    )
    obstacle_clearance = jnp.sqrt(jnp.maximum(dist_sq, 0.0)) - d_safe
    masked_hard_cbf = jnp.where(
        obs_mask[None, :], hard_cbf_margin, NO_OBSTACLE_METRIC
    )
    masked_obstacle_clearance = jnp.where(
        obs_mask[None, :], obstacle_clearance, NO_OBSTACLE_METRIC
    )
    road_clearance = _compute_body_road_clearance(
        xy_eval,
        dxy_eval,
        route_xy,
        route_normal,
        route_half_width,
        route_mask,
        config_arr[CFG_EGO_HALF_WIDTH],
        config_arr[CFG_EGO_HALF_LENGTH],
    )
    masked_road_clearance = jnp.where(
        jnp.any(route_mask), road_clearance, NO_OBSTACLE_METRIC
    )
    kinematic_slack = config_arr[CFG_KAPPA_MAX] - jnp.abs(kappa_eval)
    return (
        jnp.min(masked_hard_cbf),
        jnp.min(masked_obstacle_clearance),
        jnp.min(masked_road_clearance),
        jnp.min(kinematic_slack),
        masked_hard_cbf,
        masked_obstacle_clearance,
        masked_road_clearance,
        kinematic_slack,
    )


def _compute_objective_breakdown(
    free_pts: jnp.ndarray,
    x0_state: jnp.ndarray,
    ref_free_ctrl_pts: jnp.ndarray,
    config_arr: jnp.ndarray,
    knots: jnp.ndarray,
    t_eval: jnp.ndarray,
    obs_xy: jnp.ndarray,
    obs_radius: jnp.ndarray,
    obs_mask: jnp.ndarray,
    route_xy: jnp.ndarray,
    route_normal: jnp.ndarray,
    route_half_width: jnp.ndarray,
    route_mask: jnp.ndarray,
    degree: int,
    target_velocity: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    w_tracking = config_arr[CFG_W_TRACKING]
    w_velocity = config_arr[CFG_W_VELOCITY]
    w_risk = config_arr[CFG_W_RISK]
    w_cbf = config_arr[CFG_W_CBF]
    risk_buffer = config_arr[CFG_RISK_BUFFER]
    risk_temperature = config_arr[CFG_RISK_TEMPERATURE]
    d_clearance = config_arr[CFG_D_CLEARANCE]
    cbf_gamma = config_arr[CFG_CBF_GAMMA]
    cbf_solve_weight_scale = config_arr[CFG_CBF_SOLVE_WEIGHT_SCALE]
    w_road = config_arr[CFG_W_ROAD]
    road_buffer = config_arr[CFG_ROAD_BUFFER]
    road_temperature = config_arr[CFG_ROAD_TEMPERATURE]
    road_solve_weight_scale = config_arr[CFG_ROAD_SOLVE_WEIGHT_SCALE]

    _, xy_eval, dxy_eval, _ddxy_eval, _v_eval, _kappa_eval, _delta_eval, _accel_eval = _build_eval_terms(
        free_pts,
        x0_state,
        config_arr,
        knots,
        t_eval,
        degree,
    )
    dist_sq, d_safe, cbf_margin = _compute_obstacle_geometry_terms(
        xy_eval,
        dxy_eval,
        obs_xy,
        obs_radius,
        d_clearance,
        cbf_gamma,
        config_arr[CFG_EGO_HALF_WIDTH],
        config_arr[CFG_EGO_HALF_LENGTH],
        config_arr[CFG_HORIZON_TIME],
    )

    cost_tracking = jnp.sum((free_pts - ref_free_ctrl_pts) ** 2)
    velocity_error = dxy_eval[1:, 0] - target_velocity
    cost_velocity = jnp.mean(velocity_error**2 + 0.25 * dxy_eval[1:, 1] ** 2)
    cost_risk = _compute_distance_risk_cost(
        dist_sq,
        d_safe,
        obs_mask,
        risk_buffer=risk_buffer,
        risk_temperature=risk_temperature,
    )
    cost_cbf = _compute_cbf_soft_cost(
        cbf_margin,
        obs_mask,
        risk_temperature,
        dist_sq=dist_sq,
        d_safe=d_safe,
        risk_buffer=risk_buffer,
    )
    road_clearance = _compute_body_road_clearance(
        xy_eval,
        dxy_eval,
        route_xy,
        route_normal,
        route_half_width,
        route_mask,
        config_arr[CFG_EGO_HALF_WIDTH],
        config_arr[CFG_EGO_HALF_LENGTH],
    )
    cost_road = _compute_road_soft_cost(
        road_clearance,
        route_mask,
        road_buffer=road_buffer,
        road_temperature=road_temperature,
    )
    return (
        w_tracking * cost_tracking
        + w_velocity * cost_velocity
        + w_risk * cost_risk
        + (w_cbf * cbf_solve_weight_scale) * cost_cbf
        + (w_road * road_solve_weight_scale) * cost_road,
        cost_tracking,
        cost_cbf,
        cost_road,
    )


def _compute_objective_metric_impl(
    free_ctrl_pts: jnp.ndarray,
    x0_state: jnp.ndarray,
    ref_free_ctrl_pts: jnp.ndarray,
    config_arr: jnp.ndarray,
    knots: jnp.ndarray,
    t_eval: jnp.ndarray,
    obs_xy: jnp.ndarray,
    obs_radius: jnp.ndarray,
    obs_mask: jnp.ndarray,
    route_xy: jnp.ndarray,
    route_normal: jnp.ndarray,
    route_half_width: jnp.ndarray,
    route_mask: jnp.ndarray,
    degree: int,
) -> jnp.ndarray:
    free_shape = free_ctrl_pts.shape

    def objective_from_flat(flat_pts: jnp.ndarray) -> jnp.ndarray:
        return _objective_from_free(
            flat_pts.reshape(free_shape),
            x0_state,
            ref_free_ctrl_pts,
            config_arr,
            knots,
            t_eval,
            obs_xy,
            obs_radius,
            obs_mask,
            route_xy,
            route_normal,
            route_half_width,
            route_mask,
            degree,
            1.0,
            1.0,
        )

    return _stabilize_metric(jax.hessian(objective_from_flat)(free_ctrl_pts.reshape(-1)))


def _admm_solve_qp(
    h_mat: jnp.ndarray,
    grad_vec: jnp.ndarray,
    a_mat: jnp.ndarray,
    b_vec: jnp.ndarray,
    n_admm_steps: int,
) -> jnp.ndarray:
    dtype = h_mat.dtype
    at_mat = a_mat.T
    system = h_mat + ADMM_RHO * (at_mat @ a_mat)
    system_inv = _gauss_jordan_inverse(system + METRIC_EIGEN_FLOOR * jnp.eye(h_mat.shape[0], dtype=dtype))

    def solve_x(z_val, u_val):
        rhs = -grad_vec + ADMM_RHO * (at_mat @ (z_val - u_val))
        return system_inv @ rhs

    unconstrained = -(_gauss_jordan_inverse(h_mat) @ grad_vec)
    z0 = jnp.minimum(a_mat @ unconstrained, b_vec)
    u0 = jnp.zeros_like(b_vec)

    def admm_step(carry, _):
        z_val, u_val = carry
        x_val = solve_x(z_val, u_val)
        ax_plus_u = a_mat @ x_val + u_val
        z_new = jnp.minimum(ax_plus_u, b_vec)
        u_new = u_val + a_mat @ x_val - z_new
        return (z_new, u_new), None

    (z_final, u_final), _ = jax.lax.scan(admm_step, (z0, u0), xs=None, length=n_admm_steps)
    return solve_x(z_final, u_final)


def _solve_flat_nmpc_sqp_impl(
    free_ctrl_pts: jnp.ndarray,
    x0_state: jnp.ndarray,
    ref_free_ctrl_pts: jnp.ndarray,
    target_vel: jnp.ndarray,
    config_arr: jnp.ndarray,
    knots: jnp.ndarray,
    t_eval: jnp.ndarray,
    obs_xy: jnp.ndarray,
    obs_radius: jnp.ndarray,
    obs_mask: jnp.ndarray,
    route_xy: jnp.ndarray,
    route_normal: jnp.ndarray,
    route_half_width: jnp.ndarray,
    route_mask: jnp.ndarray,
    n_sqp_steps: int,
    n_admm_steps: int,
    degree: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    max_steering = config_arr[CFG_MAX_STEERING]
    eps = config_arr[CFG_EPS]
    a_min = config_arr[CFG_A_MIN]
    a_max = config_arr[CFG_A_MAX]

    init_flat = free_ctrl_pts.reshape(-1)
    free_shape = free_ctrl_pts.shape

    def objective_from_flat(flat_pts: jnp.ndarray) -> jnp.ndarray:
        return _objective_from_free(
            flat_pts.reshape(free_shape),
            x0_state,
            ref_free_ctrl_pts,
            config_arr,
            knots,
            t_eval,
            obs_xy,
            obs_radius,
            obs_mask,
            route_xy,
            route_normal,
            route_half_width,
            route_mask,
            degree,
            config_arr[CFG_CBF_SOLVE_WEIGHT_SCALE],
            config_arr[CFG_ROAD_SOLVE_WEIGHT_SCALE],
            target_vel,
        )

    def constraints_from_flat(flat_pts: jnp.ndarray) -> jnp.ndarray:
        reshaped = flat_pts.reshape(free_shape)
        return _compute_constraint_terms(
            reshaped,
            x0_state,
            config_arr,
            knots,
            t_eval,
            obs_xy,
            obs_radius,
            obs_mask,
            route_xy,
            route_normal,
            route_half_width,
            route_mask,
            degree,
        )[4]

    objective_grad_fn = jax.grad(objective_from_flat)
    objective_hessian_fn = jax.hessian(objective_from_flat)
    jacobian_fn = jax.jacfwd(constraints_from_flat)

    def feasibility_score(flat_pts: jnp.ndarray) -> jnp.ndarray:
        return jnp.max(jax.nn.relu(constraints_from_flat(flat_pts)))

    init_violation = feasibility_score(init_flat)
    init_objective = objective_from_flat(init_flat)

    def sqp_step(carry, _):
        flat_pts, best_flat, best_violation, best_objective = carry
        dtype = flat_pts.dtype
        grad_val = objective_grad_fn(flat_pts)
        hess_val = _stabilize_metric(objective_hessian_fn(flat_pts))
        g_val = constraints_from_flat(flat_pts)
        j_val = jacobian_fn(flat_pts)
        dx = _admm_solve_qp(hess_val, grad_val, j_val, -g_val, n_admm_steps)
        dx = jnp.clip(dx, -MAX_SQP_DX, MAX_SQP_DX)
        candidate = (flat_pts + dx).astype(dtype)
        candidate_violation = feasibility_score(candidate)
        candidate_objective = objective_from_flat(candidate)
        feasibility_improved = candidate_violation < best_violation - 1e-6
        both_feasible = (candidate_violation <= 1e-5) & (best_violation <= 1e-5)
        objective_improved = candidate_objective < best_objective
        take_candidate = feasibility_improved | (both_feasible & objective_improved)
        next_best = jnp.where(take_candidate, candidate, best_flat)
        next_violation = jnp.where(take_candidate, candidate_violation, best_violation)
        next_objective = jnp.where(take_candidate, candidate_objective, best_objective)
        return (
            candidate,
            next_best,
            next_violation,
            next_objective,
        ), None

    (_, opt_flat, _, _), _ = jax.lax.scan(
        sqp_step,
        (init_flat, init_flat, init_violation, init_objective),
        xs=None,
        length=n_sqp_steps,
    )
    opt_free_pts = opt_flat.reshape(free_shape)

    (
        full_pts,
        xy_eval,
        delta_eval,
        accel_eval,
        _constraints,
        max_kappa_violation,
        min_kappa_slack,
        min_obstacle_dist,
        min_cbf_value,
    ) = _compute_constraint_terms(
        opt_free_pts,
        x0_state,
        config_arr,
        knots,
        t_eval,
        obs_xy,
        obs_radius,
        obs_mask,
        route_xy,
        route_normal,
        route_half_width,
        route_mask,
        degree,
    )

    total, cost_tracking, cost_cbf, cost_road = _compute_objective_breakdown(
        opt_free_pts,
        x0_state,
        ref_free_ctrl_pts,
        config_arr,
        knots,
        t_eval,
        obs_xy,
        obs_radius,
        obs_mask,
        route_xy,
        route_normal,
        route_half_width,
        route_mask,
        degree,
        target_vel,
    )
    # ``_compute_constraint_terms`` intentionally keeps its established
    # nine-value diagnostic interface.  Rebuild the path derivative here for
    # the final oriented-body road-clearance diagnostic instead of referring
    # to the helper's local ``dxy_eval`` variable.
    dxy_eval = _build_eval_terms(
        opt_free_pts,
        x0_state,
        config_arr,
        knots,
        t_eval,
        degree,
    )[2]
    road_clearance = _compute_body_road_clearance(
        xy_eval,
        dxy_eval,
        route_xy,
        route_normal,
        route_half_width,
        route_mask,
        config_arr[CFG_EGO_HALF_WIDTH],
        config_arr[CFG_EGO_HALF_LENGTH],
    )
    min_road_clearance = jnp.where(jnp.any(route_mask), jnp.min(road_clearance), 0.0)
    breakdown = jnp.array(
        [
            total,
            cost_tracking,
            cost_cbf,
            cost_road,
            max_kappa_violation,
            min_kappa_slack,
            min_obstacle_dist,
            min_cbf_value,
            min_road_clearance,
        ],
        dtype=jnp.float32,
    )

    target_velocity_exec = target_vel
    steering = jnp.clip(delta_eval[0] / jnp.maximum(max_steering, eps), -1.0, 1.0)
    throttle = jnp.where(
        accel_eval[0] >= 0.0,
        accel_eval[0] / jnp.maximum(a_max, eps),
        accel_eval[0] / jnp.maximum(jnp.abs(a_min), eps),
    )
    throttle = jnp.clip(throttle, -1.0, 1.0)

    control_summary = jnp.array([steering, throttle, target_velocity_exec], dtype=jnp.float32)
    prediction = xy_eval.astype(jnp.float32)
    hj = _stabilize_metric(objective_hessian_fn(opt_flat)).astype(jnp.float32)
    return opt_free_pts, full_pts, breakdown, control_summary, prediction, hj


def _prepare_obstacles_np(
    obs_xy: np.ndarray | None,
    obs_radius: np.ndarray | None,
    obs_mask: np.ndarray | None,
    max_obstacles: int,
    include_motion: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    descriptor_dim = 4 if include_motion else 2
    xy_out = np.zeros((max_obstacles, descriptor_dim), dtype=np.float32)
    radius_out = np.zeros((max_obstacles,), dtype=np.float32)
    mask_out = np.zeros((max_obstacles,), dtype=bool)

    if obs_xy is None or obs_radius is None:
        return xy_out, radius_out, mask_out

    xy_arr = np.asarray(obs_xy, dtype=np.float32)
    if xy_arr.ndim != 2 or xy_arr.shape[1] < 2:
        xy_arr = xy_arr.reshape(-1, 2)
    radius_arr = np.asarray(obs_radius, dtype=np.float32).reshape(-1)
    if obs_mask is None:
        mask_arr = np.ones((min(xy_arr.shape[0], radius_arr.shape[0]),), dtype=bool)
    else:
        mask_arr = np.asarray(obs_mask, dtype=bool).reshape(-1)

    count = min(xy_arr.shape[0], radius_arr.shape[0], mask_arr.shape[0])
    if count <= 0:
        return xy_out, radius_out, mask_out

    valid_xy = xy_arr[:count][mask_arr[:count]]
    valid_radius = radius_arr[:count][mask_arr[:count]]
    if valid_xy.shape[0] <= 0:
        return xy_out, radius_out, mask_out

    order = np.argsort(np.linalg.norm(valid_xy[:, :2], axis=1))
    keep = order[:max_obstacles]
    n_keep = keep.shape[0]
    copy_dim = min(valid_xy.shape[1], descriptor_dim)
    xy_out[:n_keep, :copy_dim] = valid_xy[keep, :copy_dim]
    radius_out[:n_keep] = valid_radius[keep]
    mask_out[:n_keep] = True
    return xy_out, radius_out, mask_out


def _prepare_route_descriptor_np(
    route_xy: np.ndarray | None,
    route_normal: np.ndarray | None,
    route_half_width: np.ndarray | None,
    route_mask: np.ndarray | None,
    max_route_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xy_out = np.zeros((max_route_samples, 2), dtype=np.float32)
    normal_out = np.zeros((max_route_samples, 2), dtype=np.float32)
    half_width_out = np.zeros((max_route_samples,), dtype=np.float32)
    mask_out = np.zeros((max_route_samples,), dtype=bool)

    if route_xy is None or route_normal is None or route_half_width is None:
        return xy_out, normal_out, half_width_out, mask_out

    xy_arr = np.asarray(route_xy, dtype=np.float32).reshape(-1, 2)
    normal_arr = np.asarray(route_normal, dtype=np.float32).reshape(-1, 2)
    half_width_arr = np.asarray(route_half_width, dtype=np.float32).reshape(-1)
    if route_mask is None:
        mask_arr = np.ones((min(xy_arr.shape[0], normal_arr.shape[0], half_width_arr.shape[0]),), dtype=bool)
    else:
        mask_arr = np.asarray(route_mask, dtype=bool).reshape(-1)

    count = min(xy_arr.shape[0], normal_arr.shape[0], half_width_arr.shape[0], mask_arr.shape[0], max_route_samples)
    if count <= 0:
        return xy_out, normal_out, half_width_out, mask_out

    valid = mask_arr[:count]
    if not np.any(valid):
        return xy_out, normal_out, half_width_out, mask_out

    n_keep = int(np.sum(valid))
    xy_valid = xy_arr[:count][valid]
    normal_valid = normal_arr[:count][valid]
    half_width_valid = half_width_arr[:count][valid]
    normal_norm = np.linalg.norm(normal_valid, axis=1, keepdims=True)
    normal_valid = normal_valid / np.maximum(normal_norm, EPS)
    xy_out[:n_keep] = xy_valid[:n_keep]
    normal_out[:n_keep] = normal_valid[:n_keep]
    half_width_out[:n_keep] = half_width_valid[:n_keep]
    mask_out[:n_keep] = True
    return xy_out, normal_out, half_width_out, mask_out


class FlatSpaceNMPC:
    def __init__(self, config: FlatNMPCConfig = FlatNMPCConfig()):
        self.config = config
        try:
            # Preserve the established single-environment CPU default when
            # both backends are visible.
            self.device = jax.devices("cpu")[0]
        except RuntimeError:
            # Explicit JAX_PLATFORMS=cuda intentionally hides the CPU backend;
            # in that mode place the complete MPC solve on the active device.
            self.device = jax.devices()[0]
        _normalize_metric_risk_aggregation(self.config.metric_risk_aggregation)

        self._n_ctrl = config.n_ctrl_points
        self._n_free = max(config.n_ctrl_points - 2, 1)
        self._knots = jax.device_put(make_clamped_knots(self._n_ctrl, config.spline_degree), self.device)
        self._t_eval = jax.device_put(jnp.linspace(0.0, 1.0, config.n_eval_points), self.device)
        self._config_arr = jax.device_put(_pack_config(config), self.device)
        self._solve_jit = jax.jit(
            _solve_flat_nmpc_sqp_impl,
            device=self.device,
            static_argnums=(14, 15, 16),
        )
        self._metric_jit = jax.jit(
            _compute_objective_metric_impl,
            device=self.device,
            static_argnums=(13,),
        )
        self._hard_certificate_jit = jax.jit(
            _compute_hard_safety_certificate_impl,
            device=self.device,
            static_argnums=(12,),
        )

        self._default_prev_free = jax.device_put(jnp.zeros((self._n_free, 2), dtype=jnp.float32), self.device)
        self._default_obs_xy = jax.device_put(
            jnp.zeros((self.config.max_obstacles, 4), dtype=jnp.float32),
            self.device,
        )
        self._default_obs_radius = jax.device_put(
            jnp.zeros((self.config.max_obstacles,), dtype=jnp.float32),
            self.device,
        )
        self._default_obs_mask = jax.device_put(
            jnp.zeros((self.config.max_obstacles,), dtype=bool),
            self.device,
        )
        self._default_route_xy = jax.device_put(
            jnp.zeros((self.config.max_route_samples, 2), dtype=jnp.float32),
            self.device,
        )
        self._default_route_normal = jax.device_put(
            jnp.zeros((self.config.max_route_samples, 2), dtype=jnp.float32),
            self.device,
        )
        self._default_route_half_width = jax.device_put(
            jnp.zeros((self.config.max_route_samples,), dtype=jnp.float32),
            self.device,
        )
        self._default_route_mask = jax.device_put(
            jnp.zeros((self.config.max_route_samples,), dtype=bool),
            self.device,
        )
        self._warmed_up = False

    def warmup(self):
        if self._warmed_up:
            return

        free_pts = jnp.zeros((self._n_free, 2), dtype=jnp.float32)
        x0 = jnp.array([0.0, 0.0, 0.0, 10.0], dtype=jnp.float32)
        ref = jnp.zeros((self._n_free, 2), dtype=jnp.float32)
        target_vel = jnp.array(10.0, dtype=jnp.float32)
        _ = self._solve_jit(
            free_pts,
            x0,
            ref,
            target_vel,
            self._config_arr,
            self._knots,
            self._t_eval,
            self._default_obs_xy,
            self._default_obs_radius,
            self._default_obs_mask,
            self._default_route_xy,
            self._default_route_normal,
            self._default_route_half_width,
            self._default_route_mask,
            self.config.n_sqp_steps,
            self.config.n_admm_steps,
            self.config.spline_degree,
        )
        _ = self._solve_jit(
            free_pts,
            x0,
            ref,
            target_vel,
            self._config_arr,
            self._knots,
            self._t_eval,
            self._default_obs_xy,
            self._default_obs_radius,
            self._default_obs_mask,
            self._default_route_xy,
            self._default_route_normal,
            self._default_route_half_width,
            self._default_route_mask,
            0,
            self.config.n_admm_steps,
            self.config.spline_degree,
        )
        _ = self._metric_jit(
            free_pts,
            x0,
            free_pts,
            self._config_arr,
            self._knots,
            self._t_eval,
            self._default_obs_xy,
            self._default_obs_radius,
            self._default_obs_mask,
            self._default_route_xy,
            self._default_route_normal,
            self._default_route_half_width,
            self._default_route_mask,
            self.config.spline_degree,
        )
        self._warmed_up = True

    def _prepare_ref_free_ctrl_pts(self, ref_free_ctrl_pts: np.ndarray) -> jnp.ndarray:
        ref = jnp.asarray(ref_free_ctrl_pts, dtype=jnp.float32).reshape(-1, 2)
        if ref.shape[0] < self._n_free:
            ref = jnp.pad(ref, ((0, self._n_free - ref.shape[0]), (0, 0)))
        elif ref.shape[0] > self._n_free:
            ref = ref[: self._n_free]
        return ref

    def _prepare_prev_free_ctrl_pts(self, prev_free_ctrl_pts: np.ndarray | None) -> jnp.ndarray:
        if prev_free_ctrl_pts is None:
            return self._default_prev_free
        prev = jnp.asarray(prev_free_ctrl_pts, dtype=jnp.float32).reshape(-1, 2)
        if prev.shape[0] < self._n_free:
            prev = jnp.pad(prev, ((0, self._n_free - prev.shape[0]), (0, 0)))
        elif prev.shape[0] > self._n_free:
            prev = prev[: self._n_free]
        return prev

    def _prepare_obstacles(
        self,
        obs_xy: np.ndarray | None,
        obs_radius: np.ndarray | None,
        obs_mask: np.ndarray | None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if not self.config.cbf_enable:
            return self._default_obs_xy, self._default_obs_radius, self._default_obs_mask
        xy_np, radius_np, mask_np = _prepare_obstacles_np(
            obs_xy,
            obs_radius,
            obs_mask,
            self.config.max_obstacles,
            include_motion=True,
        )
        return (
            jax.device_put(jnp.asarray(xy_np, dtype=jnp.float32), self.device),
            jax.device_put(jnp.asarray(radius_np, dtype=jnp.float32), self.device),
            jax.device_put(jnp.asarray(mask_np, dtype=bool), self.device),
        )

    def _prepare_route_descriptor(
        self,
        route_xy: np.ndarray | None,
        route_normal: np.ndarray | None,
        route_half_width: np.ndarray | None,
        route_mask: np.ndarray | None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        xy_np, normal_np, half_width_np, mask_np = _prepare_route_descriptor_np(
            route_xy,
            route_normal,
            route_half_width,
            route_mask,
            self.config.max_route_samples,
        )
        return (
            jax.device_put(jnp.asarray(xy_np, dtype=jnp.float32), self.device),
            jax.device_put(jnp.asarray(normal_np, dtype=jnp.float32), self.device),
            jax.device_put(jnp.asarray(half_width_np, dtype=jnp.float32), self.device),
            jax.device_put(jnp.asarray(mask_np, dtype=bool), self.device),
        )

    def get_current_hj(
        self,
        x0: np.ndarray,
        prev_free_ctrl_pts: np.ndarray | None = None,
        obs_xy: np.ndarray | None = None,
        obs_radius: np.ndarray | None = None,
        obs_mask: np.ndarray | None = None,
        route_xy: np.ndarray | None = None,
        route_normal: np.ndarray | None = None,
        route_half_width: np.ndarray | None = None,
        route_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        x0_arr = jnp.asarray(x0, dtype=jnp.float32).flatten()
        if x0_arr.shape[0] < 4:
            x0_arr = jnp.pad(x0_arr, (0, 4 - x0_arr.shape[0]))
        else:
            x0_arr = x0_arr[:4]
        eval_free = self._prepare_prev_free_ctrl_pts(prev_free_ctrl_pts)
        obs_xy_arr, obs_radius_arr, obs_mask_arr = self._prepare_obstacles(obs_xy, obs_radius, obs_mask)
        route_xy_arr, route_normal_arr, route_half_width_arr, route_mask_arr = self._prepare_route_descriptor(
            route_xy,
            route_normal,
            route_half_width,
            route_mask,
        )
        hj = self._metric_jit(
            eval_free,
            x0_arr,
            eval_free,
            self._config_arr,
            self._knots,
            self._t_eval,
            obs_xy_arr,
            obs_radius_arr,
            obs_mask_arr,
            route_xy_arr,
            route_normal_arr,
            route_half_width_arr,
            route_mask_arr,
            self.config.spline_degree,
        )
        return np.asarray(hj, dtype=np.float32)

    def step(
        self,
        x0: np.ndarray,
        ref_free_ctrl_pts: np.ndarray,
        prev_free_ctrl_pts: np.ndarray | None = None,
        target_velocity: float | None = None,
        obs_xy: np.ndarray | None = None,
        obs_radius: np.ndarray | None = None,
        obs_mask: np.ndarray | None = None,
        route_xy: np.ndarray | None = None,
        route_normal: np.ndarray | None = None,
        route_half_width: np.ndarray | None = None,
        route_mask: np.ndarray | None = None,
        obstacle_cbf_enabled: bool = True,
    ) -> FlatNMPCResult:
        start_time = time.perf_counter()

        x0_arr = jnp.asarray(x0, dtype=jnp.float32).flatten()
        if x0_arr.shape[0] < 4:
            x0_arr = jnp.pad(x0_arr, (0, 4 - x0_arr.shape[0]))
        else:
            x0_arr = x0_arr[:4]

        ref_free = self._prepare_ref_free_ctrl_pts(ref_free_ctrl_pts)
        prev_free = self._prepare_prev_free_ctrl_pts(prev_free_ctrl_pts)
        initial_free = 0.5 * ref_free + 0.5 * prev_free
        target_vel = jnp.array(
            self.config.target_velocity if target_velocity is None else target_velocity,
            dtype=jnp.float32,
        )
        obs_xy_arr, obs_radius_arr, obs_mask_arr = self._prepare_obstacles(obs_xy, obs_radius, obs_mask)
        if not obstacle_cbf_enabled:
            obs_mask_arr = self._default_obs_mask
        route_xy_arr, route_normal_arr, route_half_width_arr, route_mask_arr = self._prepare_route_descriptor(
            route_xy,
            route_normal,
            route_half_width,
            route_mask,
        )

        opt_free, full_pts_opt, breakdown, control_summary, prediction, hj = self._solve_jit(
            initial_free,
            x0_arr,
            ref_free,
            target_vel,
            self._config_arr,
            self._knots,
            self._t_eval,
            obs_xy_arr,
            obs_radius_arr,
            obs_mask_arr,
            route_xy_arr,
            route_normal_arr,
            route_half_width_arr,
            route_mask_arr,
            self.config.n_sqp_steps,
            self.config.n_admm_steps,
            self.config.spline_degree,
        )
        opt_free.block_until_ready()

        opt_free_np = np.asarray(opt_free, dtype=np.float32)
        full_pts_np = np.asarray(full_pts_opt, dtype=np.float32)
        prediction_np = np.asarray(prediction, dtype=np.float32)
        breakdown_np = np.asarray(breakdown, dtype=np.float32)
        control_np = np.asarray(control_summary, dtype=np.float32)
        hj_np = np.asarray(hj, dtype=np.float32)

        solver_time = time.perf_counter() - start_time

        return FlatNMPCResult(
            steering=float(control_np[0]),
            throttle=float(control_np[1]),
            target_velocity_exec=float(control_np[2]),
            cost_total=float(breakdown_np[0]),
            cost_tracking=float(breakdown_np[1]),
            cost_cbf=float(breakdown_np[2]),
            cost_road=float(breakdown_np[3]),
            max_kappa_violation=float(breakdown_np[4]),
            min_kappa_slack=float(breakdown_np[5]),
            min_obstacle_dist=float(breakdown_np[6]),
            min_cbf_value=float(breakdown_np[7]),
            min_road_clearance=float(breakdown_np[8]),
            opt_free_ctrl_pts=opt_free_np,
            full_ctrl_pts=full_pts_np,
            predicted_x=prediction_np[:, 0],
            predicted_y=prediction_np[:, 1],
            hj=hj_np,
            solver_time=solver_time,
        )

    def shadow_safety_certificate(
        self,
        free_ctrl_pts: np.ndarray,
        x0: np.ndarray,
        *,
        obs_xy: np.ndarray | None = None,
        obs_radius: np.ndarray | None = None,
        obs_mask: np.ndarray | None = None,
        route_xy: np.ndarray | None = None,
        route_normal: np.ndarray | None = None,
        route_half_width: np.ndarray | None = None,
        route_mask: np.ndarray | None = None,
    ) -> FlatNMPCSafetyCertificate:
        """Verify a fixed trajectory against the real hard constraints.

        Unlike ``_prepare_obstacles``, obstacle preparation here deliberately
        ignores ``config.cbf_enable``.  The method remains a valid shadow
        verifier even when the trajectory came from a controller configured
        without obstacle-CBF authority.
        """

        x0_arr = jnp.asarray(x0, dtype=jnp.float32).flatten()
        if x0_arr.shape[0] < 4:
            x0_arr = jnp.pad(x0_arr, (0, 4 - x0_arr.shape[0]))
        else:
            x0_arr = x0_arr[:4]
        free_arr = self._prepare_ref_free_ctrl_pts(free_ctrl_pts)
        obs_xy_np, obs_radius_np, obs_mask_np = _prepare_obstacles_np(
            obs_xy,
            obs_radius,
            obs_mask,
            self.config.max_obstacles,
            include_motion=True,
        )
        obs_xy_arr = jax.device_put(
            jnp.asarray(obs_xy_np, dtype=jnp.float32), self.device
        )
        obs_radius_arr = jax.device_put(
            jnp.asarray(obs_radius_np, dtype=jnp.float32), self.device
        )
        obs_mask_arr = jax.device_put(
            jnp.asarray(obs_mask_np, dtype=bool), self.device
        )
        (
            route_xy_arr,
            route_normal_arr,
            route_half_width_arr,
            route_mask_arr,
        ) = self._prepare_route_descriptor(
            route_xy,
            route_normal,
            route_half_width,
            route_mask,
        )
        result = self._hard_certificate_jit(
            free_arr,
            x0_arr,
            self._config_arr,
            self._knots,
            self._t_eval,
            obs_xy_arr,
            obs_radius_arr,
            obs_mask_arr,
            route_xy_arr,
            route_normal_arr,
            route_half_width_arr,
            route_mask_arr,
            self.config.spline_degree,
        )
        result[0].block_until_ready()
        values = [np.asarray(value) for value in result]
        has_obstacle = bool(np.any(obs_mask_np))
        has_route = bool(np.any(np.asarray(route_mask_arr)))
        scalar_values = [float(np.asarray(value)) for value in values[:4]]
        if not has_obstacle:
            scalar_values[0] = float("nan")
            scalar_values[1] = float("nan")
        if not has_route:
            scalar_values[2] = float("nan")
        finite_scalars = [
            value for value in scalar_values if not np.isnan(value)
        ]
        finite = bool(
            all(np.isfinite(value) for value in finite_scalars)
            and all(np.all(np.isfinite(value)) for value in values[4:])
        )
        return FlatNMPCSafetyCertificate(
            min_hard_cbf_margin=scalar_values[0],
            min_obstacle_clearance=scalar_values[1],
            min_road_clearance=scalar_values[2],
            min_kinematic_slack=scalar_values[3],
            hard_cbf_margin=values[4].astype(np.float32),
            obstacle_clearance=values[5].astype(np.float32),
            road_clearance=values[6].astype(np.float32),
            kinematic_slack=values[7].astype(np.float32),
            has_obstacle=has_obstacle,
            has_route=has_route,
            finite=finite,
        )

    def step_fast(
        self,
        x0: np.ndarray,
        ref_free_ctrl_pts: np.ndarray,
        prev_free_ctrl_pts: np.ndarray | None = None,
        target_velocity: float | None = None,
        obs_xy: np.ndarray | None = None,
        obs_radius: np.ndarray | None = None,
        obs_mask: np.ndarray | None = None,
        route_xy: np.ndarray | None = None,
        route_normal: np.ndarray | None = None,
        route_half_width: np.ndarray | None = None,
        route_mask: np.ndarray | None = None,
    ) -> Tuple[float, float, np.ndarray, float]:
        result = self.step(
            x0,
            ref_free_ctrl_pts,
            prev_free_ctrl_pts=prev_free_ctrl_pts,
            target_velocity=target_velocity,
            obs_xy=obs_xy,
            obs_radius=obs_radius,
            obs_mask=obs_mask,
            route_xy=route_xy,
            route_normal=route_normal,
            route_half_width=route_half_width,
            route_mask=route_mask,
        )
        return result.steering, result.throttle, result.opt_free_ctrl_pts, result.target_velocity_exec

    def reset(self):
        pass


@dataclass
class BatchedFlatNMPCConfig:
    batch_size: int = 8
    use_gpu: bool = False
    update_period: int = 1
    print_interval: int = 5000

    n_ctrl_points: int = 5
    spline_degree: int = 3
    horizon_time: float = 0.5
    n_eval_points: int = 5

    wheelbase: float = 2.5
    max_steering: float = 0.6
    v_min: float = 1.0
    target_velocity: float = 15.0
    a_min: float = -5.0
    a_max: float = 3.0

    w_tracking: float = 10.0
    w_velocity: float = 0.0
    w_risk: float = 0.0
    w_cbf: float = 0.0
    cbf_solve_weight_scale: float = 0.0
    w_road: float = 0.0
    road_solve_weight_scale: float = 0.0
    risk_buffer: float = 1.0
    risk_temperature: float = 0.3
    road_buffer: float = 0.5
    road_temperature: float = 0.5
    metric_risk_aggregation: MetricRiskAggregation = METRIC_RISK_SMOOTHMAX

    d_clearance: float = 2.0
    cbf_gamma: float = 1.0
    max_obstacles: int = 4
    max_route_samples: int = 16
    cbf_enable: bool = False
    n_sqp_steps: int = 5
    n_admm_steps: int = 50

    def to_flat_config(self) -> FlatNMPCConfig:
        return FlatNMPCConfig(
            n_ctrl_points=self.n_ctrl_points,
            spline_degree=self.spline_degree,
            horizon_time=self.horizon_time,
            n_eval_points=self.n_eval_points,
            wheelbase=self.wheelbase,
            max_steering=self.max_steering,
            v_min=self.v_min,
            kappa_max=np.tan(self.max_steering) / self.wheelbase,
            a_min=self.a_min,
            a_max=self.a_max,
            w_tracking=self.w_tracking,
            w_velocity=self.w_velocity,
            w_risk=self.w_risk,
            w_cbf=self.w_cbf,
            cbf_solve_weight_scale=self.cbf_solve_weight_scale,
            w_road=self.w_road,
            road_solve_weight_scale=self.road_solve_weight_scale,
            risk_buffer=self.risk_buffer,
            risk_temperature=self.risk_temperature,
            road_buffer=self.road_buffer,
            road_temperature=self.road_temperature,
            metric_risk_aggregation=self.metric_risk_aggregation,
            d_clearance=self.d_clearance,
            cbf_gamma=self.cbf_gamma,
            max_obstacles=self.max_obstacles,
            max_route_samples=self.max_route_samples,
            cbf_enable=self.cbf_enable,
            n_sqp_steps=self.n_sqp_steps,
            n_admm_steps=self.n_admm_steps,
            target_velocity=self.target_velocity,
        )


class BatchedFlatNMPC:
    def __init__(self, config: BatchedFlatNMPCConfig | None = None):
        self.config = config or BatchedFlatNMPCConfig()
        self._flat_config = self.config.to_flat_config()
        _normalize_metric_risk_aggregation(self._flat_config.metric_risk_aggregation)

        if self.config.use_gpu:
            try:
                self.device = jax.devices("gpu")[0]
                print(f"[BatchedFlatNMPC] Using GPU device: {self.device}")
            except RuntimeError:
                try:
                    print("[BatchedFlatNMPC] GPU not available, falling back to CPU")
                    self.device = jax.devices("cpu")[0]
                except RuntimeError:
                    self.device = jax.devices()[0]
        else:
            try:
                self.device = jax.devices("cpu")[0]
            except RuntimeError:
                # An explicit JAX_PLATFORMS=cuda hides the CPU backend.  Keep
                # construction consistent with FlatSpaceNMPC in that mode.
                self.device = jax.devices()[0]

        self._n_ctrl = self._flat_config.n_ctrl_points
        self._n_free = max(self._n_ctrl - 2, 1)
        self._knots = jax.device_put(make_clamped_knots(self._n_ctrl, self._flat_config.spline_degree), self.device)
        self._t_eval = jax.device_put(jnp.linspace(0.0, 1.0, self._flat_config.n_eval_points), self.device)
        self._config_arr = jax.device_put(_pack_config(self._flat_config), self.device)
        self._batched_solve = jax.vmap(
            _solve_flat_nmpc_sqp_impl,
            in_axes=(0, 0, 0, 0, None, None, None, 0, 0, 0, 0, 0, 0, 0, None, None, None),
        )
        self._batched_solve_jit = jax.jit(self._batched_solve, device=self.device, static_argnums=(14, 15, 16))
        self._batched_metric = jax.vmap(
            _compute_objective_metric_impl,
            in_axes=(0, 0, 0, None, None, None, 0, 0, 0, 0, 0, 0, 0, None),
        )
        self._batched_metric_jit = jax.jit(self._batched_metric, device=self.device, static_argnums=(13,))

        self._default_prev_free_batch = jax.device_put(
            jnp.zeros((self.config.batch_size, self._n_free, 2), dtype=jnp.float32),
            self.device,
        )
        self._default_obs_xy_batch = jax.device_put(
            jnp.zeros((self.config.batch_size, self._flat_config.max_obstacles, 2), dtype=jnp.float32),
            self.device,
        )
        self._default_obs_radius_batch = jax.device_put(
            jnp.zeros((self.config.batch_size, self._flat_config.max_obstacles), dtype=jnp.float32),
            self.device,
        )
        self._default_obs_mask_batch = jax.device_put(
            jnp.zeros((self.config.batch_size, self._flat_config.max_obstacles), dtype=bool),
            self.device,
        )
        self._default_route_xy_batch = jax.device_put(
            jnp.zeros((self.config.batch_size, self._flat_config.max_route_samples, 2), dtype=jnp.float32),
            self.device,
        )
        self._default_route_normal_batch = jax.device_put(
            jnp.zeros((self.config.batch_size, self._flat_config.max_route_samples, 2), dtype=jnp.float32),
            self.device,
        )
        self._default_route_half_width_batch = jax.device_put(
            jnp.zeros((self.config.batch_size, self._flat_config.max_route_samples), dtype=jnp.float32),
            self.device,
        )
        self._default_route_mask_batch = jax.device_put(
            jnp.zeros((self.config.batch_size, self._flat_config.max_route_samples), dtype=bool),
            self.device,
        )

        self._cached_controls = None
        self._cached_opt_free = None
        self._cached_full_pts = None
        self._cached_target_vel_exec = None
        self._cached_diag = None

        self._step_count = 0
        self._solve_count = 0
        self._total_solve_time = 0.0
        self._warmed_up = False

    def _prepare_prev_batch(self, batch_size: int, prev_free_ctrl_pts_batch: np.ndarray | None) -> jnp.ndarray:
        if prev_free_ctrl_pts_batch is None:
            return self._default_prev_free_batch[:batch_size]

        prev = np.asarray(prev_free_ctrl_pts_batch, dtype=np.float32)
        expected_shape = (batch_size, self._n_free, 2)
        if prev.shape != expected_shape:
            fixed = np.zeros(expected_shape, dtype=np.float32)
            rows = min(batch_size, prev.shape[0] if prev.ndim >= 1 else 0)
            if prev.ndim == 3:
                cols = min(self._n_free, prev.shape[1])
                fixed[:rows, :cols] = prev[:rows, :cols, :2]
            prev = fixed
        return jnp.asarray(prev)

    def _prepare_ref_batch(self, batch_size: int, ref_free_ctrl_pts_batch: np.ndarray) -> jnp.ndarray:
        ref = np.asarray(ref_free_ctrl_pts_batch, dtype=np.float32)
        expected_shape = (batch_size, self._n_free, 2)
        if ref.shape != expected_shape:
            fixed = np.zeros(expected_shape, dtype=np.float32)
            rows = min(batch_size, ref.shape[0] if ref.ndim >= 1 else 0)
            if ref.ndim == 3:
                cols = min(self._n_free, ref.shape[1])
                fixed[:rows, :cols] = ref[:rows, :cols, :2]
            ref = fixed
        return jnp.asarray(ref)

    def _prepare_obstacle_batch(
        self,
        batch_size: int,
        obs_xy_batch: np.ndarray | None,
        obs_radius_batch: np.ndarray | None,
        obs_mask_batch: np.ndarray | None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if not self._flat_config.cbf_enable:
            return (
                self._default_obs_xy_batch[:batch_size],
                self._default_obs_radius_batch[:batch_size],
                self._default_obs_mask_batch[:batch_size],
            )

        xy_out = np.zeros((batch_size, self._flat_config.max_obstacles, 2), dtype=np.float32)
        radius_out = np.zeros((batch_size, self._flat_config.max_obstacles), dtype=np.float32)
        mask_out = np.zeros((batch_size, self._flat_config.max_obstacles), dtype=bool)
        if obs_xy_batch is not None and obs_radius_batch is not None:
            xy_arr = np.asarray(obs_xy_batch, dtype=np.float32)
            radius_arr = np.asarray(obs_radius_batch, dtype=np.float32)
            mask_arr = None if obs_mask_batch is None else np.asarray(obs_mask_batch, dtype=bool)
            for idx in range(batch_size):
                xy_i = xy_arr[idx] if idx < xy_arr.shape[0] else None
                radius_i = radius_arr[idx] if idx < radius_arr.shape[0] else None
                mask_i = None if mask_arr is None or idx >= mask_arr.shape[0] else mask_arr[idx]
                xy_fix, radius_fix, mask_fix = _prepare_obstacles_np(
                    xy_i,
                    radius_i,
                    mask_i,
                    self._flat_config.max_obstacles,
                )
                xy_out[idx] = xy_fix
                radius_out[idx] = radius_fix
                mask_out[idx] = mask_fix

        return (
            jax.device_put(jnp.asarray(xy_out, dtype=jnp.float32), self.device),
            jax.device_put(jnp.asarray(radius_out, dtype=jnp.float32), self.device),
            jax.device_put(jnp.asarray(mask_out, dtype=bool), self.device),
        )

    def _prepare_route_batch(
        self,
        batch_size: int,
        route_xy_batch: np.ndarray | None,
        route_normal_batch: np.ndarray | None,
        route_half_width_batch: np.ndarray | None,
        route_mask_batch: np.ndarray | None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        xy_out = np.zeros((batch_size, self._flat_config.max_route_samples, 2), dtype=np.float32)
        normal_out = np.zeros((batch_size, self._flat_config.max_route_samples, 2), dtype=np.float32)
        half_width_out = np.zeros((batch_size, self._flat_config.max_route_samples), dtype=np.float32)
        mask_out = np.zeros((batch_size, self._flat_config.max_route_samples), dtype=bool)

        if route_xy_batch is not None and route_normal_batch is not None and route_half_width_batch is not None:
            xy_arr = np.asarray(route_xy_batch, dtype=np.float32)
            normal_arr = np.asarray(route_normal_batch, dtype=np.float32)
            half_width_arr = np.asarray(route_half_width_batch, dtype=np.float32)
            mask_arr = None if route_mask_batch is None else np.asarray(route_mask_batch, dtype=bool)
            for idx in range(batch_size):
                xy_i = xy_arr[idx] if idx < xy_arr.shape[0] else None
                normal_i = normal_arr[idx] if idx < normal_arr.shape[0] else None
                half_width_i = half_width_arr[idx] if idx < half_width_arr.shape[0] else None
                mask_i = None if mask_arr is None or idx >= mask_arr.shape[0] else mask_arr[idx]
                xy_fix, normal_fix, half_width_fix, mask_fix = _prepare_route_descriptor_np(
                    xy_i,
                    normal_i,
                    half_width_i,
                    mask_i,
                    self._flat_config.max_route_samples,
                )
                xy_out[idx] = xy_fix
                normal_out[idx] = normal_fix
                half_width_out[idx] = half_width_fix
                mask_out[idx] = mask_fix

        return (
            jax.device_put(jnp.asarray(xy_out, dtype=jnp.float32), self.device),
            jax.device_put(jnp.asarray(normal_out, dtype=jnp.float32), self.device),
            jax.device_put(jnp.asarray(half_width_out, dtype=jnp.float32), self.device),
            jax.device_put(jnp.asarray(mask_out, dtype=bool), self.device),
        )

    def get_current_hj_batch(
        self,
        x0_batch: np.ndarray,
        prev_free_ctrl_pts_batch: np.ndarray | None = None,
        obs_xy_batch: np.ndarray | None = None,
        obs_radius_batch: np.ndarray | None = None,
        obs_mask_batch: np.ndarray | None = None,
        route_xy_batch: np.ndarray | None = None,
        route_normal_batch: np.ndarray | None = None,
        route_half_width_batch: np.ndarray | None = None,
        route_mask_batch: np.ndarray | None = None,
    ) -> np.ndarray:
        batch_size = x0_batch.shape[0]
        x0_batch = jnp.asarray(x0_batch, dtype=jnp.float32)
        prev_free_batch = self._prepare_prev_batch(batch_size, prev_free_ctrl_pts_batch)
        obs_xy_dev, obs_radius_dev, obs_mask_dev = self._prepare_obstacle_batch(
            batch_size,
            obs_xy_batch,
            obs_radius_batch,
            obs_mask_batch,
        )
        route_xy_dev, route_normal_dev, route_half_width_dev, route_mask_dev = self._prepare_route_batch(
            batch_size,
            route_xy_batch,
            route_normal_batch,
            route_half_width_batch,
            route_mask_batch,
        )
        hj = self._batched_metric_jit(
            prev_free_batch,
            jax.device_put(x0_batch, self.device),
            prev_free_batch,
            self._config_arr,
            self._knots,
            self._t_eval,
            obs_xy_dev,
            obs_radius_dev,
            obs_mask_dev,
            route_xy_dev,
            route_normal_dev,
            route_half_width_dev,
            route_mask_dev,
            self._flat_config.spline_degree,
        )
        return np.asarray(hj, dtype=np.float32)

    def step_batch(
        self,
        x0_batch: np.ndarray,
        ref_free_ctrl_pts_batch: np.ndarray,
        prev_free_ctrl_pts_batch: np.ndarray | None = None,
        obs_xy_batch: np.ndarray | None = None,
        obs_radius_batch: np.ndarray | None = None,
        obs_mask_batch: np.ndarray | None = None,
        route_xy_batch: np.ndarray | None = None,
        route_normal_batch: np.ndarray | None = None,
        route_half_width_batch: np.ndarray | None = None,
        route_mask_batch: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        batch_size = x0_batch.shape[0]

        self._step_count += 1
        should_solve = (
            self._cached_controls is None
            or self.config.update_period == 1
            or (self._step_count % self.config.update_period) == 0
        )
        if not should_solve and self._cached_controls is not None:
            return (
                self._cached_controls.copy(),
                self._cached_opt_free.copy(),
                None if self._cached_full_pts is None else self._cached_full_pts.copy(),
                self._cached_target_vel_exec.copy(),
                {k: v.copy() for k, v in self._cached_diag.items()},
            )

        self._solve_count += 1
        start_time = time.perf_counter()

        x0_batch = jnp.asarray(x0_batch, dtype=jnp.float32)
        ref_batch = self._prepare_ref_batch(batch_size, ref_free_ctrl_pts_batch)
        prev_free_batch = self._prepare_prev_batch(batch_size, prev_free_ctrl_pts_batch)
        obs_xy_dev, obs_radius_dev, obs_mask_dev = self._prepare_obstacle_batch(
            batch_size,
            obs_xy_batch,
            obs_radius_batch,
            obs_mask_batch,
        )
        route_xy_dev, route_normal_dev, route_half_width_dev, route_mask_dev = self._prepare_route_batch(
            batch_size,
            route_xy_batch,
            route_normal_batch,
            route_half_width_batch,
            route_mask_batch,
        )

        target_vel_batch = jnp.full((batch_size,), self._flat_config.target_velocity, dtype=jnp.float32)
        initial_free_batch = 0.5 * ref_batch + 0.5 * prev_free_batch

        opt_free_batch, full_pts_batch, breakdown_batch, control_summary_batch, _prediction_batch, hj_batch = (
            self._batched_solve_jit(
                jax.device_put(initial_free_batch, self.device),
                jax.device_put(x0_batch, self.device),
                jax.device_put(ref_batch, self.device),
                jax.device_put(target_vel_batch, self.device),
                self._config_arr,
                self._knots,
                self._t_eval,
                obs_xy_dev,
                obs_radius_dev,
                obs_mask_dev,
                route_xy_dev,
                route_normal_dev,
                route_half_width_dev,
                route_mask_dev,
                self._flat_config.n_sqp_steps,
                self._flat_config.n_admm_steps,
                self._flat_config.spline_degree,
            )
        )
        control_summary_batch.block_until_ready()
        self._total_solve_time += time.perf_counter() - start_time

        opt_free_np = np.asarray(opt_free_batch, dtype=np.float32)
        full_pts_np = np.asarray(full_pts_batch, dtype=np.float32)
        breakdown_np = np.asarray(breakdown_batch, dtype=np.float32)
        controls_np = np.asarray(control_summary_batch[:, :2], dtype=np.float32)
        target_velocity_exec_np = np.asarray(control_summary_batch[:, 2], dtype=np.float32)
        hj_np = np.asarray(hj_batch, dtype=np.float32)

        diagnostics = {
            "cost_total": breakdown_np[:, 0].astype(np.float32),
            "cost_tracking": breakdown_np[:, 1].astype(np.float32),
            "cost_cbf": breakdown_np[:, 2].astype(np.float32),
            "cost_road": breakdown_np[:, 3].astype(np.float32),
            "max_kappa_violation": breakdown_np[:, 4].astype(np.float32),
            "min_kappa_slack": breakdown_np[:, 5].astype(np.float32),
            "min_obstacle_dist": breakdown_np[:, 6].astype(np.float32),
            "min_cbf_value": breakdown_np[:, 7].astype(np.float32),
            "min_road_clearance": breakdown_np[:, 8].astype(np.float32),
            "hj": hj_np.astype(np.float32),
        }

        self._cached_controls = controls_np
        self._cached_opt_free = opt_free_np
        self._cached_full_pts = full_pts_np
        self._cached_target_vel_exec = target_velocity_exec_np
        self._cached_diag = {k: v.astype(np.float32) for k, v in diagnostics.items()}

        return (
            self._cached_controls.copy(),
            self._cached_opt_free.copy(),
            self._cached_full_pts.copy(),
            self._cached_target_vel_exec.copy(),
            {k: v.copy() for k, v in self._cached_diag.items()},
        )

    def warmup(self):
        if self._warmed_up:
            return

        batch_size = self.config.batch_size
        x0 = jnp.zeros((batch_size, 4), dtype=jnp.float32)
        x0 = x0.at[:, 3].set(10.0)
        ref = jnp.zeros((batch_size, self._n_free, 2), dtype=jnp.float32)
        free_pts = jnp.zeros((batch_size, self._n_free, 2), dtype=jnp.float32)
        target_vel = jnp.full((batch_size,), 10.0, dtype=jnp.float32)
        obs_xy = self._default_obs_xy_batch[:batch_size]
        obs_radius = self._default_obs_radius_batch[:batch_size]
        obs_mask = self._default_obs_mask_batch[:batch_size]
        route_xy = self._default_route_xy_batch[:batch_size]
        route_normal = self._default_route_normal_batch[:batch_size]
        route_half_width = self._default_route_half_width_batch[:batch_size]
        route_mask = self._default_route_mask_batch[:batch_size]

        _ = self._batched_solve_jit(
            free_pts,
            x0,
            ref,
            target_vel,
            self._config_arr,
            self._knots,
            self._t_eval,
            obs_xy,
            obs_radius,
            obs_mask,
            route_xy,
            route_normal,
            route_half_width,
            route_mask,
            self._flat_config.n_sqp_steps,
            self._flat_config.n_admm_steps,
            self._flat_config.spline_degree,
        )
        _ = self._batched_metric_jit(
            free_pts,
            x0,
            free_pts,
            self._config_arr,
            self._knots,
            self._t_eval,
            obs_xy,
            obs_radius,
            obs_mask,
            route_xy,
            route_normal,
            route_half_width,
            route_mask,
            self._flat_config.spline_degree,
        )
        self._warmed_up = True
        print(
            f"[BatchedFlatNMPC] Warmed up (batch={batch_size}, "
            f"n_ctrl={self._n_ctrl}, n_free={self._n_free}, device={self.device})"
        )

    def reset_done(self, done_mask: np.ndarray):
        done_mask = np.asarray(done_mask, dtype=bool)
        if not np.any(done_mask):
            return
        self._cached_controls = None
        self._cached_opt_free = None
        self._cached_full_pts = None
        self._cached_target_vel_exec = None
        self._cached_diag = None

    def reset(self):
        self._cached_controls = None
        self._cached_opt_free = None
        self._cached_full_pts = None
        self._cached_target_vel_exec = None
        self._cached_diag = None
        self._step_count = 0

    def get_stats(self):
        if self._solve_count == 0:
            return {"avg_solve_time_ms": 0.0, "solve_count": 0}
        return {
            "avg_solve_time_ms": self._total_solve_time / self._solve_count * 1000.0,
            "solve_count": self._solve_count,
            "step_count": self._step_count,
        }


def create_flat_nmpc(
    n_ctrl_points: int = 5,
    horizon_time: float = 0.5,
    wheelbase: float = 2.5,
    max_steering: float = 0.6,
    **kwargs,
) -> FlatSpaceNMPC:
    kappa_max = np.tan(max_steering) / wheelbase
    allowed = FlatNMPCConfig.__dataclass_fields__
    config = FlatNMPCConfig(
        n_ctrl_points=n_ctrl_points,
        horizon_time=horizon_time,
        wheelbase=wheelbase,
        max_steering=max_steering,
        kappa_max=kappa_max,
        **{k: v for k, v in kwargs.items() if k in allowed},
    )
    return FlatSpaceNMPC(config)
