"""Target-point geometry and MPC feedback for the real drive MGF-SAC path.

The module is intentionally independent of MetaDrive.  It maps the normalized
two-dimensional RL action to a point-tracking B-spline reference, then pulls
the optimized Flat NMPC geometry back to the target space.  The sensitivity
uses the objective Hessian returned by the same MPC solve; no extra MPC solve
is needed for the affine field at the executed anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DriveMGFGeometryConfig:
    margin_softmin_kappa: float = 5000.0
    # The latest Flat NMPC hard-constrains the complete oriented ego body.
    # This value retains 0.174 m beyond the 0.926 m straight-line half-width
    # for model/actuation mismatch; on turns the physical half-length support
    # replaces (rather than stacks on top of) the aligned half-width.
    road_tightening: float = 1.10
    ego_half_width: float = 0.926
    ego_half_length: float = 2.2575
    chance_beta: float = 0.0
    chance_kappa: float = 1.6448536269514722  # one-sided 95% Gaussian quantile
    conformal_delta: float = 0.0
    metric_damping: float = 1e-4
    margin_grad_step: float = 1e-3

    def __post_init__(self) -> None:
        if self.margin_softmin_kappa <= 0.0:
            raise ValueError("margin_softmin_kappa must be positive")
        if self.chance_beta < 0.0:
            raise ValueError("chance_beta must be nonnegative")
        if self.metric_damping <= 0.0:
            raise ValueError("metric_damping must be positive")
        if self.ego_half_width <= 0.0 or self.ego_half_length <= 0.0:
            raise ValueError("ego footprint dimensions must be positive")
        if self.margin_grad_step <= 0.0:
            raise ValueError("margin_grad_step must be positive")


@dataclass(frozen=True)
class DriveMGFField:
    anchor: np.ndarray
    margin: float
    margin_delta: float
    grad_margin: np.ndarray
    value: float
    grad_value: np.ndarray
    state_metric: np.ndarray
    state_metric_logdet: float
    sensitivity_condition: float
    sensitivity_valid: bool
    min_constraint: float
    min_obstacle_constraint: float
    min_road_constraint: float
    min_kinematic_constraint: float
    softmin_gap: float

    def as_info(self) -> dict[str, Any]:
        return {
            "anchor": np.asarray(self.anchor, dtype=np.float32).tolist(),
            "m": float(self.margin),
            "m_delta": float(self.margin_delta),
            "grad_m": np.asarray(self.grad_margin, dtype=np.float32).tolist(),
            "V": float(self.value),
            "grad_V": np.asarray(self.grad_value, dtype=np.float32).tolist(),
            "state_metric": np.asarray(self.state_metric, dtype=np.float32).tolist(),
            "state_metric_logdet": float(self.state_metric_logdet),
            "sensitivity_condition": float(self.sensitivity_condition),
            "sensitivity_valid": bool(self.sensitivity_valid),
            "min_constraint": float(self.min_constraint),
            "min_obstacle_constraint": float(self.min_obstacle_constraint),
            "min_road_constraint": float(self.min_road_constraint),
            "min_kinematic_constraint": float(self.min_kinematic_constraint),
            "softmin_gap": float(self.softmin_gap),
        }


def normalized_target_to_xy(
    action: np.ndarray,
    *,
    position_scale: float,
    lateral_scale: float | None = None,
) -> np.ndarray:
    action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
    lateral_scale = float(position_scale if lateral_scale is None else lateral_scale)
    return np.array(
        [0.5 * (action[0] + 1.0) * float(position_scale), action[1] * lateral_scale],
        dtype=np.float64,
    )


def target_reference_from_action(
    action: np.ndarray,
    *,
    c1: np.ndarray,
    n_free: int,
    position_scale: float,
    lateral_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return free reference points and d(ref_flat)/d(normalized target)."""
    if n_free <= 0:
        raise ValueError("n_free must be positive")
    lateral_scale = float(position_scale if lateral_scale is None else lateral_scale)
    target = normalized_target_to_xy(
        action,
        position_scale=position_scale,
        lateral_scale=lateral_scale,
    )
    c1 = np.asarray(c1, dtype=np.float64).reshape(2)
    raw_target_x = float(target[0])
    # c1 is fixed by the current velocity continuity constraint.  Letting a
    # short/braking target fall behind it folds the B-spline backward, which
    # can make the nominal emergency stop leave a curved road.  Saturating at
    # c1 produces a monotone deceleration/stopping reference instead.
    target[0] = max(raw_target_x, float(c1[0]))
    fractions = np.linspace(1.0 / n_free, 1.0, n_free, dtype=np.float64)
    ref = c1[None, :] + fractions[:, None] * (target[None, :] - c1[None, :])

    target_jac = np.diag([0.5 * float(position_scale), lateral_scale])
    if raw_target_x <= float(c1[0]):
        target_jac[0, 0] = 0.0
    ref_jac = np.zeros((2 * n_free, 2), dtype=np.float64)
    for index, fraction in enumerate(fractions):
        ref_jac[2 * index : 2 * index + 2] = fraction * target_jac
    return ref.astype(np.float32), ref_jac


def _softmin(values: np.ndarray, kappa: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 1.0
    scaled = -float(kappa) * values
    maximum = float(np.max(scaled))
    return float(-(maximum + np.log(np.sum(np.exp(scaled - maximum)))) / float(kappa))


def _route_clearance_and_tangent(
    xy_future: np.ndarray,
    route_xy: np.ndarray,
    route_normal: np.ndarray,
    route_half_width: np.ndarray,
    route_mask: np.ndarray,
    *,
    assignment_scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(route_mask, dtype=bool).reshape(-1)
    if not np.any(mask) or xy_future.size == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    route_xy = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)[mask]
    route_normal = np.asarray(route_normal, dtype=np.float64).reshape(-1, 2)[mask]
    del assignment_scale
    route_half_width = np.asarray(route_half_width, dtype=np.float64).reshape(-1)[mask]
    query = np.asarray(xy_future, dtype=np.float64)
    if route_xy.shape[0] == 1:
        normal = route_normal[0] / max(float(np.linalg.norm(route_normal[0])), 1e-8)
        tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
        tangents = np.repeat(tangent[None, :], query.shape[0], axis=0)
        return (
            route_half_width[0] - np.linalg.norm(query - route_xy[0][None, :], axis=1),
            tangents,
        )
    p0 = route_xy[:-1]
    p1 = route_xy[1:]
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
        route_half_width[:-1][None, :] * (1.0 - fraction)
        + route_half_width[1:][None, :] * fraction
    )
    segment_clearance = width - distance
    best = np.argmax(segment_clearance, axis=1)
    tangent = segment / np.maximum(np.linalg.norm(segment, axis=1, keepdims=True), 1e-8)
    return segment_clearance[np.arange(query.shape[0]), best], tangent[best]


def certificate_constraints_from_free(
    free_ctrl_pts: np.ndarray,
    *,
    c0: np.ndarray,
    c1: np.ndarray,
    basis_matrix: np.ndarray,
    obs_xy: np.ndarray,
    obs_radius: np.ndarray,
    obs_mask: np.ndarray,
    d_clearance: float,
    route_xy: np.ndarray,
    route_normal: np.ndarray,
    route_half_width: np.ndarray,
    route_mask: np.ndarray,
    road_tightening: float,
    ego_half_width: float,
    ego_half_length: float,
) -> np.ndarray:
    free = np.asarray(free_ctrl_pts, dtype=np.float64).reshape(-1, 2)
    full = np.vstack(
        [
            np.asarray(c0, dtype=np.float64).reshape(1, 2),
            np.asarray(c1, dtype=np.float64).reshape(1, 2),
            free,
        ]
    )
    xy = np.asarray(basis_matrix, dtype=np.float64) @ full
    xy_future = xy[1:]
    constraints: list[np.ndarray] = []
    path_delta = xy_future - xy[:-1]
    path_norm = np.linalg.norm(path_delta, axis=1, keepdims=True)
    path_tangent = np.where(
        path_norm > 1e-8,
        path_delta / np.maximum(path_norm, 1e-8),
        np.array([[1.0, 0.0]], dtype=np.float64),
    )

    obstacle_mask = np.asarray(obs_mask, dtype=bool).reshape(-1)
    if np.any(obstacle_mask):
        obstacle_descriptor = np.asarray(obs_xy, dtype=np.float64)
        if obstacle_descriptor.ndim != 2 or obstacle_descriptor.shape[1] < 2:
            obstacle_descriptor = obstacle_descriptor.reshape(-1, 2)
        centers = obstacle_descriptor[obstacle_mask, :2]
        horizon_delta = (
            obstacle_descriptor[obstacle_mask, 2:4]
            if obstacle_descriptor.shape[1] >= 4
            else np.zeros_like(centers)
        )
        radii = np.asarray(obs_radius, dtype=np.float64).reshape(-1)[obstacle_mask]
        fractions = np.linspace(
            1.0 / max(xy_future.shape[0], 1),
            1.0,
            xy_future.shape[0],
            dtype=np.float64,
        )
        predicted_centers = (
            centers[None, :, :]
            + fractions[:, None, None] * horizon_delta[None, :, :]
        )
        diff = xy_future[:, None, :] - predicted_centers
        distance = np.linalg.norm(diff, axis=-1)
        radial = diff / np.maximum(distance[:, :, None], 1e-8)
        longitudinal_projection = np.abs(
            np.sum(path_tangent[:, None, :] * radial, axis=-1)
        )
        lateral_projection = np.abs(
            path_tangent[:, None, 0] * radial[:, :, 1]
            - path_tangent[:, None, 1] * radial[:, :, 0]
        )
        ego_support = (
            float(ego_half_length) * longitudinal_projection
            + float(ego_half_width) * lateral_projection
        )
        clearance = distance - (
            radii[None, :] + ego_support + float(d_clearance)
        )
        constraints.append(clearance.reshape(-1))

    road_clearance, route_tangent = _route_clearance_and_tangent(
        xy_future,
        route_xy,
        route_normal,
        route_half_width,
        route_mask,
    )
    if road_clearance.size:
        # Match FlatNMPC's oriented rectangular road constraint.  A constant
        # half-width is insufficient when the vehicle yaws relative to the
        # road: the front/rear corner can cross MetaDrive's continuous edge
        # line while the centre still appears comfortably inside.  Preserve
        # the configured straight-line tightening as an additional residual
        # reserve beyond the physical half-width.
        aligned = np.abs(np.sum(path_tangent * route_tangent, axis=1))
        crossed = np.abs(
            path_tangent[:, 0] * route_tangent[:, 1]
            - path_tangent[:, 1] * route_tangent[:, 0]
        )
        body_support = (
            float(ego_half_width) * aligned
            + float(ego_half_length) * crossed
        )
        residual_tightening = max(
            float(road_tightening) - float(ego_half_width), 0.0
        )
        constraints.append(
            road_clearance - body_support - residual_tightening
        )

    if not constraints:
        return np.asarray([1.0], dtype=np.float64)
    return np.concatenate(constraints).astype(np.float64)


def certificate_margin_from_free(
    free_ctrl_pts: np.ndarray,
    *,
    geometry: DriveMGFGeometryConfig,
    **kwargs,
) -> float:
    return _softmin(
        certificate_constraints_from_free(
            free_ctrl_pts,
            road_tightening=geometry.road_tightening,
            ego_half_width=geometry.ego_half_width,
            ego_half_length=geometry.ego_half_length,
            **kwargs,
        ),
        geometry.margin_softmin_kappa,
    )


def kinematic_margin_from_free(
    free_ctrl_pts: np.ndarray,
    *,
    c0: np.ndarray,
    c1: np.ndarray,
    basis_d1: np.ndarray,
    basis_d2: np.ndarray,
    kappa_max: float,
    v_min: float,
    eps: float = 1e-6,
) -> float:
    """Signed Flat-NMPC curvature slack for an optimized B-spline."""
    free = np.asarray(free_ctrl_pts, dtype=np.float64).reshape(-1, 2)
    full = np.vstack(
        [
            np.asarray(c0, dtype=np.float64).reshape(1, 2),
            np.asarray(c1, dtype=np.float64).reshape(1, 2),
            free,
        ]
    )
    dxy = np.asarray(basis_d1, dtype=np.float64) @ full
    ddxy = np.asarray(basis_d2, dtype=np.float64) @ full
    speed = np.maximum(
        np.sqrt(np.sum(dxy * dxy, axis=1) + float(eps)),
        float(v_min),
    )
    kappa = (
        dxy[:, 0] * ddxy[:, 1] - dxy[:, 1] * ddxy[:, 0]
    ) / (speed**3 + float(eps))
    return float(float(kappa_max) - np.max(np.abs(kappa)))


def compute_drive_mgf_field(
    *,
    action: np.ndarray,
    ref_free_ctrl_pts: np.ndarray,
    ref_jacobian: np.ndarray,
    optimized_free_ctrl_pts: np.ndarray,
    objective_hessian: np.ndarray,
    objective_value: float,
    tracking_weight: float,
    c0: np.ndarray,
    c1: np.ndarray,
    basis_matrix: np.ndarray,
    obs_xy: np.ndarray,
    obs_radius: np.ndarray,
    obs_mask: np.ndarray,
    d_clearance: float,
    route_xy: np.ndarray,
    route_normal: np.ndarray,
    route_half_width: np.ndarray,
    route_mask: np.ndarray,
    geometry: DriveMGFGeometryConfig,
    kinematic_margin: float | None = None,
    basis_d1: np.ndarray | None = None,
    basis_d2: np.ndarray | None = None,
    kappa_max: float | None = None,
    v_min: float = 1.0,
    flat_eps: float = 1e-6,
) -> DriveMGFField:
    """Pull optimized trajectory geometry back to the 2-D target space."""
    action = np.asarray(action, dtype=np.float64).reshape(2)
    ref_flat = np.asarray(ref_free_ctrl_pts, dtype=np.float64).reshape(-1)
    opt_free = np.asarray(optimized_free_ctrl_pts, dtype=np.float64).reshape(-1, 2)
    opt_flat = opt_free.reshape(-1)
    ref_jacobian = np.asarray(ref_jacobian, dtype=np.float64).reshape(opt_flat.size, 2)

    hessian = np.asarray(objective_hessian, dtype=np.float64).reshape(opt_flat.size, opt_flat.size)
    hessian = 0.5 * (hessian + hessian.T)
    hessian += geometry.metric_damping * np.eye(hessian.shape[0], dtype=np.float64)
    try:
        eigenvalues = np.linalg.eigvalsh(hessian)
        condition = float(np.max(eigenvalues) / max(float(np.min(eigenvalues)), geometry.metric_damping))
        dq_dp = np.linalg.solve(hessian, 2.0 * float(tracking_weight) * ref_jacobian)
        sensitivity_valid = bool(np.all(np.isfinite(dq_dp)) and np.isfinite(condition))
    except np.linalg.LinAlgError:
        condition = float("inf")
        dq_dp = np.zeros((opt_flat.size, 2), dtype=np.float64)
        sensitivity_valid = False

    margin_kwargs = {
        "c0": c0,
        "c1": c1,
        "basis_matrix": basis_matrix,
        "obs_xy": obs_xy,
        "obs_radius": obs_radius,
        "obs_mask": obs_mask,
        "d_clearance": d_clearance,
        "route_xy": route_xy,
        "route_normal": route_normal,
        "route_half_width": route_half_width,
        "route_mask": route_mask,
    }
    constraints = certificate_constraints_from_free(
        opt_free,
        road_tightening=geometry.road_tightening,
        ego_half_width=geometry.ego_half_width,
        ego_half_length=geometry.ego_half_length,
        **margin_kwargs,
    )
    min_spatial_constraint = float(np.min(constraints))
    min_kinematic_constraint = (
        float("nan") if kinematic_margin is None else float(kinematic_margin)
    )
    # Keep one smooth certificate across obstacle, road, and Flat-MPC
    # kinematic constraints.  A hard min against the curvature slack makes
    # the target-space gradient jump exactly where obstacle avoidance and
    # steering feasibility must be traded off.
    composite_constraints = constraints
    if np.isfinite(min_kinematic_constraint):
        composite_constraints = np.concatenate(
            [constraints, np.asarray([min_kinematic_constraint], dtype=np.float64)]
        )
    margin = _softmin(composite_constraints, geometry.margin_softmin_kappa)
    min_constraint = (
        min(min_spatial_constraint, min_kinematic_constraint)
        if np.isfinite(min_kinematic_constraint)
        else min_spatial_constraint
    )
    future_samples = max(int(np.asarray(basis_matrix).shape[0]) - 1, 0)
    obstacle_count = future_samples * int(np.sum(np.asarray(obs_mask, dtype=bool)))
    if obstacle_count:
        min_obstacle_constraint = float(np.min(constraints[:obstacle_count]))
    else:
        min_obstacle_constraint = float("nan")
    if constraints.size > obstacle_count and np.any(np.asarray(route_mask, dtype=bool)):
        min_road_constraint = float(np.min(constraints[obstacle_count:]))
    else:
        min_road_constraint = float("nan")

    def composite_margin_from_free(candidate_flat: np.ndarray) -> float:
        candidate_free = np.asarray(candidate_flat, dtype=np.float64).reshape(
            opt_free.shape
        )
        candidate_constraints = certificate_constraints_from_free(
            candidate_free,
            road_tightening=geometry.road_tightening,
            ego_half_width=geometry.ego_half_width,
            ego_half_length=geometry.ego_half_length,
            **margin_kwargs,
        )
        candidate_kinematic = min_kinematic_constraint
        if (
            np.isfinite(min_kinematic_constraint)
            and basis_d1 is not None
            and basis_d2 is not None
            and kappa_max is not None
        ):
            candidate_kinematic = kinematic_margin_from_free(
                candidate_free,
                c0=c0,
                c1=c1,
                basis_d1=basis_d1,
                basis_d2=basis_d2,
                kappa_max=kappa_max,
                v_min=v_min,
                eps=flat_eps,
            )
        if np.isfinite(candidate_kinematic):
            candidate_constraints = np.concatenate(
                [
                    candidate_constraints,
                    np.asarray([candidate_kinematic], dtype=np.float64),
                ]
            )
        return _softmin(candidate_constraints, geometry.margin_softmin_kappa)

    grad_margin_q = np.zeros_like(opt_flat)
    step = float(geometry.margin_grad_step)
    for index in range(opt_flat.size):
        plus = opt_flat.copy()
        minus = opt_flat.copy()
        plus[index] += step
        minus[index] -= step
        grad_margin_q[index] = (
            composite_margin_from_free(plus)
            - composite_margin_from_free(minus)
        ) / (2.0 * step)
    grad_margin = grad_margin_q @ dq_dp

    grad_value_ref = 2.0 * float(tracking_weight) * (ref_flat - opt_flat)
    grad_value = ref_jacobian.T @ grad_value_ref

    free_basis = np.asarray(basis_matrix, dtype=np.float64)[:, 2:]
    state_sensitivity = np.empty((free_basis.shape[0], 2, 2), dtype=np.float64)
    dq_shaped = dq_dp.reshape(opt_free.shape[0], 2, 2)
    for target_index in range(2):
        state_sensitivity[:, :, target_index] = free_basis @ dq_shaped[:, :, target_index]
    if state_sensitivity.shape[0] > 1:
        state_sensitivity = state_sensitivity[1:]
    state_metric = np.mean(
        np.einsum("kdi,kdj->kij", state_sensitivity, state_sensitivity),
        axis=0,
    )
    state_metric += geometry.metric_damping * np.eye(2, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(state_metric)
    if sign <= 0 or not np.isfinite(logdet):
        logdet = float("-inf")

    margin_delta = (
        margin
        - float(geometry.chance_kappa)
        * np.sqrt(float(geometry.chance_beta))
        * float(np.linalg.norm(grad_margin))
        - float(geometry.conformal_delta)
    )
    finite = all(
        np.all(np.isfinite(value))
        for value in (grad_margin, grad_value, state_metric, dq_dp)
    ) and np.isfinite(margin) and np.isfinite(margin_delta) and np.isfinite(objective_value)
    return DriveMGFField(
        anchor=action.astype(np.float32),
        margin=float(margin),
        margin_delta=float(margin_delta),
        grad_margin=grad_margin.astype(np.float32),
        value=float(objective_value),
        grad_value=grad_value.astype(np.float32),
        state_metric=state_metric.astype(np.float32),
        state_metric_logdet=float(logdet),
        sensitivity_condition=float(condition),
        sensitivity_valid=bool(sensitivity_valid and finite),
        min_constraint=min_constraint,
        min_obstacle_constraint=min_obstacle_constraint,
        min_road_constraint=min_road_constraint,
        min_kinematic_constraint=min_kinematic_constraint,
        softmin_gap=float(min_constraint - margin),
    )
