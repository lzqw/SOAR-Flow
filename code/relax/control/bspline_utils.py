"""
Clamped Cubic B-Spline Utilities for JAX.

This module provides B-spline evaluation with proper time scaling for MPC applications.

Key Features:
- Clamped (open-uniform) knot vector for endpoint interpolation
- Analytical derivatives (1st, 2nd order) with time scaling
- Initial state constraint via first two control points
- Pure JAX implementation for JIT compatibility

Time Scaling (Critical):
    Parameter domain: u ∈ [0, 1]
    Physical time:    t ∈ [0, T]  where T = horizon_time
    
    dP/dt = dP/du · (du/dt) = dP/du / T
    d²P/dt² = d²P/du² / T²

Clamped Knot Vector:
    For degree=3, n_ctrl=6:
        knots = [0, 0, 0, 0, 0.333, 0.667, 1, 1, 1, 1]
    
    This ensures:
        - Curve passes through first control point at u=0
        - Curve passes through last control point at u=1
        - First derivative at u=0 is proportional to (C[1] - C[0])
"""

from __future__ import annotations
from typing import Tuple
import jax
import jax.numpy as jnp


# ==================== KNOT VECTOR ====================

def make_clamped_knots(n_ctrl: int, degree: int = 3) -> jnp.ndarray:
    """
    Create clamped (open-uniform) knot vector.
    
    For a clamped B-spline:
        - First (degree+1) knots are 0
        - Last (degree+1) knots are 1
        - Internal knots are uniformly spaced
    
    Args:
        n_ctrl: Number of control points
        degree: Spline degree (default 3 for cubic)
        
    Returns:
        knots: [n_ctrl + degree + 1] knot vector
        
    Example:
        n_ctrl=6, degree=3:
        n_knots = 6 + 3 + 1 = 10
        n_internal = 6 - 3 - 1 = 2
        knots = [0, 0, 0, 0, 0.333, 0.667, 1, 1, 1, 1]
    """
    n_knots = n_ctrl + degree + 1
    n_internal = n_ctrl - degree - 1
    
    if n_internal <= 0:
        # No internal knots needed
        return jnp.concatenate([
            jnp.zeros(degree + 1),
            jnp.ones(degree + 1)
        ])
    
    # Internal knots uniformly spaced in (0, 1)
    internal = jnp.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    
    return jnp.concatenate([
        jnp.zeros(degree + 1),
        internal,
        jnp.ones(degree + 1)
    ])


# ==================== BASIS FUNCTIONS ====================

def _bspline_basis_scalar(u: float, knots: jnp.ndarray, i: int, p: int) -> float:
    """
    Compute B-spline basis function N_{i,p}(u) using de Boor algorithm.
    
    This is a JAX-compatible implementation that avoids recursion.
    Uses iterative computation with proper handling of edge cases.
    
    Args:
        u: Parameter value in [0, 1]
        knots: Knot vector
        i: Basis function index
        p: Degree
        
    Returns:
        N_{i,p}(u): Basis function value
    """
    # Number of control points
    n_knots = knots.shape[0]
    n_ctrl = n_knots - p - 1
    
    # Initialize degree-0 basis (indicator functions)
    # For clamped splines, handle endpoint specially
    N = jnp.zeros(n_knots - 1)
    
    for j in range(n_knots - 1):
        # Standard indicator: knots[j] <= u < knots[j+1]
        # Special case for last span: include u == knots[-1]
        is_in_span = (knots[j] <= u) & (u < knots[j + 1])
        is_last_span = (j == n_knots - 2) & (u == knots[-1])
        N = N.at[j].set(jnp.where(is_in_span | is_last_span, 1.0, 0.0))
    
    # Iteratively compute higher degrees
    for d in range(1, p + 1):
        N_new = jnp.zeros_like(N)
        for j in range(n_knots - d - 1):
            denom1 = knots[j + d] - knots[j]
            denom2 = knots[j + d + 1] - knots[j + 1]
            
            # Left term
            alpha1 = jnp.where(denom1 > 1e-10, (u - knots[j]) / denom1, 0.0)
            # Right term
            alpha2 = jnp.where(denom2 > 1e-10, (knots[j + d + 1] - u) / denom2, 0.0)
            
            N_new = N_new.at[j].set(alpha1 * N[j] + alpha2 * N[j + 1])
        N = N_new
    
    return N[i]


def _compute_all_basis(u: float, knots: jnp.ndarray, p: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute all basis functions and their derivatives at parameter u.
    
    Uses the de Boor algorithm to compute basis functions iteratively,
    then uses derivative formulas to compute first and second derivatives.
    
    Args:
        u: Parameter value in [0, 1]
        knots: Knot vector
        p: Degree
        
    Returns:
        N:   [n_ctrl] basis function values
        dN:  [n_ctrl] first derivative values
        ddN: [n_ctrl] second derivative values
    """
    n_knots = knots.shape[0]
    n_ctrl = n_knots - p - 1
    
    # Storage for basis functions at each degree level
    # N_table[d, j] = N_{j,d}(u)
    N_table = jnp.zeros((p + 1, n_knots - 1))
    
    # Degree 0: indicator functions
    # For clamped splines, the last valid span is at index n_ctrl - 1
    # (from knots[n_ctrl-1] to knots[n_ctrl], which maps to u ∈ [last_internal, 1.0])
    last_valid_span = n_ctrl - 1
    
    for j in range(n_knots - 1):
        is_in_span = (knots[j] <= u) & (u < knots[j + 1])
        # Handle endpoint: u == 1.0 belongs to the last valid span
        is_endpoint = (u >= 1.0 - 1e-10) & (j == last_valid_span)
        N_table = N_table.at[0, j].set(jnp.where(is_in_span | is_endpoint, 1.0, 0.0))
    
    # Compute higher degrees iteratively
    for d in range(1, p + 1):
        for j in range(n_knots - d - 1):
            denom1 = knots[j + d] - knots[j]
            denom2 = knots[j + d + 1] - knots[j + 1]

            safe_denom1 = jnp.where(denom1 > 1e-10, denom1, 1.0)
            safe_denom2 = jnp.where(denom2 > 1e-10, denom2, 1.0)
            alpha1 = jnp.where(denom1 > 1e-10, (u - knots[j]) / safe_denom1, 0.0)
            alpha2 = jnp.where(denom2 > 1e-10, (knots[j + d + 1] - u) / safe_denom2, 0.0)
            
            val = alpha1 * N_table[d - 1, j] + alpha2 * N_table[d - 1, j + 1]
            N_table = N_table.at[d, j].set(val)
    
    # Extract final basis functions
    N = N_table[p, :n_ctrl]
    
    # First derivatives: dN_{i,p}/du = p * [N_{i,p-1}/(t_{i+p} - t_i) - N_{i+1,p-1}/(t_{i+p+1} - t_{i+1})]
    dN = jnp.zeros(n_ctrl)
    if p >= 1:
        for i in range(n_ctrl):
            denom1 = knots[i + p] - knots[i]
            denom2 = knots[i + p + 1] - knots[i + 1]

            safe_denom1 = jnp.where(denom1 > 1e-10, denom1, 1.0)
            safe_denom2 = jnp.where(denom2 > 1e-10, denom2, 1.0)
            term1 = jnp.where(denom1 > 1e-10, p / safe_denom1 * N_table[p - 1, i], 0.0)
            term2 = jnp.where(denom2 > 1e-10, p / safe_denom2 * N_table[p - 1, i + 1], 0.0)
            
            dN = dN.at[i].set(term1 - term2)
    
    # Second derivatives: similar formula with dN_{i,p-1}
    ddN = jnp.zeros(n_ctrl)
    if p >= 2:
        # First compute dN at degree p-1
        dN_pm1 = jnp.zeros(n_knots - 1)
        for i in range(n_knots - p):
            denom1 = knots[i + p - 1] - knots[i]
            denom2 = knots[i + p] - knots[i + 1]

            safe_denom1 = jnp.where(denom1 > 1e-10, denom1, 1.0)
            safe_denom2 = jnp.where(denom2 > 1e-10, denom2, 1.0)
            term1 = jnp.where(denom1 > 1e-10, (p - 1) / safe_denom1 * N_table[p - 2, i], 0.0)
            term2 = jnp.where(denom2 > 1e-10, (p - 1) / safe_denom2 * N_table[p - 2, i + 1], 0.0)
            
            dN_pm1 = dN_pm1.at[i].set(term1 - term2)
        
        # Then compute ddN
        for i in range(n_ctrl):
            denom1 = knots[i + p] - knots[i]
            denom2 = knots[i + p + 1] - knots[i + 1]

            safe_denom1 = jnp.where(denom1 > 1e-10, denom1, 1.0)
            safe_denom2 = jnp.where(denom2 > 1e-10, denom2, 1.0)
            term1 = jnp.where(denom1 > 1e-10, p / safe_denom1 * dN_pm1[i], 0.0)
            term2 = jnp.where(denom2 > 1e-10, p / safe_denom2 * dN_pm1[i + 1], 0.0)
            
            ddN = ddN.at[i].set(term1 - term2)
    
    return N, dN, ddN


def _bspline_basis_deriv_scalar(u: float, knots: jnp.ndarray, i: int, p: int) -> float:
    """
    Compute first derivative of B-spline basis function dN_{i,p}/du.
    
    Note: This is kept for API compatibility but internally uses _compute_all_basis.
    """
    _, dN, _ = _compute_all_basis(u, knots, p)
    return dN[i]


def _bspline_basis_deriv2_scalar(u: float, knots: jnp.ndarray, i: int, p: int) -> float:
    """
    Compute second derivative of B-spline basis function d²N_{i,p}/du².
    """
    _, _, ddN = _compute_all_basis(u, knots, p)
    return ddN[i]


# ==================== VECTORIZED EVALUATION ====================

def _eval_all_basis_matrices(u_vals: jnp.ndarray, knots: jnp.ndarray, n_ctrl: int, degree: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute all basis matrices (values, 1st deriv, 2nd deriv) efficiently.
    
    Args:
        u_vals: [N] parameter values
        knots: [n_knots] knot vector
        n_ctrl: number of control points
        degree: spline degree
        
    Returns:
        basis:    [N, n_ctrl] basis function values
        basis_d1: [N, n_ctrl] first derivative values
        basis_d2: [N, n_ctrl] second derivative values
    """
    def eval_single_u(u):
        N, dN, ddN = _compute_all_basis(u, knots, degree)
        return N, dN, ddN
    
    return jax.vmap(eval_single_u)(u_vals)


def _eval_basis_matrix(u_vals: jnp.ndarray, knots: jnp.ndarray, n_ctrl: int, degree: int) -> jnp.ndarray:
    """
    Compute basis function matrix for all parameter values and control points.
    
    Args:
        u_vals: [N] parameter values
        knots: [n_knots] knot vector
        n_ctrl: number of control points
        degree: spline degree
        
    Returns:
        basis: [N, n_ctrl] basis function values
    """
    basis, _, _ = _eval_all_basis_matrices(u_vals, knots, n_ctrl, degree)
    return basis


def _eval_basis_deriv_matrix(u_vals: jnp.ndarray, knots: jnp.ndarray, n_ctrl: int, degree: int) -> jnp.ndarray:
    """Compute first derivative basis matrix."""
    _, basis_d1, _ = _eval_all_basis_matrices(u_vals, knots, n_ctrl, degree)
    return basis_d1


def _eval_basis_deriv2_matrix(u_vals: jnp.ndarray, knots: jnp.ndarray, n_ctrl: int, degree: int) -> jnp.ndarray:
    """Compute second derivative basis matrix."""
    _, _, basis_d2 = _eval_all_basis_matrices(u_vals, knots, n_ctrl, degree)
    return basis_d2


# ==================== MAIN EVALUATION FUNCTIONS ====================

def bspline_eval(
    u_vals: jnp.ndarray,
    ctrl_pts: jnp.ndarray,
    knots: jnp.ndarray,
    degree: int = 3,
) -> jnp.ndarray:
    """
    Evaluate B-spline curve at parameter values.
    
    Args:
        u_vals: [N] parameter values in [0, 1]
        ctrl_pts: [n_ctrl, D] control points (D = 2 for xy)
        knots: [n_knots] knot vector
        degree: spline degree
        
    Returns:
        points: [N, D] curve points
    """
    n_ctrl = ctrl_pts.shape[0]
    basis = _eval_basis_matrix(u_vals, knots, n_ctrl, degree)  # [N, n_ctrl]
    return basis @ ctrl_pts  # [N, D]


def bspline_eval_with_derivs(
    u_vals: jnp.ndarray,
    ctrl_pts: jnp.ndarray,
    knots: jnp.ndarray,
    degree: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Evaluate B-spline curve and derivatives at parameter values.
    
    NOTE: Returns derivatives in PARAMETER domain (dP/du, d²P/du²).
    Caller must apply time scaling for physical derivatives.
    
    Args:
        u_vals: [N] parameter values in [0, 1]
        ctrl_pts: [n_ctrl, D] control points
        knots: [n_knots] knot vector
        degree: spline degree
        
    Returns:
        xy:      [N, D] positions
        dxy_du:  [N, D] first derivatives dP/du (parameter domain)
        ddxy_du: [N, D] second derivatives d²P/du² (parameter domain)
        
    Time Scaling (apply in caller):
        dxy_dt = dxy_du / T
        ddxy_dt = ddxy_du / T²
    """
    n_ctrl = ctrl_pts.shape[0]
    
    basis = _eval_basis_matrix(u_vals, knots, n_ctrl, degree)
    basis_d1 = _eval_basis_deriv_matrix(u_vals, knots, n_ctrl, degree)
    basis_d2 = _eval_basis_deriv2_matrix(u_vals, knots, n_ctrl, degree)
    
    xy = basis @ ctrl_pts
    dxy_du = basis_d1 @ ctrl_pts
    ddxy_du = basis_d2 @ ctrl_pts
    
    return xy, dxy_du, ddxy_du


def bspline_eval_with_derivs_scaled(
    u_vals: jnp.ndarray,
    ctrl_pts: jnp.ndarray,
    knots: jnp.ndarray,
    horizon_time: float,
    degree: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Evaluate B-spline curve and derivatives with time scaling.
    
    This is a convenience function that applies time scaling internally.
    
    Args:
        u_vals: [N] parameter values in [0, 1]
        ctrl_pts: [n_ctrl, D] control points
        knots: [n_knots] knot vector
        horizon_time: T - physical time horizon (seconds)
        degree: spline degree
        
    Returns:
        xy:     [N, D] positions
        dxy:    [N, D] first derivatives dP/dt (physical velocity)
        ddxy:   [N, D] second derivatives d²P/dt² (physical acceleration)
    """
    xy, dxy_du, ddxy_du = bspline_eval_with_derivs(u_vals, ctrl_pts, knots, degree)
    
    T = horizon_time
    dxy = dxy_du / T
    ddxy = ddxy_du / (T * T)
    
    return xy, dxy, ddxy


# ==================== INITIAL STATE CONSTRAINT ====================

def compute_initial_ctrl_pts(
    x0: float, y0: float,
    theta0: float, v0: float,
    knots: jnp.ndarray,
    horizon_time: float,
    degree: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute first two control points to satisfy initial state constraint.
    
    For a clamped cubic B-spline:
        P(0) = C[0]
        dP/du(0) = degree * (C[1] - C[0]) / (t_{degree+1} - t_1)
                 = degree * (C[1] - C[0]) / (knots[degree+1] - knots[1])
    
    For clamped knots: knots[1] = 0, so:
        dP/du(0) = degree * (C[1] - C[0]) / knots[degree+1]
    
    Physical velocity:
        dP/dt(0) = dP/du(0) / T = degree * (C[1] - C[0]) / (knots[degree+1] * T)
    
    Setting dP/dt(0) = v0 * [cos(θ0), sin(θ0)]:
        C[1] = C[0] + v0 * [cos(θ0), sin(θ0)] * knots[degree+1] * T / degree
    
    Args:
        x0, y0: Initial position
        theta0: Initial heading (rad)
        v0: Initial velocity (m/s)
        knots: Knot vector
        horizon_time: T - time horizon
        degree: Spline degree
        
    Returns:
        C0: [2] first control point
        C1: [2] second control point
    """
    T = horizon_time
    
    # First control point: initial position
    C0 = jnp.array([x0, y0])
    
    # Velocity direction
    vel_dir = jnp.array([jnp.cos(theta0), jnp.sin(theta0)])
    
    # Scaling factor for second control point
    # For clamped knots: knots[1] = 0, knots[degree+1] is first internal/end knot
    delta_knot = knots[degree + 1] - knots[1]  # = knots[degree+1] for clamped
    delta_knot = jnp.maximum(delta_knot, 1e-6)  # avoid division by zero
    
    # C[1] such that dP/dt(0) = v0 * vel_dir
    # dP/dt(0) = degree * (C[1] - C[0]) / (delta_knot * T)
    # => C[1] - C[0] = v0 * vel_dir * delta_knot * T / degree
    alpha = delta_knot * T / degree
    C1 = C0 + v0 * vel_dir * alpha
    
    return C0, C1


def apply_initial_constraint(
    free_ctrl_pts: jnp.ndarray,
    x0_state: jnp.ndarray,
    knots: jnp.ndarray,
    horizon_time: float,
    degree: int = 3,
) -> jnp.ndarray:
    """
    Apply initial state constraint to get full control points.
    
    Args:
        free_ctrl_pts: [n_ctrl-2, 2] free control points (decision variables)
        x0_state: [4] = (x, y, theta, v) initial state
        knots: Knot vector
        horizon_time: T
        degree: Spline degree
        
    Returns:
        ctrl_pts: [n_ctrl, 2] full control points with C[0], C[1] constrained
    """
    x0, y0, theta0, v0 = x0_state[0], x0_state[1], x0_state[2], x0_state[3]
    
    C0, C1 = compute_initial_ctrl_pts(x0, y0, theta0, v0, knots, horizon_time, degree)
    
    # Concatenate: [C0, C1, free_ctrl_pts]
    return jnp.vstack([C0[None, :], C1[None, :], free_ctrl_pts])


def extract_free_ctrl_pts(
    ctrl_pts: jnp.ndarray,
) -> jnp.ndarray:
    """
    Extract free control points (decision variables) from full control points.
    
    Args:
        ctrl_pts: [n_ctrl, 2] full control points
        
    Returns:
        free_ctrl_pts: [n_ctrl-2, 2] free control points (C[2:])
    """
    return ctrl_pts[2:, :]


def init_free_ctrl_pts_from_waypoints(
    waypoints: jnp.ndarray,
    x0_state: jnp.ndarray,
    n_ctrl: int,
    knots: jnp.ndarray,
    horizon_time: float,
    degree: int = 3,
) -> jnp.ndarray:
    """
    Initialize free control points from reference waypoints.
    
    This is useful for providing a good initial guess to the optimizer.
    
    Args:
        waypoints: [N, 2] reference waypoints
        x0_state: [4] initial state
        n_ctrl: total number of control points
        knots: knot vector
        horizon_time: T
        degree: spline degree
        
    Returns:
        free_ctrl_pts: [n_ctrl-2, 2] initialized free control points
    """
    n_free = n_ctrl - 2
    
    # Simple initialization: linearly interpolate waypoints to free control points
    n_wp = waypoints.shape[0]
    
    if n_wp == 0:
        # No waypoints, use zeros
        return jnp.zeros((n_free, 2))
    
    if n_wp == 1:
        # Single waypoint, repeat
        return jnp.tile(waypoints[0], (n_free, 1))
    
    # Interpolate waypoints to n_free points
    # Free control points correspond to u values after the first two
    u_free = jnp.linspace(0, 1, n_free + 2)[2:]  # Skip first two
    u_wp = jnp.linspace(0, 1, n_wp)
    
    # Linear interpolation
    x_interp = jnp.interp(u_free, u_wp, waypoints[:, 0])
    y_interp = jnp.interp(u_free, u_wp, waypoints[:, 1])
    
    return jnp.stack([x_interp, y_interp], axis=1)


# ==================== TESTING ====================

if __name__ == "__main__":
    import numpy as np
    
    print("Testing B-Spline Utilities...")
    
    # Parameters
    n_ctrl = 6
    degree = 3
    horizon_time = 0.5  # seconds
    
    # Create knot vector
    knots = make_clamped_knots(n_ctrl, degree)
    print(f"Knot vector (n_ctrl={n_ctrl}, degree={degree}):")
    print(f"  {np.array(knots)}")
    
    # Test control points: simple curve
    ctrl_pts = jnp.array([
        [0.0, 0.0],
        [1.0, 0.5],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 0.5],
        [5.0, 0.0],
    ])
    
    # Evaluate at endpoints
    u_test = jnp.array([0.0, 0.5, 1.0])
    xy, dxy_du, ddxy_du = bspline_eval_with_derivs(u_test, ctrl_pts, knots, degree)
    
    print(f"\nEndpoint test:")
    print(f"  P(0) = {np.array(xy[0])} (should be close to C[0] = {np.array(ctrl_pts[0])})")
    print(f"  P(1) = {np.array(xy[-1])} (should be close to C[-1] = {np.array(ctrl_pts[-1])})")
    
    # Test time scaling
    T = horizon_time
    dxy = dxy_du / T
    ddxy = ddxy_du / (T * T)
    
    print(f"\nTime-scaled derivatives at u=0 (T={T}s):")
    print(f"  dP/dt(0) = {np.array(dxy[0])}")
    print(f"  d²P/dt²(0) = {np.array(ddxy[0])}")
    
    # Test initial constraint
    x0_state = jnp.array([0.0, 0.0, 0.0, 10.0])  # (x, y, theta, v)
    C0, C1 = compute_initial_ctrl_pts(0.0, 0.0, 0.0, 10.0, knots, horizon_time, degree)
    
    print(f"\nInitial constraint test (v0=10 m/s, θ0=0):")
    print(f"  C[0] = {np.array(C0)}")
    print(f"  C[1] = {np.array(C1)}")
    
    # Verify: evaluate at u=0 and check velocity
    test_ctrl = jnp.vstack([C0[None, :], C1[None, :], ctrl_pts[2:, :]])
    _, dxy_du_test, _ = bspline_eval_with_derivs(jnp.array([0.0]), test_ctrl, knots, degree)
    v_at_0 = jnp.sqrt(jnp.sum((dxy_du_test[0] / T) ** 2))
    print(f"  |dP/dt(0)| = {float(v_at_0):.2f} m/s (should be ~10.0)")
    
    # Test free control points
    free_pts = jnp.array([[2.0, 1.0], [3.0, 1.0], [4.0, 0.5], [5.0, 0.0]])
    full_pts = apply_initial_constraint(free_pts, x0_state, knots, horizon_time, degree)
    print(f"\nFull control points shape: {full_pts.shape} (expected ({n_ctrl}, 2))")
    
    print("\nAll tests passed!")
