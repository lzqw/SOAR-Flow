import unittest
import importlib.util

import jax.numpy as jnp
import numpy as np

from relax.control.bspline_utils import make_clamped_knots
from relax.control.flat_nmpc import (
    BatchedFlatNMPC,
    BatchedFlatNMPCConfig,
    FlatNMPCConfig,
    FlatSpaceNMPC,
    _compute_cbf_soft_cost,
    _compute_body_road_clearance,
    _compute_constraint_terms,
    _compute_distance_risk_cost,
    _compute_road_soft_cost,
    _compute_route_edge_clearance,
    _pack_config,
)
from relax.env.drive.lane_change_mpc import (
    _compute_route_road_cost_np,
    _sample_route_corridor_from_lane_sets,
)


class _FakeStraightLane:
    def __init__(self, start_x: float, length: float = 10.0, width: float = 3.7):
        self.start_x = float(start_x)
        self.length = float(length)
        self.width = float(width)

    def local_coordinates(self, position):
        pos = np.asarray(position, dtype=np.float32)
        return float(pos[0] - self.start_x), float(pos[1])

    def width_at(self, longitudinal):
        del longitudinal
        return self.width

    def heading_theta_at(self, longitudinal):
        del longitudinal
        return 0.0

    def position(self, longitudinal, lateral):
        return np.array([self.start_x + float(longitudinal), float(lateral)], dtype=np.float32)


class TestFlatNMPC(unittest.TestCase):
    def test_distance_risk_cost_only_activates_near_warning_band(self):
        dist_sq = jnp.array(
            [
                [9.0, 16.0],
                [4.41, 25.0],
            ],
            dtype=jnp.float32,
        )
        d_safe = jnp.array([2.0, 2.0], dtype=jnp.float32)
        obs_mask = jnp.array([True, False])

        near_cost = _compute_distance_risk_cost(
            dist_sq,
            d_safe,
            obs_mask,
            risk_buffer=1.0,
            risk_temperature=0.25,
        )
        far_cost = _compute_distance_risk_cost(
            jnp.array([[25.0, 36.0]], dtype=jnp.float32),
            d_safe,
            obs_mask,
            risk_buffer=1.0,
            risk_temperature=0.25,
        )

        self.assertGreater(float(near_cost), 0.0)
        self.assertLess(float(far_cost), 1e-3)

    def test_cbf_soft_cost_penalizes_negative_margin(self):
        unsafe_cost = _compute_cbf_soft_cost(
            jnp.array([[-1.0, 3.0], [-0.5, 2.0]], dtype=jnp.float32),
            jnp.array([True, False]),
            temperature=0.25,
        )
        safe_cost = _compute_cbf_soft_cost(
            jnp.array([[3.0, 3.0], [2.0, 2.0]], dtype=jnp.float32),
            jnp.array([True, False]),
            temperature=0.25,
        )

        self.assertGreater(float(unsafe_cost), float(safe_cost))
        self.assertLessEqual(float(unsafe_cost), 1.0)
        self.assertLess(float(safe_cost), 1e-3)

    def test_cbf_soft_cost_is_gated_by_near_field_buffer(self):
        near_cost = _compute_cbf_soft_cost(
            jnp.array([[-1.0]], dtype=jnp.float32),
            jnp.array([True]),
            temperature=0.5,
            dist_sq=jnp.array([[4.0]], dtype=jnp.float32),
            d_safe=jnp.array([1.0], dtype=jnp.float32),
            risk_buffer=2.0,
        )
        far_cost = _compute_cbf_soft_cost(
            jnp.array([[-1.0]], dtype=jnp.float32),
            jnp.array([True]),
            temperature=0.5,
            dist_sq=jnp.array([[100.0]], dtype=jnp.float32),
            d_safe=jnp.array([1.0], dtype=jnp.float32),
            risk_buffer=2.0,
        )

        self.assertGreater(float(near_cost), 20.0 * float(far_cost))

    def test_route_edge_clearance_is_positive_inside_and_negative_outside(self):
        route_xy = jnp.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=jnp.float32)
        route_normal = jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=jnp.float32)
        route_half_width = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
        route_mask = jnp.array([True, True, True])

        inside_clearance = _compute_route_edge_clearance(
            jnp.array([[0.0, 0.0], [1.0, 0.2], [2.0, 0.3]], dtype=jnp.float32),
            route_xy,
            route_normal,
            route_half_width,
            route_mask,
        )
        outside_clearance = _compute_route_edge_clearance(
            jnp.array([[0.0, 0.0], [1.0, 2.6], [2.0, 3.0]], dtype=jnp.float32),
            route_xy,
            route_normal,
            route_half_width,
            route_mask,
        )

        self.assertGreater(float(jnp.min(inside_clearance)), 0.0)
        self.assertLess(float(jnp.min(outside_clearance)), 0.0)

    def test_body_road_clearance_accounts_for_heading_and_half_length(self):
        clearance = _compute_body_road_clearance(
            jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=jnp.float32),
            jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=jnp.float32),
            jnp.array([[0.0, 0.0], [3.0, 0.0]], dtype=jnp.float32),
            jnp.array([[0.0, 1.0], [0.0, 1.0]], dtype=jnp.float32),
            jnp.array([2.0, 2.0], dtype=jnp.float32),
            jnp.array([True, True]),
            jnp.asarray(0.926, dtype=jnp.float32),
            jnp.asarray(2.2575, dtype=jnp.float32),
        )

        self.assertAlmostEqual(float(clearance[0]), 2.0 - 0.926, places=5)
        self.assertAlmostEqual(
            float(clearance[1]),
            2.0 - (0.926 + 2.2575) / np.sqrt(2.0),
            places=5,
        )
        self.assertLess(float(clearance[1]), 0.0)

    def test_route_soft_cost_increases_near_and_outside_edge(self):
        route_mask = jnp.array([True, True, True])
        inside_cost = _compute_road_soft_cost(
            jnp.array([1.8, 1.6], dtype=jnp.float32),
            route_mask,
            road_buffer=0.5,
            road_temperature=0.5,
        )
        near_edge_cost = _compute_road_soft_cost(
            jnp.array([0.3, 0.2], dtype=jnp.float32),
            route_mask,
            road_buffer=0.5,
            road_temperature=0.5,
        )
        outside_cost = _compute_road_soft_cost(
            jnp.array([-0.5, -0.8], dtype=jnp.float32),
            route_mask,
            road_buffer=0.5,
            road_temperature=0.5,
        )

        self.assertLess(float(inside_cost), float(near_edge_cost))
        self.assertLess(float(near_edge_cost), float(outside_cost))

    def test_route_corridor_sampling_is_continuous_across_current_and_next_segments(self):
        current_ref_lanes = [_FakeStraightLane(0.0) for _ in range(3)]
        next_ref_lanes = [_FakeStraightLane(10.0) for _ in range(3)]

        route_xy, route_normal, route_half_width, route_mask = _sample_route_corridor_from_lane_sets(
            current_ref_lanes,
            next_ref_lanes,
            np.array([8.0, 0.0], dtype=np.float32),
            lookahead=4.0,
            max_samples=5,
        )

        self.assertTrue(np.all(route_mask))
        np.testing.assert_allclose(route_xy[:, 0], np.array([8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(route_normal[:, 0], np.zeros((5,), dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(np.abs(route_normal[:, 1]), np.ones((5,), dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(route_half_width, np.full((5,), 0.5 * 3.0 * 3.7, dtype=np.float32), atol=1e-5)

    def test_numpy_route_road_cost_penalizes_off_corridor_reference_path(self):
        route_xy = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32)
        route_normal = np.array([[0.0, 1.0]] * 4, dtype=np.float32)
        route_half_width = np.full((4,), 2.0, dtype=np.float32)
        route_mask = np.array([True, True, True, True])

        inside_cost, inside_clearance = _compute_route_road_cost_np(
            np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.2], [3.0, 0.1]], dtype=np.float32),
            route_xy,
            route_normal,
            route_half_width,
            route_mask,
            road_buffer=0.5,
            road_temperature=0.5,
        )
        outside_cost, outside_clearance = _compute_route_road_cost_np(
            np.array([[0.0, 0.0], [1.0, 2.8], [2.0, 3.0], [3.0, 2.6]], dtype=np.float32),
            route_xy,
            route_normal,
            route_half_width,
            route_mask,
            road_buffer=0.5,
            road_temperature=0.5,
        )

        self.assertLess(inside_cost, outside_cost)
        self.assertGreater(inside_clearance, 0.0)
        self.assertLess(outside_clearance, 0.0)

    def test_constraint_terms_use_cbf_and_road_hard_constraints(self):
        config = FlatNMPCConfig(
            n_ctrl_points=3,
            spline_degree=2,
            n_eval_points=5,
            cbf_enable=True,
            d_clearance=0.5,
            max_obstacles=4,
            max_route_samples=5,
        )
        free_pts = jnp.array([[6.0, 0.0]], dtype=jnp.float32)
        x0 = jnp.array([0.0, 0.0, 0.0, 4.0], dtype=jnp.float32)
        config_arr = _pack_config(config)
        knots = make_clamped_knots(config.n_ctrl_points, config.spline_degree)
        t_eval = jnp.linspace(0.0, 1.0, config.n_eval_points)
        obs_xy = jnp.array(
            [[5.5, 0.0], [7.5, 0.0], [9.5, 0.0], [11.5, 0.0]],
            dtype=jnp.float32,
        )
        obs_radius = jnp.full((4,), 1.0, dtype=jnp.float32)
        obs_mask = jnp.ones((4,), dtype=bool)
        route_xy = jnp.column_stack(
            [jnp.linspace(0.0, 8.0, config.max_route_samples), jnp.zeros(config.max_route_samples)]
        )
        route_normal = jnp.tile(
            jnp.array([[0.0, 1.0]], dtype=jnp.float32),
            (config.max_route_samples, 1),
        )
        route_half_width = jnp.full(
            (config.max_route_samples,), 3.7, dtype=jnp.float32
        )
        route_mask = jnp.ones((config.max_route_samples,), dtype=bool)

        _, _, _, _, constraints, _, _, _, _ = _compute_constraint_terms(
            free_pts,
            x0,
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
            config.spline_degree,
        )

        # Curvature contributes two constraints at every evaluation point;
        # obstacle safety contributes one exponential-CBF constraint per
        # future point/circle; the ego footprint contributes one routed-road
        # constraint per future point.  Euclidean distance is diagnostic only
        # and is deliberately not duplicated as a second obstacle hard bound.
        expected_constraints = (
            2 * config.n_eval_points
            + (config.n_eval_points - 1) * config.max_obstacles
            + (config.n_eval_points - 1)
        )
        self.assertEqual(constraints.shape[0], expected_constraints)

    def test_build_obstacle_circle_approx_splits_bus_into_passable_circles(self):
        if importlib.util.find_spec("metadrive") is None:
            self.skipTest("metadrive is required for obstacle approximation helper")
        from relax.env.drive.lane_change import build_obstacle_circle_approx

        positions, radii = build_obstacle_circle_approx(
            center_xy=np.array([0.0, 0.0], dtype=np.float32),
            heading=0.0,
            length=12.0,
            width=2.5,
            max_circles=4,
        )

        np.testing.assert_allclose(radii, np.full((4,), 1.25, dtype=np.float32))
        np.testing.assert_allclose(
            positions,
            np.array(
                [
                    [-4.5, 0.0],
                    [-1.5, 0.0],
                    [1.5, 0.0],
                    [4.5, 0.0],
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
        )

        d_clearance = 2.0
        self.assertLess(radii[0] + d_clearance, 3.7)

    def test_single_flat_nmpc_enforces_obstacle_constraints(self):
        config = FlatNMPCConfig(
            n_ctrl_points=5,
            spline_degree=3,
            horizon_time=1.0,
            wheelbase=2.5,
            max_steering=1.0,
            target_velocity=4.0,
            cbf_enable=True,
            d_clearance=0.5,
            cbf_gamma=1.0,
            max_obstacles=1,
            n_sqp_steps=8,
            n_admm_steps=80,
        )
        mpc = FlatSpaceNMPC(config)

        x0 = np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32)
        ref_free = np.array(
            [
                [4.0, 0.0],
                [8.0, 0.0],
                [12.0, 0.0],
            ],
            dtype=np.float32,
        )
        obs_xy = np.array([[7.0, 0.0]], dtype=np.float32)
        obs_radius = np.array([1.0], dtype=np.float32)
        obs_mask = np.array([True])

        result = mpc.step(
            x0,
            ref_free,
            obs_xy=obs_xy,
            obs_radius=obs_radius,
            obs_mask=obs_mask,
        )

        self.assertGreaterEqual(result.min_obstacle_dist, config.d_clearance + obs_radius[0] - 1e-3)
        self.assertTrue(np.isfinite(result.cost_cbf))
        self.assertTrue(np.isfinite(result.min_cbf_value))
        self.assertEqual(result.hj.shape, (6, 6))
        self.assertGreaterEqual(result.max_kappa_violation, 0.0)

    def test_shadow_certificate_checks_ordinary_mpc_without_cbf_authority(self):
        config = FlatNMPCConfig(
            n_ctrl_points=3,
            spline_degree=2,
            horizon_time=1.0,
            wheelbase=2.5,
            max_steering=1.0,
            target_velocity=4.0,
            cbf_enable=True,
            d_clearance=0.5,
            cbf_gamma=1.0,
            max_obstacles=1,
            n_sqp_steps=3,
            n_admm_steps=20,
        )
        mpc = FlatSpaceNMPC(config)
        x0 = np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32)
        reference = np.array([[8.0, 0.0]], dtype=np.float32)
        obstacle_xy = np.array([[5.0, 0.0]], dtype=np.float32)
        obstacle_radius = np.array([1.0], dtype=np.float32)
        obstacle_mask = np.array([True])

        ordinary = mpc.step(
            x0,
            reference,
            obs_xy=obstacle_xy,
            obs_radius=obstacle_radius,
            obs_mask=obstacle_mask,
            obstacle_cbf_enabled=False,
        )
        shadow = mpc.shadow_safety_certificate(
            ordinary.opt_free_ctrl_pts,
            x0,
            obs_xy=obstacle_xy,
            obs_radius=obstacle_radius,
            obs_mask=obstacle_mask,
        )

        self.assertGreater(ordinary.min_cbf_value, 1e5)
        self.assertTrue(shadow.has_obstacle)
        self.assertTrue(shadow.finite)
        self.assertTrue(np.isfinite(shadow.min_hard_cbf_margin))
        self.assertTrue(np.isfinite(shadow.min_obstacle_clearance))
        self.assertEqual(shadow.hard_cbf_margin.shape, (4, 1))

    def test_single_flat_nmpc_reports_road_diagnostics_and_finite_hessian(self):
        config = FlatNMPCConfig(
            n_ctrl_points=3,
            spline_degree=2,
            horizon_time=0.5,
            w_tracking=1.0,
            w_road=0.5,
            road_solve_weight_scale=0.25,
            road_buffer=0.5,
            road_temperature=0.5,
            max_route_samples=4,
            n_sqp_steps=4,
            n_admm_steps=20,
        )
        mpc = FlatSpaceNMPC(config)

        route_xy = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32)
        route_normal = np.array([[0.0, 1.0]] * 4, dtype=np.float32)
        route_half_width = np.full((4,), 2.0, dtype=np.float32)
        route_mask = np.array([True, True, True, True])

        result = mpc.step(
            np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32),
            np.array([[6.0, 2.5]], dtype=np.float32),
            route_xy=route_xy,
            route_normal=route_normal,
            route_half_width=route_half_width,
            route_mask=route_mask,
        )

        self.assertTrue(np.isfinite(result.cost_road))
        self.assertTrue(np.isfinite(result.min_road_clearance))
        self.assertEqual(result.hj.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(result.hj)))

    def test_batched_flat_nmpc_accepts_obstacle_batches(self):
        config = BatchedFlatNMPCConfig(
            batch_size=2,
            n_ctrl_points=5,
            spline_degree=3,
            horizon_time=1.0,
            max_steering=1.0,
            target_velocity=4.0,
            cbf_enable=True,
            d_clearance=0.5,
            cbf_gamma=1.0,
            max_obstacles=4,
            n_sqp_steps=4,
            n_admm_steps=40,
        )
        mpc = BatchedFlatNMPC(config)

        x0_batch = np.array(
            [
                [0.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 4.0],
            ],
            dtype=np.float32,
        )
        ref_batch = np.array(
            [
                [[4.0, 0.0], [8.0, 0.0], [12.0, 0.0]],
                [[4.0, 0.0], [8.0, 1.0], [12.0, 1.5]],
            ],
            dtype=np.float32,
        )
        obs_xy_batch = np.array(
            [
                [[5.5, 0.0], [7.5, 0.0], [9.5, 0.0], [11.5, 0.0]],
                [[5.5, 0.0], [7.5, 0.0], [9.5, 0.0], [11.5, 0.0]],
            ],
            dtype=np.float32,
        )
        obs_radius_batch = np.full((2, 4), 1.0, dtype=np.float32)
        obs_mask_batch = np.ones((2, 4), dtype=bool)

        controls, opt_free, full_pts, target_velocity_exec, diagnostics = mpc.step_batch(
            x0_batch,
            ref_batch,
            obs_xy_batch=obs_xy_batch,
            obs_radius_batch=obs_radius_batch,
            obs_mask_batch=obs_mask_batch,
        )

        self.assertEqual(controls.shape, (2, 2))
        self.assertEqual(opt_free.shape, (2, 3, 2))
        self.assertEqual(full_pts.shape, (2, 5, 2))
        self.assertEqual(target_velocity_exec.shape, (2,))
        self.assertEqual(diagnostics["min_obstacle_dist"].shape, (2,))
        self.assertEqual(diagnostics["min_cbf_value"].shape, (2,))
        self.assertEqual(diagnostics["cost_cbf"].shape, (2,))
        self.assertEqual(diagnostics["max_kappa_violation"].shape, (2,))
        self.assertEqual(diagnostics["hj"].shape, (2, 6, 6))

    def test_batched_flat_nmpc_reports_road_diagnostics(self):
        config = BatchedFlatNMPCConfig(
            batch_size=2,
            n_ctrl_points=3,
            spline_degree=2,
            horizon_time=0.5,
            w_tracking=1.0,
            w_road=0.5,
            road_solve_weight_scale=0.25,
            road_buffer=0.5,
            road_temperature=0.5,
            max_route_samples=4,
            n_sqp_steps=4,
            n_admm_steps=20,
        )
        mpc = BatchedFlatNMPC(config)

        x0_batch = np.array(
            [
                [0.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 4.0],
            ],
            dtype=np.float32,
        )
        ref_batch = np.array(
            [
                [[6.0, 0.2]],
                [[6.0, 2.6]],
            ],
            dtype=np.float32,
        )
        route_xy_batch = np.array(
            [
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            ],
            dtype=np.float32,
        )
        route_normal_batch = np.array(
            [
                [[0.0, 1.0]] * 4,
                [[0.0, 1.0]] * 4,
            ],
            dtype=np.float32,
        )
        route_half_width_batch = np.full((2, 4), 2.0, dtype=np.float32)
        route_mask_batch = np.array(
            [
                [True, True, True, True],
                [True, True, True, True],
            ]
        )

        controls, opt_free, full_pts, target_velocity_exec, diagnostics = mpc.step_batch(
            x0_batch,
            ref_batch,
            route_xy_batch=route_xy_batch,
            route_normal_batch=route_normal_batch,
            route_half_width_batch=route_half_width_batch,
            route_mask_batch=route_mask_batch,
        )

        self.assertEqual(controls.shape, (2, 2))
        self.assertEqual(opt_free.shape, (2, 1, 2))
        self.assertEqual(full_pts.shape, (2, 3, 2))
        self.assertEqual(target_velocity_exec.shape, (2,))
        self.assertEqual(diagnostics["cost_road"].shape, (2,))
        self.assertEqual(diagnostics["min_road_clearance"].shape, (2,))
        self.assertTrue(np.all(np.isfinite(diagnostics["hj"])))
        self.assertTrue(np.all(np.isfinite(diagnostics["cost_road"])))
        self.assertTrue(np.all(np.isfinite(diagnostics["min_road_clearance"])))
        # The latest SQP retains the best feasible iterate, so an off-corridor
        # reference can be corrected onto the road and need not finish with a
        # larger soft cost than an already-inside reference.  Both optimized
        # trajectories must still satisfy the hard road-domain constraint.
        self.assertTrue(np.all(diagnostics["min_road_clearance"] >= -1e-5))

    def test_cbf_disable_still_exposes_tracking_hessian_metric(self):
        config = FlatNMPCConfig(
            n_ctrl_points=3,
            spline_degree=2,
            horizon_time=0.5,
            cbf_enable=False,
            w_tracking=1.5,
        )
        mpc = FlatSpaceNMPC(config)

        result = mpc.step(
            np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32),
            np.array([[6.0, 0.0]], dtype=np.float32),
        )
        current_hj = mpc.get_current_hj(np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32))

        self.assertTrue(np.isfinite(result.steering))
        self.assertTrue(np.isfinite(result.throttle))
        np.testing.assert_allclose(
            current_hj,
            np.eye(2, dtype=np.float32) * (2.0 * config.w_tracking),
            atol=1e-4,
        )
        np.testing.assert_allclose(
            result.hj,
            np.eye(2, dtype=np.float32) * (2.0 * config.w_tracking),
            atol=1e-4,
        )

    def test_metric_hessian_stays_tracking_only_without_obstacles(self):
        x0 = np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32)
        expected = np.eye(2, dtype=np.float32) * 3.0

        config = FlatNMPCConfig(
            n_ctrl_points=3,
            spline_degree=2,
            horizon_time=0.5,
            cbf_enable=True,
            w_tracking=1.5,
            w_risk=3.0,
            max_obstacles=2,
        )
        mpc = FlatSpaceNMPC(config)
        current_hj = mpc.get_current_hj(x0)
        np.testing.assert_allclose(current_hj, expected, atol=1e-4)

    def test_metric_hessian_is_finite_for_single_obstacle(self):
        config = FlatNMPCConfig(
            n_ctrl_points=3,
            spline_degree=2,
            horizon_time=0.5,
            cbf_enable=True,
            w_tracking=1.0,
            w_risk=3.0,
            risk_buffer=1.0,
            risk_temperature=0.3,
            d_clearance=0.5,
            max_obstacles=1,
        )
        x0 = np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32)
        prev_free = np.array([[6.0, 0.25]], dtype=np.float32)
        obs_xy = np.array([[7.0, 0.5]], dtype=np.float32)
        obs_radius = np.array([1.0], dtype=np.float32)
        obs_mask = np.array([True])

        mpc = FlatSpaceNMPC(config)
        current_hj = mpc.get_current_hj(
            x0,
            prev_free_ctrl_pts=prev_free,
            obs_xy=obs_xy,
            obs_radius=obs_radius,
            obs_mask=obs_mask,
        )

        self.assertEqual(current_hj.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(current_hj)))

    def test_metric_hessian_is_finite_with_smoothmax_near_obstacle_switch(self):
        config = FlatNMPCConfig(
            n_ctrl_points=3,
            spline_degree=2,
            horizon_time=0.5,
            cbf_enable=True,
            w_tracking=1.0,
            w_risk=3.0,
            risk_buffer=1.0,
            risk_temperature=0.3,
            d_clearance=0.5,
            max_obstacles=2,
        )
        mpc = FlatSpaceNMPC(config)
        hj = mpc.get_current_hj(
            np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32),
            prev_free_ctrl_pts=np.array([[6.0, 0.0]], dtype=np.float32),
            obs_xy=np.array([[7.0, -0.9], [7.0, 0.9]], dtype=np.float32),
            obs_radius=np.array([1.0, 1.0], dtype=np.float32),
            obs_mask=np.array([True, True]),
        )

        self.assertTrue(np.all(np.isfinite(hj)))
        self.assertEqual(hj.shape, (2, 2))

    def test_solver_hessian_is_finite_near_obstacle_switch(self):
        config = FlatNMPCConfig(
            n_ctrl_points=3,
            spline_degree=2,
            horizon_time=0.5,
            cbf_enable=True,
            w_tracking=1.0,
            w_risk=3.0,
            risk_buffer=1.0,
            risk_temperature=0.3,
            d_clearance=0.5,
            max_obstacles=2,
            n_sqp_steps=4,
            n_admm_steps=20,
        )
        mpc = FlatSpaceNMPC(config)
        result = mpc.step(
            np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32),
            np.array([[6.0, 0.0]], dtype=np.float32),
            prev_free_ctrl_pts=np.array([[6.0, 0.0]], dtype=np.float32),
            obs_xy=np.array([[7.0, -0.9], [7.0, 0.9]], dtype=np.float32),
            obs_radius=np.array([1.0, 1.0], dtype=np.float32),
            obs_mask=np.array([True, True]),
        )

        self.assertEqual(result.hj.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(result.hj)))

    def test_batched_metric_hessian_is_finite_with_smoothmax_near_obstacle_switch(self):
        config = BatchedFlatNMPCConfig(
            batch_size=2,
            n_ctrl_points=3,
            spline_degree=2,
            horizon_time=0.5,
            cbf_enable=True,
            w_tracking=1.0,
            w_risk=3.0,
            risk_buffer=1.0,
            risk_temperature=0.3,
            d_clearance=0.5,
            max_obstacles=2,
        )
        mpc = BatchedFlatNMPC(config)
        hj = mpc.get_current_hj_batch(
            np.array(
                [
                    [0.0, 0.0, 0.0, 4.0],
                    [0.0, 0.0, 0.0, 4.0],
                ],
                dtype=np.float32,
            ),
            prev_free_ctrl_pts_batch=np.array(
                [
                    [[6.0, 0.0]],
                    [[6.0, 0.2]],
                ],
                dtype=np.float32,
            ),
            obs_xy_batch=np.array(
                [
                    [[7.0, -0.9], [7.0, 0.9]],
                    [[7.0, -1.0], [7.0, 1.0]],
                ],
                dtype=np.float32,
            ),
            obs_radius_batch=np.array(
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            ),
            obs_mask_batch=np.array(
                [
                    [True, True],
                    [True, True],
                ]
            ),
        )

        self.assertEqual(hj.shape, (2, 2, 2))
        self.assertTrue(np.all(np.isfinite(hj)))


if __name__ == "__main__":
    unittest.main()
