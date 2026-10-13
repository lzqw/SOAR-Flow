import unittest

import numpy as np

from relax.control.bspline_utils import bspline_eval, make_clamped_knots
from relax.control.drive_mgf import (
    DriveMGFGeometryConfig,
    compute_drive_mgf_field,
    target_reference_from_action,
)
from relax.control.flat_nmpc import FlatNMPCConfig, FlatSpaceNMPC


class TestDriveMGFGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = FlatNMPCConfig(
            n_ctrl_points=4,
            spline_degree=2,
            horizon_time=0.5,
            n_eval_points=5,
            target_velocity=3.0,
            w_tracking=1.0,
            w_risk=0.1,
            w_cbf=0.1,
            w_road=0.1,
            cbf_solve_weight_scale=0.1,
            road_solve_weight_scale=0.25,
            d_clearance=0.3,
            cbf_enable=True,
            max_obstacles=2,
            max_route_samples=6,
            n_sqp_steps=2,
            n_admm_steps=12,
        )
        cls.mpc = FlatSpaceNMPC(cls.config)
        cls.mpc.warmup()
        cls.knots = np.asarray(make_clamped_knots(4, 2), dtype=np.float32)
        cls.basis = np.asarray(
            bspline_eval(
                np.linspace(0.0, 1.0, 5, dtype=np.float32),
                np.eye(4, dtype=np.float32),
                cls.knots,
                2,
            ),
            dtype=np.float32,
        )

    def test_real_flat_mpc_returns_finite_target_space_field(self):
        x0 = np.array([0.0, 0.0, 0.0, 3.0], dtype=np.float32)
        c0 = np.array([0.0, 0.0], dtype=np.float32)
        delta_knot = max(float(self.knots[3] - self.knots[1]), 1e-6)
        c1 = np.array([x0[3] * delta_knot * 0.5 / 2.0, 0.0], dtype=np.float32)
        action = np.array([0.2, 0.0], dtype=np.float32)
        ref, ref_jac = target_reference_from_action(
            action,
            c1=c1,
            n_free=2,
            position_scale=5.0,
            lateral_scale=2.0,
        )
        obs_xy = np.array([[4.5, 1.8], [0.0, 0.0]], dtype=np.float32)
        obs_radius = np.array([0.3, 0.0], dtype=np.float32)
        obs_mask = np.array([True, False])
        route_xy = np.stack(
            [np.linspace(0.0, 6.0, 6), np.zeros(6)], axis=1
        ).astype(np.float32)
        route_normal = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (6, 1))
        route_half_width = np.full((6,), 2.0, dtype=np.float32)
        route_mask = np.ones((6,), dtype=bool)

        result = self.mpc.step(
            x0,
            ref,
            obs_xy=obs_xy,
            obs_radius=obs_radius,
            obs_mask=obs_mask,
            route_xy=route_xy,
            route_normal=route_normal,
            route_half_width=route_half_width,
            route_mask=route_mask,
        )
        field = compute_drive_mgf_field(
            action=action,
            ref_free_ctrl_pts=ref,
            ref_jacobian=ref_jac,
            optimized_free_ctrl_pts=result.opt_free_ctrl_pts,
            objective_hessian=result.hj,
            objective_value=result.cost_total,
            tracking_weight=self.config.w_tracking,
            c0=c0,
            c1=c1,
            basis_matrix=self.basis,
            obs_xy=obs_xy,
            obs_radius=obs_radius,
            obs_mask=obs_mask,
            d_clearance=self.config.d_clearance,
            route_xy=route_xy,
            route_normal=route_normal,
            route_half_width=route_half_width,
            route_mask=route_mask,
            geometry=DriveMGFGeometryConfig(
                margin_softmin_kappa=20.0,
                road_tightening=0.1,
                chance_beta=0.0,
            ),
            kinematic_margin=result.min_kappa_slack,
        )

        self.assertTrue(field.sensitivity_valid)
        self.assertTrue(np.isfinite(field.margin_delta))
        self.assertTrue(np.isfinite(field.min_kinematic_constraint))
        self.assertTrue(np.all(np.isfinite(field.grad_margin)))
        self.assertTrue(np.all(np.isfinite(field.grad_value)))
        self.assertEqual(field.grad_margin.shape, (2,))
        self.assertEqual(field.grad_value.shape, (2,))
        np.testing.assert_allclose(field.state_metric, field.state_metric.T, atol=1e-6)
        self.assertGreater(float(np.min(np.linalg.eigvalsh(field.state_metric))), 0.0)
        self.assertTrue(np.isfinite(field.state_metric_logdet))

    def test_velocity_objective_escapes_stationary_target_reference(self):
        config = FlatNMPCConfig(
            n_ctrl_points=5,
            spline_degree=2,
            horizon_time=1.0,
            n_eval_points=5,
            target_velocity=4.8,
            w_tracking=1.0,
            w_velocity=0.5,
            n_sqp_steps=5,
            n_admm_steps=50,
        )
        mpc = FlatSpaceNMPC(config)
        x0 = np.array([0.0, 0.0, 0.0, 0.1], dtype=np.float32)
        knots = np.asarray(make_clamped_knots(5, 2), dtype=np.float32)
        alpha = float(knots[3] - knots[1]) / 2.0
        c1 = np.array([x0[3] * alpha, 0.0], dtype=np.float32)
        stopping_reference = np.repeat(c1[None, :], 3, axis=0)

        result = mpc.step(
            x0,
            stopping_reference,
            prev_free_ctrl_pts=stopping_reference,
        )

        self.assertGreater(float(result.predicted_x[-1]), 0.5)
        self.assertGreater(result.throttle, 0.0)


if __name__ == "__main__":
    unittest.main()
