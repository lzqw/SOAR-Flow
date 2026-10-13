import unittest

import numpy as np

from relax.control.drive_mgf import (
    DriveMGFGeometryConfig,
    certificate_constraints_from_free,
    certificate_margin_from_free,
    normalized_target_to_xy,
    target_reference_from_action,
)
from relax.env.drive.lane_change_mpc import (
    SPEED_COUPLED_ACTION_SEMANTICS,
    decode_target_point_speed_action,
    encode_target_speed_in_action,
    target_speed_intervention_audit,
)


class TestDriveMGFTargetContract(unittest.TestCase):
    def test_speed_coupled_action_maps_progress_and_speed_monotonically(self):
        decoded = [
            decode_target_point_speed_action(
                np.asarray([longitudinal, 0.25], dtype=np.float32),
                lookahead_distance=10.0,
                lateral_scale=3.5,
                max_command_speed=4.8,
            )
            for longitudinal in (-1.0, 0.0, 1.0)
        ]
        np.testing.assert_allclose(
            [row["target_x"] for row in decoded],
            [0.0, 5.0, 10.0],
        )
        np.testing.assert_allclose(
            [row["target_speed"] for row in decoded],
            [0.0, 2.4, 4.8],
        )
        self.assertTrue(
            np.all(np.diff([row["target_x"] for row in decoded]) > 0.0)
        )
        self.assertTrue(
            np.all(np.diff([row["target_speed"] for row in decoded]) > 0.0)
        )

    def test_guardian_speed_change_is_encoded_in_executed_action(self):
        raw_action = np.asarray([1.0, -0.2], dtype=np.float32)
        executed_action = encode_target_speed_in_action(
            raw_action,
            target_speed=2.4,
            max_command_speed=4.8,
        )
        audit = target_speed_intervention_audit(
            raw_action,
            executed_action,
            raw_target_speed=4.8,
            executed_target_speed=2.4,
            action_semantics_version=SPEED_COUPLED_ACTION_SEMANTICS,
        )
        self.assertNotEqual(float(raw_action[0]), float(executed_action[0]))
        self.assertTrue(audit["actual_control_modified"])
        self.assertTrue(audit["speed_only_intervention"])
        self.assertAlmostEqual(audit["speed_command_delta"], -2.4)

    def test_hidden_speed_change_fails_closed(self):
        action = np.asarray([1.0, 0.0], dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "without changing"):
            target_speed_intervention_audit(
                action,
                action,
                raw_target_speed=4.8,
                executed_target_speed=2.4,
                action_semantics_version=SPEED_COUPLED_ACTION_SEMANTICS,
            )

    def test_subthreshold_encoded_speed_change_is_not_hidden(self):
        raw_action = np.asarray([0.5, 0.2], dtype=np.float32)
        executed_action = raw_action.copy()
        executed_action[0] = np.float32(raw_action[0] + 5e-5)
        raw_speed = 4.8 * (float(raw_action[0]) + 1.0) / 2.0
        executed_speed = 4.8 * (float(executed_action[0]) + 1.0) / 2.0

        self.assertLess(
            abs(float(executed_action[0] - raw_action[0])),
            1e-4,
        )
        audit = target_speed_intervention_audit(
            raw_action,
            executed_action,
            raw_target_speed=raw_speed,
            executed_target_speed=executed_speed,
            action_semantics_version=SPEED_COUPLED_ACTION_SEMANTICS,
            action_tolerance=1e-4,
        )

        self.assertFalse(audit["actual_control_modified"])
        self.assertTrue(audit["speed_only_intervention"])
        self.assertNotEqual(audit["speed_command_delta"], 0.0)

    def test_normalized_target_is_two_dimensional_and_bounded(self):
        target = normalized_target_to_xy(
            np.array([2.0, -2.0]),
            position_scale=6.0,
            lateral_scale=3.0,
        )
        np.testing.assert_allclose(target, np.array([6.0, -3.0]))

    def test_reference_ends_at_single_target_and_jacobian_matches_fd(self):
        action = np.array([0.2, -0.3], dtype=np.float64)
        c1 = np.array([0.5, 0.0], dtype=np.float64)
        ref, jacobian = target_reference_from_action(
            action,
            c1=c1,
            n_free=2,
            position_scale=5.0,
            lateral_scale=2.0,
        )
        np.testing.assert_allclose(ref[-1], np.array([3.0, -0.6]), atol=1e-6)
        self.assertEqual(jacobian.shape, (4, 2))

        eps = 1e-5
        numerical = np.zeros_like(jacobian)
        for index in range(2):
            plus = action.copy()
            minus = action.copy()
            plus[index] += eps
            minus[index] -= eps
            ref_plus, _ = target_reference_from_action(
                plus, c1=c1, n_free=2, position_scale=5.0, lateral_scale=2.0
            )
            ref_minus, _ = target_reference_from_action(
                minus, c1=c1, n_free=2, position_scale=5.0, lateral_scale=2.0
            )
            numerical[:, index] = ((ref_plus - ref_minus) / (2.0 * eps)).reshape(-1)
        np.testing.assert_allclose(jacobian, numerical, rtol=3e-3, atol=3e-3)

    def test_braking_target_does_not_fold_behind_velocity_control_point(self):
        c1 = np.array([0.4, 0.0], dtype=np.float64)
        ref, jacobian = target_reference_from_action(
            np.array([-1.0, 0.0]),
            c1=c1,
            n_free=2,
            position_scale=6.4,
            lateral_scale=3.7,
        )
        self.assertTrue(np.all(ref[:, 0] >= c1[0]))
        np.testing.assert_allclose(ref[:, 0], c1[0])
        np.testing.assert_allclose(jacobian[0::2, 0], 0.0)

    def test_certificate_margin_changes_sign_at_obstacle(self):
        geometry = DriveMGFGeometryConfig(
            margin_softmin_kappa=50.0,
            road_tightening=0.0,
            ego_half_width=0.1,
            ego_half_length=0.1,
        )
        common = dict(
            c0=np.array([0.0, 0.0]),
            c1=np.array([1.0, 0.0]),
            basis_matrix=np.eye(4),
            obs_xy=np.array([[3.0, 0.0]]),
            obs_radius=np.array([0.25]),
            obs_mask=np.array([True]),
            d_clearance=0.25,
            route_xy=np.zeros((1, 2)),
            route_normal=np.zeros((1, 2)),
            route_half_width=np.zeros((1,)),
            route_mask=np.array([False]),
            geometry=geometry,
        )
        collision = certificate_margin_from_free(
            np.array([[2.0, 0.0], [3.0, 0.0]]), **common
        )
        safe = certificate_margin_from_free(
            np.array([[2.0, 2.0], [3.0, 2.0]]), **common
        )
        self.assertLess(collision, 0.0)
        self.assertGreater(safe, 0.0)

    def test_certificate_predicts_obstacle_motion_over_horizon(self):
        geometry = DriveMGFGeometryConfig(
            margin_softmin_kappa=100.0,
            road_tightening=0.0,
            ego_half_width=0.1,
            ego_half_length=0.1,
        )
        common = dict(
            c0=np.array([0.0, 0.0]),
            c1=np.array([1.0, 0.0]),
            basis_matrix=np.eye(4),
            obs_radius=np.array([0.2]),
            obs_mask=np.array([True]),
            d_clearance=0.1,
            route_xy=np.zeros((1, 2)),
            route_normal=np.zeros((1, 2)),
            route_half_width=np.zeros((1,)),
            route_mask=np.array([False]),
            geometry=geometry,
        )
        free = np.array([[2.0, 0.0], [3.0, 0.0]])
        static_margin = certificate_margin_from_free(
            free,
            obs_xy=np.array([[4.0, 0.0]]),
            **common,
        )
        approaching_margin = certificate_margin_from_free(
            free,
            # The obstacle moves into the path endpoint over the horizon.
            obs_xy=np.array([[4.0, 0.0, -1.0, 0.0]]),
            **common,
        )
        self.assertGreater(static_margin, 0.0)
        self.assertLess(approaching_margin, 0.0)

    def test_route_certificate_does_not_extend_past_segment_endpoint(self):
        geometry = DriveMGFGeometryConfig(margin_softmin_kappa=50.0, road_tightening=0.0)
        common = dict(
            c0=np.array([0.0, 0.0]),
            c1=np.array([0.5, 0.0]),
            basis_matrix=np.eye(4),
            obs_xy=np.zeros((1, 2)),
            obs_radius=np.zeros((1,)),
            obs_mask=np.array([False]),
            d_clearance=0.0,
            route_xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
            route_normal=np.array([[0.0, 1.0], [0.0, 1.0]]),
            route_half_width=np.array([1.0, 1.0]),
            route_mask=np.array([True, True]),
            geometry=geometry,
        )
        inside = certificate_margin_from_free(
            np.array([[0.75, 0.0], [1.0, 0.0]]), **common
        )
        past_endpoint = certificate_margin_from_free(
            np.array([[2.25, 0.0], [2.5, 0.0]]), **common
        )
        self.assertGreater(inside, 0.0)
        self.assertLess(past_endpoint, 0.0)

    def test_route_certificate_matches_mpc_oriented_body_road_constraint(self):
        constraints = certificate_constraints_from_free(
            np.array([[0.5, 0.8], [1.0, 0.8]]),
            c0=np.array([0.0, 0.0]),
            c1=np.array([0.5, 0.5]),
            basis_matrix=np.eye(4),
            obs_xy=np.zeros((1, 2)),
            obs_radius=np.zeros((1,)),
            obs_mask=np.array([False]),
            d_clearance=0.0,
            route_xy=np.array([[0.0, 0.0], [4.0, 0.0]]),
            route_normal=np.array([[0.0, 1.0], [0.0, 1.0]]),
            route_half_width=np.array([2.0, 2.0]),
            route_mask=np.array([True, True]),
            road_tightening=1.0,
            ego_half_width=0.926,
            # Deliberately exaggerate length so a yawed body is unmistakably
            # rejected while the final road-aligned segment retains the
            # configured 1.0 m straight-line tightening.
            ego_half_length=20.0,
        )
        residual = 1.0 - 0.926
        np.testing.assert_allclose(
            constraints,
            np.array(
                [
                    1.5 - (20.0 + 0.926) / np.sqrt(2.0) - residual,
                    1.2 - 20.0 - residual,
                    1.2 - 0.926 - residual,
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
