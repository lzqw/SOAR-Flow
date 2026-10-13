from __future__ import annotations

import gymnasium as gym
import numpy as np
from types import SimpleNamespace

from relax.env.drive.metasafe import (
    FixedSeedMetaSafeEnv,
    PostResetFrameWarmup,
    conservative_body_circle_cover,
)
from scripts.collect_metasafe_cbf_dataset import (
    FAST_SPEED_SAMPLING_PROFILE,
    FAST_V231_LATERAL_NOISE_SCALES,
    FAST_V231_MAX_COMMAND_SPEED,
    FAST_V231_WITHIN_BIN_DISTRIBUTIONS,
    FastV231SpeedBinScheduler,
    SPEED_SAMPLING_PROFILES,
    CollectionConfig,
    _activate_pre_intervention_teachers,
    _ordinary_recovery_behavior,
    _state_aligned_authority,
    parse_args,
    partition_episode_targets,
    sampled_nominal_action,
    target_speed_sampling_context,
)


def test_fast_v231_sampling_profile_probabilities_are_normalized():
    profile = SPEED_SAMPLING_PROFILES[FAST_SPEED_SAMPLING_PROFILE]
    assert profile == {
        "open": (0.00, 0.03, 0.25, 0.72),
        "near": (0.10, 0.25, 0.40, 0.25),
        "critical": (0.55, 0.45, 0.00, 0.00),
    }
    assert all(np.isclose(sum(probabilities), 1.0) for probabilities in profile.values())
    assert FAST_V231_LATERAL_NOISE_SCALES == (1.0, 1.0, 0.75, 0.50)
    assert FAST_V231_WITHIN_BIN_DISTRIBUTIONS == {
        "open": "beta_3_1_low_discrepancy_golden_ratio_v1",
        "near": "beta_3_1_low_discrepancy_golden_ratio_v1",
        "critical": "uniform",
    }


def test_fast_v231_biases_open_near_draws_within_each_selected_bin():
    config = CollectionConfig(
        output_dir="unused",
        action_semantics_version="target_speed_lateral_v1",
        speed_sampling_profile=FAST_SPEED_SAMPLING_PROFILE,
    )
    initial = np.asarray([1.0, 0.0], dtype=np.float32)

    def within_bin_mean(context):
        rng = np.random.default_rng(231)
        actions = np.asarray(
            [
                sampled_nominal_action(
                    config,
                    rng,
                    initial,
                    refresh=True,
                    speed_context=context,
                )
                for _ in range(20_000)
            ]
        )
        fractions = np.clip(0.5 * (actions[:, 0] + 1.0), 0.0, 1.0)
        return float(np.mean(np.mod(fractions * 4.0, 1.0)))

    assert 0.74 < within_bin_mean("open") < 0.76
    assert 0.74 < within_bin_mean("near") < 0.76
    assert 0.48 < within_bin_mean("critical") < 0.52


def test_fast_v231_within_bin_scheduler_prevents_low_draw_streaks():
    rng = np.random.default_rng(2312)
    scheduler = FastV231SpeedBinScheduler()
    fractions = np.asarray(
        [
            scheduler.within_bin_fraction("open", rng.uniform())
            for _ in range(256)
        ]
    )
    uniforms = np.power(fractions, 3.0)
    gaps = np.diff(np.concatenate(([0.0], np.sort(uniforms), [1.0])))

    assert 0.74 < float(np.mean(fractions)) < 0.76
    assert float(np.max(gaps)) < 0.02


def test_fast_v231_weighted_fair_scheduler_bounds_contextual_bin_drift():
    config = CollectionConfig(
        output_dir="unused",
        action_semantics_version="target_speed_lateral_v1",
        speed_sampling_profile=FAST_SPEED_SAMPLING_PROFILE,
    )
    rng = np.random.default_rng(2311)
    scheduler = FastV231SpeedBinScheduler()
    actions = np.asarray(
        [
            sampled_nominal_action(
                config,
                rng,
                np.asarray([1.0, 0.0], dtype=np.float32),
                refresh=True,
                speed_context="open",
                speed_bin_scheduler=scheduler,
            )
            for _ in range(1_000)
        ]
    )
    fractions = np.clip(0.5 * (actions[:, 0] + 1.0), 0.0, 1.0)
    bins = np.minimum((fractions * 4.0).astype(np.int32), 3)
    observed = np.bincount(bins, minlength=4) / bins.size
    np.testing.assert_allclose(
        observed,
        SPEED_SAMPLING_PROFILES[FAST_SPEED_SAMPLING_PROFILE]["open"],
        atol=0.002,
        rtol=0.0,
    )


def test_fast_v231_cli_binds_the_1200_step_pilot_contract():
    args = parse_args(
        [
            "--speed-sampling-profile",
            "fast_v231",
            "--action-semantics",
            "target_speed_lateral_v1",
        ]
    )
    assert args.max_episode_steps == 1200
    assert args.exploration_hold_steps == 15
    assert args.max_command_speed == FAST_V231_MAX_COMMAND_SPEED
    assert args.target_velocity == FAST_V231_MAX_COMMAND_SPEED
    assert args.minimum_high_speed_bin_fraction == 0.45
    assert args.maximum_high_speed_bin_fraction == 0.65
    assert args.minimum_mid_high_bin_fraction == 0.18
    assert args.minimum_low_speed_bins_fraction == 0.15


def test_balanced_v23_cli_preserves_the_original_speed_ceiling():
    args = parse_args([])
    assert args.max_command_speed == 4.8
    assert args.target_velocity == 4.8


def test_rectangle_circle_cover_contains_all_corners():
    center = np.asarray([3.0, -2.0], dtype=np.float32)
    heading = 0.63
    length = 4.8
    width = 2.0
    positions, radii = conservative_body_circle_cover(
        center,
        heading,
        length,
        width,
        circle_count=2,
    )
    forward = np.asarray([np.cos(heading), np.sin(heading)])
    lateral = np.asarray([-np.sin(heading), np.cos(heading)])
    corners = np.asarray(
        [
            center + x * 0.5 * length * forward + y * 0.5 * width * lateral
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
        ]
    )
    for corner in corners:
        assert np.any(np.linalg.norm(positions - corner, axis=1) <= radii + 1e-6)


def test_fixed_metasafe_defaults_to_one_official_seed():
    env = FixedSeedMetaSafeEnv({"use_render": False})
    try:
        assert env.config["metasafe_fixed_seed"] == 0
        assert env.config["start_seed"] == 0
        assert env.config["num_scenarios"] == 1
        assert not env.config["random_traffic"]
        assert not env.config["cost_to_reward"]
        assert env.config["accident_prob"] == 0.8
        assert env.config["traffic_density"] == 0.05
        assert env.config["map"] == 3
    finally:
        env.close()


def test_parallel_targets_cover_exact_episode_count():
    assert partition_episode_targets(300, 4) == [75, 75, 75, 75]
    assert partition_episode_targets(10, 3) == [4, 3, 3]
    assert partition_episode_targets(2, 8) == [1, 1]


def test_default_cli_requests_fixed_seed_parallel_300_episode_collection():
    args = parse_args([])
    assert args.episodes == 300
    assert args.behavior_policy == "protected_nominal"
    assert args.num_envs == 4
    assert args.scene_seed == 0
    assert args.require_zero_cost
    assert args.combine_shards
    assert args.post_reset_warmup_steps == 1
    assert args.state_aligned_authority_audit
    assert args.inverse_mpc_teacher
    assert args.proactive_dagger_teacher
    assert args.dagger_pre_steps == 40
    assert args.dagger_weight_decay == 0.98
    assert args.recreate_environment_each_attempt


def test_nominal_exploration_is_held_between_refresh_steps():
    config = CollectionConfig(
        output_dir="unused",
        longitudinal_exploration_std=0.02,
        lateral_exploration_std=0.04,
    )
    rng = np.random.default_rng(7)
    initial = np.asarray([1.0, 0.0], dtype=np.float32)
    sampled = sampled_nominal_action(config, rng, initial, refresh=True)
    held = sampled_nominal_action(config, rng, sampled, refresh=False)
    np.testing.assert_array_equal(sampled, held)
    assert config.minimum_longitudinal_action <= sampled[0] <= 1.0
    assert abs(float(sampled[1])) <= config.maximum_lateral_action


def test_target_speed_sampling_is_obstacle_contextual_and_top_bin_bounded():
    config = CollectionConfig(
        output_dir="unused",
        action_semantics_version="target_speed_lateral_v1",
    )
    initial = np.asarray([1.0, 0.0], dtype=np.float32)

    def fractions(context):
        rng = np.random.default_rng(91)
        actions = np.asarray(
            [
                sampled_nominal_action(
                    config,
                    rng,
                    initial,
                    refresh=True,
                    speed_context=context,
                )
                for _ in range(10_000)
            ]
        )
        return 0.5 * (actions[:, 0] + 1.0)

    open_fraction = fractions("open")
    near_fraction = fractions("near")
    critical_fraction = fractions("critical")
    assert np.mean(open_fraction) > np.mean(near_fraction) > np.mean(
        critical_fraction
    )
    assert np.mean(open_fraction >= 0.75) < 0.70
    assert target_speed_sampling_context(
        {"raw_hard_certificate": {"obstacle_clearance": 9.0}}
    ) == "open"
    assert target_speed_sampling_context(
        {"raw_hard_certificate": {"obstacle_clearance": 5.0}}
    ) == "near"
    assert target_speed_sampling_context(
        {"raw_hard_certificate": {"obstacle_clearance": 2.0}}
    ) == "critical"


def test_state_aligned_collector_derives_counterfactual_inverse_teacher():
    class Env:
        def __init__(self):
            self.independent_calls = 0
            self.inverse_calls = 0

        def certify_independent_actor_target(self, action, **kwargs):
            del action, kwargs
            self.independent_calls += 1
            return {
                "valid": False,
                "ordinary_solved": {"solver_finite": True},
                "trajectory_rmse": 0.2,
                # A changed CBF-on/off trajectory is active low-level
                # authority only when the ordinary CBF-rate certificate is
                # negative.  Positive-certificate branch differences are
                # audit-only under active_cbf_v2.
                "hard_cbf_margin": -0.5,
                "obstacle_clearance": 0.5,
                "road_clearance": 0.5,
                "kinematic_slack": 0.5,
            }

        def derive_inverse_mpc_teacher(self, solved):
            assert solved["result"].predicted_y[-1] == np.float32(0.2)
            self.inverse_calls += 1
            return {
                "valid": True,
                "action": np.asarray([0.45, -0.3], np.float32),
                "trajectory_rmse": 0.01,
            }

    env = Env()
    certificate = {
        "projection_used": False,
        "raw_hard_certificate": {"hard_cbf_margin": 10.0},
        "solved": {
            "action": np.asarray([0.9, 0.0], np.float32),
            "result": SimpleNamespace(
                predicted_y=np.asarray([0.0, 0.2], np.float32)
            ),
        },
    }
    audit = _state_aligned_authority(
        env,
        np.asarray([0.9, 0.0], np.float32),
        certificate,
        CollectionConfig(output_dir="unused"),
    )
    assert audit["query"]
    assert audit["ordinary_cbf_violation"]
    assert audit["low_level_authority"]
    assert audit["intervention"]
    assert not audit["diagnostic_only"]
    assert audit["teacher_valid"]
    assert audit["teacher_source"] == 2
    np.testing.assert_allclose(audit["teacher_action"], [0.45, -0.3])
    assert env.independent_calls == env.inverse_calls == 1

    certificate["raw_hard_certificate"]["hard_cbf_margin"] = 100.0
    far = _state_aligned_authority(
        env,
        np.asarray([0.9, 0.0], np.float32),
        certificate,
        CollectionConfig(output_dir="unused"),
    )
    assert not far["query"]
    assert not far["intervention"]
    assert env.independent_calls == env.inverse_calls == 1


def test_state_aligned_collector_builds_latent_proactive_teacher():
    safe_solved = {"result": object()}

    class Env:
        def __init__(self):
            self.calls = []

        def certify_independent_actor_target(self, action, **kwargs):
            action = np.asarray(action, np.float32)
            self.calls.append(("independent", action.copy()))
            if np.allclose(action, [0.9, 0.0]):
                return {
                    "valid": True,
                    "ordinary_solved": {"solver_finite": True},
                    "trajectory_rmse": 0.0,
                    "hard_cbf_margin": 20.0,
                    "obstacle_clearance": 1.0,
                    "road_clearance": 1.0,
                    "kinematic_slack": 1.0,
                }
            assert kwargs["reference_solved"] is safe_solved
            return {
                "valid": False,
                "ordinary_solved": {"solver_finite": True},
                "trajectory_rmse": 0.2,
            }

        def plan_safe_recovery_trajectory(self):
            self.calls.append(("recovery", None))
            return {
                "valid": True,
                "obstacle_ahead": True,
                "action": np.asarray([1.0, -0.8], np.float32),
                "safe_solved": safe_solved,
            }

        def derive_inverse_mpc_teacher(self, solved):
            assert solved is safe_solved
            self.calls.append(("inverse", None))
            return {
                "valid": True,
                "action": np.asarray([0.75, -0.6], np.float32),
                "trajectory_rmse": 0.03,
            }

    certificate = {
        "projection_used": False,
        "raw_hard_certificate": {"hard_cbf_margin": 20.0},
        "solved": safe_solved,
    }
    audit = _state_aligned_authority(
        Env(),
        np.asarray([0.9, 0.0], np.float32),
        certificate,
        CollectionConfig(output_dir="unused"),
    )
    assert audit["query"]
    assert not audit["intervention"]
    assert audit["teacher_valid"]
    assert audit["teacher_source"] == 3
    assert not audit["proactive_failed"]
    np.testing.assert_allclose(audit["teacher_action"], [0.75, -0.6])


def test_pre_intervention_activation_preserves_each_same_state_label():
    rows = {
        "teacher_valid": [True, True, True, True],
        "teacher_source": [3, 0, 3, 2],
        "teacher_weight": [np.float32(0.0)] * 4,
        "intervention": [False, False, False, True],
    }
    activated = _activate_pre_intervention_teachers(
        rows,
        boundary_index=3,
        horizon=3,
        decay=0.9,
        boundary_confidence=0.25,
    )
    assert activated == [0, 2]
    assert rows["teacher_weight"][0] == np.float32(0.25 * 0.9**3)
    assert rows["teacher_weight"][1] == np.float32(0.0)
    assert rows["teacher_weight"][2] == np.float32(0.25 * 0.9)


def test_ordinary_recovery_behavior_executes_independently_certified_action():
    execution = {"solved": {"action": np.asarray([1.0, -0.8], np.float32)}}
    safe_solved = {"result": object()}

    class Env:
        def plan_safe_recovery_trajectory(self):
            return {
                "valid": True,
                "obstacle_ahead": True,
                "action": np.asarray([1.0, -0.8], np.float32),
                "safe_solved": safe_solved,
            }

        def certify_independent_actor_target(self, action, **kwargs):
            np.testing.assert_allclose(action, [1.0, -0.8])
            assert kwargs["reference_solved"] is safe_solved
            return {
                "valid": True,
                "execution_certificate": execution,
                "trajectory_rmse": 0.02,
            }

        def certify_ordinary_target_action(self, action):
            raise AssertionError(f"unexpected fallback: {action}")

    selected = _ordinary_recovery_behavior(
        Env(),
        np.asarray([1.0, 0.0], np.float32),
        {
            "raw_hard_certificate": {
                "has_obstacle": True,
                "hard_cbf_margin": 20.0,
            }
        },
        CollectionConfig(output_dir="unused"),
    )
    assert selected["query"]
    assert selected["used"]
    assert not selected["failed"]
    assert selected["certificate"] is execution
    np.testing.assert_allclose(selected["action"], [1.0, -0.8])


class _CertificateEnv(gym.Env):
    observation_space = gym.spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self):
        self.executed_actions = []
        self.certificate_kinds = []

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.executed_actions.clear()
        self.certificate_kinds.clear()
        return np.zeros((3,), dtype=np.float32), {"scene": "fixed"}

    def certify_target_action(self, action):
        return {
            "action": np.asarray(action, dtype=np.float32).copy(),
            "kind": "protected",
        }

    def certify_unprotected_target_action(self, action):
        return {
            "action": np.asarray(action, dtype=np.float32).copy(),
            "kind": "unprotected",
        }

    def execute_certified_target(self, certificate):
        action = certificate["action"]
        self.executed_actions.append(action)
        self.certificate_kinds.append(certificate["kind"])
        observation = np.full((3,), len(self.executed_actions), dtype=np.float32)
        return observation, 0.25, False, False, {
            "cost": 0.0,
            "mgf": {"projection_used": False},
        }


def test_post_reset_frame_warmup_hides_reset_observation_from_policy():
    base = _CertificateEnv()
    env = PostResetFrameWarmup(base, warmup_steps=1)
    observation, info = env.reset(seed=0)
    np.testing.assert_array_equal(observation, np.ones((3,), dtype=np.float32))
    np.testing.assert_array_equal(base.executed_actions, [[1.0, 0.0]])
    assert base.certificate_kinds == ["unprotected"]
    assert info["scene"] == "fixed"
    assert info["post_reset_warmup_steps"] == 1
    assert info["post_reset_observation_linf_delta"] == 1.0

    certificate = env.certify_target_action(np.asarray([0.5, -0.25], np.float32))
    observation, *_ = env.execute_certified_target(certificate)
    np.testing.assert_array_equal(observation, np.full((3,), 2.0, np.float32))
    assert base.certificate_kinds == ["unprotected", "protected"]
