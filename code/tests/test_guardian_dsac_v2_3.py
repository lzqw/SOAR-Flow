from __future__ import annotations

import inspect
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from relax.algorithm.guardian_dsac_v2_3 import (
    GuardianDSACV23,
    GuardianDSACV23Config,
    calql_mean_conservative_gap,
    proxy_actor_objective,
    proxy_raw_support_weights,
    weighted_action_proximal,
)
from relax.utils.action_semantics import encode_scene_signature
from relax.env.drive.lane_change_mpc import (
    FastMPCControlWrapper,
    decode_target_speed_lateral_action,
    encode_target_speed_lateral_action,
    target_speed_intervention_audit,
)
from relax.network.flow_policy import create_one_step_flow_policy
from relax.network.safe_sac_jax import create_guardian_dsac_net
from relax.offline.metasafe_guardian_dataset import (
    GuardianDSACMetaSafeDataset,
    REQUIRED_KEYS,
    discounted_guardian_dependence_targets,
)
from relax.utils.distributional_q import (
    OfflineSupportThresholds,
    clip_sample_target,
    gaussian_wasserstein_disagreement,
    hdsac_gaussian_loss,
    sample_gaussian_return,
    select_conservative_gaussian,
    soft_reference_weight,
)
from relax.utils.experience import GuardianDSACExperience
from relax.utils.guardian_deployment_gate import deployment_gate
from relax.utils.guardian_learning_gate import (
    apply_multi_seed_hybrid_learning_gate,
)
from scripts.train_guardian_dsac_metasafe import (
    EpisodeSeedCycle,
    _grouped_actor_diagnostics,
    _step_dsac_experience,
    _store_complete_online_episode,
    parse_args as parse_v231_args,
)


def _algorithm(
    seed: int = 0,
    config: GuardianDSACV23Config | None = None,
) -> GuardianDSACV23:
    network, params = create_guardian_dsac_net(
        jax.random.PRNGKey(seed),
        critic_obs_dim=6,
        actor_obs_dim=4,
        act_dim=2,
        hidden_sizes=(16, 16),
    )
    return GuardianDSACV23(
        network,
        params,
        config=config or GuardianDSACV23Config(),
        scene_seeds=(0, 1, 2),
        scene_signatures={0: "zero", 1: "one", 2: "two"},
        dataset_sha256="test-dataset",
    )


def _batch(size: int = 8):
    batch = GuardianDSACExperience.create_example(6, 2, batch_size=size)
    return batch._replace(
        teacher_valid=np.ones(size, dtype=bool),
        teacher_weight=np.ones(size, dtype=np.float32),
        actor_controlled=np.ones(size, dtype=bool),
        guardian_dependence_target_valid=np.ones(size, dtype=bool),
        behavior_return_valid=np.ones(size, dtype=bool),
        action_semantics_version=np.full(size, 3, dtype=np.int8),
        reachable_speed_lower=np.ones(size, dtype=np.float32),
        reachable_speed_upper=np.full(size, 4.8, dtype=np.float32),
        raw_target_speed=np.full(size, 2.0, dtype=np.float32),
        executed_target_speed=np.full(size, 2.0, dtype=np.float32),
        actual_vehicle_speed=np.full(size, 2.0, dtype=np.float32),
    )


def _tree_equal(left, right) -> bool:
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
        )
    )


def _safe_eval(*, result_return=0.0, intervention=0.1):
    return {
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "road_departure_rate": 0.0,
        "total_environment_cost": 0.0,
        "solver_finite_rate": 1.0,
        "min_physical_body_sat_margin": 0.1,
        "mean_return": result_return,
        "actual_intervention_rate": intervention,
    }


def _minimal_v231_dataset_arrays(
    *, episode_step_limit: int = 1200, speed_sampling_profile: str = "fast_v231"
):
    count = 3
    arrays = {}
    for name in REQUIRED_KEYS:
        if name in ("obs", "next_obs"):
            arrays[name] = np.zeros((count, 2), np.float32)
        elif name in ("raw_action", "action", "teacher_action"):
            arrays[name] = np.zeros((count, 2), np.float32)
        elif name in (
            "done",
            "projection_used",
            "projection_failed",
            "solver_finite",
            "critic_action_compatible",
            "bc_valid",
        ):
            arrays[name] = np.ones(count, dtype=bool)
        elif name == "episode_id":
            arrays[name] = np.arange(count, dtype=np.int32)
        elif name == "step_index":
            arrays[name] = np.ones(count, dtype=np.int32)
        else:
            arrays[name] = np.zeros(count, np.float32)
    arrays.update(
        action_semantics_version=np.full(count, "target_speed_lateral_v1"),
        min_command_speed=np.full(count, 1.0, np.float32),
        max_command_speed=np.full(count, 4.8, np.float32),
        speed_mapping=np.full(count, "state_reachable_speed_linear_v1"),
        mpc_horizon=np.full(count, 2.0, np.float32),
        scene_seed=np.zeros(count, np.int32),
        scene_signature=np.full(count, "zero"),
        reachable_speed_lower=np.full(count, 1.0, np.float32),
        reachable_speed_upper=np.full(count, 4.8, np.float32),
        raw_target_speed=np.full(count, 2.0, np.float32),
        executed_target_speed=np.full(count, 2.0, np.float32),
        actual_vehicle_speed=np.full(count, 2.0, np.float32),
        speed_command_delta=np.zeros(count, np.float32),
        emergency_stop=np.zeros(count, bool),
        episode_step_limit=np.full(count, episode_step_limit, np.int32),
        speed_sampling_profile=np.full(count, speed_sampling_profile),
    )
    return arrays


def test_distributional_q_head_has_mean_and_positive_std():
    algorithm = _algorithm()
    mean_1, std_1, mean_2, std_2 = algorithm.task_q_distribution(
        np.zeros((3, 6), np.float32), np.zeros((3, 2), np.float32)
    )
    assert mean_1.shape == std_1.shape == mean_2.shape == std_2.shape == (3,)
    assert np.all(std_1 > 0.0) and np.all(std_2 > 0.0)


def test_fixed_noise_gaussian_return_sample_is_reproducible():
    mean = jnp.array([1.0, 2.0])
    std = jnp.array([0.5, 0.25])
    noise = jnp.array([2.0, -4.0])
    first = sample_gaussian_return(None, mean, std, noise=noise)
    second = sample_gaussian_return(None, mean, std, noise=noise)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first, [2.0, 1.25])


def test_target_critic_selects_smaller_mean_branch():
    mean, std, branch = select_conservative_gaussian(
        jnp.array([1.0, -2.0]),
        jnp.array([0.1, 0.2]),
        jnp.array([0.0, -1.0]),
        jnp.array([0.3, 0.4]),
    )
    np.testing.assert_allclose(mean, [0.0, -2.0])
    np.testing.assert_allclose(std, [0.3, 0.2])
    np.testing.assert_array_equal(branch, [1, 0])


def test_three_ema_sigma_target_clipping():
    clipped = clip_sample_target(
        jnp.array([1.0, 1.0]),
        jnp.array([10.0, -10.0]),
        jnp.asarray(0.5),
        multiplier=3.0,
    )
    np.testing.assert_allclose(clipped, [2.5, -0.5])


def test_hdsac_loss_has_finite_jax_gradients():
    def loss(mean, raw_std):
        return hdsac_gaussian_loss(
            mean,
            jax.nn.softplus(raw_std) + 1e-3,
            jnp.array([1.0, -1.0]),
            jnp.array([1.5, -2.0]),
            jnp.asarray(0.7),
        )

    gradients = jax.grad(loss, argnums=(0, 1))(
        jnp.zeros(2), jnp.zeros(2)
    )
    assert all(np.all(np.isfinite(np.asarray(value))) for value in gradients)


def test_calql_conservative_term_has_no_direct_sigma_gradient():
    def loss(mean, std):
        del std
        candidates = jnp.stack((2.0 * mean + 1.0, mean - 1.0), axis=-1)
        return calql_mean_conservative_gap(mean, candidates, temperature=1.0)

    mean_grad, std_grad = jax.grad(loss, argnums=(0, 1))(
        jnp.zeros(4), jnp.ones(4)
    )
    assert np.any(np.abs(np.asarray(mean_grad)) > 0.0)
    np.testing.assert_array_equal(std_grad, np.zeros(4))


def test_gaussian_wasserstein_disagreement_is_correct():
    value = gaussian_wasserstein_disagreement(
        jnp.asarray(1.0),
        jnp.asarray(2.0),
        jnp.asarray(4.0),
        jnp.asarray(6.0),
    )
    assert float(value) == pytest.approx(5.0)


def test_familiar_reference_weight_exceeds_artificial_novel_weight():
    thresholds = OfflineSupportThresholds(1.0, 1.0, 0.2, 0.2)
    familiar = soft_reference_weight(0.1, 0.1, thresholds)
    novel = soft_reference_weight(3.0, 3.0, thresholds)
    assert float(familiar) > float(novel)


def test_scene_seed_does_not_enter_reference_weight():
    thresholds = OfflineSupportThresholds(1.0, 1.0, 0.2, 0.2)
    seed0 = soft_reference_weight(0.5, 0.5, thresholds)
    seed99 = soft_reference_weight(0.5, 0.5, thresholds)
    assert float(seed0) == float(seed99)
    assert "scene_seed" not in inspect.signature(soft_reference_weight).parameters


def test_complete_episode_proxy_targets_remain_bounded():
    raw, corrected, valid, corrected_valid = discounted_guardian_dependence_targets(
        np.array([False, True, False]),
        np.array([True, False, False]),
        np.zeros(3, dtype=bool),
        np.array([True, True, False]),
        np.array([False, False, True]),
        np.zeros(3, dtype=np.int32),
        gamma=0.97,
    )
    assert np.all((0.0 <= raw) & (raw <= 1.0))
    assert np.all((0.0 <= corrected) & (corrected <= 1.0))
    assert np.all(valid) and np.count_nonzero(corrected_valid) == 2


def test_actor_update_exposes_only_supported_sac_absolute_p_and_proximal():
    algorithm = _algorithm()
    assert algorithm.get_deterministic_action(
        np.zeros(6, dtype=np.float32)
    ).flags.writeable
    batch = jax.device_put(_batch())
    algorithm.mark_offline_reference(batch)
    metrics = algorithm.online_joint_update(jax.random.PRNGKey(9), batch)
    assert {
        "supported_sac_loss",
        "absolute_proxy_loss",
        "proxy_actor_objective_loss",
        "proxy_actor_softplus_logit",
        "proximal_loss",
        "weighted_sac_loss",
        "weighted_proxy_loss",
        "weighted_proximal_loss",
        "proximal_speed_drift",
        "proximal_lateral_drift",
        "proximal_coefficient",
        "proximal_step",
        "p_policy_mean",
        "p_policy_by_speed_bin_0",
        "p_policy_by_speed_bin_1",
        "p_policy_by_speed_bin_2",
        "p_policy_by_speed_bin_3",
    }.issubset(metrics)
    assert all(
        np.isfinite(metrics[name])
        for name in (
            "weighted_sac_loss",
            "weighted_proxy_loss",
            "weighted_proximal_loss",
        )
    )
    grouped = _grouped_actor_diagnostics(metrics)
    assert len(grouped["p_policy_by_speed_bin"]) == 4
    assert all(np.isfinite(grouped["p_policy_by_speed_bin"]))
    forbidden = ("awac", "teacher", "cvar", "uncertainty", "candidate")
    assert not any(token in name for name in metrics for token in forbidden)


def test_v231_default_proxy_and_proximal_weights():
    config = GuardianDSACV23Config()
    assert config.proxy_raw_pair_weight == 1.0
    assert config.proxy_pair_gap_weight == 0.0
    assert config.proxy_actor_weight == 0.5
    assert config.proxy_actor_objective == "probability"
    assert config.proximal_weight_initial == 1.0
    assert config.proximal_weight_min == 0.05
    assert config.proximal_decay_steps == 20_000
    assert config.proximal_speed_weight == 0.5
    assert config.proximal_lateral_weight == 1.0


def test_proxy_raw_pair_weight_emphasizes_only_valid_pairs():
    weights = proxy_raw_support_weights(
        jnp.asarray([1.0, 1.0, 0.0, 1.0]),
        jnp.asarray([0.0, 1.0, 1.0, 0.0]),
        pair_weight=8.0,
    )
    np.testing.assert_allclose(weights, [1.0, 8.0, 0.0, 1.0])


def test_softplus_logit_proxy_actor_objective_is_non_saturating():
    logits = jnp.asarray([-5.0, 0.0, 5.0], dtype=jnp.float32)
    probability = proxy_actor_objective(logits, objective="probability")
    softplus_logit = proxy_actor_objective(
        logits, objective="softplus_logit"
    )
    np.testing.assert_allclose(probability, jax.nn.sigmoid(logits))
    np.testing.assert_allclose(softplus_logit, jax.nn.softplus(logits))
    probability_gradient = jax.grad(
        lambda value: proxy_actor_objective(
            value, objective="probability"
        )
    )(jnp.asarray(5.0, dtype=jnp.float32))
    softplus_gradient = jax.grad(
        lambda value: proxy_actor_objective(
            value, objective="softplus_logit"
        )
    )(jnp.asarray(5.0, dtype=jnp.float32))
    assert float(softplus_gradient) > 100.0 * float(probability_gradient)


def test_proxy_pair_gap_loss_is_finite_and_reports_pair_target():
    algorithm = _algorithm(
        config=GuardianDSACV23Config(
            proxy_raw_pair_weight=8.0,
            proxy_pair_gap_weight=8.0,
        )
    )
    size = 8
    batch = _batch(size)._replace(
        raw_action=np.tile(
            np.asarray([0.8, 0.5], dtype=np.float32), (size, 1)
        ),
        proxy_corrected_action=np.tile(
            np.asarray([0.1, -0.2], dtype=np.float32), (size, 1)
        ),
        guardian_dependence_target=np.full(size, 0.9, dtype=np.float32),
        guardian_dependence_corrected_target=np.full(
            size, 0.2, dtype=np.float32
        ),
        guardian_dependence_corrected_valid=np.ones(size, dtype=bool),
    )
    metrics = algorithm.proxy_update(jax.device_put(batch))
    assert {
        "proxy_pair_gap_loss",
        "proxy_pair_gap_mae",
        "proxy_predicted_pair_gap",
        "proxy_target_pair_gap",
        "proxy_pair_gap_weight",
    }.issubset(metrics)
    assert all(np.isfinite(metrics[name]) for name in metrics)
    assert metrics["proxy_target_pair_gap"] == pytest.approx(0.7)
    assert metrics["proxy_pair_gap_weight"] == pytest.approx(8.0)


def test_v231_proxy_raw_pair_weight_is_explicitly_opt_in():
    args = parse_v231_args(
        [
            "--no-require-gpu",
            "--proxy-raw-pair-weight",
            "8",
            "--proxy-pair-gap-weight",
            "8",
            "--proxy-actor-objective",
            "softplus_logit",
        ]
    )
    assert args.proxy_raw_pair_weight == 8.0
    assert args.proxy_pair_gap_weight == 8.0
    assert args.proxy_actor_objective == "softplus_logit"


def test_speed_axis_has_lower_proximal_weight_than_lateral_axis():
    speed_only, speed_drift, speed_lateral_drift = weighted_action_proximal(
        jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
        speed_weight=0.5,
        lateral_weight=1.0,
    )
    lateral_only, lateral_speed_drift, lateral_drift = weighted_action_proximal(
        jnp.asarray([[0.0, 1.0]], dtype=jnp.float32),
        speed_weight=0.5,
        lateral_weight=1.0,
    )
    assert float(speed_only) == pytest.approx(0.5)
    assert float(lateral_only) == pytest.approx(1.0)
    assert float(speed_drift) == float(lateral_drift) == 1.0
    assert float(speed_lateral_drift) == float(lateral_speed_drift) == 0.0


def test_v231_main_experiment_disables_soft_reference_by_default():
    args = parse_v231_args(["--no-require-gpu"])
    assert args.reference_weight_initial == 0.0
    assert args.reference_weight_final == 0.0
    assert args.max_command_speed == 10.75
    assert args.target_velocity == 10.75
    assert not args.online_inverse_teacher


def test_v231_online_inverse_teacher_is_explicitly_opt_in():
    args = parse_v231_args(
        ["--no-require-gpu", "--online-inverse-teacher"]
    )
    assert args.online_inverse_teacher


def test_online_inverse_teacher_creates_seed_specific_proxy_pair():
    raw_action = np.array([0.6, 0.2], dtype=np.float32)
    teacher_action = np.array([0.1, -0.3], dtype=np.float32)
    hard = {
        "hard_cbf_margin": -1.0,
        "obstacle_clearance": 1.0,
        "road_clearance": 1.0,
        "kinematic_slack": 1.0,
    }
    info = {
        "mgf": {
            "solver_finite": True,
            "projection_failed": False,
            "hard_safety": {**hard, "hard_cbf_margin": 1.0},
            "raw_hard_safety": hard,
        },
        "executed_target": raw_action,
        "reachable_speed_lower": 1.0,
        "reachable_speed_upper": 10.75,
        "actual_vehicle_speed": 4.0,
        "raw_target_speed": 6.0,
        "executed_target_speed": 6.0,
        "mpc_control": {"physical_body_sat_margin": 1.0},
    }
    transition = _step_dsac_experience(
        np.zeros(3, dtype=np.float32),
        raw_action,
        1.0,
        False,
        True,
        np.ones(3, dtype=np.float32),
        {"raw_hard_certificate": hard},
        info,
        soft_pair_weight=0.25,
        authority_audit={
            "actual_low_level_modified": True,
            "audit_authority": True,
            "teacher_valid": True,
            "teacher_action": teacher_action,
            "ordinary_solver_finite": True,
            "ordinary_hard_certificate": hard,
        },
        scene_seed=1,
        scene_signature="one",
        action_semantics_version="target_speed_lateral_v1",
    )
    assert bool(transition.teacher_valid)
    assert bool(transition.proxy_pair_valid)
    assert int(transition.scene_seed) == 1
    np.testing.assert_allclose(transition.proxy_corrected_action, teacher_action)


def test_online_episode_seed_cycle_is_11110():
    cycle = EpisodeSeedCycle()
    observed = []
    for _ in range(10):
        observed.append(cycle.current_seed)
        cycle.complete_episode()
    assert observed == [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]


def test_online_seed_does_not_change_inside_an_episode():
    cycle = EpisodeSeedCycle()
    initial = cycle.current_seed
    assert [cycle.current_seed for _ in range(1200)] == [initial] * 1200
    assert cycle.complete_episode() == 1


def test_complete_episode_replay_preserves_seed_and_signature():
    signature = "seed0-signature"
    first = GuardianDSACExperience.create_example(3, 2)._replace(
        done=np.bool_(False),
        reward=np.float32(1.0),
        scene_seed=np.int32(0),
        scene_signature=encode_scene_signature(signature),
        proposal_evaluated=np.bool_(True),
        intervention_soft=np.bool_(True),
        raw_action=np.asarray([1.0, 0.0], dtype=np.float32),
        proxy_corrected_action=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    final = first._replace(
        done=np.bool_(True),
        reward=np.float32(2.0),
        intervention_soft=np.bool_(False),
        intervention_hard=np.bool_(True),
    )

    class Replay:
        def __init__(self):
            self.items = []

        def add(self, item):
            self.items.append(item)

    replay = Replay()
    stored = _store_complete_online_episode(
        replay,
        [first, final],
        scene_seed=0,
        scene_signature=signature,
        gamma=0.99,
        proxy_discount=0.97,
    )
    assert len(replay.items) == len(stored) == 2
    assert all(left is right for left, right in zip(replay.items, stored))
    assert all(int(item.scene_seed) == 0 for item in replay.items)
    assert all(
        np.array_equal(item.scene_signature, encode_scene_signature(signature))
        for item in replay.items
    )
    np.testing.assert_allclose(
        [float(item.guardian_dependence_target) for item in replay.items],
        [0.5 + 0.5 * 0.97, 1.0],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        [float(item.guardian_dependence_corrected_target) for item in replay.items],
        [0.97, 0.0],
        atol=1e-7,
    )
    with pytest.raises(ValueError, match="complete episodes"):
        _store_complete_online_episode(
            Replay(),
            [first],
            scene_seed=0,
            scene_signature=signature,
            gamma=0.99,
            proxy_discount=0.97,
        )
    wrong = final._replace(scene_seed=np.int32(1))
    with pytest.raises(RuntimeError, match="seed binding mismatch"):
        _store_complete_online_episode(
            Replay(),
            [wrong],
            scene_seed=0,
            scene_signature=signature,
            gamma=0.99,
            proxy_discount=0.97,
        )


def test_multi_seed_hybrid_gate_requires_both_seeds_to_pass():
    class AlgorithmProbe:
        def __init__(self):
            self.calls = []

        def promote_hybrid_safe(self):
            self.calls.append("promote")
            return {"rollback": "none", "snapshot": "hybrid_safe_snapshot"}

        def reject_hybrid_safety(self):
            self.calls.append("actor_only")
            return {"rollback": "actor_only", "snapshot": "hybrid_safe_snapshot"}

        def reject_numerical_disaster(self):
            self.calls.append("full")
            return {"rollback": "hybrid_safe_full_snapshot"}

    probe = AlgorithmProbe()
    unsafe = _safe_eval()
    unsafe["collision_rate"] = 1.0
    rejected = apply_multi_seed_hybrid_learning_gate(
        probe, {0: _safe_eval(), 1: unsafe}
    )
    assert not rejected["passed"]
    assert rejected["required_seeds"] == [0, 1]
    assert rejected["per_seed_failures"]["0"] == []
    assert rejected["per_seed_failures"]["1"] == ["collision"]
    assert rejected["rollback"] == "actor_only"
    assert probe.calls == ["actor_only"]

    missing = apply_multi_seed_hybrid_learning_gate(probe, {0: _safe_eval()})
    assert not missing["passed"]
    assert missing["per_seed_failures"]["1"] == ["missing_evaluation"]
    assert probe.calls == ["actor_only", "actor_only"]

    accepted = apply_multi_seed_hybrid_learning_gate(
        probe, {"0": _safe_eval(), "1": _safe_eval()}
    )
    assert accepted["passed"]
    assert accepted["per_seed_failures"] == {"0": [], "1": []}
    assert accepted["rollback"] == "none"
    assert probe.calls == ["actor_only", "actor_only", "promote"]

    numerical = apply_multi_seed_hybrid_learning_gate(
        probe,
        {0: _safe_eval(), 1: _safe_eval()},
        numerical_reasons=("non_finite_q",),
    )
    assert not numerical["passed"]
    assert numerical["rollback"] == "hybrid_safe_full_snapshot"
    assert probe.calls == ["actor_only", "actor_only", "promote", "full"]


def test_hybrid_gate_pass_does_not_reset_proximal_step():
    algorithm = _algorithm()
    algorithm.state = algorithm.state._replace(
        proximal_step=jnp.asarray(17, dtype=jnp.int32)
    )
    gate = apply_multi_seed_hybrid_learning_gate(
        algorithm, {0: _safe_eval(), 1: _safe_eval()}
    )
    assert gate["passed"]
    assert int(algorithm.state.proximal_step) == 17
    assert int(algorithm.hybrid_safe_snapshot.proximal_step) == 17


def test_ordinary_failure_does_not_rollback_learner():
    algorithm = _algorithm()
    before = algorithm.state
    disposition = algorithm.record_ordinary_failure()
    assert disposition["rollback"] == "none"
    assert _tree_equal(before, algorithm.state)


def test_hybrid_failure_restores_only_actor_and_resets_actor_optimizer():
    algorithm = _algorithm()
    algorithm.promote_hybrid_safe()
    safe_policy = algorithm.hybrid_safe_snapshot.params.policy
    changed_q = jax.tree_util.tree_map(lambda value: value + 1.0, algorithm.state.params.q1)
    changed_policy = jax.tree_util.tree_map(
        lambda value: value + 1.0, algorithm.state.params.policy
    )
    algorithm.state = algorithm.state._replace(
        params=algorithm.state.params._replace(q1=changed_q, policy=changed_policy),
        proximal_step=jnp.asarray(23, dtype=jnp.int32),
    )
    algorithm.reject_hybrid_safety()
    assert _tree_equal(algorithm.state.params.q1, changed_q)
    assert _tree_equal(algorithm.state.params.policy, safe_policy)
    assert int(algorithm.state.proximal_step) == 0


def test_numerical_disaster_restores_complete_safe_snapshot():
    algorithm = _algorithm()
    algorithm.promote_hybrid_safe()
    safe = algorithm.hybrid_safe_snapshot
    algorithm.state = jax.tree_util.tree_map(lambda value: value + 1, algorithm.state)
    algorithm.reject_numerical_disaster()
    assert _tree_equal(algorithm.state, safe)
    assert int(algorithm.state.proximal_step) == 0


def test_first_ordinary_safe_policy_needs_no_performance_threshold():
    candidate = {
        seed: {
            "hybrid": _safe_eval(result_return=-100.0, intervention=0.9),
            "ordinary": _safe_eval(result_return=-100.0, intervention=0.0),
        }
        for seed in (0, 1)
    }
    result = deployment_gate(candidate_by_seed=candidate, incumbent_by_seed=None)
    assert result["promote"]
    assert result["decision"] == "first_ordinary_safe_checkpoint"


def test_later_deployment_promotion_uses_seed_average_not_each_seed():
    incumbent = {
        0: {"hybrid": _safe_eval(intervention=0.10), "ordinary": _safe_eval(result_return=10.0)},
        1: {"hybrid": _safe_eval(intervention=0.10), "ordinary": _safe_eval(result_return=10.0)},
    }
    candidate = {
        0: {"hybrid": _safe_eval(intervention=0.11), "ordinary": _safe_eval(result_return=8.0)},
        1: {"hybrid": _safe_eval(intervention=0.08), "ordinary": _safe_eval(result_return=13.2)},
    }
    result = deployment_gate(
        candidate_by_seed=candidate,
        incumbent_by_seed=incumbent,
    )
    assert result["promote"]
    assert result["candidate_average"]["average_ordinary_return"] == pytest.approx(10.6)


def test_target_speed_action_encode_decode_is_reversible():
    action = np.array([0.3, -0.4], dtype=np.float32)
    decoded = decode_target_speed_lateral_action(
        action,
        current_speed=2.0,
        horizon_time=2.0,
        acceleration_min=-5.0,
        acceleration_max=3.0,
        command_speed_min=1.0,
        command_speed_max=4.8,
        lateral_scale=3.5,
        target_x_max=10.0,
    )
    encoded = encode_target_speed_lateral_action(
        action,
        target_speed=decoded["target_speed"],
        reachable_speed_lower=decoded["reachable_speed_lower"],
        reachable_speed_upper=decoded["reachable_speed_upper"],
    )
    np.testing.assert_allclose(encoded, action, atol=2e-6)


def test_hidden_target_speed_modification_fails_closed():
    with pytest.raises(RuntimeError, match="without changing"):
        target_speed_intervention_audit(
            np.array([0.2, 0.0], np.float32),
            np.array([0.2, 0.0], np.float32),
            raw_target_speed=2.0,
            executed_target_speed=1.5,
            action_semantics_version="target_speed_lateral_v1",
        )


def test_v23_terminal_projection_failure_is_not_executed():
    wrapper = type(
        "ProjectionFailureProbe",
        (),
        {"_step_count": 7, "target_speed_lateral_action": True},
    )()
    certificate = {
        "step_count": 7,
        "raw_action": np.zeros(2, dtype=np.float32),
        "solved": {},
        "projection_used": True,
        "projection_failed": True,
        "certificate_solves": 1,
    }
    with pytest.raises(RuntimeError, match="refuses to execute"):
        FastMPCControlWrapper.execute_certified_target(wrapper, certificate)


def test_guardian_dsac_dataset_rejects_old_action_semantics(tmp_path):
    count = 3
    arrays = {}
    for name in REQUIRED_KEYS:
        if name in ("obs", "next_obs"):
            arrays[name] = np.zeros((count, 2), np.float32)
        elif name in ("raw_action", "action", "teacher_action"):
            arrays[name] = np.zeros((count, 2), np.float32)
        elif name in ("done", "projection_used", "projection_failed", "solver_finite", "critic_action_compatible", "bc_valid"):
            arrays[name] = np.zeros(count, dtype=bool)
        elif name in ("episode_id", "step_index"):
            arrays[name] = np.arange(count, dtype=np.int32)
        else:
            arrays[name] = np.zeros(count, np.float32)
    arrays["action_semantics_version"] = np.full(
        count, "target_point_speed_coupled_v1"
    )
    path = tmp_path / "old.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="action semantics mismatch"):
        GuardianDSACMetaSafeDataset(path)


def test_guardian_dsac_dataset_rejects_v231_metadata_mismatch(tmp_path):
    wrong_limit = tmp_path / "wrong_limit.npz"
    np.savez(
        wrong_limit,
        **_minimal_v231_dataset_arrays(episode_step_limit=1800),
    )
    with pytest.raises(ValueError, match="episode_step_limit mismatch"):
        GuardianDSACMetaSafeDataset(
            wrong_limit,
            expected_episode_step_limit=1200,
            expected_speed_sampling_profile="fast_v231",
        )

    wrong_profile = tmp_path / "wrong_profile.npz"
    np.savez(
        wrong_profile,
        **_minimal_v231_dataset_arrays(
            episode_step_limit=1200,
            speed_sampling_profile="balanced_v23",
        ),
    )
    with pytest.raises(ValueError, match="speed_sampling_profile mismatch"):
        GuardianDSACMetaSafeDataset(
            wrong_profile,
            expected_episode_step_limit=1200,
            expected_speed_sampling_profile="fast_v231",
        )


def test_v2_4_flow_actor_has_single_forward_action_and_no_candidate_api():
    policy, params = create_one_step_flow_policy(
        jax.random.PRNGKey(4), actor_obs_dim=4, act_dim=2, hidden_sizes=(8, 8)
    )
    action = policy.get_deterministic_action(
        params.one_step, jnp.zeros((5, 4), dtype=jnp.float32)
    )
    assert action.shape == (5, 2)
    assert not hasattr(policy, "sample_candidates")
    assert not hasattr(policy, "candidate_bank")


def test_v7_checkpoint_contains_required_bindings_and_round_trips(tmp_path):
    algorithm = _algorithm()
    batch = jax.device_put(_batch())
    algorithm.mark_offline_reference(batch)
    algorithm.promote_deployable({"decision": "fresh_process_safe"})
    path = tmp_path / "checkpoint.pkl"
    algorithm.save(path)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    required = {
        "learner_state",
        "hybrid_safe_snapshot",
        "deployable_checkpoint",
        "offline_reference_q1",
        "offline_reference_q2",
        "offline_reference_policy",
        "offline_support_thresholds",
        "ema_q_std_1",
        "ema_q_std_2",
        "action_semantics_version",
        "scene_seeds",
        "scene_signatures",
        "dataset_sha256",
    }
    assert required.issubset(payload)
    restored = _algorithm(seed=2)
    restored.load(path)
    assert restored.loaded_checkpoint_format == restored.FORMAT
    assert restored.deployable_checkpoint["metrics"]["decision"] == "fresh_process_safe"


def test_v7_checkpoint_rejects_dataset_hash_mismatch(tmp_path):
    algorithm = _algorithm()
    algorithm.mark_offline_reference(jax.device_put(_batch()))
    path = tmp_path / "checkpoint.pkl"
    algorithm.save(path)
    restored = _algorithm()
    restored.dataset_sha256 = "different"
    with pytest.raises(ValueError, match="dataset_sha256"):
        restored.load(path)
