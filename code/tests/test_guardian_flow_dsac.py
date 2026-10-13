from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from relax.algorithm.guardian_dsac_flow_v2_4 import (
    GuardianDSACFlowV24,
    GuardianDSACFlowV24Config,
)
from relax.algorithm.guardian_dsac_v2_3 import (
    GuardianDSACV23,
    GuardianDSACV23Config,
)
from relax.buffer.tree import TreeBuffer
from relax.network.flow_policy import create_one_step_flow_policy
from relax.network.safe_sac_jax import create_guardian_dsac_net
from relax.offline.guardian_dsac_multiseed_dataset import (
    guardian_dsac_to_flow_experience,
)
from relax.utils.experience import (
    GuardianDSACExperience,
    GuardianFlowDSACExperience,
)
from scripts.train_guardian_flow_dsac_multiseed import (
    _sample_actor_flow_batch,
    parse_args as parse_flow_args,
)
from scripts.train_guardian_dsac_metasafe import _sample_mixed_batch


def _tree_equal(left, right) -> bool:
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
        )
    )


def _tree_distance(left, right) -> float:
    return float(
        sum(
            np.sum(np.square(np.asarray(a) - np.asarray(b)))
            for a, b in zip(
                jax.tree_util.tree_leaves(left),
                jax.tree_util.tree_leaves(right),
            )
        )
    )


def _dsac_batch(size: int = 16) -> GuardianDSACExperience:
    rng = np.random.default_rng(24)
    batch = GuardianDSACExperience.create_example(6, 2, batch_size=size)
    obs = rng.normal(size=(size, 6)).astype(np.float32)
    action = np.tanh(rng.normal(size=(size, 2))).astype(np.float32)
    return batch._replace(
        obs=obs,
        next_obs=(obs + 0.01).astype(np.float32),
        action=action,
        raw_action=action,
        proxy_corrected_action=action,
        reward=rng.normal(size=size).astype(np.float32),
        teacher_action=action,
        teacher_valid=np.ones(size, dtype=bool),
        teacher_weight=np.ones(size, dtype=np.float32),
        actor_controlled=np.ones(size, dtype=bool),
        proposal_evaluated=np.ones(size, dtype=bool),
        guardian_dependence_target=np.zeros(size, dtype=np.float32),
        guardian_dependence_target_valid=np.ones(size, dtype=bool),
        behavior_return_to_go=rng.normal(size=size).astype(np.float32),
        behavior_return_valid=np.ones(size, dtype=bool),
        q_anchor_valid=np.ones(size, dtype=bool),
        action_semantics_version=np.full(size, 3, dtype=np.int8),
        reachable_speed_lower=np.ones(size, dtype=np.float32),
        reachable_speed_upper=np.full(size, 4.8, dtype=np.float32),
        actual_vehicle_speed=np.full(size, 2.0, dtype=np.float32),
    )


def _base(seed: int = 0, *, ready: bool = True) -> GuardianDSACV23:
    network, params = create_guardian_dsac_net(
        jax.random.PRNGKey(seed),
        critic_obs_dim=6,
        actor_obs_dim=4,
        act_dim=2,
        hidden_sizes=(16, 16),
    )
    algorithm = GuardianDSACV23(
        network,
        params,
        config=GuardianDSACV23Config(
            reference_weight_initial=0.0,
            reference_weight_final=0.0,
            proxy_actor_weight=0.15,
            proxy_actor_objective="softplus_logit",
        ),
        scene_seeds=(0, 1),
        scene_signatures={0: "zero", 1: "one"},
        dataset_sha256="five-seed-shared-test",
    )
    if ready:
        algorithm.mark_offline_reference(jax.device_put(_dsac_batch()))
    return algorithm


def _flow(
    seed: int = 0,
    *,
    config: GuardianDSACFlowV24Config | None = None,
) -> GuardianDSACFlowV24:
    policy, params = create_one_step_flow_policy(
        jax.random.PRNGKey(seed + 100),
        actor_obs_dim=4,
        act_dim=2,
        hidden_sizes=(16, 16),
        integration_steps=8,
    )
    return GuardianDSACFlowV24(
        _base(seed),
        policy,
        params,
        config=config or GuardianDSACFlowV24Config(),
    )


def _flow_batch(size: int = 16) -> GuardianFlowDSACExperience:
    return guardian_dsac_to_flow_experience(_dsac_batch(size))


def test_flow_replay_schema_is_independent_and_round_trips():
    assert "flow_noise" not in GuardianDSACExperience._fields
    assert GuardianFlowDSACExperience._fields[-2:] == (
        "flow_noise",
        "flow_noise_valid",
    )
    batch = _flow_batch(3)._replace(
        flow_noise=np.arange(6, dtype=np.float32).reshape(3, 2),
        flow_noise_valid=np.asarray([True, False, True]),
    )
    replay = TreeBuffer.from_guardian_flow_dsac_experience(6, 2, 8, seed=3)
    replay.add_batch(batch)
    sampled = replay.sample(5)
    assert isinstance(sampled, GuardianFlowDSACExperience)
    assert sampled.flow_noise.shape == (5, 2)
    assert sampled.flow_noise_valid.dtype == np.bool_


def test_offline_conversion_uses_invalid_zero_latents():
    converted = guardian_dsac_to_flow_experience(_dsac_batch(5))
    np.testing.assert_array_equal(converted.flow_noise, np.zeros((5, 2)))
    np.testing.assert_array_equal(converted.flow_noise_valid, np.zeros(5, bool))
    with pytest.raises(ValueError, match="latent shape"):
        guardian_dsac_to_flow_experience(
            _dsac_batch(5), flow_noise=np.zeros((5, 3), np.float32)
        )


def test_flow_starts_from_shared_offline_checkpoint_not_deployable():
    base = _base()
    assert not base.has_deployable_checkpoint
    policy, params = create_one_step_flow_policy(
        jax.random.PRNGKey(8), actor_obs_dim=4, act_dim=2, hidden_sizes=(8, 8)
    )
    algorithm = GuardianDSACFlowV24(base, policy, params)
    assert algorithm.base.offline_reference_ready
    with pytest.raises(ValueError, match="shared offline Q/P"):
        GuardianDSACFlowV24(_base(ready=False), policy, params)


def test_policy_samples_one_latent_and_deterministic_mode_is_zero():
    policy, params = create_one_step_flow_policy(
        jax.random.PRNGKey(7), actor_obs_dim=4, act_dim=2, hidden_sizes=(8, 8)
    )
    obs = jnp.zeros((5, 6), dtype=jnp.float32)
    action, noise = policy.sample_action(
        params.one_step, obs, jax.random.PRNGKey(9)
    )
    deterministic = policy.get_deterministic_action(params.one_step, obs)
    zero_action = policy.get_action(
        params.one_step, obs, jnp.zeros((5, 2), dtype=jnp.float32)
    )
    assert action.shape == noise.shape == (5, 2)
    np.testing.assert_allclose(deterministic, zero_action)
    assert not hasattr(policy, "sample_candidates")
    assert not hasattr(policy, "rank_actions")


def test_teacher_freeze_is_enforced_and_joint_training_cannot_move_it():
    algorithm = _flow()
    batch = jax.device_put(_flow_batch())
    algorithm.pretrain_bc_flow(jax.random.PRNGKey(1), batch)
    algorithm.distill_one_step(jax.random.PRNGKey(2), batch)
    algorithm.freeze_bc_flow()
    teacher = algorithm.state.params.bc_flow
    with pytest.raises(RuntimeError, match="teacher is frozen"):
        algorithm.pretrain_bc_flow(jax.random.PRNGKey(3), batch)
    metrics = algorithm.offline_flow_joint_update(jax.random.PRNGKey(4), batch)
    assert _tree_equal(teacher, algorithm.state.params.bc_flow)
    assert metrics["flow_actor_entropy_free"] == 1.0
    assert metrics["flow_implicit_entropy_target"] == 0.0


def test_offline_critic_uses_target_one_step_actor(monkeypatch):
    config = GuardianDSACFlowV24Config(
        q_actor_weight=0.0,
        proxy_actor_weight=0.0,
        offline_distill_weight_initial=0.0,
        offline_distill_weight_final=0.0,
    )
    algorithm = _flow(config=config)
    algorithm.freeze_bc_flow()
    batch = jax.device_put(_flow_batch(8))
    live = jax.tree_util.tree_map(
        lambda value: value + jnp.asarray(0.05, value.dtype),
        algorithm.state.params.one_step,
    )
    algorithm.state = algorithm.state._replace(
        params=algorithm.state.params._replace(one_step=live)
    )
    captured = {}

    def fake_critic(key, data, *, next_action, policy_action):
        del key, data, policy_action
        captured["next_action"] = np.asarray(next_action)
        return {}

    monkeypatch.setattr(
        algorithm.base, "offline_critic_update_with_policy_actions", fake_critic
    )
    monkeypatch.setattr(algorithm.base, "proxy_update", lambda data: {})
    key = jax.random.PRNGKey(91)
    _, next_key, _, _ = jax.random.split(key, 4)
    noise = jax.random.normal(next_key, batch.action.shape, dtype=jnp.float32)
    expected = algorithm.flow_policy.get_action(
        algorithm.state.target_one_step, batch.next_obs, noise
    )
    live_next = algorithm.flow_policy.get_action(live, batch.next_obs, noise)
    algorithm.offline_flow_joint_update(key, batch)
    np.testing.assert_allclose(captured["next_action"], expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(np.asarray(expected), np.asarray(live_next))


def _correction_batch(*, proxy_valid: bool, latent_valid: bool = True):
    size = 32
    rng = np.random.default_rng(4)
    batch = _flow_batch(size)
    executed = np.tile(np.asarray([0.75, -0.65], np.float32), (size, 1))
    corrected = np.tile(np.asarray([-0.55, 0.45], np.float32), (size, 1))
    return batch._replace(
        obs=rng.normal(size=(size, 6)).astype(np.float32),
        action=executed,
        proxy_corrected_action=corrected,
        actor_controlled=np.zeros(size, bool),
        guardian_dependence_target_valid=np.zeros(size, bool),
        guardian_dependence_corrected_valid=np.zeros(size, bool),
        actual_control_modified=np.ones(size, bool),
        intervention_hard=np.ones(size, bool),
        intervention_soft=np.zeros(size, bool),
        proxy_pair_valid=np.full(size, proxy_valid, bool),
        flow_noise=rng.normal(size=(size, 2)).astype(np.float32),
        flow_noise_valid=np.full(size, latent_valid, bool),
    )


@pytest.mark.parametrize("proxy_valid", [False, True])
def test_same_latent_correction_moves_to_proxy_or_executed_fallback(proxy_valid):
    algorithm = _flow(
        config=GuardianDSACFlowV24Config(
            learning_rate=3e-3,
            q_actor_weight=0.0,
            proxy_actor_weight=0.0,
            correction_weight=1.0,
            proximal_weight_initial=0.0,
            proximal_weight_min=0.0,
        )
    )
    algorithm.freeze_bc_flow()
    batch = jax.device_put(_correction_batch(proxy_valid=proxy_valid))
    target = batch.proxy_corrected_action if proxy_valid else batch.action

    def error():
        prediction = algorithm.flow_policy.get_action(
            algorithm.state.params.one_step,
            batch.obs,
            batch.flow_noise,
        )
        return float(np.mean(np.square(np.asarray(prediction - target))))

    before = error()
    metrics = None
    key = jax.random.PRNGKey(110)
    for _ in range(80):
        key, update_key = jax.random.split(key)
        algorithm.state, metrics = algorithm._online_actor_update(
            update_key,
            algorithm.state,
            batch,
            algorithm.base.state.params.q1,
            algorithm.base.state.params.q2,
            algorithm.base.state.params.proxy,
        )
    after = error()
    metrics = {name: float(value) for name, value in metrics.items()}
    assert after < before * 0.45
    assert metrics["flow_correction_support_fraction"] == pytest.approx(1.0)
    assert metrics["flow_correction_proxy_target_fraction"] == pytest.approx(
        float(proxy_valid)
    )


def test_invalid_latent_has_zero_correction_support_and_no_actor_change():
    algorithm = _flow(
        config=GuardianDSACFlowV24Config(
            q_actor_weight=0.0,
            proxy_actor_weight=0.0,
            correction_weight=1.0,
            proximal_weight_initial=0.0,
            proximal_weight_min=0.0,
        )
    )
    algorithm.freeze_bc_flow()
    batch = jax.device_put(
        _correction_batch(proxy_valid=True, latent_valid=False)
    )
    before = algorithm.state.params.one_step
    algorithm.state, metrics = algorithm._online_actor_update(
        jax.random.PRNGKey(12),
        algorithm.state,
        batch,
        algorithm.base.state.params.q1,
        algorithm.base.state.params.q2,
        algorithm.base.state.params.proxy,
    )
    assert _tree_equal(before, algorithm.state.params.one_step)
    assert float(metrics["flow_correction_loss"]) == pytest.approx(0.0)
    assert float(metrics["flow_correction_support_fraction"]) == pytest.approx(0.0)


def test_hard_and_soft_corrections_use_one_and_half_weights():
    algorithm = _flow(
        config=GuardianDSACFlowV24Config(
            q_actor_weight=0.0,
            proxy_actor_weight=0.0,
            correction_weight=1.0,
            proximal_weight_initial=0.0,
            proximal_weight_min=0.0,
        )
    )
    algorithm.freeze_bc_flow()
    batch = _correction_batch(proxy_valid=True)
    half = batch.reward.shape[0] // 2
    batch = jax.device_put(
        batch._replace(
            intervention_hard=np.arange(batch.reward.shape[0]) < half,
            intervention_soft=np.arange(batch.reward.shape[0]) >= half,
        )
    )
    algorithm.state, metrics = algorithm._online_actor_update(
        jax.random.PRNGKey(19),
        algorithm.state,
        batch,
        algorithm.base.state.params.q1,
        algorithm.base.state.params.q2,
        algorithm.base.state.params.proxy,
    )
    assert float(metrics["flow_correction_weight_mean"]) == pytest.approx(0.75)


def test_target_actor_is_ema_and_gate_pass_does_not_reset_proximal():
    algorithm = _flow()
    algorithm.freeze_bc_flow()
    old_target = algorithm.state.target_one_step
    batch = jax.device_put(_flow_batch())
    algorithm.state, _ = algorithm._offline_actor_update(
        jax.random.PRNGKey(20),
        algorithm.state,
        batch,
        algorithm.base.state.params.q1,
        algorithm.base.state.params.q2,
        algorithm.base.state.params.proxy,
    )
    live = algorithm.state.params.one_step
    target = algorithm.state.target_one_step
    assert _tree_distance(old_target, target) > 0.0
    assert _tree_distance(target, live) > 0.0
    algorithm.state = algorithm.state._replace(
        proximal_step=jnp.asarray(123, dtype=jnp.int32)
    )
    algorithm.promote_hybrid_safe()
    assert int(algorithm.state.proximal_step) == 123
    assert _tree_equal(algorithm.state.safe_one_step, algorithm.state.params.one_step)


def test_hybrid_failure_restores_only_flow_actor_and_target():
    algorithm = _flow()
    algorithm.freeze_bc_flow()
    algorithm.promote_hybrid_safe()
    safe = algorithm.state.safe_one_step
    changed = jax.tree_util.tree_map(
        lambda value: value + jnp.asarray(0.01, value.dtype), safe
    )
    algorithm.state = algorithm.state._replace(
        params=algorithm.state.params._replace(one_step=changed),
        target_one_step=changed,
        proximal_step=jnp.asarray(77, dtype=jnp.int32),
    )
    base_before = algorithm.base.state
    result = algorithm.reject_hybrid_safety()
    assert result["rollback"] == "actor_only"
    assert _tree_equal(base_before, algorithm.base.state)
    assert _tree_equal(safe, algorithm.state.params.one_step)
    assert _tree_equal(safe, algorithm.state.target_one_step)
    assert int(algorithm.state.proximal_step) == 0


def test_numerical_disaster_restores_complete_safe_snapshot():
    algorithm = _flow()
    algorithm.freeze_bc_flow()
    algorithm.promote_hybrid_safe()
    safe_base = algorithm.base.state
    safe_flow = algorithm.state
    changed_q1 = jax.tree_util.tree_map(
        lambda value: value + jnp.asarray(0.02, value.dtype),
        algorithm.base.state.params.q1,
    )
    algorithm.base.state = algorithm.base.state._replace(
        params=algorithm.base.state.params._replace(q1=changed_q1)
    )
    changed_actor = jax.tree_util.tree_map(
        lambda value: value + jnp.asarray(0.03, value.dtype),
        algorithm.state.params.one_step,
    )
    algorithm.state = algorithm.state._replace(
        params=algorithm.state.params._replace(one_step=changed_actor)
    )
    result = algorithm.reject_numerical_disaster()
    assert result["rollback"] == "hybrid_safe_full_snapshot"
    assert _tree_equal(safe_base, algorithm.base.state)
    assert _tree_equal(safe_flow, algorithm.state)


def test_checkpoint_binds_forbidden_mechanisms_and_round_trips(tmp_path):
    algorithm = _flow()
    algorithm.freeze_bc_flow()
    path = tmp_path / "flow.pkl"
    algorithm.save(path)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    assert payload["format"] == GuardianDSACFlowV24.FORMAT
    for name in (
        "candidate_sampling",
        "q_reranking",
        "rejection_sampling",
        "reverse_ode_likelihood",
        "flow_internal_cbf",
        "explicit_actor_entropy",
        "teacher_online_updates",
    ):
        assert payload[name] is False
    restored = _flow(seed=2)
    restored.load(path)
    observation = np.zeros(6, np.float32)
    np.testing.assert_allclose(
        algorithm.get_deterministic_action(observation),
        restored.get_deterministic_action(observation),
    )

    payload["q_reranking"] = True
    bad = tmp_path / "bad.pkl"
    with bad.open("wb") as stream:
        pickle.dump(payload, stream)
    with pytest.raises(ValueError, match="forbidden"):
        _flow(seed=3).load(bad)


def test_formal_v24_defaults_match_v231_moving_safe_contract():
    config = GuardianDSACFlowV24Config()
    assert config.target_actor_tau == 0.005
    assert config.proximal_weight_initial == 1.0
    assert config.proximal_weight_min == 0.05
    assert config.proximal_decay_steps == 20_000
    assert config.proximal_speed_weight == 0.5
    assert config.proximal_lateral_weight == 1.0
    assert config.offline_distill_weight_initial == 1.0
    assert config.offline_distill_weight_final == 0.2


def test_formal_entrypoint_binds_single_flow_variant_and_budgets():
    args = parse_flow_args(["--shared-offline-checkpoint", "shared.pkl"])
    assert args.flow_integration_steps == 8
    assert args.flow_bc_gradient_steps == 80_000
    assert args.flow_distill_gradient_steps == 40_000
    assert args.flow_offline_joint_steps == 50_000
    assert args.online_steps == 150_000
    assert args.batch_size == 512
    assert args.correction_fraction == 0.25
    assert not hasattr(args, "online_exploration_std")
    assert not hasattr(args, "candidate_count")


def test_actor_sampler_reserves_exactly_one_quarter_for_corrections():
    class OfflineOnly:
        def sample_flow(self, size, *, split, to_jax):
            assert split == "train" and not to_jax
            return _flow_batch(size)

    ordinary_replay = TreeBuffer.from_guardian_flow_dsac_experience(6, 2, 32)
    correction_replay = TreeBuffer.from_guardian_flow_dsac_experience(6, 2, 32)
    correction = _correction_batch(proxy_valid=True)._replace(
        flow_noise_valid=np.ones(32, bool)
    )
    correction_replay.add_batch(correction)
    batch = _sample_actor_flow_batch(
        OfflineOnly(),
        ordinary_replay,
        correction_replay,
        batch_size=20,
        offline_ratio=0.5,
        correction_fraction=0.25,
        rng=np.random.default_rng(22),
    )
    assert int(np.count_nonzero(np.asarray(batch.flow_noise_valid))) == 5


def test_legacy_single_scene_sampler_adapts_offline_rows_to_flow_schema():
    class OfflineOnly:
        def sample(self, size, *, split, to_jax):
            assert split == "train" and not to_jax
            return _dsac_batch(size)

    replay = TreeBuffer.from_guardian_flow_dsac_experience(6, 2, 16)
    batch = _sample_mixed_batch(
        OfflineOnly(),
        replay,
        batch_size=8,
        offline_ratio=0.5,
        rng=np.random.default_rng(9),
        flow=True,
    )
    assert isinstance(batch, GuardianFlowDSACExperience)
    assert not np.any(np.asarray(batch.flow_noise_valid))
