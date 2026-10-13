from __future__ import annotations

import hashlib
import json
import os
import pickle

import jax
import numpy as np
import pytest

from relax.algorithm.guardian_dsac_flow_paper import (
    GuardianDSACFlowPaper,
    GuardianDSACFlowPaperConfig,
)
from relax.algorithm.guardian_dsac_v2_3 import (
    GuardianDSACV23,
    GuardianDSACV23Config,
)
from relax.buffer.seed_indexed import SeedIndexedReplay
from relax.buffer.tree import TreeBuffer
from relax.network.flow_policy import create_one_step_flow_policy
from relax.network.safe_sac_jax import create_guardian_dsac_net
from relax.offline.guardian_dsac_multiseed_dataset import (
    guardian_dsac_to_flow_long_run_experience,
)
from relax.utils.experience import GuardianDSACExperience
from relax.utils.flow_paper import (
    ALGORITHM_SEEDS,
    NOVEL_SEEDS,
    OFFLINE_SEEDS,
    PAPER_SCENE_SEEDS,
    TRANSITION_TARGET_WEIGHTS,
    PaperTransitionDeficitScheduler,
    UniformEpisodeLatentPolicy,
    validate_paper_seed_contract,
)
from scripts.aggregate_guardian_flow_paper_results import _report, _sampling_audit
from scripts.evaluate_guardian_dsac_isolated import (
    UNCERTIFIED_PROPOSAL_HANDLING,
    _aggregate,
    _policy_action,
)
from scripts.run_guardian_flow_dsac_paper_suite import (
    _resumable,
    parse_args as parse_suite_args,
)
from scripts.train_guardian_flow_dsac_paper import (
    _actor_objective_terms,
    _extension_config_failures,
    _flow_config,
    _formal_evaluation_lock,
    _load_atomic_resume_algorithm,
    _load_atomic_trainer_control,
    _load_numerical_checkpoint,
    _paper_evaluation_summary,
    _proxy_weight_at_step,
    _sampling_protocol_payload,
    _validate_paper_evaluation,
    parse_args as parse_paper_args,
)


def _tree_equal(left, right) -> bool:
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
        )
    )


def _dsac_batch(size: int = 32):
    rng = np.random.default_rng(2026)
    batch = GuardianDSACExperience.create_example(6, 2, batch_size=size)
    obs = rng.normal(size=(size, 6)).astype(np.float32)
    action = np.tanh(rng.normal(size=(size, 2))).astype(np.float32)
    half = size // 2
    scene_seed = np.concatenate(
        [
            np.resize(np.asarray(OFFLINE_SEEDS, np.int32), half),
            np.resize(np.asarray(NOVEL_SEEDS, np.int32), size - half),
        ]
    )
    modified = np.zeros(size, bool)
    modified[half:] = True
    corrected = action.copy()
    corrected[half:, 0] = -0.5
    corrected[half:, 1] = 0.4
    return batch._replace(
        obs=obs,
        next_obs=(obs + 0.01).astype(np.float32),
        action=action,
        raw_action=action,
        proxy_corrected_action=corrected,
        reward=rng.normal(size=size).astype(np.float32),
        teacher_action=action,
        teacher_valid=np.ones(size, bool),
        teacher_weight=np.ones(size, np.float32),
        actor_controlled=~modified,
        proposal_evaluated=np.ones(size, bool),
        actual_control_modified=modified,
        intervention=modified,
        intervention_hard=modified,
        intervention_soft=np.zeros(size, bool),
        proxy_pair_valid=modified,
        guardian_dependence_target=np.linspace(0.0, 1.0, size, dtype=np.float32),
        guardian_dependence_target_valid=np.ones(size, bool),
        guardian_dependence_corrected_valid=np.ones(size, bool),
        behavior_return_to_go=rng.normal(size=size).astype(np.float32),
        behavior_return_valid=np.ones(size, bool),
        q_anchor_valid=np.ones(size, bool),
        action_semantics_version=np.full(size, 3, np.int8),
        scene_seed=scene_seed,
        reachable_speed_lower=np.ones(size, np.float32),
        reachable_speed_upper=np.full(size, 4.8, np.float32),
        actual_vehicle_speed=np.full(size, 2.0, np.float32),
    )


def _base(seed: int = 0):
    network, params = create_guardian_dsac_net(
        jax.random.PRNGKey(seed),
        critic_obs_dim=6,
        actor_obs_dim=6,
        act_dim=2,
        hidden_sizes=(16, 16),
    )
    base = GuardianDSACV23(
        network,
        params,
        config=GuardianDSACV23Config(
            reference_weight_initial=0.0,
            reference_weight_final=0.0,
            proxy_actor_objective="softplus_logit",
        ),
        scene_seeds=PAPER_SCENE_SEEDS,
        scene_signatures={seed: f"paper-{seed}" for seed in PAPER_SCENE_SEEDS},
        dataset_sha256="paper-five-seed-test",
    )
    base.mark_offline_reference(jax.device_put(_dsac_batch()))
    return base


def _flow(seed: int = 0, *, config=None):
    policy, params = create_one_step_flow_policy(
        jax.random.PRNGKey(seed + 100),
        actor_obs_dim=6,
        act_dim=2,
        hidden_sizes=(16, 16),
        integration_steps=8,
    )
    return GuardianDSACFlowPaper(
        _base(seed),
        policy,
        params,
        config=config or GuardianDSACFlowPaperConfig(),
    )


def _paper_batch(size: int = 32):
    base = _dsac_batch(size)
    half = size // 2
    noise = np.zeros((size, 2), np.float32)
    noise[half:] = np.random.default_rng(42).normal(size=(size - half, 2))
    valid = np.zeros(size, bool)
    valid[half:] = True
    zero = np.zeros(size, bool)
    zero[half : half + (size - half) // 2] = True
    noise[zero] = 0.0
    return guardian_dsac_to_flow_long_run_experience(
        base,
        flow_noise=noise,
        flow_noise_valid=valid,
        flow_zero_latent=zero,
    )


def test_paper_seed_and_scheduler_contracts_are_predeclared():
    assert PAPER_SCENE_SEEDS == (0, 1, 2, 3, 4, 5, 6, 7, 8, 10)
    assert OFFLINE_SEEDS == (0, 1, 2, 3, 4)
    assert NOVEL_SEEDS == (5, 6, 7, 8, 10)
    assert ALGORITHM_SEEDS == (0, 1, 2)
    assert 9 not in PAPER_SCENE_SEEDS
    assert sum(
        TRANSITION_TARGET_WEIGHTS[seed] for seed in OFFLINE_SEEDS
    ) == pytest.approx(0.20)
    assert sum(
        TRANSITION_TARGET_WEIGHTS[seed] for seed in NOVEL_SEEDS
    ) == pytest.approx(0.80)
    with pytest.raises(ValueError, match="exactly"):
        validate_paper_seed_contract(tuple(range(10)))

    scheduler = PaperTransitionDeficitScheduler()
    for _ in range(10_000):
        scheduler.record_step(scheduler.current_seed)
        scheduler.complete_episode()
    assert scheduler.fractions() == pytest.approx(TRANSITION_TARGET_WEIGHTS, abs=1e-4)

    flow_payload = _sampling_protocol_payload(flow=True)
    assert flow_payload["transition_target_weights"] == {
        str(seed): value for seed, value in TRANSITION_TARGET_WEIGHTS.items()
    }
    assert flow_payload["correction_seed_sampling_weights"] == pytest.approx(
        {str(seed): 0.1 for seed in PAPER_SCENE_SEEDS}
    )
    assert set(flow_payload["zero_latent_probability_by_seed"].values()) == {0.5}
    gaussian_payload = _sampling_protocol_payload(flow=False)
    assert gaussian_payload["correction_seed_sampling_weights"] is None
    assert gaussian_payload["zero_latent_probability_by_seed"] is None


def test_every_scene_uses_the_same_half_zero_latent_protocol():
    policy = UniformEpisodeLatentPolicy()
    rng = np.random.default_rng(7)
    for seed in PAPER_SCENE_SEEDS:
        modes = [policy.sample(seed, rng) for _ in range(100)]
        assert modes.count("zero") == 50
        assert modes.count("random") == 50


def test_paper_actor_has_only_four_terms_and_seed0_to4_anchor():
    algorithm = _flow()
    algorithm.freeze_bc_flow()
    preparation = algorithm.prepare_online_training()
    assert preparation["performance_gate"] is False
    batch = jax.device_put(_paper_batch())
    metrics = algorithm.online_joint_update(
        jax.random.PRNGKey(11), batch, actor_data=batch, update_actor=True
    )
    assert metrics["flow_anchor_support_fraction"] == pytest.approx(0.5)
    assert metrics["flow_correction_support_fraction"] == pytest.approx(0.5)
    assert metrics["flow_proximal_enabled"] == 0.0
    assert "flow_proximal_loss" not in metrics
    assert algorithm.OBJECTIVE_TERMS == (
        "task_q",
        "guardian_p",
        "correction",
        "teacher_anchor",
    )


def test_actor_accepts_dynamic_proxy_and_correction_weights_without_changing_p_update():
    algorithm = _flow()
    algorithm.freeze_bc_flow()
    algorithm.prepare_online_training()
    batch = jax.device_put(_paper_batch())
    before_proxy_updates = algorithm.online_critic_updates_total
    metrics = algorithm.online_joint_update(
        jax.random.PRNGKey(12),
        batch,
        actor_data=batch,
        update_actor=True,
        proxy_actor_weight=0.03,
        correction_weight=1.0,
    )
    assert metrics["flow_proxy_actor_weight"] == pytest.approx(0.03)
    assert metrics["flow_correction_actor_weight"] == pytest.approx(1.0)
    assert algorithm.online_critic_updates_total == before_proxy_updates + 1
    assert metrics["flow_actor_loss"] == pytest.approx(
        metrics["flow_q_loss"]
        + 0.03 * metrics["flow_proxy_loss"]
        + metrics["flow_correction_loss"]
        + 0.05 * metrics["flow_anchor_loss"],
        abs=1e-5,
    )


@pytest.mark.parametrize(
    ("schedule", "expected"),
    (
        ("p_hard_off_300k", (0.0, 0.0, 0.0)),
        ("p_decay_300k_500k", (0.15, 0.075, 0.0)),
        ("p_decay_300k_700k", (0.15, 0.1125, 0.0)),
        ("p_decay_floor_003", (0.15, 0.09, 0.03)),
    ),
)
def test_proxy_weight_schedules_have_exact_300k_400k_and_endpoint_values(
    schedule, expected
):
    endpoint = 700_000 if schedule == "p_decay_300k_700k" else 500_000
    observed = (
        _proxy_weight_at_step(schedule, 300_000),
        _proxy_weight_at_step(schedule, 400_000),
        _proxy_weight_at_step(schedule, endpoint),
    )
    assert observed == pytest.approx(expected)


def test_flow_q_only_disables_p_and_correction_but_keeps_anchor():
    config = GuardianDSACFlowPaperConfig(
        proxy_actor_weight=0.0,
        correction_weight=0.0,
        anchor_weight=0.05,
    )
    algorithm = _flow(config=config)
    assert algorithm.config.proxy_actor_weight == 0.0
    assert algorithm.config.correction_weight == 0.0
    assert algorithm.config.anchor_weight == 0.05


def test_minimal_ablations_each_remove_exactly_one_actor_term():
    common = [
        "--shared-offline-checkpoint",
        "shared.pkl",
        "--flow-offline-checkpoint",
        "flow.pkl",
        "--seed",
        "1",
        "--online-steps",
        "300000",
    ]
    no_p_args = parse_paper_args(["--method", "flow_no_p", *common])
    no_correction_args = parse_paper_args(["--method", "flow_no_correction", *common])
    no_p = _flow_config(no_p_args, method="flow_no_p")
    no_correction = _flow_config(no_correction_args, method="flow_no_correction")

    assert no_p.proxy_actor_weight == 0.0
    assert no_p.correction_weight == pytest.approx(2.0)
    assert _actor_objective_terms("flow_no_p") == [
        "task_q",
        "correction",
        "teacher_anchor",
    ]
    assert no_correction.proxy_actor_weight == pytest.approx(0.15)
    assert no_correction.correction_weight == 0.0
    assert _actor_objective_terms("flow_no_correction") == [
        "task_q",
        "guardian_p",
        "teacher_anchor",
    ]


def test_same_seed_full_flow_reference_uses_300k_without_actor_change():
    args = parse_paper_args(
        [
            "--method",
            "full_flow",
            "--shared-offline-checkpoint",
            "shared.pkl",
            "--flow-offline-checkpoint",
            "flow.pkl",
            "--seed",
            "1",
            "--phase",
            "online",
            "--online-steps",
            "300000",
            "--experiment-profile",
            "same_seed_ablation",
        ]
    )
    config = _flow_config(args, method="full_flow")

    assert args.online_steps == 300_000
    assert config.proxy_actor_weight == pytest.approx(0.15)
    assert config.correction_weight == pytest.approx(2.0)
    assert config.anchor_weight == pytest.approx(0.05)
    assert _actor_objective_terms("full_flow") == [
        "task_q",
        "guardian_p",
        "correction",
        "teacher_anchor",
    ]


@pytest.mark.parametrize(
    ("method", "proxy_weight", "correction_weight"),
    (
        ("full_flow", 0.15, 2.0),
        ("flow_no_p", 0.0, 2.0),
        ("flow_no_correction", 0.15, 0.0),
    ),
)
def test_same_seed_ablation_1m_accepts_only_explicit_atomic_extension(
    tmp_path, method, proxy_weight, correction_weight
):
    parent = tmp_path / "parent" / method / "train_seed_1"
    child = tmp_path / "child" / method / "train_seed_1"
    common = [
        "--method",
        method,
        "--shared-offline-checkpoint",
        "shared.pkl",
        "--flow-offline-checkpoint",
        "flow.pkl",
        "--seed",
        "1",
        "--phase",
        "online",
        "--online-steps",
        "1000000",
        "--experiment-profile",
        "same_seed_ablation_1m",
        "--extend-ablation-from-300k",
        "--resume-run",
        str(child),
        "--extension-parent-run",
        str(parent),
    ]
    args = parse_paper_args(common)
    config = _flow_config(args, method=method)
    assert args.online_steps == 1_000_000
    assert config.proxy_actor_weight == pytest.approx(proxy_weight)
    assert config.correction_weight == pytest.approx(correction_weight)

    with pytest.raises(SystemExit):
        parse_paper_args(
            [value for value in common if value != "--extend-ablation-from-300k"]
        )


@pytest.mark.parametrize(
    "schedule",
    (
        "p_hard_off_300k",
        "p_decay_300k_500k",
        "p_decay_300k_700k",
        "p_decay_floor_003",
    ),
)
def test_proxy_annealing_profile_requires_explicit_atomic_extension(schedule, tmp_path):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    common = [
        "--method",
        "full_flow",
        "--shared-offline-checkpoint",
        "shared.pkl",
        "--flow-offline-checkpoint",
        "flow.pkl",
        "--seed",
        "1",
        "--phase",
        "online",
        "--online-steps",
        "1000000",
        "--experiment-profile",
        "proxy_annealing_1m",
        "--extend-proxy-annealing-from-300k",
        "--proxy-weight-schedule",
        schedule,
        "--resume-run",
        str(child),
        "--extension-parent-run",
        str(parent),
    ]
    args = parse_paper_args(common)
    config = _flow_config(args, method="full_flow")
    assert args.online_steps == 1_000_000
    assert args.proxy_weight_schedule == schedule
    assert config.proxy_actor_weight == pytest.approx(0.15)
    assert config.correction_weight == pytest.approx(2.0)

    with pytest.raises(SystemExit):
        parse_paper_args(
            [value for value in common if value != "--extend-proxy-annealing-from-300k"]
        )


def test_extension_config_comparison_json_normalizes_tuples():
    parent = {"flow_config": {"anchor_seeds": [0, 1, 2, 3, 4]}}
    desired = {"flow_config": {"anchor_seeds": (0, 1, 2, 3, 4)}}
    assert _extension_config_failures(parent, desired) == []
    desired["flow_config"]["anchor_seeds"] = (0, 1, 2, 3)
    assert _extension_config_failures(parent, desired) == ["flow_config"]


@pytest.mark.parametrize(
    ("config", "expected"),
    (
        (
            GuardianDSACFlowPaperConfig(
                proxy_actor_weight=0.0,
                correction_weight=2.0,
                anchor_weight=0.05,
            ),
            lambda metrics: (
                metrics["flow_q_loss"]
                + 2.0 * metrics["flow_correction_loss"]
                + 0.05 * metrics["flow_anchor_loss"]
            ),
        ),
        (
            GuardianDSACFlowPaperConfig(
                proxy_actor_weight=0.15,
                correction_weight=0.0,
                anchor_weight=0.05,
            ),
            lambda metrics: (
                metrics["flow_q_loss"]
                + 0.15 * metrics["flow_proxy_loss"]
                + 0.05 * metrics["flow_anchor_loss"]
            ),
        ),
    ),
)
def test_minimal_ablation_actor_loss_contains_only_declared_terms(config, expected):
    algorithm = _flow(config=config)
    algorithm.freeze_bc_flow()
    algorithm.prepare_online_training()
    batch = jax.device_put(_paper_batch())
    metrics = algorithm.online_joint_update(
        jax.random.PRNGKey(77), batch, actor_data=batch, update_actor=True
    )
    assert metrics["flow_actor_loss"] == pytest.approx(expected(metrics), abs=1e-5)


def test_no_p_actor_update_is_invariant_to_proxy_parameters():
    config = GuardianDSACFlowPaperConfig(
        proxy_actor_weight=0.0,
        correction_weight=2.0,
        anchor_weight=0.05,
    )
    algorithm = _flow(config=config)
    algorithm.freeze_bc_flow()
    algorithm.prepare_online_training()
    batch = jax.device_put(_paper_batch())
    key = jax.random.PRNGKey(713)
    state_a, _ = algorithm._online_actor_update_long_run(
        key,
        algorithm.state,
        batch,
        algorithm.base.state.params.q1,
        algorithm.base.state.params.q2,
        algorithm.base.state.params.proxy,
        np.float32(1e-4),
    )
    perturbed_proxy = jax.tree_util.tree_map(
        lambda value: value + jax.random.normal(key, value.shape) * 10.0,
        algorithm.base.state.params.proxy,
    )
    state_b, _ = algorithm._online_actor_update_long_run(
        key,
        algorithm.state,
        batch,
        algorithm.base.state.params.q1,
        algorithm.base.state.params.q2,
        perturbed_proxy,
        np.float32(1e-4),
    )
    assert _tree_equal(state_a.params.one_step, state_b.params.one_step)


def test_no_correction_actor_update_is_invariant_to_correction_target():
    config = GuardianDSACFlowPaperConfig(
        proxy_actor_weight=0.15,
        correction_weight=0.0,
        anchor_weight=0.05,
    )
    algorithm = _flow(config=config)
    algorithm.freeze_bc_flow()
    algorithm.prepare_online_training()
    batch = jax.device_put(_paper_batch())
    changed = batch._replace(
        proxy_corrected_action=-7.0 * batch.proxy_corrected_action + 3.0,
    )
    key = jax.random.PRNGKey(719)
    common = (
        key,
        algorithm.state,
    )
    trailing = (
        algorithm.base.state.params.q1,
        algorithm.base.state.params.q2,
        algorithm.base.state.params.proxy,
        np.float32(1e-4),
    )
    state_a, _ = algorithm._online_actor_update_long_run(*common, batch, *trailing)
    state_b, _ = algorithm._online_actor_update_long_run(*common, changed, *trailing)
    assert _tree_equal(state_a.params.one_step, state_b.params.one_step)


def test_minimal_ablations_reject_wrong_seed_or_budget():
    base = ["--shared-offline-checkpoint", "shared.pkl"]
    with pytest.raises(SystemExit):
        parse_paper_args(
            [
                "--method",
                "flow_no_p",
                *base,
                "--flow-offline-checkpoint",
                "flow.pkl",
                "--seed",
                "0",
                "--online-steps",
                "300000",
            ]
        )
    with pytest.raises(SystemExit):
        parse_paper_args(
            [
                "--method",
                "flow_no_correction",
                *base,
                "--flow-offline-checkpoint",
                "flow.pkl",
                "--seed",
                "1",
                "--online-steps",
                "1000000",
            ]
        )


def test_formal_evaluation_lock_is_opt_in(monkeypatch, tmp_path):
    lock = tmp_path / "evaluation.lock"
    monkeypatch.setenv("GUARDIAN_PAPER_EVALUATION_LOCK", str(lock))
    with _formal_evaluation_lock():
        assert lock.is_file()
        assert os.environ["GUARDIAN_PAPER_EVALUATION_LOCK"] == str(lock)


def test_evaluation_outcomes_never_change_training_state():
    algorithm = _flow()
    algorithm.freeze_bc_flow()
    algorithm.prepare_online_training()
    before = (algorithm.base.state, algorithm.state)
    assert algorithm.reject_hybrid_safety()["rollback"] == "none"
    assert algorithm.promote_hybrid_safe()["snapshot"] == "unchanged"
    assert algorithm.record_ordinary_failure()["evaluation_only"] is True
    assert _tree_equal(before, (algorithm.base.state, algorithm.state))


def test_only_numerical_checkpoint_can_restore_state():
    algorithm = _flow()
    algorithm.freeze_bc_flow()
    algorithm.prepare_online_training()
    snapshot = (algorithm.base.state, algorithm.state)
    batch = jax.device_put(_paper_batch())
    algorithm.online_joint_update(
        jax.random.PRNGKey(22), batch, actor_data=batch, update_actor=True
    )
    assert not _tree_equal(snapshot, (algorithm.base.state, algorithm.state))
    result = algorithm.reject_numerical_disaster()
    assert result["rollback"] == "latest_numerical_checkpoint"
    assert _tree_equal(snapshot, (algorithm.base.state, algorithm.state))


def test_paper_checkpoint_round_trip_binds_disabled_mechanisms(tmp_path):
    algorithm = _flow()
    algorithm.freeze_bc_flow()
    algorithm.prepare_online_training()
    checkpoint = tmp_path / "paper.pkl"
    algorithm.save(str(checkpoint))
    with checkpoint.open("rb") as stream:
        payload = pickle.load(stream)
    assert payload["format"] == GuardianDSACFlowPaper.FORMAT
    assert payload["paper_scene_seeds"] == list(PAPER_SCENE_SEEDS)
    assert payload["anchor_seeds"] == list(OFFLINE_SEEDS)
    assert payload["moving_safe_proximal"] is False
    assert payload["performance_rollback"] is False
    assert payload["safe_trajectory_memory"] is False
    assert payload["seed9_logic"] is False

    restored = _flow()
    restored.load(str(checkpoint))
    assert _tree_equal(algorithm.state, restored.state)
    assert _tree_equal(algorithm.base.state, restored.base.state)


def test_seed_indexed_replay_resume_is_exact_across_ring_wrap(tmp_path):
    replay = SeedIndexedReplay(
        TreeBuffer.from_guardian_flow_dsac_long_run_experience(6, 2, 5, seed=3),
        seeds=PAPER_SCENE_SEEDS,
        seed=4,
    )
    first = _paper_batch(4)
    replay.add_batch(first)
    replay.add_batch(_paper_batch(3))
    checkpoint = tmp_path / "replay.pkl"
    replay.save(checkpoint)
    assert not checkpoint.with_suffix(".pkl.tmp").exists()
    restored = SeedIndexedReplay(
        TreeBuffer.from_guardian_flow_dsac_long_run_experience(6, 2, 5, seed=999),
        seeds=PAPER_SCENE_SEEDS,
        seed=998,
    )
    restored.load(checkpoint)
    assert restored.seed_counts() == replay.seed_counts()
    assert restored.buffer.ptr == replay.buffer.ptr
    assert len(restored) == len(replay) == 5
    original_sample = replay.sample_seed_weighted(12, weights=TRANSITION_TARGET_WEIGHTS)
    restored_sample = restored.sample_seed_weighted(
        12, weights=TRANSITION_TARGET_WEIGHTS
    )
    assert _tree_equal(original_sample, restored_sample)


def test_paper_cli_binds_formal_budget_and_has_no_performance_gate_options():
    args = parse_paper_args(
        [
            "--method",
            "full_flow",
            "--shared-offline-checkpoint",
            "shared.pkl",
        ]
    )
    assert args.online_steps == 1_000_000
    assert args.batch_size == 512
    assert args.online_buffer_size == 500_000
    assert args.evaluation_every == 50_000
    assert args.checkpoint_every == 50_000
    assert args.offline_replay_ratio == pytest.approx(0.20)
    assert args.correction_fraction == pytest.approx(0.25)
    assert not hasattr(args, "hybrid_gate_every")
    assert not hasattr(args, "targeted_evaluation_every")
    assert not hasattr(args, "early_stop_consecutive_full_passes")

    suite = parse_suite_args(
        [
            "--shared-offline-checkpoint",
            "shared.pkl",
            "--manifest",
            "manifest.json",
            "--results-root",
            "smoke",
            "--component-smoke",
            "--algorithm-seeds",
            "0",
        ]
    )
    assert suite.algorithm_seeds == (0,)
    with pytest.raises(SystemExit):
        parse_suite_args(
            [
                "--shared-offline-checkpoint",
                "shared.pkl",
                "--manifest",
                "manifest.json",
                "--results-root",
                "formal",
                "--algorithm-seeds",
                "0",
            ]
        )


def _evaluation_mode(seed, mode, *, success=True, safe_termination=False):
    steps = 9 if safe_termination else 10
    safe_count = int(safe_termination)
    proposals = steps + safe_count
    intervention_rate = safe_count / proposals
    row = {
        "seed": seed,
        "mode": mode,
        "episode_index": 0,
        "policy": "learner",
        "scene_signature": f"scene-{seed}",
        "steps": steps,
        "return": 10.0,
        "success": bool(success and not safe_termination),
        "collision": False,
        "road_departure": False,
        "environment_cost": 0.0,
        "action_proposals": proposals,
        "executed_environment_steps": steps,
        "actual_intervention_rate": intervention_rate,
        "intervention_rate": intervention_rate,
        "solver_finite_rate": 1.0,
        "min_physical_body_sat_margin": 1.0,
        "q_mean": 1.0,
        "q_std": 1.0,
        "p_mean": 0.5,
        "speed_bin_counts": [0, 0, 0, 0],
        "emergency_stop_rate": 0.0,
        "inference_latency_ms_mean": 1.0,
        "inference_latency_ms_p95": 1.0,
        "guardian_certification_latency_ms_mean": 2.0,
        "guardian_certification_latency_ms_p95": 2.0,
        "certified_control_latency_ms_mean": 3.0,
        "certified_control_latency_ms_p95": 3.0,
        "flow_latent_mode": "zero",
        "actor_forward_passes": 1,
        "safe_termination": safe_termination,
        "safe_termination_reason": ("projection_failed" if safe_termination else None),
        "uncertified_action_count": safe_count,
        "terminated_without_executing_action": safe_termination,
        "error": None,
        "process_isolated": True,
        "worker_pid": 1000 + seed,
        "trajectory": None,
    }
    return _aggregate(mode, [row])


def _evaluation_payload(per_seed):
    return {
        "uncertified_proposal_handling": UNCERTIFIED_PROPOSAL_HANDLING,
        "uncertified_proposals_executed": False,
        "per_seed": per_seed,
    }


def test_all_ten_scenes_remain_in_return_and_cost_denominators():
    per_seed = {}
    for seed in PAPER_SCENE_SEEDS:
        failed = seed == 10
        ordinary = _evaluation_mode(seed, "ordinary", success=not failed)
        ordinary["collision_rate"] = float(failed)
        ordinary["rows"][0]["collision"] = failed
        ordinary["total_environment_cost"] = 10.0 if failed else 0.0
        ordinary["rows"][0]["environment_cost"] = 10.0 if failed else 0.0
        ordinary["mean_return"] = -100.0 if failed else 10.0
        ordinary["rows"][0]["return"] = -100.0 if failed else 10.0
        ordinary["min_physical_body_sat_margin"] = -1.0 if failed else 1.0
        ordinary["rows"][0]["min_physical_body_sat_margin"] = -1.0 if failed else 1.0
        hybrid = _evaluation_mode(seed, "hybrid")
        hybrid["actual_intervention_rate"] = 0.2
        hybrid["intervention_rate"] = 0.2
        hybrid["rows"][0]["actual_intervention_rate"] = 0.2
        hybrid["rows"][0]["intervention_rate"] = 0.2
        per_seed[str(seed)] = {"ordinary": ordinary, "hybrid": hybrid}
    summary = _paper_evaluation_summary(_evaluation_payload(per_seed))
    assert summary["ordinary_mean_return"] == pytest.approx(-1.0)
    assert summary["ordinary_mean_cost"] == pytest.approx(1.0)
    assert summary["ordinary_success_rate"] == pytest.approx(0.9)
    assert summary["ordinary_safe_success_rate"] == pytest.approx(0.9)
    assert summary["ordinary_collision_rate"] == pytest.approx(0.1)
    assert summary["all_scene_denominator"] is True


def test_paper_evaluation_rejects_worker_errors_and_nonfinite_metrics():
    per_seed = {}
    for seed in PAPER_SCENE_SEEDS:
        per_seed[str(seed)] = {
            mode: _evaluation_mode(seed, mode) for mode in ("ordinary", "hybrid")
        }
    evaluation = _evaluation_payload(per_seed)
    _validate_paper_evaluation(
        evaluation,
        required_modes=("ordinary", "hybrid"),
        require_value_network_metrics=True,
    )
    per_seed["10"]["ordinary"]["rows"][0]["error"] = "RuntimeError:test"
    with pytest.raises(RuntimeError, match="Seed10:ordinary:worker_error"):
        _paper_evaluation_summary(evaluation)


def test_paper_evaluation_accepts_bound_safe_termination_without_execution():
    per_seed = {
        str(seed): {
            mode: _evaluation_mode(
                seed,
                mode,
                safe_termination=(seed == 1 and mode == "hybrid"),
            )
            for mode in ("ordinary", "hybrid")
        }
        for seed in PAPER_SCENE_SEEDS
    }
    evaluation = _evaluation_payload(per_seed)
    summary = _paper_evaluation_summary(evaluation)
    assert summary["hybrid_safe_termination_rate"] == pytest.approx(0.1)
    assert summary["per_seed"]["1"]["hybrid_safe_termination"] == 1.0
    assert summary["per_seed"]["1"]["hybrid_return"] == 10.0
    assert per_seed["1"]["hybrid"]["rows"][0]["steps"] == 9
    assert per_seed["1"]["hybrid"]["rows"][0]["action_proposals"] == 10


def test_paper_evaluation_rejects_claim_that_uncertified_action_executed():
    per_seed = {
        str(seed): {
            mode: _evaluation_mode(seed, mode) for mode in ("ordinary", "hybrid")
        }
        for seed in PAPER_SCENE_SEEDS
    }
    evaluation = _evaluation_payload(per_seed)
    row = per_seed["1"]["hybrid"]["rows"][0]
    row["safe_termination"] = True
    row["uncertified_action_count"] = 1
    row["terminated_without_executing_action"] = False
    row["safe_termination_reason"] = "projection_failed"
    with pytest.raises(RuntimeError, match="safe_termination_binding"):
        _validate_paper_evaluation(
            evaluation,
            required_modes=("ordinary", "hybrid"),
            require_value_network_metrics=True,
        )


def test_teacher_policy_is_fixed_max_cruise_centreline():
    np.testing.assert_array_equal(
        _policy_action(None, "teacher", np.zeros(6, np.float32)),
        np.asarray([1.0, 0.0], np.float32),
    )


def test_teacher_aggregate_uses_json_null_for_inapplicable_value_networks():
    learner = _evaluation_mode(0, "hybrid")
    row = learner["rows"][0]
    row.update(
        policy="teacher",
        q_mean=None,
        q_std=None,
        p_mean=None,
        flow_latent_mode=None,
        actor_forward_passes=0,
    )
    teacher = _aggregate("hybrid", [row])
    assert teacher["q_mean"] is None
    assert teacher["q_std"] is None
    assert teacher["p_mean"] is None
    assert teacher["actor_forward_passes"] == 0


def test_paper_report_renders_teacher_latency_from_explicit_summary(tmp_path):
    (tmp_path / "main_table.md").write_text("| Method |\n| --- |\n", encoding="utf-8")
    methods = {
        "flow_bc": [
            {
                "config": {
                    "source_code": {
                        "git_revision": "ca168792751b56f6eedc5cc560afcf382ed53aa8"
                    }
                }
            }
        ]
    }
    teacher = {
        "teacher_guardian_certification_latency_ms_mean": 12.3454,
        "teacher_certified_control_latency_ms_mean": 23.4564,
    }
    _report(
        tmp_path,
        methods,
        [],
        [],
        tmp_path / "guardian_mpc" / "teacher_evaluation.json",
        teacher,
        True,
        {
            "observation_dim": 259,
            "action_dim": 2,
            "common_actor_q_p_hidden_sizes": [256, 256],
            "flow_teacher_hidden_sizes": [256, 256],
            "one_step_flow_actor_hidden_sizes": [256, 256],
            "flow_teacher_integration_steps": 8,
            "one_step_flow_actor_forward_passes": 1,
            "environment_backend": (
                "metadrive.envs.safe_metadrive_env.SafeMetaDriveEnv"
            ),
            "environment_factory": (
                "relax.env.drive.metasafe.make_metasafe_cbf_mpc_env"
            ),
            "action_semantics_version": "target_speed_lateral_v1",
            "guardian_environment": {
                "horizon": 2.0,
                "mpc_tracking_weight": 6.0,
                "mpc_velocity_weight": 30.0,
                "mpc_road_weight": 100.0,
                "mpc_road_buffer": 1.5,
                "mpc_max_steer_rate": 0.1,
                "cbf_gamma": 3.0,
                "sqp_steps": 12,
                "admm_steps": 120,
                "max_obstacles": 32,
            },
        },
        run_state_independence={
            "main": {
                "rows": [{"run": "main_a"}, {"run": "main_b"}],
                "expected_runs": ["main_a", "main_b"],
                "state_artifact_count": 8,
            },
            "ablations": {
                "rows": [{"run": "ablation_a"}, {"run": "ablation_b"}],
                "expected_runs": ["ablation_a", "ablation_b"],
                "state_artifact_count": 10,
            },
        },
    )
    report = (tmp_path / "EXPERIMENT_REPORT_ZH.md").read_text(encoding="utf-8")
    assert "12.345 ms/step" in report
    assert "23.456 ms/step" in report
    assert "总计 `1500` 个 episode" in report
    assert "输入观测 `259` 维、动作 `2` 维" in report
    assert "未认证 proposal 只触发安全 reset/终止并计数" in report
    assert "不构造额外复合或集成方法" in report
    assert "Ordinary/Hybrid 仅是同一 checkpoint 的两种执行评估模式" in report
    assert "## 5.1 实验设置（Experimental Setup）" in report
    assert "## 5.2 主结果（Main Results）" in report
    assert "## 5.3 学习效率（Learning Efficiency）" in report
    assert "## 5.4 Actor 表示（Actor Representation）" in report
    assert "## 5.5 最小目标消融（Ablation）" in report
    assert "两类 episode 累计计数差不超过 `1`" in report
    assert "SafeMetaDriveEnv" in report
    assert "二维动作语义为 `target_speed_lateral_v1`" in report
    assert "SQP/ADMM 步数 `12/120`" in report
    assert "主表中 Flow-Q-only 的 `±0` 只是单 run 的定义结果" in report
    assert "不能把差异分别归因给其中一项" in report
    assert "Flow learner 使用 one-step Actor 的 exact `z=0` 分支" in report
    assert "Gaussian learner 使用 squash 后的策略分布均值" in report
    assert "run-state 独立性终审：主矩阵 `2/2` 个 run、`8` 个状态文件" in report
    assert "两个消融 `2/2` 个 run、`10` 个状态文件" in report
    assert "device/inode 均不复用" in report
    assert "在线 replay、correction replay、JAX key 与 NumPy RNG" in report


def _sampling_run(method: str, seed: int, online_steps: int, *, imbalance: int = 0):
    steps = {
        str(scene): int(online_steps * (0.04 if scene < 5 else 0.16))
        for scene in PAPER_SCENE_SEEDS
    }
    online = {
        "steps_by_seed": steps,
        "transition_fraction_by_seed": {
            scene: value / online_steps for scene, value in steps.items()
        },
        "latent_episode_counts_by_seed": None,
        "latent_steps_by_seed": None,
    }
    if method != "gaussian":
        online["latent_episode_counts_by_seed"] = {
            scene: {"zero": 50 + imbalance, "random": 50} for scene in steps
        }
        online["latent_steps_by_seed"] = {
            scene: {"zero": value // 2, "random": value - value // 2}
            for scene, value in steps.items()
        }
    return {
        "path": f"/{method}/train_seed_{seed}",
        "config": {"component_smoke": False},
        "summary": {
            "algorithm_seed": seed,
            "online": online,
        },
    }


def test_final_sampling_summary_exposes_independent_run_evidence():
    evidence = _sampling_audit(
        {
            "gaussian": [_sampling_run("gaussian", 0, 1_000_000)],
            "full_flow": [_sampling_run("full_flow", 2, 1_000_000, imbalance=1)],
        },
        {"flow_no_p": [_sampling_run("flow_no_p", 1, 300_000, imbalance=1)]},
    )
    assert evidence["audit_passed"] is True
    assert evidence["formal_online_runs"] == 3
    assert evidence["maximum_observed_transition_fraction_deviation"] == 0.0
    assert evidence["maximum_flow_latent_episode_count_imbalance"] == 1
    assert evidence["independent_single_models"] is True
    assert evidence["composite_or_ensemble"] is False
    gaussian = next(row for row in evidence["rows"] if row["method"] == "gaussian")
    assert gaussian["latent_episode_and_step_evidence_by_seed"] is None
    assert all(
        row["independent_single_model"] is True
        and row["composite_or_ensemble"] is False
        for row in evidence["rows"]
    )


def test_final_sampling_summary_rejects_consistent_schedule_drift():
    run = _sampling_run("full_flow", 0, 1_000_000)
    online = run["summary"]["online"]
    online["steps_by_seed"].update({"8": 155_999, "10": 164_001})
    online["transition_fraction_by_seed"].update({"8": 0.155999, "10": 0.164001})
    online["latent_steps_by_seed"]["8"].update({"zero": 77_999, "random": 78_000})
    online["latent_steps_by_seed"]["10"].update({"zero": 82_000, "random": 82_001})
    with pytest.raises(RuntimeError, match="sampling audit schedule drift"):
        _sampling_audit({"full_flow": [run]}, {})


def test_paper_report_renders_actual_sampling_audit(tmp_path):
    (tmp_path / "main_table.md").write_text("| Method |\n| --- |\n", encoding="utf-8")
    methods = {
        "full_flow": [
            {
                "config": {
                    "source_code": {
                        "git_revision": "ca168792751b56f6eedc5cc560afcf382ed53aa8"
                    }
                }
            }
        ]
    }
    sampling = {
        "rows": [
            {
                "method": "full_flow",
                "algorithm_seed": 2,
                "online_steps": 1_000_000,
                "maximum_transition_fraction_deviation": 0.001234,
                "maximum_flow_latent_episode_count_imbalance": 1,
            }
        ],
        "maximum_observed_transition_fraction_deviation": 0.001234,
        "maximum_flow_latent_episode_count_imbalance": 1,
    }
    _report(
        tmp_path,
        methods,
        [],
        [],
        tmp_path / "guardian_mpc" / "teacher_evaluation.json",
        {
            "teacher_guardian_certification_latency_ms_mean": 1.0,
            "teacher_certified_control_latency_ms_mean": 2.0,
        },
        False,
        {
            "observation_dim": 259,
            "action_dim": 2,
            "common_actor_q_p_hidden_sizes": [256, 256],
            "flow_teacher_hidden_sizes": [256, 256],
            "one_step_flow_actor_hidden_sizes": [256, 256],
            "flow_teacher_integration_steps": 8,
            "one_step_flow_actor_forward_passes": 1,
            "environment_backend": "metadrive.envs.safe_metadrive_env.SafeMetaDriveEnv",
            "environment_factory": "relax.env.drive.metasafe.make_metasafe_cbf_mpc_env",
            "action_semantics_version": "target_speed_lateral_v1",
            "guardian_environment": {
                "horizon": 2.0,
                "mpc_tracking_weight": 6.0,
                "mpc_velocity_weight": 30.0,
                "mpc_road_weight": 100.0,
                "mpc_road_buffer": 1.5,
                "mpc_max_steer_rate": 0.1,
                "cbf_gamma": 3.0,
                "sqp_steps": 12,
                "admm_steps": 120,
                "max_obstacles": 32,
            },
        },
        sampling_audit=sampling,
    )
    report = (tmp_path / "EXPERIMENT_REPORT_ZH.md").read_text(encoding="utf-8")
    assert "最终 sampling 审计逐 run 明细" in report
    assert "| full_flow | 2 | 1000000 | 0.001234 | 1 | 是 |" in report
    assert "实际最大 transition 比例偏差为 `0.001234`" in report
    assert "实际最大累计计数差为 `1`" in report


def test_corrupt_latest_checkpoint_falls_back_to_same_step_file(tmp_path):
    algorithm = _flow()
    algorithm.freeze_bc_flow()
    algorithm.prepare_online_training()
    permanent = tmp_path / "checkpoint_00000000.pkl"
    latest = tmp_path / "checkpoint_latest.pkl"
    algorithm.save(str(permanent))
    algorithm.save(str(latest))
    latest.write_bytes(b"not-a-pickle")
    restored = _flow()
    selected = _load_numerical_checkpoint(restored, tmp_path, step=0)
    assert selected == permanent
    assert _tree_equal(restored.state, algorithm.state)


def test_atomic_resume_uses_sha_bound_backup_and_suite_detects_it(tmp_path):
    algorithm = _flow()
    algorithm.freeze_bc_flow()
    algorithm.prepare_online_training()
    permanent = tmp_path / "checkpoint_00050000.pkl"
    backup = tmp_path / "checkpoint_latest.pkl"
    algorithm.save(str(permanent))
    backup.write_bytes(permanent.read_bytes())
    digest = hashlib.sha256(permanent.read_bytes()).hexdigest()
    control = {
        "format": "guardian_flow_paper_resume_v2_atomic",
        "algorithm_checkpoint": permanent.name,
        "algorithm_checkpoint_backup": backup.name,
        "algorithm_checkpoint_sha256": digest,
        "algorithm_checkpoint_backup_sha256": digest,
    }
    with (tmp_path / "trainer_state_latest.pkl").open("wb") as stream:
        pickle.dump(control, stream)
    (tmp_path / "resume_index.json").write_text(
        json.dumps(
            {
                "format": control["format"],
                "trainer_state": "trainer_state_latest.pkl",
                "algorithm_checkpoint": permanent.name,
                "algorithm_checkpoint_backup": backup.name,
                "replay_states_embedded_atomically": True,
            }
        ),
        encoding="utf-8",
    )
    assert _resumable(tmp_path, "full_flow") is True

    permanent.write_bytes(b"corrupt-primary")
    restored = _flow()
    selected = _load_atomic_resume_algorithm(restored, tmp_path, control)
    assert selected == backup
    assert hashlib.sha256(permanent.read_bytes()).hexdigest() == digest
    assert _tree_equal(restored.state, algorithm.state)

    offline = tmp_path / "offline_only"
    offline.mkdir()
    (offline / "offline_summary.json").write_text("{}", encoding="utf-8")
    (offline / "offline_flow_bc.pkl").write_bytes(b"offline")
    assert _resumable(offline, "flow_bc") is True


def test_atomic_trainer_loader_accepts_v3_index_and_rejects_sha_mismatch(tmp_path):
    control = {
        "format": "guardian_flow_paper_resume_v3_atomic",
        "step": 350_000,
        "algorithm_checkpoint": "checkpoint_00350000.pkl",
        "algorithm_checkpoint_sha256": "a" * 64,
        "algorithm_checkpoint_backup": "checkpoint_latest.pkl",
        "algorithm_checkpoint_backup_sha256": "a" * 64,
    }
    trainer = tmp_path / "trainer_state_00350000_test.pkl"
    with trainer.open("wb") as stream:
        pickle.dump(control, stream)
    digest = hashlib.sha256(trainer.read_bytes()).hexdigest()
    index = {
        **control,
        "trainer_state": trainer.name,
        "trainer_state_sha256": digest,
        "replay_states_embedded_atomically": True,
    }
    (tmp_path / "resume_index.json").write_text(json.dumps(index), encoding="utf-8")
    assert _load_atomic_trainer_control(tmp_path) == control

    index["trainer_state_sha256"] = "b" * 64
    (tmp_path / "resume_index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA256 binding"):
        _load_atomic_trainer_control(tmp_path)
