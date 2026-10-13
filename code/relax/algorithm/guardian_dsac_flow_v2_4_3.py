"""Guardian-Flow-DSAC V2.4.3 Safe-Memory.

V2.4.3 keeps the V2.4.2 Q/P learners, Flow architecture, Guardian correction,
speed contract and episode horizon unchanged.  It replaces drifting-state
parameter-reference consolidation with certified Ordinary trajectory memory
and adds Seed9 pre-intervention teacher supervision for the deterministic
``z=0`` deployment branch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.guardian_dsac_flow_v2_4 import (
    _python_metrics,
    _weighted_mean,
)
from relax.algorithm.guardian_dsac_flow_v2_4_1 import (
    GuardianDSACFlowV241,
    _cosine_rate,
)
from relax.algorithm.guardian_dsac_flow_v2_4_2 import (
    GuardianDSACFlowV242,
    GuardianDSACFlowV242Config,
    _parameter_sha256,
)
from relax.algorithm.guardian_dsac_v2_3 import (
    proxy_actor_objective,
    weighted_action_proximal,
)
from relax.utils.flow_v2_4_3 import (
    PRE_INTERVENTION_BATCH_SIZE,
    PRE_INTERVENTION_CAPACITY,
    PRE_INTERVENTION_HORIZON,
    PRE_INTERVENTION_TAU,
    SAFE_MEMORY_BATCH_SIZE,
    PreInterventionReplay,
    SupervisionBatch,
    array_sha256,
    balanced_safe_memory_sample,
    safe_memory_sha256,
    seed9_pre_intervention_batch,
)
from relax.utils.guardian_deployment_gate import ordinary_hard_safety_failures


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GuardianDSACFlowV243Config(GuardianDSACFlowV242Config):
    """Fail-closed V2.4.3 loss and memory binding."""

    skill_retention_weight: float = 0.5
    safe_memory_batch_size: int = SAFE_MEMORY_BATCH_SIZE
    pre_intervention_weight: float = 1.0
    pre_intervention_batch_size: int = PRE_INTERVENTION_BATCH_SIZE
    pre_intervention_capacity: int = PRE_INTERVENTION_CAPACITY
    pre_intervention_horizon: int = PRE_INTERVENTION_HORIZON
    pre_intervention_tau: float = PRE_INTERVENTION_TAU

    def __post_init__(self) -> None:
        super().__post_init__()
        exact = {
            "skill_retention_weight": (self.skill_retention_weight, 0.5),
            "safe_memory_batch_size": (
                int(self.safe_memory_batch_size),
                SAFE_MEMORY_BATCH_SIZE,
            ),
            "pre_intervention_weight": (self.pre_intervention_weight, 1.0),
            "pre_intervention_batch_size": (
                int(self.pre_intervention_batch_size),
                PRE_INTERVENTION_BATCH_SIZE,
            ),
            "pre_intervention_capacity": (
                int(self.pre_intervention_capacity),
                PRE_INTERVENTION_CAPACITY,
            ),
            "pre_intervention_horizon": (
                int(self.pre_intervention_horizon),
                PRE_INTERVENTION_HORIZON,
            ),
            "pre_intervention_tau": (
                self.pre_intervention_tau,
                PRE_INTERVENTION_TAU,
            ),
        }
        mismatches = [
            name for name, (observed, expected) in exact.items() if observed != expected
        ]
        if mismatches:
            raise ValueError(
                "V2.4.3 Safe-Memory binding mismatch: " + ", ".join(mismatches)
            )


class GuardianDSACFlowV243(GuardianDSACFlowV242):
    """V2.4.2 learner with certified closed-loop capability consolidation."""

    FORMAT = "guardian_dsac_one_step_flow_v2_4_3_safe_memory_v1"
    REPLAY_SCHEMA = "guardian_flow_dsac_v2_4_3_experience_v1"
    SAFE_MEMORY_FORMAT = "guardian_flow_v2_4_3_safe_trajectory_memory_v1"

    def __init__(
        self,
        base_algorithm,
        flow_policy,
        flow_params,
        *,
        config: GuardianDSACFlowV243Config = GuardianDSACFlowV243Config(),
    ):
        super().__init__(base_algorithm, flow_policy, flow_params, config=config)
        self.config = config
        # V2.4.2 parameter references are intentionally not used by this class.
        self.ordinary_safe_references = {}
        self.ordinary_safe_reference_history = []
        self.skill_reference_cursor = 0
        self.safe_trajectory_memory: dict[int, dict[str, object]] = {}
        self.safe_trajectory_memory_history: list[dict[str, object]] = []
        self.pre_intervention_replay = PreInterventionReplay(
            self.flow_policy.actor_obs_dim,
            self.flow_policy.act_dim,
            config.pre_intervention_capacity,
        )
        self.pre_intervention_episode_events: list[dict[str, object]] = []

    def _build_long_run_actor_update(self) -> None:
        config = self.config
        optimizer = self.online_actor_optimizer
        retention_seeds = tuple(int(seed) for seed in config.retention_teacher_seeds)
        correction_axis_weight = jnp.asarray(
            [config.correction_speed_weight, config.correction_lateral_weight],
            dtype=jnp.float32,
        )

        def online_actor_update(
            key,
            state,
            data,
            q1,
            q2,
            proxy,
            learning_rate,
            safe_observation,
            safe_target_action,
            safe_support,
            pre_observation,
            pre_target_action,
            pre_weight,
            pre_support,
        ):
            action_shape = jnp.asarray(data.action).shape
            (
                policy_key,
                diversity_1_key,
                diversity_2_key,
                retention_key,
            ) = jax.random.split(key, 4)
            noise = jax.random.normal(policy_key, action_shape, dtype=jnp.float32)
            diversity_noise_1 = jax.random.normal(
                diversity_1_key, action_shape, dtype=jnp.float32
            )
            diversity_noise_2 = jax.random.normal(
                diversity_2_key, action_shape, dtype=jnp.float32
            )
            retention_noise = jax.random.normal(
                retention_key, action_shape, dtype=jnp.float32
            )
            actor_support = jnp.asarray(data.actor_controlled, dtype=jnp.float32)
            proxy_support = jnp.maximum(
                jnp.asarray(data.guardian_dependence_target_valid, jnp.float32),
                jnp.asarray(data.guardian_dependence_corrected_valid, jnp.float32),
            )
            safe_action = jax.lax.stop_gradient(
                self.flow_policy.get_action(state.safe_one_step, data.obs, noise)
            )
            coefficient = jnp.maximum(
                config.proximal_weight_min,
                config.proximal_weight_initial
                * jnp.exp(
                    -jnp.asarray(state.proximal_step, dtype=jnp.float32)
                    / float(max(config.proximal_decay_steps, 1))
                ),
            )
            hard = jnp.asarray(data.intervention_hard, dtype=bool)
            soft = jnp.asarray(data.intervention_soft, dtype=bool)
            modified = jnp.asarray(data.actual_control_modified, dtype=bool)
            latent_valid = jnp.asarray(data.flow_noise_valid, dtype=bool)
            zero_latent_row = jnp.asarray(data.flow_zero_latent, dtype=bool)
            correction_support = modified & latent_valid & (hard | soft)
            correction_sample_weight = jnp.where(
                hard,
                1.0,
                jnp.where(soft, 0.5, 0.0),
            ).astype(jnp.float32) * correction_support.astype(jnp.float32)
            pair_valid = jnp.asarray(data.proxy_pair_valid, dtype=bool)
            correction_target = jnp.where(
                pair_valid[..., None],
                jnp.asarray(data.proxy_corrected_action, dtype=jnp.float32),
                jnp.asarray(data.action, dtype=jnp.float32),
            )
            stored_noise = jnp.asarray(data.flow_noise, dtype=jnp.float32)
            scene_seed = jnp.asarray(data.scene_seed, dtype=jnp.int32)
            seed9 = scene_seed == 9
            correction_scale = jnp.where(
                seed9,
                config.seed9_correction_weight,
                config.correction_weight,
            ).astype(jnp.float32)
            zero_correction_scale = jnp.where(
                seed9,
                config.seed9_zero_latent_correction_weight,
                config.zero_latent_correction_weight,
            ).astype(jnp.float32)
            retention_seed = jnp.zeros_like(scene_seed, dtype=bool)
            for seed in retention_seeds:
                retention_seed = retention_seed | (scene_seed == seed)
            retention_support = (~latent_valid) & retention_seed
            retention_weight = retention_support.astype(jnp.float32)
            retention_teacher = jax.lax.stop_gradient(
                self.flow_policy.bc_action(
                    state.params.bc_flow,
                    data.obs,
                    retention_noise,
                )
            )
            safe_observation = jnp.asarray(safe_observation, dtype=jnp.float32)
            safe_target_action = jax.lax.stop_gradient(
                jnp.asarray(safe_target_action, dtype=jnp.float32)
            )
            safe_support = jnp.asarray(safe_support, dtype=jnp.float32)
            pre_observation = jnp.asarray(pre_observation, dtype=jnp.float32)
            pre_target_action = jax.lax.stop_gradient(
                jnp.asarray(pre_target_action, dtype=jnp.float32)
            )
            pre_weight = jnp.asarray(pre_weight, dtype=jnp.float32)
            pre_support = jnp.asarray(pre_support, dtype=jnp.float32)

            def loss_fn(params):
                action = self.flow_policy.get_action(params, data.obs, noise)
                mean_1, _ = self.base.agent.q_distribution(q1, data.obs, action)
                mean_2, _ = self.base.agent.q_distribution(q2, data.obs, action)
                q_loss = -_weighted_mean(
                    jnp.minimum(mean_1, mean_2), actor_support
                )
                proxy_logit = self.base.agent.proxy_value(proxy, data.obs, action)
                p_probability = jax.nn.sigmoid(proxy_logit)
                p_loss = _weighted_mean(
                    proxy_actor_objective(
                        proxy_logit,
                        objective=self.base.config.proxy_actor_objective,
                    ),
                    proxy_support,
                )
                delta = action - safe_action
                proximal, speed_drift, lateral_drift = weighted_action_proximal(
                    delta,
                    speed_weight=config.proximal_speed_weight,
                    lateral_weight=config.proximal_lateral_weight,
                )
                stored_prediction = self.flow_policy.get_action(
                    params, data.obs, stored_noise
                )
                zero_prediction = self.flow_policy.get_action(
                    params,
                    data.obs,
                    jnp.zeros_like(stored_noise),
                )
                target = jax.lax.stop_gradient(correction_target)
                stored_square = jnp.square(stored_prediction - target)
                zero_square = jnp.square(zero_prediction - target)
                stored_error = jnp.sum(
                    stored_square * correction_axis_weight, axis=-1
                )
                zero_error = jnp.sum(
                    zero_square * correction_axis_weight, axis=-1
                )
                stored_correction = _weighted_mean(
                    stored_error, correction_sample_weight
                )
                zero_correction = _weighted_mean(
                    zero_error, correction_sample_weight
                )
                correction = _weighted_mean(
                    correction_scale
                    * (stored_error + zero_correction_scale * zero_error),
                    correction_sample_weight,
                )
                retention_action = self.flow_policy.get_action(
                    params, data.obs, retention_noise
                )
                retention_square = jnp.sum(
                    jnp.square(retention_action - retention_teacher), axis=-1
                )
                retention = _weighted_mean(retention_square, retention_weight)
                safe_prediction = self.flow_policy.get_action(
                    params,
                    safe_observation,
                    jnp.zeros_like(safe_target_action),
                )
                safe_square = jnp.sum(
                    jnp.square(safe_prediction - safe_target_action), axis=-1
                )
                safe_memory = _weighted_mean(safe_square, safe_support)
                pre_prediction = self.flow_policy.get_action(
                    params,
                    pre_observation,
                    jnp.zeros_like(pre_target_action),
                )
                pre_square = jnp.sum(
                    jnp.square(pre_prediction - pre_target_action), axis=-1
                )
                pre_intervention = jnp.sum(
                    pre_square * pre_weight * pre_support
                ) / jnp.maximum(
                    jnp.sum(pre_support),
                    jnp.asarray(1.0, dtype=jnp.float32),
                )
                diversity_action_1 = self.flow_policy.get_action(
                    params, data.obs, diversity_noise_1
                )
                diversity_action_2 = self.flow_policy.get_action(
                    params, data.obs, diversity_noise_2
                )
                diversity = jnp.mean(
                    jnp.linalg.norm(
                        diversity_action_1 - diversity_action_2,
                        axis=-1,
                    )
                )
                total = (
                    config.q_actor_weight * q_loss
                    + config.proxy_actor_weight * p_loss
                    + coefficient * proximal
                    + correction
                    + config.retention_teacher_weight * retention
                    + config.skill_retention_weight * safe_memory
                    + config.pre_intervention_weight * pre_intervention
                )
                return total, (
                    q_loss,
                    p_loss,
                    p_probability,
                    proximal,
                    speed_drift,
                    lateral_drift,
                    correction,
                    stored_correction,
                    zero_correction,
                    stored_square,
                    zero_square,
                    retention,
                    safe_memory,
                    pre_intervention,
                    diversity,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params.one_step
            )
            adam_updates, opt_state = optimizer.update(
                grads,
                state.actor_opt_state,
                state.params.one_step,
            )
            updates = jax.tree_util.tree_map(
                lambda value: -jnp.asarray(learning_rate, jnp.float32) * value,
                adam_updates,
            )
            params = optax.apply_updates(state.params.one_step, updates)
            target = optax.incremental_update(
                params, state.target_one_step, config.target_actor_tau
            )
            (
                q_loss,
                p_loss,
                p_probability,
                proximal,
                speed_drift,
                lateral_drift,
                correction,
                stored_correction,
                zero_correction,
                stored_square,
                zero_square,
                retention,
                safe_memory,
                pre_intervention,
                diversity,
            ) = aux
            correction_weight_mean = _weighted_mean(
                correction_scale, correction_sample_weight
            )
            zero_coefficient_mean = _weighted_mean(
                zero_correction_scale, correction_sample_weight
            )
            return state._replace(
                params=state.params._replace(one_step=params),
                actor_opt_state=opt_state,
                target_one_step=target,
                gradient_step=state.gradient_step + 1,
                proximal_step=state.proximal_step + 1,
            ), {
                "flow_actor_loss": loss,
                "flow_q_loss": q_loss,
                "flow_proxy_loss": p_loss,
                "flow_absolute_proxy_loss": _weighted_mean(
                    p_probability, proxy_support
                ),
                "flow_proximal_loss": proximal,
                "flow_proximal_speed_drift": speed_drift,
                "flow_proximal_lateral_drift": lateral_drift,
                "flow_proximal_coefficient": coefficient,
                "flow_correction_loss": correction,
                "flow_correction_stored_loss": stored_correction,
                "flow_correction_zero_latent_loss": zero_correction,
                "flow_correction_unweighted_loss": (
                    stored_correction + zero_correction
                ),
                "flow_correction_effective_weight_mean": correction_weight_mean,
                "flow_zero_correction_coefficient_mean": zero_coefficient_mean,
                "flow_correction_support_fraction": jnp.mean(
                    correction_support.astype(jnp.float32)
                ),
                "flow_correction_seed9_support_fraction": jnp.mean(
                    (correction_support & seed9).astype(jnp.float32)
                ),
                "flow_correction_weight_mean": jnp.mean(correction_sample_weight),
                "flow_correction_error_speed": _weighted_mean(
                    stored_square[..., 0], correction_sample_weight
                ),
                "flow_correction_error_lateral": _weighted_mean(
                    stored_square[..., 1], correction_sample_weight
                ),
                "flow_zero_correction_error_speed": _weighted_mean(
                    zero_square[..., 0], correction_sample_weight
                ),
                "flow_zero_correction_error_lateral": _weighted_mean(
                    zero_square[..., 1], correction_sample_weight
                ),
                "flow_correction_proxy_target_fraction": _weighted_mean(
                    pair_valid.astype(jnp.float32), correction_sample_weight
                ),
                "flow_retention_loss": retention,
                "flow_retention_support_fraction": jnp.mean(retention_weight),
                "flow_safe_memory_loss": safe_memory,
                "flow_safe_memory_support_fraction": jnp.mean(safe_support),
                "flow_pre_intervention_loss": pre_intervention,
                "flow_pre_intervention_support_fraction": jnp.mean(pre_support),
                "flow_pre_intervention_weight_mean": _weighted_mean(
                    pre_weight, pre_support
                ),
                "flow_zero_latent_batch_fraction": jnp.mean(
                    (zero_latent_row & latent_valid).astype(jnp.float32)
                ),
                "flow_latent_action_diversity": diversity,
                "flow_actor_learning_rate": learning_rate,
            }

        self._online_actor_update_long_run = jax.jit(online_actor_update)

    def _supervision_arrays(
        self,
        batch: SupervisionBatch | None,
        *,
        kind: str,
        count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_dim = int(self.flow_policy.actor_obs_dim)
        action_dim = int(self.flow_policy.act_dim)
        if batch is None:
            return (
                np.zeros((count, obs_dim), dtype=np.float32),
                np.zeros((count, action_dim), dtype=np.float32),
                np.zeros(count, dtype=np.float32),
                np.zeros(count, dtype=np.float32),
            )
        observations = np.asarray(batch.obs, dtype=np.float32)
        actions = np.asarray(batch.action, dtype=np.float32)
        weights = np.asarray(batch.weight, dtype=np.float32)
        seeds = np.asarray(batch.scene_seed, dtype=np.int32)
        lags = np.asarray(batch.lag, dtype=np.int16)
        if (
            observations.shape != (count, obs_dim)
            or actions.shape != (count, action_dim)
            or weights.shape != (count,)
            or seeds.shape != (count,)
            or lags.shape != (count,)
        ):
            raise ValueError(f"{kind} supervision batch shape mismatch")
        if not (
            np.all(np.isfinite(observations))
            and np.all(np.isfinite(actions))
            and np.all(np.isfinite(weights))
        ):
            raise ValueError(f"{kind} supervision contains non-finite values")
        if kind == "safe-memory":
            expected = set(self.safe_trajectory_memory)
            observed = set(int(seed) for seed in np.unique(seeds))
            seed_counts = np.unique(seeds, return_counts=True)[1]
            if (
                not expected
                or observed != expected
                or int(np.max(seed_counts) - np.min(seed_counts)) > 1
                or not np.all(weights == 1.0)
                or not np.all(lags == -1)
            ):
                raise ValueError("safe-memory batch is not balanced over active Seeds")
        elif kind == "pre-intervention":
            expected_weights = np.exp(
                -lags.astype(np.float32) / self.config.pre_intervention_tau
            )
            if (
                not np.all(seeds == 9)
                or np.any(lags < 0)
                or np.any(lags > self.config.pre_intervention_horizon)
                or not np.allclose(weights, expected_weights, rtol=1e-6, atol=1e-7)
            ):
                raise ValueError("pre-intervention batch violates Seed9 lag binding")
        else:
            raise ValueError(f"unknown supervision kind {kind}")
        return observations, actions, weights, np.ones(count, dtype=np.float32)

    def online_joint_update(
        self,
        key,
        data,
        *,
        actor_data=None,
        update_actor: bool = True,
        safe_memory_data: SupervisionBatch | None = None,
        pre_intervention_data: SupervisionBatch | None = None,
    ):
        if not self.bc_flow_frozen:
            raise RuntimeError("online training requires a frozen Flow teacher")
        if not self.online_schedule_prepared:
            raise RuntimeError("prepare V2.4.3 online schedules before training")
        critic_key, next_key, current_key, actor_key = jax.random.split(key, 4)
        next_noise = jax.random.normal(
            next_key, jnp.asarray(data.action).shape, dtype=jnp.float32
        )
        current_noise = jax.random.normal(
            current_key, jnp.asarray(data.action).shape, dtype=jnp.float32
        )
        next_action = self.flow_policy.get_action(
            self.state.target_one_step, data.next_obs, next_noise
        )
        policy_action = self.flow_policy.get_action(
            self.state.params.one_step, data.obs, current_noise
        )
        critic_lr = float(
            np.asarray(self._critic_learning_rate(self.online_critic_updates_total))
        )
        proxy_lr = float(
            np.asarray(self._proxy_learning_rate(self.online_critic_updates_total))
        )
        q_metrics = self.base.online_critic_update_with_policy_actions(
            critic_key,
            data,
            next_action=next_action,
            policy_action=policy_action,
        )
        p_metrics = self.base.proxy_update(data)
        self.online_critic_updates_total += 1
        metrics = {
            **{f"task_{name}": value for name, value in q_metrics.items()},
            **p_metrics,
            "flow_implicit_entropy_target": 0.0,
            "flow_actor_entropy_free": 1.0,
            "flow_bc_frozen": 1.0,
            "task_critic_learning_rate": critic_lr,
            "proxy_learning_rate": proxy_lr,
        }
        if update_actor:
            actor_batch = data if actor_data is None else actor_data
            safe_obs, safe_action, _, safe_support = self._supervision_arrays(
                safe_memory_data,
                kind="safe-memory",
                count=self.config.safe_memory_batch_size,
            )
            pre_obs, pre_action, pre_weight, pre_support = (
                self._supervision_arrays(
                    pre_intervention_data,
                    kind="pre-intervention",
                    count=self.config.pre_intervention_batch_size,
                )
            )
            actor_lr = _cosine_rate(
                self.config.actor_learning_rate_initial,
                self.config.actor_learning_rate_final,
                self.online_actor_updates_total,
                self.config.actor_learning_rate_decay_steps,
            )
            self.state, actor_metrics = self._online_actor_update_long_run(
                actor_key,
                self.state,
                actor_batch,
                self.base.state.params.q1,
                self.base.state.params.q2,
                self.base.state.params.proxy,
                jnp.asarray(actor_lr, dtype=jnp.float32),
                safe_obs,
                safe_action,
                safe_support,
                pre_obs,
                pre_action,
                pre_weight,
                pre_support,
            )
            self.online_actor_updates_total += 1
            metrics.update(_python_metrics(actor_metrics))
            metrics["flow_safe_memory_active_seeds"] = int(
                len(self.safe_trajectory_memory)
            )
            metrics["flow_pre_intervention_replay_size"] = int(
                len(self.pre_intervention_replay)
            )
        return metrics

    def sample_safe_trajectory_memory(
        self, batch_size: int, rng: np.random.Generator
    ) -> SupervisionBatch | None:
        if not self.safe_trajectory_memory:
            return None
        return balanced_safe_memory_sample(
            self.safe_trajectory_memory, batch_size, rng
        )

    def sample_pre_intervention_replay(
        self, batch_size: int, rng: np.random.Generator
    ) -> SupervisionBatch | None:
        if len(self.pre_intervention_replay) == 0:
            return None
        return self.pre_intervention_replay.sample(batch_size, rng)

    def add_seed9_pre_intervention_episode(
        self, episode, *, online_step: int
    ) -> dict[str, object]:
        batch = seed9_pre_intervention_batch(
            episode,
            horizon=self.config.pre_intervention_horizon,
            tau=self.config.pre_intervention_tau,
        )
        added = self.pre_intervention_replay.add_batch(batch)
        event = {
            "online_step": int(online_step),
            "episode_rows": int(len(episode)),
            "added_rows": int(added),
            "replay_size": int(len(self.pre_intervention_replay)),
            "replay_total_added": int(self.pre_intervention_replay.total_added),
        }
        self.pre_intervention_episode_events.append(event)
        return event

    def _load_evaluated_trajectory(
        self,
        evaluation,
        ordinary,
        *,
        seed: int,
        actor_parameter_sha256: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        rows = ordinary.get("rows", [])
        if int(ordinary.get("episodes", -1)) != 1 or len(rows) != 1:
            raise ValueError("Safe-Memory requires exactly one Ordinary episode")
        row = rows[0]
        artifact = row.get("trajectory")
        if not isinstance(artifact, dict):
            raise ValueError(f"Seed{seed} Ordinary evaluation has no trajectory artifact")
        path = Path(str(artifact.get("path", ""))).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_file_hash = _file_sha256(path)
        if observed_file_hash != artifact.get("sha256"):
            raise ValueError(f"Seed{seed} trajectory file hash mismatch")
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"observations", "actions", "metadata_json"}:
                raise ValueError("trajectory artifact schema mismatch")
            observations = np.asarray(payload["observations"], dtype=np.float32)
            actions = np.asarray(payload["actions"], dtype=np.float32)
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        expected_checkpoint_hash = str(evaluation.get("checkpoint_sha256", ""))
        expected_signature = str(row.get("scene_signature", ""))
        expected_rows = int(row.get("steps", -1))
        observation_hash = array_sha256(observations)
        action_hash = array_sha256(actions)
        binding_ok = bool(
            evaluation.get("format") == "guardian_dsac_isolated_evaluation_v1"
            and evaluation.get("policy") == "learner"
            and evaluation.get("fresh_process_per_episode") is True
            and metadata.get("format")
            == "guardian_flow_v2_4_3_ordinary_trajectory_v1"
            and artifact.get("format")
            == "guardian_flow_v2_4_3_ordinary_trajectory_v1"
            and int(metadata.get("seed", -1)) == seed
            and metadata.get("mode") == "ordinary"
            and metadata.get("policy") == "learner"
            and int(metadata.get("episode_index", -1)) == 0
            and metadata.get("scene_signature") == expected_signature
            and metadata.get("checkpoint_sha256") == expected_checkpoint_hash
            and artifact.get("checkpoint_sha256") == expected_checkpoint_hash
            and metadata.get("actor_parameter_sha256")
            == actor_parameter_sha256
            and artifact.get("actor_parameter_sha256")
            == actor_parameter_sha256
            and metadata.get("worker_backend") == "cpu"
            and artifact.get("worker_backend") == "cpu"
            and metadata.get("action_generation_mode")
            == "single_observation_live_z0"
            and artifact.get("action_generation_mode")
            == "single_observation_live_z0"
            and metadata.get("action_self_check_mode")
            == "repeat_single_observation_live_z0_same_cpu_worker"
            and artifact.get("action_self_check_mode")
            == "repeat_single_observation_live_z0_same_cpu_worker"
            and metadata.get("action_self_check_exact") is True
            and artifact.get("action_self_check_exact") is True
            and float(metadata.get("action_self_check_max_abs_error", -1.0))
            == 0.0
            and float(artifact.get("action_self_check_max_abs_error", -1.0))
            == 0.0
            and float(metadata.get("action_self_check_mean_abs_error", -1.0))
            == 0.0
            and float(artifact.get("action_self_check_mean_abs_error", -1.0))
            == 0.0
            and metadata.get("observation_sha256") == observation_hash
            and artifact.get("observation_sha256") == observation_hash
            and metadata.get("action_sha256") == action_hash
            and artifact.get("action_sha256") == action_hash
            and int(artifact.get("rows", -1)) == expected_rows
        )
        if not binding_ok:
            raise ValueError(f"Seed{seed} trajectory metadata binding mismatch")
        if (
            observations.shape
            != (expected_rows, int(self.flow_policy.actor_obs_dim))
            or actions.shape != (expected_rows, int(self.flow_policy.act_dim))
            or expected_rows < 1
        ):
            raise ValueError(f"Seed{seed} trajectory array shape mismatch")
        if not (
            np.all(np.isfinite(observations))
            and np.all(np.isfinite(actions))
            and np.all(actions >= -1.0)
            and np.all(actions <= 1.0)
        ):
            raise ValueError(f"Seed{seed} trajectory contains invalid values")
        return observations, actions, {
            "trajectory_path": str(path),
            "trajectory_file_sha256": observed_file_hash,
            "evaluation_checkpoint_sha256": expected_checkpoint_hash,
            "scene_signature": expected_signature,
            "rows": expected_rows,
            "trajectory_actor_parameter_sha256": actor_parameter_sha256,
            "trajectory_worker_backend": "cpu",
            "action_verification_mode": (
                "same_cpu_worker_bitwise_exact_single_observation_live_z0"
            ),
            "action_self_check_max_abs_error": 0.0,
            "action_self_check_mean_abs_error": 0.0,
            "observation_sha256": observation_hash,
            "action_sha256": action_hash,
        }

    def consider_safe_trajectory_memory(
        self,
        evaluation,
        *,
        step: int,
        label: str,
    ) -> list[dict[str, object]]:
        """Admit only certified Ordinary rollouts and keep the best-safe one."""

        per_seed = evaluation.get("per_seed", {})
        actor_hash = _parameter_sha256(self.state.params.one_step)
        events: list[dict[str, object]] = []
        for seed_text in sorted(per_seed, key=lambda value: int(value)):
            seed = int(seed_text)
            if seed not in range(10):
                raise ValueError(f"unbound Safe-Memory Seed {seed}")
            modes = per_seed[seed_text]
            if "ordinary" not in modes:
                raise ValueError(f"Seed{seed} evaluation has no Ordinary row")
            ordinary = modes["ordinary"]
            failures = ordinary_hard_safety_failures(ordinary)
            ordinary_return = float(ordinary["mean_return"])
            if not np.isfinite(ordinary_return):
                failures = sorted(set([*failures, "non_finite_return"]))
            previous = self.safe_trajectory_memory.get(seed)
            decision = "unsafe"
            candidate_metadata = None
            if not failures:
                observations, actions, candidate_metadata = (
                    self._load_evaluated_trajectory(
                        evaluation,
                        ordinary,
                        seed=seed,
                        actor_parameter_sha256=actor_hash,
                    )
                )
                if previous is None:
                    decision = "first_ordinary_safe_trajectory"
                elif ordinary_return > float(previous["ordinary_return"]):
                    decision = "higher_return_ordinary_safe_trajectory"
                else:
                    decision = "kept_higher_return_safe_trajectory"
                if decision != "kept_higher_return_safe_trajectory":
                    self.safe_trajectory_memory[seed] = {
                        "observations": observations.copy(),
                        "actions": actions.copy(),
                        "ordinary_return": ordinary_return,
                        "ordinary_min_physical_margin": float(
                            ordinary["min_physical_body_sat_margin"]
                        ),
                        "step": int(step),
                        "label": str(label),
                        "actor_parameter_sha256": actor_hash,
                        **candidate_metadata,
                    }
            selected = self.safe_trajectory_memory.get(seed)
            event = {
                "seed": seed,
                "step": int(step),
                "label": str(label),
                "ordinary_safe": not failures,
                "ordinary_failures": failures,
                "ordinary_return": ordinary_return,
                "previous_memory_return": (
                    None if previous is None else float(previous["ordinary_return"])
                ),
                "decision": decision,
                "evaluated_actor_parameter_sha256": actor_hash,
                "candidate_trajectory_file_sha256": (
                    None
                    if candidate_metadata is None
                    else candidate_metadata["trajectory_file_sha256"]
                ),
                "selected_memory_return": (
                    None if selected is None else float(selected["ordinary_return"])
                ),
                "selected_trajectory_file_sha256": (
                    None
                    if selected is None
                    else selected["trajectory_file_sha256"]
                ),
                "selected_observation_sha256": (
                    None if selected is None else selected["observation_sha256"]
                ),
                "selected_action_sha256": (
                    None if selected is None else selected["action_sha256"]
                ),
            }
            events.append(event)
        self.safe_trajectory_memory_history.extend(events)
        return events

    def safe_trajectory_memory_metadata(self) -> dict[str, dict[str, object]]:
        return {
            str(seed): {
                key: value
                for key, value in memory.items()
                if key not in ("observations", "actions")
            }
            for seed, memory in sorted(self.safe_trajectory_memory.items())
        }

    def save_safe_trajectory_memory(self, path: str | Path) -> None:
        payload = {
            "format": self.SAFE_MEMORY_FORMAT,
            "memory_by_seed": self.safe_trajectory_memory,
            "history": list(self.safe_trajectory_memory_history),
            "memory_sha256": safe_memory_sha256(self.safe_trajectory_memory),
        }
        with Path(path).open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def save_pre_intervention_replay(self, path: str | Path) -> dict[str, object]:
        state = self.pre_intervention_replay.state_dict()
        payload = {
            **state,
            "obs_sha256": array_sha256(state["obs"]),
            "action_sha256": array_sha256(state["action"]),
            "weight_sha256": array_sha256(state["weight"]),
            "lag_sha256": array_sha256(state["lag"]),
            "episode_events": list(self.pre_intervention_episode_events),
        }
        with Path(path).open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return {
            key: value
            for key, value in payload.items()
            if key not in ("obs", "action", "weight", "lag", "episode_events")
        }

    def _checkpoint_payload(self):
        # Bypass V2.4.2's parameter-reference payload: V2.4.3 replaces that
        # mechanism instead of silently carrying two consolidation sources.
        payload = GuardianDSACFlowV241._checkpoint_payload(self)
        payload.update(
            format=self.FORMAT,
            flow_config=asdict(self.config),
            replay_schema=self.REPLAY_SCHEMA,
            safe_trajectory_memory_by_seed=self.safe_trajectory_memory,
            safe_trajectory_memory_history=list(
                self.safe_trajectory_memory_history
            ),
            safe_trajectory_memory_sha256=safe_memory_sha256(
                self.safe_trajectory_memory
            ),
            pre_intervention_replay=self.pre_intervention_replay.state_dict(),
            pre_intervention_episode_events=list(
                self.pre_intervention_episode_events
            ),
            certified_safe_trajectory_memory=True,
            balanced_safe_memory_sampling=True,
            seed9_pre_intervention_teacher_replay=True,
            dynamic_ordinary_safe_retention=False,
            seed9_weighted_correction=True,
        )
        return payload

    def _load_checkpoint_payload(self, payload, *, load_base: bool = True):
        required = {
            "safe_trajectory_memory_by_seed",
            "safe_trajectory_memory_history",
            "safe_trajectory_memory_sha256",
            "pre_intervention_replay",
            "pre_intervention_episode_events",
            "certified_safe_trajectory_memory",
            "balanced_safe_memory_sampling",
            "seed9_pre_intervention_teacher_replay",
            "dynamic_ordinary_safe_retention",
            "seed9_weighted_correction",
        }
        if not isinstance(payload, dict) or payload.get("format") != self.FORMAT:
            raise ValueError("checkpoint is not Guardian-Flow-DSAC V2.4.3")
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"V2.4.3 checkpoint missing fields: {missing}")
        required_true = (
            "certified_safe_trajectory_memory",
            "balanced_safe_memory_sampling",
            "seed9_pre_intervention_teacher_replay",
            "seed9_weighted_correction",
        )
        if any(payload[name] is not True for name in required_true):
            raise ValueError("V2.4.3 checkpoint disables a required protection")
        if payload["dynamic_ordinary_safe_retention"] is not False:
            raise ValueError("V2.4.3 checkpoint re-enables parameter-reference retention")
        GuardianDSACFlowV241._load_checkpoint_payload(
            self, payload, load_base=load_base
        )
        restored: dict[int, dict[str, object]] = {}
        for seed_text, value in payload[
            "safe_trajectory_memory_by_seed"
        ].items():
            seed = int(seed_text)
            observations = np.asarray(value["observations"], dtype=np.float32)
            actions = np.asarray(value["actions"], dtype=np.float32)
            rows = int(value["rows"])
            if seed not in range(10):
                raise ValueError("V2.4.3 checkpoint contains an unbound memory Seed")
            if (
                observations.shape
                != (rows, int(self.flow_policy.actor_obs_dim))
                or actions.shape != (rows, int(self.flow_policy.act_dim))
                or rows < 1
                or rows > 1200
                or not np.all(np.isfinite(observations))
                or not np.all(np.isfinite(actions))
                or not np.all(actions >= -1.0)
                or not np.all(actions <= 1.0)
                or array_sha256(observations) != value["observation_sha256"]
                or array_sha256(actions) != value["action_sha256"]
                or not np.isfinite(float(value["ordinary_return"]))
                or not np.isfinite(
                    float(value["ordinary_min_physical_margin"])
                )
                or float(value["ordinary_min_physical_margin"]) <= 0.0
                or int(value["step"]) < 0
                or not str(value["label"])
                or value["scene_signature"]
                != self.scene_signatures[str(seed)]
                or len(str(value["trajectory_file_sha256"])) != 64
                or len(str(value["evaluation_checkpoint_sha256"])) != 64
                or len(str(value["actor_parameter_sha256"])) != 64
            ):
                raise ValueError("V2.4.3 checkpoint contains invalid trajectory memory")
            restored[seed] = {
                **value,
                "observations": observations,
                "actions": actions,
            }
        if safe_memory_sha256(restored) != payload["safe_trajectory_memory_sha256"]:
            raise ValueError("V2.4.3 safe trajectory memory hash mismatch")
        replay = PreInterventionReplay(
            self.flow_policy.actor_obs_dim,
            self.flow_policy.act_dim,
            self.config.pre_intervention_capacity,
        )
        replay.load_state_dict(payload["pre_intervention_replay"])
        self.safe_trajectory_memory = restored
        self.safe_trajectory_memory_history = list(
            payload["safe_trajectory_memory_history"]
        )
        self.pre_intervention_replay = replay
        self.pre_intervention_episode_events = list(
            payload["pre_intervention_episode_events"]
        )
        self.ordinary_safe_references = {}
        self.ordinary_safe_reference_history = []
        self.skill_reference_cursor = 0


__all__ = [
    "GuardianDSACFlowV243",
    "GuardianDSACFlowV243Config",
]
