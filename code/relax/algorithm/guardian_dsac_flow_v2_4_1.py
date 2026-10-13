"""Guardian-Flow-DSAC V2.4.1 Long-Run protected online learner.

V2.4 remains immutable.  This version adds direct z=0 correction, Seed-bound
retention, cosine online learning rates and an independently bound checkpoint
format for the one-million-step protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.guardian_dsac_flow_v2_4 import (
    GuardianDSACFlowV24,
    GuardianDSACFlowV24Config,
    _copy_tree,
    _python_metrics,
    _weighted_mean,
)
from relax.algorithm.guardian_dsac_v2_3 import (
    proxy_actor_objective,
    weighted_action_proximal,
)
from relax.utils.flow_long_run import RETENTION_TEACHER_SEEDS


@dataclass(frozen=True)
class GuardianDSACFlowV241Config(GuardianDSACFlowV24Config):
    """Formal V2.4.1 loss and long-horizon optimization binding."""

    offline_distill_weight_final: float = 1.0
    correction_weight: float = 2.0
    proximal_decay_steps: int = 100_000

    correction_speed_weight: float = 2.0
    correction_lateral_weight: float = 1.0
    zero_latent_correction_weight: float = 1.0
    retention_teacher_weight: float = 0.05
    retention_teacher_seeds: tuple[int, ...] = RETENTION_TEACHER_SEEDS

    actor_learning_rate_initial: float = 1e-4
    actor_learning_rate_final: float = 1e-5
    actor_learning_rate_decay_steps: int = 500_000
    critic_learning_rate_initial: float = 3e-4
    critic_learning_rate_final: float = 3e-5
    critic_learning_rate_decay_steps: int = 1_000_000
    proxy_learning_rate_initial: float = 3e-4
    proxy_learning_rate_final: float = 3e-5
    proxy_learning_rate_decay_steps: int = 1_000_000

    def __post_init__(self) -> None:
        super().__post_init__()
        axis_weights = (
            self.correction_speed_weight,
            self.correction_lateral_weight,
            self.zero_latent_correction_weight,
            self.retention_teacher_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in axis_weights):
            raise ValueError("long-run loss weights must be finite and non-negative")
        if self.correction_speed_weight + self.correction_lateral_weight <= 0.0:
            raise ValueError("at least one correction axis must be weighted")
        rates = (
            (
                self.actor_learning_rate_initial,
                self.actor_learning_rate_final,
                self.actor_learning_rate_decay_steps,
            ),
            (
                self.critic_learning_rate_initial,
                self.critic_learning_rate_final,
                self.critic_learning_rate_decay_steps,
            ),
            (
                self.proxy_learning_rate_initial,
                self.proxy_learning_rate_final,
                self.proxy_learning_rate_decay_steps,
            ),
        )
        for initial, final, steps in rates:
            if not np.isfinite(initial) or not np.isfinite(final):
                raise ValueError("long-run learning rates must be finite")
            if initial <= 0.0 or final <= 0.0 or final > initial:
                raise ValueError("long-run learning-rate endpoints are invalid")
            if int(steps) < 1:
                raise ValueError("long-run learning-rate decay must be positive")
        seeds = tuple(int(seed) for seed in self.retention_teacher_seeds)
        if seeds != RETENTION_TEACHER_SEEDS:
            raise ValueError("formal retention anchor must bind exactly Seed1--4")


def _cosine_rate(initial: float, final: float, step: int, decay_steps: int) -> float:
    progress = min(max(float(step) / float(max(decay_steps, 1)), 0.0), 1.0)
    mix = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(final + (initial - final) * mix)


def _trees_equal(left, right) -> bool:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return bool(
        len(left_leaves) == len(right_leaves)
        and all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(left_leaves, right_leaves)
        )
    )


class GuardianDSACFlowV241(GuardianDSACFlowV24):
    """V2.4.1 one-step Flow Actor with protected long-run updates."""

    FORMAT = "guardian_dsac_one_step_flow_v2_4_1_long_run_v1"
    REPLAY_SCHEMA = "guardian_flow_dsac_long_run_experience_v1"

    def __init__(
        self,
        base_algorithm,
        flow_policy,
        flow_params,
        *,
        config: GuardianDSACFlowV241Config = GuardianDSACFlowV241Config(),
    ):
        super().__init__(base_algorithm, flow_policy, flow_params, config=config)
        self.config = config
        self.online_actor_optimizer = optax.chain(
            optax.clip_by_global_norm(config.gradient_clip_norm),
            optax.scale_by_adam(),
        )
        self._critic_learning_rate = optax.cosine_decay_schedule(
            init_value=config.critic_learning_rate_initial,
            decay_steps=config.critic_learning_rate_decay_steps,
            alpha=(
                config.critic_learning_rate_final
                / config.critic_learning_rate_initial
            ),
        )
        self._proxy_learning_rate = optax.cosine_decay_schedule(
            init_value=config.proxy_learning_rate_initial,
            decay_steps=config.proxy_learning_rate_decay_steps,
            alpha=(
                config.proxy_learning_rate_final
                / config.proxy_learning_rate_initial
            ),
        )
        self.online_schedule_prepared = False
        self.online_actor_updates_total = 0
        self.online_critic_updates_total = 0
        self.safe_schedule_counters = (0, 0)
        self._build_long_run_actor_update()

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
                stored_correction = _weighted_mean(
                    jnp.sum(stored_square * correction_axis_weight, axis=-1),
                    correction_sample_weight,
                )
                zero_correction = _weighted_mean(
                    jnp.sum(zero_square * correction_axis_weight, axis=-1),
                    correction_sample_weight,
                )
                correction = (
                    stored_correction
                    + config.zero_latent_correction_weight * zero_correction
                )
                retention_action = self.flow_policy.get_action(
                    params, data.obs, retention_noise
                )
                retention_square = jnp.sum(
                    jnp.square(retention_action - retention_teacher), axis=-1
                )
                retention = _weighted_mean(retention_square, retention_weight)
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
                    + config.correction_weight * correction
                    + config.retention_teacher_weight * retention
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
                diversity,
            ) = aux
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
                "flow_correction_support_fraction": jnp.mean(
                    correction_support.astype(jnp.float32)
                ),
                "flow_correction_weight_mean": jnp.mean(
                    correction_sample_weight
                ),
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
                "flow_zero_latent_batch_fraction": jnp.mean(
                    (zero_latent_row & latent_valid).astype(jnp.float32)
                ),
                "flow_latent_action_diversity": diversity,
                "flow_actor_learning_rate": learning_rate,
            }

        self._online_actor_update_long_run = jax.jit(online_actor_update)

    def _configure_base_online_optimizers(self, *, reset: bool) -> None:
        gradient_clip = self.base.config.gradient_clip_norm
        self.base.critic_optim = optax.chain(
            optax.clip_by_global_norm(gradient_clip),
            optax.adam(self._critic_learning_rate),
        )
        self.base.proxy_optim = optax.chain(
            optax.clip_by_global_norm(gradient_clip),
            optax.adam(self._proxy_learning_rate),
        )
        if reset:
            params = self.base.state.params
            self.base.state = self.base.state._replace(
                opt_state=self.base.state.opt_state._replace(
                    q1=self.base.critic_optim.init(params.q1),
                    q2=self.base.critic_optim.init(params.q2),
                    proxy=self.base.proxy_optim.init(params.proxy),
                )
            )
        self.base._build_updates()

    def prepare_online_training(self) -> dict[str, object]:
        """Reset only optimizer moments and start the bound online schedules."""

        if not self.bc_flow_frozen:
            raise RuntimeError("freeze the Flow teacher before online preparation")
        if self.online_schedule_prepared:
            raise RuntimeError("long-run online schedules are already prepared")
        self._configure_base_online_optimizers(reset=True)
        live = _copy_tree(self.state.params.one_step)
        self.state = self.state._replace(
            actor_opt_state=self.online_actor_optimizer.init(live),
            target_one_step=_copy_tree(live),
            proximal_step=jnp.asarray(0, dtype=jnp.int32),
        )
        self.online_schedule_prepared = True
        self.online_actor_updates_total = 0
        self.online_critic_updates_total = 0
        # Preserve the frozen post-distillation safe Actor.  Optional offline
        # joint updates must not silently certify their live Actor here.
        safe = _copy_tree(self.state.safe_one_step)
        safe_flow_state = self.state._replace(
            params=self.state.params._replace(one_step=_copy_tree(safe)),
            actor_opt_state=self.online_actor_optimizer.init(safe),
            target_one_step=_copy_tree(safe),
            proximal_step=jnp.asarray(0, dtype=jnp.int32),
        )
        self.base.hybrid_safe_snapshot = _copy_tree(self.base.state)
        self.safe_full_snapshot = (
            _copy_tree(self.base.state),
            _copy_tree(safe_flow_state),
        )
        self.safe_schedule_counters = (0, 0)
        return {
            "online_schedule_prepared": True,
            "actor_learning_rate": self.config.actor_learning_rate_initial,
            "critic_learning_rate": self.config.critic_learning_rate_initial,
            "proxy_learning_rate": self.config.proxy_learning_rate_initial,
        }

    def online_joint_update(
        self,
        key,
        data,
        *,
        actor_data=None,
        update_actor: bool = True,
    ):
        if not self.bc_flow_frozen:
            raise RuntimeError("online training requires a frozen Flow teacher")
        if not self.online_schedule_prepared:
            raise RuntimeError("prepare V2.4.1 online schedules before training")
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
            )
            self.online_actor_updates_total += 1
            metrics.update(_python_metrics(actor_metrics))
        return metrics

    def promote_hybrid_safe(self):
        result = super().promote_hybrid_safe()
        self.safe_schedule_counters = (
            int(self.online_actor_updates_total),
            int(self.online_critic_updates_total),
        )
        return result

    def reject_hybrid_safety(self):
        """Reset Actor moments/proximal only; keep global cosine progress."""

        safe = _copy_tree(self.state.safe_one_step)
        self.state = self.state._replace(
            params=self.state.params._replace(one_step=safe),
            actor_opt_state=self.online_actor_optimizer.init(safe),
            target_one_step=_copy_tree(safe),
            proximal_step=jnp.asarray(0, dtype=jnp.int32),
        )
        return {
            "rollback": "actor_only",
            "snapshot": "hybrid_safe_snapshot",
            "actor_schedule_step_preserved": int(
                self.online_actor_updates_total
            ),
        }

    def reject_numerical_disaster(self):
        result = super().reject_numerical_disaster()
        (
            self.online_actor_updates_total,
            self.online_critic_updates_total,
        ) = self.safe_schedule_counters
        return result

    def load_v24_distilled_checkpoint(self, path: str | Path) -> None:
        """Strictly migrate only the pre-joint frozen V2.4 distilled state."""

        from relax.algorithm.guardian_dsac_flow_v2_4 import (
            GuardianDSACFlowV24Config,
        )

        source = Path(path)
        authoritative_base_state = _copy_tree(self.base.state)
        with source.open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("format") != GuardianDSACFlowV24.FORMAT:
            raise ValueError("migration source is not a V2.4 Flow checkpoint")
        temporary = GuardianDSACFlowV24(
            self.base,
            self.flow_policy,
            self.state.params,
            config=GuardianDSACFlowV24Config(**payload["flow_config"]),
        )
        temporary._load_checkpoint_payload(payload)
        if not temporary.bc_flow_frozen:
            raise ValueError("V2.4 migration source has an unfrozen teacher")
        if int(np.asarray(temporary.state.offline_joint_step)) != 0:
            raise ValueError("V2.4 migration source is not the distilled pre-joint state")
        if int(np.asarray(temporary.state.proximal_step)) != 0:
            raise ValueError("V2.4 migration source has online proximal progress")
        if temporary.deployable_checkpoint is not None:
            raise ValueError("V2.4 migration source already has a deployment snapshot")
        if not _trees_equal(temporary.base.state, authoritative_base_state):
            raise ValueError(
                "V2.4 migration source does not contain the authoritative shared Q/P state"
            )
        if not (
            _trees_equal(
                temporary.state.params.one_step,
                temporary.state.safe_one_step,
            )
            and _trees_equal(
                temporary.state.params.one_step,
                temporary.state.target_one_step,
            )
        ):
            raise ValueError(
                "V2.4 migration source is not the synchronized distilled Actor"
            )
        self.base = temporary.base
        self.state = _copy_tree(temporary.state)
        self.safe_full_snapshot = _copy_tree(temporary.safe_full_snapshot)
        self.deployable_checkpoint = None
        self.prepare_online_training()

    def _checkpoint_payload(self):
        payload = super()._checkpoint_payload()
        payload.update(
            format=self.FORMAT,
            flow_config=asdict(self.config),
            replay_schema=self.REPLAY_SCHEMA,
            long_run_state={
                "online_schedule_prepared": bool(self.online_schedule_prepared),
                "online_actor_updates_total": int(self.online_actor_updates_total),
                "online_critic_updates_total": int(self.online_critic_updates_total),
                "safe_schedule_counters": [
                    int(value) for value in self.safe_schedule_counters
                ],
            },
            episode_level_latent_mode=True,
            zero_latent_correction=True,
            seed_indexed_replay=True,
            frozen_teacher_retention=True,
            cosine_online_learning_rates=True,
        )
        return payload

    def _load_checkpoint_payload(self, payload, *, load_base: bool = True):
        if not isinstance(payload, dict) or payload.get("format") != self.FORMAT:
            raise ValueError("checkpoint is not Guardian-Flow-DSAC V2.4.1 Long-Run")
        required = {
            "base_v2_3",
            "flow_state",
            "safe_full_snapshot",
            "deployable_checkpoint",
            "flow_config",
            "actor_observation_dim",
            "action_dim",
            "hidden_sizes",
            "integration_steps",
            "candidate_sampling",
            "q_reranking",
            "rejection_sampling",
            "reverse_ode_likelihood",
            "flow_internal_cbf",
            "explicit_actor_entropy",
            "teacher_online_updates",
            "replay_schema",
            "long_run_state",
            "episode_level_latent_mode",
            "zero_latent_correction",
            "seed_indexed_replay",
            "frozen_teacher_retention",
            "cosine_online_learning_rates",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"V2.4.1 checkpoint missing fields: {missing}")
        forbidden = (
            "candidate_sampling",
            "q_reranking",
            "rejection_sampling",
            "reverse_ode_likelihood",
            "flow_internal_cbf",
            "explicit_actor_entropy",
            "teacher_online_updates",
        )
        if any(bool(payload[name]) for name in forbidden):
            raise ValueError("V2.4.1 checkpoint enables a forbidden mechanism")
        required_true = (
            "episode_level_latent_mode",
            "zero_latent_correction",
            "seed_indexed_replay",
            "frozen_teacher_retention",
            "cosine_online_learning_rates",
        )
        if any(payload[name] is not True for name in required_true):
            raise ValueError("V2.4.1 checkpoint disables a required protection")
        if payload["replay_schema"] != self.REPLAY_SCHEMA:
            raise ValueError("V2.4.1 replay schema binding mismatch")
        expected_architecture = (
            self.flow_policy.actor_obs_dim,
            self.flow_policy.act_dim,
            tuple(self.flow_policy.hidden_sizes),
            self.flow_policy.integration_steps,
        )
        observed_architecture = (
            int(payload["actor_observation_dim"]),
            int(payload["action_dim"]),
            tuple(int(value) for value in payload["hidden_sizes"]),
            int(payload["integration_steps"]),
        )
        if observed_architecture != expected_architecture:
            raise ValueError("V2.4.1 Flow architecture binding mismatch")
        if dict(payload["flow_config"]) != asdict(self.config):
            raise ValueError("V2.4.1 Flow config binding mismatch")
        long_run = payload["long_run_state"]
        counters = (
            int(long_run["online_actor_updates_total"]),
            int(long_run["online_critic_updates_total"]),
        )
        safe_counters = tuple(int(value) for value in long_run["safe_schedule_counters"])
        if min((*counters, *safe_counters)) < 0 or len(safe_counters) != 2:
            raise ValueError("V2.4.1 checkpoint has invalid schedule counters")
        prepared = bool(long_run["online_schedule_prepared"])
        if not prepared and any(counters):
            raise ValueError("unprepared V2.4.1 checkpoint has online updates")
        if prepared:
            self._configure_base_online_optimizers(reset=False)
        if load_base:
            self.base._load_checkpoint_payload(payload["base_v2_3"])
        self.state = jax.device_put(payload["flow_state"])
        self.safe_full_snapshot = jax.device_put(payload["safe_full_snapshot"])
        deployable = payload["deployable_checkpoint"]
        self.deployable_checkpoint = (
            None
            if deployable is None
            else {
                **deployable,
                "base_state": jax.device_put(deployable["base_state"]),
                "flow_state": jax.device_put(deployable["flow_state"]),
            }
        )
        self.online_schedule_prepared = prepared
        (
            self.online_actor_updates_total,
            self.online_critic_updates_total,
        ) = counters
        self.safe_schedule_counters = safe_counters
        self._validate_loaded_state()


__all__ = [
    "GuardianDSACFlowV241",
    "GuardianDSACFlowV241Config",
]
