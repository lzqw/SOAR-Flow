"""Unified Guardian-Flow-DSAC algorithm used by the paper experiments.

This module deliberately contains no hard-Seed path, Safe-Memory path,
performance gate, or moving safe-policy proximal.  The online one-step Flow
Actor has exactly four objectives: Task Q, episode-level Guardian dependence,
generic Guardian correction, and a frozen offline Flow-teacher anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.guardian_dsac_flow_v2_4 import (
    GuardianDSACFlowV24Config,
    _copy_tree,
    _python_metrics,
    _weighted_mean,
)
from relax.algorithm.guardian_dsac_flow_v2_4_1 import (
    GuardianDSACFlowV241,
    _cosine_rate,
)
from relax.algorithm.guardian_dsac_v2_3 import proxy_actor_objective
from relax.utils.flow_paper import OFFLINE_SEEDS, PAPER_SCENE_SEEDS


@dataclass(frozen=True)
class GuardianDSACFlowPaperConfig(GuardianDSACFlowV24Config):
    """Immutable weights and long-run schedules for the paper method."""

    offline_distill_weight_final: float = 1.0
    correction_weight: float = 2.0
    correction_speed_weight: float = 2.0
    correction_lateral_weight: float = 1.0
    zero_latent_correction_weight: float = 1.0
    anchor_weight: float = 0.05
    anchor_seeds: tuple[int, ...] = OFFLINE_SEEDS

    # Kept at exactly zero only for constructor compatibility with immutable
    # V2.4 utilities.  The paper Actor update does not compute a proximal term.
    proximal_weight_initial: float = 0.0
    proximal_weight_min: float = 0.0
    proximal_decay_steps: int = 1
    proximal_speed_weight: float = 2.0
    proximal_lateral_weight: float = 1.0

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
        values = (
            self.correction_speed_weight,
            self.correction_lateral_weight,
            self.zero_latent_correction_weight,
            self.anchor_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("paper Actor loss weights must be finite and non-negative")
        if self.correction_speed_weight + self.correction_lateral_weight <= 0.0:
            raise ValueError("at least one Guardian correction axis must be weighted")
        if tuple(int(seed) for seed in self.anchor_seeds) != OFFLINE_SEEDS:
            raise ValueError("paper Flow-teacher anchor must cover Seed0--4 exactly")
        if self.proximal_weight_initial != 0.0 or self.proximal_weight_min != 0.0:
            raise ValueError("moving safe-policy proximal is disabled in paper mode")
        for initial, final, steps in (
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
        ):
            if not np.isfinite(initial) or not np.isfinite(final):
                raise ValueError("paper learning rates must be finite")
            if initial <= 0.0 or final <= 0.0 or final > initial:
                raise ValueError("paper learning-rate endpoints are invalid")
            if int(steps) < 1:
                raise ValueError("paper learning-rate decay must be positive")


class GuardianDSACFlowPaper(GuardianDSACFlowV241):
    """Paper Flow Actor with evaluation-independent online optimization."""

    FORMAT = "guardian_flow_dsac_paper_v1"
    REPLAY_SCHEMA = "guardian_flow_dsac_paper_experience_v1"
    OBJECTIVE_TERMS = ("task_q", "guardian_p", "correction", "teacher_anchor")

    def __init__(
        self,
        base_algorithm,
        flow_policy,
        flow_params,
        *,
        config: GuardianDSACFlowPaperConfig | None = None,
    ):
        if tuple(int(seed) for seed in base_algorithm.scene_seeds) != PAPER_SCENE_SEEDS:
            raise ValueError(
                "paper Flow algorithm requires the predeclared paper scenes"
            )
        super().__init__(
            base_algorithm,
            flow_policy,
            flow_params,
            config=config or GuardianDSACFlowPaperConfig(),
        )
        self.config = config or GuardianDSACFlowPaperConfig()

    def _build_long_run_actor_update(self) -> None:
        config = self.config
        optimizer = self.online_actor_optimizer
        anchor_seeds = tuple(int(seed) for seed in config.anchor_seeds)
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
            proxy_actor_weight=None,
            correction_weight=None,
        ):
            effective_proxy_weight = (
                config.proxy_actor_weight
                if proxy_actor_weight is None
                else jnp.asarray(proxy_actor_weight, dtype=jnp.float32)
            )
            effective_correction_weight = (
                config.correction_weight
                if correction_weight is None
                else jnp.asarray(correction_weight, dtype=jnp.float32)
            )
            action_shape = jnp.asarray(data.action).shape
            policy_key, diversity_1_key, diversity_2_key, anchor_key = jax.random.split(
                key, 4
            )
            noise = jax.random.normal(policy_key, action_shape, dtype=jnp.float32)
            diversity_noise_1 = jax.random.normal(
                diversity_1_key, action_shape, dtype=jnp.float32
            )
            diversity_noise_2 = jax.random.normal(
                diversity_2_key, action_shape, dtype=jnp.float32
            )
            anchor_noise = jax.random.normal(
                anchor_key, action_shape, dtype=jnp.float32
            )
            actor_support = jnp.asarray(data.actor_controlled, dtype=jnp.float32)
            proxy_support = jnp.maximum(
                jnp.asarray(data.guardian_dependence_target_valid, jnp.float32),
                jnp.asarray(data.guardian_dependence_corrected_valid, jnp.float32),
            )
            hard = jnp.asarray(data.intervention_hard, dtype=bool)
            soft = jnp.asarray(data.intervention_soft, dtype=bool)
            modified = jnp.asarray(data.actual_control_modified, dtype=bool)
            latent_valid = jnp.asarray(data.flow_noise_valid, dtype=bool)
            zero_latent_row = jnp.asarray(data.flow_zero_latent, dtype=bool)
            correction_support = modified & latent_valid & (hard | soft)
            correction_sample_weight = jnp.where(
                hard, 1.0, jnp.where(soft, 0.5, 0.0)
            ).astype(jnp.float32) * correction_support.astype(jnp.float32)
            pair_valid = jnp.asarray(data.proxy_pair_valid, dtype=bool)
            correction_target = jnp.where(
                pair_valid[..., None],
                jnp.asarray(data.proxy_corrected_action, dtype=jnp.float32),
                jnp.asarray(data.action, dtype=jnp.float32),
            )
            stored_noise = jnp.asarray(data.flow_noise, dtype=jnp.float32)
            scene_seed = jnp.asarray(data.scene_seed, dtype=jnp.int32)
            anchor_seed = jnp.zeros_like(scene_seed, dtype=bool)
            for seed in anchor_seeds:
                anchor_seed = anchor_seed | (scene_seed == seed)
            # Offline rows deliberately carry no rollout latent.  This makes
            # the 20% fixed offline replay portion the only anchor support.
            anchor_support = (~latent_valid) & anchor_seed
            anchor_sample_weight = anchor_support.astype(jnp.float32)
            frozen_teacher_action = jax.lax.stop_gradient(
                self.flow_policy.bc_action(
                    state.params.bc_flow,
                    data.obs,
                    anchor_noise,
                )
            )

            def loss_fn(params):
                action = self.flow_policy.get_action(params, data.obs, noise)
                mean_1, _ = self.base.agent.q_distribution(q1, data.obs, action)
                mean_2, _ = self.base.agent.q_distribution(q2, data.obs, action)
                q_loss = -_weighted_mean(jnp.minimum(mean_1, mean_2), actor_support)
                proxy_logit = self.base.agent.proxy_value(proxy, data.obs, action)
                p_probability = jax.nn.sigmoid(proxy_logit)
                p_loss = _weighted_mean(
                    proxy_actor_objective(
                        proxy_logit,
                        objective=self.base.config.proxy_actor_objective,
                    ),
                    proxy_support,
                )
                target = jax.lax.stop_gradient(correction_target)
                stored_prediction = self.flow_policy.get_action(
                    params, data.obs, stored_noise
                )
                zero_prediction = self.flow_policy.get_action(
                    params, data.obs, jnp.zeros_like(stored_noise)
                )
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
                anchor_action = self.flow_policy.get_action(
                    params, data.obs, anchor_noise
                )
                anchor_square = jnp.sum(
                    jnp.square(anchor_action - frozen_teacher_action), axis=-1
                )
                anchor = _weighted_mean(anchor_square, anchor_sample_weight)
                diversity_action_1 = self.flow_policy.get_action(
                    params, data.obs, diversity_noise_1
                )
                diversity_action_2 = self.flow_policy.get_action(
                    params, data.obs, diversity_noise_2
                )
                diversity = jnp.mean(
                    jnp.linalg.norm(diversity_action_1 - diversity_action_2, axis=-1)
                )
                total = (
                    config.q_actor_weight * q_loss
                    + effective_proxy_weight * p_loss
                    + effective_correction_weight * correction
                    + config.anchor_weight * anchor
                )
                return total, (
                    q_loss,
                    p_loss,
                    p_probability,
                    correction,
                    stored_correction,
                    zero_correction,
                    stored_square,
                    zero_square,
                    anchor,
                    diversity,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params.one_step
            )
            adam_updates, opt_state = optimizer.update(
                grads, state.actor_opt_state, state.params.one_step
            )
            updates = jax.tree_util.tree_map(
                lambda value: -jnp.asarray(learning_rate, jnp.float32) * value,
                adam_updates,
            )
            params = optax.apply_updates(state.params.one_step, updates)
            target_params = optax.incremental_update(
                params, state.target_one_step, config.target_actor_tau
            )
            (
                q_loss,
                p_loss,
                p_probability,
                correction,
                stored_correction,
                zero_correction,
                stored_square,
                zero_square,
                anchor,
                diversity,
            ) = aux
            return state._replace(
                params=state.params._replace(one_step=params),
                actor_opt_state=opt_state,
                target_one_step=target_params,
                gradient_step=state.gradient_step + 1,
            ), {
                "flow_actor_loss": loss,
                "flow_q_loss": q_loss,
                "flow_proxy_loss": p_loss,
                "flow_absolute_proxy_loss": _weighted_mean(
                    p_probability, proxy_support
                ),
                "flow_correction_loss": correction,
                "flow_correction_stored_loss": stored_correction,
                "flow_correction_zero_latent_loss": zero_correction,
                "flow_correction_support_fraction": jnp.mean(
                    correction_support.astype(jnp.float32)
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
                "flow_anchor_loss": anchor,
                "flow_anchor_support_fraction": jnp.mean(anchor_sample_weight),
                "flow_zero_latent_batch_fraction": jnp.mean(
                    (zero_latent_row & latent_valid).astype(jnp.float32)
                ),
                "flow_latent_action_diversity": diversity,
                "flow_actor_learning_rate": learning_rate,
                "flow_proxy_actor_weight": jnp.asarray(
                    effective_proxy_weight, dtype=jnp.float32
                ),
                "flow_correction_actor_weight": jnp.asarray(
                    effective_correction_weight, dtype=jnp.float32
                ),
                "flow_proximal_enabled": jnp.asarray(0.0, dtype=jnp.float32),
            }

        self._online_actor_update_long_run = jax.jit(online_actor_update)

    def online_joint_update(
        self,
        key,
        data,
        *,
        actor_data=None,
        update_actor: bool = True,
        proxy_actor_weight: float | None = None,
        correction_weight: float | None = None,
    ):
        """Update Q/P normally while allowing loss-only Actor coefficients.

        The scalar coefficients are JAX inputs, so a time-varying proxy schedule
        does not rebuild the optimizer or retrace once per training step.  They
        affect only the Actor objective; the P network and target machinery are
        unchanged.
        """

        if not self.bc_flow_frozen:
            raise RuntimeError("online training requires a frozen Flow teacher")
        if not self.online_schedule_prepared:
            raise RuntimeError("prepare paper online schedules before training")
        effective_proxy_weight = (
            self.config.proxy_actor_weight
            if proxy_actor_weight is None
            else float(proxy_actor_weight)
        )
        effective_correction_weight = (
            self.config.correction_weight
            if correction_weight is None
            else float(correction_weight)
        )
        if not np.isfinite(effective_proxy_weight) or effective_proxy_weight < 0.0:
            raise ValueError("proxy Actor weight must be finite and non-negative")
        if (
            not np.isfinite(effective_correction_weight)
            or effective_correction_weight < 0.0
        ):
            raise ValueError("correction Actor weight must be finite and non-negative")

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
                jnp.asarray(effective_proxy_weight, dtype=jnp.float32),
                jnp.asarray(effective_correction_weight, dtype=jnp.float32),
            )
            self.online_actor_updates_total += 1
            metrics.update(_python_metrics(actor_metrics))
        return metrics

    def prepare_online_training(self) -> dict[str, object]:
        result = super().prepare_online_training()
        result.update(self.promote_numerical_checkpoint())
        return result

    def promote_numerical_checkpoint(self) -> dict[str, object]:
        """Record the latest finite state without interpreting performance."""

        self.safe_full_snapshot = (
            _copy_tree(self.base.state),
            _copy_tree(self.state),
        )
        self.base.hybrid_safe_snapshot = _copy_tree(self.base.state)
        self.safe_schedule_counters = (
            int(self.online_actor_updates_total),
            int(self.online_critic_updates_total),
        )
        self.base._safe_online_reference_step = int(self.base.online_reference_step)
        return {
            "numerical_checkpoint": "promoted",
            "performance_gate": False,
            "online_actor_updates": int(self.online_actor_updates_total),
            "online_critic_updates": int(self.online_critic_updates_total),
        }

    def promote_hybrid_safe(self) -> dict[str, object]:
        """Paper evaluations are read-only and never move training snapshots."""

        return {"rollback": "none", "snapshot": "unchanged", "evaluation_only": True}

    def reject_hybrid_safety(self) -> dict[str, object]:
        """Performance and scene outcomes cannot alter the training trajectory."""

        return {"rollback": "none", "snapshot": "unchanged", "evaluation_only": True}

    def record_ordinary_failure(self) -> dict[str, object]:
        return {"rollback": "none", "snapshot": "unchanged", "evaluation_only": True}

    def reject_numerical_disaster(self) -> dict[str, object]:
        base_state, flow_state = self.safe_full_snapshot
        self.base.state = _copy_tree(base_state)
        self.state = _copy_tree(flow_state)
        self.online_actor_updates_total, self.online_critic_updates_total = (
            int(value) for value in self.safe_schedule_counters
        )
        self.base.online_reference_step = int(self.base._safe_online_reference_step)
        self.base._consecutive_large_critic_losses = 0
        return {
            "rollback": "latest_numerical_checkpoint",
            "reason_scope": "nan_inf_or_corrupt_optimizer_only",
        }

    def _checkpoint_payload(self):
        payload = super()._checkpoint_payload()
        payload.update(
            format=self.FORMAT,
            replay_schema=self.REPLAY_SCHEMA,
            paper_protocol=True,
            paper_scene_seeds=list(PAPER_SCENE_SEEDS),
            actor_objective_terms=list(self.OBJECTIVE_TERMS),
            frozen_flow_teacher_anchor=True,
            anchor_seeds=list(OFFLINE_SEEDS),
            moving_safe_proximal=False,
            performance_rollback=False,
            safe_trajectory_memory=False,
            seed_specific_logic=False,
            seed9_logic=False,
            evaluation_changes_training=False,
            numerical_disaster_rollback_only=True,
        )
        return payload

    def _load_checkpoint_payload(self, payload, *, load_base: bool = True):
        required = {
            "paper_protocol": True,
            "frozen_flow_teacher_anchor": True,
            "moving_safe_proximal": False,
            "performance_rollback": False,
            "safe_trajectory_memory": False,
            "seed_specific_logic": False,
            "seed9_logic": False,
            "evaluation_changes_training": False,
            "numerical_disaster_rollback_only": True,
        }
        if not isinstance(payload, dict) or payload.get("format") != self.FORMAT:
            raise ValueError("checkpoint is not Guardian-Flow-DSAC paper format")
        for name, expected in required.items():
            if payload.get(name) is not expected:
                raise ValueError(f"paper checkpoint protocol flag mismatch: {name}")
        if (
            tuple(int(seed) for seed in payload.get("paper_scene_seeds", ()))
            != PAPER_SCENE_SEEDS
        ):
            raise ValueError("paper checkpoint scene benchmark mismatch")
        if (
            tuple(int(seed) for seed in payload.get("anchor_seeds", ()))
            != OFFLINE_SEEDS
        ):
            raise ValueError("paper checkpoint anchor Seed mismatch")
        if tuple(payload.get("actor_objective_terms", ())) != self.OBJECTIVE_TERMS:
            raise ValueError("paper checkpoint Actor objective mismatch")
        super()._load_checkpoint_payload(payload, load_base=load_base)


__all__ = ["GuardianDSACFlowPaper", "GuardianDSACFlowPaperConfig"]
