"""Pure-JAX asymmetric Safe SAC with trajectory-teacher distillation."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from relax.algorithm.base import Algorithm
from relax.network.safe_sac_jax import JaxSafeSACNet, JaxSafeSACParams
from relax.utils.experience import JaxSafeSACExperience
from relax.utils.typing import Metric


class JaxSafeSACOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class JaxSafeSACTrainState(NamedTuple):
    params: JaxSafeSACParams
    opt_state: JaxSafeSACOptStates
    gradient_step: jax.Array


@dataclass(frozen=True)
class JaxSafeSACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    reward_scale: float = 1.0
    teacher_bc_weight: float = 1.0
    actor_sac_weight: float = 1.0
    gradient_clip_norm: float = 10.0
    # Two critics form a small uncertainty ensemble. beta=1 recovers min(Q1,Q2).
    q_lcb_beta: float = 1.0
    # Offline-to-online stability. Defaults preserve the original algorithm.
    champion_kl_weight: float = 0.0
    initial_policy_kl_weight: float = 0.0
    anchor_uncertainty_scale: float = 0.0
    anchor_intervention_scale: float = 0.0
    # Teacher imitation is gated off once an independently controlled Actor
    # action has no lower LCB value than the recovery teacher.
    teacher_advantage_gating: bool = False
    teacher_advantage_margin: float = 0.0
    teacher_advantage_temperature: float = 0.25
    teacher_advantage_max_weight: float = 5.0


def _weighted_mean(value: jax.Array, weight: jax.Array) -> jax.Array:
    value = jnp.asarray(value, dtype=jnp.float32)
    weight = jnp.asarray(weight, dtype=jnp.float32)
    return jnp.sum(value * weight) / jnp.maximum(jnp.sum(weight), 1.0)


def _diagonal_gaussian_kl(
    mean: jax.Array,
    std: jax.Array,
    reference_mean: jax.Array,
    reference_std: jax.Array,
) -> jax.Array:
    """KL(new || frozen reference) before the common tanh transform."""

    variance = jnp.square(std)
    reference_variance = jnp.maximum(jnp.square(reference_std), 1e-8)
    per_dimension = (
        jnp.log(jnp.maximum(reference_std, 1e-8) / jnp.maximum(std, 1e-8))
        + (variance + jnp.square(mean - reference_mean))
        / (2.0 * reference_variance)
        - 0.5
    )
    return jnp.sum(per_dimension, axis=-1)


class JaxSafeSAC(Algorithm):
    """SAC whose actor is context-free and critic is intervention-aware.

    Recovery transitions remain useful for critic learning.  Policy gradients
    are restricted to actor-controlled transitions, while valid inverse-MPC
    teacher labels supervise the deterministic policy on intervention and
    pre-intervention Safe-DAgger samples.
    """

    def __init__(
        self,
        agent: JaxSafeSACNet,
        params: JaxSafeSACParams,
        *,
        config: JaxSafeSACConfig = JaxSafeSACConfig(),
    ):
        self.agent = agent
        self.config = config
        self.optim = optax.chain(
            optax.clip_by_global_norm(config.gradient_clip_norm),
            optax.adam(config.learning_rate),
        )
        self.log_alpha_optim = optax.adam(config.alpha_learning_rate)
        self.state = JaxSafeSACTrainState(
            params=params,
            opt_state=JaxSafeSACOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.optim.init(params.policy),
                log_alpha=self.log_alpha_optim.init(params.log_alpha),
            ),
            gradient_step=jnp.asarray(0, dtype=jnp.int32),
        )
        # These snapshots are intentionally outside TrainState so checkpoints
        # from the first Safe-SAC implementation remain loadable. New saves
        # persist both anchors in a versioned payload below.
        self.initial_policy_params = params.policy
        self.champion_policy_params = params.policy

        def stateless_update(
            key: jax.Array,
            state: JaxSafeSACTrainState,
            data: JaxSafeSACExperience,
            initial_policy_params: hk.Params,
            champion_policy_params: hk.Params,
        ) -> Tuple[JaxSafeSACTrainState, Metric]:
            params = state.params
            opt_state = state.opt_state
            next_key, actor_key = jax.random.split(key)

            reward = jnp.asarray(data.reward, dtype=jnp.float32)
            done = jnp.asarray(data.done, dtype=jnp.float32)
            next_mode = jnp.asarray(data.next_mode) == 1
            actor_support = jnp.asarray(data.actor_controlled, dtype=jnp.float32)
            teacher_weight = (
                jnp.asarray(data.teacher_valid, dtype=jnp.float32)
                * jnp.maximum(jnp.asarray(data.teacher_weight, dtype=jnp.float32), 0.0)
            )

            sampled_next_action, sampled_next_logp = self.agent.evaluate(
                next_key, params.policy, data.next_obs
            )
            next_action = jnp.where(
                next_mode[..., None],
                jnp.asarray(data.next_recovery_action, dtype=jnp.float32),
                sampled_next_action,
            )
            next_logp = jnp.where(next_mode, 0.0, sampled_next_logp)
            target_q1 = self.agent.q(
                params.target_q1, data.next_obs, next_action
            )
            target_q2 = self.agent.q(
                params.target_q2, data.next_obs, next_action
            )
            alpha = jnp.exp(params.log_alpha)
            target_q = jnp.minimum(target_q1, target_q2) - alpha * next_logp
            backup = jax.lax.stop_gradient(
                config.reward_scale * reward
                + (1.0 - done) * config.gamma * target_q
            )

            def q_loss_fn(q_params: hk.Params):
                value = self.agent.q(q_params, data.obs, data.action)
                loss = jnp.mean(jnp.square(value - backup))
                return loss, value

            (q1_loss, replay_q1), q1_grads = jax.value_and_grad(
                q_loss_fn, has_aux=True
            )(params.q1)
            (q2_loss, replay_q2), q2_grads = jax.value_and_grad(
                q_loss_fn, has_aux=True
            )(params.q2)
            q1_updates, q1_opt_state = self.optim.update(
                q1_grads, opt_state.q1, params.q1
            )
            q2_updates, q2_opt_state = self.optim.update(
                q2_grads, opt_state.q2, params.q2
            )
            q1_params = optax.apply_updates(params.q1, q1_updates)
            q2_params = optax.apply_updates(params.q2, q2_updates)

            def policy_loss_fn(policy_params: hk.Params):
                new_action, new_logp = self.agent.evaluate(
                    actor_key, policy_params, data.obs
                )
                q1 = self.agent.q(q1_params, data.obs, new_action)
                q2 = self.agent.q(q2_params, data.obs, new_action)
                q_mean = 0.5 * (q1 + q2)
                q_disagreement = 0.5 * jnp.abs(q1 - q2)
                q_lcb = q_mean - config.q_lcb_beta * q_disagreement
                per_sample_sac = alpha * new_logp - q_lcb
                supported_sac_loss = _weighted_mean(
                    per_sample_sac, actor_support
                )

                deterministic_action = self.agent.get_deterministic_action(
                    policy_params, data.obs
                )
                deterministic_q1 = self.agent.q(
                    q1_params, data.obs, deterministic_action
                )
                deterministic_q2 = self.agent.q(
                    q2_params, data.obs, deterministic_action
                )
                deterministic_lcb = 0.5 * (
                    deterministic_q1 + deterministic_q2
                ) - config.q_lcb_beta * 0.5 * jnp.abs(
                    deterministic_q1 - deterministic_q2
                )
                teacher_q1 = self.agent.q(
                    q1_params, data.obs, data.teacher_action
                )
                teacher_q2 = self.agent.q(
                    q2_params, data.obs, data.teacher_action
                )
                teacher_lcb = 0.5 * (teacher_q1 + teacher_q2) - (
                    config.q_lcb_beta * 0.5 * jnp.abs(teacher_q1 - teacher_q2)
                )
                teacher_advantage = jax.lax.stop_gradient(
                    teacher_lcb - deterministic_lcb
                )
                if config.teacher_advantage_gating:
                    temperature = max(config.teacher_advantage_temperature, 1e-6)
                    maximum = max(config.teacher_advantage_max_weight, 1.0)
                    positive_advantage = teacher_advantage > (
                        config.teacher_advantage_margin
                    )
                    advantage_gate = jnp.where(
                        positive_advantage,
                        jnp.exp(
                            jnp.clip(
                                (
                                    teacher_advantage
                                    - config.teacher_advantage_margin
                                )
                                / temperature,
                                0.0,
                                math.log(maximum),
                            )
                        ),
                        0.0,
                    )
                    # Unsafe/intervened samples always keep their teacher.
                    teacher_gate = jnp.where(
                        actor_support > 0.5, advantage_gate, 1.0
                    )
                else:
                    teacher_gate = jnp.ones_like(teacher_advantage)
                gated_teacher_weight = teacher_weight * teacher_gate
                teacher_error = jnp.sum(
                    jnp.square(deterministic_action - data.teacher_action),
                    axis=-1,
                )
                teacher_bc_loss = _weighted_mean(
                    teacher_error, gated_teacher_weight
                )

                mean, std = self.agent.distribution(
                    policy_params, data.obs
                )
                initial_mean, initial_std = self.agent.distribution(
                    initial_policy_params, data.obs
                )
                champion_mean, champion_std = self.agent.distribution(
                    champion_policy_params, data.obs
                )
                initial_kl = _diagonal_gaussian_kl(
                    mean,
                    std,
                    jax.lax.stop_gradient(initial_mean),
                    jax.lax.stop_gradient(initial_std),
                )
                champion_kl = _diagonal_gaussian_kl(
                    mean,
                    std,
                    jax.lax.stop_gradient(champion_mean),
                    jax.lax.stop_gradient(champion_std),
                )
                adaptive_anchor = jax.lax.stop_gradient(
                    1.0
                    + config.anchor_uncertainty_scale * q_disagreement
                    + config.anchor_intervention_scale
                    * jnp.asarray(data.intervention, dtype=jnp.float32)
                )
                initial_kl_loss = jnp.mean(initial_kl * adaptive_anchor)
                champion_kl_loss = jnp.mean(champion_kl)
                loss = (
                    config.actor_sac_weight * supported_sac_loss
                    + config.teacher_bc_weight * teacher_bc_loss
                    + config.initial_policy_kl_weight * initial_kl_loss
                    + config.champion_kl_weight * champion_kl_loss
                )
                return loss, (
                    supported_sac_loss,
                    teacher_bc_loss,
                    new_logp,
                    q_lcb,
                    q_disagreement,
                    teacher_error,
                    teacher_advantage,
                    teacher_gate,
                    initial_kl_loss,
                    champion_kl_loss,
                )

            (policy_loss, policy_aux), policy_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True
            )(params.policy)
            (
                supported_sac_loss,
                teacher_bc_loss,
                new_logp,
                actor_q,
                q_disagreement,
                teacher_error,
                teacher_advantage,
                teacher_gate,
                initial_kl_loss,
                champion_kl_loss,
            ) = policy_aux
            policy_updates, policy_opt_state = self.optim.update(
                policy_grads, opt_state.policy, params.policy
            )
            policy_params = optax.apply_updates(params.policy, policy_updates)

            def alpha_loss_fn(log_alpha: jax.Array):
                per_sample = -log_alpha * jax.lax.stop_gradient(
                    new_logp + self.agent.target_entropy
                )
                return _weighted_mean(per_sample, actor_support)

            alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(
                params.log_alpha
            )
            alpha_updates, log_alpha_opt_state = self.log_alpha_optim.update(
                alpha_grads, opt_state.log_alpha, params.log_alpha
            )
            log_alpha = optax.apply_updates(params.log_alpha, alpha_updates)

            target_q1_params = optax.incremental_update(
                q1_params, params.target_q1, config.tau
            )
            target_q2_params = optax.incremental_update(
                q2_params, params.target_q2, config.tau
            )
            next_state = JaxSafeSACTrainState(
                params=JaxSafeSACParams(
                    q1=q1_params,
                    q2=q2_params,
                    target_q1=target_q1_params,
                    target_q2=target_q2_params,
                    policy=policy_params,
                    log_alpha=log_alpha,
                ),
                opt_state=JaxSafeSACOptStates(
                    q1=q1_opt_state,
                    q2=q2_opt_state,
                    policy=policy_opt_state,
                    log_alpha=log_alpha_opt_state,
                ),
                gradient_step=state.gradient_step + 1,
            )
            metrics = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1_mean": jnp.mean(replay_q1),
                "q2_mean": jnp.mean(replay_q2),
                "backup_mean": jnp.mean(backup),
                "policy_loss": policy_loss,
                "supported_sac_loss": supported_sac_loss,
                "teacher_bc_loss": teacher_bc_loss,
                "teacher_error": _weighted_mean(
                    teacher_error, teacher_weight * teacher_gate
                ),
                "actor_q": _weighted_mean(actor_q, actor_support),
                "q_disagreement": jnp.mean(q_disagreement),
                "entropy": -_weighted_mean(new_logp, actor_support),
                "alpha": jnp.exp(log_alpha),
                "alpha_loss": alpha_loss,
                "actor_support_fraction": jnp.mean(actor_support),
                "teacher_fraction": jnp.mean(
                    jnp.asarray(data.teacher_valid, dtype=jnp.float32)
                ),
                "teacher_gate_mean": _weighted_mean(
                    teacher_gate, teacher_weight
                ),
                "teacher_advantage": _weighted_mean(
                    teacher_advantage, teacher_weight
                ),
                "initial_policy_kl": initial_kl_loss,
                "champion_policy_kl": champion_kl_loss,
                "recovery_target_fraction": jnp.mean(
                    next_mode.astype(jnp.float32)
                ),
            }
            return next_state, metrics

        def stateless_critic_update(
            key: jax.Array,
            state: JaxSafeSACTrainState,
            data: JaxSafeSACExperience,
        ) -> Tuple[JaxSafeSACTrainState, Metric]:
            """Update Q/target-Q only while the deployed policy stays frozen."""

            params = state.params
            next_action, next_logp = self.agent.evaluate(
                key, params.policy, data.next_obs
            )
            next_mode = jnp.asarray(data.next_mode) == 1
            next_action = jnp.where(
                next_mode[..., None],
                jnp.asarray(data.next_recovery_action, dtype=jnp.float32),
                next_action,
            )
            next_logp = jnp.where(next_mode, 0.0, next_logp)
            target_q1 = self.agent.q(
                params.target_q1, data.next_obs, next_action
            )
            target_q2 = self.agent.q(
                params.target_q2, data.next_obs, next_action
            )
            backup = jax.lax.stop_gradient(
                config.reward_scale * jnp.asarray(data.reward, dtype=jnp.float32)
                + (1.0 - jnp.asarray(data.done, dtype=jnp.float32))
                * config.gamma
                * (
                    jnp.minimum(target_q1, target_q2)
                    - jnp.exp(params.log_alpha) * next_logp
                )
            )

            def q_loss(q_params: hk.Params):
                value = self.agent.q(q_params, data.obs, data.action)
                return jnp.mean(jnp.square(value - backup)), value

            (q1_loss, q1_value), q1_grads = jax.value_and_grad(
                q_loss, has_aux=True
            )(params.q1)
            (q2_loss, q2_value), q2_grads = jax.value_and_grad(
                q_loss, has_aux=True
            )(params.q2)
            q1_updates, q1_opt_state = self.optim.update(
                q1_grads, state.opt_state.q1, params.q1
            )
            q2_updates, q2_opt_state = self.optim.update(
                q2_grads, state.opt_state.q2, params.q2
            )
            q1_params = optax.apply_updates(params.q1, q1_updates)
            q2_params = optax.apply_updates(params.q2, q2_updates)
            next_state = state._replace(
                params=params._replace(
                    q1=q1_params,
                    q2=q2_params,
                    target_q1=optax.incremental_update(
                        q1_params, params.target_q1, config.tau
                    ),
                    target_q2=optax.incremental_update(
                        q2_params, params.target_q2, config.tau
                    ),
                ),
                opt_state=state.opt_state._replace(
                    q1=q1_opt_state,
                    q2=q2_opt_state,
                ),
                gradient_step=state.gradient_step + 1,
            )
            return next_state, {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1_mean": jnp.mean(q1_value),
                "q2_mean": jnp.mean(q2_value),
                "backup_mean": jnp.mean(backup),
                "critic_only": jnp.asarray(1.0, dtype=jnp.float32),
            }

        def stateless_bc_update(
            key: jax.Array,
            state: JaxSafeSACTrainState,
            data: JaxSafeSACExperience,
        ) -> Tuple[JaxSafeSACTrainState, Metric]:
            del key
            weight = (
                jnp.asarray(data.teacher_valid, dtype=jnp.float32)
                * jnp.maximum(jnp.asarray(data.teacher_weight, dtype=jnp.float32), 0.0)
            )

            def bc_loss_fn(policy_params: hk.Params):
                action = self.agent.get_deterministic_action(
                    policy_params, data.obs
                )
                error = jnp.sum(
                    jnp.square(action - data.teacher_action), axis=-1
                )
                return _weighted_mean(error, weight), error

            (bc_loss, error), grads = jax.value_and_grad(
                bc_loss_fn, has_aux=True
            )(state.params.policy)
            updates, policy_opt_state = self.optim.update(
                grads, state.opt_state.policy, state.params.policy
            )
            policy_params = optax.apply_updates(state.params.policy, updates)
            next_state = state._replace(
                params=state.params._replace(policy=policy_params),
                opt_state=state.opt_state._replace(policy=policy_opt_state),
                gradient_step=state.gradient_step + 1,
            )
            return next_state, {
                "teacher_bc_loss": bc_loss,
                "teacher_error": _weighted_mean(error, weight),
                "teacher_fraction": jnp.mean(
                    jnp.asarray(data.teacher_valid, dtype=jnp.float32)
                ),
            }

        self._stable_update = jax.jit(stateless_update)
        self._critic_update = jax.jit(stateless_critic_update)
        self._bc_update = jax.jit(stateless_bc_update)

        def compatibility_update(key, state, data):
            return stateless_update(
                key,
                state,
                data,
                state.params.policy,
                state.params.policy,
            )

        self._implement_common_behavior(
            compatibility_update,
            self.agent.get_action,
            self.agent.get_deterministic_action,
        )

    def update(self, key: jax.Array, data: JaxSafeSACExperience) -> Metric:
        self.state, info = self._stable_update(
            key,
            self.state,
            data,
            self.initial_policy_params,
            self.champion_policy_params,
        )
        return {name: float(value) for name, value in info.items()}

    def update_critics(
        self, key: jax.Array, data: JaxSafeSACExperience
    ) -> Metric:
        self.state, info = self._critic_update(key, self.state, data)
        return {name: float(value) for name, value in info.items()}

    def mark_policy_as_initial(self) -> None:
        self.initial_policy_params = jax.tree_util.tree_map(
            lambda value: jnp.array(value, copy=True), self.state.params.policy
        )
        self.champion_policy_params = self.initial_policy_params

    def promote_champion(self) -> None:
        self.champion_policy_params = jax.tree_util.tree_map(
            lambda value: jnp.array(value, copy=True), self.state.params.policy
        )

    def rollback_to_champion(self) -> None:
        """Restore the learner Actor after a failed deployment gate.

        Critics are intentionally retained: they may have learned useful
        online values even when the candidate policy itself is rejected.
        The policy optimizer is reset so stale Adam momentum cannot
        immediately push the restored Actor back toward the failed candidate.
        """

        policy = jax.tree_util.tree_map(
            lambda value: jnp.array(value, copy=True),
            self.champion_policy_params,
        )
        self.state = self.state._replace(
            params=self.state.params._replace(policy=policy),
            opt_state=self.state.opt_state._replace(
                policy=self.optim.init(policy)
            ),
        )

    def get_champion_action(
        self, key: jax.Array, observation: jax.Array
    ) -> jax.Array:
        return self._get_action(key, self.champion_policy_params, observation)

    def get_deterministic_champion_action(
        self, observation: jax.Array
    ) -> jax.Array:
        return self._get_deterministic_action(
            self.champion_policy_params, observation
        )

    def save(self, path: str) -> None:
        payload = {
            "format": "jax_safe_sac_offline_to_online_v1",
            "state": jax.device_get(self.state),
            "initial_policy_params": jax.device_get(
                self.initial_policy_params
            ),
            "champion_policy_params": jax.device_get(
                self.champion_policy_params
            ),
        }
        with open(path, "wb") as stream:
            pickle.dump(payload, stream)

    def load(self, path: str) -> None:
        with open(path, "rb") as stream:
            payload = pickle.load(stream)
        if isinstance(payload, dict) and payload.get("format") == (
            "jax_safe_sac_offline_to_online_v1"
        ):
            self.state = jax.device_put(payload["state"])
            self.initial_policy_params = jax.device_put(
                payload["initial_policy_params"]
            )
            self.champion_policy_params = jax.device_put(
                payload["champion_policy_params"]
            )
        else:
            # Backward compatibility with the original state-only checkpoints.
            self.state = jax.device_put(payload)
            self.initial_policy_params = self.state.params.policy
            self.champion_policy_params = self.state.params.policy

    def pretrain_bc(
        self, key: jax.Array, data: JaxSafeSACExperience
    ) -> Metric:
        self.state, info = self._bc_update(key, self.state, data)
        return {name: float(value) for name, value in info.items()}
