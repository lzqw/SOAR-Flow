"""Formal Guardian-Flow-DSAC V2.4 with a one-forward-pass deployed Actor.

The conditional Rectified-Flow velocity network is an offline behaviour
teacher only.  Online interaction, Bellman targets and deployment all use the
one-step Actor exactly once; there is no candidate bank, Q reranking,
rejection sampling, reverse-ODE likelihood or Flow-internal CBF path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import pickle
from pathlib import Path
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.guardian_dsac_v2_3 import (
    proxy_actor_objective,
    weighted_action_proximal,
)
from relax.network.flow_policy import (
    FlowPolicyParams,
    JaxOneStepFlowPolicy,
    flow_matching_loss,
    one_step_distillation_loss,
)


class FlowActorTrainState(NamedTuple):
    params: FlowPolicyParams
    bc_opt_state: optax.OptState
    actor_opt_state: optax.OptState
    target_one_step: hk.Params
    safe_one_step: hk.Params
    gradient_step: jax.Array
    offline_joint_step: jax.Array
    proximal_step: jax.Array
    bc_flow_frozen: jax.Array


@dataclass(frozen=True)
class GuardianDSACFlowV24Config:
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 10.0
    target_actor_tau: float = 0.005

    q_actor_weight: float = 1.0
    proxy_actor_weight: float = 0.15
    offline_distill_weight_initial: float = 1.0
    offline_distill_weight_final: float = 0.2
    offline_distill_decay_steps: int = 50_000
    correction_weight: float = 1.0

    proximal_weight_initial: float = 1.0
    proximal_weight_min: float = 0.05
    proximal_decay_steps: int = 20_000
    proximal_speed_weight: float = 0.5
    proximal_lateral_weight: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "learning_rate": self.learning_rate,
            "gradient_clip_norm": self.gradient_clip_norm,
            "target_actor_tau": self.target_actor_tau,
            "offline_distill_decay_steps": self.offline_distill_decay_steps,
            "proximal_decay_steps": self.proximal_decay_steps,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.target_actor_tau > 1.0:
            raise ValueError("target_actor_tau must not exceed one")
        nonnegative = (
            self.q_actor_weight,
            self.proxy_actor_weight,
            self.offline_distill_weight_initial,
            self.offline_distill_weight_final,
            self.correction_weight,
            self.proximal_weight_initial,
            self.proximal_weight_min,
            self.proximal_speed_weight,
            self.proximal_lateral_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("Flow loss coefficients must be finite and non-negative")
        if self.offline_distill_weight_final > self.offline_distill_weight_initial:
            raise ValueError("offline distillation weight must not increase")
        if self.proximal_weight_min > self.proximal_weight_initial:
            raise ValueError("invalid Flow proximal weights")
        if self.proximal_speed_weight + self.proximal_lateral_weight <= 0.0:
            raise ValueError("at least one Flow proximal axis must be weighted")


def _weighted_mean(value, weight):
    value = jnp.asarray(value, dtype=jnp.float32)
    weight = jnp.asarray(weight, dtype=jnp.float32)
    return jnp.sum(value * weight) / jnp.maximum(jnp.sum(weight), 1.0)


def _copy_tree(tree):
    return jax.tree_util.tree_map(lambda value: jnp.array(value, copy=True), tree)


def _python_metrics(metrics):
    return {name: float(np.asarray(value)) for name, value in metrics.items()}


class GuardianDSACFlowV24:
    """V2.3.1 Critic/P/Guardian with a density-free one-step Flow Actor."""

    FORMAT = "guardian_dsac_one_step_flow_v2_4_v2"

    def __init__(
        self,
        base_algorithm,
        flow_policy: JaxOneStepFlowPolicy,
        flow_params: FlowPolicyParams,
        *,
        config: GuardianDSACFlowV24Config = GuardianDSACFlowV24Config(),
    ):
        if not bool(getattr(base_algorithm, "offline_reference_ready", False)):
            raise ValueError(
                "V2.4 requires a five-Seed shared offline Q/P checkpoint; "
                "a deployable Gaussian checkpoint is not required"
            )
        if int(flow_policy.act_dim) != 2:
            raise ValueError("Guardian-Flow-DSAC requires a two-dimensional action")
        self.base = base_algorithm
        self.flow_policy = flow_policy
        self.config = config
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.gradient_clip_norm),
            optax.adam(config.learning_rate),
        )
        one_step = _copy_tree(flow_params.one_step)
        self.state = FlowActorTrainState(
            params=flow_params,
            bc_opt_state=self.optimizer.init(flow_params.bc_flow),
            actor_opt_state=self.optimizer.init(flow_params.one_step),
            target_one_step=_copy_tree(one_step),
            safe_one_step=_copy_tree(one_step),
            gradient_step=jnp.asarray(0, dtype=jnp.int32),
            offline_joint_step=jnp.asarray(0, dtype=jnp.int32),
            proximal_step=jnp.asarray(0, dtype=jnp.int32),
            bc_flow_frozen=jnp.asarray(False),
        )
        self.safe_full_snapshot = (
            _copy_tree(self.base.state),
            _copy_tree(self.state),
        )
        self.deployable_checkpoint = None
        self._get_action = jax.jit(self.flow_policy.get_action)
        self._get_deterministic_action = jax.jit(
            self.flow_policy.get_deterministic_action
        )
        self._build_updates()

    def _build_updates(self):
        config = self.config

        def bc_update(key, state, data):
            noise_key, time_key = jax.random.split(key)
            action_shape = jnp.asarray(data.action).shape
            noise = jax.random.normal(noise_key, action_shape, dtype=jnp.float32)
            time = jax.random.uniform(
                time_key,
                (*action_shape[:-1], 1),
                dtype=jnp.float32,
            )

            def loss_fn(params):
                return flow_matching_loss(
                    self.flow_policy,
                    params,
                    data.obs,
                    data.action,
                    noise,
                    time,
                )

            loss, grads = jax.value_and_grad(loss_fn)(state.params.bc_flow)
            updates, opt_state = self.optimizer.update(
                grads, state.bc_opt_state, state.params.bc_flow
            )
            params = optax.apply_updates(state.params.bc_flow, updates)
            return state._replace(
                params=state.params._replace(bc_flow=params),
                bc_opt_state=opt_state,
                gradient_step=state.gradient_step + 1,
            ), {"flow_bc_loss": loss}

        def distill_update(key, state, data):
            noise = jax.random.normal(
                key, jnp.asarray(data.action).shape, dtype=jnp.float32
            )

            def loss_fn(params):
                return one_step_distillation_loss(
                    self.flow_policy,
                    params,
                    state.params.bc_flow,
                    data.obs,
                    noise,
                )

            loss, grads = jax.value_and_grad(loss_fn)(state.params.one_step)
            updates, opt_state = self.optimizer.update(
                grads, state.actor_opt_state, state.params.one_step
            )
            params = optax.apply_updates(state.params.one_step, updates)
            return state._replace(
                params=state.params._replace(one_step=params),
                actor_opt_state=opt_state,
                target_one_step=_copy_tree(params),
                gradient_step=state.gradient_step + 1,
            ), {"one_step_distillation_loss": loss}

        def offline_actor_update(key, state, data, q1, q2, proxy):
            action_shape = jnp.asarray(data.action).shape
            policy_key, diversity_1_key, diversity_2_key = jax.random.split(key, 3)
            noise = jax.random.normal(policy_key, action_shape, dtype=jnp.float32)
            diversity_noise_1 = jax.random.normal(
                diversity_1_key, action_shape, dtype=jnp.float32
            )
            diversity_noise_2 = jax.random.normal(
                diversity_2_key, action_shape, dtype=jnp.float32
            )
            teacher = jax.lax.stop_gradient(
                self.flow_policy.bc_action(state.params.bc_flow, data.obs, noise)
            )
            proxy_support = jnp.maximum(
                jnp.asarray(data.guardian_dependence_target_valid, jnp.float32),
                jnp.asarray(data.guardian_dependence_corrected_valid, jnp.float32),
            )
            progress = jnp.clip(
                jnp.asarray(state.offline_joint_step, dtype=jnp.float32)
                / float(max(config.offline_distill_decay_steps, 1)),
                0.0,
                1.0,
            )
            distill_coefficient = (
                config.offline_distill_weight_initial
                + progress
                * (
                    config.offline_distill_weight_final
                    - config.offline_distill_weight_initial
                )
            )

            def loss_fn(params):
                action = self.flow_policy.get_action(params, data.obs, noise)
                mean_1, _ = self.base.agent.q_distribution(q1, data.obs, action)
                mean_2, _ = self.base.agent.q_distribution(q2, data.obs, action)
                q_loss = -jnp.mean(jnp.minimum(mean_1, mean_2))
                proxy_logit = self.base.agent.proxy_value(proxy, data.obs, action)
                p_probability = jax.nn.sigmoid(proxy_logit)
                p_loss = _weighted_mean(
                    proxy_actor_objective(
                        proxy_logit,
                        objective=self.base.config.proxy_actor_objective,
                    ),
                    proxy_support,
                )
                distill = jnp.mean(
                    jnp.sum(jnp.square(action - teacher), axis=-1)
                )
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
                    + config.proxy_actor_weight * p_loss
                    + distill_coefficient * distill
                )
                return total, (q_loss, p_loss, p_probability, distill, diversity)

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params.one_step
            )
            updates, opt_state = self.optimizer.update(
                grads, state.actor_opt_state, state.params.one_step
            )
            params = optax.apply_updates(state.params.one_step, updates)
            target = optax.incremental_update(
                params, state.target_one_step, config.target_actor_tau
            )
            q_loss, p_loss, p_probability, distill, diversity = aux
            return state._replace(
                params=state.params._replace(one_step=params),
                actor_opt_state=opt_state,
                target_one_step=target,
                gradient_step=state.gradient_step + 1,
                offline_joint_step=state.offline_joint_step + 1,
            ), {
                "flow_offline_actor_loss": loss,
                "flow_offline_q_loss": q_loss,
                "flow_offline_proxy_loss": p_loss,
                "flow_offline_absolute_proxy": _weighted_mean(
                    p_probability, proxy_support
                ),
                "flow_offline_distillation_loss": distill,
                "flow_offline_distillation_coefficient": distill_coefficient,
                "flow_latent_action_diversity": diversity,
            }

        def online_actor_update(key, state, data, q1, q2, proxy):
            action_shape = jnp.asarray(data.action).shape
            policy_key, diversity_1_key, diversity_2_key = jax.random.split(key, 3)
            noise = jax.random.normal(policy_key, action_shape, dtype=jnp.float32)
            diversity_noise_1 = jax.random.normal(
                diversity_1_key, action_shape, dtype=jnp.float32
            )
            diversity_noise_2 = jax.random.normal(
                diversity_2_key, action_shape, dtype=jnp.float32
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
                corrected_prediction = self.flow_policy.get_action(
                    params, data.obs, stored_noise
                )
                correction_square = jnp.square(
                    corrected_prediction - jax.lax.stop_gradient(correction_target)
                )
                correction = _weighted_mean(
                    jnp.sum(correction_square, axis=-1),
                    correction_sample_weight,
                )
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
                    + config.proxy_actor_weight * p_loss
                    + coefficient * proximal
                    + config.correction_weight * correction
                )
                return total, (
                    q_loss,
                    p_loss,
                    p_probability,
                    proximal,
                    speed_drift,
                    lateral_drift,
                    correction,
                    correction_square,
                    diversity,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params.one_step
            )
            updates, opt_state = self.optimizer.update(
                grads, state.actor_opt_state, state.params.one_step
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
                correction_square,
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
                "flow_correction_support_fraction": jnp.mean(
                    correction_support.astype(jnp.float32)
                ),
                "flow_correction_weight_mean": jnp.mean(correction_sample_weight),
                "flow_correction_error_speed": _weighted_mean(
                    correction_square[..., 0], correction_sample_weight
                ),
                "flow_correction_error_lateral": _weighted_mean(
                    correction_square[..., 1], correction_sample_weight
                ),
                "flow_correction_proxy_target_fraction": _weighted_mean(
                    pair_valid.astype(jnp.float32), correction_sample_weight
                ),
                "flow_latent_action_diversity": diversity,
            }

        self._bc_update = jax.jit(bc_update)
        self._distill_update = jax.jit(distill_update)
        self._offline_actor_update = jax.jit(offline_actor_update)
        self._online_actor_update = jax.jit(online_actor_update)

    @property
    def bc_flow_frozen(self) -> bool:
        return bool(np.asarray(self.state.bc_flow_frozen))

    def pretrain_bc_flow(self, key, data):
        if self.bc_flow_frozen:
            raise RuntimeError("the offline Flow teacher is frozen")
        self.state, metrics = self._bc_update(key, self.state, data)
        return _python_metrics(metrics)

    def distill_one_step(self, key, data):
        if self.bc_flow_frozen:
            raise RuntimeError("distillation cannot continue after teacher freeze")
        self.state, metrics = self._distill_update(key, self.state, data)
        return _python_metrics(metrics)

    def freeze_bc_flow(self):
        """Freeze the teacher and bind target/safe Actors to the distilled map."""

        one_step = _copy_tree(self.state.params.one_step)
        self.state = self.state._replace(
            target_one_step=_copy_tree(one_step),
            safe_one_step=_copy_tree(one_step),
            bc_flow_frozen=jnp.asarray(True),
            proximal_step=jnp.asarray(0, dtype=jnp.int32),
        )
        self.safe_full_snapshot = (
            _copy_tree(self.base.state),
            _copy_tree(self.state),
        )
        return {"bc_flow_frozen": True, "target_one_step_synchronised": True}

    def offline_flow_joint_update(self, key, data):
        if not self.bc_flow_frozen:
            raise RuntimeError("freeze the Flow teacher before offline joint training")
        critic_key, next_key, policy_key, actor_key = jax.random.split(key, 4)
        action_shape = jnp.asarray(data.action).shape
        next_noise = jax.random.normal(next_key, action_shape, dtype=jnp.float32)
        policy_noise = jax.random.normal(policy_key, action_shape, dtype=jnp.float32)
        next_action = self.flow_policy.get_action(
            self.state.target_one_step, data.next_obs, next_noise
        )
        policy_action = self.flow_policy.get_action(
            self.state.params.one_step, data.obs, policy_noise
        )
        q_metrics = self.base.offline_critic_update_with_policy_actions(
            critic_key,
            data,
            next_action=next_action,
            policy_action=policy_action,
        )
        p_metrics = self.base.proxy_update(data)
        self.state, actor_metrics = self._offline_actor_update(
            actor_key,
            self.state,
            data,
            self.base.state.params.q1,
            self.base.state.params.q2,
            self.base.state.params.proxy,
        )
        return {
            **{f"task_{name}": value for name, value in q_metrics.items()},
            **p_metrics,
            **_python_metrics(actor_metrics),
            "flow_implicit_entropy_target": 0.0,
            "flow_actor_entropy_free": 1.0,
        }

    def refresh_offline_reference(self, validation_data):
        """Freeze the post-joint Q/P support reference without certifying Actor."""

        thresholds = self.base.mark_offline_reference(validation_data)
        safe = _copy_tree(self.state.safe_one_step)
        safe_flow_state = self.state._replace(
            params=self.state.params._replace(one_step=_copy_tree(safe)),
            actor_opt_state=self.optimizer.init(safe),
            target_one_step=_copy_tree(safe),
            proximal_step=jnp.asarray(0, dtype=jnp.int32),
        )
        self.safe_full_snapshot = (
            _copy_tree(self.base.state),
            _copy_tree(safe_flow_state),
        )
        return thresholds

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
        q_metrics = self.base.online_critic_update_with_policy_actions(
            critic_key,
            data,
            next_action=next_action,
            policy_action=policy_action,
        )
        p_metrics = self.base.proxy_update(data)
        metrics = {
            **{f"task_{name}": value for name, value in q_metrics.items()},
            **p_metrics,
            "flow_implicit_entropy_target": 0.0,
            "flow_actor_entropy_free": 1.0,
            "flow_bc_frozen": 1.0,
        }
        if update_actor:
            actor_batch = data if actor_data is None else actor_data
            self.state, actor_metrics = self._online_actor_update(
                actor_key,
                self.state,
                actor_batch,
                self.base.state.params.q1,
                self.base.state.params.q2,
                self.base.state.params.proxy,
            )
            metrics.update(_python_metrics(actor_metrics))
        return metrics

    def get_action(self, observation, noise):
        observation = np.asarray(observation, dtype=np.float32)
        noise = np.asarray(noise, dtype=np.float32)
        expected = (*observation.shape[:-1], self.flow_policy.act_dim)
        if noise.shape != expected:
            raise ValueError(f"Flow noise shape {noise.shape} does not match {expected}")
        return np.array(
            self._get_action(self.state.params.one_step, observation, noise),
            copy=True,
        )

    def get_deterministic_action(self, observation):
        return np.array(
            self._get_deterministic_action(
                self.state.params.one_step, observation
            ),
            copy=True,
        )

    def get_deterministic_hybrid_safe_action(self, observation):
        return np.array(
            self._get_deterministic_action(self.state.safe_one_step, observation),
            copy=True,
        )

    def get_deterministic_deployable_action(self, observation):
        if self.deployable_checkpoint is None:
            raise RuntimeError("no V2.4 deployable checkpoint has been certified")
        flow_state = self.deployable_checkpoint["flow_state"]
        return np.array(
            self._get_deterministic_action(
                flow_state.params.one_step, observation
            ),
            copy=True,
        )

    def select_evaluation_checkpoint(self, policy: str):
        if policy == "learner":
            return
        if policy == "hybrid_safe":
            base_state, flow_state = self.safe_full_snapshot
            self.base.state = _copy_tree(base_state)
            self.state = _copy_tree(flow_state)
            return
        if policy == "deployable":
            if self.deployable_checkpoint is None:
                raise RuntimeError("no V2.4 deployable checkpoint is available")
            self.base.state = _copy_tree(self.deployable_checkpoint["base_state"])
            self.state = _copy_tree(self.deployable_checkpoint["flow_state"])
            return
        raise ValueError(f"unknown evaluation policy: {policy}")

    @property
    def scene_signatures(self):
        return self.base.scene_signatures

    @property
    def action_semantics_version(self):
        return self.base.action_semantics_version

    def task_q_distribution(self, observation, action):
        return self.base.task_q_distribution(observation, action)

    def proxy_value(self, observation, action):
        return self.base.proxy_value(observation, action)

    def numerical_disaster_reasons(self, **kwargs):
        reasons = list(self.base.numerical_disaster_reasons(**kwargs))
        leaves = jax.tree_util.tree_leaves(self.state)
        if any(not np.all(np.isfinite(np.asarray(value))) for value in leaves):
            reasons.append("non_finite_flow_actor")
        return sorted(set(reasons))

    def promote_hybrid_safe(self):
        """Move the safe one-step snapshot without resetting proximal decay."""

        self.base.promote_hybrid_safe()
        one_step = _copy_tree(self.state.params.one_step)
        self.state = self.state._replace(safe_one_step=one_step)
        self.safe_full_snapshot = (
            _copy_tree(self.base.state),
            _copy_tree(self.state),
        )
        return {"snapshot": "hybrid_safe_snapshot", "rollback": "none"}

    def reject_hybrid_safety(self):
        """Restore only the Flow Actor; shared Q/P learning is retained."""

        safe = _copy_tree(self.state.safe_one_step)
        self.state = self.state._replace(
            params=self.state.params._replace(one_step=safe),
            actor_opt_state=self.optimizer.init(safe),
            target_one_step=_copy_tree(safe),
            proximal_step=jnp.asarray(0, dtype=jnp.int32),
        )
        return {"rollback": "actor_only", "snapshot": "hybrid_safe_snapshot"}

    def reject_numerical_disaster(self):
        _, flow_state = self.safe_full_snapshot
        self.base.reject_numerical_disaster()
        self.state = _copy_tree(flow_state)
        return {"rollback": "hybrid_safe_full_snapshot"}

    def record_ordinary_failure(self):
        return {"rollback": "none", "deployment": "rejected"}

    def promote_deployable(self, metrics=None):
        self.base.promote_deployable(metrics)
        self.deployable_checkpoint = {
            "base_state": _copy_tree(self.base.state),
            "flow_state": _copy_tree(self.state),
            "metrics": dict(metrics or {}),
        }

    @property
    def has_deployable_checkpoint(self):
        return self.deployable_checkpoint is not None

    def _validate_loaded_state(self):
        deployable_state = (
            None
            if self.deployable_checkpoint is None
            else (
                self.deployable_checkpoint["base_state"],
                self.deployable_checkpoint["flow_state"],
            )
        )
        leaves = jax.tree_util.tree_leaves(
            (self.state, self.safe_full_snapshot, deployable_state)
        )
        if any(not np.all(np.isfinite(np.asarray(value))) for value in leaves):
            raise ValueError("V2.4 checkpoint contains non-finite state")
        if not self.bc_flow_frozen:
            raise ValueError("deployable V2.4 checkpoint has an unfrozen teacher")
        probe = jnp.zeros((1, self.flow_policy.actor_obs_dim), dtype=jnp.float32)
        action = self.flow_policy.get_deterministic_action(
            self.state.params.one_step, probe
        )
        target_action = self.flow_policy.get_deterministic_action(
            self.state.target_one_step, probe
        )
        if action.shape != (1, self.flow_policy.act_dim):
            raise ValueError("V2.4 one-step Actor output dimension mismatch")
        if target_action.shape != action.shape:
            raise ValueError("V2.4 target one-step Actor output mismatch")
        if not np.all(np.isfinite(np.asarray((action, target_action)))):
            raise ValueError("V2.4 one-step Actor output is non-finite")

    def _checkpoint_payload(self):
        return {
            "format": self.FORMAT,
            "base_v2_3": self.base._checkpoint_payload(),
            "flow_state": jax.device_get(self.state),
            "safe_full_snapshot": jax.device_get(self.safe_full_snapshot),
            "deployable_checkpoint": jax.device_get(self.deployable_checkpoint),
            "flow_config": asdict(self.config),
            "actor_observation_dim": int(self.flow_policy.actor_obs_dim),
            "action_dim": int(self.flow_policy.act_dim),
            "hidden_sizes": list(self.flow_policy.hidden_sizes),
            "integration_steps": int(self.flow_policy.integration_steps),
            "candidate_sampling": False,
            "q_reranking": False,
            "rejection_sampling": False,
            "reverse_ode_likelihood": False,
            "flow_internal_cbf": False,
            "explicit_actor_entropy": False,
            "teacher_online_updates": False,
            "replay_schema": "guardian_flow_dsac_experience_v1",
        }

    def save(self, path: str):
        payload = self._checkpoint_payload()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            pickle.dump(payload, stream)

    def _load_checkpoint_payload(self, payload, *, load_base: bool = True):
        if not isinstance(payload, dict) or payload.get("format") != self.FORMAT:
            raise ValueError("checkpoint is not formal Guardian-Flow-DSAC V2.4")
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
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"V2.4 checkpoint missing fields: {missing}")
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
            raise ValueError("V2.4 checkpoint enables a forbidden Flow mechanism")
        if payload["replay_schema"] != "guardian_flow_dsac_experience_v1":
            raise ValueError("V2.4 replay schema binding mismatch")
        expected = (
            self.flow_policy.actor_obs_dim,
            self.flow_policy.act_dim,
            tuple(self.flow_policy.hidden_sizes),
            self.flow_policy.integration_steps,
        )
        observed = (
            int(payload["actor_observation_dim"]),
            int(payload["action_dim"]),
            tuple(int(value) for value in payload["hidden_sizes"]),
            int(payload["integration_steps"]),
        )
        if observed != expected:
            raise ValueError("V2.4 flow architecture binding mismatch")
        if dict(payload["flow_config"]) != asdict(self.config):
            raise ValueError("V2.4 flow config binding mismatch")
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
        self._validate_loaded_state()

    def load(self, path: str):
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        self._load_checkpoint_payload(payload)


__all__ = [
    "FlowActorTrainState",
    "GuardianDSACFlowV24",
    "GuardianDSACFlowV24Config",
]
