"""Guardian-Calibrated Offline-to-Online Safe SAC.

The implementation keeps two value semantics deliberately separate:

* ``Q_R1/Q_R2`` estimate task return on the action that was really executed;
* ``P`` is either the legacy signed Guardian boundary score or the V2
  dependency regressor, selected without changing its network architecture.

A rejected raw proposal never receives the corrected transition.  It enters
only the same-state raw/corrected pairwise proxy loss.  This is the central
credit-assignment invariant of the next SAFE100 training pipeline.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import Literal, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.base import Algorithm
from relax.network.safe_sac_jax import (
    GuardianCalQLParams,
    JaxGuardianCalQLNet,
)
from relax.utils.experience import GuardianCalQLExperience
from relax.utils.typing import Metric


OfflineCriticMode = Literal["td", "cql", "calql"]
OnlineCriticMode = Literal["td", "calql"]
ProxyObjective = Literal["signed_td", "dependency_regression"]


class GuardianCalQLOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    proxy: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class GuardianCalQLTrainState(NamedTuple):
    params: GuardianCalQLParams
    opt_state: GuardianCalQLOptStates
    proxy_beta: jax.Array
    gradient_step: jax.Array
    # Default preserves unpickling compatibility with v2 checkpoints that
    # predate iterative Champion trust-region scheduling.
    trust_coefficient: jax.Array = 0.0


@dataclass(frozen=True)
class GuardianCalQLConfig:
    gamma: float = 0.99
    proxy_gamma: float = 0.97
    tau: float = 0.005
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 1e-4
    proxy_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    reward_scale: float = 1.0
    gradient_clip_norm: float = 10.0
    q_lcb_beta: float = 1.0

    calql_weight: float = 1.0
    cql_temperature: float = 1.0
    # Online defaults to ordinary TD to avoid retaining excessive Cal-QL
    # conservatism after deployment.  ``calql`` remains an explicit ablation.
    online_critic_mode: OnlineCriticMode = "td"
    q_mc_weight: float = 1.0
    q_mc_huber_delta: float = 1.0
    # Anchor online Q_R to the accepted offline Cal-QL reference.  The
    # coefficient is linearly decayed by online task-critic update count.
    q_anchor_weight_initial: float = 0.0
    q_anchor_weight_final: float = 0.0
    q_anchor_decay_steps: int = 1

    proxy_objective: ProxyObjective = "signed_td"
    proxy_td_weight: float = 1.0
    # Dependency labels are defined on [0, 1]. Bootstrap targets outside
    # that support are OOD extrapolation, not stronger-dependence evidence.
    proxy_td_target_min: float = 0.0
    proxy_td_target_max: float = 1.0
    proxy_regression_weight: float = 1.0
    proxy_pair_loss_weight: float = 1.0
    proxy_pair_margin: float = 0.5
    proxy_l2_weight: float = 1e-3
    proxy_min_value: float = 0.0
    proxy_dependency_safe_threshold: float = 0.25
    proxy_relative_margin: float = 0.0
    proxy_actor_weight_init: float = 0.1
    proxy_dual_learning_rate: float = 1e-3
    # Target average squared hinge violation, not a target proxy magnitude.
    proxy_target: float = 0.02
    proxy_beta_max: float = 100.0
    # The signed boundary proxy is supervised only near audited Guardian
    # boundaries.  Applying its arbitrary zero level to every replay state
    # creates global Actor drift without additional safety evidence.
    proxy_actor_boundary_only: bool = False

    actor_sac_weight: float = 1.0
    awac_weight: float = 0.0
    awac_temperature: float = 1.0
    awac_max_weight: float = 20.0
    teacher_bc_weight: float = 1.0
    # Direct deterministic-action anchor on every sampled transition.  This
    # complements distribution KL when the Champion variance is broad and
    # prevents a few boundary states from taking an unsafe oversized step.
    champion_action_weight: float = 1.0
    # Additional local anchor on audited intervention/pair states.  A global
    # mean can dilute the rare boundary rows that actually determine Guardian
    # dependence, while this term leaves non-boundary task improvements free.
    champion_boundary_action_weight: float = 0.0
    champion_kl_weight: float = 0.5
    champion_kl_min: float = 0.0
    champion_kl_target: float = 0.0075
    champion_kl_dual_learning_rate: float = 5e-2
    champion_kl_failure_multiplier: float = 2.0
    champion_kl_validation_relax_multiplier: float = 0.95
    champion_kl_max: float = 5.0
    initial_policy_kl_weight: float = 0.0
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
    variance = jnp.square(std)
    reference_variance = jnp.maximum(jnp.square(reference_std), 1e-8)
    per_dimension = (
        jnp.log(jnp.maximum(reference_std, 1e-8) / jnp.maximum(std, 1e-8))
        + (variance + jnp.square(mean - reference_mean)) / (2.0 * reference_variance)
        - 0.5
    )
    return jnp.sum(per_dimension, axis=-1)


def _copy_tree(tree):
    return jax.tree_util.tree_map(lambda value: jnp.array(value, copy=True), tree)


def _python_metrics(metrics: Metric) -> Metric:
    return {name: float(value) for name, value in metrics.items()}


class GuardianCalQLSafeSAC(Algorithm):
    """Cal-QL task value + PVP-style intervention proxy + safe online SAC."""

    FORMAT = "guardian_calibrated_offline_to_online_safe_sac_v5"

    def __init__(
        self,
        agent: JaxGuardianCalQLNet,
        params: GuardianCalQLParams,
        *,
        config: GuardianCalQLConfig = GuardianCalQLConfig(),
    ):
        self.agent = agent
        self.config = config
        if config.online_critic_mode not in ("td", "calql"):
            raise ValueError(
                f"unsupported online critic mode: {config.online_critic_mode}"
            )
        if config.proxy_td_target_max < config.proxy_td_target_min:
            raise ValueError(
                "proxy_td_target_max must be >= proxy_td_target_min"
            )
        clipped = lambda learning_rate: optax.chain(
            optax.clip_by_global_norm(config.gradient_clip_norm),
            optax.adam(learning_rate),
        )
        self.critic_optim = clipped(config.critic_learning_rate)
        self.actor_optim = clipped(config.actor_learning_rate)
        self.proxy_optim = clipped(config.proxy_learning_rate)
        self.log_alpha_optim = optax.adam(config.alpha_learning_rate)
        self.state = GuardianCalQLTrainState(
            params=params,
            opt_state=GuardianCalQLOptStates(
                q1=self.critic_optim.init(params.q1),
                q2=self.critic_optim.init(params.q2),
                proxy=self.proxy_optim.init(params.proxy),
                policy=self.actor_optim.init(params.policy),
                log_alpha=self.log_alpha_optim.init(params.log_alpha),
            ),
            proxy_beta=jnp.asarray(
                max(config.proxy_actor_weight_init, 0.0), dtype=jnp.float32
            ),
            gradient_step=jnp.asarray(0, dtype=jnp.int32),
            trust_coefficient=jnp.asarray(
                max(config.champion_kl_weight, 0.0), dtype=jnp.float32
            ),
        )
        self.initial_policy_params = _copy_tree(params.policy)
        self.champion_state = _copy_tree(self.state)
        self.last_safe_policy = _copy_tree(params.policy)
        self.last_safe_policy_opt_state = _copy_tree(self.state.opt_state.policy)
        self.offline_reference_q1 = _copy_tree(params.q1)
        self.offline_reference_q2 = _copy_tree(params.q2)
        self.online_q_anchor_step = 0
        self.offline_q_reference_ready = False
        self._fresh_proxy_params = _copy_tree(params.proxy)
        self.loaded_checkpoint_format: str | None = None

        self._get_action = jax.jit(self.agent.get_action)
        self._get_deterministic_action = jax.jit(self.agent.get_deterministic_action)
        self._task_q_values = jax.jit(
            lambda q1, q2, observation, action: (
                self.agent.q(q1, observation, action),
                self.agent.q(q2, observation, action),
            )
        )
        self._proxy_value_eval = jax.jit(self.agent.proxy_value)
        self._build_updates()

    def _build_updates(self) -> None:
        config = self.config

        def make_critic_update(mode: OfflineCriticMode):
            if mode not in ("td", "cql", "calql"):
                raise ValueError(f"unsupported offline critic mode: {mode}")

            def critic_update(
                key,
                state,
                data,
                offline_reference_q1,
                offline_reference_q2,
                anchor_weight,
                mc_weight,
            ):
                params = state.params
                next_key, policy_key, random_key = jax.random.split(key, 3)
                next_action, next_logp = self.agent.evaluate(
                    next_key, params.policy, data.next_obs
                )
                target_q1 = self.agent.q(params.target_q1, data.next_obs, next_action)
                target_q2 = self.agent.q(params.target_q2, data.next_obs, next_action)
                backup = jax.lax.stop_gradient(
                    config.reward_scale * jnp.asarray(data.reward, dtype=jnp.float32)
                    + (1.0 - jnp.asarray(data.done, dtype=jnp.float32))
                    * config.gamma
                    * (
                        jnp.minimum(target_q1, target_q2)
                        - jnp.exp(params.log_alpha) * next_logp
                    )
                )
                policy_action, _ = self.agent.evaluate(
                    policy_key, params.policy, data.obs
                )
                random_action = jax.random.uniform(
                    random_key,
                    shape=jnp.asarray(data.action).shape,
                    minval=-self.agent.action_scale,
                    maxval=self.agent.action_scale,
                )
                reference = jnp.asarray(data.behavior_return_to_go, dtype=jnp.float32)
                reference_valid = jnp.logical_and(
                    jnp.asarray(data.behavior_return_valid, dtype=bool),
                    jnp.isfinite(reference),
                )
                safe_reference = jnp.where(reference_valid, reference, 0.0)
                anchor_valid = jnp.asarray(data.q_anchor_valid, dtype=jnp.float32)
                temperature = max(config.cql_temperature, 1e-6)

                anchor_weight = jnp.asarray(anchor_weight, dtype=jnp.float32)
                mc_weight = jnp.asarray(mc_weight, dtype=jnp.float32)

                def loss_fn(q_params: hk.Params, offline_reference_params: hk.Params):
                    replay_q = self.agent.q(q_params, data.obs, data.action)
                    bellman_loss = 0.5 * jnp.mean(jnp.square(replay_q - backup))
                    policy_q = self.agent.q(q_params, data.obs, policy_action)
                    random_q = self.agent.q(q_params, data.obs, random_action)
                    candidates = jnp.stack([policy_q, random_q], axis=-1)
                    if mode == "calql":
                        calibrated = jnp.maximum(
                            candidates, safe_reference[..., None]
                        )
                        candidates = jnp.where(
                            reference_valid[..., None],
                            calibrated,
                            candidates,
                        )
                    conservative_gap = jnp.mean(
                        temperature
                        * jax.scipy.special.logsumexp(candidates / temperature, axis=-1)
                        - temperature * math.log(candidates.shape[-1])
                        - replay_q
                    )
                    conservative_weight = config.calql_weight if mode != "td" else 0.0
                    offline_reference_q = jax.lax.stop_gradient(
                        self.agent.q(
                            offline_reference_params,
                            data.obs,
                            data.action,
                        )
                    )
                    anchor_loss = 0.5 * _weighted_mean(
                        jnp.square(replay_q - offline_reference_q),
                        anchor_valid,
                    )
                    mc_loss = _weighted_mean(
                        optax.huber_loss(
                            replay_q,
                            safe_reference,
                            delta=max(config.q_mc_huber_delta, 1e-6),
                        ),
                        reference_valid.astype(jnp.float32),
                    )
                    loss = (
                        bellman_loss
                        + conservative_weight * conservative_gap
                        + mc_weight * mc_loss
                        + anchor_weight * anchor_loss
                    )
                    return loss, (
                        replay_q,
                        bellman_loss,
                        conservative_gap,
                        mc_loss,
                        anchor_loss,
                        policy_q,
                        random_q,
                    )

                (q1_loss, q1_aux), q1_grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    params.q1, offline_reference_q1
                )
                (q2_loss, q2_aux), q2_grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    params.q2, offline_reference_q2
                )
                q1_updates, q1_opt_state = self.critic_optim.update(
                    q1_grads, state.opt_state.q1, params.q1
                )
                q2_updates, q2_opt_state = self.critic_optim.update(
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
                        q1=q1_opt_state, q2=q2_opt_state
                    ),
                    gradient_step=state.gradient_step + 1,
                )
                (
                    q1_value,
                    q1_bellman,
                    q1_gap,
                    q1_mc,
                    q1_anchor,
                    q1_policy,
                    q1_random,
                ) = q1_aux
                (
                    q2_value,
                    q2_bellman,
                    q2_gap,
                    q2_mc,
                    q2_anchor,
                    q2_policy,
                    q2_random,
                ) = q2_aux
                q_min = jnp.minimum(q1_value, q2_value)
                return next_state, {
                    "q1_loss": q1_loss,
                    "q2_loss": q2_loss,
                    "bellman_loss": 0.5 * (q1_bellman + q2_bellman),
                    "conservative_gap": 0.5 * (q1_gap + q2_gap),
                    "mc_loss": 0.5 * (q1_mc + q2_mc),
                    "mc_coefficient": mc_weight,
                    "mc_support_fraction": jnp.mean(
                        reference_valid.astype(jnp.float32)
                    ),
                    "q_anchor_loss": 0.5 * (q1_anchor + q2_anchor),
                    "q_anchor_coefficient": anchor_weight,
                    "q_anchor_support_fraction": jnp.mean(anchor_valid),
                    "q_mean": jnp.mean(q_min),
                    "q_disagreement": jnp.mean(jnp.abs(q1_value - q2_value)),
                    "q_policy_mean": jnp.mean(jnp.minimum(q1_policy, q2_policy)),
                    "q_random_mean": jnp.mean(jnp.minimum(q1_random, q2_random)),
                    "backup_mean": jnp.mean(backup),
                    "q_calibration_mae": _weighted_mean(
                        jnp.abs(q_min - safe_reference),
                        reference_valid.astype(jnp.float32),
                    ),
                }

            return jax.jit(critic_update)

        self._offline_td_update = make_critic_update("td")
        self._offline_cql_update = make_critic_update("cql")
        self._offline_calql_update = make_critic_update("calql")

        def proxy_update(key, state, data):
            del key
            params = state.params
            # Deployment and validation use the deterministic Actor. Using
            # the same action here avoids propagating stochastic OOD proxy
            # estimates through an otherwise deterministic fixed scene.
            next_action = self.agent.get_deterministic_action(
                params.policy, data.next_obs
            )
            raw_target_next_proxy = self.agent.proxy_value(
                params.target_proxy, data.next_obs, next_action
            )
            if config.proxy_objective == "dependency_regression":
                target_next_proxy = jnp.clip(
                    raw_target_next_proxy,
                    config.proxy_td_target_min,
                    config.proxy_td_target_max,
                )
                target_clip_fraction = jnp.mean(
                    jnp.logical_or(
                        raw_target_next_proxy < config.proxy_td_target_min,
                        raw_target_next_proxy > config.proxy_td_target_max,
                    ).astype(jnp.float32)
                )
            else:
                target_next_proxy = raw_target_next_proxy
                target_clip_fraction = jnp.asarray(0.0, dtype=jnp.float32)
            # Reward-free executed-transition propagation.  The current
            # intervention label supervises only the raw proposal; it is never
            # copied onto a corrected executed action.  Pair orientation plus
            # this target carries Guardian-dependence information backward.
            proxy_target = jax.lax.stop_gradient(
                (1.0 - jnp.asarray(data.done, dtype=jnp.float32))
                * config.proxy_gamma
                * target_next_proxy
            )
            pair_weight = jnp.asarray(
                data.proxy_pair_valid, dtype=jnp.float32
            ) * jnp.maximum(
                jnp.asarray(data.proxy_pair_weight, dtype=jnp.float32),
                0.0,
            )
            dependency_label = jnp.asarray(
                data.intervention_hard, dtype=jnp.float32
            ) + 0.5 * jnp.asarray(data.intervention_soft, dtype=jnp.float32)
            regression_weight = jnp.asarray(
                data.proposal_evaluated, dtype=jnp.float32
            ) * (1.0 - jnp.asarray(data.intervention_numerical, dtype=jnp.float32))

            def loss_fn(proxy_params: hk.Params):
                executed_proxy = self.agent.proxy_value(
                    proxy_params, data.obs, data.action
                )
                corrected_proxy = self.agent.proxy_value(
                    proxy_params, data.obs, data.proxy_corrected_action
                )
                raw_proxy = self.agent.proxy_value(
                    proxy_params, data.obs, data.raw_action
                )
                executed_td_error = executed_proxy - proxy_target
                td_loss = jnp.mean(jnp.square(executed_td_error))
                if config.proxy_objective == "dependency_regression":
                    regression_error = raw_proxy - dependency_label
                    regression_loss = _weighted_mean(
                        jnp.square(regression_error), regression_weight
                    )
                    # Larger P means greater Guardian dependence in V2.
                    pair_error = jax.nn.softplus(
                        config.proxy_pair_margin - raw_proxy + corrected_proxy
                    )
                    pair_correct = raw_proxy > corrected_proxy
                    primary_loss = (
                        config.proxy_regression_weight * regression_loss
                        + config.proxy_td_weight * td_loss
                    )
                else:
                    regression_error = executed_td_error
                    regression_loss = td_loss
                    # Legacy v3 orientation: larger P means more permissive.
                    pair_error = jax.nn.softplus(
                        config.proxy_pair_margin - corrected_proxy + raw_proxy
                    )
                    pair_correct = corrected_proxy > raw_proxy
                    primary_loss = config.proxy_td_weight * regression_loss
                pair_loss = _weighted_mean(pair_error, pair_weight)
                pair_l2 = _weighted_mean(
                    0.5 * (jnp.square(corrected_proxy) + jnp.square(raw_proxy)),
                    pair_weight,
                )
                l2_loss = jnp.mean(jnp.square(executed_proxy)) + pair_l2
                loss = (
                    primary_loss
                    + config.proxy_pair_loss_weight * pair_loss
                    + config.proxy_l2_weight * l2_loss
                )
                return loss, (
                    executed_proxy,
                    corrected_proxy,
                    raw_proxy,
                    td_loss,
                    regression_loss,
                    pair_loss,
                    l2_loss,
                    executed_td_error,
                    regression_error,
                    pair_correct,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params.proxy)
            updates, proxy_opt_state = self.proxy_optim.update(
                grads, state.opt_state.proxy, params.proxy
            )
            proxy_params = optax.apply_updates(params.proxy, updates)
            (
                executed_proxy,
                corrected_proxy,
                raw_proxy,
                td_loss,
                regression_loss,
                pair_loss,
                l2_loss,
                executed_td_error,
                regression_error,
                pair_correct,
            ) = aux
            next_state = state._replace(
                params=params._replace(
                    proxy=proxy_params,
                    target_proxy=optax.incremental_update(
                        proxy_params, params.target_proxy, config.tau
                    ),
                ),
                opt_state=state.opt_state._replace(proxy=proxy_opt_state),
                gradient_step=state.gradient_step + 1,
            )
            pair_accuracy = _weighted_mean(
                pair_correct.astype(jnp.float32), pair_weight
            )
            return next_state, {
                "proxy_loss": loss,
                "proxy_td_loss": td_loss,
                "proxy_regression_loss": regression_loss,
                "proxy_pair_loss": pair_loss,
                "proxy_l2_loss": l2_loss,
                "proxy_td_mae": jnp.mean(jnp.abs(executed_td_error)),
                "proxy_regression_mae": _weighted_mean(
                    jnp.abs(regression_error), regression_weight
                ),
                "proxy_target_unclipped_mean": jnp.mean(
                    raw_target_next_proxy
                ),
                "proxy_target_mean": jnp.mean(proxy_target),
                "proxy_target_clip_fraction": target_clip_fraction,
                "proxy_dependency_label_mean": _weighted_mean(
                    dependency_label, regression_weight
                ),
                "proxy_executed_mean": jnp.mean(executed_proxy),
                "proxy_corrected_mean": _weighted_mean(corrected_proxy, pair_weight),
                "proxy_raw_mean": _weighted_mean(raw_proxy, pair_weight),
                "proxy_pair_accuracy": pair_accuracy,
                "proxy_pair_fraction": jnp.mean(
                    jnp.asarray(data.proxy_pair_valid, dtype=jnp.float32)
                ),
            }

        self._proxy_update = jax.jit(proxy_update)

        def actor_update(
            key,
            state,
            data,
            initial_policy_params,
            champion_policy_params,
            initial_kl_coefficient,
        ):
            params = state.params
            actor_support = jnp.asarray(data.actor_controlled, dtype=jnp.float32)
            raw_behavior_value = jnp.asarray(
                data.behavior_return_to_go, dtype=jnp.float32
            )
            behavior_value_valid = jnp.logical_and(
                jnp.asarray(data.behavior_return_valid, dtype=bool),
                jnp.isfinite(raw_behavior_value),
            )
            awac_support = actor_support * behavior_value_valid.astype(jnp.float32)
            proposal_weight = jnp.asarray(data.proposal_evaluated, dtype=jnp.float32)
            boundary_support = jnp.maximum(
                jnp.asarray(data.proxy_pair_valid, dtype=jnp.float32),
                jnp.asarray(data.intervention, dtype=jnp.float32),
            )
            boundary_action_weight = proposal_weight * boundary_support
            if config.proxy_actor_boundary_only:
                proposal_weight = boundary_action_weight
            teacher_weight = jnp.asarray(
                data.teacher_valid, dtype=jnp.float32
            ) * jnp.maximum(jnp.asarray(data.teacher_weight, dtype=jnp.float32), 0.0)
            alpha = jnp.exp(params.log_alpha)
            champion_action = jax.lax.stop_gradient(
                self.agent.get_deterministic_action(
                    champion_policy_params, data.obs
                )
            )

            def loss_fn(policy_params: hk.Params):
                action, logp = self.agent.evaluate(key, policy_params, data.obs)
                q1 = self.agent.q(params.q1, data.obs, action)
                q2 = self.agent.q(params.q2, data.obs, action)
                q_mean = 0.5 * (q1 + q2)
                q_disagreement = 0.5 * jnp.abs(q1 - q2)
                q_lcb = q_mean - config.q_lcb_beta * q_disagreement
                champion_q1 = self.agent.q(
                    params.q1, data.obs, champion_action
                )
                champion_q2 = self.agent.q(
                    params.q2, data.obs, champion_action
                )
                champion_q_lcb = 0.5 * (champion_q1 + champion_q2) - (
                    config.q_lcb_beta * 0.5 * jnp.abs(champion_q1 - champion_q2)
                )
                q_policy_minus_champion = q_lcb - champion_q_lcb

                behavior_q1 = self.agent.q(params.q1, data.obs, data.action)
                behavior_q2 = self.agent.q(params.q2, data.obs, data.action)
                behavior_q_mean = 0.5 * (behavior_q1 + behavior_q2)
                behavior_q_disagreement = 0.5 * jnp.abs(behavior_q1 - behavior_q2)
                behavior_q_lcb = behavior_q_mean - (
                    config.q_lcb_beta * behavior_q_disagreement
                )
                # V^mu is exact for offline/complete trajectories.  For
                # incomplete online rows use the sampled behaviour action as
                # a stop-gradient baseline; this preserves the SAC gradient
                # while keeping the reported advantage meaningful.
                behavior_value = jax.lax.stop_gradient(
                    jnp.where(
                        behavior_value_valid,
                        raw_behavior_value,
                        behavior_q_lcb,
                    )
                )
                policy_advantage = q_lcb - behavior_value
                supported_sac_loss = _weighted_mean(
                    alpha * logp - policy_advantage, actor_support
                )
                behavior_advantage = jax.lax.stop_gradient(
                    behavior_q_lcb - behavior_value
                )
                # AWAC is E_D over rows with an explicitly valid return-to-go
                # baseline.  A legitimate numerical zero remains supported.
                awac_temperature = max(config.awac_temperature, 1e-6)
                awac_max_weight = max(config.awac_max_weight, 1.0)
                awac_weight = jnp.exp(
                    jnp.clip(
                        behavior_advantage / awac_temperature,
                        -20.0,
                        math.log(awac_max_weight),
                    )
                )
                behavior_logp = self.agent.log_prob(
                    policy_params, data.obs, data.action
                )
                awac_loss = jnp.sum(
                    -behavior_logp * awac_weight * awac_support
                ) / jnp.maximum(jnp.sum(awac_support), 1.0)

                proxy_value = self.agent.proxy_value(params.proxy, data.obs, action)
                champion_proxy_value = jax.lax.stop_gradient(
                    self.agent.proxy_value(
                        params.proxy, data.obs, champion_action
                    )
                )
                p_policy_minus_champion = proxy_value - champion_proxy_value
                if config.proxy_objective == "dependency_regression":
                    proxy_risk_delta = p_policy_minus_champion
                    p_improved = p_policy_minus_champion <= 0.0
                else:
                    proxy_risk_delta = -p_policy_minus_champion
                    p_improved = p_policy_minus_champion >= 0.0
                proxy_violation = jax.nn.relu(
                    proxy_risk_delta - config.proxy_relative_margin
                )
                proxy_safe = proxy_risk_delta <= config.proxy_relative_margin
                proxy_actor_loss = _weighted_mean(
                    jnp.square(proxy_violation), proposal_weight
                )

                deterministic_action = self.agent.get_deterministic_action(
                    policy_params, data.obs
                )
                deterministic_q1 = self.agent.q(
                    params.q1, data.obs, deterministic_action
                )
                deterministic_q2 = self.agent.q(
                    params.q2, data.obs, deterministic_action
                )
                deterministic_lcb = 0.5 * (
                    deterministic_q1 + deterministic_q2
                ) - config.q_lcb_beta * 0.5 * jnp.abs(
                    deterministic_q1 - deterministic_q2
                )
                teacher_q1 = self.agent.q(params.q1, data.obs, data.teacher_action)
                teacher_q2 = self.agent.q(params.q2, data.obs, data.teacher_action)
                teacher_lcb = 0.5 * (teacher_q1 + teacher_q2) - (
                    config.q_lcb_beta * 0.5 * jnp.abs(teacher_q1 - teacher_q2)
                )
                teacher_advantage = jax.lax.stop_gradient(
                    teacher_lcb - deterministic_lcb
                )
                temperature = max(config.teacher_advantage_temperature, 1e-6)
                maximum = max(config.teacher_advantage_max_weight, 1.0)
                soft_gate = jnp.where(
                    teacher_advantage > config.teacher_advantage_margin,
                    jnp.exp(
                        jnp.clip(
                            (teacher_advantage - config.teacher_advantage_margin)
                            / temperature,
                            0.0,
                            math.log(maximum),
                        )
                    ),
                    0.0,
                )
                hard = jnp.asarray(data.intervention_hard, dtype=jnp.float32)
                numerical = jnp.asarray(data.intervention_numerical, dtype=jnp.float32)
                teacher_gate = jnp.where(
                    hard > 0.5,
                    1.0,
                    jnp.where(numerical > 0.5, 0.0, soft_gate),
                )
                # Ordinary, actor-controlled BC samples remain valid anchors.
                teacher_gate = jnp.where(
                    jnp.asarray(data.intervention, dtype=jnp.float32) < 0.5,
                    1.0,
                    teacher_gate,
                )
                gated_teacher_weight = teacher_weight * teacher_gate
                teacher_error = jnp.sum(
                    jnp.square(deterministic_action - data.teacher_action),
                    axis=-1,
                )
                teacher_bc_loss = _weighted_mean(teacher_error, gated_teacher_weight)

                mean, std = self.agent.distribution(policy_params, data.obs)
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
                initial_kl_loss = jnp.mean(initial_kl)
                champion_kl_loss = jnp.mean(champion_kl)
                champion_action_error = jnp.sum(
                    jnp.square(
                        deterministic_action - jax.lax.stop_gradient(champion_action)
                    ),
                    axis=-1,
                )
                champion_action_loss = jnp.mean(champion_action_error)
                champion_boundary_action_loss = _weighted_mean(
                    champion_action_error, boundary_action_weight
                )
                loss = (
                    config.actor_sac_weight * supported_sac_loss
                    + config.awac_weight * awac_loss
                    + state.proxy_beta * proxy_actor_loss
                    + config.teacher_bc_weight * teacher_bc_loss
                    + config.champion_action_weight * champion_action_loss
                    + config.champion_boundary_action_weight
                    * champion_boundary_action_loss
                    + state.trust_coefficient * champion_kl_loss
                    + initial_kl_coefficient * initial_kl_loss
                )
                return loss, (
                    logp,
                    q_lcb,
                    q_disagreement,
                    policy_advantage,
                    behavior_advantage,
                    behavior_q_lcb,
                    supported_sac_loss,
                    awac_loss,
                    awac_weight,
                    proxy_actor_loss,
                    teacher_bc_loss,
                    teacher_error,
                    teacher_gate,
                    teacher_advantage,
                    initial_kl_loss,
                    champion_kl_loss,
                    champion_action_loss,
                    champion_boundary_action_loss,
                    proxy_value,
                    champion_proxy_value,
                    p_policy_minus_champion,
                    p_improved,
                    proxy_violation,
                    proxy_safe,
                    q_policy_minus_champion,
                )

            (policy_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params.policy
            )
            updates, policy_opt_state = self.actor_optim.update(
                grads, state.opt_state.policy, params.policy
            )
            policy_params = optax.apply_updates(params.policy, updates)
            (
                logp,
                q_lcb,
                q_disagreement,
                policy_advantage,
                behavior_advantage,
                behavior_q_lcb,
                supported_sac_loss,
                awac_loss,
                awac_weight,
                proxy_actor_loss,
                teacher_bc_loss,
                teacher_error,
                teacher_gate,
                teacher_advantage,
                initial_kl_loss,
                champion_kl_loss,
                champion_action_loss,
                champion_boundary_action_loss,
                proxy_value,
                champion_proxy_value,
                p_policy_minus_champion,
                p_improved,
                proxy_violation,
                proxy_safe,
                q_policy_minus_champion,
            ) = aux

            def alpha_loss_fn(log_alpha):
                return _weighted_mean(
                    -log_alpha
                    * jax.lax.stop_gradient(logp + self.agent.target_entropy),
                    actor_support,
                )

            alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(
                params.log_alpha
            )
            alpha_updates, log_alpha_opt_state = self.log_alpha_optim.update(
                alpha_grads, state.opt_state.log_alpha, params.log_alpha
            )
            log_alpha = optax.apply_updates(params.log_alpha, alpha_updates)
            proxy_beta = jnp.clip(
                state.proxy_beta
                + config.proxy_dual_learning_rate
                * jax.lax.stop_gradient(proxy_actor_loss - config.proxy_target),
                0.0,
                config.proxy_beta_max,
            )
            trust_coefficient = jnp.clip(
                state.trust_coefficient
                + config.champion_kl_dual_learning_rate
                * jax.lax.stop_gradient(champion_kl_loss - config.champion_kl_target),
                max(config.champion_kl_min, 0.0),
                max(config.champion_kl_max, config.champion_kl_min),
            )
            next_state = state._replace(
                params=params._replace(policy=policy_params, log_alpha=log_alpha),
                opt_state=state.opt_state._replace(
                    policy=policy_opt_state,
                    log_alpha=log_alpha_opt_state,
                ),
                proxy_beta=proxy_beta,
                trust_coefficient=trust_coefficient,
                gradient_step=state.gradient_step + 1,
            )
            return next_state, {
                "policy_loss": policy_loss,
                "supported_sac_loss": supported_sac_loss,
                "awac_loss": awac_loss,
                "awac_beta": jnp.asarray(config.awac_weight, dtype=jnp.float32),
                "awac_weight_mean": _weighted_mean(awac_weight, awac_support),
                "awac_weight_max": jnp.max(
                    jnp.where(awac_support > 0.0, awac_weight, 0.0)
                ),
                "awac_support_fraction": _weighted_mean(
                    behavior_value_valid.astype(jnp.float32),
                    actor_support,
                ),
                "behavior_advantage_mean": _weighted_mean(
                    behavior_advantage, awac_support
                ),
                "behavior_advantage_positive_fraction": _weighted_mean(
                    (behavior_advantage > 0.0).astype(jnp.float32),
                    awac_support,
                ),
                "policy_advantage_mean": _weighted_mean(
                    policy_advantage, actor_support
                ),
                "q_policy_improvement": _weighted_mean(
                    q_lcb - behavior_q_lcb, actor_support
                ),
                "q_policy_minus_champion": _weighted_mean(
                    q_policy_minus_champion, actor_support
                ),
                "proxy_actor_loss": proxy_actor_loss,
                "proxy_beta": proxy_beta,
                "proxy_value_mean": _weighted_mean(proxy_value, proposal_weight),
                "p_champion_mean": _weighted_mean(
                    champion_proxy_value, proposal_weight
                ),
                "p_policy_minus_champion": _weighted_mean(
                    p_policy_minus_champion, proposal_weight
                ),
                "p_improvement_fraction": _weighted_mean(
                    p_improved.astype(jnp.float32), proposal_weight
                ),
                "proxy_violation_mean": _weighted_mean(
                    proxy_violation, proposal_weight
                ),
                "proxy_safe_fraction": _weighted_mean(
                    proxy_safe.astype(jnp.float32),
                    proposal_weight,
                ),
                "proxy_actor_support_fraction": jnp.mean(
                    (proposal_weight > 0.0).astype(jnp.float32)
                ),
                "teacher_bc_loss": teacher_bc_loss,
                "teacher_error": _weighted_mean(
                    teacher_error, teacher_weight * teacher_gate
                ),
                "teacher_gate_mean": _weighted_mean(teacher_gate, teacher_weight),
                "teacher_advantage": _weighted_mean(teacher_advantage, teacher_weight),
                "actor_q": _weighted_mean(q_lcb, actor_support),
                "q_disagreement": jnp.mean(q_disagreement),
                "entropy": -_weighted_mean(logp, actor_support),
                "alpha": jnp.exp(log_alpha),
                "alpha_loss": alpha_loss,
                "actor_support_fraction": jnp.mean(actor_support),
                "initial_policy_kl": initial_kl_loss,
                "champion_policy_kl": champion_kl_loss,
                "champion_action_loss": champion_action_loss,
                "champion_boundary_action_loss": champion_boundary_action_loss,
                "champion_boundary_action_support_fraction": jnp.mean(
                    (boundary_action_weight > 0.0).astype(jnp.float32)
                ),
                "champion_kl_target": jnp.asarray(
                    config.champion_kl_target, dtype=jnp.float32
                ),
                "champion_kl_coefficient": trust_coefficient,
            }

        self._actor_update = jax.jit(actor_update)

        def bc_update(key, state, data):
            del key
            weight = jnp.asarray(data.teacher_valid, dtype=jnp.float32) * jnp.maximum(
                jnp.asarray(data.teacher_weight, dtype=jnp.float32), 0.0
            )

            def loss_fn(policy_params):
                action = self.agent.get_deterministic_action(policy_params, data.obs)
                error = jnp.sum(jnp.square(action - data.teacher_action), axis=-1)
                return _weighted_mean(error, weight), error

            (loss, error), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params.policy
            )
            updates, policy_opt_state = self.actor_optim.update(
                grads, state.opt_state.policy, state.params.policy
            )
            policy_params = optax.apply_updates(state.params.policy, updates)
            next_state = state._replace(
                params=state.params._replace(policy=policy_params),
                opt_state=state.opt_state._replace(policy=policy_opt_state),
                gradient_step=state.gradient_step + 1,
            )
            return next_state, {
                "teacher_bc_loss": loss,
                "teacher_error": _weighted_mean(error, weight),
                "teacher_fraction": jnp.mean(
                    jnp.asarray(data.teacher_valid, dtype=jnp.float32)
                ),
            }

        self._bc_update = jax.jit(bc_update)

    def pretrain_bc(self, key, data: GuardianCalQLExperience) -> Metric:
        self.state, metrics = self._bc_update(key, self.state, data)
        return _python_metrics(metrics)

    def mark_offline_q_reference(self) -> None:
        """Freeze the current Cal-QL critics as the online Q_R reference."""

        self.offline_reference_q1 = _copy_tree(self.state.params.q1)
        self.offline_reference_q2 = _copy_tree(self.state.params.q2)
        self.online_q_anchor_step = 0
        self.offline_q_reference_ready = True

    def q_anchor_coefficient(self) -> float:
        """Return the linearly decayed online Q_R anchor coefficient."""

        start = max(float(self.config.q_anchor_weight_initial), 0.0)
        end = max(float(self.config.q_anchor_weight_final), 0.0)
        decay_steps = max(int(self.config.q_anchor_decay_steps), 1)
        progress = min(max(self.online_q_anchor_step, 0) / decay_steps, 1.0)
        return float(start + progress * (end - start))

    def _online_task_critic_update(
        self, key, data: GuardianCalQLExperience
    ) -> tuple[GuardianCalQLTrainState, Metric]:
        if not self.offline_q_reference_ready:
            self.mark_offline_q_reference()
        coefficient = self.q_anchor_coefficient()
        update = (
            self._offline_calql_update
            if self.config.online_critic_mode == "calql"
            else self._offline_td_update
        )
        state, metrics = update(
            key,
            self.state,
            data,
            self.offline_reference_q1,
            self.offline_reference_q2,
            jnp.asarray(coefficient, dtype=jnp.float32),
            jnp.asarray(max(self.config.q_mc_weight, 0.0), dtype=jnp.float32),
        )
        self.online_q_anchor_step += 1
        metrics = {
            **metrics,
            "online_calql_active": jnp.asarray(
                self.config.online_critic_mode == "calql",
                dtype=jnp.float32,
            ),
        }
        return state, metrics

    def offline_critic_update(
        self,
        key,
        data: GuardianCalQLExperience,
        *,
        mode: OfflineCriticMode = "calql",
    ) -> Metric:
        update = {
            "td": self._offline_td_update,
            "cql": self._offline_cql_update,
            "calql": self._offline_calql_update,
        }.get(mode)
        if update is None:
            raise ValueError(f"unsupported offline critic mode: {mode}")
        self.state, metrics = update(
            key,
            self.state,
            data,
            self.state.params.q1,
            self.state.params.q2,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return _python_metrics(metrics)

    def offline_proxy_update(self, key, data: GuardianCalQLExperience) -> Metric:
        self.state, metrics = self._proxy_update(key, self.state, data)
        return _python_metrics(metrics)

    def online_critic_warmup(self, key, data: GuardianCalQLExperience) -> Metric:
        """Update task Q and P while the deployed Actor remains frozen."""

        q_key, proxy_key = jax.random.split(key)
        self.state, q_metrics = self._online_task_critic_update(q_key, data)
        self.state, proxy_metrics = self._proxy_update(proxy_key, self.state, data)
        return _python_metrics(
            {
                **{f"task_{k}": v for k, v in q_metrics.items()},
                **proxy_metrics,
                "actor_frozen": jnp.asarray(1.0, dtype=jnp.float32),
            }
        )

    def offline_actor_update(self, key, data: GuardianCalQLExperience) -> Metric:
        """Improve the offline Actor with frozen Q/P and conservative anchors."""

        self.state, metrics = self._actor_update(
            key,
            self.state,
            data,
            self.initial_policy_params,
            self.champion_state.params.policy,
            jnp.asarray(self.config.initial_policy_kl_weight, dtype=jnp.float32),
        )
        return _python_metrics(metrics)

    def online_joint_update(
        self, key, data: GuardianCalQLExperience, *, update_actor: bool = True
    ) -> Metric:
        q_key, proxy_key, actor_key = jax.random.split(key, 3)
        self.state, q_metrics = self._online_task_critic_update(q_key, data)
        self.state, proxy_metrics = self._proxy_update(proxy_key, self.state, data)
        actor_metrics = {}
        if update_actor:
            self.state, actor_metrics = self._actor_update(
                actor_key,
                self.state,
                data,
                self.initial_policy_params,
                self.champion_state.params.policy,
                jnp.asarray(0.0, dtype=jnp.float32),
            )
        return _python_metrics(
            {
                **{f"task_{k}": v for k, v in q_metrics.items()},
                **proxy_metrics,
                **actor_metrics,
            }
        )

    def update(self, key, data: GuardianCalQLExperience) -> Metric:
        return self.online_joint_update(key, data)

    def mark_policy_as_initial(self) -> None:
        trust = jnp.asarray(max(self.config.champion_kl_weight, 0.0), dtype=jnp.float32)
        # A new run may intentionally load Q/P from a prior checkpoint while
        # selecting different Actor dual hyperparameters.  Reset both Actor
        # duals to this run's declared configuration before online training.
        proxy_beta = jnp.asarray(
            max(self.config.proxy_actor_weight_init, 0.0), dtype=jnp.float32
        )
        self.state = self.state._replace(
            trust_coefficient=trust,
            proxy_beta=proxy_beta,
        )
        self.initial_policy_params = _copy_tree(self.state.params.policy)
        self.champion_state = _copy_tree(self.state)
        self.mark_last_safe_learner()

    def restart_from_champion(self) -> None:
        """Start a new Actor lineage from the loaded Champion, preserving Q/P."""

        champion_policy = _copy_tree(self.champion_state.params.policy)
        self.restore_policy(champion_policy)
        self.mark_policy_as_initial()

    def reset_proxy_actor_dual(self) -> dict[str, float]:
        """Reset only the P-to-Actor multiplier, preserving every learned model."""

        before = float(self.state.proxy_beta)
        champion_before = float(self.champion_state.proxy_beta)
        after = max(self.config.proxy_actor_weight_init, 0.0)
        proxy_beta = jnp.asarray(after, dtype=jnp.float32)
        self.state = self.state._replace(proxy_beta=proxy_beta)
        self.champion_state = self.champion_state._replace(proxy_beta=proxy_beta)
        return {
            "before": before,
            "champion_before": champion_before,
            "after": float(after),
        }

    def reset_actor_optimizers(self) -> dict[str, bool]:
        """Drop stale Actor Adam moments without changing any policy parameters."""

        self.state = self.state._replace(
            opt_state=self.state.opt_state._replace(
                policy=self.actor_optim.init(self.state.params.policy)
            )
        )
        self.champion_state = self.champion_state._replace(
            opt_state=self.champion_state.opt_state._replace(
                policy=self.actor_optim.init(
                    self.champion_state.params.policy
                )
            )
        )
        self.last_safe_policy_opt_state = self.actor_optim.init(
            self.last_safe_policy
        )
        return {
            "learner": True,
            "champion": True,
            "last_safe": True,
        }

    def restore_policy(self, policy_params) -> None:
        """Install an explicitly selected policy and reset its Adam momentum."""

        policy = jax.device_put(policy_params)
        self.state = self.state._replace(
            params=self.state.params._replace(policy=policy),
            opt_state=self.state.opt_state._replace(
                policy=self.actor_optim.init(policy)
            ),
        )

    def promote_champion(self) -> None:
        self.state = self.state._replace(
            trust_coefficient=jnp.asarray(
                max(self.config.champion_kl_weight, 0.0), dtype=jnp.float32
            )
        )
        self.champion_state = _copy_tree(self.state)
        self.mark_last_safe_learner()

    def refresh_champion_values(self) -> None:
        """Refresh accepted Q/P state without replacing the Champion Actor."""

        champion_policy = _copy_tree(self.champion_state.params.policy)
        refreshed = _copy_tree(self.state)
        self.champion_state = refreshed._replace(
            params=refreshed.params._replace(policy=champion_policy),
            opt_state=refreshed.opt_state._replace(
                policy=self.actor_optim.init(champion_policy)
            ),
            trust_coefficient=jnp.asarray(
                max(self.config.champion_kl_weight, 0.0), dtype=jnp.float32
            ),
        )

    def mark_last_safe_learner(self) -> None:
        """Record the current Actor after it passes the basic safety gate."""

        self.last_safe_policy = _copy_tree(self.state.params.policy)
        self.last_safe_policy_opt_state = _copy_tree(self.state.opt_state.policy)

    def _increase_trust(self) -> float:
        base = max(
            float(self.state.trust_coefficient),
            float(self.champion_state.trust_coefficient),
            max(self.config.champion_kl_weight, 0.0),
        )
        increased = min(
            base * max(self.config.champion_kl_failure_multiplier, 1.0),
            max(self.config.champion_kl_max, base),
        )
        trust = jnp.asarray(increased, dtype=jnp.float32)
        self.state = self.state._replace(trust_coefficient=trust)
        self.champion_state = self.champion_state._replace(
            trust_coefficient=trust
        )
        return float(increased)

    def adapt_trust_from_validation(
        self,
        *,
        q_policy_minus_champion: float,
        p_policy_minus_champion: float,
    ) -> dict[str, float | str | bool]:
        """Adjust Champion KL using held-out relative Q/P evidence."""

        q_delta = float(q_policy_minus_champion)
        p_delta = float(p_policy_minus_champion)
        before = float(self.state.trust_coefficient)
        improving = bool(
            np.isfinite(q_delta)
            and np.isfinite(p_delta)
            and q_delta > 0.0
            and p_delta <= 0.0
        )
        if improving:
            multiplier = float(
                np.clip(
                    self.config.champion_kl_validation_relax_multiplier,
                    0.0,
                    1.0,
                )
            )
            after = max(
                max(self.config.champion_kl_min, 0.0),
                before * multiplier,
            )
            update = "relax"
        else:
            after = min(
                max(before, max(self.config.champion_kl_weight, 0.0))
                * max(self.config.champion_kl_failure_multiplier, 1.0),
                max(self.config.champion_kl_max, before),
            )
            update = "tighten"
        self.state = self.state._replace(
            trust_coefficient=jnp.asarray(after, dtype=jnp.float32)
        )
        return {
            "update": update,
            "improving": improving,
            "before": before,
            "after": float(after),
            "q_policy_minus_champion": q_delta,
            "p_policy_minus_champion": p_delta,
        }

    def retain_safe_learner_after_rejection(self) -> None:
        """Keep a safe-but-underperforming learner and strengthen its KL."""

        self.mark_last_safe_learner()
        self._increase_trust()

    def retain_learner_after_rejection(self, *, restore_critics: bool = False) -> None:
        """Backward-compatible entry point for V2 checkpoints/tests."""

        if restore_critics:
            self.rollback_to_champion(restore_critics=True)
        else:
            self.retain_safe_learner_after_rejection()

    def rollback_to_last_safe(self) -> None:
        """Restore only the latest safety-gated Actor; preserve learned Q/P."""

        self._increase_trust()
        self.state = self.state._replace(
            params=self.state.params._replace(
                policy=_copy_tree(self.last_safe_policy)
            ),
            opt_state=self.state.opt_state._replace(
                policy=_copy_tree(self.last_safe_policy_opt_state)
            ),
        )

    def rollback_to_champion(self, *, restore_critics: bool = False) -> None:
        """Rollback Actor always and optionally the full learned value state."""

        trust = jnp.asarray(self._increase_trust(), dtype=jnp.float32)
        if restore_critics:
            current_step = self.state.gradient_step
            self.state = _copy_tree(self.champion_state)._replace(
                gradient_step=current_step,
                trust_coefficient=trust,
            )
            self.last_safe_policy = _copy_tree(
                self.champion_state.params.policy
            )
            self.last_safe_policy_opt_state = _copy_tree(
                self.champion_state.opt_state.policy
            )
            return
        policy = _copy_tree(self.champion_state.params.policy)
        self.state = self.state._replace(
            params=self.state.params._replace(policy=policy),
            opt_state=self.state.opt_state._replace(
                policy=_copy_tree(self.champion_state.opt_state.policy)
            ),
            trust_coefficient=trust,
        )

    def get_champion_action(self, key, observation) -> np.ndarray:
        return np.asarray(
            self._get_action(key, self.champion_state.params.policy, observation)
        )

    def get_deterministic_champion_action(self, observation) -> np.ndarray:
        return np.asarray(
            self._get_deterministic_action(
                self.champion_state.params.policy, observation
            )
        )

    def get_deterministic_last_safe_action(self, observation) -> np.ndarray:
        return np.asarray(
            self._get_deterministic_action(
                self.last_safe_policy, observation
            )
        )

    def task_q(self, observation, action) -> tuple[np.ndarray, np.ndarray]:
        q1, q2 = self._task_q_values(
            self.state.params.q1,
            self.state.params.q2,
            observation,
            action,
        )
        return np.asarray(q1), np.asarray(q2)

    def proxy_value(self, observation, action) -> np.ndarray:
        return np.asarray(
            self._proxy_value_eval(self.state.params.proxy, observation, action)
        )

    def save(self, path: str) -> None:
        payload = {
            "format": self.FORMAT,
            "state": jax.device_get(self.state),
            "initial_policy_params": jax.device_get(self.initial_policy_params),
            "champion_state": jax.device_get(self.champion_state),
            "last_safe_policy": jax.device_get(self.last_safe_policy),
            "last_safe_policy_opt_state": jax.device_get(
                self.last_safe_policy_opt_state
            ),
            "offline_reference_q1": jax.device_get(self.offline_reference_q1),
            "offline_reference_q2": jax.device_get(self.offline_reference_q2),
            "offline_q_reference_ready": bool(self.offline_q_reference_ready),
            "online_q_anchor_step": int(self.online_q_anchor_step),
            "proxy_objective": self.config.proxy_objective,
            "online_critic_mode": self.config.online_critic_mode,
        }
        with open(path, "wb") as stream:
            pickle.dump(payload, stream)

    def load(self, path: str) -> None:
        with open(path, "rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or payload.get("format") not in {
            self.FORMAT,
            "guardian_calibrated_offline_to_online_safe_sac_v4",
            "guardian_calibrated_offline_to_online_safe_sac_v3",
            "guardian_calibrated_offline_to_online_safe_sac_v2",
        }:
            raise ValueError(
                "checkpoint is not a Guardian-CalQL v2/v3/v4/v5 checkpoint"
            )
        source_format = str(payload["format"])
        self.state = jax.device_put(payload["state"])
        self.initial_policy_params = jax.device_put(payload["initial_policy_params"])
        self.champion_state = jax.device_put(payload["champion_state"])
        self.last_safe_policy = jax.device_put(
            payload.get(
                "last_safe_policy",
                self.champion_state.params.policy,
            )
        )
        self.last_safe_policy_opt_state = jax.device_put(
            payload.get(
                "last_safe_policy_opt_state",
                self.champion_state.opt_state.policy,
            )
        )
        self.offline_reference_q1 = jax.device_put(
            payload.get("offline_reference_q1", self.state.params.q1)
        )
        self.offline_reference_q2 = jax.device_put(
            payload.get("offline_reference_q2", self.state.params.q2)
        )
        self.offline_q_reference_ready = bool(
            payload.get("offline_q_reference_ready", False)
        )
        self.online_q_anchor_step = int(payload.get("online_q_anchor_step", 0))
        self.loaded_checkpoint_format = source_format
        source_proxy_objective = payload.get("proxy_objective")
        proxy_semantics_mismatch = bool(
            source_format.endswith("_v2")
            or (
                source_proxy_objective is None
                and self.config.proxy_objective != "signed_td"
            )
            or (
                source_proxy_objective is not None
                and source_proxy_objective != self.config.proxy_objective
            )
        )
        if proxy_semantics_mismatch:
            # Proxy orientation/objective changed across v2/v3/v4.  Actor and
            # task-Q parameters remain compatible, but P must be recalibrated
            # rather than silently applying a reversed safety penalty.
            proxy = _copy_tree(self._fresh_proxy_params)
            proxy_opt_state = self.proxy_optim.init(proxy)
            self.state = self.state._replace(
                params=self.state.params._replace(
                    proxy=proxy, target_proxy=_copy_tree(proxy)
                ),
                opt_state=self.state.opt_state._replace(proxy=proxy_opt_state),
                proxy_beta=jnp.asarray(
                    max(self.config.proxy_actor_weight_init, 0.0),
                    dtype=jnp.float32,
                ),
            )
            champion_policy = _copy_tree(self.champion_state.params.policy)
            self.champion_state = _copy_tree(self.state)._replace(
                params=self.state.params._replace(policy=champion_policy),
                opt_state=self.state.opt_state._replace(
                    policy=self.actor_optim.init(champion_policy)
                ),
            )
        # Restore the configured starting coefficient instead of silently
        # disabling the Champion KL dual after a zero-valued checkpoint.
        if float(self.state.trust_coefficient) <= 0.0:
            trust = jnp.asarray(
                max(self.config.champion_kl_weight, 0.0), dtype=jnp.float32
            )
            self.state = self.state._replace(trust_coefficient=trust)
            self.champion_state = self.champion_state._replace(trust_coefficient=trust)


__all__ = [
    "GuardianCalQLConfig",
    "GuardianCalQLSafeSAC",
    "GuardianCalQLTrainState",
    "OfflineCriticMode",
    "OnlineCriticMode",
    "ProxyObjective",
]
