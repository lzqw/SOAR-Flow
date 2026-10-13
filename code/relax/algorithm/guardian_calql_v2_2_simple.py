"""Guardian-CalQL V2.2-Simple.

This module is intentionally separate from :mod:`guardian_calql`.  The latter
is the reproducible V2.1 implementation; V2.2 removes online Monte-Carlo and
Cal-QL terms, AWAC, online teacher cloning, Proxy TD/pair losses and the active
``LastSafe`` actor.

The remaining semantics are deliberately small:

* offline task critics: executed-action TD + Cal-QL;
* online task critics: executed-action TD + masked frozen-offline-Q anchor;
* P: sigmoid regression to complete-episode Guardian-dependence targets;
* online actor: supported SAC + Champion-relative P + Champion trust;
* state machine: Learner, Champion policy, and an accepted full snapshot.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import Mapping, NamedTuple, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.base import Algorithm
from relax.network.safe_sac_jax import GuardianCalQLParams, JaxGuardianCalQLNet
from relax.utils.action_semantics import SPEED_COUPLED_ACTION_SEMANTICS
from relax.utils.action_semantics import FIXED_SPEED_ACTION_SEMANTICS
from relax.utils.experience import GuardianCalQLExperience
from relax.utils.typing import Metric


class GuardianCalQLV22OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    proxy: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class GuardianCalQLV22TrainState(NamedTuple):
    params: GuardianCalQLParams
    opt_state: GuardianCalQLV22OptStates
    gradient_step: jax.Array
    trust_coefficient: jax.Array


@dataclass(frozen=True)
class GuardianCalQLV22Config:
    gamma: float = 0.99
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
    q_anchor_weight_initial: float = 1.0
    q_anchor_weight_final: float = 0.1
    q_anchor_decay_steps: int = 100_000

    proxy_huber_delta: float = 0.1
    proxy_corrected_weight: float = 1.0
    proxy_actor_weight: float = 1.0
    proxy_relative_margin: float = 0.0

    actor_sac_weight: float = 1.0
    trust_weight: float = 10.0
    trust_action_weight: float = 1.0
    trust_relax_multiplier: float = 0.95
    trust_tighten_multiplier: float = 1.25
    trust_min: float = 1.0
    trust_max: float = 200.0
    q_improvement_margin: float = 0.0
    p_non_degradation_margin: float = 0.0

    q_disaster_threshold: float = 1e4
    critic_loss_disaster_threshold: float = 1e6
    critic_loss_disaster_window: int = 3

    def __post_init__(self) -> None:
        positive = {
            "gamma": self.gamma,
            "tau": self.tau,
            "critic_learning_rate": self.critic_learning_rate,
            "actor_learning_rate": self.actor_learning_rate,
            "proxy_learning_rate": self.proxy_learning_rate,
            "alpha_learning_rate": self.alpha_learning_rate,
            "gradient_clip_norm": self.gradient_clip_norm,
            "cql_temperature": self.cql_temperature,
            "proxy_huber_delta": self.proxy_huber_delta,
            "q_disaster_threshold": self.q_disaster_threshold,
            "critic_loss_disaster_threshold": self.critic_loss_disaster_threshold,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if not 0.0 < self.trust_relax_multiplier < 1.0:
            raise ValueError("trust_relax_multiplier must be in (0, 1)")
        if self.trust_tighten_multiplier <= 1.0:
            raise ValueError("trust_tighten_multiplier must be > 1")
        if not 0.0 <= self.trust_min <= self.trust_max:
            raise ValueError("require 0 <= trust_min <= trust_max")
        if self.critic_loss_disaster_window < 1:
            raise ValueError("critic_loss_disaster_window must be positive")


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
    return jnp.sum(
        jnp.log(jnp.maximum(reference_std, 1e-8) / jnp.maximum(std, 1e-8))
        + (variance + jnp.square(mean - reference_mean)) / (2.0 * reference_variance)
        - 0.5,
        axis=-1,
    )


def _copy_tree(tree):
    return jax.tree_util.tree_map(lambda value: jnp.array(value, copy=True), tree)


def _python_metrics(metrics: Metric) -> Metric:
    return {name: float(value) for name, value in metrics.items()}


def _normalise_signatures(signatures: Mapping[int | str, str] | None) -> dict[str, str]:
    if signatures is None:
        return {}
    result = {str(int(seed)): str(signature) for seed, signature in signatures.items()}
    for seed, signature in result.items():
        if not signature:
            raise ValueError(f"empty scene signature for seed {seed}")
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


class GuardianCalQLV22SimpleSafeSAC(Algorithm):
    """The strict two-Actor V2.2-Simple algorithm."""

    FORMAT = "guardian_calibrated_offline_to_online_safe_sac_v6"

    def __init__(
        self,
        agent: JaxGuardianCalQLNet,
        params: GuardianCalQLParams,
        *,
        config: GuardianCalQLV22Config = GuardianCalQLV22Config(),
        action_semantics_version: str = SPEED_COUPLED_ACTION_SEMANTICS,
        offline_scene_seed: int = 0,
        online_scene_seeds: Sequence[int] = (),
        scene_signatures: Mapping[int | str, str] | None = None,
        scene_selection_manifest_hash: str = "",
        allow_fixed_speed_ablation: bool = False,
    ):
        if action_semantics_version != SPEED_COUPLED_ACTION_SEMANTICS and not (
            bool(allow_fixed_speed_ablation)
            and action_semantics_version == FIXED_SPEED_ACTION_SEMANTICS
        ):
            raise ValueError(
                "V2.2-Simple requires speed-coupled action semantics; got "
                f"{action_semantics_version!r}"
            )
        self.agent = agent
        self.config = config
        self.action_semantics_version = str(action_semantics_version)
        self.fixed_speed_ablation = bool(allow_fixed_speed_ablation)
        self.offline_scene_seed = int(offline_scene_seed)
        self.online_scene_seeds = tuple(int(seed) for seed in online_scene_seeds)
        self.scene_signatures = _normalise_signatures(scene_signatures)
        self.scene_selection_manifest_hash = str(scene_selection_manifest_hash)
        self.loaded_checkpoint_format: str | None = None

        clipped = lambda learning_rate: optax.chain(
            optax.clip_by_global_norm(config.gradient_clip_norm),
            optax.adam(learning_rate),
        )
        self.critic_optim = clipped(config.critic_learning_rate)
        self.actor_optim = clipped(config.actor_learning_rate)
        self.proxy_optim = clipped(config.proxy_learning_rate)
        self.log_alpha_optim = optax.adam(config.alpha_learning_rate)
        trust = float(np.clip(config.trust_weight, config.trust_min, config.trust_max))
        self.state = GuardianCalQLV22TrainState(
            params=params,
            opt_state=GuardianCalQLV22OptStates(
                q1=self.critic_optim.init(params.q1),
                q2=self.critic_optim.init(params.q2),
                proxy=self.proxy_optim.init(params.proxy),
                policy=self.actor_optim.init(params.policy),
                log_alpha=self.log_alpha_optim.init(params.log_alpha),
            ),
            gradient_step=jnp.asarray(0, dtype=jnp.int32),
            trust_coefficient=jnp.asarray(trust, dtype=jnp.float32),
        )
        self.champion_policy = _copy_tree(params.policy)
        self.accepted_full_snapshot = _copy_tree(self.state)
        self.offline_reference_q1 = _copy_tree(params.q1)
        self.offline_reference_q2 = _copy_tree(params.q2)
        self.offline_q_reference_ready = False
        self.online_q_anchor_step = 0
        self.accepted_online_q_anchor_step = 0
        self._consecutive_large_critic_losses = 0

        self._get_action = jax.jit(self.agent.get_action)
        self._get_deterministic_action = jax.jit(self.agent.get_deterministic_action)
        self._task_q_values = jax.jit(
            lambda q1, q2, observation, action: (
                self.agent.q(q1, observation, action),
                self.agent.q(q2, observation, action),
            )
        )
        self._proxy_value_eval = jax.jit(
            lambda proxy, observation, action: jax.nn.sigmoid(
                self.agent.proxy_value(proxy, observation, action)
            )
        )
        self._build_updates()

    def _build_updates(self) -> None:
        config = self.config

        def make_critic_update(*, calql: bool):
            def critic_update(
                key,
                state,
                data,
                offline_reference_q1,
                offline_reference_q2,
                anchor_weight,
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
                anchor_weight = jnp.asarray(anchor_weight, dtype=jnp.float32)
                temperature = max(config.cql_temperature, 1e-6)

                def loss_fn(q_params: hk.Params, offline_params: hk.Params):
                    replay_q = self.agent.q(q_params, data.obs, data.action)
                    bellman_loss = 0.5 * jnp.mean(jnp.square(replay_q - backup))
                    policy_q = self.agent.q(q_params, data.obs, policy_action)
                    random_q = self.agent.q(q_params, data.obs, random_action)
                    candidates = jnp.stack((policy_q, random_q), axis=-1)
                    if calql:
                        candidates = jnp.where(
                            reference_valid[..., None],
                            jnp.maximum(candidates, safe_reference[..., None]),
                            candidates,
                        )
                    conservative_gap = jnp.mean(
                        temperature
                        * jax.scipy.special.logsumexp(candidates / temperature, axis=-1)
                        - temperature * math.log(candidates.shape[-1])
                        - replay_q
                    )
                    offline_q = jax.lax.stop_gradient(
                        self.agent.q(offline_params, data.obs, data.action)
                    )
                    anchor_loss = 0.5 * _weighted_mean(
                        jnp.square(replay_q - offline_q), anchor_valid
                    )
                    loss = (
                        bellman_loss
                        + (config.calql_weight if calql else 0.0) * conservative_gap
                        + anchor_weight * anchor_loss
                    )
                    return loss, (
                        replay_q,
                        bellman_loss,
                        conservative_gap,
                        anchor_loss,
                    )

                (q1_loss, q1_aux), q1_grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    params.q1, offline_reference_q1
                )
                (q2_loss, q2_aux), q2_grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    params.q2, offline_reference_q2
                )
                q1_updates, q1_opt = self.critic_optim.update(
                    q1_grads, state.opt_state.q1, params.q1
                )
                q2_updates, q2_opt = self.critic_optim.update(
                    q2_grads, state.opt_state.q2, params.q2
                )
                q1 = optax.apply_updates(params.q1, q1_updates)
                q2 = optax.apply_updates(params.q2, q2_updates)
                next_state = state._replace(
                    params=params._replace(
                        q1=q1,
                        q2=q2,
                        target_q1=optax.incremental_update(
                            q1, params.target_q1, config.tau
                        ),
                        target_q2=optax.incremental_update(
                            q2, params.target_q2, config.tau
                        ),
                    ),
                    opt_state=state.opt_state._replace(q1=q1_opt, q2=q2_opt),
                    gradient_step=state.gradient_step + 1,
                )
                q1_value, q1_td, q1_gap, q1_anchor = q1_aux
                q2_value, q2_td, q2_gap, q2_anchor = q2_aux
                q_min = jnp.minimum(q1_value, q2_value)
                return next_state, {
                    "q1_loss": q1_loss,
                    "q2_loss": q2_loss,
                    "td_loss": 0.5 * (q1_td + q2_td),
                    "conservative_gap": 0.5 * (q1_gap + q2_gap),
                    "calql_active": jnp.asarray(calql, dtype=jnp.float32),
                    "q_anchor_loss": 0.5 * (q1_anchor + q2_anchor),
                    "q_anchor_coefficient": anchor_weight,
                    "q_anchor_support_fraction": jnp.mean(anchor_valid),
                    "q_mean": jnp.mean(q_min),
                    "q_disagreement": jnp.mean(jnp.abs(q1_value - q2_value)),
                    "backup_mean": jnp.mean(backup),
                    "q_calibration_mae": _weighted_mean(
                        jnp.abs(q_min - safe_reference),
                        reference_valid.astype(jnp.float32),
                    ),
                }

            return jax.jit(critic_update)

        self._offline_calql_update = make_critic_update(calql=True)
        self._online_td_update = make_critic_update(calql=False)

        def proxy_update(state, data):
            params = state.params
            raw_support = jnp.asarray(
                data.guardian_dependence_target_valid, dtype=jnp.float32
            )
            corrected_support = jnp.asarray(
                data.guardian_dependence_corrected_valid, dtype=jnp.float32
            )
            raw_target = jnp.clip(
                jnp.asarray(data.guardian_dependence_target, dtype=jnp.float32),
                0.0,
                1.0,
            )
            corrected_target = jnp.clip(
                jnp.asarray(
                    data.guardian_dependence_corrected_target, dtype=jnp.float32
                ),
                0.0,
                1.0,
            )

            def loss_fn(proxy_params: hk.Params):
                raw = jax.nn.sigmoid(
                    self.agent.proxy_value(proxy_params, data.obs, data.raw_action)
                )
                corrected = jax.nn.sigmoid(
                    self.agent.proxy_value(
                        proxy_params, data.obs, data.proxy_corrected_action
                    )
                )
                raw_error = raw - raw_target
                corrected_error = corrected - corrected_target
                raw_loss = _weighted_mean(
                    optax.huber_loss(
                        raw, raw_target, delta=max(config.proxy_huber_delta, 1e-6)
                    ),
                    raw_support,
                )
                corrected_loss = _weighted_mean(
                    optax.huber_loss(
                        corrected,
                        corrected_target,
                        delta=max(config.proxy_huber_delta, 1e-6),
                    ),
                    corrected_support,
                )
                loss = raw_loss + config.proxy_corrected_weight * corrected_loss
                return loss, (
                    raw,
                    corrected,
                    raw_error,
                    corrected_error,
                    raw_loss,
                    corrected_loss,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params.proxy)
            updates, proxy_opt = self.proxy_optim.update(
                grads, state.opt_state.proxy, params.proxy
            )
            proxy = optax.apply_updates(params.proxy, updates)
            raw, corrected, raw_error, corrected_error, raw_loss, corrected_loss = aux
            pair_support = corrected_support * raw_support
            next_state = state._replace(
                # ``target_proxy`` is retained only for network-shape compatibility;
                # no bootstrap target or TD path consumes it in V2.2.
                params=params._replace(proxy=proxy, target_proxy=_copy_tree(proxy)),
                opt_state=state.opt_state._replace(proxy=proxy_opt),
                gradient_step=state.gradient_step + 1,
            )
            return next_state, {
                "proxy_loss": loss,
                "proxy_raw_loss": raw_loss,
                "proxy_corrected_loss": corrected_loss,
                "proxy_mae": _weighted_mean(jnp.abs(raw_error), raw_support),
                "proxy_brier": _weighted_mean(jnp.square(raw_error), raw_support),
                "proxy_corrected_mae": _weighted_mean(
                    jnp.abs(corrected_error), corrected_support
                ),
                "proxy_pair_accuracy": _weighted_mean(
                    (raw > corrected).astype(jnp.float32), pair_support
                ),
                "proxy_raw_mean": _weighted_mean(raw, raw_support),
                "proxy_corrected_mean": _weighted_mean(corrected, corrected_support),
                "proxy_support_fraction": jnp.mean(raw_support),
                "proxy_corrected_support_fraction": jnp.mean(corrected_support),
            }

        self._proxy_update = jax.jit(proxy_update)

        def actor_update(key, state, data, champion_policy):
            params = state.params
            actor_support = jnp.asarray(data.actor_controlled, dtype=jnp.float32)
            proxy_support = jnp.maximum(
                jnp.asarray(data.guardian_dependence_target_valid, dtype=jnp.float32),
                jnp.asarray(
                    data.guardian_dependence_corrected_valid, dtype=jnp.float32
                ),
            )
            champion_action = jax.lax.stop_gradient(
                self.agent.get_deterministic_action(champion_policy, data.obs)
            )
            alpha = jnp.exp(params.log_alpha)

            def loss_fn(policy_params: hk.Params):
                action, logp = self.agent.evaluate(key, policy_params, data.obs)
                q1 = self.agent.q(params.q1, data.obs, action)
                q2 = self.agent.q(params.q2, data.obs, action)
                q_lcb = 0.5 * (q1 + q2) - (config.q_lcb_beta * 0.5 * jnp.abs(q1 - q2))
                supported_sac = _weighted_mean(alpha * logp - q_lcb, actor_support)

                p_policy = jax.nn.sigmoid(
                    self.agent.proxy_value(params.proxy, data.obs, action)
                )
                p_champion = jax.lax.stop_gradient(
                    jax.nn.sigmoid(
                        self.agent.proxy_value(params.proxy, data.obs, champion_action)
                    )
                )
                p_delta = p_policy - p_champion
                p_violation = jax.nn.relu(p_delta - config.proxy_relative_margin)
                relative_proxy = _weighted_mean(jnp.square(p_violation), proxy_support)

                mean, std = self.agent.distribution(policy_params, data.obs)
                champion_mean, champion_std = self.agent.distribution(
                    champion_policy, data.obs
                )
                champion_kl = _diagonal_gaussian_kl(
                    mean,
                    std,
                    jax.lax.stop_gradient(champion_mean),
                    jax.lax.stop_gradient(champion_std),
                )
                deterministic_action = self.agent.get_deterministic_action(
                    policy_params, data.obs
                )
                action_drift = jnp.sum(
                    jnp.square(deterministic_action - champion_action), axis=-1
                )
                trust = jnp.mean(
                    champion_kl + config.trust_action_weight * action_drift
                )
                loss = (
                    config.actor_sac_weight * supported_sac
                    + config.proxy_actor_weight * relative_proxy
                    + state.trust_coefficient * trust
                )
                return loss, (
                    logp,
                    q_lcb,
                    q1,
                    q2,
                    supported_sac,
                    relative_proxy,
                    p_policy,
                    p_champion,
                    p_delta,
                    champion_kl,
                    action_drift,
                    trust,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params.policy
            )
            updates, policy_opt = self.actor_optim.update(
                grads, state.opt_state.policy, params.policy
            )
            policy = optax.apply_updates(params.policy, updates)
            (
                logp,
                q_lcb,
                q1,
                q2,
                supported_sac,
                relative_proxy,
                p_policy,
                p_champion,
                p_delta,
                champion_kl,
                action_drift,
                trust,
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
            alpha_updates, alpha_opt = self.log_alpha_optim.update(
                alpha_grads, state.opt_state.log_alpha, params.log_alpha
            )
            log_alpha = optax.apply_updates(params.log_alpha, alpha_updates)
            next_state = state._replace(
                params=params._replace(policy=policy, log_alpha=log_alpha),
                opt_state=state.opt_state._replace(
                    policy=policy_opt, log_alpha=alpha_opt
                ),
                gradient_step=state.gradient_step + 1,
            )
            return next_state, {
                "policy_loss": loss,
                "supported_sac_loss": supported_sac,
                "relative_proxy_loss": relative_proxy,
                "champion_trust_loss": trust,
                "actor_support_fraction": jnp.mean(actor_support),
                "proxy_actor_support_fraction": jnp.mean(proxy_support),
                "p_policy_minus_champion": _weighted_mean(p_delta, proxy_support),
                "p_improvement_fraction": _weighted_mean(
                    (p_delta <= 0.0).astype(jnp.float32), proxy_support
                ),
                "actor_q": _weighted_mean(q_lcb, actor_support),
                "q_disagreement": jnp.mean(jnp.abs(q1 - q2)),
                "entropy": -_weighted_mean(logp, actor_support),
                "alpha": jnp.exp(log_alpha),
                "alpha_loss": alpha_loss,
                "champion_kl": jnp.mean(champion_kl),
                "champion_action_drift": jnp.mean(action_drift),
                "trust_coefficient": state.trust_coefficient,
                "p_policy_mean": _weighted_mean(p_policy, proxy_support),
                "p_champion_mean": _weighted_mean(p_champion, proxy_support),
            }

        self._actor_update = jax.jit(actor_update)

        def bc_update(state, data):
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
            updates, policy_opt = self.actor_optim.update(
                grads, state.opt_state.policy, state.params.policy
            )
            policy = optax.apply_updates(state.params.policy, updates)
            next_state = state._replace(
                params=state.params._replace(policy=policy),
                opt_state=state.opt_state._replace(policy=policy_opt),
                gradient_step=state.gradient_step + 1,
            )
            return next_state, {
                "offline_teacher_bc_loss": loss,
                "offline_teacher_error": _weighted_mean(error, weight),
                "offline_teacher_fraction": jnp.mean(
                    jnp.asarray(data.teacher_valid, dtype=jnp.float32)
                ),
            }

        self._bc_update = jax.jit(bc_update)

    def pretrain_bc(self, key, data: GuardianCalQLExperience) -> Metric:
        del key
        self.state, metrics = self._bc_update(self.state, data)
        return _python_metrics(metrics)

    def offline_critic_update(
        self,
        key,
        data: GuardianCalQLExperience,
        *,
        mode: str = "calql",
    ) -> Metric:
        if mode != "calql":
            raise ValueError("V2.2-Simple offline critic mode must be 'calql'")
        self.state, metrics = self._offline_calql_update(
            key,
            self.state,
            data,
            self.state.params.q1,
            self.state.params.q2,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return _python_metrics(metrics)

    def proxy_update(self, data: GuardianCalQLExperience) -> Metric:
        self.state, metrics = self._proxy_update(self.state, data)
        return _python_metrics(metrics)

    def offline_proxy_update(self, key, data: GuardianCalQLExperience) -> Metric:
        del key
        return self.proxy_update(data)

    def mark_offline_q_reference(self) -> None:
        self.offline_reference_q1 = _copy_tree(self.state.params.q1)
        self.offline_reference_q2 = _copy_tree(self.state.params.q2)
        self.offline_q_reference_ready = True
        self.online_q_anchor_step = 0

    def q_anchor_coefficient(self) -> float:
        start = max(float(self.config.q_anchor_weight_initial), 0.0)
        end = max(float(self.config.q_anchor_weight_final), 0.0)
        progress = min(
            max(self.online_q_anchor_step, 0)
            / max(int(self.config.q_anchor_decay_steps), 1),
            1.0,
        )
        return float(start + progress * (end - start))

    def online_critic_update(self, key, data: GuardianCalQLExperience) -> Metric:
        if not self.offline_q_reference_ready:
            raise RuntimeError(
                "call mark_offline_q_reference() after offline Cal-QL before online TD"
            )
        coefficient = self.q_anchor_coefficient()
        self.state, metrics = self._online_td_update(
            key,
            self.state,
            data,
            self.offline_reference_q1,
            self.offline_reference_q2,
            jnp.asarray(coefficient, dtype=jnp.float32),
        )
        self.online_q_anchor_step += 1
        return _python_metrics(metrics)

    def online_critic_warmup(self, key, data: GuardianCalQLExperience) -> Metric:
        q_metrics = self.online_critic_update(key, data)
        p_metrics = self.proxy_update(data)
        return {
            **{f"task_{name}": value for name, value in q_metrics.items()},
            **p_metrics,
            "actor_frozen": 1.0,
        }

    def online_joint_update(
        self, key, data: GuardianCalQLExperience, *, update_actor: bool = True
    ) -> Metric:
        q_key, actor_key = jax.random.split(key)
        q_metrics = self.online_critic_update(q_key, data)
        p_metrics = self.proxy_update(data)
        actor_metrics: Metric = {}
        if update_actor:
            self.state, raw_actor_metrics = self._actor_update(
                actor_key, self.state, data, self.champion_policy
            )
            actor_metrics = _python_metrics(raw_actor_metrics)
        return {
            **{f"task_{name}": value for name, value in q_metrics.items()},
            **p_metrics,
            **actor_metrics,
        }

    def update(self, key, data: GuardianCalQLExperience) -> Metric:
        return self.online_joint_update(key, data)

    def mark_policy_as_initial(self) -> None:
        trust = float(
            np.clip(
                self.config.trust_weight, self.config.trust_min, self.config.trust_max
            )
        )
        self.state = self.state._replace(
            trust_coefficient=jnp.asarray(trust, dtype=jnp.float32)
        )
        self.champion_policy = _copy_tree(self.state.params.policy)
        self.accepted_full_snapshot = _copy_tree(self.state)
        self.accepted_online_q_anchor_step = int(self.online_q_anchor_step)

    def promote_champion(self) -> None:
        """Accept Learner as the sole deployable Champion and snapshot all state."""

        self.champion_policy = _copy_tree(self.state.params.policy)
        self.accepted_full_snapshot = _copy_tree(self.state)
        self.accepted_online_q_anchor_step = int(self.online_q_anchor_step)

    def _set_trust(self, value: float) -> float:
        value = float(np.clip(value, self.config.trust_min, self.config.trust_max))
        self.state = self.state._replace(
            trust_coefficient=jnp.asarray(value, dtype=jnp.float32)
        )
        return value

    def tighten_trust(self) -> float:
        return self._set_trust(
            float(self.state.trust_coefficient) * self.config.trust_tighten_multiplier
        )

    def adapt_trust_from_validation(
        self,
        *,
        q_policy_minus_champion: float,
        p_policy_minus_champion: float,
    ) -> dict[str, float | str | bool]:
        q_delta = float(q_policy_minus_champion)
        p_delta = float(p_policy_minus_champion)
        before = float(self.state.trust_coefficient)
        improving = bool(
            np.isfinite(q_delta)
            and np.isfinite(p_delta)
            and q_delta > self.config.q_improvement_margin
            and p_delta <= self.config.p_non_degradation_margin
        )
        multiplier = (
            self.config.trust_relax_multiplier
            if improving
            else self.config.trust_tighten_multiplier
        )
        after = self._set_trust(before * multiplier)
        return {
            "update": "relax" if improving else "tighten",
            "improving": improving,
            "before": before,
            "after": after,
            "q_policy_minus_champion": q_delta,
            "p_policy_minus_champion": p_delta,
        }

    def reject_performance(self) -> dict[str, float | str]:
        """Keep Learner/Q/P and tighten trust; Champion is unchanged."""

        before = float(self.state.trust_coefficient)
        return {
            "rollback": "none",
            "trust_before": before,
            "trust_after": self.tighten_trust(),
        }

    def reject_safety(self) -> dict[str, float | str]:
        """Restore only Champion Actor; keep current Q/P and reset Actor Adam."""

        before = float(self.state.trust_coefficient)
        policy = _copy_tree(self.champion_policy)
        self.state = self.state._replace(
            params=self.state.params._replace(policy=policy),
            opt_state=self.state.opt_state._replace(
                policy=self.actor_optim.init(policy)
            ),
        )
        after = self.tighten_trust()
        return {
            "rollback": "actor_only",
            "trust_before": before,
            "trust_after": after,
        }

    def reject_numerical_disaster(self) -> dict[str, str]:
        """Restore the accepted full state exactly."""

        self.state = _copy_tree(self.accepted_full_snapshot)
        self.champion_policy = _copy_tree(self.accepted_full_snapshot.params.policy)
        self.online_q_anchor_step = int(self.accepted_online_q_anchor_step)
        self._consecutive_large_critic_losses = 0
        return {"rollback": "accepted_full_snapshot"}

    # Explicit V2.1-compatible entry point with V2.2 rollback semantics.
    def rollback_to_champion(self, *, restore_critics: bool = False) -> None:
        if restore_critics:
            self.reject_numerical_disaster()
        else:
            self.reject_safety()

    def numerical_disaster_reasons(
        self,
        *,
        metrics: Mapping[str, float] | None = None,
        observation=None,
        action=None,
    ) -> list[str]:
        reasons: list[str] = []
        leaves = jax.tree_util.tree_leaves(self.state)
        if any(not np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
            reasons.append("non_finite_state")
        if metrics is not None:
            numeric = [float(value) for value in metrics.values()]
            if any(not np.isfinite(value) for value in numeric):
                reasons.append("non_finite_metric")
            loss = max(
                [
                    abs(float(value))
                    for name, value in metrics.items()
                    if name.endswith("q1_loss") or name.endswith("q2_loss")
                ]
                or [0.0]
            )
            if loss > self.config.critic_loss_disaster_threshold:
                self._consecutive_large_critic_losses += 1
            else:
                self._consecutive_large_critic_losses = 0
            if (
                self._consecutive_large_critic_losses
                >= self.config.critic_loss_disaster_window
            ):
                reasons.append("critic_loss_window")
            disagreement = max(
                [
                    abs(float(value))
                    for name, value in metrics.items()
                    if name.endswith("q_disagreement")
                ]
                or [0.0]
            )
            if disagreement > self.config.q_disaster_threshold:
                reasons.append("q_disagreement")
        if observation is not None and action is not None:
            p = np.asarray(self.proxy_value(observation, action))
            if not np.all(np.isfinite(p)):
                reasons.append("non_finite_proxy")
            elif np.any((p < 0.0) | (p > 1.0)):
                reasons.append("proxy_out_of_range")
        return sorted(set(reasons))

    def get_champion_action(self, key, observation) -> np.ndarray:
        return np.asarray(self._get_action(key, self.champion_policy, observation))

    def get_deterministic_champion_action(self, observation) -> np.ndarray:
        return np.asarray(
            self._get_deterministic_action(self.champion_policy, observation)
        )

    def task_q(self, observation, action) -> tuple[np.ndarray, np.ndarray]:
        q1, q2 = self._task_q_values(
            self.state.params.q1, self.state.params.q2, observation, action
        )
        return np.asarray(q1), np.asarray(q2)

    def proxy_value(self, observation, action) -> np.ndarray:
        return np.asarray(
            self._proxy_value_eval(self.state.params.proxy, observation, action)
        )

    def _checkpoint_metadata(self) -> dict[str, object]:
        return {
            "action_semantics_version": self.action_semantics_version,
            "offline_scene_seed": self.offline_scene_seed,
            "online_scene_seeds": list(self.online_scene_seeds),
            "scene_signatures": dict(self.scene_signatures),
            "scene_selection_manifest_hash": self.scene_selection_manifest_hash,
            "fixed_speed_ablation": self.fixed_speed_ablation,
        }

    def save(self, path: str) -> None:
        payload = {
            "format": self.FORMAT,
            "learner_state": jax.device_get(self.state),
            "champion_policy": jax.device_get(self.champion_policy),
            "accepted_full_snapshot": jax.device_get(self.accepted_full_snapshot),
            "accepted_online_q_anchor_step": int(self.accepted_online_q_anchor_step),
            "offline_q_reference": {
                "q1": jax.device_get(self.offline_reference_q1),
                "q2": jax.device_get(self.offline_reference_q2),
                "ready": bool(self.offline_q_reference_ready),
                "online_anchor_step": int(self.online_q_anchor_step),
            },
            **self._checkpoint_metadata(),
        }
        with open(path, "wb") as stream:
            pickle.dump(payload, stream)

    def load(self, path: str) -> None:
        with open(path, "rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or payload.get("format") != self.FORMAT:
            raise ValueError(
                "checkpoint is not a Guardian-CalQL V2.2-Simple v6 checkpoint"
            )
        expected = self._checkpoint_metadata()
        observed = {
            name: payload.get(name)
            for name in (
                "action_semantics_version",
                "offline_scene_seed",
                "online_scene_seeds",
                "scene_signatures",
                "scene_selection_manifest_hash",
                "fixed_speed_ablation",
            )
        }
        observed["online_scene_seeds"] = list(observed["online_scene_seeds"] or [])
        observed["scene_signatures"] = _normalise_signatures(
            observed["scene_signatures"] or {}
        )
        if observed != expected:
            mismatches = [
                name for name in expected if observed.get(name) != expected.get(name)
            ]
            raise ValueError(
                "V2.2 checkpoint metadata mismatch: " + ", ".join(mismatches)
            )
        forbidden = {"last_safe_policy", "last_safe_policy_opt_state"}.intersection(
            payload
        )
        if forbidden:
            raise ValueError("V2.2 checkpoint contains forbidden LastSafe state")
        self.state = jax.device_put(payload["learner_state"])
        self.champion_policy = jax.device_put(payload["champion_policy"])
        self.accepted_full_snapshot = jax.device_put(payload["accepted_full_snapshot"])
        self.accepted_online_q_anchor_step = int(
            payload["accepted_online_q_anchor_step"]
        )
        reference = payload["offline_q_reference"]
        self.offline_reference_q1 = jax.device_put(reference["q1"])
        self.offline_reference_q2 = jax.device_put(reference["q2"])
        self.offline_q_reference_ready = bool(reference["ready"])
        self.online_q_anchor_step = int(reference["online_anchor_step"])
        self.loaded_checkpoint_format = self.FORMAT


__all__ = [
    "GuardianCalQLV22Config",
    "GuardianCalQLV22OptStates",
    "GuardianCalQLV22SimpleSafeSAC",
    "GuardianCalQLV22TrainState",
]
