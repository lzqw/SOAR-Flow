"""Guardian-DSAC V2.3: explicit-speed safe offline-to-online RL.

V2.3 is intentionally a new implementation rather than a compatibility mode
inside V2.2.  Its Task Critics model Gaussian return distributions, while P
keeps the complete-episode bounded Guardian-dependence regression from
V2.2-Simple.  The online state machine has one mutable Learner and two passive
artifacts: a Hybrid-safe snapshot and an optional deployable checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import pickle
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.base import Algorithm
from relax.network.safe_sac_jax import (
    GuardianDSACParams,
    JaxGuardianDSACNet,
)
from relax.utils.action_semantics import (
    TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
)
from relax.utils.distributional_q import (
    OfflineSupportThresholds,
    calibrate_offline_support,
    distributional_soft_bellman_target,
    ema_standard_deviation,
    gaussian_nll_loss,
    gaussian_reference_loss,
    gaussian_wasserstein_disagreement,
    hdsac_gaussian_loss,
    normalized_bellman_surprise,
)
from relax.utils.experience import GuardianDSACExperience
from relax.utils.typing import Metric


class GuardianDSACOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    proxy: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class GuardianDSACTrainState(NamedTuple):
    params: GuardianDSACParams
    opt_state: GuardianDSACOptStates
    gradient_step: jax.Array
    ema_q_std_1: jax.Array
    ema_q_std_2: jax.Array
    proximal_step: jax.Array


@dataclass(frozen=True)
class GuardianDSACV23Config:
    gamma: float = 0.99
    tau: float = 0.005
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 1e-4
    proxy_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    reward_scale: float = 1.0
    gradient_clip_norm: float = 10.0

    distributional_loss: str = "hdsac"
    sigma_min: float = 1e-3
    sigma_ema_tau: float = 0.005
    target_noise_clip: float = 3.0
    target_std_clip_multiplier: float = 3.0
    hdsac_stabilizer: float = 0.1

    calql_weight: float = 1.0
    cql_temperature: float = 1.0

    reference_weight_initial: float = 1.0
    reference_weight_final: float = 0.1
    reference_decay_steps: int = 100_000
    reference_std_weight: float = 0.1
    disagreement_std_weight: float = 1.0
    support_quantile: float = 0.95

    proxy_huber_delta: float = 0.1
    proxy_corrected_weight: float = 1.0
    proxy_raw_pair_weight: float = 1.0
    proxy_pair_gap_weight: float = 0.0
    proxy_actor_weight: float = 0.5
    proxy_actor_objective: str = "probability"

    actor_sac_weight: float = 1.0
    proximal_weight_initial: float = 1.0
    proximal_weight_min: float = 0.05
    proximal_decay_steps: int = 20_000
    proximal_speed_weight: float = 0.5
    proximal_lateral_weight: float = 1.0

    q_disaster_threshold: float = 1e4
    critic_loss_disaster_threshold: float = 1e6
    critic_loss_disaster_window: int = 3

    def __post_init__(self) -> None:
        if self.distributional_loss not in ("hdsac", "gaussian_nll"):
            raise ValueError(
                "distributional_loss must be 'hdsac' or 'gaussian_nll'"
            )
        if self.proxy_actor_objective not in (
            "probability",
            "softplus_logit",
        ):
            raise ValueError(
                "proxy_actor_objective must be 'probability' or "
                "'softplus_logit'"
            )
        positive = {
            "gamma": self.gamma,
            "tau": self.tau,
            "critic_learning_rate": self.critic_learning_rate,
            "actor_learning_rate": self.actor_learning_rate,
            "proxy_learning_rate": self.proxy_learning_rate,
            "alpha_learning_rate": self.alpha_learning_rate,
            "gradient_clip_norm": self.gradient_clip_norm,
            "sigma_min": self.sigma_min,
            "sigma_ema_tau": self.sigma_ema_tau,
            "target_noise_clip": self.target_noise_clip,
            "target_std_clip_multiplier": self.target_std_clip_multiplier,
            "hdsac_stabilizer": self.hdsac_stabilizer,
            "cql_temperature": self.cql_temperature,
            "reference_std_weight": self.reference_std_weight,
            "disagreement_std_weight": self.disagreement_std_weight,
            "proxy_huber_delta": self.proxy_huber_delta,
            "proxy_raw_pair_weight": self.proxy_raw_pair_weight,
            "proximal_decay_steps": self.proximal_decay_steps,
            "q_disaster_threshold": self.q_disaster_threshold,
            "critic_loss_disaster_threshold": self.critic_loss_disaster_threshold,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("gamma and tau must be in (0, 1]")
        if not 0.0 < self.sigma_ema_tau <= 1.0:
            raise ValueError("sigma_ema_tau must be in (0, 1]")
        if not 0.0 < self.support_quantile < 1.0:
            raise ValueError("support_quantile must be in (0, 1)")
        if self.reference_decay_steps < 1:
            raise ValueError("reference_decay_steps must be positive")
        if self.critic_loss_disaster_window < 1:
            raise ValueError("critic_loss_disaster_window must be positive")
        nonnegative = (
            self.calql_weight,
            self.reference_weight_initial,
            self.reference_weight_final,
            self.proxy_corrected_weight,
            self.proxy_pair_gap_weight,
            self.proxy_actor_weight,
            self.actor_sac_weight,
            self.proximal_weight_initial,
            self.proximal_weight_min,
            self.proximal_speed_weight,
            self.proximal_lateral_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("loss coefficients must be finite and non-negative")
        if self.proxy_raw_pair_weight < 1.0:
            raise ValueError("proxy_raw_pair_weight must be at least one")
        if self.proximal_weight_min > self.proximal_weight_initial:
            raise ValueError(
                "proximal_weight_min cannot exceed proximal_weight_initial"
            )
        if self.proximal_speed_weight + self.proximal_lateral_weight <= 0.0:
            raise ValueError("at least one proximal axis weight must be positive")


def _weighted_mean(value: jax.Array, weight: jax.Array) -> jax.Array:
    value = jnp.asarray(value, dtype=jnp.float32)
    weight = jnp.asarray(weight, dtype=jnp.float32)
    return jnp.sum(value * weight) / jnp.maximum(jnp.sum(weight), 1.0)


def proxy_actor_objective(
    proxy_logit: jax.Array,
    *,
    objective: str,
) -> jax.Array:
    """Map P logits to the configured Actor risk objective.

    The probability objective preserves the original V2.3.1 Actor loss.
    ``softplus_logit`` is an explicit corrective mode whose gradient does not
    vanish when the calibrated dependence probability is close to one.
    """

    proxy_logit = jnp.asarray(proxy_logit, dtype=jnp.float32)
    if objective == "probability":
        return jax.nn.sigmoid(proxy_logit)
    if objective == "softplus_logit":
        return jax.nn.softplus(proxy_logit)
    raise ValueError(f"unsupported proxy Actor objective: {objective}")


def proxy_raw_support_weights(
    target_valid: jax.Array,
    corrected_valid: jax.Array,
    *,
    pair_weight: float,
) -> jax.Array:
    """Emphasize intervention raw actions without changing their P targets."""

    if not np.isfinite(pair_weight) or pair_weight < 1.0:
        raise ValueError("proxy raw pair weight must be finite and at least one")
    target_valid = jnp.asarray(target_valid, dtype=jnp.float32)
    corrected_valid = jnp.asarray(corrected_valid, dtype=jnp.float32)
    return target_valid * (1.0 + (pair_weight - 1.0) * corrected_valid)


def weighted_action_proximal(
    delta: jax.Array,
    *,
    speed_weight: float,
    lateral_weight: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return weighted two-dimensional drift and its auditable components."""

    delta = jnp.asarray(delta, dtype=jnp.float32)
    if delta.shape[-1] != 2:
        raise ValueError("Guardian-DSAC proximal requires a two-dimensional action")
    speed_drift = jnp.mean(jnp.square(delta[..., 0]))
    lateral_drift = jnp.mean(jnp.square(delta[..., 1]))
    proximal = speed_weight * speed_drift + lateral_weight * lateral_drift
    return proximal, speed_drift, lateral_drift


def _copy_tree(tree):
    return jax.tree_util.tree_map(lambda value: jnp.array(value, copy=True), tree)


def _python_metrics(metrics: Metric) -> Metric:
    return {name: float(value) for name, value in metrics.items()}


def _normalise_signatures(signatures: Mapping[int | str, str] | None) -> dict[str, str]:
    if signatures is None:
        return {}
    result = {str(int(seed)): str(value) for seed, value in signatures.items()}
    if any(not value for value in result.values()):
        raise ValueError("scene signatures must be non-empty")
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def calql_mean_conservative_gap(
    replay_mean: jax.Array,
    candidate_means: jax.Array,
    *,
    temperature: float,
) -> jax.Array:
    """Cal-QL/CQL conservative term whose inputs are means only."""

    temperature = max(float(temperature), 1e-6)
    candidate_means = jnp.asarray(candidate_means, dtype=jnp.float32)
    return jnp.mean(
        temperature
        * jax.scipy.special.logsumexp(candidate_means / temperature, axis=-1)
        - temperature * math.log(candidate_means.shape[-1])
        - jnp.asarray(replay_mean, dtype=jnp.float32)
    )


class GuardianDSACV23(Algorithm):
    FORMAT = "guardian_dsac_safe_offline_to_online_v7"

    def __init__(
        self,
        agent: JaxGuardianDSACNet,
        params: GuardianDSACParams,
        *,
        config: GuardianDSACV23Config = GuardianDSACV23Config(),
        action_semantics_version: str = TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
        scene_seeds: Sequence[int] = (0, 1, 2),
        scene_signatures: Mapping[int | str, str] | None = None,
        dataset_sha256: str = "",
    ):
        if action_semantics_version != TARGET_SPEED_LATERAL_ACTION_SEMANTICS:
            raise ValueError(
                "Guardian-DSAC V2.3 requires target_speed_lateral_v1"
            )
        if int(agent.act_dim) != 2:
            raise ValueError("Guardian-DSAC V2.3 requires a two-dimensional action")
        if not dataset_sha256:
            raise ValueError("dataset_sha256 binding is required")
        seeds = tuple(int(seed) for seed in scene_seeds)
        if len(seeds) != len(set(seeds)) or not seeds:
            raise ValueError("scene_seeds must be non-empty and unique")
        signatures = _normalise_signatures(scene_signatures)
        missing = [seed for seed in seeds if str(seed) not in signatures]
        if missing:
            raise ValueError(f"scene signatures missing for seeds {missing}")

        self.agent = agent
        self.config = config
        self.action_semantics_version = str(action_semantics_version)
        self.scene_seeds = seeds
        self.scene_signatures = signatures
        self.dataset_sha256 = str(dataset_sha256)
        self.loaded_checkpoint_format: str | None = None

        clipped = lambda rate: optax.chain(
            optax.clip_by_global_norm(config.gradient_clip_norm),
            optax.adam(rate),
        )
        self.critic_optim = clipped(config.critic_learning_rate)
        self.actor_optim = clipped(config.actor_learning_rate)
        self.proxy_optim = clipped(config.proxy_learning_rate)
        self.log_alpha_optim = optax.adam(config.alpha_learning_rate)
        initial_std = jnp.asarray(max(1.0, config.sigma_min), dtype=jnp.float32)
        self.state = GuardianDSACTrainState(
            params=params,
            opt_state=GuardianDSACOptStates(
                q1=self.critic_optim.init(params.q1),
                q2=self.critic_optim.init(params.q2),
                proxy=self.proxy_optim.init(params.proxy),
                policy=self.actor_optim.init(params.policy),
                log_alpha=self.log_alpha_optim.init(params.log_alpha),
            ),
            gradient_step=jnp.asarray(0, dtype=jnp.int32),
            ema_q_std_1=initial_std,
            ema_q_std_2=initial_std,
            proximal_step=jnp.asarray(0, dtype=jnp.int32),
        )
        self.hybrid_safe_snapshot = _copy_tree(self.state)
        self.deployable_checkpoint: dict[str, object] | None = None
        self.offline_reference_q1 = _copy_tree(params.q1)
        self.offline_reference_q2 = _copy_tree(params.q2)
        self.offline_reference_policy = _copy_tree(params.policy)
        self.offline_reference_alpha = float(np.exp(np.asarray(params.log_alpha)))
        self.offline_support_thresholds: OfflineSupportThresholds | None = None
        self.offline_reference_ready = False
        self.online_reference_step = 0
        self._safe_online_reference_step = 0
        self._consecutive_large_critic_losses = 0

        self._get_action = jax.jit(self.agent.get_action)
        self._get_deterministic_action = jax.jit(
            self.agent.get_deterministic_action
        )
        self._q_distribution = jax.jit(
            lambda q1, q2, obs, action: (
                *self.agent.q_distribution(q1, obs, action),
                *self.agent.q_distribution(q2, obs, action),
            )
        )
        self._proxy_value_eval = jax.jit(
            lambda proxy, obs, action: jax.nn.sigmoid(
                self.agent.proxy_value(proxy, obs, action)
            )
        )
        self._build_updates()

    def _build_updates(self) -> None:
        config = self.config

        def make_critic_update(*, calql: bool, online_reference: bool):
            def critic_update(
                key,
                state,
                data,
                offline_q1,
                offline_q2,
                offline_policy,
                offline_alpha,
                threshold_values,
                reference_coefficient,
                external_next_action,
                external_next_logp,
                external_policy_action,
                use_external_policy,
            ):
                params = state.params
                (
                    next_action_key,
                    target_noise_key,
                    policy_action_key,
                    random_action_key,
                ) = jax.random.split(key, 4)
                sampled_next_action, sampled_next_logp = self.agent.evaluate(
                    next_action_key, params.policy, data.next_obs
                )
                next_action = jnp.where(
                    use_external_policy,
                    jnp.asarray(external_next_action, dtype=jnp.float32),
                    sampled_next_action,
                )
                next_logp = jnp.where(
                    use_external_policy,
                    jnp.asarray(external_next_logp, dtype=jnp.float32),
                    sampled_next_logp,
                )
                target_mean_1, target_std_1 = self.agent.q_distribution(
                    params.target_q1, data.next_obs, next_action
                )
                target_mean_2, target_std_2 = self.agent.q_distribution(
                    params.target_q2, data.next_obs, next_action
                )
                target = distributional_soft_bellman_target(
                    key=target_noise_key,
                    reward=config.reward_scale
                    * jnp.asarray(data.reward, dtype=jnp.float32),
                    done=data.done,
                    gamma=config.gamma,
                    alpha=jnp.exp(params.log_alpha),
                    next_log_prob=next_logp,
                    target_mean_1=target_mean_1,
                    target_std_1=target_std_1,
                    target_mean_2=target_mean_2,
                    target_std_2=target_std_2,
                    noise_clip=config.target_noise_clip,
                )
                sampled_policy_action, _ = self.agent.evaluate(
                    policy_action_key, params.policy, data.obs
                )
                policy_action = jnp.where(
                    use_external_policy,
                    jnp.asarray(external_policy_action, dtype=jnp.float32),
                    sampled_policy_action,
                )
                random_action = jax.random.uniform(
                    random_action_key,
                    shape=jnp.asarray(data.action).shape,
                    minval=-self.agent.action_scale,
                    maxval=self.agent.action_scale,
                )
                reference = jnp.asarray(
                    data.behavior_return_to_go, dtype=jnp.float32
                )
                reference_valid = jnp.logical_and(
                    jnp.asarray(data.behavior_return_valid, dtype=bool),
                    jnp.isfinite(reference),
                )
                safe_reference = jnp.where(reference_valid, reference, 0.0)

                frozen_mean_1, frozen_std_1 = self.agent.q_distribution(
                    offline_q1, data.obs, data.action
                )
                frozen_mean_2, frozen_std_2 = self.agent.q_distribution(
                    offline_q2, data.obs, data.action
                )
                if online_reference:
                    frozen_next_action = self.agent.get_deterministic_action(
                        offline_policy, data.next_obs
                    )
                    frozen_next_logp = self.agent.log_prob(
                        offline_policy, data.next_obs, frozen_next_action
                    )
                    frozen_next_mean_1, _ = self.agent.q_distribution(
                        offline_q1, data.next_obs, frozen_next_action
                    )
                    frozen_next_mean_2, _ = self.agent.q_distribution(
                        offline_q2, data.next_obs, frozen_next_action
                    )
                    frozen_backup = jax.lax.stop_gradient(
                        config.reward_scale
                        * jnp.asarray(data.reward, dtype=jnp.float32)
                        + (1.0 - jnp.asarray(data.done, dtype=jnp.float32))
                        * config.gamma
                        * (
                            jnp.minimum(
                                frozen_next_mean_1, frozen_next_mean_2
                            )
                            - jnp.asarray(offline_alpha, dtype=jnp.float32)
                            * frozen_next_logp
                        )
                    )
                    disagreement = gaussian_wasserstein_disagreement(
                        frozen_mean_1,
                        frozen_std_1,
                        frozen_mean_2,
                        frozen_std_2,
                        std_weight=config.disagreement_std_weight,
                    )
                    surprise = normalized_bellman_surprise(
                        frozen_mean_1,
                        frozen_mean_2,
                        frozen_std_1,
                        frozen_std_2,
                        frozen_backup,
                    )
                    tau_d, tau_e, temp_d, temp_e = threshold_values
                    reference_weight = jax.nn.sigmoid(
                        (tau_d - disagreement) / temp_d
                    ) * jax.nn.sigmoid((tau_e - surprise) / temp_e)
                else:
                    disagreement = jnp.zeros_like(frozen_mean_1)
                    surprise = jnp.zeros_like(frozen_mean_1)
                    reference_weight = jnp.zeros_like(frozen_mean_1)

                def loss_fn(
                    q_params: hk.Params,
                    frozen_params: hk.Params,
                    running_std: jax.Array,
                ):
                    mean, std = self.agent.q_distribution(
                        q_params, data.obs, data.action
                    )
                    if config.distributional_loss == "hdsac":
                        td_loss = hdsac_gaussian_loss(
                            mean,
                            std,
                            target.mean,
                            target.sample,
                            running_std,
                            stabilizer=config.hdsac_stabilizer,
                            clip_multiplier=config.target_std_clip_multiplier,
                        )
                    else:
                        td_loss = gaussian_nll_loss(
                            mean,
                            std,
                            target.sample,
                            sigma_min=config.sigma_min,
                        )
                    policy_mean, _ = self.agent.q_distribution(
                        q_params, data.obs, policy_action
                    )
                    random_mean, _ = self.agent.q_distribution(
                        q_params, data.obs, random_action
                    )
                    candidates = jnp.stack((policy_mean, random_mean), axis=-1)
                    if calql:
                        candidates = jnp.where(
                            reference_valid[..., None],
                            jnp.maximum(candidates, safe_reference[..., None]),
                            candidates,
                        )
                    conservative_gap = calql_mean_conservative_gap(
                        mean,
                        candidates,
                        temperature=config.cql_temperature,
                    )
                    frozen_mean, frozen_std = self.agent.q_distribution(
                        frozen_params, data.obs, data.action
                    )
                    ref_loss = gaussian_reference_loss(
                        mean,
                        std,
                        frozen_mean,
                        frozen_std,
                        reference_weight,
                        std_weight=config.reference_std_weight,
                    )
                    loss = (
                        td_loss
                        + (config.calql_weight if calql else 0.0)
                        * conservative_gap
                        + jnp.asarray(reference_coefficient, dtype=jnp.float32)
                        * ref_loss
                    )
                    return loss, (
                        mean,
                        std,
                        td_loss,
                        conservative_gap,
                        ref_loss,
                    )

                (q1_loss, q1_aux), q1_grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(params.q1, offline_q1, state.ema_q_std_1)
                (q2_loss, q2_aux), q2_grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(params.q2, offline_q2, state.ema_q_std_2)
                q1_updates, q1_opt = self.critic_optim.update(
                    q1_grads, state.opt_state.q1, params.q1
                )
                q2_updates, q2_opt = self.critic_optim.update(
                    q2_grads, state.opt_state.q2, params.q2
                )
                q1 = optax.apply_updates(params.q1, q1_updates)
                q2 = optax.apply_updates(params.q2, q2_updates)
                q1_mean, q1_std, q1_td, q1_gap, q1_ref = q1_aux
                q2_mean, q2_std, q2_td, q2_gap, q2_ref = q2_aux
                ema_1 = ema_standard_deviation(
                    state.ema_q_std_1,
                    q1_std,
                    tau=config.sigma_ema_tau,
                    sigma_min=config.sigma_min,
                )
                ema_2 = ema_standard_deviation(
                    state.ema_q_std_2,
                    q2_std,
                    tau=config.sigma_ema_tau,
                    sigma_min=config.sigma_min,
                )
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
                    ema_q_std_1=ema_1,
                    ema_q_std_2=ema_2,
                )
                q_min = jnp.minimum(q1_mean, q2_mean)
                return next_state, {
                    "q1_loss": q1_loss,
                    "q2_loss": q2_loss,
                    "distributional_td_loss": 0.5 * (q1_td + q2_td),
                    "conservative_gap": 0.5 * (q1_gap + q2_gap),
                    "calql_active": jnp.asarray(calql, dtype=jnp.float32),
                    "reference_loss": 0.5 * (q1_ref + q2_ref),
                    "reference_coefficient": jnp.asarray(
                        reference_coefficient, dtype=jnp.float32
                    ),
                    "reference_weight_mean": jnp.mean(reference_weight),
                    "reference_weight_min": jnp.min(reference_weight),
                    "reference_weight_max": jnp.max(reference_weight),
                    "offline_disagreement": jnp.mean(disagreement),
                    "offline_bellman_surprise": jnp.mean(surprise),
                    "q_mean": jnp.mean(q_min),
                    "q_std_1": jnp.mean(q1_std),
                    "q_std_2": jnp.mean(q2_std),
                    "ema_q_std_1": ema_1,
                    "ema_q_std_2": ema_2,
                    "q_disagreement": jnp.mean(jnp.abs(q1_mean - q2_mean)),
                    "backup_mean": jnp.mean(target.mean),
                    "sample_backup_mean": jnp.mean(target.sample),
                    "q_calibration_mae": _weighted_mean(
                        jnp.abs(q_min - safe_reference),
                        reference_valid.astype(jnp.float32),
                    ),
                }

            return jax.jit(critic_update)

        self._offline_calql_update = make_critic_update(
            calql=True, online_reference=False
        )
        self._online_distributional_update = make_critic_update(
            calql=False, online_reference=True
        )

        def proxy_update(state, data):
            params = state.params
            raw_valid = jnp.asarray(
                data.guardian_dependence_target_valid, dtype=jnp.float32
            )
            corrected_support = jnp.asarray(
                data.guardian_dependence_corrected_valid, dtype=jnp.float32
            )
            raw_support = proxy_raw_support_weights(
                raw_valid,
                corrected_support,
                pair_weight=config.proxy_raw_pair_weight,
            )
            raw_target = jnp.clip(
                jnp.asarray(data.guardian_dependence_target, dtype=jnp.float32),
                0.0,
                1.0,
            )
            corrected_target = jnp.clip(
                jnp.asarray(
                    data.guardian_dependence_corrected_target,
                    dtype=jnp.float32,
                ),
                0.0,
                1.0,
            )
            pair_support = raw_valid * corrected_support

            def loss_fn(proxy_params: hk.Params):
                raw = jax.nn.sigmoid(
                    self.agent.proxy_value(
                        proxy_params, data.obs, data.raw_action
                    )
                )
                corrected = jax.nn.sigmoid(
                    self.agent.proxy_value(
                        proxy_params, data.obs, data.proxy_corrected_action
                    )
                )
                raw_loss = _weighted_mean(
                    optax.huber_loss(
                        raw,
                        raw_target,
                        delta=max(config.proxy_huber_delta, 1e-6),
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
                pair_gap_loss = _weighted_mean(
                    optax.huber_loss(
                        raw - corrected,
                        raw_target - corrected_target,
                        delta=max(config.proxy_huber_delta, 1e-6),
                    ),
                    pair_support,
                )
                return (
                    raw_loss
                    + config.proxy_corrected_weight * corrected_loss
                    + config.proxy_pair_gap_weight * pair_gap_loss
                ), (
                    raw,
                    corrected,
                    raw_loss,
                    corrected_loss,
                    pair_gap_loss,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params.proxy
            )
            updates, proxy_opt = self.proxy_optim.update(
                grads, state.opt_state.proxy, params.proxy
            )
            proxy = optax.apply_updates(params.proxy, updates)
            raw, corrected, raw_loss, corrected_loss, pair_gap_loss = aux
            next_state = state._replace(
                params=params._replace(proxy=proxy, target_proxy=_copy_tree(proxy)),
                opt_state=state.opt_state._replace(proxy=proxy_opt),
                gradient_step=state.gradient_step + 1,
            )
            return next_state, {
                "proxy_loss": loss,
                "proxy_raw_loss": raw_loss,
                "proxy_corrected_loss": corrected_loss,
                "proxy_pair_gap_loss": pair_gap_loss,
                "proxy_mae": _weighted_mean(
                    jnp.abs(raw - raw_target), raw_support
                ),
                "proxy_brier": _weighted_mean(
                    jnp.square(raw - raw_target), raw_support
                ),
                "proxy_corrected_mae": _weighted_mean(
                    jnp.abs(corrected - corrected_target), corrected_support
                ),
                "proxy_pair_accuracy": _weighted_mean(
                    (raw > corrected).astype(jnp.float32), pair_support
                ),
                "proxy_raw_pair_mae": _weighted_mean(
                    jnp.abs(raw - raw_target), pair_support
                ),
                "proxy_pair_gap_mae": _weighted_mean(
                    jnp.abs(
                        (raw - corrected) - (raw_target - corrected_target)
                    ),
                    pair_support,
                ),
                "proxy_predicted_pair_gap": _weighted_mean(
                    raw - corrected, pair_support
                ),
                "proxy_target_pair_gap": _weighted_mean(
                    raw_target - corrected_target, pair_support
                ),
                "proxy_raw_mean": _weighted_mean(raw, raw_support),
                "proxy_corrected_mean": _weighted_mean(
                    corrected, corrected_support
                ),
                "proxy_support_fraction": jnp.mean(raw_valid),
                "proxy_raw_pair_weight": jnp.asarray(
                    config.proxy_raw_pair_weight, dtype=jnp.float32
                ),
                "proxy_pair_gap_weight": jnp.asarray(
                    config.proxy_pair_gap_weight, dtype=jnp.float32
                ),
                "proxy_corrected_support_fraction": jnp.mean(
                    corrected_support
                ),
            }

        self._proxy_update = jax.jit(proxy_update)

        def actor_update(key, state, data, hybrid_safe_policy):
            params = state.params
            actor_support = jnp.asarray(
                data.actor_controlled, dtype=jnp.float32
            )
            proxy_support = jnp.maximum(
                jnp.asarray(
                    data.guardian_dependence_target_valid, dtype=jnp.float32
                ),
                jnp.asarray(
                    data.guardian_dependence_corrected_valid,
                    dtype=jnp.float32,
                ),
            )
            safe_action = jax.lax.stop_gradient(
                self.agent.get_deterministic_action(
                    hybrid_safe_policy, data.obs
                )
            )
            proximal_coefficient = jnp.maximum(
                jnp.asarray(config.proximal_weight_min, dtype=jnp.float32),
                jnp.asarray(config.proximal_weight_initial, dtype=jnp.float32)
                * jnp.exp(
                    -jnp.asarray(state.proximal_step, dtype=jnp.float32)
                    / float(max(config.proximal_decay_steps, 1))
                ),
            )
            alpha = jnp.exp(params.log_alpha)

            def loss_fn(policy_params: hk.Params):
                action, logp = self.agent.evaluate(key, policy_params, data.obs)
                q1_mean, _ = self.agent.q_distribution(
                    params.q1, data.obs, action
                )
                q2_mean, _ = self.agent.q_distribution(
                    params.q2, data.obs, action
                )
                q_min = jnp.minimum(q1_mean, q2_mean)
                supported_sac = _weighted_mean(
                    alpha * logp - q_min, actor_support
                )
                proxy_logit = self.agent.proxy_value(
                    params.proxy, data.obs, action
                )
                p_policy = jax.nn.sigmoid(proxy_logit)
                absolute_proxy = _weighted_mean(p_policy, proxy_support)
                proxy_actor_risk = proxy_actor_objective(
                    proxy_logit,
                    objective=config.proxy_actor_objective,
                )
                proxy_actor_objective_loss = _weighted_mean(
                    proxy_actor_risk, proxy_support
                )
                deterministic_action = self.agent.get_deterministic_action(
                    policy_params, data.obs
                )
                delta = deterministic_action - safe_action
                action_drift = jnp.sum(jnp.square(delta), axis=-1)
                (
                    proximal,
                    proximal_speed_drift,
                    proximal_lateral_drift,
                ) = weighted_action_proximal(
                    delta,
                    speed_weight=config.proximal_speed_weight,
                    lateral_weight=config.proximal_lateral_weight,
                )
                weighted_sac = config.actor_sac_weight * supported_sac
                weighted_proxy = (
                    config.proxy_actor_weight * proxy_actor_objective_loss
                )
                weighted_proximal = proximal_coefficient * proximal
                loss = weighted_sac + weighted_proxy + weighted_proximal
                speed_fraction = jnp.clip(
                    0.5 * (action[..., 0] + 1.0), 0.0, 1.0 - 1e-7
                )
                speed_bin = jnp.floor(4.0 * speed_fraction).astype(jnp.int32)
                p_by_speed_bin = tuple(
                    _weighted_mean(
                        p_policy,
                        proxy_support
                        * jnp.asarray(speed_bin == index, dtype=jnp.float32),
                    )
                    for index in range(4)
                )
                p_speed_bin_fraction = tuple(
                    jnp.mean(jnp.asarray(speed_bin == index, dtype=jnp.float32))
                    for index in range(4)
                )
                return loss, (
                    logp,
                    q_min,
                    q1_mean,
                    q2_mean,
                    supported_sac,
                    absolute_proxy,
                    proxy_actor_objective_loss,
                    p_policy,
                    proximal,
                    action_drift,
                    proximal_speed_drift,
                    proximal_lateral_drift,
                    weighted_sac,
                    weighted_proxy,
                    weighted_proximal,
                    *p_by_speed_bin,
                    *p_speed_bin_fraction,
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
                q_min,
                q1_mean,
                q2_mean,
                supported_sac,
                absolute_proxy,
                proxy_actor_objective_loss,
                p_policy,
                proximal,
                action_drift,
                proximal_speed_drift,
                proximal_lateral_drift,
                weighted_sac,
                weighted_proxy,
                weighted_proximal,
                p_bin_0,
                p_bin_1,
                p_bin_2,
                p_bin_3,
                p_bin_fraction_0,
                p_bin_fraction_1,
                p_bin_fraction_2,
                p_bin_fraction_3,
            ) = aux

            def alpha_loss_fn(log_alpha):
                return _weighted_mean(
                    -log_alpha
                    * jax.lax.stop_gradient(
                        logp + self.agent.target_entropy
                    ),
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
                proximal_step=state.proximal_step + 1,
            )
            return next_state, {
                "policy_loss": loss,
                "supported_sac_loss": supported_sac,
                "absolute_proxy_loss": absolute_proxy,
                "proxy_actor_objective_loss": proxy_actor_objective_loss,
                "proxy_actor_softplus_logit": jnp.asarray(
                    config.proxy_actor_objective == "softplus_logit",
                    dtype=jnp.float32,
                ),
                "proximal_loss": proximal,
                "proximal_coefficient": proximal_coefficient,
                "proximal_step": jnp.asarray(
                    state.proximal_step, dtype=jnp.float32
                ),
                "weighted_sac_loss": weighted_sac,
                "weighted_proxy_loss": weighted_proxy,
                "weighted_proximal_loss": weighted_proximal,
                "proximal_speed_drift": proximal_speed_drift,
                "proximal_lateral_drift": proximal_lateral_drift,
                "actor_support_fraction": jnp.mean(actor_support),
                "proxy_actor_support_fraction": jnp.mean(proxy_support),
                "actor_q": _weighted_mean(q_min, actor_support),
                "q_disagreement": jnp.mean(jnp.abs(q1_mean - q2_mean)),
                "entropy": -_weighted_mean(logp, actor_support),
                "alpha": jnp.exp(log_alpha),
                "alpha_loss": alpha_loss,
                "hybrid_safe_action_drift": jnp.mean(action_drift),
                "p_policy_mean": _weighted_mean(p_policy, proxy_support),
                "p_policy_by_speed_bin_0": p_bin_0,
                "p_policy_by_speed_bin_1": p_bin_1,
                "p_policy_by_speed_bin_2": p_bin_2,
                "p_policy_by_speed_bin_3": p_bin_3,
                "p_policy_speed_bin_fraction_0": p_bin_fraction_0,
                "p_policy_speed_bin_fraction_1": p_bin_fraction_1,
                "p_policy_speed_bin_fraction_2": p_bin_fraction_2,
                "p_policy_speed_bin_fraction_3": p_bin_fraction_3,
            }

        self._actor_update = jax.jit(actor_update)

        def bc_update(state, data):
            weight = jnp.asarray(
                data.teacher_valid, dtype=jnp.float32
            ) * jnp.maximum(
                jnp.asarray(data.teacher_weight, dtype=jnp.float32), 0.0
            )

            def loss_fn(policy_params):
                action = self.agent.get_deterministic_action(
                    policy_params, data.obs
                )
                error = jnp.sum(
                    jnp.square(action - data.teacher_action), axis=-1
                )
                return _weighted_mean(error, weight), error

            (loss, error), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(state.params.policy)
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

        def support_diagnostics(
            data, offline_q1, offline_q2, offline_policy, offline_alpha
        ):
            mean_1, std_1 = self.agent.q_distribution(
                offline_q1, data.obs, data.action
            )
            mean_2, std_2 = self.agent.q_distribution(
                offline_q2, data.obs, data.action
            )
            next_action = self.agent.get_deterministic_action(
                offline_policy, data.next_obs
            )
            next_logp = self.agent.log_prob(
                offline_policy, data.next_obs, next_action
            )
            next_mean_1, _ = self.agent.q_distribution(
                offline_q1, data.next_obs, next_action
            )
            next_mean_2, _ = self.agent.q_distribution(
                offline_q2, data.next_obs, next_action
            )
            backup = jax.lax.stop_gradient(
                config.reward_scale
                * jnp.asarray(data.reward, dtype=jnp.float32)
                + (1.0 - jnp.asarray(data.done, dtype=jnp.float32))
                * config.gamma
                * (
                    jnp.minimum(next_mean_1, next_mean_2)
                    - jnp.asarray(offline_alpha, dtype=jnp.float32) * next_logp
                )
            )
            return (
                gaussian_wasserstein_disagreement(
                    mean_1,
                    std_1,
                    mean_2,
                    std_2,
                    std_weight=config.disagreement_std_weight,
                ),
                normalized_bellman_surprise(
                    mean_1, mean_2, std_1, std_2, backup
                ),
            )

        self._support_diagnostics = jax.jit(support_diagnostics)

    def _threshold_array(self) -> jax.Array:
        if self.offline_support_thresholds is None:
            raise RuntimeError("offline support thresholds are not calibrated")
        value = self.offline_support_thresholds
        return jnp.asarray(
            (
                value.disagreement_quantile,
                value.surprise_quantile,
                value.disagreement_temperature,
                value.surprise_temperature,
            ),
            dtype=jnp.float32,
        )

    def pretrain_bc(self, key, data: GuardianDSACExperience) -> Metric:
        del key
        self.state, metrics = self._bc_update(self.state, data)
        return _python_metrics(metrics)

    def offline_critic_update(
        self,
        key,
        data: GuardianDSACExperience,
        *,
        mode: str = "calql",
    ) -> Metric:
        if mode != "calql":
            raise ValueError("V2.3 offline critic mode must be 'calql'")
        self.state, metrics = self._offline_calql_update(
            key,
            self.state,
            data,
            self.state.params.q1,
            self.state.params.q2,
            self.state.params.policy,
            jnp.exp(self.state.params.log_alpha),
            jnp.ones((4,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.zeros_like(data.action),
            jnp.zeros_like(data.reward, dtype=jnp.float32),
            jnp.zeros_like(data.action),
            jnp.asarray(False),
        )
        return _python_metrics(metrics)

    def offline_critic_update_with_policy_actions(
        self,
        key,
        data: GuardianDSACExperience,
        *,
        next_action,
        policy_action,
    ) -> Metric:
        """Run the same Cal-QL DSAC update with an external density-free Actor.

        Guardian-Flow-DSAC uses its EMA one-step Actor for the Bellman next
        action and its live one-step Actor for the conservative policy sample.
        No Flow log probability is fabricated: the external entropy term is
        exactly zero while the Gaussian return-distribution Critic and Cal-QL
        machinery remain unchanged.
        """

        next_action = jnp.asarray(next_action, dtype=jnp.float32)
        policy_action = jnp.asarray(policy_action, dtype=jnp.float32)
        action_shape = jnp.asarray(data.action).shape
        if next_action.shape != action_shape:
            raise ValueError("external offline next action shape mismatch")
        if policy_action.shape != action_shape:
            raise ValueError("external offline policy action shape mismatch")
        self.state, metrics = self._offline_calql_update(
            key,
            self.state,
            data,
            self.state.params.q1,
            self.state.params.q2,
            self.state.params.policy,
            jnp.exp(self.state.params.log_alpha),
            jnp.ones((4,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            next_action,
            jnp.zeros_like(data.reward, dtype=jnp.float32),
            policy_action,
            jnp.asarray(True),
        )
        return _python_metrics(metrics)

    def proxy_update(self, data: GuardianDSACExperience) -> Metric:
        self.state, metrics = self._proxy_update(self.state, data)
        return _python_metrics(metrics)

    def offline_proxy_update(self, key, data: GuardianDSACExperience) -> Metric:
        del key
        return self.proxy_update(data)

    def offline_support_diagnostics(
        self, data: GuardianDSACExperience
    ) -> tuple[np.ndarray, np.ndarray]:
        disagreement, surprise = self._support_diagnostics(
            data,
            self.offline_reference_q1,
            self.offline_reference_q2,
            self.offline_reference_policy,
            jnp.asarray(self.offline_reference_alpha, dtype=jnp.float32),
        )
        return np.asarray(disagreement), np.asarray(surprise)

    def mark_offline_reference(
        self, validation_data: GuardianDSACExperience
    ) -> OfflineSupportThresholds:
        """Freeze offline Gaussian twins/policy and calibrate Seed0 support."""

        semantics = np.unique(
            np.asarray(validation_data.action_semantics_version).reshape(-1)
        )
        if semantics.tolist() != [3]:
            raise ValueError("support calibration requires V2.3 semantics code 3")
        self.offline_reference_q1 = _copy_tree(self.state.params.q1)
        self.offline_reference_q2 = _copy_tree(self.state.params.q2)
        self.offline_reference_policy = _copy_tree(self.state.params.policy)
        self.offline_reference_alpha = float(
            np.exp(np.asarray(self.state.params.log_alpha))
        )
        disagreement, surprise = self.offline_support_diagnostics(
            validation_data
        )
        self.offline_support_thresholds = calibrate_offline_support(
            disagreement,
            surprise,
            quantile=self.config.support_quantile,
        )
        self.offline_reference_ready = True
        self.online_reference_step = 0
        self.state = self.state._replace(
            proximal_step=jnp.asarray(0, dtype=jnp.int32)
        )
        self.hybrid_safe_snapshot = _copy_tree(self.state)
        self._safe_online_reference_step = 0
        return self.offline_support_thresholds

    def reference_coefficient(self) -> float:
        progress = min(
            max(self.online_reference_step, 0)
            / max(int(self.config.reference_decay_steps), 1),
            1.0,
        )
        return float(
            self.config.reference_weight_initial
            + progress
            * (
                self.config.reference_weight_final
                - self.config.reference_weight_initial
            )
        )

    def online_critic_update(
        self, key, data: GuardianDSACExperience
    ) -> Metric:
        if not self.offline_reference_ready:
            raise RuntimeError(
                "call mark_offline_reference() after offline Cal-QL"
            )
        coefficient = self.reference_coefficient()
        self.state, metrics = self._online_distributional_update(
            key,
            self.state,
            data,
            self.offline_reference_q1,
            self.offline_reference_q2,
            self.offline_reference_policy,
            jnp.asarray(self.offline_reference_alpha, dtype=jnp.float32),
            self._threshold_array(),
            jnp.asarray(coefficient, dtype=jnp.float32),
            jnp.zeros_like(data.action),
            jnp.zeros_like(data.reward, dtype=jnp.float32),
            jnp.zeros_like(data.action),
            jnp.asarray(False),
        )
        self.online_reference_step += 1
        return _python_metrics(metrics)

    def online_critic_update_with_policy_actions(
        self,
        key,
        data: GuardianDSACExperience,
        *,
        next_action,
        policy_action,
        next_log_prob=None,
    ) -> Metric:
        """Update the shared Critic using actions from an external Actor.

        V2.4's implicit one-step Flow Actor has no tractable density, so its
        Bellman entropy term is explicitly zero unless a log probability is
        supplied.  Frozen-offline support calibration remains unchanged.
        """

        if not self.offline_reference_ready:
            raise RuntimeError(
                "call mark_offline_reference() after offline Cal-QL"
            )
        next_action = jnp.asarray(next_action, dtype=jnp.float32)
        policy_action = jnp.asarray(policy_action, dtype=jnp.float32)
        if next_action.shape != jnp.asarray(data.action).shape:
            raise ValueError("external next action shape mismatch")
        if policy_action.shape != jnp.asarray(data.action).shape:
            raise ValueError("external policy action shape mismatch")
        if next_log_prob is None:
            next_log_prob = jnp.zeros_like(data.reward, dtype=jnp.float32)
        next_log_prob = jnp.asarray(next_log_prob, dtype=jnp.float32)
        if next_log_prob.shape != jnp.asarray(data.reward).shape:
            raise ValueError("external next log-probability shape mismatch")
        coefficient = self.reference_coefficient()
        self.state, metrics = self._online_distributional_update(
            key,
            self.state,
            data,
            self.offline_reference_q1,
            self.offline_reference_q2,
            self.offline_reference_policy,
            jnp.asarray(self.offline_reference_alpha, dtype=jnp.float32),
            self._threshold_array(),
            jnp.asarray(coefficient, dtype=jnp.float32),
            next_action,
            next_log_prob,
            policy_action,
            jnp.asarray(True),
        )
        self.online_reference_step += 1
        return _python_metrics(metrics)

    def online_critic_warmup(
        self, key, data: GuardianDSACExperience
    ) -> Metric:
        q_metrics = self.online_critic_update(key, data)
        p_metrics = self.proxy_update(data)
        return {
            **{f"task_{name}": value for name, value in q_metrics.items()},
            **p_metrics,
            "actor_frozen": 1.0,
        }

    def online_joint_update(
        self,
        key,
        data: GuardianDSACExperience,
        *,
        update_actor: bool = True,
    ) -> Metric:
        q_key, actor_key = jax.random.split(key)
        q_metrics = self.online_critic_update(q_key, data)
        p_metrics = self.proxy_update(data)
        actor_metrics: Metric = {}
        if update_actor:
            self.state, raw_actor_metrics = self._actor_update(
                actor_key,
                self.state,
                data,
                self.hybrid_safe_snapshot.params.policy,
            )
            actor_metrics = _python_metrics(raw_actor_metrics)
        return {
            **{f"task_{name}": value for name, value in q_metrics.items()},
            **p_metrics,
            **actor_metrics,
        }

    def update(self, key, data: GuardianDSACExperience) -> Metric:
        return self.online_joint_update(key, data)

    def promote_hybrid_safe(self) -> dict[str, str]:
        """Move the passive safe snapshot without restarting proximal decay."""

        self.hybrid_safe_snapshot = _copy_tree(self.state)
        self._safe_online_reference_step = int(self.online_reference_step)
        return {"snapshot": "hybrid_safe_snapshot", "rollback": "none"}

    def reject_hybrid_safety(self) -> dict[str, str]:
        """Restore Actor only; Task Critics and P continue learning."""

        safe_policy = _copy_tree(self.hybrid_safe_snapshot.params.policy)
        self.state = self.state._replace(
            params=self.state.params._replace(policy=safe_policy),
            opt_state=self.state.opt_state._replace(
                policy=self.actor_optim.init(safe_policy)
            ),
            proximal_step=jnp.asarray(0, dtype=jnp.int32),
        )
        return {"rollback": "actor_only", "snapshot": "hybrid_safe_snapshot"}

    def record_ordinary_failure(self) -> dict[str, str]:
        """Ordinary failure affects deployability, never the Learner state."""

        return {"rollback": "none", "deployment": "rejected"}

    def reject_numerical_disaster(self) -> dict[str, str]:
        self.state = _copy_tree(self.hybrid_safe_snapshot)._replace(
            proximal_step=jnp.asarray(0, dtype=jnp.int32)
        )
        self.online_reference_step = int(self._safe_online_reference_step)
        self._consecutive_large_critic_losses = 0
        return {"rollback": "hybrid_safe_full_snapshot"}

    # Compatibility spelling used by orchestration code.
    def rollback_to_hybrid_safe(self, *, full_state: bool = False) -> None:
        if full_state:
            self.reject_numerical_disaster()
        else:
            self.reject_hybrid_safety()

    def promote_deployable(
        self, metrics: Mapping[str, object] | None = None
    ) -> None:
        self.deployable_checkpoint = {
            "state": _copy_tree(self.state),
            "metrics": dict(metrics or {}),
            "online_reference_step": int(self.online_reference_step),
        }

    @property
    def has_deployable_checkpoint(self) -> bool:
        return self.deployable_checkpoint is not None

    def numerical_disaster_reasons(
        self,
        *,
        metrics: Mapping[str, float] | None = None,
        observation=None,
        action=None,
    ) -> list[str]:
        reasons: list[str] = []
        if any(
            not np.all(np.isfinite(np.asarray(leaf)))
            for leaf in jax.tree_util.tree_leaves(self.state)
        ):
            reasons.append("non_finite_state")
        for name, value in (
            ("ema_q_std_1", self.state.ema_q_std_1),
            ("ema_q_std_2", self.state.ema_q_std_2),
        ):
            numeric = float(np.asarray(value))
            if not np.isfinite(numeric) or numeric <= 0.0:
                reasons.append(f"invalid_{name}")
        if metrics is not None:
            numeric_metrics = [float(value) for value in metrics.values()]
            if any(not np.isfinite(value) for value in numeric_metrics):
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
                reasons.append("distribution_disagreement")
            for suffix in ("q_std_1", "q_std_2", "ema_q_std_1", "ema_q_std_2"):
                values = [
                    float(value)
                    for name, value in metrics.items()
                    if name.endswith(suffix)
                ]
                if any(not np.isfinite(value) or value <= 0.0 for value in values):
                    reasons.append("invalid_distribution_std")
        if observation is not None and action is not None:
            mean_1, std_1, mean_2, std_2 = self.task_q_distribution(
                observation, action
            )
            if not all(
                np.all(np.isfinite(value))
                for value in (mean_1, std_1, mean_2, std_2)
            ):
                reasons.append("non_finite_distribution")
            if np.any(std_1 <= 0.0) or np.any(std_2 <= 0.0):
                reasons.append("non_positive_distribution_std")
            p = self.proxy_value(observation, action)
            if not np.all(np.isfinite(p)):
                reasons.append("non_finite_proxy")
            elif np.any((p < 0.0) | (p > 1.0)):
                reasons.append("proxy_out_of_range")
        return sorted(set(reasons))

    def get_action(self, key, observation) -> np.ndarray:
        return np.array(
            self._get_action(key, self.state.params.policy, observation),
            copy=True,
        )

    def get_deterministic_action(self, observation) -> np.ndarray:
        return np.array(
            self._get_deterministic_action(
                self.state.params.policy, observation
            ),
            copy=True,
        )

    def get_deterministic_hybrid_safe_action(self, observation) -> np.ndarray:
        return np.array(
            self._get_deterministic_action(
                self.hybrid_safe_snapshot.params.policy, observation
            ),
            copy=True,
        )

    # V2.2 evaluator compatibility: V2.3's passive safe snapshot replaces Champion.
    def get_deterministic_champion_action(self, observation) -> np.ndarray:
        return self.get_deterministic_hybrid_safe_action(observation)

    def get_deterministic_deployable_action(self, observation) -> np.ndarray:
        if self.deployable_checkpoint is None:
            raise RuntimeError("no deployable checkpoint has been certified")
        state = self.deployable_checkpoint["state"]
        return np.array(
            self._get_deterministic_action(state.params.policy, observation),
            copy=True,
        )

    def select_evaluation_checkpoint(self, policy: str) -> None:
        """Activate the exact passive artifact inside an isolated evaluator."""

        if policy == "learner":
            return
        if policy == "hybrid_safe":
            self.state = _copy_tree(self.hybrid_safe_snapshot)
            return
        if policy == "deployable":
            if self.deployable_checkpoint is None:
                raise RuntimeError("no deployable checkpoint has been certified")
            self.state = _copy_tree(self.deployable_checkpoint["state"])
            return
        raise ValueError(f"unknown evaluation policy: {policy}")

    def task_q_distribution(
        self, observation, action
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        result = self._q_distribution(
            self.state.params.q1,
            self.state.params.q2,
            observation,
            action,
        )
        return tuple(np.asarray(value) for value in result)

    def task_q(self, observation, action) -> tuple[np.ndarray, np.ndarray]:
        mean_1, _, mean_2, _ = self.task_q_distribution(observation, action)
        return mean_1, mean_2

    def proxy_value(self, observation, action) -> np.ndarray:
        return np.asarray(
            self._proxy_value_eval(
                self.state.params.proxy, observation, action
            )
        )

    def _metadata(self) -> dict[str, object]:
        return {
            "action_semantics_version": self.action_semantics_version,
            "scene_seeds": list(self.scene_seeds),
            "scene_signatures": dict(self.scene_signatures),
            "dataset_sha256": self.dataset_sha256,
            "observation_dim": int(self.agent.critic_obs_dim),
            "actor_observation_dim": int(self.agent.actor_obs_dim),
            "action_dim": int(self.agent.act_dim),
            "distributional_q_output_dim": 2,
        }

    @staticmethod
    def _threshold_payload(
        thresholds: OfflineSupportThresholds | None,
    ) -> dict[str, float] | None:
        return None if thresholds is None else asdict(thresholds)

    def _validate_loaded_state(self) -> None:
        leaves = jax.tree_util.tree_leaves(
            (
                self.state,
                self.hybrid_safe_snapshot,
                None
                if self.deployable_checkpoint is None
                else self.deployable_checkpoint["state"],
                self.offline_reference_q1,
                self.offline_reference_q2,
                self.offline_reference_policy,
            )
        )
        if any(not np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
            raise ValueError("V2.3 checkpoint contains non-finite parameters/state")
        observation = jnp.zeros(
            (1, self.agent.critic_obs_dim), dtype=jnp.float32
        )
        action = jnp.zeros((1, self.agent.act_dim), dtype=jnp.float32)
        for name, params in (
            ("learner_q1", self.state.params.q1),
            ("learner_q2", self.state.params.q2),
            ("offline_q1", self.offline_reference_q1),
            ("offline_q2", self.offline_reference_q2),
        ):
            mean, std = self.agent.q_distribution(params, observation, action)
            mean_np = np.asarray(mean)
            std_np = np.asarray(std)
            if mean_np.shape != (1,) or std_np.shape != (1,):
                raise ValueError(f"{name} distributional output dimension mismatch")
            if not np.all(np.isfinite(mean_np)) or not np.all(np.isfinite(std_np)):
                raise ValueError(f"{name} distribution output is non-finite")
            if np.any(std_np <= 0.0):
                raise ValueError(f"{name} distribution std is not positive")
        for value in (self.state.ema_q_std_1, self.state.ema_q_std_2):
            if not np.isfinite(float(np.asarray(value))) or float(value) <= 0.0:
                raise ValueError("checkpoint EMA Q std is invalid")

    def _checkpoint_payload(self) -> dict[str, object]:
        """Return the complete v7 payload for self-contained wrappers."""

        return {
            "format": self.FORMAT,
            "learner_state": jax.device_get(self.state),
            "hybrid_safe_snapshot": jax.device_get(
                self.hybrid_safe_snapshot
            ),
            "deployable_checkpoint": jax.device_get(
                self.deployable_checkpoint
            ),
            "offline_reference_q1": jax.device_get(self.offline_reference_q1),
            "offline_reference_q2": jax.device_get(self.offline_reference_q2),
            "offline_reference_policy": jax.device_get(
                self.offline_reference_policy
            ),
            "offline_reference_alpha": float(self.offline_reference_alpha),
            "offline_support_thresholds": self._threshold_payload(
                self.offline_support_thresholds
            ),
            "ema_q_std_1": float(np.asarray(self.state.ema_q_std_1)),
            "ema_q_std_2": float(np.asarray(self.state.ema_q_std_2)),
            "offline_reference_ready": bool(self.offline_reference_ready),
            "online_reference_step": int(self.online_reference_step),
            "safe_online_reference_step": int(
                self._safe_online_reference_step
            ),
            "config": asdict(self.config),
            **self._metadata(),
        }

    def save(self, path: str) -> None:
        payload = self._checkpoint_payload()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            pickle.dump(payload, stream)

    def _load_checkpoint_payload(self, payload: object) -> None:
        """Load an already-decoded v7 payload with all binding checks."""

        if not isinstance(payload, dict) or payload.get("format") != self.FORMAT:
            raise ValueError("checkpoint is not Guardian-DSAC V2.3 v7")
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
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"V2.3 checkpoint missing fields: {missing}")
        expected = self._metadata()
        observed = {
            name: payload.get(name)
            for name in expected
        }
        observed["scene_seeds"] = [int(seed) for seed in observed["scene_seeds"]]
        observed["scene_signatures"] = _normalise_signatures(
            observed["scene_signatures"]
        )
        mismatches = [
            name for name in expected if observed.get(name) != expected.get(name)
        ]
        if mismatches:
            raise ValueError(
                "V2.3 checkpoint binding mismatch: " + ", ".join(mismatches)
            )
        self.state = jax.device_put(payload["learner_state"])
        self.hybrid_safe_snapshot = jax.device_put(
            payload["hybrid_safe_snapshot"]
        )
        deployable = payload["deployable_checkpoint"]
        self.deployable_checkpoint = (
            None
            if deployable is None
            else {
                **deployable,
                "state": jax.device_put(deployable["state"]),
            }
        )
        self.offline_reference_q1 = jax.device_put(
            payload["offline_reference_q1"]
        )
        self.offline_reference_q2 = jax.device_put(
            payload["offline_reference_q2"]
        )
        self.offline_reference_policy = jax.device_put(
            payload["offline_reference_policy"]
        )
        self.offline_reference_alpha = float(payload["offline_reference_alpha"])
        threshold_payload = payload["offline_support_thresholds"]
        self.offline_support_thresholds = (
            None
            if threshold_payload is None
            else OfflineSupportThresholds(**threshold_payload)
        )
        self.offline_reference_ready = bool(
            payload.get("offline_reference_ready", threshold_payload is not None)
        )
        self.online_reference_step = int(payload.get("online_reference_step", 0))
        self._safe_online_reference_step = int(
            payload.get("safe_online_reference_step", 0)
        )
        if not np.isclose(
            float(np.asarray(self.state.ema_q_std_1)),
            float(payload["ema_q_std_1"]),
            atol=1e-8,
        ) or not np.isclose(
            float(np.asarray(self.state.ema_q_std_2)),
            float(payload["ema_q_std_2"]),
            atol=1e-8,
        ):
            raise ValueError("V2.3 checkpoint EMA std binding mismatch")
        self._validate_loaded_state()
        self.loaded_checkpoint_format = self.FORMAT

    def load(self, path: str) -> None:
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        self._load_checkpoint_payload(payload)


__all__ = [
    "GuardianDSACOptStates",
    "GuardianDSACTrainState",
    "GuardianDSACV23",
    "GuardianDSACV23Config",
    "calql_mean_conservative_gap",
    "proxy_actor_objective",
    "proxy_raw_support_weights",
    "weighted_action_proximal",
]
