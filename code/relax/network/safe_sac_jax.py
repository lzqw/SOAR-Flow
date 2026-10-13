"""Asymmetric JAX networks for intervention-aware Safe SAC.

The policy deliberately receives only the task observation.  Safety mode,
authorization and intervention context may be useful to the critic while
training, but exposing them to the actor creates a policy that cannot be
evaluated honestly with the safety layer disabled.  The Q functions therefore
consume the complete observation while the policy slices the first
``actor_obs_dim`` features in every public action path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from relax.network.blocks import Activation, DistributionalQNet2, PolicyNet, QNet


class JaxSafeSACParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    log_alpha: jax.Array


class GuardianCalQLParams(NamedTuple):
    """Parameters with task critics and a semantically separate proxy P."""

    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    proxy: hk.Params
    target_proxy: hk.Params
    policy: hk.Params
    log_alpha: jax.Array


class GuardianDSACParams(NamedTuple):
    """V2.3 parameters with two Gaussian return-distribution Critics."""

    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    proxy: hk.Params
    target_proxy: hk.Params
    policy: hk.Params
    log_alpha: jax.Array


@dataclass(frozen=True)
class JaxSafeSACNet:
    policy: Callable[[hk.Params, jax.Array], Tuple[jax.Array, jax.Array]]
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    actor_obs_dim: int
    target_entropy: float
    action_scale: float = 1.0
    log_prob_epsilon: float = 1e-6

    def actor_observation(self, observation: jax.Array) -> jax.Array:
        return observation[..., : self.actor_obs_dim]

    def distribution(
        self, policy_params: hk.Params, observation: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        return self.policy(policy_params, self.actor_observation(observation))

    def get_action(
        self,
        key: jax.Array,
        policy_params: hk.Params,
        observation: jax.Array,
    ) -> jax.Array:
        mean, std = self.distribution(policy_params, observation)
        pre_tanh = mean + std * jax.random.normal(key, mean.shape)
        return jnp.tanh(pre_tanh) * self.action_scale

    def get_deterministic_action(
        self, policy_params: hk.Params, observation: jax.Array
    ) -> jax.Array:
        mean, _ = self.distribution(policy_params, observation)
        return jnp.tanh(mean) * self.action_scale

    def evaluate(
        self,
        key: jax.Array,
        policy_params: hk.Params,
        observation: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Reparameterized action and correctly squashed log probability."""

        mean, std = self.distribution(policy_params, observation)
        noise = jax.random.normal(key, mean.shape)
        pre_tanh = mean + std * noise
        squashed = jnp.tanh(pre_tanh)
        normal_log_prob = -0.5 * (
            noise**2 + 2.0 * jnp.log(std) + math.log(2.0 * math.pi)
        )
        scale = jnp.asarray(self.action_scale, dtype=pre_tanh.dtype)
        squash_log_det = jnp.log(
            jnp.maximum(scale * (1.0 - squashed**2), self.log_prob_epsilon)
        )
        log_prob = jnp.sum(normal_log_prob - squash_log_det, axis=-1)
        return squashed * scale, log_prob


@dataclass(frozen=True)
class JaxGuardianCalQLNet:
    """Actor, task-Q ensemble and guardian-dependence proxy network bundle."""

    policy: Callable[[hk.Params, jax.Array], Tuple[jax.Array, jax.Array]]
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    proxy: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    actor_obs_dim: int
    target_entropy: float
    action_scale: float = 1.0
    log_prob_epsilon: float = 1e-6

    def actor_observation(self, observation: jax.Array) -> jax.Array:
        return observation[..., : self.actor_obs_dim]

    def distribution(
        self, policy_params: hk.Params, observation: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        return self.policy(policy_params, self.actor_observation(observation))

    def get_action(
        self,
        key: jax.Array,
        policy_params: hk.Params,
        observation: jax.Array,
    ) -> jax.Array:
        mean, std = self.distribution(policy_params, observation)
        pre_tanh = mean + std * jax.random.normal(key, mean.shape)
        return jnp.tanh(pre_tanh) * self.action_scale

    def get_deterministic_action(
        self, policy_params: hk.Params, observation: jax.Array
    ) -> jax.Array:
        mean, _ = self.distribution(policy_params, observation)
        return jnp.tanh(mean) * self.action_scale

    def evaluate(
        self,
        key: jax.Array,
        policy_params: hk.Params,
        observation: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        mean, std = self.distribution(policy_params, observation)
        noise = jax.random.normal(key, mean.shape)
        pre_tanh = mean + std * noise
        squashed = jnp.tanh(pre_tanh)
        normal_log_prob = -0.5 * (
            noise**2 + 2.0 * jnp.log(std) + math.log(2.0 * math.pi)
        )
        scale = jnp.asarray(self.action_scale, dtype=pre_tanh.dtype)
        squash_log_det = jnp.log(
            jnp.maximum(scale * (1.0 - squashed**2), self.log_prob_epsilon)
        )
        log_prob = jnp.sum(normal_log_prob - squash_log_det, axis=-1)
        return squashed * scale, log_prob

    def log_prob(
        self,
        policy_params: hk.Params,
        observation: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        """Log probability of a recorded action under the squashed policy.

        AWAC evaluates dataset actions rather than reparameterized policy
        samples.  Inverting the tanh transform here keeps that likelihood
        exactly consistent with :meth:`evaluate` without changing the policy
        network or action space.
        """

        mean, std = self.distribution(policy_params, observation)
        scale = jnp.asarray(self.action_scale, dtype=mean.dtype)
        squashed = jnp.clip(
            jnp.asarray(action, dtype=mean.dtype) / scale,
            -1.0 + self.log_prob_epsilon,
            1.0 - self.log_prob_epsilon,
        )
        pre_tanh = jnp.arctanh(squashed)
        noise = (pre_tanh - mean) / jnp.maximum(std, self.log_prob_epsilon)
        normal_log_prob = -0.5 * (
            noise**2 + 2.0 * jnp.log(std) + math.log(2.0 * math.pi)
        )
        squash_log_det = jnp.log(
            jnp.maximum(scale * (1.0 - squashed**2), self.log_prob_epsilon)
        )
        return jnp.sum(normal_log_prob - squash_log_det, axis=-1)

    def proxy_value(
        self, proxy_params: hk.Params, observation: jax.Array, action: jax.Array
    ) -> jax.Array:
        """Return the signed Guardian-boundary score (higher means safer).

        The score must remain signed: same-state ranking places corrected
        actions above rejected raw actions, while a small L2 anchor keeps the
        boundary close to zero.  A softplus output would erase that calibrated
        boundary and would make the Actor objective globally optimize an
        arbitrary non-negative magnitude.
        """

        return self.proxy(proxy_params, observation, action)


@dataclass(frozen=True)
class JaxGuardianDSACNet(JaxGuardianCalQLNet):
    """Gaussian distributional Task Critics plus the unchanged V2.2 P/Actor."""

    critic_obs_dim: int = 0
    act_dim: int = 0
    sigma_min: float = 1e-3

    def q_distribution(
        self,
        q_params: hk.Params,
        observation: jax.Array,
        action: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        mean, std = self.q(q_params, observation, action)
        return mean, std


def create_jax_safe_sac_net(
    key: jax.Array,
    *,
    critic_obs_dim: int,
    actor_obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int] = (256, 256),
    activation: Activation = jax.nn.relu,
    action_scale: float = 1.0,
    init_alpha: float = 0.1,
    min_log_std: float = -10.0,
    max_log_std: float = 1.0,
) -> Tuple[JaxSafeSACNet, JaxSafeSACParams]:
    if not 0 < actor_obs_dim <= critic_obs_dim:
        raise ValueError("require 0 < actor_obs_dim <= critic_obs_dim")
    if act_dim < 1:
        raise ValueError("act_dim must be positive")

    q = hk.without_apply_rng(
        hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act))
    )
    policy = hk.without_apply_rng(
        hk.transform(
            lambda obs: PolicyNet(
                act_dim,
                hidden_sizes,
                activation,
                min_log_std=min_log_std,
                max_log_std=max_log_std,
                log_std_mode="shared",
            )(obs)
        )
    )

    q1_key, q2_key, policy_key = jax.random.split(key, 3)
    critic_observation = jnp.zeros((1, critic_obs_dim), dtype=jnp.float32)
    actor_observation = jnp.zeros((1, actor_obs_dim), dtype=jnp.float32)
    action = jnp.zeros((1, act_dim), dtype=jnp.float32)
    q1_params = q.init(q1_key, critic_observation, action)
    q2_params = q.init(q2_key, critic_observation, action)
    policy_params = policy.init(policy_key, actor_observation)
    params = JaxSafeSACParams(
        q1=q1_params,
        q2=q2_params,
        target_q1=q1_params,
        target_q2=q2_params,
        policy=policy_params,
        log_alpha=jnp.asarray(math.log(float(init_alpha)), dtype=jnp.float32),
    )
    network = JaxSafeSACNet(
        policy=policy.apply,
        q=q.apply,
        actor_obs_dim=int(actor_obs_dim),
        target_entropy=-float(act_dim),
        action_scale=float(action_scale),
    )
    return network, params


def create_guardian_calql_net(
    key: jax.Array,
    *,
    critic_obs_dim: int,
    actor_obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int] = (256, 256),
    activation: Activation = jax.nn.relu,
    action_scale: float = 1.0,
    init_alpha: float = 0.1,
    min_log_std: float = -10.0,
    max_log_std: float = 1.0,
) -> Tuple[JaxGuardianCalQLNet, GuardianCalQLParams]:
    """Create networks for Guardian-Calibrated Offline-to-Online Safe SAC."""

    if not 0 < actor_obs_dim <= critic_obs_dim:
        raise ValueError("require 0 < actor_obs_dim <= critic_obs_dim")
    if act_dim < 1:
        raise ValueError("act_dim must be positive")

    q = hk.without_apply_rng(
        hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act))
    )
    proxy = hk.without_apply_rng(
        hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act))
    )
    policy = hk.without_apply_rng(
        hk.transform(
            lambda obs: PolicyNet(
                act_dim,
                hidden_sizes,
                activation,
                min_log_std=min_log_std,
                max_log_std=max_log_std,
                log_std_mode="shared",
            )(obs)
        )
    )

    q1_key, q2_key, proxy_key, policy_key = jax.random.split(key, 4)
    critic_observation = jnp.zeros((1, critic_obs_dim), dtype=jnp.float32)
    actor_observation = jnp.zeros((1, actor_obs_dim), dtype=jnp.float32)
    action = jnp.zeros((1, act_dim), dtype=jnp.float32)
    q1_params = q.init(q1_key, critic_observation, action)
    q2_params = q.init(q2_key, critic_observation, action)
    proxy_params = proxy.init(proxy_key, critic_observation, action)
    policy_params = policy.init(policy_key, actor_observation)
    params = GuardianCalQLParams(
        q1=q1_params,
        q2=q2_params,
        target_q1=q1_params,
        target_q2=q2_params,
        proxy=proxy_params,
        target_proxy=proxy_params,
        policy=policy_params,
        log_alpha=jnp.asarray(math.log(float(init_alpha)), dtype=jnp.float32),
    )
    network = JaxGuardianCalQLNet(
        policy=policy.apply,
        q=q.apply,
        proxy=proxy.apply,
        actor_obs_dim=int(actor_obs_dim),
        target_entropy=-float(act_dim),
        action_scale=float(action_scale),
    )
    return network, params


def create_guardian_dsac_net(
    key: jax.Array,
    *,
    critic_obs_dim: int,
    actor_obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int] = (256, 256),
    activation: Activation = jax.nn.relu,
    action_scale: float = 1.0,
    init_alpha: float = 0.1,
    min_log_std: float = -10.0,
    max_log_std: float = 1.0,
    sigma_min: float = 1e-3,
) -> Tuple[JaxGuardianDSACNet, GuardianDSACParams]:
    """Create Guardian-DSAC V2.3 networks without altering V2.2 factories."""

    if not 0 < actor_obs_dim <= critic_obs_dim:
        raise ValueError("require 0 < actor_obs_dim <= critic_obs_dim")
    if act_dim < 1:
        raise ValueError("act_dim must be positive")
    if not math.isfinite(float(sigma_min)) or float(sigma_min) <= 0.0:
        raise ValueError("sigma_min must be finite and positive")

    q = hk.without_apply_rng(
        hk.transform(
            lambda obs, act: DistributionalQNet2(
                hidden_sizes,
                activation,
                sigma_min=float(sigma_min),
            )(obs, act)
        )
    )
    proxy = hk.without_apply_rng(
        hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act))
    )
    policy = hk.without_apply_rng(
        hk.transform(
            lambda obs: PolicyNet(
                act_dim,
                hidden_sizes,
                activation,
                min_log_std=min_log_std,
                max_log_std=max_log_std,
                log_std_mode="shared",
            )(obs)
        )
    )

    q1_key, q2_key, proxy_key, policy_key = jax.random.split(key, 4)
    critic_observation = jnp.zeros((1, critic_obs_dim), dtype=jnp.float32)
    actor_observation = jnp.zeros((1, actor_obs_dim), dtype=jnp.float32)
    action = jnp.zeros((1, act_dim), dtype=jnp.float32)
    q1_params = q.init(q1_key, critic_observation, action)
    q2_params = q.init(q2_key, critic_observation, action)
    proxy_params = proxy.init(proxy_key, critic_observation, action)
    policy_params = policy.init(policy_key, actor_observation)
    params = GuardianDSACParams(
        q1=q1_params,
        q2=q2_params,
        target_q1=q1_params,
        target_q2=q2_params,
        proxy=proxy_params,
        target_proxy=proxy_params,
        policy=policy_params,
        log_alpha=jnp.asarray(math.log(float(init_alpha)), dtype=jnp.float32),
    )
    network = JaxGuardianDSACNet(
        policy=policy.apply,
        q=q.apply,
        proxy=proxy.apply,
        actor_obs_dim=int(actor_obs_dim),
        target_entropy=-float(act_dim),
        action_scale=float(action_scale),
        critic_obs_dim=int(critic_obs_dim),
        act_dim=int(act_dim),
        sigma_min=float(sigma_min),
    )
    return network, params
