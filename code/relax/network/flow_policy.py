"""Conditional BC flow prior and one-step deployment Actor for V2.4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class FlowPolicyParams(NamedTuple):
    bc_flow: hk.Params
    one_step: hk.Params


@dataclass(frozen=True)
class JaxOneStepFlowPolicy:
    """Flow bundle with no candidate bank or rejection-sampling path."""

    bc_velocity: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    one_step: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    actor_obs_dim: int
    act_dim: int
    hidden_sizes: tuple[int, ...]
    integration_steps: int = 8

    def actor_observation(self, observation: jax.Array) -> jax.Array:
        return observation[..., : self.actor_obs_dim]

    def bc_action(
        self,
        params: hk.Params,
        observation: jax.Array,
        noise: jax.Array,
    ) -> jax.Array:
        """Integrate only the frozen offline BC prior used for distillation."""

        observation = self.actor_observation(observation)
        step_size = 1.0 / float(self.integration_steps)

        def body(index, action):
            time = jnp.full(
                (*action.shape[:-1], 1),
                (jnp.asarray(index, dtype=action.dtype) + 0.5) * step_size,
                dtype=action.dtype,
            )
            velocity = self.bc_velocity(
                params, observation, action, time
            )
            return action + step_size * velocity

        action = jax.lax.fori_loop(0, self.integration_steps, body, noise)
        return jnp.clip(action, -1.0, 1.0)

    def get_action(
        self,
        params: hk.Params,
        observation: jax.Array,
        noise: jax.Array,
    ) -> jax.Array:
        """One network forward pass; the online Actor never integrates an ODE."""

        return self.one_step(
            params, self.actor_observation(observation), noise
        )

    def sample_action(
        self,
        params: hk.Params,
        observation: jax.Array,
        key: jax.Array,
        *,
        latent_scale: float = 1.0,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample one Gaussian latent and execute exactly one Actor forward."""

        observation = jnp.asarray(observation, dtype=jnp.float32)
        if not np.isfinite(latent_scale) or latent_scale <= 0.0:
            raise ValueError("latent_scale must be finite and positive")
        noise = latent_scale * jax.random.normal(
            key,
            (*observation.shape[:-1], self.act_dim),
            dtype=observation.dtype,
        )
        return self.get_action(params, observation, noise), noise

    def latent_action_diversity(
        self,
        params: hk.Params,
        observation: jax.Array,
        noise_1: jax.Array,
        noise_2: jax.Array,
    ) -> jax.Array:
        """Diagnostic only: mean action distance between two latent draws."""

        action_1 = self.get_action(params, observation, noise_1)
        action_2 = self.get_action(params, observation, noise_2)
        return jnp.mean(jnp.linalg.norm(action_1 - action_2, axis=-1))

    def get_deterministic_action(
        self, params: hk.Params, observation: jax.Array
    ) -> jax.Array:
        observation = jnp.asarray(observation)
        noise = jnp.zeros(
            (*observation.shape[:-1], self.act_dim), dtype=observation.dtype
        )
        return self.get_action(params, observation, noise)


def flow_matching_loss(
    policy: JaxOneStepFlowPolicy,
    params: hk.Params,
    observation: jax.Array,
    action: jax.Array,
    noise: jax.Array,
    time: jax.Array,
) -> jax.Array:
    """Conditional linear flow-matching objective on safe executed actions."""

    action = jnp.asarray(action, dtype=jnp.float32)
    noise = jnp.asarray(noise, dtype=action.dtype)
    time = jnp.asarray(time, dtype=action.dtype)
    interpolated = (1.0 - time) * noise + time * action
    target_velocity = action - noise
    predicted = policy.bc_velocity(
        params,
        policy.actor_observation(observation),
        interpolated,
        time,
    )
    return jnp.mean(jnp.sum(jnp.square(predicted - target_velocity), axis=-1))


def one_step_distillation_loss(
    policy: JaxOneStepFlowPolicy,
    one_step_params: hk.Params,
    bc_params: hk.Params,
    observation: jax.Array,
    noise: jax.Array,
) -> jax.Array:
    target = jax.lax.stop_gradient(
        policy.bc_action(bc_params, observation, noise)
    )
    predicted = policy.get_action(
        one_step_params, observation, noise
    )
    return jnp.mean(jnp.sum(jnp.square(predicted - target), axis=-1))


def create_one_step_flow_policy(
    key: jax.Array,
    *,
    actor_obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int] = (256, 256),
    integration_steps: int = 8,
) -> tuple[JaxOneStepFlowPolicy, FlowPolicyParams]:
    if actor_obs_dim < 1 or act_dim < 1:
        raise ValueError("flow observation/action dimensions must be positive")
    if integration_steps < 1:
        raise ValueError("integration_steps must be positive")
    hidden_sizes = tuple(int(value) for value in hidden_sizes)
    if not hidden_sizes or any(value < 1 for value in hidden_sizes):
        raise ValueError("flow hidden sizes must be positive")

    def bc_fn(obs, action, time):
        inputs = jnp.concatenate((obs, action, time), axis=-1)
        return hk.nets.MLP(
            (*hidden_sizes, act_dim),
            activation=jax.nn.relu,
            activate_final=False,
        )(inputs)

    def one_step_fn(obs, noise):
        inputs = jnp.concatenate((obs, noise), axis=-1)
        raw = hk.nets.MLP(
            (*hidden_sizes, act_dim),
            activation=jax.nn.relu,
            activate_final=False,
        )(inputs)
        return jnp.tanh(raw)

    bc = hk.without_apply_rng(hk.transform(bc_fn))
    one_step = hk.without_apply_rng(hk.transform(one_step_fn))
    bc_key, one_key = jax.random.split(key)
    observation = jnp.zeros((1, actor_obs_dim), dtype=jnp.float32)
    action = jnp.zeros((1, act_dim), dtype=jnp.float32)
    time = jnp.zeros((1, 1), dtype=jnp.float32)
    params = FlowPolicyParams(
        bc_flow=bc.init(bc_key, observation, action, time),
        one_step=one_step.init(one_key, observation, action),
    )
    return (
        JaxOneStepFlowPolicy(
            bc_velocity=bc.apply,
            one_step=one_step.apply,
            actor_obs_dim=int(actor_obs_dim),
            act_dim=int(act_dim),
            hidden_sizes=hidden_sizes,
            integration_steps=int(integration_steps),
        ),
        params,
    )


__all__ = [
    "FlowPolicyParams",
    "JaxOneStepFlowPolicy",
    "create_one_step_flow_policy",
    "flow_matching_loss",
    "one_step_distillation_loss",
]
