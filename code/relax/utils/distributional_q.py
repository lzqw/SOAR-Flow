"""Pure Gaussian distributional-Q utilities for Guardian-DSAC V2.3.

The helpers in this module deliberately know nothing about scene identifiers.
Soft offline-reference support is inferred from frozen twin-distribution
disagreement and normalized Bellman surprise, never from the seed that
produced a transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


DEFAULT_SIGMA_MIN = 1e-3
DEFAULT_NOISE_CLIP = 3.0


class GaussianTarget(NamedTuple):
    mean: jax.Array
    sample: jax.Array
    std: jax.Array
    branch: jax.Array


@dataclass(frozen=True)
class OfflineSupportThresholds:
    disagreement_quantile: float
    surprise_quantile: float
    disagreement_temperature: float
    surprise_temperature: float

    def __post_init__(self) -> None:
        values = (
            self.disagreement_quantile,
            self.surprise_quantile,
            self.disagreement_temperature,
            self.surprise_temperature,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("offline support thresholds must be finite")
        if self.disagreement_temperature <= 0.0 or self.surprise_temperature <= 0.0:
            raise ValueError("offline support temperatures must be positive")


def positive_std(raw_scale: jax.Array, sigma_min: float = DEFAULT_SIGMA_MIN) -> jax.Array:
    """Convert an unconstrained head to a finite strictly-positive std."""

    sigma_min = float(sigma_min)
    if not np.isfinite(sigma_min) or sigma_min <= 0.0:
        raise ValueError("sigma_min must be finite and positive")
    return jax.nn.softplus(jnp.asarray(raw_scale)) + sigma_min


def sample_gaussian_return(
    key: jax.Array | None,
    mean: jax.Array,
    std: jax.Array,
    *,
    noise: jax.Array | None = None,
    noise_clip: float = DEFAULT_NOISE_CLIP,
) -> jax.Array:
    """Reparameterized Gaussian return sample with injectable test noise."""

    mean = jnp.asarray(mean)
    std = jnp.asarray(std, dtype=mean.dtype)
    if noise is None:
        if key is None:
            raise ValueError("key is required when fixed noise is not supplied")
        noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    else:
        noise = jnp.broadcast_to(jnp.asarray(noise, dtype=mean.dtype), mean.shape)
    bound = jnp.asarray(float(noise_clip), dtype=mean.dtype)
    return mean + jnp.clip(noise, -bound, bound) * std


def select_conservative_gaussian(
    mean_1: jax.Array,
    std_1: jax.Array,
    mean_2: jax.Array,
    std_2: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Select the target-Critic branch with the smaller return mean."""

    mean_1 = jnp.asarray(mean_1)
    mean_2 = jnp.asarray(mean_2, dtype=mean_1.dtype)
    std_1 = jnp.asarray(std_1, dtype=mean_1.dtype)
    std_2 = jnp.asarray(std_2, dtype=mean_1.dtype)
    choose_second = mean_2 < mean_1
    return (
        jnp.where(choose_second, mean_2, mean_1),
        jnp.where(choose_second, std_2, std_1),
        choose_second.astype(jnp.int8),
    )


def distributional_soft_bellman_target(
    *,
    key: jax.Array | None,
    reward: jax.Array,
    done: jax.Array,
    gamma: float,
    alpha: jax.Array,
    next_log_prob: jax.Array,
    target_mean_1: jax.Array,
    target_std_1: jax.Array,
    target_mean_2: jax.Array,
    target_std_2: jax.Array,
    noise: jax.Array | None = None,
    noise_clip: float = DEFAULT_NOISE_CLIP,
) -> GaussianTarget:
    """Build paired mean and sampled soft-Bellman return targets."""

    selected_mean, selected_std, branch = select_conservative_gaussian(
        target_mean_1,
        target_std_1,
        target_mean_2,
        target_std_2,
    )
    selected_sample = sample_gaussian_return(
        key,
        selected_mean,
        selected_std,
        noise=noise,
        noise_clip=noise_clip,
    )
    reward = jnp.asarray(reward, dtype=selected_mean.dtype)
    continuation = 1.0 - jnp.asarray(done, dtype=selected_mean.dtype)
    entropy = jnp.asarray(alpha, dtype=selected_mean.dtype) * jnp.asarray(
        next_log_prob, dtype=selected_mean.dtype
    )
    discount = jnp.asarray(float(gamma), dtype=selected_mean.dtype) * continuation
    return GaussianTarget(
        mean=jax.lax.stop_gradient(reward + discount * (selected_mean - entropy)),
        sample=jax.lax.stop_gradient(reward + discount * (selected_sample - entropy)),
        std=jax.lax.stop_gradient(selected_std),
        branch=branch,
    )


def ema_standard_deviation(
    running_std: jax.Array,
    batch_std: jax.Array,
    *,
    tau: float = 0.005,
    sigma_min: float = DEFAULT_SIGMA_MIN,
) -> jax.Array:
    """Update the non-network executed-transition standard-deviation scale."""

    tau = float(tau)
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    batch_mean = jnp.mean(jnp.asarray(batch_std, dtype=jnp.float32))
    updated = (1.0 - tau) * jnp.asarray(running_std, dtype=jnp.float32) + tau * batch_mean
    return jnp.maximum(updated, float(sigma_min))


def clip_sample_target(
    current_mean: jax.Array,
    sampled_target: jax.Array,
    running_std: jax.Array,
    *,
    multiplier: float = 3.0,
) -> jax.Array:
    """Clip a stochastic target to ``mean +/- multiplier * ema_std``."""

    current_mean = jnp.asarray(current_mean)
    sampled_target = jnp.asarray(sampled_target, dtype=current_mean.dtype)
    bound = jnp.asarray(float(multiplier), dtype=current_mean.dtype) * jnp.asarray(
        running_std, dtype=current_mean.dtype
    )
    return current_mean + jnp.clip(sampled_target - current_mean, -bound, bound)


def hdsac_gaussian_loss(
    mean: jax.Array,
    std: jax.Array,
    mean_target: jax.Array,
    sampled_target: jax.Array,
    running_std: jax.Array,
    *,
    stabilizer: float = 0.1,
    clip_multiplier: float = 3.0,
) -> jax.Array:
    """H-DSAC-style stop-gradient loss for one Gaussian return Critic."""

    mean = jnp.asarray(mean, dtype=jnp.float32)
    std = jnp.asarray(std, dtype=mean.dtype)
    mean_target = jnp.asarray(mean_target, dtype=mean.dtype)
    clipped_target = clip_sample_target(
        mean,
        sampled_target,
        running_std,
        multiplier=clip_multiplier,
    )
    detached_std = jax.lax.stop_gradient(std)
    detached_mean_error = jax.lax.stop_gradient(mean_target - mean)
    detached_sample_error = jax.lax.stop_gradient(mean - clipped_target)
    b = jnp.asarray(float(stabilizer), dtype=mean.dtype)
    scale = jnp.square(jnp.asarray(running_std, dtype=mean.dtype)) + b
    mean_term = -(detached_mean_error / (jnp.square(detached_std) + b)) * mean
    std_term = -(
        (jnp.square(detached_sample_error) - jnp.square(detached_std))
        / (jnp.power(detached_std, 3) + b)
    ) * std
    return jnp.mean(scale * (mean_term + std_term))


def gaussian_nll_loss(
    mean: jax.Array,
    std: jax.Array,
    target: jax.Array,
    *,
    sigma_min: float = DEFAULT_SIGMA_MIN,
) -> jax.Array:
    """Configuration ablation: conventional diagonal Gaussian NLL."""

    mean = jnp.asarray(mean, dtype=jnp.float32)
    std = jnp.maximum(jnp.asarray(std, dtype=mean.dtype), float(sigma_min))
    target = jax.lax.stop_gradient(jnp.asarray(target, dtype=mean.dtype))
    return jnp.mean(0.5 * jnp.square((target - mean) / std) + jnp.log(std))


def gaussian_wasserstein_disagreement(
    mean_1: jax.Array,
    std_1: jax.Array,
    mean_2: jax.Array,
    std_2: jax.Array,
    *,
    std_weight: float = 1.0,
) -> jax.Array:
    """One-dimensional Gaussian W2 twin disagreement."""

    return jnp.sqrt(
        jnp.square(jnp.asarray(mean_1) - jnp.asarray(mean_2))
        + float(std_weight) * jnp.square(jnp.asarray(std_1) - jnp.asarray(std_2))
    )


def normalized_bellman_surprise(
    current_mean_1: jax.Array,
    current_mean_2: jax.Array,
    current_std_1: jax.Array,
    current_std_2: jax.Array,
    frozen_mean_target: jax.Array,
    *,
    epsilon: float = 1e-6,
) -> jax.Array:
    """Frozen-offline Bellman error normalized by average predicted std."""

    mean = 0.5 * (jnp.asarray(current_mean_1) + jnp.asarray(current_mean_2))
    std = 0.5 * (jnp.asarray(current_std_1) + jnp.asarray(current_std_2))
    return jnp.abs(mean - jnp.asarray(frozen_mean_target)) / (
        jnp.maximum(std, 0.0) + float(epsilon)
    )


def calibrate_offline_support(
    disagreement: np.ndarray,
    surprise: np.ndarray,
    *,
    quantile: float = 0.95,
    minimum_temperature: float = 1e-3,
) -> OfflineSupportThresholds:
    """Fit validation-only quantiles and robust sigmoid temperatures."""

    disagreement = np.asarray(disagreement, dtype=np.float64).reshape(-1)
    surprise = np.asarray(surprise, dtype=np.float64).reshape(-1)
    if not disagreement.size or not surprise.size:
        raise ValueError("offline support calibration arrays must be non-empty")
    if not np.all(np.isfinite(disagreement)) or not np.all(np.isfinite(surprise)):
        raise ValueError("offline support calibration arrays must be finite")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")

    def mad(value: np.ndarray) -> float:
        median = float(np.median(value))
        return max(float(np.median(np.abs(value - median))), minimum_temperature)

    return OfflineSupportThresholds(
        disagreement_quantile=float(np.quantile(disagreement, quantile)),
        surprise_quantile=float(np.quantile(surprise, quantile)),
        disagreement_temperature=mad(disagreement),
        surprise_temperature=mad(surprise),
    )


def soft_reference_weight(
    disagreement: jax.Array,
    surprise: jax.Array,
    thresholds: OfflineSupportThresholds,
) -> jax.Array:
    """Continuous familiar-support weight, independent of scene identity."""

    disagreement_term = jax.nn.sigmoid(
        (thresholds.disagreement_quantile - jnp.asarray(disagreement))
        / thresholds.disagreement_temperature
    )
    surprise_term = jax.nn.sigmoid(
        (thresholds.surprise_quantile - jnp.asarray(surprise))
        / thresholds.surprise_temperature
    )
    return disagreement_term * surprise_term


def gaussian_reference_loss(
    mean: jax.Array,
    std: jax.Array,
    frozen_mean: jax.Array,
    frozen_std: jax.Array,
    weight: jax.Array,
    *,
    std_weight: float = 0.1,
    epsilon: float = 1e-6,
) -> jax.Array:
    """Weighted 1-D Gaussian W2-squared reference loss."""

    weight = jnp.asarray(weight, dtype=jnp.float32)
    distance = jnp.square(jnp.asarray(mean) - jax.lax.stop_gradient(frozen_mean))
    distance += float(std_weight) * jnp.square(
        jnp.asarray(std) - jax.lax.stop_gradient(frozen_std)
    )
    return jnp.sum(weight * distance) / (jnp.sum(weight) + float(epsilon))


__all__ = [
    "DEFAULT_NOISE_CLIP",
    "DEFAULT_SIGMA_MIN",
    "GaussianTarget",
    "OfflineSupportThresholds",
    "calibrate_offline_support",
    "clip_sample_target",
    "distributional_soft_bellman_target",
    "ema_standard_deviation",
    "gaussian_nll_loss",
    "gaussian_reference_loss",
    "gaussian_wasserstein_disagreement",
    "hdsac_gaussian_loss",
    "normalized_bellman_surprise",
    "positive_std",
    "sample_gaussian_return",
    "select_conservative_gaussian",
    "soft_reference_weight",
]
