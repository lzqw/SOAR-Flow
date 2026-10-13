"""Bound Safe-Memory utilities for Guardian-Flow-DSAC V2.4.3.

The ordinary trajectory memory and Seed9 pre-intervention replay are separate
from the task replay on purpose.  Neither structure is a Bellman source: both
contain supervised ``z=0`` Actor targets only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np


SAFE_MEMORY_BATCH_SIZE = 128
PRE_INTERVENTION_BATCH_SIZE = 128
PRE_INTERVENTION_CAPACITY = 100_000
PRE_INTERVENTION_HORIZON = 15
PRE_INTERVENTION_TAU = 5.0


@dataclass(frozen=True)
class SupervisionBatch:
    """Shape-stable supervised Actor batch held outside task replay."""

    obs: np.ndarray
    action: np.ndarray
    weight: np.ndarray
    scene_seed: np.ndarray
    lag: np.ndarray

    def __len__(self) -> int:
        return int(self.obs.shape[0])


def array_sha256(value: np.ndarray) -> str:
    """Hash an array with an explicit dtype/shape binding."""

    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def safe_memory_sha256(memories: Mapping[int, Mapping[str, object]]) -> str:
    """Return a stable digest of all active certified trajectory rows."""

    digest = hashlib.sha256()
    for seed in sorted(int(value) for value in memories):
        memory = memories[seed]
        binding = {
            "seed": seed,
            "metadata": {
                key: value
                for key, value in memory.items()
                if key not in ("observations", "actions")
            },
            "observation_sha256": array_sha256(memory["observations"]),
            "action_sha256": array_sha256(memory["actions"]),
        }
        digest.update(
            json.dumps(binding, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def balanced_safe_memory_sample(
    memories: Mapping[int, Mapping[str, object]],
    batch_size: int,
    rng: np.random.Generator,
) -> SupervisionBatch:
    """Sample every active Seed with counts differing by at most one.

    Sampling is with replacement inside each certified trajectory.  The final
    row order is shuffled, but the per-Seed quota is exact and auditable.
    """

    if int(batch_size) < 1:
        raise ValueError("safe-memory batch size must be positive")
    seeds = tuple(sorted(int(seed) for seed in memories))
    if not seeds:
        raise ValueError("cannot sample an empty safe trajectory memory")
    quotient, remainder = divmod(int(batch_size), len(seeds))
    extra = set(
        int(value)
        for value in rng.permutation(np.asarray(seeds, dtype=np.int32))[:remainder]
    )
    observations = []
    actions = []
    sampled_seeds = []
    for seed in seeds:
        count = quotient + int(seed in extra)
        if count == 0:
            continue
        memory = memories[seed]
        obs = np.asarray(memory["observations"], dtype=np.float32)
        action = np.asarray(memory["actions"], dtype=np.float32)
        if obs.ndim != 2 or action.ndim != 2 or obs.shape[0] != action.shape[0]:
            raise ValueError(f"Seed{seed} safe-memory array shape mismatch")
        if obs.shape[0] < 1:
            raise ValueError(f"Seed{seed} safe-memory trajectory is empty")
        indices = rng.integers(0, obs.shape[0], size=count)
        observations.append(obs[indices])
        actions.append(action[indices])
        sampled_seeds.append(np.full(count, seed, dtype=np.int32))
    result = SupervisionBatch(
        obs=np.concatenate(observations, axis=0),
        action=np.concatenate(actions, axis=0),
        weight=np.ones(int(batch_size), dtype=np.float32),
        scene_seed=np.concatenate(sampled_seeds, axis=0),
        lag=np.full(int(batch_size), -1, dtype=np.int16),
    )
    permutation = rng.permutation(int(batch_size))
    return SupervisionBatch(
        **{
            name: np.asarray(getattr(result, name))[permutation]
            for name in result.__dataclass_fields__
        }
    )


def seed9_pre_intervention_batch(
    episode: Sequence[object],
    *,
    horizon: int = PRE_INTERVENTION_HORIZON,
    tau: float = PRE_INTERVENTION_TAU,
) -> SupervisionBatch | None:
    """Extract deduplicated teacher targets before Seed9 interventions.

    For overlapping neighborhoods, a historical row is stored once with the
    nearest-intervention lag (and therefore the largest exponential weight).
    """

    if int(horizon) < 0:
        raise ValueError("pre-intervention horizon must be non-negative")
    if not np.isfinite(tau) or float(tau) <= 0.0:
        raise ValueError("pre-intervention tau must be finite and positive")
    if not episode:
        return None
    seeds = [int(np.asarray(item.scene_seed)) for item in episode]
    if any(seed != 9 for seed in seeds):
        raise ValueError("pre-intervention replay accepts only Seed9 episodes")
    nearest_lag: dict[int, int] = {}
    for index, item in enumerate(episode):
        intervention = bool(np.asarray(item.actual_control_modified)) and (
            bool(np.asarray(item.intervention_hard))
            or bool(np.asarray(item.intervention_soft))
        )
        if not intervention:
            continue
        for lag in range(int(horizon) + 1):
            historical = index - lag
            if historical < 0:
                break
            source = episode[historical]
            if not bool(np.asarray(source.teacher_valid)):
                continue
            nearest_lag[historical] = min(
                lag, nearest_lag.get(historical, int(horizon) + 1)
            )
    if not nearest_lag:
        return None
    indices = np.asarray(sorted(nearest_lag), dtype=np.int32)
    lags = np.asarray([nearest_lag[int(index)] for index in indices], dtype=np.int16)
    observations = np.stack(
        [np.asarray(episode[int(index)].obs, dtype=np.float32) for index in indices]
    )
    actions = np.stack(
        [
            np.asarray(episode[int(index)].teacher_action, dtype=np.float32)
            for index in indices
        ]
    )
    weights = np.exp(-lags.astype(np.float32) / float(tau)).astype(np.float32)
    if not (
        np.all(np.isfinite(observations))
        and np.all(np.isfinite(actions))
        and np.all(np.isfinite(weights))
    ):
        raise ValueError("pre-intervention supervision contains non-finite values")
    return SupervisionBatch(
        obs=observations,
        action=actions,
        weight=weights,
        scene_seed=np.full(len(indices), 9, dtype=np.int32),
        lag=lags,
    )


class PreInterventionReplay:
    """Fixed-capacity ring for Seed9 pre-intervention supervision."""

    FORMAT = "guardian_flow_v2_4_3_pre_intervention_replay_v1"

    def __init__(self, obs_dim: int, action_dim: int, capacity: int):
        if min(int(obs_dim), int(action_dim), int(capacity)) < 1:
            raise ValueError("pre-intervention replay dimensions must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.action = np.zeros(
            (self.capacity, self.action_dim), dtype=np.float32
        )
        self.weight = np.zeros(self.capacity, dtype=np.float32)
        self.lag = np.zeros(self.capacity, dtype=np.int16)
        self.position = 0
        self.size = 0
        self.total_added = 0

    def __len__(self) -> int:
        return int(self.size)

    def add_batch(self, batch: SupervisionBatch | None) -> int:
        if batch is None:
            return 0
        observations = np.asarray(batch.obs, dtype=np.float32)
        actions = np.asarray(batch.action, dtype=np.float32)
        weights = np.asarray(batch.weight, dtype=np.float32)
        lags = np.asarray(batch.lag, dtype=np.int16)
        seeds = np.asarray(batch.scene_seed, dtype=np.int32)
        count = observations.shape[0]
        if (
            observations.shape != (count, self.obs_dim)
            or actions.shape != (count, self.action_dim)
            or weights.shape != (count,)
            or lags.shape != (count,)
            or seeds.shape != (count,)
        ):
            raise ValueError("pre-intervention replay batch shape mismatch")
        if not np.all(seeds == 9):
            raise ValueError("pre-intervention replay is bound to Seed9")
        if not (
            np.all(np.isfinite(observations))
            and np.all(np.isfinite(actions))
            and np.all(np.isfinite(weights))
            and np.all(weights > 0.0)
            and np.all(weights <= 1.0)
            and np.all(lags >= 0)
            and np.all(lags <= PRE_INTERVENTION_HORIZON)
        ):
            raise ValueError("invalid pre-intervention replay values")
        for index in range(count):
            slot = self.position
            self.obs[slot] = observations[index]
            self.action[slot] = actions[index]
            self.weight[slot] = weights[index]
            self.lag[slot] = lags[index]
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        self.total_added += count
        return int(count)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> SupervisionBatch:
        if self.size < 1:
            raise ValueError("cannot sample an empty pre-intervention replay")
        indices = rng.integers(0, self.size, size=int(batch_size))
        return SupervisionBatch(
            obs=self.obs[indices].copy(),
            action=self.action[indices].copy(),
            weight=self.weight[indices].copy(),
            scene_seed=np.full(int(batch_size), 9, dtype=np.int32),
            lag=self.lag[indices].copy(),
        )

    def state_dict(self) -> dict[str, object]:
        active = slice(0, self.size)
        return {
            "format": self.FORMAT,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "capacity": self.capacity,
            "position": self.position,
            "size": self.size,
            "total_added": self.total_added,
            "obs": self.obs[active].copy(),
            "action": self.action[active].copy(),
            "weight": self.weight[active].copy(),
            "lag": self.lag[active].copy(),
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if payload.get("format") != self.FORMAT:
            raise ValueError("invalid pre-intervention replay format")
        if (
            int(payload["obs_dim"]) != self.obs_dim
            or int(payload["action_dim"]) != self.action_dim
            or int(payload["capacity"]) != self.capacity
        ):
            raise ValueError("pre-intervention replay architecture mismatch")
        size = int(payload["size"])
        position = int(payload["position"])
        total_added = int(payload["total_added"])
        if not (0 <= size <= self.capacity and 0 <= position < self.capacity):
            raise ValueError("invalid pre-intervention replay counters")
        if (
            total_added < size
            or size != min(total_added, self.capacity)
            or position != total_added % self.capacity
        ):
            raise ValueError("invalid pre-intervention total-added counter")
        batch = SupervisionBatch(
            obs=np.asarray(payload["obs"], dtype=np.float32),
            action=np.asarray(payload["action"], dtype=np.float32),
            weight=np.asarray(payload["weight"], dtype=np.float32),
            scene_seed=np.full(size, 9, dtype=np.int32),
            lag=np.asarray(payload["lag"], dtype=np.int16),
        )
        if len(batch) != size:
            raise ValueError("pre-intervention replay size mismatch")
        if not np.allclose(
            batch.weight,
            np.exp(-batch.lag.astype(np.float32) / PRE_INTERVENTION_TAU),
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError("pre-intervention replay weight/lag mismatch")
        self.obs.fill(0.0)
        self.action.fill(0.0)
        self.weight.fill(0.0)
        self.lag.fill(0)
        if size:
            # Validate through the normal insertion path, then restore the exact
            # persisted ring counters (active rows are canonicalized at 0:size).
            self.position = 0
            self.size = 0
            self.total_added = 0
            self.add_batch(batch)
        self.position = position
        self.size = size
        self.total_added = total_added


__all__ = [
    "PRE_INTERVENTION_BATCH_SIZE",
    "PRE_INTERVENTION_CAPACITY",
    "PRE_INTERVENTION_HORIZON",
    "PRE_INTERVENTION_TAU",
    "SAFE_MEMORY_BATCH_SIZE",
    "PreInterventionReplay",
    "SupervisionBatch",
    "array_sha256",
    "balanced_safe_memory_sample",
    "safe_memory_sha256",
    "seed9_pre_intervention_batch",
]
