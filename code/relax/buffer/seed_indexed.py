"""Exact ring-slot Seed ledgers for balanced long-run replay sampling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import pickle
from typing import Generic, TypeVar

import numpy as np

from relax.buffer.tree import TreeBuffer


T = TypeVar("T")


class SeedIndexedReplay(Generic[T]):
    """Wrap a TreeBuffer and keep per-Seed sets aligned across ring wrap."""

    def __init__(
        self,
        buffer: TreeBuffer[T],
        *,
        seeds: Sequence[int],
        seed: int = 0,
    ) -> None:
        values = tuple(int(value) for value in seeds)
        if not values or len(values) != len(set(values)):
            raise ValueError("Seed ledger requires non-empty unique Seeds")
        self.buffer = buffer
        self.seeds = values
        self.rng = np.random.default_rng(seed)
        self.seed_by_index = np.full(buffer.max_len, -1, dtype=np.int32)
        self.indices_by_seed = {value: set() for value in values}

    def __len__(self) -> int:
        return len(self.buffer)

    def _record_slot(self, index: int, seed: int) -> None:
        if seed not in self.indices_by_seed:
            raise ValueError(f"unbound replay Seed {seed}")
        old = int(self.seed_by_index[index])
        if old >= 0:
            self.indices_by_seed[old].discard(index)
        self.seed_by_index[index] = seed
        self.indices_by_seed[seed].add(index)

    def add(self, sample: T, *, from_jax: bool = False) -> None:
        seed = int(np.asarray(getattr(sample, "scene_seed")))
        self._record_slot(self.buffer.ptr, seed)
        self.buffer.add(sample, from_jax=from_jax)

    def add_batch(self, samples: T, *, from_jax: bool = False) -> None:
        if from_jax:
            import jax

            samples = jax.device_get(samples)
        seeds = np.asarray(getattr(samples, "scene_seed"), dtype=np.int32)
        if seeds.ndim != 1:
            raise ValueError("batched replay Seed labels must be one-dimensional")
        if seeds.size > self.buffer.max_len:
            raise ValueError("batch exceeds replay capacity")
        start = self.buffer.ptr
        for offset, seed in enumerate(seeds):
            self._record_slot((start + offset) % self.buffer.max_len, int(seed))
        self.buffer.add_batch(samples, from_jax=False)

    def sample(self, size: int, *, to_jax: bool = False) -> T:
        return self.buffer.sample(size, to_jax=to_jax)

    def _weighted_indices(
        self,
        size: int,
        weights: Mapping[int, float],
    ) -> np.ndarray:
        if size < 0:
            raise ValueError("sample size must be non-negative")
        if size == 0:
            return np.empty(0, dtype=np.int64)
        available = [seed for seed in self.seeds if self.indices_by_seed[seed]]
        if not available:
            raise ValueError("cannot sample an empty Seed-indexed replay")
        probabilities = np.asarray(
            [max(float(weights.get(seed, 0.0)), 0.0) for seed in available],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Seed sampling weights must be finite")
        if float(np.sum(probabilities)) <= 0.0:
            probabilities = np.ones(len(available), dtype=np.float64)
        probabilities /= np.sum(probabilities)
        selected_seeds = self.rng.choice(
            np.asarray(available, dtype=np.int32),
            size=size,
            replace=True,
            p=probabilities,
        )
        indices = np.empty(size, dtype=np.int64)
        for index, selected_seed in enumerate(selected_seeds):
            slots = np.fromiter(
                sorted(self.indices_by_seed[int(selected_seed)]),
                dtype=np.int64,
            )
            indices[index] = slots[self.rng.integers(0, slots.size)]
        return indices

    def sample_seed_weighted(
        self,
        size: int,
        *,
        weights: Mapping[int, float],
        to_jax: bool = False,
    ) -> T:
        return self.buffer.take(
            self._weighted_indices(size, weights),
            to_jax=to_jax,
        )

    def has_seed(self, seed: int) -> bool:
        """Return whether at least one live ring slot belongs to ``seed``."""

        seed = int(seed)
        if seed not in self.indices_by_seed:
            raise ValueError(f"unbound replay Seed {seed}")
        return bool(self.indices_by_seed[seed])

    def sample_seed(
        self,
        size: int,
        *,
        seed: int,
        to_jax: bool = False,
    ) -> T:
        """Sample only one Seed, with replacement, from live ring slots."""

        seed = int(seed)
        if size < 1:
            raise ValueError("sample size must be positive")
        if not self.has_seed(seed):
            raise ValueError(f"cannot sample absent replay Seed {seed}")
        slots = np.fromiter(
            sorted(self.indices_by_seed[seed]),
            dtype=np.int64,
        )
        indices = slots[self.rng.integers(0, slots.size, size=int(size))]
        return self.buffer.take(indices, to_jax=to_jax)

    def sample_global_seed_mix(
        self,
        size: int,
        *,
        balanced_fraction: float,
        seed_weights: Mapping[int, float],
        to_jax: bool = False,
    ) -> T:
        if not 0.0 <= balanced_fraction <= 1.0:
            raise ValueError("balanced replay fraction must be in [0, 1]")
        if size < 1 or len(self.buffer) < 1:
            raise ValueError("cannot sample a non-positive or empty replay")
        balanced = min(int(round(size * balanced_fraction)), size)
        uniform = size - balanced
        pieces = []
        if uniform:
            pieces.append(self.rng.integers(0, len(self.buffer), size=uniform))
        if balanced:
            pieces.append(self._weighted_indices(balanced, seed_weights))
        indices = np.concatenate(pieces)
        self.rng.shuffle(indices)
        return self.buffer.take(indices, to_jax=to_jax)

    def seed_counts(self) -> dict[int, int]:
        return {seed: len(self.indices_by_seed[seed]) for seed in self.seeds}

    def state_dict(self) -> dict[str, object]:
        """Return an exact resumable ring/ledger/RNG snapshot."""

        length = int(len(self.buffer))
        leaves = tuple(
            np.array(
                value if length == self.buffer.max_len else value[:length],
                copy=True,
            )
            for value in self.buffer.buffers
        )
        return {
            "format": "seed_indexed_replay_v1",
            "seeds": list(self.seeds),
            "capacity": int(self.buffer.max_len),
            "length": length,
            "pointer": int(self.buffer.ptr),
            "leaves": leaves,
            "seed_by_index": np.array(self.seed_by_index, copy=True),
            "buffer_rng_state": self.buffer.rng.bit_generator.state,
            "ledger_rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        """Restore a snapshot produced by :meth:`state_dict` exactly."""

        if payload.get("format") != "seed_indexed_replay_v1":
            raise ValueError("unsupported Seed-indexed replay checkpoint")
        if tuple(int(seed) for seed in payload["seeds"]) != self.seeds:
            raise ValueError("Seed-indexed replay checkpoint Seed mismatch")
        if int(payload["capacity"]) != int(self.buffer.max_len):
            raise ValueError("Seed-indexed replay checkpoint capacity mismatch")
        length = int(payload["length"])
        pointer = int(payload["pointer"])
        if not 0 <= length <= self.buffer.max_len:
            raise ValueError("Seed-indexed replay checkpoint length is invalid")
        if not 0 <= pointer < self.buffer.max_len:
            raise ValueError("Seed-indexed replay checkpoint pointer is invalid")
        leaves = tuple(payload["leaves"])
        if len(leaves) != len(self.buffer.buffers):
            raise ValueError("Seed-indexed replay checkpoint structure mismatch")
        expected_rows = self.buffer.max_len if length == self.buffer.max_len else length
        for source, destination in zip(leaves, self.buffer.buffers):
            source = np.asarray(source)
            if source.shape != (expected_rows, *destination.shape[1:]):
                raise ValueError("Seed-indexed replay leaf shape mismatch")
            if source.dtype != destination.dtype:
                raise ValueError("Seed-indexed replay leaf dtype mismatch")
            destination[:expected_rows] = source
        seed_by_index = np.asarray(payload["seed_by_index"], dtype=np.int32)
        if seed_by_index.shape != (self.buffer.max_len,):
            raise ValueError("Seed-indexed replay ledger shape mismatch")
        if np.any(~np.isin(seed_by_index[:length], np.asarray(self.seeds, np.int32))):
            raise ValueError("Seed-indexed replay ledger contains an unbound Seed")
        if length < self.buffer.max_len and np.any(seed_by_index[length:] != -1):
            raise ValueError("Seed-indexed replay has live ledger rows after length")
        self.buffer.len = length
        self.buffer.ptr = pointer
        self.seed_by_index = np.array(seed_by_index, copy=True)
        self.indices_by_seed = {
            seed: set(np.flatnonzero(self.seed_by_index == seed).astype(int).tolist())
            for seed in self.seeds
        }
        if sum(len(indices) for indices in self.indices_by_seed.values()) != length:
            raise ValueError("Seed-indexed replay ledger count mismatch")
        self.buffer.rng.bit_generator.state = payload["buffer_rng_state"]
        self.rng.bit_generator.state = payload["ledger_rng_state"]

    def save(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(self.state_dict(), stream)
        os.replace(temporary, target)

    def load(self, path: Path) -> None:
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        self.load_state_dict(payload)


__all__ = ["SeedIndexedReplay"]
