from pathlib import Path
import pickle
from typing import Any, Callable, Tuple, TypeVar

import numpy as np
import jax, jax.tree_util as tree
from jax import ShapeDtypeStruct

from relax.buffer.base import Buffer
from relax.utils.experience import (
    AEFLACTrajExperience,
    Experience,
    GuardianCalQLExperience,
    GuardianDSACExperience,
    GuardianFlowDSACExperience,
    GuardianFlowDSACLongRunExperience,
    JaxSafeSACExperience,
    MGFDriveExperience,
    MGFToyExperience,
)

T = TypeVar("T")
S = TypeVar("S")
ShapeDtypeStructTree = (
    Any  # A tree of same structure as T, but with ShapeDtypeStruct leaves
)


class TreeBuffer(Buffer[T]):
    def __init__(self, spec: ShapeDtypeStructTree, size: int, seed: int = 0) -> None:
        def create_buffer(sd: ShapeDtypeStruct):
            shape, dtype = (size, *sd.shape), sd.dtype
            return np.empty(shape, dtype=dtype)

        leaves, treedef = tree.tree_flatten(spec)
        self.buffers: Tuple[np.ndarray] = tuple(create_buffer(sd) for sd in leaves)
        self.treedef = treedef

        self.rng = np.random.default_rng(seed)

        self.max_len = size
        self.len = 0
        self.ptr = 0

    def add(self, sample: T, *, from_jax: bool = False) -> None:
        if from_jax:
            sample = jax.device_get(sample)
        leaves = self.treedef.flatten_up_to(sample)
        for leaf, buf in zip(leaves, self.buffers):
            buf[self.ptr] = leaf
        self._advance()

    def add_batch(self, samples: T, *, from_jax: bool = False) -> None:
        if from_jax:
            samples = jax.device_get(samples)
        leaves = self.treedef.flatten_up_to(samples)
        batch_size = leaves[0].shape[0]
        assert batch_size <= self.max_len and all(
            leaf.shape[0] == batch_size for leaf in leaves
        )

        start, end = self.ptr, self.ptr + batch_size
        if end > self.max_len:
            # Need to wrap around
            split, remain = self.max_len - start, end - self.max_len
            for leaf, buf in zip(leaves, self.buffers):
                buf[start:] = leaf[:split]
                buf[:remain] = leaf[split:]
        else:
            for leaf, buf in zip(leaves, self.buffers):
                buf[start:end] = leaf

        self._advance(batch_size)

    def sample(self, size: int, *, to_jax: bool = False) -> T:
        return self.sample_with_indices(size, to_jax=to_jax)[0]

    def sample_with_indices(
        self, size: int, *, to_jax: bool = False
    ) -> Tuple[T, np.ndarray]:
        indices = self.rng.integers(0, self.len, size=size)
        return self.take(indices, to_jax=to_jax), indices

    def take(self, indices: np.ndarray, *, to_jax: bool = False) -> T:
        """Return exact valid slots without changing replay RNG state."""

        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("replay indices must be one-dimensional")
        if np.any(indices < 0) or np.any(indices >= self.len):
            raise IndexError("replay index is outside the valid range")
        leaves = tuple(np.take(buf, indices, axis=0) for buf in self.buffers)
        samples = tree.tree_unflatten(self.treedef, leaves)
        if to_jax:
            samples = jax.device_put(samples)
        return samples

    def replace(
        self, indices: np.ndarray, samples: T, *, from_jax: bool = False
    ) -> None:
        if from_jax:
            samples = jax.device_get(samples)
        leaves = self.treedef.flatten_up_to(samples)
        for leaf, buf in zip(leaves, self.buffers):
            if leaf is None:
                continue  # Skip replacing if leaf is None
            buf[indices] = leaf

    def save(self, path: Path) -> None:
        if self.len < self.max_len:
            # Only save the valid part of the buffer
            leaves = tuple(buf[: self.len] for buf in self.buffers)
        else:
            leaves = self.buffers
        with path.open("wb") as f:
            pickle.dump(leaves, f)

    def load(self, path: Path) -> None:
        """Restore leaves written by :meth:`save` into this buffer."""

        with path.open("rb") as f:
            leaves = pickle.load(f)
        if len(leaves) != len(self.buffers):
            raise ValueError("replay buffer structure does not match checkpoint")
        lengths = {int(np.asarray(leaf).shape[0]) for leaf in leaves}
        if len(lengths) != 1:
            raise ValueError("checkpoint replay leaves have inconsistent lengths")
        length = lengths.pop()
        if length > self.max_len:
            raise ValueError(
                f"checkpoint has {length} samples but capacity is {self.max_len}"
            )
        for source, destination in zip(leaves, self.buffers):
            source = np.asarray(source)
            if source.shape[1:] != destination.shape[1:]:
                raise ValueError("replay leaf shape does not match checkpoint")
            destination[:length] = source
        self.len = length
        self.ptr = length % self.max_len

    def _advance(self, size: int = 1):
        self.len = min(self.len + size, self.max_len)
        self.ptr = (self.ptr + size) % self.max_len

    def __len__(self):
        return self.len

    def __repr__(self):
        return f"TreeBuffer(size={self.max_len}, len={self.len}, ptr={self.ptr}, treedef={self.treedef})"

    @staticmethod
    def from_example(
        example: T, size: int, seed: int = 0, remove_batch_dim: bool = True
    ) -> "TreeBuffer[T]":
        def create_shape_dtype_struct(x: np.ndarray):
            if remove_batch_dim:
                assert x.ndim >= 1
                return ShapeDtypeStruct(x.shape[1:], x.dtype)
            else:
                return ShapeDtypeStruct(x.shape, x.dtype)

        spec = tree.tree_map(create_shape_dtype_struct, example)
        return TreeBuffer(spec, size, seed)

    @staticmethod
    def from_experience(
        obs_dim: int, act_dim: int, size: int, seed: int = 0
    ) -> "TreeBuffer[Experience]":
        example = Experience.create_example(obs_dim, act_dim)
        return TreeBuffer.from_example(example, size, seed, remove_batch_dim=False)

    @staticmethod
    def from_ae_flac_traj_experience(
        obs_dim: int, act_dim: int, size: int, seed: int = 0
    ) -> "TreeBuffer[AEFLACTrajExperience]":
        example = AEFLACTrajExperience.create_example(obs_dim, act_dim)
        return TreeBuffer.from_example(example, size, seed, remove_batch_dim=False)

    @staticmethod
    def from_mgf_toy_experience(
        obs_dim: int, act_dim: int, size: int, seed: int = 0
    ) -> "TreeBuffer[MGFToyExperience]":
        example = MGFToyExperience.create_example(obs_dim, act_dim)
        return TreeBuffer.from_example(example, size, seed, remove_batch_dim=False)

    @staticmethod
    def from_mgf_drive_experience(
        obs_dim: int, act_dim: int, size: int, seed: int = 0
    ) -> "TreeBuffer[MGFDriveExperience]":
        example = MGFDriveExperience.create_example(obs_dim, act_dim)
        return TreeBuffer.from_example(example, size, seed, remove_batch_dim=False)

    @staticmethod
    def from_jax_safe_sac_experience(
        obs_dim: int, act_dim: int, size: int, seed: int = 0
    ) -> "TreeBuffer[JaxSafeSACExperience]":
        example = JaxSafeSACExperience.create_example(obs_dim, act_dim)
        return TreeBuffer.from_example(example, size, seed, remove_batch_dim=False)

    @staticmethod
    def from_guardian_calql_experience(
        obs_dim: int, act_dim: int, size: int, seed: int = 0
    ) -> "TreeBuffer[GuardianCalQLExperience]":
        example = GuardianCalQLExperience.create_example(obs_dim, act_dim)
        return TreeBuffer.from_example(example, size, seed, remove_batch_dim=False)

    @staticmethod
    def from_guardian_dsac_experience(
        obs_dim: int, act_dim: int, size: int, seed: int = 0
    ) -> "TreeBuffer[GuardianDSACExperience]":
        example = GuardianDSACExperience.create_example(obs_dim, act_dim)
        return TreeBuffer.from_example(example, size, seed, remove_batch_dim=False)

    @staticmethod
    def from_guardian_flow_dsac_experience(
        obs_dim: int, act_dim: int, size: int, seed: int = 0
    ) -> "TreeBuffer[GuardianFlowDSACExperience]":
        example = GuardianFlowDSACExperience.create_example(obs_dim, act_dim)
        return TreeBuffer.from_example(example, size, seed, remove_batch_dim=False)

    @staticmethod
    def from_guardian_flow_dsac_long_run_experience(
        obs_dim: int, act_dim: int, size: int, seed: int = 0
    ) -> "TreeBuffer[GuardianFlowDSACLongRunExperience]":
        example = GuardianFlowDSACLongRunExperience.create_example(
            obs_dim, act_dim
        )
        return TreeBuffer.from_example(example, size, seed, remove_batch_dim=False)

    @staticmethod
    def connect(
        src: "TreeBuffer[S]", dst: "TreeBuffer[T]", converter: Callable[[S], T]
    ):
        original_add = src.add
        original_add_batch = src.add_batch

        def add(self, experience: Experience):
            original_add(experience)
            dst.add(converter(experience))

        def add_batch(self, experiences: Experience):
            original_add_batch(experiences)
            dst.add_batch(converter(experiences))

        import types

        src.add = types.MethodType(add, src)
        src.add_batch = types.MethodType(add_batch, src)
