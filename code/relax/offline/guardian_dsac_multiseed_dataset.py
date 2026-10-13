"""Uniform multi-Seed adapter for Guardian-DSAC V2.3.1 datasets.

The underlying :class:`GuardianDSACMetaSafeDataset` intentionally remains a
single-scene loader.  This module only coordinates several already validated
single-Seed datasets and never rewrites their replay representation.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Iterator, Sequence

import numpy as np

from relax.offline.metasafe_guardian_dataset import (
    GuardianDSACMetaSafeDataset,
)
from relax.utils.action_semantics import (
    TARGET_SPEED_LATERAL_ACTION_SEMANTICS,
)
from relax.utils.experience import (
    GuardianDSACExperience,
    GuardianFlowDSACExperience,
    GuardianFlowDSACLongRunExperience,
)


def guardian_dsac_to_flow_experience(
    batch: GuardianDSACExperience,
    *,
    flow_noise: np.ndarray | None = None,
    flow_noise_valid: np.ndarray | None = None,
) -> GuardianFlowDSACExperience:
    """Add V2.4-only latent fields without changing the V2.3 replay type.

    Offline rows have no causal Flow proposal, so their latent is zero and
    invalid by default.  Online orchestration supplies the actual sampled
    latent and a true validity flag instead.
    """

    action = np.asarray(batch.action)
    reward = np.asarray(batch.reward)
    noise = (
        np.zeros_like(action, dtype=np.float32)
        if flow_noise is None
        else np.asarray(flow_noise, dtype=np.float32)
    )
    valid = (
        np.zeros_like(reward, dtype=np.bool_)
        if flow_noise_valid is None
        else np.asarray(flow_noise_valid, dtype=np.bool_)
    )
    if noise.shape != action.shape:
        raise ValueError("Flow latent shape must match the action shape")
    if valid.shape != reward.shape:
        raise ValueError("Flow latent-valid shape must match the reward shape")
    return GuardianFlowDSACExperience(
        **batch._asdict(),
        flow_noise=noise,
        flow_noise_valid=valid,
    )


def guardian_dsac_to_flow_long_run_experience(
    batch: GuardianDSACExperience,
    *,
    flow_noise: np.ndarray | None = None,
    flow_noise_valid: np.ndarray | None = None,
    flow_zero_latent: np.ndarray | None = None,
) -> GuardianFlowDSACLongRunExperience:
    """Add V2.4.1 causal latent fields without mutating older schemas."""

    base = guardian_dsac_to_flow_experience(
        batch,
        flow_noise=flow_noise,
        flow_noise_valid=flow_noise_valid,
    )
    reward = np.asarray(batch.reward)
    zero_latent = (
        np.zeros_like(reward, dtype=np.bool_)
        if flow_zero_latent is None
        else np.asarray(flow_zero_latent, dtype=np.bool_)
    )
    if zero_latent.shape != reward.shape:
        raise ValueError("Flow zero-latent shape must match the reward shape")
    if np.any(zero_latent & ~np.asarray(base.flow_noise_valid, dtype=np.bool_)):
        raise ValueError("zero-latent rows must have a valid causal Flow latent")
    if np.any(
        zero_latent
        & np.any(np.asarray(base.flow_noise, dtype=np.float32) != 0.0, axis=-1)
    ):
        raise ValueError("zero-latent rows must store an exact all-zero latent")
    return GuardianFlowDSACLongRunExperience(
        **base._asdict(),
        flow_zero_latent=zero_latent,
    )


def _combined_sha256(datasets: Sequence[GuardianDSACMetaSafeDataset]) -> str:
    binding = [
        {
            "scene_seed": int(dataset.scene_seed),
            "scene_signature": str(dataset.scene_signature),
            "sha256": str(dataset.sha256),
        }
        for dataset in datasets
    ]
    payload = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _concatenate(
    batches: Sequence[GuardianDSACExperience],
    *,
    permutation: np.ndarray | None = None,
) -> GuardianDSACExperience:
    if not batches:
        raise ValueError("cannot concatenate an empty batch sequence")
    values = {}
    for name in batches[0]._fields:
        field = np.concatenate(
            [np.asarray(getattr(batch, name)) for batch in batches], axis=0
        )
        values[name] = field if permutation is None else field[permutation]
    return GuardianDSACExperience(**values)


class GuardianDSACMultiSeedDataset:
    """Lightweight uniform sampler over validated single-Seed archives.

    Remainder rows are assigned in ascending Seed order.  Consequently a
    batch of 512 over Seed0--4 contains ``103, 103, 102, 102, 102`` rows.
    """

    def __init__(
        self,
        datasets: Sequence[GuardianDSACMetaSafeDataset],
        *,
        seed: int = 0,
        expected_seeds: Sequence[int] | None = None,
    ) -> None:
        if not datasets:
            raise ValueError("multi-Seed dataset requires at least one dataset")
        ordered = tuple(sorted(datasets, key=lambda value: int(value.scene_seed)))
        seeds = tuple(int(dataset.scene_seed) for dataset in ordered)
        if len(seeds) != len(set(seeds)):
            raise ValueError("multi-Seed dataset contains duplicate scene seeds")
        if expected_seeds is not None:
            expected = tuple(sorted(int(value) for value in expected_seeds))
            if seeds != expected:
                raise ValueError(
                    f"multi-Seed dataset seeds {seeds} do not match {expected}"
                )
        obs_dims = {int(dataset.summary.obs_dim) for dataset in ordered}
        action_dims = {int(dataset.summary.action_dim) for dataset in ordered}
        semantics = {str(dataset.action_semantics_version) for dataset in ordered}
        if len(obs_dims) != 1 or len(action_dims) != 1:
            raise ValueError("single-Seed datasets have incompatible dimensions")
        if semantics != {TARGET_SPEED_LATERAL_ACTION_SEMANTICS}:
            raise ValueError("multi-Seed dataset requires target_speed_lateral_v1")

        self.datasets = ordered
        self.by_seed = {int(dataset.scene_seed): dataset for dataset in self.datasets}
        self.scene_seeds = seeds
        self.scene_signatures = {
            str(dataset.scene_seed): str(dataset.scene_signature)
            for dataset in self.datasets
        }
        self.obs_dim = obs_dims.pop()
        self.action_dim = action_dims.pop()
        self.sha256 = _combined_sha256(self.datasets)
        self.rng = np.random.default_rng(int(seed) + 51_007)

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def batch_allocation(self, batch_size: int) -> dict[int, int]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        base, remainder = divmod(int(batch_size), len(self.datasets))
        return {
            seed: base + int(index < remainder)
            for index, seed in enumerate(self.scene_seeds)
        }

    def sample(
        self,
        batch_size: int,
        *,
        split: str = "train",
        pair_fraction: float = 0.0,
        teacher_fraction: float = 0.0,
        to_jax: bool = True,
    ) -> GuardianDSACExperience:
        allocation = self.batch_allocation(batch_size)
        batches = [
            self.by_seed[seed].sample(
                count,
                split=split,
                pair_fraction=pair_fraction,
                teacher_fraction=teacher_fraction,
                to_jax=False,
            )
            for seed, count in allocation.items()
            if count
        ]
        permutation = self.rng.permutation(batch_size)
        result = _concatenate(batches, permutation=permutation)
        if to_jax:
            import jax

            result = jax.device_put(result)
        return result

    def sample_flow(
        self,
        batch_size: int,
        *,
        split: str = "train",
        pair_fraction: float = 0.0,
        teacher_fraction: float = 0.0,
        to_jax: bool = True,
    ) -> GuardianFlowDSACExperience:
        """Sample offline V2.4 rows with explicitly invalid zero latents."""

        result = guardian_dsac_to_flow_experience(
            self.sample(
                batch_size,
                split=split,
                pair_fraction=pair_fraction,
                teacher_fraction=teacher_fraction,
                to_jax=False,
            )
        )
        if to_jax:
            import jax

            result = jax.device_put(result)
        return result

    def sample_flow_long_run(
        self,
        batch_size: int,
        *,
        split: str = "train",
        pair_fraction: float = 0.0,
        teacher_fraction: float = 0.0,
        to_jax: bool = True,
    ) -> GuardianFlowDSACLongRunExperience:
        """Sample V2.4.1 offline rows with invalid, non-causal latents."""

        result = guardian_dsac_to_flow_long_run_experience(
            self.sample(
                batch_size,
                split=split,
                pair_fraction=pair_fraction,
                teacher_fraction=teacher_fraction,
                to_jax=False,
            )
        )
        if to_jax:
            import jax

            result = jax.device_put(result)
        return result

    def iter_validation(
        self,
        *,
        batch_size: int = 4096,
        to_jax: bool = True,
    ) -> Iterator[tuple[int, GuardianDSACExperience]]:
        """Yield validation batches without mixing Seed identities."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for seed in self.scene_seeds:
            for batch in self.by_seed[seed].iter_split(
                "validation", batch_size=batch_size, to_jax=to_jax
            ):
                yield seed, batch

    def validation_reference_batch(
        self,
        *,
        rows_per_seed: int | None = None,
        to_jax: bool = True,
    ) -> GuardianDSACExperience:
        """Build a deterministic, uniformly capped support-calibration batch."""

        if rows_per_seed is not None and rows_per_seed < 1:
            raise ValueError("rows_per_seed must be positive when supplied")
        batches = []
        for seed in self.scene_seeds:
            dataset = self.by_seed[seed]
            indices = dataset.split_transition_indices("validation")
            if rows_per_seed is not None and indices.size > rows_per_seed:
                # Evenly spaced rows cover complete validation trajectories
                # without introducing a second random sampling contract.
                positions = np.linspace(
                    0, indices.size - 1, rows_per_seed, dtype=np.int64
                )
                indices = indices[positions]
            batches.append(dataset.experience(indices))
        result = _concatenate(batches)
        if to_jax:
            import jax

            result = jax.device_put(result)
        return result

    def flow_validation_reference_batch(
        self,
        *,
        rows_per_seed: int | None = None,
        to_jax: bool = True,
    ) -> GuardianFlowDSACExperience:
        result = guardian_dsac_to_flow_experience(
            self.validation_reference_batch(
                rows_per_seed=rows_per_seed,
                to_jax=False,
            )
        )
        if to_jax:
            import jax

            result = jax.device_put(result)
        return result

    def flow_long_run_validation_reference_batch(
        self,
        *,
        rows_per_seed: int | None = None,
        to_jax: bool = True,
    ) -> GuardianFlowDSACLongRunExperience:
        result = guardian_dsac_to_flow_long_run_experience(
            self.validation_reference_batch(
                rows_per_seed=rows_per_seed,
                to_jax=False,
            )
        )
        if to_jax:
            import jax

            result = jax.device_put(result)
        return result

    def speed_coverage(self) -> dict[str, object]:
        per_seed = {
            str(seed): self.by_seed[seed].speed_coverage() for seed in self.scene_seeds
        }
        counts = np.sum(
            [
                np.asarray(value["counts"], dtype=np.int64)
                for value in per_seed.values()
            ],
            axis=0,
        )
        total = max(int(np.sum(counts)), 1)
        return {
            "counts": counts.astype(int).tolist(),
            "fractions": (counts / total).astype(float).tolist(),
            "all_bins_visited": bool(np.all(counts > 0)),
            "highest_bin_fraction": float(counts[3] / total),
            "emergency_stop_count": int(
                sum(int(value["emergency_stop_count"]) for value in per_seed.values())
            ),
            "per_seed": per_seed,
        }

    def summary(self) -> dict[str, object]:
        return {
            "format": "guardian_dsac_multiseed_dataset_v1",
            "scene_seeds": list(self.scene_seeds),
            "scene_signatures": dict(self.scene_signatures),
            "dataset_sha256": self.sha256,
            "dataset_count": len(self.datasets),
            "episodes": sum(dataset.summary.episodes for dataset in self.datasets),
            "transitions": sum(
                dataset.summary.transitions for dataset in self.datasets
            ),
            "decision_transitions": len(self),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "speed_coverage": self.speed_coverage(),
            "per_seed": {
                str(seed): asdict(self.by_seed[seed].summary)
                for seed in self.scene_seeds
            },
        }


__all__ = [
    "GuardianDSACMultiSeedDataset",
    "guardian_dsac_to_flow_experience",
    "guardian_dsac_to_flow_long_run_experience",
]
