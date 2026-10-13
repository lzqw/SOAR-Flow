"""Immutable benchmark and scheduling contracts for the paper experiments.

The paper benchmark intentionally replaces scene Seed9 with the predeclared
Seed10.  Nothing in this module searches candidate scenes or adapts weights to
observed performance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from relax.utils.flow_long_run import TransitionDeficitScheduler


PAPER_SCENE_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10)
OFFLINE_SEEDS = (0, 1, 2, 3, 4)
NOVEL_SEEDS = (5, 6, 7, 8, 10)
ALGORITHM_SEEDS = (0, 1, 2)

TRANSITION_TARGET_WEIGHTS = {
    **{seed: 0.04 for seed in OFFLINE_SEEDS},
    **{seed: 0.16 for seed in NOVEL_SEEDS},
}
CORRECTION_SEED_WEIGHTS = {
    seed: 1.0 / len(PAPER_SCENE_SEEDS) for seed in PAPER_SCENE_SEEDS
}
ZERO_LATENT_PROBABILITY = 0.50


def validate_paper_seed_contract(
    scene_seeds: Sequence[int],
    *,
    offline_seeds: Sequence[int] = OFFLINE_SEEDS,
    novel_seeds: Sequence[int] = NOVEL_SEEDS,
) -> None:
    """Reject any accidental return to the historical Seed0--9 benchmark."""

    scenes = tuple(int(seed) for seed in scene_seeds)
    offline = tuple(int(seed) for seed in offline_seeds)
    novel = tuple(int(seed) for seed in novel_seeds)
    if scenes != PAPER_SCENE_SEEDS:
        raise ValueError("paper benchmark must be exactly Seeds 0,1,2,3,4,5,6,7,8,10")
    if offline != OFFLINE_SEEDS or novel != NOVEL_SEEDS:
        raise ValueError("paper offline/novel Seed roles do not match the protocol")
    if set(offline).intersection(novel) or set(offline).union(novel) != set(scenes):
        raise ValueError("paper Seed roles must partition the benchmark")
    if 9 in scenes:
        raise ValueError("Seed9 is forbidden in the paper benchmark")


def validate_probability_weights(
    weights: Mapping[int, float],
    *,
    expected_seeds: Sequence[int] = PAPER_SCENE_SEEDS,
) -> None:
    expected = tuple(int(seed) for seed in expected_seeds)
    observed = {int(seed): float(value) for seed, value in weights.items()}
    if set(observed) != set(expected):
        raise ValueError("paper scheduling weights do not cover the benchmark")
    values = np.asarray([observed[seed] for seed in expected], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("paper scheduling weights must be finite and positive")
    if not np.isclose(float(np.sum(values)), 1.0, atol=1e-12):
        raise ValueError("paper scheduling weights must sum to one")


class PaperTransitionDeficitScheduler(TransitionDeficitScheduler):
    """80/20 novel/retention transition scheduler with no hard-Seed bias."""

    def __init__(self) -> None:
        validate_probability_weights(TRANSITION_TARGET_WEIGHTS)
        super().__init__(
            TRANSITION_TARGET_WEIGHTS,
            tie_break_order=PAPER_SCENE_SEEDS,
        )


class UniformEpisodeLatentPolicy:
    """Stratify every scene to the same 50% exact-zero episode rate."""

    def __init__(
        self,
        seeds: Sequence[int] = PAPER_SCENE_SEEDS,
        *,
        zero_probability: float = ZERO_LATENT_PROBABILITY,
    ) -> None:
        self.seeds = tuple(int(seed) for seed in seeds)
        validate_paper_seed_contract(self.seeds)
        self.zero_probability = float(zero_probability)
        if (
            not np.isfinite(self.zero_probability)
            or not 0.0 <= self.zero_probability <= 1.0
        ):
            raise ValueError("zero-latent probability must be in [0, 1]")
        self.counts = {seed: {"zero": 0, "random": 0} for seed in self.seeds}

    def sample(self, seed: int, rng: np.random.Generator) -> str:
        seed = int(seed)
        if seed not in self.counts:
            raise ValueError(f"unbound paper scene Seed{seed}")
        counts = self.counts[seed]
        total = counts["zero"] + counts["random"]
        deficits = {
            "zero": self.zero_probability * total - counts["zero"],
            "random": (1.0 - self.zero_probability) * total - counts["random"],
        }
        if np.isclose(deficits["zero"], deficits["random"]):
            mode = "zero" if float(rng.random()) < self.zero_probability else "random"
        else:
            mode = max(("zero", "random"), key=deficits.__getitem__)
        counts[mode] += 1
        return mode


validate_paper_seed_contract(PAPER_SCENE_SEEDS)
validate_probability_weights(TRANSITION_TARGET_WEIGHTS)
validate_probability_weights(CORRECTION_SEED_WEIGHTS)


__all__ = [
    "ALGORITHM_SEEDS",
    "CORRECTION_SEED_WEIGHTS",
    "NOVEL_SEEDS",
    "OFFLINE_SEEDS",
    "PAPER_SCENE_SEEDS",
    "PaperTransitionDeficitScheduler",
    "TRANSITION_TARGET_WEIGHTS",
    "UniformEpisodeLatentPolicy",
    "ZERO_LATENT_PROBABILITY",
    "validate_paper_seed_contract",
    "validate_probability_weights",
]
