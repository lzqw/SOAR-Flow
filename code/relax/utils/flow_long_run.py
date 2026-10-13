"""Pure scheduling contracts for Guardian-Flow-DSAC V2.4.1 Long-Run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


ALL_LONG_RUN_SEEDS = tuple(range(10))
HARD_SEEDS = (0, 1, 9)
RETENTION_TEACHER_SEEDS = (1, 2, 3, 4)
TRANSITION_TARGET_WEIGHTS = {
    0: 0.15,
    1: 0.15,
    9: 0.20,
    **{seed: 0.50 / 7.0 for seed in range(2, 9)},
}
CORRECTION_SEED_WEIGHTS = {
    0: 0.20,
    1: 0.15,
    9: 0.25,
    **{seed: 0.40 / 7.0 for seed in range(2, 9)},
}


def _normalise_weights(
    weights: Mapping[int, float],
    seeds: Sequence[int],
) -> dict[int, float]:
    values = {int(seed): float(weights[int(seed)]) for seed in seeds}
    if any(not np.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("all long-run Seed weights must be finite and positive")
    total = float(sum(values.values()))
    return {seed: value / total for seed, value in values.items()}


class TransitionDeficitScheduler:
    """Choose the Seed with the largest target-transition deficit."""

    def __init__(
        self,
        weights: Mapping[int, float] = TRANSITION_TARGET_WEIGHTS,
        *,
        tie_break_order: Sequence[int] = (9, 0, 1, 2, 3, 4, 5, 6, 7, 8),
    ) -> None:
        self.seeds = tuple(int(seed) for seed in tie_break_order)
        if set(self.seeds) != set(int(seed) for seed in weights):
            raise ValueError("deficit scheduler Seeds and weights differ")
        self.weights = _normalise_weights(weights, self.seeds)
        self.steps = {seed: 0 for seed in self.seeds}
        self.episodes = {seed: 0 for seed in self.seeds}
        self.current_seed = self._select()

    @property
    def total_steps(self) -> int:
        return int(sum(self.steps.values()))

    def deficit(self, seed: int) -> float:
        seed = int(seed)
        return self.weights[seed] * self.total_steps - self.steps[seed]

    def _select(self) -> int:
        order = {seed: index for index, seed in enumerate(self.seeds)}
        return max(
            self.seeds,
            key=lambda seed: (
                self.deficit(seed),
                self.weights[seed],
                -order[seed],
            ),
        )

    def record_step(self, seed: int, count: int = 1) -> None:
        seed = int(seed)
        count = int(count)
        if seed != self.current_seed:
            raise RuntimeError("recorded transition does not match active Seed")
        if count < 1:
            raise ValueError("transition count must be positive")
        self.steps[seed] += count

    def complete_episode(self) -> int:
        self.episodes[self.current_seed] += 1
        self.current_seed = self._select()
        return self.current_seed

    def abort_episode(self) -> int:
        """Reschedule after discarding an incomplete episode."""

        self.current_seed = self._select()
        return self.current_seed

    def fractions(self) -> dict[int, float]:
        total = max(self.total_steps, 1)
        return {seed: self.steps[seed] / total for seed in self.seeds}


class EpisodeLatentModePolicy:
    """Stratify episode modes to Seed-bound rates with random initial ties."""

    def __init__(
        self,
        *,
        hard_seeds: Sequence[int] = HARD_SEEDS,
        hard_zero_probability: float = 0.50,
        other_zero_probability: float = 0.20,
        zero_probability_by_seed: Mapping[int, float] | None = None,
    ) -> None:
        self.hard_seeds = frozenset(int(seed) for seed in hard_seeds)
        self.hard_zero_probability = float(hard_zero_probability)
        self.other_zero_probability = float(other_zero_probability)
        for value in (
            self.hard_zero_probability,
            self.other_zero_probability,
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("zero-latent probabilities must be in [0, 1]")
        self.zero_probability_by_seed = {
            int(seed): float(value)
            for seed, value in (zero_probability_by_seed or {}).items()
        }
        if any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.zero_probability_by_seed.values()
        ):
            raise ValueError("per-Seed zero-latent probabilities must be in [0, 1]")
        self.counts: dict[int, dict[str, int]] = {}

    def zero_probability(self, seed: int) -> float:
        if int(seed) in self.zero_probability_by_seed:
            return self.zero_probability_by_seed[int(seed)]
        return (
            self.hard_zero_probability
            if int(seed) in self.hard_seeds
            else self.other_zero_probability
        )

    def sample(self, seed: int, rng: np.random.Generator) -> str:
        seed = int(seed)
        counts = self.counts.setdefault(seed, {"zero": 0, "random": 0})
        total = counts["zero"] + counts["random"]
        probability = self.zero_probability(seed)
        deficits = {
            "zero": probability * total - counts["zero"],
            "random": (1.0 - probability) * total - counts["random"],
        }
        if np.isclose(deficits["zero"], deficits["random"]):
            mode = "zero" if float(rng.random()) < probability else "random"
        else:
            mode = max(("zero", "random"), key=deficits.__getitem__)
        counts[mode] += 1
        return mode


__all__ = [
    "ALL_LONG_RUN_SEEDS",
    "CORRECTION_SEED_WEIGHTS",
    "EpisodeLatentModePolicy",
    "HARD_SEEDS",
    "RETENTION_TEACHER_SEEDS",
    "TRANSITION_TARGET_WEIGHTS",
    "TransitionDeficitScheduler",
]
