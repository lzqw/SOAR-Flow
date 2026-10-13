"""Immutable scheduling bindings for Guardian-Flow-DSAC V2.4.2."""

from __future__ import annotations

from relax.utils.flow_long_run import CORRECTION_SEED_WEIGHTS


SEED9 = 9
SEED9_CORRECTION_FRACTION = 0.50
SEED9_ZERO_LATENT_PROBABILITY = 0.70


def _v242_correction_weights() -> dict[int, float]:
    """Preserve V2.4.1 relative weights inside the non-Seed9 half."""

    remaining = {
        seed: weight
        for seed, weight in CORRECTION_SEED_WEIGHTS.items()
        if seed != SEED9
    }
    total = float(sum(remaining.values()))
    return {
        **{
            seed: (1.0 - SEED9_CORRECTION_FRACTION) * weight / total
            for seed, weight in remaining.items()
        },
        SEED9: SEED9_CORRECTION_FRACTION,
    }


CORRECTION_SEED_WEIGHTS_V242 = _v242_correction_weights()
ZERO_LATENT_PROBABILITY_BY_SEED_V242 = {
    0: 0.50,
    1: 0.50,
    9: SEED9_ZERO_LATENT_PROBABILITY,
    **{seed: 0.20 for seed in range(2, 9)},
}


__all__ = [
    "CORRECTION_SEED_WEIGHTS_V242",
    "SEED9",
    "SEED9_CORRECTION_FRACTION",
    "SEED9_ZERO_LATENT_PROBABILITY",
    "ZERO_LATENT_PROBABILITY_BY_SEED_V242",
]
