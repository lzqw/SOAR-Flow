"""Pure-data scenario specifications for Safe100 MetaDrive experiments.

The scenario grammar is intentionally shell friendly:

``lane:fraction+lane:fraction;lane:fraction``

``+`` joins parked obstacles inside one scene and ``;`` separates scenes.
Legacy one-obstacle strings remain valid.
"""

from __future__ import annotations

from typing import Iterable, Sequence, TypeAlias


ObstacleSpec: TypeAlias = tuple[int, float]
ScenarioSpec: TypeAlias = tuple[ObstacleSpec, ...]
ScenarioSuite: TypeAlias = tuple[ScenarioSpec, ...]


DEFAULT_OFFLINE_SCENARIOS_TEXT = (
    "0:0.30+2:0.62;"
    "2:0.32+0:0.66;"
    "0:0.27+2:0.50+0:0.76;"
    "2:0.28+0:0.53+2:0.78"
)
DEFAULT_ONLINE_SCENARIOS_TEXT = (
    "0:0.20+1:0.48;"
    "2:0.20+1:0.52"
)
DEFAULT_VALIDATION_SCENARIOS_TEXT = (
    "0:0.22+1:0.46;"
    "2:0.22+1:0.50"
)


def _validate_obstacle(lane: int, fraction: float) -> ObstacleSpec:
    lane = int(lane)
    fraction = float(fraction)
    if lane not in (0, 1, 2):
        raise ValueError(f"invalid lane index {lane}; expected 0, 1, or 2")
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            f"invalid obstacle fraction {fraction}; expected a value in (0, 1)"
        )
    return lane, fraction


def normalize_scenario(
    scenario: Sequence[ObstacleSpec] | ObstacleSpec,
) -> ScenarioSpec:
    """Normalize a legacy obstacle pair or a multi-obstacle scenario."""

    if (
        len(scenario) == 2
        and isinstance(scenario[0], (int, float))
        and isinstance(scenario[1], (int, float))
    ):
        return (_validate_obstacle(int(scenario[0]), float(scenario[1])),)
    normalized = tuple(
        _validate_obstacle(int(obstacle[0]), float(obstacle[1]))
        for obstacle in scenario
    )
    if not normalized:
        raise ValueError("a scenario must contain at least one obstacle")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"scenario contains a duplicate obstacle: {scenario!r}")
    return normalized


def parse_scenario(text: str) -> ScenarioSpec:
    obstacles: list[ObstacleSpec] = []
    for raw in text.split("+"):
        item = raw.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"invalid obstacle {item!r}; expected lane:fraction"
            )
        obstacles.append(_validate_obstacle(int(parts[0]), float(parts[1])))
    return normalize_scenario(tuple(obstacles))


def parse_scenarios(text: str) -> ScenarioSuite:
    scenarios = tuple(
        parse_scenario(item) for item in text.split(";") if item.strip()
    )
    if not scenarios:
        raise ValueError("at least one scenario is required")
    return scenarios


def scenario_to_text(scenario: Sequence[ObstacleSpec] | ObstacleSpec) -> str:
    return "+".join(
        f"{lane}:{fraction:g}" for lane, fraction in normalize_scenario(scenario)
    )


def scenarios_to_text(scenarios: Iterable[ScenarioSpec]) -> str:
    return ";".join(scenario_to_text(scenario) for scenario in scenarios)


def scenario_to_dict(
    scenario: Sequence[ObstacleSpec] | ObstacleSpec,
) -> dict[str, object]:
    normalized = normalize_scenario(scenario)
    return {
        "text": scenario_to_text(normalized),
        "obstacle_count": len(normalized),
        "obstacles": [
            {"lane": int(lane), "fraction": float(fraction)}
            for lane, fraction in normalized
        ],
    }


__all__ = [
    "DEFAULT_OFFLINE_SCENARIOS_TEXT",
    "DEFAULT_ONLINE_SCENARIOS_TEXT",
    "DEFAULT_VALIDATION_SCENARIOS_TEXT",
    "ObstacleSpec",
    "ScenarioSpec",
    "ScenarioSuite",
    "normalize_scenario",
    "parse_scenario",
    "parse_scenarios",
    "scenario_to_dict",
    "scenario_to_text",
    "scenarios_to_text",
]
