""" Scenario generators for BlueSky-Gym conflict resolution environments. """

from __future__ import annotations

from typing import Type

from .base_scenario import BaseScenario
from .head_on_scenario import HeadOnScenario
from .crossing_scenario import CrossingScenario
from .merging_scenario import MergingScenario
from .overtaking_scenario import OvertakingScenario

__all__ = [
    'BaseScenario',
    'HeadOnScenario',
    'CrossingScenario',
    'MergingScenario',
    'OvertakingScenario',
    'get_scenario',
]


def get_scenario(scenario_type: str) -> Type[BaseScenario]:
    """Return the scenario *class* (un-instantiated) for *scenario_type*.

    Args:
        scenario_type: One of ``'head_on'``, ``'crossing'``,
            ``'merging'``, or ``'overtaking'`` (case-insensitive).

    Returns:
        The concrete :class:`BaseScenario` subclass.

    Raises:
        ValueError: If *scenario_type* is not one of the supported values.
    """
    scenarios: dict[str, Type[BaseScenario]] = {
        'head_on': HeadOnScenario,
        'crossing': CrossingScenario,
        'merging': MergingScenario,
        'overtaking': OvertakingScenario,
    }

    key = scenario_type.lower()
    if key not in scenarios:
        raise ValueError(
            f"Unknown scenario type: {scenario_type!r}. "
            f"Available types: {list(scenarios.keys())}"
        )
    return scenarios[key]
