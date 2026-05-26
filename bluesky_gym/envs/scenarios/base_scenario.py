"""Abstract base class for BlueSky-Gym conflict scenario generators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseScenario(ABC):
    """Abstract interface for conflict scenario generators.

    All concrete scenario classes must implement :meth:`generate`, which
    creates intruder aircraft around a specified ownship and returns the
    list of intruder call signs.

    Concrete subclasses are free to define scenario-specific constructor
    parameters (e.g., ``speed_range`` vs ``speed_delta_range`` for
    :class:`OvertakingScenario`).
    """

    @abstractmethod
    def generate(self, target_acid: str = 'AC0', actype: str = 'A320') -> List[str]:
        """Generate intruder aircraft around *target_acid*.

        Args:
            target_acid: Call sign of the ownship / protagonist aircraft.
            actype: ICAO aircraft type designator for all intruders.

        Returns:
            List of created intruder call signs.
        """
        ...
