"""Common utility functions shared across BlueSky-Gym environments."""
from __future__ import annotations

import numpy as np
from .constants import ALTITUDE_LAYERS


def get_altitude_layer_index(altitude: float) -> int:
    """Return the index of the altitude layer closest to *altitude* (metres)."""
    distances = [abs(altitude - layer_alt) for layer_alt in ALTITUDE_LAYERS]
    return int(np.argmin(distances))
