"""pytest session configuration: BlueSky simulator singleton initialisation."""
from __future__ import annotations

import bluesky as bs
import pytest


@pytest.fixture(scope="session", autouse=True)
def _init_bluesky():
    """Initialise BlueSky once for the entire test session."""
    if bs.sim is None:
        bs.init(mode='sim', detached=True)
    yield


@pytest.fixture(autouse=True)
def _reset_traffic_state():
    """Reset traffic and NFZ state before each test to ensure isolation."""
    bs.traf.reset()
    if hasattr(bs.traf, 'nfz'):
        bs.traf.nfz.clear_zones()
    yield
