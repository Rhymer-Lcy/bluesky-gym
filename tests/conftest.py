"""pytest session configuration: BlueSky simulator singleton initialisation."""
from __future__ import annotations

import importlib.util

import pytest

_HAS_BLUESKY = importlib.util.find_spec("bluesky") is not None
if _HAS_BLUESKY:
    import bluesky as bs


@pytest.fixture(scope="session", autouse=True)
def _init_bluesky():
    """Initialise BlueSky once for the entire test session (no-op if bluesky is not installed)."""
    if not _HAS_BLUESKY:
        yield
        return
    if bs.sim is None:
        bs.init(mode='sim', detached=True)
    yield


@pytest.fixture(autouse=True)
def _reset_traffic_state():
    """Reset traffic and NFZ state before each test (no-op if bluesky is not installed)."""
    if not _HAS_BLUESKY:
        yield
        return
    bs.traf.reset()
    if hasattr(bs.traf, 'nfz'):
        bs.traf.nfz.clear_zones()
    yield
