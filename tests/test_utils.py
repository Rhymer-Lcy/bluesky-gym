"""Unit tests for bluesky_gym.envs.common.utils and bluesky_gym.envs.scenarios.base_scenario.

These tests cover the two modules added in the recent refactoring pass:
  - get_altitude_layer_index()  (extracted from ConflictResolutionEnv / Discrete25DEnv)
  - BaseScenario ABC            (abstract interface satisfied by all 4 scenario classes)

The get_altitude_layer_index and BaseScenario ABC tests do not require a live BlueSky
simulator and run in any Python environment that has bluesky_gym installed.

The parametrized test that checks each built-in scenario class inherits from BaseScenario
imports the concrete scenario modules (which depend on bluesky); those tests are skipped
automatically when bluesky is not installed.
"""
from __future__ import annotations

import importlib.util

import pytest

from bluesky_gym.envs.common.constants import ALTITUDE_LAYERS
from bluesky_gym.envs.common.utils import get_altitude_layer_index

# BaseScenario itself has no bluesky dependency; import directly from the ABC module.
from bluesky_gym.envs.scenarios.base_scenario import BaseScenario

# Concrete scenario classes import bluesky at module level.
_HAS_BLUESKY = importlib.util.find_spec("bluesky") is not None
if _HAS_BLUESKY:
    from bluesky_gym.envs.scenarios import (
        CrossingScenario,
        HeadOnScenario,
        MergingScenario,
        OvertakingScenario,
    )
    _CONCRETE_SCENARIOS = [HeadOnScenario, CrossingScenario, MergingScenario, OvertakingScenario]
else:
    _CONCRETE_SCENARIOS = []


# ---------------------------------------------------------------------------
# get_altitude_layer_index
# ---------------------------------------------------------------------------

class TestGetAltitudeLayerIndex:
    """Tests for the shared get_altitude_layer_index() utility."""

    def test_exact_match_each_layer(self):
        """Exact altitude for every defined layer returns the correct index."""
        for i, alt in enumerate(ALTITUDE_LAYERS):
            assert get_altitude_layer_index(alt) == i

    def test_below_lowest_layer_returns_zero(self):
        """Altitude below the lowest layer must return index 0."""
        assert get_altitude_layer_index(0.0) == 0
        assert get_altitude_layer_index(-500.0) == 0

    def test_above_highest_layer_returns_last_index(self):
        """Altitude above the highest layer must return the last valid index."""
        last_idx = len(ALTITUDE_LAYERS) - 1
        assert get_altitude_layer_index(99_999.0) == last_idx

    def test_return_type_is_plain_int(self):
        """Must return plain Python int, not numpy int64 (SB3 compatibility)."""
        result = get_altitude_layer_index(ALTITUDE_LAYERS[0])
        assert type(result) is int

    def test_closest_layer_is_chosen(self):
        """An altitude slightly above a layer boundary maps to the nearer layer."""
        # One metre above ALTITUDE_LAYERS[0] is still closer to layer 0 than layer 1
        assert get_altitude_layer_index(ALTITUDE_LAYERS[0] + 1) == 0
        # One metre below ALTITUDE_LAYERS[-1] is still closer to the last layer
        assert get_altitude_layer_index(ALTITUDE_LAYERS[-1] - 1) == len(ALTITUDE_LAYERS) - 1

    @pytest.mark.parametrize("altitude,expected_idx", [
        (ALTITUDE_LAYERS[0], 0),
        (ALTITUDE_LAYERS[1], 1),
        (ALTITUDE_LAYERS[2], 2),
        (ALTITUDE_LAYERS[3], 3),
        (ALTITUDE_LAYERS[4], 4),
    ])
    def test_parametrized_exact_matches(self, altitude, expected_idx):
        assert get_altitude_layer_index(altitude) == expected_idx


# ---------------------------------------------------------------------------
# BaseScenario ABC
# ---------------------------------------------------------------------------

class TestBaseScenario:
    """Tests for the BaseScenario abstract interface."""

    def test_cannot_instantiate_abstract_class_directly(self):
        """Instantiating BaseScenario without implementing generate() raises TypeError."""
        with pytest.raises(TypeError):
            BaseScenario()  # type: ignore[abstract]

    def test_incomplete_subclass_remains_abstract(self):
        """A subclass that omits generate() is still abstract and cannot be instantiated."""
        class IncompleteScenario(BaseScenario):
            pass  # generate() not implemented

        with pytest.raises(TypeError):
            IncompleteScenario()

    def test_minimal_concrete_subclass_is_instantiable(self):
        """A subclass that implements generate() can be constructed normally."""
        class MinimalScenario(BaseScenario):
            def generate(self, target_acid: str = 'AC0', actype: str = 'A320'):
                return []

        s = MinimalScenario()
        assert isinstance(s, BaseScenario)

    def test_generate_may_return_empty_list(self):
        """generate() returning [] is a valid result (no intruders generated)."""
        class EmptyScenario(BaseScenario):
            def generate(self, target_acid: str = 'AC0', actype: str = 'A320'):
                return []

        assert EmptyScenario().generate() == []

    def test_generate_returns_list_of_strings(self):
        """generate() must return a list; a concrete implementation returning call signs works."""
        class StubScenario(BaseScenario):
            def generate(self, target_acid: str = 'AC0', actype: str = 'A320'):
                return ['INTRUDER1', 'INTRUDER2']

        result = StubScenario().generate()
        assert isinstance(result, list)
        assert all(isinstance(cs, str) for cs in result)

    @pytest.mark.skipif(not _HAS_BLUESKY, reason="bluesky not installed")
    @pytest.mark.parametrize("cls", _CONCRETE_SCENARIOS)
    def test_builtin_scenarios_are_subclasses_of_base_scenario(self, cls):
        """All four built-in scenario classes must be proper subclasses of BaseScenario."""
        assert issubclass(cls, BaseScenario), (
            f"{cls.__name__} does not inherit from BaseScenario"
        )
