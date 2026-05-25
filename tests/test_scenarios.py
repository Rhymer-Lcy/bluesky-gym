"""Tests for the canonical conflict scenario generators."""
from __future__ import annotations

import numpy as np
import pytest

import bluesky as bs
from bluesky.tools.geo import qdrdist
from bluesky_gym.envs.scenarios import (
    CrossingScenario,
    HeadOnScenario,
    MergingScenario,
    OvertakingScenario,
    get_scenario,
)


SCENARIO_CASES = [
    pytest.param(
        HeadOnScenario(
            num_intruders=3,
            dpsi_range=(170, 190),
            dcpa_range=(0, 3),
            tlosh_range=(80, 120),
        ),
        (150, 210),
        id="head_on",
    ),
    pytest.param(
        CrossingScenario(
            num_intruders=3,
            dpsi_range=(80, 100),
            dcpa_range=(0, 3),
            tlosh_range=(80, 120),
        ),
        (60, 120),
        id="crossing",
    ),
    pytest.param(
        MergingScenario(
            num_intruders=3,
            dpsi_range=(40, 50),
            dcpa_range=(0, 2),
            tlosh_range=(80, 120),
        ),
        (20, 70),
        id="merging",
    ),
    pytest.param(
        OvertakingScenario(
            num_intruders=3,
            dpsi_range=(0, 10),
            dcpa_range=(0, 2),
            tlosh_range=(150, 250),
        ),
        (0, 25),
        id="overtaking",
    ),
]


@pytest.fixture
def cd_enabled():
    """Enable conflict detection."""
    bs.stack.stack("CDMETHOD ON")
    yield


def _create_target(acid: str = "TARGET") -> int:
    bs.traf.cre(
        acid=acid, actype="A320",
        aclat=52.0, aclon=4.0,
        achdg=0, acalt=3000, acspd=150,
    )
    return bs.traf.id2idx(acid)


@pytest.mark.parametrize("scenario, expected_dpsi", SCENARIO_CASES)
def test_scenario_parameters_in_valid_range(scenario, expected_dpsi):
    """Scenario parameters should lie within acceptable bounds."""
    assert 1 <= scenario.num_intruders <= 10
    assert 0 <= scenario.dpsi_range[0] <= scenario.dpsi_range[1] <= 360
    assert 0 <= scenario.dcpa_range[0] <= scenario.dcpa_range[1] <= 50
    assert 10 <= scenario.tlosh_range[0] <= scenario.tlosh_range[1] <= 600


@pytest.mark.parametrize("scenario, expected_dpsi", SCENARIO_CASES)
def test_scenario_generation_creates_intruders(scenario, expected_dpsi):
    """After generating a scenario the expected number of intruders should exist in BlueSky."""
    target_idx = _create_target()
    intruders = scenario.generate(target_acid="TARGET")

    assert len(intruders) == scenario.num_intruders
    assert bs.traf.ntraf == 1 + scenario.num_intruders

    target_lat = bs.traf.lat[target_idx]
    target_lon = bs.traf.lon[target_idx]
    target_hdg = bs.traf.hdg[target_idx]

    for acid in intruders:
        idx = bs.traf.id2idx(acid)
        assert idx >= 0, f"intruder {acid} not found"

        _, dist_nm = qdrdist(target_lat, target_lon, bs.traf.lat[idx], bs.traf.lon[idx])
        assert dist_nm > 0, "intruder should not be at target position"

        hdg_diff = abs(bs.traf.hdg[idx] - target_hdg)
        if hdg_diff > 180:
            hdg_diff = 360 - hdg_diff
        lo, hi = expected_dpsi
        assert lo - 20 <= hdg_diff <= hi + 20, (
            f"intruder heading diff {hdg_diff:.1f}° out of expected band {expected_dpsi}"
        )


@pytest.mark.parametrize("scenario, expected_dpsi", SCENARIO_CASES)
def test_scenario_triggers_conflict_detection(scenario, expected_dpsi, cd_enabled):
    """Each scenario should trigger at least one conflict detection event within 600 sim steps."""
    _create_target()
    scenario.generate(target_acid="TARGET")

    detected = False
    for _ in range(600):
        bs.sim.step()
        if len(bs.traf.cd.confpairs) > 0:
            detected = True
            break

    assert detected, f"scenario did not produce a conflict within 600 steps"
    assert len(bs.traf.cd.tcpa) > 0
    # tcpa may be negative (past closest approach) — assert finite only
    assert np.all(np.isfinite(bs.traf.cd.tcpa))


@pytest.mark.parametrize(
    "scenario_type, expected_cls",
    [
        ("head_on", HeadOnScenario),
        ("crossing", CrossingScenario),
        ("merging", MergingScenario),
        ("overtaking", OvertakingScenario),
    ],
)
def test_get_scenario_factory(scenario_type, expected_cls):
    """Scenario factory should return the correct class."""
    cls = get_scenario(scenario_type)
    assert cls is expected_cls
