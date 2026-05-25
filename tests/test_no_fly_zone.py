"""No-Fly Zone module tests."""
from __future__ import annotations

import pytest

import bluesky as bs


# ---- circular NFZ -----------------------------------------------------------------

CIRCULAR_ZONE_CASES = [
    pytest.param(52.30, 4.80, 1000, True,  id="center_in_altitude_range"),
    pytest.param(52.35, 4.85, 1500, True,  id="inside_circle"),
    pytest.param(52.50, 5.00, 2000, False, id="outside_horizontally"),
    pytest.param(52.30, 4.80, 3500, False, id="above_altitude_max"),
    pytest.param(52.30, 4.80,  100, True,  id="below_default_min"),
]


@pytest.mark.parametrize("lat, lon, alt, expected_violation", CIRCULAR_ZONE_CASES)
def test_circular_zone_point_check(lat, lon, alt, expected_violation):
    bs.traf.nfz.create_circular_zone(
        name="EHAM_RESTRICTED", lat=52.3, lon=4.8, radius=10, alt_min=0, alt_max=3000
    )
    violations = bs.traf.nfz.check_point(lat, lon, alt)
    assert bool(violations) is expected_violation


# ---- polygon NFZ -----------------------------------------------------------------

POLYGON_ZONE_CASES = [
    pytest.param(52.10, 4.20, 2000, True,  id="inside_polygon"),
    pytest.param(52.15, 4.35, 3000, True,  id="inside_near_edge"),
    pytest.param(52.30, 4.20, 2000, False, id="outside_north"),
    pytest.param(52.10, 3.90, 2000, False, id="outside_west"),
]


@pytest.mark.parametrize("lat, lon, alt, expected_violation", POLYGON_ZONE_CASES)
def test_polygon_zone_point_check(lat, lon, alt, expected_violation):
    bs.traf.nfz.create_polygon_zone(
        name="RESTRICTED_AREA",
        lats=[52.0, 52.2, 52.2, 52.0],
        lons=[4.0, 4.0, 4.4, 4.4],
        alt_min=0, alt_max=99999,
    )
    violations = bs.traf.nfz.check_point(lat, lon, alt)
    assert bool(violations) is expected_violation


# ---- multiple NFZs and per-aircraft violation tracking ---------------------------

def test_multiple_zones_and_aircraft_tracking():
    """Manage multiple NFZs simultaneously and correctly identify violations per aircraft."""
    bs.traf.nfz.create_circular_zone("ZONE_A", 52.0, 4.0, 5, 0, 2000)
    bs.traf.nfz.create_circular_zone("ZONE_B", 52.2, 4.3, 3, 1000, 3000)
    bs.traf.nfz.create_polygon_zone(
        "ZONE_C",
        [52.3, 52.4, 52.4, 52.3],
        [4.0, 4.0, 4.2, 4.2],
        0, 99999,
    )
    assert len(bs.traf.nfz.zones) == 3

    bs.traf.cre("AC001", actype="A320", aclat=52.0, aclon=4.0, achdg=90, acalt=1000, acspd=250)
    bs.traf.cre("AC002", actype="A320", aclat=52.2, aclon=4.3, achdg=90, acalt=2000, acspd=250)
    bs.traf.cre("AC003", actype="A320", aclat=52.35, aclon=4.1, achdg=90, acalt=3000, acspd=250)
    assert bs.traf.ntraf == 3

    # AC001 在 ZONE_A 内
    v1 = bs.traf.nfz.check_aircraft("AC001", bs.traf.lat[0], bs.traf.lon[0], bs.traf.alt[0])
    assert v1, "AC001 should be inside ZONE_A"

    # AC002 在 ZONE_B 内
    v2 = bs.traf.nfz.check_aircraft("AC002", bs.traf.lat[1], bs.traf.lon[1], bs.traf.alt[1])
    assert v2, "AC002 should be inside ZONE_B"

    # AC003 在 ZONE_C 内
    v3 = bs.traf.nfz.check_aircraft("AC003", bs.traf.lat[2], bs.traf.lon[2], bs.traf.alt[2])
    assert v3, "AC003 should be inside ZONE_C"


def test_violation_count_accumulates_over_time():
    """对同一飞机重复 check 应累积违规次数。"""
    bs.traf.nfz.create_circular_zone("ZONE_X", 52.0, 4.0, 5, 0, 5000)
    bs.traf.cre("AC001", actype="A320", aclat=52.0, aclon=4.0, achdg=0, acalt=1000, acspd=200)

    for _ in range(3):
        bs.traf.nfz.check_aircraft("AC001", bs.traf.lat[0], bs.traf.lon[0], bs.traf.alt[0])

    assert bs.traf.nfz.get_violation_count("AC001") >= 1
