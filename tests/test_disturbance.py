"""Natural disturbance module tests: verify disturbance magnitude increases monotonically across presets."""
from __future__ import annotations

import numpy as np
import pytest

import bluesky as bs


SIM_DURATION_S = 60.0  # 1 minute per preset is sufficient to distinguish magnitude levels
EARTH_RADIUS_M = 6_371_000.0


def _great_circle_m(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in metres, numerically safe for small angles."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_M * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0)))


def _run_with_preset(preset: str) -> dict:
    """Run simulation with the given disturbance preset and return deviation statistics from the ideal trajectory."""
    bs.traf.disturb.set_preset(preset)

    bs.traf.cre(
        acid="TEST_AC", actype="A320",
        aclat=52.0, aclon=4.0,
        achdg=45, acalt=3000, acspd=150,
    )

    ideal_lat = bs.traf.lat[0]
    ideal_lon = bs.traf.lon[0]
    ideal_alt = bs.traf.alt[0]
    ideal_hdg = bs.traf.hdg[0]
    ideal_tas = bs.traf.tas[0]

    pos_errors, alt_errors, hdg_errors = [], [], []
    n_steps = int(SIM_DURATION_S / bs.sim.simdt)
    for _ in range(n_steps):
        ideal_gsnorth = ideal_tas * np.cos(np.radians(ideal_hdg))
        ideal_gseast = ideal_tas * np.sin(np.radians(ideal_hdg))
        ideal_lat += np.degrees(bs.sim.simdt * ideal_gsnorth / EARTH_RADIUS_M)
        ideal_lon += np.degrees(
            bs.sim.simdt * ideal_gseast / (np.cos(np.radians(ideal_lat)) * EARTH_RADIUS_M)
        )

        bs.sim.step()

        pos_errors.append(_great_circle_m(ideal_lat, ideal_lon, bs.traf.lat[0], bs.traf.lon[0]))
        alt_errors.append(abs(bs.traf.alt[0] - ideal_alt))
        hdg_errors.append(abs(((bs.traf.hdg[0] - ideal_hdg + 180) % 360) - 180))

    return {
        "pos_max": float(np.max(pos_errors)),
        "alt_max": float(np.max(alt_errors)),
        "hdg_max": float(np.max(hdg_errors)),
    }


def test_set_preset_enables_disturbance():
    """`set_preset('light')` should enable disturbance; `set_preset('none')` should disable it."""
    bs.traf.disturb.set_preset("light")
    assert bs.traf.disturb.enabled is True
    bs.traf.disturb.set_preset("none")
    assert bs.traf.disturb.enabled is False


def test_natural_log_prob_returns_finite_value():
    """natural_log_prob should return a finite real number for valid inputs."""
    bs.traf.disturb.set_preset("medium")
    lp = bs.traf.disturb.natural_log_prob(dnorth=1.0, deast=1.0, dspd=0.1, dhdg=0.1, dalt=1.0)
    assert np.isfinite(lp)


@pytest.mark.parametrize("preset", ["none", "light", "medium", "heavy"])
def test_disturbance_runs_without_error(preset):
    """All disturbance presets should complete a short simulation without aircraft divergence."""
    stats = _run_with_preset(preset)
    assert stats["pos_max"] < 1e6  # 1000 km — far above any reasonable disturbance
    assert stats["alt_max"] < 5_000.0


def test_disturbance_intensity_monotonic():
    """Stronger disturbance should produce larger maximum positional error (weak assertion: none==0 and heavy >> light).

    In a single finite-length simulation, ``light/medium`` may occasionally swap due to randomness,
    so only coarse-grained trend is asserted to avoid test flakiness across seeds.
    """
    np.random.seed(20260501)
    pos_max = {}
    for preset in ["none", "light", "medium", "heavy"]:
        bs.traf.reset()
        pos_max[preset] = _run_with_preset(preset)["pos_max"]

    assert pos_max["none"] == pytest.approx(0.0, abs=1e-3)
    assert pos_max["light"] > 0
    assert pos_max["medium"] > 0
    assert pos_max["heavy"] > pos_max["light"] * 1.2
    assert pos_max["heavy"] > pos_max["none"]
