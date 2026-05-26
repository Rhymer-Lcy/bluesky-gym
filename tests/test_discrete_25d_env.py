"""Tests for the 2.5D discrete altitude-layer environment."""
from __future__ import annotations

import numpy as np
import pytest

from bluesky_gym.envs.discrete_25d_env import ALTITUDE_LAYERS, Discrete25DEnv


@pytest.fixture
def env():
    e = Discrete25DEnv()
    yield e
    e.close()


def test_initialization_yields_valid_observation(env):
    obs, info = env.reset()
    assert isinstance(obs, dict)
    layer = int(obs["current_altitude_layer"][0])
    assert 0 <= layer < len(ALTITUDE_LAYERS)


REQUIRED_OBS_KEYS = [
    "intruder_distance",
    "cos_difference_pos",
    "sin_difference_pos",
    "current_altitude_layer",
    "altitude_difference",
    "intruder_altitude_layer",
    "waypoint_distance",
    "cos_drift",
    "sin_drift",
]


@pytest.mark.parametrize("key", REQUIRED_OBS_KEYS)
def test_observation_contains_required_keys(env, key):
    obs, _ = env.reset()
    assert key in obs


def test_action_space_accepts_three_altitude_choices(env):
    """altitude ∈ {0=descend, 1=maintain, 2=climb}。"""
    env.reset()
    for altitude in (0, 1, 2):
        action = {"heading": np.array([0.0]), "altitude": altitude}
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float))
        assert "current_altitude_layer" in info
        if terminated or truncated:
            env.reset()


def test_climb_increases_target_altitude_layer(env):
    """Repeated climb actions should advance the target altitude layer to the maximum."""
    env.reset()
    info = {}
    for _ in range(len(ALTITUDE_LAYERS) + 2):
        _, _, terminated, truncated, info = env.step({"heading": np.array([0.0]), "altitude": 2})
        if terminated or truncated:
            break
    assert info["target_altitude_layer"] == len(ALTITUDE_LAYERS) - 1


def test_descend_decreases_target_altitude_layer(env):
    """Repeated descend actions from the top layer should bring the target altitude layer to 0."""
    env.reset()
    # Climb to top layer first
    for _ in range(len(ALTITUDE_LAYERS) + 2):
        _, _, term, trunc, _ = env.step({"heading": np.array([0.0]), "altitude": 2})
        if term or trunc:
            env.reset()
    info = {}
    for _ in range(len(ALTITUDE_LAYERS) + 2):
        _, _, term, trunc, info = env.step({"heading": np.array([0.0]), "altitude": 0})
        if term or trunc:
            break
    assert info["target_altitude_layer"] == 0


def test_full_episode_with_random_actions(env):
    """A full episode with random actions should terminate normally and include summary statistics."""
    rng = np.random.default_rng(seed=0)
    env.reset(seed=0)
    info = {}
    for _ in range(50):
        action = {
            "heading": rng.uniform(-1, 1, size=(1,)),
            "altitude": int(rng.integers(0, 3)),
        }
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    for key in ("total_intrusions", "total_altitude_changes", "average_drift"):
        assert key in info
