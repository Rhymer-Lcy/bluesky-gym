"""Integration tests for ConflictResolutionEnv."""
from __future__ import annotations

import numpy as np
import pytest

from bluesky_gym.envs.conflict_resolution_env import ConflictResolutionEnv


# expected obs vector length: 6 ego + 4 target + 4 conflict + 2 NFZ + 4 × num_intruders
def _expected_obs_size(num_intruders: int) -> int:
    return 6 + 4 + 4 + 2 + 4 * num_intruders


CONFIG_CASES = [
    pytest.param(
        {"scenario_type": "head_on", "num_intruders": 3, "disturbance_level": "none", "enable_nfz": False},
        id="head_on_clean",
    ),
    pytest.param(
        {"scenario_type": "crossing", "num_intruders": 5, "disturbance_level": "light", "enable_nfz": True},
        id="crossing_light_nfz",
    ),
    pytest.param(
        {"scenario_type": "random", "num_intruders": 4, "disturbance_level": "medium", "enable_nfz": True},
        id="random_medium_nfz",
    ),
]


@pytest.mark.parametrize("config", CONFIG_CASES)
def test_environment_creation_and_reset(config):
    env = ConflictResolutionEnv(**config)
    try:
        obs, info = env.reset()
        assert obs.shape == (_expected_obs_size(config["num_intruders"]),)
        assert np.isfinite(obs).all()
    finally:
        env.close()


def test_observation_vector_layout():
    """Observation vector segment lengths must strictly match the documented layout."""
    env = ConflictResolutionEnv(
        scenario_type="head_on", num_intruders=5,
        disturbance_level="light", enable_nfz=True,
    )
    try:
        obs, _ = env.reset()
        assert obs.shape == (_expected_obs_size(5),)
        # ego 6 dims: normalised values should not be far from 1
        assert np.max(np.abs(obs[0:6])) <= 5.0
    finally:
        env.close()


@pytest.mark.parametrize(
    "action",
    [
        {"heading": np.array([0.5]),  "speed": np.array([0.0]),  "altitude": 1},
        {"heading": np.array([-0.5]), "speed": np.array([0.3]),  "altitude": 2},
        {"heading": np.array([0.0]),  "speed": np.array([-0.2]), "altitude": 0},
    ],
)
def test_step_returns_valid_transition(action):
    env = ConflictResolutionEnv(scenario_type="crossing", num_intruders=3)
    try:
        env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_full_episode_terminates():
    """A random policy should reach max_episode_steps and terminate normally."""
    rng = np.random.default_rng(seed=0)
    env = ConflictResolutionEnv(scenario_type="head_on", num_intruders=3)
    try:
        env.reset(seed=0)
        info = {}
        terminated = truncated = False
        for _ in range(env.max_episode_steps + 5):
            action = {
                "heading": rng.uniform(-1, 1, size=(1,)),
                "speed": rng.uniform(-1, 1, size=(1,)),
                "altitude": int(rng.integers(0, 3)),
            }
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert terminated or truncated
        for key in ("total_conflicts", "total_nfz_violations", "waypoint_reached"):
            assert key in info
    finally:
        env.close()


@pytest.mark.parametrize("scenario_type", ["head_on", "crossing", "merging", "overtaking", "random"])
def test_all_scenario_types_produce_valid_obs(scenario_type):
    env = ConflictResolutionEnv(scenario_type=scenario_type, num_intruders=3)
    try:
        obs, _ = env.reset()
        assert obs.shape == (_expected_obs_size(3),)
        assert np.isfinite(obs).all()
    finally:
        env.close()
