"""Tests for the dense-RL critical-state filter."""
from __future__ import annotations

import numpy as np
import pytest

from bluesky_gym.envs.multi_agent_env import MultiAgentEnv


@pytest.fixture
def env():
    e = MultiAgentEnv(
        scenario_type="head_on",
        num_intruders=3,
        disturbance_level="none",
        enable_nfz=False,
        enable_adversarial=True,
    )
    yield e
    e.close()


def _random_protagonist_action(rng: np.random.Generator) -> dict:
    return {
        "heading": rng.uniform(-1, 1, size=(1,)),
        "speed": rng.uniform(-1, 1, size=(1,)),
        "altitude": int(rng.integers(0, 3)),
    }


def _random_adversary_actions(num: int, rng: np.random.Generator) -> dict:
    return {
        f"INTRUDER_{i + 1}": {
            "heading": rng.uniform(-1, 1, size=(1,)),
            "speed": rng.uniform(-1, 1, size=(1,)),
            "altitude": int(rng.integers(0, 3)),
        }
        for i in range(num)
    }


def test_info_contains_is_critical_state_flag(env):
    """Every step's info dict should carry the `is_critical_state` boolean field."""
    rng = np.random.default_rng(seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(
        _random_protagonist_action(rng),
        _random_adversary_actions(env.num_intruders, rng),
    )
    assert "is_critical_state" in info
    assert isinstance(info["is_critical_state"], (bool, np.bool_))


def test_critical_state_ratio_is_reasonable(env):
    """Critical state ratio over 100 steps in head-on scenario should be ≤ 80% (prevent all steps triggering)."""
    rng = np.random.default_rng(seed=42)
    env.reset(seed=42)

    critical_count = 0
    total = 0
    for _ in range(100):
        _, _, terminated, truncated, info = env.step(
            _random_protagonist_action(rng),
            _random_adversary_actions(env.num_intruders, rng),
        )
        critical_count += int(bool(info.get("is_critical_state", False)))
        total += 1
        if terminated or truncated:
            break

    ratio = critical_count / max(total, 1)
    # No lower bound required (random policy may never trigger); only assert it doesn't over-fire
    assert 0.0 <= ratio <= 0.80, f"critical state ratio {ratio:.2%} out of bounds"
