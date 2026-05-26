"""RL framework compatibility tests (Gymnasium API + optional SB3 integration)."""
from __future__ import annotations

import importlib.util

import gymnasium as gym
import numpy as np
import pytest

from bluesky_gym.envs.conflict_resolution_env import ConflictResolutionEnv
from bluesky_gym.utils.wrappers import FlattenDictActionWrapper


_HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None


@pytest.fixture
def env():
    e = ConflictResolutionEnv(scenario_type="head_on", num_intruders=3)
    yield e
    e.close()


# ---- Gymnasium API ------------------------------------------------------------

def test_required_methods_and_attributes(env):
    for name in ("reset", "step", "close", "observation_space", "action_space"):
        assert hasattr(env, name)


def test_reset_returns_obs_and_info_dict(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)


def test_step_returns_5tuple_with_correct_types(env):
    env.reset()
    action = {"heading": np.array([0.0]), "speed": np.array([0.0]), "altitude": 1}
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (int, float, np.number))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ---- observation/action space consistency ----------------------------------------

def test_observation_in_observation_space(env):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)


def test_sampled_action_in_action_space(env):
    action = env.action_space.sample()
    assert env.action_space.contains(action)


# ---- custom flatten wrapper (for SB3) -------------------------------------------


def test_flatten_dict_action_wrapper_round_trip():
    base = ConflictResolutionEnv(scenario_type="merging", num_intruders=3)
    try:
        wrapped = FlattenDictActionWrapper(base)
        wrapped.reset()
        flat_action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(flat_action)
        assert wrapped.observation_space.contains(obs)
    finally:
        base.close()


# ---- optional SB3 integration ---------------------------------------------------

@pytest.mark.skipif(not _HAS_SB3, reason="stable-baselines3 not installed")
def test_stablebaselines3_short_training_and_save(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv

    base = ConflictResolutionEnv(
        scenario_type="head_on", num_intruders=3,
        disturbance_level="none", enable_nfz=False,
    )
    try:
        env = FlattenDictActionWrapper(base)
        check_env(env, warn=False)

        vec_env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=64, batch_size=32, n_epochs=2)
        model.learn(total_timesteps=128)

        save_path = tmp_path / "ppo_test.zip"
        model.save(str(save_path))
        assert save_path.exists()

        loaded = PPO.load(str(save_path))
        obs = vec_env.reset()
        action, _ = loaded.predict(obs, deterministic=True)
        vec_env.step(action)

        vec_env.close()
    finally:
        base.close()


# ---- multi-episode training loop ------------------------------------------------

def test_manual_training_loop_runs_multiple_episodes():
    """Random policy should produce reasonable statistics over several episodes."""
    rng = np.random.default_rng(seed=0)
    env = ConflictResolutionEnv(
        scenario_type="random", num_intruders=4,
        disturbance_level="light", enable_nfz=True,
    )
    try:
        for _ in range(2):
            env.reset(seed=int(rng.integers(0, 10_000)))
            for _ in range(30):
                action = {
                    "heading": rng.uniform(-0.5, 0.5, size=(1,)),
                    "speed": rng.uniform(-0.3, 0.3, size=(1,)),
                    "altitude": int(rng.integers(0, 3)),
                }
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            for key in ("total_conflicts", "total_nfz_violations", "waypoint_reached"):
                assert key in info
    finally:
        env.close()
