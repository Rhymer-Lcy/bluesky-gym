"""Tests for adversarial training components (AdversarialPolicy + MultiAgentEnv)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from bluesky_gym.envs.adversarial_policy import AdversarialPolicy
from bluesky_gym.envs.multi_agent_env import MultiAgentEnv


ADVERSARY_OBS_DIM = 12


# ---- AdversarialPolicy --------------------------------------------------------

@pytest.fixture
def policy() -> AdversarialPolicy:
    return AdversarialPolicy(obs_dim=ADVERSARY_OBS_DIM, device="cpu")


def test_select_action_returns_valid_dict(policy):
    obs = np.random.default_rng(0).standard_normal(ADVERSARY_OBS_DIM).astype(np.float32)
    action = policy.select_action(obs, deterministic=False)
    assert set(action.keys()) >= {"heading", "speed", "altitude"}
    assert np.all(np.abs(action["heading"]) <= 1.0)
    assert np.all(np.abs(action["speed"]) <= 1.0)
    assert int(action["altitude"]) in (0, 1, 2)


def test_deterministic_action_is_reproducible(policy):
    obs = np.zeros(ADVERSARY_OBS_DIM, dtype=np.float32)
    a1 = policy.select_action(obs, deterministic=True)
    a2 = policy.select_action(obs, deterministic=True)
    assert np.allclose(a1["heading"], a2["heading"])
    assert np.allclose(a1["speed"], a2["speed"])
    assert a1["altitude"] == a2["altitude"]


def test_value_function_returns_finite_scalar(policy):
    obs = np.random.default_rng(0).standard_normal(ADVERSARY_OBS_DIM).astype(np.float32)
    value = policy.get_value(obs)
    assert np.isfinite(value)


def test_save_and_load_round_trip(tmp_path, policy):
    obs = np.random.default_rng(0).standard_normal(ADVERSARY_OBS_DIM).astype(np.float32)
    value_before = policy.get_value(obs)

    save_path = tmp_path / "policy.pt"
    policy.save(str(save_path))
    assert save_path.exists()

    loaded = AdversarialPolicy(obs_dim=ADVERSARY_OBS_DIM, device="cpu")
    loaded.load(str(save_path))
    assert loaded.get_value(obs) == pytest.approx(value_before, abs=1e-5)


# ---- MultiAgentEnv ------------------------------------------------------------

@pytest.fixture
def env():
    e = MultiAgentEnv(
        scenario_type="head_on",
        num_intruders=3,
        enable_adversarial=True,
        importance_sampling=True,
    )
    yield e
    e.close()


def _random_protagonist_action(rng):
    return {
        "heading": rng.uniform(-1, 1, size=(1,)),
        "speed": rng.uniform(-1, 1, size=(1,)),
        "altitude": int(rng.integers(0, 3)),
    }


def _random_adversary_actions(env, rng):
    return {
        agent_id: {
            "heading": rng.uniform(-1, 1, size=(1,)),
            "speed": rng.uniform(-1, 1, size=(1,)),
            "altitude": int(rng.integers(0, 3)),
        }
        for agent_id in env.adversarial_agents
    }


def test_reset_provides_protagonist_and_adversary_obs(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert "adversary_obs" in info
    assert len(info["adversary_obs"]) == env.num_intruders
    for adv_obs in info["adversary_obs"].values():
        assert adv_obs.shape == (env.adversary_obs_dim,)


def test_step_returns_protagonist_and_adversary_rewards(env):
    rng = np.random.default_rng(0)
    env.reset()
    _, reward, _, _, info = env.step(
        _random_protagonist_action(rng),
        _random_adversary_actions(env, rng),
    )
    assert np.isfinite(reward)
    assert "adversary_rewards" in info
    assert len(info["adversary_rewards"]) == env.num_intruders
    for r in info["adversary_rewards"].values():
        assert np.isfinite(r)


def test_extreme_actions_do_not_break_simulation(env):
    env.reset()
    extreme_protagonist = {
        "heading": np.array([1.0]), "speed": np.array([1.0]), "altitude": 2,
    }
    extreme_adversaries = {
        agent_id: {"heading": np.array([-1.0]), "speed": np.array([-1.0]), "altitude": 0}
        for agent_id in env.adversarial_agents
    }
    obs, reward, _, _, info = env.step(extreme_protagonist, extreme_adversaries)
    assert np.isfinite(obs).all()
    assert np.isfinite(reward)


def test_importance_ratio_update(env, policy):
    """`update_importance_ratio` 调用后，`info['importance_ratios']` 应包含每个对手。"""
    rng = np.random.default_rng(0)
    _, info = env.reset()

    for agent_id, adv_obs in info["adversary_obs"].items():
        with torch.no_grad():
            obs_tensor = torch.from_numpy(adv_obs).float().unsqueeze(0)
            _, _, _, log_prob = policy.actor.forward(obs_tensor)
        env.update_importance_ratio(agent_id, log_prob.item(), log_prob.item())

    _, _, _, _, info = env.step(
        _random_protagonist_action(rng),
        _random_adversary_actions(env, rng),
    )
    assert "importance_ratios" in info
    for ratio in info["importance_ratios"].values():
        assert np.isfinite(ratio)
        assert ratio >= 0


def test_episode_statistics_have_expected_keys(env):
    rng = np.random.default_rng(0)
    env.reset()
    for _ in range(5):
        env.step(
            _random_protagonist_action(rng),
            _random_adversary_actions(env, rng),
        )
    stats = env.get_episode_statistics()
    assert "protagonist" in stats
    for key in ("goal_reached", "conflicts", "nfz_violations"):
        assert key in stats["protagonist"]
