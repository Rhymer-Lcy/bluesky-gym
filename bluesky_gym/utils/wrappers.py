"""Gymnasium action-space wrappers for BlueSky-Gym environments."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlattenDictActionWrapper(gym.Wrapper):
    """Flatten a Dict action space to a continuous Box for SB3 compatibility.

    Converts the 3-component Dict action space
    ``{"heading": Box(1,), "speed": Box(1,), "altitude": Discrete(3)}``
    used by ConflictResolutionEnv and MultiAgentEnv into a flat
    ``Box(-1, 1, shape=(3,))`` that SB3 PPO/SAC/TD3 can drive directly.

    The inverse mapping applied in ``step`` is::

        env["heading"]  = action[0:1]                          (pass-through)
        env["speed"]    = action[1:2]                          (pass-through)
        env["altitude"] = clip(round(action[2] + 1.0), 0, 2)  ([-1,1] → {0,1,2})

    Works with both ConflictResolutionEnv (single-agent) and MultiAgentEnv
    (whose ``step`` accepts an optional ``adversary_actions`` defaulting to
    ``None``).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = env.observation_space

    def step(self, action):
        dict_action = {
            "heading": np.array([action[0]], dtype=np.float64),
            "speed": np.array([action[1]], dtype=np.float64),
            "altitude": int(np.clip(np.round(action[2] + 1.0), 0, 2)),
        }
        return self.env.step(dict_action)
