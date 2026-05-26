"""BlueSky-Gym upstream example: minimal train + eval loop for a single environment.

This file originates from the upstream repository (TUDelft-CNS-ATM/bluesky-gym).
It is kept as a quick smoke-test to verify the installation and demonstrate the
standard Gymnasium API. Change ``env_name`` / ``algorithm`` to switch environments
and algorithms.

The thesis experiments live under ``scripts/``, not here:

* ``scripts/train/train_adversarial.py``                protagonist + adversary training
* ``scripts/eval/run_comparison_experiments.py``        natural vs adversarial comparison
* ``scripts/report/generate_phase5_comprehensive_report.py``  figure generation
* automated tests: ``pytest tests/``
* visualisation demos: ``examples/`` pygame / matplotlib scripts

NOTE (upstream gap):
* rgb_array rendering is not yet implemented in upstream environments; video saving is unsupported
"""

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DDPG

import numpy as np

import bluesky_gym
import bluesky_gym.envs

from bluesky_gym.utils import logger

bluesky_gym.register_envs()

env_name = 'StaticObstacleEnv-v0'
algorithm = SAC

# Initialize logger
log_dir = f'./logs/{env_name}/'
file_name = f'{env_name}_{str(algorithm.__name__)}.csv'
csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

TRAIN = False
EVAL_EPISODES = 10


if __name__ == "__main__":
    env = gym.make(env_name, render_mode=None)
    obs, info = env.reset()
    model = algorithm("MultiInputPolicy", env, verbose=1,learning_rate=3e-4)
    if TRAIN:
        model.learn(total_timesteps=2e6, callback=csv_logger_callback)
        model.save(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model")
        del model
    env.close()
    
    # Test the trained model
    model = algorithm.load(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model", env=env)
    env = gym.make(env_name, render_mode="human")
    for i in range(EVAL_EPISODES):

        done = truncated = False
        obs, info = env.reset()
        tot_rew = 0
        while not (done or truncated):
            # action = np.array(np.random.randint(-100,100,size=(2))/100)
            # action = np.array([0,-1])
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action[()])
            tot_rew += reward
        print(tot_rew)
    env.close()