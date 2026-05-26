# BlueSky-Gym

A Gymnasium-compatible reinforcement learning library for Air Traffic Management research, built on [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) and the Farama Foundation's [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

<p align="center">
    <img src="https://github.com/user-attachments/assets/6ae83579-78af-4cb7-8096-3a10af54a5c5" width=50% height=50%><br/>
    <em>A trained agent in the merge environment.</em>
</p>

This repository is a research fork of [TUDelft-CNS-ATM/bluesky-gym](https://github.com/TUDelft-CNS-ATM/bluesky-gym). Additions relative to the upstream release:

- **Extended BlueSky submodule** — `bluesky/traffic/` augmented with `disturbance.py` (stochastic wind disturbance model) and `no_fly_zone.py` (dynamic no-fly zone enforcement).
- **Additional environments** — `ConflictResolutionEnv-v0`, `Discrete25DEnv-v0`, `MultiAgentEnv-v0`.
- **Training and evaluation scripts** — adversarial co-evolutionary training, multi-scenario comparison experiments, and automated report generation (see [`scripts/README.md`](scripts/README.md)).

For the full list of available environments, see [`bluesky_gym/envs/README.md`](bluesky_gym/envs/README.md).

---

## Installation

### Standard (upstream PyPI package)

```bash
pip install bluesky-gym
```

The PyPI package name is `bluesky-gym`; the importable module name is `bluesky_gym`.

### From source (this fork)

This fork bundles a customised BlueSky branch as a Git submodule under `bluesky/`. The submodule **must** be installed instead of the upstream `bluesky-simulator` package; without it, `bs.traf.disturb` and `bs.traf.nfz` will be unavailable at runtime.

```bash
git clone --recurse-submodules https://github.com/Rhymer-Lcy/bluesky-gym.git
cd bluesky-gym

# 1. Install the bundled BlueSky submodule (required)
pip install -e bluesky/

# 2. Install bluesky-gym in editable mode
pip install -e .
```

---

## Usage

All environments conform to the standard Gymnasium API:

```python
import gymnasium as gym
import bluesky_gym

bluesky_gym.register_envs()

env = gym.make('MergeEnv-v0', render_mode='human')
obs, info = env.reset()
done = truncated = False

while not (done or truncated):
    action = ...  # agent policy
    obs, reward, done, truncated, info = env.step(action)
```

Integration with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) or [RLlib](https://docs.ray.io/en/latest/rllib/index.html) follows the same pattern:

```python
import gymnasium as gym
import bluesky_gym
from stable_baselines3 import DDPG

bluesky_gym.register_envs()

env = gym.make('MergeEnv-v0', render_mode=None)
model = DDPG("MultiInputPolicy", env)
model.learn(total_timesteps=2_000_000)
model.save("ddpg_merge")
```

For background on the library design and environment construction, refer to the [workshop slides](https://docs.google.com/presentation/d/1Jpwdrx__OMdgHWtQ1yCVQyxsdDFk2ieX/edit?usp=drive_link&ouid=109800667545002770848&rtpof=true&sd=true).

---

## Contributing

Contributions and assistance requests are welcome via GitHub Issues or the BlueSky-Gym [Discord](https://discord.gg/s7CdxcSX). The upstream [roadmap](https://github.com/TUDelft-CNS-ATM/bluesky-gym/issues/24) outlines planned development directions.

---

## Citation

```bibtex
@misc{bluesky-gym,
  author  = {Groot, DJ and Leto, G and Vlaskin, A and Moec, A and Ellerbroek, J},
  title   = {BlueSky-Gym: Reinforcement Learning Environments for Air Traffic Applications},
  year    = {2024},
  journal = {SESAR Innovation Days 2024},
}
```

Publications using BlueSky-Gym (open a pull request to add entries):
- _none listed_
