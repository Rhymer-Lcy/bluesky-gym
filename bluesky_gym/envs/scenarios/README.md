# Conflict Scenario Generators

This module provides four canonical UAV conflict scenario generators built on
top of BlueSky's `creconfs()` command.

## Implemented scenarios

### 1. Head-On Scenario
- **Description**: Two aircraft flying toward each other head-on (track angle ~180°)
- **Severity**: High
- **Class**: `HeadOnScenario`
- **Default ranges**:
  - Conflict angle: 150° – 210°
  - CPA distance: 0 – 3 NM
  - Time to CPA: 60 – 180 s

### 2. Crossing Scenario
- **Description**: Two aircraft crossing paths at ~90° (intersection-like geometry)
- **Severity**: Medium
- **Class**: `CrossingScenario`
- **Default ranges**:
  - Conflict angle: 60° – 120°
  - CPA distance: 0 – 3 NM
  - Time to CPA: 60 – 180 s

### 3. Merging Scenario
- **Description**: Intruder merging into the ownship's route from the side (30°–60°)
- **Severity**: Low
- **Class**: `MergingScenario`
- **Default ranges**:
  - Conflict angle: 30° – 60°
  - CPA distance: 0 – 2 NM
  - Time to CPA: 60 – 180 s

### 4. Overtaking Scenario
- **Description**: Faster intruder overtaking a slower ownship from behind (0°–20°)
- **Severity**: Low
- **Class**: `OvertakingScenario`
- **Default ranges**:
  - Conflict angle: 0° – 20°
  - CPA distance: 0 – 2 NM
  - Time to CPA: 100 – 300 s
  - Speed differential: +10 – +30 m/s

## Usage

### Option 1: Direct instantiation

```python
import bluesky as bs
from bluesky_gym.envs.scenarios import HeadOnScenario

bs.init(mode='sim', detached=True)

bs.traf.cre('AC0', actype='A320', aclat=52.0, aclon=4.0,
            achdg=0, acalt=3000, acspd=150)

scenario = HeadOnScenario(num_intruders=3)
intruders = scenario.generate(target_acid='AC0')

print(f"Generated intruders: {intruders}")
# Output: Generated intruders: ['INTRUDER_1', 'INTRUDER_2', 'INTRUDER_3']

for i in range(200):
    bs.sim.step()
```

### Option 2: Scenario factory

```python
from bluesky_gym.envs.scenarios import get_scenario

ScenarioClass = get_scenario('crossing')  # 'head_on', 'crossing', 'merging', 'overtaking'
scenario = ScenarioClass(num_intruders=5)
intruders = scenario.generate(target_acid='AC0')
```

### Option 3: Custom parameters

```python
from bluesky_gym.envs.scenarios import HeadOnScenario

scenario = HeadOnScenario(
    num_intruders=5,
    dpsi_range=(170, 190),     # narrower angle range (closer to pure head-on)
    dcpa_range=(0, 1),         # smaller CPA distance (more dangerous)
    tlosh_range=(30, 60),      # shorter time to CPA (more urgent)
    speed_range=(150, 180),    # explicit speed range
    altitude_range=(-100, 100) # closer altitude separation
)

intruders = scenario.generate(target_acid='AC0', actype='B738')
```

## Use inside a Gym environment

```python
import numpy as np
import gymnasium as gym
import bluesky as bs
from bluesky_gym.envs.scenarios import get_scenario

class DroneConflictEnv(gym.Env):
    def reset(self):
        bs.traf.reset()

        bs.traf.cre('OWNSHIP', actype='A320', aclat=52.0, aclon=4.0,
                    achdg=0, acalt=3000, acspd=150)

        scenario_type = np.random.choice(['head_on', 'crossing', 'merging', 'overtaking'])
        ScenarioClass = get_scenario(scenario_type)
        scenario = ScenarioClass(num_intruders=3)
        self.intruders = scenario.generate(target_acid='OWNSHIP')

        return self._get_obs(), {}
```

## Parameters

### Constructor parameters

| Parameter | Type | Description | Example |
|---|---|---|---|
| `num_intruders` | int | Number of intruder aircraft | 3 |
| `dpsi_range` | tuple | Conflict angle range [deg] | (150, 210) |
| `dcpa_range` | tuple | CPA distance range [NM] | (0, 3) |
| `tlosh_range` | tuple | Time-to-CPA range [s] | (60, 180) |
| `speed_range` | tuple\|None | Speed range [m/s]; None = auto | (100, 200) |
| `altitude_range` | tuple | Relative altitude range [m] | (-500, 500) |

### `generate()` parameters

| Parameter | Type | Description | Example |
|---|---|---|---|
| `target_acid` | str | Callsign of the ownship | 'AC0' |
| `actype` | str | Aircraft type for intruders | 'A320' |

## Advanced: mixed scenario

```python
from bluesky_gym.envs.scenarios import HeadOnScenario, CrossingScenario

bs.traf.cre('AC0', actype='A320', aclat=52.0, aclon=4.0,
            achdg=0, acalt=3000, acspd=150)

head_on  = HeadOnScenario(num_intruders=2)
crossing = CrossingScenario(num_intruders=2)

intruders1 = head_on.generate(target_acid='AC0')
intruders2 = crossing.generate(target_acid='AC0')

print(f"Total intruders: {len(intruders1) + len(intruders2)}")
```

## Implementation note

All scenario generators delegate to BlueSky's `creconfs()` command, which
automatically back-calculates the intruder's initial position, speed, and
heading so that the conflict geometry matches the requested parameters:

```python
bs.traf.creconfs(
    acid='INTRUDER_1',
    actype='A320',
    targetidx=0,   # index of the ownship
    dpsi=180,      # conflict angle [deg]
    dcpa=2,        # CPA distance [NM]
    tlosh=120,     # time to CPA [s]
    dH=0,          # relative altitude [m]
    spd=150        # speed [m/s]
)
```

## Notes

1. **Ownship must exist first**: call `bs.traf.cre()` before `generate()`.
2. **Intruder callsigns are auto-generated**: `INTRUDER_1`, `INTRUDER_2`, …
3. **Conflict detection latency**: allow at least 50 simulation steps before
   checking for active conflicts.

## References

- [BlueSky documentation](https://github.com/TUDelft-CNS-ATM/bluesky/wiki)
- BlueSky built-in `creconfs()` interface and conflict detection logic
- Scenario validation scripts in `examples/` and experiment scripts in `scripts/`
