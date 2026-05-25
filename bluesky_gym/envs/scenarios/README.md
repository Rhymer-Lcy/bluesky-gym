# 典型冲突场景生成器

本模块提供4种典型无人机冲突场景的自动生成器，基于 BlueSky 的 `creconfs()` 函数。

## 📦 已实现的场景

### 1. 对头场景 (Head-On Scenario)
- **特点**：两机相向飞行，正面相遇（航迹夹角180°）
- **危险等级**：⚠️⚠️⚠️ 最高
- **类名**：`HeadOnScenario`
- **典型参数**：
  - 冲突角度：150° ~ 210°
  - 最近距离：0 ~ 3 NM
  - 时间范围：60 ~ 180秒

### 2. 交叉场景 (Crossing Scenario)
- **特点**：两机交叉通过，类似十字路口（航迹夹角90°）
- **危险等级**：⚠️⚠️ 中等
- **类名**：`CrossingScenario`
- **典型参数**：
  - 冲突角度：60° ~ 120°
  - 最近距离：0 ~ 3 NM
  - 时间范围：60 ~ 180秒

### 3. 汇入场景 (Merging Scenario)
- **特点**：入侵者从侧向汇入主航路（航迹夹角30°-60°）
- **危险等级**：⚠️ 较低
- **类名**：`MergingScenario`
- **典型参数**：
  - 冲突角度：30° ~ 60°
  - 最近距离：0 ~ 2 NM
  - 时间范围：60 ~ 180秒

### 4. 追尾场景 (Overtaking Scenario)
- **特点**：后机速度大于前机，逐渐接近（航迹夹角0°-20°）
- **危险等级**：⚠️ 较低
- **类名**：`OvertakingScenario`
- **典型参数**：
  - 冲突角度：0° ~ 20°
  - 最近距离：0 ~ 2 NM
  - 时间范围：100 ~ 300秒（较长）
  - 速度差：+10 ~ +30 m/s

## 🚀 使用方法

### 方法1：直接使用场景类

```python
import bluesky as bs
from bluesky_gym.envs.scenarios import HeadOnScenario

# 初始化 BlueSky
bs.init(mode='sim', detached=True)

# 创建目标飞行器
bs.traf.cre('AC0', actype='A320', aclat=52.0, aclon=4.0, 
            achdg=0, acalt=3000, acspd=150)

# 创建对头场景（3个入侵者）
scenario = HeadOnScenario(num_intruders=3)
intruders = scenario.generate(target_acid='AC0')

print(f"生成的入侵者: {intruders}")
# 输出: 生成的入侵者: ['INTRUDER_1', 'INTRUDER_2', 'INTRUDER_3']

# 运行仿真
for i in range(200):
    bs.sim.step()
```

### 方法2：使用场景工厂

```python
from bluesky_gym.envs.scenarios import get_scenario

# 根据场景名称获取场景类
ScenarioClass = get_scenario('crossing')  # 'head_on', 'crossing', 'merging', 'overtaking'
scenario = ScenarioClass(num_intruders=5)
intruders = scenario.generate(target_acid='AC0')
```

### 方法3：自定义参数

```python
from bluesky_gym.envs.scenarios import HeadOnScenario

# 自定义场景参数
scenario = HeadOnScenario(
    num_intruders=5,               # 5个入侵者
    dpsi_range=(170, 190),         # 更窄的角度范围（更接近正对头）
    dcpa_range=(0, 1),             # 更小的最近距离（更危险）
    tlosh_range=(30, 60),          # 更短的时间（更紧急）
    speed_range=(150, 180),        # 指定速度范围
    altitude_range=(-100, 100)     # 高度更接近
)

intruders = scenario.generate(target_acid='AC0', actype='B738')
```

## 🎯 在 Gym 环境中使用

```python
import gymnasium as gym
import bluesky as bs
from bluesky_gym.envs.scenarios import HeadOnScenario

class DroneConflictEnv(gym.Env):
    def reset(self):
        bs.traf.reset()
        
        # 创建被控飞行器
        bs.traf.cre('OWNSHIP', actype='A320', aclat=52.0, aclon=4.0, 
                    achdg=0, acalt=3000, acspd=150)
        
        # 随机选择场景类型
        scenario_type = np.random.choice(['head_on', 'crossing', 'merging', 'overtaking'])
        ScenarioClass = get_scenario(scenario_type)
        scenario = ScenarioClass(num_intruders=3)
        
        # 生成冲突场景
        self.intruders = scenario.generate(target_acid='OWNSHIP')
        
        return self._get_obs(), {}
```

## 📊 场景参数说明

### 通用参数

| 参数 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| `num_intruders` | int | 入侵者数量 | 3 |
| `dpsi_range` | tuple | 冲突角度范围[deg] | (150, 210) |
| `dcpa_range` | tuple | 最近接近距离范围[NM] | (0, 3) |
| `tlosh_range` | tuple | 到达最近点时间范围[s] | (60, 180) |
| `speed_range` | tuple\|None | 速度范围[m/s]，None表示自动 | (100, 200) |
| `altitude_range` | tuple | 相对高度范围[m] | (-500, 500) |

### 生成方法参数

| 参数 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| `target_acid` | str | 目标飞行器呼号 | 'AC0' |
| `actype` | str | 入侵者飞行器类型 | 'A320' |

## 🔧 高级用法

### 创建混合场景

```python
from bluesky_gym.envs.scenarios import HeadOnScenario, CrossingScenario

# 创建目标飞行器
bs.traf.cre('AC0', actype='A320', aclat=52.0, aclon=4.0, 
            achdg=0, acalt=3000, acspd=150)

# 同时生成多种场景
head_on = HeadOnScenario(num_intruders=2)
crossing = CrossingScenario(num_intruders=2)

intruders1 = head_on.generate(target_acid='AC0')
intruders2 = crossing.generate(target_acid='AC0')

print(f"总共 {len(intruders1) + len(intruders2)} 个入侵者")
```

### 查看场景描述

```python
scenario = HeadOnScenario()
print(scenario.get_description())
```

输出：
```
对头冲突场景 (Head-On Scenario)
================================
入侵者数量: 3
冲突角度范围: 150° ~ 210° (中心180°)
最近距离范围: 0 ~ 3 NM
时间范围: 60 ~ 180 秒
...
```

## ⚙️ 实现原理

所有场景生成器都基于 BlueSky 的 `creconfs()` 函数：

```python
bs.traf.creconfs(
    acid='INTRUDER_1',     # 入侵者呼号
    actype='A320',         # 飞行器类型
    targetidx=0,           # 目标飞行器索引
    dpsi=180,              # 冲突角度[deg]
    dcpa=2,                # 最近接近距离[NM]
    tlosh=120,             # 到达最近点时间[s]
    dH=0,                  # 相对高度[m]
    spd=150                # 速度[m/s]
)
```

`creconfs()` 会自动计算入侵者的初始位置、速度、航向，使其与目标飞行器在指定时间、距离产生冲突。

## 📝 注意事项

1. **必须先创建目标飞行器**：在调用 `generate()` 前，目标飞行器必须已存在
2. **入侵者呼号自动生成**：格式为 `INTRUDER_1`, `INTRUDER_2`, ...
3. **参数范围验证**：确保参数范围合理，避免物理不可行的配置
4. **冲突检测延迟**：冲突检测有一定延迟，建议运行至少50步再检查

## 🧪 测试示例

运行测试脚本：

```bash
python examples/test_scenarios.py
```

查看各场景的生成效果和冲突检测结果。

## 📚 参考资料

- [BlueSky 文档](https://github.com/TUDelft-CNS-ATM/bluesky/wiki)
- BlueSky 内置 `creconfs()` 接口与冲突检测逻辑
- 本仓库 `examples/` 下的场景验证脚本与 `scripts/` 下的实验脚本
