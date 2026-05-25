# 演示脚本 (examples/)

本目录提供面向人类用户的可视化与交互演示。所有脚本可独立运行，不依赖 pytest。

> 自动化测试请使用根目录 `tests/`，运行 `pytest -v`。

## 演示脚本一览

| 脚本 | 演示内容 | 运行方式 |
|---|---|---|
| `disturbance_visualization.py` | 不同 `disturbance preset` 下飞机轨迹偏离量的 matplotlib 对比图 | `python examples/disturbance_visualization.py` |
| `disturbance_pygame_demo.py`   | 实时 pygame 可视化扰动对单机航迹的影响（可切换 preset） | `python examples/disturbance_pygame_demo.py` |
| `scenarios_pygame_demo.py`     | 循环展示 4 类典型冲突场景（head-on / crossing / merging / overtaking） | `python examples/scenarios_pygame_demo.py` |
| `no_fly_zone_pygame_demo.py`   | 圆形 + 多边形禁飞区与多机违规检测的实时渲染 | `python examples/no_fly_zone_pygame_demo.py` |

## 通用要求

- 已通过 `pip install -e .` 安装本仓库与 `bluesky-simulator`
- pygame 演示需要桌面环境（无头服务器请改为 `tests/` 中的 pytest 验证）

## 控制方式（pygame 演示）

- 关闭窗口 / `ESC`：退出
- 详细按键说明见各脚本顶部 docstring
