# scripts

毕设新增（snake_case 命名，与原仓库扁平脚本并列）：

| 路径 | 说明 |
| --- | --- |
| `train/train_adversarial.py` | 对抗训练入口（protagonist / adversary 双模式） |
| `eval/run_comparison_experiments.py` | 自然背景 vs 对抗背景对比实验 |
| `report/generate_phase5_comprehensive_report.py` | 多场景综合报告（自动扫描 `results/comparison_*`） |
| `viz/visualize_comparison.py` | 对比实验可视化 |
| `run_full_retrain.ps1` | 一键重训：4 protagonists → 4 adversaries → 4 comparisons → report |

原仓库脚本保持原样：`common/logger.py`、`descent_env_sac.py`、`vertical_cr_env_sac.py`、`multi_processing_example.py`、`result_plotter.py`、`workshop_pt1.py`、`workshop_pt2.py`。

一键复现：
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_retrain.ps1
```
