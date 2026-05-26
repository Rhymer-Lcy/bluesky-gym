# scripts/

This directory contains training, evaluation, and reporting scripts added in this fork,
alongside the original upstream utility scripts retained without modification.

## Added scripts (this fork)

| Path | Description |
| --- | --- |
| `train/train_adversarial.py` | Adversarial co-evolutionary training. Phase 1 trains a protagonist conflict-resolution policy via SB3 PPO against random background aircraft; Phase 2 freezes the protagonist and trains per-intruder adversary policies (custom Actor-Critic with critical-state filtering and per-step IS-weight accumulation); Phase 3 runs alternating co-evolutionary updates. |
| `eval/run_comparison_experiments.py` | Natural-background vs adversarial-background comparison experiments. Evaluates CDR, GRR, CSR, average episode reward, episode length, and sample efficiency (IS-corrected required-episode estimates) over 200 evaluation episodes per condition. |
| `report/generate_phase5_comprehensive_report.py` | Aggregates comparison results across all four conflict scenarios by scanning `results/comparison_*/` directories and produces publication-ready figures and a summary CSV. |
| `viz/visualize_comparison.py` | Generates per-scenario figures from comparison experiment JSON outputs: training curves, metric bar charts, performance radar chart, and critical-state frequency distribution. |
| `run_full_retrain.ps1` | End-to-end pipeline: trains four protagonist policies, four sets of adversary policies, runs four comparison experiments, and generates the comprehensive report. Supports `-StartStage <n>` to resume from a specific stage without rerunning earlier stages. |

## Upstream scripts (retained unchanged)

`common/logger.py`, `descent_env_sac.py`, `vertical_cr_env_sac.py`,
`multi_processing_example.py`, `result_plotter.py`, `workshop_pt1.py`, `workshop_pt2.py`.

## Reproducing experiments

Run the full pipeline from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_retrain.ps1
```

To resume from a specific stage (e.g., skip protagonist training if checkpoints already exist):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_retrain.ps1 -StartStage 2
```
