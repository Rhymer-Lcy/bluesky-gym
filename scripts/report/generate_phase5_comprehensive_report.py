"""
Phase 5 comprehensive experiment report generator.
Aggregates Natural vs Adversarial comparison results across 4 scenarios
and produces publication-ready charts.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# matplotlib font config
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# scenario display names
SCENARIO_NAMES = {
    'head_on': 'Head-On',
    'crossing': 'Crossing',
    'merging': 'Merging',
    'overtaking': 'Overtaking'
}

# metric display names
METRIC_NAMES = {
    'avg_reward': 'Avg Reward',
    'conflict_discovery_rate': 'Conflict Discovery Rate',
    'critical_state_ratio': 'Critical State Ratio',
    'goal_reach_rate': 'Goal Reach Rate',
    'avg_episode_length': 'Avg Episode Length'
}




def _scan_results_by_scenario(base_dir='results'):
    """Scan results/ directory and return {scenario: [Path, ...]} sorted by mtime."""
    results_dir = Path(base_dir)
    mapping = {}
    for item in sorted(results_dir.glob('comparison_*')):
        if not item.is_dir():
            continue
        results_file = item / 'results.json'
        if not results_file.exists():
            continue
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sc = (data.get('config') or {}).get('scenario') or \
                 (data.get('natural', {}).get('config') or {}).get('scenario')
            if sc:
                mapping.setdefault(sc, []).append(item)
        except Exception:
            continue
    return mapping


def _find_timestamp_for_scenario(scenario, base_dir='results'):
    """Return timestamp string for the latest comparison dir of *scenario*, or None."""
    mapping = _scan_results_by_scenario(base_dir)
    dirs = mapping.get(scenario, [])
    if not dirs:
        return None
    latest = max(dirs, key=lambda p: p.stat().st_mtime)
    return latest.name.replace('comparison_', '')


def find_latest_comparison_result(scenario, base_dir='results'):
    """Find the latest comparison result directory for *scenario*."""
    mapping = _scan_results_by_scenario(base_dir)
    dirs = mapping.get(scenario, [])
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def load_scenario_results(scenario, base_dir='results'):
    """Load comparison results for a single scenario."""
    result_dir = find_latest_comparison_result(scenario, base_dir)
    
    if result_dir is None:
        print(f"Warning: no results found for scenario {scenario}")
        return None
    
    results_file = result_dir / 'results.json'
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # transform data format to canonical structure
        if 'natural' in data and 'adversarial' in data:
            # new format: transform to canonical structure
            transformed_data = {
                'scenario': scenario,
                'timestamp': data.get('timestamp', ''),
                'natural_background': {
                    'avg_reward': data['natural']['evaluation']['avg_reward'],
                    'std_reward': data['natural']['evaluation']['std_reward'],
                    'avg_episode_length': data['natural']['evaluation']['avg_length'],
                    'conflict_discovery_rate': data['natural']['evaluation']['conflict_rate'] * 100,
                    'critical_state_ratio': data['natural']['evaluation']['avg_critical_ratio'] * 100,
                    'goal_reach_rate': data['natural']['evaluation']['goal_rate'] * 100
                },
                'adversarial_background': {
                    'avg_reward': data['adversarial']['evaluation']['avg_reward'],
                    'std_reward': data['adversarial']['evaluation']['std_reward'],
                    'avg_episode_length': data['adversarial']['evaluation']['avg_length'],
                    'conflict_discovery_rate': data['adversarial']['evaluation']['conflict_rate'] * 100,
                    'critical_state_ratio': data['adversarial']['evaluation']['avg_critical_ratio'] * 100,
                    'goal_reach_rate': data['adversarial']['evaluation']['goal_rate'] * 100
                }
            }
            print(f"OK loaded {scenario} results: {result_dir.name}")
            return transformed_data
        else:
            # old format: use directly
            print(f"OK loaded {scenario} results: {result_dir.name}")
            return data
            
    except Exception as e:
        print(f"Error: failed to load {scenario} results: {e}")
        return None


def extract_tensorboard_metrics(log_dir, metrics=['rollout/ep_rew_mean']):
    """Extract metrics from TensorBoard event files."""
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    data = {}
    for metric in metrics:
        if metric in ea.Tags()['scalars']:
            events = ea.Scalars(metric)
            data[metric] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events]
            }
    
    return data


def load_training_curves(scenario, base_dir='logs'):
    """Load training curve data for *scenario*."""
    logs_dir = Path(base_dir)

    # resolve timestamp from results/ scan
    timestamp = _find_timestamp_for_scenario(scenario)
    if timestamp is None:
        print(f"Warning: no comparison results found for scenario {scenario}")
        return None, None

    # build log paths from timestamp
    natural_dir = logs_dir / f'comparison_natural_{timestamp}' / 'PPO_1'
    adversarial_dir = logs_dir / f'comparison_adversarial_{timestamp}' / 'PPO_1'
    
    if not natural_dir.exists() or not adversarial_dir.exists():
        print(f"Warning: training logs not found for scenario {scenario}")
        print(f"  expected: {natural_dir.parent.name}, {adversarial_dir.parent.name}")
        return None, None
    
    try:
        natural_data = extract_tensorboard_metrics(
            natural_dir,
            ['rollout/ep_rew_mean', 'rollout/ep_len_mean']
        )
        adversarial_data = extract_tensorboard_metrics(
            adversarial_dir,
            ['rollout/ep_rew_mean', 'rollout/ep_len_mean']
        )
        
        print(f"OK loaded {scenario} training curves")
        return natural_data, adversarial_data
    except Exception as e:
        print(f"Error: failed to load {scenario} training curves: {e}")
        return None, None


def create_radar_chart(all_results, output_path):
    """Create a radar chart comparing key metrics across all scenarios."""
    scenarios = ['head_on', 'crossing', 'merging', 'overtaking']
    metrics = ['avg_reward', 'conflict_discovery_rate', 'critical_state_ratio', 
               'goal_reach_rate']
    
    # prepare data
    natural_values = []
    adversarial_values = []
    
    for scenario in scenarios:
        if scenario not in all_results or all_results[scenario] is None:
            natural_values.append([0, 0, 0, 0])
            adversarial_values.append([0, 0, 0, 0])
            continue
        
        data = all_results[scenario]
        nat = data['natural_background']
        adv = data['adversarial_background']
        
        # normalise to [0, 1] (reward assumed range -10 to +10)
        nat_vals = [
            (nat['avg_reward'] + 10) / 20,
            nat['conflict_discovery_rate'] / 100,
            nat['critical_state_ratio'] / 100,
            nat['goal_reach_rate'] / 100
        ]
        adv_vals = [
            (adv['avg_reward'] + 10) / 20,
            adv['conflict_discovery_rate'] / 100,
            adv['critical_state_ratio'] / 100,
            adv['goal_reach_rate'] / 100
        ]
        
        natural_values.append(nat_vals)
        adversarial_values.append(adv_vals)
    
    # create 4 subplots (one radar per scenario)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                            subplot_kw=dict(projection='polar'))
    fig.suptitle('Natural vs Adversarial Radar Chart (4 Scenarios)', fontsize=16, y=0.98)
    
    labels = ['Avg Reward', 'Conflict Rate', 'Critical State', 'Goal Reach']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # close polygon
    
    for idx, (ax, scenario) in enumerate(zip(axes.flat, scenarios)):
        if all_results.get(scenario) is None:
            ax.text(0.5, 0.5, f'{SCENARIO_NAMES[scenario]}\nNo data',
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        nat_vals = natural_values[idx] + natural_values[idx][:1]
        adv_vals = adversarial_values[idx] + adversarial_values[idx][:1]
        
        ax.plot(angles, nat_vals, 'o-', linewidth=2, label='Natural', color='#3498db')
        ax.fill(angles, nat_vals, alpha=0.15, color='#3498db')
        
        ax.plot(angles, adv_vals, 's-', linewidth=2, label='Adversarial', color='#e74c3c')
        ax.fill(angles, adv_vals, alpha=0.15, color='#e74c3c')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(SCENARIO_NAMES[scenario], pad=20, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved radar chart: {output_path}")
    plt.close()


def create_metrics_comparison_bars(all_results, output_path):
    """Create bar charts comparing key metrics across scenarios."""
    scenarios = ['head_on', 'crossing', 'merging', 'overtaking']
    metrics = [
        ('conflict_discovery_rate', 'Conflict Discovery Rate (%)', 100),
        ('critical_state_ratio', 'Critical State Ratio (%)', 100),
        ('goal_reach_rate', 'Goal Reach Rate (%)', 100)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Key Metrics: Natural vs Adversarial (4 Scenarios)', fontsize=16, y=1.02)

    x = np.arange(len(scenarios))
    width = 0.35

    for ax, (metric, title, scale) in zip(axes, metrics):
        natural_vals = []
        adversarial_vals = []

        for scenario in scenarios:
            if scenario in all_results and all_results[scenario] is not None:
                data = all_results[scenario]
                natural_vals.append(data['natural_background'][metric] * scale)
                adversarial_vals.append(data['adversarial_background'][metric] * scale)
            else:
                natural_vals.append(0)
                adversarial_vals.append(0)

        bars1 = ax.bar(x - width/2, natural_vals, width, label='Natural',
                      color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, adversarial_vals, width, label='Adversarial',
                      color='#e74c3c', alpha=0.8)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Scenario', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_NAMES[s] for s in scenarios], rotation=15)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved bar chart: {output_path}")
    plt.close()


def create_training_curves(all_training_data, output_path):
    """Create training curve comparison plots."""
    scenarios = ['head_on', 'crossing', 'merging', 'overtaking']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Training Curves: Natural vs Adversarial (4 Scenarios)', fontsize=16, y=0.995)

    for ax, scenario in zip(axes.flat, scenarios):
        if scenario not in all_training_data or all_training_data[scenario] is None:
            ax.text(0.5, 0.5, f'{SCENARIO_NAMES[scenario]}\nNo training data',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        natural_data, adversarial_data = all_training_data[scenario]

        if natural_data and 'rollout/ep_rew_mean' in natural_data:
            nat_steps = natural_data['rollout/ep_rew_mean']['steps']
            nat_rewards = natural_data['rollout/ep_rew_mean']['values']
            ax.plot(nat_steps, nat_rewards, label='Natural',
                   color='#3498db', linewidth=2, alpha=0.8)

        if adversarial_data and 'rollout/ep_rew_mean' in adversarial_data:
            adv_steps = adversarial_data['rollout/ep_rew_mean']['steps']
            adv_rewards = adversarial_data['rollout/ep_rew_mean']['values']
            ax.plot(adv_steps, adv_rewards, label='Adversarial',
                   color='#e74c3c', linewidth=2, alpha=0.8)

        ax.set_xlabel('Training steps', fontsize=10)
        ax.set_ylabel('Mean episode reward', fontsize=10)
        ax.set_title(SCENARIO_NAMES[scenario], fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves: {output_path}")
    plt.close()


def create_improvement_table(all_results):
    """Create improvement metrics summary table."""
    scenarios = ['head_on', 'crossing', 'merging', 'overtaking']

    table_data = []
    for scenario in scenarios:
        if scenario not in all_results or all_results[scenario] is None:
            continue

        data = all_results[scenario]
        nat = data['natural_background']
        adv = data['adversarial_background']

        conflict_improve = adv['conflict_discovery_rate'] - nat['conflict_discovery_rate']
        critical_improve = adv['critical_state_ratio'] - nat['critical_state_ratio']
        goal_change = adv['goal_reach_rate'] - nat['goal_reach_rate']
        reward_change = adv['avg_reward'] - nat['avg_reward']

        table_data.append({
            'Scenario': SCENARIO_NAMES[scenario],
            'Conflict Disc. Gain (%)': f"{conflict_improve:+.1f}",
            'Critical State Gain (%)': f"{critical_improve:+.1f}",
            'Goal Reach Change (%)': f"{goal_change:+.1f}",
            'Reward Change': f"{reward_change:+.2f}"
        })

    df = pd.DataFrame(table_data)
    return df


def generate_comprehensive_report():
    """Generate the comprehensive Phase 5 report."""
    print("\n" + "="*70)
    print("Phase 5 Comprehensive Report Generator")
    print("="*70 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f'results/phase5_comprehensive_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    print("Step 1/5: loading experiment results...")
    scenarios = ['head_on', 'crossing', 'merging', 'overtaking']
    all_results = {}

    for scenario in scenarios:
        result = load_scenario_results(scenario)
        all_results[scenario] = result

    print()

    print("Step 2/5: loading training curves...")
    all_training_data = {}
    for scenario in scenarios:
        natural_data, adversarial_data = load_training_curves(scenario)
        all_training_data[scenario] = (natural_data, adversarial_data)

    print()

    print("Step 3/5: generating charts...")
    create_radar_chart(all_results, output_dir / 'radar_chart.png')
    create_metrics_comparison_bars(all_results, output_dir / 'metrics_bars.png')
    create_training_curves(all_training_data, output_dir / 'training_curves.png')
    
    print()

    # build improvement table
    print("Step 4/5: building improvement table...")
    improvement_df = create_improvement_table(all_results)
    print("\n" + "="*70)
    print("Adversarial Training Improvement Summary")
    print("="*70)
    print(improvement_df.to_string(index=False))
    print("="*70 + "\n")

    improvement_df.to_csv(output_dir / 'improvement_summary.csv',
                         index=False, encoding='utf-8-sig')

    print("Step 5/5: saving comprehensive results...")
    comprehensive_results = {
        'timestamp': timestamp,
        'scenarios': all_results,
        'summary': {
            'total_scenarios': len([s for s in all_results.values() if s is not None]),
            'completed_scenarios': [s for s in scenarios if all_results.get(s) is not None]
        }
    }

    with open(output_dir / 'comprehensive_results.json', 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_dir / 'comprehensive_results.json'}")
    print(f"Saved: {output_dir / 'improvement_summary.csv'}")

    print("\n" + "="*70)
    print("Report generation complete.")
    print("="*70)
    print(f"\nAll files saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - radar_chart.png              (4-scenario radar chart)")
    print("  - metrics_bars.png             (key metrics bar chart)")
    print("  - training_curves.png          (training curve comparison)")
    print("  - improvement_summary.csv      (improvement metrics table)")
    print("  - comprehensive_results.json   (comprehensive results data)")
    print()


if __name__ == '__main__':
    generate_comprehensive_report()
