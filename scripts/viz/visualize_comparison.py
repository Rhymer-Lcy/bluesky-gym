"""
Stage 5: Visualization Script for Comparison Experiments

Generates publication-ready figures for thesis:
1. Training curves comparison
2. Metric bar charts
3. Performance radar chart
4. Critical state distribution
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Use non-interactive backend for saving figures
matplotlib.use('Agg')

# Set font for Chinese characters (if available)
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# Style settings for publication
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'natural': '#3498db',      # Blue
    'adversarial': '#e74c3c',  # Red
    'highlight': '#2ecc71'     # Green
}


def load_results(results_path):
    """Load experiment results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_training_curves(natural_rewards, adversarial_rewards, output_path):
    """Plot training reward curves comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Smooth the curves
    def smooth(data, window=10):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    nat_smooth = smooth(natural_rewards)
    adv_smooth = smooth(adversarial_rewards)
    
    ax.plot(nat_smooth, label='Natural Background', color=COLORS['natural'], linewidth=2)
    ax.plot(adv_smooth, label='Adversarial Background', color=COLORS['adversarial'], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Training Reward Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_metric_bars(natural_metrics, adversarial_metrics, output_path):
    """Plot bar chart comparing key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    metrics = [
        ('conflict_rate', 'Conflict Discovery Rate', '%'),
        ('avg_critical_ratio', 'Critical State Ratio', '%'),
        ('goal_rate', 'Goal Reach Rate', '%')
    ]
    
    for ax, (key, title, unit) in zip(axes, metrics):
        nat_val = natural_metrics.get(key, 0)
        adv_val = adversarial_metrics.get(key, 0)
        
        bars = ax.bar(['Natural', 'Adversarial'], [nat_val * 100, adv_val * 100],
                     color=[COLORS['natural'], COLORS['adversarial']], 
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, [nat_val, adv_val]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel(f'{title} ({unit})', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(nat_val * 100, adv_val * 100) * 1.2 + 5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_radar_chart(natural_metrics, adversarial_metrics, output_path):
    """Plot radar/spider chart for multi-dimensional comparison."""
    # Define metrics for radar
    metrics = ['Conflict Rate', 'Critical Ratio', 'Goal Rate', 'Avg Reward', 'Episode Length']
    
    # Normalize values to 0-1 range
    nat_values = [
        natural_metrics.get('conflict_rate', 0),
        natural_metrics.get('avg_critical_ratio', 0),
        natural_metrics.get('goal_rate', 0),
        max(0, min(1, (natural_metrics.get('avg_reward', 0) + 100) / 200)),  # Normalize reward
        min(1, natural_metrics.get('avg_length', 0) / 500)  # Normalize length
    ]
    
    adv_values = [
        adversarial_metrics.get('conflict_rate', 0),
        adversarial_metrics.get('avg_critical_ratio', 0),
        adversarial_metrics.get('goal_rate', 0),
        max(0, min(1, (adversarial_metrics.get('avg_reward', 0) + 100) / 200)),
        min(1, adversarial_metrics.get('avg_length', 0) / 500)
    ]
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    nat_values += nat_values[:1]  # Close the loop
    adv_values += adv_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, nat_values, 'o-', linewidth=2, label='Natural', color=COLORS['natural'])
    ax.fill(angles, nat_values, alpha=0.25, color=COLORS['natural'])
    
    ax.plot(angles, adv_values, 'o-', linewidth=2, label='Adversarial', color=COLORS['adversarial'])
    ax.fill(angles, adv_values, alpha=0.25, color=COLORS['adversarial'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), fontsize=11)
    ax.set_title('Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_reward_distribution(natural_rewards, adversarial_rewards, output_path):
    """Plot reward distribution comparison using violin plots."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = [natural_rewards, adversarial_rewards]
    positions = [1, 2]
    
    parts = ax.violinplot(data, positions, showmeans=True, showmedians=True)
    
    # Color the violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([COLORS['natural'], COLORS['adversarial']][i])
        pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Natural', 'Adversarial'], fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Reward Distribution Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add statistics text
    nat_mean, nat_std = np.mean(natural_rewards), np.std(natural_rewards)
    adv_mean, adv_std = np.mean(adversarial_rewards), np.std(adversarial_rewards)
    
    stats_text = f"Natural: μ={nat_mean:.1f}, σ={nat_std:.1f}\nAdversarial: μ={adv_mean:.1f}, σ={adv_std:.1f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_sample_efficiency(natural_rewards, adversarial_rewards, threshold=0, output_path=None):
    """Plot sample efficiency - steps to reach threshold reward."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def cumulative_max(rewards):
        result = []
        max_so_far = float('-inf')
        for r in rewards:
            max_so_far = max(max_so_far, r)
            result.append(max_so_far)
        return result
    
    nat_cummax = cumulative_max(natural_rewards)
    adv_cummax = cumulative_max(adversarial_rewards)
    
    ax.plot(nat_cummax, label='Natural', color=COLORS['natural'], linewidth=2)
    ax.plot(adv_cummax, label='Adversarial', color=COLORS['adversarial'], linewidth=2)
    ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1, label=f'Threshold ({threshold})')
    
    # Find first episode reaching threshold
    nat_first = next((i for i, r in enumerate(nat_cummax) if r >= threshold), len(nat_cummax))
    adv_first = next((i for i, r in enumerate(adv_cummax) if r >= threshold), len(adv_cummax))
    
    if nat_first < len(nat_cummax):
        ax.axvline(x=nat_first, color=COLORS['natural'], linestyle=':', alpha=0.7)
    if adv_first < len(adv_cummax):
        ax.axvline(x=adv_first, color=COLORS['adversarial'], linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Best Reward So Far', fontsize=12)
    ax.set_title('Sample Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    efficiency_text = f"Episodes to threshold:\n  Natural: {nat_first}\n  Adversarial: {adv_first}"
    ax.text(0.02, 0.98, efficiency_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(natural_results, adversarial_results, output_path):
    """Generate markdown summary table."""
    nat_eval = natural_results['evaluation']
    adv_eval = adversarial_results['evaluation']
    
    # Calculate improvements
    conflict_imp = (adv_eval['conflict_rate'] - nat_eval['conflict_rate']) / max(nat_eval['conflict_rate'], 0.001) * 100
    critical_imp = (adv_eval['avg_critical_ratio'] - nat_eval['avg_critical_ratio']) / max(nat_eval['avg_critical_ratio'], 0.001) * 100
    
    md_content = f"""# Comparison Experiment Results

## Summary Table

| Metric | Natural Background | Adversarial Background | Improvement |
|--------|-------------------|----------------------|-------------|
| Avg Episode Reward | {nat_eval['avg_reward']:.2f} ± {nat_eval['std_reward']:.2f} | {adv_eval['avg_reward']:.2f} ± {adv_eval['std_reward']:.2f} | {adv_eval['avg_reward'] - nat_eval['avg_reward']:+.2f} |
| Avg Episode Length | {nat_eval['avg_length']:.1f} | {adv_eval['avg_length']:.1f} | {adv_eval['avg_length'] - nat_eval['avg_length']:+.1f} |
| Conflict Discovery Rate | {nat_eval['conflict_rate']*100:.1f}% | {adv_eval['conflict_rate']*100:.1f}% | {conflict_imp:+.1f}% |
| Goal Reach Rate | {nat_eval['goal_rate']*100:.1f}% | {adv_eval['goal_rate']*100:.1f}% | {(adv_eval['goal_rate'] - nat_eval['goal_rate'])*100:+.1f}pp |
| Critical State Ratio | {nat_eval['avg_critical_ratio']*100:.1f}% | {adv_eval['avg_critical_ratio']*100:.1f}% | {critical_imp:+.1f}% |

## Key Findings

1. **Conflict Discovery**: {"Adversarial background significantly increases conflict discovery rate" if conflict_imp > 10 else "Comparable conflict discovery rates"}
2. **Critical States**: {"Higher exposure to critical states enables more focused training" if critical_imp > 10 else "Similar critical state exposure"}
3. **Sample Efficiency**: Based on training curves, {"adversarial training shows faster convergence" if adv_eval['avg_reward'] > nat_eval['avg_reward'] else "both methods show similar learning speed"}

## Experiment Configuration

- Scenario: {natural_results['config']['scenario']}
- Intruders: {natural_results['config']['num_intruders']}
- Training Steps: {natural_results['config']['train_steps']}
- Evaluation Episodes: {natural_results['config']['eval_episodes']}

---
Generated: {natural_results.get('timestamp', 'N/A')}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate comparison visualizations')
    parser.add_argument('--results', type=str, required=True, help='Path to results.json')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 70)
    
    # Extract data
    nat_eval = results['natural']['evaluation']
    adv_eval = results['adversarial']['evaluation']
    nat_rewards = results['natural']['raw_data']['episode_rewards']
    adv_rewards = results['adversarial']['raw_data']['episode_rewards']
    
    # Generate all plots
    plot_training_curves(nat_rewards, adv_rewards, 
                        os.path.join(args.output_dir, 'training_curves.png'))
    
    plot_metric_bars(nat_eval, adv_eval,
                    os.path.join(args.output_dir, 'metric_comparison.png'))
    
    plot_radar_chart(nat_eval, adv_eval,
                    os.path.join(args.output_dir, 'radar_chart.png'))
    
    plot_reward_distribution(nat_rewards, adv_rewards,
                            os.path.join(args.output_dir, 'reward_distribution.png'))
    
    plot_sample_efficiency(nat_rewards, adv_rewards, threshold=0,
                          output_path=os.path.join(args.output_dir, 'sample_efficiency.png'))
    
    # Generate summary table
    generate_summary_table(results['natural'], results['adversarial'],
                          os.path.join(args.output_dir, 'summary.md'))
    
    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS GENERATED")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
