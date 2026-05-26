"""
Stage 5: Comparison Experiments - Natural vs Adversarial Background

This script runs comparative experiments between:
1. Natural background: Random intruder behaviors
2. Adversarial background: Trained adversarial policies

Metrics collected:
- Conflict discovery rate
- Critical state ratio
- Average episode reward
- Sample efficiency
- Goal reach rate
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from bluesky_gym.envs.conflict_resolution_env import ConflictResolutionEnv
from bluesky_gym.envs.multi_agent_env import MultiAgentEnv


class ExperimentMetricsCallback(BaseCallback):
    """Callback to collect training metrics."""
    
    def __init__(self, eval_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.conflict_counts = []
        self.goal_reached = []
        self.critical_states = []
        
        # Per-step tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_conflicts = 0
        self.current_critical = 0
    
    def _on_step(self) -> bool:
        # Accumulate episode stats
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        self.current_episode_length += 1
        
        # Check for episode end
        dones = self.locals.get('dones', [False])
        infos = self.locals.get('infos', [{}])
        
        if dones[0]:
            info = infos[0] if infos else {}
            
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.conflict_counts.append(info.get('total_conflicts', 0))
            self.goal_reached.append(info.get('waypoint_reached', False))
            self.critical_states.append(info.get('critical_state_ratio', 0))
            
            # Reset
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_conflicts = 0
            self.current_critical = 0
        
        return True
    
    def get_metrics(self):
        """Return collected metrics."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'conflict_counts': self.conflict_counts,
            'goal_reached': self.goal_reached,
            'critical_states': self.critical_states
        }


def create_natural_env(scenario_type='head_on', num_intruders=3, disturbance='none'):
    """Create environment with natural (random) background."""
    env = ConflictResolutionEnv(
        scenario_type=scenario_type,
        num_intruders=num_intruders,
        disturbance_level=disturbance,
        enable_nfz=False
    )
    return env


def create_adversarial_env(scenario_type='head_on', num_intruders=3, adversary_model_path=None):
    """Create environment with adversarial background."""
    env = MultiAgentEnv(
        scenario_type=scenario_type,
        num_intruders=num_intruders,
        enable_adversarial=True,
        disturbance_level='none',
        enable_nfz=False,
        importance_sampling=True,  # track IS weights during evaluation
    )
    return env


def flatten_action_wrapper(env):
    """Wrap environment for SB3 compatibility."""
    import gymnasium as gym
    from gymnasium import spaces
    
    class FlattenDictActionWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
            self.observation_space = env.observation_space
        
        def reset(self, **kwargs):
            return self.env.reset(**kwargs)
        
        def step(self, action):
            dict_action = {
                "heading": np.array([action[0]], dtype=np.float64),
                "speed": np.array([action[1]], dtype=np.float64),
                "altitude": int(np.clip(np.round((action[2] + 1) * 1), 0, 2))
            }
            return self.env.step(dict_action)
    
    return FlattenDictActionWrapper(env)


def _unwrap_to_multi_agent(env):
    """Unwrap the gym Wrapper stack to reach the underlying MultiAgentEnv; return None if not found."""
    cur = env
    while hasattr(cur, 'env'):
        if isinstance(cur, MultiAgentEnv):
            return cur
        cur = cur.env
    return cur if isinstance(cur, MultiAgentEnv) else None


def evaluate_policy(model, env, n_episodes=100, adversary_policy=None):
    """Evaluate a trained protagonist policy, optionally with an adversarial background agent.

    The returned metrics include two extra fields:
        - ``importance_weights``: per-episode importance-sampling weight.
        - ``conflict_indicator``: whether the episode had a conflict. 1/0.
    """
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'conflicts': [],
        'goal_reached': [],
        'critical_state_ratios': [],
        'importance_weights': [],
        'conflict_indicator': [],
    }

    multi_env = _unwrap_to_multi_agent(env)
    use_adv = adversary_policy is not None and multi_env is not None

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        critical_count = 0

        # Fetch initial adversary_obs from the underlying env
        adv_obs = info.get('adversary_obs', {}) if use_adv else {}

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)

            if use_adv:
                dict_action = {
                    'heading': np.array([action[0]], dtype=np.float64),
                    'speed':   np.array([action[1]], dtype=np.float64),
                    'altitude': int(np.clip(np.round((action[2] + 1) * 1), 0, 2)),
                }
                # Generate adversarial action + log_prob
                adv_actions = {}
                for agent_id, observation in adv_obs.items():
                    obs_t = torch.FloatTensor(observation).unsqueeze(0)
                    with torch.no_grad():
                        h, s, a, lp = adversary_policy.actor.forward(obs_t, deterministic=False)
                    adv_actions[agent_id] = {
                        'heading': h.squeeze(0).cpu().numpy(),
                        'speed':   s.squeeze(0).cpu().numpy(),
                        'altitude': int(a.cpu().item()),
                    }
                    multi_env.update_importance_ratio(
                        agent_id=agent_id,
                        adversarial_log_prob=float(lp.cpu().item()),
                        action=adv_actions[agent_id],
                    )
                obs, reward, terminated, truncated, info = multi_env.step(dict_action, adv_actions)
                adv_obs = info.get('adversary_obs', {})
            else:
                obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            if info.get('is_critical_state', False):
                critical_count += 1
            done = terminated or truncated

        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        n_conflicts = int(info.get('total_conflicts', 0))
        metrics['conflicts'].append(n_conflicts)
        metrics['goal_reached'].append(bool(info.get('waypoint_reached', False)))
        metrics['critical_state_ratios'].append(critical_count / max(episode_length, 1))
        metrics['conflict_indicator'].append(1.0 if n_conflicts > 0 else 0.0)
        ep_w = float(info.get('episode_importance_weight', 1.0))
        metrics['importance_weights'].append(ep_w)

        if (ep + 1) % 20 == 0:
            print(f"  Evaluated {ep + 1}/{n_episodes} episodes")

    return metrics


def compute_is_diagnostics(weights):
    """Diagnose importance-sampling weight degeneracy.

    Key metrics:
        - ``ess``:        Kish effective sample size ``(Σw)² / Σw²``.
        - ``ess_ratio``:  ``ess / N``; <0.1 indicates severe weight degeneracy.
        - ``max_ratio``:  fraction of total weight held by the largest sample; >0.3 is a warning sign.
        - ``degenerate``: True if any of the above thresholds are exceeded.
    """
    weights = np.asarray(weights, dtype=float)
    n = int(weights.size)
    if n == 0 or weights.sum() <= 0:
        return {
            'n': n, 'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0,
            'ess': 0.0, 'ess_ratio': 0.0, 'max_ratio': 0.0, 'degenerate': True,
        }
    total = float(weights.sum())
    ess = float(total ** 2 / np.square(weights).sum())
    ess_ratio = ess / n
    max_ratio = float(weights.max() / total)
    return {
        'n': n,
        'mean': float(weights.mean()),
        'median': float(np.median(weights)),
        'min': float(weights.min()),
        'max': float(weights.max()),
        'ess': ess,
        'ess_ratio': ess_ratio,
        'max_ratio': max_ratio,
        'degenerate': bool(ess_ratio < 0.1 or max_ratio > 0.3),
    }


def compute_rate_statistics(indicators, weights=None, target_relative_halfwidth=0.1):
    """Compute mean / 95% relative half-width / sample-size needed to reach target_relative_halfwidth from binary indicators."""
    indicators = np.asarray(indicators, dtype=float)
    n = len(indicators)
    if n == 0:
        return {'n': 0, 'mean': 0.0, 'std': 0.0, 'relative_halfwidth': float('inf'),
                'episodes_to_target': float('inf')}
    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        return {'n': n, 'mean': 0.0, 'std': 0.0, 'relative_halfwidth': float('inf'),
                'episodes_to_target': float('inf')}

    # Weighted mean & effective sample size (Kish ESS)
    mean = float(np.average(indicators, weights=weights))
    ess = float((weights.sum() ** 2) / (np.square(weights).sum()))
    var = float(np.average((indicators - mean) ** 2, weights=weights))
    std = float(np.sqrt(var))
    halfwidth = 1.96 * std / max(np.sqrt(ess), 1e-9)
    rel_half = halfwidth / mean if mean > 1e-9 else float('inf')
    # Minimum sample size to reach target_relative_halfwidth (assuming constant variance)
    if mean > 1e-9 and std > 0:
        n_required = (1.96 * std / (target_relative_halfwidth * mean)) ** 2
    else:
        n_required = float('inf')
    return {
        'n': n,
        'mean': mean,
        'std': std,
        'effective_sample_size': ess,
        'relative_halfwidth_95': rel_half,
        'episodes_to_relative_halfwidth_{:.0%}'.format(target_relative_halfwidth): n_required,
    }


def run_natural_background_experiment(args):
    """Run experiment with natural background."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Natural Background (Random Intruders)")
    print("=" * 70)
    
    # Create environment
    base_env = create_natural_env(
        scenario_type=args.scenario,
        num_intruders=args.num_intruders,
        disturbance=args.disturbance
    )
    env = flatten_action_wrapper(base_env)
    
    # Create callback
    callback = ExperimentMetricsCallback(eval_freq=1000)
    
    # Train
    print(f"\nTraining PPO for {args.train_steps} steps...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,
        tensorboard_log=f"logs/comparison_natural_{args.experiment_id}"
    )
    
    model.learn(total_timesteps=args.train_steps, callback=callback)
    
    # Save model
    model_path = f"models/comparison_natural_{args.experiment_id}"
    os.makedirs(model_path, exist_ok=True)
    model.save(f"{model_path}/final_model")
    print(f"Model saved to {model_path}")
    
    # Evaluate
    print(f"\nEvaluating for {args.eval_episodes} episodes...")
    eval_metrics = evaluate_policy(model, env, n_episodes=args.eval_episodes)
    
    # Combine metrics
    training_metrics = callback.get_metrics()
    
    results = {
        'experiment_type': 'natural',
        'config': {
            'scenario': args.scenario,
            'num_intruders': args.num_intruders,
            'disturbance': args.disturbance,
            'train_steps': args.train_steps,
            'eval_episodes': args.eval_episodes
        },
        'training': {
            'final_reward': np.mean(training_metrics['episode_rewards'][-10:]) if training_metrics['episode_rewards'] else 0,
            'total_episodes': len(training_metrics['episode_rewards']),
            'avg_episode_length': np.mean(training_metrics['episode_lengths']) if training_metrics['episode_lengths'] else 0
        },
        'evaluation': {
            'avg_reward': np.mean(eval_metrics['episode_rewards']),
            'std_reward': np.std(eval_metrics['episode_rewards']),
            'avg_length': np.mean(eval_metrics['episode_lengths']),
            'conflict_rate': np.mean([c > 0 for c in eval_metrics['conflicts']]),
            'goal_rate': np.mean(eval_metrics['goal_reached']),
            'avg_critical_ratio': np.mean(eval_metrics['critical_state_ratios']),
            # Raw conflict rate statistics (unweighted)
            'rate_stats': compute_rate_statistics(eval_metrics['conflict_indicator']),
        },
        'raw_data': eval_metrics
    }
    
    env.close()
    return results


def run_adversarial_background_experiment(args):
    """Run experiment with adversarial background."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Adversarial Background (Trained Adversaries)")
    print("=" * 70)
    
    # Load adversary policy if available
    adversary_policy = None
    if args.adversary_model:
        if not os.path.exists(args.adversary_model):
            raise FileNotFoundError(
                f"Adversary model checkpoint not found: {args.adversary_model}"
            )
        print(f"Loading adversary policy from {args.adversary_model}...")
        from bluesky_gym.envs.adversarial_policy import AdversarialPolicy
        adversary_policy = AdversarialPolicy(obs_dim=12, device='cpu')
        adversary_policy.load(args.adversary_model)
        print("Adversary policy loaded")
    else:
        raise ValueError(
            "--adversary-model is required to run the adversarial background experiment."
            " Without a trained adversary, the adversary degenerates to random noise."
        )
    
    # Create environment
    env = MultiAgentEnv(
        scenario_type=args.scenario,
        num_intruders=args.num_intruders,
        enable_adversarial=True,
        disturbance_level='none',
        enable_nfz=False
    )
    
    # Wrap for SB3 (protagonist only)
    wrapped_env = flatten_action_wrapper(env)
    
    # Create callback
    callback = ExperimentMetricsCallback(eval_freq=1000)
    
    # Train protagonist with adversarial background
    print(f"\nTraining PPO for {args.train_steps} steps with adversarial background...")
    model = PPO(
        "MlpPolicy",
        wrapped_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,
        tensorboard_log=f"logs/comparison_adversarial_{args.experiment_id}"
    )
    
    model.learn(total_timesteps=args.train_steps, callback=callback)
    
    # Save model
    model_path = f"models/comparison_adversarial_{args.experiment_id}"
    os.makedirs(model_path, exist_ok=True)
    model.save(f"{model_path}/final_model")
    print(f"Model saved to {model_path}")
    
    # Evaluate
    print(f"\nEvaluating for {args.eval_episodes} episodes...")
    eval_metrics = evaluate_policy(model, wrapped_env, n_episodes=args.eval_episodes, adversary_policy=adversary_policy)
    
    # Combine metrics
    training_metrics = callback.get_metrics()
    
    results = {
        'experiment_type': 'adversarial',
        'config': {
            'scenario': args.scenario,
            'num_intruders': args.num_intruders,
            'adversary_model': args.adversary_model,
            'train_steps': args.train_steps,
            'eval_episodes': args.eval_episodes
        },
        'training': {
            'final_reward': np.mean(training_metrics['episode_rewards'][-10:]) if training_metrics['episode_rewards'] else 0,
            'total_episodes': len(training_metrics['episode_rewards']),
            'avg_episode_length': np.mean(training_metrics['episode_lengths']) if training_metrics['episode_lengths'] else 0
        },
        'evaluation': {
            'avg_reward': np.mean(eval_metrics['episode_rewards']),
            'std_reward': np.std(eval_metrics['episode_rewards']),
            'avg_length': np.mean(eval_metrics['episode_lengths']),
            'conflict_rate': np.mean([c > 0 for c in eval_metrics['conflicts']]),
            'goal_rate': np.mean(eval_metrics['goal_reached']),
            'avg_critical_ratio': np.mean(eval_metrics['critical_state_ratios']),
            # Raw conflict rate statistics (under adversarial distribution)
            'rate_stats': compute_rate_statistics(eval_metrics['conflict_indicator']),
            # IS-weighted conflict rate statistics (corrected back to natural distribution)
            'rate_stats_is_weighted': compute_rate_statistics(
                eval_metrics['conflict_indicator'],
                weights=eval_metrics['importance_weights'],
            ),
            # IS weight degeneracy diagnostics
            'is_diagnostics': compute_is_diagnostics(eval_metrics['importance_weights']),
        },
        'raw_data': eval_metrics
    }
    
    env.close()
    return results


def generate_comparison_report(natural_results, adversarial_results, output_path):
    """Generate comparison report."""
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print("\n{:<30} {:>15} {:>15}".format("Metric", "Natural", "Adversarial"))
    print("-" * 60)
    
    metrics = [
        ('Avg Episode Reward', 'avg_reward'),
        ('Reward Std Dev', 'std_reward'),
        ('Avg Episode Length', 'avg_length'),
        ('Conflict Discovery Rate', 'conflict_rate'),
        ('Goal Reach Rate', 'goal_rate'),
        ('Critical State Ratio', 'avg_critical_ratio')
    ]
    
    for name, key in metrics:
        nat_val = natural_results['evaluation'].get(key, 0)
        adv_val = adversarial_results['evaluation'].get(key, 0)
        
        if 'rate' in key.lower() or 'ratio' in key.lower():
            print("{:<30} {:>14.1%} {:>14.1%}".format(name, nat_val, adv_val))
        else:
            print("{:<30} {:>15.2f} {:>15.2f}".format(name, nat_val, adv_val))
    
    # Calculate improvements
    print("\n" + "-" * 60)
    print("ANALYSIS:")
    
    conflict_improvement = (adversarial_results['evaluation']['conflict_rate'] - 
                           natural_results['evaluation']['conflict_rate'])
    critical_improvement = (adversarial_results['evaluation']['avg_critical_ratio'] - 
                           natural_results['evaluation']['avg_critical_ratio'])
    
    print(f"  Conflict Discovery Improvement: {conflict_improvement:+.1%}")
    print(f"  Critical State Improvement: {critical_improvement:+.1%}")

    # ===== Relative half-width / sample-size statistics (validate dense RL sample efficiency) =====
    print("\n" + "-" * 60)
    print("STATISTICAL EFFICIENCY (95% CI, target relative half-width = 10%):")
    nat_stats = natural_results['evaluation'].get('rate_stats', {})
    adv_stats = adversarial_results['evaluation'].get('rate_stats', {})
    adv_is_stats = adversarial_results['evaluation'].get('rate_stats_is_weighted', {})
    key_n = 'episodes_to_relative_halfwidth_10%'
    print(f"  [Natural ]  rate={nat_stats.get('mean', 0):.4f}  "
          f"rel_halfwidth={nat_stats.get('relative_halfwidth_95', float('inf')):.3f}  "
          f"episodes_to_10%={nat_stats.get(key_n, float('inf')):.0f}")
    print(f"  [Adv     ]  rate={adv_stats.get('mean', 0):.4f}  "
          f"rel_halfwidth={adv_stats.get('relative_halfwidth_95', float('inf')):.3f}  "
          f"episodes_to_10%={adv_stats.get(key_n, float('inf')):.0f}")
    print(f"  [Adv+IS  ]  rate={adv_is_stats.get('mean', 0):.4f}  "
          f"rel_halfwidth={adv_is_stats.get('relative_halfwidth_95', float('inf')):.3f}  "
          f"episodes_to_10%={adv_is_stats.get(key_n, float('inf')):.0f}  (IS-weighted -> natural distribution)")
    if nat_stats.get(key_n, float('inf')) > 0 and adv_is_stats.get(key_n, float('inf')) > 0:
        speedup = nat_stats[key_n] / adv_is_stats[key_n]
        print(f"  Sample efficiency speedup (Natural / Adv+IS): {speedup:.1f}x")

    # ===== IS weight degeneracy diagnostics =====
    is_diag = adversarial_results['evaluation'].get('is_diagnostics', {})
    print("\n" + "-" * 60)
    print("IMPORTANCE SAMPLING WEIGHT DIAGNOSTICS:")
    print(f"  N={is_diag.get('n', 0)}  ESS={is_diag.get('ess', 0):.1f}  "
          f"ESS/N={is_diag.get('ess_ratio', 0):.3f}  "
          f"max_ratio={is_diag.get('max_ratio', 0):.3f}")
    print(f"  weight stats: min={is_diag.get('min', 0):.3e}  "
          f"median={is_diag.get('median', 0):.3e}  "
          f"mean={is_diag.get('mean', 0):.3e}  "
          f"max={is_diag.get('max', 0):.3e}")
    if is_diag.get('degenerate', False):
        print("  ⚠  IS weights are degenerate (ESS/N<0.1 or max_ratio>0.3).")
        print("     The IS-weighted estimate may have inflated variance;"
              " consider lowering adversary entropy or increasing eval episodes.")
    else:
        print("  ✓  IS weights healthy.")

    # ===== Unbiasedness check: natural ground truth vs adversarial+IS estimate =====
    print("\n" + "-" * 60)
    print("UNBIASEDNESS CHECK (Natural ground truth vs IS-weighted estimate):")
    p_nat = float(nat_stats.get('mean', 0.0))
    p_adv_is = float(adv_is_stats.get('mean', 0.0))
    bias = p_adv_is - p_nat
    nat_hw = float(nat_stats.get('relative_halfwidth_95', float('inf'))) * abs(p_nat)
    adv_is_hw = float(adv_is_stats.get('relative_halfwidth_95', float('inf'))) * abs(p_adv_is)
    combined_hw = float(np.sqrt(nat_hw ** 2 + adv_is_hw ** 2))
    print(f"  p_natural          = {p_nat:.4f}  ± {nat_hw:.4f} (95% CI half-width)")
    print(f"  p_adversarial+IS   = {p_adv_is:.4f}  ± {adv_is_hw:.4f} (95% CI half-width)")
    print(f"  bias (adv_IS - nat) = {bias:+.4f}   tolerance ±{combined_hw:.4f}")
    if abs(bias) <= combined_hw:
        print("  ✓  IS-weighted estimate is statistically consistent with the natural ground truth.")
        unbiased_ok = True
    else:
        print("  ⚠  IS-weighted estimate deviates from the natural ground truth"
              " beyond the combined 95% CI; check adversary policy or sample size.")
        unbiased_ok = False

    if adversarial_results['evaluation']['conflict_rate'] > natural_results['evaluation']['conflict_rate']:
        print("  ✓ Adversarial background increases conflict discovery")
    else:
        print("  ⚠ Adversarial background did not increase conflict discovery")
    
    if adversarial_results['evaluation']['avg_critical_ratio'] > natural_results['evaluation']['avg_critical_ratio']:
        print("  ✓ Adversarial background increases critical state exposure")
    else:
        print("  ⚠ Adversarial background did not increase critical states")
    
    # Save results - extract only serializable data
    def make_serializable(obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items() if k != 'raw_data'}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        else:
            return obj
    
    combined_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            # Preserve scenario and key evaluation parameters for downstream reporting scripts
            # (e.g., generate_phase5_comprehensive_report.py) to identify the scenario
            'scenario': natural_results.get('config', {}).get('scenario'),
            'num_intruders': natural_results.get('config', {}).get('num_intruders'),
            'eval_episodes': natural_results.get('config', {}).get('eval_episodes'),
        },
        'natural': {
            'config': make_serializable(natural_results.get('config', {})),
            'training': make_serializable(natural_results.get('training', {})),
            'evaluation': make_serializable(natural_results.get('evaluation', {}))
        },
        'adversarial': {
            'config': make_serializable(adversarial_results.get('config', {})),
            'training': make_serializable(adversarial_results.get('training', {})),
            'evaluation': make_serializable(adversarial_results.get('evaluation', {}))
        },
        'comparison': {
            'conflict_improvement': float(conflict_improvement),
            'critical_improvement': float(critical_improvement),
            'unbiasedness_check': {
                'p_natural': p_nat,
                'p_adversarial_is_weighted': p_adv_is,
                'bias': float(bias),
                'combined_95ci_halfwidth': combined_hw,
                'consistent': bool(unbiased_ok),
            },
            'sample_efficiency_speedup_natural_over_adv_is': float(
                nat_stats.get(key_n, float('inf')) / adv_is_stats[key_n]
            ) if (nat_stats.get(key_n, float('inf')) > 0
                  and adv_is_stats.get(key_n, float('inf')) > 0) else None,
            'is_diagnostics': make_serializable(
                adversarial_results['evaluation'].get('is_diagnostics', {})
            ),
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return combined_results


def main():
    parser = argparse.ArgumentParser(description='Run comparison experiments')
    parser.add_argument('--scenario', type=str, default='head_on', 
                       choices=['head_on', 'crossing', 'merging', 'overtaking'])
    parser.add_argument('--num-intruders', type=int, default=3)
    parser.add_argument('--disturbance', type=str, default='none',
                       choices=['none', 'light', 'medium', 'heavy'])
    parser.add_argument('--train-steps', type=int, default=50000)
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--adversary-model', type=str, default=None,
                       help='Path to trained adversary policy')
    parser.add_argument('--skip-natural', action='store_true',
                       help='Skip natural background experiment')
    parser.add_argument('--skip-adversarial', action='store_true',
                       help='Skip adversarial background experiment')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    args.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("STAGE 5: COMPARISON EXPERIMENTS")
    print("Natural Background vs Adversarial Background")
    print("=" * 70)
    print(f"\nExperiment ID: {args.experiment_id}")
    print(f"Scenario: {args.scenario}")
    print(f"Intruders: {args.num_intruders}")
    print(f"Training steps: {args.train_steps}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    natural_results = None
    adversarial_results = None
    
    # Run experiments
    if not args.skip_natural:
        natural_results = run_natural_background_experiment(args)
    
    if not args.skip_adversarial:
        adversarial_results = run_adversarial_background_experiment(args)
    
    # Generate report
    if natural_results and adversarial_results:
        output_path = f"results/comparison_{args.experiment_id}/results.json"
        generate_comparison_report(natural_results, adversarial_results, output_path)
    
    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
