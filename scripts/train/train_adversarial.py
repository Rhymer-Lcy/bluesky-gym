"""Adversarial training script.

Protagonist (conflict-resolution policy): trained with standard SB3 PPO.
Adversary  (background aircraft policy): trained with a custom Actor-Critic
  algorithm featuring advantage estimation, gradient clipping, critical-state
  weighting and per-step importance-sampling weight accumulation.
  Note: this is NOT PPO. For strict PPO a rollout buffer / clip ratio /
  multi-epoch update would be needed.

Training Strategy:
    - Phase 1: train protagonist against random background (baseline)
    - Phase 2: freeze protagonist, train adversary to maximise conflict exposure
    - Phase 3: alternating co-evolutionary training

Features:
    - Protagonist via SB3 PPO
    - Adversary logs per-step IS weights to reweight toward natural distribution
    - Critical-state filtering (TCPA<90s / TAU<120s): skip non-critical updates
    - Gradient clipping + Tensorboard + Checkpoint
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

_log = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from bluesky_gym.envs.multi_agent_env import MultiAgentEnv
from bluesky_gym.envs.adversarial_policy import AdversarialPolicy
from bluesky_gym.utils.wrappers import FlattenDictActionWrapper
import gymnasium as gym


class AdversarialTrainingCallback(BaseCallback):
    """
    Custom callback for adversarial training.
    
    Logs protagonist and adversary performance metrics.
    """
    
    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_conflicts = []
        self.episode_goal_reached = []
        self.critical_state_count = 0
        self.total_state_count = 0
    
    def _on_step(self) -> bool:
        # Track critical states for dense RL
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            self.total_state_count += 1
            if info.get('is_critical_state', False):
                self.critical_state_count += 1
        
        # Log every log_freq steps
        if self.n_calls % self.log_freq == 0:
            # Extract episode statistics from info
            if len(self.locals.get('infos', [])) > 0:
                info = self.locals['infos'][0]
                
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    self.episode_rewards.append(ep_reward)
                    
                    # Log to tensorboard
                    self.logger.record('train/episode_reward', ep_reward)
                    self.logger.record('train/episode_length', ep_length)
                
                # Log critical state ratio (key metric for dense RL)
                if self.total_state_count > 0:
                    critical_ratio = self.critical_state_count / self.total_state_count
                    self.logger.record('dense_rl/critical_state_ratio', critical_ratio)
                    self.logger.record('dense_rl/critical_state_count', self.critical_state_count)
                
                # Log adversary statistics if available
                if 'adversary_rewards' in info:
                    avg_adversary_reward = np.mean(list(info['adversary_rewards'].values()))
                    self.logger.record('adversary/avg_reward', avg_adversary_reward)
        
        return True


def make_env(
    scenario_type: str = 'head_on',
    num_intruders: int = 3,
    enable_adversarial: bool = True,
    importance_sampling: bool = False
) -> gym.Env:
    """
    Create and wrap environment for training.
    
    Args:
        scenario_type: Type of conflict scenario
        num_intruders: Number of intruder aircraft
        enable_adversarial: Whether to enable adversarial training
        importance_sampling: Whether to track importance ratios
    
    Returns:
        Wrapped environment
    """
    env = MultiAgentEnv(
        scenario_type=scenario_type,
        num_intruders=num_intruders,
        disturbance_level='none',
        enable_nfz=False,
        enable_adversarial=enable_adversarial,
        importance_sampling=importance_sampling
    )
    
    # Wrap to flatten action space
    env = FlattenDictActionWrapper(env)
    
    # Monitor for episode statistics
    env = Monitor(env)
    
    return env


def train_protagonist(
    model_dir: str,
    log_dir: str,
    total_timesteps: int = 100000,
    scenario_type: str = 'head_on',
    num_intruders: int = 3,
    learning_rate: float = 3e-4,
    n_steps: int = 512,  # Reduced for faster adaptation in adversarial scenarios
    batch_size: int = 64,
    n_epochs: int = 10,
    device: str = 'cpu',
    save_freq: int = 10000,
    log_interval: int = 10,
    eval_freq: int = 5000,
    n_eval_episodes: int = 10
) -> PPO:
    """
    Train protagonist policy (conflict resolution).
    
    Args:
        model_dir: Directory to save model files
        log_dir: Directory to save logs
        total_timesteps: Total training timesteps
        scenario_type: Conflict scenario type
        num_intruders: Number of background aircraft
        learning_rate: PPO learning rate
        n_steps: Number of steps per rollout
        batch_size: Minibatch size
        n_epochs: Number of optimization epochs
        device: Device to train on
        save_freq: Frequency to save checkpoints
        log_interval: Frequency to log training info
        eval_freq: Frequency to evaluate policy
        n_eval_episodes: Number of evaluation episodes
    
    Returns:
        Trained PPO model
    """
    _log.info("Training protagonist policy...")
    _log.info("  Scenario: %s", scenario_type)
    _log.info("  Intruders: %d", num_intruders)
    _log.info("  Total timesteps: %d", total_timesteps)
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create training environment
    env = make_env(scenario_type, num_intruders, enable_adversarial=False)
    
    # Create evaluation environment
    eval_env = make_env(scenario_type, num_intruders, enable_adversarial=False)
    
    # Configure tensorboard logger (logs only)
    logger = configure(log_dir, ["tensorboard", "stdout"])
    
    # Create PPO model with optimized hyperparameters for adversarial training
    # Learning rate decay: linear from initial_lr to 0
    def lr_schedule(progress_remaining: float) -> float:
        return progress_remaining * learning_rate
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule,  # Linear decay
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.05,  # Increased for more exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log=None  # We'll use custom logger
    )
    
    model.set_logger(logger)
    
    # Setup callbacks - save models to model_dir
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="protagonist_ppo"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    
    training_callback = AdversarialTrainingCallback(log_freq=log_interval)
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, training_callback],
        log_interval=log_interval
    )
    
    # Save final model to model_dir
    final_model_path = os.path.join(model_dir, "final_model")
    try:
        model.save(final_model_path)
        _log.info("Training complete!")
        _log.info("  Model saved to: %s.zip", final_model_path)
    except OSError as exc:
        _log.warning("Training complete! Could not save model to %s: %s", final_model_path, exc)
    _log.info("  Logs saved to: %s", log_dir)
    
    env.close()
    eval_env.close()
    
    return model


def train_adversary(
    model_dir: str,
    log_dir: str,
    protagonist_model_path: str,
    total_timesteps: int = 50000,
    scenario_type: str = 'head_on',
    num_intruders: int = 3,
    learning_rate: float = 3e-4,
    device: str = 'cpu'
) -> AdversarialPolicy:
    """
    Train adversarial policy against fixed protagonist.
    
    Args:
        model_dir: Directory to save models
        log_dir: Directory to save logs
        protagonist_model_path: Path to trained protagonist model
        total_timesteps: Total training timesteps
        scenario_type: Conflict scenario type
        num_intruders: Number of adversarial aircraft
        learning_rate: Learning rate
        device: Device to train on
    
    Returns:
        Trained adversarial policy
    """
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    
    _log.info("Training adversarial policy...")
    _log.info("  Loading protagonist from: %s", protagonist_model_path)
    _log.info("  Total timesteps: %d", total_timesteps)
    
    # Load protagonist model
    protagonist = PPO.load(protagonist_model_path, device=device)
    _log.info("  Protagonist loaded")
    
    # Create multi-agent environment
    env = MultiAgentEnv(
        scenario_type=scenario_type,
        num_intruders=num_intruders,
        enable_adversarial=True,
        importance_sampling=True
    )
    
    # Wrap for protagonist
    env_wrapped = FlattenDictActionWrapper(env)
    
    # Create adversarial policy
    adversarial_policy = AdversarialPolicy(
        obs_dim=12,  # Adversary observation dimension
        device=device
    )
    
    _log.info("%s", adversarial_policy.info())
    
    # Setup optimizer
    optimizer = optim.Adam(
        list(adversarial_policy.actor.parameters()) + 
        list(adversarial_policy.critic.parameters()),
        lr=learning_rate
    )
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir)
    
    # Training loop
    _log.info("Starting adversarial training...")
    episode_count = 0
    step_count = 0
    best_adversary_reward = -np.inf
    
    while step_count < total_timesteps:
        # Reset environment
        obs, info = env.reset()
        adversary_obs = info['adversary_obs']
        
        episode_reward_protagonist = 0
        episode_reward_adversaries = {aid: 0.0 for aid in env.adversarial_agents}
        episode_critical_states = 0
        episode_steps = 0
        
        done = False
        
        while not done and episode_steps < 1000:
            # Protagonist action (from trained model)
            protagonist_action_flat, _ = protagonist.predict(obs, deterministic=False)
            protagonist_action = {
                'heading': np.array([protagonist_action_flat[0]]),
                'speed': np.array([protagonist_action_flat[1]]),
                'altitude': int(np.clip(np.round((protagonist_action_flat[2] + 1) * 1), 0, 2))
            }
            
            # Adversary actions (from trainable policy)
            adversary_actions = {}
            adversary_action_log_probs = {}
            for agent_id in env.adversarial_agents:
                if agent_id in adversary_obs:
                    adv_obs = adversary_obs[agent_id]
                    obs_tensor = torch.FloatTensor(adv_obs).unsqueeze(0).to(device)
                    with torch.no_grad():
                        h, s, a, lp = adversarial_policy.actor.forward(
                            obs_tensor, deterministic=False
                        )
                    adversary_actions[agent_id] = {
                        'heading': h.squeeze(0).cpu().numpy(),
                        'speed': s.squeeze(0).cpu().numpy(),
                        'altitude': int(a.cpu().item()),
                    }
                    adversary_action_log_probs[agent_id] = float(lp.cpu().item())

            # Step environment
            next_obs, protagonist_reward, terminated, truncated, info = env.step(
                protagonist_action, adversary_actions
            )

            # accumulate importance-sampling weights pi_natural / pi_adv via env interface
            for agent_id, adv_lp in adversary_action_log_probs.items():
                env.update_importance_ratio(
                    agent_id=agent_id,
                    adversarial_log_prob=adv_lp,
                    action=adversary_actions[agent_id],
                )
            
            done = terminated or truncated
            
            # Update statistics
            episode_reward_protagonist += protagonist_reward
            if 'adversary_rewards' in info:
                for aid, reward in info['adversary_rewards'].items():
                    episode_reward_adversaries[aid] += reward
            
            if info.get('is_critical_state', False):
                episode_critical_states += 1
            
            # Get next adversary observations
            next_adversary_obs = info.get('adversary_obs', {})

            # Dense RL: update gradients only on critical states; skip non-critical steps.
            is_critical = bool(info.get('is_critical_state', False))
            if is_critical:
                for agent_id in env.adversarial_agents:
                    if agent_id in adversary_obs and agent_id in info.get('adversary_rewards', {}):
                        adv_obs_tensor = torch.FloatTensor(adversary_obs[agent_id]).to(device)
                        adv_reward = info['adversary_rewards'][agent_id]

                        # Compute value and loss
                        value = adversarial_policy.critic(adv_obs_tensor.unsqueeze(0))
                        advantage = adv_reward - value.detach()

                        # Actor loss (policy gradient with log probability)
                        heading, speed, altitude, log_prob = adversarial_policy.actor.forward(
                            adv_obs_tensor.unsqueeze(0), deterministic=False
                        )
                        actor_loss = -(advantage * log_prob)
                        critic_loss = (value - adv_reward) ** 2
                        loss = actor_loss + 0.5 * critic_loss

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(adversarial_policy.actor.parameters()) +
                            list(adversarial_policy.critic.parameters()),
                            max_norm=0.5
                        )
                        optimizer.step()
            
            # Update for next step
            obs = next_obs
            adversary_obs = next_adversary_obs
            episode_steps += 1
            step_count += 1
        
        # Episode complete
        episode_count += 1
        
        # Compute average adversary reward
        avg_adversary_reward = np.mean(list(episode_reward_adversaries.values()))
        critical_ratio = episode_critical_states / max(episode_steps, 1)
        
        # Log to tensorboard
        writer.add_scalar('adversary/avg_episode_reward', avg_adversary_reward, episode_count)
        writer.add_scalar('protagonist/episode_reward', episode_reward_protagonist, episode_count)
        writer.add_scalar('episode/critical_state_ratio', critical_ratio, episode_count)
        writer.add_scalar('episode/steps', episode_steps, episode_count)
        
        # Print progress
        if episode_count % 10 == 0:
            _log.info(
                "Episode %d, Steps %d/%d | adversary_reward=%.2f "
                "protagonist_reward=%.2f critical_states=%d/%d (%.2f%%)",
                episode_count, step_count, total_timesteps,
                avg_adversary_reward, episode_reward_protagonist,
                episode_critical_states, episode_steps, critical_ratio * 100,
            )
        
        # Save best model
        if avg_adversary_reward > best_adversary_reward:
            best_adversary_reward = avg_adversary_reward
            best_policy_path = os.path.join(model_dir, "best_adversarial_policy.pt")
            adversarial_policy.save(best_policy_path)
    
    writer.close()
    
    # Save final adversarial policy
    final_policy_path = os.path.join(model_dir, "final_adversarial_policy.pt")
    adversarial_policy.save(final_policy_path)
    
    _log.info("Adversarial training complete!")
    _log.info("  Final policy saved to: %s", final_policy_path)
    _log.info("  Best policy saved to: %s", best_policy_path)
    _log.info("  Logs saved to: %s", log_dir)
    
    env.close()
    
    return adversarial_policy


def alternating_training(
    model_dir: str,
    log_dir: str,
    protagonist_timesteps_per_iter: int = 20000,
    adversary_timesteps_per_iter: int = 10000,
    n_iterations: int = 5,
    scenario_type: str = 'head_on',
    num_intruders: int = 3,
    device: str = 'cpu'
):
    """
    Alternating training between protagonist and adversary.
    
    Strategy:
    1. Train protagonist for N timesteps
    2. Train adversary against current protagonist
    3. Repeat for M iterations
    
    Args:
        output_dir: Directory to save models and logs
        protagonist_timesteps_per_iter: Timesteps per protagonist iteration
        adversary_timesteps_per_iter: Timesteps per adversary iteration
        n_iterations: Number of alternating iterations
        scenario_type: Conflict scenario type
        num_intruders: Number of intruder aircraft
        device: Device to train on
    """
    _log.info("=" * 70)
    _log.info("ALTERNATING ADVERSARIAL TRAINING")
    _log.info("=" * 70)
    _log.info("Iterations: %d", n_iterations)
    _log.info("Protagonist timesteps per iter: %d", protagonist_timesteps_per_iter)
    _log.info("Adversary timesteps per iter: %d", adversary_timesteps_per_iter)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    protagonist_model = None
    adversary_policy = None

    for iteration in range(n_iterations):
        _log.info("=" * 70)
        _log.info("ITERATION %d/%d", iteration + 1, n_iterations)
        _log.info("=" * 70)

        # Train protagonist
        prot_model_dir = os.path.join(model_dir, f"iter_{iteration + 1}", "protagonist")
        prot_log_dir   = os.path.join(log_dir,   f"iter_{iteration + 1}", "protagonist")

        protagonist_model = train_protagonist(
            model_dir=prot_model_dir,
            log_dir=prot_log_dir,
            total_timesteps=protagonist_timesteps_per_iter,
            scenario_type=scenario_type,
            num_intruders=num_intruders,
            device=device
        )

        # Train adversary against current protagonist
        if iteration < n_iterations - 1:
            adv_model_dir = os.path.join(model_dir, f"iter_{iteration + 1}", "adversary")
            adv_log_dir   = os.path.join(log_dir,   f"iter_{iteration + 1}", "adversary")
            protagonist_path = os.path.join(prot_model_dir, "final_model.zip")

            adversary_policy = train_adversary(
                model_dir=adv_model_dir,
                log_dir=adv_log_dir,
                protagonist_model_path=protagonist_path,
                total_timesteps=adversary_timesteps_per_iter,
                scenario_type=scenario_type,
                num_intruders=num_intruders,
                device=device
            )

    _log.info("=" * 70)
    _log.info("ALTERNATING TRAINING COMPLETE")
    _log.info("=" * 70)
    _log.info("Models saved in: %s", model_dir)
    _log.info("Logs   saved in: %s", log_dir)


def main():
    parser = argparse.ArgumentParser(description="Adversarial Training for Conflict Resolution")
    
    parser.add_argument('--mode', type=str, default='protagonist',
                       choices=['protagonist', 'adversary', 'alternating'],
                       help='Training mode')
    parser.add_argument('--output-dir', type=str, default='logs/adversarial_training',
                       help='Output directory for models and logs')
    parser.add_argument('--scenario', type=str, default='head_on',
                       choices=['head_on', 'crossing', 'merging', 'overtaking'],
                       help='Conflict scenario type')
    parser.add_argument('--num-intruders', type=int, default=3,
                       help='Number of intruder aircraft')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--protagonist-timesteps', type=int, default=20000,
                       help='Protagonist timesteps per iteration (alternating mode)')
    parser.add_argument('--adversary-timesteps', type=int, default=10000,
                       help='Adversary timesteps per iteration (alternating mode)')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations (alternating mode)')
    parser.add_argument('--protagonist-model', type=str, default=None,
                       help='Path to protagonist model (for adversary training)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to train on')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Configure root logging so _log messages appear on stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.mode}_{args.scenario}_{timestamp}"
    
    # Separate directories for models and logs
    model_dir = os.path.join("models", run_name)
    log_dir = os.path.join("logs", run_name)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    _log.info("Model directory: %s", model_dir)
    _log.info("Log directory:   %s", log_dir)
    _log.info("Device: %s  seed: %d", args.device, args.seed)
    
    # Run training based on mode
    if args.mode == 'protagonist':
        train_protagonist(
            model_dir=model_dir,
            log_dir=log_dir,
            total_timesteps=args.timesteps,
            scenario_type=args.scenario,
            num_intruders=args.num_intruders,
            learning_rate=args.learning_rate,
            device=args.device
        )
    
    elif args.mode == 'adversary':
        if args.protagonist_model is None:
            _log.error("--protagonist-model required for adversary training")
            sys.exit(1)
        
        train_adversary(
            model_dir=model_dir,
            log_dir=log_dir,
            protagonist_model_path=args.protagonist_model,
            total_timesteps=args.timesteps,
            scenario_type=args.scenario,
            num_intruders=args.num_intruders,
            learning_rate=args.learning_rate,
            device=args.device
        )
    
    elif args.mode == 'alternating':
        alternating_training(
            model_dir=model_dir,
            log_dir=log_dir,
            protagonist_timesteps_per_iter=args.protagonist_timesteps,
            adversary_timesteps_per_iter=args.adversary_timesteps,
            n_iterations=args.iterations,
            scenario_type=args.scenario,
            num_intruders=args.num_intruders,
            device=args.device
        )


if __name__ == "__main__":
    main()
