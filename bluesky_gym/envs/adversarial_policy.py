"""
Adversarial Policy Network for Intelligent Background Aircraft

This module implements an Actor-Critic network that generates challenging but
realistic behaviors for background aircraft to train robust conflict resolution agents.

Design Philosophy:
    - Increase conflict complexity without causing crashes
    - Learn behaviors that expose weaknesses in the protagonist policy
    - Maintain flight safety and realism (altitude limits, speed constraints)
    - Balance adversarial objective with trajectory smoothness

Key Features:
    - Actor network: Outputs continuous heading/speed + discrete altitude actions
    - Critic network: Estimates expected cumulative adversarial reward
    - Observation: Relative position, speed, conflict potential with protagonist
    - Constraints: Action clipping, safety margins, physics-based limits
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Tuple, Dict, Optional


class AdversarialActor(nn.Module):
    """
    Actor network for adversarial policy.
    
    Outputs continuous heading/speed adjustments and discrete altitude action
    to create challenging scenarios for the protagonist aircraft.
    """
    
    def __init__(
        self,
        obs_dim: int = 12,
        hidden_dims: Tuple[int, ...] = (128, 128),
        heading_range: float = 30.0,  # ±30° heading change
        speed_range: float = 10.0,    # ±10 m/s speed change
        altitude_actions: int = 3,    # {down, maintain, up}
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        device: str = 'cpu'
    ):
        """
        Initialize adversarial actor network.
        
        Args:
            obs_dim: Observation dimension (relative state)
            hidden_dims: Hidden layer dimensions
            heading_range: Maximum heading change in degrees
            speed_range: Maximum speed change in m/s
            altitude_actions: Number of discrete altitude actions
            log_std_min: Minimum log std for continuous actions
            log_std_max: Maximum log std for continuous actions
            device: Device to run on
        """
        super().__init__()
        
        self.device = device
        self.heading_range = heading_range
        self.speed_range = speed_range
        self.altitude_actions = altitude_actions
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.log_std_max = log_std_max
        
        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        
        # Continuous action heads (heading, speed)
        self.heading_mean = nn.Linear(in_dim, 1)
        self.heading_log_std = nn.Linear(in_dim, 1)
        self.speed_mean = nn.Linear(in_dim, 1)
        self.speed_log_std = nn.Linear(in_dim, 1)
        
        # Discrete action head (altitude)
        self.altitude_logits = nn.Linear(in_dim, altitude_actions)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Small initialization for output layers
        nn.init.orthogonal_(self.heading_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.speed_mean.weight, gain=0.01)
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to get actions and log probabilities.
        
        Args:
            obs: Observation tensor (batch_size, obs_dim)
            deterministic: If True, return mean actions; else sample
        
        Returns:
            heading_action: Heading adjustment in [-1, 1]
            speed_action: Speed adjustment in [-1, 1]
            altitude_action: Altitude action in {0, 1, 2}
            log_prob: Total log probability of the action
        """
        # Ensure obs is on the correct device
        device_str = str(self.device) if not isinstance(self.device, str) else self.device
        if not obs.is_cuda and 'cuda' in device_str:
            obs = obs.to(self.device)
        
        features = self.feature_net(obs)
        
        # Continuous actions: heading and speed
        heading_mean = torch.tanh(self.heading_mean(features))
        heading_log_std = torch.clamp(
            self.heading_log_std(features),
            self.log_std_min,
            self.log_std_max
        )
        heading_std = torch.exp(heading_log_std)
        
        speed_mean = torch.tanh(self.speed_mean(features))
        speed_log_std = torch.clamp(
            self.speed_log_std(features),
            self.log_std_min,
            self.log_std_max
        )
        speed_std = torch.exp(speed_log_std)
        
        if deterministic:
            heading_action = heading_mean
            speed_action = speed_mean
            heading_log_prob = torch.zeros_like(heading_mean)
            speed_log_prob = torch.zeros_like(speed_mean)
        else:
            heading_dist = Normal(heading_mean, heading_std)
            heading_action = heading_dist.rsample()
            heading_action = torch.clamp(heading_action, -1.0, 1.0)
            heading_log_prob = heading_dist.log_prob(heading_action)
            
            speed_dist = Normal(speed_mean, speed_std)
            speed_action = speed_dist.rsample()
            speed_action = torch.clamp(speed_action, -1.0, 1.0)
            speed_log_prob = speed_dist.log_prob(speed_action)
        
        # Discrete action: altitude
        altitude_logits = self.altitude_logits(features)
        altitude_dist = Categorical(logits=altitude_logits)
        
        if deterministic:
            altitude_action = torch.argmax(altitude_logits, dim=-1)
        else:
            altitude_action = altitude_dist.sample()
        
        altitude_log_prob = altitude_dist.log_prob(altitude_action)
        
        # Total log probability
        log_prob = heading_log_prob.sum(-1) + speed_log_prob.sum(-1) + altitude_log_prob
        
        return heading_action, speed_action, altitude_action, log_prob
    
    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Get action for a single observation (for inference).
        
        Args:
            obs: Observation array (obs_dim,)
            deterministic: If True, return mean actions
        
        Returns:
            Dictionary with 'heading', 'speed', 'altitude' actions
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            heading, speed, altitude, _ = self.forward(obs_tensor, deterministic)
            
            return {
                'heading': heading.squeeze(0).cpu().numpy(),
                'speed': speed.squeeze(0).cpu().numpy(),
                'altitude': altitude.cpu().item()
            }


class AdversarialCritic(nn.Module):
    """
    Critic network for adversarial policy.
    
    Estimates the expected cumulative adversarial reward given the current state.
    """
    
    def __init__(
        self,
        obs_dim: int = 12,
        hidden_dims: Tuple[int, ...] = (128, 128)
    ):
        """
        Initialize adversarial critic network.
        
        Args:
            obs_dim: Observation dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        # Value network
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.value_net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get value estimate.
        
        Args:
            obs: Observation tensor (batch_size, obs_dim)
        
        Returns:
            value: Estimated value (batch_size, 1)
        """
        return self.value_net(obs)


class AdversarialPolicy:
    """
    Complete adversarial policy with actor and critic.
    
    This class encapsulates the actor-critic architecture and provides
    convenient methods for action selection and training.
    """
    
    def __init__(
        self,
        obs_dim: int = 12,
        hidden_dims: Tuple[int, ...] = (128, 128),
        heading_range: float = 30.0,
        speed_range: float = 10.0,
        altitude_actions: int = 3,
        device: str = 'cpu'
    ):
        """
        Initialize adversarial policy.
        
        Args:
            obs_dim: Observation dimension
            hidden_dims: Hidden layer dimensions for both actor and critic
            heading_range: Maximum heading change in degrees
            speed_range: Maximum speed change in m/s
            altitude_actions: Number of discrete altitude actions
            device: Device to run the networks on
        """
        self.device = device
        
        self.actor = AdversarialActor(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims,
            heading_range=heading_range,
            speed_range=speed_range,
            altitude_actions=altitude_actions,
            device=device
        ).to(device)
        
        self.critic = AdversarialCritic(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims
        ).to(device)
    
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Select action given observation.
        
        Args:
            obs: Observation array
            deterministic: If True, return deterministic action
        
        Returns:
            Action dictionary compatible with environment
        """
        return self.actor.get_action(obs, deterministic)
    
    def get_value(self, obs: np.ndarray) -> float:
        """
        Get value estimate for observation.
        
        Args:
            obs: Observation array
        
        Returns:
            Value estimate
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            value = self.critic(obs_tensor)
            return value.cpu().item()
    
    def save(self, path: str):
        """Save policy to file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load policy from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
    
    def info(self) -> str:
        """Return policy information string."""
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        
        return (
            f"AdversarialPolicy(\n"
            f"  Device: {self.device}\n"
            f"  Actor parameters: {actor_params:,}\n"
            f"  Critic parameters: {critic_params:,}\n"
            f"  Total parameters: {actor_params + critic_params:,}\n"
            f")"
        )


# Example usage
if __name__ == "__main__":
    # Create adversarial policy
    policy = AdversarialPolicy(obs_dim=12, device='cpu')
    print(policy.info())
    
    # Test action selection
    obs = np.random.randn(12)
    action = policy.select_action(obs, deterministic=False)
    print(f"\nSample action: {action}")
    
    value = policy.get_value(obs)
    print(f"Value estimate: {value:.4f}")
