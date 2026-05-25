"""
Multi-Agent Environment for Adversarial Training

This module extends ConflictResolutionEnv to support simultaneous decision-making
for both the protagonist aircraft (conflict resolution policy) and background aircraft
(adversarial policy).

Design Philosophy:
    - Protagonist observes all aircraft and makes resolution decisions
    - Each background aircraft observes its local environment and acts adversarially
    - Separate reward functions for protagonist and adversarial agents
    - Synchronized episode management and termination
    - Support for importance sampling for unbiased evaluation

Key Features:
    - Multi-agent observation spaces (protagonist + N background aircraft)
    - Parallel action execution with conflict checking
    - Separate reward shaping for protagonist and adversaries
    - Episode termination based on protagonist's goal achievement
    - Importance ratio tracking for off-policy correction
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Any
import bluesky as bs
from bluesky.tools import geo

from .conflict_resolution_env import ConflictResolutionEnv
from .common.constants import (
    ALTITUDE_LAYERS,
    ACTION_FREQUENCY,
    D_HEADING_DEG,
    D_SPEED_MS,
    CRITICAL_TCPA_S,
    CRITICAL_TAU_S,
    CRITICAL_DIST_NM,
    NM2M,
)


class MultiAgentEnv(ConflictResolutionEnv):
    """
    Multi-agent environment for adversarial training.
    
    Extends ConflictResolutionEnv to support both protagonist and adversarial policies.
    The protagonist aims to reach its goal safely, while adversarial background aircraft
    try to create challenging scenarios.
    """
    
    # Adversarial reward components
    INCREASE_COMPLEXITY_REWARD = 0.5    # Reward for creating more conflicts
    APPROACH_PROTAGONIST_REWARD = 0.3   # Reward for getting closer to protagonist
    PROTAGONIST_FAILURE_REWARD = 1.0    # Reward when protagonist fails
    ADVERSARY_COLLISION_PENALTY = -2.0  # Penalty for adversary collision
    ADVERSARY_RULE_VIOLATION_PENALTY = -1.0  # Penalty for violating flight rules
    ADVERSARY_SMOOTH_CONTROL_BONUS = 0.02    # Bonus for smooth control
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        scenario_type: str = 'head_on',
        num_intruders: int = 3,
        disturbance_level: str = 'none',
        enable_nfz: bool = False,
        nfz_config: Optional[Dict] = None,
        enable_adversarial: bool = True,
        adversarial_agents: Optional[List[str]] = None,
        importance_sampling: bool = False
    ):
        """
        Initialize multi-agent environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            scenario_type: Type of conflict scenario
            num_intruders: Number of intruder aircraft
            disturbance_level: Natural disturbance level
            enable_nfz: Whether to enable no-fly zones
            nfz_config: No-fly zone configuration
            enable_adversarial: Whether to enable adversarial training
            adversarial_agents: List of aircraft IDs to be controlled by adversarial policy
                               If None, all intruders become adversarial
            importance_sampling: Whether to track importance ratios
        """
        super().__init__(
            render_mode=render_mode,
            scenario_type=scenario_type,
            num_intruders=num_intruders,
            disturbance_level=disturbance_level,
            enable_nfz=enable_nfz,
            nfz_config=nfz_config
        )
        
        self.enable_adversarial = enable_adversarial
        self.importance_sampling = importance_sampling
        
        # Determine adversarial agents
        if adversarial_agents is not None:
            self.adversarial_agents = adversarial_agents
        else:
            # By default, all intruders except protagonist are adversarial
            self.adversarial_agents = []
        
        # Adversarial observation dimension:
        # [relative_lat(1), relative_lon(1), relative_alt(1), relative_hdg(1),
        #  relative_spd(1), distance(1), conflict_tcpa(1), conflict_dcpa(1),
        #  protagonist_relative_pos(3), altitude_layer(1)]
        self.adversary_obs_dim = 12
        
        # Store previous adversarial observations for computing rewards
        self.prev_adversary_obs = {}
        self.adversary_conflicts_count = {}
        self.adversary_prev_actions = {}
        
        # Importance sampling weights (for each adversarial agent)
        self.importance_ratios = {}
        self.behavior_log_probs = {}
        
        # Episode statistics for adversaries
        self.adversary_episode_rewards = {}
        self.adversary_episode_conflicts = {}
        
        # Episode statistics for protagonist (tracked by parent but not initialized)
        self.episode_reward = 0.0
        self.conflict_count = 0
        self.nfz_violation_count = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment and return initial observations.
        
        Returns:
            Tuple of (protagonist_obs, info_dict)
        """
        protagonist_obs, info = super().reset(seed=seed, options=options)
        
        # Identify adversarial agents from current scenario
        if self.enable_adversarial:
            # Get all intruder IDs (exclude protagonist 'OWNSHIP')
            all_agents = [acid for acid in bs.traf.id if acid != 'OWNSHIP']
            if not self.adversarial_agents:
                self.adversarial_agents = all_agents
        
        # Initialize adversarial tracking
        self.prev_adversary_obs = {}
        self.adversary_conflicts_count = {}
        self.adversary_prev_actions = {}
        self.adversary_episode_rewards = {acid: 0.0 for acid in self.adversarial_agents}
        self.adversary_episode_conflicts = {acid: 0 for acid in self.adversarial_agents}
        
        # Reset protagonist episode statistics
        self.episode_reward = 0.0
        self.conflict_count = 0
        self.nfz_violation_count = 0
        
        # Reset importance sampling
        if self.importance_sampling:
            self.importance_ratios = {acid: 1.0 for acid in self.adversarial_agents}
            self.behavior_log_probs = {}
        
        # Add adversarial observations to info
        if self.enable_adversarial:
            info['adversary_obs'] = self._get_adversary_observations()
        
        return protagonist_obs, info
    
    def step(
        self,
        action: Dict[str, np.ndarray],
        adversary_actions: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step with synchronised protagonist + adversary actions.

        Unlike the parent class, this method issues both protagonist and adversary
        commands before advancing the simulation by ``ACTION_FREQUENCY`` BlueSky steps.
        This ensures both agents act within the same time window and prevents the
        adversary from being delayed by one action cycle.

        Args:
            action: protagonist action dict (heading/speed/altitude).
            adversary_actions: ``{agent_id: action_dict}``; when ``None`` the
                adversary aircraft hold their current commands.

        Returns:
            ``(protagonist_obs, protagonist_reward, terminated, truncated, info)``
        """
        # 1. 记录上一拍对抗者观测（用于对抗奖励差分计算）
        if self.enable_adversarial:
            self.prev_adversary_obs = self._get_adversary_observations()

        # 2. 同步发送 protagonist 与 adversary 的指令到 BlueSky 命令栈
        self._get_action(action)
        if self.enable_adversarial and adversary_actions:
            for agent_id, agent_action in adversary_actions.items():
                if agent_id in bs.traf.id:
                    self._apply_adversary_action(agent_id, agent_action)

        # 3. 统一推进 ACTION_FREQUENCY 个仿真步
        for _ in range(ACTION_FREQUENCY):
            bs.sim.step()
            if self.enable_nfz and 'OWNSHIP' in bs.traf.id:
                idx = bs.traf.id2idx('OWNSHIP')
                if bs.traf.nfz.check_aircraft(
                    'OWNSHIP', bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.alt[idx]
                ):
                    self.total_nfz_violations += 1
                    self.nfz_violation_count += 1
            if self.render_mode == 'human':
                self._render_frame()

        self.episode_step += 1

        # 4. 计算 protagonist 观测与奖励（在删除飞机之前）
        obs = self._get_obs()
        protagonist_reward, terminated = self._get_reward()
        truncated = self.episode_step >= self.max_episode_steps
        info = self._get_info()

        # 5. 收集统计与对抗奖励
        self.episode_reward += protagonist_reward
        if bs.traf.cd.confpairs:
            ownship_pairs = [p for p in bs.traf.cd.confpairs if 'OWNSHIP' in p]
            self.conflict_count += len(ownship_pairs)

        if self.enable_adversarial:
            adversary_rewards = self._calculate_adversary_rewards(terminated)
            info['adversary_rewards'] = adversary_rewards
            info['adversary_obs'] = self._get_adversary_observations()
            for agent_id, reward in adversary_rewards.items():
                if agent_id in self.adversary_episode_rewards:
                    self.adversary_episode_rewards[agent_id] += reward

        if self.importance_sampling:
            # Do NOT filter importance_ratios to currently-alive adversaries.
            # IS weights accumulated before an adversary is removed by bs.traf.delete
            # must be retained; otherwise weights for the high-value "failure" scenarios
            # are silently dropped, biasing the pi_natural / pi_adversarial estimator.
            info['importance_ratios'] = self.importance_ratios.copy()
            info['episode_importance_weight'] = float(
                np.prod(list(self.importance_ratios.values()))
                if self.importance_ratios else 1.0
            )

        info['is_critical_state'] = self._is_critical_state()

        # 6. Episode 结束清理（与父类一致）
        if terminated or truncated:
            for acid in list(bs.traf.id):
                bs.traf.delete(bs.traf.id2idx(acid))

        return obs, protagonist_reward, terminated, truncated, info
    
    def _get_adversary_observations(self) -> Dict[str, np.ndarray]:
        """
        Get observations for all adversarial agents.
        
        Returns:
            Dictionary mapping agent_id -> observation array
        """
        adversary_obs = {}
        
        for agent_id in self.adversarial_agents:
            if agent_id not in bs.traf.id:
                continue
            
            idx = bs.traf.id.index(agent_id)
            obs = self._compute_adversary_observation(idx)
            adversary_obs[agent_id] = obs
        
        return adversary_obs
    
    def _compute_adversary_observation(self, agent_idx: int) -> np.ndarray:
        """
        Compute observation for a single adversarial agent.
        
        Observation includes:
        - Relative position to protagonist (lat, lon, alt)
        - Relative heading and speed
        - Distance to protagonist
        - Conflict information (tcpa, dcpa)
        - Protagonist's relative position to its goal
        - Current altitude layer
        
        Args:
            agent_idx: Index of the adversarial agent in bs.traf
        
        Returns:
            Observation array (12,)
        """
        # Get protagonist index
        try:
            protagonist_idx = bs.traf.id.index('OWNSHIP')
        except ValueError:
            # Protagonist not found, return zero observation
            return np.zeros(self.adversary_obs_dim, dtype=np.float32)
        
        # Get agent and protagonist positions
        agent_lat = bs.traf.lat[agent_idx]
        agent_lon = bs.traf.lon[agent_idx]
        agent_alt = bs.traf.alt[agent_idx]
        agent_hdg = bs.traf.hdg[agent_idx]
        agent_spd = bs.traf.gs[agent_idx]
        
        prot_lat = bs.traf.lat[protagonist_idx]
        prot_lon = bs.traf.lon[protagonist_idx]
        prot_alt = bs.traf.alt[protagonist_idx]
        prot_hdg = bs.traf.hdg[protagonist_idx]
        prot_spd = bs.traf.gs[protagonist_idx]
        
        # Compute relative position (normalized)
        distance = geo.latlondist(agent_lat, agent_lon, prot_lat, prot_lon) / 1852.0  # nm
        bearing_result = geo.qdrpos(agent_lat, agent_lon, prot_lat, prot_lon)  # degrees
        bearing = bearing_result[0] if isinstance(bearing_result, tuple) else bearing_result
        
        relative_lat = (prot_lat - agent_lat) / 0.1  # Normalize by ~10 nm
        relative_lon = (prot_lon - agent_lon) / 0.1
        relative_alt = (prot_alt - agent_alt) / 1000.0  # Normalize by 1000m
        relative_hdg = ((prot_hdg - agent_hdg + 180) % 360 - 180) / 180.0  # [-1, 1]
        relative_spd = (prot_spd - agent_spd) / 50.0  # Normalize by typical speed
        
        # Conflict information (注意 confpairs 元素是 (acid_str, acid_str)，
        # tcpa/dcpa 是按 pair 位置索引的一维数组)
        tcpa = np.inf
        dcpa = np.inf
        agent_id = bs.traf.id[agent_idx]
        for k, pair in enumerate(bs.traf.cd.confpairs):
            if agent_id in pair and 'OWNSHIP' in pair:
                if k < len(bs.traf.cd.tcpa):
                    tcpa = float(bs.traf.cd.tcpa[k]) / 300.0
                if k < len(bs.traf.cd.dcpa):
                    dcpa = float(bs.traf.cd.dcpa[k]) / NM2M
                break

        tcpa = float(np.clip(tcpa, 0, 1))
        dcpa = float(np.clip(dcpa, 0, 1))
        
        # Protagonist relative position to goal
        if hasattr(self, 'target_lat') and hasattr(self, 'target_lon'):
            prot_goal_dist = geo.latlondist(prot_lat, prot_lon, 
                                            self.target_lat, self.target_lon) / 1852.0
            prot_goal_bearing_result = geo.qdrpos(prot_lat, prot_lon, 
                                          self.target_lat, self.target_lon)
            prot_goal_bearing = prot_goal_bearing_result[0] if isinstance(prot_goal_bearing_result, tuple) else prot_goal_bearing_result
            prot_drift = abs((prot_goal_bearing - prot_hdg + 180) % 360 - 180) / 180.0
        else:
            prot_goal_dist = 0.0
            prot_goal_bearing = 0.0
            prot_drift = 0.0
        
        prot_goal_dist = np.clip(prot_goal_dist / 50.0, 0, 1)  # Normalize by 50 nm
        
        # Current altitude layer (for discrete altitude environment)
        altitude_layers = [500, 1000, 1500, 2000, 2500]
        altitude_layer = np.argmin([abs(agent_alt - alt) for alt in altitude_layers]) / 4.0  # [0, 1]
        
        # Construct observation vector
        obs = np.array([
            relative_lat,
            relative_lon,
            relative_alt,
            relative_hdg,
            relative_spd,
            np.clip(distance / 20.0, 0, 1),  # Normalize by 20 nm
            tcpa,
            dcpa,
            prot_goal_dist,
            prot_drift,
            np.clip(bearing / 360.0, 0, 1),
            altitude_layer
        ], dtype=np.float32)
        
        return obs
    
    def _apply_adversary_action(
        self,
        agent_id: str,
        action: Dict[str, np.ndarray]
    ):
        """
        Apply adversarial action to a background aircraft.
        
        Args:
            agent_id: Aircraft ID
            action: Action dict with 'heading', 'speed', 'altitude'
        """
        # Check aircraft exists before any operations
        if agent_id not in bs.traf.id:
            return
        
        try:
            idx = bs.traf.id.index(agent_id)
            
            # Extract actions (handle both array and scalar inputs)
            heading_delta = float(action['heading'][0] if hasattr(action['heading'], '__len__') else action['heading']) * 30.0
            speed_delta = float(action['speed'][0] if hasattr(action['speed'], '__len__') else action['speed']) * 10.0
            altitude_action = int(action['altitude'])        # 0: down, 1: maintain, 2: up
            
            # Apply heading change (verify aircraft still exists)
            if agent_id in bs.traf.id:
                current_hdg = bs.traf.hdg[idx]
                new_heading = (current_hdg + heading_delta) % 360
                bs.stack.stack(f"HDG {agent_id} {new_heading:.1f}")
            
            # Apply speed change (verify aircraft still exists)
            if agent_id in bs.traf.id:
                idx = bs.traf.id.index(agent_id)  # Re-get index in case it changed
                current_spd = bs.traf.gs[idx]
                new_speed = np.clip(current_spd + speed_delta, 50, 250)  # Realistic speed limits
                bs.stack.stack(f"SPD {agent_id} {new_speed:.1f}")
            
            # Apply altitude change (verify aircraft still exists)
            if agent_id in bs.traf.id:
                idx = bs.traf.id.index(agent_id)  # Re-get index in case it changed
                current_alt = bs.traf.alt[idx]
                altitude_layers = [500, 1000, 1500, 2000, 2500]
                current_layer = np.argmin([abs(current_alt - alt) for alt in altitude_layers])
                
                if altitude_action == 0:  # Down
                    target_layer = max(0, current_layer - 1)
                elif altitude_action == 2:  # Up
                    target_layer = min(len(altitude_layers) - 1, current_layer + 1)
                else:  # Maintain
                    target_layer = current_layer
                
                target_altitude = altitude_layers[target_layer]
                bs.stack.stack(f"ALT {agent_id} {target_altitude}")
            
            # Store action for smooth control reward (only if aircraft still exists)
            if agent_id in bs.traf.id:
                self.adversary_prev_actions[agent_id] = action
        except (ValueError, IndexError):
            # Aircraft disappeared during action application, silently ignore
            pass
    
    def _calculate_adversary_rewards(
        self,
        protagonist_terminated: bool
    ) -> Dict[str, float]:
        """
        Calculate rewards for all adversarial agents.
        
        Reward components:
        - Increase conflict complexity: +0.5 for creating new conflicts
        - Approach protagonist: +0.3 for reducing distance
        - Protagonist failure: +1.0 if protagonist crashes or times out
        - Collision: -2.0 for adversary collision
        - Rule violation: -1.0 for violating NFZ or altitude limits
        - Smooth control: +0.02 for consistent actions
        
        Args:
            protagonist_terminated: Whether protagonist episode terminated
        
        Returns:
            Dictionary mapping agent_id -> reward
        """
        adversary_rewards = {}
        
        for agent_id in self.adversarial_agents:
            if agent_id not in bs.traf.id:
                adversary_rewards[agent_id] = 0.0
                continue
            
            reward = 0.0
            idx = bs.traf.id.index(agent_id)
            
            # 1. Conflict complexity reward
            current_obs = self._compute_adversary_observation(idx)
            if agent_id in self.prev_adversary_obs:
                prev_obs = self.prev_adversary_obs[agent_id]
                
                # Check if TCPA decreased (approaching conflict)
                prev_tcpa = prev_obs[6]
                curr_tcpa = current_obs[6]
                if curr_tcpa < prev_tcpa and curr_tcpa < 0.5:  # Within 2.5 min
                    reward += self.INCREASE_COMPLEXITY_REWARD
                
                # 2. Approach protagonist reward
                prev_dist = prev_obs[5]
                curr_dist = current_obs[5]
                if curr_dist < prev_dist:
                    reward += self.APPROACH_PROTAGONIST_REWARD * (prev_dist - curr_dist)
            
            # 3. Protagonist failure reward
            if protagonist_terminated:
                # Check if it was a crash (not goal achievement)
                try:
                    prot_idx = bs.traf.id.index('OWNSHIP')
                    goal_reached = hasattr(self, '_goal_reached') and self._goal_reached
                    if not goal_reached:
                        reward += self.PROTAGONIST_FAILURE_REWARD
                except ValueError:
                    pass
            
            # 4. Collision penalty —— 用 confpairs 位置索引取 dcpa
            for k, pair in enumerate(bs.traf.cd.confpairs):
                if agent_id in pair and k < len(bs.traf.cd.dcpa):
                    dcpa_nm = float(bs.traf.cd.dcpa[k]) / NM2M
                    if dcpa_nm < 0.1:
                        reward += self.ADVERSARY_COLLISION_PENALTY
                        self.adversary_episode_conflicts[agent_id] = (
                            self.adversary_episode_conflicts.get(agent_id, 0) + 1
                        )
                        break

            # 5. Rule violation penalty
            # Check NFZ violation
            if self.enable_nfz:
                lat = bs.traf.lat[idx]
                lon = bs.traf.lon[idx]
                alt = bs.traf.alt[idx]
                if bs.traf.nfz.check_aircraft(agent_id, lat, lon, alt):
                    reward += self.ADVERSARY_RULE_VIOLATION_PENALTY
            
            # Check altitude limits
            alt = bs.traf.alt[idx]
            if alt < 300 or alt > 3000:  # Outside safe altitude range
                reward += self.ADVERSARY_RULE_VIOLATION_PENALTY
            
            # 6. Smooth control bonus
            if agent_id in self.adversary_prev_actions:
                # Reward small action changes
                reward += self.ADVERSARY_SMOOTH_CONTROL_BONUS
            
            adversary_rewards[agent_id] = float(reward)
        
        return adversary_rewards
    
    def update_importance_ratio(
        self,
        agent_id: str,
        adversarial_log_prob: float,
        natural_log_prob: Optional[float] = None,
        action: Optional[Dict[str, np.ndarray]] = None,
    ) -> float:
        """累计重要性采样权重 ``w = \u03c0_natural(a|s) / \u03c0_adversarial(a|s)``。

        该权重用于将在对抗环境中估计的衡量映射回自然分布，从而保证
        对真实冲突/失效率的估计理论上无偏（Feng et al., Nature 2023）。

        Args:
            agent_id: 对抗代理 ID。
            adversarial_log_prob: 对抗策略下动作的 log概率。
            natural_log_prob: 自然分布下动作的 log概率；若缺省则从
                ``action`` 与 ``bs.traf.disturb`` 推算。
            action: ``{'heading': ..., 'speed': ..., 'altitude': ...}``。

        Returns:
            本步重要性权重（已计入累积量）。
        """
        if not self.importance_sampling:
            return 1.0

        if natural_log_prob is None:
            if action is None:
                natural_log_prob = 0.0
            else:
                natural_log_prob = self._action_natural_log_prob(action)

        log_ratio = float(natural_log_prob) - float(adversarial_log_prob)
        # 限幅避免数值爆炸
        step_ratio = float(np.exp(np.clip(log_ratio, -10.0, 10.0)))

        prev = self.importance_ratios.get(agent_id, 1.0)
        self.importance_ratios[agent_id] = prev * step_ratio
        self.behavior_log_probs[agent_id] = natural_log_prob
        return step_ratio

    def _action_natural_log_prob(self, action: Dict[str, np.ndarray]) -> float:
        """把背景机输出的 (heading_delta, speed_delta) 当作对名义航迹的扰动量，
        调用 ``bs.traf.disturb.natural_log_prob`` 返回自然分布下的 log概率。“高度动作”为
        离散变量，正常巡航以 ``maintain`` 为众数，简化为在 ``maintain`` 上赋
        较高概率（从而令改变高度的 IS 权重偏小）。”"""
        if not hasattr(bs.traf, 'disturb'):
            return 0.0

        # 连续动作 → 实际偏移量
        h_norm = float(action['heading'][0] if hasattr(action['heading'], '__len__') else action['heading'])
        s_norm = float(action['speed'][0] if hasattr(action['speed'], '__len__') else action['speed'])
        dhdg = h_norm * D_HEADING_DEG
        dspd = s_norm * D_SPEED_MS

        log_p = bs.traf.disturb.natural_log_prob(dhdg=dhdg, dspd=dspd)

        # 离散高度动作：maintain=0.9, up/down 各 0.05 作为自然先验
        altitude_action = int(action['altitude'])
        prior = (0.05, 0.9, 0.05)
        log_p += float(np.log(prior[altitude_action])) if 0 <= altitude_action < 3 else 0.0
        return float(log_p)
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        Get episode statistics for both protagonist and adversaries.
        
        Returns:
            Dictionary with episode statistics
        """
        stats = {
            'protagonist': {
                'total_reward': self.episode_reward,
                'conflicts': self.conflict_count,
                'nfz_violations': self.nfz_violation_count,
                'goal_reached': getattr(self, '_goal_reached', False)
            },
            'adversaries': {}
        }
        
        for agent_id in self.adversarial_agents:
            stats['adversaries'][agent_id] = {
                'total_reward': self.adversary_episode_rewards.get(agent_id, 0.0),
                'conflicts': self.adversary_episode_conflicts.get(agent_id, 0),
                'importance_ratio': self.importance_ratios.get(agent_id, 1.0)
            }
        
        return stats
    
    def _is_critical_state(self) -> bool:
        """判定当前是否为“关键状态”。

        任一条件成立即返回 True：
        - 距 protagonist 平面距离 < ``CRITICAL_DIST_NM``。
        - 存在涉及 protagonist 的官方冲突检测对。
        - 表示官方 TCPA 中最小值 < ``CRITICAL_TCPA_S``。
        - 任一背景机与 protagonist 接近率导出的 TAU < ``CRITICAL_TAU_S``。
        """
        if 'OWNSHIP' not in bs.traf.id:
            return False

        own_idx = bs.traf.id.index('OWNSHIP')
        own_lat, own_lon, own_alt = (
            bs.traf.lat[own_idx], bs.traf.lon[own_idx], bs.traf.alt[own_idx]
        )
        own_gs, own_hdg = bs.traf.gs[own_idx], bs.traf.hdg[own_idx]

        # 1) 直接平面距离
        for i, acid in enumerate(bs.traf.id):
            if acid == 'OWNSHIP':
                continue
            dist_nm = geo.latlondist(
                own_lat, own_lon, bs.traf.lat[i], bs.traf.lon[i]
            ) / NM2M
            if dist_nm < CRITICAL_DIST_NM:
                return True

        # 2) 官方冲突对
        if any('OWNSHIP' in pair for pair in bs.traf.cd.confpairs):
            return True

        # 3) 官方 TCPA
        if len(bs.traf.cd.tcpa) > 0:
            min_tcpa = float(np.min(np.abs(bs.traf.cd.tcpa)))
            if min_tcpa < CRITICAL_TCPA_S:
                return True

        # 4) 备用 TAU判判
        for intruder_id in bs.traf.id:
            if intruder_id == 'OWNSHIP':
                continue
            i = bs.traf.id.index(intruder_id)
            hor_dist_m = geo.latlondist(
                own_lat, own_lon, bs.traf.lat[i], bs.traf.lon[i]
            )
            ver_dist_m = abs(own_alt - bs.traf.alt[i])
            total_dist = float(np.hypot(hor_dist_m, ver_dist_m))
            own_vx = own_gs * np.sin(np.radians(own_hdg))
            own_vy = own_gs * np.cos(np.radians(own_hdg))
            int_vx = bs.traf.gs[i] * np.sin(np.radians(bs.traf.hdg[i]))
            int_vy = bs.traf.gs[i] * np.cos(np.radians(bs.traf.hdg[i]))
            closing = float(np.hypot(int_vx - own_vx, int_vy - own_vy))
            if closing > 1.0 and total_dist / closing < CRITICAL_TAU_S:
                return True
        return False


# Example usage
if __name__ == "__main__":
    # Create multi-agent environment
    env = MultiAgentEnv(
        scenario_type='head_on',
        num_intruders=3,
        enable_adversarial=True,
        importance_sampling=True
    )
    
    print("Multi-agent environment created")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Adversary obs dim: {env.adversary_obs_dim}")
