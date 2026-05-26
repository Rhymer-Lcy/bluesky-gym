""" 
2.5D Conflict Resolution Environment with discrete altitude layers.

This environment extends horizontal conflict resolution with vertical maneuvers
using discrete altitude layers (500m, 1000m, 1500m, 2000m, 2500m).
"""

import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

from bluesky_gym.envs.common.constants import (
    NM2KM,
    ALTITUDE_LAYERS,
    ACTION_FREQUENCY,
    VERTICAL_TRANSITION_RATE,
    DISTANCE_MARGIN_KM as DISTANCE_MARGIN,
    INTRUSION_DISTANCE_NM as INTRUSION_DISTANCE,
    VERTICAL_MARGIN_M as VERTICAL_MARGIN,
)


# Environment constants (env-specific)
DEFAULT_ALTITUDE_LAYER = 1500  # m (index 2)

# Rewards and penalties
REACH_REWARD = 1
DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1
ALTITUDE_CHANGE_PENALTY = -0.05  # Small penalty for altitude changes

# Aircraft parameters
NUM_INTRUDERS = 5
NUM_WAYPOINTS = 1

WAYPOINT_DISTANCE_MIN = 100
WAYPOINT_DISTANCE_MAX = 150

# Action parameters (env-specific: 25D uses 45° heading delta vs 30° in others)
D_HEADING = 45  # degrees
AC_SPD = 150  # m/s


class Discrete25DEnv(gym.Env):
    """
    2.5D Conflict Resolution Environment
    
    Action space:
    - Continuous heading change: [-1, 1] → [-45°, +45°]
    - Discrete altitude layer: {0, 1, 2} → {descend, maintain, climb}
    
    Observation space includes:
    - Horizontal state (intruder positions, waypoint info)
    - Vertical state (current altitude layer, relative altitudes)
    """
    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height)

        # Observation space includes horizontal and vertical information
        self.observation_space = spaces.Dict({
            # Horizontal observations
            "intruder_distance": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "cos_difference_pos": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "sin_difference_pos": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "x_difference_speed": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "y_difference_speed": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            
            # Vertical observations
            "current_altitude_layer": spaces.Box(0, len(ALTITUDE_LAYERS)-1, shape=(1,), dtype=np.float64),
            "altitude_difference": spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "intruder_altitude_layer": spaces.Box(0, len(ALTITUDE_LAYERS)-1, shape=(NUM_INTRUDERS,), dtype=np.float64),
            
            # Waypoint observations
            "waypoint_distance": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
            "cos_drift": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
            "sin_drift": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64)
        })
       
        # Action space: [heading_change, altitude_action]
        # heading_change: continuous [-1, 1]
        # altitude_action: discrete {0, 1, 2} → {descend one layer, maintain, climb one layer}
        self.action_space = spaces.Dict({
            "heading": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
            "altitude": spaces.Discrete(3)  # 0=down, 1=maintain, 2=up
        })

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize BlueSky
        if bs.sim is None:
            bs.init(mode='sim', detached=True)

        bs.scr = ScreenDummy()
        bs.stack.stack('DT 5;FF')

        # Logging variables
        self.total_reward = 0
        self.total_intrusions = 0
        self.total_altitude_changes = 0
        self.average_drift = np.array([])
        
        # Altitude layer tracking
        self.target_altitude_layer_idx = 2  # Start at 1500m
        self.current_altitude_layer_idx = 2

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        bs.traf.reset()

        # Reset logging
        self.total_reward = 0
        self.total_intrusions = 0
        self.total_altitude_changes = 0
        self.average_drift = np.array([])

        # Reset altitude layer
        self.target_altitude_layer_idx = 2  # 1500m
        self.current_altitude_layer_idx = 2

        # Create ownship at default altitude layer
        bs.traf.cre('KL001', actype="A320", acspd=AC_SPD, acalt=ALTITUDE_LAYERS[self.target_altitude_layer_idx])

        self._generate_conflicts()
        self._generate_waypoint()
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self._get_action(action)

        # Run simulation steps
        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
            if self.render_mode == "human":
                observation = self._get_obs()
                self._render_frame()

        observation = self._get_obs()
        reward, terminated = self._get_reward()

        info = self._get_info()

        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return observation, reward, terminated, False, info

    def _generate_conflicts(self, acid='KL001'):
        """ Generate intruders at various altitude layers """
        target_idx = bs.traf.id2idx(acid)
        
        for i in range(NUM_INTRUDERS):
            dpsi = np.random.randint(45, 315)
            cpa = np.random.randint(0, INTRUSION_DISTANCE)
            tlosh = np.random.randint(100, 1000)
            
            # Randomly assign altitude layers to intruders
            intruder_alt_layer_idx = np.random.randint(0, len(ALTITUDE_LAYERS))
            intruder_alt = ALTITUDE_LAYERS[intruder_alt_layer_idx]
            
            # No vertical separation for conflict generation
            dH = 0
            
            bs.traf.creconfs(
                acid=f'INTRUDER_{i+1}',
                actype="A320",
                targetidx=target_idx,
                dpsi=dpsi,
                dcpa=cpa,
                tlosh=tlosh,
                dH=dH,
                spd=AC_SPD
            )
            
            # Set intruder altitude after creation
            int_idx = bs.traf.id2idx(f'INTRUDER_{i+1}')
            bs.traf.alt[int_idx] = intruder_alt
            bs.traf.selalt[int_idx] = intruder_alt

    def _generate_waypoint(self, acid='KL001'):
        """ Generate waypoint for navigation """
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_reach = []
        
        for i in range(NUM_WAYPOINTS):
            wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
            wpt_hdg_init = 0

            ac_idx = bs.traf.id2idx(acid)

            wpt_lat, wpt_lon = fn.get_point_at_distance(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init
            )
            
            self.wpt_lat.append(wpt_lat)
            self.wpt_lon.append(wpt_lon)
            self.wpt_reach.append(0)

    def _get_obs(self):
        """ Get current observation """
        ac_idx = bs.traf.id2idx('KL001')

        # Horizontal observations
        intruder_distance = []
        cos_bearing = []
        sin_bearing = []
        x_difference_speed = []
        y_difference_speed = []
        
        # Vertical observations
        altitude_difference = []
        intruder_altitude_layer = []

        ac_hdg = bs.traf.hdg[ac_idx]
        ac_gs = bs.traf.gs[ac_idx]
        ac_alt = bs.traf.alt[ac_idx]
        
        # Update current altitude layer based on actual altitude
        self.current_altitude_layer_idx = self._get_altitude_layer_index(ac_alt)

        # Get intruder information
        for i in range(NUM_INTRUDERS):
            int_acid = f'INTRUDER_{i+1}'
            if int_acid in bs.traf.id:
                int_idx = bs.traf.id2idx(int_acid)
                
                # Horizontal info
                int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                    bs.traf.lat[int_idx], bs.traf.lon[int_idx]
                )
                intruder_distance.append(int_dis * NM2KM)
                
                bearing = ac_hdg - int_qdr
                bearing = fn.bound_angle_positive_negative_180(bearing)
                cos_bearing.append(np.cos(np.deg2rad(bearing)))
                sin_bearing.append(np.sin(np.deg2rad(bearing)))

                # Speed difference
                int_hdg = bs.traf.hdg[int_idx]
                int_gs = bs.traf.gs[int_idx]
                
                ac_vx = ac_gs * np.sin(np.deg2rad(ac_hdg))
                ac_vy = ac_gs * np.cos(np.deg2rad(ac_hdg))
                int_vx = int_gs * np.sin(np.deg2rad(int_hdg))
                int_vy = int_gs * np.cos(np.deg2rad(int_hdg))
                
                x_difference_speed.append(int_vx - ac_vx)
                y_difference_speed.append(int_vy - ac_vy)
                
                # Vertical info
                int_alt = bs.traf.alt[int_idx]
                altitude_difference.append(int_alt - ac_alt)
                int_alt_layer_idx = self._get_altitude_layer_index(int_alt)
                intruder_altitude_layer.append(float(int_alt_layer_idx))
            else:
                # Intruder deleted, fill with safe values
                intruder_distance.append(1000.0)
                cos_bearing.append(0.0)
                sin_bearing.append(0.0)
                x_difference_speed.append(0.0)
                y_difference_speed.append(0.0)
                altitude_difference.append(1000.0)
                intruder_altitude_layer.append(float(self.current_altitude_layer_idx))

        # Waypoint information
        waypoint_distance = []
        cos_drift = []
        sin_drift = []
        
        for i in range(NUM_WAYPOINTS):
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                self.wpt_lat[i], self.wpt_lon[i]
            )
            
            waypoint_distance.append(wpt_dis * NM2KM)
            
            drift = ac_hdg - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)
            cos_drift.append(np.cos(np.deg2rad(drift)))
            sin_drift.append(np.sin(np.deg2rad(drift)))

        return {
            "intruder_distance": np.array(intruder_distance, dtype=np.float64),
            "cos_difference_pos": np.array(cos_bearing, dtype=np.float64),
            "sin_difference_pos": np.array(sin_bearing, dtype=np.float64),
            "x_difference_speed": np.array(x_difference_speed, dtype=np.float64),
            "y_difference_speed": np.array(y_difference_speed, dtype=np.float64),
            "current_altitude_layer": np.array([self.current_altitude_layer_idx], dtype=np.float64),
            "altitude_difference": np.array(altitude_difference, dtype=np.float64),
            "intruder_altitude_layer": np.array(intruder_altitude_layer, dtype=np.float64),
            "waypoint_distance": np.array(waypoint_distance, dtype=np.float64),
            "cos_drift": np.array(cos_drift, dtype=np.float64),
            "sin_drift": np.array(sin_drift, dtype=np.float64)
        }

    def _get_altitude_layer_index(self, altitude):
        """ Get the closest altitude layer index for given altitude """
        distances = [abs(altitude - layer_alt) for layer_alt in ALTITUDE_LAYERS]
        return np.argmin(distances)

    def _get_action(self, action):
        """ Execute action: heading change + altitude layer change """
        # Extract actions
        if isinstance(action, dict):
            heading_action = action["heading"]
            altitude_action = action["altitude"]
        else:
            # Fallback if action format is different
            heading_action = action[0] if hasattr(action, '__len__') else action
            altitude_action = 1  # maintain

        # Heading change - convert to float to avoid txt2hdg error
        ac_idx = bs.traf.id2idx('KL001')
        current_heading = float(bs.traf.hdg[ac_idx])
        heading_change = float(heading_action[0] if hasattr(heading_action, '__len__') else heading_action)
        new_heading = current_heading + heading_change * D_HEADING
        
        # Use format to ensure proper float representation
        bs.stack.stack(f"HDG KL001 {new_heading:.1f}")

        # Altitude layer change
        if altitude_action == 0:  # Descend one layer
            self.target_altitude_layer_idx = max(0, self.target_altitude_layer_idx - 1)
            self.total_altitude_changes += 1
        elif altitude_action == 2:  # Climb one layer
            self.target_altitude_layer_idx = min(len(ALTITUDE_LAYERS) - 1, self.target_altitude_layer_idx + 1)
            self.total_altitude_changes += 1
        # altitude_action == 1: maintain current layer

        # Set target altitude
        target_alt = ALTITUDE_LAYERS[self.target_altitude_layer_idx]
        bs.stack.stack(f"ALT KL001 {target_alt:.1f}")
        bs.stack.stack(f"VS KL001 {VERTICAL_TRANSITION_RATE:.1f}")

    def _get_reward(self):
        """ Calculate reward based on conflicts, waypoint progress, and altitude changes """
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        terminated = False

        # Check for waypoint reach
        for i in range(NUM_WAYPOINTS):
            if self.wpt_reach[i] == 0:
                wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                    self.wpt_lat[i], self.wpt_lon[i]
                )
                
                if wpt_dis * NM2KM < DISTANCE_MARGIN:
                    reward += REACH_REWARD
                    self.wpt_reach[i] = 1
                    terminated = True
                else:
                    # Drift penalty
                    drift = bs.traf.hdg[ac_idx] - wpt_qdr
                    drift = fn.bound_angle_positive_negative_180(drift)
                    drift_norm = abs(drift) / 180.0
                    reward += DRIFT_PENALTY * drift_norm
                    self.average_drift = np.append(self.average_drift, abs(drift))

        # Check for intrusions (considering both horizontal and vertical separation)
        for i in range(NUM_INTRUDERS):
            int_acid = f'INTRUDER_{i+1}'
            if int_acid in bs.traf.id:
                int_idx = bs.traf.id2idx(int_acid)
                
                # Horizontal distance
                int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                    bs.traf.lat[int_idx], bs.traf.lon[int_idx]
                )
                
                # Vertical distance
                alt_diff = abs(bs.traf.alt[int_idx] - bs.traf.alt[ac_idx])
                
                # Conflict if within horizontal AND vertical margins
                if int_dis < INTRUSION_DISTANCE and alt_diff < VERTICAL_MARGIN:
                    reward += INTRUSION_PENALTY
                    self.total_intrusions += 1

        # Small penalty for altitude changes to encourage efficiency
        if self.target_altitude_layer_idx != self.current_altitude_layer_idx:
            reward += ALTITUDE_CHANGE_PENALTY

        self.total_reward += reward
        return reward, terminated

    def _get_info(self):
        """ Return logging information """
        return {
            "total_reward": self.total_reward,
            "total_intrusions": self.total_intrusions,
            "total_altitude_changes": self.total_altitude_changes,
            "average_drift": np.mean(self.average_drift) if len(self.average_drift) > 0 else 0,
            "current_altitude_layer": self.current_altitude_layer_idx,
            "target_altitude_layer": self.target_altitude_layer_idx
        }

    def _render_frame(self):
        """ Render environment (basic pygame visualization) """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 200  # km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135, 206, 235))

        ac_idx = bs.traf.id2idx('KL001')
        ac_lat = bs.traf.lat[ac_idx]
        ac_lon = bs.traf.lon[ac_idx]

        # Draw ownship (center)
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self.window_width // 2, self.window_height // 2),
            10
        )

        # Draw altitude layer indicator
        font = pygame.font.Font(None, 24)
        alt_text = font.render(
            f"Alt Layer: {self.current_altitude_layer_idx} ({ALTITUDE_LAYERS[self.current_altitude_layer_idx]}m)",
            True,
            (0, 0, 0)
        )
        canvas.blit(alt_text, (10, 10))

        # Draw intruders
        for i in range(NUM_INTRUDERS):
            int_acid = f'INTRUDER_{i+1}'
            if int_acid in bs.traf.id:
                int_idx = bs.traf.id2idx(int_acid)
                int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                    ac_lat, ac_lon,
                    bs.traf.lat[int_idx], bs.traf.lon[int_idx]
                )
                
                if int_dis * NM2KM < max_distance:
                    x = self.window_width // 2 + int(np.sin(np.deg2rad(int_qdr)) * int_dis * NM2KM / max_distance * self.window_width / 2)
                    y = self.window_height // 2 - int(np.cos(np.deg2rad(int_qdr)) * int_dis * NM2KM / max_distance * self.window_height / 2)
                    
                    # Color based on altitude difference
                    alt_diff = abs(bs.traf.alt[int_idx] - bs.traf.alt[ac_idx])
                    if alt_diff < VERTICAL_MARGIN:
                        color = (255, 165, 0)  # Orange - same layer, dangerous
                    else:
                        color = (0, 0, 255)  # Blue - different layer, safer
                    
                    pygame.draw.circle(canvas, color, (x, y), 8)

        # Draw waypoints
        for i in range(NUM_WAYPOINTS):
            if self.wpt_reach[i] == 0:
                wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(
                    ac_lat, ac_lon, self.wpt_lat[i], self.wpt_lon[i]
                )
                
                if wpt_dis * NM2KM < max_distance:
                    x = self.window_width // 2 + int(np.sin(np.deg2rad(wpt_qdr)) * wpt_dis * NM2KM / max_distance * self.window_width / 2)
                    y = self.window_height // 2 - int(np.cos(np.deg2rad(wpt_qdr)) * wpt_dis * NM2KM / max_distance * self.window_height / 2)
                    pygame.draw.circle(canvas, (0, 255, 0), (x, y), 8)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
