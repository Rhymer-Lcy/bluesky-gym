"""
Comprehensive Conflict Resolution Environment for UAV Testing.

This environment integrates all features from Phase 2:
- Discrete 2.5D altitude layers (500m-2500m)
- Typical conflict scenarios (head-on, crossing, merging, overtaking)
- Natural disturbances (GPS errors, wind, control delays)
- No-fly zones (circular and polygonal)

Designed for testing RL-based conflict resolution agents with:
- Standardized observation space (ego state, conflicts, NFZ, waypoints)
- Standardized action space (heading, speed, altitude layer)
- Comprehensive reward shaping (safety, efficiency, smoothness)
- Episode management (initialization, termination, reset)
"""

import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
from bluesky_gym.envs.scenarios import get_scenario

import gymnasium as gym
from gymnasium import spaces

from bluesky_gym.envs.common.constants import (
    NM2KM,
    ALTITUDE_LAYERS,
    ACTION_FREQUENCY,
    D_HEADING_DEG as D_HEADING,
    D_SPEED_MS as D_SPEED,
)


# Environment constants (env-specific; shared scalars are imported above)
DISTANCE_MARGIN = 5  # km
VERTICAL_TRANSITION_RATE = 5  # m/s

# Conflict detection parameters
NUM_INTRUDERS = 5
INTRUSION_DISTANCE = 5  # NM
VERTICAL_MARGIN = 300  # m

# Navigation parameters
WAYPOINT_DISTANCE_MIN = 100  # km
WAYPOINT_DISTANCE_MAX = 150  # km

# Action parameters
AC_SPD = 150  # m/s, default cruise speed

# Reward parameters
REACH_REWARD = 10.0
CONFLICT_PENALTY = -5.0
NFZ_VIOLATION_PENALTY = -2.0
DRIFT_PENALTY = -0.1
ALTITUDE_CHANGE_PENALTY = -0.05
SMOOTH_CONTROL_BONUS = 0.05


class ConflictResolutionEnv(gym.Env):
    """
    Standard conflict resolution environment for testing UAV agents.
    
    Features:
    - 2.5D airspace with discrete altitude layers
    - Multiple conflict scenarios with configurable difficulty
    - Natural disturbances (position, speed, heading, altitude)
    - No-fly zones (circular and polygonal)
    - Comprehensive observation space (ego, conflicts, NFZ, waypoints)
    - Flexible action space (heading, speed, altitude)
    - Shaped rewards (safety, efficiency, smoothness)
    
    Action Space:
        - heading: continuous [-1, 1] → [-30°, +30°]
        - speed: continuous [-1, 1] → [-10 m/s, +10 m/s]
        - altitude: discrete {0, 1, 2} → {down, maintain, up}
    
    Observation Space:
        - ego_state: [lat, lon, alt, hdg, spd, current_alt_layer]
        - target_state: [distance, bearing_cos, bearing_sin, drift]
        - conflict_info: [closest_tcpa, closest_dcpa, num_conflicts, closest_alt_diff]
        - nfz_info: [min_distance_to_nfz, is_in_nfz]
        - intruder_relative: [distance, bearing, alt_diff, speed_diff] x N
    """
    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(self, 
                 render_mode=None,
                 scenario_type='head_on',
                 num_intruders=5,
                 disturbance_level='none',
                 enable_nfz=True,
                 nfz_config=None):
        """
        Initialize conflict resolution environment.
        
        Arguments:
        - render_mode: 'human' for pygame display, None for no rendering
        - scenario_type: 'head_on', 'crossing', 'merging', 'overtaking', or 'random'
        - num_intruders: Number of intruder aircraft
        - disturbance_level: 'none', 'light', 'medium', 'heavy'
        - enable_nfz: Whether to enable no-fly zones
        - nfz_config: Dict with NFZ configuration (optional)
        """
        self.window_width = 1200
        self.window_height = 900
        self.window_size = (self.window_width, self.window_height)

        # Environment configuration
        self.scenario_type = scenario_type
        self.num_intruders = num_intruders
        self.disturbance_level = disturbance_level
        self.enable_nfz = enable_nfz
        self.nfz_config = nfz_config or {}

        # Observation space
        # 6 ego state + 4 target state + 4 conflict info + 2 NFZ info + 4*N intruder info
        obs_dim = 6 + 4 + 4 + 2 + 4 * self.num_intruders
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float64
        )
       
        # Action space
        self.action_space = spaces.Dict({
            "heading": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
            "speed": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
            "altitude": spaces.Discrete(3)  # 0=down, 1=maintain, 2=up
        })

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize BlueSky
        if bs.sim is None:
            bs.init(mode='sim', detached=True)

        bs.scr = ScreenDummy()
        bs.stack.stack('DT 5;FF')

        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 1000
        
        # Logging
        self.total_reward = 0
        self.total_conflicts = 0
        self.total_nfz_violations = 0
        self.total_altitude_changes = 0
        self.trajectory_deviation = []
        
        # Altitude layer tracking
        self.target_altitude_layer_idx = 2
        self.current_altitude_layer_idx = 2

        # Rendering
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        bs.traf.reset()

        # Reset tracking
        self.episode_step = 0
        self.total_reward = 0
        self.total_conflicts = 0
        self.total_nfz_violations = 0
        self.total_altitude_changes = 0
        self.trajectory_deviation = []

        # Reset altitude
        self.target_altitude_layer_idx = 2
        self.current_altitude_layer_idx = 2

        # Create ownship
        bs.traf.cre(
            'OWNSHIP',
            actype="A320",
            aclat=52.0,
            aclon=4.0,
            achdg=0,
            acalt=ALTITUDE_LAYERS[self.target_altitude_layer_idx],
            acspd=AC_SPD
        )

        # Generate conflict scenario
        self._generate_scenario()
        
        # Generate waypoint
        self._generate_waypoint()
        
        # Setup no-fly zones
        if self.enable_nfz:
            self._setup_nfz()
        
        # Set disturbance level
        bs.traf.disturb.set_preset(self.disturbance_level)
        
        # Enable conflict detection
        bs.stack.stack('CDMETHOD ON')

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self._get_action(action)

        # Run simulation
        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
            
            # Check NFZ violations during simulation
            if self.enable_nfz and 'OWNSHIP' in bs.traf.id:
                idx = bs.traf.id2idx('OWNSHIP')
                violations = bs.traf.nfz.check_aircraft(
                    'OWNSHIP', bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.alt[idx]
                )
                if violations:
                    self.total_nfz_violations += 1
            
            if self.render_mode == "human":
                self._render_frame()

        self.episode_step += 1

        observation = self._get_obs()
        reward, terminated = self._get_reward()
        truncated = self.episode_step >= self.max_episode_steps

        info = self._get_info()

        if terminated or truncated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return observation, reward, terminated, truncated, info

    def _generate_scenario(self):
        """ Generate conflict scenario based on configuration """
        # Select scenario type
        if self.scenario_type == 'random':
            scenario_types = ['head_on', 'crossing', 'merging', 'overtaking']
            selected_type = np.random.choice(scenario_types)
        else:
            selected_type = self.scenario_type
        
        # Get scenario class
        ScenarioClass = get_scenario(selected_type)
        
        # Create scenario instance with random parameters
        scenario = ScenarioClass(num_intruders=self.num_intruders)
        
        # Generate scenario
        scenario.generate(target_acid='OWNSHIP', actype='A320')
        
        # Randomly assign altitude layers to intruders
        for i in range(self.num_intruders):
            int_acid = f'INTRUDER_{i+1}'
            if int_acid in bs.traf.id:
                int_idx = bs.traf.id2idx(int_acid)
                alt_layer_idx = np.random.randint(0, len(ALTITUDE_LAYERS))
                bs.traf.alt[int_idx] = ALTITUDE_LAYERS[alt_layer_idx]
                bs.traf.selalt[int_idx] = ALTITUDE_LAYERS[alt_layer_idx]

    def _generate_waypoint(self):
        """ Generate target waypoint """
        wpt_dis = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
        wpt_hdg = 0  # Straight ahead

        ac_idx = bs.traf.id2idx('OWNSHIP')
        self.wpt_lat, self.wpt_lon = fn.get_point_at_distance(
            bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis, wpt_hdg
        )
        self.wpt_reached = False

    def _setup_nfz(self):
        """ Setup no-fly zones """
        bs.traf.nfz.clear_zones()
        
        # Default NFZ configuration if not provided
        if not self.nfz_config:
            # Create 1-2 random no-fly zones
            num_zones = np.random.randint(1, 3)
            
            for i in range(num_zones):
                if np.random.random() < 0.5:
                    # Circular zone
                    lat = 52.0 + np.random.uniform(-0.3, 0.3)
                    lon = 4.0 + np.random.uniform(-0.3, 0.3)
                    radius = np.random.uniform(5, 15)
                    alt_max = np.random.choice([2000, 3000, 99999])
                    
                    bs.traf.nfz.create_circular_zone(
                        name=f'NFZ_CIRCLE_{i}',
                        lat=lat,
                        lon=lon,
                        radius=radius,
                        alt_min=0,
                        alt_max=alt_max
                    )
                else:
                    # Polygonal zone
                    center_lat = 52.0 + np.random.uniform(-0.3, 0.3)
                    center_lon = 4.0 + np.random.uniform(-0.3, 0.3)
                    size = 0.1
                    
                    lats = [center_lat, center_lat + size, center_lat + size, center_lat]
                    lons = [center_lon, center_lon, center_lon + size, center_lon + size]
                    
                    bs.traf.nfz.create_polygon_zone(
                        name=f'NFZ_POLY_{i}',
                        lats=lats,
                        lons=lons,
                        alt_min=0,
                        alt_max=99999
                    )
        else:
            # Use provided NFZ configuration
            for zone_config in self.nfz_config.get('zones', []):
                if zone_config['type'] == 'circle':
                    bs.traf.nfz.create_circular_zone(**zone_config)
                elif zone_config['type'] == 'polygon':
                    bs.traf.nfz.create_polygon_zone(**zone_config)

    def _get_obs(self):
        """
        Get standardized observation.
        
        Observation structure:
        [0-5]: ego_state (lat_norm, lon_norm, alt_norm, hdg_norm, spd_norm, alt_layer_norm)
        [6-9]: target_state (dist_norm, bearing_cos, bearing_sin, drift_norm)
        [10-13]: conflict_info (tcpa_norm, dcpa_norm, num_conflicts_norm, alt_diff_norm)
        [14-15]: nfz_info (min_dist_norm, is_violated)
        [16+]: intruder_relative (dist, bearing_cos, bearing_sin, alt_diff) x N
        """
        if 'OWNSHIP' not in bs.traf.id:
            return np.zeros(self.observation_space.shape[0], dtype=np.float64)
        
        ac_idx = bs.traf.id2idx('OWNSHIP')
        
        # Ego state
        lat = bs.traf.lat[ac_idx] / 90.0  # Normalize to [-1, 1]
        lon = bs.traf.lon[ac_idx] / 180.0
        alt = bs.traf.alt[ac_idx] / 5000.0  # Normalize to ~[0, 1]
        hdg = bs.traf.hdg[ac_idx] / 180.0  # Normalize to [-1, 1]
        spd = bs.traf.gs[ac_idx] / 200.0  # Normalize to ~[0, 1]
        
        self.current_altitude_layer_idx = self._get_altitude_layer_index(bs.traf.alt[ac_idx])
        alt_layer = self.current_altitude_layer_idx / (len(ALTITUDE_LAYERS) - 1)
        
        ego_state = np.array([lat, lon, alt, hdg, spd, alt_layer])
        
        # Target state
        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(
            bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
            self.wpt_lat, self.wpt_lon
        )
        
        dist_norm = wpt_dis * NM2KM / 200.0  # Normalize by 200km
        bearing_cos = np.cos(np.deg2rad(wpt_qdr))
        bearing_sin = np.sin(np.deg2rad(wpt_qdr))
        
        drift = bs.traf.hdg[ac_idx] - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)
        drift_norm = drift / 180.0
        
        target_state = np.array([dist_norm, bearing_cos, bearing_sin, drift_norm])
        
        # Conflict info
        # tcpa/dcpa are 1-D arrays indexed by confpairs position
        closest_tcpa = 999.0
        closest_dcpa_nm = 999.0      # 真实 DCPA (NM)
        closest_range_nm = 999.0     # 当前欧式距离 (NM)
        num_conflicts = 0
        closest_alt_diff = 999.0

        # find minimum TCPA/DCPA for pairs involving OWNSHIP
        for k, pair in enumerate(bs.traf.cd.confpairs):
            if 'OWNSHIP' not in pair:
                continue
            num_conflicts += 1
            if k < len(bs.traf.cd.tcpa):
                tcpa_k = float(bs.traf.cd.tcpa[k])
                if tcpa_k < closest_tcpa:
                    closest_tcpa = tcpa_k
            if k < len(bs.traf.cd.dcpa):
                dcpa_k = float(bs.traf.cd.dcpa[k]) / 1852.0
                if dcpa_k < closest_dcpa_nm:
                    closest_dcpa_nm = dcpa_k

        for i in range(self.num_intruders):
            int_acid = f'INTRUDER_{i+1}'
            if int_acid in bs.traf.id:
                int_idx = bs.traf.id2idx(int_acid)
                int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                    bs.traf.lat[int_idx], bs.traf.lon[int_idx]
                )
                alt_diff = abs(bs.traf.alt[int_idx] - bs.traf.alt[ac_idx])
                if int_dis < closest_range_nm:
                    closest_range_nm = int_dis
                if alt_diff < closest_alt_diff:
                    closest_alt_diff = alt_diff

        conflict_info = np.array([
            closest_tcpa / 300.0,        # Normalize by 5 minutes
            closest_dcpa_nm / 10.0,      # Normalize by 10 NM (真实 DCPA)
            num_conflicts / 10.0,
            closest_alt_diff / 3000.0
        ])
        
        # NFZ info
        min_nfz_dist = 999.0
        is_violated = 0.0
        
        if self.enable_nfz:
            violations = bs.traf.nfz.check_point(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.alt[ac_idx]
            )
            is_violated = 1.0 if violations else 0.0
            
            # Calculate minimum distance to any NFZ
            for zone in bs.traf.nfz.zones:
                if zone['type'] == 'circle':
                    dist = bs.tools.geo.kwikdist(
                        zone['lat'], zone['lon'],
                        bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]
                    )
                    dist_to_edge = abs(dist - zone['radius'])
                    min_nfz_dist = min(min_nfz_dist, dist_to_edge)
        
        nfz_info = np.array([min_nfz_dist / 50.0, is_violated])  # Normalize by 50 NM
        
        # Intruder relative states
        intruder_states = []
        for i in range(self.num_intruders):
            int_acid = f'INTRUDER_{i+1}'
            if int_acid in bs.traf.id:
                int_idx = bs.traf.id2idx(int_acid)
                
                int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                    bs.traf.lat[int_idx], bs.traf.lon[int_idx]
                )
                
                bearing = bs.traf.hdg[ac_idx] - int_qdr
                bearing = fn.bound_angle_positive_negative_180(bearing)
                
                alt_diff = bs.traf.alt[int_idx] - bs.traf.alt[ac_idx]
                spd_diff = bs.traf.gs[int_idx] - bs.traf.gs[ac_idx]
                
                intruder_states.extend([
                    int_dis * NM2KM / 100.0,  # Normalize by 100km
                    np.cos(np.deg2rad(bearing)),
                    np.sin(np.deg2rad(bearing)),
                    alt_diff / 3000.0  # Normalize by 3000m
                ])
            else:
                intruder_states.extend([10.0, 0.0, 0.0, 0.0])  # Far away if deleted
        
        intruder_states = np.array(intruder_states)
        
        # Combine all observations
        obs = np.concatenate([
            ego_state,
            target_state,
            conflict_info,
            nfz_info,
            intruder_states
        ])
        
        return obs.astype(np.float64)

    def _get_altitude_layer_index(self, altitude):
        """ Get closest altitude layer index """
        distances = [abs(altitude - layer_alt) for layer_alt in ALTITUDE_LAYERS]
        return np.argmin(distances)

    def _get_action(self, action):
        """ Execute action """
        if 'OWNSHIP' not in bs.traf.id:
            return
        
        ac_idx = bs.traf.id2idx('OWNSHIP')
        
        # Extract actions
        if isinstance(action, dict):
            heading_action = action["heading"][0] if hasattr(action["heading"], '__len__') else action["heading"]
            speed_action = action["speed"][0] if hasattr(action["speed"], '__len__') else action["speed"]
            altitude_action = action["altitude"]
        else:
            heading_action = 0.0
            speed_action = 0.0
            altitude_action = 1

        # Heading change
        current_heading = float(bs.traf.hdg[ac_idx])
        heading_change = float(heading_action) * D_HEADING
        new_heading = current_heading + heading_change
        bs.stack.stack(f"HDG OWNSHIP {new_heading:.1f}")

        # Speed change
        current_speed = float(bs.traf.gs[ac_idx])
        speed_change = float(speed_action) * D_SPEED
        new_speed = np.clip(current_speed + speed_change, 50, 250)  # Limit to reasonable range
        bs.stack.stack(f"SPD OWNSHIP {new_speed:.1f}")

        # Altitude layer change
        if altitude_action == 0:  # Descend
            self.target_altitude_layer_idx = max(0, self.target_altitude_layer_idx - 1)
            self.total_altitude_changes += 1
        elif altitude_action == 2:  # Climb
            self.target_altitude_layer_idx = min(len(ALTITUDE_LAYERS) - 1, self.target_altitude_layer_idx + 1)
            self.total_altitude_changes += 1

        target_alt = ALTITUDE_LAYERS[self.target_altitude_layer_idx]
        bs.stack.stack(f"ALT OWNSHIP {target_alt:.1f}")
        bs.stack.stack(f"VS OWNSHIP {VERTICAL_TRANSITION_RATE:.1f}")

    def _get_reward(self):
        """ Calculate comprehensive reward """
        if 'OWNSHIP' not in bs.traf.id:
            return -100.0, True  # Large penalty for crash
        
        ac_idx = bs.traf.id2idx('OWNSHIP')
        reward = 0.0
        terminated = False

        # 1. Goal reaching reward
        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(
            bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
            self.wpt_lat, self.wpt_lon
        )
        
        if wpt_dis * NM2KM < DISTANCE_MARGIN and not self.wpt_reached:
            reward += REACH_REWARD
            self.wpt_reached = True
            terminated = True
        
        # 2. Trajectory drift penalty
        drift = bs.traf.hdg[ac_idx] - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)
        drift_norm = abs(drift) / 180.0
        reward += DRIFT_PENALTY * drift_norm
        self.trajectory_deviation.append(abs(drift))
        
        # 3. Conflict penalty
        num_conflicts = 0
        for i in range(self.num_intruders):
            int_acid = f'INTRUDER_{i+1}'
            if int_acid in bs.traf.id:
                int_idx = bs.traf.id2idx(int_acid)
                
                int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                    bs.traf.lat[int_idx], bs.traf.lon[int_idx]
                )
                
                alt_diff = abs(bs.traf.alt[int_idx] - bs.traf.alt[ac_idx])
                
                # 3D conflict check
                if int_dis < INTRUSION_DISTANCE and alt_diff < VERTICAL_MARGIN:
                    reward += CONFLICT_PENALTY
                    num_conflicts += 1
                    self.total_conflicts += 1
        
        # 4. NFZ violation penalty
        violations = False
        if self.enable_nfz:
            violations = bs.traf.nfz.check_point(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.alt[ac_idx]
            )
            if violations:
                reward += NFZ_VIOLATION_PENALTY
        
        # 5. Altitude change penalty (encourage efficiency)
        if self.target_altitude_layer_idx != self.current_altitude_layer_idx:
            reward += ALTITUDE_CHANGE_PENALTY
        
        # 6. Smooth control bonus (small reward for stability)
        if num_conflicts == 0 and not violations:
            reward += SMOOTH_CONTROL_BONUS

        self.total_reward += reward
        return reward, terminated

    def _get_info(self):
        """ Return episode information """
        return {
            "episode_step": self.episode_step,
            "total_reward": self.total_reward,
            "total_conflicts": self.total_conflicts,
            "total_nfz_violations": self.total_nfz_violations,
            "total_altitude_changes": self.total_altitude_changes,
            "avg_trajectory_deviation": np.mean(self.trajectory_deviation) if self.trajectory_deviation else 0.0,
            "waypoint_reached": self.wpt_reached,
            "scenario_type": self.scenario_type,
            "disturbance_level": self.disturbance_level
        }

    def _render_frame(self):
        """ Render environment """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Conflict Resolution Environment")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((240, 248, 255))

        if 'OWNSHIP' not in bs.traf.id:
            return

        ac_idx = bs.traf.id2idx('OWNSHIP')
        
        # Map parameters
        map_scale = 200  # km per screen height
        map_width = int(self.window_width * 0.7)
        
        # Draw no-fly zones
        if self.enable_nfz:
            for zone in bs.traf.nfz.zones:
                if zone['type'] == 'circle':
                    center_x, center_y = self._latlon_to_screen(
                        zone['lat'], zone['lon'], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                        map_width, self.window_height, map_scale
                    )
                    radius_km = zone['radius'] * NM2KM
                    radius_px = int(radius_km / map_scale * self.window_height)
                    pygame.draw.circle(canvas, (255, 150, 150), (int(center_x), int(center_y)), radius_px, 2)
                    pygame.draw.circle(canvas, (255, 200, 200, 50), (int(center_x), int(center_y)), radius_px, 0)
        
        # Draw waypoint
        wpt_x, wpt_y = self._latlon_to_screen(
            self.wpt_lat, self.wpt_lon, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
            map_width, self.window_height, map_scale
        )
        pygame.draw.circle(canvas, (0, 200, 0), (int(wpt_x), int(wpt_y)), 12, 3)
        
        # Draw ownship
        own_x, own_y = map_width // 2, self.window_height // 2
        pygame.draw.circle(canvas, (255, 0, 0), (own_x, own_y), 15)
        
        # Draw intruders
        for i in range(self.num_intruders):
            int_acid = f'INTRUDER_{i+1}'
            if int_acid in bs.traf.id:
                int_idx = bs.traf.id2idx(int_acid)
                int_x, int_y = self._latlon_to_screen(
                    bs.traf.lat[int_idx], bs.traf.lon[int_idx],
                    bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                    map_width, self.window_height, map_scale
                )
                pygame.draw.circle(canvas, (0, 100, 255), (int(int_x), int(int_y)), 10)
        
        # Draw info panel
        font = pygame.font.Font(None, 24)
        info_x = map_width + 20
        info_y = 20
        
        info_texts = [
            f"Step: {self.episode_step}/{self.max_episode_steps}",
            f"Reward: {self.total_reward:.2f}",
            f"Conflicts: {self.total_conflicts}",
            f"NFZ Violations: {self.total_nfz_violations}",
            f"Alt Layer: {self.current_altitude_layer_idx} ({ALTITUDE_LAYERS[self.current_altitude_layer_idx]}m)",
            f"Speed: {bs.traf.gs[ac_idx]:.1f} m/s",
            f"Heading: {bs.traf.hdg[ac_idx]:.1f}°",
            "",
            f"Scenario: {self.scenario_type}",
            f"Disturbance: {self.disturbance_level}",
        ]
        
        for text in info_texts:
            label = font.render(text, True, (0, 0, 0))
            canvas.blit(label, (info_x, info_y))
            info_y += 30

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def _latlon_to_screen(self, lat, lon, center_lat, center_lon, width, height, scale):
        """ Convert lat/lon to screen coordinates """
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * np.cos(np.deg2rad(center_lat))
        
        dy_km = (lat - center_lat) * km_per_deg_lat
        dx_km = (lon - center_lon) * km_per_deg_lon
        
        x = width / 2 + dx_km / scale * height
        y = height / 2 - dy_km / scale * height
        
        return x, y

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
