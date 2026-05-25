"""Shared constants for the dense-RL UAV conflict-resolution environments.

Centralises altitude layers, separation thresholds and action magnitudes so that
ConflictResolutionEnv, Discrete25DEnv and MultiAgentEnv all use identical values.
"""

# Unit conversions
NM2KM = 1.852  # 1 nautical mile in km
NM2M = 1852.0
KM2M = 1000.0

# 2.5D discrete altitude layers [m]
ALTITUDE_LAYERS = (500, 1000, 1500, 2000, 2500)
DEFAULT_ALTITUDE_LAYER_IDX = 2  # 1500 m
VERTICAL_TRANSITION_RATE = 5.0  # m/s

# conflict detection thresholds
INTRUSION_DISTANCE_NM = 5.0     # horizontal separation minimum [NM]
VERTICAL_MARGIN_M = 300.0       # vertical separation minimum [m]
DISTANCE_MARGIN_KM = 5.0        # waypoint reach threshold [km]

# action magnitude limits
D_HEADING_DEG = 30.0
D_SPEED_MS = 10.0
AC_DEFAULT_SPEED_MS = 150.0
ACTION_FREQUENCY = 10           # BlueSky sim steps per RL action

# critical state thresholds for dense RL
CRITICAL_TCPA_S = 90.0
CRITICAL_TAU_S = 120.0
CRITICAL_DIST_NM = 10.0
