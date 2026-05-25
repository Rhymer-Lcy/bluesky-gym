"""Overtaking conflict scenario generator.

Faster intruders approach from behind at a very shallow angle (~0–20°),
typical of same-route overtaking situations.
"""

import numpy as np
import bluesky as bs


class OvertakingScenario:
    """Generate overtaking conflicts with configurable parameters.

    Args:
        num_intruders: Number of intruder aircraft.
        dpsi_range: Conflict angle range [deg], default (0, 20).
        dcpa_range: Distance at CPA [NM], default (0, 2).
        tlosh_range: Time to CPA [s], default (100, 300).
        speed_delta_range: Extra speed above ownship speed [m/s], default (10, 30).
        altitude_range: Relative altitude range [m], default (-300, 300).
    """
    
    def __init__(self, 
                 num_intruders: int = 3,
                 dpsi_range: tuple = (0, 20),
                 dcpa_range: tuple = (0, 2),
                 tlosh_range: tuple = (100, 300),
                 speed_delta_range: tuple = (10, 30),
                 altitude_range: tuple = (-300, 300)):
        
        self.num_intruders = num_intruders
        self.dpsi_range = dpsi_range
        self.dcpa_range = dcpa_range
        self.tlosh_range = tlosh_range
        self.speed_delta_range = speed_delta_range
        self.altitude_range = altitude_range
    
    def generate(self, target_acid: str = 'AC0', actype: str = 'A320'):
        """Generate overtaking conflict intruders around *target_acid*.

        Args:
            target_acid: Call sign of the ownship.
            actype: Aircraft type for all intruders.

        Returns:
            List of created intruder call signs.
        """
        target_idx = bs.traf.id2idx(target_acid)
        if target_idx < 0:
            raise ValueError(f"Aircraft {target_acid} not found")

        target_speed = bs.traf.tas[target_idx]
        
        intruder_ids = []
        
        for i in range(self.num_intruders):
            dpsi = np.random.uniform(*self.dpsi_range)
            dcpa = np.random.uniform(*self.dcpa_range)
            tlosh = np.random.uniform(*self.tlosh_range)
            dH = np.random.uniform(*self.altitude_range)

            # intruder speed = ownship speed + delta (faster)
            speed_delta = np.random.uniform(*self.speed_delta_range)
            spd = target_speed + speed_delta

            intruder_id = f'INTRUDER_{i+1}'

            # create conflicting aircraft via BlueSky creconfs
            bs.traf.creconfs(
                acid=intruder_id,
                actype=actype,
                targetidx=target_idx,
                dpsi=dpsi,
                dcpa=dcpa,
                tlosh=tlosh,
                dH=dH,
                spd=spd
            )
            
            intruder_ids.append(intruder_id)
        
        return intruder_ids
    
    def get_description(self) -> str:
        """Return a plain-text summary of this scenario configuration."""
        return (
            f"Overtaking Scenario\n"
            f"  intruders    : {self.num_intruders}\n"
            f"  angle        : {self.dpsi_range[0]}\u2013{self.dpsi_range[1]} deg (near-same-track)\n"
            f"  dcpa         : {self.dcpa_range[0]}\u2013{self.dcpa_range[1]} NM\n"
            f"  time to CPA  : {self.tlosh_range[0]}\u2013{self.tlosh_range[1]} s\n"
            f"  speed excess : +{self.speed_delta_range[0]}\u2013+{self.speed_delta_range[1]} m/s\n"
            f"  alt range    : {self.altitude_range[0]}\u2013{self.altitude_range[1]} m\n"
        )
