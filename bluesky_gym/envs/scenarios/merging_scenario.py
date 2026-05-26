"""Merging conflict scenario generator.

Intruders merge from the side at a shallow track angle (~30–60°),
typical of route merge points or approach procedures.
"""

import numpy as np
import bluesky as bs

from .base_scenario import BaseScenario


class MergingScenario(BaseScenario):
    """Generate merging conflicts with configurable parameters.

    Args:
        num_intruders: Number of intruder aircraft.
        dpsi_range: Conflict angle range [deg], default (30, 60).
        dcpa_range: Distance at CPA [NM], default (0, 2).
        tlosh_range: Time to CPA [s], default (60, 180).
        speed_range: Intruder speed range [m/s], default (120, 180).
        altitude_range: Relative altitude range [m], default (-200, 200).
    """
    
    def __init__(self, 
                 num_intruders: int = 3,
                 dpsi_range: tuple = (30, 60),
                 dcpa_range: tuple = (0, 2),
                 tlosh_range: tuple = (60, 180),
                 speed_range: tuple = (120, 180),
                 altitude_range: tuple = (-200, 200)):
        
        self.num_intruders = num_intruders
        self.dpsi_range = dpsi_range
        self.dcpa_range = dcpa_range
        self.tlosh_range = tlosh_range
        self.speed_range = speed_range
        self.altitude_range = altitude_range
    
    def generate(self, target_acid: str = 'AC0', actype: str = 'A320'):
        """Generate merging conflict intruders around *target_acid*.

        Args:
            target_acid: Call sign of the ownship.
            actype: Aircraft type for all intruders.

        Returns:
            List of created intruder call signs.
        """
        target_idx = bs.traf.id2idx(target_acid)
        if target_idx < 0:
            raise ValueError(f"Aircraft {target_acid} not found")
        
        intruder_ids = []
        
        for i in range(self.num_intruders):
            dpsi = np.random.uniform(*self.dpsi_range)
            dcpa = np.random.uniform(*self.dcpa_range)
            tlosh = np.random.uniform(*self.tlosh_range)
            dH = np.random.uniform(*self.altitude_range)

            spd = np.random.uniform(*self.speed_range) if self.speed_range else None

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
            f"Merging Scenario\n"
            f"  intruders  : {self.num_intruders}\n"
            f"  angle      : {self.dpsi_range[0]}\u2013{self.dpsi_range[1]} deg (shallow merge)\n"
            f"  dcpa       : {self.dcpa_range[0]}\u2013{self.dcpa_range[1]} NM\n"
            f"  time to CPA: {self.tlosh_range[0]}\u2013{self.tlosh_range[1]} s\n"
            f"  speed      : {self.speed_range[0] if self.speed_range else 'auto'}\u2013"
            f"{self.speed_range[1] if self.speed_range else 'auto'} m/s\n"
            f"  alt range  : {self.altitude_range[0]}\u2013{self.altitude_range[1]} m\n"
        )
