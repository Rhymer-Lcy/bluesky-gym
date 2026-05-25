"""Head-on conflict scenario generator.

Intruders approach from nearly the opposite direction (~180° track angle),
representing the highest-severity conflict type.
"""

import numpy as np
import bluesky as bs


class HeadOnScenario:
    """Generate head-on conflicts with configurable parameters.

    Args:
        num_intruders: Number of intruder aircraft.
        dpsi_range: Conflict angle range [deg], default (150, 210) centred at 180°.
        dcpa_range: Distance at CPA [NM], default (0, 3).
        tlosh_range: Time to CPA [s], default (60, 180).
        speed_range: Intruder speed range [m/s], default (100, 200).
        altitude_range: Relative altitude range [m], default (-500, 500).
    """
    
    def __init__(self, 
                 num_intruders: int = 3,
                 dpsi_range: tuple = (150, 210),
                 dcpa_range: tuple = (0, 3),
                 tlosh_range: tuple = (60, 180),
                 speed_range: tuple = (100, 200),
                 altitude_range: tuple = (-500, 500)):
        
        self.num_intruders = num_intruders
        self.dpsi_range = dpsi_range
        self.dcpa_range = dcpa_range
        self.tlosh_range = tlosh_range
        self.speed_range = speed_range
        self.altitude_range = altitude_range
    
    def generate(self, target_acid: str = 'AC0', actype: str = 'A320'):
        """Generate head-on conflict intruders around *target_acid*.

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
            f"Head-On Scenario\n"
            f"  intruders  : {self.num_intruders}\n"
            f"  angle      : {self.dpsi_range[0]}\u2013{self.dpsi_range[1]} deg (centred 180\u00b0)\n"
            f"  dcpa       : {self.dcpa_range[0]}\u2013{self.dcpa_range[1]} NM\n"
            f"  time to CPA: {self.tlosh_range[0]}\u2013{self.tlosh_range[1]} s\n"
            f"  speed      : {self.speed_range[0] if self.speed_range else 'auto'}\u2013"
            f"{self.speed_range[1] if self.speed_range else 'auto'} m/s\n"
            f"  alt range  : {self.altitude_range[0]}\u2013{self.altitude_range[1]} m\n"
        )


# usage example
if __name__ == "__main__":
    import bluesky as bs

    bs.init(mode='sim', detached=True)

    bs.traf.cre('AC0', actype='A320', aclat=52.0, aclon=4.0,
                achdg=0, acalt=3000, acspd=150)

    scenario = HeadOnScenario(num_intruders=3)
    print(scenario.get_description())

    intruders = scenario.generate(target_acid='AC0')
    print(f"\nCreated intruders: {intruders}")

    for i in range(100):
        bs.sim.step()
        if len(bs.traf.cd.confpairs) > 0:
            print(f"\nConflict detected at step {i}: {bs.traf.cd.confpairs}")
            print(f"TCPA: {bs.traf.cd.tcpa}")
            print(f"DCPA: {bs.traf.cd.dcpa/1852} NM")
            break
