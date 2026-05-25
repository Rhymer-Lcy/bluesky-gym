""" Scenario generators for BlueSky-Gym conflict resolution environments. """

from .head_on_scenario import HeadOnScenario
from .crossing_scenario import CrossingScenario
from .merging_scenario import MergingScenario
from .overtaking_scenario import OvertakingScenario

__all__ = [
    'HeadOnScenario',
    'CrossingScenario', 
    'MergingScenario',
    'OvertakingScenario',
    'get_scenario'
]


def get_scenario(scenario_type: str):
    """
    Factory function to get scenario class by type.
    
    Arguments:
    - scenario_type: Type of scenario ('head_on', 'crossing', 'merging', 'overtaking')
    
    Returns:
    - Scenario class (not instantiated)
    """
    scenarios = {
        'head_on': HeadOnScenario,
        'crossing': CrossingScenario,
        'merging': MergingScenario,
        'overtaking': OvertakingScenario
    }
    
    scenario_type = scenario_type.lower()
    if scenario_type not in scenarios:
        raise ValueError(
            f"Unknown scenario type: {scenario_type}. "
            f"Available types: {list(scenarios.keys())}"
        )
    
    return scenarios[scenario_type]
