# Demo scripts (examples/)

This directory contains human-readable visualisation and interactive demos.
All scripts run standalone and do not depend on pytest.

> For automated testing use the root-level `tests/` directory (`pytest -v`).

## Script overview

| Script | What it demonstrates | How to run |
|---|---|---|
| `disturbance_visualization.py` | matplotlib comparison of trajectory deviation under each `disturbance preset` | `python examples/disturbance_visualization.py` |
| `disturbance_pygame_demo.py` | Real-time pygame visualisation of disturbance effects on a single aircraft (switchable preset) | `python examples/disturbance_pygame_demo.py` |
| `scenarios_pygame_demo.py` | Cyclic display of the 4 canonical conflict scenarios (head-on / crossing / merging / overtaking) | `python examples/scenarios_pygame_demo.py` |
| `no_fly_zone_pygame_demo.py` | Real-time rendering of circular + polygonal NFZs and per-aircraft violation detection | `python examples/no_fly_zone_pygame_demo.py` |

## Requirements

- The package and `bluesky-simulator` must be installed via `pip install -e .`
- pygame demos require a desktop environment (headless servers: use `pytest -v` in `tests/` instead)

## Controls (pygame demos)

- Close window / `ESC`: quit
- For detailed key bindings see the docstring at the top of each script
