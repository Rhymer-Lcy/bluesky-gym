"""
Disturbance effect test script.

Test and visualise the impact of different disturbance levels on flight trajectories.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import bluesky as bs
from bluesky.tools.geo import qdrdist


def test_disturbance_effects(preset='medium', duration=300):
    """
    Run a disturbance test.

    Args:
        preset: disturbance preset ('none', 'light', 'medium', 'heavy')
        duration: simulation time in seconds
    """
    print(f"\n{'='*70}")
    print(f"Testing disturbance - preset: {preset.upper()}")
    print(f"{'='*70}\n")

    bs.init(mode='sim', detached=True)

    bs.traf.cre(
        acid='TEST_AC',
        actype='A320',
        aclat=52.0,
        aclon=4.0,
        achdg=45,  # north-east
        acalt=3000,
        acspd=150
    )

    bs.traf.disturb.set_preset(preset)
    bs.traf.disturb.enabled = True

    print(bs.traf.disturb.info())

    history = {
        'time': [],
        'lat': [],
        'lon': [],
        'alt': [],
        'hdg': [],
        'tas': [],
        'ideal_lat': [],
        'ideal_lon': [],
        'ideal_alt': [],
        'ideal_hdg': [],
        'ideal_tas': []
    }

    ideal_lat = bs.traf.lat[0]
    ideal_lon = bs.traf.lon[0]
    ideal_alt = bs.traf.alt[0]
    ideal_hdg = bs.traf.hdg[0]
    ideal_tas = bs.traf.tas[0]

    print(f"\nRunning simulation for {duration} s...")
    for i in range(int(duration / bs.sim.simdt)):
        history['time'].append(bs.sim.simt)
        history['lat'].append(bs.traf.lat[0])
        history['lon'].append(bs.traf.lon[0])
        history['alt'].append(bs.traf.alt[0])
        history['hdg'].append(bs.traf.hdg[0])
        history['tas'].append(bs.traf.tas[0])

        ideal_gsnorth = ideal_tas * np.cos(np.radians(ideal_hdg))
        ideal_gseast = ideal_tas * np.sin(np.radians(ideal_hdg))
        ideal_lat += np.degrees(bs.sim.simdt * ideal_gsnorth / 6371000.0)
        ideal_lon += np.degrees(bs.sim.simdt * ideal_gseast / np.cos(np.radians(ideal_lat)) / 6371000.0)

        history['ideal_lat'].append(ideal_lat)
        history['ideal_lon'].append(ideal_lon)
        history['ideal_alt'].append(ideal_alt)
        history['ideal_hdg'].append(ideal_hdg)
        history['ideal_tas'].append(ideal_tas)

        bs.sim.step()

        if (i + 1) % 50 == 0:
            qdr, dist = qdrdist(ideal_lat, ideal_lon,
                               bs.traf.lat[0], bs.traf.lon[0])
            print(f"  step: {i+1:4d}, t={bs.sim.simt:6.1f}s, "
                  f"pos_err={dist*1852:.1f}m, "
                  f"alt_err={bs.traf.alt[0]-ideal_alt:.1f}m, "
                  f"hdg_err={(bs.traf.hdg[0]-ideal_hdg+180)%360-180:.2f}deg")

    print(f"\nSimulation complete\n")

    for key in history:
        history[key] = np.array(history[key])

    pos_errors = []
    for i in range(len(history['time'])):
        _, dist = qdrdist(history['ideal_lat'][i], history['ideal_lon'][i],
                         history['lat'][i], history['lon'][i])
        pos_errors.append(dist * 1852)  # convert NM to m
    pos_errors = np.array(pos_errors)

    alt_errors = history['alt'] - history['ideal_alt']
    hdg_errors = (history['hdg'] - history['ideal_hdg'] + 180) % 360 - 180
    spd_errors = history['tas'] - history['ideal_tas']

    print(f"Statistics:")
    print(f"  Position: mean={np.mean(pos_errors):.2f}m, "
          f"std={np.std(pos_errors):.2f}m, max={np.max(pos_errors):.2f}m")
    print(f"  Altitude: mean={np.mean(np.abs(alt_errors)):.2f}m, "
          f"std={np.std(alt_errors):.2f}m, max={np.max(np.abs(alt_errors)):.2f}m")
    print(f"  Heading:  mean={np.mean(np.abs(hdg_errors)):.3f}deg, "
          f"std={np.std(hdg_errors):.3f}deg, max={np.max(np.abs(hdg_errors)):.3f}deg")
    print(f"  Speed:    mean={np.mean(np.abs(spd_errors)):.2f}m/s, "
          f"std={np.std(spd_errors):.2f}m/s, max={np.max(np.abs(spd_errors)):.2f}m/s")

    return history, pos_errors, alt_errors, hdg_errors, spd_errors


def visualize_results(results_dict):
    """
    Plot comparison charts for multiple disturbance presets.

    Args:
        results_dict: {preset_name: (history, pos_errors, alt_errors, hdg_errors, spd_errors)}
    """
    print(f"\nGenerating charts...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Disturbance Effect Comparison', fontsize=16, fontweight='bold')

    colors = {'none': 'green', 'light': 'blue', 'medium': 'orange', 'heavy': 'red'}

    for preset_name, (history, pos_errors, alt_errors, hdg_errors, spd_errors) in results_dict.items():
        color = colors.get(preset_name, 'gray')
        label = f'{preset_name.upper()}'

        ax = axes[0, 0]
        if preset_name == 'none':
            ax.plot(history['ideal_lon'], history['ideal_lat'],
                   'k--', linewidth=2, alpha=0.5, label='Ideal')
        ax.plot(history['lon'], history['lat'],
               color=color, linewidth=2, alpha=0.7, label=label)
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        ax.set_title('Flight trajectory (top view)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')

        ax = axes[0, 1]
        ax.plot(history['time'], pos_errors,
               color=color, linewidth=2, alpha=0.7, label=label)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position error [m]')
        ax.set_title('Position error vs time')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[0, 2]
        ax.plot(history['time'], alt_errors,
               color=color, linewidth=2, alpha=0.7, label=label)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Altitude error [m]')
        ax.set_title('Altitude error vs time')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 0]
        ax.plot(history['time'], hdg_errors,
               color=color, linewidth=2, alpha=0.7, label=label)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Heading error [deg]')
        ax.set_title('Heading error vs time')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 1]
        ax.plot(history['time'], spd_errors,
               color=color, linewidth=2, alpha=0.7, label=label)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Speed error [m/s]')
        ax.set_title('Speed error vs time')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 2]
        ax.hist(pos_errors, bins=30, alpha=0.6, color=color, label=label, density=True)
        ax.set_xlabel('Position error [m]')
        ax.set_ylabel('Probability density')
        ax.set_title('Position error distribution')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()

    output_file = 'disturbance_test_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}\n")

    plt.show()


def main():
    """Entry point."""
    print("\n" + "="*70)
    print("Disturbance Effect Test")
    print("="*70)

    presets = ['none', 'light', 'medium', 'heavy']
    duration = 300  # 5 minutes

    results_dict = {}

    for preset in presets:
        result = test_disturbance_effects(preset, duration)
        results_dict[preset] = result
        print("\n" + "-"*70 + "\n")

    visualize_results(results_dict)

    print("="*70)
    print("Disturbance test complete.")
    print("="*70)


if __name__ == "__main__":
    main()
