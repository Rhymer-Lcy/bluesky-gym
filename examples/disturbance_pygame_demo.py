"""
Disturbance effect visualisation demo (Pygame).

Shows how different disturbance levels affect flight trajectory in real time.
Press 1-4 to switch disturbance level, Q to quit.
"""

import sys
import os
import pygame
import numpy as np

# add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import bluesky as bs
from bluesky.tools.geo import qdrdist


class DisturbanceVisualizer:
    """Disturbance effect visualiser."""

    def __init__(self):
        # disturbance presets
        self.presets = ['none', 'light', 'medium', 'heavy']
        self.preset_names = {
            'none': 'None',
            'light': 'Light',
            'medium': 'Medium',
            'heavy': 'Heavy'
        }
        self.current_preset_idx = 2  # default: medium

        # initialise pygame
        pygame.init()
        self.window_width = 1400
        self.window_height = 900
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("BlueSky Disturbance Visualiser")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 36)

        # initialise BlueSky
        print("Initialising BlueSky...")
        bs.init(mode='sim', detached=True)
        print("BlueSky ready\n")

        # view parameters
        self.view_range = 20  # NM
        self.center_lat = 52.0
        self.center_lon = 4.0

        # trajectory history
        self.history = {
            'actual': {'lat': [], 'lon': [], 'alt': [], 'hdg': [], 'tas': []},
            'ideal': {'lat': [], 'lon': [], 'alt': [], 'hdg': [], 'tas': []}
        }
        self.step_count = 0
        self.max_history = 500

        # statistics
        self.stats = {
            'pos_error': 0.0,
            'alt_error': 0.0,
            'hdg_error': 0.0,
            'spd_error': 0.0,
            'max_pos_error': 0.0,
            'avg_pos_error': 0.0
        }
        self.pos_errors = []

        self.setup_scenario()
    def setup_scenario(self):
        """Set up test scenario."""
        preset = self.presets[self.current_preset_idx]

        print(f"\n{'='*70}")
        print(f"Switching to: {self.preset_names[preset]}")
        print(f"{'='*70}")

        # reset
        bs.traf.reset()
        self.history['actual'] = {'lat': [], 'lon': [], 'alt': [], 'hdg': [], 'tas': []}
        self.history['ideal'] = {'lat': [], 'lon': [], 'alt': [], 'hdg': [], 'tas': []}
        self.step_count = 0
        self.pos_errors = []

        # create test aircraft
        bs.traf.cre(
            acid='TEST',
            actype='A320',
            aclat=self.center_lat,
            aclon=self.center_lon,
            achdg=45,  # north-east
            acalt=3000,
            acspd=150
        )

        # apply disturbance preset
        success, msg = bs.traf.disturb.set_preset(preset)
        bs.traf.disturb.enabled = (preset != 'none')

        # record initial ideal state
        self.ideal_lat = bs.traf.lat[0]
        self.ideal_lon = bs.traf.lon[0]
        self.ideal_alt = bs.traf.alt[0]
        self.ideal_hdg = bs.traf.hdg[0]
        self.ideal_tas = bs.traf.tas[0]

        print(f"{msg}")
        print(f"Disturbance: {'enabled' if bs.traf.disturb.enabled else 'disabled'}")
        
    def lat_lon_to_screen(self, lat, lon):
        """Convert lat/lon to screen coordinates."""
        qdr, dist = qdrdist(self.center_lat, self.center_lon, lat, lon)

        qdr_rad = np.radians(qdr)
        x_nm = dist * np.sin(qdr_rad)
        y_nm = dist * np.cos(qdr_rad)

        # map area occupies left 70% of window width
        map_width = self.window_width * 0.7
        map_height = self.window_height
        scale = min(map_width, map_height) / (2 * self.view_range)

        screen_x = map_width / 2 + x_nm * scale
        screen_y = map_height / 2 - y_nm * scale

        return int(screen_x), int(screen_y)

    def draw_map(self):
        """Draw map background."""
        map_width = int(self.window_width * 0.7)

        pygame.draw.rect(self.window, (20, 30, 50), (0, 0, map_width, self.window_height))

        center_x, center_y = map_width // 2, self.window_height // 2
        scale = min(map_width, self.window_height) / (2 * self.view_range)

        for radius_nm in [5, 10, 15, 20]:
            radius_px = int(radius_nm * scale)
            if radius_px < max(map_width, self.window_height):
                pygame.draw.circle(
                    self.window,
                    (40, 50, 70),
                    (center_x, center_y),
                    radius_px,
                    1
                )
                label = self.font_small.render(f"{radius_nm}NM", True, (80, 90, 110))
                self.window.blit(label, (center_x + radius_px + 5, center_y - 10))

        pygame.draw.circle(self.window, (100, 100, 120), (center_x, center_y), 5)

    def draw_trails(self):
        """Draw flight trails."""
        if len(self.history['ideal']['lat']) < 2:
            return

        # ideal trail (green dashed)
        ideal_points = []
        for i in range(len(self.history['ideal']['lat'])):
            x, y = self.lat_lon_to_screen(
                self.history['ideal']['lat'][i],
                self.history['ideal']['lon'][i]
            )
            ideal_points.append((x, y))

        if len(ideal_points) > 1:
            for i in range(0, len(ideal_points) - 1, 3):  # dashed effect
                if i + 1 < len(ideal_points):
                    pygame.draw.line(
                        self.window,
                        (100, 200, 100),
                        ideal_points[i],
                        ideal_points[i + 1],
                        2
                    )

        # actual trail (red solid)
        actual_points = []
        for i in range(len(self.history['actual']['lat'])):
            x, y = self.lat_lon_to_screen(
                self.history['actual']['lat'][i],
                self.history['actual']['lon'][i]
            )
            actual_points.append((x, y))

        if len(actual_points) > 1:
            pygame.draw.lines(self.window, (255, 100, 100), False, actual_points, 2)

    def draw_aircraft(self):
        """Draw aircraft symbols."""
        if bs.traf.ntraf == 0:
            return

        # ideal position (green circle)
        ideal_x, ideal_y = self.lat_lon_to_screen(self.ideal_lat, self.ideal_lon)
        pygame.draw.circle(self.window, (100, 255, 100), (ideal_x, ideal_y), 12, 2)

        # actual position (red dot + heading line)
        actual_x, actual_y = self.lat_lon_to_screen(
            bs.traf.lat[0],
            bs.traf.lon[0]
        )
        pygame.draw.circle(self.window, (255, 100, 100), (actual_x, actual_y), 10)

        hdg = bs.traf.hdg[0]
        line_length = 30
        end_x = actual_x + line_length * np.sin(np.radians(hdg))
        end_y = actual_y - line_length * np.cos(np.radians(hdg))
        pygame.draw.line(
            self.window,
            (255, 150, 150),
            (actual_x, actual_y),
            (int(end_x), int(end_y)),
            3
        )

        # deviation line connecting actual to ideal
        pygame.draw.line(
            self.window,
            (255, 255, 100),
            (ideal_x, ideal_y),
            (actual_x, actual_y),
            1
        )

    def draw_info_panel(self):
        """Draw info panel (right 30% of window)."""
        panel_x = int(self.window_width * 0.7)
        panel_width = self.window_width - panel_x

        pygame.draw.rect(
            self.window,
            (30, 30, 40),
            (panel_x, 0, panel_width, self.window_height)
        )

        y_offset = 20
        x_margin = panel_x + 20

        preset = self.presets[self.current_preset_idx]
        title = self.font_large.render(
            self.preset_names[preset],
            True,
            (100, 255, 100) if preset == 'none' else
            (100, 200, 255) if preset == 'light' else
            (255, 200, 100) if preset == 'medium' else
            (255, 100, 100)
        )
        self.window.blit(title, (x_margin, y_offset))
        y_offset += 50

        pygame.draw.line(
            self.window,
            (80, 80, 90),
            (x_margin, y_offset),
            (self.window_width - 20, y_offset),
            2
        )
        y_offset += 20

        info_texts = [
            f"Steps: {self.step_count}",
            f"Time: {bs.sim.simt:.1f} s",
            "",
            "Live deviation:",
            f"  Position: {self.stats['pos_error']:.2f} m",
            f"  Altitude: {self.stats['alt_error']:.2f} m",
            f"  Heading:  {self.stats['hdg_error']:.2f} deg",
            f"  Speed:    {self.stats['spd_error']:.2f} m/s",
            "",
            "Cumulative stats:",
            f"  Avg pos error: {self.stats['avg_pos_error']:.2f} m",
            f"  Max pos error: {self.stats['max_pos_error']:.2f} m",
            "",
            "Disturbance params:",
            f"  Pos noise:  {bs.traf.disturb.pos_noise_std:.1f} m",
            f"  Spd noise:  {bs.traf.disturb.spd_noise_std:.1f} m/s",
            f"  Hdg noise:  {bs.traf.disturb.hdg_noise_std:.1f} deg",
            f"  Alt noise:  {bs.traf.disturb.alt_noise_std:.1f} m",
        ]
        
        for text in info_texts:
            surface = self.font_small.render(text, True, (200, 200, 210))
            self.window.blit(surface, (x_margin, y_offset))
            y_offset += 28
        
        # legend
        y_offset += 30
        pygame.draw.line(
            self.window,
            (80, 80, 90),
            (x_margin, y_offset),
            (self.window_width - 20, y_offset),
            2
        )
        y_offset += 20

        legend_title = self.font.render("Legend:", True, (200, 200, 210))
        self.window.blit(legend_title, (x_margin, y_offset))
        y_offset += 35

        # ideal trail
        pygame.draw.line(
            self.window,
            (100, 200, 100),
            (x_margin, y_offset + 10),
            (x_margin + 40, y_offset + 10),
            2
        )
        text = self.font_small.render("Ideal trail", True, (180, 180, 190))
        self.window.blit(text, (x_margin + 50, y_offset))
        y_offset += 30

        # actual trail
        pygame.draw.line(
            self.window,
            (255, 100, 100),
            (x_margin, y_offset + 10),
            (x_margin + 40, y_offset + 10),
            2
        )
        text = self.font_small.render("Actual trail", True, (180, 180, 190))
        self.window.blit(text, (x_margin + 50, y_offset))
        y_offset += 30

        # deviation line
        pygame.draw.line(
            self.window,
            (255, 255, 100),
            (x_margin, y_offset + 10),
            (x_margin + 40, y_offset + 10),
            1
        )
        text = self.font_small.render("Position error", True, (180, 180, 190))
        self.window.blit(text, (x_margin + 50, y_offset))

        y_offset = self.window_height - 150
        pygame.draw.line(
            self.window,
            (80, 80, 90),
            (x_margin, y_offset),
            (self.window_width - 20, y_offset),
            2
        )
        y_offset += 20

        controls = [
            "Controls:",
            "1 - No disturbance",
            "2 - Light",
            "3 - Medium",
            "4 - Heavy",
            "Q - Quit"
        ]
        
        for text in controls:
            surface = self.font_small.render(text, True, (150, 150, 160))
            self.window.blit(surface, (x_margin, y_offset))
            y_offset += 25
    
    def update(self):
        """Step the simulation forward."""
        if bs.traf.ntraf == 0:
            return

        bs.sim.step()
        self.step_count += 1

        self.history['actual']['lat'].append(bs.traf.lat[0])
        self.history['actual']['lon'].append(bs.traf.lon[0])
        self.history['actual']['alt'].append(bs.traf.alt[0])
        self.history['actual']['hdg'].append(bs.traf.hdg[0])
        self.history['actual']['tas'].append(bs.traf.tas[0])

        ideal_gsnorth = self.ideal_tas * np.cos(np.radians(self.ideal_hdg))
        ideal_gseast = self.ideal_tas * np.sin(np.radians(self.ideal_hdg))
        self.ideal_lat += np.degrees(bs.sim.simdt * ideal_gsnorth / 6371000.0)
        self.ideal_lon += np.degrees(bs.sim.simdt * ideal_gseast /
                                     np.cos(np.radians(self.ideal_lat)) / 6371000.0)

        self.history['ideal']['lat'].append(self.ideal_lat)
        self.history['ideal']['lon'].append(self.ideal_lon)
        self.history['ideal']['alt'].append(self.ideal_alt)
        self.history['ideal']['hdg'].append(self.ideal_hdg)
        self.history['ideal']['tas'].append(self.ideal_tas)

        # trim history to max_history entries
        if len(self.history['actual']['lat']) > self.max_history:
            for key in self.history['actual']:
                self.history['actual'][key].pop(0)
                self.history['ideal'][key].pop(0)

        _, dist = qdrdist(self.ideal_lat, self.ideal_lon,
                         bs.traf.lat[0], bs.traf.lon[0])
        pos_error = dist * 1852  # convert NM to m

        self.stats['pos_error'] = pos_error
        self.stats['alt_error'] = abs(bs.traf.alt[0] - self.ideal_alt)
        self.stats['hdg_error'] = abs((bs.traf.hdg[0] - self.ideal_hdg + 180) % 360 - 180)
        self.stats['spd_error'] = abs(bs.traf.tas[0] - self.ideal_tas)

        self.pos_errors.append(pos_error)
        self.stats['max_pos_error'] = max(self.pos_errors)
        self.stats['avg_pos_error'] = np.mean(self.pos_errors)

    def render(self):
        """Render one frame."""
        self.window.fill((15, 20, 30))
        self.draw_map()
        self.draw_trails()
        self.draw_aircraft()
        self.draw_info_panel()
        pygame.display.flip()
    
    def switch_preset(self, preset_idx):
        """Switch to a different disturbance preset."""
        if 0 <= preset_idx < len(self.presets):
            self.current_preset_idx = preset_idx
            self.setup_scenario()

    def run(self):
        """Main loop."""
        running = True
        paused = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.switch_preset(0)
                    elif event.key == pygame.K_2:
                        self.switch_preset(1)
                    elif event.key == pygame.K_3:
                        self.switch_preset(2)
                    elif event.key == pygame.K_4:
                        self.switch_preset(3)
                    elif event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_p:
                        paused = not paused

            if not paused:
                self.update()

            self.render()
            self.clock.tick(30)

        pygame.quit()
        print("\nVisualisation test complete.")


def main():
    """Entry point."""
    print("\n" + "="*70)
    print("BlueSky Disturbance Visualiser")
    print("="*70)
    print("\nControls:")
    print("  1: No disturbance")
    print("  2: Light")
    print("  3: Medium")
    print("  4: Heavy")
    print("  P: Pause/resume")
    print("  Q: Quit")
    print("\nStarting...")

    try:
        visualizer = DisturbanceVisualizer()
        visualizer.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
