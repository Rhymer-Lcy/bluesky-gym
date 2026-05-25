"""
Scenario generator Pygame visualisation demo.

Displays the evolution of 4 typical conflict scenarios in a pygame window.
Press SPACE to switch scenario, Q to quit.
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
from bluesky_gym.envs.scenarios import (
    HeadOnScenario, 
    CrossingScenario, 
    MergingScenario, 
    OvertakingScenario
)
from bluesky.tools.geo import qdrdist


class ScenarioPygameVisualizer:
    """Pygame-based scenario visualiser."""

    def __init__(self):
        self.scenarios = [
            ('Head-On', HeadOnScenario(num_intruders=3)),
            ('Crossing', CrossingScenario(num_intruders=3)),
            ('Merging', MergingScenario(num_intruders=3)),
            ('Overtaking', OvertakingScenario(num_intruders=3))
        ]
        self.current_scenario_idx = 0
        self.target_acid = 'TARGET'

        pygame.init()
        self.window_width = 1200
        self.window_height = 800
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("BlueSky Conflict Scenario Visualiser")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        print("Initialising BlueSky...")
        bs.init(mode='sim', detached=True)
        bs.stack.stack("CDMETHOD ON")
        print("BlueSky ready, conflict detection enabled\n")

        self.view_range = 30  # NM
        self.center_lat = 52.0
        self.center_lon = 4.0

        self.history = {}
        self.step_count = 0
        self.conflict_detected = False
        self.conflict_info = {}

        self.setup_scenario(0)
    def setup_scenario(self, scenario_idx):
        """Set up the selected scenario."""
        name, scenario = self.scenarios[scenario_idx]

        print(f"\n{'='*70}")
        print(f"Scenario {scenario_idx + 1}/4: {name}")
        print(f"{'='*70}")

        bs.traf.reset()
        self.history = {}
        self.step_count = 0
        self.conflict_detected = False
        self.conflict_info = {}

        bs.traf.cre(
            acid=self.target_acid,
            actype='A320',
            aclat=self.center_lat,
            aclon=self.center_lon,
            achdg=0,
            acalt=3000,
            acspd=150
        )

        intruders = scenario.generate(target_acid=self.target_acid)

        for acid in bs.traf.id:
            self.history[acid] = {
                'lat': [bs.traf.lat[bs.traf.id2idx(acid)]],
                'lon': [bs.traf.lon[bs.traf.id2idx(acid)]]
            }

        print(f"Generated intruders: {intruders}")
        print(f"Total aircraft: {bs.traf.ntraf}")
        
    def lat_lon_to_screen(self, lat, lon):
        """Convert lat/lon to screen pixel coordinates."""
        qdr, dist = qdrdist(self.center_lat, self.center_lon, lat, lon)

        qdr_rad = np.radians(qdr)
        x_nm = dist * np.sin(qdr_rad)
        y_nm = dist * np.cos(qdr_rad)

        scale = min(self.window_width, self.window_height) / (2 * self.view_range)
        screen_x = self.window_width / 2 + x_nm * scale
        screen_y = self.window_height / 2 - y_nm * scale

        return int(screen_x), int(screen_y)

    def draw_aircraft(self, acid, color, size=8):
        """Draw an aircraft symbol."""
        idx = bs.traf.id2idx(acid)
        if idx >= 0:
            lat, lon = bs.traf.lat[idx], bs.traf.lon[idx]
            x, y = self.lat_lon_to_screen(lat, lon)

            pygame.draw.circle(self.window, color, (x, y), size)

            hdg = bs.traf.hdg[idx]
            line_length = 20
            end_x = x + line_length * np.sin(np.radians(hdg))
            end_y = y - line_length * np.cos(np.radians(hdg))
            pygame.draw.line(self.window, color, (x, y), (int(end_x), int(end_y)), 2)

            label = self.font_small.render(acid, True, color)
            self.window.blit(label, (x + 10, y - 10))

            return x, y
        return None, None

    def draw_trail(self, acid, color):
        """Draw flight trail."""
        if acid in self.history and len(self.history[acid]['lat']) > 1:
            points = []
            for i in range(len(self.history[acid]['lat'])):
                lat = self.history[acid]['lat'][i]
                lon = self.history[acid]['lon'][i]
                x, y = self.lat_lon_to_screen(lat, lon)
                points.append((x, y))

            if len(points) > 1:
                pygame.draw.lines(self.window, color, False, points, 1)

    def draw_info_panel(self):
        """Draw HUD info panel."""
        name, _ = self.scenarios[self.current_scenario_idx]
        title = self.font.render(f"Scenario: {name}", True, (255, 255, 255))
        self.window.blit(title, (10, 10))

        step_text = self.font_small.render(f"Steps: {self.step_count}", True, (255, 255, 255))
        self.window.blit(step_text, (10, 40))

        ac_text = self.font_small.render(f"Aircraft: {bs.traf.ntraf}", True, (255, 255, 255))
        self.window.blit(ac_text, (10, 60))

        if self.conflict_detected:
            conf_text = self.font_small.render(
                f"Conflict: detected ({len(bs.traf.cd.confpairs)} pairs)",
                True, (255, 100, 100)
            )
            self.window.blit(conf_text, (10, 80))

            if 'min_tcpa' in self.conflict_info:
                tcpa_text = self.font_small.render(
                    f"Min TCPA: {self.conflict_info['min_tcpa']:.1f}s",
                    True, (255, 200, 100)
                )
                self.window.blit(tcpa_text, (10, 100))

                dcpa_text = self.font_small.render(
                    f"Min DCPA: {self.conflict_info['min_dcpa']:.2f}NM",
                    True, (255, 200, 100)
                )
                self.window.blit(dcpa_text, (10, 120))
        else:
            conf_text = self.font_small.render("Conflict: none", True, (100, 255, 100))
            self.window.blit(conf_text, (10, 80))

        help_text = [
            "SPACE: next scenario",
            "Q: quit",
            "Range: ±30 NM"
        ]
        y_offset = self.window_height - 80
        for text in help_text:
            help_surf = self.font_small.render(text, True, (200, 200, 200))
            self.window.blit(help_surf, (10, y_offset))
            y_offset += 20

    def draw_grid(self):
        """Draw range rings."""
        center_x, center_y = self.window_width // 2, self.window_height // 2
        scale = min(self.window_width, self.window_height) / (2 * self.view_range)

        for radius_nm in [5, 10, 15, 20, 25, 30]:
            radius_px = int(radius_nm * scale)
            pygame.draw.circle(
                self.window,
                (50, 50, 50),
                (center_x, center_y),
                radius_px,
                1
            )
            label = self.font_small.render(f"{radius_nm}NM", True, (80, 80, 80))
            self.window.blit(label, (center_x + radius_px + 5, center_y - 10))

    def update(self):
        """Step simulation."""
        bs.sim.step()
        self.step_count += 1

        for acid in bs.traf.id:
            idx = bs.traf.id2idx(acid)
            if idx >= 0:
                if acid not in self.history:
                    self.history[acid] = {'lat': [], 'lon': []}
                self.history[acid]['lat'].append(bs.traf.lat[idx])
                self.history[acid]['lon'].append(bs.traf.lon[idx])

        if len(bs.traf.cd.confpairs) > 0 and not self.conflict_detected:
            self.conflict_detected = True
            print(f"\nConflict detected at step {self.step_count}!")
            print(f"  Conflict pairs: {len(bs.traf.cd.confpairs)}")

            if len(bs.traf.cd.tcpa) > 0:
                min_tcpa = min(bs.traf.cd.tcpa)
                min_dcpa = min(bs.traf.cd.dcpa) / 1852  # convert to NM
                self.conflict_info['min_tcpa'] = min_tcpa
                self.conflict_info['min_dcpa'] = min_dcpa
                print(f"  Min TCPA: {min_tcpa:.2f}s")
                print(f"  Min DCPA: {min_dcpa:.2f}NM")

    def render(self):
        """Render one frame."""
        self.window.fill((20, 20, 40))
        self.draw_grid()

        for acid in bs.traf.id:
            if acid == self.target_acid:
                self.draw_trail(acid, (255, 100, 100))
            else:
                self.draw_trail(acid, (100, 150, 255))

        for acid in bs.traf.id:
            if acid == self.target_acid:
                self.draw_aircraft(acid, (255, 50, 50), size=10)
            else:
                self.draw_aircraft(acid, (50, 150, 255), size=8)

        self.draw_info_panel()
        pygame.display.flip()

    def next_scenario(self):
        """Switch to the next scenario."""
        self.current_scenario_idx = (self.current_scenario_idx + 1) % len(self.scenarios)
        self.setup_scenario(self.current_scenario_idx)

    def run(self):
        """Main loop."""
        running = True
        paused = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.next_scenario()
                    elif event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_p:
                        paused = not paused

            if not paused and self.step_count < 500:
                self.update()

            self.render()
            self.clock.tick(30)

        pygame.quit()
        print("\nVisualisation complete.")


def main():
    """Entry point."""
    print("\n" + "="*70)
    print("BlueSky Conflict Scenario Visualiser")
    print("="*70)
    print("\nControls:")
    print("  SPACE: next scenario")
    print("  P:     pause/resume")
    print("  Q:     quit")
    print("\nStarting...")

    try:
        visualizer = ScenarioPygameVisualizer()
        visualizer.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
