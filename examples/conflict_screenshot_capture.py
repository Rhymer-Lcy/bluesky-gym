"""
Non-interactive conflict screenshot capture.

Runs the Crossing scenario (target + 3 intruders) inside BlueSky, steps the
simulation until a conflict is detected (or a step limit is reached), then
saves a single PNG frame of the pygame visualisation. Unlike the interactive
demos, this script takes no keyboard input and exits automatically, which makes
it convenient for generating a static figure.

Usage:
    python examples/conflict_screenshot_capture.py [output.png]

The output path defaults to ``conflict_screenshot.png`` in the current
directory. A desktop environment is still required because the frame is
rendered through a pygame window surface.
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
from bluesky_gym.envs.scenarios import CrossingScenario
from bluesky.tools.geo import qdrdist


def main(out_path):
    pygame.init()
    window_width, window_height = 1200, 800
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("BlueSky Conflict Scenario Visualiser")
    font = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 18)

    print("Initialising BlueSky...")
    bs.init(mode='sim', detached=True)
    bs.stack.stack("CDMETHOD ON")
    print("BlueSky ready, conflict detection enabled\n")

    view_range = 30  # NM
    center_lat, center_lon = 52.0, 4.0
    target_acid = 'TARGET'

    bs.traf.cre(
        acid=target_acid, actype='A320',
        aclat=center_lat, aclon=center_lon,
        achdg=0, acalt=3000, acspd=150
    )

    scenario = CrossingScenario(num_intruders=3)
    intruders = scenario.generate(target_acid=target_acid)
    print(f"Generated intruders: {intruders}")
    print(f"Total aircraft: {bs.traf.ntraf}")

    history = {acid: {'lat': [bs.traf.lat[bs.traf.id2idx(acid)]],
                       'lon': [bs.traf.lon[bs.traf.id2idx(acid)]]}
               for acid in bs.traf.id}

    def lat_lon_to_screen(lat, lon):
        qdr, dist = qdrdist(center_lat, center_lon, lat, lon)
        qdr_rad = np.radians(qdr)
        x_nm = dist * np.sin(qdr_rad)
        y_nm = dist * np.cos(qdr_rad)
        scale = min(window_width, window_height) / (2 * view_range)
        screen_x = window_width / 2 + x_nm * scale
        screen_y = window_height / 2 - y_nm * scale
        return int(screen_x), int(screen_y)

    def draw_grid():
        center_x, center_y = window_width // 2, window_height // 2
        scale = min(window_width, window_height) / (2 * view_range)
        for radius_nm in [5, 10, 15, 20, 25, 30]:
            radius_px = int(radius_nm * scale)
            pygame.draw.circle(window, (50, 50, 50), (center_x, center_y), radius_px, 1)
            label = font_small.render(f"{radius_nm}NM", True, (80, 80, 80))
            window.blit(label, (center_x + radius_px + 5, center_y - 10))

    def draw_trail(acid, color):
        if acid in history and len(history[acid]['lat']) > 1:
            points = []
            for i in range(len(history[acid]['lat'])):
                lat = history[acid]['lat'][i]
                lon = history[acid]['lon'][i]
                points.append(lat_lon_to_screen(lat, lon))
            if len(points) > 1:
                pygame.draw.lines(window, color, False, points, 1)

    def draw_aircraft(acid, color, size=8):
        idx = bs.traf.id2idx(acid)
        if idx >= 0:
            lat, lon = bs.traf.lat[idx], bs.traf.lon[idx]
            x, y = lat_lon_to_screen(lat, lon)
            pygame.draw.circle(window, color, (x, y), size)
            hdg = bs.traf.hdg[idx]
            line_length = 20
            end_x = x + line_length * np.sin(np.radians(hdg))
            end_y = y - line_length * np.cos(np.radians(hdg))
            pygame.draw.line(window, color, (x, y), (int(end_x), int(end_y)), 2)
            label = font_small.render(acid, True, color)
            window.blit(label, (x + 10, y - 10))

    def draw_info_panel(step_count, conflict_detected, conflict_info):
        title = font.render("Scenario: Crossing", True, (255, 255, 255))
        window.blit(title, (10, 10))
        step_text = font_small.render(f"Steps: {step_count}", True, (255, 255, 255))
        window.blit(step_text, (10, 40))
        ac_text = font_small.render(f"Aircraft: {bs.traf.ntraf}", True, (255, 255, 255))
        window.blit(ac_text, (10, 60))

        if conflict_detected:
            conf_text = font_small.render(
                f"Conflict: detected ({len(bs.traf.cd.confpairs)} pairs)",
                True, (255, 100, 100))
            window.blit(conf_text, (10, 80))
            if 'min_tcpa' in conflict_info:
                tcpa_text = font_small.render(
                    f"Min TCPA: {conflict_info['min_tcpa']:.1f}s", True, (255, 200, 100))
                window.blit(tcpa_text, (10, 100))
                dcpa_text = font_small.render(
                    f"Min DCPA: {conflict_info['min_dcpa']:.2f}NM", True, (255, 200, 100))
                window.blit(dcpa_text, (10, 120))
        else:
            conf_text = font_small.render("Conflict: none", True, (100, 255, 100))
            window.blit(conf_text, (10, 80))

    def render(step_count, conflict_detected, conflict_info):
        window.fill((20, 20, 40))
        draw_grid()
        for acid in bs.traf.id:
            color = (255, 100, 100) if acid == target_acid else (100, 150, 255)
            draw_trail(acid, color)
        for acid in bs.traf.id:
            if acid == target_acid:
                draw_aircraft(acid, (255, 50, 50), size=10)
            else:
                draw_aircraft(acid, (50, 150, 255), size=8)
        draw_info_panel(step_count, conflict_detected, conflict_info)
        pygame.display.flip()

    conflict_detected = False
    conflict_info = {}
    conflict_step = 0
    step_count = 0
    max_steps = 500
    capture_step_after_conflict = 70  # settle steps before the frame is saved

    saved = False
    while step_count < max_steps and not saved:
        bs.sim.step()
        step_count += 1

        for acid in bs.traf.id:
            idx = bs.traf.id2idx(acid)
            if idx >= 0:
                history.setdefault(acid, {'lat': [], 'lon': []})
                history[acid]['lat'].append(bs.traf.lat[idx])
                history[acid]['lon'].append(bs.traf.lon[idx])

        if len(bs.traf.cd.confpairs) > 0 and not conflict_detected:
            conflict_detected = True
            conflict_step = step_count
            if len(bs.traf.cd.tcpa) > 0:
                conflict_info['min_tcpa'] = min(bs.traf.cd.tcpa)
                conflict_info['min_dcpa'] = min(bs.traf.cd.dcpa) / 1852  # convert to NM
            print(f"Conflict detected at step {step_count}")

        render(step_count, conflict_detected, conflict_info)

        # drain the event queue so the window stays responsive
        for _ in pygame.event.get():
            pass

        if conflict_detected and step_count >= conflict_step + capture_step_after_conflict:
            pygame.image.save(window, out_path)
            print(f"Screenshot saved to {out_path}")
            saved = True

    if not saved:
        pygame.image.save(window, out_path)
        print(f"Screenshot saved to {out_path} (no conflict reached within {max_steps} steps)")

    pygame.quit()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "conflict_screenshot.png"
    main(out)
