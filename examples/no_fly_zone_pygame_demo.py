"""No-fly zone visualisation demo (Pygame).

Demonstrates:
- Circular and polygon NFZ rendering on a map
- Real-time violation detection and highlighting when aircraft enter NFZs
- Cumulative violation count and status display

Exit: close window or press ESC.
"""
from __future__ import annotations

import bluesky as bs
import numpy as np
import pygame


# display parameters
WIDTH, HEIGHT = 1200, 900
MAP_CENTER_LAT, MAP_CENTER_LON = 52.15, 4.3
MAP_SCALE_KM = 150.0  # km per screen height
MAX_STEPS = 500


def latlon_to_screen(lat: float, lon: float) -> tuple[float, float]:
    """Equirectangular projection: lat/lon to screen pixels."""
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(MAP_CENTER_LAT))
    dy_km = (lat - MAP_CENTER_LAT) * km_per_deg_lat
    dx_km = (lon - MAP_CENTER_LON) * km_per_deg_lon
    x = WIDTH / 2 + dx_km / MAP_SCALE_KM * HEIGHT
    y = HEIGHT / 2 - dy_km / MAP_SCALE_KM * HEIGHT
    return x, y


def setup_simulation() -> None:
    """Initialise BlueSky and create NFZs and test aircraft."""
    if bs.sim is None:
        bs.init(mode='sim', detached=True)
    bs.traf.reset()
    bs.traf.nfz.clear_zones()

    bs.traf.nfz.create_circular_zone("DANGER_ZONE", 52.2, 4.2, 8, 0, 2500)
    bs.traf.nfz.create_polygon_zone(
        "MILITARY_AREA",
        [52.0, 52.1, 52.15, 52.05],
        [4.4, 4.5, 4.6, 4.5],
        500, 3500,
    )

    bs.traf.cre('KL001', actype='A320', aclat=52.0, aclon=4.0, achdg=45, acalt=1500, acspd=200)
    bs.traf.cre('KL002', actype='B737', aclat=52.3, aclon=4.0, achdg=90, acalt=2000, acspd=220)
    bs.traf.cre('KL003', actype='A320', aclat=52.0, aclon=4.5, achdg=0,  acalt=1000, acspd=180)


def draw_zones(screen: pygame.Surface, font: pygame.font.Font) -> None:
    for zone in bs.traf.nfz.zones:
        if zone['type'] == 'circle':
            cx, cy = latlon_to_screen(zone['lat'], zone['lon'])
            radius_px = int(zone['radius'] * 1.852 / MAP_SCALE_KM * HEIGHT)
            pygame.draw.circle(screen, (255, 100, 100), (int(cx), int(cy)), radius_px, 2)
            label = font.render(zone['name'], True, (200, 0, 0))
            screen.blit(label, (int(cx) - 50, int(cy) - 10))
        elif zone['type'] == 'polygon':
            points = [latlon_to_screen(la, lo) for la, lo in zip(zone['lats'], zone['lons'])]
            points = [(int(x), int(y)) for x, y in points]
            if len(points) >= 3:
                pygame.draw.polygon(screen, (255, 100, 100), points, 2)
                cx = sum(p[0] for p in points) / len(points)
                cy = sum(p[1] for p in points) / len(points)
                label = font.render(zone['name'], True, (200, 0, 0))
                screen.blit(label, (int(cx) - 50, int(cy)))


def draw_aircraft(screen: pygame.Surface, font: pygame.font.Font) -> None:
    for i in range(bs.traf.ntraf):
        x, y = latlon_to_screen(bs.traf.lat[i], bs.traf.lon[i])
        violations = bs.traf.nfz.get_violations(bs.traf.id[i])
        color = (255, 0, 0) if violations else (0, 100, 255)

        pygame.draw.circle(screen, color, (int(x), int(y)), 8)
        hdg_rad = np.deg2rad(bs.traf.hdg[i])
        end_x = x + 15 * np.sin(hdg_rad)
        end_y = y - 15 * np.cos(hdg_rad)
        pygame.draw.line(screen, color, (int(x), int(y)), (int(end_x), int(end_y)), 2)

        label_text = f"{bs.traf.id[i]} ({int(bs.traf.alt[i])}m)"
        if violations:
            label_text += "  VIOLATION"
        label = font.render(label_text, True, color)
        screen.blit(label, (int(x) + 12, int(y) - 8))


def draw_info(screen: pygame.Surface, font: pygame.font.Font, step: int) -> None:
    info_lines = [
        f"Step: {step}",
        f"Aircraft: {bs.traf.ntraf}",
        f"No-Fly Zones: {len(bs.traf.nfz.zones)}",
        "",
        "Violations:",
    ]
    for acid in bs.traf.id:
        count = bs.traf.nfz.get_violation_count(acid)
        if count > 0:
            current = bs.traf.nfz.get_violations(acid) or []
            zone_names = [bs.traf.nfz.zones[v]['name'] for v in current]
            tail = f" (IN: {', '.join(zone_names)})" if zone_names else ""
            info_lines.append(f"  {acid}: {count} violations{tail}")

    y = 10
    for line in info_lines:
        screen.blit(font.render(line, True, (0, 0, 0)), (10, y))
        y += 25


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("No-Fly Zone Demo")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    setup_simulation()

    print("Controls: close window or press ESC to exit.")

    running = True
    step = 0
    while running and step < MAX_STEPS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

        bs.sim.step()
        step += 1

        # update violation status
        for i in range(bs.traf.ntraf):
            bs.traf.nfz.check_aircraft(
                bs.traf.id[i], bs.traf.lat[i], bs.traf.lon[i], bs.traf.alt[i]
            )

        screen.fill((240, 248, 255))
        draw_zones(screen, font)
        draw_aircraft(screen, font)
        draw_info(screen, font, step)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    print("\nFinal violation statistics:")
    for acid in bs.traf.id:
        print(f"  {acid}: {bs.traf.nfz.get_violation_count(acid)} violations")


if __name__ == "__main__":
    main()
