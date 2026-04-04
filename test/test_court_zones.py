# test/test_court_zones.py
# -*- coding: utf-8 -*-
"""
Court zones unit tests
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.court_zones import CourtZones, _point_in_polygon, _lerp


# Real court_config from the project
REAL_COURT_CONFIG = {
    "court_boundary_polygon": [[560, 357], [214, 896], [1658, 916], [1496, 370]],
    "net_y": 274,
}

# Simple rectangular court for easier testing
SIMPLE_COURT_CONFIG = {
    "court_boundary_polygon": [[200, 100], [200, 700], [800, 700], [800, 100]],
    "net_y": 400,
}


class TestHelpers(unittest.TestCase):

    def test_lerp(self):
        self.assertEqual(_lerp((0, 0), (10, 10), 0.5), (5.0, 5.0))
        self.assertEqual(_lerp((0, 0), (10, 10), 0.0), (0, 0))
        self.assertEqual(_lerp((0, 0), (10, 10), 1.0), (10.0, 10.0))

    def test_point_in_polygon_square(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertTrue(_point_in_polygon((5, 5), square))
        self.assertFalse(_point_in_polygon((15, 5), square))
        self.assertFalse(_point_in_polygon((-1, 5), square))


class TestCourtZonesInit(unittest.TestCase):

    def test_init_real_config(self):
        zones = CourtZones(REAL_COURT_CONFIG)
        self.assertEqual(zones.net_y, 274)
        self.assertIsNotNone(zones.far_left)
        self.assertIsNotNone(zones.near_right)

    def test_init_simple_config(self):
        zones = CourtZones(SIMPLE_COURT_CONFIG)
        self.assertEqual(zones.net_y, 400)

    def test_init_insufficient_points(self):
        with self.assertRaises(ValueError):
            CourtZones({"court_boundary_polygon": [[0, 0], [1, 1]], "net_y": 100})


class TestServeZones(unittest.TestCase):

    def test_simple_court_serve_zones(self):
        zones = CourtZones(SIMPLE_COURT_CONFIG)

        # near side serve zone (below bottom line, Y > 700)
        # Left third
        zone = zones.get_serve_zone((300, 720), 'near')
        self.assertEqual(zone, 1, "Should be left serve zone")

        # Middle third
        zone = zones.get_serve_zone((500, 720), 'near')
        self.assertEqual(zone, 2, "Should be middle serve zone")

        # Right third
        zone = zones.get_serve_zone((700, 720), 'near')
        self.assertEqual(zone, 3, "Should be right serve zone")

    def test_fallback_serves(self):
        """Test fallback when position is not strictly in serve zone polygon"""
        zones = CourtZones(SIMPLE_COURT_CONFIG)
        # Position far from serve area should still get a zone from fallback
        zone = zones.get_serve_zone((250, 800), 'near')
        self.assertIn(zone, [1, 2, 3])

    def test_serve_zone_polygons_exist(self):
        zones = CourtZones(REAL_COURT_CONFIG)
        for side in ('far', 'near'):
            for z in range(1, 4):
                poly = zones.get_zone_polygon(side, f'serve_{z}')
                self.assertIsNotNone(poly, f"Serve zone {side} {z} should exist")
                self.assertEqual(len(poly), 4)


class TestReceptionZones(unittest.TestCase):

    def test_simple_court_reception_zones(self):
        zones = CourtZones(SIMPLE_COURT_CONFIG)

        # Near side, front row (close to net, Y ~400-550), left
        zone = zones.get_reception_zone((250, 450), 'near')
        self.assertIn(zone, [1, 2, 3], "Should be front row")

        # Near side, back row (far from net, Y ~550-700), left
        zone = zones.get_reception_zone((250, 650), 'near')
        self.assertIn(zone, [4, 5, 6], "Should be back row")

    def test_all_six_zones_reachable(self):
        """Each zone 1-6 should be reachable on each side"""
        zones = CourtZones(SIMPLE_COURT_CONFIG)

        for side in ('near', 'far'):
            zone_polygons = zones.get_all_zone_polygons(side)
            reception_zones = {k: v for k, v in zone_polygons.items() if isinstance(k, int)}
            self.assertEqual(len(reception_zones), 6,
                             f"{side} side should have 6 reception zones")

    def test_reception_zone_polygons_exist(self):
        zones = CourtZones(REAL_COURT_CONFIG)
        for side in ('far', 'near'):
            for z in range(1, 7):
                poly = zones.get_zone_polygon(side, z)
                self.assertIsNotNone(poly, f"Zone {side} {z} should exist")
                self.assertEqual(len(poly), 4)


class TestSideDetection(unittest.TestCase):

    def test_get_side(self):
        zones = CourtZones(REAL_COURT_CONFIG)
        self.assertEqual(zones.get_side_for_position((500, 200)), 'far')
        self.assertEqual(zones.get_side_for_position((500, 600)), 'near')

    def test_get_side_at_net(self):
        zones = CourtZones(REAL_COURT_CONFIG)
        # At net_y exactly, should be near (>=)
        self.assertEqual(zones.get_side_for_position((500, 274)), 'near')


class TestRealCourtConfig(unittest.TestCase):

    def test_real_config_zone_count(self):
        zones = CourtZones(REAL_COURT_CONFIG)
        all_far = zones.get_all_zone_polygons('far')
        all_near = zones.get_all_zone_polygons('near')

        # 6 reception + 3 serve = 9 zones per side
        self.assertEqual(len(all_far), 9)
        self.assertEqual(len(all_near), 9)

    def test_center_of_court_near_side(self):
        """Center of near side court should be in some zone"""
        zones = CourtZones(REAL_COURT_CONFIG)
        # Center of near side: roughly midpoint between net_y and bottom
        center_y = (274 + 906) / 2  # ~590
        center_x = (214 + 1658) / 2  # ~936
        zone = zones.get_reception_zone((center_x, center_y), 'near')
        self.assertIn(zone, range(1, 7))


if __name__ == '__main__':
    unittest.main()
