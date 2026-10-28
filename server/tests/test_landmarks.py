"""buildings: landmarks that are not boxes (landmark_kind): a moored ship is a hull, deck and superstructure, a round
hall a drum and a dome, a small round industrial footprint a tank, an arched roof (roof:shape=round) a barrel vault;
houses, round towers and plain boxes stay buildings."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import buildings as bld  # noqa: E402
import geodesy  # noqa: E402


def rect(x, y, w, d):
    return np.array([(x, y), (x + w, y), (x + w, y + d), (x, y + d)], np.float64)


def hull(L=270.0, B=33.0):
    """A ship's outline along x: a rounded stern, parallel sides, a bow tapering to a point (the Iowa's proportions)."""
    xs = np.linspace(-L / 2, L / 2, 41)
    x0, x1 = -L / 2 + 0.1 * L, L / 2 - 0.25 * L

    def w(x):
        if x < x0:
            return B / 2 * math.sqrt(max(0.0, 1 - ((x0 - x) / (x0 + L / 2)) ** 2))
        if x > x1:
            return B / 2 * (1 - (x - x1) / (L / 2 - x1)) ** 1.3
        return B / 2
    top = [(x, w(x)) for x in xs[::-1] if w(x) > 0.05]
    ring = np.array([(L / 2, 0.0)] + top + [(-L / 2, 0.0)] + [(x, -y) for x, y in top[::-1]], np.float64)
    return ring[::-1] if bld._area2(ring) < 0 else ring


def circle(r, n=40, c=(0.0, 0.0)):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)])


def outward(test, spec, P, T, c):
    n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    good = ((P[T].mean(1) - c) * n).sum(1) >= -1e-6
    test.assertTrue(good.all(), f"{(~good).sum()} of {len(T)} triangles face in ({spec.cls})")


class KindTest(unittest.TestCase):
    def test_ship_by_name_or_water(self):
        r = hull()
        self.assertEqual(bld.landmark_kind(r, {"name": "Battleship USS Iowa", "class": ""}), "ship")
        self.assertEqual(bld.landmark_kind(r, {"name": "", "class": ""}, on_water=True), "ship")
        self.assertIsNone(bld.landmark_kind(r, {"name": "", "class": ""}), "a pointed building on land: a building")
        self.assertIsNone(bld.landmark_kind(r, {"name": "USS Iowa"}, ml=True), "machine-learned: never")

    def test_not_ships(self):
        self.assertIsNone(bld.landmark_kind(rect(0, 0, 200, 30), {"name": "Ship Warehouse"}, on_water=True),
                          "square ends: a shed on a pier")
        walk = hull(90.0, 6.3)
        self.assertIsNone(bld.landmark_kind(walk, {"name": ""}, on_water=True), "a 90 x 6 m walkway: too slender")
        self.assertIsNone(bld.landmark_kind(hull(20.0, 5.0), {"name": "SS Tiny"}), "too short")
        self.assertIsNone(bld.landmark_kind(hull(), {"name": "USS Iowa", "class": "apartments"}))

    def test_round(self):
        self.assertEqual(bld.landmark_kind(circle(26), {"class": "commercial", "name": "Century 21", "height": 16.8}),
                         "dome")
        self.assertIsNone(bld.landmark_kind(circle(26), {"class": "apartments", "name": "Round Tower Flats"}))
        self.assertIsNone(bld.landmark_kind(circle(15), {"class": "office", "name": "Capitol Records",
                                                         "height": 45.0}), "a round tower")
        self.assertEqual(bld.landmark_kind(circle(12), {"class": "", "name": ""}), "tank")
        self.assertEqual(bld.landmark_kind(circle(30), {"class": "storage_tank", "name": "Tank 4"}), "tank")
        self.assertEqual(bld.landmark_kind(rect(0, 0, 30, 30), {"class": "civic", "roof_shape": "dome"}), "dome")
        self.assertIsNone(bld.landmark_kind(rect(0, 0, 30, 20), {"class": "retail", "name": "Mall"}))

    def test_round_with_a_lobby(self):
        """The Century 21 outline: a circle with an entrance block on one side (4 pi A / P^2 under 0.8)."""
        t = np.linspace(0.45, 2 * math.pi - 0.45, 32)
        arc = np.column_stack([26 * np.cos(t), 26 * np.sin(t)])
        ring = bld.clean_ring(np.vstack([arc, [(23.4, -11.4), (42, -14), (42, 14), (23.4, 11.4)]]))
        self.assertLess(bld.roundness(ring), 0.86)
        fit = bld.circle_fit(ring)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit[1], 26.0, delta=1.0)
        self.assertEqual(bld.landmark_kind(ring, {"class": "commercial", "name": "Former Century 21"}), "dome")


class MeshTest(unittest.TestCase):
    def test_ship(self):
        r = hull()
        spec = bld.plan(r, {"name": "Battleship USS Iowa", "height": 17.4, "landmark": "ship"})
        self.assertEqual(spec.cls, "ship")
        P, UV, T, M, mats, C = bld.build([(spec, 0.0, 0.0)])
        self.assertAlmostEqual(P[:, 2].max(), spec.height, delta=0.5)
        self.assertGreaterEqual(spec.height, 0.15 * 270 - 0.5, "masts about 0.15 of its length")
        self.assertNotIn(bld.MATERIALS.index(bld.FACADE), set(M.tolist()), "no window sheet")
        deck = P[np.abs(P[:, 2] - spec.floor_h) < 1e-6]
        self.assertGreater(np.ptp(deck[:, 0]), 260, "the deck the hull's length")
        grey = bld._Looks(spec.seed)[f"{bld.STEEL}|grey"][1]
        self.assertIn(tuple(grey), {tuple(c) for c in C.tolist()}, "a warship: painted grey")
        # its collision: the hull to the deck, not a block to the mast tops
        cm = bld.collision_mesh([(spec, 0.0, 0.0)])
        self.assertAlmostEqual(cm[0][:, 2].max(), spec.floor_h, delta=1e-6)

    def test_liner_paint_and_mapped_funnels(self):
        r = hull(310.0, 36.0)
        funnels = [(circle(4.5, 12, (x, 0.0)), 15.0) for x in (-30.0, 10.0, 50.0)]
        spec = bld.plan(r, {"name": "The Queen Mary", "height": 10.0, "landmark": "ship", "ship_parts": funnels})
        P, UV, T, M, mats, C = bld.build([(spec, 0.0, 0.0)])
        red = tuple(bld._Looks(spec.seed)[f"{bld.STEEL}|red"][1])
        reds = np.array([tuple(C[i]) == red for i in range(len(C))])
        self.assertTrue(reds.any())
        xs = sorted({round(float(x)) for x in P[reds][:, 0]})
        for fx in (-30, 10, 50):
            self.assertTrue(any(abs(x - fx) < 10 for x in xs), f"a funnel at x={fx}")

    def test_dome(self):
        ring = circle(26)
        spec = bld.plan(ring, {"class": "commercial", "height": 16.8, "landmark": "dome"})
        self.assertEqual(spec.cls, "dome")
        P, UV, T, M, mats, C = bld.build([(spec, 0.0, 0.0)])
        self.assertAlmostEqual(P[:, 2].max(), 16.8, delta=0.01)
        top = P[P[:, 2] > 16.0]
        self.assertLess(np.abs(top[:, :2]).max(), 12.0, "a dome, not a flat roof: only its crown that high")
        self.assertNotIn(bld.MATERIALS.index(bld.FACADE), set(M.tolist()))
        dome = P[T].mean(1)[:, 2] > bld.DOME_DRUM_SHARE * 16.8
        outward(self, spec, P, T[dome], np.array([0.0, 0.0, 0.0]))

    def test_tank(self):
        spec = bld.plan(circle(12), {"class": "", "landmark": "tank"})
        P, UV, T, M, mats, C = bld.build([(spec, 0.0, 0.0)])
        self.assertGreater(spec.height, 5.0)
        self.assertAlmostEqual(np.hypot(P[:, 0], P[:, 1]).max(), 12.0, delta=0.5)
        outward(self, spec, P, T, np.array([0.0, 0.0, spec.height / 2]))

    def test_barrel_roof(self):
        ring = rect(0, 0, 60, 30)
        spec = bld.plan(ring, {"class": "", "height": 14.0, "roof_shape": "round"})
        self.assertEqual(spec.roof, "barrel")
        P, UV, T, M, mats, C = bld.build([(spec, 0.0, 0.0)])
        self.assertAlmostEqual(P[:, 2].max(), spec.height + 15.0 * bld.BARREL_RISE, delta=0.3)
        self.assertAlmostEqual(P[:, 2].max(), 14.0, delta=0.5, msg="the vault's crown at the mapped height")
        crown = P[P[:, 2] > P[:, 2].max() - 0.01]
        self.assertGreater(np.ptp(crown[:, 0]), 59.0, "the vault runs along the long side")
        outward(self, spec, P, T, np.array([30.0, 15.0, spec.height / 2]))

    def test_plain_buildings_unchanged(self):
        spec = bld.plan(rect(0, 0, 30, 20), {"class": "retail", "height": 9.0})
        self.assertNotIn(spec.cls, bld.LANDMARK_CLASSES)


class TileTest(unittest.TestCase):
    """tile_buildings marks the landmarks (OSM footprints only: a ship by its name, or on water)."""

    def test_tile_marks_ship_and_dome(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_buildings import Feats
        f = geodesy.EnuFrame(33.74, -118.28)

        def ll(r):
            r = np.vstack([r, r[:1]])
            la, lo, _ = f.enu_to_geodetic(r[:, 0], r[:, 1], 0.0)
            return np.stack([lo, la], 1)
        rings = [hull() + (128.0, 70.0), circle(26, 40, (60.0, 190.0)), rect(150, 170, 30, 20)]
        b = Feats(["S", "D", "R"], [ll(r) for r in rings], **{
            "class": ["", "commercial", "retail"], "name": ["Battleship USS Iowa", "Former Century 21", "Mall"],
            "height": [17.4, 16.8, 6.0], "src_dataset": ["OpenStreetMap"] * 3})
        items = bld.tile_buildings(f, 0, 0, 256, 256, [b], lambda la, lo: np.zeros(np.shape(la)))
        self.assertEqual(sorted(s.cls for s, _, _ in items), ["dome", "retail", "ship"])
        ml = Feats(["S"], [ll(rings[0])], **{"class": [""], "name": ["Battleship USS Iowa"], "height": [17.4],
                                             "src_dataset": ["Microsoft ML Buildings"]})
        items = bld.tile_buildings(f, 0, 0, 256, 256, [ml], lambda la, lo: np.zeros(np.shape(la)))
        self.assertNotIn("ship", [s.cls for s, _, _ in items])


if __name__ == "__main__":
    unittest.main()
