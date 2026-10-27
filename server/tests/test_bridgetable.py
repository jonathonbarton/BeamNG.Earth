"""The Interstate Bridge (I-5, Columbia River) from bridgetable, built offline on its own axis as twin decks over flat
water: the hump's clearance, the lift span level, the lift towers, the trusses, an open railing instead of the
parapet, no waves (wave_lint's limits), and a deck mesh with finite vertices drawn the right way up / out (the cdae
round trip: the drawing side is the reversed cross product of the decoded order)."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bridgetable  # noqa: E402
import cdae  # noqa: E402
import geodesy  # noqa: E402
import roadmesh  # noqa: E402
import roads  # noqa: E402

WATER_H = 2.0                   # m: the river's height (geodetic) under the bridge
# BeamMaps' tools/earth_harvest/wave_lint.py limits (2026-10-25)
KINK2, KINK10, WAVE_M, MAX_GRADE = 0.02, 0.01, 0.05, 0.15


def lint(xy, z):
    """wave_lint.lint_line on a deck line, and its waves in the 20-100 m band (an extremum of the residual against a
    100 m running mean, swinging more than 2 * WAVE_M from the one 20-100 m before it)."""
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy[:, :2], axis=0).T))]
    s = np.arange(0, d[-1], 2.0)
    zz = np.interp(s, d, z)
    g = np.diff(zz) / 2.0
    s10 = np.arange(0, d[-1], 10.0)
    g10 = np.diff(np.interp(s10, d, z)) / 10.0
    k = 50
    pad = np.r_[np.full(k, zz[0]), zz, np.full(k, zz[-1])]
    r = zz - np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), "valid")
    last, band = None, 0
    for i in range(1, len(r) - 1):
        if (r[i] - r[i - 1]) * (r[i + 1] - r[i]) < 0:
            if last is not None and abs(r[i] - last[1]) > 2 * WAVE_M and 20.0 <= s[i] - last[0] <= 100.0:
                band += 1
            last = (s[i], r[i])
    return dict(kink2=int((np.abs(np.diff(g)) > KINK2).sum()), kink10=int((np.abs(np.diff(g10)) > KINK10).sum()),
                grade=int((np.abs(g10) > MAX_GRADE).sum()), band=band)


class InterstateBridge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        e = bridgetable.TABLE[0]
        assert e["name"] == "Interstate Bridge"
        la, lo = zip(*e["axis"])
        cls.entry = e
        cls.frame = geodesy.EnuFrame(float(np.mean(la)), float(np.mean(lo)), 0.0)
        a, u, L = bridgetable._axis(cls.frame, e)
        nrm = np.array([-u[1], u[0]])
        s = np.arange(-10.0, L + 10.0, 4.0)
        cls.lines = [a + np.outer(s, u) + nrm * off for off in (15.0, -15.0)]   # northbound, southbound
        cls.lines[1] = cls.lines[1][::-1]                                         # (the other way along)
        cls.bp = roads.BridgeProfiles(cls.frame, lambda key: None, lambda la_, lo_: np.full(np.shape(la_), WATER_H))
        cls.water = float(np.min(cls.bp._ground(cls.lines[0])))
        start = [np.full(len(ln), cls.water + 12.0) for ln in cls.lines]       # abutments 12 m over the river
        cls.comp = [("i5nb", 0), ("i5sb", 0)]
        cls.zs = cls.bp._table(cls.comp, cls.lines, start)
        cls.beams, _ = cls.bp._structure([(ln, z, 12.0) for ln, z in zip(cls.lines, cls.zs)])

    def test_matched(self):
        self.assertEqual(set(bridgetable.match(self.frame, self.lines)), {0, 1})

    def test_hump_clearance(self):
        for z in self.zs:
            clear = float(z.max()) - self.water - bridgetable.UNDER_M
            self.assertAlmostEqual(clear, self.entry["clear_m"], delta=1.0)

    def test_lift_span_level(self):
        a, b = self.entry["level"][0]
        for ln, z in zip(self.lines, self.zs):
            s, _ = bridgetable.project(self.frame, self.entry, ln)
            m = (s >= a) & (s <= b)
            self.assertLess(float(np.ptp(z[m])), 0.02)
            self.assertAlmostEqual(float(z[m].mean()) - self.water, 14.1, delta=0.1)

    def test_no_waves(self):
        for ln, z in zip(self.lines, self.zs):
            got = lint(ln, z)
            self.assertEqual(got, dict(kink2=0, kink10=0, grade=0, band=0), got)

    def test_truss_and_towers(self):
        self.assertGreater(len(self.beams), 500)
        tops = np.array([max(b[0][2], b[1][2]) for b in self.beams]) - self.water
        # the towers: legs to 74 m over the water (a machinery house over them), four a tower, two towers a deck
        legs = [b for b in self.beams if abs(b[0][0] - b[1][0]) < 1e-6 and abs(b[0][1] - b[1][1]) < 1e-6
                and max(b[0][2], b[1][2]) - self.water > 70.0]
        self.assertEqual(len(legs), 2 * 2 * 4)
        self.assertGreater(float(tops.max()), self.entry["tower_top_m"])
        # the trusses: top chords 7-12 m over the deck; every member green
        chords = tops[(tops > 14.0) & (tops < 40.0)]
        self.assertGreater(len(chords), 200)
        self.assertEqual({b[4] for b in self.beams}, {"green"})
        for b in self.beams:
            self.assertTrue(np.isfinite(np.r_[b[0], b[1], b[2], b[3]]).all())

    def test_railing_and_mesh(self):
        chunks = []
        for (fid, _), ln, z in zip(self.comp, self.lines, self.zs):
            ch = dict(roads.DECK, w=12.0, deck=True, fixed=True, fid=fid, n=np.c_[ln, z].tolist())
            self.assertIn(fid, roads.RAILS)
            ch["rail"] = list(roads.RAILS[fid])
            chunks.append(ch)
        P, T = roadmesh.deck(chunks, (0.0, 0.0, 0.0), (), (), self.beams)[:2]
        P, T = np.asarray(P, np.float64), np.asarray(T, np.int64)
        self.assertTrue(np.isfinite(P).all())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "deck.cdae")
            cdae.write(path, P.astype(np.float32), T)
            with open(path, "rb") as f:
                ma = cdae.mesh_arrays(cdae.decode(f.read())["meshes"][0])
        V = ma["verts"].astype(np.float64)
        Ix = ma["indices"].astype(np.int64).reshape(-1, 3)
        A, B, C = V[Ix[:, 0]], V[Ix[:, 1]], V[Ix[:, 2]]
        nrm = -np.cross(B - A, C - A)
        ln_ = np.linalg.norm(nrm, axis=1)
        self.assertEqual(int((ln_ < 1e-9).sum()), 0)
        nz = nrm[:, 2] / ln_
        cen = (A + B + C) / 3
        road = np.concatenate([np.c_[ln, z] for ln, z in zip(self.lines, self.zs)])
        from scipy.spatial import cKDTree
        dist, near = cKDTree(road[:, :2]).query(cen[:, :2])
        top = (np.abs(nz) > 0.9) & (np.abs(cen[:, 2] - road[near, 2]) < 0.05) & (dist < 5.0)
        self.assertGreater(int(top.sum()), 100)
        self.assertEqual(int((top & (nz < 0)).sum()), 0)            # the road deck draws up
        # every edge shared by two triangles is run both ways (one winding through the mesh)
        ed = np.concatenate([Ix[:, [0, 1]], Ix[:, [1, 2]], Ix[:, [2, 0]]])
        uq, inv, cnt = np.unique(np.sort(ed, 1), axis=0, return_inverse=True, return_counts=True)
        fwd = np.bincount(inv.ravel(), weights=(ed[:, 0] < ed[:, 1]).astype(float), minlength=len(uq))
        self.assertEqual(int(((cnt == 2) & ((fwd == 0) | (fwd == 2))).sum()), 0)
        # the parapet's place: no concrete over the curb at the deck's edges, the railing in steel
        mats = roadmesh.deck(chunks, (0.0, 0.0, 0.0))
        Pm, Tm, _, names, Mi = mats
        Pm, Tm, Mi = np.asarray(Pm), np.asarray(Tm), np.asarray(Mi)
        conc = names.index(roadmesh.DECK_SIDE)
        cm = (Pm[Tm[Mi == conc]].mean(1))
        dd, nn = cKDTree(road[:, :2]).query(cm[:, :2])
        hi = (dd > 5.5) & (dd < 7.0) & (cm[:, 2] - road[nn, 2] > roadmesh.CURB_H + 0.05)
        self.assertEqual(int(hi.sum()), 0)
        self.assertGreater(len(names), 1)                            # the railing's steel


if __name__ == "__main__":
    unittest.main()
