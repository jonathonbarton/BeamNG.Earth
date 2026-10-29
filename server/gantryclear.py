"""Overhead sign structures against the other roads round them (a gantry over I-5 must not cut an I-84 flyover).

A gantry (throughsigns 'through' record: centre, travel, poles, panels, bottom) occupies a box: across its road
from pole to pole, DEPTH_M along it (the truss and panels), from its panels' bottom to the top of its truss. Every
road deck that passes through that box in plan must either be high enough over the structure (its underside above
the truss) or low enough under it (CLEARANCE_M from its surface to the panels' bottom, MUTCD / AASHTO 17 ft 4 in).
Ground roads under the span need the same clearance from their own ground. The columns standing off every
roadway is throughsigns' own test (signs._clear on all pavement).

Deck heights: the sign queue does not solve bridge profiles (signstream: 'conservative deck footprints'), so a
deck without a known height ('top' m, or per-point z with 'z': True) is taken one level above its surroundings
(DECK_LEVEL_M), which no gantry clears: a gantry is never planned through an unknown deck.

Numpy only (tools/check_sign_layout.py loads this file directly to measure any checkout).
"""
import math

import numpy as np

CLEARANCE_M = 5.3          # vertical clearance under an overhead sign (MUTCD 2A.18: 17 ft min over roadway)
DEPTH_M = 1.6              # truss and panels along the road (the structure's half depth either side of centre)
DECK_LEVEL_M = 7.0         # an unknown deck: one level up (5.3 m clearance + its structure)
DECK_DEPTH_M = 1.6         # a deck's structure under its surface
TRUSS_ABOVE_M = .6         # the truss above the panels' top
SAMPLE_M = 1.


def _panel_top(sign):
    return float(sign['bottom']) + max(float(p['height']) for p in sign['panels']) + TRUSS_ABOVE_M


def _frame(sign):
    c = np.asarray(sign['centre'][:2], float)
    t = np.asarray(sign['travel'], float)
    t = t / max(float(np.linalg.norm(t)), 1e-9)
    return c, t, np.array([t[1], -t[0]])


def _densify(line, step=SAMPLE_M):
    line = np.asarray(line, float)
    out = [line[:1]]
    for a, b in zip(line[:-1], line[1:]):
        n = max(1, int(math.ceil(float(np.linalg.norm(b[:2] - a[:2])) / step)))
        f = np.linspace(0., 1., n + 1)[1:, None]
        out.append(a + (b - a) * f)
    return np.vstack(out)


def _box_distance(points, sign, depth=DEPTH_M):
    """Plan distance of each point from the gantry's footprint rectangle (0 inside)."""
    c, t, r = _frame(sign)
    lo, hi = min(sign['poles']), max(sign['poles'])
    d = points[:, :2] - c
    u, v = d @ r, d @ t
    du = np.maximum(np.maximum(lo - u, u - hi), 0.)
    dv = np.maximum(np.abs(v) - depth, 0.)
    return np.hypot(du, dv)


def _deck_heights(chunk, pts, road_z):
    if chunk.get('top') is not None:
        return np.full(len(pts), float(chunk['top']))
    if chunk.get('z') and pts.shape[1] >= 3:
        return pts[:, 2]
    return np.full(len(pts), road_z + DECK_LEVEL_M)


def clashes(sign, chunks=(), pavements=(), ground=None):
    """Reasons a planned overhead structure meets another road (empty: it clears every one).

    chunks: road decks ({'deck': True, 'n': polyline, 'w': width, optional 'top' / 'z'}).
    pavements: signroads rows (fid, spec, line, width) of ground roads; ground(xy) -> z (None: flat at the sign's
    road). Tunnels (spec.tunnel, if any) pass under."""
    out = []
    road_z = float(sign['centre'][2]) if len(sign['centre']) > 2 else 0.
    bottom, top = float(sign['bottom']), _panel_top(sign)
    for chunk in chunks:
        if not chunk.get('deck') or len(chunk.get('n', ())) < 2:
            continue
        pts = _densify(chunk['n'])
        near = _box_distance(pts, sign) <= float(chunk.get('w', 0.)) / 2.
        if not near.any():
            continue
        z = _deck_heights(chunk, pts[near], road_z)
        under_ok = z + CLEARANCE_M <= bottom + 1e-6              # the deck passes under, with clearance
        over_ok = z - DECK_DEPTH_M >= top + .3                   # the deck passes over the truss
        if not np.all(under_ok | over_ok):
            out.append(('deck', float(np.min(np.abs(z - road_z)))))
    if ground is not None:
        for row in pavements:
            fid, sp, line, width = row
            if getattr(sp, 'tunnel', False):
                continue
            pts = _densify(np.asarray(line, float))
            near = _box_distance(pts, sign) <= float(width) / 2.
            if not near.any():
                continue
            z = np.array([ground(p[:2]) for p in pts[near]])
            if np.any(bottom - z < CLEARANCE_M - 1e-6):
                out.append(('ground', float(np.max(z))))
    return out


def sight_blocked(sign, chunks, sight_m=100.):
    """A deck crossing the sign's own road within sight_m before it hides the sign (only for a moved sign)."""
    c, t, r = _frame(sign)
    lo, hi = min(sign['poles']), max(sign['poles'])
    for chunk in chunks:
        if not chunk.get('deck') or len(chunk.get('n', ())) < 2:
            continue
        pts = _densify(chunk['n'])
        d = pts[:, :2] - c
        u, v = d @ r, d @ t
        w = float(chunk.get('w', 0.)) / 2.
        tangent = np.gradient(pts[:, :2], axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1e-9)[:, None]
        crossing = np.abs(tangent @ t) < .7                      # (not the sign's own road on a bridge)
        if np.any(crossing & (u >= lo - w) & (u <= hi + w) & (v < -DEPTH_M) & (v > -sight_m)):
            return True
    return False
