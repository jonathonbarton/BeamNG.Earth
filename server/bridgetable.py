"""Named bridges drawn from a table (bridges the lookup gets wrong and the rules cannot draw): their deck profile
(heights over the water along the crossing), their structure (through trusses, a lift span and its towers) and
their railing. A table entry is placed on the map by its axis (two lon/lat points, the ends of OpenStreetMap's
bridge ways): a deck part lying along the axis is that bridge's, every twin carriageway included.

Interstate Bridge (I-5, Columbia River, Vancouver WA - Portland OR), the first entry (the user's report
2026-10-27 in 1.6.3: decks 40-44 m over the river, "bad, visible waves at the top", jersey walls, no truss):
  sources: Wikipedia (2026-10-09): 72 ft (21.9 m) under the highest fixed span (the hump), 176 ft open lift;
  lift span 278 ft (84 m), the second span from the Washington shore, towers 190 ft over the roadway, 242 ft (74 m)
  overall, 3,538 ft (1,078 m) long, twin Parker through trusses (1917 northbound, 1958 southbound). HAER WA-427
  photos (bridge_refs/vancouver/interstate.md): curved-chord Parker trusses, open steel railing on a low curb
  inside the trusses, green paint. BeamMaps' layout/river_bridges.json (the kitbash map): deck 24 m at the hump,
  lift deck ~14 m (39 ft closed), truss 7-12 m over the deck. NBI's navigation clearances for it (41.5 / 42.7 m)
  are the lift span raised; bridgetypes.nav_clearance took them.

Pure: no I/O."""
import numpy as np

AXIS_M = 45.0           # m: a deck part this close to a table bridge's axis (over most of its length) is that bridge's
AXIS_SHARE = 0.7        # ... over this share of its points
CURVE_M = 100.0         # m: the deck's vertical curves (a moving average this long over the table's grades)
UNDER_M = 2.2           # m from the road surface to the underside (roadmesh.DECK_DEPTH + GIRDER_D)
PANEL_M = 10.5          # m between a truss's panel points
PLANE_GAP_M = 0.8       # m from the deck's edge to a truss's plane (outside the slab's parapet ledge)
TOWER_D = 6.0           # m along the bridge, a lift tower's legs apart
TOWER_BRACE_M = 8.0     # m between a tower's horizontal braces
CHORD = (0.8, 1.0)
POST = (0.5, 0.6)
DIAG = (0.4, 0.5)
LEG = (1.2, 1.2)
BRACE = (0.4, 0.5)
RAIL_POST_M = 2.5       # m between an open railing's posts
RAIL_H = (0.6, 1.05)    # m over the road: its two rails

TABLE = [
    dict(name="Interstate Bridge", axis=((45.621652, -122.672841), (45.612855, -122.678732)), length_m=1078.0,
         paint="green", rail="steel",
         # (s m from the north end along the axis, road surface m over the water); None: the abutment's own height
         profile=[(0.0, None), (121.0, 14.1), (205.0, 14.1), (650.0, 24.1), (1078.0, None)],
         level=[(121.0, 205.0)],                          # held level: the lift span
         # the hump: the crest over the highest fixed span, one vertical curve (a level crest between two curves
         # read as a dip to wave_lint's 100 m fit)
         hump=(610.0, 691.0), clear_m=21.9,              # (650: its middle, a profile point)
         spans=[(40.0, 121.0, "truss")] + [(121.0, 205.0, "lift")] +
               [(205.0 + 81.0 * i, 286.0 + 81.0 * i, "truss") for i in range(8)],
         truss=(7.0, 12.0), lift_truss=10.0, tower_top_m=74.0),
]


def _axis(frame, entry):
    (la0, lo0), (la1, lo1) = entry["axis"]
    e, n, _ = frame.geodetic_to_enu(np.array([la0, la1]), np.array([lo0, lo1]), np.zeros(2))
    a, b = np.array([e[0], n[0]]), np.array([e[1], n[1]])
    d = b - a
    L = float(np.hypot(*d))
    return a, d / max(L, 1e-9), L


def project(frame, entry, xy):
    """(s along the axis from its first point, m off it) of points xy (k, 2)."""
    a, u, _ = _axis(frame, entry)
    q = np.asarray(xy, np.float64)[:, :2] - a
    return q @ u, np.abs(q[:, 0] * u[1] - q[:, 1] * u[0])


def match(frame, lines):
    """{part index: table entry} of a chain's deck lines lying along a table bridge's axis."""
    out = {}
    for entry in TABLE:
        _, _, L = _axis(frame, entry)
        for k, ln in enumerate(lines):
            if len(ln) < 2:
                continue
            s, off = project(frame, entry, ln)
            near = (off < AXIS_M) & (s > -60.0) & (s < L + 60.0)
            if near.mean() >= AXIS_SHARE and np.ptp(s) > 0.3 * L:
                out[k] = entry
    return out


def water_level(ground):
    """The water's height under a table bridge: the lowest ground under its decks."""
    return float(np.min(ground))


def profile(frame, entry, xy, water, z0, z1):
    """Road surface heights (k,) of one deck part xy (k, 2) of a table bridge: the table's heights over the water
    (straight grades between its points, the level stretches held level), joined by CURVE_M vertical curves (a
    moving average on a 1 m grid; each level stretch padded by half of it so it stays exactly level), and the part's
    own ends z0, z1 (its abutments) kept. Pure."""
    xy = np.asarray(xy, np.float64)[:, :2]
    s, _ = project(frame, entry, xy)
    rev = s[-1] < s[0]
    sa, sb = float(min(s[0], s[-1])), float(max(s[0], s[-1]))
    za, zb = (z1, z0) if rev else (z0, z1)
    half = CURVE_M / 2
    pts = [(si, water + h) for si, h in entry["profile"] if h is not None]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    levels = [(a, b, float(np.interp(a, xs, ys))) for a, b in entry["level"]]
    # padded: the curves start and end outside each level stretch
    pts = [p for p in pts if not any(a - half < p[0] < b + half for a, b, _ in levels)]
    pts = sorted(pts + [q for a, b, h in levels for q in ((a - half, h), (b + half, h))])
    pts = [p for p in pts if sa + half < p[0] < sb - half]
    ctrl = [(sa, za)] + pts + [(sb, zb)]
    g = np.arange(sa, sb + 1e-9, 1.0)
    if g[-1] < sb:
        g = np.r_[g, sb]
    r = int(half)
    crest = entry.get("hump")
    cy = [c[1] for c in ctrl]
    ci = None
    if crest is not None:
        hx = 0.5 * (crest[0] + crest[1])
        ci = next((i for i, c in enumerate(ctrl) if abs(c[0] - hx) < 1.0), None)
    for _ in range(6):                          # the crest point raised until the curve's crest is at its height
        t = np.interp(g, [c[0] for c in ctrl], cy)
        ext = np.r_[np.full(r, t[0]), t, np.full(r, t[-1])]
        sm = np.convolve(ext, np.ones(2 * r + 1) / (2 * r + 1), mode="valid")
        if ci is None:
            break
        want = ctrl[ci][1]
        got = float(np.interp(ctrl[ci][0], g, sm))
        if abs(want - got) < 0.01:
            break
        cy[ci] += want - got
    t = np.interp(g, [c[0] for c in ctrl], [c[1] for c in ctrl])
    # the ends back on their abutments, the correction faded out over CURVE_M (not across the whole bridge: that
    # tilted the level stretches by ~0.2 m and left steps at their ends)
    w0 = np.clip(1.0 - (g - sa) / CURVE_M, 0.0, 1.0)
    w1 = np.clip(1.0 - (sb - g) / CURVE_M, 0.0, 1.0)
    w0, w1 = w0 * w0 * (3 - 2 * w0), w1 * w1 * (3 - 2 * w1)
    sm = sm + (t[0] - sm[0]) * w0 + (t[-1] - sm[-1]) * w1
    for a, b, h in levels:                      # (exactly level there: the end correction leaves a few mm)
        m = (g >= a) & (g <= b)
        if m.any() and a > sa + half and b < sb - half:
            sm[m] = h
    z = np.interp(s, g, sm)
    z[0], z[-1] = z0, z1
    return z


def _beam(out, p, q, size, paint):
    out.append((np.asarray(p, np.float64), np.asarray(q, np.float64), float(size[0]), float(size[1]), paint))


def _along(xy, z, s_part, si):
    """Point (x, y), road height and unit direction (along increasing s) of a part at axis station si."""
    order = np.argsort(s_part)
    ss = s_part[order]
    x = float(np.interp(si, ss, xy[order, 0]))
    y = float(np.interp(si, ss, xy[order, 1]))
    zz = float(np.interp(si, ss, z[order]))
    d = 1.0
    x2 = float(np.interp(si + d, ss, xy[order, 0]))
    y2 = float(np.interp(si + d, ss, xy[order, 1]))
    t = np.array([x2 - x, y2 - y])
    t = t / max(np.hypot(*t), 1e-9)
    return np.array([x, y]), zz, t


def structure(frame, entry, parts, water):
    """The steel of a table bridge (beams (p, q, w, h, paint), roadmesh.deck's form) over its deck parts
    [(xy (k, 2), z (k,), width)]: a Parker through truss on each side of each part over each "truss" span (the top
    chord curved from entry truss[0] over the deck at the ends to truss[1] at mid-span, verticals, diagonals falling
    to mid-span, cross beams over the top), a Pratt truss over the "lift" span, and a lift tower at each end of the
    lift span (four legs from the deck to tower_top_m over the water, braced, the machinery across the top). Pure."""
    out = []
    paint = entry.get("paint", "grey")
    lo, hi = entry["truss"]
    for xy, z, width in parts:
        xy, z = np.asarray(xy, np.float64)[:, :2], np.asarray(z, np.float64)
        s, _ = project(frame, entry, xy)
        smin, smax = float(s.min()), float(s.max())
        off = width / 2 + PLANE_GAP_M
        for a, b, kind in entry["spans"]:
            if a < smin - 1 or b > smax + 1:
                continue
            n = max(2, int(round((b - a) / PANEL_M)))
            st = np.linspace(a, b, n + 1)
            hts = (np.full(n + 1, entry["lift_truss"]) if kind == "lift"
                   else lo + (hi - lo) * np.sin(np.pi * (st - a) / (b - a)))
            sides = []
            for sgn in (1.0, -1.0):
                bot, top = [], []
                for i, si in enumerate(st):
                    c, zz, t = _along(xy, z, s, si)
                    nrm = np.array([-t[1], t[0]]) * sgn * off
                    p = np.r_[c + nrm, zz - 0.3]
                    bot.append(p)
                    top.append(p + [0, 0, hts[i] if 0 < i < n else 0.0])
                top[0] = bot[0]
                top[-1] = bot[-1]
                for i in range(n):
                    _beam(out, bot[i], bot[i + 1], CHORD, paint)
                    if 0 < i < n - 1:
                        _beam(out, top[i], top[i + 1], CHORD, paint)
                _beam(out, bot[0], top[1], CHORD, paint)                 # the end posts, inclined
                _beam(out, top[n - 1], bot[n], CHORD, paint)
                for i in range(1, n):
                    _beam(out, bot[i], top[i], POST, paint)
                    if i < n / 2:                                         # diagonals falling toward mid-span
                        _beam(out, top[i], bot[i + 1], DIAG, paint)
                    elif i > n / 2:
                        _beam(out, top[i], bot[i - 1], DIAG, paint)
                sides.append(top)
            for i in range(1, n):                                         # cross beams over the lanes
                _beam(out, sides[0][i], sides[1][i], BRACE, paint)
                if i < n - 1:
                    _beam(out, sides[0][i], sides[1][i + 1], BRACE, paint)
            if kind == "lift":
                for si in (a, b):
                    _tower(out, xy, z, s, si, off, water + entry["tower_top_m"], paint, inward=1.0 if si == a else -1.0)
    return out


def _tower(out, xy, z, s, si, off, top_z, paint, inward):
    """One lift tower over a part at station si: four legs (two each side, TOWER_D apart along the bridge, toward
    the lift span), braced every TOWER_BRACE_M on its faces, with the machinery house across its top."""
    legs = []
    for ds in (0.0, inward * TOWER_D):
        c, zz, t = _along(xy, z, s, si + ds)
        for sgn in (1.0, -1.0):
            nrm = np.array([-t[1], t[0]]) * sgn * (off + 0.6)
            legs.append((np.r_[c + nrm, zz - 0.3], sgn, ds))
    base = min(p[2] for p, _, _ in legs)
    for p, _, _ in legs:
        _beam(out, p, [p[0], p[1], top_z], LEG, paint)
    hs = np.arange(base + TOWER_BRACE_M, top_z - 1.0, TOWER_BRACE_M)
    # the faces: each side's pair (along the bridge), braced from the deck up; across the lanes only high up
    pairs = [(0, 2), (1, 3)]
    clear_z = base + 16.0                       # (across the lanes only over the truss and the traffic)
    for i, h in enumerate(np.r_[base, hs]):
        h2 = hs[i] if i < len(hs) else top_z
        for a, b in pairs:
            pa, pb = legs[a][0], legs[b][0]
            _beam(out, [pa[0], pa[1], h2], [pb[0], pb[1], h2], BRACE, paint)
            _beam(out, [pa[0], pa[1], h], [pb[0], pb[1], h2], BRACE, paint)
            _beam(out, [pb[0], pb[1], h], [pa[0], pa[1], h2], BRACE, paint)
        if h2 > clear_z:
            for a, b in ((0, 1), (2, 3)):
                pa, pb = legs[a][0], legs[b][0]
                _beam(out, [pa[0], pa[1], h2], [pb[0], pb[1], h2], BRACE, paint)
    # the machinery house: one box across the top, TOWER_D deep
    left = (legs[0][0][:2] + legs[2][0][:2]) / 2
    right = (legs[1][0][:2] + legs[3][0][:2]) / 2
    _beam(out, np.r_[left, top_z + 2.5], np.r_[right, top_z + 2.5], (TOWER_D + 1.0, 5.0), paint)


def railing(edge, z, paint, step=RAIL_POST_M):
    """An open steel railing along a deck edge (points (k, 2), road heights z): posts every `step` m and two rails
    (RAIL_H over the road), as beams (p, q, w, h, paint). Pure."""
    out = []
    edge = np.asarray(edge, np.float64)
    z = np.asarray(z, np.float64)
    if len(edge) < 2:
        return out
    a = np.r_[0.0, np.cumsum(np.hypot(*np.diff(edge, axis=0).T))]
    st = np.arange(0.0, a[-1] + 1e-9, step)
    if st[-1] < a[-1] - 0.5:
        st = np.r_[st, a[-1]]
    P = np.c_[np.interp(st, a, edge[:, 0]), np.interp(st, a, edge[:, 1]), np.interp(st, a, z)]
    for p in P:
        _beam(out, p, p + [0, 0, RAIL_H[1] + 0.05], (0.12, 0.12), paint)
    for h in RAIL_H:
        for i in range(len(P) - 1):
            _beam(out, P[i] + [0, 0, h], P[i + 1] + [0, 0, h], (0.1, 0.14), paint)
    return out
