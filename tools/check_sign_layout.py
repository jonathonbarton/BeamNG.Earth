"""Offline check of the highway guide signs planned on real cached map data (no game, no server).

    python tools/check_sign_layout.py [--server DIR] [--box LAT0,LON0,LAT1,LON1] [--png OUT.png] [--json OUT.json]

Plans every road tile of the box with highwaysigns (exactly as signstream does, decks from OSM bridges,
flat ground), then checks each sign:
  * footprint: no post, and no ground-mounted panel edge, inside a road pavement outline
    (an overhead's panels may span the road; its columns may not);
  * spacing:  no two guide signs on the same roadway and direction closer than exitsigns.MIN_SPACING_M
    (MUTCD 2E.39 / 2E.11: advance guide signs on urban freeways at least 800 ft apart where possible,
    never closer than the driver can read; 150 m is our hard floor);
  * size:     panel width/height follow from the legend (lines, text width, shields, rows), so not all equal.
--server points at another checkout's server/ to measure the before state.
"""
import argparse
import importlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Portland: I-405 at W Burnside to the Fremont bridge, I-5/I-84 by OMSI and the Banfield.
PORTLAND = (45.505, -122.692, 45.535, -122.655)
MIN_SPACING_M = 150.


def load(server):
    sys.path.insert(0, server)
    mods = {n: importlib.import_module(n) for n in
            ('mapdata', 'geodesy', 'roads', 'highwaysigns', 'throughsigns', 'exitsigns', 'signstream')}
    return mods


def decks_for(m, context, frame, tx, ty):
    roads = m['roads']
    x0, y0, x1, y1 = roads.tile_bbox(tx, ty)
    la, lo, _ = frame.enu_to_geodetic(np.array([x0, x1]), np.array([y0, y1]), 0.)
    feats = context.around(np.c_[lo, la])
    decks = []
    for i, spec in enumerate(roads.bridge_specs(feats)):
        if spec is None:
            continue
        for ring in roads.bridge_parts(feats, i):
            if len(ring) >= 2:
                pts = np.column_stack(frame.geodetic_to_enu(ring[:, 1], ring[:, 0], 0.))
                decks.append({'deck': True, 'n': pts, 'w': spec.width})
    return decks


def plan(m, box):
    md = m['mapdata'].MapData()
    source = md.highway_signs
    lat0, lon0, lat1, lon1 = box
    frame = m['geodesy'].EnuFrame((lat0+lat1)/2, (lon0+lon1)/2, 0.)
    e0, n0, _ = frame.geodetic_to_enu(lat0, lon0, 0.)
    e1, n1, _ = frame.geodetic_to_enu(lat1, lon1, 0.)
    T = m['roads'].TILE
    elev = lambda la, lo: np.zeros(np.shape(la))
    captured = []
    hs, ts = m['highwaysigns'], m['throughsigns']
    real_build = ts.build

    def capture(exits, overheads, origin):
        captured.append((list(exits), list(overheads)))
        return None
    ts.build = capture
    context = source.context(None, None)
    try:
        for tx in range(int(math.floor(e0/T)), int(math.floor(e1/T))+1):
            for ty in range(int(math.floor(n0/T)), int(math.floor(n1/T))+1):
                chunks = decks_for(m, context, frame, tx, ty)
                hs.build_tile(md, frame, tx, ty, elev, chunks, False, True, None, None)
    finally:
        ts.build = real_build
    signs = []
    for exits, overheads in captured:
        signs.extend(exits)
        signs.extend(overheads)
    # Pavement of the whole box for the footprint check and the drawing.
    lons = np.linspace(lon0, lon1, 12)
    lats = np.linspace(lat0, lat1, 12)
    pts = np.array([(lo, la) for lo in lons for la in lats])
    pav = context.pavements(frame, pts, False)
    outlines = m['exitsigns']._outlines(pav)
    return frame, signs, pav, outlines


def _inside(p, poly):
    a, b = poly[:-1]-p, poly[1:]-p
    crosses = (a[:, 1] > 0.) != (b[:, 1] > 0.)
    ab = b-a
    hit = a[crosses, 0]-a[crosses, 1]*ab[crosses, 0]/ab[crosses, 1]
    return bool(np.count_nonzero(hit > 0.) % 2)


def _hits(points, outlines):
    for p in points:
        near = np.flatnonzero(np.all(p >= outlines.lo, axis=1) & np.all(p <= outlines.hi, axis=1))
        if any(_inside(p, outlines[i]) for i in near):
            return True
    return False


def footprint(sign):
    """(post points, panel edge points or None for an overhead) in plan."""
    c = np.asarray(sign['centre'][:2], float)
    t = np.asarray(sign['travel'], float)
    right = np.array([t[1], -t[0]])
    if sign['kind'] == 'through':
        return [c+right*p for p in sign['poles']], None
    w = sign['width']
    posts = [c+right*w*o for o in sign.get('post_offsets', (-.32, .32))]
    panel = [c+right*w*f for f in np.linspace(-.5, .5, 9)]
    return posts, panel


def legend_rows(sign):
    if sign['kind'] == 'through':
        return [p.get('rows') or p['label'].get('names') for p in sign['panels']]
    return sign.get('rows') or sign['label'].get('names')


def legend_width(es, label, rows=None, state=''):
    if rows:
        return max(es.guide_width(r['ref'], .52) for r in rows)+max(es.guide_width(r['distance'], .52) for r in rows)
    routes = es.route_symbols(label['routes'], state)
    names = label['names'] or ([] if routes else ['EXIT '+label['ref']])
    return max([es.guide_width(n, .58) for n in names]+[.86*bool(routes)]+[0.])


def misfits(es, signs):
    """Panels whose width does not follow the legend: wider than the text by more than a blank's margin
    (the 'one size' symptom), or narrower than the text."""
    out = []
    for s in signs:
        if s['kind'] == 'through':
            panels = [(p['label'], p.get('rows'), p['width']) for p in s['panels'] if p['exit']]
        elif s['kind'] in ('advance', 'decision') and es.sign_style(s['label']) == 'exit':
            panels = [(s['label'], s.get('rows'), s['width'])]
        else:
            continue
        for label, rows, width in panels:
            need = legend_width(es, label, rows, s.get('state', ''))
            if width < min(need, 6.) or width > max(need+2.2, 3.6):
                out.append((s, need, width))
    return out


def check(signs, outlines):
    # one record per physical sign (tiles may meet at a sign's centre only once, but be safe)
    uniq = {}
    for s in signs:
        key = (str(s['id']), s['kind'], s.get('distance', ''), round(s['centre'][0], 1), round(s['centre'][1], 1))
        uniq[key] = s
    signs = list(uniq.values())
    road_hits, spacing, sizes = [], [], set()
    for s in signs:
        posts, panel = footprint(s)
        if _hits(posts, outlines) or (panel is not None and _hits(panel, outlines)):
            road_hits.append(s)
        if s['kind'] == 'through':
            sizes.update((round(p['width'], 2), round(p['height'], 2)) for p in s['panels'] if p['exit'])
        elif s['kind'] in ('advance', 'decision'):
            sizes.add((round(s['width'], 2), round(s['height'], 2)))
    guide = [s for s in signs if s['kind'] in ('advance', 'decision', 'through')]
    for i, a in enumerate(guide):
        for b in guide[i+1:]:
            ta, tb = np.asarray(a['travel']), np.asarray(b['travel'])
            if ta@tb < .7:
                continue
            d = np.asarray(b['centre'][:2])-np.asarray(a['centre'][:2])
            right = np.array([ta[1], -ta[0]])
            if abs(d@right) < 30. and abs(d@ta) < MIN_SPACING_M:
                spacing.append((a, b, float(np.linalg.norm(d))))
    return signs, road_hits, spacing, sizes


def describe(s):
    lab = s.get('label') or next((p['label'] for p in s.get('panels', ()) if p.get('exit')), {})
    ref = lab.get('ref', '') if isinstance(lab, dict) else ''
    return f"{s['kind']} {ref} {s.get('distance', '')} @({s['centre'][0]:.0f},{s['centre'][1]:.0f})"


def render(path, frame, signs, pav, road_hits, spacing, box_xy, scale=2.):
    from PIL import Image, ImageDraw
    (x0, y0), (x1, y1) = box_xy
    W, H = int((x1-x0)*scale/4), int((y1-y0)*scale/4)
    k = scale/4

    def px(p):
        return (float((p[0]-x0)*k), float((y1-p[1])*k))
    img = Image.new('RGB', (W, H), (250, 250, 247))
    d = ImageDraw.Draw(img)
    for (fid, sp, line, width), poly in zip(pav, pav.outlines or []):
        main = sp.cls in ('motorway', 'trunk')
        d.polygon([px(p) for p in poly], fill=(150, 150, 150) if main else (205, 205, 205))
    bad = {id(s) for s in road_hits}
    close = {id(x) for a, b, _ in spacing for x in (a, b)}
    for s in signs:
        c = np.asarray(s['centre'][:2])
        t = np.asarray(s['travel'])
        right = np.array([t[1], -t[0]])
        colour = (220, 0, 0) if id(s) in bad else (255, 140, 0) if id(s) in close else \
            (0, 90, 200) if s['kind'] == 'through' else (0, 140, 60)
        if s['kind'] == 'through':
            a, b = c+right*s['poles'][0], c+right*s['poles'][1]
            d.line([px(a), px(b)], fill=colour, width=3)
        else:
            w = s['width']
            d.line([px(c-right*w/2), px(c+right*w/2)], fill=colour, width=max(3, int(w*k)))
            d.line([px(c), px(c+t*8)], fill=colour, width=1)
        if s.get('rows') or any(p.get('rows') for p in s.get('panels', ())):
            d.ellipse([px(c-6)[0], px(c+6)[1], px(c+6)[0], px(c-6)[1]], outline=(120, 0, 160), width=2)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default=os.path.join(HERE, '..', 'server'))
    ap.add_argument('--box', default=','.join(map(str, PORTLAND)))
    ap.add_argument('--png')
    ap.add_argument('--json')
    ap.add_argument('--scale', type=float, default=2., help='pixels per 2 m (8: 1 px = 25 cm)')
    args = ap.parse_args()
    m = load(os.path.abspath(args.server))
    box = tuple(map(float, args.box.split(',')))
    frame, signs, pav, outlines = plan(m, box)
    signs, road_hits, spacing, sizes = check(signs, outlines)
    misfit = misfits(m['exitsigns'], signs)
    kinds = {}
    for s in signs:
        kinds[s['kind']] = kinds.get(s['kind'], 0)+1
    combined = sum(1 for s in signs if s.get('rows') or any(p.get('rows') for p in s.get('panels', ())))
    report = dict(signs=len(signs), kinds=kinds, combined=combined, in_road=len(road_hits),
                  too_close_pairs=len(spacing), distinct_sizes=len(sizes), size_misfits=len(misfit),
                  misfit_list=[f'{describe(a)} legend {n:.1f} m panel {w:.1f} m' for a, n, w in misfit],
                  in_road_list=[describe(s) for s in road_hits],
                  too_close_list=[f'{describe(a)} | {describe(b)} {dist:.0f} m' for a, b, dist in spacing])
    print(json.dumps(report, indent=1))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=1)
    if args.png:
        lat0, lon0, lat1, lon1 = box
        e0, n0, _ = frame.geodetic_to_enu(lat0, lon0, 0.)
        e1, n1, _ = frame.geodetic_to_enu(lat1, lon1, 0.)
        render(args.png, frame, signs, pav, road_hits, spacing, ((e0, n0), (e1, n1)), args.scale)
    return 1 if road_hits or spacing or misfit else 0


if __name__ == '__main__':
    sys.exit(main())
