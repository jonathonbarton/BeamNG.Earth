"""Exit, overhead destination and mile-marker signs of a road tile (from cody196's PR #363), from the weekly sign
shards (railtags.SignTags: tools/build_sign_tags.py over the OSM planet) and the tile's generated road geometry."""
import logging
import threading
import time
from concurrent.futures import CancelledError

import numpy as np

import exitsigns
import milemarkers
import railtags
import roads
import signroads
import throughsigns

log = logging.getLogger('earthd.highwaysigns')


class Source:
    """The sign records (SignTags) and the stable road neighbourhoods signs are placed in (signroads.Context)."""

    def __init__(self, md, tags=None):
        self.tags = tags if tags is not None else railtags.SignTags(md.cache_dir, md.fetcher)
        self.md = md
        self.contexts = {}
        self.lock = threading.Lock()
        self.warned = -float('inf')

    def close(self):
        pass

    def version(self):
        """The sign data's build in use, or None (none yet)."""
        return self.tags.version()

    def context(self, urban, widths):
        key = (urban, widths)
        with self.lock:
            if key not in self.contexts:
                self.contexts[key] = signroads.Context(
                    lambda cell: self.md.cell(cell, layers=('roads',), kind='demand').layers.get('roads'),
                    urban, widths)
            return self.contexts[key]

    def objects(self, frame, box):
        """The sign records in box (x0, y0, x1, y1 of frame), in the sign modules' form."""
        x0, y0, x1, y1 = box
        lat, lon, _ = frame.enu_to_geodetic(np.array([x0, x1, x0, x1]), np.array([y0, y0, y1, y1]), 0.)
        return self.tags.query(float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat)))


def tile_objects(source, frame, tx, ty):
    """Guide-sign and mile-reference sources that may place a sign in this tile (an exit's signs stand up to
    exitsigns.APPROACH_M before its junction)."""
    x0, y0, x1, y1 = roads.tile_bbox(tx, ty)
    m = exitsigns.QUERY_MARGIN
    return source.objects(frame, (x0 - m, y0 - m, x1 + m, y1 + m))


def plan_tile(source, frame, tx, ty, elev, chunks=(), left_hand=False, mph=True, urban=None, widths=None):
    """The sign records of a tile: (roadside exit signs, overhead assemblies, mile markers).

    Every exit's advance signs are planned on the whole source neighbourhood first (not only this
    tile), then exitsigns.consolidate spaces and combines them across exits, and only then is the
    tile's share kept: closely spaced urban exits get one combined panel, never a stack.
    """
    box = roads.tile_bbox(tx, ty)
    objects = tile_objects(source, frame, tx, ty)
    if not objects:
        return [], [], []
    context = source.context(urban, widths)
    everywhere = (-float('inf'), -float('inf'), float('inf'), float('inf'))
    roadside_all, overhead_all, miles = [], [], []
    hosts = {}
    for obj in sorted({o['id']: o for o in objects if o.get('exit')}.values(), key=lambda o: str(o['id'])):
        pavements, mains = exitsigns.source_pavements(frame, obj, context, left_hand)
        hosts[obj['id']] = (obj, pavements)
        roadside = exitsigns.place(frame, [obj], (), everywhere, elev, chunks, left_hand, mph,
                                   pavements=pavements, ramp_mains=mains)
        distance = '1/2 MILE' if mph else '500 m'
        paired = throughsigns.place(frame, [obj], pavements, box, elev, chunks, mph, allow_outside=True)
        for s in roadside:
            if paired and s['kind'] == 'advance' and s['distance'] == distance:
                continue
            if s.get('blocked'):
                gantry = throughsigns.exit_gantry(frame, obj, pavements, s, elev, chunks)
                if gantry:
                    overhead_all.append(gantry)
                continue
            roadside_all.append(s)
        overhead_all.extend(paired)
    through_all = []
    for obj in sorted({o['id']: o for o in objects if o.get('through')}.values(), key=lambda o: str(o['id'])):
        pavements = throughsigns.source_pavements(frame, obj, context, left_hand)
        through_all.extend(throughsigns.place(frame, [obj], pavements, box, elev, chunks, mph, allow_outside=True))
    roadside_all, overhead_all = exitsigns.consolidate(roadside_all, overhead_all, mph, through_all)
    roadside_all, overhead_all = arrange(frame, hosts, roadside_all, overhead_all, elev, chunks)

    def inside(s):
        return box[0] <= s['centre'][0] < box[2] and box[1] <= s['centre'][1] < box[3]
    exits = [s for s in roadside_all if inside(s)]
    overheads = [s for s in overhead_all if inside(s)]
    overheads += [s for s in through_all if inside(s)]
    for obj in milemarkers.unique_objects(objects):
        pavements = milemarkers.source_pavements(frame, obj, context, left_hand)
        miles.extend(milemarkers.place(frame, [obj], pavements, box, elev, chunks))
    return exits, overheads, miles


def arrange(frame, hosts, roadside, overheads, elev, chunks=()):
    """Combined panels after exitsigns.consolidate, as signed in practice: a split at one point, or a list too
    wide for one panel, becomes one exit panel per branch side by side on one gantry; a gantry whose list
    panel grew is re-planned so its columns and clearances still hold (throughsigns.refit). Where no gantry
    fits, a roadside list stays a list and a gantry falls back to its own panels."""
    out_roadside, out_overheads = [], []
    for sign in roadside+overheads:
        through = sign['kind'] == 'through'
        rows = sign.get('rows') or next((p.get('rows') for p in sign.get('panels', ()) if p.get('rows')), None)
        host = hosts.get(sign['id'])
        if not rows or host is None:
            (out_overheads if through else out_roadside).append(sign)
            continue
        obj, pavements = host
        base = [p for p in sign.get('base_panels', sign.get('panels', ())) if not p['exit']]
        tries = []
        if sign.get('split') or sign.get('wide'):
            exits = throughsigns.exit_panels(rows, sign.get('state', ''))
            side = 1 if sign.get('side', 1) > 0 else -1
            tries.append(base+exits if side > 0 else exits+base)
        if through:
            tries.append(sorted(sign['panels'], key=lambda p: p['offset']))
        placed = None
        for panels in tries:
            placed = throughsigns.refit(frame, obj, pavements, sign, elev, chunks, panels)
            if placed:
                break
        if placed:
            placed['distance'] = sign.get('distance') or rows[0]['distance']
            if sum(1 for p in placed['panels'] if p['exit']) > 1:
                placed['split_panels'] = True
            out_overheads.append(placed)
        elif through:
            fallback = {k: v for k, v in sign.items() if k not in ('base_panels', 'split', 'wide')}
            fallback['panels'] = sign['base_panels']
            out_overheads.append(fallback)
        else:
            out_roadside.append(sign)
    return out_roadside, out_overheads


def build_tile(md, frame, tx, ty, elev, chunks=(), left_hand=False, mph=True, urban=None, widths=None):
    """Return (exit mesh, mile-marker mesh); optional scenery cannot stop road generation."""
    source = getattr(md, 'highway_signs', None)
    if source is None:
        return None, None
    try:
        exits, overheads, miles = plan_tile(source, frame, tx, ty, elev, chunks, left_hand, mph, urban, widths)
        if not (exits or overheads or miles) and not tile_objects(source, frame, tx, ty):
            return None, None
        box = roads.tile_bbox(tx, ty)
        origin = (box[0], box[1], 0.)
        return throughsigns.build(exits, overheads, origin), milemarkers.build(miles, origin)
    except CancelledError:
        raise
    except Exception as error:
        # Keep provider outages, incomplete optional data or missing glyphs from blanking roads.
        now = time.monotonic()
        if now - source.warned >= 60.:
            source.warned = now
            log.warning('Highway signs unavailable; roads continue: %s: %s', type(error).__name__, error)
        return None, None
