"""Closely spaced (urban) exits: advance signs are spaced and combined, never stacked or put in the lanes;
panel sizes follow the legend. Rural exits far apart keep their 1 mile / 1/2 mile / 1/4 mile signs."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import exitsigns
import roads
import throughsigns
from test_exit_signs import FRAME, fixture, ll

flat = lambda lat, lon: np.zeros_like(np.asarray(lat, dtype=float))


def advance(ref, distance_m, along, side=1, names=('Downtown',), kind='advance'):
    """A roadside advance sign of exit ref, standing at x=along on an eastbound freeway."""
    label = dict(ref=ref, names=list(names), routes=[])
    width, height = exitsigns.dimensions(label, kind, 'OR')
    return dict(id=hash(ref) % 1000, kind=kind, label=label, style='exit', state='OR',
                centre=[float(along), -12., 0.], travel=[1., 0.], side=side, bottom=2.4,
                width=width, height=height, bases=[0., 0.], post_offsets=[-.32, 0., .32],
                distance=exitsigns.rounded_distance(distance_m) if kind == 'advance' else '',
                advance_m=float(distance_m))


def test_rounded_distances():
    assert exitsigns.rounded_distance(402.) == '1/4 MILE'
    assert exitsigns.rounded_distance(760.) == '1/2 MILE'
    assert exitsigns.rounded_distance(1609.) == '1 MILE'
    assert exitsigns.rounded_distance(2000.) == '1 1/4 MILES'
    assert exitsigns.rounded_distance(90.) == '1/4 MILE'
    assert exitsigns.rounded_distance(480., mph=False) == '500 m'


def test_stacked_advance_signs_of_close_exits_become_one_panel():
    # Exit 300A's 1/2 mile and 300B's 1/4 mile stations fall a few metres apart (I-5 by OMSI).
    signs = [advance('300A', 804.7, 0.), advance('300B', 402.3, 4.), advance('301', 1609.3, 60.)]
    roadside, overheads = exitsigns.consolidate(signs, [])
    assert overheads == []
    assert len(roadside) == 1
    rows = roadside[0]['rows']
    assert [r['ref'] for r in rows] == ['EXIT 300B', 'EXIT 300A', 'EXIT 301']
    assert [r['distance'] for r in rows] == ['1/4 MILE', '1/2 MILE', '1 MILE']
    w, h = exitsigns.list_dimensions(rows)
    assert (roadside[0]['width'], roadside[0]['height']) == (w, h)
    mesh = exitsigns.build(roadside, (0., 0., 0.))
    assert mesh is not None


def test_never_two_guide_signs_within_minimum_spacing():
    signs = [advance('1', 402.3, 0.), advance('1A', 804.7, 120.), advance('2', 1609.3, 600.),
             advance('2', 804.7, 1400.), advance('1', 804.7, 400.)]
    roadside, _ = exitsigns.consolidate(signs, [])
    xs = sorted(s['centre'][0] for s in roadside)
    assert all(b-a >= exitsigns.MIN_SPACING_M for a, b in zip(xs, xs[1:]))


def test_advance_sign_next_to_another_exits_decision_sign_is_dropped():
    decision = advance('2A', 35., 100., kind='decision')
    signs = [decision, advance('1', 402.3, 40.), advance('3', 1609.3, 2000.)]
    roadside, _ = exitsigns.consolidate(signs, [])
    assert sorted(s['label']['ref'] for s in roadside) == ['2A', '3']


def test_far_apart_rural_exits_are_unchanged():
    signs = []
    for k, ref in enumerate(('10', '12', '15')):
        base = k*5000.
        signs += [advance(ref, d, base-d) for d in (1609.3, 804.7, 402.3)]
        signs.append(advance(ref, 35., base-35., kind='decision'))
    roadside, overheads = exitsigns.consolidate(signs, [])
    assert overheads == []
    assert len(roadside) == len(signs)
    assert not any(s.get('rows') for s in roadside)
    key = lambda s: (s['label']['ref'], s['kind'], s['advance_m'])
    assert sorted(map(key, roadside)) == sorted(map(key, signs))


def test_opposite_direction_and_parallel_roadways_are_not_combined():
    a, b = advance('1', 402.3, 0.), advance('5', 804.7, 10.)
    b['travel'] = [-1., 0.]
    c = advance('7', 804.7, 20.)
    c['centre'][1] = 60.                    # a parallel freeway 72 m away
    roadside, _ = exitsigns.consolidate([a, b, c], [])
    assert len(roadside) == 3 and not any(s.get('rows') for s in roadside)


def test_combined_rows_on_an_exit_gantry():
    gantry = dict(id=1, kind='through', centre=[0., 0., 0.], travel=[1., 0.], side=1, state='OR',
                  distance='1/2 MILE', advance_m=804.7, bottom=6., bases=[0., 0.], poles=[-10., 10.],
                  panels=[dict(label=dict(names=['Portland'], routes=[]), width=6., height=3.7, offset=-3., exit=False),
                          dict(label=dict(ref='300A', names=['OMSI'], routes=[]), width=4., height=3.7, offset=3.,
                               exit=True)])
    roadside, overheads = exitsigns.consolidate([advance('300B', 402.3, 5.)], [gantry])
    assert roadside == []
    exit_panel = [p for p in overheads[0]['panels'] if p['exit']][0]
    assert [r['ref'] for r in exit_panel['rows']] == ['EXIT 300B', 'EXIT 300A']
    assert throughsigns.build([], overheads, (0., 0., 0.)) is not None


def test_panel_size_follows_the_legend():
    short = exitsigns.dimensions(dict(ref='1', names=['Gary'], routes=[]), 'advance', 'OR')
    long = exitsigns.dimensions(dict(ref='1', names=['Oregon City'], routes=[]), 'advance', 'OR')
    two = exitsigns.dimensions(dict(ref='1', names=['Oregon City', 'OMSI'], routes=[]), 'advance', 'OR')
    assert short[0] < long[0]
    assert two[1] > long[1]
    assert short[1] < short[0]*1.22         # (no fixed portrait blank)
    land_short = exitsigns.dimensions(dict(ref='1', names=['Gary'], routes=[]), 'advance', 'MI')
    land_long = exitsigns.dimensions(dict(ref='1', names=['Kalamazoo Downtown'], routes=[]), 'advance', 'MI')
    assert land_short[0] < land_long[0]


def test_advance_sign_never_stands_in_a_neighbouring_roadway():
    rows, obj = fixture()
    # A frontage road right where the ground panels would stand (the right side of the eastbound freeway).
    frontage = roads.Spec(10.0, "paved", False, "primary", True, 2.5, 1, 50.0)
    rows = rows+[("frontage", frontage, np.array([[-1900., -14., 0.], [0., -14., 0.]]), -85.0)]
    signs = exitsigns.place(FRAME, [obj], rows, (-1800., -300., 256., 100.), flat, link_ids={"ramp"})
    adv = [s for s in signs if s['kind'] == 'advance']
    assert adv and all(s.get('blocked') for s in adv)
    import signroads
    pav = []
    for fid, sp, line, _ in rows:
        sp = signroads.spec(sp, link=fid == 'ramp')
        p, width = signroads.pavement(sp, line, FRAME, False)
        pav.append(signroads.polygon(p, width))
    for s in signs:
        if s['kind'] == 'advance':
            continue
        assert exitsigns._clear(np.asarray(s['centre'][:2]), s['width']/2, pav)


def _split_fixture():
    import signroads
    rows, obj = fixture()
    ribbons = [(fid, signroads.spec(sp, fid == 'ramp'), line, sp.width) for fid, sp, line, _ in rows]
    south = advance('302A', 402.3, -402.3, names=['Salem'])
    north = advance('302B', 402.3, -399., names=['Seattle'])
    south['id'] = obj['id']
    south['label']['routes'], north['label']['routes'] = ['I 5 SOUTH'], ['I 5 NORTH']
    return obj, ribbons, [south, north]


def test_split_at_one_point_is_one_panel_per_branch_on_one_gantry():
    # I-84 west at the I-5 split: Salem and Seattle at the same gore. Two panels side by side on one gantry,
    # each with its shield, direction, control city and own distance; never a two-row list.
    import highwaysigns
    obj, ribbons, signs = _split_fixture()
    roadside, overheads = exitsigns.consolidate(signs, [])
    assert len(roadside) == 1 and roadside[0].get('split')
    roadside, overheads = highwaysigns.arrange(FRAME, {obj['id']: (obj, ribbons)}, roadside, overheads, flat)
    assert roadside == []
    gantry, = overheads
    exits = [p for p in gantry['panels'] if p['exit']]
    assert [p['label']['names'] for p in exits] == [['Salem'], ['Seattle']]
    assert [p['distance'] for p in exits] == ['1/4 MILE', '1/4 MILE']
    assert not any(p.get('rows') for p in gantry['panels'])
    a, b = exits
    assert b['offset']-a['offset'] >= (a['width']+b['width'])/2      # side by side, not overlapping
    assert throughsigns.build([], overheads, (-1024., -256., 0.)) is not None


def test_split_keeps_its_list_where_no_gantry_fits():
    import highwaysigns
    obj, ribbons, signs = _split_fixture()
    roadside, overheads = exitsigns.consolidate(signs, [])
    deck = dict(deck=True, n=[[-2000., 0., 0.], [500., 0., 0.]], w=14.)     # the whole road on a viaduct
    roadside, overheads = highwaysigns.arrange(FRAME, {obj['id']: (obj, ribbons)}, roadside, overheads, flat,
                                               [deck])
    assert overheads == [] and len(roadside[0]['rows']) == 2


def test_combined_rows_carry_exit_number_destination_and_distance():
    signs = [advance('302A', 402.3, 0., names=['Broadway', 'Weidler St']), advance('303', 804.7, 4., names=['Moda'])]
    roadside, _ = exitsigns.consolidate(signs, [])
    sign = roadside[0]
    rows = [dict(r, dest=exitsigns.row_destination(r['label'])) for r in sign['rows']]
    assert [(r['ref'], r['dest'], r['distance']) for r in rows] ==         [('EXIT 302A', 'Broadway - Weidler St', '1/4 MILE'), ('EXIT 303', 'Moda', '1/2 MILE')]
    bare = [dict(r, dest='') for r in rows]
    assert exitsigns.list_dimensions(rows)[0] > exitsigns.list_dimensions(bare)[0]   # the panel grows to fit
    texts = [t['value'] for t in exitsigns.list_layout(rows, *exitsigns.list_dimensions(rows))]
    assert texts[:3] == ['EXIT 302A', 'Broadway - Weidler St', '1/4 MILE']
    # Too wide for one panel (MUTCD letter sizes): marked for one panel per exit side by side, and the list
    # kept for where no gantry fits is trimmed to fit.
    assert sign.get('wide') and not sign.get('split')
    assert exitsigns.list_dimensions(sign['rows'])[0] <= exitsigns.LIST_MAX_W or         all(not r['dest'] for r in sign['rows'])


def test_wide_list_becomes_side_by_side_panels_each_with_its_distance():
    import highwaysigns
    obj, ribbons, signs = _split_fixture()
    signs[1].update(advance_m=804.7, distance='1/2 MILE')
    signs[1]['centre'][0] = -399.
    roadside, overheads = exitsigns.consolidate(signs, [])
    assert roadside[0].get('wide')
    _, overheads = highwaysigns.arrange(FRAME, {obj['id']: (obj, ribbons)}, roadside, overheads, flat)
    assert [p['distance'] for p in overheads[0]['panels'] if p['exit']] == ['1/4 MILE', '1/2 MILE']


def test_untagged_link_is_signed_with_the_freeway_it_reaches():
    # A link from I-84 to I-5 with only a junction number: "I 5 SOUTH" from the reached way's ref and the
    # direction driven on it (an odd Interstate runs north-south). No destination tags, no place names.
    coords = {1: ll(-1000., 0.), 2: ll(0., 0.), 3: ll(500., 0.), 10: ll(60., -60.), 11: ll(200., -300.),
              12: ll(200., -1000.)}
    main = [(100, [1, 2, 3], dict(highway='motorway', oneway='yes', ref='I 84')),
            (300, [11, 12], dict(highway='motorway', oneway='yes', ref='I 5'))]
    links = [(200, [2, 10, 11], dict(highway='motorway_link', oneway='yes'))]
    found = exitsigns.index_records(main, links, coords, {2: dict(highway='motorway_junction', ref='302A')})
    label = found[0]['ways'][0]['label']
    assert label['routes'] == ['I 5 SOUTH'] and label['source'] == 'target' and label['ref'] == '302A'
    # Posted tags always win over the derived route.
    links = [(200, [2, 10, 11], dict(highway='motorway_link', oneway='yes', destination='Salem'))]
    label = exitsigns.index_records(main, links, coords, {2: dict(ref='302A')})[0]['ways'][0]['label']
    assert label['names'] == ['Salem'] and 'source' not in label


def test_gantry_clears_two_roads_at_different_heights_crossing_it():
    import gantryclear
    from test_through_signs import pavement, through_object
    obj, box = through_object(), (-300., -100., 300., 100.)
    free, = throughsigns.place(FRAME, [obj], [pavement()], box, flat)
    x = free['centre'][0]

    def crossing(at, top):          # a road crossing the freeway on a deck at height top
        return dict(deck=True, n=[[at, -60., top], [at, 60., top]], w=12., top=top)
    low, high = crossing(x, 7.), crossing(x+3., 30.)
    assert gantryclear.clashes(free, [low]) and not gantryclear.clashes(free, [high])
    assert not gantryclear.clashes(free, [crossing(x, -9.)])         # far below: 5.3 m clear over it
    assert gantryclear.clashes(free, [dict(deck=True, n=[[x, -60.], [x, 60.]], w=12.)])   # unknown height
    # Both roads cross the span: the gantry slides along its own road to clear the low one.
    moved, = throughsigns.place(FRAME, [obj], [pavement()], box, flat, [low, high])
    assert abs(moved['centre'][0]-x) >= 6. and moved.get('slid_m')
    assert not gantryclear.clashes(moved, [low, high])
    # No clear spot within reach: no gantry at all.
    wall = [crossing(x+s, 7.) for s in range(-160, 170, 10)]
    assert throughsigns.place(FRAME, [obj], [pavement()], box, flat, wall) == []
