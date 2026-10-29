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
