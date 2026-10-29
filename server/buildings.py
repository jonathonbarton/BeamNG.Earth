"""Buildings (v1.9; the user, 2026-10-13: "make the world look alive"): Overture footprints extruded into
walls with a window band per floor and a flat, gabled or hipped roof, one mesh per road tile (roads.TILE),
written next to the tile's road mesh and shown as a colliding TSStatic by the mod (roads.lua, "bld").

Overture rarely knows a building's height (Fells Point, 2026-10-13: 2 % of 12458; towns 50-80 %) and
almost never its roof shape or facade, so most of it is predicted from the footprint (area, narrowness,
class), deterministically per building (ROADMAP "deterministic generation": a hash of the footprint's
position, so it is the same building whichever way the player came).

Facades use the stock window sheets (asset survey 2026-10-13): m_highrise_plaster's four 256 px bands are
four floor styles (tall windows on plaster, square windows on panels, sash windows on plaster, sash windows
on red brick), m_highrise_metal's are office curtain wall and a shopfront; each floor of a wall is one band,
its windows a whole number of bays along every wall (no window cut at a corner). Pure functions of the
footprints and a ground-height function; no game state.
"""
import os
import hashlib
import math
import re

import numpy as np

import fast
import mapgeom
import staging

FLOOR_M = 3.0               # m per storey (residential)
GROUND_FLOOR_SHOP_M = 4.2   # m, a shop's ground floor
FOUNDATION_M = 0.6          # m the walls reach under the lowest ground of the footprint
# The walls reach this far under the lowest ground of the footprint (a skirt): beyond the physics block (+-512 m) the
# ground drawn is the far field's mesh, coarser with distance (tens of metres between vertices at 1-2 km), which on any
# slope runs metres under the heights the buildings stand on: they floated (the user, 2026-10-18: "houses and
# buildings in many parts of the world are floating"). Under a correct ground the skirt is hidden; the floors stay
# where they were (anchored at the eaves, the rest a plain plinth). The lite mesh (far tiles, a near tile's distant
# detail) stands on the coarsest ground: the deepest skirt.
SKIRT_M = 3.0
SKIRT_LITE_M = 10.0
MIN_AREA_M2 = 12.0          # footprints smaller than this are skipped (kiosks, sheds' slivers)
MAX_VERTS = 64              # footprints with more vertices are simplified until under this
SIMPLIFY_M = 0.35           # m: vertices closer than this to the line of their neighbours are dropped
TILE_PX_M = 12.0            # m of wall per 1024 px of the window sheets (5 sash bays of 2.4 m)
ROOF_UV_M = 6.0             # m per repeat of the roof textures
SIDING_UV_M = 3.0           # m per repeat of the plain wall textures (sheds, garages)
PITCH = {"gabled": 32.0, "hipped": 26.0}     # degrees
BARREL_RISE = 0.45          # an arched roof's rise, of its half span (a Quonset hut 1.0; halls and hangars lower)
BARREL_MAX_HW = 45.0        # m: an arched roof spans at most twice this
EAVE_M = 0.35               # m a pitched roof overhangs its walls
CORNICE_OUT_M = 0.28        # a flat roof's cornice: how far it stands out, its depth below and above the eave
CORNICE_DOWN_M = 0.45
CORNICE_UP_M = 0.2

# Materials (mod/.../art/shapes/beamngEarth/buildings.materials.json): the window sheets in a few tints, the
# shop / office sheet, roofs, plain walls.
FACADE_TINTS = ("bne_bld_fac_white", "bne_bld_fac_cream", "bne_bld_fac_sand", "bne_bld_fac_grey")
PAINT_TINTS = ("bne_bld_fac_blue", "bne_bld_fac_green", "bne_bld_fac_rose", "bne_bld_fac_yellow")   # painted houses
BRICK_TINTS = ("bne_bld_fac_white", "bne_bld_fac_brickdark", "bne_bld_fac_brickwarm")   # the brick band's
SHOP = "bne_bld_shop"
OFFICE = "bne_bld_office"
ROOF_FLAT = "bne_bld_roof_flat"
ROOF_SLATE = "bne_bld_roof_slate"
ROOF_SHINGLE = "bne_bld_roof_shingle"
SIDING = "bne_bld_siding"
FACADE = "bne_bld_facade"    # the plaster window sheet, tinted per building by vertex colour
STUCCO = "bne_bld_stucco"   # plain stucco walls (windows laid over them from the sheet)
ROOF_METAL = "bne_bld_roof_metal"   # standing-seam metal (barns, sheds, farm buildings)
PANEL = "bne_bld_panel"     # precast concrete panels (big-box stores, strip malls)
METALWALL = "bne_bld_metalwall"     # corrugated metal walls (warehouses, factories)
STONE = "bne_bld_stone"     # dressed stone (old villages and towns of southern and central Europe: Gordes came out as
                            # grey plaster, 2026-10-15)
STONE_FLOORS = 4            # stone walls on buildings this low at most
BIG_BOX_M2 = 1200.0         # m2: a low commercial building this big is a big box (blank walls, one storefront)
CHURCH_CLASSES = ("church", "chapel", "cathedral")
BARN_CLASSES = {"barn", "farm_auxiliary", "stable", "cowshed"}
# The mesh's materials (vertColor: each building's colours are its vertices'); the names above are the
# looks plan() picks, each one of these with a colour (LOOKS).
MARBLE = "bne_bld_marble"   # cut white marble (the stock m_marble_cut): Baltimore's row house steps
BRICK_WALL = "bne_bld_brick"    # plain red brick (the stock brick_plain): a garage's skin
CONCRETE = "bne_bridge_concrete"  # plain concrete (tunnel.materials.json): a precast garage's skin
LINE_PAINT = "line_white"   # the roads' white lane line (stock decal material, loaded with the road materials): garage stalls
SKY_WALL = "bne_bld_sky"    # the skyline's walls: plain (vertex colours), their windows lit at night (_sky_mesh)
MATERIALS = (FACADE, SHOP, OFFICE, ROOF_FLAT, ROOF_SLATE, ROOF_SHINGLE, SIDING, STUCCO, ROOF_METAL, PANEL, METALWALL,
             STONE, MARBLE, BRICK_WALL, CONCRETE, SKY_WALL, LINE_PAINT)
LOOKS = {
    "bne_bridge_steel": (METALWALL, (0.50, 0.52, 0.54)),
    "bne_marble": (MARBLE, (1.0, 1.0, 1.0)),
    "bne_brick_red": (BRICK_WALL, (1.0, 1.0, 1.0)), "bne_brick_dark": (BRICK_WALL, (0.78, 0.72, 0.70)),
    "bne_brick_tan": (BRICK_WALL, (1.0, 0.92, 0.80)),
    "bne_precast": (CONCRETE, (1.0, 1.0, 1.0)), "bne_precast_warm": (CONCRETE, (1.0, 0.95, 0.86)),
    "bne_bld_fac_white": (FACADE, (1.0, 1.0, 1.0)), "bne_bld_fac_cream": (FACADE, (1.0, 0.93, 0.80)),
    "bne_bld_fac_sand": (FACADE, (0.92, 0.82, 0.68)), "bne_bld_fac_grey": (FACADE, (0.80, 0.82, 0.84)),
    "bne_bld_fac_brickdark": (FACADE, (0.78, 0.72, 0.70)), "bne_bld_fac_brickwarm": (FACADE, (1.0, 0.88, 0.80)),
    "bne_bld_fac_blue": (FACADE, (0.80, 0.87, 0.96)), "bne_bld_fac_green": (FACADE, (0.84, 0.92, 0.82)),
    "bne_bld_fac_rose": (FACADE, (0.99, 0.86, 0.82)), "bne_bld_fac_yellow": (FACADE, (1.0, 0.95, 0.72)),
    LINE_PAINT: (LINE_PAINT, (1.0, 1.0, 1.0)),
    SHOP: (SHOP, (1.0, 1.0, 1.0)), OFFICE: (OFFICE, (1.0, 1.0, 1.0)), SIDING: (SIDING, (0.93, 0.92, 0.88)),
}
ROOF_TINTS = {               # per roof material: the colours a building's roof takes (hashed)
    ROOF_FLAT: ((0.85, 0.85, 0.85), (0.62, 0.62, 0.64), (0.95, 0.93, 0.88), (0.45, 0.45, 0.46), (0.75, 0.70, 0.64)),
    # 4..: clay tiles (terracotta, red-brown, faded orange) and dark grey / anthracite / brown-black (regional)
    ROOF_SLATE: ((1.0, 1.0, 1.0), (0.80, 0.66, 0.56), (0.76, 0.80, 0.90), (0.62, 0.62, 0.62),
                 (0.95, 0.52, 0.36), (0.78, 0.40, 0.30), (1.0, 0.66, 0.46), (0.42, 0.43, 0.46), (0.30, 0.31, 0.33),
                 (0.48, 0.38, 0.32)),
    ROOF_SHINGLE: ((0.75, 0.72, 0.70), (0.55, 0.50, 0.48), (0.70, 0.56, 0.48), (0.62, 0.66, 0.72),
                   (0.95, 0.52, 0.36), (0.78, 0.40, 0.30), (0.42, 0.43, 0.46), (0.36, 0.42, 0.52)),
    ROOF_METAL: ((0.85, 0.86, 0.88), (0.55, 0.57, 0.60), (0.45, 0.58, 0.48), (0.62, 0.30, 0.26), (0.40, 0.42, 0.48)),
}
WALL_TINTS = {               # houses with plain walls and laid-over windows: their colours (hashed)
    SIDING: ((0.96, 0.95, 0.92), (0.90, 0.86, 0.76), (0.78, 0.84, 0.90), (0.95, 0.92, 0.74), (0.78, 0.84, 0.74),
             (0.82, 0.78, 0.72), (0.70, 0.72, 0.75), (0.62, 0.20, 0.17), (0.66, 0.24, 0.20)),   # the last: barn red
    STUCCO: ((1.0, 0.97, 0.90), (0.96, 0.88, 0.74), (0.93, 0.80, 0.68), (0.98, 0.95, 0.88), (0.88, 0.84, 0.78),
             # 5..: bright paints (Latin America, the Mediterranean, the tropics): ochre, terracotta, sky blue,
             # mint, pink, yellow
             (1.0, 0.78, 0.45), (0.92, 0.58, 0.44), (0.66, 0.82, 0.95), (0.70, 0.90, 0.78), (0.98, 0.72, 0.76),
             (1.0, 0.92, 0.52)),
    PANEL: ((0.96, 0.94, 0.89), (0.86, 0.86, 0.86), (0.92, 0.85, 0.74), (0.78, 0.80, 0.82), (0.88, 0.78, 0.68),
            (0.62, 0.62, 0.60), (0.56, 0.54, 0.50)),    # the last two: weathered concrete (parking decks)
    METALWALL: ((0.88, 0.89, 0.90), (0.72, 0.74, 0.77), (0.84, 0.80, 0.72), (0.66, 0.72, 0.78)),
    # limestone, honey, ochre and grey stone (the texture is a light ashlar: tinted down)
    STONE: ((1.0, 0.93, 0.80), (0.98, 0.86, 0.66), (0.95, 0.80, 0.60), (0.86, 0.85, 0.82), (0.92, 0.88, 0.80)),
    # curtain walls (the user, 2026-10-14: "the skyscrapers all use the same texture and look the same colour"):
    # silver, blue, green, bronze, charcoal and steel-blue glass, the common tints of real towers
    OFFICE: ((1.0, 1.0, 1.0), (0.74, 0.84, 0.98), (0.76, 0.92, 0.86), (0.98, 0.84, 0.66), (0.56, 0.58, 0.62),
             (0.66, 0.74, 0.86), (0.88, 0.90, 0.92)),
}
for _m, _cs in list(ROOF_TINTS.items()) + list(WALL_TINTS.items()):
    for _k, _c in enumerate(_cs):
        LOOKS[f"{_m}|{_k}"] = (_m, _c)
# stained wood boards (Alpine chalets; kept apart from the siding tints' indices): larch brown, dark brown, grey-brown
WOOD_TINTS = ((0.62, 0.44, 0.30), (0.46, 0.33, 0.24), (0.58, 0.52, 0.46))
for _k, _c in enumerate(WOOD_TINTS):
    LOOKS[f"{SIDING}|w{_k}"] = (SIDING, _c)
# Shop signs (Bishop's and Moab's main streets: a painted sign band over every shopfront, each shop its own
# colour): boxes over the ground floor of a shop's street wall, a colour per shop width (hashed).
SIGN_TINTS = ((0.62, 0.16, 0.14), (0.16, 0.22, 0.42), (0.18, 0.36, 0.24), (0.95, 0.90, 0.75), (0.42, 0.30, 0.22),
              (0.85, 0.65, 0.20), (0.13, 0.13, 0.14), (0.96, 0.96, 0.96), (0.15, 0.45, 0.48))
for _k, _c in enumerate(SIGN_TINTS):
    LOOKS[f"sign|{_k}"] = (FACADE, _c)
SIGN_H = 0.9                # m, the band's height (its top 0.15 m under the ground floor's top)
SIGN_OUT = 0.18             # m out from the wall
SIGN_SHOP_M = (6.0, 11.0)   # one shop's width (hashed in this range)
BARN_REDS = 2               # the last siding tints are barn red (barns only)
US_WALL_TINTS = {STUCCO: 5}  # the first tints of these kinds are the US's (later ones: regional paints)
JITTER = 0.06               # a building's colour varies this much (one factor, hashed)

COLOR_NAMES = {
    "white": (0.95, 0.95, 0.95), "black": (0.12, 0.12, 0.12),
    "grey": (0.50, 0.50, 0.50), "gray": (0.50, 0.50, 0.50),
    "lightgrey": (0.75, 0.75, 0.75), "lightgray": (0.75, 0.75, 0.75),
    "darkgrey": (0.30, 0.30, 0.30), "darkgray": (0.30, 0.30, 0.30),
    "silver": (0.75, 0.75, 0.78), "charcoal": (0.20, 0.20, 0.22), "anthracite": (0.22, 0.23, 0.25),
    "red": (0.75, 0.15, 0.15), "darkred": (0.50, 0.10, 0.10), "lightred": (0.85, 0.35, 0.35),
    "maroon": (0.45, 0.10, 0.15), "burgundy": (0.45, 0.12, 0.20), "crimson": (0.70, 0.10, 0.20),
    "pink": (0.90, 0.60, 0.65), "rose": (0.90, 0.55, 0.60), "coral": (0.90, 0.45, 0.35),
    "salmon": (0.90, 0.50, 0.45), "brick": (0.65, 0.25, 0.18),
    "green": (0.18, 0.55, 0.22), "darkgreen": (0.10, 0.35, 0.12), "lightgreen": (0.45, 0.75, 0.40),
    "olive": (0.45, 0.50, 0.20), "lime": (0.55, 0.75, 0.20), "forestgreen": (0.13, 0.40, 0.15),
    "teal": (0.12, 0.45, 0.45), "mint": (0.60, 0.85, 0.70), "sage": (0.55, 0.65, 0.55), "moss": (0.35, 0.45, 0.25),
    "blue": (0.18, 0.38, 0.72), "darkblue": (0.10, 0.20, 0.45), "lightblue": (0.50, 0.70, 0.88),
    "navy": (0.08, 0.15, 0.35), "skyblue": (0.50, 0.75, 0.90), "cyan": (0.20, 0.70, 0.75),
    "azure": (0.30, 0.60, 0.85), "cobalt": (0.15, 0.30, 0.65), "turquoise": (0.25, 0.70, 0.65),
    "steel": (0.45, 0.50, 0.55), "steel_blue": (0.28, 0.45, 0.60),
    "yellow": (0.92, 0.82, 0.20), "lightyellow": (0.95, 0.92, 0.60), "gold": (0.85, 0.70, 0.18),
    "orange": (0.90, 0.48, 0.12), "darkorange": (0.80, 0.35, 0.08), "amber": (0.88, 0.60, 0.12),
    "brown": (0.45, 0.30, 0.20), "darkbrown": (0.30, 0.20, 0.12), "lightbrown": (0.60, 0.45, 0.32),
    "rust": (0.60, 0.25, 0.15), "beige": (0.90, 0.85, 0.75), "cream": (0.95, 0.92, 0.82),
    "tan": (0.82, 0.70, 0.55), "terracotta": (0.78, 0.40, 0.26), "ochre": (0.80, 0.55, 0.20),
    "sand": (0.88, 0.80, 0.65), "khaki": (0.75, 0.70, 0.50), "copper": (0.55, 0.75, 0.65),
    "bronze": (0.55, 0.42, 0.25), "purple": (0.50, 0.20, 0.55), "violet": (0.55, 0.25, 0.60),
    "magenta": (0.75, 0.15, 0.55), "indigo": (0.25, 0.10, 0.45), "lavender": (0.75, 0.70, 0.85),
    "plum": (0.45, 0.20, 0.40),
}


def parse_color(val):
    """Parse an OSM/Overture color (hex or CSS name) into an (r, g, b) float tuple in [0, 1], or None."""
    if not val or not isinstance(val, str):
        return None
    s = val.strip().lower().replace(" ", "").replace("-", "_")
    if s in COLOR_NAMES:
        return COLOR_NAMES[s]
    if s.startswith("#"):
        hex_str = s[1:]
        if len(hex_str) in (3, 4):
            try:
                return tuple(int(hex_str[i] * 2, 16) / 255.0 for i in range(3))
            except ValueError:
                return None
        elif len(hex_str) in (6, 8):
            try:
                return tuple(int(hex_str[2 * i:2 * i + 2], 16) / 255.0 for i in range(3))
            except ValueError:
                return None
    return None


def custom_look(mat, rgb):
    """A look key in LOOKS for material mat with RGB tuple in [0, 1]."""
    key = f"{mat}|c_{rgb[0]:.3f}_{rgb[1]:.3f}_{rgb[2]:.3f}"
    if key not in LOOKS:
        LOOKS[key] = (mat, rgb)
    return key


class _Looks:
    """name -> (material index, RGBA bytes) for one building (its hashed brightness jitter)."""

    def __init__(self, seed):
        self.f = 1.0 + JITTER * (2.0 * _hash01(seed, "jit") - 1.0)

    def __getitem__(self, name):
        if name in LOOKS:
            mat, c = LOOKS[name]
        elif name.startswith("custom|") or ("|c_" in name):
            mat = name.split("|")[0]
            c = (1.0, 1.0, 1.0)
        else:
            mat, c = LOOKS.get(name, (FACADE, (1.0, 1.0, 1.0)))
        return MATERIALS.index(mat), tuple(int(max(0, min(255, round(255 * v * self.f)))) for v in c) + (255,)

# The window sheets' bands, top of the image first (v in Collada convention: v = 1 at the image top).
# (sheet, band index, bays per 1024 px)
BAND_TALL = (0, 3)          # m_highrise_plaster: tall windows on plaster (3 bays of uneven width)
BAND_PANEL = (1, 4)         # square windows on panels
BAND_SASH = (2, 5, (0, 2, 3))   # sash windows on plaster (bays 1 and 4 are black "open" windows: skipped)
BAND_BRICK = (3, 5)         # sash windows on red brick
BAND_OFFICE = (0, 5)        # m_highrise_metal: curtain wall
BAND_OFFICE2 = (1, 4)
BAND_SHOP = (2, 2)          # shopfront with awnings
BANDS = 4

# Overture "bridge_structure" (OSM man_made=bridge parts with a height: the Queensboro Bridge's towers, 110 m) was a
# glass tower the decks ran through (the user's report, 2026-10-17)
BRIDGE_CLASSES = {"bridge_structure", "bridge_tower"}
BRIDGE_PIER_M = 12.0        # a bridge structure's height when not mapped
STEEL_MATERIALS = ("", "steel", "metal", "iron")     # a part over a bridge support of these (or none mapped): steel
MASONRY_MATERIALS = ("concrete", "stone", "brick", "masonry", "granite")
STONE_FACADES = ("stone", "sandstone", "limestone", "granite", "marble", "masonry", "tuff", "slate")
STEEL = "bne_bridge_steel"  # (a LOOKS name: the corrugated-metal material, painted steel grey)
# a bridge structure in its mapped paint (bridgetypes.PAINTS: the Golden Gate Bridge's towers, building:colour #bf4e3b,
# were steel grey beside its orange cables, 2026-10-18)
for _k, _c in __import__("bridgetypes").PAINTS.items():
    LOOKS[f"{STEEL}|{_k}"] = (METALWALL, tuple(v / 255.0 for v in _c))
STEEL_LEG_MAX = 3.0         # m: a frame's legs at most this thick (a quarter of its short side)
STEEL_BAY_M = 14.0          # m between a frame's bracing levels
STEEL_TOP_SHARE = 0.85      # cross beams over the decks only from this share of the height up
SHED_CLASSES = {"shed", "garage", "garages", "carport", "roof", "hut", "cabin", "barn", "farm_auxiliary",
                "greenhouse", "service", "storage_tank", "toilets", "kiosk", "parking", "shelter"}
# A grain elevator (the plains' small towns: Strasburg, CO by I-70 had two 8-storey brick apartment blocks with windows
# in the farmland): a tall slender footprint of no class (or a silo's) in a US small town, plain concrete
# A pyramid (Giza's, GitHub #47: drawn as 66-140 m boxes with windows): a pyramidal roof mapped as (nearly) the whole
# height over a monument-sized footprint is a stone pyramid on a low plinth (pitched roofs are otherwise flat past 28 m)
PYRAMID_ROOF_SHARE = 0.8
PYRAMID_MIN_M2 = 500.0
PYRAMID_PLINTH_M = 1.0
ELEVATOR_CLASSES = {"", "industrial", "silo", "grain_silo", "storage_tank", "farm_auxiliary", "agricultural"}
ELEVATOR_MIN_H = 15.0       # m mapped (a height, no floors)
ELEVATOR_MAX_M2 = 1500.0
ELEVATOR_SLENDER = 0.7      # height / sqrt(area) at least (11 x 11 m, 25 m: 2.3; 28 x 31 m, 23 m: 0.78; a 15 m motel: 0.4)
GLASS_FLOORS = (20, 40)     # towers from 20 floors: glass, but for OLD_TOWER_GLASS of the old ones below 40 ...
OLD_TOWER_GLASS = 0.65      # ... (Midtown's pre-war skyscrapers are stone)
MIDRISE_GLASS_OFFICE = 0.45 # offices / hotels of 7+ floors: this share glass (modern), the rest masonry
TOWER_SHOP_FLOORS = 6       # a building this tall in a dense core: a shop / lobby ground floor whatever its class
SHOP_CLASSES = {"retail", "commercial", "supermarket", "kiosk", "office", "hotel", "civic", "restaurant"}
HOUSE_CLASSES = {"house", "detached", "semidetached_house", "bungalow", "residential", "farm", "static_caravan",
                 "cabin", "dormitory"}
ROW_CLASSES = {"terrace"}
GARAGE_MIN_M2 = 400.0       # a "parking" building this big is a parking structure
INDUSTRIAL_CLASSES = {"industrial", "warehouse", "manufacture", "factory", "hangar"}


def _hash01(*vals):
    """A number in [0, 1) from the values (stable across runs and machines)."""
    h = hashlib.blake2b(repr(vals).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2.0 ** 64


# ------------------------------------------------------------------------------------------- polygons

def _area2(p):
    """Twice the signed area of the ring p (n, 2), counter-clockwise positive."""
    x, y = p[:, 0], p[:, 1]
    # np.roll(v, -1) by hand: the same arrays without roll's axis handling (~28k calls a 9-tile set)
    return float(np.dot(x, np.concatenate((y[1:], y[:1]))) - np.dot(np.concatenate((x[1:], x[:1])), y))


def clean_ring(p):
    """The ring (n, 2) without its closing point, duplicate and nearly collinear vertices, counter-clockwise;
    None if fewer than 3 are left. The simplification in Rust (earthfast.clean_ring_core: numpy's arithmetic)."""
    fast.require(76, "clean_ring_core")
    p = fast._ef.clean_ring_core(np.ascontiguousarray(np.asarray(p, np.float64)[:, :2]), float(SIMPLIFY_M), int(MAX_VERTS))
    if p is None:
        return None
    p = np.asarray(p, np.float64)
    if _area2(p) < 0:
        p = p[::-1]
    return p


def triangulate(p):
    """Ear clipping of the counter-clockwise simple ring p (n, 2): (n-2, 3) indices into p (fewer if the
    ring is not simple: the rest is left out). The Rust loop (earthfast.triangulate)."""
    fast.require(76, "triangulate")
    return np.asarray(fast._ef.triangulate(np.ascontiguousarray(np.asarray(p, np.float64)[:, :2])), np.int64)


def oriented_box(p):
    """The minimum-area rectangle around ring p along one of its edges: (centre (2,), axis u (2,) along the
    long side, half length, half width). The edge search in Rust (earthfast.oriented_box_core: the same products and
    sums, math.hypot's own rounding), the rest here as before."""
    fast.require(76, "oriented_box_core")
    best = fast._ef.oriented_box_core(np.ascontiguousarray(np.asarray(p, np.float64)[:, :2]))
    if best is None:
        raise ValueError("oriented_box: a ring without an edge")
    ux, uy, vx, vy, amin, amax, bmin, bmax = best
    sa, sb = amax + amin, bmax + bmin
    c = np.array([ux * sa / 2 + vx * sb / 2, uy * sa / 2 + vy * sb / 2])
    hl, hw = np.float64((amax - amin) / 2), np.float64((bmax - bmin) / 2)     # (numpy scalars, as before)
    u = np.array([ux, uy])
    if hw > hl:
        u, hl, hw = np.array([vx, vy]), hw, hl
    return c, u, hl, hw


# ------------------------------------------------------------------------------------------- the model

class Spec:
    """What is built for one footprint (decided by plan())."""
    __slots__ = ("ring", "floors", "floor_h", "height", "roof", "roof_mat", "facade", "band", "shop", "plain", "min_h", "wall", "door", "windows", "front", "steeple", "stoop",
                 "sign", "porch", "cls", "seed", "deck", "pitch", "portal", "gring", "circle")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class Style:
    """A region's building habits (2026-10-15, the user: "the other parts of the world need more love": Swiss
    villages came out as clapboard houses with porches). Shares are of buildings whose data does not say:
    brick: of facades; walls: (kind, share) for detached houses' plain walls (SIDING, STUCCO; the rest the
    plaster window sheet), wall_tints: the tint indices the plain walls take (None: all but barn red);
    paint: plaster houses painted (PAINT_TINTS); porch; flat: detached houses with flat roofs; hipped: of the
    pitched; roofs: (roof material, (tint indices)) to pick from; pitch: the roof pitch in degrees (None: PITCH);
    stone: the share of low older buildings (houses and attached ones, up to STONE_FLOORS, no shops) with dressed stone
    walls; wood: siding walls are stained wood (WOOD_TINTS, barns too); barn_red: the share of barns painted barn red
    (the US's and Scandinavia's; elsewhere their siding colours: a Lauterbrunnen garage row came out a 150 m red
    wall, 2026-10-15); attached: low attached buildings (terraces, old town
    blocks) with a pitched roof (the US: flat; Amsterdam's canal houses, Europe's old towns: gables).
    Coarse boxes by region until a facade-material map; the US ones keep their earlier behaviour."""

    def __init__(self, name, brick, walls, porch, flat, hipped, roofs, pitch=None, paint=0.3, wall_tints=None,
                 wood=False, attached=0.0, barn_red=0.0, stone=0.0):
        self.name, self.brick, self.walls, self.porch, self.flat = name, brick, walls, porch, flat
        self.barn_red, self.stone = barn_red, stone
        self.hipped, self.roofs, self.pitch, self.paint, self.wall_tints = hipped, roofs, pitch, paint, wall_tints
        self.wood, self.attached = wood, attached


_US_ROOFS = ((ROOF_SLATE, (0, 1, 2, 3)), (ROOF_SHINGLE, (0, 1, 2, 3)))
_CLAY = (ROOF_SLATE, (4, 5, 6))
_DARK = (ROOF_SLATE, (7, 8, 9))
STYLES = {
    "us_east": Style("us_east", 0.75, ((SIDING, 0.75),), 0.55, 0.0, 0.4, _US_ROOFS, barn_red=0.5),
    "us_south": Style("us_south", 0.35, ((SIDING, 0.75),), 0.55, 0.0, 0.4, _US_ROOFS, barn_red=0.5),
    "us_plains": Style("us_plains", 0.15, ((SIDING, 0.75),), 0.55, 0.0, 0.4, _US_ROOFS, barn_red=0.5),
    "us_west": Style("us_west", 0.15, ((STUCCO, 0.65), (SIDING, 0.25)), 0.25, 0.0, 0.4, _US_ROOFS, barn_red=0.5),
    # Alpine villages: white plaster ground floors, dark wooden chalets, low wide gables of dark or brown roofs
    "alpine": Style("alpine", 0.05, ((SIDING, 0.45),), 0.0, 0.03, 0.15, (_DARK, (ROOF_SLATE, (9, 5))), pitch=24.0,
                    paint=0.05, wood=True, attached=0.85, stone=0.15),
    # north-west / central Europe: brick (Low Countries, northern Germany, Britain) or plaster, steep tiled or
    # slate roofs
    "europe_north": Style("europe_north", 0.5, (), 0.02, 0.05, 0.3, (_CLAY, _DARK, (ROOF_SLATE, (5,))), pitch=42.0,
                          paint=0.15, attached=0.75, stone=0.12),
    # Scandinavia and Finland: painted wooden houses (siding), little brick, dark or tiled roofs, falu-red barns
    "nordic": Style("nordic", 0.12, ((SIDING, 0.65),), 0.0, 0.08, 0.2, (_DARK, _CLAY), pitch=35.0, paint=0.2,
                    attached=0.3, barn_red=0.7),
    "mediterranean": Style("mediterranean", 0.08, ((STUCCO, 0.85),), 0.0, 0.35, 0.5, (_CLAY,), pitch=22.0,
                           paint=0.1, wall_tints=(0, 1, 2, 3, 4, 5), attached=0.8, stone=0.4),
    "latin_america": Style("latin_america", 0.25, ((STUCCO, 0.9),), 0.02, 0.7, 0.3, (_CLAY, (ROOF_METAL, (0, 1))),
                           pitch=22.0, paint=0.0, wall_tints=(0, 1, 2, 3, 5, 6, 7, 8, 9, 10)),
    "east_asia": Style("east_asia", 0.08, ((SIDING, 0.5), (STUCCO, 0.3)), 0.0, 0.35, 0.5,
                       (_DARK, (ROOF_SHINGLE, (6, 7)), (ROOF_METAL, (1, 4))), pitch=25.0, paint=0.0,
                       wall_tints=(0, 1, 6, 5), attached=0.3),
    "oceania": Style("oceania", 0.55, ((STUCCO, 0.5), (SIDING, 0.3)), 0.25, 0.05, 0.7, (_CLAY, (ROOF_METAL, (0, 1, 4))),
                     pitch=24.0, paint=0.1),
    # South Asia (India, Pakistan, Bangladesh, Sri Lanka, Nepal): a concrete frame and brick under plaster, flat terrace
    # roofs, pastel and whitewashed walls, metal roofs on sheds (OmdevShastri, PR #146)
    "south_asia": Style("south_asia", 0.08, ((STUCCO, 0.92),), 0.0, 0.94, 0.2, (_CLAY, (ROOF_METAL, (0, 1, 3))),
                        pitch=18.0, paint=0.15, wall_tints=(0, 1, 2, 3, 5, 6, 8, 10), attached=0.90),
    # elsewhere (Africa, the Middle East, ...): concrete and plaster, mostly flat roofs
    "warm": Style("warm", 0.15, ((STUCCO, 0.85),), 0.0, 0.75, 0.4, (_CLAY, (ROOF_METAL, (0, 1, 3))), pitch=20.0,
                  paint=0.0, wall_tints=(0, 1, 2, 3, 4, 5, 6, 8)),
    "cool": Style("cool", 0.3, ((SIDING, 0.4),), 0.05, 0.1, 0.3, (_DARK, (ROOF_METAL, (0, 1, 2, 3))), pitch=30.0,
                  paint=0.3, barn_red=0.5),
}


def region_style(lat, lon):
    """The Style of the region around (lat, lon) (None: the US East's)."""
    if lat is None or lon is None:
        return STYLES["us_east"]
    if 24.0 <= lat <= 72.0 and -170.0 <= lon <= -52.0:          # North America (and Canada)
        if lon < -100.0:
            return STYLES["us_west"]
        if lon <= -95.0:
            return STYLES["us_plains"]
        return STYLES["us_east"] if lat > 36.0 else STYLES["us_south"]
    if -56.0 <= lat < 24.0 and -120.0 <= lon <= -30.0:          # Mexico, the Caribbean, Central and South America
        return STYLES["latin_america"]
    if 45.4 <= lat <= 48.0 and 5.8 <= lon <= 16.5:              # the Alps
        return STYLES["alpine"]
    if 56.0 <= lat <= 72.0 and 4.5 <= lon <= 32.0:              # Norway, Sweden, Finland (brick Copenhagen, Malmo: not)
        return STYLES["nordic"]
    if 46.0 <= lat <= 72.0 and -11.0 <= lon <= 32.0:
        return STYLES["europe_north"]
    if 30.0 <= lat < 46.0 and -10.0 <= lon <= 40.0:             # southern Europe, North Africa's coast, Levant
        return STYLES["mediterranean"]
    if 20.0 <= lat <= 46.0 and 100.0 <= lon <= 146.0:           # China, Korea, Japan, Taiwan
        return STYLES["east_asia"]
    if -50.0 <= lat <= -10.0 and 110.0 <= lon <= 180.0:         # Australia, New Zealand
        return STYLES["oceania"]
    if 6.0 <= lat <= 37.5 and 68.0 <= lon <= 92.5:              # South Asia (not Myanmar, east of 92.5)
        return STYLES["south_asia"]
    if abs(lat) >= 50.0:
        return STYLES["cool"]
    return STYLES["warm"]


def brick_share(lat, lon):
    """The share of brick facades in a region (old East Coast / Midwest towns are mostly brick; the South and
    the West plaster, siding and stucco; region_style elsewhere). A coarse stand-in until a facade-material map."""
    if lat is None:
        return 0.4
    return region_style(lat, lon).brick


def roof_pitch(spec):
    """The roof pitch (degrees) of a Spec: its region's, else PITCH."""
    return spec.pitch or PITCH.get(spec.roof, PITCH["gabled"])


ALPINE_HUT_M2 = 120.0        # an isolated building this small above the tree line (a summit shelter, a mountain hut) ...
ALPINE_COVER = 0.02          # ... in a tile this empty gets dressed stone walls (the user, 2026-10-14: Mount Whitney's
                             # stone summit hut was a plain grey box)


def tree_line_m(lat):
    """A rough tree line (m) by latitude: ~3,600 m to 32 deg, falling ~110 m per degree (the Alps at 46 deg ~2,050 m,
    Scandinavia at 60 deg ~520 m), sea level towards 70 deg. Pure."""
    return max(0.0, 3600.0 - 110.0 * max(0.0, abs(float(lat)) - 32.0))


def plan(ring, attrs, neighbours=0, lat=None, lon=None, look=None, town=False, dense=False, west_town=False,
         alpine=False):
    """The Spec of a footprint (local ring (n, 2) in metres, counter-clockwise) from its Overture attributes
    (dict: class, height, num_floors, min_height, roof_shape, roof_height; garage: a mapped multi-storey car park
    covers it, garage_rings) and the number of buildings
    sharing a wall with it. Height and roof are predicted when not given. west_town: a small town of the
    arid West (district_town): single-storey main streets and ranch houses."""
    area = _area2(ring) / 2
    c = ring.mean(0)
    seed = _hash01(round(float(c[0]) * 2), round(float(c[1]) * 2), round(area)) if look is None else look
    style = region_style(lat, lon)
    _, _, hl, hw = oriented_box(ring)
    box_fill = area / max(4 * hl * hw, 1e-6)
    cls = (attrs.get("class") or "").lower()
    name = str(attrs.get("name") or "").lower()
    terminal = any(w in name for w in ("terminal", "concourse"))
    church = cls in CHURCH_CLASSES or any(w in name for w in ("church", "chapel", "cathedral"))
    if "hangar" in name:
        cls = "hangar"
    # a parking structure (a small "parking" building is a carport or a booth: a shed); "parking" is also a
    # shed class, which made big garages barns in barn-red siding (Columbia, 2026-10-14)
    garage = (cls in ("parking", "parking_garage") and area >= GARAGE_MIN_M2) or "parking garage" in name or (
        "garage" in name and area > 800) or (bool(attrs.get("garage")) and cls in GARAGE_TAG_CLASSES)
    narrow = hw < 4.2 and hl > 2.2 * hw                 # a row house: ~4-8 m wide, deep
    # in a dense core small buildings are row houses even where the footprints leave gaps (Fells Point)
    row = cls in ROW_CLASSES or (not west_town and ((narrow and (neighbours >= 1 or dense)) or (
        dense and area < 260 and cls not in SHOP_CLASSES)))
    shed = (cls in SHED_CLASSES and not garage) or (area < 40 and not row)
    shop = cls in SHOP_CLASSES
    industrial = cls in INDUSTRIAL_CLASSES or (area > 2500 and cls in ("", "commercial"))
    industrial = industrial and not (garage or terminal)

    h = attrs.get("height")
    nf = attrs.get("num_floors")
    if terminal and (h is None or not np.isfinite(h)) and (nf is None or not np.isfinite(nf)):
        h, nf = 13.0, 2                                 # an airport terminal: two tall glass storeys
    if garage and (h is None or not np.isfinite(h)) and (nf is None or not np.isfinite(nf)):
        nf = 4 + int(seed * 2)
    if cls == "hangar" and (h is None or not np.isfinite(h)):
        h, nf = 12.0, 1
    if church and (h is None or not np.isfinite(h)) and (nf is None or not np.isfinite(nf)) and area < 2500:
        nf = 1                                          # a nave: one tall storey under a steep gable
    h = None if h is None or not np.isfinite(h) or h <= 0 else float(h)
    nf = None if nf is None or not np.isfinite(nf) or nf <= 0 else int(round(nf))
    rh = attrs.get("roof_height")
    rh = float(rh) if rh is not None and np.isfinite(rh) else None
    if ((attrs.get("roof_shape") or "").lower() == "pyramidal" and h is not None and rh is not None
            and rh >= PYRAMID_ROOF_SHARE * h and area >= PYRAMID_MIN_M2 and hw > 1.0):
        look = f"{STONE}|0"
        return Spec(ring=ring, floors=1, floor_h=PYRAMID_PLINTH_M, height=PYRAMID_PLINTH_M, roof="hipped", roof_mat=look,
                    facade=look, band=None, shop=False, plain=True, seed=seed, min_h=0.0, wall=None, windows=False,
                    steeple=False, porch=False, cls="pyramid", deck=False,
                    pitch=math.degrees(math.atan2(max(h - PYRAMID_PLINTH_M, 0.1), hw)))
    if (town and style.name.startswith("us_") and cls in ELEVATOR_CLASSES and h is not None and h >= ELEVATOR_MIN_H
            and nf is None and area <= ELEVATOR_MAX_M2 and h / math.sqrt(max(area, 1.0)) >= ELEVATOR_SLENDER
            and not (church or terminal or garage)):
        return Spec(ring=ring, floors=1, floor_h=h, height=h, roof="flat", roof_mat="bne_precast", facade="bne_precast",
                    band=None, shop=False, plain=True, seed=seed, min_h=0.0, wall=None, windows=False, steeple=False,
                    porch=False, cls="elevator", deck=False, pitch=None)
    if attrs.get("landmark") in LANDMARK_KINDS:       # a ship, a dome, a tank, a vault, a pier (landmark_kind)
        return landmark_spec(attrs["landmark"], ring, attrs, seed)
    if attrs.get("bridge_part"):    # a mapped part of a bridge's tower (support_rings): a plain box of painted steel
        return Spec(ring=ring, floors=1, floor_h=h or BRIDGE_PIER_M, height=h or BRIDGE_PIER_M, roof="flat",
                    roof_mat=f"{STEEL}|{attrs['bridge_part']}", facade=f"{STEEL}|{attrs['bridge_part']}", band=None,
                    shop=False, plain=True,
                    seed=seed, min_h=0.0, wall=None, windows=False, steeple=False, porch=False, cls="bridge_part",
                    deck=False, pitch=None)
    mat_ = str(attrs.get("facade_material") or "").lower()
    if attrs.get("bridge_masonry") or (cls in BRIDGE_CLASSES and mat_ in MASONRY_MATERIALS):
        # a mapped masonry pier (Tower Bridge's 9 m concrete piers under its stone towers): a solid block, not a frame
        look = f"{STONE}|0" if mat_ in ("stone", "brick") else "bne_precast"
        return Spec(ring=ring, floors=1, floor_h=h or BRIDGE_PIER_M, height=h or BRIDGE_PIER_M, roof="flat",
                    roof_mat=look, facade=look, band=None, shop=False, plain=True, seed=seed, min_h=0.0, wall=None,
                    windows=False, steeple=False, porch=False, cls="bridge_pier", deck=False, pitch=None)
    if cls in BRIDGE_CLASSES:       # a bridge's tower or pier: an open steel frame (_steel_frame), the decks through it
        import bridgetypes
        pt = bridgetypes.paint(attrs.get("facade_color"))
        return Spec(ring=ring, floors=1, floor_h=h or BRIDGE_PIER_M, height=h or BRIDGE_PIER_M, roof="flat",
                    roof_mat=ROOF_FLAT, facade=f"{STEEL}|{pt}" if pt else STEEL, band=None, shop=False, plain=True,
                    seed=seed, min_h=0.0,
                    wall=None, windows=False, steeple=False, porch=False, cls=cls, deck=False, pitch=None)
    floor_h = FLOOR_M
    if industrial:
        floor_h = 4.5
    if church:
        floor_h = 7.5
    # (a big shed is a barn; a row of garages, a carport or a canopy is not)
    barn = cls in BARN_CLASSES or (shed and area >= 150 and cls not in ("garage", "garages", "carport", "roof", "shelter"))
    if barn:
        floor_h = 5.5 if area >= 150 else 3.5          # one tall storey; the gable above it
    if h is None and nf is None:
        if shed:
            nf = 1
        elif row:
            nf = 2 if seed < 0.25 else (3 if seed < 0.85 else 4)
        elif west_town and not industrial and cls != "apartments":
            # Bishop's and Moab's photos (2026-10-14): one-storey shops with false fronts, ranch houses
            nf = 1 if seed < WEST_TOWN_ONE_STOREY else 2
        elif cls in HOUSE_CLASSES or (area < 220 and not shop):
            nf = 1 if seed < 0.3 else 2
        elif industrial:
            nf = 2 if area > 6000 else 1
        elif cls == "apartments":
            nf = 3 + int(seed * 3)
        elif town:
            nf = 1 if seed < 0.6 else 2                 # small towns: mostly single-storey shops and offices
        elif area < 900:
            nf = 2 + int(seed * 2)
        else:
            nf = 2 + int(seed * 3)
    roof_given = (attrs.get("roof_shape") or "").lower()
    if roof_given in ("gabled", "hipped", "flat"):
        roof = roof_given
    elif roof_given in ("pyramidal", "half-hipped", "mansard", "gambrel"):
        roof = "hipped"
    elif roof_given in ("round", "barrel", "arched"):
        roof = "barrel"                         # an arched hall (OSM roof:shape=round): _roof's vault
    elif roof_given in ("skillion", "dome", "onion", "saltbox"):
        roof = "gabled"
    else:
        detached_house = (cls in HOUSE_CLASSES or (area < 260 and not shop)) and not industrial and not row
        if church and neighbours == 0:
            roof = "gabled"
        elif neighbours >= 1:
            # attached: flat in the US (a pitched roof would show its gable to the neighbour); old European towns
            # are gabled (the ridge on the long axis: a narrow deep house shows its gable to the street)
            low = (nf or (h or 0) / FLOOR_M or 3) <= 6
            roof = "gabled" if (style.attached and low and not industrial and not shop
                                and _hash01(seed, "attached roof") < style.attached) else "flat"
        elif barn:
            roof = "gabled"
        elif shed and area < 80:
            roof = "gabled" if seed > 0.4 else "flat"
        elif detached_house:
            if style.flat and _hash01(seed, "flat roof") < style.flat:
                roof = "flat"
            else:
                roof = "gabled" if seed < 1.0 - style.hipped else "hipped"
        else:
            roof = "flat"
    # pitched roofs only on near-rectangular footprints (a church's nave keeps its gable beside a tower or an apse)
    if roof != "flat" and (box_fill < (0.6 if church else 0.8) or hw < 2.0
                           or hw > (BARREL_MAX_HW if roof == "barrel" else 18.0 if barn else 20.0 if church else 14.0)):
        roof = "flat"
    # a city tower's ground floor is shops or a lobby (Midtown, 2026-10-14: most towers have no class, and their
    # curtain wall came down to the pavement)
    tall = (nf or (h or 0) / FLOOR_M) >= TOWER_SHOP_FLOORS
    if dense and tall and cls not in HOUSE_CLASSES and not (shed or barn or garage or terminal or church):
        shop = True
    if church:
        shop = False                                # (no shopfronts on a church, Cluj 2026-10-15)
    if h is None:
        h = nf * floor_h + (GROUND_FLOOR_SHOP_M - floor_h if shop and nf > 1 else 0.0) + (0.6 if roof == "flat" else 0.2)
        if west_town and roof == "flat" and nf == 1 and area >= 80 and not (shed or barn):
            h += FALSE_FRONT_M                      # a main-street shop's tall parapet
    else:
        if attrs.get("roof_height") and np.isfinite(attrs["roof_height"]) and roof != "flat":
            h = max(2.5, h - float(attrs["roof_height"]))
        elif roof == "barrel":
            h = max(2.5, h - hw * BARREL_RISE)
        elif roof != "flat":
            h = max(2.5, h - hw * math.tan(math.radians(style.pitch or PITCH[roof])))
        if nf is None:
            # a church: one tall storey (Cluj's St. Michael's, 25 m, came out as three 8 m floors)
            nf = 1 if church else max(1, int(round(h / floor_h)))
    nf = max(1, min(nf, 120))
    floor_h = h / nf
    if shed or barn:
        facade, band, plain = SIDING, None, True
    else:
        # glass curtain walls on towers; the mid-rises of old city cores (and a share of the older towers) are
        # masonry with punched windows (Midtown's street photos, 2026-10-14: the theatre district's 5-15 storey
        # brick and stone buildings had all come out as glass towers)
        u_glass = _hash01(seed, "glass")
        # a building drawn as its parts: the style of the whole building on every part (NYC 2026-10-14: a brick
        # part on a glass tower, each part decided by its own height)
        bh = attrs.get("building_height")
        snf = max(nf, int(bh / FLOOR_M)) if bh is not None and np.isfinite(bh) else nf
        modern = cls in ("office", "commercial", "hotel") and u_glass < MIDRISE_GLASS_OFFICE
        tall = snf >= GLASS_FLOORS[1] or (snf >= GLASS_FLOORS[0] and (modern or u_glass < OLD_TOWER_GLASS)) \
            or (snf >= 7 and modern)
        if snf >= 7 and not tall and not industrial:
            if _hash01(seed, "brick") < brick_share(lat, lon) * 1.2:
                facade, band = BRICK_TINTS[int(seed * 7919) % len(BRICK_TINTS)], BAND_BRICK
            else:
                facade = FACADE_TINTS[int(seed * 7919) % len(FACADE_TINTS)]
                band = BAND_TALL if seed < 0.6 else BAND_SASH
        elif tall or (industrial and not shed):
            facade = f"{OFFICE}|{int(_hash01(seed, 'glass tint') * len(WALL_TINTS[OFFICE]))}" if tall \
                else FACADE_TINTS[3]
            band = (BAND_OFFICE if seed < 0.5 else BAND_OFFICE2) if tall else BAND_PANEL
        else:
            u_brick = _hash01(seed, "brick")
            detached = (cls in HOUSE_CLASSES or (area < 260 and not shop)) and not row and neighbours == 0
            brick = u_brick < brick_share(lat, lon) * (1.15 if row else 0.35 if detached else 1.0)
            if brick:
                band = BAND_BRICK
                facade = BRICK_TINTS[int(seed * 7919) % len(BRICK_TINTS)]
            else:
                facade = FACADE_TINTS[int(seed * 7919) % len(FACADE_TINTS)]
                if (row or cls in HOUSE_CLASSES) and _hash01(seed, "paint") < style.paint:
                    facade = PAINT_TINTS[int(_hash01(seed, "pc") * len(PAINT_TINTS))]
                if row or cls in HOUSE_CLASSES or roof != "flat":
                    band = BAND_SASH if seed < 0.7 else BAND_PANEL
                else:
                    band = (BAND_SASH, BAND_TALL, BAND_PANEL)[int(seed * 104729) % 3]
        plain = False
    wall = None
    house = (cls in HOUSE_CLASSES or (area < 260 and not shop and not industrial)) and not row and not shed \
        and neighbours == 0 and area < 300         # a long "residential" block is not a house (Fells Point)
    if house and band is not BAND_BRICK and nf <= 3:
        # detached houses: siding in the East and South, stucco in the West; windows laid over the wall
        u = _hash01(seed, "wall")
        kind, acc = None, 0.0
        for k, share in style.walls:            # (the US East: siding 75 %; the West: stucco 65 %, siding 25 %)
            acc += share
            if u < acc:
                kind = k
                break
        if kind is not None:
            tints = style.wall_tints
            if style.wood and kind == SIDING:   # chalets: stained wood
                wall = f"{SIDING}|w{int(_hash01(seed, 'wc') * len(WOOD_TINTS))}"
            elif tints is not None:
                tints = [t for t in tints if t < len(WALL_TINTS[kind]) - (BARN_REDS if kind == SIDING else 0)] or [0]
                wall = f"{kind}|{tints[int(_hash01(seed, 'wc') * len(tints))]}"
            else:
                n = len(WALL_TINTS[kind]) - (BARN_REDS if kind == SIDING else 0)     # houses: not barn red
                n = min(n, US_WALL_TINTS.get(kind, n))
                wall = f"{kind}|{int(_hash01(seed, 'wc') * n)}"
            facade = wall
    # a hut above the tree line, alone: stone walls (alpine: the tile is nearly empty and high, tile_buildings)
    stone_hut = wall is None and alpine and area < ALPINE_HUT_M2 and neighbours == 0
    if stone_hut:                       # (sheds too: a summit shelter is shed-sized)
        wall = f"{STONE}|{int(_hash01(seed, 'stc') * len(WALL_TINTS[STONE]))}"
        facade = wall
    # dressed stone: a regional share of low older buildings, attached ones too (their plaster sheet otherwise)
    if wall is None and style.stone and nf <= STONE_FLOORS and band is not BAND_BRICK \
            and not (shop or industrial or terminal or garage or shed or barn or church) \
            and cls not in ("apartments", "office", "commercial", "hotel", "retail") \
            and _hash01(seed, "stone") < style.stone:
        wall = f"{STONE}|{int(_hash01(seed, 'stc') * len(WALL_TINTS[STONE]))}"
        facade = wall
    # the mapped facade material (OSM building:material, Overture facade_material) over the guesses (Tower Bridge's
    # stone towers came out as glass towers, 2026-10-18)
    fm = str(attrs.get("facade_material") or "").lower()
    if fm in STONE_FACADES and not (barn or shed):
        wall = f"{STONE}|{int(_hash01(seed, 'stc') * len(WALL_TINTS[STONE]))}"
        facade, plain = wall, False
    elif fm == "brick" and not (barn or shed):
        facade, band, plain, wall = BRICK_TINTS[int(seed * 7919) % len(BRICK_TINTS)], BAND_BRICK, False, None
    elif fm == "glass":
        facade = f"{OFFICE}|{int(_hash01(seed, 'glass tint') * len(WALL_TINTS[OFFICE]))}"
        band, plain, wall = BAND_OFFICE, False, None
    elif fm in ("wood", "timber", "log", "wood_boards") and not (barn or shed):
        wall = f"{SIDING}|w{int(_hash01(seed, 'wc') * len(WOOD_TINTS))}"
        facade, plain = wall, False
    elif fm in ("concrete", "cement", "precast") and not (barn or shed):
        wall = f"{CONCRETE}|0"
        facade, plain = wall, False
    elif fm in ("plaster", "stucco", "render") and not (barn or shed):
        wall = f"{STUCCO}|{int(_hash01(seed, 'wc') * len(WALL_TINTS[STUCCO]))}"
        facade = wall
    elif fm in ("metal", "steel", "corrugated_iron", "aluminium") and not (barn or shed):
        wall = f"{METALWALL}|{int(_hash01(seed, 'wc') * len(WALL_TINTS[METALWALL]))}"
        facade, plain = wall, False
    fc = parse_color(attrs.get("facade_color"))
    if fc is not None and not attrs.get("bridge_part"):
        if wall is not None:
            base = wall.split("|")[0]
            wall = custom_look(base, fc)
            facade = wall
        elif fm == "brick":
            facade = custom_look(BRICK_WALL, fc)
            wall = facade
        elif fm == "glass":
            facade = custom_look(OFFICE, fc)
        else:
            facade = custom_look(FACADE, fc)
    if terminal:
        facade, band, plain = OFFICE, BAND_OFFICE, False
    elif garage:
        facade, band, plain = FACADE_TINTS[3], BAND_PANEL, False
    windows = True
    if not (shed or barn or row or house or terminal or garage or church) and nf <= 2 and roof == "flat" \
            and (area >= BIG_BOX_M2 or industrial) \
            and cls not in HOUSE_CLASSES and cls != "apartments":
        kind = METALWALL if industrial else PANEL
        wall = f"{kind}|{int(_hash01(seed, 'wc') * len(WALL_TINTS[kind]))}"
        facade, windows = wall, False
    rmat = str(attrs.get("roof_material") or "").lower()
    if rmat in ("metal", "tin", "steel", "zinc", "corrugated_iron", "sheet"):
        roof_mat = f"{ROOF_METAL}|{int(_hash01(seed, 'roof') * len(ROOF_TINTS[ROOF_METAL]))}"
    elif rmat == "copper":
        roof_mat = custom_look(ROOF_METAL, COLOR_NAMES["copper"])
    elif rmat in ("tile", "tiles", "roof_tiles", "clay", "terracotta"):
        roof_mat = f"{ROOF_SLATE}|{4 + int(_hash01(seed, 'roof') * 3)}"
    elif rmat in ("slate", "stone"):
        roof_mat = f"{ROOF_SLATE}|{int(_hash01(seed, 'roof') * 4)}"
    elif rmat in ("shingle", "shingles", "wood"):
        roof_mat = f"{ROOF_SHINGLE}|{int(_hash01(seed, 'roof') * len(ROOF_TINTS[ROOF_SHINGLE]))}"
    elif rmat in ("concrete", "cement", "tar_paper", "bitumen", "asphalt", "felt", "gravel"):
        roof_mat = f"{ROOF_FLAT}|{int(_hash01(seed, 'roof') * len(ROOF_TINTS[ROOF_FLAT]))}"
    elif rmat == "glass":
        roof_mat = f"{OFFICE}|0"
    elif roof == "flat" or barn or shed:
        roof_mat = ROOF_FLAT if roof == "flat" else ROOF_METAL
        roof_mat = f"{roof_mat}|{int(_hash01(seed, 'roof') * len(ROOF_TINTS[roof_mat]))}"
    else:       # the region's pitched roofs (the US: slate or shingle by the seed, their first four tints)
        mat, tints = style.roofs[min(int(seed * len(style.roofs)), len(style.roofs) - 1)]
        roof_mat = f"{mat}|{tints[int(_hash01(seed, 'roof') * len(tints))]}"
    rc = parse_color(attrs.get("roof_color"))
    if rc is not None:
        base_rmat = roof_mat.split("|")[0]
        roof_mat = custom_look(base_rmat, rc)
    if (barn or shed) and not stone_hut:    # plain walls tinted: barns the region's barn red or wood, sheds the siding colours
        tints = WALL_TINTS[SIDING]
        if barn and _hash01(seed, "barn") < style.barn_red:
            facade = f"{SIDING}|{len(tints) - 1 - int(_hash01(seed, 'red') * 2)}"
        elif style.wood:
            facade = f"{SIDING}|w{int(_hash01(seed, 'wc') * len(WOOD_TINTS))}"
        else:
            facade = f"{SIDING}|{int(_hash01(seed, 'sc') * (len(tints) - 2))}"
    # a front porch (the photos of Lancaster, Ocean City, Frederick, Moab: most houses have one): pitched-roof
    # houses, a regional share; doors() puts it on the front door's wall
    porch = (house and roof != "flat" and not barn and not shed and area < 300
             and _hash01(seed, "porch") < style.porch)
    if porch:   # pitched roofs draw the walls on the oriented box: an L-shaped house's door wall can be inside it
        _, _, hl, hw = oriented_box(ring)
        porch = area >= 0.9 * 4 * hl * hw
    return Spec(ring=ring, floors=nf, floor_h=floor_h, height=h, roof=roof, roof_mat=roof_mat, facade=facade,
                band=band, shop=shop and not shed and nf >= 1, plain=plain, seed=seed, min_h=0.0, wall=wall,
                windows=windows, steeple=church and area < 2500, porch=porch, cls=cls,
                deck=garage and roof == "flat", pitch=style.pitch)


# ------------------------------------------------------------------------------------------- landmarks
# Footprints that are not a box with a roof, told from the outline (2026-10-28, the user's San Pedro and San Jose
# shots): the Battleship USS Iowa (OSM building=ship, historic=ship) and the Queen Mary (historic=ship) were 9-storey
# apartment blocks, the Century 21 dome theatre a windowed round block. Overture keeps neither tag (no "ship" class;
# OSM itself maps no roof:shape on the dome), so a ship is a long pointed hull on water or with a ship's name, a dome
# a round footprint that is not a house or a tower, a tank a small round one of a tank's class or without a name.
SHIP_SLENDER = 3.0          # a hull: at least this many times as long as wide ...
SHIP_MAX_SLENDER = 10.0     # ... at most this many (real hulls 5-9: the Iowa 8.2, the Queen Mary 8.6; a 90 x 6 m
                            # pontoon walkway beside the Queen Mary is not one) ...
SHIP_MIN_L = 25.0           # ... at least this long (m) ...
SHIP_FILL = (0.5, 0.92)     # ... filling this share of its oriented box (pointed or rounded ends) ...
SHIP_END_SHARE = 0.65       # ... one end at most this share of the beam wide over its last SHIP_END_L of length
SHIP_END_L = 0.06
SHIP_NAME = re.compile(r"\b(uss|ss|hms|rms|mv|ms|uscgc|battleship|ship|schooner|steamer|steamship|lightship|"
                       r"submarine|frigate|destroyer|cruiser|aircraft carrier|liner|queen mary|ferry|sailing)\b")
NAVY_NAME = re.compile(r"\b(uss|hms|uscgc|battleship|submarine|frigate|destroyer|cruiser|carrier)\b")
SHIP_FREEBOARD = (3.0, 12.0)    # m: the deck over the water (0.6 of the half beam, in this range)
SHIP_TIER_M = 3.2           # m per deck of the superstructure
SHIP_PART_BIG_M = 10.0      # m: a ship's mapped part this long (half) is superstructure; smaller ones turrets, guns
SHIP_TURRET_M = 3.5         # m: a turret's height over the deck
SHIP_TOP_SHARE = 0.15       # a ship over 60 m: its mast tops this share of its length over the water
SHIP_SHEER = (1.35, 1.1)    # an unknown ship's deck at the bow and the stern, of its freeboard amidships
SHIP_STATIONS = 48          # hull sections along a ship (lite: SHIP_STATIONS_LITE)
SHIP_STATIONS_LITE = 16
SHIP_KEEL_SHARE = 0.5       # the hull's width at its keel (under the water, out of sight), of the waterline's
PIER_DECK_M = 1.5           # m: a quay structure alongside a ship (pier_rings): a low deck
PIER_TOUCH_M = 1.0          # m: its outline this near the ship's hull somewhere ...
PIER_GAP_M = 12.0           # ... its centre at most this past the ship's side and its own half width ...
PIER_SLENDER = 3.0          # ... and at least this many times as long as wide
CIRCLE_SEGS = 32            # a dome's or a tank's circle drawn with this many sides
DOME_ANNEX_M = 5.0          # m: the rest of a dome's outline (a lobby, an entrance) one plain storey
ROUND_MIN_VERTS = 10        # a round footprint (circle_fit) is drawn with at least this many corners ...
ROUND_TOLS = (0.35, 0.2, 0.12, 0.12, 0.12)  # circle_fit: corners this share of the radius off the circle are on it
                            # (the refits closing in) ...
ROUND_SHARE_MIN = 0.35      # ... this share of the corners at least (the cruise terminal's dome: 0.46, its annex the
                            # rest) ...
ROUND_ARC = 0.7             # ... going this share of the way round ...
ROUND_GAP_DEG = 120.0       # ... with no wider gap round the centre (the cruise terminal's annex: 86) ...
ROUND_DISC_SHARE = 0.6      # ... the disc this share of the footprint at least ...
ROUND_TURN_DEG = 55.0       # ... its round corners turning at most this ...
ROUND_GENTLE = 0.55         # ... this share of them at least
HOLE_SHARE = 0.03           # holes this share of the outline: a ring round a courtyard (Apple Park), not a dome
ROUND_INSIDE = 0.9          # ... and this share of it inside the footprint
DOME_MIN_D = 18.0           # m: a round building this wide is a dome (smaller: a tank or a kiosk) ...
DOME_MAX_SLENDER = 0.9      # ... unless taller than this many times its width (a round tower) ...
DOME_MAX_D = 230.0          # ... or wider than this (the Superdome 210 m)
ROUND_NOT_CLASSES = {"roof", "carport", "shelter", "garage", "garages", "parking", "parking_garage", "bridge_structure",
                     "bridge_tower"}
DOME_DRUM_SHARE = 0.35      # of the height: the drum under the cap
TANK_CLASSES = {"storage_tank", "silo", "grain_silo", "slurry_tank", "digester", "tank", "gasometer"}
TANK_MAX_D = 90.0
ROUND_HOME_CLASSES = HOUSE_CLASSES | {"apartments", "terrace", "dormitory", "hotel"}
ARCH_SEGS = 10              # segments of a barrel roof's arch / a dome's quarter
VAULT_RISE_SHARE = 0.3      # a hangar this tall for its span (mapped height / width) is arched (landmark_kind) ...
VAULT_MIN_SPAN = 20.0       # ... spanning at least this (m) ...
VAULT_MAX_HW = 70.0         # ... and at most twice this ...
VAULT_FILL = 0.75           # ... a rectangle or a stadium (this share of its oriented box at least)
VAULT_EAVE_M = 3.0          # m: the walls under its arch (an airship hangar's arch springs from near the ground)
VAULT_RISE_HW = 1.3         # its arch at most this many half spans high (Hangar One: 59 m over a 46 m half span)
ROUNDED_END_SHARE = 0.85    # _rounded_ends: an end at most this share of the width over its last quarter half width


def _open_ring(ring):
    """The ring without a repeated last corner, counter-clockwise."""
    r = np.asarray(ring, np.float64)
    if len(r) > 1 and np.allclose(r[0], r[-1]):
        r = r[:-1]
    return r if _area2(r) >= 0 else r[::-1]


def _arc_seed(r):
    """The corners on the best circle through three corners of the ring (each corner and the ones k and 2k on, the
    circle with most corners within 4 % of it): circle_fit's first fit, which an annex's corners would otherwise pull
    off the round part (the cruise terminal's dome came out 80 m round for 63)."""
    n = len(r)
    span = float(np.ptp(r, 0).max())
    best, best_n = r, 0
    idx = np.arange(n)
    for k in (1, 2, 3, max(4, n // 8)):
        a, b, c = r[idx], r[(idx + k) % n], r[(idx + 2 * k) % n]
        d = 2 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
        ok = np.abs(d) > 1e-9
        if not ok.any():
            continue
        a2, b2, c2 = (a * a).sum(1), (b * b).sum(1), (c * c).sum(1)
        ux = (a2 * (b[:, 1] - c[:, 1]) + b2 * (c[:, 1] - a[:, 1]) + c2 * (a[:, 1] - b[:, 1])) / np.where(ok, d, 1)
        uy = (a2 * (c[:, 0] - b[:, 0]) + b2 * (a[:, 0] - c[:, 0]) + c2 * (b[:, 0] - a[:, 0])) / np.where(ok, d, 1)
        cen = np.column_stack([ux, uy])
        R = np.hypot(*(a - cen).T)
        dist = np.hypot(r[None, :, 0] - cen[:, None, 0], r[None, :, 1] - cen[:, None, 1])
        ok &= R <= 0.55 * span                  # (three corners on a straight wall: a huge circle through all)
        cnt = (np.abs(dist - R[:, None]) <= 0.04 * R[:, None]).sum(1) * ok
        i = int(np.argmax(cnt))
        if cnt[i] > best_n:
            best_n, best = int(cnt[i]), r[np.abs(dist[i] - R[i]) <= 0.04 * R[i]]
    return best if best_n >= ROUND_MIN_VERTS else r


def circle_fit(ring):
    """(centre (2,), radius) of the round part of a footprint, or None: a least-squares circle through its corners,
    refitted on the corners near it with the tolerance closing in (ROUND_TOLS: so an annex does not pull it: the Long
    Beach cruise terminal's dome carries a terminal block over a quarter of its rim), its corners on the circle going
    ROUND_ARC of the way round with no gap over ROUND_GAP_DEG, the disc most of the footprint and itself almost all
    inside it (a curved wall of a bigger building is not a round building)."""
    r = _open_ring(ring)
    if len(r) < ROUND_MIN_VERTS:
        return None
    pts = _arc_seed(r)
    for tol in ROUND_TOLS:
        A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
        try:
            sol = np.linalg.lstsq(A, (pts ** 2).sum(1), rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        c = sol[:2]
        R = math.sqrt(max(sol[2] + c @ c, 0.0))
        if R <= 0:
            return None
        d = np.hypot(*(r - c).T)
        keep = np.abs(d - R) <= tol * R
        if keep.sum() < ROUND_MIN_VERTS:
            return None
        pts = r[keep]
    ang = np.sort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * math.pi]))
    # the arc the round corners cover: gaps of a plain polygon's step count, a gap far over the usual one (an annex
    # in the way) does not
    step = float(np.median(gaps))
    arc = float(gaps[gaps <= max(2.5 * step, math.radians(40.0))].sum()) / (2 * math.pi)
    area = _area2(r) / 2
    # the round corners turn gently (a polygon's few tens of degrees; a jagged house outline whose corners happen
    # to lie near a circle turns 90 at each: 83 of 672 "tanks" in a 700-cell census were such houses)
    e_in, e_out = r - np.roll(r, 1, 0), np.roll(r, -1, 0) - r
    turn = np.abs(np.arctan2(e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0], (e_in * e_out).sum(1)))
    if float((turn[keep] <= math.radians(ROUND_TURN_DEG)).mean()) < ROUND_GENTLE:
        return None
    if keep.mean() < ROUND_SHARE_MIN or arc < ROUND_ARC or math.degrees(gaps.max()) > ROUND_GAP_DEG \
            or math.pi * R * R < ROUND_DISC_SHARE * area:
        return None
    from polygon import Path
    t = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    probe = np.concatenate([c[None], c + 0.5 * R * np.column_stack([np.cos(t), np.sin(t)]),
                            c + 0.85 * R * np.column_stack([np.cos(t + 0.13), np.sin(t + 0.13)])])
    if Path(r).contains_points(probe).mean() < ROUND_INSIDE:
        return None
    return c, R


def roundness(ring):
    """4 pi A / P^2 of the footprint (1 a circle, a square 0.785)."""
    r = _open_ring(ring)
    per = float(np.hypot(*(np.roll(r, -1, 0) - r).T).sum())
    return 4 * math.pi * (_area2(r) / 2) / max(per * per, 1e-9)


def _end_widths(ring, c, u, hl):
    """The footprint's width across u over its last SHIP_END_L of length at each end (sampled along its edges)."""
    r = _open_ring(ring)
    v = np.array([-u[1], u[0]])
    t = np.linspace(0, 1, 6)[:, None, None]
    pts = (r[None] + (np.roll(r, -1, 0) - r)[None] * t).reshape(-1, 2) - c
    s, d = pts @ u, pts @ v
    out = []
    for sgn in (-1, 1):
        k = (sgn * s) >= hl * (1 - 2 * SHIP_END_L)
        out.append(float(d[k].max() - d[k].min()) if k.any() else 0.0)
    return out


def _rounded_ends(r, c, u, hl, hw):
    """Whether both ends of a footprint are rounded (a stadium: Hangar One): its width over the last quarter of the
    half width at each end at most ROUNDED_END_SHARE of the full width (a square end: all of it)."""
    v = np.array([-u[1], u[0]])
    t = np.linspace(0, 1, 8)[:, None, None]
    pts = (r[None] + (np.roll(r, -1, 0) - r)[None] * t).reshape(-1, 2) - c
    s, d = pts @ u, pts @ v
    for sgn in (-1, 1):
        k = (sgn * s) >= hl - 0.25 * hw
        if not k.any() or float(d[k].max() - d[k].min()) > ROUNDED_END_SHARE * 2 * hw:
            return False
    return True


def landmark_kind(ring, attrs, on_water=False, ml=False):
    """"ship", "dome", "tank", "vault" (an arched hangar) or None for a footprint (local ring) and its attributes (class, name, height,
    roof_shape). Machine-learned outlines are never landmarks (their round and pointed blobs are trees and boats)."""
    if ml:
        return None
    rs = str(attrs.get("roof_shape") or "").lower()
    cls = str(attrs.get("class") or "").lower()
    name = str(attrs.get("name") or "").lower()
    r = _open_ring(ring)
    area = _area2(r) / 2
    c, u, hl, hw = oriented_box(r)
    if 2 * hl >= SHIP_MIN_L and SHIP_SLENDER * hw <= hl <= SHIP_MAX_SLENDER * hw and cls in ("", "yes", "ship", "museum") \
            and SHIP_FILL[0] <= area / max(4 * hl * hw, 1e-6) <= SHIP_FILL[1] \
            and (SHIP_NAME.search(name) or (on_water() if callable(on_water) else on_water)) \
            and min(_end_widths(r, c, u, hl)) <= SHIP_END_SHARE * 2 * hw:
        return "ship"
    if rs in ("dome", "onion"):
        return "dome"
    h = attrs.get("height")
    h = float(h) if h is not None and np.isfinite(h) and h > 0 else None
    hangar = cls == "hangar" or "hangar" in name
    if (hangar or rs in ("round", "barrel", "arched")) and VAULT_MIN_SPAN <= 2 * hw <= 2 * VAULT_MAX_HW             and hl >= hw and area / max(4 * hl * hw, 1e-6) >= VAULT_FILL:
        # an arched hangar: tall for its span (Hangar One 59 m over 93, Hangar 2 54 over 91; a flat-roofed modern
        # hangar 8 m over 78), or arched by its tag with rounded ends (a straight-ended one keeps plan's barrel)
        rounded = _rounded_ends(r, c, u, hl, hw)
        if (hangar and h is not None and h >= VAULT_RISE_SHARE * 2 * hw) or (not hangar and rounded):
            return "vault"
    if (attrs.get("holed") or 0.0) >= HOLE_SHARE or cls in ROUND_NOT_CLASSES:
        return None                 # a ring round a courtyard (Apple Park), a canopy
    fit = circle_fit(r)
    if fit is None:
        return None
    d = 2 * fit[1]
    if cls in TANK_CLASSES or (not name and cls in ("", "yes", "industrial") and (h is None or h <= 1.5 * d)):
        # (unnamed round buildings without a class are tanks: OSM building=yes + man_made=storage_tank, Overture
        # keeping neither; Cupertino's 40-70 m ones came out as domes)
        return "tank" if d <= TANK_MAX_D else None
    if d < DOME_MIN_D or d > DOME_MAX_D or cls in ROUND_HOME_CLASSES or (h is not None and h > DOME_MAX_SLENDER * d):
        return None
    return "dome"


def _ring_points(r, step=2.0):
    """Points every ~step m round a closed ring."""
    out = []
    for a, b in zip(r, np.roll(r, -1, 0)):
        k = max(1, int(math.ceil(float(np.hypot(*(b - a))) / step)))
        out.append(a + (b - a) * (np.arange(k)[:, None] / k))
    return np.concatenate(out)


def pier_alongside(ring, hull):
    """Whether a footprint is a quay structure along a ship's side (local rings): long and thin (PIER_SLENDER),
    parallel to the ship, touching its hull (PIER_TOUCH_M), lying beside it (its length within the ship's, its
    centre at most PIER_GAP_M past the ship's side and its own half width). The Queen Mary's landing stage (OSM
    building=yes, a LiDAR height of 22.7 m: the ship's own flank) stood as a 7-storey block beside her."""
    r, h = _open_ring(ring), _open_ring(hull)
    c, u, hl, hw = oriented_box(r)
    hc, hu, hhl, hhw = oriented_box(h)
    if hl < PIER_SLENDER * hw or abs(float(u @ hu)) < math.cos(math.radians(15.0)):
        return False
    hv = np.array([-hu[1], hu[0]])
    along = (r - hc) @ hu
    if along.min() < -hhl - PIER_GAP_M or along.max() > hhl + PIER_GAP_M \
            or abs(float((c - hc) @ hv)) > hhw + hw + PIER_GAP_M:
        return False
    pts = _ring_points(r)
    a, b = h, np.roll(h, -1, 0)
    ab = b - a
    t = np.clip(((pts[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0, 1)
    d = np.hypot(*(pts[:, None] - (a[None] + ab[None] * t[..., None])).transpose(2, 0, 1)).min()
    return float(d) <= PIER_TOUCH_M


def landmark_spec(kind, ring, attrs, seed):
    """plan()'s Spec for a landmark_kind footprint (cls the kind; build() draws it: _ship, _dome, _tank)."""
    r = _open_ring(ring)
    _, _, hl, hw = oriented_box(r)
    h = attrs.get("height")
    h = float(h) if h is not None and np.isfinite(h) and h > 0 else None
    fit = circle_fit(r) if kind != "ship" else None
    if fit is None:
        fit = (oriented_box(r)[0], hw)          # (a dome by its tag on an outline that is not round)
    d = 2 * fit[1]
    if kind == "ship":
        import shipprofiles
        navy = bool(NAVY_NAME.search(str(attrs.get("name") or "").lower()))
        prof = shipprofiles.profile(attrs.get("name"), navy)
        navy = prof["navy"]
        if prof.get("L"):
            # a known ship: its own freeboard and its masts' and funnels' tops, scaled to its mapped length
            k = 2 * hl / prof["L"]
            fb = prof["freeboard"][1] * k
            h = k * max([z for _, z in prof["masts"]] + [f[3] for f in prof["funnels"]]
                        + ([prof["island"][3]] if prof.get("island") else []))
        else:
            fb = float(np.clip(0.6 * hw, *SHIP_FREEBOARD))
            # the mapped height of a ship is its hull's or a guess (the Iowa 17.4 m, the Queen Mary 10 m); its masts
            # and funnels stand about SHIP_TOP_SHARE of its length over the water
            h = max(h or 0.0, fb + 3 * SHIP_TIER_M, SHIP_TOP_SHARE * 2 * hl if 2 * hl >= 60 else 0.0)
        hull, upper = (f"{STEEL}|grey", f"{STEEL}|grey") if navy else (f"{STEEL}|black", f"{STEEL}|white")
        # (floor_h: the freeboard amidships; wall: the superstructure's paint; circle: its profile and mapped parts)
        return Spec(ring=r, floors=1, floor_h=fb, height=h, roof="flat", roof_mat=f"{ROOF_METAL}|1", facade=hull,
                    wall=upper, band=None, shop=False, plain=True, seed=seed, min_h=0.0, windows=False,
                    steeple=False, porch=False, cls="ship", deck=False, pitch=None,
                    circle={"profile": prof, "parts": attrs.get("ship_parts")})
    if kind == "vault":
        c_, u_, hl_, hw_ = oriented_box(r)
        h = h or 2 * hw_ * VAULT_RISE_SHARE
        eave = max(VAULT_EAVE_M, h - VAULT_RISE_HW * hw_)
        return Spec(ring=r, floors=1, floor_h=eave, height=h, roof="flat", roof_mat=f"{ROOF_METAL}|1",
                    facade=f"{STEEL}|grey", band=None, shop=False, plain=True, seed=seed, min_h=0.0, wall=None,
                    windows=False, steeple=False, porch=False, cls="vault", deck=False, pitch=None,
                    circle=_rounded_ends(r, c_, u_, hl_, hw_))
    if kind == "pier":
        # a quay structure alongside a moored ship (pier_rings): a low concrete deck, whatever height its scan got
        return Spec(ring=r, floors=1, floor_h=PIER_DECK_M, height=PIER_DECK_M, roof="flat", roof_mat="bne_precast",
                    facade="bne_precast", band=None, shop=False, plain=True, seed=seed, min_h=0.0, wall=None,
                    windows=False, steeple=False, porch=False, cls="pier", deck=False, pitch=None)
    if kind == "tank":
        h = h or float(np.clip(0.6 * d, 6.0, 20.0))
        return Spec(ring=r, floors=1, floor_h=h, height=h, roof="flat", roof_mat=f"{STEEL}|white",
                    facade=f"{STEEL}|white", band=None, shop=False, plain=True, seed=seed, min_h=0.0, wall=None,
                    windows=False, steeple=False, porch=False, cls="tank", deck=False, pitch=None, circle=fit)
    h = h or float(np.clip(0.4 * d, 8.0, 30.0))
    return Spec(ring=r, floors=1, floor_h=h, height=h, roof="flat", roof_mat=f"{ROOF_FLAT}|0",
                facade=f"{PANEL}|0", band=None, shop=False, plain=True, seed=seed, min_h=0.0, wall=None,
                windows=False, steeple=False, porch=False, cls="dome", deck=False, pitch=None, circle=fit)


LANDMARK_CLASSES = {"ship", "dome", "tank", "vault"}     # drawn by their own (build: _ship, _dome, _tank, _vault)
LANDMARK_KINDS = LANDMARK_CLASSES | {"pier"}            # landmark_spec's (a pier: a plain low box)


def _plain_walls(m, ring, z0, z1, mat, uv_m=SIDING_UV_M):
    """Plain walls round a ring, facing out."""
    r = _open_ring(ring)
    s = 0.0
    for i in range(len(r)):
        a, b = r[i], r[(i + 1) % len(r)]
        L = float(np.hypot(*(b - a)))
        m.quad((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1),
               (s / uv_m, z0 / uv_m), ((s + L) / uv_m, z0 / uv_m), ((s + L) / uv_m, z1 / uv_m), (s / uv_m, z1 / uv_m),
               mat)
        s += L


def _cap(m, ring, z0, rise, mat, segs=ARCH_SEGS, top_share=0.0):
    """A dome over a ring from z0: each corner drawn toward the centre on a quarter ellipse rise high, closed at the
    top (top_share > 0: cut flat at that share of the width, a tank's low cone)."""
    r = _open_ring(ring)
    c = r.mean(0)
    th_max = math.acos(top_share) if top_share > 0 else math.pi / 2
    prev = None
    for k in range(segs + 1):
        th = th_max * k / segs
        f, z = math.cos(th), z0 + rise * math.sin(th) / math.sin(th_max)
        cur = [(float(c[0] + (p[0] - c[0]) * f), float(c[1] + (p[1] - c[1]) * f), z) for p in r]
        if prev is not None:
            for i in range(len(r)):
                j = (i + 1) % len(r)
                m.quad(prev[i], prev[j], cur[j], cur[i], (i / 4.0, k / 4.0), (j / 4.0, k / 4.0),
                       (j / 4.0, (k + 1) / 4.0), (i / 4.0, (k + 1) / 4.0), mat)
        prev = cur
    m.poly(prev, [(p[0] / ROOF_UV_M, p[1] / ROOF_UV_M) for p in prev], mat)


def _circle(spec, segs=CIRCLE_SEGS):
    c, R = spec.circle
    t = np.linspace(0, 2 * math.pi, segs, endpoint=False)
    return np.column_stack([c[0] + R * np.cos(t), c[1] + R * np.sin(t)])


def _dome(m, spec, gmin, gmax, mat_idx, lite=False):
    """A round hall: the outline one plain storey (its lobby: a flat roof round the drum), a plain drum on its fitted
    circle (circle_fit) and a dome over it to the mapped height."""
    z0, zb = gmin - (SKIRT_LITE_M if lite else SKIRT_M), base_z(spec, gmin, gmax)
    drum = DOME_DRUM_SHARE * spec.height
    annex = min(DOME_ANNEX_M, drum)
    circ = _circle(spec, 16 if lite else CIRCLE_SEGS)
    r = _open_ring(spec.ring)
    _plain_walls(m, r, z0, zb + annex, mat_idx[spec.facade])
    m.poly([(p[0], p[1], zb + annex) for p in r], [(p[0] / ROOF_UV_M, p[1] / ROOF_UV_M) for p in r],
           mat_idx[f"{ROOF_FLAT}|1"])
    _plain_walls(m, circ, zb + annex - 0.3, zb + drum, mat_idx[spec.facade])
    _cap(m, circ, zb + drum, spec.height - drum, mat_idx[spec.roof_mat], 4 if lite else ARCH_SEGS)


def _tank(m, spec, gmin, gmax, mat_idx, lite=False):
    """A storage tank: a plain cylinder on its fitted circle under a low cone."""
    z0, zb = gmin - (SKIRT_LITE_M if lite else SKIRT_M), base_z(spec, gmin, gmax)
    circ = _circle(spec, 12 if lite else CIRCLE_SEGS)
    _plain_walls(m, circ, z0, zb + spec.height, mat_idx[spec.facade])
    _cap(m, circ, zb + spec.height, min(0.16 * spec.circle[1], 2.5), mat_idx[spec.roof_mat], 2, top_share=0.15)


_UV0 = ((0.002, 0.995), (0.012, 0.995), (0.012, 0.999), (0.002, 0.999))


def _quad_out(m, a, b, c, d, hint, mat):
    """A quad facing hint (a direction: its corners reordered when they face away)."""
    n = np.cross(np.subtract(b, a), np.subtract(c, a))
    if float(np.dot(n, hint)) < 0:
        a, b, c, d = a, d, c, b
    m.quad(a, b, c, d, *_UV0, mat)


def _cylinder(m, c, u, rx, ry, z0, z1, mat, segs=12, top=True):
    """An upright elliptic cylinder (rx along u, ry across) from z0 to z1, its top closed."""
    v = np.array([-u[1], u[0]])
    t = np.linspace(0, 2 * math.pi, segs, endpoint=False)
    ring = np.array([c + u * rx * math.cos(a) + v * ry * math.sin(a) for a in t])
    if _area2(ring) < 0:
        ring = ring[::-1]
    _plain_walls(m, ring, z0, z1, mat)
    if top:
        m.poly([(p[0], p[1], z1) for p in ring], [(p[0] / ROOF_UV_M, p[1] / ROOF_UV_M) for p in ring], mat)


def _ship_axis(r):
    """(centre, unit vector to the bow, half length, half beam) of a ship's footprint: the bow its narrower end."""
    c, u, hl, hw = oriented_box(r)
    ew = _end_widths(r, c, u, hl)
    return c, (-u if ew[0] < ew[1] else u), hl, hw


def _hull_widths(r, c, ub, hl, xs):
    """The footprint's (centre offset, half width) across the ship at stations xs (0 the bow, 1 the stern)."""
    v = np.array([-ub[1], ub[0]])
    a, b = r - c, np.roll(r, -1, 0) - c
    sa, sb, ta, tb = a @ ub, b @ ub, a @ v, b @ v
    out = []
    for x in xs:
        sx = hl * (1 - 2 * float(np.clip(x, 0.003, 0.997)))
        k = ((sa - sx) * (sb - sx) <= 0) & (sa != sb)
        if not k.any():
            out.append((0.0, 0.0))
            continue
        t = ta[k] + (tb[k] - ta[k]) * (sx - sa[k]) / (sb[k] - sa[k])
        out.append(((t.max() + t.min()) / 2, (t.max() - t.min()) / 2))
    return np.array(out)


def _ship(m, spec, gmin, gmax, mat_idx, lite=False):
    """A moored ship on its footprint (the plan at its deck: its length, beam, which end is the bow), drawn from
    spec.circle's profile (shipprofiles: a known ship's own, else a warship's or a liner's): a hull of sections that
    narrows to the waterline and the keel, its deck rising toward the bow (the sheer), the stem raked back under the
    water; on the deck the profile's superstructure, funnels (a liner's red with black tops), turrets and guns, a
    carrier's island, masts. An unknown ship with mapped parts (spec.circle["parts"]: [(local ring, height or None)])
    gets its mapped funnels and superstructure where they are."""
    r = _open_ring(spec.ring)
    c, ub, hl, hw = _ship_axis(r)
    v = np.array([-ub[1], ub[0]])                      # to port (the left, facing the bow)
    info = spec.circle if isinstance(spec.circle, dict) else {"parts": spec.circle}
    prof = info.get("profile") or {}
    parts = info.get("parts") or []
    known = bool(prof.get("L"))
    k = 2 * hl / prof["L"] if known else 1.0
    zw = base_z(spec, gmin, gmax)
    zk = gmin - (SKIRT_LITE_M if lite else SKIRT_M)
    hull, upper = mat_idx[spec.facade], mat_idx[spec.wall]
    navy = spec.facade == f"{STEEL}|grey"
    paints = {"upper": upper, "hull": hull, "deck": mat_idx[spec.roof_mat],
              "funnel": mat_idx[f"{STEEL}|grey" if navy else f"{STEEL}|red"]}
    black = mat_idx[f"{STEEL}|black"]
    if known:
        fb = [f * k for f in prof["freeboard"]]
    else:
        fb = [spec.floor_h * SHIP_SHEER[0], spec.floor_h, spec.floor_h * SHIP_SHEER[1]]

    def zd(x):                                          # the deck edge over the water: the sheer
        x = float(np.clip(x, 0.0, 1.0))
        return zw + (fb[1] + (fb[0] - fb[1]) * ((0.5 - x) / 0.5) ** 2 if x < 0.5
                     else fb[1] + (fb[2] - fb[1]) * ((x - 0.5) / 0.5) ** 2)

    def zabs(z):                                        # a profile's height (m over the water) on this ship
        return zw + z * k

    n = SHIP_STATIONS_LITE if lite else SHIP_STATIONS
    xs = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, n + 1))     # (closer at the ends)
    tw = _hull_widths(r, c, ub, hl, xs)
    tc, b = tw[:, 0], tw[:, 1]
    # the waterline: the deck's plan narrowed, its stem raked back from the deck's, the stern cut up a little
    rake, wl = prof.get("rake", 0.03), prof.get("waterline", 0.93)
    xq = xs - rake * np.clip(1 - xs / 0.15, 0, 1) + 0.012 * np.clip(1 - (1 - xs) / 0.1, 0, 1)
    w = wl * np.interp(xq, xs, b, left=0.0, right=0.0)
    keel = SHIP_KEEL_SHARE * w

    def P(x, t, z):
        p = c + ub * (hl * (1 - 2 * x)) + v * t
        return (float(p[0]), float(p[1]), float(z))
    # (the waterline's and the keel's stations: the stem's foot drawn back by the rake, the stern's cut up)
    xw = np.clip(xs, rake, 1 - 0.012)
    rows = [(keel, zk, xw), (w, zw, xw), (b, None, xs)]     # (half widths, z (None: the deck's), stations)
    deck_mat = paints["deck"]
    for i in range(n):
        x0, x1 = xs[i], xs[i + 1]
        for sgn in (1, -1):
            hint = np.array([*(v * sgn), 0.0])
            for (w0, z0, X0), (w1, z1, X1) in zip(rows[:-1], rows[1:]):
                a = P(X0[i], tc[i] + sgn * w0[i], z0 if z0 is not None else zd(x0))
                bb = P(X0[i + 1], tc[i + 1] + sgn * w0[i + 1], z0 if z0 is not None else zd(x1))
                cc = P(X1[i + 1], tc[i + 1] + sgn * w1[i + 1], z1 if z1 is not None else zd(x1))
                d = P(X1[i], tc[i] + sgn * w1[i], z1 if z1 is not None else zd(x0))
                _quad_out(m, a, bb, cc, d, hint, hull)
        _quad_out(m, P(x0, tc[i] - b[i], zd(x0)), P(x1, tc[i + 1] - b[i + 1], zd(x1)),
                  P(x1, tc[i + 1] + b[i + 1], zd(x1)), P(x0, tc[i] + b[i], zd(x0)), (0, 0, 1), deck_mat)
    for i, sg in ((0, 1), (n, -1)):                     # a flat stem or transom where the plan ends wide
        if b[i] > 0.3:
            hint = np.array([*(ub * sg), 0.0])
            for (w0, z0, X0), (w1, z1, X1) in zip(rows[:-1], rows[1:]):
                za = z0 if z0 is not None else zd(xs[i])
                zb_ = z1 if z1 is not None else zd(xs[i])
                _quad_out(m, P(X0[i], tc[i] - w0[i], za), P(X0[i], tc[i] + w0[i], za),
                          P(X1[i], tc[i] + w1[i], zb_), P(X1[i], tc[i] - w1[i], zb_), hint, hull)

    def half_b(x):
        return float(np.interp(x, xs, b))

    def at(x, t=0.0):
        return c + ub * (hl * (1 - 2 * x)) + v * (float(np.interp(x, xs, tc)) + t)
    beam = 2 * hw
    top = zw + spec.height
    blocks = [oriented_box(np.asarray(pr, np.float64)) for pr, ph in parts if ph is None]
    mapped_funnels = [np.asarray(pr, np.float64) for pr, ph in parts if ph is not None]
    if not known and any(phl >= SHIP_PART_BIG_M for _, _, phl, _ in blocks):
        zdm = zd(0.5)
        for pc, pu, phl, phw in blocks:     # mapped superstructure: tiers to 0.6 of the height; turrets; small guns
            if phl >= SHIP_PART_BIG_M:
                zz, L, W = zdm - 0.3, phl, phw
                while zz + SHIP_TIER_M <= zdm + 0.6 * (top - zdm) and W > 1.5:
                    _box(m, pc, pu, L, W, zz, zz + SHIP_TIER_M, upper)
                    zz, L, W = zz + SHIP_TIER_M, L * 0.9, W * 0.85
            else:
                _box(m, pc, pu, phl, phw, zdm - 0.3, zdm + (SHIP_TURRET_M if phl >= 4.0 else 2.0), upper)
    else:
        for x0, x1, wsh, z0, z1, paint in prof.get("blocks", []):
            xm = (x0 + x1) / 2
            deck = min(zd(x0), zd(x1), zd(xm)) - 0.3
            if prof.get("relative"):
                za = deck if z0 is None else zd(xm) + z0
                zb_ = zd(xm) + z1
            else:
                za = deck if z0 is None else zabs(z0)
                zb_ = zabs(z1)
            hy = min(wsh * beam, 0.98 * min(half_b(x0), half_b(x1)))
            if zb_ > za + 0.5 and hy > 0.5:
                _box(m, at(xm), ub, (x1 - x0) * hl, hy, za, zb_, paints.get(paint, upper))
    isl = prof.get("island")
    if isl:                                             # a carrier's island on its starboard edge
        x0, x1, wsh, ztop = isl
        xm = (x0 + x1) / 2
        hy = wsh * beam
        _box(m, at(xm, -(half_b(xm) - hy)), ub, (x1 - x0) * hl, hy, zd(xm) - 0.3, zabs(ztop), upper)
    if lite:
        return
    funnels = []
    if mapped_funnels and not known:
        zdm = zd(0.5)
        for pr in mapped_funnels:
            pc, pu, phl, phw = oriented_box(pr)
            x = float(np.clip((1 - ((pc - c) @ ub) / hl) / 2, 0, 1))
            funnels.append((pc, max(phl, 0.9), max(phw, 0.9), zd(x) - 0.3, zdm + 0.85 * (top - zdm),
                            0.12 * (top - zdm), paints["funnel"]))
    else:
        for x, flen, fwid, ztop, paint, btop in prof.get("funnels", []):
            if prof.get("relative"):
                flen, fwid, z1, btop = flen * 2 * hl, fwid * 2 * hl, zd(x) + ztop, btop * ztop
            else:
                flen, fwid, z1, btop = flen * k, fwid * k, zabs(ztop), btop * k
            funnels.append((at(x), flen / 2, fwid / 2, zd(x) - 0.3, z1, btop, paints.get(paint, upper)))
    for pc, rx, ry, z0, z1, btop, mat in funnels:
        if btop > 0 and mat is paints["funnel"] and not navy:      # a liner's: red with a black top
            _cylinder(m, pc, ub, rx, ry, z0, z1 - btop, mat, top=False)
            _cylinder(m, pc, ub, rx, ry, z1 - btop, z1, black)
        else:
            _cylinder(m, pc, ub, rx, ry, z0, z1, mat)
    for x, diam, hgt, lift, way, barrel in prof.get("turrets", []):
        diam, hgt, lift, barrel = diam * k, hgt * k, lift * k, barrel * k
        pc = at(x)
        z0 = zd(x) - 0.3
        if lift > 0:                                    # a superfiring turret's barbette
            _cylinder(m, pc, ub, 0.4 * diam, 0.4 * diam, z0, z0 + lift + 0.3, upper)
        zt = z0 + lift + 0.3
        _cylinder(m, pc, ub, diam / 2, 0.45 * diam, zt, zt + hgt, upper)
        sg = 1 if way == "bow" else -1
        for off in (-0.22, 0.0, 0.22):                  # its three guns
            _box(m, pc + ub * sg * (0.3 * diam + barrel / 2) + v * off * diam, ub, barrel / 2, 0.35 * k,
                 zt + 0.45 * hgt, zt + 0.45 * hgt + 0.7 * k, upper)
    mt = max(0.35, 0.012 * beam)
    for x, ztop in prof.get("masts", []):
        z1 = zabs(ztop) if (known and ztop is not None) else top
        z0 = zd(x) - 0.3
        pc = at(x)
        _box(m, pc, ub, mt, mt, z0, z1, upper)
        yard = z1 - 0.15 * (z1 - z0)
        _box(m, pc, ub, mt, min(0.4 * beam, 0.9 * half_b(x) + 2.0), yard, yard + 2 * mt, upper)


def _vault(m, spec, gmin, gmax, mat_idx, lite=False):
    """An arched hangar: plain walls round its outline to spec.floor_h, then an arch over it to its height, every
    corner drawn in toward the footprint's spine (its long axis; shortened by the half span when its ends are rounded,
    spec.circle: those become half domes, Hangar One's; at square ends the arch's own profile closes the end)."""
    r = _open_ring(spec.ring)
    c, u, hl, hw = oriented_box(r)
    z0, zb = gmin - (SKIRT_LITE_M if lite else SKIRT_M), base_z(spec, gmin, gmax)
    ze, top = zb + spec.floor_h, zb + spec.height
    fac, rm = mat_idx[spec.facade], mat_idx[spec.roof_mat]
    _plain_walls(m, r, z0, ze, fac)
    spine = max(0.0, hl - hw) if spec.circle else hl
    q = c + u * np.clip((r - c) @ u, -spine, spine)[:, None]
    segs = 4 if lite else ARCH_SEGS + 4
    prev = None
    for kk in range(segs + 1):
        th = (math.pi / 2) * kk / segs
        f, z = math.cos(th), ze + (top - ze) * math.sin(th)
        cur = [(float(qq[0] + (p[0] - qq[0]) * f), float(qq[1] + (p[1] - qq[1]) * f), z) for p, qq in zip(r, q)]
        if prev is not None:
            for i in range(len(r)):
                j = (i + 1) % len(r)
                a, b_, cc, d = prev[i], prev[j], cur[j], cur[i]
                if np.allclose(a, b_):
                    continue
                mid = (np.add(a, b_) + np.add(cc, d)) / 4
                ax = c + u * float(np.clip((mid[:2] - c) @ u, -spine, spine))
                hint = np.array([mid[0] - ax[0], mid[1] - ax[1], max(mid[2] - ze, 0.0)])
                _quad_out(m, a, b_, cc, d, hint, rm)
        prev = cur


class Mesh:
    """Positions, uvs, colours, triangles and their material index, appended to. A material is a token
    (index, RGBA) as _Looks gives. Kept as flat lists of numbers (x y z x y z ..., one material per triangle):
    a road tile adds ~25k quads, and lists of tuples flattened at the end took ~2.5 s of a Steam Deck's 87 s."""

    def __init__(self):
        self.P, self.UV, self.C, self.T, self.M = [], [], [], [], []
        self.n = 0

    def quad(self, a, b, c, d, uva, uvb, uvc, uvd, mat):
        """Corners counter-clockwise seen from outside."""
        i = self.n
        self.P += (*a, *b, *c, *d)
        self.UV += (*uva, *uvb, *uvc, *uvd)
        self.C += mat[1] * 4
        self.T += (i, i + 1, i + 2, i, i + 2, i + 3)
        self.M += (mat[0], mat[0])
        self.n += 4

    def quads(self, P, UV, mid, mats):
        """Quads from a kernel (earthfast.building_walls): P 12 numbers per quad, UV 8, mid the index into mats (material
        tokens) per quad; the same lists quad() would make."""
        q = len(mid)
        if not q:
            return
        i = self.n
        self.P += P
        self.UV += UV
        cols = [tuple(mt[1]) * 4 if mt is not None else None for mt in mats]
        for k in mid:
            self.C += cols[k]
        self.T += (np.arange(q, dtype=np.int64)[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3], np.int64) + i).ravel().tolist()
        idx = [mt[0] if mt is not None else None for mt in mats]
        self.M += [idx[k] for k in mid for _ in (0, 1)]
        self.n += 4 * q

    def tri(self, a, b, c, uva, uvb, uvc, mat):
        i = self.n
        self.P += (*a, *b, *c)
        self.UV += (*uva, *uvb, *uvc)
        self.C += mat[1] * 3
        self.T += (i, i + 1, i + 2)
        self.M.append(mat[0])
        self.n += 3

    def poly(self, pts3, uv, mat):
        """A flat polygon facing up when counter-clockwise seen from above, down when clockwise (its underside:
        the ear clipping wants counter-clockwise rings and gave a clockwise one no triangles at all, so the
        parking decks' and floating parts' undersides were missing, the user's report 2026-10-14)."""
        xy = np.asarray(pts3)[:, :2]
        if _area2(xy) < 0:
            k = len(xy)
            tris = np.array([(k - 1 - c, k - 1 - b, k - 1 - a) for a, b, c in triangulate(xy[::-1])],
                            np.int64).reshape(-1, 3)
        else:
            tris = triangulate(xy)
        i = self.n
        self.P += np.asarray(pts3, np.float64)[:, :3].ravel().tolist()
        self.UV += np.asarray(uv, np.float64)[:, :2].ravel().tolist()
        self.C += tuple(mat[1]) * len(pts3)
        self.T += (np.asarray(tris, np.int64).reshape(-1, 3) + i).ravel().tolist()
        self.M += [mat[0]] * len(tris)
        self.n += len(pts3)

    def arrays(self):
        return (np.array(self.P, np.float64).reshape(-1, 3), np.array(self.UV, np.float64).reshape(-1, 2),
                np.array(self.T, np.int64).reshape(-1, 3), np.array(self.M, np.int64),
                np.array(self.C, np.uint8).reshape(-1, 4))


# A sash window cut from the plaster sheet (band 2, bay 0, its sill included): u, v ranges (Collada v up)
WIN_UV = (53 / 1024, 154 / 1024, 1.0 - (512 + 238) / 1024, 1.0 - (512 + 45) / 1024)
WIN_W, WIN_H, WIN_SILL = 1.05, 1.75, 0.85   # m
HOUSE_BAY_M = 3.4           # m between a house's windows
DOOR_GAP_M = 0.75           # m each side of a front door kept free of the ground floor's windows
WIN_OUT_M = 0.03            # m the window stands proud of its wall


_WALL_CONSTS = None


def _walls(m, spec, ring, z0, z_eave, mat_idx, lite=False):
    """Each wall of the ring, floor by floor: one band of the window sheet, a whole number of bays; a plain wall with
    sash windows laid over it for a plain tileable material (spec.wall; a big box's storefront band); a plain plinth
    under the floors on a slope; a shopfront ground floor; the good bays only where a band has some; the front door's
    bay as the band's plain strip, at the sheet's scale. The Rust loop (earthfast.building_walls: the quads the Python
    made, in its order and arithmetic)."""
    global _WALL_CONSTS
    if _WALL_CONSTS is None:
        _WALL_CONSTS = [float(v) for v in (SIDING_UV_M, TILE_PX_M, FOUNDATION_M, BANDS, BAND_SHOP[0], BAND_SHOP[1],
                                           DOOR_GAP_M, PLAIN_STRIP, GROUND_FLOOR_SHOP_M, WIN_OUT_M, HOUSE_BAY_M,
                                           *WIN_UV, WIN_W, WIN_H, WIN_SILL)]
    fast.require(76, "building_walls")
    band = spec.band or (0, 1)
    good = list(band[2]) if len(band) > 2 else []
    door = spec.door
    P, UV, mid = fast._ef.building_walls(
        np.ascontiguousarray(np.asarray(ring, np.float64)[:, :2]), float(z0), float(z_eave), bool(spec.wall),
        bool(spec.plain), int(band[0]), int(band[1]), [int(g) for g in good], int(spec.floors), float(spec.floor_h),
        bool(spec.shop), int(door[0]) if door else -1, float(door[1]) if door else 0.0, spec.roof == "flat",
        float(spec.seed) if isinstance(spec.seed, float) else 0.0, spec.windows is False,
        int(spec.front) if spec.front is not None else -1, bool(lite), _WALL_CONSTS)
    if mid:
        m.quads(P, UV, mid, [mat_idx[spec.facade], mat_idx[SHOP] if (spec.shop or spec.wall) else None,
                             mat_idx["bne_bld_fac_white"] if spec.wall else None])


PLAIN_STRIP = 0.3           # of a bay: the plain wall between two windows of a sheet's band (centred on a bay edge)


def _cornice(m, spec, ring, z_eave, mat_idx):
    """A box moulding round a flat roof's edge (row houses' cornices, parapets), the plain white plaster."""
    mat = mat_idx["bne_bld_fac_white"]
    o, z0, z1 = CORNICE_OUT_M, z_eave - CORNICE_DOWN_M, z_eave + CORNICE_UP_M
    uv = ((0.002, 0.995), (0.012, 0.995), (0.012, 0.999), (0.002, 0.999))
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        e = b - a
        L = float(math.hypot(*e))
        if L < 0.3:
            continue
        d = e / L
        nn = np.array([d[1], -d[0]])                  # outward (counter-clockwise ring)
        a2, b2 = a + nn * o - d * o, b + nn * o + d * o
        m.quad((a2[0], a2[1], z0), (b2[0], b2[1], z0), (b2[0], b2[1], z1), (a2[0], a2[1], z1), *uv, mat)
        m.quad((a[0], a[1], z1), (a2[0], a2[1], z1), (b2[0], b2[1], z1), (b[0], b[1], z1), *uv, mat)      # top
        m.quad((a[0], a[1], z0), (b[0], b[1], z0), (b2[0], b2[1], z0), (a2[0], a2[1], z0), *uv, mat)      # under
        # the lip's inner face, above the roof (seen from on the roof it was missing: the user's report, 2026-10-14)
        m.quad((b[0], b[1], z_eave), (a[0], a[1], z_eave), (a[0], a[1], z1), (b[0], b[1], z1), *uv, mat)


def _inside(ring, p):
    """Point p inside the ring (even-odd)."""
    x, y = p
    return fast._ef.point_inside(np.asarray(ring, np.float64), float(x), float(y))     # (Rust; even-odd)


def _box(m, c, u, hx, hy, z0, z1, mat):
    """An axis-u box (half sizes hx along u, hy across) from z0 to z1: four sides and a top."""
    v = np.array([-u[1], u[0]])
    k = [c + u * sx * hx + v * sy * hy for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    uv = ((0.002, 0.995), (0.012, 0.995), (0.012, 0.999), (0.002, 0.999))
    for i in range(4):
        a, b = k[i], k[(i + 1) % 4]
        m.quad((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1), *uv, mat)
    m.quad(*[(p[0], p[1], z1) for p in k], *uv, mat)


def _signs(m, spec, ring, z_top, mat_idx):
    """The sign band over a shop's street wall (spec.sign: its edge), shop by shop, a few without one."""
    edge = spec.sign
    a, b = ring[edge], ring[(edge + 1) % len(ring)]
    L = float(math.hypot(*(b - a)))
    if L < 3.0:
        return
    d = (b - a) / L
    out = np.array([d[1], -d[0]])
    t, k = 0.3, 0
    while t < L - 1.5:
        w = SIGN_SHOP_M[0] + (SIGN_SHOP_M[1] - SIGN_SHOP_M[0]) * _hash01(spec.seed, "signw", k)
        t1 = min(L - 0.3, t + w)
        if L - 0.3 - t1 < 2.5:                         # no sliver of a shop at the end
            t1 = L - 0.3
        if _hash01(spec.seed, "signon", k) < 0.85:
            c = a + d * ((t + t1) / 2) + out * (SIGN_OUT / 2)
            look = f"sign|{int(_hash01(spec.seed, 'signc', k) * len(SIGN_TINTS))}"
            _box(m, c, d, (t1 - t) / 2 - 0.15, SIGN_OUT / 2, z_top - 0.15 - SIGN_H, z_top - 0.15, mat_idx[look])
        t, k = t1, k + 1


PORCH_DEPTH_M = 2.2
PORCH_H_M = 2.7             # the porch roof's underside above the floor
PORCH_DECK_M = 0.35         # the deck's top above the ground


PORTAL_NEAR_M = 25.0        # m round a building where the road at a tunnel's mouth is looked for


def _portal_lift(frame, ring, x0, y0, tunnels_xy, elev):
    """How far (m) over its lowest ground a building over a tunnel stands: to OVER_TUNNEL_CLEAR_M over the lowest ground
    along the tunnel lines within PORTAL_NEAR_M of it (the road at the mouth), 0 when its ground is that high."""
    lo = ring.min(0) - PORTAL_NEAR_M
    hi = ring.max(0) + PORTAL_NEAR_M
    pts = []
    for line in tunnels_xy:
        ln = np.asarray(line, np.float64)
        if len(ln) < 2:
            continue
        seg = np.hypot(*np.diff(ln, axis=0).T)
        s = np.concatenate(([0.0], np.cumsum(seg)))
        t = np.arange(0.0, s[-1] + 1e-9, 2.0)
        p = np.stack([np.interp(t, s, ln[:, 0]), np.interp(t, s, ln[:, 1])], axis=1)
        p = p[(p[:, 0] >= lo[0]) & (p[:, 0] <= hi[0]) & (p[:, 1] >= lo[1]) & (p[:, 1] <= hi[1])]
        if len(p):
            pts.append(p)
    if not pts:
        return 0.0
    q = np.concatenate(pts + [ring])
    la, lo_, _ = frame.enu_to_geodetic(q[:, 0] + x0, q[:, 1] + y0, np.zeros(len(q)))
    z = np.asarray(elev(la, lo_), np.float64)
    nr = len(ring)
    road, ground = z[:-nr], z[-nr:]
    if not len(road) or not np.isfinite(road).any() or not np.isfinite(ground).any():
        return 0.0
    return max(0.0, float(np.nanmin(road)) + OVER_TUNNEL_CLEAR_M - float(np.nanmin(ground)))


def base_z(spec, gmin, gmax):
    """The ground a building's floors stand on: the footprint's highest, but the lowest for one over a tunnel's portal
    (the road at its mouth: from the hill over it, Wabash's portal house floated ~10 m over the portal and
    Eisenhower's portal building over its frames, 2026-10-17)."""
    return gmin if spec.portal else gmax


def drawn_ring(spec):
    """The footprint its walls are drawn on: a pitched roof's house is its oriented box (build); its door, steps and
    porch go on these walls (on the map's outline they stood off a box wall at an angle, or inside it, only the
    door's frame showing: Stelvio pass, the user's reports 2026-10-17)."""
    ring = spec.ring
    if spec.roof != "flat":
        c, u, hl, hw = oriented_box(ring)
        v = np.array([-u[1], u[0]])
        ring = np.array([c - u * hl - v * hw, c + u * hl - v * hw, c + u * hl + v * hw, c - u * hl + v * hw])
    return ring


def _porch(m, spec, base, mat_idx):
    """A front porch on the front door's wall (spec.door: (edge, distance along it) of spec.ring): a deck, three
    posts and a shallow roof, the house's trim white and its roof's colour."""
    edge, sd = spec.door
    ring = drawn_ring(spec)
    a, b = ring[edge], ring[(edge + 1) % len(ring)]
    L = float(math.hypot(*(b - a)))
    if L < 4.0:
        return
    d = (b - a) / L
    out = np.array([d[1], -d[0]])
    w = min(L * 0.75, 7.5)
    c0 = float(np.clip(sd, w / 2, L - w / 2))
    c = a + d * c0
    white = mat_idx["bne_bld_fac_white"]
    D = PORCH_DEPTH_M
    _box(m, c + out * (D / 2), d, w / 2, D / 2, base - 0.3, base + PORCH_DECK_M, white)              # deck
    top = base + PORCH_DECK_M + PORCH_H_M
    for t in (-w / 2 + 0.15, 0.0, w / 2 - 0.15):                                                   # posts
        _box(m, c + d * t + out * (D - 0.15), d, 0.08, 0.08, base + PORCH_DECK_M, top, white)
    roof = mat_idx[spec.roof_mat] if spec.roof_mat else white
    _box(m, c + out * (D / 2 + 0.1), d, w / 2 + 0.15, D / 2 + 0.1, top, top + 0.18, roof)          # roof


def door_ground(spec, edge, along, gmax):
    """The ground at a point `along` m on edge `edge` of the footprint (its corners' ground, spec.gring), or gmax
    when not known: a row house's door and steps (from the footprint's highest ground, its back yard's on a slope,
    the steps on Bradford Street stood 1.5 m over the sidewalk, five or six of them; Mapillary: one or two)."""
    g = getattr(spec, "gring", None)
    ring = spec.ring
    if g is None or len(g) != len(ring):
        return gmax
    a, b = ring[edge], ring[(edge + 1) % len(ring)]
    L = float(math.hypot(*(b - a)))
    t = 0.0 if L < 1e-6 else min(max(along / L, 0.0), 1.0)
    return float(g[edge] + t * (g[(edge + 1) % len(ring)] - g[edge]))


def _stoop(m, spec, ring, gmin, base, mat_idx):
    """Marble steps up to the front door (spec.stoop: (edge, distance along it); base: the ground there, the door
    STOOP_H over it): STOOP_STEPS treads to a landing, each block down under the footprint's lowest ground (they floated where the street falls away, Fells Point, 2026-10-17)."""
    edge, sd = spec.stoop
    a, b = ring[edge], ring[(edge + 1) % len(ring)]
    L = float(math.hypot(*(b - a)))
    if L < 1e-6:
        return
    d = (b - a) / L
    out = np.array([d[1], -d[0]])
    c = a + d * sd
    mat = mat_idx[MARBLE_LOOK]
    base = door_ground(spec, edge, sd, base)     # (the ground at the door)
    top = base + STOOP_H
    z0 = min(gmin, base) - 0.3
    # as many steps as it takes down to the street (the footprint's lowest ground): from the highest one only the
    # bottom step stood a metre over the sidewalk on a white block (Bradford Street, Mapillary 2026-10-17)
    n = int(np.clip(round((top - min(gmin, base)) / STOOP_RISE), STOOP_STEPS - 1, STOOP_MAX_STEPS))
    rise = min((top - min(gmin, base)) / n, STOOP_RISE_MAX)    # (n risers: the landing's and n - 1 steps' fronts;
                                                                # higher, the bottom step on a plinth)
    for k in range(n):
        depth = STOOP_LANDING + k * STOOP_TREAD
        zt = top - k * rise
        _box_tiled(m, c + out * depth / 2, d, STOOP_W / 2, depth / 2, z0, zt, mat)


MARBLE_LOOK = "bne_marble"


def _box_tiled(m, c, u, hx, hy, z0, z1, mat, tile=1.0):
    """_box with the material tiled at `tile` m per repeat (a stone texture, not the facade sheets' colour patches)."""
    v = np.array([-u[1], u[0]])
    k = [c + u * sx * hx + v * sy * hy for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for i in range(4):
        p, q = k[i], k[(i + 1) % 4]
        w = float(math.hypot(*(q - p))) / tile
        hz0, hz1 = z0 / tile, z1 / tile
        m.quad((p[0], p[1], z0), (q[0], q[1], z0), (q[0], q[1], z1), (p[0], p[1], z1),
               (0.0, hz0), (w, hz0), (w, hz1), (0.0, hz1), mat)
    m.quad(*[(p[0], p[1], z1) for p in k], *[(p[0] / tile, p[1] / tile) for p in k], mat)


def _steeple(m, spec, z_eave, mat_idx):
    """A church's square tower with a pyramid spire, at one end of its nave (hashed)."""
    c, u, hl, hw = oriented_box(spec.ring)
    if _hash01(spec.seed, "steeple") < 0.5:
        u = -u
    side = float(np.clip(1.1 * hw, 3.0, 6.0)) / 2
    p = c + u * (hl - side)
    for k in range(1, 12):              # on the footprint (an irregular one's box end can be outside it)
        if _inside(spec.ring, p):
            break
        p = c + u * (hl - side) * (1 - k / 12)
    rise = (min(hw, 12.0) + EAVE_M) * math.tan(math.radians(spec.pitch or PITCH["gabled"]))
    top = z_eave + rise + 6.0
    _box(m, p, u, side, side, z_eave - 1.0, top, mat_idx[spec.facade])
    v = np.array([-u[1], u[0]])
    k = [p + u * sx * side + v * sy * side for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    apex = (float(p[0]), float(p[1]), top + 2.6 * side * 2)
    rm = mat_idx[f"{ROOF_SLATE}|3"]
    for i in range(4):
        a, b = k[i], k[(i + 1) % 4]
        m.tri((a[0], a[1], top), (b[0], b[1], top), apex, (0, 0), (1, 0), (0.5, 2.0), rm)


ROOF_CLUTTER_M2 = 250.0     # flat roofs this large get boxes: air-conditioning units, stair bulkheads


def _clutter(m, spec, ring, z_eave, mat_idx):
    area = _area2(ring) / 2
    if spec.roof != "flat" or spec.plain or area < ROOF_CLUTTER_M2 or spec.cls in CHURCH_CLASSES:
        return
    c, u, hl, hw = oriented_box(ring)
    v = np.array([-u[1], u[0]])
    n = min(6, 1 + int(area / 600.0))
    mat = mat_idx["bne_bld_fac_grey"]
    for k in range(n):
        s = (_hash01(spec.seed, "cx", k) * 2 - 1) * (hl - 3.0)
        t = (_hash01(spec.seed, "cy", k) * 2 - 1) * (hw - 3.0)
        p = c + u * s + v * t
        if not _inside(ring, p):
            continue
        big = _hash01(spec.seed, "cb", k) < 0.25
        hx, hy, h = (2.0, 1.6, 2.6) if big else (1.1, 0.7, 1.1)
        _box(m, p, u, hx, hy, z_eave, z_eave + h, mat)


def _roof(m, spec, ring, z_eave, mat_idx):
    rm = mat_idx[spec.roof_mat]
    if spec.roof == "flat":
        pts = [(p[0], p[1], z_eave) for p in ring]
        uv = [(p[0] / ROOF_UV_M, p[1] / ROOF_UV_M) for p in ring]
        m.poly(pts, uv, rm)
        return
    c, u, hl, hw = oriented_box(ring)
    v = np.array([-u[1], u[0]])
    hl2, hw2 = hl + EAVE_M, hw + EAVE_M
    rise = hw2 * math.tan(math.radians(roof_pitch(spec)))
    ze = z_eave - EAVE_M * math.tan(math.radians(roof_pitch(spec)))
    zr = ze + rise

    def at(s, t, z):
        p = c + u * s + v * t
        return (float(p[0]), float(p[1]), float(z))
    slope = math.hypot(hw2, rise)

    def uv(s, t):            # along the ridge, up the slope (m / ROOF_UV_M)
        return (s / ROOF_UV_M, (hw2 - abs(t)) / hw2 * slope / ROOF_UV_M)
    if spec.roof == "barrel":
        _barrel(m, spec, c, u, v, hl, hw, z_eave, rm, mat_idx[spec.facade])
        return
    if spec.roof == "gabled":
        r = hl2
        m.quad(at(-hl2, -hw2, ze), at(hl2, -hw2, ze), at(r, 0, zr), at(-r, 0, zr),
               uv(-hl2, -hw2), uv(hl2, -hw2), uv(r, 0), uv(-r, 0), rm)
        m.quad(at(hl2, hw2, ze), at(-hl2, hw2, ze), at(-r, 0, zr), at(r, 0, zr),
               uv(hl2, hw2), uv(-hl2, hw2), uv(-r, 0), uv(r, 0), rm)
        # the gable ends: wall triangles from the eave line up to the ridge, the facade's plain colour
        fac = mat_idx[spec.facade]
        for sgn in (-1, 1):
            s = sgn * hl
            a, b, top = at(s, -sgn * hw, z_eave), at(s, sgn * hw, z_eave), at(s, 0, zr - EAVE_M * 0.2)
            if sgn > 0:
                m.tri(a, b, top, (0, 0.995), (0.02, 0.995), (0.01, 0.999), fac)
            else:
                m.tri(a, b, top, (0, 0.995), (0.02, 0.995), (0.01, 0.999), fac)
    else:                    # hipped
        r = max(hl2 - hw2, 0.0)
        m.quad(at(-hl2, -hw2, ze), at(hl2, -hw2, ze), at(r, 0, zr), at(-r, 0, zr),
               uv(-hl2, -hw2), uv(hl2, -hw2), uv(r, 0), uv(-r, 0), rm)
        m.quad(at(hl2, hw2, ze), at(-hl2, hw2, ze), at(-r, 0, zr), at(r, 0, zr),
               uv(hl2, hw2), uv(-hl2, hw2), uv(-r, 0), uv(r, 0), rm)
        m.tri(at(hl2, -hw2, ze), at(hl2, hw2, ze), at(r, 0, zr),
              (-hw2 / ROOF_UV_M, 0), (hw2 / ROOF_UV_M, 0), (0, slope / ROOF_UV_M), rm)
        m.tri(at(-hl2, hw2, ze), at(-hl2, -hw2, ze), at(-r, 0, zr),
              (-hw2 / ROOF_UV_M, 0), (hw2 / ROOF_UV_M, 0), (0, slope / ROOF_UV_M), rm)


def _barrel(m, spec, c, u, v, hl, hw, z_eave, rm, fac):
    """An arched (barrel) roof along the oriented box's long axis: an elliptic arch BARREL_RISE of the half span
    high, overhanging the walls by EAVE_M, its end walls the facade's plain colour."""
    hw2, hl2, rise = hw + EAVE_M, hl + EAVE_M, hw * BARREL_RISE
    ths = [math.pi * k / ARCH_SEGS for k in range(ARCH_SEGS + 1)]
    arch = [(-hw2 * math.cos(t), z_eave + rise * math.sin(t)) for t in ths]       # (across, z) from -v to +v

    def at(s, t, z):
        p = c + u * s + v * t
        return (float(p[0]), float(p[1]), float(z))
    run = 0.0
    for (t0, z0), (t1, z1) in zip(arch[:-1], arch[1:]):
        d = math.hypot(t1 - t0, z1 - z0)
        m.quad(at(-hl2, t0, z0), at(hl2, t0, z0), at(hl2, t1, z1), at(-hl2, t1, z1),
               (-hl2 / ROOF_UV_M, run / ROOF_UV_M), (hl2 / ROOF_UV_M, run / ROOF_UV_M),
               (hl2 / ROOF_UV_M, (run + d) / ROOF_UV_M), (-hl2 / ROOF_UV_M, (run + d) / ROOF_UV_M), rm)
        run += d
    for sgn in (-1, 1):             # the end walls under the arch (a fan from the eave line's middle)
        s = sgn * hl
        mid = at(s, 0, z_eave)
        pts = [at(s, t * hw / hw2, z) for t, z in arch]
        for a, b in zip(pts[:-1], pts[1:]):
            if sgn > 0:
                m.tri(mid, b, a, (0, 0.995), (0.02, 0.995), (0.01, 0.999), fac)
            else:
                m.tri(mid, a, b, (0, 0.995), (0.02, 0.995), (0.01, 0.999), fac)


# The engine draws a shape's mesh with 16-bit indices: past 65,535 vertices they wrap and triangles join far-apart
# vertices (the user's report 2026-10-14: slabs across a Fells Point street; dense tiles had up to 81k vertices).
MAX_MESH_VERTS = 60000


def _concat(meshes):
    """build() meshes (the same MATERIALS) as one."""
    if len(meshes) == 1:
        return meshes[0]
    off = np.cumsum([0] + [len(m[0]) for m in meshes[:-1]])
    return (np.concatenate([m[0] for m in meshes]), np.concatenate([m[1] for m in meshes]),
            np.concatenate([m[2] + o for m, o in zip(meshes, off)]), np.concatenate([m[3] for m in meshes]),
            meshes[0][4], np.concatenate([m[5] for m in meshes]))


def build_parts(specs_and_ground, lite=False, max_verts=MAX_MESH_VERTS, with_lod=False):
    """build() split into meshes of at most max_verts vertices each (whole buildings per mesh): a list, [] if empty.
    Each building is built once (build draws them one by one) and the meshes joined. with_lod: each mesh carries the
    lite mesh of its buildings as a 7th element (write: the shape's coarse detail)."""
    parts, cur, cur_lite, n = [], [], [], 0

    def flush():
        if cur:
            mesh = _concat(cur)
            parts.append(mesh + (_concat(cur_lite) if cur_lite else None,) if with_lod and not lite else mesh)

    for item in specs_and_ground:
        one = build([item], lite)
        if one is None:
            continue
        if cur and n + len(one[0]) > max_verts:
            flush()
            cur, cur_lite, n = [], [], 0
        cur.append(one)
        n += len(one[0])
        if with_lod and not lite:
            o = build([item], lite=True)
            if o is not None:
                cur_lite.append(o)
    flush()
    return parts


# A parking structure you can drive into (the user, 2026-10-14: "garage buildings ... are solid"): open storeys of
# concrete decks on perimeter columns, a parapet round each upper deck, and a two-lane switchback ramp along the
# longest wall (one lane up per storey, a U-turn landing at each end). Only where the ramp fits inside the footprint
# and the ground under it is near level; otherwise the closed box as before. The tile's mesh collides as drawn.
DECK_LANE_M = 3.6           # a ramp lane
DECK_EDGE_M = 0.8           # the ramp lanes start this far in from the wall (the columns stand there)
DECK_LANDING_M = 7.0        # flat deck at each end of the ramps (the U-turn)
DECK_SLOPE = 0.17           # the steepest switchback ramp (CCDC p.18: up to 20 % for a covered non-parking ramp)
DECK_SLAB_M = 0.3
DECK_COLUMN_M = 9.0         # perimeter column spacing
DECK_LEVEL_M = 3.5          # ground this uneven (max - min) under the footprint: no deck (up to it: the ground deck
                            # level at the highest ground on a plinth, driven into from the uphill side)
DECK_HEADROOM_M = 2.6       # the lowest storey height that still drives
DECK_STOREY_M = 3.0         # a deck's storey where the mapped floors are too many for the height


DECK_RAMP_EDGES = 4         # the longest edges tried for the ramp, longest first


def _deck_layout(ring, storey):
    """The ramp's frame on the longest edge of the counter-clockwise ring it fits along: (edge index i, A, along
    unit u, inward unit n, s0, s1), the ramps between arc lengths s0 and s1; None when it fits along none of the
    DECK_RAMP_EDGES longest (too short for DECK_SLOPE, or the well outside the ring: the longest edge of an
    L-shaped garage can be the short leg's, 3 of downtown Baltimore's 15 garages were boxes, 2026-10-17)."""
    lens = [float(np.hypot(*(ring[(k + 1) % len(ring)] - ring[k]))) for k in range(len(ring))]
    depth = DECK_EDGE_M + 2 * DECK_LANE_M
    for i in sorted(range(len(ring)), key=lambda k: -lens[k])[:DECK_RAMP_EDGES]:
        a, b = ring[i], ring[(i + 1) % len(ring)]
        L = lens[i]
        s0, s1 = 0.5 + DECK_LANDING_M, L - 0.5 - DECK_LANDING_M
        if s1 - s0 - TRANSITION_M < storey / DECK_SLOPE:   # (its middle steeper by the p.18 transitions)
            return None                         # the rest are shorter
        u = (b - a) / L
        n = np.array([-u[1], u[0]])             # the inside is on the left
        if all(_inside(ring, a + u * s + n * d) for s in np.linspace(0.5, L - 0.5, 9)
               for d in (0.3, depth / 2, depth + 0.5, depth + DECK_LANDING_M / 2)):
            return i, a, u, n, s0, s1
    return None


def _deck_wall(m, p, q, z0, z1, mat, t=0.2):
    """A thin wall (both faces and a top) from p to q, z0 to z1."""
    d = q - p
    L = float(np.hypot(*d))
    if L > 0.05:
        _box(m, (p + q) / 2, d / L, L / 2, t / 2, z0, z1, mat)


# ------------------------------------------------------------------------------------------- parking structures
# The parking-module garage (the user, 2026-10-28, a garage by Century Winchester in San Jose: "no room to turn",
# "ramp far too narrow, a slot between tall walls", "gaps"; then: encode garages "variously" to the CCDC standard).
#
# The standard: CCDC (Capital City Development Corporation, Boise) Parking Structure Design Guidelines, update
# 2019-07-18, ccdcboise.com/wp-content/uploads/2022/04/CCDC-PDG-Update-7-18-2019.pdf. "p.N" below is the document's
# own printed page number (the PDF's page N + 2). Feet in the table, metres in code (FT).
#
# | rule                                              | value                                   | page |
# |---------------------------------------------------|-----------------------------------------|------|
# | parking (sloped-floor) ramp, preferred            | <= 5 %                                  | 17   |
# | parking ramp, maximum (IBC)                       | 6.67 % (7 % tolerated, dense urban)     | 17   |
# | non-parking ramp (helix, express, speed)          | 12-14 % max (20 % if covered/snow-melt) | 18   |
# | ramp-end transitions                              | grade >= 8 %: >= 10' long, half grade   | 18   |
# | structural clear height                           | 8'-2" (vans); 7'-4" w/o vans; 13'-6"    | 18-19|
# |                                                   | for trucks/buses on the ground floor    |      |
# | floor-to-floor                                    | ~10' minimum (one 360 deg turn = 10')   | 21   |
# | floor-to-floor for adaptive reuse                 | 15' ground, 12' upper                   | 11   |
# | one-way traffic                                   | never with 90 deg stalls                | 20   |
# | single-threaded, 2 bays two-way 90 deg            | ~122' x 155' min, 5' rise per bay;      | 21   |
# |                                                   | ~255' long: one bay sloped 10', one flat|      |
# | single-threaded levels                            | <= 6 preferred                          | 21   |
# | double-threaded (one-way, angled)                 | 54-58' modules, 60/70/75 deg preferred, | 23   |
# |                                                   | 10' rise per module, ~240' long,        |      |
# |                                                   | rises 2 levels per 360 deg; 7+ levels   |      |
# | double-loaded bay width                           | 54-60'; ideal length >= 240'            | 13   |
# | module / stall / aisle by angle and UCF           | CCDC_LAYOUT below                       | 30   |
# | end-bay drive aisle, two-way                      | >= 26'                                  | 31   |
# | striped buffer, last stall to aisle               | >= 3'                                   | 31   |
# | turning radius, cars                              | inside 15', outside 30'                 | 31   |
# | turning radius, single-unit trucks/buses          | inside 28', outside 42'                 | 31   |
# | stall beside wall/column/stair                    | +1' wider                               | 31   |
# | column spacing, short span                        | ~27' x 30' (a column every 3 stalls)    | 63   |
# | column spacing, cast-in-place / precast           | 20-27' / 18-22'                         | 52,54|
# | entry/exit with control equipment                 | <= 3 % slope; 75-100' (150') from corner| 27   |
# | accessible stall                                  | 8' + 5' aisle (van: 8' aisle), <= 2 %   | 34   |
# | accessible route                                  | <= 5 % running, <= 2 % cross, >= 3' wide| 35   |
# | stairs                                            | >= 44" wide, in dead corners            | 33   |
# | elevator                                          | >= 5'-0" x 7'-0" car, 3,500 lb          | 33   |
# | floor drainage slope                              | 2 % desired, 1.5 % min (not drawn)      | 48   |
# | ground-floor retail on street sides               | encouraged (not drawn)                  | 58   |
# | barrier/parapet height                            | not given: IBC guard 42"                | -    |
FT = 0.3048


def _ft(s):
    """Feet from 12'-9" style text."""
    f, i = s.replace('"', "").split("'")
    return int(f) * FT + (int(i.strip("-") or 0) * FT / 12.0)


# p.30 Parking Layout Dimensions (wall-to-wall double-loaded module): (user comfort factor, angle) -> stall width
# projection WP, module width MW, vehicle projection VP, aisle width AW. UCF 4 = 9'-0" stalls (LOS A), UCF 3 =
# 8'-9" (LOS B: visitors, retail), UCF 2 = 8'-6" (LOS C: commuters, residents) (p.29).
CCDC_LAYOUT = {(u, a): tuple(_ft(v) for v in row.split()) for u, rows in {
    4: {45: "12'-9\" 49'-10\" 17'-7\" 14'-8\"", 50: "11'-9\" 51'-7\" 18'-2\" 15'-3\"",
        55: "11'-0\" 53'-0\" 18'-8\" 15'-8\"", 60: "10'-5\" 54'-6\" 19'-0\" 16'-6\"",
        65: "9'-11\" 55'-9\" 19'-2\" 17'-5\"", 70: "9'-7\" 57'-0\" 19'-3\" 18'-6\"",
        75: "9'-4\" 58'-0\" 19'-1\" 19'-10\"", 90: "9'-0\" 62'-0\" 18'-0\" 26'-0\""},
    3: {45: "12'-4\" 48'-10\" 17'-7\" 13'-8\"", 50: "11'-5\" 50'-7\" 18'-2\" 14'-3\"",
        55: "10'-8\" 52'-0\" 18'-8\" 14'-8\"", 60: "10'-1\" 53'-6\" 19'-0\" 15'-6\"",
        65: "9'-8\" 54'-9\" 19'-2\" 16'-5\"", 70: "9'-4\" 56'-0\" 19'-3\" 17'-6\"",
        75: "9'-1\" 57'-0\" 19'-1\" 18'-10\"", 90: "8'-9\" 61'-0\" 18'-0\" 25'-0\""},
    2: {45: "12'-0\" 47'-10\" 17'-7\" 12'-8\"", 50: "11'-1\" 49'-7\" 18'-2\" 13'-3\"",
        55: "10'-5\" 51'-0\" 18'-8\" 13'-8\"", 60: "9'-10\" 52'-6\" 19'-0\" 14'-6\"",
        65: "9'-5\" 53'-9\" 19'-2\" 15'-5\"", 70: "9'-1\" 55'-0\" 19'-3\" 16'-6\"",
        75: "8'-10\" 56'-0\" 19'-1\" 17'-10\"", 90: "8'-6\" 60'-0\" 18'-0\" 24'-0\""},
}.items() for a, row in rows.items()}
CCDC_ONE_WAY_ANGLES = (60, 70, 75)          # p.23: the preferred angles for one-way angled parking
PARK_RAMP_PREF = 0.05                       # p.17
PARK_RAMP_MAX = 0.0667                      # p.17 (IBC)
SPEED_RAMP_MAX = 0.12                       # p.18 (12-14 %): non-parking ramps
SPEED_RAMP_MAX2 = 0.14
TRANSITION_FROM = 0.08                      # p.18: a grade from this needs transitions at both ends ...
TRANSITION_M = 10 * FT                      # ... at least 10' long, at half the grade
CLEAR_M = 8 * FT + 2 * FT / 12              # p.18: 8'-2" structural clear height (accessible vans)
STOREY_STD_M = 10 * FT                      # p.21: floor-to-floor ~10' (one 360 deg turn rises 10')
STOREY_MAX_M = 12 * FT                      # p.11: 12' upper floors (adaptive reuse): the tallest we draw
SINGLE_THREAD_MAX_FLOORS = 6                # p.21
DOUBLE_THREAD_MIN_FLOORS = 4                # p.23 recommends 7+; mapped garages are rarely that tall
LONG_GARAGE_M = 240 * FT                    # p.13 / p.23: an ideal (double-threaded) length
END_AISLE_M = 26 * FT                       # p.31: an end bay's two-way drive aisle at least
TURN_OUT_M = 30 * FT                        # p.31: outside turning radius (inside 15')
ROW_BUFFER_M = 3 * FT                       # p.31: striped buffer between the last stall and the aisle
SPEED_END_M = TURN_OUT_M + ROW_BUFFER_M     # flat deck at each end of a speed ramp: room for the U-turn
COLUMN_EVERY_STALLS = 3                     # p.63: short span, a column about every three stalls (27' x 30')
DECK_PARAPET_M = 42 / 12 * FT               # not in CCDC: the IBC's 42" guard (a low parapet, not a wall)

# The 90-degree UCF-4 module (62'): kept as the reference module
PARK_BAY_M = CCDC_LAYOUT[(4, 90)][2]
AISLE_LANE_M = CCDC_LAYOUT[(4, 90)][3] / 2
MODULE_M = CCDC_LAYOUT[(4, 90)][1]
RAMP_PARAPET_IN_M = 0.15    # m the ramp's parapets stand in from its edges
MERGE_COLLINEAR_M = 0.6     # m off the line through its neighbours a vertex may be and still be dropped

# The circulation schemes (p.21-26). Levels are counted in half storeys (h): deck k is h = 2k; the ground deck
# (h = 0) is the whole outline. Along the chosen edge the garage is: a west end bay (s < s0), the middle (s0..s1)
# cut into strips (modules) inward from the edge, an east end bay (s > s1). A strip is sloped (one slab per storey,
# from half-level ha at s0 to hb at s1) or flat; the rest of the middle is flat at every deck.
#   one_bay        single-threaded, one bay sloped a whole storey, the others flat (p.21 "one level bay"), two-way 90
#   two_ramps      the same with the first and last bays sloped (inbound and outbound ramp): long garages
#   both_bays      single-threaded with sloping floors (p.21): every bay sloped half a storey, the end bays half a
#                  storey apart, one-way loop; 90 two-way or angled one-way
#   double_thread  double-threaded (p.23): every bay sloped a whole storey, alternate bays the other way, two
#                  intertwined threads; 90 two-way or angled one-way
#   speed_ramp     flat parking floors and a non-parking speed ramp (p.18, <= 12-14 %, transitions) one aisle wide
#                  along the edge: small footprints that cannot take a sloped parking floor
SCHEMES = ("one_bay", "two_ramps", "both_bays", "double_thread", "speed_ramp")


class GarageLayout(tuple):
    """(edge i, A, along u, inward n, s0, s1, edge length L, modules, grade, scheme, angle, ucf, module width mw,
    strips, west levels, east levels, flat levels): strips ((da, db, slabs | None), ...) inward from edge i, slabs
    ((ha, hb), ...) per storey (None: flat at `flat`); the end bays at the half-levels west / east."""
    __slots__ = ()

    def __new__(cls, *v):
        return super().__new__(cls, v)

    i = property(lambda t: t[0])
    A = property(lambda t: t[1])
    u = property(lambda t: t[2])
    n = property(lambda t: t[3])
    s0 = property(lambda t: t[4])
    s1 = property(lambda t: t[5])
    L = property(lambda t: t[6])
    modules = property(lambda t: t[7])
    grade = property(lambda t: t[8])
    scheme = property(lambda t: t[9])
    angle = property(lambda t: t[10])
    ucf = property(lambda t: t[11])
    mw = property(lambda t: t[12])
    strips = property(lambda t: t[13])
    west = property(lambda t: t[14])
    east = property(lambda t: t[15])
    flat = property(lambda t: t[16])

    @property
    def dims(self):
        """(stall width projection, module, vehicle projection, aisle) of its module (CCDC_LAYOUT)."""
        return CCDC_LAYOUT[(self.ucf, self.angle)]


ModuleLayout = GarageLayout


def _merge_collinear(ring, tol=MERGE_COLLINEAR_M):
    """The ring without vertices within tol of the line through their neighbours (a mapped outline's long wall is
    often several edges: the Winchester garage's 111 m north wall was 39 + 61 m)."""
    pts = [np.asarray(p, np.float64) for p in ring]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for k in range(len(pts)):
            a, p, b = pts[k - 1], pts[k], pts[(k + 1) % len(pts)]
            ab = b - a
            L = float(np.hypot(*ab))
            if L < 1e-9:
                del pts[k]
                changed = True
                break
            off = abs(float(ab[0] * (p - a)[1] - ab[1] * (p - a)[0])) / L
            if off < tol and 0.0 <= float((p - a) @ ab) <= L * L:
                del pts[k]
                changed = True
                break
    return np.array(pts)


def _depth_fits(ring, a, u, n, L, d0, d1):
    """The band d0..d1 inward from the edge (a, u, n, L) inside the ring all along it."""
    return all(_inside(ring, a + u * s + n * d) for s in np.linspace(0.6, L - 0.6, max(9, int(L // 6)))
               for d in np.linspace(d0 + 0.3, d1 - 0.3, 5))


def _scheme_strips(scheme, floors, mods, mw, rw=0.0):
    """(strips, west, east, flat) of a scheme (see SCHEMES)."""
    F = floors
    even = tuple(range(2, 2 * F + 1, 2))
    up = tuple((2 * k, 2 * k + 2) for k in range(F))
    down = tuple((2 * k + 2, 2 * k) for k in range(F))
    if scheme == "speed_ramp":
        return ((0.0, rw, up),), even, even, even
    bands = [(j * mw, (j + 1) * mw) for j in range(mods)]
    if scheme == "one_bay":
        slabs = [up] + [None] * (mods - 1)
    elif scheme == "two_ramps":
        slabs = [up] + [None] * (mods - 2) + [up]
    elif scheme == "both_bays":
        a = tuple((2 * k, 2 * k + 1) for k in range(F))
        b = tuple((2 * k + 2, 2 * k + 1) for k in range(F))
        slabs = [a if j % 2 == 0 else b for j in range(mods)]
        return (tuple((d0, d1, s) for (d0, d1), s in zip(bands, slabs)), even, tuple(range(1, 2 * F, 2)), even)
    else:                                               # double_thread
        slabs = [up if j % 2 == 0 else down for j in range(mods)]
    return tuple((d0, d1, s) for (d0, d1), s in zip(bands, slabs)), even, even, even


def _ramp_profile(s0, s1, rise):
    """[(s, fraction of the rise)] along a ramp: straight below TRANSITION_FROM, else a TRANSITION_M at half the
    grade at each end (p.18)."""
    R = s1 - s0
    g = rise / R
    if g < TRANSITION_FROM or R < 3 * TRANSITION_M:
        return [(s0, 0.0), (s1, 1.0)]
    gm = rise / (R - TRANSITION_M)                      # rise = gm (R - 2T) + gm/2 2T
    f = gm / 2 * TRANSITION_M / rise
    return [(s0, 0.0), (s0 + TRANSITION_M, f), (s1 - TRANSITION_M, 1.0 - f), (s1, 1.0)]


def _garage_layout(ring, storey, floors=4, seed=0.0, edges=4, schemes=SCHEMES):
    """GarageLayout of the counter-clockwise ring, or None: on the longest edge (of the `edges` longest) where any
    CCDC scheme fits, one of the schemes that fit there, picked by the seed (then one of its module types);
    preferring the schemes at or under PARK_RAMP_PREF; a speed ramp when no parking ramp fits."""
    k_ring = len(ring)
    lens = [float(np.hypot(*(ring[(k + 1) % k_ring] - ring[k]))) for k in range(k_ring)]
    order = sorted(range(k_ring), key=lambda k: -lens[k])[:edges]
    types = [(u_, 90) for u_ in (4, 3, 2)] + [(u_, a) for u_ in (4, 3, 2) for a in CCDC_ONE_WAY_ANGLES]
    for i in order:
        L = lens[i]
        a = ring[i]
        u = (ring[(i + 1) % k_ring] - a) / L
        n = np.array([-u[1], u[0]])                     # the inside is on the left
        cands = {}
        for ucf, ang in types:
            mw = CCDC_LAYOUT[(ucf, ang)][1]
            R = L - 2 * mw
            if R <= 0 or storey / 2 / R > PARK_RAMP_MAX:
                continue
            mods = 0
            while mods < 8 and _depth_fits(ring, a, u, n, L, mods * mw, (mods + 1) * mw):
                mods += 1
            if mods < 2:
                continue
            for sch, rise in (("one_bay", storey), ("two_ramps", storey), ("both_bays", storey / 2),
                              ("double_thread", storey)):
                g = rise / R
                if g > PARK_RAMP_MAX + 1e-9 or sch not in schemes:
                    continue
                if ang != 90 and sch in ("one_bay", "two_ramps"):
                    continue                            # p.20: those are two-way; one-way never at 90 deg
                if sch == "two_ramps" and (mods < 3 or L < LONG_GARAGE_M):
                    continue
                if sch == "both_bays" and floors > SINGLE_THREAD_MAX_FLOORS:
                    continue
                if sch == "double_thread" and floors < DOUBLE_THREAD_MIN_FLOORS:
                    continue
                cands.setdefault(sch, []).append((g, ucf, ang, mw, mods))
        if cands:
            pref = {s: [c for c in v if c[0] <= PARK_RAMP_PREF + 1e-9] for s, v in cands.items()}
            if any(pref.values()):
                cands = {s: v for s, v in pref.items() if v}
            names = sorted(cands)
            sch = names[int(_hash01(seed, "garage scheme") * len(names)) % len(names)]
            opts = cands[sch]
            g, ucf, ang, mw, mods = opts[int(_hash01(seed, "garage module") * len(opts)) % len(opts)]
            strips, west, east, flat = _scheme_strips(sch, floors, mods, mw)
            return GarageLayout(i, a, u, n, mw, L - mw, L, mods, g, sch, ang, ucf, mw, strips, west, east, flat)
    for cap in (SPEED_RAMP_MAX, SPEED_RAMP_MAX2) if "speed_ramp" in schemes else ():   # a speed ramp (p.18)
        for i in order:
            L = lens[i]
            s0, s1 = SPEED_END_M, L - SPEED_END_M
            if s1 - s0 < 3 * TRANSITION_M:
                continue
            prof = _ramp_profile(s0, s1, storey)
            g = max((f1 - f0) * storey / (t1 - t0) for (t0, f0), (t1, f1) in zip(prof, prof[1:]))
            if g > cap + 1e-9:
                continue
            a = ring[i]
            u = (ring[(i + 1) % k_ring] - a) / L
            n = np.array([-u[1], u[0]])
            ucf = (4, 3, 2)[int(_hash01(seed, "garage module") * 3) % 3]
            mw = CCDC_LAYOUT[(ucf, 90)][1]
            rw = END_AISLE_M
            if not (_depth_fits(ring, a, u, n, L, 0.0, rw) and _depth_fits(ring, a, u, n, L, rw, rw + mw)):
                continue
            strips, west, east, flat = _scheme_strips("speed_ramp", floors, 1, mw, rw)
            return GarageLayout(i, a, u, n, s0, s1, L, 1, g, "speed_ramp", 90, ucf, mw, strips, west, east, flat)
    return None


def _module_layout(ring, storey, floors=4, seed=0.0):
    """(the earlier name)"""
    return _garage_layout(ring, storey, floors, seed)


def _clip(poly, p, nrm, c):
    """Sutherland-Hodgman: the part of poly with (x - p) . nrm >= c."""
    out = []
    k = len(poly)
    for j in range(k):
        a, b = poly[j], poly[(j + 1) % k]
        fa, fb = float((a - p) @ nrm) - c, float((b - p) @ nrm) - c
        if fa >= 0:
            out.append(a)
        if (fa >= 0) != (fb >= 0):
            out.append(a + (b - a) * (fa / (fa - fb)))
    return out


def _slab(m, top, thick, mat, uvp):
    """A closed slab: the planar polygon top ((k, 3), counter-clockwise from above) facing up, the same thick lower
    facing down, a side quad per edge (each edge shared by exactly two of its triangles)."""
    top = [tuple(float(c) for c in p) for p in top]
    bot = [(x, y, z - thick) for x, y, z in top]
    m.poly(top, [uvp(p) for p in top], mat)
    m.poly(bot[::-1], [uvp(p) for p in bot[::-1]], mat)
    for j in range(len(top)):
        k = (j + 1) % len(top)
        m.quad(bot[j], bot[k], top[k], top[j], (0, 0), (1, 0), (1, 0.1), (0, 0.1), mat)


def _sloped_wall(m, p, q, zp, zq, h, mat, t=0.2, sink=0.1):
    """A closed wall from p (ground zp) to q (zq), h over that line and sink under it, t thick: a parapet along a
    ramp (its top follows the ramp)."""
    d = q - p
    L = float(np.hypot(*d))
    if L < 0.05:
        return
    v = np.array([-d[1], d[0]]) / L * (t / 2)
    c = [p - v, q - v, q + v, p + v]           # counter-clockwise from above
    zb = [zp - sink, zq - sink, zq - sink, zp - sink]
    zt = [zp + h, zq + h, zq + h, zp + h]
    uv = ((0.002, 0.995), (0.012, 0.995), (0.012, 0.999), (0.002, 0.999))
    for j in range(4):
        k = (j + 1) % 4
        m.quad((*c[j], zb[j]), (*c[k], zb[k]), (*c[k], zt[k]), (*c[j], zt[j]), *uv, mat)
    m.quad(*[(*c[j], zt[j]) for j in range(4)], *uv, mat)
    m.quad(*[(*c[j], zb[j]) for j in (3, 2, 1, 0)], *uv, mat)


def _valid_poly(poly):
    """A simple polygon the ear clipping triangulates whole (Sutherland-Hodgman leaves a non-convex ring's parts
    joined by bridges along the cut line: not simple)."""
    k = len(poly)
    if k < 3:
        return False
    tri = triangulate(poly)
    if len(tri) != k - 2:
        return False
    tri = np.asarray(tri, np.int64).reshape(-1, 3)
    a = poly[tri]
    e1, e2 = a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]
    tot = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum() / 2
    return abs(tot - abs(_area2(poly)) / 2) < 1e-3 * max(1.0, tot)


def _clip_all(ring, planes):
    """The ring cut by the half-planes [(nrm, c)] ((x - A) . nrm >= c with A = planes' origin): [polygon], one
    simple polygon when that is, else the convex parts of the ring's triangles."""
    A, cuts = planes
    pts = [np.asarray(p, np.float64) for p in ring]
    p = pts
    for nrm, c in cuts:
        p = _clip(p, A, nrm, c)
    p = _dedup(p)
    if len(p) >= 3 and _valid_poly(np.array(p)):
        return [np.array(p)]
    out = []
    for a, b, c_ in np.asarray(triangulate(np.array(pts)), np.int64).reshape(-1, 3):
        q = [pts[a], pts[b], pts[c_]]
        for nrm, c in cuts:
            q = _clip(q, A, nrm, c)
        q = _dedup(q)
        if len(q) >= 3 and abs(_area2(np.array(q))) > 1e-4:
            out.append(np.array(q))
    return out


def _dedup(p, tol=1e-3):
    """The polygon without consecutive near-duplicate points."""
    out = []
    for q in p:
        if not out or float(np.hypot(*(q - out[-1]))) > tol:
            out.append(q)
    while len(out) > 1 and float(np.hypot(*(out[0] - out[-1]))) <= tol:
        out.pop()
    return out


def _garage_pieces(ring, lay):
    """The flat deck pieces of a GarageLayout: [(name, [polygon (k, 2)], half-levels)]: the end bays and the
    middle's parts between the sloped strips (and beyond them)."""
    A, u, n, s0, s1 = lay.A, lay.u, lay.n, lay.s0, lay.s1
    out = [("west", _clip_all(ring, (A, [(-u, -s0)])), lay.west), ("east", _clip_all(ring, (A, [(u, s1)])), lay.east)]
    sloped = sorted((d0, d1) for d0, d1, sl in lay.strips if sl is not None)
    gaps, d = [], 0.0
    for d0, d1 in sloped:
        if d0 > d + 1e-6:
            gaps.append((d, d0))
        d = max(d, d1)
    gaps.append((d, None))
    for j, (g0, g1) in enumerate(gaps):
        cuts = [(u, s0), (-u, -s1), (n, g0)] + ([(-n, -g1)] if g1 is not None else [])
        out.append((f"mid{j}", _clip_all(ring, (A, cuts)), lay.flat))
    return [(nm, [p for p in ps if abs(_area2(p)) > 1e-3], lv) for nm, ps, lv in out]


def _on_outline(ring, p, q, tol=1e-3):
    """The segment p-q along an edge of the ring."""
    k = len(ring)
    for j in range(k):
        a, b = ring[j], ring[(j + 1) % k]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        ok = True
        for x in (p, q, (p + q) / 2):
            t = float(np.clip((x - a) @ ab / L2, 0.0, 1.0))
            if float(np.hypot(*(x - a - ab * t))) > tol:
                ok = False
                break
        if ok:
            return True
    return False


def _surface_at(lay, s_end, d, h):
    """Whether a sloped strip at depth d ends (s_end 0: at s0, 1: at s1) at half-level h, or a flat part of the
    middle lies there at h."""
    for d0, d1, sl in lay.strips:
        if d0 - 1e-6 <= d <= d1 + 1e-6 and sl is not None:
            return any((hb if s_end else ha) == h for ha, hb in sl)
    return h in lay.flat


def garage_levels(lay, storey, z0):
    """The height of each half-level."""
    return lambda h: z0 + h * storey / 2


def _module_floors(m, ring, floors, storey, lay, z0, mat, uvp):
    """The drivable structure of a CCDC garage (closed slabs and parapets): the whole ground deck; each flat piece
    (_garage_pieces) at each of its half-levels; each sloped strip's slab per storey (with transitions on a speed
    ramp); low parapets round the outline, along the sloped strips and wherever a deck piece ends at a drop.
    Returns the decks' heights (whole storeys)."""
    A, u, n, s0, s1 = lay.A, lay.u, lay.n, lay.s0, lay.s1
    zh = garage_levels(lay, storey, z0)
    zs = [zh(2 * k) for k in range(floors + 1)]
    top_h = 2 * floors

    def at(s, d):
        return A + u * s + n * d

    def sd(p):
        r = p - A
        return float(r @ u), float(r @ n)

    _slab(m, [(p[0], p[1], zs[0]) for p in ring], DECK_SLAB_M, mat, uvp)
    for name, polys, levels in _garage_pieces(ring, lay):
        for poly in polys:
            for h in levels:
                if h <= 0 or h > top_h:
                    continue
                z = zh(h)
                _slab(m, [(p[0], p[1], z) for p in poly], DECK_SLAB_M, mat, uvp)
                k = len(poly)
                for j in range(k):
                    p, q = poly[j], poly[(j + 1) % k]
                    (sp, dp), (sq, dq) = sd(p), sd(q)
                    inward = np.array([-(q - p)[1], (q - p)[0]])
                    inward = inward / max(float(np.hypot(*inward)), 1e-9) * 0.1
                    if abs(sp - s0) < 1e-4 and abs(sq - s0) < 1e-4 or abs(sp - s1) < 1e-4 and abs(sq - s1) < 1e-4:
                        end = 0 if abs(sp - s0) < 1e-4 else 1
                        s_line = s0 if end == 0 else s1
                        cuts = sorted({dp, dq} | {c for d0, d1, _ in lay.strips for c in (d0, d1)
                                                  if min(dp, dq) < c < max(dp, dq)})
                        for c0, c1 in zip(cuts, cuts[1:]):
                            mid_d = (c0 + c1) / 2
                            if name in ("west", "east"):
                                ok = _surface_at(lay, end, mid_d, h)
                            else:
                                ok = h in (lay.west if end == 0 else lay.east)
                            if not ok:
                                _sloped_wall(m, at(s_line, c0) + inward, at(s_line, c1) + inward, z, z,
                                             DECK_PARAPET_M, mat)
                    elif abs(dp - dq) < 1e-4 and s0 - 1e-4 <= min(sp, sq) and max(sp, sq) <= s1 + 1e-4 \
                            and any(abs(dp - c) < 1e-4 for d0, d1, sl in lay.strips if sl is not None for c in (d0, d1)):
                        _sloped_wall(m, p + inward, q + inward, z, z, DECK_PARAPET_M, mat)      # beside a ramp
                    elif _on_outline(ring, p, q):
                        _sloped_wall(m, p, q, z, z, DECK_PARAPET_M, mat)                    # round the outline
    for d0, d1, sl in lay.strips:                       # the sloped strips: a slab per storey, parapets both sides
        if sl is None:
            continue                                    # (part of the flat middle pieces)
        for ha, hb in sl:
            za, zb = zh(ha), zh(hb)
            prof = _ramp_profile(s0, s1, 1.0) if lay.scheme == "speed_ramp" else [(s0, 0.0), (s1, 1.0)]
            for (ta, fa), (tb, fb) in zip(prof, prof[1:]):
                z_a, z_b = za + (zb - za) * fa, za + (zb - za) * fb
                _slab(m, [(*at(ta, d0), z_a), (*at(tb, d0), z_b), (*at(tb, d1), z_b), (*at(ta, d1), z_a)],
                      DECK_SLAB_M, mat, uvp)
                for d in (d0 + RAMP_PARAPET_IN_M, d1 - RAMP_PARAPET_IN_M):
                    _sloped_wall(m, at(ta, d), at(tb, d), z_a, z_b, DECK_PARAPET_M, mat)
    return zs


def module_columns(ring, lay):
    """The interior columns' centres: on each line between modules (between back-to-back stall rows), about every
    COLUMN_EVERY_STALLS stalls (p.63) from s0 to s1; never in an aisle, none in the end bays."""
    wp = lay.dims[0]
    own = {c for d0, d1, _ in lay.strips for c in (d0, d1)}
    if lay.scheme == "speed_ramp":                  # its own edge is the ramp's parapet; its modules beyond
        own = {lay.strips[0][1] + j * lay.mw for j in range(1, lay.modules + 1)}
    beyond = {max(own) + j * lay.mw for j in range(1, 8)}         # the module grid carried on into an L's leg
    out = []
    nseg = max(1, int(round((lay.s1 - lay.s0) / (COLUMN_EVERY_STALLS * wp))))
    for c in sorted(own | beyond):
        if c < 1e-6:
            continue
        ts = np.linspace(lay.s0, lay.s1, nseg + 1)
        for t in (ts if c in own else ts[1:-1]):    # (the leg's columns clear of the end bays' edges)
            p = lay.A + lay.u * t + lay.n * c
            if _inside(ring, p) and _inside(ring, p + lay.n * 1.0) and _inside(ring, p - lay.n * 1.0):
                out.append(p)
    return out


def garage_stalls(lay):
    """The stall lines of a GarageLayout's middle strips (for plans and tests): [(p, q)] each a stall's side line,
    from the module's outer edge across its vehicle projection, at its angle; and the aisles [(d0, d1)]."""
    wp, mw, vp, aw = lay.dims
    lines, aisles = [], []
    rows = [(d0, d1, 1.0 if not sl or sl[0][1] >= sl[0][0] else -1.0) for d0, d1, sl in lay.strips
            if d1 - d0 > mw - 0.5]
    if lay.scheme == "speed_ramp":
        rows = [(lay.strips[0][1] + j * mw, lay.strips[0][1] + (j + 1) * mw, 1.0) for j in range(lay.modules)]
    ang = math.radians(lay.angle)
    for d0, d1, lean in rows:                       # (angled stalls lean with their one-way traffic)
        aisles.append((d0 + vp, d1 - vp))
        for side, e, sgn in ((0, d0, 1.0), (1, d1, -1.0)):
            s = lay.s0 + ROW_BUFFER_M
            while s + wp <= lay.s1 - ROW_BUFFER_M:
                p = lay.A + lay.u * s + lay.n * e
                tip = p + lay.n * (sgn * vp) + lay.u * (lean * vp / math.tan(ang) if lay.angle != 90 else 0.0)
                lines.append((p, tip))
                s += wp
    return lines, aisles


def _module_deck(m, spec, gmin, gmax, mat_idx, ring, floors, storey, lay, paint=False):
    """The CCDC garage: its floors, skin, perimeter columns and interior columns on the module lines between
    back-to-back stall rows (never in an aisle, none in the end bays where the cars turn)."""
    mat = mat_idx[f"{PANEL}|{len(WALL_TINTS[PANEL]) - 1 - int(_hash01(spec.seed, 'deck') * 2)}"]

    def uvp(p):
        return (float(p[0]) / 4.0, float(p[1]) / 4.0)

    k_ring = len(ring)
    zs = _module_floors(m, ring, floors, storey, lay, gmax + 0.03, mat, uvp)
    if gmax - gmin > 0.05:                      # the ground deck's plinth down to the lowest ground
        for j in range(k_ring):
            p, q = ring[j], ring[(j + 1) % k_ring]
            m.quad((p[0], p[1], gmin - FOUNDATION_M), (q[0], q[1], gmin - FOUNDATION_M),
                   (q[0], q[1], zs[0] - DECK_SLAB_M), (p[0], p[1], zs[0] - DECK_SLAB_M),
                   (0, 0), (1, 0), (1, 0.1), (0, 0.1), mat)
    _garage_skin(m, spec, ring, zs, lay.i, mat_idx)
    for j in range(k_ring):                     # perimeter columns
        p, q = ring[j], ring[(j + 1) % k_ring]
        L = float(np.hypot(*(q - p)))
        if L < 0.5:
            continue
        d = (q - p) / L
        inward = np.array([-d[1], d[0]])
        for t in np.arange(0.0, L, DECK_COLUMN_M):
            _box(m, p + d * t + inward * 0.35, d, 0.25, 0.25, zs[0] - FOUNDATION_M, zs[-1], mat)
    for c in module_columns(ring, lay):
        _box(m, c, lay.u, 0.25, 0.25, zs[0] - FOUNDATION_M, zs[-1], mat)
    if paint:                                   # (drawn only: collision_mesh passes paint=False)
        _paint(m, ring, lay, garage_levels(lay, storey, gmax + 0.03), mat_idx[LINE_PAINT])


def garage_floors(height):
    """(floors, storey) for a mapped height: the standard's floor-to-floor (STOREY_STD_M, p.21), the floor count
    fitted to the mapped height; the storey stretched up to STOREY_MAX_M where the count would fall short."""
    floors = max(1, int(round(height / STOREY_STD_M)))
    storey = STOREY_STD_M
    if floors * storey < height - STOREY_STD_M / 2 and height / floors <= STOREY_MAX_M:
        storey = height / floors
    return floors, storey


def deck_plan(spec, gmin, gmax):
    """(counter-clockwise ring, floors, storey height, layout) of a parking structure drawn as open decks, or None
    (not a garage, ground too uneven, the ramp does not fit: a closed box). The layout is a GarageLayout (CCDC
    modules, see SCHEMES) or the old switchback (_deck_layout) on footprints too small for any of them."""
    if not spec.deck or gmax - gmin > DECK_LEVEL_M or not spec.height or (spec.min_h or 0.0) > 0.5:
        return None
    floors, storey = garage_floors(spec.height)
    ring = np.asarray(spec.ring, np.float64)
    if _area2(ring) < 0:
        ring = ring[::-1]
    simple = _merge_collinear(ring)
    mod = _garage_layout(simple, storey, floors, spec.seed)
    if mod is not None:
        return simple, floors, storey, mod
    lay = _deck_layout(ring, storey)            # too small for that: the two-lane switchback along one wall
    return None if lay is None else (ring, floors, storey, lay)


# Garage paint (2026-10-28): the stall lines garage_stalls plans, drawn as the roads draw their lane lines: thin quads
# in the stock line_white decal material (u 0..1 across, v = length / its texture length), lying PAINT_LIFT_M over the
# slab's top as roadmesh lays its markings DECK_SKIN over a bridge deck (no z-fighting), on every level the stall row
# has a floor, following a sloped floor's plane. Drawn only (build); collision_mesh never gets them.
PAINT_W_M = 0.10            # a stall line (4")
PAINT_LIFT_M = 0.01         # m over the slab (roadmesh.DECK_SKIN)
PAINT_TL_M = 6.4            # m of texture along (roads.EDGE_LINE)
PAINT_CLEAR_M = 0.65        # stall lines start this far in from a module line (the columns: 0.25 half width there,
                            # perimeter columns 0.35 + 0.25 in from the outline; parapets 0.15 + 0.1)
ARROW_LEN_M = 3.0           # one-way aisles: an arrow down the aisle's centre ...
ARROW_HEAD_M = 1.0          # ... its head strokes this long ...
ARROW_EVERY_M = 20.0        # ... about this far apart


def garage_paint(ring, lay):
    """The garage's painted lines in plan with the floors they lie on: [(p (2,), q (2,), [(ha, hb) | (h, h)])] per
    line, each floor a pair of half-levels at s0 and s1 (a sloped strip's slab; equal for a flat floor). Stall side
    lines (each row's last stall closed too), PAINT_CLEAR_M clear of the module lines (columns, parapets), kept
    inside the outline; one-way schemes (angled stalls) an arrow every ARROW_EVERY_M down each aisle's centre,
    pointing with the traffic. Never across an aisle: a stall line ends at the aisle's edge (the vehicle projection)."""
    wp, mw, vp, aw = lay.dims
    flat_floors = [(0, 0)] + [(h, h) for h in lay.flat if h > 0]
    if lay.scheme == "speed_ramp":
        r1 = lay.strips[0][1]
        rows = [(r1 + j * mw, r1 + (j + 1) * mw, 1.0, flat_floors) for j in range(lay.modules)]
    else:
        rows = [(d0, d1, 1.0 if not sl or sl[0][1] >= sl[0][0] else -1.0, list(sl) if sl else flat_floors)
                for d0, d1, sl in lay.strips if d1 - d0 > mw - 0.5]
    ang = math.radians(lay.angle)
    out = []

    def keep(p, q, floors):
        sp, sq = float((p - lay.A) @ lay.u), float((q - lay.A) @ lay.u)
        if min(sp, sq) < lay.s0 - 1e-6 or max(sp, sq) > lay.s1 + 1e-6:
            return                                  # (an angled row's last lines: not out into an end bay)
        if _inside(ring, p) and _inside(ring, q) and _inside(ring, (p + q) / 2):
            out.append((p, q, floors))

    for d0, d1, lean, floors in rows:
        for e, sgn in ((d0, 1.0), (d1, -1.0)):
            n = int((lay.s1 - lay.s0 - 2 * ROW_BUFFER_M) // wp)
            for k in range(n + 1):
                s = lay.s0 + ROW_BUFFER_M + k * wp
                p = lay.A + lay.u * s + lay.n * e
                tip = p + lay.n * (sgn * vp) + lay.u * (lean * vp / math.tan(ang) if lay.angle != 90 else 0.0)
                p = p + (tip - p) * (PAINT_CLEAR_M / vp)
                keep(p, tip, floors)
        if lay.angle != 90:                         # one-way: arrows with the traffic (the stalls' lean)
            c = (d0 + d1) / 2
            for s in np.arange(lay.s0 + ARROW_EVERY_M / 2, lay.s1 - ARROW_LEN_M, ARROW_EVERY_M):
                a = lay.A + lay.u * s + lay.n * c
                b = a + lay.u * (lean * ARROW_LEN_M)
                for side in (1.0, -1.0):
                    back = b - lay.u * (lean * ARROW_HEAD_M * 0.7) + lay.n * (side * ARROW_HEAD_M * 0.7)
                    keep(back, b, floors)
                keep(a, b, floors)
    return out


def _paint(m, ring, lay, zh, mat):
    """garage_paint's lines as quads PAINT_LIFT_M over each of their floors (a sloped slab's plane: z linear in s
    from s0 to s1)."""
    for p, q, floors in garage_paint(ring, lay):
        d = q - p
        L = float(np.hypot(*d))
        if L < 1e-6:
            continue
        w = np.array([-d[1], d[0]]) / L * (PAINT_W_M / 2)
        corners = (p - w, q - w, q + w, p + w)          # counter-clockwise from above
        uv = ((1.0, 0.0), (1.0, L / PAINT_TL_M), (0.0, L / PAINT_TL_M), (0.0, 0.0))
        for ha, hb in floors:
            za, zb = zh(ha), zh(hb)
            pts = []
            for c in corners:
                f = min(1.0, max(0.0, float((c - lay.A) @ lay.u - lay.s0) / max(lay.s1 - lay.s0, 1e-9)))
                pts.append((float(c[0]), float(c[1]), za + (zb - za) * f + PAINT_LIFT_M))
            m.quad(*pts, *uv, mat)


def _parking_deck(m, spec, gmin, gmax, mat_idx, paint=False):
    """The open parking structure (see DECK_*); False when it does not apply (the caller draws the box)."""
    plan_ = deck_plan(spec, gmin, gmax)
    if plan_ is None:
        return False
    if isinstance(plan_[3], GarageLayout):
        _module_deck(m, spec, gmin, gmax, mat_idx, *plan_, paint=paint)
        return True
    ring, floors, storey, (i, A, u, n, s0, s1) = plan_
    # weathered concrete (Baltimore's street photos: the white decks read as a skeleton; city garages are grey)
    mat = mat_idx[f"{PANEL}|{len(WALL_TINTS[PANEL]) - 1 - int(_hash01(spec.seed, 'deck') * 2)}"]
    W, e = DECK_LANE_M, DECK_EDGE_M
    depth = e + 2 * W
    zs = [gmax + 0.03 + k * storey for k in range(floors + 1)]

    def at(s, d):
        return A + u * s + n * d

    def uvp(p):
        return (float(p[0]) / 4.0, float(p[1]) / 4.0)

    k_ring = len(ring)
    _slab(m, [(p[0], p[1], zs[0]) for p in ring], DECK_SLAB_M, mat, uvp)          # the ground deck, whole
    if gmax - gmin > 0.05:                      # its plinth down to the lowest ground (a sloping street)
        for j in range(k_ring):
            p, q = ring[j], ring[(j + 1) % k_ring]
            m.quad((p[0], p[1], gmin - FOUNDATION_M), (q[0], q[1], gmin - FOUNDATION_M),
                   (q[0], q[1], zs[0] - DECK_SLAB_M), (p[0], p[1], zs[0] - DECK_SLAB_M),
                   (0, 0), (1, 0), (1, 0.1), (0, 0.1), mat)
    # the upper decks: the ring with the ramp well (both lanes, s0..s1) cut in from the longest edge
    well = [at(s0, 0.0), at(s0, depth), at(s1, depth), at(s1, 0.0)]
    pts = [A] + well + [ring[(i + 1 + j) % k_ring] for j in range(k_ring - 1)]
    for k in range(1, floors + 1):
        z = zs[k]
        _slab(m, [(p[0], p[1], z) for p in pts], DECK_SLAB_M, mat, uvp)
        for j in range(len(pts)):
            if not (1 <= j <= 3):               # the parapet, not round the well (below)
                _sloped_wall(m, pts[j], pts[(j + 1) % len(pts)], z, z, DECK_PARAPET_M, mat)
        # round the well: its inner side and the end away from the ramps' landing (ramp k - 1 comes up at s1 when
        # k - 1 is even, and ramp k leaves from that same end); the top deck has no ramp leaving: its lane closed
        land = s1 if (k - 1) % 2 == 0 else s0
        far = s0 if land == s1 else s1
        _sloped_wall(m, at(s0, depth + 0.1), at(s1, depth + 0.1), z, z, DECK_PARAPET_M, mat)
        _sloped_wall(m, at(far, 0.0), at(far, depth), z, z, DECK_PARAPET_M, mat)
        if k == floors:
            d_free = e + (k % 2) * W
            _sloped_wall(m, at(land, d_free), at(land, d_free + W), z, z, DECK_PARAPET_M, mat)
    # the ramps: ramp k from deck k to deck k + 1 on lane k % 2, even ones climbing from s0 to s1, walled both sides,
    # with p.18's half-grade transitions at both ends
    prof = _ramp_profile(s0, s1, 1.0)
    for k in range(floors):
        d0 = e + (k % 2) * W
        za, zb = zs[k], zs[k + 1]
        if k % 2 == 1:                          # climbing from s1 to s0: the same profile mirrored
            segs = [(s0 + s1 - tb, s0 + s1 - ta, za + (zb - za) * fb, za + (zb - za) * fa)
                    for (ta, fa), (tb, fb) in zip(prof, prof[1:])]
        else:
            segs = [(ta, tb, za + (zb - za) * fa, za + (zb - za) * fb) for (ta, fa), (tb, fb) in zip(prof, prof[1:])]
        for ta, tb, z_a, z_b in segs:
            _slab(m, [(*at(ta, d0), z_a), (*at(tb, d0), z_b), (*at(tb, d0 + W), z_b), (*at(ta, d0 + W), z_a)],
                  DECK_SLAB_M, mat, uvp)
            for d in (d0 + 0.1, d0 + W - 0.1):
                _sloped_wall(m, at(ta, d), at(tb, d), z_a, z_b, DECK_PARAPET_M, mat, 0.15)
    _garage_skin(m, spec, ring, zs, i, mat_idx)
    # the columns: round the outline every DECK_COLUMN_M, from the foundation to the top deck
    for j in range(k_ring):
        p, q = ring[j], ring[(j + 1) % k_ring]
        L = float(np.hypot(*(q - p)))
        if L < 0.5:
            continue
        d = (q - p) / L
        inward = np.array([-d[1], d[0]])
        for t in np.arange(0.0, L, DECK_COLUMN_M):
            _box(m, p + d * t + inward * 0.35, d, 0.25, 0.25, zs[0] - FOUNDATION_M, zs[-1], mat)
    return True


# A garage's skin (the user's reports #51/#52, 2026-10-17: "Garages look ugly and basic", "all garages look the
# same"; Canton's on Mapillary: brick or precast concrete spandrels with long openings between, solid corners, a
# stair tower over the roof): per garage (hashed) one of GARAGE_SKINS
GARAGE_SKINS = ("bne_brick_red", "bne_brick_dark", "bne_brick_tan", "bne_precast", "bne_precast_warm")
SPANDREL_H = 1.1            # m of spandrel over each deck (the slab's edge behind it)
SPANDREL_T = 0.3            # m thick, its face just outside the outline
PIER_W = 0.9                # m: the solid corners and every PIER_EVERY_M along the outline
PIER_EVERY_M = 18.0
STAIR_TOWER_M = 4.5         # m square, at the outline's corner farthest from the ramps, ...
STAIR_TOWER_UP = 3.2        # ... this far over the top deck


def _garage_skin(m, spec, ring, zs, ramp_edge, mat_idx):
    """The facade of an open parking structure: spandrels round its outline at every deck above the ground,
    piers at its corners and along it, a stair tower; brick or precast concrete by its seed (GARAGE_SKINS)."""
    mat = mat_idx[GARAGE_SKINS[int(_hash01(spec.seed, "skin") * len(GARAGE_SKINS)) % len(GARAGE_SKINS)]]
    k_ring = len(ring)
    for j in range(k_ring):
        p, q = ring[j], ring[(j + 1) % k_ring]
        L = float(np.hypot(*(q - p)))
        if L < 0.5:
            continue
        d = (q - p) / L
        out = np.array([d[1], -d[0]])           # (counter-clockwise ring: the outside on the right)
        c = (p + q) / 2 + out * (SPANDREL_T / 2 - 0.05)
        for z in zs[1:]:
            _box_tiled(m, c, d, L / 2, SPANDREL_T / 2, z - DECK_SLAB_M - 0.1, z + SPANDREL_H, mat, 2.0)
        for t in [0.0] + list(np.arange(PIER_EVERY_M, L - PIER_EVERY_M / 2, PIER_EVERY_M)):
            _box_tiled(m, p + d * t + out * 0.05, d, PIER_W / 2, PIER_W / 2, zs[0] - FOUNDATION_M,
                       zs[-1] + SPANDREL_H + 0.2, mat, 2.0)
    # the stair tower: at the ring's vertex farthest from the ramps' edge
    a, b = ring[ramp_edge], ring[(ramp_edge + 1) % k_ring]
    ab = b - a
    L2 = max(float(ab @ ab), 1e-9)

    def dist(v):
        t = float(np.clip(((v - a) @ ab) / L2, 0.0, 1.0))
        return float(np.hypot(*(v - (a + t * ab))))
    jv = max(range(k_ring), key=lambda j: dist(ring[j]))
    v = ring[jv]
    inward = ring.mean(0) - v
    inward = inward / max(float(np.hypot(*inward)), 1e-9)
    c = v + inward * (STAIR_TOWER_M * 0.75)
    if _inside(ring, c):
        e = ring[(jv + 1) % k_ring] - v
        e = e / max(float(np.hypot(*e)), 1e-9)
        _box_tiled(m, c, e, STAIR_TOWER_M / 2, STAIR_TOWER_M / 2, zs[0] - FOUNDATION_M, zs[-1] + STAIR_TOWER_UP,
                   mat, 2.0)


def garage_drives(items, lines, x0, y0):
    """Driveway chunks (road tile chunk dicts, draped by roadmesh) into each open parking structure: from the
    middle of its outline's edge nearest a road out to that road's edge (report #51, 2026-10-17: "No ramp leading
    into the garage")."""
    out = []
    if not lines:
        return out
    segs = []
    for ln in lines:
        pts, half = np.asarray(ln[0], np.float64)[:, :2], ln[1]
        for k in range(len(pts) - 1):
            segs.append((pts[k], pts[k + 1], half))
    if not segs:
        return out
    A = np.array([s[0] for s in segs])
    B = np.array([s[1] for s in segs])
    H = np.array([s[2] for s in segs])
    D = B - A
    L2 = np.maximum((D * D).sum(1), 1e-12)
    for spec, gmin, gmax in items:
        plan_ = deck_plan(spec, gmin, gmax)
        if plan_ is None:
            continue
        ring = plan_[0] + np.array([x0, y0])
        best = None
        for j in range(len(ring)):
            p, q = ring[j], ring[(j + 1) % len(ring)]
            if np.hypot(*(q - p)) < 2 * DRIVEWAY_W:
                continue
            mid = (p + q) / 2
            t = np.clip(((mid - A) * D).sum(1) / L2, 0.0, 1.0)
            near = A + t[:, None] * D
            dd = np.hypot(*(mid - near).T) - H
            k = int(np.argmin(dd))
            if best is None or dd[k] < best[0]:
                best = (float(dd[k]), mid, near[k], H[k], p, q)
        if best is None or not (0.5 < best[0] < GARAGE_DRIVE_MAX_M):
            continue
        _, mid, qn, half, p, q = best
        d = (q - p) / max(float(np.hypot(*(q - p))), 1e-9)
        inside = np.array([-d[1], d[0]])
        v = mid - qn
        dl = float(np.hypot(*v))
        p0 = mid + inside * 3.0             # a few metres in, onto the ground deck
        p1 = qn + v / max(dl, 1e-9) * (half + 0.1)
        n = max(2, int(np.hypot(*(p1 - p0)) // 4.0) + 1)
        pts = [p0 + (p1 - p0) * (j / (n - 1)) for j in range(n)]
        out.append({"m": DRIVEWAY["m"], "w": GARAGE_DRIVE_W, "tl": DRIVEWAY["tl"], "p": DRIVEWAY["p"],
                    "n": [[round(float(pp[0]), 3), round(float(pp[1]), 3), 0.0] for pp in pts]})
    return out


GARAGE_DRIVE_W = 6.5        # m: a garage entrance's two lanes
GARAGE_DRIVE_MAX_M = 25.0   # m from the outline to a road's edge at most


def build(specs_and_ground, lite=False):
    """The tile's mesh from [(Spec, ground z min, ground z max)] with rings in the tile's local frame:
    (positions, uvs, tris, tri material indices, MATERIALS) or None if empty. lite (far tiles): no
    cornices, whole walls instead of one quad per bay."""
    m = Mesh()
    for spec, gmin, gmax in specs_and_ground:
        if spec.cls == "toll_booth":        # a stock prop (toll_booth_rows)
            continue
        mat_idx = _Looks(spec.seed)
        if spec.cls in BRIDGE_CLASSES:
            # a tower the decks pass (bridge_tower): its legs only; the bridge's own structure draws its portal beams
            # over the decks (a frame's bracing crossed the Vincent Thomas Bridge's lanes at truck height, drawn
            # but not solid, 2026-10-28)
            _steel_frame(m, spec, gmin, mat_idx[spec.facade if spec.facade in LOOKS else STEEL], lite,
                         legs_only=spec.cls == "bridge_tower")
            continue
        if spec.cls in LANDMARK_CLASSES:
            {"ship": _ship, "dome": _dome, "tank": _tank, "vault": _vault}[spec.cls](m, spec, gmin, gmax, mat_idx, lite)
            continue
        ring = drawn_ring(spec)
        mh = spec.min_h or 0.0
        if spec.deck and not lite and _parking_deck(m, spec, gmin, gmax, mat_idx, paint=True):
            continue
        z0 = gmin - (SKIRT_LITE_M if lite else SKIRT_M) if mh <= 0.5 else base_z(spec, gmin, gmax) + mh
        z_eave = base_z(spec, gmin, gmax) + mh + spec.height
        _walls(m, spec, ring, z0, z_eave, mat_idx, lite)
        if mh > 0.5:                                  # a part standing on another / over a street: its underside
            m.poly([(p[0], p[1], z0) for p in ring[::-1]], [(0.002, 0.995)] * len(ring), mat_idx[spec.facade])
        _roof(m, spec, ring, z_eave, mat_idx)
        if spec.roof == "flat" and not spec.plain and spec.floors <= 6 and not lite:
            _cornice(m, spec, ring, z_eave, mat_idx)
        if not lite:
            _clutter(m, spec, ring, z_eave, mat_idx)
        if spec.steeple:
            _steeple(m, spec, z_eave, mat_idx)
        if spec.stoop and not lite:
            _stoop(m, spec, ring, gmin, gmax, mat_idx)
        if spec.porch and spec.door and not lite:
            _porch(m, spec, gmax, mat_idx)
        if spec.sign is not None and spec.roof == "flat" and not lite:
            if spec.windows is False:           # a big box: over its storefront band (_plain_wall_with_windows)
                z_top = min(gmax + GROUND_FLOOR_SHOP_M + 1.15, z_eave)
            else:
                z_top = z_eave - (spec.floors - 1) * spec.floor_h if spec.floors > 1 else z_eave
            _signs(m, spec, ring, z_top, mat_idx)
    if m.n == 0:
        return None
    P, UV, T, M, C = m.arrays()
    return P, UV, T, M, MATERIALS, C


def _steel_frame(m, spec, gmin, mat, lite=False, legs_only=False):
    """A bridge structure (BRIDGE_CLASSES) as an open steel frame on its oriented box: a leg at each corner, bracing
    along the short sides every STEEL_BAY_M (beside the decks running along them), cross beams over the long sides
    only near the top (STEEL_TOP_SHARE, over any deck); a narrow footprint is one pier. lite: legs and top only."""
    c, u, hl, hw = oriented_box(np.asarray(spec.ring, np.float64))
    v = np.array([-u[1], u[0]])
    z0, z1 = gmin - FOUNDATION_M, gmin + spec.height
    if hl < 6.0:                                            # a pier
        _box(m, c, u, hl, hw, z0, z1, mat)
        return
    if hw < 3.0:    # a thin wall across the decks (the Queensboro Bridge's towers, 33 x 4.6 m): a leg at each end
        t = float(min(hw, STEEL_LEG_MAX))
        for sx in (-1, 1):
            _box(m, c + u * sx * (hl - t), u, t, hw, z0, z1, mat)
        if not legs_only:
            b = t * 0.6
            top = gmin + STEEL_TOP_SHARE * spec.height
            for z in [z1 - b] + ([] if lite else list(np.arange(z1 - b - STEEL_BAY_M / 2, top, -STEEL_BAY_M / 2))):
                _box(m, c, u, hl - 2 * t, b, z - b, z + b, mat)
        return
    t = float(np.clip(hw * 0.5, 0.5, STEEL_LEG_MAX))        # leg half size
    corners = [c + u * sx * (hl - t) + v * sy * (hw - t) for sx in (-1, 1) for sy in (-1, 1)]
    for k in corners:
        _box(m, k, u, t, t, z0, z1, mat)
    if legs_only:
        return
    b = t * 0.6                                             # beam half size
    levels = [z1 - b] if lite else list(np.arange(gmin + STEEL_BAY_M, z1 - b, STEEL_BAY_M)) + [z1 - b]
    for z in levels:
        for sx in (-1, 1):          # along the short sides (along v): beside a deck running through the frame
            _box(m, c + u * sx * (hl - t), v, hw - 2 * t, b, z - b, z + b, mat)
        if z >= gmin + STEEL_TOP_SHARE * spec.height:       # across the top only
            for sy in (-1, 1):
                _box(m, c + v * sy * (hw - t), u, hl - 2 * t, b, z - b, z + b, mat)


def collision_mesh(specs_and_ground):
    """The tile's buildings for physics only: one quad per wall and a flat lid at the eaves (pitched roofs on their
    oriented box), parking decks as drawn; (positions, uvs, tris, tri materials, MATERIALS, colours) or None.
    The drawn mesh (windows a quad per bay and floor, cornices, signs) cost the game's collision rebuild ~30 ms of
    a 40 ms rebuild in Fells Point (2026-10-15), every few seconds while driving; this one is ~1/20 of its triangles."""
    m = Mesh()
    tok = (MATERIALS.index(PANEL), (128, 128, 128, 255))
    zero = (0.0, 0.0)
    for spec, gmin, gmax in specs_and_ground:
        if spec.cls in BRIDGE_CLASSES:          # its legs (the decks run between them)
            _steel_frame(m, spec, gmin, tok, True, legs_only=True)
            continue
        if deck_plan(spec, gmin, gmax) is not None:
            _parking_deck(m, spec, gmin, gmax, _Looks(spec.seed))
            continue
        ring = np.asarray(spec.ring, np.float64)
        if spec.roof != "flat":
            c, u, hl, hw = oriented_box(ring)
            v = np.array([-u[1], u[0]])
            ring = np.array([c - u * hl - v * hw, c + u * hl - v * hw, c + u * hl + v * hw, c - u * hl + v * hw])
        if _area2(ring) < 0:
            ring = ring[::-1]
        mh = spec.min_h or 0.0
        z0 = gmin - FOUNDATION_M if mh <= 0.5 else base_z(spec, gmin, gmax) + mh
        z1 = base_z(spec, gmin, gmax) + mh + (spec.floor_h if spec.cls in ("ship", "vault") else spec.height)
        # (a ship: its deck; an arched hangar: its walls under the arch)
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            m.quad((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1), zero, zero, zero, zero, tok)
        m.poly([(p[0], p[1], z1) for p in ring], [zero] * len(ring), tok)
        if mh > 0.5:                # over a street: its underside
            m.poly([(p[0], p[1], z0) for p in ring[::-1]], [zero] * len(ring), tok)
    if m.n == 0:
        return None
    P, UV, T, M, C = m.arrays()
    return P, UV, T, M, MATERIALS, C


# ------------------------------------------------------------------------------------------- a tile

def _shared_walls(rings, tol=0.6):
    """Per ring, the number of other rings with a vertex within tol of one of its edges (row houses)."""
    n = len(rings)
    cnt = np.zeros(n, np.int64)
    if n < 2:
        return cnt
    cents = np.array([r.mean(0) for r in rings])
    rads = np.array([np.hypot(*(r - r.mean(0)).T).max() for r in rings])
    order = np.argsort(cents[:, 0])
    fast.require(1, "shared_walls")
    offs = np.zeros(n + 1, np.int64)
    offs[1:] = np.cumsum([len(r) for r in rings])
    coords = np.ascontiguousarray(np.concatenate(rings)[:, :2], np.float64)
    return np.asarray(fast._ef.shared_walls(coords, offs, np.ascontiguousarray(cents, np.float64),
                                            np.ascontiguousarray(rads, np.float64), order.astype(np.int64),
                                            float(tol)), np.int64)

def _part_heights(a, i):
    """A building part's (own height, its bottom) from its attributes, or None without a height."""
    top = a["height"][i] if "height" in a else np.nan
    mh = a["min_height"][i] if "min_height" in a else np.nan
    mh = float(mh) if np.isfinite(mh) and mh > 0 else 0.0
    if not np.isfinite(top) or top <= mh + 1.0:
        nf, f0 = (a[k][i] if k in a else np.nan for k in ("num_floors", "min_floor"))
        if np.isfinite(nf) and nf > 0:
            f0 = float(f0) if np.isfinite(f0) and f0 > 0 else 0.0
            mh = mh or f0 * FLOOR_M
            return max(FLOOR_M, (float(nf) - f0) * FLOOR_M), mh
        return None
    return float(top) - mh, mh


ROAD_THROUGH_M = 15.0       # a footprint that public roads run through this far is a structure over them
OVER_ROAD_CLEAR_M = 5.5     # ... lifted this high (an expressway's or a station's slab over the streets: Baltimore's
                            # 17,500 m2 one by I-83, the AI drove into its wall at 35 mph, 2026-10-14)
ACROSS_W = 8.0              # m: a footprint at most this wide that a public road crosses (its long middle line) is a
                            # walkway or canopy over it (GitHub #48: the 69th Street station's 3 m shelters stood across
                            # the Dan Ryan as brick walls; a road crossing one runs only its width inside it)
THROUGH_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary", "residential", "unclassified",
                   "living_street"}


def tunnel_lines(frame, road_layers, x0, y0, margin=50.0, size=256.0):
    """Polylines (k, 2) relative to (x0, y0) of the road tunnels near the tile (roads.tunnel_parts; for
    road_through: a building a tunnel runs into is its portal building, its floor over the tube), but those of
    PORTAL_SKIP_CLASSES."""
    import roads as roadsmod
    out = []
    for f in road_layers:
        if f is None or len(f.kind) == 0:
            continue
        ts = roadsmod.tunnel_specs(f)
        for i in range(len(f.kind)):
            if ts[i] is None or ts[i].cls in PORTAL_SKIP_CLASSES:
                continue
            for ring in roadsmod.tunnel_parts(f, i):
                e, n = mapgeom.enu_xy(f, frame, ring)
                xy = np.stack([e - x0, n - y0], 1)
                if xy[:, 0].max() < -margin or xy[:, 0].min() > size + margin or xy[:, 1].max() < -margin \
                        or xy[:, 1].min() > size + margin:
                    continue
                out.append(xy)
    return out


GARAGE_TAGS = ("multi-storey", "multistorey", "multi_storey", "multi-level")
GARAGE_TAG_CLASSES = {"", "yes", "commercial", "garage", "garages", "parking", "parking_garage", "carport",
                      "service", "transportation", "industrial", "retail"}     # (not the apartments on a car park podium)


def garage_rings(frame, infra_layers, x0, y0, margin=50.0, size=256.0):
    """Outer rings (k, 2) relative to (x0, y0) of the mapped multi-storey car parks near the tile (Overture
    infrastructure, class parking, OSM parking=multi-storey). OSM often keeps the car park's tags on its own area
    and the building beside it as building=yes or =garage (a shed), so the class alone missed 25 of downtown
    Baltimore's 40 garages (2026-10-17, the user's 'parking garages are solid boxes')."""
    import json
    out, seen = [], set()
    for f in infra_layers:
        if f is None or len(f.kind) == 0:
            continue
        a = f.attrs
        cls, tags = a.get("class"), a.get("tags")
        if cls is None or tags is None:
            continue
        for i in np.flatnonzero((cls.astype(str) == "parking") & (f.kind == 3)):
            t = str(tags[i])
            if '"parking"' not in t:
                continue
            try:
                if str(json.loads(t).get("parking", "")).lower() not in GARAGE_TAGS:
                    continue
            except (ValueError, AttributeError):
                continue
            fid = f.ids[i]
            if fid in seen:
                continue
            seen.add(fid)
            for part in f.parts(i):
                if not part or len(part[0]) < 3:
                    continue
                e, n = mapgeom.enu_xy(f, frame, part[0])
                xy = np.stack([e - x0, n - y0], 1)
                if xy[:, 0].max() < -margin or xy[:, 0].min() > size + margin or xy[:, 1].max() < -margin \
                        or xy[:, 1].min() > size + margin:
                    continue
                out.append(xy)
    return out


def support_rings(frame, infra_layers, x0, y0, margin=50.0, size=256.0):
    """[(ring (k, 2) relative to (x0, y0), paint)] of the mapped bridge supports near the tile (Overture
    infrastructure, class bridge_support: OSM bridge:support towers and piers, their building:colour as a
    bridgetypes paint, grey without one). The Golden Gate Bridge's towers are mapped as building parts over them and
    were drawn as a 225 m glass tower (2026-10-18)."""
    import json
    import bridgetypes
    out, seen = [], set()
    for f in infra_layers:
        if f is None or len(f.kind) == 0:
            continue
        a = f.attrs
        cls, tags = a.get("class"), a.get("tags")
        if cls is None:
            continue
        cs = cls.astype(str)
        # a bridge's supports, and its own outlines mapped as buildings (Tower Bridge's high walkways: building:colour
        # lightblue, 45 m); not a bare bridge outline (Ponte Vecchio's shops stand on one)
        three_d = f.__dict__.get("_three_d")         # (once per cell's features: every tile of it asked)
        if three_d is None:
            three_d = f._three_d = (np.array(['"building' in str(t) for t in tags], bool) if tags is not None
                                    else np.zeros(len(cs), bool))
        for i in np.flatnonzero(((cs == "bridge_support") | ((cs == "bridge") & three_d)) & (f.kind == 3)):
            fid = f.ids[i]
            if fid in seen:
                continue
            seen.add(fid)
            pt = None
            if tags is not None and "colour" in str(tags[i]):
                try:
                    td = json.loads(str(tags[i]))
                    pt = bridgetypes.paint(td.get("building:colour") or td.get("colour"))
                except ValueError:
                    pt = None
            for part in f.parts(i):
                if not part or len(part[0]) < 3:
                    continue
                e, n = mapgeom.enu_xy(f, frame, part[0])
                xy = np.stack([e - x0, n - y0], 1)
                if xy[:, 0].max() < -margin or xy[:, 0].min() > size + margin or xy[:, 1].max() < -margin \
                        or xy[:, 1].min() > size + margin:
                    continue
                out.append((xy, pt or "grey"))
    return out


def _on_support(ring, supports):
    """The paint of the bridge support (support_rings) holding the footprint's centre, else None."""
    if not supports:
        return None
    from polygon import Path
    c = ring.mean(0)
    for r, pt in supports:
        if r[:, 0].min() <= c[0] <= r[:, 0].max() and r[:, 1].min() <= c[1] <= r[:, 1].max() \
                and Path(r).contains_points(c[None])[0]:
            return pt
    return None


def _covers_garage(ring, garages):
    """A footprint inside a mapped car park's area (its centre in it) or holding the car park's centre."""
    c = ring.mean(0)
    for g in garages:
        if _inside(g, c) or _inside(ring, g.mean(0)):
            return True
    return False


TOWER_FRAME_M = 40.0        # m: a steel bridge support this tall is a tower the decks pass (25 de Abril's 190 m pylons
                            # as buildings were solid blocks across one carriageway, 2026-10-28): an open frame
TUNNEL_THROUGH_M = 5.0      # a footprint a road tunnel runs into this far: its portal building, lifted over the tube
PORTAL_SKIP_CLASSES = ("service",)     # tunnels that lift no building: a driveway, loading dock or covered drop-off
                            # under a building, which stands on the ground (GitHub #60: a 15 m drive-through under an
                            # Oklahoma City church floated it 10 m up; in downtown Baltimore and Chicago 20 office and
                            # garage blocks of up to 7,156 m2 stood 7.4-10.5 m up over loading-dock tunnels, 2026-10-20)
OVER_TUNNEL_CLEAR_M = 7.5   # (the tube's vault is roadmesh.TUNNEL_H = 6.8 m over the floor; 2026-10-17: Eisenhower's
                            # east portal building closed the tunnel)


def road_lines(frame, road_layers, x0, y0, margin=50.0, size=256.0):
    """Polylines (k, 2) relative to (x0, y0) of the public roads on the ground near the tile (for road_through)."""
    import roads as roadsmod
    out = []
    for f in road_layers:
        if f is None or len(f.kind) == 0:
            continue
        cls = f.attrs.get("class")
        if cls is None:
            continue
        bb = f.boxes(frame)             # (only the roads near the tile: the test below, all at once)
        near = np.flatnonzero((bb[:, 2] - x0 >= -margin) & (bb[:, 0] - x0 <= size + margin)
                              & (bb[:, 3] - y0 >= -margin) & (bb[:, 1] - y0 <= size + margin))
        for i in near.tolist():
            if str(cls[i]) not in THROUGH_CLASSES:
                continue
            if roadsmod.level_ranges(f, i):
                # its pieces on the ground (a motorway segment with a short overpass in it was left out whole, and the
                # lorries machine-learned as buildings on its lanes stayed: the A7, GitHub #239; PR #71's finding)
                lls = [np.asarray(p, np.float64) for p in roadsmod.ground_parts(f, i)]
            else:
                a = f.coord_off[f.ring_off[f.part_off[i]]]
                b = f.coord_off[f.ring_off[f.part_off[i + 1]]]
                lls = [f.lonlat[a:b]]
            for ll in lls:
                if len(ll) < 2:
                    continue
                e, n = mapgeom.enu_xy(f, frame, ll)
                xy = np.stack([e - x0, n - y0], 1)
                if xy[:, 0].max() < -margin or xy[:, 0].min() > size + margin or xy[:, 1].max() < -margin \
                        or xy[:, 1].min() > size + margin:
                    continue
                out.append(xy)
    return out


PASSAGE_W = 11.0           # m: the passage a bridge's road takes through its tower (Tower Bridge: 2 lanes + walks) ...
PASSAGE_H = 7.0            # ... this high over the tower's mapped bottom (its deck: the part starts at the road), ...
PASSAGE_CLEAR = 6.5        # m: the way over a deck through a building (with the decks' heights)
PASSAGE_BOTTOM_M = 20.0    # ... in a structure whose bottom is this low (not its high walkways or tops)


def _clip_half(ring, p0, nrm, keep_ge):
    """The part of the ring (k, 2) on one side of the line through p0 with normal nrm (Sutherland-Hodgman)."""
    out = []
    pts = list(ring[:-1]) if np.allclose(ring[0], ring[-1]) else list(ring)
    side = [float((q - p0) @ nrm) for q in pts]
    for k in range(len(pts)):
        a, b = pts[k], pts[(k + 1) % len(pts)]
        sa, sb = side[k], side[(k + 1) % len(pts)]
        ina, inb = (sa >= 0) == keep_ge or sa == 0, (sb >= 0) == keep_ge or sb == 0
        if ina:
            out.append(a)
        if (sa > 0) != (sb > 0) and sa != 0 and sb != 0:
            t = sa / (sa - sb)
            out.append(a + t * (b - a))
    if len(out) < 3:
        return None
    r = np.array(out + [out[0]], np.float64)
    return r if _area2(r) / 2 >= 2.0 else None


DECK_STAND_M = 2.0      # m: a deck this far over the ground under a footprint is what the building stands on ...
DECK_LANE_M = 1.0       # ... and it is left out where it reaches this far inside the deck's road (its width / 2) ...
DECK_REACH_M = 6.0      # m past the road's edge a footprint's centre still counts as on its deck
DECK_SEA_M = 0.5        # ... only over ground this low: water, a beach (a building on dry land under a viaduct stays)


def on_deck(ring, decks, ground):
    """(over the lanes, deck height) when footprint ring (k, 2) has its centre on a deck (decks: [(xy (k, 2), z (k,),
    width m)], the same frame) standing DECK_STAND_M or more over ground (m, its highest ground, DECK_SEA_M at most:
    over water), else None. A building
    on the old Sunshine Skyway span kept as a fishing pier (its bait shop) was raised from the seabed: its top 0.5 m
    stood over the deck (2026-10-27). Over the lanes: a corner more than DECK_LANE_M inside the road's edge. Pure."""
    r = np.asarray(ring, np.float64)[:, :2]
    c = r.mean(0)
    best = None
    for d in decks:
        if len(d) < 3 or len(d[0]) < 2:
            continue
        xy, z, w = np.asarray(d[0], np.float64), np.asarray(d[1], np.float64), float(d[2])
        a, b = xy[:-1], xy[1:]
        ab = b - a
        t = np.clip(((c - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
        dist = np.hypot(*(a + t[:, None] * ab - c).T)
        k = int(np.argmin(dist))
        half = w / 2 + DECK_REACH_M         # (a pier's map width is its lane: its deck reaches past it)
        if dist[k] > half:
            continue
        zd = float(z[k] + t[k] * (z[k + 1] - z[k]))
        if zd - ground < DECK_STAND_M:
            continue
        if best is None or dist[k] < best[0]:
            best = (dist[k], zd, a[k], ab[k] / max(float(np.hypot(*ab[k])), 1e-9), w)
    if best is None or ground > DECK_SEA_M:
        return None
    _, zd, p, u, w = best
    off = (r - p) @ np.array([-u[1], u[0]])            # (signed: a footprint across the lanes has corners both sides)
    inner = w / 2 - DECK_LANE_M
    return bool(off.min() < inner and off.max() > -inner), zd


def _passages(rings, attrs, looks, mins, bridges_xy, decks=None, ground_at=None):
    """Buildings a road bridge runs through split round a PASSAGE_W wide way (Tower Bridge's road ran into its towers'
    stone and its Southern Gate, the user 2026-10-18: "the buildings are still in the middle of the bridge", "this
    building is still blocking the bridge"). With the decks' heights (decks: [(xy (k, 2), z (k,))] relative to the
    tile, ground_at(c (2,)) -> the ground under a footprint's centre): only where the deck passes through the
    building's height (a building under a viaduct stays whole): solid under the deck, the way PASSAGE_CLEAR over it,
    the whole footprint above. Without them, a bridge's own structure (bridge_part / bridge_masonry) starting within
    PASSAGE_BOTTOM_M gets the way PASSAGE_H over its bottom (its deck: the part starts at the road)."""
    R, A, L, M = [], [], [], []
    lines = [d[0] for d in decks] if decks else bridges_xy
    for ring, at, look, mh in zip(rings, attrs, looks, mins):
        bridge_own = bool(at.get("bridge_part") or at.get("bridge_masonry"))
        if not lines or road_through(ring, lines) < 3.0 or (not decks and (not bridge_own or mh > PASSAGE_BOTTOM_M)):
            R.append(ring), A.append(at), L.append(look), M.append(mh)
            continue
        c = ring.mean(0)
        best = None
        for k, ln in enumerate(lines):
            for q in range(len(ln) - 1):
                a, b = ln[q], ln[q + 1]
                ab = b - a
                t = float(np.clip((c - a) @ ab / max(ab @ ab, 1e-9), 0, 1))
                d = float(np.hypot(*(a + t * ab - c)))
                if best is None or d < best[0]:
                    zd = None
                    if decks:
                        z = decks[k][1]
                        zd = float(z[q] + t * (z[q + 1] - z[q]))
                    best = (d, a + t * ab, ab / max(float(np.hypot(*ab)), 1e-9), zd)
        _, p0, u, zd = best
        nrm = np.array([-u[1], u[0]])
        h = at.get("height")
        own = look is not None          # (a building's part carries its own height over its bottom, a building its top)
        top = (mh + float(h) if own else float(h)) if h is not None and np.isfinite(h) else mh + PASSAGE_H + 3.0
        if zd is not None and ground_at is not None and np.isfinite(zd):
            dh = zd - float(ground_at(c))           # the deck's height over the building's ground
            if top <= dh + 1.0 or mh >= dh + PASSAGE_CLEAR:
                R.append(ring), A.append(at), L.append(look), M.append(mh)
                continue                            # under the deck, or over its way
            low, cut = max(mh, dh - 1.5), dh + PASSAGE_CLEAR
        else:
            low, cut = mh, mh + PASSAGE_H
        cut = min(cut, top)
        if low > mh + 0.5:                          # solid under the deck
            R.append(ring), A.append(dict(at, height=(low - mh) if own else low)), L.append(look), M.append(mh)
        for sgn in (1, -1):
            piece = _clip_half(ring, p0 + nrm * sgn * PASSAGE_W / 2, nrm * sgn, True)
            if piece is not None and cut > low + 0.5:
                R.append(piece), A.append(dict(at, height=(cut - low) if own else cut)), L.append(look), M.append(low)
        if top > cut + 1.0:
            R.append(ring), A.append(dict(at, height=(top - cut) if own else top)), L.append(look), M.append(cut)
    return R, A, L, M


ML_BRIDGE_M = 6.0          # m of a road bridge's line through a machine-learned footprint: the deck, not a building
ML_ROAD_M = 4.0            # m from a public road's centre line to a machine-learned footprint's centre: a vehicle on it
ML_MEDIAN_M = 9.0          # m: a machine-learned footprint between two parallel roads this close on either side lies in a
                           # dual carriageway's median (GitHub #90: a house between Moscow's Bolshaya Tulskaya carriageways)


def _line_dist(p, lines):
    """Distance from point p (2,) to the nearest of the polylines lines (k, 2), or inf."""
    best = np.inf
    for ln in lines:
        a, b = ln[:-1], ln[1:]
        d = b - a
        t = np.clip(((p - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0.0, 1.0)
        q = a + t[:, None] * d
        best = min(best, float(np.hypot(*(q - p).T).min()))
    return best


def _nearest_on(p, ln):
    """(distance, unit direction of the segment) of polyline ln (k, 2) nearest point p (2,)."""
    a, b = ln[:-1], ln[1:]
    d = b - a
    t = np.clip(((p - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0.0, 1.0)
    q = a + t[:, None] * d
    dist = np.hypot(*(q - p).T)
    k = int(np.argmin(dist))
    u = d[k] / max(float(np.hypot(*d[k])), 1e-9)
    return float(dist[k]), q[k], u


def _in_median(c, roads_xy):
    """Does point c (2,) lie between two parallel roads each within ML_MEDIAN_M, on opposite sides of it?"""
    near = []
    for ln in roads_xy:
        if len(ln) < 2 or ln[:, 0].min() > c[0] + ML_MEDIAN_M or ln[:, 0].max() < c[0] - ML_MEDIAN_M or \
                ln[:, 1].min() > c[1] + ML_MEDIAN_M or ln[:, 1].max() < c[1] - ML_MEDIAN_M:
            continue
        dist, q, u = _nearest_on(c, ln)
        if dist <= ML_MEDIAN_M:
            near.append((q - c, u))
    for i in range(len(near)):
        for j in range(i + 1, len(near)):
            (vi, ui), (vj, uj) = near[i], near[j]
            if abs(float(ui @ uj)) > 0.9 and float(vi @ vj) < 0:        # parallel, one on each side
                return True
    return False


def _ml_false(ring, bridges_xy, water_xy, roads_xy=None):
    """A machine-learned footprint (Microsoft's, Google's) that is a bridge deck, a boat or a vehicle: a road bridge
    runs through it, its centre lies on mapped water (Tower Bridge's river: barges as brick buildings, 2026-10-18), on
    a public road (a lorry on a Norwegian trunk road stood as a white building over the lanes, GitHub #2) or in a dual
    carriageway's median (_in_median, GitHub #90)."""
    if bridges_xy and road_through(ring, bridges_xy) >= ML_BRIDGE_M:
        return True
    if roads_xy and (_line_dist(ring.mean(0), roads_xy) <= ML_ROAD_M or _in_median(ring.mean(0), roads_xy)):
        return True
    return _on_water(ring.mean(0), water_xy)


def _on_water(c, water_xy):
    """Whether point c (local) lies on mapped water (water_rings), not on an island in it."""
    if not water_xy:
        return False
    from polygon import Path
    for w, holes in water_xy:
        if w[:, 0].min() <= c[0] <= w[:, 0].max() and w[:, 1].min() <= c[1] <= w[:, 1].max() \
                and Path(w).contains_points(c[None])[0] \
                and not any(Path(h).contains_points(c[None])[0] for h in holes):    # (an island: land)
            return True
    return False


ML_STEEP_DEG = 25.0         # a machine-learned footprint on ground this steep ...
ML_FAR_FROM_WAY_M = 60.0    # ... this far from any mapped way (a road, track or path) is a rock or a shadow, not a hut
ML_CLIFF_DEG = 36.0         # ... and on ground this steep, wherever it is
# (GitHub #242: 64 machine-learned outlines (Shi et al.'s East Asia set) on Mount Fuji's slopes, median 34 deg, half
# over 60 m from a trail; the real huts on 6 deg next to the trail. Steep places with real ones, 2026-10-24: La Paz 0 of
# 5453, Gorkha (Nepal) 0 of 50, Rwanda's hills 1 of 293, Medellin and Valparaiso 0)


def _ml_steep(ring, g, ways_xy):
    """Is a machine-learned footprint (ring, relative to the tile's corner; g its ground at each corner) on ground too
    steep for it (ML_*)? The slope of the plane through its corners; the distance from its centre to the nearest
    mapped way (ways_xy: way_lines, the same corner). Pure."""
    r = np.asarray(ring, np.float64)[:, :2]
    g = np.asarray(g, np.float64)
    if len(r) < 3 or len(g) != len(r):
        return False
    A = np.c_[r - r.mean(0), np.ones(len(r))]
    try:
        (a, b, _), *_ = np.linalg.lstsq(A, g, rcond=None)
    except np.linalg.LinAlgError:
        return False
    slope = math.degrees(math.atan(math.hypot(a, b)))
    if slope > ML_CLIFF_DEG:
        return True
    if slope <= ML_STEEP_DEG:
        return False
    return not ways_xy or _line_dist(r.mean(0), ways_xy) > ML_FAR_FROM_WAY_M


def way_lines(frame, road_layers, x0, y0, margin=ML_FAR_FROM_WAY_M + 20.0, size=256.0):
    """Polylines (k, 2) relative to (x0, y0) of every mapped way near the tile (roads of any class, tracks, paths,
    steps: _ml_steep's access test)."""
    out = []
    for f in road_layers:
        if f is None or len(f.kind) == 0:
            continue
        bb = f.boxes(frame)
        near = np.flatnonzero((bb[:, 2] - x0 >= -margin) & (bb[:, 0] - x0 <= size + margin)
                              & (bb[:, 3] - y0 >= -margin) & (bb[:, 1] - y0 <= size + margin))
        for i in near.tolist():
            a = f.coord_off[f.ring_off[f.part_off[i]]]
            b = f.coord_off[f.ring_off[f.part_off[i + 1]]]
            ll = f.lonlat[a:b]
            if len(ll) < 2:
                continue
            e, n = mapgeom.enu_xy(f, frame, ll)
            out.append(np.stack([e - x0, n - y0], 1))
    return out


def bridge_lines(frame, road_layers, x0, y0, margin=50.0, size=256.0):
    """Polylines (k, 2) relative to (x0, y0) of the road bridges near the tile (their bridge ranges)."""
    import roads as roadsmod
    out = []
    for f in road_layers:
        if f is None or len(f.kind) == 0:
            continue
        bs = roadsmod.bridge_specs(f)
        for i in range(len(f.kind)):
            if bs[i] is None:
                continue
            for ring in roadsmod.bridge_parts(f, i):
                e, n = mapgeom.enu_xy(f, frame, ring)
                xy = np.stack([e - x0, n - y0], 1)
                if len(xy) < 2 or xy[:, 0].max() < -margin or xy[:, 0].min() > size + margin or \
                        xy[:, 1].max() < -margin or xy[:, 1].min() > size + margin:
                    continue
                out.append(xy)
    return out


def water_rings(frame, water_layers, x0, y0, margin=50.0, size=256.0):
    """[(outer ring (k, 2), [hole rings])] relative to (x0, y0) of the mapped water areas (sea, rivers, lakes, docks)
    near the tile (their holes: islands)."""
    out = []
    for f in water_layers:
        if f is None or len(f.kind) == 0:
            continue
        for i in np.flatnonzero(f.kind == 3):
            for part in f.parts(i):
                if not part or len(part[0]) < 4:
                    continue
                e, n = mapgeom.enu_xy(f, frame, part[0])
                xy = np.stack([e - x0, n - y0], 1)
                if xy[:, 0].max() < -margin or xy[:, 0].min() > size + margin or xy[:, 1].max() < -margin \
                        or xy[:, 1].min() > size + margin:
                    continue
                holes = []
                for h in part[1:]:
                    if len(h) >= 4:
                        he, hn = mapgeom.enu_xy(f, frame, h)
                        holes.append(np.stack([he - x0, hn - y0], 1))
                out.append((xy, holes))
    return out


_BOXES = {}                     # id(list of polylines) -> (that list, its boxes (n, 4)): the last few lists asked


def line_boxes(lines):
    """(n, 4) x0 y0 x1 y1 of each polyline (k, 2+) of the list, made once per list (a tile's buildings test the same
    lists of roads one footprint after another: ~1.2M small numpy min / max calls a Baltimore tile set)."""
    got = _BOXES.get(id(lines))
    if got is not None and got[0] is lines:
        return got[1]
    b = np.array([(ln[:, 0].min(), ln[:, 1].min(), ln[:, 0].max(), ln[:, 1].max()) for ln in lines],
                 np.float64).reshape(-1, 4)
    if len(_BOXES) >= 16:
        _BOXES.clear()
    _BOXES[id(lines)] = (lines, b)      # (the list kept: its id cannot be another list's meanwhile)
    return b


def road_through(ring, lines, step=2.0):
    """Metres of the polylines lines (k, 2) inside ring (sampled every step m)."""
    lo, hi = ring.min(0), ring.max(0)
    b = line_boxes(lines)
    near = np.flatnonzero((b[:, 2] >= lo[0]) & (b[:, 0] <= hi[0]) & (b[:, 3] >= lo[1]) & (b[:, 1] <= hi[1]))
    if not len(near):
        return 0.0
    fast.require(74, "road_through")         # (the Rust loop: every sample's even-odd test there)
    sel = [np.asarray(lines[li], np.float64)[:, :2] for li in near]
    starts = np.r_[0, np.cumsum([len(ln) for ln in sel])].astype(np.int64)
    return float(fast._ef.road_through(np.ascontiguousarray(np.asarray(ring, np.float64)[:, :2]),
                                       np.ascontiguousarray(np.concatenate(sel)), starts, float(step)))


def road_across(ring, lines, max_w=ACROSS_W):
    """Does one of the polylines lines (k, 2) cross the narrow footprint ring (at most max_w wide) along its length,
    over its long middle line?"""
    c, u, hl, hw = oriented_box(ring)
    if 2 * hw > max_w or hl <= hw:
        return False
    a, b = c - u * hl, c + u * hl
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    bx = line_boxes(lines)
    for li in np.flatnonzero((bx[:, 2] >= lo[0]) & (bx[:, 0] <= hi[0]) & (bx[:, 3] >= lo[1]) & (bx[:, 1] <= hi[1])):
        ln = lines[li]
        p, q = ln[:-1], ln[1:]
        d = q - p
        e = b - a
        den = d[:, 0] * e[1] - d[:, 1] * e[0]
        ok = np.abs(den) > 1e-9
        den = np.where(ok, den, 1.0)
        w = a - p
        t = (w[:, 0] * e[1] - w[:, 1] * e[0]) / den         # along the road segment
        s_ = (w[:, 0] * d[:, 1] - w[:, 1] * d[:, 0]) / den  # along the middle line
        if np.any(ok & (t >= 0) & (t <= 1) & (s_ >= 0) & (s_ <= 1)):
            return True
    return False


CENTRE_PREFILTER_M = 1.0       # m of slack for Features.outer_centres (cumulative sums: within ~1e-5 m of the exact mean)


def _parts_index(f):
    """{building id: [rows]} of a building parts layer, kept on it (the cells stay decoded in memory: every tile of
    them looks parts up again)."""
    memo = f.__dict__.get("_bnge_parts_by_bid")
    if memo is None:
        memo = {}
        col = f.attrs.get("building_id")
        if col is not None:
            for i in range(len(f.kind)):
                bid = str(col[i])
                if bid:
                    memo.setdefault(bid, []).append(i)
        f.__dict__["_bnge_parts_by_bid"] = memo
    return memo


def _parts_tops(f):
    """Each part's own height + bottom (_part_heights' sum; nan without a height) for the whole layer, kept on it."""
    tops = f.__dict__.get("_bnge_parts_tops")
    if tops is None:
        a, n = f.attrs, len(f.kind)

        def col(k):
            return np.asarray(a[k], np.float64) if k in a else np.full(n, np.nan)
        with np.errstate(invalid="ignore"):
            top, mh, nf, f0 = col("height"), col("min_height"), col("num_floors"), col("min_floor")
            mh = np.where(np.isfinite(mh) & (mh > 0), mh, 0.0)
            f0 = np.where(np.isfinite(f0) & (f0 > 0), f0, 0.0)
            floors = np.maximum(FLOOR_M, (nf - f0) * FLOOR_M) + np.where(mh > 0, mh, f0 * FLOOR_M)
            tops = np.where(np.isfinite(top) & (top > mh + 1.0), top,
                            np.where(np.isfinite(nf) & (nf > 0), floors, np.nan))
        f.__dict__["_bnge_parts_tops"] = tops
    return tops


class _PartsOf:
    """The building parts of a tile's cells by building id, each [(ring in the anchor frame, _part_heights)], made
    when asked for: a tile needs those of its own buildings, and the toll booths (booths), not all the cells' parts
    (converting every part for every tile was half of tile_buildings' time in Manhattan, 2026-10-24). A part in two
    cells' layers is taken once (the first)."""

    def __init__(self, frame, part_layers):
        self.frame = frame
        self.layers = [f for f in part_layers if f is not None and len(f.kind)]
        self.memo = {}

    def _part(self, f, i):
        pp = f.parts(i)
        if not pp or not pp[0]:
            return None
        outer = np.asarray(pp[0][0], np.float64)
        if len(outer) < 4:
            return None
        e, n = mapgeom.enu_xy(f, self.frame, outer)
        a = f.attrs
        # (its own roof shape and mapped colours: hudsontadams' PR #167)
        own = {k: (a[k][i] if k in a else None) for k in ("roof_shape", "facade_color", "roof_color")}
        return np.stack([e, n], 1), _part_heights(a, i), own

    def get(self, bid, default=None):
        if bid not in self.memo:
            out, seen = [], set()
            for f in self.layers:
                for i in _parts_index(f).get(bid, ()):
                    pid = f.ids[i]
                    if pid in seen:
                        continue
                    seen.add(pid)
                    p = self._part(f, i)
                    if p is not None:
                        out.append(p)
            self.memo[bid] = out or None
        got = self.memo[bid]
        return default if got is None else got

    def booths(self, lo, hi):
        """The parts (ring, _part_heights) whose top (own height + bottom) is within [lo, hi], of any building."""
        out, seen = [], set()
        for f in self.layers:
            index = _parts_index(f)
            if not index:
                continue
            col = f.attrs["building_id"]
            tops = _parts_tops(f)
            with np.errstate(invalid="ignore"):
                rows = np.flatnonzero((tops >= lo - 1e-6) & (tops <= hi + 1e-6))
            for i in rows.tolist():
                pid = f.ids[i]
                if pid in seen or not str(col[i]):
                    continue
                seen.add(pid)
                p = self._part(f, i)
                if p is not None:
                    out.append(p[:2])
        return out


def tile_buildings(frame, x0, y0, x1, y1, layers, elev, water=None, part_layers=(), west_town=False, roads_xy=None,
                   tunnels_xy=None, garages_xy=None, supports=None, bridges_xy=None, water_xy=None, decks=None,
                   ml_roads_xy=None, ways_xy=None, skip=frozenset(), min_size=None):
    """[(Spec, gmin, gmax)] of the buildings whose footprint centroid is in the tile [x0, x1) x [y0, y1)
    (anchor frame), rings relative to (x0, y0). layers: mapgeom.Features of buildings (one per map cell; a
    building in two cells is taken once, by id); elev(lat, lon) the physics ground. part_layers: the
    building parts of the same cells: a building whose parts all have heights is drawn as its parts
    (each from its bottom to its top, the building's look). garages_xy: garage_rings (a building one covers, or
    covering one's centre, is a parking structure). supports: support_rings (a building or part on one is the
    bridge's painted steel). bridges_xy / water_xy (bridge_lines, water_rings): a machine-learned footprint (not
    OSM) on a road bridge, over water or on a public road (roads_xy, or ml_roads_xy for that test alone: the far
    buildings) is a deck, a boat or a vehicle, left out (_ml_false). skip: ids of buildings left out (a gas station's
    canopy, drawn by the mod from the stock levels' props: gasstations.tile_sites). min_size (min height m, min
    footprint m²): only buildings at least that tall or that large (the skyline tiles: skyline_tile), tested before
    any other work on a building."""
    parts_of = _PartsOf(frame, part_layers)     # building id -> [(ring in the anchor frame, (own height, bottom))]
    seen = set()
    rings, attrs, looks, mins = [], [], [], []
    for f in layers:
        if f is None or len(f.kind) == 0:
            continue
        a = f.attrs
        ids = getattr(f, "ids", None)
        rows = range(len(f.kind))
        if isinstance(f, mapgeom.Features):     # only the buildings whose centre may be in the tile (the exact test below)
            cx_, cy_ = f.outer_centres(frame)
            m = CENTRE_PREFILTER_M
            rows = np.flatnonzero((cx_ >= x0 - m) & (cx_ < x1 + m) & (cy_ >= y0 - m) & (cy_ < y1 + m)).tolist()
        for i in rows:
            if a.get("is_underground") is not None and bool(a["is_underground"][i]):
                continue
            lv = a["level"][i] if a.get("level") is not None else None
            if lv is not None and np.isfinite(lv) and lv < 0:
                continue                # below ground (Shot Tower Metro, level -2, stood across a Baltimore street)
            key = ids[i] if ids is not None else None
            if key is not None and (key in seen or key in skip):
                continue
            parts = f.parts(i)
            if not parts or not parts[0]:
                continue
            outer = np.asarray(parts[0][0], np.float64)
            if len(outer) < 4:
                continue
            e, n = mapgeom.enu_xy(f, frame, outer)
            k = len(e) - 1             # the same reduction as np.mean, without its wrapper (36k calls a tile set)
            cx, cy = float(e[:-1].sum() / k), float(n[:-1].sum() / k)
            if not (x0 <= cx < x1 and y0 <= cy < y1):
                continue
            if min_size is not None:
                h_ = a["height"][i] if a.get("height") is not None else None
                if h_ is None or not np.isfinite(h_):
                    fl_ = a["num_floors"][i] if a.get("num_floors") is not None else None
                    h_ = float(fl_) * 3.2 if fl_ is not None and np.isfinite(fl_) else 0.0
                if h_ < min_size[0] and abs(float(np.dot(e[:-1], n[1:]) - np.dot(e[1:], n[:-1]))) / 2 < min_size[1]:
                    continue
            at = {k: (a[k][i] if k in a else None) for k in
                  ("class", "height", "num_floors", "min_height", "roof_shape", "roof_height", "name",
                   "facade_material", "facade_color", "roof_material", "roof_color")}
            if len(parts) > 1:
                at["holed"] = 1.0               # (landmark_kind: a ring round a courtyard is no dome)
            elif len(parts[0]) > 1:
                # the holes' share of the outline (a stair well or a light well in a dome is not a courtyard: the
                # Long Beach cruise terminal's 56 m2 hole in 15,100 m2)
                a_out = abs(float(np.dot(e[:-1], n[1:]) - np.dot(e[1:], n[:-1]))) / 2
                a_in = 0.0
                for hole in parts[0][1:]:
                    he, hn = mapgeom.enu_xy(f, frame, np.asarray(hole, np.float64))
                    a_in += abs(float(np.dot(he[:-1], hn[1:]) - np.dot(he[1:], hn[:-1]))) / 2
                at["holed"] = a_in / max(a_out, 1e-6)
            hh_ = at.get("height")
            if hh_ is not None and np.isfinite(hh_) and 0 < hh_ < FLAT_FOOTPRINT_M and not parts_of.get(str(key)):
                continue                # a mapped area, not a building (the Golden Gate toll plaza's 0.01 m footprint
                                        # drew a slab with a rim round the booths, 2026-10-21)
            bparts = parts_of.get(str(key)) if key is not None else None
            funnels = None
            if bparts and "src_dataset" in a and str(a["src_dataset"][i]) == "OpenStreetMap":
                # a ship drawn as its hull with parts on it (the Iowa's turrets, the Queen Mary's funnels): the ship,
                # its parts on its deck (_ship)
                ring_ = clean_ring(np.stack([e - x0, n - y0], 1))
                if ring_ is not None and landmark_kind(ring_, at) == "ship":
                    funnels = [(pr - np.array([x0, y0]), None if hh is None else hh[0]) for pr, hh, *_ in bparts]
                    bparts = None
            if bparts and any(p[1] is not None for p in bparts):
                # drawn as its parts, with the building's class and one look for all of them; a part
                # without a height takes the building's (else its lowest known part's)
                look = _hash01(str(key), "look")
                bh = at.get("height")
                fill = float(bh) if bh is not None and np.isfinite(bh) and bh > 0 else \
                    min(p[1][0] + p[1][1] for p in bparts if p[1] is not None)
                top = max([fill] + [p[1][0] + p[1][1] for p in bparts if p[1] is not None])
                for ppart in bparts:
                    pring, hh = ppart[0], ppart[1]
                    pat = ppart[2] if len(ppart) > 2 else {}
                    ph, pmh = hh if hh is not None else (fill, 0.0)
                    if ph + pmh < FLAT_FOOTPRINT_M:
                        continue        # (a part taking a flat area's height: FLAT_FOOTPRINT_M)
                    r = clean_ring(pring - np.array([x0, y0]))
                    if r is None or _area2(r) / 2 < 4.0:
                        continue
                    rings.append(r)
                    p_rshape = str(pat.get("roof_shape") or "").lower() or "flat"
                    p_fcol = pat.get("facade_color") or at.get("facade_color")
                    p_rcol = pat.get("roof_color") or at.get("roof_color")
                    attrs.append({"class": at["class"], "height": ph, "num_floors": None,
                                  "roof_shape": p_rshape,
                                  "building_height": top, "facade_material": at.get("facade_material"),
                                  "facade_color": p_fcol,
                                  "roof_material": at.get("roof_material"),
                                  "roof_color": p_rcol,
                                  "bridge_part": _on_support(r, supports) if str(at.get("facade_material") or "")
                                  .lower() in STEEL_MATERIALS else None,
                                  "bridge_masonry": str(at.get("facade_material") or "").lower()
                                  if _on_support(r, supports) is not None and str(at.get("facade_material") or "")
                                  .lower() not in STEEL_MATERIALS else None})
                    looks.append(look)
                    mins.append(pmh)
                if key is not None:
                    seen.add(key)
                continue
            ring = clean_ring(np.stack([e - x0, n - y0], 1))
            if ring is None or _area2(ring) / 2 < MIN_AREA_M2:
                continue
            is_ml = "src_dataset" in a and str(a["src_dataset"][i]) != "OpenStreetMap"
            if is_ml and _ml_false(ring, bridges_xy, water_xy, ml_roads_xy if ml_roads_xy is not None else roads_xy):
                continue
            if key is not None:
                seen.add(key)
            rings.append(ring)
            mh = at.get("min_height")
            mh = float(mh) if mh is not None and np.isfinite(mh) and mh > 0 else 0.0
            through = False
            if mh <= 0.5 and roads_xy and (road_through(ring, roads_xy) >= ROAD_THROUGH_M or road_across(ring, roads_xy)):
                through = True
                mh = OVER_ROAD_CLEAR_M
                h = at.get("height")
                if h is None or not np.isfinite(h) or h < mh + 3.0:
                    at = dict(at, height=mh + 3.0)      # keeps its top; at least a slab's worth over the clearance
            if mh < OVER_TUNNEL_CLEAR_M and tunnels_xy and road_through(ring, tunnels_xy) >= TUNNEL_THROUGH_M:
                # over a tunnel: its floor OVER_TUNNEL_CLEAR_M over the road at the mouth near it (the lowest ground
                # along the tunnel's line round it), from its own lowest ground (base_z); none where its ground is
                # that high already (a house on the hill over the tube: Wabash's floated ~10 m, 2026-10-17)
                lift = _portal_lift(frame, ring, x0, y0, tunnels_xy, elev)
                if lift > 0.5:
                    mh = lift
                    at = dict(at, portal=True)
                    h = at.get("height")
                    if h is None or not np.isfinite(h) or h < mh + 3.0:
                        at = dict(at, height=mh + 3.0)
            if mh <= 0.5 and garages_xy and _covers_garage(ring, garages_xy):
                at = dict(at, garage=True)
            on = _on_support(ring, supports)
            if on is not None:              # a bridge's own: no flats' windows (Tower Bridge's towers, the user 2026-10-18)
                fm_ = str(at.get("facade_material") or "").lower()
                h_ = at.get("height")
                tall = h_ is not None and np.isfinite(h_) and h_ >= TOWER_FRAME_M and fm_ in STEEL_MATERIALS
                if through or tall:
                    # a tower the deck runs through (25 de Abril's pylons: building, bridge:support=pylon, 190 m) was
                    # a solid block with a tunnel for the road, 2026-10-28: an open frame on the ground, the decks
                    # between its legs
                    at = dict(at, **{"class": "bridge_tower"}, facade_color=at.get("facade_color"))
                    mh = 0.0
                else:
                    at = dict(at, bridge_part=on) if fm_ in STEEL_MATERIALS else dict(at, bridge_masonry=fm_)
            if is_ml:
                at = dict(at, ml=True)          # (_ml_steep once its ground is known)
            elif on is None and mh <= 0.5:
                lk = landmark_kind(ring, at, on_water=lambda: _on_water(ring.mean(0), water_xy))     # (asked last: slow)
                if lk:
                    at = dict(at, landmark=lk, ship_parts=funnels)
            attrs.append(at)
            looks.append(None)
            mins.append(mh)
    if not rings:
        return []
    hulls = [r for r, at in zip(rings, attrs) if at.get("landmark") == "ship"]
    if hulls:       # a quay structure alongside a moored ship: a low deck (its scanned height is the ship's flank)
        for k_, (r, at) in enumerate(zip(rings, attrs)):
            if not at.get("landmark") and not at.get("ml") and looks[k_] is None and mins[k_] <= 0.5 \
                    and any(pier_alongside(r, hr) for hr in hulls):
                attrs[k_] = dict(at, landmark="pier")
    if bridges_xy or decks:             # a road bridge through a building: a passage (Tower Bridge's towers and gate)
        def ground_at(c):
            la, lo, _ = frame.enu_to_geodetic(float(c[0]) + x0, float(c[1]) + y0, 0.0)
            z = float(np.asarray(elev(np.array([float(la)]), np.array([float(lo)])), np.float64)[0])
            return float(frame.geodetic_to_enu(float(la), float(lo), z)[2])       # (the decks' frame up)
        rings, attrs, looks, mins = _passages(rings, attrs, looks, mins, bridges_xy, decks, ground_at)
    shared = _shared_walls(rings)
    cover = sum(_area2(r) / 2 for r, mh in zip(rings, mins) if mh <= 0.5) / ((x1 - x0) * (y1 - y0))
    town, dense = cover < TOWN_COVERAGE, cover > BRICK_WALK_COVERAGE
    clat, clon, _ = frame.enu_to_geodetic((x0 + x1) / 2, (y0 + y1) / 2, 0.0)
    clat, clon = float(clat), float(clon)
    alpine = False
    if cover < ALPINE_COVER:
        try:
            gz = float(np.asarray(elev(np.array([clat]), np.array([clon])), np.float64)[0])
            alpine = np.isfinite(gz) and gz > tree_line_m(clat)
        except Exception:
            alpine = False
    # the ground at every building's corners in one call (elev and the frame's conversions are point by point; a
    # call per building cost a far buildings tile ~20 % of its time, 2026-10-20)
    ends = np.cumsum([len(r) for r in rings])
    allr = np.concatenate(rings)
    la_all, lo_all, _ = frame.enu_to_geodetic(allr[:, 0] + x0, allr[:, 1] + y0, np.zeros(len(allr)))
    la_all, lo_all = np.asarray(la_all, np.float64), np.asarray(lo_all, np.float64)
    g_all = np.asarray(elev(la_all, lo_all), np.float64)
    ok = np.isfinite(g_all)
    # heights above the ellipsoid -> the anchor frame's up (the Earth's curvature, as the terrain and the roads):
    # without it a building stood (d^2 / 2R) over the ground at d from the anchor's origin, 5 m at 8 km, up to
    # 18 m at the 15 km rebase distance (the user, 2026-10-18 at Santa Maria Val Muestair: floating houses)
    up_all = np.full(len(allr), np.nan)
    if ok.any():
        up_all[ok] = np.asarray(frame.geodetic_to_enu(la_all[ok], lo_all[ok], g_all[ok])[2], np.float64)
    out = []
    for k, (ring, at, sh, look, mh) in enumerate(zip(rings, attrs, shared, looks, mins)):
        if mh > 0.5 and at.get("height") is not None and np.isfinite(at["height"]) and look is None:
            at = dict(at, height=max(2.5, float(at["height"]) - mh))     # a building's own height range
        spec = plan(ring, at, int(sh) if look is None else 0, clat, clon, look, town, dense, west_town, alpine)
        spec.min_h = mh
        spec.portal = bool(at.get("portal"))
        i0, i1 = (ends[k - 1] if k else 0), ends[k]
        if not ok[i0:i1].all():
            continue
        g = up_all[i0:i1].copy()
        if at.get("ml") and ways_xy is not None and _ml_steep(ring, g, ways_xy):
            continue
        if water is not None:
            c = ring.mean(0)
            cla, clo, _ = frame.enu_to_geodetic(c[0] + x0, c[1] + y0, 0.0)
            if water(float(cla), float(clo)):
                continue
        if decks and mh <= 0.5 and not (at.get("bridge_part") or at.get("bridge_masonry")):
            got = on_deck(ring, decks, float(g.max()))
            if got is not None:
                if got[0]:              # over the deck's lanes: a vehicle, a booth on the road
                    continue
                g = np.full(len(g), got[1])     # its floor on the deck (a pier's bait shop stood on the seabed)
        spec.gring = g              # the ground at each corner (door_ground: the door's, not the footprint's highest)
        out.append((spec, float(g.min()), float(g.max())))
    lines = roads_xy if roads_xy else ml_roads_xy
    if lines and out:
        isl = _toll_islands(frame, layers, parts_of, lines, x0, y0)
        if isl:
            out = _toll_mark(out, isl)
    return out


# Toll plazas (the user's report at the Golden Gate Bridge, 2026-10-21: "Toll booths need work. west coast has one,
# reference that"): OSM maps a plaza's islands as low "buildings" (0.2 m there) with the booths as building parts
# standing on them; drawn as buildings they were bare boxes on slabs. An island with a booth on it, along a road, is
# West Coast USA's toll booth instead (trees.lua group "tollbooth": the island, its booth and kerbs in one shape, 22 m
# long along its local y), placed along the island (toll_booth_rows); the island and its booth parts are not built.
FLAT_FOOTPRINT_M = 0.1     # m: a footprint mapped lower than this is an area (a plaza), not drawn
G_TOLLBOOTH = "tollbooth"
TOLLBOOTH_H = 4.96          # m: the stock shape's height (s_toll_booth_center)
TOLL_ISLAND_MAX_H = 1.0     # m: an island is a "building" no taller than this ...
TOLL_ISLAND_MIN_L = 8.0     # ... at least this long ...
TOLL_ISLAND_MAX_W = 4.0     # ... and at most this wide
TOLL_BOOTH_H = (2.0, 8.0)   # m: a booth on it: a footprint this tall ...
TOLL_BOOTH_MAX_AREA = 40.0  # ... and this small (m2), its centre on the island
TOLL_ROAD_M = 25.0          # m: a road this near the island's centre ...
TOLL_ROAD_DEG = 25.0        # ... running along it within this


def _on_box(p, box, pad=1.0):
    """Is point p (2,) within the oriented box (centre, axis u, half length, half width) grown by pad?"""
    c, u, hl, hw = box
    d = np.asarray(p, np.float64) - c
    return abs(float(d @ u)) <= hl + pad and abs(float(d[0] * u[1] - d[1] * u[0])) <= hw + pad


def _toll_islands(frame, layers, parts_of, lines, x0, y0):
    """The toll islands among all the buildings of the tile's cells, as oriented boxes relative to (x0, y0): a low
    building (TOLL_ISLAND_*) with a booth (TOLL_BOOTH_*) among all the cells' building parts centred on it, along a
    road (lines, relative to (x0, y0)). Not only the tile's own buildings: a part is in its building's tile, and the
    Golden Gate's booths are parts of the plaza's footprint, centred in the next tile from most islands."""
    booths = []
    for pring, hh in parts_of.booths(*TOLL_BOOTH_H):
        if hh is None or not (TOLL_BOOTH_H[0] <= hh[0] + hh[1] <= TOLL_BOOTH_H[1]):
            continue
        r = pring - np.array([x0, y0])
        if _area2(r) / 2 <= TOLL_BOOTH_MAX_AREA:
            booths.append(r[:-1].mean(0) if len(r) > 1 and np.allclose(r[0], r[-1]) else r.mean(0))
    if not booths:
        return []
    segs = [np.concatenate([np.asarray(ln, np.float64)[:-1], np.asarray(ln, np.float64)[1:]], 1)
            for ln in lines if len(ln) >= 2]
    if not segs:
        return []
    segs = np.concatenate(segs)
    a, b = segs[:, :2], segs[:, 2:]
    ab = b - a
    L = np.maximum(np.hypot(ab[:, 0], ab[:, 1]), 1e-9)
    cos_max = math.cos(math.radians(TOLL_ROAD_DEG))
    out, seen = [], set()
    for f in layers:
        if f is None or len(f.kind) == 0 or f.attrs.get("height") is None:
            continue
        h = np.asarray(f.attrs["height"], np.float64)
        ids = getattr(f, "ids", None)
        for i in np.flatnonzero(np.isfinite(h) & (h > 0) & (h <= TOLL_ISLAND_MAX_H)).tolist():
            key = ids[i] if ids is not None else None
            if key is not None and key in seen:
                continue
            seen.add(key)
            parts = f.parts(i)
            if not parts or not parts[0] or len(parts[0][0]) < 4:
                continue
            e, n = mapgeom.enu_xy(f, frame, np.asarray(parts[0][0], np.float64))
            box = oriented_box(np.stack([e - x0, n - y0], 1)[:-1])
            c, u, hl, hw = box
            if 2 * hl < TOLL_ISLAND_MIN_L or 2 * hw > TOLL_ISLAND_MAX_W:
                continue
            if not any(_on_box(q, box) for q in booths):
                continue
            tt = np.clip(((c - a) * ab).sum(1) / (L * L), 0.0, 1.0)
            dist = np.hypot(*(a + tt[:, None] * ab - c).T)
            if ((dist <= TOLL_ROAD_M) & (np.abs((ab @ u) / L) >= cos_max)).any():
                out.append(box)
    return out


def _toll_mark(items, islands):
    """items with each toll island (islands: _toll_islands' boxes) marked cls "toll_booth" and the booths standing on
    them left out (West Coast USA's booth stands there instead: toll_booth_rows). Pure."""
    out = []
    for spec, gmin, gmax in items:
        p = spec.ring.mean(0)
        on = next((bx for bx in islands if _on_box(p, bx)), None)
        if on is None:
            out.append((spec, gmin, gmax))
            continue
        h = spec.height or 0.0
        if h <= TOLL_ISLAND_MAX_H:
            _c, _u, hl, hw = oriented_box(spec.ring)
            if 2 * hl >= TOLL_ISLAND_MIN_L and 2 * hw <= TOLL_ISLAND_MAX_W and (spec.min_h or 0) <= 0.5:
                spec.cls = "toll_booth"         # the island itself (not a plaza's flat footprint round it)
            out.append((spec, gmin, gmax))
        elif not (TOLL_BOOTH_H[0] <= h <= TOLL_BOOTH_H[1] and _area2(spec.ring) / 2 <= TOLL_BOOTH_MAX_AREA):
            out.append((spec, gmin, gmax))
    return out


def toll_booth_rows(items, x0, y0):
    """Prop rows (trees.lua rows, group G_TOLLBOOTH) of the toll islands (_toll_mark) among items (rings relative to
    (x0, y0)): West Coast USA's booth along each island, on its lowest ground."""
    rows = []
    for spec, gmin, _gmax in items:
        if spec.cls != "toll_booth":
            continue
        c, u, _hl, _hw = oriented_box(spec.ring)
        if u[0] < 0 or (u[0] == 0 and u[1] < 0):    # one way along the axis for every island: the booths face alike
            u = -u
        yaw = (math.degrees(math.atan2(u[1], u[0])) - 90.0) % 360.0       # (its length is the shape's local y)
        rows.append([G_TOLLBOOTH, round(float(c[0] + x0), 2), round(float(c[1] + y0), 2), round(gmin, 2),
                     round(yaw, 1), 1.0, TOLLBOOTH_H, 0.5])
    return rows


def collider_boxes(items, x0, y0):
    """Boxes [x0, y0, z0, x1, y1, z1] (anchor frame) around each building: the mod does not make the tile's
    mesh solid while the player's car is inside one (roads.lua playerIn)."""
    out = []
    for spec, gmin, gmax in items:
        if deck_plan(spec, gmin, gmax) is not None:     # an open parking deck: driven into on purpose
            continue
        if spec.cls in BRIDGE_CLASSES:                  # a bridge's frame: driven through between its legs
            continue
        r = spec.ring
        out.append([round(float(r[:, 0].min() + x0), 2), round(float(r[:, 1].min() + y0), 2),
                    round(gmin - FOUNDATION_M, 2), round(float(r[:, 0].max() + x0), 2),
                    round(float(r[:, 1].max() + y0), 2), round(gmax + spec.height + 6.0, 2)])
    return out


def garage_sound_boxes(items, x0, y0, kind):
    """The sound spaces (roads.sound_boxes rows [kind, x, y, z, length, width, height, yaw], anchor frame) of the
    tile's parking structures drawn as open decks (deck_plan: the ones driven into): each its footprint's oriented
    box from its lowest ground to its top."""
    out = []
    for spec, gmin, gmax in items:
        if deck_plan(spec, gmin, gmax) is None:
            continue
        c, u, hl, hw = oriented_box(spec.ring)
        _, fl, st, _ = deck_plan(spec, gmin, gmax)
        z0, z1 = gmin - 0.5, gmax + max(spec.height, fl * st) + 1.5
        out += [float(kind), round(float(c[0]) + x0, 2), round(float(c[1]) + y0, 2), round((z0 + z1) / 2, 2),
                round(2 * float(hl), 2), round(2 * float(hw), 2), round(z1 - z0, 2),
                round(math.atan2(float(u[1]), float(u[0])), 4)]
    return out


# The engine's LOD size of a near tile's shape below which its buildings are drawn as the lite mesh (the far tiles' one:
# no window bays, steps, porches, signs, cornices or roof clutter): the engine's own camera LOD (cdae lods), the tile's
# whole shape at once. Calibrated in game (2026-10-18, 1440p, the game's mesh detail at its default 2): a 21 m radius
# probe switched between 450 and 550 m, so a tile of ~190 m radius switches ~500 m from its centre (less with a lower
# mesh detail or screen resolution, as the stock shapes do).
BLD_LOD_PX = 2700.0


def write(path, mesh, collision_only=False):
    """The tile's .cdae (cdae.encode with one primitive per material; collision_only: a never-drawn collision detail).
    A mesh with a 7th element (build_parts with_lod) gets that lite mesh as its coarse detail below BLD_LOD_PX."""
    import cdae
    P, UV, T, M, mats, C = mesh[:6]
    mats = list(mats)
    lods = None
    lite = mesh[6] if len(mesh) > 6 and not collision_only else None     # (None: no lite mesh)
    if lite is not None:
        Pl, UVl, Tl, Ml, matsl, Cl = lite
        for m in matsl:
            if m not in mats:
                mats.append(m)
        remap = np.array([mats.index(m) for m in matsl], np.int64)
        lods = [(BLD_LOD_PX, {"positions": Pl, "tris": Tl, "uvs": UVl, "tri_materials": remap[np.asarray(Ml)],
                              "colors": Cl})]
    return cdae.write(path, P, T, uvs=UV, material=mats, tri_materials=M, object_name="buildings", colors=C,
                      collision_only=collision_only, lods=lods)


DOOR_OUT_M = 0.09            # m the door model's origin stands off the wall: its panel is 6.9 cm behind the origin
                             # (both Building Architect doors), so at 3 cm it was inside the wall, only the frame
                             # showing (Fells Point, 2026-10-17)
STOOP_SHARE = 0.8           # row houses with a front stoop (Baltimore's white marble steps)
# A row house's marble stoop (the user's report #47, 2026-10-17: "the steps model isnt good. Try and find one from the
# offical maps"): built as the official maps' stair blocks are (West Coast USA's stairblock1, Italy's old stairs:
# treads and risers up to a landing between stepped side walls); their own models carry race-track paint and logos
STOOP_H = 0.54              # m: three steps up to the door (Fells Point's on Mapillary, 2026-10-17: low white marble
STOOP_STEPS = 3             # steps, no side walls; ours were four between walls, big white blocks)
STOOP_RISE = 0.18           # m a step (more of them where the street falls away from the house)
STOOP_MAX_STEPS = 7
STOOP_RISE_MAX = 0.23       # m: the steepest outdoor step
STOOP_TREAD = 0.3           # m deep each step
STOOP_LANDING = 0.6         # m deep the landing at the door
STOOP_W = 1.3               # m wide the steps
DOOR_ROAD_M = 30.0          # a front door on the wall facing the nearest road within this
DOOR_CLASSES_SKIP = SHED_CLASSES | INDUSTRIAL_CLASSES
G_DOOR = "door"
FAR_SEND_M = float(os.environ.get("BNE_BLDFAR_SEND_M", "1600"))    # far building tiles (render-only, the lite mesh)
                                                                    # sent within this (Euclidean: roads.tiles_near)
FAR_DROP_M = float(os.environ.get("BNE_BLDFAR_DROP_M", str(FAR_SEND_M + 300)))     # ... dropped beyond
FAR_PER_POSE = 4


MAIN_STREET_CLASSES = {"primary", "secondary", "tertiary"}
BACK_ROAD_CLASSES = {"service", "track", "path", "footway", "cycleway", "pedestrian", "steps", "alley"}
MAIN_STREET_SETBACK_M = 7.0  # m from a main street's edge within which a front wall has a shopfront
MAIN_STREET_SHOPS = 0.75     # the share of such buildings that do


DRIVEWAY_W = 3.2            # m
DRIVEWAY = {"m": "road_asphalt_2lane", "tl": 11.0, "p": 19}     # a road tile chunk's look (roads.LOOK)


def doors(items, lines, x0, y0, drives=None):
    """Prop rows (trees.lua rows, group "door") of a front door per house: on the wall whose middle is
    nearest a road and faces it, a row house's to one side, others' in the middle. lines: [(centreline
    (k, 2) in the anchor frame, half width[, class, urban])]. A building close to an urban main street gets
    a shopfront ground floor instead (spec.shop, most of them). drives (a list) gets a driveway chunk (road
    tile chunk dicts, draped by roadmesh) from each detached house's front to the road's edge."""
    if not lines:
        return []
    segs, smain, shalf, sminor = [], [], [], []
    for ln in lines:
        pts, half = ln[0], ln[1]
        main = len(ln) > 3 and ln[3] and ln[2] in MAIN_STREET_CLASSES     # an urban main street
        p = np.asarray(pts, np.float64)
        if len(p) >= 2:
            segs.append(np.concatenate([p[:-1], p[1:]], 1))
            smain.append(np.full(len(p) - 1, bool(main)))
            shalf.append(np.full(len(p) - 1, float(half)))
            sminor.append(np.full(len(p) - 1, len(ln) > 2 and ln[2] in BACK_ROAD_CLASSES))
    if not segs:
        return []
    S = np.concatenate(segs)                             # (n, 4) x0 y0 x1 y1
    SMAIN, SHALF, SMINOR = np.concatenate(smain), np.concatenate(shalf), np.concatenate(sminor)
    A, B = S[:, :2], S[:, 2:]
    D = B - A
    L2 = np.maximum((D * D).sum(1), 1e-12)
    rows = []
    others = [(spec, spec.ring + np.array([x0, y0])) for spec, _, _ in items if (spec.min_h or 0.0) <= 0.5]
    for spec, gmin, gmax in items:
        no_door = False
        if spec.plain or spec.floors >= 7 or spec.band in (BAND_OFFICE, BAND_OFFICE2) or spec.wall:
            if not spec.wall or spec.plain or spec.floors >= 7:
                no_door = True
        # shops and big boxes: their sign faces the street, not the alley or the parking aisle behind
        storefront = (spec.shop or spec.windows is False) and not spec.plain and spec.roof == "flat"
        if no_door and not storefront:
            continue
        ring = drawn_ring(spec) + np.array([x0, y0])
        best = None
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            e = b - a
            Le = float(math.hypot(*e))
            if Le < 2.5:
                continue
            mid = (a + b) / 2
            t = np.clip(((mid - A) * D).sum(1) / L2, 0, 1)
            q = A + t[:, None] * D
            dist = np.hypot(*(q - mid).T)
            k = int(dist.argmin())
            if dist[k] > DOOR_ROAD_M:
                continue
            out = np.array([e[1], -e[0]]) / Le
            if float(np.dot(q[k] - mid, out)) <= 0:
                continue                                 # the wall faces away from that road
            rank = (bool(SMINOR[k]) if storefront else False, float(dist[k]))
            if best is None or rank < best[0]:
                best = (rank, a, e / Le, Le, out, i, bool(SMAIN[k]) and dist[k] < SHALF[k] + MAIN_STREET_SETBACK_M,
                        k)
        if best is None:
            continue
        _, a, d, Le, out, edge, on_main, kseg = best
        if spec.windows is False:
            spec.front = edge                   # a big box: its storefront faces the road
            if storefront:
                spec.sign = edge
            rows.extend(_sign_in_open(_wall_sign(spec, a, d, Le, out, gmax), spec, d, out, others))
            continue
        if spec.cls in RESTAURANT_CLASSES and spec.floors == 1:
            rows.extend(_sign_in_open(_wall_sign(spec, a, d, Le, out, gmax), spec, d, out, others))
        if spec.shop:
            if storefront and not spec.wall:
                spec.sign = edge                # the sign band over the street wall's shopfront
            continue
        if no_door:
            continue
        if on_main and not spec.wall and spec.roof == "flat" and spec.cls not in CHURCH_CLASSES \
                and _hash01(spec.seed, "shop") < MAIN_STREET_SHOPS:
            spec.shop = True                    # a main street's ground floor: a shopfront, no house door
            if not spec.plain:
                spec.sign = edge
            continue
        narrow = Le < 9.0
        f = (0.22 if _hash01(spec.seed, "door") < 0.5 else 0.78) if narrow else 0.5
        spec.door = (edge, Le * f)          # _walls leaves the ground floor's band plain there
        # brick row houses (an eastern city's): white marble steps up to the door
        if narrow and spec.band == BAND_BRICK and spec.roof == "flat" and _hash01(spec.seed, "stoop") < STOOP_SHARE:
            spec.stoop = (edge, Le * f)
        if drives is not None and spec.wall and Le >= 7.0:
            # the driveway beside the door, straight out to the road's edge
            g = 0.82 if f <= 0.5 else 0.18
            p0 = a + d * (Le * g) + out * 0.2
            t = float(np.clip(((p0 - A[kseg]) * D[kseg]).sum() / L2[kseg], 0, 1))
            q = A[kseg] + t * D[kseg]
            v = p0 - q
            dl = float(np.hypot(*v))
            if SHALF[kseg] + 1.0 < dl < DOOR_ROAD_M:
                p1 = q + v / dl * (SHALF[kseg] + 0.1)
                n = max(2, int(dl // 4.0) + 1)
                pts = [p0 + (p1 - p0) * (j / (n - 1)) for j in range(n)]
                drives.append({"m": DRIVEWAY["m"], "w": DRIVEWAY_W, "tl": DRIVEWAY["tl"], "p": DRIVEWAY["p"],
                               "n": [[round(float(pp[0]), 3), round(float(pp[1]), 3), 0.0] for pp in pts]})
        if spec.roof != "flat" and not narrow and _hash01(spec.seed, "fence") < FENCE_SHARE:
            rows.extend(_front_fence(spec, a, d, Le, f, A[kseg], D[kseg], L2[kseg], SHALF[kseg], gmin,
                                     drives is not None and spec.wall and Le >= 7.0, (A, B, SHALF)))
        p = a + d * (Le * f) + out * DOOR_OUT_M
        gd = door_ground(spec, edge, Le * f, gmax) if spec.stoop else gmax     # a stoop's door: over its own ground
        base = gd + (STOOP_H if spec.stoop else 0.0) + (PORCH_DECK_M if spec.porch else 0.0)   # (the deck / stoop)
        rows.append([G_DOOR, round(float(p[0]), 2), round(float(p[1]), 2), round(base, 2),
                     round(math.degrees(math.atan2(d[1], d[0])) % 360.0, 1), 1.0, 2.2,
                     round(_hash01(spec.seed, "dv"), 3)])
    return rows


# Company signs on walls (West Coast USA's, for its fictional firms): (prop group, width m, height m, panel thin
# along its local x (True) or y (False)). Warehouses: Logistics or Rentabox; big retail boxes: Lemon; one-storey
# restaurants: Turboburger.
WALL_SIGNS = {"logistics": ("wallsign_logistics", 10.8, 5.8, True), "rentabox": ("wallsign_rentabox", 8.2, 5.7, True),
              "lemon": ("wallsign_lemon", 11.5, 3.6, True), "burger": ("wallsign_turboburger", 5.8, 2.4, False)}
WALL_SIGN_SHARE = 0.6
RESTAURANT_CLASSES = {"restaurant", "fast_food", "cafe"}


def _sign_in_open(rows, spec, d, out, others):
    """rows (a _wall_sign) kept only when the panel and the air in front of it are clear of the other buildings:
    a sign on a wall hidden behind a neighbour stood out through the neighbour at the wall's angle (Stelvio
    pass, a big box's sign through the shop in front of it, the user's report 2026-10-17)."""
    if not rows:
        return rows
    _, x, y, _, _, _, h, _ = rows[0]
    w = WALL_SIGNS[next(k for k, v in WALL_SIGNS.items() if v[0] == rows[0][0])][1] * h / \
        WALL_SIGNS[next(k for k, v in WALL_SIGNS.items() if v[0] == rows[0][0])][2]
    c = np.array([x, y])
    probes = [c + d * (w / 2 * f) + out * o for f in (-1.0, 0.0, 1.0) for o in (0.0, 1.5)]
    for other, ring in others:
        if other is spec:
            continue
        if any(_inside(ring, p) for p in probes):
            return []
    return rows


def _wall_sign(spec, a, d, Le, out, gmax):
    """A company sign centred high on a big box's or restaurant's street wall (WALL_SIGNS), if it fits."""
    if _hash01(spec.seed, "wallsign") >= WALL_SIGN_SHARE:
        return []
    if spec.cls in RESTAURANT_CLASSES:
        key = "burger"
    elif spec.wall and str(spec.wall).startswith(METALWALL):
        key = "logistics" if _hash01(spec.seed, "ws") < 0.5 else "rentabox"
    else:
        key = "lemon"
    group, w, h0, thin_x = WALL_SIGNS[key]
    h = min(h0, spec.height * 0.45)                     # scaled down to the wall (trees.lua scales to the height)
    w *= h / h0
    if h < 1.8 or Le < w + 2.0:
        return []
    zc = gmax + spec.height - 0.6 - h / 2               # its top 0.6 m under the parapet
    p = a + d * (Le / 2) + out * 0.35
    # the logo faces the model's local -x (Lemon, Logistics, Rentabox: seen in game 2026-10-14) or its -y
    yaw = math.degrees(math.atan2(out[1], out[0])) + (180.0 if thin_x else 90.0)
    return [[group, round(float(p[0]), 2), round(float(p[1]), 2), round(float(zc), 2), round(yaw % 360.0, 1), 1.0,
             round(h, 2), 0.0]]


G_FENCE = "fence"           # trees.lua: East Coast USA's low wooden fence (eca_bld_wood_fence_c, 5.4 m, along x)
FENCE_SEG_M = 5.4
FENCE_SHARE = 0.3           # detached houses with a fence along the front of their yard (hashed)
FENCE_BACK_M = 3.2          # m from the road's edge (the back of the sidewalk)
FENCE_MIN_YARD_M = 2.5      # the front yard at least this deep between the fence and the house
FENCE_CLEAR_M = 2.0         # m every piece keeps from any road's edge (a house on a bend or a corner: its fence ran
                            # along one road out across another, GitHub #105 Falcon Drive, Coquitlam)


def _front_fence(spec, a, d, Le, f, A, D, L2, half, z, driveway, roads=None):
    """Prop rows of a fence along the front of a house's yard, parallel to the road at the back of the sidewalk,
    a little wider than the house, with gaps for the path to the door and the driveway. roads: (starts (n, 2), ends
    (n, 2), half widths (n,)) of the roads near: no piece within FENCE_CLEAR_M of their edges."""
    mid = a + d * (Le / 2)
    t = float(np.clip(((mid - A) * D).sum() / L2, 0, 1))
    q = A + t * D
    v = mid - q
    dist = float(np.hypot(*v))
    if dist - (half + FENCE_BACK_M) < FENCE_MIN_YARD_M:
        return []
    nv = v / dist
    dr = D / math.sqrt(L2)
    centre = q + nv * (half + FENCE_BACK_M)
    yaw = round(math.degrees(math.atan2(dr[1], dr[0])) % 360.0, 1)
    gaps = [float((a + d * (Le * f) - centre) @ dr)]                          # the path to the door
    if driveway:
        gaps.append(float((a + d * (Le * (0.82 if f <= 0.5 else 0.18)) - centre) @ dr))
    n = max(1, int(math.ceil((Le + 3.0) / FENCE_SEG_M)))
    rows = []
    for k in range(n):
        sk = (k - (n - 1) / 2) * FENCE_SEG_M
        if any(abs(sk - g) < FENCE_SEG_M / 2 + 0.6 for g in gaps):
            continue
        p = centre + dr * sk
        if roads is not None and len(roads[0]):
            ends = np.array([p - dr * (FENCE_SEG_M / 2), p, p + dr * (FENCE_SEG_M / 2)])
            fast.require(0, "points_near_segments")
            if np.asarray(fast._ef.points_near_segments(ends, np.ascontiguousarray(roads[0], np.float64),
                                                        np.ascontiguousarray(roads[1], np.float64),
                                                        np.asarray(roads[2], np.float64) + FENCE_CLEAR_M)).any():
                continue
        rows.append([G_FENCE, round(float(p[0]), 2), round(float(p[1]), 2), round(float(z), 2), yaw, 1.0, 1.3,
                     round(_hash01(spec.seed, "fr", k), 3)])
    return rows


def road_tile_items(frame, tx, ty, md, ground, decks=None, skip=frozenset()):
    """tile_buildings of road tile (tx, ty), from the map cells under it. decks: the bridge decks near it
    [(xy (k, 2), z (k,))] in the anchor frame (roads.BridgeProfiles.parts): buildings they run through get a way.
    skip: building ids left out (gas station canopies)."""
    import roads
    x0, y0, x1, y1 = roads.tile_bbox(tx, ty)
    cds = [md.cell(k, layers=("buildings", "building_parts", "roads", "infrastructure", "water"), kind="demand")
           for k in roads.cells_of_tile(frame, tx, ty, margin=0.0)]
    return tile_buildings(frame, x0, y0, x1, y1, [c.layers.get("buildings") for c in cds], ground,
                          part_layers=[c.layers.get("building_parts") for c in cds],
                          west_town=district_town(frame, tx, ty, md),
                          roads_xy=road_lines(frame, [c.layers.get("roads") for c in cds], x0, y0),
                          ways_xy=way_lines(frame, [c.layers.get("roads") for c in cds], x0, y0),
                          tunnels_xy=tunnel_lines(frame, [c.layers.get("roads") for c in cds], x0, y0),
                          garages_xy=garage_rings(frame, [c.layers.get("infrastructure") for c in cds], x0, y0),
                          supports=support_rings(frame, [c.layers.get("infrastructure") for c in cds], x0, y0),
                          bridges_xy=bridge_lines(frame, [c.layers.get("roads") for c in cds], x0, y0),
                          water_xy=water_rings(frame, [c.layers.get("water") for c in cds], x0, y0),
                          decks=[(np.asarray(d[0], np.float64)[:, :2] - [x0, y0], np.asarray(d[1], np.float64))
                                 + tuple(d[2:3]) for d in decks] if decks else None, skip=skip)


TOWN_COVERAGE = 0.16        # a tile whose footprints cover less than this share is a town / suburb (heights)
# Small towns of the arid West (Bishop, Moab: one-storey main streets with false fronts, ranch houses), told
# from cities by the footprint share of the district around a tile (1 km: Bishop 0.14, Moab 0.11; Frederick
# 0.22, Fells Point 0.27): a tile's own share cannot (a main-street block is as dense as a city's).
DISTRICT_M = 1000.0
WEST_TOWN_COVERAGE = 0.18
WEST_TOWN_ONE_STOREY = 0.75     # share of predicted one-storey buildings there
FALSE_FRONT_M = 1.2             # m of parapet over a one-storey shop there
POLES_COVERAGE = 0.30       # ... and its streets get wired wooden poles (furniture; Ocean City's photos: poles
                            # everywhere; Fells Point's dense core ~0.45 keeps lamp posts)


def coverage(items, tile=256.0):
    """The share of the tile covered by building footprints."""
    return sum(_area2(s.ring) / 2 for s, _, _ in items if (s.min_h or 0) <= 0.5) / (tile * tile)


def centres(items, tx, ty):
    """The anchor-frame (x, y) centres of a road tile's buildings (road_tile_items' rings are tile-local)."""
    import roads
    x0, y0, _, _ = roads.tile_bbox(tx, ty)
    return [(float(s.ring[:, 0].mean() + x0), float(s.ring[:, 1].mean() + y0)) for s, _, _ in items]


def road_tile(frame, tx, ty, md, ground, lines=None, props=None, items=None):
    """The buildings of road tile (tx, ty) (roads.TILE squares of the anchor frame): (mesh, collider boxes) or
    None. md: mapdata.MapData; ground(lat, lon): the physics ground without the road lift. lines (the road
    centrelines, roads.build_tile) and props (its prop rows): the houses' front doors are added to props.
    items: road_tile_items' result when the caller has it."""
    import roads
    x0, y0, x1, y1 = roads.tile_bbox(tx, ty)
    if items is None:
        items = road_tile_items(frame, tx, ty, md, ground)
    if props is not None and lines:
        props.extend(doors(items, lines, x0, y0))
    meshes = build_parts(items, with_lod=True)
    if not meshes:
        return None
    return meshes, collider_boxes(items, x0, y0), collision_mesh(items)


def far_tile(frame, tx, ty, md, ground, out_dir, vfs_dir, aid, lamp_region=None):
    """A far (render-only) building tile written to the session folder: (vfs path of its JSON, building
    count) or None. The lite mesh of the same buildings as road_tile's."""
    import json
    import os
    import roads
    x0, y0, x1, y1 = roads.tile_bbox(tx, ty)
    cds = [md.cell(k, layers=("buildings", "building_parts", "infrastructure", "roads", "water"), kind="demand")
           for k in roads.cells_of_tile(frame, tx, ty, margin=0.0)]
    items = tile_buildings(frame, x0, y0, x1, y1, [c.layers.get("buildings") for c in cds], ground,
                           part_layers=[c.layers.get("building_parts") for c in cds],
                           west_town=district_town(frame, tx, ty, md),
                           garages_xy=garage_rings(frame, [c.layers.get("infrastructure") for c in cds], x0, y0),
                           supports=support_rings(frame, [c.layers.get("infrastructure") for c in cds], x0, y0),
                           bridges_xy=bridge_lines(frame, [c.layers.get("roads") for c in cds], x0, y0),
                           water_xy=water_rings(frame, [c.layers.get("water") for c in cds], x0, y0),
                           ml_roads_xy=road_lines(frame, [c.layers.get("roads") for c in cds], x0, y0),
                           ways_xy=way_lines(frame, [c.layers.get("roads") for c in cds], x0, y0, size=x1 - x0))
    mesh = build(items, lite=True)
    data = {"v": 1}
    lamps = far_lamps(frame, [c.layers.get("roads") for c in cds], x0, y0, x1 - x0,
                      built_squares(items, 0.0, 0.0, x1 - x0), ground, lamp_region)
    if lamps:
        data["lamps"] = lamps
    if mesh is not None:
        shape = f"bldfar_{aid}_{tx}_{ty}.cdae"
        write(os.path.join(out_dir, shape), mesh)
        data["shape"] = f"{vfs_dir}/{shape}"
        data["pos"] = [x0, y0, 0.0]
    name = f"bldfar_{aid}_{tx}_{ty}.json"
    path = os.path.join(out_dir, name)
    tmp = staging.tmp_for(path)
    with open(tmp, "w") as f:
        f.write(json.dumps(data))       # (json.dump to a file: the pure-Python encoder)
    staging.replace(tmp, path)
    return f"{vfs_dir}/{name}", len(items)


# ------------------------------------------------------------------------------------------- far street lamps
# The street lamps past the road tiles (the user, 2026-10-27: "extend the street and window lights as far as the
# buildings render"): the far and skyline tiles carry their streets' lamps as glow points (trees.lua's lampglow group,
# night only), placed along the mapped roads by furniture.py's spacing: the town streets in built-up squares (at least
# FAR_LAMP_MIN_BLD buildings in its 256 m square: the near rule's land cover is not at hand here), the highways there
# and every ramp. FAR_LAMP_Z over the ground, over the carriageway's edge.
FAR_LAMP_MIN_BLD = 8
FAR_LAMP_Z = 8.0
FAR_LAMP_MAX = 4000         # a tile's most (the skyline's 1 km squares: a dense city's ~3000)


def built_squares(items, x0, y0, size, sq=256.0):
    """The set of (i, j) 256 m squares of a tile (x0, y0 its corner; items' rings tile-relative) holding at least
    FAR_LAMP_MIN_BLD buildings."""
    import collections
    n = collections.Counter()
    for spec, _gmin, _gmax in items:
        c = np.asarray(spec.ring, np.float64)[:, :2].mean(0)
        n[(int(c[0] // sq), int(c[1] // sq))] += 1
    return {k for k, v in n.items() if v >= FAR_LAMP_MIN_BLD}


def far_lamps(frame, road_layers, x0, y0, size, built, ground, region=None, sq=256.0):
    """The lamps of a far tile's streets: [x, y, z] in the anchor frame (rounded), at most FAR_LAMP_MAX. built: the
    tile's built-up 256 m squares (built_squares); ground(lat, lon) the ground (arrays); region: furniture.lamp_region's
    (the spacing)."""
    import furniture
    import roads as roadsmod
    import fast
    if not built:
        return []
    table = furniture.lamp_spacing(region)
    hw = "na" if furniture._na(region) else None
    pts, off, spacing, phase, side, lit_any = [], [0], [], [], [], []
    for f in road_layers:
        if f is None or len(f.kind) == 0:
            continue
        specs = roadsmod.road_specs(f)
        bb = f.boxes(frame)
        near = np.flatnonzero((bb[:, 2] - x0 >= -20) & (bb[:, 0] - x0 <= size + 20)
                              & (bb[:, 3] - y0 >= -20) & (bb[:, 1] - y0 <= size + 20))
        for i in near.tolist():
            sp = specs[i]
            if sp is None:
                continue
            highway = sp.cls in furniture.HIGHWAY_CLASSES
            if highway and not furniture.highway_lit(region, sp.link, True):
                continue
            if highway:
                every = (furniture.LINK_LAMP_SPACING if sp.link else furniture.HIGHWAY_LAMP_SPACING)[hw]
                rules = [(every, -float(sp.dir) if sp.dir else (1.0 if furniture._h01(str(f.ids[i]), "pside") < 0.5
                                                               else -1.0), 0.0)]
            elif sp.cls in table:
                every, both = table[sp.cls]
                sd = 1.0 if furniture._h01(str(f.ids[i]), "side") < 0.5 else -1.0
                rules = [(every, sd, 0.0)] + ([(every, -sd, every / 2)] if both else [])
            else:
                continue
            ph0 = furniture._h01(str(f.ids[i]), "pphase")
            reach = max(1.0, sp.width / 2 - 1.0)
            parts = roadsmod.ground_parts(f, i) if roadsmod.level_ranges(f, i) else [None]
            for part in parts:
                if part is None:
                    a = f.coord_off[f.ring_off[f.part_off[i]]]
                    b = f.coord_off[f.ring_off[f.part_off[i + 1]]]
                    part = f.lonlat[a:b]
                if len(part) < 2:
                    continue
                e, n = mapgeom.enu_xy(f, frame, np.asarray(part, np.float64))
                xy = np.stack([np.asarray(e, np.float64), np.asarray(n, np.float64)], 1)
                for every, sd, shift in rules:
                    pts.append(xy)
                    off.append(off[-1] + len(xy))
                    spacing.append(every)
                    phase.append(ph0 * every + shift)
                    side.append(sd * reach)
                    lit_any.append(bool(highway and sp.link))
    if not pts:
        return []
    fast.require(87, "lamp_stations")
    st = fast._ef.lamp_stations(np.ascontiguousarray(np.concatenate(pts)), np.asarray(off, np.int64),
                                np.asarray(spacing, np.float64), np.asarray(phase, np.float64),
                                np.asarray(side, np.float64), [x0, y0, x0 + size, y0 + size])
    if not len(st):
        return []
    keep = [k for k in range(len(st)) if lit_any[int(st[k, 2])]
            or (int((st[k, 0] - x0) // sq), int((st[k, 1] - y0) // sq)) in built]
    st = st[keep][:FAR_LAMP_MAX]
    if not len(st):
        return []
    lat, lon, _ = frame.enu_to_geodetic(st[:, 0], st[:, 1], 0.0)
    z = np.asarray(ground(np.asarray(lat), np.asarray(lon)), np.float64) + FAR_LAMP_Z
    return np.round(np.c_[st[:, 0], st[:, 1], z], 1).tolist()


# ------------------------------------------------------------------------------------------- the skyline
# Past the far buildings, a lighter ring out to SKY_SEND_M (the user, 2026-10-27: "see how far away we can render
# buildings"): one mesh per SKY_UNIT x SKY_UNIT road tiles (1 km), only the buildings big enough to see that far
# (SKY_MIN_H tall or SKY_MIN_AREA_M2 large), the lite mesh, the tallest first up to the 16-bit vertex budget. Far
# buildings to 4.8 km as 256 m tiles cost +37 % frame time at Federal Hill (1129 objects, ~0.7 GB of meshes); a
# skyline tile stands for 16 of them and holds a fraction of their buildings. The far buildings are sent for every
# tile of a skyline square that comes within FAR_SEND_M (vegetation.Tracker group), and the mod shows a skyline tile
# only while none of its squares' far buildings are there: no overlap, no hole.
SKY_UNIT = 4
SKY_SEND_M = float(os.environ.get("BNE_SKY_SEND_M", "4800"))
SKY_DROP_M = SKY_SEND_M + 800.0
SKY_PER_POSE = 2
SKY_MIN_AREA_M2 = 40.0      # (sheds left out)
SKY_BOX_AREA_M2 = 600.0     # a footprint smaller than this is drawn as its oriented box
SKY_MAX_CORNERS = 12        # a larger one with more corners: every k-th kept


# The mean colour of each building material's base texture (tools: a 64 x 64 resize of each material's
# baseColorMap from the game's content zips, 2026-10-27): a skyline building is drawn in its facade's and its roof's
# mean colour times their tints (MARBLE, nearly white, sampled at one texel: the vertex colour is the colour)
MAT_MEAN = {"bne_bld_facade": (0.679, 0.676, 0.683), "bne_bld_shop": (0.669, 0.691, 0.699),
            "bne_bld_office": (0.47, 0.493, 0.492), "bne_bld_roof_flat": (0.432, 0.431, 0.43),
            "bne_bld_roof_slate": (0.359, 0.36, 0.388), "bne_bld_roof_shingle": (0.585, 0.577, 0.571),
            "bne_bld_siding": (0.822, 0.822, 0.822), "bne_bld_stucco": (0.795, 0.795, 0.795),
            "bne_bld_roof_metal": (0.642, 0.642, 0.642), "bne_bld_panel": (0.815, 0.815, 0.814),
            "bne_bld_metalwall": (0.841, 0.841, 0.841), "bne_bld_stone": (0.83, 0.818, 0.803),
            "bne_bld_marble": (0.978, 0.978, 0.978), "bne_bld_brick": (0.717, 0.569, 0.513)}
SKY_UV = (0.5, 0.5)
# the skyline walls' windows (tools/build_sky_windows.py: SKY_WIN_BAYS x SKY_WIN_FLOORS of them in its map), lit at
# night (nightlights.lua; the user, 2026-10-27: "extend the street and window lights as far as the buildings render")
SKY_BAY_M = 3.5
SKY_WIN_BAYS = SKY_WIN_FLOORS = 16


def _sky_colour(look, f):
    """RGBA bytes of a LOOKS name drawn plain: its material's mean colour x its tint x the jitter f."""
    if look not in LOOKS:
        return (150, 150, 150, 255)
    mat, tint = LOOKS[look]
    mean = MAT_MEAN.get(mat, (0.7, 0.7, 0.7))
    return tuple(int(max(0, min(255, round(255 * m * t * f)))) for m, t in zip(mean, tint)) + (255,)


def _sky_mesh(items):
    """The skyline mesh of [(Spec, gmin, gmax)]: per building a quad per wall and its flat roof, in MARBLE at one
    texel, coloured per vertex (~4 per corner + the roof's corners: a row house is 20 vertices, the lite mesh ~72).
    (positions, uvs, tris, tri material indices, MATERIALS, colours) as build, or None."""
    m = Mesh()
    mi = MATERIALS.index(MARBLE)
    ms = MATERIALS.index(SKY_WALL)
    uv = SKY_UV
    for spec, gmin, gmax in items:
        ring = np.asarray(spec.ring, np.float64)[:, :2]
        if len(ring) < 3:
            continue
        if _area2(ring) < 0:
            ring = ring[::-1]
        f = 1.0 + JITTER * (2.0 * _hash01(spec.seed, "jit") - 1.0)
        windows = spec.windows is not False
        wall = (ms if windows else mi, _sky_colour(spec.facade, f))
        roof = (mi, _sky_colour(spec.roof_mat or ROOF_FLAT, f))
        z0 = gmin - SKIRT_LITE_M
        z1 = base_z(spec, gmin, gmax) + float(spec.min_h or 0.0) + float(spec.height or 0.0)
        if z1 <= z0 + 0.5:
            continue
        n = len(ring)
        # (windows: u along the walls in bays, v up in storeys from the ground, both over the map's count)
        zb = gmin
        fh = float(spec.floor_h or FLOOR_M) * SKY_WIN_FLOORS
        bw = SKY_BAY_M * SKY_WIN_BAYS
        v0, v1 = (z0 - zb) / fh, (z1 - zb) / fh
        u = 0.0
        for i in range(n):
            a, b = ring[i], ring[(i + 1) % n]
            if windows:
                du = float(math.hypot(*(b - a))) / bw
                m.quad((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1),
                       (u, v0), (u + du, v0), (u + du, v1), (u, v1), wall)
                u += du
            else:
                m.quad((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1), uv, uv, uv, uv, wall)
        m.poly([(p[0], p[1], z1) for p in ring], [uv] * n, roof)
    if not m.n:
        return None
    P, UV, T, M, C = m.arrays()
    return P, UV, T, M, MATERIALS, C


def skyline_tile(frame, ux, uy, md, ground, out_dir, vfs_dir, aid, lamp_region=None):
    """A skyline tile (render-only) written to the session folder: (vfs path of its JSON, building count) or None
    when its cells are not at hand (fetched in the background: asked again later). Key (ux, uy): SKY_UNIT x SKY_UNIT
    road tiles from road tile (ux * SKY_UNIT, uy * SKY_UNIT)."""
    import json
    import roads
    x0, y0, _, _ = roads.tile_bbox(ux * SKY_UNIT, uy * SKY_UNIT)
    x1, y1 = x0 + SKY_UNIT * roads.TILE, y0 + SKY_UNIT * roads.TILE
    keys = sorted({k for i in range(SKY_UNIT) for j in range(SKY_UNIT)
                   for k in roads.cells_of_tile(frame, ux * SKY_UNIT + i, uy * SKY_UNIT + j, margin=0.0)}, key=str)
    # (and the roads: their street lamps, far_lamps)
    if not all(md.has(k, "buildings") and md.has(k, "roads") for k in keys):
        for k in keys:                      # (the background lane: never before what the player is near)
            md.prefetch(k, layers=("buildings", "roads"))
        return None
    cds = [md.cell(k, layers=("buildings", "roads"), kind="background") for k in keys]
    layers = [c.layers.get("buildings") for c in cds]
    items = tile_buildings(frame, x0, y0, x1, y1, layers, ground, min_size=(1e9, SKY_MIN_AREA_M2))
    lamps = far_lamps(frame, [c.layers.get("roads") for c in cds], x0, y0, x1 - x0,
                      built_squares(items, 0.0, 0.0, x1 - x0), ground, lamp_region)
    for spec, _gmin, _gmax in items:        # (the simplest shape: a box or a few corners, a flat roof, nothing on it)
        ring = np.asarray(spec.ring, np.float64)
        if len(ring) > 4 and abs(_area2(ring)) / 2 < SKY_BOX_AREA_M2:
            c, u, hl, hw = oriented_box(ring)
            v = np.array([-u[1], u[0]])
            ring = np.array([c - u * hl - v * hw, c + u * hl - v * hw, c + u * hl + v * hw, c - u * hl + v * hw])
        elif len(ring) > SKY_MAX_CORNERS:     # (rings are open: the last corner is not the first again)
            ring = ring[::int(np.ceil(len(ring) / SKY_MAX_CORNERS))]
        spec.ring = ring
        spec.roof, spec.steeple, spec.stoop, spec.porch, spec.sign, spec.door = "flat", None, None, None, None, None
    items.sort(key=lambda it: (-(float(it[0].height or 0.0) + float(it[0].min_h or 0.0)), str(it[0].seed)))
    picked, n = [], 0
    for it in items:                        # (the tallest first, up to the 16-bit vertex budget: 5 per corner)
        k = 5 * len(it[0].ring)
        if n + k > MAX_MESH_VERTS:
            break
        picked.append(it)
        n += k
    mesh = _sky_mesh(picked) if picked else None
    data = {"v": 1}
    if mesh is not None:
        shape = f"bldsky_{aid}_{ux}_{uy}.cdae"
        if lamps:
            data["lamps"] = lamps
        write(os.path.join(out_dir, shape), mesh)
        data["shape"] = f"{vfs_dir}/{shape}"
        data["pos"] = [x0, y0, 0.0]
    name = f"bldsky_{aid}_{ux}_{uy}.json"
    path = os.path.join(out_dir, name)
    tmp = staging.tmp_for(path)
    with open(tmp, "w") as f:
        f.write(json.dumps(data))
    staging.replace(tmp, path)
    return f"{vfs_dir}/{name}", len(picked) if mesh is not None else 0


# ------------------------------------------------------------------------------------------- far bridges
# The road bridges past the road tiles (their decks only came with the road tiles, ~0.8 km: a city's bridges were
# gone beyond that): one render-only slab per road tile that has bridge decks, out to BRIDGE_FAR_SEND_M, from the same
# deck solver as the road tiles (roads.BridgeProfiles), each part's segments in the tile of their midpoint (offsets
# from the whole part: tiles meet without a gap). A slab: the deck's top, its sides and its underside down to its
# girders, plain grey (MARBLE at one texel, vertex colours). The mod hides one while its road tile has a deck.
BRIDGE_FAR_SEND_M = float(os.environ.get("BNE_BRIDGE_FAR_SEND_M", "4800"))
BRIDGE_FAR_DROP_M = BRIDGE_FAR_SEND_M + 500.0
BRIDGE_FAR_PER_POSE = 2
BRIDGE_TOP_RGBA = (92, 92, 94, 255)        # the deck's asphalt from afar
BRIDGE_SIDE_RGBA = (176, 174, 168, 255)    # its concrete


def _bridge_slab(parts, x0, y0, x1, y1):
    """The slab mesh of the bridge parts [(xy (k, 2), z (k,), width m)] (anchor frame) for the segments whose midpoint
    is in [x0, x1) x [y0, y1), relative to (x0, y0), or None."""
    import roadmesh
    m = Mesh()
    mi = MATERIALS.index(MARBLE)
    uv = SKY_UV
    top, side = (mi, BRIDGE_TOP_RGBA), (mi, BRIDGE_SIDE_RGBA)
    depth = roadmesh.DECK_DEPTH + roadmesh.GIRDER_D
    for xy, z, w in parts:
        xy = np.asarray(xy, np.float64)[:, :2]
        z = np.asarray(z, np.float64)
        if len(xy) < 2:
            continue
        nrm = roadmesh._normals(xy) * (w / 2.0)
        L, R = xy + nrm - (x0, y0), xy - nrm - (x0, y0)
        mid = (xy[:-1] + xy[1:]) / 2.0
        inside = (mid[:, 0] >= x0) & (mid[:, 0] < x1) & (mid[:, 1] >= y0) & (mid[:, 1] < y1)
        for i in np.flatnonzero(inside):
            j = i + 1
            a, b, zt0, zt1 = L[i], L[j], z[i], z[j]
            c, d = R[i], R[j]
            zb0, zb1 = zt0 - depth, zt1 - depth
            m.quad((a[0], a[1], zt0), (c[0], c[1], zt0), (d[0], d[1], zt1), (b[0], b[1], zt1), uv, uv, uv, uv, top)
            m.quad((a[0], a[1], zb0), (a[0], a[1], zt0), (b[0], b[1], zt1), (b[0], b[1], zb1), uv, uv, uv, uv, side)
            m.quad((d[0], d[1], zb1), (d[0], d[1], zt1), (c[0], c[1], zt0), (c[0], c[1], zb0), uv, uv, uv, uv, side)
            m.quad((c[0], c[1], zb0), (a[0], a[1], zb0), (b[0], b[1], zb1), (d[0], d[1], zb1), uv, uv, uv, uv, side)
    if not m.n:
        return None
    P, UV, T, M, C = m.arrays()
    return P, UV, T, M, MATERIALS, C


def far_bridge_tile(frame, tx, ty, md, bridges, out_dir, vfs_dir, aid):
    """A far bridge tile (render-only) written to the session folder: (vfs path of its JSON, part count) or None when
    its road cells are not at hand (fetched in the background: asked again later). bridges: the anchor's
    roads.BridgeProfiles."""
    import json
    import roads
    x0, y0, x1, y1 = roads.tile_bbox(tx, ty)
    keys = roads.cells_of_tile(frame, tx, ty, margin=0.0)
    if not all(md.has(k, "roads") for k in keys):
        for k in keys:
            md.prefetch(k, layers=("roads",))
        return None
    cells = [md.cell(k, layers=("roads",), kind="background").layers.get("roads") for k in keys]
    parts = [(xy, z, float(spec.width)) for _fid, spec, xy, z, _lon in bridges.parts(cells, (x0, y0, x1, y1))]
    mesh = _bridge_slab(parts, x0, y0, x1, y1) if parts else None
    data = {"v": 1}
    if mesh is not None:
        shape = f"brfar_{aid}_{tx}_{ty}.cdae"
        write(os.path.join(out_dir, shape), mesh)
        data["shape"] = f"{vfs_dir}/{shape}"
        data["pos"] = [x0, y0, 0.0]
    name = f"brfar_{aid}_{tx}_{ty}.json"
    path = os.path.join(out_dir, name)
    tmp = staging.tmp_for(path)
    with open(tmp, "w") as f:
        f.write(json.dumps(data))
    staging.replace(tmp, path)
    return f"{vfs_dir}/{name}", len(parts) if mesh is not None else 0


# ------------------------------------------------------------------------------------------- the ground photo

PHOTO_GROW_M = 1.5          # m a footprint is grown on the ground photo (edges, eaves) ...
PHOTO_LEAN = 0.12           # ... plus this share of its height (the photo's lean of tall buildings), at most
PHOTO_LEAN_MAX_M = 6.0      # ... this
PHOTO_FILL_M = 12.0         # m: the blur that fills them from the ground around


def photo_mask(frame, x0, y0, size, px, md):
    """A (px, px) bool mask (row 0 = north) of the ground photo's square [x0, x0 + size) x [y0, y0 + size)
    (anchor frame) where buildings stand, grown by their edges and lean: there the photo shows roofs and
    walls that the 3D buildings now stand on or next to. None without buildings."""
    from PIL import Image, ImageDraw
    import mapcells
    la, lo, _ = frame.enu_to_geodetic(np.array([x0, x0 + size, x0, x0 + size]),
                                      np.array([y0, y0, y0 + size, y0 + size]), 0.0)
    keys = mapcells.cells_for_bbox(float(la.min()), float(lo.min()), float(la.max()), float(lo.max()))
    img = Image.new("L", (px, px), 0)
    draw = ImageDraw.Draw(img)
    mpp = size / px
    seen = set()
    n = 0
    for k in keys:
        f = md.cell(k, layers=("buildings",), kind="demand").layers.get("buildings")
        if f is None:
            continue
        hs = f.attrs.get("height")
        ids = f.ids.tolist()
        rows, a, b = _photo_candidates(f, frame, x0, y0, size)
        keep = [j for j, i in enumerate(rows) if ids[i] not in seen]  # (a building in two cells: the first piece)
        if not keep:
            seen.update(ids)
            continue
        rows, a, b = rows[keep], a[keep], b[keep]
        # every kept outer ring grown out from its middle, all at once
        h = np.full(len(rows), 8.0) if hs is None else np.asarray(hs, np.float64)[rows]
        h = np.where(np.isfinite(h), h, 8.0)
        grow = (PHOTO_GROW_M + np.minimum(PHOTO_LEAN * h, PHOTO_LEAN_MAX_M)) / mpp
        ln = b - a
        idx = np.repeat(a - np.r_[0, np.cumsum(ln)[:-1]], ln) + np.arange(int(ln.sum()))
        xy = f.xy_in(frame)[idx]
        cx, cy = (xy[:, 0] - x0) / mpp, (y0 + size - xy[:, 1]) / mpp
        starts = np.r_[0, np.cumsum(ln)[:-1]]
        mx = np.repeat(np.add.reduceat(cx, starts) / ln, ln)
        my = np.repeat(np.add.reduceat(cy, starts) / ln, ln)
        dx, dy = cx - mx, cy - my
        g = np.repeat(grow, ln) / np.maximum(np.hypot(dx, dy), 1e-6)
        pts = np.stack([cx + dx * g, cy + dy * g], 1).tolist()
        for s0, m in zip(starts.tolist(), ln.tolist()):
            draw.polygon([tuple(q) for q in pts[s0:s0 + m]], fill=255)
        n += len(rows)
        seen.update(ids)
    if n == 0:
        return None
    return np.asarray(img) > 0


def _photo_candidates(f, frame, x0, y0, size):
    """Rows of f (a cell's buildings) whose outer ring's box meets the square [x0, x0 + size]² and that come first
    among the rows of their id, ascending, with their outer rings' coordinate ranges [a, b): photo_mask's buildings, its test made for all rows at once (one ring at a
    time it was ~19k Python steps a block, most of a block photo's roof mask; 2026-10-23)."""
    m = len(f.kind)
    none = (np.zeros(0, np.int64),) * 3
    if m == 0:
        return none
    p = f.part_off[:-1]
    has = (f.part_off[1:] > p)
    r = f.ring_off[np.minimum(p, len(f.ring_off) - 2)]
    has &= f.ring_off[np.minimum(p, len(f.ring_off) - 2) + 1] > r
    a = f.coord_off[np.minimum(r, len(f.coord_off) - 2)]
    b = f.coord_off[np.minimum(r, len(f.coord_off) - 2) + 1]
    has &= b > a
    _, first = np.unique(f.ids, return_index=True)
    head = np.zeros(m, bool)
    head[first] = True
    rows = np.flatnonzero(has & head)
    if len(rows) == 0:
        return none
    xy = f.xy_in(frame)
    a, b = a[rows], b[rows]
    order = np.argsort(a, kind="stable")       # (reduceat: ascending starts)
    e_lo = np.empty(len(rows)); e_hi = np.empty(len(rows)); n_lo = np.empty(len(rows)); n_hi = np.empty(len(rows))
    e_lo[order] = mapgeom._reduce(np.minimum, xy[:, 0], a[order], b[order])
    e_hi[order] = mapgeom._reduce(np.maximum, xy[:, 0], a[order], b[order])
    n_lo[order] = mapgeom._reduce(np.minimum, xy[:, 1], a[order], b[order])
    n_hi[order] = mapgeom._reduce(np.maximum, xy[:, 1], a[order], b[order])
    inside = ~((e_hi < x0) | (e_lo > x0 + size) | (n_hi < y0) | (n_lo > y0 + size))
    return rows[inside], a[inside], b[inside]


def fill_photo(rgb, mask, size, px):
    """rgb (px, px, 3) uint8 with the masked pixels replaced by a smooth fill from the unmasked ground
    around them: a push-pull pyramid, so a footprint of any size fills (one PHOTO_FILL_M blur left black
    holes in big blocks), softened by that blur (earthfast.fill_photo: the numpy / Pillow version's numbers,
    a third of its time; 2026-10-23)."""
    import fast
    fast.require(68, "the ground photo's roof fill")
    r = max(2.0, PHOTO_FILL_M / (size / px))
    return fast._ef.fill_photo(np.ascontiguousarray(rgb, np.uint8), np.ascontiguousarray(mask, bool), float(r))


BRICK_WALK_COVERAGE = 0.35  # dense old cores (row houses) of brick regions ...
BRICK_WALK_SHARE = 0.55     # ... pave this share of their streets' sidewalks with brick (hashed per street)


def brick_walks(walks, items, lat, lon):
    """Marks the sidewalk runs (roads.build_tile's walks) of a dense brick-region tile that are brick pavers
    (Fells Point's historic streets): every run of a street the same."""
    if not walks or brick_share(lat, lon) < 0.7 or coverage(items) < BRICK_WALK_COVERAGE:
        return
    for wk in walks:
        fid = wk.get("fid")
        if fid is not None and _hash01(str(fid), "brickwalk") < BRICK_WALK_SHARE:
            wk["brick"] = True


_cell_stats = {}                # map cell key -> (lon, lat, area m2) per footprint (district_coverage)
_cell_lock = None


def _cell_lock_get():
    global _cell_lock
    if _cell_lock is None:
        import threading
        _cell_lock = threading.Lock()
    return _cell_lock


def cell_footprints(key, layer):
    """(lon, lat, area m2) arrays of a cell's building footprints (outer rings), cached per cell (layer may
    be None once cached). Pure."""
    _cell_lock_get()
    with _cell_lock:
        got = _cell_stats.get(key)
    if got is not None:
        return got
    if layer is None or len(layer.ids) == 0:
        got = (np.zeros(0), np.zeros(0), np.zeros(0))
    else:
        f = layer
        first = f.ring_off[f.part_off[:-1]]
        a = f.coord_off[first]
        b = f.coord_off[first + 1]
        ll = f.lonlat
        lat0 = float(np.mean(ll[:, 1]))
        x = ll[:, 0] * 111320.0 * math.cos(math.radians(lat0))
        y = ll[:, 1] * 111320.0
        cross = np.r_[x[:-1] * y[1:] - x[1:] * y[:-1], 0.0]
        C = np.concatenate(([0.0], np.cumsum(cross)))
        CX, CY = np.concatenate(([0.0], np.cumsum(ll[:, 0]))), np.concatenate(([0.0], np.cumsum(ll[:, 1])))
        n = np.maximum(b - a, 1)
        last = np.maximum(b - 1, a)
        area = np.abs(C[last] - C[a] + x[last] * y[a] - x[a] * y[last]) / 2
        got = ((CX[b] - CX[a]) / n, (CY[b] - CY[a]) / n, area)
    with _cell_lock:
        if len(_cell_stats) > 64:
            _cell_stats.clear()
        _cell_stats[key] = got
    return got


def district_coverage(lat, lon, cells):
    """The share of the DISTRICT_M disc around (lat, lon) covered by building footprints; cells: [(key,
    buildings layer)] covering the disc."""
    seen = 0.0
    mx = 111320.0 * math.cos(math.radians(lat))
    for key, layer in cells:
        lo, la, ar = cell_footprints(key, layer)
        if len(ar):
            d2 = ((lo - lon) * mx) ** 2 + ((la - lat) * 111320.0) ** 2
            seen += float(ar[d2 <= DISTRICT_M * DISTRICT_M].sum())
    return seen / (math.pi * DISTRICT_M * DISTRICT_M)


def district_town(frame, tx, ty, md):
    """Is road tile (tx, ty) in a small town of the arid West (WEST_TOWN_COVERAGE)? The cells of the district
    are loaded like the tile's own (the far building tiles need them anyway); an unreadable one counts as
    empty."""
    import mapcells
    import roads
    import surface
    x0, y0, x1, y1 = roads.tile_bbox(tx, ty)
    lat, lon, _ = frame.enu_to_geodetic((x0 + x1) / 2, (y0 + y1) / 2, 0.0)
    lat, lon = float(lat), float(lon)
    if not bool(surface.dry_west(lat, lon)):
        return False
    dlat = DISTRICT_M / 111320.0
    dlon = DISTRICT_M / (111320.0 * math.cos(math.radians(lat)))
    cells = []
    for key in mapcells.cells_for_bbox(lat - dlat, lon - dlon, lat + dlat, lon + dlon):
        with _cell_lock_get():
            known = key in _cell_stats
        if known:
            cells.append((key, None))
            continue
        try:
            cells.append((key, md.cell(key, layers=("buildings",), kind="demand").layers.get("buildings")))
        except Exception:           # not cached / unreadable: counts as empty
            continue
    return district_coverage(lat, lon, cells) < WEST_TOWN_COVERAGE
