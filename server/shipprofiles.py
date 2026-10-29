"""shipprofiles: how a moored museum ship looks, as a small parametric profile (buildings._ship draws it on the
ship's own OSM footprint, which gives its length, beam, plan and which end is the bow).

A profile is measured along the ship from the bow (x 0) to the stern (x 1); heights are metres over the waterline
for a ship of length `L` (scaled by the footprint's own length); widths are shares of the beam. Every number is my
own, worked out from published overall dimensions (length, beam, freeboard, the stations of turrets, funnels and
masts read off general-arrangement proportions as stated in the ships' public specifications and the US Navy's
public-domain ship data: NavSource / Naval History and Heritage Command fact sheets, DANFS). No third-party model,
drawing or mesh is copied. Approximate by design: a ship that reads right from a car, not a scale model.

Known ships are found by name (the footprint's OSM name; Overture keeps no wikidata on buildings). Detection is not
by name (buildings.landmark_kind: a long pointed hull on water); the name only picks the look. Anything else gets
GENERIC_WARSHIP or GENERIC_LINER.

Blocks: (x0, x1, half width share of the beam, z0, z1, paint) — paint "upper" (the superstructure's paint),
"hull", "deck". z0 None: from the deck at that station.
Funnels: (x, length m, width m, top z m, paint, black top m).
Turrets: (x, diameter m, height over the deck m, superfiring lift m, guns to "bow"/"stern", barrel length m).
Masts: (x, top z m).
Freeboard: (bow, midships, stern) m: the deck edge over the waterline (the sheer: rising toward the bow).
Flight deck: True: the footprint is the flight deck (a carrier), its hull narrowing to `waterline` of it.
Island: (x0, x1, width share, top z, starboard): a carrier's island on its starboard edge.
"""
import re

# The Iowa class (BB-61 Iowa, San Pedro; BB-62 New Jersey, Camden; BB-63 Missouri, Pearl Harbor; BB-64 Wisconsin,
# Norfolk). Published: 270.4 m overall, 33.0 m beam; freeboard about 11 m at the bow falling to about 6 m aft;
# three 16-inch turrets (two forward, the second superfiring; one aft), each about 15 m across (the Iowa's mapped
# turret outlines 15 m), guns about 20 m long; the conning tower and forward fire-control tower amidships forward,
# two funnels, a tower foremast about 45 m over the water, a mainmast aft of the second funnel.
IOWA_CLASS = dict(
    L=270.4, freeboard=(11.0, 8.0, 6.0), waterline=0.94, rake=0.03,
    blocks=[
        (0.335, 0.665, 0.30, None, 11.5, "upper"),     # the 01 level deckhouse
        (0.355, 0.47, 0.22, 11.5, 15.0, "upper"),      # 02-03 levels forward
        (0.37, 0.45, 0.17, 15.0, 21.0, "upper"),       # the conning tower and the bridge
        (0.38, 0.44, 0.11, 21.0, 27.0, "upper"),       # the forward fire-control tower
        (0.395, 0.425, 0.07, 27.0, 31.0, "upper"),     # the main battery director
        (0.555, 0.665, 0.19, 11.5, 15.0, "upper"),     # the aft superstructure
        (0.60, 0.65, 0.10, 15.0, 19.5, "upper"),       # the aft director
    ],
    funnels=[(0.495, 11.0, 7.0, 29.0, "upper", 0.0), (0.575, 10.0, 6.5, 28.0, "upper", 0.0)],
    turrets=[(0.245, 15.0, 4.0, 0.0, "bow", 20.0), (0.32, 15.0, 4.0, 3.0, "bow", 20.0),
             (0.735, 15.0, 4.0, 0.0, "stern", 20.0)],
    masts=[(0.455, 46.0), (0.615, 40.0)],
    navy=True,
)

# RMS Queen Mary (Long Beach). Published: 310.7 m overall, 36.1 m beam, draught 11.9 m; a black hull to the main
# deck (about 16 m over the water amidships, rising to the bow), white upperworks of the promenade and sun decks,
# three funnels (the third a dummy) about 0.36, 0.48 and 0.60 of the length from the bow, red with black tops, their
# tops about 45 m over the water; the bridge front at about 0.23; foremast near the bow, mainmast aft.
QUEEN_MARY = dict(
    L=310.7, freeboard=(21.0, 16.0, 17.0), waterline=0.95, rake=0.035,
    blocks=[
        (0.215, 0.80, 0.47, None, 20.0, "upper"),      # the promenade deck (A deck) the beam's width
        (0.225, 0.775, 0.43, 20.0, 23.0, "upper"),     # the sun deck house
        (0.23, 0.70, 0.36, 23.0, 26.0, "upper"),       # the sports deck
        (0.225, 0.26, 0.47, 23.0, 28.5, "upper"),      # the bridge, its wings the full beam
        (0.235, 0.255, 0.12, 28.5, 31.0, "upper"),     # the wheelhouse top
    ],
    funnels=[(0.36, 11.0, 7.0, 45.0, "funnel", 4.5), (0.48, 11.0, 7.0, 45.0, "funnel", 4.5),
             (0.60, 10.5, 7.0, 43.0, "funnel", 4.5)],
    turrets=[],
    masts=[(0.13, 48.0), (0.86, 44.0)],
    navy=False,
)

# Essex-class carriers (CV-10 Yorktown, Charleston; CV-11 Intrepid, New York; CV-12 Hornet, Alameda; CV-16
# Lexington, Corpus Christi) and USS Midway (CV-41, San Diego). Published: the Essex class 266-270 m overall, hull
# beam 28 m under a flight deck about 45 m wide (the footprint) some 17 m over the water; Midway 296 m, 41 m hull
# beam under a 72 m angled flight deck about 18 m up. The island on the starboard edge a little forward of
# amidships, its top about 35 m over the water, a mast over it.
ESSEX_CLASS = dict(
    L=266.0, freeboard=(17.0, 17.0, 17.0), waterline=0.62, rake=0.03, flight_deck=True,
    blocks=[], funnels=[], turrets=[],
    island=(0.36, 0.50, 0.11, 33.0),
    masts=[(0.43, 45.0)],
    navy=True,
)
MIDWAY = dict(ESSEX_CLASS, L=296.0, freeboard=(18.0, 18.0, 18.0), waterline=0.57, island=(0.34, 0.46, 0.07, 36.0),
              masts=[(0.40, 50.0)])

# Ships not known by name. A warship: a deckhouse amidships, a bridge tower forward of it, one funnel, two masts.
GENERIC_WARSHIP = dict(
    L=None, freeboard=None, waterline=0.92, rake=0.03,
    blocks=[(0.33, 0.66, 0.30, None, 3.0, "upper"), (0.36, 0.48, 0.22, 3.0, 6.0, "upper"),
            (0.38, 0.45, 0.15, 6.0, 9.5, "upper")],
    funnels=[(0.53, 0.04, 0.025, 11.0, "upper", 0.0)],
    turrets=[], masts=[(0.43, None), (0.66, None)], navy=True, relative=True,
)
# A merchant ship or a liner: white upperworks amidships, decks narrowing upward, funnels on top.
GENERIC_LINER = dict(
    L=None, freeboard=None, waterline=0.94, rake=0.035,
    blocks=[(0.22, 0.78, 0.45, None, 3.0, "upper"), (0.25, 0.74, 0.40, 3.0, 6.0, "upper"),
            (0.27, 0.66, 0.33, 6.0, 9.0, "upper"), (0.26, 0.30, 0.45, 6.0, 10.5, "upper")],
    funnels=[(0.45, 0.04, 0.025, 15.0, "funnel", 0.12)],
    turrets=[], masts=[(0.14, None), (0.84, None)], navy=False, relative=True,
)

KNOWN = [
    (re.compile(r"\b(iowa|new jersey|missouri|wisconsin)\b|\bbb-6[1-4]\b"), IOWA_CLASS, ("uss", "battleship", "bb-6")),
    (re.compile(r"\bqueen mary\b"), QUEEN_MARY, None),
    (re.compile(r"\bmidway\b"), MIDWAY, ("uss", "carrier", "cv-41")),
    (re.compile(r"\b(yorktown|intrepid|hornet|lexington)\b"), ESSEX_CLASS, ("uss", "carrier", "cv-1")),
]


def profile(name, navy):
    """The profile for a ship by its name (else by navy alone). A known class needs a tell in the name too (an
    "Iowa" building is not the battleship: "uss", "battleship" or the hull number)."""
    n = str(name or "").lower()
    for rx, prof, tells in KNOWN:
        if rx.search(n) and (tells is None or any(t in n for t in tells)):
            return prof
    return GENERIC_WARSHIP if navy else GENERIC_LINER
