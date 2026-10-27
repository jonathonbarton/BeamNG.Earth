"""Bridge lookup (branch bridge-lookup): what a road bridge is and what it must clear, from public data,
for the deck profile (roads.deck_curve) and the structure template.

Sources, best first, every one optional (offline / no network / nothing found: the rules alone):
  NBI       FHWA National Bridge Inventory, the delimited state files (www.fhwa.dot.gov/bridge/nbi/
            ascii.cfm), read from <cache>/nbi/raw/<ST><yy>.txt and indexed only for the counties in
            NBI_COUNTIES (Multnomah OR, Clark WA for now). Structure type (043B), max span (048),
            length (049), deck width (052), min vertical underclearance (054B), navigation vertical
            clearance (039; 116, a lift span raised, is kept for the record only).
  OSM       Overpass: the bridge ways and man_made=bridge outlines at the deck (bridge:structure,
            bridge:movable, layer, seamark:bridge:clearance_height, maxheight, wikidata, name) and the
            waterways under it. Overture keeps only a feature's first source id, often a route
            relation (the Interstate Bridge's is r12140029), so the way is found by place, not id; a
            first source that is a way ("w123@4") is read from the OSM API directly.
  Wikidata  P2793 vertical clearance, P2787 longest span, P2043 length, P2048 height, P31 type.
Every answer is cached as JSON under <cache>/bridgeinfo/ (keyed by the deck's rounded midpoint); network
requests go through netfetch (its per-host policies). Nothing here runs per frame: once per bridge chain.

info(...) returns a dict: name, template (beam / truss / arch / suspension / cable-stayed / movable),
clearance_m (to the deck's underside, over water or a road, with clearance_src), span_m, length_m,
width_m, water (the waterway class under it, if any), sources (which answered)."""
import csv
import json
import math
import os
import threading
import urllib.parse

OVERPASS_SERVERS = ("https://overpass.kumi.systems/api/interpreter",      # first: overpass-api.de took
                    "https://overpass-api.de/api/interpreter")           # 40-55 s a query, 2026-10-18
OSM_API = "https://api.openstreetmap.org/api/0.6"
WIKIDATA = "https://www.wikidata.org/w/api.php"
# (state FIPS, county) indexed from the state files: Multnomah OR, Clark WA (Portland / Vancouver), and the
# landmark test bridges: Calcasieu LA, Orleans + Jefferson LA, Pinellas / Manatee / Hillsborough FL, Los
# Angeles, San Francisco, Alameda, San Diego CA, Allegheny PA. More: add the county here and its state's file to nbi/raw.
NBI_COUNTIES = {("41", "051"), ("53", "011"), ("22", "019"), ("22", "071"), ("22", "051"), ("12", "103"),
                ("12", "081"), ("12", "057"), ("06", "037"), ("06", "075"), ("06", "001"), ("06", "073"), ("42", "003")}
NBI_NEAR_M = 120.0          # an NBI record this close to the deck line can be it
DECK_DEPTH = 1.2            # m from the road surface to the deck's underside (roadmesh.DECK_DEPTH)
CLEAR_SPAN_MIN = 0.06       # a navigation clearance is believed between these shares of the main span
CLEAR_SPAN_MAX = 0.45
CLEAR_SPAN_MAX_MOVABLE = 0.30   # a movable bridge's 039 over this share is the span raised (Hawthorne 33.5 m / 74 m)
WATER_CLEAR = {"river": 7.0, "canal": 4.5, "stream": 2.5, "ditch": 1.5, "drain": 1.5}   # m, by rule
TEMPLATES = ("beam", "truss", "arch", "suspension", "cable-stayed", "movable")
NBI_TYPE = {"09": "truss", "10": "truss", "11": "arch", "12": "arch", "13": "suspension",
            "14": "cable-stayed", "15": "movable", "16": "movable", "17": "movable"}
NBI_TYPE_NAME = {"01": "slab", "02": "stringer/multi-beam or girder", "03": "girder and floorbeam",
                 "04": "tee beam", "05": "box beam (multiple)", "06": "box beam (single/spread)",
                 "07": "frame", "08": "orthotropic", "09": "truss - deck", "10": "truss - thru",
                 "11": "arch - deck", "12": "arch - thru", "13": "suspension", "14": "stayed girder",
                 "15": "movable - lift", "16": "movable - bascule", "17": "movable - swing",
                 "18": "tunnel", "19": "culvert", "20": "mixed types", "21": "segmental box girder",
                 "22": "channel beam"}

def _offline():
    return os.environ.get("BNE_BRIDGE_LOOKUP", "on").lower() == "off" or os.environ.get("BNE_OFFLINE") == "1"


def _overpass_on():
    """Overpass only from the tools (BNE_BRIDGE_OVERPASS=1, tools/bridge_profiles.py): netfetch's rule is no Overpass
    at runtime (its POLICIES deny overpass-api.de), so under earthd the lookup asks NBI, the OSM API and Wikidata."""
    return os.environ.get("BNE_BRIDGE_OVERPASS") == "1"


def _get(url, data=None):
    """JSON from url (data: form fields sent as the query string), or None on any failure. Through netfetch (the
    project's one door to the network: its User-Agent, per-host rate limits, retries and breakers, HTTPS verified);
    kind "background" under earthd: never in the way of the world being built (a refused one is asked again later)."""
    if _offline():
        return None
    import netfetch
    if data:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(data)
    try:
        kind = "demand" if os.environ.get("BNE_BRIDGE_LOOKUP_WAIT") == "1" else "background"   # (a tool waits)
        r = netfetch.default_fetcher().get(url, kind=kind, headers={"Accept": "application/json"})
        return json.loads(r.body.decode("utf-8"))
    except Exception:
        return None


def _num(v):
    """Metres from an OSM / NBI value ('12.5', '12.5 m', "41'", '13 ft'), or None."""
    if v is None:
        return None
    s = str(v).strip().lower()
    try:
        if "'" in s or "ft" in s:
            parts = s.replace("ft", "'").replace('"', "").split("'")
            ft = float(parts[0] or 0)
            inch = float(parts[1]) if len(parts) > 1 and parts[1].strip() else 0.0
            return ft * 0.3048 + inch * 0.0254
        return float(s.replace("m", "").strip())
    except ValueError:
        return None


# ---- NBI -----------------------------------------------------------------------------------------------

def _dms(v, deg_digits):
    """NBI LAT_016 (DDMMSSss) / LONG_017 (DDDMMSSss) to degrees, or None."""
    s = str(v).strip().strip("'")
    if not s.isdigit() or int(s) == 0:
        return None
    s = s.zfill(deg_digits + 6)
    d, m, sec = int(s[:deg_digits]), int(s[deg_digits:deg_digits + 2]), int(s[deg_digits + 2:]) / 100.0
    return d + m / 60.0 + sec / 3600.0


class NBI:
    """The NBI records of NBI_COUNTIES from <cache>/nbi/raw/*.txt (the delimited state files: untrusted
    data, read as text only), indexed once to <cache>/nbi/index.json."""

    def __init__(self, cache_dir, counties=NBI_COUNTIES):
        self.dir = os.path.join(cache_dir, "nbi")
        self.counties = set(counties)
        self.rows = None

    def _load(self):
        if self.rows is not None:
            return self.rows
        idx = os.path.join(self.dir, "index.json")
        tag = sorted("%s-%s" % c for c in self.counties)
        try:
            with open(idx, encoding="utf-8") as f:
                got = json.load(f)
            if got.get("counties") == tag:
                self.rows = got["rows"]
                return self.rows
        except (OSError, ValueError, KeyError):
            pass
        rows = []
        raw = os.path.join(self.dir, "raw")
        for name in sorted(os.listdir(raw)) if os.path.isdir(raw) else ():
            if not name.lower().endswith(".txt"):
                continue
            with open(os.path.join(raw, name), encoding="latin-1", newline="") as f:
                for r in csv.DictReader(f, quotechar="'"):
                    st = (r.get("STATE_CODE_001") or "").strip().zfill(2)
                    co = (r.get("COUNTY_CODE_003") or "").strip().zfill(3)
                    if (st, co) not in self.counties:
                        continue
                    lat, lon = _dms(r.get("LAT_016"), 2), _dms(r.get("LONG_017"), 3)
                    if lat is None or lon is None:
                        continue
                    rows.append({
                        "id": (r.get("STRUCTURE_NUMBER_008") or "").strip(), "state": st, "county": co,
                        "lat": lat, "lon": -lon,
                        "carried": (r.get("FACILITY_CARRIED_007") or "").strip(),
                        "crosses": (r.get("FEATURES_DESC_006A") or "").strip(),
                        "type": (r.get("STRUCTURE_TYPE_043B") or "").strip().zfill(2),
                        "span_m": _num(r.get("MAX_SPAN_LEN_MT_048")), "length_m": _num(r.get("STRUCTURE_LEN_MT_049")),
                        "width_m": _num(r.get("DECK_WIDTH_MT_052")),
                        "under_ref": (r.get("VERT_CLR_UND_REF_054A") or "").strip(),
                        "under_m": _num(r.get("VERT_CLR_UND_054B")),
                        "nav": (r.get("NAVIGATION_038") or "").strip(),
                        "nav_m": _num(r.get("NAV_VERT_CLR_MT_039")), "nav_lift_open_m": _num(r.get("MIN_NAV_CLR_MT_116")),
                        "year": (r.get("YEAR_BUILT_027") or "").strip()})
        self.rows = rows
        if rows:
            try:
                os.makedirs(self.dir, exist_ok=True)
                with open(idx, "w", encoding="utf-8") as f:
                    json.dump({"counties": tag, "rows": rows}, f)
            except OSError:
                pass
        return rows

    def match(self, lonlat, name=None, water_gap=False):
        """The record carrying this deck (lon/lat polyline): within NBI_NEAR_M of it, the facility
        carried sharing a word with the name breaking ties; the longest wins among close ones."""
        rows = self._load()
        if not rows or len(lonlat) == 0:
            return None
        ll = [(float(p[0]), float(p[1])) for p in lonlat]
        lat0 = sum(p[1] for p in ll) / len(ll)
        kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
        lo0, lo1 = min(p[0] for p in ll) - 0.003, max(p[0] for p in ll) + 0.003
        la0, la1 = min(p[1] for p in ll) - 0.003, max(p[1] for p in ll) + 0.003
        words = {w for w in (name or "").upper().replace("-", " ").split() if len(w) > 2}
        best = None
        for r in rows:
            if not (lo0 <= r["lon"] <= lo1 and la0 <= r["lat"] <= la1):
                continue
            d = min(_seg_dist((r["lon"] - a[0]) * kx, (r["lat"] - a[1]) * ky, 0.0, 0.0,
                              (b[0] - a[0]) * kx, (b[1] - a[1]) * ky) for a, b in zip(ll[:-1], ll[1:])) \
                if len(ll) > 1 else math.hypot((r["lon"] - ll[0][0]) * kx, (r["lat"] - ll[0][1]) * ky)
            if d > NBI_NEAR_M:
                continue
            if r["type"] in ("18", "19"):       # tunnels and culverts are not decks
                continue
            text = (r["carried"] + " " + r["crosses"]).upper().replace("-", " ")
            hit = sum(1 for w in words if w in text)
            nav = 1 if water_gap and (r["nav_m"] or 0) > 0 else 0       # over water: the river span's record
            score = (-nav, -hit, d - 0.02 * (r["length_m"] or 0.0))
            if best is None or score < best[0]:
                best = (score, r, d)
        if best is None:
            return None
        return dict(best[1], dist_m=round(best[2], 1))


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


# ---- OSM / Wikidata ------------------------------------------------------------------------------------

TILE_DEG = 0.02            # OSM is asked per tile of this many degrees (~2 x 1.5 km at 45 N): one query
                           # serves every bridge in it


def osm_tile(lat0, lon0):
    """The bridge ways, man_made=bridge outlines and waterways of the tile at (lat0, lon0) (its south-west
    corner): [{'tags', 'geom': [(lon, lat)...]}], or None (no answer from any server)."""
    s, w, n, e = lat0, lon0, lat0 + TILE_DEG, lon0 + TILE_DEG
    bb = f"{s:.4f},{w:.4f},{n:.4f},{e:.4f}"
    q = (f'[out:json][timeout:90];'
         f'(way({bb})[highway][bridge];way({bb})[man_made=bridge];);out tags geom;'
         f'way({bb})[waterway~"^(river|canal|stream|ditch|drain)$"];out tags geom({bb});')
    if not _overpass_on():
        return None
    for url in OVERPASS_SERVERS:
        got = _get(url, {"data": q})
        if got is not None and "elements" in got:
            break
    else:
        return None
    out = []
    for el in got.get("elements", []):
        g = [(float(p["lon"]), float(p["lat"])) for p in el.get("geometry", []) if p]
        out.append({"id": "w%d" % el.get("id", 0), "tags": el.get("tags", {}), "geom": g})
    return out


def _dist_m(lonlat_pt, geom):
    """Metres from a (lon, lat) point to a (lon, lat) polyline."""
    if not geom:
        return float("inf")
    lon0, lat0 = lonlat_pt
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    if len(geom) == 1:
        return math.hypot((geom[0][0] - lon0) * kx, (geom[0][1] - lat0) * ky)
    return min(_seg_dist(0.0, 0.0, (a[0] - lon0) * kx, (a[1] - lat0) * ky, (b[0] - lon0) * kx, (b[1] - lat0) * ky)
               for a, b in zip(geom[:-1], geom[1:]))


def _crosses(line_a, line_b):
    """Whether two (lon, lat) polylines cross."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    for p, q in zip(line_a[:-1], line_a[1:]):
        for r, s in zip(line_b[:-1], line_b[1:]):
            d1, d2, d3, d4 = ccw(p, q, r), ccw(p, q, s), ccw(r, s, p), ccw(r, s, q)
            if d1 * d2 < 0 and d3 * d4 < 0:
                return True
    return False


def osm_select(elements, lonlat):
    """{'bridges': tags of the bridge ways / outlines at this deck (lon/lat polyline), 'water': tags of
    the waterways crossing under it}. Pure."""
    ll = [(float(p[0]), float(p[1])) for p in lonlat]
    probes = [ll[len(ll) // 2], ll[len(ll) // 4], ll[(3 * len(ll)) // 4]]
    br, wa = [], []
    for el in elements:
        tg = el["tags"]
        if "waterway" in tg:
            if len(el["geom"]) >= 2 and _crosses(ll, el["geom"]):
                wa.append(dict(tg, _id=el["id"]))
            continue
        reach = 45.0 if tg.get("man_made") == "bridge" else 20.0
        if min(_dist_m(p, el["geom"]) for p in probes) <= reach:
            br.append(dict(tg, _id=el["id"]))
    return {"bridges": br, "water": wa}


def osm_way(record):
    """Tags of the OSM way of an Overture source record id ('w123@4'), or None."""
    if not record or not str(record).startswith("w"):
        return None
    wid = str(record)[1:].split("@")[0]
    got = _get(f"{OSM_API}/way/{wid}.json")
    try:
        return got["elements"][0].get("tags", {})
    except (TypeError, KeyError, IndexError):
        return None


def _claim_amount(claims, prop):
    """The first quantity of a Wikidata property in metres (ft converted), or None."""
    for c in claims.get(prop, []):
        try:
            v = c["mainsnak"]["datavalue"]["value"]
            amt = float(v["amount"])
            unit = v.get("unit", "")
            if unit.endswith("Q3710"):          # foot
                amt *= 0.3048
            return amt
        except (KeyError, TypeError, ValueError):
            continue
    return None


def wikidata(qid):
    """{'clearance_m', 'span_m', 'length_m', 'height_m', 'types': [english labels of P31]} or None."""
    if not qid or not str(qid).startswith("Q"):
        return None
    got = _get(f"{WIKIDATA}?action=wbgetentities&ids={qid}&props=claims|labels&languages=en&format=json")
    try:
        claims = got["entities"][qid]["claims"]
    except (TypeError, KeyError):
        return None
    out = {"clearance_m": _claim_amount(claims, "P2793"), "span_m": _claim_amount(claims, "P2787"),
           "length_m": _claim_amount(claims, "P2043"), "height_m": _claim_amount(claims, "P2048"), "types": []}
    def ids_of(prop):
        got = []
        for c in claims.get(prop, []):
            try:
                got.append(c["mainsnak"]["datavalue"]["value"]["id"])
            except (KeyError, TypeError):
                pass
        return got
    types, colours = ids_of("P31"), ids_of("P462")          # instance of, colour
    out["colours"] = []
    ids = (types + colours)[:40]
    if ids:
        lab = _get(f"{WIKIDATA}?action=wbgetentities&ids={'|'.join(ids)}&props=labels&languages=en&format=json")
        try:
            name = {q: e["labels"]["en"]["value"] for q, e in lab["entities"].items() if "en" in e.get("labels", {})}
            out["types"] = [name[q] for q in types if q in name]
            out["colours"] = [name[q] for q in colours if q in name]
        except (TypeError, KeyError, AttributeError):
            pass
    return out


def template_of(words):
    """A structure template from free words (OSM bridge:structure, Wikidata type labels). Pure."""
    t = " ".join(w for w in words if w).lower()
    for key, tpl in (("lift", "movable"), ("bascule", "movable"), ("swing", "movable"), ("movable", "movable"),
                     ("cable-stayed", "cable-stayed"), ("cable stayed", "cable-stayed"), ("suspension", "suspension"),
                     ("arch", "arch"), ("truss", "truss"), ("lenticular", "truss"), ("pauli", "truss")):
        if key in t:
            return tpl
    return None


# ---- the answer ------------------------------------------------------------------------------------------

class Lookup:
    """info() per bridge chain, cached. cache_dir: the server's cache root."""

    def __init__(self, cache_dir, nbi=True):
        self.dir = os.path.join(cache_dir, "bridgeinfo")
        self.nbi = NBI(cache_dir) if nbi else None
        self.mem = {}
        self.tiles = {}                         # tile key -> elements, or None (failed this session: not asked again)
        self.qids = {}
        self.lock = threading.Lock()
        self.fetch_lock = threading.Lock()      # one network fetch at a time (polite, and no duplicate tiles)
        # network fetches off the caller's thread unless BNE_BRIDGE_LOOKUP_WAIT=1 (the offline tools): a road tile
        # is never held for an Overpass answer (40-55 s a tile, 2026-10-19: tiles waited 5 min at the Interstate
        # Bridge); the answer is cached for the next build of that bridge
        self.wait = os.environ.get("BNE_BRIDGE_LOOKUP_WAIT") == "1"
        self.queue, self.queued = [], set()
        self.qcond = threading.Condition()
        self.worker = None

    def info(self, lonlat, name=None, record=None, length_m=None, water_gap=False):
        """What is known of the bridge whose deck is the lon/lat polyline (k, 2). Never raises."""
        try:
            return self._info(lonlat, name, record, length_m, water_gap)
        except Exception as e:                  # a lookup never costs a deck
            return dict(rules(None, None, None), error=f"{type(e).__name__}: {e}")

    def _json(self, sub, key, fetch):
        """A cached JSON answer (memory, then <cache>/bridgeinfo/<sub>/<key>.json), else fetch() saved there
        (a None answer is remembered for this session only)."""
        store = self.tiles if sub == "osm" else self.qids
        with self.lock:
            if key in store:
                return store[key]
        path = os.path.join(self.dir, sub, key + ".json")
        try:
            with open(path, encoding="utf-8") as f:
                got = json.load(f)
        except (OSError, ValueError):
            if not self.wait:
                self._later(sub, key, fetch, path)
                return None
            with self.fetch_lock:
                with self.lock:
                    if key in store:
                        return store[key]
                got = fetch()
                if got is not None:
                    try:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(got, f)
                    except OSError:
                        pass
        with self.lock:
            store[key] = got
        return got

    def _later(self, sub, key, fetch, path):
        """Queue fetch() for the background thread (once per key); its answer goes to disk and memory."""
        with self.qcond:
            if (sub, key) in self.queued:
                return
            self.queued.add((sub, key))
            self.queue.append((sub, key, fetch, path))
            if self.worker is None:
                self.worker = threading.Thread(target=self._run, name="bridgeinfo", daemon=True)
                self.worker.start()
            self.qcond.notify()

    def _run(self):
        while True:
            with self.qcond:
                while not self.queue:
                    self.qcond.wait()
                sub, key, fetch, path = self.queue.pop(0)
            try:
                got = fetch()
            except Exception:
                got = None
            if got is not None:
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(got, f)
                except OSError:
                    pass
                store = self.tiles if sub == "osm" else self.qids
                with self.lock:
                    store[key] = got
                    self.mem.clear()            # answers built without it are asked again

    def _osm(self, lonlat):
        elements, tried = [], set()
        ll = [lonlat[0], lonlat[len(lonlat) // 2], lonlat[-1]]
        ok = False
        for lon, lat in ((float(p[0]), float(p[1])) for p in ll):
            la0, lo0 = math.floor(lat / TILE_DEG) * TILE_DEG, math.floor(lon / TILE_DEG) * TILE_DEG
            key = "%.2f_%.2f" % (la0, lo0)
            if key in tried:
                continue
            tried.add(key)
            got = self._json("osm", key, lambda: osm_tile(la0, lo0))
            if got is not None:
                ok = True
                elements += got
        return osm_select(elements, lonlat) if ok else None

    def _info(self, lonlat, name, record, length_m, water_gap=False):
        mid = lonlat[len(lonlat) // 2]
        key = ("%.5f_%.5f" % (float(mid[1]), float(mid[0])), water_gap)
        with self.lock:
            if key in self.mem:
                return self.mem[key]
        near = self._osm(lonlat)
        qids = []
        for tg in (near["bridges"] if near else []):
            if tg.get("wikidata") and tg["wikidata"] not in qids:
                qids.append(tg["wikidata"])
        wd = self._json("wikidata", qids[0], lambda: wikidata(qids[0])) if qids else None
        net = {"osm": near, "way": None, "wikidata": wd, "qid": qids[0] if qids else None}
        nb = self.nbi.match(lonlat, name, water_gap) if self.nbi is not None else None
        out = rules(net, nb, name)
        with self.lock:
            self.mem[key] = out
        return out


def rules(net, nb, name):
    """The answer from what was found (net: the cached OSM / Wikidata answer; nb: the NBI record), the
    rules filling the gaps. Pure."""
    net = net or {}
    osm = net.get("osm") or {}
    tags = [t for t in ([net.get("way")] if net.get("way") else []) + list(osm.get("bridges", []))]
    wd = net.get("wikidata") or {}
    src = []
    out = {"name": name, "template": "beam", "template_src": "default", "clearance_m": None,
           "clearance_src": None, "span_m": None, "length_m": None, "width_m": None, "water": None,
           "nbi": None, "qid": net.get("qid")}
    for t in osm.get("water", []):
        w = t.get("waterway")
        if w in WATER_CLEAR and (out["water"] is None or WATER_CLEAR[w] > WATER_CLEAR[out["water"]]):
            out["water"] = w
    # template: NBI type, then OSM bridge:movable / bridge:structure, then Wikidata types
    if nb and nb.get("type") in NBI_TYPE:
        out["template"], out["template_src"] = NBI_TYPE[nb["type"]], "nbi"
    else:
        mv = next((t.get("bridge:movable") for t in tags if t.get("bridge:movable") and t.get("bridge:movable") != "no"), None)
        st = template_of([t.get("bridge:structure") for t in tags] + [t.get("bridge") for t in tags])
        wt = template_of(wd.get("types", []))
        if mv:
            out["template"], out["template_src"] = "movable", "osm"
        elif st:
            out["template"], out["template_src"] = st, "osm"
        elif wt:
            out["template"], out["template_src"] = wt, "wikidata"
    if nb:
        src.append("nbi")
        out["nbi"] = {k: nb.get(k) for k in ("id", "carried", "crosses", "type", "span_m", "length_m", "width_m",
                                               "under_ref", "under_m", "nav_m", "nav_lift_open_m", "year", "dist_m")}
        out["nbi"]["type_name"] = NBI_TYPE_NAME.get(nb.get("type"), "?")
        out["span_m"], out["length_m"], out["width_m"] = nb.get("span_m"), nb.get("length_m"), nb.get("width_m")
    out["span_m"] = out["span_m"] or wd.get("span_m")
    out["length_m"] = out["length_m"] or wd.get("length_m")
    # clearance to the deck's underside: navigation (closed, for a lift) > seamark > wikidata > NBI under
    # a road / rail > the waterway rule
    cands = []
    if nb:
        # item 039 is the clearance a boat has with the bridge down (for a lift bridge item 116 is the
        # lift span raised: not what a car drives on)
        v = nb.get("nav_m")
        if v and v > 0.0:
            cands.append((v, "nbi:039 nav"))
    for t in tags:
        v = _num(t.get("seamark:bridge:clearance_height"))
        if v and v > 0.0:
            cands.append((v, "osm:seamark"))
    if wd.get("clearance_m") and wd["clearance_m"] > 0.0:
        cands.append((wd["clearance_m"], "wikidata:P2793"))
    if out["water"]:
        cands.append((WATER_CLEAR[out["water"]], "rule:" + out["water"]))
    # sanity: a navigation clearance under CLEAR_SPAN_MIN or over CLEAR_SPAN_MAX of the main span is not the
    # driving deck's (2026-10-19: the Interstate Bridge's 039 is 4.3 m under a 162 m span, the Steel Bridge's 48.8 m
    # is both its decks raised over an 88 m span); 80 m at most, 3 m at least. Rejected ones are listed.
    span = out["span_m"] or 0.0
    rejected = []
    for v, s in list(cands):
        bad = not 3.0 <= v <= 80.0
        if not bad and span >= 40.0 and not s.startswith("rule:"):
            hi = CLEAR_SPAN_MAX_MOVABLE if (nb or {}).get("type") in ("15", "16", "17") else CLEAR_SPAN_MAX
            bad = not min(CLEAR_SPAN_MIN * span, 15.0) <= v <= hi * span   # (a 1.3 km span's 67 m is right)
        if bad:
            cands.remove((v, s))
            rejected.append([round(v, 2), s])
    out["clearance_rejected"] = rejected
    if cands:                                   # over water
        out["clearance_m"], out["clearance_src"] = cands[0]
    out["road_clear_m"] = None                  # over a road / railway (NBI 054B), kept apart: a river bridge's
    if nb and nb.get("under_m") and 3.0 < nb["under_m"] < 30.0 and nb.get("under_ref") in ("H", "R"):
        out["road_clear_m"] = nb["under_m"]     # record also names the street on its bank
    # what the structure drawing needs (bridgeart): the truss's shape (a lens when the sources call it both a
    # truss and an arch: the Smithfield Street Bridge is NBI truss-thru, OSM arch, Wikidata truss + arch), a deck
    # arch (NBI 11) or a through arch, a lift span (NBI 15 / bridge:movable=lift), the colour if tagged
    words = " ".join([str(t.get("bridge:structure", "")) for t in tags] + list(wd.get("types", []))).lower()
    nbt = (nb or {}).get("type")
    out["truss_shape"] = "lens" if (out["template"] == "truss" and "arch" in words) or "lenticular" in words else "parker"
    out["arch_deck"] = nbt == "11" or "deck arch" in words
    out["lift"] = nbt == "15" or any(t.get("bridge:movable") == "lift" for t in tags)
    out["colours"] = list(wd.get("colours") or [])
    out["colour"] = next((t.get("colour") or t.get("bridge:colour") for t in tags if t.get("colour") or t.get("bridge:colour")), None)
    for name_, have in (("osm", bool(tags or osm.get("water"))), ("wikidata", bool(wd))):
        if have:
            src.append(name_)
    out["sources"] = src
    return out


def nbi_source_note():
    """Where the NBI data comes from (for the credits / report)."""
    return "FHWA National Bridge Inventory (public domain), delimited state files, counties " + \
        ", ".join("%s-%s" % c for c in sorted(NBI_COUNTIES))
