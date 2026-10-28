"""Build server/bridges_nl.npz from Rijkswaterstaat's Fairway Information System (FIS, vaarweginformatie.nl; open
data, CC0 1.0, data.overheid.nl "FIS_VNDS"): the Netherlands' counterpart of the NBI's navigation clearances.

    python tools/build_bridge_nl.py

Reads the fis_vnds map service's bridge (brug) and opening (opening: a bridge's passages, their clearance under the
closed bridge and width) layers. Per bridge with a measured clearance: lat, lon, a type code as bridgetypes' (NBI item
43B: 15 lift, 16 bascule, 17 swing for a movable opening, else 2, a plain deck), the widest opening as its main span,
its length and width (0 where not given) and the highest closed clearance (m over the fairway's reference level) as
nav. Same columns as bridges_us.npz, sorted by latitude."""
import json
import os
import sys
import urllib.parse
import urllib.request

import numpy as np

BASE = "https://geo.rijkswaterstaat.nl/arcgis/rest/services/GDR/fis_vnds/MapServer"
BRIDGE, OPENING = 3, 15
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server", "bridges_nl.npz")
UA = "BeamNG.Earth (bridge inventory build tool; https://github.com/TecnaGamer)"
# FIS opening types: VST fixed; HEF lift; BC / DBC / RBC / KLP / OPH / ROL bascule-like (ophaal-, klap-, basculebrug,
# rolbrug); DR / DDR / PDR swing (draaibrug); others (DOP, OKW, PON: openings in other works, pontoons) no type
MOVABLE = {"HEF": 15, "BC": 16, "DBC": 16, "RBC": 16, "KLP": 16, "OPH": 16, "ROL": 16, "DR": 17, "DDR": 17, "PDR": 17}
PLAIN = 2


def _all(layer, fields):
    out, off = [], 0
    while True:
        q = urllib.parse.urlencode({"where": "1=1", "outFields": fields, "outSR": 4326, "f": "json",
                                    "resultOffset": off, "resultRecordCount": 2000, "orderByFields": "objectid"})
        req = urllib.request.Request(f"{BASE}/{layer}/query?{q}", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
        if "error" in d:
            raise RuntimeError(d["error"])
        feats = d.get("features") or []
        out += feats
        if not feats or not d.get("exceededTransferLimit"):
            return out
        off += len(feats)


def main():
    bridges = _all(BRIDGE, "id,name,length,width,condition")
    openings = _all(OPENING, "parentid,parentgeotype,type,width,clearanceheightclosed,condition")
    per = {}
    for f in openings:
        a = f["attributes"]
        if a.get("parentgeotype") != "bridge" or a.get("condition") not in (None, "CONSTRUCTED"):
            continue
        p = per.setdefault(a["parentid"], {"nav": 0.0, "span": 0.0, "type": PLAIN})
        c = a.get("clearanceheightclosed")
        if c is not None and c > p["nav"]:
            p["nav"] = float(c)
        w = a.get("width")
        if w is not None and w > p["span"]:
            p["span"] = float(w)
        if a.get("type") in MOVABLE:
            p["type"] = MOVABLE[a["type"]]
    rows = []
    for f in bridges:
        a, g = f["attributes"], f.get("geometry") or {}
        p = per.get(a["id"])
        if p is None or p["nav"] <= 0 or "x" not in g or a.get("condition") not in (None, "CONSTRUCTED"):
            continue
        rows.append((g["y"], g["x"], p["type"], p["span"], a.get("length") or 0.0, a.get("width") or 0.0, p["nav"]))
    a = np.array(rows, np.float64)
    o = np.argsort(a[:, 0])
    a = a[o]
    np.savez_compressed(OUT, lat=a[:, 0].astype(np.float32), lon=a[:, 1].astype(np.float32),
                        type=a[:, 2].astype(np.uint8), span=a[:, 3].astype(np.float32),
                        length=a[:, 4].astype(np.float32), width=a[:, 5].astype(np.float32),
                        nav=a[:, 6].astype(np.float32))
    print(f"{len(a)} bridges ({int((a[:, 2] != PLAIN).sum())} movable) of {len(bridges)} -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
