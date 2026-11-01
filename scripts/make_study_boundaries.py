#!/usr/bin/env python3
"""Build data/study_boundaries.geojson from the Natural Earth 1:10m admin-1 states and provinces (public domain).

Each feature gets a `group`: the study place it belongs to, or "" for places that are outside the study but kept so that border
squares are not pulled into the wrong place. The study places are those asked for: Peninsular Malaysia, Sabah, Sarawak, Brunei,
Singapore, Sumatra, Kalimantan and Java. Geometry is simplified to about 1 km.

Usage: python scripts/make_study_boundaries.py ne_10m_admin_1_states_provinces.geojson data/study_boundaries.geojson
Source: https://github.com/nvkelso/natural-earth-vector (geojson/ne_10m_admin_1_states_provinces.geojson)
"""
import json
import sys

from shapely.geometry import mapping, shape

SUMATRA = {"Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Jambi", "Sumatera Selatan", "Bengkulu", "Lampung",
           "Bangka-Belitung", "Kepulauan Riau"}
KALIMANTAN = {"Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan", "Kalimantan Timur", "Kalimantan Utara"}
JAVA = {"Banten", "Jakarta Raya", "Jawa Barat", "Jawa Tengah", "Jawa Timur", "Yogyakarta"}
SABAH = {"Sabah", "Labuan"}
KEEP_OUTSIDE = {"Indonesia", "Thailand", "Myanmar", "Philippines", "Vietnam", "Cambodia", "Timor-Leste", "Australia"}


def group_of(admin, name):
    if admin == "Indonesia":
        return "Sumatra" if name in SUMATRA else "Kalimantan" if name in KALIMANTAN else "Java" if name in JAVA else ""
    if admin == "Malaysia":
        return "Sabah" if name in SABAH else "Sarawak" if name == "Sarawak" else "Peninsular Malaysia"
    if admin == "Brunei":
        return "Brunei"
    if admin == "Singapore":
        return "Singapore"
    return ""


def main(src, dst):
    out = []
    for f in json.load(open(src))["features"]:
        p = f["properties"]
        admin = p["admin"]
        if admin not in KEEP_OUTSIDE | {"Malaysia", "Brunei", "Singapore"}:
            continue
        geom = shape(f["geometry"])
        if not (geom.bounds[2] >= 93 and geom.bounds[0] <= 121 and geom.bounds[3] >= -12 and geom.bounds[1] <= 9):
            continue                                  # nowhere near the study box
        geom = geom.simplify(0.01, preserve_topology=True)
        out.append({"type": "Feature", "properties": {"admin": admin, "name": p["name"], "group": group_of(admin, p["name"])},
                    "geometry": json.loads(json.dumps(mapping(geom), separators=(",", ":")))})
    json.dump({"type": "FeatureCollection", "features": out}, open(dst, "w"), separators=(",", ":"))
    groups = sorted({f["properties"]["group"] for f in out if f["properties"]["group"]})
    print(f"{len(out)} features, groups {groups}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
