"""Which grid squares belong to the study, and to which place.

The study places are Peninsular Malaysia, Sabah, Sarawak, Brunei, Singapore, Sumatra, Kalimantan and Java. A square belongs to the
place whose boundary holds its centre (boundaries: data/study_boundaries.geojson, from Natural Earth). A coastal square whose
centre is just off shore goes to the nearest boundary if that is within NEAREST_DEG; squares nearest to anything outside the
study (Thailand, the Philippines, Sulawesi and the rest of Indonesia and so on) are left out.

Land: ERA5 land fraction of at least LAND_MIN. Singapore is a single, mostly coastal square (land fraction 0.33), so for Singapore
alone LAND_SINGAPORE is used.

For the regression the very small places share the term of their neighbour (MODEL_GROUP): Brunei with Sarawak and Singapore with
Peninsular Malaysia. Each has too few squares and fires to estimate its own term, and a place with no fires at all would push its
coefficient to minus infinity.
"""
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

LAND_MIN = 0.5
LAND_SINGAPORE = 0.1
NEAREST_DEG = 0.2
MODEL_GROUP = {"Brunei": "Sarawak", "Singapore": "Peninsular Malaysia"}
BOUNDARIES = Path(__file__).resolve().parents[1] / "data" / "study_boundaries.geojson"


def load(path=BOUNDARIES):
    feats = json.load(open(path))["features"]
    return [shape(f["geometry"]) for f in feats], [f["properties"]["group"] for f in feats]


def label_cells(lat, lon, land_fraction, boundaries=None):
    """Array (same shape as `land_fraction`) of the study place of each square, "" where the square is not in the study."""
    geoms, groups = boundaries or load()
    tree = STRtree(geoms)
    out = np.full(land_fraction.shape, "", dtype=object)
    for i in range(land_fraction.shape[0]):
        for j in range(land_fraction.shape[1]):
            lf = land_fraction[i, j]
            if lf < LAND_SINGAPORE:
                continue
            pt = Point(lon[j], lat[i])
            hit = [k for k in tree.query(pt, predicate="within")]
            if hit:
                k = hit[0]
            else:
                k = int(tree.nearest(pt))
                if geoms[k].distance(pt) > NEAREST_DEG:
                    continue
            g = groups[k]
            if g == "Singapore" or (g and lf >= LAND_MIN):
                out[i, j] = g
    return out
