"""Offline tests of the FIRMS filtering, de-duplication and cell aggregation (no network)."""
import datetime as dt

import numpy as np
import pandas as pd

import download_firms as fb
import grid


def rows(points, daynight="D", conf="n"):
    return pd.DataFrame({"latitude": [p[0] for p in points], "longitude": [p[1] for p in points],
                         "daynight": daynight, "confidence": conf, "acq_date": "2019-09-01"})


def test_good_daytime_filters():
    df = pd.DataFrame({"latitude": [0.0] * 6, "longitude": [100.0] * 6,
                       "daynight": ["D", "N", "D", "D", "D", "D"],
                       "confidence": ["n", "n", "l", "h", "n", "n"],
                       "type": [0, 0, 0, 0, 2, 0]})
    out = fb.good_daytime(df)
    assert list(out.index) == [0, 3, 5]          # night, low confidence and the static source (type 2) dropped


def test_good_daytime_without_type_column():
    df = rows([(0.0, 100.0), (1.0, 101.0)])
    assert len(fb.good_daytime(df)) == 2


def test_dedupe_priority_and_radius():
    near = (0.0, 100.0)
    # the same fire seen 300 m away by the lower-priority sensors, plus a fire 5 km away that only snpp sees
    n21 = rows([near])
    n20 = rows([(0.0, 100.0027)])                # ~300 m east
    snpp = rows([(0.0, 100.0027), (0.0, 100.045)])
    out = fb.dedupe([("n21", n21), ("n20", n20), ("snpp", snpp)])
    assert sorted(out["sensor"]) == ["n21", "snpp"]
    assert len(out) == 2


def test_same_sensor_pixels_do_not_drop_each_other():
    n21 = rows([(0.0, 100.0), (0.0, 100.0034), (0.0, 100.0068)])      # a fire front, 375 m apart
    out = fb.dedupe([("n21", n21), ("n20", rows([]))])
    assert len(out) == 3


def test_aggregate_day_counts():
    d = dt.date(2019, 9, 1)
    n21 = fb.good_daytime(rows([(0.5, 101.5)]))
    n20 = fb.good_daytime(rows([(0.5, 101.5), (0.6, 101.55)]))          # one repeat of the n21 fire, one new fire in the same cell
    snpp = fb.good_daytime(rows([(0.5, 101.5)]))
    table = fb.aggregate_day(d, [("n21", n21), ("n20", n20), ("snpp", snpp)])
    assert len(table) == 1
    r = table.iloc[0]
    assert (r.n_unique, r.n_n21, r.n_n20, r.n_snpp) == (2, 1, 1, 0)
    assert (r.raw_n21, r.raw_n20, r.raw_snpp) == (1, 2, 1)
    assert r.lat == 0.5 and r.lon == 101.5


def test_aggregate_day_outside_box_is_ignored():
    d = dt.date(2019, 9, 1)
    out = fb.aggregate_day(d, [("n21", fb.good_daytime(rows([(20.0, 101.5)])))])
    assert out.empty or out["n_unique"].sum() == 0


def test_source_segments_prefer_sp_then_nrt():
    avail = {"A_SP": (dt.date(2019, 1, 1), dt.date(2019, 1, 3)), "A_NRT": (dt.date(2019, 1, 3), dt.date(2019, 1, 10))}
    days = [dt.date(2019, 1, 1) + dt.timedelta(days=i) for i in range(5)]
    segs = fb.source_segments(avail, "A_SP", "A_NRT", days)
    assert [(s, a.day, b.day) for s, a, b in segs] == [("A_SP", 1, 3), ("A_NRT", 4, 5)]


def test_grid_cells():
    r, c = grid.cell_index([8.0, -11.0, 0.5, 9.0], [94.0, 120.0, 101.5, 101.5])
    assert list(r) == [0, 76, 30, -1] and list(c) == [0, 104, 30, -1]
    assert grid.lat_axis()[0] == 8.0 and grid.lat_axis()[-1] == -11.0 and len(grid.lon_axis()) == 105
    assert list(np.unique(grid.noon_utc_hour(grid.lon_axis()))) == [4, 5, 6]
