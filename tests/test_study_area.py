import numpy as np
from shapely.geometry import box

import study_area as sa

# two study squares and one outside square, as a stand-in for the real boundary file
GEOMS = [box(100, 0, 102, 2), box(102, 0, 104, 2), box(104, 0, 106, 2)]
GROUPS = ["Sumatra", "Java", ""]
LAT = np.array([1.0])
LON = np.array([101.0, 103.0, 105.0, 102.0, 99.9, 100.1])


def label(lf, lon=LON):
    return sa.label_cells(LAT, lon, np.array([lf]), (GEOMS, GROUPS))[0]


def test_inside_a_boundary_takes_its_group_and_outside_is_left_out():
    out = label(np.ones(6))
    assert list(out[:3]) == ["Sumatra", "Java", ""]


def test_a_square_just_off_shore_goes_to_the_nearest_place_within_the_limit():
    out = label(np.ones(6))
    assert out[4] == "Sumatra"           # 99.9 is 0.1 from the Sumatra box and not inside it
    assert label(np.ones(1), np.array([98.0]))[0] == ""      # 2 degrees away: too far


def test_mostly_sea_squares_are_dropped_except_singapore():
    out = label(np.full(6, 0.3))
    assert all(v == "" for v in out)
    geoms, groups = [box(0, 0, 2, 2)], ["Singapore"]
    sg = sa.label_cells(np.array([1.0]), np.array([1.0]), np.array([[0.3]]), (geoms, groups))
    assert sg[0, 0] == "Singapore"
    sg2 = sa.label_cells(np.array([1.0]), np.array([1.0]), np.array([[0.05]]), (geoms, groups))
    assert sg2[0, 0] == ""


def test_real_file_places_cells_correctly():
    lat, lon = np.array([0.5, -2.0, -7.5, 5.5, 3.0, 1.25, 4.5]), np.array([101.5, 114.0, 110.0, 117.0, 113.0, 103.75, 114.75])
    lf = np.ones((len(lat), len(lon)))
    out = sa.label_cells(lat, lon, lf)
    pick = lambda la, lo: out[list(lat).index(la), list(lon).index(lo)]
    assert pick(0.5, 101.5) == "Sumatra"           # Riau
    assert pick(-2.0, 114.0) == "Kalimantan"       # South Kalimantan
    assert pick(-7.5, 110.0) == "Java"
    assert pick(5.5, 117.0) == "Sabah"
    assert pick(3.0, 113.0) == "Sarawak"
    assert pick(4.5, 114.75) == "Brunei"
