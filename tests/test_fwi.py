"""Reproduce the 49-day test run of Table 7 of NOR-X-424 (Wang, Anderson and Suddaby 2015)."""
import re
from pathlib import Path

import numpy as np
import pytest

import fwi

DATA = Path(__file__).parent / "data" / "table7_nor_x_424.txt"


def table7():
    rows = []
    for line in DATA.read_text().splitlines():
        t = line.split()
        if len(t) == 12 and all(re.fullmatch(r"-?\d+\.?\d*", x) for x in t):
            rows.append([float(x) for x in t])
    return rows


def test_table7_has_49_days():
    assert len(table7()) == 49


def test_reproduces_table7():
    state = (fwi.FFMC_START, fwi.DMC_START, fwi.DC_START)
    for r in table7():
        month, day, T, H, W, Ro = int(r[0]), int(r[1]), r[2], r[3], r[4], r[5]
        state, out = fwi.step(state, T, H, W, Ro, le=fwi.EL_MONTHLY[month - 1], lf=fwi.LF_MONTHLY[month - 1])
        f = float(fwi.fwi(out["isi"], out["bui"]))
        got = [float(out[k]) for k in ("ffmc", "dmc", "dc", "isi", "bui")] + [f]
        for name, g, e in zip(("ffmc", "dmc", "dc", "isi", "bui", "fwi"), got, r[6:]):
            assert round(g, 1) == pytest.approx(e, abs=0.051), f"{month}/{day} {name}: got {g:.3f}, report {e}"


def test_grid_matches_scalar():
    """One vectorised call over a grid gives the same numbers as cell-by-cell calls."""
    rng = np.random.default_rng(1)
    n = 50
    T, H, W = rng.uniform(15, 35, n), rng.uniform(30, 100, n), rng.uniform(0, 25, n)
    Ro = np.where(rng.random(n) < 0.4, rng.uniform(0, 30, n), 0.0)
    state = (rng.uniform(20, 95, n), rng.uniform(1, 120, n), rng.uniform(5, 700, n))
    _, grid = fwi.step(state, T, H, W, Ro)
    for k in range(n):
        _, one = fwi.step((state[0][k], state[1][k], state[2][k]), T[k], H[k], W[k], Ro[k])
        for name in grid:
            assert float(grid[name][k]) == pytest.approx(float(one[name]), rel=1e-12)


def test_equatorial_constants_dry_day():
    """A dry day in the equatorial setting raises DC by 0.5 * (0.36 * (T + 2.8) + 1.39)."""
    d = fwi.dc(30.0, 0.0, 100.0, fwi.LF_EQUATORIAL)
    assert float(d) == pytest.approx(100.0 + 0.5 * (0.36 * 32.8 + 1.39))


def test_rain_lowers_codes():
    f_dry = float(fwi.ffmc(30, 60, 10, 0.0, 90.0))
    f_wet = float(fwi.ffmc(30, 60, 10, 20.0, 90.0))
    assert f_wet < f_dry
    assert float(fwi.dc(30, 50.0, 400.0, 1.39)) < 400.0
    assert float(fwi.dmc(30, 60, 50.0, 80.0, 9.0)) < 80.0
