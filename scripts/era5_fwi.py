#!/usr/bin/env python3
"""Daily FFMC, DMC, DC, ISI and BUI over the study box from hourly ARCO-ERA5.

ARCO-ERA5 is Google's public analysis-ready Zarr copy of ERA5 (0.25 deg, hourly); no account is needed.

The FWI System wants the weather at noon local standard time and the rain of the 24 h before it. Here "noon" is solar noon of
each grid column rounded to a whole UTC hour (the hours are the same for every cell of a column):
    T, RH, wind  from the 2 m temperature, 2 m dewpoint and 10 m wind of that hour (RH by the Magnus formula)
    rain         the sum of the 24 hourly ERA5 accumulations that end at that hour, in mm

The codes are carried day by day from `--spinup-days` before the first requested day (FFMC, DMC and DC start at the values of
the NOR-X-424 test run and forget them within days, weeks and months). Only the requested days are written.

Output: one NetCDF file per run with dims (time, latitude, longitude), float32, plus the land-sea mask.
"""
import argparse
import datetime as dt
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

import fwi
import grid

STORE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
INSTANT = {"t2m": "2m_temperature", "d2m": "2m_dewpoint_temperature",
           "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind"}
PRECIP = "total_precipitation"
WORKERS = 8
BATCH_DAYS = 16                  # days fetched in parallel at a time


def box(da):
    return da.sel(latitude=slice(grid.NORTH, grid.SOUTH), longitude=slice(grid.WEST, grid.EAST))


def fetch_day(ds, day, hours):
    """One UTC day: 24 hourly rain accumulations and the instantaneous fields at the wanted hours, all cut to the box."""
    t0 = np.datetime64(day.isoformat())
    out = {"tp": box(ds[PRECIP]).sel(time=slice(t0, t0 + np.timedelta64(23, "h"))).values.astype("float64")}
    if out["tp"].shape[0] != 24 or np.isnan(out["tp"]).any():
        raise RuntimeError(f"{day}: ERA5 rain is missing or incomplete (ERA5 runs about 6 days behind real time)")
    for key, name in INSTANT.items():
        arr = box(ds[name]).sel(time=[t0 + np.timedelta64(int(h), "h") for h in hours]).values.astype("float64")
        if np.isnan(arr).any():
            raise RuntimeError(f"{day}: ERA5 {name} is missing")
        out[key] = arr
    return out


def relative_humidity(t_c, td_c):
    """Magnus formula, over water."""
    return np.clip(100.0 * np.exp(17.625 * td_c / (243.04 + td_c)) / np.exp(17.625 * t_c / (243.04 + t_c)), 1.0, 100.0)


def noon_inputs(cur, prev, hours, hour_of_col):
    """T (deg C), RH (%), wind (km/h) and 24 h rain (mm) at each cell's noon, from today's fetch and yesterday's."""
    ny, nx = cur["tp"].shape[1:]
    sel = {h: np.broadcast_to(hour_of_col == h, (ny, nx)) for h in hours}
    pick = lambda a: sum(np.where(sel[h], a[i], 0.0) for i, h in enumerate(hours))
    t = pick(cur["t2m"]) - 273.15
    td = pick(cur["d2m"]) - 273.15
    wind = np.hypot(pick(cur["u10"]), pick(cur["v10"])) * 3.6
    rain = np.zeros((ny, nx))
    for h in hours:
        # accumulations ending at hours h+1..23 of yesterday and 0..h of today: 24 values
        total = prev["tp"][h + 1:].sum(axis=0) + cur["tp"][:h + 1].sum(axis=0)
        rain = np.where(sel[h], total, rain)
    return t, relative_humidity(t, td), wind, np.maximum(rain * 1000.0, 0.0)     # float noise can leave -0.0


def run(start, end, spinup_days, out_path):
    ds = xr.open_zarr(STORE, chunks=None)
    lat, lon = grid.lat_axis(), grid.lon_axis()
    hour_of_col = grid.noon_utc_hour(lon)[None, :]
    hours = sorted(int(h) for h in np.unique(hour_of_col))
    print(f"box {grid.SOUTH}..{grid.NORTH}N {grid.WEST}..{grid.EAST}E, noon UTC hours {hours}, "
          f"requested {start}..{end}, spin-up {spinup_days} days", flush=True)

    first = start - dt.timedelta(days=spinup_days)
    days = [first + dt.timedelta(days=i) for i in range((end - first).days + 1)]
    state = tuple(np.full((len(lat), len(lon)), v) for v in (fwi.FFMC_START, fwi.DMC_START, fwi.DC_START))
    kept, kept_days, prev = {k: [] for k in ("ffmc", "dmc", "dc", "isi", "bui", "wind", "rain24", "rh", "temp")}, [], None
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        # one extra day before the first day, for its rain before noon
        fetch_days = [first - dt.timedelta(days=1)] + days
        for b in range(0, len(fetch_days), BATCH_DAYS):
            batch = fetch_days[b:b + BATCH_DAYS]
            fetched = list(pool.map(lambda d: fetch_day(ds, d, hours), batch))
            for d, cur in zip(batch, fetched):
                if prev is not None:
                    t, rh, w, r = noon_inputs(cur, prev, hours, hour_of_col)
                    state, idx = fwi.step(state, t, rh, w, r, le=fwi.LE_EQUATORIAL, lf=fwi.LF_EQUATORIAL)
                    if d >= start:
                        for k in ("ffmc", "dmc", "dc", "isi", "bui"):
                            kept[k].append(idx[k].astype("float32"))
                        kept["wind"].append(w.astype("float32"))
                        kept["rain24"].append(r.astype("float32"))
                        kept["rh"].append(rh.astype("float32"))
                        kept["temp"].append(t.astype("float32"))
                        kept_days.append(d)
                prev = cur
            print(f"  fetched through {batch[-1]}  ({time.time() - t_start:.0f}s)", flush=True)

    lsm = ds["land_sea_mask"].sel(time="2020-01-01T00:00")
    lsm = box(lsm).values.astype("float32")
    coords = {"time": np.array(kept_days, dtype="datetime64[ns]"), "latitude": lat, "longitude": lon}
    out = xr.Dataset({k: (("time", "latitude", "longitude"), np.stack(v)) for k, v in kept.items()}, coords=coords)
    out["land_sea_mask"] = (("latitude", "longitude"), lsm)
    out.attrs["description"] = ("FWI System codes from ARCO-ERA5 at solar noon; equatorial day length (DMC Le = 9.0, DC Lf = 1.39); "
                                f"spin-up {spinup_days} days; code of NOR-X-424")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = {k: {"zlib": True, "complevel": 4} for k in out.data_vars}
    out.to_netcdf(out_path, encoding=enc)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB), {len(kept_days)} days", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, help="whole calendar year")
    p.add_argument("--start", help="first day, YYYY-MM-DD")
    p.add_argument("--end", help="last day (inclusive), YYYY-MM-DD")
    p.add_argument("--spinup-days", type=int, default=184, help="days run before the first day to forget the starting codes")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.year:
        start, end = dt.date(a.year, 1, 1), dt.date(a.year, 12, 31)
    elif a.start and a.end:
        start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    else:
        p.error("give --year or --start and --end")
    run(start, end, a.spinup_days, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
