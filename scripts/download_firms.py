#!/usr/bin/env python3
"""Daily VIIRS active-fire counts per 0.25 degree cell from NASA FIRMS (S-NPP, NOAA-20, NOAA-21), de-duplicated across sensors.

Needs a FIRMS MAP_KEY in the environment variable FIRMS_MAP_KEY (https://firms.modaps.eosdis.nasa.gov/api/map_key/). The key
is part of FIRMS request URLs, so this script never prints a URL.

What is counted (the same rules as hotspot_aod_plots):
  * daytime detections only (daynight == 'D'): the afternoon passes of all three satellites are at about 13:30 local time
  * nominal or high confidence (confidence 'n'/'h'; the numeric 'nominal' band of old files is 30..100)
  * vegetation fires only when the file has a `type` column (type 0); volcanoes, static land sources and offshore are dropped
  * a day is the UTC acquisition date (the afternoon passes over the box fall at 04-08 UTC, so it is also the local date)
  * one fire is usually seen by several sensors/passes: a pixel is dropped when a higher-priority sensor (NOAA-21, then NOAA-20,
    then S-NPP) already has a pixel within DEDUPE_RADIUS_M of it. Pixels of the same sensor never drop each other, so a fire
    front covering many pixels still counts as many.

Sources: for each sensor and date the standard-processing product (SP) is used where FIRMS says it has the date, else
near-real-time (NRT). `data_availability` is asked first and printed.

Outputs (under --outdir):
  hotspots_cells_<tag>.csv.gz   date, row, col, lat, lon, n_unique, n_snpp, n_n20, n_n21 (n_* are the pixels each sensor keeps
                                after de-duplication, so they add up to n_unique), raw_snpp, raw_n20, raw_n21 (before it)
  hotspots_days_<tag>.csv       date, rows_* (all detections in the box, day or night, any confidence: a sensor with 0 rows
                                that day is not "active"), active_* (0/1), src_* (SP/NRT/none)
"""
import argparse
import csv
import datetime as dt
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import grid

BASE = "https://firms.modaps.eosdis.nasa.gov"
# (name, priority order for de-duplication, SP source, NRT source)
SENSORS = [
    ("n21", "VIIRS_NOAA21_SP", "VIIRS_NOAA21_NRT"),
    ("n20", "VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"),
    ("snpp", "VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"),
]
DEDUPE_RADIUS_M = 500.0
M_PER_DEG = 111195.0
KEEP_CONFIDENCE = {"n", "h", "nominal", "high"}
EDGE = grid.STEP / 2                       # request half a cell beyond the box so the edge cells are whole
RETRIES = 6


# ------------------------------------------------------------------------------------------------------------- requests
def get(url, what, session):
    """GET with retries. The URL holds the key, so errors name `what` and the status only."""
    last = None
    for attempt in range(RETRIES):
        try:
            r = session.get(url, timeout=180)
            if r.status_code == 200:
                return r.text
            last = f"HTTP {r.status_code}"
            if r.status_code in (401, 403):
                raise SystemExit(f"[X] FIRMS refused the request for {what} ({last}): check FIRMS_MAP_KEY")
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"{what}: failed after {RETRIES} tries ({last})")


def availability(key, session):
    """{source: (first date, last date)} from FIRMS data_availability."""
    text = get(f"{BASE}/api/data_availability/csv/{key}/ALL", "data_availability", session)
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            out[row["data_id"]] = (dt.date.fromisoformat(row["min_date"]), dt.date.fromisoformat(row["max_date"]))
        except (KeyError, ValueError):
            continue
    return out


def pick_source(avail, sp, nrt, day):
    for src in (sp, nrt):
        if src in avail and avail[src][0] <= day <= avail[src][1]:
            return src
    return None


def source_segments(avail, sp, nrt, days):
    """Consecutive runs of days that share one source: [(source or None, first day, last day)]."""
    segs = []
    for d in days:
        s = pick_source(avail, sp, nrt, d)
        if segs and segs[-1][0] == s and (d - segs[-1][2]).days == 1:
            segs[-1][2] = d
        else:
            segs.append([s, d, d])
    return segs


def fetch_range(key, source, first, last, session, step_days):
    """All rows of one source for first..last (inclusive), `step_days` days per request, as one DataFrame."""
    west, south, east, north = grid.WEST - EDGE, grid.SOUTH - EDGE, grid.EAST + EDGE, grid.NORTH + EDGE
    frames = []
    d = first
    while d <= last:
        n = min(step_days, (last - d).days + 1)
        text = get(f"{BASE}/api/area/csv/{key}/{source}/{west},{south},{east},{north}/{n}/{d.isoformat()}",
                   f"{source} {d}+{n}d", session)
        head = text.lstrip()[:200].lower()
        if "latitude" in head:
            frames.append(pd.read_csv(io.StringIO(text)))
        elif head and (head.startswith("<") or any(w in head for w in ("invalid", "error", "exceed", "limit"))):
            raise RuntimeError(f"{source} {d}: unexpected FIRMS answer: {head[:80]!r}")       # a message, not data
        # else: an empty answer, no detections in those days
        d += dt.timedelta(days=n)
        time.sleep(0.2)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------------------------------------- filtering
def good_daytime(df):
    """Rows that count: daytime, nominal/high confidence, vegetation fire."""
    if df.empty:
        return df
    keep = df["daynight"].astype(str).str.upper().eq("D")
    conf = df["confidence"]
    if pd.api.types.is_numeric_dtype(conf):
        keep &= conf >= 30
    else:
        keep &= conf.astype(str).str.lower().isin(KEEP_CONFIDENCE)
    if "type" in df.columns:
        keep &= df["type"].fillna(0).astype(int).eq(0)
    return df[keep]


def metres(lat, lon):
    lat = np.asarray(lat, "float64")
    return np.column_stack([np.asarray(lon, "float64") * M_PER_DEG * np.cos(np.radians(lat)), lat * M_PER_DEG])


def dedupe(parts, radius_m=DEDUPE_RADIUS_M):
    """parts: [(name, DataFrame with latitude/longitude)] in priority order.

    Returns one DataFrame of the kept pixels with a `sensor` column. A pixel is dropped when a higher-priority part already
    has a pixel within radius_m; pixels of the same part never drop each other.
    """
    from scipy.spatial import cKDTree
    kept, kept_xy = [], np.empty((0, 2))
    for name, df in parts:
        if df is None or df.empty:
            continue
        xy = metres(df["latitude"].values, df["longitude"].values)
        keep = np.ones(len(df), bool)
        if len(kept_xy):
            dist, _ = cKDTree(kept_xy).query(xy, distance_upper_bound=radius_m)
            keep = ~np.isfinite(dist)
        mine = df[keep].assign(sensor=name)
        kept.append(mine)
        kept_xy = np.vstack([kept_xy, xy[keep]])
    cols = ["latitude", "longitude", "sensor"]
    return pd.concat(kept, ignore_index=True)[cols] if kept else pd.DataFrame(columns=cols)


def aggregate_day(day, parts):
    """The cell table of one UTC day from the good daytime rows of each sensor (parts in priority order)."""
    uniq = dedupe(parts)
    rows = {}
    for name, df in parts:
        if df is not None and not df.empty:
            r, c = grid.cell_index(df["latitude"].values, df["longitude"].values)
            for rr, cc in zip(r[r >= 0], c[r >= 0]):
                rows.setdefault((rr, cc), {})[f"raw_{name}"] = rows.get((rr, cc), {}).get(f"raw_{name}", 0) + 1
    if not uniq.empty:
        r, c = grid.cell_index(uniq["latitude"].values, uniq["longitude"].values)
        for rr, cc, s in zip(r, c, uniq["sensor"].values):
            if rr >= 0:
                rows.setdefault((rr, cc), {})[f"n_{s}"] = rows.get((rr, cc), {}).get(f"n_{s}", 0) + 1
    lats, lons = grid.lat_axis(), grid.lon_axis()
    out = []
    for (rr, cc), v in sorted(rows.items()):
        rec = {"date": day.isoformat(), "row": int(rr), "col": int(cc), "lat": float(lats[rr]), "lon": float(lons[cc])}
        for name, _, _ in SENSORS:
            rec[f"n_{name}"] = v.get(f"n_{name}", 0)
        rec["n_unique"] = sum(rec[f"n_{name}"] for name, _, _ in SENSORS)
        for name, _, _ in SENSORS:
            rec[f"raw_{name}"] = v.get(f"raw_{name}", 0)
        out.append(rec)
    cols = ["date", "row", "col", "lat", "lon", "n_unique"] + [f"n_{n}" for n, _, _ in SENSORS] + [f"raw_{n}" for n, _, _ in SENSORS]
    return pd.DataFrame(out, columns=cols)


# -------------------------------------------------------------------------------------------------------------- main
def run(first, last, outdir, tag, step_days):
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key:
        raise SystemExit("[X] set FIRMS_MAP_KEY (a FIRMS MAP_KEY) in the environment")
    session = requests.Session()
    avail = availability(key, session)
    print("FIRMS data availability for the sources used:", flush=True)
    for _, sp, nrt in SENSORS:
        for s in (sp, nrt):
            print(f"  {s:20s} {avail[s][0]} .. {avail[s][1]}" if s in avail else f"  {s:20s} not listed", flush=True)

    days = [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]
    raw = {}                                   # sensor name -> DataFrame of every row in the box for first..last
    src_of = {}                                # (sensor, day) -> source used
    for name, sp, nrt in SENSORS:
        frames = []
        for src, a, b in source_segments(avail, sp, nrt, days):
            for d in days:
                if a <= d <= b:
                    src_of[(name, d)] = src or "none"
            if src is None:
                print(f"  {name}: no FIRMS data for {a}..{b}", flush=True)
                continue
            t0 = time.time()
            df = fetch_range(key, src, a, b, session, step_days)
            print(f"  {name}: {src} {a}..{b}: {len(df)} rows ({time.time() - t0:.0f}s)", flush=True)
            if not df.empty:
                frames.append(df)
        raw[name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    cells, day_rows = [], []
    for d in days:
        iso = d.isoformat()
        parts, rec = [], {"date": iso}
        for name, _, _ in SENSORS:
            df = raw[name]
            today = df[df["acq_date"].astype(str) == iso] if not df.empty else df
            rec[f"rows_{name}"] = int(len(today))
            rec[f"active_{name}"] = int(len(today) > 0)
            rec[f"src_{name}"] = src_of.get((name, d), "none")
            parts.append((name, good_daytime(today)))
        day_rows.append(rec)
        cells.append(aggregate_day(d, parts))

    outdir.mkdir(parents=True, exist_ok=True)
    cell_df = pd.concat(cells, ignore_index=True)
    cell_df.to_csv(outdir / f"hotspots_cells_{tag}.csv.gz", index=False)
    pd.DataFrame(day_rows).to_csv(outdir / f"hotspots_days_{tag}.csv", index=False)
    total = cell_df["n_unique"].sum()
    print(f"wrote {len(cell_df)} cell-days, {int(total)} unique daytime detections, to {outdir}", flush=True)
    day_df = pd.DataFrame(day_rows)
    for name, _, _ in SENSORS:
        print(f"  {name}: active on {int(day_df['active_' + name].sum())} of {len(day_df)} days, "
              f"{int(cell_df['raw_' + name].sum())} daytime detections before de-duplication, "
              f"{int(cell_df['n_' + name].sum())} kept", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--step-days", type=int, default=5, help="days per FIRMS request (the API allows up to 10)")
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args()
    if a.year:
        first, last, tag = dt.date(a.year, 1, 1), dt.date(a.year, 12, 31), str(a.year)
    elif a.start and a.end:
        first, last = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
        tag = f"{first:%Y%m%d}_{last:%Y%m%d}"
    else:
        p.error("give --year or --start and --end")
    if not 1 <= a.step_days <= 10:
        p.error("--step-days must be 1..10")
    run(first, last, a.outdir, tag, a.step_days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
