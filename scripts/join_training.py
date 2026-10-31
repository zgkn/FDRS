#!/usr/bin/env python3
"""Join the FIRMS cell counts and the ERA5 FWI indices into one training table, one row per land cell and day.

Reads  <hotspots-dir>/**/hotspots_cells_<year>.csv.gz   (from download_firms.py)
       <fwi-dir>/**/fwi_<year>.nc                       (from era5_fwi.py)
Writes <out> (parquet): date, year, row, col, lat, lon, region, y (a unique daytime hotspot was seen in the cell that day),
       n_unique, and the day's ffmc, dmc, dc, isi, bui, wind, rain24, rh, temp.

Land cells are those with ERA5 land-sea mask >= --land-threshold (default 0.5). Hotspots in other (coastal or sea) cells are
not in the table; how many that is, is printed.

Regions are the islands of the land mask (connected cells), named by where their centre is, so no outside boundary file is
needed; the table of islands is printed so the naming can be checked.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

import grid

SMALL_ISLAND_CELLS = 40            # smaller islands are lumped into "Other"


def name_islands(land, lat, lon):
    """Island label array (strings) of the land mask, plus a printable table."""
    comp, n = ndimage.label(land)                 # 4-connected
    names = np.full(land.shape, "", dtype=object)
    table = []
    for k in range(1, n + 1):
        rr, cc = np.where(comp == k)
        clat, clon, size = lat[rr].mean(), lon[cc].mean(), len(rr)
        if size < SMALL_ISLAND_CELLS:
            name = "Other"
        elif clat < -6.0 and clon > 104.5:
            name = "Java and Nusa Tenggara"
        elif clon > 117.5 and clat < 1.0:
            name = "Sulawesi (west edge)"
        elif clon >= 108.0 and clat > -4.5:
            name = "Borneo"
        elif clat > 2.5 and 98.0 <= clon <= 104.5:
            name = "Peninsular Malaysia"
        elif clon < 108.0 and clat < 6.5:
            name = "Sumatra"
        else:
            name = "Other"
        names[comp == k] = name
        table.append((k, name, size, round(float(clat), 2), round(float(clon), 2)))
    return names, table


def find(root, pattern):
    return {re.search(r"(\d{4})", p.name).group(1): p for p in sorted(Path(root).rglob(pattern))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hotspots-dir", type=Path, required=True)
    ap.add_argument("--fwi-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--land-threshold", type=float, default=0.5)
    a = ap.parse_args()

    hot, fw = find(a.hotspots_dir, "hotspots_cells_*.csv.gz"), find(a.fwi_dir, "fwi_*.nc")
    years = sorted(set(hot) & set(fw))
    print(f"hotspot files for {sorted(hot)}, FWI files for {sorted(fw)}; joining {years}", flush=True)
    if not years:
        raise SystemExit("[X] no year has both a hotspot file and an FWI file")

    frames, islands_printed = [], False
    for y in years:
        ds = xr.open_dataset(fw[y])
        lat, lon = ds["latitude"].values, ds["longitude"].values
        assert len(lat) == len(grid.lat_axis()) and len(lon) == len(grid.lon_axis()), "FWI grid is not the study grid"
        land = ds["land_sea_mask"].values >= a.land_threshold
        names, table = name_islands(land, lat, lon)
        if not islands_printed:
            print("land islands (id, name, cells, centre lat, centre lon):")
            for t in table:
                print("  ", t)
            print(f"land cells: {int(land.sum())} of {land.size}", flush=True)
            islands_printed = True
        times = ds["time"].values
        counts = np.zeros((len(times), len(lat), len(lon)), dtype="int32")
        h = pd.read_csv(hot[y])
        day_index = {d: i for i, d in enumerate(pd.to_datetime(times).strftime("%Y-%m-%d"))}
        h = h[h["date"].isin(day_index)]
        counts[h["date"].map(day_index).values, h["row"].values, h["col"].values] = h["n_unique"].values
        on_land = counts[:, land].sum()
        print(f"{y}: {int(counts.sum())} unique detections in the box, {int(on_land)} "
              f"({100 * on_land / max(counts.sum(), 1):.1f}%) in land cells", flush=True)

        t_idx, r_idx, c_idx = np.where(np.broadcast_to(land, counts.shape))
        df = pd.DataFrame({
            "date": pd.to_datetime(times)[t_idx], "year": int(y), "row": r_idx.astype("int16"), "col": c_idx.astype("int16"),
            "lat": lat[r_idx].astype("float32"), "lon": lon[c_idx].astype("float32"),
            "region": pd.Categorical(names[r_idx, c_idx]),
            "n_unique": counts[t_idx, r_idx, c_idx],
        })
        df["y"] = (df["n_unique"] > 0).astype("int8")
        for v in ("ffmc", "dmc", "dc", "isi", "bui", "wind", "rain24", "rh", "temp"):
            df[v] = ds[v].values[t_idx, r_idx, c_idx]
        frames.append(df)
        ds.close()

    out = pd.concat(frames, ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(out)} cell-days, {len(years)} years, fire fraction {out['y'].mean():.3f}", flush=True)
    print(out.groupby("year")["y"].agg(["mean", "sum"]).rename(columns={"mean": "fire_frac", "sum": "fire_cell_days"}).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
