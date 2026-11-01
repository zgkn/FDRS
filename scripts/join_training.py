#!/usr/bin/env python3
"""Join the FIRMS cell counts and the ERA5 FWI indices into one training table, one row per land cell and day.

Reads  <hotspots-dir>/**/hotspots_cells_<year>.csv.gz   (from download_firms.py)
       <fwi-dir>/**/fwi_<year>.nc                       (from era5_fwi.py)
Writes <out> (parquet): date, year, row, col, lat, lon, region (the place), region_model (the place whose term the regression uses:
       Brunei and Singapore share their neighbour's), y (a unique daytime hotspot was seen in the cell that day),
       n_unique, and the day's ffmc, dmc, dc, isi, bui, fwi (the final Fire Weather Index, from ISI and BUI), wind, rain24, rh, temp.

Only squares of the study area are kept: land squares (ERA5 land fraction >= 0.5; Singapore, a mostly coastal square, >= 0.1) that
lie in Peninsular Malaysia, Sabah, Sarawak, Brunei, Singapore, Sumatra, Kalimantan or Java (scripts/study_area.py,
data/study_boundaries.geojson). Hotspots in any other square are not in the table; how many that is, is printed.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import fwi as fwi_system
import grid
import study_area


def find(root, pattern):
    return {re.search(r"(\d{4})", p.name).group(1): p for p in sorted(Path(root).rglob(pattern))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hotspots-dir", type=Path, required=True)
    ap.add_argument("--fwi-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boundaries", type=Path, default=study_area.BOUNDARIES)
    a = ap.parse_args()

    hot, fw = find(a.hotspots_dir, "hotspots_cells_*.csv.gz"), find(a.fwi_dir, "fwi_*.nc")
    years = sorted(set(hot) & set(fw))
    print(f"hotspot files for {sorted(hot)}, FWI files for {sorted(fw)}; joining {years}", flush=True)
    if not years:
        raise SystemExit("[X] no year has both a hotspot file and an FWI file")

    boundaries, frames, printed = study_area.load(a.boundaries), [], False
    for y in years:
        ds = xr.open_dataset(fw[y])
        lat, lon = ds["latitude"].values, ds["longitude"].values
        assert len(lat) == len(grid.lat_axis()) and len(lon) == len(grid.lon_axis()), "FWI grid is not the study grid"
        names = study_area.label_cells(lat, lon, ds["land_sea_mask"].values, boundaries)
        land = names != ""
        if not printed:
            print("study squares by place:")
            for place, n in pd.Series(names[land]).value_counts().items():
                print(f"   {place}: {n}")
            print(f"study squares: {int(land.sum())}", flush=True)
            printed = True
        times = ds["time"].values
        counts = np.zeros((len(times), len(lat), len(lon)), dtype="int32")
        h = pd.read_csv(hot[y])
        day_index = {d: i for i, d in enumerate(pd.to_datetime(times).strftime("%Y-%m-%d"))}
        h = h[h["date"].isin(day_index)]
        counts[h["date"].map(day_index).values, h["row"].values, h["col"].values] = h["n_unique"].values
        on_land = counts[:, land].sum()
        print(f"{y}: {int(counts.sum())} unique detections in the box, {int(on_land)} "
              f"({100 * on_land / max(counts.sum(), 1):.1f}%) in the study squares", flush=True)

        t_idx, r_idx, c_idx = np.where(np.broadcast_to(land, counts.shape))
        df = pd.DataFrame({
            "date": pd.to_datetime(times)[t_idx], "year": int(y), "row": r_idx.astype("int16"), "col": c_idx.astype("int16"),
            "lat": lat[r_idx].astype("float32"), "lon": lon[c_idx].astype("float32"),
            "region": pd.Categorical(names[r_idx, c_idx]),
            "region_model": pd.Categorical([study_area.MODEL_GROUP.get(n, n) for n in names[r_idx, c_idx]]),
            "n_unique": counts[t_idx, r_idx, c_idx],
        })
        df["y"] = (df["n_unique"] > 0).astype("int8")
        for v in ("ffmc", "dmc", "dc", "isi", "bui", "wind", "rain24", "rh", "temp"):
            df[v] = ds[v].values[t_idx, r_idx, c_idx]
        df["fwi"] = fwi_system.fwi(df["isi"].values, df["bui"].values).astype("float32")     # final FWI from ISI and BUI
        frames.append(df)
        ds.close()

    out = pd.concat(frames, ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(out)} cell-days, {len(years)} years, fire fraction {out['y'].mean():.3f}", flush=True)
    by = out.groupby("region", observed=True).agg(cells=("row", lambda v: v.nunique()), fire_cell_days=("y", "sum"),
                                                   fire_share=("y", "mean"), hotspots=("n_unique", "sum"))
    print(by.to_string())
    by.to_csv(a.out.parent / "region_summary.csv")
    print(out.groupby("year")["y"].agg(["mean", "sum"]).rename(columns={"mean": "fire_frac", "sum": "fire_cell_days"}).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
