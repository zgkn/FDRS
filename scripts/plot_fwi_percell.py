#!/usr/bin/env python3
"""One dot per grid cell: the cell's average FWI against its total number of hotspots over all the years in the table.

Input: the parquet of join_training.py (FWI is calculated from ISI and BUI as in plot_fwi_scatter.py).
Writes fwi_percell.png and fwi_percell.csv (one row per land cell: row, col, lat, lon, region, mean_fwi, total_hotspots,
fire_days, days).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy import stats

import fwi as fwi_system

INK, MUTED, GRID, ACCENT = "#1f2328", "#6b7280", "#e5e7eb", "#1d4e89"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(a.table, columns=["year", "row", "col", "lat", "lon", "region", "n_unique", "isi", "bui"])
    df["fwi"] = fwi_system.fwi(df["isi"].values, df["bui"].values).astype("float32")
    df["fire"] = df["n_unique"] > 0
    cells = df.groupby(["row", "col"], observed=True).agg(
        lat=("lat", "first"), lon=("lon", "first"), region=("region", "first"), mean_fwi=("fwi", "mean"),
        total_hotspots=("n_unique", "sum"), fire_days=("fire", "sum"), days=("fire", "size")).reset_index()
    years = f"{int(df['year'].min())}-{int(df['year'].max())}"
    cells.to_csv(a.outdir / "fwi_percell.csv", index=False)

    has = cells[cells["total_hotspots"] > 0]
    rho, _ = stats.spearmanr(cells["mean_fwi"], cells["total_hotspots"])
    rho_pos, _ = stats.spearmanr(has["mean_fwi"], has["total_hotspots"])
    # median of the dots in bands of mean FWI, to show the trend
    edges = np.quantile(has["mean_fwi"], np.linspace(0, 1, 11))
    band = pd.cut(has["mean_fwi"], np.unique(edges), include_lowest=True)
    trend = has.groupby(band, observed=True).agg(x=("mean_fwi", "median"), y=("total_hotspots", "median"))

    fig, ax = plt.subplots(figsize=(9.5, 6.6), facecolor="white")
    ax.scatter(has["mean_fwi"], has["total_hotspots"], s=14, color=ACCENT, alpha=0.4, linewidths=0)
    ax.plot(trend["x"], trend["y"], color=INK, linewidth=2, marker="o", markersize=5, markeredgecolor="white")
    ax.annotate("median of the dots,\nin tenths of the cells", (trend["x"].iloc[-3], trend["y"].iloc[-3]),
                textcoords="offset points", xytext=(-10, -42), ha="right", fontsize=8.5, color=INK)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Average FWI of the cell over all days", color=INK, fontsize=10)
    ax.set_ylabel(f"Total hotspots detected in the cell, {years} (log)", color=INK, fontsize=10)
    ax.set_title(f"FWI vs hotspots, one dot per grid cell ({len(has):,} cells with a fire)", loc="left", fontsize=12,
                 color=INK, fontweight="bold")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.text(0.02, 0.97, f"Spearman rho = {rho_pos:.2f}", transform=ax.transAxes, va="top", color=INK, fontsize=9.5)
    fig.text(0.01, 0.01, f"{len(cells):,} land cells of 0.25 degrees (about 28 km); {len(cells) - len(has)} with no hotspot at all "
             f"are not shown (rho {rho:.2f} with them). Hotspots: unique daytime VIIRS detections, three satellites de-duplicated.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(a.outdir / "fwi_percell.png", dpi=150)

    text = (f"cells {len(cells)}; with a fire {len(has)}; Spearman rho {rho_pos:.3f} (cells with fire), {rho:.3f} (all)\n"
            f"mean FWI of cells: median {cells['mean_fwi'].median():.2f}, range {cells['mean_fwi'].min():.2f}-{cells['mean_fwi'].max():.2f}\n"
            f"total hotspots per cell (cells with fire): median {has['total_hotspots'].median():.0f}, max {has['total_hotspots'].max()}\n"
            f"by island (cells, median mean FWI, median total hotspots, rho):\n")
    for r, g in has.groupby("region", observed=True):
        rr = stats.spearmanr(g["mean_fwi"], g["total_hotspots"])[0] if len(g) > 5 else float("nan")
        text += f"  {r}: {len(g)}, {g['mean_fwi'].median():.2f}, {g['total_hotspots'].median():.0f}, {rr:.2f}\n"
    (a.outdir / "percell_summary.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
