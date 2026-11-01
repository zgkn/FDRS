#!/usr/bin/env python3
"""Fire Weather Index (FWI) against the number of hotspots detected.

Input: the parquet of join_training.py. The FWI is the final index of the Canadian system, calculated from ISI and BUI
(scripts/fwi.py, checked against the 49-day test of NOR-X-424). Hotspots are the unique daytime VIIRS detections per 0.25 degree
cell and day (all three satellites, de-duplicated).

Figure, four panels:
  A  one dot per day: mean FWI over the land cells vs the day's hotspots in the whole box (log scale)
  B  every cell-day that had at least one hotspot: FWI vs its number of hotspots (density, log scale)
  C  mean hotspots per cell-day (zeros included) in bands of FWI
  D  share of cell-days with at least one hotspot in the same bands
Also written: fwi_binned.csv (the numbers of C and D), fwi_daily.csv (the dots of A), summary.txt, fwi_cellday.parquet.
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
EDGES = [0, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 100]


def plain_log_axis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def style(ax, xlabel, ylabel, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    cols = ["date", "year", "row", "col", "n_unique", "isi", "bui"]
    df = pd.read_parquet(a.table, columns=cols)
    df["fwi"] = fwi_system.fwi(df["isi"].values, df["bui"].values).astype("float32")
    n = len(df)
    years = f"{int(df['year'].min())}-{int(df['year'].max())}"
    fire = df["n_unique"].values > 0
    rho_cell, _ = stats.spearmanr(df["fwi"].values, df["n_unique"].values)

    daily = df.groupby("date").agg(fwi=("fwi", "mean"), hotspots=("n_unique", "sum"), year=("year", "first")).reset_index()
    daily_pos = daily[daily["hotspots"] > 0]
    rho_day, _ = stats.spearmanr(daily["fwi"], daily["hotspots"])

    df["band"] = pd.cut(df["fwi"], EDGES, right=False)
    binned = df.groupby("band", observed=True).agg(cell_days=("n_unique", "size"), mean_hotspots=("n_unique", "mean"),
                                                   share_with_fire=("n_unique", lambda v: float((v > 0).mean()))).reset_index()
    binned["band"] = binned["band"].astype(str)

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.2), facecolor="white")
    fig.suptitle(f"Fire Weather Index vs hotspots detected, southern ASEAN, {years}  ({n:,} land cell-days)",
                 x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")

    ax = axs[0, 0]
    ax.scatter(daily_pos["fwi"], daily_pos["hotspots"], s=9, color=ACCENT, alpha=0.45, linewidths=0)
    ax.set_yscale("log")
    plain_log_axis(ax)
    style(ax, "Mean FWI over land cells that day", "Hotspots in the whole box that day (log)",
          f"A  One dot per day ({len(daily_pos):,} days)")
    ax.text(0.03, 0.95, f"Spearman rho = {rho_day:.2f}", transform=ax.transAxes, va="top", color=INK, fontsize=9)

    ax = axs[0, 1]
    sub = df[fire]
    hb = ax.hexbin(sub["fwi"], sub["n_unique"], gridsize=55, yscale="log", bins="log", mincnt=1, cmap="Blues", linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label("cell-days (log)", color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    plain_log_axis(ax)
    style(ax, "FWI of the cell that day", "Hotspots in the cell that day (log)",
          f"B  Cell-days with a fire ({fire.sum():,}, {100 * fire.mean():.1f}% of all)")

    x = np.arange(len(binned))
    labels = [b.strip("[)").replace(", ", "-") for b in binned["band"]]
    ax = axs[1, 0]
    ax.plot(x, binned["mean_hotspots"], color=ACCENT, linewidth=2, marker="o", markersize=6, markeredgecolor="white")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    style(ax, "FWI band", "Mean hotspots per cell-day", "C  Mean hotspots per cell-day, by FWI band")
    ax = axs[1, 1]
    ax.plot(x, 100 * binned["share_with_fire"], color=ACCENT, linewidth=2, marker="o", markersize=6, markeredgecolor="white")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    style(ax, "FWI band", "Cell-days with at least one hotspot (%)", "D  Share of cell-days with a fire, by FWI band")
    for k in (0, len(x) // 2, len(x) - 1):
        ax.annotate(f"{100 * binned['share_with_fire'].iloc[k]:.0f}%", (x[k], 100 * binned["share_with_fire"].iloc[k]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK)
    fig.text(0.01, 0.005, f"Spearman rho over all cell-days = {rho_cell:.2f}. Bands with few cell-days are noisy "
             "(counts in fwi_binned.csv). Hotspots: unique daytime VIIRS detections, three satellites de-duplicated.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(a.outdir / "fwi_scatter.png", dpi=150)

    binned.to_csv(a.outdir / "fwi_binned.csv", index=False)
    daily.to_csv(a.outdir / "fwi_daily.csv", index=False)
    df[["date", "row", "col", "fwi", "n_unique"]].astype({"n_unique": "int16"}).to_parquet(a.outdir / "fwi_cellday.parquet", index=False)
    q = df["fwi"].quantile([0.5, 0.9, 0.99, 0.999]).round(1).to_dict()
    text = (f"cell-days {n:,}; with a fire {fire.sum():,} ({100 * fire.mean():.2f}%)\n"
            f"FWI percentiles (50/90/99/99.9): {q}; max {df['fwi'].max():.1f}\n"
            f"Spearman rho, cell-days: {rho_cell:.3f}; days: {rho_day:.3f}\n"
            f"FWI of fire cell-days: median {sub['fwi'].median():.1f}; of no-fire cell-days: {df.loc[~fire, 'fwi'].median():.1f}\n"
            f"{binned.round(4).to_string(index=False)}\n")
    (a.outdir / "summary.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
