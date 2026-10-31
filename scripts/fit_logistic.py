#!/usr/bin/env python3
"""Logistic regression of "a unique daytime hotspot in the cell that day" on the FWI-System indices.

Input: the parquet of join_training.py. Models (all with island intercepts):
    region     the island intercept only: what the weather has to beat
    fine       + ln of the fine-fuel moisture content (from FFMC) and wind
    fine_dc    + DC/100                      (the smoke and peat dryness code)
    full       + ln(1 + DMC)                 (DMC and DC and FFMC are not collinear enough to matter; ISI and BUI are functions
                                              of these and the wind, so they are not added)

Validation: leave-one-year-out over the training years, and the hold-out years fitted on all training years. Per model it
reports AUC, Brier score and the ratio of mean predicted to observed fire frequency (1.0 = right on average), and a
calibration table by predicted-risk decile. Standard errors are clustered by date (the cells of one day are not independent).

Outputs in --outdir: report.md, metrics.csv, coefficients.csv, calibration.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score

MODELS = {
    "region": [],
    "fine": ["ln_fmc", "wind"],
    "fine_dc": ["ln_fmc", "wind", "dc100"],
    "full": ["ln_fmc", "wind", "dc100", "ln_dmc"],
}


def features(df):
    f = pd.DataFrame(index=df.index)
    moisture = 147.2 * (101.0 - df["ffmc"].astype("float64")) / (59.5 + df["ffmc"].astype("float64"))   # Eq. 1 of NOR-X-424
    f["ln_fmc"] = np.log(np.maximum(moisture, 0.5))
    f["wind"] = df["wind"].astype("float64")
    f["dc100"] = df["dc"].astype("float64") / 100.0
    f["ln_dmc"] = np.log1p(df["dmc"].astype("float64"))
    return f


def design(df, feats, regions, cols):
    x = feats[cols].copy() if cols else pd.DataFrame(index=df.index)
    for r in regions[1:]:                               # first island is the reference
        x[f"island_{r}"] = (df["region"].astype(str) == r).astype("float64")
    return sm.add_constant(x, has_constant="add")


def fit(x, y, groups):
    model = sm.GLM(y, x, family=sm.families.Binomial())
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


def score(y, p):
    return {"n": int(len(y)), "fire_frac": float(y.mean()), "mean_pred": float(p.mean()),
            "pred_over_obs": float(p.mean() / max(y.mean(), 1e-12)),
            "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
            "brier": float(brier_score_loss(y, p))}


def calibration(y, p, label):
    bins = pd.qcut(p, 10, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "bin": bins}).groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")).reset_index(drop=True)
    t.insert(0, "split", label)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--train-years", default="2019,2020,2021,2022,2023")
    ap.add_argument("--holdout-years", default="2015")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    train_years = [int(y) for y in a.train_years.split(",") if y]
    hold_years = [int(y) for y in a.holdout_years.split(",") if y]

    df = pd.read_parquet(a.table)
    missing = sorted(set(train_years + hold_years) - set(df["year"].unique()))
    if missing:
        raise SystemExit(f"[X] years not in the table: {missing}")
    df = df[df["year"].isin(train_years + hold_years)].reset_index(drop=True)
    regions = sorted(df["region"].astype(str).unique(), key=lambda r: -(df["region"].astype(str) == r).sum())
    print(f"{len(df)} cell-days; islands: {regions}", flush=True)
    feats = features(df)
    y_all = df["y"].values.astype("float64")
    date_code = pd.factorize(df["date"])[0]

    metrics, coefs, calib = [], [], []
    for name, cols in MODELS.items():
        x = design(df, feats, regions, cols)
        # leave one training year out
        for hold in (train_years if len(train_years) > 1 else []):
            tr, te = (df["year"].isin(train_years) & (df["year"] != hold)).values, (df["year"] == hold).values
            res = fit(x[tr], y_all[tr], date_code[tr])
            p = res.predict(x[te])
            metrics.append({"model": name, "split": f"leave-out {hold}", **score(y_all[te], p.values)})
        # all training years, then the hold-out years
        tr = df["year"].isin(train_years).values
        res = fit(x[tr], y_all[tr], date_code[tr])
        p_tr = res.predict(x[tr])
        metrics.append({"model": name, "split": "train (in sample)", **score(y_all[tr], p_tr.values)})
        for hy in hold_years:
            te = (df["year"] == hy).values
            p = res.predict(x[te])
            metrics.append({"model": name, "split": f"hold-out {hy}", **score(y_all[te], p.values)})
            calib.append(calibration(y_all[te], p.values, f"{name} hold-out {hy}"))
        calib.append(calibration(y_all[tr], p_tr.values, f"{name} train"))
        ci = res.conf_int()
        coefs.append(pd.DataFrame({"model": name, "term": res.params.index, "coef": res.params.values,
                                   "se": res.bse.values, "lo95": ci[0].values, "hi95": ci[1].values}))
        print(f"fitted {name}", flush=True)

    a.outdir.mkdir(parents=True, exist_ok=True)
    m, c, cal = pd.DataFrame(metrics), pd.concat(coefs, ignore_index=True), pd.concat(calib, ignore_index=True)
    m.to_csv(a.outdir / "metrics.csv", index=False)
    c.to_csv(a.outdir / "coefficients.csv", index=False)
    cal.to_csv(a.outdir / "calibration.csv", index=False)

    lines = ["# Fire occurrence vs FWI-System indices", "",
             f"Training years {train_years}, hold-out {hold_years}. {len(df):,} land cell-days; a fire is at least one unique "
             "daytime VIIRS detection in the 0.25 degree cell that day.", "",
             "## Skill (AUC: 0.5 = no skill; pred/obs: 1.0 = right on average)", "",
             m.pivot_table(index="split", columns="model", values="auc", sort=False)[list(MODELS)].round(3).to_markdown(), "",
             m.pivot_table(index="split", columns="model", values="pred_over_obs", sort=False)[list(MODELS)].round(2).to_markdown(), "",
             "## Coefficients (model fine_dc, all training years)", "",
             c[c["model"] == "fine_dc"].drop(columns="model").round(3).to_markdown(index=False), "",
             "## Calibration of the full model on the hold-out year(s)", "",
             cal[cal["split"].str.startswith("full hold-out")].round(3).to_markdown(index=False), ""]
    (a.outdir / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:14]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
