#!/usr/bin/env python3
"""Logistic regression of "a unique daytime hotspot in the cell that day" on the FWI-System indices.

Input: the parquet of join_training.py. Models (all with island intercepts):
    region     the island intercept only: what the weather has to beat
    fine       + ln of the fine-fuel moisture content (from FFMC) and wind
    fine_dc    + DC/100                      (the smoke and peat dryness code)
    full       + ln(1 + DMC)                 (ISI and BUI are functions of FFMC, DMC, DC and the wind, so they are not added)

Validation: leave-one-year-out over the training years, and the hold-out years scored with the fit on all training years. Per
model it reports AUC, Brier score and the ratio of mean predicted to observed fire frequency (1.0 = right on average), and a
calibration table by predicted-risk decile. Standard errors are clustered by date (the cells of one day are not independent).

The fit is a small Newton (IRLS) solver on every row, not statsmodels: with millions of rows statsmodels needs about 1 KB per
row and the runner runs out of memory, while this one needs the design matrix and a few vectors. It is checked against
statsmodels in tests/test_fit_logistic.py.

Outputs in --outdir: report.md, metrics.csv, coefficients.csv, calibration.csv.
"""
import argparse
import gc
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

MODELS = {
    "region": [],
    "fine": ["ln_fmc", "wind"],
    "fine_dc": ["ln_fmc", "wind", "dc100"],
    "full": ["ln_fmc", "wind", "dc100", "ln_dmc"],
}
CHUNK = 500_000


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def features(df):
    f = pd.DataFrame(index=df.index)
    moisture = 147.2 * (101.0 - df["ffmc"].astype("float64")) / (59.5 + df["ffmc"].astype("float64"))   # Eq. 1 of NOR-X-424
    f["ln_fmc"] = np.log(np.maximum(moisture, 0.5))
    f["wind"] = df["wind"].astype("float64")
    f["dc100"] = df["dc"].astype("float64") / 100.0
    f["ln_dmc"] = np.log1p(df["dmc"].astype("float64"))
    return f


def design(df, feats, regions, cols):
    """Design matrix (numpy, float64) and its column names: constant, the chosen features, one dummy per island but the first."""
    parts, names = [np.ones((len(df), 1))], ["const"]
    for c in cols:
        parts.append(feats[c].values.reshape(-1, 1).astype("float64"))
        names.append(c)
    region = df["region_model"].astype(str).values
    for r in regions[1:]:                               # the first (largest) island is the reference
        parts.append((region == r).astype("float64").reshape(-1, 1))
        names.append(f"island_{r}")
    return np.hstack(parts), names


def predict(params, X):
    return 1.0 / (1.0 + np.exp(-(X @ params)))


def logit_fit(X, y, groups=None, max_iter=60, tol=1e-9):
    """Logistic regression by Newton's method. Returns (params, standard errors or None, number of iterations).

    With `groups` (integer cluster codes) the standard errors are the cluster-robust sandwich, with the usual small-sample factor.
    Memory: X plus a few vectors; the weighted cross-product is built in chunks.
    """
    n, k = X.shape
    beta = np.zeros(k)
    mean = min(max(y.mean(), 1e-6), 1 - 1e-6)
    beta[0] = np.log(mean / (1 - mean))                 # the first column is the constant
    ll_old = -np.inf
    for it in range(1, max_iter + 1):
        grad, hess, ll = np.zeros(k), np.zeros((k, k)), 0.0
        for a in range(0, n, CHUNK):
            Xc, yc = X[a:a + CHUNK], y[a:a + CHUNK]
            eta = Xc @ beta
            p = 1.0 / (1.0 + np.exp(-eta))
            ll += float(np.sum(yc * eta - np.logaddexp(0.0, eta)))
            w = np.maximum(p * (1.0 - p), 1e-10)
            grad += Xc.T @ (yc - p)
            hess += (Xc * w[:, None]).T @ Xc
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while True:                                     # halve the step if the likelihood fell
            cand = beta + t * step
            ll_new = float(sum(np.sum(y[a:a + CHUNK] * (X[a:a + CHUNK] @ cand)
                                      - np.logaddexp(0.0, X[a:a + CHUNK] @ cand)) for a in range(0, n, CHUNK)))
            if ll_new >= ll - 1e-9 or t < 1e-4:
                break
            t /= 2
        beta = cand
        if abs(ll_new - ll) < tol * (1 + abs(ll)):
            ll_old = ll_new
            break
        ll_old = ll_new
    else:
        print(f"  [!] logit_fit did not converge in {max_iter} iterations", flush=True)
    if groups is None:
        return beta, None, it
    # sandwich: bread = inverse Hessian, meat = sum over clusters of the squared cluster score
    hess, scores = np.zeros((k, k)), np.zeros((int(groups.max()) + 1, k))
    for a in range(0, n, CHUNK):
        Xc, yc, gc_ = X[a:a + CHUNK], y[a:a + CHUNK], groups[a:a + CHUNK]
        p = 1.0 / (1.0 + np.exp(-(Xc @ beta)))
        w = np.maximum(p * (1.0 - p), 1e-10)
        hess += (Xc * w[:, None]).T @ Xc
        r = yc - p
        for j in range(k):
            scores[:, j] += np.bincount(gc_, weights=r * Xc[:, j], minlength=scores.shape[0])
    bread = np.linalg.inv(hess)
    n_clusters = int((np.abs(scores).sum(axis=1) > 0).sum())
    meat = scores.T @ scores
    factor = n_clusters / max(n_clusters - 1, 1) * (n - 1) / max(n - k, 1)
    cov = factor * bread @ meat @ bread
    return beta, np.sqrt(np.diag(cov)), it


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

    df = pd.read_parquet(a.table, columns=["date", "year", "region_model", "y", "ffmc", "dmc", "dc", "wind"])
    missing = sorted(set(train_years + hold_years) - set(df["year"].unique()))
    if missing:
        raise SystemExit(f"[X] years not in the table: {missing}")
    df = df[df["year"].isin(train_years + hold_years)].reset_index(drop=True)
    regions = sorted(df["region_model"].astype(str).unique(), key=lambda r: -(df["region_model"].astype(str) == r).sum())
    print(f"{len(df)} cell-days; islands: {regions}; peak memory {peak_gb():.1f} GB", flush=True)
    feats = features(df)
    y_all = df["y"].values.astype("float64")
    date_code = pd.factorize(df["date"])[0]
    year = df["year"].values
    in_train = np.isin(year, train_years)

    metrics, coefs, calib = [], [], []
    for name, cols in MODELS.items():
        X, names = design(df, feats, regions, cols)
        for hold in (train_years if len(train_years) > 1 else []):          # leave one training year out
            tr, te = in_train & (year != hold), year == hold
            params, _, _ = logit_fit(X[tr], y_all[tr])
            metrics.append({"model": name, "split": f"leave-out {hold}", **score(y_all[te], predict(params, X[te]))})
        params, se, iters = logit_fit(X[in_train], y_all[in_train], groups=date_code[in_train])    # all training years
        p_tr = predict(params, X[in_train])
        metrics.append({"model": name, "split": "train (in sample)", **score(y_all[in_train], p_tr)})
        for hy in hold_years:
            te = year == hy
            p = predict(params, X[te])
            metrics.append({"model": name, "split": f"hold-out {hy}", **score(y_all[te], p)})
            calib.append(calibration(y_all[te], p, f"{name} hold-out {hy}"))
        calib.append(calibration(y_all[in_train], p_tr, f"{name} train"))
        coefs.append(pd.DataFrame({"model": name, "term": names, "coef": params, "se": se,
                                   "lo95": params - 1.96 * se, "hi95": params + 1.96 * se}))
        del X, p_tr
        gc.collect()
        print(f"fitted {name} ({iters} Newton steps; peak memory so far {peak_gb():.1f} GB)", flush=True)

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
             "## Coefficients (model fine_dc, all training years; standard errors clustered by date)", "",
             c[c["model"] == "fine_dc"].drop(columns="model").round(3).to_markdown(index=False), "",
             "## Coefficients (model full)", "",
             c[c["model"] == "full"].drop(columns="model").round(3).to_markdown(index=False), "",
             "## Calibration of the full model on the hold-out year(s)", "",
             cal[cal["split"].str.startswith("full hold-out")].round(3).to_markdown(index=False), ""]
    (a.outdir / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:14]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
