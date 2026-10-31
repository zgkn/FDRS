"""The Newton solver of fit_logistic.py against statsmodels, on a small made-up problem."""
import numpy as np
import pytest
import statsmodels.api as sm

import fit_logistic as fl


def problem(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    dummy = (rng.random(n) < 0.3).astype(float)
    X = np.column_stack([np.ones(n), x1, x2, dummy])
    p = 1 / (1 + np.exp(-(-1.5 + 0.8 * x1 - 0.5 * x2 + 0.3 * dummy)))
    y = (rng.random(n) < p).astype(float)
    groups = rng.integers(0, 40, n)
    return X, y, groups


def test_coefficients_match_statsmodels():
    X, y, _ = problem()
    params, se, _ = fl.logit_fit(X, y, groups=np.arange(len(y)))        # every row its own cluster: close to the plain SEs
    ref = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    assert params == pytest.approx(ref.params, abs=1e-6)
    assert se == pytest.approx(ref.bse, rel=0.03)


def test_cluster_standard_errors_match_statsmodels():
    X, y, groups = problem()
    _, se, _ = fl.logit_fit(X, y, groups=groups)
    ref = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    assert se == pytest.approx(ref.bse, rel=0.03)


def test_chunking_does_not_change_the_answer(monkeypatch):
    X, y, groups = problem()
    a, se_a, _ = fl.logit_fit(X, y, groups=groups)
    monkeypatch.setattr(fl, "CHUNK", 777)
    b, se_b, _ = fl.logit_fit(X, y, groups=groups)
    assert a == pytest.approx(b, abs=1e-10) and se_a == pytest.approx(se_b, rel=1e-8)


def test_predict_is_a_probability():
    X, y, _ = problem()
    params, _, _ = fl.logit_fit(X, y)
    p = fl.predict(params, X)
    assert p.min() > 0 and p.max() < 1 and p.mean() == pytest.approx(y.mean(), abs=1e-6)
