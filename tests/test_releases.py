"""The two results once implemented and withheld, released and accepted against two references (P6; divergences D121, D126;
docs/cens-trunc-families.md, docs/post-processing.md):

- a lower truncation bound of a Poisson or negative binomial response, ``trunc(lb = )``: the park ``formula.PARKED_TRUNCATION``
  is empty, the formulas build, each row's log mass is divided by ``P(Y >= lb)`` and ``predict(kind="mean")`` is the mean of
  the distribution the likelihood truncates, ``E[Y | Y >= lb]`` (SciPy's, by hand), where brms's ``posterior_epred`` is
  ``E[Y | Y >= lb + 1]`` (the frozen case ``negbinomial_truncated`` passes G4 on its 50-digit reference); ``describe()`` says so;
- the exact stacking weights: the park ``post_processing.PARKED_WEIGHTS`` is empty, ``ab.loo_model_weights`` is exported and
  ``model_weights``'s default is ``"stacking"``, brms's (the frozen case ``gauss_loo_compare_x_stacking`` passes the post gate
  with its weights on the 50-digit optimum);
- the park mechanisms themselves still refuse what a park would hold.

The oracle gates of the two cases and the P6 contract of the new ``"weights"`` kind are in tests/oracle (``test_baseline.py``,
``test_harness.py``). Nothing samples."""
from types import MappingProxyType

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats

import autobayes as ab
import autobayes.formula as F
from autobayes import post_processing
from autobayes.formula import parse_formula
from autobayes.post_processing import loo_from_log_lik, stacking_weights

FRAME = pd.DataFrame({
    "x": [-1.0, 0.5, 2.0, 0.3, -0.2, 1.1, 0.0, -0.7, 0.9, -1.4],
    "ytr": [1, 2, 5, 1, 3, 6, 1, 2, 4, 1], "ymid": [2, 3, 5, 2, 3, 6, 2, 4, 4, 2],
})
PRIORS = {"poisson": ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1)),
          "negbinomial": ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), shape=ab.Gamma(0.01, 0.01))}
SHAPE = 1.4


def _model(family, term, response="ytr"):
    return getattr(ab, family)(f"{response} | {term} ~ x", FRAME, PRIORS[family])


def _distribution(family, eta):
    mu = np.exp(eta)
    return stats.poisson(mu) if family == "poisson" else stats.nbinom(SHAPE, SHAPE / (SHAPE + mu))


def _draws(model, family):
    post = {"Intercept": np.full((1, 2), 0.3), "x": np.full((1, 2), -0.4)}
    if family == "negbinomial":
        post[model.names["shape"]] = np.full((1, 2), SHAPE)
    return az.from_dict(posterior=post)


# -- the lower bound of a count response (D121) ------------------------------------------------------------------------
def test_the_truncation_park_is_empty_and_its_mechanism_still_refuses(monkeypatch):
    assert dict(F.PARKED_TRUNCATION) == {}
    for family in ("poisson", "negbinomial"):
        assert parse_formula("ytr | trunc(lb = 1) ~ x", family=family).addition("trunc").lb == 1.0
    monkeypatch.setattr(F, "PARKED_TRUNCATION", MappingProxyType({"count_lower_bound": "held back (a reason)"}))
    with pytest.raises(ab.FormulaError, match=r"'trunc\(lb = 1\)': held back \(a reason\)"):
        parse_formula("ytr | trunc(lb = 1) ~ x", family="poisson")


@pytest.mark.parametrize("family", ["poisson", "negbinomial"])
@pytest.mark.parametrize("lb,ub", [(1, None), (2, 7)])
def test_a_count_lower_bound_builds_with_the_truncated_likelihood_and_mean(family, lb, ub):
    """Built without lifting anything: each row's log mass minus log P(lb <= Y <= ub), and the mean E[Y | lb <= Y <= ub] over
    the support (SciPy), the mean of the distribution the likelihood truncates."""
    term, response = (f"trunc(lb = {lb})", "ytr") if ub is None else (f"trunc(lb = {lb}, ub = {ub})", "ymid")
    m = _model(family, term, response)
    assert np.array_equal(m.design.lb, np.full(10, float(lb))) and np.array_equal(m.design.ub, np.full(10, np.inf if ub is None else ub))
    eta = 0.3 - 0.4 * FRAME.x.to_numpy()                       # the raw-scale intercept 0.3 and the slope -0.4
    d = _distribution(family, eta)
    mass = (d.cdf(ub) if ub is not None else 1.0) - d.cdf(lb - 1)
    post = _draws(m, family)
    np.testing.assert_allclose(ab.log_lik(m, post).to_numpy()[0, 0], d.logpmf(FRAME[response].to_numpy()) - np.log(mass), rtol=1e-12)
    ks = np.arange(lb, (ub if ub is not None else 600) + 1)[:, None]
    mean = (ks * d.pmf(ks)).sum(0) / d.pmf(ks).sum(0)
    np.testing.assert_allclose(m.predict(post, kind="mean")["mu"].to_numpy()[0, 0], mean, rtol=1e-12)


@pytest.mark.parametrize("term,mean", [
    ("trunc(lb = 1)", "the truncated mean E[Y | Y >= 1] (brms's posterior_epred is E[Y | Y >= 2] instead: divergence D121)"),
    ("trunc(lb = 1, ub = 8)", "the truncated mean E[Y | 1 <= Y <= 8] (brms's posterior_epred is E[Y | 2 <= Y <= 8] instead: "
                              "divergence D121)"),
    ("trunc(lb = 0)", "the truncated mean E[Y | Y >= 0] (brms's posterior_epred is E[Y | Y >= 1] instead: divergence D121)"),
    ("trunc(lb = -2)", "the truncated mean"),
    ("trunc(ub = 8)", "the truncated mean"),
])
def test_describe_names_the_mean_of_a_lower_bound_and_brms_difference(term, mean):
    """describe()'s truncation line: the counts' range (a bound below 0 keeps the whole support, from 0) and, for a lower bound
    at or above 0, the mean predict() gives and brms's, which moves the bound up by one (D121)."""
    lb = {"trunc(lb = 1)": "1", "trunc(lb = 1, ub = 8)": "1"}.get(term, "0")
    ub = "8" if "ub = 8" in term else "inf"
    line = (f"  truncation: {term}: the counts lie in {{{lb}, ..., {ub}}}, both bounds included; each row's mass is divided by the "
            f"probability of that range (brms trunc()); predict(kind='mean') is {mean} and kind='response' draws from the truncated "
            "distribution, as brms's posterior_predict does")
    assert line in _model("negbinomial", term).describe().splitlines()


# -- the stacking weights (D126) -----------------------------------------------------------------------------------------
def _loo(ll):
    log_lik = xr.DataArray(ll[None], dims=("chain", "draw", "__obs__"))
    return loo_from_log_lik(log_lik, r_eff=1.0)


def test_stacking_is_offered_exported_and_the_default(monkeypatch):
    rng = np.random.default_rng(4)
    results = {"a": _loo(rng.normal(-1.0, 0.4, (30, 20))), "b": _loo(rng.normal(-1.1, 0.4, (30, 20)))}
    assert dict(post_processing.PARKED_WEIGHTS) == {}
    assert "loo_model_weights" in ab.__all__ and ab.loo_model_weights is post_processing.loo_model_weights
    w = ab.loo_model_weights(results)
    lpd = np.column_stack([r.pointwise["elpd_loo"].to_numpy() for r in results.values()])
    assert w.name == "stacking" and list(w.index) == ["a", "b"] and np.array_equal(w.to_numpy(), stacking_weights(lpd))
    assert ab.model_weights(results).equals(w) and ab.model_weights(results, "stacking").equals(w)
    monkeypatch.setattr(post_processing, "PARKED_WEIGHTS", MappingProxyType({"stacking": "held back (a reason)"}))
    with pytest.raises(ValueError, match=r"model_weights\(weights='stacking'\) is withheld: held back \(a reason\)"):
        ab.model_weights(results)
