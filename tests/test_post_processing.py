"""C07 post-processing (docs/post-processing.md): ``log_lik``, ``bayes_R2`` and ``loo`` beyond the oracle gate, which compares
them with brms on the frozen cases ``gauss_varint_post``, ``poisson_varint_post`` and ``gauss_varint_post_psis``.

- ``log_lik`` at the frozen brms points is the model's own pointwise log density (the per-row terms of the likelihood whose
  sum gate G2 checks against brms's ``log_prob``), on one case of every family and of each feature (weights, distributional
  formulas, monotonic terms, offsets, correlated and nested group terms) and on the three mixture cases (C05);
- PSIS-LOO and the relative efficiencies against loo 2.9.0's outputs on the deterministic inputs of ``tests/loo_probes``
  (ties at the Pareto tail's cutoff, tails whose fit does not exist, log ratios beyond the double range, ``r_eff`` given,
  NA and computed), and the PSIS steps against ArviZ's ``psislw`` where the two take the same tail;
- the API: weights multiply ``log_lik``, ``bayes_R2`` of an ordinal model and its refusal for the categorical family, the
  ``Loo`` summary, the refusals.
No sampler runs here: every posterior is a set of frozen or synthetic draws.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import arviz as az
import numpy as np
import pytest
import xarray as xr

import autobayes as ab
from autobayes.post_processing import POINTWISE_COLUMNS, gpd_fit, gpd_quantile, loo_from_log_lik, psis, relative_eff
from tests.loo_probes.inputs import FINGERPRINTS, fingerprint, loo_inputs, relative_eff_inputs
from tests.oracle import adapter
from tests.oracle import fixture as fx
from tests.oracle import gates
from tests.oracle.tolerances import PRED_REL, rel_err

RECORD = json.loads((Path(__file__).parent / "loo_probes" / "loo-2.9.0.json").read_text())
LOO_INPUTS = loo_inputs()
RELATIVE_EFF_INPUTS = relative_eff_inputs()
INPUT_PRINTS = json.loads(FINGERPRINTS.read_text())   # platform-independent: the inputs rounded to 8 decimals

# One frozen case per family and per feature the log-likelihood has to carry.
CROSS_CHECK = ["gauss_weighted", "gauss_distributional_sigma", "gauss_slope_correlated", "gauss_nested_colon", "student_learned_nu",
               "poisson_offset", "negbinomial_distributional_shape", "gamma_weighted", "lognormal_varint", "beta_distributional_phi",
               "bernoulli_probit", "bernoulli_weighted", "binomial_monotonic", "binomial_probit", "zero_inflated_poisson_varint",
               "hurdle_poisson_varint", "cumulative_monotonic", "categorical_poponly_categorical", "mixture_gauss_gauss",
               "mixture_gauss3_categorical", "mixture_gauss_default_priors",
               # D01: the Weibull, exponential, skew-normal, shifted lognormal and ex-Gaussian families
               "weibull_varint", "exponential_varint", "skew_normal_varint", "shifted_lognormal_varint", "exgaussian_varint",
               # D02: the beta-binomial, geometric and negbinomial2 families
               "beta_binomial_varint", "geometric_varint", "negbinomial2_varint",
               # D04: the sratio, cratio and acat families, with category-specific effects cs()
               "sratio_varint", "cratio_varint", "acat_varint", "acat_cs", "sratio_cs", "cratio_cs",
               # D03: six more zero-inflated and hurdle families, and weights() for B06's two
               "zero_inflated_negbinomial_varint", "hurdle_negbinomial_varint", "zero_inflated_binomial_varint",
               "zero_inflated_beta_varint", "hurdle_gamma_varint", "hurdle_lognormal_varint", "zero_inflated_poisson_weighted",
               "hurdle_poisson_weighted",
               # E06: the Poisson sqrt, negative binomial identity, Gamma inverse and identity, Beta probit and cloglog links
               "poisson_sqrt", "negbinomial_identity", "gamma_inverse", "gamma_identity", "beta_probit", "beta_cloglog"]


def _bundle(case_id):
    if not fx.has_fixture(case_id) or fx.load_case(case_id).get("gate"):
        pytest.skip(f"{case_id} is not frozen or still gated")
    return gates.Bundle(case_id)


def _same(ours, theirs, tol=PRED_REL):
    """Equal to ``tol`` where finite, and the same non-finite values elsewhere."""
    ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    assert ours.shape == theirs.shape
    assert np.array_equal(np.isfinite(ours), np.isfinite(theirs))
    assert np.array_equal(ours[~np.isfinite(ours)], theirs[~np.isfinite(theirs)], equal_nan=True)
    finite = np.isfinite(ours)
    assert rel_err(ours[finite], theirs[finite]) <= tol


def _as_log_lik(ll):
    """A (draws, rows) matrix as a one-chain log-likelihood."""
    return xr.DataArray(np.asarray(ll)[None], dims=("chain", "draw", "__obs__"))


@pytest.mark.parametrize("case_id", CROSS_CHECK)
def test_log_lik_is_the_models_own_pointwise_log_density(case_id):
    """At every frozen point, ``log_lik`` row by row equals the PyMC model's elementwise log density of the response (with
    the weights, for a weighted case), whose sum G2 checks against brms's log_prob."""
    b = _bundle(case_id)
    pmm = b.model.pymc_model
    pointwise = pmm.compile_fn(pmm.logp(vars=pmm.observed_RVs, sum=False), inputs=pmm.value_vars, point_fn=True,
                               on_unused_input="ignore")
    ll = ab.log_lik(b.model, adapter.draws(b.fixture, b.model, b.cmap, b.gmap))
    assert ll.dims == ("chain", "draw", "__obs__") and ll.shape == (1, len(b.fixture.points), b.model.design.n)
    for k, _, vp in b.points():
        assert rel_err(ll.to_numpy()[0, k], np.asarray(pointwise(vp)[0], dtype=float).ravel()) <= 1e-12, k


def test_weights_multiply_the_log_likelihood():
    weighted = _bundle("gauss_weighted")
    pr = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
    plain = ab.gaussian("y ~ x + (1 | g)", weighted.data, pr)
    idata = adapter.draws(weighted.fixture, weighted.model, weighted.cmap, weighted.gmap)
    w = weighted.model.design.weights
    assert w is not None and not np.allclose(w, 1.0)
    np.testing.assert_allclose(ab.log_lik(weighted.model, idata).to_numpy(), w * ab.log_lik(plain, idata).to_numpy(), rtol=1e-14)


@pytest.mark.parametrize("probe", RECORD["loo_probes"], ids=lambda p: p["label"])
def test_psis_loo_equals_loo_2_9_0_on_the_probe_inputs(probe):
    """Every pointwise column, the estimates, k-hat and n_eff equal loo 2.9.0's on the recorded probe; where loo's MCSE is
    NaN because ``exp(elpd_loo_i)^2`` underflows (``elpd_loo_i`` below about -372), this one is finite (divergence D50)."""
    ll, r_eff = LOO_INPUTS[probe["label"]]
    assert [fingerprint(ll), fingerprint(r_eff)] == INPUT_PRINTS["loo"][probe["label"]], "the probe inputs changed: re-record them"
    res = loo_from_log_lik(_as_log_lik(ll), r_eff=r_eff)
    assert list(res.pointwise.columns) == probe["columns"] == list(POINTWISE_COLUMNS)
    assert list(res.estimates.index) == probe["rows"]
    table = fx.decode(probe["pointwise"])
    for j, column in enumerate(probe["columns"]):
        ours, theirs = res.pointwise[column].to_numpy(), table[:, j]
        if column == "mcse_elpd_loo" and np.isnan(theirs).any():          # D50
            under = np.isnan(theirs)
            assert np.isfinite(ours).all() and (res.pointwise["elpd_loo"].to_numpy()[under] < -350).all()
            ours, theirs = ours[~under], theirs[~under]
        _same(ours, theirs)
    _same(res.estimates.to_numpy(), fx.decode(probe["estimates"]))
    _same(res.pareto_k, fx.decode(probe["pareto_k"]))
    _same(res.n_eff, fx.decode(probe["n_eff"]))


def test_the_probes_reach_the_cases_arviz_does_not_take_like_loo():
    """Ties at the tail's cutoff and tails whose fit does not exist are in the record: loo's k-hat is Inf (raw ratios) on
    some tied columns and finite on others, and ArviZ's psislw, which keeps only the draws strictly above the cutoff, gives
    a different k-hat on at least one column that loo fits."""
    (ties,) = [p for p in RECORD["loo_probes"] if p["label"] == "ties S=100"]
    k = fx.decode(ties["pareto_k"])
    assert np.isinf(k).any() and np.isfinite(k).any()
    ll, _ = LOO_INPUTS["ties S=100"]
    with np.errstate(all="ignore"):
        k_arviz = np.array([az.psislw(-ll[:, i].copy(), 1.0)[1] for i in range(ll.shape[1])])
    fitted = np.isfinite(k)
    assert not np.allclose(k_arviz[fitted], k[fitted], rtol=0, atol=1e-9)


@pytest.mark.parametrize("probe", RECORD["relative_eff_probes"], ids=lambda p: p["label"])
def test_relative_eff_equals_loo_2_9_0(probe):
    lik = RELATIVE_EFF_INPUTS[probe["label"]]
    assert fingerprint(lik) == INPUT_PRINTS["relative_eff"][probe["label"]], "the probe inputs changed: re-record them"
    with np.errstate(divide="ignore"):
        log_lik = xr.DataArray(np.log(lik), dims=("chain", "draw", "__obs__"))
    _same(relative_eff(log_lik), fx.decode(probe["r_eff"]))


def test_short_chains_have_relative_efficiency_one():
    """Chains of fewer than four draws (brms's fixed-parameter fits have one each) have no relative efficiency: loo's NA,
    which it replaces by 1."""
    rng = np.random.default_rng(3)
    for draws in (1, 2, 3):
        ll = xr.DataArray(rng.normal(size=(5, draws, 4)), dims=("chain", "draw", "__obs__"))
        assert (relative_eff(ll) == 1.0).all()
    constant = xr.DataArray(np.full((2, 50, 3), -1.0), dims=("chain", "draw", "__obs__"))
    assert (relative_eff(constant) == 1.0).all()


@pytest.mark.parametrize("S, r_eff", [(30, 1.0), (100, 1.0), (100, 0.4), (1000, 1.0), (4000, 0.8)])
def test_psis_equals_arviz_where_the_tails_coincide(S, r_eff):
    """Without ties, ArviZ's psislw takes the same tail: the smoothed log weights and k-hat agree to rounding."""
    rng = np.random.default_rng(S)
    for scale in (0.3, 1.0, 3.0):
        x = rng.normal(0.0, scale, S)
        lw, k = psis(x, r_eff)
        lw_arviz, k_arviz = az.psislw(x.copy(), r_eff)
        assert abs(k - k_arviz) <= 1e-12 * max(1.0, abs(k))
        np.testing.assert_allclose(lw, lw_arviz, rtol=0, atol=1e-12)


def test_gpd_fit_equals_arviz_where_the_fit_exists():
    """The generalized Pareto fit has the steps of ArviZ's ``_gpdfit``: equal to rounding on tails that have a fit."""
    from arviz.stats.stats import _gpdfit

    rng = np.random.default_rng(5)
    for n in (5, 13, 50, 190, 800):
        for scale in (0.1, 1.0, 10.0):
            x = np.sort(rng.pareto(1.5, n) * scale)
            np.testing.assert_allclose(gpd_fit(x), _gpdfit(x), rtol=1e-12)


def test_psis_leaves_short_or_degenerate_tails_raw():
    x = np.random.default_rng(0).normal(size=20)                           # tail of ceil(min(4, 13.4)) = 4 draws
    lw, k = psis(x)
    assert k == math.inf
    np.testing.assert_allclose(lw, x - np.log(np.exp(x - x.max()).sum()) - x.max(), rtol=0, atol=1e-14)
    lw, k = psis(np.zeros(100))                                             # every tail draw ties with the cutoff
    assert k == math.inf
    np.testing.assert_allclose(lw, np.full(100, -math.log(100.0)), rtol=0, atol=1e-15)
    assert all(math.isnan(v) for v in gpd_fit(np.r_[np.zeros(10), np.linspace(0.1, 1.0, 10)]))   # quartile 0: no fit
    np.testing.assert_allclose(gpd_quantile(np.array([0.5]), 0.0, 2.0), [2.0 * math.log(2.0)])
    np.testing.assert_allclose(gpd_quantile(np.array([0.5]), 0.5, 2.0), [2.0 * (2.0 ** 0.5 - 1.0) / 0.5])


def test_bayes_R2_of_ordinal_and_categorical_models():
    """brms's bayes_R2 of a cumulative model is that of the expected category ``sum_k k P(y = k)`` against the category (the
    frozen case cumulative_varint_post gates it); brms refuses it for the categorical family, and so does this one."""
    b = _bundle("cumulative_varint")
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    p = b.model.predict(idata, kind="mean")["p"].transpose("chain", "draw", "__obs__", "category").to_numpy()
    yhat = (p * np.arange(1, p.shape[-1] + 1)).sum(-1)
    e = b.model.design.y - yhat
    expected = np.var(yhat, axis=-1, ddof=1) / (np.var(yhat, axis=-1, ddof=1) + np.var(e, axis=-1, ddof=1))
    np.testing.assert_allclose(ab.bayes_R2(b.model, idata).to_numpy(), expected, rtol=1e-14)
    c = _bundle("categorical_poponly")
    with pytest.raises(ValueError, match="not defined for unordered categorical models"):
        ab.bayes_R2(c.model, adapter.draws(c.fixture, c.model, c.cmap, c.gmap))


def test_loo_result_and_refusals():
    b = _bundle("gauss_varint_post_psis")
    idata = gates.fixed_parameter_draws(b)
    res = ab.loo(b.model, idata)
    assert isinstance(res, ab.Loo) and res.n_draws == 64 and len(res.pointwise) == b.model.design.n
    assert res.elpd_loo == pytest.approx(res.pointwise["elpd_loo"].sum())
    assert res.looic == pytest.approx(-2 * res.elpd_loo) and res.p_loo == pytest.approx(res.pointwise["p_loo"].sum())
    assert res.se_elpd_loo > 0 and (res.r_eff == 1.0).all()
    assert res.pareto_k_threshold == pytest.approx(1 - 1 / math.log10(64))
    assert "PSIS-LOO of 36 observations from 64 posterior draws" in str(res)
    same = ab.loo(b.model, idata, r_eff=1.0)
    _same(same.pointwise.to_numpy(), res.pointwise.to_numpy(), tol=0.0)
    halved = ab.loo(b.model, idata, r_eff=0.5)                # the same 13-draw tail (S / 5 < 3 sqrt(2 S)): n_eff halves
    np.testing.assert_allclose(halved.n_eff, 0.5 * res.n_eff, rtol=1e-12)
    np.testing.assert_allclose(halved.pointwise["elpd_loo"], res.pointwise["elpd_loo"], rtol=0, atol=0)
    assert (halved.pointwise["mcse_elpd_loo"] > res.pointwise["mcse_elpd_loo"]).all()
    for bad in (0.0, -1.0, np.inf, [1.0, 2.0]):
        with pytest.raises(ValueError):
            ab.loo(b.model, idata, r_eff=bad)
    with pytest.raises(ValueError, match="finite log-likelihood"):
        loo_from_log_lik(_as_log_lik(np.array([[0.0, -np.inf], [-1.0, -2.0]])), r_eff=1.0)
