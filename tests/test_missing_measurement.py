"""brms's measurement-error term ``me(x, sdx)`` and missing-value term ``mi(col)`` with the response's ``| mi()`` (brms-parity
feature C06; docs/missing-and-measurement-error.md).

The grammar and its refusals, the term model, the priors (``meanme``, ``sdme``, the coefficients' ``b``), the graphs against
densities written out here with scipy (the latent values and the noisy values' density of ``me()``; the observed and imputed
values of ``mi()``, one variable of the missing values shared by the responses), ``value_point``, predictions on fitted and new
rows, brms's default priors and prior table, ``describe()``, the target identity, and the refusals of what is not part of the
slice. Nothing samples. The numbers against brms are the oracle's (tests/oracle, cases gauss_me and gauss_mi)."""
import re

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import norm

import autobayes as ab
from autobayes.formula import FormulaError, parse_formula


def me_frame(n=24, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.5, 1.2, n)
    sdx = rng.uniform(0.2, 0.6, n)
    return pd.DataFrame({"y": 0.3 + 0.8 * latent + rng.normal(0, 0.5, n), "xobs": latent + rng.normal(0, sdx), "sdx": sdx,
                         "w": rng.normal(size=n), "g": ["a", "b", "c"] * (n // 3)})


def mi_frame(n=24, seed=4, missing=(0, 5, 6, 17)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, n)
    xm = 0.4 * w + rng.normal(0, 0.6, n)
    y = -0.5 + 0.2 * w - 0.6 * xm + rng.normal(0, 0.4, n)
    xm[list(missing)] = np.nan
    return pd.DataFrame({"y": y, "xm": xm, "w": w, "v": rng.normal(size=n), "g": ["a", "b", "c", "d"] * (n // 4)})


ME = "y ~ me(xobs, sdx)"
MI = ["y ~ mi(xm) + w", "xm | mi() ~ w"]


def me_priors(**changes):
    return ab.Priors(**{"intercept": ab.Normal(0, 5), "b": ab.Normal(0, 2), "sigma": ab.HalfNormal(1), "meanme": ab.Normal(0, 2),
                        "sdme": ab.HalfNormal(1), **changes})


def response(sigma=1.0):
    return {"intercept": ab.Normal(0, 5), "b": ab.Normal(0, 2), "sigma": ab.HalfNormal(sigma)}


def mi_priors(responses=("y", "xm")):
    return ab.Priors(resp={r: response(1.0 + k) for k, r in enumerate(responses)})


@pytest.fixture(scope="module")
def me_data():
    return me_frame()


@pytest.fixture(scope="module")
def mi_data():
    return mi_frame()


@pytest.fixture(scope="module")
def me_model(me_data):
    return ab.gaussian(ME, me_data, me_priors())


@pytest.fixture(scope="module")
def mi_model(mi_data):
    return ab.gaussian(MI, mi_data, mi_priors(), rescor=False)


# -- 1. the grammar and the term model ------------------------------------------------------------------------------
def test_terms(me_data, mi_data):
    s = parse_formula("y ~ me( xobs ,sdx) + w", me_data)
    (t, w) = s.pop_terms
    assert (t.name, t.kind, t.operands, t.columns, t.is_special) == ("me(xobs, sdx)", "measurement_error", ("xobs", "sdx"), (), True)
    assert s.terms == ("me(xobs, sdx)", "w") and s.columns == ("w",) and s.coefficient_labels == ("me(xobs, sdx)", "w")
    assert s.me_terms == (t,) and s.bambi_formula == "y ~ w" and s.canonical_formula == "y ~ me(xobs, sdx) + w"
    assert s.column_terms == (("w", "w", None),) and s.missing_columns == ()
    s = parse_formula("y ~ mi(xm) + w", mi_data)
    assert s.mi_terms[0].kind == "missing" and s.mi_terms[0].operands == ("xm",) and s.bambi_formula == "y ~ w"
    assert s.missing_columns == ("xm",) and s.coefficient_labels == ("mi(xm)", "w")
    s = parse_formula("xm | mi() ~ w", mi_data)
    assert [a.term for a in s.additions] == ["mi()"] and s.canonical_formula == "xm | mi() ~ w" and s.missing_columns == ("xm",)
    assert s.bambi_formula == "xm ~ w"


@pytest.mark.parametrize("formula, message", [
    ("y ~ me(x, sdx, gr = g)", "brms's gr argument (me(x, sdx, gr = g)"),
    ("y ~ me(x)", "me() takes two column names"),
    ("y ~ me(log(x), sdx)", "me() takes column names here, not expressions"),
    ("y ~ me(x, x)", "the noisy column and the column of its standard deviations must differ"),
    ("y ~ me(y, sdx)", "the response 'y' cannot be read by me()"),
    ("y ~ mi(x, idx)", "brms's idx argument (mi(x, idx)"),
    ("y ~ mi(log(x))", "mi() takes a column name here"),
    ("y ~ mi(y)", "the response 'y' cannot be its own mi() predictor"),
    ("y ~ w * mi(x)", "an interaction with a missing-value term ('mi(x)') is not supported"),
    ("y ~ me(x, s):w", "an interaction with a measurement-error term ('me(x, s)') is not supported"),
    ("y ~ (1 + mi(x) | g)", "a missing-value term cannot vary by 'g'"),
    ("y ~ (1 + me(x, s) || g)", "a measurement-error term cannot vary by 'g'"),
    ("y | mi(s) ~ x", "brms's mi(sdy), a response measured with known noise, is not part of this slice"),
    ("y | mi() + weights(v) ~ x", "mi() together with weights(v) on the left-hand side is not verified"),
    ("y ~ me(x, s) + me(w, s2)", "several measurement-error terms"),
    ("y ~ mo(o) + mi(x)", "mo() and mi() terms in one formula are not supported"),
    ("y ~ me(x, s) + mi(w)", "me() and mi() terms in one formula are not supported"),
    ("y | mi() ~ me(x, s)", "a measurement-error term in a formula whose response has mi()"),
    ("y ~ mi(x) + x", "'x' is read by 'mi(x)' and by another term of the formula"),
    ("y ~ me(x, s) + log(s)", "'s' is read by 'me(x, s)' and by another term"),
    ("y ~ mi(x) + (1 | x)", "the grouping factor cannot also be a predictor"),
])
def test_grammar_refusals(formula, message):
    with pytest.raises(FormulaError, match=re.escape(message)):
        parse_formula(formula)


@pytest.mark.parametrize("family, formula, message", [
    ("student", "y ~ me(x, s)", "the measurement-error term 'me(x, s)' is not available for the student family; brms's me() is "
                                "verified here for the gaussian family"),
    ("poisson", "y ~ mi(x)", "the missing-value term 'mi(x)' is not available for the poisson family"),
    ("bernoulli", "y | mi() ~ x", "'mi()' is not available for the bernoulli family; brms's mi() is verified here for the gaussian"),
])
def test_other_families_refused(family, formula, message):
    with pytest.raises(FormulaError, match=re.escape(message)):
        parse_formula(formula, family=family)


def test_categorical_declaration_refused(me_data):
    with pytest.raises(FormulaError, match=re.escape("'me(xobs, sdx)': 'xobs' is declared categorical; me() reads numbers")):
        parse_formula("y ~ me(xobs, sdx)", categorical={"xobs": 0.5})


def test_distributional_formulas_refused(me_data, mi_data):
    with pytest.raises(FormulaError, match=re.escape("a measurement-error term in a distributional formula (me(xobs, sdx))")):
        ab.gaussian(["y ~ w", "sigma ~ me(xobs, sdx)"], me_data, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2),
                                                                          sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1)}))
    with pytest.raises(FormulaError, match=re.escape("a list of formulas with mi() is brms's multivariate model")):
        ab.student(MI, mi_data, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)), nu=4)


@pytest.mark.parametrize("change, message", [
    (lambda d: d.assign(sdx=d["sdx"].where(d.index != 3, 0.0)), "the known standard deviations of 'me(xobs, sdx)' must be positive; row 3"),
    (lambda d: d.assign(xobs=d["xobs"].where(d.index != 2, np.nan)), "column 'xobs' (noisy values of 'me(xobs, sdx)') has missing or non-finite"),
    (lambda d: d.assign(sdx=d["sdx"].astype(str)), "column 'sdx' (known standard deviations of 'me(xobs, sdx)') has dtype"),
])
def test_me_data_refusals(me_data, change, message):
    with pytest.raises(FormulaError, match=re.escape(message)):
        ab.gaussian(ME, change(me_data), me_priors())


@pytest.mark.parametrize("change, message", [
    (lambda d: d.assign(xm=np.nan), "column 'xm' has no observed value"),
    (lambda d: d.assign(xm=d["xm"].where(d.index != 3, np.inf)), "column 'xm' has an infinite value at row 3"),
    (lambda d: d.assign(xm=d["xm"].astype(str)), "a column with missing values modelled by mi() must be numeric"),
    (lambda d: d.assign(w=d["w"].where(d.index != 2, np.nan)), "missing values in columns ['w']"),
])
def test_mi_data_refusals(mi_data, change, message):
    with pytest.raises(FormulaError, match=re.escape(message)):
        ab.gaussian(MI, change(mi_data), mi_priors(), rescor=False)


# -- 2. the scope of mi(): a multivariate list of formulas ------------------------------------------------------------
def test_mi_needs_its_formula(mi_data):
    with pytest.raises(FormulaError, match=re.escape("'mi(xm)' reads a column whose missing values are modelled by a formula of its own")
                       + ".*rescor=False.*divergence D72"):
        ab.gaussian("y ~ mi(xm) + w", mi_data, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)))
    with pytest.raises(FormulaError, match=re.escape("a univariate y | mi() ~ ... model is not part of this slice")):
        ab.gaussian("xm | mi() ~ w", mi_data, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)))
    with pytest.raises(FormulaError, match=re.escape("'mi(xm)' reads a column whose missing values are modelled")):
        ab.default_priors("y ~ mi(xm) + w", mi_data)


@pytest.mark.parametrize("formulas, rescor, error, message", [
    (MI, True, FormulaError, "mi() with a residual correlation (rescor=True, brms's set_rescor(TRUE)) is not supported"),
    (MI, None, ValueError, "a model with mi() takes rescor=False"),
    (["y ~ mi(xm) + w", "xm ~ w"], False, FormulaError, "whose formula must say that its missing values are parameters: write it 'xm | mi() ~ ...'"),
    (["y ~ mi(v) + w", "xm | mi() ~ w"], False, FormulaError, "'mi(v)' reads 'v', which no formula of the list models"),
    (["y ~ mi(xm) + w", "xm | mi() ~ mi(v)", "v | mi() ~ mi(xm)"], False, FormulaError,
     "the imputations form a cycle (xm -> v -> xm)"),
    (["y ~ mi(xm) + w", "xm | mi() ~ y"], False, FormulaError, "the response(s) ['y'] of another formula are read here other than through mi()"),
    (["y ~ mi(xm) + me(v, w)", "xm | mi() ~ w"], False, FormulaError, "me() and mi() terms in one formula are not supported"),
    (["y ~ mi(xm) + w", "xm | mi() ~ me(v, w)"], False, FormulaError, "a measurement-error term in a formula whose response has mi()"),
    (["y ~ mi(xm) + w", "xm | mi() ~ w", "y ~ v"], False, FormulaError, "the response(s) ['y'] have more than one formula"),
    (["mvbind(y, v) ~ mi(xm)", "xm | mi() ~ w"], False, FormulaError, "(docs/multivariate.md, divergence D56)"),   # C01's refusal
    (["y ~ mi(xm) + w", "xm | mi() + weights(w) ~ w"], False, FormulaError, "mi() together with weights(w) on the left-hand side"),
])
def test_multivariate_refusals(mi_data, formulas, rescor, error, message):
    responses = [re.match(r"\s*(\w+)", f).group(1) for f in formulas]
    with pytest.raises(error, match=re.escape(message)):
        ab.gaussian(formulas, mi_data, mi_priors(tuple(dict.fromkeys(responses))), rescor=rescor)


def test_me_refused_in_multivariate(me_data):
    with pytest.raises(FormulaError, match=re.escape("a measurement-error term in a multivariate model is not supported")):
        ab.gaussian("mvbind(y, w) ~ me(xobs, sdx)", me_data, mi_priors(("y", "w")), rescor=False)


def test_a_formula_per_response_without_mi_stays_refused(mi_data):
    with pytest.raises(FormulaError, match=re.escape("a formula of its own per response (brms's bf(y ~ ...) + bf(y2 ~ ...)) is not "
                                                     "supported in this slice") + ".*divergence D56"):
        ab.gaussian(["y ~ w", "v ~ w"], mi_data, mi_priors(("y", "v")), rescor=False)


# -- 3. me(): the graph, its density and its points -----------------------------------------------------------------------
def test_me_graph(me_model, me_data):
    m = me_model
    assert m.me_names == {"xobs": {"b": "me(xobs, sdx)", "mean": "meanme_xobs", "sd": "sdme_xobs", "z": "zme_xobs",
                                   "latent": "Xme_xobs", "noisy": "Xn_xobs"}}
    assert {v.name for v in m.pymc_model.value_vars} == {"Intercept", "sigma_log__", "me(xobs, sdx)", "meanme_xobs", "sdme_xobs_log__",
                                                         "zme_xobs"}
    assert sorted(rv.name for rv in m.pymc_model.observed_RVs) == ["Xn_xobs", "y"]
    block = m.design.measurement_error["xobs"]
    assert np.array_equal(block.Xn, me_data["xobs"].to_numpy()) and np.array_equal(block.noise, me_data["sdx"].to_numpy())
    assert m.design.X_raw.shape == (len(me_data), 0) and m.names["Xme:xobs"] == "Xme_xobs" and m.names["b:me(xobs, sdx)"] == "me(xobs, sdx)"


def me_point(model, seed=0):
    rng = np.random.default_rng(seed)
    nat = dict(intercept=rng.normal(), b=[], sigma=float(rng.uniform(0.5, 1.5)), bsp=float(rng.normal()), meanme=float(rng.normal()),
               sdme=float(rng.uniform(0.5, 1.5)), zme=rng.normal(size=model.design.n))
    return model.value_point(**nat), nat


def test_me_density(me_model, me_data):
    m = me_model
    vp, nat = me_point(m)
    latent = nat["meanme"] + nat["sdme"] * nat["zme"]
    mu = nat["intercept"] + nat["bsp"] * latent
    lik = (norm.logpdf(me_data["y"], mu, nat["sigma"]).sum() + norm.logpdf(me_data["xobs"], latent, me_data["sdx"]).sum()
           + norm.logpdf(nat["zme"]).sum())
    prior = (norm.logpdf(nat["intercept"], 0, 5) + norm.logpdf(nat["bsp"], 0, 2) + norm.logpdf(nat["sigma"], 0, 1) + np.log(2)
             + norm.logpdf(nat["meanme"], 0, 2) + norm.logpdf(nat["sdme"], 0, 1) + np.log(2))
    jac = np.log(nat["sigma"]) + np.log(nat["sdme"])
    assert m.pymc_model.compile_logp(jacobian=False)(vp) == pytest.approx(lik + prior, rel=1e-12)
    assert m.pymc_model.compile_logp()(vp) == pytest.approx(lik + prior + jac, rel=1e-12)
    pmm = m.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["Xme_xobs"], pmm["mu"]]), inputs=pmm.value_vars, point_fn=True,
                        on_unused_input="ignore")
    got_latent, got_mu = fn(vp)
    assert np.allclose(got_latent, latent, rtol=1e-13) and np.allclose(got_mu, mu, rtol=1e-13)


@pytest.mark.parametrize("change, message", [
    (dict(meanme=None), "meanme is required, keyed by ['xobs']"),
    (dict(sdme=-1.0), "sdme for 'xobs' must be a finite positive scalar"),
    (dict(zme=np.zeros(3)), "zme for 'xobs' must have 24 finite entries"),
    (dict(bsp={"x": 0.1}), "bsp (the coefficient of each me() / mi() term) must cover exactly ['xobs']"),
    (dict(simo=[0.5, 0.5]), "this model has no monotonic term mo(x); bsp and simo do not apply"),
    (dict(Ymi=[0.1]), "this model holds no missing values; Ymi does not apply"),
])
def test_me_value_point_refusals(me_model, change, message):
    _, nat = me_point(me_model)
    with pytest.raises(ValueError, match=re.escape(message)):
        me_model.value_point(**{**nat, **change})


def test_me_value_point_keyed(me_model):
    _, nat = me_point(me_model)
    a = me_model.value_point(**nat)
    b = me_model.value_point(**{**nat, **{k: {"xobs": nat[k]} for k in ("bsp", "meanme", "sdme", "zme")}})
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    assert a["sdme_xobs_log__"] == pytest.approx(np.log(nat["sdme"]))


def test_me_priors(me_data):
    for priors, message in ((me_priors(meanme=None), "need a prior on the location of their latent values: pass meanme="),
                            (ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1), meanme=ab.HalfNormal(1),
                                       sdme=ab.HalfNormal(1)), "prior for 'meanme:xobs' must be a location prior"),
                            (ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1), meanme=ab.Normal(0, 1),
                                       sdme=ab.Normal(0, 1)), "prior for 'sdme:xobs' must be a positive-support prior or Flat()"),
                            (ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1), meanme={"w": ab.Normal(0, 1)},
                                       sdme=ab.HalfNormal(1)), "meanme must cover exactly the noisy columns of the measurement-error terms ['xobs']")):
        with pytest.raises(ab.PriorError, match=re.escape(message)):
            ab.gaussian(ME, me_data, priors)
    with pytest.raises(ab.PriorError, match=re.escape("the formula has no measurement-error term me(x, sdx); do not pass sdme")):
        ab.gaussian("y ~ w", me_data, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1), sdme=ab.HalfNormal(1)))
    keyed = ab.gaussian(ME, me_data, me_priors(b={"me(xobs, sdx)": ab.Normal(0, 2)}))
    assert keyed.target_identity == ab.gaussian(ME, me_data, me_priors()).target_identity


def test_me_default_priors_and_table(me_data):
    d = ab.default_priors(ME, me_data)
    assert d.meanme == {"xobs": ab.Flat()} and d.sdme == {"xobs": ab.Flat()} and d.b == ab.Flat()
    m = ab.gaussian(ME, me_data, d)
    rows = [(r["prior"], r["class"], r["coef"], r["lb"]) for r in m.prior_table()]
    assert rows[3:7] == [("", "meanme", "", ""), ("", "meanme", "mexobs", ""), ("", "sdme", "", "0"), ("", "sdme", "mexobs", "")]
    assert ("", "b", "mexobssdx", "") in rows
    assert m.default_prior_labels == ["intercept", "b:me(xobs, sdx)", "meanme:xobs", "sdme:xobs", "sigma"]
    assert m.priors.improper(m.spec) == ["b:me(xobs, sdx)", "meanme:xobs", "sdme:xobs"]
    # a flat sdme is flat on the positive half line, sampled on the log scale: its density is the Jacobian alone
    vp, nat = me_point(m)
    full, nojac = m.pymc_model.compile_logp()(vp), m.pymc_model.compile_logp(jacobian=False)(vp)
    assert full - nojac == pytest.approx(np.log(nat["sigma"]) + np.log(nat["sdme"]), rel=1e-12)
    with pytest.raises(ValueError, match=re.escape("the prior of ['b:me(xobs, sdx)', 'meanme:xobs', 'sdme:xobs'] is flat")):
        m.prior_predictive(draws=5)


def test_me_describe(me_model):
    text = me_model.describe()
    for line in ("AutoBayes M0 model with measurement-error term(s)", "mu_i = a_c + b_me(xobs, sdx) * Xme_xobs,i",
                 "Xme_xobs,i = meanme_xobs + sdme_xobs * zme_xobs,i", "xobs_i ~ Normal(Xme_xobs,i, sdx_i)",
                 "b_me(xobs, sdx) ~ Normal(mu=0, sigma=2)   (brms's bsp_mexobssdx: class b, coefficient mexobssdx)",
                 "meanme_xobs ~ Normal(mu=0, sigma=2)   (brms class meanme, coefficient mexobs)", "sdme_xobs_log__",
                 "zme_xobs                 zme_xobs[0..23]", "Xme_xobs = meanme_xobs + sdme_xobs * zme_xobs"):
        assert line in text, line


def draws(model, points):
    """Posterior draws of ``model`` (one chain) from natural-scale points, under the graph's names, as a fit holds them."""
    data = {}
    for k, v in points[0].items():
        arr = np.stack([np.asarray(p[k], dtype=float) for p in points])[None]
        dims = ("chain", "draw") + (("__obs__",) if arr.ndim == 3 else ())
        data[k] = (dims, arr)
    return az.InferenceData(posterior=xr.Dataset(data))


def test_me_predictions(me_model, me_data):
    m = me_model
    pts = [me_point(m, s)[1] for s in range(3)]
    post = draws(m, [{"Intercept": p["intercept"], "sigma": p["sigma"], "me(xobs, sdx)": p["bsp"], "meanme_xobs": p["meanme"],
                      "sdme_xobs": p["sdme"], "zme_xobs": p["zme"]} for p in pts])
    eta = m.predict(post, kind="linear")["eta"].to_numpy()[0]
    want = np.stack([p["intercept"] + p["bsp"] * (p["meanme"] + p["sdme"] * p["zme"]) for p in pts])
    assert np.allclose(eta, want, rtol=1e-13)
    mean = m.predict(post, kind="mean")["mu"].to_numpy()[0]
    assert np.allclose(mean, want, rtol=1e-13)
    with pytest.raises(FormulaError, match=re.escape("predict(newdata=...) of a model with 'me(xobs, sdx)' is not offered: brms draws "
                                                     "the latent values of new rows")):
        m.predict(post, newdata=me_data.head(3), kind="linear")
    with pytest.raises(ValueError, match=re.escape("log_lik() of a model with me() or mi() is not offered")):
        ab.log_lik(m, post)
    with pytest.raises(ValueError, match=re.escape("bayes_R2() of a model with me() or mi() is not offered")):
        ab.bayes_R2(m, post)


def test_me_prior_predictive(me_model):
    pp = me_model.prior_predictive(draws=7, seed=2)
    assert pp.prior_predictive["y"].shape == (1, 7, me_model.design.n) and "Xn_xobs" in pp.prior_predictive
    assert np.isfinite(pp.prior["Xme_xobs"]).all()


def test_me_identity(me_model, me_data):
    other = me_data.assign(sdx=me_data["sdx"] * 1.01)
    assert ab.gaussian(ME, other, me_priors()).target_identity != me_model.target_identity
    again = ab.gaussian(ME, me_data, me_priors())
    assert again.target_identity == me_model.target_identity
    changed = ab.gaussian(ME, me_data, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1),
                                                 meanme=ab.Normal(0, 3), sdme=ab.HalfNormal(1)))
    assert changed.target_identity != me_model.target_identity


# -- 4. mi(): the graph, its density and its points ------------------------------------------------------------------------
MISSING = [0, 5, 6, 17]


def test_mi_graph(mi_model, mi_data):
    m = mi_model
    assert m.spec.separate and m.spec.imputed == ("xm",) and m.responses == ("y", "xm")
    assert m.spec.canonical_formula == "bf(y ~ mi(xm) + w) + bf(xm | mi() ~ w)"
    assert m.missing_names == {"xm": {"values": "Ymi_xm", "coordinate": "Ymi_xm_dim", "response": "xm"}}
    for sub in m.submodels.values():                     # each response's model holds the missing values it reads
        assert sub.missing_names == {"xm": {"values": "Ymi_xm", "coordinate": "Ymi_xm_dim"}}
        assert sub.design.missing["xm"].index.tolist() == MISSING
    names = [v.name for v in m.pymc_model.value_vars]
    assert names.count("Ymi_xm") == 1 and "y_Ymi_xm" not in names and "xm_Ymi_xm" not in names
    assert set(names) == {"y_sigma_log__", "y_Intercept", "y_w", "y_mi(xm)", "Ymi_xm", "xm_sigma_log__", "xm_Intercept", "xm_w"}
    assert [int(c) for c in m.pymc_model.coords["Ymi_xm_dim"]] == MISSING
    assert m.response_names["y"]["b:mi(xm)"] == "y_mi(xm)" and np.isnan(m.design.Y[MISSING, 1]).all()
    held = np.asarray(m.pymc_model.rvs_to_values[m.pymc_model["xm"]].data)
    assert np.array_equal(held[MISSING], np.zeros(4))                      # Bambi's placeholder, never read (below)


def mi_point(model, seed=0):
    rng = np.random.default_rng(seed)
    resp = {r: {"intercept": rng.normal(), "b": rng.normal(size=len(sub.spec.columns)), "sigma": float(rng.uniform(0.5, 1.5)),
                **({"bsp": float(rng.normal())} if sub.spec.mi_terms else {})} for r, sub in model.submodels.items()}
    ymi = rng.normal(size=len(MISSING))
    return model.value_point(resp=resp, Ymi={"xm": ymi}), resp, ymi


def test_mi_density(mi_model, mi_data):
    m = mi_model
    vp, resp, ymi = mi_point(m)
    xm = mi_data["xm"].to_numpy().copy()
    xm[MISSING] = ymi
    wc = mi_data["w"] - mi_data["w"].mean()
    ry, rx = resp["y"], resp["xm"]
    mu_y = ry["intercept"] + wc * ry["b"][0] + ry["bsp"] * xm
    mu_x = rx["intercept"] + wc * rx["b"][0]
    lik = norm.logpdf(mi_data["y"], mu_y, ry["sigma"]).sum() + norm.logpdf(xm, mu_x, rx["sigma"]).sum()
    prior = (norm.logpdf(ry["intercept"], 0, 5) + norm.logpdf(ry["b"][0], 0, 2) + norm.logpdf(ry["bsp"], 0, 2)
             + norm.logpdf(ry["sigma"], 0, 1) + np.log(2) + norm.logpdf(rx["intercept"], 0, 5) + norm.logpdf(rx["b"][0], 0, 2)
             + norm.logpdf(rx["sigma"], 0, 2) + np.log(2))
    assert m.pymc_model.compile_logp(jacobian=False)(vp) == pytest.approx(lik + prior, rel=1e-12)
    assert m.pymc_model.compile_logp()(vp) == pytest.approx(lik + prior + np.log(ry["sigma"]) + np.log(rx["sigma"]), rel=1e-12)
    # the gradient in each missing value: its own row of xm's likelihood and of y's
    grad = m.pymc_model.compile_dlogp(vars=[m.pymc_model["Ymi_xm"]])(vp)
    want = (-(ymi - mu_x[MISSING]) / rx["sigma"] ** 2
            + ry["bsp"] * (mi_data["y"].to_numpy()[MISSING] - mu_y[MISSING]) / ry["sigma"] ** 2)
    assert np.allclose(grad, want, rtol=1e-12)


def test_mi_value_point_refusals(mi_model):
    _, resp, ymi = mi_point(mi_model)
    with pytest.raises(ValueError, match=re.escape("Ymi (the missing values of ['xm']) is required")):
        mi_model.value_point(resp=resp)
    with pytest.raises(ValueError, match=re.escape("resp['y'] holds Ymi: the missing values are the model's")):
        mi_model.value_point(resp={**resp, "y": {**resp["y"], "Ymi": {"xm": ymi}}}, Ymi=ymi)
    with pytest.raises(ValueError, match=re.escape("Ymi for 'xm' must have 4 finite entries, one per missing row [0, 5, 6, 17]")):
        mi_model.value_point(resp=resp, Ymi={"xm": ymi[:3]})
    with pytest.raises(ValueError, match=re.escape("Ymi must cover exactly the columns with missing values ['xm']")):
        mi_model.value_point(resp=resp, Ymi={"v": ymi})
    bare = mi_model.value_point(resp=resp, Ymi=ymi)                      # one column: the bare array
    assert np.array_equal(bare["Ymi_xm"], ymi)


def test_mi_predictions(mi_model, mi_data):
    m = mi_model
    pts = [mi_point(m, s) for s in range(3)]
    data = {}
    for vp, resp, ymi in pts:
        nat = {"y_Intercept": resp["y"]["intercept"] - resp["y"]["b"][0] * mi_data["w"].mean(), "y_w": resp["y"]["b"][0],
               "y_sigma": resp["y"]["sigma"], "y_mi(xm)": resp["y"]["bsp"],
               "xm_Intercept": resp["xm"]["intercept"] - resp["xm"]["b"][0] * mi_data["w"].mean(), "xm_w": resp["xm"]["b"][0],
               "xm_sigma": resp["xm"]["sigma"]}
        for k, v in nat.items():
            data.setdefault(k, []).append(v)
        data.setdefault("Ymi_xm", []).append(ymi)
    post = az.InferenceData(posterior=xr.Dataset({k: (("chain", "draw") + (("Ymi_xm_dim",) if k == "Ymi_xm" else ()), np.asarray(v)[None])
                                                  for k, v in data.items()}, coords={"Ymi_xm_dim": MISSING}))
    eta = m.predict(post, kind="linear")["eta"].transpose("chain", "draw", "__obs__", "response").to_numpy()[0]
    for d, (vp, resp, ymi) in enumerate(pts):
        xm = mi_data["xm"].to_numpy().copy()
        xm[MISSING] = ymi
        wc = mi_data["w"] - mi_data["w"].mean()
        assert np.allclose(eta[d, :, 0], resp["y"]["intercept"] + wc * resp["y"]["b"][0] + resp["y"]["bsp"] * xm, rtol=1e-12)
        assert np.allclose(eta[d, :, 1], resp["xm"]["intercept"] + wc * resp["xm"]["b"][0], rtol=1e-12)
    new = pd.DataFrame({"w": [1.0, 1.5], "xm": [0.2, -0.4]})
    lin = m.predict(post, newdata=new, kind="linear")["eta"].transpose("chain", "draw", "__obs__", "response").to_numpy()[0]
    vp, resp, _ = pts[1]
    assert lin[1, :, 0] == pytest.approx(resp["y"]["intercept"] + (new["w"] - mi_data["w"].mean()) * resp["y"]["b"][0]
                                         + resp["y"]["bsp"] * new["xm"], rel=1e-12)
    with pytest.raises(FormulaError, match=re.escape("newdata column 'xm', read by 'mi(xm)', has missing values: on new rows an mi() "
                                                     "predictor must be observed")):
        m.predict(post, newdata=new.assign(xm=[0.2, np.nan]), kind="linear")
    with pytest.raises(FormulaError, match=re.escape("newdata lacks columns ['xm']")):
        m.predict(post, newdata=new[["w"]], kind="linear")
    draws_ = m.predict(post, kind="response", seed=1)["y"]
    assert draws_.shape == (1, 3, len(mi_data), 2) and np.isfinite(draws_).all()
    with pytest.raises(ValueError, match=re.escape("log_lik() of a model with me() or mi() is not offered")):
        ab.log_lik(m, post)


def test_mi_default_priors_and_table(mi_data):
    d = ab.default_priors(MI, mi_data, rescor=False)
    xm = d.resp["xm"]
    assert xm.intercept == ab.StudentT(3, 0, 2.5) and xm.sigma == ab.HalfStudentT(3, 2.5)   # brms's NA median and MAD: 0 and 2.5
    m = ab.gaussian(MI, mi_data, d, rescor=False)
    rows = [(r["prior"], r["class"], r["coef"], r["resp"]) for r in m.prior_table()]
    assert ("", "b", "mixm", "y") in rows and ("student_t(3, 0, 2.5)", "Intercept", "", "xm") in rows
    with pytest.raises(ValueError, match=re.escape("a model with mi() takes rescor=False")):
        ab.default_priors(MI, mi_data)


def test_mi_describe_and_identity(mi_model, mi_data):
    text = mi_model.describe()
    for line in ("formula   : ['y ~ mi(xm) + w', 'xm | mi() ~ w']", "expanded  : bf(y ~ mi(xm) + w) + bf(xm | mi() ~ w)",
                 "Missing values of xm (brms's mi())", "4 of 24 rows of xm are missing (0-based [0, 5, 6, 17]",
                 "mu_i = a_c + b_mi(xm) * xm_i + (x_w - mean(x_w)) * b_w", "Ymi_xm                       the missing values of xm (shared)",
                 "xm | mi() : the response's 4 missing row(s)"):
        assert line in text, line
    again = ab.gaussian(MI, mi_data, mi_priors(), rescor=False)
    assert again.target_identity == mi_model.target_identity
    changed = mi_data.copy()
    changed.loc[3, "xm"] += 0.01
    assert ab.gaussian(MI, changed, mi_priors(), rescor=False).target_identity != mi_model.target_identity
    moved = mi_frame(missing=(0, 5, 6, 18))
    assert ab.gaussian(MI, moved, mi_priors(), rescor=False).target_identity != mi_model.target_identity


def test_mi_several_terms_and_chains():
    """Two mi() terms in one formula and an imputed column read by another imputation formula: one variable of each column's
    missing values, and the joint density written out."""
    rng = np.random.default_rng(8)
    n = 20
    w = rng.normal(size=n)
    x1, x2, y = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    x1[[2, 9]] = np.nan
    x2[[1, 9, 13]] = np.nan
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "w": w})
    m = ab.gaussian(["y ~ mi(x1) + mi(x2)", "x1 | mi() ~ w", "x2 | mi() ~ mi(x1) + w"], df, mi_priors(("y", "x1", "x2")), rescor=False)
    names = [v.name for v in m.pymc_model.value_vars]
    assert names.count("Ymi_x1") == 1 and names.count("Ymi_x2") == 1 and set(m.missing_names) == {"x1", "x2"}
    resp = {"y": {"intercept": 0.2, "b": [], "sigma": 0.9, "bsp": {"x1": 0.5, "x2": -0.3}},
            "x1": {"intercept": -0.1, "b": [0.4], "sigma": 1.1}, "x2": {"intercept": 0.3, "b": [0.2], "sigma": 0.8, "bsp": 0.7}}
    y1, y2 = np.array([0.5, -0.2]), np.array([0.1, 0.9, -1.0])
    vp = m.value_point(resp=resp, Ymi={"x1": y1, "x2": y2})
    f1, f2 = x1.copy(), x2.copy()
    f1[[2, 9]], f2[[1, 9, 13]] = y1, y2
    wc = w - w.mean()
    lik = (norm.logpdf(y, 0.2 + 0.5 * f1 - 0.3 * f2, 0.9).sum() + norm.logpdf(f1, -0.1 + 0.4 * wc, 1.1).sum()
           + norm.logpdf(f2, 0.3 + 0.7 * f1 + 0.2 * wc, 0.8).sum())
    prior = (norm.logpdf(0.2, 0, 5) + norm.logpdf(0.5, 0, 2) + norm.logpdf(-0.3, 0, 2) + norm.logpdf(0.9, 0, 1) + np.log(2)
             + norm.logpdf(-0.1, 0, 5) + norm.logpdf(0.4, 0, 2) + norm.logpdf(1.1, 0, 2) + np.log(2)
             + norm.logpdf(0.3, 0, 5) + norm.logpdf(0.2, 0, 2) + norm.logpdf(0.7, 0, 2) + norm.logpdf(0.8, 0, 3) + np.log(2))
    assert m.pymc_model.compile_logp(jacobian=False)(vp) == pytest.approx(lik + prior, rel=1e-12)


def test_mi_with_group_terms_per_response(mi_data):
    """Different group terms per response: predictions take each response's factors, with one policy for the model's union."""
    pr = ab.Priors(resp={"y": {**response(), "sd": ab.HalfNormal(1)}, "xm": response()})
    m = ab.gaussian(["y ~ mi(xm) + w + (1 | g)", "xm | mi() ~ w"], mi_data, pr, rescor=False)
    assert m.spec.groups == ("g",)
    rng = np.random.default_rng(0)
    post = xr.Dataset({"y_Intercept": (("chain", "draw"), rng.normal(size=(1, 2))), "y_w": (("chain", "draw"), rng.normal(size=(1, 2))),
                       "y_sigma": (("chain", "draw"), np.ones((1, 2))), "y_mi(xm)": (("chain", "draw"), rng.normal(size=(1, 2))),
                       "y_1|g": (("chain", "draw", "g__factor_dim"), rng.normal(size=(1, 2, 4))),
                       "y_1|g_sigma": (("chain", "draw"), np.ones((1, 2))),
                       "xm_Intercept": (("chain", "draw"), rng.normal(size=(1, 2))), "xm_w": (("chain", "draw"), rng.normal(size=(1, 2))),
                       "xm_sigma": (("chain", "draw"), np.ones((1, 2))), "Ymi_xm": (("chain", "draw", "Ymi_xm_dim"), rng.normal(size=(1, 2, 4)))},
                      coords={"g__factor_dim": ["a", "b", "c", "d"], "Ymi_xm_dim": MISSING})
    post = az.InferenceData(posterior=post)
    new = pd.DataFrame({"w": [1.0], "xm": [0.3], "g": ["e"]})
    with pytest.raises(FormulaError, match=re.escape("newdata has levels of 'g' not seen in training: ['e']")):
        m.predict(post, newdata=new, kind="linear")
    out = m.predict(post, newdata=new, kind="linear", new_groups="population", seed=3)
    assert out["eta"].shape == (1, 2, 1, 2) and out.attrs["new_groups"] == "population"


def test_me_fit_record(me_model):
    import json

    pts = [me_point(me_model, s)[1] for s in range(4)]
    idata = draws(me_model, [{"Intercept": p["intercept"], "sigma": p["sigma"], "me(xobs, sdx)": p["bsp"], "meanme_xobs": p["meanme"],
                              "sdme_xobs": p["sdme"], "zme_xobs": p["zme"]} for p in pts])
    fit = ab.FitResult(idata, "pymc", "cpu", "cpu", "cpu", "float64", None, 1, 4, 0, 0.9, 0, seed=0,
                       _target_identity_json=json.dumps(me_model.target_identity))
    rec = ab.fit_record(me_model, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    json.dumps(rec)
    model = rec["model"]
    assert model["priors"]["meanme:xobs"] == "Normal(mu=0, sigma=2)" and model["priors"]["sdme:xobs"] == "HalfNormal(sigma=1)"
    assert model["priors"]["b:me(xobs, sdx)"] == "Normal(mu=0, sigma=2)" and model["fit_binding_status"] == "matched"
    assert model["measurement_error"] == {"me(xobs, sdx)": {"column": "xobs", "noise_column": "sdx", "names": me_model.me_names["xobs"]}}
    assert {"meanme_xobs", "sdme_xobs", "me(xobs, sdx)"} <= set(rec["posterior"])
