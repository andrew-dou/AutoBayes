"""Beta-binomial, geometric and negbinomial2 responses (brms-parity D02, docs/discrete-families.md): each family's mass in brms's
parameterisation against SciPy's own distribution (whose moments confirm the parameterisation) on both paths, its normalisation,
negbinomial2 and geometric as the negative binomial at shape 1 / sigma and 1, the beta-binomial's trials (zero-trial rows,
separation, new rows), the term model composing with the new likelihoods, every other construct refused by name, the priors and
value_point of phi and sigma, by-hand predictions and response draws, prior-predictive draws, brms's default priors, describe(),
the target identity, the fit record and log_lik(). No MCMC; the brms comparison is the oracle cases beta_binomial_varint,
geometric_varint and negbinomial2_varint."""
import json
import math
import warnings
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit

import autobayes as ab
from autobayes.families import get_family

FAMILIES = ("beta_binomial", "geometric", "negbinomial2")
LHS = {"beta_binomial": "k | trials(size)", "geometric": "ynb", "negbinomial2": "ynb"}
OWN = {"beta_binomial": {"phi": ab.Gamma(2, 0.5)}, "geometric": {}, "negbinomial2": {"sigma": ab.Gamma(0.4, 0.3)}}
VALUES = {"beta_binomial": {"phi": 3.5}, "geometric": {}, "negbinomial2": {"sigma": 0.6}}


def frame(n=24, seed=5):
    """The frame whose brms defaults are recorded below (written to CSV and read by R)."""
    rng = np.random.default_rng(seed)
    size = rng.integers(1, 13, n)
    return pd.DataFrame({"x": np.round(rng.normal(size=n), 4), "w": np.round(rng.uniform(-1, 1, n), 4), "o": np.round(rng.normal(0, 0.2, n), 4),
                         "z": np.array(["p", "q", "r"])[np.arange(n) % 3], "g": [f"l{k}" for k in np.arange(n) % 4],
                         "h": np.array(["u", "v"])[(np.arange(n) // 4) % 2],
                         "size": size, "k": rng.binomial(size, 0.4), "ynb": rng.negative_binomial(2, 0.4, n)})


def build(family, formula=None, frame_=None, **extra):
    priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **{**OWN[family], **extra})
    return getattr(ab, family)(formula or f"{LHS[family]} ~ x", frame() if frame_ is None else frame_, priors)


def scipy_distribution(family, eta, values, trials=None):
    """SciPy's distribution of each row, from brms's parameterisation, derived without the library's formulas."""
    if family == "beta_binomial":
        mu = expit(eta)
        return stats.betabinom(trials, mu * values["phi"], (1 - mu) * values["phi"])
    mu = np.exp(eta)
    shape = 1.0 if family == "geometric" else 1.0 / values["sigma"]
    return stats.nbinom(shape, shape / (shape + mu))


def moments(family, eta, values, trials=None):
    """brms's posterior_epred and the variance its parameterisation implies (brms's vignette)."""
    if family == "beta_binomial":
        mu, phi = expit(eta), values["phi"]
        return trials * mu, trials * mu * (1 - mu) * (trials + phi) / (1 + phi)
    mu = np.exp(eta)
    return mu, mu + (1.0 if family == "geometric" else values["sigma"]) * mu ** 2


def graph(m, point, names):
    pmm = m.pymc_model
    return [np.asarray(a, dtype=float) for a in
            pmm.compile_fn(pmm.replace_rvs_by_values([pmm[n] for n in names]), inputs=pmm.value_vars, on_unused_input="ignore")(point)]


def rows_logp(m, point):
    pmm = m.pymc_model
    f = pmm.compile_fn(pmm.logp(vars=pmm.observed_RVs, sum=False), inputs=pmm.value_vars, on_unused_input="ignore")
    return np.asarray(f(point)[0], dtype=float).ravel()


def response(family, df):
    return df["k" if family == "beta_binomial" else "ynb"].to_numpy()


# -- the masses: SciPy, the parameterisation by moments, both paths ----------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_mass_is_brms_parameterisation_on_both_paths(family):
    m = build(family)
    df, v = frame(), VALUES[family]
    p = m.value_point(intercept=0.4, b=[0.3], **v)
    x = df.x.to_numpy()
    eta = 0.4 + 0.3 * (x - x.mean())
    trials = df["size"].to_numpy()
    dist = scipy_distribution(family, eta, v, trials)
    mean, var = moments(family, eta, v, trials)
    np.testing.assert_allclose(dist.mean(), mean, rtol=1e-12)               # the parameterisation, by moments
    np.testing.assert_allclose(dist.var(), var, rtol=1e-12)
    np.testing.assert_allclose(rows_logp(m, p), dist.logpmf(response(family, df)), rtol=1e-12)
    from pymc.sampling.jax import get_jaxified_logp
    import jax

    pmm = m.pymc_model
    names = [w.name for w in pmm.value_vars]
    args = [p[n] for n in names]
    jf = get_jaxified_logp(pmm)
    np.testing.assert_allclose(float(jf(args)), float(pmm.compile_logp()(p)), rtol=1e-13)
    g_jax = np.concatenate([np.ravel(a) for a in jax.grad(lambda q: jf(q))(args)])
    # PyTensor's digamma (the gradient of the log-gamma terms of phi and of 1 / sigma) is off by about 1e-10 absolute (SciPy's
    # digamma at 0.05 to 7.5: 6e-11 to 1.5e-10); brms's gradient tolerance (G2) is 1e-7
    np.testing.assert_allclose(g_jax, np.asarray(pmm.compile_dlogp()(p), dtype=float), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_mass_sums_to_one(family):
    """Over the whole support at a few linear predictors and parameter values (the negative binomials' tail beyond the
    enumerated counts is below 1e-14)."""
    import pymc as pm

    from autobayes import likelihoods as lk

    for eta in (-2.0, 0.3, 1.5):
        if family == "beta_binomial":
            n = 9
            dist = lk.BetaBinomialLogit.dist(np.float64(eta), np.float64(2.5), np.int64(n))
            support = np.arange(n + 1)
        else:
            dist = lk.Geometric.dist(np.float64(math.exp(eta))) if family == "geometric" else lk.NegBinomial2.dist(
                np.float64(math.exp(eta)), np.float64(0.6))
            support = np.arange(400)
        logp = pm.logp(dist, support)
        total = float(np.exp(np.asarray(_evaluate(logp), dtype=float)).sum())
        assert abs(total - 1.0) < 1e-12, (family, eta, total)


def _evaluate(var):
    import pytensor

    return pytensor.function([], var)()


def test_negbinomial2_and_geometric_are_the_negative_binomial_at_shape_one_over_sigma_and_one():
    """brms's relation, read off the masses: negbinomial2(mu, sigma) is negbinomial(mu, shape = 1 / sigma), geometric(mu) is
    negbinomial(mu, shape = 1)."""
    df = frame()
    nb = ab.negbinomial("ynb ~ x", df, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), shape=ab.Gamma(2, 0.5)))
    for sigma in (0.25, 1.0, 4.0):
        two = build("negbinomial2")
        np.testing.assert_allclose(rows_logp(two, two.value_point(intercept=0.4, b=[0.3], sigma=sigma)),
                                   rows_logp(nb, nb.value_point(intercept=0.4, b=[0.3], shape=1.0 / sigma)), rtol=1e-13)
    geo = build("geometric")
    np.testing.assert_allclose(rows_logp(geo, geo.value_point(intercept=0.4, b=[0.3])),
                               rows_logp(nb, nb.value_point(intercept=0.4, b=[0.3], shape=1.0)), rtol=1e-13)


# Stan 2.32's beta_binomial_lpmf(5 | 12, mu phi, (1 - mu) phi) at mu = inv_logit(0.3) (brms's form; rstan::expose_stan_functions,
# printed with 17 digits), with the largest difference allowed from it: PyMC's log-gamma differences lose about
# eps * phi * log(phi) as phi grows, Stan's lbeta less (against a 50-digit evaluation of the same mass, PyMC's errors are 1.4e-13,
# 1.3e-11, 1.5e-9 and 1.4e-7 at these four phi, Stan's 8e-15, 4e-13, 1e-11 and 9e-9; divergence D82)
STAN_BETA_BINOMIAL = {1e2: (-2.0739606318007962, 1e-12), 1e4: (-2.0776264564640545, 5e-11), 1e6: (-2.0777007866447779, 3e-9),
                      1e8: (-2.0777015432279544, 3e-7)}


def test_beta_binomial_mass_equals_stans_and_tends_to_the_binomial():
    from autobayes import likelihoods as lk
    import pymc as pm

    for phi, (want, tol) in STAN_BETA_BINOMIAL.items():
        got = float(_evaluate(pm.logp(lk.BetaBinomialLogit.dist(np.float64(0.3), np.float64(phi), np.int64(12)), 5)))
        assert abs(got - want) <= tol, (phi, got, want)
    df = frame()
    bb = build("beta_binomial")
    binom = ab.binomial("k | trials(size) ~ x", df, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1)))
    limit = rows_logp(binom, binom.value_point(intercept=0.4, b=[0.3]))
    for phi in (1e3, 1e5):                                  # the binomial as phi grows: the difference is O(n^2 / phi)
        gap = np.abs(rows_logp(bb, bb.value_point(intercept=0.4, b=[0.3], phi=phi)) - limit)
        assert (gap <= df["size"].to_numpy() ** 2 / phi).all() and gap.max() > 0


# -- the beta-binomial's trials ---------------------------------------------------------------------------------------------
def test_beta_binomial_trials_zero_trial_rows_and_separation():
    df = frame()
    df.loc[[2, 7], ["k", "size"]] = 0                                      # two zero-trial rows, dropped as for the binomial (D4)
    m = build("beta_binomial", frame_=df)
    assert m.dropped_rows == (2, 7) and m.design.n == len(df) - 2 and int(m.design.trials.sum()) == int(df["size"].sum())
    np.testing.assert_array_equal(m.design.trials, df["size"].drop(index=[2, 7]).to_numpy())
    assert "2 zero-trial row(s) dropped (positions [2, 7])" in m.describe()
    zeros = frame().assign(k=0)
    assert any("every fitted row has zero successes" in w for w in build("beta_binomial", frame_=zeros).warnings)
    with pytest.raises(ab.FormulaError, match=r"successes 'k' = 13 exceed trials 'size'"):
        build("beta_binomial", frame_=frame().assign(k=13))
    with pytest.raises(ab.FormulaError, match="trials"):
        ab.beta_binomial("k ~ x", frame(), ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), phi=ab.Gamma(2, 0.5)))
    with pytest.raises(ab.FormulaError, match="trials"):
        ab.geometric("ynb | trials(size) ~ x", frame(), ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1)))


# -- the term model composes with the new likelihoods ---------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_slopes_categorical_interactions_transforms_offset_and_gp_compose(family):
    """Correlated varying slopes, a combined grouping factor g:h, a declared categorical predictor and its interaction, a
    transform, an offset and a Gaussian process all reach the likelihood through the linear predictor: the observed rows'
    mass at a point is SciPy's at the graph's linear predictor."""
    extra = dict(sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}, "g:h": ab.HalfNormal(0.5)}, cor={"g": ab.LKJ(2)},
                 sdgp=ab.HalfNormal(1), lscale=ab.InverseGamma(3, 1))
    priors = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **OWN[family], **extra)
    m = getattr(ab, family)(f"{LHS[family]} ~ x * z + I(w^2) + offset(o) + gp(w) + (1 + x | g) + (1 | g:h)", frame(), priors,
                            categorical={"z": "p"})
    pmm = m.pymc_model
    rng = np.random.default_rng(7)
    p = {k: v + 0.2 * rng.standard_normal(np.shape(v)) for k, v in pmm.initial_point().items()}
    (parent,) = graph(m, p, [m.names["linpred"]])
    eta = parent.ravel() if family == "beta_binomial" else np.log(parent.ravel())      # Bambi's mu is exp(eta) under the log link
    values = {"phi": math.exp(p["phi_log__"])} if family == "beta_binomial" else {}
    if family == "negbinomial2":
        values["sigma"] = math.exp(p["sigma_log__"])
    df = frame()
    np.testing.assert_allclose(rows_logp(m, p), scipy_distribution(family, eta, values, df["size"].to_numpy()).logpmf(response(family, df)),
                               rtol=1e-11)
    assert m.spec.gp_terms and m.design.offset is not None and "g" in m.correlation_names and {"x:z", "I(w^2)"} <= set(m.spec.terms)
    assert "g:h" in m.spec.groups


@pytest.mark.parametrize("family", FAMILIES)
def test_uncentred_design_and_no_intercept(family):
    """center_predictors=False and a formula without an intercept reach the likelihood as the centred one does."""
    df = frame()
    for formula, centre, values in ((f"{LHS[family]} ~ x + w", False, dict(intercept=0.2, b=[0.3, -0.4])),
                                    (f"{LHS[family]} ~ 0 + x + w", True, dict(b=[0.3, -0.4]))):
        priors = ab.Priors(intercept=ab.Normal(0, 2) if "intercept" in values else None, b=ab.Normal(0, 1), **OWN[family])
        m = getattr(ab, family)(formula, df, priors, center_predictors=centre)
        eta = values.get("intercept", 0.0) + 0.3 * df.x.to_numpy() - 0.4 * df.w.to_numpy()
        np.testing.assert_allclose(rows_logp(m, m.value_point(**values, **VALUES[family])),
                                   scipy_distribution(family, eta, VALUES[family], df["size"].to_numpy()).logpmf(response(family, df)),
                                   rtol=1e-12)


# -- every other construct is refused, naming it (D80, D81) ---------------------------------------------------------------------
def _lhs(family, addition):
    return f"k | trials(size) + {addition}" if family == "beta_binomial" else f"ynb | {addition}"


REFUSALS = [
    ("distributional", lambda f: [f"{LHS[f]} ~ x", {"beta_binomial": "phi ~ w", "geometric": "sigma ~ w", "negbinomial2": "sigma ~ w"}[f]],
     {"beta_binomial": r"brms's phi ~ \.\.\. is not offered here", "geometric": r"the geometric family has no parameter 'sigma'",
      "negbinomial2": r"brms's sigma ~ \.\.\. is not offered here"}),
    ("weights", lambda f: f"{_lhs(f, 'weights(w)')} ~ x", r"'weights\(w\)' is not available for the {f} family"),
    ("cens", lambda f: f"{_lhs(f, 'cens(c)')} ~ x", r"'cens\(c\)' is not available for the {f} family"),
    ("trunc", lambda f: f"{_lhs(f, 'trunc(lb = 0)')} ~ x", r"'trunc\(lb = 0\)' is not available for the {f} family"),
    ("mo", lambda f: f"{LHS[f]} ~ mo(ord)", r"the monotonic term 'mo\(ord\)' is not available for the {f} family"),
    ("me", lambda f: f"{LHS[f]} ~ me(x, sdx)", r"the measurement-error term 'me\(x, sdx\)' is not available for the {f} family"),
    ("mi", lambda f: f"{LHS[f]} ~ mi(xm)", r"the missing-value term 'mi\(xm\)' is not available for the {f} family"),
    ("response_mi", lambda f: f"{_lhs(f, 'mi()')} ~ x", r"'mi\(\)' is not available for the {f} family"),
    ("ar", lambda f: f"{LHS[f]} ~ x + ar(p = 1)", r"the autocorrelation term 'ar\(p = 1\)' is not available for the {f} family"),
    ("mvbind", lambda f: "mvbind(ynb, size) ~ x", r"a multivariate {f} model is not offered"),
    ("mm", lambda f: f"{LHS[f]} ~ x + (1 | mm(g, h))", r"the multi-membership term '\(1 \| mm\(g, h\)\)' is not available for the {f} family"),
    ("gr_by", lambda f: f"{LHS[f]} ~ x + (1 | gr(g, by = h))", r"the term with gr\(by = \) .* is not available for the {f} family"),
]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("construct", [r[0] for r in REFUSALS])
def test_other_constructs_are_refused_by_name(family, construct):
    if family == "beta_binomial" and construct in ("distributional", "weights"):   # E03 lifted them (tests/test_contexts_newer_families.py)
        pytest.skip("E03: the beta-binomial's phi ~ ... and weights() are taken (beta_binomial_distributional_phi, beta_binomial_weighted)")
    _, make, needle = next(r for r in REFUSALS if r[0] == construct)
    needle = needle[family] if isinstance(needle, dict) else needle.format(f=family)
    df = frame().assign(c=0, ord=np.arange(24) % 4, sdx=0.3, xm=np.where(np.arange(24) % 5 == 0, np.nan, 0.5))
    with pytest.raises((ab.FormulaError, ab.PriorError, ValueError), match=needle):
        getattr(ab, family)(make(family), df, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **OWN[family]))


@pytest.mark.parametrize("family", FAMILIES)
def test_other_links_mixtures_and_nonlinear_models_are_refused(family):
    df, priors = frame(), ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **OWN[family])
    own = get_family(family).link
    for link in ("identity", "probit", "sqrt", "cloglog"):
        if link != own:
            with pytest.raises(ValueError, match=rf"the {family} family has the {own} link only; link='{link}' is not available"):
                ab.Model(f"{LHS[family]} ~ x", df, priors, family=family, link=link)
            with pytest.raises(ValueError, match=rf"the {family} family has the {own} link only"):
                ab.default_priors(f"{LHS[family]} ~ x", df, family=family, link=link)
    comp = ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), **OWN[family])
    with pytest.raises(ValueError, match=rf"a {family} mixture component is not offered.*D52"):
        ab.mixture("ynb ~ x", df, ab.Priors(components=[comp, comp], theta=ab.Dirichlet(2)), families=[family, family])
    with pytest.raises(ab.FormulaError, match=rf"the {family} family is not offered with nl=True \(divergence D62\)"):
        ab.NonlinearModel(["ynb ~ a * x", "a ~ 1"], df, ab.Priors(nlpars={"a": {"b": ab.Normal(0, 1)}}, **OWN[family]), family=family)


# -- the priors of phi and sigma, and value_point ---------------------------------------------------------------------------------
def test_priors_of_the_auxiliary_parameters():
    df = frame()
    base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1))
    with pytest.raises(ab.PriorError, match=r"the beta_binomial family needs a prior on the precision parameter phi: pass phi=\.\.\."):
        ab.beta_binomial("k | trials(size) ~ x", df, ab.Priors(**base))
    with pytest.raises(ab.PriorError, match=r"the negbinomial2 family needs a prior on the dispersion sigma .*sigma = 1 / shape of "
                                            r"negbinomial\): pass sigma=\.\.\."):
        ab.negbinomial2("ynb ~ x", df, ab.Priors(**base))
    with pytest.raises(ab.PriorError, match=r"the negbinomial2 family has no shape parameter; do not pass shape\. Its auxiliary "
                                            r"parameter is sigma: pass sigma=\.\.\."):
        ab.negbinomial2("ynb ~ x", df, ab.Priors(**base, sigma=ab.Gamma(0.4, 0.3), shape=ab.Gamma(2, 0.5)))
    with pytest.raises(ab.PriorError, match=r"the geometric family has no residual SD; do not pass sigma\."):
        ab.geometric("ynb ~ x", df, ab.Priors(**base, sigma=ab.Gamma(0.4, 0.3)))
    with pytest.raises(ab.PriorError, match=r"the beta_binomial family has no residual SD; do not pass sigma\. Its auxiliary "
                                            r"parameter is phi"):
        ab.beta_binomial("k | trials(size) ~ x", df, ab.Priors(**base, phi=ab.Gamma(2, 0.5), sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.PriorError, match=r"prior for 'sigma' must be a positive-support prior"):
        ab.negbinomial2("ynb ~ x", df, ab.Priors(**base, sigma=ab.Normal(0, 1)))
    with pytest.raises(ab.PriorError, match=r"prior for 'phi' must be a positive-support prior"):
        ab.beta_binomial("k | trials(size) ~ x", df, ab.Priors(**base, phi=ab.Beta(2, 2)))
    with pytest.raises(ab.PriorError, match=r"prior for 'phi' is a mapping"):
        ab.beta_binomial("k | trials(size) ~ x", df, ab.Priors(**base, phi={"intercept": ab.Normal(0, 1)}))


def test_value_point_coordinates_and_refusals():
    m = build("negbinomial2")
    np.testing.assert_allclose(m.value_point(intercept=0.1, b=[0.2], sigma=0.6)["sigma_log__"], math.log(0.6), rtol=1e-15)
    with pytest.raises(ValueError, match=r"sigma \(the dispersion, brms's 1 / shape\) is required for the negbinomial2 family"):
        m.value_point(intercept=0.1, b=[0.2], shape=2.0)
    with pytest.raises(ValueError, match="sigma must be a finite positive scalar"):
        m.value_point(intercept=0.1, b=[0.2], sigma=0.0)
    m = build("beta_binomial")
    np.testing.assert_allclose(m.value_point(intercept=0.1, b=[0.2], phi=3.5)["phi_log__"], math.log(3.5), rtol=1e-15)
    with pytest.raises(ValueError, match=r"phi \(the precision\) is required for the beta_binomial family"):
        m.value_point(intercept=0.1, b=[0.2])
    m = build("geometric")
    assert set(m.value_point(intercept=0.1, b=[0.2])) == {"Intercept", "x"}
    with pytest.raises(ValueError, match="the geometric family has no auxiliary parameter"):
        m.value_point(intercept=0.1, b=[0.2], sigma=0.5)


# -- predictions ------------------------------------------------------------------------------------------------------------
def _posterior(m, values, n_draw=3):
    """Synthetic posterior draws: one chain, the same values in every draw but the intercept."""
    post = {m.names["intercept"]: np.linspace(0.2, 0.6, n_draw)[None], "x": np.full((1, n_draw), 0.3)}
    for name, v in values.items():
        post[m.names[name]] = np.full((1, n_draw), v)
    return az.from_dict(posterior=post)


@pytest.mark.parametrize("family", FAMILIES)
def test_predictions_by_hand_and_response_draws(family):
    m = build(family)
    v = VALUES[family]
    idata = _posterior(m, v)
    nd = pd.DataFrame({"x": [-1.0, 0.0, 2.0], "size": [3, 7, 12]})
    eta = np.linspace(0.2, 0.6, 3)[:, None] + 0.3 * nd.x.to_numpy()[None]   # the intercept draws are on the raw scale
    lin = m.predict(idata, newdata=nd, kind="linear")["eta"].to_numpy()[0]
    np.testing.assert_allclose(lin, eta, rtol=1e-14)
    out = m.predict(idata, newdata=nd, kind="mean")
    trials = nd["size"].to_numpy()
    if family == "beta_binomial":                      # brms's posterior_epred is trials * mu, predict()'s expected (D5, D82)
        np.testing.assert_allclose(out["mu"].to_numpy()[0], expit(eta), rtol=1e-14)
        np.testing.assert_allclose(out["expected"].to_numpy()[0], trials * expit(eta), rtol=1e-14)
        np.testing.assert_allclose(m.predict(idata, newdata=nd[["x"]], kind="probability")["mu"].to_numpy()[0], expit(eta), rtol=1e-14)
        with pytest.raises(ab.FormulaError, match="newdata lacks the trials column 'size'"):
            m.predict(idata, newdata=nd[["x"]], kind="mean")
    else:
        np.testing.assert_allclose(out["mu"].to_numpy()[0], np.exp(eta), rtol=1e-14)
        assert "expected" not in out
        with pytest.raises(ValueError, match="kind must be 'mean', 'response' or 'linear'"):
            m.predict(idata, newdata=nd, kind="probability")
    big = pd.DataFrame({"x": np.zeros(20000), "size": np.full(20000, 9)})
    draws = m.predict(_posterior(m, v, 1), newdata=big, kind="response", seed=5)["y"].to_numpy().ravel()
    assert draws.dtype == np.int64 and (draws >= 0).all()
    ref = scipy_distribution(family, np.float64(0.2), v, 9)
    assert abs(draws.mean() - ref.mean()) < 4.5 * ref.std() / math.sqrt(draws.size)
    assert abs(draws.var() - ref.var()) < 0.1 * ref.var()
    if family == "beta_binomial":
        assert draws.max() <= 9
        assert draws.var() > 1.5 * stats.binom(9, expit(0.2)).var()          # overdispersed relative to the binomial


@pytest.mark.parametrize("family", FAMILIES)
def test_prior_predictive(family):
    m = build(family, f"{LHS[family]} ~ x + (1 | g)", sd=ab.HalfNormal(0.5))
    out = m.prior_predictive(draws=30, seed=3)
    y = out.prior_predictive[m.names["y"]].values
    assert y.shape == (1, 30, 24) and (y >= 0).all() and (y == np.floor(y)).all()
    if family == "beta_binomial":
        assert (y <= frame()["size"].to_numpy()).all()


# -- brms's default priors (A07) ----------------------------------------------------------------------------------------
# brms 2.23.0's default_prior() of <response> ~ x + (1 | g) on frame() and wide_frame() (written to CSV and read by R 4.5.2), its
# class rows with a prior, as (class, prior, lb, ub) in brms's order. The beta-binomial's are constants (logit link); the
# geometric and negbinomial2 Intercept and sd read log(y), zeros as 0.1 (wide_frame: median 1.67, MAD 4.60); sigma's
# gamma(0.4, 0.3) is a constant
BRMS_CLASS_ROWS = {
    "beta_binomial": [("Intercept", "student_t(3, 0, 2.5)", "", ""), ("phi", "gamma(0.01, 0.01)", "0", ""),
                      ("sd", "student_t(3, 0, 2.5)", "0", "")],
    "geometric": [("Intercept", "student_t(3, 0.9, 2.5)", "", ""), ("sd", "student_t(3, 0, 2.5)", "0", "")],
    "negbinomial2": [("Intercept", "student_t(3, 0.9, 2.5)", "", ""), ("sd", "student_t(3, 0, 2.5)", "0", ""),
                     ("sigma", "gamma(0.4, 0.3)", "0", "")],
}
BRMS_WIDE_ROWS = {
    "beta_binomial": BRMS_CLASS_ROWS["beta_binomial"],
    "geometric": [("Intercept", "student_t(3, 1.7, 4.6)", "", ""), ("sd", "student_t(3, 0, 4.6)", "0", "")],
    "negbinomial2": [("Intercept", "student_t(3, 1.7, 4.6)", "", ""), ("sd", "student_t(3, 0, 4.6)", "0", ""),
                     ("sigma", "gamma(0.4, 0.3)", "0", "")],
}


def wide_frame():
    rng = np.random.default_rng(13)
    n = 30
    size = rng.integers(1, 200, n)
    return pd.DataFrame({"x": np.round(rng.normal(size=n), 4), "g": [f"l{k}" for k in np.arange(n) % 3],
                         "size": size, "k": rng.binomial(size, 0.9),
                         "ynb": np.round(np.exp(rng.normal(2.0, 4.0, n))).astype(int)})


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("which", ["frame", "wide"])
def test_default_priors_build_the_model_brms_describes(family, which):
    df, want = (frame(), BRMS_CLASS_ROWS) if which == "frame" else (wide_frame(), BRMS_WIDE_ROWS)
    formula = f"{LHS[family]} ~ x + (1 | g)"
    pri = ab.default_priors(formula, df, family=family)
    m = getattr(ab, family)(formula, df, pri)
    rows = [(r["class"], r["prior"], r["lb"], r["ub"]) for r in m.prior_table() if r["coef"] == "" and r["group"] == "" and r["prior"]]
    assert rows == want[family]
    assert all(r["source"] == "default" for r in m.prior_table())
    scale = get_family(family).scale_parameter
    if scale is not None:
        assert scale in m.default_prior_labels and getattr(pri, scale).source == "default"


# -- describe(), the target identity and the fit record ----------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_describe(family):
    m = build(family)
    text = m.describe()
    spec = get_family(family)
    assert f"AutoBayes {spec.tag} model" in text and f"family    : {family}, {spec.link} link" in text
    if spec.scale_parameter is not None:
        assert f"  {spec.scale_parameter + '_log__':<24} log({spec.scale_parameter})" in text
    lines = {"beta_binomial": "k_i ~ BetaBinomial(size_i, mu_i phi, (1 - mu_i) phi)",
             "geometric": "y_i ~ Geometric(mu_i) = NegativeBinomial(mu_i, shape = 1)",
             "negbinomial2": "y_i ~ NegativeBinomial(mu_i, shape = 1 / sigma)"}
    assert lines[family] in text
    if family == "beta_binomial":
        assert f"{int(frame()['size'].sum())} trials in total" in text
    assert ab.Model(f"{LHS[family]} ~ x", frame(), m.priors, family=family).describe() == text


@pytest.mark.parametrize("family", ["beta_binomial", "negbinomial2"])
def test_target_identity_and_fit_record(family, monkeypatch):
    m = build(family)
    scale = get_family(family).scale_parameter
    other = {"phi": ab.Gamma(0.01, 0.01), "sigma": ab.Exponential(1)}[scale]
    assert build(family).target_identity == m.target_identity
    assert build(family, **{scale: other}).target_identity != m.target_identity
    assert build("geometric").target_identity != ab.negbinomial("ynb ~ x", frame(), ab.Priors(
        intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), shape=ab.Gamma(2, 0.5))).target_identity
    post = _posterior(m, VALUES[family], n_draw=4)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    assert scale in rows
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=1, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == family and rec["model"]["priors"][scale] == str(OWN[family][scale])
    if family == "beta_binomial":
        assert rec["model"]["trials"] == "size"
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


@pytest.mark.parametrize("family", FAMILIES)
def test_log_lik_is_the_models_own_pointwise_mass(family):
    m = build(family, f"{LHS[family]} ~ x + (1 | g)", sd=ab.HalfNormal(0.5))
    v = VALUES[family]
    post = _posterior(m, v, n_draw=2)
    post.posterior["1|g_sigma"] = (("chain", "draw"), np.full((1, 2), 0.5))
    effects = np.array([0.3, -0.2, 0.1, 0.0])
    coord = m.group_names["g"]["coordinate"]
    post.posterior["1|g"] = (("chain", "draw", coord), np.tile(effects, (1, 2, 1)))
    post.posterior = post.posterior.assign_coords({coord: list(m.design.levels)})
    ll = ab.log_lik(m, post).to_numpy()[0]
    x = frame().x.to_numpy()
    for d, a in enumerate([0.2, 0.6]):
        p = m.value_point(intercept=a + 0.3 * x.mean(), b=[0.3], sd=0.5, z=effects / 0.5, **v)
        np.testing.assert_allclose(ll[d], rows_logp(m, p), rtol=1e-12)


@pytest.mark.parametrize("case", [f"{f}_varint" for f in FAMILIES])
def test_bayes_R2_loo_and_waic_on_the_frozen_points(case):
    """bayes_R2 at the frozen brms points is brms's definition on brms's own posterior_epred (the fixture's: trials * mu for the
    beta-binomial), and loo and waic run on the families' log_lik (brms's loo and waic of these cases are not frozen)."""
    from tests.oracle import adapter, gates

    b = gates.Bundle(case)
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    epred, y = b.fixture.prediction("fitted", "epred"), b.model.design.y
    want = np.var(epred, axis=-1, ddof=1) / (np.var(epred, axis=-1, ddof=1) + np.var(y - epred, axis=-1, ddof=1))
    np.testing.assert_allclose(ab.bayes_R2(b.model, idata).to_numpy()[0], want, rtol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                       # eight one-chain draws: Pareto k is large, as it should be
        result = ab.loo(b.model, idata)
        waic = ab.waic(b.model, idata)
    assert np.isfinite(result.estimates.loc["elpd_loo", "Estimate"]) and np.isfinite(waic.estimates.loc["elpd_waic", "Estimate"])
