"""brms-parity A12: the probit link of the Bernoulli and binomial families, and P6: their cloglog link (docs/links.md).

Refusals and the registry; the log density and gradient on both paths at linear predictors up to |eta| = 300 against
scipy's log_ndtr and the analytic Mills ratios; value_point; predictions on fitted and new rows under both unseen-group
policies; describe(); the fit record and the target identity; prior-predictive support; and the numerics of the
cloglog link against a 50-digit Decimal reference on |eta| <= 40, its models, describe() and predictions (the oracle
cases bernoulli_cloglog and binomial_cloglog compare them with brms). Nothing samples.
"""
import json
from decimal import Decimal, getcontext
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.special import erfcx, log_ndtr, ndtr, ndtri
from scipy.stats import halfnorm, norm

import autobayes as ab
from autobayes import likelihoods as lk
from autobayes.families import FAMILIES, get_family
from autobayes.families.links import INVERSE_LINKS, LINKS, PARKED_LINKS, inverse_cloglog
from tests.reference_binomial import log_binom_dec

PRIORS = ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Normal(0, 1.5), sd=ab.HalfNormal(1))
ETAS = [0.0, 0.5, -0.5, 3.0, -3.0, 12.0, -12.0, 40.0, -40.0, 300.0, -300.0]
BERNOULLI, BINOMIAL = "y ~ x + (1 | g)", "k | trials(n) ~ x + (1 | g)"


def frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    g = np.array([f"l{i % 6}" for i in range(n)], dtype=object)
    rng.shuffle(g)
    u = dict(zip(sorted(set(g)), rng.normal(0, 0.6, 6)))
    eta = -0.2 + 0.7 * x + np.array([u[v] for v in g])
    size = rng.integers(1, 9, n)
    return pd.DataFrame({"x": x, "g": g, "y": rng.binomial(1, ndtr(eta)), "k": rng.binomial(size, ndtr(eta)), "n": size})


@pytest.fixture(scope="module")
def df():
    return frame()


@pytest.fixture(scope="module")
def pmodel(df):
    return ab.bernoulli(BERNOULLI, df, PRIORS, link="probit")


@pytest.fixture(scope="module")
def bpmodel(df):
    return ab.binomial(BINOMIAL, df, PRIORS, link="probit")


# -- references written from the mathematics ------------------------------------------------------------------
def _eta(model, nat):
    d = model.design
    return nat["intercept"] + d.X_model @ nat["b"] + nat["sd"] * nat["z"][d.group_index]


def _mills(z):
    """phi(z) / Phi(z), the derivative of log Phi(z), as sqrt(2 / pi) / erfcx(-z / sqrt 2): accurate to a few ulps for
    every z (exp(logpdf(z) - log_ndtr(z)) subtracts two numbers near -z^2 / 2 and is off by 1.4e-8 relative at -1e4)."""
    return np.sqrt(2.0 / np.pi) / erfcx(-np.asarray(z, dtype=float) / np.sqrt(2.0))


def reference(model, nat, *, jacobian=True):
    """The target in PyMC's value space: the probit likelihood through scipy's log_ndtr (log C(n, k) from the 60-digit
    decimal reference for binomial rows), the priors of PRIORS, z ~ Normal(0, 1), plus log(sd) with the Jacobian."""
    d = model.design
    eta, y = _eta(model, nat), d.y
    if d.trials is None:
        lik = np.sum(np.where(y == 1, log_ndtr(eta), log_ndtr(-eta)))
    else:
        lik = (sum(float(log_binom_dec(int(n), int(k))) for n, k in zip(d.trials, y))
               + np.sum(y * log_ndtr(eta) + (d.trials - y) * log_ndtr(-eta)))
    prior = (norm.logpdf(nat["intercept"], 0, 2.5) + norm.logpdf(nat["b"], 0, 1.5).sum()
             + halfnorm.logpdf(nat["sd"], scale=1) + norm.logpdf(nat["z"]).sum())
    return lik + prior + (np.log(nat["sd"]) if jacobian else 0.0)


def analytic_gradient(model, nat):
    """d log p / d (value variables): per row, d/d eta of y log Phi(eta) + (n - y) log Phi(-eta) is y r(eta) - (n - y) r(-eta)
    with r the Mills ratio phi / Phi; HalfNormal(1) on sd plus the Jacobian gives -sd^2 + 1 in log(sd)."""
    d = model.design
    eta, y = _eta(model, nat), d.y
    n = np.ones_like(y) if d.trials is None else d.trials
    r = y * _mills(eta) - (n - y) * _mills(-eta)
    a, sd, z = nat["intercept"], nat["sd"], nat["z"]
    return {"Intercept": r.sum() - a / 2.5 ** 2, "x": (r * d.X_model[:, 0]).sum() - nat["b"][0] / 1.5 ** 2,
            "1|g_sigma_log__": (r * z[d.group_index]).sum() * sd - sd ** 2 + 1.0,
            "1|g_offset": np.array([r[d.group_index == j].sum() * sd - z[j] for j in range(d.J)])}


def _random_points(model, k=4, seed=3):
    rng = np.random.default_rng(seed)
    return [dict(intercept=rng.normal(0, 0.8), b=rng.normal(0, 0.6, 1), sd=np.exp(rng.normal(-0.5, 0.4)),
                 z=rng.normal(size=model.design.J)) for _ in range(k)]


def _both_paths(model):
    """The value-variable order and logp / gradient evaluators on the PyTensor and the JAX path."""
    import jax
    import jax.numpy as jnp
    from pymc.sampling.jax import get_jaxified_logp

    pmm = model.pymc_model
    names = [v.name for v in pmm.value_vars]
    logp, dlogp = pmm.compile_logp(), pmm.compile_dlogp()
    jf = get_jaxified_logp(pmm)
    jgrad = jax.grad(lambda *args: jf(list(args)), argnums=tuple(range(len(names))))

    def jax_value(point):
        return float(jf([point[n] for n in names]))

    def jax_grad(point):
        return np.concatenate([np.atleast_1d(np.asarray(g, dtype=float)) for g in jgrad(*[jnp.asarray(point[n]) for n in names])])

    return names, logp, dlogp, jax_value, jax_grad


def _gvec(model, nat, names):
    g = analytic_gradient(model, nat)
    return np.concatenate([np.atleast_1d(g[n]) for n in names])


# -- refusals and the registry ----------------------------------------------------------------------------------
def test_links_offered_parked_and_refused(df):
    assert set(LINKS) == {"logit", "probit", "cloglog"} and not PARKED_LINKS          # P6 lifted the cloglog link's park
    assert set(INVERSE_LINKS) == {"logit", "probit", "cloglog"} and INVERSE_LINKS["probit"] is ndtr
    for family in ("bernoulli", "binomial"):
        spec = FAMILIES[family]
        assert spec.link == "logit" and set(spec.links) == {"probit", "cloglog"} and not spec.parked_links
        probit = get_family(family, "probit")
        assert probit.name == family and probit.parent == "probit_p" and probit.link == probit.link_scale == "probit"
        assert set(probit.links) == {"logit", "cloglog"} and probit.parked_links is spec.parked_links
        cloglog = get_family(family, "cloglog")
        assert cloglog.name == family and cloglog.parent == "cloglog_p" and cloglog.link == cloglog.link_scale == "cloglog"
        assert set(cloglog.links) == {"logit", "probit"} and get_family(family, "cloglog") is cloglog
        assert cloglog.inverse_link is inverse_cloglog and cloglog.with_link("probit") is probit
        assert get_family(family, "probit") is probit and probit.with_link("logit") is spec     # one spec per link
        assert get_family(family, None) is spec and get_family(family, "logit") is spec and probit.with_link("probit") is probit
    ordinal = {"cumulative", "sratio", "cratio", "acat",          # D04: brms's other ordinal links are withheld, with reasons (D38)
               "hurdle_cumulative"}                               # E01: and hurdle_cumulative's (D109)
    more = {"poisson": {"sqrt"}, "negbinomial": {"identity"}, "gamma": {"inverse", "identity"}, "beta": {"probit", "cloglog"}}   # E06
    for name, spec in FAMILIES.items():
        if name in more:
            assert set(spec.links) == more[name] and spec.with_link(spec.link) is spec
        elif name not in ("bernoulli", "binomial", "cumulative", "inverse_gaussian"):
            assert not spec.links and spec.with_link(spec.link) is spec
        if name in ordinal:
            assert set(spec.parked_links) == ({"probit_approx", "cauchit"} if name == "cumulative"
                                              else {"probit", "probit_approx", "cloglog", "cauchit"})
        elif name not in ("bernoulli", "binomial"):
            assert not spec.parked_links
    cumulative = FAMILIES["cumulative"]                           # D04, P6: its probit link is offered (D88)
    assert set(cumulative.links) == {"probit", "cloglog"} and cumulative.with_link("probit").link == "probit"   # E06: cloglog
    assert cumulative.with_link("probit") is cumulative.with_link("probit") and cumulative.with_link("logit") is cumulative
    ig = FAMILIES["inverse_gaussian"]                             # E02: brms's default 1/mu^2 and its log link
    assert ig.link == "1/mu^2" and set(ig.links) == {"log"} and ig.with_link("log").with_link("1/mu^2") is ig
    with pytest.raises(ValueError, match=r"link must be one of \['cloglog', 'logit', 'probit'\] for the bernoulli family, got 'cauchit'"):
        ab.bernoulli(BERNOULLI, df, PRIORS, link="cauchit")
    with pytest.raises(ValueError, match=r"for the binomial family, got 1$"):
        ab.binomial(BINOMIAL, df, PRIORS, link=1)
    for ctor, formula in ((ab.bernoulli, BERNOULLI), (ab.binomial, BINOMIAL)):     # P6: built, no longer refused (D18)
        assert ctor(formula, df, PRIORS, link="cloglog").link == "cloglog"
    gpriors = ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Normal(0, 1.5), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
    with pytest.raises(ValueError, match=r"the gaussian family has the identity link only; link='probit' is not available "
                                         r"\(link= applies to the families with more than one link: poisson \['log', 'sqrt'\], negbinomial "
                                         r"\['identity', 'log'\], gamma \['identity', 'inverse', 'log'\], beta \['cloglog', 'logit', 'probit'\], "
                                         r"bernoulli \['cloglog', 'logit', 'probit'\], binomial \['cloglog', 'logit', 'probit'\], cumulative "
                                         r"\['cloglog', 'logit', 'probit'\], inverse_gaussian \['1/mu\^2', 'log'\]\)"):                    # E06 added the Poisson, NB, Gamma and Beta links
        ab.Model(BERNOULLI, df, gpriors, family="gaussian", link="probit")
    with pytest.raises(ValueError, match=r"link must be one of \['log', 'sqrt'\] for the poisson family, got 'identity'"):   # E06: sqrt
        ab.Model("k ~ x + (1 | g)", df, PRIORS, family="poisson", link="identity")
    with pytest.raises(ab.PriorError, match="has no residual SD; do not pass sigma. Its intercept and slope priors are on the probit scale"):
        ab.bernoulli(BERNOULLI, df, gpriors, link="probit")
    assert ab.Model(BERNOULLI, df, gpriors, family="gaussian", link="identity").link == "identity"   # a family's own link, explicitly
    m = ab.bernoulli(BERNOULLI, df, PRIORS, link="probit")
    same = ab.Model(BERNOULLI, df, PRIORS, family="bernoulli", link="probit")
    assert same.target_identity == m.target_identity and same.describe() == m.describe()


def test_model_attributes_and_target_identity(df, pmodel, bpmodel):
    for m, y in ((pmodel, "y"), (bpmodel, "p(k, n)")):
        assert m.link == "probit" and m.family_spec is get_family(m.family, "probit") and m.family_spec.link_scale == "probit"
        assert m.names["linpred"] == "probit_p" and m.names["mean"] == "p" and m.names["y"] == y and "sigma" not in m.names
        assert {v.name for v in m.pymc_model.free_RVs} == {"Intercept", "x", "1|g_sigma", "1|g_offset"}
        assert {v.name for v in m.pymc_model.deterministics} == {"1|g", "probit_p"}
    logit = ab.bernoulli(BERNOULLI, df, PRIORS)
    assert logit.target_identity == ab.bernoulli(BERNOULLI, df, PRIORS, link="logit").target_identity
    assert logit.target_identity != pmodel.target_identity                                 # the link is part of the target
    assert ab.binomial(BINOMIAL, df, PRIORS).target_identity != bpmodel.target_identity


# -- the density and its gradient --------------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["bernoulli", "binomial"])
def test_logp_and_gradient_match_the_reference_on_both_paths(which, pmodel, bpmodel):
    pytest.importorskip("jax")
    m = pmodel if which == "bernoulli" else bpmodel
    names, logp, dlogp, jax_value, jax_grad = _both_paths(m)
    logp_nojac = m.pymc_model.compile_logp(jacobian=False)
    for nat in _random_points(m):
        point = m.value_point(**nat)
        ref, ref_nojac = reference(m, nat), reference(m, nat, jacobian=False)
        assert abs(float(logp(point)) - ref) < 1e-8 and abs(float(logp_nojac(point)) - ref_nojac) < 1e-8
        assert abs(ref - ref_nojac - np.log(nat["sd"])) < 1e-12 and abs(jax_value(point) - ref) < 1e-8
        np.testing.assert_allclose(np.asarray(dlogp(point), dtype=float), _gvec(m, nat, names), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(jax_grad(point), _gvec(m, nat, names), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("which", ["bernoulli", "binomial"])
def test_extreme_linear_predictors_are_finite_and_exact_on_both_paths(which, pmodel, bpmodel):
    """At |eta| up to 300 (Stan's Phi is exactly 0 below -37.5 and 1 above 8.25, where brms's density is -inf) the
    value and the gradient equal the stable reference on both paths."""
    pytest.importorskip("jax")
    m = pmodel if which == "bernoulli" else bpmodel
    names, logp, dlogp, jax_value, jax_grad = _both_paths(m)
    rng = np.random.default_rng(5)
    for a in ETAS:
        for b, sd in ((0.0, 0.1), (0.7, 0.5)):
            nat = dict(intercept=a, b=np.array([b]), sd=sd, z=rng.normal(size=m.design.J))
            point = m.value_point(**nat)
            ref, gvec = reference(m, nat), _gvec(m, nat, names)
            tol = 1e-8 * max(1.0, abs(ref))
            v_pt, g_pt = float(logp(point)), np.asarray(dlogp(point), dtype=float)
            v_jx, g_jx = jax_value(point), jax_grad(point)
            assert np.isfinite(v_pt) and np.isfinite(v_jx) and np.all(np.isfinite(g_pt)) and np.all(np.isfinite(g_jx))
            assert abs(v_pt - ref) < tol and abs(v_jx - ref) < tol, (a, b, sd, v_pt, v_jx, ref)
            np.testing.assert_allclose(g_pt, gvec, rtol=1e-9, atol=1e-8, err_msg=f"PyTensor gradient at intercept {a}")
            np.testing.assert_allclose(g_jx, gvec, rtol=1e-9, atol=1e-8, err_msg=f"JAX gradient at intercept {a}")


@pytest.mark.parametrize("n", [1, 10, 10**3, 10**6])
def test_binomial_probit_large_counts_on_both_paths(n):
    """One row per grid count k in its own group, every row at the same eta: counts up to the validated cap times the
    probit log probabilities, against the reference, on both paths."""
    pytest.importorskip("jax")
    ks = sorted({0, 1, n // 2, n - 1, n})
    rows = len(ks)
    data = pd.DataFrame({"k": ks, "n": [n] * rows, "x": np.linspace(-0.3, 0.3, rows), "g": [f"g{i}" for i in range(rows)]})
    m = ab.binomial(BINOMIAL, data, PRIORS, link="probit")
    names, logp, dlogp, jax_value, jax_grad = _both_paths(m)
    for eta in ETAS:
        nat = dict(intercept=eta, b=np.array([0.0]), sd=0.1, z=np.zeros(rows))
        point = m.value_point(**nat)
        ref, gvec = reference(m, nat), _gvec(m, nat, names)
        tol = max(1e-8, 1e-12 * abs(ref))
        v_pt, v_jx = float(logp(point)), jax_value(point)
        assert np.isfinite(v_pt) and abs(v_pt - ref) <= tol and abs(v_jx - ref) <= tol, (n, eta, v_pt, v_jx, ref)
        gt = np.maximum(1e-8, 1e-10 * np.abs(gvec))
        assert np.all(np.abs(np.asarray(dlogp(point), dtype=float) - gvec) <= gt), (n, eta)
        assert np.all(np.abs(jax_grad(point) - gvec) <= gt), (n, eta)


# -- value_point and describe() ---------------------------------------------------------------------------------
def test_value_point_and_describe(df, pmodel, bpmodel):
    for m in (pmodel, bpmodel):
        point = m.value_point(intercept=0.3, b=[0.2], sd=0.5, z=np.zeros(6))
        assert set(point) == {v.name for v in m.pymc_model.value_vars} == {"Intercept", "x", "1|g_sigma_log__", "1|g_offset"}
        with pytest.raises(ValueError, match="sigma"):
            m.value_point(intercept=0.3, b=[0.2], sd=0.5, z=np.zeros(6), sigma=1.0)
    text = pmodel.describe()
    assert "family    : bernoulli, probit link" in text
    assert "P(y_i = 1) = p_i = Phi(eta_i), the standard Normal CDF; eta is on the probit scale." in text
    assert "a_c is the probit (Phi^-1) of the probability of y = 1 at the predictor means (group effect zero)" in text
    assert "Priors (location coefficients on the probit scale)" in text
    assert "y_i log Phi(eta_i) + (1 - y_i) log Phi(-eta_i)" in text and "Stan's Phi(eta) is exactly 0 below" in text
    assert text.endswith("Deterministic: 1|g = u, probit_p = linear predictor; p = Phi(probit_p) are computed by predict()")
    assert "logit" not in text and "log-odds" not in text and "softplus" not in text
    btext = bpmodel.describe()
    assert "family    : binomial, probit link" in btext
    assert "P(success) = p_i = Phi(eta_i); k_i ~ Binomial(n_i, p_i); eta is on the probit scale." in btext
    assert "a_c is the per-trial success probability on the probit scale at the predictor means" in btext
    assert "+ k_i log Phi(eta_i) + (n_i - k_i) log Phi(-eta_i)" in btext
    assert btext.endswith("probit_p = linear predictor; p = Phi(probit_p), expected = n p and count draws are computed by predict()")
    cat = ab.bernoulli("y ~ x + z + (1 | g)", df.assign(z=np.where(df.x > 0, "hi", "lo")), PRIORS, link="probit", categorical={"z": "lo"})
    assert "Phi(a_c) is the probability at that proportion-weighted predictor" in cat.describe()
    # the logit models read exactly as before
    assert "P(y_i = 1) = p_i = expit(eta_i) = 1 / (1 + exp(-eta_i)); eta is on the log-odds scale." in ab.bernoulli(BERNOULLI, df, PRIORS).describe()


# -- predictions --------------------------------------------------------------------------------------------------
def _posterior(model, draws=400, seed=1):
    rng = np.random.default_rng(seed)
    shape = (1, draws)
    post = {"Intercept": rng.normal(0.2, 0.3, shape), "x": rng.normal(0.5, 0.2, shape), "1|g_sigma": np.exp(rng.normal(-0.5, 0.3, shape))}
    post["1|g"] = rng.normal(size=shape + (model.design.J,)) * post["1|g_sigma"][..., None]
    return az.from_dict(posterior=post, coords={"g__factor_dim": list(model.design.levels)}, dims={"1|g": ["g__factor_dim"]})


def test_predictions_apply_the_probit_inverse_link(df, pmodel):
    idata = _posterior(pmodel)
    post = idata.posterior
    a, b, u, sd = (post[k].to_numpy() for k in ("Intercept", "x", "1|g", "1|g_sigma"))
    eta = a[..., None] + b[..., None] * df.x.to_numpy() + u[..., pmodel.design.group_index]
    lin = pmodel.predict(idata, kind="linear")
    assert np.allclose(lin["eta"].to_numpy(), eta, atol=1e-12) and lin.attrs["link"] == "probit" and lin.attrs["prediction_scale"] == "linear"
    mean = pmodel.predict(idata, kind="mean")
    assert set(mean.data_vars) == {"p"} and np.allclose(mean["p"].to_numpy(), ndtr(eta), atol=1e-12)
    resp = pmodel.predict(idata, kind="response", seed=0)
    y = resp["y"].to_numpy()
    assert set(np.unique(y)) <= {0, 1} and np.issubdtype(y.dtype, np.integer) and abs(y.mean() - ndtr(eta).mean()) < 0.02
    nog = pmodel.predict(idata, kind="mean", include_group_effects=False)
    assert np.allclose(nog["p"].to_numpy(), ndtr(a[..., None] + b[..., None] * df.x.to_numpy()), atol=1e-12)
    # new rows: an observed level, and unseen levels under both policies, with the effects on the probit scale
    new = pd.DataFrame({"x": [0.0, 1.0, 0.0, 0.0], "g": ["l0", "NEW_A", "NEW_B", "NEW_A"]})
    with pytest.raises(ab.FormulaError, match="not seen in training"):
        pmodel.predict(idata, newdata=new, kind="mean")
    lv = list(pmodel.design.levels)
    res = pmodel.predict(idata, newdata=new, kind="mean", new_groups="resample_observed", seed=4)
    v = res["p"].to_numpy()
    assert res.attrs["new_levels"] == ["NEW_A", "NEW_B"] and np.allclose(v[..., 0], ndtr(a + u[..., lv.index("l0")]), atol=1e-12)
    cand = ndtr(a[..., None] + b[..., None] + u)                        # row 1: x = 1 and one observed level's effect
    assert np.isclose(v[..., 1][..., None], cand, atol=1e-12).any(-1).all()
    assert np.allclose(ndtri(v[..., 1]) - b, ndtri(v[..., 3]), atol=1e-8)   # the repeated label shares its draw
    pop = pmodel.predict(idata, newdata=new, kind="response", new_groups="population", seed=4)
    dev = ndtri(pop["p"][..., 2].to_numpy()) - a                          # a fresh Normal(0, sd) effect, read back through Phi^-1
    assert abs(dev.std() - np.sqrt(np.mean(sd ** 2))) < 0.2 * np.sqrt(np.mean(sd ** 2))
    assert set(np.unique(pop["y"].to_numpy())) <= {0, 1}


def test_binomial_probit_predictions(df, bpmodel):
    idata = _posterior(bpmodel)
    post = idata.posterior
    a, b, u = (post[k].to_numpy() for k in ("Intercept", "x", "1|g"))
    p = ndtr(a[..., None] + b[..., None] * df.x.to_numpy() + u[..., bpmodel.design.group_index])
    prob = bpmodel.predict(idata, kind="probability")
    assert set(prob.data_vars) == {"p"} and np.allclose(prob["p"].to_numpy(), p, atol=1e-12)
    mean = bpmodel.predict(idata, kind="mean")
    assert set(mean.data_vars) == {"p", "expected"} and np.allclose(mean["expected"].to_numpy(), p * df.n.to_numpy(), atol=1e-10)
    resp = bpmodel.predict(idata, kind="response", seed=2)
    y = resp["y"].to_numpy()
    assert np.issubdtype(y.dtype, np.integer) and np.all(y >= 0) and np.all(y <= df.n.to_numpy())
    assert abs(y.mean() - (p * df.n.to_numpy()).mean()) < 0.05
    new = pd.DataFrame({"x": [0.0, 2.0], "n": [10, 0], "g": ["l1", "l2"]})
    out = bpmodel.predict(idata, newdata=new, kind="mean")
    lv = list(bpmodel.design.levels)
    assert np.allclose(out["p"][..., 0].to_numpy(), ndtr(a + u[..., lv.index("l1")]), atol=1e-12)
    assert np.all(out["expected"][..., 1].to_numpy() == 0)
    with pytest.raises(ab.FormulaError, match="lacks the trials column"):
        bpmodel.predict(idata, newdata=new.drop(columns="n"), kind="mean")


# -- prior predictive and the fit record ------------------------------------------------------------------------
def test_prior_predictive_and_fit_record(df, pmodel, bpmodel, monkeypatch):
    pp = pmodel.prior_predictive(draws=25, seed=3)
    y = pp.prior_predictive[pmodel.names["y"]].values
    assert y.shape == (1, 25, len(df)) and set(np.unique(y)) <= {0, 1}
    k = bpmodel.prior_predictive(draws=25, seed=3).prior_predictive[bpmodel.names["y"]].values
    assert k.shape == (1, 25, len(df)) and np.all(k >= 0) and np.all(k <= df.n.to_numpy())
    import autobayes.session as session
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})

    def record(m):
        fit = SimpleNamespace(target_identity=m.target_identity, idata=None, backend="synthetic", device_requested="cpu",
                              device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=0, draws=0,
                              tune=0, target_accept=0.9, seed=1, wall_seconds=0, notes=["synthetic record; not a fit"])
        return ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})

    rec = record(pmodel)
    json.dumps(rec)
    assert rec["model"]["family"] == "bernoulli" and rec["model"]["link"] == "probit" and "sigma" not in rec["model"]["priors"]
    assert rec["model"]["intercept_prior_applies_to"].endswith(", probit scale")
    brec = record(bpmodel)
    assert brec["model"]["link"] == "probit" and brec["model"]["trials"] == "n"
    logit_rec = record(ab.bernoulli(BERNOULLI, df, PRIORS))
    assert logit_rec["model"]["link"] == "logit"
    with pytest.raises(ValueError, match="different models"):
        ab.compare_records(rec, logit_rec)


# -- the numerics themselves, on both paths --------------------------------------------------------------------
# The branch boundaries of log_normal_cdf (-20, -1) and cloglog_log_p1 (-20) are hit exactly and approached from both
# sides: PyTensor's Clip gradient and its JAX lowering (a nested where) both pass the whole gradient at a tie.
getcontext().prec = 50


def _cloglog_exact(eta):
    """log P(y = 1), log P(y = 0) and their eta-derivatives under the cloglog link, at 50 digits."""
    e = Decimal(float(eta)).exp()
    q = (-e).exp()                                             # P(y = 0) = exp(-exp(eta))
    return float((1 - q).ln()), float(-e), float(e * q / (1 - q)), float(-e)


@pytest.mark.parametrize("mode", ["FAST_RUN", "JAX"])
def test_log_normal_cdf_matches_scipy_on_both_paths(mode):
    if mode == "JAX":
        pytest.importorskip("jax")
    z = pt.dvector("z")
    f = pytensor.function([z], [lk.log_normal_cdf(z), pt.grad(lk.log_normal_cdf(z).sum(), z)], mode=mode)
    grid = np.array(ETAS + [-20.0 - 1e-9, -20.0, -19.999, -1.0 - 1e-9, -1.0, -0.999, -37.5, -38.0, 8.25, 9.0, -1e4, 1e4])
    v, g = (np.asarray(a, dtype=float) for a in f(grid))
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(g))
    np.testing.assert_allclose(v, log_ndtr(grid), rtol=1e-13, atol=1e-300)
    np.testing.assert_allclose(g, _mills(grid), rtol=1e-11, atol=1e-300)


@pytest.mark.parametrize("mode", ["FAST_RUN", "JAX"])
def test_cloglog_numerics_are_exact_on_both_paths(mode):
    if mode == "JAX":
        pytest.importorskip("jax")
    eta = pt.dvector("eta")
    p1, p0 = lk.cloglog_log_p1(eta), lk.cloglog_log_p0(eta)
    f = pytensor.function([eta], [p1, pt.grad(p1.sum(), eta), p0, pt.grad(p0.sum(), eta)], mode=mode)
    grid = np.array([e for e in ETAS if abs(e) <= 40] + [-20.0 - 1e-9, -20.0, -19.999, 2.7, 3.0, 3.7, 20.0])
    v1, g1, v0, g0 = (np.asarray(a, dtype=float) for a in f(grid))
    for i, e in enumerate(grid):
        x1, x0, d1, d0 = _cloglog_exact(e)
        for got, want in ((v1[i], x1), (g1[i], d1), (v0[i], x0), (g0[i], d0)):
            assert np.isfinite(got) and abs(got - want) <= 1e-13 * max(1.0, abs(want)), (mode, e, got, want)


def test_cloglog_likelihood_object_and_inverse_link():
    import pymc as pm

    eta = np.array([-3.0, -0.5, 0.0, 1.0, 2.5])
    dist = lk._bernoulli_link_dist("cloglog").dist(cloglog_p=eta)
    for y in (0, 1):
        got = pytensor.function([], pm.logp(dist, np.full(5, y)))()
        want = np.array([_cloglog_exact(e)[1 - y] for e in eta])
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-14)
    draws = pm.draw(dist, draws=2000, random_seed=1)
    assert draws.shape == (2000, 5) and set(np.unique(draws)) <= {0, 1}
    assert np.all(np.abs(draws.mean(0) - inverse_cloglog(eta)) < 4 * np.sqrt(0.25 / 2000))
    exact = [float(1 - (-Decimal(e).exp()).exp()) for e in (-40.0, 0.0, 3.0)]
    np.testing.assert_allclose(inverse_cloglog(np.array([-40.0, 0.0, 3.0])), exact, rtol=1e-14)
    assert INVERSE_LINKS["cloglog"] is inverse_cloglog


# -- the cloglog link in models (P6; the oracle cases bernoulli_cloglog and binomial_cloglog compare them with brms) --------
def _cloglog_reference(model, nat, *, jacobian=True):
    """The target and its gradient in PyMC's value space under the cloglog link: the row terms and their eta-derivatives
    from the 50-digit ``_cloglog_exact`` (log C(n, k) from the 60-digit binomial reference), the priors of PRIORS,
    z ~ Normal(0, 1), plus log(sd) with the Jacobian."""
    d = model.design
    eta, y = _eta(model, nat), d.y
    n = np.ones_like(y) if d.trials is None else d.trials
    rows = [_cloglog_exact(e) for e in eta]                  # log P(y = 1), log P(y = 0) and their eta-derivatives
    value = sum(yi * r[0] + (ni - yi) * r[1] for yi, ni, r in zip(y, n, rows))
    if d.trials is not None:
        value += sum(float(log_binom_dec(int(ni), int(yi))) for ni, yi in zip(n, y))
    value += (norm.logpdf(nat["intercept"], 0, 2.5) + norm.logpdf(nat["b"], 0, 1.5).sum() + halfnorm.logpdf(nat["sd"], scale=1)
              + norm.logpdf(nat["z"]).sum() + (np.log(nat["sd"]) if jacobian else 0.0))
    dl = np.array([yi * r[2] + (ni - yi) * r[3] for yi, ni, r in zip(y, n, rows)])
    a, sd, z = nat["intercept"], nat["sd"], nat["z"]
    grad = {"Intercept": dl.sum() - a / 2.5 ** 2, "x": (dl * d.X_model[:, 0]).sum() - nat["b"][0] / 1.5 ** 2,
            "1|g_sigma_log__": (dl * z[d.group_index]).sum() * sd - sd ** 2 + 1.0,
            "1|g_offset": np.array([dl[d.group_index == j].sum() * sd - z[j] for j in range(d.J)])}
    return value, grad


@pytest.mark.parametrize("which", ["bernoulli", "binomial"])
def test_cloglog_models_density_describe_and_predictions(which, df):
    """``link="cloglog"`` builds the model (P6 lifted the park, D18): its density and gradient on both paths equal the
    50-digit reference up to linear predictors of about 6, where brms's 1 - inv_cloglog(eta) has lost every digit;
    ``describe()`` writes the link; ``predict()`` applies ``1 - exp(-exp(eta))``; prior-predictive draws are counts."""
    pytest.importorskip("jax")
    ctor, formula = (ab.bernoulli, BERNOULLI) if which == "bernoulli" else (ab.binomial, BINOMIAL)
    m = ctor(formula, df, PRIORS, link="cloglog")
    assert m.link == "cloglog" and m.family_spec is get_family(m.family, "cloglog")
    assert m.names["linpred"] == "cloglog_p" and m.names["mean"] == "p" and "sigma" not in m.names
    assert {v.name for v in m.pymc_model.deterministics} == {"1|g", "cloglog_p"}
    assert m.target_identity != ctor(formula, df, PRIORS, link="probit").target_identity
    names, logp, dlogp, jax_value, jax_grad = _both_paths(m)
    rng = np.random.default_rng(11)
    for a in (-4.0, -1.0, 0.0, 1.5, 3.0):
        nat = dict(intercept=a, b=np.array([0.6]), sd=0.7, z=rng.normal(size=m.design.J))
        point = m.value_point(**nat)
        ref, g = _cloglog_reference(m, nat)
        gvec = np.concatenate([np.atleast_1d(g[n]) for n in names])
        tol = 1e-8 * max(1.0, abs(ref))
        assert abs(float(logp(point)) - ref) < tol and abs(jax_value(point) - ref) < tol, (a, float(logp(point)), ref)
        np.testing.assert_allclose(np.asarray(dlogp(point), dtype=float), gvec, rtol=1e-9, atol=1e-9, err_msg=f"intercept {a}")
        np.testing.assert_allclose(jax_grad(point), gvec, rtol=1e-9, atol=1e-9, err_msg=f"JAX, intercept {a}")
    text = m.describe()
    assert f"family    : {which}, cloglog link" in text and "Priors (location coefficients on the cloglog scale)" in text
    assert "eta is on the complementary log-log scale." in text and "log-odds" not in text and "Phi" not in text
    if which == "bernoulli":
        assert "P(y_i = 1) = p_i = 1 - exp(-exp(eta_i)), the inverse complementary log-log; eta" in text
        assert "a_c is the complementary log-log (log(-log(1 - p))) of the probability of y = 1 at the predictor means" in text
        assert "log p(y_i | eta_i) = y_i log(1 - exp(-exp(eta_i))) - (1 - y_i) exp(eta_i)" in text
        assert text.endswith("Deterministic: 1|g = u, cloglog_p = linear predictor; p = 1 - exp(-exp(cloglog_p)) are computed by predict()")
    else:
        assert "P(success) = p_i = 1 - exp(-exp(eta_i)); k_i ~ Binomial(n_i, p_i)" in text
        assert "log p(k_i | n_i, eta_i) = log C(n_i, k_i) + k_i log(1 - exp(-exp(eta_i))) - (n_i - k_i) exp(eta_i)" in text
    idata = _posterior(m)
    post = idata.posterior
    a, b, u = (post[k].to_numpy() for k in ("Intercept", "x", "1|g"))
    eta = a[..., None] + b[..., None] * df.x.to_numpy() + u[..., m.design.group_index]
    lin = m.predict(idata, kind="linear")
    assert np.allclose(lin["eta"].to_numpy(), eta, atol=1e-12) and lin.attrs["link"] == "cloglog"
    mean = m.predict(idata, kind="mean")
    assert np.allclose(mean["p"].to_numpy(), -np.expm1(-np.exp(eta)), atol=1e-12)
    if which == "binomial":
        assert np.allclose(mean["expected"].to_numpy(), -np.expm1(-np.exp(eta)) * df.n.to_numpy(), atol=1e-10)
    y = m.prior_predictive(draws=25, seed=3).prior_predictive[m.names["y"]].values
    upper = 1 if which == "bernoulli" else df.n.to_numpy()
    assert y.shape == (1, 25, len(df)) and np.all(y >= 0) and np.all(y <= upper)
