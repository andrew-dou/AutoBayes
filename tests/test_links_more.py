"""brms-parity E06 (docs/links.md): more links for the Tier A families.

``poisson("sqrt")``, ``negbinomial("identity")``, ``Gamma("inverse")``, ``Gamma("identity")``, ``Beta("probit")`` and
``Beta("cloglog")`` are offered (the oracle cases poisson_sqrt, negbinomial_identity, gamma_inverse, gamma_identity, beta_probit
and beta_cloglog compare them with brms); ``cumulative("cloglog")`` is built but withheld (divergence D132). Here: the registry
and the refusals (the link= message, the constructs no frozen case verifies under a link, D133, and non-positive means, D134);
each likelihood row on both paths against references written from the published densities (50-digit Decimal where brms's
forms lose digits), finite with a finite gradient where the forms say so; the models' density and gradient, describe(),
value_point, predictions on fitted and new rows, prior-predictive draws, log_lik, brms's default priors under each link and the
fit record; and the withheld cumulative cloglog link against its frozen case and a 50-digit reference. Nothing samples.
"""
import math
import re
from decimal import Decimal, getcontext
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
from scipy import stats
from scipy.special import betaln, gammaln, log_ndtr, ndtr

import autobayes as ab
from autobayes import likelihoods as lk
from autobayes.families import FAMILIES, get_family
from autobayes.families import pointwise as pw
from autobayes.families.cumulative import _cloglog as cumulative_cloglog_spec
from autobayes.families.links import COMPLEMENTS, INVERSE_LINKS, positive_mean, square_mean
from autobayes.families.spec import FamilySpec

getcontext().prec = 50

# (family, link) -> (formula, auxiliary prior), on FRAME
MODELS = {
    ("poisson", "sqrt"): ("cnt ~ x + (1 | g)", {}),
    ("negbinomial", "identity"): ("ynb ~ x + offset(eoff) + (1 | g)", {"shape": ab.Gamma(2, 0.5)}),
    ("gamma", "inverse"): ("ypos ~ x + offset(eoff) + (1 | g)", {"shape": ab.Gamma(2, 0.5)}),
    ("gamma", "identity"): ("ypos ~ x + offset(eoff) + (1 | g)", {"shape": ab.Gamma(2, 0.5)}),
    ("beta", "probit"): ("prop ~ x + (1 | g)", {"phi": ab.Gamma(2, 0.5)}),
    ("beta", "cloglog"): ("prop ~ x + (1 | g)", {"phi": ab.Gamma(2, 0.5)}),
}
PARENT = {("poisson", "sqrt"): "sqrt_mu", ("negbinomial", "identity"): "mu", ("gamma", "inverse"): "inverse_mu",
          ("gamma", "identity"): "mu", ("beta", "probit"): "probit_mu", ("beta", "cloglog"): "cloglog_mu"}
BASE = dict(intercept=ab.Normal(0, 1), b=ab.Normal(0, 0.5), sd=ab.HalfNormal(0.5))
EULER = 0.5772156649015329


def frame(n=48, seed=5):
    rng = np.random.default_rng(seed)
    g = np.array([f"l{i % 6}" for i in range(n)], dtype=object)
    rng.shuffle(g)
    return pd.DataFrame({"x": rng.normal(size=n), "g": g, "g2": np.roll(g, 3), "eoff": rng.uniform(4.0, 5.0, n),
                         "w": rng.uniform(0.5, 2.0, n), "ordx": rng.integers(1, 4, n), "cnt": rng.poisson(2.0, n),
                         "ynb": rng.negative_binomial(3, 0.4, n), "ypos": rng.gamma(4.0, 0.06, n), "prop": rng.uniform(0.05, 0.95, n),
                         "ord": np.r_[[1, 2, 3, 4], rng.integers(1, 5, n - 4)]})


FRAME = frame()


def build(family, link, formula=None, data=None, priors=None, **kwargs):
    f, aux = MODELS[(family, link)]
    return getattr(ab, family)(formula or f, FRAME if data is None else data, priors or ab.Priors(**BASE, **aux), link=link, **kwargs)


@pytest.fixture(scope="module")
def models():
    return {key: build(*key) for key in MODELS}


# -- the registry and the refusals ----------------------------------------------------------------------------------------
def test_registry_offers_each_link_once():
    offered = {"poisson": {"sqrt"}, "negbinomial": {"identity"}, "gamma": {"inverse", "identity"}, "beta": {"probit", "cloglog"}}
    for family, links in offered.items():
        spec = FAMILIES[family]
        assert set(spec.links) == links and not spec.parked_links
        for link in links:
            other = get_family(family, link)
            assert isinstance(other, FamilySpec) and other is get_family(family, link) and other.name == family and other.link == link
            assert other.parent == PARENT[(family, link)] and other.with_link(spec.link) is spec and other.with_link(None) is other
            assert set(other.links) == (links - {link}) | {spec.link}
            assert all(other.with_link(o).with_link(link) is other for o in other.links)
            assert other.check_formula is not None and other.addition_terms == frozenset() and not other.monotonic
            assert other.distributional_link is None and other.log_likelihood is not None
        assert FAMILIES[family] is spec and spec.check_formula is None                      # the registered spec is unchanged
    assert get_family("negbinomial", "identity").mean_name == get_family("gamma", "inverse").mean_name == "mu"
    cumulative = FAMILIES["cumulative"]                                    # E06: cloglog built, withheld with its numbers (D132)
    assert set(cumulative.links) == {"probit"} and set(cumulative.parked_links) == {"probit_approx", "cloglog", "cauchit"}
    assert "5.9e-7 relative off the 50-digit value" in cumulative.parked_links["cloglog"]
    with pytest.raises(ValueError, match=r"the cloglog link of the cumulative family is withheld: it is implemented \(E06\), but brms's "
                                         r"own log density of the frozen case cumulative_cloglog is 5\.9e-7 relative off"):
        get_family("cumulative", "cloglog")
    with pytest.raises(ValueError, match="the cloglog link of the cumulative family is withheld"):
        ab.cumulative("ord ~ x", FRAME, ab.Priors(intercept=ab.Normal(0, 3), b=ab.Normal(0, 1)), link="cloglog")
    for family in ("sratio", "cratio", "acat"):                            # the other ordinal families keep every park (D38)
        assert "cloglog" in FAMILIES[family].parked_links


def test_other_links_are_refused_naming_each_familys_links():
    priors = ab.Priors(**BASE)
    with pytest.raises(ValueError, match=r"link must be one of \['log', 'sqrt'\] for the poisson family, got 'identity'$"):
        ab.poisson("cnt ~ x + (1 | g)", FRAME, priors, link="identity")
    with pytest.raises(ValueError, match=r"link must be one of \['identity', 'log'\] for the negbinomial family, got 'sqrt'$"):
        ab.negbinomial("ynb ~ x + (1 | g)", FRAME, ab.Priors(**BASE, shape=ab.Gamma(2, 0.5)), link="sqrt")
    with pytest.raises(ValueError, match=r"link must be one of \['identity', 'inverse', 'log'\] for the gamma family, got 'softplus'"):
        ab.gamma("ypos ~ x + (1 | g)", FRAME, ab.Priors(**BASE, shape=ab.Gamma(2, 0.5)), link="softplus")
    with pytest.raises(ValueError, match=r"link must be one of \['cloglog', 'logit', 'probit'\] for the beta family, got 'cauchit'"):
        ab.beta("prop ~ x + (1 | g)", FRAME, ab.Priors(**BASE, phi=ab.Gamma(2, 0.5)), link="cauchit")
    with pytest.raises(ValueError, match=r"link must be one of \['identity', 'inverse', 'log'\] for the gamma family"):
        ab.default_priors("ypos ~ x", FRAME, family="gamma", link="sqrt")
    with pytest.raises(ValueError, match=r"the lognormal family has the identity link only; link='log' is not available \(link= applies "
                                         r"to the families with more than one link: poisson \['log', 'sqrt'\], negbinomial \['identity', 'log'\], "
                                         r"gamma \['identity', 'inverse', 'log'\], beta \['cloglog', 'logit', 'probit'\], bernoulli "
                                         r"\['cloglog', 'logit', 'probit'\], binomial \['cloglog', 'logit', 'probit'\], cumulative "
                                         r"\['logit', 'probit'\]\)"):
        ab.Model("ypos ~ x + (1 | g)", FRAME, ab.Priors(**BASE, sigma=ab.HalfNormal(1)), family="lognormal", link="log")
    assert ab.poisson("cnt ~ x + (1 | g)", FRAME, priors, link="log").family_spec is FAMILIES["poisson"]    # the default, explicitly


CONSTRUCTS = [   # (what, formula maker, message fragment)
    ("mo()", lambda f: f.replace(" + (1 | g)", " + mo(ordx) + (1 | g)"), "the monotonic term 'mo(ordx)'"),
    ("weights()", lambda f: f.replace(" ~", " | weights(w) ~", 1), "the addition term 'weights(w)'"),
    ("gp()", lambda f: f.replace(" + (1 | g)", " + gp(x)"), "the Gaussian process 'gp(x)'"),
    ("mm()", lambda f: f.replace("(1 | g)", "(1 | mm(g, g2))"), "the group term '(1 | mm(g, g2))'"),
]


@pytest.mark.parametrize("key", list(MODELS), ids=lambda k: f"{k[0]}-{k[1]}")
@pytest.mark.parametrize("construct", [c[0] for c in CONSTRUCTS])
def test_unverified_constructs_are_refused_under_the_link(key, construct):
    """D133: under an E06 link the formula takes population- and group-level terms and offset(); the special terms and the
    addition terms are refused, naming the construct and the link (the family's own link keeps taking them)."""
    family, link = key
    _, make, needle = next(c for c in CONSTRUCTS if c[0] == construct)
    formula = make(MODELS[key][0])
    message = f"{needle} is not available under the {link} link of the {family} family: no frozen brms case verifies it"
    if family == "beta" and construct == "mm()":                     # D08 refuses mm() for the Beta family under every link
        message = "the multi-membership term '(1 | mm(g, g2))' is not available for the beta family"
    with pytest.raises(ab.FormulaError, match=re.escape(message)):
        build(family, link, formula=formula)
    with pytest.raises(ab.FormulaError, match=re.escape(message)):
        ab.default_priors(formula, FRAME, family=family, link=link)


@pytest.mark.parametrize("key", [k for k in MODELS if k[0] != "poisson"], ids=lambda k: f"{k[0]}-{k[1]}")
def test_a_distributional_formula_is_refused_under_the_link(key):
    family, link = key
    aux = "phi" if family == "beta" else "shape"
    formula = [MODELS[key][0], f"{aux} ~ x"]
    with pytest.raises(ab.FormulaError, match=re.escape(f"the distributional formula '{aux} ~ x' is not available under the {link} link "
                                                        f"of the {family} family")):
        getattr(ab, family)(formula, FRAME, ab.Priors(**BASE, **{aux: {"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1)}}), link=link)
    getattr(ab, family)(formula, FRAME, ab.Priors(**BASE, **{aux: {"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1)}}))   # own link


def test_the_term_model_composes_under_the_links():
    """What the links take: categorical predictors, interactions, transforms, varying slopes (correlated or not) and nested
    group terms enter the linear predictor as under the family's own link."""
    df = FRAME.assign(z=np.where(FRAME.x > 0, "hi", "lo"), h=np.where(np.arange(len(FRAME)) % 2, "u", "v"))
    slopes = {"g": {"Intercept": ab.HalfNormal(0.5), "x": ab.HalfNormal(0.5)}}
    for (family, link), (formula, aux) in MODELS.items():
        lhs, rest = formula.split(" ~ ")
        offset = " + offset(eoff)" if "offset" in rest else ""
        m = getattr(ab, family)(f"{lhs} ~ x * z + log(eoff){offset} + (1 + x | g)", df,
                                ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 0.3), sd=slopes, cor=ab.LKJ(2), **aux),
                                link=link, categorical={"z": "lo"})
        assert m.link == link and m.names["linpred"] == PARENT[(family, link)]
        nested = getattr(ab, family)(f"{lhs} ~ x{offset} + (1 | g/h)", df,
                                     ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 0.5),
                                               sd={"g": ab.HalfNormal(0.5), "g:h": ab.HalfNormal(0.5)}, **aux), link=link)
        assert set(nested.group_names) == {"g", "g:h"}


# -- the likelihood rows, on both paths -----------------------------------------------------------------------------------------
def _compile(build_rows, inputs, mode):
    if mode == "JAX":
        pytest.importorskip("jax")
    rows = build_rows(*inputs)
    return pytensor.function(list(inputs), [rows, pt.grad(rows.sum(), inputs[1])], mode=mode)


@pytest.mark.parametrize("mode", ["FAST_RUN", "JAX"])
def test_poisson_sqrt_rows(mode):
    yv, ev = pt.dvector("y"), pt.dvector("eta")
    f = _compile(lk.poisson_sqrt_logp, (yv, ev), mode)
    etas = [-1e3, -40.0, -3.0, -0.5, -1e-3, 1e-3, 0.5, 3.0, 40.0, 1e3]
    y = np.repeat([0.0, 1.0, 7.0, 250.0], len(etas))
    eta = np.tile(etas, 4)
    v, g = (np.asarray(a, dtype=float) for a in f(y, eta))
    for i in range(len(y)):
        e = Decimal(eta[i])
        want = (2 * Decimal(y[i]) * abs(e).ln() if y[i] > 0 else Decimal(0)) - e * e - Decimal(float(gammaln(y[i] + 1)))
        dwant = (2 * Decimal(y[i]) / e if y[i] > 0 else Decimal(0)) - 2 * e
        assert abs(v[i] - float(want)) <= 1e-13 * max(1.0, abs(float(want))), (y[i], eta[i], v[i], want)
        assert abs(g[i] - float(dwant)) <= 1e-13 * max(1.0, abs(float(dwant))), (y[i], eta[i], g[i], dwant)
    v0, g0 = (np.asarray(a, dtype=float) for a in f(np.array([0.0, 3.0]), np.array([0.0, 0.0])))
    assert v0[0] == 0.0 and g0[0] == 0.0 and v0[1] == -np.inf          # mu = 0: a zero count is certain, a positive one impossible
    np.testing.assert_allclose(pw.poisson_sqrt(y, eta), v, rtol=1e-13)
    np.testing.assert_allclose(v, stats.poisson.logpmf(y, eta ** 2), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("mode", ["FAST_RUN", "JAX"])
@pytest.mark.parametrize("which", ["gamma_inverse", "gamma_identity", "negbinomial_identity"])
def test_identity_and_inverse_rows(which, mode):
    """The Gamma rows against SciPy's Gamma density with scale mu / shape, the negative binomial's against SciPy's NB(shape, shape /
    (shape + mu)); a non-positive linear predictor gives -inf with a zero (finite) gradient, where brms's Stan rejects."""
    yv, ev, av = pt.dvector("y"), pt.dvector("eta"), pt.dvector("alpha")
    if which == "negbinomial_identity":
        logp, y = lk.negbinomial_identity_logp, np.array([0.0, 1.0, 4.0, 30.0, 0.0, 2.0])
    else:
        logp, y = lk.gamma_link_logp(which.split("_")[1]), np.array([0.01, 0.4, 1.0, 3.0, 25.0, 0.2])
    f = _compile(logp, (yv, ev, av), mode)
    alpha = np.array([0.3, 2.0, 7.5, 50.0, 1.0, 1e4])
    eta = np.array([0.05, 0.8, 2.5, 9.0, 120.0, 4.0])
    v, g = (np.asarray(a, dtype=float) for a in f(y, eta, alpha))
    if which == "negbinomial_identity":
        want = stats.nbinom.logpmf(y, alpha, alpha / (alpha + eta))
        num = y / eta - (y + alpha) / (alpha + eta)                           # d/dmu of log NB2(y | mu, shape)
        np.testing.assert_allclose(pw.neg_binomial_2_identity(y, eta, alpha), want, rtol=1e-11)
    else:
        mu = 1.0 / eta if which == "gamma_inverse" else eta
        want = stats.gamma.logpdf(y, alpha, scale=mu / alpha)
        rate = alpha * eta if which == "gamma_inverse" else alpha / eta
        # d/deta of shape log(rate) - rate y: shape / eta - shape y (inverse), -shape / eta + shape y / eta^2 (identity)
        num = alpha / eta - alpha * y if which == "gamma_inverse" else -alpha / eta + alpha * y / eta ** 2
        assert np.all(rate > 0)
        np.testing.assert_allclose(pw.gamma_link(y, eta, alpha, which.split("_")[1]), want, rtol=1e-12)
    np.testing.assert_allclose(v, want, rtol=1e-11)
    np.testing.assert_allclose(g, num, rtol=1e-12)
    bad = np.array([0.0, -1e-9, -3.0])
    vb, gb = (np.asarray(a, dtype=float) for a in f(y[:3], bad, alpha[:3]))
    assert np.all(vb == -np.inf) and np.all(gb == 0.0)


def _beta_reference(y, eta, phi, link):
    """log Beta(y | mu phi, (1 - mu) phi) from SciPy: mu and 1 - mu from log_ndtr (probit) or exp(eta), exp(-exp(eta)) (cloglog),
    SciPy's betaln for shape parameters it can hold, and for one that has underflowed (cloglog, eta above about 6.6) the limit
    lgamma(b) -> -log b, log b = log phi - exp(eta)."""
    if link == "probit":
        log_mu, log_cm = log_ndtr(eta), log_ndtr(-eta)
    else:
        log_mu = float(Decimal(1 - (-Decimal(eta).exp()).exp()).ln()) if eta > -30 else eta - 0.5 * math.exp(eta)
        log_cm = -math.exp(min(eta, 700.0))
    a, b = math.exp(log_mu) * phi, math.exp(log_cm) * phi
    if b > 1e-300 and a > 1e-300:
        return (a - 1) * math.log(y) + (b - 1) * math.log1p(-y) - betaln(a, b)
    log_b = math.log(phi) + log_cm                                             # only b underflows on this grid
    return (a - 1) * math.log(y) + (b - 1) * math.log1p(-y) - (gammaln(a) + (-log_b - EULER * b) - gammaln(a + b))


BETA_ETAS = {"probit": [-45.0, -38.5, -37.0, -8.3, -3.0, 0.0, 0.7, 3.0, 8.3, 30.0, 37.0, 38.5, 45.0],
             "cloglog": [-800.0, -40.0, -21.0, -19.0, -3.0, 0.0, 1.5, 3.0, 3.6, 5.0, 6.6, 6.7, 8.0, 20.0, 600.0]}


@pytest.mark.parametrize("mode", ["FAST_RUN", "JAX"])
@pytest.mark.parametrize("link", ["probit", "cloglog"])
def test_beta_rows_are_finite_and_exact_on_both_paths(link, mode):
    """Finite with a finite gradient on the whole grid, past where 1 - mu underflows (probit 38, cloglog 6.6) and where brms's 1 -
    mu by subtraction is 0 (8.25, 3.6); within 1e-12 relative of the reference; the gradient equals a central difference of the
    reference where it is smooth and the JAX gradient everywhere."""
    yv, ev, pv = pt.dvector("y"), pt.dvector("eta"), pt.dvector("phi")
    f = _compile(lk.beta_link_logp(link), (yv, ev, pv), mode)
    etas = BETA_ETAS[link]
    grid = [(y, e, p) for y in (0.02, 0.5, 0.97) for e in etas for p in (0.5, 30.0)]
    y, eta, phi = (np.array(c) for c in zip(*grid))
    v, g = (np.asarray(a, dtype=float) for a in f(y, eta, phi))
    assert np.isfinite(v).all() and np.isfinite(g).all()
    for i in range(len(y)):
        want = _beta_reference(y[i], eta[i], phi[i], link)
        assert abs(v[i] - want) <= 1e-12 * max(1.0, abs(want)), (link, y[i], eta[i], phi[i], v[i], want)
        if abs(eta[i]) < 25 and not (link == "cloglog" and 6.5 < eta[i] < 6.8):
            h = 1e-6
            num = (_beta_reference(y[i], eta[i] + h, phi[i], link) - _beta_reference(y[i], eta[i] - h, phi[i], link)) / (2 * h)
            assert abs(g[i] - num) <= 1e-5 * max(1.0, abs(num)), (link, y[i], eta[i], phi[i], g[i], num)
    np.testing.assert_allclose(pw.beta_link(y, eta, phi, link), v, rtol=1e-13, atol=1e-13)


def test_beta_rows_continue_across_the_underflow():
    """Under cloglog, 1 - mu underflows at eta of about 6.62: the density on either side of the switch to the logarithm of the
    shape parameter is one smooth curve (here, its second differences are small)."""
    eta = np.linspace(6.55, 6.70, 31)
    yv, ev, pv = pt.dvector("y"), pt.dvector("eta"), pt.dvector("phi")
    f = pytensor.function([yv, ev, pv], lk.beta_link_logp("cloglog")(yv, ev, pv))
    v = np.asarray(f(np.full(31, 0.6), eta, np.full(31, 3.0)), dtype=float)
    assert np.isfinite(v).all() and np.all(np.diff(v) < 0) and np.max(np.abs(np.diff(v, 2))) < 1.0


# -- the cumulative cloglog rows (withheld link, D132) --------------------------------------------------------------------------
def _dec_neg_expm1(t):
    """1 - exp(-t) for t >= 0, at 50 digits, by its series below 1e-6 (where 1 - exp(-t) would cancel)."""
    if t < Decimal("1e-6"):
        term, total, k = t, Decimal(0), 1
        while abs(term) > Decimal("1e-80") * abs(t):
            total += term
            k += 1
            term = -term * t / k
        return total
    return 1 - (-t).exp()


def _dec_log_mass(lo, hi):
    """log(F(hi) - F(lo)), F(x) = 1 - exp(-exp(x)), lo < hi, at 50 digits: -exp(lo) + log(1 - exp(-(exp(hi) - exp(lo))))."""
    lo, hi = Decimal(lo), Decimal(hi)
    return -lo.exp() + _dec_neg_expm1(hi.exp() - lo.exp()).ln()


def _dec_log_f(x):
    return _dec_neg_expm1(Decimal(x).exp()).ln()


MASS_PAIRS = [(-900.0, -800.0), (-40.0, -39.999), (-25.0, -20.000001), (-21.0, -19.0), (-3.0, -2.9), (-1.0, 1e-12), (0.0, 1e-9),
              (0.5, 2.0), (2.0, 3.6), (3.0, 3.0000001), (5.0, 9.0), (6.0, 6.5), (-50.0, 40.0), (650.0, 699.0), (699.5, 700.5),
              (705.0, 800.0)]


@pytest.mark.parametrize("mode", ["FAST_RUN", "JAX"])
def test_cloglog_log_mass_is_exact_and_finite_on_both_paths(mode):
    if mode == "JAX":
        pytest.importorskip("jax")
    lo, hi = pt.dvector("lo"), pt.dvector("hi")
    m = lk.cloglog_log_mass(lo, hi)
    f = pytensor.function([lo, hi], [m, pt.grad(m.sum(), lo), pt.grad(m.sum(), hi)], mode=mode)
    a, b = (np.array(c) for c in zip(*MASS_PAIRS))
    v, gl, gh = (np.asarray(x, dtype=float) for x in f(a, b))
    assert np.isfinite(v).all() and np.isfinite(gl).all() and np.isfinite(gh).all()
    for i, (l, u) in enumerate(MASS_PAIRS):
        if u > 700:                                                    # beyond exp's cap: -exp(min(lower, 700)), finite
            assert v[i] == -math.exp(min(l, 700.0)) and gh[i] == 0.0
            continue
        want = _dec_log_mass(l, u)
        assert abs(v[i] - float(want)) <= 1e-13 * max(1.0, abs(float(want))), (l, u, v[i], want)
        h = Decimal("1e-20")
        for got, d in ((gl[i], (_dec_log_mass(Decimal(l) + h, u) - _dec_log_mass(Decimal(l) - h, u)) / (2 * h)),
                       (gh[i], (_dec_log_mass(l, Decimal(u) + h) - _dec_log_mass(l, Decimal(u) - h)) / (2 * h))):
            assert abs(got - float(d)) <= 1e-9 * max(1.0, abs(float(d))), (l, u, got, d)


@pytest.mark.parametrize("mode", ["FAST_RUN", "JAX"])
def test_cumulative_cloglog_rows_and_probabilities(mode):
    """Every category's row against the 50-digit reference, finite with a finite gradient at |eta| = 1000; the NumPy log
    probabilities (predict(), log_lik) equal the rows and their exponentials sum to one."""
    if mode == "JAX":
        pytest.importorskip("jax")
    tau = np.array([-1.2, 0.3, 2.0])
    etas = [-1000.0, -40.0, -3.0, -0.5, 0.0, 0.7, 3.0, 40.0, 1000.0]
    y = np.repeat(np.arange(1, 5, dtype=np.int64), len(etas))
    eta = np.tile(etas, 4)
    yv, ev, tv = pt.lvector("y"), pt.dvector("eta"), pt.dmatrix("tau")
    rows = lk.ordinal_logp_rows("cumulative_cloglog", yv, lk.ordinal_gaps(ev, tv, False))
    f = pytensor.function([yv, ev, tv], [rows, pt.grad(rows.sum(), ev), pt.grad(rows.sum(), tv)], mode=mode)
    taus = np.tile(tau, (len(eta), 1))
    v, ge, gt = (np.asarray(x, dtype=float) for x in f(y, eta, taus))
    assert np.isfinite(v).all() and np.isfinite(ge).all() and np.isfinite(gt).all()
    for i in range(len(y)):
        d = [t - eta[i] for t in tau]
        k = int(y[i])
        if abs(eta[i]) > 700:
            continue
        want = _dec_log_f(d[0]) if k == 1 else (-Decimal(d[-1]).exp() if k == 4 else _dec_log_mass(d[k - 2], d[k - 1]))
        assert abs(v[i] - float(want)) <= 1e-13 * max(1.0, abs(float(want))), (k, eta[i], v[i], want)
    probs = lk.ordinal_log_probabilities("cumulative_cloglog", tau[None, :] - np.array(etas)[:, None])
    np.testing.assert_allclose(probs[np.tile(np.arange(len(etas)), 4), y - 1], v, rtol=1e-13, atol=1e-300)
    finite = np.abs(np.array(etas)) < 700
    np.testing.assert_allclose(np.exp(probs[finite]).sum(-1), 1.0, rtol=1e-14)


# -- the models ---------------------------------------------------------------------------------------------------------------
def _eta(model, nat, df=None):
    d = model.design
    df = model.data if df is None else df
    eta = nat["intercept"] - float(d.x_means @ nat["b"]) + df["x"].to_numpy() * nat["b"][0] + nat["sd"] * nat["z"][d.group_index]
    return eta + (df["eoff"].to_numpy() if model.spec.offset is not None else 0.0)


def _row_reference(family, link, y, eta, aux):
    if family == "poisson":
        return stats.poisson.logpmf(y, eta ** 2)
    if family == "negbinomial":
        return stats.nbinom.logpmf(y, aux, aux / (aux + eta))
    if family == "gamma":
        mu = 1.0 / eta if link == "inverse" else eta
        return stats.gamma.logpdf(y, aux, scale=mu / aux)
    return np.array([_beta_reference(yi, e, aux, link) for yi, e in zip(y, eta)])


def _target(model, family, link, nat):
    """The model's log density in PyMC's value space from SciPy: the likelihood rows, the priors of BASE and the auxiliary
    parameter's Gamma(2, 0.5) (rate 0.5), z ~ Normal(0, 1), and the log-scale Jacobians of sd and the auxiliary parameter."""
    aux = nat.get("aux", 1.0)
    rows = _row_reference(family, link, model.design.y.astype(float), _eta(model, nat), aux)
    out = (rows.sum() + stats.norm.logpdf(nat["intercept"], 0, 1) + stats.norm.logpdf(nat["b"], 0, 0.5).sum()
           + stats.halfnorm.logpdf(nat["sd"], scale=0.5) + stats.norm.logpdf(nat["z"]).sum() + math.log(nat["sd"]))
    if family != "poisson":
        out += stats.gamma.logpdf(aux, 2.0, scale=2.0) + math.log(aux)
    return out


# natural parameters near the data (each model's eta stays positive under the identity and inverse links: offsets in (4, 5))
NATURAL = {("poisson", "sqrt"): dict(intercept=1.3, b=np.array([0.4]), sd=0.4),
           ("negbinomial", "identity"): dict(intercept=0.0, b=np.array([0.3]), sd=0.4, aux=3.0),
           ("gamma", "inverse"): dict(intercept=-0.5, b=np.array([0.3]), sd=0.4, aux=4.0),
           ("gamma", "identity"): dict(intercept=-3.85, b=np.array([0.02]), sd=0.05, aux=4.0),
           ("beta", "probit"): dict(intercept=0.2, b=np.array([0.6]), sd=0.4, aux=8.0),
           ("beta", "cloglog"): dict(intercept=-0.3, b=np.array([0.6]), sd=0.4, aux=8.0)}


@pytest.mark.parametrize("key", list(MODELS), ids=lambda k: f"{k[0]}-{k[1]}")
def test_models_density_names_and_describe(models, key):
    family, link = key
    m = models[key]
    spec = get_family(family, link)
    assert m.family == family and m.link == link and m.family_spec is spec
    assert m.names["linpred"] == PARENT[key] and m.names["mean"] == "mu"
    assert {v.name for v in m.pymc_model.deterministics} == {"1|g", PARENT[key]}
    default = getattr(ab, family)(MODELS[key][0], FRAME, ab.Priors(**BASE, **MODELS[key][1]))
    assert m.target_identity != default.target_identity
    nat = dict(NATURAL[key], z=np.linspace(-1, 1, m.design.J))
    kwargs = {k: v for k, v in nat.items() if k != "aux"}
    if family != "poisson":
        kwargs[spec.scale_parameter] = nat["aux"]
    point = m.value_point(**kwargs)
    assert np.all(_eta(m, nat) > 0) or family in ("poisson", "beta")      # inside the support of every row
    want = _target(m, family, link, nat)
    got = float(m.pymc_model.compile_logp()(point))
    assert abs(got - want) <= 1e-10 * max(1.0, abs(want)), (got, want)
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    names = [v.name for v in m.pymc_model.value_vars]
    assert abs(float(get_jaxified_logp(m.pymc_model)([point[n] for n in names])) - want) <= 1e-10 * max(1.0, abs(want))
    text = m.describe()
    assert f"family    : {family}, {link} link" in text and f"Deterministic: 1|g = u, {PARENT[key]} = linear predictor" in text
    assert f"(brms's {'Gamma' if family == 'gamma' else 'Beta' if family == 'beta' else family}(\"{link}\"))" in text
    if spec.link_scale is not None:
        assert f"Priors (location coefficients on the {spec.link_scale} scale)" in text
    else:
        assert "Priors\n" in text
    assert "exp(eta" not in text or link == "cloglog"


def test_a_non_positive_mean_has_no_density(models):
    """D134: under the identity and inverse links a row with eta <= 0 has log density -inf (brms's Stan rejects the draw)."""
    for key in [("negbinomial", "identity"), ("gamma", "inverse"), ("gamma", "identity")]:
        m = models[key]
        spec = m.family_spec
        point = m.value_point(intercept=-30.0, b=[0.0], sd=0.4, z=np.zeros(m.design.J), **{spec.scale_parameter: 2.0})
        assert float(m.pymc_model.compile_logp()(point)) == -np.inf


def _posterior(model, key, draws=300, seed=2):
    """Synthetic draws near NATURAL[key] (the raw-scale intercept; under the Gamma identity link, which needs eta > 0 on every
    row and draw, the intercept is moved up to 0 so that the mean is about 4.5)."""
    rng = np.random.default_rng(seed)
    shape = (1, draws)
    nat = NATURAL[key]
    a0 = 0.0 if key == ("gamma", "identity") else nat["intercept"]
    post = {"Intercept": rng.normal(a0, 0.05, shape), "x": rng.normal(nat["b"][0], 0.05, shape),
            "1|g_sigma": np.exp(rng.normal(-1.5, 0.2, shape))}
    post["1|g"] = rng.normal(size=shape + (model.design.J,)) * post["1|g_sigma"][..., None]
    spec = model.family_spec
    if spec.scale_parameter is not None:
        post[spec.bambi_scale_parameter] = np.exp(rng.normal(math.log(nat["aux"]), 0.1, shape))
    return az.from_dict(posterior=post, coords={"g__factor_dim": list(model.design.levels)}, dims={"1|g": ["g__factor_dim"]})


MEANS = {"sqrt": np.square, "identity": lambda e: e, "inverse": lambda e: 1.0 / e, "probit": ndtr,
         "cloglog": lambda e: -np.expm1(-np.exp(e))}


@pytest.mark.parametrize("key", list(MODELS), ids=lambda k: f"{k[0]}-{k[1]}")
def test_predictions_apply_the_inverse_link(models, key):
    family, link = key
    m = models[key]
    idata = _posterior(m, key)
    post = idata.posterior
    a, b, u = (post[k].to_numpy() for k in ("Intercept", "x", "1|g"))
    off = FRAME.eoff.to_numpy() if m.spec.offset is not None else 0.0
    eta = a[..., None] + b[..., None] * FRAME.x.to_numpy() + u[..., m.design.group_index] + off
    lin = m.predict(idata, kind="linear")
    assert np.allclose(lin["eta"].to_numpy(), eta, atol=1e-12) and lin.attrs["link"] == link
    mean = m.predict(idata, kind="mean")
    assert set(mean.data_vars) == {"mu"} and np.allclose(mean["mu"].to_numpy(), MEANS[link](eta), rtol=1e-12, atol=1e-300)
    y = m.predict(idata, kind="response", seed=0)["y"].to_numpy()
    if family in ("poisson", "negbinomial"):
        assert np.issubdtype(y.dtype, np.integer) and np.all(y >= 0)
    elif family == "gamma":
        assert np.all(y > 0)
    else:                                     # numpy's Beta generator rounds a draw within eps of 1 to 1, as for the logit link
        assert np.all((y > 0) & (y <= 1)) and np.mean(y == 1) < 1e-3
    assert abs(y.mean() / MEANS[link](eta).mean() - 1) < 0.05
    nog = m.predict(idata, kind="mean", include_group_effects=False)
    assert np.allclose(nog["mu"].to_numpy(), MEANS[link](a[..., None] + b[..., None] * FRAME.x.to_numpy() + off), rtol=1e-12)
    new = pd.DataFrame({"x": [0.0, 1.0, 0.0], "g": ["l0", "NEW", "NEW"], "eoff": [4.5, 4.5, 4.5]})
    res = m.predict(idata, newdata=new, kind="mean", new_groups="resample_observed", seed=4)
    lv = list(m.design.levels)
    noff = 4.5 if m.spec.offset is not None else 0.0
    assert np.allclose(res["mu"][..., 0].to_numpy(), MEANS[link](a + u[..., lv.index("l0")] + noff), rtol=1e-12)
    assert res.attrs["new_levels"] == ["NEW"]
    pop = m.predict(idata, newdata=new, kind="response", new_groups="population", seed=4)
    assert pop["y"].shape == (1, 300, 3)


@pytest.mark.parametrize("key", [("negbinomial", "identity"), ("gamma", "inverse"), ("gamma", "identity")], ids=lambda k: f"{k[0]}-{k[1]}")
def test_predictions_refuse_a_non_positive_mean(models, key):
    """D134: brms's posterior_epred returns a non-positive mean unchecked and posterior_predict draws NaN there; predict()
    refuses it, naming the link and the rows (the linear predictor itself is returned)."""
    family, link = key
    m = models[key]
    idata = _posterior(m, key)
    new = pd.DataFrame({"x": [0.0, 0.0], "g": ["l0", "l1"], "eoff": [4.5, -40.0]})
    assert np.all(m.predict(idata, newdata=new, kind="linear")["eta"].to_numpy()[..., 1] < 0)
    for kind in ("mean", "response"):
        with pytest.raises(ValueError, match=rf"the {link} link gives a non-positive or nonfinite mean \(linear predictor <= 0\) for some "
                                             r"draws at 1 row\(s\), the first \[1\].*divergence D134"):
            m.predict(idata, newdata=new, kind=kind)
    with pytest.raises(ValueError, match="non-positive"):
        positive_mean(np.array([1.0, 0.0]), link, "x")
    assert np.array_equal(positive_mean(np.array([2.0, 4.0]), "inverse", "x"), [0.5, 0.25])
    with pytest.raises(ValueError, match="overflow"):
        square_mean(np.array([1e200]), "Poisson expected count")


def test_inverse_links_and_complements():
    eta = np.array([-40.0, -3.0, 0.0, 0.4, 3.0, 6.0, 30.0])
    np.testing.assert_allclose(COMPLEMENTS["probit"](eta), ndtr(-eta), rtol=1e-15)
    exact = np.array([float((-Decimal(e).exp()).exp()) for e in eta])
    np.testing.assert_allclose(COMPLEMENTS["cloglog"](eta), exact, rtol=4 * 403 * 2.2e-16, atol=0)   # exp(eta) eps relative, eta = 6
    assert np.all(COMPLEMENTS["cloglog"](eta[:-1]) > 0) and INVERSE_LINKS["cloglog"](np.array([-800.0]))[0] == 0.0


@pytest.mark.parametrize("key", list(MODELS), ids=lambda k: f"{k[0]}-{k[1]}")
def test_log_lik_prior_predictive_and_fit_record(models, key, monkeypatch):
    family, link = key
    m = models[key]
    idata = _posterior(m, key)
    ll = ab.log_lik(m, idata)
    post = idata.posterior
    eta = m.predict(idata, kind="linear")["eta"].to_numpy()
    aux = post[m.family_spec.bambi_scale_parameter].to_numpy()[..., None] if family != "poisson" else 1.0
    want = np.stack([_row_reference(family, link, m.design.y.astype(float), eta[0, k], float(np.squeeze(aux[0, k])) if family != "poisson" else 1.0)
                     for k in range(eta.shape[1])])[None]
    np.testing.assert_allclose(ll.to_numpy(), want, rtol=1e-10)
    pointwise = m.pymc_model.compile_fn(m.pymc_model.logp(vars=m.pymc_model.observed_RVs, sum=False), inputs=m.pymc_model.value_vars,
                                        point_fn=True, on_unused_input="ignore")
    spec = m.family_spec
    raw, slope = float(post["Intercept"][0, 0]), float(post["x"][0, 0])     # value_point takes the centred intercept
    kwargs = dict(intercept=raw + float(m.design.x_means[0]) * slope, b=[slope], sd=float(post["1|g_sigma"][0, 0]),
                  z=post["1|g"][0, 0].to_numpy() / float(post["1|g_sigma"][0, 0]))
    if family != "poisson":
        kwargs[spec.scale_parameter] = float(post[spec.bambi_scale_parameter][0, 0])
    rows = np.asarray(pointwise(m.value_point(**kwargs))[0], dtype=float)
    np.testing.assert_allclose(ll.to_numpy()[0, 0], rows, rtol=1e-11)
    narrow = m if family in ("poisson", "beta") else build(family, link, priors=ab.Priors(   # eta near the offset, 4.5
        intercept=ab.Normal(0, 0.2), b=ab.Normal(0, 0.05), sd=ab.HalfNormal(0.05), **MODELS[key][1]))
    y = narrow.prior_predictive(draws=20, seed=3).prior_predictive[m.names["y"]].values
    assert y.shape == (1, 20, len(FRAME)) and np.isfinite(y).all()
    if family in ("negbinomial", "gamma"):     # D134: under BASE's Normal(0, 1) intercept some prior draws have eta <= 0
        with pytest.raises(ValueError, match="prior predictive generation failed") as info:
            m.prior_predictive(draws=200, seed=3)
        assert f"the {link} link gives a non-positive mean" in str(info.value.__cause__)
    import autobayes.session as session
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=None, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=0, draws=0, tune=0,
                          target_accept=0.9, seed=1, wall_seconds=0, notes=["synthetic record; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["family"] == family and rec["model"]["link"] == link


# -- brms's default priors under each link ------------------------------------------------------------------------------------
def test_default_priors_read_the_response_through_the_link():
    """brms's def_scale_prior reads sqrt(y) (zeros as they are) under the sqrt link, 1 / y (zeros as 0.1) under the inverse link
    and y under the identity link, minus the mean offset; the probability links give 0 and 2.5 (brms 2.23's default_prior() on
    these data: student_t(3, 0.5, 2.5) for the counts 0, 0, 0, 1, 4, 9 under sqrt and identity, student_t(3, 4.2, 5.9) and
    student_t(3, 0.2, 2.5) for the positive values below under inverse and identity, and student_t(3, 0, 2.5) for Beta)."""
    counts = pd.DataFrame({"y": [0, 0, 0, 1, 4, 9], "x": [0.1, -0.3, 0.5, 1.2, -1.0, 0.2]})
    for family, link in (("poisson", "sqrt"), ("negbinomial", "identity")):
        p = ab.default_priors("y ~ x", counts, family=family, link=link)
        assert (p.intercept.name, p.intercept.params) == ("StudentT", {"nu": 3.0, "mu": 0.5, "sigma": 2.5})
    pos = pd.DataFrame({"y": [0.05, 0.2, 0.3, 4.0, 9.0, 0.01], "x": [0.1, -0.3, 0.5, 1.2, -1.0, 0.2]})
    inv = ab.default_priors("y ~ x", pos, family="gamma", link="inverse")
    ident = ab.default_priors("y ~ x", pos, family="gamma", link="identity")
    assert inv.intercept.params == {"nu": 3.0, "mu": 4.2, "sigma": 5.9} and ident.intercept.params == {"nu": 3.0, "mu": 0.2, "sigma": 2.5}
    assert inv.shape.name == "Gamma" and inv.shape.params == {"alpha": 0.01, "beta": 0.01}
    props = pd.DataFrame({"y": [0.05, 0.2, 0.3, 0.4, 0.9, 0.01], "x": [0.1, -0.3, 0.5, 1.2, -1.0, 0.2]})
    for link in ("probit", "cloglog"):
        assert ab.default_priors("y ~ x", props, family="beta", link=link).intercept.params == {"nu": 3.0, "mu": 0.0, "sigma": 2.5}
    off = pos.assign(o=[1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    assert ab.default_priors("y ~ x + offset(o)", off, family="gamma", link="inverse").intercept.params["mu"] == 4.2 - 2.0


# -- the withheld cumulative cloglog link against its frozen case (D132) -------------------------------------------------------
@pytest.fixture(scope="module")
def cumulative_bundle():
    """The frozen case cumulative_cloglog's model, built with the withheld spec (a test-only path: Model reaches a link through
    FamilySpec.with_link, patched here for this one link)."""
    from tests.oracle import gates

    original = FamilySpec.with_link

    def with_link(self, link):
        if self.name == "cumulative" and link == "cloglog":
            return cumulative_cloglog_spec()
        return original(self, link)

    mp = pytest.MonkeyPatch()
    mp.setattr(FamilySpec, "with_link", with_link)
    try:
        yield gates.Bundle("cumulative_cloglog")
    finally:
        mp.undo()


def _dec_cumulative_loglik(fixture, point):
    sd = fixture.standata
    X, Y, J, Z = sd["X"], sd["Y"].astype(int), sd["J_1"].astype(int) - 1, sd["Z_1_1"]
    means = X.mean(0)
    b = np.atleast_1d(point.pars["b"])
    thr, r = np.atleast_1d(point.pars["Intercept"]), np.atleast_1d(point.pars["r_1_1"])
    total = Decimal(0)
    for i in range(len(Y)):
        mu = sum((Decimal(X[i, k]) - Decimal(means[k])) * Decimal(b[k]) for k in range(len(b))) + Decimal(r[J[i]]) * Decimal(Z[i])
        d = [Decimal(t) - mu for t in thr]
        k = int(Y[i])
        if k == 1:
            total += _dec_neg_expm1(d[0].exp()).ln()
        elif k == len(thr) + 1:
            total += -d[-1].exp()
        else:
            total += -d[k - 2].exp() + _dec_neg_expm1(d[k - 1].exp() - d[k - 2].exp()).ln()
    return total


def test_withheld_cumulative_cloglog_against_its_frozen_case(cumulative_bundle):
    """The implementation is checked as an offered link would be: G1, G3, G4, G5 and G6 pass against brms; the likelihood equals
    a 50-digit reference at every point (written from the published definition, brms's design and the points' parameters), and
    the whole log density equals brms's wherever brms is accurate. At point 7 (0-based) brms's is 5.9e-7 relative off, because
    brms forms P(y = 4) = 1 - inv_cloglog(d) by subtraction (rows with that probability near 2e-13); which is why the link is
    withheld (D132)."""
    from tests.oracle import gates
    from tests.oracle.tolerances import LOGP_REL, rel_err

    b = cumulative_bundle
    assert b.model.link == "cloglog" and b.model.family_spec is cumulative_cloglog_spec()
    for check in (gates.check_design, gates.check_derived, gates.check_predictions, gates.check_default_prior):
        check(b)
    pytest.importorskip("jax")
    gates.check_jax(b)
    pmm = b.model.pymc_model
    lik = pmm.compile_fn(pmm.logp(vars=pmm.observed_RVs), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    brms_err = []
    for k, p, vp in b.points():
        exact = float(_dec_cumulative_loglik(b.fixture, p))
        assert rel_err(float(lik(vp)), exact) <= 1e-14, (k, float(lik(vp)), exact)
        brms_err.append(rel_err(float(b.logp(vp)), p.lp))
    assert max(brms_err[:7]) <= LOGP_REL and 5e-7 < brms_err[7] < 7e-7, brms_err
