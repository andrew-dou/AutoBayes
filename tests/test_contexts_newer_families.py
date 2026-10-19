"""E03 (docs/contexts-newer-families.md): distributional formulas on the parameters of the Tier D families that a frozen brms
case verifies (the Weibull's ``shape`` and the beta-binomial's ``phi`` under the log link, the zero-inflated negative
binomial's ``zi`` and the hurdle lognormal's ``hu`` under the logit link, the skew normal's ``alpha`` under the identity
link) and ``weights()`` for the Weibull and beta-binomial families. The term model of the formulas and what stays refused,
the priors (``Logistic`` on a ``zi`` / ``hu`` intercept), each family's density with one parameter per row written by hand
from SciPy's distributions on both paths (with ``weights()`` as well), ``value_point``, by-hand predictions with each row's
parameter (means, seeded draws, ``dpar=``), ``log_lik`` / ``loo`` / ``waic``, ``describe()``, the target identity, fit records,
prior-predictive draws, brms's default priors (the tables brms 2.23.0's ``default_prior()`` printed for
``tests/contexts_probes``), the gates on corrupted fixtures, and the post-processing functions that keep refusing these
models (D99). Nothing samples; the brms comparison is the oracle cases ``weibull_distributional_shape``,
``beta_binomial_distributional_phi``, ``zero_inflated_negbinomial_distributional_zi``, ``hurdle_lognormal_distributional_hu``,
``skew_normal_distributional_alpha``, ``weibull_weighted`` and ``beta_binomial_weighted``."""
import copy
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit, gamma as gamma_fn

import autobayes as ab
from autobayes import likelihoods as lk
from autobayes.families import FAMILIES, get_family
from autobayes.formula import parse_formula
from tests.oracle import adapter, datasets
from tests.oracle import fixture as fx
from tests.oracle import gates
from tests.oracle.adapter import OracleMismatch

DATA = datasets.tier_e(7, 40)                # ypos > 0, y real, zcnt counts, zpos >= 0 with zeros, k of size, x, w > 0, z, g
Z = {"z": "p"}
DPAR = {"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}
# family -> (response formula on DATA, the parameter with a formula, the family's other priors, their values at a point)
SPECS = {
    "weibull": ("ypos ~ x", "shape", {}, {}),
    "beta_binomial": ("k | trials(size) ~ x", "phi", {}, {}),
    "zero_inflated_negbinomial": ("zcnt ~ x", "zi", {"shape": ab.InverseGamma(0.4, 0.3)}, {"shape": 1.3}),
    "hurdle_lognormal": ("zpos ~ x", "hu", {"sigma": ab.HalfNormal(1)}, {"sigma": 0.7}),
    "skew_normal": ("y ~ x", "alpha", {"sigma": ab.HalfNormal(1)}, {"sigma": 0.8}),
}
FAMILY_NAMES = tuple(SPECS)
LINK = {"weibull": "log", "beta_binomial": "log", "zero_inflated_negbinomial": "logit", "hurdle_lognormal": "logit",
        "skew_normal": "identity"}
INVERSE = {"log": np.exp, "logit": expit, "identity": lambda eta: eta}
A, B, A_D, B_D = 0.2, 0.3, 0.1, np.array([0.4, -0.3, 0.2])       # intercepts and coefficients (dpar: w, z[q], z[r])
CASES = {"weibull": "weibull_distributional_shape", "beta_binomial": "beta_binomial_distributional_phi",
         "zero_inflated_negbinomial": "zero_inflated_negbinomial_distributional_zi",
         "hurdle_lognormal": "hurdle_lognormal_distributional_hu", "skew_normal": "skew_normal_distributional_alpha"}
PROBE = json.loads((Path(__file__).parent / "contexts_probes" / "brms-2.23.0-contexts.json").read_text())


def _lhs_weighted(lhs):
    response = lhs.split(" ~ ")[0]
    return lhs.replace(response, f"{response} + weights(w)" if "|" in response else f"{response} | weights(w)", 1)


def build(family, rhs="w + z", *, weighted=False, frame=None, **override):
    lhs, name, own, _ = SPECS[family]
    formula = [_lhs_weighted(lhs) if weighted else lhs, f"{name} ~ {rhs}"]
    pri = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), **{**own, name: DPAR, **override})
    return getattr(ab, family)(formula, DATA if frame is None else frame, pri, categorical=Z if "z" in rhs else None)


def rows_by_hand(family, df, eta, theta, values):
    """Each row's log density from SciPy's distributions, in brms's parameterisation, with the formula's parameter ``theta``
    one value per row (derived without the library's formulas)."""
    if family == "weibull":
        mu = np.exp(eta)
        return stats.weibull_min(theta, scale=mu / gamma_fn(1 + 1 / theta)).logpdf(df.ypos.to_numpy())
    if family == "beta_binomial":
        mu = expit(eta)
        return stats.betabinom(df["size"].to_numpy(), mu * theta, (1 - mu) * theta).logpmf(df.k.to_numpy())
    if family == "zero_inflated_negbinomial":
        mu, shape, y = np.exp(eta), values["shape"], df.zcnt.to_numpy()
        nb = stats.nbinom(shape, shape / (shape + mu))
        return np.where(y == 0, np.log(theta + (1 - theta) * nb.pmf(0)), np.log1p(-theta) + nb.logpmf(y))
    if family == "hurdle_lognormal":
        y = df.zpos.to_numpy()
        positive = np.log1p(-theta) + stats.lognorm(values["sigma"], scale=np.exp(eta)).logpdf(np.where(y > 0, y, 1.0))
        return np.where(y == 0, np.log(theta), positive)
    sigma, y = values["sigma"], df.y.to_numpy()                       # the skew normal, its location and scale from SciPy's moments
    out = np.empty(len(y))
    for i, (m, a) in enumerate(zip(np.broadcast_to(eta, y.shape), np.broadcast_to(theta, y.shape))):
        omega = sigma / stats.skewnorm.std(a)
        out[i] = stats.skewnorm(a, loc=m - omega * stats.skewnorm.mean(a), scale=omega).logpdf(y[i])
    return out


def mean_by_hand(family, df, eta, theta, values):
    """brms's posterior_epred of each family with the formula's parameter one value per row."""
    if family == "weibull":
        return np.exp(eta)
    if family == "beta_binomial":
        return df["size"].to_numpy() * expit(eta)
    if family == "zero_inflated_negbinomial":
        return (1 - theta) * np.exp(eta)
    if family == "hurdle_lognormal":
        return (1 - theta) * np.exp(eta + values["sigma"] ** 2 / 2)
    return eta


def other_priors(family, values):
    """The log densities of the family's other parameters' priors at ``values`` and the log Jacobian of their coordinates."""
    if family == "zero_inflated_negbinomial":
        return stats.invgamma(0.4, scale=0.3).logpdf(values["shape"]), np.log(values["shape"])
    if family in ("hurdle_lognormal", "skew_normal"):
        return stats.halfnorm(scale=1).logpdf(values["sigma"]), np.log(values["sigma"])
    return 0.0, 0.0


# -- which parameters take a formula, and what stays refused -------------------------------------------------------------
def test_the_registry_offers_exactly_the_verified_formulas():
    scale = {name for name, spec in FAMILIES.items() if spec.distributional_link is not None}
    assert scale == {"gaussian", "student", "lognormal", "negbinomial", "gamma", "beta", "weibull", "beta_binomial"}
    others = {(name, f.parameter, f.link) for name, spec in FAMILIES.items() for f in spec.distributional_formulas}
    assert others == {("zero_inflated_negbinomial", "zi", "logit"), ("hurdle_lognormal", "hu", "logit"),
                      ("skew_normal", "alpha", "identity")}
    for family, (_, name, _, _) in SPECS.items():
        fam = get_family(family)
        assert fam.distributional_parameters == (name,) and fam.distributional_parameter_link(name) == LINK[family]
    assert not FAMILIES["weibull"].distributional_group_terms and not FAMILIES["beta_binomial"].distributional_group_terms
    assert all(spec.distributional_group_terms for name, spec in FAMILIES.items() if name not in ("weibull", "beta_binomial"))
    assert {name for name, spec in FAMILIES.items() if "weights" in spec.addition_terms} >= {"weibull", "beta_binomial"}
    for name in ("exponential", "skew_normal", "shifted_lognormal", "exgaussian", "geometric", "negbinomial2"):
        assert "weights" not in FAMILIES[name].addition_terms, name


REFUSED = [
    ("zero_inflated_negbinomial", ["zcnt ~ x", "shape ~ w"], r"brms's shape ~ \.\.\. is not offered here.*D33.*and for the "
     r"skew_normal \(alpha, identity link\), zero_inflated_negbinomial \(zi, logit link\), hurdle_lognormal \(hu, logit link\) families.*D116"),
    ("hurdle_lognormal", ["zpos ~ x", "sigma ~ w"], r"brms's sigma ~ \.\.\. is not offered here"),
    ("skew_normal", ["y ~ x", "sigma ~ w"], r"brms's sigma ~ \.\.\. is not offered here"),
    ("exgaussian", ["y ~ x", "beta ~ w"], r"brms's beta ~ \.\.\. is not offered here"),
    ("shifted_lognormal", ["ypos ~ x", "ndt ~ w"], r"brms's ndt ~ \.\.\. is not offered here"),
    ("zero_inflated_poisson", ["zcnt ~ x", "zi ~ w"], r"brms's zi ~ \.\.\. is not offered here"),
    ("hurdle_negbinomial", ["zcnt ~ x", "hu ~ w"], r"brms's hu ~ \.\.\. is not offered here"),
    ("zero_inflated_beta", ["zprop ~ x", "zi ~ w"], r"brms's zi ~ \.\.\. is not offered here"),
    ("weibull", ["ypos ~ x", "sigma ~ w"], r"the weibull family has no parameter 'sigma'"),
    ("zero_inflated_negbinomial", ["zcnt ~ x", "hu ~ w"], r"the zero_inflated_negbinomial family has no parameter 'hu'"),
    ("zero_inflated_negbinomial", ["zcnt ~ x", "zi ~ w + (1 | g)"], r"group-level terms in a distributional formula .*D32"),
    ("zero_inflated_negbinomial", ["zcnt ~ x + (1 |p| g)", "zi ~ 1 + (1 |p| g)"],
     r"group-level terms in the formula of zi of the zero_inflated_negbinomial family \(\(1 \|p\| g\)\) are not supported.*D116"),
    ("weibull", ["ypos ~ x + (1 |p| g)", "shape ~ 1 + (1 |p| g)"],
     r"group-level terms in the formula of shape of the weibull family .*are not supported.*D116"),
    ("beta_binomial", ["k | trials(size) ~ x + (1 |p| g)", "phi ~ 1 + (1 |p| g)"], r"the formula of phi of the beta_binomial family"),
    ("skew_normal", ["y ~ x", "alpha ~ w + offset(xpos)"], r"'offset\(xpos\)' in a distributional formula is not supported.*D32"),
    ("hurdle_lognormal", ["zpos ~ x", "hu ~ gp(x)"], r"a Gaussian process in a distributional formula .*D64"),
    ("beta_binomial", ["k | trials(size) ~ x", "phi ~ size"], r"'size' is the trials column; it cannot also be a predictor of phi.*D35"),
    ("zero_inflated_negbinomial", ["zcnt ~ x", "zi ~ zcnt"], r"'zcnt' is the response; it cannot also be a predictor of zi.*D35"),
    ("exponential", "ypos | weights(w) ~ x", r"'weights\(w\)' is not available for the exponential family"),
    ("skew_normal", "y | weights(w) ~ x", r"'weights\(w\)' is not available for the skew_normal family"),
    ("weibull", "ypos | cens(cens_r) ~ x", r"'cens\(cens_r\)' is not available for the weibull family"),
    ("beta_binomial", "k | trials(size) + trunc(ub = 9) ~ x", r"'trunc\(ub = 9\)' is not available for the beta_binomial family"),
]


@pytest.mark.parametrize("family, formula, message", REFUSED)
def test_what_stays_refused_is_refused_naming_it(family, formula, message):
    with pytest.raises(ab.FormulaError, match=message):
        parse_formula(formula, DATA, family)


def test_the_formulas_parse_with_their_own_terms():
    for family, (lhs, name, _, _) in SPECS.items():
        s = parse_formula([lhs, f"{name} ~ w * z + log(xpos)"], DATA, family, Z)
        d = s.dpar(name)
        assert s.dpar_names == (name,) and d.canonical_formula == f"{name} ~ w + z + log(xpos) + w:z" and d.intercept
        assert d.columns == ("w", "z[q]", "z[r]", "log(xpos)", "w:z[q]", "w:z[r]")
        assert s.trials == ("size" if family == "beta_binomial" else None) and d.trials is None
    with pytest.raises(ab.FormulaError, match=r"center_predictors=False with the formula 'zi ~ w' is not supported.*D34"):
        ab.zero_inflated_negbinomial(["zcnt ~ x", "zi ~ w"], DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                                                                            shape=ab.Gamma(1, 1), zi=DPAR), center_predictors=False)
    m = ab.zero_inflated_negbinomial(["zcnt ~ x", "zi ~ 0 + z"], DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                                                                               shape=ab.Gamma(1, 1), zi={"b": ab.Normal(0, 1)}),
                                     categorical=Z, center_predictors=False)
    assert m.design.dpars["zi"].columns == ("z[p]", "z[q]", "z[r]") and not m.design.dpars["zi"].centered


# -- priors -------------------------------------------------------------------------------------------------------------------
def test_priors_of_the_formulas():
    base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), shape=ab.Gamma(1, 1))
    formula = ["zcnt ~ x", "zi ~ w"]
    with pytest.raises(ab.PriorError, match=r"the formula 'zi ~ w' gives zi its own linear predictor, which needs priors on its "
                                            r"intercept and coefficients: pass zi=\{'intercept': \.\.\., 'b': \.\.\.\}"):
        ab.zero_inflated_negbinomial(formula, DATA, ab.Priors(**base))
    with pytest.raises(ab.PriorError, match=r"zi has its own formula 'zi ~ w': its priors are a mapping"):
        ab.zero_inflated_negbinomial(formula, DATA, ab.Priors(**base, zi=ab.Beta(1, 1)))
    with pytest.raises(ab.PriorError, match=r"prior for 'zi:intercept' must be a location prior \(\['Flat', 'Logistic', 'Normal', "
                                            r"'StudentT'\]\), got Beta"):
        ab.zero_inflated_negbinomial(formula, DATA, ab.Priors(**base, zi={"intercept": ab.Beta(1, 1), "b": ab.Normal(0, 1)}))
    with pytest.raises(ab.PriorError, match=r"prior for 'zi' must be a Prior, got dict \(a mapping is the form of the priors of zi's "
                                            r"own formula.*E03; this model gives it none\)"):
        ab.zero_inflated_negbinomial("zcnt ~ x", DATA, ab.Priors(**base, zi=DPAR))
    with pytest.raises(ab.PriorError, match=r"prior for 'zi' must be a Prior, got dict$"):          # a family whose zi takes none
        ab.zero_inflated_poisson("zcnt ~ x", DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), zi=DPAR))
    with pytest.raises(ab.PriorError, match=r"prior for 'alpha' must be a Prior, got dict \(a mapping is the form of the priors of "
                                            r"alpha's own formula.*no formula of its own, divergence D79\)"):
        ab.skew_normal("y ~ x", DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1), alpha=DPAR))
    with pytest.raises(ab.PriorError, match=r"alpha has its own formula 'alpha ~ w': its priors are a mapping"):
        ab.skew_normal(["y ~ x", "alpha ~ w"], DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1),
                                                              alpha=ab.Normal(0, 4)))
    with pytest.raises(ab.PriorError, match=r"shape has its own formula 'shape ~ w': its priors are a mapping"):
        ab.weibull(["ypos ~ x", "shape ~ w"], DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), shape=ab.Gamma(1, 1)))
    with pytest.raises(ab.PriorError, match=r"the hurdle_lognormal family has no zero-inflation probability; do not pass zi"):
        ab.hurdle_lognormal(["zpos ~ x", "hu ~ w"], DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1),
                                                                   hu=DPAR, zi=ab.Beta(1, 1)))
    # Logistic: brms's default on a zi / hu formula's intercept, and nowhere else here (D118)
    lg = ab.Logistic(0, 1)
    assert lg.name == "Logistic" and lg.params == {"mu": 0.0, "s": 1.0} and ab.priors.brms_prior(ab.Logistic(-0.5, 2)) == "logistic(-0.5, 2)"
    with pytest.raises(ab.PriorError, match="s must be positive"):
        ab.Logistic(0, 0)
    m = ab.zero_inflated_negbinomial(formula, DATA, ab.Priors(**base, zi={"intercept": lg, "b": ab.Normal(0, 1)}))
    assert m.priors.zi["intercept"] == lg
    for where in ({"intercept": lg}, {"b": lg}):
        with pytest.raises(ab.PriorError, match=r"cannot be Logistic\(mu=0, s=1\): Logistic\(\) is offered for the intercept of a zi or "
                                                r"hu formula only.*D118"):
            ab.zero_inflated_negbinomial(formula, DATA, ab.Priors(**{**base, **where}, zi={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1)}))
    with pytest.raises(ab.PriorError, match=r"prior for 'zi:b:w' cannot be Logistic"):
        ab.zero_inflated_negbinomial(formula, DATA, ab.Priors(**base, zi={"intercept": lg, "b": lg}))
    with pytest.raises(ab.PriorError, match=r"prior for 'alpha:intercept' cannot be Logistic"):
        ab.skew_normal(["y ~ x", "alpha ~ w"], DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1),
                                                              alpha={"intercept": lg, "b": ab.Normal(0, 1)}))
    with pytest.raises(ab.PriorError, match=r"prior for 'sigma:intercept' cannot be Logistic"):
        ab.gaussian(["y ~ x", "sigma ~ w"], DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma={"intercept": lg, "b": ab.Normal(0, 1)}))
    # the logistic density is Stan's (SciPy's) at moderate and far points
    pmm = m.pymc_model
    f = pmm.compile_logp(vars=[pmm["zi_Intercept"]], jacobian=False)
    point = m.value_point(intercept=0.1, b=[0.2], shape=1.0, zi={"intercept": 0.0, "b": [0.1]})
    for value in (-30.0, -2.5, 0.0, 0.3, 4.0, 30.0):
        point["zi_Intercept"] = np.asarray(value)
        assert float(f(point)) == pytest.approx(stats.logistic.logpdf(value), rel=1e-13, abs=1e-13)


# -- the densities, by hand, on both paths -------------------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILY_NAMES)
@pytest.mark.parametrize("weighted", [False, True])
def test_the_density_is_the_one_written_by_hand(family, weighted):
    if weighted and family == "skew_normal":
        pytest.skip("the skew normal takes no weights() (D79)")
    m = build(family, weighted=weighted)
    _, name, _, values = SPECS[family]
    dd = m.design.dpars[name]
    assert dd.link == LINK[family] and dd.centered and dd.columns == ("w", "z[q]", "z[r]")
    point = m.value_point(intercept=A, b=[B], **values, **{name: {"intercept": A_D, "b": B_D}})
    roles = m.dpar_names[name]
    assert {roles["intercept"], roles["b:w"], roles["b:z"]} <= set(point)
    eta = A + B * (DATA.x.to_numpy() - DATA.x.mean())
    theta = INVERSE[LINK[family]](A_D + dd.X_model @ B_D)
    rows = rows_by_hand(family, DATA, eta, theta, values)
    w = DATA.w.to_numpy() if weighted else 1.0
    other, jac = other_priors(family, values)
    want = ((w * rows).sum() + stats.norm.logpdf(A, 0, 2) + stats.norm.logpdf(B, 0, 1) + stats.norm.logpdf(A_D, 0, 1)
            + stats.norm.logpdf(B_D, 0, 0.5).sum() + other)
    assert float(m.pymc_model.compile_logp(jacobian=False)(point)) == pytest.approx(want, rel=1e-12)
    assert float(m.pymc_model.compile_logp()(point)) == pytest.approx(want + jac, rel=1e-12)
    pmm = m.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[roles["value"]]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    np.testing.assert_allclose(np.asarray(fn(point)[0], dtype=float), theta, rtol=1e-14)
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    names = [v.name for v in pmm.value_vars]
    assert float(get_jaxified_logp(pmm)([point[n] for n in names])) == pytest.approx(want + jac, rel=1e-12)


@pytest.mark.parametrize("family, lhs, own", [("weibull", "ypos ~ x + (1 | g)", {"shape": ab.Gamma(1, 1)}),
                                              ("beta_binomial", "k | trials(size) ~ x + (1 | g)", {"phi": ab.Gamma(1, 0.1)})])
def test_weights_multiply_each_rows_log_density(family, lhs, own):
    """brms adds weights[n] * lpdf(Y[n] | ...) row by row: the weighted density is the unweighted one plus sum((w - 1) * log p(y_i)),
    on both paths; unit weights give the unweighted density."""
    pri = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), **own)
    m0, mw = (getattr(ab, family)(f, DATA, pri) for f in (lhs, _lhs_weighted(lhs)))
    ones = getattr(ab, family)(_lhs_weighted(lhs), DATA.assign(w=1.0), pri)
    rng = np.random.default_rng(3)
    point = {k: np.asarray(v + rng.normal(0.0, 0.3, np.shape(v))) for k, v in m0.pymc_model.initial_point().items()}
    l0, lw, l1 = (float(m.pymc_model.compile_logp()(point)) for m in (m0, mw, ones))
    pmm = m0.pymc_model
    rows = np.asarray(pmm.compile_fn(pmm.logp(vars=pmm.observed_RVs, sum=False)[0], inputs=pmm.value_vars,
                                     on_unused_input="ignore")(point), dtype=float)
    w = mw.design.weights
    assert np.array_equal(w, DATA.w.to_numpy()) and not np.allclose(w, 1.0) and np.isfinite(rows).all()
    assert abs((lw - l0) - np.sum((w - 1.0) * rows)) <= 1e-10 * max(1.0, abs(lw))
    assert abs(l1 - l0) <= 1e-12 * max(1.0, abs(l0))
    from pymc.sampling.jax import get_jaxified_logp

    names = [v.name for v in mw.pymc_model.value_vars]
    assert abs(float(get_jaxified_logp(mw.pymc_model)([point[n] for n in names])) - lw) <= 1e-12 * max(1.0, abs(lw))
    assert mw.target_identity["sha256"] != m0.target_identity["sha256"]
    assert any(line.startswith("  weights   : w, each fitted row's log likelihood multiplied by its weight")
               for line in mw.describe().splitlines())


def test_value_point_refusals():
    m = build("zero_inflated_negbinomial")
    with pytest.raises(ValueError, match=r"zi has its own formula 'zi ~ w \+ z': give zi=\{'intercept': float, 'b': array\}"):
        m.value_point(intercept=A, b=[B], shape=1.3, zi=0.3)
    with pytest.raises(ValueError, match=r"zi\['b'\] must have 3 finite entries, one per design column \['w', 'z\[q\]', 'z\[r\]'\]"):
        m.value_point(intercept=A, b=[B], shape=1.3, zi={"intercept": 0.1, "b": [0.1]})
    with pytest.raises(ValueError, match=r"zi \(the zero-inflation probability\) is required"):
        m.value_point(intercept=A, b=[B], shape=1.3)
    s = build("skew_normal")
    with pytest.raises(ValueError, match=r"alpha has its own formula 'alpha ~ w \+ z': give alpha=\{'intercept': float"):
        s.value_point(intercept=A, b=[B], sigma=0.8, alpha=1.0)
    w = build("weibull")
    with pytest.raises(ValueError, match=r"shape has its own formula 'shape ~ w \+ z'"):
        w.value_point(intercept=A, b=[B], shape=1.5)


# -- predictions and log_lik with each row's parameter ---------------------------------------------------------------------
def draws(model, family):
    """Two chains of four constant draws: raw-scale intercepts (as a posterior from fit() holds them), coefficients, and the
    family's other parameters."""
    _, name, _, values = SPECS[family]
    roles, n = model.dpar_names[name], model.names
    post = {"Intercept": np.full((2, 4), A), "x": np.full((2, 4), B), roles["intercept"]: np.full((2, 4), A_D),
            roles["b:w"]: np.full((2, 4), B_D[0]), roles["b:z"]: np.broadcast_to(B_D[1:], (2, 4, 2))}
    post.update({n[k]: np.full((2, 4), v) for k, v in values.items()})
    return az.from_dict(posterior=post, coords={f"{roles['b:z']}_dim": ["q", "r"]}, dims={roles["b:z"]: [f"{roles['b:z']}_dim"]})


def _raw(df, family):
    """The formula's parameter on the rows of ``df`` from the raw-scale intercept and the raw design."""
    z = df.z.to_numpy()
    eta_d = A_D + B_D[0] * df.w.to_numpy() + B_D[1] * (z == "q") + B_D[2] * (z == "r")
    return eta_d, INVERSE[LINK[family]](eta_d)


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_predictions_take_each_rows_parameter(family):
    m = build(family)
    _, name, _, values = SPECS[family]
    post = draws(m, family)
    shape3 = (2, 4, len(DATA))
    eta = np.broadcast_to(A + B * DATA.x.to_numpy(), shape3)
    eta_d, theta = (np.broadcast_to(v, shape3) for v in _raw(DATA, family))
    np.testing.assert_allclose(m.predict(post, kind="linear").eta, eta, rtol=1e-14)
    mean = m.predict(post, kind="mean")
    want = mean_by_hand(family, DATA, eta, theta, values)
    np.testing.assert_allclose(mean["expected" if family == "beta_binomial" else m.names["mean"]], want, rtol=1e-13)
    lin = m.predict(post, kind="linear", dpar=name)
    np.testing.assert_allclose(lin.eta, eta_d, rtol=1e-13)
    assert lin.attrs["dpar"] == name and lin.attrs["link"] == LINK[family]
    np.testing.assert_allclose(m.predict(post, kind="mean", dpar=name)[name], theta, rtol=1e-13)
    new = DATA.iloc[[3, 0, 7]].reset_index(drop=True)
    np.testing.assert_allclose(m.predict(post, newdata=new, kind="mean", dpar=name)[name][0, 0], _raw(new, family)[1], rtol=1e-13)
    # seeded response draws, each row's parameter in the family's own generator
    out = m.predict(post, kind="response", seed=5).y.to_numpy()
    rng = np.random.default_rng(5)
    if family == "weibull":
        mu = np.exp(eta)
        y = mu * np.exp(-np.log(gamma_fn(1 + 1 / theta))) * rng.weibull(theta)
    elif family == "beta_binomial":
        p = rng.beta(expit(eta) * theta, expit(-eta) * theta)
        y = rng.binomial(DATA["size"].to_numpy(), p)
    elif family == "zero_inflated_negbinomial":
        mu, shape = np.exp(eta), values["shape"]
        zero = rng.uniform(size=shape3) < theta
        y = np.where(zero, 0, rng.negative_binomial(shape, shape / (shape + mu)))
    elif family == "hurdle_lognormal":
        zero = rng.uniform(size=shape3) < theta
        y = np.where(zero, 0.0, rng.lognormal(eta, values["sigma"]))
    else:
        y = lk.skew_normal_random(eta, np.full(shape3, values["sigma"]), theta, rng=rng)
    np.testing.assert_allclose(out, y, rtol=1e-12)


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_log_lik_loo_and_waic_read_each_rows_parameter_and_the_weights(family):
    m = build(family)
    _, name, _, values = SPECS[family]
    post = draws(m, family)
    eta = A + B * DATA.x.to_numpy()
    rows = rows_by_hand(family, DATA, eta, _raw(DATA, family)[1], values)
    ll = ab.log_lik(m, post)
    assert ll.dims == ("chain", "draw", "__obs__")
    np.testing.assert_allclose(ll.to_numpy(), np.broadcast_to(rows, (2, 4, len(DATA))), rtol=1e-11)
    assert np.isfinite(ab.waic(m, post).estimates.to_numpy()).all()
    if family == "skew_normal":
        return
    mw = build(family, weighted=True)
    np.testing.assert_allclose(ab.log_lik(mw, draws(mw, family)).to_numpy(), DATA.w.to_numpy() * ll.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("case_id", ["weibull_weighted", "beta_binomial_weighted"])
def test_the_weighted_cases_weight_log_lik_loo_and_waic_and_not_the_predictions(case_id):
    """brms's log_lik() multiplies each row's log density by its weight (C07), and loo() and waic() read it; posterior_epred
    ignores the weights (G4 checks it against brms on these cases)."""
    from autobayes.post_processing import loo_from_log_lik, waic_from_log_lik

    b = gates.Bundle(case_id)
    lhs = "ypos ~ x + (1 | g)" if b.model.family == "weibull" else "k | trials(size) ~ x + (1 | g)"
    plain = getattr(ab, b.model.family)(lhs, b.data, b.model.priors)
    idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
    w = b.model.design.weights
    assert not np.allclose(w, 1.0)
    weighted = ab.log_lik(b.model, idata)
    expected = w * ab.log_lik(plain, idata).to_numpy()
    np.testing.assert_allclose(weighted.to_numpy(), expected, rtol=1e-14)
    ll = weighted.copy(data=expected)
    pd.testing.assert_frame_equal(ab.waic(b.model, idata).estimates, waic_from_log_lik(ll).estimates)
    pd.testing.assert_frame_equal(ab.loo(b.model, idata, r_eff=1).estimates, loo_from_log_lik(ll, r_eff=1).estimates)
    for kind in ("linear", "mean"):
        got, want = b.model.predict(idata, kind=kind), plain.predict(idata, kind=kind)
        for var in got.data_vars:
            np.testing.assert_allclose(got[var].to_numpy(), want[var].to_numpy(), rtol=1e-14)


# -- describe, identity, records, prior predictive ------------------------------------------------------------------------------
DESCRIBED = {
    "weibull": ["  shape     : shape ~ w + z   (distributional formula: shape has its own linear predictor, log link)",
                "  log(shape_i) = eta_shape,i = a_shape_c + (x_w - mean(x_w)) * b_shape_w + (1[z_i = q] - p_z[q]) * b_shape_z[q] "
                "+ (1[z_i = r] - p_z[r]) * b_shape_z[r]",
                "  shape_i = exp(eta_shape,i), one per row: the formula 'shape ~ w + z' (Distributional parameter shape, above)",
                "  y_i ~ Weibull(shape_i, scale = mu_i / Gamma(1 + 1 / shape_i)); mu_i = exp(eta_i); E[y_i] = mu_i"],
    "beta_binomial": ["  phi_i = exp(eta_phi,i), one per row: the formula 'phi ~ w + z' (Distributional parameter phi, above)",
                      "  k_i ~ BetaBinomial(size_i, mu_i phi_i, (1 - mu_i) phi_i): size_i trials whose success probability is "
                      "Beta(mu_i phi_i, (1 - mu_i) phi_i);"],
    "zero_inflated_negbinomial": [
        "  zi        : zi ~ w + z   (distributional formula: zi has its own linear predictor, logit link)",
        "Distributional parameter zi (its own formula 'zi ~ w + z', logit link; brms's bf(..., zi ~ w + z))",
        "  logit(zi_i) = eta_zi,i = a_zi_c + (x_w - mean(x_w)) * b_zi_w + (1[z_i = q] - p_z[q]) * b_zi_z[q] + (1[z_i = r] - p_z[r]) * b_zi_z[r]",
        "  zi_i = inv_logit(eta_zi,i), in (0, 1), is row i's zi in the likelihood below; the coefficients act on logit zi.",
        "  a_zi_c ~ Normal(mu=0, sigma=1)   (on the logit scale of zi)",
        "  zi_i = inv_logit(eta_zi,i), one per row: the formula 'zi ~ w + z' (Distributional parameter zi, above)",
        "  y_i ~ ZeroInflatedNegBinomial(mu_i, shape, zi_i); mu_i = exp(eta_i): a structural zero with probability zi_i,",
        "  zi_Intercept             a_zi_c (the intercept of zi's formula)",
        "  zi_w                     b_zi_w"],
    "hurdle_lognormal": ["  hu_i = inv_logit(eta_hu,i), one per row: the formula 'hu ~ w + z' (Distributional parameter hu, above)",
                         "  y_i ~ HurdleLogNormal(mu_i, sigma, hu_i): an exact zero with probability hu_i, LogNormal(mu_i, sigma) otherwise"],
    "skew_normal": ["  alpha_i = eta_alpha,i = a_alpha_c + (x_w - mean(x_w)) * b_alpha_w + (1[z_i = q] - p_z[q]) * b_alpha_z[q] "
                    "+ (1[z_i = r] - p_z[r]) * b_alpha_z[r]",
                    "  alpha_i = eta_alpha,i is row i's alpha in the likelihood below; the coefficients act on alpha itself.",
                    "  alpha_i = eta_alpha,i, one per row: the formula 'alpha ~ w + z' (Distributional parameter alpha, above)",
                    "  y_i ~ SkewNormal(mean mu_i, SD sigma, skewness alpha_i): Azzalini's SN(xi_i, omega_i, alpha_i) with"],
}
DETERMINISTIC = {"weibull": "alpha = exp(eta_shape), one shape per row (B01)", "beta_binomial": "phi = exp(eta_phi), one phi per row (B01)",
                 "zero_inflated_negbinomial": "zi = inv_logit(eta_zi), one zi per row (B01)",
                 "hurdle_lognormal": "hu = inv_logit(eta_hu), one hu per row (B01)", "skew_normal": "alpha = eta_alpha, one alpha per row (B01)"}


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_describe_writes_the_formula_out(family):
    text = build(family).describe()
    lines = text.splitlines()
    _, name, _, _ = SPECS[family]
    assert lines[0].endswith(f" model with distributional formula(s) for {name} (population-level only)")   # z is the formula's only
    for line in DESCRIBED[family]:
        assert line in lines, line
    assert lines[-1].endswith(DETERMINISTIC[family])
    assert f"{name}_logodds__" not in text and f"  {name} ~ " not in text


def test_the_target_identity_holds_the_formula_and_its_priors():
    base = build("zero_inflated_negbinomial").target_identity
    assert build("zero_inflated_negbinomial").target_identity == base
    assert build("zero_inflated_negbinomial", rhs="w").target_identity != base
    assert build("zero_inflated_negbinomial", zi={"intercept": ab.Logistic(0, 1), "b": ab.Normal(0, 0.5)}).target_identity != base
    assert build("zero_inflated_negbinomial", weighted=True).target_identity != base
    plain = ab.zero_inflated_negbinomial("zcnt ~ x", DATA, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                                                                     shape=ab.InverseGamma(0.4, 0.3), zi=ab.Beta(1, 1))).target_identity
    assert plain != base
    assert build("skew_normal").target_identity != build("skew_normal", rhs="w").target_identity


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_fit_records_and_summary_rows(family, monkeypatch):
    m = build(family)
    _, name, own, _ = SPECS[family]
    post = draws(m, family)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # constant synthetic draws: ArviZ's R-hat divides by zero
        rows = session._summary_rows(m, post)
    roles = m.dpar_names[name]
    assert {roles["intercept"], roles["b:w"], f"{roles['b:z']}[q]", f"{roles['b:z']}[r]"} <= set(rows) and roles["value"] not in rows
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=4,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    pri = rec["model"]["priors"]
    assert pri[f"{name}:intercept"] == "Normal(mu=0, sigma=1)" and pri[f"{name}:b:z[q]"] == "Normal(mu=0, sigma=0.5)" and name not in pri
    assert all(pri[k] == str(v) for k, v in own.items())
    dist = rec["model"]["distributional"][name]
    assert dist["formula"] == f"{name} ~ w + z" and dist["link"] == LINK[family] and dist["design_columns"] == ["w", "z[q]", "z[r]"]
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


def test_prior_predictive_draws_each_rows_parameter():
    pp = build("zero_inflated_negbinomial").prior_predictive(draws=20, seed=4)
    zi = pp.prior["zi"].to_numpy()
    assert zi.shape == (1, 20, len(DATA)) and ((zi > 0) & (zi < 1)).all()
    assert pp.prior_predictive["zcnt"].shape == (1, 20, len(DATA))
    pp = build("skew_normal").prior_predictive(draws=20, seed=4)
    assert pp.prior["alpha"].shape == (1, 20, len(DATA)) and np.isfinite(pp.prior["alpha"]).all()
    flat = build("hurdle_lognormal", hu={"intercept": ab.Logistic(0, 1), "b": ab.Flat()})
    with pytest.raises(ValueError, match=r"the prior of \['hu:b:w', 'hu:b:z\[q\]', 'hu:b:z\[r\]'\] is flat"):
        flat.prior_predictive(draws=5, seed=1)


# -- brms's default priors (tests/contexts_probes, printed by brms 2.23.0's default_prior()) -------------------------------------
PROBED = {   # the probe's table name -> (formula, family, categorical)
    "weibull, shape ~ x": (["ypos ~ x", "shape ~ x"], "weibull", None),
    "weibull, shape ~ 1, a wide response": (["ybig ~ x", "shape ~ 1"], "weibull", None),
    "beta_binomial, phi ~ z": (["k | trials(size) ~ x", "phi ~ z"], "beta_binomial", Z),
    "zero_inflated_negbinomial, zi ~ x": (["cnt ~ x", "zi ~ x"], "zero_inflated_negbinomial", None),
    "zero_inflated_negbinomial, zi ~ 0 + z": (["cnt ~ x", "zi ~ 0 + z"], "zero_inflated_negbinomial", Z),
    "zero_inflated_negbinomial, zi ~ 1": (["cnt ~ x", "zi ~ 1"], "zero_inflated_negbinomial", None),
    "hurdle_lognormal, hu ~ x + z": (["zpos ~ x", "hu ~ x + z"], "hurdle_lognormal", Z),
    "skew_normal, alpha ~ x": (["y ~ x", "alpha ~ x"], "skew_normal", None),
    "skew_normal, alpha ~ 1, a wide response": (["ybig ~ x", "alpha ~ 1"], "skew_normal", None),
}
PROBE_COLUMNS = ("prior", "class", "coef", "group", "resp", "dpar", "nlpar", "lb", "ub", "source")


def test_the_probe_covers_every_table():
    assert PROBE["brms"] == "2.23.0" and sorted(PROBE["tables"]) == sorted(PROBED)


@pytest.mark.parametrize("label", sorted(PROBED))
def test_the_default_prior_table_is_brms(label):
    formula, family, categorical = PROBED[label]
    frame = pd.DataFrame(PROBE["frame"])
    pri = ab.default_priors(formula, frame, family=family, categorical=categorical)
    model = getattr(ab, family)(formula, frame, pri, categorical=categorical)
    table = [tuple(r[c] for c in PROBE_COLUMNS) for r in model.prior_table()]
    assert table == [tuple(r[c] for c in PROBE_COLUMNS) for r in PROBE["tables"][label]]
    name = formula[1].split(" ~ ")[0]
    ds = model.spec.dpar(name)
    want = ([f"{name}:intercept"] if ds.intercept else []) + [f"{name}:b:{col}" for col in ds.columns]
    assert [lab for lab in model.default_prior_labels if lab.startswith(f"{name}:")] == want and name not in model.default_prior_labels
    if ds.intercept:
        default = get_family(family).distributional_formula(name)
        intercept = getattr(pri, name)["intercept"]
        assert intercept == (default.default_intercept() if default is not None else ab.StudentT(3, 0, 2.5))
        assert intercept.source == "default"
        assert f"  a_{name}{'_c' if model.design.dpars[name].centered else ''} ~ {intercept}   (brms default " in model.describe()


# -- the oracle's extension to these formulas fails on a corrupted fixture ---------------------------------------------------
def _corrupted(bundle, mutate):
    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def test_the_gates_see_a_corrupted_zi_fixture():
    b = gates.Bundle("zero_inflated_negbinomial_distributional_zi")
    for gate in gates.ALL:
        gate(b)

    def design(raw):
        raw["standata"]["X_zi"]["values"][-1] += 1e-11
    with pytest.raises(OracleMismatch, match=r"zi's design column 'z\[r\]' matches 0 brms X_zi columns"):
        _corrupted(b, design)

    def coefficient(raw):
        raw["points"][3]["pars"]["b_zi"]["values"][0] += 1e-6
    with pytest.raises(OracleMismatch, match="point 3: log density"):
        gates.check_logp(_corrupted(b, coefficient))

    def gradient(raw):
        (i,) = [int(e["index"]) - 1 for e in raw["upars"] if e["par"] == "Intercept_zi"]
        raw["points"][1]["grad"]["values"][i] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 1: gradient of zi_Intercept\[0\]"):
        gates.check_gradient(_corrupted(b, gradient))

    def raw_intercept(raw):
        raw["points"][2]["pars"]["b_zi_Intercept"]["values"][0] += 1e-6
    with pytest.raises(OracleMismatch, match="point 2: raw-scale intercept of zi"):
        gates.check_derived(_corrupted(b, raw_intercept))

    def epred(raw):
        block = raw["predictions"]["fitted"]["epred"]
        block["values"][5] *= 1 + 1e-7
    with pytest.raises(OracleMismatch, match=r"fitted: 'mean' differs from brms posterior_epred"):
        gates.check_predictions(_corrupted(b, epred))

    def default(raw):
        (row,) = [r for r in raw["default_prior"] if r["class"] == "Intercept" and r["dpar"] == "zi"]
        row["prior"] = "logistic(0, 2)"
    with pytest.raises(OracleMismatch, match="differs from brms's default_prior"):
        gates.check_default_prior(_corrupted(b, default))

    def no_rows(raw):
        raw["default_prior"] = [r for r in raw["default_prior"] if r["dpar"] != "zi"]
    with pytest.raises(OracleMismatch, match=r"brms default_prior lacks the rows \[\('Intercept', 'zi'\), \('b', 'zi'\)\]"):
        gates.check_default_prior(_corrupted(b, no_rows))

    plain = ab.zero_inflated_negbinomial("zcnt ~ x + (1 | g)", b.data, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1),
                                                                                 sd=ab.HalfNormal(1), shape=ab.InverseGamma(0.4, 0.3),
                                                                                 zi=ab.Beta(1, 1)))
    with pytest.raises(OracleMismatch, match=r"brms gives zi its own formula \(standata X_zi\); the AutoBayes model has none"):
        gates.Bundle("zero_inflated_negbinomial_distributional_zi", model=plain)


def test_the_gates_see_a_model_without_the_alpha_formula_or_the_weights():
    b = gates.Bundle("skew_normal_distributional_alpha")
    plain = ab.skew_normal("y ~ x + (1 | g)", b.data, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1),
                                                                sigma=ab.HalfNormal(1), alpha=ab.Normal(0, 2)))
    with pytest.raises(OracleMismatch, match=r"brms gives alpha its own formula \(standata X_alpha\); the AutoBayes model has none"):
        gates.Bundle("skew_normal_distributional_alpha", model=plain)
    w = gates.Bundle("weibull_weighted")
    unweighted = ab.weibull("ypos ~ x + (1 | g)", w.data, w.model.priors)
    with pytest.raises(OracleMismatch, match="brms has weights but AutoBayes does not"):
        gates.check_design(gates.Bundle("weibull_weighted", model=unweighted))

    def weights(raw):
        raw["standata"]["weights"]["values"][4] += 1e-6
    with pytest.raises(OracleMismatch, match="weights differ by"):
        gates.check_design(_corrupted(w, weights))


# -- the post-processing functions that keep refusing these models (D99) ----------------------------------------------------------
@pytest.mark.parametrize("case_id, match", [
    ("zero_inflated_negbinomial_distributional_zi", "a distributional formula for zi"),
    ("hurdle_lognormal_distributional_hu", "a distributional formula for hu"),
    ("skew_normal_distributional_alpha", "a distributional formula for alpha"),
    ("weibull_distributional_shape", "a distributional formula for shape"),
    ("weibull_weighted", r"the addition term\(s\) weights\(w\)"),
    ("beta_binomial_weighted", "conditional_effects\\(\\) of a beta_binomial model is not offered: its expected response needs the trials"),
])
def test_conditional_effects_and_prior_draws_refuse_them(case_id, match):
    case = fx.load_case(case_id)
    model = adapter.build_model(case, fx.read_frame(case, fx.fixture_dir(case_id) / "data.csv"))
    with pytest.raises(ValueError, match=match) as refused:
        ab.conditional_effects(model, None, ["x"])
    assert "divergence D99" in str(refused.value)
    if model.spec.dpars:
        with pytest.raises(ValueError, match=r"prior_draws\(\) of a model with .* is not offered"):   # its formula (or alpha, D124)
            ab.prior_draws(model, draws=10, seed=1)
        b = gates.Bundle(case_id, model=model)
        idata = adapter.draws(b.fixture, b.model, b.cmap, b.gmap)
        name = model.spec.dpar_names[0]
        with pytest.raises(ValueError, match=f"'{name}_Intercept' is a coefficient of the distributional formula for {name}"):
            ab.hypothesis(model, idata, f"{name}_Intercept > 0")
        assert ab.hypothesis(model, idata, "x > 0").table.shape == (1, 8)
