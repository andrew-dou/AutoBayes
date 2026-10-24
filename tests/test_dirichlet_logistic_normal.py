"""Responses on the simplex (brms-parity F01b, docs/dirichlet-logistic-normal.md): brms's ``dirichlet()`` and
``logistic_normal()`` on a ``cbind(y1, ..., yK)`` response. The formula's matrix response and its refusals, the response's
checks, the categories and ``refcat``, the priors and their refusals, the graph (names, dims, the per-category group terms) and
its log density with and without the Jacobian against SciPy's Dirichlet and multivariate normal densities with the simplex
Jacobian written out, the JAX density, ``value_point``, by-hand predictions on fitted and new rows, the draws on the simplex
(their moments and the generator's stream), ``log_lik()``, the refusals downstream (``kind="mean"`` for the logistic-normal
family, new levels, ``bayes_R2()``, ``prior_predictive()``), ``describe()``, the target identity, brms's default prior tables
and the fit record. No MCMC; the brms comparison is the oracle cases ``dirichlet_varint``, ``logistic_normal_varint`` and
``logistic_normal_refcat``."""
import json
import math
import warnings
from dataclasses import asdict
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from scipy import stats
from scipy.special import betaln, gammaln, softmax

import autobayes as ab
from autobayes.families import FAMILIES, get_family
from autobayes.families import simplex as sx
from autobayes.formula import FormulaError, matrix_response_name, parse_formula

F = "cbind(a, b, c) ~ x + (1 | g)"


# -- data and models ----------------------------------------------------------------------------------------------------
def frame(n=24, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([2.0, 3.0, 4.0], size=n)
    return pd.DataFrame({"a": p[:, 0], "b": p[:, 1], "c": 1.0 - p[:, 0] - p[:, 1], "x": rng.normal(size=n),
                         "w": rng.normal(size=n), "z": np.array(["p", "q", "r"])[np.arange(n) % 3],
                         "g": np.array(list("uvwxyz"))[np.arange(n) % 6], "h": np.array(["h1", "h2", "h3", "h4"])[(np.arange(n) // 2) % 4],
                         "e": np.exp(rng.normal(size=n)), "o": rng.integers(1, 4, n), "t": np.arange(n), "sx": np.full(n, 0.5)})


def dir_priors(**kw):
    return ab.Priors(**{"intercept": ab.Normal(0, 2), "b": ab.Normal(0, 1), "sd": ab.HalfNormal(1), "phi": ab.Gamma(2, 0.5), **kw})


def ln_priors(**kw):
    return ab.Priors(**{"intercept": ab.Normal(0, 2), "b": ab.Normal(0, 1), "sd": ab.HalfNormal(1), "sigma": ab.HalfNormal(1),
                        "lncor": ab.LKJ(2), **kw})


@pytest.fixture(scope="module")
def dir_model():
    return ab.dirichlet(F, frame(), dir_priors())


@pytest.fixture(scope="module")
def ln_model():
    return ab.logistic_normal(F, frame(), ln_priors())


@pytest.fixture(scope="module")
def ln_refcat():
    return ab.logistic_normal(F, frame(), ln_priors(), refcat="b")


def cholesky(r):
    return np.array([[1.0, 0.0], [r, math.sqrt(1.0 - r * r)]])


def natural(model, rng, r=0.3):
    """Random natural-scale values for ``model``: centred intercepts, coefficients (columns x categories), one SD per category
    and standardised effects (levels x categories) per factor, and the family's own parameters."""
    K1, P = len(model.categories) - 1, len(model.spec.columns)
    out = dict(intercept=rng.normal(size=K1), b=rng.normal(size=(P, K1)) if P else None,
               sd={t.factor: rng.uniform(0.3, 1.5, K1) for t in model.spec.group_terms} or None,
               z={t.factor: rng.normal(size=(model.design.group_designs[t.factor].J, K1)) for t in model.spec.group_terms} or None)
    if model.family == "dirichlet":
        out["phi"] = float(rng.uniform(1.0, 6.0))
    else:
        out.update(sigma=rng.uniform(0.3, 1.5, K1), Llncor=cholesky(r))
    return out


def as_point(model, nat):
    kw = {k: v for k, v in nat.items() if v is not None}
    if "b" not in kw:
        kw["b"] = np.zeros((0, len(model.categories) - 1))
    if not model.spec.group_terms:
        kw.pop("sd", None), kw.pop("z", None)
    return model.value_point(**kw)


def linear_predictor(model, nat, frame_=None):
    """The K - 1 linear predictors by hand on the fitted rows: the centred design times b, the intercepts and each factor's
    sd[k] * z[level, k]."""
    d = model.design
    mu = np.asarray(nat["intercept"], dtype=float)[None, :] + (d.X_model @ nat["b"] if nat["b"] is not None else np.zeros((d.n, 1)))
    for t in model.spec.group_terms:
        mu = mu + (nat["sd"][t.factor] * nat["z"][t.factor])[d.group_designs[t.factor].group_index]
    return mu


def category_response(model):
    return np.asarray(model.design.y)[:, model.category_order]


def hand_logp(model, nat, jacobian=True):
    """The model's log density by hand: SciPy's Dirichlet / multivariate normal densities of the rows (the logistic-normal one
    with the Jacobian -sum log y written out), the priors, the standardised effects and, with ``jacobian``, the log
    transforms' and the cholesky_corr transform's Jacobians."""
    y, mu = category_response(model), linear_predictor(model, nat)
    lp = 0.0
    if model.family == "dirichlet":
        phi = nat["phi"]
        for i in range(len(y)):
            lp += stats.dirichlet.logpdf(y[i], phi * softmax(np.r_[0.0, mu[i]]))
        lp += stats.gamma.logpdf(phi, 2.0, scale=2.0) + (math.log(phi) if jacobian else 0.0)
    else:
        sig, L = np.asarray(nat["sigma"]), np.asarray(nat["Llncor"])
        S = np.diag(sig) @ L @ L.T @ np.diag(sig)
        zr = np.log(y[:, 1:]) - np.log(y[:, :1])
        for i in range(len(y)):
            lp += stats.multivariate_normal.logpdf(zr[i], mu[i], S) - np.log(y[i]).sum()
        r = L[1, 0]
        lp += stats.halfnorm.logpdf(sig).sum() + (np.log(sig).sum() if jacobian else 0.0)
        lp += (2.0 - 1.0) * math.log(1.0 - r * r) - (3.0 * math.log(2.0) + betaln(2.0, 2.0))   # LKJ(2) of the one correlation
        lp += math.log(1.0 - r * r) if jacobian else 0.0                                         # tanh's Jacobian
    lp += stats.norm.logpdf(nat["intercept"], 0, 2).sum()
    if nat["b"] is not None:
        lp += stats.norm.logpdf(nat["b"], 0, 1).sum()
    for t in model.spec.group_terms:
        sd, z = nat["sd"][t.factor], nat["z"][t.factor]
        lp += stats.halfnorm.logpdf(sd).sum() + stats.norm.logpdf(z).sum() + (np.log(sd).sum() if jacobian else 0.0)
    return lp


def draws_of(model, nats):
    """The posterior draws (one chain) of ``nats`` under the model's names, the intercepts on the raw scale as a fit's
    posterior holds them."""
    n, d = model.names, model.design
    cat = n["category_dim"]
    others = list(model.categories[1:])
    data = {}
    raw = [np.asarray(v["intercept"]) - (d.x_means @ v["b"] if v["b"] is not None else 0.0) for v in nats]
    data[n["intercept"]] = (("chain", "draw", cat), np.stack(raw)[None])
    coords = {"chain": [0], "draw": np.arange(len(nats)), cat: others}
    i = 0
    for term in model.spec.bambi_terms:
        k = len(term.columns)
        vals = np.stack([v["b"][i:i + k] for v in nats])
        if term.levels:
            data[term.name] = (("chain", "draw", f"{term.name}_dim", cat), vals[None])
            coords[f"{term.name}_dim"] = list(term.levels)
        else:
            data[term.name] = (("chain", "draw", cat), vals[None, :, 0])
        i += k
    for (g, coef), roles in model.group_coefficient_names.items():
        data[roles["sd"]] = (("chain", "draw", cat), np.stack([v["sd"][g] for v in nats])[None])
        data[roles["group_effect"]] = (("chain", "draw", roles["coordinate"], cat), np.stack([v["sd"][g] * v["z"][g] for v in nats])[None])
        coords[roles["coordinate"]] = list(d.group_designs[g].levels)
    if model.family == "dirichlet":
        data["phi"] = (("chain", "draw"), np.array([v["phi"] for v in nats])[None])
    else:
        roles = model.simplex_names
        for k, s in enumerate(roles["sigma"]):
            data[s] = (("chain", "draw"), np.array([v["sigma"][k] for v in nats])[None])
        data[roles["L"]] = (("chain", "draw", roles["coordinate"], roles["coordinate_T"]), np.stack([v["Llncor"] for v in nats])[None])
        coords[roles["coordinate"]] = coords[roles["coordinate_T"]] = others
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


# -- 1. the formula --------------------------------------------------------------------------------------------------
def test_the_cbind_response_is_parsed_with_its_columns():
    s = parse_formula(F, None, "dirichlet")
    assert s.response == matrix_response_name(["a", "b", "c"]) == "cbind_a_b_c"
    assert s.response_columns == ("a", "b", "c") and s.formula == F
    assert s.canonical_formula == "cbind(a, b, c) ~ x + (1 | g)" and s.groups == ("g",)
    assert hash(s) == hash(parse_formula(F, None, "dirichlet")) and hash(s) != hash(parse_formula("cbind(a, c, b) ~ x + (1 | g)", None, "dirichlet"))
    replica = ab.ModelSpec(**asdict(s))
    assert replica == s and replica.response_columns == ("a", "b", "c")
    with_data = parse_formula(F, frame(), "logistic_normal")
    assert with_data.response_columns == ("a", "b", "c") and with_data.levels_by_group["g"] == tuple("uvwxyz")
    assert parse_formula("y ~ x", None, "gaussian").response_columns == ()          # every other formula has none
    assert parse_formula("  cbind( a ,b,  c ) ~ 1", None, "dirichlet").response_columns == ("a", "b", "c")


@pytest.mark.parametrize("formula,family,needle", [
    ("cbind(a, b, c) ~ x", "gaussian", r"a cbind\(\.\.\.\) response of several columns is the response of the dirichlet and "
                                       r"logistic_normal families .* the gaussian family takes one response column"),
    ("cbind(a, (b), c) ~ x", "dirichlet", r"expected 'cbind\(y1, y2, \.\.\.\) ~ terms'"),
    ("cbind(a, 2b, c) ~ x", "dirichlet", r"cbind\(\) takes the response's column names, got '2b'"),
    ("cbind(a, b, a) ~ x", "dirichlet", r"cbind\(\) names the column 'a' twice"),
    ("cbind(a) ~ x", "dirichlet", r"the dirichlet family needs at least 2 response columns, got 1$"),
    ("cbind(a, b) ~ x", "logistic_normal", r"the logistic_normal family needs at least 3 response columns, got 2 \(with two, brms's "
                                           r"correlation of the log-ratios is 1 x 1 and has no parameter, which no frozen case verifies\)"),
    ("cbind(a, b, c) | weights(w) ~ x", "dirichlet", r"addition terms after the cbind\(\) response \('\| weights\(w\)'\) are not "
                                                    r"offered for the dirichlet family .*divergence D138"),
    ("cbind(a, b, c) | trials(o) ~ x", "logistic_normal", r"addition terms after the cbind\(\) response .*divergence D139"),
    ("cbind(a, b, c) ~ a + x", "dirichlet", r"the response column 'a' cannot also be a predictor, a grouping factor or the offset"),
    ("cbind(a, b, c) ~ x + (1 | b)", "dirichlet", r"the response column 'b' cannot also be"),
    ("cbind(a, b, c) ~ x + log(c)", "logistic_normal", r"the response column 'c' cannot also be"),
    ("cbind(a, b, c) ~ x + gp(x)", "dirichlet", r"^'cbind\(a, b, c\) ~ x \+ gp\(x\)': a Gaussian process"),   # quoted as written
    (["cbind(a, b, c) ~ x", "phi ~ w"], "dirichlet", r"a distributional formula is not offered for the dirichlet family .*D138"),
    (["cbind(a, b, c) ~ x", "sigmab ~ w"], "logistic_normal", r"a distributional formula is not offered for the logistic_normal family"),
])
def test_the_matrix_response_refusals(formula, family, needle):
    with pytest.raises(FormulaError, match=needle):
        parse_formula(formula, None, family)


@pytest.mark.parametrize("formula,needle", [
    ("cbind(a, b, c) ~ x + (1 + x | g)", r"a varying slope \('\(1 \+ x \| g\)'\) is not offered for the {f} family"),
    ("cbind(a, b, c) ~ x + (1 + x || g)", r"a varying slope \('\(1 \+ x \|\| g\)'\)"),
    ("cbind(a, b, c) ~ x + (1 |p| g)", r"a correlation ID \('\(1 \|p\| g\)'\) is not offered for the {f} family"),
    ("cbind(a, b, c) ~ x + offset(log(e))", r"an offset term is not offered for the {f} family .*divergence {d}"),
    ("cbind(a, b, c) ~ x + (1 | mm(g, h))", r"the multi-membership term '\(1 \| mm\(g, h\)\)' is not available for the {f} family"),
    ("cbind(a, b, c) ~ x + (1 | gr(g, by = h))", r"the term with gr\(by = \) .* is not available for the {f} family"),
    ("cbind(a, b, c) ~ mo(o)", r"the monotonic term 'mo\(o\)' is not available for the {f} family"),
    ("cbind(a, b, c) ~ cs(x)", r"category-specific effect cs\(\) is not supported for the {f} family"),
    ("cbind(a, b, c) ~ me(x, sx)", r"the measurement-error term 'me\(x, sx\)' is not available for the {f} family"),
    ("cbind(a, b, c) ~ mi(x)", r"the missing-value term 'mi\(x\)' is not available for the {f} family"),
    ("cbind(a, b, c) ~ x + ar(t, g, 1)", r"is not available for the {f} family; brms's ARMA terms"),
    ("mvbind(a, b) ~ x", r"a multivariate {f} model is not offered"),
])
@pytest.mark.parametrize("family,div", [("dirichlet", "D138"), ("logistic_normal", "D139")])
def test_the_family_refuses_what_no_frozen_case_verifies(formula, needle, family, div):
    with pytest.raises(FormulaError, match=needle.format(f=family, d=div)):
        parse_formula(formula, frame(), family)


def test_the_slope_refusal_directly():
    # the family's own check (simplex._check_formula) on a term the generic parse let through: gr(by = ) is refused by the
    # generic check first, so the family's branch for custom terms is reached here on its own
    check = sx._check_formula("dirichlet", "D138")
    custom = SimpleNamespace(formula="f", offset=None, group_terms=(SimpleNamespace(slopes=(), id=None, custom=True,
                                                                                  canonical="(1 | gr(g, by = h))"),))
    with pytest.raises(FormulaError, match=r"the group term '\(1 \| gr\(g, by = h\)\)' \(mm\(\) or gr\(by = \)\) is not offered"):
        check(custom)
    check(SimpleNamespace(formula="f", offset=None, group_terms=(SimpleNamespace(slopes=(), id=None, custom=False, canonical="(1 | g)"),)))


@pytest.mark.parametrize("change,needle", [
    (lambda d: d.assign(b=d["b"].astype(str)), r"response column 'b' has dtype .*; the dirichlet family's response is 3 numeric "
                                               r"proportions per row"),
    (lambda d: d.assign(a=np.r_[0.0, d["a"].to_numpy()[1:]]), r"response column 'a' has the value 0.0 at row 0; the dirichlet family's "
                                                              r"proportions lie strictly between 0 and 1"),
    (lambda d: d.assign(c=np.r_[np.nan, d["c"].to_numpy()[1:]]), r"response column 'c' has the value nan at row 0"),
    (lambda d: d.assign(c=d["c"] + np.r_[0.0, 2e-8, np.zeros(len(d) - 2)]), r"the response cbind\(a, b, c\) sums to .* at row 1; each "
                                                                             r"row of a dirichlet response must sum to 1 \(within 1e-08"),
    (lambda d: d.assign(a=d["a"] + 0j), r"response 'a' is complex-valued"),
    (lambda d: d.drop(columns="c"), r"columns not in data: \['c'\]"),
])
def test_the_response_must_lie_on_the_simplex(change, needle):
    with pytest.raises(FormulaError, match=needle):
        ab.dirichlet(F, change(frame()), dir_priors())


def test_a_sum_within_stans_tolerance_is_taken():
    d = frame()
    d["c"] = d["c"] + np.r_[5e-9, np.zeros(len(d) - 1)]
    assert sx.response_matrix(parse_formula(F, None, "dirichlet"), d, "dirichlet").shape == (len(d), 3)


def test_a_one_column_response_is_refused_for_the_simplex_families():
    from autobayes.formula import validate_data

    d = frame().assign(y=0.5)
    for fam in ("dirichlet", "logistic_normal"):
        needle = rf"the {fam} family's response is written cbind\(y1, \.\.\., yK\) ~ \.\.\., one column per category"
        with pytest.raises(FormulaError, match=r"^'y ~ x': " + needle):
            parse_formula("y ~ x", d, fam)                         # at parse time, before the data are read (a matrix column)
        with pytest.raises(FormulaError, match=r"^'ydir ~ x': " + needle):
            parse_formula("ydir ~ x", None, fam)
        with pytest.raises(FormulaError, match=needle):                # the response check behind it, for a spec built elsewhere
            validate_data(parse_formula("y ~ x"), d, fam)


# -- 2. categories and refcat -----------------------------------------------------------------------------------------
def test_the_categories_and_the_reference(dir_model, ln_model, ln_refcat):
    assert dir_model.categories == ln_model.categories == ("a", "b", "c")
    assert ln_refcat.categories == ("b", "a", "c") and ln_refcat.category_order.tolist() == [1, 0, 2]
    for m in (dir_model, ln_model, ln_refcat):
        np.testing.assert_array_equal(m.design.y, frame()[["a", "b", "c"]].to_numpy())      # brms's Y, in the columns' order
        np.testing.assert_array_equal(m.data[m.spec.response], frame()[m.categories[0]])  # Bambi's: the reference's column
        np.testing.assert_array_equal(m.response_matrix, m.design.y)
    spec = parse_formula(F, None, "logistic_normal")
    assert sx.categories_of(spec, None, get_family("logistic_normal"))[0] == ("a", "b", "c")
    assert sx.categories_of(spec, "c", get_family("logistic_normal"))[1].tolist() == [2, 0, 1]
    with pytest.raises(ValueError, match=r"refcat must name one of the response columns \['a', 'b', 'c'\], got 'd'"):
        sx.categories_of(spec, "d", get_family("logistic_normal"))
    with pytest.raises(ValueError, match=r"refcat must name one of the response columns .*got 2"):
        sx.categories_of(spec, 2, get_family("logistic_normal"))
    assert sx.categories_of(spec, "a", get_family("dirichlet"))[0] == ("a", "b", "c")        # the first column is its default
    with pytest.raises(ValueError, match=r"refcat='b': another reference category than the first column \('a'\) is not offered for "
                                         r"the dirichlet family .*divergence D138\); reorder the columns of cbind\(\)"):
        ab.dirichlet(F, frame(), dir_priors(), refcat="b")
    clash = frame().rename(columns={"b": "b_", "c": "b."})
    with pytest.raises(FormulaError, match=r"would share brms's parameter name 'mub'"):
        ab.dirichlet("cbind(a, b_, b.) ~ x", clash, dir_priors(sd=None))


def test_refcat_applies_to_the_simplex_families_only():
    with pytest.raises(ValueError, match=r"refcat= names the reference category of the logistic_normal family's cbind\(\) response; "
                                         r"the gaussian family takes none"):
        ab.Model("y ~ x", frame().assign(y=1.0 * np.arange(24)), ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1),
                 sigma=ab.HalfNormal(1)), family="gaussian", refcat="a")
    with pytest.raises(ValueError, match=r"refcat= names the reference category .* the poisson family takes none"):
        ab.default_priors("y ~ x", frame().assign(y=1), family="poisson", refcat="a")
    sx.check_refcat(get_family("dirichlet"), "a")                                        # checked further by categories_of
    assert get_family("dirichlet").simplex is not None and get_family("logistic_normal").simplex is not None
    assert not get_family("dirichlet").matrix_response and get_family("gaussian").simplex is None   # F01a's count matrices


# -- 3. priors ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("priors,needle", [
    (ln_priors(sigma=None), r"the logistic_normal family needs a prior on the SDs of its log-ratios: pass sigma=\.\.\."),
    (ln_priors(sigma={"b": ab.HalfNormal(1)}), r"prior for 'sigma' must be one Prior, shared by every non-reference category's "
                                               r"sigma<k> .*divergence D139\); got dict"),
    (ln_priors(sigma=ab.Normal(0, 1)), r"prior for 'sigma' must be a positive-support prior"),
    (ln_priors(lncor=None), r"the logistic_normal family needs lncor=ab.LKJ\(eta\)"),
    (ln_priors(lncor=ab.HalfNormal(1)), r"prior for 'lncor' must be ab.LKJ\(eta\)"),
    (ln_priors(phi=ab.Gamma(1, 1)), r"the logistic_normal family has no precision parameter; do not pass phi"),
])
def test_the_logistic_normal_priors(priors, needle):
    with pytest.raises(ab.PriorError, match=needle):
        ab.logistic_normal(F, frame(), priors)


def test_lncor_belongs_to_the_logistic_normal_family_alone():
    with pytest.raises(ab.PriorError, match=r"lncor is the prior of the logistic_normal family's correlation of its log-ratios "
                                            r"\(brms class lncor\); the dirichlet family has none"):
        ab.dirichlet(F, frame(), dir_priors(lncor=ab.LKJ(1)))
    with pytest.raises(ab.PriorError, match=r"the gaussian family has none"):
        ab.Priors(intercept=ab.Normal(0, 1), sigma=ab.HalfNormal(1), lncor=ab.LKJ(1)).validate(parse_formula("y ~ 1"), "gaussian")
    mv = ab.Priors(resp={"y": {"intercept": ab.Normal(0, 1), "sigma": ab.HalfNormal(1)},
                         "y2": {"intercept": ab.Normal(0, 1), "sigma": ab.HalfNormal(1)}}, lncor=ab.LKJ(1))
    with pytest.raises(ab.PriorError, match=r"lncor is the prior .*; a multivariate model has none"):
        ab.gaussian("mvbind(y, y2) ~ 1", frame().assign(y=np.arange(24.0), y2=np.arange(24.0) ** 0.5), mv, rescor=False)
    with pytest.raises(ab.PriorError, match=r"the dirichlet family needs a prior on its precision phi"):
        ab.dirichlet(F, frame(), dir_priors(phi=None))


# -- 4. the graph -------------------------------------------------------------------------------------------------------
def test_the_graph_and_its_names(dir_model, ln_model, ln_refcat):
    n = dir_model.names
    assert n["y"] == "cbind_a_b_c" and n["category_dim"] == "cbind_a_b_c_reduced_dim" and n["response_dim"] == "cbind_a_b_c_simplex_dim"
    assert n["phi"] == "phi" and "sigma" not in n and dir_model.simplex_names == {"response_dim": "cbind_a_b_c_simplex_dim"}
    pmm = dir_model.pymc_model
    assert {v.name for v in pmm.value_vars} == {"phi_log__", "Intercept", "x", "1|g_sigma_log__", "1|g_offset"}
    dims = pmm.named_vars_to_dims
    assert dims["1|g_sigma"] == (n["category_dim"],) and dims["1|g_offset"] == ("g__factor_dim", n["category_dim"])
    assert dims[n["y"]] == ("__obs__", n["response_dim"]) and list(pmm.coords[n["response_dim"]]) == ["a", "b", "c"]
    np.testing.assert_array_equal(np.asarray(pmm.rvs_to_values[pmm[n["y"]]].data), category_response(dir_model))
    roles = ln_refcat.simplex_names
    assert roles == {"response_dim": "cbind_a_b_c_simplex_dim", "sigma": ["sigmaa", "sigmac"], "L": "Llncor",
                     "coordinate": "Llncor_dim", "coordinate_T": "Llncor_dim_T"}
    assert ln_model.simplex_names["sigma"] == ["sigmab", "sigmac"] == sx.sigma_names(("a", "b", "c"))
    ln_n = ln_refcat.names
    assert ln_n["sigmaa"] == "sigmaa" and ln_n["Llncor"] == "Llncor" and "sigma" not in ln_n and "phi" not in ln_n
    vv = {v.name for v in ln_refcat.pymc_model.value_vars}
    assert vv == {"Intercept", "x", "1|g_sigma_log__", "1|g_offset", "sigmaa_log__", "sigmac_log__", "Llncor_cholesky_corr__"}
    assert list(ln_refcat.pymc_model.coords["Llncor_dim"]) == ["a", "c"]
    assert ab.gaussian("y ~ x", frame().assign(y=np.arange(24.0)), ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1),
                       sigma=ab.HalfNormal(1))).simplex_names == {}
    # the observed response is brms's Y in category order
    obs = ln_refcat.pymc_model.rvs_to_values[ln_refcat.pymc_model[ln_n["y"]]]
    np.testing.assert_array_equal(np.asarray(obs.data), category_response(ln_refcat))


def test_generated_names_may_not_collide():
    d = frame().rename(columns={"x": "sigmab"})
    with pytest.raises(FormulaError, match=r"generated variable/coordinate name collision: \['sigmab'\]"):
        ab.logistic_normal("cbind(a, b, c) ~ sigmab", d, ln_priors(sd=None))


@pytest.mark.parametrize("which", ["dir", "ln", "ln_refcat"])
def test_the_log_density_by_hand(which, dir_model, ln_model, ln_refcat):
    model = {"dir": dir_model, "ln": ln_model, "ln_refcat": ln_refcat}[which]
    rng = np.random.default_rng(7)
    logp, logp0 = model.pymc_model.compile_logp(), model.pymc_model.compile_logp(jacobian=False)
    dlogp = model.pymc_model.compile_dlogp()
    from pymc.sampling.jax import get_jaxified_logp

    jf = get_jaxified_logp(model.pymc_model)
    names = [v.name for v in model.pymc_model.value_vars]
    for r in (0.3, -0.6):
        nat = natural(model, rng, r)
        pt_ = as_point(model, nat)
        assert abs(float(logp(pt_)) - hand_logp(model, nat)) <= 1e-10 * abs(hand_logp(model, nat))
        assert abs(float(logp0(pt_)) - hand_logp(model, nat, jacobian=False)) <= 1e-10 * abs(hand_logp(model, nat, jacobian=False))
        assert np.isfinite(dlogp(pt_)).all()
        assert abs(float(jf([pt_[k] for k in names])) - float(logp(pt_))) <= 1e-12 * abs(float(logp(pt_)))


def test_more_terms_in_every_category():
    d = frame()
    m = ab.dirichlet("cbind(a, b, c) ~ x + z + (1 | g) + (1 | h)", d, dir_priors(sd={"g": ab.HalfNormal(1), "h": ab.HalfNormal(1)}),
                     categorical={"z": "p"})
    assert m.spec.columns == ("x", "z[q]", "z[r]") and list(m.design.group_blocks) == [("g", "Intercept"), ("h", "Intercept")]
    nat = natural(m, np.random.default_rng(3))
    assert abs(float(m.pymc_model.compile_logp()(as_point(m, nat))) - hand_logp(m, nat)) <= 1e-10 * abs(hand_logp(m, nat))
    only = ab.logistic_normal("cbind(c, a, b) ~ 1", d, ln_priors(b=None, sd=None))
    nat = natural(only, np.random.default_rng(4))
    assert only.categories == ("c", "a", "b") and nat["b"] is None
    assert abs(float(only.pymc_model.compile_logp()(as_point(only, nat))) - hand_logp(only, nat)) <= 1e-10 * abs(hand_logp(only, nat))


def test_the_densities_stay_finite_for_extreme_predictors():
    """A Dirichlet concentration below 1e-10 takes its lgamma from its logarithm (likelihoods._lgamma_of_shape): the density and
    its gradient stay finite on both paths where alpha itself underflows."""
    m = ab.dirichlet("cbind(a, b, c) ~ x", frame(), dir_priors(sd=None))
    pt_ = m.value_point(intercept=[800.0, -800.0], b=np.array([[0.1, -0.2]]), phi=3.0)
    assert np.isfinite(float(m.pymc_model.compile_logp()(pt_))) and np.isfinite(m.pymc_model.compile_dlogp()(pt_)).all()
    eta = np.array([[800.0, -800.0]])
    y = category_response(m)[:1]
    ours = sx.dirichlet_log_density(y, eta, 3.0)
    log_p = np.r_[0.0, 800.0, -800.0] - 800.0
    alpha_log = math.log(3.0) + log_p
    by_hand = (gammaln(3.0) - sum(-a - 0.5772156649015329 * math.exp(a) if a < math.log(1e-10) else gammaln(math.exp(a)) for a in alpha_log)
               + sum((math.exp(a) - 1.0) * math.log(v) for a, v in zip(alpha_log, y[0])))
    assert np.isfinite(ours).all() and abs(ours[0] - by_hand) <= 1e-12 * abs(by_hand)
    # the series is lgamma's to its last digit there: SciPy's lgamma at 1e-12 and 1e-11
    for x in (1e-12, 1e-11):
        assert abs(sx._lgamma(x, math.log(x)) - gammaln(x)) <= 1e-15 * gammaln(x)


def test_the_numpy_densities_equal_the_graphs(dir_model, ln_refcat):
    for model in (dir_model, ln_refcat):
        nat = natural(model, np.random.default_rng(11), -0.4)
        pmm = model.pymc_model
        rows = pmm.compile_fn(pmm.logp(vars=pmm.observed_RVs, sum=False), inputs=pmm.value_vars, point_fn=True,
                              on_unused_input="ignore")(as_point(model, nat))[0]
        eta = linear_predictor(model, nat)
        if model.family == "dirichlet":
            ours = sx.dirichlet_log_density(category_response(model), eta, nat["phi"])
        else:
            ours = sx.logistic_normal_log_density(category_response(model), eta, nat["sigma"], nat["Llncor"])
        np.testing.assert_allclose(ours, rows, rtol=1e-13)


# -- 5. value_point -----------------------------------------------------------------------------------------------------
def test_value_point_takes_one_column_per_category(dir_model, ln_refcat):
    nat = natural(ln_refcat, np.random.default_rng(5))
    pt_ = as_point(ln_refcat, nat)
    np.testing.assert_allclose(pt_["1|g_sigma_log__"], np.log(nat["sd"]["g"]))
    np.testing.assert_array_equal(pt_["1|g_offset"], nat["z"]["g"])
    np.testing.assert_allclose([pt_["sigmaa_log__"], pt_["sigmac_log__"]], np.log(nat["sigma"]))
    np.testing.assert_allclose(pt_["Llncor_cholesky_corr__"], [math.atanh(0.3)])
    base = {k: v for k, v in nat.items()}
    bad = [({"z": {"g": np.zeros(6)}}, r"z for 'g' must be a finite array of shape \(6, 2\): one row per level, one column per "
                                       r"non-reference category \['a', 'c'\]"),
           ({"z": {"g": np.full((6, 2), np.nan)}}, r"z for 'g' must be a finite array of shape \(6, 2\)"),
           ({"sd": {"g": 0.5}}, r"sd for 'g' must be 2 finite positive SDs, one per non-reference category \['a', 'c'\]"),
           ({"sd": {"g": [0.5, -1.0]}}, r"sd for 'g' must be 2 finite positive SDs"),
           ({"sigma": [0.5]}, r"sigma must be 2 finite positive SDs, one per non-reference category \['a', 'c'\]"),
           ({"sigma": [0.5, 0.0]}, r"sigma must be 2 finite positive SDs"),
           ({"sigma": None}, r"sigma \(2 SDs, one per non-reference category \['a', 'c'\]: brms's sigmaa, sigmac\) and Llncor \(the 2 x 2 "
                             r"Cholesky factor of their correlation\) are required for the logistic_normal family"),
           ({"Llncor": None}, r"are required for the logistic_normal family"),
           ({"Llncor": np.eye(3)}, r"Llncor must be a finite 2 x 2 matrix"),
           ({"Llncor": np.array([[1.0, 0.0], [0.6, 0.6]])}, r"Llncor must have unit-norm rows"),
           ({"phi": 2.0}, r"the logistic_normal family's SDs are given as sigma .*; shape and phi do not apply")]
    for change, needle in bad:
        with pytest.raises(ValueError, match=needle):
            ln_refcat.value_point(**{**base, **change})
    dnat = natural(dir_model, np.random.default_rng(6))
    with pytest.raises(ValueError, match=r"Llncor is the Cholesky factor of the logistic_normal family's correlation of its "
                                         r"log-ratios; the dirichlet family has none"):
        dir_model.value_point(**dnat, Llncor=np.eye(2))
    with pytest.raises(ValueError, match=r"phi \(the precision\) is required for the dirichlet family"):
        dir_model.value_point(**{**dnat, "phi": None, "sigma": 1.0})


def test_the_graph_checks_the_per_category_dims():
    m = ab.dirichlet(F, frame(), dir_priors())
    dims = m.pymc_model.named_vars_to_dims
    for name, wrong, needle in (("1|g_offset", ("g__factor_dim",), r"PyMC dims of '1\|g_offset' are \('g__factor_dim',\); expected "
                                                                    r"\('g__factor_dim', 'cbind_a_b_c_reduced_dim'\)"),
                                ("1|g_sigma", (), r"PyMC dims of '1\|g_sigma' are not \('cbind_a_b_c_reduced_dim',\): one SD per "
                                                  r"non-reference category")):
        saved = dims[name]
        dims[name] = wrong
        try:
            with pytest.raises(RuntimeError, match=needle):
                m._extract_design()
        finally:
            dims[name] = saved
    assert m._extract_design().y.shape == (24, 3)


# -- 6. predictions -----------------------------------------------------------------------------------------------------
def _nats(model, k=3, seed=21):
    rng = np.random.default_rng(seed)
    return [natural(model, rng, r) for r in np.linspace(-0.5, 0.5, k)]


def _hand_eta(model, nat, df, groups=True):
    """The linear predictors on the rows of ``df`` from the raw design (``predict.raw_design``) and raw-scale intercepts."""
    from autobayes.predict import raw_design

    X = raw_design(model, df)
    a = np.asarray(nat["intercept"]) - model.design.x_means @ nat["b"]
    eta = a[None, :] + X @ nat["b"]
    if groups:
        for t in model.spec.group_terms:
            levels = list(model.design.group_designs[t.factor].levels)
            idx = [levels.index(v) for v in df[t.factor].astype(str)]
            eta = eta + (nat["sd"][t.factor] * nat["z"][t.factor])[idx]
    return eta


def test_linear_predictions_and_the_dirichlet_mean(dir_model, ln_refcat):
    new = frame(8, seed=5).drop(columns=["a", "b", "c"])
    for model in (dir_model, ln_refcat):
        nats = _nats(model)
        idata = draws_of(model, nats)
        for df in (None, new):
            rows = model.data if df is None else new
            for groups in (True, False):
                lin = ab.predict(model, idata, newdata=df, kind="linear", include_group_effects=groups)
                assert lin["eta"].dims == ("chain", "draw", "__obs__", "category") and list(lin["category"].values) == list(model.categories[1:])
                for k, nat in enumerate(nats):
                    np.testing.assert_allclose(lin["eta"].to_numpy()[0, k], _hand_eta(model, nat, rows, groups), rtol=1e-12, atol=1e-12)
                if model.family == "dirichlet":
                    mean = ab.predict(model, idata, newdata=df, kind="mean", include_group_effects=groups)
                    assert list(mean["category"].values) == ["a", "b", "c"]
                    for k, nat in enumerate(nats):
                        np.testing.assert_allclose(mean["p"].to_numpy()[0, k], softmax(np.column_stack(
                            [np.zeros(len(rows)), _hand_eta(model, nat, rows, groups)]), axis=1), rtol=1e-12)
        if model.family == "dirichlet":
            np.testing.assert_allclose(sx._softmax(np.array([[0.2, -1.0]])), softmax([[0.0, 0.2, -1.0]], axis=1), rtol=1e-15)


def test_the_logistic_normal_mean_is_refused_as_brms_refuses_it(ln_model):
    idata = draws_of(ln_model, _nats(ln_model, 1))
    with pytest.raises(ValueError, match=r"kind must be 'response' \(draws on the simplex\) or 'linear' .*; kind='mean' is not offered: "
                                         r"the mean of a logistic-normal vector has no closed form, and brms refuses posterior_epred"):
        ab.predict(ln_model, idata, kind="mean")
    with pytest.raises(ValueError, match=r"Cannot compute expected values of the posterior predictive distribution for family "
                                         r"'logistic_normal'"):
        get_family("logistic_normal").inverse_link(np.zeros((2, 2)))


def test_response_draws_on_the_simplex(dir_model, ln_refcat):
    for model in (dir_model, ln_refcat):
        nats = _nats(model, 2)
        idata = draws_of(model, nats)
        ds = ab.predict(model, idata, kind="response", seed=4)
        assert ds["y"].dims == ("chain", "draw", "__obs__", "category") and list(ds["category"].values) == list(model.categories)
        y = ds["y"].to_numpy()
        assert y.shape == (1, 2, 24, 3) and np.allclose(y.sum(-1), 1.0) and (y > 0).all()
        np.testing.assert_array_equal(y, ab.predict(model, idata, kind="response", seed=4)["y"].to_numpy())
        eta = np.stack([_hand_eta(model, nat, model.data) for nat in nats])[None]
        rng = np.random.default_rng(4)
        if model.family == "dirichlet":
            assert "p" in ds
            want = sx.dirichlet_draws(eta, np.array([[nat["phi"] for nat in nats]])[..., None], rng)
        else:
            assert set(ds.data_vars) == {"y"}                                  # no mean: brms has no posterior_epred here
            sig = np.array([[nat["sigma"] for nat in nats]])
            L = np.array([[nat["Llncor"] for nat in nats]])
            want = sx.logistic_normal_draws(eta, sig, L, rng)
            e = np.random.default_rng(4).standard_normal(eta.shape)          # the log-ratios are eta + diag(sigma) L e
            np.testing.assert_allclose(np.log(y[..., 1:] / y[..., :1]), eta + sig[..., None, :] * np.einsum("...ij,...nj->...ni", L, e),
                                       rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(y, want, rtol=1e-14)


def test_the_draws_have_the_families_moments():
    rng = np.random.default_rng(0)
    eta = np.tile(np.array([0.4, -0.3]), (40000, 1))
    p = softmax([0.0, 0.4, -0.3])
    y = sx.dirichlet_draws(eta, 3.0, rng)
    np.testing.assert_allclose(y.mean(0), p, atol=0.005)
    np.testing.assert_allclose(y.var(0), p * (1 - p) / 4.0, rtol=0.03)                 # Var = p (1 - p) / (phi + 1)
    tiny = sx.dirichlet_draws(np.array([[700.0, -700.0]] * 10), 1e-3, rng)             # a concentration far below 1e-300
    assert np.isfinite(tiny).all() and np.allclose(tiny.sum(-1), 1.0)
    L = cholesky(0.5)
    z = sx.logistic_normal_draws(np.zeros((40000, 2)), np.array([0.5, 2.0]), L, rng)
    alr = np.log(z[:, 1:] / z[:, :1])
    np.testing.assert_allclose(np.cov(alr.T), np.diag([0.5, 2.0]) @ L @ L.T @ np.diag([0.5, 2.0]), rtol=0.03, atol=0.01)


def test_the_graphs_own_draws(dir_model, ln_refcat):
    """The likelihoods' and Llncor's forward draws (PyMC's random functions), with and without ``size`` and a generator."""
    for model in (dir_model, ln_refcat):
        y = pm.draw(model.pymc_model[model.names["y"]], random_seed=3)
        assert y.shape == (24, 3) and np.allclose(y.sum(1), 1.0) and (y > 0).all()
    L = pm.draw(ln_refcat.pymc_model["Llncor"], random_seed=5)
    assert L.shape == (2, 2) and L[0, 1] == 0 and abs((L ** 2).sum(1) - 1).max() < 1e-12
    mu = np.zeros(2)
    a = sx._dirichlet_random(mu, 2.0, np.zeros(3), rng=np.random.default_rng(1), size=(4,))
    b = sx._logistic_normal_random(mu, np.ones(2), np.eye(2), np.zeros(3), rng=np.random.default_rng(1), size=(4,))
    assert a.shape == b.shape == (4, 3) and np.allclose(a.sum(-1), 1.0) and np.allclose(b.sum(-1), 1.0)
    assert sx._dirichlet_random(mu, 2.0, np.zeros(3)).shape == sx._logistic_normal_random(mu, np.ones(2), np.eye(2), np.zeros(3)).shape == (3,)


def test_new_levels_are_refused_under_every_policy(dir_model, ln_model):
    new = frame(4, seed=9).drop(columns=["a", "b", "c"]).assign(g=["u", "new", "v", "new"])
    for model in (dir_model, ln_model):
        idata = draws_of(model, _nats(model, 1))
        for policy in ("population", "resample_observed"):
            with pytest.raises(FormulaError, match=rf"newdata has levels of 'g' not seen in training: \['new'\]; new levels in a "
                                                   rf"{model.family} model, whose every non-reference category has its own group term, "
                                                   rf"are not offered .*divergence {model.family_spec.simplex.divergence}"):
                ab.predict(model, idata, newdata=new, kind="linear", new_groups=policy)
        with pytest.raises(FormulaError, match=r"Pass new_groups='population'"):
            ab.predict(model, idata, newdata=new, kind="linear")
        assert ab.predict(model, idata, newdata=new, kind="linear", include_group_effects=False)["eta"].shape == (1, 1, 4, 2)


# -- 7. post-processing -------------------------------------------------------------------------------------------------
def test_log_lik_is_each_rows_density(dir_model, ln_refcat):
    for model in (dir_model, ln_refcat):
        nats = _nats(model)
        ll = ab.log_lik(model, draws_of(model, nats)).to_numpy()
        pmm = model.pymc_model
        rows = pmm.compile_fn(pmm.logp(vars=pmm.observed_RVs, sum=False), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
        for k, nat in enumerate(nats):
            np.testing.assert_allclose(ll[0, k], rows(as_point(model, nat))[0], rtol=1e-12)


def test_the_refusals_downstream(dir_model, ln_model):
    for model in (dir_model, ln_model):
        idata = draws_of(model, _nats(model, 2))
        div = model.family_spec.simplex.divergence
        with pytest.raises(ValueError, match=rf"bayes_R2\(\) is not offered for the {model.family} family: its response is K proportions "
                                             rf"per row, .*divergence {div}\)"):
            ab.bayes_R2(model, idata)
        with pytest.raises(ValueError, match=rf"prior_predictive\(\) is not offered for the {model.family} family: its response is K "
                                             rf"proportions per row .*divergence {div}\)"):
            ab.prior_predictive(model, draws=5, seed=1)
        with pytest.raises(Exception, match=rf"the {model.family} family's categories"):
            ab.hypothesis(model, idata, "x > 0")
        with pytest.raises(Exception, match=rf"the {model.family} family's categories"):
            ab.conditional_effects(model, idata, "x")
        with pytest.raises(Exception, match=rf"the {model.family} family's categories"):
            ab.prior_draws(model)


# -- 8. describe(), the identity and the fit record ---------------------------------------------------------------------------
def test_describe(dir_model, ln_refcat):
    text = dir_model.describe()
    for needle in ("AutoBayes Dirichlet model", "response  : cbind(a, b, c), 3 proportions per row summing to 1 (brms's matrix Y, in the "
                   "columns' order); categories a, b, c, the reference 'a' (the first column); the non-reference categories' linear "
                   "predictors are brms's mub, muc",
                   "u_g[j, k] = sd_g[k] * z_g[j, k], z_g[j, k] ~ Normal(0, 1), j = 0..5, for each non-reference category k = b, c",
                   "E[y_i] = softmax(0, mu_i) (brms's posterior_epred)", "sd_g[k] ~ HalfNormal(sigma=1)",
                   "y_i ~ Dirichlet(phi * softmax(0, mu_i)): brms's dirichlet_logit_lpdf", "phi ~ Gamma(alpha=2, beta=0.5)",
                   "1|g_sigma_log__          log(sd_g[b, c]) (vector; dims cbind_a_b_c_reduced_dim)",
                   "1|g_offset               z_g[j, k] (matrix; dims g__factor_dim x cbind_a_b_c_reduced_dim)"):
        assert needle in text, needle
    text = ln_refcat.describe()
    for needle in ("AutoBayes logistic-normal model", "categories b, a, c, the reference 'b' (refcat)", "brms's mua, muc",
                   "mu_i[k] is the mean of log(y_i[k] / y_i[b]); E[y_i] has no closed form (brms refuses posterior_epred).",
                   "sigmaa, sigmac ~ HalfNormal(sigma=1), each   (brms classes sigmaa, sigmac; one prior for all)",
                   "Lncor ~ LKJ(eta=2) (brms lkj(2) on the correlation matrix of the log-ratios",
                   "z_i = log(y_i[k] / y_i[b]), k = a, c: z_i ~ MultiNormal(mu_i, diag(sigma) Lncor diag(sigma))",
                   "sigmaa_log__             log(sigmaa), the SD of log(y[a] / y[b]) (brms's sigmaa)",
                   "Llncor_cholesky_corr__   Llncor, the Cholesky factor of the log-ratios' correlation"):
        assert needle in text, needle


def test_the_target_identity(dir_model, ln_model, ln_refcat):
    ids = {m.target_identity["sha256"] for m in (dir_model, ln_model, ln_refcat)}
    assert len(ids) == 3
    assert ab.logistic_normal(F, frame(), ln_priors()).target_identity == ln_model.target_identity
    assert ab.logistic_normal(F, frame(), ln_priors(lncor=ab.LKJ(1))).target_identity != ln_model.target_identity
    assert ab.logistic_normal("cbind(b, a, c) ~ x + (1 | g)", frame(), ln_priors()).target_identity != ln_refcat.target_identity
    assert ab.dirichlet(F, frame(), dir_priors(phi=ab.Gamma(1, 1))).target_identity != dir_model.target_identity


def test_the_fit_record(ln_refcat, monkeypatch):
    import autobayes.session as session

    idata = draws_of(ln_refcat, _nats(ln_refcat, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                   # four synthetic draws: R-hat divides by little
        rows = session._summary_rows(ln_refcat, idata)
    assert {"sigmaa", "sigmac", "1|g_sigma[a]", "Intercept[c]"} <= set(rows)
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=ln_refcat.target_identity, idata=idata, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=1, draws=4, tune=0,
                          target_accept=0.9, seed=1, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(ln_refcat, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["priors"]["sigma"] == "HalfNormal(sigma=1)" and rec["model"]["priors"]["lncor"] == "LKJ(eta=2)"
    assert rec["model"]["simplex"] == {"columns": ["a", "b", "c"], "categories": ["b", "a", "c"], "reference": "b",
                                       "names": {"response_dim": "cbind_a_b_c_simplex_dim", "sigma": ["sigmaa", "sigmac"], "L": "Llncor",
                                                 "coordinate": "Llncor_dim", "coordinate_T": "Llncor_dim_T"}}
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


# -- 9. brms's default priors -------------------------------------------------------------------------------------------
def _rows(table):
    return [(r["class"], r["coef"], r["group"], r["dpar"], r["prior"], r["lb"], r["source"]) for r in table]


def test_brms_default_prior_tables():
    """brms 2.23's default_prior() of these formulas (the frozen cases' tables, read here as brms printed them)."""
    d = frame()
    common = lambda dp: [("Intercept", "", "", dp, "student_t(3, 0, 2.5)", "", "default"), ("b", "", "", dp, "", "", "default"),
                         ("b", "x", "", dp, "", "", "default"), ("sd", "", "", dp, "student_t(3, 0, 2.5)", "0", "default"),
                         ("sd", "", "g", dp, "", "", "default"), ("sd", "Intercept", "g", dp, "", "", "default")]
    pri = ab.default_priors(F, d, family="dirichlet")
    m = ab.dirichlet(F, d, pri)
    assert _rows(m.prior_table()) == [("phi", "", "", "", "gamma(0.01, 0.01)", "0", "default"), *common("mub"), *common("muc")]
    assert m.default_prior_labels == ["intercept", "b:x", "sd:g", "phi"]
    pri = ab.default_priors(F, d, family="logistic_normal", refcat="b")
    m = ab.logistic_normal(F, d, pri, refcat="b")
    assert pri.sigma == ab.HalfStudentT(3, 2.5) and pri.lncor.params == {"eta": 1.0} and pri.lncor.source == "default"
    assert _rows(m.prior_table()) == [("lncor", "", "", "", "lkj(1)", "", "default"), ("sigmaa", "", "", "", "student_t(3, 0, 2.5)", "0", "default"),
                                      ("sigmac", "", "", "", "student_t(3, 0, 2.5)", "0", "default"), *common("mua"), *common("muc")]
    assert m.default_prior_labels == ["intercept", "b:x", "sd:g", "sigma", "lncor"]
    assert ab.logistic_normal(F, d, ln_priors()).default_prior_labels == []
    with pytest.raises(ValueError, match=r"refcat='c': another reference category .* the dirichlet family"):
        ab.default_priors(F, d, family="dirichlet", refcat="c")
    no_groups = ab.default_priors("cbind(a, b, c) ~ x", d, family="dirichlet")
    assert no_groups.sd is None and _rows(ab.dirichlet("cbind(a, b, c) ~ x", d, no_groups).prior_table())[0][0] == "phi"


def test_a_user_prior_table_per_category():
    m = ab.logistic_normal("cbind(a, b, c) ~ x + (1 | g) + (1 | h)", frame(), ln_priors(sd={"g": ab.HalfNormal(1), "h": ab.HalfNormal(2)}))
    rows = [(r["class"], r["coef"], r["group"], r["dpar"], r["prior"]) for r in m.prior_table() if r["class"] == "sd"]
    assert rows == [(c, coef, g, dp, p) for dp in ("mub", "muc") for c, coef, g, p in
                    [("sd", "", "", ""), ("sd", "", "g", "normal(0, 1)"), ("sd", "Intercept", "g", ""), ("sd", "", "h", "normal(0, 2)"),
                     ("sd", "Intercept", "h", "")]]
    with pytest.raises(ab.PriorError, match=r"the logistic_normal family's prior table needs the response categories"):
        m.priors.table(m.spec, m.family_spec)


# -- 10. the registry ---------------------------------------------------------------------------------------------------
def test_the_registry_entries():
    for name in ("dirichlet", "logistic_normal"):
        spec = FAMILIES[name]
        assert spec.per_category and spec.continuous and spec.simplex is not None and spec.manual_prediction
        assert spec.log_likelihood is not None and spec.response_categories is None and not spec.monotonic
    assert get_family("dirichlet").simplex is sx.DIRICHLET_PARTS and sx.DIRICHLET_PARTS.epred and not sx.LOGISTIC_NORMAL_PARTS.epred
    assert get_family("dirichlet").default_scale_prior(7.0) == ab.Gamma(0.01, 0.01)
    assert get_family("dirichlet").response_upper_bound(None) is None and get_family("dirichlet").response_warnings(None, None) == []
    assert get_family("logistic_normal").response_upper_bound(None) is None
    assert get_family("logistic_normal").response_warnings(None, None) == []


# -- 11. the shared paths these families extend keep their behaviour for every other model ------------------------------------
def test_the_shared_paths_for_other_models():
    d = frame().assign(y=np.linspace(-1.0, 2.0, 24))
    # an sd prior per coefficient of one factor (the default-prior table's last branch, without a dpar)
    pri = ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1),
                    sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(2)}})
    m = ab.gaussian("y ~ x + (1 + x || g)", d, pri)
    rows = [(r["class"], r["coef"], r["group"], r["dpar"], r["prior"]) for r in m.prior_table() if r["class"] == "sd"]
    assert rows == [("sd", "", "", "", ""), ("sd", "", "g", "", ""), ("sd", "Intercept", "g", "", "normal(0, 1)"),
                    ("sd", "x", "g", "", "normal(0, 2)")]
    # a one-column model's standardised effects are one vector per coefficient
    with pytest.raises(ValueError, match=r"z for 'g' must have 6 finite entries"):
        m.value_point(intercept=0.0, b=[0.1], sigma=1.0, sd={"g": {"Intercept": 1.0, "x": 1.0}},
                      z={"g": {"Intercept": np.zeros((6, 2)), "x": np.zeros(6)}})
    # a categorical model's response line, and its Bambi family's reference category
    c = ab.categorical("z ~ x", d, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1)))
    assert "z         : response categories 1=p, 2=q, 3=r (the reference is 'p')" in c.describe()
    dm = ab.dirichlet(F, frame(), dir_priors())
    fam = dm.bambi_model.family
    assert fam.get_reference(None) == "a" and fam.get_levels(None) == ["b", "c"]
    assert fam.get_coords(None) == {dm.names["category_dim"]: ["b", "c"]}
