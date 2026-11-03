"""Multivariate extensions (brms-parity feature G08, docs/multivariate-ext.md): a formula per response under a residual
correlation, a correlation ID across responses (``(1 |p| g)`` in ``mvbind()`` or in each response's formula) and ``weights()``
in multivariate models.

The grammar and every refusal, the split of the ``cor`` priors between the IDs across responses and the responses' own terms,
the assembled graph (the ID's ``L|g`` and ``cor|g``, the correlated effects), the log density of each construct against an
independent computation with scipy (the effects ``(diag(sd) L z)^T``, the joint density weighted once, the independent
densities weighted per response), value points, the refusal of joint predictions at new levels of an ID's factor,
brms's default priors and prior table, ``describe()`` and the target identity. Nothing samples. The numbers against brms are
the oracle's (tests/oracle, cases gauss_mv_separate_rescor, gauss_mv_id_across, gauss_mv_id_across_rescor, gauss_mv_weights
and gauss_mv_weights_rescor)."""
import re
from dataclasses import replace

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.special import betaln
from scipy.stats import multivariate_normal, norm

import autobayes as ab
from autobayes.formula import FormulaError
from autobayes.multivariate import (CrossId, MultivariateModel, _cor_rows, multivariate_prior_table, parse_multivariate, split_cor,
                                    weighted_logp)
from autobayes.priors import PriorError

LEVELS = ("a", "b", "c", "d", "e")


def frame(n=30, seed=11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = [LEVELS[i % len(LEVELS)] for i in range(n)]
    h = ["h1", "h2", "h3"][: 3] * (n // 3)
    x = rng.normal(size=n)
    e = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
    return pd.DataFrame({"x": x, "g": g, "h": h, "z": ["p", "q", "r"] * (n // 3), "w": rng.uniform(0.5, 2.0, n),
                         "w2": rng.uniform(0.5, 2.0, n), "s": [1, 0] * (n // 2), "c": [0] * n,
                         "y": 0.4 + 0.8 * x + e[:, 0], "y2": -0.2 + 0.3 * x + 0.8 * e[:, 1], "y3": rng.normal(size=n),
                         "y4": rng.normal(size=n)})


@pytest.fixture(scope="module")
def df():
    return frame()


def response(sd=0.5, group=True, b=True):
    out = {"intercept": ab.Normal(0, 5), "sigma": ab.HalfNormal(1)}
    out.update({"b": ab.Normal(0, 2)} if b else {})
    out.update({"sd": ab.HalfNormal(sd)} if group else {})
    return out


def priors(responses=("y", "y2"), cor=None, rescor=None, sds=(1.0, 0.5), **kwargs):
    return ab.Priors(resp={r: response(sd, **kwargs) for r, sd in zip(responses, sds)}, cor=cor, rescor=rescor)


ID = "mvbind(y, y2) ~ x + (1 |p| g)"


@pytest.fixture(scope="module")
def id_model(df):
    return ab.gaussian(ID, df, priors(cor={"g": ab.LKJ(2)}), rescor=False)


@pytest.fixture(scope="module")
def id_rescor(df):
    return ab.gaussian(ID, df, priors(cor={"g": ab.LKJ(2)}, rescor=ab.LKJ(3)), rescor=True)


def half(x, s):
    return float(norm.logpdf(x, 0, s)) + np.log(2.0)


def lkj2(L, eta):
    """Stan's lkj_corr_cholesky_lpdf of a 2 x 2 factor, with its constant: the LKJ density of r = L[1, 0]."""
    return -(2 * eta - 1) * np.log(2.0) - betaln(eta, eta) + (2 * eta - 2) * np.log(L[1, 1])


def corr(r):
    return np.array([[1.0, 0.0], [r, np.sqrt(1 - r * r)]])


PARS = {"y": dict(intercept=0.3, b=[0.7], sigma=0.9, sd=0.6), "y2": dict(intercept=-0.4, b=[0.2], sigma=1.2, sd=0.8)}


def id_point(model, L, Lrescor=None, seed=0):
    rng = np.random.default_rng(seed)
    resp = {r: {**PARS[r], "z": rng.normal(size=len(LEVELS))} for r in model.responses}
    return model.value_point(resp=resp, L={"g": L}, **({"Lrescor": Lrescor} if Lrescor is not None else {})), resp


# -- 1. the grammar --------------------------------------------------------------------------------------------------
def test_ids_across_responses_are_parsed_and_written_back(df):
    s = parse_multivariate(ID, df, rescor=False)
    assert s.cross_ids == (CrossId("p", "g", ("y", "y2")),)
    assert s.cross_ids[0].coefs == (("y", "Intercept"), ("y2", "Intercept")) and s.cross_ids[0].labels == ("y_Intercept", "y2_Intercept")
    assert s.formulas == {"y": "y ~ x + (1 | g)", "y2": "y2 ~ x + (1 | g)"}       # each response's own (1 | g)
    assert s.canonical_formula == ID and s.rhs == "x + (1 |p| g)"
    sep = parse_multivariate(["y ~ x + (1 |p| g)", "y2 ~ z + (1 |p| g)", "y3 ~ x"], df, categorical={"z": "p"}, rescor=False)
    assert sep.cross_ids == (CrossId("p", "g", ("y", "y2")),) and sep.formulas["y2"] == "y2 ~ z + (1 | g)"
    assert sep.canonical_formula == "bf(y ~ x + (1 |p| g)) + bf(y2 ~ z + (1 |p| g)) + bf(y3 ~ x)"
    w = parse_multivariate("mvbind(y, y2) | weights(w) ~ x", df, rescor=True)
    assert w.formulas["y2"] == "y2 | weights(w) ~ x" and w.canonical_formula == "mvbind(y, y2) | weights(w) ~ x" and w.rescor
    assert parse_multivariate(["y | weights(w) ~ x", "y2 | weights(w) ~ z"], df, categorical={"z": "p"}, rescor=True).rescor


@pytest.mark.parametrize("formula, rescor, message", [
    ("mvbind(y, y2) ~ x + (1 + x |p| g)", False, "'(1 + x |p| g)': varying slopes (or a term without the varying intercept) in a "
                                                 "correlation ID across responses are not supported"),
    ("mvbind(y, y2) ~ x + (0 + x |p| g)", False, "in a correlation ID across responses are not supported"),
    ("mvbind(y, y2) ~ x + (1 |p| g:h)", False, "a correlation ID across responses is offered on a grouping column, '(1 |p| g)'; the "
                                               "grouping 'g:h'"),
    ("mvbind(y, y2) ~ x + (1 |p| g) + (1 |p| h)", False, "the correlation ID 'p' appears in two terms of one formula"),
    ("mvbind(y, y2) ~ x + (1 ||p| g)", False, "a correlation ID across responses is written '(1 |p| g)', the ID between two single bars"),
    (["y ~ x + (1 |p| g)", "y2 ~ x + (1 |p| h)"], False, "the correlation ID 'p' joins the grouping factors ['g', 'h']; brms refuses it"),
    (["y ~ x + (1 |p| g)", "y2 ~ x"], False, "the correlation ID 'p' of '(1 |p| g)' is used by the response 'y' alone"),
    (["y ~ (1 |p| g)", "y2 ~ (1 |p| g)", "y3 ~ (1 |q| g)", "y4 ~ (1 |q| g)"], False,
     "two correlation IDs across responses on the grouping factor 'g' are not supported"),
    (["y | subset(s) ~ x + (1 |p| g)", "y2 ~ x + (1 |p| g)"], False, "a correlation ID across responses with subset() is not supported"),
    (["y ~ mi(x) + (1 |p| g)", "x | mi() ~ (1 |p| g)"], False, "a correlation ID across responses in a model with mi() is not supported"),
    ("mvbind(y, y2) | cens(c) ~ x", False, "addition terms other than weights() (cens(c)) on the responses of a multivariate model"),
    ("mvbind(y, y2) foo ~ x", False, "'foo' after mvbind(): addition terms other than weights() (foo)"),
    ("mvbind(y, y2) | weights(y2) ~ x", False, "the response(s) ['y2'] cannot also be the weights of the responses"),
    (["y | weights(y2) ~ x", "y2 ~ x"], False, "the response(s) ['y2'] cannot also be the weights of a response"),
    (["y | weights(w) ~ x", "y2 | weights(w2) ~ x"], True, "every response must have the same weights, got y: weights(w), y2: "
                                                           "weights(w2); brms refuses differing weights columns"),
    (["y | weights(w) ~ x", "y2 ~ x"], True, "got y: weights(w), y2: no weights"),
    (["y | subset(s) ~ x", "y2 | weights(w) ~ x"], False, "weights() (on ['y2']) in a multivariate model with subset() is not supported"),
])
def test_refusals(df, formula, rescor, message):
    """Refused on the structure (no data), and on the data where the data do not refuse first."""
    with pytest.raises(FormulaError, match=re.escape(message)):
        parse_multivariate(formula, None, rescor=rescor)
    if "weights(y2)" not in str(formula):                      # a response's values are no weights: refused on the data first
        with pytest.raises(FormulaError, match=re.escape(message)):
            parse_multivariate(formula, df, rescor=rescor)


# -- 2. the priors -----------------------------------------------------------------------------------------------------
def test_split_cor(df):
    spec = parse_multivariate(ID, df, rescor=False)
    own = parse_multivariate("mvbind(y, y2) ~ x + (1 |p| g) + (1 + x | h)", df, rescor=False)
    plain = parse_multivariate("mvbind(y, y2) ~ x + (1 + x | h)", df, rescor=False)
    lkj2_, lkj3 = ab.LKJ(2), ab.LKJ(3)
    assert split_cor(ab.Priors(cor={"h": lkj2_}), plain) == ({}, {"h": lkj2_})            # no ID across responses: as before
    assert split_cor(ab.Priors(cor={"g": lkj2_}), spec) == ({"g": lkj2_}, None)
    assert split_cor(ab.Priors(cor={"*": lkj3}), spec) == ({"g": lkj3}, None)              # the class-wide prior, no term of its own
    assert split_cor(ab.Priors(cor=lkj2_), spec) == ({"g": lkj2_}, None)                    # one matrix in all: a bare prior
    assert split_cor(ab.Priors(cor={"*": lkj3, "h": lkj2_}), own) == ({"g": lkj3}, {"*": lkj3, "h": lkj2_})
    assert split_cor(ab.Priors(cor={"g": lkj2_, "h": lkj3}), own) == ({"g": lkj2_}, {"h": lkj3})
    for pr, sp, message in [
        (ab.Priors(), spec, "the correlation ID 'p' across the responses ['y', 'y2'] gives their varying intercepts of 'g' a correlation "
                            "matrix, which needs an LKJ prior: pass cor={'g': ab.LKJ(eta)}"),
        (ab.Priors(cor={"h": lkj2_}), own, "cor has no prior for ['g'], whose varying intercepts a correlation ID correlates"),
        (ab.Priors(cor=lkj2_), own, "multiple correlated grouping factors (['g', 'h']) require an exact-key cor mapping"),
        (ab.Priors(cor={"g": ab.Normal(0, 1)}), spec, "prior for 'cor:g' must be a correlation prior"),
    ]:
        with pytest.raises(PriorError, match=re.escape(message)):
            split_cor(pr, sp)


def test_priors_of_a_model_with_an_id(df):
    with pytest.raises(PriorError, match=re.escape("needs an LKJ prior: pass cor={'g': ab.LKJ(eta)}")):
        ab.gaussian(ID, df, priors(), rescor=False)
    sd = {"g": ab.HalfNormal(1), "h": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}}
    pr = ab.Priors(resp={r: {**response(), "sd": sd} for r in ("y", "y2")}, cor={"g": ab.LKJ(2), "h": ab.LKJ(3)})
    m = ab.gaussian("mvbind(y, y2) ~ x + (1 |p| g) + (1 + x | h)", df, pr, rescor=False)
    assert m.id_cor_priors == {"g": ab.LKJ(2)} and dict(m.submodels["y"].priors.cor) == {"h": ab.LKJ(3)}   # each its own
    assert {"L|g_cholesky_corr__", "y_L|h_cholesky_corr__", "y2_L|h_cholesky_corr__"} <= {v.name for v in m.pymc_model.value_vars}


# -- 3. the graph and its density ---------------------------------------------------------------------------------------
def manual_logp(model, df, resp, L, Lrescor=None, eta=2.0, eta_rescor=3.0, weights=None):
    """The log density without the Jacobian, written out: each response's priors (Normal intercept and slope, half-normal sigma
    and sd), the ID's LKJ density, the standard normal z, and the likelihood with brms's effects r = (diag(sd) L z)^T: the joint
    multivariate normal under a residual correlation (weighted once), independent normals otherwise (each weighted)."""
    x = df["x"].to_numpy() - df["x"].mean()
    codes = pd.Index(LEVELS).get_indexer(df["g"])
    sds = {r: (1.0, 0.5)[k] for k, r in enumerate(model.responses)}
    out = lkj2(L, eta)
    zs = [resp[r]["z"] for r in model.responses]
    effects = (np.diag([resp[r]["sd"] for r in model.responses]) @ L @ np.vstack(zs)).T     # (levels, responses)
    mus = []
    for k, r in enumerate(model.responses):
        a = resp[r]
        out += float(norm.logpdf(a["intercept"], 0, 5) + norm.logpdf(a["b"][0], 0, 2)) + half(a["sigma"], 1.0) + half(a["sd"], sds[r])
        out += float(norm.logpdf(a["z"]).sum())
        mus.append(a["intercept"] + x * a["b"][0] + effects[codes, k])
    w = np.ones(len(df)) if weights is None else weights
    Y = df[list(model.responses)].to_numpy()
    sig = np.array([resp[r]["sigma"] for r in model.responses])
    if Lrescor is None:
        return out + float((w[:, None] * norm.logpdf(Y, np.column_stack(mus), sig)).sum())
    S = np.diag(sig) @ Lrescor @ Lrescor.T @ np.diag(sig)
    ll = np.array([multivariate_normal(mean=m, cov=S).logpdf(y) for m, y in zip(np.column_stack(mus), Y)])
    return out + lkj2(Lrescor, eta_rescor) + float((w * ll).sum())


def test_id_graph_names_and_effects(id_model):
    m, pmm = id_model, id_model.pymc_model
    roles = m.id_names["g"]
    assert roles == {"L": "L|g", "cor": "cor|g", "coordinate": "g__id_dim", "coordinate_T": "g__id_dim_T", "id": "p",
                     "effects": ("y_1|g", "y2_1|g")}
    assert m.names["L:g"] == "L|g" and m.names["cor:g"] == "cor|g"
    assert list(pmm.coords["g__id_dim"]) == ["y_Intercept", "y2_Intercept"]
    assert "L|g_cholesky_corr__" in {v.name for v in pmm.value_vars}
    L = corr(0.55)
    vp, resp = id_point(m, L)
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["y_1|g"], pmm["y2_1|g"], pmm["cor|g"]]), inputs=pmm.value_vars, point_fn=True,
                        on_unused_input="ignore")
    ry, ry2, cor = fn(vp)
    np.testing.assert_allclose(ry, 0.6 * resp["y"]["z"], rtol=1e-14)                        # the first response's: sd z
    np.testing.assert_allclose(ry2, 0.8 * (0.55 * resp["y"]["z"] + L[1, 1] * resp["y2"]["z"]), rtol=1e-14)
    np.testing.assert_allclose(cor, L @ L.T, rtol=1e-14)


@pytest.mark.parametrize("rho, seed", [(0.55, 0), (-0.8, 3)])
def test_id_log_density_against_scipy(id_model, id_rescor, df, rho, seed):
    L, Lr = corr(rho), corr(0.3)
    vp, resp = id_point(id_model, L, seed=seed)
    got = float(id_model.pymc_model.compile_logp(jacobian=False)(vp))
    expected = manual_logp(id_model, df, resp, L)
    assert abs(got - expected) <= 1e-10 * max(1.0, abs(expected))
    vp, resp = id_point(id_rescor, L, Lrescor=Lr, seed=seed)
    got = float(id_rescor.pymc_model.compile_logp(jacobian=False)(vp))
    expected = manual_logp(id_rescor, df, resp, L, Lrescor=Lr)
    assert abs(got - expected) <= 1e-10 * max(1.0, abs(expected))


def test_value_point_takes_the_id_s_cholesky_factor(id_model, df):
    _, resp = id_point(id_model, corr(0.2))
    for kwargs, message in [({}, "L must map exactly the grouping factors of the correlation IDs across responses ['g']"),
                            ({"L": {"h": corr(0.2)}}, "L must map exactly the grouping factors"),
                            ({"L": {"g": np.eye(3)}}, "L for 'g' must be a finite 2 x 2 matrix")]:
        with pytest.raises(ValueError, match=re.escape(message)):
            id_model.value_point(resp=resp, **kwargs)
    plain = ab.gaussian("mvbind(y, y2) ~ x + (1 | g)", df, priors(), rescor=False)
    with pytest.raises(ValueError, match=re.escape("this model has no correlation ID across responses; L does not apply")):
        plain.value_point(resp=resp, L={"g": corr(0.2)})


def test_the_graph_check_names_what_is_wrong(id_model, monkeypatch):
    real = MultivariateModel.id_names.fget(id_model)
    for broken, message in [({**real["g"], "coordinate": "g__factor_dim"}, "is not a free 2 x 2 correlation Cholesky factor"),
                            ({**real["g"], "effects": ("y_1|g", "y_mu")}, "the effects 'y_mu' are not correlated through 'L|g'")]:
        monkeypatch.setattr(MultivariateModel, "id_names", property(lambda self, b=broken: {"g": b}))
        with pytest.raises(RuntimeError, match=re.escape(message)):
            id_model._check_graph()
        monkeypatch.undo()


def test_weights_without_rescor_weight_each_response(df):
    m = ab.gaussian("mvbind(y, y2) | weights(w) ~ x + (1 | g)", df, priors(), rescor=False)
    assert m.joint_weights is None and m.observed_names == ("y", "y2")
    for r in m.responses:
        np.testing.assert_array_equal(m.submodels[r].design.weights, df["w"].to_numpy())
    rng = np.random.default_rng(1)
    resp = {r: {**PARS[r], "z": rng.normal(size=len(LEVELS))} for r in m.responses}
    got = float(m.pymc_model.compile_logp(jacobian=False)(m.value_point(resp=resp)))
    # the same density as the ID model's at L = I (uncorrelated effects), minus the LKJ term, with each response weighted
    expected = manual_logp(m, df, resp, np.eye(2), weights=df["w"].to_numpy()) - lkj2(np.eye(2), 2.0)
    assert abs(got - expected) <= 1e-10 * max(1.0, abs(expected))
    assert "weights(w)" in m.describe() and m.spec.canonical_formula == "mvbind(y, y2) | weights(w) ~ x + (1 | g)"


def test_weights_with_rescor_weight_the_joint_density_once(df):
    m = ab.gaussian("mvbind(y, y2) | weights(w) ~ x + (1 | g)", df, priors(rescor=ab.LKJ(3)), rescor=True)
    np.testing.assert_array_equal(m.joint_weights, df["w"].to_numpy())
    rng = np.random.default_rng(2)
    resp = {r: {**PARS[r], "z": rng.normal(size=len(LEVELS))} for r in m.responses}
    Lr = corr(-0.45)
    got = float(m.pymc_model.compile_logp(jacobian=False)(m.value_point(resp=resp, Lrescor=Lr)))
    expected = manual_logp(m, df, resp, np.eye(2), Lrescor=Lr, weights=df["w"].to_numpy()) - lkj2(np.eye(2), 2.0)
    assert abs(got - expected) <= 1e-10 * max(1.0, abs(expected))
    twice = manual_logp(m, df, resp, np.eye(2), Lrescor=Lr, weights=2 * df["w"].to_numpy()) - lkj2(np.eye(2), 2.0)
    assert abs(got - twice) > 1.0                                                           # not weighted per response
    text = m.describe()
    assert "weights(w): each row's joint log density is multiplied by its weight, once" in text
    same = ab.gaussian(["y | weights(w) ~ x + (1 | g)", "y2 | weights(w) ~ x + (1 | g)"], df, priors(rescor=ab.LKJ(3)), rescor=True)
    np.testing.assert_array_equal(same.joint_weights, df["w"].to_numpy())


def test_weighted_logp_is_the_weighted_multi_normal():
    import pytensor
    import pytensor.tensor as pt

    rng = np.random.default_rng(4)
    w, L, sigma = rng.uniform(0.5, 2, 5), corr(0.35), np.array([0.8, 1.4])
    mu, y = rng.normal(size=(5, 2)), rng.normal(size=(5, 2))
    v, m, s, l = pt.matrix("v"), pt.matrix("m"), pt.vector("s"), pt.matrix("l")
    got = pytensor.function([v, m, s, l], weighted_logp(w)(pt.specify_shape(v, (None, 2)), m, s, l))(y, mu, sigma, L)
    S = np.diag(sigma) @ L @ L.T @ np.diag(sigma)
    np.testing.assert_allclose(got, w * np.array([multivariate_normal(mean=a, cov=S).logpdf(b) for a, b in zip(mu, y)]), rtol=1e-12)


def test_a_formula_per_response_with_rescor(df):
    pr = ab.Priors(resp={"y": response(1.0), "y2": response(group=False)}, rescor=ab.LKJ(2))
    m = ab.gaussian(["y ~ x + (1 | g)", "y2 ~ z"], df, pr, rescor=True, categorical={"z": "p"})
    assert m.rescor and m.spec.separate and m.observed_names == ("mvbind(y, y2)",)
    assert m.spec.canonical_formula == "bf(y ~ x + (1 | g)) + bf(y2 ~ z)"
    rng = np.random.default_rng(5)
    resp = {"y": {**PARS["y"], "z": rng.normal(size=len(LEVELS))}, "y2": {"intercept": 0.2, "b": [0.4, -0.7], "sigma": 1.1}}
    Lr = corr(0.6)
    got = float(m.pymc_model.compile_logp(jacobian=False)(m.value_point(resp=resp, Lrescor=Lr)))
    x = df["x"].to_numpy() - df["x"].mean()
    q, r = (df["z"] == "q").to_numpy(float), (df["z"] == "r").to_numpy(float)
    mu_y = 0.3 + 0.7 * x + 0.6 * resp["y"]["z"][pd.Index(LEVELS).get_indexer(df["g"])]
    mu_y2 = 0.2 + 0.4 * (q - q.mean()) - 0.7 * (r - r.mean())
    S = np.diag([0.9, 1.1]) @ Lr @ Lr.T @ np.diag([0.9, 1.1])
    expected = sum(multivariate_normal(mean=[a, b], cov=S).logpdf(yy) for a, b, yy in zip(mu_y, mu_y2, df[["y", "y2"]].to_numpy()))
    expected += float(norm.logpdf(0.3, 0, 5) + norm.logpdf(0.7, 0, 2)) + half(0.9, 1.0) + half(0.6, 1.0) + float(norm.logpdf(resp["y"]["z"]).sum())
    expected += float(norm.logpdf(0.2, 0, 5) + norm.logpdf([0.4, -0.7], 0, 2).sum()) + half(1.1, 1.0) + lkj2(Lr, 2.0)
    assert abs(got - expected) <= 1e-10 * max(1.0, abs(expected))


# -- 4. predictions -----------------------------------------------------------------------------------------------------
def test_joint_predictions_refuse_new_levels_of_an_id_s_factor(id_model, df):
    post = {f"{r}_{k}": (("chain", "draw"), np.full((1, 2), v)) for r in ("y", "y2")
            for k, v in (("Intercept", 0.1), ("x", 0.5), ("sigma", 1.0), ("1|g_sigma", 0.5))}
    post.update({f"{r}_1|g": (("chain", "draw", "g__factor_dim"), np.zeros((1, 2, len(LEVELS)))) for r in ("y", "y2")})
    idata = az.InferenceData(posterior=xr.Dataset(post, coords={"chain": [0], "draw": [0, 1], "g__factor_dim": list(LEVELS)}))
    new = df.head(3).assign(g=["a", "b", "zz"])
    with pytest.raises(FormulaError, match=re.escape("newdata has levels of 'g' not seen in training: ['zz'], whose effects a correlation "
                                                     "ID correlates across the responses; drawing them jointly is not supported")):
        id_model.predict(idata, newdata=new, new_groups="population", kind="linear")
    one = id_model.predict(idata, newdata=new, new_groups="population", kind="linear", resp="y2", seed=1)   # one response's marginal
    assert one["eta"].shape == (1, 2, 3)
    both = id_model.predict(idata, newdata=df.head(3), kind="linear")
    np.testing.assert_allclose(both["eta"].sel(response="y2").to_numpy(), np.broadcast_to(0.1 + 0.5 * df["x"].head(3).to_numpy(), (1, 2, 3)))


# -- 5. default priors, the prior table, describe() and the target identity ---------------------------------------------
def test_default_priors_and_prior_table(df):
    pr = ab.default_priors(ID, df, rescor=False)
    assert dict(pr.cor) == {"g": ab.LKJ(1)} and pr.cor["g"].source == "default"
    rows = [r for r in ab.gaussian(ID, df, pr, rescor=False).prior_table() if r["class"] == "cor"]
    assert [(r["prior"], r["group"], r["source"]) for r in rows] == [("lkj(1)", "", "default"), ("", "g", "default")]
    spec = parse_multivariate("mvbind(y, y2) ~ x + (1 |p| g) + (1 + x | h)", df, rescor=False)
    sd = {"g": ab.HalfNormal(1), "h": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}}
    base = ab.Priors(resp={r: {**response(), "sd": sd} for r in ("y", "y2")})

    def table(cor):
        return [(r["prior"], r["group"], r["source"]) for r in multivariate_prior_table(replace(base, cor=cor), spec) if r["class"] == "cor"]

    assert table({"g": ab.LKJ(2), "h": ab.LKJ(3)}) == [("", "", "default"), ("lkj(2)", "g", "user"), ("lkj(3)", "h", "user")]
    assert table({"g": ab.LKJ(2), "h": ab.LKJ(2)}) == [("lkj(2)", "", "user"), ("", "g", "user"), ("", "h", "user")]
    assert table({"*": ab.LKJ(2), "h": ab.LKJ(3)}) == [("lkj(2)", "", "user"), ("", "g", "user"), ("lkj(3)", "h", "user")]
    assert _cor_rows({"g": ab.LKJ(2)}, None)[0]["prior"] == "lkj(2)"


def test_describe(id_model, df):
    text = id_model.describe()
    assert "Group-level effects correlated across responses (brms's correlation ID)" in text
    assert "(u_y,g[j], u_y2,g[j]) = diag(sd_y,g, sd_y2,g) L_g z[j]" in text and "L|g ~ LKJ(eta=2)" in text
    assert "L|g_cholesky_corr__" in text and "cor|g = L|g L|g^T (brms Cor_m)" in text
    assert "uncorrelated across responses (no shared ID)" not in text
    pr = ab.Priors(resp={r: {**response(), "sd": {"g": ab.HalfNormal(1), "h": ab.HalfNormal(1)}} for r in ("y", "y2")}, cor={"g": ab.LKJ(2)})
    other = ab.gaussian("mvbind(y, y2) ~ x + (1 |p| g) + (1 | h)", df, pr, rescor=False)
    assert "The group-level terms of ['h'] are separate per response and uncorrelated across responses." in other.describe()


def test_target_identity(id_model, df):
    same = ab.gaussian(ID, df, priors(cor={"g": ab.LKJ(2)}), rescor=False)
    assert same.target_identity == id_model.target_identity
    assert ab.gaussian(ID, df, priors(cor={"g": ab.LKJ(4)}), rescor=False).target_identity != id_model.target_identity
    assert ab.gaussian("mvbind(y, y2) ~ x + (1 | g)", df, priors(), rescor=False).target_identity != id_model.target_identity


def test_the_syntax_audit_compares_ids_across_responses():
    """The corpus's mv_id_across reads as brms's; the same formula without the ID does not (tests/brms_syntax/audit.py)."""
    from tests.brms_syntax import audit

    store, frame_ = audit.load_json(audit.STANDATA), audit.python_frame()
    entry = next(e for e in audit.load_json(audit.CORPUS)["entries"] if e["id"] == "mv_id_across")
    assert audit.run_entry(entry, frame_, store)["issues"] == []
    plain = ab.gaussian("mvbind(y, y2) ~ x + (1 | g)", frame_, ab.default_priors("mvbind(y, y2) ~ x + (1 | g)", frame_, rescor=False),
                        rescor=False)
    issues, _ = audit.compare(plain, "mv_id_across", store)
    assert issues == ["brms correlates the group-level effects [('g', ('y', 'y2'))] across responses; AutoBayes's correlation IDs "
                      "across responses are []"]
