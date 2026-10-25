"""Multivariate Gaussian models, brms's ``mvbind(y, y2) ~ ...`` with and without residual correlation (brms-parity feature
C01; docs/multivariate.md).

The grammar and its refusals, the priors by response (coverage, the shared ``cor``, ``rescor``), the assembled graph
(names, coordinates, value variables) against the responses' own graphs, the joint density against scipy's multivariate
normal and the closed-form LKJ density, ``value_point``, predictions (the response axis, joint draws, seeds, unseen levels),
the fit path's sampling call and cleaning with ``pm.sample`` replaced (nothing samples), brms's default priors and prior
table, ``describe()``, the target identity and the refusals of what is not part of the slice. The numbers against brms
are the oracle's (tests/oracle, cases gauss_mv_rescor, gauss_mv_norescor and gauss_mv_slopes)."""
import re

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.special import betaln
from scipy.stats import multivariate_normal

import autobayes as ab
from autobayes.correlation import cholesky_corr_constrain, cholesky_corr_free
from autobayes.identity import FIT_IDENTITY_ATTR
from autobayes.multivariate import (RESPONSE_DIM, RESPONSE_DIM_T, MultivariateSpec, multi_normal_cholesky_logp,
                                    parse_multivariate)

LEVELS = ("a", "b", "c", "d")


def frame(n=28, seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = [LEVELS[i % len(LEVELS)] for i in range(n)]
    x, w = rng.normal(size=n), rng.normal(0.5, 1.0, n)
    e = rng.multivariate_normal([0, 0, 0], [[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]], size=n)
    return pd.DataFrame({"x": x, "w": w, "g": g, "z": ["p", "q", "r", "q"] * (n // 4), "y": 0.4 + 0.8 * x + e[:, 0],
                         "y2": -0.2 + 0.3 * x + 0.8 * e[:, 1], "y3": 1.0 - 0.5 * x + 1.2 * e[:, 2]})


def response(sigma=1.0, sd=None, intercept=0.0):
    return {"intercept": ab.Normal(intercept, 5), "b": ab.Normal(0, 2), "sd": sd or ab.HalfNormal(1), "sigma": ab.HalfNormal(sigma)}


def priors(rescor=True, eta=2.0, responses=("y", "y2"), **extra):
    return ab.Priors(resp={r: response(1.0 + k) for k, r in enumerate(responses)}, rescor=ab.LKJ(eta) if rescor else None, **extra)


F = "mvbind(y, y2) ~ x + (1 | g)"


@pytest.fixture(scope="module")
def data():
    return frame()


@pytest.fixture(scope="module")
def rescor(data):
    return ab.gaussian(F, data, priors(), rescor=True)


@pytest.fixture(scope="module")
def norescor(data):
    return ab.gaussian(F, data, priors(rescor=False), rescor=False)


@pytest.fixture(scope="module")
def three(data):
    """Three responses, correlated slopes per response under one shared cor prior, and a residual correlation."""
    sd = {"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}}
    pr = ab.Priors(resp={r: response(sd=sd) for r in ("y", "y2", "y3")}, cor={"g": ab.LKJ(2)}, rescor=ab.LKJ(1.5))
    return ab.gaussian("mvbind(y, y2, y3) ~ x + (1 + x | g)", data, pr, rescor=True)


def point(model, seed=0, rho=None):
    """A value point of ``model`` with every response's parameters drawn at random, and ``Lrescor`` with the correlations
    ``rho`` (K = 2: a number; otherwise a valid Cholesky factor)."""
    rng = np.random.default_rng(seed)
    resp = {}
    for r, sub in model.submodels.items():
        args = {"intercept": rng.normal(), "b": rng.normal(size=len(sub.spec.columns)), "sigma": float(rng.uniform(0.5, 1.5))}
        sd, z = {}, {}
        for t in sub.spec.group_terms:
            J = sub.design.group_designs[t.factor].J
            sd[t.factor] = {c: float(rng.uniform(0.3, 1.0)) for c in t.coefs} if t.slopes else float(rng.uniform(0.3, 1.0))
            z[t.factor] = {c: rng.normal(size=J) for c in t.coefs} if t.slopes else rng.normal(size=J)
        if sd:
            args.update(sd=sd, z=z)
        if sub.spec.correlated_terms:
            args["L"] = {t.factor: np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 1.0]])) for t in sub.spec.correlated_terms}
        resp[r] = args
    if not model.rescor:
        return model.value_point(resp=resp), resp
    K = len(model.responses)
    if K == 2:
        rho = 0.4 if rho is None else rho
        L = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho ** 2)]])
    else:
        L = np.linalg.cholesky(np.array([[1.0, 0.4, -0.2], [0.4, 1.0, 0.25], [-0.2, 0.25, 1.0]]))
    return model.value_point(resp=resp, Lrescor=L), resp | {"Lrescor": L}


# -- 1. the grammar --------------------------------------------------------------------------------------------------
def test_parse_multivariate(data):
    s = parse_multivariate("mvbind(y, y2) ~ x + z + (1 | g)", data, categorical={"z": "p"}, rescor=True)
    assert isinstance(s, MultivariateSpec) and s.responses == ("y", "y2") and s.rescor
    assert s.formulas == {"y": "y ~ x + z + (1 | g)", "y2": "y2 ~ x + z + (1 | g)"}
    assert [sp.response for sp in s.specs.values()] == ["y", "y2"] and s.specs["y2"].columns == ("x", "z[q]", "z[r]")
    assert s.canonical_formula == "mvbind(y, y2) ~ x + z + (1 | g)" and s.groups == ("g",)
    assert parse_multivariate(" mvbind( y , y2,y3 )~x", rescor=False).responses == ("y", "y2", "y3")


@pytest.mark.parametrize("formula, message", [
    ("mvbind(y) ~ x", "mvbind() needs at least two responses"),
    ("mvbind(y, y) ~ x", "the response(s) ['y'] appear more than once"),
    ("mvbind(y, log(y2)) ~ x", "expected 'mvbind(y1, y2, ...) ~ <terms>', the responses as column names inside mvbind()"),
    ("mvbind(y, 2y) ~ x", "mvbind() takes response column names, got ['2y']"),
    ("mvbind(y, y2) | weights(w) ~ x", "addition terms (weights(), cens(), trunc(), ...) on the responses of a multivariate model are not supported"),
    ("mvbind(y, y2) ~ x + mi(w)", "brms's mi() (missing values modelled by a response's own formula) is not supported (docs/multivariate.md"),
    ("mvbind(y, y2) ~ x + mo(w)", "a monotonic term in a multivariate formula is not supported"),
    ("mvbind(y, y2) ~ x + (1 |p| g)", "a correlation ID, brms's (1 |ID| g), which correlates the responses' group-level effects, is not supported"),
    ("mvbind(y, y2) ~ x + y2", "the response(s) ['y2'] cannot also be predictors, grouping factors or offsets"),
    ("mvbind(y, y2) ~ x + y", "the formula of the response 'y' ('y ~ x + y'): 'y': the response cannot also be a predictor"),
    ("mvbind(y, y2) ~ x + (1 | y2)", "cannot also be predictors, grouping factors or offsets"),
])
def test_grammar_refusals(data, formula, message):
    with pytest.raises(ab.FormulaError, match=re.escape(message)):
        ab.gaussian(formula, data, priors(), rescor=True)


def test_refused_forms_and_families(data):
    with pytest.raises(ab.FormulaError, match=re.escape("a formula of its own per response (brms's bf(y ~ ...) + bf(y2 ~ ...)) is not "
                                                        "supported") + ".*divergence D56"):
        ab.gaussian(["y ~ x", "y2 ~ w"], data, priors(), rescor=True)
    with pytest.raises(ab.FormulaError, match=re.escape("a distributional formula in a multivariate model")):
        ab.gaussian(["mvbind(y, y2) ~ x", "sigma ~ w"], data, priors(), rescor=True)
    with pytest.raises(ab.FormulaError, match=re.escape("if 'y2' is a second response, a formula of its own per response")):
        ab.gaussian(["y ~ x", "y2 ~ w"], data, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)))
    with pytest.raises(ValueError, match=re.escape("a multivariate model needs rescor=True") + ".*divergence D58"):
        ab.gaussian(F, data, priors())
    with pytest.raises(ValueError, match=re.escape("rescor must be True or False, got 'yes'")):
        ab.gaussian(F, data, priors(), rescor="yes")
    with pytest.raises(ab.FormulaError, match=re.escape("rescor= applies to a multivariate formula")):
        ab.gaussian("y ~ x", data, priors(), rescor=True)
    for build in (lambda: ab.student(F, data, priors(), nu=4), lambda: ab.poisson(F, data, priors())):
        with pytest.raises(ab.FormulaError, match=re.escape("ab.gaussian(..., rescor=...) for the gaussian family only") + ".*D58"):
            build()
    with pytest.raises(ab.FormulaError, match=re.escape("is built by ab.gaussian(formula, data, ab.Priors(resp={...}), rescor=True | False)")):
        ab.Model(F, data, priors())
    with pytest.raises(ab.FormulaError, match=re.escape("a multivariate student model is not (docs/multivariate.md, divergence D58)")):
        parse_multivariate(F, family="student", rescor=True)


# -- 2. the priors by response ---------------------------------------------------------------------------------------
def test_priors_by_response(data):
    for pr, message in [
        (ab.Priors(resp={"y": response(), "y2": response()}, sigma=ab.HalfNormal(1), rescor=ab.LKJ(2)),
         "a multivariate model takes its priors by response, as brms's resp: Priors(resp={'y': {...}, 'y2': {...}}, rescor=ab.LKJ(eta)); got ['sigma']"),
        (ab.Priors(rescor=ab.LKJ(2)), "a multivariate model needs the priors of each response: pass resp={'y': {...}, 'y2': {...}}"),
        (ab.Priors(resp={"y": response()}, rescor=ab.LKJ(2)), "resp must cover exactly the responses ['y', 'y2']; got ['y']"),
        (ab.Priors(resp={"y": response(), "y2": response(), "y3": response()}, rescor=ab.LKJ(2)), "resp must cover exactly the responses"),
        (ab.Priors(resp={"y": response(), "y2": response()}), "rescor=True gives the responses a residual correlation matrix, which needs an LKJ prior"),
        (ab.Priors(resp={"y": response(), "y2": response()}, rescor=ab.Normal(0, 1)), "prior for 'rescor' must be a correlation prior"),
        (ab.Priors(resp={"y": response(), "y2": {**response(), "Intercept": ab.Normal(0, 1)}}, rescor=ab.LKJ(2)),
         "the priors of the response 'y2' take the classes ['intercept', 'b', 'sd', 'sigma'"),
        (ab.Priors(resp={"y": response(), "y2": {**response(), "cor": {"g": ab.LKJ(2)}}}, rescor=ab.LKJ(2)),
         "brms's correlation prior, class cor, has no resp: pass it once at the top level"),
        (ab.Priors(resp={"y": response(), "y2": ab.Priors(**response(), cor=ab.LKJ(2))}, rescor=ab.LKJ(2)),
         "the priors of the response 'y2' set cor: brms's correlation prior (class cor) has no resp"),
        (ab.Priors(resp={"y": response(), "y2": {k: v for k, v in response().items() if k != "sigma"}}, rescor=ab.LKJ(2)),
         "response 'y2': the Gaussian family needs a prior on the residual SD"),
        (ab.Priors(resp={"y": response(), "y2": response()}, rescor=ab.LKJ(2), cor={"g": ab.LKJ(2)}),
         "response 'y': the formula has no correlated group term"),
    ]:
        with pytest.raises(ab.PriorError, match=re.escape(message)):
            ab.gaussian(F, data, pr, rescor=True)
    with pytest.raises(ab.PriorError, match=re.escape("this model has no residual correlation (rescor=False): do not pass a rescor prior")):
        ab.gaussian(F, data, priors(), rescor=False)
    with pytest.raises(ab.PriorError, match=re.escape("resp and rescor are the priors of a multivariate model")):
        ab.gaussian("y ~ x + (1 | g)", data, ab.Priors(**response(), rescor=ab.LKJ(2)))


def test_a_response_entry_may_be_a_priors_or_a_mapping(data, rescor):
    as_priors = ab.Priors(resp={r: ab.Priors(**response(1.0 + k)) for k, r in enumerate(("y", "y2"))}, rescor=ab.LKJ(2))
    same = ab.gaussian(F, data, as_priors, rescor=True)
    assert same.target_identity == rescor.target_identity and same.describe() == rescor.describe()
    assert rescor.priors.response_priors("y2").sigma == ab.HalfNormal(2.0)


# -- 3. the assembled graph ------------------------------------------------------------------------------------------
def test_graph_names_and_coordinates(rescor, norescor):
    per = ["sigma_log__", "Intercept", "x", "1|g_sigma_log__", "1|g_offset"]
    assert [v.name for v in rescor.pymc_model.value_vars] == [f"{r}_{n}" for r in ("y", "y2") for n in per] + ["Lrescor_cholesky_corr__"]
    assert [v.name for v in norescor.pymc_model.value_vars] == [f"{r}_{n}" for r in ("y", "y2") for n in per]
    assert rescor.observed_names == ("mvbind(y, y2)",) and [v.name for v in rescor.pymc_model.observed_RVs] == ["mvbind(y, y2)"]
    assert norescor.observed_names == ("y", "y2") and sorted(v.name for v in norescor.pymc_model.observed_RVs) == ["y", "y2"]
    pmm = rescor.pymc_model
    assert [d.name for d in pmm.deterministics] == ["y_1|g", "y_mu", "y2_1|g", "y2_mu", "rescor"]
    assert pmm.named_vars_to_dims["Lrescor"] == pmm.named_vars_to_dims["rescor"] == (RESPONSE_DIM, RESPONSE_DIM_T)
    assert pmm.named_vars_to_dims["mvbind(y, y2)"] == ("__obs__", RESPONSE_DIM)
    assert tuple(pmm.named_vars_to_dims["y_1|g"]) == tuple(pmm.named_vars_to_dims["y2_1|g"]) == ("g__factor_dim",)
    assert tuple(pmm.coords[RESPONSE_DIM]) == ("y", "y2") and tuple(pmm.coords["g__factor_dim"]) == LEVELS
    assert rescor.response_names["y2"] == {"intercept": "y2_Intercept", "sd": "y2_1|g_sigma", "offset": "y2_1|g_offset",
                                           "group_effect": "y2_1|g", "linpred": "y2_mu", "b:x": "y2_x", "sigma": "y2_sigma",
                                           "mu": "y2_mu", "y": "mvbind(y, y2)"}
    assert norescor.response_names["y2"]["y"] == "y2" and "Lrescor" not in norescor.names
    assert rescor.names["mean"] == "mu" and rescor.names["Lrescor"] == "Lrescor" and rescor.names["y:sigma"] == "y_sigma"
    assert rescor.variable("y2", "1|g_sigma") == "y2_1|g_sigma"
    assert rescor.design.Y.shape == (28, 2) and np.array_equal(rescor.design.Y[:, 1], rescor.submodels["y2"].design.y)
    assert rescor.spec.formulas == {"y": "y ~ x + (1 | g)", "y2": "y2 ~ x + (1 | g)"} and rescor.rescor and not norescor.rescor


def test_collisions_are_refused(data):
    df = data.assign(y_x=data["y"] + 1.0)                     # response y's coefficient of x and the response y_x: 'y_x' twice
    with pytest.raises(ab.FormulaError, match=re.escape("generated variable/coordinate name collision in the multivariate graph: ['y_x']")):
        no_sd = {k: v for k, v in response().items() if k != "sd"}
        ab.gaussian("mvbind(y, y_x) ~ x", df, ab.Priors(resp={"y": no_sd, "y_x": no_sd}), rescor=False)


def _sub_point(model, vp, r):
    sub = model.submodels[r]
    return {v.name: vp[model.variable(r, v.name)] for v in sub.pymc_model.value_vars}


def _layout(pmm, point):
    """Where each value variable sits in the flattened gradient (value_vars order)."""
    out, i = {}, 0
    for v in pmm.value_vars:
        size = int(np.size(point[v.name]))
        out[v.name], i = slice(i, i + size), i + size
    return out


def test_without_rescor_the_density_is_the_responses_densities(norescor):
    vp, _ = point(norescor, seed=1)
    for jacobian in (True, False):
        joint = float(norescor.pymc_model.compile_logp(jacobian=jacobian)(vp))
        parts = sum(float(sub.pymc_model.compile_logp(jacobian=jacobian)(_sub_point(norescor, vp, r))) for r, sub in norescor.submodels.items())
        assert abs(joint - parts) <= 1e-10 * max(1.0, abs(joint))
    grad, lay = norescor.pymc_model.compile_dlogp()(vp), _layout(norescor.pymc_model, vp)
    for r, sub in norescor.submodels.items():                       # each coordinate's gradient is its response's own
        sp = _sub_point(norescor, vp, r)
        sub_grad = sub.pymc_model.compile_dlogp()(sp)
        for name, where in _layout(sub.pymc_model, sp).items():
            assert np.allclose(grad[lay[norescor.variable(r, name)]], sub_grad[where], rtol=1e-12, atol=1e-12)


def _likelihood_free_logp(model, vp, r, jacobian=False):
    """Response r's own model's density without its likelihood."""
    sub = model.submodels[r]
    return float(sub.pymc_model.compile_logp(vars=list(sub.pymc_model.free_RVs), jacobian=jacobian)(_sub_point(model, vp, r)))


def _mu(model, vp, r):
    sub = model.submodels[r]
    pmm = sub.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["mu"]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    return np.asarray(fn(_sub_point(model, vp, r))[0], dtype=float)


@pytest.mark.parametrize("which", ["rescor", "three"])
def test_with_rescor_the_density_is_the_multivariate_normal(which, request):
    """Without the Jacobian: each response's priors, the multivariate normal of every row (scipy) and the LKJ density of
    Lrescor; with it, plus Stan's cholesky_corr Jacobian. K = 2 checks the LKJ term against its closed form, (eta - 1)
    log(1 - rho^2) - log(2^(2 eta - 1) B(eta, eta))."""
    model = request.getfixturevalue(which)
    vp, args = point(model, seed=2)
    K, L = len(model.responses), args["Lrescor"]
    sigma = np.array([args[r]["sigma"] for r in model.responses])
    Sigma = np.diag(sigma) @ L @ L.T @ np.diag(sigma)
    mus = np.column_stack([_mu(model, vp, r) for r in model.responses])
    lik = sum(multivariate_normal(mean=m, cov=Sigma).logpdf(y) for m, y in zip(mus, model.design.Y))
    eta = float(model.priors.rescor.params["eta"])
    if K == 2:
        rho = L[1, 0]
        lkj = (eta - 1) * np.log(1 - rho ** 2) - ((2 * eta - 1) * np.log(2) + betaln(eta, eta))
    else:
        from autobayes.correlation import lkj_log_normalizer          # (checked against closed forms in test_correlated_slopes)
        lkj = sum((K - d + 2 * eta - 2) * np.log(L[d - 1, d - 1]) for d in range(2, K + 1)) - lkj_log_normalizer(eta, K)
    expected = sum(_likelihood_free_logp(model, vp, r) for r in model.responses) + lik + lkj
    got = float(model.pymc_model.compile_logp(jacobian=False)(vp))
    assert abs(got - expected) <= 1e-9 * max(1.0, abs(expected))
    jacobian = float(model.pymc_model.compile_logp(jacobian=True)(vp)) - got           # the responses' and Lrescor's (Stan's)
    expected = sum(_likelihood_free_logp(model, vp, r, True) - _likelihood_free_logp(model, vp, r) for r in model.responses)
    expected += np.log(1 - L[1, 0] ** 2) if K == 2 else cholesky_corr_constrain(cholesky_corr_free(L), K)[1]
    assert abs(jacobian - expected) <= 1e-9 * max(1.0, abs(expected))


def test_multi_normal_cholesky_logp_against_scipy():
    import pytensor
    import pytensor.tensor as pt

    rng = np.random.default_rng(7)
    L = np.linalg.cholesky(np.array([[1.0, 0.5, -0.3], [0.5, 1.0, 0.2], [-0.3, 0.2, 1.0]]))
    sigma, mu, y = np.array([0.7, 1.3, 2.0]), rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
    v, m, s, l = pt.matrix("v"), pt.matrix("m"), pt.vector("s"), pt.matrix("l")
    got = pytensor.function([v, m, s, l], multi_normal_cholesky_logp(pt.specify_shape(v, (None, 3)), m, s, l))(y, mu, sigma, L)
    Sigma = np.diag(sigma) @ L @ L.T @ np.diag(sigma)
    expected = [multivariate_normal(mean=mm, cov=Sigma).logpdf(yy) for mm, yy in zip(mu, y)]
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_value_point_refusals(rescor, norescor):
    _, args = point(rescor)
    resp = {r: args[r] for r in rescor.responses}
    vp = rescor.value_point(resp=resp, Lrescor=args["Lrescor"])
    assert np.allclose(vp["Lrescor_cholesky_corr__"], cholesky_corr_free(args["Lrescor"]))
    for kwargs, message in [({"resp": {"y": resp["y"]}, "Lrescor": args["Lrescor"]}, "resp must map exactly the responses ['y', 'y2']"),
                            ({"resp": resp}, "Lrescor (the Cholesky factor of the residual correlation matrix) is required"),
                            ({"resp": resp, "Lrescor": np.eye(3)}, "Lrescor must be a finite 2 x 2 matrix"),
                            ({"resp": resp, "Lrescor": np.array([[1.0, 0.0], [0.6, 0.6]])}, "Lrescor must have unit-norm rows"),
                            ({"resp": {**resp, "y2": 3}, "Lrescor": args["Lrescor"]}, "resp['y2'] must be a mapping")]:
        with pytest.raises(ValueError, match=re.escape(message)):
            rescor.value_point(**kwargs)
    with pytest.raises(ValueError, match=re.escape("this model has no residual correlation (rescor=False); Lrescor does not apply")):
        norescor.value_point(resp=resp, Lrescor=args["Lrescor"])


# -- 4. predictions --------------------------------------------------------------------------------------------------
PARS = {"y": (0.5, -0.3, 0.7, 0.4), "y2": (1.5, 0.6, 1.1, 0.8)}      # raw-scale intercept, slope of x, sigma, sd of g


def posterior(model, draws=4, rho=0.6, seed=0):
    """A posterior as ``fit()`` returns it (raw-scale intercepts, no offsets, no linear predictors) with the same parameters
    in every draw and every response's group effects drawn once."""
    rng = np.random.default_rng(seed)
    shape, data = (1, draws), {}
    for r, (a, b, s, sd) in PARS.items():
        u = rng.normal(0, sd, len(LEVELS))
        data.update({f"{r}_Intercept": (("chain", "draw"), np.full(shape, a)), f"{r}_x": (("chain", "draw"), np.full(shape, b)),
                     f"{r}_sigma": (("chain", "draw"), np.full(shape, s)), f"{r}_1|g_sigma": (("chain", "draw"), np.full(shape, sd)),
                     f"{r}_1|g": (("chain", "draw", "g__factor_dim"), np.broadcast_to(u, shape + (len(LEVELS),)).copy())})
    coords = {"chain": [0], "draw": np.arange(draws), "g__factor_dim": list(LEVELS)}
    if model.rescor:
        L = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho ** 2)]])
        dims = ("chain", "draw", RESPONSE_DIM, RESPONSE_DIM_T)
        data.update({"Lrescor": (dims, np.broadcast_to(L, shape + (2, 2)).copy()), "rescor": (dims, np.broadcast_to(L @ L.T, shape + (2, 2)).copy())})
        coords.update({RESPONSE_DIM: ["y", "y2"], RESPONSE_DIM_T: ["y", "y2"]})
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


def _eta(idata, df, r):
    post = idata.posterior
    u = dict(zip(LEVELS, post[f"{r}_1|g"].to_numpy()[0, 0]))
    return PARS[r][0] + PARS[r][1] * df["x"].to_numpy() + np.array([u[g] for g in df["g"]])


def test_linear_and_mean_predictions_carry_a_response_axis(rescor, data):
    idata = posterior(rescor)
    newdata = data.iloc[[3, 0, 9]].reset_index(drop=True)
    for nd, df in ((None, data), (newdata, newdata)):
        lin = rescor.predict(idata, newdata=nd, kind="linear")
        assert lin["eta"].dims == ("chain", "draw", "__obs__", "response") and list(lin["response"].values) == ["y", "y2"]
        for k, r in enumerate(("y", "y2")):
            assert np.allclose(lin["eta"].to_numpy()[0, :, :, k], _eta(idata, df, r)[None, :], rtol=1e-12)
        mean = rescor.predict(idata, newdata=nd, kind="mean")
        assert np.array_equal(mean["mu"].to_numpy(), lin["eta"].to_numpy()) and lin.attrs["link"] == "identity"
    nogroup = rescor.predict(idata, kind="linear", include_group_effects=False)["eta"].to_numpy()[0, 0, :, 1]
    assert np.allclose(nogroup, PARS["y2"][0] + PARS["y2"][1] * data["x"].to_numpy())
    assert np.array_equal(ab.predict(rescor, idata, kind="linear")["eta"].to_numpy(), rescor.predict(idata, kind="linear")["eta"].to_numpy())
    with pytest.raises(ValueError, match=re.escape("dpar does not apply to a multivariate model")):
        ab.predict(rescor, idata, kind="linear", dpar="sigma")


@pytest.mark.parametrize("which, rho", [("rescor", 0.6), ("norescor", 0.0)])
def test_response_draws_are_joint(which, rho, request, data):
    model = request.getfixturevalue(which)
    idata = posterior(model, draws=4000, rho=0.6)
    out = model.predict(idata, kind="response", seed=11)
    y = out["y"].to_numpy()[0]                                     # (draw, row, response)
    resid = y - out["mu"].to_numpy()[0]
    assert np.isclose(np.corrcoef(resid[:, :, 0].ravel(), resid[:, :, 1].ravel())[0, 1], rho, atol=0.02)
    assert np.allclose(resid.std(axis=(0, 1)), [PARS["y"][2], PARS["y2"][2]], rtol=0.02)
    again = model.predict(idata, kind="response", seed=11)["y"].to_numpy()
    assert np.array_equal(again, out["y"].to_numpy())
    assert not np.array_equal(model.predict(idata, kind="response", seed=12)["y"].to_numpy(), out["y"].to_numpy())


def test_unseen_levels(rescor, data):
    idata = posterior(rescor, draws=200)
    nd = pd.DataFrame({"x": [0.1, 0.2, -1.0], "g": ["a", "zz", "zz"]})
    with pytest.raises(ab.FormulaError, match=re.escape("newdata has levels of 'g' not seen in training: ['zz']")):
        rescor.predict(idata, newdata=nd, kind="linear")
    pop = rescor.predict(idata, newdata=nd, kind="linear", new_groups="population", seed=3)
    eta = pop["eta"].to_numpy()[0]
    effect = eta[:, 1, :] - np.column_stack([PARS[r][0] + PARS[r][1] * 0.2 for r in ("y", "y2")]).ravel()
    assert not np.allclose(effect[:, 0] / PARS["y"][3], effect[:, 1] / PARS["y2"][3])   # each response draws its own effects
    assert np.allclose(eta[:, 1, :] - eta[:, 2, :], np.array([PARS[r][1] for r in ("y", "y2")]) * 1.2)   # one effect per label
    assert pop.attrs["new_groups"] == "population" and pop.attrs["new_levels"] == ["zz"]
    assert np.array_equal(rescor.predict(idata, newdata=nd, kind="linear", new_groups="population", seed=3)["eta"].to_numpy(), pop["eta"].to_numpy())
    rs = rescor.predict(idata, newdata=nd, kind="mean", new_groups="resample_observed", seed=4)["mu"].to_numpy()[0]
    observed = {r: idata.posterior[f"{r}_1|g"].to_numpy()[0, 0] for r in ("y", "y2")}
    for k, r in enumerate(("y", "y2")):
        assert np.all(np.isin(np.round(rs[:, 1, k] - PARS[r][0] - PARS[r][1] * 0.2, 12), np.round(observed[r], 12)))


# -- 5. fitting, without sampling ------------------------------------------------------------------------------------
@pytest.mark.nosampling                                    # pm.sample is replaced: nothing samples
def test_fit_samples_the_assembled_graph_and_cleans_it(rescor, data, monkeypatch):
    import pymc as pm

    calls = {}
    rng = np.random.default_rng(0)
    centred = {r: (rng.normal(size=(2, 5)), rng.normal(size=(2, 5))) for r in rescor.responses}      # (Intercept_c, b_x)

    def fake_sample(**kwargs):
        calls.update(kwargs)
        data_vars = {}
        for r, (a, b) in centred.items():
            data_vars.update({f"{r}_Intercept": (("chain", "draw"), a), f"{r}_x": (("chain", "draw"), b),
                              f"{r}_1|g_offset": (("chain", "draw", "g__factor_dim"), np.zeros((2, 5, 4)))})
        return az.InferenceData(posterior=xr.Dataset(data_vars, coords={"chain": [0, 1], "draw": np.arange(5), "g__factor_dim": list(LEVELS)}))

    monkeypatch.setattr(pm, "sample", fake_sample)
    fit = ab.fit(rescor, backend="pymc", chains=2, draws=5, tune=5, seed=1)
    assert calls["nuts_sampler"] == "pymc" and calls["draws"] == 5 and calls["chains"] == 2 and calls["random_seed"] == 1
    assert "y_mu" not in calls["var_names"] and "y2_mu" not in calls["var_names"]
    assert {"Lrescor", "rescor", "y_1|g", "y2_1|g", "y_Intercept", "y2_sigma"} <= set(calls["var_names"])
    post = fit.idata.posterior
    assert not [v for v in post.data_vars if v.endswith("_offset")]
    for r, (a, b) in centred.items():                                    # brms's b_y_Intercept = Intercept_y - mean(x) b_y
        assert np.allclose(post[f"{r}_Intercept"].to_numpy(), a - data["x"].mean() * b, rtol=1e-12)
    assert fit.idata.attrs[FIT_IDENTITY_ATTR] and fit.target_identity == rescor.target_identity
    assert rescor.predict(fit, kind="linear", include_group_effects=False).attrs["fit_binding_status"] == "matched"
    import jax

    monkeypatch.setattr("autobayes.inference.resolve_jax_device", lambda *args: (jax.devices("cpu")[0], "sequential"))
    ab.fit(rescor, backend="numpyro", chains=2, draws=5, tune=5)
    assert calls["nuts_sampler"] == "numpyro" and "chain_method" in calls["nuts_sampler_kwargs"]


# -- 6. default priors, the prior table, describe() and the identity --------------------------------------------------------
def test_default_priors(data):
    from autobayes.default_priors import location_scale

    pr = ab.default_priors("mvbind(y, y2) ~ x + (1 + x | g)", data, rescor=True)
    assert set(pr.resp) == {"y", "y2"} and pr.rescor == ab.LKJ(1) and pr.rescor.source == "default" and pr.cor == {"g": ab.LKJ(1)}
    for r in ("y", "y2"):
        loc, scale = location_scale(data[r].to_numpy(), "identity")
        assert pr.resp[r].intercept == ab.StudentT(3, loc, scale) and pr.resp[r].cor is None
    model = ab.gaussian("mvbind(y, y2) ~ x + (1 + x | g)", data, pr, rescor=True)
    rows = model.prior_table()
    assert rows[0] == {"prior": "lkj(1)", "class": "cor", "coef": "", "group": "", "resp": "", "dpar": "", "nlpar": "", "lb": "",
                       "ub": "", "tag": "", "source": "default"}
    assert [row["class"] for row in rows if row["resp"] == ""] == ["cor", "cor", "rescor"]
    assert [row["resp"] for row in rows if row["class"] == "sigma"] == ["y", "y2"]
    assert ab.default_priors(F, data, rescor=False).rescor is None
    with pytest.raises(ValueError, match=re.escape("a multivariate model needs rescor=True")):
        ab.default_priors(F, data)


def test_describe(rescor, norescor):
    text = rescor.describe()
    for line in ["AutoBayes multivariate Gaussian model of 2 responses (y, y2), with residual correlation (brms set_rescor(TRUE))",
                 "  (y_i, y2_i) ~ MultiNormal((mu_y,i, mu_y2,i), Sigma), Sigma = diag(sigma) Rescor diag(sigma), sigma = (sigma_y, sigma_y2):",
                 "  Lrescor ~ LKJ(eta=2) (brms lkj(2) on Rescor, placed on Lrescor as lkj_corr_cholesky), sampled on Stan's cholesky_corr coordinates",
                 "Response y2: its univariate model; in the multivariate graph its PyMC variables are named y2_<name> and its likelihood is the joint one above",
                 "    formula   : y2 ~ x + (1 | g)",
                 "  y2_1|g_sigma_log__           y2's 1|g_sigma_log__ (above)",
                 "  Lrescor_cholesky_corr__      Lrescor on Stan's cholesky_corr coordinates (canonical partial correlations, arctanh scale)"]:
        assert line in text.splitlines(), line
    assert "rescor = Lrescor Lrescor^T (brms Rescor)" in text
    other = norescor.describe()
    assert "each response its own Normal likelihood, independent given the parameters (no residual correlation): " \
           "y_i ~ Normal(mu_y,i, sigma_y); y2_i ~ Normal(mu_y2,i, sigma_y2)" in other and "Lrescor" not in other


def test_target_identity(rescor, norescor, data):
    assert rescor.target_identity != norescor.target_identity
    assert ab.gaussian(F, data, priors(), rescor=True).target_identity == rescor.target_identity
    assert ab.gaussian(F, data, priors(eta=3.0), rescor=True).target_identity != rescor.target_identity
    swapped = ab.gaussian("mvbind(y2, y) ~ x + (1 | g)", data, priors(), rescor=True)
    assert swapped.target_identity != rescor.target_identity and swapped.responses == ("y2", "y")


def test_what_the_slice_does_not_offer(rescor):
    with pytest.raises(ValueError, match=re.escape("prior_predictive() of a multivariate model is not supported")):
        ab.prior_predictive(rescor, draws=10)
    with pytest.raises(ValueError, match=re.escape("fit_record() of a multivariate model is not supported")):
        ab.fit_record(rescor, None)
