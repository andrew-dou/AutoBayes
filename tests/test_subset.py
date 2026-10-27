"""brms's ``subset()`` in a multivariate model with a formula per response (brms-parity feature F06c; docs/subset.md).

The grammar (``subset(col)`` on a response of a list of formulas, ``MultivariateSpec.subsets``, the canonical ``bf() + bf()``)
and its refusals, the indicator's rows, the model each response gets (its own formula on its subset's rows: its design, its
priors' defaults, its levels), the assembled graph (one observation coordinate per response, on its rows) and its density
against the responses' own models and scipy's Normal written out, ``predict(resp=)`` by hand on the fitted rows and on new
rows (by their own indicator), the refusal of joint predictions, ``describe()``, the target identity, the fit path's cleaning
with ``pm.sample`` replaced (nothing samples), and a formula per response without ``subset()``. The numbers against brms are
the oracle's (tests/oracle, case gauss_mv_subset)."""
import re

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import norm

import autobayes as ab
from autobayes.formula import FormulaError, parse_formula
from autobayes.multivariate import SUBSET_REFUSED, is_multivariate_formula, parse_multivariate, subset_rows

LEVELS = ("a", "b", "c", "d")
F = ["y | subset(s1) ~ x + (1 | g)", "y2 | subset(s2) ~ x"]


def frame(n=36, seed=11) -> pd.DataFrame:
    """Each response observed on its subset's rows only (NaN elsewhere, as brms keeps such rows): ``s1`` an integer 0/1
    indicator that reaches every level of ``g``, ``s2`` a logical one; ``y2f`` and ``wf`` complete columns."""
    rng = np.random.default_rng(seed)
    g = np.array([LEVELS[i % 4] for i in range(n)])
    x, w = rng.normal(size=n), rng.normal(0.5, 1.0, n)
    u = dict(zip(LEVELS, rng.normal(0, 0.7, 4)))
    s1 = np.array([1, 1, 0] * (n // 3))
    s2 = rng.uniform(size=n) < 0.55
    y = 0.4 + 0.8 * x + np.array([u[v] for v in g]) + rng.normal(0, 0.5, n)
    y2 = -0.2 + 0.3 * x + rng.normal(0, 0.8, n)
    return pd.DataFrame({"x": x, "w": w, "g": g, "h": rng.choice(["u", "v"], n), "k": rng.integers(0, 3, n), "t": np.arange(n),
                         "y": np.where(s1 == 1, y, np.nan), "y2": np.where(s2, y2, np.nan), "y2f": y2, "wf": w + y,
                         "s1": s1, "s2": s2})


def response(sd=False, sigma=1.0):
    out = {"intercept": ab.Normal(0, 5), "b": ab.Normal(0, 2), "sigma": ab.HalfNormal(sigma)}
    return {**out, "sd": ab.HalfNormal(1)} if sd else out


def priors(responses=("y", "y2"), groups=("y",)):
    return ab.Priors(resp={r: response(sd=r in groups, sigma=1.0 + k) for k, r in enumerate(responses)})


UNI = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1))


@pytest.fixture(scope="module")
def data():
    return frame()


@pytest.fixture(scope="module")
def model(data):
    return ab.gaussian(F, data, priors(), rescor=False)


def rows(data, column):
    return np.flatnonzero(data[column].to_numpy() == 1)


# -- 1. the grammar --------------------------------------------------------------------------------------------------
def test_parse(data):
    s = parse_multivariate(F, data, rescor=False)
    assert s.responses == ("y", "y2") and s.subsets == {"y": "s1", "y2": "s2"} and s.separate and not s.rescor
    assert s.formulas == {"y": "y ~ x + (1 | g)", "y2": "y2 ~ x"} and s.groups == ("g",)
    assert s.canonical_formula == "bf(y | subset(s1) ~ x + (1 | g)) + bf(y2 | subset(s2) ~ x)"
    mixed = parse_multivariate(["y | subset( s1 ) ~ x", "y2f ~ x + w"], data, rescor=False)
    assert mixed.subsets == {"y": "s1"} and mixed.response_formula("y2f") == "y2f ~ x + w"
    assert mixed.canonical_formula == "bf(y | subset(s1) ~ x) + bf(y2f ~ x + w)"
    assert parse_multivariate(F, rescor=False).subsets == {"y": "s1", "y2": "s2"}          # the structure, without the data
    assert is_multivariate_formula(F) and is_multivariate_formula(["y | subset(s1) ~ x", "y2 ~ x"])
    assert not is_multivariate_formula(["y ~ x", "sigma ~ w"]) and not is_multivariate_formula("y ~ x")


def test_subset_rows(data):
    assert subset_rows(data, "s1").tolist() == rows(data, "s1").tolist()
    assert subset_rows(data, "s2").tolist() == np.flatnonzero(data["s2"]).tolist()      # logical
    assert subset_rows(data.assign(s1=data["s1"].astype(float)), "s1").tolist() == rows(data, "s1").tolist()
    for values, message in [([2] + [1] * 35, "must be logical (True/False) or numeric 0/1 without missing values, got the values ['1', '2']"),
                            ([np.nan] + [1.0] * 35, "without missing values"),
                            (["yes", "no"] * 18, "got the values [\"'no'\", \"'yes'\"]"),
                            ([0] * 36, "no row of the data is in the subset (brms: \"All rows of 'data' were removed via 'subset'\")")]:
        with pytest.raises(FormulaError, match=re.escape(message)):
            subset_rows(data.assign(s1=values), "s1", "y")
    with pytest.raises(FormulaError, match=re.escape("subset(nope) of the response 'y': data lacks the indicator column 'nope'")):
        subset_rows(data, "nope", "y")
    with pytest.raises(FormulaError, match=re.escape("subset(s1): no row of the newdata is in the subset") + "$"):
        subset_rows(data.assign(s1=0), "s1", where="newdata")


@pytest.mark.parametrize("formulas, rescor, error, message", [
    (F, True, FormulaError, "subset() with a residual correlation (rescor=True, brms's set_rescor(TRUE)) is refused, as brms refuses "
                            "it (\"Only 'se', 'weights', 'mi' are supported addition arguments when 'rescor' is estimated.\")"),
    (["y2f ~ x", "wf ~ w"], True, FormulaError, "a formula of its own per response (brms's bf(y ~ ...) + bf(y2 ~ ...)) is not supported"
                                                " with a residual correlation (rescor=True)"),
    (F, None, ValueError, "a formula per response takes rescor=False (brms's set_rescor(FALSE); docs/subset.md)"),
    (F, "no", ValueError, "rescor must be True or False, got 'no'"),
    (F[:1], False, FormulaError, "one formula is a univariate model; a multivariate model is a list of formulas, one per response"),
    ([F[0], "mvbind(y2, wf) ~ x"], False, FormulaError, "mvbind() inside a list of formulas is not supported"),
    (["subset(s1) ~ x", F[1]], False, FormulaError, "subset() belongs to a response's left-hand side, '<response> | subset(<indicator>) ~ <terms>'"),
    (["y | subset(s1) + subset(s2) ~ x", F[1]], False, FormulaError, "subset() appears more than once on the left-hand side"),
    (["y | subset(s1) + weights(w) ~ x", F[1]], False, FormulaError, "subset() together with weights(w) on the left-hand side is not supported"),
    (["y | subset(s1 == 1) ~ x", F[1]], False, FormulaError, "subset() takes the name of a 0/1 or logical column of the data, got 's1 == 1'"),
    ([F[0], "y2 | subset(s2) ~ x + foo(x)"], False, FormulaError, "the formula 'y2 | subset(s2) ~ x + foo(x)': 'foo(x)': the function foo() is not supported"),
    ([F[0], "y | subset(s2) ~ w"], False, FormulaError, "the response(s) ['y'] have more than one formula"),
    ([F[0], "sigma ~ w"], False, FormulaError, "'sigma ~ w': a distributional formula in a multivariate model is not supported"),
    ([F[0], "y2 | weights(w) ~ x"], False, FormulaError, "addition terms (weights(w)) on a response of a multivariate model are not supported"),
    (["y | subset(s1) ~ x + (1 | mm(g, h))", F[1]], False, FormulaError, "'(1 | mm(g, h))' in a multivariate model is not supported"),
    (["y | subset(s1) ~ me(x, w)", F[1]], False, FormulaError, "a measurement-error term in a multivariate model is not supported"),
    (["y | subset(s1) ~ mo(k)", F[1]], False, FormulaError, "a monotonic term in a multivariate model is not supported"),
    (["y | subset(s1) ~ gp(x)", F[1]], False, FormulaError, "the Gaussian process 'gp(x)' in a multivariate model is not supported"),
    (["y | subset(s1) ~ x + ar(t, g)", F[1]], False, FormulaError, "the autocorrelation term 'ar(time = t, gr = g, p = 1)' in a multivariate model"),
    (["y | subset(s1) ~ x + sar(M)", F[1]], False, FormulaError, "the SAR term 'sar(M, type = \"lag\")' in a multivariate model"),
    (["y | subset(s1) ~ x + fcor(M)", F[1]], False, FormulaError, "fcor(M) in a multivariate model is not supported"),
    (["y | subset(s1) ~ x + car(W, gr = g)", F[1]], False, FormulaError, "the CAR term 'car(W, gr = g, type = \"escar\")' in a multivariate"),
    (["y | subset(s1) ~ x + cosy(t, g)", F[1]], False, FormulaError, "the residual correlation term 'cosy(time = t, gr = g)' in a multivariate"),
    (["y | subset(s1) ~ x + y2", F[1]], False, FormulaError, "the response(s) ['y2'] of another formula cannot also be predictors"),
    (["y | subset(y2) ~ x", "y2 ~ x"], False, FormulaError, "the subset() indicator(s) ['y2'] are responses of the model"),
    (["y | subset(s1) ~ mi(xm) + w", "xm | mi() ~ w"], False, FormulaError, "subset() in a model with mi() is refused, as brms refuses it "
                                                                           "(\"mi() terms in subsetted formulas require the 'idx'"),
    (["y ~ x", "xm | mi() ~ w"], False, FormulaError, "the missing values of ['xm'] are parameters ('| mi()') that no formula reads"),
])
def test_refusals(data, formulas, rescor, error, message):
    with pytest.raises(error, match=re.escape(message)):
        ab.gaussian(formulas, data.assign(xm=np.where(data.index % 5 == 0, np.nan, data["w"])), priors(), rescor=rescor)


def test_univariate_subset_and_a_second_response_are_named(data):
    with pytest.raises(FormulaError, match=re.escape("'y | subset(s1) ~ x': 'subset(s1)': brms's subset() is offered here in a multivariate "
                                                     "model with a formula per response") + ".*divergence D160"):
        ab.gaussian("y | subset(s1) ~ x", data, UNI)
    with pytest.raises(FormulaError, match=re.escape("'subset(s1)': brms's subset() is offered here")):
        parse_formula(["y | weights(w) + subset(s1) ~ x", "sigma ~ w"], data)
    with pytest.raises(FormulaError, match=re.escape("if 'y2' is a second response, a formula of its own per response (brms's bf(y ~ ...) "
                                                     "+ bf(y2 ~ ...) + set_rescor(FALSE)) is a multivariate model: pass rescor=False")):
        ab.gaussian(["y ~ x", "y2 ~ w"], data, UNI)


def test_data_refusals(data):
    with pytest.raises(FormulaError, match=re.escape("the formula of the response 'y' ('y | subset(s1) ~ x + (1 | g)', on the rows of "
                                                     "subset(s1)): ") + ".*missing values"):
        ab.default_priors(F, data.assign(x=np.where(data.index == 0, np.nan, data["x"])), rescor=False)   # row 0 is in s1
    with pytest.raises(FormulaError, match=re.escape("the formula of the response 'y2f' ('y2f ~ x'): ") + ".*missing values"):
        ab.default_priors(["y | subset(s1) ~ x", "y2f ~ x"], data.assign(x=np.where(data.index == 2, np.nan, data["x"])), rescor=False)
    lacking = data.assign(s1=np.where(data["g"] == "d", 0, data["s1"]))
    with pytest.raises(FormulaError, match=re.escape("the response 'y': the level(s) ['d'] of the grouping factor 'g' occur in the data but in "
                                                     "no row of subset(s1)") + ".*divergence D161"):
        ab.gaussian(F, lacking, priors(), rescor=False)
    outside = data.assign(g=np.where((data["s1"] == 0) & (data.index == 2), None, data["g"]))   # a missing label outside the subset
    assert ab.gaussian(F, outside, priors(), rescor=False).submodels["y"].design.group_designs["g"].levels == LEVELS


# -- 2. the model: each response on its rows -------------------------------------------------------------------------
def test_each_response_is_its_own_model_on_its_rows(model, data):
    r1, r2 = rows(data, "s1"), np.flatnonzero(data["s2"])
    assert model.design.n == len(data) and model.design.rows["y"].tolist() == r1.tolist() and model.design.rows["y2"].tolist() == r2.tolist()
    for r, rr in (("y", r1), ("y2", r2)):
        sub = model.submodels[r]
        assert sub.design.n == rr.size and np.array_equal(sub.design.y, data[r].to_numpy()[rr])
        assert np.isclose(sub.design.x_means[0], data["x"].to_numpy()[rr].mean(), rtol=1e-14)     # centred on the subset's rows
    Y = model.design.Y
    assert np.array_equal(np.isnan(Y[:, 0]), data["s1"].to_numpy() == 0) and np.array_equal(np.isnan(Y[:, 1]), ~data["s2"].to_numpy())
    assert model.spec.subsets == {"y": "s1", "y2": "s2"} and model.spec.canonical_formula == "bf(y | subset(s1) ~ x + (1 | g)) + bf(y2 | subset(s2) ~ x)"
    pmm = model.pymc_model
    assert model.observation_dims == {"y": "__obs_y__", "y2": "__obs_y2__"}
    assert list(pmm.coords["__obs_y__"]) == r1.tolist() and list(pmm.coords["__obs_y2__"]) == r2.tolist() and "__obs__" not in pmm.coords
    dims = {name: tuple(d) for name, d in pmm.named_vars_to_dims.items()}
    assert dims["y"] == ("__obs_y__",) and dims["y2_mu"] == ("__obs_y2__",) and dims["y2"] == ("__obs_y2__",)
    assert dims["y_1|g"] == ("g__factor_dim",) and list(pmm.coords["g__factor_dim"]) == list(LEVELS)
    assert [v.name for v in pmm.value_vars] == ["y_sigma_log__", "y_Intercept", "y_x", "y_1|g_sigma_log__", "y_1|g_offset",
                                                "y2_sigma_log__", "y2_Intercept", "y2_x"]
    assert model.observed_names == ("y", "y2") and model.response_names["y2"]["y"] == "y2"


def test_the_graph_check_reads_the_rows(model, data):
    from dataclasses import replace

    good = model.design
    try:
        model.design = replace(good, rows={"y": good.rows["y"][1:], "y2": good.rows["y2"]})
        with pytest.raises(RuntimeError, match=re.escape("the observed 'y' is not on its rows, the coordinate '__obs_y__'")):
            model._check_graph()
    finally:
        model.design = good
    model._check_graph()


def test_missing_values_outside_the_subsets_are_not_read(data):
    neither = int(np.flatnonzero((data["s1"] == 0) & ~data["s2"])[0])
    holes = data.assign(x=np.where(data.index == neither, np.nan, data["x"]), g=np.where(data.index == neither, None, data["g"]))
    m = ab.gaussian(F, holes, priors(), rescor=False)            # what no response reads is not part of the target
    assert m.design.n == len(data) and m.target_identity == ab.gaussian(F, data, priors(), rescor=False).target_identity


def point(model, seed=0):
    rng = np.random.default_rng(seed)
    resp = {}
    for r, sub in model.submodels.items():
        args = {"intercept": float(rng.normal()), "b": rng.normal(size=len(sub.spec.columns)), "sigma": float(rng.uniform(0.5, 1.5))}
        if sub.spec.group_terms:
            args.update(sd={"g": float(rng.uniform(0.3, 1.0))}, z={"g": rng.normal(size=len(LEVELS))})
        resp[r] = args
    return model.value_point(resp=resp), resp


def test_the_density_is_each_response_on_its_rows(model, data):
    vp, resp = point(model, seed=3)
    pmm = model.pymc_model
    for jacobian in (True, False):
        joint = float(pmm.compile_logp(jacobian=jacobian)(vp))
        parts = sum(float(sub.pymc_model.compile_logp(jacobian=jacobian)({v.name: vp[model.variable(r, v.name)] for v in sub.pymc_model.value_vars}))
                    for r, sub in model.submodels.items())
        assert abs(joint - parts) <= 1e-12 * abs(joint)
    likelihood = float(pmm.compile_logp(jacobian=False)(vp)) - float(pmm.compile_logp(vars=list(pmm.free_RVs), jacobian=False)(vp))
    expected = 0.0
    for r, column in (("y", "s1"), ("y2", "s2")):                    # scipy's Normal on the subset's rows, the mean written out
        rr = np.flatnonzero(data[column].to_numpy() == 1)
        x = data["x"].to_numpy()[rr]
        mu = resp[r]["intercept"] + (x - x.mean()) * resp[r]["b"][0]
        if r == "y":
            u = resp[r]["sd"]["g"] * resp[r]["z"]["g"]
            mu = mu + u[[LEVELS.index(v) for v in data["g"].to_numpy()[rr]]]
        expected += norm.logpdf(data[r].to_numpy()[rr], mu, resp[r]["sigma"]).sum()
    assert abs(likelihood - expected) <= 1e-12 * abs(expected)


# -- 3. predictions, one response at a time ----------------------------------------------------------------------------
PARS = {"y": (0.3, 0.7, 0.6), "y2": (-0.4, 0.2, 0.9)}       # raw-scale intercept, slope, sigma


def posterior(draws=4, seed=0, pars=None, groups=("y",), slope="x"):
    """A posterior as ``fit()`` returns it: each response's raw-scale intercept, slope (named ``<response>_<slope>``) and sigma
    in every draw (``pars``, ``PARS`` by default), and the group effects of ``groups``, drawn once."""
    rng = np.random.default_rng(seed)
    shape, out = (1, draws), {}
    for r, (a, b, s) in (PARS if pars is None else pars).items():
        out.update({f"{r}_Intercept": (("chain", "draw"), np.full(shape, a)), f"{r}_{slope}": (("chain", "draw"), np.full(shape, b)),
                    f"{r}_sigma": (("chain", "draw"), np.full(shape, s))})
        if r in groups:
            out.update({f"{r}_1|g_sigma": (("chain", "draw"), np.full(shape, 0.8)),
                        f"{r}_1|g": (("chain", "draw", "g__factor_dim"), np.broadcast_to(rng.normal(0, 0.8, 4), shape + (4,)).copy())})
    return az.InferenceData(posterior=xr.Dataset(out, coords={"chain": [0], "draw": np.arange(draws), "g__factor_dim": list(LEVELS)}))


def _eta(idata, df, r, groups=True):
    a, b, _ = PARS[r]
    eta = a + b * df["x"].to_numpy()
    if r == "y" and groups:
        u = dict(zip(LEVELS, idata.posterior["y_1|g"].to_numpy()[0, 0]))
        eta = eta + np.array([u[v] for v in df["g"]])
    return eta


def test_predict_one_response_on_its_rows(model, data):
    idata = posterior()
    newdata = data.iloc[[0, 2, 5, 7, 10]].reset_index(drop=True).assign(s1=[1, 0, 1, 0, 1], s2=[False, True, True, False, True])
    for r, column in (("y", "s1"), ("y2", "s2")):
        rr, nr = rows(data, column), rows(newdata, column)
        for nd, df, where in ((None, data.iloc[rr], rr), (newdata, newdata.iloc[nr], nr)):
            for include in (True, False):
                lin = model.predict(idata, newdata=nd, kind="linear", resp=r, include_group_effects=include)
                assert lin["eta"].dims == ("chain", "draw", "__obs__") and lin["__obs__"].values.tolist() == where.tolist()
                assert np.allclose(lin["eta"].to_numpy()[0], _eta(idata, df, r, include)[None, :], rtol=1e-13, atol=1e-13)
                assert lin.attrs["response"] == r and lin.attrs["link"] == "identity" and lin.attrs["fit_binding_status"] == "unverified"
                mean = ab.predict(model, idata, newdata=nd, kind="mean", resp=r, include_group_effects=include)
                assert np.array_equal(mean["mu"].to_numpy(), lin["eta"].to_numpy())
    assert model.predict(idata, kind="linear", resp="y").attrs["new_groups"] == "in_sample"
    assert model.predict(idata, kind="linear", resp="y2").attrs["new_groups"] == "not_applicable"


def test_response_draws(model, data):
    idata = posterior(draws=3000)
    out = model.predict(idata, kind="response", resp="y2", seed=5)
    resid = out["y"].to_numpy()[0] - out["mu"].to_numpy()[0]
    assert resid.shape == (3000, rows(data, "s2").size) and np.isclose(resid.std(), PARS["y2"][2], rtol=0.02)
    assert np.array_equal(model.predict(idata, kind="response", resp="y2", seed=5)["y"].to_numpy(), out["y"].to_numpy())


def test_joint_predictions_are_refused_and_resp_checked(model, data):
    idata = posterior()
    for kind in ("linear", "mean", "response"):
        with pytest.raises(ValueError, match=re.escape(SUBSET_REFUSED + " (\"Argument 'resp' must be a single variable name for models "
                                                       "using addition argument 'subset'.\"): each response is predicted on its own rows; "
                                                       "pass resp= one of ['y', 'y2']")):
            model.predict(idata, kind=kind)
    with pytest.raises(ValueError, match=re.escape("resp must name one of the responses ['y', 'y2'], got 'w'")):
        model.predict(idata, kind="linear", resp="w")
    with pytest.raises(FormulaError, match=re.escape("subset(s1) of the response 'y': newdata lacks the indicator column 's1'")):
        model.predict(idata, newdata=data.drop(columns="s1"), kind="linear", resp="y")
    with pytest.raises(FormulaError, match=re.escape("subset(s2) of the response 'y2': no row of the newdata is in the subset")):
        model.predict(idata, newdata=data.assign(s2=False), kind="linear", resp="y2")
    uni = ab.gaussian("y2f ~ x", data, UNI)
    with pytest.raises(ValueError, match=re.escape("resp applies to a multivariate model, naming one of its responses (docs/subset.md); "
                                                   "this model has one response (got 'y2f')")):
        ab.predict(uni, posterior(pars={}), kind="linear", resp="y2f")


def test_unseen_levels(model, data):
    idata = posterior(draws=50)
    nd = pd.DataFrame({"x": [0.1, 0.2, -1.0], "g": ["a", "zz", "zz"], "s1": [1, 1, 1], "s2": [True, True, True]})
    with pytest.raises(FormulaError, match=re.escape("newdata has levels of 'g' not seen in training by the response 'y': ['zz']")):
        model.predict(idata, newdata=nd, kind="linear", resp="y")
    pop = model.predict(idata, newdata=nd, kind="linear", resp="y", new_groups="population", seed=2)
    eta = pop["eta"].to_numpy()[0]
    assert np.allclose(eta[:, 1] - eta[:, 2], PARS["y"][1] * 1.2) and pop.attrs["new_levels"] == ["zz"]
    assert np.allclose(model.predict(idata, newdata=nd, kind="linear", resp="y2", new_groups={"g": "population"})["eta"].to_numpy()[0],
                       PARS["y2"][0] + PARS["y2"][1] * nd["x"].to_numpy())    # y2 has no group term: its new rows are as they are


# -- 4. a formula per response without subset() -----------------------------------------------------------------------
def test_a_formula_per_response_without_subset(data):
    m = ab.gaussian(["y2f ~ x", "wf ~ w + (1 | g)"], data, priors(("y2f", "wf"), groups=("wf",)), rescor=False)
    assert m.design.rows is None and m.spec.subsets == {} and m.observation_dims == {"y2f": "__obs__", "wf": "__obs__"}
    assert m.spec.canonical_formula == "bf(y2f ~ x) + bf(wf ~ w + (1 | g))" and m.design.Y.shape == (len(data), 2)
    vp, _ = point(m, seed=1)
    joint = float(m.pymc_model.compile_logp()(vp))
    parts = sum(float(sub.pymc_model.compile_logp()({v.name: vp[m.variable(r, v.name)] for v in sub.pymc_model.value_vars}))
                for r, sub in m.submodels.items())
    assert abs(joint - parts) <= 1e-12 * abs(joint)
    idata = posterior(pars={"wf": (0.1, -0.3, 1.1)}, groups=("wf",), slope="w")
    idata.posterior["y2f_Intercept"], idata.posterior["y2f_x"], idata.posterior["y2f_sigma"] = (
        idata.posterior["wf_Intercept"] * 2, idata.posterior["wf_w"] * 3, idata.posterior["wf_sigma"])
    joint = m.predict(idata, kind="linear")["eta"].to_numpy()
    one = m.predict(idata, kind="linear", resp="wf")
    assert np.array_equal(one["eta"].to_numpy(), joint[..., 1]) and one["__obs__"].values.tolist() == list(range(len(data)))
    nd = data.iloc[[1, 4]]
    assert np.array_equal(m.predict(idata, newdata=nd, kind="linear", resp="y2f")["eta"].to_numpy(),
                          m.predict(idata, newdata=nd, kind="linear")["eta"].to_numpy()[..., 0])
    mixed = ab.gaussian(["y | subset(s1) ~ x + (1 | g)", "y2f ~ x"], data, priors(("y", "y2f")), rescor=False)
    assert mixed.design.rows["y2f"].tolist() == list(range(len(data))) and mixed.observation_dims["y2f"] == "__obs_y2f__"
    out = mixed.predict(posterior(pars={"y": PARS["y"], "y2f": PARS["y2"]}), newdata=data.iloc[[3, 4]], kind="linear", resp="y2f")
    assert out["__obs__"].values.tolist() == [0, 1] and np.allclose(out["eta"].to_numpy()[0], _eta(None, data.iloc[[3, 4]], "y2"))


# -- 5. priors, describe(), the identity and the fit path --------------------------------------------------------------
def test_default_priors_read_each_subset(data):
    from autobayes.default_priors import location_scale

    pr = ab.default_priors(F, data, rescor=False)
    assert set(pr.resp) == {"y", "y2"} and pr.rescor is None
    for r, column in (("y", "s1"), ("y2", "s2")):
        loc, scale = location_scale(data[r].to_numpy()[rows(data, column)], "identity")
        assert pr.resp[r].intercept == ab.StudentT(3, loc, scale)
    assert pr.resp["y"].intercept != ab.default_priors(["y ~ x", "y2f ~ x"], data.assign(y=data["y"].fillna(5.0)), rescor=False).resp["y"].intercept
    m = ab.gaussian(F, data, pr, rescor=False)
    table = m.prior_table()
    assert [(row["class"], row["resp"]) for row in table if row["prior"]] == [("Intercept", "y"), ("sd", "y"), ("sigma", "y"),
                                                                              ("Intercept", "y2"), ("sigma", "y2")]


def test_describe(model, data):
    text = model.describe().splitlines()
    for line in ["  formula   : ['y | subset(s1) ~ x + (1 | g)', 'y2 | subset(s2) ~ x']",
                 "  expanded  : bf(y | subset(s1) ~ x + (1 | g)) + bf(y2 | subset(s2) ~ x)",
                 "  responses : y ~ x + (1 | g); y2 ~ x   (one univariate model per response, of its own formula on the rows of its subset: "
                 "brms's bf(y | subset(s) ~ ...) + bf(...))",
                 "  data      : n = 36 rows (brms's N); each response's likelihood covers the rows of its subset only",
                 f"  subset    : y on the 24 rows with s1 = 1 (brms's N_y; 0-based {rows(data, 's1')[:12].tolist()}...), the rows of its "
                 "observed variable '__obs_y__'"]:
        assert line in text, line
    assert any(line.startswith(f"  subset    : y2 on the {int(data['s2'].sum())} rows with s2 = 1") for line in text)


def test_target_identity(model, data):
    assert ab.gaussian(F, data, priors(), rescor=False).target_identity == model.target_identity
    moved = data.assign(s2=data["s2"] | (data.index == 0), y2=np.where(data.index == 0, 0.5, data["y2"]))
    assert ab.gaussian(F, moved, priors(), rescor=False).target_identity != model.target_identity
    assert ab.gaussian(["y | subset(s1) ~ x + (1 | g)", "y2 | subset(s1) ~ x"], data.assign(y2=data["y2f"]), priors(),
                       rescor=False).target_identity != model.target_identity


@pytest.mark.nosampling                                    # pm.sample is replaced: nothing samples
def test_fit_cleans_each_response_on_its_rows(model, data, monkeypatch):
    import pymc as pm

    rng = np.random.default_rng(0)
    centred = {r: (rng.normal(size=(1, 3)), rng.normal(size=(1, 3))) for r in model.responses}

    def fake_sample(**kwargs):
        out = {}
        for r, (a, b) in centred.items():
            out.update({f"{r}_Intercept": (("chain", "draw"), a), f"{r}_x": (("chain", "draw"), b)})
        return az.InferenceData(posterior=xr.Dataset(out, coords={"chain": [0], "draw": np.arange(3)}))

    monkeypatch.setattr(pm, "sample", fake_sample)
    post = ab.fit(model, chains=1, draws=3, tune=3).idata.posterior
    for r, column in (("y", "s1"), ("y2", "s2")):                    # brms's b_<resp>_Intercept, centred on the subset's means
        a, b = centred[r]
        assert np.allclose(post[f"{r}_Intercept"].to_numpy(), a - data["x"].to_numpy()[rows(data, column)].mean() * b, rtol=1e-12)


def test_what_is_not_offered(model):
    for call, message in [(lambda: ab.log_lik(model, posterior()), "log_lik() of a multivariate model is not offered"),
                          (lambda: ab.prior_predictive(model, draws=5), "prior_predictive() of a multivariate model is not supported")]:
        with pytest.raises(ValueError, match=re.escape(message)):
            call()
