"""The per-category families read a posterior by dimension name: ``predict()``, ``log_lik()``, ``waic()`` and ``loo()`` give the
same numbers on draws laid out as a fit leaves them as on the same draws in the graph's order. Bambi's post-processing of every
fit moves each ``__factor_dim`` after the other dimensions, so a dirichlet or logistic_normal model's group effects, ``(level,
category)`` in the graph (and in the oracle's fixture draws), come out of a fit as ``(category, level)``; reading them by position
gave wrong numbers when the factor has K - 1 levels, a broadcast error when it has fewer and an IndexError when it has more. The
categorical family takes no group-level term, but a fit moves its categorical coefficients' category dimension first; the
count-matrix families (multinomial, dirichlet_multinomial) are sampled with PyMC's ``sample``, which keeps the graph's order; a
Gaussian model with correlated varying slopes is the control of the families with one effect per level. New levels: refused in
the same words under both layouts for the dirichlet and logistic_normal families (their every category has its own group
term), drawn identically under both for the count-matrix families and the Gaussian control. No MCMC: the draws are computed
from seeded points of each graph's value space."""
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import autobayes as ab
from autobayes.formula import FormulaError


def frame(n=24, seed=1, categories=3, levels=6):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(categories, 3.0), size=n)
    counts = rng.multinomial(10, [0.2, 0.3, 0.5], size=n)
    return pd.DataFrame({**{f"s{k}": p[:, k] for k in range(categories)}, "x": rng.normal(size=n),
                         "z": np.array(["p", "q", "r"])[np.arange(n) % 3],
                         "g": np.array([f"g{j}" for j in range(levels)])[np.arange(n) % levels],
                         "h": np.array(["h1", "h2", "h3", "h4"])[(np.arange(n) // 2) % 4],
                         "k1": counts[:, 0], "k2": counts[:, 1], "k3": counts[:, 2], "N": counts.sum(1),
                         "y": rng.normal(size=n), "cat": np.array(["r", "s", "t"])[np.arange(n) % 3]})


def priors(**kw):
    return ab.Priors(**{"intercept": ab.Normal(0, 2), "b": ab.Normal(0, 1), "sd": ab.HalfNormal(1), **kw})


SIMPLEX = "cbind(s0, s1, s2)"
# J levels of g against K - 1 non-reference categories: J > K - 1 (IndexError before), J < K - 1 (broadcast error), J = K - 1
# (silently wrong numbers); the categorical family's coefficients of z, the count-matrix families and the Gaussian control
BUILDERS = {
    "dirichlet_two_factors": lambda: ab.dirichlet(f"{SIMPLEX} ~ x + z + (1 | g) + (1 | h)", frame(),
                                                  priors(sd={"g": ab.HalfNormal(1), "h": ab.HalfNormal(1)}, phi=ab.Gamma(2, 0.5)),
                                                  categorical={"z": "p"}),
    "dirichlet_few_levels": lambda: ab.dirichlet("cbind(s0, s1, s2, s3) ~ x + (1 | g)", frame(categories=4, levels=2),
                                                 priors(phi=ab.Gamma(2, 0.5))),
    "logistic_normal_square": lambda: ab.logistic_normal(f"{SIMPLEX} ~ x + (1 | g)", frame(levels=2),
                                                         priors(sigma=ab.HalfNormal(1), lncor=ab.LKJ(2)), refcat="s1"),
    "categorical": lambda: ab.categorical("cat ~ x + z", frame(), priors(sd=None), categorical={"z": "p"}),
    "multinomial": lambda: ab.multinomial("cbind(k1, k2, k3) | trials(N) ~ x + z + (1 | g)", frame(), priors(), categorical={"z": "p"}),
    "dirichlet_multinomial": lambda: ab.dirichlet_multinomial("cbind(k1, k2, k3) | trials(N) ~ x + (1 | g)", frame(),
                                                              priors(phi=ab.Gamma(2, 0.5))),
    "gaussian_correlated": lambda: ab.gaussian("y ~ x + (1 + x | g)", frame(),
                                               priors(sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}},
                                                      sigma=ab.HalfNormal(1), cor=ab.LKJ(2))),
}


def graph_posterior(model, seed=7, chains=2, draws=10) -> az.InferenceData:
    """Draws of every unobserved variable without an observation dimension under the graph's names, dims, dim order and
    coordinates (the layout of the oracle's fixture draws), computed from seeded points of the value space so that the
    deterministics (the group effects ``sd * z``, a correlated term's ``L``) agree with the free variables."""
    pmm = model.pymc_model
    dims_of = pmm.named_vars_to_dims
    keep = [v for v in pmm.unobserved_RVs if not any(str(d) == "__obs__" for d in (dims_of.get(v.name) or ()))]
    fn = pmm.compile_fn(pmm.replace_rvs_by_values(keep), inputs=pmm.value_vars, on_unused_input="ignore", point_fn=True)
    start, rng = pmm.initial_point(), np.random.default_rng(seed)
    values = [fn({k: np.asarray(v, dtype=float) + 0.4 * rng.normal(size=np.shape(v)) for k, v in start.items()})
              for _ in range(chains * draws)]
    data, used = {}, set()
    for i, v in enumerate(keep):
        arr = np.stack([np.asarray(point[i]) for point in values]).reshape(chains, draws, *np.shape(values[0][i]))
        dims = tuple(dims_of.get(v.name) or ())
        data[v.name] = (("chain", "draw", *dims), arr)
        used.update(dims)
    coords = {k: list(c) for k, c in pmm.coords.items() if k in used}
    return az.InferenceData(posterior=xr.Dataset(data, coords={**coords, "chain": np.arange(chains), "draw": np.arange(draws)}))


def fit_layout(model, idata) -> az.InferenceData:
    """The same draws as a fit lays them out. A Bambi-built model's fit ends in Bambi's ``_clean_results`` (``ab.fit`` through
    ``bambi_model.fit``, either backend), which drops the ``_offset`` variables and transposes every variable to its own order;
    the dims and the kept variables are taken from it, the values from ``idata`` (it also uncentres the intercept, which is
    not the point here). A count-matrix model is sampled with PyMC's ``sample`` (``sample_posterior``), which keeps the
    graph's order and drops the offsets."""
    if getattr(model, "bambi_model", None) is None:
        kept = [k for k in idata.posterior.data_vars if not k.endswith("_offset")]
        return az.InferenceData(posterior=idata.posterior[kept])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = model.bambi_model.backend._clean_results(idata.copy(), omit_offsets=True, include_response_params=False)
    return az.InferenceData(posterior=xr.Dataset({k: idata.posterior[k].transpose(*clean.posterior[k].dims)
                                                  for k in clean.posterior.data_vars}))


@pytest.fixture(scope="module", params=list(BUILDERS))
def case(request):
    model = BUILDERS[request.param]()
    graph = graph_posterior(model)
    return request.param, model, graph, fit_layout(model, graph)


def new_rows(model, unseen=False):
    """Six new rows: fitted levels of g (and h) only, or two levels the fit has not seen among them."""
    rows = frame(6, seed=5).drop(columns=["s0", "s1", "s2", "k1", "k2", "k3", "y", "cat"]).assign(N=10)
    if unseen:
        return rows.assign(g=["g0", "new1", "g1", "new1", "new2", "g0"])
    if "g" in model.data:                                              # the categorical model has no factor
        levels = sorted(model.data["g"].astype(str).unique())
        rows["g"] = [levels[i % len(levels)] for i in range(len(rows))]
    return rows


def has_groups(model) -> bool:
    return bool(model.spec.factors if isinstance(model, ab.MultinomialModel) else model.spec.group_terms)


def simplex_family(name: str) -> bool:
    """The families whose every non-reference category has its own group term (and which refuse new levels)."""
    return name.startswith(("dirichlet_two", "dirichlet_few", "logistic_normal"))


def assert_same(a, b):
    """Every variable, dimension, coordinate and attribute equal, the values bit for bit."""
    if isinstance(a, xr.Dataset):
        assert list(a.data_vars) == list(b.data_vars) and a.attrs == b.attrs
        for k in a.data_vars:
            assert a[k].dims == b[k].dims
            np.testing.assert_array_equal(a[k].to_numpy(), b[k].to_numpy())
        for c in a.coords:
            np.testing.assert_array_equal(a[c].to_numpy(), b[c].to_numpy())
    elif isinstance(a, xr.DataArray):
        assert a.dims == b.dims and a.attrs == b.attrs
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    else:                                                             # Waic / Loo: the pointwise table and the estimates
        pd.testing.assert_frame_equal(a.pointwise, b.pointwise, check_exact=True)
        pd.testing.assert_frame_equal(a.estimates, b.estimates, check_exact=True)


def test_the_fit_layout_is_what_a_fit_leaves(case):
    """The layouts differ where the bug was: a per-category family's group effects come out of a fit as (category, level), the
    categorical family's vector coefficients as (category, coefficient level); the others keep the graph's order."""
    name, model, graph, fit = case
    moved = {k: fit.posterior[k].dims for k in fit.posterior.data_vars if fit.posterior[k].dims != graph.posterior[k].dims}
    assert not any(k.endswith("_offset") for k in fit.posterior.data_vars)
    if simplex_family(name):
        category = model.names["category_dim"]
        for (g, coef), roles in model.group_coefficient_names.items():
            assert graph.posterior[roles["group_effect"]].dims == ("chain", "draw", roles["coordinate"], category)
            assert moved[roles["group_effect"]] == ("chain", "draw", category, roles["coordinate"])
    elif name == "categorical":
        assert moved == {"z": ("chain", "draw", model.names["category_dim"], "z_dim")}
    else:
        assert moved == {}


def test_predictions_are_the_same_in_both_layouts(case):
    name, model, graph, fit = case
    kinds = [k for k in ("linear", "mean", "response") if k in model.family_spec.prediction_kinds]
    groups = (True, False) if has_groups(model) else (True,)
    for kind in kinds:
        for newdata in (None, new_rows(model)):
            for include in groups:
                call = dict(newdata=newdata, kind=kind, include_group_effects=include, seed=3)
                assert_same(ab.predict(model, graph, **call), ab.predict(model, fit, **call))


def test_the_group_effects_read_by_name_on_the_fit_layout(case):
    """A hand-computed expectation independent of either layout's positions: on the fit layout, the linear predictor with group
    effects minus the one without is each row's effect of its level, selected by label (``.sel``) for every grouping factor."""
    name, model, graph, fit = case
    if not has_groups(model):
        pytest.skip("the categorical family has no group-level term")
    rows = new_rows(model)
    post = fit.posterior
    lin = {inc: ab.predict(model, fit, newdata=rows, kind="linear", include_group_effects=inc)["eta"] for inc in (True, False)}

    def effect(roles, t, coef):                                        # each row's effect of its level, by label
        u = post[roles["group_effect"]].sel({roles["coordinate"]: xr.DataArray(rows[t.factor].astype(str).to_numpy(), dims="__obs__")})
        return u * (1.0 if coef == "Intercept" else xr.DataArray(rows[coef].to_numpy(), dims="__obs__"))

    if isinstance(model, ab.MultinomialModel):                         # each category's formula's own terms, in category order
        want = xr.concat([sum(effect(model.mu_group_names[(mu, t.factor, "Intercept")], t, "Intercept")
                              for t in model.spec.mu(mu).group_terms) for mu in model.spec.mu_names], dim="category")
    else:
        want = sum(effect(model.group_coefficient_names[(t.factor, c)], t, c) for t in model.spec.group_terms for c in t.coefs)
        if model.family_spec.per_category:
            want = want.rename({model.names["category_dim"]: "category"})
    want = want.transpose("chain", "draw", "__obs__", *(("category",) if "category" in want.dims else ()))
    np.testing.assert_allclose((lin[True] - lin[False]).to_numpy(), want.to_numpy(), rtol=1e-12, atol=1e-12)


def test_log_lik_and_the_information_criteria_are_the_same_in_both_layouts(case):
    name, model, graph, fit = case
    assert_same(ab.log_lik(model, graph), ab.log_lik(model, fit))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_same(ab.waic(model, graph), ab.waic(model, fit))
        assert_same(ab.loo(model, graph), ab.loo(model, fit))


def test_new_levels_are_treated_alike_in_both_layouts(case):
    """Drawn identically (the same random numbers, the same effects) where a family offers new levels, refused in the same
    words under both layouts where it does not (the dirichlet and logistic_normal families)."""
    name, model, graph, fit = case
    if not has_groups(model):
        pytest.skip("the categorical family has no group-level term")
    unseen = new_rows(model, unseen=True)
    kinds = [k for k in ("linear", "mean", "response") if k in model.family_spec.prediction_kinds]
    for policy in ("population", "resample_observed"):
        for kind in kinds:
            call = dict(newdata=unseen, kind=kind, new_groups=policy, seed=8)
            if simplex_family(name):
                errors = []
                for idata in (graph, fit):
                    with pytest.raises(FormulaError, match=r"new levels in a .* model, whose every non-reference category has its own "
                                                           r"group term, are not offered") as e:
                        ab.predict(model, idata, **call)
                    errors.append(str(e.value))
                assert errors[0] == errors[1]
            else:
                a, b = ab.predict(model, graph, **call), ab.predict(model, fit, **call)
                assert_same(a, b)
                assert a.attrs["new_levels"] == ["new1", "new2"]
    for idata in (graph, fit):                                         # the default policy refuses them for every family
        with pytest.raises(FormulaError, match=r"not seen in training: \['new1', 'new2'\]"):
            ab.predict(model, idata, newdata=unseen, kind="linear")
