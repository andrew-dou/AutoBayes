"""brms's multi-membership terms (1 | mm(g, g2, weights = cbind(w1, w2))), (1 | gr(g, by = gby)) and a correlation ID shared
between the response formula and sigma's formula (brms-parity D08; docs/multi-membership-and-by.md).

The term model, the refusals naming each construct, the design blocks and PyMC names, ``value_point`` and the log density
against a hand computation (a log-link family too), brms's normalisation constant of the by-level SDs, predictions by hand
(weights, unseen labels, the shared ID's sigma), the default priors and their table, ``describe()`` and the target identity.
Build-only; nothing samples. The numbers against brms are the oracle's business (tests/oracle: gauss_mm, gauss_gr_by,
gauss_shared_id; acat_cs, parked by D08 (D107), passes since D04, tests/test_more_ordinal.py)."""
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import halfnorm, halfcauchy, norm, poisson

import autobayes as ab
from autobayes.formula import GroupTerm, SharedId

LEVELS = ("a", "b", "c", "d", "e")


def frame(seed=3, n=30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = rng.choice(LEVELS[:4], n)
    g2 = rng.choice(LEVELS[1:], n)                              # 'e' only as a second member: the levels are the union
    w1 = rng.uniform(0.2, 2.0, n)
    df = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n), "z": rng.normal(size=n), "g": g, "g2": g2,
                       "w1": w1, "w2": rng.uniform(0.2, 2.0, n), "cnt": rng.poisson(2.0, n), "h": rng.choice(["p", "q", "r"], n)})
    df["gby"] = np.where(np.isin(df.g, ["a", "b"]), "u", "v")   # g nested in gby
    return df


PRIORS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
MM = "y ~ x + (1 | mm(g, g2, weights = cbind(w1, w2)))"
BY = "y ~ x + (1 | gr(g, by = gby))"
SHARED = ["y ~ x + (1 + x |p| g)", "sigma ~ 1 + (1 |p| g)"]
SHARED_PRIORS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}},
                          cor={"g": ab.LKJ(2)}, sigma={"intercept": ab.Normal(0, 1), "sd": ab.HalfStudentT(3, 2.5)})


def test_the_term_model():
    s = ab.parse_formula(MM, frame())
    t = s.group_terms[0]
    assert t == GroupTerm("mmgg2", ("Intercept",), True, members=("g", "g2"), weights=("w1", "w2"))
    assert t.is_mm and t.custom and not t.derived and t.operands == ("g", "g2") and t.data_columns == ("g", "g2", "w1", "w2")
    assert t.canonical == "(1 | mm(g, g2, weights = cbind(w1, w2)))" and s.canonical_formula == MM
    assert s.levels_by_group["mmgg2"] == LEVELS and s.bambi_formula == "y ~ x" and s.custom_group_terms == (t,)
    assert ab.parse_formula("y ~ x + (1 | mm(g, g2))").group_terms[0].weights == ()
    b = ab.parse_formula(BY, frame()).group_terms[0]
    assert b == GroupTerm("g", ("Intercept",), True, by="gby") and b.canonical == "(1 | gr(g, by = gby))" and b.custom
    assert ab.parse_formula("y ~ x + (1 | gr(g))").group_terms[0] == GroupTerm("g")          # brms's gr(g) is g
    sh = ab.parse_formula(SHARED, frame())
    assert sh.group_terms[0] == GroupTerm("g", ("Intercept", "x"), True, id="p") and not sh.group_terms[0].has_correlation
    assert sh.group_terms[0].canonical == "(1 + x |p| g)" and sh.bambi_formula == "y ~ x + (1 + x | g)"
    assert sh.dpar("sigma").group_terms == (GroupTerm("g", ("Intercept",), True, id="p"),)
    assert sh.shared_ids == (SharedId("p", "g", (("", "Intercept"), ("", "x"), ("sigma", "Intercept"))),)
    assert sh.shared_ids[0].labels == ("Intercept", "x", "sigma_Intercept") and sh.correlated_terms == ()


def bad_weights():
    df = frame()
    df.loc[3, "w1"] = -1.0
    return df


def zero_weights():
    df = frame()
    df.loc[[2, 5], ["w1", "w2"]] = 0.0
    return df


def unnested():
    df = frame()
    df.loc[0, "gby"] = "v" if df.loc[0, "g"] in ("a", "b") else "u"
    return df


REFUSALS = [
    ("y ~ x + (1 + x |p| g)", {}, "in a formula on its own it is not supported here"),
    ("y ~ x + (1 + x ||p| g)", {}, "'||' with an ID is not brms syntax"),
    (["y ~ x + (1 + x |p| g)", "sigma ~ 1 + (1 |p| h)"], {}, "two grouping factors; brms refuses it too"),
    (["y ~ x + (1 + x |p| g)", "sigma ~ z"], {}, "is used by no distributional formula of the list"),
    (["y ~ x + (1 + x | g)", "sigma ~ 1 + (1 |p| g)"], {}, "names no group-level term of the response formula"),
    (["y ~ x + z + (1 + x |p| g)", "sigma ~ z + (1 + z |p| g)"], {}, "varying slopes in a distributional formula are not supported"),
    (["y ~ x + (1 + x | g)", "sigma ~ 1 + (1 | g)"], {}, "group-level terms in a distributional formula ((1 | g)) are not supported"),
    ("y ~ x + (1 |p| mm(g, g2))", {}, "a correlation ID with mm() or gr() is not supported in this slice"),
    ("y ~ x + (1 + x | mm(g, g2))", {}, "only the varying intercept '(1 | mm(...))' is supported"),
    ("y ~ x + (1 | mm(g, g2, scale = FALSE))", {}, "mm()'s argument 'scale' is not supported"),
    ("y ~ x + (1 | mm(g))", {}, "mm() takes two or more member columns"),
    ("y ~ x + (1 | mm(g, g))", {}, "a member column appears twice in mm()"),
    ("y ~ x + (1 | mm(g, g2, weights = cbind(w1)))", {}, "2 member columns and 1 weight columns"),
    ("y ~ x + (1 | mm(g, g2, weights = w1))", {}, "weights = takes cbind(<column>, <column>, ...)"),
    ("y ~ x + (1 | gr(g, cor = FALSE))", {}, "brms's gr() options ['cor'] are not supported"),
    ("y ~ x + (1 + x | gr(g, by = gby))", {}, "varying slopes with gr(by = ) are not supported in this slice"),
    ("y ~ x + (1 | gr(g, by = g))", {}, "cannot be its own by column"),
    ("y ~ x + gby + (1 | gr(g, by = gby))", {}, "'gby' is a by column of the group term and cannot also be a predictor"),
    ("y ~ x + mo(cnt) + (1 | mm(g, g2))", {}, "next to the monotonic term 'mo(cnt)' is not supported in this slice"),
    ("y | weights(w1) ~ x + (1 | gr(g, by = gby))", {}, "next to the addition term 'weights(w1)' is not supported"),
    ("cnt ~ x + (1 | mm(g, g2))", {"family": "bernoulli"}, "is not available for the bernoulli family (logit link)"),
    (["y ~ x + (1 | mm(g, g2))", "sigma ~ z"], {}, "next to a distributional formula is not supported in this slice"),
    ("y ~ x + cs(z)", {}, "brms's category-specific effect cs() is not supported"),
]


@pytest.mark.parametrize("formula,kwargs,message", REFUSALS, ids=[str(r[0]) for r in REFUSALS])
def test_refusals_name_the_construct(formula, kwargs, message):
    with pytest.raises(ab.FormulaError, match=re.escape(message)):
        ab.parse_formula(formula, **kwargs)


@pytest.mark.parametrize("df,message", [(bad_weights, "must be finite and at least 0 on every row"),
                                        (zero_weights, "sum to 0 on row(s) [2, 5]"),
                                        (unnested, "needs each level of 'g' within one level of 'gby'")])
def test_data_refusals(df, message):
    formula = BY if df is unnested else MM
    with pytest.raises(ab.FormulaError, match=re.escape(message)):
        ab.parse_formula(formula, df())


def test_refused_in_multivariate_and_nonlinear_models():
    from autobayes.multivariate import parse_multivariate
    from autobayes.nonlinear import parse_nonlinear

    with pytest.raises(ab.FormulaError, match=re.escape("in a multivariate formula is not supported in this slice")):
        parse_multivariate("mvbind(y, z) ~ x + (1 | mm(g, g2))", frame(), rescor=False)
    with pytest.raises(ab.FormulaError, match=re.escape("in a non-linear parameter's formula is not supported in this slice")):
        parse_nonlinear(["y ~ a * exp(c * x)", "a ~ 1 + (1 | gr(g, by = gby))", "c ~ 1"], frame())
    with pytest.raises(ab.FormulaError, match=re.escape("in a formula on its own it is not supported here")):
        parse_nonlinear(["y ~ a * exp(c * x)", "a ~ 1 + (1 |p| g)", "c ~ 1 + (1 |p| g)"], frame())


def test_mm_design_weights_and_density():
    df = frame()
    m = ab.gaussian(MM, df, PRIORS)
    block = m.design.group_blocks[("mmgg2", "Intercept")]
    W = np.column_stack([df.w1, df.w2]) / (df.w1 + df.w2).to_numpy()[:, None]   # brms's scale = TRUE
    idx = [np.array([LEVELS.index(v) for v in df[c]]) for c in ("g", "g2")]
    Z = np.zeros((len(df), 5))
    for k in range(2):
        np.add.at(Z, (np.arange(len(df)), idx[k]), W[:, k])
    assert block.levels == LEVELS and np.allclose(block.Z, Z) and np.allclose(m.design.Z, Z)
    assert all(np.array_equal(a, b) for a, b in zip(block.memberships, idx)) and np.allclose(np.column_stack(block.weights), W)
    roles = m.group_coefficient_names[("mmgg2", "Intercept")]
    assert roles == {"sd": "1|mmgg2_sigma", "offset": "1|mmgg2_offset", "group_effect": "1|mmgg2", "coordinate": "mmgg2__factor_dim"}
    z = np.linspace(-1, 1, 5)
    vp = m.value_point(intercept=0.3, b=[0.5], sd=0.8, z=z, sigma=1.3)
    eta = 0.3 + 0.5 * (df.x - df.x.mean()).to_numpy() + Z @ (0.8 * z)
    ref = (norm.logpdf(df.y, eta, 1.3).sum() + norm.logpdf(0.3, 0, 5) + norm.logpdf(0.5, 0, 2) + halfnorm.logpdf(0.8)
           + norm.logpdf(z).sum() + halfnorm.logpdf(1.3))
    assert np.isclose(float(m.pymc_model.compile_logp(jacobian=False)(vp)), ref, rtol=1e-12)
    # the weights scaled by any positive factor give the same weights (to rounding); no weights are equal weights
    scaled = df.assign(w1=3 * df.w1, w2=3 * df.w2)
    assert np.allclose(ab.gaussian(MM, scaled, PRIORS).design.Z, m.design.Z, rtol=1e-15, atol=0)
    even = ab.gaussian("y ~ x + (1 | mm(g, g2))", df, PRIORS)
    assert np.allclose(np.column_stack(even.design.group_blocks[("mmgg2", "Intercept")].weights), 0.5)
    assert even.target_identity != m.target_identity


def test_mm_in_a_log_link_family_and_next_to_another_group_term():
    df = frame()
    pr = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd={"mmgg2": ab.HalfNormal(1), "h": ab.HalfCauchy(1)})
    m = ab.poisson("cnt ~ x + (1 | mm(g, g2, weights = cbind(w1, w2))) + (1 | h)", df, pr)
    Z = m.design.group_blocks[("mmgg2", "Intercept")].Z
    H = m.design.group_blocks[("h", "Intercept")].Z
    assert m.design.Z.shape == (len(df), 8) and np.array_equal(m.design.Z[:, :5], Z) and np.array_equal(m.design.Z[:, 5:], H)
    zg, zh = np.linspace(-1, 1, 5), np.array([0.4, -0.2, 0.1])
    vp = m.value_point(intercept=0.2, b=[0.3], sd={"mmgg2": 0.6, "h": 0.9}, z={"mmgg2": zg, "h": zh})
    eta = 0.2 + 0.3 * (df.x - df.x.mean()).to_numpy() + Z @ (0.6 * zg) + H @ (0.9 * zh)
    ref = (poisson.logpmf(df.cnt, np.exp(eta)).sum() + norm.logpdf(0.2, 0, 2) + norm.logpdf(0.3, 0, 1) + halfnorm.logpdf(0.6)
           + halfcauchy.logpdf(0.9) + norm.logpdf(zg).sum() + norm.logpdf(zh).sum())
    assert np.isclose(float(m.pymc_model.compile_logp(jacobian=False)(vp)), ref, rtol=1e-12)


def test_by_design_names_and_brms_normalisation_constant():
    df = frame()
    m = ab.gaussian(BY, df, PRIORS)
    block = m.design.group_blocks[("g", "Intercept")]
    assert block.by_levels == ("u", "v") and block.by_index.tolist() == [0, 0, 1, 1]
    roles = m.group_coefficient_names[("g", "Intercept")]
    assert roles["by_coordinate"] == "g__by_dim" and m.pymc_model.named_vars_to_dims[roles["sd"]] == ("g__by_dim",)
    z = np.array([0.3, -0.4, 1.1, -0.9])
    vp = m.value_point(intercept=0.1, b=[0.2], sd={"g": [0.7, 1.4]}, z=z, sigma=0.9)
    sds = np.array([0.7, 0.7, 1.4, 1.4])
    eta = 0.1 + 0.2 * (df.x - df.x.mean()).to_numpy() + (sds * z)[[("a", "b", "c", "d").index(v) for v in df.g]]
    # brms's lprior of sd_1 (1 x 2) is normal_lpdf of both entries minus normal_lccdf(0 | 0, 1) once: log 2 below the half-normals
    ref = (norm.logpdf(df.y, eta, 0.9).sum() + norm.logpdf(0.1, 0, 5) + norm.logpdf(0.2, 0, 2) + halfnorm.logpdf([0.7, 1.4]).sum()
           - np.log(2) + norm.logpdf(z).sum() + halfnorm.logpdf(0.9))
    assert np.isclose(float(m.pymc_model.compile_logp(jacobian=False)(vp)), ref, rtol=1e-12)
    assert roles["by_normalisation"] in m.pymc_model.named_vars
    gamma = ab.gaussian(BY, df, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.Gamma(2, 1), sigma=ab.HalfNormal(1)))
    assert roles["by_normalisation"] not in gamma.pymc_model.named_vars     # a prior on (0, inf) needs no correction
    with pytest.raises(ValueError, match=re.escape("must be 2 finite positive SDs, one per level ['u', 'v']")):
        m.value_point(intercept=0.1, b=[0.2], sd={"g": 0.7}, z=z, sigma=0.9)


def shared_model():
    return ab.gaussian(SHARED, frame(), SHARED_PRIORS)


def test_shared_id_names_correlation_and_density():
    m, df = shared_model(), frame()
    assert m.correlation_coefs("g") == ("Intercept", "x", "sigma_Intercept")
    assert m.dpar_group_names == {("sigma", "g", "Intercept"): {"sd": "sigma_1|g_sigma", "offset": "sigma_1|g_offset",
                                                                "group_effect": "sigma_1|g", "coordinate": "g__factor_dim"}}
    assert [str(v) for v in m.pymc_model.coords["g__cor_dim"]] == ["Intercept", "x", "sigma_Intercept"]
    L = np.linalg.cholesky(np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]]))
    zs = {c: np.linspace(-1, 1, 4) * (k + 1) / 3 for k, c in enumerate(("Intercept", "x", "sigma"))}
    vp = m.value_point(intercept=0.2, b=[0.4], sd={"g": {"Intercept": 0.8, "x": 0.5}}, z={"g": {"Intercept": zs["Intercept"], "x": zs["x"]}},
                       L={"g": L}, sigma={"intercept": -0.3, "b": [], "sd": {"g": 0.6}, "z": {"g": zs["sigma"]}})
    r = (np.diag([0.8, 0.5, 0.6]) @ L @ np.vstack([zs["Intercept"], zs["x"], zs["sigma"]])).T      # brms's r = (diag(sd) L z)^T
    j = np.array([("a", "b", "c", "d").index(v) for v in df.g])
    mu = 0.2 + 0.4 * (df.x - df.x.mean()).to_numpy() + r[j, 0] + df.x.to_numpy() * r[j, 1]
    sigma = np.exp(-0.3 + r[j, 2])
    fn = m.pymc_model.compile_fn(m.pymc_model.replace_rvs_by_values([m.pymc_model["mu"], m.pymc_model["sigma"]]),
                                 inputs=m.pymc_model.value_vars, point_fn=True, on_unused_input="ignore")
    got_mu, got_sigma = fn(vp)
    assert np.allclose(got_mu, mu, rtol=1e-12) and np.allclose(got_sigma, sigma, rtol=1e-12)
    assert np.allclose(m.pymc_model.compile_fn(m.pymc_model.replace_rvs_by_values([m.pymc_model["cor|g"]]), inputs=m.pymc_model.value_vars,
                                               point_fn=True, on_unused_input="ignore")(vp)[0], L @ L.T)


def shared_draws(m):
    rng = np.random.default_rng(0)
    coords = {"chain": [0], "draw": [0, 1], "g__factor_dim": list(LEVELS[:4])}
    data = {"Intercept": (("chain", "draw"), [[0.1, -0.2]]), "x": (("chain", "draw"), [[0.5, 0.3]]),
            "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(size=(1, 2, 4))),
            "x|g": (("chain", "draw", "g__factor_dim"), rng.normal(size=(1, 2, 4))),
            "sigma_1|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.3, size=(1, 2, 4))),
            "sigma_Intercept": (("chain", "draw"), [[-0.2, 0.1]])}
    import arviz as az
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


def test_shared_id_predictions_by_hand():
    m, df = shared_model(), frame()
    idata = shared_draws(m)
    post = idata.posterior
    j = [("a", "b", "c", "d").index(v) for v in df.g]
    want = np.exp(post["sigma_Intercept"].values[..., None] + post["sigma_1|g"].values[..., j])
    assert np.allclose(m.predict(idata, kind="mean", dpar="sigma")["sigma"].values, want)
    nog = m.predict(idata, kind="mean", dpar="sigma", include_group_effects=False)["sigma"].values
    assert np.allclose(nog, np.exp(post["sigma_Intercept"].values[..., None]) * np.ones(len(df)))
    nd = df.iloc[:3].copy()
    nd.loc[nd.index[0], "g"] = "zz"
    with pytest.raises(ab.FormulaError, match=re.escape("correlation ID 'p' is shared with a distributional formula are not offered")):
        m.predict(idata, newdata=nd, kind="linear", new_groups="population")


def mm_draws():
    rng = np.random.default_rng(1)
    coords = {"chain": [0], "draw": [0, 1, 2], "mmgg2__factor_dim": list(LEVELS)}
    data = {"Intercept": (("chain", "draw"), [[0.1, 0.2, 0.3]]), "x": (("chain", "draw"), [[0.5, 0.4, 0.3]]),
            "1|mmgg2": (("chain", "draw", "mmgg2__factor_dim"), rng.normal(size=(1, 3, 5))),
            "1|mmgg2_sigma": (("chain", "draw"), [[0.7, 0.8, 0.9]]), "sigma": (("chain", "draw"), [[1.0, 1.1, 1.2]])}
    import arviz as az
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


def test_mm_predictions_weights_and_unseen_labels():
    df = frame()
    m = ab.gaussian(MM, df, PRIORS)
    idata = mm_draws()
    post = idata.posterior
    u = post["1|mmgg2"].values
    nd = df.iloc[:4].copy()
    nd["g"] = ["a", "zz", "zz", "b"]
    nd["g2"] = ["zz", "e", "c", "yy"]
    W = np.column_stack([nd.w1, nd.w2]) / (nd.w1 + nd.w2).to_numpy()[:, None]
    base = post["Intercept"].values[..., None] + post["x"].values[..., None] * nd.x.to_numpy()
    with pytest.raises(ab.FormulaError, match="not seen in training: \\['yy', 'zz'\\]"):
        m.predict(idata, newdata=nd, kind="linear")
    ds = m.predict(idata, newdata=nd, kind="linear", new_groups="resample_observed", seed=5)
    eta = ds["eta"].values
    assert ds.attrs["new_levels"] == ["yy", "zz"]
    # rows 1 and 2 share the unseen 'zz' as their first member: the same effect in a draw; the seen labels are their own
    shared = (eta[..., 1] - base[..., 1] - W[1, 1] * u[..., 4]) / W[1, 0]
    assert np.allclose(shared, (eta[..., 2] - base[..., 2] - W[2, 1] * u[..., 2]) / W[2, 0])
    assert all(np.isclose(v, u[0, d]).any() for d, v in enumerate(shared[0]))     # resampled: one observed level's effect
    pop = m.predict(idata, newdata=nd, kind="linear", new_groups="population", seed=5)["eta"].values
    drawn = (pop[..., 1] - base[..., 1] - W[1, 1] * u[..., 4]) / W[1, 0]      # 'zz' drawn once per draw, shared by rows 1 and 2
    assert np.allclose(drawn, (pop[..., 2] - base[..., 2] - W[2, 1] * u[..., 2]) / W[2, 0])
    assert not any(np.isclose(v, u[0, d]).any() for d, v in enumerate(drawn[0]))   # a fresh Normal(0, sd) draw, not a level's
    fitted = m.predict(idata, kind="linear")["eta"].values
    Z = m.design.group_blocks[("mmgg2", "Intercept")].Z
    ref = post["Intercept"].values[..., None] + post["x"].values[..., None] * df.x.to_numpy() + np.einsum("cdj,nj->cdn", u, Z)
    assert np.allclose(fitted, ref)
    assert np.allclose(m.predict(idata, kind="linear", include_group_effects=False)["eta"].values,
                       post["Intercept"].values[..., None] + post["x"].values[..., None] * df.x.to_numpy())


def test_by_predictions_refuse_new_levels_and_contradicting_by():
    df = frame()
    m = ab.gaussian(BY, df, PRIORS)
    rng = np.random.default_rng(2)
    import arviz as az
    idata = az.InferenceData(posterior=xr.Dataset(
        {"Intercept": (("chain", "draw"), [[0.1]]), "x": (("chain", "draw"), [[0.5]]),
         "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(size=(1, 1, 4))), "sigma": (("chain", "draw"), [[1.0]]),
         "1|g_sigma": (("chain", "draw", "g__by_dim"), [[[0.5, 0.9]]])},
        coords={"chain": [0], "draw": [0], "g__factor_dim": list("abcd"), "g__by_dim": ["u", "v"]}))
    nd = df.iloc[:2].copy()
    eta = m.predict(idata, newdata=nd, kind="linear")["eta"].values
    u = idata.posterior["1|g"].values
    assert np.allclose(eta, 0.1 + 0.5 * nd.x.to_numpy() + u[..., ["abcd".index(v) for v in nd.g]])
    bad = nd.copy()
    bad.loc[bad.index[0], "gby"] = "v" if bad.g.iloc[0] in ("a", "b") else "u"
    with pytest.raises(ab.FormulaError, match="it was fitted within"):
        m.predict(idata, newdata=bad, kind="linear")
    new = nd.copy()
    new.loc[new.index[0], "g"] = "zz"
    with pytest.raises(ab.FormulaError, match=re.escape("(one SD per level of 'gby') are not offered in this slice")):
        m.predict(idata, newdata=new, kind="linear", new_groups="population")


def test_default_priors_and_their_table():
    df = frame()
    pr = ab.default_priors(SHARED, df)
    log_scale = ab.default_priors(["y ~ x", "sigma ~ 1"], df).sigma["intercept"].params["sigma"]
    assert pr.sigma["sd"] == {"g": ab.HalfStudentT(3, log_scale)} and pr.cor == {"g": ab.LKJ(1)}
    m = ab.gaussian(SHARED, df, pr)
    rows = [(r["class"], r["coef"], r["group"], r["dpar"], r["prior"], r["lb"]) for r in m.prior_table()]
    assert ("sd", "", "", "sigma", f"student_t(3, 0, {log_scale:g})", "0") in rows
    assert ("sd", "", "g", "sigma", "", "") in rows and ("sd", "Intercept", "g", "sigma", "", "") in rows
    assert ("cor", "", "", "", "lkj(1)", "") in rows and ("cor", "", "g", "", "", "") in rows
    assert "sigma:sd:g" in m.default_prior_labels
    mm = ab.gaussian(MM, df, ab.default_priors(MM, df))
    assert ("sd", "", "mmgg2") in [(r["class"], r["coef"], r["group"]) for r in mm.prior_table()]


def test_describe_and_target_identity():
    df = frame()
    text = ab.gaussian(MM, df, PRIORS).describe()
    assert "multi-membership grouping mm(g, g2, weights = cbind(w1, w2))" in text and "W_g,i * u[g_i] + W_g2,i * u[g2_i]" in text
    by = ab.gaussian(BY, df, PRIORS)
    assert "gr(g, by = gby): one SD per level of 'gby' (u: a, b; v: c, d)" in by.describe()
    assert "log(sd_g[u, v]) (vector; dims g__by_dim)" in by.describe()
    plain = ab.gaussian("y ~ x + (1 | g)", df, PRIORS)
    assert plain.target_identity != by.target_identity
    shared = shared_model().describe()
    assert "shares one correlation matrix Cor_g = L_g L_g^T over (Intercept, x, sigma_Intercept)" in shared
    assert "sd_sigma,g ~ HalfStudentT(nu=3, sigma=2.5)   (brms class sd with dpar = sigma)" in shared
