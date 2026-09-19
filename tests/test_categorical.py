"""M3a-1: one treatment-coded categorical predictor. Checks C1-C8 and C12 of docs/m3a1-decision.md."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit

import autobayes as ab
from tests.reference_binomial import reference_logp_binomial
from tests.reference_logp import build_design, design_columns, prior_logpdf, reference_logp, reference_logp_bernoulli

LEVELS, PROPS = ("a", "b", "c"), (0.5, 0.3, 0.2)
BETA = {"a": 0.0, "b": -0.8, "c": 0.6}
TRUTH = dict(a=1.0, bx=0.5, sd=0.5, sigma=0.4)
G_PRIORS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
G_LEVEL = ab.Priors(intercept=ab.Normal(0, 5), b={"x": ab.Normal(0, 2), "z": {"b": ab.Normal(0, 1), "c": ab.Normal(0.5, 2)}},
                    sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
L_PRIORS = ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
L_LEVEL = ab.Priors(intercept=ab.Normal(0, 1.5), b={"x": ab.Normal(0, 1), "z": {"b": ab.StudentT(4, 0, 1), "c": ab.StudentT(6, 0.2, 2)}},
                    sd=ab.HalfNormal(1))


def ref_priors(priors: ab.Priors, spec) -> dict:
    """The reference's prior description, keyed by design column, from the library's own Priors."""
    coef = priors.coefficient_priors(spec)
    out = {"intercept": (priors.intercept.name, dict(priors.intercept.params)),
           "b": {col: (pr.name, dict(pr.params)) for col, pr in coef.items()},
           "sd": (priors.sd.name, dict(priors.sd.params))}
    if priors.sigma is not None:
        out["sigma"] = (priors.sigma.name, dict(priors.sigma.params))
    return out


def simulate_cat(n=90, J=6, seed=5, trials=20):
    """Every level and every group present (asserted); unequal level proportions."""
    rng = np.random.default_rng(seed)
    z = rng.choice(LEVELS, size=n, p=PROPS)
    g = np.array([f"g{i % J}" for i in range(n)]); rng.shuffle(g)
    x = rng.normal(size=n)
    u = rng.normal(0, TRUTH["sd"], J)
    eta = TRUTH["a"] + TRUTH["bx"] * x + np.array([BETA[v] for v in z]) + u[[int(s[1:]) for s in g]]
    df = pd.DataFrame({"y": eta + rng.normal(0, TRUTH["sigma"], n), "x": x, "z": z, "g": g,
                       "y01": (rng.uniform(size=n) < expit(eta)).astype(int),
                       "k": rng.binomial(trials, expit(eta)), "n": trials})
    assert set(df.z) == set(LEVELS) and df.g.nunique() == J
    return df


def build(family, df, priors, **kw):
    f = {"gaussian": "y ~ x + z + (1 | g)", "bernoulli": "y01 ~ x + z + (1 | g)", "binomial": "k | trials(n) ~ x + z + (1 | g)"}[family]
    return getattr(ab, family)(f, df, priors, **kw)


def hand_design(df, terms, categorical, center):
    _, X_raw, _, gidx, _ = build_design(df, "y", list(terms), "g", False, categorical)
    _, Xc, xbar, _, _ = build_design(df, "y", list(terms), "g", center, categorical)
    return X_raw, Xc, xbar, gidx


# -- C1: declaration and validation --------------------------------------------------------------
def test_declaration_and_validation():
    df = simulate_cat()
    with pytest.raises(ab.FormulaError, match="declared categorical with its reference level"):
        ab.gaussian("y ~ x + z + (1 | g)", df, G_PRIORS)                    # undeclared string column
    with pytest.raises(ab.FormulaError, match="does not occur in the fitted rows; observed levels: \\['a', 'b', 'c'\\]"):
        build("gaussian", df, G_PRIORS, categorical={"z": "q"})
    with pytest.raises(ab.FormulaError, match="needs an explicit reference level"):
        build("gaussian", df, G_PRIORS, categorical={"z": None})
    with pytest.raises(ab.FormulaError, match="is the grouping factor"):
        build("gaussian", df, G_PRIORS, categorical={"z": "a", "g": "g0"})
    with pytest.raises(ab.FormulaError, match="is the response"):
        build("gaussian", df, G_PRIORS, categorical={"z": "a", "y": 1})
    with pytest.raises(ab.FormulaError, match="is the trials column"):
        build("binomial", df, L_PRIORS, categorical={"z": "a", "n": 20})
    with pytest.raises(ab.FormulaError, match="not a term of"):
        build("gaussian", df, G_PRIORS, categorical={"z": "a", "w": "a"})
    with pytest.raises(ab.FormulaError, match="must be a mapping"):
        build("gaussian", df, G_PRIORS, categorical=["z"])
    d = df.copy(); d["z"] = d["z"].map({"a": True, "b": False, "c": True})
    with pytest.raises(ab.FormulaError, match="boolean"):
        build("gaussian", d, G_PRIORS, categorical={"z": True})
    d = df.copy(); d["z"] = d["z"].map({"a": 1.0, "b": 1.5, "c": 3.0})
    with pytest.raises(ab.FormulaError, match="must be integral"):
        build("gaussian", d, G_PRIORS, categorical={"z": 1})
    d = df.copy(); d["z"] = d["z"].map({"a": 1 + 0j, "b": 2 + 0j, "c": 3 + 0j})
    with pytest.raises(ab.FormulaError, match="complex"):
        build("gaussian", d, G_PRIORS, categorical={"z": 1})
    d = df.copy(); d["z"] = "a"
    with pytest.raises(ab.FormulaError, match="constant"):
        build("gaussian", d, G_PRIORS, categorical={"z": "a"})
    d = df.copy(); d.loc[3, "z"] = None
    with pytest.raises(ab.FormulaError, match="missing values"):
        build("gaussian", d, G_PRIORS, categorical={"z": "a"})


def test_dtypes_levels_and_unused_categories():
    df = simulate_cat()
    d = df.copy(); d["zi"] = d["z"].map({"a": 1, "b": 2, "c": 3})
    m = ab.gaussian("y ~ x + zi + (1 | g)", d, G_PRIORS)                     # an integer column stays numeric
    assert m.spec.columns == ("x", "zi") and m.spec.categoricals == () and m.design.X_raw[:, 1].tolist() == d.zi.tolist()
    m = ab.gaussian("y ~ x + zi + (1 | g)", d, G_PRIORS, categorical={"zi": 2})   # unless declared
    cat = m.spec.categorical("zi")
    assert cat.levels == ("1", "2", "3") and cat.reference == "2" and cat.coefficients == ("1", "3")
    assert m.spec.columns == ("x", "zi[1]", "zi[3]") and list(m.pymc_model.coords["zi_dim"]) == ["1", "3"]
    assert np.array_equal(m.design.X_raw[:, 1], (d.zi == 1).astype(float))
    d["zf"] = d["zi"].astype(float)                                             # integral floats: same labels
    m2 = ab.gaussian("y ~ x + zf + (1 | g)", d, G_PRIORS, categorical={"zf": 2.0})
    assert m2.spec.categorical("zf").levels == ("1", "2", "3") and np.array_equal(m2.design.X_raw, m.design.X_raw)
    # pandas Categorical: the declared order is not used (sorted labels), unused categories are reported
    d = df.copy(); d["z"] = pd.Categorical(d["z"], categories=["c", "b", "a", "unused"])
    m = ab.gaussian("y ~ x + z + (1 | g)", d, G_PRIORS, categorical={"z": "b"})
    cat = m.spec.categorical("z")
    assert cat.levels == ("a", "b", "c") and cat.coefficients == ("a", "c") and cat.unused == ("unused",)
    assert m.spec.columns == ("x", "z[a]", "z[c]") and m.design.X_raw.shape[1] == 3
    assert any("['unused']" in w for w in m.warnings) and "unused declared categories ['unused']" in m.describe()
    with pytest.raises(ab.FormulaError, match="does not occur in the fitted rows"):
        ab.gaussian("y ~ x + z + (1 | g)", d, G_PRIORS, categorical={"z": "unused"})
    d["z"] = pd.Categorical(d["z"].astype(str), categories=["c", "b", "a"], ordered=True)
    m3 = ab.gaussian("y ~ x + z + (1 | g)", d, G_PRIORS, categorical={"z": "b"})
    assert m3.spec.categorical("z").levels == ("a", "b", "c") and np.array_equal(m3.design.X_raw, m.design.X_raw)


# -- C2: the design equals an independent construction ------------------------------------------
@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
@pytest.mark.parametrize("terms", [("x", "z"), ("z", "x"), ("z",)])
@pytest.mark.parametrize("reference", LEVELS)
@pytest.mark.parametrize("center", [True, False])
def test_design_matches_independent_construction(family, terms, reference, center):
    df = simulate_cat()
    lhs = {"gaussian": "y", "bernoulli": "y01", "binomial": "k | trials(n)"}[family]
    priors = G_PRIORS if family == "gaussian" else L_PRIORS
    m = getattr(ab, family)(f"{lhs} ~ {' + '.join(terms)} + (1 | g)", df, priors, categorical={"z": reference}, center_predictors=center)
    cat = m.spec.categorical("z")
    others = tuple(l for l in LEVELS if l != reference)
    assert cat.levels == LEVELS and cat.coefficients == others and cat.columns == tuple(f"z[{l}]" for l in others)
    assert list(m.spec.columns) == design_columns(df, list(terms), {"z": reference})
    X_raw, X_model, xbar, gidx = hand_design(df, terms, {"z": reference}, center)
    assert np.array_equal(m.design.X_raw, X_raw) and np.allclose(m.design.X_model, X_model) and np.allclose(m.design.x_means, xbar)
    assert np.array_equal(m.design.group_index, gidx)
    for j, l in enumerate(others):                                     # unequal proportions, exactly
        col = list(m.spec.columns).index(f"z[{l}]")
        assert np.allclose(m.design.x_means[col], (df.z == l).mean() if center else 0.0)
    bterm = m.bambi_model.components[m.names["linpred"]].design.common.terms["z"]
    assert list(bterm.levels) == list(others) and list(bterm.labels) == list(cat.columns)
    assert list(m.pymc_model.coords["z_dim"]) == list(others)
    assert m.coefficients == tuple((c, "z", j) if c.startswith("z[") else (c, c, None)
                                   for j, c in ((list(cat.columns).index(c) if c.startswith("z[") else None, c) for c in m.spec.columns))
    assert m.pymc_model["z"].type.shape == (len(others),) or m.pymc_model["z"].eval().shape == (len(others),)


def test_two_categorical_terms():
    df = simulate_cat()
    df["w"] = np.where(df.index % 2 == 0, "lo", "hi")
    m = ab.gaussian("y ~ z + x + w + (1 | g)", df, G_PRIORS, categorical={"z": "c", "w": "lo"})
    assert m.spec.columns == ("z[a]", "z[b]", "x", "w[hi]") and m.spec.terms == ("z", "x", "w")
    X_raw, _, _, _ = hand_design(df, ("z", "x", "w"), {"z": "c", "w": "lo"}, True)
    assert np.array_equal(m.design.X_raw, X_raw)
    assert list(m.pymc_model.coords["w_dim"]) == ["hi"]


# -- C3: density, Jacobian and gradients against the reference on both paths -----------------------
def _ref_point(model, point):
    b = np.concatenate([np.atleast_1d(point[term]) for term in model.spec.terms]) if model.spec.terms else np.zeros(0)
    out = {"a_c": float(point["Intercept"]), "b": b, "log_sd": float(point[f"{model.names['sd']}_log__"]),
           "z": np.asarray(point[model.names["offset"]])}
    if model.family == "gaussian":
        out["log_sigma"] = float(point["sigma_log__"])
    return out


def _reference(model, df, priors_dict, point, jacobian):
    kw = dict(response=model.spec.response, predictors=list(model.spec.terms), group="g", priors=priors_dict,
              center=model.center_predictors, jacobian=jacobian, categorical={c.name: c.reference for c in model.spec.categoricals})
    if model.family == "gaussian":
        return reference_logp(_ref_point(model, point), df, **kw)
    if model.family == "bernoulli":
        return reference_logp_bernoulli(_ref_point(model, point), df, **kw)
    return reference_logp_binomial(_ref_point(model, point), df, trials="n", **kw)


def _random_nat(model, rng):
    nat = dict(intercept=rng.normal(0.5, 0.7), b=rng.normal(0, 0.6, len(model.spec.columns)),
               sd=np.exp(rng.normal(-0.6, 0.3)), z=rng.normal(size=model.design.J))
    if model.family == "gaussian":
        nat["sigma"] = np.exp(rng.normal(-0.8, 0.3))
    return nat


@pytest.mark.parametrize("family,priors", [("gaussian", G_PRIORS), ("gaussian", G_LEVEL), ("bernoulli", L_LEVEL),
                                           ("binomial", L_PRIORS), ("binomial", L_LEVEL)])
@pytest.mark.parametrize("center", [True, False])
def test_density_jacobian_and_gradients_both_paths(family, priors, center):
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    df = simulate_cat()
    per_level = isinstance(priors.b, dict) and isinstance(priors.b.get("z"), dict)
    m = build(family, df, priors, categorical={"z": "a" if per_level else "b"}, center_predictors=center)   # per-level priors are keyed b, c
    pd_ = ref_priors(priors, m.spec)
    assert set(pd_["b"]) == set(m.spec.columns)
    logp, logp_nojac, dlogp = m.pymc_model.compile_logp(), m.pymc_model.compile_logp(jacobian=False), m.pymc_model.compile_dlogp()
    names = [v.name for v in m.pymc_model.value_vars]
    jf = get_jaxified_logp(m.pymc_model)
    jgrad = jax.grad(lambda *args: jf(list(args)), argnums=tuple(range(len(names))))
    rng = np.random.default_rng(7)
    for _ in range(3):
        nat = _random_nat(m, rng)
        point = m.value_point(**nat)
        ref, ref_nojac = _reference(m, m.data, pd_, point, True), _reference(m, m.data, pd_, point, False)
        assert np.isfinite(ref)
        assert abs(float(logp(point)) - ref) < 1e-8, (family, float(logp(point)), ref)
        assert abs(float(logp_nojac(point)) - ref_nojac) < 1e-8
        assert abs(float(jf([point[n] for n in names])) - ref) < 1e-8
        # gradients: PyMC analytic, JAX analytic, central differences of the reference
        grad_pt = np.asarray(dlogp(point), dtype=float)
        grad_jx = np.concatenate([np.atleast_1d(np.asarray(g, dtype=float)).ravel() for g in jgrad(*[point[n] for n in names])])
        flat = np.concatenate([np.atleast_1d(np.asarray(point[n], dtype=float)).ravel() for n in names])
        def f(vec):
            q, i = {}, 0
            for n_ in names:
                size = int(np.prod(np.shape(point[n_])))
                q[n_] = vec[i:i + size].reshape(np.shape(point[n_])) if np.ndim(point[n_]) else np.asarray(vec[i])
                i += size
            return _reference(m, m.data, pd_, q, True)
        h = 1e-5
        fd = np.array([(f(flat + h * e) - f(flat - h * e)) / (2 * h) for e in np.eye(len(flat))])
        assert np.allclose(grad_pt, fd, rtol=1e-6, atol=1e-5), (family, grad_pt, fd)
        assert np.allclose(grad_jx, grad_pt, rtol=1e-9, atol=1e-9)


# -- C4: what a change of reference level does and does not preserve ------------------------------
def reference_map(levels, ref, new_ref, centered):
    """theta = (intercept, beta over sorted non-reference levels) -> theta' for the new reference,
    from the derivation in docs/m3a1-decision.md: beta'_l = beta_l - beta_new (l not in {ref, new}),
    beta'_ref = -beta_new; centred intercept invariant, raw intercept a' = a + beta_new."""
    old = [l for l in levels if l != ref]
    new = [l for l in levels if l != new_ref]
    M = np.zeros((1 + len(new), 1 + len(old)))
    M[0, 0] = 1.0
    j_new = 1 + old.index(new_ref)
    if not centered:
        M[0, j_new] = 1.0
    for i, l in enumerate(new):
        if l == ref:
            M[1 + i, j_new] = -1.0
        else:
            M[1 + i, 1 + old.index(l)] = 1.0
            M[1 + i, j_new] = -1.0
    return M


@pytest.mark.parametrize("family", ["gaussian", "bernoulli"])
@pytest.mark.parametrize("center", [True, False])
def test_reference_change_design_vs_model_invariance(family, center):
    df = simulate_cat()
    priors = G_PRIORS if family == "gaussian" else L_PRIORS
    ma = build(family, df, priors, categorical={"z": "a"}, center_predictors=center)
    mc = build(family, df, priors, categorical={"z": "c"}, center_predictors=center)
    M = reference_map(LEVELS, "a", "c", center)
    assert abs(abs(np.linalg.det(M)) - 1.0) < 1e-12
    y_name = ma.names["y"]
    lik_a = ma.pymc_model.compile_logp(vars=[ma.pymc_model[y_name]], jacobian=False)
    lik_c = mc.pymc_model.compile_logp(vars=[mc.pymc_model[y_name]], jacobian=False)
    lin_a = ma.pymc_model.compile_fn(ma.pymc_model.replace_rvs_by_values([ma.pymc_model[ma.names["linpred"]]])[0], inputs=ma.pymc_model.value_vars, point_fn=True, on_unused_input="ignore")
    lin_c = mc.pymc_model.compile_fn(mc.pymc_model.replace_rvs_by_values([mc.pymc_model[mc.names["linpred"]]])[0], inputs=mc.pymc_model.value_vars, point_fn=True, on_unused_input="ignore")
    post_a, post_c = ma.pymc_model.compile_logp(), mc.pymc_model.compile_logp()
    coef = priors.coefficient_priors(ma.spec)
    mu_a = np.array([priors.intercept.params["mu"], coef["x"].params["mu"], coef["z[b]"].params["mu"], coef["z[c]"].params["mu"]])
    S_a = np.diag([priors.intercept.params["sigma"], coef["x"].params["sigma"], coef["z[b]"].params["sigma"], coef["z[c]"].params["sigma"]]) ** 2
    Mfull = np.eye(4); Mfull[np.ix_([0, 2, 3], [0, 2, 3])] = M          # (intercept, x, z[b], z[c]) -> (intercept, x, z[a], z[b])
    cov_c = Mfull @ S_a @ Mfull.T
    assert abs(cov_c[2, 3] - coef["z[c]"].params["sigma"] ** 2) < 1e-12   # induced covariance between beta'_a and beta'_b: +s_c^2 (both contain -beta_c)
    if not center:
        assert abs(cov_c[0, 2] + coef["z[c]"].params["sigma"] ** 2) < 1e-12   # intercept coupling: Cov(a', beta'_a) = -s_c^2
    rng = np.random.default_rng(11)
    differ = []
    for _ in range(4):
        nat = _random_nat(ma, rng)
        theta = np.array([nat["intercept"], *nat["b"]])                    # (intercept, x, z[b], z[c])
        theta_c = Mfull @ theta
        pa = ma.value_point(intercept=theta[0], b=theta[1:], sd=nat["sd"], z=nat["z"], sigma=nat.get("sigma"))
        pc = mc.value_point(intercept=theta_c[0], b=theta_c[1:], sd=nat["sd"], z=nat["z"], sigma=nat.get("sigma"))
        assert np.allclose(lin_a(pa), lin_c(pc), atol=1e-12)               # design invariance: same linear predictor ...
        assert abs(float(lik_a(pa)) - float(lik_c(pc))) < 1e-9             # ... and likelihood
        # model non-invariance: independent priors in each coding give different posteriors
        prior_a = stats.multivariate_normal(mu_a, S_a).logpdf(theta)
        prior_c_indep = stats.multivariate_normal(Mfull @ mu_a, S_a).logpdf(theta_c)   # what the c-coded model uses (same marginal priors)
        # sanity: the two coded models' total log densities differ exactly by the prior difference
        assert abs((float(post_c(pc)) - float(post_a(pa))) - (prior_c_indep - prior_a)) < 1e-8
        differ.append(abs(prior_c_indep - prior_a))
        # posterior invariance only under the transformed joint prior (unit Jacobian: no correction)
        prior_c_transformed = stats.multivariate_normal(Mfull @ mu_a, cov_c).logpdf(theta_c)
        assert abs(prior_c_transformed - prior_a) < 1e-9
        assert abs((float(post_c(pc)) - prior_c_indep + prior_c_transformed) - float(post_a(pa))) < 1e-8
    assert max(differ) > 1e-3                                              # the difference is real, not rounding


# -- C5: prior forms -------------------------------------------------------------------------------
def test_prior_forms_and_refusals():
    df = simulate_cat()
    m1 = build("gaussian", df, G_PRIORS, categorical={"z": "a"})
    same = ab.Priors(intercept=ab.Normal(0, 5), b={"x": ab.Normal(0, 2), "z": {"b": ab.Normal(0, 2), "c": ab.Normal(0, 2)}}, sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
    m2 = build("gaussian", df, same, categorical={"z": "a"})
    lp1, lp2 = m1.pymc_model.compile_logp(), m2.pymc_model.compile_logp()
    pt = m1.value_point(intercept=0.3, b=[0.1, -0.4, 0.7], sd=0.5, z=np.zeros(m1.design.J), sigma=0.4)
    assert abs(float(lp1(pt)) - float(lp2(pt))) < 1e-12                    # term-wide == identical per-level priors
    assert m2.priors.coefficient_priors(m2.spec) == {"x": ab.Normal(0, 2), "z[b]": ab.Normal(0, 2), "z[c]": ab.Normal(0, 2)}
    intk = build("gaussian", df.assign(zi=df.z.map({"a": 1, "b": 2, "c": 3})).drop(columns="z").rename(columns={"zi": "z"}),
                 ab.Priors(intercept=ab.Normal(0, 5), b={"x": ab.Normal(0, 2), "z": {2: ab.Normal(0, 1), "3": ab.Normal(0, 2)}}, sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)),
                 categorical={"z": 1})                                     # integer and string keys both match labels
    assert [str(p) for p in intk.priors.coefficient_priors(intk.spec).values()] == ["Normal(mu=0, sigma=2)", "Normal(mu=0, sigma=1)", "Normal(mu=0, sigma=2)"]

    def pri(b):
        return ab.Priors(intercept=ab.Normal(0, 5), b=b, sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
    cases = [
        ({"x": ab.Normal(0, 2)}, "missing \\['z'\\]"),
        ({"x": ab.Normal(0, 2), "z": ab.Normal(0, 2), "w": ab.Normal(0, 1)}, "unexpected \\['w'\\]"),
        ({"x": ab.Normal(0, 2), "z[b]": ab.Normal(0, 2), "z[c]": ab.Normal(0, 2)}, "looks like a design-column label"),
        ({"x": ab.Normal(0, 2), "z": {"b": ab.Normal(0, 1)}}, "missing \\['c'\\]"),
        ({"x": ab.Normal(0, 2), "z": {"a": ab.Normal(0, 1), "b": ab.Normal(0, 1), "c": ab.Normal(0, 1)}}, "reference level and has no coefficient"),
        ({"x": ab.Normal(0, 2), "z": {"b": ab.Normal(0, 1), "c": ab.Normal(0, 1), "d": ab.Normal(0, 1)}}, "unexpected \\['d'\\]"),
        ({"x": ab.Normal(0, 2), "z": {"b": ab.Normal(0, 1), "c": ab.StudentT(3, 0, 1)}}, "share one distribution family"),
        ({"x": ab.Normal(0, 2), "z": {"b": ab.HalfNormal(1), "c": ab.HalfNormal(2)}}, "location prior"),
        ({"x": {"b": ab.Normal(0, 1)}, "z": ab.Normal(0, 1)}, "takes one prior"),
        ({"x": ab.Normal(0, 2), "z": ab.HalfNormal(1)}, "location prior"),
        ({"x": ab.Normal(0, 2), "z": [ab.Normal(0, 1)]}, "takes one prior or a mapping"),
    ]
    for b, msg in cases:
        with pytest.raises(ab.PriorError, match=msg):
            build("gaussian", df, pri(b), categorical={"z": "a"})
    d = df.assign(zi=df.z.map({"a": 1, "b": 2, "c": 3})).drop(columns="z").rename(columns={"zi": "z"})
    with pytest.raises(ab.PriorError, match="ambiguous"):
        build("gaussian", d, pri({"x": ab.Normal(0, 2), "z": {2: ab.Normal(0, 1), "2": ab.Normal(0, 1), 3: ab.Normal(0, 1)}}), categorical={"z": 1})
    with pytest.raises(ab.PriorError, match="only for a categorical term"):
        build("gaussian", df.assign(z=df.x), pri({"x": ab.Normal(0, 2), "z": {"b": ab.Normal(0, 1)}}))


# -- C6: describe() and the fit record ---------------------------------------------------------------
def test_describe_and_fit_record():
    df = simulate_cat()
    priors = ab.Priors(intercept=ab.Normal(0, 1.5), b={"x": ab.Normal(0, 1), "z": {"a": ab.StudentT(4, 0, 1), "c": ab.StudentT(6, 0.2, 2)}},
                       sd=ab.HalfNormal(1))                                     # per-level priors keyed for reference 'b'
    m = build("bernoulli", df, priors, categorical={"z": "b"})
    text = m.describe()
    for s in ["reference 'b'", "levels (sorted) a, b, c", "coefficients b_z[a, c]", "proportions among fitted rows: a 0.",
              "(1[z_i = a] - p_z[a]) * b_z[a]", "difference between level l and the reference 'b' on the log-odds scale",
              "sample proportions of the categorical levels", "NOT the", "average probability across the levels",
              "b_z[a] ~ StudentT(nu=4, mu=0, sigma=1)", "b_z[c] ~ StudentT(nu=6, mu=0.2, sigma=2)",
              "posterior rows z[a], z[c]", "b_x ~ Normal(mu=0, sigma=1)"]:
        assert s in text, s
    mg = build("gaussian", df, G_PRIORS, categorical={"z": "a"}, center_predictors=False)
    assert "raw-scale intercept (x = 0, reference level, group effect zero)" in mg.describe()
    fit = m.fit(backend="pymc", chains=2, draws=60, tune=60, seed=2, cores=1)
    rec = ab.fit_record(m, fit)
    assert rec["model"]["categorical"] == {"z": {"reference": "b", "levels": ["a", "b", "c"], "coefficients": ["a", "c"], "unused_levels": [], "coding": "treatment"}}
    assert rec["model"]["design_columns"] == ["x", "z[a]", "z[c]"] and rec["model"]["terms"] == ["x", "z"]
    assert rec["model"]["priors"]["b:z[a]"] == "StudentT(nu=4, mu=0, sigma=1)" and rec["model"]["priors"]["b:z[c]"] == "StudentT(nu=6, mu=0.2, sigma=2)"
    assert list(rec["posterior"]) == ["Intercept", "x", "z[a]", "z[c]", "1|g_sigma"]
    assert "sample proportions of the categorical levels" in rec["model"]["intercept_prior_applies_to"]
    json.dumps(rec)


# -- C7: predictions per level, mixed rows, unseen levels --------------------------------------------
def _hand_linpred(model, post, df):
    """Raw intercept + coefficients on the raw design + the row's group effect, per draw."""
    from autobayes.predict import raw_design
    a = post["Intercept"].to_numpy()[..., None]
    b = np.concatenate([post[t].to_numpy()[..., None] if t not in model.spec.categorical_names else post[t].to_numpy()
                        for t in model.spec.terms], axis=-1)
    X = raw_design(model, df)
    u = post[model.names["group_effect"]]
    ueff = np.stack([u.sel({u.dims[-1]: lvl}).to_numpy() for lvl in df[model.spec.group].astype(str)], axis=-1)
    return a + np.einsum("cdp,np->cdn", b, X) + ueff


@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
def test_predictions_per_level_mixed_rows_and_unseen_levels(family):
    df = simulate_cat()
    priors = G_PRIORS if family == "gaussian" else L_PRIORS
    m = build(family, df, priors, categorical={"z": "c"})
    fit = m.fit(backend="pymc", chains=2, draws=80, tune=80, seed=3, cores=1)
    post = fit.idata.posterior
    mean_name = "mu" if family == "gaussian" else "p"
    link = (lambda v: v) if family == "gaussian" else expit
    ins = m.predict(fit, kind="mean")
    assert np.allclose(ins[mean_name].to_numpy(), link(_hand_linpred(m, post, m.data)), atol=1e-10)
    new = pd.DataFrame({"x": [0.3, -1.0, 0.0, 2.0, 0.3], "z": ["c", "a", "b", "a", "c"], "g": ["g1", "g0", "g3", "g0", "g1"], "n": [10, 0, 5, 7, 10]})
    out = m.predict(fit, newdata=new, kind="mean", seed=1)
    hand = link(_hand_linpred(m, post, new))
    assert np.allclose(out[mean_name].to_numpy(), hand, atol=1e-10)
    if family == "binomial":
        assert np.allclose(out["expected"].to_numpy(), hand * new.n.to_numpy(), atol=1e-10)
        assert np.allclose(m.predict(fit, newdata=new.drop(columns="n"), kind="probability")["p"].to_numpy(), hand, atol=1e-10)
    assert np.allclose(out[mean_name].to_numpy()[..., 0], out[mean_name].to_numpy()[..., 4])   # identical rows, identical predictions
    dfi = new.assign(z=pd.Categorical(new.z, categories=["b", "a", "c"]))                       # any dtype/order of labels
    assert np.allclose(m.predict(fit, newdata=dfi, kind="mean", seed=1)[mean_name].to_numpy(), hand, atol=1e-10)
    for kind in (["mean", "response"] + (["probability"] if family == "binomial" else [])):
        for ng in ["error", "population", "resample_observed"]:
            with pytest.raises(ab.FormulaError, match="not in the fitted rows: \\['d'\\]; fitted levels \\['a', 'b', 'c'\\]"):
                m.predict(fit, newdata=new.assign(z=["d", "a", "b", "a", "c"]), kind=kind, new_groups=ng, seed=1)
    with pytest.raises(ab.FormulaError, match="categorical predictor 'z'"):                    # a group label is not a category level
        m.predict(fit, newdata=new.assign(z=["g1", "a", "b", "a", "c"]), kind="mean", new_groups="population")
    newg = new.assign(g=["g1", "brand_new", "g3", "brand_new", "g1"])
    with pytest.raises(ab.FormulaError, match="levels of 'g' not seen in training"):
        m.predict(fit, newdata=newg, kind="mean")
    pop = m.predict(fit, newdata=newg, kind="mean", new_groups="population", seed=5)              # unseen GROUP still works
    assert pop.attrs["new_levels"] == ["brand_new"]
    assert np.allclose(pop[mean_name].to_numpy()[..., [0, 2, 4]], hand[..., [0, 2, 4]], atol=1e-10)
    resp = m.predict(fit, newdata=new, kind="response", seed=4)
    assert resp["y"].shape == out[mean_name].shape


# -- C8: the binomial design is built from the fitted rows -----------------------------------------
def test_binomial_zero_trial_rows_determine_levels_and_proportions():
    df = simulate_cat()
    d = df.copy()
    d.loc[d.z == "c", "n"] = 0; d.loc[d.z == "c", "k"] = 0                  # every 'c' row has no trials
    m = ab.binomial("k | trials(n) ~ x + z + (1 | g)", d, L_PRIORS, categorical={"z": "a"})
    assert m.spec.categorical("z").levels == ("a", "b") and m.spec.columns == ("x", "z[b]")
    assert len(m.dropped_rows) == int((df.z == "c").sum()) and any("['c']" in w and "zero-trial" in w for w in m.warnings)
    kept = d[d.n > 0]
    assert np.allclose(m.design.x_means[1], (kept.z == "b").mean()) and abs((kept.z == "b").mean() - (d.z == "b").mean()) > 1e-9
    fit = m.fit(backend="pymc", chains=1, draws=30, tune=30, seed=1, cores=1)
    with pytest.raises(ab.FormulaError, match="not in the fitted rows: \\['c'\\]"):
        m.predict(fit, newdata=pd.DataFrame({"x": [0.0], "z": ["c"], "g": ["g0"], "n": [5]}), kind="mean")
    with pytest.raises(ab.FormulaError, match="occurs only in zero-trial rows"):
        ab.binomial("k | trials(n) ~ x + z + (1 | g)", d, L_PRIORS, categorical={"z": "c"})


# -- C12: value_point and the coefficient map ------------------------------------------------------
def test_value_point_and_coefficient_map():
    df = simulate_cat()
    m = build("gaussian", df, G_PRIORS, categorical={"z": "b"})
    assert m.coefficients == (("x", "x", None), ("z[a]", "z", 0), ("z[c]", "z", 1))
    pt = m.value_point(intercept=1.0, b=[0.5, -0.8, 0.6], sd=0.5, z=np.zeros(m.design.J), sigma=0.4)
    assert pt["x"] == 0.5 and np.array_equal(pt["z"], [-0.8, 0.6]) and set(pt) == {v.name for v in m.pymc_model.value_vars}
    with pytest.raises(ValueError, match="one per design column"):
        m.value_point(intercept=1.0, b=[0.5], sd=0.5, z=np.zeros(m.design.J), sigma=0.4)
    assert m.names["b:z"] == "z" and m.names["b:x"] == "x"
