"""M3a-2: one numeric-by-categorical interaction. Checks D1-D8 and D12 of docs/m3a2-decision.md."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit

import autobayes as ab
from tests.reference_binomial import reference_logp_binomial
from tests.reference_logp import build_design, design_columns, reference_logp, reference_logp_bernoulli
from tests.test_categorical import ref_priors

LEVELS, PROPS = ("a", "b", "c"), (0.5, 0.3, 0.2)
XMEAN = {"a": 0.0, "b": 1.0, "c": -0.5}                     # different within-level x means
BETA = {"a": 0.0, "b": -0.8, "c": 0.6}
DELTA = {"a": 0.0, "b": 0.4, "c": -0.7}
TRUTH = dict(a=1.0, bx=0.5, sd=0.5, sigma=0.4)
G_PRIORS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
G_LEVEL = ab.Priors(intercept=ab.Normal(0, 5), b={"x": ab.Normal(0, 2), "z": {"a": ab.Normal(0, 1), "c": ab.Normal(0.5, 2)},
                                                    "x:z": {"a": ab.Normal(0.1, 1.5), "c": ab.Normal(-0.2, 2.5)}},
                    sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
L_PRIORS = ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
L_LEVEL = ab.Priors(intercept=ab.Normal(0, 1.5), b={"x": ab.Normal(0, 1), "z": ab.StudentT(5, 0, 1),
                                                      "x:z": {"a": ab.StudentT(4, 0, 1), "c": ab.StudentT(6, 0.2, 2)}},
                    sd=ab.HalfNormal(1))
FORMULAS = {"gaussian": "y ~ x + z + x:z + (1 | g)", "bernoulli": "y01 ~ x + z + x:z + (1 | g)",
            "binomial": "k | trials(n) ~ x + z + x:z + (1 | g)"}


def simulate_int(n=120, J=6, seed=7, trials=20):
    """Unequal level proportions, different x means within levels, every level in every group."""
    rng = np.random.default_rng(seed)
    z = rng.choice(LEVELS, size=n, p=PROPS)
    g = np.array([f"g{i % J}" for i in range(n)]); rng.shuffle(g)
    x = rng.normal(size=n) + np.array([XMEAN[v] for v in z])
    u = rng.normal(0, TRUTH["sd"], J)
    eta = (TRUTH["a"] + TRUTH["bx"] * x + np.array([BETA[v] for v in z]) + np.array([DELTA[v] for v in z]) * x
           + u[[int(s[1:]) for s in g]])
    df = pd.DataFrame({"y": eta + rng.normal(0, TRUTH["sigma"], n), "x": x, "z": z, "g": g,
                       "y01": (rng.uniform(size=n) < expit(eta)).astype(int), "k": rng.binomial(trials, expit(eta)), "n": trials})
    cov = pd.crosstab(df.g, df.z)
    assert (cov > 0).all().all() and df.groupby("z").x.mean().round(1).nunique() == 3
    return df


def build(family, df, priors, reference="b", center=True, formula=None):
    return getattr(ab, family)(formula or FORMULAS[family], df, priors, categorical={"z": reference}, center_predictors=center)


def hand(df, mains, categorical, interactions, center):
    _, X_raw, _, gidx, _ = build_design(df, "y", mains, "g", False, categorical, interactions)
    _, Xc, xbar, _, _ = build_design(df, "y", mains, "g", center, categorical, interactions)
    return X_raw, Xc, xbar, gidx


def coef_draws(post, model):
    """(chain, draw, P) draws in design-column order, read straight from the posterior variables."""
    parts = []
    for col, term, j in model.coefficients:
        arr = post[term].to_numpy()
        parts.append(arr if j is None else arr[..., j])
    return np.stack(parts, axis=-1)


def hand_predict(model, post, df):
    """Raw intercept + slope * x + level intercept and slope differences + the row's group effect, per draw,
    written from the model's meaning without any library design helper."""
    a = post["Intercept"].to_numpy()
    bx = post["x"].to_numpy()
    bz = post["z"]; bxz = post["x:z"]
    ref = model.spec.categorical("z").reference
    u = post[model.names["group_effect"]]
    out = []
    for _, row in df.iterrows():
        lvl = str(row["z"])
        eta = a + bx * float(row["x"])
        if lvl != ref:
            eta = eta + bz.sel(z_dim=lvl).to_numpy() + bxz.sel({"x:z_dim": lvl}).to_numpy() * float(row["x"])
        eta = eta + u.sel({u.dims[-1]: str(row["g"])}).to_numpy()
        out.append(eta)
    return np.stack(out, axis=-1)


# -- D1: grammar, canonical expansion, refusals ------------------------------------------------------
def test_canonical_expansion_equivalence_and_refusals():
    df = simulate_int()
    base = build("gaussian", df, G_PRIORS)
    assert base.spec.terms == ("x", "z", "x:z") and base.spec.columns == ("x", "z[a]", "z[c]", "x:z[a]", "x:z[c]")
    assert base.spec.rhs == "x + z + x:z + (1 | g)" and base.spec.canonical_formula == "y ~ x + z + x:z + (1 | g)"
    shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    for formula, terms in [("y ~ x*z + (1 | g)", ("x", "z", "x:z")), ("y ~ z*x + (1 | g)", ("z", "x", "x:z")),
                           ("y ~ z + x + z:x + (1 | g)", ("z", "x", "x:z")), ("y ~ x:z + z + x + (1 | g)", ("z", "x", "x:z")),
                           ("y ~ 1 + x + z + x:z + (1 | g)", ("x", "z", "x:z"))]:
        for data in (df, shuffled):
            m = build("gaussian", data, G_PRIORS, formula=formula)
            assert m.spec.terms == terms and m.spec.interactions == (("x", "z"),) and m.spec.rhs.endswith("x:z + (1 | g)")
            assert set(m.spec.columns) == set(base.spec.columns)
            X_raw, Xc, xbar, gidx = hand(data, list(terms[:2]), {"z": "b"}, [("x", "z")], True)
            assert np.array_equal(m.design.X_raw, X_raw) and np.allclose(m.design.X_model, Xc)
            assert m.bambi_model.formula.main.endswith("x:z + (1 | g)") if hasattr(m.bambi_model, "formula") else True
    with pytest.raises(ab.FormulaError, match="main effect 'z' is missing"):
        build("gaussian", df, G_PRIORS, formula="y ~ x + x:z + (1 | g)")
    with pytest.raises(ab.FormulaError, match="main effect 'x' is missing"):
        build("gaussian", df, G_PRIORS, formula="y ~ z + x:z + (1 | g)")
    with pytest.raises(ab.FormulaError, match="main effect 'x' is missing"):
        build("gaussian", df, G_PRIORS, formula="y ~ x:z + (1 | g)")
    with pytest.raises(ab.FormulaError, match="duplicated predictor after expanding 'x\\*z'"):
        build("gaussian", df, G_PRIORS, formula="y ~ x + z + x*z + (1 | g)")
    with pytest.raises(ab.FormulaError, match="appears more than once after expansion"):
        build("gaussian", df, G_PRIORS, formula="y ~ x + z + x:z + z:x + (1 | g)")
    df_w = df.assign(w=np.linspace(-1, 2, len(df)), v=np.where(df.index % 2 == 0, "p", "q"))
    for formula, cat in [("y ~ x + w + z + x:z + w:z + (1 | g)", {"z": "a"}),                  # shared categorical side
                         ("y ~ x + z + v + x:z + x:v + (1 | g)", {"z": "a", "v": "p"}),          # shared numeric side
                         ("y ~ x + z + w + v + x:z + w:v + (1 | g)", {"z": "a", "v": "p"}),      # disjoint
                         ("y ~ x*z + w*v + (1 | g)", {"z": "a", "v": "p"})]:                     # shorthand twice
        with pytest.raises(ab.FormulaError, match="2 interactions after expansion .* at most one"):
            ab.gaussian(formula, df_w, G_PRIORS, categorical=cat)
    with pytest.raises(ab.FormulaError, match="two numeric terms"):
        ab.gaussian("y ~ x + z + x:z + (1 | g)", df.assign(z=df.x ** 2), G_PRIORS)
    df2 = df.assign(w=np.where(df.index % 2 == 0, "p", "q"))
    with pytest.raises(ab.FormulaError, match="two categorical terms"):
        ab.gaussian("y ~ z + w + z:w + (1 | g)", df2, G_PRIORS, categorical={"z": "a", "w": "p"})
    with pytest.raises(ab.FormulaError, match="grouping factor 'g' cannot enter"):
        build("gaussian", df, G_PRIORS, formula="y ~ x + z + x:g + (1 | g)")
    with pytest.raises(ab.FormulaError, match="cannot interact with itself"):
        build("gaussian", df, G_PRIORS, formula="y ~ x + z + x:x + (1 | g)")
    with pytest.raises(ab.FormulaError, match="only a two-way interaction"):
        build("gaussian", df, G_PRIORS, formula="y ~ x + z + x:z:g + (1 | g)")
    with pytest.raises(ab.FormulaError, match="mixing '\\*' and ':'"):
        build("gaussian", df, G_PRIORS, formula="y ~ x + z + x*z:g + (1 | g)")
    with pytest.raises(ab.PriorError, match="no population intercept"):              # removal now supported; old intercept prior refused
        build("gaussian", df, G_PRIORS, formula="y ~ x + z + x:z + (1 | g) - 1")


# -- D2: the design equals an independent construction; the reported intercept ----------------------
@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
@pytest.mark.parametrize("reference", ["b", "c"])
@pytest.mark.parametrize("center", [True, False])
def test_design_matches_independent_construction(family, reference, center):
    df = simulate_int().sample(frac=1.0, random_state=5).reset_index(drop=True)
    priors = G_PRIORS if family == "gaussian" else L_PRIORS
    m = build(family, df, priors, reference=reference, center=center)
    others = [l for l in LEVELS if l != reference]
    assert m.spec.columns == ("x", *[f"z[{l}]" for l in others], *[f"x:z[{l}]" for l in others])
    assert list(m.spec.columns) == design_columns(df, ["x", "z"], {"z": reference}, [("x", "z")])
    X_raw, Xc, xbar, gidx = hand(df, ["x", "z"], {"z": reference}, [("x", "z")], center)
    assert np.array_equal(m.design.X_raw, X_raw) and np.allclose(m.design.X_model, Xc) and np.allclose(m.design.x_means, xbar)
    assert np.array_equal(m.design.group_index, gidx)
    lab = df.z.to_numpy(); xv = df.x.to_numpy()
    for j, l in enumerate(others):                                     # product means = p_l * mean(x | l), not mean(x) * p_l
        q = m.design.x_means[3 + j]
        if center:
            assert np.isclose(q, (lab == l).mean() * xv[lab == l].mean()) and abs(q - xv.mean() * (lab == l).mean()) > 0.02
        else:
            assert q == 0.0
    comp = m.bambi_model.components[m.names["linpred"]].design.common
    assert comp.terms["x:z"].kind == "interaction" and list(comp.terms["x:z"].levels) == others
    assert list(comp.terms["x:z"].labels) == [f"x:z[{l}]" for l in others]
    assert list(m.pymc_model.coords["x:z_dim"]) == others and list(m.pymc_model.coords["z_dim"]) == others
    assert m.coefficients == (("x", "x", None), *[(f"z[{l}]", "z", j) for j, l in enumerate(others)],
                              *[(f"x:z[{l}]", "x:z", j) for j, l in enumerate(others)])


def test_reported_intercept_is_the_sampled_one_minus_expanded_column_means():
    """Bambi reports the raw intercept; recover the sampled a_c from it with the ACTUAL expanded-column
    means and check the model's log density at that point against the sampler's own lp."""
    df = simulate_int()
    m = build("gaussian", df, G_PRIORS, reference="b")
    fit = m.fit(backend="pymc", chains=1, draws=60, tune=60, seed=4, cores=1, discard_tuned_samples=True)
    post = fit.idata.posterior; stats_ = fit.idata.sample_stats
    logp = m.pymc_model.compile_logp()
    B = coef_draws(post, m)
    a_c = post["Intercept"].to_numpy() + B @ m.design.x_means          # a_c = a + xbar b + sum p beta + sum q delta
    a_c_wrong = post["Intercept"].to_numpy() + B @ np.concatenate([m.design.x_means[:3], m.design.x_means[0] * m.design.x_means[1:3]])
    ok = wrong = 0
    for i in range(0, 60, 6):
        z_off = (post["1|g"][0, i] / post["1|g_sigma"][0, i]).to_numpy()            # Bambi reports u = sd * z, not z
        pt = m.value_point(intercept=float(a_c[0, i]), b=B[0, i], sd=float(post["1|g_sigma"][0, i]), z=z_off, sigma=float(post["sigma"][0, i]))
        lp = float(logp(pt)); lp_stat = float(stats_["lp"][0, i])
        ok += abs(lp - lp_stat) < 1e-6
        pt_w = dict(pt); pt_w["Intercept"] = np.asarray(float(a_c_wrong[0, i]))
        wrong += abs(float(logp(pt_w)) - lp_stat) < 1e-6
    assert ok == 10 and wrong == 0


# -- D3: density, Jacobian and gradients on both paths ---------------------------------------------
def _ref_point(model, point):
    b = np.concatenate([np.atleast_1d(point[term]) for term in model.spec.terms])
    out = {"a_c": float(point["Intercept"]), "b": b, "log_sd": float(point[f"{model.names['sd']}_log__"]), "z": np.asarray(point[model.names["offset"]])}
    if model.family == "gaussian":
        out["log_sigma"] = float(point["sigma_log__"])
    return out


def _reference(model, df, pd_, point, jacobian):
    kw = dict(response=model.spec.response, predictors=[t for t in model.spec.terms if ":" not in t], group="g", priors=pd_,
              center=model.center_predictors, jacobian=jacobian, categorical={c.name: c.reference for c in model.spec.categoricals},
              interactions=list(model.spec.interactions))
    if model.family == "gaussian":
        return reference_logp(_ref_point(model, point), df, **kw)
    if model.family == "bernoulli":
        return reference_logp_bernoulli(_ref_point(model, point), df, **kw)
    return reference_logp_binomial(_ref_point(model, point), df, trials="n", **kw)


def _random_nat(model, rng):
    nat = dict(intercept=rng.normal(0.5, 0.7), b=rng.normal(0, 0.5, len(model.spec.columns)), sd=np.exp(rng.normal(-0.6, 0.3)), z=rng.normal(size=model.design.J))
    if model.family == "gaussian":
        nat["sigma"] = np.exp(rng.normal(-0.8, 0.3))
    return nat


@pytest.mark.parametrize("family,priors", [("gaussian", G_PRIORS), ("gaussian", G_LEVEL), ("bernoulli", L_LEVEL), ("binomial", L_PRIORS), ("binomial", L_LEVEL)])
@pytest.mark.parametrize("center", [True, False])
def test_density_jacobian_and_gradients_both_paths(family, priors, center):
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    df = simulate_int()
    if family == "binomial":                                 # the ORIGINAL frame decides what the reference sees
        df = df.copy(); df.loc[df.index[[4, 17, 30]], ["n", "k"]] = 0            # three zero-trial rows the model must drop
    m = build(family, df, priors, reference="b", center=center)
    fitted = df[df.n > 0].reset_index(drop=True) if family == "binomial" else df    # selected here, not from model.data
    assert len(fitted) == m.design.n
    pd_ = ref_priors(priors, m.spec)
    assert set(pd_["b"]) == set(m.spec.columns)
    logp, logp_nojac, dlogp = m.pymc_model.compile_logp(), m.pymc_model.compile_logp(jacobian=False), m.pymc_model.compile_dlogp()
    names = [v.name for v in m.pymc_model.value_vars]
    jf = get_jaxified_logp(m.pymc_model)
    jgrad = jax.grad(lambda *args: jf(list(args)), argnums=tuple(range(len(names))))
    rng = np.random.default_rng(9)
    for _ in range(3):
        nat = _random_nat(m, rng)
        point = m.value_point(**nat)
        ref, ref_nojac = _reference(m, fitted, pd_, point, True), _reference(m, fitted, pd_, point, False)
        assert np.isfinite(ref)
        assert abs(float(logp(point)) - ref) < 1e-8, (family, float(logp(point)), ref)
        assert abs(float(logp_nojac(point)) - ref_nojac) < 1e-8
        assert abs(float(jf([point[n] for n in names])) - ref) < 1e-8
        grad_pt = np.asarray(dlogp(point), dtype=float)
        grad_jx = np.concatenate([np.atleast_1d(np.asarray(g, dtype=float)).ravel() for g in jgrad(*[point[n] for n in names])])
        flat = np.concatenate([np.atleast_1d(np.asarray(point[n], dtype=float)).ravel() for n in names])
        def f(vec):
            q, i = {}, 0
            for n_ in names:
                size = int(np.prod(np.shape(point[n_])))
                q[n_] = vec[i:i + size].reshape(np.shape(point[n_])) if np.ndim(point[n_]) else np.asarray(vec[i]); i += size
            return _reference(m, fitted, pd_, q, True)
        h = 1e-5
        fd = np.array([(f(flat + h * e) - f(flat - h * e)) / (2 * h) for e in np.eye(len(flat))])
        assert np.allclose(grad_pt, fd, rtol=1e-6, atol=1e-5), (family, grad_pt, fd)
        assert np.allclose(grad_jx, grad_pt, rtol=1e-9, atol=1e-9)


# -- D4: change of reference with the interaction ---------------------------------------------------
def reference_map_interaction(levels, ref, new_ref, centered):
    """theta = (intercept, b, beta over old non-ref levels, delta over old non-ref levels) -> theta' for the
    new reference (docs/m3a2-decision.md): beta'_l = beta_l - beta_new, delta'_l = delta_l - delta_new,
    beta'_ref = -beta_new, delta'_ref = -delta_new, b' = b + delta_new, a' = a + beta_new (raw) or a'_c = a_c."""
    old = [l for l in levels if l != ref]; new = [l for l in levels if l != new_ref]
    K = len(old); n = 2 + 2 * K
    M = np.zeros((n, n)); M[0, 0] = 1.0; M[1, 1] = 1.0
    jb, jd = 2 + old.index(new_ref), 2 + K + old.index(new_ref)
    if not centered:
        M[0, jb] = 1.0
    M[1, jd] = 1.0
    for i, l in enumerate(new):
        rb, rd = 2 + i, 2 + K + i
        if l == ref:
            M[rb, jb] = -1.0; M[rd, jd] = -1.0
        else:
            M[rb, 2 + old.index(l)] = 1.0; M[rb, jb] = -1.0
            M[rd, 2 + K + old.index(l)] = 1.0; M[rd, jd] = -1.0
    return M


@pytest.mark.parametrize("family", ["gaussian", "bernoulli"])
@pytest.mark.parametrize("center", [True, False])
def test_reference_change_with_interaction(family, center):
    df = simulate_int()
    priors = G_PRIORS if family == "gaussian" else L_PRIORS
    ma = build(family, df, priors, reference="a", center=center)
    mc = build(family, df, priors, reference="c", center=center)
    M = reference_map_interaction(LEVELS, "a", "c", center)
    assert abs(abs(np.linalg.det(M)) - 1.0) < 1e-12
    y = ma.names["y"]
    lik_a = ma.pymc_model.compile_logp(vars=[ma.pymc_model[y]], jacobian=False); lik_c = mc.pymc_model.compile_logp(vars=[mc.pymc_model[y]], jacobian=False)
    lin = {}
    for tag, mm in (("a", ma), ("c", mc)):
        v = mm.pymc_model.replace_rvs_by_values([mm.pymc_model[mm.names["linpred"]]])[0]
        lin[tag] = mm.pymc_model.compile_fn(v, inputs=mm.pymc_model.value_vars, point_fn=True, on_unused_input="ignore")
    post_a, post_c = ma.pymc_model.compile_logp(), mc.pymc_model.compile_logp()
    coef = priors.coefficient_priors(ma.spec)                                   # (intercept, x, z[b], z[c], x:z[b], x:z[c])
    mu = np.array([priors.intercept.params["mu"]] + [coef[c].params["mu"] for c in ma.spec.columns])
    S = np.diag([priors.intercept.params["sigma"]] + [coef[c].params["sigma"] for c in ma.spec.columns]) ** 2
    cov_c = M @ S @ M.T
    sb, sd_ = coef["z[c]"].params["sigma"] ** 2, coef["x:z[c]"].params["sigma"] ** 2
    assert np.isclose(cov_c[2, 3], sb) and np.isclose(cov_c[4, 5], sd_) and np.isclose(cov_c[1, 4], -sd_)   # induced covariances
    assert np.isclose(cov_c[0, 2], 0.0 if center else -sb)
    # centred intercept invariance derived from the actual expanded-column means: a_c = sum_l p_l mu_l + sum_l q_l s_l
    lab = df.z.to_numpy(); xv = df.x.to_numpy()
    rng = np.random.default_rng(13); differ = []
    for _ in range(4):
        nat = _random_nat(ma, rng)
        theta = np.array([nat["intercept"], *nat["b"]]); theta_c = M @ theta
        if center:
            a_raw = theta[0] - ma.design.x_means @ theta[1:]
            mu_l = {"a": a_raw, "b": a_raw + theta[2], "c": a_raw + theta[3]}; s_l = {"a": theta[1], "b": theta[1] + theta[4], "c": theta[1] + theta[5]}
            a_c_from_levels = sum((lab == l).mean() * mu_l[l] + (xv * (lab == l)).mean() * s_l[l] for l in LEVELS)
            assert np.isclose(a_c_from_levels, theta[0]) and np.isclose(theta_c[0], theta[0])
        pa = ma.value_point(intercept=theta[0], b=theta[1:], sd=nat["sd"], z=nat["z"], sigma=nat.get("sigma"))
        pc = mc.value_point(intercept=theta_c[0], b=theta_c[1:], sd=nat["sd"], z=nat["z"], sigma=nat.get("sigma"))
        assert np.allclose(lin["a"](pa), lin["c"](pc), atol=1e-10)                  # design invariance
        assert abs(float(lik_a(pa)) - float(lik_c(pc))) < 1e-8
        prior_a = stats.multivariate_normal(mu, S).logpdf(theta)
        prior_c_indep = stats.multivariate_normal(M @ mu, S).logpdf(theta_c)         # what the c-coded model with independent priors uses
        assert abs((float(post_c(pc)) - float(post_a(pa))) - (prior_c_indep - prior_a)) < 1e-8
        differ.append(abs(prior_c_indep - prior_a))
        prior_c_joint = stats.multivariate_normal(M @ mu, cov_c).logpdf(theta_c)      # transformed joint prior, unit Jacobian
        assert abs(prior_c_joint - prior_a) < 1e-9
        assert abs((float(post_c(pc)) - prior_c_indep + prior_c_joint) - float(post_a(pa))) < 1e-8
    assert max(differ) > 1e-3


# -- D5: prior forms ---------------------------------------------------------------------------------
def test_prior_forms_and_refusals():
    df = simulate_int()
    m1 = build("gaussian", df, G_PRIORS)
    same = ab.Priors(intercept=ab.Normal(0, 5), b={"x": ab.Normal(0, 2), "z": ab.Normal(0, 2), "x:z": {"a": ab.Normal(0, 2), "c": ab.Normal(0, 2)}}, sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
    m2 = build("gaussian", df, same)
    pt = m1.value_point(intercept=0.3, b=[0.1, -0.4, 0.7, 0.2, -0.3], sd=0.5, z=np.zeros(m1.design.J), sigma=0.4)
    assert abs(float(m1.pymc_model.compile_logp()(pt)) - float(m2.pymc_model.compile_logp()(pt))) < 1e-12
    assert list(G_LEVEL.coefficient_priors(build("gaussian", df, G_LEVEL).spec)) == ["x", "z[a]", "z[c]", "x:z[a]", "x:z[c]"]

    def pri(b):
        return ab.Priors(intercept=ab.Normal(0, 5), b=b, sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
    for b, msg in [({"x": ab.Normal(0, 2), "z": ab.Normal(0, 2)}, "missing \\['x:z'\\]"),
                   ({"x": ab.Normal(0, 2), "z": ab.Normal(0, 2), "z:x": ab.Normal(0, 2)}, "canonical spelling 'x:z'"),
                   ({"x": ab.Normal(0, 2), "z": ab.Normal(0, 2), "x:z": ab.Normal(0, 2), "x:z[a]": ab.Normal(0, 1)}, "design-column label"),
                   ({"x": ab.Normal(0, 2), "z": ab.Normal(0, 2), "x:z": {"a": ab.Normal(0, 1)}}, "missing \\['c'\\]"),
                   ({"x": ab.Normal(0, 2), "z": ab.Normal(0, 2), "x:z": {"a": ab.Normal(0, 1), "b": ab.Normal(0, 1), "c": ab.Normal(0, 1)}}, "reference level and has no coefficient"),
                   ({"x": ab.Normal(0, 2), "z": ab.Normal(0, 2), "x:z": {"a": ab.Normal(0, 1), "c": ab.StudentT(3, 0, 1)}}, "share one distribution family"),
                   ({"x": ab.Normal(0, 2), "z": ab.Normal(0, 2), "x:z": ab.HalfNormal(1)}, "location prior"),
                   ({"x": {"a": ab.Normal(0, 1)}, "z": ab.Normal(0, 2), "x:z": ab.Normal(0, 2)}, "takes one prior")]:
        with pytest.raises(ab.PriorError, match=msg):
            build("gaussian", df, pri(b))


# -- D6: describe() and the fit record ---------------------------------------------------------------
def test_describe_and_fit_record():
    df = simulate_int()
    m = build("bernoulli", df, L_LEVEL, reference="b")
    text = m.describe()
    for s in ["x:z: the reference level 'b' has the raw intercept a and slope b_x", "slope difference b_x:z[l]", "slope in x is b_x + b_x:z[l]",
              "b_z[l] is the difference between level l and the reference 'b' on the log-odds scale at x = 0 (same group)",
              "at another value of x the difference is b_z[l] + x * b_x:z[l]",
              "(x_x * 1[z_i = a] - q_x:z[a]) * b_x:z[a]", "centred at its OWN mean q", "not mean(x)", "not a realisable covariate row",
              "fitted-row means of every expanded design column", "NOT the", "average probability across the levels",
              "b_x:z[a] ~ StudentT(nu=4, mu=0, sigma=1)", "b_x:z[c] ~ StudentT(nu=6, mu=0.2, sigma=2)", "posterior rows x:z[a], x:z[c]"]:
        assert s in text, s
    m2 = build("gaussian", df, G_PRIORS, formula="y ~ z*x + (1 | g)")
    assert "expanded  : y ~ z + x + x:z + (1 | g)" in m2.describe()
    fit = m.fit(backend="pymc", chains=1, draws=40, tune=40, seed=2, cores=1)
    rec = ab.fit_record(m, fit)
    assert rec["model"]["interactions"] == [["x", "z"]] and rec["model"]["canonical_formula"] == "y01 ~ x + z + x:z + (1 | g)"
    assert rec["model"]["design_columns"] == ["x", "z[a]", "z[c]", "x:z[a]", "x:z[c]"]
    assert rec["model"]["priors"]["b:x:z[c]"] == "StudentT(nu=6, mu=0.2, sigma=2)"
    assert list(rec["posterior"]) == ["Intercept", "x", "z[a]", "z[c]", "x:z[a]", "x:z[c]", "1|g_sigma"]
    json.dumps(rec)


# -- D7: predictions by hand ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "binomial"])
def test_predictions_slopes_per_level_mixed_rows_and_unseen(family):
    df = simulate_int()
    priors = G_PRIORS if family == "gaussian" else L_PRIORS
    m = build(family, df, priors, reference="c")
    fit = m.fit(backend="pymc", chains=2, draws=80, tune=80, seed=3, cores=1)
    post = fit.idata.posterior
    mean_name = "mu" if family == "gaussian" else "p"
    link = (lambda v: v) if family == "gaussian" else expit
    ins = m.predict(fit, kind="mean")
    assert np.allclose(ins[mean_name].to_numpy(), link(hand_predict(m, post, m.data)), atol=1e-10)
    new = pd.DataFrame({"x": [0.3, 1.3, -1.0, 0.0, 2.0, 0.3], "z": ["c", "c", "a", "b", "a", "c"], "g": ["g1", "g1", "g0", "g3", "g0", "g1"], "n": [10, 10, 0, 5, 7, 10]})
    out = m.predict(fit, newdata=new, kind="mean", seed=1)
    eta = hand_predict(m, post, new)
    assert np.allclose(out[mean_name].to_numpy(), link(eta), atol=1e-10)
    # slopes: rows 0 and 1 differ by one unit of x in the reference level -> b_x; a non-reference level -> b_x + delta
    assert np.allclose(eta[..., 1] - eta[..., 0], post["x"].to_numpy(), atol=1e-10)
    new2 = pd.DataFrame({"x": [0.0, 1.0], "z": ["a", "a"], "g": ["g0", "g0"], "n": [5, 5]})
    eta2 = hand_predict(m, post, new2)
    slope_a = post["x"].to_numpy() + post["x:z"].sel({"x:z_dim": "a"}).to_numpy()
    assert np.allclose(eta2[..., 1] - eta2[..., 0], slope_a, atol=1e-10)
    lin2 = m.predict(fit, newdata=new2, kind="mean")[mean_name].to_numpy()
    assert np.allclose(lin2, link(eta2), atol=1e-10)
    if family == "binomial":
        assert np.allclose(out["expected"].to_numpy(), link(eta) * new.n.to_numpy(), atol=1e-10)
        resp = m.predict(fit, newdata=new, kind="response", seed=4)
        assert resp["y"].dtype.kind == "i" and int(resp["y"][..., 2].max()) == 0 and float(resp["expected"][..., 2].max()) == 0.0
        assert np.all(resp["y"].to_numpy() <= new.n.to_numpy())
        assert np.allclose(m.predict(fit, newdata=new.drop(columns="n"), kind="probability")["p"].to_numpy(), link(eta), atol=1e-10)
    else:
        resp = m.predict(fit, newdata=new, kind="response", seed=4)
        assert resp["y"].shape == out[mean_name].shape
    for kind in (["mean", "response"] + (["probability"] if family == "binomial" else [])):
        for ng in ["error", "population", "resample_observed"]:
            with pytest.raises(ab.FormulaError, match="not in the fitted rows: \\['d'\\]"):
                m.predict(fit, newdata=new.assign(z=["d", "c", "a", "b", "a", "c"]), kind=kind, new_groups=ng, seed=1)
    newg = new.assign(g=["g1", "brand_new", "g0", "g3", "g0", "g1"])
    with pytest.raises(ab.FormulaError, match="levels of 'g' not seen in training"):
        m.predict(fit, newdata=newg, kind="mean")
    pop = m.predict(fit, newdata=newg, kind="mean", new_groups="population", seed=5)
    assert pop.attrs["new_levels"] == ["brand_new"] and np.allclose(pop[mean_name].to_numpy()[..., [0, 2, 3, 4, 5]], link(eta)[..., [0, 2, 3, 4, 5]], atol=1e-10)


# -- D8: binomial zero-trial rows are dropped before means and levels ----------------------------------
def test_binomial_zero_trial_rows_removed_before_means_and_levels():
    df = simulate_int()
    d = df.copy()
    drop = d.index[(d.z == "a") & (d.x > 0.5)]
    d.loc[drop, ["n", "k"]] = 0
    m = ab.binomial("k | trials(n) ~ x + z + x:z + (1 | g)", d, L_PRIORS, categorical={"z": "b"})
    kept = d[d.n > 0]
    assert len(m.dropped_rows) == len(drop) and m.design.n == len(kept)
    X_raw, Xc, xbar, _ = hand(kept.reset_index(drop=True), ["x", "z"], {"z": "b"}, [("x", "z")], True)
    assert np.array_equal(m.design.X_raw, X_raw) and np.allclose(m.design.x_means, xbar)
    _, _, xbar_all, _ = hand(d, ["x", "z"], {"z": "b"}, [("x", "z")], True)
    assert not np.allclose(xbar, xbar_all)                                          # the drop changes the means
    d2 = df.copy(); d2.loc[d2.z == "c", ["n", "k"]] = 0
    m2 = ab.binomial("k | trials(n) ~ x + z + x:z + (1 | g)", d2, L_PRIORS, categorical={"z": "a"})
    assert m2.spec.columns == ("x", "z[b]", "x:z[b]") and any("['c']" in w for w in m2.warnings)


# -- D12: value_point and the coefficient map -----------------------------------------------------------
def test_value_point_and_coefficient_map():
    df = simulate_int()
    m = build("gaussian", df, G_PRIORS, reference="b")
    assert m.coefficients == (("x", "x", None), ("z[a]", "z", 0), ("z[c]", "z", 1), ("x:z[a]", "x:z", 0), ("x:z[c]", "x:z", 1))
    pt = m.value_point(intercept=1.0, b=[0.5, -0.8, 0.6, 0.4, -0.7], sd=0.5, z=np.zeros(m.design.J), sigma=0.4)
    assert pt["x"] == 0.5 and np.array_equal(pt["z"], [-0.8, 0.6]) and np.array_equal(pt["x:z"], [0.4, -0.7])
    assert set(pt) == {v.name for v in m.pymc_model.value_vars}
    assert m.original_labels["z"].tolist() == df.z.tolist() and m.names["b:x:z"] == "x:z"
