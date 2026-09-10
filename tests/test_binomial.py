"""M2: grouped binomial counts with a logit link. Checks A1-A9 and C1 of docs/m2-decision.md, and N1-N5 of its correction record."""
import copy
import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import autobayes as ab
from autobayes.formula import COUNT_CAP
from tests.reference_binomial import (EXACT_UPTO, lgamma_dec, log_binom_dec, reference_grad_binomial,
                                      reference_logp_binomial, row_grad_dec, row_loglik_dec, stirling_remainder_bound)
from tests.reference_logp import reference_logp_bernoulli

PRIORS = ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
PRIORS_DICT = {"intercept": ("Normal", {"mu": 0.0, "sigma": 1.5}),
               "b": {"x": ("Normal", {"mu": 0.0, "sigma": 1.0}), "w": ("Normal", {"mu": 0.0, "sigma": 1.0})},
               "sd": ("HalfNormal", {"sigma": 1.0})}
FORMULA = "k | trials(n) ~ x + (1 | g)"
ETAS = [0.0, 0.5, -0.5, 12.0, -12.0, 40.0, -40.0, 300.0, -300.0, 1000.0, -1000.0, 1e6, -1e6]
GRID_N = [1, 2, 10, 10**3, 10**5, COUNT_CAP - 1, COUNT_CAP]          # acceptance grid (docs/m2-decision.md, repair record)
MODE_OFFSETS = [0.0, 1e-3, -1e-3, 0.1, -0.1, 1.0, -1.0]


def grid_ks(n):
    ks = sorted({0, 1, 2, 10, n // 4, n // 2, (3 * n) // 4, n - 2, n - 1, n})
    return [k for k in ks if 0 <= k <= n]


def etas_for(n, k):
    out = list(ETAS)
    if 0 < k < n:
        mode = math.log(k) - math.log(n - k)
        out += [mode + d for d in MODE_OFFSETS]
    return out


def tol(ref: float) -> float:
    return max(1e-8, 1e-12 * abs(ref))


def simulate_counts(rows=120, J=6, seed=0, a=-0.5, b=0.8, sd=0.7, extra=False, lo=5, hi=50, equal_trials=None):
    rng = np.random.default_rng(seed)
    g = np.array([i % J for i in range(rows)]); rng.shuffle(g)          # every group present
    x = rng.normal(size=rows)
    u = rng.normal(0, sd, J)
    eta = a + b * x + u[g]
    df = pd.DataFrame({"x": x, "g": [f"lvl{j:02d}" for j in g]})
    if extra:
        df["w"] = rng.normal(size=rows)
        eta = eta - 0.5 * df["w"].to_numpy()
    n = np.full(rows, equal_trials) if equal_trials else rng.integers(lo, hi + 1, rows)
    df["n"] = n
    df["k"] = rng.binomial(n, expit(eta))
    return df


@pytest.fixture(scope="module")
def cdata():
    return simulate_counts()


@pytest.fixture(scope="module")
def cmodel(cdata):
    return ab.binomial(FORMULA, cdata, PRIORS)


@pytest.fixture(scope="module")
def cfit(cmodel):
    return cmodel.fit(backend="pymc", chains=2, draws=200, tune=200, seed=1, cores=1)


def _random_points(model, k=5, seed=3):
    rng = np.random.default_rng(seed)
    return [dict(intercept=rng.normal(0, 1), b=rng.normal(0, 0.7, len(model.spec.predictors)),
                 sd=np.exp(rng.normal(-0.5, 0.4)), z=rng.normal(size=model.design.J)) for _ in range(k)]


def _ref_point(nat):
    return {"a_c": nat["intercept"], "b": nat["b"], "log_sd": np.log(nat["sd"]), "z": nat["z"]}


def _eta(model, nat):
    d = model.design
    return nat["intercept"] + d.X_model @ nat["b"] + nat["sd"] * nat["z"][d.group_index]


# -- the reference itself ------------------------------------------------------------------------
def test_reference_lgamma_branches_agree_and_remainder_is_bounded():
    for z in (EXACT_UPTO + 1, EXACT_UPTO + 7):
        exact = sum((float(math.log(i)) for i in range(2, z)), 0.0)       # float64 sum, ~1e-13 relative
        assert abs(float(lgamma_dec(z)) - exact) < 1e-9 * exact
    assert stirling_remainder_bound(EXACT_UPTO + 1) < 1e-45
    assert float(log_binom_dec(10, 3)) == pytest.approx(math.log(120), rel=1e-15)
    assert float(log_binom_dec(50, 25)) == pytest.approx(math.log(math.comb(50, 25)), rel=1e-14)
    assert float(row_grad_dec(3, 10, 0.0)) == pytest.approx(3 * 0.5 - 7 * 0.5)


# -- A1, A2 --------------------------------------------------------------------------------------
@pytest.mark.parametrize("formula,extra,center", [
    (FORMULA, False, True), (FORMULA, False, False),
    ("k | trials(n) ~ x + w + (1 | g)", True, True), ("k | trials(n) ~ 1 + (1 | g)", False, True),
])
def test_logp_matches_reference(formula, extra, center):
    df = simulate_counts(extra=extra)
    m = ab.binomial(formula, df, PRIORS, center_predictors=center)
    preds = list(m.spec.predictors)
    logp = m.pymc_model.compile_logp()
    logp_nojac = m.pymc_model.compile_logp(jacobian=False)
    kw = dict(response="k", trials="n", predictors=preds, group="g", priors=PRIORS_DICT, center=center)
    for nat in _random_points(m):
        assert np.max(np.abs(_eta(m, nat))) < 15
        point = m.value_point(**nat)
        ref = reference_logp_binomial(_ref_point(nat), m.data, jacobian=True, **kw)
        ref_nojac = reference_logp_binomial(_ref_point(nat), m.data, jacobian=False, **kw)
        assert abs(float(logp(point)) - ref) < tol(ref), (float(logp(point)), ref)
        assert abs(float(logp_nojac(point)) - ref_nojac) < tol(ref_nojac)
        assert abs(ref - ref_nojac - np.log(nat["sd"])) < 1e-12


# -- A3, A4 --------------------------------------------------------------------------------------
def test_gradient_matches_finite_differences_and_analytic(cmodel):
    dlogp = cmodel.pymc_model.compile_dlogp()
    names = [v.name for v in cmodel.pymc_model.value_vars]
    kw = dict(response="k", trials="n", predictors=["x"], group="g", priors=PRIORS_DICT, center=True)
    for nat in _random_points(cmodel, k=3):
        point = cmodel.value_point(**nat)
        grad = np.asarray(dlogp(point), dtype=float)
        sizes = [int(np.asarray(point[n_]).size) for n_ in names]

        def f(vec):
            q, i = {}, 0
            for n_, s in zip(names, sizes):
                q[n_] = vec[i:i + s] if s > 1 else float(vec[i])
                i += s
            nat2 = dict(intercept=q["Intercept"], b=[q["x"]], sd=np.exp(q["1|g_sigma_log__"]), z=np.asarray(q["1|g_offset"]))
            return reference_logp_binomial(_ref_point(nat2), cmodel.data, jacobian=True, **kw)
        vec = np.concatenate([np.atleast_1d(np.asarray(point[n_], dtype=float)).ravel() for n_ in names])
        fd = np.zeros_like(vec)
        for i in range(len(vec)):
            e = np.zeros_like(vec); e[i] = 1e-5
            fd[i] = (f(vec + e) - f(vec - e)) / 2e-5
        assert np.allclose(grad, fd, rtol=1e-5, atol=1e-6), (grad, fd)
        an = reference_grad_binomial(_ref_point(nat), cmodel.data, response="k", trials="n", predictors=["x"],
                                     group="g", prior_intercept_sigma=1.5, prior_b_sigma=1.0)
        an_vec = np.concatenate([np.atleast_1d(an[n_]) for n_ in names])
        assert np.all(np.abs(grad - an_vec) <= np.maximum(1e-8, 1e-12 * np.abs(an_vec)))


def test_jax_logp_matches_pytensor(cmodel):
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    jf = get_jaxified_logp(cmodel.pymc_model)
    logp = cmodel.pymc_model.compile_logp()
    names = [v.name for v in cmodel.pymc_model.value_vars]
    for nat in _random_points(cmodel, k=3):
        point = cmodel.value_point(**nat)
        a, b = float(jf([point[n_] for n_ in names])), float(logp(point))
        assert abs(a - b) < tol(b)


# -- A5: the validation domain grid, both paths, values and full gradients ---------------------------
def grid_model(n):
    """One row per grid k, each row its own group, so per-row terms and derivatives are visible."""
    ks = grid_ks(n)
    rows = len(ks)
    df = pd.DataFrame({"k": ks, "n": [n] * rows, "x": np.linspace(-0.3, 0.3, rows),
                       "g": [f"g{i:02d}" for i in range(rows)]})
    return ab.binomial(FORMULA, df, PRIORS)


def per_row_functions(m):
    """PyTensor and JAX evaluators of the per-row likelihood terms and of the total gradient."""
    import jax
    from pymc.sampling.jax import get_jaxified_graph, get_jaxified_logp

    pmm = m.pymc_model
    names = [v.name for v in pmm.value_vars]
    rows_term = pmm.logp(vars=[pmm[m.names["y"]]], sum=False)[0]
    rows_term_v = pmm.replace_rvs_by_values([rows_term])[0]
    f_rows_pt = pmm.compile_fn(rows_term_v, inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    f_rows_jx = get_jaxified_graph(inputs=pmm.value_vars, outputs=[rows_term_v])
    dlogp = pmm.compile_dlogp()
    jf = get_jaxified_logp(pmm)
    jgrad = jax.grad(lambda *args: jf(list(args)), argnums=tuple(range(len(names))))
    return names, f_rows_pt, (lambda point: np.asarray(f_rows_jx(*[point[n_] for n_ in names])[0])), dlogp, jgrad


@pytest.mark.parametrize("n", GRID_N)
def test_extreme_logits_and_large_counts_both_paths(n):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pymc.sampling.jax import get_jaxified_logp

    m = grid_model(n)
    pmm = m.pymc_model
    logp, dlogp = pmm.compile_logp(), pmm.compile_dlogp()
    names = [v.name for v in pmm.value_vars]
    jf = get_jaxified_logp(pmm)
    jgrad = jax.grad(lambda *args: jf(list(args)), argnums=tuple(range(len(names))))
    kw = dict(response="k", trials="n", predictors=["x"], group="g", priors=PRIORS_DICT, center=True)
    z = np.zeros(m.design.J)
    for eta in sorted({e for k in grid_ks(n) for e in etas_for(n, k)}):
        nat = dict(intercept=eta, b=np.array([0.0]), sd=0.1, z=z)           # every row at eta
        point = m.value_point(**nat)
        ref = reference_logp_binomial(_ref_point(nat), m.data, jacobian=True, **kw)
        gref = reference_grad_binomial(_ref_point(nat), m.data, response="k", trials="n", predictors=["x"],
                                       group="g", prior_intercept_sigma=1.5, prior_b_sigma=1.0)
        gref_vec = np.concatenate([np.atleast_1d(gref[n_]) for n_ in names])
        v_pt, g_pt = float(logp(point)), np.asarray(dlogp(point), dtype=float)
        v_jx = float(jf([point[n_] for n_ in names]))
        g_jx = np.concatenate([np.atleast_1d(np.asarray(g, dtype=float)) for g in jgrad(*[jnp.asarray(point[n_]) for n_ in names])])
        assert np.isfinite(v_pt) and np.isfinite(v_jx) and np.all(np.isfinite(g_pt)) and np.all(np.isfinite(g_jx))
        assert abs(v_pt - ref) <= tol(ref), (n, eta, v_pt, ref)
        assert abs(v_jx - ref) <= tol(ref), (n, eta, v_jx, ref)
        gt = np.maximum(1e-8, 1e-12 * np.abs(gref_vec))
        assert np.all(np.abs(g_pt - gref_vec) <= gt), (n, eta, g_pt, gref_vec)
        assert np.all(np.abs(g_jx - gref_vec) <= gt), (n, eta, g_jx, gref_vec)


def test_count_cap_is_the_domain_boundary():
    assert COUNT_CAP == 10**6
    df = pd.DataFrame({"k": [1, 2], "n": [COUNT_CAP, COUNT_CAP], "x": [0.0, 1.0], "g": ["a", "b"]})
    ab.binomial(FORMULA, df, PRIORS)                                        # at the cap: accepted
    with pytest.raises(ab.FormulaError, match="above the tested limit 1000000"):
        ab.binomial(FORMULA, df.assign(n=[COUNT_CAP + 1, COUNT_CAP]), PRIORS)
    with pytest.raises(ab.FormulaError, match="above the tested limit"):
        ab.binomial(FORMULA, df.assign(n=[float(COUNT_CAP) * 4, 5.0], k=[1.0, 2.0]), PRIORS)
    for n in (10**9, 2**31 - 2, 2**31 - 1):                                # the reviewer's failing cases are outside the domain
        with pytest.raises(ab.FormulaError, match="above the tested limit"):
            ab.binomial(FORMULA, df.assign(n=[n, n], k=[1, 2]), PRIORS)


# -- per-row normalised checks over the whole acceptance grid (repair record) ----------------------
@pytest.mark.parametrize("n", GRID_N)
def test_per_row_terms_and_derivatives_both_paths(n):
    """Every (n, k, eta) grid point checked as an individual row term, unmasked by other rows:
    per-row likelihood values (PyTensor and JAX), and per-row derivatives read from the offset
    gradient components (sd = 1, z = 0 make d/dz_j the row-j derivative), plus the total."""
    pytest.importorskip("jax")
    m = grid_model(n)
    names, f_rows_pt, f_rows_jx, dlogp, jgrad = per_row_functions(m)
    ks = grid_ks(n)
    J = m.design.J
    i_off = names.index("1|g_offset")
    kw = dict(response="k", trials="n", predictors=["x"], group="g", priors=PRIORS_DICT, center=True)
    etas = sorted({e for k in ks for e in etas_for(n, k)})
    for eta in etas:
        nat = dict(intercept=eta, b=np.array([0.0]), sd=1.0, z=np.zeros(J))   # every row exactly at eta
        point = m.value_point(**nat)
        refs = np.array([float(row_loglik_dec(int(k), n, eta)) for k in ks])
        grefs = np.array([float(row_grad_dec(int(k), n, eta)) for k in ks])
        rows_pt, rows_jx = np.asarray(f_rows_pt(point)).ravel(), np.asarray(f_rows_jx(point)).ravel()
        assert rows_pt.shape == (len(ks),)
        for j, k in enumerate(ks):
            tv = tol(refs[j])
            assert np.isfinite(rows_pt[j]) and abs(rows_pt[j] - refs[j]) <= tv, (n, k, eta, rows_pt[j], refs[j])
            assert np.isfinite(rows_jx[j]) and abs(rows_jx[j] - refs[j]) <= tv, (n, k, eta, rows_jx[j], refs[j])
        g_pt = np.asarray(dlogp(point), dtype=float)
        g_jx = jgrad(*[__import__("jax").numpy.asarray(point[n_]) for n_ in names])
        off_pt = g_pt[sum(int(np.asarray(point[n_]).size) for n_ in names[:i_off]):][:J]
        off_jx = np.asarray(g_jx[i_off], dtype=float)
        for j, k in enumerate(ks):                                            # row-j derivative, minus z_j = 0
            tg = tol(grefs[j])
            assert np.isfinite(off_pt[j]) and abs(off_pt[j] - grefs[j]) <= tg, (n, k, eta, off_pt[j], grefs[j])
            assert np.isfinite(off_jx[j]) and abs(off_jx[j] - grefs[j]) <= tg, (n, k, eta, off_jx[j], grefs[j])
        ref_total = reference_logp_binomial(_ref_point(nat), m.data, jacobian=True, **kw)
        assert abs(float(m.pymc_model.compile_logp()(point)) - ref_total) <= tol(ref_total)


@pytest.mark.parametrize("n", [10**3, 10**5, COUNT_CAP - 1, COUNT_CAP])
@pytest.mark.parametrize("kfrac", ["one", "quarter", "half", "n-1"])
def test_homogeneous_near_mode_models_with_nontrivial_parameters(n, kfrac):
    """All rows at one (n, k), sd != 1, z != 0, a slope on a varying predictor: rows sit at
    different logits near the mode. Model density and full gradient on both paths."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from pymc.sampling.jax import get_jaxified_logp

    k = {"one": 1, "quarter": n // 4, "half": n // 2, "n-1": n - 1}[kfrac]
    rows = 6
    df = pd.DataFrame({"k": [k] * rows, "n": [n] * rows, "x": np.linspace(-1.0, 1.0, rows),
                       "g": [f"g{i}" for i in range(rows)]})
    m = ab.binomial(FORMULA, df, PRIORS)
    pmm = m.pymc_model
    logp, dlogp = pmm.compile_logp(), pmm.compile_dlogp()
    names = [v.name for v in pmm.value_vars]
    jf = get_jaxified_logp(pmm)
    jgrad = jax.grad(lambda *args: jf(list(args)), argnums=tuple(range(len(names))))
    mode = math.log(k) - math.log(n - k)
    rng = np.random.default_rng(11)
    kw = dict(response="k", trials="n", predictors=["x"], group="g", priors=PRIORS_DICT, center=True)
    for d in MODE_OFFSETS:
        nat = dict(intercept=mode + d, b=np.array([0.05]), sd=0.3, z=rng.normal(size=rows) * 0.5)
        point = m.value_point(**nat)
        ref = reference_logp_binomial(_ref_point(nat), m.data, jacobian=True, **kw)
        gref = reference_grad_binomial(_ref_point(nat), m.data, response="k", trials="n", predictors=["x"],
                                       group="g", prior_intercept_sigma=1.5, prior_b_sigma=1.0)
        gref_vec = np.concatenate([np.atleast_1d(gref[n_]) for n_ in names])
        v_pt, g_pt = float(logp(point)), np.asarray(dlogp(point), dtype=float)
        v_jx = float(jf([point[n_] for n_ in names]))
        g_jx = np.concatenate([np.atleast_1d(np.asarray(g, dtype=float)) for g in jgrad(*[jnp.asarray(point[n_]) for n_ in names])])
        assert abs(v_pt - ref) <= tol(ref), (n, k, d, v_pt, ref)
        assert abs(v_jx - ref) <= tol(ref), (n, k, d, v_jx, ref)
        gt = np.maximum(1e-8, 1e-12 * np.abs(gref_vec))
        assert np.all(np.abs(g_pt - gref_vec) <= gt), (n, k, d, g_pt - gref_vec)
        assert np.all(np.abs(g_jx - gref_vec) <= gt), (n, k, d, g_jx - gref_vec)


@pytest.mark.parametrize("n", [COUNT_CAP - 1, COUNT_CAP])
def test_reviewer_two_row_near_mode_construction(n):
    """The construction that failed at n = 1e9 on the reviewed head, at the new cap: two identical
    rows in two groups, intercept-only, sd = 1, z = 0, at the mode. Both paths, unmasked."""
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    k = n // 2
    df = pd.DataFrame({"k": [k, k], "n": [n, n], "g": ["a", "b"]})
    m = ab.binomial("k | trials(n) ~ 1 + (1 | g)", df, PRIORS)
    pdct = {"intercept": PRIORS_DICT["intercept"], "b": {}, "sd": PRIORS_DICT["sd"]}
    for eta in [0.0, 1e-3, -0.1]:
        point = m.value_point(intercept=eta, b=[], sd=1.0, z=np.zeros(2))
        ref = reference_logp_binomial({"a_c": eta, "b": np.array([]), "log_sd": 0.0, "z": np.zeros(2)}, m.data,
                                      response="k", trials="n", predictors=[], group="g", priors=pdct)
        v_pt = float(m.pymc_model.compile_logp()(point))
        v_jx = float(get_jaxified_logp(m.pymc_model)([point[v.name] for v in m.pymc_model.value_vars]))
        assert abs(v_pt - ref) <= tol(ref) and abs(v_jx - ref) <= tol(ref), (n, eta, v_pt, v_jx, ref)


# -- A6: design, levels, logit_p from the graph, zero-trial rows -------------------------------------
def test_design_levels_and_zero_trial_rows(cdata):
    df = cdata.copy()
    df.loc[[2, 7], ["k", "n"]] = 0                                           # two zero-trial rows
    m = ab.binomial(FORMULA, df, PRIORS)
    assert m.dropped_rows == (2, 7) and m.design.n == len(df) - 2 and int(m.design.trials.sum()) == int(df.n.sum())
    kept = df.drop(index=[2, 7]).reset_index(drop=True)
    assert m.design.levels == tuple(sorted(kept.g.unique()))
    assert np.array_equal(m.design.group_index, np.array([m.design.levels.index(v) for v in kept.g]))
    assert np.allclose(m.design.X_model[:, 0], kept.x.to_numpy() - kept.x.mean())      # centred over fitted rows
    assert np.allclose(m.design.x_means, [kept.x.mean()]) and abs(kept.x.mean() - df.x.mean()) > 1e-12
    assert np.array_equal(m.design.y, kept.k.to_numpy()) and np.array_equal(m.design.trials, kept.n.to_numpy())
    assert "2 zero-trial row(s) dropped (positions [2, 7])" in m.describe()
    nat = _random_points(m, k=1)[0]
    point = m.value_point(**nat)
    pmm = m.pymc_model
    eta_v, u_v = pmm.replace_rvs_by_values([pmm["logit_p"], pmm["1|g"]])
    fn = pmm.compile_fn([eta_v, u_v], inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    eta_graph, u_graph = (np.asarray(v).ravel() for v in fn(point))
    assert np.allclose(eta_graph, _eta(m, nat), atol=1e-12) and np.allclose(u_graph, nat["sd"] * nat["z"], atol=1e-12)
    assert {v.name for v in pmm.free_RVs} == {"Intercept", "x", "1|g_sigma", "1|g_offset"}
    assert m.names["y"] == "p(k, n)" and m.spec.bambi_formula == "p(k, n) ~ x + (1 | g)"


# -- A6a: n = 1 equivalence with the Bernoulli family on the shared domain ---------------------------
def test_n_equals_one_matches_bernoulli_on_the_shared_domain():
    rng = np.random.default_rng(4)
    rows, J = 80, 5
    g = np.array([i % J for i in range(rows)]); rng.shuffle(g)
    x = rng.normal(size=rows)
    k = rng.binomial(1, expit(-0.3 + 0.8 * x + rng.normal(0, 0.7, J)[g]))
    df = pd.DataFrame({"k": k, "n": np.ones(rows, dtype=int), "x": x, "g": [f"lvl{j:02d}" for j in g]})
    assert set(np.unique(k)) == {0, 1}
    bin_m = ab.binomial(FORMULA, df, PRIORS)
    ber_m = ab.bernoulli("k ~ x + (1 | g)", df, PRIORS)
    names_bin = [v.name for v in bin_m.pymc_model.value_vars]
    assert names_bin == [v.name for v in ber_m.pymc_model.value_vars]
    lp_bin, lp_ber = bin_m.pymc_model.compile_logp(), ber_m.pymc_model.compile_logp()
    dl_bin, dl_ber = bin_m.pymc_model.compile_dlogp(), ber_m.pymc_model.compile_dlogp()
    for nat in _random_points(bin_m):
        pt_ = bin_m.value_point(**nat)
        a, b = float(lp_bin(pt_)), float(lp_ber(pt_))
        assert abs(a - b) <= tol(b)                                           # log C(1, k) = 0
        ga, gb = np.asarray(dl_bin(pt_)), np.asarray(dl_ber(pt_))
        assert np.all(np.abs(ga - gb) <= np.maximum(1e-8, 1e-12 * np.abs(gb)))
    fit = bin_m.fit(backend="pymc", chains=2, draws=60, tune=60, seed=2, cores=1)
    p_bin = bin_m.predict(fit, kind="probability")["p"].to_numpy()
    p_ber = ber_m.predict(fit, kind="mean")["p"].to_numpy()
    assert np.allclose(p_bin, p_ber, atol=1e-10)
    # outside the shared domain the policies differ by design
    const = df.assign(k=0)
    m = ab.binomial(FORMULA, const, PRIORS)
    assert m.warnings and "zero successes" in m.warnings[0] and "WARNING" in m.describe()
    with pytest.raises(ab.FormulaError, match="constant"):
        ab.bernoulli("k ~ x + (1 | g)", const, PRIORS)
    m2 = ab.binomial(FORMULA, df.assign(k=df.n), PRIORS)
    assert "all successes" in m2.warnings[0]


# -- A6b: expansion to Bernoulli rows with aligned centring ------------------------------------------
def expand(df):
    rows = []
    for _, r in df.iterrows():
        rows += [{"k": 1, "x": r.x, "g": r.g}] * int(r.k) + [{"k": 0, "x": r.x, "g": r.g}] * int(r.n - r.k)
    return pd.DataFrame(rows)


def test_expansion_to_bernoulli_rows_with_aligned_centring():
    for center, equal in [(False, None), (True, 4)]:                          # equal trials => equal means
        agg = simulate_counts(rows=40, J=4, seed=6, lo=2, hi=6, equal_trials=equal)
        agg = agg[agg.n > 0].reset_index(drop=True)
        exp_df = expand(agg)
        bin_m = ab.binomial(FORMULA, agg, PRIORS, center_predictors=center)
        ber_m = ab.bernoulli("k ~ x + (1 | g)", exp_df, PRIORS, center_predictors=center)
        if center:
            assert abs(agg.x.mean() - exp_df.x.mean()) < 1e-12               # aligned by construction
        lp_bin = bin_m.pymc_model.compile_logp(vars=[bin_m.pymc_model["p(k, n)"]], jacobian=False)
        lp_ber = ber_m.pymc_model.compile_logp(vars=[ber_m.pymc_model["k"]], jacobian=False)
        const = float(sum(log_binom_dec(int(n), int(k)) for n, k in zip(agg.n, agg.k)))
        for nat in _random_points(bin_m, k=3):
            pt_ = bin_m.value_point(**nat)
            assert abs(float(lp_bin(pt_)) - const - float(lp_ber(pt_))) <= tol(float(lp_ber(pt_)))
    # with unequal trials and centring on, the two centrings differ and so do the likelihoods
    agg = simulate_counts(rows=40, J=4, seed=6, lo=2, hi=6)
    agg = agg[agg.n > 0].reset_index(drop=True)
    exp_df = expand(agg)
    assert abs(agg.x.mean() - exp_df.x.mean()) > 1e-6
    bin_m = ab.binomial(FORMULA, agg, PRIORS)
    ber_m = ab.bernoulli("k ~ x + (1 | g)", exp_df, PRIORS)
    lp_bin = bin_m.pymc_model.compile_logp(vars=[bin_m.pymc_model["p(k, n)"]], jacobian=False)
    lp_ber = ber_m.pymc_model.compile_logp(vars=[ber_m.pymc_model["k"]], jacobian=False)
    const = float(sum(log_binom_dec(int(n), int(k)) for n, k in zip(agg.n, agg.k)))
    nat = dict(intercept=0.4, b=np.array([0.9]), sd=0.5, z=np.zeros(4))
    assert abs(float(lp_bin(bin_m.value_point(**nat))) - const - float(lp_ber(bin_m.value_point(**nat)))) > 1e-6


# -- A7: invalid inputs, refused before any cast -----------------------------------------------------
def test_invalid_inputs(cdata):
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        warnings.simplefilter("error", RuntimeWarning)
        bad = [
            (cdata.assign(k=cdata.k.astype(bool)), "boolean"),
            (cdata.assign(n=cdata.n.astype(complex)), "complex"),
            (cdata.assign(k=cdata.k + 0.5), "non-integral"),
            (cdata.assign(n=cdata.n.astype(float).where(cdata.index > 0, np.inf)), "non-finite"),
            (cdata.assign(k=cdata.k.where(cdata.index > 0, -1)), "negative"),
            (cdata.assign(k=cdata.n + 1), "exceed trials"),
            (cdata.assign(n=cdata.n.where(cdata.index > 0, 0)), "zero trials is impossible"),
            (cdata.assign(n=0, k=0), "no trials in the data"),
            (cdata.assign(k=cdata.k.astype(float).where(cdata.index > 0, np.nan)), "missing values"),
        ]
        for df, needle in bad:
            with pytest.raises(ab.FormulaError, match=needle):
                ab.binomial(FORMULA, df, PRIORS)
    with pytest.raises(ab.FormulaError, match="above the tested limit"):
        ab.binomial(FORMULA, cdata.assign(n=cdata.n.astype(np.int64).where(cdata.index > 0, COUNT_CAP + 5)), PRIORS)
    with pytest.raises(ab.PriorError, match="no residual SD"):
        ab.binomial(FORMULA, cdata, ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)))
    with pytest.raises(TypeError):
        ab.Priors(intercept=ab.Normal(0, 1.5), b=ab.Normal(0, 1))
    for bad_formula, needle in [
        ("k ~ x + (1 | g)", "needs 'successes | trials"), ("p(k, n) ~ x + (1 | g)", "Bambi's 'p"),
        ("k | trials(n) ~ x + (1 + x | g)", "varying slopes"), ("k | trials(n) ~ x + (1 | g/h)", "nested"),
        ("k | trials(n) ~ 0 + x + (1 | g)", "no intercept"), ("k | trials(n) ~ x", "exactly one varying-intercept"),
        ("k | trials(n) ~ n + (1 | g)", "cannot also be a predictor"), ("k | trials(n) ~ x + (1 | n)", "cannot also be a predictor"),
        ("k | trials(k) ~ x + (1 | g)", "cannot also be the response"), ("k | n ~ x + (1 | g)", "exactly 'successes | trials"),
        ("k | weights(n) ~ x + (1 | g)", "exactly 'successes | trials"),
    ]:
        with pytest.raises(ab.FormulaError, match=needle):
            ab.binomial(bad_formula, cdata, PRIORS)
    with pytest.raises(ab.FormulaError, match="binomial family's left-hand side"):
        ab.gaussian(FORMULA, cdata.assign(y=cdata.k.astype(float)), ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)))
    with pytest.raises(ab.FormulaError, match="binomial family's left-hand side"):
        ab.bernoulli(FORMULA, cdata, PRIORS)
    m = ab.binomial(FORMULA, cdata, PRIORS)
    with pytest.raises(ValueError, match="sigma"):
        m.value_point(intercept=0.0, b=[0.0], sd=1.0, z=np.zeros(m.design.J), sigma=1.0)


# -- A8: predictions ---------------------------------------------------------------------------------
def test_prediction_meaning(cmodel, cfit, cdata):
    post = cfit.idata.posterior
    a, b, u = (post[k_].to_numpy() for k_ in ("Intercept", "x", "1|g"))
    p_hand = expit(a[..., None] + b[..., None] * cdata.x.to_numpy() + u[..., cmodel.design.group_index])
    prob = cmodel.predict(cfit, kind="probability")
    assert set(prob.data_vars) == {"p"} and prob["p"].dims == ("chain", "draw", "__obs__") and prob.attrs["new_groups"] == "in_sample"
    assert np.allclose(prob["p"].to_numpy(), p_hand, atol=1e-10) and float(prob["p"].min()) >= 0 and float(prob["p"].max()) <= 1
    mean = cmodel.predict(cfit, kind="mean")
    assert set(mean.data_vars) == {"p", "expected"}
    assert np.array_equal(mean["expected"].to_numpy(), mean["p"].to_numpy() * cdata.n.to_numpy())
    resp = cmodel.predict(cfit, kind="response", seed=0)
    assert set(resp.data_vars) == {"p", "expected", "y"}
    yv = resp["y"].to_numpy()
    assert np.issubdtype(yv.dtype, np.integer) and np.all(yv >= 0) and np.all(yv <= cdata.n.to_numpy())
    new = pd.DataFrame({"x": [0.0, 1.0, 0.0], "n": [10, 0, 25], "g": ["lvl00", "lvl03", "lvl00"]})
    obs = cmodel.predict(cfit, newdata=new, kind="response", seed=1)
    lv = list(cmodel.design.levels)
    for i, (xv, gv, nv) in enumerate(zip(new.x, new.g, new.n)):
        assert np.allclose(obs["p"][..., i].to_numpy(), expit(a + b * xv + u[..., lv.index(gv)]), atol=1e-10)
    assert np.all(obs["expected"][..., 1].to_numpy() == 0) and np.all(obs["y"][..., 1].to_numpy() == 0)   # n = 0 rows
    assert cmodel.predict(cfit, newdata=new.drop(columns="n"), kind="probability").sizes["__obs__"] == 3
    with pytest.raises(ab.FormulaError, match="lacks the trials column"):
        cmodel.predict(cfit, newdata=new.drop(columns="n"), kind="mean")
    with pytest.raises(ab.FormulaError, match="non-integral"):
        cmodel.predict(cfit, newdata=new.assign(n=[10.5, 0, 25]), kind="mean")
    with pytest.raises(ValueError, match="kind must be"):
        cmodel.predict(cfit, kind="counts")


def test_new_group_semantics(cmodel, cfit):
    from tests.test_predict import check_resample_observed_identity, synthetic_posterior

    idata = synthetic_posterior(cmodel, effects=np.linspace(-2.0, 2.0, cmodel.design.J))
    check_resample_observed_identity(cmodel, idata, "p", expit, kind="probability")
    new = pd.DataFrame({"x": [0.0, 0.0, 0.0], "n": [20, 20, 20], "g": ["lvl00", "NEW_A", "NEW_B"]})
    with pytest.raises(ab.FormulaError, match="not seen in training"):
        cmodel.predict(cfit, newdata=new, kind="mean")
    post = cfit.idata.posterior
    a, u, sd = (post[k_].to_numpy() for k_ in ("Intercept", "1|g", "1|g_sigma"))
    pop = cmodel.predict(cfit, newdata=new, kind="response", new_groups="population", seed=1)
    assert np.allclose(pop["p"][..., 0].to_numpy(), expit(a + u[..., 0]), atol=1e-10)
    dev = np.log(pop["p"][..., 1].to_numpy() / (1 - pop["p"][..., 1].to_numpy())) - a
    target = np.sqrt(np.mean(sd ** 2))
    assert abs(dev.std() - target) < 0.15 * target
    assert np.all(pop["y"].to_numpy() <= 20) and np.array_equal(pop["expected"].to_numpy(), pop["p"].to_numpy() * 20)


# -- A9: predictive check conditional on the saved draws, with an exact tail bound ------------------
def bernstein_t(V, bound, delta=1e-6):
    c = math.log(2.0 / delta)
    return (2 * bound * c / 3 + math.sqrt((2 * bound * c / 3) ** 2 + 8 * c * V)) / 2


def test_predictive_counts_conditional_on_posterior(cmodel, cfit, cdata):
    resp = cmodel.predict(cfit, kind="response", seed=5)
    p, y = resp["p"].to_numpy(), resp["y"].to_numpy()
    n = cdata.n.to_numpy()
    E = (p * n).sum(axis=(0, 1)); V = (p * (1 - p) * n).sum(axis=(0, 1)); T = y.sum(axis=(0, 1))
    zero_var = V == 0
    assert np.array_equal(T[zero_var], E[zero_var])                            # exact where the variance is zero
    for i in np.flatnonzero(~zero_var):
        assert abs(T[i] - E[i]) <= bernstein_t(V[i], n[i])                      # false-failure bound 1e-6 per row
    assert abs(T.sum() - E.sum()) <= bernstein_t(V.sum(), n.max())              # pooled: total variance, max bound
    # saturated rows: deterministic outputs
    sat = pd.DataFrame({"x": [0.0, 0.0], "n": [7, 7], "g": ["lvl00", "lvl01"]})
    idata = copy.deepcopy(cfit.idata)
    idata.posterior["Intercept"].values[:] = 60.0
    idata.posterior["1|g"].values[:] = 0.0
    idata.posterior["x"].values[:] = 0.0
    ds = cmodel.predict(idata, newdata=sat, kind="response", seed=2)
    assert np.all(ds["p"].to_numpy() == 1.0) and np.all(ds["y"].to_numpy() == 7) and np.all(ds["expected"].to_numpy() == 7.0)


# -- records and devices ----------------------------------------------------------------------------
def test_fit_record_and_report(cmodel, cfit):
    rec = ab.fit_record(cmodel, cfit)
    json.dumps(rec)
    assert rec["model"]["family"] == "binomial" and rec["model"]["link"] == "logit" and rec["model"]["trials"] == "n"
    assert rec["model"]["zero_trial_rows_dropped"] == 0 and rec["model"]["warnings"] == []
    assert set(rec["posterior"]) == {"Intercept", "x", "1|g_sigma"} and "logit scale" in rec["model"]["intercept_prior_applies_to"]
    assert cfit.dtype == "float64" and "seed             : 1" in cfit.report()


def test_gpu_unavailable_is_error_for_this_family(cmodel):
    jax = pytest.importorskip("jax")
    if any(d.platform in {"gpu", "cuda", "rocm"} for d in jax.devices()):
        pytest.skip("a GPU is visible")
    with pytest.raises(ab.DeviceError, match="does not fall back"):
        cmodel.fit(backend="numpyro", device="gpu", chains=1, draws=5, tune=5)


def test_numpyro_cpu_path(cmodel):
    pytest.importorskip("numpyro")
    r = cmodel.fit(backend="numpyro", device="cpu", chains=2, draws=40, tune=40, seed=3)
    assert r.platform == "cpu" and r.dtype == "float64" and r.chain_method == "parallel"
    assert set(r.idata.posterior.data_vars) >= {"Intercept", "x", "1|g_sigma", "1|g"}


# -- N1-N5 (correction record): the exposed RV is one normalised distribution -----------------
def exact_pmf(n, eta):
    """Independent probabilities: math.comb and a 60-digit decimal expit, no SciPy."""
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    p = 1 / (1 + (-Decimal(eta)).exp())
    return np.array([float(Decimal(math.comb(n, k)) * p ** k * (1 - p) ** (n - k)) for k in range(n + 1)])


def two_row_model(n, k_obs):
    df = pd.DataFrame({"k": [k_obs, k_obs], "n": [n, n], "g": ["a", "b"]})
    return ab.binomial("k | trials(n) ~ 1 + (1 | g)", df, PRIORS)


def compiled_value_functions(m):
    """Functions of a symbolic value vector (integer and float) on both paths, at fixed params."""
    import pymc as pm
    import pytensor.tensor as pt
    from pymc.sampling.jax import get_jaxified_graph

    pmm = m.pymc_model
    rv = pmm[m.names["y"]]
    out = {}
    for dtype in ("int64", "float64"):
        v = pt.vector("v", dtype=dtype)
        lp = pmm.replace_rvs_by_values([pm.logp(rv, v)])[0]
        f_pt = pmm.compile_fn(lp, inputs=[v, *pmm.value_vars], point_fn=False, on_unused_input="ignore")
        f_jx = get_jaxified_graph(inputs=[v, *pmm.value_vars], outputs=[lp])
        out[dtype] = (f_pt, lambda vals, *args, f=f_jx: np.asarray(f(vals, *args)[0]))
    return out


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 25])
@pytest.mark.parametrize("eta", [0.0, 0.7, -0.7, 3.0, -3.0, -12.0])
def test_support_enumeration_matches_exact_binomial_and_sums_to_one(n, eta):
    pytest.importorskip("jax")
    m = two_row_model(n, min(1, n))
    fns = compiled_value_functions(m)
    point = m.value_point(intercept=eta, b=[], sd=1.0, z=np.zeros(2))
    args = [point[x.name] for x in m.pymc_model.value_vars]
    exact = exact_pmf(n, eta)
    for dtype, (f_pt, f_jx) in fns.items():
        for f in (f_pt, f_jx):
            pmf = np.array([np.exp(np.asarray(f(np.array([k, k], dtype=dtype), *args))[0]) for k in range(n + 1)])
            assert np.allclose(pmf, exact, rtol=0, atol=1e-12), (n, eta, dtype, pmf, exact)
            assert abs(pmf.sum() - 1.0) < 1e-12


@pytest.mark.parametrize("n,eta", [(2, 0.0), (5, 0.7), (10, -3.0)])
def test_pmf_does_not_depend_on_the_observed_outcome(n, eta):
    pytest.importorskip("jax")
    pmfs = []
    for k_obs in range(n + 1):
        m = two_row_model(n, k_obs)
        f_pt, f_jx = compiled_value_functions(m)["int64"]
        point = m.value_point(intercept=eta, b=[], sd=1.0, z=np.zeros(2))
        args = [point[x.name] for x in m.pymc_model.value_vars]
        pmfs.append(np.array([np.exp(np.asarray(f(np.array([k, k]), *args))[0]) for f in (f_pt, f_jx) for k in range(n + 1)]))
    exact = np.tile(exact_pmf(n, eta), 2)
    for pmf in pmfs:
        assert np.allclose(pmf, pmfs[0], rtol=0, atol=1e-14) and np.allclose(pmf, exact, rtol=0, atol=1e-12)


def test_symbolic_value_evaluated_repeatedly_after_compilation():
    pytest.importorskip("jax")
    n, eta = 5, 0.7
    m = two_row_model(n, 2)
    f_pt, f_jx = compiled_value_functions(m)["int64"]
    point = m.value_point(intercept=eta, b=[], sd=1.0, z=np.zeros(2))
    args = [point[x.name] for x in m.pymc_model.value_vars]
    exact = exact_pmf(n, eta)
    for _ in range(2):                                     # two passes over the support with the same compiled functions
        for k in range(n + 1):
            for f in (f_pt, f_jx):
                lp = np.asarray(f(np.array([k, n - k]), *args))
                assert np.allclose(np.exp(lp), [exact[k], exact[n - k]], rtol=0, atol=1e-12)
    # a different point of the parameter space through the same compiled function
    point2 = m.value_point(intercept=-3.0, b=[], sd=1.0, z=np.zeros(2))
    args2 = [point2[x.name] for x in m.pymc_model.value_vars]
    assert np.allclose([np.exp(np.asarray(f_pt(np.array([k, k]), *args2))[0]) for k in range(n + 1)], exact_pmf(n, -3.0), atol=1e-12)


def test_random_draws_match_the_mass():
    import pymc as pm
    from autobayes.likelihoods import BinomialLogit

    N, delta = 20000, 1e-6
    for n, eta in [(2, 0.0), (10, -0.7), (25, 3.0)]:
        dist = BinomialLogit.dist(np.full(N, eta), np.full(N, n, dtype=np.int64))
        draws = pm.draw(dist, random_seed=1)
        assert draws.shape == (N,) and np.issubdtype(draws.dtype, np.integer) and draws.min() >= 0 and draws.max() <= n
        exact = exact_pmf(n, eta)
        bound = math.sqrt(math.log(2 * (n + 1) / delta) / (2 * N))          # Hoeffding, per outcome
        freq = np.bincount(draws, minlength=n + 1) / N
        assert np.all(np.abs(freq - exact) <= bound), (n, eta, freq, exact, bound)
        assert abs(draws.mean() - n / (1 + math.exp(-eta))) <= n * math.sqrt(math.log(2 / delta) / (2 * N))


def test_invalid_values_have_zero_mass():
    pytest.importorskip("jax")
    n, eta = 5, 0.7
    m = two_row_model(n, 2)
    fns = compiled_value_functions(m)
    point = m.value_point(intercept=eta, b=[], sd=1.0, z=np.zeros(2))
    args = [point[x.name] for x in m.pymc_model.value_vars]
    exact = exact_pmf(n, eta)
    for f in fns["float64"]:
        for bad, ok in (([-1.0, 2.0], [False, True]), ([6.0, 3.0], [False, True]), ([1.5, 1.0], [False, True]),
                        ([2.0, 5.5], [True, False]), ([-0.5, 5.0], [False, True])):
            lp = np.asarray(f(np.array(bad), *args))
            ok = np.array(ok)
            assert np.all(np.isneginf(lp[~ok])), (bad, lp)
            assert np.allclose(np.exp(lp[ok]), exact[np.array(bad)[ok].astype(int)], rtol=0, atol=1e-12), (bad, lp)
    for f in fns["int64"]:
        assert np.all(np.isneginf(np.asarray(f(np.array([-1, 6]), *args))))
        assert np.allclose(np.exp(np.asarray(f(np.array([0, 5]), *args))), [exact[0], exact[5]], rtol=0, atol=1e-12)
