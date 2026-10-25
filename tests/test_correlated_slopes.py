"""Correlated varying intercept and slopes, ``(1 + x | g)`` under an LKJ prior (brms-parity feature A02;
docs/correlated-slopes.md).

The grammar and the ``cor`` prior's coverage rules, Stan's ``cholesky_corr`` transform (round trip, Jacobian against
finite differences) and the LKJ normalising constant against closed forms, the rebuilt graph (names, dims, value
variables), the density against an explicit reference with the LKJ term and the Jacobian (two and three
coefficients, and two correlated factors), every family's likelihood kept through the graph edit, ``value_point``,
by-hand predictions under both ``new_groups`` policies and their agreement with the Bambi path, the fit record, the
target identity and ``describe()``. Build-only; nothing samples. The numbers against brms are the oracle's business
(tests/oracle, case ``gauss_slope_correlated``)."""
import json
import re
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr
from scipy.special import betaln
from scipy.stats import halfnorm, norm

import autobayes as ab
from autobayes.correlation import (CholeskyCorrTransform, check_correlation_cholesky, cholesky_corr_constrain,
                                   cholesky_corr_free, lkj_cholesky_rng, lkj_log_normalizer, lower_indices, n_free)
from autobayes.formula import GroupTerm
from autobayes.identity import FIT_IDENTITY_ATTR

J_G = 4
LEVELS_G = tuple(f"g{i}" for i in range(J_G))


def frame(n=24, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(0.4, 1.0, n), "w": rng.normal(size=n),
                         "g": [LEVELS_G[i % J_G] for i in range(n)], "h": [f"h{(i // 2) % 3}" for i in range(n)]})


def indicator(df, factor, levels):
    return np.column_stack([df[factor].to_numpy() == l for l in levels]).astype(float)


SD2 = {"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}}
PRIORS2 = ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5), sd=SD2, cor={"g": ab.LKJ(2)})
SD3 = {"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5), "w": ab.HalfNormal(0.7)}}
PRIORS3 = ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5), sd=SD3, cor=ab.LKJ(2))


@pytest.fixture(scope="module")
def two():
    """``y ~ x + (1 + x | g)``: one factor, intercept and one slope correlated."""
    return ab.gaussian("y ~ x + (1 + x | g)", frame(), PRIORS2)


@pytest.fixture(scope="module")
def three():
    """``y ~ x + w + (1 + x + w | g)``: three correlated coefficients, the bare ``cor`` prior."""
    return ab.gaussian("y ~ x + w + (1 + x + w | g)", frame(), PRIORS3)


# -- 1. the grammar and the cor prior ------------------------------------------------------------------------------
def test_grammar_accepts_the_correlated_form():
    s = ab.parse_formula("y ~ x + w + (1 + x | g) + (1 | h)")
    assert s.group_terms == (GroupTerm("g", ("Intercept", "x"), True), GroupTerm("h", ("Intercept",), True))
    assert s.correlated_terms == (s.group_terms[0],) and s.group_terms[0].has_correlation and not s.group_terms[1].has_correlation
    assert s.group_terms[0].canonical == "(1 + x | g)" and s.canonical_formula == "y ~ x + w + (1 + x | g) + (1 | h)"
    assert s.bambi_formula == "y ~ x + w + (1 + x | g) + (1 | h)"                      # Bambi builds the independent terms; Model correlates them
    assert ab.parse_formula("y ~ x + (1 + x || g)").correlated_terms == ()               # '||' stays independent
    assert not ab.parse_formula("y ~ x + (1 || g)").group_terms[0].has_correlation       # one coefficient: nothing to correlate
    nested = ab.parse_formula("y ~ x + (1 + x | g/h)")                                   # brms: both terms correlated
    assert nested.group_terms == (GroupTerm("g", ("Intercept", "x"), True), GroupTerm("g:h", ("Intercept", "x"), True))
    assert ab.parse_formula("y ~ x + w + (1 + x + w | g:h)").group_terms == (GroupTerm("g:h", ("Intercept", "x", "w"), True),)
    for formula, message in [("y ~ x + (1 + x |p| g)", "in a formula on its own it is not supported here"),   # D08: shared with sigma ~ only
                             ("y ~ x + (1 | gr(g, cov = A))", "brms's gr() options ['cov'] are not supported; gr() takes one grouping "
                                                              "column, by = <column>, cor = FALSE (brms's '||')"),   # F08 takes cor and id
                             ("y ~ x + (0 + x | g)", "varying slopes without the varying intercept are not supported; write '(1 + x | g)'"),
                             ("y ~ w + (1 + x | g)", "the varying slope 'x' must also be a population-level term")]:
        with pytest.raises(ab.FormulaError, match=re.escape(message)):
            ab.parse_formula(formula)


def test_cor_prior_coverage_rules():
    cor_spec = ab.parse_formula("y ~ x + (1 + x | g)")
    ind_spec = ab.parse_formula("y ~ x + (1 + x || g)")
    one_spec = ab.parse_formula("y ~ x + (1 | g)")
    both = ab.parse_formula("y ~ x + (1 + x | g) + (1 + x | h)")

    def validate(cor, spec=cor_spec, sd=SD2):
        ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=sd, cor=cor, sigma=ab.HalfNormal(1)).validate(spec, "gaussian")

    def refused(cor, message, spec=cor_spec, sd=SD2):
        with pytest.raises(ab.PriorError, match=re.escape(message)):
            validate(cor, spec, sd)

    validate({"g": ab.LKJ(2)})
    validate(ab.LKJ(1))                                                                  # one correlated factor: bare form
    refused(None, "the group term '(1 + x | g)' gives its coefficients ['Intercept', 'x'] a correlation matrix, which needs an LKJ prior: pass cor={'g': ab.LKJ(eta)}")
    refused({"g": ab.LKJ(2)}, "the formula has no correlated group term '(1 + x | g)'; do not pass cor; '(1 + x || g)' uses '||', "
                              "independent slopes with no correlation matrix: write '(1 + x | g)' for brms's correlated form", ind_spec)
    refused(ab.LKJ(2), "do not pass cor; only a group term with varying slopes written with '|', '(1 + x | g)', has a correlation matrix",
            one_spec, ab.HalfNormal(1))
    refused({"h": ab.LKJ(2)}, "cor must cover exactly the correlated grouping factors ['g']; got ['h']")
    refused({"g": ab.LKJ(2), "h": ab.LKJ(2)}, "cor must cover exactly the correlated grouping factors ['g']; got ['g', 'h']")
    refused({"g": ab.HalfNormal(1)}, "prior for 'cor:g' must be a correlation prior (['LKJ'], brms's lkj(eta)), got HalfNormal(sigma=1)")
    refused({"g": 2}, "prior for 'cor:g' must be a Prior, got int")
    sd_both = {"g": SD2["g"], "h": SD2["g"]}
    validate({"g": ab.LKJ(2), "h": ab.LKJ(1)}, both, sd_both)
    refused(ab.LKJ(2), "multiple correlated grouping factors require an exact-key cor mapping over ['g', 'h']; a bare prior is not broadcast", both, sd_both)
    refused({"g": ab.LKJ(2)}, "cor must cover exactly the correlated grouping factors ['g', 'h']; got ['g']", both, sd_both)
    # a correlated factor next to an independent one: cor for the independent one is refused with the form to write
    mixed = ab.parse_formula("y ~ x + (1 + x | g) + (1 + x || h)")
    refused({"g": ab.LKJ(2), "h": ab.LKJ(2)}, "got ['g', 'h'] ('(1 + x || h)' has no correlation matrix: write '(1 + x | h)' for correlated slopes)", mixed, sd_both)
    with pytest.raises(ab.PriorError, match="eta must be positive"):
        ab.LKJ(0)
    assert str(ab.LKJ(2)) == "LKJ(eta=2)" and ab.LKJ(2).params == {"eta": 2.0}
    assert PRIORS2.cor_priors(cor_spec) == {"g": ab.LKJ(2)} and PRIORS3.cor_priors(ab.parse_formula("y ~ x + w + (1 + x + w | g)")) == {"g": ab.LKJ(2)}
    assert ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1)).cor_priors(one_spec) == {}
    # the sd class is untouched by cor: it still covers every coefficient
    refused({"g": ab.LKJ(2)}, "SD priors for 'g' must cover exactly its coefficients ['Intercept', 'x']; missing ['x']", cor_spec, {"g": {"Intercept": ab.HalfNormal(1)}})


# -- 2. Stan's cholesky_corr transform and the LKJ constant -----------------------------------------------------------
@pytest.mark.parametrize("M", [2, 3, 4])
def test_cholesky_corr_transform_round_trip_and_jacobian(M):
    rng = np.random.default_rng(M)
    y = rng.normal(size=n_free(M))
    L, lj = cholesky_corr_constrain(y, M)
    assert np.allclose(np.tril(L), L) and np.allclose((L ** 2).sum(1), 1.0) and (np.diag(L) > 0).all()
    assert np.allclose(cholesky_corr_free(L), y, atol=1e-12)
    check_correlation_cholesky(L, M, "L")
    # the row-wise strictly-lower entries are the free coordinates; Stan's Jacobian equals log|det dL/dy| by finite differences
    idx = lower_indices(M)
    assert len(idx) == n_free(M) and idx[:3] == [(1, 0), (2, 0), (2, 1)][:len(idx)]
    J = np.zeros((n_free(M), n_free(M)))
    h = 1e-6
    for c in range(n_free(M)):
        e = np.zeros(n_free(M))
        e[c] = h
        Lp, Lm = cholesky_corr_constrain(y + e, M)[0], cholesky_corr_constrain(y - e, M)[0]
        J[:, c] = [(Lp[i, j] - Lm[i, j]) / (2 * h) for i, j in idx]
    assert abs(lj - np.log(abs(np.linalg.det(J)))) < 1e-7
    # the PyTensor transform agrees with the NumPy one: backward, forward and log_jac_det
    tr = CholeskyCorrTransform(M)
    yv, Lv = pt.vector("y"), pt.matrix("L")
    backward, forward, jac = (pytensor.function([yv], tr.backward(yv)), pytensor.function([Lv], tr.forward(Lv)),
                              pytensor.function([yv], tr.log_jac_det(yv)))
    assert np.allclose(backward(y), L, atol=1e-12) and np.allclose(forward(L), y, atol=1e-12) and abs(float(jac(y)) - lj) < 1e-12
    assert tr.name == "cholesky_corr" and tr.ndim_supp == 2
    with pytest.raises(ValueError, match="unit-norm rows"):
        check_correlation_cholesky(np.array([[1.0, 0.0], [0.5, 0.5]]), 2, "L")
    with pytest.raises(ValueError, match="lower triangular"):
        check_correlation_cholesky(np.array([[1.0, 0.3], [0.0, 1.0]]), 2, "L")


def test_lkj_normalizer_against_closed_forms():
    # M = 2: the correlation is 2 Beta(eta, eta) - 1, so c_2(eta) = 2^(2 eta - 1) B(eta, eta)
    for eta in (0.5, 1.0, 2.0, 3.5):
        assert abs(lkj_log_normalizer(eta, 2) - ((2 * eta - 1) * np.log(2) + betaln(eta, eta))) < 1e-12
    # M = 3: the elliptope has volume pi^2 / 2 (eta = 1), and the integral of det(R) over it is 3 pi^2 / 16 (eta = 2)
    assert abs(lkj_log_normalizer(1.0, 3) - np.log(np.pi ** 2 / 2)) < 1e-12
    assert abs(lkj_log_normalizer(2.0, 3) - np.log(3 * np.pi ** 2 / 16)) < 1e-12


def test_lkj_cholesky_draws_are_correlation_cholesky_factors_with_the_right_spread():
    rng = np.random.default_rng(7)
    L = lkj_cholesky_rng(2.0, 3, rng, size=20000)
    assert L.shape == (20000, 3, 3) and np.allclose((L ** 2).sum(-1), 1.0) and np.allclose(np.triu(L, 1), 0.0)
    cor = L @ np.swapaxes(L, -1, -2)
    # every marginal correlation of LKJ(eta) in dimension d is 2 Beta(a, a) - 1 with a = eta - 1 + d / 2: variance 1 / (2 a + 1)
    a = 2.0 - 1 + 1.5
    for i, j in [(1, 0), (2, 0), (2, 1)]:
        assert abs(cor[:, i, j].mean()) < 0.02 and abs((cor[:, i, j] ** 2).mean() - 1 / (2 * a + 1)) < 0.01
    one = lkj_cholesky_rng(2.0, 2, rng)
    assert one.shape == (2, 2) and abs(one[1, 1] - np.sqrt(1 - one[1, 0] ** 2)) < 1e-12


# -- 3. the graph ---------------------------------------------------------------------------------------------------
def test_graph_names_dims_and_value_variables(two, three):
    m = two
    pmm = m.pymc_model
    assert m.correlation_names == {"g": {"L": "L|g", "cor": "cor|g", "coordinate": "g__cor_dim", "coordinate_T": "g__cor_dim_T"}}
    assert m.group_names["g"] == {"sd": "1|g_sigma", "offset": "1|g_offset", "group_effect": "1|g", "coordinate": "g__factor_dim",
                                  "sd:x": "x|g_sigma", "offset:x": "x|g_offset", "group_effect:x": "x|g", "L": "L|g", "cor": "cor|g"}
    assert m.names["L"] == "L|g" and m.names["cor"] == "cor|g" and m.names["group_effect:x"] == "x|g"
    assert sorted(v.name for v in pmm.value_vars) == sorted(["Intercept", "x", "sigma_log__", "1|g_sigma_log__", "1|g_offset",
                                                             "x|g_sigma_log__", "x|g_offset", "L|g_cholesky_corr__"])
    assert pmm.rvs_to_values[pmm["L|g"]].type.shape == (1,) and pmm["L|g"].type.shape == (2, 2)
    assert tuple(pmm.named_vars_to_dims["L|g"]) == tuple(pmm.named_vars_to_dims["cor|g"]) == ("g__cor_dim", "g__cor_dim_T")
    assert list(pmm.coords["g__cor_dim"]) == list(pmm.coords["g__cor_dim_T"]) == ["Intercept", "x"]
    assert tuple(pmm.named_vars_to_dims["x|g"]) == tuple(pmm.named_vars_to_dims["1|g"]) == ("g__factor_dim",)
    assert {d.name for d in pmm.deterministics} == {"1|g", "x|g", "cor|g", "mu"}
    assert m.bambi_model.backend.model is pmm                                        # fit() samples the rebuilt graph
    # the design is the independent model's: the slope block multiplies the raw x
    d, df = m.design, m.data
    ind = indicator(df, "g", LEVELS_G)
    assert np.array_equal(d.group_blocks[("g", "x")].Z, ind * df.x.to_numpy()[:, None]) and d.columns == ("x",)
    # three coefficients: three free coordinates for L, dims over the three coefficients
    p3 = three.pymc_model
    assert p3.rvs_to_values[p3["L|g"]].type.shape == (3,) and list(p3.coords["g__cor_dim"]) == ["Intercept", "x", "w"]
    assert three.names["group_effect:w"] == "w|g" and three.priors.cor_priors(three.spec) == {"g": ab.LKJ(2)}


# -- 4. value_point and the density -----------------------------------------------------------------------------------
def natural2(k=0):
    L = cholesky_corr_constrain([0.4 - 0.2 * k], 2)[0]
    return dict(intercept=0.3 + 0.1 * k, b=np.array([0.5]) + 0.05 * k,
                sd={"g": {"Intercept": 0.8 - 0.1 * k, "x": 0.3 + 0.05 * k}},
                z={"g": {"Intercept": np.array([-0.4, 0.2, 0.9, -0.1]) + 0.1 * k, "x": np.array([0.7, -0.5, 0.1, 0.3]) - 0.1 * k}},
                L={"g": L}, sigma=0.9 + 0.1 * k)


def reference_logp2(m, nat, jacobian):
    """The two-coefficient model at natural arguments: r = diag(sd) L z per level, the LKJ term with its constant, Stan's Jacobian."""
    df, d = m.data, m.design
    x = df.x.to_numpy(dtype=float)
    ind = indicator(df, "g", LEVELS_G)
    sd_u, sd_v = nat["sd"]["g"]["Intercept"], nat["sd"]["g"]["x"]
    z1, z2, L = nat["z"]["g"]["Intercept"], nat["z"]["g"]["x"], nat["L"]["g"]
    u = sd_u * z1
    v = sd_v * (L[1, 0] * z1 + L[1, 1] * z2)
    eta = nat["intercept"] + d.X_model @ nat["b"] + ind @ u + x * (ind @ v)
    eta_lkj = 2.0
    lkj = -((2 * eta_lkj - 1) * np.log(2) + betaln(eta_lkj, eta_lkj)) + (2 * eta_lkj - 2) * np.log(L[1, 1])
    lp = (norm.logpdf(nat["intercept"], 0.2, 5) + norm.logpdf(nat["b"], 0.1, 2).sum() + norm.logpdf(z1).sum() + norm.logpdf(z2).sum()
          + halfnorm.logpdf(sd_u, scale=1) + halfnorm.logpdf(sd_v, scale=0.5) + halfnorm.logpdf(nat["sigma"], scale=1.5)
          + lkj + norm.logpdf(df.y.to_numpy(), eta, nat["sigma"]).sum())
    if jacobian:
        y_L = np.arctanh(L[1, 0])
        lp += np.log(sd_u) + np.log(sd_v) + np.log(nat["sigma"]) + np.log(1 - np.tanh(y_L) ** 2)
    return float(lp), eta, v


def test_value_point_and_density_two_coefficients(two):
    m = two
    pmm = m.pymc_model
    logp, nojac = pmm.compile_logp(), pmm.compile_logp(jacobian=False)
    dets = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["mu"], pmm["x|g"], pmm["cor|g"]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    for k in range(3):
        nat = natural2(k)
        point = m.value_point(**nat)
        assert set(point) == {v.name for v in pmm.value_vars}
        assert np.allclose(point["L|g_cholesky_corr__"], [np.arctanh(nat["L"]["g"][1, 0])])      # Stan's coordinate: arctanh of the partial correlation
        ref, eta, v = reference_logp2(m, nat, True)
        assert abs(float(logp(point)) - ref) < 1e-8 and abs(float(nojac(point)) - reference_logp2(m, nat, False)[0]) < 1e-8
        out = dets(point)
        assert np.allclose(np.asarray(out[0]).ravel(), eta, atol=1e-12) and np.allclose(np.asarray(out[1]), v, atol=1e-12)
        assert np.allclose(np.asarray(out[2]), nat["L"]["g"] @ nat["L"]["g"].T, atol=1e-12)
    # the bare matrix for the single correlated factor is the mapping
    nat = natural2()
    alt = m.value_point(**{**nat, "L": nat["L"]["g"]})
    assert all(np.array_equal(alt[k], v) for k, v in m.value_point(**nat).items())


def test_value_point_refusals(two):
    m, nat = two, natural2()

    def refused(message, **override):
        with pytest.raises(ValueError, match=re.escape(message)):
            m.value_point(**{**nat, **override})

    refused("L (the Cholesky factor of the correlation matrix) is required for the correlated factors ['g']", L=None)
    refused("L must cover exactly the correlated grouping factors ['g']; got ['h']", L={"h": nat["L"]["g"]})
    refused("L for 'g' must be a finite 2 x 2 matrix", L={"g": np.eye(3)})
    refused("L for 'g' must be lower triangular", L={"g": np.array([[1.0, 0.3], [0.0, 1.0]])})
    refused("L for 'g' must have unit-norm rows", L={"g": np.array([[1.0, 0.0], [0.6, 0.6]])})
    refused("L for 'g' must have a positive diagonal", L={"g": np.array([[1.0, 0.0], [0.6, -0.8]])})
    ind = ab.gaussian("y ~ x + (1 + x || g)", frame(), ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5), sd=SD2))
    with pytest.raises(ValueError, match=re.escape("this model has no correlated group term '(1 + x | g)'; L does not apply")):
        ind.value_point(**nat)


def test_density_three_coefficients(three):
    """Three correlated coefficients: r = diag(sd) L z with a 3 x 3 L, the LKJ term's closed-form constant (3 pi^2 / 16 at
    eta = 2), Stan's Jacobian from the NumPy transform."""
    m = three
    pmm = m.pymc_model
    logp, nojac = pmm.compile_logp(), pmm.compile_logp(jacobian=False)
    df, d = m.data, m.design
    x, w = df.x.to_numpy(dtype=float), df.w.to_numpy(dtype=float)
    ind = indicator(df, "g", LEVELS_G)
    y_L = np.array([0.3, -0.5, 0.8])
    L, jac_L = cholesky_corr_constrain(y_L, 3)
    sds = np.array([0.8, 0.3, 0.5])
    Z = np.array([[-0.4, 0.2, 0.9, -0.1], [0.7, -0.5, 0.1, 0.3], [0.2, 0.4, -0.6, 0.5]])
    nat = dict(intercept=0.3, b=np.array([0.5, -0.2]), sd={"g": dict(zip(("Intercept", "x", "w"), sds))},
               z={"g": dict(zip(("Intercept", "x", "w"), Z))}, L={"g": L}, sigma=0.9)
    point = m.value_point(**nat)
    assert np.allclose(point["L|g_cholesky_corr__"], y_L)
    R = (np.diag(sds) @ L @ Z)                                               # (3, J): r_c[j] = sd_c * sum_k L[c, k] z_k[j]
    eta = nat["intercept"] + d.X_model @ nat["b"] + ind @ R[0] + x * (ind @ R[1]) + w * (ind @ R[2])
    lkj = -np.log(3 * np.pi ** 2 / 16) + (3 - 2 + 2 * 2 - 2) * np.log(L[1, 1]) + (3 - 3 + 2 * 2 - 2) * np.log(L[2, 2])
    ref = (norm.logpdf(0.3, 0.2, 5) + norm.logpdf(nat["b"], 0.1, 2).sum() + norm.logpdf(Z).sum()
           + halfnorm.logpdf(sds[0], scale=1) + halfnorm.logpdf(sds[1], scale=0.5) + halfnorm.logpdf(sds[2], scale=0.7)
           + halfnorm.logpdf(0.9, scale=1.5) + lkj + norm.logpdf(df.y.to_numpy(), eta, 0.9).sum())
    assert abs(float(nojac(point)) - ref) < 1e-8
    assert abs(float(logp(point)) - (ref + np.log(sds).sum() + np.log(0.9) + jac_L)) < 1e-8
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["1|g"], pmm["x|g"], pmm["w|g"], pmm["cor|g"]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    out = fn(point)
    assert all(np.allclose(np.asarray(o), R[c], atol=1e-12) for c, o in enumerate(out[:3])) and np.allclose(np.asarray(out[3]), L @ L.T, atol=1e-12)


def test_two_correlated_factors_one_derived():
    """``(1 + x | g/h)`` is brms's two correlated terms ``(1 + x | g) + (1 + x | g:h)``, each with its own ``L`` and
    ``eta``: names, and the density at a point against the closed form with both LKJ terms and both Jacobians."""
    df = frame()
    sd = {"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(0.5)}, "g:h": {"Intercept": ab.HalfNormal(0.8), "x": ab.HalfNormal(0.4)}}
    m = ab.gaussian("y ~ x + (1 + x | g/h)", df, ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5),
                                                           sd=sd, cor={"g": ab.LKJ(2), "g:h": ab.LKJ(1)}))
    assert m.correlation_names["g:h"] == {"L": "L|g_h", "cor": "cor|g_h", "coordinate": "g_h__cor_dim", "coordinate_T": "g_h__cor_dim_T"}
    assert {"L|g_cholesky_corr__", "L|g_h_cholesky_corr__"} <= {v.name for v in m.pymc_model.value_vars}
    d, rng = m.design, np.random.default_rng(5)
    etas, ys, sds = {"g": 2.0, "g:h": 1.0}, {"g": 0.4, "g:h": -0.7}, {"g": (0.8, 0.3), "g:h": (0.6, 0.2)}
    nat = dict(intercept=0.3, b=np.array([0.5]), sigma=0.9, sd={}, z={}, L={})
    for F in ("g", "g:h"):
        J = d.group_blocks[(F, "Intercept")].J
        nat["sd"][F] = dict(zip(("Intercept", "x"), sds[F]))
        nat["z"][F] = {"Intercept": rng.normal(size=J), "x": rng.normal(size=J)}
        nat["L"][F] = cholesky_corr_constrain([ys[F]], 2)[0]
    point = m.value_point(**nat)
    eta = nat["intercept"] + d.X_model @ nat["b"]
    ref = norm.logpdf(0.3, 0.2, 5) + norm.logpdf(0.5, 0.1, 2) + halfnorm.logpdf(0.9, scale=1.5)
    jac = np.log(0.9)
    for F, scales in {"g": (1, 0.5), "g:h": (0.8, 0.4)}.items():
        (s0, s1), (z0, z1), L, e = sds[F], nat["z"][F].values(), nat["L"][F], etas[F]
        eta = eta + d.group_blocks[(F, "Intercept")].Z @ (s0 * z0) + d.group_blocks[(F, "x")].Z @ (s1 * (L[1, 0] * z0 + L[1, 1] * z1))
        ref += (norm.logpdf(z0).sum() + norm.logpdf(z1).sum() + halfnorm.logpdf(s0, scale=scales[0]) + halfnorm.logpdf(s1, scale=scales[1])
                - ((2 * e - 1) * np.log(2) + betaln(e, e)) + (2 * e - 2) * np.log(L[1, 1]))
        jac += np.log(s0) + np.log(s1) + np.log1p(-L[1, 0] ** 2)
    ref += norm.logpdf(df.y.to_numpy(), eta, 0.9).sum()
    pmm = m.pymc_model
    assert abs(float(pmm.compile_logp(jacobian=False)(point)) - ref) < 1e-8
    assert abs(float(pmm.compile_logp()(point)) - (ref + jac)) < 1e-8


FAMILIES = {   # family: (the formula's response side, the family's scale prior, constructor kwargs)
    "gaussian": ("y", {"sigma": ab.HalfNormal(1)}, {}),
    "student": ("y", {"sigma": ab.HalfNormal(1)}, {"nu": 4.0}),
    "lognormal": ("ypos", {"sigma": ab.HalfNormal(1)}, {}),
    "bernoulli": ("yb", {}, {}),
    "binomial": ("k | trials(n)", {}, {}),
    "poisson": ("c", {}, {}),
    "negbinomial": ("c", {"shape": ab.Gamma(2, 0.5)}, {}),
    "gamma": ("ypos", {"shape": ab.Gamma(2, 0.5)}, {}),
    "beta": ("yu", {"phi": ab.Gamma(2, 0.5)}, {}),
}


@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_family_keeps_its_likelihood_through_the_graph_edit(family):
    """The correlated model at ``(z, L)`` is the independent model at ``z' = L z`` (per level) up to the priors that
    differ: ``log p_cor(z, L) - log p_ind(z') = sum log N(z) - sum log N(z') + lkj(L)``, plus Stan's Jacobian
    ``log(1 - L[1, 0]^2)`` on the sampled scale; the slope effects and the linear predictor agree. So the graph edit
    keeps every family's likelihood (custom ones included) and changes only the slope's construction."""
    lhs, scale, kwargs = FAMILIES[family]
    rng = np.random.default_rng(12)
    df = frame()
    n = rng.integers(3, 9, len(df))
    df = df.assign(ypos=np.exp(rng.normal(size=len(df))), yb=rng.integers(0, 2, len(df)), n=n, k=rng.binomial(n, 0.4),
                   c=rng.poisson(2.0, len(df)), yu=rng.uniform(0.1, 0.9, len(df)))
    common = dict(intercept=ab.Normal(0.2, 2), b=ab.Normal(0.1, 2), sd=SD2, **scale)
    build = getattr(ab, family)
    cor = build(f"{lhs} ~ x + (1 + x | g)", df, ab.Priors(**common, cor={"g": ab.LKJ(1.5)}), **kwargs)
    ind = build(f"{lhs} ~ x + (1 + x || g)", df, ab.Priors(**common), **kwargs)
    J = cor.design.J
    z0, z1 = rng.normal(size=J), rng.normal(size=J)
    L = cholesky_corr_constrain([0.6], 2)[0]
    aux = {k: 0.8 for k in scale}
    nat = dict(intercept=0.3, b=np.array([0.4]), sd={"g": {"Intercept": 0.7, "x": 0.4}}, **aux)
    pc = cor.value_point(**nat, z={"g": {"Intercept": z0, "x": z1}}, L={"g": L})
    pi = ind.value_point(**nat, z={"g": {"Intercept": z0, "x": L[1, 0] * z0 + L[1, 1] * z1}})
    lkj = 0.5 * np.log(1 - L[1, 0] ** 2) - (2 * np.log(2) + betaln(1.5, 1.5))           # (eta - 1) log(1 - r^2) - log c_2(eta)
    prior_gap = norm.logpdf(z1).sum() - norm.logpdf(L[1, 0] * z0 + L[1, 1] * z1).sum() + lkj
    c_pm, i_pm = cor.pymc_model, ind.pymc_model
    assert abs(float(c_pm.compile_logp(jacobian=False)(pc)) - float(i_pm.compile_logp(jacobian=False)(pi)) - prior_gap) < 1e-8
    assert abs(float(c_pm.compile_logp()(pc)) - float(i_pm.compile_logp()(pi)) - prior_gap - np.log1p(-L[1, 0] ** 2)) < 1e-8
    outs = []
    for m, point in ((cor, pc), (ind, pi)):
        pmm = m.pymc_model
        fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[m.names["linpred"]], pmm["x|g"]]), inputs=pmm.value_vars,
                            point_fn=True, on_unused_input="ignore")
        outs.append([np.asarray(o, dtype=float) for o in fn(point)])
    assert all(np.allclose(a, b, atol=1e-12) for a, b in zip(*outs))


def test_jax_density_equals_pytensor(two):
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    pmm = two.pymc_model
    jf, logp = get_jaxified_logp(pmm), pmm.compile_logp()
    names = [v.name for v in pmm.value_vars]
    for k in range(2):
        point = two.value_point(**natural2(k))
        assert abs(float(jf([point[n] for n in names])) - float(logp(point))) < 1e-8


# -- 5. predictions -----------------------------------------------------------------------------------------------------
def synthetic(model, draws=5, seed=8):
    """A posterior whose draws are distinct constants, with a correlation Cholesky factor per draw."""
    rng = np.random.default_rng(seed)
    J = model.design.J
    shape = (1, draws)
    L = lkj_cholesky_rng(2.0, 2, rng, size=shape)
    var = {"Intercept": (("chain", "draw"), rng.normal(1.0, 0.1, shape)), "x": (("chain", "draw"), rng.normal(0.5, 0.1, shape)),
           "1|g_sigma": (("chain", "draw"), rng.uniform(0.3, 0.7, shape)), "x|g_sigma": (("chain", "draw"), rng.uniform(0.2, 0.4, shape)),
           "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.5, shape + (J,))), "x|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.3, shape + (J,))),
           "L|g": (("chain", "draw", "g__cor_dim", "g__cor_dim_T"), L), "cor|g": (("chain", "draw", "g__cor_dim", "g__cor_dim_T"), L @ np.swapaxes(L, -1, -2)),
           "sigma": (("chain", "draw"), rng.uniform(0.3, 0.5, shape))}
    return az.InferenceData(posterior=xr.Dataset(var, coords={"chain": [0], "draw": np.arange(draws), "g__factor_dim": list(model.design.levels),
                                                              "g__cor_dim": ["Intercept", "x"], "g__cor_dim_T": ["Intercept", "x"]}))


def test_predictions_on_fitted_rows_agree_with_the_bambi_path(two):
    m, idata = two, synthetic(two)
    P = idata.posterior
    a, b, U, V = P["Intercept"].to_numpy(), P["x"].to_numpy(), P["1|g"].to_numpy(), P["x|g"].to_numpy()
    x, gi = m.data.x.to_numpy(dtype=float), m.design.group_index
    eta = a[..., None] + b[..., None] * x + U[..., gi] + x * V[..., gi]
    assert np.allclose(m.predict(idata, kind="linear")["eta"].to_numpy(), eta, atol=1e-12)
    assert np.allclose(m.predict(idata, kind="mean")["mu"].to_numpy(), eta, atol=1e-10)             # Bambi's path reads the posterior's effects
    assert np.allclose(m.predict(idata, kind="mean", include_group_effects=False)["mu"].to_numpy(), a[..., None] + b[..., None] * x, atol=1e-12)
    resp = m.predict(idata, kind="response", seed=4)
    assert np.allclose(resp["mu"].to_numpy(), eta, atol=1e-10) and resp["y"].shape == eta.shape


@pytest.mark.parametrize("policy", ["population", "resample_observed"])
def test_predictions_for_unseen_levels(two, policy):
    m, idata = two, synthetic(two)
    P = idata.posterior
    a, b, U, V, L = (P[k].to_numpy() for k in ("Intercept", "x", "1|g", "x|g", "L|g"))
    sd_u, sd_v = P["1|g_sigma"].to_numpy(), P["x|g_sigma"].to_numpy()
    levels = list(m.design.levels)
    new = pd.DataFrame({"x": [0.0, 1.0, 1.0, 3.0, 2.0], "g": [levels[1], "newA", "newA", "newA", "newB"]})
    seed = 11
    out = m.predict(idata, newdata=new, kind="linear", new_groups=policy, seed=seed)
    eta = out["eta"].to_numpy()
    assert out.attrs["new_groups"] == policy and list(out.attrs["new_levels"]) == ["newA", "newB"]
    assert np.allclose(eta[..., 0], a + U[..., 1], atol=1e-12)
    rng = np.random.default_rng(seed)
    pairs = {}
    for lvl in ("newA", "newB"):
        if policy == "population":                     # z per coefficient, then diag(sd) L z: the intercept's draw feeds the slope
            z1, z2 = rng.normal(0.0, 1.0, size=sd_u.shape), rng.normal(0.0, 1.0, size=sd_v.shape)
            pairs[lvl] = (sd_u * z1, sd_v * (L[..., 1, 0] * z1 + L[..., 1, 1] * z2))
        else:                                          # one observed level's whole vector
            chosen = rng.integers(0, len(levels), size=sd_u.shape)
            pairs[lvl] = (np.take_along_axis(U, chosen[..., None], -1)[..., 0], np.take_along_axis(V, chosen[..., None], -1)[..., 0])
    for i in range(1, 5):
        u_star, v_star = pairs[new.g[i]]
        assert np.allclose(eta[..., i], a + b * new.x[i] + u_star + new.x[i] * v_star, atol=1e-12)
    assert np.allclose(eta[..., 1], eta[..., 2]) and not np.allclose(eta[..., 4] - a - 2 * b, eta[..., 1] - a - b)
    # kind="mean" goes by hand too, and the default refuses unseen levels
    assert np.allclose(m.predict(idata, newdata=new, kind="mean", new_groups=policy, seed=seed)["mu"].to_numpy(), eta, atol=1e-12)
    with pytest.raises(ab.FormulaError, match="not seen in training"):
        m.predict(idata, newdata=new, kind="mean")


def test_population_policy_uses_the_same_random_numbers_as_the_independent_model():
    """With L the identity the correlated draw is the independent one, so the independent model's draws are unchanged."""
    df = frame()
    cor = ab.gaussian("y ~ x + (1 + x | g)", df, PRIORS2)
    ind = ab.gaussian("y ~ x + (1 + x || g)", df, ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5), sd=SD2))
    idata = synthetic(cor)
    eye = np.broadcast_to(np.eye(2), idata.posterior["L|g"].shape).copy()
    idata.posterior["L|g"] = (idata.posterior["L|g"].dims, eye)
    new = pd.DataFrame({"x": [1.0, -2.0], "g": ["newA", "newB"]})
    a = cor.predict(idata, newdata=new, kind="linear", new_groups="population", seed=3)["eta"].to_numpy()
    b = ind.predict(idata, newdata=new, kind="linear", new_groups="population", seed=3)["eta"].to_numpy()
    assert np.allclose(a, b, atol=1e-12)


# -- 6. the fit record, the target identity and describe() ------------------------------------------------------------
def test_fit_record_identity_and_describe(two, three):
    m = two
    idata = synthetic(m, draws=6)
    idata.posterior = xr.concat([idata.posterior, idata.posterior.assign_coords(chain=[1])], dim="chain")
    idata.attrs[FIT_IDENTITY_ATTR] = json.dumps(m.target_identity, sort_keys=True)
    fit = SimpleNamespace(idata=idata, backend="pymc", device_requested="cpu", device_actual="cpu", platform="cpu", dtype="float64",
                          chain_method=None, chains=2, draws=6, tune=0, target_accept=0.9, seed=1, wall_seconds=0.1, notes=[], target_identity=None)
    panel = {"values": {"divergences": 0.0}, "status": {"divergences": "pass"}, "overall": "pass"}
    rec = ab.fit_record(m, fit, panel=panel)
    json.dumps(rec)
    (grp,) = rec["model"]["groups"]
    assert grp["correlated"] is True and grp["cor_prior"] == "LKJ(eta=2)" and grp["correlation_names"] == m.correlation_names["g"]
    assert grp["slopes"] == [{"coef": "x", "term": "x|g", "names": m.group_coefficient_names[("g", "x")], "sd_prior": "HalfNormal(sigma=0.5)"}]
    assert rec["model"]["priors"] == {"intercept": "Normal(mu=0.2, sigma=5)", "b:x": "Normal(mu=0.1, sigma=2)", "sd": "HalfNormal(sigma=1)",
                                      "sd:x|g": "HalfNormal(sigma=0.5)", "cor:g": "LKJ(eta=2)", "sigma": "HalfNormal(sigma=1.5)"}
    assert rec["model"]["canonical_formula"] == "y ~ x + (1 + x | g)"
    # the independent model and a different eta are different targets; the same call is the same target
    ind = ab.gaussian("y ~ x + (1 + x || g)", frame(), ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5), sd=SD2))
    other = ab.gaussian("y ~ x + (1 + x | g)", frame(), ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sigma=ab.HalfNormal(1.5), sd=SD2, cor={"g": ab.LKJ(1)}))
    same = ab.gaussian("y ~ x + (1 + x | g)", frame(), PRIORS2)
    assert m.target_identity != ind.target_identity != other.target_identity and m.target_identity == same.target_identity
    text = m.describe()
    for s in ["AutoBayes M0 model with correlated varying slope(s)",
              "mu_i = a_c + (x_x - mean(x_x)) * b_x + u[g_i] + x_x,i * v_x|g[g_i]",
              "u_g[j] = sd_g * z_g[j], z_g[j] ~ Normal(0, 1), j = 0..3",
              "v_x|g[j] = sd_x|g * (L_g[1,0] * z_g[j] + L_g[1,1] * z_x|g[j]), z_x|g[j] ~ Normal(0, 1), j = 0..3: the slope of x varies by g ('|': correlated with u_g)",
              "(u_g[j], v_x|g[j]) = diag(sd_g, sd_x|g) L_g (z_g[j], z_x|g[j]): L_g is the Cholesky factor of the correlation matrix",
              "Cor_g = L_g L_g^T of the coefficients (Intercept, x) (brms's scale_r_cor; PyMC cor|g)",
              "the correlation is a statement", "at x = 0",
              "sd_g ~ HalfNormal(sigma=1)", "sd_x|g ~ HalfNormal(sigma=0.5)",
              "Cor_g ~ LKJ(eta=2) (brms lkj(2) on the correlation matrix, placed on L_g as lkj_corr_cholesky)",
              "L|g_cholesky_corr__      L_g on Stan's cholesky_corr coordinates (canonical partial correlations, arctanh scale)",
              "Deterministic: 1|g = u, x|g = v_x|g, cor|g = Cor_g = L_g L_g^T, mu = linear predictor"]:
        assert s in text, s
    assert "independent varying slope" not in text and "'||'" not in text
    t3 = three.describe()
    assert "v_w|g[j] = sd_w|g * (L_g[2,0] * z_g[j] + L_g[2,1] * z_x|g[j] + L_g[2,2] * z_w|g[j])" in t3
    assert "(u_g[j], v_x|g[j], v_w|g[j]) = diag(sd_g, sd_x|g, sd_w|g) L_g (z_g[j], z_x|g[j], z_w|g[j])" in t3
    assert "Deterministic: 1|g = u, x|g = v_x|g, w|g = v_w|g, cor|g = Cor_g = L_g L_g^T, mu = linear predictor" in t3


def test_prior_predictive_draws_through_the_rebuilt_graph(two):
    out = two.prior_predictive(draws=6, seed=5)
    assert out.prior_predictive["y"].shape == (1, 6, two.design.n)
    L = out.prior["L|g"].to_numpy()[0]
    assert L.shape == (6, 2, 2) and np.allclose((L ** 2).sum(-1), 1.0) and np.allclose(out.prior["cor|g"].to_numpy()[0], L @ np.swapaxes(L, -1, -2))
    # the slope's effect in each draw is the correlated construction
    sd_v, z1, z2 = (out.prior[k].to_numpy()[0] for k in ("x|g_sigma", "1|g_offset", "x|g_offset"))
    assert np.allclose(out.prior["x|g"].to_numpy()[0], sd_v[:, None] * (L[:, 1, 0, None] * z1 + L[:, 1, 1, None] * z2), atol=1e-12)
