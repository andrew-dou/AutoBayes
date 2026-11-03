"""brms's gr(g, cov = A) (a known covariance of a grouping factor's levels, A from data2), gr(g, dist = "student"), varying slopes
in mm(), mmc(x1, x2) and mm(..., scale = FALSE) (docs/gr-cov-dist.md; divergences D185, D186).

The term model and every refusal, the reading and checks of A, the design blocks and PyMC names, ``value_point`` and the log
density against hand computations (the Kronecker product A kron Sigma checked on the graph's map from z to the effects, and the
frozen gauss_gr_cov_slope_signed case, where brms 2.23.0 drops the negative entries of chol(A), against both constructions), the
Student-t scales and df's truncated prior, the multi-membership slopes with each member's covariate and the raw weights,
predictions by hand and the refusal of new levels, the default priors and their table, ``describe()``, the target identity and
the graph checks the model runs at build. Build-only; nothing samples. The numbers against brms are the oracle's (tests/oracle:
gauss_gr_cov, gauss_gr_cov_slope, gauss_gr_cov_slope_signed, gauss_gr_student, gauss_mm_slope, gauss_mmc, gauss_mm_scale_false)."""
import dataclasses
import math
import re
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats
from scipy.integrate import quad
from scipy.special import betaln

import autobayes as ab
from autobayes.formula import GroupTerm, mm_weights, mmc_columns, parse_formula
from autobayes.group_cov import known_covariance

LEVELS = ("a", "b", "c", "d", "e")
# a relationship matrix of the five levels with small negative entries (the first five levels of tests/oracle/datasets.py's
# g_relationship): positive definite, its lower Cholesky factor has negative entries
A = pd.DataFrame([[1.00, 0.50, 0.50, 0.05, -0.10],
                  [0.50, 1.00, 0.10, 0.00, 0.05],
                  [0.50, 0.10, 1.00, -0.05, 0.00],
                  [0.05, 0.00, -0.05, 0.95, 0.25],
                  [-0.10, 0.05, 0.00, 0.25, 1.05]], index=list(LEVELS), columns=list(LEVELS))


def frame(seed=11, n=40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = np.array([LEVELS[i % 5] for i in range(n)])
    g2 = rng.choice(LEVELS, n)
    df = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n), "w": rng.normal(size=n), "g": g, "g2": g2,
                       "x1": rng.normal(size=n), "x2": rng.normal(size=n), "w1": rng.uniform(0.2, 1.2, n),
                       "w2": rng.uniform(0.2, 1.2, n), "wn": rng.uniform(-0.5, 1.0, n), "cnt": rng.poisson(2.0, n),
                       "h": rng.choice(["p", "q", "r"], n)})
    df["gby"] = np.where(np.isin(df.g, ["a", "b"]), "u", "v")
    return df


HN = ab.HalfNormal(1)
BASE = dict(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1))
COV = "y ~ x + (1 | gr(g, cov = A))"
COV_SLOPE = "y ~ x + (1 + x | gr(g, cov = A))"
STUDENT = 'y ~ x + (1 | gr(g, dist = "student"))'
MM_SLOPE = "y ~ x + (1 + x | mm(g, g2, weights = cbind(w1, w2)))"
MMC = "y ~ x + (1 + mmc(x1, x2) || mm(g, g2))"
UNSCALED = "y ~ x + (1 | mm(g, g2, weights = cbind(wn, w2), scale = FALSE))"


def slope_priors(factor, coefs=("Intercept", "x"), correlated=True, **extra):
    return ab.Priors(**BASE, sd={factor: {c: HN for c in coefs}}, cor={factor: ab.LKJ(2)} if correlated else None, **extra)


def lkj2(L):
    """Stan's lkj_corr_cholesky_lpdf(L | 2) of a 2 x 2 factor: the density (1 - r^2)^(eta - 1) / (2^(2 eta - 1) B(eta, eta)) of
    the correlation r = L[1, 0], written on L[1, 1] = sqrt(1 - r^2)."""
    eta = 2.0
    return -(2 * eta - 1) * np.log(2.0) - betaln(eta, eta) + (2 * eta - 2) * np.log(L[1, 1])


def lower(r):
    return np.array([[1.0, 0.0], [r, np.sqrt(1 - r * r)]])


# -- the term model --------------------------------------------------------------------------------------------------------
def test_the_term_model_and_its_spellings():
    t = parse_formula(COV).group_terms[0]
    assert t == GroupTerm("g", ("Intercept",), True, cov="A") and t.custom and not t.is_student and t.grouping == "gr(g, cov = A)"
    assert t.canonical == "(1 | gr(g, cov = A))" and parse_formula(COV).data2_names == ("A",)
    s = parse_formula(COV_SLOPE.replace("|", "||") + " + (1 | gr(h, cov = A))")
    assert s.group_terms[0].canonical == "(1 + x || gr(g, cov = A))" and not s.group_terms[0].correlated
    assert s.data2_names == ("A",)                                   # one matrix read by two terms is named once
    st = parse_formula(STUDENT).group_terms[0]
    assert st.dist == "student" and st.is_student and st.custom and st.canonical == '(1 | gr(g, dist = "student"))'
    assert parse_formula('y ~ x + (1 | gr(g, dist = "gaussian"))').group_terms[0] == GroupTerm("g")   # brms's default: the plain term
    mm = parse_formula(MM_SLOPE).group_terms[0]
    assert mm == GroupTerm("mmgg2", ("Intercept", "x"), True, members=("g", "g2"), weights=("w1", "w2"))
    assert mm.covariate_columns("Intercept") == (None, None) and mm.covariate_columns("x") == ("x", "x")
    mc = parse_formula(MMC.replace("1 + ", "")).group_terms[0]      # the intercept is implied, as in brms
    assert mc.coefs == ("Intercept", "mmc(x1, x2)") and not mc.correlated and mc.canonical == "(1 + mmc(x1, x2) || mm(g, g2))"
    assert mc.covariate_columns("mmc(x1, x2)") == ("x1", "x2") and mc.data_columns == ("g", "g2", "x1", "x2")
    assert mmc_columns("mmc(x1, x2)") == ("x1", "x2") and mmc_columns("x") is None
    un = parse_formula(UNSCALED).group_terms[0]
    assert un.unscaled and un.grouping == "mm(g, g2, weights = cbind(wn, w2), scale = FALSE)"
    assert not parse_formula("y ~ x + (1 | mm(g, g2, scale = TRUE))").group_terms[0].unscaled
    assert parse_formula("y ~ x + (1 || mm(g, g2))").group_terms[0].correlated     # one coefficient: '||' is '|'
    assert parse_formula("y ~ x + (1 | g)").group_terms[0].covariate_columns("Intercept") == (None,)
    with pytest.raises(ValueError, match="dist is 'student' or None"):
        GroupTerm("g", dist="cauchy")


REFUSALS = [
    ("y ~ x + (1 | gr(g, cov = A, by = gby))", "gr()'s by and cov together are not supported"),
    ('y ~ x + (1 | gr(g, cov = A, dist = "student"))', "gr()'s cov and dist together are not supported"),
    ('y ~ x + (1 + x | gr(g, dist = "student", id = "p"))', "a correlation ID with gr(dist = ) is not supported"),
    ("y ~ x + (1 | gr(g, cov = 2))", "gr()'s cov = takes the name of a matrix in data2"),
    ("y ~ x + (1 | gr(g, dist = student))", "gr()'s dist = takes \"gaussian\" or \"student\""),
    ("y ~ x + (1 | gr(g, foo = 1))", "brms's gr() options ['foo'] are not supported"),
    ("y ~ x + (0 + x | mm(g, g2))", "a multi-membership term without its varying intercept is not supported"),
    ("y ~ x + (x - 1 | mm(g, g2))", "a multi-membership term without its varying intercept is not supported"),
    ("y ~ x + (1 + | mm(g, g2))", "empty term inside the group term"),
    ("y ~ x + (1 + 1 + x | mm(g, g2))", "'1' appears more than once"),
    ("y ~ x + (1 | mm(g, g2, scale = F))", "mm()'s scale = takes TRUE or FALSE"),
    ("y ~ x + (1 | mm(g, g2, dist = 'student'))", "mm()'s argument 'dist' is not supported"),
    ("y ~ x + (1 + mmc(x1) | mm(g, g2))", "has 1 columns for the 2 members of mm() (brms: 'Expected 2 coefficients but found 1')"),
    ("y ~ x + (1 + mmc(x1, 2) | mm(g, g2))", "mmc() takes column names here"),
    ("y ~ x + (1 + mmc(x1, y) | mm(g, g2))", "the response 'y' cannot be a column of 'mmc(x1, y)'"),
    ("y ~ x + (1 + mmc(x1, g) | mm(g, g2))", "'g', a member or weight column of mm(), cannot be a column of 'mmc(x1, g)'"),
    ("y ~ x + (1 + log(x) | mm(g, g2))", "'log(x)' cannot vary by a multi-membership term"),
    ("y ~ x + w + x:w + (1 + x:w | mm(g, g2))", "'x:w' cannot vary by a multi-membership term"),
    ("y ~ x + g + (1 + g | mm(g, g2))", "the member or weight column 'g' of mm() cannot also be a varying slope"),
    ("y ~ x + (1 + y | mm(g, g2))", "the response 'y' cannot be a varying slope"),
    ("y ~ x + (1 + x + x | mm(g, g2))", "duplicated varying slope 'x'"),
    ("y ~ x + (0 + x | gr(g, cov = A))", "'(0 + x | gr(g, cov = A))' without its varying intercept is not supported"),
    ('y ~ x + I(x^2) + (1 + I(x^2) | gr(g, dist = "student"))', "the varying slope 'I(x^2)' of '(1 + I(x^2) | gr(g, dist = \"student\"))'"
                                                               " is not supported: in mm(), gr(cov = ) and gr(dist = \"student\") a varying "
                                                               "slope is a numeric column"),
    ("y ~ x + (1 + mmc(x1, x2) | g)", "mmc() is only supported in multi-membership terms"),
    ("y ~ x + (1 + mmc(h, x2) | mm(g, g2)) + (1 | h)", "'h', a column of an mmc() coefficient, cannot also be a grouping column"),
    ("y ~ x + mo(cnt) + (1 | gr(g, cov = A))", "the term with gr(cov = ) '(1 | gr(g, cov = A))' next to the monotonic term"),
    ('y ~ x + mo(cnt) + (1 | gr(g, dist = "student"))', "next to the monotonic term 'mo(cnt)' is not supported: no frozen brms case "
                                                         "verifies the pair (docs/gr-cov-dist.md, divergence D186)"),
]


@pytest.mark.parametrize("formula,message", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusals_name_the_construct(formula, message):
    with pytest.raises(ab.FormulaError, match=re.escape(message)):
        parse_formula(formula)


def test_refused_for_other_families_and_model_kinds():
    with pytest.raises(ab.FormulaError, match=re.escape("the term with gr(dist = \"student\") '(1 | gr(g, dist = \"student\"))' is "
                                                        "not available for the bernoulli family (logit link)")):
        parse_formula('ybin ~ x + (1 | gr(g, dist = "student"))', family="bernoulli")
    with pytest.raises(ab.FormulaError, match=re.escape("multi-membership term '(1 + x | mm(g, g2))' is not available for the "
                                                        "bernoulli family (logit link): it is added to the linear predictor as the "
                                                        "library builds it and checked there under the identity and log links, for the "
                                                        "families whose linear predictor takes such terms (docs/gr-cov-dist.md, "
                                                        "divergence D186)")):
        parse_formula("ybin ~ x + (1 + x | mm(g, g2))", family="bernoulli")
    with pytest.raises(ab.FormulaError, match=r"in a multivariate formula is not supported"):
        ab.gaussian("mvbind(y, w) ~ x + (1 | gr(g, cov = A))", frame(), ab.Priors(resp={}), rescor=False)
    with pytest.raises(ab.PriorError, match=r"df is the prior of the degrees of freedom .* a multivariate model takes none"):
        ab.Priors(resp={}, df=ab.Gamma(2, 0.1)).validate_multivariate(SimpleNamespace(responses=("y", "w"), rescor=False))


# -- the known covariance ----------------------------------------------------------------------------------------------------
def test_known_covariance_is_read_and_checked():
    t = parse_formula(COV).group_terms[0]
    cov, notes = known_covariance(t, LEVELS, {"A": A})
    assert cov.levels == LEVELS and np.array_equal(cov.A, A.to_numpy()) and np.allclose(cov.L, np.linalg.cholesky(A.to_numpy()))
    assert notes == [] and (cov.L < -1e-10).any()                    # the case brms 2.23.0 gets wrong with several coefficients
    big = pd.DataFrame(np.eye(6) * 2.0, index=[*LEVELS, "z"], columns=[*LEVELS, "z"])
    big.iloc[:5, :5] = A.to_numpy()
    sub, notes = known_covariance(t, LEVELS, {"A": big})             # a label no fitted level has is dropped, as brms drops it
    assert np.array_equal(sub.A, A.to_numpy()) and sub.unused == ("z",)
    assert notes == ["data2['A']: the labels ['z'] are no fitted level of 'g'; their rows and columns are dropped (brms drops them too)"]
    perm = A.iloc[[2, 0, 4, 1, 3], [2, 0, 4, 1, 3]]                  # labelled, in another order: read by label
    assert np.array_equal(known_covariance(t, LEVELS, {"A": perm})[0].A, A.to_numpy())
    positional = pd.DataFrame(A.to_numpy(), index=list(LEVELS))      # the columns by position
    assert np.array_equal(known_covariance(t, LEVELS, {"A": positional})[0].A, A.to_numpy())
    not_pd = A.copy()
    not_pd.loc["a", "b"] = not_pd.loc["b", "a"] = 1.5
    singular = pd.DataFrame(np.ones((5, 5)), index=list(LEVELS), columns=list(LEVELS))
    asym = A.copy()
    asym.loc["a", "b"] = 0.4
    bad = [(None, "reads the known covariance 'A' from data2: pass data2={'A': <a pandas DataFrame"),
           ({"B": A}, "reads the known covariance 'A' from data2"),
           ({"A": A.to_numpy()}, "data2['A'] must be a square pandas DataFrame whose index labels the levels of 'g' (brms: 'Row or "
                                 "column names are required for within-group covariance matrices.')"),
           ({"A": pd.DataFrame(A.to_numpy())}, "must be a square pandas DataFrame"),
           ({"A": A.iloc[:, :4]}, "must be a square pandas DataFrame"),
           ({"A": A[list("bacde")]}, "the columns must be labelled as the rows, in the same order (or by position)"),
           ({"A": pd.DataFrame(A.to_numpy(), index=list("aacde"), columns=list("aacde"))}, "each level must label one row"),
           ({"A": A.astype(str)}, "data2['A'] must hold finite real numbers"),
           ({"A": A.replace(1.0, np.inf)}, "data2['A'] must hold finite real numbers"),
           ({"A": asym}, "must be symmetric (brms: 'Within-group covariance matrices must be symmetric.')"),
           ({"A": A.iloc[:4, :4]}, "the levels ['e'] of 'g' are not labels of data2['A'] (brms: 'Levels of the within-group "
                                   "covariance matrix for 'g' do not match names of the grouping levels.')"),
           ({"A": not_pd}, "is not positive definite (brms: 'Within-group covariance matrices must be positive definite.')"),
           ({"A": singular}, "is not positive definite")]
    for data2, message in bad:
        with pytest.raises(ab.FormulaError, match=re.escape(message)):
            known_covariance(t, LEVELS, data2)
    with pytest.raises(ab.FormulaError, match=re.escape("data2 holds ['B'], which no term of the formula reads")):
        ab.gaussian(COV, frame(), ab.Priors(**BASE, sd=HN), data2={"A": A, "B": A})
    with pytest.raises(ab.FormulaError, match=re.escape("gr(g, cov = A) reads its known covariance A from data2 too")):
        ab.gaussian("y ~ x + (1 | g)", frame(), ab.Priors(**BASE, sd=HN), data2={"A": A})
    with pytest.raises(ab.FormulaError, match=re.escape("is not positive definite")):
        ab.default_priors(COV, frame(), data2={"A": not_pd})    # default_priors checks the matrix as the constructor does


def kron_effects(Lcov, sd, L, z):
    """(levels, coefficients) effects from the mathematics: vec(r) = (Lcov kron diag(sd) L) vec(z), z (coefficients, levels)."""
    return (np.kron(Lcov, np.diag(sd) @ L) @ z.flatten(order="F")).reshape(z.shape[1], z.shape[0])


def test_cov_design_density_and_the_kronecker_product():
    df = frame()
    m = ab.gaussian(COV_SLOPE, df, slope_priors("g"), data2={"A": A})
    Lcov = np.linalg.cholesky(A.to_numpy())
    assert np.allclose(m.design.group_cov["g"].L, Lcov) and m.design.group_cov["g"].matrix == "A"
    idx = np.array([LEVELS.index(v) for v in df.g])
    ind = np.eye(5)[idx]
    assert np.array_equal(m.design.group_blocks[("g", "Intercept")].Z, ind)
    assert np.array_equal(m.design.group_blocks[("g", "x")].Z, ind * df.x.to_numpy()[:, None])
    assert m.correlation_names["g"]["L"] == "L|g" and "L|g_cholesky_corr__" in [v.name for v in m.pymc_model.value_vars]
    rng = np.random.default_rng(4)
    z = rng.normal(size=(2, 5))
    sd, L = np.array([0.8, 0.4]), lower(0.35)
    vp = m.value_point(intercept=0.2, b=[0.5], sd={"g": {"Intercept": sd[0], "x": sd[1]}}, z={"g": {"Intercept": z[0], "x": z[1]}},
                       L=L, sigma=1.1)
    r = kron_effects(Lcov, sd, L, z)
    eta = 0.2 + 0.5 * (df.x - df.x.mean()).to_numpy() + r[idx, 0] + r[idx, 1] * df.x.to_numpy()
    ref = (stats.norm.logpdf(df.y, eta, 1.1).sum() + stats.norm.logpdf(0.2, 0, 5) + stats.norm.logpdf(0.5, 0, 2)
           + stats.halfnorm.logpdf([1.1, *sd]).sum() + lkj2(L) + stats.norm.logpdf(z).sum())
    assert np.isclose(float(m.pymc_model.compile_logp(jacobian=False)(vp)), ref, rtol=1e-12)
    # the graph's map from vec(z) to vec(r) (level by level) is Lcov kron diag(sd) L, so vec(r) has covariance A kron Sigma
    pmm = m.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["1|g"], pmm["x|g"]]), inputs=pmm.value_vars, point_fn=True,
                        on_unused_input="ignore")
    T = np.empty((10, 10))
    for i in range(10):
        e = np.zeros(10)
        e[i] = 1.0
        ez = e.reshape(5, 2).T                                       # vec(z) level by level: z[c, j] at 2 j + c
        u, v = fn({**vp, "1|g_offset": ez[0], "x|g_offset": ez[1]})
        T[:, i] = np.column_stack([u, v]).ravel()
    Sigma = np.diag(sd) @ L @ L.T @ np.diag(sd)
    assert np.allclose(T, np.kron(Lcov, np.diag(sd) @ L), atol=1e-14) and np.allclose(T @ T.T, np.kron(A.to_numpy(), Sigma))
    # the Gaussian density of vec(r) under A kron Sigma is the standard normal density of z less log |det T|
    vec_r = r.ravel()
    assert np.isclose(stats.multivariate_normal(np.zeros(10), np.kron(A.to_numpy(), Sigma)).logpdf(vec_r),
                      stats.norm.logpdf(z).sum() - np.log(abs(np.linalg.det(T))), rtol=1e-12)
    # '||': each coefficient's effects are sd_c Lcov z_c (brms's dense product)
    ind_model = ab.gaussian(COV_SLOPE.replace("| gr", "|| gr"), df, slope_priors("g", correlated=False), data2={"A": A})
    vp2 = ind_model.value_point(intercept=0.2, b=[0.5], sd={"g": {"Intercept": sd[0], "x": sd[1]}}, z={"g": {"Intercept": z[0], "x": z[1]}},
                                sigma=1.1)
    eta2 = 0.2 + 0.5 * (df.x - df.x.mean()).to_numpy() + (sd[0] * Lcov @ z[0])[idx] + (sd[1] * Lcov @ z[1])[idx] * df.x.to_numpy()
    ref2 = (stats.norm.logpdf(df.y, eta2, 1.1).sum() + stats.norm.logpdf(0.2, 0, 5) + stats.norm.logpdf(0.5, 0, 2)
            + stats.halfnorm.logpdf([1.1, *sd]).sum() + stats.norm.logpdf(z).sum())
    assert np.isclose(float(ind_model.pymc_model.compile_logp(jacobian=False)(vp2)), ref2, rtol=1e-12)


def test_cov_in_a_log_link_family():
    df = frame()
    m = ab.Model("cnt ~ x + (1 | gr(g, cov = A))", df, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=HN), family="poisson",
                 data2={"A": A})
    z = np.linspace(-1, 1, 5)
    vp = m.value_point(intercept=0.3, b=[0.2], sd=0.7, z=z)
    u = 0.7 * np.linalg.cholesky(A.to_numpy()) @ z
    eta = 0.3 + 0.2 * (df.x - df.x.mean()).to_numpy() + u[[LEVELS.index(v) for v in df.g]]
    ref = (stats.poisson.logpmf(df.cnt, np.exp(eta)).sum() + stats.norm.logpdf(0.3, 0, 2) + stats.norm.logpdf(0.2, 0, 1)
           + stats.halfnorm.logpdf(0.7) + stats.norm.logpdf(z).sum())
    assert np.isclose(float(m.pymc_model.compile_logp(jacobian=False)(vp)), ref, rtol=1e-12)


def test_the_signed_case_density_against_both_constructions():
    """gauss_gr_cov_slope_signed (A with five negative Cholesky entries): at every frozen point the model's log density without
    the Jacobian is the one computed here with the Kronecker product (Lcov kron diag(sd) L) vec(z), and brms's is the one computed
    with Lcov's negative entries dropped (brms 2.23.0's scale_r_cor_cov, divergence D185), so the case's G2 and G3 fail on
    brms's arithmetic alone."""
    from tests.oracle import adapter
    from tests.oracle import fixture as fx

    case = fx.load_case("gauss_gr_cov_slope_signed")
    d = fx.fixture_dir(case["id"])
    data, data2 = fx.read_frame(case, d / "data.csv"), fx.read_data2(case, d)
    f = fx.Fixture.load(case["id"])
    m = adapter.build_model({**case, "autobayes": {**case["autobayes"], "kwargs": {"data2": data2}}}, data)
    levels = list(m.design.group_blocks[("g", "Intercept")].levels)
    Lcov = np.linalg.cholesky(data2["A"].loc[levels, levels].to_numpy())
    assert int((Lcov < -1e-10).sum()) == 5
    plus = np.where(Lcov > 1e-10, Lcov, 0.0)
    idx = np.array([levels.index(v) for v in data.g])
    x, y = data.x.to_numpy(), data.y.to_numpy()
    logp = m.pymc_model.compile_logp(jacobian=False)

    def density(p, Lc):
        sd, z, L = np.atleast_1d(p.pars["sd_1"]), np.atleast_2d(p.pars["z_1"]), np.atleast_2d(p.pars["L_1"])
        r = kron_effects(Lc, sd, L, z)
        eta = float(p.pars["Intercept"]) + float(np.ravel(p.pars["b"])[0]) * (x - x.mean()) + r[idx, 0] + r[idx, 1] * x
        sigma = float(p.pars["sigma"])
        return (stats.norm.logpdf(y, eta, sigma).sum() + stats.norm.logpdf(float(p.pars["Intercept"]), 0, 5)
                + stats.norm.logpdf(float(np.ravel(p.pars["b"])[0]), 0, 2) + stats.halfnorm.logpdf([sigma, *sd]).sum() + lkj2(L)
                + stats.norm.logpdf(z).sum())

    for p in f.points:
        sd, z = np.atleast_1d(p.pars["sd_1"]), np.atleast_2d(p.pars["z_1"])
        vp = m.value_point(intercept=float(p.pars["Intercept"]), b=[float(np.ravel(p.pars["b"])[0])],
                           sd={"g": {"Intercept": sd[0], "x": sd[1]}}, z={"g": {"Intercept": z[0], "x": z[1]}},
                           L=np.atleast_2d(p.pars["L_1"]), sigma=float(p.pars["sigma"]))
        mine, brms = density(p, Lcov), density(p, plus)
        assert np.isclose(float(logp(vp)), mine, rtol=1e-12)        # AutoBayes: the Kronecker product, covariance A kron Sigma
        assert np.isclose(p.lp_nojac, brms, rtol=1e-12)             # brms: Lcov+ kron diag(sd) L
        assert abs(mine - brms) > 1e-3


# -- Student-t effects ---------------------------------------------------------------------------------------------------------
def truncated_gamma(v, a=2.0, rate=0.1):
    g = stats.gamma(a, scale=1 / rate)
    return g.logpdf(v) - g.logsf(1.0)


def test_student_effects_names_and_density():
    df = frame()
    pr = ab.Priors(**BASE, sd=HN, df={"g": ab.Gamma(2, 0.1)})
    m = ab.gaussian(STUDENT, df, pr)
    names = m.group_dist_names["g"]
    assert names == {"df": "df|g", "udf": "udf|g", "dfm": "dfm|g", "coordinate": "g__factor_dim"}
    assert m.group_names["g"]["df"] == "df|g" and m.names["udf"] == "udf|g"
    vals = [v.name for v in m.pymc_model.value_vars]
    assert "df|g_interval__" in vals and "udf|g_log__" in vals and m.pymc_model["dfm|g"] in m.pymc_model.deterministics
    z, udf = np.linspace(-1.2, 0.9, 5), np.array([0.3, 0.6, 1.1, 0.2, 2.0])
    vp = m.value_point(intercept=0.1, b=[0.4], sd=0.9, z=z, sigma=1.2, df=4.5, udf=udf)
    assert np.isclose(vp["df|g_interval__"], np.log(3.5)) and np.allclose(vp["udf|g_log__"], np.log(udf))
    u = np.sqrt(4.5 * udf) * 0.9 * z
    eta = 0.1 + 0.4 * (df.x - df.x.mean()).to_numpy() + u[[LEVELS.index(v) for v in df.g]]
    udf_density = stats.invgamma(4.5 / 2, scale=0.5).logpdf(udf).sum()
    # Stan's inv_chi_square_lpdf(u | df) written out: -df/2 log 2 - lgamma(df/2) - (df/2 + 1) log u - 1 / (2 u)
    assert np.isclose(udf_density, sum(-2.25 * np.log(2) - math.lgamma(2.25) - 3.25 * np.log(v) - 1 / (2 * v) for v in udf))
    ref = (stats.norm.logpdf(df.y, eta, 1.2).sum() + stats.norm.logpdf(0.1, 0, 5) + stats.norm.logpdf(0.4, 0, 2)
           + stats.halfnorm.logpdf([0.9, 1.2]).sum() + truncated_gamma(4.5) + udf_density + stats.norm.logpdf(z).sum())
    assert np.isclose(float(m.pymc_model.compile_logp(jacobian=False)(vp)), ref, rtol=1e-12)
    # each level's effect is Student-t(df, 0, sd) once udf is integrated out: the scale mixture's density, numerically
    mixture, _ = quad(lambda v: stats.invgamma(4.5 / 2, scale=0.5).pdf(v) * stats.norm.pdf(0.7, 0, 0.9 * np.sqrt(4.5 * v)), 0, np.inf,
                      epsabs=0, epsrel=1e-11)
    assert np.isclose(mixture, stats.t.pdf(0.7, 4.5, 0, 0.9), rtol=1e-9)
    # with slopes: each level's vector of effects is scaled by its dfm (brms's rep_matrix(dfm, M) .* scale_r_cor)
    sm = ab.gaussian('y ~ x + (1 + x | gr(g, dist = "student"))', df, slope_priors("g", df=ab.Gamma(2, 0.1)))
    zz, L, sd = np.vstack([z, z[::-1]]), lower(-0.3), np.array([0.9, 0.5])
    vp2 = sm.value_point(intercept=0.1, b=[0.4], sd={"g": {"Intercept": 0.9, "x": 0.5}}, z={"g": {"Intercept": zz[0], "x": zz[1]}},
                         L=L, sigma=1.2, df={"g": 4.5}, udf={"g": udf})
    r = (np.diag(sd) @ L @ zz).T * np.sqrt(4.5 * udf)[:, None]
    idx = [LEVELS.index(v) for v in df.g]
    eta2 = 0.1 + 0.4 * (df.x - df.x.mean()).to_numpy() + r[idx, 0] + r[idx, 1] * df.x.to_numpy()
    ref2 = (stats.norm.logpdf(df.y, eta2, 1.2).sum() + stats.norm.logpdf(0.1, 0, 5) + stats.norm.logpdf(0.4, 0, 2)
            + stats.halfnorm.logpdf([0.9, 0.5, 1.2]).sum() + lkj2(L) + truncated_gamma(4.5) + udf_density + stats.norm.logpdf(zz).sum())
    assert np.isclose(float(sm.pymc_model.compile_logp(jacobian=False)(vp2)), ref2, rtol=1e-12)
    base = dict(intercept=0.1, b=[0.4], sd=0.9, z=z, sigma=1.2)
    for extra, message in [({}, "df is required"), ({"df": 4.5}, "udf is required"),
                           ({"df": {"h": 4.5}, "udf": udf}, "df must cover exactly the factors ['g']"),
                           ({"df": 1.0, "udf": udf}, "df for 'g' must be a finite scalar > 1"),
                           ({"df": 4.5, "udf": udf[:4]}, "udf for 'g' must be 5 finite positive numbers, one per level"),
                           ({"df": 4.5, "udf": -udf}, "udf for 'g' must be 5 finite positive numbers")]:
        with pytest.raises(ValueError, match=re.escape(message)):
            m.value_point(**base, **extra)
    plain = ab.gaussian("y ~ x + (1 | g)", df, ab.Priors(**BASE, sd=HN))
    with pytest.raises(ValueError, match="df and udf are the degrees of freedom and scales of a gr"):
        plain.value_point(intercept=0.1, b=[0.4], sd=0.9, z=z, sigma=1.2, df=4.5)
    two = ab.gaussian('y ~ x + (1 | gr(g, dist = "student")) + (1 | gr(h, dist = "student"))', df,
                      ab.Priors(**BASE, sd={"g": HN, "h": HN}, df={"g": ab.Gamma(2, 0.1), "h": ab.Gamma(2, 0.1)}))
    with pytest.raises(ValueError, match=re.escape("df requires an exact-key mapping over the factors ['g', 'h']")):
        two.value_point(intercept=0.1, b=[0.4], sd={"g": 0.9, "h": 0.5}, z={"g": z, "h": z[:3]}, sigma=1.2, df=4.5, udf=udf)


def test_df_priors_are_checked():
    spec = parse_formula(STUDENT)
    assert ab.Priors(df=ab.Gamma(2, 0.1)).df_priors(spec) == {"g": ab.Gamma(2, 0.1)}
    for priors, message in [(ab.Priors(), "need a prior on their degrees of freedom: pass df={'g': ab.Gamma(2, 0.1)}"),
                            (ab.Priors(df={"h": ab.Gamma(2, 0.1)}), "df must cover exactly the factors with gr(g, dist = \"student\") ['g']"),
                            (ab.Priors(df=ab.HalfStudentT(3, 1)), "prior for 'df:g' must be a positive-support prior"),
                            (ab.Priors(df=ab.Normal(0, 1)), "prior for 'df:g' must be a positive-support prior")]:
        with pytest.raises(ab.PriorError, match=re.escape(message)):
            priors.df_priors(spec)
    two = parse_formula('y ~ x + (1 | gr(g, dist = "student")) + (1 | gr(h, dist = "student"))')
    with pytest.raises(ab.PriorError, match="need an exact-key df mapping; a bare prior is not broadcast"):
        ab.Priors(df=ab.Gamma(2, 0.1)).df_priors(two)
    with pytest.raises(ab.PriorError, match="the formula has none"):
        ab.Priors(df=ab.Gamma(2, 0.1)).df_priors(parse_formula("y ~ x + (1 | g)"))
    with pytest.raises(ab.PriorError, match="the formula has none"):
        ab.gaussian("y ~ x + (1 | g)", frame(), ab.Priors(**BASE, sd=HN, df=ab.Gamma(2, 0.1)))


# -- multi-membership slopes, mmc() and scale = FALSE --------------------------------------------------------------------------
def memberships(df):
    return [np.array([LEVELS.index(v) for v in df[c]]) for c in ("g", "g2")]


def test_mm_slopes_design_and_density():
    df = frame()
    m = ab.gaussian(MM_SLOPE, df, slope_priors("mmgg2"))
    W = np.column_stack([df.w1, df.w2]) / (df.w1 + df.w2).to_numpy()[:, None]
    idx, n = memberships(df), len(df)
    blocks = [m.design.group_blocks[("mmgg2", c)] for c in ("Intercept", "x")]
    Zs = [np.zeros((n, 5)), np.zeros((n, 5))]
    for k in range(2):
        np.add.at(Zs[0], (np.arange(n), idx[k]), W[:, k])
        np.add.at(Zs[1], (np.arange(n), idx[k]), W[:, k] * df.x.to_numpy())
    assert np.allclose(blocks[0].Z, Zs[0]) and np.allclose(blocks[1].Z, Zs[1]) and np.allclose(m.design.Z, np.hstack(Zs))
    assert blocks[0].covariates == () and all(np.array_equal(c, df.x.to_numpy()) for c in blocks[1].covariates)
    assert blocks[1].column_slice == slice(5, 10) and m.correlation_names["mmgg2"]["cor"] == "cor|mmgg2"
    rng = np.random.default_rng(8)
    z, sd, L = rng.normal(size=(2, 5)), np.array([0.7, 0.3]), lower(0.5)
    vp = m.value_point(intercept=0.4, b=[0.2], sd={"mmgg2": {"Intercept": 0.7, "x": 0.3}}, z={"mmgg2": {"Intercept": z[0], "x": z[1]}},
                       L=L, sigma=0.8)
    r = (np.diag(sd) @ L @ z).T
    eta = 0.4 + 0.2 * (df.x - df.x.mean()).to_numpy() + sum(W[:, k] * (r[idx[k], 0] + r[idx[k], 1] * df.x.to_numpy()) for k in range(2))
    ref = (stats.norm.logpdf(df.y, eta, 0.8).sum() + stats.norm.logpdf(0.4, 0, 5) + stats.norm.logpdf(0.2, 0, 2)
           + stats.halfnorm.logpdf([0.7, 0.3, 0.8]).sum() + lkj2(L) + stats.norm.logpdf(z).sum())
    assert np.isclose(float(m.pymc_model.compile_logp(jacobian=False)(vp)), ref, rtol=1e-12)
    pmm = m.pymc_model                                                # the graph's correlation matrix is L L^T
    cor = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["cor|mmgg2"]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")(vp)[0]
    assert np.allclose(cor, L @ L.T)


def test_mmc_and_raw_weights_density():
    df = frame()
    m = ab.gaussian(MMC, df, slope_priors("mmgg2", ("Intercept", "mmc(x1, x2)"), correlated=False))
    idx, n = memberships(df), len(df)
    block = m.design.group_blocks[("mmgg2", "mmc(x1, x2)")]
    assert np.array_equal(block.covariates[0], df.x1.to_numpy()) and np.array_equal(block.covariates[1], df.x2.to_numpy())
    assert m.group_coefficient_names[("mmgg2", "mmc(x1, x2)")]["sd"] == "mmc(x1, x2)|mmgg2_sigma" and "mmgg2" not in m.correlation_names
    z = np.vstack([np.linspace(-1, 1, 5), np.linspace(0.5, -0.7, 5)])
    vp = m.value_point(intercept=0.4, b=[0.2], sd={"mmgg2": {"Intercept": 0.7, "mmc(x1, x2)": 0.3}},
                       z={"mmgg2": {"Intercept": z[0], "mmc(x1, x2)": z[1]}}, sigma=0.8)
    u, v = 0.7 * z[0], 0.3 * z[1]
    eta = (0.4 + 0.2 * (df.x - df.x.mean()).to_numpy() + 0.5 * (u[idx[0]] + v[idx[0]] * df.x1.to_numpy())
           + 0.5 * (u[idx[1]] + v[idx[1]] * df.x2.to_numpy()))
    ref = (stats.norm.logpdf(df.y, eta, 0.8).sum() + stats.norm.logpdf(0.4, 0, 5) + stats.norm.logpdf(0.2, 0, 2)
           + stats.halfnorm.logpdf([0.7, 0.3, 0.8]).sum() + stats.norm.logpdf(z).sum())
    assert np.isclose(float(m.pymc_model.compile_logp(jacobian=False)(vp)), ref, rtol=1e-12)
    raw = ab.gaussian(UNSCALED, df, ab.Priors(**BASE, sd=HN))      # scale = FALSE: the weights as given, negative ones too
    W = raw.design.group_blocks[("mmgg2", "Intercept")].weights
    assert np.array_equal(W[0], df.wn.to_numpy()) and np.array_equal(W[1], df.w2.to_numpy()) and (df.wn < 0).any()
    vp2 = raw.value_point(intercept=0.4, b=[0.2], sd=0.7, z=z[0], sigma=0.8)
    eta2 = 0.4 + 0.2 * (df.x - df.x.mean()).to_numpy() + df.wn.to_numpy() * u[idx[0]] + df.w2.to_numpy() * u[idx[1]]
    ref2 = (stats.norm.logpdf(df.y, eta2, 0.8).sum() + stats.norm.logpdf(0.4, 0, 5) + stats.norm.logpdf(0.2, 0, 2)
            + stats.halfnorm.logpdf([0.7, 0.8]).sum() + stats.norm.logpdf(z[0]).sum())
    assert np.isclose(float(raw.pymc_model.compile_logp(jacobian=False)(vp2)), ref2, rtol=1e-12)
    even = parse_formula("y ~ x + (1 | mm(g, g2, scale = FALSE))").group_terms[0]   # no weights: 1 / members each, as brms's
    assert np.allclose(mm_weights(even, df), 0.5)
    bad = df.assign(wn=np.where(np.arange(n) == 3, np.inf, df.wn))
    with pytest.raises(ab.FormulaError, match=re.escape("the weights ['wn', 'w2'] of (1 | mm(g, g2, weights = cbind(wn, w2), scale = FALSE)) "
                                                        "must be finite on every row")):
        ab.gaussian(UNSCALED, bad, ab.Priors(**BASE, sd=HN))
    with pytest.raises(ab.FormulaError, match=re.escape("mm(..., scale = FALSE) takes them as given, negative ones included")):
        ab.gaussian(UNSCALED.replace(", scale = FALSE", ""), df, ab.Priors(**BASE, sd=HN))
    with pytest.raises(ab.FormulaError, match=re.escape("column 'x1' of (1 + mmc(x1, x2) || mm(g, g2)) must be numeric and finite")):
        ab.gaussian(MMC, df.assign(x1=df.h), slope_priors("mmgg2", ("Intercept", "mmc(x1, x2)"), correlated=False))


# -- predictions ---------------------------------------------------------------------------------------------------------------
def draws(model, rng, n_draw=3):
    """Synthetic posterior draws of the effects, SDs and population terms the predictions read."""
    data, coords = {}, {"chain": [0], "draw": np.arange(n_draw)}
    data["Intercept"] = (("chain", "draw"), rng.normal(size=(1, n_draw)))
    data["x"] = (("chain", "draw"), rng.normal(size=(1, n_draw)))
    data["sigma"] = (("chain", "draw"), np.full((1, n_draw), 1.0))
    for (g, c), roles in model.group_coefficient_names.items():
        J = model.design.group_blocks[(g, c)].J
        data[roles["group_effect"]] = (("chain", "draw", roles["coordinate"]), rng.normal(size=(1, n_draw, J)))
        data[roles["sd"]] = (("chain", "draw"), np.full((1, n_draw), 0.7))
        coords[roles["coordinate"]] = list(model.design.group_blocks[(g, c)].levels)
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


def test_mm_slope_and_mmc_predictions_by_hand():
    df = frame()
    rng = np.random.default_rng(3)
    for formula, cols in ((MM_SLOPE, ("x", "x")), (MMC, ("x1", "x2"))):
        coefs = ("Intercept", "x") if cols == ("x", "x") else ("Intercept", "mmc(x1, x2)")
        m = ab.gaussian(formula, df, slope_priors("mmgg2", coefs, correlated=formula == MM_SLOPE))
        idata = draws(m, rng)
        post = idata.posterior
        u = post[m.group_coefficient_names[("mmgg2", "Intercept")]["group_effect"]].values
        v = post[m.group_coefficient_names[("mmgg2", coefs[1])]["group_effect"]].values
        nd = df.iloc[:6].copy()
        W = (np.column_stack([nd.w1, nd.w2]) / (nd.w1 + nd.w2).to_numpy()[:, None] if formula == MM_SLOPE
             else np.full((len(nd), 2), 0.5))
        idx = memberships(nd)
        ref = post["Intercept"].values[..., None] + post["x"].values[..., None] * nd.x.to_numpy()
        for k in range(2):
            ref = ref + W[:, k] * (u[..., idx[k]] + v[..., idx[k]] * nd[cols[k]].to_numpy())
        assert np.allclose(m.predict(idata, newdata=nd, kind="linear")["eta"].values, ref)
        assert np.allclose(m.predict(idata, newdata=nd, kind="mean")["mu"].values, ref)
    with pytest.raises(ab.FormulaError, match=re.escape("newdata lacks columns ['x2']")):
        m.predict(idata, newdata=nd.drop(columns="x2"), kind="linear")


def test_new_levels_of_cov_and_student_terms_are_refused():
    df = frame()
    rng = np.random.default_rng(5)
    for m, why in ((ab.gaussian(COV, df, ab.Priors(**BASE, sd=HN), data2={"A": A}), "a level outside the known covariance"),
                   (ab.gaussian(STUDENT, df, ab.Priors(**BASE, sd=HN, df=ab.Gamma(2, 0.1))), "how its Student-t scales are drawn")):
        idata = draws(m, rng)
        nd = df.iloc[:3].copy()
        u = idata.posterior["1|g"].values
        fitted = m.predict(idata, newdata=nd, kind="linear")["eta"].values
        ref = (idata.posterior["Intercept"].values[..., None] + idata.posterior["x"].values[..., None] * nd.x.to_numpy()
               + u[..., [LEVELS.index(v) for v in nd.g]])
        assert np.allclose(fitted, ref)                               # the effects of fitted levels, as drawn
        nd.loc[nd.index[0], "g"] = "zz"
        for policy in ("population", "resample_observed"):
            with pytest.raises(ab.FormulaError, match=re.escape(why) + ".*divergence D186"):
                m.predict(idata, newdata=nd, kind="linear", new_groups=policy)
        with pytest.raises(ValueError, match=re.escape(f"prior_draws() of a model with the group term '{m.spec.group_terms[0].canonical}' "
                                                       "(gr(cov = ) or gr(dist = \"student\")) is not offered")):
            ab.prior_draws(m)


# -- default priors, describe() and the target identity ---------------------------------------------------------------------
def test_default_priors_and_their_table():
    df = frame()
    pr = ab.default_priors(STUDENT, df)
    assert pr.df == {"g": ab.Gamma(2, 0.1)} and pr.df["g"].source == "default"
    sm = ab.gaussian(STUDENT, df, pr)
    assert "df:g" in sm.default_prior_labels and "sd:g" in sm.default_prior_labels
    rows = [(r["class"], r["coef"], r["group"], r["prior"], r["lb"]) for r in sm.prior_table()]
    assert ("df", "", "g", "gamma(2, 0.1)", "1") in rows and not any(r[0] == "df" and r[2] == "" for r in rows)
    mm = ab.default_priors(MMC, df)
    assert set(mm.sd["mmgg2"]) == {"Intercept", "mmc(x1, x2)"} and mm.cor is None
    rows = [(r["class"], r["coef"], r["group"]) for r in ab.gaussian(MMC, df, mm).prior_table()]
    assert ("sd", "mmcx1x2", "mmgg2") in rows                         # brms's label of the mmc() coefficient
    cov = ab.default_priors(COV_SLOPE, df, data2={"A": A})
    assert cov.cor == {"g": ab.LKJ(1)} and ab.gaussian(COV_SLOPE, df, cov, data2={"A": A}).prior_table()


def test_describe_and_target_identity():
    df = frame()
    big = pd.DataFrame(np.eye(6) * 2.0, index=[*LEVELS, "z"], columns=[*LEVELS, "z"])
    big.iloc[:5, :5] = A.to_numpy()
    cov = ab.gaussian(COV_SLOPE, df, slope_priors("g"), data2={"A": big})
    text = cov.describe()
    assert "with correlated varying slope(s) and a known covariance of a factor's levels (gr(cov = ))" in text
    assert "gr(g, cov = A): the levels' known covariance data2['A'] on the 5 fitted levels, Lcov_g its lower Cholesky factor" in text
    assert "its labels ['z'] are no fitted level and are dropped" in text and cov.warnings[-1].startswith("data2['A']: the labels ['z']")
    assert "vec(r) = (Lcov_g kron diag(sd) L_g) vec(z) has covariance A kron Sigma" in text and "divergence D185" in text
    one = ab.gaussian(COV, df, ab.Priors(**BASE, sd=HN), data2={"A": A}).describe()
    assert "so u_g has covariance sd_g^2 A." in one
    st = ab.gaussian('y ~ x + (1 + x | gr(g, dist = "student"))', df, slope_priors("g", df=ab.Gamma(2, 0.1))).describe()
    assert "Student-t group effects (gr(dist = \"student\"))" in st and "dfm_g[j] = sqrt(df_g * udf_g[j])" in st
    assert "each level's vector of effects is multivariate Student-t" in st and "dfm|g = sqrt(df_g * udf_g)" in st
    assert "df_g ~ Gamma(alpha=2, beta=0.1), truncated to df_g > 1 and renormalised there (brms class df, group g)" in st
    assert "log(df_g - 1), the degrees of freedom of the Student-t effects of g (brms's df)" in st
    mm = ab.gaussian(MMC, df, slope_priors("mmgg2", ("Intercept", "mmc(x1, x2)"), correlated=False)).describe()
    assert "multiplies, for each member, 'x1' for 'g', 'x2' for 'g2' (brms's Z_m_c_k)" in mm
    assert "W_g,i * (u[g_i] + x_x1,i * v_mmc(x1, x2)|mmgg2[g_i]) + W_g2,i * (u[g2_i] + x_x2,i * v_mmc(x1, x2)|mmgg2[g2_i])" in mm
    assert "Z_c,k,i coefficient c's covariate for member k" in mm
    raw = ab.gaussian(UNSCALED, df, ab.Priors(**BASE, sd=HN)).describe()
    assert "the weight columns wn, w2 as given (brms's scale = FALSE)" in raw and "the weights as given (brms's scale = FALSE)." in raw
    # the target identity: the matrix, df's prior, scale and the members' covariates each change it
    ids = [ab.gaussian(COV, df, ab.Priors(**BASE, sd=HN), data2={"A": a}).target_identity
           for a in (A, A * 2.0)]
    assert ids[0] != ids[1]
    st1 = ab.gaussian(STUDENT, df, ab.Priors(**BASE, sd=HN, df=ab.Gamma(2, 0.1))).target_identity
    st2 = ab.gaussian(STUDENT, df, ab.Priors(**BASE, sd=HN, df=ab.Gamma(3, 0.1))).target_identity
    assert st1 != st2
    scaled = ab.gaussian(UNSCALED.replace("wn", "w1").replace(", scale = FALSE", ""), df, ab.Priors(**BASE, sd=HN)).target_identity
    assert scaled != ab.gaussian(UNSCALED.replace("wn", "w1"), df, ab.Priors(**BASE, sd=HN)).target_identity


# -- the graph checks at build -------------------------------------------------------------------------------------------------
class Graph:
    """A PyMC model whose named attributes are overridden (``override``), to show the build checks refuse a wrong graph."""

    def __init__(self, pmm, **override):
        self._pmm, self._override = pmm, override

    def __getattr__(self, name):
        return self._override.get(name, getattr(self._pmm, name))

    def __getitem__(self, key):
        return self._pmm[key]


def test_the_graph_checks_refuse_a_wrong_graph():
    df = frame()
    st = ab.gaussian('y ~ x + (1 + x | gr(g, dist = "student"))', df, slope_priors("g", df=ab.Gamma(2, 0.1)))
    by = ab.gaussian("y ~ x + (1 | gr(g, by = gby))", df, ab.Priors(**BASE, sd=HN))
    pmm = st.pymc_model
    dims = dict(pmm.named_vars_to_dims)
    cases = [
        (st, dict(free_RVs=[]), r"'1\|g_sigma' is not a free positive variable of shape \(\) sampled on its log"),
        (st, dict(coords={**pmm.coords, "g__factor_dim": ("b", "a", "c", "d", "e")}), r"coordinate 'g__factor_dim' differs from the fitted"),
        (st, dict(deterministics=[]), r"'1\|g_offset' is not a free variable or '1\|g' not a deterministic"),
        (st, dict(named_vars_to_dims={k: v for k, v in dims.items() if k != "1|g_offset"}), r"PyMC dims of '1\|g_offset' are not"),
        (st, dict(deterministics=[v for v in pmm.deterministics if v.name != "dfm|g"]),
         r"'df\|g' and 'udf\|g' are not free variables on their interval and log coordinates, or 'dfm\|g' is not a deterministic"),
        (st, dict(named_vars_to_dims={k: v for k, v in dims.items() if k != "udf|g"}), r"PyMC dims of 'udf\|g' are not"),
        (by, dict(named_vars_to_dims={k: v for k, v in by.pymc_model.named_vars_to_dims.items() if k != "1|g_sigma"}),
         r"PyMC dims of '1\|g_sigma' are not \('g__by_dim',\) with the levels of 'gby'"),
    ]
    for m, override, needle in cases:
        saved = m.pymc_model
        m.pymc_model = Graph(saved, **override)
        try:
            with pytest.raises(RuntimeError, match=needle):
                m._custom_group_blocks()
        finally:
            m.pymc_model = saved
    blocks = st._custom_group_blocks()
    assert set(blocks) == {("g", "Intercept"), ("g", "x")}
    wrong = {k: dataclasses.replace(b, Z=2.0 * b.Z) for k, b in blocks.items()}
    with pytest.raises(RuntimeError, match=r"the graph's 'mu' does not move by the weighted effects of 'g' at its rows' levels"):
        st._check_custom_group_effects(wrong)
    cov = ab.gaussian(COV, df, ab.Priors(**BASE, sd=HN), data2={"A": A})
    cov._group_cov["g"] = SimpleNamespace(L=np.eye(5))               # a covariance the graph does not carry
    with pytest.raises(RuntimeError, match=r"does not move by the weighted effects of 'g'"):
        cov._custom_group_blocks()


def test_fit_record_holds_the_df_prior(monkeypatch):
    import autobayes.session as session

    m = ab.gaussian(STUDENT, frame(), ab.Priors(**BASE, sd=HN, df={"g": ab.Gamma(2, 0.1)}))
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=draws(m, np.random.default_rng(1)), backend="synthetic",
                          device_requested="cpu", device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=1,
                          draws=3, tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = session.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["priors"]["df:g"] == "Gamma(alpha=2, beta=0.1)"


# -- with the group-term syntax, class-wide priors and dropna ----------------------------------------------------------------
def test_slopes_in_these_terms_are_numeric_columns():
    """A factor slope is refused in a term the model builds itself (its expansion is verified for the terms Bambi builds); a
    numeric column without its population-level term varies there as anywhere, brms's Z the raw column, b_w absent."""
    with pytest.raises(ab.FormulaError, match=re.escape("the varying slope 'z' of '(1 + z | gr(g, cov = A))' is not supported")):
        parse_formula("y ~ x + z + (1 + z | gr(g, cov = A))", categorical={"z": "p"})
    with pytest.raises(ab.FormulaError, match=re.escape("the varying slope 'z' of '(1 + z | mm(g, g2))' is not supported")):
        parse_formula("y ~ x + z + (1 + z | mm(g, g2))", categorical={"z": "p"})
    df = frame()
    m = ab.gaussian("y ~ x + (1 + w | mm(g, g2))", df, slope_priors("mmgg2", ("Intercept", "w")))
    assert m.spec.group_terms[0].slope_columns == ("w",) and "w" not in m.spec.columns
    idx = memberships(df)
    z, sd, L = np.vstack([np.linspace(-1, 1, 5), np.linspace(0.6, -0.4, 5)]), np.array([0.7, 0.3]), lower(0.2)
    vp = m.value_point(intercept=0.4, b=[0.2], sd={"mmgg2": {"Intercept": 0.7, "w": 0.3}}, z={"mmgg2": {"Intercept": z[0], "w": z[1]}},
                       L=L, sigma=0.8)
    r = (np.diag(sd) @ L @ z).T
    eta = 0.4 + 0.2 * (df.x - df.x.mean()).to_numpy() + sum(0.5 * (r[idx[k], 0] + r[idx[k], 1] * df.w.to_numpy()) for k in range(2))
    ref = (stats.norm.logpdf(df.y, eta, 0.8).sum() + stats.norm.logpdf(0.4, 0, 5) + stats.norm.logpdf(0.2, 0, 2)
           + stats.halfnorm.logpdf([0.7, 0.3, 0.8]).sum() + lkj2(L) + stats.norm.logpdf(z).sum())
    assert np.isclose(float(m.pymc_model.compile_logp(jacobian=False)(vp)), ref, rtol=1e-12)


def test_class_wide_priors_cover_df_and_these_terms_sds():
    two = parse_formula('y ~ x + (1 | gr(g, dist = "student")) + (1 | gr(h, dist = "student"))')
    pr = ab.Priors(df={"*": ab.Gamma(2, 0.1), "h": ab.Exponential(0.5)})
    assert pr.df_priors(two) == {"g": ab.Gamma(2, 0.1), "h": ab.Exponential(0.5)}
    with pytest.raises(ab.PriorError, match=re.escape("the override(s) ['k'] of df={'*': ...} name no factor")):
        ab.Priors(df={"*": ab.Gamma(2, 0.1), "k": ab.Gamma(2, 0.1)}).df_priors(two)
    df = frame()
    m = ab.gaussian(MMC, df, ab.Priors(**BASE, sd={"*": ab.HalfNormal(1), "mmgg2": {"mmc(x1, x2)": ab.HalfNormal(0.5)}}))
    assert m.priors.sd_priors(m.spec) == {("mmgg2", "Intercept"): ab.HalfNormal(1), ("mmgg2", "mmc(x1, x2)"): ab.HalfNormal(0.5)}


def test_dropna_with_these_terms():
    """dropna=True drops the rows as for any term and the model is the model of the rows kept; with gr(cov = A) it is refused when
    the dropped rows hold every row of a level (the levels of A would change with the missing values)."""
    df = frame()
    holes = df.copy()
    holes.loc[[1, 7], "x"] = np.nan                                  # levels b and c keep other rows
    kept = holes.dropna(subset=["x"]).reset_index(drop=True)
    pr = ab.Priors(**BASE, sd=HN)
    with pytest.warns(UserWarning, match="Rows containing NAs were excluded from the model"):
        m = ab.gaussian(COV, holes, pr, data2={"A": A}, dropna=True)
    clean = ab.gaussian(COV, kept, pr, data2={"A": A})
    z = np.linspace(-1, 1, 5)
    point = dict(intercept=0.2, b=[0.3], sd=0.6, z=z, sigma=1.1)
    assert m.dropped_na_rows == (1, 7)
    assert np.isclose(float(m.pymc_model.compile_logp()(m.value_point(**point))), float(clean.pymc_model.compile_logp()(clean.value_point(**point))),
                      rtol=1e-12)
    lost = df.copy()
    lost.loc[df.g == "e", "x"] = np.nan                              # every row of level e
    with pytest.raises(ab.FormulaError, match=re.escape("dropna=True drops every row of the level(s) ['e'] of 'g'")):
        ab.gaussian(COV, lost, pr, data2={"A": A}, dropna=True)
    mmh = df.copy()
    mmh.loc[[2, 5], "w1"] = np.nan
    with pytest.warns(UserWarning):
        mm = ab.gaussian(MM_SLOPE, mmh, slope_priors("mmgg2"), dropna=True)
    mm_clean = ab.gaussian(MM_SLOPE, mmh.dropna(subset=["w1"]).reset_index(drop=True), slope_priors("mmgg2"))
    assert np.allclose(mm.design.Z, mm_clean.design.Z) and mm.spec.levels_by_group == mm_clean.spec.levels_by_group
