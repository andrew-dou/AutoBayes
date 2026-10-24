"""Residual correlation over time within groups (brms-parity F03b): brms's ``unstr(time, gr)`` and ``cosy(time, gr)``. The
term and its parsing, the refusals (D146), the groups of the fitted rows, the compound-symmetric Cholesky factor, the joint
densities of both families against SciPy's multivariate normal and t (and the JAX density), the LOO log-likelihood against the
joint density less the marginal of the other rows, joint draws against their covariance, priors and brms's default priors,
``value_point``, predictions, ``describe()``, names, the target identity, fit records, prior-predictive draws, the other
modules' refusals, and the oracle's extension (row order, the groups' Stan data, ``Lcortime`` and ``cosy`` coordinates, and
brms's ``bayes_R2`` on unsorted rows, D147). Nothing here samples; the brms comparison is the oracle cases ``gauss_unstr``,
``gauss_cosy``, ``student_cosy``, ``gauss_unstr_post``, ``gauss_unstr_site_varint_post`` and ``student_cosy_post``."""
import copy
import json
import math
import warnings
from dataclasses import asdict
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from scipy import stats

import autobayes as ab
from autobayes import timecor as tc
from autobayes.correlation import cholesky_corr_free, lkj_cholesky_rng
from autobayes.families import FAMILIES
from autobayes.formula import parse_formula

# five subjects: s1 and s2 at every time 1..4, s3 without time 2, s4 at time 2 alone, s5 at times 1 and 2; rows shuffled
_SUBJ = ["s1"] * 4 + ["s2"] * 4 + ["s3"] * 3 + ["s4"] + ["s5"] * 2
_TT = [1, 2, 3, 4, 1, 2, 3, 4, 1, 3, 4, 2, 1, 2]
_PERM = [5, 12, 0, 9, 3, 13, 7, 11, 1, 10, 4, 8, 2, 6]
_RNG = np.random.default_rng(20261024)
FRAME = pd.DataFrame({
    "subj": [_SUBJ[i] for i in _PERM], "tt": [_TT[i] for i in _PERM],
    "x": np.round(_RNG.normal(0.0, 1.0, 14), 4), "w": np.round(_RNG.normal(0.5, 1.0, 14), 4),
    "y": np.round(_RNG.normal(0.3, 1.2, 14), 4), "h": ["u", "v"] * 7, "ord": [i % 4 for i in range(14)],
})
LINEAR = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))
UNSTR = "y ~ x + unstr(time = tt, gr = subj)"
COSY = "y ~ x + cosy(time = tt, gr = subj)"
L4 = np.linalg.cholesky(np.array([[1.0, 0.5, 0.2, -0.1], [0.5, 1.0, 0.3, 0.1], [0.2, 0.3, 1.0, 0.4], [-0.1, 0.1, 0.4, 1.0]]))


def unstr_model(formula=UNSTR, frame=None, family="gaussian", **extra):
    priors = ab.Priors(**{**LINEAR, "cortime": ab.LKJ(2), **extra})
    build = ab.gaussian if family == "gaussian" else ab.student
    return build(formula, FRAME if frame is None else frame, priors)


def cosy_model(formula=COSY, frame=None, family="gaussian", **extra):
    priors = ab.Priors(**{**LINEAR, "cosy": ab.Flat(), **extra})
    build = ab.gaussian if family == "gaussian" else ab.student
    return build(formula, FRAME if frame is None else frame, priors)


def cs_matrix(rho, k):
    return (1.0 - rho) * np.eye(k) + rho * np.ones((k, k))


def groups_of(frame, gr="subj", time="tt"):
    """Each group's data rows in time order, by label (pandas, independent of ``timecor.time_layout``)."""
    return {g: sub.sort_values(time).index.to_numpy() for g, sub in frame.groupby(gr)}


def reference_logp(frame, mu, sigma, cor_of, nu=None):
    """The joint log density written from the definition: per group, SciPy's multivariate normal (or t with ``nu``) of its
    residuals with covariance (or scale matrix) ``sigma^2 cor_of(rows)``."""
    y, total = frame["y"].to_numpy(dtype=float), 0.0
    for rows in groups_of(frame).values():
        S = sigma ** 2 * cor_of(rows)
        e = y[rows] - mu[rows]
        total += (stats.multivariate_normal.logpdf(e, np.zeros(rows.size), S) if nu is None else
                  stats.multivariate_t.logpdf(e, np.zeros(rows.size), S, df=nu))
    return total


def reference_loo(frame, mu, sigma, cor_of, nu=None):
    """Each row's log density given its group's other rows, as the joint density less the other rows' marginal (SciPy)."""
    y, out = frame["y"].to_numpy(dtype=float), np.empty(len(frame))

    def logpdf(e, S):
        if e.size == 0:
            return 0.0
        return (stats.multivariate_normal.logpdf(e, np.zeros(e.size), S) if nu is None else
                stats.multivariate_t.logpdf(e, np.zeros(e.size), S, df=nu))

    for rows in groups_of(frame).values():
        S, e = sigma ** 2 * cor_of(rows), y[rows] - mu[rows]
        whole = logpdf(e, S)
        for j, i in enumerate(rows):
            keep = np.arange(rows.size) != j
            out[i] = whole - logpdf(e[keep], S[np.ix_(keep, keep)])
    return out


def unstr_cor(L, frame=FRAME):
    times = np.sort(frame["tt"].unique())
    C = L @ L.T
    return lambda rows: C[np.ix_(*(np.searchsorted(times, frame["tt"].to_numpy()[rows]),) * 2)]


def cosy_cor(rho):
    return lambda rows: cs_matrix(rho, rows.size)


def logp_fn(model, var=None, jacobian=True):
    """The log density of one variable (the response by default) at a value point, as a float."""
    pmm = model.pymc_model
    fn = pmm.compile_fn(pmm.logp(vars=[pmm[var or model.names["y"]]], jacobian=jacobian), inputs=pmm.value_vars, point_fn=True,
                        on_unused_input="ignore")
    return lambda point: float(np.asarray(fn(point)).ravel()[0])


def mu_of(frame, a, bx):
    return a + bx * (frame["x"].to_numpy() - frame["x"].mean())


# -- the term ---------------------------------------------------------------------------------------------------------
def test_the_term():
    s = parse_formula(UNSTR, FRAME)
    assert s.timecor == tc.TimeCor("unstr", "tt", "subj") and s.timecor.columns == ("tt", "subj")
    assert s.timecor.prior_class == "cortime" and s.autocor is None
    assert s.terms == ("x",) and not s.groups
    assert s.canonical_formula == UNSTR and s.bambi_formula == "y ~ x"
    for replica in (copy.deepcopy(s), ab.ModelSpec(**asdict(s))):         # asdict hands the term back as a mapping
        assert replica == s and replica.timecor == s.timecor and hash(replica) == hash(s)
    c = parse_formula("y ~ cosy(tt, subj) + (1 | h) + x", FRAME)           # R's positional arguments; the term is written last
    assert c.timecor == tc.TimeCor("cosy", "tt", "subj") and c.timecor.prior_class == "cosy"
    assert c.canonical_formula == "y ~ x + (1 | h) + cosy(time = tt, gr = subj)" and c.bambi_formula == "y ~ x + (1 | h)"
    assert hash(c) != hash(parse_formula("y ~ x + (1 | h)", FRAME))
    assert parse_formula("y ~ x", FRAME).timecor is None


@pytest.mark.parametrize("piece, expected", [
    ("unstr(tt, subj)", tc.TimeCor("unstr", "tt", "subj")),
    ("unstr(time = tt, gr = subj)", tc.TimeCor("unstr", "tt", "subj")),
    ("unstr(gr = subj, tt)", tc.TimeCor("unstr", "tt", "subj")),          # R: the named argument first, then the others in order
    ("cosy( time=tt,gr=subj )", tc.TimeCor("cosy", "tt", "subj")),
    ("cosy (tt, gr = subj)", tc.TimeCor("cosy", "tt", "subj")),
])
def test_brms_arguments(piece, expected):
    assert tc.parse_timecor(piece, "f") == expected
    assert tc.parse_timecor(expected.term, "f") == expected                # the canonical spelling reads back as itself


FORMULA_REFUSALS = [
    ("y ~ x + unstr(tt, subj, 3)", r"too many arguments; brms's unstr\(\) takes time, gr"),
    ("y ~ x + unstr(tt, subj, cov = TRUE)", r"unstr\(\) has no argument 'cov'; brms's unstr\(\) takes time, gr"),
    ("y ~ x + unstr(time = tt, time = w)", r"the argument 'time' is given twice"),
    ("y ~ x + cosy(gr = subj)", r"cosy\(\) needs its time column here \(brms's cosy\(\) defaults to the data order and one group; .*D146"),
    ("y ~ x + cosy()", r"cosy\(\) needs its time column here"),
    ("y ~ x + unstr(tt)", r"unstr\(\) needs its gr column here"),
    ("y ~ x + unstr(tt, NA)", r"unstr\(\) needs its gr column here"),
    ("y ~ x + unstr(tt, subj:h)", r"gr = subj:h: gr must be one column name here \(brms's gr = g:h groups the rows by two columns.*D146"),
    ("y ~ x + unstr(log(tt), subj)", r"time = log\(tt\): time must be one column name here \(docs/unstr-cosy.md"),
    ("y ~ x + unstr(tt, tt)", r"the time column cannot also be the grouping column gr"),
    ("y ~ x + unstr(tt, subj) + cosy(tt, subj)", r"one residual correlation term per formula in this slice .*D146"),
    ("y ~ x + unstr(y, subj)", r"the response 'y' cannot be the time or grouping column of a residual correlation term"),
    ("y ~ x + cosy(tt, y)", r"the response 'y' cannot be the time or grouping column"),
    ("y | weights(w) ~ x + unstr(tt, subj)", r"with the response addition term\(s\) weights\(w\) is not supported in this slice: no "
                                             r"frozen brms case verifies the pair .*D146"),
    ("y ~ gp(w) + unstr(tt, subj)", r"next to the Gaussian process 'gp\(w\)' is not supported in this slice: .*D146"),
    ("y ~ x + me(w, h) + cosy(tt, subj)", r"next to the term 'me\(w, h\)' is not supported"),
    ("y ~ x + ar(tt, subj) + unstr(tt, subj)", r"next to the autocorrelation term 'ar\(time = tt, gr = subj, p = 1\)'"),
    ("y ~ x + sar(M) + cosy(tt, subj)", r"the residual correlation term 'cosy\(time = tt, gr = subj\)' next to the SAR term 'sar\(M"),
    ("y ~ x + car(W, gr = site) + cosy(tt, subj)", r"the residual correlation term 'cosy\(time = tt, gr = subj\)' next to the CAR term"),
    ("y ~ x + fcor(V) + unstr(tt, subj)", r"the residual correlation term 'unstr\(time = tt, gr = subj\)' next to the fixed residual "
                                          r"covariance 'fcor\(V\)' is not supported"),
    ("y ~ x + (1 | mm(subj, h)) + unstr(tt, subj)", r"the multi-membership term .* next to the residual correlation term "
                                                    r"'unstr\(time = tt, gr = subj\)' is not supported"),
]


@pytest.mark.parametrize("formula, message", FORMULA_REFUSALS)
def test_formula_refusals(formula, message):
    with pytest.raises(ab.FormulaError, match=message):
        parse_formula(formula)


def test_a_piece_that_is_not_a_call():
    with pytest.raises(ab.FormulaError, match=r"a residual correlation term is written unstr\(time, gr\) or cosy\(time, gr\)"):
        tc.parse_timecor("unstr(tt, subj)^2", "f")


def test_the_gaussian_and_student_families_take_the_term():
    """D146: brms's unstr() and cosy() are verified here for the Gaussian and Student-t families (brms also takes others); a
    mixture's linear predictors are mu1, mu2, ... and brms refuses the term there too."""
    assert {name for name, spec in FAMILIES.items() if spec.time_correlation is not None} == {"gaussian", "student"}
    assert parse_formula(UNSTR, None, "student").timecor == tc.TimeCor("unstr", "tt", "subj")
    for family, formula in (("poisson", "cnt ~ x + unstr(tt, subj)"), ("lognormal", "y ~ x + cosy(tt, subj)")):
        with pytest.raises(ab.FormulaError, match=rf"the residual correlation term '.*' is not available for the {family} family; "
                                                  r"brms's .*\(\) is verified here for the gaussian, student families .*D146"):
            parse_formula(formula, None, family)
    with pytest.raises(ab.FormulaError, match=r"the trials column 'size' cannot be the time or grouping column"):
        parse_formula("k | trials(size) ~ x + unstr(size, subj)", None, "binomial")
    with pytest.raises(ab.FormulaError, match=r"a residual correlation term in a mixture \(cosy\(time = tt, gr = subj\)\) is not "
                                              r"supported, as in brms \('Explicit covariance terms can only be specified on mu'\)"):
        parse_formula(COSY, None, "mixture")


def test_distributional_formulas_refuse_the_term():
    """D146: next to a distributional formula (brms accepts bf(y ~ x + unstr(...), sigma ~ z)) and inside one (brms refuses
    that too: 'Explicit covariance terms can only be specified on mu')."""
    pr = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1)},
                   cortime=ab.LKJ(2))
    with pytest.raises(ab.FormulaError, match=r"the residual correlation term 'unstr\(time = tt, gr = subj\)' with a distributional "
                                              r"formula is not supported in this slice \(brms accepts .*D146"):
        ab.gaussian([UNSTR, "sigma ~ x"], FRAME, pr)
    with pytest.raises(ab.FormulaError, match=r"a residual correlation term in a distributional formula \(cosy\(time = tt, gr = subj\)\) "
                                              r"is not supported: it belongs to the response formula, as in brms"):
        ab.gaussian(["y ~ x", "sigma ~ x + cosy(tt, subj)"], FRAME, pr)


_S1 = FRAME.index[FRAME.subj == "s1"]
DATA_REFUSALS = [
    (FRAME.assign(tt=FRAME.tt.where(FRAME.index != _S1[1], FRAME.tt[_S1[0]])), rf"rows \d+ and \d+ of group 's1' have the same time "
                                                                              rf"{float(FRAME.tt[_S1[0]])}; time points within a group "
                                                                              r"must be unique \(brms's rule too\)"),
    (FRAME.assign(tt=[str(t) for t in FRAME.tt]), r"time column 'tt' of 'unstr\(time = tt, gr = subj\)' has dtype \w+; it must be "
                                                   r"numeric \(brms orders any column, text as text.*D146"),
    (FRAME.assign(tt=FRAME.tt.astype(float).where(FRAME.index != 3)), r"time column 'tt' of .* has missing or non-finite values"),
    (FRAME.assign(subj=FRAME.subj.where(FRAME.index != 3)), r"missing"),
]


@pytest.mark.parametrize("frame, message", DATA_REFUSALS)
def test_data_refusals(frame, message):
    with pytest.raises(ab.FormulaError, match=message):
        parse_formula(UNSTR, frame)


def test_a_column_the_term_reads_must_be_there():
    with pytest.raises(ab.FormulaError, match=r"columns not in data: \['tt'\] \(the residual correlation term 'unstr\(time = tt, gr = "
                                              r"subj\)' reads it\)"):
        parse_formula(UNSTR, FRAME.drop(columns="tt"))
    with pytest.raises(ab.FormulaError, match=r"columns not in data: \['subj'\]"):
        tc.time_layout(tc.TimeCor("unstr", "tt", "subj"), FRAME.drop(columns="subj"))


def test_columns_the_term_shares():
    """The time column may also be a predictor and the grouping column a grouping factor (brms takes both)."""
    m = unstr_model("y ~ tt + (1 | subj) + unstr(time = tt, gr = subj)", sd=ab.HalfNormal(1))
    assert m.spec.terms == ("tt",) and m.spec.groups == ("subj",) and m.design.timecor.groups == ("s1", "s2", "s3", "s4", "s5")


# -- the groups -------------------------------------------------------------------------------------------------------
def test_the_groups_are_brms_sorted_rows():
    d = tc.time_layout(tc.TimeCor("unstr", "tt", "subj"), FRAME)
    want = FRAME.sort_values(["subj", "tt"]).index.to_numpy()               # brms's order_data(): by group, then time
    np.testing.assert_array_equal(d.order, want)
    assert d.groups == ("s1", "s2", "s3", "s4", "s5") and d.times == (1.0, 2.0, 3.0, 4.0) and d.dim == 4 and d.n == 14
    np.testing.assert_array_equal(d.nobs, [4, 4, 3, 1, 2])
    np.testing.assert_array_equal(d.begin, [0, 4, 8, 11, 12])
    np.testing.assert_array_equal(d.group_index, [int(g[1]) - 1 for g in FRAME.subj])
    np.testing.assert_array_equal(d.index, FRAME.tt.to_numpy() - 1)
    # brms's Jtime_tg of these groups (make_standata pads a short group with 0, as gauss_unstr_site_varint_post records)
    np.testing.assert_array_equal(d.Jtime, [[1, 2, 3, 4], [1, 2, 3, 4], [1, 3, 4, 0], [2, 0, 0, 0], [1, 2, 0, 0]])
    assert d.labels == ["1", "2", "3", "4"]
    assert [key for key, _ in d.patterns] == [(0, 1, 2, 3), (0, 2, 3), (1,), (0, 1)]
    groups = groups_of(FRAME)
    np.testing.assert_array_equal(d.patterns[0][1], [groups["s1"], groups["s2"]])
    np.testing.assert_array_equal(d.patterns[1][1], [groups["s3"]])
    c = tc.time_layout(tc.TimeCor("cosy", "tt", "subj"), FRAME.assign(tt=FRAME.tt * 10.0 - 3))   # cosy: positions, any times
    np.testing.assert_array_equal(c.order, want)
    assert c.times == () and c.dim == 4 and c.labels == ["1", "2", "3", "4"]
    assert [key for key, _ in c.patterns] == [(0, 1, 2, 3), (0, 1, 2), (0,), (0, 1)]
    odd = tc.time_layout(tc.TimeCor("unstr", "tt", "subj"), FRAME.assign(tt=FRAME.tt * 0.25 + 0.1))
    assert odd.labels == ["0.35", "0.6", "0.85", "1.1"]                      # the shortest decimal of each time


# -- the correlation's Cholesky factor ------------------------------------------------------------------------------------
@pytest.mark.parametrize("rho", [0.05, 0.4, 0.93])
def test_the_compound_symmetric_cholesky_factor(rho):
    import pytensor

    for dim in (1, 2, 5):
        want = np.linalg.cholesky(cs_matrix(rho, dim))
        np.testing.assert_allclose(tc.cosy_cholesky_numpy(rho, dim), want, rtol=1e-13, atol=1e-15)
        r = pytensor.tensor.dscalar("r")
        got = pytensor.function([r], tc.cosy_cholesky(r, dim))(rho)
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-15)
    batch = tc.cosy_cholesky_numpy(np.array([[rho, 0.2]]), 3)                # any leading shape
    assert batch.shape == (1, 2, 3, 3)
    np.testing.assert_allclose(batch[0, 1], np.linalg.cholesky(cs_matrix(0.2, 3)), rtol=1e-13)


# -- the density ----------------------------------------------------------------------------------------------------------
def test_the_unstr_density_is_scipys_per_group():
    """Groups with every time (Lcortime itself), without a middle time (a Cholesky factor of their rows and columns of
    Cortime), of one row at a later time and of a leading block; SciPy's multivariate normal of each group's residuals."""
    m = unstr_model()
    fn = logp_fn(m)
    for a, bx, sigma, L in ((0.3, 0.7, 0.9, L4), (-0.5, 1.4, 2.1, np.eye(4)), (1.0, -0.2, 0.4, lkj_cholesky_rng(1.5, 4, np.random.default_rng(3)))):
        vp = m.value_point(intercept=a, b=[bx], sigma=sigma, Lcortime=L)
        want = reference_logp(FRAME, mu_of(FRAME, a, bx), sigma, unstr_cor(L))
        assert fn(vp) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("nu", [None, 4.0])
def test_the_student_densities_are_scipys_multivariate_t(nu):
    """The Student-t kernel: the multivariate t with scale matrix sigma^2 Cor_g and one nu, learned (nu=None) or fixed."""
    for structure in ("unstr", "cosy"):
        extra = ({"cortime": ab.LKJ(2)} if structure == "unstr" else {"cosy": ab.Flat()})
        priors = ab.Priors(**LINEAR, **extra, nu=None if nu is not None else ab.Gamma(2, 0.1))
        m = ab.student(UNSTR if structure == "unstr" else COSY, FRAME, priors, nu=nu)
        fn = logp_fn(m)
        value = dict(Lcortime=L4) if structure == "unstr" else dict(cosy=0.35)
        vp = m.value_point(intercept=0.2, b=[0.6], sigma=0.8, **value, **({} if nu is not None else {"nu": 5.5}))
        cor = unstr_cor(L4) if structure == "unstr" else cosy_cor(0.35)
        want = reference_logp(FRAME, mu_of(FRAME, 0.2, 0.6), 0.8, cor, nu=5.5 if nu is None else nu)
        assert fn(vp) == pytest.approx(want, rel=1e-12)


def test_the_cosy_density_and_its_prior():
    m = cosy_model(cosy=ab.Beta(2, 3))
    pmm = m.pymc_model
    assert pmm.rvs_to_values[pmm["cosy"]].name == "cosy_logodds__"
    vp = m.value_point(intercept=0.1, b=[0.5], sigma=1.3, cosy=0.62)
    assert logp_fn(m)(vp) == pytest.approx(reference_logp(FRAME, mu_of(FRAME, 0.1, 0.5), 1.3, cosy_cor(0.62)), rel=1e-12)
    assert logp_fn(m, "cosy", jacobian=False)(vp) == pytest.approx(stats.beta.logpdf(0.62, 2, 3), rel=1e-12)
    flat = cosy_model()
    vp = flat.value_point(intercept=0.1, b=[0.5], sigma=1.3, cosy=0.62)
    fp = flat.pymc_model
    assert fp.rvs_to_values[fp["cosy"]].name == "cosy_interval__"
    assert logp_fn(flat, "cosy")(vp) == pytest.approx(math.log(0.62 * 0.38), rel=1e-12)   # flat: Stan's logit Jacobian only


def test_the_density_and_its_gradient_with_group_effects():
    """A varying intercept next to the term (as gauss_unstr_site_varint_post); the gradient against central differences."""
    m = unstr_model("y ~ x + (1 | h) + unstr(tt, subj)", sd=ab.HalfNormal(1))
    pmm = m.pymc_model
    vp = m.value_point(intercept=0.2, b=[0.5], sigma=1.1, sd=0.7, z=np.array([0.4, -0.9]), Lcortime=L4)
    mu = mu_of(FRAME, 0.2, 0.5) + 0.7 * np.where(FRAME.h == "u", 0.4, -0.9)
    assert logp_fn(m)(vp) == pytest.approx(reference_logp(FRAME, mu, 1.1, unstr_cor(L4)), rel=1e-12)
    logp, dlogp = pmm.compile_logp(), pmm.compile_dlogp()
    grad, names, h = np.asarray(dlogp(vp)), [v.name for v in pmm.value_vars], 1e-6
    i = 0
    for name in names:
        flat = np.atleast_1d(vp[name]).astype(float).ravel()
        for j in range(flat.size):
            up, down = dict(vp), dict(vp)
            for point, step in ((up, h), (down, -h)):
                moved = flat.copy()
                moved[j] += step
                point[name] = moved.reshape(np.shape(vp[name]))
            assert grad[i] == pytest.approx((float(logp(up)) - float(logp(down))) / (2 * h), rel=1e-6, abs=1e-6), (name, j)
            i += 1


def test_the_likelihood_reads_every_term_of_the_linear_predictor():
    """mo(), offset() and correlated varying slopes next to the term: the joint density of the residuals from the graph's own
    mu (whose terms other features verify against brms), here with SciPy's multivariate normal."""
    priors = ab.Priors(**LINEAR, cortime=ab.LKJ(2), simo=ab.Dirichlet(1), cor=ab.LKJ(2),
                       sd={"h": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}})
    m = ab.gaussian("y ~ x + mo(ord) + offset(w) + (1 + x | h) + unstr(tt, subj)", FRAME, priors)
    pmm = m.pymc_model
    rng = np.random.default_rng(11)
    point = {k: np.asarray(v) + 0.4 * rng.standard_normal(np.shape(v)) for k, v in pmm.initial_point().items()}
    mu = pmm.compile_fn(pmm.replace_rvs_by_values([pmm["mu"]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")(point)
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = float(np.exp(point["sigma_log__"]))
    L = lkj_like(point["Lcortime_cholesky_corr__"])
    assert logp_fn(m)(point) == pytest.approx(reference_logp(FRAME, mu, sigma, unstr_cor(L)), rel=1e-12)


def lkj_like(u):
    from autobayes.correlation import cholesky_corr_constrain

    return cholesky_corr_constrain(u, 4)[0]


@pytest.mark.parametrize("build, rtol", [(unstr_model, 1e-12), (cosy_model, 1e-12),
                                         (lambda: unstr_model(family="student", nu=ab.Gamma(2, 0.1)), 1e-6)])
def test_the_jax_density_is_the_pytensor_one(build, rtol):
    jax = pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    m = build()
    pmm = m.pymc_model
    jf = get_jaxified_logp(pmm)
    point = pmm.initial_point()
    rng = np.random.default_rng(7)
    point = {k: np.asarray(v) + 0.3 * rng.standard_normal(np.shape(v)) for k, v in point.items()}
    names = [v.name for v in pmm.value_vars]
    assert float(jf([point[n] for n in names])) == pytest.approx(float(pmm.compile_logp()(point)), rel=1e-12)
    grads = jax.grad(lambda *a: jf(list(a)), argnums=tuple(range(len(names))))(*[point[n] for n in names])
    # the Student-t's nu: PyTensor's digamma is good to about 1e-10 relative (5.9e-10 at 1.7, against SciPy's), which nu's
    # gradient, a sum of nearly cancelling digamma terms, carries to 7.4e-8 here; JAX's is SciPy's to the last digits
    np.testing.assert_allclose(np.concatenate([np.ravel(g) for g in grads]), np.asarray(pmm.compile_dlogp()(point)), rtol=rtol, atol=1e-12)


def test_the_likelihood_needs_the_familys_parameters():
    """The likelihood Bambi calls takes the family's parameters by name; another set is a mismatch, not a silent model."""
    m = unstr_model()
    dist = m.bambi_model.family.likelihood.dist
    with pm.Model(), pytest.raises(RuntimeError, match=r"Bambi handed the likelihood \['mu'\]; the residual correlation term needs "
                                                      r"\['mu', 'sigma'\]"):
        dist("y", mu=np.zeros(14), observed=FRAME.y.to_numpy())


# -- priors ---------------------------------------------------------------------------------------------------------------
PRIOR_REFUSALS = [
    (UNSTR, dict(), r"the residual correlation term 'unstr\(time = tt, gr = subj\)' needs a prior on its correlation matrix: pass "
                    r"cortime=\.\.\., an LKJ prior"),
    (COSY, dict(), r"needs a prior on its correlation: pass cosy=\.\.\., Flat\(\) \(brms's default: uniform on \(0, 1\)\) or a Beta prior"),
    (UNSTR, dict(cortime=ab.Normal(0, 1)), r"prior for 'cortime' must be an LKJ prior, ab.LKJ\(eta\) \(brms's default lkj\(1\)\); got Normal"),
    (COSY, dict(cosy=ab.Normal(0, 1)), r"prior for 'cosy' must be Flat\(\) .* or a Beta prior; got Normal"),
    (COSY, dict(cosy=ab.Flat(), cortime=ab.LKJ(2)), r"do not pass cortime; 'cosy\(time = tt, gr = subj\)' has the class cosy"),
    (UNSTR, dict(cortime=ab.LKJ(2), cosy=ab.Flat()), r"do not pass cosy; 'unstr\(time = tt, gr = subj\)' has the class cortime"),
    ("y ~ x", dict(cortime=ab.LKJ(2)), r"do not pass cortime; the formula has no residual correlation term unstr\(\) or cosy\(\)"),
    ("y ~ x", dict(cosy=ab.Flat()), r"do not pass cosy; the formula has no residual correlation term"),
    (UNSTR, dict(cortime="lkj(2)"), r"prior for 'cortime' must be a Prior, got str"),
]


@pytest.mark.parametrize("formula, extra, message", PRIOR_REFUSALS)
def test_prior_refusals(formula, extra, message):
    with pytest.raises(ab.PriorError, match=message):
        ab.gaussian(formula, FRAME, ab.Priors(**LINEAR, **extra))


@pytest.mark.parametrize("formula, rows", [
    (UNSTR, [{"prior": "lkj(1)", "class": "cortime", "lb": "", "ub": ""}]),
    (COSY, [{"prior": "", "class": "cosy", "lb": "0", "ub": "1"}]),
])
def test_the_default_prior_table_is_brms(formula, rows):
    """brms 2.23's default_prior() rows (gauss_unstr, gauss_cosy): cortime lkj(1) without bounds; cosy flat with lb 0, ub 1."""
    priors = ab.default_priors(formula, FRAME)
    cls = rows[0]["class"]
    assert getattr(priors, cls).source == "default"
    assert getattr(priors, "cosy" if cls == "cortime" else "cortime") is None
    table = [r for r in priors.table(parse_formula(formula, FRAME), "gaussian") if r["class"] == cls]
    assert table == [{**dict.fromkeys(("coef", "group", "resp", "dpar", "nlpar", "tag"), ""), **rows[0], "source": "default"}]
    m = ab.gaussian(formula, FRAME, priors)
    assert cls in m.default_prior_labels and cls in m.describe() and "(brms default)" in m.describe()
    user = ab.Priors(**LINEAR, cosy=ab.Beta(2, 2)).table(parse_formula(COSY, FRAME), "gaussian")
    assert {"prior": "beta(2, 2)", "class": "cosy", "lb": "0", "ub": "1", "source": "user"}.items() <= next(
        r for r in user if r["class"] == "cosy").items()
    assert "cosy" not in ab.gaussian(COSY, FRAME, ab.Priors(**LINEAR, cosy=ab.Beta(2, 2))).default_prior_labels
    assert not ab.gaussian(COSY, FRAME, ab.Priors(**LINEAR, cosy=ab.Flat())).priors.improper(parse_formula(COSY, FRAME))


# -- value_point ----------------------------------------------------------------------------------------------------------
def test_value_point():
    m = unstr_model()
    vp = m.value_point(intercept=0.3, b=[0.7], sigma=0.9, Lcortime=L4)
    np.testing.assert_allclose(vp["Lcortime_cholesky_corr__"], cholesky_corr_free(L4))
    with pytest.raises(ValueError, match="Lcortime is required: the model has the term 'unstr\\(time = tt, gr = subj\\)'"):
        m.value_point(intercept=0.3, b=[0.7], sigma=0.9)
    with pytest.raises(ValueError, match="Lcortime must be a finite 4 x 4 matrix"):
        m.value_point(intercept=0.3, b=[0.7], sigma=0.9, Lcortime=np.eye(3))
    with pytest.raises(ValueError, match="this model has no cosy \\('unstr\\(time = tt, gr = subj\\)'\\); cosy does not apply"):
        m.value_point(intercept=0.3, b=[0.7], sigma=0.9, Lcortime=L4, cosy=0.5)
    c = cosy_model()
    assert c.value_point(intercept=0.3, b=[0.7], sigma=0.9, cosy=0.25)["cosy_interval__"] == pytest.approx(math.log(0.25 / 0.75))
    for bad in (0.0, 1.0, np.nan, [0.2, 0.3]):
        with pytest.raises(ValueError, match="cosy must be a finite scalar strictly between 0 and 1"):
            c.value_point(intercept=0.3, b=[0.7], sigma=0.9, cosy=bad)
    with pytest.raises(ValueError, match="this model has no Lcortime"):
        c.value_point(intercept=0.3, b=[0.7], sigma=0.9, cosy=0.25, Lcortime=L4)
    plain = ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR))
    with pytest.raises(ValueError, match="this model has no cosy \\(no residual correlation term unstr\\(\\) or cosy\\(\\)\\)"):
        plain.value_point(intercept=0.3, b=[0.7], sigma=0.9, cosy=0.25)


# -- predictions and post-processing --------------------------------------------------------------------------------------
def draws(model, a=(0.3, 0.1), bx=(0.7, 0.4), sigma=(0.9, 1.3), structure=None, nu=None, S=None, u=None):
    """Draws held as a posterior: one chain of len(a) draws (or S copies of the first), raw-scale intercept, slope, sigma,
    the term's parameter (Lcortime and Cortime, or cosy), a learned nu and a varying intercept of h."""
    n = len(a) if S is None else S
    pick = (lambda v: np.asarray(v, dtype=float)[None, :]) if S is None else (lambda v: np.full((1, S), float(v[0])))
    post = {"Intercept": pick(a), "x": pick(bx), "sigma": pick(sigma)}
    coords, dims = {}, {}
    if "L" in model.timecor_names:
        L = np.stack([structure] * n)[None]
        post.update(Lcortime=L, Cortime=L @ np.swapaxes(L, -1, -2))
        coords.update(cortime_dim=model.design.timecor.labels, cortime_dim_T=model.design.timecor.labels)
        dims.update(Lcortime=["cortime_dim", "cortime_dim_T"], Cortime=["cortime_dim", "cortime_dim_T"])
    else:
        post["cosy"] = np.full((1, n), structure)
    if nu is not None:
        post["nu"] = np.full((1, n), nu)
    if u is not None:
        post.update({"1|h": np.broadcast_to(np.asarray(u, dtype=float), (1, n, 2)).copy(), "1|h_sigma": np.ones((1, n))})
        coords["h__factor_dim"], dims["1|h"] = ["u", "v"], ["h__factor_dim"]
    return az.from_dict(posterior=post, coords=coords, dims=dims)


def test_predictions_are_mu_on_the_fitted_rows():
    m = unstr_model("y ~ x + (1 | h) + unstr(tt, subj)", sd=ab.HalfNormal(1))
    idata = draws(m, structure=L4, u=(0.25, -0.15))
    a, bx = np.array([0.3, 0.1]), np.array([0.7, 0.4])
    fixed = a[:, None] + bx[:, None] * FRAME.x.to_numpy()                 # the raw-scale intercept and the raw x
    full = fixed + np.where(FRAME.h == "u", 0.25, -0.15)
    for include, want in ((True, full), (False, fixed)):
        np.testing.assert_allclose(ab.predict(m, idata, kind="linear", include_group_effects=include)["eta"].to_numpy()[0], want,
                                   rtol=1e-13)
        np.testing.assert_allclose(ab.predict(m, idata, kind="mean", include_group_effects=include)["mu"].to_numpy()[0], want,
                                   rtol=1e-13)
    with pytest.raises(ab.FormulaError, match=r"predictions on new data are not offered for a model with the residual correlation term "
                                              r"'unstr\(time = tt, gr = subj\)': .*divergence D146"):
        ab.predict(m, idata, newdata=FRAME, kind="mean")


@pytest.mark.parametrize("structure, value, family, nu", [("unstr", L4, "gaussian", None), ("cosy", 0.45, "gaussian", None),
                                                          ("cosy", 0.45, "student", 9.0), ("unstr", L4, "student", 9.0)])
def test_new_responses_are_drawn_jointly(structure, value, family, nu):
    """kind="response" draws each group's rows jointly: mean mu, covariance sigma^2 Cor_g (times nu / (nu - 2) for the
    Student-t), the rows of different groups uncorrelated."""
    extra = {"nu": ab.Gamma(2, 0.1)} if family == "student" else {}
    m = (unstr_model if structure == "unstr" else cosy_model)(family=family, **extra)
    idata = draws(m, a=(0.3,), bx=(0.7,), sigma=(0.9,), structure=value, nu=nu, S=40000)
    y = ab.predict(m, idata, kind="response", seed=11)["y"].to_numpy()[0]
    mu = 0.3 + 0.7 * FRAME.x.to_numpy()
    np.testing.assert_allclose(y.mean(0), mu, atol=0.03)
    cor = unstr_cor(value) if structure == "unstr" else cosy_cor(value)
    inflate = 1.0 if nu is None else nu / (nu - 2.0)
    for rows in groups_of(FRAME).values():
        np.testing.assert_allclose(np.cov(y[:, rows].T).reshape(rows.size, rows.size), 0.81 * inflate * cor(rows), atol=0.04)
    s1, s2 = groups_of(FRAME)["s1"], groups_of(FRAME)["s2"]
    assert abs(np.corrcoef(y[:, s1[0]], y[:, s2[0]])[0, 1]) < 0.03


@pytest.mark.parametrize("structure, value, family", [("unstr", L4, "gaussian"), ("cosy", 0.45, "gaussian"),
                                                      ("unstr", L4, "student"), ("cosy", 0.7, "student")])
def test_log_lik_is_each_rows_density_given_its_groups_other_rows(structure, value, family):
    """brms's log_lik() of the non-factorised model (gauss_unstr_post, student_cosy_post): the joint density less the
    marginal density of the group's other rows, here from SciPy's multivariate normal and t; loo reads it."""
    from autobayes.post_processing import loo_from_log_lik

    extra = {"nu": ab.Gamma(2, 0.1)} if family == "student" else {}
    m = (unstr_model if structure == "unstr" else cosy_model)(family=family, **extra)
    idata = draws(m, structure=value, nu=6.5 if family == "student" else None)
    ll = ab.log_lik(m, idata)
    assert ll.dims == ("chain", "draw", "__obs__") and ll.shape == (1, 2, 14)
    cor = unstr_cor(value) if structure == "unstr" else cosy_cor(value)
    for k, (a, bx, sigma) in enumerate(((0.3, 0.7, 0.9), (0.1, 0.4, 1.3))):
        want = reference_loo(FRAME, a + bx * FRAME.x.to_numpy(), sigma, cor, nu=6.5 if family == "student" else None)
        np.testing.assert_allclose(ll.to_numpy()[0, k], want, rtol=1e-11)
    res, reference = ab.loo(m, idata), loo_from_log_lik(ll)
    np.testing.assert_array_equal(res.pointwise.to_numpy(), reference.pointwise.to_numpy())


def test_log_lik_without_the_term_is_the_familys_own():
    """The branch of log_lik() a model without the term takes: each row's own Normal density."""
    m = ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR))
    post = az.from_dict(posterior={"Intercept": np.array([[0.3]]), "x": np.array([[0.7]]), "sigma": np.array([[0.9]])})
    np.testing.assert_allclose(ab.log_lik(m, post).to_numpy()[0, 0], stats.norm.logpdf(FRAME.y, 0.3 + 0.7 * FRAME.x, 0.9), rtol=1e-12)


def test_a_fixed_nu_and_a_single_row_group():
    """A fixed nu (student(..., nu = 4)); a group of one row is the family's own univariate density."""
    m = ab.student(COSY, FRAME, ab.Priors(**LINEAR, cosy=ab.Flat()), nu=4)
    ll = ab.log_lik(m, draws(m, structure=0.3)).to_numpy()[0]
    mu = 0.3 + 0.7 * FRAME.x.to_numpy()
    single = int(groups_of(FRAME)["s4"][0])
    assert ll[0, single] == pytest.approx(stats.t.logpdf(FRAME.y[single], 4, mu[single], 0.9), rel=1e-12)
    np.testing.assert_allclose(ll[0], reference_loo(FRAME, mu, 0.9, cosy_cor(0.3), nu=4.0), rtol=1e-11)


def test_bayes_R2_pairs_each_rows_response_with_its_own_mean():
    """bayes_R2 is var(mu) / (var(mu) + var(y - mu)) over the fitted rows as given (brms's misalignment on unsorted rows:
    test_brms_bayes_R2_pairs_the_rows_in_its_sorted_order)."""
    m = cosy_model()
    r2 = ab.bayes_R2(m, draws(m, structure=0.3)).to_numpy()[0]
    y = FRAME.y.to_numpy()
    for k, (a, bx) in enumerate(((0.3, 0.7), (0.1, 0.4))):
        mu = a + bx * FRAME.x.to_numpy()
        assert r2[k] == pytest.approx(np.var(mu, ddof=1) / (np.var(mu, ddof=1) + np.var(y - mu, ddof=1)), rel=1e-12)


# -- describe, names, identity, records, prior predictive -------------------------------------------------------------------
def test_describe_and_names():
    m = unstr_model("y ~ x + (1 + x | h) + unstr(tt, subj)", sd={"h": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}},
                    cor=ab.LKJ(2))
    text = m.describe()
    for line in ("with correlated varying slope(s) and residual correlation term unstr(time = tt, gr = subj)",
                 "timecor   : unstr(time = tt, gr = subj): 5 group(s) of 'subj' of 1 to 4 rows, each ordered by 'tt'; each row is its "
                 "times 1, 2, 3, 4 (brms's n_unique_t = 4: the rows and columns of Cortime); 4 distinct set(s)",
                 "Residual correlation (unstr(time = tt, gr = subj); brms unstr())",
                 "e_g ~ MultiNormal(0, sigma^2 Cor_g), Cor_g = Cortime[J_g, J_g]",
                 "Cortime ~ LKJ(eta=2) (brms lkj(2) on the correlation matrix, class cortime, placed on Lcortime as lkj_corr_cholesky)",
                 "Lcortime_cholesky_corr__ Lcortime, the Cholesky factor of Cortime, on Stan's cholesky_corr coordinates",
                 "Cortime = Lcortime Lcortime^T (brms's Cortime)"):
        assert line in text, line
    assert m.timecor_names == {"L": "Lcortime", "cor": "Cortime", "coordinate": "cortime_dim", "coordinate_T": "cortime_dim_T"}
    assert m.names["L"] == "L|h" and m.names["cor"] == "cor|h"              # the correlated group term keeps its roles
    assert m.names["Lcortime"] == "Lcortime" and m.names["Cortime"] == "Cortime"
    assert set(m.pymc_model.named_vars) >= {"Lcortime", "Cortime", "L|h", "cor|h"}
    s = cosy_model(family="student", cosy=ab.Beta(2, 2), nu=ab.Gamma(2, 0.1))
    text = s.describe()
    for line in ("Student-t model with residual correlation term cosy(time = tt, gr = subj)",
                 "each row is its positions 1..4 in time order (the largest group)",
                 "e_g ~ MultiStudentT(nu, 0, sigma^2 Cor_g), Cor_g = the compound-symmetric matrix of the group's size",
                 "cosy ~ Beta(alpha=2, beta=2)   (brms class cosy, lb 0, ub 1)",
                 "cosy_logodds__           logit(cosy), the compound-symmetric correlation (brms's cosy)"):
        assert line in text, line
    assert s.timecor_names == {"cosy": "cosy"} and s.names["cosy"] == "cosy"
    assert "cosy ~ Flat(): uniform on (0, 1), proper" in cosy_model().describe()
    plain = ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR))
    assert plain.timecor_names == {} and "timecor" not in plain.describe() and "Residual correlation" not in plain.describe()


def test_generated_names_must_not_collide():
    with pytest.raises(ab.FormulaError, match=r"generated variable/coordinate name collision: \['Cortime'\]"):
        unstr_model("y ~ Cortime + unstr(tt, subj)", frame=FRAME.assign(Cortime=FRAME.x))


def test_the_target_identity_holds_the_term():
    base = unstr_model().target_identity
    assert unstr_model().target_identity == base
    assert unstr_model(cortime=ab.LKJ(3)).target_identity != base
    assert unstr_model(frame=FRAME.assign(tt=FRAME.tt.map({1: 1, 2: 3, 3: 2, 4: 4}))).target_identity != base   # other times
    assert cosy_model().target_identity != base
    assert cosy_model(frame=FRAME.assign(tt=FRAME.tt * 3)).target_identity == cosy_model().target_identity   # the same positions
    assert ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR)).target_identity != base


def test_fit_records_and_summary_rows(monkeypatch):
    import autobayes.session as session

    m = cosy_model()
    post = draws(m, structure=0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert "cosy" in session._summary_rows(m, post)
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=1, draws=2,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["priors"]["cosy"] == "Flat()"
    assert rec["model"]["time_correlation"] == {"term": "cosy(time = tt, gr = subj)", "time": "tt", "gr": "subj", "groups": 5, "dim": 4,
                                                "names": {"cosy": "cosy"}}
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]
    u = unstr_model()
    rec = ab.fit_record(u, SimpleNamespace(**{**vars(fit), "target_identity": u.target_identity}),
                        panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["priors"]["cortime"] == "LKJ(eta=2)" and rec["model"]["time_correlation"]["dim"] == 4


@pytest.mark.parametrize("build", [lambda: unstr_model(), lambda: cosy_model(cosy=ab.Beta(2, 2))])
def test_prior_predictive_draws_each_group_jointly(build):
    pp = build().prior_predictive(draws=30, seed=4)
    assert pp.prior_predictive["y"].shape == (1, 30, 14) and np.isfinite(pp.prior_predictive["y"]).all()
    if "Lcortime" in pp.prior:
        L = pp.prior["Lcortime"].to_numpy()[0]
        np.testing.assert_allclose((L ** 2).sum(-1), 1.0, rtol=1e-12)
        np.testing.assert_allclose(pp.prior["Cortime"].to_numpy()[0], L @ np.swapaxes(L, -1, -2), rtol=1e-12)
    else:
        assert ((pp.prior["cosy"] > 0) & (pp.prior["cosy"] < 1)).all()


def test_the_draws_take_one_response_vector_at_a_time():
    """The likelihood's draws (PyMC's random of the response) are one vector per call; a batched size is refused."""
    from pymc.distributions.shape_utils import change_dist_size

    m = cosy_model()
    with pytest.raises(ValueError, match=r"the residual correlation term draws one response vector at a time"):
        pm.draw(change_dist_size(m.pymc_model["y"], (2,)), random_seed=1)


# -- the other features ---------------------------------------------------------------------------------------------------
def test_other_features_refuse_the_term():
    """D146: conditional_effects(), prior_draws(), a shrinkage prior on b, multivariate and non-linear models."""
    m = cosy_model()
    with pytest.raises(ValueError, match=r"conditional_effects\(\) of a model with the residual correlation term 'cosy\(time = tt, gr = "
                                         r"subj\)' is not offered"):
        ab.conditional_effects(m, draws(m, structure=0.3), "x")
    with pytest.raises(ValueError, match=r"prior_draws\(\) of a model with the residual correlation term 'cosy\(time = tt, gr = subj\)'"):
        ab.prior_draws(m)
    with pytest.raises(ab.PriorError, match=r"a residual correlation term unstr\(time = tt, gr = subj\), which no frozen case verifies"):
        ab.gaussian(UNSTR, FRAME, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Horseshoe(), sigma=ab.HalfNormal(1), cortime=ab.LKJ(2)))
    resp = {"y": dict(LINEAR), "w": dict(LINEAR)}
    with pytest.raises(ab.FormulaError, match=r"'unstr\(tt, subj\)': a residual correlation term in a multivariate formula is not "
                                              r"verified in this slice; .*D146"):
        ab.gaussian("mvbind(y, w) ~ x + unstr(tt, subj)", FRAME, ab.Priors(resp=resp, rescor=ab.LKJ(2)), rescor=True)
    with pytest.raises(ab.FormulaError, match=r"the residual correlation term 'cosy\(time = tt, gr = subj\)' in a multivariate model is "
                                              r"not verified in this slice; .*D146"):        # C06's list of formulas, one per response
        ab.gaussian(["y ~ x + cosy(tt, subj)", "w | mi() ~ x"], FRAME, ab.Priors(resp=resp), rescor=False)
    with pytest.raises(ab.PriorError, match=r"cortime and cosy are the priors of a residual correlation term .*; a multivariate model "
                                            r"takes neither"):
        ab.gaussian("mvbind(y, w) ~ x", FRAME, ab.Priors(resp=resp, rescor=ab.LKJ(2), cosy=ab.Flat()), rescor=True)
    nlpars = {"a": {"b": ab.Normal(0, 1)}, "b": {"b": ab.Normal(0, 1)}}
    frame = FRAME.assign(ypos=np.abs(FRAME.y) + 0.1)
    with pytest.raises(ab.FormulaError, match=r"the residual correlation term 'unstr\(time = tt, gr = subj\)' in a non-linear parameter's "
                                              r"formula is not supported, as in brms"):
        ab.gaussian(["ypos ~ a * exp(b * x)", "a ~ 1 + unstr(tt, subj)", "b ~ 1"], frame, ab.Priors(nlpars=nlpars, sigma=ab.HalfNormal(1)),
                    nl=True)
    with pytest.raises(ab.PriorError, match=r"cortime is the prior of a residual correlation term .*; a non-linear model takes no "
                                            r"residual correlation term \(divergence D146\)"):
        ab.gaussian(["ypos ~ a * exp(b * x)", "a ~ 1", "b ~ 1"], frame, ab.Priors(nlpars=nlpars, sigma=ab.HalfNormal(1), cortime=ab.LKJ(2)),
                    nl=True)


def test_mixtures_refuse_the_terms_priors_and_values():
    components = [ab.Priors(**LINEAR), ab.Priors(**LINEAR)]
    with pytest.raises(ab.PriorError, match=r"cortime and cosy are the priors of a residual correlation term .*; a mixture takes neither, "
                                            r"as in brms"):
        ab.mixture("y ~ x", FRAME, ab.Priors(components=components, theta=ab.Dirichlet(1), cortime=ab.LKJ(2)),
                   families=["gaussian", "gaussian"])
    m = ab.mixture("y ~ x", FRAME, ab.Priors(components=components, theta=ab.Dirichlet(1)), families=["gaussian", "gaussian"])
    with pytest.raises(ValueError, match="Lcortime and cosy are a residual correlation term's parameters; a mixture has none"):
        m.value_point(intercept=[0.0, 1.0], b=np.zeros((1, 2)), sigma=[1.0, 1.0], theta=[0.5, 0.5], cosy=0.3)


# -- the oracle's extension -----------------------------------------------------------------------------------------------
def _bundle(case):
    from tests.oracle import gates

    return gates.Bundle(case)


def _corrupted(bundle, mutate):
    from tests.oracle import fixture as fx

    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def test_the_oracle_compares_brms_sorted_rows_through_the_derived_permutation():
    """brms's Stan rows are the data sorted by (site, tt); the adapter derives that order from the data, puts the per-row
    arrays back in data order and leaves the per-group ones (N_tg, begin_tg, nobs_tg, Jtime_tg) brms's."""
    from tests.oracle import adapter
    from tests.oracle import fixture as fx

    b = _bundle("gauss_unstr_site_varint_post")
    raw = fx.Fixture.load("gauss_unstr_site_varint_post").standata
    order = adapter.brms_row_order(b.case, b.data, b.model.spec.timecor)
    np.testing.assert_array_equal(b.fixture.stan_order, order)
    np.testing.assert_array_equal(order, b.model.design.timecor.order)        # AutoBayes sorts the groups as brms does here
    assert not np.array_equal(order, np.arange(order.size))
    np.testing.assert_array_equal(b.fixture.standata["Y"][order], raw["Y"])
    np.testing.assert_array_equal(b.fixture.standata["Jtime_tg"], raw["Jtime_tg"])
    np.testing.assert_array_equal(b.model.design.timecor.Jtime, raw["Jtime_tg"])
    np.testing.assert_array_equal(b.model.design.timecor.nobs, raw["nobs_tg"])


def test_the_gates_see_a_corrupted_fixture():
    from tests.oracle import gates
    from tests.oracle.adapter import OracleMismatch

    b = _bundle("gauss_unstr_site_varint_post")
    for gate in gates.ALL:
        gate(b)

    def entry(name, index, value):
        def mutate(raw):
            values = raw["standata"][name]["values"]
            values[index] = value
        return mutate

    for mutate, message in ((entry("Jtime_tg", 0, 2), r"brms's Jtime_tg of group 1 is \[2, 3\]"),
                            (entry("nobs_tg", 0, 1), r"do not tile its 36 rows"),
                            (entry("N_tg", 0, 24), r"brms's 24 groups"),
                            (entry("n_unique_t", 0, 5), r"brms has n_unique_t = 5")):
        with pytest.raises(OracleMismatch, match=message):
            gates.check_design(_corrupted(b, mutate))

    def drop(raw):
        del raw["standata"]["Jtime_tg"]
    with pytest.raises(OracleMismatch, match=r"brms's standata has .*, not \["):
        gates.check_design(_corrupted(b, drop))

    def swap_groups(raw):                                     # brms's first group's two rows moved to the third group's place
        values = raw["standata"]["Y"]["values"]
        values[0], values[3] = values[3], values[0]
    with pytest.raises(OracleMismatch, match=r"brms's Y is not the response sorted by"):
        gates.check_design(_corrupted(b, swap_groups))

    def lcortime(raw):
        raw["points"][0]["u"]["values"][2] += 0.01
    with pytest.raises(OracleMismatch, match=r"point 0: Stan's cholesky_corr coordinate 0 of Lcortime is"):
        gates.check_gradient(_corrupted(b, lcortime))

    def cortime(raw):
        raw["points"][1]["pars"]["Cortime"]["values"][1] += 0.01
    with pytest.raises(OracleMismatch, match=r"point 1: the graph's Cortime differs from brms's"):
        gates.check_derived(_corrupted(b, cortime))

    c = _bundle("student_cosy_post")
    for gate in gates.ALL:
        gate(c)

    def cosy_step(raw):
        raw["upars"][2]["du"] = 1.0
    with pytest.raises(OracleMismatch, match=r"brms's probe moved the coordinate of cosy by 1.0"):
        gates.check_gradient(_corrupted(c, cosy_step))

    def cosy_u(raw):
        raw["points"][2]["u"]["values"][2] += 0.01
    with pytest.raises(OracleMismatch, match=r"point 2: logit\(cosy\) is"):
        gates.check_gradient(_corrupted(c, cosy_u))

    def cosy_L(raw):
        raw["points"][3]["pars"]["Lcortime"]["values"][1] += 0.01
    with pytest.raises(OracleMismatch, match=r"point 3: the graph's Lcortime differs from brms's"):
        gates.check_derived(_corrupted(c, cosy_L))

    def grouped(raw):
        raw["standata"]["begin_tg"]["values"][1] += 1
    with pytest.raises(OracleMismatch, match=r"do not tile"):
        gates.check_design(_corrupted(c, grouped))


def test_a_model_and_a_fixture_must_agree_on_the_term():
    from tests.oracle import adapter
    from tests.oracle.adapter import OracleMismatch

    b = _bundle("gauss_unstr")
    plain = ab.gaussian("y ~ x", b.data, ab.Priors(**LINEAR))
    with pytest.raises(OracleMismatch, match=r"brms has a residual correlation term's data"):
        adapter.check_timecor_design(b.fixture, plain)
    sar = _bundle("gauss_sar_lag")                         # a SAR model's N_tg (one group) is no residual correlation term's data
    assert "N_tg" in sar.fixture.standata and adapter.check_timecor_design(sar.fixture, sar.model) is None
    with pytest.raises(OracleMismatch, match=r"must be compared in data order"):
        from tests.oracle import fixture as fx
        adapter.check_timecor_design(fx.Fixture.load("gauss_unstr"), b.model)
    cosy = ab.gaussian("y ~ x + cosy(tt, subj)", b.data, ab.Priors(**LINEAR, cosy=ab.Flat()))
    with pytest.raises(OracleMismatch, match=r"brms samples Lcortime \(unstr\(\)\) but the AutoBayes model has no unstr\(\) term"):
        adapter.coordinate_map(b.fixture, cosy, adapter.match_columns(b.fixture, cosy), adapter.match_groups(b.fixture, cosy))
    c = _bundle("gauss_cosy")
    with pytest.raises(OracleMismatch, match=r"brms samples cosy but the AutoBayes model has no cosy\(\) term"):
        unstr = ab.gaussian("y ~ x + unstr(tt, subj)", c.data, ab.Priors(**LINEAR, cortime=ab.LKJ(2)))
        adapter.coordinate_map(c.fixture, unstr, adapter.match_columns(c.fixture, unstr), adapter.match_groups(c.fixture, unstr))


def test_brms_bayes_R2_pairs_the_rows_in_its_sorted_order():
    """D147 (upstream): on gauss_unstr_site_varint_post, whose rows are in neither site nor time order, brms 2.23's
    bayes_R2() equals var(epred) / (var(epred) + var(y - epred[sorted rows])) to 5e-16: each row's response paired with the
    mean of another row (brms's posterior_epred in its rows sorted by site and time), 0.0455 where the aligned R2 is 0.0381
    at the first point; ab.bayes_R2 is the aligned R2, from the means the oracle's gate G4 checks against brms's."""
    from tests.oracle import adapter
    from tests.oracle import fixture as fx

    b = _bundle("gauss_unstr_site_varint_post")
    epred, y = b.fixture.prediction("fitted", "epred"), b.model.design.y
    order = b.fixture.stan_order

    def r2(resid, fitted):
        vf = np.var(fitted, axis=-1, ddof=1)
        return vf / (vf + np.var(resid, axis=-1, ddof=1))

    brms = fx.decode(b.fixture.raw["post"]["bayes_R2"]).ravel()
    np.testing.assert_allclose(brms, r2(y - epred[:, order], epred), rtol=1e-14)
    assert np.max(np.abs(r2(y - epred, epred) - brms) / brms) > 0.2
    ours = ab.bayes_R2(b.model, adapter.draws(b.fixture, b.model, b.cmap, b.gmap)).to_numpy()[0]
    np.testing.assert_allclose(ours, r2(y - epred, epred), rtol=1e-12)
    assert brms[0] == pytest.approx(0.04549101, abs=1e-8) and ours[0] == pytest.approx(0.03811231, abs=1e-8)
