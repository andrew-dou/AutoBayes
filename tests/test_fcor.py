"""brms's fixed residual covariance ``fcor(M)`` (brms-parity F03a, docs/fcor.md). The term and its parsing, ``data2`` and the
matrix's checks (brms's and Stan's refusals, probed on brms 2.23.0), the refusals of what no frozen case verifies with it
(D144) and of what brms refuses too, the likelihood against SciPy's multivariate normal and its rows against the sequential
conditionals written by hand, the gradient against the analytic one, the JAX density, the build-time check, default priors,
predictions (the fitted rows, joint draws, the new-data refusal, D145), ``describe()``, the target identity, fit records,
prior-predictive draws, the post-processing refusals (D145) and the oracle's extension to ``Mfcor``. Nothing here samples;
the brms comparison is the oracle cases ``gauss_fcor`` and ``gauss_fcor_varint``."""
import copy
import json
import warnings
from dataclasses import asdict
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytest
from scipy import stats

import autobayes as ab
from autobayes.families import FAMILIES
from autobayes.fcor import Fcor, FcorDesign, fcor_design, fcor_draws, fcor_family, fcor_values, parse_fcor
from autobayes.formula import check_data2, parse_formula

N = 8
FRAME = pd.DataFrame({
    "y": [0.42, -1.10, 0.35, 1.72, 0.08, -0.61, 1.15, 0.27],
    "x": [-0.62, -1.21, 1.13, -0.05, 0.42, 0.94, 0.82, -0.59],
    "z": [0.3, 1.2, -0.4, 0.8, -1.1, 0.5, 0.0, -0.7],
    "w": [1.0, 2.0, 1.5, 0.5, 1.0, 1.0, 2.5, 0.8],
    "t": [1, 2, 3, 4, 5, 6, 7, 8],
    "k": [0, 1, 2, 3, 1, 2, 0, 3],
    "g": ["a", "b", "c", "a", "b", "c", "a", "b"],
    "g2": ["b", "c", "a", "c", "a", "b", "b", "c"],
    "cnt": [1, 0, 3, 2, 5, 1, 0, 2],
    "sdx": [0.2] * N,
    "xm": [0.1, np.nan, 0.4, 0.2, -0.3, 0.5, np.nan, 0.0],
    "y2": [0.1, -0.4, 0.9, 1.1, -0.2, 0.3, 0.7, -0.5],
})
SCALES = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 0.8, 0.9, 1.0])
M = np.outer(SCALES, SCALES) * 0.4 ** np.abs(np.subtract.outer(np.arange(N), np.arange(N)))   # positive definite
LABELS = [str(i + 1) for i in range(N)]
LINEAR = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))


def gaussian(formula="y ~ x + fcor(V)", matrix=None, frame=None, **extra):
    data2 = {"V": pd.DataFrame(M, index=LABELS, columns=LABELS) if matrix is None else matrix}
    return ab.gaussian(formula, FRAME if frame is None else frame, ab.Priors(**{**LINEAR, **extra}), data2=data2)


def point(model, a=0.3, bx=0.7, sigma=0.9, u=None):
    """A value point of the model: the centred intercept, the slope, log sigma and, with (1 | g), the standardised effects."""
    names = model.pymc_model.initial_point()
    p = {"Intercept": np.float64(a), "x": np.array(bx), "sigma_log__": np.log(sigma)}
    if u is not None:
        p.update({"1|g_sigma_log__": np.log(0.5), "1|g_offset": np.asarray(u, dtype=float)})
    assert set(p) == set(names)
    return p


def sequential_rows(y, mu, S):
    """Each row's log density given the rows before it under y ~ MultiNormal(mu, S), written from the conditional normal."""
    out = []
    for i in range(len(y)):
        if i == 0:
            mean, var = mu[0], S[0, 0]
        else:
            w = np.linalg.solve(S[:i, :i], S[:i, i])
            mean, var = mu[i] + w @ (y[:i] - mu[:i]), S[i, i] - w @ S[:i, i]
        out.append(stats.norm.logpdf(y[i], mean, np.sqrt(var)))
    return np.array(out)


# -- the term ---------------------------------------------------------------------------------------------------------
def test_the_term():
    s = parse_formula("y ~ x + fcor(V) + (1 | g)", FRAME)
    assert s.fcor == Fcor("V") and s.fcor.term == "fcor(V)" and s.data2_names == ("V",)
    assert s.terms == ("x",) and s.groups == ("g",)
    assert s.canonical_formula == "y ~ x + (1 | g) + fcor(V)" and s.bambi_formula == "y ~ x + (1 | g)"
    for replica in (copy.deepcopy(s), ab.ModelSpec(**asdict(s))):         # asdict hands the term back as a mapping
        assert replica == s and replica.fcor == s.fcor and hash(replica) == hash(s)
    plain = parse_formula("y ~ x", FRAME)
    assert plain.fcor is None and plain.data2_names == () and hash(plain) != hash(s)
    assert parse_formula("y ~ x + fcor(M = V)").fcor == Fcor("V")            # brms's argument by name
    assert parse_formula("y ~ 1 + fcor( my.cov )").fcor == Fcor("my.cov")    # an R name, spaces around it


@pytest.mark.parametrize("piece, message", [
    ("fcor()", r"'fcor\(\)': fcor\(\) needs the matrix M, the name of a matrix in data2 \(brms: \"Argument 'M' is missing in fcor\(\)\""),
    ("fcor(V, W)", r"fcor\(\) takes one argument, M, the name of a matrix in data2"),
    ("fcor(cov = V)", r"fcor\(\) has no argument 'cov'; brms's fcor\(\) takes M"),
    ("fcor(V + W)", r"M = V \+ W: fcor\(\) reads its matrix from data2 by name"),
    ("fcor(diag(v))", r"M = diag\(v\): fcor\(\) reads its matrix from data2 by name"),
    ("fcor(V):x", r"'fcor\(V\):x': brms's fixed residual covariance is written fcor\(M\)"),
])
def test_the_term_as_written(piece, message):
    with pytest.raises(ab.FormulaError, match=message):
        parse_formula(f"y ~ x + {piece}")
    with pytest.raises(ab.FormulaError, match=message):
        parse_fcor(piece)


def test_the_gaussian_family_alone_takes_the_term():
    """D144: fcor() is verified for the Gaussian family; brms also takes it for student() (student_t_fcor_hom) and refuses it
    for every other family ('FCOR terms are not implemented for this family')."""
    assert {name for name, spec in FAMILIES.items() if spec.fcor} == {"gaussian"}
    for family, formula in (("student", "y ~ x + fcor(V)"), ("poisson", "cnt ~ x + fcor(V)"), ("lognormal", "w ~ x + fcor(V)")):
        with pytest.raises(ab.FormulaError, match=rf"fcor\(V\) is not available for the {family} family; the fixed residual "
                                                  r"covariance is verified here for the gaussian family \(brms also takes it for "
                                                  r"student\(\).*divergence D144"):
            parse_formula(formula, None, family)
    with pytest.raises(ab.FormulaError, match=r"fcor\(V\) is not available for the student family"):
        ab.student("y ~ x + fcor(V)", FRAME, ab.Priors(**LINEAR), nu=4)


FORMULA_REFUSALS = [
    ("y ~ x + fcor(V) + fcor(W)", r"'fcor\(V\)' and 'fcor\(W\)': one fcor\(\) term per formula \(brms: 'Can only model one term "
                                                  r"per autocorrelation class'"),
    ("y | weights(w) ~ x + fcor(V)", r"fcor\(V\) with the response addition term\(s\) weights\(w\) is not supported \(brms refuses "
                                     r"weights\(\) with it too: 'Invalid addition arguments for this model'.*D144"),
    ("y | trunc(lb = -5) ~ x + fcor(V)", r"with the response addition term\(s\) trunc\(lb = -5\) is not supported"),
    ("y ~ x + sar(W) + fcor(V)", r"fcor\(V\) next to the SAR term 'sar\(W, type = \"lag\"\)' is not supported in this slice: "
                                 r"no frozen brms case verifies the pair \(docs/fcor.md, divergence D144\)"),
    ("y ~ x + ar(t) + fcor(V)", r"fcor\(V\) next to the autocorrelation term 'ar\(time = t, p = 1\)' is not supported in this slice: "
                                r"no frozen brms case verifies the pair \(docs/fcor.md, divergence D144\)"),
    ("y ~ x + mo(k) + fcor(V)", r"fcor\(V\) next to the monotonic term 'mo\(k\)'"),
    ("y ~ x + gp(z) + fcor(V)", r"fcor\(V\) next to the Gaussian process 'gp\(z\)'"),
    ("y ~ me(z, sdx) + fcor(V)", r"fcor\(V\) next to the measurement-error term 'me\(z, sdx\)'"),
    ("y ~ x + mi(xm) + fcor(V)", r"fcor\(V\) next to the missing-value term 'mi\(xm\)'"),
    ("y ~ x + (1 | mm(g, g2)) + fcor(V)", r"fcor\(V\) next to the group term '\(1 \| mm\(g, g2\)\)'"),
]


@pytest.mark.parametrize("formula, message", FORMULA_REFUSALS)
def test_formula_refusals(formula, message):
    with pytest.raises(ab.FormulaError, match=message):
        parse_formula(formula)


def test_mixtures_and_distributional_formulas_refuse_the_term():
    """brms refuses fcor() in a mixture and in a distributional parameter's formula ('Explicit covariance terms can only be
    specified on mu'); next to one, brms's heterogeneous normal_fcor_het is not verified (D144)."""
    with pytest.raises(ab.FormulaError, match=r"fcor\(V\) in a mixture is not supported, as in brms \('Explicit covariance terms can "
                                              r"only be specified on mu'"):
        parse_formula("y ~ x + fcor(V)", None, "mixture")
    with pytest.raises(ab.FormulaError, match=r"'y ~ x \+ fcor\(V\)': fcor\(V\) with a distributional formula is not supported in this "
                                              r"slice \(brms accepts bf\(y ~ x \+ fcor\(M\), sigma ~ z\), its normal_fcor_het.*D144"):
        gaussian(["y ~ x + fcor(V)", "sigma ~ z"], sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 1)})
    with pytest.raises(ab.FormulaError, match=r"'sigma ~ z \+ fcor\(V\)': fcor\(V\) in a distributional formula is not supported: it "
                                              r"belongs to the response formula, as in brms"):
        parse_formula(["y ~ x", "sigma ~ z + fcor(V)"])


def test_multivariate_and_nonlinear_models_refuse_the_term():
    """brms gives each response of mvbind() its own Mfcor_y (D144); a list of response formulas reads each formula and refuses
    it there; a non-linear model takes no fcor() in its expression (outside the grammar) or in a parameter's formula."""
    resp = {"y": dict(LINEAR), "y2": dict(LINEAR)}
    with pytest.raises(ab.FormulaError, match=r"'fcor\(V\)': a fixed residual covariance in a multivariate formula is not verified in "
                                              r"this slice \(brms gives each response its own matrix, Mfcor_y and Mfcor_y2\).*D144"):
        ab.gaussian("mvbind(y, y2) ~ x + fcor(V)", FRAME, ab.Priors(resp=resp, rescor=ab.LKJ(2)), rescor=True)
    with pytest.raises(ab.FormulaError, match=r"'w ~ x \+ fcor\(V\)': fcor\(V\) in a multivariate model is not verified in this "
                                              r"slice; write it in a univariate model.*D144"):
        ab.gaussian(["y ~ mi(xm) + x", "xm | mi() ~ z", "w ~ x + fcor(V)"], FRAME, ab.Priors(resp={**resp, "xm": dict(LINEAR),
                                                                                                      "w": dict(LINEAR)}), rescor=False)
    with pytest.raises(ab.FormulaError, match=r"'xm \| mi\(\) ~ z \+ fcor\(V\)': fcor\(V\) with the response addition term\(s\) "
                                              r"mi\(\) is not supported"):
        ab.gaussian(["y ~ mi(xm) + x", "xm | mi() ~ z + fcor(V)"], FRAME, ab.Priors(resp=resp), rescor=False)
    nlpars = {"a": {"b": ab.Normal(1, 1)}, "b": {"b": ab.Normal(0, 1)}}
    priors = ab.Priors(nlpars=nlpars, sigma=ab.HalfNormal(1))
    with pytest.raises(ab.FormulaError, match=r"fcor\(\) at position \d+ is outside the grammar"):
        ab.gaussian(["y ~ a * exp(b * x) + fcor(V)", "a ~ 1", "b ~ 1"], FRAME, priors, nl=True)
    with pytest.raises(ab.FormulaError, match=r"'a ~ 1 \+ fcor\(V\)': fcor\(V\) in a non-linear parameter's formula is not supported: "
                                              r"a non-linear model takes no fixed residual covariance in this slice.*D144"):
        ab.gaussian(["y ~ a * exp(b * x)", "a ~ 1 + fcor(V)", "b ~ 1"], FRAME, priors, nl=True)
    for kwargs in (dict(nl=True), dict(rescor=False)):          # data2 belongs to a univariate linear model
        for build in (lambda: ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR), data2={"V": M}, **kwargs),
                      lambda: ab.default_priors("y ~ x", FRAME, data2={"V": M}, **kwargs)):
            with pytest.raises(ab.FormulaError, match=r"data2 is read by a SAR term sar\(M, type = \.\.\.\), a fixed residual "
                                                      r"covariance fcor\(M\) or a CAR term car\(W, gr = \.\.\.\), which a non-linear or "
                                                      r"multivariate model does not take in this slice .*divergences D141, D143, D144"):
                build()


def test_a_shrinkage_prior_is_refused_with_the_term():
    with pytest.raises(ab.PriorError, match=r"this formula has a fixed residual covariance fcor\(V\), which no frozen case verifies "
                                            r"with it .*D102"):
        gaussian(b=ab.Horseshoe(df=3))


# -- data2 and the matrix ---------------------------------------------------------------------------------------------
def test_the_matrix_as_brms_reads_it():
    """A data frame, an array, nested lists and a vector of variances (brms's diagonal matrix) are read by position: the labels
    are not read, as in brms."""
    term = Fcor("V")
    design = fcor_design(term, {"V": pd.DataFrame(M, index=LABELS, columns=LABELS)}, N)
    assert isinstance(design, FcorDesign) and design.n == N and design.term == "fcor(V)" and design.name == "V"
    np.testing.assert_array_equal(design.M, M)
    np.testing.assert_allclose(design.L @ design.L.T, M, rtol=1e-14, atol=1e-15)
    assert np.allclose(np.triu(design.L, 1), 0.0)
    np.testing.assert_allclose(2.0 * design.log_diagonal.sum(), np.linalg.slogdet(M)[1], rtol=1e-13)
    shuffled = pd.DataFrame(M, index=LABELS[::-1], columns=LABELS[::-1])      # other labels, the same entries
    for value in (M, M.tolist(), shuffled, M.astype(np.float32).astype(np.float64)):
        np.testing.assert_array_equal(fcor_design(term, {"V": value}, N).M, np.asarray(value, dtype=float))
    v = np.arange(1, N + 1, dtype=np.int64)
    np.testing.assert_array_equal(fcor_design(term, {"V": v}, N).M, np.diag(v.astype(float)))
    np.testing.assert_allclose(fcor_design(term, {"V": v}, N).L, np.diag(np.sqrt(v)), rtol=1e-15)


@pytest.mark.parametrize("data2, message", [
    (None, r"fcor\(V\) reads the matrix 'V' from data2, which is not given: pass data2=\{'V': matrix\} \(fcor\(\) reads its matrix "
           r"from data2; brms: \"Object 'V' was not found in 'data2'\""),
    ([("V", M)], r"data2 must be a mapping from a matrix's name to the matrix, got list"),
    ({"W": M}, r"fcor\(V\) reads the matrix 'V' from data2, which holds \['W'\]"),
    ({"V": np.full((N, N), "a")}, r"data2\['V'\] of fcor\(V\) has dtype <U1; the covariance matrix must be numeric"),
    ({"V": M.astype(bool)}, r"has dtype bool; the covariance matrix must be numeric"),
    ({"V": M[:6, :6]}, r"data2\['V'\] of fcor\(V\) has shape \(6, 6\); it must be 8 x 8, one row and column per fitted row, or a vector "
                       r"of 8 variances \(brms: \"Dimensions of 'M' for FCOR terms must be equal to the number of observations\""),
    ({"V": M[:, :6]}, r"has shape \(8, 6\); it must be 8 x 8"),
    ({"V": np.ones(5)}, r"has shape \(5, 5\); it must be 8 x 8"),
    ({"V": np.ones((2, 2, 2))}, r"has shape \(2, 2, 2\); it must be 8 x 8"),
    ({"V": np.where(np.eye(N) == 1, np.nan, M)}, r"data2\['V'\] of fcor\(V\) has missing or non-finite entries"),
    ({"V": M + np.triu(np.full((N, N), 1e-9), 1)}, r"data2\['V'\] of fcor\(V\) is not symmetric \(brms: \"'M' for FCOR terms must "
                                                   r"be symmetric\""),
    ({"V": M - 2.0 * np.eye(N)}, r"data2\['V'\] of fcor\(V\) is not positive definite: it has no Cholesky factor \(Stan's "
                                 r"cholesky_decompose refuses it"),
])
def test_matrix_refusals(data2, message):
    with pytest.raises(ab.FormulaError, match=message):
        fcor_design(Fcor("V"), data2, N)


def test_symmetry_is_read_to_rounding():
    """An asymmetry of a few ulps of the largest entry (a matrix computed in floating point) is rounding, not asymmetry; the
    matrix is kept as given."""
    nudged = M + np.triu(np.full((N, N), 1e-16), 1)
    assert fcor_design(Fcor("V"), {"V": nudged}, N).M[0, 1] == nudged[0, 1]


def test_data2_refusals():
    with pytest.raises(ab.FormulaError, match=r"data2 must be a mapping from a matrix's name to the matrix, got list"):
        ab.gaussian("y ~ x + fcor(V)", FRAME, ab.Priors(**LINEAR), data2=[M])
    with pytest.raises(ab.FormulaError, match=r"data2 holds \['W'\], which no term of the formula reads \(data2 is read by sar\(M, \.\.\.\), "
                                              r"fcor\(M\) and car\(W, \.\.\.\) here; docs/spatial-sar.md, docs/fcor.md, docs/spatial-car.md\)"):
        ab.gaussian("y ~ x + fcor(V)", FRAME, ab.Priors(**LINEAR), data2={"V": M, "W": M})
    with pytest.raises(ab.FormulaError, match=r"data2 holds \['V'\], which no term of the formula reads"):
        ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR), data2={"V": M})
    with pytest.raises(ab.FormulaError, match=r"reads the matrix 'V' from data2, which is not given"):
        ab.gaussian("y ~ x + fcor(V)", FRAME, ab.Priors(**LINEAR))
    check_data2(parse_formula("y ~ x"), None)                               # nothing to read, nothing given
    with pytest.raises(ab.FormulaError, match=r"data2 must be a mapping from a matrix's name to the matrix, got str"):
        check_data2(parse_formula("y ~ x + fcor(V)"), "V")


# -- the likelihood ---------------------------------------------------------------------------------------------------
def test_the_density_is_the_multivariate_normal():
    """The observed variable's rows are the sequential conditionals of MultiNormal(mu, sigma^2 M), written by hand, and sum
    to SciPy's density; with (1 | g) mu holds the group effects."""
    for formula, u in (("y ~ x + fcor(V)", None), ("y ~ x + (1 | g) + fcor(V)", [0.4, -0.3, 0.2])):
        m = gaussian(formula, sd=ab.HalfNormal(1)) if u is not None else gaussian(formula)
        pmm = m.pymc_model
        rows = pmm.compile_fn(pmm.logp(vars=pmm.observed_RVs, sum=False), inputs=pmm.value_vars, point_fn=True,
                              on_unused_input="ignore")
        for a, bx, sigma in ((0.3, 0.7, 0.9), (-1.2, 2.5, 0.35), (4.0, -3.0, 2.2)):
            p = point(m, a, bx, sigma, u)
            mu = a + bx * (FRAME.x - FRAME.x.mean()).to_numpy()
            if u is not None:
                mu = mu + 0.5 * pd.Series(dict(zip("abc", u)))[FRAME.g].to_numpy()
            y = FRAME.y.to_numpy()
            got = np.asarray(rows(p)[0], dtype=float).ravel()
            np.testing.assert_allclose(got, sequential_rows(y, mu, sigma ** 2 * M), rtol=1e-12)
            np.testing.assert_allclose(got.sum(), stats.multivariate_normal.logpdf(y, mu, sigma ** 2 * M), rtol=1e-13)
            np.testing.assert_allclose(fcor_values(y, mu, sigma, m.design.fcor), got.sum(), rtol=1e-13)


@pytest.mark.parametrize("formula, extra", [
    ("y ~ x + (1 + x | g) + fcor(V)", dict(cor=ab.LKJ(2))),                 # the correlated slopes' post-build edit of the graph
    ("y ~ x + z + offset(w) + (1 + x || g) + fcor(V)", {}),
])
def test_the_likelihood_reads_every_term_of_mu(formula, extra):
    """Group-level slopes, correlated (A02, whose graph edit rebuilds the model) or not, and an offset enter mu, which the
    joint likelihood reads: at a point off the initial one it is MultiNormal(y | mu, sigma^2 M) at the graph's mu and sigma."""
    m = gaussian(formula, sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}}, **extra)
    pmm = m.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm.logp(vars=[pmm[m.names["y"]]]), pmm["mu"], pmm["sigma"]]),
                        inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    p = {k: v + 0.1 * (1 + np.arange(np.size(v))).reshape(np.shape(v)) for k, v in pmm.initial_point().items()}
    like, mu, sigma = fn(p)
    assert np.ptp(mu) > 0.5
    np.testing.assert_allclose(float(like), stats.multivariate_normal.logpdf(FRAME.y, mu, float(sigma) ** 2 * M), rtol=1e-13)


def test_the_numpy_density_takes_draws():
    design = fcor_design(Fcor("V"), {"V": M}, N)
    y = FRAME.y.to_numpy()
    mu = np.stack([np.full(N, 0.2), np.linspace(-1, 1, N), np.zeros(N)])
    sigma = np.array([0.5, 1.0, 2.0])
    want = [stats.multivariate_normal.logpdf(y, mu[k], sigma[k] ** 2 * M) for k in range(3)]
    np.testing.assert_allclose(fcor_values(y, mu, sigma, design), want, rtol=1e-13)
    np.testing.assert_allclose(fcor_values(y, mu.reshape(1, 3, N), sigma.reshape(1, 3), design), [want], rtol=1e-13)


def test_the_gradient_is_the_analytic_one():
    """d/dIntercept = 1' S^-1 r, d/db = (x - mean x)' S^-1 r and d/dlog(sigma) = -n + r' M^-1 r / sigma^2, with S = sigma^2 M
    and r = y - mu, of the likelihood alone."""
    m = gaussian()
    pmm = m.pymc_model
    rv = pmm[m.names["y"]]
    like = pmm.replace_rvs_by_values([pmm.logp(vars=[rv])])[0]
    values = [pmm.rvs_to_values[pmm[name]] for name in ("Intercept", "x", "sigma")]
    fn = pytensor.function(values, pytensor.grad(like, values))
    xc, y = (FRAME.x - FRAME.x.mean()).to_numpy(), FRAME.y.to_numpy()
    for a, bx, sigma in ((0.3, 0.7, 0.9), (-1.2, 2.5, 0.35)):
        r = y - a - bx * xc
        s = np.linalg.solve(sigma ** 2 * M, r)
        got = fn(np.float64(a), np.array(bx), np.log(sigma))
        np.testing.assert_allclose([float(g) for g in got], [s.sum(), xc @ s, -N + r @ np.linalg.solve(M, r) / sigma ** 2], rtol=1e-12)


def test_the_jax_density_is_the_pytensor_one():
    pytest.importorskip("jax")
    m = gaussian("y ~ x + (1 | g) + fcor(V)", sd=ab.HalfNormal(1))
    pmm = m.pymc_model
    logp, dlogp = pmm.compile_logp(), pmm.compile_dlogp()
    jlogp, jdlogp = pmm.compile_logp(mode="JAX"), pmm.compile_dlogp(mode="JAX")
    for a, bx, sigma in ((0.3, 0.7, 0.9), (-1.2, 2.5, 0.35)):
        p = point(m, a, bx, sigma, [0.4, -0.3, 0.2])
        np.testing.assert_allclose(float(jlogp(p)), float(logp(p)), rtol=1e-13)
        np.testing.assert_allclose(np.asarray(jdlogp(p)), np.asarray(dlogp(p)), rtol=1e-12)


def test_the_build_checks_the_likelihood(monkeypatch):
    import autobayes.fcor as fcor

    monkeypatch.setattr(fcor, "fcor_values", lambda y, mu, sigma, design: -1.0)
    with pytest.raises(RuntimeError, match=r"the graph's likelihood of fcor\(V\) is -?\d.*; MultiNormal\(y \| mu, sigma\^2 V\) is -1.0"):
        gaussian()


def test_the_family_takes_the_gaussian_likelihood_alone():
    design = fcor_design(Fcor("V"), {"V": M}, N)
    with pytest.raises(ValueError, match=r"fcor\(\) needs the Gaussian likelihood's parameters \(mu, sigma\); this family's are \('mu',\)"):
        fcor_family("poisson", design)
    import bambi as bmb
    from bambi.defaults import get_builtin_family

    base = get_builtin_family("gaussian")
    family = fcor_family(base, design)                                       # a family object is copied, not changed
    assert family is not base and family.likelihood is not base.likelihood and family.likelihood.params == ["mu", "sigma"]
    assert isinstance(family, bmb.Family)


# -- priors -----------------------------------------------------------------------------------------------------------
def test_the_default_priors_have_no_class_of_the_term():
    """brms's default_prior() of y ~ x + fcor(V) is that of y ~ x (probed): the term adds no class; default_priors() reads and
    checks data2 as the constructor does."""
    data2 = {"V": M}
    with_term = ab.default_priors("y ~ x + (1 | g) + fcor(V)", FRAME, data2=data2)
    assert with_term == ab.default_priors("y ~ x + (1 | g)", FRAME)
    m = ab.gaussian("y ~ x + (1 | g) + fcor(V)", FRAME, with_term, data2=data2)
    assert "  sigma ~ HalfStudentT(nu=3, sigma=2.5)   (brms default student_t(3, 0, 2.5), lb 0)" in m.describe()
    with pytest.raises(ab.FormulaError, match=r"reads the matrix 'V' from data2, which is not given"):
        ab.default_priors("y ~ x + fcor(V)", FRAME)
    with pytest.raises(ab.FormulaError, match=r"has shape \(6, 6\); it must be 8 x 8"):
        ab.default_priors("y ~ x + fcor(V)", FRAME, data2={"V": M[:6, :6]})
    with pytest.raises(ab.FormulaError, match=r"data2 holds \['V'\], which no term of the formula reads"):
        ab.default_priors("y ~ x", FRAME, data2=data2)


# -- predictions ------------------------------------------------------------------------------------------------------
def draws(model, a=0.3, bx=0.7, sigma=(0.9, 1.1, 0.7), u=(0.25, -0.15, 0.1)):
    """Two chains of three draws: the raw-scale intercept (as a posterior from fit() holds it), the slope, sigma and, with a
    group term, the varying intercepts of g."""
    post = {"Intercept": np.full((2, 3), a), "x": np.full((2, 3), bx), "sigma": np.broadcast_to(np.array(sigma), (2, 3)).copy()}
    coords, dims = {}, {}
    if model.spec.groups:
        post.update({"1|g": np.broadcast_to(np.array(u), (2, 3, 3)), "1|g_sigma": np.ones((2, 3))})
        coords["g__factor_dim"], dims["1|g"] = ["a", "b", "c"], ["g__factor_dim"]
    return az.from_dict(posterior=post, coords=coords, dims=dims)


def test_predictions_on_the_fitted_rows():
    """The linear predictor and expected response are those without the term (brms's posterior_linpred and posterior_epred);
    kind="response" draws the rows jointly, mu + sigma L z with z standard normal from the seeded generator."""
    m = gaussian("y ~ x + (1 | g) + fcor(V)", sd=ab.HalfNormal(1))
    post = draws(m)
    fixed = 0.3 + 0.7 * FRAME.x.to_numpy()
    full = fixed + pd.Series({"a": 0.25, "b": -0.15, "c": 0.1})[FRAME.g].to_numpy()
    for kind, var in (("linear", "eta"), ("mean", "mu")):
        np.testing.assert_allclose(m.predict(post, kind=kind)[var], np.broadcast_to(full, (2, 3, N)), rtol=1e-13)
        np.testing.assert_allclose(m.predict(post, kind=kind, include_group_effects=False)[var], np.broadcast_to(fixed, (2, 3, N)),
                                   rtol=1e-13)
    out = m.predict(post, kind="response", seed=7)
    z = np.random.default_rng(7).standard_normal((2, 3, N))
    sigma = np.broadcast_to(np.array([0.9, 1.1, 0.7]), (2, 3))[..., None]
    L = np.linalg.cholesky(M)
    np.testing.assert_allclose(out["y"], full + sigma * np.einsum("ij,cdj->cdi", L, z), rtol=1e-13)
    np.testing.assert_allclose(out["mu"], np.broadcast_to(full, (2, 3, N)), rtol=1e-13)


def test_joint_draws_have_the_covariance():
    """fcor_draws: many draws of mu + sigma L z have mean mu and covariance sigma^2 M (to their Monte Carlo error); sigma may
    come per draw or per draw and row."""
    design = fcor_design(Fcor("V"), {"V": M}, N)
    mu = np.linspace(-1.0, 1.0, N)
    draws_ = fcor_draws(np.broadcast_to(mu, (40000, N)), np.full(40000, 1.5), design, np.random.default_rng(11))
    np.testing.assert_allclose(draws_.mean(0), mu, atol=0.03)
    np.testing.assert_allclose(np.cov(draws_.T), 2.25 * M, atol=0.05)
    same = fcor_draws(np.broadcast_to(mu, (5, N)), np.full((5, 1), 1.5), design, np.random.default_rng(3))
    np.testing.assert_array_equal(same, fcor_draws(np.broadcast_to(mu, (5, N)), np.full(5, 1.5), design, np.random.default_rng(3)))


def test_new_rows_are_refused():
    m = gaussian()
    post = draws(m)
    for kind in ("linear", "mean", "response"):
        with pytest.raises(ab.FormulaError, match=r"predictions on new data are not offered for a model with fcor\(V\): its matrix is "
                                                  r"the covariance of the fitted rows' residuals.*divergence D145\); predict on the "
                                                  r"fitted rows"):
            m.predict(post, newdata=FRAME, kind=kind)


# -- describe(), identity, records, prior predictive ------------------------------------------------------------------
def test_describe():
    text = gaussian().describe()
    for line in ("AutoBayes M0 model with fixed residual covariance fcor(V) (population-level only)",
                 "  formula   : y ~ x + fcor(V)",
                 "  fcor      : fcor(V): data2['V'], the 8 x 8 covariance of the residuals before the factor sigma^2 (brms's Mfcor), its "
                 "rows and columns the fitted rows in data order; diagonal 0.64 to 1.44, log det "
                 f"{np.linalg.slogdet(M)[1]:.6g}",
                 "  y ~ MultiNormal(mu, sigma^2 * V) over the 8 fitted rows: fcor(V), brms's normal_fcor_hom; V is",
                 "  Lfcor) gives z = L^-1 (y - mu) / sigma and log p(y) = sum_i (-z_i^2 / 2 - log sigma - log L_ii - log(2 pi) / 2)"):
        assert line in text, line
    assert "y_i ~ Normal(mu_i" not in text
    assert "  y_i ~ Normal(mu_i, sigma)" in ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR)).describe()


def test_the_target_identity_holds_the_matrix():
    base = gaussian().target_identity
    assert gaussian().target_identity == base
    assert gaussian(matrix=M.tolist()).target_identity == base                 # the same entries, however given
    assert gaussian(matrix=M * 1.5).target_identity != base
    assert gaussian(matrix=np.diag(np.diag(M))).target_identity != base
    assert ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR)).target_identity != base


def test_fit_records():
    m = gaussian()
    import autobayes.session as session
    fit = SimpleNamespace(target_identity=m.target_identity, idata=draws(m), backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=3,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    with warnings.catch_warnings(), pytest.MonkeyPatch.context() as mp:
        warnings.simplefilter("ignore", RuntimeWarning)
        mp.setattr(session, "_summary_rows", lambda *args: {})
        mp.setattr(session, "environment", lambda: {})
        rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["fcor"] == {"term": "fcor(V)", "matrix": "V", "rows": N}
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


def test_prior_predictive_draws_the_rows_jointly():
    """Each prior-predictive draw is MultiNormal(mu, sigma^2 M) at its own mu and sigma: whitened by L and sigma, the draws'
    residuals are standard normal and uncorrelated across rows (to their Monte Carlo error)."""
    m = gaussian()
    pp = m.prior_predictive(draws=4000, seed=5)
    y = pp.prior_predictive["y"].to_numpy()[0]
    mu, sigma = pp.prior["mu"].to_numpy()[0], pp.prior["sigma"].to_numpy()[0]
    z = np.linalg.solve(np.linalg.cholesky(M), (y - mu).T).T / sigma[:, None]
    np.testing.assert_allclose(z.mean(0), 0.0, atol=0.06)
    np.testing.assert_allclose(np.cov(z.T), np.eye(N), atol=0.07)


# -- post-processing --------------------------------------------------------------------------------------------------
def test_post_processing_is_refused():
    """D145: brms's log_lik() of the joint normal conditions each row on all the others; with no frozen case of it, log_lik,
    loo, waic and bayes_R2 are refused, and conditional_effects() and prior_draws() name the term; hypothesis() reads the
    coefficients alone and is offered."""
    m = gaussian()
    post = draws(m)
    for fn, what in ((ab.log_lik, "log_lik"), (ab.loo, "log_lik"), (ab.waic, "waic"), (ab.bayes_R2, "bayes_R2")):
        with pytest.raises(ValueError, match=rf"{what}\(\) of a model with fcor\(V\) is not offered: brms's log_lik\(\) of the joint "
                                             r"normal is each row's density given every other row.*divergence D145"):
            fn(m, post)
    with pytest.raises(ValueError, match=r"conditional_effects\(\) of a model with the fixed residual covariance 'fcor\(V\)' is not offered"):
        ab.conditional_effects(m, post, "x")
    with pytest.raises(ValueError, match=r"the fixed residual covariance 'fcor\(V\)'"):
        ab.prior_draws(m)
    h = ab.hypothesis(m, post, "x > 0")
    assert h.table["Estimate"].tolist() == [pytest.approx(0.7)]


# -- the oracle's extension -------------------------------------------------------------------------------------------
def _bundle(case="gauss_fcor", model=None):
    from tests.oracle import gates

    return gates.Bundle(case, model=model)


def _corrupted(bundle, mutate):
    from tests.oracle import fixture as fx

    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


@pytest.mark.parametrize("case", ["gauss_fcor", "gauss_fcor_varint"])
def test_the_gates_pass_and_see_a_corrupted_matrix(case):
    from tests.oracle import gates
    from tests.oracle.adapter import OracleMismatch

    b = _bundle(case)
    for gate in gates.ALL:
        gate(b)

    def entry(raw):
        raw["standata"]["Mfcor"]["values"][1] += 1e-9
    with pytest.raises(OracleMismatch, match=r"brms's Mfcor \(\(36, 36\)\) differs from fcor\(V\)'s matrix \(\(36, 36\)\) by 1"):
        gates.check_design(_corrupted(b, entry))

    def shape(raw):
        raw["standata"]["Mfcor"] = {"dims": [35, 35], "values": raw["standata"]["Mfcor"]["values"][:35 * 35]}
    with pytest.raises(OracleMismatch, match=r"brms's Mfcor \(\(35, 35\)\) differs from fcor\(V\)'s matrix \(\(36, 36\)\)$"):
        gates.check_design(_corrupted(b, shape))

    def blocks(raw):
        raw["standata"]["N_tg"]["values"][0] = 2
    with pytest.raises(OracleMismatch, match=r"brms has N_tg = .*2.*; fcor\(V\) is one block of every row \(N_tg = 1\)"):
        gates.check_design(_corrupted(b, blocks))

    def no_matrix(raw):
        del raw["standata"]["Mfcor"]
    with pytest.raises(OracleMismatch, match=r"the AutoBayes model has fcor\(V\); brms's standata has no Mfcor"):
        gates.check_design(_corrupted(b, no_matrix))

    def point_(raw):
        raw["points"][2]["pars"]["sigma"]["values"][0] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match="point 2: log density"):
        gates.check_logp(_corrupted(b, point_))

    scaled = adapter_model(b, b.data2["V"] * 1.01)                           # another matrix: G1 and G2 fail
    with pytest.raises(OracleMismatch, match=r"brms's Mfcor .* differs from fcor\(V\)'s matrix"):
        gates.check_design(_bundle(case, model=scaled))
    with pytest.raises(OracleMismatch, match=r"point 0: log density"):
        gates.check_logp(_bundle(case, model=scaled))
    spec = b.case["autobayes"]
    plain = ab.gaussian(spec["formula"].replace(" + fcor(V)", ""), b.data, b.model.priors)
    with pytest.raises(OracleMismatch, match=r"brms has the fixed residual covariance Mfcor; the AutoBayes model has no fcor\(\) term"):
        gates.check_design(_bundle(case, model=plain))


def adapter_model(bundle, matrix):
    from tests.oracle import adapter

    spec = bundle.case["autobayes"]
    return adapter.build_model({**bundle.case, "autobayes": {**spec, "kwargs": {"data2": {"V": matrix}}}}, bundle.data)
