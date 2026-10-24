"""F02b: brms's simultaneous autoregressive term sar(M, type = "lag") and sar(M, type = "error") (docs/spatial-sar.md).

Every expectation here is computed independently of ``autobayes.sar``: the joint density from SciPy's multivariate normal with
the covariance ``sigma^2 (A^T A)^-1`` written out (brms's density is that plus ``n/2 log(2 pi)``, D142), the log determinant
from NumPy's ``slogdet`` rather than from the eigenvalues, the conditional densities of ``log_lik()`` from the covariance's
partitioned blocks, the expected responses by a dense inverse. Nothing samples (no MCMC): posterior draws are written by hand,
and response draws are PyMC's forward draws.
"""
from __future__ import annotations

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
import pytensor
import pytest
from scipy import stats
from scipy.special import expit

import autobayes as ab
from autobayes.families import FAMILIES
from autobayes.formula import ModelSpec, parse_formula
from autobayes.formula import check_data2                     # Tier F: shared with fcor() (F03a)
from autobayes.sar import (Sar, SarDesign, _solve, parse_sar, sar_design, sar_log_lik, sar_mean, sar_response)

# a 4 x 3 rook grid of sites, one row per site: the adjacency W, its symmetric normalisation D^-1/2 W D^-1/2 (eigenvalues in
# [-1, 1], the largest 1) and its row standardisation D^-1 W (not symmetric; the same eigenvalues: D^-1 W = D^-1/2 M D^1/2)
SITES = [(r, c) for r in range(4) for c in range(3)]
W = np.array([[1.0 if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 else 0.0 for b in SITES] for a in SITES])
DEG = W.sum(1)
M_SYM = W / np.sqrt(np.outer(DEG, DEG))
M_ROW = W / DEG[:, None]
N = len(SITES)
_rng = np.random.default_rng(20261024)
FRAME = pd.DataFrame({
    "y": np.round(_rng.normal(0.5, 1.2, N), 4), "x": np.round(_rng.normal(0.0, 1.0, N), 4), "w": np.round(_rng.uniform(0.5, 1.5, N), 4),
    "z": ["a", "b", "c"] * 4, "g": ["u", "v"] * 6, "t": np.arange(N), "k": [1, 2, 3, 2] * 3, "sdx": np.full(N, 0.3),
    "y2": np.round(_rng.normal(0.0, 1.0, N), 4),
})
LINEAR = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))
PARAMETER = {"lag": "lagsar", "error": "errorsar"}
LOG_2PI = math.log(2.0 * math.pi)


def priors(kind="lag", **extra):
    return ab.Priors(**{**LINEAR, PARAMETER[kind]: ab.Flat(), **extra})


def sar_model(kind="lag", M=M_SYM, formula=None, frame=None, **extra):
    formula = formula or f'y ~ x + sar(M, type = "{kind}")'
    return ab.gaussian(formula, FRAME if frame is None else frame, priors(kind, **extra), data2={"M": M})


@pytest.fixture(scope="module")
def lag():
    return sar_model("lag")


@pytest.fixture(scope="module")
def error_row():
    return sar_model("error", M=M_ROW)


def exact_logpdf(y, mu, sigma, rho, M, kind):
    """The exact joint log density of the SAR model: y ~ MultiNormal(E[y], sigma^2 (A^T A)^-1), A = I - rho M."""
    A = np.eye(len(y)) - rho * M
    cov = sigma ** 2 * np.linalg.inv(A.T @ A)
    mean = np.linalg.inv(A) @ mu if kind == "lag" else mu
    return float(stats.multivariate_normal(mean, cov).logpdf(y))


def conditional_log_lik(y, mu, sigma, rho, M, kind):
    """Each row's log density given every other row, from the covariance's blocks (the conditional normal)."""
    A = np.eye(len(y)) - rho * M
    S = sigma ** 2 * np.linalg.inv(A.T @ A)
    m = np.linalg.inv(A) @ mu if kind == "lag" else mu
    out = []
    for i in range(len(y)):
        rest = [j for j in range(len(y)) if j != i]
        w = np.linalg.solve(S[np.ix_(rest, rest)], S[rest, i])
        out.append(stats.norm.logpdf(y[i], m[i] + w @ (y[rest] - m[rest]), math.sqrt(S[i, i] - S[i, rest] @ w)))
    return np.array(out)


def centred_mu(model, a_c, bx):
    x = model.design.X_raw[:, 0]
    return a_c + (x - x.mean()) * bx


# -- the term ---------------------------------------------------------------------------------------------------------
def test_the_term():
    t = Sar("M", "lag")
    assert t.term == 'sar(M, type = "lag")' and t.parameter == "lagsar"
    assert Sar("W", "error").parameter == "errorsar"
    assert parse_sar("x", "f") is None and parse_sar("sarx(M)", "f") is None


@pytest.mark.parametrize("piece, expected", [
    ("sar(M)", Sar("M", "lag")),
    ('sar(M, type = "error")', Sar("M", "error")),
    ("sar(W, 'error')", Sar("W", "error")),
    ('sar(type = "lag", M = W)', Sar("W", "lag")),
    ('sar(M, type = "err")', Sar("M", "error")),                      # R's match.arg takes a prefix (brms accepts "err")
    ('sar(M, "l")', Sar("M", "lag")),
    ("sar( M.sym , type='lag' )", Sar("M.sym", "lag")),
])
def test_brms_arguments(piece, expected):
    assert parse_sar(piece, "f") == expected
    assert parse_sar(expected.term, "f") == expected                    # the canonical spelling reads back as itself


FORMULA_REFUSALS = [
    ('y ~ x + sar(M, type = "foo")', r"type = \"foo\": type must be one of \"lag\", \"error\" \(brms: 'arg' should be one of"),
    ("y ~ x + sar(M, type = lag)", r"type = lag: type must be one of"),
    ('y ~ x + sar(M, type = "")', r"type must be one of"),
    ("y ~ x + sar(M, k = 1)", r"sar\(\) has no argument 'k'; brms's sar\(\) takes M, type"),
    ("y ~ x + sar(M = W, M = V)", r"the argument 'M' is given twice"),
    ('y ~ x + sar(M, "lag", 3)', r"too many arguments; brms's sar\(\) takes M, type"),
    ("y ~ x + sar()", r"sar\(\) needs its weight matrix"),
    ('y ~ x + sar(type = "lag")', r"sar\(\) needs its weight matrix"),
    ("y ~ x + sar(M + 1)", r"M = M \+ 1: M must be the name of a matrix handed in data2"),
    ("y ~ x + sar(M):x", r"a SAR term is written sar\(M, type = \"lag\"\) or sar\(M, type = \"error\"\)"),
    ('y ~ x + sar(M) + sar(M, type = "error")', r"one SAR term per formula \(brms: 'Can only model one term per autocorrelation class'\)"),
    ("y ~ x + (1 | g) + sar(M)", r"next to the group-level term '\(1 \| g\)' is not supported in this slice: no frozen brms case verifies "
                                 r"the pair \(brms takes it\); a SAR model takes population-level terms here .*divergence D143"),
    ("y ~ x + offset(w) + sar(M)", r"next to the offset 'offset\(w\)' is not supported .*D143"),
    ("y | weights(w) ~ x + sar(M)", r"next to the response addition term 'weights\(w\)' is not supported .*D143"),
    ("y ~ mo(k) + sar(M)", r"next to the term 'mo\(k\)' is not supported .*D143"),
    ("y ~ x + mi(y2) + sar(M)", r"next to the term 'mi\(y2\)' is not supported .*D143"),
    ("y ~ w + me(x, sdx) + sar(M)", r"next to the term 'me\(x, sdx\)' is not supported .*D143"),
    ("y ~ x + gp(w) + sar(M)", r"next to the Gaussian process 'gp\(w\)' is not supported .*D143"),
    ("y ~ x + ar(t, g) + sar(M)", r"next to the autocorrelation term 'ar\(time = t, gr = g, p = 1\)' is not supported .*D143"),
]


@pytest.mark.parametrize("formula, message", FORMULA_REFUSALS)
def test_formula_refusals(formula, message):
    with pytest.raises(ab.FormulaError, match=message):
        parse_formula(formula)


def test_the_gaussian_family_alone_takes_the_term():
    """D143: brms's sar() is verified here for the Gaussian family only (brms also takes the Student-t); every other family,
    a mixture's included, is refused naming the construct."""
    assert [name for name, fam in FAMILIES.items() if fam.sar] == ["gaussian"]
    for family in ("student", "poisson", "mixture"):
        with pytest.raises(ab.FormulaError, match=rf"the SAR term 'sar\(M, type = \"lag\"\)' is not available for the {family} family; "
                                                  r"brms's sar\(\) is verified here for the gaussian family \(brms also takes the student "
                                                  r"family; docs/spatial-sar.md, divergence D143\)"):
            parse_formula("y ~ x + sar(M)", family=family)
    with pytest.raises(ab.FormulaError, match=r"not available for the mixture family"):
        ab.mixture("y ~ x + sar(M)", FRAME, ab.Priors(components=[ab.Priors(**LINEAR), ab.Priors(**LINEAR)], theta=ab.Dirichlet(1)),
                   families=["gaussian", "gaussian"])


def test_the_canonical_formula_and_the_spec():
    spec = parse_formula('y ~ x + sar(W, type = "err")', FRAME)
    assert spec.sar == Sar("W", "error") and spec.canonical_formula == 'y ~ x + sar(W, type = "error")'
    assert "sar" not in spec.bambi_formula                                  # Bambi never sees the term
    rebuilt = ModelSpec(**asdict(spec))                                    # rebuilt from a mapping (dataclasses.asdict output)
    assert rebuilt == spec and hash(rebuilt) == hash(spec) and rebuilt.sar == spec.sar
    assert hash(spec) != hash(parse_formula("y ~ x", FRAME))


def test_distributional_formulas_refuse_the_term():
    with pytest.raises(ab.FormulaError, match=r"the SAR term 'sar\(M, type = \"lag\"\)' with a distributional formula is not supported, "
                                              r"as in brms \('SAR models are not implemented when predicting 'sigma' or 'nu''.*D143"):
        parse_formula(["y ~ x + sar(M)", "sigma ~ w"])
    with pytest.raises(ab.FormulaError, match=r"a SAR term in a distributional formula \(sar\(M, type = \"lag\"\)\) is not supported: it "
                                              r"belongs to the response formula .*D143"):
        parse_formula(["y ~ x", "sigma ~ w + sar(M)"])


def test_multivariate_and_nonlinear_models_refuse_the_term():
    P = dict(LINEAR)
    with pytest.raises(ab.FormulaError, match=r"'sar\(M\)': a SAR term in a multivariate formula is not verified in this slice.*D143"):
        ab.gaussian("mvbind(y, y2) ~ x + sar(M)", FRAME, ab.Priors(resp={"y": P, "y2": P}), rescor=False)
    frame = FRAME.assign(xm=np.r_[np.nan, FRAME.x.to_numpy()[1:]])
    with pytest.raises(ab.FormulaError, match=r"'y2 ~ x \+ sar\(M\)': the SAR term 'sar\(M, type = \"lag\"\)' in a multivariate model is "
                                              r"not verified in this slice.*D143"):
        ab.gaussian(["y ~ mi(xm) + w", "xm | mi() ~ w", "y2 ~ x + sar(M)"], frame, ab.Priors(resp={"y": P, "xm": P, "y2": P}), rescor=False)
    for kwargs in (dict(rescor=False), dict(nl=True)):
        with pytest.raises(ab.FormulaError, match=r"data2 is read by a SAR term .* a non-linear or multivariate model does not take.*D143"):
            ab.gaussian("y ~ x", FRAME, priors(), data2={"M": M_SYM}, **kwargs)
        with pytest.raises(ab.FormulaError, match=r"data2 is read by a SAR term .*D143"):
            ab.default_priors("y ~ x", FRAME, data2={"M": M_SYM}, **kwargs)
    nl_priors = ab.Priors(nlpars={"a": {"b": ab.Normal(1, 1)}, "b": {"b": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1))
    frame = FRAME.assign(ypos=np.exp(FRAME.y))
    with pytest.raises(ab.FormulaError, match=r"sar\(\) at position 17 is brms's SAR term \(F02b\), not a function of the non-linear "
                                              r"expression; a non-linear model takes no SAR term in this slice.*D143"):
        ab.gaussian(["ypos ~ a * exp(b * x) + sar(M)", "a ~ 1", "b ~ 1"], frame, nl_priors, nl=True)
    with pytest.raises(ab.FormulaError, match=r"'a ~ 1 \+ sar\(M\)': the SAR term 'sar\(M, type = \"lag\"\)' in a non-linear parameter's "
                                              r"formula is not supported in this slice.*D143"):
        ab.gaussian(["ypos ~ a * exp(b * x)", "a ~ 1 + sar(M)", "b ~ 1"], frame, nl_priors, nl=True)


# -- the weight matrix ------------------------------------------------------------------------------------------------
def test_the_design_of_a_symmetric_and_a_row_standardised_matrix(lag, error_row):
    d = lag.design.sar
    assert isinstance(d, SarDesign) and (d.term, d.type, d.parameter, d.matrix, d.n) == ('sar(M, type = "lag")', "lag", "lagsar", "M", N)
    assert d.symmetric and not error_row.design.sar.symmetric
    np.testing.assert_array_equal(d.M, M_SYM)
    reference = np.sort(np.linalg.eigvals(M_SYM).real)[::-1]
    np.testing.assert_allclose(d.eigenvalues, reference, atol=1e-13)
    np.testing.assert_allclose(error_row.design.sar.eigenvalues, reference, atol=1e-12)   # D^-1 W is similar to M_SYM
    assert np.all(np.diff(d.eigenvalues) <= 0) and d.lower == d.eigenvalues[-1] and d.upper == d.eigenvalues[0]
    assert abs(d.upper - 1.0) < 1e-12 and d.lower > -1.0 - 1e-12
    frame_m = pd.DataFrame(M_SYM, index=[f"s{i}" for i in range(N)], columns=[f"s{i}" for i in range(N)])
    assert sar_model(M=frame_m).target_identity == lag.target_identity   # a labelled DataFrame reads as its values, in data order


DATA2_REFUSALS = [
    (None, r"the SAR term 'sar\(M, type = \"lag\"\)' reads the matrix 'M' from data2, and the model was given no data2: pass "
           r"data2=\{'M': matrix\}, an 12 x 12 matrix"),
    ({"W": M_SYM}, r"reads the matrix 'M' from data2, and the model was given data2 with \['W'\]"),
    ({"M": M_SYM, "W": M_SYM}, r"data2 holds \['W'\], which no term of the formula reads"),
    ({"M": M_SYM[:11, :11]}, r"data2\['M'\] of 'sar\(M, type = \"lag\"\)' has shape \(11, 11\); the model has 12 fitted rows, so M must be "
                             r"12 x 12 \(brms: 'Dimensions of 'M' for SAR terms must be equal to the number of observations'\)"),
    ({"M": M_SYM[:, :11]}, r"has shape \(12, 11\)"),
    ({"M": np.where(np.eye(N) == 1, np.nan, M_SYM)}, r"has missing or infinite entries \(brms refuses them too\)"),
    ({"M": M_SYM.astype(str)}, r"has dtype <U\d+; the SAR weight matrix must be real numbers"),
    ({"M": M_SYM + 0j}, r"has dtype complex128; the SAR weight matrix must be real numbers"),
    ({"M": 0.5 * np.roll(np.eye(N), 1, axis=1)}, r"has complex eigenvalues \(largest imaginary part 0.5\); the SAR log determinant .*brms "
                                                 r"passes complex ones to Stan, which refuses them.*D143"),
    ({"M": W}, r"the eigenvalues of data2\['M'\] of 'sar\(M, type = \"lag\"\)' lie in \[-3.\d+, 3.\d+\], outside \[-1, 1\]: brms bounds the "
               r"SAR parameter by the smallest and largest eigenvalue, and 1 - rho \* lambda is then not positive .*normalise M.*D143"),
    ({"M": np.zeros((N, N))}, r"the eigenvalues of data2\['M'\] of 'sar\(M, type = \"lag\"\)' are all 0: they bound no interval"),
]


@pytest.mark.parametrize("data2, message", DATA2_REFUSALS)
def test_data2_refusals(data2, message):
    with pytest.raises(ab.FormulaError, match=message):
        ab.gaussian("y ~ x + sar(M)", FRAME, priors(), data2=data2)


def test_data2_without_a_term_that_reads_it():
    spec = parse_formula("y ~ x")
    check_data2(spec, None)
    check_data2(spec, {})
    with pytest.raises(ab.FormulaError, match=r"data2 holds \['M'\], which no term of the formula reads \(data2 is read by sar\(M, \.\.\.\)"):
        ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR), data2={"M": M_SYM})
    with pytest.raises(ab.FormulaError, match=r"data2 must be a mapping from a matrix's name to the matrix, got list"):
        check_data2(spec, [M_SYM])
    with pytest.raises(ab.FormulaError, match=r"data2 must be a mapping from a matrix's name to the matrix, got tuple"):
        ab.gaussian("y ~ x + sar(M)", FRAME, priors(), data2=(M_SYM,))
    with pytest.raises(ab.FormulaError, match=r"has shape \(12, 12\); the model has 5 fitted rows"):
        sar_design(Sar("M", "lag"), {"M": M_SYM}, 5)


# -- the model --------------------------------------------------------------------------------------------------------
POINTS = [(0.3, 0.7, 0.9, 0.45), (-0.2, -1.1, 1.6, -0.8), (1.1, 0.2, 0.5, 0.95)]   # (a_c, b_x, sigma, rho)


@pytest.mark.parametrize("kind, M", [("lag", M_SYM), ("error", M_SYM), ("lag", M_ROW), ("error", M_ROW)])
def test_the_density_is_the_exact_one_without_the_constant(kind, M):
    """The graph's likelihood is the exact joint density plus n/2 log(2 pi), the constant brms 2.23 leaves out (D142); the
    full log density adds the priors, flat on rho, and the coordinates' Jacobians (log sigma; Stan's lower-upper transform)."""
    m = sar_model(kind, M=M)
    pmm, d = m.pymc_model, m.design.sar
    like = pmm.compile_logp(vars=[pmm[m.names["y"]]], jacobian=False)
    full, nojac = pmm.compile_logp(), pmm.compile_logp(jacobian=False)
    y = m.design.y
    for a_c, bx, sigma, rho in POINTS:
        vp = m.value_point(intercept=a_c, b=[bx], sigma=sigma, **{d.parameter: rho})
        mu = centred_mu(m, a_c, bx)
        brms = exact_logpdf(y, mu, sigma, rho, M, kind) + 0.5 * N * LOG_2PI
        assert float(like(vp)) == pytest.approx(brms, rel=1e-12, abs=1e-12)
        A = np.eye(N) - rho * M
        sign, logdet = np.linalg.slogdet(A)
        r = A @ y - mu if kind == "lag" else A @ (y - mu)
        assert sign > 0 and brms == pytest.approx(logdet - N * math.log(sigma) - 0.5 * r @ r / sigma ** 2, rel=1e-12)
        prior = stats.norm.logpdf(a_c, 0, 2) + stats.norm.logpdf(bx, 0, 1) + stats.halfnorm.logpdf(sigma, scale=1)
        assert float(nojac(vp)) == pytest.approx(prior + brms, rel=1e-12)
        t = (rho - d.lower) / (d.upper - d.lower)
        jacobian = math.log(sigma) + math.log(d.upper - d.lower) + math.log(t) + math.log1p(-t)
        assert float(full(vp)) == pytest.approx(prior + brms + jacobian, rel=1e-12)


@pytest.mark.parametrize("kind", ["lag", "error"])
def test_the_gradient_is_the_reference_densitys(kind):
    """dlogp against a central difference of the independent density in the value coordinates (Intercept, x, log sigma and
    rho's interval coordinate)."""
    m = sar_model(kind)
    d, y = m.design.sar, m.design.y
    names = [v.name for v in m.pymc_model.value_vars]

    def reference(values):
        u = dict(zip(names, values))
        a_c, bx, sigma = u["Intercept"], u["x"], math.exp(u["sigma_log__"])
        rho = d.lower + (d.upper - d.lower) * expit(u[f"{d.parameter}_interval__"])
        t = (rho - d.lower) / (d.upper - d.lower)
        prior = stats.norm.logpdf(a_c, 0, 2) + stats.norm.logpdf(bx, 0, 1) + stats.halfnorm.logpdf(sigma, scale=1)
        jac = math.log(sigma) + math.log(d.upper - d.lower) + math.log(t) + math.log1p(-t)
        return prior + jac + exact_logpdf(y, centred_mu(m, a_c, bx), sigma, rho, M_SYM, kind)

    dlogp = m.pymc_model.compile_dlogp()
    for a_c, bx, sigma, rho in POINTS:
        vp = m.value_point(intercept=a_c, b=[bx], sigma=sigma, **{d.parameter: rho})
        u = np.array([float(vp[k]) for k in names])
        h = 1e-6
        numeric = np.array([(reference(u + h * e) - reference(u - h * e)) / (2 * h) for e in np.eye(len(u))])
        np.testing.assert_allclose(np.asarray(dlogp(vp), dtype=float), numeric, rtol=1e-6, atol=1e-6)


def test_the_jax_density_is_the_pytensor_one(lag, error_row):
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    for m in (lag, error_row):
        d = m.design.sar
        jf, logp = get_jaxified_logp(m.pymc_model), m.pymc_model.compile_logp()
        names = [v.name for v in m.pymc_model.value_vars]
        for a_c, bx, sigma, rho in POINTS:
            vp = m.value_point(intercept=a_c, b=[bx], sigma=sigma, **{d.parameter: rho})
            assert float(jf([vp[k] for k in names])) == pytest.approx(float(logp(vp)), rel=1e-12)


def test_the_parameter_is_flat_on_its_interval(lag):
    """rho's own log density is 0 on (lower, upper) and -inf outside; its initial value is the midpoint."""
    pmm, d = lag.pymc_model, lag.design.sar
    rv = pmm[d.parameter]
    for value, expected in ((0.0, 0.0), (0.9, 0.0), (d.upper + 0.1, -np.inf), (d.lower - 0.1, -np.inf)):
        assert pytensor.function([], pm.logp(rv, np.float64(value)))() == expected
    start = pmm.initial_point()[f"{d.parameter}_interval__"]
    assert float(start) == pytest.approx(math.log((0.5 * (d.lower + d.upper) - d.lower) / (d.upper - 0.5 * (d.lower + d.upper))), abs=1e-12)


@pytest.mark.parametrize("kind", ["lag", "error"])
def test_forward_draws_are_the_joint_models(kind):
    """PyMC's draws of the response (its random function) at fixed parameters: the innovations recovered from them, A y - mu
    (lag) or A (y - mu) (error), divided by sigma, are standard normal and independent across rows."""
    m = sar_model(kind, M=M_ROW)
    d = m.design.sar
    fixed = pm.do(m.pymc_model, {"Intercept": 0.4, "x": 0.8, "sigma": 0.7, d.parameter: 0.6})
    ys = pm.draw(fixed[m.names["y"]], draws=4000, random_seed=11)
    assert ys.shape == (4000, N)
    mu, A = centred_mu(m, 0.4, 0.8), np.eye(N) - 0.6 * M_ROW
    z = (ys @ A.T - mu) / 0.7 if kind == "lag" else (ys - mu) @ A.T / 0.7
    assert np.abs(z.mean(0)).max() < 0.08
    assert np.abs(np.cov(z.T) - np.eye(N)).max() < 0.12
    np.testing.assert_array_equal(pm.draw(fixed[m.names["y"]], draws=3, random_seed=5), pm.draw(fixed[m.names["y"]], draws=3, random_seed=5))
    assert pm.draw(fixed[m.names["y"]], random_seed=5).shape == (N,)


# -- priors -----------------------------------------------------------------------------------------------------------
PRIOR_REFUSALS = [
    (dict(), r"the SAR term 'sar\(M, type = \"lag\"\)' has the parameter lagsar, which needs a prior: pass lagsar=ab.Flat\(\) \(brms class "
             r"lagsar, bounded by the smallest and largest eigenvalue of M; brms's default is flat\)"),
    (dict(lagsar=ab.Normal(0, 0.5)), r"prior for 'lagsar' must be ab.Flat\(\), brms's default, flat on the interval .*; got "
                                     r"Normal\(mu=0, sigma=0.5\): another prior on the bounded lagsar is not verified here.*D143"),
    (dict(lagsar="flat"), r"prior for 'lagsar' must be ab.Flat\(\)"),
    (dict(lagsar=ab.Flat(), errorsar=ab.Flat()), r"do not pass errorsar; the SAR term 'sar\(M, type = \"lag\"\)' has lagsar"),
]


@pytest.mark.parametrize("extra, message", PRIOR_REFUSALS)
def test_prior_refusals(extra, message):
    with pytest.raises(ab.PriorError, match=message):
        ab.gaussian("y ~ x + sar(M)", FRAME, ab.Priors(**LINEAR, **extra), data2={"M": M_SYM})


def test_the_priors_of_a_model_without_the_term():
    for name in ("lagsar", "errorsar"):
        with pytest.raises(ab.PriorError, match=rf"do not pass {name}; the formula has no SAR term sar\(M, type = \.\.\.\)"):
            ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR, **{name: ab.Flat()}))
    P = dict(LINEAR)
    with pytest.raises(ab.PriorError, match=r"lagsar and errorsar are the priors of a SAR term .*; a multivariate model takes neither.*D143"):
        ab.gaussian("mvbind(y, y2) ~ x", FRAME, ab.Priors(resp={"y": P, "y2": P}, lagsar=ab.Flat()), rescor=False)
    with pytest.raises(ab.PriorError, match=r"the priors of the response 'y' take the classes .*; unexpected \['errorsar'\]"):
        ab.gaussian("mvbind(y, y2) ~ x", FRAME, ab.Priors(resp={"y": {**P, "errorsar": ab.Flat()}, "y2": P}), rescor=False)
    with pytest.raises(ab.PriorError, match=r"do not pass lagsar; the formula has no SAR term"):
        ab.mixture("y ~ x", FRAME, ab.Priors(components=[ab.Priors(**P), ab.Priors(**P)], theta=ab.Dirichlet(1), lagsar=ab.Flat()),
                   families=["gaussian", "gaussian"])
    with pytest.raises(ab.PriorError, match=r"do not pass errorsar; the formula has no SAR term"):
        ab.gaussian(["ypos ~ a * exp(b * x)", "a ~ 1", "b ~ 1"], FRAME.assign(ypos=np.exp(FRAME.y)),
                    ab.Priors(nlpars={"a": {"b": ab.Normal(1, 1)}, "b": {"b": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1),
                              errorsar=ab.Flat()), nl=True)


def test_the_flat_prior_is_improper(lag):
    assert lag.priors.improper(lag.spec) == ["lagsar"]
    with pytest.raises(ValueError, match=r"the prior of \['lagsar'\] is flat"):
        lag.prior_predictive(draws=5, seed=1)


def test_brms_default_priors_and_table(lag):
    for kind in ("lag", "error"):
        defaults = ab.default_priors(f'y ~ x + sar(M, type = "{kind}")', FRAME, data2={"M": M_SYM})
        prior = getattr(defaults, PARAMETER[kind])
        assert prior == ab.Flat() and prior.source == "default" and getattr(defaults, PARAMETER[{"lag": "error", "error": "lag"}[kind]]) is None
        m = ab.gaussian(f'y ~ x + sar(M, type = "{kind}")', FRAME, defaults, data2={"M": M_SYM})
        rows = [r for r in m.prior_table() if r["class"] == PARAMETER[kind]]
        assert rows == [{"prior": "", "class": PARAMETER[kind], "coef": "", "group": "", "resp": "", "dpar": "", "nlpar": "",
                         "lb": "min_eigenMsar", "ub": "max_eigenMsar", "tag": "", "source": "default"}]
        assert [r["class"] for r in m.prior_table()] == ["Intercept", "b", "b", PARAMETER[kind], "sigma"]
    assert [r for r in lag.prior_table() if r["class"] == "lagsar"][0]["source"] == "user"
    with pytest.raises(ab.FormulaError, match=r"reads the matrix 'M' from data2, and the model was given no data2"):
        ab.default_priors("y ~ x + sar(M)", FRAME)
    with pytest.raises(ab.FormulaError, match=r"outside \[-1, 1\]"):
        ab.default_priors("y ~ x + sar(M)", FRAME, data2={"M": W})


# -- values, names, describe, identity, records -----------------------------------------------------------------------
def test_value_point(lag):
    d = lag.design.sar
    vp = lag.value_point(intercept=0.1, b=[0.2], sigma=1.0, lagsar=0.3)
    assert float(vp["lagsar_interval__"]) == pytest.approx(math.log((0.3 - d.lower) / (d.upper - 0.3)), rel=1e-14)
    for kwargs, message in ((dict(), r"lagsar is required: the term 'sar\(M, type = \"lag\"\)' has it"),
                            (dict(lagsar=1.5), r"lagsar must be a scalar strictly between -1.0\d* and 1.0\d* \(the smallest and largest "
                                               r"eigenvalue of M\), got 1.5"),
                            (dict(lagsar=[0.1, 0.2]), r"lagsar must be a scalar strictly between"),
                            (dict(lagsar=0.3, errorsar=0.3), r"this model has no errorsar \('sar\(M, type = \"lag\"\)' has lagsar\); "
                                                             r"errorsar does not apply")):
        with pytest.raises(ValueError, match=message):
            lag.value_point(intercept=0.1, b=[0.2], sigma=1.0, **kwargs)
    plain = ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR))
    with pytest.raises(ValueError, match=r"this model has no lagsar \(no SAR term sar\(M, type = \.\.\.\)\); lagsar does not apply"):
        plain.value_point(intercept=0.1, b=[0.2], sigma=1.0, lagsar=0.3)
    mix = ab.mixture("y ~ x", FRAME, ab.Priors(components=[ab.Priors(**LINEAR), ab.Priors(**LINEAR)], theta=ab.Dirichlet(1)),
                     families=["gaussian", "gaussian"])
    with pytest.raises(ValueError, match="lagsar and errorsar are a SAR term's parameters; a mixture has none"):
        mix.value_point(intercept=[0.0, 1.0], b=np.zeros((1, 2)), sigma=[1.0, 1.0], theta=[0.5, 0.5], errorsar=0.2)


def test_names(lag, error_row):
    assert lag.sar_names == {"lagsar": "lagsar"} and lag.names["lagsar"] == "lagsar" and "errorsar" not in lag.names
    assert error_row.sar_names == {"errorsar": "errorsar"}
    assert lag.names["linpred"] == "mu"                          # Bambi's mu is the linear predictor, brms's posterior_linpred
    assert ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR)).sar_names == {}
    with pytest.raises(ab.FormulaError, match=r"generated variable/coordinate name collision: \['lagsar'\]"):
        sar_model(formula="y ~ lagsar + sar(M)", frame=FRAME.assign(lagsar=FRAME.x))


def test_describe(lag, error_row):
    text = lag.describe()
    assert text.splitlines()[0] == 'AutoBayes M0 model with simultaneous autoregressive term sar(M, type = "lag") (population-level only)'
    assert ("  sar       : sar(M, type = \"lag\"): data2['M'], 12 x 12 (symmetric) over the fitted rows in data order (brms's Msar); "
            "its eigenvalues lie in [") in text
    assert "  lagsar ~ Flat() on (" in text and "(brms class lagsar, bounded by min_eigenMsar and max_eigenMsar" in text
    assert "A y = mu + e: y ~ MultiNormal(A^-1 mu, sigma^2 A^-1 A^-T), an autoregression of the responses" in text
    assert "with r = A y - mu" in text and 'predict(kind="mean") is A^-1 mu (brms\'s posterior_epred).' in text
    assert "y_i ~ Normal(mu_i, sigma)" not in text                # the Gaussian family's likelihood line is replaced
    assert "  lagsar_interval__        log((lagsar - lower) / (upper - lagsar))" in text
    other = error_row.describe()
    assert "12 x 12 (non-symmetric, with real eigenvalues)" in other and "an autoregression of the residuals" in other
    assert "with r = A (y - mu)" in other and "is mu (brms's posterior_epred fails here; divergence D142)" in other
    assert "  errorsar ~ Flat() on (" in other and "normal_errorsar_lpdf" in other
    defaults = ab.gaussian("y ~ x + sar(M)", FRAME, ab.default_priors("y ~ x + sar(M)", FRAME, data2={"M": M_SYM}), data2={"M": M_SYM})
    assert "  lagsar ~ Flat()   (brms default: flat, improper) on (" in defaults.describe()


def test_the_target_identity_holds_the_term(lag):
    assert sar_model("error").target_identity != lag.target_identity
    assert sar_model(M=M_ROW).target_identity != lag.target_identity
    assert sar_model().target_identity == lag.target_identity
    assert ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR)).target_identity != lag.target_identity


def draws(model, rho=(0.2, -0.5, 0.7, 0.1, -0.9, 0.95), a=0.3, bx=0.7, sigma=(0.9, 1.1, 0.8, 1.2, 1.0, 0.7)):
    """Two chains of three draws: the raw-scale intercept (as a posterior from fit() holds it), the slope, sigma and the SAR
    parameter, which differ by draw."""
    name = model.design.sar.parameter
    post = {"Intercept": np.full((2, 3), a), "x": np.full((2, 3), bx), "sigma": np.reshape(sigma, (2, 3)),
            name: np.reshape(rho, (2, 3))}
    return az.from_dict(posterior=post)


def test_fit_records_and_summary_rows(lag, monkeypatch):
    post = draws(lag)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = session._summary_rows(lag, post)
    assert "lagsar" in rows
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=lag.target_identity, idata=post, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=3,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(lag, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    d = lag.design.sar
    assert rec["model"]["priors"]["lagsar"] == "Flat()"
    assert rec["model"]["sar"] == {"term": 'sar(M, type = "lag")', "type": "lag", "matrix": "M", "n": N, "bounds": [d.lower, d.upper],
                                   "names": {"lagsar": "lagsar"}}
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


# -- predictions and post-processing ----------------------------------------------------------------------------------
@pytest.mark.parametrize("kind, M", [("lag", M_SYM), ("error", M_ROW)])
def test_predictions(kind, M):
    """linear: the linear predictor (brms's posterior_linpred); mean: A^-1 mu per draw (lag; brms's posterior_epred) or mu
    (error; D142); response: draws whose recovered innovations are the seeded standard normals times sigma."""
    m = sar_model(kind, M=M)
    post = draws(m)
    rho, sigma = post.posterior[m.design.sar.parameter].to_numpy(), post.posterior["sigma"].to_numpy()
    x = FRAME.x.to_numpy()
    eta = 0.3 + 0.7 * x
    lin = m.predict(post, kind="linear")["eta"].to_numpy()
    np.testing.assert_allclose(lin, np.broadcast_to(eta, (2, 3, N)), rtol=1e-14)
    mean = m.predict(post, kind="mean")["mu"].to_numpy()
    for c in range(2):
        for k in range(3):
            want = np.linalg.inv(np.eye(N) - rho[c, k] * M) @ eta if kind == "lag" else eta
            np.testing.assert_allclose(mean[c, k], want, rtol=1e-12, atol=1e-13)
    out = m.predict(post, kind="response", seed=4)
    np.testing.assert_array_equal(out["y"].to_numpy(), m.predict(post, kind="response", seed=4)["y"].to_numpy())
    z = np.random.default_rng(4).standard_normal((2, 3, N))
    for c in range(2):
        for k in range(3):
            A = np.eye(N) - rho[c, k] * M
            y = out["y"].to_numpy()[c, k]
            innovation = A @ y - eta if kind == "lag" else A @ (y - eta)
            np.testing.assert_allclose(innovation, sigma[c, k] * z[c, k], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out["mu"].to_numpy(), mean, rtol=1e-14)


def test_new_rows_are_refused(lag):
    with pytest.raises(ab.FormulaError, match=r"predictions on new data are not offered for a model with the SAR term 'sar\(M, type = "
                                              r"\"lag\"\)': its weight matrix M relates the fitted rows to one another.*D143"):
        lag.predict(draws(lag), newdata=FRAME.head(3), kind="mean")


@pytest.mark.parametrize("kind, M", [("lag", M_SYM), ("error", M_ROW)])
def test_log_lik_is_each_rows_density_given_the_others(kind, M):
    m = sar_model(kind, M=M)
    post = draws(m)
    rho, sigma = post.posterior[m.design.sar.parameter].to_numpy(), post.posterior["sigma"].to_numpy()
    ll = ab.log_lik(m, post)
    assert ll.dims == ("chain", "draw", "__obs__") and ll.shape == (2, 3, N)
    eta = 0.3 + 0.7 * FRAME.x.to_numpy()
    for c in range(2):
        for k in range(3):
            want = conditional_log_lik(m.design.y, eta, sigma[c, k], rho[c, k], M, kind)
            np.testing.assert_allclose(ll.to_numpy()[c, k], want, rtol=1e-10)
    from autobayes.post_processing import loo_from_log_lik, waic_from_log_lik
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pd.testing.assert_frame_equal(ab.loo(m, post).pointwise, loo_from_log_lik(ll).pointwise)
        pd.testing.assert_frame_equal(ab.waic(m, post).pointwise, waic_from_log_lik(ll).pointwise)
    with pytest.raises(ValueError, match=rf"bayes_R2\(\) of a model with the SAR term 'sar\(M, type = \"{kind}\"\)' is not offered: no frozen "
                                         r"brms case verifies it.*D143"):
        ab.bayes_R2(m, post)


def test_a_model_without_the_term_keeps_the_gaussian_likelihood():
    """The paths the term branches from: the Gaussian family's likelihood lines and its row-wise log_lik."""
    plain = ab.gaussian("y ~ x", FRAME, ab.Priors(**LINEAR))
    assert "  y_i ~ Normal(mu_i, sigma)" in plain.describe() and "sar" not in plain.describe()
    post = az.from_dict(posterior={"Intercept": np.full((1, 2), 0.3), "x": np.full((1, 2), 0.7), "sigma": np.array([[0.9, 1.2]])})
    ll = ab.log_lik(plain, post).to_numpy()
    eta = 0.3 + 0.7 * FRAME.x.to_numpy()
    np.testing.assert_allclose(ll[0, 1], stats.norm.logpdf(FRAME.y.to_numpy(), eta, 1.2), rtol=1e-12)


def test_the_helpers_broadcast_over_draws():
    d = sar_design(Sar("M", "lag"), {"M": M_SYM}, N)
    e = sar_design(Sar("M", "error"), {"M": M_SYM}, N)
    rho = np.array([[0.1, -0.4], [0.8, 0.0]])
    eta = np.random.default_rng(1).normal(size=(2, 2, N))
    for c in range(2):
        for k in range(2):
            inv = np.linalg.inv(np.eye(N) - rho[c, k] * M_SYM)
            np.testing.assert_allclose(_solve(rho, eta, d)[c, k], inv @ eta[c, k], rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(sar_mean(eta, rho, d)[c, k], inv @ eta[c, k], rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(sar_response(eta, eta, rho, e)[c, k], eta[c, k] + inv @ eta[c, k], rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(sar_response(eta, eta, rho, d)[c, k], inv @ (2 * eta[c, k]), rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(sar_mean(eta, rho, e), eta)
    np.testing.assert_allclose(_solve(0.5, eta[0, 0], d), np.linalg.solve(np.eye(N) - 0.5 * M_SYM, eta[0, 0]), rtol=1e-14)
    ll = sar_log_lik(FRAME.y.to_numpy(), eta, np.full((2, 2), 0.8), rho, e)
    np.testing.assert_allclose(ll[1, 0], conditional_log_lik(FRAME.y.to_numpy(), eta[1, 0], 0.8, 0.8, M_SYM, "error"), rtol=1e-10)


def test_the_other_features_refuse_the_term(lag):
    post = draws(lag)
    with pytest.raises(ValueError, match=r"conditional_effects\(\) of a model with the SAR term 'sar\(M, type = \"lag\"\)' is not offered"):
        ab.conditional_effects(lag, post, "x")
    with pytest.raises(ValueError, match=r"the SAR term 'sar\(M, type = \"lag\"\)'"):
        ab.prior_draws(lag, draws=5, seed=1)
    with pytest.raises(ab.PriorError, match=r"Horseshoe\(\) on b is offered with population-level terms that are columns of the design only; "
                                            r"this formula has a SAR term sar\(M, type = \"lag\"\), which no frozen case verifies with it"):
        ab.gaussian("y ~ x + w + sar(M)", FRAME, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Horseshoe(), sigma=ab.HalfNormal(1),
                                                          lagsar=ab.Flat()), data2={"M": M_SYM})


# -- the oracle's extension to the SAR term ---------------------------------------------------------------------------
def _bundle(case="gauss_sar_lag", model=None):
    from tests.oracle import gates

    return gates.Bundle(case, model=model)


def _corrupted(bundle, mutate):
    from tests.oracle import fixture as fx

    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def test_the_gates_see_a_corrupted_sar_fixture():
    from tests.oracle import adapter, gates
    from tests.oracle.adapter import OracleMismatch

    b = _bundle()
    gates.check_design(b)
    gates.check_gradient(b)

    def msar(raw):
        raw["standata"]["Msar"]["values"][8] = 0.3

    def eigen(raw):
        raw["standata"]["eigenMsar"]["values"][3] += 1e-6

    def n_tg(raw):
        raw["standata"]["N_tg"]["values"] = [2]

    def declaration(raw):
        raw["declarations"]["lagsar"] = "real<lower=-1,upper=1> lagsar;"

    def no_msar(raw):
        del raw["standata"]["Msar"]

    for mutate, message in ((msar, r"brms's Msar \(\(25, 25\)\) differs from the model's weight matrix \(\(25, 25\)\) by 0.0499"),
                            (eigen, r"brms's eigenMsar differs from the model's eigenvalues of M"),
                            (n_tg, r"brms's N_tg is 2"),
                            (declaration, r"brms declares 'real<lower=-1,upper=1> lagsar;'; the AutoBayes model bounds lagsar"),
                            (no_msar, r"the AutoBayes model has the SAR term .*; brms's Stan data have no Msar / eigenMsar")):
        with pytest.raises(OracleMismatch, match=message):
            gates.check_design(_corrupted(b, mutate))

    def step(raw):
        next(e for e in raw["upars"] if e["par"] == "lagsar")["du"] = 0.2

    def coordinate(raw):
        raw["points"][2]["u"]["values"][2] += 1e-6

    for mutate, message in ((step, r"brms's probe moved the coordinate of lagsar by 0.2"),
                            (coordinate, r"point 2: log\(\(lagsar - lb\) / \(ub - lagsar\)\) is .*; brms's unconstrained value is")):
        with pytest.raises(OracleMismatch, match=message):
            gates.check_gradient(_corrupted(b, mutate))
    plain = ab.gaussian("y ~ x", b.data, ab.Priors(**LINEAR))
    with pytest.raises(OracleMismatch, match=r"brms has a SAR term \(Msar, eigenMsar\); the AutoBayes model has none"):
        adapter.check_sar_design(b.fixture, plain)
    adapter.check_sar_design(_bundle("gauss_ar1").fixture, _bundle("gauss_ar1").model)   # neither side has one


def test_the_density_at_brms_points_is_brms_plus_nothing_else():
    """At the frozen points of the three cases, the model's log density is brms's (G2), and brms's is the exact density plus
    n/2 log(2 pi), the constant its normal_lagsar_lpdf and normal_errorsar_lpdf leave out (D142, upstream)."""
    from tests.oracle import adapter

    for case, kind in (("gauss_sar_lag", "lag"), ("gauss_sar_error", "error"), ("gauss_sar_lag_post", "lag"), ("gauss_sar_error_post", "error")):
        b = _bundle(case)
        M = b.model.design.sar.M
        for p in b.fixture.points:
            nat = adapter.natural_point(b.model, p, b.cmap, b.gmap)
            rho = nat[PARAMETER[kind]]
            mu = centred_mu(b.model, nat["intercept"], nat["b"][0])
            exact = exact_logpdf(b.model.design.y, mu, nat["sigma"], rho, M, kind)
            assert p.lp_nojac - float(p.pars["lprior"]) - exact == pytest.approx(0.5 * len(mu) * LOG_2PI, abs=1e-9)
