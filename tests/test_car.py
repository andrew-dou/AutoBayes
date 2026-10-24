"""Spatial CAR terms (brms-parity F02a): brms's ``car(M, gr, type)`` with ``type`` escar, esicar, icar and bym2. The term and
its parsing, the refusals (D140, D141), the adjacency matrix and the locations against brms's own Stan data, the BYM2 scaling
factor against an exact rational evaluation of its definition, each type's log density and gradient against an independent
transcription of the published densities (a dense multivariate normal for escar), the JAX density, priors and brms's default
priors, ``value_point``, predictions and post-processing, ``describe()``, names, the target identity, fit records, the term in
the contexts that refuse it, and the oracle's extension (corrupted fixtures). The Stan data and default-prior tables below were
printed by brms 2.23.0 (R 4.5.2) for ``FRAME`` and ``W``; nothing here samples. The brms comparison is the oracle cases
``gauss_car_escar``, ``gauss_car_esicar``, ``gauss_car_icar``, ``gauss_car_bym2`` and ``poisson_car_bym2``."""
import copy
import json
import math
import warnings
from fractions import Fraction
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import autobayes as ab
from autobayes import car as carmod
from autobayes.car import Car, bym2_scale, car_layout, car_values, components, parse_car
from autobayes.formula import FormulaError, ModelSpec, parse_formula


def grid(r, c, prefix="s"):
    """The 0/1 rook adjacency of an r by c grid, labelled s1, s2, ... row by row."""
    n = r * c
    W = np.zeros((n, n))
    for a in range(n):
        i, j = divmod(a, c)
        if i + 1 < r:
            W[a, a + c] = W[a + c, a] = 1
        if j + 1 < c:
            W[a, a + 1] = W[a + 1, a] = 1
    labels = [f"{prefix}{k}" for k in range(1, n + 1)]
    return pd.DataFrame(W, index=labels, columns=labels)


W = grid(3, 3)
SITES = ["s3", "s1", "s2", "s9", "s5", "s4", "s7", "s6", "s8", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
FRAME = pd.DataFrame({
    "y": [-0.3, 1.2, 0.4, -1.1, 0.8, 0.0, 2.1, -0.6, 0.5, 1.4, -0.2, 0.9, -1.7, 0.3, 1.0, -0.9, 0.2, 0.6],
    "x": np.linspace(-1.0, 1.0, 18), "site": SITES,
    "cnt": [1, 0, 3, 2, 5, 1, 0, 2, 4, 1, 2, 0, 3, 1, 1, 6, 2, 0],
    "g": ["a", "b", "c"] * 6, "w": np.round(np.linspace(0.1, 0.9, 18), 4), "k": [3, 1, 4, 2, 5, 1, 0, 2, 4] * 2,
    "n": [5] * 18,
})
# make_standata(y ~ x + car(W, gr = site, type = ...), FRAME, data2 = list(W = W)), brms 2.23.0: escar's and bym2's
BRMS = {"Jloc": [3, 1, 2, 9, 5, 4, 7, 6, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9], "edges1": [2, 4, 3, 5, 6, 5, 7, 6, 8, 9, 8, 9],
        "edges2": [1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7, 8], "Nneigh": [2, 3, 2, 3, 4, 3, 2, 3, 2],
        "eigenMcar": [0.99999999999999978, 0.57735026918962595, 0.57735026918962573, 1.4204320418277011e-16, 5.9445091065589311e-17,
                      -4.2037167569147664e-17, -0.57735026918962562, -0.57735026918962573, -1.0000000000000002],
        "car_scale": 0.40970896306544979}
# the same without the rows of s9: brms drops s9 from the graph, with its edges
BRMS_NO_S9 = {"Jloc": [3, 1, 2, 5, 4, 7, 6, 8, 1, 2, 3, 4, 5, 6, 7, 8], "edges1": [2, 4, 3, 5, 6, 5, 7, 6, 8, 8],
              "edges2": [1, 1, 2, 2, 3, 4, 4, 5, 5, 7], "Nneigh": [2, 3, 2, 3, 4, 2, 2, 2],
              "eigenMcar": [1.0000000000000004, 0.64549722436790258, 0.40824829046386296, 1.2160665340679751e-16,
                            -1.6167086729097125e-16, -0.40824829046386313, -0.64549722436790313, -1],
              "car_scale": 0.43272349083068218}
EXTRA = {"escar": {"car": ab.Flat()}, "esicar": {}, "icar": {}, "bym2": {"rhocar": ab.Beta(1, 1)}}
TYPES = ("escar", "esicar", "icar", "bym2")


def priors(type="icar", family="gaussian", **extra):
    base = dict(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sdcar=ab.HalfNormal(1), **EXTRA[type])
    if family == "gaussian":
        base["sigma"] = ab.HalfNormal(1)
    return ab.Priors(**{**base, **extra})


def model(type="icar", formula=None, frame=None, family="gaussian", data2=None, **extra):
    response = "y" if family == "gaussian" else "cnt"
    formula = formula or f'{response} ~ x + car(W, gr = site, type = "{type}")'
    return getattr(ab, family)(formula, FRAME if frame is None else frame, priors(type, family, **extra),
                               data2={"W": W} if data2 is None else data2)


@pytest.fixture(scope="module")
def models():
    return {t: model(t) for t in TYPES}


# -- the term ---------------------------------------------------------------------------------------------------------
def test_the_term_and_its_arguments():
    assert parse_car("car(W, gr = site)", "f") == Car("W", "site", "escar")                 # brms's default type
    assert parse_car('car(W, site, "bym2")', "f") == Car("W", "site", "bym2")                # R's positional arguments
    assert parse_car("car(type = 'icar', M = W, gr = site)", "f") == Car("W", "site", "icar")
    assert parse_car('car(gr = site, W, type = "esicar")', "f") == Car("W", "site", "esicar")   # named first, then the rest in order
    t = Car("W", "site", "bym2")
    assert t.term == 'car(W, gr = site, type = "bym2")' and t.columns == ("site",)
    assert t.classes == ("sdcar", "rhocar") and t.latent == ("zcar", "nszcar") and t.meaning == "BYM2 convolution (Riebler et al. 2016)"
    assert [Car("W", "s", k).classes for k in TYPES] == [("sdcar", "car"), ("sdcar",), ("sdcar",), ("sdcar", "rhocar")]
    assert [Car("W", "s", k).latent for k in TYPES] == [("rcar",), ("zcar",), ("zcar",), ("zcar", "nszcar")]
    s = parse_formula('y ~ x + car(W, gr = site, type = "icar") + (1 | g)', FRAME)
    assert s.car == Car("W", "site", "icar") and s.terms == ("x",) and s.groups == ("g",)
    assert s.canonical_formula == 'y ~ x + (1 | g) + car(W, gr = site, type = "icar")' and s.bambi_formula == "y ~ x + (1 | g)"
    rebuilt = ModelSpec(formula=s.formula, response="y", pop_terms=s.pop_terms, group_terms=s.group_terms, levels_by_group=dict(s.levels_by_group),
                        car={"matrix": "W", "gr": "site", "type": "icar"})                   # from a mapping (asdict output)
    assert rebuilt.car == s.car and hash(rebuilt) != hash(ModelSpec(formula=s.formula, response="y", pop_terms=s.pop_terms,
                                                                    group_terms=s.group_terms, levels_by_group=dict(s.levels_by_group)))
    with pytest.raises(FormulaError, match=r"columns not in data: \['site'\]"):
        parse_formula("y ~ x + car(W, gr = site)", FRAME.drop(columns="site"))


@pytest.mark.parametrize("piece, message", [
    ("car(W)x", r"a CAR term is written car\(M, gr = <column>"),
    ("car(W, gr = site, rho = 1)", r"car\(\) has no argument 'rho'; brms's car\(\) takes M, gr, type"),
    ("car(W, gr = site, gr = g)", r"the argument 'gr' is given twice"),
    ('car(W, site, "icar", 4)', r"too many arguments"),
    ("car(gr = site)", r"car\(\) needs M, the name of the adjacency matrix"),
    ('car("W", gr = site)', r"car\(\) needs M"),
    ("car(W)", r"car\(\) without gr \(each row its own location\) is not offered; brms deprecates it too .* divergence D141"),
    ("car(W, gr = NA)", r"without gr"),
    ("car(W, gr = g:h)", r"gr = g:h: gr must be one column name, the column of locations .* D141"),
    ('car(W, gr = site, type = "sar")', r"type = \"sar\": type must be one of 'escar', 'esicar', 'icar', 'bym2' \(brms's rule too\)"),
    ("car(W, gr = site, type = icar)", r"type must be one of"),
])
def test_the_term_refused(piece, message):
    with pytest.raises(FormulaError, match=message):
        parse_car(piece, "f")


@pytest.mark.parametrize("formula, family, message", [
    ("y ~ x + car(W, gr = site) + car(W, gr = site, type = 'icar')", "gaussian",
     r"one car\(\) term per formula \(brms: 'Can only model one term per autocorrelation class'\)"),
    ("y ~ x + car(W, gr = y)", "gaussian", r"the response 'y' cannot be the column of locations"),
    ("k | trials(n) ~ x + car(W, gr = n)", "binomial", r"the trials column 'n' cannot be the column of locations"),
    ("y ~ x + car(W, gr = site)", "student", r"not available for the student family; brms's car\(\) is verified here for the "
                                             r"gaussian and poisson families .* D141"),
    ("y | weights(w) ~ x + car(W, gr = site)", "gaussian", r"next to the addition term 'weights\(w\)' is not supported .* D141"),
    ("y ~ x + ar(gr = g) + car(W, gr = site)", "gaussian", r"next to the autocorrelation term 'ar\(gr = g, p = 1\)'"),
    ("y ~ mo(k) + car(W, gr = site)", "gaussian", r"next to the monotonic term 'mo\(k\)'"),
    ("y ~ gp(x) + car(W, gr = site)", "gaussian", r"next to the Gaussian process 'gp\(x\)'"),
    ("y ~ me(x, w) + car(W, gr = site)", "gaussian", r"next to the term 'me\(x, w\)'"),
    ("y ~ x + (1 | mm(g, site)) + car(W, gr = site)", "gaussian", r"next to the group term '\(1 \| mm\(g, site\)\)'"),
    ("y ~ x + sar(M) + car(W, gr = site)", "gaussian", r"next to the SAR term 'sar\(M, type = \"lag\"\)'"),
    ("y ~ x + fcor(V) + car(W, gr = site)", "gaussian", r"next to the fixed residual covariance 'fcor\(V\)'"),
])
def test_formula_contexts_refused(formula, family, message):
    with pytest.raises(FormulaError, match=message):
        parse_formula(formula, None, family)


def test_the_column_of_locations_may_be_a_grouping_factor():
    m = model("icar", formula='y ~ x + (1 | site) + car(W, gr = site, type = "icar")', sd=ab.HalfNormal(1))
    assert m.spec.groups == ("site",) and m.design.car.Nloc == 9


def test_other_models_refuse_the_term():
    lin = dict(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))
    frame = FRAME.assign(y2=FRAME.y * 2)
    with pytest.raises(FormulaError, match=r"the CAR term 'car\(W, gr = site, type = \"escar\"\)' with a distributional formula is not "
                                           r"supported .* D141"):
        ab.gaussian(["y ~ x + car(W, gr = site)", "sigma ~ x"], FRAME, ab.Priors(**lin), data2={"W": W})
    with pytest.raises(FormulaError, match=r"a CAR term in a distributional formula \(car\(W, gr = site, type = \"escar\"\)\) is not "
                                           r"supported: it belongs to the response formula here"):
        ab.gaussian(["y ~ x", "sigma ~ x + car(W, gr = site)"], FRAME, ab.Priors(**lin), data2={"W": W})
    with pytest.raises(FormulaError, match=r"a CAR term in a multivariate formula is not verified in this slice; .* D141"):
        ab.gaussian("mvbind(y, y2) ~ x + car(W, gr = site)", frame, ab.Priors(resp={"y": lin, "y2": lin}), rescor=False)
    with pytest.raises(FormulaError, match=r"the CAR term 'car\(W, gr = site, type = \"escar\"\)' in a multivariate model is not verified"):
        mi = frame.assign(xm=[np.nan, 0.2] + list(np.linspace(0, 1, 16)))           # C06's list of formulas, one per response
        ab.gaussian(["y ~ mi(xm) + w", "xm | mi() ~ w", "y2 ~ x + car(W, gr = site)"], mi, ab.Priors(resp={"y": lin, "xm": lin, "y2": lin}),
                    rescor=False)
    nl = dict(nlpars={"a": {"b": ab.Normal(1, 1)}, "b": {"b": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1))
    with pytest.raises(FormulaError, match=r"car\(\) at position \d+ is brms's spatial CAR term \(F02a\), not a function of the "
                                           r"non-linear expression"):
        ab.gaussian(["y ~ a * exp(b * x) + car(W, site)", "a ~ 1", "b ~ 1"], FRAME, ab.Priors(**nl), nl=True)
    with pytest.raises(FormulaError, match=r"the CAR term 'car\(W, gr = site, type = \"escar\"\)' in a non-linear parameter's formula"):
        ab.gaussian(["y ~ a * exp(b * x)", "a ~ 1 + car(W, gr = site)", "b ~ 1"], FRAME, ab.Priors(**nl), nl=True)
    with pytest.raises(PriorError_, match=r"sdcar is the prior of a CAR term car\(\) \(docs/spatial-car.md\); a non-linear model takes no "
                                          r"CAR term \(divergence D141\)"):
        ab.gaussian(["y ~ a * exp(b * x)", "a ~ 1", "b ~ 1"], FRAME, ab.Priors(**nl, sdcar=ab.HalfNormal(1)), nl=True)
    with pytest.raises(FormulaError, match=r"the CAR term 'car\(W, gr = site, type = \"escar\"\)' is not available under the sqrt link "
                                           r"of the poisson family"):
        ab.poisson("cnt ~ x + car(W, gr = site)", FRAME, priors("escar", "poisson"), link="sqrt", data2={"W": W})
    with pytest.raises(FormulaError, match=r"not available for the mixture family"):
        parse_formula("y ~ x + car(W, gr = site)", None, "mixture")


PriorError_ = ab.PriorError


# -- the adjacency matrix and the locations ---------------------------------------------------------------------------
def test_the_locations_and_graph_are_brms_stan_data():
    for frame, want in ((FRAME, BRMS), (FRAME[FRAME.site != "s9"], BRMS_NO_S9)):
        for type in ("escar", "bym2"):
            d, notes = car_layout(Car("W", "site", type), frame.reset_index(drop=True), {"W": W})
            np.testing.assert_array_equal(d.index + 1, want["Jloc"])
            np.testing.assert_array_equal(d.edges[:, 0] + 1, want["edges1"])
            np.testing.assert_array_equal(d.edges[:, 1] + 1, want["edges2"])
            np.testing.assert_array_equal(d.neighbours, want["Nneigh"])
            assert d.Nloc == len(want["Nneigh"]) and d.locations == tuple(sorted(set(frame.site)))
            if type == "escar":
                np.testing.assert_allclose(d.eigenvalues, want["eigenMcar"], rtol=0, atol=1e-15)
                assert d.scale is None
            else:
                assert d.eigenvalues is None and abs(d.scale - want["car_scale"]) < 1e-9        # brms's rounding (D140)
        assert bool(notes) == ("s9" not in set(frame.site))
    _, notes = car_layout(Car("W", "site", "icar"), FRAME[FRAME.site != "s9"].reset_index(drop=True), {"W": W})
    assert notes == ["'car(W, gr = site, type = \"icar\")': the locations ['s9'] of data2['W'] have no fitted row and are dropped from the "
                     "graph, with their edges (brms drops them too)"]
    big = grid(3, 4)
    _, notes = car_layout(Car("W", "site", "icar"), pd.DataFrame({"site": ["s1", "s2", "s5", "s6"]}), {"W": big})
    assert "['s3', 's4', 's7', 's8', 's9'] ..." in notes[0]


def test_the_matrix_is_read_by_its_labels():
    numbered = pd.DataFrame(W.to_numpy(), index=range(1, 10), columns=range(1, 10))     # integer labels, as R prints them
    d, _ = car_layout(Car("W", "site", "icar"), pd.DataFrame({"site": [3, 1, 2, 9, 5, 4, 7, 6, 8]}), {"W": numbered})
    assert d.locations == tuple(str(k) for k in range(1, 10)) and d.index[0] == 2
    positional = pd.DataFrame(W.to_numpy(), index=W.index)                               # columns by position
    shuffled = W.loc[W.index[::-1], W.index[::-1]]                                       # brms matches the rows by name
    weighted = W * 2.5                                                                   # brms reads any non-zero entry as 1
    for M in (positional, shuffled, weighted):
        d, notes = car_layout(Car("W", "site", "escar"), FRAME, {"W": M})
        np.testing.assert_array_equal(d.edges[:, 0] + 1, BRMS["edges1"])
        np.testing.assert_array_equal(d.W, W.to_numpy())
    assert notes == ["data2['W']: entries other than 0 and 1 are read as 1, an edge (brms: 'Converting all non-zero values in 'M' to 1.')"]
    with pytest.raises(FormulaError, match=r"data2 holds \['unused'\], which no term of the formula reads \(data2 is read by "
                                           r"sar\(M, \.\.\.\), fcor\(M\) and car\(W, \.\.\.\) here"):
        model("icar", data2={"W": weighted, "unused": W})                                # an entry no term reads is refused (F02b)
    m = model("icar", data2={"W": weighted})
    assert m.warnings[-1].startswith("data2['W']: entries other than 0 and 1")


def _bad(cells):
    M = W.copy()
    for (i, j), v in cells.items():
        M.iloc[i, j] = v
    return M


@pytest.mark.parametrize("data2, message", [
    (None, r"reads the adjacency matrix 'W' from data2: pass data2=\{'W': <a pandas DataFrame labelled by location on both axes>\}"),
    ({}, r"reads the adjacency matrix 'W' from data2"),
    ({"M": W}, r"data2 holds \['M'\], which no term of the formula reads"),                # F02b's rule: data2_names
    ({"W": W.to_numpy()}, r"data2\['W'\] must be a square pandas DataFrame whose index labels the locations .* 'Row names are required"),
    ({"W": W.reset_index(drop=True)}, r"must be a square pandas DataFrame"),
    ({"W": W.iloc[:, :8]}, r"must be a square pandas DataFrame"),
    ({"W": W.set_axis(["s1"] * 9, axis=0)}, r"the locations must label the rows once each, and the columns in the same order"),
    ({"W": W.set_axis([f"t{k}" for k in range(9)], axis=1)}, r"the columns in the same order"),
    ({"W": W.astype(object).where(W == 0, "1")}, r"data2\['W'\] must hold finite real numbers"),
    ({"W": W.replace(1.0, np.inf)}, r"must hold finite real numbers"),
    ({"W": W.astype(complex)}, r"must hold finite real numbers"),
    ({"W": _bad({(0, 1): 0.0})}, r"data2\['W'\] must be symmetric \(brms: ''M' for CAR terms must be symmetric.'\)"),
    ({"W": _bad({(4, 4): 1.0})}, r"has a non-zero diagonal: a location is not its own neighbour here \(brms counts the diagonal entry as "
                                   r"an edge; .* D141\)"),
])
def test_the_matrix_refused(data2, message):
    with pytest.raises(FormulaError, match=message):
        ab.gaussian('y ~ x + car(W, gr = site)', FRAME, priors("escar"), data2=data2)


def test_the_locations_refused():
    with pytest.raises(FormulaError, match=r"the locations \['s9'\] of 'site' are not labels of data2\['W'\] \(brms: 'Row names of 'M' for "
                                           r"CAR terms do not match the names of the grouping levels.'\)"):
        car_layout(Car("W", "site", "icar"), FRAME, {"W": W.iloc[:8, :8]})
    with pytest.raises(FormulaError, match=r"the fitted rows have 1 location; a CAR term needs at least two"):
        car_layout(Car("W", "site", "icar"), pd.DataFrame({"site": ["s1", "s1"]}), {"W": W})
    lonely = _bad({(2, 5): 0.0, (5, 2): 0.0, (5, 8): 0.0, (8, 5): 0.0, (4, 5): 0.0, (5, 4): 0.0})   # s6 without a neighbour
    with pytest.raises(FormulaError, match=r"the locations \['s6'\] have no neighbour among the fitted locations \(brms refuses them for "
                                           r"escar and esicar and accepts them for icar and bym2"):
        car_layout(Car("W", "site", "bym2"), FRAME, {"W": lonely})
    split = _bad({(1, 2): 0.0, (2, 1): 0.0, (4, 5): 0.0, (5, 4): 0.0, (7, 8): 0.0, (8, 7): 0.0})   # columns 1-2 and column 3
    with pytest.raises(FormulaError, match=r"has 2 connected components; one connected graph is offered .* divergences D140, D141"):
        car_layout(Car("W", "site", "escar"), FRAME, {"W": split})
    with pytest.raises(FormulaError, match=r"location column 'site' of 'car\(W, gr = site, type = \"icar\"\)' has missing values"):
        car_layout(Car("W", "site", "icar"), FRAME.assign(site=[None] + SITES[1:]), {"W": W})
    with pytest.raises(FormulaError, match=r"missing values in columns \['site'\]"):
        model("icar", frame=FRAME.assign(site=[None] + SITES[1:]))


def test_components():
    assert components(W.to_numpy()) == 1 and components(np.zeros((3, 3))) == 3
    assert components(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])) == 2


def _exact_scale(W):
    """brms's (INLA's) BYM2 scaling by its definition in exact rational arithmetic: the inverse of Q + eps I (eps = max(diag Q) 2^-26,
    sqrt of the double epsilon) conditioned on sum = 0, by Gauss-Jordan elimination over fractions; then the geometric mean of the
    diagonal in double precision."""
    n = len(W)
    eps = Fraction(int(W.sum(1).max()), 2 ** 26)
    A = [[Fraction(int(W[i].sum()) if i == j else -int(W[i, j])) + (eps if i == j else 0) for j in range(n)]
         + [Fraction(int(i == k)) for k in range(n)] for i in range(n)]
    for col in range(n):
        A[col] = [v / A[col][col] for v in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [a - f * b for a, b in zip(A[r], A[col])]
    S = [row[n:] for row in A]
    S1 = [sum(row) for row in S]
    total = sum(S1)
    return math.exp(math.fsum(math.log(float(S[i][i] - S1[i] * S1[i] / total)) for i in range(n)) / n)


def test_the_bym2_scaling_factor():
    """D140: brms's car_scale is INLA's perturbed scaling with its rounding; the definition's exact value is what bym2_scale
    computes, to 1e-15, and brms's values are within its rounding of it (printed by brms 2.23.0: the 3 by 3 grid, the grid without
    s9, the frozen 5 by 5 grid, a path of four); the exact geometric mean of Q's generalized inverse lies above them."""
    cases = [(W.to_numpy(), BRMS["car_scale"], 1e-9), (W.iloc[:8, :8].to_numpy(), BRMS_NO_S9["car_scale"], 1e-9),
             (grid(5, 5).to_numpy(), 0.51638590096426951, 2e-10),
             (np.diag(np.ones(3), 1) + np.diag(np.ones(3), -1), 0.57282194195281777, 1e-15)]
    for M, brms, tol in cases:
        mine = bym2_scale(M)
        assert abs(mine - _exact_scale(M)) < 1e-15 and abs(mine - brms) < tol
        Q = np.diag(M.sum(1)) - M
        riebler = math.exp(np.mean(np.log(np.diag(np.linalg.pinv(Q)))))
        assert 1e-8 < (riebler - mine) / riebler < 1e-7
    assert abs(bym2_scale(grid(5, 5).to_numpy()) - 0.5163859011368434) < 1e-15


def test_rcar_from_the_parameters():
    d = {t: car_layout(Car("W", "site", t), FRAME, {"W": W})[0] for t in TYPES}
    z, ns = np.linspace(-1.0, 1.0, 9), np.cos(np.arange(9.0))
    np.testing.assert_array_equal(car_values(d["escar"], {"rcar": z}), z)
    np.testing.assert_allclose(car_values(d["esicar"], {"zcar": z[:8]}), np.r_[z[:8], -z[:8].sum()], rtol=1e-15)
    np.testing.assert_allclose(car_values(d["icar"], {"zcar": z, "sdcar": 0.7}), 0.7 * z, rtol=1e-15)
    want = 0.7 * (math.sqrt(1 - 0.3) * ns + math.sqrt(0.3 / d["bym2"].scale) * z)
    np.testing.assert_allclose(car_values(d["bym2"], {"zcar": z, "nszcar": ns, "sdcar": 0.7, "rhocar": 0.3}), want, rtol=1e-14)
    two = car_values(d["icar"], {"zcar": np.stack([z, 2 * z]), "sdcar": np.array([1.0, 0.5])})   # one row per draw
    np.testing.assert_allclose(two, np.stack([z, z]), rtol=1e-15)


# -- the density ------------------------------------------------------------------------------------------------------
RNG = np.random.default_rng(20261024)
POINT = {"intercept": 0.4, "b": [0.6], "sigma": 0.8, "sdcar": 0.7, "car": 0.35, "rhocar": 0.6,
         "rcar": RNG.normal(size=9), "zcar": RNG.normal(size=9) * 0.3, "nszcar": RNG.normal(size=9)}


def car_entries(type):
    out = {name: POINT[name] for name in (*Car("W", "s", type).classes, *Car("W", "s", type).latent)}
    if type == "esicar":
        out["zcar"] = POINT["zcar"][:8]
    return out


def reference(type, values, *, family="gaussian", frame=FRAME, jacobian=True, extra_eta=0.0, extra_lp=0.0):
    """The log density from the published densities, written here without autobayes.car: the likelihood, the priors, and the
    CAR density (escar: the multivariate normal of precision tau (D - car W) with scipy, plus brms's dropped constants
    N log(2 pi) / 2 - log det(D) / 2; esicar: the intrinsic form; icar and bym2: the pairwise differences over every ordered pair
    of neighbours, halved, and the soft sum-to-zero normal), with the Jacobians of the coordinates."""
    M = W.to_numpy()
    D, N = M.sum(1), len(M)
    labels = sorted(set(frame.site))
    J = np.array([labels.index(s) for s in frame.site])
    tau = 1.0 / values["sdcar"] ** 2
    if type == "escar":
        r = np.asarray(values["rcar"])
        prec = tau * (np.diag(D) - values["car"] * M)
        lp_car = (stats.multivariate_normal(np.zeros(N), np.linalg.inv(prec)).logpdf(r) + N / 2 * math.log(2 * math.pi)
                  - 0.5 * np.sum(np.log(D)))
    elif type == "esicar":
        r = np.r_[values["zcar"], -np.sum(values["zcar"])]
        lp_car = 0.5 * ((N - 1) * math.log(tau) - tau * r @ (np.diag(D) - M) @ r)
    else:
        z = np.asarray(values["zcar"])
        lp_car = -0.25 * np.sum(M * (z[:, None] - z[None, :]) ** 2) + stats.norm.logpdf(z.sum(), 0, 0.001 * N)
        if type == "icar":
            r = values["sdcar"] * z
        else:
            ns, rho = np.asarray(values["nszcar"]), values["rhocar"]
            r = values["sdcar"] * (math.sqrt(1 - rho) * ns + math.sqrt(rho / _exact_scale(M)) * z)
            lp_car += stats.norm.logpdf(ns).sum() + stats.beta.logpdf(rho, 1, 1)
    x = frame.x.to_numpy()
    eta = values["intercept"] + (x - x.mean()) * values["b"][0] + r[J] + extra_eta
    lp = lp_car + stats.norm.logpdf(values["intercept"], 0, 2) + stats.norm.logpdf(values["b"][0], 0, 1)
    lp += stats.halfnorm.logpdf(values["sdcar"]) + extra_lp
    if family == "gaussian":
        lp += stats.norm.logpdf(frame.y, eta, values["sigma"]).sum() + stats.halfnorm.logpdf(values["sigma"])
    else:
        lp += stats.poisson.logpmf(frame.cnt, np.exp(eta)).sum()
    if jacobian:
        lp += math.log(values["sdcar"]) + (math.log(values["sigma"]) if family == "gaussian" else 0.0)
        lp += sum(math.log(values[c] * (1 - values[c])) for c in ("car", "rhocar") if c in Car("W", "s", type).classes)
    return lp


@pytest.mark.parametrize("type", TYPES)
def test_the_log_density_is_the_published_one(models, type):
    m = models[type]
    vp = m.value_point(intercept=POINT["intercept"], b=POINT["b"], sigma=POINT["sigma"], car=car_entries(type))
    values = {**POINT, **car_entries(type)}
    for jac in (True, False):
        got = float(m.pymc_model.compile_logp(jacobian=jac)(vp))
        assert got == pytest.approx(reference(type, values, jacobian=jac), rel=1e-12, abs=1e-10)
    # the gradient: central differences of the reference in the value space (sdcar and sigma on the log scale, car and rhocar on
    # the logit scale), coordinate by coordinate
    dlogp = m.pymc_model.compile_dlogp()(vp)
    layout, i = {}, 0
    for v in m.pymc_model.value_vars:
        layout[v.name] = slice(i, i + np.asarray(vp[v.name]).size)
        i += np.asarray(vp[v.name]).size

    def ref_at(point):
        nat = {"intercept": float(point["Intercept"]), "b": [float(point["x"])], "sigma": math.exp(float(point["sigma_log__"])),
               "sdcar": math.exp(float(point["sdcar_log__"]))}
        for c in ("car", "rhocar"):
            key = next((k for k in point if k.startswith(c + "_")), None)
            if key is not None:
                nat[c] = 1.0 / (1.0 + math.exp(-float(point[key])))
        nat.update({v: np.asarray(point[v]) for v in Car("W", "s", type).latent})
        return reference(type, nat)
    h = 1e-6
    for name, sl in layout.items():
        for k in range(sl.stop - sl.start):
            up, down = copy.deepcopy(vp), copy.deepcopy(vp)
            up[name] = np.array(up[name], dtype=float)
            down[name] = np.array(down[name], dtype=float)
            up[name].flat[k] += h
            down[name].flat[k] -= h
            fd = (ref_at(up) - ref_at(down)) / (2 * h)
            assert dlogp[sl.start + k] == pytest.approx(fd, rel=1e-5, abs=1e-5), (name, k)


def test_the_poisson_density_and_group_terms_and_offset():
    m = model("bym2", family="poisson")
    vp = m.value_point(intercept=POINT["intercept"], b=POINT["b"], car=car_entries("bym2"))
    got = float(m.pymc_model.compile_logp()(vp))
    assert got == pytest.approx(reference("bym2", {**POINT, **car_entries("bym2")}, family="poisson"), rel=1e-12)
    m = model("icar", formula='y ~ x + offset(w) + (1 | g) + car(W, gr = site, type = "icar")', sd=ab.HalfNormal(1))
    u = np.array([0.3, -0.2, 0.5])
    vp = m.value_point(intercept=POINT["intercept"], b=POINT["b"], sigma=POINT["sigma"], sd=0.9, z=u / 0.9, car=car_entries("icar"))
    extra_eta = FRAME.w.to_numpy() + u[np.array([["a", "b", "c"].index(g) for g in FRAME.g])]
    extra_lp = stats.halfnorm.logpdf(0.9) + stats.norm.logpdf(u / 0.9).sum() + math.log(0.9)
    want = reference("icar", {**POINT, **car_entries("icar")}, extra_eta=extra_eta, extra_lp=extra_lp)
    assert float(m.pymc_model.compile_logp()(vp)) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("type", ("escar", "bym2"))
def test_the_jax_density(models, type):
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    m = models[type]
    vp = m.value_point(intercept=POINT["intercept"], b=POINT["b"], sigma=POINT["sigma"], car=car_entries(type))
    jf = get_jaxified_logp(m.pymc_model)
    assert float(jf([vp[v.name] for v in m.pymc_model.value_vars])) == pytest.approx(float(m.pymc_model.compile_logp()(vp)), rel=1e-13)


def test_the_graph_is_checked_when_built(monkeypatch):
    monkeypatch.setattr(carmod, "car_values", lambda design, values: np.zeros(design.Nloc) + 1.0)
    with pytest.raises(RuntimeError, match=r"the graph's 'mu' does not add 'car\(W, gr = site, type = \"icar\"\)' as car.car_values computes it"):
        model("icar")


# -- priors -----------------------------------------------------------------------------------------------------------
def test_priors_of_the_term():
    spec = parse_formula('y ~ x + car(W, gr = site, type = "bym2")', FRAME)
    assert ab.Priors(sdcar=ab.HalfNormal(1), rhocar=ab.Beta(2, 2)).car_priors(spec) == {"sdcar": ab.HalfNormal(1), "rhocar": ab.Beta(2, 2)}
    plain = parse_formula("y ~ x", FRAME)
    assert ab.Priors().car_priors(plain) == {}
    for given, message in ((dict(sdcar=ab.HalfNormal(1)), r"do not pass sdcar: the formula has no CAR term car\(\)"),):
        with pytest.raises(ab.PriorError, match=message):
            ab.Priors(**given).car_priors(plain)
    for given, message in (
            (dict(sdcar=ab.HalfNormal(1)), r"the CAR term 'car\(W, gr = site, type = \"bym2\"\)' needs a prior on rhocar: pass rhocar=... "
                                           r"\(brms class rhocar; brms's default is beta\(1, 1\)"),
            (dict(rhocar=ab.Beta(1, 1)), r"needs a prior on sdcar: pass sdcar=... \(brms class sdcar; brms's default is student_t\(3, 0, "
                                         r"scale\), as for sd"),
            (dict(sdcar=ab.HalfNormal(1), rhocar=ab.Beta(1, 1), car=ab.Flat()), r"do not pass car: the CAR term 'car\(W, gr = site, "
                                                                               r"type = \"bym2\"\)' has no car"),
            (dict(sdcar=ab.Normal(0, 1), rhocar=ab.Beta(1, 1)), r"prior for 'sdcar' must be a positive-support prior"),
            (dict(sdcar=ab.HalfNormal(1), rhocar=ab.Normal(0, 1)), r"prior for 'rhocar' must be a prior on \(0, 1\) or Flat\(\), the uniform "
                                                                   r"there \(\['Beta', 'Flat'\]\)"),
            (dict(sdcar=ab.HalfNormal(1), rhocar="beta"), r"prior for 'rhocar' must be a prior on \(0, 1\)")):
        with pytest.raises(ab.PriorError, match=message):
            ab.Priors(**given).car_priors(spec)
    escar = parse_formula('y ~ x + car(W, gr = site)', FRAME)
    with pytest.raises(ab.PriorError, match=r"needs a prior on car: pass car=... \(brms class car; brms's default is flat on \(0, 1\), "
                                            r"ab.Flat\(\)"):
        ab.Priors(sdcar=ab.HalfNormal(1)).car_priors(escar)
    with pytest.raises(ab.PriorError, match=r"sdcar, car and rhocar are the priors of a CAR term car\(\) .* a multivariate model takes none"):
        lin = dict(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))
        ab.gaussian("mvbind(y, y2) ~ x", FRAME.assign(y2=FRAME.x), ab.Priors(resp={"y": lin, "y2": lin}, sdcar=ab.HalfNormal(1)), rescor=False)
    with pytest.raises(ab.PriorError, match=r"a mixture's priors are per component.*\['sdcar'\] at the top level do not apply"):
        comp = ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1))
        ab.mixture("y ~ x", FRAME, ab.Priors(components=[comp, comp], theta=ab.Dirichlet(1), sdcar=ab.HalfNormal(1)),
                   families=["gaussian", "gaussian"])


def test_the_priors_build_the_parameters():
    """car and rhocar under Beta are sampled on PyMC's log-odds coordinate, under Flat() as the uniform on its interval coordinate
    (both Stan's logit); every prior's density is the scipy one, the uniform's 0 (brms's flat prior adds nothing to lprior)."""
    for type, extra, name, coordinate, density in (
            ("escar", {"car": ab.Beta(2, 3)}, "car", "car_logodds__", stats.beta(2, 3).logpdf(0.35)),
            ("escar", {}, "car", "car_interval__", 0.0),
            ("bym2", {"rhocar": ab.Flat()}, "rhocar", "rhocar_interval__", 0.0),
            ("icar", {"sdcar": ab.HalfStudentT(3, 2.5)}, "sdcar", "sdcar_log__", stats.t(3, 0, 2.5).logpdf(0.7) + math.log(2))):
        m = model(type, **extra)
        assert coordinate in {v.name for v in m.pymc_model.value_vars}
        rv = m.pymc_model[name]
        logp = m.pymc_model.compile_logp(vars=[rv], jacobian=False)
        vp = m.value_point(intercept=POINT["intercept"], b=POINT["b"], sigma=POINT["sigma"], car=car_entries(type))
        assert float(logp(vp)) == pytest.approx(density, rel=1e-8, abs=1e-12)


def test_brms_default_priors():
    """brms 2.23.0's default_prior() for FRAME (printed above the tests): Intercept student_t(3, 0.3, 2.5), flat b, car flat with lb 0
    and ub 1 (escar), rhocar beta(1, 1) with lb 0 and ub 1 (bym2), sdcar and sigma student_t(3, 0, 2.5) with lb 0."""
    def rows(type, family="gaussian", response="y"):
        f = f'{response} ~ x + car(W, gr = site, type = "{type}")'
        d = ab.default_priors(f, FRAME, family=family, data2={"W": W})
        m = getattr(ab, family)(f, FRAME, d, data2={"W": W})
        return d, [(r["prior"], r["class"], r["coef"], r["lb"], r["ub"], r["source"]) for r in m.prior_table()]
    d, table = rows("escar")
    assert d.sdcar == ab.HalfStudentT(3, 2.5) and d.car == ab.Flat() and d.rhocar is None and d.sdcar.source == "default"
    assert table == [("student_t(3, 0.3, 2.5)", "Intercept", "", "", "", "default"), ("", "b", "", "", "", "default"),
                     ("", "b", "x", "", "", "default"), ("", "car", "", "0", "1", "default"),
                     ("student_t(3, 0, 2.5)", "sdcar", "", "0", "", "default"), ("student_t(3, 0, 2.5)", "sigma", "", "0", "", "default")]
    d, table = rows("bym2")
    assert d.rhocar == ab.Beta(1, 1) and d.car is None and ("beta(1, 1)", "rhocar", "", "0", "1", "default") in table
    d, table = rows("esicar")
    assert d.car is None and d.rhocar is None and [r[1] for r in table] == ["Intercept", "b", "b", "sdcar", "sigma"]
    d, _ = rows("icar", "poisson", "cnt")
    assert d.sdcar == ab.HalfStudentT(3, 2.5)
    with pytest.raises(FormulaError, match=r"reads the adjacency matrix 'W' from data2"):
        ab.default_priors("y ~ x + car(W, gr = site)", FRAME)
    m = model("bym2", sdcar=ab.HalfStudentT(3, 2.5), rhocar=ab.Prior("Beta", {"alpha": 1.0, "beta": 1.0}, source="default"))
    assert m.default_prior_labels == ["rhocar"]


# -- value_point ------------------------------------------------------------------------------------------------------
def test_value_point(models):
    m = models["bym2"]
    vp = m.value_point(intercept=0.1, b=[0.2], sigma=0.5, car=car_entries("bym2"))
    assert vp["sdcar_log__"] == pytest.approx(math.log(0.7)) and vp["rhocar_logodds__"] == pytest.approx(math.log(0.6 / 0.4))
    np.testing.assert_array_equal(vp["zcar"], POINT["zcar"])
    good = car_entries("bym2")
    for bad, message in ((None, r"car must map exactly \['sdcar', 'rhocar', 'zcar', 'nszcar'\] to their values .* got NoneType"),
                         ({"sdcar": 0.7}, r"car must map exactly .* got \['sdcar'\]"),
                         ({**good, "sdcar": -1.0}, r"car\['sdcar'\] must be a positive finite number"),
                         ({**good, "sdcar": np.inf}, r"car\['sdcar'\] must be a positive finite number"),
                         ({**good, "rhocar": 1.0}, r"car\['rhocar'\] must be a number in \(0, 1\)"),
                         ({**good, "zcar": np.ones(8)}, r"car\['zcar'\] must be 9 finite numbers"),
                         ({**good, "nszcar": np.full(9, np.nan)}, r"car\['nszcar'\] must be 9 finite numbers")):
        with pytest.raises(ValueError, match=message):
            m.value_point(intercept=0.1, b=[0.2], sigma=0.5, car=bad)
    with pytest.raises(ValueError, match=r"car\['car'\] must be a number in \(0, 1\)"):
        models["escar"].value_point(intercept=0.1, b=[0.2], sigma=0.5, car={**car_entries("escar"), "car": 0.0})
    with pytest.raises(ValueError, match=r"car\['zcar'\] must be 8 finite numbers"):
        models["esicar"].value_point(intercept=0.1, b=[0.2], sigma=0.5, car={"sdcar": 0.7, "zcar": np.ones(9)})
    plain = ab.gaussian("y ~ x", FRAME, ab.Priors(intercept=ab.Normal(0, 1), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)))
    with pytest.raises(ValueError, match=r"car holds a CAR term's values; this model has no car\(\) term"):
        plain.value_point(intercept=0.1, b=[0.2], sigma=0.5, car=good)


# -- predictions and post-processing ----------------------------------------------------------------------------------
def draws(m, rcar):
    """Two chains of three draws: the raw-scale intercept, the slope, sigma (Gaussian), the term's parameters and its rcar (one
    vector per draw), as a fit's posterior holds them."""
    loc = list(m.design.car.locations)
    scale = np.array([[1.0, 0.9, 1.1], [0.8, 1.2, 1.0]])
    post = {"Intercept": np.full((2, 3), 0.3), "x": np.full((2, 3), 0.7), "rcar": scale[..., None] * rcar, "sdcar": scale}
    coords, dims = {"rcar_dim": loc}, {"rcar": ["rcar_dim"]}
    if "sigma" in m.names:
        post["sigma"] = np.full((2, 3), 0.9)
    if m.spec.groups:
        post.update({"1|g": np.broadcast_to(np.array([0.25, -0.15, 0.1]), (2, 3, 3)), "1|g_sigma": np.ones((2, 3))})
        coords["g__factor_dim"], dims["1|g"] = ["a", "b", "c"], ["g__factor_dim"]
    return az.from_dict(posterior=post, coords=coords, dims=dims), scale[..., None] * rcar


def test_predictions_add_rcar_at_each_rows_location():
    m = model("icar", formula='y ~ x + (1 | g) + car(W, gr = site, type = "icar")', sd=ab.HalfNormal(1))
    idata, rcar = draws(m, np.linspace(-0.4, 0.4, 9))
    J = np.array([int(s[1:]) - 1 for s in FRAME.site])
    fixed = 0.3 + 0.7 * FRAME.x.to_numpy() + rcar[..., J]
    groups = np.array([0.25, -0.15, 0.1])[np.array([["a", "b", "c"].index(g) for g in FRAME.g])]
    np.testing.assert_allclose(m.predict(idata, kind="linear")["eta"], fixed + groups, rtol=1e-14)
    np.testing.assert_allclose(m.predict(idata, kind="mean")["mu"], fixed + groups, rtol=1e-14)
    # without group effects the CAR term stays (brms's re_formula = NA keeps it, probed with a fixed-parameter fit)
    np.testing.assert_allclose(m.predict(idata, kind="linear", include_group_effects=False)["eta"], fixed, rtol=1e-14)
    new = pd.DataFrame({"x": [0.5, -0.2], "site": ["s4", "s9"], "g": ["b", "a"]})
    want = 0.3 + 0.7 * new.x.to_numpy() + rcar[..., [3, 8]] + np.array([-0.15, 0.25])
    np.testing.assert_allclose(m.predict(idata, newdata=new, kind="linear")["eta"], want, rtol=1e-14)
    assert m.predict(idata, newdata=new, kind="response", seed=1)["y"].shape == (2, 3, 2)
    with pytest.raises(FormulaError, match=r"newdata has locations of 'site' the model was not fitted on: \['s10'\]; .* \(brms: 'Cannot "
                                           r"handle new locations in CAR models.'\)"):
        m.predict(idata, newdata=new.assign(site=["s4", "s10"]), kind="linear")
    with pytest.raises(FormulaError, match=r"newdata lacks columns \['site'\]"):
        m.predict(idata, newdata=new.drop(columns="site"), kind="linear", include_group_effects=False)
    p = model("bym2", family="poisson")
    idata, rcar = draws(p, np.linspace(-0.4, 0.4, 9))
    eta = 0.3 + 0.7 * FRAME.x.to_numpy() + rcar[..., J]
    np.testing.assert_allclose(p.predict(idata, kind="mean")["mu"], np.exp(eta), rtol=1e-14)


def test_log_lik_bayes_R2_and_loo_read_the_term():
    """brms's log_lik() of a CAR model is each row's density at the predictor with rcar (probed with a fixed-parameter fit: dnorm(y,
    posterior_epred, sigma)); bayes_R2 and loo follow from it and from the expected values."""
    m = model("icar")
    idata, rcar = draws(m, np.linspace(-0.4, 0.4, 9))
    J = np.array([int(s[1:]) - 1 for s in FRAME.site])
    mu = 0.3 + 0.7 * FRAME.x.to_numpy() + rcar[..., J]
    np.testing.assert_allclose(ab.log_lik(m, idata), stats.norm.logpdf(FRAME.y.to_numpy(), mu, 0.9), rtol=1e-13)
    resid = FRAME.y.to_numpy() - mu
    want = mu.var(-1, ddof=1) / (mu.var(-1, ddof=1) + resid.var(-1, ddof=1))
    np.testing.assert_allclose(ab.bayes_R2(m, idata), want, rtol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isfinite(ab.loo(m, idata).elpd_loo)


# -- describe(), names, identity, records -----------------------------------------------------------------------------
def test_describe_and_names(models):
    text = models["escar"].describe()
    for line in ('AutoBayes M0 model with spatial CAR term car(W, gr = site, type = "escar") (population-level only)',
                 '  car       : car(W, gr = site, type = "escar"): 9 locations (the labels of site, sorted) and 12 edges of data2[\'W\'] on '
                 "them, 2 to 4 neighbours each (brms's Nloc, Jloc, edges1, edges2, Nneigh)",
                 "  mu_i = a_c + (x_x - mean(x_x)) * b_x + rcar[site_i]",
                 "  The CAR log density is the potential rcar_lpdf (in brms's model block, not its lprior); new locations are refused.",
                 "  sdcar ~ HalfNormal(sigma=1)   (brms class sdcar: the SD of the CAR effects)",
                 "  car ~ Flat()   (brms class car: the spatial dependence; Flat() is the uniform on (0, 1))",
                 "  car_interval__           logit(car) (brms's car, in (0, 1))",
                 "  rcar                     rcar[1..9], brms's rcar (vector; dims rcar_dim; flat, its density the potential rcar_lpdf)"):
        assert line in text, line
    assert "(Nloc log tau + sum_k log(1 - car lambda_k)" in text
    assert "((Nloc - 1) log tau - tau rcar' (D - W) rcar) / 2" in models["esicar"].describe()
    assert "zcar[1..8], brms's zcar (vector; dims zcar_dim" in models["esicar"].describe()
    assert "rcar = sdcar * zcar, and zcar's -sum over the edges" in models["icar"].describe()
    bym2 = models["bym2"].describe()
    assert "BYM2 scaling factor car_scale = 0.409708963" in bym2 and "sqrt(rhocar / 0.409708963) zcar" in bym2
    assert "  rhocar ~ Beta(alpha=1, beta=1)   (brms class rhocar: the spatial share)" in bym2
    assert "nszcar[1..9], brms's nszcar (vector; dims rcar_dim; Normal(0, 1))" in bym2
    default = model("escar", sdcar=ab.default_priors("y ~ x + car(W, gr = site)", FRAME, data2={"W": W}).sdcar)
    assert "  sdcar ~ HalfStudentT(nu=3, sigma=2.5)   (brms default student_t(3, 0, 2.5), lb 0)   (brms class sdcar" in default.describe()
    assert models["bym2"].car_names == {"sdcar": "sdcar", "rhocar": "rhocar", "zcar": "zcar", "nszcar": "nszcar", "rcar": "rcar",
                                        "coordinate": "rcar_dim", "lpdf": "zcar_lpdf"}
    assert models["esicar"].car_names["zcar_coordinate"] == "zcar_dim" and models["escar"].names["car"] == "car"
    assert carmod.roles(None) == {} and carmod.describe(SimpleNamespace(spec=parse_formula("y ~ x", FRAME)), "roles") == {}
    assert carmod.describe(SimpleNamespace(spec=parse_formula("y ~ x", FRAME)), "data") == []
    with pytest.raises(FormulaError, match=r"generated variable/coordinate name collision: \['rcar'\]"):
        model("icar", formula='y ~ x + rcar + car(W, gr = site, type = "icar")', frame=FRAME.assign(rcar=FRAME.x ** 2))


def test_the_target_identity_holds_the_term(models):
    base = models["icar"].target_identity
    assert model("icar").target_identity == base
    assert models["escar"].target_identity != base and models["bym2"].target_identity != base
    assert model("icar", sdcar=ab.HalfNormal(2)).target_identity != base
    assert model("icar", data2={"W": W.loc[W.index[::-1], W.index[::-1]]}).target_identity == base   # the same graph
    assert model("icar", frame=FRAME.assign(site=FRAME.site.map({"s1": "s2", "s2": "s1"}).fillna(FRAME.site))).target_identity != base


def test_fit_records_and_summary_rows(monkeypatch):
    m = model("escar")
    idata, _ = draws(m, np.linspace(-0.4, 0.4, 9))
    post = idata.posterior.assign(car=idata.posterior["sdcar"] * 0.3)
    idata = az.InferenceData(posterior=post)
    import autobayes.session as session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = session._summary_rows(m, idata)
    assert {"sdcar", "car"} <= set(rows)
    monkeypatch.setattr(session, "_summary_rows", lambda *args: {})
    monkeypatch.setattr(session, "environment", lambda: {})
    fit = SimpleNamespace(target_identity=m.target_identity, idata=idata, backend="synthetic", device_requested="cpu",
                          device_actual="cpu", platform="cpu", dtype="float64", chain_method="none", chains=2, draws=3,
                          tune=0, target_accept=0.9, seed=9, wall_seconds=0, notes=["fixed synthetic draws; not a fit"])
    rec = ab.fit_record(m, fit, panel={"values": {}, "status": {}, "overall": "not evaluated"})
    assert rec["model"]["priors"]["sdcar"] == "HalfNormal(sigma=1)" and rec["model"]["priors"]["car"] == "Flat()"
    assert rec["model"]["car"] == {"term": 'car(W, gr = site, type = "escar")', "type": "escar", "gr": "site", "matrix": "W",
                                   "locations": 9, "edges": 12, "names": m.car_names}
    assert ab.compare_records(rec, json.loads(json.dumps(rec)))["agree"]


def test_functions_that_refuse_the_term(models):
    m = models["icar"]
    with pytest.raises(ValueError, match=r"prior_predictive\(\) of a model with the CAR term 'car\(W, gr = site, type = \"icar\"\)' is "
                                         r"not offered: .* divergence D141"):
        m.prior_predictive(draws=10, seed=1)
    idata, _ = draws(m, np.linspace(-0.4, 0.4, 9))
    with pytest.raises(ValueError, match=r"conditional_effects\(\) of a model with the CAR term 'car\(W, gr = site, type = \"icar\"\)' is "
                                         r"not offered"):
        ab.conditional_effects(m, idata, "x")
    with pytest.raises(ValueError, match=r"prior_draws\(\) of a model with the CAR term 'car\(W, gr = site, type = \"icar\"\)' is not"):
        ab.prior_draws(m, draws=5, seed=1)
    with pytest.raises(ab.PriorError, match=r"a CAR term car\(W, gr = site, type = \"icar\"\), which no frozen case verifies with it"):
        model("icar", b=ab.Horseshoe(df=1))


# -- the oracle's extension -------------------------------------------------------------------------------------------
def _bundle(case="gauss_car_escar", model=None):
    from tests.oracle import gates

    return gates.Bundle(case, model=model)


def _corrupted(bundle, mutate):
    from tests.oracle import fixture as fx

    raw = copy.deepcopy(bundle.fixture.raw)
    mutate(raw)
    return bundle.with_fixture(fx.Fixture(raw))


def test_the_gates_see_corrupted_car_fixtures():
    from tests.oracle import adapter, gates
    from tests.oracle.adapter import OracleMismatch

    b = _bundle()

    def standata(key, k, delta):
        def mutate(raw):
            raw["standata"][key]["values"][k] += delta
        return mutate
    for key, k, delta, message in (("Jloc", 0, 1, r"brms's Nloc 25 and Jloc differ from the model's 25 locations"),
                                   ("edges1", 3, 1, r"brms edges1 is .* \(largest difference 1.0\)"),
                                   ("Nneigh", 0, 1, r"brms Nneigh is"),
                                   ("eigenMcar", 2, 1e-11, r"brms eigenMcar is")):
        with pytest.raises(OracleMismatch, match=message):
            gates.check_design(_corrupted(b, standata(key, k, delta)))

    def no_eigen(raw):
        del raw["standata"]["eigenMcar"]
    with pytest.raises(OracleMismatch, match=r"brms has the CAR data \['Jloc', 'Nedges', 'Nloc', 'Nneigh', 'edges1', 'edges2'\]; the model's "
                                             r"car\(W, gr = site, type = \"escar\"\) has"):
        gates.check_design(_corrupted(b, no_eigen))

    def rcar(raw):
        raw["points"][2]["pars"]["rcar"]["values"][4] += 1e-3
    with pytest.raises(OracleMismatch, match=r"point 2: log density"):
        gates.check_logp(_corrupted(b, rcar))

    def car_u(raw):
        i = next(k for k, e in enumerate(raw["upars"]) if e["par"] == "car")
        raw["points"][1]["u"]["values"][i] += 1e-6
    with pytest.raises(OracleMismatch, match=r"point 1: logit\(car\) is .*; brms's unconstrained value is"):
        gates.check_gradient(_corrupted(b, car_u))

    def car_du(raw):
        next(e for e in raw["upars"] if e["par"] == "car")["du"] += 1e-6
    with pytest.raises(OracleMismatch, match=r"brms's probe moved the coordinate of car by"):
        gates.check_gradient(_corrupted(b, car_du))

    def gradient(raw):
        i = next(k for k, e in enumerate(raw["upars"]) if e["par"] == "sdcar")
        raw["points"][3]["grad"]["values"][i] *= 1 + 1e-6
    with pytest.raises(OracleMismatch, match=r"point 3: gradient of sdcar_log__\[0\]"):
        gates.check_gradient(_corrupted(b, gradient))

    def default(raw):
        next(r for r in raw["default_prior"] if r["class"] == "sdcar")["prior"] = "normal(0, 1)"
    with pytest.raises(OracleMismatch, match="differs from brms's default_prior"):
        gates.check_default_prior(_corrupted(b, default))

    icar = _bundle("gauss_car_icar")

    def derived(raw):
        raw["points"][5]["pars"]["rcar"]["values"][7] += 1e-6
    with pytest.raises(OracleMismatch, match=r"point 5: the graph's rcar differs from brms's rcar"):
        gates.check_car_effects(_corrupted(icar, derived))
    gates.check_car_effects(_bundle("poisson_car_bym2"))
    with pytest.raises(OracleMismatch, match=r"brms car_scale is \[0.5163859009642695\]...; the model has \[0.5163859011368434\]"):
        gates.check_design(_bundle("gauss_car_bym2"))                    # D140: G1 expected to fail on the bym2 cases

    other = model("icar", frame=b.data, data2=b.data2)                    # another type against escar's fixture
    with pytest.raises(OracleMismatch, match=r"brms has the CAR data .*; the model's car\(W, gr = site, type = \"icar\"\) has"):
        adapter.check_car_design(b.fixture, other)
    with pytest.raises(OracleMismatch, match=r"brms samples car, which the AutoBayes model has not \(its CAR term: "):
        adapter.car_coordinate(b.fixture, other, {"index": 1, "du": 0.0}, "car", 0)
    plain = ab.gaussian("y ~ x", b.data, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sigma=ab.HalfNormal(1)))
    with pytest.raises(OracleMismatch, match=r"brms has the CAR data .*; the AutoBayes model has no CAR term"):
        adapter.check_car_design(b.fixture, plain)
    adapter.check_car_design(fx_without_car(), plain)
    assert gates.check_car_effects(SimpleNamespace(model=plain)) is None
    adapter.car_draws(plain, [], {}, {})


def fx_without_car():
    from tests.oracle import fixture as fx

    return fx.Fixture.load("gauss_ar1")
