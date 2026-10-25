"""F08 (brms-parity, docs/ordered-factors.md, docs/syntax-fixes.md): the fixes of the syntax audit (#85).

1. Ordered factors: brms codes an ordered factor predictor with R's polynomial contrasts (``contr.poly``); so does AutoBayes,
   for an ordered pandas Categorical declared with ``categorical=`` and for ``ordered(col)`` in the formula: the contrasts against
   an independent exact construction, the term model, the design, the priors, brms's names, ``describe()``, new rows, the other
   model kinds, and every refusal.
2. Group-level syntax as brms reads it: ``(x | g)`` is ``(1 + x | g)``, ``gr(g, cor = FALSE)`` is ``||``, ``gr(g, id = "p")`` is
   ``|p|``, and the advice of a slope-only term's refusal is brms's reading.
3. The refusals that now name their construct (``.``, backticks, ``Intercept``, ``::``, ``%in%``, ``^``, smooths, a transformed or
   ``cbind()`` response, ``disc ~``, ``nlf()``).
4. ``trials(<number>)``, brms's censoring labels and ``cat(K)``.
5. The syntax audit's ``data2`` entries.

Build-only; nothing samples. The numbers against brms are the oracle's (case ``gauss_ordered_factor``) and the corpus's
(tests/test_brms_syntax_corpus.py)."""
from fractions import Fraction
import math

import numpy as np
import pandas as pd
import pytest

import autobayes as ab
import importlib
from autobayes.default_priors import brms_coefficient_names
from autobayes.formula import (Categorical, FormulaError, PopTerm, POLY_LEVELS_MAX, contrast_terms, contrast_values, parse_formula,
                               poly_contrast_labels, poly_contrasts, trials_number, with_trials)
from autobayes.priors import PriorError
from tests.brms_syntax import audit

predict_mod = importlib.import_module("autobayes.predict")
PRIORS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1))


@pytest.fixture(scope="module")
def corpus_frame():
    df = audit.python_frame()
    df["fs"] = df["f"].astype(str)                       # the levels a, b, c as text
    return df


def frame(n=30, seed=8):
    rng = np.random.default_rng(seed)
    lv = ["lo", "mid", "hi"]
    df = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n), "z": rng.normal(size=n),
                       "f": [lv[i % 3] for i in range(n)], "g": [f"g{i % 5}" for i in range(n)],
                       "k": rng.integers(0, 6, n), "n": np.full(n, 10), "o": [(2, 10, 1, 3)[i % 4] for i in range(n)]})
    df["fo"] = pd.Categorical(df["f"], categories=lv, ordered=True)
    return df


# -- 1. polynomial contrasts ----------------------------------------------------------------------------------------
def exact_contr_poly(k):
    """An independent construction: Gram-Schmidt of the monomials 1, i, i^2, ... over the scores 1..k in exact rationals, each
    column then divided by its norm."""
    scores = [Fraction(i) for i in range(1, k + 1)]
    basis = []
    for d in range(k):
        v = [s ** d for s in scores]
        for b in basis:
            c = sum(x * y for x, y in zip(v, b)) / sum(y * y for y in b)
            v = [x - c * y for x, y in zip(v, b)]
        basis.append(v)
    return np.array([[float(x) / math.sqrt(sum(y * y for y in b)) for x in b] for b in basis[1:]]).T


@pytest.mark.parametrize("k", range(2, POLY_LEVELS_MAX + 1))
def test_poly_contrasts_are_the_orthonormal_polynomials(k):
    C = poly_contrasts(k)
    assert C.shape == (k, k - 1)
    assert np.max(np.abs(C - exact_contr_poly(k))) <= 4e-16
    assert np.max(np.abs(C.T @ C - np.eye(k - 1))) <= 1e-15 and np.max(np.abs(C.sum(0))) <= 1e-15
    assert (C[-1] > 0).all()                             # a positive leading coefficient: positive at the highest score


def test_poly_contrasts_closed_forms_and_labels():
    assert np.allclose(poly_contrasts(3), np.array([[-1, 1], [0, -2], [1, 1]]) / np.sqrt([2.0, 6.0]), atol=1e-16)
    assert np.allclose(poly_contrasts(4)[:, 2], np.array([-1, 3, -3, 1]) / np.sqrt(20.0), atol=1e-16)
    assert np.allclose(poly_contrasts(5)[:, 0], np.array([-2, -1, 0, 1, 2]) / np.sqrt(10.0), atol=1e-16)
    assert poly_contrast_labels(6) == (".L", ".Q", ".C", "^4", "^5")
    for k in (1, POLY_LEVELS_MAX + 1):
        with pytest.raises(ValueError, match="polynomial contrasts are built for 2 to 16 levels"):
            poly_contrasts(k)


def test_categorical_and_contrast_terms():
    cat = Categorical("fo", "lo", ("lo", "mid", "hi"), coding="poly")
    assert cat.term_name == "fo" and cat.is_poly and cat.contrast_labels == (".L", ".Q")
    assert cat.contrast_columns == ("fo.L", "fo.Q")
    other = Categorical("z", "", coding="poly", term="ordered(z)")
    assert other.term_name == "ordered(z)" and other.contrast_labels == () and other.contrast_columns == ()
    assert [t.name for t in contrast_terms(other)] == ["ordered(z)"]                     # the placeholder before the levels
    assert not Categorical("f", "a", ("a", "b")).is_poly and Categorical("f", "a", ("a", "b")).contrast_labels == ()
    (t,) = contrast_terms(other)
    assert t.kind == "contrast" and t.operands == ("z",) and t.bambi_name == "`ordered(z)`" and not t.is_vector
    for bad in (dict(name="fo.L", kind="contrast", columns=("fo.L",)), dict(name="x", kind="numeric", columns=("x",), factor="x"),
                dict(name="fo.L", kind="contrast", columns=("other",), factor="fo")):
        with pytest.raises(ValueError, match="a contrast term"):
            PopTerm(**bad)
    df = pd.DataFrame({"fo": ["lo", "hi", "mid", "other"]})
    with pytest.raises(FormulaError, match=r"levels \['other'\] that the ordered factor 'fo' was not fitted with"):
        contrast_values(contrast_terms(cat)[0], df, (cat,))
    assert np.allclose(contrast_values(contrast_terms(cat)[1], df.iloc[:3], (cat,)), poly_contrasts(3)[[0, 2, 1], 1])


def test_ordered_pandas_categorical_has_polynomial_contrasts():
    df = frame()
    m = ab.gaussian("y ~ x + fo", df, PRIORS, categorical={"fo": "lo"})
    assert m.spec.terms == ("x", "fo.L", "fo.Q") and m.spec.columns == ("x", "fo.L", "fo.Q")
    assert m.spec.canonical_formula == "y ~ x + fo" and m.spec.categorical("fo").levels == ("lo", "mid", "hi")
    idx = df["f"].map({"lo": 0, "mid": 1, "hi": 2}).to_numpy()
    assert np.max(np.abs(m.design.X_raw[:, 1:] - poly_contrasts(3)[idx])) == 0.0
    assert m.names["b:fo.L"] == "fo.L" and brms_coefficient_names(m.spec) == {"x": "x", "fo.L": "fo.L", "fo.Q": "fo.Q"}
    assert [r["coef"] for r in m.prior_table() if r["class"] == "b"] == ["", "fo.L", "fo.Q", "x"]
    text = m.describe()
    assert "ordered factor(s) with polynomial contrasts" in text and "lo < mid < hi" in text
    assert "(fo.L[fo_i] - mean(fo.L)) * b_fo.L" in text and "(brms's b_fo.L, b_fo.Q)" in text
    assert "a dummy column's mean" not in text and "the reference level" not in text
    from autobayes import summaries
    with pytest.raises(ValueError, match="fixef\\(\\) of the contrast term 'fo.L' is not offered.*D162"):
        summaries.refuse_unsupported(m, "fixef()")
    # the full-rank factor of a '0 +' formula is coded by indicators, as R codes it
    m0 = ab.gaussian("y ~ 0 + fo", df, ab.Priors(b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)), categorical={"fo": "lo"})
    assert m0.spec.columns == ("fo[lo]", "fo[hi]", "fo[mid]")
    # and an ordered factor next to a treatment-coded one: both kinds of text in describe()
    both = ab.gaussian("y ~ f2 + fo", df.assign(f2=df["g"]), PRIORS, categorical={"fo": "lo", "f2": "g0"}).describe()
    assert both.splitlines()[0].startswith("AutoBayes M0 model with categorical predictor(s) and ordered factor(s) with polynomial")
    assert "treatment coding" in both and "a dummy column's mean is its level's proportion" in both
    inter = ab.gaussian("y ~ x * f2 + fo", df.assign(f2=df["g"]), PRIORS, categorical={"fo": "lo", "f2": "g0"}).describe()
    assert "every expanded design column (x, 1[z = l], x * 1[z = l])" in inter and "(fo.L[fo_i] - mean(fo.L))" in inter
    raw = ab.gaussian("y ~ x + fo", df, PRIORS, categorical={"fo": "lo"}, center_predictors=False).describe()
    assert "a is the raw-scale intercept." in raw and "reference level" not in raw and "fo.L[fo_i] * b_fo.L" in raw


def test_ordered_factor_refusals():
    df = frame()
    with pytest.raises(FormulaError, match=r"ordered pandas Categorical: brms codes an ordered factor with polynomial contrasts over its "
                                           r"levels \['lo', 'mid', 'hi'\].*declare its lowest level, categorical=\{'fo': 'lo'\}"):
        ab.gaussian("y ~ fo", df, PRIORS, categorical={"fo": "mid"})
    big = pd.DataFrame({"y": np.arange(34.0) % 5, "u": [f"l{i % 17:02d}" for i in range(34)]})
    with pytest.raises(FormulaError, match="'ordered\\(u\\)' has 17 levels; polynomial contrasts are offered for at most 16.*D168"):
        ab.gaussian("y ~ ordered(u)", big, PRIORS)
    assert ab.gaussian("y ~ ordered(u)", big[big.u != "l16"], PRIORS).spec.categorical("u").levels[-1] == "l15"
    for formula, kw, message in [
        ("y ~ x * fo", {"categorical": {"fo": "lo"}}, "'x:fo': an interaction with the ordered factor 'fo' is not supported"),
        ("y ~ x * ordered(f)", {}, r"an interaction with an ordered factor \('ordered\(f\)'\) is not supported"),
        ("y ~ ordered(x + z)", {}, "ordered\\(\\) takes one column name here, not 'x \\+ z'"),
        ("y ~ ordered(y)", {}, "the response 'y' cannot be an ordered factor predictor"),
        ("k | trials(n) ~ ordered(n)", {}, "the trials column 'n' cannot be an ordered factor predictor"),
        ("y ~ x + ordered(f)", {"categorical": {"f": "lo"}}, "'f' is declared categorical; ordered\\(\\) reads it"),
        ("y ~ f + ordered(f)", {"categorical": {"f": "lo"}}, "'f' is read by 'ordered\\(f\\)' and by another term"),
        ("y ~ x + ordered(x)", {}, "'x' is read by 'ordered\\(x\\)' and by another term"),
        ("y ~ 0 + ordered(f)", {}, "'ordered\\(f\\)' is the first factor of a formula without an intercept"),
    ]:
        with pytest.raises(FormulaError, match=message):
            ab.gaussian(formula, df, PRIORS, **kw) if not formula.startswith("k |") else ab.binomial(formula, df, PRIORS, **kw)
    with pytest.raises(FormulaError, match="the polynomial contrast 'fo.L' of an ordered factor .* has the name of another term"):
        ab.gaussian("y ~ fo + fo.L", df.assign(**{"fo.L": df["x"]}), PRIORS, categorical={"fo": "lo"})


def test_ordered_in_the_formula_reads_rs_order():
    df = frame()
    text = ab.gaussian("y ~ ordered(f)", df, PRIORS)                        # text: sorted, as R's ordered() of text in C order
    assert text.spec.categorical("f").levels == ("hi", "lo", "mid") and text.spec.terms == ("ordered(f).L", "ordered(f).Q")
    assert brms_coefficient_names(text.spec) == {"ordered(f).L": "orderedf.L", "ordered(f).Q": "orderedf.Q"}
    num = ab.gaussian("y ~ ordered(o)", df, PRIORS)                         # whole numbers in numeric order, not as text
    assert num.spec.categorical("o").levels == ("1", "2", "3", "10")
    assert brms_coefficient_names(num.spec)["ordered(o).C"] == "orderedo.C"
    cat = ab.gaussian("y ~ ordered(f)", df.assign(f=pd.Categorical(df["f"], categories=["mid", "lo", "hi"])), PRIORS)
    assert cat.spec.categorical("f").levels == ("mid", "lo", "hi")         # a pandas Categorical: its own order
    with pytest.raises(FormulaError, match="categorical values must be integral or labels"):
        ab.gaussian("y ~ ordered(x)", df, PRIORS)
    five = pd.DataFrame({"y": np.arange(20.0) % 3, "u": [f"u{i % 5}" for i in range(20)]})
    assert brms_coefficient_names(ab.gaussian("y ~ ordered(u)", five, PRIORS).spec)["ordered(u)^4"] == "ordereduE4"
    full = ab.gaussian("y ~ 0 + x + f + ordered(g)", df, ab.Priors(b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)), categorical={"f": "lo"})
    assert full.spec.columns == ("x", "f[lo]", "f[hi]", "f[mid]", *(f"ordered(g){c}" for c in (".L", ".Q", ".C", "^4")))


def test_ordered_factor_priors():
    df = frame()
    pr = ab.Priors(intercept=ab.Normal(0, 5), b={"x": ab.Normal(0, 1), "fo": {".L": ab.Normal(1, 1), ".Q": ab.Normal(-1, 2)}},
                   sigma=ab.HalfNormal(1))
    m = ab.gaussian("y ~ x + fo", df, pr, categorical={"fo": "lo"})
    coef = m.priors.coefficient_priors(m.spec)
    assert coef["fo.L"].params["mu"] == 1 and coef["fo.Q"].params["sigma"] == 2
    one = ab.gaussian("y ~ ordered(f)", df, ab.Priors(intercept=ab.Normal(0, 5), b={"ordered(f)": ab.Normal(0, 3)}, sigma=ab.HalfNormal(1)))
    assert {k: v.params["sigma"] for k, v in one.priors.coefficient_priors(one.spec).items()} == {"ordered(f).L": 3, "ordered(f).Q": 3}
    mapped = ab.Priors(intercept=ab.Normal(0, 5), b={"ordered(f)": {".L": ab.Normal(0, 1), ".Q": ab.Normal(0, 1)}}, sigma=ab.HalfNormal(1))
    assert ab.gaussian("y ~ ordered(f)", df, mapped).spec.terms == ("ordered(f).L", "ordered(f).Q")
    for b, message in [({"x": ab.Normal(0, 1), "fo": {".L": ab.Normal(0, 1)}}, r"must cover exactly its polynomial contrasts \['.L', '.Q'\]"),
                       ({"x": ab.Normal(0, 1), "fo": {".L": ab.Normal(0, 1), ".Q": ab.Normal(0, 1), "^4": ab.Normal(0, 1)}},
                        r"unexpected \['\^4'\]"),
                       ({"x": ab.Normal(0, 1), "fo.L": ab.Normal(0, 1), "fo.Q": ab.Normal(0, 1)}, "must cover exactly the predictor terms")]:
        with pytest.raises(PriorError, match=message):
            ab.gaussian("y ~ x + fo", df, ab.Priors(intercept=ab.Normal(0, 5), b=b, sigma=ab.HalfNormal(1)), categorical={"fo": "lo"})
    for entry, message in [([ab.Normal(0, 1)], "takes one prior or a mapping \\{contrast: prior\\}"),
                           ({".L": 1.0, ".Q": ab.Normal(0, 1)}, "must be a Prior, got float"), ({}, "are empty")]:
        with pytest.raises(PriorError, match=message):
            ab.gaussian("y ~ ordered(f)", df, ab.Priors(intercept=ab.Normal(0, 5), b={"ordered(f)": entry}, sigma=ab.HalfNormal(1)))


def test_ordered_factor_new_rows_and_distributional_formula(corpus_frame):
    df = corpus_frame
    m = ab.gaussian(["y ~ x + fo", "sigma ~ ordered(fs)"], df, ab.default_priors(["y ~ x + fo", "sigma ~ ordered(fs)"], df,
                    categorical={"fo": "lo"}), categorical={"fo": "lo"})
    C = poly_contrasts(3)
    assert np.max(np.abs(m.design.dpars["sigma"].X_raw - C[df["fs"].map({"a": 0, "b": 1, "c": 2}).to_numpy()])) == 0.0
    assert "orderedfs.L" in [r["coef"] for r in m.prior_table() if r.get("dpar") == "sigma"]
    new = df.iloc[[5, 1, 2]][["x", "fo", "fs"]].reset_index(drop=True)
    checked = predict_mod._check_newdata(m, new)
    assert np.allclose(checked["fo.Q"], C[new["fo"].map({"lo": 0, "mid": 1, "hi": 2}).to_numpy(), 1])
    assert np.allclose(checked["ordered(fs).L"], C[new["fs"].map({"a": 0, "b": 1, "c": 2}).to_numpy(), 0])
    assert np.allclose(predict_mod.raw_design(m, checked)[:, 1:], C[new["fo"].map({"lo": 0, "mid": 1, "hi": 2}).to_numpy()])


@pytest.mark.parametrize("ctor, formula, kw", [
    ("cumulative", "yord ~ x + ordered(fs)", {}), ("categorical", "ycat ~ x + ordered(fs)", {}),
    ("mixture", "y ~ ordered(fs)", {"families": ["gaussian", "gaussian"]}), ("bernoulli", "ybin ~ ordered(fs)", {}),
    ("gaussian", "mvbind(y, y2) ~ ordered(fs)", {"rescor": True}),
    ("gaussian", ["y ~ a * exp(b * x)", "a ~ ordered(fs)", "b ~ 1"], {"nl": True}),
    ("multinomial", "cbind(ymn1, ymn2, ymn3) | trials(ntr) ~ ordered(fs)", {}),
])
def test_ordered_factor_in_other_models(corpus_frame, ctor, formula, kw):
    df = corpus_frame
    pk = {k: v for k, v in kw.items() if k in audit.DEFAULT_PRIOR_KWARGS}
    m = getattr(ab, ctor)(formula, df, ab.default_priors(formula, df, family=ctor, **pk), **kw)
    want = poly_contrasts(3)[df["fs"].map({"a": 0, "b": 1, "c": 2}).to_numpy()]
    if kw.get("nl"):
        X = m.design.nlpars["a"].X[:, 1:]
    elif kw.get("rescor"):
        X = m.submodels["y2"].design.X_raw
    elif ctor == "multinomial":
        X = next(iter(m.design.mus.values())).X_raw
    else:
        X = m.design.X_raw[:, -2:]
    assert np.max(np.abs(X - want)) == 0.0
    text = m.describe()
    assert "ordered(fs)" in text and ("ordered(fs).L[fs_i]" in text or ctor in ("cumulative", "categorical", "bernoulli")
                                      or kw.get("rescor"))


# -- 2. group-level syntax ------------------------------------------------------------------------------------------
def test_implicit_intercept_of_a_group_term():
    df = frame()
    for written, canonical, correlated in [("(x | g)", "(1 + x | g)", True), ("(x || g)", "(1 + x || g)", False)]:
        (t,) = parse_formula(f"y ~ x + {written}").group_terms
        assert t.coefs == ("Intercept", "x") and t.correlated is correlated and t.canonical == canonical
        pr = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}},
                       sigma=ab.HalfNormal(1), cor=ab.LKJ(1) if correlated else None)
        a = ab.gaussian(f"y ~ x + {written}", df, pr)
        b = ab.gaussian(f"y ~ x + {canonical}", df, a.priors)
        assert a.names == b.names and a.group_names == b.group_names and np.array_equal(a.design.Z, b.design.Z)


@pytest.mark.parametrize("formula, message", [
    ("y ~ x + (0 + x | g)", "'0' removes the varying intercept of 'g': varying slopes without the varying intercept are not supported; "
                            r"write '\(1 \+ x \| g\)'"),
    ("y ~ x + (x - 1 | g)", r"'\(x - 1 \| g\)': '- 1' removes the varying intercept of 'g'"),
    ("y ~ x + (x - z | g)", r"'-' inside a group term removes only the varying intercept, '- 1'"),
    ("y ~ x + (- 1 | g)", r"'-' inside a group term removes only the varying intercept"),
    ("y ~ x + (1 | g) + (0 + x | g)", r"next to '\(1 \| g\)' it is brms's varying intercept and slope without a correlation, which is "
                                      r"one term here: write '\(1 \+ x \|\| g\)'"),
    ("y ~ x + z + (1 + x |p| g) + (0 + z |p| g)", r"share the correlation ID 'p' as one block .* write '\(1 \+ x \+ z \| g\)'"),
    ("y ~ x + z + (1 + z | g) + (0 + x | g)", r"varying slopes without the varying intercept are not supported; write '\(1 \+ x \| g\)'"),
])
def test_slope_only_refusals_give_brms_reading(formula, message):
    with pytest.raises(FormulaError, match=message):
        parse_formula(formula)


def test_gr_cor_and_id_spellings():
    df = frame()
    (t,) = parse_formula("y ~ x + (1 + x | gr(g, cor = FALSE))").group_terms
    assert t.correlated is False and t.canonical == "(1 + x || g)"
    (u,) = parse_formula("y ~ x + (1 + x | gr(g, cor = TRUE))").group_terms
    assert u.correlated is True
    for formula, message in [("y ~ x + (1 + x | gr(g, cor = F))", "gr\\(\\)'s cor = takes TRUE or FALSE, got 'F'"),
                             ("y ~ x + (1 + x || gr(g, cor = TRUE))", "'\\|\\|' leaves the coefficients uncorrelated"),
                             ("y ~ x + (1 + x | gr(g, id = p))", "gr\\(\\)'s id = takes a quoted name"),
                             ("y ~ x + (1 + x || gr(g, id = 'p'))", "'\\|\\|' with an ID is not brms syntax"),
                             ("y ~ x + (1 | gr(g, dist = 'student'))", r"brms's gr\(\) options \['dist'\] are not supported")]:
        with pytest.raises(FormulaError, match=message):
            parse_formula(formula)
    pr = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd={"g": {"Intercept": ab.HalfNormal(1), "x": ab.HalfNormal(1)}},
                   sigma={"intercept": ab.Normal(0, 1), "sd": ab.HalfNormal(1)}, cor={"g": ab.LKJ(2)})
    spelt = ab.gaussian(["y ~ x + (1 + x | gr(g, id = \"p\"))", "sigma ~ 1 + (1 | gr(g, id = 'p'))"], df, pr)
    bars = ab.gaussian(["y ~ x + (1 + x |p| g)", "sigma ~ 1 + (1 |p| g)"], df, pr)
    assert spelt.spec.shared_ids == bars.spec.shared_ids and spelt.names == bars.names


# -- 3. refusals that name their construct --------------------------------------------------------------------------
@pytest.mark.parametrize("formula, family, message", [
    ("y ~ .", "gaussian", "R's '.' \\(every column of the data not otherwise in the formula\\) is not supported"),
    ("y ~ `x`", "gaussian", "backtick-quoted names are not supported; 'x' is a column name as it stands"),
    ("y ~ `my x`", "gaussian", "backtick-quoted names are not supported; rename the column"),
    ("y ~ 0 + Intercept + x", "gaussian", "brms's reserved term 'Intercept'"),
    ("y ~ splines::ns(x, df = 3)", "gaussian", "R's namespace operator '::'"),
    ("y ~ x %in% f", "gaussian", "R's operator %in% is not supported in a formula; nesting x %in% f is the interaction x:f"),
    ("y ~ x %o% z", "gaussian", "R's operator %o% is not supported in a formula$"),
    ("y ~ (x + z)^2", "gaussian", "R's formula operator '\\^'"),
    ("y ~ x^2", "gaussian", "R's formula operator '\\^'"),
    ("y ~ s(x)", "gaussian", "brms's smooth term s\\(\\) \\(mgcv's penalised splines"),
    ("y ~ t2(x, z)", "gaussian", "brms's smooth term t2\\(\\)"),
    ("log(y) ~ x", "gaussian", "the response 'log\\(y\\)' is an expression; a transformed response"),
    ("cbind(y, z) ~ x", "gaussian", "the gaussian family takes one response column; several Gaussian responses are brms's mvbind"),
    ("cbind(k, k) ~ x", "poisson", "the poisson family takes one response column$"),
])
def test_refusals_name_the_construct(formula, family, message):
    with pytest.raises(FormulaError, match=message):
        parse_formula(formula, None, family)


def test_distributional_and_nonlinear_refusals_name_the_construct(corpus_frame):
    df = corpus_frame
    with pytest.raises(FormulaError, match="brms's disc ~ ... \\(the discrimination of an ordinal family"):
        parse_formula(["yord ~ x", "disc ~ z"], None, "cumulative")
    with pytest.raises(FormulaError, match="'a' is a parameter of a non-linear formula of sigma \\(brms's nlf\\(sigma ~ ...\\)"):
        parse_formula(["y ~ x", "sigma ~ a * exp(c * z)", "a ~ 1", "c ~ 1"])
    with pytest.raises(FormulaError, match="'c' is a parameter of the formula of the non-linear parameter 'a' \\(brms's nlf\\(a ~ ...\\)"):
        ab.gaussian(["y ~ a * exp(b * x)", "b ~ 1", "a ~ c * z", "c ~ 1"], df, ab.Priors(), nl=True)


# -- 4. trials(<number>), censoring labels, cat(K) ------------------------------------------------------------------
def test_constant_trials(corpus_frame):
    df = corpus_frame
    spec = parse_formula("k | trials(15) ~ x", None, "binomial")
    assert spec.trials == "15" and trials_number(spec) == 15 and spec.canonical_formula == "k | trials(15) ~ x"
    assert trials_number(parse_formula("k | trials(ntr) ~ x", None, "binomial")) is None
    assert with_trials(parse_formula("k | trials(ntr) ~ x", None, "binomial"), df) is df
    m = ab.binomial("k | trials(15) ~ x", df, ab.default_priors("k | trials(15) ~ x", df, family="binomial"))
    assert (m.design.trials == 15).all() and "15" not in df.columns
    checked = predict_mod._check_newdata(m, df[["x"]].iloc[:3], need_trials=True)
    assert checked["15"].tolist() == [15, 15, 15]
    with pytest.raises(FormulaError, match="trials\\(\\) takes one column name or a whole number, trials\\(15\\)"):
        parse_formula("k | trials(15.5) ~ x", None, "binomial")


def test_censoring_labels():
    df = frame()
    df["c"] = ["left", "none", "right", "none", "none"] * 6
    m = ab.gaussian("y | cens(c) ~ x", df, PRIORS)
    assert m.design.cens.tolist() == [-1, 0, 1, 0, 0] * 6
    lg = ab.gaussian("y | cens(c) ~ x", df.assign(c=df["c"] == "right"), PRIORS)
    assert lg.design.cens.tolist() == [0, 0, 1, 0, 0] * 6
    with pytest.raises(FormulaError, match="the censoring labels of 'cens\\(c\\)' are brms's 'left', 'none', 'right' and 'interval'.*'Left'"):
        ab.gaussian("y | cens(c) ~ x", df.assign(c=df["c"].str.capitalize()), PRIORS)
    with pytest.raises(FormulaError, match="code 2 \\(interval censoring\\) needs brms's cens\\(c, y2\\)"):
        ab.gaussian("y | cens(c) ~ x", df.assign(c=df["c"].replace("left", "interval")), PRIORS)
    iv = ab.gaussian("y | cens(c, y2) ~ x", df.assign(c=df["c"].replace("left", "interval"), y2=df["y"] + 1), PRIORS)
    assert iv.design.cens.tolist()[:3] == [2, 0, 1]


def test_cat_is_the_datas_number_of_categories(corpus_frame):
    df = corpus_frame
    spec = parse_formula("yord | cat(4) ~ x", None, "cumulative")
    assert spec.addition("cat").term == "cat(4)" and spec.canonical_formula == "yord | cat(4) ~ x"
    pr = ab.default_priors("yord ~ x", df, family="cumulative")
    plain, cat = ab.cumulative("yord ~ x", df, pr), ab.cumulative("yord | cat(4) ~ x", df, pr)
    text = cat.describe()
    assert plain.names == cat.names and plain.categories == cat.categories and "  categories: cat(4): the response's 4 categories" in text
    assert "trunc()" not in text and "truncation" not in text
    with pytest.raises(FormulaError, match="the response has 4 categories here .*cat\\(\\) with another number.*D169"):
        ab.cumulative("yord | cat(5) ~ x", df, pr)
    for formula, family, message in [("yord | cat(x) ~ x", "cumulative", "cat\\(\\) takes the number of categories, a whole number of at least 2"),
                                     ("yord | cat(1) ~ x", "cumulative", "a whole number of at least 2"),
                                     ("y | cat(4) ~ x", "gaussian", "brms's cat\\(\\), an ordinal response's number of categories, is taken"),
                                     ("yord | cat(4) + thres(gr = f) ~ x", "cumulative", "without thres\\(\\)")]:
        with pytest.raises(FormulaError, match=message):
            parse_formula(formula, None, family)


# -- 5. the syntax audit's data2 entries ----------------------------------------------------------------------------
def test_audit_passes_data2_and_compares_the_terms_data(corpus_frame, monkeypatch):
    corpus, store = audit.load_json(audit.CORPUS), audit.load_json(audit.STANDATA)
    entries = {e["id"]: e for e in corpus["entries"]}
    for eid in ("ac_car", "ac_sar", "ac_fcor"):
        assert audit.run_entry(entries[eid], corpus_frame, store)["issues"] == []
    objects = audit.make_data2()
    assert list(objects["M_car"].index) == audit.FACTORS["g"] and np.allclose(objects["M_sar"].sum(axis=1), 1.0)
    assert objects["V_fcor"][0, 0] == 0.6 and objects["A_cov"].iloc[0, 2] == 0.25
    m, _ = audit.build(entries["ac_sar"], corpus_frame)
    assert set(audit.data2_data(m)) == {"Msar", "eigenMsar", "N_tg"}
    changed = {**objects, "V_fcor": objects["V_fcor"] + 0.01 * np.eye(len(corpus_frame))}   # another matrix, another reading
    monkeypatch.setattr(audit, "make_data2", lambda n=audit.N_ROWS: changed)
    model, _ = audit.build(entries["ac_fcor"], corpus_frame)
    issues, _ = audit.compare(model, "ac_fcor", store)
    assert issues == ["brms's Mfcor differs from AutoBayes's Mfcor of its data2 term"]
