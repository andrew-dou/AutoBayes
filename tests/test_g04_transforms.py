"""In-formula expressions and fitted transforms (docs/expressions.md; divergences D10, D16, D181, D182): ``I()`` of arithmetic,
comparisons and fractional powers, ``log()``, ``exp()``, ``sqrt()`` and ``abs()`` of expressions, ``scale()``, ``poly()``
(orthogonal and raw) and ``factor()`` with their fitted parameters on new rows, and several offsets and expressions inside
``offset()``.

The grammar and its refusals by message, the spelling (R's deparse and brms's coefficient names), every operator and function
against NumPy, the fitted transforms against hand-computed values and an independent QR construction, the term model and its
round trip, the model's design, graph, priors and ``describe()``, predictions on fitted and new rows computed by hand,
``default_priors()``, ``conditional_effects()``'s model-frame columns, and the combinations refused. Build-only; nothing
samples. The numbers against brms are the oracle's (tests/oracle, cases ``gauss_I_expressions``, ``gauss_log_abs``,
``gauss_scale_poly``, ``gauss_scale_poly_post``, ``gauss_poly_raw``, ``gauss_factor_call``, ``gauss_factor_call_newlevels``,
``poisson_two_offsets``)."""
import math
import re
from dataclasses import asdict

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import autobayes as ab
from autobayes import expressions as ex
from autobayes.formula import (FormulaError, ModelSpec, Offset, PopTerm, Transform, expression_term, fit_expressions,
                               is_expression_term, offset_expression, parse_formula, pop_term, pop_term_design)

LEVELS = ("g0", "g1", "g2", "g3")


def frame(n=32, seed=11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": rng.normal(size=n), "x": np.round(rng.normal(0.2, 1.0, n), 4), "w": np.round(rng.normal(1.0, 0.5, n), 4),
                       "xpos": np.round(np.exp(rng.normal(0.0, 0.5, n)), 4), "o": np.array([1, 2, 3, 2, 1, 3, 3, 2] * (n // 8)),
                       "off": np.round(rng.uniform(-0.5, 0.5, n), 4), "g": [LEVELS[i % 4] for i in range(n)],
                       "z": rng.choice(["p", "q"], size=n), "k": rng.integers(0, 5, n), "nn": np.full(n, 5)})
    df["cnt"] = rng.poisson(np.exp(0.3 + 0.2 * df.x + df.off)).astype(np.int64)
    df["y2"] = df.y + rng.normal(size=n)
    assert (df.x > 0).any() and (df.x < 0).any() and set(df.o) == {1, 2, 3}
    return df


FORMULA = ("y ~ x + I(x * w) + log(xpos + 1) + scale(w) + poly(x, 2) + poly(w, 2, raw = TRUE) + factor(o) + I(x > 0) "
           "+ offset(2 * off) + offset(log(xpos)) + (1 | g)")
COLUMNS = ("x", "I(x * w)", "log(xpos + 1)", "scale(w)", "poly(x, 2)1", "poly(x, 2)2", "poly(w, 2, raw = TRUE)1", "poly(w, 2, raw = TRUE)2",
           "factor(o)2", "factor(o)3", "I(x > 0)TRUE")
PRIORS = ab.Priors(intercept=ab.Normal(0.2, 5), b=ab.Normal(0.1, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1.5))


@pytest.fixture(scope="module")
def model():
    return ab.gaussian(FORMULA, frame(), PRIORS)


def poly_reference(x_fit, degree, x):
    """R's construction of ``poly(x_fit, degree)`` at ``x`` (a QR decomposition of the centred powers, each column of Q times R's
    diagonal, normalised over the fitted rows), written independently of ``expressions._poly_fit``'s recurrence."""
    centre = x_fit.mean()
    V = np.vander(x_fit - centre, degree + 1, increasing=True)
    Q, R = np.linalg.qr(V)
    C = np.linalg.solve(R, np.diag(np.diag(R)))           # Q diag(R) = V C: the monic orthogonal polynomials' coefficients
    Z_fit = V @ C
    norms = np.sqrt((Z_fit ** 2).sum(0))
    return (np.vander(x - centre, degree + 1, increasing=True) @ C / norms)[:, 1:]


# -- 1. the grammar: parsing, spelling and what is refused -------------------------------------------------------------------
@pytest.mark.parametrize("text,canonical,r_spelling", [
    ("I(x*w)", "I(x * w)", "I(x * w)"), ("I(xpos^.5)", "I(xpos^.5)", "I(xpos^0.5)"), ("I( x>0 )", "I(x > 0)", "I(x > 0)"),
    ("log(xpos+1)", "log(xpos + 1)", "log(xpos + 1)"), ("abs(x)", "abs(x)", "abs(x)"), ("I((x + 1)^2)", "I((x + 1)^2)", "I((x + 1)^2)"),
    ("I(-x^2)", "I(-x^2)", "I(-x^2)"), ("I((-x)^2)", "I((-x)^2)", "I((-x)^2)"), ("I(x^-0.5)", "I(x^-0.5)", "I(x^-0.5)"),
    ("exp(-(x * w))", "exp(-(x * w))", "exp(-(x * w))"), ("I(x - (w - 1))", "I(x - (w - 1))", "I(x - (w - 1))"),
    ("I(2^x^2)", "I(2^x^2)", "I(2^x^2)"), ("I((x^2)^3)", "I((x^2)^3)", "I((x^2)^3)"), ("I(x / 2 / w)", "I(x / 2 / w)", "I(x / 2 / w)"),
    ("I(x / (2 / w))", "I(x / (2 / w))", "I(x / (2 / w))"), ("I((x < w) > 0)", "I((x < w) > 0)", "I((x < w) > 0)"),
    ("I(x**2.0)", "I(x^2.0)", "I(x^2)"), ("I(x.1 * 1e5)", "I(x.1 * 1e5)", "I(x.1 * 1e+05)"), ("sqrt(abs(x))", "sqrt(abs(x))", "sqrt(abs(x))"),
    ("poly(x,3,raw=TRUE)", "poly(x, 3, raw = TRUE)", "poly(x, 3, raw = TRUE)"), ("poly(x, degree = 2)", "poly(x, degree = 2)", "poly(x, degree = 2)"),
    ("scale( x )", "scale(x)", "scale(x)"), ("factor(o)", "factor(o)", "factor(o)"), ("I(-x * w)", "I(-x * w)", "I(-x * w)"),
])
def test_canonical_and_r_spelling(text, canonical, r_spelling):
    form, spelled, _ = ex.read_term(text)
    assert spelled == canonical and ex.read_term(canonical)[1] == canonical                     # the spelling is stable
    assert ex.deparse(ex.parse(canonical), r_numbers=True) == r_spelling


def test_trees_walk_and_columns():
    e = ex.parse("I((x > 0) * log(w) - -x)")
    assert e == ("call", "I", (("bin", "-", ("bin", "*", ("cmp", ">", ("col", "x"), ("num", "0")), ("call", "log", (("col", "w"),))),
                                ("neg", ("col", "x"))),))
    assert ex.columns_of(e) == ("x", "w") and [n[0] for n in ex.walk(e)] == ["call", "bin", "bin", "cmp", "col", "num", "call", "col", "neg", "col"]
    assert ex.deparse(e) == "I((x > 0) * log(w) - -x)" and ex.deparse(e, row=True) == "I((x_i > 0) * log(w_i) - -x_i)"
    assert ex.parse("poly(x, 2, raw = T)") == ("call", "poly", (("col", "x"), ("num", "2")), (("raw", ("col", "T")),))


@pytest.mark.parametrize("text,message", [
    ("sin(x)", "the function sin() is not supported"), ("log(x, 2)", "log() takes one argument here; log(x, base) is not supported"),
    ("I(x = 2)", "I() takes one argument here"), ("exp()", "exp() takes one argument here"),
    ("I(x %% 2)", "the character '%' is not part of the expressions read here"), ("I(x & w)", "the character '&'"),
    ("I(x[1])", "the character '['"), ("I('x')", "the character \"'\""), ("I(x <- 2)", "R's assignment operator"), ("I(2 -> x)", "R's assignment operator"),
    ("I(x < w < 1)", "a chained or unknown comparison is not supported"), ("I(1L * x)", "'I(1L * x)' is not an expression of this grammar"),
    ("I(0x10 * x)", "the literal '0x10' is not a number here"), ("I(1j * x)", "the literal '1j' is not a number here"),
    ("I(+x)", "this unary operator is not part of the expressions read here"), ("I(x // 2)", "this operator is not part of the expressions"),
    ("I(x << 2)", "this operator"), ("I((x, w))", "Tuple is not part of the expressions read here"), ("I(*x)", "Starred is not part"),
    ("ns(x, 3)", "the spline basis ns() is not supported"), ("bs(x)", "the spline basis bs() is not supported"),
    ("I(scale(x))", "scale() is a term of its own here"), ("log(factor(o))", "factor() is a term of its own here"),
    ("I(2)", "'I(2)' reads no column"), ("x + 1", "'x + 1' is not a call"), ("np.log(x)", "the function np.log() is not supported"),
    ("scale(x, center = FALSE)", "scale() takes one column and no other argument here (scale(x)); scale()'s center = and scale = are not supported"),
    ("factor(o, levels = 3)", "factor()'s levels =, labels = and ordered = are not supported"), ("factor(o + 1)", "factor() takes one column name first"),
    ("scale()", "scale() takes one column name first here"), ("poly(x, w)", "several columns (R's polym), coefs = and simple = are not supported"),
    ("poly(x, 2, coefs = 1)", "several columns (R's polym)"), ("poly(x, 2, degree = 2)", "several columns"), ("poly(x, 2, 3)", "several columns"),
    ("poly(x, 2.5)", "poly()'s degree must be a whole number from 1 to 10 here"), ("poly(x, 0)", "poly()'s degree must be a whole number"),
    ("poly(x, 11)", "from 1 to 10"), ("poly(x, 2, raw = 1)", "poly()'s raw = takes TRUE or FALSE"), ("poly(x, 2, raw = yes)", "raw = takes TRUE or FALSE"),
    ("poly(x, **k)", "poly(**...) is not supported"),
])
def test_grammar_refusals(text, message):
    with pytest.raises(ex.ExpressionError, match=re.escape(message)):
        ex.read_term(text)


def test_r_number_is_r_deparse():
    cases = {".5": "0.5", "2.0": "2", "2": "2", "100000": "1e+05", "1e5": "1e+05", "10000": "10000", "0.0001": "1e-04", "0.001": "0.001",
             "123456.789": "123456.789", "1e-10": "1e-10", "0": "0", "0.1": "0.1", "1.5e3": "1500", "3.14159265358979": "3.14159265358979"}
    assert {k: ex.r_number(k) for k in cases} == cases


# -- 2. evaluation over the data -------------------------------------------------------------------------------------------------
def test_every_operator_and_function_against_numpy():
    df = frame()
    x, w, xp = (df[c].to_numpy(dtype=float) for c in ("x", "w", "xpos"))
    for text, expected in [("I(x * w)", x * w), ("I(x + w)", x + w), ("I(x - w)", x - w), ("I(x / w)", x / w), ("I(xpos^0.5)", xp ** 0.5),
                           ("I(xpos^w)", xp ** w), ("I(2^x)", 2.0 ** x), ("I(-x)", -x), ("I(x > 0)", (x > 0) * 1.0), ("I(x < w)", (x < w) * 1.0),
                           ("I(x >= 0.2)", (x >= 0.2) * 1.0), ("I(x <= 0.2)", (x <= 0.2) * 1.0), ("I(o == 2)", (df.o == 2) * 1.0),
                           ("I(o != 2)", (df.o != 2) * 1.0), ("log(xpos + 1)", np.log(xp + 1)), ("exp(-x)", np.exp(-x)),
                           ("sqrt(abs(x))", np.sqrt(np.abs(x))), ("abs(x)", np.abs(x)), ("I((x > 0) * w)", (x > 0) * w),
                           ("log(log(xpos + 2))", np.log(np.log(xp + 2)))]:
        tree = ex.read_term(text)[2][0]
        assert np.array_equal(ex.evaluate(tree, df, term=text), np.asarray(expected, dtype=float)), text


def test_evaluation_refuses_columns_and_values_outside_the_domain():
    df = frame()
    for text, data, message in [
        ("I(x * w)", df.drop(columns="w"), "column 'w' of the term 'I(x * w)' is not in the data"),
        ("I(x * w)", df.assign(w=df.w.astype(str)), "column 'w' has dtype str; the term 'I(x * w)' needs a numeric column"),
        ("I(x * w)", df.assign(w=df.w > 1), "column 'w' has dtype bool; the term 'I(x * w)' needs a numeric column"),
        ("I(x * w)", df.assign(w=df.w.astype(complex)), "is complex-valued"),
        ("I(x * w)", df.assign(w=df.w.where(df.index != 2)), "column 'w' of the term 'I(x * w)' has missing or non-finite values"),
        ("log(x + 1)", df.assign(x=df.x.where(df.index != 3, -1.0)), "the term 'log(x + 1)' is not finite at row 3 (x = -1.0)"),
        ("I(x / w)", df.assign(w=df.w.where(df.index != 4, 0.0)), "the term 'I(x / w)' is not finite at row 4 (x = "),
        ("I(x^0.5)", df.assign(x=-df.x.abs() - 1), "the term 'I(x^0.5)' is not finite at row 0"),
    ]:
        with pytest.raises(FormulaError, match=re.escape(message)):
            ex.evaluate(ex.read_term(text)[2][0], data, term=text)


# -- 3. the fitted transforms ------------------------------------------------------------------------------------------------------
def test_scale_keeps_the_fitted_centre_and_scale():
    small = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    e = ex.Expression("scale(x)")
    assert e.form == "scale" and e.needs_fit and e.columns == ("x",) and e.suffixes() == ("",)
    centre, scale = e.estimate(small)
    assert centre == 2.5 and scale == pytest.approx(math.sqrt(5 / 3), rel=1e-15)
    fitted = ex.Expression("scale(x)", (centre, scale))
    assert np.allclose(fitted.values(small), (small.x - 2.5) / math.sqrt(5 / 3), rtol=0, atol=1e-15)
    new = pd.DataFrame({"x": [10.0, 2.5]})                  # new rows keep the fitted centre and scale (brms's predvars)
    assert np.allclose(fitted.values(new), [7.5 / math.sqrt(5 / 3), 0.0], rtol=0, atol=1e-15) and fitted.frame_value(new).shape == (2, 1)
    df = frame()
    c, s = ex.Expression("scale(w)").estimate(df)
    assert c == pytest.approx(df.w.mean(), rel=1e-15) and s == pytest.approx(df.w.std(ddof=1), rel=1e-14)
    with pytest.raises(FormulaError, match=re.escape("the term 'scale(x)' is constant in the fitted rows")):
        e.estimate(pd.DataFrame({"x": [1.0, 1.0, 1.0]}))
    with pytest.raises(FormulaError, match=re.escape("the term 'scale(x)' is constant in the fitted rows")):
        e.estimate(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(FormulaError, match=re.escape("the term 'scale(x)' needs its fitted parameters, known once the model has seen its data")):
        e.values(small)
    assert e.row_expression() == "(x_i - centre) / scale"
    assert fitted.row_expression() == "(x_i - 2.5) / 1.29099" and ex.Expression("scale(x)", (-0.5, 2.0)).row_expression() == "(x_i + 0.5) / 2"


def test_orthogonal_poly_is_r_construction_and_keeps_its_coefficients():
    small = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    fitted = ex.Expression("poly(x, 2)").estimate(small)          # hand-computed: (-1, 0, 1)/sqrt(2) and (1, -2, 1)/sqrt(6)
    assert fitted == pytest.approx((1.0, 1.0, 3.0, 2.0, 2.0 / 3.0), rel=1e-15)
    basis = ex.Expression("poly(x, 2)", fitted).basis(small)
    assert np.allclose(basis, np.column_stack([[-1, 0, 1], [1, -2, 1]]) / np.sqrt([2.0, 6.0]), rtol=0, atol=1e-15)
    df = frame()
    x = df.x.to_numpy(dtype=float)
    for degree in (1, 2, 3, 4):
        e = ex.Expression(f"poly(x, {degree})")
        f = e.estimate(df)
        assert len(f) == 2 * degree + 1
        b = ex.Expression(e.term, f).basis(df)
        assert np.allclose(b.T @ b, np.eye(degree), atol=1e-12) and np.allclose(b.sum(0), 0, atol=1e-12)   # orthonormal, orthogonal to 1
        assert np.allclose(b, poly_reference(x, degree, x), rtol=0, atol=1e-12)
        new = np.array([-3.0, 0.1, 2.7])                      # new rows: the fitted polynomials continued, not refitted
        assert np.allclose(ex.Expression(e.term, f).basis(pd.DataFrame({"x": new})), poly_reference(x, degree, new), rtol=0, atol=1e-11)
    two = ex.Expression("poly(x, 2)", ex.Expression("poly(x, 2)").estimate(df), "2")
    assert two.column == "poly(x, 2)2" and two.row_expression() == "P_2(x_i)" and ex.Expression("poly(x, 2)").row_expression() == "P_1(x_i) .. P_2(x_i)"
    with pytest.raises(FormulaError, match=re.escape("'poly(x, 2)': the degree 2 is not below the number of distinct values of 'x' in the fitted "
                                                    "rows (2); R refuses it too")):
        ex.Expression("poly(x, 2)").estimate(pd.DataFrame({"x": [0.0, 1.0, 0.0, 1.0]}))


def test_raw_poly_is_the_powers():
    df = frame()
    e = ex.Expression("poly(x, 3, raw = TRUE)")
    assert e.raw and not e.needs_fit and e.estimate(df) == () and e.suffixes() == ("1", "2", "3")
    x = df.x.to_numpy(dtype=float)
    assert np.allclose(e.basis(df), np.column_stack([x, x ** 2, x ** 3]), rtol=1e-15, atol=0)
    assert ex.Expression("poly(x, 3, raw = T)").raw and not ex.Expression("poly(x, 3, raw = FALSE)").raw
    assert ex.Expression(e.term, (), "2").row_expression() == "x_i^2" and e.row_expression() == "x_i^1 .. x_i^3"
    with pytest.raises(FormulaError, match=re.escape("the term 'poly(x, 2, raw = TRUE)' is not finite at row 1")):
        ex.Expression("poly(x, 2, raw = TRUE)").basis(pd.DataFrame({"x": [1.0, 1e200]}))


def test_factor_levels_sort_as_numbers_and_new_rows_use_them():
    df = pd.DataFrame({"o": [3, 1, 2, 10, 1, 3]})
    e = ex.Expression("factor(o)")
    levels = e.estimate(df)
    assert levels == ("1", "2", "3", "10")                    # numeric order, not text order ('10' before '2')
    cols = ex.expand("factor(o)", levels)
    assert [c.column for c in cols] == ["factor(o)2", "factor(o)3", "factor(o)10"] and [c.brms_name for c in cols] == ["factoro2", "factoro3", "factoro10"]
    assert np.array_equal(ex.Expression("factor(o)", levels).basis(df), np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0]], float))
    new = pd.DataFrame({"o": [2.0, 2.0, 3.0]})               # rows lacking levels 1 and 10: coded through the fitted levels (D181)
    assert np.array_equal(ex.Expression("factor(o)", levels).basis(new), [[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(FormulaError, match=re.escape("newdata column 'o' has values ['4', '12'] that 'factor(o)' was not fitted with (levels "
                                                    "['1', '2', '3', '10']); its columns are the fitted levels' indicators")):
        ex.Expression("factor(o)", levels).basis(pd.DataFrame({"o": [12, 4, 1]}), where="newdata column")
    with pytest.raises(FormulaError, match=re.escape("'factor(o)' has a single level ('2') in the fitted rows")):
        e.estimate(pd.DataFrame({"o": [2, 2]}))
    with pytest.raises(ValueError, match=re.escape("'factor(o)' has no fitted levels")):
        e.suffixes()
    with pytest.raises(FormulaError, match=re.escape("column 'o' of the term 'factor(o)' is not in the data")):
        e.estimate(pd.DataFrame({"x": [1, 2]}))
    for data, message in [(pd.DataFrame({"o": ["a", "b"]}), "column 'o' has dtype str; 'factor(o)' is offered for a column of whole numbers"),
                          (pd.DataFrame({"o": [True, False]}), "column 'o' has dtype bool"),
                          (pd.DataFrame({"o": [1.0, np.nan]}), "column 'o' of the term 'factor(o)' has missing values"),
                          (pd.DataFrame({"o": [1.0, 2.5]}), "holds values that are not whole numbers; factor() is offered for whole numbers here")]:
        with pytest.raises(FormulaError, match=re.escape(message)):
            e.estimate(data)
    assert ex.Expression("factor(o)", levels, "3").row_expression() == "1[o_i = 3]" and e.row_expression() == "indicators of the levels of o_i"


def test_a_comparison_is_a_logical_coded_true():
    df = frame()
    e = ex.Expression("I(x > 0)")
    assert e.form == "logical" and e.suffixes() == ("TRUE",) and e.estimate(df) == ()
    (col,) = ex.expand("I(x > 0)", ())
    assert col.column == "I(x > 0)TRUE" and col.brms_name == "Ix>0TRUE" and col.row_expression() == "1[x_i > 0]"
    assert np.array_equal(col.values(df), (df.x > 0).to_numpy(dtype=float))
    assert np.array_equal(col.values(pd.DataFrame({"x": [1.0, 2.0]})), [1.0, 1.0])     # new rows all TRUE are coded 1, not refused
    for data, value in [(df.assign(x=df.x.abs() + 1), "TRUE"), (df.assign(x=-df.x.abs() - 1), "FALSE")]:
        with pytest.raises(FormulaError, match=re.escape(f"the comparison 'I(x > 0)' is {value} on every fitted row: R codes a logical as a "
                                                        "factor, and one of a single level has no contrast")):
            e.estimate(data)
    assert ex.Expression("I(x * w)").estimate(df) == () and ex.Expression("I(x * w)").frame_value(df).shape == (len(df),)


# -- 4. the term model ---------------------------------------------------------------------------------------------------------------
def test_term_model_before_and_after_the_data():
    df = frame()
    s0 = parse_formula(FORMULA)
    assert s0.columns == ("x", "I(x * w)", "log(xpos + 1)", "scale(w)", "poly(x, 2)", "poly(w, 2, raw = TRUE)", "factor(o)", "I(x > 0)")
    assert [t.basis for t in s0.expression_terms] == list(s0.columns[1:]) and all(not t.fitted for t in s0.pop_terms)
    assert s0.pop_columns == ("x", "w", "xpos", "o") and s0.offset == Offset(parts=("2 * off", "log(xpos)"))
    s = parse_formula(FORMULA, df)
    assert s.columns == COLUMNS and s.terms == COLUMNS and s.predictors == ("x",) and s.transform_terms == s.pop_terms[1:]
    by = {t.name: t for t in s.pop_terms}
    assert by["poly(x, 2)1"] == PopTerm("poly(x, 2)1", "transform", ("poly(x, 2)1",), basis="poly(x, 2)", fitted=by["poly(x, 2)1"].fitted)
    assert by["poly(x, 2)1"].fitted == ex.Expression("poly(x, 2)").estimate(df) and by["factor(o)3"].fitted == ("1", "2", "3")
    assert by["scale(w)"].fitted == ex.Expression("scale(w)").estimate(df) and by["I(x * w)"].fitted == () and by["I(x > 0)TRUE"].basis == "I(x > 0)"
    assert by["I(x * w)"].operands == ("x", "w") and by["factor(o)2"].operands == ("o",) and by["poly(x, 2)2"].transform.suffix == "2"
    assert s.bambi_formula == ("y ~ x + `I(x * w)` + `log(xpos + 1)` + `scale(w)` + `poly(x, 2)1` + `poly(x, 2)2` + `poly(w, 2, raw = TRUE)1` + "
                               "`poly(w, 2, raw = TRUE)2` + `factor(o)2` + `factor(o)3` + `I(x > 0)TRUE` + offset(`2 * off + log(xpos)`) + (1 | g)")
    assert s.canonical_formula == ("y ~ x + I(x * w) + log(xpos + 1) + scale(w) + poly(x, 2) + poly(w, 2, raw = TRUE) + factor(o) + I(x > 0) + "
                                   "offset(2 * off) + offset(log(xpos)) + (1 | g)")
    for replica in (ModelSpec(**asdict(s)),):                  # asdict hands the terms back as mappings, the fitted parameters kept
        assert replica == s and replica.pop_terms == s.pop_terms
    assert fit_expressions(s, df) is s and fit_expressions(s0, df).columns == COLUMNS and fit_expressions(parse_formula("y ~ x"), df).columns == ("x",)
    # an A05 form keeps its Transform, whatever the spelling
    assert isinstance(expression_term("I(x**2)"), Transform) and isinstance(expression_term("I(x * w)"), ex.Expression)
    assert is_expression_term("poly(x, 2)") and not is_expression_term("sin(x)") and pop_term("I(x * w)", "transform").basis == "I(x * w)"
    assert pop_term("log(x)", "transform").basis is None
    # a duplicated term in another spelling, and a term next to its raw column
    with pytest.raises(FormulaError, match=re.escape("'I(x * w)': duplicated predictor ('I(x*w)' spells 'I(x * w)' again)")):
        parse_formula("y ~ I(x * w) + I(x*w)")
    assert parse_formula("y ~ w + poly(w, 2)", df).columns == ("w", "poly(w, 2)1", "poly(w, 2)2")


def test_pop_term_validation():
    with pytest.raises(ValueError, match=re.escape("'poly(x, 2)1' is not a design column of the in-formula transform 'poly(x,2)'")):
        PopTerm("poly(x, 2)1", "transform", ("poly(x, 2)1",), basis="poly(x,2)")
    with pytest.raises(ValueError, match="is not a design column of the in-formula transform 'sin\\(x\\)'"):
        PopTerm("sin(x)", "transform", ("sin(x)",), basis="sin(x)")
    with pytest.raises(ValueError, match="is not a design column"):
        PopTerm("scale(x)", "numeric", ("scale(x)",), basis="scale(x)")
    with pytest.raises(ValueError, match="is not a design column"):
        PopTerm("w1", "transform", ("w1",), basis="scale(w)")
    with pytest.raises(ValueError, match=re.escape("'sin(x)' is not a supported in-formula transform")):
        PopTerm("sin(x)", "transform", ("sin(x)",))


@pytest.mark.parametrize("formula,family,kwargs,message", [
    ("y ~ sin(x)", "gaussian", {}, "'sin(x)': the function sin() is not supported; the in-formula transforms supported are I(), log()"),
    ("y ~ ns(x, 3)", "gaussian", {}, "'ns(x, 3)': the spline basis ns() is not supported"),
    ("y ~ splines::bs(x)", "gaussian", {}, "R's namespace operator '::'"),
    ("y ~ I(y * x)", "gaussian", {}, "'I(y * x)': the response 'y' cannot be transformed into a predictor"),
    ("k | trials(nn) ~ I(nn * x)", "binomial", {}, "'I(nn * x)': the trials column 'nn' cannot be transformed into a predictor"),
    ("y ~ I(x * g) + (1 | g)", "gaussian", {}, "'g': the grouping factor cannot also be a predictor or the response"),
    ("y ~ scale(z)", "gaussian", {"categorical": {"z": "p"}}, "'scale(z)': 'z' is declared categorical; an in-formula transform applies to a numeric column"),
    ("y ~ x + scale(w) + x:scale(w)", "gaussian", {}, "'x:scale(w)': an interaction with a transformed term ('scale(w)') is not supported"),
    ("y ~ x + x:sin(w)", "gaussian", {}, "'sin(w)': the in-formula transforms supported are"),
    ("y ~ x + (1 + scale(x) | g)", "gaussian", {}, "function calls and transformations inside a group term are not supported"),
    ("y ~ 0 + factor(o)", "gaussian", {}, "'factor(o)' in a formula without an intercept is not supported: R codes the first factor (a logical "
                                          "included) of a '0 +' formula with one indicator per level"),
    ("y ~ x + I(x > 0) - 1", "gaussian", {}, "'I(x > 0)' in a formula without an intercept is not supported"),
    (["y ~ x", "sigma ~ scale(w)"], "gaussian", {}, "'sigma ~ scale(w)': the in-formula transform 'scale(w)' in a distributional parameter's "
                                                     "formula is not supported"),
    ("cnt ~ x + offset(cnt * 2)", "poisson", {}, "'offset(cnt * 2)': the response 'cnt' cannot be the offset"),
    ("k | trials(nn) ~ x + offset(nn / 10)", "binomial", {}, "'offset(nn / 10)': the trials column 'nn' cannot be the offset"),
    ("cnt ~ x + offset(off) + offset(log(g)) + (1 | g)", "poisson", {}, "the grouping factor 'g' cannot be the offset"),
    ("cnt ~ x + z + offset(off) + offset(z)", "poisson", {"categorical": {"z": "p"}}, "'z' is declared categorical; an offset is a known numeric column"),
    ("cnt ~ x + offset(x > 0)", "poisson", {}, "'offset(x > 0)': a comparison inside offset() is not supported; offset() takes an expression"),
    ("cnt ~ x + offset(scale(x))", "poisson", {}, "'offset(scale(x))': scale() inside offset() is not supported"),
    ("cnt ~ x + offset(2 + 1)", "poisson", {}, "'offset(2 + 1)': the offset reads no column"),
    ("cnt ~ x + offset(sin(x))", "poisson", {}, "'offset(sin(x))': the function sin() is not supported; offset() takes an expression"),
    ("cnt ~ x + offset(2*off) + offset(2 * off)", "poisson", {}, "'offset(2 * off)': the offset is written twice ('offset(2*off)' and "
                                                                "'offset(2 * off)'); R reads it once, so write it once"),
])
def test_formula_refusals(formula, family, kwargs, message):
    with pytest.raises(FormulaError, match=re.escape(message)):
        parse_formula(formula, family=family, **kwargs)


def test_refused_in_the_formulas_no_case_verifies():
    df = frame()
    from autobayes.multinomial import parse_multinomial
    from autobayes.multivariate import parse_multivariate
    from autobayes.nonlinear import parse_nonlinear

    with pytest.raises(FormulaError, match=re.escape("the in-formula transform 'scale(w)' in a non-linear parameter's formula is not supported")):
        parse_nonlinear(["y ~ a * exp(b * x)", "a ~ 1", "b ~ scale(w)"], None, "gaussian")
    with pytest.raises(FormulaError, match=re.escape("the in-formula transform 'poly(x, 2)' is not supported for the multinomial families")):
        parse_multinomial("cbind(k, nn) | trials(t) ~ poly(x, 2)", None)
    sub = df.assign(s1=(df.index % 2 == 0), s2=True)
    with pytest.raises(FormulaError, match=re.escape("'scale(x)' with subset(s1) is not supported: its fitted parameters (centre and scale, "
                                                    "polynomial coefficients, levels) would be the subset's")):
        parse_multivariate(["y | subset(s1) ~ scale(x)", "y2 | subset(s2) ~ x"], sub, "gaussian", rescor=False)
    ok = parse_multivariate(["y | subset(s1) ~ I(x * w) + offset(2 * off)", "y2 | subset(s2) ~ x"], sub, "gaussian", rescor=False)
    assert ok.specs["y"].columns == ("I(x * w)",) and ok.specs["y"].offset == Offset(parts=("2 * off",))
    mi = parse_multivariate(["y ~ mi(y2) + I(x * w) + offset(2 * off)", "y2 | mi() ~ x"], df, "gaussian", rescor=False)
    assert mi.specs["y"].offset.columns == ("off",)
    from autobayes.grouped_thresholds import check_thres          # the offset's columns are read next to thres(gr = z)
    thres = parse_formula("k | thres(gr = z) ~ x + offset(2 * off)", family="cumulative")
    check_thres(thres, "k | thres(gr = z) ~ x + offset(2 * off)")
    with pytest.raises(FormulaError, match=re.escape("'z' is the grouping column of 'thres(gr = z)' and cannot also be")):
        check_thres(parse_formula("k | thres(gr = z) ~ x + offset(I(z))", family="cumulative"), "f")


def test_the_multivariate_formula_reads_the_offsets_columns():
    from autobayes.multivariate import parse_multivariate

    s = parse_multivariate("mvbind(y, y2) ~ scale(x) + offset(2 * off)", frame(), "gaussian", rescor=True)
    assert s.specs["y"].columns == s.specs["y2"].columns == ("scale(x)",) and s.specs["y2"].offset.columns == ("off",)
    with pytest.raises(FormulaError, match=re.escape("the response(s) ['y2'] cannot also be predictors, grouping factors or offsets")):
        parse_multivariate("mvbind(y, y2) ~ x + offset(y2 / 2)", frame(), "gaussian", rescor=True)


# -- 5. offsets ------------------------------------------------------------------------------------------------------------------------
def test_offsets_add_up():
    df = frame()
    one = parse_formula("cnt ~ x + offset(log(xpos))", family="poisson")
    assert one.offset == Offset("xpos", log=True) and one.offset.columns == ("xpos",) and one.offset.bambi_term == "offset(log(xpos))"
    assert one.offset.row_expression() == "log(xpos_i)" and Offset("off").row_expression() == "off_i" and Offset("off").bambi_name == "offset(off)"
    two = parse_formula("cnt ~ x + offset(off) + offset(log(xpos))", df, family="poisson")
    o = two.offset
    assert o == Offset(parts=("off", "log(xpos)")) and o.columns == ("off", "xpos") and o.expression == "off + log(xpos)"
    assert o.term == "offset(off) + offset(log(xpos))" and o.bambi_term == "offset(`off + log(xpos)`)" and o.bambi_name == "offset(off + log(xpos))"
    assert o.row_expression() == "off_i + log(xpos_i)" and np.array_equal(o.values(df), df.off.to_numpy() + np.log(df.xpos.to_numpy()))
    expr = parse_formula("cnt ~ x + offset(2*off - x / 4)", df, family="poisson").offset
    assert expr == Offset(parts=("2 * off - x / 4",)) and np.array_equal(expr.values(df), 2 * df.off.to_numpy() - df.x.to_numpy() / 4)
    assert offset_expression("exp( xpos )", "offset(exp( xpos ))") == "exp(xpos)"
    with pytest.raises(FormulaError, match=re.escape("newdata column 'off' of the term 'offset(2 * off)' is not in the data")):
        Offset(parts=("2 * off",)).values(df.drop(columns="off"), where="newdata column")
    for bad in (dict(column="x", parts=("x",)), dict(), dict(parts=("x",), log=True)):
        with pytest.raises(ValueError, match="an Offset is one column"):
            Offset(**bad)


def test_offsets_in_the_model_and_its_predictions():
    df = frame()
    pri = ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1), sd=ab.HalfNormal(1))
    m = ab.poisson("cnt ~ x + offset(log(xpos)) + offset(2 * off) + (1 | g)", df, pri)
    o = np.log(df.xpos.to_numpy()) + 2 * df.off.to_numpy()
    assert np.array_equal(m.design.offset, o) and np.array_equal(m.data["log(xpos) + 2 * off"].to_numpy(), o)
    assert list(m.bambi_model.components["mu"].offset_terms) == ["offset(log(xpos) + 2 * off)"]
    text = m.describe()
    assert "  offset    : log(xpos) + 2 * off, added to the linear predictor as written" in text
    assert "  offset_i = log(xpos_i) + 2 * off_i: the offset term" in text
    rng = np.random.default_rng(4)
    post = xr.Dataset({"Intercept": (("chain", "draw"), rng.normal(0.3, 0.1, (1, 3))), "x": (("chain", "draw"), rng.normal(0.2, 0.1, (1, 3))),
                       "1|g_sigma": (("chain", "draw"), np.full((1, 3), 0.5)), "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.3, (1, 3, 4)))},
                      coords={"chain": [0], "draw": np.arange(3), "g__factor_dim": list(LEVELS)})
    idata = az.InferenceData(posterior=post)
    new = pd.DataFrame({"x": [0.5, -1.0], "xpos": [2.0, 0.5], "off": [0.1, -0.3], "g": ["g1", "g3"]})
    eta = (post["Intercept"].to_numpy()[..., None] + post["x"].to_numpy()[..., None] * new.x.to_numpy() + np.log(new.xpos.to_numpy())
           + 2 * new.off.to_numpy() + post["1|g"].to_numpy()[..., [1, 3]])
    assert np.allclose(m.predict(idata, newdata=new, kind="linear")["eta"].to_numpy(), eta, atol=1e-12)
    assert np.allclose(m.predict(idata, newdata=new, kind="mean")["mu"].to_numpy(), np.exp(eta), rtol=1e-10)   # Bambi reads the offset column
    with pytest.raises(FormulaError, match=re.escape("newdata lacks columns ['off'] (the offset term 'offset(log(xpos)) + offset(2 * off)' "
                                                    "needs 'off' on every row to predict)")):
        m.predict(idata, newdata=new.drop(columns="off"), kind="linear")
    loc = ab.default_priors("cnt ~ x + offset(log(xpos)) + offset(2 * off) + (1 | g)", df, family="poisson").intercept
    from autobayes.default_priors import location_scale
    assert loc.params["mu"] == location_scale(df.cnt.to_numpy(dtype=float), "log", list(o))[0]


# -- 6. the model: design, graph, priors, describe() and predictions -------------------------------------------------------------
def test_design_frame_and_names(model):
    m, d, df = model, model.design, frame()
    s = m.spec
    assert d.columns == COLUMNS and d.X_raw.shape == (32, 11)
    expected = {"x": df.x, "I(x * w)": df.x * df.w, "log(xpos + 1)": np.log(df.xpos + 1), "scale(w)": (df.w - df.w.mean()) / df.w.std(ddof=1),
                "poly(w, 2, raw = TRUE)1": df.w, "poly(w, 2, raw = TRUE)2": df.w ** 2, "factor(o)2": df.o == 2, "factor(o)3": df.o == 3,
                "I(x > 0)TRUE": df.x > 0}
    for name, values in expected.items():
        assert np.allclose(d.X_raw[:, COLUMNS.index(name)], np.asarray(values, dtype=float), rtol=0, atol=1e-14), name
    ref = poly_reference(df.x.to_numpy(dtype=float), 2, df.x.to_numpy(dtype=float))
    assert np.allclose(d.X_raw[:, 4:6], ref, rtol=0, atol=1e-12)
    for name in COLUMNS[1:]:                                    # the fitted frame carries every column Bambi reads, by name
        assert np.array_equal(m.data[name].to_numpy(), d.X_raw[:, COLUMNS.index(name)]), name
    assert np.array_equal(np.concatenate([pop_term_design(t, m.data) for t in s.pop_terms], axis=1), d.X_raw)
    assert np.array_equal(d.offset, 2 * df.off.to_numpy() + np.log(df.xpos.to_numpy()))
    assert m.coefficients == tuple((c, c, None) for c in COLUMNS)
    from autobayes.default_priors import brms_coefficient_names
    assert list(brms_coefficient_names(s).values()) == ["x", "IxMUw", "logxposP1", "scalew", "polyx21", "polyx22", "polyw2rawEQTRUE1",
                                                          "polyw2rawEQTRUE2", "factoro2", "factoro3", "Ix>0TRUE"]


def test_graph_linear_predictor_at_a_point(model):
    m, d = model, model.design
    pmm = m.pymc_model
    fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[m.names["linpred"]]]), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore")
    b = np.linspace(-0.5, 0.5, len(COLUMNS))
    nat = dict(intercept=0.3, b=b, sd=0.7, z=np.array([-0.4, 0.2, 0.9, -0.1]), sigma=0.9)
    pt = m.value_point(**nat)
    eta = nat["intercept"] + d.X_model @ b + d.offset + d.group_designs["g"].Z @ (nat["sd"] * nat["z"])
    assert np.allclose(np.asarray(fn(pt)[0]).ravel(), eta, atol=1e-12)


def synthetic(names, draws=3, seed=9):
    rng = np.random.default_rng(seed)
    var = {n: (("chain", "draw"), rng.normal(0, 0.3, (1, draws))) for n in names}
    var.update({"Intercept": (("chain", "draw"), rng.normal(0.2, 0.1, (1, draws))), "1|g_sigma": (("chain", "draw"), np.full((1, draws), 0.5)),
                "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.5, (1, draws, 4))), "sigma": (("chain", "draw"), np.full((1, draws), 0.8))})
    return az.InferenceData(posterior=xr.Dataset(var, coords={"chain": [0], "draw": np.arange(draws), "g__factor_dim": list(LEVELS)}))


def test_predictions_on_new_rows_use_the_fitted_transforms(model):
    m, df = model, frame()
    idata = synthetic(COLUMNS)
    P = idata.posterior
    new = pd.DataFrame({"x": [2.5, -0.3, 0.7], "w": [0.2, 1.9, 1.0], "xpos": [0.5, 2.0, 1.0], "o": [2, 2, 3], "off": [0.1, 0.0, -0.2],
                        "g": ["g1", "g0", "g3"]})                  # o lacks level 1: coded through the fitted levels (D181)
    x, w = new.x.to_numpy(dtype=float), new.w.to_numpy(dtype=float)
    cols = {"x": x, "I(x * w)": x * w, "log(xpos + 1)": np.log(new.xpos + 1), "scale(w)": (w - df.w.mean()) / df.w.std(ddof=1),
            "poly(w, 2, raw = TRUE)1": w, "poly(w, 2, raw = TRUE)2": w ** 2, "factor(o)2": (new.o == 2) * 1.0, "factor(o)3": (new.o == 3) * 1.0,
            "I(x > 0)TRUE": (x > 0) * 1.0}
    ref = poly_reference(df.x.to_numpy(dtype=float), 2, x)
    cols["poly(x, 2)1"], cols["poly(x, 2)2"] = ref[:, 0], ref[:, 1]
    eta = P["Intercept"].to_numpy()[..., None] + 2 * new.off.to_numpy() + np.log(new.xpos.to_numpy())
    for c in COLUMNS:
        eta = eta + P[c].to_numpy()[..., None] * np.asarray(cols[c], dtype=float)
    eta_g = eta + P["1|g"].to_numpy()[..., [1, 0, 3]]
    assert np.allclose(m.predict(idata, newdata=new, kind="linear")["eta"].to_numpy(), eta_g, atol=1e-11)
    assert np.allclose(m.predict(idata, newdata=new, kind="mean")["mu"].to_numpy(), eta_g, atol=1e-10)           # Bambi reads the columns
    assert np.allclose(m.predict(idata, newdata=new.drop(columns="g"), kind="linear", include_group_effects=False)["eta"].to_numpy(), eta, atol=1e-11)
    stale = new.assign(**{c: 99.0 for c in COLUMNS[1:]})          # columns named like the design's are recomputed, never read
    assert np.allclose(m.predict(idata, newdata=stale, kind="mean")["mu"].to_numpy(), eta_g, atol=1e-10)
    with pytest.raises(FormulaError, match=re.escape("newdata column 'o' has values ['7'] that 'factor(o)' was not fitted with")):
        m.predict(idata, newdata=new.assign(o=[2, 7, 3]), kind="linear")
    with pytest.raises(FormulaError, match=re.escape("newdata lacks columns ['o']")):
        m.predict(idata, newdata=new.drop(columns="o"), kind="linear")


def test_priors_by_term_and_by_column():
    df = frame()

    def build(b):
        return ab.gaussian("y ~ poly(w, 2) + factor(o) + I(x > 0) + scale(x)", df, ab.Priors(intercept=ab.Normal(0, 5), b=b, sigma=ab.HalfNormal(1)))
    m = build({"poly(w, 2)": {"1": ab.Normal(0, 1), "2": ab.Normal(0, 3)}, "factor(o)": ab.Normal(0, 2), "I(x > 0)": {"TRUE": ab.Normal(1, 1)},
               "scale(x)": ab.StudentT(3, 0, 1)})
    assert m.priors.coefficient_priors(m.spec) == {"poly(w, 2)1": ab.Normal(0, 1), "poly(w, 2)2": ab.Normal(0, 3), "factor(o)2": ab.Normal(0, 2),
                                                   "factor(o)3": ab.Normal(0, 2), "I(x > 0)TRUE": ab.Normal(1, 1), "scale(x)": ab.StudentT(3, 0, 1)}
    assert "b_poly(w, 2)2 ~ Normal(mu=0, sigma=3)" in m.describe()
    full = {"factor(o)": ab.Normal(0, 2), "I(x > 0)": ab.Normal(0, 1), "scale(x)": ab.Normal(0, 1)}
    for poly, message in [({"1": ab.Normal(0, 1), "3": ab.Normal(0, 3)}, "per-column priors for 'poly(w, 2)' must cover exactly its columns' "
                                                                            "suffixes ['1', '2'] (poly(w, 2)1, poly(w, 2)2); missing ['2'], unexpected ['3']"),
                          ({}, "per-column priors for 'poly(w, 2)' must be a non-empty mapping of Priors by its columns' suffixes"),
                          ({"1": "normal(0, 1)"}, "must be a non-empty mapping of Priors by its columns' suffixes, got str"),
                          ([ab.Normal(0, 1)], "the term 'poly(w, 2)' takes one prior or a mapping {column: prior} by its columns' suffixes, got list")]:
        with pytest.raises(ab.PriorError, match=re.escape(message)):
            build({"poly(w, 2)": poly, **full})
    with pytest.raises(ab.PriorError, match=re.escape("coefficient priors must cover exactly the predictor terms ['poly(w, 2)', 'factor(o)', "
                                                     "'I(x > 0)', 'scale(x)']; missing ['poly(w, 2)'], unexpected ['poly(w, 2)1']")):
        build({"poly(w, 2)1": ab.Normal(0, 1), **full})
    with pytest.raises(ab.PriorError, match=re.escape("predictor 'scale(x)' takes one prior, got dict")):
        build({"poly(w, 2)": ab.Normal(0, 1), "factor(o)": ab.Normal(0, 2), "I(x > 0)": ab.Normal(0, 1), "scale(x)": {"1": ab.Normal(0, 1)}})


def test_describe_says_what_each_column_is(model):
    text, df = model.describe(), frame()
    c, s = ex.Expression("scale(w)").estimate(df)
    lines = [
        "AutoBayes Gaussian model with in-formula transform(s) and offset",
        "  I(x * w)  : in-formula expression of 'x', 'w', x_i * w_i, range",
        f"  scale(w)  : (w_i - {c:.6g}) / {s:.6g}: 'w' centred and scaled by its mean and SD (divisor n - 1) over the fitted rows; new rows use the "
        "same centre and scale; centred like any column; brms names its coefficient scalew",
        "  poly(x, 2)1: P_1(x_i): the orthonormal polynomial of degree 1 in 'x' over the fitted rows (the three-term recurrence of ?poly); new rows "
        "use the fitted recurrence coefficients; centred like any column; brms names its coefficient polyx21",
        "  poly(w, 2, raw = TRUE)2: w_i^2: the raw power of 'w', recomputed on new rows; centred like any column; brms names its coefficient "
        "polyw2rawEQTRUE2",
        "  factor(o)3: 1[o_i = 3]: factor(o) with treatment contrasts against its lowest fitted level '1' (levels 1, 2, 3, in numeric order); "
        "new rows are coded through these levels, whichever of them they hold (divergence D181)",
        f"  I(x > 0)TRUE: the comparison I(x > 0), a logical coded as R codes one (FALSE the reference, this column TRUE): 1[x_i > 0], TRUE on "
        f"{int((df.x > 0).sum())} of 32 fitted rows",
        "(x_i * w_i - mean(I(x * w))) * b_I(x * w)", "(P_1(x_i) - mean(poly(x, 2)1)) * b_poly(x, 2)1",
        "  offset_i = 2 * off_i + log(xpos_i): the offset term",
        "  I(x * w): b_I(x * w) is the slope in I(x * w), not in x or w;",
        "  poly(x, 2)2: b_poly(x, 2)2 multiplies the orthonormal polynomial of degree 2 in x; the polynomial's slope in x combines every degree's coefficient.",
        "  factor(o)2: b_factor(o)2 is the difference on the mu scale between level 2 of factor(o) and its reference level '1'",
        "  I(x > 0)TRUE: b_I(x > 0)TRUE is the difference on the mu scale between rows where x > 0 holds and rows where it does not",
        "  poly(w, 2, raw = TRUE)1: b_poly(w, 2, raw = TRUE)1 is the slope in poly(w, 2, raw = TRUE)1, not in w;",
    ]
    for line in lines:
        assert line in text, line
    uncentred = ab.gaussian("y ~ scale(w) + (1 | g)", df, PRIORS, center_predictors=False).describe()
    assert f"mu_i = a + (w_i - {c:.6g}) / {s:.6g} * b_scale(w) + u[g_i]" in uncentred


def test_default_priors_name_the_columns_as_brms():
    from autobayes.families import get_family

    f = "y ~ I(x * w) + I(xpos^.5) + I(x > 0) + poly(w, 2) + factor(o) + (1 | g)"
    rows = ab.default_priors(f, frame()).table(parse_formula(f, frame()), get_family("gaussian"))
    coefs = [r["coef"] for r in rows if r["class"] == "b" and r["coef"]]
    assert coefs == sorted(["IxMUw", "IxposE0.5", "Ix>0TRUE", "polyw21", "polyw22", "factoro2", "factoro3"])


# -- 7. conditional_effects() -----------------------------------------------------------------------------------------------------
def test_conditional_effects_carry_the_model_frame_columns():
    df = frame()
    m = ab.gaussian("y ~ xpos + scale(x) + poly(w, 2) + abs(x) + (1 | g)", df, PRIORS)
    idata = synthetic(m.spec.columns, draws=5)
    res = ab.conditional_effects(m, idata, ["w"])
    table = res["w"]                                            # the terms' model-frame columns in formula order, then the raw columns
    assert list(table.columns) == ["w", "y", "xpos", "scale(x)", "poly(w, 2)", "abs(x)", "x", "g", "cond__", "effect1__", "estimate__", "se__",
                                   "lower__", "upper__"]
    held = {"x": math.fsum(df.x) / len(df), "w": math.fsum(df.w) / len(df)}
    assert np.allclose(np.asarray(table["scale(x)"].tolist()), (held["x"] - df.x.mean()) / df.x.std(ddof=1), atol=1e-15)
    assert np.asarray(table["scale(x)"].tolist()).shape == (100, 1) and np.asarray(table["poly(w, 2)"].tolist()).shape == (100, 2)
    ref = poly_reference(df.w.to_numpy(dtype=float), 2, np.array([held["w"]]))
    assert np.allclose(np.asarray(table["poly(w, 2)"].tolist()), np.repeat(ref, 100, axis=0), atol=1e-12)
    assert np.allclose(table["abs(x)"].to_numpy(dtype=float), abs(held["x"])) and table["g"].isna().all()
    for formula, term in [("y ~ x + factor(o)", "factor(o)"), ("y ~ x + I(x > 0)", "I(x > 0)")]:
        m2 = ab.gaussian(formula, df, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1)))
        with pytest.raises(ValueError, match=re.escape(f"conditional_effects() of a model with the term {term!r} is not offered: no frozen case "
                                                      "checks how brms holds it on the grid")):
            ab.conditional_effects(m2, synthetic(m2.spec.columns), ["x"])


# -- 8. the shared paths the offsets' and transforms' columns pass through -------------------------------------------------------
def test_shared_paths_read_every_column():
    df = frame()
    assert parse_formula("y ~ log(xpos)").pop_terms[0].operands == ("xpos",)          # A05's Transform reads its one column
    from autobayes.model import _linear_terms
    assert _linear_terms(parse_formula("y ~ log(xpos) + scale(w)", df), True) == [
        "(log(xpos_i) - mean(log(xpos))) * b_log(xpos)", f"((w_i - {df.w.mean():.6g}) / {df.w.std(ddof=1):.6g} - mean(scale(w))) * b_scale(w)"]
    # a list of formulas with a declared column: the transforms' and the offset's columns are checked against the declaration
    lst = parse_formula(["y ~ I(x * w) + z + offset(2 * off)", "sigma ~ z"], df, categorical={"z": "p"})
    assert lst.columns == ("I(x * w)", "z[q]") and lst.offset.columns == ("off",)
    with pytest.raises(FormulaError, match=re.escape("'I(x * z)': 'z' is declared categorical; an in-formula transform applies to a numeric column")):
        parse_formula(["y ~ I(x * z) + z", "sigma ~ z"], categorical={"z": "p"})
    with pytest.raises(FormulaError, match=re.escape("'offset(2 * z)': 'z' is declared categorical; the response formula reads it as a number")):
        parse_formula(["y ~ x + z + offset(2 * z)", "sigma ~ x"], categorical={"z": "p"})
    # a cbind() response reads its columns apart from the right-hand side's, an offset's included (refused for the family itself)
    assert parse_formula("cbind(p1, p2, p3) ~ I(x * w)", family="dirichlet").columns == ("I(x * w)",)
    with pytest.raises(FormulaError, match=re.escape("an offset term is not offered for the dirichlet family")):
        parse_formula("cbind(p1, p2, p3) ~ x + offset(2 * off)", family="dirichlet")
    # cs() and ordered() read their column alone: an offset expression that reads it is refused
    with pytest.raises(FormulaError, match=re.escape("'x' is read by 'cs(x)' and by another term of the formula")):
        parse_formula("yo ~ cs(x) + offset(2 * x)", family="sratio")
    with pytest.raises(FormulaError, match=re.escape("'o' is read by 'ordered(o)' and by another term of the formula")):
        parse_formula("y ~ ordered(o) + offset(o / 2)")
    # an ordered factor's contrasts keep their factor's prior key next to an expression's columns
    s = parse_formula("y ~ ordered(o) + poly(w, 2)", df)
    pri = ab.Priors(intercept=ab.Normal(0, 5), b={"ordered(o)": ab.Normal(0, 1), "poly(w, 2)": ab.Normal(0, 2)}, sigma=ab.HalfNormal(1))
    assert pri.coefficient_priors(s) == {"ordered(o).L": ab.Normal(0, 1), "ordered(o).Q": ab.Normal(0, 1), "poly(w, 2)1": ab.Normal(0, 2),
                                         "poly(w, 2)2": ab.Normal(0, 2)}
