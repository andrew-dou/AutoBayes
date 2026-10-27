"""R's term algebra and its coding of a factor in an interaction (brms-parity G03; docs/term-algebra.md, divergences D179, D180).

1. The algebra: ``*``, ``:``, ``^``, ``-``, parentheses, repeated terms and R's term order, against R 4.5.2's ``terms()`` labels.
2. The coding of a factor whose margin is absent and brms's names and column order, against brms 2.23's ``make_standata()``
   column names on probe formulas (the columns themselves built here from those names, independently of the library).
3. The refusals, each by its message: what R reads differently from what it looks like (``x^2``, ``x:x``, ``0 + x + x:z``), and
   the combinations no frozen case verifies.
4. Models: Bambi's design, priors per column, warnings, ``describe()``, predictions on new rows, the fit record, brms's names,
   distributional formulas and the other model kinds.

Build-only; nothing samples. The numbers against brms are the oracle's (cases ``gauss_three_way_numeric``,
``gauss_three_way_factor``, ``gauss_power_expansion``, ``gauss_colon_only``, ``gauss_factor_slopes``,
``gauss_numeric_colon_factor_one_main``, ``bernoulli_three_way_varint``) and the syntax corpus's."""
import json
import re
from dataclasses import asdict, replace

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import autobayes as ab
from autobayes import term_algebra as ta
from autobayes.default_priors import brms_coefficient_names
from autobayes.formula import FormulaError, ModelSpec, PopTerm, parse_formula, pop_term_design
from autobayes.multinomial import parse_multinomial
from autobayes.multivariate import parse_multivariate
from autobayes.nonlinear import parse_nonlinear
from autobayes.priors import PriorError
from autobayes.term_algebra import Product, expand_rhs, factor_coding, r_variable_order
from tests.test_population_only import PANEL_STUB

NUMERIC = ("x", "w", "v")
LEVELS = {"a": ("a1", "a2", "a3"), "b": ("b1", "b2"), "c": ("c1", "c2"), "z": ("p", "q", "r")}
CATS = {"a": "a1", "b": "b1", "c": "c1", "z": "p"}


def frame(n=48, seed=3) -> pd.DataFrame:
    """Every cell of a, b, c and z observed (each block of twelve rows runs through the cells of a, b and c at one level of z)."""
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    return pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(0.3, 1.0, n), "w": rng.normal(-0.2, 1.0, n), "v": rng.normal(size=n),
                         "a": np.array(LEVELS["a"])[i % 3], "b": np.array(LEVELS["b"])[(i // 3) % 2],
                         "c": np.array(LEVELS["c"])[(i // 6) % 2], "z": np.array(LEVELS["z"])[(i // 12) % 3],
                         "g": np.array(["g0", "g1", "g2", "g3"])[i % 4], "sdx": 0.2 + rng.uniform(size=n), "o": i % 4 + 1,
                         "yb": rng.integers(0, 2, n), "label": np.array(["u", "t"])[i % 2]})


def declared(formula) -> dict:
    """The factors among the formula's terms after the algebra, declared with their first level as the reference (as R's)."""
    atoms = {a for piece, _ in expand_rhs(formula.split("~", 1)[1], formula) for a in piece.split(":")}
    return {k: v for k, v in CATS.items() if k in atoms}


def r_column(name: str, df: pd.DataFrame) -> np.ndarray:
    """A model-matrix column from R's name of it: the product of the numeric columns and factor-level indicators it names."""
    out = np.ones(len(df))
    for piece in name.split(":"):
        if piece in NUMERIC:
            out = out * df[piece].to_numpy(dtype=float)
            continue
        ((f, level),) = [(f, piece[len(f):]) for f in LEVELS if piece.startswith(f) and piece[len(f):] in LEVELS[f]]
        out = out * (df[f] == level).to_numpy(dtype=float)
    return out


def design(spec, df) -> np.ndarray:
    return np.column_stack([pop_term_design(t, df, spec.categoricals) for t in spec.pop_terms if not t.is_special])


PRIORS = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1))
POP = ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2), sigma=ab.HalfNormal(1))


# -- 1. the algebra -------------------------------------------------------------------------------------------------------------
# R 4.5.2's attr(terms(y ~ <rhs>), "term.labels") of each right-hand side (probed; docs/term-algebra.md)
R_TERMS = [
    ("(a+b):(c+d)", "a:c a:d b:c b:d"),
    ("(a+b)*(c+d)", "a b c d a:c a:d b:c b:d"),
    ("a*b*c*d", "a b c d a:b a:c b:c a:d b:d c:d a:b:c a:b:d a:c:d b:c:d a:b:c:d"),
    ("(a+b+c+d)^3", "a b c d a:b a:c a:d b:c b:d c:d a:b:c a:b:d a:c:d b:c:d"),
    ("(a+b+c)^4", "a b c a:b a:c b:c a:b:c"),
    ("b:c + a:b", "b:c b:a"),
    ("(c + a):b", "c:b a:b"),
    ("a:b:c + a:b + c", "c a:b a:b:c"),
    ("a*(b+c) - a:b", "a b c a:c"),
    ("(a+b)^2:c", "a:c b:c a:b:c"),
    ("a:(b*c)", "a:b a:c a:b:c"),
    ("a*b:c", "a b:c a:b:c"),
    ("a:b*c", "c a:b a:b:c"),
    ("(a*b)^2", "a b a:b"),
    ("(a:b + c)^2", "c a:b a:b:c"),
    ("x + z - z + z", "x z"),
    ("x - z + z", "x z"),
    ("(x + z - z):w", "x:w"),
    ("(x + w):(x + z)", "x x:z x:w w:z"),
    ("x*z + x", "x z x:z"),
    ("x:z + z + z:x + x", "z x x:z"),
    ("a + b + c - (b + c)", "a"),
    ("a*b - b - a", "a:b"),
    ("a*b - a*b", ""),
    ("(x + w + z)^2 - x:w", "x w z x:z w:z"),
    ("(a + b + c)^2 - (a + b)^2", "c a:c b:c"),
    ("x + w + x*w:z", "x w w:z x:w:z"),
    ("x:(w + v)", "x:w x:v"),
    ("a + (b)", "a b"),
    ("((a + b)):c", "a:c b:c"),
    ("x + x", "x"),
]


@pytest.mark.parametrize("rhs, labels", R_TERMS, ids=[r[0] for r in R_TERMS])
def test_the_algebra_gives_r_s_terms_in_r_s_order(rhs, labels):
    assert [p for p, _ in expand_rhs(rhs, rhs)] == labels.split()


def test_the_intercept_pieces_read_on_their_own_and_the_pieces_as_written():
    assert expand_rhs("-1 + x:z", "f") == [("0", "1"), ("x:z", "x:z")]                  # '- 1' is '0', as before
    assert expand_rhs("a*b*c - a:b - 1", "f")[-1] == ("a:b:c", "a*b*c")
    assert expand_rhs("x + (1 | g) + log(w) + log(w)", "f") == [("x", "x"), ("(1 | g)", "(1 | g)"), ("log(w)", "log(w)"),
                                                                ("log(w)", "log(w)")]     # a call written twice reaches the parser twice
    assert expand_rhs("log(x) * z", "f") == [("log(x)", "log(x) * z"), ("z", "log(x) * z"), ("log(x):z", "log(x) * z")]
    assert expand_rhs("(1 | g):x", "f") == [("(1 | g):x", "(1 | g):x")]                  # a group term inside the algebra is one variable
    assert expand_rhs("x - y", "f") == [("x", "x")]                                      # removing a term the formula lacks: R's no-op
    assert r_variable_order([("z",), ("x", "z"), ("w", "x")]) == {"z": 0, "x": 1, "w": 2}


@pytest.mark.parametrize("rhs, message", [
    ("x + + z", "empty term"),
    ("x - 0", "'- 0': '-' removes the intercept"),
    ("x - log(w)", "'- log\\(w\\)': '-' removes the intercept \\('- 1'\\) or population-level terms of columns"),
    ("x - (1 | g)", "'- \\(1 \\| g\\)': '-' removes"),
    ("(x + w)^1", "R's '\\^' takes a whole number of at least 2"),
    ("(x + w)^2.5", "R's '\\^' takes a whole number of at least 2"),
    ("(x + w)^(2)", "R's '\\^' takes a whole number of at least 2"),
    ("x^2", "R reads x\\^2 as x itself: the power of a formula term multiplies a sum of terms .* write I\\(x\\^2\\)"),
    ("(a:b)^2", "R reads \\(a:b\\)\\^2 as \\(a:b\\) itself"),
    ("x:x", "a term cannot interact with itself \\('x' on both sides of ':'\\); R reads x:x as x, not as a square"),
    ("x*x", "a term cannot interact with itself \\('x' on both sides of '\\*'\\)"),
    ("a:b:a", "a term cannot interact with itself \\('a' on both sides of ':'\\)"),
    ("(x y):z", "'y' is not read: R's term algebra takes columns"),
    ("x:z w", "'w' is not read"),
    ("x:2", "'2' is not a column name: R's term algebra takes columns, function calls"),
    ("(1 + x):z", "'1 \\+ x\\):z' is not a column name"),
    ("x:`z`", "backtick-quoted names are not supported"),
    ("(x + z", "unbalanced parentheses"),
    ("x + z)", "unbalanced parentheses"),
    ("(x + z))*w", "unbalanced parentheses"),
])
def test_the_algebra_refuses_what_r_reads_otherwise_or_not_at_all(rhs, message):
    with pytest.raises(FormulaError, match=message):
        expand_rhs(rhs, rhs)


def test_the_helpers_by_hand():
    with pytest.raises(FormulaError, match="unbalanced"):
        ta._closing("(x", 0)
    assert ta._signed_summands("-1 + x - z") == [("-", "1"), ("+", "x"), ("-", "z")]
    assert ta._is_algebra("x") and ta._is_algebra("x:z") and ta._is_algebra("(x + z)") and ta._is_algebra("(x)^2")
    assert not ta._is_algebra("log(x)") and not ta._is_algebra("(1 | g)") and not ta._is_algebra("(x) + (z)")
    a, b = frozenset({"a"}), frozenset({"b"})
    assert ta._union([a], [b, a]) == [a, b] and ta._cross([a, b], [a, b]) == [a, a | b, b]


# -- 2. the coding and brms's names ----------------------------------------------------------------------------------------
# brms 2.23's colnames(make_standata(<formula>, data)$X) without the intercept (probed; docs/term-algebra.md)
BRMS_X = [
    ("y ~ w:x + x", "x x:w"),
    ("y ~ b:a + a", "aa2 aa3 aa1:bb2 aa2:bb2 aa3:bb2"),
    ("y ~ z:x + x", "x x:zq x:zr"),
    ("y ~ a:x + b:x", "aa1:x aa2:x aa3:x x:bb2"),
    ("y ~ a:b + a:c", "aa1:bb1 aa2:bb1 aa3:bb1 aa1:bb2 aa2:bb2 aa3:bb2 aa1:cc2 aa2:cc2 aa3:cc2"),
    ("y ~ x * z - x", "zq zr zp:x zq:x zr:x"),
    ("y ~ (x + w + z)^2 - x:w", "x w zq zr x:zq x:zr w:zq w:zr"),
    ("y ~ x:w:v + w:x", "x:w x:w:v"),
    ("y ~ v:w:x + x + w", "x w x:w:v"),
    ("y ~ x + w:x:z", "x x:w:zp x:w:zq x:w:zr"),
    ("y ~ (x + z):w", "x:w w:zq w:zr"),
    ("y ~ 0 + x:z + z", "zp zq zr zp:x zq:x zr:x"),
    ("y ~ z:a + a", "aa2 aa3 aa1:zq aa2:zq aa3:zq aa1:zr aa2:zr aa3:zr"),
    ("y ~ x:z + z + x", "zq zr x zq:x zr:x"),
    ("y ~ x + z - z", "x"),
    # the last three are R 4.5.2's colnames(model.matrix()): brms's rebuilt formula lists their variables in the same order
    ("y ~ a:b:c", "aa1:bb1:cc1 aa2:bb1:cc1 aa3:bb1:cc1 aa1:bb2:cc1 aa2:bb2:cc1 aa3:bb2:cc1 aa1:bb1:cc2 aa2:bb1:cc2 aa3:bb1:cc2 "
                  "aa1:bb2:cc2 aa2:bb2:cc2 aa3:bb2:cc2"),
    ("y ~ a + a:b:c", "aa2 aa3 aa1:bb1:cc1 aa2:bb1:cc1 aa3:bb1:cc1 aa1:bb2:cc1 aa2:bb2:cc1 aa3:bb2:cc1 aa1:bb1:cc2 aa2:bb1:cc2 "
                      "aa3:bb1:cc2 aa1:bb2:cc2 aa2:bb2:cc2 aa3:bb2:cc2"),
    ("y ~ a*b*c - a:b:c", "aa2 aa3 bb2 cc2 aa2:bb2 aa3:bb2 aa2:cc2 aa3:cc2 bb2:cc2"),
]


@pytest.mark.parametrize("formula, names", BRMS_X, ids=[r[0] for r in BRMS_X])
def test_columns_are_brms_s_by_name_order_and_content(formula, names):
    df = frame()
    s = parse_formula(formula, df, "gaussian", declared(formula))
    brms = brms_coefficient_names(s)
    assert [brms[c] for c in s.columns] == names.split()
    X = design(s, df)
    for j, name in enumerate(names.split()):
        assert np.array_equal(X[:, j], r_column(name, df)), name


def test_the_coding_rule_by_hand():
    terms = [("a", "b"), ("a", "c")]                                         # a:c's margin a is contained in the earlier a:b
    assert factor_coding(terms, {"a", "b", "c"}, True, "f") == [frozenset({"a", "b"}), frozenset({"a"})]
    assert factor_coding([("z",), ("z", "x")], {"z"}, True, "f") == [frozenset(), frozenset({"z"})]
    assert factor_coding([("x",), ("x", "z")], {"z"}, True, "f") == [frozenset(), frozenset()]
    assert factor_coding([("x", "z")], {"z"}, False, "f") == [frozenset({"z"})]                 # 0 + x:z: z by indicators already
    assert factor_coding([("x",), ("z",)], {"z"}, False, "f") == [frozenset(), frozenset({"z"})]   # 0 + z
    assert factor_coding([("x",)], set(), False, "f") == [frozenset()]


def test_without_an_intercept_r_s_rank_deficient_coding_is_refused():
    with pytest.raises(FormulaError, match=re.escape("'y ~ 0 + x + x:z': without an intercept R and brms code the first factor of the "
                                                     "formula by an indicator of every level, here 'z' in 'x:z', although its margin 'x' "
                                                     "is a term")):
        parse_formula("y ~ 0 + x + x:z", None, "gaussian", {"z": "p"})
    s = parse_formula("y ~ 0 + x:z", frame(), "gaussian", {"z": "p"})                          # brms: x:zp, x:zq, x:zr
    assert s.columns == ("x:zp", "x:zq", "x:zr") and not s.intercept and not s.categoricals[0].full_rank
    s = parse_formula("y ~ 0 + z*x*a", frame(), "gaussian", {"z": "p", "a": "a1"})
    assert s.categoricals[0].full_rank and s.columns[:3] == ("z[p]", "z[q]", "z[r]") and "zq:x:aa2" in s.columns


def test_the_term_model_of_a_product_term():
    s = parse_formula("y ~ z + x:z", None, "gaussian", {"z": "p"})
    (t,) = [t for t in s.pop_terms if t.kind == "product"]
    assert t.name == "z:x" and t.columns == ("z:x",) and t.operands == ("z", "x") and t.bambi_name == "`z:x`" and not t.is_vector
    assert t.product == Product("z:x", ("z", "x"), full=("z",)) and not t.product.known and t.product.column == "z:x"
    with pytest.raises(ValueError, match="has no design column until its factors' levels are known"):
        t.product.values(frame())
    s = parse_formula("y ~ z + x:z", frame(), "gaussian", {"z": "p"})
    cols = [t for t in s.pop_terms if t.kind == "product"]
    assert [c.name for c in cols] == ["zp:x", "zq:x", "zr:x"] and [c.product.label for c in cols] == ["p", "q", "r"]
    assert cols[1].product.cell == ("q", None) and cols[1].product.factors == ("z",) and cols[1].product.known
    rebuilt = ModelSpec(**asdict(s))                                                          # asdict and back: the Product too
    assert rebuilt.pop_terms == s.pop_terms
    assert ta.product_columns(cols[0], s.categoricals) == tuple(cols)                        # a column re-expands to its term's
    numeric = parse_formula("y ~ x:w:v", None)
    assert numeric.pop_terms[0].product.known and numeric.pop_terms[0].product.label == ""
    assert ta.product_columns(numeric.pop_terms[0], ()) == numeric.pop_terms


@pytest.mark.parametrize("kwargs, message", [
    (dict(term="x", operands=("x",)), "is the interaction of its distinct operands"),
    (dict(term="x:x", operands=("x", "x")), "is the interaction of its distinct operands"),
    (dict(term="x:w", operands=("w", "x")), "is the interaction of its distinct operands"),
    (dict(term="x:z", operands=("x", "z"), full=("z",), contrasts=("z",)), "each factor is coded once"),
    (dict(term="x:z", operands=("x", "z"), full=("h",)), "each factor is coded once"),
    (dict(term="x:z", operands=("x", "z"), full=("z",), cell=("q",)), "a cell gives each factor a level"),
    (dict(term="x:z", operands=("x", "z"), full=("z",), cell=("x", "q")), "a cell gives each factor a level"),
    (dict(term="x:w", operands=("x", "w"), cell=(None, None)), "a cell gives each factor a level"),
])
def test_a_product_is_checked(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Product(**kwargs)


def test_a_product_term_is_checked():
    p = Product("x:z", ("x", "z"), full=("z",), cell=(None, "q"))
    assert PopTerm("x:zq", "product", ("x:zq",), product=p).operands == ("x", "z")
    for bad in (dict(name="x:z", kind="product", columns=("x:z",), product=p), dict(name="x:zq", kind="product", columns=("x:zq",)),
                dict(name="x:zq", kind="numeric", columns=("x:zq",), product=p),
                dict(name="x:zq", kind="product", columns=("x:zq",), levels=("q",), product=p)):
        with pytest.raises(ValueError, match="a product term"):
            PopTerm(**bad)


# -- 3. refusals -----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("formula, kwargs, message", [
    ("y ~ x + x:g + (1 | g)", {}, "'x:g': the grouping factor 'g' cannot enter a fixed interaction"),
    ("y ~ x:w:g + (1 | g)", {}, "'x:w:g': the grouping factor 'g' cannot enter a fixed interaction"),
    ("y ~ x + x:w:y", {}, "'x:w:y': the response or trials column cannot enter an interaction"),
    ("y ~ mo(o):x:w", {}, "'mo\\(o\\):x:w': an interaction with a monotonic term \\('mo\\(o\\)'\\) is not supported"),
    ("y ~ x * me(w, sdx)", {}, "'x \\* me\\(w, sdx\\)': an interaction with a measurement-error term"),
    ("y ~ x * mi(w)", {}, "an interaction with a missing-value term"),
    ("y ~ ordered(o):x:w", {}, "an interaction with an ordered factor \\('ordered\\(o\\)'\\) is not supported"),
    ("y ~ (x + w):log(v)", {}, "an interaction with a transformed term \\('log\\(v\\)'\\) is not supported"),
    ("y ~ x %in% z", {}, "R's operator %in% is not supported in a formula; nesting x %in% f is R's interaction x:f: write that"),
    ("y ~ x + (1 + x:z | g)", {}, "an interaction cannot be a varying slope"),
    ("y ~ me(x, sdx) + x:w", {}, "'x' is read by 'me\\(x, sdx\\)' and by another term of the formula"),
    ("y ~ ordered(o) + x:o", {}, "'o' is read by 'ordered\\(o\\)' and by another term of the formula"),
    ("y ~ x + x:g_h + (1 | g:h)", {}, "the derived grouping factor 'g:h' is handed to Bambi as the column 'g_h', which is also a column"),
])
def test_unverified_combinations_are_refused_by_name(formula, kwargs, message):
    with pytest.raises(FormulaError, match=message):
        parse_formula(formula, None, "gaussian", **kwargs)


def test_special_terms_keep_their_columns_when_interactions_are_present():
    assert parse_formula("y ~ me(xobs, sdx) + w:v").terms == ("me(xobs, sdx)", "w:v")
    assert parse_formula("y ~ cs(x) + w:v", None, "acat").terms == ("cs(x)", "w:v")
    with pytest.raises(FormulaError, match="'x' is read by 'cs\\(x\\)' and by another term"):
        parse_formula("y ~ cs(x) + x:w", None, "acat")
    s = parse_formula("y ~ ordered(o) + x:z", frame(), "gaussian", {"z": "p"})
    assert [c.name for c in s.categoricals] == ["o", "z"] and s.columns[-3:] == ("x:zp", "x:zq", "x:zr")


def test_data_refusals():
    df = frame()
    with pytest.raises(FormulaError, match="column 'label' of the interaction 'z:label' has dtype"):
        parse_formula("y ~ z + label:z", df, "gaussian", {"z": "p"})
    with pytest.raises(FormulaError, match=re.escape("interaction 'x:w:v' is constant (x * w * v takes a single value in the fitted rows)")):
        parse_formula("y ~ x:w:v", df.assign(v=1.0 / (df.x * df.w)))
    with pytest.raises(FormulaError, match=re.escape("columns not in data: ['q']")):
        parse_formula("y ~ x:q", df)
    ordered = df.assign(o=pd.Categorical(df.o, categories=[1, 2, 3, 4], ordered=True))
    with pytest.raises(FormulaError, match="'x:o': an interaction with the ordered factor 'o' is not supported"):
        parse_formula("y ~ x:o", ordered, "gaussian", {"o": 1})
    with pytest.raises(FormulaError, match="the column 'x:z`q' has a backtick in a level label"):
        parse_formula("y ~ x:z", df.assign(z=df.z.replace("q", "`q")), "gaussian", {"z": "p"})
    clash = df.assign(za=np.where(df.a == "a1", "b", "c"), z=np.where(df.z == "p", "ab", df.z))   # z level ab and za level b: zab
    with pytest.raises(FormulaError, match="the interaction column 'x:zab' of 'x:z' .* is also the name of another column or term"):
        parse_formula("y ~ x:z + x:za", clash, "gaussian", {"z": "q", "za": "c"})


def test_the_other_model_kinds_refuse_a_product_term():
    df = frame()
    with pytest.raises(FormulaError, match="the interaction 'x:w:v' .* is not supported in a non-linear parameter's formula"):
        parse_nonlinear(["y ~ a * exp(b * x)", "a ~ 1 + x:w:v", "b ~ 1"])
    with pytest.raises(FormulaError, match="the interaction 'x:w:v' .* is not supported in a category's formula"):
        parse_multinomial("cbind(m1, m2, m3) | trials(n) ~ x:w:v")
    with pytest.raises(FormulaError, match="the interaction 'x:w:z' .* is not supported in a multivariate model"):
        parse_multivariate("mvbind(y, v) ~ x*w*z", df.assign(z=df.x * 2), rescor=False)
    with pytest.raises(FormulaError, match="the interaction 'x:z' .* is not supported in a multivariate model"):
        parse_multivariate(["y ~ x:z", "w ~ x"], None, rescor=False)
    with pytest.raises(FormulaError, match="the interaction 'x:z' .* is not supported in a multivariate model"):
        parse_multivariate(["y ~ mi(v) + x:z", "v | mi() ~ x"], None, rescor=False)


# -- 4. models -------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def slopes():
    """``y ~ z + x:z + (1 | g)``: a slope for every level of z, the numeric's main effect absent."""
    return ab.gaussian("y ~ z + x:z + (1 | g)", frame(), PRIORS, categorical={"z": "p"})


def synthetic(draws=4, seed=8):
    rng = np.random.default_rng(seed)
    shape = (1, draws)
    var = {"Intercept": (("chain", "draw"), rng.normal(0.2, 0.1, shape)),
           "z": (("chain", "draw", "z_dim"), rng.normal(0, 0.5, shape + (2,))),
           **{c: (("chain", "draw"), rng.normal(0.3 * k, 0.1, shape)) for k, c in enumerate(("zp:x", "zq:x", "zr:x"))},
           "1|g_sigma": (("chain", "draw"), rng.uniform(0.3, 0.7, shape)), "1|g": (("chain", "draw", "g__factor_dim"), rng.normal(0, 0.5, shape + (4,))),
           "sigma": (("chain", "draw"), rng.uniform(0.3, 0.5, shape))}
    coords = {"chain": [0], "draw": np.arange(draws), "z_dim": ["q", "r"], "g__factor_dim": ["g0", "g1", "g2", "g3"]}
    return az.InferenceData(posterior=xr.Dataset(var, coords=coords))


def test_the_model_s_design_names_and_graph(slopes):
    m = slopes
    assert m.spec.columns == ("z[q]", "z[r]", "zp:x", "zq:x", "zr:x") and m.spec.rhs == "z + z:x + (1 | g)"
    df = frame()
    assert np.array_equal(m.design.X_raw[:, 2:], np.column_stack([r_column(c, df) for c in ("zp:x", "zq:x", "zr:x")]))
    assert list(m.bambi_model.components["mu"].common_terms) == ["z", "zp:x", "zq:x", "zr:x"]
    assert m.coefficients[2:] == (("zp:x", "zp:x", None), ("zq:x", "zq:x", None), ("zr:x", "zr:x", None))
    assert m.warnings == []


def test_predictions_on_new_rows_recompute_the_columns(slopes):
    m, idata = slopes, synthetic()
    P = idata.posterior
    new = pd.DataFrame({"x": [0.5, -1.0, 2.0], "z": ["p", "q", "r"], "g": ["g0", "g1", "g2"]})
    eta = m.predict(idata, newdata=new, kind="linear", include_group_effects=False)["eta"].to_numpy()
    a, bz = P["Intercept"].to_numpy(), P["z"].to_numpy()
    hand = np.stack([a + 0.5 * P["zp:x"].to_numpy(), a + bz[..., 0] - 1.0 * P["zq:x"].to_numpy(), a + bz[..., 1] + 2.0 * P["zr:x"].to_numpy()],
                    axis=-1)
    assert np.allclose(eta, hand, atol=1e-12)
    with pytest.raises(FormulaError, match="newdata column 'x' must be numeric"):
        m.predict(idata, newdata=new.assign(x=["a", "b", "c"]), kind="linear")


def test_priors_per_column_and_their_refusals(slopes):
    df = frame()
    b = {"z": ab.Normal(0, 1), "z:x": {"p": ab.Normal(0, 1), "q": ab.Normal(0, 2), "r": ab.Normal(0, 3)}}
    m = ab.gaussian("y ~ z + x:z", df, ab.Priors(intercept=ab.Normal(0, 5), b=b, sigma=ab.HalfNormal(1)), categorical={"z": "p"})
    assert {k: v.params["sigma"] for k, v in m.priors.coefficient_priors(m.spec).items()} == {"z[q]": 1, "z[r]": 1, "zp:x": 1, "zq:x": 2,
                                                                                               "zr:x": 3}
    assert m.priors.coefficient_priors(parse_formula("y ~ z + x:z", None, "gaussian", {"z": "p"}))["z:x"] == ab.Normal(0, 1)
    spec = m.spec
    one = ab.Priors(b={"z": ab.Normal(0, 1), "z:x": ab.Normal(0, 3)}).coefficient_priors(spec)     # one prior for every column
    assert [one[c] for c in ("zp:x", "zq:x", "zr:x")] == [ab.Normal(0, 3)] * 3

    def check(entry, message):
        with pytest.raises(PriorError, match=message):
            ab.Priors(b={"z": ab.Normal(0, 1), "z:x": entry}).coefficient_priors(spec)
    check({"p": ab.Normal(0, 1), "q": ab.Normal(0, 1)}, re.escape("must cover exactly its columns' levels ['p', 'q', 'r']"))
    check({"p": ab.Normal(0, 1), "q": ab.Normal(0, 1), "r": ab.Normal(0, 1), "s": ab.Normal(0, 1)}, "unexpected \\['s'\\]")
    check({"p": 1.0}, "map each column's levels to a Prior; got float")
    check({}, "got an empty mapping")
    check(2.0, "the interaction 'z:x' takes one prior or a mapping")
    with pytest.raises(PriorError, match="the interaction 'x:w' takes one prior, got dict"):
        ab.Priors(b={"x:w": {"": ab.Normal(0, 1)}}).coefficient_priors(parse_formula("y ~ x:w", df))
    with pytest.raises(PriorError, match="write the interaction key as R orders its variables, 'z:x'"):
        ab.Priors(b={"z": ab.Normal(0, 1), "x:z": ab.Normal(0, 1)}).coefficient_priors(spec)


def test_warnings_describe_brms_names_and_the_fit_record(slopes):
    df = frame()
    cells = parse_formula("y ~ x:a:b + x", df, "gaussian", {"a": "a1", "b": "b1"})
    assert cells.columns[1:] == ("x:aa1:bb1", "x:aa2:bb1", "x:aa3:bb1", "x:aa1:bb2", "x:aa2:bb2", "x:aa3:bb2")
    with pytest.raises(FormulaError, match=re.escape("column(s) ['aa3:bb2'] of the interaction 'a:b' are zero on every fitted row (no "
                                                     "fitted row has the level combination")):
        parse_formula("y ~ x:w + a:b", df[(df.a != "a3") | (df.b != "b2")].reset_index(drop=True), "gaussian", {"a": "a1", "b": "b1"})
    with pytest.raises(FormulaError, match=re.escape("column(s) ['x:zr'] of the interaction 'x:z' are zero on every fitted row")):
        parse_formula("y ~ x:z", df.assign(x=np.where(df.z == "r", 0.0, df.x)), "gaussian", {"z": "p"})
    colon = ab.gaussian("y ~ x:w + a:b", df, POP, categorical={"a": "a1", "b": "b1"})
    assert colon.warnings == [
        "the population-level design has rank 7 of its 8 columns (with the intercept): 1 combination(s) of the coefficients are "
        "identified by their priors only, not by the data (brms builds the same design and warns of nothing; docs/term-algebra.md, "
        "divergence D179)."]
    text = colon.describe()
    for line in ["AutoBayes Gaussian model with interaction(s) (population-level only)",
                 "  x:w       : interaction of x, w: the product column x * w (brms's x:w), recomputed from the raw columns on new rows",
                 "  a:b       : interaction of a, b coded by R's rule, a by an indicator of every level, as no earlier term holds b; b by an "
                 "indicator of every level, as no earlier term holds a: 6 column(s) aa1:bb1, aa2:bb1, aa3:bb1, aa1:bb2, aa2:bb2, aa3:bb2 "
                 "(brms's names), each the product of one level's indicator of each factor",
                 "  a         : categorical, in interactions only (no main effect; each interaction's line gives its coding), reference 'a1'",
                 "(x_x * x_w - q_x:w) * b_x:w + (1[a_i = a1] * 1[b_i = b1] - q_aa1:bb1) * b_aa1:bb1",
                 "Each interaction column of R's coding is centred at its own mean over the fitted rows: q_x:w = ",
                 "(the interaction columns of R's coding)", "- sum_c q_c * b_c (the interaction columns c)",
                 "WARNING: the population-level design has rank 7 of its 8 columns"]:
        assert line in text, line
    assert "b_a[l] is the difference" not in text
    text = slopes.describe()
    assert ("  z:x       : interaction of z, x coded by R's rule, z by an indicator of every level, as no earlier term holds x: 3 column(s) "
            "zp:x, zq:x, zr:x (brms's names), each the product of the numeric operands and one level's indicator of each factor") in text
    assert "(1[z_i = p] * x_x - q_zp:x) * b_zp:x" in text and "b_z[l] is the difference between level l and the reference 'p'" in text
    contrasts = ab.gaussian("y ~ x + x:z", df, POP, categorical={"z": "p"}, center_predictors=False).describe()
    assert "z by treatment contrasts against 'p', as x is in an earlier term: 2 column(s) x:zq, x:zr" in contrasts
    assert "x_x * 1[z_i = q] * b_x:zq" in contrasts
    assert brms_coefficient_names(slopes.spec) == {"z[q]": "zq", "z[r]": "zr", "zp:x": "zp:x", "zq:x": "zq:x", "zr:x": "zr:x"}
    idata = synthetic(draws=50)
    fit = ab.FitResult(idata, "pymc", "cpu", "cpu", "cpu", "float64", None, 1, 50, 0, 0.9, 0, seed=0,
                       _target_identity_json=json.dumps(slopes.target_identity))
    rec = ab.fit_record(slopes, fit, panel=PANEL_STUB)
    json.dumps(rec)
    assert rec["model"]["products"] == {"z:x": {"operands": ["z", "x"], "indicator_coded": ["z"], "columns": ["zp:x", "zq:x", "zr:x"]}}
    assert rec["model"]["interactions"] == [] and rec["model"]["terms"] == ["z", "zp:x", "zq:x", "zr:x"]
    with pytest.raises(ValueError, match="of the product term 'zp:x' is not offered.*divergence D162"):
        ab.fixef(slopes, None)


def test_a_distributional_formula_takes_product_terms():
    df = frame()
    m = ab.gaussian(["y ~ x", "sigma ~ z + x:z"], df, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2),
                                                                 sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}),
                    categorical={"z": "p"})
    d = m.spec.dpar("sigma")
    assert d.columns == ("z[q]", "z[r]", "zp:x", "zq:x", "zr:x") and m.warnings == []
    assert np.array_equal(m.design.dpars["sigma"].X_raw[:, 2:], np.column_stack([r_column(c, df) for c in ("zp:x", "zq:x", "zr:x")]))
    assert "(1[z_i = q] * x_x - q_zq:x) * b_sigma_zq:x" in m.describe()
    rank = ab.gaussian(["y ~ x", "sigma ~ a:b"], df, ab.Priors(intercept=ab.Normal(0, 5), b=ab.Normal(0, 2),
                                                             sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)}),
                       categorical={"a": "a1", "b": "b1"})
    assert rank.warnings == ["the population-level design of sigma has rank 6 of its 7 columns (with the intercept): 1 combination(s) of "
                             "the coefficients are identified by their priors only, not by the data (brms builds the same design and warns "
                             "of nothing; docs/term-algebra.md, divergence D179)."]


def test_a_bernoulli_three_way_model_builds_and_names_brms_s_columns():
    df = frame()
    m = ab.bernoulli("yb ~ x * w * z", df, ab.Priors(intercept=ab.Normal(0, 2), b=ab.Normal(0, 1)), categorical={"z": "p"})
    names = brms_coefficient_names(m.spec)
    assert [names[c] for c in m.spec.columns] == "x w zq zr x:w x:zq x:zr w:zq w:zr x:w:zq x:w:zr".split()
    assert np.array_equal(m.design.X_raw[:, -2:], np.column_stack([r_column(c, df) for c in ("x:w:zq", "x:w:zr")]))
    assert parse_formula("yb ~ x*w*z", None, "bernoulli", {"z": "p"}).rhs == "x + w + z + x:w + x:z + w:z + x:w:z"


def test_product_notes_and_lines_by_hand():
    s = parse_formula("y ~ x:w + a:b", frame(), "gaussian", {"a": "a1", "b": "b1"})
    X = design(s, frame())
    assert ta.product_notes(s, X, False) == []                                               # 7 columns of rank 7 without the intercept
    assert ta.product_notes(s, X, True, " of sigma")[0].startswith("the population-level design of sigma has rank 7 of its 8 columns")
    plain = parse_formula("y ~ x + w + x:w", frame())                                         # no product term: nothing is checked
    assert ta.product_notes(plain, np.zeros((48, 3)), True) == []
    assert list(ta.product_groups(s)) == ["x:w", "a:b"] and len(ta.product_lines(s)) == 2
