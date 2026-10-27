"""In-formula expressions and fitted transforms (docs/expressions.md, divergences D16, D181, D182).

A term written as a call, ``I(x * w)``, ``I(xpos^0.5)``, ``I(x > 0)``, ``log(xpos + 1)``, ``abs(x)``, ``scale(x)``, ``poly(w, 2)``,
``poly(x, 3, raw = TRUE)`` or ``factor(ord)``, and the expression inside ``offset()``, are read here. Nothing is evaluated as code:
the R text, with ``^`` read as ``**`` and every name replaced by a placeholder, is parsed by Python's ``ast`` into a tree of a
few whitelisted node kinds (numbers, column names, ``+ - * / ^``, unary minus, one comparison, and the calls ``I()``, ``log()``,
``exp()``, ``sqrt()`` and ``abs()``), which ``evaluate`` computes over the data's columns with NumPy. Anything else is refused,
naming it.

A tree is a tuple: ``("num", text)``, ``("col", name)``, ``("neg", e)``, ``("bin", op, a, b)`` with ``op`` one of ``+ - * / ^``,
``("cmp", op, a, b)`` with ``op`` one of ``> < >= <= == !=``, and ``("call", fn, (args...))``.

A term (``Expression``) is one of five forms. ``expression``: a numeric value, one design column named as written. ``logical``:
``I()`` of a comparison, which R makes a logical and codes as a factor against ``FALSE``, one column ``<term>TRUE`` holding 0 and
1. ``scale``: ``scale(col)``, ``(x - centre) / scale`` with the centre and the standard deviation (divisor ``n - 1``) of the fitted
rows. ``poly``: ``poly(col, k)``, the orthonormal polynomials of degree 1 to ``k`` over the fitted rows (the three-term
recurrence ``?poly`` documents, Kennedy and Gentle 1980), columns ``<term>1`` to ``<term>k``; with ``raw = TRUE`` the powers
``x^1`` to ``x^k``. ``factor``: ``factor(col)`` of whole numbers, treatment-coded against the lowest fitted value, one indicator
column ``<term><level>`` per other fitted level. The fitted parameters (the centre and scale, the recurrence's coefficients, the
levels) are kept and used on new rows, as R's ``predict()`` and brms do for ``scale()`` and ``poly()`` (brms does not for
``factor()``: divergence D181).
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

FUNCTIONS = ("I", "log", "exp", "sqrt", "abs")            # elementwise, anywhere in an expression
FITTED = ("scale", "poly", "factor")                       # a whole term, of one column
BASES = ("ns", "bs", "s", "te", "ti", "t2")                # spline bases and smooths, refused by name
POLY_DEGREE_MAX = 10                                       # the highest degree of poly() offered
SUPPORTED = (
    "the in-formula transforms supported are I(), log(), exp(), sqrt() and abs() of an expression in numeric columns and numbers "
    "with + - * / ^ and unary minus (and inside I() one comparison, > < >= <= == !=, a logical coded <term>TRUE), and scale(col), "
    "poly(col, k), poly(col, k, raw = TRUE) and factor(col) of one column (docs/expressions.md, docs/transforms.md, divergence "
    "D16); other function calls and transformations inside the formula are not supported: compute the column in the data "
    "frame and name it instead.")

_NAME = re.compile(r"(?<![A-Za-z0-9_.])(?:[A-Za-z_]|\.(?![0-9]))[A-Za-z0-9_.]*")
_NUMBER_TEXT = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_ALLOWED_CHARS = re.compile(r"[A-Za-z0-9_.\s+\-*/^()<>=!,]*")
_BINARY = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^"}
_COMPARE = {ast.Gt: ">", ast.Lt: "<", ast.GtE: ">=", ast.LtE: "<=", ast.Eq: "==", ast.NotEq: "!="}
_PRECEDENCE = {"^": 14, "neg": 13, "*": 11, "/": 11, "+": 10, "-": 10, "cmp": 9}


class ExpressionError(ValueError):
    """An expression outside the grammar; ``formula.FormulaError`` re-raises it with the term."""


# -- parsing ----------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def parse(text: str) -> tuple:
    """The tree of ``text`` (a call with keyword arguments only at its top, ``poly(x, 2, raw = TRUE)``); ``ExpressionError`` names
    what is outside the grammar."""
    text = text.strip()
    bad = re.sub(_ALLOWED_CHARS, "", text)
    if bad:
        raise ExpressionError(f"the character {bad[0]!r} is not part of the expressions read here")
    if "<-" in text or "->" in text:
        raise ExpressionError("R's assignment operator is not part of an expression here")
    names: list[str] = []

    def hold(m):
        names.append(m.group(0))
        return f"_v{len(names) - 1}_"

    source = _NAME.sub(hold, text).replace("^", "**")
    try:
        tree = ast.parse(source, mode="eval").body
    except SyntaxError:
        raise ExpressionError(f"{text!r} is not an expression of this grammar") from None
    return _node(tree, source, names, top=True)


def _name(node, names) -> str:
    return names[int(node.id[2:-1])]


def _node(n, source: str, names: list[str], top: bool = False) -> tuple:
    if isinstance(n, ast.Constant):
        piece = ast.get_source_segment(source, n)
        if not isinstance(n.value, (int, float)) or not _NUMBER_TEXT.fullmatch(piece or ""):     # names are placeholders: no bools
            raise ExpressionError(f"the literal {piece!r} is not a number here")
        return ("num", piece)
    if isinstance(n, ast.Name):
        return ("col", _name(n, names))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return ("neg", _node(n.operand, source, names))
    if isinstance(n, ast.BinOp) and type(n.op) in _BINARY:
        return ("bin", _BINARY[type(n.op)], _node(n.left, source, names), _node(n.right, source, names))
    if isinstance(n, ast.Compare):
        if len(n.ops) != 1 or type(n.ops[0]) not in _COMPARE:
            raise ExpressionError("a chained or unknown comparison is not supported; write one comparison")
        return ("cmp", _COMPARE[type(n.ops[0])], _node(n.left, source, names), _node(n.comparators[0], source, names))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
        fn = _name(n.func, names)
        if fn in BASES:
            raise ExpressionError(f"the spline basis {fn}() is not supported: its knots and basis are another package's (splines, mgcv); "
                                  "compute the basis columns in the data frame and name them, or use poly(col, k) (divergence D182)")
        if fn in FITTED and not top:
            raise ExpressionError(f"{fn}() is a term of its own here, of one column (scale(x), poly(x, 2), factor(x)), not part of an "
                                  "expression")
        if fn not in FUNCTIONS and fn not in FITTED:
            raise ExpressionError(f"the function {fn}() is not supported")
        if fn in FUNCTIONS and (n.keywords or len(n.args) != 1):
            raise ExpressionError(f"{fn}() takes one argument here" + ("; log(x, base) is not supported" if fn == "log" else ""))
        args = tuple(_node(a, source, names) for a in n.args)
        keywords = tuple((k.arg if k.arg is None else names[int(k.arg[2:-1])], _node(k.value, source, names)) for k in n.keywords)
        if any(k is None for k, _ in keywords):
            raise ExpressionError(f"{fn}(**...) is not supported")
        return ("call", fn, args, keywords) if keywords else ("call", fn, args)
    kind = {ast.UnaryOp: "this unary operator", ast.BinOp: "this operator"}.get(type(n), type(n).__name__)   # (dots are in names,
                                                                                                           # brackets refused above)
    raise ExpressionError(f"{kind} is not part of the expressions read here")


# -- spelling -----------------------------------------------------------------------------------------------------------
def r_number(text: str) -> str:
    """R's ``deparse()`` of a numeric literal: at most 15 significant digits, fixed notation unless the scientific one is shorter
    (``0.5`` for ``.5``, ``2`` for ``2.0``, ``1e+05`` for ``100000``, ``1e-04`` for ``0.0001``)."""
    v = float(text)
    if v == 0:
        return "0"
    target = float(f"{v:.14e}")
    digits = next(k for k in range(1, 16) if float(f"{v:.{k - 1}e}") == target)
    mantissa, exponent = f"{v:.{digits - 1}e}".split("e")
    exponent = int(exponent)
    scientific = f"{mantissa}e{'-' if exponent < 0 else '+'}{abs(exponent):02d}"
    fixed = f"{v:.{max(0, digits - 1 - exponent)}f}"
    return fixed if len(fixed) <= len(scientific) else scientific


def _precedence(e) -> int:
    if e[0] == "bin":
        return _PRECEDENCE[e[1]]
    return {"neg": _PRECEDENCE["neg"], "cmp": _PRECEDENCE["cmp"]}.get(e[0], 100)


def deparse(e, *, r_numbers: bool = False, row: bool = False) -> str:
    """The tree written back as R writes it (``deparse()``): spaces around ``+ - * /`` and comparisons, none around ``^``, the
    fewest parentheses that keep its meaning. ``r_numbers`` writes each number as R deparses it (``r_number``), else as written;
    ``row`` writes each column ``c`` as ``c_i`` (for ``describe()``)."""
    def go(e) -> str:
        if e[0] == "num":
            return r_number(e[1]) if r_numbers else e[1]
        if e[0] == "col":
            return f"{e[1]}_i" if row else e[1]
        if e[0] == "neg":
            inner = go(e[1])
            return f"-({inner})" if _precedence(e[1]) < _PRECEDENCE["neg"] else f"-{inner}"
        if e[0] in ("bin", "cmp"):
            op, a, b = e[1], e[2], e[3]
            p = _precedence(e)
            left, right = go(a), go(b)
            if _precedence(a) < p or (_precedence(a) == p and (op == "^" or e[0] == "cmp")):   # ^ is right-associative, and a
                left = f"({left})"                                                       # comparison of a comparison keeps its parentheses
            if (_precedence(b) < p and not (op == "^" and b[0] == "neg")) or (_precedence(b) == p and op != "^"):
                right = f"({right})"
            return f"{left}{op}{right}" if op == "^" else f"{left} {op} {right}"
        args = [go(a) for a in e[2]] + [f"{k} = {go(v)}" for k, v in (e[3] if len(e) > 3 else ())]
        return f"{e[1]}({', '.join(args)})"
    return go(e)


def walk(e):
    """Every node of the tree, parents before children (a call's keyword values are not visited)."""
    yield e
    if e[0] == "neg":
        yield from walk(e[1])
    elif e[0] in ("bin", "cmp"):
        yield from walk(e[2])
        yield from walk(e[3])
    elif e[0] == "call":
        for a in e[2]:
            yield from walk(a)


def columns_of(e) -> tuple[str, ...]:
    """The data columns the tree reads, in the order they first appear."""
    return tuple(dict.fromkeys(n[1] for n in walk(e) if n[0] == "col"))


# -- evaluation ---------------------------------------------------------------------------------------------------------
_UFUNC = {"log": np.log, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs, "I": lambda v: v}
_OPS = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "^": np.power}
_CMP = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal, "==": np.equal, "!=": np.not_equal}


def column_values(df: pd.DataFrame, column: str, term: str, where: str) -> np.ndarray:
    """A numeric column of ``df`` that ``term`` reads, as floats; a column that is missing, not numeric (text, bool, complex) or
    has missing or non-finite values is refused, naming it."""
    from autobayes.formula import FormulaError, _is_numeric, reject_complex

    if column not in df.columns:
        raise FormulaError(f"{where} {column!r} of the term {term!r} is not in the data")
    s = df[column]
    reject_complex(f"{where} {column!r}", s)
    if not _is_numeric(s):
        raise FormulaError(f"{where} {column!r} has dtype {s.dtype}; the term {term!r} needs a numeric column")
    v = s.to_numpy(dtype=float)
    if not np.all(np.isfinite(v)):
        raise FormulaError(f"{where} {column!r} of the term {term!r} has missing or non-finite values")
    return v


def evaluate(e, df: pd.DataFrame, *, term: str, where: str = "column") -> np.ndarray:
    """The tree's value on every row of ``df`` as floats (a comparison 0 or 1); every column it reads must be numeric and finite
    (``column_values``) and so must the result, else the first row where it is not is named."""
    from autobayes.formula import FormulaError

    cols = {c: column_values(df, c, term, where) for c in columns_of(e)}

    def go(e) -> np.ndarray:
        if e[0] == "num":
            return np.full(len(df), float(e[1]))
        if e[0] == "col":
            return cols[e[1]]
        if e[0] == "neg":
            return -go(e[1])
        if e[0] == "bin":
            return _OPS[e[1]](go(e[2]), go(e[3]))
        if e[0] == "cmp":
            return _CMP[e[1]](go(e[2]), go(e[3])).astype(float)
        return _UFUNC[e[1]](go(e[2][0]))

    with np.errstate(all="ignore"):
        out = np.asarray(go(e), dtype=float)
    if not np.all(np.isfinite(out)):
        i = int(np.flatnonzero(~np.isfinite(out))[0])
        at = ", ".join(f"{c} = {float(v[i])!r}" for c, v in cols.items())
        raise FormulaError(f"the term {term!r} is not finite at row {i}" + (f" ({at})" if at else ""))
    return out


# -- terms ---------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def read_term(text: str) -> tuple[str, str, tuple]:
    """``(form, canonical, details)`` of a term written as a call: the form (module docstring), the term as the canonical formula
    writes it, and what evaluates it: the tree (``expression``, ``logical``), the column (``scale``, ``factor``), or the column, the
    degree and whether the powers are raw (``poly``). ``ExpressionError`` for anything else."""
    e = parse(text)
    if e[0] != "call":
        raise ExpressionError(f"{text!r} is not a call")
    fn = e[1]
    if fn in FUNCTIONS:
        canonical = deparse(e)
        logical = fn == "I" and e[2][0][0] == "cmp"
        if not columns_of(e):
            raise ExpressionError(f"{canonical!r} reads no column")
        return ("logical" if logical else "expression", canonical, (e,))
    args, keywords = e[2], dict(e[3]) if len(e) > 3 else {}
    if not args or args[0][0] != "col":
        raise ExpressionError(f"{fn}() takes one column name first here, {fn}(<column>)")
    column = args[0][1]
    if fn in ("scale", "factor"):
        if len(args) > 1 or keywords:
            raise ExpressionError(f"{fn}() takes one column and no other argument here ({fn}({column})); "
                                  + ("scale()'s center = and scale = are not supported" if fn == "scale" else
                                     "factor()'s levels =, labels = and ordered = are not supported"))
        return (fn, f"{fn}({column})", (column,))
    extra = sorted(set(keywords) - {"degree", "raw"})
    if extra or len(args) > 2 or (len(args) == 2 and ("degree" in keywords or args[1][0] == "col")):
        raise ExpressionError("poly() takes one column, its degree and raw = TRUE or FALSE here, poly(x, 2) or poly(x, 2, raw = TRUE); "
                              "several columns (R's polym), coefs = and simple = are not supported")
    degree = args[1] if len(args) == 2 else keywords.get("degree", ("num", "1"))
    if degree[0] != "num" or not re.fullmatch(r"\d+", degree[1]) or not 1 <= int(degree[1]) <= POLY_DEGREE_MAX:
        raise ExpressionError(f"poly()'s degree must be a whole number from 1 to {POLY_DEGREE_MAX} here")
    raw = keywords.get("raw", ("col", "FALSE"))
    if raw[0] != "col" or raw[1] not in ("TRUE", "FALSE", "T", "F"):
        raise ExpressionError("poly()'s raw = takes TRUE or FALSE")
    return ("poly", deparse(e), (column, int(degree[1]), raw[1] in ("TRUE", "T")))


def _poly_fit(x: np.ndarray, degree: int) -> tuple[float, ...]:
    """The recurrence's coefficients over the fitted values ``x``: ``alpha_0 .. alpha_{k-1}`` and ``norm2_0 .. norm2_k``, where the
    monic orthogonal polynomials are ``p_0 = 1``, ``p_{d+1}(x) = (x - alpha_d) p_d(x) - (norm2_d / norm2_{d-1}) p_{d-1}(x)``,
    ``alpha_d = sum x p_d^2 / norm2_d`` and ``norm2_d = sum p_d^2`` over the fitted rows."""
    alpha, norm2 = [], [float(x.size)]
    prev, cur, prev_norm = np.zeros_like(x), np.ones_like(x), 1.0
    for _ in range(degree):
        a = float(np.sum(x * cur * cur) / norm2[-1])
        prev, cur, prev_norm = cur, (x - a) * cur - (norm2[-1] / prev_norm) * prev, norm2[-1]
        alpha.append(a)
        norm2.append(float(np.sum(cur * cur)))
    return (*alpha, *norm2)


def _poly_basis(x: np.ndarray, degree: int, fitted: tuple[float, ...]) -> np.ndarray:
    """The orthonormal columns of degree 1 to ``degree`` at ``x`` from the fitted coefficients (``_poly_fit``): ``p_d / sqrt(norm2_d)``."""
    alpha, norm2 = fitted[:degree], fitted[degree:]
    prev, cur, cols = np.zeros_like(x), np.ones_like(x), []
    for d in range(degree):
        prev, cur = cur, (x - alpha[d]) * cur - (norm2[d] / (norm2[d - 1] if d else 1.0)) * prev
        cols.append(cur / math.sqrt(norm2[d + 1]))
    return np.column_stack(cols)


def factor_labels(column: str, s: pd.Series, term: str, where: str = "column") -> np.ndarray:
    """``factor()``'s labels of each row: R's ``as.character()`` of whole numbers; any other column (text, bool, a pandas Categorical,
    fractional numbers) is refused naming the declaration to write instead."""
    from autobayes.formula import FormulaError, _is_numeric, _label

    if not _is_numeric(s):
        raise FormulaError(f"{where} {column!r} has dtype {s.dtype}; {term!r} is offered for a column of whole numbers here (its levels "
                           f"sort as numbers); declare a text or categorical column instead, categorical={{{column!r}: <reference "
                           "level>}} (divergence D182)")
    if s.isna().any():
        raise FormulaError(f"{where} {column!r} of the term {term!r} has missing values")
    v = s.to_numpy()
    if np.issubdtype(v.dtype, np.floating) and not np.all(np.isfinite(v) & (v == np.floor(v))):
        raise FormulaError(f"{where} {column!r} of {term!r} holds values that are not whole numbers; factor() is offered for whole "
                           "numbers here (divergence D182)")
    return np.array([_label(x) for x in v], dtype=object)


@dataclass(frozen=True)
class Expression:
    """One design column of an in-formula transform term (module docstring): ``term`` the term as the canonical formula writes it
    (the key of its priors), ``fitted`` its fitted parameters once the data are seen (scale's centre and scale, poly's recurrence
    coefficients, factor()'s levels; ``()`` before, or where it has none), ``suffix`` the column's within the term (poly's degree
    ``"1"``, factor()'s level, a logical's ``"TRUE"``; ``None`` for a term of one column named as written, and before the data
    are seen)."""

    term: str
    fitted: tuple = ()
    suffix: str | None = None

    @property
    def form(self) -> str:
        return read_term(self.term)[0]

    @property
    def details(self) -> tuple:
        return read_term(self.term)[2]

    @property
    def raw(self) -> bool:
        """``poly(..., raw = TRUE)``: plain powers, no fitted parameters."""
        return self.form == "poly" and self.details[2]

    @property
    def needs_fit(self) -> bool:
        """Its columns depend on the fitted rows (scale(), orthogonal poly(), factor())."""
        return self.form in ("scale", "factor") or (self.form == "poly" and not self.raw)

    @property
    def columns(self) -> tuple[str, ...]:
        """The data columns the term reads."""
        return columns_of(self.details[0]) if self.form in ("expression", "logical") else (self.details[0],)

    @property
    def column(self) -> str:
        """The design column's name: the term itself, or the term and the column's suffix (``poly(w, 2)1``, ``factor(ord)2``,
        ``I(x > 0)TRUE``), R's model-matrix names."""
        return self.term + (self.suffix or "")

    @property
    def brms_name(self) -> str:
        """brms's name of the coefficient: its ``rename()`` of R's model-matrix column (R's deparse of the term, numbers as R writes
        them, then the column's suffix): ``IxMUw``, ``IxposE0.5``, ``Ix>0TRUE``, ``logxposP1``, ``scalex``, ``polyw21``,
        ``polyx3rawEQTRUE1``, ``factorord2``."""
        from autobayes.default_priors import brms_rename

        return brms_rename(deparse(parse(self.term), r_numbers=True) + (self.suffix or ""))

    def suffixes(self) -> tuple[str, ...]:
        """The suffixes of the term's columns once fitted (``("",)`` for one column named as written)."""
        form = self.form
        if form == "logical":
            return ("TRUE",)
        if form == "poly":
            return tuple(str(d) for d in range(1, self.details[1] + 1))
        if form == "factor":
            if not self.fitted:
                raise ValueError(f"{self.term!r} has no fitted levels")
            return tuple(self.fitted[1:])
        return ("",)

    def estimate(self, df: pd.DataFrame) -> tuple:
        """The fitted parameters on the fitted rows ``df``, the values checked as on any rows; refuses a degree of ``poly()`` that is
        not below the number of distinct values (as R does), a ``factor()`` of fewer than two levels and a constant ``scale()``."""
        from autobayes.formula import FormulaError

        form = self.form
        if form == "factor":
            levels = tuple(sorted(set(self._factor_labels(df, "column").tolist()), key=int))
            if len(levels) < 2:
                raise FormulaError(f"{self.term!r} has a single level ({levels[0]!r}) in the fitted rows")
            return levels
        if form == "expression":
            return ()
        if form == "logical":
            values = evaluate(self.details[0][2][0], df, term=self.term)
            if np.all(values == values[0]):
                raise FormulaError(f"the comparison {self.term!r} is {'TRUE' if values[0] else 'FALSE'} on every fitted row: R codes a "
                                   "logical as a factor, and one of a single level has no contrast")
            return ()
        x = column_values(df, self.details[0], self.term, "column")
        if form == "scale":                                # R's scale(): the mean, and the SD with divisor n - 1
            centre = math.fsum(x) / x.size
            scale = math.sqrt(math.fsum((x - centre) ** 2) / (x.size - 1)) if x.size > 1 else 0.0
            if not scale > 0:
                raise FormulaError(f"the term {self.term!r} is constant in the fitted rows (its column has no spread to scale by)")
            return (centre, scale)
        if form == "poly" and not self.raw:
            degree = self.details[1]
            if degree >= np.unique(x).size:
                raise FormulaError(f"{self.term!r}: the degree {degree} is not below the number of distinct values of "
                                   f"{self.details[0]!r} in the fitted rows ({np.unique(x).size}); R refuses it too")
            return _poly_fit(x, degree)
        return ()

    def basis(self, df: pd.DataFrame, *, where: str = "column") -> np.ndarray:
        """Every column of the term on the rows of ``df``, ``(n, k)`` in the order of ``suffixes()``, from the fitted parameters."""
        from autobayes.formula import FormulaError

        form = self.form
        if self.needs_fit and not self.fitted:
            raise FormulaError(f"the term {self.term!r} needs its fitted parameters, known once the model has seen its data")
        if form == "expression":
            return evaluate(self.details[0], df, term=self.term, where=where)[:, None]
        if form == "logical":
            return evaluate(self.details[0][2][0], df, term=self.term, where=where)[:, None]
        if form == "factor":                               # D181: through the fitted levels, whichever the rows hold
            labels = self._factor_labels(df, where)
            unseen = sorted(set(labels.tolist()) - set(self.fitted), key=int)
            if unseen:
                raise FormulaError(f"{where} {self.details[0]!r} has values {unseen} that {self.term!r} was not fitted with (levels "
                                   f"{list(self.fitted)}); its columns are the fitted levels' indicators")
            return np.column_stack([(labels == lvl).astype(float) for lvl in self.fitted[1:]])
        x = column_values(df, self.details[0], self.term, where)
        if form == "scale":
            return ((x - self.fitted[0]) / self.fitted[1])[:, None]
        degree = self.details[1]
        if self.raw:
            with np.errstate(all="ignore"):
                out = np.column_stack([x ** float(d) for d in range(1, degree + 1)])
            if not np.all(np.isfinite(out)):
                raise FormulaError(f"the term {self.term!r} is not finite at row {int(np.flatnonzero(~np.isfinite(out).all(1))[0])}")
            return out
        return _poly_basis(x, degree, self.fitted)

    def _factor_labels(self, df: pd.DataFrame, where: str) -> np.ndarray:
        from autobayes.formula import FormulaError

        column = self.details[0]
        if column not in df.columns:
            raise FormulaError(f"{where} {column!r} of the term {self.term!r} is not in the data")
        return factor_labels(column, df[column], self.term, where)

    def values(self, df: pd.DataFrame, *, where: str = "column") -> np.ndarray:
        """This design column on the rows of ``df``."""
        out = self.basis(df, where=where)
        return out[:, self.suffixes().index(self.suffix or "")]

    def frame_value(self, df: pd.DataFrame) -> np.ndarray:
        """The term's model-frame column on the rows of ``df`` (brms's ``conditional_effects()`` tables carry it): ``(n,)`` for an
        expression, ``(n, k)`` for ``scale()`` (``k = 1``) and ``poly()``, whose values R holds as a matrix."""
        out = self.basis(df)
        return out if self.form in ("scale", "poly") else out[:, 0]

    def row_expression(self) -> str:
        """The column's value at row ``i``, for ``describe()``: ``x_i * w_i``, ``1[x_i > 0]``, ``(x_i - 0.3) / 1.06``, ``P_1(w_i)``,
        ``x_i^2``, ``1[ord_i = 2]``."""
        form = self.form
        if form == "expression":                           # I() of an expression is the expression
            e = self.details[0]
            return deparse(e[2][0] if e[1] == "I" else e, row=True)
        if form == "logical":
            return f"1[{deparse(self.details[0][2][0], row=True)}]"
        column = f"{self.details[0]}_i"
        if form == "scale":
            if not self.fitted:
                return f"({column} - centre) / scale"
            centre = self.fitted[0]
            return f"({column} {'-' if centre >= 0 else '+'} {abs(centre):.6g}) / {self.fitted[1]:.6g}"
        if form == "factor":
            return f"1[{column} = {self.suffix}]" if self.suffix else f"indicators of the levels of {column}"
        degree = int(self.suffix) if self.suffix else None
        if self.raw:
            return f"{column}^{degree}" if degree else f"{column}^1 .. {column}^{self.details[1]}"
        return f"P_{degree}({column})" if degree else f"P_1({column}) .. P_{self.details[1]}({column})"


def expand(term: str, fitted: tuple) -> tuple[Expression, ...]:
    """The design columns of a fitted term, in R's model-matrix order."""
    base = Expression(term, fitted)
    suffixes = base.suffixes()
    return tuple(Expression(term, fitted, s or None) for s in suffixes)


# -- describe() -----------------------------------------------------------------------------------------------------------
def describe_data(ex: Expression, values: np.ndarray) -> str:
    """The data line of one design column in ``Model.describe()``: what it is, its values over the fitted rows, what new rows
    reuse, and brms's name of its coefficient."""
    form, name, cols = ex.form, ex.column, ", ".join(repr(c) for c in ex.columns)
    head = f"  {name:<10}: "
    tail = f"; centred like any column; brms names its coefficient {ex.brms_name}"
    if form == "expression":
        return (head + f"in-formula expression of {cols}, {ex.row_expression()}, range {values.min():.4f} to {values.max():.4f} over the "
                f"fitted rows; recomputed from {cols} on new rows" + tail)
    if form == "logical":
        return (head + f"the comparison {ex.term}, a logical coded as R codes one (FALSE the reference, this column TRUE): "
                f"{ex.row_expression()}, TRUE on {int(values.sum())} of {values.size} fitted rows; recomputed on new rows" + tail)
    if form == "scale":
        return (head + f"{ex.row_expression()}: {cols} centred and scaled by its mean and SD (divisor n - 1) over the fitted rows; new "
                "rows use the same centre and scale" + tail)
    if form == "factor":
        return (head + f"{ex.row_expression()}: factor({ex.columns[0]}) with treatment contrasts against its lowest fitted level "
                f"{ex.fitted[0]!r} (levels {', '.join(ex.fitted)}, in numeric order); new rows are coded through these levels, "
                "whichever of them they hold (divergence D181)" + tail)
    if ex.raw:
        return head + f"{ex.row_expression()}: the raw power of {cols}, recomputed on new rows" + tail
    return (head + f"{ex.row_expression()}: the orthonormal polynomial of degree {ex.suffix} in {cols} over the fitted rows (the "
            "three-term recurrence of ?poly); new rows use the fitted recurrence coefficients" + tail)


def describe_meaning(ex: Expression, lp: str) -> str:
    """The line of ``Model.describe()`` saying what the column's coefficient means."""
    form, name = ex.form, ex.column
    if form == "factor":
        return (f"  {name}: b_{name} is the difference on the {lp} scale between level {ex.suffix} of factor({ex.columns[0]}) and its "
                f"reference level {ex.fitted[0]!r}, other predictors and group effects fixed.")
    if form == "logical":
        return (f"  {name}: b_{name} is the difference on the {lp} scale between rows where {ex.term[2:-1]} holds and rows where it "
                "does not, other predictors and group effects fixed.")
    if form == "poly" and not ex.raw:
        return (f"  {name}: b_{name} multiplies the orthonormal polynomial of degree {ex.suffix} in {ex.columns[0]}; the polynomial's "
                f"slope in {ex.columns[0]} combines every degree's coefficient.")
    return (f"  {name}: b_{name} is the slope in {name}, not in {' or '.join(ex.columns)}; a unit change of {name} changes {lp} by "
            f"b_{name}, other predictors and group effects fixed.")
