"""R's term algebra and its rule for coding a factor in an interaction (docs/term-algebra.md, divergences D179 and D180).

The population-level part of a formula is read as R reads it (``?formula``, ``?terms.formula``): ``a + b`` is the union of
the terms of ``a`` and ``b``; ``a:b`` the interaction of every term of ``a`` with every term of ``b``; ``a * b`` is ``a + b +
a:b``; ``(a + b + c)^k`` the sum multiplied by itself ``k`` times, every interaction of its terms up to order ``k``; ``- t``
removes the terms of ``t`` from the terms before it; a term written twice is one term; and the terms are ordered by their order
(the number of variables) and then by appearance. ``expand_rhs`` rewrites a right-hand side as the pieces
``formula.parse_formula`` reads, in that order: a column, an interaction ``a:b:c`` of columns, or a piece that is a term of its
own (a function call, a group-level term, the intercept ``1`` or ``0``).

A factor ``F`` in an interaction ``T`` is coded by treatment contrasts when ``T`` without ``F`` is contained in a term before
``T`` (a main effect, whose margin is the intercept, takes contrasts), and by an indicator of every level otherwise: ``z +
x:z`` gives ``zq, zr`` and ``zp:x, zq:x, zr:x``; ``x + x:z`` gives ``x:zq, x:zr``; ``x:w + z:h`` twelve cell indicators.
Without an intercept the first term with a factor codes it by indicators (``0 + z``). That is the rule R's ``model.matrix``
applies, as brms 2.23 shows on probe formulas (``factor_coding``; docs/term-algebra.md lists them). The variables of a term are
written in the order of brms's population-level formula, which brms rebuilds from R's ordered terms (``r_variable_order``:
``x:z + z`` is brms's ``z + z:x``), and a term's columns in R's order, the first variable's levels changing fastest.

A two-way interaction of two main effects is A04's ``"interaction"`` term (docs/interactions.md), unchanged. Every other
interaction (three or more variables, or a variable whose main effect is absent) is a ``"product"`` term: before the levels
are known one placeholder ``PopTerm``, then one ``PopTerm`` per design column (``product_columns``), the product of its numeric
operands and one level's indicator of each factor, named as R names the model-matrix column (``x:zq:hh2``, ``zp:x``) and handed
to Bambi as a precomputed column (``Product``)."""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from autobayes.formula import (_IDENT, FormulaError, _is_numeric, _refuse_r_construct, _split_top_level, categorical_labels,
                               reject_complex)

DOC = "docs/term-algebra.md"
_IDENT_RE = re.compile(_IDENT)
_POWER_RE = re.compile(r"\s*(\d+)(?![\d.])")


@dataclass(frozen=True)
class Product:
    """One design column of an interaction coded by R's rule (``"product"`` terms; module docstring).

    ``term`` is the interaction, its variables joined by ``:`` in R's order (``operands``): ``x:z:h``, ``z:x``. ``full`` are the
    factors coded by an indicator of every level (the term without the factor is in no term before it), ``contrasts`` the factors
    coded by treatment contrasts; every other operand is numeric. ``cell`` is the level of each operand that this column takes
    (``None`` for a numeric operand), known once the data are seen: empty for the placeholder of a term with factors, and empty
    for a term of numeric operands only, which is one column throughout."""

    term: str
    operands: tuple[str, ...]
    full: tuple[str, ...] = ()
    contrasts: tuple[str, ...] = ()
    cell: tuple[str | None, ...] = ()

    def __post_init__(self):
        for name in ("operands", "full", "contrasts", "cell"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.operands) < 2 or len(set(self.operands)) != len(self.operands) or self.term != ":".join(self.operands):
            raise ValueError(f"Product: {self.term!r} is the interaction of its distinct operands {list(self.operands)} joined by ':'")
        coded = self.full + self.contrasts
        if len(set(coded)) != len(coded) or any(f not in self.operands for f in coded):
            raise ValueError(f"Product {self.term!r}: each factor is coded once, by indicators or by contrasts, and is an operand")
        if self.cell and (len(self.cell) != len(self.operands) or not self.factors
                          or any((level is None) != (v not in coded) for v, level in zip(self.operands, self.cell))):
            raise ValueError(f"Product {self.term!r}: a cell gives each factor a level and each numeric operand None")

    @property
    def factors(self) -> tuple[str, ...]:
        """The factor operands, in the term's order."""
        return tuple(v for v in self.operands if v in self.full or v in self.contrasts)

    @property
    def known(self) -> bool:
        """Whether this is a design column (its cell known, or no factor), rather than the placeholder of its term."""
        return bool(self.cell) or not self.factors

    @property
    def column(self) -> str:
        """R's model-matrix name of the column, each factor's name pasted to its level: ``x:zq:hh2``, ``zp:x``, ``x:w:v`` (the term
        itself before the levels are known)."""
        if not self.cell:
            return self.term
        return ":".join(v if level is None else f"{v}{level}" for v, level in zip(self.operands, self.cell))

    @property
    def label(self) -> str:
        """The column's levels joined by ``, `` (``q, h2``): its key in a per-column prior mapping (``""`` without a factor)."""
        return ", ".join(level for level in self.cell if level is not None)

    def values(self, df: pd.DataFrame) -> np.ndarray:
        """The column on the rows of ``df``: the product of the numeric operands and of each factor's indicator of its level (a
        categorical column is read as its string labels, as the fitted frame and checked new rows hold them)."""
        if not self.known:
            raise ValueError(f"the interaction {self.term!r} has no design column until its factors' levels are known")
        out = np.ones(len(df))
        for v, level in zip(self.operands, self.cell or (None,) * len(self.operands)):
            out = out * (df[v].to_numpy(dtype=float) if level is None else (df[v].astype(str).to_numpy() == level))
        return out


# -- the term algebra ----------------------------------------------------------------------------------------------------------
def _closing(text: str, i: int) -> int:
    """The index of the ``)`` that closes the ``(`` at ``text[i]``."""
    depth = 0
    for j in range(i, len(text)):
        depth += {"(": 1, ")": -1}.get(text[j], 0)
        if depth == 0:
            return j
    raise FormulaError(f"unbalanced parentheses in {text!r}")


def _signed_summands(rhs: str) -> list[tuple[str, str]]:
    """``rhs`` split at its top-level ``+`` and ``-`` into ``(sign, text)``; a leading ``-`` (``-1 + x``) signs the first."""
    out, depth, buf, sign = [], 0, [], "+"
    for ch in rhs:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth < 0:
            raise FormulaError(f"unbalanced parentheses in {rhs!r}")
        if depth == 0 and ch in "+-":
            out.append((sign, "".join(buf).strip()))
            buf, sign = [], ch
            continue
        buf.append(ch)
    if depth != 0:
        raise FormulaError(f"unbalanced parentheses in {rhs!r}")
    out.append((sign, "".join(buf).strip()))
    return out[1:] if len(out) > 1 and out[0] == ("+", "") and out[1][0] == "-" else out


def _is_algebra(text: str) -> bool:
    """Whether a summand is read by the term algebra (a column, or an expression of columns with ``*``, ``:``, ``^`` and
    parentheses) rather than as a term of its own (a function call, a group-level term)."""
    if _IDENT_RE.fullmatch(text) or any(len(_split_top_level(text, op)) > 1 for op in "*:^"):
        return True
    return text.startswith("(") and _closing(text, 0) == len(text) - 1 and len(_split_top_level(text[1:-1], "|")) == 1


def _union(a: list, b: list) -> list:
    out = list(a)
    for t in b:
        if t not in out:
            out.append(t)
    return out


def _cross(a: list, b: list) -> list:
    return _union([], [s | t for s in a for t in b])


class _Algebra:
    """One summand of a right-hand side read as R's term algebra: ``parse`` gives its terms, each a frozenset of variables, in the
    order the expansion writes them; ``order`` records each variable's first appearance across the formula."""

    def __init__(self, text: str, order: dict[str, int]):
        self.s, self.i, self.order, self.local = text, 0, order, {}

    def fail(self, message: str):
        raise FormulaError(f"{self.s!r}: {message}")

    def peek(self) -> str:
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1
        return self.s[self.i] if self.i < len(self.s) else ""

    def parse(self) -> list[frozenset]:
        terms = self.sum()
        if self.peek():
            self.fail(f"{self.s[self.i:]!r} is not read: R's term algebra takes columns, '+', '-', '*', ':', '^' and parentheses")
        return terms

    def sum(self) -> list[frozenset]:
        terms = self.product()
        while self.peek() and self.peek() in "+-":
            op = self.s[self.i]
            self.i += 1
            right = self.product()
            terms = _union(terms, right) if op == "+" else [t for t in terms if t not in right]
        return terms

    def product(self) -> list[frozenset]:
        left = self.interaction()
        while self.peek() == "*":
            self.i += 1
            right = self.interaction()
            self.distinct(left, right, "*")
            left = _union(_union(left, right), _cross(left, right))
        return left

    def interaction(self) -> list[frozenset]:
        left = self.power()
        while self.peek() == ":":
            self.i += 1
            right = self.power()
            self.distinct(left, right, ":")
            left = _cross(left, right)
        return left

    def power(self) -> list[frozenset]:
        start = self.i
        base = self.atom()
        written = self.s[start:self.i].strip()
        if self.peek() != "^":
            return base
        self.i += 1
        m = _POWER_RE.match(self.s, self.i)
        if m is None or int(m.group(1)) < 2:
            self.fail(f"R's '^' takes a whole number of at least 2 here, '(a + b + c)^2': every interaction of the terms up to that "
                      f"order ({DOC})")
        self.i = m.end()
        if len(base) < 2:
            self.fail(f"R reads {written}^{m.group(1)} as {written} itself: the power of a formula term multiplies a sum of terms "
                      f"into their interactions and is not the power of a column; write I({written}^{m.group(1)}) for that "
                      f"(docs/transforms.md), or the sum '(a + b)^{m.group(1)}' for interactions (divergence D180)")
        out = base
        for _ in range(int(m.group(1)) - 1):
            out = _union(_union(out, base), _cross(out, base))
        return out

    def atom(self) -> list[frozenset]:
        ch = self.peek()
        if ch == "(":
            j = _closing(self.s, self.i)
            if len(_split_top_level(self.s[self.i + 1:j], "|")) > 1:   # a group-level term: a term of its own, refused here
                text, self.i = self.s[self.i:j + 1], j + 1
                return [self.variable(text)]
            self.i += 1
            terms = self.sum()
            if self.peek() != ")" or self.i != j:
                self.fail(f"{self.s[self.i:j]!r} is not read: R's term algebra takes columns, '+', '-', '*', ':', '^' and parentheses")
            self.i = j + 1
            return terms
        m = _IDENT_RE.match(self.s, self.i)
        if m is None:
            self.fail(f"{self.s[self.i:]!r} is not a column name: R's term algebra takes columns, function calls, '+', '-', '*', ':', "
                      "'^' and parentheses (an intercept, '1' or '0', is a term of its own, outside them)")
        rest = self.s[m.end():]
        if rest.lstrip().startswith("("):                               # a function call: one variable, as R reads it
            j = _closing(self.s, m.end() + len(rest) - len(rest.lstrip()))
            text, self.i = self.s[self.i:j + 1], j + 1
            return [self.variable(text)]
        self.i = m.end()
        return [self.variable(m.group(0))]

    def variable(self, text: str) -> frozenset:
        self.order.setdefault(text, len(self.order))
        self.local.setdefault(text, len(self.local))
        return frozenset({text})

    def distinct(self, left: list, right: list, op: str) -> None:
        """``x:x``, ``x*x`` and ``x:z:x``, two single terms that share a variable, are refused: R reads ``x:x`` as ``x``."""
        if len(left) == 1 and len(right) == 1 and left[0] & right[0]:
            v = min(left[0] & right[0], key=self.order.__getitem__)
            self.fail(f"a term cannot interact with itself ({v!r} on both sides of '{op}'); R reads {v}:{v} as {v}, not as a "
                      f"square: write I({v}^2) for that (docs/transforms.md, divergence D180)")


def expand_rhs(rhs: str, formula: str) -> list[tuple[str, str]]:
    """The right-hand side ``rhs`` as the pieces ``parse_formula`` reads, each ``(piece, as written)``, in R's term order: a column,
    an interaction ``a:b:c`` of R's term algebra (its variables in the order they first appear in the formula; an interaction with a
    function call or a group-level term, which the parser refuses naming it, in the order its summand writes them), or a piece read
    on its own (a function call, a group-level term, ``1`` or ``0``; ``- 1`` is ``0``). Terms of the algebra written twice are one;
    ``- t`` removes the terms of ``t`` (a column or an interaction of columns); pieces read on their own keep every occurrence, so
    that a call written twice is refused by the parser as before."""
    order: dict[str, int] = {}
    items: list[tuple[frozenset | None, str, str, dict]] = []
    for sign, text in _signed_summands(rhs):
        if text == "":
            raise FormulaError(f"{formula!r}: empty term")
        if text in ("0", "1") or not _is_algebra(text):
            if sign == "-" and text != "1":
                raise FormulaError(f"{formula!r}: '- {text}': '-' removes the intercept ('- 1') or population-level terms of columns "
                                   f"and their interactions ('- x', '- x:z'; {DOC})")
            items.append((None, "0" if sign == "-" else text, text, {}))
            continue
        _refuse_r_construct(text)
        algebra = _Algebra(text, order)
        terms = algebra.parse()
        if sign == "-":
            items = [it for it in items if it[0] not in terms]
            continue
        have = {it[0] for it in items}
        items += [(t, "", text, algebra.local) for t in terms if t not in have]
    items.sort(key=lambda it: 1 if it[0] is None else len(it[0]))       # stable: by order, then by appearance
    return [(piece if term is None else ":".join(sorted(term, key=(order if all(_IDENT_RE.fullmatch(a) for a in term) else local).__getitem__)),
             written) for term, piece, written, local in items]


def r_variable_order(terms) -> dict[str, int]:
    """The position of each variable in brms's population-level formula, which brms rebuilds from R's ordered terms: each term's
    variables in the order they first appear in the formula, the terms in R's order (``x:z + z`` is brms's ``z + z:x``)."""
    order: dict[str, int] = {}
    for t in terms:
        for v in t:
            order.setdefault(v, len(order))
    return order


def factor_coding(terms: list[tuple[str, ...]], factors, intercept: bool, formula: str) -> list[frozenset]:
    """For each term (its variables, in R's term order) the factors coded by an indicator of every level; the others are coded by
    treatment contrasts. A factor ``F`` of an interaction ``T`` takes contrasts when ``T`` without ``F`` is contained in a term
    before ``T``; a main effect always does, except that without an intercept the first term with a factor codes its factors by
    indicators (``0 + z``). Where that turns a factor whose margin is present into indicators (``0 + x + x:z``: ``x``, ``x:zp``,
    ``x:zq``, ``x:zr``, whose columns add up to ``x``), R and brms build the rank-deficient design; that is refused (D180)."""
    sets = [frozenset(t) for t in terms]
    out = [frozenset(f for f in t if f in factors and len(t) > 1 and not any(s - {f} <= u for u in sets[:j]))
           for j, (t, s) in enumerate(zip(terms, sets))]
    first = next((j for j, s in enumerate(sets) if s & set(factors)), None)
    if not intercept and first is not None and len(sets[first]) > 1 and out[first] != sets[first] & set(factors):
        (f,) = (sets[first] & set(factors)) - out[first]
        margin = ":".join(v for v in terms[first] if v != f)
        raise FormulaError(
            f"{formula!r}: without an intercept R and brms code the first factor of the formula by an indicator of every level, here "
            f"{f!r} in {':'.join(terms[first])!r}, although its margin {margin!r} is a term: the columns of {':'.join(terms[first])!r} then "
            f"add up to {margin!r} (a rank-deficient design). Keep the intercept, or drop {margin!r} so that each level of {f!r} has "
            f"its own coefficient ({DOC}, divergence D180)")
    if not intercept and first is not None:
        out[first] = sets[first] & set(factors)
    return out


def product_columns(t, categoricals) -> tuple:
    """The design columns of the product term ``t`` once its factors' levels are known, one ``PopTerm`` each in R's order (the
    first operand's level changing fastest): a factor coded by contrasts takes its non-reference levels, one coded by indicators
    every level, reference first (``Categorical.categories``). A term of numeric operands is its own column. An ordered factor
    (polynomial contrasts) is refused as in A04's interactions (D168), and so is a level whose label holds a backtick."""
    from autobayes.formula import PopTerm

    p = t.product
    if not p.factors:
        return (t,)
    cats = {c.name: c for c in categoricals}
    poly = next((cats[f] for f in p.factors if cats[f].is_poly), None)
    if poly is not None:
        raise FormulaError(
            f"{p.term!r}: an interaction with the ordered factor {poly.name!r} is not supported: brms codes an ordered factor with "
            "polynomial contrasts and gives each its own interaction column, which no frozen case verifies here "
            "(docs/ordered-factors.md, divergence D168)")
    choices = [[None] if v not in p.factors else list(cats[v].categories if v in p.full else cats[v].contrasts) for v in p.operands]
    out = []
    for combo in itertools.product(*reversed(choices)):
        q = replace(p, cell=tuple(reversed(combo)))
        if "`" in q.column:
            raise FormulaError(f"{p.term!r}: the column {q.column!r} has a backtick in a level label, which a column name handed "
                               "to Bambi cannot hold; relabel the level")
        out.append(PopTerm(q.column, "product", (q.column,), product=q))
    return tuple(out)


def check_product_names(terms, reads) -> None:
    """Refuse two columns of the population-level terms with one name, or a product column named like a column the formula
    reads: R's names paste a factor's name to its level, so ``z`` with level ``ab`` and ``za`` with level ``b`` both give
    ``zab``."""
    names = [t.name for t in terms]
    for t in terms:
        if t.kind == "product" and (names.count(t.name) > 1 or t.name in reads):
            raise FormulaError(
                f"the interaction column {t.name!r} of {t.product.term!r} (R's name: each factor's name pasted to its level) is also "
                f"the name of another column or term of the formula; relabel a level or rename a column so that the names differ ({DOC})")


def check_products(spec, data: pd.DataFrame, categoricals=()) -> None:
    """The data of the product terms, with the factors' levels known (``categoricals``): a numeric operand that is not a main
    effect is a numeric column, as a main effect is; a product of numeric operands only must vary over the fitted rows, as A04's
    ``x:w`` must; and no column may be zero on every fitted row (a cell no row reaches). brms keeps such a column, its
    coefficient informed by its prior alone, as A04's ``z:h`` keeps an empty cell; Bambi takes no constant column handed to it,
    so it is refused here, naming the column (divergence D179)."""
    for t in spec.pop_terms:
        if t.kind != "product":
            continue
        for v in t.product.operands:
            if v in t.product.factors or v in spec.predictors:
                continue
            reject_complex(f"column {v!r}", data[v])
            if not _is_numeric(data[v]):
                raise FormulaError(
                    f"column {v!r} of the interaction {t.product.term!r} has dtype {data[v].dtype}; a column of an interaction is "
                    f"numeric unless it is declared categorical with its reference level, categorical={{{v!r}: <level>}}")
        if not t.product.factors and np.isclose(np.std(t.product.values(data)), 0.0):
            raise FormulaError(f"interaction {t.name!r} is constant ({' * '.join(t.product.operands)} takes a single value in the "
                               "fitted rows)")
        if not t.product.factors:
            continue
        labelled = data[list(t.product.operands)].copy()
        for f in t.product.factors:                               # the factors' labels as the fitted frame holds them
            labelled[f] = categorical_labels(f"categorical predictor {f!r}", data[f])[0]
        empty = [c.name for c in product_columns(t, categoricals) if not c.product.values(labelled).any()]
        if empty:
            raise FormulaError(
                f"column(s) {empty} of the interaction {t.product.term!r} are zero on every fitted row (no fitted row has the level "
                "combination, or the numeric operands are zero on its rows): brms keeps such a column, informed by its prior alone; "
                "it is refused here, as Bambi takes no constant column. Drop the unused levels from the data, or write the "
                f"interaction so that no column is empty ({DOC}, divergence D179)")


def refuse_products(specs, where: str) -> None:
    """The model kinds whose own design code no frozen brms case verifies with a product term (a non-linear parameter's formula,
    a count-matrix category's, a multivariate response's) refuse it, naming the term (divergence D179)."""
    for s in specs:
        t = next((t for t in s.pop_terms if t.kind == "product"), None)
        if t is not None:
            raise FormulaError(
                f"{s.formula!r}: the interaction {t.product.term!r} (of three or more variables, or with a variable whose main effect "
                f"is absent, coded by R's rule; {DOC}) is not supported in {where}: no frozen brms case verifies it there "
                "(divergence D179)")


# -- what a model with product terms says about them ---------------------------------------------------------------------------
def product_groups(spec) -> dict[str, list]:
    """The product terms of ``spec`` by interaction, ``{"x:z:h": [PopTerm, ...]}``, in term order."""
    out: dict[str, list] = {}
    for t in spec.pop_terms:
        if t.kind == "product":
            out.setdefault(t.product.term, []).append(t)
    return out


def product_notes(spec, X: np.ndarray, intercept: bool, what: str = "") -> list[str]:
    """``Model.warnings`` about the product terms of one formula's design ``X`` (raw, in ``spec.columns`` order): a design of less
    than full rank, which brms builds and samples without a word (divergence D179), its coefficients then identified only by
    their priors along the directions the data do not reach (``gauss_colon_only``: the cells of ``z:h`` add up to the
    intercept). Checked only for a formula with a product term, so that every other model's warnings are as they were."""
    notes = []
    if product_groups(spec):
        full = np.column_stack([np.ones(len(X)), X]) if intercept else X
        rank = int(np.linalg.matrix_rank(full))
        if rank < full.shape[1]:
            notes.append(
                f"the population-level design{what} has rank {rank} of its {full.shape[1]} columns{' (with the intercept)' if intercept else ''}:"
                f" {full.shape[1] - rank} combination(s) of the coefficients are identified by their priors only, not by the data "
                f"(brms builds the same design and warns of nothing; {DOC}, divergence D179).")
    return notes


def product_lines(spec) -> list[str]:
    """``describe()``'s lines of the product terms: each interaction's coding and columns."""
    lines = []
    for term, cols in product_groups(spec).items():
        p = cols[0].product
        if not p.factors:
            lines += [f"  {term:<10}: interaction of {', '.join(p.operands)}: the product column {' * '.join(p.operands)} (brms's {term}), "
                      "recomputed from the raw columns on new rows; centred like any column"]
            continue
        coding = []
        for f in p.factors:
            margin = ":".join(v for v in p.operands if v != f)
            coding.append(f"{f} by an indicator of every level, as no earlier term holds {margin}" if f in p.full else
                          f"{f} by treatment contrasts against {spec.categorical(f).reference!r}, as {margin} is in an earlier term")
        lines += [f"  {term:<10}: interaction of {', '.join(p.operands)} coded by R's rule, {'; '.join(coding)}: {len(cols)} column(s) "
                  f"{', '.join(c.name for c in cols)} (brms's names), each the product of "
                  + ("the numeric operands and " if len(p.factors) < len(p.operands) else "")
                  + "one level's indicator of each factor; recomputed from the raw columns on new rows; centred like any column"]
    return lines
