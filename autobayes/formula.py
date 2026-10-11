"""The formula grammar, its validation, and the term model a parsed formula is made of.

A parsed formula is a ``ModelSpec`` holding population-level terms (``PopTerm``: ``x``, ``z``, ``x:z``) and
group-level terms (``GroupTerm``: ``(1 | g)``, ``(1 + x || g)``, ``(1 + x | g)``); every flat view of the formula (``predictors``,
``terms``, ``columns``, ``groups``, ...) is derived from those two tuples. See docs/term-model.md.

Supported::

    response ~ [1 + | 0 +] predictor [+ predictor ...] [+ a:b ...] [+ offset(col) | offset(log(col))]
               [+ (1 | factor) | (1 + x [+ w ...] || factor) | (1 + x [+ w ...] | factor) ...]
    successes | trials(n) ~ ...  the same right-hand side  (binomial, brms syntax)
    response | weights(col) ~ ...  (and successes | trials(n) + weights(col) ~ ...)

``offset(col)`` and ``offset(log(col))`` (A06) add a known column to the linear predictor with no coefficient and
no prior, as brms's ``offset()`` does; one offset term per formula, held in ``ModelSpec.offset`` (docs/offset.md).

brms's response addition terms other than ``trials()`` (B02, docs/addition-terms.md) are ``AdditionTerm`` entries of
``ModelSpec.additions``, joined with ``+`` after the ``|`` as brms writes them: ``weights(col)`` multiplies each row's
log likelihood by its weight. ``cens(col)`` and ``trunc(lb = a, ub = b)`` are parsed, represented and implemented
(Gaussian) but withheld (``PARKED_ADDITIONS``): brms's own gradients of their frozen fixtures are not exact enough to
certify them.

An interaction ``a:b`` (or ``a*b`` = ``a + b + a:b``) is two-way between any two main effects of the formula (A04,
docs/interactions.md): numeric by declared categorical (``x:z``, the numeric column times each non-reference
indicator), numeric by numeric (``x:w``, the elementwise product, one coefficient) or categorical by categorical
(``z:h``, one indicator product per cell of non-reference levels, R's treatment coding of the interaction). Both
main effects are required; three-way interactions are refused.

A predictor may be an in-formula transform of one numeric column (A05, docs/transforms.md): ``log(col)``,
``exp(col)``, ``sqrt(col)``, ``I(col^k)`` with integer ``k``, ``I(col * k)``, ``I(col + k)``, ``I(col - k)`` with a
numeric constant ``k``. It is a term of kind ``"transform"`` named as written, with one design column (the
evaluated function of the raw column, recomputed on new rows) and one coefficient; anything else inside a call is
refused with the supported list.

A grouping factor is a column, or two columns combined (A03, docs/grouping-syntax.md): ``(1 | g:h)`` groups by the
observed ``(g, h)`` combinations, labelled ``g_h`` as brms labels them, and ``(1 | g/h)`` is brms's nesting
shorthand for ``(1 | g) + (1 | g:h)``.

A model may also be given a list of formulas (B01, docs/distributional.md): the response formula first, then one
formula per distributional parameter, ``["y ~ x + (1 | g)", "sigma ~ z"]`` (brms's ``bf(y ~ x + (1 | g), sigma ~ z)``).
Each distributional formula is parsed as a formula of its own (population-level terms only, its own intercept) into
``ModelSpec.dpars``; which parameters take one is the family's (``FamilySpec.distributional_link``).

``mo(col)`` is brms's monotonic effect of an ordinal predictor (B03, docs/monotonic.md): a population-level term of kind
``"monotonic"`` with no column in the design ``X``; its coefficient (brms's ``bsp``) multiplies ``mo(col)_i = D * sum_{k <=
x_i} simo[k]``, the share of the simplex ``simo`` up to row ``i``'s value coded ``0..D``. The column holds whole numbers
(coded as the value minus the smallest fitted value) or an ordered pandas Categorical (coded by its declared categories
that the fitted rows take); ``ModelSpec.monotonics`` keeps that coding once the data are seen. Interactions with a
``mo()`` term, ``mo()`` in a group-level or distributional formula and ``mo()``'s ``id`` argument are refused.

Group-level terms are optional and additive across distinct factors (M3c-3): a formula without them is a population-only
model, as in brms ``brmsformula`` where group-level terms are optional; ``y ~ 1`` is intercept-only. ``(1 + x || g)``
is brms's independent varying intercept and slope (A01): each numeric population-level term named varies by level
with its own SD and no correlation (docs/varying-slopes.md); brms's correlated form ``(1 + x | g)`` (A02) gives the
coefficients a correlation matrix under an LKJ prior (docs/correlated-slopes.md).

Predictors are numeric columns, or categorical columns declared with ``categorical={column:
reference_level}`` (treatment coding, milestone M3a-1); each grouping variable is a column
and is coded independently with sorted fitted string levels. Bambi (via formulae) accepts a
much larger language; this module exists so that a formula outside the verified subset is
refused with an error naming the construct, rather than silently built.
"""
from __future__ import annotations

import re
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import pandas as pd

_IDENT = r"[A-Za-z_][A-Za-z0-9_.]*"
_FORMULA_RE = re.compile(rf"^\s*({_IDENT})\s*~\s*(.+?)\s*$", re.S)
_ADDITIONS_FORMULA_RE = re.compile(rf"^\s*({_IDENT})\s*\|\s*([^~]+?)\s*~\s*(.+?)\s*$", re.S)   # response | addition terms ~ rhs
_ADDITION_CALL_RE = re.compile(rf"^({_IDENT})\s*\((.*)\)$", re.S)                               # trials(n), weights(w), trunc(lb = 0)
_BAMBI_PROPORTION_RE = re.compile(r"^\s*(p|prop|proportion)\s*\(", re.S)
_GROUP_RE = re.compile(rf"^\(\s*([^|\s][^|]*?)\s*(\|\|?)\s*(.+?)\s*\)$", re.S)   # (inner | factor) or (inner || factor)
_OFFSET_RE = re.compile(r"^offset\s*\((.*)\)$", re.S)                             # offset(<inner>)
_OFFSET_LOG_RE = re.compile(rf"^log\s*\(\s*({_IDENT})\s*\)$", re.S)              # log(col) inside offset()
_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"                           # a numeric literal: 2, 2.5, .5, 1e-1, -1
_TRANSFORM_FN_RE = re.compile(rf"^(log|exp|sqrt)\s*\(\s*({_IDENT})\s*\)$", re.S)                       # log(col)
_TRANSFORM_POW_RE = re.compile(rf"^I\s*\(\s*({_IDENT})\s*\^\s*([+-]?\d+)\s*\)$", re.S)               # I(col^k), integer k
_TRANSFORM_OP_RE = re.compile(rf"^I\s*\(\s*({_IDENT})\s*([*+-])\s*({_NUMBER})\s*\)$", re.S)          # I(col * k), I(col + k), I(col - k)
_TRANSFORM_OP_FIRST_RE = re.compile(rf"^I\s*\(\s*({_NUMBER})\s*([*+])\s*({_IDENT})\s*\)$", re.S)     # I(k * col), I(k + col)
_MO_RE = re.compile(r"^mo\s*\((.*)\)$", re.S)                                                        # mo(<inside>), brms's monotonic effect (B03)
_MO_TERM_RE = re.compile(rf"^mo\(({_IDENT})\)$")                                                    # the canonical spelling mo(col)
SUPPORTED_TRANSFORMS = ("only log(col), exp(col), sqrt(col), I(col^k) with integer k, and I(col * k), I(col + k), I(col - k) with a "
                        "numeric constant k are supported as in-formula transforms of one numeric column (A05, docs/transforms.md); "
                        "other function calls and transformations inside the formula are not supported: compute the column in the "
                        "data frame and name it instead.")

COUNT_CAP = 10**6   # the validated domain for binomial counts per row (docs/m2-decision.md, repair record)


class FormulaError(ValueError):
    """A formula outside the M0 subset, or one that does not match the data."""


@dataclass(frozen=True)
class Categorical:
    """A categorical main effect: treatment or full-level coding (docs/m3c2-decision.md).

    ``levels`` are the distinct string labels present in the fitted rows in ``sorted`` order
    (empty until the data are seen); ``reference`` is the declared reference label, which
    must be one of them; ``coefficients`` are the levels without the reference, in order,
    one dummy column and one coefficient each, unless ``full_rank`` includes every level
    in reference-first order; ``unused`` lists a pandas Categorical's
    declared categories that do not occur in the fitted rows.
    """

    name: str
    reference: str
    levels: tuple[str, ...] = ()
    unused: tuple[str, ...] = ()
    full_rank: bool = False

    @property
    def coefficients(self) -> tuple[str, ...]:
        return self.categories if self.full_rank else self.contrasts

    @property
    def contrasts(self) -> tuple[str, ...]:
        """Levels with a difference coefficient relative to the reference."""
        return tuple(l for l in self.levels if l != self.reference)

    @property
    def categories(self) -> tuple[str, ...]:
        """Reference first, then the other levels: the ordered Categorical handed to formulae."""
        return (self.reference, *self.contrasts)

    @property
    def columns(self) -> tuple[str, ...]:
        """Design-column labels, Bambi/formulae's own: ``z[level]`` per coefficient level."""
        return tuple(f"{self.name}[{l}]" for l in self.coefficients)


POP_TERM_KINDS = ("numeric", "categorical", "interaction", "transform", "monotonic")


def monotonic_column(name: str) -> str | None:
    """The column of a monotonic term spelled ``mo(col)`` (B03), or None for any other term name."""
    m = _MO_TERM_RE.match(name)
    return m.group(1) if m else None


@dataclass(frozen=True)
class Monotonic:
    """How a monotonic predictor ``mo(col)`` (B03, docs/monotonic.md) is coded, brms's ``get_mo_values``: each row's value
    becomes an integer ``0..D`` (brms's ``Xmo``), ``D`` (brms's ``Jmo``) the number of steps between the lowest and the
    highest value, each step with its share of the simplex. Whole numbers are coded as the value minus ``minimum``, the
    smallest fitted value, so every integer up to ``maximum`` is a value whether or not a fitted row takes it (brms's rule);
    an ordered pandas Categorical (brms's ordered factor) by the position among ``categories``, its declared categories that
    occur in the fitted rows, lowest first (brms drops a factor's unused levels before coding it). Nothing but the column is
    known until the data are seen; new rows are coded the same way and a value outside the fitted coding is refused (brms
    refuses it too)."""

    column: str
    minimum: int | None = None
    maximum: int | None = None
    categories: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "categories", tuple(self.categories))

    @property
    def term(self) -> str:
        """The term as the canonical formula writes it: ``mo(col)``."""
        return f"mo({self.column})"

    @property
    def ordered(self) -> bool:
        """Whether the column is an ordered Categorical (rather than whole numbers)."""
        return bool(self.categories)

    @property
    def known(self) -> bool:
        """Whether the coding has been read from the data."""
        return self.ordered or self.minimum is not None

    @property
    def D(self) -> int:
        """brms's ``Jmo``: the number of steps, which is the length of the simplex."""
        if not self.known:
            raise ValueError(f"the coding of {self.term!r} is known once the data are seen")
        return len(self.categories) - 1 if self.ordered else int(self.maximum) - int(self.minimum)

    @property
    def lowest(self) -> str:
        """The lowest value, as text (coded 0)."""
        return self.categories[0] if self.ordered else str(self.minimum)

    @property
    def highest(self) -> str:
        """The highest value, as text (coded D)."""
        return self.categories[-1] if self.ordered else str(self.maximum)

    def values(self) -> tuple[str, ...]:
        """Every value of the coding as text, lowest first: the value coded ``k`` is ``values()[k]``."""
        return self.categories if self.ordered else tuple(str(v) for v in range(int(self.minimum), int(self.maximum) + 1))

    def step_labels(self) -> tuple[str, ...]:
        """The steps the simplex splits the effect over, ``"1 -> 2"``, ..., in order (simplex entry ``k``, 1-based, is the
        share of the step from the value coded ``k - 1`` to the value coded ``k``)."""
        values = self.values()
        return tuple(f"{a} -> {b}" for a, b in zip(values[:-1], values[1:]))

    def notes(self, s: pd.Series) -> list[str]:
        """The fitted column ``s``'s warnings: whole numbers in ``minimum..maximum`` that no row takes (brms keeps them as
        values), or an ordered Categorical's declared categories that no row takes (brms drops them)."""
        if self.ordered:
            declared = [_scalar(self.term, c, where="declared category") for c in s.cat.categories]
            dropped = [c for c in declared if c not in self.categories]
            return [f"declared categories {dropped} of the monotonic predictor {self.column!r} do not occur in the fitted rows: they are "
                    f"not values of {self.term} (brms drops them too), and new rows taking them are refused"] if dropped else []
        unseen = np.setdiff1d(np.arange(self.D + 1), self.codes(s))
        shown = [str(int(self.minimum) + int(k)) for k in unseen[:10]] + (["..."] if unseen.size > 10 else [])
        return [f"values {shown} of the monotonic predictor {self.column!r} have no fitted rows: they are still values of "
                f"{self.term} (brms keeps them), and the data inform only the sum of the two steps next to each, which the prior "
                "on the simplex splits"] if unseen.size else []

    def codes(self, s: pd.Series, *, where: str = "column") -> np.ndarray:
        """brms's ``Xmo`` of ``s`` under this coding: an int64 array of codes ``0..D``. Refuses a value that is not one of
        the coding's (a non-whole number, a value outside ``minimum..maximum``, a label outside ``categories``), a missing
        value, and a boolean or complex column."""
        if not self.known:
            raise ValueError(f"the coding of {self.term!r} is known once the data are seen")
        label = f"{where} {self.column!r} of the monotonic term {self.term!r}"
        reject_complex(label, s)
        if pd.api.types.is_bool_dtype(s):
            raise FormulaError(f"{label} is boolean; a monotonic predictor holds whole numbers or ordered categories")
        if s.isna().any():
            raise FormulaError(f"{label} has missing values; drop or impute them explicitly")
        if self.ordered:
            position = {c: k for k, c in enumerate(self.categories)}
            labels = [_scalar(label, v) for v in s.astype(object).tolist()]
            unknown = sorted({l for l in labels if l not in position})
            if unknown:
                raise FormulaError(f"{label} has values {unknown} that are not among its ordered categories {list(self.categories)}; "
                                   "a monotonic predictor takes only the categories it was fitted with (brms refuses others too)")
            return np.array([position[l] for l in labels], dtype=np.int64)
        if not _is_numeric(s):
            raise FormulaError(f"{label} has dtype {s.dtype}; it was fitted as whole numbers {self.minimum}..{self.maximum}")
        v = s.to_numpy(dtype=float)
        bad = v[~np.isfinite(v) | (v != np.floor(v)) | (v < self.minimum) | (v > self.maximum)]
        if bad.size:
            raise FormulaError(f"{label} has the value {float(bad[0])!r}, which is not a whole number from {self.minimum} to {self.maximum} "
                               "(the fitted values; brms refuses such values on new rows too)")
        return (v - self.minimum).astype(np.int64)


_TWO_VALUES = ("with two values a monotonic effect has a single step and no simplex to estimate (brms's simplex[1] is the "
               "constant 1, and mo() is then the predictor coded 0/1 times its coefficient): write that 0/1 column as a numeric "
               "term instead (docs/monotonic.md, divergence D42)")


def monotonic_coding(column: str, s: pd.Series) -> Monotonic:
    """The coding of the fitted column ``s`` of ``mo(column)`` (brms's ``get_mo_values``): an ordered pandas Categorical by
    its declared categories that occur in the fitted rows (brms drops the unused levels of a factor), whole numbers (of any
    numeric dtype) from their smallest to their largest value, every integer in between included; anything else is refused
    as brms refuses it ("Monotonic predictors must be integers or ordered factors"), as is a predictor with fewer than three
    values (one step has no simplex to estimate) or one spanning more than ``COUNT_CAP`` steps."""
    term = f"mo({column})"
    label = f"column {column!r} of the monotonic term {term!r}"
    reject_complex(label, s)
    need = ("a monotonic predictor holds whole numbers or an ordered pandas Categorical (pd.Categorical(values, categories=[lowest, "
            "..., highest], ordered=True), brms's ordered factor)")
    if pd.api.types.is_bool_dtype(s):
        raise FormulaError(f"{label} is boolean; {need}")
    if s.isna().any():
        raise FormulaError(f"{label} has missing values; drop or impute them explicitly before fitting.")
    if isinstance(s.dtype, pd.CategoricalDtype):
        if not s.cat.ordered:
            raise FormulaError(f"{label} is an unordered pandas Categorical, whose category order means nothing; {need}")
        categories = tuple(_scalar(label, c, where="declared category") for c in s.cat.categories)
        if len(set(categories)) != len(categories):
            raise FormulaError(f"{label} has declared categories that collide after normalisation: {list(categories)}")
        if len({_kind(c) for c in s.cat.categories}) > 1:
            raise FormulaError(f"{label} mixes text labels and numeric codes in its declared categories; recode it explicitly")
        observed = set(_scalar(label, v) for v in s.astype(object).tolist())
        categories = tuple(c for c in categories if c in observed)       # brms codes the factor after dropping unused levels
        if len(categories) < 3:
            raise FormulaError(f"{label} has the categories {list(categories)} in the fitted rows; "
                               + (_TWO_VALUES if len(categories) == 2 else "a monotonic effect needs at least three"))
        return Monotonic(column, categories=categories)
    if not _is_numeric(s):
        raise FormulaError(f"{label} has dtype {s.dtype}; {need}. Text labels have no order: declare it with an ordered Categorical")
    v = s.to_numpy(dtype=float)
    bad = v[~np.isfinite(v) | (v != np.floor(v))]
    if bad.size:
        raise FormulaError(f"{label} has the value {float(bad[0])!r}; {need} (brms: 'Monotonic predictors must be integers or ordered factors')")
    lo, hi = int(v.min()), int(v.max())
    if hi - lo < 2:
        raise FormulaError(f"{label} takes the values {sorted({lo, hi})} in the fitted rows; "
                           + (_TWO_VALUES if hi > lo else "a monotonic effect needs at least three values"))
    if hi - lo > COUNT_CAP:
        raise FormulaError(f"{label} spans {lo} to {hi}, {hi - lo} steps, each with its own simplex entry; more than {COUNT_CAP} "
                           "steps is refused")
    return Monotonic(column, minimum=lo, maximum=hi)


@dataclass(frozen=True)
class Transform:
    """An in-formula transform of one numeric column (A05, docs/transforms.md): ``fn`` is ``"log"``, ``"exp"``,
    ``"sqrt"`` (``log(col)``), or the operator of an ``I()`` expression, ``"^"`` with an integer ``constant``
    (``I(col^2)``), ``"*"``, ``"+"`` or ``"-"`` with a numeric ``constant`` (``I(col * 2)``, ``I(col + 1)``,
    ``I(col - 1)``; ``constant_first`` for ``I(2 * col)`` and ``I(1 + col)``). The constant is kept as written, so
    the term is spelled as the user spelled it, up to whitespace."""

    fn: str
    column: str
    constant: str | None = None
    constant_first: bool = False

    @property
    def term(self) -> str:
        """The term as the canonical formula writes it: ``log(xpos)``, ``I(x^2)``, ``I(x * 2)``, ``I(2 * x)``."""
        return self.expression(self.column) if self.fn in ("log", "exp", "sqrt") else f"I({self.expression(self.column)})"

    def expression(self, value: str) -> str:
        """The transform written for a value: ``log(xpos_i)``, ``xpos_i^2``, ``2 * x_i``, ``x_i + 1``."""
        if self.fn in ("log", "exp", "sqrt"):
            return f"{self.fn}({value})"
        if self.fn == "^":
            return f"{value}^{self.constant}"
        return f"{self.constant} {self.fn} {value}" if self.constant_first else f"{value} {self.fn} {self.constant}"

    @property
    def brms_name(self) -> str:
        """brms's label for the coefficient: its ``rename()`` of the term, which drops spaces and parentheses and
        writes ``^`` as ``E``, ``*`` as ``MU``, ``+`` as ``P`` and ``-`` as ``M`` (``logxpos``, ``IxE2``, ``IxMU2``, ``IxP1``)."""
        out = self.term
        for old, new in ((" ", ""), ("(", ""), (")", ""), ("^", "E"), ("*", "MU"), ("+", "P"), ("-", "M")):
            out = out.replace(old, new)
        return out

    def values(self, df: pd.DataFrame, *, where: str = "column") -> np.ndarray:
        """The design column on ``df``: the transform of the raw column as a float array. Refuses a missing,
        non-numeric or missing-valued column, non-positive values under ``log`` and ``sqrt``, and a non-finite
        result; ``where`` names the frame in messages."""
        if self.column not in df.columns:
            raise FormulaError(f"{where} {self.column!r} of the term {self.term!r} is not in the data")
        s = df[self.column]
        reject_complex(f"{where} {self.column!r}", s)
        if not _is_numeric(s):
            raise FormulaError(f"{where} {self.column!r} has dtype {s.dtype}; the term {self.term!r} needs a numeric column")
        v = s.to_numpy(dtype=float)
        if not np.all(np.isfinite(v)):
            raise FormulaError(f"{where} {self.column!r} of the term {self.term!r} has missing or non-finite values")
        if self.fn in ("log", "sqrt") and np.any(v <= 0):
            i = int(np.flatnonzero(v <= 0)[0])
            raise FormulaError(f"{where} {self.column!r} must be positive for the term {self.term!r}; row {i} has {float(v[i])!r}")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if self.fn in ("log", "exp", "sqrt"):
                out = {"log": np.log, "exp": np.exp, "sqrt": np.sqrt}[self.fn](v)
            elif self.fn == "^":
                out = v ** float(int(self.constant))
            else:
                k = float(self.constant)
                out = v * k if self.fn == "*" else v + k if self.fn == "+" else v - k
        if not np.all(np.isfinite(out)):
            i = int(np.flatnonzero(~np.isfinite(out))[0])
            raise FormulaError(f"the term {self.term!r} is not finite at row {i} ({self.column} = {float(v[i])!r})")
        return out


def parse_transform(text: str) -> Transform | None:
    """The ``Transform`` a piece of formula spells, or None when it is not one of the supported forms."""
    text = text.strip()
    if m := _TRANSFORM_FN_RE.match(text):
        return Transform(m.group(1), m.group(2))
    if m := _TRANSFORM_POW_RE.match(text):
        return Transform("^", m.group(1), m.group(2))
    if m := _TRANSFORM_OP_RE.match(text):
        return Transform(m.group(2), m.group(1), m.group(3))
    if m := _TRANSFORM_OP_FIRST_RE.match(text):
        return Transform(m.group(2), m.group(3), m.group(1), constant_first=True)
    return None


@dataclass(frozen=True)
class PopTerm:
    """One population-level term as written in the formula: ``x`` (``"numeric"``), ``z`` (``"categorical"``,
    declared with its reference level), a two-way ``"interaction"``: ``x:z`` (numeric by categorical), ``x:w``
    (numeric by numeric) or ``z:h`` (categorical by categorical; A04, docs/interactions.md), or an in-formula
    ``"transform"`` of one numeric column, ``log(xpos)``, ``I(x^2)`` (A05, docs/transforms.md), whose single column
    is the evaluated transform and whose ``transform`` says how to evaluate it, or brms's ``"monotonic"`` effect
    ``mo(col)`` of an ordinal predictor (B03, docs/monotonic.md), which has no design column (``columns`` is empty): its
    coefficient (brms's ``bsp``) and simplex act through its own block, ``ModelSpec.monotonic(col)`` its coding.

    ``columns`` are the design-column labels the term expands to, Bambi/formulae's own, in design order:
    ``x``; ``z[l]`` per coefficient level ``l``; ``x:z[l]``; ``x:w``; ``z[l]:h[m]`` per cell, the first operand's
    level changing slowest. ``levels`` are the coefficient levels of a vector term (a categorical term's
    coefficient levels; an interaction's non-reference levels of its categorical side, or its cells ``"l, m"``
    labelled as Bambi labels them), empty for a scalar term. A categorical term's levels are known once the data
    are seen; until then it has no levels and its single column is its own name, a placeholder, as
    ``ModelSpec.columns`` always listed it. ``categorical`` names the operands that are declared categorical
    (``()``, ``("z",)``, ``("z", "h")``), which is what makes a term a vector term before its levels are known;
    when it is not given it is read from the kind, with the last operand of an interaction taken as the
    categorical side (the ``x:z`` form). The coefficients of the term are brms's ``b`` entries for its columns.
    """

    name: str
    kind: str
    columns: tuple[str, ...]
    levels: tuple[str, ...] = ()
    categorical: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.kind not in POP_TERM_KINDS:
            raise ValueError(f"PopTerm kind must be one of {list(POP_TERM_KINDS)}, got {self.kind!r}")
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "levels", tuple(self.levels))
        if self.kind == "transform" and parse_transform(self.name) is None:
            raise ValueError(f"PopTerm: {self.name!r} is not a supported in-formula transform")
        if self.kind == "monotonic" and (monotonic_column(self.name) is None or self.columns or self.levels):
            raise ValueError(f"PopTerm: a monotonic term is spelled 'mo(col)' and has no design column or level, got {self.name!r}")
        if self.categorical is None:
            default = {"numeric": (), "categorical": (self.name,), "interaction": (self.operands[-1],), "transform": (), "monotonic": ()}
            object.__setattr__(self, "categorical", default[self.kind])
        else:
            object.__setattr__(self, "categorical", tuple(self.categorical))
        if any(c not in self.operands for c in self.categorical):
            raise ValueError(f"PopTerm {self.name!r}: categorical operands {list(self.categorical)} are not among its operands {list(self.operands)}")

    @property
    def operands(self) -> tuple[str, ...]:
        """The data columns the term is built from: ``(x,)``, ``(z,)``, ``(x, z)`` (numeric first), ``(x, w)`` or
        ``(z, h)`` (main-effect order), ``(xpos,)`` for the transform ``log(xpos)``, ``(ord,)`` for ``mo(ord)``."""
        if self.kind == "transform":
            return (self.transform.column,)
        if self.kind == "monotonic":
            return (monotonic_column(self.name),)
        return tuple(self.name.split(":"))

    @property
    def is_monotonic(self) -> bool:
        """A monotonic term ``mo(col)`` (B03): no design column; its own coefficient and simplex, built by this library (Bambi
        never sees it)."""
        return self.kind == "monotonic"

    @property
    def transform(self) -> Transform:
        """How a ``"transform"`` term is evaluated (``parse_transform`` of its name); ``ValueError`` for other kinds."""
        tr = parse_transform(self.name) if self.kind == "transform" else None
        if tr is None:
            raise ValueError(f"{self.name!r} is not a transform term")
        return tr

    @property
    def bambi_name(self) -> str:
        """The term as handed to Bambi: a transform's evaluated column is named as written and quoted in backticks
        (the fitted frame carries it under that name), so Bambi's term and PyMC's variable carry the term's own
        name; every other term is itself."""
        return f"`{self.name}`" if self.kind == "transform" else self.name

    @property
    def is_vector(self) -> bool:
        """A term with one coefficient per level (a categorical term, or an interaction with a categorical
        operand), as opposed to one coefficient (a numeric term or a numeric-by-numeric interaction)."""
        return bool(self.categorical)


@dataclass(frozen=True)
class GroupTerm:
    """One group-level term ``(coefs | factor)``, brms's group ``m``: ``factor`` is the grouping column, or the
    derived factor ``g:h`` built from two columns (``operands``; its labels are the observed combinations joined
    with ``_`` and it lives in the fitted frame, and is handed to Bambi, under ``column`` = ``g_h``),
    ``coefs`` the population-level coefficients that vary by its level (brms ``M_m`` of them; ``("Intercept",)``
    for ``(1 | g)``, ``("Intercept", "x")`` for ``(1 + x || g)`` and ``(1 + x | g)``), ``correlated`` whether they
    share a correlation matrix (``|``, brms's default: an LKJ prior on the correlation of the coefficients, A02,
    docs/correlated-slopes.md) or are independent (``||``); moot for a single coefficient, so ``(1 || g)`` parses to
    the same term as ``(1 | g)`` and ``has_correlation`` is what says a correlation matrix exists. The level labels
    are not part of the term: they are known once the data are seen (``ModelSpec.levels_by_group``,
    ``GroupBlock.levels``). Its parameters are brms's ``sd_m`` (one per coefficient), ``z_m`` (the
    standardised effects by coefficient and level), ``r_m_c`` (the effects of coefficient ``c``) and, when
    correlated, ``L_m`` (the Cholesky factor of the correlation matrix).
    """

    factor: str
    coefs: tuple[str, ...] = ("Intercept",)
    correlated: bool = True

    def __post_init__(self):
        object.__setattr__(self, "coefs", tuple(self.coefs))

    @property
    def slopes(self) -> tuple[str, ...]:
        """The coefficients other than the varying intercept: the numeric terms whose slope varies by level."""
        return tuple(c for c in self.coefs if c != "Intercept")

    @property
    def has_correlation(self) -> bool:
        """Whether the term carries a correlation matrix: ``|`` with two or more coefficients (brms's ``L_m``,
        ``NC_m > 0`` with ``cor = TRUE``). A single coefficient has nothing to correlate with."""
        return self.correlated and len(self.coefs) > 1

    @property
    def canonical(self) -> str:
        """The term as the canonical formula writes it: ``(1 | g)``, ``(1 + x || g)`` for independent slopes,
        ``(1 + x | g)`` for correlated ones."""
        inner = " + ".join("1" if c == "Intercept" else c for c in self.coefs)
        bar = "|" if self.correlated or len(self.coefs) == 1 else "||"
        return f"({inner} {bar} {self.factor})"

    @property
    def operands(self) -> tuple[str, ...]:
        """The data columns the factor is read from: ``("g",)``, or ``("g", "h")`` for the derived factor ``g:h``."""
        return tuple(self.factor.split(":"))

    @property
    def derived(self) -> bool:
        """Whether the factor is built from two columns (``g:h``) rather than read from one."""
        return len(self.operands) > 1

    @property
    def column(self) -> str:
        """The fitted-frame column holding the factor's labels, which is also the name Bambi sees (``factor_column``)."""
        return factor_column(self.factor)

    @property
    def bambi_canonical(self) -> str:
        """``canonical`` as handed to Bambi: the factor under its column name and ``||`` as ``|`` (formulae has no ``||``)."""
        return self.canonical.replace(f" {self.factor})", f" {self.column})").replace("||", "|")


def factor_column(factor: str) -> str:
    """The frame column that holds a grouping factor's labels, and the name it is handed to Bambi under: the
    factor itself for a column, ``g_h`` for the derived factor ``g:h`` (formulae would read ``(1 | g:h)`` as the
    full Cartesian product of levels with ``:``-joined labels, which is not brms's model; docs/grouping-syntax.md)."""
    return factor.replace(":", "_")


def group_labels(factor: str, df: pd.DataFrame) -> pd.Series:
    """The string labels of grouping factor ``factor`` on ``df``: the column as text, or for the derived factor
    ``g:h`` the row's ``(g, h)`` pair joined with ``_``, brms's ``paste(g, h, sep = "_")`` over the observed
    combinations. Two distinct pairs whose joined labels coincide are refused rather than merged (brms merges them)."""
    parts = [df[c].astype(str) for c in factor.split(":")]
    if len(parts) == 1:
        return parts[0]
    labels = parts[0].str.cat(parts[1:], sep="_")
    seen: dict[str, tuple[str, ...]] = {}
    for lab, *pair in zip(labels.to_numpy(), *(p.to_numpy() for p in parts)):
        pair = tuple(str(v) for v in pair)
        if seen.setdefault(lab, pair) != pair:
            raise FormulaError(
                f"grouping factor {factor!r}: the combinations {seen[lab]} and {pair} both label as {lab!r} "
                "(labels are joined with '_', as brms does); rename the levels so that the labels are distinct"
            )
    return labels


@dataclass(frozen=True)
class Offset:
    """The offset term of a formula, brms's ``offset(col)`` or ``offset(log(col))``: the column (``log``-transformed
    when ``log`` is set) is added to the linear predictor as written, with no coefficient and no prior. It is not a
    population-level term: it has no design column and no entry in ``ModelSpec.columns`` or ``Model.coefficients``.
    """

    column: str
    log: bool = False

    @property
    def expression(self) -> str:
        """``log(col)`` or ``col``: what is inside ``offset()``."""
        return f"log({self.column})" if self.log else self.column

    @property
    def term(self) -> str:
        """The term as the canonical formula writes it: ``offset(log(col))``."""
        return f"offset({self.expression})"

    def values(self, df: pd.DataFrame, *, where: str = "column") -> np.ndarray:
        """The offset vector on ``df``: the column as a float array, logged when ``log`` is set. Refuses a
        non-numeric, missing-valued or (for ``log``) non-positive column; ``where`` names the frame in messages."""
        if self.column not in df.columns:
            raise FormulaError(f"{where} {self.column!r} of the offset term {self.term!r} is not in the data")
        s = df[self.column]
        reject_complex(f"{where} {self.column!r}", s)
        if not _is_numeric(s):
            raise FormulaError(f"{where} {self.column!r} has dtype {s.dtype}; the offset term {self.term!r} needs a numeric column")
        v = s.to_numpy(dtype=float)
        if not np.all(np.isfinite(v)):
            raise FormulaError(f"{where} {self.column!r} of the offset term {self.term!r} has missing or non-finite values")
        if self.log:
            if np.any(v <= 0):
                i = int(np.flatnonzero(v <= 0)[0])
                raise FormulaError(
                    f"{where} {self.column!r} must be positive for the offset term {self.term!r}; row {i} has {float(v[i])!r}"
                )
            return np.log(v)
        return v


ADDITION_KINDS = ("weights", "cens", "trunc")    # brms's response addition terms besides trials() that are parsed (B02)

# Addition terms that are parsed, represented and implemented (Gaussian family) but withheld, each with the reason its
# refusal carries (docs/addition-terms.md, divergences D24 and D25). Removing an entry un-parks the term;
# tests/test_addition_terms.py does that under monkeypatch to check the withheld forms against the frozen fixtures.
PARKED_ADDITIONS = MappingProxyType({
    "cens": ("brms's own gradient of a censored Gaussian model is not exact enough to certify it: for a left-censored row "
             "Stan's normal_lcdf differentiates through piecewise polynomial approximations (Abramowitz and Stegun 7.1.26 "
             "and Taylor fits), off by up to 4e-4 absolute, so the frozen fixture gauss_censored records gradients that "
             "differ from the exact ones by up to 2.5e-6 relative at 5 of its 8 points (the oracle's GRAD_REL is 1e-7), "
             "while its log densities agree to 3e-16; AutoBayes's censored likelihood is exact on both paths and is "
             "withheld until Andrew decides how such a feature is certified (docs/addition-terms.md, divergence D24)"),
    "trunc": ("brms's own gradient and expected value of a truncated Gaussian model are not exact enough to certify it far "
              "in the tail: Stan's normal_lccdf forms 1 - erf(z / sqrt(2)) by subtraction and brms's posterior_epred "
              "1 - pnorm(z), so at point 4 of the frozen fixture gauss_truncated, where a row's location sits 6.17 SD below "
              "the bound, brms's gradient is 4.2e-7 relative off the exact one (GRAD_REL 1e-7) and its posterior_epred up "
              "to 1.0e-3 (PRED_REL 1e-9), while its log densities agree to 7e-11; AutoBayes's truncated likelihood and mean "
              "are exact and are withheld until a decision (docs/addition-terms.md, divergence D25)"),
})


def _number_text(v: float) -> str:
    """A truncation bound as the canonical formula writes it: ``0`` for 0.0, ``2.5``, ``-1e-05``."""
    text = repr(float(v))
    return text[:-2] if text.endswith(".0") else text


@dataclass(frozen=True)
class AdditionTerm:
    """A response addition term of brms's left-hand side other than ``trials()`` (B02, docs/addition-terms.md), data
    of the formula rather than a population-level term: it has no coefficient, no prior and no design column.

    ``kind`` is ``"weights"`` (``weights(col)``: each row's log likelihood is multiplied by its weight), ``"cens"``
    (``cens(col)``: a censoring indicator per row, -1 left-, 0 un-, 1 right-censored) or ``"trunc"`` (``trunc(lb = a,
    ub = b)``: the response truncated to numeric bounds, ``None`` for an open side). ``column`` is the data column of
    ``weights`` and ``cens``. ``addition_data`` turns the terms into brms's Stan data on a frame."""

    kind: str
    column: str | None = None
    lb: float | None = None
    ub: float | None = None

    @property
    def term(self) -> str:
        """The term as the canonical formula writes it: ``weights(w)``, ``cens(c)``, ``trunc(lb = 0)``, ``trunc(lb = 0, ub = 5)``."""
        if self.kind == "trunc":
            sides = [f"{side} = {_number_text(v)}" for side, v in (("lb", self.lb), ("ub", self.ub)) if v is not None]
            return f"trunc({', '.join(sides)})"
        return f"{self.kind}({self.column})"

    @property
    def columns(self) -> tuple[str, ...]:
        """The data columns the term reads (none for ``trunc``, whose bounds are numbers)."""
        return () if self.column is None else (self.column,)


def _parse_trunc(args: list[str], piece: str, formula: str) -> AdditionTerm:
    """``trunc(lb = a, ub = b)``, ``trunc(lb = a)``, ``trunc(ub = b)`` or brms's positional ``trunc(a, b)``, with numeric
    bounds (``Inf`` / ``-Inf`` for an open side, brms's defaults)."""
    bounds: dict[str, float] = {}
    for i, arg in enumerate(args):
        km = re.fullmatch(rf"({_IDENT})\s*=\s*(.+)", arg, re.S)
        side, value = (km.group(1), km.group(2).strip()) if km else (("lb", "ub")[i] if i < 2 else None, arg)
        if side not in ("lb", "ub") or side in bounds:
            raise FormulaError(f"{formula!r}: {piece!r}: trunc() takes the bounds lb and ub, each at most once")
        if re.fullmatch(_NUMBER, value):
            bounds[side] = float(value)
        elif value in ("Inf", "-Inf"):
            bounds[side] = math.inf if value == "Inf" else -math.inf
        else:
            raise FormulaError(
                f"{formula!r}: {piece!r}: a truncation bound must be a number; per-row bounds read from a column "
                f"({value!r}) are not part of this slice (docs/addition-terms.md)"
            )
    lb, ub = bounds.get("lb", -math.inf), bounds.get("ub", math.inf)
    if not lb < ub:
        raise FormulaError(f"{formula!r}: {piece!r}: the truncation bounds must satisfy lb < ub (brms refuses lb >= ub too)")
    if lb == -math.inf and ub == math.inf:
        raise FormulaError(f"{formula!r}: {piece!r} has no finite bound and truncates nothing; drop it")
    return AdditionTerm("trunc", lb=None if lb == -math.inf else lb, ub=None if ub == math.inf else ub)


def _parse_additions(text: str, formula: str) -> tuple[str | None, tuple[AdditionTerm, ...]]:
    """The addition terms after the ``|`` of brms's left-hand side, joined with ``+``: the ``trials()`` column (or None)
    and the other terms in the order written. Each term at most once; anything else is refused naming the term."""
    trials, additions, seen = None, [], set()
    for piece in _split_top_level(text):
        m = _ADDITION_CALL_RE.match(piece)
        kind = m.group(1) if m else None
        if kind not in ("trials", *ADDITION_KINDS):
            raise FormulaError(
                f"{formula!r}: {piece!r} is not a supported response addition term. brms's left-hand side "
                "'response | term + term' takes weights(col), cens(col) and trunc(lb = a, ub = b) here "
                "(docs/addition-terms.md); a binomial response is exactly 'successes | trials(n)', with any other terms after it."
            )
        if kind in seen:
            raise FormulaError(f"{formula!r}: {kind}() appears more than once on the left-hand side")
        seen.add(kind)
        args = [a.strip() for a in _split_top_level(m.group(2), ",")] if m.group(2).strip() else []
        if kind == "trunc":
            additions.append(_parse_trunc(args, piece, formula))
            continue
        if len(args) != 1 or not re.fullmatch(_IDENT, args[0]):
            hint = {"weights": "; brms's weights(col, scale = TRUE), which rescales the weights to sum to the number of rows, "
                               "is not supported: rescale the column",
                    "cens": "; interval censoring, cens(col, y2), is not part of this slice", "trials": ""}[kind]
            raise FormulaError(f"{formula!r}: {piece!r}: {kind}() takes one column name{hint}")
        if kind == "trials":
            trials = args[0]
        else:
            additions.append(AdditionTerm(kind, column=args[0]))
    return trials, tuple(additions)


def _check_additions(additions: tuple[AdditionTerm, ...], fam, formula: str) -> None:
    """Refuse a withheld addition term (``PARKED_ADDITIONS``, with its reason), a term the family's likelihood does not
    take (``FamilySpec.addition_terms``, naming the families that take it), and ``cens()`` with ``trunc()``: brms writes a
    censored row of a truncated model as the untruncated distribution's tail minus the truncation mass, which is not the
    censored probability of the truncated distribution (docs/addition-terms.md)."""
    from autobayes.families import FAMILIES

    for a in additions:
        if a.kind in PARKED_ADDITIONS:
            raise FormulaError(f"{formula!r}: {a.term!r}: brms's {a.kind}() is withheld: {PARKED_ADDITIONS[a.kind]}")
        if a.kind not in fam.addition_terms:
            offering = [name for name, other in FAMILIES.items() if a.kind in other.addition_terms]
            raise FormulaError(
                f"{formula!r}: {a.term!r} is not available for the {fam.name} family; brms's {a.kind}() is verified here "
                f"for the {', '.join(offering) or 'no'} famil{'y' if len(offering) == 1 else 'ies'} (docs/addition-terms.md)"
            )
    if {"cens", "trunc"} <= {a.kind for a in additions}:
        raise FormulaError(
            f"{formula!r}: cens() together with trunc() is not supported: brms adds a censored row of a truncated model as "
            "the untruncated tail log P(Y <= y) or log P(Y > y) minus the log mass between the bounds, which is not the "
            "censored probability of the truncated distribution (docs/addition-terms.md, divergence D25)"
        )


def _weights_values(column: str, s: pd.Series, where: str) -> np.ndarray:
    reject_complex(f"{where} {column!r}", s)
    if not _is_numeric(s):
        raise FormulaError(f"{where} {column!r} has dtype {s.dtype}; the weights of 'weights({column})' must be numeric")
    v = s.to_numpy(dtype=float)
    if not np.all(np.isfinite(v)):
        raise FormulaError(f"{where} {column!r} of 'weights({column})' has missing or non-finite values")
    if np.any(v <= 0):
        i = int(np.flatnonzero(v <= 0)[0])
        raise FormulaError(
            f"{where} {column!r}: weights must be positive; row {i} has {float(v[i])!r} (brms accepts a zero weight, which "
            "removes the row from the likelihood but not from the centring means and level sets: drop the row instead; "
            "docs/addition-terms.md, divergence D26)"
        )
    return v


def _cens_values(column: str, s: pd.Series, where: str) -> np.ndarray:
    reject_complex(f"{where} {column!r}", s)
    if not _is_numeric(s):
        raise FormulaError(
            f"{where} {column!r} has dtype {s.dtype}; the censoring indicator of 'cens({column})' must be numeric, -1 (left), "
            "0 (none) or 1 (right); recode brms's labels 'left' / 'none' / 'right' and TRUE / FALSE explicitly"
        )
    v = s.to_numpy(dtype=float)
    if not np.all(np.isfinite(v)):
        raise FormulaError(f"{where} {column!r} of 'cens({column})' has missing or non-finite values")
    if np.any(v == 2):
        raise FormulaError(f"{where} {column!r}: code 2 (interval censoring) needs brms's cens({column}, y2), which is not part of this slice")
    bad = v[~np.isin(v, (-1.0, 0.0, 1.0))]
    if bad.size:
        raise FormulaError(f"{where} {column!r}: censoring codes must be -1 (left), 0 (none) or 1 (right); found {float(bad[0])!r}")
    return v.astype(np.int64)


def addition_data(spec: "ModelSpec", df: pd.DataFrame, *, where: str = "column") -> dict[str, np.ndarray]:
    """brms's Stan data of the formula's addition terms on ``df`` (B02), under brms's names: ``weights`` (positive finite
    weights), ``cens`` (-1, 0 or 1 per row, int64) and ``lb`` / ``ub`` (the truncation bounds per row, ``-inf`` / ``inf``
    on an open side, as brms writes both whenever ``trunc()`` is present). Each is checked here: a response outside the
    truncation bounds is refused, as brms refuses it. Empty without addition terms."""
    out: dict[str, np.ndarray] = {}
    for a in spec.additions:
        if a.kind == "weights":
            out["weights"] = _weights_values(a.column, df[a.column], where)
        elif a.kind == "cens":
            out["cens"] = _cens_values(a.column, df[a.column], where)
        else:
            lb = -math.inf if a.lb is None else a.lb
            ub = math.inf if a.ub is None else a.ub
            y = df[spec.response].to_numpy(dtype=float)
            outside = np.flatnonzero((y < lb) | (y > ub))
            if outside.size:
                i = int(outside[0])
                raise FormulaError(
                    f"response {spec.response!r} at row {i} is {float(y[i])!r}, outside the bounds of {a.term!r} "
                    f"({outside.size} row(s) outside); brms refuses responses outside the truncation bounds too"
                )
            out["lb"], out["ub"] = np.full(len(df), lb), np.full(len(df), ub)
    return out


def bambi_group_name(factor: str, coef: str) -> str:
    """Bambi's name of one group-level coefficient: ``1|g`` for the varying intercept of ``g``, ``x|g`` for the
    slope of ``x``; a derived factor ``g:h`` appears under its column name, ``1|g_h``. PyMC's variables are this
    name (the effects per level), ``<name>_sigma`` and ``<name>_offset``."""
    return f"{'1' if coef == 'Intercept' else coef}|{factor_column(factor)}"


def sd_tag(factor: str, coef: str) -> str:
    """How messages, ``describe()`` and fit records label one group SD: ``g`` for the varying intercept of ``g``
    (``sd:g``, ``sd_g``, as they have always read) and ``x|g`` for the slope of ``x`` (``sd:x|g``, ``sd_x|g``)."""
    return factor if coef == "Intercept" else f"{coef}|{factor}"


def _find_categorical(categoricals, name: str) -> Categorical:
    for c in categoricals:
        if c.name == name:
            return c
    raise KeyError(name)


def interaction_cells(z: Categorical, h: Categorical) -> tuple[tuple[str, str], ...]:
    """The cells of the categorical-by-categorical interaction ``z:h``: every pair of non-reference levels, the
    first operand's level changing slowest (the order Bambi/formulae give the columns ``z[l]:h[m]``). R codes
    the interaction of two treatment-coded factors whose main effects are present the same way."""
    return tuple((a, b) for a in z.contrasts for b in h.contrasts)


def pop_term(name: str, kind: str, categoricals=()) -> PopTerm:
    """The population-level term ``name`` of the given kind, with its coefficient levels and design columns
    read from the declared categorical terms: a categorical term's coefficient levels; for an interaction, the
    non-reference levels of its one categorical operand (``x:z[l]``), the cells of its two categorical operands
    (``z[l]:h[m]``, levels ``"l, m"``), or no levels at all when both operands are numeric (``x:w``); none for a
    numeric term; a monotonic term ``mo(col)`` has neither (B03)."""
    if kind in ("numeric", "transform"):
        return PopTerm(name, kind, (name,), (), ())
    if kind == "monotonic":
        return PopTerm(name, kind, (), (), ())
    if kind == "categorical":
        levels = _find_categorical(categoricals, name).coefficients
        return PopTerm(name, kind, tuple(f"{name}[{l}]" for l in levels) if levels else (name,), levels, (name,))
    if kind != "interaction":
        raise ValueError(f"PopTerm kind must be one of {list(POP_TERM_KINDS)}, got {kind!r}")
    operands = tuple(name.split(":"))
    cats = tuple(c for c in operands if any(k.name == c for k in categoricals))
    if not cats:                                                   # x:w: the product column, one coefficient
        return PopTerm(name, kind, (name,), (), ())
    if len(cats) == 1:                                             # x:z: the numeric column times each indicator
        levels = _find_categorical(categoricals, cats[0]).contrasts
        return PopTerm(name, kind, tuple(f"{name}[{l}]" for l in levels) if levels else (name,), levels, cats)
    z, h = (_find_categorical(categoricals, c) for c in cats)      # z:h: one indicator product per cell
    cells = interaction_cells(z, h)
    levels = tuple(f"{a}, {b}" for a, b in cells)
    if len(set(levels)) != len(levels):
        dup = next(l for l in levels if levels.count(l) > 1)
        raise FormulaError(
            f"interaction {name!r}: two cells both label as {dup!r} (cells are labelled 'level, level', as Bambi labels "
            "them); rename the levels so that the cell labels are distinct"
        )
    columns = tuple(f"{z.name}[{a}]:{h.name}[{b}]" for a, b in cells) if cells else (name,)
    return PopTerm(name, kind, columns, levels, cats)


def pop_term_design(term: PopTerm, df: pd.DataFrame, categoricals=()) -> np.ndarray:
    """The raw design block of one population-level term on ``df``: ``(n, len(term.columns))``, one column per
    entry of ``term.columns``. A numeric term is its column as given; a categorical term one 0/1 indicator
    per coefficient level (reference rows all zero); an interaction ``x:z`` the numeric column times each
    indicator, ``x:w`` the elementwise product of the two columns, ``z:h`` the product of the two indicators
    of each cell, which needs the declared ``Categorical`` terms for the operands' non-reference levels.
    Categorical columns are read as string labels (a fitted frame holds them recoded). A monotonic term ``mo(col)`` has no
    column in the design (``(n, 0)``): its block is its codes, ``Monotonic.codes`` (B03)."""
    n = len(df)
    if term.kind == "numeric":
        return df[term.name].to_numpy(dtype=float)[:, None]
    if term.kind == "monotonic":
        return np.zeros((n, 0))
    if term.kind == "transform":                                 # evaluated from the raw column, never read from a column of its name
        return term.transform.values(df)[:, None]
    if term.kind == "interaction" and not term.categorical:
        a, b = (df[c].to_numpy(dtype=float) for c in term.operands)
        return (a * b)[:, None]
    if len(term.categorical) == 2:
        if not categoricals:
            raise ValueError(f"pop_term_design needs the declared Categorical terms to build the cells of {term.name!r}")
        z, h = (_find_categorical(categoricals, c) for c in term.categorical)
        lz, lh = df[z.name].astype(str).to_numpy(), df[h.name].astype(str).to_numpy()
        cells = interaction_cells(z, h)
        if len(cells) != len(term.levels):
            raise ValueError(f"{term.name!r} has {len(term.levels)} cells but its categorical terms give {len(cells)}")
        if not cells:
            return np.zeros((n, 0))
        return np.column_stack([(lz == a) & (lh == b) for a, b in cells]).astype(float)
    (z,) = term.categorical
    numeric = [c for c in term.operands if c != z]
    labels = df[z].astype(str).to_numpy()
    x = df[numeric[0]].to_numpy(dtype=float) if numeric else np.ones(n)
    if not term.levels:
        return np.zeros((n, 0))
    return np.column_stack([x * (labels == lvl) for lvl in term.levels])


def _flat_pop_terms(predictors, terms, categoricals, interactions) -> tuple[PopTerm, ...]:
    """The term model of the flat description (``predictors``, ``terms``, ``categoricals``, ``interactions``)
    that the original ``ModelSpec`` constructor takes."""
    names = tuple(terms) or tuple(predictors)
    cat_names = {c.name for c in categoricals}
    inter_names = {f"{x}:{z}" for x, z in interactions}
    return tuple(pop_term(t, "categorical" if t in cat_names else "interaction" if t in inter_names else "numeric",
                          categoricals) for t in names)


class _GroupLevels(Mapping):
    """Owned immutable level tuples, with ordinary copy/pickle reconstruction."""

    __slots__ = ("_items",)

    def __init__(self, items):
        object.__setattr__(self, "_items", tuple((g, tuple(v)) for g, v in items))

    def __setattr__(self, name, value):
        raise TypeError("group levels are immutable")

    def __delattr__(self, name):
        raise TypeError("group levels are immutable")

    def __getitem__(self, key):
        for group, levels in self._items:
            if group == key:
                return levels
        raise KeyError(key)

    def __iter__(self):
        return (group for group, _ in self._items)

    def __len__(self):
        return len(self._items)

    def __reduce__(self):
        return type(self), (self._items,)

    def __repr__(self):
        return repr(dict(self))


@dataclass(frozen=True, init=False)
class ModelSpec:
    """A parsed formula. ``pop_terms`` and ``group_terms`` are the source of truth; ``predictors``, ``terms``,
    ``columns``, ``categorical_names``, ``interactions``, ``groups`` and the per-term lookups are views of them.
    ``categoricals`` carries what a categorical term needs beyond its name (reference, levels, coding) and
    ``levels_by_group`` the fitted levels of each grouping factor, both filled in once the data are seen.
    ``offset`` is the formula's ``offset()`` term, or None (docs/offset.md). ``additions`` are the response addition
    terms after ``|`` other than ``trials()`` (``weights()``, ``cens()``, ``trunc()``; B02, docs/addition-terms.md).
    ``dpars`` are the formulas of distributional parameters (B01, docs/distributional.md: brms's ``bf(y ~ x, sigma ~ z)``),
    each a ``ModelSpec`` of its own whose ``response`` is the parameter's brms name (``sigma``, ``shape``, ``phi``) and
    whose population-level terms make that parameter's linear predictor; empty for a single formula. ``monotonics`` holds
    the coding of each monotonic term's column (B03, ``mo(col)``; ``Monotonic``), in term order, known once the data are
    seen, as ``categoricals`` does for the categorical terms."""

    formula: str
    response: str
    intercept: bool
    levels_by_group: Mapping[str, tuple[str, ...]]
    trials: str | None
    rhs: str
    categoricals: tuple[Categorical, ...]
    pop_terms: tuple[PopTerm, ...]
    group_terms: tuple[GroupTerm, ...]
    offset: Offset | None
    additions: tuple[AdditionTerm, ...]
    dpars: tuple["ModelSpec", ...]
    monotonics: tuple[Monotonic, ...]

    def __init__(self, formula, response, predictors=(), group=None, intercept=True,
                 levels=(), trials=None, rhs="", terms=(), categoricals=(), interactions=(),
                 *, groups=None, levels_by_group=None, pop_terms=None, group_terms=None, offset=None, additions=(), dpars=(),
                 monotonics=None):
        """Keep the original positional constructor: without ``pop_terms`` / ``group_terms`` the term model is
        built from the flat ``predictors``, ``terms``, ``categoricals``, ``interactions`` and ``group(s)``.
        Terms given as mappings (``dataclasses.asdict`` output) are rebuilt, the distributional formulas and the
        monotonic codings too; without ``monotonics`` each monotonic term's coding is its column alone (not yet seen)."""
        categoricals = tuple(c if isinstance(c, Categorical) else Categorical(**c) for c in categoricals)
        if offset is not None and not isinstance(offset, Offset):
            offset = Offset(**offset)
        additions = tuple(a if isinstance(a, AdditionTerm) else AdditionTerm(**a) for a in additions)
        dpars = tuple(d if isinstance(d, ModelSpec) else ModelSpec(**d) for d in dpars)
        if group_terms is not None:
            if groups is not None or group is not None:
                raise ValueError("supply group_terms or group/groups, not both")
            group_terms = tuple(t if isinstance(t, GroupTerm) else GroupTerm(**t) for t in group_terms)
        else:
            if groups is not None and group is not None:
                raise ValueError("supply group or groups, not both")
            groups = tuple(groups) if groups is not None else (() if group is None else (group,))
            group_terms = tuple(GroupTerm(g) for g in groups)
        factors = tuple(t.factor for t in group_terms)
        if len(set(factors)) != len(factors):
            raise ValueError("duplicate grouping factors")
        if pop_terms is not None:
            if predictors or terms or interactions:
                raise ValueError("supply pop_terms or predictors/terms/interactions, not both")
            pop_terms = tuple(t if isinstance(t, PopTerm) else PopTerm(**t) for t in pop_terms)
        else:
            pop_terms = _flat_pop_terms(predictors, terms, categoricals, interactions)
        if levels_by_group is not None and levels:
            raise ValueError("supply levels or levels_by_group, not both")
        if levels and len(factors) != 1:
            raise ValueError("singular levels require one grouping factor")
        level_map = ({g: tuple(levels) for g in factors} if levels_by_group is None
                     else {g: tuple(v) for g, v in levels_by_group.items()})
        if set(level_map) != set(factors):
            raise ValueError("levels_by_group must cover exactly the grouping factors")
        mo_columns = tuple(t.operands[0] for t in pop_terms if t.is_monotonic)
        if monotonics is None:
            monotonics = tuple(Monotonic(c) for c in mo_columns)
        monotonics = tuple(m if isinstance(m, Monotonic) else Monotonic(**m) for m in monotonics)
        if tuple(m.column for m in monotonics) != mo_columns:
            raise ValueError(f"monotonics must give the coding of exactly the monotonic terms' columns {list(mo_columns)}, in order")
        values = dict(formula=formula, response=response, intercept=intercept,
                      levels_by_group=_GroupLevels((g, level_map[g]) for g in factors),
                      trials=trials, rhs=rhs, categoricals=categoricals,
                      pop_terms=pop_terms, group_terms=group_terms, offset=offset, additions=additions, dpars=dpars,
                      monotonics=monotonics)
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def __hash__(self):
        # Preserve hashability of the frozen specification; factor metadata is ordered.
        return hash((self.formula, self.response, self.intercept,
                     tuple(self.levels_by_group[g] for g in self.groups), self.trials, self.rhs,
                     self.categoricals, self.pop_terms, self.group_terms, self.offset, self.additions)
                    + ((self.dpars,) if self.dpars else ()) + ((self.monotonics,) if self.monotonics else ()))

    def addition(self, kind: str) -> AdditionTerm | None:
        """The addition term of ``kind`` (``"weights"``, ``"cens"``, ``"trunc"``), or None when the formula has none."""
        return next((a for a in self.additions if a.kind == kind), None)

    # -- distributional formulas (B01) ---------------------------------------------------------
    @property
    def dpar_names(self) -> tuple[str, ...]:
        """The distributional parameters that have their own formula, by brms name, in the order written."""
        return tuple(d.response for d in self.dpars)

    def dpar(self, name: str) -> "ModelSpec":
        """The formula of distributional parameter ``name`` (``KeyError`` when it has none)."""
        for d in self.dpars:
            if d.response == name:
                return d
        raise KeyError(name)

    @property
    def formulas(self) -> tuple[str, ...]:
        """Every formula as written: the response formula, then each distributional parameter's."""
        return (self.formula, *(d.formula for d in self.dpars))

    @property
    def all_categoricals(self) -> tuple[Categorical, ...]:
        """The declared categorical columns of every formula, each once (the response formula's first, then those only a
        distributional formula uses): what the fitted frame recodes."""
        out = list(self.categoricals)
        for d in self.dpars:
            out += [c for c in d.categoricals if all(c.name != o.name for o in out)]
        return tuple(out)

    @property
    def model_columns(self) -> tuple[str, ...]:
        """The data columns the population-level terms of every formula read, without repeats (``pop_columns`` of the
        response formula, then of each distributional formula)."""
        return tuple(dict.fromkeys([*self.pop_columns, *(c for d in self.dpars for c in d.pop_columns)]))

    # -- views of the term model -------------------------------------------------------------
    @property
    def terms(self) -> tuple[str, ...]:
        """Population-level term names in design order (main effects, then interactions)."""
        return tuple(t.name for t in self.pop_terms)

    @property
    def predictors(self) -> tuple[str, ...]:
        """The numeric terms."""
        return tuple(t.name for t in self.pop_terms if t.kind == "numeric")

    @property
    def interactions(self) -> tuple[tuple[str, str], ...]:
        """The operand pair of every interaction term, in term order: ``(numeric, categorical)`` for ``x:z``,
        ``(x, w)`` or ``(z, h)`` in main-effect order otherwise."""
        return tuple(t.operands for t in self.pop_terms if t.kind == "interaction")

    @property
    def pop_columns(self) -> tuple[str, ...]:
        """The data columns the population-level terms read, in formula order without repeats: the numeric and
        categorical main effects and the raw columns inside transforms (an interaction reads its main effects'
        columns)."""
        return tuple(dict.fromkeys(c for t in self.pop_terms for c in t.operands))

    @property
    def transform_terms(self) -> tuple[PopTerm, ...]:
        """The in-formula transform terms (A05), in formula order."""
        return tuple(t for t in self.pop_terms if t.kind == "transform")

    # -- monotonic terms (B03) -------------------------------------------------------------
    @property
    def mo_terms(self) -> tuple[PopTerm, ...]:
        """The monotonic terms ``mo(col)`` (B03), in formula order: population-level terms with no design column."""
        return tuple(t for t in self.pop_terms if t.is_monotonic)

    def monotonic(self, column: str) -> Monotonic:
        """The coding of the monotonic term of ``column`` (``KeyError`` when the formula has no ``mo(column)``)."""
        for m in self.monotonics:
            if m.column == column:
                return m
        raise KeyError(column)

    @property
    def bambi_terms(self) -> tuple[PopTerm, ...]:
        """The population-level terms Bambi builds: all but the monotonic ones, which this library adds to the graph (B03)."""
        return tuple(t for t in self.pop_terms if not t.is_monotonic)

    @property
    def coefficient_labels(self) -> tuple[str, ...]:
        """brms's class ``b`` of this formula, one label per coefficient in term order: each design column (``columns``) and
        each monotonic term by its name (``mo(ord)``, brms's ``bsp``), which has a coefficient but no design column."""
        return tuple(label for t in self.pop_terms for label in (t.columns if not t.is_monotonic else (t.name,)))

    @property
    def groups(self) -> tuple[str, ...]:
        """Grouping factors in formula order."""
        return tuple(t.factor for t in self.group_terms)

    def pop_term(self, name: str) -> PopTerm:
        for t in self.pop_terms:
            if t.name == name:
                return t
        raise KeyError(name)

    def group_term_of(self, factor: str) -> GroupTerm:
        for t in self.group_terms:
            if t.factor == factor:
                return t
        raise KeyError(factor)

    @property
    def group_coefficients(self) -> tuple[tuple[str, str], ...]:
        """``(factor, coefficient)`` per group-level coefficient in formula order: the keys of
        ``Design.group_blocks`` and ``Model.group_coefficient_names``."""
        return tuple((t.factor, coef) for t in self.group_terms for coef in t.coefs)

    @property
    def has_slopes(self) -> bool:
        """Whether any group-level term has a varying slope."""
        return any(t.slopes for t in self.group_terms)

    @property
    def correlated_terms(self) -> tuple[GroupTerm, ...]:
        """The group-level terms whose coefficients share a correlation matrix, ``(1 + x | g)`` (A02), in formula order."""
        return tuple(t for t in self.group_terms if t.has_correlation)

    @property
    def has_group(self) -> bool:
        return bool(self.group_terms)

    @property
    def group(self) -> str | None:
        if len(self.groups) > 1:
            raise ValueError("group is ambiguous with multiple grouping factors; use groups")
        return self.groups[0] if self.groups else None

    @property
    def levels(self) -> tuple[str, ...]:
        if len(self.groups) > 1:
            raise ValueError("levels is ambiguous with multiple grouping factors; use levels_by_group")
        return self.levels_by_group[self.groups[0]] if self.groups else ()

    @property
    def group_term(self) -> str:
        """Bambi's name of the single varying intercept, ``1|g``."""
        if self.group is None:
            raise ValueError("this model has no group-level term")
        return f"1|{self.group}"

    @property
    def categorical_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.pop_terms if t.kind == "categorical")

    def categorical(self, name: str) -> Categorical:
        return _find_categorical(self.categoricals, name)

    @property
    def interaction_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.pop_terms if t.kind == "interaction")

    def interaction_of(self, term: str) -> tuple[str, str]:
        """The operands of an interaction term (``(numeric, categorical)`` for ``x:z``); ``KeyError`` for any other name."""
        t = self.pop_term(term)
        if t.kind != "interaction":
            raise KeyError(term)
        return t.operands

    def is_vector_term(self, term: str) -> bool:
        """A categorical or interaction term: coefficient levels given by levels_of."""
        return any(t.name == term and t.is_vector for t in self.pop_terms)

    def levels_of(self, term: str) -> tuple[str, ...]:
        """Coefficient levels of a vector-valued term: a categorical term's coefficient levels, the non-reference
        levels of an interaction's categorical side, or the cells ``"l, m"`` of a categorical-by-categorical
        interaction. ``()`` for a scalar term."""
        return next((t.levels for t in self.pop_terms if t.name == term), ())

    def term_columns(self, term: str) -> tuple[str, ...]:
        """Design-column labels of one term, Bambi/formulae's own: ``x``; ``z[l]``; ``x:z[l]``; ``x:w``; ``z[l]:h[m]``."""
        return next((t.columns for t in self.pop_terms if t.name == term), (term,))

    @property
    def columns(self) -> tuple[str, ...]:
        """Design-column labels in design order: the terms' columns, one after another (a categorical
        term expanded into ``z[level]`` dummies, reference omitted; an interaction into ``x:z[level]``,
        ``x:w`` or ``z[l]:h[m]``)."""
        return tuple(col for t in self.pop_terms for col in t.columns)

    @property
    def column_terms(self) -> tuple[tuple[str, str, int | None], ...]:
        """``(column label, term, index within the term or None for a numeric term)`` per column (a monotonic term has
        none)."""
        out: list[tuple[str, str, int | None]] = []
        for t in self.pop_terms:
            if t.is_monotonic:
                continue
            if t.levels:
                out.extend((col, t.name, j) for j, col in enumerate(t.columns))
            else:
                out.append((t.name, t.name, None))
        return tuple(out)

    @property
    def canonical_formula(self) -> str:
        """The formula as expanded: main effects, then interactions (``numeric:categorical``; otherwise the two
        operands in main-effect order), then the offset and the group terms; on the left, ``trials()`` first and the
        other addition terms in the order written."""
        extra = ([f"trials({self.trials})"] if self.trials is not None else []) + [a.term for a in self.additions]
        lhs = self.response if not extra else f"{self.response} | {' + '.join(extra)}"
        return f"{lhs} ~ {self.rhs}"

    @property
    def bambi_formula(self) -> str:
        """What is handed to Bambi: the canonical right-hand side rebuilt term by term; brms's ``k | trials(n)``
        becomes Bambi's ``p(k, n)``, ``(1 + x || g)`` becomes ``(1 + x | g)``, which Bambi (formulae has no
        ``||``) builds as the independent terms ``1|g`` and ``x|g``, a derived factor ``g:h`` appears under its
        column name, ``(1 | g_h)``, and a transform term under its own name in backticks, the evaluated column
        the fitted frame carries; ``Model._extract_design`` verifies all of that. The other addition terms never reach
        Bambi's formula: they act on the likelihood (``likelihoods.addition_family``)."""
        if self.trials is None:
            return f"{self.response} ~ {self.bambi_rhs}"
        return f"p({self.response}, {self.trials}) ~ {self.bambi_rhs}"

    @property
    def bambi_rhs(self) -> str:
        """The right-hand side of ``bambi_formula``, rebuilt term by term; a distributional formula (B01) is handed to Bambi
        as ``<parameter> ~ <bambi_rhs>`` under Bambi's name for the parameter (``alpha`` for brms's ``shape``). A monotonic
        term (B03) is not handed to Bambi: the model adds it to the linear predictor Bambi builds."""
        parts = ([] if self.intercept else ["0"]) + [t.bambi_name for t in self.bambi_terms]
        parts += ([self.offset.term] if self.offset is not None else []) + [t.bambi_canonical for t in self.group_terms]
        return " + ".join(parts) if parts else "1"


def _split_top_level(rhs: str, sep: str = "+") -> list[str]:
    """Split on ``sep`` outside parentheses."""
    parts, depth, buf = [], 0, []
    for ch in rhs:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise FormulaError(f"unbalanced parentheses in {rhs!r}")
        if ch == sep and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if depth != 0:
        raise FormulaError(f"unbalanced parentheses in {rhs!r}")
    parts.append("".join(buf).strip())
    return parts


def _parse_group_term(inner: str, bars: str, factor: str, piece: str, response: str, trials: str | None) -> list[GroupTerm]:
    """The ``GroupTerm``s of one ``(inner | factor)`` or ``(inner || factor)`` piece: the varying intercept
    ``(1 | g)`` (``(1 || g)`` is the same term, one coefficient having nothing to correlate with), the
    independent varying intercept and numeric slopes ``(1 + x [+ w ...] || g)`` (A01), or the correlated ones
    ``(1 + x [+ w ...] | g)`` (A02, an LKJ prior on their correlation matrix). The factor is a column ``g``,
    the derived factor ``g:h`` (the observed combinations of two columns: one term) or brms's nesting shorthand
    ``g/h`` (``(1 | g) + (1 | g:h)``: two terms, in that order). Everything else brms accepts here is refused
    with a message naming the construct. Whether a slope is a numeric population-level term of the formula is
    checked by ``parse_formula`` once every term is read."""
    if "|" in factor:
        raise FormulaError(f"{piece!r}: correlation IDs are not supported")
    if factor.startswith("mm("):
        raise FormulaError(f"{piece!r}: multimembership grouping is not supported")
    if re.match(r"gr\s*\(", factor):                             # brms's gr(g, cor = FALSE, by = ..., ...)
        raise FormulaError(
            f"{piece!r}: brms's gr() grouping options are not supported; write the grouping column itself, with "
            "'||' for independent effects (gr(g, cor = FALSE)) or '|' for correlated ones"
        )
    if "(" in factor:
        raise FormulaError(f"{piece!r}: transformed grouping factors are not supported; use a column name")
    if "/" in factor and ":" in factor:
        raise FormulaError(
            f"{piece!r}: mixing '/' and ':' in a grouping factor ({factor!r}) is not supported in this slice; write the "
            "group terms out, '(1 | g) + (1 | g:h)'"
        )
    sep = "/" if "/" in factor else ":" if ":" in factor else None
    operands = [p.strip() for p in factor.split(sep)] if sep else [factor]
    if len(operands) > 2:
        raise FormulaError(
            f"{piece!r}: only two factors can be combined in a grouping factor in this slice ({factor!r} names "
            f"{len(operands)} columns); three-way nesting or interaction of grouping factors is not supported"
        )
    for p in operands:
        if not re.fullmatch(_IDENT, p):
            raise FormulaError(f"{piece!r}: {p!r} is not a column name" if sep else f"{piece!r}: grouping factor must be a single column name")
    if len(operands) == 2 and operands[0] == operands[1]:
        raise FormulaError(f"{piece!r}: a grouping factor cannot be combined with itself ({factor!r})")
    combined = ":".join(operands)                                   # the derived factor as the term model spells it
    factors = [operands[0], combined] if sep == "/" else [combined]  # (1 | g/h) is (1 | g) + (1 | g:h)
    tokens = [t.strip() for t in _split_top_level(inner)]
    if any(t == "" for t in tokens):
        raise FormulaError(f"{piece!r}: empty term inside the group term")
    slopes = [t for t in tokens if t not in ("0", "1")]
    for s in slopes:                                            # each slope token is a column name
        if _OFFSET_RE.match(s):
            raise FormulaError(
                f"{piece!r}: offset() cannot be placed inside a group term; write it as a population-level term, "
                f"'{response} ~ ... + {s} + (1 {bars} {factor})'"
            )
        if _MO_RE.match(s):
            raise FormulaError(
                f"{piece!r}: a monotonic effect cannot vary by {factor!r} in this slice: brms's '(1 + {s} {bars} {factor})' gives "
                "every level its own coefficient of the monotonic term, which is not verified here (docs/monotonic.md, "
                f"divergence D41); keep {s} as a population-level term, '{response} ~ ... + {s} + (1 | {factor})'"
            )
        if "(" in s or ")" in s:
            raise FormulaError(
                f"{piece!r}: function calls and transformations inside a group term are not supported; "
                "transform the column in the data frame instead."
            )
        if ":" in s or "*" in s:
            raise FormulaError(
                f"{piece!r}: an interaction cannot be a varying slope in this subset; only a numeric "
                f"population-level term of the formula can vary by {factor!r}"
            )
        if not re.fullmatch(_IDENT, s):
            raise FormulaError(f"{piece!r}: {s!r} is not a column name")
        if s in operands:
            raise FormulaError(f"{piece!r}: the grouping factor {s!r} cannot also be a varying slope")
        if s == response:
            raise FormulaError(f"{piece!r}: the response {response!r} cannot be a varying slope")
        if trials is not None and s == trials:
            raise FormulaError(f"{piece!r}: the trials column {trials!r} cannot be a varying slope")
    duplicated = [s for s in dict.fromkeys(slopes) if slopes.count(s) > 1]
    if duplicated:
        raise FormulaError(f"{piece!r}: duplicated varying slope {duplicated[0]!r}")
    if "0" in tokens or "1" not in tokens:
        if not slopes:
            raise FormulaError(
                f"{piece!r}: '0' removes the varying intercept and leaves no group-level coefficient; a group term is "
                f"'(1 | {factor})', or '(1 + x || {factor})' for varying slopes"
            )
        raise FormulaError(                                     # the hint keeps the bars as written: '|' correlated (A02), '||' not
            f"{piece!r}: varying slopes without the varying intercept are not supported; write "
            f"'(1 + {' + '.join(slopes)} {bars} {factor})' so that the intercept varies by {factor!r} as well"
        )
    if tokens.count("1") > 1:
        raise FormulaError(f"{piece!r}: '1' appears more than once in the group term")
    if not slopes:
        return [GroupTerm(f, ("Intercept",), correlated=True) for f in factors]
    # '|' is brms's correlated form (A02): the intercept and slope(s) share a correlation matrix under an LKJ prior;
    # '||' the independent form (A01). '(1 + x | g/h)' gives '(1 + x | g) + (1 + x | g:h)', both correlated, as brms.
    return [GroupTerm(f, ("Intercept", *slopes), correlated=(bars == "|")) for f in factors]


def _normalise_categorical(categorical, terms: list[str], response: str, groups: list[str], trials: str | None,
                           formula: str) -> tuple[Categorical, ...]:
    if categorical is None:
        return ()
    if not isinstance(categorical, Mapping):
        raise FormulaError(
            "categorical must be a mapping {column: reference_level}, for example categorical={'z': 'a'}"
        )
    out: list[Categorical] = []
    for name, ref in categorical.items():
        if not isinstance(name, str):
            raise FormulaError(f"categorical keys must be column names, got {name!r}")
        if name == response:
            raise FormulaError(f"{name!r} is the response; only predictor terms can be declared categorical")
        if trials is not None and name == trials:
            raise FormulaError(f"{name!r} is the trials column; only predictor terms can be declared categorical")
        if name in groups:
            raise FormulaError(
                f"{name!r} is the grouping factor of '(1 | {name})', which is already a factor with a "
                "varying intercept; only fixed-effect predictor terms are declared categorical"
            )
        if name not in terms:
            raise FormulaError(
                f"{name!r} is declared categorical but is not a term of {formula!r}; add it to the formula"
            )
        if ref is None:
            raise FormulaError(
                f"categorical predictor {name!r} needs an explicit reference level: categorical={{{name!r}: <level>}}"
            )
        out.append(Categorical(name=name, reference=_scalar(f"categorical predictor {name!r}", ref, where="reference level")))
    return tuple(c for t in terms for c in out if c.name == t)          # formula order


_FLOAT_EXACT = 2 ** 53          # float64 represents every integer up to here exactly


def _label(v) -> str:
    """The string label of one categorical value (docs/m3a1-decision.md, repair record).

    Integers of any width and sign map to their exact decimal string; floats only when finite,
    integral and at most 2**53 in magnitude; strings are themselves. Booleans, complex numbers,
    bytes, missing values and every other type are refused. No float intermediate anywhere."""
    if isinstance(v, (bool, np.bool_)):
        raise FormulaError("boolean values are not accepted as categorical labels; recode them explicitly")
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            raise FormulaError(f"{v!r} is not a finite categorical value")
        if float(v) != math.floor(float(v)):
            raise FormulaError(f"categorical values must be integral or labels, got {v!r}")
        if abs(float(v)) > _FLOAT_EXACT:
            raise FormulaError(
                f"float categorical value {v!r} exceeds 2**53 in magnitude, beyond which float64 does not represent "
                "integers exactly; store large codes as integers (int64/uint64/object) or as strings"
            )
        return str(int(v))
    if isinstance(v, str):
        return str(v)
    if isinstance(v, (complex, np.complexfloating)):
        raise FormulaError(f"complex value {v!r} is not accepted as a categorical label")
    if v is None or v is pd.NA or v is pd.NaT:
        raise FormulaError("missing value is not accepted as a categorical label")
    raise FormulaError(f"unsupported categorical value type {type(v).__name__} ({v!r}); use integers or strings")


def _kind(v) -> str:
    return "text" if isinstance(v, str) else "number"


def categorical_labels(label: str, s: pd.Series) -> tuple[np.ndarray, tuple[str, ...]]:
    """String labels of a categorical predictor column, checked scalar by scalar before any
    coding (``_label``), and the column's declared-but-absent categories (pandas Categorical
    only). Containers are read as Python objects, which is exact for every integer width and
    for nullable types; a column mixing text labels with numeric codes is refused as ambiguous."""
    reject_complex(label, s)
    if pd.api.types.is_bool_dtype(s):
        raise FormulaError(
            f"{label} is boolean; a categorical predictor needs explicit labels or integer codes "
            "(recode it, for example data[col].map({False: 'no', True: 'yes'}))"
        )
    declared: tuple[str, ...] = ()
    if isinstance(s.dtype, pd.CategoricalDtype):
        declared = tuple(_scalar(label, c, where="declared category") for c in s.cat.categories)
        if len(set(declared)) != len(declared):
            raise FormulaError(f"{label} has declared categories that collide after normalisation: {list(declared)}")
        if len({_kind(c) for c in s.cat.categories}) > 1:
            raise FormulaError(f"{label} mixes text labels and numeric codes in its declared categories; recode it explicitly")
    values = s.astype(object)
    na = pd.isna(values.to_numpy())
    if na.any():
        raise FormulaError(f"{label} has missing values; drop or impute them explicitly before fitting.")
    raw = values.tolist()
    labels = np.array([_scalar(label, v) for v in raw], dtype=object)
    if len({_kind(v) for v in raw}) > 1:
        raise FormulaError(
            f"{label} mixes text labels and numeric codes (for example a string '1' next to an integer 1), which "
            "would silently merge under label normalisation; recode the column to one kind"
        )
    unused = tuple(c for c in declared if c not in set(labels.tolist()))
    return labels, unused


def _scalar(label: str, v, *, where: str = "value") -> str:
    try:
        return _label(v)
    except FormulaError as e:
        raise FormulaError(f"{label}: {where} {v!r} refused: {e}") from None


DISTRIBUTIONAL_FORM = ("a list of formulas, the response formula first and then one formula per distributional parameter, "
                       "['y ~ x + (1 | g)', 'sigma ~ z'] (brms's bf(y ~ x + (1 | g), sigma ~ z); docs/distributional.md)")


def _distributional_offer(fam) -> str:
    """Which parameters take a formula here, for the refusals below."""
    from autobayes.families import FAMILIES

    offering = [f"{name} ({spec.scale_parameter})" for name, spec in FAMILIES.items() if spec.distributional_link is not None]
    return (f"a distributional formula is offered for the auxiliary parameter of the {', '.join(offering)} families, with a "
            "log link (docs/distributional.md, divergence D33)")


def _parse_formulas(formulas, data: pd.DataFrame | None, family: str, categorical) -> ModelSpec:
    """A list of formulas (B01, docs/distributional.md): the response formula and the formulas of distributional
    parameters, brms's ``bf(y ~ x + (1 | g), sigma ~ z)``. Each distributional formula is ``<parameter> ~ <terms>`` with
    the parameter's brms name on the left, and is parsed as a formula of its own (population-level terms of every kind the
    response formula takes: numeric, declared categorical, interactions, transforms; its own intercept and centring),
    held in ``ModelSpec.dpars``. ``categorical`` is shared: a declared column is declared in every formula that names it
    as a main effect. Refused, naming the construct: a parameter the family has not, or one whose formula is not offered
    here (D33); group-level terms and ``offset()`` in a distributional formula (D32); a distributional formula that reads
    the response, the trials column or a grouping factor of the response formula (D35)."""
    from autobayes.families import get_family

    fam = get_family(family)
    if not isinstance(formulas, (list, tuple)) or not formulas or not all(isinstance(f, str) for f in formulas):
        raise FormulaError(f"a formula is a string, or {DISTRIBUTIONAL_FORM}; got {formulas!r}")
    main, *rest = formulas
    if not rest:
        return parse_formula(main, data, family, categorical)
    head = _ADDITIONS_FORMULA_RE.match(main) or _FORMULA_RE.match(main)
    response = head.group(1) if head else None
    parsed: list[tuple[str, str, str]] = []                   # (parameter, right-hand side, formula as written)
    for text in rest:
        m = _FORMULA_RE.match(text)
        if not m or "|" in text.split("~", 1)[0]:
            raise FormulaError(
                f"{text!r}: a distributional formula is '<parameter> ~ <terms>' with the parameter's name on the left, as "
                "brms writes 'sigma ~ z'; addition terms belong to the response formula, which comes first in the list"
            )
        name, rhs = m.group(1), m.group(2)
        if name == response or name in ("mu", fam.parent):
            raise FormulaError(
                f"{text!r}: {name!r} is the response formula's; the first formula of the list is the response's ({main!r}) "
                f"and the others are distributional parameters' ({DISTRIBUTIONAL_FORM})"
            )
        if name in (p for p, _, _ in parsed):
            raise FormulaError(f"{text!r}: {name!r} has two formulas; give each distributional parameter one")
        if name != fam.scale_parameter or fam.distributional_link is None:
            if name in (fam.scale_parameter, fam.zero_parameter, *fam.fixed_parameters, *fam.learnable_parameters):
                why = f"brms's {name} ~ ... is not offered here"
            else:
                own = [p for p in (fam.scale_parameter, fam.zero_parameter, *fam.fixed_parameters) if p is not None]
                why = (f"the {fam.name} family has no parameter {name!r}"
                       + (f" (its parameters besides the mean are {own})" if own else " (it has no parameter besides the mean)"))
            raise FormulaError(f"{text!r}: {why}; {_distributional_offer(fam)}")
        parsed.append((name, rhs, text.strip()))

    # Route the declared categorical columns: each formula receives the declarations of its own main effects (read
    # without the data); a declaration that names no formula's main effect stays with the response formula, which
    # refuses it as it always has.
    main0 = parse_formula(main, None, family, None)
    dpar0 = [parse_formula(f"{name} ~ {rhs}", None, family, None) for name, rhs, _ in parsed]
    mains = [{t.name for t in s.pop_terms if t.kind == "numeric"} for s in (main0, *dpar0)]
    if isinstance(categorical, Mapping):                    # a declared column is categorical in every formula
        for t in (t for s in (main0, *dpar0) for t in s.transform_terms):
            if t.transform.column in categorical:
                raise FormulaError(f"{t.name!r}: {t.transform.column!r} is declared categorical; an in-formula transform applies to a numeric column")
        reads = (([(main0.offset.term, main0.offset.column)] if main0.offset is not None else [])
                 + [(a.term, c) for a in main0.additions for c in a.columns])        # columns the response formula reads as numbers
        for term, col in reads:
            if col in categorical:
                raise FormulaError(f"{term!r}: {col!r} is declared categorical; the response formula reads it as a number")
        main_cat = {k: v for k, v in categorical.items() if k in mains[0] or not any(k in m for m in mains[1:])}
        dpar_cat = [{k: v for k, v in categorical.items() if k in m} for m in mains[1:]]
    else:
        main_cat, dpar_cat = categorical, [None] * len(parsed)
    spec = parse_formula(main, data, family, main_cat)
    operands = {c for t in spec.group_terms for c in t.operands}
    dpars = []
    for (name, rhs, text), cat in zip(parsed, dpar_cat):
        d = parse_formula(f"{name} ~ {rhs}", None, family, cat or None)
        if d.group_terms:
            raise FormulaError(
                f"{text!r}: group-level terms in a distributional formula ({', '.join(t.canonical for t in d.group_terms)}) are "
                f"not supported in this slice: brms would give {name} its own varying effects; a distributional formula "
                "takes population-level terms only here (docs/distributional.md, divergence D32)"
            )
        if d.offset is not None:
            raise FormulaError(
                f"{text!r}: {d.offset.term!r} in a distributional formula is not supported in this slice; an offset belongs "
                "to the response formula here (docs/distributional.md, divergence D32)"
            )
        if d.mo_terms:
            raise FormulaError(
                f"{text!r}: a monotonic term in a distributional formula ({', '.join(t.name for t in d.mo_terms)}) is not supported "
                f"in this slice: brms would give {name} a monotonic coefficient and simplex of its own; mo() belongs to the response "
                "formula here (docs/monotonic.md, divergence D42)"
            )
        for col in d.pop_columns:
            role = ("the response" if col == spec.response else "the trials column" if col == spec.trials
                    else "a grouping factor of the response formula" if col in operands else None)
            if role is not None:
                raise FormulaError(
                    f"{text!r}: {col!r} is {role}; it cannot also be a predictor of {name} in this slice "
                    "(docs/distributional.md, divergence D35)"
                )
        if data is not None:
            d = validate_data(d, data, family, dpar=True)
        dpars.append(d)
    return replace(spec, dpars=tuple(dpars))


def parse_formula(formula: str | list[str], data: pd.DataFrame | None = None, family: str = "gaussian",
                  categorical: Mapping[str, object] | None = None) -> ModelSpec:
    """Parse and validate a formula in the supported subset, optionally against a data frame.

    ``categorical`` maps a predictor column to its reference level (treatment coding); every
    other predictor term must be numeric. ``formula`` may also be a list (B01): the response formula, then one formula
    per distributional parameter, ``["y ~ x + (1 | g)", "sigma ~ z"]`` (``_parse_formulas``, docs/distributional.md).
    """
    from autobayes.families import get_family

    if not isinstance(formula, str):
        return _parse_formulas(formula, data, family, categorical)
    fam = get_family(family)
    trials, additions = None, ()
    if _BAMBI_PROPORTION_RE.match(formula):
        raise FormulaError(
            f"{formula!r}: Bambi's 'p(successes, trials)' left-hand side is not accepted; write the brms "
            "form 'successes | trials(n) ~ ...' (ab.binomial)."
        )
    ma = _ADDITIONS_FORMULA_RE.match(formula)              # brms's 'response | term + term ~ ...' (trials(), B02's terms)
    if ma:
        response, rhs = ma.group(1), ma.group(3)
        trials, additions = _parse_additions(ma.group(2), formula)
        if trials is not None and not fam.has_trials:
            raise FormulaError(
                f"{formula!r}: 'successes | trials(n)' is the binomial family's left-hand side; use "
                "ab.binomial, or drop '| trials(...)' for a Gaussian or Bernoulli response."
            )
        if trials is None and fam.has_trials:
            raise FormulaError(
                f"{formula!r}: the {fam.name} family needs exactly 'successes | trials(n)' on the left-hand side (brms "
                "syntax), with any other addition terms after it: 'k | trials(n) + weights(w) ~ ...'."
            )
        if trials == response:
            raise FormulaError(f"{trials!r}: the trials column cannot also be the response")
        _check_additions(additions, fam, formula)
    else:
        if "~" in formula and "|" in formula.split("~", 1)[0]:
            raise FormulaError(
                f"{formula!r}: the left-hand side must be 'response | term + term' with a column name as the response "
                "and brms's addition terms after the '|' (weights(col), cens(col), trunc(lb = a, ub = b); exactly "
                "'successes | trials(n)' for a binomial response; docs/addition-terms.md)."
            )
        m = _FORMULA_RE.match(formula)
        if not m:
            raise FormulaError(
                f"{formula!r}: expected 'response ~ terms'. Supported: "
                "'y ~ [1 +] x1 [+ x2 ...] [+ (1 | factor) ...]' and, for ab.binomial, "
                "'successes | trials(n) ~ [1 +] x1 [+ x2 ...] [+ (1 | factor) ...]'."
            )
        if fam.has_trials:
            raise FormulaError(
                f"{formula!r}: the {fam.name} family needs 'successes | trials(n) ~ ...' (brms syntax); "
                f"counts without a trials column are not a {fam.name} model."
            )
        response, rhs = m.group(1), m.group(2)
    # Only top-level subtraction of the literal 1 is supported. Do not rewrite
    # group terms, transformations, negative numbers, or predictor subtraction.
    minus_parts = _split_top_level(rhs, "-")
    if len(minus_parts) > 1:
        pieces = [minus_parts[0]] if minus_parts[0] else []
        for part in minus_parts[1:]:
            if not re.match(r"^1(?=\s*(?:\+|$))", part):
                raise FormulaError(f"{formula!r}: '-' supports only subtracting the population intercept '- 1'")
            pieces.append(re.sub(r"^1", "0", part, count=1))
        rhs = " + ".join(pieces)

    predictors: list[str] = []                             # main effects as written: numeric, categorical, transform and monotonic terms
    transforms: dict[str, Transform] = {}                  # the transform terms among them, by canonical name
    monotonics: dict[str, str] = {}                        # the monotonic terms among them (B03): canonical name -> column
    group_terms: list[GroupTerm] = []
    group_pieces: list[str] = []                           # each group term as written, for the messages below
    groups: list[str] = []                                 # the factors of group_terms, for the checks below
    raw_interactions: list[tuple[str, str, str]] = []      # (left, right, piece as written)
    offsets: list[tuple[Offset, str]] = []                 # (offset term, piece as written); at most one
    saw_intercept_token = False
    remove_intercept = False

    def _main(name: str, piece: str, *, transform: Transform | None = None, call: bool = False) -> None:
        if transform is None and not call and not re.fullmatch(_IDENT, name):
            raise FormulaError(f"{piece!r}: {name!r} is not a column name")
        if name == response:
            raise FormulaError(f"{name!r}: the response cannot also be a predictor")
        if name in predictors:
            if transform is not None or call:
                hint = f" ({piece!r} spells {name!r} again)" if piece != name else ""
            else:
                hint = f" after expanding {piece!r} (a*b means a + b + a:b; write 'a + b + a:b' or 'a*b', not both)" if piece != name else ""
            raise FormulaError(f"{name!r}: duplicated predictor{hint}")
        predictors.append(name)

    for piece in _split_top_level(rhs):
        if piece == "":
            raise FormulaError(f"{formula!r}: empty term")
        if piece == "1":
            saw_intercept_token = True
            continue
        if piece == "0":
            remove_intercept = True
            continue
        gm = _GROUP_RE.match(piece)
        if gm:
            inner, bars, factor = gm.group(1).strip(), gm.group(2), gm.group(3).strip()
            for term in _parse_group_term(inner, bars, factor, piece, response, trials):   # '(1 | g)', '(1 + x || g)', '(1 | g:h)'; '(1 | g/h)' gives two
                group_terms.append(term)
                group_pieces.append(piece)
                groups.append(term.factor)
            continue
        if piece.startswith("("):
            raise FormulaError(f"{piece!r}: parenthesised term is not a '(1 | group)' or '(1 + x || group)' term")
        if "|" in piece:
            raise FormulaError(f"{piece!r}: a '|' term must be written '(1 | group)' or '(1 + x || group)'")
        om = _OFFSET_RE.match(piece)
        if om:                                                  # offset(col) or offset(log(col)), brms's offset()
            inner = om.group(1).strip()
            lm = _OFFSET_LOG_RE.match(inner)
            if lm:
                offset = Offset(lm.group(1), log=True)
            elif re.fullmatch(_IDENT, inner):
                offset = Offset(inner)
            else:
                raise FormulaError(
                    f"{piece!r}: only 'offset(col)' and 'offset(log(col))' are supported inside offset(); other "
                    "expressions (general in-formula transforms) are not part of this slice. Compute the column in "
                    "the data frame and write 'offset(<column>)'."
                )
            if offsets:
                raise FormulaError(
                    f"{piece!r}: more than one offset term ({offsets[0][1]!r} and {piece!r}); add the offsets into one "
                    "column and write a single 'offset(<column>)'"
                )
            if offset.column == response:
                raise FormulaError(f"{piece!r}: the response {response!r} cannot be the offset")
            if trials is not None and offset.column == trials:
                raise FormulaError(f"{piece!r}: the trials column {trials!r} cannot be the offset")
            offsets.append((offset, piece))
            continue
        stars, colons = _split_top_level(piece, "*"), _split_top_level(piece, ":")
        if len(stars) > 1 or len(colons) > 1:                   # an interaction a:b, or the shorthand a*b
            if len(stars) > 1 and len(colons) > 1:
                raise FormulaError(f"{piece!r}: mixing '*' and ':' in one term is not supported; write 'x*z' or 'x + z + x:z'")
            factors = [f.strip() for f in (stars if len(stars) > 1 else colons)]
            if len(factors) != 2 or any(f == "" for f in factors):
                raise FormulaError(
                    f"{piece!r}: only a two-way interaction is supported in this slice ({piece!r} names {len(factors)} terms); "
                    "three-way and higher-order interactions are not."
                )
            left, right = factors
            for side in (left, right):
                if side == response or (trials is not None and side == trials):
                    raise FormulaError(f"{piece!r}: the response or trials column cannot enter an interaction")
                if _OFFSET_RE.match(side):
                    raise FormulaError(
                        f"{piece!r}: offset() cannot enter an interaction; an offset is a known column added to the linear "
                        "predictor with no coefficient"
                    )
                if _MO_RE.match(side):
                    raise FormulaError(
                        f"{piece!r}: an interaction with a monotonic term ({side!r}) is not supported in this slice: brms's "
                        "mo(x):z gives each interaction its own coefficient (and mo(x):mo(w) a simplex per factor), which is not "
                        "verified here (docs/monotonic.md, divergence D40)"
                    )
                if "(" in side or ")" in side:
                    if parse_transform(side) is None:
                        raise FormulaError(f"{side!r}: {SUPPORTED_TRANSFORMS}")
                    raise FormulaError(
                        f"{piece!r}: an interaction with a transformed term ({side!r}) is not supported in this slice; "
                        "compute the column in the data frame and interact the column instead."
                    )
            if left == right:
                raise FormulaError(f"{piece!r}: a term cannot interact with itself")
            if len(stars) > 1:                                      # a*b is shorthand for a + b + a:b
                _main(left, piece)
                _main(right, piece)
            raw_interactions.append((left, right, piece))
            continue
        mm = _MO_RE.match(piece)
        if mm:                                                  # brms's monotonic effect mo(col) (B03)
            inner = mm.group(1).strip()
            if len(_split_top_level(inner, ",")) > 1:
                raise FormulaError(
                    f"{piece!r}: mo() takes one column here; brms's id argument (mo(x, id = 'a'), one simplex shared by the "
                    "terms with the same id) is not supported in this slice (docs/monotonic.md, divergence D42)"
                )
            if not re.fullmatch(_IDENT, inner):
                raise FormulaError(
                    f"{piece!r}: mo() takes a column name here, not an expression ({inner!r}; brms evaluates one): compute the "
                    "column in the data frame and write mo(<column>) (docs/monotonic.md, divergence D42)"
                )
            if inner == response:
                raise FormulaError(f"{piece!r}: the response {response!r} cannot be a monotonic predictor")
            if trials is not None and inner == trials:
                raise FormulaError(f"{piece!r}: the trials column {trials!r} cannot be a monotonic predictor")
            _main(f"mo({inner})", piece, call=True)                # under its canonical spelling, so 'mo( x )' duplicates 'mo(x)'
            monotonics[f"mo({inner})"] = inner
            continue
        if "(" in piece or ")" in piece:                       # an in-formula transform, log(col), I(col^2), ... (A05)
            tr = parse_transform(piece)
            if tr is None:
                raise FormulaError(f"{piece!r}: {SUPPORTED_TRANSFORMS}")
            if tr.column == response:
                raise FormulaError(f"{piece!r}: the response {response!r} cannot be transformed into a predictor")
            if trials is not None and tr.column == trials:
                raise FormulaError(f"{piece!r}: the trials column {trials!r} cannot be transformed into a predictor")
            _main(tr.term, piece, transform=tr)                   # under its canonical spelling, so 'log( x )' duplicates 'log(x)'
            transforms[tr.term] = tr
            continue
        _main(piece, piece)

    if saw_intercept_token and remove_intercept:
        raise FormulaError(f"{formula!r}: conflicting intercept inclusion/removal; use '0 + terms' or 'terms - 1'")

    operands_all = list(dict.fromkeys(c for t in group_terms for c in t.operands))   # every column a grouping factor reads
    for term, piece in zip(group_terms, group_pieces):       # a varying slope is a population-level term of the formula
        for s in term.slopes:
            if s not in predictors:
                raise FormulaError(
                    f"{piece!r}: the varying slope {s!r} must also be a population-level term of the formula, "
                    f"as in '{response} ~ {s} + (1 + {s} {'|' if term.correlated else '||'} {term.factor})' (brms would add the group-level slope "
                    "without the population-level one; this subset does not)"
                )

    for left, right, piece in raw_interactions:              # both main effects are required in this slice
        for side in (left, right):
            if side not in predictors and re.fullmatch(_IDENT, side) and side not in operands_all:
                raise FormulaError(
                    f"{piece!r}: the main effect {side!r} is missing. This slice requires both main effects of an "
                    f"interaction ('{left} + {right} + {left}:{right}' or '{left}*{right}'): omitting one is a different "
                    "model (the level intercepts or the reference slope are forced to zero and formulae changes the "
                    "interaction's coding) and is not part of the verified subset (docs/m3a2-decision.md)."
                )
    if remove_intercept and not predictors:
        raise FormulaError(f"{formula!r}: no intercept requires at least one population-level predictor")
    seen_partition: dict[frozenset, int] = {}                # a factor, or the same pair of columns in either order, once
    for i, (term, piece) in enumerate(zip(group_terms, group_pieces)):
        key = frozenset(term.operands)
        if key not in seen_partition:
            seen_partition[key] = i
            continue
        other, other_piece = group_terms[seen_partition[key]], group_pieces[seen_partition[key]]
        if other.factor == term.factor:
            hint = ""
            slash = next((p for p in (piece, other_piece) if "/" in _GROUP_RE.match(p).group(3)), None)
            if slash is not None:
                a, b = [p.strip() for p in _GROUP_RE.match(slash).group(3).split("/")]
                hint = f" ('{slash}' expands to '(1 | {a}) + (1 | {a}:{b})')"
            raise FormulaError(
                f"{formula!r}: duplicate grouping factor {term.factor!r}; a factor appears in one group term, "
                f"'(1 | {term.factor})' or '(1 + x || {term.factor})'{hint}"
            )
        raise FormulaError(
            f"{formula!r}: '{piece}' groups by the same columns as '{other_piece}' ({term.factor!r} and {other.factor!r} "
            "induce the same partition of the rows); keep one of them"
        )
    if len(predictors) == 0 and not saw_intercept_token:
        raise FormulaError(
            f"{formula!r}: no population-level predictor; write 'y ~ 1 + (1 | group)' "
            "for an intercept-only model."
        )
    transform_columns = [tr.column for tr in transforms.values()]   # the raw columns the transform terms read
    mo_columns = list(monotonics.values())                           # and the monotonic terms (B03)
    for col in operands_all:
        if col in predictors or col in transform_columns or col in mo_columns or col == response:
            raise FormulaError(f"{col!r}: the grouping factor cannot also be a predictor or the response")
    if trials is not None and (trials in predictors or trials in operands_all):
        raise FormulaError(f"{trials!r}: the trials column cannot also be a predictor or the grouping factor")
    offset = offsets[0][0] if offsets else None
    # A derived factor's column (g_h) is added to the fitted frame: it must not shadow a column the formula uses.
    in_use = {response, *predictors, *transform_columns, *mo_columns, *operands_all, *([trials] if trials is not None else []),
              *([offset.column] if offset is not None else []), *(c for a in additions for c in a.columns)}
    derived_columns: dict[str, GroupTerm] = {}
    for term in group_terms:
        if not term.derived:
            continue
        if term.column in derived_columns:
            raise FormulaError(
                f"{formula!r}: the derived grouping factors {derived_columns[term.column].factor!r} and {term.factor!r} would "
                f"both be handed to Bambi as the column {term.column!r}; rename one of the columns"
            )
        if term.column in in_use:
            raise FormulaError(
                f"{formula!r}: the derived grouping factor {term.factor!r} is handed to Bambi as the column {term.column!r}, "
                "which is also a column of the formula; rename that column"
            )
        derived_columns[term.column] = term
    if offset is not None and offset.column in operands_all:
        raise FormulaError(
            f"{offsets[0][1]!r}: the grouping factor {offset.column!r} cannot be the offset; an offset is a known "
            "numeric column added to the linear predictor"
        )

    if isinstance(categorical, Mapping):                     # a transform applies to a numeric column, never to a declared categorical one
        for name, tr in transforms.items():
            if tr.column in categorical:
                raise FormulaError(f"{name!r}: {tr.column!r} is declared categorical; an in-formula transform applies to a numeric column")
        for name, col in monotonics.items():                 # nor does mo(), which reads the column's own values (B03)
            if col in categorical:
                raise FormulaError(f"{name!r}: {col!r} is declared categorical; mo() reads it as an ordinal predictor (whole numbers "
                                   "or an ordered pandas Categorical), and one column cannot be both")
    categoricals = _normalise_categorical(categorical, predictors, response, operands_all, trials, formula)
    if offset is not None and any(c.name == offset.column for c in categoricals):
        raise FormulaError(
            f"{offsets[0][1]!r}: {offset.column!r} is declared categorical; an offset is a known numeric column added "
            "to the linear predictor"
        )
    if remove_intercept and categoricals:
        categoricals = (replace(categoricals[0], full_rank=True), *categoricals[1:])
    cat_names = {c.name for c in categoricals}
    for term, piece in zip(group_terms, group_pieces):
        for s in term.slopes:
            if s in cat_names:
                raise FormulaError(
                    f"{piece!r}: {s!r} is declared categorical; a categorical varying slope (one varying coefficient "
                    f"per level of {s!r}) is not supported in this subset; only a numeric population-level term can "
                    f"vary by {term.factor!r}"
                )
    interactions: list[str] = []                             # canonical interaction names, in formula order
    for left, right, piece in raw_interactions:
        for side in (left, right):
            if not re.fullmatch(_IDENT, side):
                raise FormulaError(f"{piece!r}: {side!r} is not a column name")
            if side in operands_all:
                raise FormulaError(f"{piece!r}: the grouping factor {side!r} cannot enter a fixed interaction")
            if side == response or (trials is not None and side == trials):
                raise FormulaError(f"{piece!r}: the response or trials column cannot enter an interaction")
        cats = [s for s in (left, right) if s in cat_names]
        if len(cats) == 1:                                       # x:z: numeric first, as the term model has always spelled it
            name = f"{[s for s in (left, right) if s not in cat_names][0]}:{cats[0]}"
        else:                                                    # x:w, z:h: main-effect order, as R orders a term's variables
            name = ":".join(sorted((left, right), key=predictors.index))
        if name in interactions:
            raise FormulaError(f"{piece!r}: the interaction {name!r} appears more than once after expansion")
        interactions.append(name)
    parts = [*predictors, *interactions] + ([offset.term] if offset is not None else []) + [t.canonical for t in group_terms]
    canonical_rhs = " + ".join(parts) if parts else "1"           # intercept-only population model: 'y ~ 1'
    if remove_intercept:
        canonical_rhs = "0 + " + canonical_rhs
    kinds = {**{p: "transform" for p in transforms}, **{p: "monotonic" for p in monotonics}, **{p: "categorical" for p in cat_names}}
    pop_terms = (*(pop_term(p, kinds.get(p, "numeric"), categoricals) for p in predictors),
                 *(pop_term(name, "interaction", categoricals) for name in interactions))
    spec = ModelSpec(
        formula=formula.strip(), response=response, intercept=not remove_intercept, trials=trials,
        rhs=canonical_rhs, categoricals=categoricals, pop_terms=pop_terms, group_terms=tuple(group_terms),
        offset=offset, additions=additions,
    )
    if spec.mo_terms and not fam.monotonic:                  # mo() where a frozen brms case verifies it (B03)
        from autobayes.families import FAMILIES

        offering = [name for name, other in FAMILIES.items() if other.monotonic]
        raise FormulaError(
            f"{formula!r}: the monotonic term {spec.mo_terms[0].name!r} is not available for the {fam.name} family; brms's mo() is "
            f"verified here for the {', '.join(offering) or 'no'} famil{'y' if len(offering) == 1 else 'ies'} "
            "(docs/monotonic.md, divergence D42)"
        )
    if fam.check_formula is not None:                        # a family that does not take every formula (B05)
        fam.check_formula(spec)
    if data is not None:
        spec = validate_data(spec, data, family)
    return spec


def reject_complex(label: str, s: pd.Series) -> None:
    """Refuse a complex-valued column before anything casts it (a cast would silently drop
    the imaginary part and pandas counts complex as numeric)."""
    if pd.api.types.is_complex_dtype(s):
        raise FormulaError(f"{label} is complex-valued ({s.dtype}); only real numeric columns are supported.")


def _is_numeric(s: pd.Series) -> bool:
    return (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            and not pd.api.types.is_complex_dtype(s))


def validate_numeric_response(name: str, s: pd.Series, family_label: str) -> None:
    """A continuous family's response: a real numeric column."""
    if not _is_numeric(s):
        raise FormulaError(
            f"column {name!r} has dtype {s.dtype}; the {family_label} family needs a numeric response."
        )


def validate_positive_response(name: str, s: pd.Series, family_label: str) -> None:
    """A positive continuous family's response (Gamma): a real numeric column of finite values strictly greater
    than zero. Missing values are left to the shared missing-data check that follows."""
    validate_numeric_response(name, s, family_label)
    v = s.to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    if np.any(np.isinf(v)):
        raise FormulaError(f"response {name!r} has a non-finite value; the {family_label} family needs finite positive responses.")
    if np.any(v <= 0):
        raise FormulaError(
            f"response {name!r} has a non-positive value ({v[v <= 0][0]!r}); the {family_label} family needs strictly "
            "positive responses (a zero is refused, not fitted)."
        )


def validate_proportion_response(name: str, s: pd.Series, family_label: str) -> None:
    """A proportion family's response (Beta): a real numeric column of values strictly inside (0, 1). Missing values
    are left to the shared missing-data check that follows."""
    validate_numeric_response(name, s, family_label)
    v = s.to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    if np.any(np.isinf(v)):
        raise FormulaError(f"response {name!r} has a non-finite value; the {family_label} family needs proportions strictly inside (0, 1).")
    bad = v[(v <= 0) | (v >= 1)]
    if bad.size:
        raise FormulaError(
            f"response {name!r} has a value outside the open interval (0, 1) ({bad[0]!r}); the {family_label} family needs "
            "proportions strictly between 0 and 1 (an exact 0 or 1 is refused, not fitted)."
        )


def validate_binary_response(name: str, s: pd.Series) -> None:
    reject_complex(f"response {name!r}", s)
    if pd.api.types.is_bool_dtype(s):
        raise FormulaError(
            f"response {name!r} is boolean; code it as integers 0/1 explicitly "
            f"(data[{name!r}].astype(int)) so that 1 is unambiguously the modelled outcome."
        )
    if not pd.api.types.is_numeric_dtype(s):
        raise FormulaError(
            f"response {name!r} has dtype {s.dtype}; the Bernoulli family needs a numeric 0/1 "
            "column with 1 the modelled outcome. Recode labels explicitly rather than relying on "
            "the sorted first level."
        )
    vals = set(np.unique(s.dropna().to_numpy(dtype=float)))
    if not vals <= {0.0, 1.0}:
        raise FormulaError(
            f"response {name!r} takes values outside {{0, 1}}: {sorted(vals - {0.0, 1.0})[:5]}. "
            "The Bernoulli family needs exactly 0/1 (binomial trials are not supported in M1)."
        )
    if vals != {0.0, 1.0}:
        raise FormulaError(
            f"response {name!r} is constant ({sorted(vals)}); a Bernoulli model needs both outcomes."
        )


def ordinal_categories(name: str, s: pd.Series) -> tuple[tuple[str, ...], np.ndarray, list[str]]:
    """brms's ordinal response (the cumulative family, B05): an ordered pandas Categorical, whose categories in their
    declared order are the response's (every declared category, observed or not: brms's ordered factor), or positive
    whole numbers, whose categories are ``1..max(y)`` (brms's rule for a numeric response, so a value in between that no
    row takes is still a category); anything else is refused as brms refuses it. Returns the categories as labels, each
    row's 1-based category (brms's ``Y``) and a warning naming the categories no fitted row takes."""
    label = f"response {name!r}"
    reject_complex(label, s)
    need = ("the cumulative family needs an ordinal response: positive integers 1..K or an ordered pandas Categorical "
            "(pd.Categorical(values, categories=[lowest, ..., highest], ordered=True), brms's ordered factor)")
    if isinstance(s.dtype, pd.CategoricalDtype):
        if not s.cat.ordered:
            raise FormulaError(f"{label} is an unordered pandas Categorical, whose category order means nothing; {need}")
        categories = tuple(_scalar(label, c, where="declared category") for c in s.cat.categories)
        if len(set(categories)) != len(categories):
            raise FormulaError(f"{label} has declared categories that collide after normalisation: {list(categories)}")
        if len({_kind(c) for c in s.cat.categories}) > 1:
            raise FormulaError(f"{label} mixes text labels and numeric codes in its declared categories; recode it explicitly")
        if s.isna().any():
            raise FormulaError(f"{label} has missing values; drop or impute them explicitly before fitting.")
        codes = s.cat.codes.to_numpy().astype(np.int64) + 1
    else:
        if pd.api.types.is_bool_dtype(s):
            raise FormulaError(f"{label} is boolean; {need}")
        if not pd.api.types.is_numeric_dtype(s):
            raise FormulaError(f"{label} has dtype {s.dtype}; {need}. Text labels have no order: declare it with an ordered "
                               "Categorical")
        v = s.to_numpy(dtype=float)
        if np.isnan(v).any():
            raise FormulaError(f"{label} has missing values; drop or impute them explicitly before fitting.")
        bad = v[~np.isfinite(v) | (v != np.floor(v)) | (v < 1)]
        if bad.size:
            raise FormulaError(f"{label} has the value {bad[0]:g}; {need} (brms: 'requires either positive integers or "
                               "ordered factors as responses')")
        if v.max() > COUNT_CAP:
            raise FormulaError(f"{label} has the value {v.max():.0f}, which would make {v.max():.0f} categories; an ordinal "
                               f"response above {COUNT_CAP} is refused")
        categories = tuple(str(k) for k in range(1, int(v.max()) + 1))
        codes = v.astype(np.int64)
    if len(categories) < 2:
        raise FormulaError(f"{label} has one category ({categories[0]!r}); at least two response categories are required")
    unobserved = [c for k, c in enumerate(categories, start=1) if not (codes == k).any()]
    notes = [f"categories {unobserved} of the ordinal response {name!r} have no fitted rows: the thresholds next to them "
             "are informed by the ordering and the prior alone (brms keeps them too)"] if unobserved else []
    return categories, codes, notes


def categorical_categories(name: str, s: pd.Series) -> tuple[tuple[str, ...], np.ndarray, list[str]]:
    """brms's categorical response (the categorical family, B05): labels (text, or integer codes) checked as the library
    checks every categorical value (``categorical_labels``). The categories are the distinct labels in the order brms's
    ``as.factor`` gives them (text in C-locale order, which is Python's ``sorted``; integer codes numerically), the first
    the reference; a pandas Categorical's declared order is not used and its unused categories are not categories
    (brms drops them too). Returns the categories, each row's 1-based category (brms's ``Y``) and a warning naming the
    declared categories no fitted row takes."""
    label = f"response {name!r}"
    labels, unused = categorical_labels(label, s)
    numeric = all(_kind(v) == "number" for v in s.astype(object).tolist())
    categories = tuple(sorted(set(labels.tolist()), key=(lambda l: int(l)) if numeric else None))
    if len(categories) < 2:
        raise FormulaError(f"{label} has one category ({categories[0]!r}); at least two response categories are required")
    position = {c: k for k, c in enumerate(categories, start=1)}
    codes = np.array([position[l] for l in labels], dtype=np.int64)
    notes = [f"declared categories {list(unused)} of the categorical response {name!r} do not occur in the fitted rows: "
             "they are not categories of the model (brms drops them too)"] if unused else []
    return categories, codes, notes


def check_counts(label: str, s: pd.Series, *, cap: int = COUNT_CAP, domain: str = "binomial") -> np.ndarray:
    """A count column: real numeric, finite, integral, in ``[0, cap]``. Checked before any cast;
    returns the values as ``int64``."""
    reject_complex(label, s)
    if pd.api.types.is_bool_dtype(s):
        raise FormulaError(f"{label} is boolean; counts must be integers")
    if not pd.api.types.is_numeric_dtype(s):
        raise FormulaError(f"{label} has dtype {s.dtype}; counts must be numeric integers")
    v = s.to_numpy()
    if not np.all(np.isfinite(v.astype(float))):
        bad = v[~np.isfinite(v.astype(float))][0]
        raise FormulaError(f"{label} has a non-finite value ({bad!r}); counts must be finite integers")
    vf = v.astype(float)
    if not np.all(np.mod(vf, 1.0) == 0.0):
        bad = vf[np.mod(vf, 1.0) != 0.0][0]
        raise FormulaError(f"{label} has a non-integral value ({bad!r}); counts must be whole numbers")
    if np.any(vf < 0):
        raise FormulaError(f"{label} has a negative value ({vf[vf < 0][0]!r}); counts must be non-negative")
    if np.any(vf > cap):
        if domain == "poisson":
            raise FormulaError(f"{label} exceeds the Poisson input-count limit {cap}; see docs/poisson-decision.md")
        raise FormulaError(
            f"{label} has a value above the tested limit {cap} ({vf[vf > cap][0]:.0f}); larger counts per row "
            "are outside the validated domain (the float64 density and gradient checks in docs/m2-decision.md) "
            "and are refused rather than assumed. Note that splitting a row into several rows changes the "
            "unweighted centring and therefore what the centred-intercept prior refers to."
        )
    return vf.astype(np.int64)


def binomial_counts(spec: ModelSpec, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Validated ``(successes, trials)`` for a binomial formula, without dropping any row.

    Refuses ``successes > trials``, a zero-trial row with successes, and a data set with no
    trials at all. Rows with ``trials = 0`` and ``successes = 0`` pass; the model drops and
    reports them (they carry no likelihood and would otherwise enter the centring means).
    """
    na = data[[spec.response, spec.trials]].isna().any()
    if na.any():
        raise FormulaError(
            f"missing values in columns {list(na[na].index)}; drop or impute them explicitly before fitting."
        )
    k = check_counts(f"response {spec.response!r}", data[spec.response])
    n = check_counts(f"trials {spec.trials!r}", data[spec.trials])
    impossible = np.flatnonzero((n == 0) & (k > 0))
    if impossible.size:
        raise FormulaError(f"row {int(impossible[0])}: successes with zero trials is impossible data")
    over = np.flatnonzero(k > n)
    if over.size:
        i = int(over[0])
        raise FormulaError(
            f"row {i}: successes {spec.response!r} = {k[i]} exceed trials {spec.trials!r} = {n[i]}"
        )
    if not np.any(n > 0):
        raise FormulaError(
            f"no trials in the data ({spec.trials!r} is zero everywhere); a prior-only fit must be requested "
            "explicitly, it is not a binomial fit"
        )
    return k, n


def validate_data(spec: ModelSpec, data: pd.DataFrame, family: str = "gaussian", *, dpar: bool = False) -> ModelSpec:
    """Check columns exist, are numeric where required, and have no missing values. ``dpar=True`` validates a
    distributional parameter's formula (B01), whose left-hand side names a parameter rather than a column: its terms
    only."""
    from autobayes.families import get_family

    fam = get_family(family)
    if not dpar and (spec.trials is not None) != fam.has_trials:
        raise FormulaError("a trials column is present exactly for the binomial family")
    operands = [c for t in spec.group_terms for c in t.operands]   # the columns the grouping factors are read from
    needed = (([] if dpar else [spec.response]) + [*spec.pop_columns] + operands
              + ([spec.trials] if spec.trials else []) + ([spec.offset.column] if spec.offset is not None else [])
              + [c for a in spec.additions for c in a.columns])
    needed = list(dict.fromkeys(needed))                         # the offset column may also be a predictor
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise FormulaError(f"columns not in data: {missing}")
    for c in spec.predictors:
        reject_complex(f"column {c!r}", data[c])
        if not _is_numeric(data[c]):
            raise FormulaError(
                f"column {c!r} has dtype {data[c].dtype}; a predictor is numeric unless it is declared "
                f"categorical with its reference level, for example categorical={{{c!r}: <level>}} "
                "(treatment coding; docs/m3a1-decision.md)."
            )
    if spec.offset is not None:
        spec.offset.values(data)                                 # numeric, finite and (for log) positive, or refused
    for t in spec.transform_terms:
        t.transform.values(data)                                 # numeric, finite, in the function's domain and finite after it, or refused
    categoricals = []
    for cat in spec.categoricals:
        labels, unused = categorical_labels(f"categorical predictor {cat.name!r}", data[cat.name])
        levels = tuple(sorted(set(labels.tolist())))
        if len(levels) < 2:
            raise FormulaError(f"predictor {cat.name!r} is constant (single level {levels[0]!r} in the fitted rows)")
        if cat.reference not in levels:
            raise FormulaError(
                f"reference level {cat.reference!r} of {cat.name!r} does not occur in the fitted rows; "
                f"observed levels: {list(levels)}"
            )
        categoricals.append(Categorical(name=cat.name, reference=cat.reference, levels=levels, unused=unused, full_rank=cat.full_rank))
    if not dpar:
        reject_complex(f"response {spec.response!r}", data[spec.response])
        fam.validate_response(spec, data)
    na = data[needed].isna().any()
    if na.any():
        raise FormulaError(
            f"missing values in columns {list(na[na].index)}; M0 does not handle missing "
            "data. Drop or impute them explicitly before fitting."
        )
    if len(data) < 2:
        raise FormulaError("need at least two observations")
    addition_data(spec, data)                                     # positive weights, censoring codes, responses inside the bounds (B02)
    levels_by_group = {}
    for t in spec.group_terms:                                    # a derived factor's levels: the observed combinations
        levels = tuple(sorted(group_labels(t.factor, data).unique()))
        if len(levels) < 2:
            raise FormulaError(f"grouping factor {t.factor!r} has a single level; a varying intercept needs at least two.")
        levels_by_group[t.factor] = levels
    for c in spec.predictors:
        if np.isclose(np.std(data[c].to_numpy(dtype=float)), 0.0):
            raise FormulaError(f"predictor {c!r} is constant")
    for t in spec.pop_terms:                                      # a numeric-by-numeric product column is a predictor too
        if t.kind == "interaction" and not t.categorical and np.isclose(np.std(pop_term_design(t, data)), 0.0):
            raise FormulaError(f"interaction {t.name!r} is constant ({' * '.join(t.operands)} takes a single value in the fitted rows)")
    for t in spec.transform_terms:                                # and so is a transformed column
        if np.isclose(np.std(pop_term_design(t, data)), 0.0):
            raise FormulaError(f"term {t.name!r} is constant in the fitted rows")
    monotonics = tuple(monotonic_coding(m.column, data[m.column]) for m in spec.monotonics)   # brms's Xmo coding (B03)
    return ModelSpec(
        formula=spec.formula, response=spec.response, intercept=spec.intercept, levels_by_group=levels_by_group,
        trials=spec.trials, rhs=spec.rhs, categoricals=tuple(categoricals),
        pop_terms=tuple(pop_term(t.name, t.kind, categoricals) for t in spec.pop_terms),   # levels now known
        group_terms=spec.group_terms, offset=spec.offset, additions=spec.additions, dpars=spec.dpars, monotonics=monotonics,
    )
