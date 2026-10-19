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
log likelihood by its weight. ``cens(col)`` gives each row a censoring code and ``trunc(lb = a, ub = b)`` truncates the
response (Gaussian); both were withheld (``PARKED_ADDITIONS``) until P6 accepted them against two references, brms where
brms's own numbers are exact and a committed 50-digit reference where they are not (divergences D24, D25).

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

``gp(col)`` and ``gp(col, k = K, c = C)`` are brms's Gaussian process of one numeric column, exact or Hilbert-space
approximate (C03, docs/gaussian-processes.md), with the squared-exponential kernel and brms's scaling of the input: terms of
the linear predictor without a coefficient of class ``b``, held in ``ModelSpec.gp_terms`` (``GaussianProcess``, whose
inputs and scaling are known once the data are seen). Several covariates in one ``gp()``, ``by``, other kernels, ``iso``,
``gr = FALSE``, ``cmc``, ``scale = FALSE``, interactions with ``gp()``, and ``gp()`` in a group-level or distributional
formula are refused.

brms's autocorrelation terms ``ar(time, gr, p)``, ``ma(time, gr, q)`` and ``arma(time, gr, p, q)`` in its regression
formulation (C04, docs/autocorrelation.md) are held in ``ModelSpec.autocor`` (``autocor.Autocor``), one per formula: not
a population-level term (no design column, no ``b`` coefficient), the coefficients ``ar`` and ``ma`` are its own.

``me(x, sdx)`` and ``mi(col)`` are brms's measurement-error and missing-value terms (C06,
docs/missing-and-measurement-error.md), special terms like ``mo()``: kinds ``"measurement_error"`` and ``"missing"``, a
coefficient (brms's ``bsp``) and no design column. ``me(x, sdx)`` multiplies the latent values of a predictor ``x``
observed with the known standard deviations ``sdx`` (one such term, Gaussian family); ``mi(col)`` multiplies a column whose
missing values are parameters of the model, which another formula of a multivariate model, ``col | mi() ~ ...``, models
(the addition term ``mi()``, ``AdditionTerm("mi")``; ``ModelSpec.missing_columns``). Both are refused in interactions,
group-level and distributional formulas, with brms's ``gr`` and ``idx`` arguments, next to another kind of special term,
and for families that do not take them.

brms's multi-membership term ``(1 | mm(g, g2, weights = cbind(w1, w2)))``, ``(1 | gr(g, by = gby))`` (one SD per level of
``gby``) and a correlation ID ``(1 + x |p| g)`` shared with a distributional formula's ``sigma ~ 1 + (1 |p| g)`` (D08,
docs/multi-membership-and-by.md) are ``GroupTerm`` fields (``members``/``weights``, ``by``, ``id``); the first two are built by the
model itself (``GroupTerm.custom``), the third by Bambi and then correlated across the formulas (``ModelSpec.shared_ids``).
brms's category-specific effect ``cs(x)`` (D04, docs/ordinal-categorical.md) is a special term too, kind
``"category_specific"``: no design column and one coefficient per threshold (brms's ``bcs``, class ``b``), for the families that
take it (``FamilySpec.category_specific``: sratio, cratio, acat); elsewhere, with an expression inside, next to the column as a
population-level term and in interactions and group-level terms it is refused naming the construct (divergences D89, D107).

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
_CS_RE = re.compile(r"^cs\s*\((.*)\)$", re.S)                                                        # cs(<inside>), brms's category-specific effect (D04)
_CS_TERM_RE = re.compile(rf"^cs\(({_IDENT})\)$")                                                    # the canonical spelling cs(col)
_MO_RE = re.compile(r"^mo\s*\((.*)\)$", re.S)                                                        # mo(<inside>), brms's monotonic effect (B03)
_MO_TERM_RE = re.compile(rf"^mo\(({_IDENT})\)$")                                                    # the canonical spelling mo(col)
_ME_RE = re.compile(r"^me\s*\((.*)\)$", re.S)                                                        # me(<inside>), brms's measurement error (C06)
_ME_TERM_RE = re.compile(rf"^me\(({_IDENT}), ({_IDENT})\)$")                                         # the canonical spelling me(x, sdx)
_MI_RE = re.compile(r"^mi\s*\((.*)\)$", re.S)                                                        # mi(<inside>), brms's missing values (C06)
_MI_TERM_RE = re.compile(rf"^mi\(({_IDENT})\)$")                                                    # the canonical spelling mi(col)
MI_CALL_RE = re.compile(r"(?<![A-Za-z0-9_.])mi\s*\(")                                                 # an mi( call anywhere in a formula (C06)
MVBIND_RE = re.compile(r"^\s*mvbind\s*\(", re.S)                                                       # brms's multivariate mvbind(y, y2) ~ ... (C01)
_GP_RE = re.compile(r"^gp\s*\((.*)\)$", re.S)                                                        # gp(<inside>), brms's Gaussian process (C03)
_AUTOCOR_RE = re.compile(r"^(ar|ma|arma|cosy|car|sar|fcor|unstr)\s*\(", re.S)   # brms's autocorrelation terms (C04; autobayes.autocor)
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


POP_TERM_KINDS = ("numeric", "categorical", "interaction", "transform", "monotonic", "measurement_error", "missing",
                  "category_specific")
# brms's special terms (its bsp coefficients, and D04's category-specific bcs): a coefficient and no design column, built by
# this library, never by Bambi
SPECIAL_KINDS = ("monotonic", "measurement_error", "missing", "category_specific")


def monotonic_column(name: str) -> str | None:
    """The column of a monotonic term spelled ``mo(col)`` (B03), or None for any other term name."""
    m = _MO_TERM_RE.match(name)
    return m.group(1) if m else None


def measurement_error_columns(name: str) -> tuple[str, str] | None:
    """The noisy column and the noise column of a measurement-error term spelled ``me(x, sdx)`` (C06), or None."""
    m = _ME_TERM_RE.match(name)
    return (m.group(1), m.group(2)) if m else None


def missing_column(name: str) -> str | None:
    """The column of a missing-value term spelled ``mi(col)`` (C06), or None for any other term name."""
    m = _MI_TERM_RE.match(name)
    return m.group(1) if m else None


def category_specific_column(name: str) -> str | None:
    """The column of a category-specific term spelled ``cs(col)`` (D04), or None for any other term name."""
    m = _CS_TERM_RE.match(name)
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
class GaussianProcess:
    """brms's Gaussian process term of one numeric column (C03, docs/gaussian-processes.md): ``gp(x)``, the exact GP, or
    ``gp(x, k = K, c = C)``, its Hilbert-space approximation with ``K`` basis functions on ``[-L, L]``, ``L = C`` times the
    largest distance between scaled inputs, which is 1 (brms's ``c`` defaults to 5/4); the squared-exponential kernel and
    brms's ``scale = TRUE``. It is a term of the linear predictor without a coefficient of class ``b``, held in
    ``ModelSpec.gp_terms`` rather than among the population-level terms; its parameters are brms's ``sdgp``, ``lscale`` and
    ``zgp`` (``autobayes.gaussian_process`` has the mathematics).

    Nothing but ``column``, ``k`` and ``c`` is known until the data are seen. Then ``inputs`` holds the distinct fitted
    values in the order they first appear (brms's grouping of equal inputs, ``gr = TRUE``: one latent value per distinct
    input, in brms's order, which is also the order of the exact GP's ``zgp`` and fixes its Cholesky factor), ``dmax`` the
    largest distance between them (brms's ``dmax``: the inputs are scaled by it, so that the largest distance is 1) and
    ``cmean`` the mean of the distinct scaled inputs (brms's ``cmeans``, at which the approximate GP's basis is centred)."""

    column: str
    k: int | None = None
    c: float | None = None
    inputs: tuple[float, ...] = ()
    dmax: float | None = None
    cmean: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(float(v) for v in self.inputs))
        if (self.k is None) != (self.c is None):
            raise ValueError(f"a Gaussian process of {self.column!r} is exact (no k, no c) or approximate (k and c), got k = "
                             f"{self.k!r}, c = {self.c!r}")

    @property
    def term(self) -> str:
        """The term as the canonical formula writes it: ``gp(x)``, or ``gp(x, k = 10, c = 1.25)`` with brms's default ``c``
        written out."""
        return f"gp({self.column})" if self.k is None else f"gp({self.column}, k = {self.k}, c = {_number_text(self.c)})"

    @property
    def approximate(self) -> bool:
        """Whether this is the Hilbert-space approximation (``k`` given) rather than the exact GP."""
        return self.k is not None

    @property
    def known(self) -> bool:
        """Whether the inputs and their scaling have been read from the data."""
        return self.dmax is not None

    def _need_data(self):
        if not self.known:
            raise ValueError(f"the inputs of {self.term!r} are known once the data are seen")

    @property
    def size(self) -> int:
        """The length of ``zgp``: the number of distinct inputs (exact) or of basis functions (approximate)."""
        if self.approximate:
            return int(self.k)
        self._need_data()
        return len(self.inputs)

    @property
    def L(self) -> float:
        """The approximate GP's boundary: ``c`` times the largest distance between scaled inputs, which is 1 (brms's ``Lgp``)."""
        if not self.approximate:
            raise ValueError(f"{self.term!r} is an exact Gaussian process; L belongs to the approximate one, gp(x, k = ...)")
        return float(self.c)

    @property
    def slambda(self) -> np.ndarray:
        """The approximate GP's frequencies ``sqrt(lambda_j) = pi j / (2 L)`` (brms's ``slambda``)."""
        from autobayes.gaussian_process import sqrt_eigenvalues

        return sqrt_eigenvalues(self.L, self.k)

    @property
    def scaled_inputs(self) -> np.ndarray:
        """The distinct inputs divided by ``dmax`` (brms's ``Xgp_prior``, and the exact GP's ``Xgp``), in order."""
        self._need_data()
        return np.asarray(self.inputs, dtype=float) / self.dmax

    def values(self, s: pd.Series, *, where: str = "column") -> np.ndarray:
        """The column ``s`` as the GP's inputs, a float array: refuses a non-numeric, boolean, complex, missing or infinite
        column; ``where`` names the frame in messages."""
        label = f"{where} {self.column!r} of the Gaussian process {self.term!r}"
        reject_complex(label, s)
        if not _is_numeric(s):
            raise FormulaError(f"{label} has dtype {s.dtype}; the input of a Gaussian process is a numeric column")
        v = s.to_numpy(dtype=float)
        if not np.all(np.isfinite(v)):
            raise FormulaError(f"{label} has missing or non-finite values")
        return v

    def index(self, s: pd.Series, *, where: str = "column") -> np.ndarray:
        """Each row's 0-based position among the distinct fitted inputs (brms's ``Jgp`` less 1). A value that is not a
        fitted input is refused: the exact GP has no value there (brms draws one from the conditional normal; divergence
        D65)."""
        self._need_data()
        v = self.values(s, where=where)
        position = {x: j for j, x in enumerate(self.inputs)}
        unknown = sorted({float(x) for x in v if float(x) not in position})
        if unknown:
            shown = unknown[:5] + (["..."] if len(unknown) > 5 else [])
            raise FormulaError(
                f"{where} {self.column!r} has the values {shown}, which are not inputs of the exact Gaussian process "
                f"{self.term!r}: it has values at its {len(self.inputs)} fitted inputs only (brms draws a value at a new input "
                "from the conditional normal, which is not offered here); predict at fitted inputs, or fit the approximate GP "
                f"gp({self.column}, k = ...), whose basis is evaluated at any input inside its boundary (docs/gaussian-processes.md, "
                "divergence D65)")
        return np.array([position[float(x)] for x in v], dtype=np.int64)

    def centred(self, values) -> np.ndarray:
        """Inputs as the approximate GP's basis reads them: divided by ``dmax`` and less ``cmean``."""
        self._need_data()
        return np.asarray(values, dtype=float) / self.dmax - self.cmean

    def basis(self, s: pd.Series, *, where: str = "column") -> np.ndarray:
        """``(rows, k)``: the approximate GP's basis functions at the rows of ``s`` (brms's ``Xgp`` at those inputs). An input
        outside ``(-L, L)`` is refused: the Hilbert-space approximation holds inside its boundary only (divergence D66)."""
        from autobayes.gaussian_process import basis

        x = self.centred(self.values(s, where=where))
        outside = np.flatnonzero(np.abs(x) >= self.L)
        if outside.size:
            i = int(outside[0])
            raise FormulaError(
                f"{where} {self.column!r} has the value {float(s.iloc[i])!r} (row {i}), outside the boundary of the approximate "
                f"Gaussian process {self.term!r}: its basis lives on [-L, L] = [-{_number_text(self.L)}, {_number_text(self.L)}] "
                f"of the scaled, centred input ({self.column} / {self.dmax!r} - {self.cmean!r}), that is {self.column} in "
                f"({(self.cmean - self.L) * self.dmax!r}, {(self.cmean + self.L) * self.dmax!r}); beyond it the basis repeats "
                "itself and approximates nothing (brms evaluates it anyway; divergence D66): refit with a larger c")
        return basis(x, self.L, self.k)


def gaussian_process_coding(gp: GaussianProcess, s: pd.Series) -> GaussianProcess:
    """``gp`` with its inputs read from the fitted column ``s``: the distinct values in the order they first appear (equal
    values, ``0.0`` and ``-0.0`` included, are one input, as R's ``unique`` takes them), ``dmax`` the largest distance between
    them and ``cmean`` the mean of the distinct scaled inputs. Refused: a column that is not numeric or has missing or
    infinite values, fewer than two distinct values (brms divides by their largest distance, which is then zero), and an
    approximate GP whose ``c`` leaves a fitted input outside its boundary (with ``c >= 1`` none is: the centred, scaled
    inputs lie strictly inside ``(-1, 1)``)."""
    v = gp.values(s)
    inputs = tuple(dict.fromkeys(float(x) for x in v))           # first appearance; 0.0 == -0.0 is one key
    if len(inputs) < 2:
        raise FormulaError(f"column {gp.column!r} of the Gaussian process {gp.term!r} takes the single value {inputs[0]!r} in the "
                           "fitted rows; a Gaussian process needs at least two distinct inputs (brms scales the inputs by the largest "
                           "distance between them, which is zero here)")
    dmax = max(inputs) - min(inputs)
    scaled = np.asarray(inputs) / dmax
    coded = replace(gp, inputs=inputs, dmax=float(dmax), cmean=math.fsum(scaled.tolist()) / len(inputs))
    if gp.approximate:
        centred = coded.centred(inputs)
        if np.max(np.abs(centred)) >= coded.L:
            raise FormulaError(
                f"{gp.term!r}: c = {_number_text(gp.c)} puts fitted inputs outside the approximate Gaussian process's boundary "
                f"[-L, L] = [-{_number_text(coded.L)}, {_number_text(coded.L)}] of the scaled, centred input (the largest is "
                f"{float(np.max(np.abs(centred)))!r}), where the approximation does not hold (brms fits it anyway; divergence D66); "
                "use c >= 1 (brms's default is 5/4)")
    return coded


_GP_ARGUMENTS = "by, k, cov, iso, gr, cmc, scale and c"
_TRUE, _FALSE = ("TRUE", "T"), ("FALSE", "F")


def _parse_gp(piece: str, inner: str, response: str, trials: str | None) -> GaussianProcess:
    """The ``GaussianProcess`` of ``gp(<inner>)`` (C03): one column, then brms's named arguments. Accepted: ``k`` (a positive
    whole number: the approximate GP), ``c`` (a positive number or a ratio of two, ``5/4``; with ``k`` only, brms's default
    5/4), and the defaults written out, ``cov = "exp_quad"``, ``gr = TRUE``, ``scale = TRUE``. Refused, naming the construct
    (divergence D64): an expression, several covariates, ``by``, another kernel, ``iso``, ``gr = FALSE``, ``cmc``,
    ``scale = FALSE``, ``c`` without ``k``, the response or the trials column."""
    hint = "(docs/gaussian-processes.md, divergence D64)"
    args = [a.strip() for a in _split_top_level(inner, ",")] if inner.strip() else []
    if not args or not args[0] or re.fullmatch(rf"{_IDENT}\s*=.*", args[0], re.S):
        raise FormulaError(f"{piece!r}: gp() takes the column of its input first, gp(x)")
    column, rest = args[0], args[1:]
    if not re.fullmatch(_IDENT, column):
        raise FormulaError(f"{piece!r}: gp() takes a column name here, not an expression ({column!r}; brms evaluates one): compute "
                           f"the column in the data frame and write gp(<column>) {hint}")
    named: dict[str, str] = {}
    for arg in rest:
        km = re.fullmatch(rf"({_IDENT})\s*=\s*(.+)", arg, re.S)
        if km is None:
            raise FormulaError(
                f"{piece!r}: one covariate per gp() here; brms's gp({column}, {arg}, ...) is one Gaussian process of several "
                f"covariates (one kernel over their joint distance, or one length-scale each with iso = FALSE), which is not part "
                f"of this slice; add a separate gp() per covariate if that is the model meant {hint}")
        name, value = km.group(1), km.group(2).strip()
        if name in named:
            raise FormulaError(f"{piece!r}: gp()'s argument {name!r} is given twice")
        named[name] = value
    k = c = None
    for name, value in named.items():
        if name == "k":
            if value == "NA":
                raise FormulaError(f"{piece!r}: k = NA is brms's exact Gaussian process: drop k, gp({column})")
            if not re.fullmatch(r"\d+", value) or int(value) < 1:
                raise FormulaError(f"{piece!r}: k is the number of basis functions of the approximate Gaussian process, a positive "
                                   f"whole number, got {value!r}")
            k = int(value)
        elif name == "c":
            m = re.fullmatch(rf"({_NUMBER})(?:\s*/\s*({_NUMBER}))?", value)
            c = None if m is None else float(m.group(1)) / (float(m.group(2)) if m.group(2) is not None else 1.0)
            if c is None or not math.isfinite(c) or c <= 0:
                raise FormulaError(f"{piece!r}: c is the approximate Gaussian process's boundary factor, a positive number such "
                                   f"as 1.25 or 5/4, got {value!r}")
        elif name == "cov":
            kernel = value.strip("'\"")
            if value[:1] not in "'\"" or value[-1:] != value[:1] or kernel != "exp_quad":
                raise FormulaError(f"{piece!r}: cov = {value} is not offered here; only brms's default squared-exponential kernel "
                                   f"cov = \"exp_quad\" is part of this slice (brms's matern32, matern52 and exponential kernels "
                                   f"are not) {hint}")
        elif name in ("gr", "scale"):
            if value in _FALSE:
                why = ("gives every row its own latent value even where inputs repeat; here rows with equal inputs share one, "
                       "brms's default gr = TRUE, which is the same model" if name == "gr" else
                       "keeps the inputs unscaled, which changes what lscale measures and the default lscale prior; here the "
                       "inputs are scaled so that their largest distance is 1, brms's default scale = TRUE")
                raise FormulaError(f"{piece!r}: {name} = FALSE {why} {hint}")
            if value not in _TRUE:
                raise FormulaError(f"{piece!r}: {name} takes TRUE or FALSE, got {value!r}")
        elif name == "by":
            raise FormulaError(f"{piece!r}: brms's by = {value} (a Gaussian process scaled by a numeric column, or one per level "
                               f"of a factor) is not part of this slice {hint}")
        elif name in ("iso", "cmc"):
            why = ("decides whether several covariates share one length-scale, and with one covariate brms ignores it"
                   if name == "iso" else "only matters with a factor by =, which is not part of this slice")
            raise FormulaError(f"{piece!r}: gp()'s {name} {why}: drop it {hint}")
        else:
            raise FormulaError(f"{piece!r}: gp() has no argument {name!r}; brms's gp() takes {_GP_ARGUMENTS}")
    if c is not None and k is None:
        raise FormulaError(f"{piece!r}: c is the boundary factor of the approximate Gaussian process, which k = <number of basis "
                           f"functions> asks for; give k as well, or drop c for the exact Gaussian process (brms ignores c there)")
    if column == response:
        raise FormulaError(f"{piece!r}: the response {response!r} cannot be the input of a Gaussian process")
    if trials is not None and column == trials:
        raise FormulaError(f"{piece!r}: the trials column {trials!r} cannot be the input of a Gaussian process")
    return GaussianProcess(column, k=k, c=(1.25 if c is None else c) if k is not None else None)


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
    coefficient (brms's ``bsp``) and simplex act through its own block, ``ModelSpec.monotonic(col)`` its coding. brms's
    ``"measurement_error"`` term ``me(x, sdx)`` and ``"missing"`` term ``mi(col)`` (C06, docs/missing-and-measurement-error.md)
    are special terms like it: a coefficient (brms's ``bsp``) and no design column; the first multiplies the latent values of
    a predictor observed with known noise, the second a column whose missing values are parameters of the model.

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
        if self.kind == "measurement_error" and (measurement_error_columns(self.name) is None or self.columns or self.levels):
            raise ValueError(f"PopTerm: a measurement-error term is spelled 'me(x, sdx)' and has no design column or level, got {self.name!r}")
        if self.kind == "missing" and (missing_column(self.name) is None or self.columns or self.levels):
            raise ValueError(f"PopTerm: a missing-value term is spelled 'mi(col)' and has no design column or level, got {self.name!r}")
        if self.kind == "category_specific" and (category_specific_column(self.name) is None or self.columns or self.levels):
            raise ValueError(f"PopTerm: a category-specific term is spelled 'cs(col)' and has no design column or level, got {self.name!r}")
        if self.categorical is None:
            default = {"numeric": (), "categorical": (self.name,), "interaction": (self.operands[-1],), "transform": (), "monotonic": (),
                       "measurement_error": (), "missing": (), "category_specific": ()}
            object.__setattr__(self, "categorical", default[self.kind])
        else:
            object.__setattr__(self, "categorical", tuple(self.categorical))
        if any(c not in self.operands for c in self.categorical):
            raise ValueError(f"PopTerm {self.name!r}: categorical operands {list(self.categorical)} are not among its operands {list(self.operands)}")

    @property
    def operands(self) -> tuple[str, ...]:
        """The data columns the term is built from: ``(x,)``, ``(z,)``, ``(x, z)`` (numeric first), ``(x, w)`` or
        ``(z, h)`` (main-effect order), ``(xpos,)`` for the transform ``log(xpos)``, ``(ord,)`` for ``mo(ord)``, ``(x, sdx)``
        for ``me(x, sdx)`` and ``(col,)`` for ``mi(col)`` (C06)."""
        if self.kind == "transform":
            return (self.transform.column,)
        if self.kind == "monotonic":
            return (monotonic_column(self.name),)
        if self.kind == "measurement_error":
            return measurement_error_columns(self.name)
        if self.kind == "missing":
            return (missing_column(self.name),)
        if self.kind == "category_specific":                      # D04: cs(col)
            return (category_specific_column(self.name),)
        return tuple(self.name.split(":"))

    @property
    def is_monotonic(self) -> bool:
        """A monotonic term ``mo(col)`` (B03): no design column; its own coefficient and simplex, built by this library (Bambi
        never sees it)."""
        return self.kind == "monotonic"

    @property
    def is_special(self) -> bool:
        """One of brms's special terms (its ``bsp`` coefficients): ``mo(col)`` (B03), ``me(x, sdx)`` or ``mi(col)`` (C06); or a
        category-specific effect ``cs(col)`` (D04, brms's ``bcs``, one coefficient per threshold): a coefficient and no design
        column, built by this library (Bambi never sees it)."""
        return self.kind in SPECIAL_KINDS

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
    # D08 (docs/multi-membership-and-by.md): brms's multi-membership grouping (1 | mm(g, g2, weights = cbind(w1, w2))): the
    # member columns, in order, and their weight columns (empty: equal weights); the factor is then brms's group name "mmgg2"
    members: tuple[str, ...] = ()
    weights: tuple[str, ...] = ()
    # D08: brms's (1 | gr(g, by = gby)): one SD per level of the column ``by``, in which the levels of g are nested
    by: str | None = None
    # D08: brms's correlation ID of (1 + x |p| g), shared with a distributional formula's (1 |p| g): one correlation matrix
    id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "coefs", tuple(self.coefs))
        object.__setattr__(self, "members", tuple(self.members))
        object.__setattr__(self, "weights", tuple(self.weights))

    @property
    def slopes(self) -> tuple[str, ...]:
        """The coefficients other than the varying intercept: the numeric terms whose slope varies by level."""
        return tuple(c for c in self.coefs if c != "Intercept")

    @property
    def has_correlation(self) -> bool:
        """Whether the term carries a correlation matrix: ``|`` with two or more coefficients (brms's ``L_m``,
        ``NC_m > 0`` with ``cor = TRUE``). A single coefficient has nothing to correlate with. A term whose ID it shares
        with a distributional formula (D08) has its correlation matrix through ``ModelSpec.shared_ids`` instead."""
        return self.correlated and len(self.coefs) > 1 and self.id is None

    @property
    def is_mm(self) -> bool:
        """Whether the term is brms's multi-membership term ``(1 | mm(g, g2, ...))`` (D08)."""
        return bool(self.members)

    @property
    def custom(self) -> bool:
        """Whether this library builds the term itself (a multi-membership term, or one with ``by``; D08) rather than Bambi."""
        return self.is_mm or self.by is not None

    @property
    def grouping(self) -> str:
        """The grouping as written inside the term: ``g``, ``g:h``, ``mm(g, g2, weights = cbind(w1, w2))`` or
        ``gr(g, by = gby)`` (D08)."""
        if self.is_mm:
            weights = f", weights = cbind({', '.join(self.weights)})" if self.weights else ""
            return f"mm({', '.join(self.members)}{weights})"
        if self.by is not None:
            return f"gr({self.factor}, by = {self.by})"
        return self.factor

    @property
    def canonical(self) -> str:
        """The term as the canonical formula writes it: ``(1 | g)``, ``(1 + x || g)`` for independent slopes,
        ``(1 + x | g)`` for correlated ones; ``(1 + x |p| g)`` with a correlation ID, ``(1 | mm(g, g2))`` and
        ``(1 | gr(g, by = gby))`` (D08)."""
        inner = " + ".join("1" if c == "Intercept" else c for c in self.coefs)
        bar = "|" if self.correlated or len(self.coefs) == 1 else "||"
        if self.id is not None:
            bar = f"|{self.id}|"
        return f"({inner} {bar} {self.grouping})"

    @property
    def operands(self) -> tuple[str, ...]:
        """The data columns the factor's labels are read from: ``("g",)``, ``("g", "h")`` for the derived factor ``g:h``,
        the member columns of a multi-membership term (D08)."""
        if self.is_mm:
            return self.members
        return tuple(self.factor.split(":"))

    @property
    def data_columns(self) -> tuple[str, ...]:
        """Every data column the term reads: its ``operands``, then a multi-membership term's weight columns and the
        ``by`` column (D08)."""
        return (*self.operands, *self.weights, *((self.by,) if self.by is not None else ()))

    @property
    def derived(self) -> bool:
        """Whether the factor is built from two columns (``g:h``) rather than read from one."""
        return not self.is_mm and len(self.operands) > 1

    @property
    def column(self) -> str:
        """The fitted-frame column holding the factor's labels, which is also the name Bambi sees (``factor_column``).
        A multi-membership term (D08) has none: its labels are its member columns; this is its group name."""
        return factor_column(self.factor)

    @property
    def bambi_canonical(self) -> str:
        """``canonical`` as handed to Bambi: the factor under its column name and ``||`` as ``|`` (formulae has no ``||``);
        a correlation ID is dropped (D08: the model correlates the coefficients across formulas after Bambi builds them)."""
        inner = " + ".join("1" if c == "Intercept" else c for c in self.coefs)
        return f"({inner} | {self.column})"


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
class SharedId:
    """D08: a correlation ID shared by the response formula's term ``(1 + x |p| g)`` and distributional formulas' terms
    ``sigma ~ (1 |p| g)``: brms's one group with one correlation matrix over every coefficient. ``coefs`` are ``(dpar,
    coefficient)`` pairs in brms's order, ``dpar`` empty for the response formula's; ``labels`` are brms's names of them
    (``Intercept``, ``x``, ``sigma_Intercept``), the coordinates of the correlation matrix."""
    id: str
    factor: str
    coefs: tuple[tuple[str, str], ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(c if not dpar else f"{dpar}_{c}" for dpar, c in self.coefs)


def mm_weights(term: GroupTerm, df: pd.DataFrame, *, where: str = "column") -> np.ndarray:
    """D08: the ``(rows, members)`` weights of a multi-membership term on ``df``, as brms's ``W_m_k``: the weight columns
    divided by their sum on each row (brms's default ``scale = TRUE``), or ``1 / members`` each without weight columns.
    A weight must be a finite number at least 0 and each row's weights must have a positive sum, or the frame is refused."""
    k = len(term.members)
    if not term.weights:
        return np.full((len(df), k), 1.0 / k)
    cols = []
    for c in term.weights:
        reject_complex(f"{where} {c!r}", df[c])
        if not _is_numeric(df[c]):
            raise FormulaError(f"{where} {c!r}, a weight of {term.canonical}, must be numeric")
        cols.append(df[c].to_numpy(dtype=float))
    w = np.column_stack(cols) if cols else np.zeros((len(df), 0))
    if not np.isfinite(w).all() or (w < 0).any():
        raise FormulaError(f"the weights {list(term.weights)} of {term.canonical} must be finite and at least 0 on every row "
                           "(brms's scale = TRUE; negative weights need scale = FALSE, which is not offered: divergence D104)")
    total = w.sum(axis=1)
    if (total <= 0).any():
        raise FormulaError(f"the weights {list(term.weights)} of {term.canonical} sum to 0 on row(s) "
                           f"{np.flatnonzero(total <= 0)[:5].tolist()}; each row needs a positive total (brms divides by it)")
    return w / total[:, None]


def by_levels_of(term: GroupTerm, df: pd.DataFrame, *, where: str = "column") -> dict[str, str]:
    """D08: the level of ``term.by`` of each level of a ``gr(g, by = gby)`` term's factor on ``df`` (labels as text), refused
    when a level of ``g`` meets two levels of ``by`` (brms: "Some levels of 'g' correspond to multiple levels of 'gby'")."""
    labels, by = group_labels(term.factor, df).to_numpy(), group_labels(term.by, df).to_numpy()
    out: dict[str, str] = {}
    for lab, b in zip(labels, by):
        if out.setdefault(lab, b) != b:
            raise FormulaError(f"{where}s {term.factor!r} and {term.by!r}: the level {lab!r} of {term.factor!r} meets the levels "
                               f"{out[lab]!r} and {b!r} of {term.by!r}; gr({term.factor}, by = {term.by}) needs each level of "
                               f"{term.factor!r} within one level of {term.by!r} (brms refuses it too)")
    return out


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


ADDITION_KINDS = ("weights", "cens", "trunc", "mi")   # brms's response addition terms besides trials() that are parsed (B02; mi(), C06)

# Addition terms that are parsed, represented and implemented but withheld, each with the reason its refusal carries
# (docs/addition-terms.md): none since P6, which lifted the parks of cens() and trunc() (divergences D24 and D25).
PARKED_ADDITIONS = MappingProxyType({})


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
    ``weights`` and ``cens``. ``addition_data`` turns the terms into brms's Stan data on a frame. ``"mi"`` (``mi()``, C06,
    docs/missing-and-measurement-error.md) says that the response's missing values are parameters of the model (brms's
    ``Ymi``) rather than rows to drop; it reads no column and is no likelihood weight (``addition_data`` skips it)."""

    kind: str
    column: str | None = None
    lb: float | None = None
    ub: float | None = None

    @property
    def term(self) -> str:
        """The term as the canonical formula writes it: ``weights(w)``, ``cens(c)``, ``trunc(lb = 0)``, ``trunc(lb = 0, ub = 5)``,
        ``mi()``."""
        if self.kind == "trunc":
            sides = [f"{side} = {_number_text(v)}" for side, v in (("lb", self.lb), ("ub", self.ub)) if v is not None]
            return f"trunc({', '.join(sides)})"
        if self.kind == "mi":
            return "mi()"
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
                "(docs/addition-terms.md), and mi() in a multivariate list of formulas (docs/missing-and-measurement-error.md); "
                "a binomial response is exactly 'successes | trials(n)', with any other terms after it."
            )
        if kind in seen:
            raise FormulaError(f"{formula!r}: {kind}() appears more than once on the left-hand side")
        seen.add(kind)
        args = [a.strip() for a in _split_top_level(m.group(2), ",")] if m.group(2).strip() else []
        if kind == "trunc":
            additions.append(_parse_trunc(args, piece, formula))
            continue
        if kind == "mi":                                        # C06: the response's missing values are parameters (brms's Ymi)
            if args:
                raise FormulaError(
                    f"{formula!r}: {piece!r}: mi() on the response takes no argument here; brms's mi(sdy), a response measured "
                    "with known noise, is not part of this slice (docs/missing-and-measurement-error.md, divergence D72)"
                )
            additions.append(AdditionTerm("mi"))
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
    others = [a.term for a in additions if a.kind != "mi"]
    if others and len(others) < len(additions):                # C06: mi() is verified alone on the left-hand side
        raise FormulaError(
            f"{formula!r}: mi() together with {', '.join(others)} on the left-hand side is not verified in this slice: brms "
            "would weight, censor or truncate the imputed rows too (docs/missing-and-measurement-error.md, divergence D72)"
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
    truncation bounds is refused, as brms refuses it. Empty without addition terms; ``mi()`` (C06) has no such data (its missing
    rows are the model's ``Design.missing``)."""
    out: dict[str, np.ndarray] = {}
    for a in spec.additions:
        if a.kind == "weights":
            out["weights"] = _weights_values(a.column, df[a.column], where)
        elif a.kind == "cens":
            out["cens"] = _cens_values(a.column, df[a.column], where)
        elif a.kind == "mi":
            continue
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
    numeric term; a special term (``mo(col)``, B03; ``me(x, sdx)`` and ``mi(col)``, C06) has neither."""
    if kind in ("numeric", "transform"):
        return PopTerm(name, kind, (name,), (), ())
    if kind in SPECIAL_KINDS:
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
    column in the design (``(n, 0)``): its block is its codes, ``Monotonic.codes`` (B03); nor have ``me(x, sdx)`` and
    ``mi(col)`` (C06), whose values are the model's (``Design.measurement_error``, ``Design.missing``)."""
    n = len(df)
    if term.kind == "numeric":
        return df[term.name].to_numpy(dtype=float)[:, None]
    if term.kind in SPECIAL_KINDS:
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
    seen, as ``categoricals`` does for the categorical terms. ``gp_terms`` are the formula's Gaussian processes (C03,
    ``gp(x)``, ``gp(x, k = 10, c = 5/4)``; ``GaussianProcess``), in formula order, their inputs known once the data are seen:
    terms of the linear predictor that are not population-level terms (no coefficient of class ``b``, no design column).
    ``autocor`` is the formula's autocorrelation term (C04, ``ar()``, ``ma()`` or ``arma()``; ``autocor.Autocor``), or None."""

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
    gp_terms: tuple[GaussianProcess, ...]
    autocor: "Autocor | None" = None

    def __init__(self, formula, response, predictors=(), group=None, intercept=True,
                 levels=(), trials=None, rhs="", terms=(), categoricals=(), interactions=(),
                 *, groups=None, levels_by_group=None, pop_terms=None, group_terms=None, offset=None, additions=(), dpars=(),
                 monotonics=None, gp_terms=(), autocor=None):
        """Keep the original positional constructor: without ``pop_terms`` / ``group_terms`` the term model is
        built from the flat ``predictors``, ``terms``, ``categoricals``, ``interactions`` and ``group(s)``.
        Terms given as mappings (``dataclasses.asdict`` output) are rebuilt, the distributional formulas and the
        monotonic codings too; without ``monotonics`` each monotonic term's coding is its column alone (not yet seen)."""
        categoricals = tuple(c if isinstance(c, Categorical) else Categorical(**c) for c in categoricals)
        if offset is not None and not isinstance(offset, Offset):
            offset = Offset(**offset)
        additions = tuple(a if isinstance(a, AdditionTerm) else AdditionTerm(**a) for a in additions)
        dpars = tuple(d if isinstance(d, ModelSpec) else ModelSpec(**d) for d in dpars)
        if autocor is not None:                                  # C04: rebuilt from a mapping (dataclasses.asdict output)
            from autobayes.autocor import Autocor
            if not isinstance(autocor, Autocor):
                autocor = Autocor(**autocor)
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
        gp_terms = tuple(g if isinstance(g, GaussianProcess) else GaussianProcess(**g) for g in gp_terms)   # C03
        if len({g.column for g in gp_terms}) != len(gp_terms):
            raise ValueError("gp_terms must name each column once")
        values = dict(formula=formula, response=response, intercept=intercept,
                      levels_by_group=_GroupLevels((g, level_map[g]) for g in factors),
                      trials=trials, rhs=rhs, categoricals=categoricals,
                      pop_terms=pop_terms, group_terms=group_terms, offset=offset, additions=additions, dpars=dpars,
                      monotonics=monotonics, gp_terms=gp_terms, autocor=autocor)
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def __hash__(self):
        # Preserve hashability of the frozen specification; factor metadata is ordered.
        return hash((self.formula, self.response, self.intercept,
                     tuple(self.levels_by_group[g] for g in self.groups), self.trials, self.rhs,
                     self.categoricals, self.pop_terms, self.group_terms, self.offset, self.additions)
                    + ((self.dpars,) if self.dpars else ()) + ((self.monotonics,) if self.monotonics else ())
                    + ((self.gp_terms,) if self.gp_terms else ())
                    + ((self.autocor,) if self.autocor is not None else ()))

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
        columns), then the inputs of the Gaussian processes (C03)."""
        return tuple(dict.fromkeys([*(c for t in self.pop_terms for c in t.operands), *(g.column for g in self.gp_terms)]))

    def gp(self, column: str) -> GaussianProcess:
        """The Gaussian process of ``column`` (C03; ``KeyError`` when the formula has no ``gp(column)``)."""
        for g in self.gp_terms:
            if g.column == column:
                return g
        raise KeyError(column)

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

    # -- measurement-error and missing-value terms (C06) ------------------------------------
    @property
    def me_terms(self) -> tuple[PopTerm, ...]:
        """The measurement-error terms ``me(x, sdx)`` (C06), in formula order: population-level terms with no design column."""
        return tuple(t for t in self.pop_terms if t.kind == "measurement_error")

    @property
    def mi_terms(self) -> tuple[PopTerm, ...]:
        """The missing-value terms ``mi(col)`` (C06), in formula order: population-level terms with no design column."""
        return tuple(t for t in self.pop_terms if t.kind == "missing")

    @property
    def cs_terms(self) -> tuple[PopTerm, ...]:
        """The category-specific terms ``cs(col)`` (D04), in formula order: population-level terms with no design column and
        one coefficient per threshold (brms's ``bcs``; its ``Xcs`` columns in this order)."""
        return tuple(t for t in self.pop_terms if t.kind == "category_specific")

    @property
    def missing_columns(self) -> tuple[str, ...]:
        """The columns whose missing values are parameters of this formula's model (C06): the response's under ``| mi()``,
        then each ``mi(col)`` term's column, in formula order without repeats."""
        own = (self.response,) if self.addition("mi") is not None else ()
        return tuple(dict.fromkeys([*own, *(t.operands[0] for t in self.mi_terms)]))

    @property
    def bambi_terms(self) -> tuple[PopTerm, ...]:
        """The population-level terms Bambi builds: all but the special ones (``mo()``, B03; ``me()``, ``mi()``, C06), which this
        library adds to the graph."""
        return tuple(t for t in self.pop_terms if not t.is_special)

    @property
    def coefficient_labels(self) -> tuple[str, ...]:
        """brms's class ``b`` of this formula, one label per coefficient in term order: each design column (``columns``) and
        each special term by its name (``mo(ord)``, B03; ``me(x, sdx)``, ``mi(col)``, C06; brms's ``bsp``), which has a
        coefficient but no design column."""
        return tuple(label for t in self.pop_terms for label in (t.columns if not t.is_special else (t.name,)))

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
    def bambi_group_terms(self) -> tuple[GroupTerm, ...]:
        """The group-level terms Bambi builds: all but the multi-membership and ``by`` terms (D08), which the model adds to
        the linear predictor itself."""
        return tuple(t for t in self.group_terms if not t.custom)

    @property
    def custom_group_terms(self) -> tuple[GroupTerm, ...]:
        """The group-level terms the model builds itself (D08): ``(1 | mm(g, g2, ...))`` and ``(1 | gr(g, by = gby))``."""
        return tuple(t for t in self.group_terms if t.custom)

    @property
    def shared_ids(self) -> tuple["SharedId", ...]:
        """D08: each correlation ID the response formula shares with distributional formulas, ``(1 + x |p| g)`` and
        ``sigma ~ (1 |p| g)``: one group of brms (one ``sd``, ``z``, ``L``) whose coefficients are the response formula's
        term's, then each distributional formula's, in the order of the list. Empty without one."""
        out = []
        for t in self.group_terms:
            if t.id is None or not self.dpars:                # a distributional formula's own spec lists none: its ID is the response's
                continue
            coefs = [("", c) for c in t.coefs]
            coefs += [(d.response, c) for d in self.dpars for u in d.group_terms if u.id == t.id for c in u.coefs]
            out.append(SharedId(t.id, t.factor, tuple(coefs)))
        return tuple(out)

    def shared_id_of(self, factor: str) -> "SharedId | None":
        """The shared correlation ID of ``factor`` (D08), or None."""
        return next((s for s in self.shared_ids if s.factor == factor), None)

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
        """``(column label, term, index within the term or None for a numeric term)`` per column (a special term, ``mo()``,
        ``me()`` or ``mi()``, has none)."""
        out: list[tuple[str, str, int | None]] = []
        for t in self.pop_terms:
            if t.is_special:
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
        parts += ([self.offset.term] if self.offset is not None else []) + [t.bambi_canonical for t in self.bambi_group_terms]
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
    checked by ``parse_formula`` once every term is read.

    D08: brms's correlation ID ``(1 + x |p| g)`` (``id``), the multi-membership term ``(1 | mm(g, g2, weights = cbind(w1,
    w2)))`` and ``(1 | gr(g, by = gby))``, each in the subset docs/multi-membership-and-by.md describes (divergences
    D104-D106); ``gr(g)`` alone is ``g``."""
    term_id = None
    if "|" in factor:                                            # D08: brms's correlation ID, (1 + x |p| g)
        term_id, factor = (p.strip() for p in factor.split("|", 1))
        if bars == "||":
            raise FormulaError(f"{piece!r}: a correlation ID is written '(1 + x |{term_id}| g)'; '||' with an ID is not brms "
                               "syntax ('Invalid syntax used in group-level terms')")
        if not re.fullmatch(_IDENT, term_id) or "|" in factor:
            raise FormulaError(f"{piece!r}: a correlation ID is a name between two bars, '(1 + x |p| g)'")
        if "(" in factor:
            raise FormulaError(f"{piece!r}: a correlation ID with mm() or gr() is not supported in this slice; the ID's grouping "
                               "factor is a column (docs/multi-membership-and-by.md, divergence D106)")
    if re.match(r"mm\s*\(", factor):                              # D08: brms's multi-membership grouping
        return [_parse_mm_term(inner, bars, factor, piece, response, trials)]
    by = None
    gm = re.fullmatch(r"gr\s*\((.*)\)", factor, re.S)
    if gm:                                                       # D08: brms's gr(g) and gr(g, by = gby); its other options refused
        args = [a.strip() for a in _split_top_level(gm.group(1), ",")]
        named = {}
        for a in args[1:]:
            na = re.fullmatch(rf"({_IDENT})\s*=\s*(.+)", a, re.S)
            if not na:
                raise FormulaError(f"{piece!r}: gr() takes the grouping column and by = <column> here (docs/multi-membership-and-by.md)")
            named[na.group(1)] = na.group(2).strip()
        other = [k for k in named if k != "by"]
        if other or not args or not re.fullmatch(_IDENT, args[0]):
            raise FormulaError(
                f"{piece!r}: brms's gr() options {other or ''} are not supported; gr() takes one grouping column and by = <column> "
                "here (gr(g, cor = FALSE) is '||', gr(g, id = 'p') is '|p|'; cov, dist and a grouping expression are not "
                "offered; docs/multi-membership-and-by.md, divergence D105)"
            )
        factor = args[0]
        if "by" in named:
            by = named["by"]
            if not re.fullmatch(_IDENT, by):
                raise FormulaError(f"{piece!r}: by = takes a column name here, got {by!r}")
            if by == factor:
                raise FormulaError(f"{piece!r}: the grouping column {factor!r} cannot be its own by column")
            if by in (response, trials):
                raise FormulaError(f"{piece!r}: the {'response' if by == response else 'trials column'} {by!r} cannot be the by column")
            if term_id is not None:
                raise FormulaError(f"{piece!r}: a correlation ID with gr(by = ) is not supported in this slice (divergence D106)")
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
        if _ME_RE.match(s) or _MI_RE.match(s):                   # C06
            which = ("a measurement-error term", "D73") if _ME_RE.match(s) else ("a missing-value term", "D72")
            raise FormulaError(
                f"{piece!r}: {which[0]} cannot vary by {factor!r} in this slice: brms's '(1 + {s} {bars} {factor})' gives every "
                f"level its own coefficient of it, which is not verified here (docs/missing-and-measurement-error.md, divergence "
                f"{which[1]}); keep {s} as a population-level term"
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
    if (by is not None or term_id is not None) and sep is not None:
        raise FormulaError(f"{piece!r}: {'gr(by = )' if by is not None else 'a correlation ID'} with a combined grouping factor "
                           f"({factor!r}) is not supported in this slice (docs/multi-membership-and-by.md)")
    if by is not None and slopes:
        raise FormulaError(
            f"{piece!r}: varying slopes with gr(by = ) are not supported in this slice: brms gives each level of {by!r} its own "
            "SD of every coefficient and, with '|', its own correlation matrix, which no frozen case verifies; "
            f"'(1 | gr({factor}, by = {by}))' is offered (docs/multi-membership-and-by.md, divergence D105)"
        )
    if not slopes:
        return [GroupTerm(f, ("Intercept",), correlated=True, by=by, id=term_id) for f in factors]
    # '|' is brms's correlated form (A02): the intercept and slope(s) share a correlation matrix under an LKJ prior;
    # '||' the independent form (A01). '(1 + x | g/h)' gives '(1 + x | g) + (1 + x | g:h)', both correlated, as brms.
    return [GroupTerm(f, ("Intercept", *slopes), correlated=(bars == "|"), id=term_id) for f in factors]


def _parse_mm_term(inner: str, bars: str, factor: str, piece: str, response: str, trials: str | None) -> GroupTerm:
    """D08: brms's multi-membership term ``(1 | mm(g, g2, ...))`` or ``(1 | mm(g, g2, ..., weights = cbind(w1, w2, ...)))``:
    each row belongs to the levels of its member columns with the weights given (brms's default ``scale = TRUE`` divides
    each row's weights by their sum; no weights: equal weights), and the group's levels are the union of the members'
    labels. brms's group name is ``mm`` followed by the member columns (``mmgg2``), which is the term's factor here. The
    varying intercept only: slopes (brms's ``mmc()`` and covariates that vary by membership) and ``scale = FALSE`` are
    refused (divergence D104)."""
    m = re.fullmatch(r"mm\s*\((.*)\)", factor, re.S)
    args = [a.strip() for a in _split_top_level(m.group(1), ",")] if m and m.group(1).strip() else []
    members, weights = [], []
    for a in args:
        na = re.fullmatch(rf"({_IDENT})\s*=\s*(.+)", a, re.S)
        if na is None:
            if weights:
                raise FormulaError(f"{piece!r}: the member columns of mm() come before weights = ")
            if not re.fullmatch(_IDENT, a):
                raise FormulaError(f"{piece!r}: mm() takes column names here, not {a!r} (docs/multi-membership-and-by.md)")
            members.append(a)
            continue
        key, value = na.group(1), na.group(2).strip()
        if key != "weights":
            raise FormulaError(
                f"{piece!r}: mm()'s argument {key!r} is not supported in this slice; mm(g, g2, ..., weights = cbind(w1, w2, ...)) "
                "with brms's default scale = TRUE is offered (docs/multi-membership-and-by.md, divergence D104)")
        cm = re.fullmatch(r"cbind\s*\((.*)\)", value, re.S)
        cols = [c.strip() for c in _split_top_level(cm.group(1), ",")] if cm else []
        if not cols or any(not re.fullmatch(_IDENT, c) for c in cols):
            raise FormulaError(f"{piece!r}: weights = takes cbind(<column>, <column>, ...), one weight column per member")
        weights = cols
    if len(members) < 2:
        raise FormulaError(f"{piece!r}: mm() takes two or more member columns, mm(g, g2)")
    if len(set(members)) != len(members):
        raise FormulaError(f"{piece!r}: a member column appears twice in mm()")
    if weights and len(weights) != len(members):
        raise FormulaError(f"{piece!r}: mm() has {len(members)} member columns and {len(weights)} weight columns; give one weight "
                           "column per member")
    for c in (*members, *weights):
        if c in (response, trials):
            raise FormulaError(f"{piece!r}: the {'response' if c == response else 'trials column'} {c!r} cannot be read by mm()")
    if set(members) & set(weights):
        raise FormulaError(f"{piece!r}: a column cannot be both a member and a weight of mm()")
    tokens = [t.strip() for t in _split_top_level(inner)]
    if tokens != ["1"]:
        raise FormulaError(
            f"{piece!r}: only the varying intercept '(1 | mm(...))' is supported for a multi-membership term in this slice; varying "
            "slopes (and brms's mmc()) are not verified by a frozen case (docs/multi-membership-and-by.md, divergence D104)")
    return GroupTerm("mm" + "".join(members), ("Intercept",), correlated=True, members=tuple(members), weights=tuple(weights))


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
    if any(MI_CALL_RE.search(text) for text in formulas):     # C06: a formula per response with mi() is a multivariate model
        raise FormulaError(
            f"{list(formulas)!r}: a list of formulas with mi() is brms's multivariate model with one formula per response, "
            "bf(y ~ mi(x) + ...) + bf(x | mi() ~ ...): it is built by ab.gaussian([...], data, ab.Priors(resp={...}), "
            "rescor=False) for the gaussian family only (docs/missing-and-measurement-error.md, divergence D72)"
        )
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
            nl_hint = ""
            extras = [e.name for e in fam.extra_parameters]            # D01: alpha, beta, ndt take no formula here (D79)
            if name in (fam.scale_parameter, fam.zero_parameter, *fam.fixed_parameters, *fam.learnable_parameters, *extras):
                why = f"brms's {name} ~ ... is not offered here"
            else:
                own = [p for p in (fam.scale_parameter, fam.zero_parameter, *fam.fixed_parameters, *extras) if p is not None]
                why = (f"the {fam.name} family has no parameter {name!r}"
                       + (f" (its parameters besides the mean are {own})" if own else " (it has no parameter besides the mean)"))
                from autobayes.families import FAMILIES

                if not any(name in (o.scale_parameter, o.zero_parameter, *o.fixed_parameters, *o.learnable_parameters,
                                    *(e.name for e in o.extra_parameters)) for o in FAMILIES.values()):   # C01: a second response
                    why += (f"; if {name!r} is a second response, a formula of its own per response (brms's bf(y ~ ...) + "
                            f"bf({name} ~ ...)) is not supported: a multivariate model's responses share one right-hand side, "
                            f"'mvbind(y, {name}) ~ ...' (docs/multivariate.md, divergence D56)")
                nl_hint = (f". If {name!r} is a non-linear parameter of the first formula, build the model with nl=True "
                           "(brms's nl = TRUE; docs/nonlinear.md)")              # C02
            raise FormulaError(f"{text!r}: {why}; {_distributional_offer(fam)}{nl_hint}")
        parsed.append((name, rhs, text.strip()))

    # Route the declared categorical columns: each formula receives the declarations of its own main effects (read
    # without the data); a declaration that names no formula's main effect stays with the response formula, which
    # refuses it as it always has.
    main0 = parse_formula(main, None, family, None, _ids=True)
    if main0.autocor is not None:                             # C04: not next to a distributional formula (D70)
        raise FormulaError(
            f"{main!r}: the autocorrelation term {main0.autocor.term!r} with a distributional formula is not supported in this "
            "slice (brms accepts bf(y ~ x + ar(...), sigma ~ z); docs/autocorrelation.md, divergence D70)"
        )
    dpar0 = [parse_formula(f"{name} ~ {rhs}", None, family, None, _ids=True) for name, rhs, _ in parsed]
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
    spec = parse_formula(main, data, family, main_cat, _ids=True)
    operands = {c for t in spec.group_terms for c in t.operands}
    dpars = []
    for (name, rhs, text), cat in zip(parsed, dpar_cat):
        d = parse_formula(f"{name} ~ {rhs}", None, family, cat or None, _ids=True)
        unshared = [t for t in d.group_terms if t.id is None]
        if unshared:
            raise FormulaError(
                f"{text!r}: group-level terms in a distributional formula ({', '.join(t.canonical for t in unshared)}) are "
                f"not supported in this slice: brms would give {name} its own varying effects; a distributional formula "
                "takes population-level terms only here, and group-level terms only with a correlation ID shared with the "
                f"response formula's term of the same factor, '{name} ~ ... + (1 |p| g)' next to '... + (1 + x |p| g)' "
                "(docs/distributional.md, divergence D32; docs/multi-membership-and-by.md, D106)"
            )
        for t in d.group_terms:                              # D08: the ID's term in the response formula, same factor
            partner = next((u for u in spec.group_terms if u.id == t.id), None)
            if partner is None:
                raise FormulaError(f"{text!r}: the correlation ID {t.id!r} of '{t.canonical}' names no group-level term of the response "
                                   f"formula {main!r}; an ID shares a correlation matrix with that formula's term here (divergence D106)")
            if partner.factor != t.factor:
                raise FormulaError(f"{text!r}: the correlation ID {t.id!r} joins '{partner.canonical}' and '{t.canonical}', two grouping "
                                   "factors; brms refuses it too ('Can only combine group-level terms of the same grouping factors')")
            if t.slopes:
                raise FormulaError(f"{text!r}: '{t.canonical}': varying slopes in a distributional formula are not supported in this "
                                   f"slice; '(1 |{t.id}| {t.factor})' is offered (docs/multi-membership-and-by.md, divergence D106)")
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
        if d.gp_terms:
            raise FormulaError(
                f"{text!r}: a Gaussian process in a distributional formula ({', '.join(g.term for g in d.gp_terms)}) is not supported "
                f"in this slice: brms would give {name} a Gaussian process of its own; gp() belongs to the response formula here "
                "(docs/gaussian-processes.md, divergence D64)"
            )
        if d.autocor is not None:                             # C04: brms too ('Explicit covariance terms can only be specified on mu')
            raise FormulaError(
                f"{text!r}: an autocorrelation term in a distributional formula ({d.autocor.term}) is not supported: it belongs "
                "to the response formula, as in brms (docs/autocorrelation.md, divergence D70)"
            )
        if d.me_terms:                                       # C06
            raise FormulaError(
                f"{text!r}: a measurement-error term in a distributional formula ({', '.join(t.name for t in d.me_terms)}) is not "
                "supported in this slice; me() belongs to the response formula of a univariate Gaussian model here "
                "(docs/missing-and-measurement-error.md, divergence D73)"
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
    shared = {t.id for d in dpars for t in d.group_terms}
    for t in spec.group_terms:                               # D08: an ID of the response formula is shared, or refused
        if t.id is not None and t.id not in shared:
            raise FormulaError(f"{main!r}: the correlation ID {t.id!r} of '{t.canonical}' is used by no distributional formula of the "
                               "list; an ID shares one correlation matrix between the response formula and a distributional formula's "
                               f"term of {t.factor!r} here (write '{t.canonical.replace(f'|{t.id}|', '|')}' without one; divergence D106)")
    custom = [t for t in spec.group_terms if t.custom]
    if custom:                                               # D08: mm() and gr(by = ) are verified without distributional formulas
        raise FormulaError(f"{main!r}: '{custom[0].canonical}' next to a distributional formula is not supported in this slice: no "
                           "frozen brms case verifies the pair (docs/multi-membership-and-by.md, divergences D104, D105)")
    return replace(spec, dpars=tuple(dpars))


def _check_special_terms(spec: ModelSpec, fam, formula: str) -> None:
    """C06: ``me(x, sdx)`` and ``mi(col)`` where a frozen brms case verifies them (``FamilySpec.measurement_error``,
    ``FamilySpec.missing_values``): one ``me()`` term at most (brms correlates the latent values of several, its default
    ``set_mecor(TRUE)``), and one kind of special term per formula (``mo()``, ``me()`` and ``mi()`` share brms's ``bsp``
    vector, whose order is verified for one kind only); ``me()`` is not taken in a formula whose response has ``| mi()``;
    neither term, nor the response's ``| mi()``, next to a Gaussian process ``gp()`` (C03) or an autocorrelation term
    ``ar()``, ``ma()``, ``arma()`` (C04), as no frozen case verifies either pair."""
    from autobayes.families import FAMILIES

    for terms, flag, what, did in ((spec.me_terms, "measurement_error", "measurement-error", "D73"),
                                   (spec.mi_terms, "missing_values", "missing-value", "D72")):
        if terms and not getattr(fam, flag):
            offering = [name for name, other in FAMILIES.items() if getattr(other, flag)]
            raise FormulaError(
                f"{formula!r}: the {what} term {terms[0].name!r} is not available for the {fam.name} family; brms's "
                f"{terms[0].name.split('(')[0]}() is verified here for the {', '.join(offering) or 'no'} "
                f"famil{'y' if len(offering) == 1 else 'ies'} (docs/missing-and-measurement-error.md, divergence {did})"
            )
    if len(spec.me_terms) > 1:
        raise FormulaError(
            f"{formula!r}: several measurement-error terms ({', '.join(t.name for t in spec.me_terms)}) are not supported in this "
            "slice: brms correlates their latent values by default (set_mecor(TRUE), a correlation matrix Lme), which is not "
            "verified here; one me() term per formula (docs/missing-and-measurement-error.md, divergence D73)"
        )
    kinds = [k for k, terms in (("mo()", spec.mo_terms), ("me()", spec.me_terms), ("mi()", spec.mi_terms)) if terms]
    if len(kinds) > 1:
        raise FormulaError(
            f"{formula!r}: {' and '.join(kinds)} terms in one formula are not supported in this slice: they share brms's vector of "
            "special coefficients (bsp), whose order is verified for one kind of term only "
            f"(docs/missing-and-measurement-error.md, divergence {'D73' if 'me()' in kinds else 'D72'})"
        )
    if spec.me_terms and spec.addition("mi") is not None:
        raise FormulaError(
            f"{formula!r}: a measurement-error term in a formula whose response has mi() is not supported in this slice; me() "
            "belongs to a univariate Gaussian model here (docs/missing-and-measurement-error.md, divergence D73)"
        )
    special = ([f"the measurement-error term {t.name!r}" for t in spec.me_terms]
               + [f"the missing-value term {t.name!r}" for t in spec.mi_terms]
               + ([f"the response's mi() ('{spec.response} | mi()')"] if spec.addition("mi") is not None else []))
    others = ([f"the Gaussian process {g.term!r} (C03)" for g in spec.gp_terms]
              + ([f"the autocorrelation term {spec.autocor.term!r} (C04)"] if spec.autocor is not None else []))
    if special and others:                                   # C03 / C04: no frozen brms case verifies either pair
        raise FormulaError(
            f"{formula!r}: {special[0]} next to {others[0]} is not supported in this slice: no frozen brms case verifies the pair "
            f"(docs/missing-and-measurement-error.md, divergence {'D73' if spec.me_terms else 'D72'})"
        )


def check_imputation(spec: ModelSpec, imputed=()) -> None:
    """C06: the missing values a formula's model holds must be imputed by a formula of the same (multivariate) model:
    ``imputed`` names the responses of that model's ``col | mi() ~ ...`` formulas (empty for a univariate model). An
    ``mi(col)`` term needs ``col`` among them (brms refuses an ``mi()`` predictor without its sub-model too), and a response
    with ``| mi()`` must be one of them: a univariate ``y | mi() ~ ...`` is not part of this slice."""
    for t in spec.mi_terms:
        col = t.operands[0]
        if col not in imputed:
            raise FormulaError(
                f"{spec.formula!r}: {t.name!r} reads a column whose missing values are modelled by a formula of its own, "
                f"'{col} | mi() ~ ...', in the same model (brms's bf({spec.formula}) + bf({col} | mi() ~ ...)); write the model as "
                f"a list of formulas, ab.gaussian(['{spec.formula}', '{col} | mi() ~ <terms>'], data, priors, rescor=False) "
                "(docs/missing-and-measurement-error.md, divergence D72)"
            )
    if spec.addition("mi") is not None and spec.response not in imputed:
        raise FormulaError(
            f"{spec.formula!r}: the response's own missing values are imputed here only by the formula of an mi() predictor of "
            f"another response, ab.gaussian(['y ~ mi({spec.response}) + ...', '{spec.formula}'], data, priors, rescor=False); a "
            "univariate y | mi() ~ ... model is not part of this slice (docs/missing-and-measurement-error.md, divergence D72)"
        )


def parse_formula(formula: str | list[str], data: pd.DataFrame | None = None, family: str = "gaussian",
                  categorical: Mapping[str, object] | None = None, *, _ids: bool = False) -> ModelSpec:
    """Parse and validate a formula in the supported subset, optionally against a data frame.

    ``categorical`` maps a predictor column to its reference level (treatment coding); every
    other predictor term must be numeric. ``formula`` may also be a list (B01): the response formula, then one formula
    per distributional parameter, ``["y ~ x + (1 | g)", "sigma ~ z"]`` (``_parse_formulas``, docs/distributional.md).
    ``_ids`` is set by ``_parse_formulas`` alone: a correlation ID ``(1 + x |p| g)`` (D08) is taken in a list of formulas,
    where a distributional formula shares it, and refused in a formula on its own.
    """
    from autobayes.families import get_family

    if not isinstance(formula, str):
        return _parse_formulas(formula, data, family, categorical)
    fam = get_family(family)
    if MVBIND_RE.match(formula):                           # C01: a multivariate formula is one model per response, built elsewhere
        from autobayes.families import FAMILIES

        offering = [name for name, other in FAMILIES.items() if other.multivariate]
        raise FormulaError(
            f"{formula!r}: brms's multivariate formula mvbind(...) ~ ... is built by "
            + ("ab.gaussian(formula, data, ab.Priors(resp={...}), rescor=True | False), one model per response "
               "(docs/multivariate.md); a univariate formula has one response" if fam.multivariate else
               f"ab.{offering[0] if offering else 'gaussian'}(..., rescor=...) for the {', '.join(offering) or 'no'} "
               f"family only; a multivariate {fam.name} model is not offered (docs/multivariate.md, divergence D58)")
        )
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
    special_columns: dict[str, tuple[str, ...]] = {}       # the me() and mi() terms among them (C06): canonical name -> columns
    category_specific: dict[str, str] = {}                 # the cs() terms among them (D04): canonical name -> column
    group_terms: list[GroupTerm] = []
    group_pieces: list[str] = []                           # each group term as written, for the messages below
    groups: list[str] = []                                 # the factors of group_terms, for the checks below
    raw_interactions: list[tuple[str, str, str]] = []      # (left, right, piece as written)
    offsets: list[tuple[Offset, str]] = []                 # (offset term, piece as written); at most one
    gps: list[tuple[GaussianProcess, str]] = []            # the Gaussian processes (C03): (term, piece as written)
    autocors: list = []                                    # (autocorrelation term, piece as written); at most one (C04)
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
        if _AUTOCOR_RE.match(piece):                            # brms's ar(), ma(), arma() (C04); its other structures refused (D69)
            from autobayes.autocor import parse_autocor

            term = parse_autocor(piece, formula)
            if term is None:
                raise FormulaError(f"{piece!r}: an autocorrelation term is written ar(...), ma(...) or arma(...) (docs/autocorrelation.md)")
            autocors.append((term, piece))
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
                if _CS_RE.match(side):                              # D04
                    raise FormulaError(
                        f"{piece!r}: an interaction with a category-specific term ({side!r}) is not supported in this slice: brms's "
                        "cs(x):z is not verified here; compute the product column in the data frame and write cs(<column>) "
                        "(docs/ordinal-categorical.md, divergence D89)"
                    )
                if _GP_RE.match(side):
                    raise FormulaError(
                        f"{piece!r}: an interaction with a Gaussian process ({side!r}) is not supported in this slice; a Gaussian "
                        "process that changes with another variable is brms's gp(x, by = z), which is not offered either "
                        "(docs/gaussian-processes.md, divergence D64)"
                    )
                if _ME_RE.match(side) or _MI_RE.match(side):        # C06
                    which = ("a measurement-error term", "D73") if _ME_RE.match(side) else ("a missing-value term", "D72")
                    raise FormulaError(
                        f"{piece!r}: an interaction with {which[0]} ({side!r}) is not supported in this slice: brms gives it a "
                        "special coefficient of its own times the product, which is not verified here "
                        f"(docs/missing-and-measurement-error.md, divergence {which[1]})"
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
        csm = _CS_RE.match(piece)
        if csm:                                                 # brms's category-specific effect cs(col) (D04; D08 parked it, D107)
            if not fam.category_specific:
                from autobayes.families import FAMILIES

                offering = [name for name, other in FAMILIES.items() if other.category_specific]
                raise FormulaError(
                    f"{piece!r}: brms's category-specific effect cs() is not supported for the {fam.name} family: brms takes it in its "
                    f"ordinal families, and this library in the {', '.join(offering)} families (one coefficient per threshold, brms's "
                    "bcs; docs/ordinal-categorical.md, divergence D89)"
                )
            inner = csm.group(1).strip()
            if not re.fullmatch(_IDENT, inner):
                raise FormulaError(
                    f"{piece!r}: cs() takes one column name here, not {inner!r}; brms's cs(x + w), cs(1) and expressions inside cs() "
                    "are not supported in this slice: write one cs(<column>) per numeric column (docs/ordinal-categorical.md, "
                    "divergence D89)"
                )
            if inner == response:
                raise FormulaError(f"{piece!r}: the response {response!r} cannot be a category-specific predictor")
            if trials is not None and inner == trials:
                raise FormulaError(f"{piece!r}: the trials column {trials!r} cannot be a category-specific predictor")
            _main(f"cs({inner})", piece, call=True)                # under its canonical spelling, 'cs( x )' duplicates 'cs(x)'
            category_specific[f"cs({inner})"] = inner
            continue
        gpm = _GP_RE.match(piece)
        if gpm:                                                 # brms's Gaussian process gp(x), gp(x, k = 10, c = 5/4) (C03)
            gp = _parse_gp(piece, gpm.group(1), response, trials)
            twin = next((p for g, p in gps if g.column == gp.column), None)
            if twin is not None:
                raise FormulaError(f"{piece!r}: a second Gaussian process of {gp.column!r} (after {twin!r}); one gp() per column here")
            gps.append((gp, piece))
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
        me = _ME_RE.match(piece)
        if me:                                                  # brms's measurement error me(x, sdx) (C06)
            args = [a.strip() for a in _split_top_level(me.group(1), ",")] if me.group(1).strip() else []
            named = [a for a in args if re.match(rf"^{_IDENT}\s*=", a)]
            if named or len(args) > 2:
                raise FormulaError(
                    f"{piece!r}: me() takes the noisy column and the column of its known standard deviations here, me(x, sdx); "
                    "brms's gr argument (me(x, sdx, gr = g), latent values shared within the levels of g) is not supported in "
                    "this slice (docs/missing-and-measurement-error.md, divergence D73)"
                )
            if len(args) != 2:
                raise FormulaError(f"{piece!r}: me() takes two column names, the noisy values and their known standard deviations, "
                                   "me(x, sdx) (brms's me())")
            bad = [a for a in args if not re.fullmatch(_IDENT, a)]
            if bad:
                raise FormulaError(
                    f"{piece!r}: me() takes column names here, not expressions ({bad[0]!r}; brms evaluates one): compute the "
                    "column in the data frame and write me(<column>, <noise column>) (docs/missing-and-measurement-error.md, "
                    "divergence D73)"
                )
            x, sdx = args
            for col in (x, sdx):
                if col == response:
                    raise FormulaError(f"{piece!r}: the response {response!r} cannot be read by me()")
                if trials is not None and col == trials:
                    raise FormulaError(f"{piece!r}: the trials column {trials!r} cannot be read by me()")
            if x == sdx:
                raise FormulaError(f"{piece!r}: the noisy column and the column of its standard deviations must differ")
            _main(f"me({x}, {sdx})", piece, call=True)             # under its canonical spelling, 'me(x,sdx)' is 'me(x, sdx)'
            special_columns[f"me({x}, {sdx})"] = (x, sdx)
            continue
        mi = _MI_RE.match(piece)
        if mi:                                                  # brms's missing values mi(col) (C06)
            inner = mi.group(1).strip()
            if len(_split_top_level(inner, ",")) > 1:
                raise FormulaError(
                    f"{piece!r}: mi() takes one column here; brms's idx argument (mi(x, idx), the rows of x's own formula that "
                    "each row of this one reads) is not supported in this slice (docs/missing-and-measurement-error.md, "
                    "divergence D72)"
                )
            if not re.fullmatch(_IDENT, inner):
                raise FormulaError(
                    f"{piece!r}: mi() takes a column name here, the response of another formula of the model, written "
                    "'<column> | mi() ~ ...' (docs/missing-and-measurement-error.md, divergence D72)"
                )
            if inner == response:
                raise FormulaError(f"{piece!r}: the response {response!r} cannot be its own mi() predictor")
            if trials is not None and inner == trials:
                raise FormulaError(f"{piece!r}: the trials column {trials!r} cannot be an mi() predictor")
            _main(f"mi({inner})", piece, call=True)
            special_columns[f"mi({inner})"] = (inner,)
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
    if remove_intercept and not predictors and not gps:     # 'y ~ 0 + gp(x)' leaves the Gaussian process (C03)
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
    if len(predictors) == 0 and not saw_intercept_token and not gps:
        raise FormulaError(
            f"{formula!r}: no population-level predictor; write 'y ~ 1 + (1 | group)' "
            "for an intercept-only model."
        )
    transform_columns = [tr.column for tr in transforms.values()]   # the raw columns the transform terms read
    mo_columns = list(monotonics.values())                           # and the monotonic terms (B03)
    gp_columns = [g.column for g, _ in gps]                          # and the Gaussian processes (C03)
    special_cols = [c for cols in special_columns.values() for c in cols]   # and the me() and mi() terms (C06)
    special_cols += list(category_specific.values())                 # and the cs() terms (D04)
    for col in operands_all:
        if (col in predictors or col in transform_columns or col in mo_columns or col in gp_columns or col in special_cols
                or col == response):
            raise FormulaError(f"{col!r}: the grouping factor cannot also be a predictor or the response")
    for term, piece in zip(group_terms, group_pieces):       # D08: an mm() term's weights and a gr() term's by column
        for col in (*term.weights, *((term.by,) if term.by is not None else ())):
            if (col in predictors or col in transform_columns or col in mo_columns or col in gp_columns or col in special_cols
                    or col in operands_all):
                raise FormulaError(f"{piece!r}: {col!r} is a {'weight' if col in term.weights else 'by'} column of the group term and "
                                   "cannot also be a predictor or a grouping column in this slice")
    ids: dict[str, GroupTerm] = {}
    for term, piece in zip(group_terms, group_pieces):       # D08: one grouping factor per correlation ID, as brms requires
        if term.id is None:
            continue
        if term.id in ids:
            raise FormulaError(f"{formula!r}: the correlation ID {term.id!r} is used by '{ids[term.id].canonical}' and '{piece}' in one "
                               "formula; brms writes the coefficients of one factor in one term ('Can only combine group-level terms "
                               "of the same grouping factors' for different factors); docs/multi-membership-and-by.md, divergence D106")
        ids[term.id] = term
    custom = [(t, p) for t, p in zip(group_terms, group_pieces) if t.custom]
    if custom:                                               # D08: mm() and gr(by = ) next to the terms no frozen case verifies with them
        t, p = custom[0]
        what = "the multi-membership term" if t.is_mm else "the term with gr(by = )"
        others = ([f"the monotonic term {next(iter(monotonics))!r}"] if monotonics else []) + [f"the Gaussian process {g.term!r}" for g, _ in gps]
        others += [f"the term {name!r}" for name in special_columns] + [f"the autocorrelation term {a.term!r}" for a, _ in autocors]
        others += [f"the addition term {a.term!r}" for a in additions]
        if others:
            raise FormulaError(f"{formula!r}: {what} '{p}' next to {others[0]} is not supported in this slice: no frozen brms case "
                               "verifies the pair (docs/multi-membership-and-by.md, divergences D104, D105)")
        if fam.link not in ("identity", "log") or not fam.monotonic or fam.per_category or fam.thresholds or fam.mixture:
            raise FormulaError(
                f"{formula!r}: {what} '{p}' is not available for the {fam.name} family ({fam.link} link): it is added to the linear "
                "predictor as the library builds it and checked there under the identity and log links, for the families whose "
                "linear predictor takes such terms (docs/multi-membership-and-by.md, divergences D104, D105)")
    if trials is not None and (trials in predictors or trials in operands_all):
        raise FormulaError(f"{trials!r}: the trials column cannot also be a predictor or the grouping factor")
    offset = offsets[0][0] if offsets else None
    autocor = None
    if autocors:                                             # C04: one autocorrelation term, for a family that takes it (D70)
        from autobayes.autocor import check_autocor

        if len(autocors) > 1:
            raise FormulaError(
                f"{formula!r}: {autocors[0][1]!r} and {autocors[1][1]!r}: one autocorrelation term per formula (brms: 'Can only "
                "model one term per autocorrelation class'); write arma(time, gr, p, q) for an AR and an MA part"
            )
        autocor = autocors[0][0]
        check_autocor(autocor, autocors[0][1], response=response, trials=trials, fam=fam, additions=additions, formula=formula)
        if gps:                                              # C03: not verified together (brms accepts the pair; D70)
            raise FormulaError(
                f"{formula!r}: the autocorrelation term {autocor.term!r} next to the Gaussian process {gps[0][0].term!r} is not "
                "supported in this slice: no frozen brms case verifies the pair (brms takes it, the residuals then those of the "
                "predictor with the process); docs/autocorrelation.md, divergence D70"
            )
    for name, cols in special_columns.items():                # C06: a column me() or mi() reads is read by nothing else
        elsewhere = [*(p for p in predictors if p != name and p not in special_columns), *transform_columns, *mo_columns, *gp_columns,
                     *(c for other, oc in special_columns.items() if other != name for c in oc),
                     *([offset.column] if offset is not None else []), *(c for a in additions for c in a.columns),
                     *(c for left, right, _ in raw_interactions for c in (left, right))]
        shared = [c for c in cols if c in elsewhere]
        if shared:
            raise FormulaError(
                f"{formula!r}: {shared[0]!r} is read by {name!r} and by another term of the formula; a column inside me() or "
                "mi() is read by that term alone in this slice (docs/missing-and-measurement-error.md)"
            )
    for name, col in category_specific.items():                # D04: a column inside cs() is read by that term alone
        elsewhere = [*(p for p in predictors if p != name and p not in category_specific), *transform_columns, *mo_columns, *gp_columns,
                     *(c for cols in special_columns.values() for c in cols), *([offset.column] if offset is not None else []),
                     *(c for a in additions for c in a.columns), *(c for left, right, _ in raw_interactions for c in (left, right)),
                     *(s for t in group_terms for s in t.slopes)]
        if col in elsewhere:
            raise FormulaError(
                f"{formula!r}: {col!r} is read by {name!r} and by another term of the formula; a column inside cs() is read by that "
                "term alone in this slice (brms accepts x + cs(x), whose b_x and bcs_x are not separately identified; "
                "docs/ordinal-categorical.md, divergence D89)"
            )
    # A derived factor's column (g_h) is added to the fitted frame: it must not shadow a column the formula uses.
    in_use = {response, *predictors, *transform_columns, *mo_columns, *gp_columns, *special_cols, *operands_all,
              *([trials] if trials is not None else []),
              *([offset.column] if offset is not None else []), *(c for a in additions for c in a.columns)}
    in_use |= set(autocor.columns) if autocor is not None else set()   # the time and series columns (C04)
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
        for g, piece in gps:                                 # nor does gp(), whose input is a number (C03)
            if g.column in categorical:
                raise FormulaError(f"{piece!r}: {g.column!r} is declared categorical; the input of a Gaussian process is a numeric "
                                   "column, and one column cannot be both")
        for name, cols in special_columns.items():           # nor do me() and mi(), which read numbers (C06)
            declared = [c for c in cols if c in categorical]
            if declared:
                raise FormulaError(f"{name!r}: {declared[0]!r} is declared categorical; {name.split('(')[0]}() reads numbers")
        for name, col in category_specific.items():          # nor does cs() here (D04): brms's cs(z) of a factor is not verified
            if col in categorical:
                raise FormulaError(f"{name!r}: {col!r} is declared categorical; cs() takes a numeric column in this slice (brms's cs() "
                                   "of a factor, one coefficient per level and threshold, is not verified here; divergence D89)")
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
    parts = ([*predictors, *interactions] + [g.term for g, _ in gps] + ([offset.term] if offset is not None else [])
             + [t.canonical for t in group_terms])
    parts += [autocor.term] if autocor is not None else []        # the autocorrelation term, last (C04)
    canonical_rhs = " + ".join(parts) if parts else "1"           # intercept-only population model: 'y ~ 1'
    if remove_intercept:
        canonical_rhs = "0 + " + canonical_rhs
    kinds = {**{p: "transform" for p in transforms}, **{p: "monotonic" for p in monotonics}, **{p: "categorical" for p in cat_names},
             **{p: "measurement_error" if p.startswith("me(") else "missing" for p in special_columns},   # C06
             **{p: "category_specific" for p in category_specific}}                                     # D04
    pop_terms = (*(pop_term(p, kinds.get(p, "numeric"), categoricals) for p in predictors),
                 *(pop_term(name, "interaction", categoricals) for name in interactions))
    spec = ModelSpec(
        formula=formula.strip(), response=response, intercept=not remove_intercept, trials=trials,
        rhs=canonical_rhs, categoricals=categoricals, pop_terms=pop_terms, group_terms=tuple(group_terms),
        offset=offset, additions=additions, gp_terms=tuple(g for g, _ in gps), autocor=autocor,
    )
    if spec.gp_terms and fam.per_category:                   # one linear predictor per category (C03)
        raise FormulaError(
            f"{formula!r}: a Gaussian process ({spec.gp_terms[0].term}) is not available for the {fam.name} family: brms would give "
            "each non-reference category its own Gaussian process, which is not part of this slice (docs/gaussian-processes.md, "
            "divergence D64)"
        )
    _check_special_terms(spec, fam, formula)                 # C06: me() and mi() where a frozen brms case verifies them
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
    with_id = next((t for t in spec.group_terms if t.id is not None), None)
    if with_id is not None and not _ids:                     # D08: an ID is shared with a distributional formula of a list
        raise FormulaError(
            f"{formula!r}: the correlation ID {with_id.id!r} of '{with_id.canonical}' is offered between the response formula and a "
            f"distributional formula of a list, ['y ~ x + (1 + x |{with_id.id}| g)', 'sigma ~ (1 |{with_id.id}| g)'] (brms's bf(...)); "
            f"in a formula on its own it is not supported here (docs/multi-membership-and-by.md, divergence D106)")
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


def validate_nonnegative_response(name: str, s: pd.Series, family_label: str) -> None:
    """A hurdle family's continuous response (hurdle Gamma, hurdle lognormal; D03): a real numeric column of finite values,
    zero or positive (the zeros are the hurdle's). Missing values are left to the shared missing-data check that follows."""
    validate_numeric_response(name, s, family_label)
    v = s.to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    if np.any(np.isinf(v)):
        raise FormulaError(f"response {name!r} has a non-finite value; the {family_label} family needs finite non-negative responses.")
    if np.any(v < 0):
        raise FormulaError(
            f"response {name!r} has a negative value ({float(v[v < 0][0])!r}); the {family_label} family needs non-negative responses "
            "(an exact zero is the hurdle's zero part)."
        )


def validate_zero_proportion_response(name: str, s: pd.Series, family_label: str) -> None:
    """A zero-inflated proportion's response (zero-inflated Beta; D03): a real numeric column of values in [0, 1), the exact
    zeros being the zero part. Missing values are left to the shared missing-data check that follows."""
    validate_numeric_response(name, s, family_label)
    v = s.to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    if np.any(np.isinf(v)):
        raise FormulaError(f"response {name!r} has a non-finite value; the {family_label} family needs proportions in [0, 1).")
    bad = v[(v < 0) | (v >= 1)]
    if bad.size:
        raise FormulaError(
            f"response {name!r} has a value outside [0, 1) ({float(bad[0])!r}); the {family_label} family needs proportions in [0, 1): "
            "an exact 0 is the zero part, an exact 1 is refused (brms's zero_one_inflated_beta and xbeta model exact 1s: "
            "ab.zero_one_inflated_beta(), ab.xbeta())."
        )


def validate_unit_interval_response(name: str, s: pd.Series, family_label: str) -> None:
    """A response on the closed unit interval (E01: the zero-one-inflated Beta and the extended-support Beta, xbeta): a real
    numeric column of values in [0, 1], exact zeros and ones included. Missing values are left to the shared missing-data check
    that follows."""
    validate_numeric_response(name, s, family_label)
    v = s.to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    if np.any(np.isinf(v)):
        raise FormulaError(f"response {name!r} has a non-finite value; the {family_label} family needs proportions in [0, 1].")
    bad = v[(v < 0) | (v > 1)]
    if bad.size:
        raise FormulaError(f"response {name!r} has a value outside [0, 1] ({float(bad[0])!r}); the {family_label} family needs "
                           "proportions in [0, 1] (exact 0s and 1s included).")


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
    need = ("the ordinal families (cumulative, sratio, cratio, acat) need an ordinal response: positive integers 1..K or an "
            "ordered pandas Categorical (pd.Categorical(values, categories=[lowest, ..., highest], ordered=True), brms's ordered "
            "factor)")
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


def hurdle_ordinal_categories(name: str, s: pd.Series) -> tuple[tuple[str, ...], np.ndarray, list[str], str]:
    """brms's hurdle_cumulative response (E01, docs/bounded-families.md), as brms 2.23's ``standata()`` codes it (probed): whole
    numbers ``0..max(y)``, 0 the hurdle category and ``1..max(y)`` the ordinal part's categories (every one, observed or not,
    as for the cumulative family; brms: "requires either non-negative integers or ordered factors as responses"), or an ordered
    pandas Categorical, whose first declared category is the hurdle (coded 0, as brms codes an ordered factor's first level) and
    whose others, in their declared order, the ordinal part's. The ordinal part needs at least two categories (one threshold;
    brms: "Could not extract the number of thresholds"). Returns the ordinal part's categories as labels, each row's code
    (brms's ``Y``: 0 for the hurdle, ``k`` for the ordinal part's k-th category), notes naming the ordinal categories no fitted
    row takes, and the hurdle category's label."""
    label = f"response {name!r}"
    reject_complex(label, s)
    need = ("the hurdle_cumulative family needs whole numbers 0..K (0 the hurdle category) or an ordered pandas Categorical (its "
            "first declared category the hurdle), brms's non-negative integers or ordered factor")
    if isinstance(s.dtype, pd.CategoricalDtype):
        if not s.cat.ordered:
            raise FormulaError(f"{label} is an unordered pandas Categorical, whose category order means nothing; {need}")
        declared = tuple(_scalar(label, c, where="declared category") for c in s.cat.categories)
        if len(set(declared)) != len(declared):
            raise FormulaError(f"{label} has declared categories that collide after normalisation: {list(declared)}")
        if len({_kind(c) for c in s.cat.categories}) > 1:
            raise FormulaError(f"{label} mixes text labels and numeric codes in its declared categories; recode it explicitly")
        if s.isna().any():
            raise FormulaError(f"{label} has missing values; drop or impute them explicitly before fitting.")
        hurdle, categories = declared[0], declared[1:]
        codes = s.cat.codes.to_numpy().astype(np.int64)
    else:
        if pd.api.types.is_bool_dtype(s):
            raise FormulaError(f"{label} is boolean; {need}")
        if not pd.api.types.is_numeric_dtype(s):
            raise FormulaError(f"{label} has dtype {s.dtype}; {need}. Text labels have no order: declare it with an ordered "
                               "Categorical")
        v = s.to_numpy(dtype=float)
        if np.isnan(v).any():
            raise FormulaError(f"{label} has missing values; drop or impute them explicitly before fitting.")
        bad = v[~np.isfinite(v) | (v != np.floor(v)) | (v < 0)]
        if bad.size:
            raise FormulaError(f"{label} has the value {bad[0]:g}; {need} (brms: 'requires either non-negative integers or "
                               "ordered factors as responses')")
        if v.max() > COUNT_CAP:
            raise FormulaError(f"{label} has the value {v.max():.0f}, which would make {v.max():.0f} categories; an ordinal "
                               f"response above {COUNT_CAP} is refused")
        hurdle, categories = "0", tuple(str(k) for k in range(1, int(v.max()) + 1))
        codes = v.astype(np.int64)
    if len(categories) < 2:
        raise FormulaError(f"{label} has {len(categories)} categor{'y' if len(categories) == 1 else 'ies'} above the hurdle category "
                           f"{hurdle!r}; the ordinal part needs at least two (one threshold; brms: 'Could not extract the number of "
                           "thresholds')")
    notes = []
    unobserved = [c for k, c in enumerate(categories, start=1) if not (codes == k).any()]
    if unobserved:
        notes.append(f"categories {unobserved} of the ordinal response {name!r} have no fitted rows: the thresholds next to them "
                     "are informed by the ordering and the prior alone (brms keeps them too)")
    if (codes == 0).all():
        notes.append(f"every fitted row of {name!r} is the hurdle category {hurdle!r}: the ordinal part has no data, so its "
                     "thresholds and coefficients are informed by their priors alone; check sensitivity to the explicit priors")
    return categories, codes, notes, hurdle


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


def missing_values(label: str, s: pd.Series, response: bool = False) -> np.ndarray:
    """A column whose missing values are parameters of the model (C06: the response of ``col | mi() ~ ...``, or the column of
    an ``mi(col)`` term): real numeric, NaN (or pandas's NA) marking a missing row, every other value finite, and at least one
    observed value. Returns the values as float64 with NaN at the missing rows."""
    reject_complex(label, s)
    if not _is_numeric(s):
        raise FormulaError(f"{label} has dtype {s.dtype}; a column with missing values modelled by mi() must be numeric, NaN "
                           "marking a missing value")
    v = s.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(v).any():
        raise FormulaError(f"{label} has an infinite value at row {int(np.flatnonzero(np.isinf(v))[0])}; a missing value is NaN")
    if np.isnan(v).all():
        raise FormulaError(f"{label} has no observed value; {'its formula' if response else 'mi()'} needs observed rows")
    return v


def measurement_values(term: str, x: pd.Series, sd: pd.Series, *, where: str = "column") -> tuple[np.ndarray, np.ndarray]:
    """The noisy values and their known standard deviations of ``me(x, sdx)`` (C06, brms's ``Xn`` and ``noise``): real numeric
    and finite (missing noisy values are brms's ``me()`` with ``mi()``, not part of this slice), the standard deviations
    positive (Stan's normal density needs a positive scale)."""
    out = []
    for s, what in ((x, "noisy values"), (sd, "known standard deviations")):
        reject_complex(f"{where} {s.name!r}", s)
        if not _is_numeric(s):
            raise FormulaError(f"{where} {s.name!r} ({what} of {term!r}) has dtype {s.dtype}; it must be numeric")
        v = s.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(v).all():
            raise FormulaError(f"{where} {s.name!r} ({what} of {term!r}) has missing or non-finite values (missing noisy values "
                               "are brms's me() with mi(), not part of this slice; docs/missing-and-measurement-error.md, "
                               "divergence D73)")
        out.append(v)
    if np.any(out[1] <= 0):
        i = int(np.flatnonzero(out[1] <= 0)[0])
        raise FormulaError(f"{where} {sd.name!r}: the known standard deviations of {term!r} must be positive; row {i} has "
                           f"{float(out[1][i])!r}")
    return out[0], out[1]


def validate_data(spec: ModelSpec, data: pd.DataFrame, family: str = "gaussian", *, dpar: bool = False) -> ModelSpec:
    """Check columns exist, are numeric where required, and have no missing values. ``dpar=True`` validates a
    distributional parameter's formula (B01), whose left-hand side names a parameter rather than a column: its terms
    only."""
    from autobayes.families import get_family

    fam = get_family(family)
    if not dpar and (spec.trials is not None) != fam.has_trials:
        raise FormulaError("a trials column is present exactly for the binomial family")
    operands = [c for t in spec.group_terms for c in t.data_columns]   # the columns the grouping factors are read from (D08: and
    needed = (([] if dpar else [spec.response]) + [*spec.pop_columns] + operands   # mm()'s weights, gr()'s by column)
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
    for c in spec.missing_columns:                               # C06: the columns whose missing values are parameters
        missing_values(f"column {c!r}", data[c], c == spec.response)
    for t in spec.me_terms:                                       # C06: the noisy column and its known noise
        measurement_values(t.name, data[t.operands[0]], data[t.operands[1]])
    na = data[[c for c in needed if c not in spec.missing_columns]].isna().any()
    if na.any():
        raise FormulaError(
            f"missing values in columns {list(na[na].index)}; M0 does not handle missing "
            "data. Drop or impute them explicitly before fitting."
        )
    if len(data) < 2:
        raise FormulaError("need at least two observations")
    addition_data(spec, data)                                     # positive weights, censoring codes, responses inside the bounds (B02)
    if spec.autocor is not None:                                  # C04: the series and their time order, or refused
        from autobayes.autocor import series_layout

        series_layout(spec.autocor, data)
    levels_by_group = {}
    for t in spec.group_terms:                                    # a derived factor's levels: the observed combinations
        if t.is_mm:                                               # D08: the union of the member columns' labels, sorted
            levels = tuple(sorted(set().union(*(group_labels(c, data).unique() for c in t.members))))
            mm_weights(t, data)                                   # finite, non-negative, a positive sum on every row, or refused
        else:
            levels = tuple(sorted(group_labels(t.factor, data).unique()))
        if len(levels) < 2:
            raise FormulaError(f"grouping factor {t.factor!r} has a single level; a varying intercept needs at least two.")
        if t.by is not None:                                      # D08: each level of g within one level of by, as brms requires
            by_levels_of(t, data)
        levels_by_group[t.factor] = levels
    for c in spec.predictors:
        if np.isclose(np.std(data[c].to_numpy(dtype=float)), 0.0):
            raise FormulaError(f"predictor {c!r} is constant")
    for t in getattr(spec, "cs_terms", ()):                       # D04: a category-specific column is a numeric predictor too
        c = t.operands[0]
        reject_complex(f"column {c!r}", data[c])
        if not _is_numeric(data[c]):
            raise FormulaError(f"column {c!r} of {t.name!r} has dtype {data[c].dtype}; cs() takes a numeric column in this slice "
                               "(divergence D89)")
        values = data[c].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise FormulaError(f"column {c!r} of {t.name!r} has non-finite values")
        if np.isclose(np.std(values), 0.0):
            raise FormulaError(f"predictor {c!r} of {t.name!r} is constant")
    for t in spec.pop_terms:                                      # a numeric-by-numeric product column is a predictor too
        if t.kind == "interaction" and not t.categorical and np.isclose(np.std(pop_term_design(t, data)), 0.0):
            raise FormulaError(f"interaction {t.name!r} is constant ({' * '.join(t.operands)} takes a single value in the fitted rows)")
    for t in spec.transform_terms:                                # and so is a transformed column
        if np.isclose(np.std(pop_term_design(t, data)), 0.0):
            raise FormulaError(f"term {t.name!r} is constant in the fitted rows")
    monotonics = tuple(monotonic_coding(m.column, data[m.column]) for m in spec.monotonics)   # brms's Xmo coding (B03)
    gp_terms = tuple(gaussian_process_coding(g, data[g.column]) for g in spec.gp_terms)      # the GPs' inputs and scaling (C03)
    return ModelSpec(
        formula=spec.formula, response=spec.response, intercept=spec.intercept, levels_by_group=levels_by_group,
        trials=spec.trials, rhs=spec.rhs, categoricals=tuple(categoricals),
        pop_terms=tuple(pop_term(t.name, t.kind, categoricals) for t in spec.pop_terms),   # levels now known
        group_terms=spec.group_terms, offset=spec.offset, additions=spec.additions, dpars=spec.dpars, monotonics=monotonics,
        gp_terms=gp_terms,
        autocor=spec.autocor,
    )
