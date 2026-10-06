"""The formula grammar, its validation, and the term model a parsed formula is made of.

A parsed formula is a ``ModelSpec`` holding population-level terms (``PopTerm``: ``x``, ``z``, ``x:z``) and
group-level terms (``GroupTerm``: ``(1 | g)``, ``(1 + x || g)``); every flat view of the formula (``predictors``,
``terms``, ``columns``, ``groups``, ...) is derived from those two tuples. See docs/term-model.md.

Supported::

    response ~ [1 + | 0 +] predictor [+ predictor ...] [+ x:z ...] [+ offset(col) | offset(log(col))]
               [+ (1 | factor) | (1 + x [+ w ...] || factor) ...]
    successes | trials(n) ~ ...  the same right-hand side  (binomial, brms syntax)

``offset(col)`` and ``offset(log(col))`` (A06) add a known column to the linear predictor with no coefficient and
no prior, as brms's ``offset()`` does; one offset term per formula, held in ``ModelSpec.offset`` (docs/offset.md).

Group-level terms are optional and additive across distinct factors (M3c-3): a formula without them is a population-only
model, as in brms ``brmsformula`` where group-level terms are optional; ``y ~ 1`` is intercept-only. ``(1 + x || g)``
is brms's independent varying intercept and slope (A01): each numeric population-level term named varies by level
with its own SD and no correlation; brms's correlated form ``(1 + x | g)`` is refused (docs/varying-slopes.md).

with ``x:z`` (or the shorthand ``x*z`` = ``x + z + x:z``) distinct interactions between numeric terms
and declared categorical terms whose main effects are both present (milestone M3a-3).

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

import numpy as np
import pandas as pd

_IDENT = r"[A-Za-z_][A-Za-z0-9_.]*"
_FORMULA_RE = re.compile(rf"^\s*({_IDENT})\s*~\s*(.+?)\s*$", re.S)
_TRIALS_FORMULA_RE = re.compile(rf"^\s*({_IDENT})\s*\|\s*trials\(\s*({_IDENT})\s*\)\s*~\s*(.+?)\s*$", re.S)
_BAMBI_PROPORTION_RE = re.compile(r"^\s*(p|prop|proportion)\s*\(", re.S)
_GROUP_RE = re.compile(rf"^\(\s*([^|\s][^|]*?)\s*(\|\|?)\s*(.+?)\s*\)$", re.S)   # (inner | factor) or (inner || factor)
_OFFSET_RE = re.compile(r"^offset\s*\((.*)\)$", re.S)                             # offset(<inner>)
_OFFSET_LOG_RE = re.compile(rf"^log\s*\(\s*({_IDENT})\s*\)$", re.S)              # log(col) inside offset()

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


POP_TERM_KINDS = ("numeric", "categorical", "interaction")


@dataclass(frozen=True)
class PopTerm:
    """One population-level term as written in the formula: ``x`` (``"numeric"``), ``z`` (``"categorical"``,
    declared with its reference level) or ``x:z`` (``"interaction"``: one numeric by one categorical term).

    ``columns`` are the design-column labels the term expands to, Bambi/formulae's own, in design order:
    ``x``; ``z[l]`` per coefficient level ``l``; ``x:z[l]``. ``levels`` are the coefficient levels of a
    vector term (a categorical term's coefficient levels, an interaction's non-reference levels), empty for
    a scalar term. A categorical term's levels are known once the data are seen; until then it has no
    levels and its single column is its own name, a placeholder, as ``ModelSpec.columns`` always listed it.
    The coefficients of the term are brms's ``b`` entries for its columns.
    """

    name: str
    kind: str
    columns: tuple[str, ...]
    levels: tuple[str, ...] = ()

    def __post_init__(self):
        if self.kind not in POP_TERM_KINDS:
            raise ValueError(f"PopTerm kind must be one of {list(POP_TERM_KINDS)}, got {self.kind!r}")
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "levels", tuple(self.levels))

    @property
    def operands(self) -> tuple[str, ...]:
        """The data columns the term is built from: ``(x,)``, ``(z,)`` or ``(x, z)`` (numeric first)."""
        return tuple(self.name.split(":"))

    @property
    def is_vector(self) -> bool:
        """A term with one coefficient per level (categorical or interaction), as opposed to one slope."""
        return self.kind != "numeric"


@dataclass(frozen=True)
class GroupTerm:
    """One group-level term ``(coefs | factor)``, brms's group ``m``: ``factor`` is the grouping column,
    ``coefs`` the population-level coefficients that vary by its level (brms ``M_m`` of them; ``("Intercept",)``
    for ``(1 | g)``, ``("Intercept", "x")`` for ``(1 + x || g)``), ``correlated`` whether they share a
    correlation matrix (``|``, brms's default, not supported with slopes yet) or are independent (``||``);
    moot for a single coefficient, so ``(1 || g)`` parses to the same term as ``(1 | g)``. The level labels
    are not part of the term: they are known once the data are seen (``ModelSpec.levels_by_group``,
    ``GroupBlock.levels``). Its parameters are brms's ``sd_m`` (one per coefficient), ``z_m`` (the
    standardised effects by coefficient and level) and ``r_m_c`` (the effects of coefficient ``c``).
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
    def canonical(self) -> str:
        """The term as the canonical formula writes it: ``(1 | g)``, or ``(1 + x || g)`` for independent slopes."""
        inner = " + ".join("1" if c == "Intercept" else c for c in self.coefs)
        bar = "|" if self.correlated or len(self.coefs) == 1 else "||"
        return f"({inner} {bar} {self.factor})"


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


def bambi_group_name(factor: str, coef: str) -> str:
    """Bambi's name of one group-level coefficient: ``1|g`` for the varying intercept of ``g``, ``x|g`` for the
    slope of ``x``. PyMC's variables are this name (the effects per level), ``<name>_sigma`` and ``<name>_offset``."""
    return f"{'1' if coef == 'Intercept' else coef}|{factor}"


def sd_tag(factor: str, coef: str) -> str:
    """How messages, ``describe()`` and fit records label one group SD: ``g`` for the varying intercept of ``g``
    (``sd:g``, ``sd_g``, as they have always read) and ``x|g`` for the slope of ``x`` (``sd:x|g``, ``sd_x|g``)."""
    return factor if coef == "Intercept" else f"{coef}|{factor}"


def _find_categorical(categoricals, name: str) -> Categorical:
    for c in categoricals:
        if c.name == name:
            return c
    raise KeyError(name)


def pop_term(name: str, kind: str, categoricals=()) -> PopTerm:
    """The population-level term ``name`` of the given kind, with its coefficient levels and design columns
    read from the declared categorical terms (a categorical term's coefficient levels; the non-reference
    levels of an interaction's categorical side; none for a numeric term)."""
    if kind == "numeric":
        levels: tuple[str, ...] = ()
    elif kind == "categorical":
        levels = _find_categorical(categoricals, name).coefficients
    elif kind == "interaction":
        levels = _find_categorical(categoricals, name.split(":", 1)[1]).contrasts
    else:
        raise ValueError(f"PopTerm kind must be one of {list(POP_TERM_KINDS)}, got {kind!r}")
    columns = tuple(f"{name}[{l}]" for l in levels) if levels else (name,)
    return PopTerm(name, kind, columns, levels)


def pop_term_design(term: PopTerm, df: pd.DataFrame) -> np.ndarray:
    """The raw design block of one population-level term on ``df``: ``(n, len(term.columns))``, one column per
    entry of ``term.columns``. A numeric term is its column as given; a categorical term one 0/1 indicator
    per coefficient level (reference rows all zero); an interaction ``x:z`` the numeric column times each
    indicator. Categorical columns are read as string labels (a fitted frame holds them recoded)."""
    if term.kind == "numeric":
        return df[term.name].to_numpy(dtype=float)[:, None]
    *numeric, z = term.operands
    labels = df[z].astype(str).to_numpy()
    x = df[numeric[0]].to_numpy(dtype=float) if numeric else np.ones(len(labels))
    if not term.levels:
        return np.zeros((len(df), 0))
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
    ``offset`` is the formula's ``offset()`` term, or None (docs/offset.md)."""

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

    def __init__(self, formula, response, predictors=(), group=None, intercept=True,
                 levels=(), trials=None, rhs="", terms=(), categoricals=(), interactions=(),
                 *, groups=None, levels_by_group=None, pop_terms=None, group_terms=None, offset=None):
        """Keep the original positional constructor: without ``pop_terms`` / ``group_terms`` the term model is
        built from the flat ``predictors``, ``terms``, ``categoricals``, ``interactions`` and ``group(s)``.
        Terms given as mappings (``dataclasses.asdict`` output) are rebuilt."""
        categoricals = tuple(c if isinstance(c, Categorical) else Categorical(**c) for c in categoricals)
        if offset is not None and not isinstance(offset, Offset):
            offset = Offset(**offset)
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
        values = dict(formula=formula, response=response, intercept=intercept,
                      levels_by_group=_GroupLevels((g, level_map[g]) for g in factors),
                      trials=trials, rhs=rhs, categoricals=categoricals,
                      pop_terms=pop_terms, group_terms=group_terms, offset=offset)
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def __hash__(self):
        # Preserve hashability of the frozen specification; factor metadata is ordered.
        return hash((self.formula, self.response, self.intercept,
                     tuple(self.levels_by_group[g] for g in self.groups), self.trials, self.rhs,
                     self.categoricals, self.pop_terms, self.group_terms, self.offset))

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
        """``(numeric, categorical)`` per interaction term, in term order."""
        return tuple(t.operands for t in self.pop_terms if t.kind == "interaction")

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
        """``(numeric, categorical)`` of an interaction term; ``KeyError`` for any other name."""
        t = self.pop_term(term)
        if t.kind != "interaction":
            raise KeyError(term)
        return t.operands

    def is_vector_term(self, term: str) -> bool:
        """A categorical or interaction term: coefficient levels given by levels_of."""
        return any(t.name == term and t.is_vector for t in self.pop_terms)

    def levels_of(self, term: str) -> tuple[str, ...]:
        """Coefficient levels of a vector-valued term: a categorical term's coefficient levels,
        or the non-reference levels of an interaction's categorical side. ``()`` for a numeric term."""
        return next((t.levels for t in self.pop_terms if t.name == term), ())

    def term_columns(self, term: str) -> tuple[str, ...]:
        """Design-column labels of one term, Bambi/formulae's own: ``x``; ``z[l]``; ``x:z[l]``."""
        return next((t.columns for t in self.pop_terms if t.name == term), (term,))

    @property
    def columns(self) -> tuple[str, ...]:
        """Design-column labels in design order: the terms' columns, one after another (a categorical
        term expanded into ``z[level]`` dummies, reference omitted; an interaction into ``x:z[level]``)."""
        return tuple(col for t in self.pop_terms for col in t.columns)

    @property
    def column_terms(self) -> tuple[tuple[str, str, int | None], ...]:
        """``(column label, term, index within the term or None for a numeric term)`` per column."""
        out: list[tuple[str, str, int | None]] = []
        for t in self.pop_terms:
            if t.levels:
                out.extend((col, t.name, j) for j, col in enumerate(t.columns))
            else:
                out.append((t.name, t.name, None))
        return tuple(out)

    @property
    def canonical_formula(self) -> str:
        """The formula as expanded: main effects, then interactions spelled ``numeric:categorical``."""
        lhs = self.response if self.trials is None else f"{self.response} | trials({self.trials})"
        return f"{lhs} ~ {self.rhs}"

    @property
    def bambi_formula(self) -> str:
        """What is handed to Bambi: the canonical right-hand side; brms's ``k | trials(n)`` becomes
        Bambi's ``p(k, n)``, and ``(1 + x || g)`` becomes ``(1 + x | g)``, which Bambi (formulae has no
        ``||``) builds as the independent terms ``1|g`` and ``x|g``; ``Model._extract_design`` verifies that."""
        rhs = self.rhs.replace("||", "|")
        if self.trials is None:
            return f"{self.response} ~ {rhs}"
        return f"p({self.response}, {self.trials}) ~ {rhs}"


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


def _parse_group_term(inner: str, bars: str, factor: str, piece: str, response: str, trials: str | None) -> GroupTerm:
    """The ``GroupTerm`` of one ``(inner | factor)`` or ``(inner || factor)`` piece: the varying intercept
    ``(1 | g)`` (``(1 || g)`` is the same term, one coefficient having nothing to correlate with), or the
    independent varying intercept and numeric slopes ``(1 + x [+ w ...] || g)``. Everything else brms accepts
    here is refused with a message naming the construct. Whether a slope is a numeric population-level term of
    the formula is checked by ``parse_formula`` once every term is read."""
    if "/" in factor or ":" in factor:
        raise FormulaError(
            f"{piece!r}: nested/grouping interaction syntax ('g/h', 'g:h') is not "
            "supported; use distinct additive '(1 | group)' terms."
        )
    if "|" in factor:
        raise FormulaError(f"{piece!r}: correlation IDs are not supported")
    if factor.startswith("mm("):
        raise FormulaError(f"{piece!r}: multimembership grouping is not supported")
    if "(" in factor:
        raise FormulaError(f"{piece!r}: transformed grouping factors are not supported; use a column name")
    if not re.fullmatch(_IDENT, factor):
        raise FormulaError(f"{piece!r}: grouping factor must be a single column name")
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
        if s == factor:
            raise FormulaError(f"{piece!r}: the grouping factor {factor!r} cannot also be a varying slope")
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
        raise FormulaError(
            f"{piece!r}: varying slopes without the varying intercept are not supported; write "
            f"'(1 + {' + '.join(slopes)} || {factor})' so that the intercept varies by {factor!r} as well"
        )
    if tokens.count("1") > 1:
        raise FormulaError(f"{piece!r}: '1' appears more than once in the group term")
    if not slopes:
        return GroupTerm(factor, ("Intercept",), correlated=True)
    if bars == "|":
        raise FormulaError(
            f"{piece!r}: brms's '|' gives the varying intercept and slope(s) a correlation matrix (an LKJ prior on "
            f"their correlation), which is not supported yet; write '({inner} || {factor})' for independent "
            "varying slopes, each with its own SD and no correlation"
        )
    return GroupTerm(factor, ("Intercept", *slopes), correlated=False)


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


def parse_formula(formula: str, data: pd.DataFrame | None = None, family: str = "gaussian",
                  categorical: Mapping[str, object] | None = None) -> ModelSpec:
    """Parse and validate a formula in the supported subset, optionally against a data frame.

    ``categorical`` maps a predictor column to its reference level (treatment coding); every
    other predictor term must be numeric.
    """
    from autobayes.families import get_family

    fam = get_family(family)
    trials = None
    if _BAMBI_PROPORTION_RE.match(formula):
        raise FormulaError(
            f"{formula!r}: Bambi's 'p(successes, trials)' left-hand side is not accepted; write the brms "
            "form 'successes | trials(n) ~ ...' (ab.binomial)."
        )
    mt = _TRIALS_FORMULA_RE.match(formula)
    if mt:
        if not fam.has_trials:
            raise FormulaError(
                f"{formula!r}: 'successes | trials(n)' is the binomial family's left-hand side; use "
                "ab.binomial, or drop '| trials(...)' for a Gaussian or Bernoulli response."
            )
        response, trials, rhs = mt.group(1), mt.group(2), mt.group(3)
        if trials == response:
            raise FormulaError(f"{trials!r}: the trials column cannot also be the response")
    else:
        if "~" in formula and "|" in formula.split("~", 1)[0]:
            raise FormulaError(
                f"{formula!r}: a '|' on the left-hand side must be exactly 'successes | trials(n)'."
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

    predictors: list[str] = []                             # main effects as written, numeric and categorical
    group_terms: list[GroupTerm] = []
    group_pieces: list[str] = []                           # each group term as written, for the messages below
    groups: list[str] = []                                 # the factors of group_terms, for the checks below
    raw_interactions: list[tuple[str, str, str]] = []      # (left, right, piece as written)
    offsets: list[tuple[Offset, str]] = []                 # (offset term, piece as written); at most one
    saw_intercept_token = False
    remove_intercept = False

    def _main(name: str, piece: str) -> None:
        if not re.fullmatch(_IDENT, name):
            raise FormulaError(f"{piece!r}: {name!r} is not a column name")
        if name == response:
            raise FormulaError(f"{name!r}: the response cannot also be a predictor")
        if name in predictors:
            raise FormulaError(
                f"{name!r}: duplicated predictor" + (f" after expanding {piece!r} (a*b means a + b + a:b; write "
                                                     f"'a + b + a:b' or 'a*b', not both)" if piece != name else "")
            )
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
            group_terms.append(_parse_group_term(inner, bars, factor, piece, response, trials))   # '(1 | g)' or '(1 + x || g)'
            group_pieces.append(piece)
            groups.append(factor)
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
        if "(" in piece or ")" in piece:
            raise FormulaError(
                f"{piece!r}: function calls and transformations inside the formula are "
                "not supported in M0; transform the column in the data frame instead."
            )
        if "*" in piece or ":" in piece:
            sep = "*" if "*" in piece else ":"
            if "*" in piece and ":" in piece:
                raise FormulaError(f"{piece!r}: mixing '*' and ':' in one term is not supported; write 'x*z' or 'x + z + x:z'")
            factors = [f.strip() for f in piece.split(sep)]
            if len(factors) != 2 or any(f == "" for f in factors):
                raise FormulaError(
                    f"{piece!r}: only a two-way interaction between one numeric and one categorical term is "
                    "supported in this slice (docs/m3a2-decision.md); higher-order interactions are not."
                )
            left, right = factors
            if left == right:
                raise FormulaError(f"{piece!r}: a term cannot interact with itself")
            if sep == "*":                                          # a*b is shorthand for a + b + a:b
                _main(left, piece)
                _main(right, piece)
            raw_interactions.append((left, right, piece))
            continue
        _main(piece, piece)

    if saw_intercept_token and remove_intercept:
        raise FormulaError(f"{formula!r}: conflicting intercept inclusion/removal; use '0 + terms' or 'terms - 1'")

    for term, piece in zip(group_terms, group_pieces):       # a varying slope is a population-level term of the formula
        for s in term.slopes:
            if s not in predictors:
                raise FormulaError(
                    f"{piece!r}: the varying slope {s!r} must also be a population-level term of the formula, "
                    f"as in '{response} ~ {s} + (1 + {s} || {term.factor})' (brms would add the group-level slope "
                    "without the population-level one; this subset does not)"
                )

    for left, right, piece in raw_interactions:              # both main effects are required in this slice
        for side in (left, right):
            if side not in predictors and re.fullmatch(_IDENT, side) and side not in groups:
                raise FormulaError(
                    f"{piece!r}: the main effect {side!r} is missing. This slice requires both main effects of an "
                    f"interaction ('{left} + {right} + {left}:{right}' or '{left}*{right}'): omitting one is a different "
                    "model (the level intercepts or the reference slope are forced to zero and formulae changes the "
                    "interaction's coding) and is not part of the verified subset (docs/m3a2-decision.md)."
                )
    if remove_intercept and not predictors:
        raise FormulaError(f"{formula!r}: no intercept requires at least one population-level predictor")
    if len(set(groups)) != len(groups):
        repeated = next(g for g in groups if groups.count(g) > 1)
        raise FormulaError(
            f"{formula!r}: duplicate grouping factor {repeated!r}; a factor appears in one group term, "
            f"'(1 | {repeated})' or '(1 + x || {repeated})'"
        )
    if len(predictors) == 0 and not saw_intercept_token:
        raise FormulaError(
            f"{formula!r}: no population-level predictor; write 'y ~ 1 + (1 | group)' "
            "for an intercept-only model."
        )
    for group in groups:
        if group in predictors or group == response:
            raise FormulaError(f"{group!r}: the grouping factor cannot also be a predictor or the response")
    if trials is not None and (trials in predictors or trials in groups):
        raise FormulaError(f"{trials!r}: the trials column cannot also be a predictor or the grouping factor")
    offset = offsets[0][0] if offsets else None
    if offset is not None and offset.column in groups:
        raise FormulaError(
            f"{offsets[0][1]!r}: the grouping factor {offset.column!r} cannot be the offset; an offset is a known "
            "numeric column added to the linear predictor"
        )

    categoricals = _normalise_categorical(categorical, predictors, response, groups, trials, formula)
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
    interactions: list[tuple[str, str]] = []
    for left, right, piece in raw_interactions:
        for side in (left, right):
            if not re.fullmatch(_IDENT, side):
                raise FormulaError(f"{piece!r}: {side!r} is not a column name")
            if side in groups:
                raise FormulaError(f"{piece!r}: the grouping factor {side!r} cannot enter a fixed interaction")
            if side == response or (trials is not None and side == trials):
                raise FormulaError(f"{piece!r}: the response or trials column cannot enter an interaction")
        cats = [s for s in (left, right) if s in cat_names]
        nums = [s for s in (left, right) if s not in cat_names]
        if len(cats) != 1:
            kind = "two categorical terms" if len(cats) == 2 else "two numeric terms"
            raise FormulaError(
                f"{piece!r}: an interaction between {kind} is not supported in this slice; only one numeric "
                "by one declared categorical term is (docs/m3a2-decision.md). A categorical side must be "
                "declared in categorical=."
            )
        x, z = nums[0], cats[0]
        if (x, z) in interactions:
            raise FormulaError(f"{piece!r}: the interaction '{x}:{z}' appears more than once after expansion")
        interactions.append((x, z))
    parts = ([*predictors, *(f"{x}:{z}" for x, z in interactions)] + ([offset.term] if offset is not None else [])
             + [t.canonical for t in group_terms])
    canonical_rhs = " + ".join(parts) if parts else "1"           # intercept-only population model: 'y ~ 1'
    if remove_intercept:
        canonical_rhs = "0 + " + canonical_rhs
    pop_terms = (*(pop_term(p, "categorical" if p in cat_names else "numeric", categoricals) for p in predictors),
                 *(pop_term(f"{x}:{z}", "interaction", categoricals) for x, z in interactions))
    spec = ModelSpec(
        formula=formula.strip(), response=response, intercept=not remove_intercept, trials=trials,
        rhs=canonical_rhs, categoricals=categoricals, pop_terms=pop_terms, group_terms=tuple(group_terms),
        offset=offset,
    )
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


def validate_data(spec: ModelSpec, data: pd.DataFrame, family: str = "gaussian") -> ModelSpec:
    """Check columns exist, are numeric where required, and have no missing values."""
    from autobayes.families import get_family

    fam = get_family(family)
    if (spec.trials is not None) != fam.has_trials:
        raise FormulaError("a trials column is present exactly for the binomial family")
    needed = ([spec.response, *spec.predictors, *spec.categorical_names] + list(spec.groups)
              + ([spec.trials] if spec.trials else []) + ([spec.offset.column] if spec.offset is not None else []))
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
    levels_by_group = {}
    for group in spec.groups:
        levels = tuple(sorted(data[group].astype(str).unique()))
        if len(levels) < 2:
            raise FormulaError(f"grouping factor {group!r} has a single level; a varying intercept needs at least two.")
        levels_by_group[group] = levels
    for c in spec.predictors:
        if np.isclose(np.std(data[c].to_numpy(dtype=float)), 0.0):
            raise FormulaError(f"predictor {c!r} is constant")
    return ModelSpec(
        formula=spec.formula, response=spec.response, intercept=spec.intercept, levels_by_group=levels_by_group,
        trials=spec.trials, rhs=spec.rhs, categoricals=tuple(categoricals),
        pop_terms=tuple(pop_term(t.name, t.kind, categoricals) for t in spec.pop_terms),   # levels now known
        group_terms=spec.group_terms, offset=spec.offset,
    )
