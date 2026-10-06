"""The formula grammar and its validation.

Supported::

    response ~ [1 + | 0 +] predictor [+ predictor ...] [+ x:z ...] [+ (1 | factor) ...]               (Gaussian, Bernoulli)
    successes | trials(n) ~ [1 + | 0 +] predictor [+ predictor ...] [+ x:z ...] [+ (1 | factor) ...]  (binomial, brms syntax)

Varying-intercept terms are optional and additive across distinct factors (M3c-3): a formula without them is a population-only
model, as in brms ``brmsformula`` where group-level terms are optional; ``y ~ 1`` is intercept-only.

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
_GROUP_RE = re.compile(rf"^\(\s*(.+?)\s*\|\s*(.+?)\s*\)$", re.S)

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
    formula: str
    response: str
    predictors: tuple[str, ...]
    groups: tuple[str, ...]
    intercept: bool
    levels_by_group: Mapping[str, tuple[str, ...]]
    trials: str | None
    rhs: str
    terms: tuple[str, ...]
    categoricals: tuple[Categorical, ...]
    interactions: tuple[tuple[str, str], ...]

    def __init__(self, formula, response, predictors, group=None, intercept=True,
                 levels=(), trials=None, rhs="", terms=(), categoricals=(), interactions=(),
                 *, groups=None, levels_by_group=None):
        """Keep the original positional constructor; canonical fields use ordered factors."""
        if groups is not None and group is not None:
            raise ValueError("supply group or groups, not both")
        groups = tuple(groups) if groups is not None else (() if group is None else (group,))
        if len(set(groups)) != len(groups):
            raise ValueError("duplicate grouping factors")
        if levels_by_group is not None and levels:
            raise ValueError("supply levels or levels_by_group, not both")
        if levels and len(groups) != 1:
            raise ValueError("singular levels require one grouping factor")
        level_map = ({g: tuple(levels) for g in groups} if levels_by_group is None
                     else {g: tuple(v) for g, v in levels_by_group.items()})
        if set(level_map) != set(groups):
            raise ValueError("levels_by_group must cover exactly the grouping factors")
        values = dict(formula=formula, response=response, predictors=tuple(predictors),
                      groups=groups, intercept=intercept,
                      levels_by_group=_GroupLevels((g, level_map[g]) for g in groups),
                      trials=trials, rhs=rhs, terms=tuple(terms) or tuple(predictors),
                      categoricals=tuple(categoricals), interactions=tuple(interactions))
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def __hash__(self):
        # Preserve hashability of the old frozen specification; factor metadata is ordered.
        return hash((self.formula, self.response, self.predictors, self.groups, self.intercept,
                     tuple(self.levels_by_group[g] for g in self.groups), self.trials, self.rhs,
                     self.terms, self.categoricals, self.interactions))

    @property
    def has_group(self) -> bool:
        return bool(self.groups)

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
        if self.group is None:
            raise ValueError("this model has no group-level term")
        return f"1|{self.group}"

    @property
    def categorical_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.categoricals)

    def categorical(self, name: str) -> Categorical:
        for c in self.categoricals:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def interaction_names(self) -> tuple[str, ...]:
        return tuple(f"{x}:{z}" for x, z in self.interactions)

    def interaction_of(self, term: str) -> tuple[str, str]:
        for x, z in self.interactions:
            if f"{x}:{z}" == term:
                return x, z
        raise KeyError(term)

    def is_vector_term(self, term: str) -> bool:
        """A categorical or interaction term: coefficient levels given by levels_of."""
        return term in self.categorical_names or term in self.interaction_names

    def levels_of(self, term: str) -> tuple[str, ...]:
        """Coefficient levels of a vector-valued term: a categorical term's non-reference levels,
        or those of an interaction's categorical side. ``()`` for a numeric term."""
        if term in self.categorical_names:
            return self.categorical(term).coefficients
        if term in self.interaction_names:
            return self.categorical(self.interaction_of(term)[1]).contrasts
        return ()

    def term_columns(self, term: str) -> tuple[str, ...]:
        """Design-column labels of one term, Bambi/formulae's own: ``x``; ``z[l]``; ``x:z[l]``."""
        levels = self.levels_of(term)
        return tuple(f"{term}[{l}]" for l in levels) if levels else (term,)

    @property
    def columns(self) -> tuple[str, ...]:
        """Design-column labels in design order: canonical term order, a categorical term
        expanded into ``z[level]`` dummies (reference omitted), an interaction into ``x:z[level]``."""
        return tuple(col for term in self.terms for col in self.term_columns(term))

    @property
    def column_terms(self) -> tuple[tuple[str, str, int | None], ...]:
        """``(column label, term, index within the term or None for a numeric term)`` per column."""
        out: list[tuple[str, str, int | None]] = []
        for term in self.terms:
            cols = self.term_columns(term)
            if self.levels_of(term):
                out.extend((col, term, j) for j, col in enumerate(cols))
            else:
                out.append((term, term, None))
        return tuple(out)

    @property
    def canonical_formula(self) -> str:
        """The formula as expanded: main effects, then interactions spelled ``numeric:categorical``."""
        lhs = self.response if self.trials is None else f"{self.response} | trials({self.trials})"
        return f"{lhs} ~ {self.rhs}"

    @property
    def bambi_formula(self) -> str:
        """What is handed to Bambi: the canonical right-hand side; brms's ``k | trials(n)`` becomes
        Bambi's ``p(k, n)``."""
        if self.trials is None:
            return f"{self.response} ~ {self.rhs}"
        return f"p({self.response}, {self.trials}) ~ {self.rhs}"


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


def _reject_group_term(inner: str, factor: str, piece: str) -> None:
    if "||" in piece:
        raise FormulaError(
            f"{piece!r}: the '||' (uncorrelated varying effects) syntax is not supported "
            "in this subset; use varying intercepts '(1 | group)'."
        )
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
    if inner != "1":
        raise FormulaError(
            f"{piece!r}: varying slopes are not supported in M0; only a varying "
            f"intercept '(1 | {factor})' is."
        )


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

    predictors: list[str] = []
    groups: list[str] = []
    raw_interactions: list[tuple[str, str, str]] = []      # (left, right, piece as written)
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
            inner, factor = gm.group(1).strip(), gm.group(2).strip()
            _reject_group_term(inner, factor, piece)
            groups.append(factor)
            continue
        if piece.startswith("("):
            raise FormulaError(f"{piece!r}: parenthesised term is not a '(1 | group)' term")
        if "|" in piece:
            raise FormulaError(f"{piece!r}: a '|' term must be written '(1 | group)'")
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
        raise FormulaError(f"{formula!r}: duplicate grouping factor; use each '(1 | factor)' once")
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

    categoricals = _normalise_categorical(categorical, predictors, response, groups, trials, formula)
    if remove_intercept and categoricals:
        categoricals = (replace(categoricals[0], full_rank=True), *categoricals[1:])
    cat_names = {c.name for c in categoricals}
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
    parts = [*predictors, *(f"{x}:{z}" for x, z in interactions)] + [f"(1 | {group})" for group in groups]
    canonical_rhs = " + ".join(parts) if parts else "1"           # intercept-only population model: 'y ~ 1'
    if remove_intercept:
        canonical_rhs = "0 + " + canonical_rhs
    spec = ModelSpec(
        formula=formula.strip(), response=response,
        predictors=tuple(p for p in predictors if p not in cat_names),
        groups=tuple(groups), intercept=not remove_intercept, trials=trials, rhs=canonical_rhs,
        terms=(*predictors, *(f"{x}:{z}" for x, z in interactions)), categoricals=categoricals,
        interactions=tuple(interactions),
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
              + ([spec.trials] if spec.trials else []))
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
        formula=spec.formula, response=spec.response, predictors=spec.predictors,
        groups=spec.groups, intercept=spec.intercept, levels_by_group=levels_by_group, trials=spec.trials, rhs=spec.rhs,
        terms=spec.terms, categoricals=tuple(categoricals), interactions=spec.interactions,
    )
