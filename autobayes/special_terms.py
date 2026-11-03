"""brms's special terms in interactions and with ids (docs/special-term-ext.md): ``mo(ord):x``, ``mo(ord) * mo(o2)``,
``mo(ord):z``, ``mo(ord, id = "a")``, ``me(x, sdx):w``, and ``mo()`` in a distributional formula.

brms reads every term of a formula that holds a special function (``mo()``, ``me()``, ``mi()``) as one special term: one
coefficient (its entry of ``bsp``) times the product of the term's special parts, ``mo(simo, Xmo)`` for each ``mo()`` and the
latent values ``Xme`` for each ``me()``, and of its other variables, the covariate brms calls ``Csp`` (a numeric column, or the
product of several; a factor's column of one level). A main effect ``mo(ord)`` or ``me(x, sdx)`` without an id keeps its own
machinery (B03, C06); every other special term is a ``SpecialProduct``, a population-level term of kind ``"special_product"``
(``PopTerm.special``): no design column, its coefficient keyed by its name in ``Priors(b={...})`` (brms's class ``b``,
coefficient ``rename(name)``: ``moord:x``, ``moordidEQa``).

Each ``mo()`` part of a special product has a simplex of its own (brms's ``simo_k``, one per ``Xmo_k``) unless it carries an id:
the ``mo()`` parts of one formula with the same id share one simplex (brms's ``id`` argument of ``mo()``; ``?mo``: "All monotonic
terms with the same id within one formula will be modeled as having the same simplex"); a distributional formula's ids are its
own. A simplex is keyed, in ``Priors(simo={...})``, ``value_point(simo=...)`` and ``Model.names``, by its id, by the term's name for
a term with one ``mo()`` part (``"mo(ord):x"``), and by the term's name and then the part's column for a term with several
(``{"mo(ord):mo(o2)": {"ord": ..., "o2": ...}}``; flat key ``"mo(ord):mo(o2)[ord]"``). brms names the simplex's prior row
``coef = <id>`` or ``rename(name)`` followed by the part's position in the term (``moord:x1``, ``moord:moo22``).

A factor in a special product (declared ``categorical=``) gives one special term per column of R's coding of the term: an
indicator per level when the term without the factor is not in the formula (``x + mo(ord):z``: ``mo(ord):zp``, ``mo(ord):zq``,
``mo(ord):zr``), the non-reference levels' indicators when it is (``mo(ord) * z``). brms 2.23.0 gets both wrong (divergence
D197): it multiplies every level's term by the first level's indicator in the first case and stops with an R error in the
second; this library builds the interaction as written.

Nothing here samples. ``add_effects`` builds the terms inside the PyMC model Bambi is building, ``check_effects`` reads each
term's contribution back off the graph, and ``contribution`` gives the terms' part of the linear predictor per draw.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from autobayes.formula import (FormulaError, Monotonic, PopTerm, _IDENT, _ME_RE, _MO_RE, _label, _split_top_level,
                               monotonic_coding)

_ID_RE = re.compile(r"""^id\s*=\s*(["'])([A-Za-z][A-Za-z0-9_.]*)\1$""")
_GR_RE = re.compile(rf"^gr\s*=\s*({_IDENT})$")
DOC = "docs/special-term-ext.md"


# -- the calls ----------------------------------------------------------------------------------------------------
def mo_call(text: str, piece: str) -> tuple[str, str | None]:
    """The column and id of brms's ``mo(col)`` or ``mo(col, id = "a")`` written as ``text``; anything else is refused naming
    it (an expression inside ``mo()``, another argument)."""
    m = _MO_RE.match(text.strip())
    args = [a.strip() for a in _split_top_level(m.group(1), ",")]
    if not re.fullmatch(_IDENT, args[0]):
        raise FormulaError(f"{piece!r}: mo() takes a column name here, not an expression ({args[0]!r}; brms evaluates one): compute "
                           "the column in the data frame and write mo(<column>) (docs/monotonic.md, divergence D42)")
    ids = [_ID_RE.match(a) for a in args[1:]]
    if len(args) > 2 or not all(ids):
        raise FormulaError(f"{piece!r}: mo() takes a column and optionally brms's id, mo(<column>, id = \"<name>\"), the name a "
                           f"letter followed by letters, digits, '.' or '_' ({DOC}, divergence D198)")
    return args[0], (ids[0].group(2) if ids else None)


def mo_name(column: str, id: str | None) -> str:
    """The canonical spelling of a ``mo()`` call: ``mo(ord)``, ``mo(ord, id = "a")``."""
    return f"mo({column})" if id is None else f'mo({column}, id = "{id}")'


def me_call(text: str, piece: str) -> tuple[str, str, str | None]:
    """The noisy column, the noise column and the grouping column of brms's ``me(x, sdx)`` or ``me(x, sdx, gr = g)`` written as
    ``text``; anything else is refused naming it."""
    m = _ME_RE.match(text.strip())
    args = [a.strip() for a in _split_top_level(m.group(1), ",")] if m.group(1).strip() else []
    gr = _GR_RE.match(args[2]) if len(args) == 3 else None
    if len(args) not in (2, 3) or (len(args) == 3 and gr is None):
        raise FormulaError(f"{piece!r}: me() takes the noisy column, the column of its known standard deviations and optionally "
                           "brms's gr, me(x, sdx) or me(x, sdx, gr = g) (docs/missing-and-measurement-error.md, divergence D73)")
    bad = [a for a in args[:2] if not re.fullmatch(_IDENT, a)]
    if bad:
        raise FormulaError(f"{piece!r}: me() takes column names here, not expressions ({bad[0]!r}; brms evaluates one): compute the "
                           "column in the data frame and write me(<column>, <noise column>) (docs/missing-and-measurement-error.md, "
                           "divergence D73)")
    return args[0], args[1], (gr.group(1) if gr else None)


def me_name(x: str, sdx: str, gr: str | None) -> str:
    """The canonical spelling of an ``me()`` call: ``me(x, sdx)``, ``me(x, sdx, gr = g)``."""
    return f"me({x}, {sdx})" if gr is None else f"me({x}, {sdx}, gr = {gr})"


# -- the term -----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpecialProduct:
    """One special term of brms's ``bsp`` that is no plain ``mo(col)`` or ``me(x, sdx)`` main effect: ``sides`` are its variables
    as R orders them, the canonical ``mo()`` / ``me()`` calls and the other columns; ``factor`` the declared factor among the
    columns (one at most), ``level`` the factor's level whose indicator this term multiplies once the data are seen (None before,
    and without a factor), ``coding`` how R codes the factor in the term (``"indicator"``: every level; ``"contrast"``: the
    levels but the reference); ``codings`` the ``formula.Monotonic`` coding of each ``mo()`` part once the data are seen;
    ``dpar`` the distributional parameter whose formula holds the term (None for the response formula)."""

    sides: tuple[str, ...]
    factor: str | None = None
    level: str | None = None
    coding: str = "indicator"
    codings: tuple = ()
    dpar: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "sides", tuple(self.sides))
        object.__setattr__(self, "codings", tuple(m if isinstance(m, Monotonic) else Monotonic(**m) for m in self.codings))

    @property
    def written(self) -> str:
        """The term as the canonical formula writes it: ``mo(ord):x``, ``mo(ord):z`` (before its factor is expanded)."""
        return ":".join(self.sides)

    @property
    def name(self) -> str:
        """The term's name, its factor side replaced by the level's column once expanded (``mo(ord):zp``)."""
        return ":".join(f"{s}{self.level}" if s == self.factor and self.level is not None else s for s in self.sides)

    @property
    def parts(self) -> tuple[str, ...]:
        """The special calls, in order."""
        return tuple(s for s in self.sides if _MO_RE.match(s) or _ME_RE.match(s))

    @property
    def mo_parts(self) -> tuple[tuple[str, str | None], ...]:
        """``(column, id)`` of each ``mo()`` part, in order."""
        return tuple(mo_call(s, s) for s in self.parts if _MO_RE.match(s))

    @property
    def me_parts(self) -> tuple[str, ...]:
        """The ``me()`` calls among the parts, as written."""
        return tuple(s for s in self.parts if _ME_RE.match(s))

    @property
    def numeric(self) -> tuple[str, ...]:
        """The numeric columns the term multiplies (brms's ``Csp``)."""
        return tuple(s for s in self.sides if s not in self.parts and s != self.factor)

    @property
    def operands(self) -> tuple[str, ...]:
        """Every data column the term reads, in order without repeats."""
        cols = []
        for s in self.sides:
            if _MO_RE.match(s):
                cols.append(mo_call(s, s)[0])
            elif _ME_RE.match(s):
                x, sdx, gr = me_call(s, s)
                cols += [x, sdx] + ([gr] if gr else [])
            else:
                cols.append(s)
        return tuple(dict.fromkeys(cols))

    @property
    def has_covariate(self) -> bool:
        """Whether the term multiplies a covariate (brms gives it a ``Csp_k``)."""
        return bool(self.numeric) or self.factor is not None

    def simplex_key(self, j: int) -> str:
        """The flat key of the simplex of ``mo()`` part ``j``: its id, the term's name for a term with one ``mo()`` part, or
        ``<name>[<column>]``."""
        column, id = self.mo_parts[j]
        if id is not None:
            return id
        return self.name if len(self.mo_parts) == 1 else f"{self.name}[{column}]"

    def brms_simplex_coef(self, j: int) -> str:
        """brms's ``coef`` of that simplex's prior row: the id, or ``rename(name)`` and the part's position (``moord:x1``)."""
        from autobayes.default_priors import brms_rename

        column, id = self.mo_parts[j]
        return id if id is not None else f"{brms_rename(self.name)}{j + 1}"

    def coding_of(self, column: str) -> Monotonic:
        """The fitted coding of the ``mo()`` part of ``column``."""
        return next(m for m in self.codings if m.column == column)

    def covariate(self, frame: pd.DataFrame, levels=None, *, where: str = "column") -> np.ndarray | None:
        """brms's ``Csp`` of the term on ``frame``: the product of its numeric columns and, with a factor, the indicator of its
        level (``levels``, the factor's fitted levels: a label outside them is refused, as brms refuses new factor levels);
        None for a term without a covariate."""
        if not self.has_covariate:
            return None
        out = np.ones(len(frame))
        for c in self.numeric:
            out = out * frame[c].to_numpy(dtype=float)
        if self.factor is not None:
            labels = np.array([_label(v) for v in frame[self.factor].astype(object).tolist()], dtype=object)
            unknown = sorted(set(labels.tolist()) - set(levels or ()))
            if levels is not None and unknown:
                raise FormulaError(f"{where} {self.factor!r} of {self.name!r} has levels {unknown} the model was not fitted with "
                                   f"(its levels {list(levels)}; brms refuses new factor levels too)")
            out = out * (labels == self.level)
        return out


def special_term(sp: SpecialProduct) -> PopTerm:
    """The population-level term of ``sp``."""
    return PopTerm(sp.name, "special_product", (), special=sp)


def parse_product(sides, piece: str, response: str, trials: str | None) -> SpecialProduct:
    """The special product of an interaction ``a:b[:c ...]`` with at least one ``mo()`` or ``me()`` side, its sides in R's order;
    a side that is neither a ``mo()`` or ``me()`` call nor a column name, and a variable written twice, are refused."""
    out = []
    for side in (s.strip() for s in sides):
        if _MO_RE.match(side):
            out.append(mo_name(*mo_call(side, piece)))
        elif _ME_RE.match(side):
            out.append(me_name(*me_call(side, piece)))
        elif re.fullmatch(_IDENT, side):
            if side == response or (trials is not None and side == trials):
                raise FormulaError(f"{piece!r}: the response or trials column cannot enter an interaction")
            out.append(side)
        else:
            raise FormulaError(f"{piece!r}: {side!r} in an interaction with a special term (mo(), me()) is not supported: such an "
                               f"interaction takes mo() and me() calls and column names here ({DOC}, divergence D198)")
    columns = [mo_call(s, s)[0] if _MO_RE.match(s) else s for s in out if not _ME_RE.match(s)]
    if len(set(out)) != len(out) or len(set(columns)) != len(columns):
        raise FormulaError(f"{piece!r}: a variable written twice in one interaction is not supported ({DOC}, divergence D198)")
    if any(_MO_RE.match(s) for s in out) and any(_ME_RE.match(s) for s in out):
        raise FormulaError(f"{piece!r}: mo() and me() in one term are not supported: they share brms's vector of special coefficients "
                           "(bsp), whose order is verified for one kind of term only (docs/missing-and-measurement-error.md, "
                           "divergence D73)")
    return SpecialProduct(tuple(out))


def resolve(products: dict, categorical, names: set, me_terms: dict, formula: str, dpar: str | None) -> dict:
    """The special products of a formula with their factors and codings decided: ``products`` by written name, ``categorical``
    the declared factors (a mapping or None), ``names`` every term name of the formula, ``me_terms`` the formula's ``me()``
    terms by name. One declared factor per term at most; an ``me()`` part must be a main effect of the formula (whose latent
    values it multiplies); a factor is coded by indicators unless the term without it is a term of the formula (R's rule)."""
    declared = set(categorical) if isinstance(categorical, Mapping) else set()
    out = {}
    for name, sp in products.items():
        factors = [s for s in sp.sides if s in declared]
        if len(factors) > 1:
            raise FormulaError(f"{formula!r}: {name!r} multiplies the factors {factors}; a special term with more than one factor is "
                               f"not supported ({DOC}, divergence D198)")
        missing = [m for m in sp.me_parts if m not in me_terms]
        if missing:
            raise FormulaError(f"{formula!r}: {name!r} multiplies the latent values of {missing[0]!r}, which is no main effect of the "
                               f"formula; write {missing[0]} as a term of the formula too ({DOC}, divergence D198)")
        factor = factors[0] if factors else None
        rest = ":".join(s for s in sp.sides if s != factor)
        coding = "contrast" if factor is not None and rest in names else "indicator"
        out[name] = replace(sp, factor=factor, coding=coding, dpar=dpar)
    seen: dict[str, str] = {}
    for sp in out.values():                                 # one column per id within a formula
        for column, id in sp.mo_parts:
            if id is not None and seen.setdefault(id, column) != column:
                raise FormulaError(f"{formula!r}: the id {id!r} is given to mo({seen[id]}) and mo({column}); an id shares one simplex "
                                   f"between the mo() terms of one column here ({DOC}, divergence D198)")
    return out


def seen_terms(terms, categoricals, data: pd.DataFrame) -> tuple[PopTerm, ...]:
    """The population-level terms once the data are seen: each special product with its ``mo()`` codings, and a special product
    with a factor expanded into one term per column of the factor's coding (its fitted levels, the reference dropped under
    ``"contrast"``)."""
    out = []
    for t in terms:
        if t.kind != "special_product":
            out.append(t)
            continue
        sp = t.special
        codings = tuple(monotonic_coding(c, data[c]) for c, _ in sp.mo_parts)
        if sp.factor is None or sp.level is not None:
            out.append(special_term(replace(sp, codings=codings)))
            continue
        cat = next(c for c in categoricals if c.name == sp.factor)
        levels = [l for l in cat.levels if sp.coding == "indicator" or l != cat.reference]
        out.extend(special_term(replace(sp, level=l, codings=codings)) for l in levels)
    return tuple(out)


def check_data(spec, terms, data: pd.DataFrame) -> None:
    """The fitted columns of the special products ``terms`` (seen, ``seen_terms``): each numeric covariate numeric, finite and not
    constant (a constant one would make the term its main effect's twin), and a term's covariate not zero on every row."""
    from autobayes.formula import _is_numeric, reject_complex

    for t in terms:
        if t.kind != "special_product":
            continue
        sp = t.special
        for c in sp.numeric:
            reject_complex(f"column {c!r}", data[c])
            values = data[c].to_numpy(dtype=float, na_value=np.nan) if _is_numeric(data[c]) else None
            if values is None or not np.isfinite(values).all():
                raise FormulaError(f"column {c!r} of {t.name!r} must be numeric and finite (a covariate of a special term)")
            if np.isclose(np.std(values), 0.0):
                raise FormulaError(f"column {c!r} of {t.name!r} is constant in the fitted rows")


def measurement_levels(name: str, data: pd.DataFrame, *, where: str = "column"):
    """``me(x, sdx, gr = g)``'s levels and values (brms's ``Xn_k``, ``noise_k``, one per level of g, and ``Jme_k``): the sorted
    labels of g, each row's level (0-based) and each level's noisy value and noise, which must be the same on every row of the
    level (brms: "Measured values and measurement error should be unique for each group"); refused naming the level otherwise."""
    from autobayes.formula import group_labels, measurement_error_columns, measurement_values

    x, sdx, g = measurement_error_columns(name)
    noisy, noise = measurement_values(name, data[x], data[sdx], where=where)
    labels = group_labels(g, data).to_numpy()
    levels = tuple(sorted(set(labels.tolist())))
    index = np.array([levels.index(v) for v in labels], dtype=np.int64)
    Xn, sd = np.empty(len(levels)), np.empty(len(levels))
    for k, level in enumerate(levels):
        rows = index == k
        if np.unique(noisy[rows]).size != 1 or np.unique(noise[rows]).size != 1:
            raise FormulaError(f"{name!r}: the noisy values and their noise must be the same on every row of a level of {g!r}; they "
                               f"vary within the level {level!r} (brms: 'Measured values and measurement error should be unique for "
                               f"each group. Occured for level {level!r} of group {g!r}.')")
        Xn[k], sd[k] = noisy[rows][0], noise[rows][0]
    return levels, index, Xn, sd


def check_context(spec, fam, formula: str, *, dpar: bool = False) -> None:
    """Where a special product is taken: a family whose linear predictor takes ``mo()`` (``FamilySpec.monotonic``) for a term
    with a ``mo()`` part and ``me()`` (``FamilySpec.measurement_error``) for one with an ``me()`` part, a univariate model, and
    nothing next to it that no frozen case verifies it with (an addition term other than ``trials()``, a Gaussian process, an
    autocorrelation, SAR, CAR, fcor() or residual correlation term, a group-level term the model builds itself, mm(), gr(by = ),
    gr(cov = ), gr(dist = )); each refused naming it (divergence D198)."""
    terms = spec.sp_terms                                   # (an me() part's family is its main effect's, checked before)
    first = terms[0].name
    if any(t.special.mo_parts for t in terms) and not fam.monotonic:
        raise FormulaError(f"{formula!r}: the monotonic term {first!r} is not available for the {fam.name} family (docs/monotonic.md, "
                           "divergence D42)")
    others = ([f"the addition term {a.term!r}" for a in spec.additions] + [f"the Gaussian process {g.term!r}" for g in spec.gp_terms]
              + [f"the term {x.term!r}" for x in (spec.autocor, spec.sar, spec.car, spec.fcor, spec.timecor) if x is not None]
              + [f"the group term '{t.canonical}'" for t in spec.group_terms if t.custom])
    if others:
        raise FormulaError(f"{formula!r}: the special term {first!r} next to {others[0]} is not supported: no frozen brms case verifies "
                           f"the pair ({DOC}, divergence D198)")


def check_dpar_link(name: str, link: str, d) -> None:
    """A distributional formula takes special products under the log link only (``sigma``, ``shape``, ``phi``: the frozen case
    is the Gaussian ``sigma``); refused naming the parameter otherwise."""
    if d.sp_terms and link != "log":
        raise FormulaError(f"'{d.formula}': the monotonic term {d.sp_terms[0].name!r} in the formula of {name} ({link} link) is not "
                           f"supported: mo() is taken in the formulas of parameters with the log link ({DOC}, divergence D198)")


def refuse_in(spec, where: str, row: str = "D198") -> None:
    """Refuse the special products of ``spec`` in a model they are not verified in (``where``: multivariate, non-linear, ...)."""
    if getattr(spec, "sp_terms", ()):
        raise FormulaError(f"{spec.formula!r}: the special term {spec.sp_terms[0].name!r} in {where} is not supported ({DOC}, "
                           f"divergence {row})")


# -- simplexes and priors -----------------------------------------------------------------------------------------
def simplexes(spec) -> dict[str, tuple]:
    """Every simplex of the formula's special products, by flat key in first-use order: ``(coding, brms coef, term, j)`` of
    the first ``mo()`` part that uses it (an id's later parts reuse it)."""
    out = {}
    for t in spec.sp_terms:
        sp = t.special
        for j, (column, _) in enumerate(sp.mo_parts):
            out.setdefault(sp.simplex_key(j), (sp.coding_of(column) if sp.codings else Monotonic(column), sp.brms_simplex_coef(j),
                                               t.name, j))
    return out


def prior_keys(spec) -> list:
    """The keys of ``Priors(simo={...})`` the special products need: a flat key, or ``(term, column)`` for a part of a term
    with several ``mo()`` parts and no id."""
    keys = []
    for t in spec.sp_terms:
        sp = t.special
        for j, (column, id) in enumerate(sp.mo_parts):
            key = sp.simplex_key(j) if id is not None or len(sp.mo_parts) == 1 else (sp.name, column)
            if key not in keys:
                keys.append(key)
    return keys


def flat(key) -> str:
    """The flat key of a prior key: ``(term, column)`` is ``term[column]``."""
    return key if isinstance(key, str) else f"{key[0]}[{key[1]}]"


def split_simo(simo, spec, plain: list[str]):
    """``Priors.simo`` (or ``value_point``'s ``simo``) split into the plain monotonic terms' entries (by column, ``plain``) and
    the special products' (by flat key): a mapping's nested entry ``{term: {column: ...}}`` is the parts of a term with several
    ``mo()`` parts. Returns ``(plain entries or None, special entries)``; a key that is neither is refused naming it."""
    keys = prior_keys(spec)
    if not isinstance(simo, Mapping):
        if len(keys) + len(plain) != 1:
            raise ValueError(f"several simplexes {plain + [flat(k) for k in keys]} need an exact-key simo mapping")
        return (simo if plain else None), ({flat(keys[0]): simo} if keys else {})
    terms = {k[0] for k in keys if not isinstance(k, str)}
    flats = {flat(k) for k in keys}
    special, rest = {}, {}
    for k, v in simo.items():
        if k in terms:
            if not isinstance(v, Mapping):
                raise ValueError(f"simo[{k!r}] must map each mo() column of the term to its simplex's entry, got {type(v).__name__}")
            special.update({flat((k, c)): w for c, w in v.items()})
        elif k in flats:                                    # a flat key, term[column] for a part of a term with several
            special[k] = v
        else:
            rest[k] = v
    want = {flat(k) for k in keys}
    if set(special) != want or set(rest) != set(plain):
        raise ValueError(f"simo must cover exactly the simplexes {plain + [k if isinstance(k, str) else {k[0]: k[1]} for k in keys]}; "
                         f"got {list(simo)}")
    return (rest if plain else None), special


# -- names ------------------------------------------------------------------------------------------------------------
def names(spec, prefix: str = "") -> dict:
    """PyMC names of a formula's special products: ``"b"`` by term name (the coefficient, brms's ``bsp``: the term's name, or
    ``<prefix><name>`` in a distributional formula) and ``"simplex"`` / ``"coordinate"`` by flat simplex key (``simo_<key>``,
    sampled on Stan's stick-breaking coordinates ``simo_<key>_stickbreaking__``; ``<prefix>simo_<key>``)."""
    keys = simplexes(spec)
    return {"b": {t.name: f"{prefix}{t.name}" for t in spec.sp_terms},
            "simplex": {k: f"{prefix}simo_{k}" for k in keys}, "coordinate": {k: f"{prefix}simo_{k}_dim" for k in keys}}


# -- the graph --------------------------------------------------------------------------------------------------------
def add_effects(model, eta, spec, coefficients, priors, data: pd.DataFrame, prefix: str = ""):
    """``eta`` plus the special products of ``spec`` (the response formula, or a distributional formula with ``prefix``), built
    inside the PyMC model Bambi is building: each simplex once (a ``Dirichlet`` under its ``simo`` prior on Stan's stick-breaking
    coordinates, ``monotonic.simplex_rv``), each coefficient under its ``b`` prior, and per row ``b * prod(parts) * Csp``: a
    ``mo()`` part's ``mo(simo, Xmo)``, an ``me()`` part's latent values (``model.me_names``, at the row's level for
    ``me(x, sdx, gr = g)``)."""
    import pymc as pm

    from autobayes.monotonic import coefficient_rv, monotonic_effect, simplex_rv
    from autobayes.priors import dirichlet_concentration

    pmm = pm.modelcontext(None)
    roles, built = names(spec, prefix), {}
    for key, (m, _, _, _) in simplexes(spec).items():
        pmm.add_coord(roles["coordinate"][key], [str(j) for j in range(1, m.D + 1)])
        built[key] = simplex_rv(roles["simplex"][key], dirichlet_concentration(priors[key], m.D), roles["coordinate"][key])
    for t in spec.sp_terms:
        sp, value = t.special, None
        for j, (column, _) in enumerate(sp.mo_parts):
            m = sp.coding_of(column)
            part = monotonic_effect(built[sp.simplex_key(j)], m.codes(data[column]), m.D)
            value = part if value is None else value * part
        for call in sp.me_parts:
            part = latent_rows(model, pmm, call)
            value = part if value is None else value * part
        cov = sp.covariate(data)
        if cov is not None:
            value = value * np.asarray(cov, dtype=np.float64)
        eta = eta + coefficient_rv(roles["b"][t.name], coefficients[t.name]) * value
    return eta


def special_link(model, d):
    """The Bambi link of distributional parameter ``d.response`` whose formula ``d`` has special products (its link the log,
    ``check_dpar_link``): Bambi hands its linear predictor to the link's ``linkinv_backend`` inside the model it builds, which
    adds the products there (``add_effects``, under the formula's own priors, ``Priors.dpar_priors``) before the exponential.
    Named ``log_special``, which ``Model._dpar_design`` expects of such a formula."""
    import bambi as bmb
    import pytensor.tensor as pt

    prefix = f"{model._bambi_dpar(d.response)}_"

    def linkinv_backend(eta):
        sub = model.priors.dpar_priors(model.spec, d.response)
        return pt.exp(add_effects(model, eta, d, sub.coefficient_priors(d), sub.special_simo_priors(d), model.data, prefix))

    return bmb.Link("log_special", link=lambda v: np.log(np.asarray(v, dtype=float)), linkinv=np.exp,
                    linkinv_backend=linkinv_backend)


def latent_rows(model, pmm, call: str):
    """The latent values of the ``me()`` term ``call`` per fitted row in the graph ``pmm``."""
    from autobayes.formula import measurement_error_columns

    x = measurement_error_columns(call)[0]
    latent = pmm[model.me_names[x]["latent"]]
    block = model._measurement_rows(x)
    return latent if block is None else latent[block]


def check_effects(model, spec, value_name: str, link: str, data: pd.DataFrame, prefix: str = "") -> dict:
    """Each special product's contribution read back off the graph: at the graph's initial point with every special coefficient
    0 and every simplex at unequal steps, the value ``value_name`` (``mu``, or a distributional parameter's per-row
    deterministic; on its link scale, identity or log) must move, row by row, by exactly ``c * prod(parts) * Csp`` recomputed
    from the data (the latent values of an ``me()`` part read at that point) when the term's coefficient goes from 0 to ``c``.
    Returns the terms' blocks by name (``SpecialBlock``)."""
    from autobayes.monotonic import monotonic_values, stick_breaking_free

    if link not in ("identity", "log"):
        raise RuntimeError(f"the link of {value_name!r} is {link}; the special terms are checked under the identity and log links")
    pmm = model.pymc_model
    roles = names(spec, prefix)
    for key, name in roles["simplex"].items():
        if pmm[name] not in pmm.free_RVs or pmm.rvs_to_values[pmm[name]].name != f"{name}_stickbreaking__":
            raise RuntimeError(f"{name!r} is not a free simplex of the graph on Stan's stick-breaking coordinates")
    for name in roles["b"].values():
        if pmm[name] not in pmm.free_RVs or pmm[name].type.ndim != 0:
            raise RuntimeError(f"{name!r} is not a free scalar of the graph")
    outputs = [pmm[value_name]] + [pmm[model.me_names[x]["latent"]] for x in model.me_names]
    fn = pmm.compile_fn(pmm.replace_rvs_by_values(outputs), inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore",
                        mode="FAST_COMPILE")
    point = pmm.initial_point()
    point.update({name: np.asarray(0.0) for name in roles["b"].values()})
    tests = {}
    for key, (m, _, _, _) in simplexes(spec).items():
        tests[key] = np.arange(1.0, m.D + 1.0) / (m.D * (m.D + 1) / 2.0)
        point[f"{roles['simplex'][key]}_stickbreaking__"] = stick_breaking_free(tests[key])
    base = [np.asarray(v, dtype=float) for v in fn(point)]
    latent = dict(zip(model.me_names, base[1:]))
    blocks = {}
    for t in spec.sp_terms:
        sp = t.special
        values, codes = np.ones(len(data)), {}
        for j, (column, _) in enumerate(sp.mo_parts):
            m = sp.coding_of(column)
            codes[sp.simplex_key(j)] = m.codes(data[column])
            values = values * monotonic_values(tests[sp.simplex_key(j)], codes[sp.simplex_key(j)])
        for call in sp.me_parts:
            from autobayes.formula import measurement_error_columns

            x = measurement_error_columns(call)[0]
            rows = model._measurement_rows(x)
            values = values * (latent[x] if rows is None else latent[x][rows])
        cov = sp.covariate(data)
        if cov is not None:
            values = values * cov
        c = 1.0 / max(1.0, float(np.max(np.abs(values))))
        hi = np.asarray(fn({**point, roles["b"][t.name]: np.asarray(c)})[0], dtype=float).ravel()
        lo = base[0].ravel()
        moved = hi - lo if link == "identity" else np.log(hi) - np.log(lo)
        if moved.shape != values.shape or not np.allclose(moved, c * values, rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"the graph's {value_name!r} does not move by the values of {t.name!r} times its coefficient")
        blocks[t.name] = SpecialBlock(t.name, tuple((sp.mo_parts[j][0], sp.simplex_key(j), codes[sp.simplex_key(j)],
                                                     sp.coding_of(sp.mo_parts[j][0]).D) for j in range(len(sp.mo_parts))),
                                      sp.me_parts, cov)
    return blocks


@dataclass(frozen=True)
class SpecialBlock:
    """The design of one special product on the fitted rows: ``mo`` holds ``(column, simplex key, codes, D)`` per ``mo()`` part
    (brms's ``Xmo_k`` and ``Jmo[k]``), ``me`` its ``me()`` parts and ``covariate`` brms's ``Csp`` (None without one)."""
    term: str
    mo: tuple
    me: tuple
    covariate: np.ndarray | None


# -- predictions ------------------------------------------------------------------------------------------------------
def contribution(model, post, df: pd.DataFrame, spec=None, prefix: str = "", *, in_sample: bool = True):
    """(chain, draw, row) draws of the special products' part of a linear predictor on the rows of ``df`` (the response formula's,
    or ``spec``'s with ``prefix``): each coefficient's draws times its parts and covariate, ``mo()`` from the simplex draws at the
    rows' codes (the fitted coding, refusing others), ``me()`` from the latent values' draws on the fitted rows. 0 without one."""
    from autobayes.monotonic import monotonic_values

    spec = model.spec if spec is None else spec
    roles, out = names(spec, prefix), 0.0
    for t in spec.sp_terms:
        sp = t.special
        value = post[roles["b"][t.name]].to_numpy()[..., None]
        for j, (column, _) in enumerate(sp.mo_parts):
            simplex = post[roles["simplex"][sp.simplex_key(j)]].transpose("chain", "draw", roles["coordinate"][sp.simplex_key(j)])
            value = value * monotonic_values(simplex.to_numpy(), sp.coding_of(column).codes(df[column], where="newdata column"))
        for call in sp.me_parts:                              # the fitted rows only (new rows are refused before)
            from autobayes.formula import measurement_error_columns
            from autobayes.predict import latent_draws

            value = value * latent_draws(model, post, measurement_error_columns(call)[0])
        cats = {c.name: c.levels for c in spec.categoricals}
        cov = sp.covariate(df, cats.get(sp.factor), where="newdata column")
        out = out + (value if cov is None else value * cov)
    return out


# -- describe -----------------------------------------------------------------------------------------------------
def row_expression(sp: SpecialProduct, prefix: str = "") -> str:
    """Row i's value of a special product as ``describe()`` writes it: ``mo(simo_<key>, ord_i) * x_i * 1[z_i = q]``."""
    out = [f"mo({prefix}simo_{sp.simplex_key(j)}, {c}_i)" for j, (c, _) in enumerate(sp.mo_parts)]
    out += [f"Xme_{me_call(call, call)[0]},i" for call in sp.me_parts]
    out += [f"{c}_i" for c in sp.numeric] + ([f"1[{sp.factor}_i = {sp.level}]"] if sp.factor is not None else [])
    return " * ".join(out)


def prior_lines(spec, coefficients, priors, describe_prior, prefix: str = "") -> list[str]:
    """``describe()``'s prior lines of the special products: each coefficient's (brms class b) and each simplex's (class simo)."""
    from autobayes.default_priors import brms_rename
    from autobayes.priors import dirichlet_concentration

    roles = names(spec, prefix)
    lines = [f"  b_{t.name} ~ {describe_prior(coefficients[t.name])}   (brms's bsp: class b, coefficient {brms_rename(t.name)})"
             for t in spec.sp_terms]
    for key, (m, coef, _, _) in simplexes(spec).items():
        con = ", ".join(f"{v:g}" for v in dirichlet_concentration(priors[key], m.D))
        lines.append(f"  {roles['simplex'][key]} ~ {describe_prior(priors[key])}   (brms class simo, coefficient {coef}; "
                     f"concentrations {con})")
    return lines


def describe_lines(spec, prefix: str = "") -> list[str]:
    """``describe()``'s lines for the special products of a formula."""
    lines = []
    for t in spec.sp_terms:
        sp = t.special
        ids = [id for _, id in sp.mo_parts if id is not None]
        lines.append(f"  {t.name}: brms's special term, b_{t.name} * {row_expression(sp, prefix)} (brms's bsp times its parts and "
                     "covariate Csp; not centred)" + (f"; the simplex of id {ids[0]!r} is shared by the formula's mo() terms with "
                                                       "that id" if ids else "")
                     + ("; the factor coded by an indicator per level, as written (brms 2.23.0 multiplies every level's term by "
                        "the first level's indicator; divergence D197)" if sp.factor is not None and sp.coding == "indicator" else ""))
    return lines
