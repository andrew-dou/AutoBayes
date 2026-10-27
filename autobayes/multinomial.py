"""Count-matrix responses, brms's ``multinomial()`` and ``dirichlet_multinomial()`` (brms-parity F01a, docs/multinomial.md).

The formula's left-hand side is brms's matrix response ``cbind(y_1, ..., y_K) | trials(n)``: the K count columns, in the
order written, are the categories (the first the reference, brms's default ``refcat``), and ``n`` each row's total, which
brms requires and checks against the row sums. Each non-reference category ``k`` has its own linear predictor (brms's dpar
``mu<k>``: ``mu`` and the category's letters and digits, ``default_priors.brms_dpar_names``), by default on the response
formula's right-hand side; a list of formulas gives a category one of its own, ``["cbind(m1, m2, m3) | trials(size) ~ x +
(1 | g)", "mum3 ~ z"]`` (brms's ``bf(..., mum3 ~ z)``)::

    mu_k,i = a_k + (X_k,i - mean(X_k)) b_k + sum_g u_k,g[g_i]          k = 2..K   (brms's mu<k>; mu_1,i = 0)
    u_k,g[j] = sd_k,g z_k,g[j],  z_k,g[j] ~ Normal(0, 1)                     (each category's group effects, its own SD)
    p_i = softmax(0, mu_2,i, ..., mu_K,i)
    y_i ~ Multinomial(n_i, p_i)                    or   y_i ~ DirichletMultinomial(n_i, phi p_i)

Each category's design is centred on its own column means (brms's ``Xc_mu<k>``, ``Intercept_mu<k>``), and each of its
group-level terms is a group of its own (brms numbers them ``1, 2, ...`` category by category: ``sd_1``, ``z_1`` for
``mum2``'s ``(1 | g)``, ``sd_2`` for ``mum3``'s). The likelihoods are ``families.multinomial``'s.

Bambi has no such model, so ``MultinomialModel`` builds the PyMC graph itself (``model.bambi_model`` is None), as
``NonlinearModel`` does (C02), and reuses its helpers. Names: per category ``mum2_Intercept`` (the centred intercept),
``mum2_x`` (a vector over ``mum2_z_dim`` for a categorical term), ``mum2_1|g_sigma``, ``mum2_1|g_offset`` and ``mum2_1|g``
(Bambi's names under the category's prefix, the levels on ``g__factor_dim``), the deterministics ``mum2_b_Intercept`` (the
raw-scale intercept, brms's ``b_mum2_Intercept``), ``mum2`` (the row values) and ``mu`` (``(rows, K - 1)`` over ``mu_dim``),
``phi`` for the Dirichlet-multinomial, and the observed counts under the response's name, ``cbind(m1, m2, m3)``, over
``(__obs__, category)``.

Priors (``Priors(intercept=..., b=..., sd=...)``) apply to every category whose formula is the response formula's, one prior
per class shared by those categories as for B05's categorical family (brms takes one per ``dpar``: divergence D137); a
category with a formula of its own takes its priors from ``Priors(dpars={"mum3": {"intercept": ..., "b": ..., "sd": ...}})``
(brms's ``dpar = mum3``), which any category may also be given. ``phi`` takes one positive-support prior (brms's default
``gamma(0.01, 0.01)``).

The reference category is the first column; the columns written in another order choose another (brms's ``refcat``).
Refused, naming the construct (divergence D136, unless brms refuses it too): another link than the logit (brms's only), a
response without ``trials()`` (brms: "Specifying 'trials' is required for this model."), counts that are not whole,
non-negative and summing to the trials (brms: "Number of trials does not match the number of events."; brms accepts a
negative count, for which Stan's multinomial log density raises and brms's Dirichlet-multinomial one is -Inf), fewer than two
categories, other addition terms (``weights()``,
``cens()``, ``trunc()``, ``mi()``), a formula of the reference category (brms refuses it) or of ``phi`` (brms takes it), a
response formula's right-hand side that no category uses (brms ignores it silently), the removal of a category's intercept,
``center_predictors=False``, varying slopes ``(1 + x | g)`` and ``(1 + x || g)``, correlation IDs, ``mm()`` and ``gr(by =
)``, ``offset()``, ``mo()``, ``gp()``, ``me()``, ``mi()``, ``cs()``, autocorrelation and ``sar()`` terms in a category's formula
(and ``data2``), and
(elsewhere) multivariate, mixture and non-linear models of these families, ``hypothesis()``, ``conditional_effects()``,
``prior_draws()`` and ``bayes_R2()`` (brms: "'bayes_R2' is not defined for unordered categorical models.").
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from autobayes.formula import (_CS_RE, _FORMULA_RE, _GROUP_RE, _IDENT, FormulaError, ModelSpec, _split_top_level, categorical_labels,
                               check_counts, factor_column, group_labels, parse_formula, pop_term_design, reject_complex, validate_data)
from autobayes.model import Design, GroupBlock, Model
from autobayes.priors import _SCALE, AUXILIARY, Prior, PriorError, Priors, describe_prior

DIVERGENCE = "D136"
_CBIND_RE = re.compile(r"^\s*cbind\s*\((?P<columns>[^()]*)\)\s*(?:\|(?P<additions>[^~]*))?~\s*(?P<rhs>.+?)\s*$", re.S)
_TRIALS_RE = re.compile(rf"^trials\s*\(\s*({_IDENT})\s*\)$", re.S)
FORM = ("'cbind(y1, ..., yK) | trials(n) ~ terms' (brms's matrix response: the count columns, the first the reference "
        "category, and each row's total n), or a list of formulas: that formula first, then formulas of non-reference "
        "categories' linear predictors, 'mu<category> ~ terms' (brms's bf(cbind(...) | trials(n) ~ ..., mum3 ~ ...); "
        "docs/multinomial.md)")


def _refuse(text: str, what: str) -> None:
    raise FormulaError(f"{text!r}: {what} is not supported for the multinomial families (docs/multinomial.md, "
                       f"divergence {DIVERGENCE})")


# -- the formula ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MultinomialSpec:
    """A parsed count-matrix model (F01a): ``formula`` the response formula as written and ``formulas`` every formula,
    ``response`` the canonical matrix ``cbind(m1, m2, m3)``, ``categories`` its columns in order (the first the reference),
    ``trials`` the trials column, ``rhs`` the response formula's right-hand side as written, ``mus`` one ``ModelSpec`` per
    non-reference category (its ``response`` is brms's name of the category's linear predictor, ``mum2``; levels and codings
    known once the data are seen) and ``own`` the names of those with a formula of their own.

    It stands where a ``ModelSpec`` stands on a model, as ``NonlinearSpec`` does: the views of a single linear formula the
    rest of the library reads are empty here; the categories' formulas are in ``mus``."""

    formula: str
    formulas: tuple
    response: str
    categories: tuple
    trials: str
    rhs: str
    mus: tuple
    own: tuple

    is_multinomial = True                 # what the shared code dispatches on (getattr(spec, "is_multinomial", False))
    is_nonlinear = False
    intercept = False
    pop_terms = ()
    group_terms = ()
    terms = ()
    columns = ()
    column_terms = ()
    coefficient_labels = ()
    predictors = ()
    interactions = ()
    interaction_names = ()
    categorical_names = ()
    categoricals = ()
    transform_terms = ()
    bambi_terms = ()
    mo_terms = ()
    monotonics = ()
    gp_terms = ()
    autocor = None
    sar = None                            # a SAR term (F02b): none, as a count-matrix model takes none (D136)
    me_terms = ()
    mi_terms = ()
    cs_terms = ()
    missing_columns = ()
    custom_group_terms = ()
    shared_ids = ()
    groups = ()
    group_coefficients = ()
    correlated_terms = ()
    has_group = False
    has_slopes = False
    levels_by_group = MappingProxyType({})
    dpars = ()
    dpar_names = ()
    offset = None
    additions = ()

    def addition(self, kind: str):
        """The response's addition terms besides ``trials()`` are refused (D136)."""
        return None

    @property
    def reference(self) -> str:
        return self.categories[0]

    @property
    def mu_names(self) -> tuple[str, ...]:
        """brms's names of the non-reference categories' linear predictors, in category order (``mum2``, ``mum3``)."""
        return tuple(d.response for d in self.mus)

    def mu(self, name: str) -> ModelSpec:
        """The formula of the category whose linear predictor is ``name`` (``KeyError`` when there is none)."""
        for d in self.mus:
            if d.response == name:
                return d
        raise KeyError(name)

    @property
    def mu_group_terms(self) -> tuple:
        """``(category name, GroupTerm)`` for every group-level term of every category's formula, in category order."""
        return tuple((d.response, t) for d in self.mus for t in d.group_terms)

    @property
    def factors(self) -> tuple[str, ...]:
        """The grouping factors of every category's formula, each once, in order."""
        return tuple(dict.fromkeys(t.factor for _, t in self.mu_group_terms))

    @property
    def all_categoricals(self) -> tuple:
        """The declared categorical columns of every category's formula, each once (the first formula's coding)."""
        out = []
        for d in self.mus:
            out += [c for c in d.categoricals if all(c.name != o.name for o in out)]
        return tuple(out)

    @property
    def model_columns(self) -> tuple[str, ...]:
        """The data columns the categories' population-level terms read, each once, in order."""
        return tuple(dict.fromkeys(c for d in self.mus for c in d.pop_columns))

    @property
    def group_columns(self) -> tuple[str, ...]:
        """The columns the grouping factors are read from, each once."""
        return tuple(dict.fromkeys(c for _, t in self.mu_group_terms for c in t.operands))

    @property
    def data_columns(self) -> tuple[str, ...]:
        """Every column the model reads: the counts, the trials, ``model_columns`` and ``group_columns``."""
        return tuple(dict.fromkeys([*self.categories, self.trials, *self.model_columns, *self.group_columns]))

    @property
    def canonical_rhs(self) -> str:
        """The response formula's right-hand side as the canonical formula writes it (a category's that uses it, or the
        written one when every category has a formula of its own)."""
        shared = [d for d in self.mus if d.response not in self.own]
        return shared[0].canonical_formula.split("~", 1)[1].strip() if shared else self.rhs

    @property
    def canonical_formula(self) -> str:
        return f"{self.response} | trials({self.trials}) ~ {self.canonical_rhs}"


def _split_formulas(formula) -> tuple[str, tuple[str, ...]]:
    if isinstance(formula, str):
        return formula, ()
    if isinstance(formula, (list, tuple)) and formula and all(isinstance(f, str) for f in formula):
        return formula[0], tuple(formula[1:])
    raise FormulaError(f"the multinomial families take {FORM}; got {formula!r}")


def _response_side(main: str) -> tuple[tuple[str, ...], str, str]:
    """The categories, the trials column and the right-hand side of the response formula ``main``, or refused."""
    m = _CBIND_RE.match(main)
    if not m:
        raise FormulaError(f"{main!r}: the multinomial families' response is a count matrix written over its count columns with "
                           f"cbind() (brms's matrix response; a data frame holds no matrix column): {FORM}")
    columns = [c.strip() for c in m.group("columns").split(",")]
    if any(not re.fullmatch(_IDENT, c) for c in columns):
        raise FormulaError(f"{main!r}: cbind() takes the count columns by name, cbind(y1, y2, ...); got {m.group('columns')!r}")
    if len(columns) < 2:
        raise FormulaError(f"{main!r}: At least 2 response categories are required (brms); cbind() has {len(columns)}")
    if len(set(columns)) != len(columns):
        raise FormulaError(f"{main!r}: cbind() names a column twice; each category is one count column")
    additions = m.group("additions")
    if additions is None:
        raise FormulaError(f"{main!r}: Specifying 'trials' is required for this model (brms): write 'cbind(...) | trials(n) ~ ...' "
                           "with n each row's total")
    pieces = [p.strip() for p in _split_top_level(additions.strip())]
    trials = [t.group(1) for t in map(_TRIALS_RE.match, pieces) if t]
    others = [p for p in pieces if not _TRIALS_RE.match(p)]
    if others:
        _refuse(main, f"the addition term {others[0]!r} (brms's weights(), cens(), trunc(), mi() and the others besides trials())")
    if len(trials) != 1:
        raise FormulaError(f"{main!r}: the left-hand side takes exactly one trials(n) after the '|'")
    if trials[0] in columns:
        raise FormulaError(f"{main!r}: the trials column {trials[0]!r} cannot also be a category's counts")
    return tuple(columns), trials[0], m.group("rhs")


def _check_category_terms(text: str, rhs: str) -> None:
    """Refuse, before the term model reads a category's right-hand side, what it would refuse in other words: a
    category-specific term ``cs(x)`` (an ordinal family's) and a correlation ID ``(1 |p| g)`` (D136)."""
    for piece in _split_top_level(rhs):
        if _CS_RE.match(piece):
            _refuse(text, f"the category-specific term {piece!r} (brms's ordinal families' cs())")
        gm = _GROUP_RE.match(piece)
        if gm and "|" in gm.group(3):
            _refuse(text, f"the correlation ID of {piece!r}")


def _check_category_formula(text: str, d: ModelSpec) -> None:
    """Refuse what a category's formula takes in brms but no frozen case verifies here (D136)."""
    if not d.intercept:
        _refuse(text, "a category's linear predictor without an intercept ('0 +' or '- 1')")
    if d.offset is not None:
        _refuse(text, f"the offset term {d.offset.term!r}")
    if d.expression_terms:                                             # G04 (divergence D182)
        _refuse(text, f"the in-formula transform {d.expression_terms[0].basis!r}")
    for kind, terms in (("monotonic term", d.mo_terms), ("measurement-error term", d.me_terms), ("missing-value term", d.mi_terms)):
        if terms:
            _refuse(text, f"the {kind} {terms[0].name!r}")
    if d.gp_terms:
        _refuse(text, f"the Gaussian process {d.gp_terms[0].term!r}")
    if d.autocor is not None:
        _refuse(text, f"the autocorrelation term {d.autocor.term!r}")
    if getattr(d, "sar", None) is not None:
        _refuse(text, f"the SAR term {d.sar.term!r}")
    for t in d.group_terms:
        if t.custom:
            _refuse(text, f"the group-level term '{t.canonical}' (mm() and gr(by = ))")
        if t.slopes:
            _refuse(text, f"the varying slope(s) of '{t.canonical}' (brms gives each category its own SDs and, for '|', its own "
                          "correlation matrix)")


def parse_multinomial(formula, data: pd.DataFrame | None = None, family: str = "multinomial",
                      categorical: Mapping[str, object] | None = None) -> MultinomialSpec:
    """Parse and validate a count-matrix model (``FORM``), optionally against a data frame (the counts and trials, and each
    category's formula, checked as a distributional formula is: ``formula.validate_data(..., dpar=True)``). ``categorical``
    is shared: a declared column is categorical in every category's formula that names it as a main effect, and must be
    one."""
    from autobayes.default_priors import brms_dpar_names
    from autobayes.families import get_family

    fam = get_family(family)
    main, rest = _split_formulas(formula)
    categories, trials, rhs = _response_side(main)
    dpar_names = brms_dpar_names(categories)
    reference, names = dpar_names[0], dpar_names[1:]
    own: dict[str, str] = {}
    texts: dict[str, str] = {}
    for text in rest:
        m = _FORMULA_RE.match(text)
        if not m or "|" in text.split("~", 1)[0]:
            raise FormulaError(f"{text!r}: a category's formula is '<mu<category>> ~ <terms>' with brms's name of its linear "
                               f"predictor on the left ({', '.join(names)}); {FORM}")
        name, prhs = m.group(1), m.group(2)
        if name == fam.scale_parameter:
            _refuse(text, f"a formula of {name} (brms's distributional {name})")
        if name not in names:
            hint = (f" ({name!r} is the reference category's, whose linear predictor is 0)" if name == reference else "")
            raise FormulaError(f"{text!r}: The parameter {name!r} is not a valid distributional or non-linear parameter (brms){hint}; "
                               f"the categories' linear predictors are {list(names)}")
        if name in own:
            raise FormulaError(f"{text!r}: {name!r} has two formulas; give each category one")
        own[name], texts[name] = prhs, text.strip()
    if len(own) == len(names) and re.sub(r"\s+", "", rhs) != "1":
        raise FormulaError(f"{main!r}: every non-reference category has a formula of its own, so the response formula's terms "
                           f"'{rhs}' apply to none of them (brms ignores them silently); write the response formula as "
                           f"'{main.split('~', 1)[0].strip()} ~ 1'")
    rhs_of = {name: own.get(name, rhs) for name in names}
    for name in names:
        _check_category_terms(texts.get(name, main.strip()), rhs_of[name])
    spec0 = {name: parse_formula(f"{name} ~ {rhs_of[name]}", None, "gaussian", None) for name in names}
    mains = {name: {t.name for t in s.pop_terms if t.kind == "numeric"} for name, s in spec0.items()}
    if isinstance(categorical, Mapping):
        stray = [col for col in categorical if not any(col in m for m in mains.values())]
        if stray:
            raise FormulaError(f"{stray} declared categorical, but no category's formula names "
                               f"{'it' if len(stray) == 1 else 'them'} as a main effect")
        routed = {name: ({k: v for k, v in categorical.items() if k in m} or None) for name, m in mains.items()}
    elif categorical is not None:
        raise FormulaError(f"categorical must be a mapping {{column: reference level}}, got {type(categorical).__name__}")
    else:
        routed = {name: None for name in names}
    reserved = set(categories) | {trials}
    mus = []
    for name in names:
        text = texts.get(name, main.strip())
        d = parse_formula(f"{name} ~ {rhs_of[name]}", None, "gaussian", routed[name])
        _check_category_formula(text, d)
        reads = [*d.pop_columns, *(c for t in d.group_terms for c in t.operands)]
        clash = [c for c in reads if c in reserved]
        if clash:
            raise FormulaError(f"{text!r}: {clash[0]!r} is {'the trials column' if clash[0] == trials else 'a category of the response'}; "
                               "it cannot also be a predictor or a grouping factor")
        if data is not None:
            d = validate_data(d, data, "gaussian", dpar=True)
        mus.append(d)
    response = f"cbind({', '.join(categories)})"
    spec = MultinomialSpec(formula=main.strip(), formulas=(main.strip(), *(t.strip() for t in rest)), response=response,
                           categories=categories, trials=trials, rhs=rhs, mus=tuple(mus), own=tuple(n for n in names if n in own))
    if data is not None:
        response_counts(spec, data)
    return spec


def response_counts(spec: MultinomialSpec, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The count matrix ``(rows, K)`` and the trials of ``data``, validated: every column present, whole non-negative numbers
    (``formula.check_counts``: brms refuses a non-integer; it accepts a negative count, which this library refuses), no missing
    value, and each row's
    counts summing to its trials (brms: "Number of trials does not match the number of events.")."""
    columns = [*spec.categories, spec.trials]
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise FormulaError(f"columns not in data: {missing}")
    na = data[columns].isna().any()
    if na.any():
        raise FormulaError(f"missing values in columns {list(na[na].index)}; drop or impute them explicitly before fitting.")
    Y = np.column_stack([check_counts(f"count column {c!r}", data[c]) for c in spec.categories])
    n = check_counts(f"trials {spec.trials!r}", data[spec.trials])
    bad = np.flatnonzero(Y.sum(axis=1) != n)
    if bad.size:
        i = int(bad[0])
        raise FormulaError(f"Number of trials does not match the number of events (brms): row {i} has counts "
                           f"{Y[i].tolist()} summing to {int(Y[i].sum())} and trials {spec.trials!r} = {int(n[i])}")
    return Y, n


# -- priors -----------------------------------------------------------------------------------------------------------------
_CATEGORY_KEYS = ("intercept", "b", "sd")
_NOT_HERE = (("nlpars", "the priors of a non-linear model (nl=True)"), ("components", "the priors of a mixture"),
             ("theta", "the priors of a mixture"), ("resp", "the priors of a multivariate model"),
             ("rescor", "the priors of a multivariate model"), ("cor", "an LKJ prior of a correlated term (1 + x | g), refused here"),
             ("simo", "the simplex prior of a monotonic term"), ("sdgp", "a Gaussian process's prior"),
             ("lscale", "a Gaussian process's prior"), ("ar", "an autocorrelation term's prior"),
             ("ma", "an autocorrelation term's prior"), ("meanme", "a measurement-error term's prior"),
             ("sdme", "a measurement-error term's prior"), ("nu", "the Student-t degrees of freedom"))


def _shared(spec: MultinomialSpec, priors: Priors) -> list[str]:
    """The categories that take the top-level intercept, b and sd: those without an entry in ``dpars``."""
    entries = priors.dpars or {}
    return [name for name in spec.mu_names if name not in entries]


def category_priors(priors: Priors, spec: MultinomialSpec, name: str) -> Priors:
    """The priors of category ``name``'s formula as a ``Priors`` of that formula (``intercept``, ``b``, ``sd``): its entry of
    ``Priors(dpars={name: {...}})`` when it has one, else the top-level ``intercept``, ``b`` and ``sd``."""
    entry = (priors.dpars or {}).get(name)
    if entry is None:
        return Priors(intercept=priors.intercept, b=priors.b, sd=priors.sd)
    return Priors(intercept=entry.get("intercept"), b=entry.get("b"), sd=entry.get("sd"))


def check_multinomial_priors(priors: Priors, spec: MultinomialSpec, fam) -> None:
    """``Priors.validate`` for a count-matrix model: ``dpars`` names non-reference categories only, each a mapping over
    ``"intercept"``, ``"b"`` and ``"sd"``; a category with a formula of its own needs an entry; the top-level ``intercept``,
    ``b`` and ``sd`` are given exactly when some category takes them; each category's priors are checked by the rules of a
    linear formula (location priors on the intercept and coefficients, positive-support priors on the SDs, exact coverage);
    ``phi`` is the Dirichlet-multinomial's one positive-support prior; every other class is refused."""
    for name, what in _NOT_HERE:
        if getattr(priors, name) is not None:
            raise PriorError(f"{name} is {what}; a {fam.name} model takes intercept, b, sd"
                             + (", phi" if fam.scale_parameter else "") + " and dpars (docs/multinomial.md)")
    for name in AUXILIARY:
        if getattr(priors, name) is not None and name != fam.scale_parameter:
            raise PriorError(f"the {fam.name} family has no {AUXILIARY[name]}; do not pass {name}")
    if fam.scale_parameter is not None:
        phi = priors.phi
        if phi is None:
            raise PriorError(fam.scale_missing_error)
        if not isinstance(phi, Prior) or phi.name not in _SCALE:
            raise PriorError(f"prior for 'phi' must be one positive-support prior ({sorted(_SCALE)}; brms's real<lower=0> phi, "
                             f"which takes no formula of its own here, divergence {DIVERGENCE}), got {phi!r}")
    entries = priors.dpars
    if entries is not None:
        if not isinstance(entries, Mapping):
            raise PriorError(f"dpars must be a mapping from a category's linear predictor ({list(spec.mu_names)}) to its priors, "
                             f"got {type(entries).__name__}")
        unknown = [k for k in entries if k not in spec.mu_names]
        if unknown:
            raise PriorError(f"dpars names {unknown}; the non-reference categories' linear predictors are {list(spec.mu_names)} "
                             f"(brms's dpar names; the reference category {spec.reference!r} has none)")
        for name, entry in entries.items():
            if not isinstance(entry, Mapping) or set(entry) - set(_CATEGORY_KEYS):
                raise PriorError(f"dpars[{name!r}] is a mapping with the keys {list(_CATEGORY_KEYS)} (brms's class Intercept, b and "
                                 f"sd with dpar = {name}), got {entry!r}")
    missing = [name for name in spec.own if name not in (entries or {})]
    if missing:
        raise PriorError(f"the categories {missing} have formulas of their own ({', '.join(repr(spec.mu(n).formula) for n in missing)}), "
                         f"which need their own priors: pass dpars={{{', '.join(f'{n!r}: {{...}}' for n in missing)}}} with "
                         "'intercept', 'b' and 'sd' as the formula needs them (brms's dpar =)")
    shared = _shared(spec, priors)
    if not shared:
        stray = [k for k in _CATEGORY_KEYS if getattr(priors, k) is not None]
        if stray:
            raise PriorError(f"every category's priors are in dpars, so {stray} at the top level apply to none; omit them")
    for d in spec.mus:
        sub = category_priors(priors, spec, d.response)
        try:
            Priors(intercept=sub.intercept, b=sub.b, sd=sub.sd,
                   **({fam.scale_parameter: priors.phi} if fam.scale_parameter else {})).validate(d, fam)
        except PriorError as e:
            where = f"dpars[{d.response!r}]" if d.response not in shared else "the top-level intercept, b and sd"
            raise PriorError(f"category {d.response!r} ('{d.formula}', its priors from {where}): {e}") from None


def prior_labels(priors: Priors, spec: MultinomialSpec, fam) -> list[tuple[str, Prior]]:
    """Every prior of a count-matrix model with the label messages and fit records give it: ``mum2:intercept``,
    ``mum2:b:x``, ``mum2:sd:g``, ..., ``phi``, in that order."""
    from autobayes.formula import sd_tag

    out = []
    for d in spec.mus:
        sub = category_priors(priors, spec, d.response)
        out.append((f"{d.response}:intercept", sub.intercept))
        out += [(f"{d.response}:b:{col}", p) for col, p in sub.coefficient_priors(d).items()]
        out += [(f"{d.response}:sd:{sd_tag(g, c)}", p) for (g, c), p in sub.sd_priors(d).items()]
    if fam.scale_parameter is not None:
        out.append((fam.scale_parameter, priors.phi))
    return out


def multinomial_improper(priors: Priors, spec: MultinomialSpec) -> list[str]:
    """``Priors.improper`` for a count-matrix model: the intercepts and coefficients under a flat prior, ``mum2:intercept``,
    ``mum2:b:x``."""
    out = []
    for d in spec.mus:
        sub = category_priors(priors, spec, d.response)
        labelled = [("intercept", sub.intercept)] + [(f"b:{col}", p) for col, p in sub.coefficient_priors(d).items()]
        out += [f"{d.response}:{label}" for label, p in labelled if isinstance(p, Prior) and p.name == "Flat"]
    return out


# -- the design --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MuDesign:
    """The design of one category's formula: ``columns`` its design columns (no intercept column), ``X_raw`` brms's
    ``X_<name>`` without its intercept column, ``x_means`` their means over the fitted rows, ``X_model`` the centred design
    (brms's ``Xc_<name>``), ``group_blocks`` one ``GroupBlock`` per group-level coefficient keyed ``(factor, coefficient)``
    and ``Z`` the blocks side by side."""
    name: str
    columns: tuple
    X_raw: np.ndarray
    X_model: np.ndarray
    x_means: np.ndarray
    group_blocks: Mapping
    Z: np.ndarray


@dataclass(frozen=True)
class MultinomialDesign(Design):
    """What a count-matrix model was fit to: ``y`` the counts ``(rows, K)``, ``trials`` each row's total, ``categories``
    the count columns, and each category's design (``mus``). The linear design fields of ``Design`` are empty."""
    categories: tuple = ()
    mus: Mapping = field(default_factory=lambda: MappingProxyType({}))


def mu_design(d: ModelSpec, frame: pd.DataFrame) -> np.ndarray:
    """The raw design of category formula ``d`` on ``frame`` without its intercept column: each population-level term's
    block (``pop_term_design``)."""
    blocks = [pop_term_design(t, frame, d.categoricals) for t in d.pop_terms]
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(frame), 0))


# -- the model ---------------------------------------------------------------------------------------------------------------
class MultinomialModel(Model):
    """A count-matrix model (F01a, docs/multinomial.md), built by ``ab.multinomial(...)``, ``ab.dirichlet_multinomial(...)``
    or ``ab.Model(..., family="multinomial")``: ``spec`` is a ``MultinomialSpec``, ``design`` a ``MultinomialDesign``,
    ``pymc_model`` the graph this class builds (``bambi_model`` is None; ``fit()`` samples ``pymc_model`` through
    ``sample_posterior``), ``categories`` the count columns. ``describe()``, ``value_point()``, ``predict()``,
    ``prior_table()``, ``prior_predictive()``, ``fit()``, the target identity and fit records are offered."""

    def __init__(self, formula, data: pd.DataFrame, priors: Priors, *, family: str = "multinomial", center_predictors: bool = True,
                 categorical: Mapping[str, object] | None = None, nu: float | None = None, link: str | None = None,
                 families: list[str] | None = None, order: str | bool | None = None, imputed: tuple[str, ...] = (),
                 quantile: float | None = None, data2: Mapping[str, object] | None = None):
        from autobayes.families import FAMILIES, get_family

        fam = get_family(family)
        if not fam.matrix_response:
            offering = [name for name, other in FAMILIES.items() if other.matrix_response]
            raise ValueError(f"MultinomialModel builds the {' and '.join(offering)} families; got {family!r}")
        if link is not None and link != fam.link:
            raise ValueError(f"{link!r} is not a supported link for family {fam.name!r}. Supported links are: {fam.link!r} (brms)")
        if nu is not None:
            raise ValueError(f"nu applies only to the {' or '.join(n for n, s in FAMILIES.items() if 'nu' in s.fixed_parameters)} family")
        if quantile is not None:
            raise ValueError(f"quantile= fixes the asym_laplace family's quantile; the {fam.name} family has none")
        if families is not None or order is not None:
            raise ValueError(f"families= and order= are a mixture's (ab.mixture, docs/mixtures.md); the {fam.name} family takes neither")
        if imputed:
            raise FormulaError(f"a {fam.name} model in a multivariate formula is not supported (divergence {DIVERGENCE})")
        if data2:
            raise FormulaError(f"data2 holds the matrices of terms such as sar(M), which a {fam.name} model does not take "
                               f"(docs/multinomial.md, divergence {DIVERGENCE})")
        if not center_predictors:
            raise FormulaError(f"center_predictors=False with the {fam.name} family is not supported: brms's bf(..., "
                               "center = FALSE) makes each category's intercept a coefficient of class b, which no frozen case "
                               f"verifies (docs/multinomial.md, divergence {DIVERGENCE})")
        if not isinstance(priors, Priors):
            raise PriorError(f"priors must be an ab.Priors, got {type(priors).__name__}")
        self.family = family
        self.family_spec = fam
        self.priors = priors
        self.center_predictors = True
        self.effective_center_predictors = True
        self.dropped_rows: tuple[int, ...] = ()
        self.warnings: list[str] = []
        self.components: tuple[str, ...] = ()
        self.imputed: tuple[str, ...] = ()
        self.hurdle_category = None
        self.bambi_model = None
        spec0 = parse_multinomial(formula, None, family, categorical)
        priors.validate(spec0, fam)                       # structure now; per-level coverage once the levels are known
        missing = [c for c in spec0.data_columns if c not in data.columns]
        if missing:
            raise FormulaError(f"columns not in data: {missing}")
        df = data[list(spec0.data_columns)].copy().reset_index(drop=True)
        self.spec: MultinomialSpec = parse_multinomial(formula, df, family, categorical)
        priors.validate(self.spec, fam)
        self.categories: tuple[str, ...] = self.spec.categories
        for d in self.spec.mus:                           # a cell of z:h with no fitted rows (brms keeps the zero column too)
            for t in d.pop_terms:
                if t.kind == "interaction" and len(t.categorical) == 2:
                    empty = [lvl for lvl, col in zip(t.levels, pop_term_design(t, df, d.categoricals).T) if not col.any()]
                    if empty:
                        self.warnings.append(f"cell(s) {empty} of {t.name!r} in the formula of {d.response} have no fitted rows: their "
                                             "coefficients are informed by the prior alone (the design column is zero, as in brms).")
        Y, trials = response_counts(self.spec, df)
        if not Y.sum(axis=0).all():
            empty = [c for c, total in zip(self.spec.categories, Y.sum(axis=0)) if not total]
            self.warnings.append(f"the categories {empty} have no counts in the fitted rows: the likelihood alone would send their "
                                 "linear predictors to -infinity, and their priors decide where they stop")
        self.original_labels: dict[str, np.ndarray] = {}
        for cat in self.spec.all_categoricals:            # each declared column as its labels (brms's levels), once
            labels, unused = categorical_labels(f"categorical predictor {cat.name!r}", df[cat.name])
            self.original_labels[cat.name] = labels
            df[cat.name] = labels
            if unused:
                self.warnings.append(f"declared categories {list(unused)} of {cat.name!r} do not occur in the fitted rows: they get no "
                                     "coefficient, and predictions for them are refused.")
        for _, t in self.spec.mu_group_terms:             # labels as text; a derived factor g:h gets its own column g_h
            df[t.column] = group_labels(t.factor, df)
        self.data = df
        self._check_names()
        self.design = self._build_design(Y, trials)
        self.pymc_model = self._build_graph()
        self._target_identity = MappingProxyType(freeze_multinomial_target(self))

    # -- names ------------------------------------------------------------------------------------------------------------
    @property
    def names(self) -> dict[str, str]:
        """PyMC names of the model's roles: ``"linpred"`` the deterministic ``mu`` (the non-reference categories' linear
        predictors, ``(rows, K - 1)`` over ``"mu_dim"``), ``"mean"`` the name ``predict()`` gives the category probabilities
        (``p``; ``expected`` holds ``n p``), ``"y"`` the observed counts (the response's name, over ``"category_dim"``),
        ``"phi"`` the Dirichlet-multinomial's precision and ``"mu:<name>"`` each category's values per row (the deterministic
        named after it). Each category's formula's variables are in ``mu_names``, ``mu_coefficients`` and ``mu_group_names``."""
        out = {"linpred": "mu", "mean": self.family_spec.mean_name, "y": self.spec.response, "category_dim": "category",
               "mu_dim": "mu_dim"}
        if self.family_spec.scale_parameter is not None:
            out["phi"] = "phi"
        out.update({f"mu:{name}": name for name in self.spec.mu_names})
        return out

    @property
    def mu_names(self) -> dict[str, dict[str, str]]:
        """PyMC names of each category's formula, by brms's name of its linear predictor in category order: ``"value"`` its
        values per row (the deterministic ``mum2``), ``"intercept"`` its centred intercept (``mum2_Intercept``, brms's
        ``Intercept_mum2``), ``"raw_intercept"`` the raw-scale intercept (the deterministic ``mum2_b_Intercept``, brms's
        ``b_mum2_Intercept``) and ``"b:<term>"`` each population-level term's coefficient(s) (``mum2_x``; a vector over
        ``mum2_z_dim`` for a categorical or interaction term; brms's ``b_mum2``)."""
        return {d.response: {"value": d.response, "intercept": f"{d.response}_Intercept", "raw_intercept": f"{d.response}_b_Intercept",
                             **{f"b:{t.name}": f"{d.response}_{t.name}" for t in d.pop_terms}}
                for d in self.spec.mus}

    @property
    def mu_coefficients(self) -> dict[str, tuple[tuple[str, str, int | None], ...]]:
        """``(design column, PyMC variable, index into it or None)`` per column of each category's design, in column order
        (``("x", "mum2_x", None)``, ``("z[q]", "mum3_z", 0)``): where brms's ``b_<name>[k]`` lives."""
        return {d.response: tuple((col, self.mu_names[d.response][f"b:{term}"], j) for col, term, j in d.column_terms)
                for d in self.spec.mus}

    @property
    def mu_group_names(self) -> dict[tuple[str, str, str], dict[str, str]]:
        """PyMC roles of every group-level coefficient of every category's formula, keyed ``(category name, factor,
        coefficient)``: ``sd`` (brms's ``sd_<id>``), ``offset`` (``z_<id>``) and ``group_effect`` (``r_<id>_<name>_1``), Bambi's
        names under the category's prefix (``mum2_1|g_sigma``, ``mum2_1|g_offset``, ``mum2_1|g``), and ``coordinate``, the
        factor's level dimension (``g__factor_dim``, shared by every category)."""
        from autobayes.formula import bambi_group_name

        out = {}
        for name, t in self.spec.mu_group_terms:
            for coef in t.coefs:
                base = f"{name}_{bambi_group_name(t.factor, coef)}"
                out[(name, t.factor, coef)] = {"sd": f"{base}_sigma", "offset": f"{base}_offset", "group_effect": base,
                                               "coordinate": f"{factor_column(t.factor)}__factor_dim"}
        return out

    def _check_names(self) -> None:
        """Refuse generated names that would alias one another (a variable, a coordinate or the observed response), and
        retained names ending in ``_offset`` (the sampler drops those from the posterior, as Bambi does)."""
        n = self.names
        variables = [n["linpred"], n["y"], *self.spec.mu_names, *([n["phi"]] if "phi" in n else [])]
        retained = [v for roles in self.mu_names.values() for k, v in roles.items() if k != "value"]
        retained += [roles["group_effect"] for roles in self.mu_group_names.values()]
        variables += retained + [roles[k] for roles in self.mu_group_names.values() for k in ("sd", "offset")]
        coords = ["__obs__", n["category_dim"], n["mu_dim"], *dict.fromkeys(r["coordinate"] for r in self.mu_group_names.values())]
        coords += [f"{self.mu_names[d.response][f'b:{t.name}']}_dim" for d in self.spec.mus for t in d.pop_terms if t.is_vector]
        names = variables + coords
        duplicates = sorted({x for x in names if names.count(x) > 1})
        if duplicates:
            raise FormulaError(f"generated variable/coordinate name collision: {duplicates}; rename a column")
        if any(x.endswith("_offset") for x in retained):
            raise FormulaError("generated name collision: the suffix '_offset' is reserved for the standardised group effects; rename "
                               "the column")

    # -- design and graph -----------------------------------------------------------------------------------------------------
    def _build_design(self, Y: np.ndarray, trials: np.ndarray) -> MultinomialDesign:
        s, df = self.spec, self.data
        n = len(df)
        mus = {}
        for d in s.mus:
            X = mu_design(d, df)
            means = X.mean(axis=0)
            group_blocks, start = {}, 0
            for t in d.group_terms:
                levels = d.levels_by_group[t.factor]
                labels = df[t.column].to_numpy()
                indicator = np.column_stack([labels == level for level in levels]).astype(float)
                group_blocks[(t.factor, "Intercept")] = GroupBlock(levels, indicator, indicator.argmax(1), slice(start, start + len(levels)))
                start += len(levels)
            Z = np.concatenate([b.Z for b in group_blocks.values()], axis=1) if group_blocks else np.zeros((n, 0))
            mus[d.response] = MuDesign(name=d.response, columns=d.columns, X_raw=X, X_model=X - means, x_means=means,
                                       group_blocks=MappingProxyType(group_blocks), Z=Z)
        return MultinomialDesign(y=Y, X_raw=np.zeros((n, 0)), X_model=np.zeros((n, 0)), x_means=np.zeros(0), predictors=(),
                                 Z=np.zeros((n, 0)), group_blocks=MappingProxyType({}), centered=True, trials=trials,
                                 categories=s.categories, mus=MappingProxyType(mus))

    def _build_graph(self):
        """The PyMC graph: each category's centred intercept, coefficients, raw-scale intercept, group effects and values per
        row, the linear predictors ``mu`` side by side, ``phi`` (Dirichlet-multinomial) and the likelihood on the counts."""
        import pymc as pm
        import pytensor.tensor as pt

        from autobayes.families import multinomial as fm
        from autobayes.nonlinear import _location_rv, _positive_rv

        s, design, fam, n = self.spec, self.design, self.family_spec, self.design.n
        names = self.names
        coords = {"__obs__": np.arange(n), names["category_dim"]: list(s.categories), names["mu_dim"]: list(s.categories[1:])}
        with pm.Model(coords=coords) as pmm:
            etas = []
            for d in s.mus:
                name, md, roles = d.response, design.mus[d.response], self.mu_names[d.response]
                sub = category_priors(self.priors, s, name)
                per_column = sub.coefficient_priors(d)
                intercept = _location_rv(roles["intercept"], sub.intercept)
                eta, raw, col = intercept * np.ones(n), intercept, 0
                for t in d.pop_terms:
                    var, k = roles[f"b:{t.name}"], len(t.columns)
                    block = np.asarray(md.X_model[:, col:col + k], dtype=np.float64)
                    means = np.asarray(md.x_means[col:col + k], dtype=np.float64)
                    if t.is_vector:
                        pmm.add_coord(f"{var}_dim", list(t.levels))
                        coefficient = _location_rv(var, [per_column[c] for c in t.columns], dims=f"{var}_dim")
                        eta = eta + pm.math.dot(block, coefficient)
                        raw = raw - pt.dot(means, coefficient)
                    else:
                        coefficient = _location_rv(var, per_column[t.columns[0]])
                        eta = eta + coefficient * block[:, 0]
                        raw = raw - coefficient * means[0]
                    col += k
                pm.Deterministic(roles["raw_intercept"], raw)
                sd_priors = sub.sd_priors(d)
                for t in d.group_terms:
                    r = self.mu_group_names[(name, t.factor, "Intercept")]
                    if r["coordinate"] not in pmm.coords:
                        pmm.add_coord(r["coordinate"], list(d.levels_by_group[t.factor]))
                    sd = _positive_rv(r["sd"], sd_priors[(t.factor, "Intercept")])
                    z = pm.Normal(r["offset"], np.float64(0.0), np.float64(1.0), dims=r["coordinate"])
                    u = pm.Deterministic(r["group_effect"], sd * z, dims=r["coordinate"])
                    eta = eta + u[np.asarray(md.group_blocks[(t.factor, "Intercept")].group_index, dtype=np.int64)]
                etas.append(pm.Deterministic(roles["value"], eta, dims="__obs__"))
            mu = pm.Deterministic(names["linpred"], pt.stack(etas, axis=1), dims=("__obs__", names["mu_dim"]))
            full = pt.concatenate([pt.zeros((n, 1), dtype="float64"), mu], axis=1)
            trials = np.asarray(design.trials, dtype=np.int64)
            dims = ("__obs__", names["category_dim"])
            if fam.scale_parameter is not None:
                phi = _positive_rv(names["phi"], self.priors.phi)
                pm.CustomDist(names["y"], full, phi, trials, logp=fm.dirichlet_multinomial_logit_logp,
                              random=fm.dirichlet_multinomial_logit_random, signature="(k),(),()->(k)", dtype="int64",
                              observed=np.asarray(design.y, dtype=np.int64), dims=dims)
            else:
                pm.CustomDist(names["y"], full, trials, logp=fm.multinomial_logit_logp, random=fm.multinomial_logit_random,
                              signature="(k),()->(k)", dtype="int64", observed=np.asarray(design.y, dtype=np.int64), dims=dims)
        return pmm

    # -- points ---------------------------------------------------------------------------------------------------------------
    def value_point(self, *, dpars=None, phi=None, **unexpected) -> dict[str, np.ndarray]:
        """A point in PyMC's sampled space from natural-scale values: ``dpars={name: {"intercept": ..., "b": ..., "sd": ...,
        "z": ...}}`` for every non-reference category, ``intercept`` its centred intercept (brms's ``Intercept_<name>``),
        ``b`` its coefficients in its design's column order (``design.mus[name].columns``; brms's ``b_<name>``), ``sd`` and
        ``z`` its group-level terms' SDs and standardised effects by factor (a bare value for a single factor), and the
        Dirichlet-multinomial's ``phi``. Scales are stored on the log scale."""
        if unexpected:
            raise ValueError(f"a {self.family} model's point takes dpars={{...}}" + (" and phi" if "phi" in self.names else "")
                             + f"; unexpected {sorted(unexpected)}")
        s = self.spec
        if not isinstance(dpars, Mapping) or set(dpars) != set(s.mu_names):
            raise ValueError(f"dpars must be a mapping over exactly the categories' linear predictors {list(s.mu_names)}")
        point: dict[str, np.ndarray] = {}

        def scalar(value, label, positive=False):
            arr = np.asarray(value, dtype=float)
            if arr.shape != () or not np.isfinite(arr) or (positive and arr <= 0):
                raise ValueError(f"{label} must be a finite {'positive ' if positive else ''}scalar")
            return np.asarray(np.log(arr)) if positive else arr

        for d in s.mus:
            name, entry = d.response, dpars[d.response]
            if not isinstance(entry, Mapping) or set(entry) - {"intercept", "b", "sd", "z"}:
                raise ValueError(f"dpars[{name!r}] is a mapping with the keys 'intercept', 'b', and 'sd', 'z' for its group-level terms")
            md, roles = self.design.mus[name], self.mu_names[name]
            point[roles["intercept"]] = scalar(entry.get("intercept"), f"the intercept of {name!r}")
            b = np.atleast_1d(np.asarray(entry.get("b", ()), dtype=float))
            if b.shape != (len(md.columns),) or not np.isfinite(b).all():
                raise ValueError(f"dpars[{name!r}]['b'] must have {len(md.columns)} finite entries, one per design column {list(md.columns)}")
            i = 0
            for t in d.pop_terms:                         # b in design-column order: one entry per term column
                k = len(t.columns)
                point[roles[f"b:{t.name}"]] = np.array(b[i:i + k]) if t.is_vector else np.asarray(b[i])
                i += k
            if not d.group_terms and (entry.get("sd") is not None or entry.get("z") is not None):
                raise ValueError(f"'{d.formula}' has no group-level term; sd and z do not apply to {name!r}")
            sds, zs = (_by_factor(d, entry.get(k), k, name) for k in ("sd", "z")) if d.group_terms else ({}, {})
            for t in d.group_terms:
                r, block = self.mu_group_names[(name, t.factor, "Intercept")], md.group_blocks[(t.factor, "Intercept")]
                z = np.asarray(zs[t.factor], dtype=float)
                if z.shape != (block.J,) or not np.isfinite(z).all():
                    raise ValueError(f"z of {t.factor!r} in {name!r} must have {block.J} finite entries")
                point[f"{r['sd']}_log__"] = scalar(sds[t.factor], f"sd of {t.factor!r} in {name!r}", positive=True)
                point[r["offset"]] = z
        if (phi is not None) != ("phi" in self.names):
            raise ValueError(self.family_spec.scale_rule)
        if phi is not None:
            point["phi_log__"] = scalar(phi, "phi", positive=True)
        expected = {v.name for v in self.pymc_model.value_vars}
        if set(point) != expected:
            raise RuntimeError(f"value names {sorted(point)} do not match PyMC's {sorted(expected)}")
        return point

    # -- description ----------------------------------------------------------------------------------------------------------
    @property
    def default_prior_labels(self) -> list[str]:
        """The priors that came from ``ab.default_priors`` (brms's defaults), labelled as ``prior_labels`` labels them."""
        return [label for label, p in prior_labels(self.priors, self.spec, self.family_spec) if getattr(p, "source", "user") == "default"]

    def describe(self) -> str:
        return describe_multinomial(self)

    def sample_posterior(self, *, inference_method: str = "pymc", draws: int = 1000, tune: int = 1000, chains=None, cores=None,
                         random_seed=None, init: str = "auto", n_init: int = 50000, discard_tuned_samples: bool = True,
                         omit_offsets: bool = True, **kwargs):
        """Sample ``pymc_model`` with PyMC's ``sample`` (``nuts_sampler`` = ``inference_method``, ``"pymc"`` or ``"numpyro"``), as
        ``NonlinearModel.sample_posterior`` does: the per-row deterministics (the categories' values and ``mu``) are not kept,
        and the standardised group effects (``_offset``) are dropped unless ``omit_offsets=False``."""
        import pymc as pm
        from pymc.util import get_default_varnames

        per_row = {self.names["linpred"], *self.spec.mu_names}
        var_names = [v.name for v in get_default_varnames(self.pymc_model.unobserved_value_vars, include_transformed=False)
                     if v.name not in per_row]
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, random_seed=random_seed, init=init, n_init=n_init,
                          discard_tuned_samples=discard_tuned_samples, var_names=var_names, nuts_sampler=inference_method,
                          model=self.pymc_model, **kwargs)
        if omit_offsets and hasattr(idata, "posterior"):
            idata.posterior = idata.posterior.drop_vars([v for v in idata.posterior.data_vars if v.endswith("_offset")])
        return idata


def _by_factor(d: ModelSpec, value, label: str, name: str) -> dict[str, object]:
    """``value_point``'s ``sd`` / ``z`` of one category's formula, ``{factor: value}``: a mapping over exactly its factors, or a
    bare value for a single factor."""
    factors = [t.factor for t in d.group_terms]
    if value is None:
        raise ValueError(f"{label} is required for the group-level terms of {name!r}: {[t.canonical for t in d.group_terms]}")
    if isinstance(value, Mapping):
        if set(value) != set(factors):
            raise ValueError(f"{label} of {name!r} must be a mapping over exactly its grouping factors {factors}")
        return dict(value)
    if len(factors) != 1:
        raise ValueError(f"{label} of {name!r} must be a mapping over exactly its grouping factors {factors}")
    return {factors[0]: value}


def multinomial(formula, data: pd.DataFrame, priors: Priors, *, center_predictors: bool = True,
                categorical: Mapping[str, object] | None = None) -> MultinomialModel:
    """brms's ``multinomial()`` (F01a, docs/multinomial.md): ``cbind(y1, ..., yK) | trials(n) ~ terms`` with the count columns
    as the categories (the first the reference) and one linear predictor ``mu<k>`` per non-reference category, each with its
    own centred intercept, coefficients and group-level terms; ``y_i ~ Multinomial(n_i, softmax(0, mu_i))``. A list of
    formulas gives a category a formula of its own, ``["cbind(m1, m2, m3) | trials(size) ~ x + (1 | g)", "mum3 ~ z"]``.
    ``priors``: ``Priors(intercept=..., b=..., sd=...)`` for the categories on the response formula, ``dpars={"mum3":
    {...}}`` for a category's own; ``ab.default_priors(formula, data, family="multinomial")`` returns brms's defaults."""
    return MultinomialModel(formula, data, priors, family="multinomial", center_predictors=center_predictors, categorical=categorical)


def dirichlet_multinomial(formula, data: pd.DataFrame, priors: Priors, *, center_predictors: bool = True,
                          categorical: Mapping[str, object] | None = None) -> MultinomialModel:
    """brms's ``dirichlet_multinomial()`` (F01a): the multinomial model's formulas and categories with ``y_i ~
    DirichletMultinomial(n_i, phi softmax(0, mu_i))``, ``phi`` the precision (``Priors(phi=...)``, brms's default ``gamma(0.01,
    0.01)``); the multinomial is its limit as ``phi`` grows."""
    return MultinomialModel(formula, data, priors, family="dirichlet_multinomial", center_predictors=center_predictors,
                            categorical=categorical)


# -- describe() ----------------------------------------------------------------------------------------------------------------
def mean_lines(model) -> list[str]:
    """``describe()``'s lines relating the probabilities and expected counts to the linear predictors."""
    s = model.spec
    others = ", ".join(f"{c} ({m})" for c, m in zip(s.categories[1:], s.mu_names))
    return [f"  The reference category {s.reference!r} has the linear predictor 0; each other category k = {others} has its own,",
            "  mu_k,i, on the log-odds scale of k against the reference. p_i = softmax(0, mu_2,i, ..., mu_K,i), the category",
            f"  probabilities, and E[y_k,i] = {s.trials}_i p_k,i (brms's posterior_epred)."]


def likelihood_lines(model) -> list[str]:
    """``describe()``'s likelihood paragraph (and ``phi``'s prior line)."""
    s, fam = model.spec, model.family_spec
    lines = []
    if fam.scale_parameter is not None:
        lines += [f"  phi ~ {describe_prior(model.priors.phi, lb=0)}   (the precision; sampled as log(phi))"]
    lines += ["", "Likelihood"]
    coefficient = f"log C(y_i) = log({s.trials}_i! / prod_k y_k,i!), written as sum_k log C(y_k,i + ... + y_K,i, y_k,i)"
    if fam.scale_parameter is None:
        lines += [f"  y_i ~ Multinomial({s.trials}_i, p_i): log p(y_i) = log C(y_i) + sum_k y_k,i log p_k,i (brms's",
                  f"  multinomial_logit2_lpmf), {coefficient};",
                  "  log p_i is the log-softmax with the largest entry factored out."]
    else:
        lines += [f"  y_i ~ DirichletMultinomial({s.trials}_i, phi p_i): log p(y_i) = log C(y_i) + lgamma(phi) - lgamma({s.trials}_i + phi)",
                  "  + sum_k [lgamma(y_k,i + phi p_k,i) - lgamma(phi p_k,i)] (brms's dirichlet_multinomial_logit2_lpmf),",
                  f"  {coefficient}; each lgamma difference is a log rising factorial through Stirling's remainder",
                  "  (lgamma(y) + log(phi p_k,i) where phi p_k,i < exp(-40)). phi is the precision (the multinomial as phi -> infinity)."]
    lines += [f"  Each row's counts sum to {s.trials}_i (checked); rows of zero trials contribute 0, as in brms."]
    return lines


def describe_multinomial(model: MultinomialModel) -> str:
    """``describe()`` of a count-matrix model: the formulas, data, categories and levels; each category's linear predictor
    (centred), group effects and priors under brms's names; the likelihood; the sampled variables and the deterministics."""
    from autobayes.default_priors import BRMS_VERSION, brms_coefficient_names
    from autobayes.formula import sd_tag
    from autobayes.nonlinear import _column_text

    s, d, pr, fam = model.spec, model.design, model.priors, model.family_spec
    lines = [f"AutoBayes {fam.tag} model: {len(s.categories)} categories ({', '.join(s.categories)}; reference {s.reference!r}), "
             f"{len(s.mus)} linear predictor(s) ({', '.join(s.mu_names)})",
             f"  formula   : {s.formula}"]
    if s.canonical_formula != s.formula:
        lines += [f"  expanded  : {s.canonical_formula}"]
    for name in s.own:
        lines += [f"  {name:<10}: {s.mu(name).formula}   (the category's own formula)"]
    totals = d.y.sum(axis=0)
    lines += [f"  family    : {model.family}, {model.link} link (softmax over the categories)",
              f"  data      : n = {d.n} rows, {int(d.trials.sum())} counts in all ("
              + ", ".join(f"{c} {int(t)}" for c, t in zip(s.categories, totals)) + f"); trials {s.trials!r}"]
    for factor in s.factors:
        levels = next(md.group_blocks[k].levels for md in d.mus.values() for k in md.group_blocks if k[0] == factor)
        lines += [f"  {'levels ' + factor:<10}: {', '.join(f'{j}={lvl}' for j, lvl in enumerate(levels))}"]
    for cat in s.all_categoricals:
        owners = [ds.response for ds in s.mus if cat.name in ds.categorical_names]
        labels = model.data[cat.name].astype(str).to_numpy()
        lines += [f"  {cat.name:<10}: categorical in the formula(s) of {', '.join(owners)}, reference {cat.reference!r}; levels "
                  f"(sorted) {', '.join(cat.levels)}; proportions among fitted rows "
                  + ", ".join(f"{lvl} {np.mean(labels == lvl):.3f}" for lvl in cat.levels)]
    lines += ["", "Categories"] + fam.mean_lines(model)
    for ds in s.mus:
        name, md = ds.response, d.mus[ds.response]
        names = brms_coefficient_names(ds)
        parts = [f"Intercept_{name}"]
        for t in ds.pop_terms:
            for col, level in zip(t.columns, t.levels or (None,)):
                parts.append(f"({_column_text(t, level)} - mean) * b_{name}[{names[col]}]")
        parts += [f"u_{name}_{t.factor}[{t.factor}_i]" for t in ds.group_terms]
        lines += ["", f"Linear predictor {name} (category {s.categories[1 + s.mu_names.index(name)]!r}): {ds.formula}",
                  f"  {name}_i = {' + '.join(parts)}"]
        if md.columns:
            lines += [f"  The design is centred on its fitted-row means ({', '.join(f'{c} {m:.6g}' for c, m in zip(md.columns, md.x_means))}): "
                      f"Intercept_{name} is the category's log-odds at the means (brms's Intercept_{name}); the raw-scale intercept",
                      f"  b_{name}_Intercept = Intercept_{name} - sum_p mean_p b_{name}[p] is the deterministic {model.mu_names[name]['raw_intercept']}."]
        for t in ds.group_terms:
            block = md.group_blocks[(t.factor, "Intercept")]
            lines += [f"  u_{name}_{t.factor}[j] = sd_{name}_{t.factor} * z_{name}_{t.factor}[j], z ~ Normal(0, 1), j = 0..{block.J - 1}: "
                      "a group of its own (brms numbers each category's terms separately)"]
    lines += ["", "Priors"]
    for ds in s.mus:
        name, sub = ds.response, category_priors(pr, s, ds.response)
        names = brms_coefficient_names(ds)
        source = (f"dpars[{name!r}]" if name in (pr.dpars or {}) else "the top-level intercept, b and sd")
        lines += [f"  Intercept_{name} ~ {describe_prior(sub.intercept)}   (from {source})"]
        lines += [f"  b_{name}[{names[col]}] ~ {describe_prior(p)}" for col, p in sub.coefficient_priors(ds).items()]
        lines += [f"  sd_{name}_{sd_tag(g, c)} ~ {describe_prior(p, lb=0)}" for (g, c), p in sub.sd_priors(ds).items()]
    lines += fam.likelihood_lines(model)
    if model.default_prior_labels:
        lines += ["", "Default priors (ab.default_priors; docs/default-priors.md, docs/multinomial.md)",
                  f"  {', '.join(model.default_prior_labels)}: brms {BRMS_VERSION}'s default_prior() for this model: student_t(3, 0, 2.5) on",
                  "  each category's intercept and group SDs, flat coefficients (improper: prior_predictive() is refused)"
                  + (", gamma(0.01, 0.01) on phi" if fam.scale_parameter else "") + "."]
    for w in model.warnings:
        lines += ["", "WARNING: " + w]
    lines += ["", "PyMC variables sampled (value space)"]
    roles = {}
    for ds in s.mus:
        name, mroles, names = ds.response, model.mu_names[ds.response], brms_coefficient_names(ds)
        roles[mroles["intercept"]] = f"Intercept_{name} (centred)"
        for t in ds.pop_terms:
            var = mroles[f"b:{t.name}"]
            roles[var] = (f"b_{name}[{', '.join(names[c] for c in t.columns)}] (vector; dims {var}_dim, coordinates {', '.join(t.levels)})"
                          if t.is_vector else f"b_{name}[{names[t.columns[0]]}]")
    for (name, g, _), r in model.mu_group_names.items():
        roles[f"{r['sd']}_log__"] = f"log(sd_{name}_{g})"
        roles[r["offset"]] = f"z_{name}_{g}"
    if fam.scale_parameter is not None:
        roles["phi_log__"] = "log(phi)"
    for v in model.pymc_model.value_vars:
        lines.append(f"  {v.name:<24} {roles.get(v.name, v.name)}")
    lines.append("Deterministic: " + ", ".join(
        [f"{model.mu_names[name]['raw_intercept']} = b_{name}_Intercept" for name in s.mu_names]
        + [f"{r['group_effect']} = u_{name}_{g}" for (name, g, _), r in model.mu_group_names.items()]
        + [f"{name} = {name}_i per row" for name in s.mu_names]
        + [f"mu = ({', '.join(s.mu_names)}) per row; p and n p are computed by predict()"]))
    return "\n".join(lines)


# -- predictions ----------------------------------------------------------------------------------------------------------------
def check_multinomial_newdata(model: MultinomialModel, newdata: pd.DataFrame, *, include_group_effects: bool = True,
                              need_trials: bool = False) -> pd.DataFrame:
    """The rows of ``newdata`` a count-matrix model is predicted on: every category's columns (and the grouping factors'
    unless group effects are excluded, the trials for the expected counts and draws), numeric where they are numbers, the
    transforms in their domain, the categorical labels fitted ones; each grouping factor's labels in its column. The counts
    themselves are not needed."""
    s = model.spec
    operands = list(s.group_columns) if include_group_effects else []
    cols = list(dict.fromkeys([*s.model_columns, *operands, *([s.trials] if need_trials else [])]))
    missing = [c for c in cols if c not in newdata.columns]
    if missing:
        hint = f" (the trials {s.trials!r} give the expected counts and the draws)" if s.trials in missing else ""
        raise FormulaError(f"newdata lacks columns {missing}{hint}")
    df = newdata[cols].copy()
    if df.isna().any().any():
        raise FormulaError("newdata contains missing values")
    categorical = {c.name for c in s.all_categoricals}
    for c in (c for c in s.model_columns if c not in categorical):
        reject_complex(f"newdata column {c!r}", df[c])
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise FormulaError(f"newdata column {c!r} must be numeric")
    if need_trials:
        df[s.trials] = check_counts(f"newdata trials {s.trials!r}", df[s.trials])
    for d in s.mus:
        for t in d.transform_terms:
            t.transform.values(df, where="newdata column")
    for cat in s.all_categoricals:
        labels, _ = categorical_labels(f"newdata column {cat.name!r}", df[cat.name])
        unseen = sorted(set(labels.tolist()) - set(cat.levels))
        if unseen:
            raise FormulaError(
                f"newdata has levels of the categorical predictor {cat.name!r} that were not in the fitted rows: {unseen}; fitted "
                f"levels {list(cat.levels)} (reference {cat.reference!r}). A categorical predictor has a coefficient only for its "
                "fitted levels; it is not a grouping factor, so new_groups does not apply to it.")
        df[cat.name] = labels
    for _, t in (s.mu_group_terms if include_group_effects else ()):
        df[t.column] = group_labels(t.factor, df)
    return df.reset_index(drop=True)


def mu_draws(model: MultinomialModel, post, frame: pd.DataFrame, name: str, *, include_group_effects: bool,
             policies: dict[str, str], rng) -> np.ndarray:
    """(chain, draw, row) draws of category ``name``'s linear predictor on ``frame``: its centred intercept plus its design,
    centred on the fitted means, times its coefficients, plus each group-level term's effect at the row's level (an unseen
    level by its factor's policy: ``population`` a fresh ``Normal(0, sd)`` per draw, ``resample_observed`` one observed
    level's effect per draw and label, as ``predict()`` draws them)."""
    d, md, roles = model.spec.mu(name), model.design.mus[name], model.mu_names[name]
    X = mu_design(d, frame) - md.x_means
    eta = np.broadcast_to(post[roles["intercept"]].to_numpy()[..., None], post[roles["intercept"]].shape + (len(frame),)).copy()
    if md.columns:
        B = np.stack([post[var].to_numpy() if j is None else post[var].to_numpy()[..., j] for _, var, j in model.mu_coefficients[name]],
                     axis=-1)
        eta += np.einsum("cdk,nk->cdn", B, X)
    if not include_group_effects:
        return eta
    for t in d.group_terms:
        r, block = model.mu_group_names[(name, t.factor, "Intercept")], md.group_blocks[(t.factor, "Intercept")]
        levels, labels = list(block.levels), frame[t.column].to_numpy()
        u = post[r["group_effect"]].transpose("chain", "draw", r["coordinate"]).to_numpy()
        index = {lvl: i for i, lvl in enumerate(levels)}
        position = np.array([index.get(lvl, -1) for lvl in labels], dtype=np.int64)
        seen = position >= 0
        eta[..., seen] += u[..., position[seen]]
        sd = post[r["sd"]].to_numpy()
        for lvl in sorted(set(labels[~seen].tolist())):
            if policies[t.factor] == "population":
                effect = sd * rng.normal(0.0, 1.0, size=sd.shape)
            else:
                chosen = rng.integers(0, len(levels), size=sd.shape)
                effect = np.take_along_axis(u, chosen[..., None], axis=-1)[..., 0]
            eta[..., labels == lvl] += effect[..., None]
    return eta


def predict_multinomial(model: MultinomialModel, fit_result, *, newdata: pd.DataFrame | None = None, kind: str = "response",
                        new_groups="error", seed: int | None = None, include_group_effects: bool = True, dpar: str | None = None,
                        nlpar: str | None = None):
    """``predict()`` of a count-matrix model, by hand from the posterior: ``kind="linear"`` the non-reference categories'
    linear predictors (brms's ``posterior_linpred``, ``eta`` over ``(chain, draw, __obs__, category)``), ``kind="mean"`` the
    category probabilities ``p`` and the expected counts ``expected = n p`` over every category (brms's ``posterior_epred``),
    ``kind="response"`` count draws ``y`` too (brms's ``posterior_predict``). Group-level policies (``new_groups``,
    ``include_group_effects``) are ``predict()``'s, by grouping factor across the categories' formulas; the random numbers of
    unseen levels follow the categories in order, then their group-level terms, then the sorted unseen labels (a factor in two
    categories' formulas draws each one's effects independently, as brms's two group ids do), then the response draws."""
    import xarray as xr

    from autobayes.families import family_of
    from autobayes.identity import check_fit_binding
    from autobayes.nonlinear import _policies

    fam = family_of(model)
    if kind not in fam.prediction_kinds:
        raise ValueError(fam.kind_error)
    if dpar is not None or nlpar is not None:
        raise ValueError(f"a {model.family} model's predictions are per category (kind='linear' gives every category's linear "
                         f"predictor); dpar and nlpar do not apply (got {dpar or nlpar!r})")
    if not isinstance(include_group_effects, (bool, np.bool_)):
        raise ValueError("include_group_effects must be boolean")
    if not include_group_effects and new_groups != "error":
        raise ValueError("new_groups must stay at its default when group effects are excluded")
    s = model.spec
    factors = s.factors
    policies = _policies(factors, new_groups)
    binding = check_fit_binding(model, fit_result, allow_unverified=True)
    idata = getattr(fit_result, "idata", fit_result)
    need_trials = kind in ("mean", "response")
    df = None if newdata is None else check_multinomial_newdata(model, newdata, include_group_effects=include_group_effects,
                                                                need_trials=need_trials)
    frame = model.data if df is None else df
    new_levels: dict[str, list] = {}
    if include_group_effects:
        for name, t in s.mu_group_terms:
            fitted = set(model.design.mus[name].group_blocks[(t.factor, "Intercept")].levels)
            new_levels[t.factor] = sorted(set(new_levels.get(t.factor, [])) | (set(frame[t.column]) - fitted))
    for g, unseen in new_levels.items():
        if unseen and policies[g] == "error":
            raise FormulaError(
                f"newdata has levels of {g!r} not seen in training: {unseen}. Pass new_groups='population' to draw their effects "
                "from Normal(0, sd), or new_groups='resample_observed' for brms's default 'uncertainty' behaviour; an exact-key "
                "mapping can set the policy separately for each factor.")
    post = idata.posterior
    rng = np.random.default_rng(seed)
    eta = np.stack([mu_draws(model, post, frame, name, include_group_effects=include_group_effects, policies=policies, rng=rng)
                    for name in s.mu_names], axis=-1)
    if not np.isfinite(eta).all():
        raise ValueError("the categories' linear predictors contain nonfinite draws; inspect the posterior draws")
    dims = ("chain", "draw", "__obs__", "category")
    base = {"chain": post["chain"].values, "draw": post["draw"].values, "__obs__": np.arange(len(frame))}
    if kind == "linear":
        ds = xr.Dataset({"eta": (dims, eta)}, coords={**base, "category": list(s.categories[1:])})
    else:
        p = fam.inverse_link(eta)
        trials = frame[s.trials].to_numpy(dtype=np.int64)
        out = {fam.mean_name: (dims, p), "expected": (dims, trials[:, None] * p)}
        if kind == "response":
            out["y"] = (dims, fam.draw_response(model, post, eta, p, frame, rng))
        ds = xr.Dataset(out, coords={**base, "category": list(s.categories)})
    ds.attrs["include_group_effects"] = int(bool(include_group_effects))
    if not factors or not include_group_effects:
        ds.attrs.update(new_groups="not_applicable", new_levels=[])
    elif len(factors) == 1:
        ds.attrs["new_groups"] = "in_sample" if df is None else policies[factors[0]]
        ds.attrs["new_levels"] = new_levels.get(factors[0], [])
    else:
        ds.attrs["new_groups"] = "in_sample" if df is None else "per_factor"
        ds.attrs["new_groups_by_factor_json"] = json.dumps(policies)
        ds.attrs["new_levels_by_factor_json"] = json.dumps(new_levels)
    ds.attrs["categories_json"] = json.dumps(list(s.categories))
    ds.attrs["fit_binding_status"] = binding
    if kind == "linear":
        ds.attrs.update(prediction_scale="linear", link=model.link)
    return ds


# -- brms's default priors and prior table ----------------------------------------------------------------------------------
def multinomial_default_priors(formula, data: pd.DataFrame, *, family: str, categorical=None, center_predictors: bool = True,
                               imputed=()) -> Priors:
    """brms's ``default_prior()`` of a count-matrix model (brms 2.23, checked against R on the frozen cases and probes):
    ``student_t(3, 0, 2.5)`` on each category's intercept and on its group SDs (``HalfStudentT(3, 2.5)``; brms reads no
    response through the logit link), flat coefficients, and ``gamma(0.01, 0.01)`` on the Dirichlet-multinomial's ``phi``.
    The categories on the response formula take them at the top level, a category with a formula of its own under
    ``dpars``. Every prior is marked ``source="default"``."""
    from autobayes.default_priors import DF, _default
    from autobayes.families import get_family
    from autobayes.priors import Flat, HalfStudentT, StudentT

    fam = get_family(family)
    if not center_predictors:
        raise FormulaError(f"center_predictors=False with the {fam.name} family is not supported (docs/multinomial.md, "
                           f"divergence {DIVERGENCE})")
    if imputed:
        raise FormulaError(f"a {fam.name} model in a multivariate formula is not supported (divergence {DIVERGENCE})")
    spec = parse_multinomial(formula, data, family, categorical)

    def defaults(d: ModelSpec) -> dict:
        return {"intercept": _default(StudentT(DF, 0.0, 2.5)), **({"b": _default(Flat())} if d.pop_terms else {}),
                **({"sd": {t.factor: _default(HalfStudentT(DF, 2.5)) for t in d.group_terms}} if d.group_terms else {})}

    shared = [d for d in spec.mus if d.response not in spec.own]
    top = defaults(shared[0]) if shared else {}
    priors = Priors(**top, dpars={name: defaults(spec.mu(name)) for name in spec.own} or None,
                    **({"phi": _default(fam.default_scale_prior(2.5))} if fam.scale_parameter else {}))
    priors.validate(spec, fam)
    return priors


def multinomial_prior_table(priors: Priors, spec: MultinomialSpec, fam) -> list[dict[str, str]]:
    """A count-matrix model's priors in brms's table form (``default_priors.prior_table``'s rows and order): each category's
    class ``Intercept``, ``b`` and ``sd`` rows under its ``dpar``, and ``phi``'s row. A prior shared by a class sits on the class
    row, as brms writes its defaults."""
    from autobayes.default_priors import COLUMNS, brms_coefficient_names, brms_rename
    from autobayes.priors import brms_prior

    check_multinomial_priors(priors, spec, fam)
    rows: list[dict[str, str]] = []

    def add(prior, cls, *, coef="", group="", dpar="", lb="", source=None):
        text = brms_prior(prior) if isinstance(prior, Prior) else ""
        src = source if source is not None else (prior.source if isinstance(prior, Prior) else "default")
        rows.append(dict(zip(COLUMNS, (text, cls, coef, group, "", dpar, "", lb, "", "", src))))

    def same(ps):
        return all(p == ps[0] for p in ps)

    for d in spec.mus:
        name, sub = d.response, category_priors(priors, spec, d.response)
        names = brms_coefficient_names(d)
        add(sub.intercept, "Intercept", dpar=name)
        coefs = [(names[col], p) for col, p in sub.coefficient_priors(d).items()]
        if coefs:
            ps = [p for _, p in coefs]
            add(ps[0] if same(ps) else None, "b", dpar=name)
            for coef, p in coefs:
                add(None if same(ps) else p, "b", coef=coef, dpar=name, source=p.source if same(ps) else None)
        sd = sub.sd_priors(d)
        if sd:
            ps = list(sd.values())
            add(ps[0] if same(ps) else None, "sd", dpar=name, lb="0")
            for t in d.group_terms:
                p = sd[(t.factor, "Intercept")]
                add(None if same(ps) else p, "sd", group=t.factor, dpar=name, source=p.source if same(ps) else None)
                add(None, "sd", coef=brms_rename("Intercept"), group=t.factor, dpar=name, source=p.source)
    if fam.scale_parameter is not None:
        add(priors.phi, "phi", lb="0")
    rows.sort(key=lambda r: (r["resp"], r["dpar"], r["nlpar"], r["class"], r["group"], r["coef"], r["tag"]))
    return rows


# -- the target identity, fit records and the prior predictive ---------------------------------------------------------------
def freeze_multinomial_target(model: MultinomialModel) -> dict[str, str]:
    """The target identity of a count-matrix model (``identity.freeze_target``'s schema and encoding): the family and link,
    the response's categories, counts and trials, each category's canonical formula, design, means, coding and group blocks,
    and every prior. Only the digest is retained."""
    from autobayes.identity import SCHEMA, _canonical, _prior

    s, d, fam = model.spec, model.design, model.family_spec
    mus = []
    for ds in s.mus:
        md = d.mus[ds.response]
        mus.append({"name": ds.response, "formula": ds.canonical_formula, "own": ds.response in s.own, "columns": list(md.columns),
                    "X": md.X_raw, "means": md.x_means,
                    "categorical": [{"name": c.name, "reference": c.reference, "levels": c.levels, "coefficients": c.coefficients,
                                     "full_rank": c.full_rank} for c in ds.categoricals],
                    "groups": [{"factor": g, "coef": c, "levels": b.levels, "index": b.group_index, "noncentered": True}
                               for (g, c), b in md.group_blocks.items()]})
    payload = {"schema": SCHEMA, "family": model.family, "link": model.link,
               "multinomial": {"response": s.response, "formula": s.canonical_formula, "categories": list(s.categories),
                               "trials": s.trials, "y": d.y, "n": d.trials, "mus": mus},
               "priors": {label: _prior(p) for label, p in prior_labels(model.priors, s, fam)}}
    encoded = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return {"schema": SCHEMA, "sha256": hashlib.sha256(encoded).hexdigest()}


def multinomial_summary_rows(model: MultinomialModel, idata) -> dict:
    """The fit record's posterior summary: every category's intercepts and coefficients, its group SDs and ``phi``."""
    import arviz as az

    names = [model.mu_names[name][k] for name in model.spec.mu_names for k in ("intercept", "raw_intercept")]
    names += list(dict.fromkeys(var for rows in model.mu_coefficients.values() for _, var, _ in rows))
    names += [roles["sd"] for roles in model.mu_group_names.values()]
    names += [model.names["phi"]] if "phi" in model.names else []
    s = az.summary(idata, var_names=names, hdi_prob=0.94, round_to="none")
    return {name: {k: float(r[k]) for k in ("mean", "sd", "hdi_3%", "hdi_97%", "mcse_mean", "ess_bulk", "ess_tail", "r_hat")}
            for name, r in s.iterrows()}


def multinomial_fit_record(model: MultinomialModel, fit, panel: dict, binding: str) -> dict:
    """``session.fit_record`` of a count-matrix model: its own model section (every formula, the categories and trials, each
    category's formula, design columns, groups and PyMC names, every prior by label) and posterior summary; the execution,
    environment and diagnostics sections are ``fit_record``'s."""
    from autobayes import __version__, session
    from autobayes.default_priors import BRMS_VERSION

    s, d = model.spec, model.design
    defaults = model.default_prior_labels

    def groups(ds):
        md = d.mus[ds.response]
        return [{"factor": t.factor, "levels": list(md.group_blocks[(t.factor, "Intercept")].levels),
                 "names": model.mu_group_names[(ds.response, t.factor, "Intercept")]} for t in ds.group_terms]

    return {
        "autobayes_version": __version__,
        "model": {
            "formula": list(s.formulas) if len(s.formulas) > 1 else s.formula,
            "family": model.family,
            **({"target_identity": model.target_identity} if binding == "matched" else
               {"requested_target_identity": model.target_identity}),
            "fit_binding_status": binding,
            "link": model.link,
            "n": int(d.n),
            "multinomial": {"response": s.response, "canonical_formula": s.canonical_formula, "categories": list(s.categories),
                            "reference": s.reference, "trials": s.trials, "trials_total": int(d.trials.sum()), "names": dict(model.names),
                            "mus": {ds.response: {"formula": ds.formula, "canonical_formula": ds.canonical_formula,
                                                  "own_formula": ds.response in s.own,
                                                  "design_columns": list(d.mus[ds.response].columns),
                                                  "names": model.mu_names[ds.response], "groups": groups(ds),
                                                  "categorical": {c.name: {"reference": c.reference, "levels": list(c.levels),
                                                                           "coefficients": list(c.coefficients),
                                                                           "coding": "full_level" if c.full_rank else "treatment"}
                                                                  for c in ds.categoricals}}
                                    for ds in s.mus}},
            "priors": {label: str(p) for label, p in prior_labels(model.priors, s, model.family_spec)},
            **({"default_priors": {"brms": BRMS_VERSION, "parameters": list(defaults)}} if defaults else {}),
            "center_predictors": True,
            "noncentered": True if s.factors else None,
            "warnings": list(model.warnings),
        },
        "execution": {
            "backend": fit.backend, "device_requested": fit.device_requested,
            "device_actual": fit.device_actual, "platform": fit.platform, "dtype": fit.dtype,
            "chain_method": fit.chain_method, "chains": fit.chains, "draws": fit.draws,
            "tune": fit.tune, "target_accept": fit.target_accept, "seed": fit.seed,
            "wall_seconds": round(float(fit.wall_seconds), 3), "notes": list(fit.notes),
        },
        "environment": session.environment(),
        "diagnostics": {"values": {k: (float(v) if v is not None else None) for k, v in panel["values"].items()},
                        "status": dict(panel["status"]), "overall": panel["overall"]},
        "posterior": multinomial_summary_rows(model, fit.idata),
    }


def multinomial_prior_predictive(model: MultinomialModel, *, draws=500, seed=None):
    """``prior_predictive()`` of a count-matrix model: forward draws of the graph on the fitted rows (the parameters as sampled,
    the counts by the likelihood's ``random``), refused under a flat prior, and checked: the draws' shapes, the counts whole,
    non-negative and summing to each row's trials, the observed counts the model's."""
    from numbers import Integral

    import pymc as pm

    from autobayes import __version__

    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, Integral) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    improper = model.priors.improper(model.spec)
    if improper:
        raise ValueError(f"the prior of {improper} is flat (Flat(), brms's default for b): an improper prior has no prior predictive "
                         "distribution; give proper priors for these parameters to draw from the prior")
    identity = json.dumps(model.target_identity, sort_keys=True)
    try:
        result = pm.sample_prior_predictive(draws=int(draws), random_seed=None if seed is None else int(seed), model=model.pymc_model,
                                            return_inferencedata=True)
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise ValueError("prior predictive generation failed; inspect priors and numerical support; no retry performed") from exc
    y = result.prior_predictive[model.names["y"]].values
    if y.shape != (1, draws, model.design.n, len(model.categories)):
        raise ValueError("prior predictive response dimensions do not match modeled rows and categories")
    if (y < 0).any() or (y.sum(axis=-1) != model.design.trials).any():
        raise ValueError("prior predictive counts violate the response's support")
    for ds in (result.prior, result.prior_predictive):
        for arr in ds.data_vars.values():
            if not np.isfinite(arr.values).all():
                raise ValueError("prior predictive output contains nonfinite draws; inspect priors")
    result.attrs.update(autobayes_generation_kind="prior_predictive", autobayes_built_target_identity_json=identity,
                        autobayes_seed_json=json.dumps(None if seed is None else int(seed)), autobayes_requested_draws=int(draws),
                        autobayes_version=__version__, autobayes_parameterization="native_graph")
    return result
