"""Multivariate Gaussian models (C01, docs/multivariate.md): brms's ``bf(mvbind(y, y2) ~ ...) + set_rescor(TRUE | FALSE)``.

``ab.gaussian("mvbind(y, y2) ~ x + (1 | g)", data, priors, rescor=True)`` builds one univariate Gaussian model per response
on the same data (``MultivariateModel.submodels``: each response's own formula ``y ~ x + (1 | g)`` with its design, group
terms, priors and Bambi graph, exactly the univariate model) and assembles their PyMC graphs into one
(``pymc.model.fgraph``), every variable named after its response: ``y_Intercept``, ``y_x``, ``y_sigma``, ``y_1|g``,
``y_1|g_sigma``, ``y_mu``, and ``y2_...`` for ``y2``. The coordinates (``__obs__``, ``g__factor_dim``, ...) are shared, the
responses having the same right-hand side on the same rows. As in brms, each response has its own group-level terms
(``sd_1``/``z_1`` for ``y``, ``sd_2``/``z_2`` for ``y2``), uncorrelated across responses.

With ``rescor=True`` the responses' Normal likelihoods are replaced by brms's ``multi_normal_cholesky(Y_n | Mu_n,
diag(sigma) Lrescor)`` over each row's responses (the observed variable ``mvbind(y, y2)``, ``multi_normal_cholesky_logp``),
``Lrescor`` the Cholesky factor of the residual correlation matrix under ``lkj_corr_cholesky(eta)`` (``Priors(rescor=
ab.LKJ(eta))``, brms's class ``rescor``) sampled on Stan's ``cholesky_corr`` coordinates (``autobayes.correlation``, as A02's
``L_m``), and the deterministic ``rescor = Lrescor Lrescor^T`` (brms's ``Rescor``). With ``rescor=False`` each response keeps
its own likelihood (the observed variables ``y`` and ``y2``), as brms's ``set_rescor(FALSE)``.

Priors are given by response, as brms's ``resp =`` (``Priors(resp={"y": {...}, "y2": {...}}, rescor=...)``), with a
correlated term's ``cor`` at the top level, shared by the responses (brms's ``cor`` has no ``resp``). ``predict()`` returns
draws x rows x responses (a ``response`` dimension in the order written, brms's ``posterior_linpred`` / ``posterior_epred``
array); ``kind="response"`` draws each row's responses jointly from the multivariate normal. ``fit()`` samples the
assembled graph with PyMC's ``pm.sample`` (``sample_posterior``) and cleans the result as Bambi does.

Refused, naming the construct: distributional formulas and addition terms other than ``weights()`` (D56), ``mi()`` in an
``mvbind()`` formula, ``mo()`` (not verified in a multivariate model; D56), correlated group effects across responses other
than G08's (D57, D193), families other than the Gaussian (``FamilySpec.multivariate``; D58), and a model without an explicit
``rescor`` (D58).

Missing values (C06, docs/missing-and-measurement-error.md): brms's ``bf(y ~ mi(xm) + w) + bf(xm | mi() ~ w) +
set_rescor(FALSE)`` is a list of formulas, one per response, ``ab.gaussian(["y ~ mi(xm) + w", "xm | mi() ~ w"], data,
ab.Priors(resp={...}), rescor=False)``: the response ``xm`` has missing values (NaN), which are the parameters ``Ymi_xm``
(brms's ``Ymi_xm``, flat), and ``mi(xm)`` reads the column completed by them (brms's ``Yl_xm``). Each response's model holds
the missing values it reads; the assembled graph has one ``Ymi_xm``, by brms's name, shared by them. Refused: an ``mi()``
term without its column's ``| mi()`` formula, with ``rescor=True``, in a cycle of imputations, ``me()`` in a multivariate
model, and a response read by another formula other than through ``mi()`` (D72, D73).

A formula per response and brms's ``subset()`` (F06c, docs/subset.md): brms's ``bf(y | subset(sub1) ~ x + (1 | g)) + bf(y2 |
subset(sub2) ~ x) + set_rescor(FALSE)`` is ``ab.gaussian(["y | subset(sub1) ~ x + (1 | g)", "y2 | subset(sub2) ~ x"], data,
ab.Priors(resp={...}), rescor=False)``: each response's model is the univariate model of its own formula on the rows of its
subset (those whose 0/1 or logical indicator is 1; every row without ``subset()``), so its likelihood covers its rows only, and
the assembled graph gives each response its own observation coordinate (``observation_dims``). ``predict()`` takes one response
at a time (``resp=``), on its rows of the fitted data or of ``newdata`` (by the new rows' own indicator), as brms's
``posterior_linpred(resp = )``; brms refuses the joint predictions of such a model, and so does ``predict()``. A list of formulas
without ``subset()`` or ``mi()`` is the same model on every row. Refused, naming the construct: ``rescor=True`` with
``subset()`` (brms refuses it), ``subset()`` with ``mi()`` (brms requires ``mi(idx = )``) or with other addition terms, an
indicator other than a 0/1 or logical column, an empty subset, a grouping level of the data that a response's subset does not
reach (brms keeps it as a prior-only effect), and the univariate ``subset()`` (D160, D161).

Multivariate extensions (G08, docs/multivariate-ext.md): a list of formulas without ``subset()`` or ``mi()`` takes ``rescor=True``
(each response's own design in the joint likelihood); brms's correlation ID across responses, ``(1 |p| g)`` in ``mvbind()`` or in
each response's formula (``CrossId``, ``MultivariateSpec.cross_ids``), is each response's plain ``(1 | g)`` correlated across the
responses by the model (``L|g``, ``cor|g``, ``id_names``; the ID's LKJ prior is the top-level ``cor`` of its factor,
``split_cor``); ``weights()`` after ``mvbind()`` or on a response's own formula is each response's weighted likelihood, or under
a residual correlation each row's joint density weighted once (``joint_weights``, ``weighted_logp``). Refused, naming the
construct: slopes, ``gr()``, ``mm()``, ``g:h`` with an ID across responses, an ID in one response alone, two IDs on one factor,
an ID with ``subset()`` or ``mi()``, joint predictions at new levels of an ID's factor (D193), differing weights under a residual
correlation, ``weights()`` with ``subset()`` (D194).
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd

from autobayes.correlation import check_correlation_cholesky, cholesky_corr_free, cholesky_corr_rv, n_free
from autobayes.families import FAMILIES, get_family
from autobayes.formula import (_GROUP_RE, _IDENT, MI_CALL_RE, MVBIND_RE, FormulaError, ModelSpec, _split_top_level, check_imputation,
                               parse_formula)
from autobayes.formula import SUBSET_CALL_RE, group_labels   # F06c
from autobayes.term_algebra import refuse_products           # G03
from autobayes.formula import extended_group_syntax   # G05a
from autobayes.priors import PriorError, Priors, describe_prior

_MVBIND_FORMULA_RE = re.compile(r"^\s*mvbind\s*\(([^()]*)\)\s*([^~]*?)\s*~\s*(.*?)\s*$", re.S)   # mvbind(<responses>) <rest> ~ <rhs>
_MI_RE = re.compile(r"(?<![A-Za-z0-9_.])mi\s*\(")                                                 # brms's mi() (C06)
_MO_RE = re.compile(r"(?<![A-Za-z0-9_.])mo\s*\(")                                                 # brms's mo() (B03)
_GP_RE = re.compile(r"(?<![A-Za-z0-9_.])gp\s*\(")                                                 # brms's gp() (C03)
_AUTOCOR_RE = re.compile(r"(?<![A-Za-z0-9_.])(ar|ma|arma)\s*\(")                                     # brms's ARMA terms (C04)
_SAR_RE = re.compile(r"(?<![A-Za-z0-9_.])sar\s*\(")                                                      # brms's SAR term (F02b)
_FCOR_RE = re.compile(r"(?<![A-Za-z0-9_.])fcor\s*\(")                                                 # brms's fcor() (F03a)
_ME_RE = re.compile(r"(?<![A-Za-z0-9_.])me\s*\(")                                                 # brms's me() (C06)
_CAR_RE = re.compile(r"(?<![A-Za-z0-9_.])car\s*\(")                                               # brms's car() (F02a)
_TIMECOR_RE = re.compile(r"(?<![A-Za-z0-9_.])(unstr|cosy)\s*\(")                                    # brms's unstr(), cosy() (F03b)
RESPONSE_DIM, RESPONSE_DIM_T = "__resp__", "__resp_T__"     # the graph's response coordinates (Lrescor, rescor, the joint response)
PREDICTION_DIM = "response"                                 # predict()'s response dimension, brms's last axis of its prediction arrays
_LHS_RE = re.compile(rf"^\s*({_IDENT})\s*\|\s*(.*?)\s*~\s*(.*?)\s*$", re.S)                       # <response> | <terms> ~ <rhs> (F06c)
_SUBSET_TERM_RE = re.compile(r"^subset\s*\((.*)\)$", re.S)                                          # subset(<inside>) (F06c)
SUBSET_REFUSED = "brms refuses the joint predictions of a model with subset()"               # F06c: predict() without resp=
_ID_PIECE_RE = re.compile(rf"^\(\s*([^|]*?)\s*\|\s*({_IDENT})\s*\|\s*([^|]*?)\s*\)$", re.S)        # (<inner> |<ID>| <factor>) (G08)
_WEIGHTS_TERM_RE = re.compile(r"^weights\s*\(", re.S)                                                # weights(...) (G08)
ID_ROW = "docs/multivariate-ext.md, divergence D193"                                                 # G08: IDs across responses
WEIGHTS_ROW = "docs/multivariate-ext.md, divergence D194"                                            # G08: weights() under rescor


@dataclass(frozen=True)
class CrossId:
    """G08: brms's correlation ID shared by the group-level terms of several responses, ``(1 |p| g)`` in ``mvbind(y, y2) ~ ...``
    or in each response's own formula: one group of brms (``sd_m`` with one entry per response, each under that response's
    ``sd`` prior, ``z_m``, one ``L_m``) over the varying intercepts of ``responses`` (in the model's order of the responses) on
    the grouping column ``factor``. Each response's model has the plain ``(1 | g)``; the model correlates them."""
    id: str
    factor: str
    responses: tuple[str, ...]

    @property
    def coefs(self) -> tuple[tuple[str, str], ...]:
        """``(response, coefficient)`` in brms's order (``cn`` 1, 2, ...): each response's varying intercept."""
        return tuple((r, "Intercept") for r in self.responses)

    @property
    def labels(self) -> tuple[str, ...]:
        """The coordinates of the correlation matrix, ``y_Intercept``, ``y2_Intercept`` (brms's ``cor_g__y_Intercept__y2_Intercept``)."""
        return tuple(f"{r}_{c}" for r, c in self.coefs)


def _strip_ids(rhs: str, what: str) -> tuple[str, dict[str, str]]:
    """G08: ``rhs`` with each group-level term that carries a correlation ID, ``(1 |p| g)``, written without it, ``(1 | g)``,
    and the IDs met, ``{"p": "g"}``; ``rhs`` unchanged without one. Taken: the varying intercept alone on a grouping column.
    Refused, naming the construct: varying slopes or no varying intercept with the ID, an ID on ``gr()``, ``mm()``, ``g:h`` or
    ``g/h``, and one ID twice in a formula."""
    pieces, ids = [], {}
    for piece in _split_top_level(rhs):
        m = _ID_PIECE_RE.match(piece)
        if m is None:
            pieces.append(piece)
            continue
        inner, term_id, factor = (s.strip() for s in m.groups())
        if re.sub(r"\s+", "", inner) != "1":
            raise FormulaError(f"{what}: {piece!r}: varying slopes (or a term without the varying intercept) in a correlation ID across "
                               f"responses are not supported; the ID is offered on the varying intercepts, '(1 |{term_id}| {factor})' "
                               f"({ID_ROW})")
        if not re.fullmatch(_IDENT, factor):
            raise FormulaError(f"{what}: {piece!r}: a correlation ID across responses is offered on a grouping column, "
                               f"'(1 |{term_id}| g)'; the grouping {factor!r} (gr(), mm(), g:h, g/h) is not supported with it ({ID_ROW})")
        if term_id in ids:
            raise FormulaError(f"{what}: the correlation ID {term_id!r} appears in two terms of one formula; brms writes the "
                               "coefficients of one factor in one term (\"Can only combine group-level terms of the same grouping "
                               f"factors\" for two factors; {ID_ROW})")
        ids[term_id] = factor
        pieces.append(f"(1 | {factor})")
    return (" + ".join(pieces), ids) if ids else (rhs, ids)


def _strip_formula_ids(text: str, what: str) -> tuple[str, dict[str, str]]:
    """G08: a response's formula ``<lhs> ~ <rhs>`` with its right-hand side's correlation IDs stripped (``_strip_ids``)."""
    lhs, rhs = text.split("~", 1)
    stripped, ids = _strip_ids(rhs.strip(), what)
    return (f"{lhs.strip()} ~ {stripped}" if ids else text), ids


def _cross_ids(found: Mapping[str, Mapping[str, str]], responses, what) -> tuple[CrossId, ...]:
    """G08: the correlation IDs across responses from each response's IDs (``found``: response -> {ID: grouping column}), in the
    order met. An ID must join the same grouping column in two or more responses (brms refuses two columns: "Can only combine
    group-level terms of the same grouping factors."); an ID in one response alone is refused as a univariate formula refuses
    it (divergence D106), and two IDs on one grouping column (D193)."""
    uses: dict[str, list[tuple[str, str]]] = {}
    for r in responses:
        for term_id, factor in found.get(r, {}).items():
            uses.setdefault(term_id, []).append((r, factor))
    out = []
    for term_id, pairs in uses.items():
        factors = sorted({f for _, f in pairs})
        if len(factors) > 1:
            raise FormulaError(f"{what}: the correlation ID {term_id!r} joins the grouping factors {factors}; brms refuses it (\"Can "
                               f"only combine group-level terms of the same grouping factors.\"); an ID joins one factor's terms ({ID_ROW})")
        if len(pairs) < 2:
            raise FormulaError(f"{what}: the correlation ID {term_id!r} of '(1 |{term_id}| {factors[0]})' is used by the response "
                               f"{pairs[0][0]!r} alone; an ID correlates the varying intercepts of two or more responses here (in one "
                               f"formula it is not supported, divergence D106): write '(1 | {factors[0]})' without it")
        out.append(CrossId(term_id, factors[0], tuple(r for r, _ in pairs)))
    twice = sorted({c.factor for c in out if sum(o.factor == c.factor for o in out) > 1})
    if twice:
        raise FormulaError(f"{what}: two correlation IDs across responses on the grouping factor {twice[0]!r} are not supported ({ID_ROW})")
    return tuple(out)


def _restore_ids(text: str, spec: ModelSpec, response: str, cross_ids) -> str:
    """G08: ``text`` (a response's canonical formula or right-hand side) with the response's terms of a correlation ID across
    responses written with their ID, ``(1 |p| g)`` for ``(1 | g)``."""
    for cid in cross_ids:
        if response in cid.responses:
            t = spec.group_term_of(cid.factor)
            text = text.replace(t.canonical, replace(t, id=cid.id).canonical)
    return text


def _mvbind_additions(formula: str, rest: str) -> str:
    """G08: the addition terms after ``mvbind()``, ``| weights(w)``, as written after the bar: ``weights()`` alone is taken (each
    response's formula carries it, brms's ``weights_y``, ``weights_y2``); anything else is refused naming it (D56)."""
    terms = _split_top_level(rest[1:]) if rest.startswith("|") else [rest]
    others = [t for t in terms if not _WEIGHTS_TERM_RE.match(t)]
    if others or not rest.startswith("|"):
        raise FormulaError(f"{formula!r}: {rest!r} after mvbind(): addition terms other than weights() ({', '.join(others)}) on the "
                           "responses of a multivariate model are not supported (docs/multivariate.md, divergence D56)")
    return rest[1:].strip()


def split_cor(priors: Priors, spec) -> tuple[dict, object]:
    """G08: ``priors.cor`` split between the correlation IDs across responses (each one's LKJ prior, keyed by its grouping
    column, brms's ``class = cor, group = g``) and the ``cor`` the responses' own correlated terms take (``response_priors``):
    a factor's own entry, else the class-wide ``"*"`` entry (D173); a bare prior when the model has one correlation matrix in
    all. Without an ID across responses: ``({}, priors.cor)``."""
    from autobayes.priors import _CORRELATION, CLASS_WIDE, Prior, class_default

    cross = [c.factor for c in getattr(spec, "cross_ids", ())]
    if not cross:
        return {}, priors.cor
    own = sorted({t.factor for s in spec.specs.values() for t in s.group_terms if t.has_correlation})
    cor, first = priors.cor, spec.cross_ids[0]
    if cor is None:
        raise PriorError(f"the correlation ID {first.id!r} across the responses {list(first.responses)} gives their varying intercepts "
                         f"of {first.factor!r} a correlation matrix, which needs an LKJ prior: pass cor={{{first.factor!r}: ab.LKJ(eta)}} "
                         f"(brms class cor, group {first.factor}; its default is lkj(1))")
    if isinstance(cor, Mapping):
        default = class_default(cor, "cor")
        out = {g: cor.get(g, default) for g in cross}
        missing = [g for g, p in out.items() if p is None]
        if missing:
            raise PriorError(f"cor has no prior for {missing}, whose varying intercepts a correlation ID correlates across the "
                             f"responses: pass cor={{{missing[0]!r}: ab.LKJ(eta), ...}}")
        rest = {k: v for k, v in cor.items() if k not in cross}
        rest = None if not rest or (set(rest) == {CLASS_WIDE} and not own) else rest
    elif len(cross) + len(own) != 1:
        raise PriorError(f"multiple correlated grouping factors ({cross + own}) require an exact-key cor mapping; a bare prior is not "
                         "broadcast")
    else:
        out, rest = {cross[0]: cor}, None
    for g, p in out.items():
        if not isinstance(p, Prior) or p.name not in _CORRELATION:
            raise PriorError(f"prior for 'cor:{g}' must be a correlation prior ({sorted(_CORRELATION)}, brms's lkj(eta)), got {p!r}")
    return out, rest


def _check_rescor_weights(specs: Mapping[str, ModelSpec], responses, what) -> None:
    """G08: with a residual correlation brms weights each row's joint density once, so every response must have the same weights:
    the same ``weights()`` column, or none. brms refuses two columns ("All responses should use the same weights if 'rescor' is
    estimated."); with only some responses weighted it weights the joint density by the first response's weights, or writes Stan
    code that does not compile when only a later one is weighted (divergence D194): refused here, naming the columns."""
    cols = {r: (a.column if (a := specs[r].addition("weights")) is not None else None) for r in responses}
    if len(set(cols.values())) > 1:
        shown = ", ".join(f"{r}: {'weights(' + c + ')' if c else 'no weights'}" for r, c in cols.items())
        raise FormulaError(f"{what}: with a residual correlation (rescor=True) every response must have the same weights, got {shown}; "
                           "brms refuses differing weights columns (\"All responses should use the same weights if 'rescor' is "
                           f"estimated.\") and the responses' weights must be one column here, or none ({WEIGHTS_ROW})")


def _has_subset(formula: str) -> bool:
    """Whether brms's ``subset()`` is on the left-hand side of ``formula`` (F06c)."""
    return "~" in formula and SUBSET_CALL_RE.search(formula.split("~", 1)[0]) is not None


def is_multivariate_formula(formula) -> bool:
    """Whether ``formula`` is brms's ``mvbind(y, y2) ~ ...`` (C01), alone or first in a list of formulas, or a list of
    formulas with ``mi()`` (C06: one formula per response, ``["y ~ mi(xm) + w", "xm | mi() ~ w"]``) or with ``subset()`` on a
    response (F06c: ``["y | subset(sub1) ~ x", "y2 | subset(sub2) ~ x"]``)."""
    if isinstance(formula, (list, tuple)) and formula:
        if all(isinstance(f, str) for f in formula) and any(MI_CALL_RE.search(f) or _has_subset(f) for f in formula):
            return True
        formula = formula[0]
    return isinstance(formula, str) and MVBIND_RE.match(formula) is not None


def _offering() -> list[str]:
    return [name for name, spec in FAMILIES.items() if spec.multivariate]


@dataclass(frozen=True)
class MultivariateSpec:
    """A parsed multivariate formula (C01): brms's ``mvbind(y, y2) ~ rhs``. ``responses`` in the order written (the last
    axis of brms's prediction arrays, its ``responses``), ``rhs`` the right-hand side the responses share, as written,
    ``specs`` each response's own formula ``y ~ rhs`` parsed as a univariate formula (``ModelSpec``; with the data seen once
    the model is built, the sub-models' own), ``rescor`` whether the responses' residuals are correlated (brms's
    ``set_rescor``). A formula of its own per response (C06's ``bf(y ~ mi(x)) + bf(x | mi() ~ w)``) would fill ``specs`` from
    a list instead; the model reads nothing but ``specs``. ``subsets`` (F06c) maps each response written with brms's
    ``subset(<indicator>)`` to its indicator column; ``specs`` hold its formula without the term, which the response's model is
    built from on the subset's rows."""
    formula: str
    responses: tuple[str, ...]
    rhs: str
    specs: Mapping[str, ModelSpec]
    rescor: bool
    subsets: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    cross_ids: tuple[CrossId, ...] = ()      # G08: the correlation IDs across responses, each response's term (1 | g) in ``specs``

    @property
    def separate(self) -> bool:
        """A formula of its own per response (C06): ``formula`` is the list as written, ``rhs`` empty."""
        return isinstance(self.formula, tuple)

    @property
    def formulas(self) -> dict[str, str]:
        """Each response's own formula, ``{"y": "y ~ x + (1 | g)", ...}``, from which its univariate model is built (F06c: without
        its ``subset()`` term, on the subset's rows)."""
        return {r: self.specs[r].formula for r in self.responses}

    def response_formula(self, response: str) -> str:
        """``response``'s canonical formula as brms writes it in ``bf()``, its ``subset()`` term included (F06c):
        ``y | subset(sub1) ~ x + (1 | g)``."""
        text, column = self.specs[response].canonical_formula, self.subsets.get(response)
        text = _restore_ids(text, self.specs[response], response, self.cross_ids)     # G08: (1 |p| g)
        if column is None:
            return text
        lhs, rhs = text.split("~", 1)
        return f"{lhs.strip()} | subset({column}) ~{rhs}"

    @property
    def canonical_formula(self) -> str:
        """``mvbind(y, y2) ~ <the canonical right-hand side>``; a formula per response (C06, F06c) as brms writes it,
        ``bf(y ~ mi(xm) + w) + bf(xm | mi() ~ w)``, ``bf(y | subset(sub1) ~ x) + bf(y2 | subset(sub2) ~ x)``. G08: with the addition
        terms after ``mvbind()`` (``mvbind(y, y2) | weights(w) ~ ...``) and the correlation IDs across responses (``(1 |p| g)``)."""
        if self.separate:
            return " + ".join(f"bf({self.response_formula(r)})" for r in self.responses)
        first = self.specs[self.responses[0]]
        adds = " + ".join(a.term for a in first.additions)
        rhs = _restore_ids(first.rhs, first, self.responses[0], self.cross_ids)
        return f"mvbind({', '.join(self.responses)}){' | ' + adds if adds else ''} ~ {rhs}"

    @property
    def groups(self) -> tuple[str, ...]:
        """The grouping factors of every response's formula (the same right-hand side, so the same factors; with a formula
        per response, C06, every response's in the order met)."""
        return tuple(dict.fromkeys(g for r in self.responses for g in self.specs[r].groups))

    @property
    def imputed(self) -> tuple[str, ...]:
        """The responses whose missing values are parameters, ``col | mi() ~ ...`` (C06), in the order written."""
        return tuple(r for r in self.responses if self.specs[r].addition("mi") is not None)


def parse_multivariate(formula, data: pd.DataFrame | None = None, family: str = "gaussian", categorical=None, *,
                       rescor) -> MultivariateSpec:
    """Parse brms's ``mvbind(y1, y2, ...) ~ rhs`` (C01): two or more distinct response columns sharing one right-hand side,
    each response's formula ``y ~ rhs`` parsed (and validated against ``data``) as the univariate formula it is; ``rescor``
    must be True or False. G08: ``mvbind(y, y2) | weights(w) ~ rhs`` gives each response ``y | weights(w) ~ rhs``, and a
    correlation ID ``(1 |p| g)`` is each response's ``(1 | g)`` and one ``CrossId`` over the responses. Everything outside the
    verified subset is refused naming the construct (module docstring)."""
    fam = get_family(family)
    if not fam.multivariate:
        raise FormulaError(f"a multivariate model (brms's mvbind() or mvbf()) is offered for the {', '.join(_offering()) or 'no'} family "
                           f"here; a multivariate {fam.name} model is not (docs/multivariate.md, divergence D58)")
    if (isinstance(formula, (list, tuple)) and formula and all(isinstance(f, str) for f in formula)
            and not MVBIND_RE.match(formula[0])):                                     # a formula per response
        if not any(MI_CALL_RE.search(f) for f in formula):
            return _parse_separate(formula, data, family, categorical, rescor=rescor)   # F06c: with or without subset()
        if any(_has_subset(f) for f in formula):
            raise FormulaError(f"{formula!r}: subset() in a model with mi() is refused, as brms refuses it (\"mi() terms in subsetted "
                               "formulas require the 'idx' argument to be specified\"); brms's mi(x, idx = ) is not supported"
                               " (docs/subset.md, divergence D160)")
        return _parse_imputation(formula, data, family, categorical, rescor=rescor)      # C06: a formula per response, with mi()
    if not isinstance(formula, str):
        first = formula[0] if isinstance(formula, (list, tuple)) and formula else formula
        what = ("a distributional formula in a multivariate model (brms's bf(mvbind(y, y2) ~ ..., sigma ~ z))"
                if isinstance(first, str) and MVBIND_RE.match(first) else
                "a formula of its own per response (brms's bf(y ~ ...) + bf(y2 ~ ...))")
        raise FormulaError(f"{formula!r}: {what} is not supported; a multivariate model is one formula, "
                           "'mvbind(y, y2) ~ <terms>', whose right-hand side every response shares (docs/multivariate.md, divergence D56)")
    if rescor is None:
        raise ValueError("a multivariate model needs rescor=True (the responses' residuals correlated, brms's set_rescor(TRUE)) or "
                         "rescor=False (independent likelihoods); brms 2.23 sets it to TRUE for a Gaussian model by default with a "
                         "warning that the default will change, and AutoBayes takes no default (docs/multivariate.md, divergence D58)")
    if not isinstance(rescor, (bool, np.bool_)):
        raise ValueError(f"rescor must be True or False, got {rescor!r}")
    if not MVBIND_RE.match(formula):
        raise FormulaError(f"{formula!r}: rescor= applies to a multivariate formula, 'mvbind(y, y2) ~ <terms>' (brms's mvbind()); "
                           "this formula has one response: drop rescor")
    m = _MVBIND_FORMULA_RE.match(formula)
    if not m:
        raise FormulaError(f"{formula!r}: expected 'mvbind(y1, y2, ...) ~ <terms>', the responses as column names inside mvbind()")
    inside, rest, rhs = m.group(1), m.group(2).strip(), m.group(3).strip()
    additions = _mvbind_additions(formula, rest) if rest else ""      # G08: '| weights(w)', each response's
    responses = tuple(p.strip() for p in inside.split(","))
    bad = [p for p in responses if not re.fullmatch(_IDENT, p)]
    if bad:
        raise FormulaError(f"{formula!r}: mvbind() takes response column names, got {bad}")
    if len(responses) < 2:
        raise FormulaError(f"{formula!r}: mvbind() needs at least two responses; one response is a univariate model, '{responses[0]} ~ ...'")
    dup = sorted({r for r in responses if responses.count(r) > 1})
    if dup:
        raise FormulaError(f"{formula!r}: the response(s) {dup} appear more than once in mvbind()")
    if not rhs:
        raise FormulaError(f"{formula!r}: empty right-hand side")
    stripped, ids = _strip_ids(rhs, repr(formula))                     # G08: (1 |p| g), every response's term in one block
    for piece in _split_top_level(stripped):
        if _MI_RE.search(piece):
            raise FormulaError(f"{formula!r}: {piece!r}: brms's mi() (missing values modelled by a response's own formula) is not supported"
                               " (docs/multivariate.md, divergence D56): not in an mvbind() formula, "
                               "whose responses share one right-hand side; it is taken in a list of formulas, one per response, "
                               "['y ~ mi(x) + w', 'x | mi() ~ w'] with rescor=False (docs/missing-and-measurement-error.md)")
        if _GP_RE.search(piece):
            raise FormulaError(f"{formula!r}: {piece!r}: a Gaussian process in a multivariate formula is not supported "
                               "(each response would have its own process, brms's sdgp_y and lscale_y); write it in a univariate "
                               "model (docs/multivariate.md, divergence D56)")
        if _CAR_RE.search(piece):                             # F02a
            raise FormulaError(f"{formula!r}: {piece!r}: a CAR term in a multivariate formula is not supported; write it "
                               "in a univariate model (docs/spatial-car.md, divergence D141)")
        if _AUTOCOR_RE.search(piece):
            raise FormulaError(f"{formula!r}: {piece!r}: an autocorrelation term in a multivariate formula is not supported"
                               " (brms gives each response its own ARMA term, ar_y and ar_y2, with J_lag_y and J_lag_y2); write it "
                               "in a univariate model (docs/autocorrelation.md, divergence D70)")
        if _SAR_RE.search(piece):                             # F02b
            raise FormulaError(f"{formula!r}: {piece!r}: a SAR term in a multivariate formula is not supported; write it "
                               "in a univariate model (docs/spatial-sar.md, divergence D143)")
        if _FCOR_RE.search(piece):                            # F03a: brms gives each response its own Mfcor_y
            raise FormulaError(f"{formula!r}: {piece!r}: a fixed residual covariance in a multivariate formula is not supported"
                               " (brms gives each response its own matrix, Mfcor_y and Mfcor_y2); write it in a univariate "
                               "model (docs/fcor.md, divergence D144)")
        if _TIMECOR_RE.search(piece):                     # F03b
            raise FormulaError(f"{formula!r}: {piece!r}: a residual correlation term in a multivariate formula is not supported"
                               "; write it in a univariate model (docs/unstr-cosy.md, divergence D146)")
        if _ME_RE.search(piece):
            raise FormulaError(f"{formula!r}: {piece!r}: a measurement-error term in a multivariate model is not supported"
                               "; me() belongs to a univariate Gaussian model here (docs/missing-and-measurement-error.md, "
                               "divergence D73)")
        if _MO_RE.search(piece):
            raise FormulaError(f"{formula!r}: {piece!r}: a monotonic term in a multivariate formula is not supported (each "
                               "response would have its own coefficient and simplex, brms's bsp_y and simo_y); write it in a "
                               "univariate model (docs/multivariate.md, divergence D56)")
        gm = _GROUP_RE.match(piece)
        if gm and "|" in gm.group(3):                                  # G08: a correlation ID not written (1 |p| g)
            raise FormulaError(f"{formula!r}: {piece!r}: a correlation ID across responses is written '(1 |p| g)', the ID between "
                               f"two single bars ({ID_ROW})")
    specs = {}
    for r in responses:
        text = f"{r}{' | ' + additions if additions else ''} ~ {stripped}"
        try:
            s = parse_formula(text, data, family, categorical)
        except FormulaError as e:
            raise FormulaError(f"{formula!r}, the formula of the response {r!r} ('{text}'): {e}") from None
        refuse_products([s], "a multivariate model")                    # G03: verified in univariate models only
        if s.custom_group_terms:                                         # D08: verified in univariate models only
            raise FormulaError(f"{formula!r}: '{s.custom_group_terms[0].canonical}' in a multivariate formula is not supported"
                               " (docs/multi-membership-and-by.md, divergences D104, D105)")
        if extended_group_syntax(s) is not None:                         # G05a: verified in univariate models only
            raise FormulaError(f"{formula!r}: {extended_group_syntax(s)} in a multivariate formula is not supported "
                               "(docs/group-term-syntax.md, divergence D183)")
        reads = {*s.pop_columns, *(c for t in s.group_terms for c in t.operands), *(s.offset.columns if s.offset is not None else ())}
        others = sorted(reads & set(responses))
        if others:
            raise FormulaError(f"{formula!r}: the response(s) {others} cannot also be predictors, grouping factors or offsets of the "
                               "right-hand side every response shares")
        weighted = sorted({c for a in s.additions for c in a.columns} & set(responses))
        if weighted:                                                   # G08
            raise FormulaError(f"{formula!r}: the response(s) {weighted} cannot also be the weights of the responses")
        specs[r] = s
    cross = _cross_ids({r: ids for r in responses}, responses, repr(formula))
    return MultivariateSpec(formula.strip(), responses, rhs, MappingProxyType(specs), bool(rescor), cross_ids=cross)


IMPUTATION_FORM = ("a list of formulas, one per response, the formula of a response with missing values written 'x | mi() ~ ...' "
                   "and read by another as mi(x): ab.gaussian(['y ~ mi(x) + w', 'x | mi() ~ w'], data, ab.Priors(resp={...}), "
                   "rescor=False) (brms's bf(y ~ mi(x) + w) + bf(x | mi() ~ w) + set_rescor(FALSE); docs/missing-and-measurement-error.md)")


def response_categorical(spec: MultivariateSpec, categorical) -> dict:
    """Each response's declared categorical columns (C06): with one right-hand side (``mvbind()``) every response takes
    ``categorical``; with a formula per response, each takes the declarations of its own main effects, and a declaration no
    formula uses stays with the first, which refuses it as a univariate formula does."""
    if not spec.separate or not isinstance(categorical, Mapping):
        return {r: categorical for r in spec.responses}
    mains = {r: {t.name for t in s.pop_terms if t.kind in ("numeric", "categorical")} for r, s in spec.specs.items()}
    used = {k for m in mains.values() for k in m}
    out = {r: {k: v for k, v in categorical.items() if k in mains[r]} or None for r in spec.responses}
    stray = {k: v for k, v in categorical.items() if k not in used}
    if stray:
        first = spec.responses[0]
        out[first] = {**(out[first] or {}), **stray}
    return out


def _parse_imputation(formulas, data, family, categorical, *, rescor) -> MultivariateSpec:
    """A formula per response with brms's ``mi()`` (C06): each ``'<response> ~ <terms>'`` or ``'<response> | mi() ~ <terms>'``
    parsed as the univariate formula it is (and validated against ``data``), the responses distinct, ``rescor=False``. Every
    ``mi(col)`` term needs ``col``'s own formula with ``| mi()`` in the list (brms refuses it otherwise too); refused, naming the
    construct: no ``mi()`` term (a formula per response alone, D56), ``rescor=True`` and imputations in a cycle (D72), ``me()``
    (D73), ``mo()``, other addition terms (D56), a Gaussian process ``gp()`` (C03, D56) or an autocorrelation term (C04, D70) in
    any formula of the list, as in an ``mvbind()`` formula, and a response read by another formula other than through ``mi()``."""
    texts = list(formulas)
    if len(texts) < 2 or not all(isinstance(f, str) for f in texts):
        raise FormulaError(f"{formulas!r}: {IMPUTATION_FORM}")
    if any(MVBIND_RE.match(f) for f in texts):
        raise FormulaError(f"{formulas!r}: mvbind() inside a list of formulas is not supported; {IMPUTATION_FORM}")
    if rescor is None:
        raise ValueError("a multivariate model needs rescor=True or rescor=False (divergence D58); a model with mi() takes rescor=False "
                         "(brms's set_rescor(FALSE); docs/missing-and-measurement-error.md)")
    if not isinstance(rescor, (bool, np.bool_)):
        raise ValueError(f"rescor must be True or False, got {rescor!r}")
    if rescor:
        raise FormulaError(f"{formulas!r}: mi() with a residual correlation (rescor=True, brms's set_rescor(TRUE)) is not supported"
                           ": brms would correlate the imputed values' residuals with the other responses'; pass "
                           "rescor=False (docs/missing-and-measurement-error.md, divergence D72)")
    for text in texts:                                         # G08: not verified with mi()
        if "~" in text and any(_ID_PIECE_RE.match(p) for p in _split_top_level(text.split("~", 1)[1].strip())):
            raise FormulaError(f"{formulas!r}, the formula {text!r}: a correlation ID across responses in a model with mi() is not "
                               f"supported ({ID_ROW})")
    specs0 = []
    for text in texts:
        try:
            specs0.append(parse_formula(text, None, family, None))     # the structure, before the data and the declarations
        except FormulaError as e:
            raise FormulaError(f"{formulas!r}, the formula {text!r}: {e}") from None
        refuse_products(specs0[-1:], "a multivariate model")            # G03: verified in univariate models only
        if specs0[-1].custom_group_terms:                               # D08: verified in univariate models only
            raise FormulaError(f"{formulas!r}, the formula {text!r}: '{specs0[-1].custom_group_terms[0].canonical}' in a multivariate "
                               "model is not supported (docs/multi-membership-and-by.md, divergences D104, D105)")
        if extended_group_syntax(specs0[-1]) is not None:              # G05a: verified in univariate models only
            raise FormulaError(f"{formulas!r}, the formula {text!r}: {extended_group_syntax(specs0[-1])} in a multivariate model is "
                               "not supported (docs/group-term-syntax.md, divergence D183)")
    responses = tuple(s.response for s in specs0)
    dup = sorted({r for r in responses if responses.count(r) > 1})
    if dup:
        raise FormulaError(f"{formulas!r}: the response(s) {dup} have more than one formula; give each response one")
    imputed = tuple(s.response for s in specs0 if s.addition("mi") is not None)
    for s in specs0:
        others = [a.term for a in s.additions if a.kind != "mi"]
        if others or s.trials is not None:
            raise FormulaError(f"{s.formula!r}: addition terms ({', '.join(others or ['trials()'])}) on a response of a multivariate model "
                               "are not supported (docs/multivariate.md, divergence D56)")
        if s.me_terms:
            raise FormulaError(f"{s.formula!r}: a measurement-error term in a multivariate model is not supported; me() "
                               "belongs to a univariate Gaussian model here (docs/missing-and-measurement-error.md, divergence D73)")
        if s.mo_terms:
            raise FormulaError(f"{s.formula!r}: a monotonic term in a multivariate model is not supported (docs/multivariate.md, "
                               "divergence D56)")
        if s.gp_terms:                                        # C03: refused as in an mvbind() formula (a formula with mi() already was)
            raise FormulaError(f"{s.formula!r}: the Gaussian process {s.gp_terms[0].term!r} in a multivariate model is not supported"
                               " (each response would have its own process, brms's sdgp_y and lscale_y); write it in a univariate "
                               "model (docs/multivariate.md, divergence D56)")
        if s.fcor is not None:                                # F03a: likewise
            raise FormulaError(f"{s.formula!r}: {s.fcor.term} in a multivariate model is not supported; write it in a "
                               "univariate model (docs/fcor.md, divergence D144)")
        if s.car is not None:                                 # F02a: likewise
            raise FormulaError(f"{s.formula!r}: the CAR term {s.car.term!r} in a multivariate model is not supported; write "
                               "it in a univariate model (docs/spatial-car.md, divergence D141)")
        if s.autocor is not None:                             # C04: likewise
            raise FormulaError(f"{s.formula!r}: the autocorrelation term {s.autocor.term!r} in a multivariate model is not supported"
                               " (brms gives each response its own ARMA term, ar_y and ar_y2); write it in a univariate model "
                               "(docs/autocorrelation.md, divergence D70)")
        if getattr(s, "sar", None) is not None:               # F02b: likewise
            raise FormulaError(f"{s.formula!r}: the SAR term {s.sar.term!r} in a multivariate model is not supported; "
                               "write it in a univariate model (docs/spatial-sar.md, divergence D143)")
        if s.timecor is not None:                             # F03b: likewise
            raise FormulaError(f"{s.formula!r}: the residual correlation term {s.timecor.term!r} in a multivariate model is not supported"
                               "; write it in a univariate model (docs/unstr-cosy.md, divergence D146)")
        for t in s.mi_terms:
            col = t.operands[0]
            if col not in responses:
                raise FormulaError(f"{s.formula!r}: {t.name!r} reads {col!r}, which no formula of the list models; add its formula, "
                                   f"'{col} | mi() ~ <terms>' (brms refuses an mi() term without it too; divergence D72)")
            if col not in imputed:
                raise FormulaError(f"{s.formula!r}: {t.name!r} reads {col!r}, whose formula must say that its missing values are "
                                   f"parameters: write it '{col} | mi() ~ ...' (divergence D72)")
        check_imputation(s, imputed)
        columns = {*(c for t in s.pop_terms if t.kind != "missing" for c in t.operands), *(c for t in s.group_terms for c in t.operands),
                   *(s.offset.columns if s.offset is not None else ())}       # what the formula reads other than through mi()
        plain = sorted(columns & (set(responses) - {s.response}))
        if plain:
            raise FormulaError(f"{s.formula!r}: the response(s) {plain} of another formula are read here other than through mi(); "
                               f"write mi({plain[0]}) to read {plain[0]!r} with its imputed values (docs/missing-and-measurement-error.md)")
    if not any(s.mi_terms for s in specs0):                   # F06c: a formula per response without '| mi()' is _parse_separate's
        raise FormulaError(f"{formulas!r}: the missing values of {list(imputed)} are parameters ('| mi()') that no formula reads through "
                           "mi(), which is not supported (divergence D72); a formula per response without '| mi()' is taken "
                           f"with rescor=False (docs/subset.md), and {IMPUTATION_FORM}")
    reads = {s.response: [t.operands[0] for t in s.mi_terms] for s in specs0}
    state: dict[str, int] = {}

    def visit(r, path):                                       # depth-first search for a cycle of mi() reads
        state[r] = 1
        for c in reads[r]:
            if state.get(c) == 1:
                cycle = path[path.index(c):] + [c] if c in path else [r, c]
                raise FormulaError(f"{formulas!r}: the imputations form a cycle ({' -> '.join(cycle)}): each formula reads the next "
                                   "one's mi() column; this is not one joint model, and is refused (divergence D72)")
            if state.get(c, 0) == 0:
                visit(c, path + [c])
        state[r] = 2

    for r in responses:
        if state.get(r, 0) == 0:
            visit(r, [r])
    spec0 = MultivariateSpec(tuple(t.strip() for t in texts), responses, "", MappingProxyType({s.response: s for s in specs0}), False)
    cats = response_categorical(spec0, categorical)
    specs = {}
    for text, r in zip(texts, responses):
        try:
            specs[r] = parse_formula(text, data, family, cats[r])
        except FormulaError as e:
            raise FormulaError(f"{formulas!r}, the formula of the response {r!r} ({text!r}): {e}") from None
    return replace(spec0, specs=MappingProxyType(specs))


SEPARATE_FORM = ("a list of formulas, one per response, each '<response> ~ <terms>' or '<response> | subset(<indicator>) ~ <terms>', "
                 "with rescor=False: ab.gaussian(['y | subset(s1) ~ x + (1 | g)', 'y2 | subset(s2) ~ x'], data, ab.Priors(resp={...}), "
                 "rescor=False) (brms's bf(y | subset(s1) ~ ...) + bf(y2 | subset(s2) ~ ...) + set_rescor(FALSE); docs/subset.md)")


def _split_subset(text: str) -> tuple[str, str | None]:
    """A formula of a list (F06c) without its ``subset()`` term, and the term's indicator column (None without one): brms's
    ``y | subset(sub1) ~ rhs`` is ``("y ~ rhs", "sub1")``. The term must stand alone on the left-hand side, its argument a column."""
    if not _has_subset(text):
        return text, None
    m = _LHS_RE.match(text)
    if m is None:
        raise FormulaError(f"{text!r}: subset() belongs to a response's left-hand side, '<response> | subset(<indicator>) ~ <terms>'")
    response, terms, rhs = m.groups()
    pieces = _split_top_level(terms)
    found = [p for p in pieces if _SUBSET_TERM_RE.match(p)]
    others = [p for p in pieces if not _SUBSET_TERM_RE.match(p)]
    if len(found) > 1:
        raise FormulaError(f"{text!r}: subset() appears more than once on the left-hand side")
    if others:
        raise FormulaError(f"{text!r}: subset() together with {', '.join(others)} on the left-hand side is not supported; "
                           "a response with subset() takes no other addition term here (docs/subset.md, divergence D160)")
    inside = _SUBSET_TERM_RE.match(found[0]).group(1).strip()
    if not re.fullmatch(_IDENT, inside):
        raise FormulaError(f"{text!r}: subset() takes the name of a 0/1 or logical column of the data, got {inside!r}; brms's "
                           "subset(<expression>) is not supported: compute the indicator as a column (divergence D160)")
    return f"{response} ~ {rhs}", inside


def subset_rows(data: pd.DataFrame, column: str, response: str = "", *, where: str = "data") -> np.ndarray:
    """The rows of brms's ``subset(column)`` (F06c), as 0-based positions in ``data``: those whose indicator is 1 or True, in data
    order. The indicator must be a logical column or a numeric one of 0 and 1, without missing values (brms refuses a missing
    value and text, and takes any non-zero number as TRUE: here anything but 0, 1, True and False is refused; divergence D160),
    and at least one row must be in the subset (brms: "All rows of 'data' were removed via 'subset'")."""
    what = f"subset({column})" + (f" of the response {response!r}" if response else "")
    if column not in data.columns:
        raise FormulaError(f"{what}: {where} lacks the indicator column {column!r}")
    s = data[column]
    numeric = pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)
    if s.isna().any() or not (pd.api.types.is_bool_dtype(s) or (numeric and s.isin([0, 1]).all())):
        shown = sorted({repr(v) for v in s.tolist()})[:6]
        raise FormulaError(f"{what}: the {where} column {column!r} must be logical (True/False) or numeric 0/1 without missing "
                           f"values, got the values {shown}: brms refuses a missing value and text, and takes any non-zero number "
                           "as TRUE, which is refused here (docs/subset.md, divergence D160)")
    rows = np.flatnonzero(s.to_numpy() == 1)
    if rows.size == 0:
        raise FormulaError(f"{what}: no row of the {where} is in the subset"
                           + (" (brms: \"All rows of 'data' were removed via 'subset'\")" if where == "data" else ""))
    return rows


def _check_subset_levels(spec: ModelSpec, data: pd.DataFrame, rows: np.ndarray, response: str, column: str) -> None:
    """F06c: brms gives a response's group-level term a level for every level of its grouping factor in the data (``N_1`` counts
    them, probed: with a level missing from the response's subset, or present only outside it, its effect is informed by its prior
    alone), while the response's model here has the levels of its own rows; a level the subset does not reach is refused."""
    for t in spec.group_terms:
        full = data.loc[data[list(t.operands)].notna().all(axis=1)]          # brms's levels: the data's, missing labels aside
        absent = sorted(set(group_labels(t.factor, full)) - set(group_labels(t.factor, data.iloc[rows])))
        if absent:
            raise FormulaError(f"the response {response!r}: the level(s) {absent[:8]} of the grouping factor {t.factor!r} occur in the "
                               f"data but in no row of subset({column}): brms gives every level of the data a group-level effect of "
                               f"{response!r}, which only its prior informs where the subset has no row; this is not supported here "
                               "(docs/subset.md, divergence D161): drop those rows, or the levels, from the data")


def _parse_separate(formulas, data, family, categorical, *, rescor) -> MultivariateSpec:
    """A formula per response without ``mi()`` (F06c): each ``'<response> ~ <terms>'`` or ``'<response> | subset(<indicator>) ~
    <terms>'`` parsed as the univariate formula it is, without its ``subset()`` term, and validated against the rows of its
    subset (every row without one), the responses distinct. G08: ``rescor=True`` without ``subset()``, ``weights()`` (one column
    for every response under ``rescor``) and correlation IDs across responses (``_strip_formula_ids``, ``_cross_ids``). Refused,
    naming the construct: ``rescor=True`` with ``subset()`` (brms refuses it), other addition terms, ``me()`` (D73), ``mo()``,
    ``gp()`` (D56), autocorrelation (D70) and SAR terms (D143), ``fcor()`` (D144), ``car()`` (D141), ``unstr()`` and ``cosy()``
    (D146), ``mm()`` and ``gr(by = )`` (D104, D105), a distributional formula, a response read by another formula, and an
    indicator that is a response."""
    from autobayes.families import get_family

    texts = list(formulas)
    if len(texts) < 2:
        raise FormulaError(f"{formulas!r}: one formula is a univariate model; a multivariate model is {SEPARATE_FORM}")
    if any(MVBIND_RE.match(f) for f in texts):
        raise FormulaError(f"{formulas!r}: mvbind() inside a list of formulas is not supported; {SEPARATE_FORM}")
    if rescor is None:
        raise ValueError("a multivariate model needs rescor=True or rescor=False (divergence D58); a formula per response takes "
                         "rescor=False (brms's set_rescor(FALSE); docs/subset.md)")
    if not isinstance(rescor, (bool, np.bool_)):
        raise ValueError(f"rescor must be True or False, got {rescor!r}")
    split = [_split_subset(text) for text in texts]
    if rescor and any(column is not None for _, column in split):
        raise FormulaError(f"{formulas!r}: subset() with a residual correlation (rescor=True, brms's set_rescor(TRUE)) is refused, "
                           "as brms refuses it (\"Only 'se', 'weights', 'mi' are supported addition arguments when 'rescor' is "
                           "estimated.\"): pass rescor=False (docs/subset.md, divergence D160)")
    found = []                                                            # G08: each formula's correlation IDs, stripped
    for i, (stripped, column) in enumerate(split):
        text, ids = _strip_formula_ids(stripped, repr(formulas))
        if ids and column is not None:
            raise FormulaError(f"{formulas!r}, the formula {texts[i]!r}: a correlation ID across responses with subset() is not "
                               f"supported ({ID_ROW})")
        split[i], found = (text, column), found + [ids]
    fam = get_family(family)
    specs0 = []
    for text, (stripped, _) in zip(texts, split):
        try:
            specs0.append(parse_formula(stripped, None, family, None))   # the structure, before the data and the declarations
        except FormulaError as e:
            raise FormulaError(f"{formulas!r}, the formula {text!r}: {e}") from None
        refuse_products(specs0[-1:], "a multivariate model")              # G03: verified in univariate models only
    responses = tuple(s.response for s in specs0)
    dup = sorted({r for r in responses if responses.count(r) > 1})
    if dup:
        raise FormulaError(f"{formulas!r}: the response(s) {dup} have more than one formula; give each response one")
    parameters = {p for p in (fam.scale_parameter, *fam.distributional_parameters) if p is not None}
    for s in specs0:
        if s.response in parameters:
            raise FormulaError(f"{s.formula!r}: a distributional formula in a multivariate model is not supported "
                               f"({s.response!r} is a parameter of the {fam.name} family, not a response; docs/multivariate.md, "
                               "divergence D56)")
        others = [a.term for a in s.additions if a.kind != "weights"]  # G08: weights() is taken
        if others or s.trials is not None:
            raise FormulaError(f"{s.formula!r}: addition terms ({', '.join(others or ['trials()'])}) on a response of a multivariate model "
                               "are not supported (docs/multivariate.md, divergence D56)")
        if s.custom_group_terms:                                      # D08: verified in univariate models only
            raise FormulaError(f"{s.formula!r}: '{s.custom_group_terms[0].canonical}' in a multivariate model is not supported"
                               " (docs/multi-membership-and-by.md, divergences D104, D105)")
        if extended_group_syntax(s) is not None:                      # G05a: verified in univariate models only
            raise FormulaError(f"{s.formula!r}: {extended_group_syntax(s)} in a multivariate model is not supported "
                               "(docs/group-term-syntax.md, divergence D183)")
        if s.me_terms:
            raise FormulaError(f"{s.formula!r}: a measurement-error term in a multivariate model is not supported; me() "
                               "belongs to a univariate Gaussian model here (docs/missing-and-measurement-error.md, divergence D73)")
        if s.mo_terms:
            raise FormulaError(f"{s.formula!r}: a monotonic term in a multivariate model is not supported (docs/multivariate.md, "
                               "divergence D56)")
        if s.gp_terms:
            raise FormulaError(f"{s.formula!r}: the Gaussian process {s.gp_terms[0].term!r} in a multivariate model is not supported"
                               "; write it in a univariate model (docs/multivariate.md, divergence D56)")
        if s.autocor is not None:
            raise FormulaError(f"{s.formula!r}: the autocorrelation term {s.autocor.term!r} in a multivariate model is not supported"
                               " (brms gives each response its own ARMA term, ar_y and ar_y2); write it in a univariate model "
                               "(docs/autocorrelation.md, divergence D70)")
        if getattr(s, "sar", None) is not None:
            raise FormulaError(f"{s.formula!r}: the SAR term {s.sar.term!r} in a multivariate model is not supported; "
                               "write it in a univariate model (docs/spatial-sar.md, divergence D143)")
        if s.fcor is not None:                                        # F03a, as C06's list of formulas refuses it
            raise FormulaError(f"{s.formula!r}: {s.fcor.term} in a multivariate model is not supported; write it in a "
                               "univariate model (docs/fcor.md, divergence D144)")
        if s.car is not None:                                         # F02a, likewise
            raise FormulaError(f"{s.formula!r}: the CAR term {s.car.term!r} in a multivariate model is not supported; write "
                               "it in a univariate model (docs/spatial-car.md, divergence D141)")
        if s.timecor is not None:                                     # F03b, likewise
            raise FormulaError(f"{s.formula!r}: the residual correlation term {s.timecor.term!r} in a multivariate model is not supported"
                               "; write it in a univariate model (docs/unstr-cosy.md, divergence D146)")
        reads = {*s.pop_columns, *(c for t in s.group_terms for c in t.operands), *(s.offset.columns if s.offset is not None else ())}
        plain = sorted(reads & (set(responses) - {s.response}))
        if plain:
            raise FormulaError(f"{s.formula!r}: the response(s) {plain} of another formula cannot also be predictors, grouping factors or "
                               "offsets here (a response's missing values are modelled with mi(), docs/missing-and-measurement-error.md)")
        weighted = sorted({c for a in s.additions for c in a.columns} & set(responses))
        if weighted:                                                  # G08
            raise FormulaError(f"{s.formula!r}: the response(s) {weighted} cannot also be the weights of a response")
    if rescor:                                                        # G08: the joint density is weighted once
        _check_rescor_weights({s.response: s for s in specs0}, responses, repr(formulas))
    weighted = [s.response for s in specs0 if s.addition("weights") is not None]
    if weighted and any(column is not None for _, column in split):   # G08: not verified together
        raise FormulaError(f"{formulas!r}: weights() (on {weighted}) in a multivariate model with subset() is not supported ({WEIGHTS_ROW})")
    cross = _cross_ids(dict(zip(responses, found)), responses, repr(formulas))
    subsets = {r: column for r, (_, column) in zip(responses, split) if column is not None}
    indicator_responses = sorted(set(subsets.values()) & set(responses))
    if indicator_responses:
        raise FormulaError(f"{formulas!r}: the subset() indicator(s) {indicator_responses} are responses of the model; an indicator "
                           "is a 0/1 or logical column of the data")
    spec0 = MultivariateSpec(tuple(t.strip() for t in texts), responses, "", MappingProxyType({s.response: s for s in specs0}),
                             bool(rescor), MappingProxyType(subsets), cross_ids=cross)
    cats = response_categorical(spec0, categorical)
    specs = {}
    for text, (stripped, column), r in zip(texts, split, responses):
        frame = data if data is None or column is None else data.iloc[subset_rows(data, column, r)]   # its subset's rows
        try:
            specs[r] = parse_formula(stripped, frame, family, cats[r])
        except FormulaError as e:
            on = "" if column is None else f", on the rows of subset({column})"
            raise FormulaError(f"{formulas!r}, the formula of the response {r!r} ({text!r}{on}): {e}") from None
        fitted = next((t.basis for t in specs[r].expression_terms if t.transform.needs_fit), None) if column is not None else None
        if fitted is not None:                                       # G04: fitted on which rows brms fits them is not verified
            raise FormulaError(f"{formulas!r}, the formula of the response {r!r} ({text!r}): {fitted!r} with subset({column}) is not "
                               "supported: its fitted parameters (centre and scale, polynomial coefficients, levels) would be the "
                               "subset's, which no frozen case verifies (docs/expressions.md, divergence D182)")
    return replace(spec0, specs=MappingProxyType(specs))


# -- the joint likelihood --------------------------------------------------------------------------------------
def multi_normal_cholesky_logp(value, mu, sigma, L):
    """Stan's ``multi_normal_cholesky_lpdf(y | mu, diag(sigma) L)`` per row, with its constant: ``value`` and ``mu`` are
    ``(..., K)``, ``sigma`` ``(..., K)``, ``L`` ``(..., K, K)`` lower triangular. With ``LS = diag(sigma) L`` the residual is
    solved forward, ``e_k = (r_k - sum_{j<k} LS[k, j] e_j) / LS[k, k]``, and the density is ``-K/2 log(2 pi) - sum_k
    log LS[k, k] - |e|^2 / 2``, as Stan computes it."""
    import pytensor.tensor as pt

    K = int(value.type.shape[-1])
    LS = sigma[..., :, None] * L
    resid = value - mu
    es = []
    for k in range(K):
        acc = resid[..., k]
        for j in range(k):
            acc = acc - LS[..., k, j] * es[j]
        es.append(acc / LS[..., k, k])
    quad = es[0] ** 2
    logdet = pt.log(LS[..., 0, 0])
    for k in range(1, K):
        quad = quad + es[k] ** 2
        logdet = logdet + pt.log(LS[..., k, k])
    return np.float64(-0.5 * K * np.log(2.0 * np.pi)) - logdet - np.float64(0.5) * quad


def weighted_logp(weights):
    """G08: ``multi_normal_cholesky_logp`` with each row's joint log density multiplied by its weight, brms's ``weights[n] *
    multi_normal_cholesky_lpdf(Y[n] | Mu[n], LSigma)`` under ``weights()`` and a residual correlation (``weights`` the ``(n,)``
    weights of every response)."""
    w = np.asarray(weights, dtype=np.float64)

    def logp(value, mu, sigma, L):
        return w * multi_normal_cholesky_logp(value, mu, sigma, L)

    return logp


def _ancestors(var) -> set:
    """G08: the variables ``var`` is computed from (PyTensor's ``ancestors``)."""
    from pytensor.graph.traversal import ancestors

    return set(ancestors([var]))


def _rename_dim(fg, old: str, new: str, coords) -> None:
    """F06c: the coordinate ``old`` of a response's function graph (``pymc.model.fgraph``) renamed ``new``, with ``coords`` as its
    values: every model variable on it (its dims are string constants among its node's inputs) and the graph's coordinates and
    dimension lengths."""
    import pytensor
    from pymc.model.fgraph import ModelVar
    from pytensor.graph.basic import Constant

    for node in list(fg.apply_nodes):
        if isinstance(node.op, ModelVar):
            for i, inp in enumerate(node.inputs):
                if isinstance(inp, Constant) and isinstance(inp.data, str) and inp.data == old:
                    fg.change_node_input(node, i, pytensor.as_symbolic(new), import_missing=True)
    fg._coords = {(new if k == old else k): (tuple(int(c) for c in coords) if k == old else v) for k, v in fg._coords.items()}
    fg._dim_lengths = {(new if k == old else k): v for k, v in fg._dim_lengths.items()}


@dataclass(frozen=True)
class MultivariateDesign:
    """What a multivariate model was fit to (C01): ``Y`` the ``(n, K)`` responses in ``responses`` order (brms's ``Y``) and
    each response's own ``Design`` (``designs``: its univariate model's, the same design columns, group blocks and centring for
    every response, which share the right-hand side and the rows). With ``subset()`` (F06c) ``rows`` maps each response to its
    rows of the data (0-based, in data order; every row without ``subset()``), ``Y`` holds each response on its rows and NaN on
    the others, and each design is its response's model's, on its rows; None without ``subset()``, every response on every row."""
    responses: tuple[str, ...]
    Y: np.ndarray
    designs: Mapping[str, object]
    rows: Mapping[str, np.ndarray] | None = None

    @property
    def n(self) -> int:
        return int(self.Y.shape[0])


class MultivariateModel:
    """brms's multivariate Gaussian model ``mvbind(y, y2) ~ ...`` with or without residual correlation (module docstring).

    Attributes
    ----------
    family, family_spec, link : the family (Gaussian) and its spec, as for ``Model``
    responses : tuple[str]          the responses in the order written (the ``response`` dimension of ``predict()``)
    rescor : bool                   whether the residuals are correlated (``Lrescor``, ``rescor`` in the graph)
    spec : MultivariateSpec         the formula, with each response's ``ModelSpec``
    priors : Priors                 ``Priors(resp={...}, cor=..., rescor=...)``
    submodels : {response: Model}   each response's univariate model (its own Bambi graph, design and names)
    design : MultivariateDesign     ``Y`` and each response's design
    pymc_model : pymc.Model         the assembled graph that is sampled; inspect it directly
    id_cor_priors : {factor: Prior} G08: the LKJ prior of each correlation ID across responses (``id_names``), by factor
    joint_weights : ndarray | None  G08: the weights of each row's joint density under a residual correlation, or None
    """

    def __init__(self, formula, data: pd.DataFrame, priors: Priors, *, family: str = "gaussian", rescor: bool | None = None,
                 center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False):
        from autobayes.column_types import typed_categorical
        from autobayes.missing_rows import check_dropna
        from autobayes.model import Model

        fam = get_family(family)
        self.dropna = check_dropna(dropna)                 # D174: one set of rows for every response (docs/missing-rows.md)
        self.dropped_na_rows: tuple[int, ...] = ()
        # D172: an undeclared pandas Categorical or boolean predictor is declared by its dtype, as in a univariate model
        categorical = typed_categorical(data, categorical, lambda cat: parse_multivariate(formula, None, family, cat, rescor=rescor))
        spec0 = parse_multivariate(formula, None, family, categorical, rescor=rescor)   # the structure, before the data
        if not isinstance(priors, Priors):
            raise PriorError(f"priors must be an ab.Priors, got {type(priors).__name__}")
        priors.validate_multivariate(spec0, fam)
        self.id_cor_priors, own_cor = split_cor(priors, spec0)     # G08: the IDs across responses', and the responses' own
        own = replace(priors, cor=own_cor) if spec0.cross_ids else priors
        self.family, self.family_spec, self.priors = family, fam, priors
        self.responses: tuple[str, ...] = spec0.responses
        self.rescor = bool(rescor)
        self.center_predictors = bool(center_predictors)
        cats = response_categorical(spec0, categorical)          # each response's declarations (C06: its own main effects')
        note = None
        if self.dropna:                                          # D174: but the missing values mi() models (C06)
            import warnings

            from autobayes.missing_rows import drop_missing_multivariate, dropped_warning

            n = len(data)
            data, self.dropped_na_rows, where = drop_missing_multivariate(spec0, data)
            if self.dropped_na_rows:
                note = dropped_warning(self.dropped_na_rows, n, where)
                warnings.warn(note, UserWarning, stacklevel=3)
        subsets = spec0.subsets                                  # F06c: a response with subset() is built on its subset's rows
        rows = {r: subset_rows(data, subsets[r], r) for r in subsets}
        self.submodels = MappingProxyType({r: Model(spec0.formulas[r], data.iloc[rows[r]] if r in subsets else data,
                                                    own.response_priors(r), family=family,
                                                    center_predictors=center_predictors, categorical=cats[r], imputed=spec0.imputed)
                                           for r in self.responses})
        subs = list(self.submodels.values())
        for r in subsets:                                        # F06c: brms's levels of a grouping factor are the data's
            _check_subset_levels(self.submodels[r].spec, data, rows[r], r, subsets[r])
        if not subsets and (len({sub.design.n for sub in subs}) != 1 or any(len(sub.dropped_rows) for sub in subs)):
            raise RuntimeError("the responses' models were fit to different rows")
        self.spec = MultivariateSpec(spec0.formula, spec0.responses, spec0.rhs,
                                     MappingProxyType({r: sub.spec for r, sub in self.submodels.items()}), self.rescor, subsets,
                                     cross_ids=spec0.cross_ids)
        # G08: with a residual correlation each row's joint density is weighted once, by the responses' one weights column
        self.joint_weights = subs[0].design.weights if self.rescor else None
        # one right-hand side: the responses' centring; a formula per response (C06): the flag, which each response's table
        # applies when its formula has an intercept
        self.effective_center_predictors = self.center_predictors if self.spec.separate else subs[0].effective_center_predictors
        self.warnings: list[str] = list(dict.fromkeys(w for sub in subs for w in sub.warnings))
        if note is not None:                                     # D174: first, as the drop came first
            self.warnings.insert(0, note)
        designs = MappingProxyType({r: sub.design for r, sub in self.submodels.items()})
        if subsets:                                              # F06c: each response on its rows, NaN elsewhere
            every = {r: rows.get(r, np.arange(len(data))) for r in self.responses}
            Y = np.full((len(data), len(self.responses)), np.nan)
            for k, (r, sub) in enumerate(self.submodels.items()):
                Y[every[r], k] = sub.design.y
            self.design = MultivariateDesign(self.responses, Y, designs, MappingProxyType(every))
        else:
            self.design = MultivariateDesign(self.responses, np.column_stack([sub.design.y for sub in subs]), designs)
        self._names: dict[str, dict[str, str]] = {}       # per response: its univariate model's PyMC name -> the graph's
        self.pymc_model = self._assemble()
        self._check_graph()
        self._target_identity = MappingProxyType(_freeze_target(self))

    # -- names ---------------------------------------------------------------------------------------------
    @property
    def link(self) -> str:
        return self.family_spec.link

    @staticmethod
    def variable(response: str, name: str) -> str:
        """The multivariate graph's name of a variable of ``response``'s univariate model: ``y_Intercept`` for its
        ``Intercept``, ``y_1|g_sigma`` for its ``1|g_sigma`` (value variables alike, ``y_sigma_log__``)."""
        return f"{response}_{name}"

    def value_name(self, response: str, name: str) -> str:
        """The graph's name of a value variable of ``response``'s model: ``variable(response, name)``, but the missing values
        ``Ymi_<col>`` (C06) keep their name, one variable of the model shared by the responses that hold them."""
        shared = {roles["values"] for roles in self.submodels[response].missing_names.values()}
        return name if name in shared else self.variable(response, name)

    @property
    def missing_names(self) -> dict[str, dict[str, str]]:
        """The missing values of the model (C06), by column: ``"values"`` the free variable ``Ymi_<col>`` (brms's
        ``Ymi_<col>``, flat), ``"coordinate"`` its dimension, whose coordinates are the missing rows (0-based), and
        ``"response"`` the response whose formula imputes the column (``<col> | mi() ~ ...``). Empty without one."""
        out = {}
        for r, sub in self.submodels.items():
            for col, roles in sub.missing_names.items():
                out.setdefault(col, {**roles, "response": col})
        return out

    @property
    def observed_names(self) -> tuple[str, ...]:
        """The graph's observed variables: the joint ``mvbind(y, y2)`` with ``rescor=True``, each response (``y``, ``y2``)
        otherwise."""
        return (f"mvbind({', '.join(self.responses)})",) if self.rescor else self.responses

    @property
    def observation_dims(self) -> dict[str, str]:
        """Each response's observation coordinate in the graph: ``__obs__``, shared by the responses on the same rows, or with
        ``subset()`` (F06c) one per response, ``__obs_<response>__``, whose coordinates are the response's rows of the data."""
        return {r: f"__obs_{r}__" if self.spec.subsets else "__obs__" for r in self.responses}

    @property
    def response_names(self) -> dict[str, dict[str, str]]:
        """Each response's roles (its univariate model's ``names``: ``intercept``, ``b:x``, ``sigma``, ``sd``, ``linpred``,
        ...) with the multivariate graph's names, ``{"y": {"intercept": "y_Intercept", ...}, ...}``; ``"y"`` is the
        observed variable that holds the response."""
        out = {}
        for r, sub in self.submodels.items():
            roles = {role: self._names[r][name] for role, name in sub.names.items() if role not in ("y", "mean") and name in self._names[r]}
            roles["y"] = self.observed_names[0] if self.rescor else r
            out[r] = roles
        return out

    @property
    def names(self) -> dict[str, str]:
        """``"mean"`` (the name ``predict(kind="mean")`` gives the expected responses, with a ``response`` dimension), the
        residual correlation's ``"Lrescor"`` and ``"rescor"`` and the joint observed ``"y"`` (``rescor=True``), and every
        response's roles as ``"<response>:<role>"`` (``"y:intercept"``: ``"y_Intercept"``; ``response_names``)."""
        out = {"mean": self.family_spec.mean_name}
        if self.rescor:
            out.update(y=self.observed_names[0], Lrescor="Lrescor", rescor="rescor")
        for r, roles in self.response_names.items():
            out.update({f"{r}:{role}": name for role, name in roles.items()})
        for g, roles in self.id_names.items():                # G08: "L:g" and "cor:g" of a correlation ID across responses
            out.update({f"L:{g}": roles["L"], f"cor:{g}": roles["cor"]})
        return out

    @property
    def id_names(self) -> dict[str, dict[str, object]]:
        """G08: each correlation ID across responses, by grouping factor: ``L`` the Cholesky factor of the correlation matrix of the
        responses' varying intercepts (``L|g``, brms's ``L_m``, sampled as ``L|g_cholesky_corr__``), ``cor`` the matrix (``cor|g``,
        brms's ``Cor_m``), ``coordinate`` / ``coordinate_T`` its dimensions (``g__id_dim``, labelled ``y_Intercept``, ...), ``id``
        the ID and ``effects`` the graph's names of the correlated effects, in order (``y_1|g``, ``y2_1|g``). Empty without one."""
        out = {}
        for cid in self.spec.cross_ids:
            g = cid.factor
            out[g] = {"L": f"L|{g}", "cor": f"cor|{g}", "coordinate": f"{g}__id_dim", "coordinate_T": f"{g}__id_dim_T", "id": cid.id,
                      "effects": tuple(self._names[r][self.submodels[r].group_coefficient_names[(g, c)]["group_effect"]]
                                       for r, c in cid.coefs)}
        return out

    @property
    def target_identity(self) -> dict[str, str]:
        """Frozen built-target digest (the responses' own targets, the responses' order, ``Y`` and the residual correlation);
        a defensive copy."""
        return dict(self._target_identity)

    # -- the graph -----------------------------------------------------------------------------------------
    def _assemble(self):
        """The responses' Bambi graphs as one PyMC model: each converted to a function graph, its variables renamed after the
        response (``variable``) and its coordinates checked against the other responses', all of them made one graph (the
        dimension lengths of a shared coordinate unified) and rebuilt as a model; then, with ``rescor=True``, ``Lrescor`` (a
        free ``K x K`` correlation Cholesky factor under its LKJ prior, on Stan's ``cholesky_corr`` coordinates), ``rescor``
        and the joint likelihood replacing the responses' own."""
        import pymc as pm
        import pytensor.tensor as pt
        from pymc.model.fgraph import ModelFreeRV, ModelObservedRV, fgraph_from_model, model_from_fgraph
        from pytensor.graph import FunctionGraph

        outputs, coords, lengths, aliases = [], {}, {}, []
        shared, replaced = {}, []                            # C06: a column's missing values, one variable shared by the responses
        for r, sub in self.submodels.items():
            fg, _ = fgraph_from_model(sub.pymc_model, inlined_views=True)
            if self.spec.subsets:                            # F06c: the response's rows are its own coordinate
                _rename_dim(fg, "__obs__", self.observation_dims[r], self.design.rows[r])
            renamed = {}
            missing = {roles["values"] for roles in sub.missing_names.values()}
            for out in fg.outputs:
                op = out.owner.op
                if isinstance(op, ModelObservedRV):
                    if self.rescor:                          # replaced by the joint likelihood
                        continue
                    renamed[out.name] = out.name             # the response's own likelihood keeps the response's name
                elif out.name in missing:                    # Ymi_<col> keeps brms's name; the first response's variable is the model's
                    renamed[out.name] = out.name
                    if out.name in shared:
                        replaced.append((out, shared[out.name]))
                        continue
                    shared[out.name] = out
                else:
                    new = self.variable(r, out.name)
                    renamed[out.name] = new
                    out.name = new
                    if isinstance(op, ModelFreeRV):          # its value variable: <name>_<transform>__, or the name itself
                        value, transform = out.owner.inputs[1], op.transform
                        value.name = new if transform is None else f"{new}_{transform.name}__"
                outputs.append(out)
            self._names[r] = renamed
            for dim, values in fg._coords.items():
                if dim in coords:
                    if tuple(coords[dim]) != tuple(values):
                        raise RuntimeError(f"the responses' graphs disagree on the coordinate {dim!r}: {coords[dim]} vs {values}")
                    aliases.append((fg._dim_lengths[dim], lengths[dim]))
                else:
                    coords[dim], lengths[dim] = values, fg._dim_lengths[dim]
        added = ([*self.observed_names, "Lrescor", "rescor", "Lrescor_cholesky_corr__", RESPONSE_DIM, RESPONSE_DIM_T]
                 if self.rescor else [])
        added += [n for roles in self.id_names.values()                                 # G08
                  for n in (roles["L"], roles["cor"], f"{roles['L']}_cholesky_corr__", roles["coordinate"], roles["coordinate_T"])]
        value_names = [o.owner.inputs[1].name for o in outputs                 # the value variables named apart from their variable
                       if isinstance(o.owner.op, ModelFreeRV) and o.owner.inputs[1].name != o.name]
        names = [o.name for o in outputs] + value_names + list(coords) + added
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise FormulaError(f"generated variable/coordinate name collision in the multivariate graph: {duplicates} (each response's "
                               "variables are named <response>_<name>); rename the conflicting column")
        merged = FunctionGraph(outputs=outputs[::-1], clone=False)   # (model_from_fgraph's toposort then lists the responses in order)
        for old, new in aliases:                             # one dimension length per shared coordinate
            if old in merged.variables:
                merged.replace(old, new, import_missing=True)
        for old, new in replaced:                            # C06: one variable of the missing values of each column
            if old in merged.variables:
                merged.replace(old, new, import_missing=True)
        merged._coords, merged._dim_lengths = coords, lengths
        if self.spec.cross_ids:                              # G08: the responses' varying intercepts in one correlation matrix
            self._correlate_across(merged)
        model = model_from_fgraph(merged, mutate_fgraph=True)
        if self.rescor:
            K = len(self.responses)
            with model:
                model.add_coord(RESPONSE_DIM, list(self.responses))
                model.add_coord(RESPONSE_DIM_T, list(self.responses))
                rv, transform, _ = cholesky_corr_rv("Lrescor", float(self.priors.rescor.params["eta"]), K)
                L = model.register_rv(rv, "Lrescor", dims=(RESPONSE_DIM, RESPONSE_DIM_T), default_transform=None, transform=transform)
                pm.Deterministic("rescor", L @ L.T, dims=(RESPONSE_DIM, RESPONSE_DIM_T))
                mu = pt.stack([model[self._names[r][sub.names["linpred"]]] for r, sub in self.submodels.items()], axis=-1)
                sigma = pt.stack([model[self._names[r][sub.names["sigma"]]] for r, sub in self.submodels.items()])
                logp = multi_normal_cholesky_logp if self.joint_weights is None else weighted_logp(self.joint_weights)
                pm.CustomDist(self.observed_names[0], mu, sigma, L, logp=logp, observed=self.design.Y,
                              signature="(k),(k),(k,k)->(k)", dims=("__obs__", RESPONSE_DIM))
        return model

    def _correlate_across(self, fgraph) -> None:
        """G08: brms's correlation ID across responses as an edit of the assembled function graph, as A02 and D08 correlate a
        univariate model's coefficients (``Model._correlate_group_effects``): each response's model built its varying intercept
        ``(1 | g)`` as ``r_k = sd_k * z_k`` (``y2_1|g = y2_1|g_sigma * y2_1|g_offset``); brms's ``r = (diag(sd) L z)^T`` over the
        responses is ``r_k = sd_k * sum_{j <= k} L[k, j] z_j``, so the first response's effects are unchanged (``L[0, 0] = 1``)
        and each later one's deterministic is replaced, everywhere it is used, by the correlated expression. ``L|g`` is a free
        ``M x M`` Cholesky factor under ``lkj_corr_cholesky(eta)`` (the ID's ``cor`` prior), sampled on Stan's ``cholesky_corr``
        coordinates, and ``cor|g = L L^T`` a deterministic (brms's ``Cor_m``), both on the ID's coordinates."""
        import pytensor
        from pymc.model.fgraph import model_deterministic, model_free_rv
        from pymc.pytensorf import toposort_replace

        by_name = {o.name: o for o in fgraph.outputs}
        replacements, added = [], []
        for cid in self.spec.cross_ids:
            roles, M = self.id_names[cid.factor], len(cid.coefs)
            rv, transform, value = cholesky_corr_rv(roles["L"], float(self.id_cor_priors[cid.factor].params["eta"]), M)
            for dim in (roles["coordinate"], roles["coordinate_T"]):
                fgraph._coords[dim] = cid.labels
                fgraph._dim_lengths[dim] = pytensor.shared(M, name=dim)
            L = model_free_rv(rv, value, transform, roles["coordinate"], roles["coordinate_T"])
            L.name = roles["L"]
            cor = L @ L.T
            cor_det = model_deterministic(cor, roles["coordinate"], roles["coordinate_T"])
            cor_det.name = roles["cor"]
            added += [L, cor_det]
            coef = [{k: self._names[r][v] if k != "coordinate" else v for k, v in self.submodels[r].group_coefficient_names[(cid.factor, c)].items()
                     if k in ("sd", "offset", "group_effect", "coordinate")} for r, c in cid.coefs]
            zs = [by_name[c["offset"]] for c in coef]
            for k in range(1, M):
                expr = by_name[coef[k]["sd"]] * sum(L[k, j] * zs[j] for j in range(k + 1))
                det = model_deterministic(expr, coef[k]["coordinate"])
                det.name = coef[k]["group_effect"]
                replacements.append((by_name[coef[k]["group_effect"]], det))
        toposort_replace(fgraph, tuple(replacements))
        for var in added:
            fgraph.add_output(var)

    def _check_graph(self) -> None:
        """The assembled graph holds exactly the responses' variables under their new names with their transforms, the
        observed variables ``observed_names``, and with ``rescor=True`` ``Lrescor`` (free, ``K x K``, sampled on the
        ``K (K - 1) / 2`` cholesky_corr coordinates) and the deterministic ``rescor``, each on the response coordinates."""
        pmm, K = self.pymc_model, len(self.responses)
        expected = {self.value_name(r, v.name) for r, sub in self.submodels.items() for v in sub.pymc_model.value_vars}
        expected |= {"Lrescor_cholesky_corr__"} if self.rescor else set()
        expected |= {f"{roles['L']}_cholesky_corr__" for roles in self.id_names.values()}      # G08
        got = {v.name for v in pmm.value_vars}
        if got != expected:
            raise RuntimeError(f"the multivariate graph's value variables {sorted(got)} differ from the responses' {sorted(expected)}")
        for r, sub in self.submodels.items():
            for rv in sub.pymc_model.free_RVs:
                new = pmm[self._names[r][rv.name]]
                if type(pmm.rvs_to_transforms[new]) is not type(sub.pymc_model.rvs_to_transforms[rv]) or new.type.shape != rv.type.shape:
                    raise RuntimeError(f"{self._names[r][rv.name]!r} is not {r}'s {rv.name!r} with its transform and shape")
        if sorted(rv.name for rv in pmm.observed_RVs) != sorted(self.observed_names):
            raise RuntimeError(f"the multivariate graph observes {[rv.name for rv in pmm.observed_RVs]}, expected {list(self.observed_names)}")
        for r, dim in self.observation_dims.items():          # F06c: each response observed on its own rows
            if self.spec.subsets and (tuple(pmm.named_vars_to_dims.get(r, ())) != (dim,)
                                      or list(pmm.coords[dim]) != self.design.rows[r].tolist()):
                raise RuntimeError(f"the observed {r!r} is not on its rows, the coordinate {dim!r}")
        if self.rescor:
            L = pmm["Lrescor"]
            if L not in pmm.free_RVs or tuple(L.type.shape) != (K, K):
                raise RuntimeError(f"'Lrescor' is not a free {K} x {K} variable of the graph")
            if tuple(pmm.rvs_to_values[L].type.shape) != (n_free(K),):
                raise RuntimeError(f"'Lrescor' is not sampled on the {n_free(K)} cholesky_corr coordinates")
            for name in ("Lrescor", "rescor"):
                if tuple(pmm.named_vars_to_dims.get(name, ())) != (RESPONSE_DIM, RESPONSE_DIM_T):
                    raise RuntimeError(f"PyMC dims of {name!r} are not the response coordinates")
            if tuple(pmm.named_vars_to_dims.get(self.observed_names[0], ())) != ("__obs__", RESPONSE_DIM):
                raise RuntimeError(f"PyMC dims of {self.observed_names[0]!r} are not (__obs__, {RESPONSE_DIM})")
        for cid in self.spec.cross_ids:                       # G08: L|g free on the ID's coordinates, the later effects correlated
            roles, M = self.id_names[cid.factor], len(cid.coefs)
            L = pmm[roles["L"]]
            if (L not in pmm.free_RVs or tuple(L.type.shape) != (M, M) or tuple(pmm.rvs_to_values[L].type.shape) != (n_free(M),)
                    or any(tuple(pmm.named_vars_to_dims.get(n, ())) != (roles["coordinate"], roles["coordinate_T"]) for n in (roles["L"], roles["cor"]))
                    or tuple(pmm.coords[roles["coordinate"]]) != cid.labels):
                raise RuntimeError(f"{roles['L']!r} is not a free {M} x {M} correlation Cholesky factor on {roles['coordinate']!r}")
            for name in roles["effects"][1:]:
                if L not in _ancestors(pmm[name]):
                    raise RuntimeError(f"the effects {name!r} are not correlated through {roles['L']!r}")

    # -- points --------------------------------------------------------------------------------------------
    def value_point(self, *, resp: Mapping[str, Mapping], Lrescor=None, Ymi=None, L=None) -> dict[str, np.ndarray]:
        """A point in the graph's value space: ``resp`` maps each response to the keyword arguments of its univariate
        model's ``Model.value_point`` (``{"y": {"intercept": ..., "b": ..., "sd": ..., "z": ..., "sigma": ...}, ...}``),
        ``Lrescor`` is the Cholesky factor of the residual correlation matrix (lower triangular, unit-norm rows, in the
        responses' order), required exactly with ``rescor=True`` and stored on Stan's ``cholesky_corr`` coordinates. ``Ymi``
        (C06) gives the missing values of each column of ``missing_names`` (``{"xm": array(Nmi)}``, one per missing row; a
        single column also takes the bare array), which every response holding them shares: it is not part of ``resp``. ``L``
        (G08) gives the Cholesky factor of each correlation ID across responses by grouping factor (``{"g": array(M, M)}``, in
        the order of ``id_names[g]["effects"]``), required exactly when the model has one; each response's ``z`` holds its own
        standardised effects (brms's row of ``z_m``)."""
        if not isinstance(resp, Mapping) or set(resp) != set(self.responses):
            raise ValueError(f"resp must map exactly the responses {list(self.responses)} to their value_point arguments")
        missing = list(self.missing_names)
        if missing:
            if Ymi is None:
                raise ValueError(f"Ymi (the missing values of {missing}) is required")
            values = dict(Ymi) if isinstance(Ymi, Mapping) else ({missing[0]: Ymi} if len(missing) == 1 else None)
            if values is None or set(values) != set(missing):
                raise ValueError(f"Ymi must cover exactly the columns with missing values {missing}")
        elif Ymi is not None:
            raise ValueError("this model has no missing values; Ymi does not apply")
        point = {}
        for r, sub in self.submodels.items():
            if not isinstance(resp[r], Mapping):
                raise ValueError(f"resp[{r!r}] must be a mapping of the keyword arguments of {r}'s Model.value_point")
            if "Ymi" in resp[r]:
                raise ValueError(f"resp[{r!r}] holds Ymi: the missing values are the model's, shared by its responses; pass them at "
                                 "the top level, value_point(resp=..., Ymi={...})")
            args = {**resp[r], **({"Ymi": {c: values[c] for c in sub.missing_names}} if sub.missing_names else {})}
            point.update({self.value_name(r, name): value for name, value in sub.value_point(**args).items()})
        if self.rescor:
            if Lrescor is None:
                raise ValueError("Lrescor (the Cholesky factor of the residual correlation matrix) is required: rescor=True")
            point["Lrescor_cholesky_corr__"] = cholesky_corr_free(check_correlation_cholesky(Lrescor, len(self.responses), "Lrescor"))
        elif Lrescor is not None:
            raise ValueError("this model has no residual correlation (rescor=False); Lrescor does not apply")
        ids = self.id_names                                  # G08: each ID's L on Stan's cholesky_corr coordinates
        if ids and (not isinstance(L, Mapping) or set(L) != set(ids)):
            raise ValueError(f"L must map exactly the grouping factors of the correlation IDs across responses {list(ids)} to their "
                             "Cholesky factors")
        if not ids and L is not None:
            raise ValueError("this model has no correlation ID across responses; L does not apply")
        for g, roles in ids.items():
            point[f"{roles['L']}_cholesky_corr__"] = cholesky_corr_free(check_correlation_cholesky(L[g], len(roles["effects"]), f"L for {g!r}"))
        expected = {v.name for v in self.pymc_model.value_vars}
        if set(point) != expected:
            raise RuntimeError(f"value names {sorted(point)} do not match PyMC's {sorted(expected)}")
        return point

    def response_posterior(self, posterior, response: str):
        """The draws of ``response``'s variables in ``posterior`` (an xarray Dataset of this graph's names) under its
        univariate model's names, which ``Model``'s prediction code reads."""
        names = {new: old for old, new in self._names[response].items() if new in posterior.data_vars}   # C06: Ymi_<col> keeps its name
        return posterior[list(names)].rename({new: old for new, old in names.items() if new != old})

    # -- sampling (ab.fit) ---------------------------------------------------------------------------------
    def sample_posterior(self, *, inference_method: str = "pymc", cores=None, **kwargs):
        """What ``ab.fit()`` samples for a multivariate model, in place of Bambi's ``fit()`` (no one Bambi model holds this
        graph): ``pm.sample`` on ``pymc_model`` with ``nuts_sampler=inference_method`` ("pymc" or "numpyro"), recording the free
        variables and the deterministics but each response's linear predictor (Bambi leaves the likelihood's parameters out
        too), then ``clean_posterior``."""
        import pymc as pm

        linpred = {self._names[r][sub.names["linpred"]] for r, sub in self.submodels.items()}
        var_names = [v.name for v in self.pymc_model.free_RVs] + [d.name for d in self.pymc_model.deterministics if d.name not in linpred]
        with self.pymc_model:
            idata = pm.sample(var_names=var_names, nuts_sampler=inference_method, cores=cores, **kwargs)
        return self.clean_posterior(idata)

    def clean_posterior(self, idata):
        """A fit's posterior cleaned as Bambi cleans its own, response by response: the standardised group effects
        (``*_offset``) dropped, and each centred intercept moved to the raw scale, ``a = a_c - sum_p mean(x_p) b_p`` (brms's
        ``b_<response>_Intercept``), which ``predict()`` reads."""
        from autobayes.predict import _coefficient_draws

        post = idata.posterior
        post = post.drop_vars([v for v in post.data_vars if str(v).endswith("_offset")])
        for r, sub in self.submodels.items():
            if not (sub.spec.intercept and sub.design.centered and sub.spec.columns):
                continue
            name = self._names[r][sub.names["intercept"]]
            b = _coefficient_draws(self.response_posterior(post, r), sub)          # (chain, draw, P)
            post[name] = post[name].copy(data=post[name].to_numpy() - b @ sub.design.x_means)
        idata.posterior = post
        return idata

    def fit(self, **kwargs):
        from autobayes.inference import fit

        return fit(self, **kwargs)

    def predict(self, fit_result, **kwargs):
        return predict_multivariate(self, fit_result, **kwargs)

    def prior_predictive(self, **kwargs):
        raise ValueError("prior_predictive() of a multivariate model is not supported (docs/multivariate.md); draw from "
                         "each response's univariate model, model.submodels[response].prior_predictive(), which ignores the "
                         "residual correlation")

    # -- priors and description ----------------------------------------------------------------------------
    def prior_table(self) -> list[dict[str, str]]:
        """The priors in brms's table form: each response's rows under its ``resp``, the shared ``cor`` rows and ``rescor``."""
        return self.priors.table(self.spec, self.family_spec, center_predictors=self.effective_center_predictors)

    def describe(self) -> str:
        s, K, R = self.spec, len(self.responses), self.responses
        lines = [f"AutoBayes multivariate {self.family_spec.tag} model of {K} responses ({', '.join(R)}), "
                 + ("with residual correlation (brms set_rescor(TRUE))" if self.rescor else "without residual correlation (brms set_rescor(FALSE))"),
                 f"  formula   : {list(s.formula) if s.separate else s.formula}"]
        if s.canonical_formula != s.formula:
            lines += [f"  expanded  : {s.canonical_formula}"]
        lines += [f"  responses : {'; '.join(s.formulas.values())}   (one univariate model per response on the same rows and "
                  + ("its own formula: brms's bf(...) + bf(...))" if s.separate else "right-hand side: brms's mvbind())")
                  if not s.subsets else
                  f"  responses : {'; '.join(s.formulas.values())}   (one univariate model per response, of its own formula on the "
                  "rows of its subset: brms's bf(y | subset(s) ~ ...) + bf(...))",
                  f"  data      : n = {self.design.n} rows, each with its {K} responses" if not s.subsets else
                  f"  data      : n = {self.design.n} rows (brms's N); each response's likelihood covers the rows of its subset only"]
        if self.dropped_na_rows:                             # D174: the rows dropna=True dropped, for every response
            lines.append(f"  dropped   : {len(self.dropped_na_rows)} row(s) with missing values (dropna=True, positions "
                         f"{list(self.dropped_na_rows)[:10]}{'...' if len(self.dropped_na_rows) > 10 else ''})")
        for r, column in s.subsets.items():                  # F06c: brms's N_<response> rows of each subset
            rows = self.design.rows[r]
            lines.append(f"  subset    : {r} on the {rows.size} rows with {column} = 1 (brms's N_{r}; 0-based {rows[:12].tolist()}"
                         f"{'...' if rows.size > 12 else ''}), the rows of its observed variable {self.observation_dims[r]!r}")
        lines.append("")
        if self.rescor:
            eta = float(self.priors.rescor.params["eta"])
            lines += ["Residual correlation",
                      f"  ({', '.join(f'{r}_i' for r in R)}) ~ MultiNormal(({', '.join(f'mu_{r},i' for r in R)}), Sigma), "
                      f"Sigma = diag(sigma) Rescor diag(sigma), sigma = ({', '.join(f'sigma_{r}' for r in R)}):",
                      "  brms's multi_normal_cholesky(Y_i | Mu_i, diag(sigma) Lrescor); each response's mu_i and sigma are its own model's "
                      "(below), whose Normal likelihood this joint one replaces (it is each response's marginal).",
                      f"  Rescor = Lrescor Lrescor^T is the correlation matrix of the residuals y_i - mu_i (PyMC 'rescor', brms Rescor): "
                      f"Rescor[{R[0]}, {R[1]}] is the correlation of",
                      f"  {R[0]} and {R[1]} at the same predictors and group effects, not of the responses themselves.",
                      f"  Lrescor ~ {describe_prior(self.priors.rescor)} (brms lkj({eta:g}) on Rescor, placed on Lrescor as "
                      "lkj_corr_cholesky), sampled on Stan's cholesky_corr coordinates"]
            if self.joint_weights is not None:                # G08: brms's weights[n] * multi_normal_cholesky_lpdf(...)
                w, column = self.joint_weights, self.spec.specs[R[0]].addition("weights").column
                lines += [f"  weights({column}): each row's joint log density is multiplied by its weight, once (brms's weights[n] * "
                          f"multi_normal_cholesky_lpdf(Y[n] | ...), the weights of every response); weights {w.min():.4f} to "
                          f"{w.max():.4f}, sum {w.sum():.4f}"]
        else:
            lines += ["Likelihood",
                      "  each response its own Normal likelihood, independent given the parameters (no residual correlation): "
                      + "; ".join(f"{r}_i ~ Normal(mu_{r},i, sigma_{r})" for r in R)]
        if s.groups and not s.cross_ids:
            lines += ["  Group-level effects are separate per response (each response's group terms have their own sd, z and effects, brms's "
                      "sd_m, z_m and r_m of that response) and uncorrelated across responses (no shared ID)."]
        if s.cross_ids:                                       # G08: brms's correlation ID across responses
            lines += ["", "Group-level effects correlated across responses (brms's correlation ID)"]
            for cid in s.cross_ids:
                g, roles, prior = cid.factor, self.id_names[cid.factor], self.id_cor_priors[cid.factor]
                lines += [f"  ({', '.join(f'u_{r},{g}[j]' for r in cid.responses)}) = diag({', '.join(f'sd_{r},{g}' for r in cid.responses)}) "
                          f"L_{g} z[j], z[j] ~ Normal(0, I): the ID {cid.id!r} of (1 |{cid.id}| {g}) puts the responses' varying intercepts "
                          f"of {g!r} in one correlation matrix",
                          f"  (brms's one group: sd_m with one entry per response, each under that response's sd prior, and one L_m). "
                          f"{roles['L']} ~ {describe_prior(prior)} (brms lkj({float(prior.params['eta']):g}) on cor_{g}), sampled on "
                          f"Stan's cholesky_corr coordinates;",
                          f"  {roles['cor']} = L L^T (brms Cor_m) on {roles['coordinate']} ({', '.join(cid.labels)}). The responses' "
                          f"effects {', '.join(roles['effects'])} are these correlated effects (each response's own (1 | {g}) below, "
                          "with its sd and z)."]
            others = [g for g in s.groups if g not in {c.factor for c in s.cross_ids}]
            if others:
                lines += [f"  The group-level terms of {others} are separate per response and uncorrelated across responses."]
        for col, roles in self.missing_names.items():         # C06: the missing values, shared by the responses that hold them
            rows = self.submodels[col].design.missing[col].index.tolist()
            readers = [r for r, sub in self.submodels.items() if any(t.operands[0] == col for t in sub.spec.mi_terms)]
            lines += ["", f"Missing values of {col} (brms's mi())",
                      f"  {len(rows)} of {self.design.n} rows of {col} are missing (0-based {rows[:12]}{'...' if len(rows) > 12 else ''}; brms's "
                      f"Jmi_{col} is 1-based): their values are the parameters {roles['values']} (brms's Ymi_{col}), flat, with no prior of",
                      f"  their own. {col}'s formula evaluates its likelihood at the observed and imputed values together (brms's Yl_{col}), "
                      f"and mi({col}) in {', '.join(readers)} reads the same completed column: one variable of the model, informed by "
                      "every formula that holds it."]
        for r, sub in self.submodels.items():
            lines += ["", f"Response {r}: its univariate model; in the multivariate graph its PyMC variables are named {r}_<name> "
                      + ("and its likelihood is the joint one above" if self.rescor else f"and its likelihood is the observed variable {r!r}")]
            lines += [f"  {line}" if line else "" for line in sub.describe().splitlines()]
        lines += ["", "PyMC variables sampled (value space)"]
        listed = set()
        for r, sub in self.submodels.items():                # each response's in its own model's order, then Lrescor's
            for v in sub.pymc_model.value_vars:
                name = self.value_name(r, v.name)
                if name not in listed:                       # C06: the shared missing values once, under brms's name
                    listed.add(name)
                    lines.append(f"  {name:<28} {r}'s {v.name} (above)" if name != v.name else
                                 f"  {name:<28} the missing values of {v.name[len('Ymi_'):]} (shared)")
        if self.rescor:
            lines.append(f"  {'Lrescor_cholesky_corr__':<28} Lrescor on Stan's cholesky_corr coordinates (canonical partial correlations, arctanh scale)")
        for g, roles in self.id_names.items():               # G08
            lines.append(f"  {roles['L'] + '_cholesky_corr__':<28} {roles['L']} on Stan's cholesky_corr coordinates (the ID across responses)")
        deterministic = {d.name for d in self.pymc_model.deterministics}
        dets = [f"{new} = {r}'s {old}" for r, names in self._names.items() for old, new in names.items() if new in deterministic]
        dets += [f"{roles['cor']} = {roles['L']} {roles['L']}^T (brms Cor_m)" for roles in self.id_names.values()]   # G08
        lines.append("Deterministic: " + ", ".join(dets + (["rescor = Lrescor Lrescor^T (brms Rescor)"] if self.rescor else [])))
        for w in self.warnings:
            lines += ["", "WARNING: " + w]
        return "\n".join(lines)


# -- predictions ---------------------------------------------------------------------------------------------
def predict_multivariate(model: MultivariateModel, fit_result, *, newdata: pd.DataFrame | None = None, kind: str = "response",
                         new_groups="error", seed: int | None = None, include_group_effects: bool = True, dpar=None, resp=None):
    """``predict()`` of a multivariate model: each response's linear predictor computed by its univariate model's prediction
    code from its own draws (``MultivariateModel.response_posterior``), with the same policies for unseen levels and one random
    stream for every response (each response's new levels draw their own effects), stacked on a last ``response`` dimension
    in the order of the responses (brms's prediction arrays). ``kind="linear"`` returns ``eta``, ``kind="mean"`` the expected
    responses ``mu`` (brms's ``posterior_linpred`` / ``posterior_epred``), ``kind="response"`` also ``y``, each row's responses
    drawn jointly, ``mu + diag(sigma) Lrescor z`` with ``z`` standard normal (the identity for ``Lrescor`` without rescor).
    ``resp`` (F06c) names one response, whose predictions alone are returned on its rows (``predict_response``); a model with
    ``subset()`` takes nothing else, as brms refuses its joint predictions."""
    import xarray as xr

    from autobayes.identity import check_fit_binding
    from autobayes.predict import _check_newdata, _group_policies, _predict_by_hand, _prediction_attrs, expected_response

    fam = model.family_spec
    if dpar is not None:
        raise ValueError("dpar does not apply to a multivariate model: its responses have no distributional formulas (divergence D56)")
    if kind not in fam.prediction_kinds:
        raise ValueError(fam.kind_error)
    if not isinstance(include_group_effects, (bool, np.bool_)):
        raise ValueError("include_group_effects must be boolean")
    if not include_group_effects and new_groups != "error":
        raise ValueError("new_groups must stay at its default when group effects are excluded")
    if resp is not None:                                       # F06c: one response, on its rows
        return predict_response(model, fit_result, resp, newdata=newdata, kind=kind, new_groups=new_groups, seed=seed,
                                include_group_effects=include_group_effects)
    if model.spec.subsets:
        raise ValueError(f"{SUBSET_REFUSED} (\"Argument 'resp' must be a single variable name for models using addition argument "
                         f"'subset'.\"): each response is predicted on its own rows; pass resp= one of {list(model.responses)} "
                         "(docs/subset.md, divergence D160)")
    # Every response's grouping factors (one right-hand side: the same; a formula per response, C06: each its own) and new
    # rows checked against every response's formula, which reads its own columns
    union = SimpleNamespace(spec=SimpleNamespace(groups=model.spec.groups))
    policies = _group_policies(union, new_groups)
    binding = check_fit_binding(model, fit_result, allow_unverified=True)
    post = getattr(fit_result, "idata", fit_result).posterior
    dfs = {r: None if newdata is None else _check_newdata(sub, newdata, include_group_effects=include_group_effects)
           for r, sub in model.submodels.items()}
    new_levels = {}
    for r, sub in model.submodels.items():
        for t in (sub.spec.group_terms if include_group_effects else ()):
            if t.factor not in new_levels:
                new_levels[t.factor] = ([] if dfs[r] is None else
                                        sorted(set(dfs[r][t.column]) - set(sub.design.group_designs[t.factor].levels)))
    for g, unseen in new_levels.items():
        if unseen and policies[g] == "error":
            raise FormulaError(
                f"newdata has levels of {g!r} not seen in training: {unseen}. Pass new_groups='population' to draw their effects "
                "from Normal(0, sd), or new_groups='resample_observed' for brms's default 'uncertainty' behaviour; an exact-key "
                "mapping can set the policy separately for each factor.")
        if unseen and g in model.id_names:                    # G08: the responses' effects would be drawn independently
            raise FormulaError(f"newdata has levels of {g!r} not seen in training: {unseen}, whose effects a correlation ID correlates "
                               "across the responses; drawing them jointly is not supported (each response's own predictions, "
                               f"predict(resp=...), draw its effects from its own marginal; {ID_ROW})")
    rng = np.random.default_rng(seed)
    posts = {r: model.response_posterior(post, r) for r in model.responses}
    eta = np.stack([_predict_by_hand(sub, SimpleNamespace(posterior=posts[r]), dfs[r], "linear", rng,
                                     {g: policies[g] for g in sub.spec.groups}, include_group_effects=include_group_effects)
                    ["eta"].transpose("chain", "draw", "__obs__").to_numpy()
                    for r, sub in model.submodels.items()], axis=-1)                        # (chain, draw, row, response)
    dims = ("chain", "draw", "__obs__", PREDICTION_DIM)
    coords = {"chain": post["chain"], "draw": post["draw"], "__obs__": np.arange(eta.shape[2]), PREDICTION_DIM: list(model.responses)}
    if kind == "linear":
        out = {"eta": (dims, eta)}
    else:
        mean = np.stack([np.asarray(expected_response(fam, sub, posts[r], eta[..., k]), dtype=float)
                         for k, (r, sub) in enumerate(model.submodels.items())], axis=-1)
        out = {fam.mean_name: (dims, mean)}
        if kind == "response":
            sigma = np.stack([posts[r][sub.names["sigma"]].transpose("chain", "draw").to_numpy() for r, sub in model.submodels.items()], axis=-1)
            K = len(model.responses)
            if model.rescor:
                sel = {RESPONSE_DIM: list(model.responses), RESPONSE_DIM_T: list(model.responses)}
                L = post["Lrescor"].sel(sel).transpose("chain", "draw", RESPONSE_DIM, RESPONSE_DIM_T).to_numpy()
            else:
                L = np.broadcast_to(np.eye(K), sigma.shape + (K,))
            z = rng.standard_normal(mean.shape)
            out["y"] = (dims, mean + np.einsum("cdkj,cdnj->cdnk", sigma[..., :, None] * L, z))
    ds = xr.Dataset(out, coords=coords)
    _prediction_attrs(union, ds, policies, new_levels, in_sample=newdata is None, manual=True, include_group_effects=include_group_effects)
    ds.attrs["fit_binding_status"] = binding
    if kind == "linear":
        ds.attrs.update(prediction_scale="linear", link=model.link)
    return ds


def predict_response(model: MultivariateModel, fit_result, resp: str, *, newdata: pd.DataFrame | None = None, kind: str = "response",
                     new_groups="error", seed: int | None = None, include_group_effects: bool = True):
    """``predict(resp=...)`` of a multivariate model (F06c; brms's ``posterior_linpred``, ``posterior_epred`` and
    ``posterior_predict`` with ``resp =``): one response's predictions by its univariate model's prediction code from its own
    draws, on its rows: the fitted rows of its subset (every row without ``subset()``), or the rows of ``newdata`` whose own
    indicator is 1 (the response's ``subset()`` column, which ``newdata`` must hold), as brms selects them. Dimensions ``(chain,
    draw, __obs__)``, the ``__obs__`` coordinates those rows' 0-based positions in the data or in ``newdata``; ``new_groups`` as for
    the joint predictions (covering every factor of the model when a mapping); ``kind="response"`` draws the response from its
    Normal at each row (its marginal, with or without the residual correlation)."""
    import xarray as xr

    from autobayes.identity import check_fit_binding
    from autobayes.predict import _check_newdata, _group_policies, _predict_by_hand, _prediction_attrs, expected_response

    if resp not in model.responses:
        raise ValueError(f"resp must name one of the responses {list(model.responses)}, got {resp!r}")
    fam, sub = model.family_spec, model.submodels[resp]
    policies = _group_policies(SimpleNamespace(spec=SimpleNamespace(groups=model.spec.groups)), new_groups)
    policies = {g: policies[g] for g in sub.spec.groups}             # the response's own factors
    binding = check_fit_binding(model, fit_result, allow_unverified=True)
    post = getattr(fit_result, "idata", fit_result).posterior
    column = model.spec.subsets.get(resp)
    if newdata is None:
        df, rows = None, (np.arange(model.design.n) if model.design.rows is None else model.design.rows[resp])
    else:
        rows = np.arange(len(newdata)) if column is None else subset_rows(newdata, column, resp, where="newdata")
        df = _check_newdata(sub, newdata.iloc[rows], include_group_effects=include_group_effects)
    new_levels = {t.factor: ([] if df is None else sorted(set(df[t.column]) - set(sub.design.group_designs[t.factor].levels)))
                  for t in (sub.spec.group_terms if include_group_effects else ())}
    for g, unseen in new_levels.items():
        if unseen and policies[g] == "error":
            raise FormulaError(
                f"newdata has levels of {g!r} not seen in training by the response {resp!r}: {unseen}. Pass new_groups='population' "
                "to draw their effects from Normal(0, sd), or new_groups='resample_observed' for brms's default 'uncertainty' "
                "behaviour; an exact-key mapping can set the policy separately for each factor.")
    rng = np.random.default_rng(seed)
    part = model.response_posterior(post, resp)
    eta = (_predict_by_hand(sub, SimpleNamespace(posterior=part), df, "linear", rng, policies, include_group_effects=include_group_effects)
           ["eta"].transpose("chain", "draw", "__obs__").to_numpy())
    dims = ("chain", "draw", "__obs__")
    coords = {"chain": post["chain"], "draw": post["draw"], "__obs__": np.asarray(rows)}
    if kind == "linear":
        out = {"eta": (dims, eta)}
    else:
        mean = np.asarray(expected_response(fam, sub, part, eta), dtype=float)
        out = {fam.mean_name: (dims, mean)}
        if kind == "response":
            sigma = part[sub.names["sigma"]].transpose("chain", "draw").to_numpy()
            out["y"] = (dims, mean + sigma[..., None] * rng.standard_normal(mean.shape))
    ds = xr.Dataset(out, coords=coords)
    _prediction_attrs(SimpleNamespace(spec=SimpleNamespace(groups=sub.spec.groups)), ds, policies, new_levels, in_sample=newdata is None,
                      manual=True, include_group_effects=include_group_effects)
    ds.attrs.update(fit_binding_status=binding, response=resp)
    if kind == "linear":
        ds.attrs.update(prediction_scale="linear", link=model.link)
    return ds


# -- priors --------------------------------------------------------------------------------------------------
def multivariate_default_priors(formula, data: pd.DataFrame, *, family: str = "gaussian", categorical=None, nu=None, link=None,
                                center_predictors: bool = True, rescor=None, dropna: bool = False) -> Priors:
    """brms's default priors of a multivariate model (``ab.default_priors`` with an ``mvbind()`` formula or ``rescor``): each
    response's own defaults (``default_priors`` of its formula, the location and scale read from that response), a correlated
    term's ``lkj(1)`` once for every response (brms's ``cor`` has no ``resp``) and, with ``rescor=True``, ``lkj(1)`` on the
    residual correlation (brms's class ``rescor``). A response with ``subset()`` (F06c) has its defaults read from the rows of its
    subset, as brms reads them (its Intercept's location is the subset's median response). ``categorical`` and ``dropna`` as the
    model takes them (D172, D174): the defaults are read from the rows the model keeps."""
    from autobayes.column_types import typed_categorical
    from autobayes.default_priors import default_priors
    from autobayes.missing_rows import check_dropna, drop_missing_multivariate
    from autobayes.priors import LKJ

    categorical = typed_categorical(data, categorical, lambda cat: parse_multivariate(formula, None, family, cat, rescor=rescor))
    if check_dropna(dropna):
        data, _, _ = drop_missing_multivariate(parse_multivariate(formula, None, family, categorical, rescor=rescor), data)
    spec = parse_multivariate(formula, data, family, categorical, rescor=rescor)
    cats = response_categorical(spec, categorical)              # each response's declarations (C06: its own main effects')
    frames = {r: data.iloc[subset_rows(data, c, r)] for r, c in spec.subsets.items()}   # F06c: each subset's rows
    resp = {r: default_priors(spec.formulas[r], frames.get(r, data), family=family, categorical=cats[r], nu=nu, link=link,
                              center_predictors=center_predictors, imputed=spec.imputed) for r in spec.responses}
    cor = resp[spec.responses[0]].cor
    if spec.separate:                                           # C06: every response's correlated terms, each factor once
        cor = {g: p for r in spec.responses for g, p in dict(resp[r].cor or {}).items()} or None
    if spec.cross_ids:                                          # G08: brms's lkj(1) on each correlation ID across responses
        cor = {**dict(cor or {}), **{c.factor: replace(LKJ(1), source="default") for c in spec.cross_ids}}
    priors = Priors(resp={r: replace(p, cor=None) for r, p in resp.items()}, cor=cor,
                    rescor=replace(LKJ(1), source="default") if spec.rescor else None)
    priors.validate_multivariate(spec, family)
    return priors


def multivariate_prior_table(priors: Priors, spec: MultivariateSpec, family="gaussian", *, center_predictors: bool = True):
    """A multivariate model's priors in brms's table form: each response's table (``prior_table`` of its own priors and formula)
    with its ``resp``, the ``cor`` rows once without ``resp`` (brms's correlation prior is per factor, shared by the responses),
    and the ``rescor`` row; sorted as brms sorts them."""
    from autobayes.default_priors import COLUMNS, prior_table
    from autobayes.priors import brms_prior, class_default

    priors.validate_multivariate(spec, family)
    rows, cor_rows = [], set()
    cross, own_cor = split_cor(priors, spec)                    # G08: the IDs across responses' cor, and the responses' own
    own = replace(priors, cor=own_cor) if cross else priors
    for r in spec.responses:
        for row in prior_table(own.response_priors(r), spec.specs[r], family, center_predictors=center_predictors):
            if row["class"] == "cor":                       # once per factor (C06: whichever response has the correlated term)
                if cross:                                   # G08: written with the IDs' rows, below
                    continue
                if (row["group"], row["coef"]) not in cor_rows:
                    cor_rows.add((row["group"], row["coef"]))
                    rows.append(row)
                continue
            rows.append({**row, "resp": r})
    if cross:
        every = {g: p for r in spec.responses for g, p in own.response_priors(r).cor_priors(spec.specs[r]).items()}
        rows += _cor_rows({**every, **cross}, class_default(priors.cor, "cor"))
    if spec.rescor:
        rows.append(dict(zip(COLUMNS, (brms_prior(priors.rescor), "rescor", "", "", "", "", "", "", "", "", priors.rescor.source))))
    rows.sort(key=lambda row: (row["resp"], row["dpar"], row["nlpar"], row["class"], row["group"], row["coef"], row["tag"]))
    return rows


def _cor_rows(cor: Mapping, default) -> list[dict[str, str]]:
    """G08: brms's ``cor`` rows of a model with a correlation ID across responses, written as a univariate table writes them
    (``default_priors.prior_table``) for every correlation matrix of the model at once (the IDs' and the responses' own): the
    class-wide prior and each factor's override (D173), the prior every factor shares on the class row over empty group rows, or
    an empty class row over each factor's own."""
    from autobayes.default_priors import COLUMNS
    from autobayes.priors import brms_prior

    def row(prior, group="", source=None):
        text = brms_prior(prior) if prior is not None else ""
        src = source if source is not None else (prior.source if prior is not None else "default")
        return dict(zip(COLUMNS, (text, "cor", "", group, "", "", "", "", "", "", src)))

    ps = list(cor.values())
    if default is not None:
        return [row(default)] + [row(None if p == default else p, g, default.source if p == default else None) for g, p in cor.items()]
    if all(p == ps[0] for p in ps):
        return [row(ps[0])] + [row(None, g, p.source) for g, p in cor.items()]
    return [row(None)] + [row(p, g) for g, p in cor.items()]


# -- identity ------------------------------------------------------------------------------------------------
def _freeze_target(model: MultivariateModel) -> dict[str, str]:
    """The target identity of a multivariate model: the responses' own targets (each a univariate model's digest: its
    formula, design, response, priors), their order, ``Y``, and the residual correlation and its prior."""
    from autobayes.identity import SCHEMA, _prior, digest

    Y = model.design.Y
    missing = np.isnan(Y)                                   # C06: the missing values are parameters; their rows enter instead
    if model.spec.subsets:                                  # F06c: NaN outside a response's rows, which enter instead
        missing = np.zeros(Y.shape, dtype=bool)
        Y = np.where(np.isnan(Y), 0.0, Y)
    payload = {"schema": SCHEMA, "family": model.family,
               "multivariate": {"formula": model.spec.canonical_formula, "responses": list(model.responses), "rescor": model.rescor,
                                "rescor_prior": _prior(model.priors.rescor) if model.rescor else None,
                                "Y": np.where(missing, 0.0, Y) if missing.any() else Y,
                                **({"missing": {r: np.flatnonzero(missing[:, k]) for k, r in enumerate(model.responses)}}
                                   if missing.any() else {}),
                                "targets": {r: sub.target_identity["sha256"] for r, sub in model.submodels.items()},
                                **({"subsets": {r: {"column": c, "rows": model.design.rows[r]} for r, c in model.spec.subsets.items()}}
                                   if model.spec.subsets else {}),
                                # G08: the correlation IDs across responses and their priors (the responses' own targets lack them)
                                **({"cross_ids": [{"id": c.id, "factor": c.factor, "responses": list(c.responses),
                                                   "prior": _prior(model.id_cor_priors[c.factor])} for c in model.spec.cross_ids]}
                                   if model.spec.cross_ids else {})}}
    return {"schema": SCHEMA, "sha256": digest(payload)}   # G11: the canonical bytes, streamed (identity.digest)
