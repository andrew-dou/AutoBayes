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

Refused, naming the construct: a formula per response (brms's ``bf(y ~ ...) + bf(y2 ~ ...)``) without ``mi()``,
distributional formulas and addition terms (D56), ``mi()`` in an ``mvbind()`` formula, ``mo()`` (not verified in a
multivariate model; D56), correlated group effects across responses ``(1 |ID| g)`` (D57), families other than the Gaussian
(``FamilySpec.multivariate``; D58), and a model without an explicit ``rescor`` (D58).

Missing values (C06, docs/missing-and-measurement-error.md): brms's ``bf(y ~ mi(xm) + w) + bf(xm | mi() ~ w) +
set_rescor(FALSE)`` is a list of formulas, one per response, ``ab.gaussian(["y ~ mi(xm) + w", "xm | mi() ~ w"], data,
ab.Priors(resp={...}), rescor=False)``: the response ``xm`` has missing values (NaN), which are the parameters ``Ymi_xm``
(brms's ``Ymi_xm``, flat), and ``mi(xm)`` reads the column completed by them (brms's ``Yl_xm``). Each response's model holds
the missing values it reads; the assembled graph has one ``Ymi_xm``, by brms's name, shared by them. Refused: an ``mi()``
term without its column's ``| mi()`` formula, with ``rescor=True``, in a cycle of imputations, ``me()`` in a multivariate
model, and a response read by another formula other than through ``mi()`` (D72, D73).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd

from autobayes.correlation import check_correlation_cholesky, cholesky_corr_free, cholesky_corr_rv, n_free
from autobayes.families import FAMILIES, get_family
from autobayes.formula import (_GROUP_RE, _IDENT, MI_CALL_RE, MVBIND_RE, FormulaError, ModelSpec, _split_top_level, check_imputation,
                               parse_formula)
from autobayes.priors import PriorError, Priors, describe_prior

_MVBIND_FORMULA_RE = re.compile(r"^\s*mvbind\s*\(([^()]*)\)\s*([^~]*?)\s*~\s*(.*?)\s*$", re.S)   # mvbind(<responses>) <rest> ~ <rhs>
_MI_RE = re.compile(r"(?<![A-Za-z0-9_.])mi\s*\(")                                                 # brms's mi() (C06)
_MO_RE = re.compile(r"(?<![A-Za-z0-9_.])mo\s*\(")                                                 # brms's mo() (B03)
_GP_RE = re.compile(r"(?<![A-Za-z0-9_.])gp\s*\(")                                                 # brms's gp() (C03)
_AUTOCOR_RE = re.compile(r"(?<![A-Za-z0-9_.])(ar|ma|arma)\s*\(")                                     # brms's ARMA terms (C04)
_SAR_RE = re.compile(r"(?<![A-Za-z0-9_.])sar\s*\(")                                                      # brms's SAR term (F02b)
_FCOR_RE = re.compile(r"(?<![A-Za-z0-9_.])fcor\s*\(")                                                 # brms's fcor() (F03a)
_ME_RE = re.compile(r"(?<![A-Za-z0-9_.])me\s*\(")                                                 # brms's me() (C06)
RESPONSE_DIM, RESPONSE_DIM_T = "__resp__", "__resp_T__"     # the graph's response coordinates (Lrescor, rescor, the joint response)
PREDICTION_DIM = "response"                                 # predict()'s response dimension, brms's last axis of its prediction arrays


def is_multivariate_formula(formula) -> bool:
    """Whether ``formula`` is brms's ``mvbind(y, y2) ~ ...`` (C01), alone or first in a list of formulas, or a list of
    formulas with ``mi()`` (C06: one formula per response, ``["y ~ mi(xm) + w", "xm | mi() ~ w"]``)."""
    if isinstance(formula, (list, tuple)) and formula:
        if all(isinstance(f, str) for f in formula) and any(MI_CALL_RE.search(f) for f in formula):
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
    a list instead; the model reads nothing but ``specs``."""
    formula: str
    responses: tuple[str, ...]
    rhs: str
    specs: Mapping[str, ModelSpec]
    rescor: bool

    @property
    def separate(self) -> bool:
        """A formula of its own per response (C06): ``formula`` is the list as written, ``rhs`` empty."""
        return isinstance(self.formula, tuple)

    @property
    def formulas(self) -> dict[str, str]:
        """Each response's own formula, ``{"y": "y ~ x + (1 | g)", ...}``, from which its univariate model is built."""
        return {r: self.specs[r].formula for r in self.responses}

    @property
    def canonical_formula(self) -> str:
        """``mvbind(y, y2) ~ <the canonical right-hand side>``; a formula per response (C06) as brms writes it,
        ``bf(y ~ mi(xm) + w) + bf(xm | mi() ~ w)``."""
        if self.separate:
            return " + ".join(f"bf({self.specs[r].canonical_formula})" for r in self.responses)
        return f"mvbind({', '.join(self.responses)}) ~ {self.specs[self.responses[0]].rhs}"

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
    must be True or False. Everything outside the verified subset is refused naming the construct (module docstring)."""
    fam = get_family(family)
    if not fam.multivariate:
        raise FormulaError(f"a multivariate model (brms's mvbind() or mvbf()) is offered for the {', '.join(_offering()) or 'no'} family "
                           f"here; a multivariate {fam.name} model is not (docs/multivariate.md, divergence D58)")
    if is_multivariate_formula(formula) and not isinstance(formula, str) and not MVBIND_RE.match(formula[0]):
        return _parse_imputation(formula, data, family, categorical, rescor=rescor)      # C06: a formula per response, with mi()
    if not isinstance(formula, str):
        first = formula[0] if isinstance(formula, (list, tuple)) and formula else formula
        what = ("a distributional formula in a multivariate model (brms's bf(mvbind(y, y2) ~ ..., sigma ~ z))"
                if isinstance(first, str) and MVBIND_RE.match(first) else
                "a formula of its own per response (brms's bf(y ~ ...) + bf(y2 ~ ...))")
        raise FormulaError(f"{formula!r}: {what} is not supported in this slice; a multivariate model is one formula, "
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
    if rest:
        raise FormulaError(f"{formula!r}: {rest!r} after mvbind(): addition terms (weights(), cens(), trunc(), ...) on the responses of a "
                           "multivariate model are not supported in this slice (docs/multivariate.md, divergence D56)")
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
    for piece in _split_top_level(rhs):
        if _MI_RE.search(piece):
            raise FormulaError(f"{formula!r}: {piece!r}: brms's mi() (missing values modelled by a response's own formula) is not "
                               "part of this slice (feature C06; docs/multivariate.md, divergence D56): not in an mvbind() formula, "
                               "whose responses share one right-hand side; C06 takes it in a list of formulas, one per response, "
                               "['y ~ mi(x) + w', 'x | mi() ~ w'] with rescor=False (docs/missing-and-measurement-error.md)")
        if _GP_RE.search(piece):
            raise FormulaError(f"{formula!r}: {piece!r}: a Gaussian process in a multivariate formula is not verified in this slice "
                               "(each response would have its own process, brms's sdgp_y and lscale_y); write it in a univariate "
                               "model (docs/multivariate.md, divergence D56)")
        if _AUTOCOR_RE.search(piece):
            raise FormulaError(f"{formula!r}: {piece!r}: an autocorrelation term in a multivariate formula is not verified in this "
                               "slice (brms gives each response its own ARMA term, ar_y and ar_y2, with J_lag_y and J_lag_y2); write it "
                               "in a univariate model (docs/autocorrelation.md, divergence D70)")
        if _SAR_RE.search(piece):                             # F02b
            raise FormulaError(f"{formula!r}: {piece!r}: a SAR term in a multivariate formula is not verified in this slice; write it "
                               "in a univariate model (docs/spatial-sar.md, divergence D143)")
        if _FCOR_RE.search(piece):                            # F03a: brms gives each response its own Mfcor_y
            raise FormulaError(f"{formula!r}: {piece!r}: a fixed residual covariance in a multivariate formula is not verified in "
                               "this slice (brms gives each response its own matrix, Mfcor_y and Mfcor_y2); write it in a univariate "
                               "model (docs/fcor.md, divergence D144)")
        if _ME_RE.search(piece):
            raise FormulaError(f"{formula!r}: {piece!r}: a measurement-error term in a multivariate model is not supported in this "
                               "slice; me() belongs to a univariate Gaussian model here (docs/missing-and-measurement-error.md, "
                               "divergence D73)")
        if _MO_RE.search(piece):
            raise FormulaError(f"{formula!r}: {piece!r}: a monotonic term in a multivariate formula is not verified in this slice (each "
                               "response would have its own coefficient and simplex, brms's bsp_y and simo_y); write it in a "
                               "univariate model (docs/multivariate.md, divergence D56)")
        gm = _GROUP_RE.match(piece)
        if gm and "|" in gm.group(3):
            raise FormulaError(f"{formula!r}: {piece!r}: a correlation ID, brms's (1 |ID| g), which correlates the responses' group-level "
                               "effects, is not supported; each response's group terms are separate and uncorrelated across responses, "
                               "as brms's are without an ID (docs/multivariate.md, divergence D57)")
    specs = {}
    for r in responses:
        try:
            s = parse_formula(f"{r} ~ {rhs}", data, family, categorical)
        except FormulaError as e:
            raise FormulaError(f"{formula!r}, the formula of the response {r!r} ('{r} ~ {rhs}'): {e}") from None
        if s.custom_group_terms:                                         # D08: verified in univariate models only
            raise FormulaError(f"{formula!r}: '{s.custom_group_terms[0].canonical}' in a multivariate formula is not supported in this "
                               "slice (docs/multi-membership-and-by.md, divergences D104, D105)")
        reads = {*s.pop_columns, *(c for t in s.group_terms for c in t.operands), *([s.offset.column] if s.offset is not None else [])}
        others = sorted(reads & set(responses))
        if others:
            raise FormulaError(f"{formula!r}: the response(s) {others} cannot also be predictors, grouping factors or offsets of the "
                               "right-hand side every response shares")
        specs[r] = s
    return MultivariateSpec(formula.strip(), responses, rhs, MappingProxyType(specs), bool(rescor))


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
        raise FormulaError(f"{formulas!r}: mvbind() inside a list of formulas is not supported in this slice; {IMPUTATION_FORM}")
    if rescor is None:
        raise ValueError("a multivariate model needs rescor=True or rescor=False (divergence D58); a model with mi() takes rescor=False "
                         "(brms's set_rescor(FALSE); docs/missing-and-measurement-error.md)")
    if not isinstance(rescor, (bool, np.bool_)):
        raise ValueError(f"rescor must be True or False, got {rescor!r}")
    if rescor:
        raise FormulaError(f"{formulas!r}: mi() with a residual correlation (rescor=True, brms's set_rescor(TRUE)) is not supported in "
                           "this slice: brms would correlate the imputed values' residuals with the other responses'; pass "
                           "rescor=False (docs/missing-and-measurement-error.md, divergence D72)")
    specs0 = []
    for text in texts:
        try:
            specs0.append(parse_formula(text, None, family, None))     # the structure, before the data and the declarations
        except FormulaError as e:
            raise FormulaError(f"{formulas!r}, the formula {text!r}: {e}") from None
        if specs0[-1].custom_group_terms:                               # D08: verified in univariate models only
            raise FormulaError(f"{formulas!r}, the formula {text!r}: '{specs0[-1].custom_group_terms[0].canonical}' in a multivariate "
                               "model is not supported in this slice (docs/multi-membership-and-by.md, divergences D104, D105)")
    responses = tuple(s.response for s in specs0)
    dup = sorted({r for r in responses if responses.count(r) > 1})
    if dup:
        raise FormulaError(f"{formulas!r}: the response(s) {dup} have more than one formula; give each response one")
    imputed = tuple(s.response for s in specs0 if s.addition("mi") is not None)
    for s in specs0:
        others = [a.term for a in s.additions if a.kind != "mi"]
        if others or s.trials is not None:
            raise FormulaError(f"{s.formula!r}: addition terms ({', '.join(others or ['trials()'])}) on a response of a multivariate model "
                               "are not supported in this slice (docs/multivariate.md, divergence D56)")
        if s.me_terms:
            raise FormulaError(f"{s.formula!r}: a measurement-error term in a multivariate model is not supported in this slice; me() "
                               "belongs to a univariate Gaussian model here (docs/missing-and-measurement-error.md, divergence D73)")
        if s.mo_terms:
            raise FormulaError(f"{s.formula!r}: a monotonic term in a multivariate model is not verified in this slice (docs/multivariate.md, "
                               "divergence D56)")
        if s.gp_terms:                                        # C03: refused as in an mvbind() formula (a formula with mi() already was)
            raise FormulaError(f"{s.formula!r}: the Gaussian process {s.gp_terms[0].term!r} in a multivariate model is not verified in this "
                               "slice (each response would have its own process, brms's sdgp_y and lscale_y); write it in a univariate "
                               "model (docs/multivariate.md, divergence D56)")
        if s.fcor is not None:                                # F03a: likewise
            raise FormulaError(f"{s.formula!r}: {s.fcor.term} in a multivariate model is not verified in this slice; write it in a "
                               "univariate model (docs/fcor.md, divergence D144)")
        if s.autocor is not None:                             # C04: likewise
            raise FormulaError(f"{s.formula!r}: the autocorrelation term {s.autocor.term!r} in a multivariate model is not verified in "
                               "this slice (brms gives each response its own ARMA term, ar_y and ar_y2); write it in a univariate model "
                               "(docs/autocorrelation.md, divergence D70)")
        if getattr(s, "sar", None) is not None:               # F02b: likewise
            raise FormulaError(f"{s.formula!r}: the SAR term {s.sar.term!r} in a multivariate model is not verified in this slice; "
                               "write it in a univariate model (docs/spatial-sar.md, divergence D143)")
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
                   *([s.offset.column] if s.offset is not None else [])}       # what the formula reads other than through mi()
        plain = sorted(columns & (set(responses) - {s.response}))
        if plain:
            raise FormulaError(f"{s.formula!r}: the response(s) {plain} of another formula are read here other than through mi(); "
                               f"write mi({plain[0]}) to read {plain[0]!r} with its imputed values (docs/missing-and-measurement-error.md)")
    if not any(s.mi_terms for s in specs0):
        raise FormulaError(f"{formulas!r}: a formula of its own per response (brms's bf(y ~ ...) + bf(y2 ~ ...)) is not supported in this "
                           "slice without mi(); a multivariate model is one formula, 'mvbind(y, y2) ~ <terms>', whose right-hand side every "
                           f"response shares (docs/multivariate.md, divergence D56), or {IMPUTATION_FORM}")
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


@dataclass(frozen=True)
class MultivariateDesign:
    """What a multivariate model was fit to (C01): ``Y`` the ``(n, K)`` responses in ``responses`` order (brms's ``Y``) and
    each response's own ``Design`` (``designs``: its univariate model's, the same design columns, group blocks and centring for
    every response, which share the right-hand side and the rows)."""
    responses: tuple[str, ...]
    Y: np.ndarray
    designs: Mapping[str, object]

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
    """

    def __init__(self, formula, data: pd.DataFrame, priors: Priors, *, family: str = "gaussian", rescor: bool | None = None,
                 center_predictors: bool = True, categorical: Mapping[str, object] | None = None):
        from autobayes.model import Model

        fam = get_family(family)
        spec0 = parse_multivariate(formula, None, family, categorical, rescor=rescor)   # the structure, before the data
        if not isinstance(priors, Priors):
            raise PriorError(f"priors must be an ab.Priors, got {type(priors).__name__}")
        priors.validate_multivariate(spec0, fam)
        self.family, self.family_spec, self.priors = family, fam, priors
        self.responses: tuple[str, ...] = spec0.responses
        self.rescor = bool(rescor)
        self.center_predictors = bool(center_predictors)
        cats = response_categorical(spec0, categorical)          # each response's declarations (C06: its own main effects')
        self.submodels = MappingProxyType({r: Model(spec0.formulas[r], data, priors.response_priors(r), family=family,
                                                    center_predictors=center_predictors, categorical=cats[r], imputed=spec0.imputed)
                                           for r in self.responses})
        subs = list(self.submodels.values())
        if len({sub.design.n for sub in subs}) != 1 or any(len(sub.dropped_rows) for sub in subs):
            raise RuntimeError("the responses' models were fit to different rows")
        self.spec = MultivariateSpec(spec0.formula, spec0.responses, spec0.rhs,
                                     MappingProxyType({r: sub.spec for r, sub in self.submodels.items()}), self.rescor)
        # one right-hand side: the responses' centring; a formula per response (C06): the flag, which each response's table
        # applies when its formula has an intercept
        self.effective_center_predictors = self.center_predictors if self.spec.separate else subs[0].effective_center_predictors
        self.warnings: list[str] = list(dict.fromkeys(w for sub in subs for w in sub.warnings))
        self.design = MultivariateDesign(self.responses, np.column_stack([sub.design.y for sub in subs]),
                                         MappingProxyType({r: sub.design for r, sub in self.submodels.items()}))
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
                pm.CustomDist(self.observed_names[0], mu, sigma, L, logp=multi_normal_cholesky_logp, observed=self.design.Y,
                              signature="(k),(k),(k,k)->(k)", dims=("__obs__", RESPONSE_DIM))
        return model

    def _check_graph(self) -> None:
        """The assembled graph holds exactly the responses' variables under their new names with their transforms, the
        observed variables ``observed_names``, and with ``rescor=True`` ``Lrescor`` (free, ``K x K``, sampled on the
        ``K (K - 1) / 2`` cholesky_corr coordinates) and the deterministic ``rescor``, each on the response coordinates."""
        pmm, K = self.pymc_model, len(self.responses)
        expected = {self.value_name(r, v.name) for r, sub in self.submodels.items() for v in sub.pymc_model.value_vars}
        expected |= {"Lrescor_cholesky_corr__"} if self.rescor else set()
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

    # -- points --------------------------------------------------------------------------------------------
    def value_point(self, *, resp: Mapping[str, Mapping], Lrescor=None, Ymi=None) -> dict[str, np.ndarray]:
        """A point in the graph's value space: ``resp`` maps each response to the keyword arguments of its univariate
        model's ``Model.value_point`` (``{"y": {"intercept": ..., "b": ..., "sd": ..., "z": ..., "sigma": ...}, ...}``),
        ``Lrescor`` is the Cholesky factor of the residual correlation matrix (lower triangular, unit-norm rows, in the
        responses' order), required exactly with ``rescor=True`` and stored on Stan's ``cholesky_corr`` coordinates. ``Ymi``
        (C06) gives the missing values of each column of ``missing_names`` (``{"xm": array(Nmi)}``, one per missing row; a
        single column also takes the bare array), which every response holding them shares: it is not part of ``resp``."""
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
        raise ValueError("prior_predictive() of a multivariate model is not part of this slice (docs/multivariate.md); draw from "
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
                  + ("its own formula: brms's bf(...) + bf(...))" if s.separate else "right-hand side: brms's mvbind())"),
                  f"  data      : n = {self.design.n} rows, each with its {K} responses", ""]
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
        else:
            lines += ["Likelihood",
                      "  each response its own Normal likelihood, independent given the parameters (no residual correlation): "
                      + "; ".join(f"{r}_i ~ Normal(mu_{r},i, sigma_{r})" for r in R)]
        if s.groups:
            lines += ["  Group-level effects are separate per response (each response's group terms have their own sd, z and effects, brms's "
                      "sd_m, z_m and r_m of that response) and uncorrelated across responses (no shared ID)."]
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
        deterministic = {d.name for d in self.pymc_model.deterministics}
        dets = [f"{new} = {r}'s {old}" for r, names in self._names.items() for old, new in names.items() if new in deterministic]
        lines.append("Deterministic: " + ", ".join(dets + (["rescor = Lrescor Lrescor^T (brms Rescor)"] if self.rescor else [])))
        for w in self.warnings:
            lines += ["", "WARNING: " + w]
        return "\n".join(lines)


# -- predictions ---------------------------------------------------------------------------------------------
def predict_multivariate(model: MultivariateModel, fit_result, *, newdata: pd.DataFrame | None = None, kind: str = "response",
                         new_groups="error", seed: int | None = None, include_group_effects: bool = True, dpar=None):
    """``predict()`` of a multivariate model: each response's linear predictor computed by its univariate model's prediction
    code from its own draws (``MultivariateModel.response_posterior``), with the same policies for unseen levels and one random
    stream for every response (each response's new levels draw their own effects), stacked on a last ``response`` dimension
    in the order of the responses (brms's prediction arrays). ``kind="linear"`` returns ``eta``, ``kind="mean"`` the expected
    responses ``mu`` (brms's ``posterior_linpred`` / ``posterior_epred``), ``kind="response"`` also ``y``, each row's responses
    drawn jointly, ``mu + diag(sigma) Lrescor z`` with ``z`` standard normal (the identity for ``Lrescor`` without rescor)."""
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


# -- priors --------------------------------------------------------------------------------------------------
def multivariate_default_priors(formula, data: pd.DataFrame, *, family: str = "gaussian", categorical=None, nu=None, link=None,
                                center_predictors: bool = True, rescor=None) -> Priors:
    """brms's default priors of a multivariate model (``ab.default_priors`` with an ``mvbind()`` formula or ``rescor``): each
    response's own defaults (``default_priors`` of its formula, the location and scale read from that response), a correlated
    term's ``lkj(1)`` once for every response (brms's ``cor`` has no ``resp``) and, with ``rescor=True``, ``lkj(1)`` on the
    residual correlation (brms's class ``rescor``)."""
    from autobayes.default_priors import default_priors
    from autobayes.priors import LKJ

    spec = parse_multivariate(formula, data, family, categorical, rescor=rescor)
    cats = response_categorical(spec, categorical)              # each response's declarations (C06: its own main effects')
    resp = {r: default_priors(spec.formulas[r], data, family=family, categorical=cats[r], nu=nu, link=link,
                              center_predictors=center_predictors, imputed=spec.imputed) for r in spec.responses}
    cor = resp[spec.responses[0]].cor
    if spec.separate:                                           # C06: every response's correlated terms, each factor once
        cor = {g: p for r in spec.responses for g, p in dict(resp[r].cor or {}).items()} or None
    priors = Priors(resp={r: replace(p, cor=None) for r, p in resp.items()}, cor=cor,
                    rescor=replace(LKJ(1), source="default") if spec.rescor else None)
    priors.validate_multivariate(spec, family)
    return priors


def multivariate_prior_table(priors: Priors, spec: MultivariateSpec, family="gaussian", *, center_predictors: bool = True):
    """A multivariate model's priors in brms's table form: each response's table (``prior_table`` of its own priors and formula)
    with its ``resp``, the ``cor`` rows once without ``resp`` (brms's correlation prior is per factor, shared by the responses),
    and the ``rescor`` row; sorted as brms sorts them."""
    from autobayes.default_priors import COLUMNS, prior_table
    from autobayes.priors import brms_prior

    priors.validate_multivariate(spec, family)
    rows, cor_rows = [], set()
    for r in spec.responses:
        for row in prior_table(priors.response_priors(r), spec.specs[r], family, center_predictors=center_predictors):
            if row["class"] == "cor":                       # once per factor (C06: whichever response has the correlated term)
                if (row["group"], row["coef"]) not in cor_rows:
                    cor_rows.add((row["group"], row["coef"]))
                    rows.append(row)
                continue
            rows.append({**row, "resp": r})
    if spec.rescor:
        rows.append(dict(zip(COLUMNS, (brms_prior(priors.rescor), "rescor", "", "", "", "", "", "", "", "", priors.rescor.source))))
    rows.sort(key=lambda row: (row["resp"], row["dpar"], row["nlpar"], row["class"], row["group"], row["coef"], row["tag"]))
    return rows


# -- identity ------------------------------------------------------------------------------------------------
def _freeze_target(model: MultivariateModel) -> dict[str, str]:
    """The target identity of a multivariate model: the responses' own targets (each a univariate model's digest: its
    formula, design, response, priors), their order, ``Y``, and the residual correlation and its prior."""
    from autobayes.identity import SCHEMA, _canonical, _prior

    Y = model.design.Y
    missing = np.isnan(Y)                                   # C06: the missing values are parameters; their rows enter instead
    payload = {"schema": SCHEMA, "family": model.family,
               "multivariate": {"formula": model.spec.canonical_formula, "responses": list(model.responses), "rescor": model.rescor,
                                "rescor_prior": _prior(model.priors.rescor) if model.rescor else None,
                                "Y": np.where(missing, 0.0, Y) if missing.any() else Y,
                                **({"missing": {r: np.flatnonzero(missing[:, k]) for k, r in enumerate(model.responses)}}
                                   if missing.any() else {}),
                                "targets": {r: sub.target_identity["sha256"] for r, sub in model.submodels.items()}}}
    encoded = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return {"schema": SCHEMA, "sha256": hashlib.sha256(encoded).hexdigest()}
