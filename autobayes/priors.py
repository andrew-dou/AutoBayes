"""Explicit priors, by parameter class, in brms terms.

brms classes: ``Intercept``, ``b`` (slopes and dummy coefficients), ``sd`` (one group SD per group-level
coefficient: the varying intercept of ``(1 | g)``, the intercept and each slope of ``(1 + x || g)``), ``cor`` (the
LKJ prior on the correlation matrix of the coefficients of ``(1 + x | g)``, brms's ``lkj(eta)``; A02) and the
family's auxiliary parameter: ``sigma`` for Gaussian/Student-t (residual SD/scale respectively) and for the
lognormal (the SD of ``log y``), ``shape`` for the negative binomial and Gamma families, ``phi`` for the Beta family
(the precision), and ``nu`` for a Student-t model that learns its degrees of freedom (``nu=None``; brms class ``nu``,
truncated to ``nu > 1``); ``zi`` or ``hu``, the zero-inflation or hurdle probability (B06, a ``Beta`` prior). An
auxiliary parameter with its own formula (B01, ``["y ~ x", "sigma ~ z"]``) takes the priors of that formula's intercept
and coefficients instead, ``sigma={"intercept": ..., "b": ...}`` (brms's ``Intercept`` and ``b`` with ``dpar = sigma``).
Every class the model has must be given, and a class the model does
not have is refused (M3c-1: ``sd`` without a group term; ``shape`` for a Gaussian model); there are no silent
defaults here (Bambi would supply some, and that is exactly what this layer refuses to let happen). brms's own
defaults are available on request, explicitly: ``ab.default_priors(formula, data, ...)`` returns them as a ``Priors``
(A07, ``autobayes.default_priors``), each prior marked ``source="default"``. An intercept-only model has no ``b``
coefficient and needs no ``b`` prior (D2). For the
Bernoulli family the intercept and slope priors are on the logit scale and ``sigma`` must be omitted.

The parameterisation follows PyMC/Bambi: ``Normal(mu, sigma)`` is mean and standard
deviation, ``StudentT(nu, mu, sigma)`` has a scale ``sigma``, ``Exponential(lam)`` is a
rate, ``HalfCauchy(beta)`` a scale, ``Gamma(alpha, beta)`` is shape and rate, ``InverseGamma(alpha, beta)`` shape
and scale. This is also brms's (Stan's) convention. ``Flat()`` is brms's flat prior (its default for ``b``): improper,
it adds nothing to the log density. ``brms_prior`` writes any of them in brms's syntax. ``Beta(alpha, beta)`` (B06) has
the two shapes of Stan's ``beta(alpha, beta)``, on (0, 1).
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

_LOCATION = {"Normal", "StudentT", "Flat"}       # Flat: brms's improper flat prior, its default for b (A07)
_SCALE = {"HalfNormal", "HalfStudentT", "Exponential", "HalfCauchy", "Gamma", "InverseGamma"}
_CORRELATION = {"LKJ"}
_UNIT = {"Beta"}                                 # priors on (0, 1): the zero-inflation and hurdle probabilities (B06)
# The Priors fields a family's scale_parameter can name, and (B06) the probabilities a family's zero_parameter can name
AUXILIARY = {"sigma": "residual SD", "shape": "shape parameter", "phi": "precision parameter",
             "zi": "zero-inflation probability", "hu": "hurdle probability"}
# The priors of a learned parameter with a lower bound (the Student-t nu, A13): the positive-support priors whose
# truncation at the bound PyMC can renormalise (it needs their CDF; PyMC has none for HalfStudentT).
TRUNCATABLE = frozenset({"Gamma", "Exponential", "HalfNormal", "HalfCauchy"})
_NO_CDF = frozenset({"HalfStudentT"})            # positive-support priors whose CDF PyMC lacks, so no truncation of them


class PriorError(ValueError):
    """A missing, extra, or ill-typed prior."""


@dataclass(frozen=True)
class Prior:
    name: str
    params: Mapping[str, float] = field(default_factory=dict)
    # Who chose the prior: "user" (a constructor call) or "default" (``ab.default_priors``, brms's default for its class;
    # A07). It is not part of the prior: equality, ``str()``, the target identity and the graph ignore it, while
    # ``describe()``, ``Priors.table`` and fit records report it.
    source: str = field(default="user", compare=False, repr=False)

    def __str__(self) -> str:
        inside = ", ".join(f"{k}={v:g}" for k, v in self.params.items())
        return f"{self.name}({inside})"

    def to_bambi(self):
        import bambi as bmb
        import numpy as np

        if self.name in _UNIT:
            # B06: float64 parameters. PyTensor makes a Python 2.0 a float32 constant, and the Beta's normalising constant
            # lgamma(a) + lgamma(b) - lgamma(a + b) would then lose 2e-8 at Beta(2, 2).
            return bmb.Prior(self.name, **{k: np.float64(v) for k, v in self.params.items()})
        return bmb.Prior(self.name, **dict(self.params))

    def to_bambi_truncated(self, lower: float):
        """Bambi's prior for a parameter bounded below by ``lower`` (brms's ``real<lower=1> nu``, A13): this distribution
        truncated to ``(lower, inf)`` and renormalised there, ``pm.Truncated``, which PyMC samples as
        ``log(value - lower)`` (its interval transform, whose Jacobian is that coordinate), as Stan does. Bambi builds an
        auxiliary parameter from a ``Prior`` with ``dist`` by calling ``dist(name, **params)``."""
        import bambi as bmb
        import pymc as pm

        base = getattr(pm, self.name)

        def truncated(name, **params):
            return pm.Truncated(name, base.dist(**params), lower=lower)

        return bmb.Prior(f"Truncated{self.name}", dist=truncated, **dict(self.params))


def _positive(name: str, value: float) -> float:
    if not value > 0:
        raise PriorError(f"{name} must be positive, got {value!r}")
    return float(value)


def Normal(mu: float, sigma: float) -> Prior:
    return Prior("Normal", {"mu": float(mu), "sigma": _positive("sigma", sigma)})


def StudentT(nu: float, mu: float, sigma: float) -> Prior:
    return Prior("StudentT", {"nu": _positive("nu", nu), "mu": float(mu), "sigma": _positive("sigma", sigma)})


def HalfNormal(sigma: float) -> Prior:
    return Prior("HalfNormal", {"sigma": _positive("sigma", sigma)})


def HalfStudentT(nu: float, sigma: float) -> Prior:
    return Prior("HalfStudentT", {"nu": _positive("nu", nu), "sigma": _positive("sigma", sigma)})


def Exponential(lam: float) -> Prior:
    return Prior("Exponential", {"lam": _positive("lam", lam)})


def HalfCauchy(beta: float) -> Prior:
    return Prior("HalfCauchy", {"beta": _positive("beta", beta)})


def Gamma(alpha: float, beta: float) -> Prior:
    """Shape ``alpha`` and rate ``beta`` (mean ``alpha / beta``), as Stan's ``gamma(alpha, beta)`` and PyMC's ``Gamma``."""
    return Prior("Gamma", {"alpha": _positive("alpha", alpha), "beta": _positive("beta", beta)})


def InverseGamma(alpha: float, beta: float) -> Prior:
    """Shape ``alpha`` and scale ``beta`` (density proportional to ``x^(-alpha - 1) exp(-beta / x)``), as Stan's
    ``inv_gamma(alpha, beta)`` and PyMC's ``InverseGamma``; brms's default prior on the negative binomial ``shape`` is
    ``inv_gamma(0.4, 0.3)``."""
    return Prior("InverseGamma", {"alpha": _positive("alpha", alpha), "beta": _positive("beta", beta)})


def Flat() -> Prior:
    """brms's flat prior, an empty prior string (its default for ``b``, and for the intercept when predictors are not
    centred): the improper uniform density on the real line, PyMC's ``Flat``, which adds nothing to the log density. A
    location prior (intercept and ``b``); a model with one has no prior predictive distribution, so
    ``prior_predictive()`` refuses it."""
    return Prior("Flat", {})


def Beta(alpha: float, beta: float) -> Prior:
    """A prior on (0, 1) with shapes ``alpha`` and ``beta`` (mean ``alpha / (alpha + beta)``), as Stan's ``beta(alpha,
    beta)`` and PyMC's ``Beta``: the prior of a zero-inflation or hurdle probability (``Priors(zi=...)``, ``Priors(hu=...)``;
    B06), whose brms default is ``beta(1, 1)``, the uniform. Not accepted for any other class."""
    return Prior("Beta", {"alpha": _positive("alpha", alpha), "beta": _positive("beta", beta)})


_BRMS = {   # prior name -> (brms's distribution, its arguments in Stan's order: a parameter name, or a literal number)
    "Normal": ("normal", ("mu", "sigma")), "StudentT": ("student_t", ("nu", "mu", "sigma")),
    "HalfNormal": ("normal", (0, "sigma")), "HalfStudentT": ("student_t", ("nu", 0, "sigma")),
    "HalfCauchy": ("cauchy", (0, "beta")), "Exponential": ("exponential", ("lam",)),
    "Gamma": ("gamma", ("alpha", "beta")), "InverseGamma": ("inv_gamma", ("alpha", "beta")), "LKJ": ("lkj", ("eta",)),
    "Beta": ("beta", ("alpha", "beta")),
}


def brms_prior(prior: Prior) -> str:
    """The prior in brms's syntax with its numbers as R prints them: ``normal(0, 5)``, ``student_t(3, 0.9, 2.5)``,
    ``gamma(2, 0.1)``, ``inv_gamma(0.4, 0.3)``, ``lkj(1)``; a half prior is the full distribution at location 0, as brms
    writes it on a class bounded below by 0 (``HalfStudentT(3, 2.5)`` is ``student_t(3, 0, 2.5)``, ``HalfNormal(1)``
    ``normal(0, 1)``); ``Flat()`` is the empty string, brms's flat prior."""
    if prior.name == "Flat":
        return ""
    if prior.name not in _BRMS:
        raise PriorError(f"{prior} has no brms spelling")
    from autobayes.default_priors import r_number

    fn, args = _BRMS[prior.name]
    return f"{fn}({', '.join(r_number(float(prior.params[a] if isinstance(a, str) else a)) for a in args)})"


def describe_prior(prior: Prior, lb: float | None = None, ub: float | None = None) -> str:
    """How ``describe()`` prints a prior: ``str(prior)``, followed for a brms default (``ab.default_priors``) by brms's
    spelling and bounds, ``HalfStudentT(nu=3, sigma=2.5)   (brms default student_t(3, 0, 2.5), lb 0)``. A user's prior
    prints as it always has."""
    if getattr(prior, "source", "user") != "default":
        return str(prior)
    text = brms_prior(prior)
    if not text:
        return f"{prior}   (brms default: flat, improper)"
    from autobayes.default_priors import r_number

    return f"{prior}   (brms default {text}{'' if lb is None else ', lb ' + r_number(lb)}{'' if ub is None else ', ub ' + r_number(ub)})"


def LKJ(eta: float) -> Prior:
    """brms's ``lkj(eta)`` on the correlation matrix of a correlated group term, placed on its Cholesky factor as
    Stan's ``lkj_corr_cholesky(eta)``: ``eta = 1`` is uniform over correlation matrices, ``eta > 1`` concentrates on
    the identity (for two coefficients the correlation is ``2 Beta(eta, eta) - 1``)."""
    return Prior("LKJ", {"eta": _positive("eta", eta)})


@dataclass(frozen=True)
class Priors:
    """Priors for the parameter classes.

    ``b`` is either one prior applied to every coefficient (numeric slopes and the dummy
    coefficients of categorical predictors) or a mapping from formula term to prior, which
    must cover every predictor term exactly. For a categorical term the entry is one prior
    (the same for every non-reference level) or a mapping ``{level: prior}`` covering exactly
    the actual coefficient levels (all levels for the first categorical main effect
    without an intercept; non-reference levels otherwise); an intercept-only formula has no coefficient, and ``b``
    may then be omitted (brms needs no ``b`` prior there either, D2). Omit intercept exactly
    when the formula removes it. ``sd`` is keyed by grouping factor (an exact-key mapping; a single
    intercept-only factor also accepts a bare Prior): an intercept-only factor takes one Prior (or the
    one-entry mapping ``{"Intercept": prior}``), a factor with varying slopes a mapping over exactly its
    coefficients, ``{"g": {"Intercept": HalfNormal(1), "x": HalfNormal(0.5)}}`` (brms ``class = sd, group =
    g, coef = x``). ``cor`` is keyed by grouping factor too and covers exactly the correlated group terms
    ``(1 + x | g)``: ``cor={"g": LKJ(2)}`` (brms ``class = cor, group = g``; a single correlated factor also accepts
    the bare prior); it is refused for an independent (``||``) or intercept-only term, which has no correlation
    matrix. No automatic priors are filled in silently and nothing is broadcast. ``sigma``, ``shape`` and
    ``phi`` are the auxiliary parameters: exactly the one the family has (``FamilySpec.scale_parameter``) must be given.
    When that parameter has its own formula (B01, ``["y ~ x", "sigma ~ z"]``, docs/distributional.md) its entry is instead
    a mapping over that formula's intercept and coefficients, ``sigma={"intercept": Normal(0, 1), "b": Normal(0, 0.5)}``
    (brms ``class = Intercept`` and ``class = b`` with ``dpar = sigma``), read as the response formula's ``intercept`` and
    ``b`` are (``dpar_priors``): ``"intercept"`` exactly when that formula has one, ``"b"`` one prior or a mapping by term.
    ``nu`` is the prior of a Student-t model's learned degrees of freedom (``ab.student(..., nu=None)``): one of
    ``TRUNCATABLE``, truncated to ``nu > 1`` and renormalised there (brms-parity A13); refused for every other model.
    ``zi`` and ``hu`` are the zero-inflation and hurdle probabilities (B06): exactly the one the family has
    (``FamilySpec.zero_parameter``) must be given, a prior on (0, 1) (``Beta``), and is refused for every other family.
    """

    intercept: Prior | None = None
    b: Prior | Mapping[str, Prior | Mapping[str, Prior]] | None = None
    sd: Prior | Mapping[str, Prior | Mapping[str, Prior]] | None = None   # required with a group term, refused without one (M3c-1)
    # Each auxiliary parameter: one prior, or with its own formula (B01) a mapping {"intercept": ..., "b": ...} over it
    sigma: Prior | Mapping[str, object] | None = None       # Gaussian, Student-t and lognormal
    shape: Prior | Mapping[str, object] | None = None       # negative binomial and Gamma (brms class ``shape``)
    phi: Prior | Mapping[str, object] | None = None         # Beta (brms class ``phi``, the precision)
    # LKJ(eta) per correlated group term ``(1 + x | g)`` (A02), refused otherwise; after the classic fields, so the positional order
    # ``Priors(intercept, b, sd, sigma)`` existing callers use is unchanged
    cor: Prior | Mapping[str, Prior] | None = None
    nu: Prior | None = None          # Student-t with nu=None: the learned degrees of freedom (brms class ``nu``)
    zi: Prior | None = None          # zero-inflated families: the zero-inflation probability (brms class ``zi``; B06)
    hu: Prior | None = None          # hurdle families: the hurdle probability (brms class ``hu``; B06)

    # -- coefficient priors -----------------------------------------------------------------
    def coefficient_priors(self, spec) -> dict[str, Prior]:
        """One prior per design column, keyed by column label, in design order. ``spec`` is a
        ``ModelSpec`` (or a tuple of numeric predictor names); categorical levels must be
        known (data seen) for per-level mappings to be checked."""
        from autobayes.formula import ModelSpec

        if not isinstance(spec, ModelSpec):
            spec = ModelSpec(formula="", response="", predictors=tuple(spec), group=None)
        if isinstance(self.b, Prior):
            return {col: self.b for col in spec.columns}
        if self.b is None and not spec.pop_terms:             # an intercept-only model: no coefficient, so no b prior (D2)
            return {}
        if not isinstance(self.b, Mapping):
            raise PriorError(f"b must be a Prior or a mapping from term to prior, got {type(self.b).__name__}")
        given = dict(self.b)
        missing = [t for t in spec.terms if t not in given]
        extra = [k for k in given if k not in spec.terms]
        if missing or extra:
            hint = ""
            for k in extra:
                stripped = re.sub(r"\[[^\]]*\]", "", k) if isinstance(k, str) else k    # 'z[q]:h[h2]' -> 'z:h', 'x:z[q]' -> 'x:z'
                if isinstance(k, str) and "[" in k and any(t.name == stripped and t.is_vector for t in spec.pop_terms):
                    hint = (f"; {k!r} looks like a design-column label: give per-level priors as "
                            f"b={{{stripped!r}: {{level: prior}}}}")
                    break
                if isinstance(k, str) and ":" in k and ":".join(reversed(k.split(":", 1))) in spec.interaction_names:
                    canonical = spec.pop_term(":".join(reversed(k.split(":", 1))))
                    order = "numeric first" if len(canonical.categorical) == 1 else "the main effects' order"
                    hint = f"; write the interaction key in its canonical spelling {canonical.name!r} ({order})"
                    break
            raise PriorError(
                f"coefficient priors must cover exactly the predictor terms {list(spec.terms)}; "
                f"missing {missing}, unexpected {extra}{hint}"
            )
        out: dict[str, Prior] = {}
        for t in spec.pop_terms:
            term, entry = t.name, given[t.name]
            if not t.is_vector:
                if not isinstance(entry, Prior):
                    raise PriorError(
                        f"predictor {term!r} takes one prior, got {type(entry).__name__}; a mapping {{level: prior}} "
                        f"is only for a categorical term, declared with categorical={{{term!r}: <reference level>}}"
                    )
                out[term] = entry
                continue
            cats = [spec.categorical(c) for c in t.categorical]   # the categorical side(s) of a categorical or interaction term
            cat = cats[-1]
            if isinstance(entry, Prior):
                for col in t.columns:
                    out[col] = entry
                continue
            if not isinstance(entry, Mapping):
                raise PriorError(
                    f"categorical predictor {term!r} takes one prior or a mapping {{level: prior}}, "
                    f"got {type(entry).__name__}"
                )
            keyed: dict[str, Prior] = {}
            for k, v in entry.items():
                try:
                    from autobayes.formula import FormulaError, _label
                    lab = _label(k)               # the same scalar rule as the data (no float intermediate)
                except FormulaError as e:         # bool / complex / fractional / large-float keys
                    raise PriorError(f"prior key {k!r} for {term!r} is not a level label: {e}") from None
                if lab in keyed:
                    raise PriorError(f"ambiguous prior keys for {term!r}: more than one key maps to level {lab!r}")
                keyed[lab] = v
            if not cat.levels:                     # structure only; coverage needs the data
                for lab, v in keyed.items():
                    if not isinstance(v, Prior):
                        raise PriorError(f"prior for {term!r} level {lab!r} must be a Prior, got {type(v).__name__}")
                continue
            wanted = list(t.levels)
            missing_l = [l for l in wanted if l not in keyed]
            extra_l = [l for l in keyed if l not in wanted]
            if missing_l or extra_l:
                if len(cats) == 2:                   # z:h: one key per cell 'l, m' of non-reference levels
                    z, h = cats
                    raise PriorError(
                        f"per-cell priors for {term!r} must cover exactly its cells {wanted} (each 'l, m' pairs a non-reference level "
                        f"of {z.name!r} with one of {h.name!r}; the references {z.reference!r} and {h.reference!r} have no cell coefficient); "
                        f"missing {missing_l}, unexpected {extra_l}"
                    )
                ref_note = (f" ({cat.reference!r} is the reference level and has no coefficient)"
                            if cat.reference in extra_l and cat.reference not in wanted else "")
                raise PriorError(
                    f"per-level priors for {term!r} must cover exactly the {'coefficient levels' if cat.reference in wanted else 'non-reference levels'} {wanted}; "
                    f"missing {missing_l}, unexpected {extra_l}{ref_note}"
                )
            names = {keyed[l].name for l in wanted if isinstance(keyed[l], Prior)}
            if len(names) > 1:
                raise PriorError(
                    f"per-level priors of {term!r} must share one distribution family (Bambi takes one "
                    f"prior per term, with per-level parameters); got {sorted(names)}"
                )
            for l, col in zip(wanted, t.columns):             # 'z[q]' and 'x:z[q]' for level q; 'z[q]:h[h2]' for cell 'q, h2'
                out[col] = keyed[l]
        return out

    def slope_priors(self, predictors: tuple[str, ...]) -> dict[str, Prior]:
        """Numeric-only compatibility form of ``coefficient_priors``."""
        return self.coefficient_priors(tuple(predictors))

    def sd_priors(self, spec) -> dict[tuple[str, str], Prior]:
        """One SD prior per group-level coefficient, keyed ``(factor, coefficient)`` in formula order (the order
        of ``Design.group_blocks``). Coverage is checked exactly, by factor and then by coefficient; the
        values are checked to be positive-support priors by ``validate``. No scalar is broadcast."""
        group_terms = getattr(spec, "group_terms", ())
        groups = [t.factor for t in group_terms]
        if not group_terms:
            if self.sd is not None:
                raise PriorError("the formula has no group-level term '(1 | group)'; do not pass sd")
            return {}
        if self.sd is None:
            if len(groups) == 1:
                t = group_terms[0]
                if t.slopes:
                    raise PriorError(
                        f"the group term '{t.canonical}' needs a prior on the SD of each of its coefficients "
                        f"{list(t.coefs)}: pass sd={{{t.factor!r}: {{{', '.join(f'{c!r}: <prior>' for c in t.coefs)}}}}}"
                    )
                raise PriorError(f"the varying intercept '(1 | {groups[0]})' needs a prior on the SD of its varying intercept: pass sd=...")
            raise PriorError(f"each varying intercept needs a prior on the SD: pass sd for {list(groups)}")
        if isinstance(self.sd, Mapping):
            if set(self.sd) != set(groups):
                hint = ""
                if len(group_terms) == 1 and set(self.sd) <= set(group_terms[0].coefs):
                    t = group_terms[0]
                    hint = (f"; {list(self.sd)} look like the coefficients of {t.factor!r}: nest them under the factor, "
                            f"sd={{{t.factor!r}: {{{', '.join(f'{c!r}: <prior>' for c in t.coefs)}}}}}")
                raise PriorError(f"sd must cover exactly the grouping factors {list(groups)}; got {list(self.sd)}{hint}")
            by_factor = {g: self.sd[g] for g in groups}
        else:
            if len(groups) != 1:
                raise PriorError("multiple grouping factors require an exact-key sd mapping; scalar sd is not broadcast")
            by_factor = {groups[0]: self.sd}
        out: dict[tuple[str, str], Prior] = {}
        for t in group_terms:
            entry = by_factor[t.factor]
            if isinstance(entry, Mapping):
                missing = [c for c in t.coefs if c not in entry]
                extra = [k for k in entry if k not in t.coefs]
                if missing or extra:
                    hint = ""
                    for k in extra:
                        if isinstance(k, str) and k in getattr(spec, "predictors", ()):
                            hint = (f" ({k!r} is a population-level term, but '{t.canonical}' has no varying slope on it; "
                                    f"write '(1 + {k} || {t.factor})' to add one)")
                            break
                    raise PriorError(
                        f"SD priors for {t.factor!r} must cover exactly its coefficients {list(t.coefs)}; "
                        f"missing {missing}, unexpected {extra}{hint}"
                    )
                for c in t.coefs:
                    out[(t.factor, c)] = entry[c]
            elif t.slopes:
                raise PriorError(
                    f"sd for {t.factor!r} must be a mapping over its coefficients {list(t.coefs)} ('{t.canonical}' has "
                    f"varying slopes), got a single {type(entry).__name__}; the same prior for every coefficient must be spelled out"
                )
            else:
                out[(t.factor, "Intercept")] = entry
        return out

    def cor_priors(self, spec) -> dict[str, Prior]:
        """The LKJ prior of each correlated group term (``(1 + x | g)``, two or more coefficients sharing a
        correlation matrix), keyed by factor in formula order. Coverage is exact: a factor whose term is independent
        (``||``) or intercept-only has no correlation matrix and is refused here, as is a missing or non-LKJ prior."""
        group_terms = tuple(getattr(spec, "group_terms", ()))
        terms = [t for t in group_terms if t.has_correlation]
        factors = [t.factor for t in terms]
        if not terms:
            if self.cor is not None:
                others = [t for t in group_terms if t.slopes]
                hint = (f"; '{others[0].canonical}' uses '||', independent slopes with no correlation matrix: write "
                        f"'{others[0].canonical.replace('||', '|')}' for brms's correlated form" if others else
                        "; only a group term with varying slopes written with '|', '(1 + x | g)', has a correlation matrix")
                raise PriorError(f"the formula has no correlated group term '(1 + x | g)'; do not pass cor{hint}")
            return {}
        if self.cor is None:
            t = terms[0]
            raise PriorError(
                f"the group term '{t.canonical}' gives its coefficients {list(t.coefs)} a correlation matrix, which needs "
                f"an LKJ prior: pass cor={{{t.factor!r}: ab.LKJ(eta)}}" + (f" (and for {factors[1:]})" if len(terms) > 1 else "")
            )
        if isinstance(self.cor, Mapping):
            if set(self.cor) != set(factors):
                extra = [k for k in self.cor if k not in factors]
                hint = ""
                for k in extra:
                    t = next((t for t in group_terms if t.factor == k), None)
                    if t is not None:
                        hint = (f" ('{t.canonical}' has no correlation matrix: " +
                                (f"write '{t.canonical.replace('||', '|')}' for correlated slopes)" if t.slopes else
                                 "one coefficient has nothing to correlate with)"))
                        break
                raise PriorError(f"cor must cover exactly the correlated grouping factors {factors}; got {list(self.cor)}{hint}")
            by_factor = {g: self.cor[g] for g in factors}
        else:
            if len(factors) != 1:
                raise PriorError(f"multiple correlated grouping factors require an exact-key cor mapping over {factors}; a bare prior is not broadcast")
            by_factor = {factors[0]: self.cor}
        for g, prior in by_factor.items():
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for 'cor:{g}' must be a Prior, got {type(prior).__name__}")
            if prior.name not in _CORRELATION:
                raise PriorError(f"prior for 'cor:{g}' must be a correlation prior ({sorted(_CORRELATION)}, brms's lkj(eta)), got {prior}")
        return by_factor

    def dpar_priors(self, spec, name: str) -> "Priors":
        """The priors of distributional parameter ``name``'s own formula (B01, ``spec.dpar(name)``) as a ``Priors`` with
        ``intercept`` and ``b`` only, read from ``Priors(<name>={"intercept": ..., "b": ...})``, so that the response
        formula's rules apply to it unchanged (``coefficient_priors(spec.dpar(name))``, location priors). ``"intercept"`` is
        required exactly when that formula has an intercept; ``"b"`` when it has a population-level term (an intercept-only
        formula needs none, and one given anyway changes nothing, as for the response formula, D2)."""
        dspec = spec.dpar(name)
        entry = getattr(self, name)
        brms = f"brms class Intercept and b with dpar = {name}"
        if entry is None:
            raise PriorError(
                f"the formula '{dspec.formula}' gives {name} its own linear predictor, which needs priors on its intercept and "
                f"coefficients: pass {name}={{'intercept': ..., 'b': ...}} ({brms}; docs/distributional.md)")
        if not isinstance(entry, Mapping):
            raise PriorError(
                f"{name} has its own formula '{dspec.formula}': its priors are a mapping {{'intercept': prior, 'b': prior}} over "
                f"that formula's intercept and coefficients ({brms}), not {type(entry).__name__} {entry}")
        extra = [k for k in entry if k not in ("intercept", "b")]
        if extra:
            raise PriorError(f"the priors of {name}'s formula '{dspec.formula}' take the keys 'intercept' and 'b' ({brms}); "
                             f"unexpected {extra}")
        if dspec.intercept and entry.get("intercept") is None:
            raise PriorError(f"'{dspec.formula}' has an intercept, which needs a prior: pass {name}={{'intercept': ..., ...}}")
        if not dspec.intercept and entry.get("intercept") is not None:
            raise PriorError(f"'{dspec.formula}' has no intercept; omit 'intercept' from the priors of {name}'s formula")
        if dspec.pop_terms and entry.get("b") is None:
            raise PriorError(f"'{dspec.formula}' has the population-level term(s) {list(dspec.terms)}, which need priors: pass "
                             f"{name}={{..., 'b': ...}} (one prior, or a mapping by term as for the response formula's b)")
        return Priors(intercept=entry.get("intercept"), b=entry.get("b"))

    def table(self, spec, family="gaussian", *, center_predictors: bool = True, categories=None) -> list[dict[str, str]]:
        """These priors in brms's table form, the rows of brms's ``default_prior()`` (prior, class, coef, group, resp,
        dpar, nlpar, lb, ub, tag, source; all strings), sorted as brms sorts them: ``autobayes.default_priors.prior_table``.
        ``spec`` is the parsed formula with its data seen (``model.spec``; ``Model.prior_table()`` passes the model's own),
        ``family`` a family name or the model's ``FamilySpec``, ``center_predictors`` whether the intercept is centred
        (otherwise it is brms's ``b`` coefficient ``Intercept``); ``categories`` the response's categories, which the
        ordinal and categorical families' tables need (``model.categories``; B05)."""
        from autobayes.default_priors import prior_table

        return prior_table(self, spec, family, center_predictors=center_predictors, categories=categories)

    def improper(self, spec) -> list[str]:
        """The parameters under a flat prior (``Flat()``, brms's default for ``b``), labelled as fit records label them
        (``intercept``, ``b:x``); such a model has no prior predictive distribution."""
        out = ["intercept"] if getattr(spec, "intercept", True) and isinstance(self.intercept, Prior) and self.intercept.name == "Flat" else []
        out += [f"b:{col}" for col, p in self.coefficient_priors(spec).items() if p.name == "Flat"]
        for name in getattr(spec, "dpar_names", ()):                  # a distributional parameter's formula (B01): sigma:b:zq
            out += [f"{name}:{label}" for label in self.dpar_priors(spec, name).improper(spec.dpar(name))]
        return out

    def group_priors(self, spec) -> dict[str, Prior]:
        """The SD prior of each factor's varying intercept, in formula order (the per-factor view that
        ``describe()``, fit records and the target identity have always used); ``sd_priors`` has every
        group-level coefficient, slopes included."""
        return {g: prior for (g, coef), prior in self.sd_priors(spec).items() if coef == "Intercept"}

    def validate(self, spec, family="gaussian") -> None:
        """``family`` is a registered family name, or the model's resolved ``FamilySpec`` (which carries its link)."""
        from autobayes.families import FamilySpec, get_family

        fam = family if isinstance(family, FamilySpec) else get_family(family)
        has_intercept = getattr(spec, "intercept", True)
        if has_intercept and self.intercept is None:
            raise PriorError("the formula has a population intercept; an intercept prior is required")
        if not has_intercept and self.intercept is not None:
            raise PriorError("the formula has no population intercept; omit the intercept prior (or pass None)")
        sd_priors = self.sd_priors(spec)
        self.cor_priors(spec)                                         # exact coverage of the correlated terms (A02)
        scale = fam.scale_parameter                                   # "sigma", "shape", "phi" or None
        zero = fam.zero_parameter                                     # "zi", "hu" or None (B06)
        auxiliary = {name: getattr(self, name) for name in AUXILIARY}
        predicted = scale is not None and scale in getattr(spec, "dpar_names", ())   # the scale parameter has its own formula (B01)
        if scale is not None and auxiliary[scale] is None and not predicted:
            raise PriorError(fam.scale_missing_error)
        if zero is not None and auxiliary[zero] is None:
            raise PriorError(f"the {fam.name} family needs a prior on the {AUXILIARY[zero]} {zero}: pass {zero}=..., "
                             "a prior on (0, 1) such as ab.Beta(1, 1)")
        for name, prior in auxiliary.items():
            if prior is not None and name not in (scale, zero):
                raise PriorError(
                    f"the {fam.name} family has no {AUXILIARY[name]}; do not pass {name}."
                    + (f" Its auxiliary parameter is {scale}: pass {scale}=..." if scale is not None else "")
                    + (f" Its {AUXILIARY[zero]} is {zero}: pass {zero}=..." if zero is not None else "")
                    + (f" Its intercept and slope priors are on the {fam.link_scale} scale." if fam.link_scale is not None else "")
                )
        if self.nu is not None:                                       # the learned Student-t degrees of freedom (A13)
            if "nu" not in fam.learnable_parameters:
                raise PriorError(
                    f"the {fam.name} family has no degrees of freedom; do not pass nu."
                    + (f" Its auxiliary parameter is {scale}: pass {scale}=..." if scale is not None else "")
                )
            lower = fam.learnable_parameters["nu"]
            if not isinstance(self.nu, Prior):
                raise PriorError(f"prior for 'nu' must be a Prior, got {type(self.nu).__name__}")
            if self.nu.name in _SCALE and self.nu.name not in TRUNCATABLE:
                why = (f"PyMC has no CDF for it, so its truncation to nu > {lower:g} cannot be renormalised as brms's is"
                       if self.nu.name in _NO_CDF else f"its truncation to nu > {lower:g} is not one this library renormalises")
                raise PriorError(f"prior for 'nu' cannot be {self.nu.name}: {why}; use one of {sorted(TRUNCATABLE)}")
            if self.nu.name not in TRUNCATABLE:
                raise PriorError(
                    f"prior for 'nu' must be a positive-support prior {sorted(TRUNCATABLE)}; got {self.nu} (nu is learned "
                    f"on nu > {lower:g}, under its prior truncated there and renormalised, as brms does)"
                )
        location = ([("intercept", self.intercept)] if has_intercept else []) + list(self.coefficient_priors(spec).items())
        if predicted:                                                 # its formula's intercept and coefficients (B01): location priors
            dspec, sub = spec.dpar(scale), self.dpar_priors(spec, scale)
            location += (([(f"{scale}:intercept", sub.intercept)] if dspec.intercept else [])
                         + [(f"{scale}:b:{col}", p) for col, p in sub.coefficient_priors(dspec).items()])
        elif scale is not None and isinstance(auxiliary[scale], Mapping):
            raise PriorError(
                f"prior for {scale!r} is a mapping, the form of the priors of {scale}'s own formula ('{scale} ~ ...', B01); "
                f"this model gives {scale} none: pass one positive-support prior, or add the formula "
                f"(['<response formula>', '{scale} ~ ...'], docs/distributional.md)")
        for label, prior in location:
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for {label!r} must be a Prior, got {type(prior).__name__}")
            if prior.name not in _LOCATION:
                raise PriorError(
                    f"prior for {label!r} must be a location prior ({sorted(_LOCATION)}), got {prior}"
                )
        from autobayes.formula import sd_tag

        scales = [(f"sd:{sd_tag(g, c)}", prior) for (g, c), prior in sd_priors.items()] + ([(scale, auxiliary[scale])] if scale is not None and not predicted else [])
        for label, prior in scales:
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for {label!r} must be a Prior, got {type(prior).__name__}")
            if prior.name not in _SCALE:
                raise PriorError(
                    f"prior for {label!r} must be a positive-support prior ({sorted(_SCALE)}), got {prior}"
                )
        if zero is not None:                                          # the zero-inflation / hurdle probability (B06)
            prior = auxiliary[zero]
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for {zero!r} must be a Prior, got {type(prior).__name__}")
            if prior.name not in _UNIT:
                raise PriorError(f"prior for {zero!r} must be a prior on (0, 1) ({sorted(_UNIT)}; the {AUXILIARY[zero]} "
                                 f"is brms's real<lower=0,upper=1> {zero}), got {prior}")

    def to_bambi(self, spec, family="gaussian") -> dict:
        """Bambi's prior dict: one prior per term (array-valued parameters for the levels of a
        categorical term). Each group-level coefficient (``1|g``, and ``x|g`` for a varying slope) gets
        Normal(0, sd) with its own ``sd`` as hyperprior; without a group term no group prior is emitted
        (Bambi would accept a stray ``1|g`` entry silently). ``family`` as for ``validate``. An auxiliary parameter with its
        own formula (B01) gets the dict of its formula's term priors, which Bambi reads for that distributional component."""
        import bambi as bmb
        from autobayes.families import FamilySpec, get_family
        from autobayes.formula import ModelSpec, bambi_group_name

        self.validate(spec, family)
        fam = family if isinstance(family, FamilySpec) else get_family(family)
        if not isinstance(spec, ModelSpec):
            spec = ModelSpec(formula="", response="", predictors=tuple(spec), group=None)
        out = {"Intercept": self.intercept.to_bambi()} if spec.intercept else {}
        if fam.scale_parameter in spec.dpar_names:         # its own formula (B01): Bambi's distributional component's priors
            sub, dspec = self.dpar_priors(spec, fam.scale_parameter), spec.dpar(fam.scale_parameter)
            out[fam.bambi_scale_parameter] = {**({"Intercept": sub.intercept.to_bambi()} if dspec.intercept else {}),
                                              **sub._bambi_terms(dspec)}
        elif fam.scale_parameter is not None:              # under Bambi's name: "sigma", or "alpha" for brms's "shape"
            out[fam.bambi_scale_parameter] = getattr(self, fam.scale_parameter).to_bambi()
        if fam.zero_parameter is not None:                  # the zero-inflation / hurdle probability, under brms's name (B06)
            out[fam.zero_parameter] = getattr(self, fam.zero_parameter).to_bambi()
        if self.nu is not None:                             # a learned Student-t nu (A13): its prior truncated at the bound
            out["nu"] = self.nu.to_bambi_truncated(fam.learnable_parameters["nu"])
        out.update(self._bambi_terms(spec))
        for (group, coef), prior in self.sd_priors(spec).items():
            out[bambi_group_name(group, coef)] = bmb.Prior("Normal", mu=0.0, sigma=prior.to_bambi())
        return out

    def _bambi_terms(self, spec) -> dict:
        """Bambi's priors of the population-level terms of ``spec`` (a formula's own, the response's or a distributional
        parameter's): one prior per term, with array-valued parameters for the levels of a vector term."""
        import bambi as bmb
        import numpy as np

        per_column = self.coefficient_priors(spec)
        out = {}
        for term in spec.pop_terms:
            if not term.is_vector:
                out[term.name] = per_column[term.name].to_bambi()
                continue
            priors = [per_column[col] for col in term.columns]
            name = priors[0].name
            params = {k: np.array([float(pr.params[k]) for pr in priors]) for k in priors[0].params}
            out[term.name] = bmb.Prior(name, **params)
        return out
