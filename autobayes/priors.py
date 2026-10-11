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
from dataclasses import dataclass, field, replace

import numpy as np

_LOCATION = {"Normal", "StudentT", "Flat"}       # Flat: brms's improper flat prior, its default for b (A07)
_SCALE = {"HalfNormal", "HalfStudentT", "Exponential", "HalfCauchy", "Gamma", "InverseGamma"}
_CORRELATION = {"LKJ"}
_UNIT = {"Beta"}                                 # priors on (0, 1): the zero-inflation and hurdle probabilities (B06)
_SIMPLEX = {"Dirichlet"}                         # priors on a simplex: a monotonic term's simo (B03)
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


def Dirichlet(*alpha) -> Prior:
    """brms's ``dirichlet(...)`` on the simplex of a monotonic term ``mo(x)`` (B03, ``Priors(simo={"x": ...})``, brms class
    ``simo``): ``Dirichlet(2, 2, 2)`` or ``Dirichlet([2, 2, 2])`` gives one positive concentration per step of the term,
    and one number, ``Dirichlet(1)``, is brms's shorthand for the same concentration on every step (brms's default
    ``dirichlet(1)``, the uniform distribution on the simplex). Its density is Stan's ``dirichlet_lpdf`` with its
    normalising constant. Not accepted for any other class."""
    values = alpha[0] if len(alpha) == 1 and isinstance(alpha[0], (list, tuple, np.ndarray)) else alpha
    values = [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]
    if not values:
        raise PriorError("Dirichlet needs at least one concentration")
    if not all(np.isfinite(v) and v > 0 for v in values):
        raise PriorError(f"Dirichlet concentrations must be positive finite numbers, got {values}")
    if len(values) == 1:                                       # brms's shorthand: the same concentration on every step
        return Prior("Dirichlet", {"a": values[0]})
    return Prior("Dirichlet", {f"a[{k}]": v for k, v in enumerate(values, start=1)})


def dirichlet_concentration(prior: Prior, D: int) -> np.ndarray:
    """The ``D`` concentrations of a ``Dirichlet`` prior on a simplex of ``D`` entries (brms's ``con_simo``): one number
    repeated, or the vector given, which must have ``D`` entries (brms's ``eval_dirichlet``)."""
    values = list(prior.params.values())
    if len(values) == 1 and "a" in prior.params:
        return np.full(int(D), float(values[0]))
    if len(values) != int(D):
        raise PriorError(f"{prior} gives {len(values)} concentrations for a simplex of {D} steps; give {D}, or one number for "
                         "the same concentration on every step (brms: 'Invalid Dirichlet prior. Expected input of length "
                         f"{D}.')")
    return np.asarray(values, dtype=float)


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
    ``normal(0, 1)``); ``Flat()`` is the empty string, brms's flat prior; ``Dirichlet(2, 2, 2)`` is ``dirichlet(2, 2, 2)``
    and ``Dirichlet(1)`` brms's ``dirichlet(1)`` (B03)."""
    if prior.name == "Flat":
        return ""
    from autobayes.default_priors import r_number

    if prior.name == "Dirichlet":                         # its concentrations in order, or brms's one-number shorthand
        return f"dirichlet({', '.join(r_number(float(v)) for v in prior.params.values())})"
    if prior.name not in _BRMS:
        raise PriorError(f"{prior} has no brms spelling")
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
    A monotonic term ``mo(x)`` (B03, docs/monotonic.md) has a coefficient of class ``b`` (brms's ``bsp``, coefficient
    ``mox``), which ``b`` covers under the term's name, ``b={"mo(x)": Normal(0, 2)}``, and a simplex whose prior is ``simo``,
    a ``Dirichlet`` keyed by the term's column, ``simo={"x": Dirichlet(2, 2, 2)}`` (brms ``class = simo, coef = mox1``; a
    single monotonic term also accepts the bare prior), with one concentration per step of the term or one number for all;
    refused without a monotonic term.
    A mixture (C05, ``ab.mixture``, docs/mixtures.md) takes ``components``, one ``Priors`` per component in the order of
    ``families``, each the component family's own (a Gaussian component's ``intercept``, ``b`` and ``sigma``: brms's
    ``class = Intercept`` and ``b`` with ``dpar = mu<k>``, and ``class = sigma<k>``), and ``theta``, the ``Dirichlet`` prior of
    the mixing proportions (brms class ``theta``; one concentration per component or one number for all); nothing else at
    the top level. Per class, the components' priors share one distribution family (the intercepts are one ordered vector
    and each term one variable with a component axis), each with its own parameters.

    A multivariate model (C01, docs/multivariate.md, ``ab.gaussian("mvbind(y, y2) ~ ...", ..., rescor=...)``) takes its
    priors by response, as brms's ``resp =``: ``resp={"y": {...}, "y2": {...}}`` holds each response's classes (a mapping
    of the fields above, or a ``Priors``), ``rescor`` the LKJ prior of the residual correlation (brms ``class = rescor``,
    ``rescor=True`` only), and ``cor`` stays at the top level, keyed by factor and shared by the responses' correlated terms
    on that factor (brms's ``cor`` has no ``resp``). Both are refused for a model with one response.
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
    simo: Prior | Mapping[str, Prior] | None = None   # the simplex of each monotonic term mo(x), by column (brms class ``simo``; B03)
    # A mixture (C05, docs/mixtures.md): one Priors per component, the component family's own (intercept, b, sigma: brms's
    # dpar = mu<k> and class sigma<k>), and the mixing proportions' Dirichlet (brms class ``theta``); refused otherwise
    components: tuple | list | None = None
    theta: Prior | None = None
    resp: Mapping[str, object] | None = None   # C01: a multivariate model's priors by response (brms ``resp =``), each a mapping or Priors
    rescor: Prior | None = None               # C01: LKJ(eta) on the residual correlation of a multivariate model (brms class ``rescor``)

    def __post_init__(self):
        if isinstance(self.components, list):                         # one value for equal component lists (C05)
            object.__setattr__(self, "components", tuple(self.components))

    # -- multivariate models (C01, docs/multivariate.md) ---------------------------------------------
    _RESPONSE_CLASSES = ("intercept", "b", "sd", "sigma", "shape", "phi", "nu", "zi", "hu", "simo")

    def response_priors(self, response: str) -> "Priors":
        """The priors of one response of a multivariate model (C01): its entry of ``resp`` as a ``Priors`` (a mapping of
        the response classes is converted), with the top-level ``cor`` (brms's ``cor`` has no ``resp``: one prior per
        factor, shared by every response's correlated term on it). Refused when ``resp`` has no entry for ``response``."""
        if not isinstance(self.resp, Mapping) or response not in self.resp:
            raise PriorError(f"no priors for the response {response!r}: pass resp={{{response!r}: {{...}}, ...}}")
        entry = self.resp[response]
        if isinstance(entry, Priors):
            if entry.resp is not None or entry.rescor is not None:
                raise PriorError(f"the priors of the response {response!r} cannot hold resp or rescor themselves")
            if entry.cor is not None:
                raise PriorError(f"the priors of the response {response!r} set cor: brms's correlation prior (class cor) has no "
                                 "resp; pass it once at the top level, Priors(resp={...}, cor={'g': ab.LKJ(eta)}), for every response")
            sub = entry
        elif isinstance(entry, Mapping):
            keys = [k for k in entry if not isinstance(k, str) or k not in self._RESPONSE_CLASSES]
            if keys:
                hint = (" (brms's correlation prior, class cor, has no resp: pass it once at the top level, "
                        "Priors(resp={...}, cor={'g': ab.LKJ(eta)}))" if "cor" in keys else "")
                raise PriorError(f"the priors of the response {response!r} take the classes {list(self._RESPONSE_CLASSES)}; "
                                 f"unexpected {keys}{hint}")
            sub = Priors(**dict(entry))
        else:
            raise PriorError(f"the priors of the response {response!r} must be a mapping of its classes "
                             f"({{'intercept': ..., 'b': ..., 'sd': ..., 'sigma': ...}}) or a Priors, got {type(entry).__name__}")
        return replace(sub, cor=self.cor)

    def validate_multivariate(self, spec, family="gaussian") -> None:
        """A multivariate model's priors (C01): ``resp`` covers exactly the responses ``spec.responses``, each response's
        priors are valid for its own formula (``response_priors``, then ``validate`` against ``spec.specs[response]``),
        ``rescor`` is an ``LKJ`` exactly when the model has a residual correlation (``spec.rescor``), and nothing else is
        given at the top level but ``cor``, which the responses share."""
        from autobayes.families import FamilySpec, get_family

        fam = family if isinstance(family, FamilySpec) else get_family(family)
        responses = list(spec.responses)
        if self.components is not None or self.theta is not None:
            raise PriorError("components and theta are the priors of a mixture (ab.mixture, docs/mixtures.md); "
                             "a multivariate model takes neither")
        top = [name for name in self._RESPONSE_CLASSES if getattr(self, name) is not None]
        if top:
            raise PriorError(
                f"a multivariate model takes its priors by response, as brms's resp: Priors(resp={{{', '.join(f'{r!r}: {{...}}' for r in responses)}}}"
                + (", rescor=ab.LKJ(eta)" if spec.rescor else "") + f"); got {top} at the top level (docs/multivariate.md)")
        if not isinstance(self.resp, Mapping):
            raise PriorError(f"a multivariate model needs the priors of each response: pass resp={{{', '.join(f'{r!r}: {{...}}' for r in responses)}}}")
        if set(self.resp) != set(responses):
            raise PriorError(f"resp must cover exactly the responses {responses}; got {list(self.resp)}")
        for r in responses:
            try:
                self.response_priors(r).validate(spec.specs[r], fam)
            except PriorError as e:
                raise PriorError(f"response {r!r}: {e}") from None
        if spec.rescor:
            if self.rescor is None:
                raise PriorError("rescor=True gives the responses a residual correlation matrix, which needs an LKJ prior: pass "
                                 "Priors(..., rescor=ab.LKJ(eta)) (brms class rescor; its default is lkj(1))")
            if not isinstance(self.rescor, Prior) or self.rescor.name not in _CORRELATION:
                raise PriorError(f"prior for 'rescor' must be a correlation prior ({sorted(_CORRELATION)}, brms's lkj(eta)), got {self.rescor}")
        elif self.rescor is not None:
            raise PriorError("this model has no residual correlation (rescor=False): do not pass a rescor prior")

    # -- coefficient priors -----------------------------------------------------------------
    def coefficient_priors(self, spec) -> dict[str, Prior]:
        """One prior per coefficient of brms's class ``b`` (``spec.coefficient_labels``): per design column, keyed by column
        label, and per monotonic term, keyed by its name (``mo(x)``, brms's ``bsp``; B03), in term order. ``spec`` is a
        ``ModelSpec`` (or a tuple of numeric predictor names); categorical levels must be
        known (data seen) for per-level mappings to be checked."""
        from autobayes.formula import ModelSpec

        if not isinstance(spec, ModelSpec):
            spec = ModelSpec(formula="", response="", predictors=tuple(spec), group=None)
        if isinstance(self.b, Prior):
            return {col: self.b for col in spec.coefficient_labels}
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
                mo = next((t for t in spec.mo_terms if k in (t.operands[0], f"mo{t.operands[0]}")), None)
                if mo is not None:                          # the column, or brms's coefficient name 'moord' (B03)
                    hint = f"; the monotonic term's coefficient is keyed by the term, b={{{mo.name!r}: prior}}"
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

    def simo_priors(self, spec) -> dict[str, Prior]:
        """The ``Dirichlet`` prior of each monotonic term's simplex (B03, brms class ``simo``), keyed by the term's column in
        formula order. Coverage is exact; a single monotonic term also accepts the bare prior; refused when the formula has
        none. Once the data are seen (the term's coding known), a vector of concentrations must have one entry per step."""
        terms = tuple(getattr(spec, "mo_terms", ()))
        columns = [t.operands[0] for t in terms]
        if not terms:
            if self.simo is not None:
                raise PriorError("the formula has no monotonic term mo(x); do not pass simo (the prior of a monotonic term's simplex)")
            return {}
        if self.simo is None:
            raise PriorError(f"the monotonic term(s) {[t.name for t in terms]} need a prior on their simplex: pass "
                             f"simo={{{', '.join(f'{c!r}: ab.Dirichlet(...)' for c in columns)}}} (brms class simo; brms's default "
                             "is dirichlet(1), ab.Dirichlet(1))")
        if isinstance(self.simo, Mapping):
            if set(self.simo) != set(columns):
                extra = [k for k in self.simo if k not in columns]
                hint = next((f"; key the prior by the column, simo={{{c!r}: ...}}" for k in extra for t, c in zip(terms, columns)
                             if k in (t.name, f"mo{c}", f"mo{c}1")), "")
                raise PriorError(f"simo must cover exactly the columns of the monotonic terms {columns}; got {list(self.simo)}{hint}")
            by_column = {c: self.simo[c] for c in columns}
        else:
            if len(columns) != 1:
                raise PriorError(f"several monotonic terms {[t.name for t in terms]} need an exact-key simo mapping over {columns}; "
                                 "a bare prior is not broadcast")
            by_column = {columns[0]: self.simo}
        for c, prior in by_column.items():
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for 'simo:{c}' must be a Prior, got {type(prior).__name__}")
            if prior.name not in _SIMPLEX:
                raise PriorError(f"prior for 'simo:{c}' must be a prior on a simplex ({sorted(_SIMPLEX)}, brms's dirichlet()), got {prior}")
            coding = spec.monotonic(c)
            if coding.known:                                                 # one concentration per step, or one for all
                try:
                    dirichlet_concentration(prior, coding.D)
                except PriorError as e:
                    raise PriorError(f"prior for 'simo:{c}' ({coding.term}, {coding.D} steps from {coding.lowest} to "
                                     f"{coding.highest}): {e}") from None
        return by_column

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
        (``intercept``, ``b:x``); such a model has no prior predictive distribution. A mixture's (C05) carry their component's
        brms name, ``mu1:b:x``."""
        if self.components is not None:
            return [f"mu{k}:{label}" for k, c in enumerate(self.components, start=1) for label in c.improper(spec)]
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
        if self.resp is not None or self.rescor is not None:          # C01: the priors of a multivariate model
            raise PriorError("resp and rescor are the priors of a multivariate model, 'mvbind(y, y2) ~ ...' built with "
                             "ab.gaussian(..., rescor=True | False) (docs/multivariate.md); this formula has one response")
        if fam.mixture:                                               # C05: the components' priors and theta's
            return self._validate_mixture(spec)
        if self.components is not None or self.theta is not None:
            raise PriorError("components and theta are the priors of a mixture (ab.mixture, docs/mixtures.md); the "
                             f"{fam.name} family takes neither")
        has_intercept = getattr(spec, "intercept", True)
        if has_intercept and self.intercept is None:
            raise PriorError("the formula has a population intercept; an intercept prior is required")
        if not has_intercept and self.intercept is not None:
            raise PriorError("the formula has no population intercept; omit the intercept prior (or pass None)")
        sd_priors = self.sd_priors(spec)
        self.cor_priors(spec)                                         # exact coverage of the correlated terms (A02)
        self.simo_priors(spec)                                        # and of the monotonic terms' simplexes (B03)
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
        if fam.mixture:                                    # C05: each term's priors with one value per component (the
            return self._mixture_bambi(spec)               # intercepts, sigmas and theta are built by the family)
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
        for term in spec.bambi_terms:                         # a monotonic term's coefficient is built by the model (B03)
            if not term.is_vector:
                out[term.name] = per_column[term.name].to_bambi()
                continue
            priors = [per_column[col] for col in term.columns]
            name = priors[0].name
            params = {k: np.array([float(pr.params[k]) for pr in priors]) for k in priors[0].params}
            out[term.name] = bmb.Prior(name, **params)
        return out

    # -- a mixture (C05, docs/mixtures.md) ---------------------------------------------------------------------------------
    def _validate_mixture(self, spec) -> None:
        """A mixture's priors: nothing at the top level but ``components`` (at least two ``Priors``, each valid for the
        component family: a Gaussian component's intercept, coefficients and ``sigma``, read as a Gaussian model's are) and
        ``theta`` (a ``Dirichlet`` with one concentration per component or one for all). Per class, the components share one
        distribution family: the intercepts are one ordered vector (brms's ``ordered_Intercept``) and each term's coefficients
        one variable with a component axis, so only their parameters differ by component."""
        from autobayes.families import get_family
        from autobayes.families.mixture import COMPONENT_FAMILIES

        form = ("Priors(components=[ab.Priors(intercept=..., b=..., sigma=...), ...], theta=ab.Dirichlet(...)), one Priors per "
                "component (brms's dpar = mu<k> and class sigma<k>; docs/mixtures.md)")
        top = [name for name in ("intercept", "b", "sd", "sigma", "shape", "phi", "cor", "nu", "zi", "hu", "simo")
               if getattr(self, name) is not None]
        if top:
            raise PriorError(f"a mixture's priors are per component, {form}; {top} at the top level do not apply")
        comps = self.components
        if comps is None:
            raise PriorError(f"a mixture needs the priors of each component: {form}")
        if not isinstance(comps, (list, tuple)) or len(comps) < 2 or not all(isinstance(c, Priors) for c in comps):
            raise PriorError(f"components must be a list of at least two Priors, one per component: {form}")
        component = get_family(COMPONENT_FAMILIES[0])
        for k, c in enumerate(comps, start=1):
            if c.components is not None or c.theta is not None:
                raise PriorError(f"component {k}'s priors cannot hold components or theta of their own")
            try:
                c.validate(spec, component)
            except PriorError as e:
                raise PriorError(f"component {k} (brms's mu{k} and sigma{k}): {e}") from None
        intercepts = {c.intercept.name for c in comps} if getattr(spec, "intercept", True) else set()
        if len(intercepts) > 1:
            raise PriorError(f"the components' intercept priors must share one distribution family, each with its own parameters "
                             f"(they are brms's ordered_Intercept, one vector); got {sorted(intercepts)}")
        per = [c.coefficient_priors(spec) for c in comps]
        for col in spec.columns:
            names = {p[col].name for p in per if col in p}             # per-level priors are known once the data are seen
            if len(names) > 1:
                raise PriorError(f"the components' priors of {col!r} must share one distribution family, each with its own "
                                 f"parameters (a term's coefficients are one variable with a component axis); got {sorted(names)}")
        if self.theta is None:
            raise PriorError(f"a mixture needs a prior on its mixing proportions: theta=ab.Dirichlet(...) (brms class theta; its "
                             "default is dirichlet(1), ab.Dirichlet(1))")
        if not isinstance(self.theta, Prior) or self.theta.name not in _SIMPLEX:
            raise PriorError(f"prior for 'theta' must be a prior on a simplex ({sorted(_SIMPLEX)}, brms's dirichlet()), got {self.theta}")
        try:
            dirichlet_concentration(self.theta, len(comps))
        except PriorError as e:
            raise PriorError(f"prior for 'theta' ({len(comps)} components): {e}") from None

    def _mixture_bambi(self, spec) -> dict:
        """Bambi's prior of each population-level term of a mixture: one distribution per term with one value of each
        parameter per (coefficient, component) (``likelihoods.component_prior``)."""
        from autobayes.likelihoods import component_prior

        per = [c.coefficient_priors(spec) for c in self.components]
        out = {}
        for term in spec.bambi_terms:
            grid = [[p[col] for p in per] for col in term.columns]            # (columns of the term, components)
            params = {k: np.array([[float(pr.params[k]) for pr in row] for row in grid]) for k in grid[0][0].params}
            out[term.name] = component_prior(grid[0][0].name, params if term.levels else {k: v[0] for k, v in params.items()})
        return out
