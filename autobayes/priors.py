"""Explicit priors, by parameter class, in brms terms.

brms classes: ``Intercept``, ``b`` (slopes and dummy coefficients), ``sd`` (one group SD per group-level
coefficient: the varying intercept of ``(1 | g)``, the intercept and each slope of ``(1 + x || g)``), ``cor`` (the
LKJ prior on the correlation matrix of the coefficients of ``(1 + x | g)``, brms's ``lkj(eta)``; A02) and the
family's auxiliary parameter: ``sigma`` for Gaussian/Student-t (residual SD/scale respectively) and for the
lognormal (the SD of ``log y``), ``shape`` for the negative binomial and Gamma families, ``phi`` for the Beta family
(the precision), and ``nu`` for a Student-t model that learns its degrees of freedom (``nu=None``; brms class ``nu``,
truncated to ``nu > 1``); ``zi`` or ``hu``, the zero-inflation or hurdle probability (B06, a ``Beta`` prior); ``alpha``,
``beta`` or ``ndt``, the skewness, exponential component's mean or non-decision time of the skew-normal, ex-Gaussian and shifted
lognormal families (D01; ``ndt``'s ``Uniform(0, "min_Y")`` is brms's ``uniform(0, min_Y)``); ``zoi`` and ``coi``, the
zero-one-inflated Beta's two probabilities (E01, ``Beta`` priors); ``kappa``, the extended-support Beta's (``xbeta``, E01) or
the von Mises precision (E02); ``quantile`` or ``xi``, the quantile or shape of the asymmetric Laplace and generalized extreme
value families (E02; ``xi`` is the prior of brms's unscaled ``tmp_xi``). An
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

import math
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
             "zi": "zero-inflation probability", "hu": "hurdle probability",
             # D01: a family's extra parameters (FamilySpec.extra_parameters): skew_normal, exgaussian, shifted_lognormal
             "alpha": "skewness", "beta": "mean of the exponential component", "ndt": "non-decision time",
             # E01: the zero-one-inflated Beta's two probabilities; E01, E02: kappa (xbeta's, von_mises's precision); E02:
             # asym_laplace's quantile, gen_extreme_value's xi (the prior of brms's tmp_xi)
             "zoi": "zero-one-inflation probability", "coi": "conditional one-inflation probability", "kappa": "kappa parameter",
             "quantile": "quantile", "xi": "extreme value shape"}
# The priors of a learned parameter with a lower bound (the Student-t nu, A13): the positive-support priors whose
# truncation at the bound PyMC can renormalise (it needs their CDF; PyMC has none for HalfStudentT).
TRUNCATABLE = frozenset({"Gamma", "Exponential", "HalfNormal", "HalfCauchy"})
_NO_CDF = frozenset({"HalfStudentT"})            # positive-support priors whose CDF PyMC lacks, so no truncation of them
# D01: the priors of a parameter bounded on both sides by brms (the shifted lognormal's real<lower=0,upper=min_Y> ndt), each
# truncated to the interval and renormalised there as brms does, the log of its mass there computed with SciPy
# (``bounded_log_mass``); and ``Uniform(0, "min_Y")``, brms's default, the uniform distribution on the interval itself.
BOUNDABLE = frozenset({"Normal", "StudentT", "HalfNormal", "HalfCauchy", "Exponential", "Gamma", "InverseGamma", "Uniform"})
MIN_Y = "min_Y"                                  # brms's name of the smallest response, the upper bound of ndt
# D01: the priors each support of a family's extra parameter takes (families.spec.EXTRA_SUPPORTS)
EXTRA_PRIORS = {"real": frozenset({"Normal", "StudentT"}), "positive": frozenset(_SCALE), "below_min_y": BOUNDABLE,
                "unit": frozenset(_UNIT)}                       # E01, E02: brms's real<lower=0,upper=1> zoi, coi, quantile


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
        inside = ", ".join(f"{k}={v}" if isinstance(v, str) else f"{k}={v:g}" for k, v in self.params.items())   # D01: min_Y
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

    def to_bambi_bounded(self, lower: float, upper: float):
        """Bambi's prior for a parameter bounded to ``(lower, upper)`` (D01: the shifted lognormal's ``ndt``, brms's
        ``real<lower=0,upper=min_Y>``): this distribution truncated to the interval and renormalised there, as brms's
        ``lprior`` does (the density minus ``log_diff_exp`` of the CDF at the bounds), or for ``Uniform(0, "min_Y")`` the
        uniform density ``-log(upper - lower)``. It is a ``pm.CustomDist`` sampled on PyMC's interval coordinate (Stan's
        lower-upper transform, ``log((x - lower) / (upper - x))``), whose log density is PyMC's own density of the
        distribution minus the log of its mass between the bounds, a float64 constant computed with SciPy
        (``bounded_log_mass``): PyMC's ``Truncated`` would compute it through PyMC's normal log-CDF, whose float32 constant
        ``sqrt(2.0)`` costs 1e-8 (measured on the frozen case ``shifted_lognormal_varint``). Draws are SciPy's inverse CDF
        between the bounds; the support point is the interval's midpoint."""
        import bambi as bmb
        import pymc as pm
        import pytensor.tensor as pt
        from pymc.distributions.transforms import Interval

        lower, upper = np.float64(lower), np.float64(upper)
        if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
            raise PriorError(f"the bounds of a bounded parameter must be finite with lower < upper, got ({lower}, {upper})")
        if self.name == "Uniform":
            log_density = np.float64(-np.log(upper - lower))
            base = log_mass = dist = None
        else:
            log_density = None
            log_mass = np.float64(bounded_log_mass(self, lower, upper))
            params = {k: np.float64(v) for k, v in self.params.items()}
            base = getattr(pm, self.name)
            dist = _scipy_distribution(self)
        middle = np.float64(0.5) * (lower + upper)

        def logp(value):
            inside = pt.and_(pt.ge(value, lower), pt.le(value, upper))
            body = pt.zeros_like(value) + log_density if base is None else pm.logp(base.dist(**params), value) - log_mass
            return pt.switch(inside, body, -np.inf)

        def random(rng=None, size=None):
            if dist is None:
                return rng.uniform(lower, upper, size=size)
            return np.clip(dist.ppf(rng.uniform(dist.cdf(lower), dist.cdf(upper), size=size)), lower, upper)

        def support_point(rv, size):
            return pt.full_like(rv, middle)

        def bounded(name, **_):
            return pm.CustomDist(name, logp=logp, random=random, support_point=support_point, transform=Interval(lower, upper),
                                 dtype="float64")

        return bmb.Prior(f"Bounded{self.name}", dist=bounded)


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


def Uniform(lower: float, upper) -> Prior:
    """brms's ``uniform(lower, upper)``, the prior of the shifted lognormal's non-decision time ``ndt`` (D01,
    ``Priors(ndt=...)``): brms's default ``uniform(0, min_Y)`` is ``Uniform(0, "min_Y")``, the uniform distribution on ndt's
    own support (0, min y), whose upper bound is the smallest response of the data the model is built on (brms's Stan data
    ``min_Y``). Only that form is accepted, and only for ``ndt`` (a narrower interval would put the density at -inf on part
    of the parameter's support, where brms's sampler cannot start; divergence D78)."""
    if not (isinstance(lower, (int, float, np.integer, np.floating)) and float(lower) == 0.0 and isinstance(upper, str)
            and upper == MIN_Y):
        raise PriorError(f'Uniform is brms\'s uniform(0, min_Y), the default prior of ndt: write Uniform(0, "min_Y"); got '
                         f"Uniform({lower!r}, {upper!r}) (another interval is not offered, divergence D78)")
    return Prior("Uniform", {"lower": 0.0, "upper": MIN_Y})


def _scipy_distribution(prior: Prior):
    """The SciPy distribution of a prior of ``BOUNDABLE`` (but ``Uniform``), with Stan's (and PyMC's) parameterisation."""
    from scipy import stats

    p = {k: float(v) for k, v in prior.params.items()}
    if prior.name == "Normal":
        return stats.norm(p["mu"], p["sigma"])
    if prior.name == "StudentT":
        return stats.t(p["nu"], p["mu"], p["sigma"])
    if prior.name == "HalfNormal":
        return stats.halfnorm(scale=p["sigma"])
    if prior.name == "HalfCauchy":
        return stats.halfcauchy(scale=p["beta"])
    if prior.name == "Exponential":
        return stats.expon(scale=1.0 / p["lam"])
    if prior.name == "Gamma":
        return stats.gamma(p["alpha"], scale=1.0 / p["beta"])
    if prior.name == "InverseGamma":
        return stats.invgamma(p["alpha"], scale=p["beta"])
    raise PriorError(f"{prior} cannot be truncated to an interval here; use one of {sorted(BOUNDABLE)}")


def bounded_log_mass(prior: Prior, lower: float, upper: float) -> float:
    """``log P(lower < X < upper)`` under ``prior`` (D01): the log of brms's ``log_diff_exp(lcdf(upper), lcdf(lower))``,
    from the lower tail's CDF, or from the upper tail's survival function where the interval lies above the median, so that
    no difference of two numbers near 1 is formed. Refused when the prior puts no mass on the interval."""
    d = _scipy_distribution(prior)
    if d.cdf(lower) > 0.5:
        hi, lo = float(d.logsf(lower)), float(d.logsf(upper))
    else:
        hi, lo = float(d.logcdf(upper)), float(d.logcdf(lower))
    if not hi > lo:
        raise PriorError(f"{prior} puts no mass on ({lower:g}, {upper:g}), the support of the parameter it is the prior of")
    return hi + (math.log1p(-math.exp(lo - hi)) if np.isfinite(lo) else 0.0)


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
    from autobayes.shrinkage import brms_spelling, is_shrinkage

    if is_shrinkage(prior):                               # D07: horseshoe(...) / R2D2(...) on the whole class b
        return brms_spelling(prior)

    if prior.name == "Dirichlet":                         # its concentrations in order, or brms's one-number shorthand
        return f"dirichlet({', '.join(r_number(float(v)) for v in prior.params.values())})"
    if prior.name == "Uniform":                           # D01: brms's uniform(0, min_Y), ndt's default
        return f"uniform({r_number(float(prior.params['lower']))}, {prior.params['upper']})"
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

    A Gaussian process ``gp(x)`` (C03, docs/gaussian-processes.md) has a marginal SD and a length-scale, each with its own
    positive-support prior: ``sdgp`` (brms class ``sdgp``) and ``lscale`` (brms class ``lscale``, on the scaled input), keyed
    by the process's column, ``sdgp={"x": HalfNormal(1)}, lscale={"x": InverseGamma(3, 1)}`` (brms ``coef = gpx``); a single
    Gaussian process also accepts the bare priors, several need exact-key mappings (nothing is broadcast, as for ``sd``);
    both are required with a Gaussian process and refused without one.

    A non-linear model (C02, ``nl=True``, docs/nonlinear.md) has no ``intercept``, ``b`` or ``sd`` of its own: ``nlpars``
    gives each non-linear parameter's, ``nlpars={"a": {"b": Normal(1, 1), "sd": HalfNormal(1)}, "b": {"b": Normal(0, 1)}}``
    (brms's ``class = b`` and ``class = sd`` with ``nlpar = "a"``; ``"b"`` covers the parameter's intercept too, brms's
    coefficient ``Intercept``, as one prior or as a mapping by term with an ``"Intercept"`` entry), while ``cor`` stays keyed
    by factor (brms's class ``cor`` has no ``nlpar``); refused for every other model.

    ``ar`` and ``ma`` are the priors of an autocorrelation term's coefficients (C04, ``ar()``, ``ma()``, ``arma()``;
    docs/autocorrelation.md): one location prior for every entry of the vector (brms's classes ``ar`` and ``ma``,
    unbounded; flat by default, ``Flat()``), required exactly for the parts the term has and refused for every other model.

    A measurement-error term ``me(x, sdx)`` (C06, docs/missing-and-measurement-error.md) has a coefficient of class ``b``
    (brms's ``bsp``, coefficient ``mexsdx``), covered by ``b`` under the term's name, and its latent values a location and a
    scale, ``meanme`` (a location prior) and ``sdme`` (a positive-support prior, or ``Flat()``, brms's default, flat on the
    positive half line), each keyed by the noisy column (``meanme={"x": Normal(0, 2)}``, brms ``class = meanme, coef = mex``)
    or bare for the model's one ``me()`` term; refused without one. A missing-value term ``mi(col)`` has a coefficient of
    class ``b`` (brms's ``bsp``, coefficient ``micol``) and nothing else: the missing values have brms's flat prior.

    ``alpha``, ``beta`` and ``ndt`` are the extra auxiliary parameters of the skew-normal, ex-Gaussian and shifted lognormal
    families (D01, ``FamilySpec.extra_parameters``, docs/continuous-families.md), required exactly for their family and refused
    for every other: ``alpha``, the skewness, a location prior (``Normal``, ``StudentT``; brms class ``alpha``, default
    ``normal(0, 4)``); ``beta``, the mean of the ex-Gaussian's exponential component, a positive-support prior (brms class
    ``beta``, default ``gamma(1, 0.1)``); ``ndt``, the non-decision time, brms's ``real<lower=0,upper=min_Y>``: one of
    ``BOUNDABLE``, truncated to (0, min y) and renormalised there as brms does, or ``Uniform(0, "min_Y")``, brms's default
    ``uniform(0, min_Y)``. None takes a formula of its own (D79). E01 (docs/bounded-families.md): ``zoi`` and ``coi``, the
    zero-one-inflated Beta's zero-one-inflation probability and conditional one-inflation probability (brms classes ``zoi`` and
    ``coi``, ``real<lower=0,upper=1>``: a ``Beta`` prior, brms's default ``beta(1, 1)``), and ``kappa``, the extended-support
    Beta's kappa (brms class ``kappa`` of ``xbeta()``, ``real<lower=0>``: a positive-support prior, brms's default
    ``gamma(0.01, 0.01)``), each required exactly for its family and refused for every other.
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
    sdgp: Prior | Mapping[str, Prior] | None = None   # each Gaussian process's marginal SD, by column (brms class ``sdgp``; C03)
    lscale: Prior | Mapping[str, Prior] | None = None  # each Gaussian process's length-scale, by column (brms class ``lscale``; C03)
    nlpars: Mapping[str, Mapping[str, object]] | None = None   # each non-linear parameter's {"b": ..., "sd": ...} (brms's nlpar =; C02)
    ar: Prior | None = None          # an autocorrelation term's autoregressive coefficients (brms class ``ar``; C04)
    ma: Prior | None = None          # an autocorrelation term's moving-average coefficients (brms class ``ma``; C04)
    # C06, a measurement-error term me(x, sdx) (docs/missing-and-measurement-error.md): the location and the scale of its latent
    # values, by noisy column (brms classes ``meanme`` and ``sdme``, coefficient ``mex``); a single me() term takes the bare prior
    meanme: Prior | Mapping[str, Prior] | None = None
    sdme: Prior | Mapping[str, Prior] | None = None
    # D01, a family's extra parameters (FamilySpec.extra_parameters, docs/continuous-families.md): the skew normal's skewness
    # (brms class ``alpha``, a location prior), the ex-Gaussian's exponential component's mean (class ``beta``, a positive-support prior)
    # and the shifted lognormal's non-decision time (class ``ndt``, truncated to (0, min y)); refused for every other family
    alpha: Prior | None = None
    beta: Prior | None = None
    ndt: Prior | None = None
    # E01 (docs/bounded-families.md): the zero-one-inflated Beta's zero-one-inflation probability and conditional one-inflation
    # probability (brms classes ``zoi`` and ``coi``, real<lower=0,upper=1>, a Beta prior) and the extended-support Beta's kappa
    # (brms class ``kappa`` of ``xbeta()``, real<lower=0>, a positive-support prior); E02 (docs/continuous-families.md): the von
    # Mises precision (class ``kappa`` too, a positive-support prior), the asymmetric Laplace quantile (class ``quantile``, a
    # prior on (0, 1); none when the model fixes it) and the generalized extreme value shape (class ``xi``, a location prior on
    # brms's unscaled tmp_xi); each refused for every other family
    zoi: Prior | None = None
    coi: Prior | None = None
    kappa: Prior | None = None
    quantile: Prior | None = None
    xi: Prior | None = None

    def __post_init__(self):
        if isinstance(self.components, list):                         # one value for equal component lists (C05)
            object.__setattr__(self, "components", tuple(self.components))

    # -- multivariate models (C01, docs/multivariate.md) ---------------------------------------------
    _RESPONSE_CLASSES = ("intercept", "b", "sd", "sigma", "shape", "phi", "nu", "zi", "hu", "simo", "alpha", "beta", "ndt",
                         "kappa", "quantile", "xi")

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
        from autobayes.shrinkage import check_prior

        check_prior(self)                                             # D07: horseshoe() / R2D2() in a response's priors
        if self.nlpars is not None:                                   # C02
            raise PriorError("nlpars gives the priors of non-linear parameters (ab.gaussian(..., nl=True), docs/nonlinear.md); "
                             "a multivariate model takes none: a non-linear model in a multivariate formula is not supported "
                             "(divergence D62)")
        if self.components is not None or self.theta is not None:
            raise PriorError("components and theta are the priors of a mixture (ab.mixture, docs/mixtures.md); "
                             "a multivariate model takes neither")
        if self.ar is not None or self.ma is not None:              # C04
            raise PriorError("ar and ma are the priors of an autocorrelation term ar(), ma(), arma() (docs/autocorrelation.md); a "
                             "multivariate model takes neither: an autocorrelation term in a multivariate formula is not supported "
                             "(divergence D70)")
        if self.sdgp is not None or self.lscale is not None:          # C03: a multivariate formula takes no gp()
            raise PriorError("sdgp and lscale are the priors of a Gaussian process gp(x) (docs/gaussian-processes.md); a multivariate "
                             "model takes neither: a Gaussian process in a multivariate formula is not supported (divergence D56)")
        if self.meanme is not None or self.sdme is not None:          # C06
            raise PriorError("meanme and sdme are the priors of a measurement-error term me(x, sdx) (docs/missing-and-measurement-error.md); "
                             "a multivariate model takes neither: a measurement-error term in a multivariate model is not supported "
                             "(divergence D73)")
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
        # D08: a correlation ID shared with a distributional formula has one correlation matrix too, keyed by its factor
        terms += [t for t in group_terms if t.id is not None and any(s.factor == t.factor for s in getattr(spec, "shared_ids", ()))]
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

    def sdgp_priors(self, spec) -> dict[str, Prior]:
        """The prior of each Gaussian process's marginal SD (C03, brms class ``sdgp``), keyed by its column in formula order."""
        return self._gp_priors(spec, "sdgp", "marginal SD", "student_t(3, 0, scale) as for sd")

    def lscale_priors(self, spec) -> dict[str, Prior]:
        """The prior of each Gaussian process's length-scale (C03, brms class ``lscale``), keyed by its column in formula order."""
        return self._gp_priors(spec, "lscale", "length-scale", "an inverse gamma tuned to the inputs")

    def _gp_priors(self, spec, cls: str, what: str, default: str) -> dict[str, Prior]:
        """``sdgp_priors`` / ``lscale_priors``: exact coverage of the Gaussian processes' columns (a single one also takes the
        bare prior), positive-support priors only; refused when the formula has no Gaussian process."""
        terms = tuple(getattr(spec, "gp_terms", ()))
        columns = [g.column for g in terms]
        given = getattr(self, cls)
        if not terms:
            if given is not None:
                raise PriorError(f"the formula has no Gaussian process gp(x); do not pass {cls} (the prior of a Gaussian process's {what})")
            return {}
        if given is None:
            raise PriorError(f"the Gaussian process(es) {[g.term for g in terms]} need a prior on their {what}: pass "
                             f"{cls}={{{', '.join(f'{c!r}: ...' for c in columns)}}} (brms class {cls}; brms's default is {default}, "
                             "which ab.default_priors() returns)")
        if isinstance(given, Mapping):
            if set(given) != set(columns):
                extra = [k for k in given if k not in columns]
                hint = next((f"; key the prior by the column, {cls}={{{c!r}: ...}}" for k in extra for g, c in zip(terms, columns)
                             if k in (g.term, f"gp({c})", f"gp{c}")), "")
                raise PriorError(f"{cls} must cover exactly the columns of the Gaussian processes {columns}; got {list(given)}{hint}")
            by_column = {c: given[c] for c in columns}
        else:
            if len(columns) != 1:
                raise PriorError(f"several Gaussian processes {[g.term for g in terms]} need an exact-key {cls} mapping over "
                                 f"{columns}; a bare prior is not broadcast")
            by_column = {columns[0]: given}
        for c, prior in by_column.items():
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for '{cls}:{c}' must be a Prior, got {type(prior).__name__}")
            if prior.name not in _SCALE:
                raise PriorError(f"prior for '{cls}:{c}' must be a positive-support prior ({sorted(_SCALE)}; brms's {cls} is "
                                 f"bounded below by 0), got {prior}")
        return by_column

    def autocor_priors(self, spec) -> dict[str, Prior]:
        """The priors of an autocorrelation term's coefficients (C04, ``spec.autocor``): ``{"ar": ..., "ma": ...}`` for the
        parts the term has (``p > 0``, ``q > 0``), each one prior for every entry of the vector, as brms's class ``ar`` /
        ``ma`` is. Coverage is exact: a part the formula has needs its prior, and a prior for a part it lacks is refused."""
        term = getattr(spec, "autocor", None)
        wanted = dict(term.parts) if term is not None else {}
        out: dict[str, Prior] = {}
        for name, meaning in (("ar", "autoregressive"), ("ma", "moving-average")):
            prior = getattr(self, name)
            if name not in wanted:
                if prior is not None:
                    has = (f"; '{term.term}' has no {meaning} part" if term is not None else
                           "; the formula has no autocorrelation term ar(), ma() or arma()")
                    raise PriorError(f"do not pass {name}{has}")
                continue
            if prior is None:
                raise PriorError(f"the autocorrelation term '{term.term}' has {wanted[name]} {meaning} coefficient(s), which need a "
                                 f"prior: pass {name}=... (brms class {name}, unbounded; brms's default is flat, ab.Flat())")
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for {name!r} must be a Prior, got {type(prior).__name__}")
            out[name] = prior
        return out

    def me_priors(self, spec, cls: str) -> dict[str, Prior]:
        """The prior of class ``cls`` (``"meanme"``, the location of a measurement-error term's latent values, or ``"sdme"``,
        their scale; C06) of each ``me(x, sdx)`` term, keyed by its noisy column ``x`` in formula order: a location prior
        (``meanme``), a positive-support prior or ``Flat()`` (``sdme``; brms's default is flat on the positive half line).
        Coverage is exact; a single ``me()`` term also takes the bare prior; refused when the formula has none."""
        terms = tuple(getattr(spec, "me_terms", ()))
        columns = [t.operands[0] for t in terms]
        given = getattr(self, cls)
        what = "location" if cls == "meanme" else "scale"
        if not terms:
            if given is not None:
                raise PriorError(f"the formula has no measurement-error term me(x, sdx); do not pass {cls} (the prior of the "
                                 f"{what} of its latent values)")
            return {}
        if given is None:
            raise PriorError(f"the measurement-error term(s) {[t.name for t in terms]} need a prior on the {what} of their latent "
                             f"values: pass {cls}={{{', '.join(f'{c!r}: ...' for c in columns)}}} (brms class {cls}, whose default "
                             "is flat: ab.default_priors() returns it, ab.Flat())")
        if isinstance(given, Mapping):
            if set(given) != set(columns):
                raise PriorError(f"{cls} must cover exactly the noisy columns of the measurement-error terms {columns}; got {list(given)}")
            by_column = {c: given[c] for c in columns}
        else:
            if len(columns) != 1:
                raise PriorError(f"several measurement-error terms need an exact-key {cls} mapping over {columns}")
            by_column = {columns[0]: given}
        allowed = _LOCATION if cls == "meanme" else _SCALE | {"Flat"}
        for c, prior in by_column.items():
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for '{cls}:{c}' must be a Prior, got {type(prior).__name__}")
            if prior.name not in allowed:
                kind = "a location prior" if cls == "meanme" else "a positive-support prior or Flat() (flat on the positive half line)"
                raise PriorError(f"prior for '{cls}:{c}' must be {kind} ({sorted(allowed)}), got {prior}")
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
        keys = ("intercept", "b", "sd") if dspec.group_terms else ("intercept", "b")   # D08: a shared ID's terms have an sd
        extra = [k for k in entry if k not in keys]
        if extra:
            keys_text = "'intercept', 'b' and 'sd'" if dspec.group_terms else "'intercept' and 'b'"
            raise PriorError(f"the priors of {name}'s formula '{dspec.formula}' take the keys {keys_text} ({brms}"
                             f"{', and class sd with dpar = ' + name if dspec.group_terms else ''}); unexpected {extra}")
        if dspec.group_terms and entry.get("sd") is None:
            raise PriorError(f"'{dspec.formula}' has the group-level term(s) {[t.canonical for t in dspec.group_terms]}, whose SDs need "
                             f"priors: pass {name}={{..., 'sd': ...}} (brms class sd with dpar = {name}; docs/multi-membership-and-by.md)")
        if dspec.intercept and entry.get("intercept") is None:
            raise PriorError(f"'{dspec.formula}' has an intercept, which needs a prior: pass {name}={{'intercept': ..., ...}}")
        if not dspec.intercept and entry.get("intercept") is not None:
            raise PriorError(f"'{dspec.formula}' has no intercept; omit 'intercept' from the priors of {name}'s formula")
        if dspec.pop_terms and entry.get("b") is None:
            raise PriorError(f"'{dspec.formula}' has the population-level term(s) {list(dspec.terms)}, which need priors: pass "
                             f"{name}={{..., 'b': ...}} (one prior, or a mapping by term as for the response formula's b)")
        return Priors(intercept=entry.get("intercept"), b=entry.get("b"), sd=entry.get("sd") if dspec.group_terms else None)

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
        brms name, ``mu1:b:x``; a non-linear model's (C02) their parameter's, ``a:b:Intercept``."""
        if getattr(spec, "is_nonlinear", False):                       # a non-linear model (C02): a:b:Intercept, a:b:x
            from autobayes.nonlinear import nonlinear_improper

            return nonlinear_improper(self, spec)
        if self.components is not None:
            return [f"mu{k}:{label}" for k, c in enumerate(self.components, start=1) for label in c.improper(spec)]
        out = ["intercept"] if getattr(spec, "intercept", True) and isinstance(self.intercept, Prior) and self.intercept.name == "Flat" else []
        out += [f"b:{col}" for col, p in self.coefficient_priors(spec).items() if p.name == "Flat"]
        out += [name for name, p in self.autocor_priors(spec).items() if p.name == "Flat"]   # an ARMA term's ar / ma (C04)
        for name in getattr(spec, "dpar_names", ()):                  # a distributional parameter's formula (B01): sigma:b:zq
            out += [f"{name}:{label}" for label in self.dpar_priors(spec, name).improper(spec.dpar(name))]
        for cls in ("meanme", "sdme"):                                 # a measurement-error term's latent values (C06)
            out += [f"{cls}:{c}" for c, p in self.me_priors(spec, cls).items() if p.name == "Flat"]
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
        from autobayes.shrinkage import check_model, check_prior, is_shrinkage

        check_prior(self)                                             # D07: a shrinkage prior anywhere but the bare b is refused
        if getattr(spec, "is_nonlinear", False):                       # a non-linear model (C02): priors per non-linear parameter
            from autobayes.nonlinear import check_nonlinear_priors

            return check_nonlinear_priors(self, spec, fam)
        if self.nlpars is not None:                                   # C02, refused for every other model, a mixture's included
            raise PriorError("nlpars gives the priors of non-linear parameters, which a model has only with nl=True "
                             "(docs/nonlinear.md); this formula is linear")
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
        self.sdgp_priors(spec)                                        # and of the Gaussian processes' SDs and length-scales (C03)
        self.lscale_priors(spec)
        self.me_priors(spec, "meanme")                                # and of the measurement-error terms' latent values (C06)
        self.me_priors(spec, "sdme")
        scale = fam.scale_parameter                                   # "sigma", "shape", "phi" or None
        zero = fam.zero_parameter                                     # "zi", "hu" or None (B06)
        extras = {e.name: e for e in fam.extra_parameters}            # alpha, beta or ndt (D01)
        auxiliary = {name: getattr(self, name) for name in AUXILIARY}
        predicted = scale is not None and scale in getattr(spec, "dpar_names", ())   # the scale parameter has its own formula (B01)
        if scale is not None and auxiliary[scale] is None and not predicted:
            raise PriorError(fam.scale_missing_error)
        if zero is not None and auxiliary[zero] is None:
            raise PriorError(f"the {fam.name} family needs a prior on the {AUXILIARY[zero]} {zero}: pass {zero}=..., "
                             "a prior on (0, 1) such as ab.Beta(1, 1)")
        for name, extra in extras.items():                            # D01: each is required, with a prior its support takes
            self._validate_extra(fam, extra, auxiliary[name])
        for name, prior in auxiliary.items():
            if prior is not None and name in fam.fixed_parameters:   # E02: a quantile the model fixes (bf(quantile = p))
                raise PriorError(f"the {fam.name} family's {name} is fixed in this model ({name}=...), not learned: do not pass "
                                 f"a prior on it (brms's bf({name} = ...) has none); leave {name}=None to learn it under its prior")
            if prior is not None and name not in (scale, zero, *extras):
                raise PriorError(
                    f"the {fam.name} family has no {AUXILIARY[name]}; do not pass {name}."
                    + (f" Its auxiliary parameter is {scale}: pass {scale}=..." if scale is not None else "")
                    + (f" Its {AUXILIARY[zero]} is {zero}: pass {zero}=..." if zero is not None else "")
                    + "".join(f" Its {e.meaning} is {e.name}: pass {e.name}=..." for e in extras.values())
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
        shrinkage = is_shrinkage(self.b)                              # D07: horseshoe() / R2D2() on the whole class b
        if shrinkage:
            check_model(self.b, spec, fam)
        location = ([("intercept", self.intercept)] if has_intercept else [])
        location += [] if shrinkage else list(self.coefficient_priors(spec).items())
        location += list(self.autocor_priors(spec).items())          # an ARMA term's ar and ma, unbounded (C04)
        dpar_sds = []
        if predicted:                                                 # its formula's intercept and coefficients (B01): location priors
            dspec, sub = spec.dpar(scale), self.dpar_priors(spec, scale)
            location += (([(f"{scale}:intercept", sub.intercept)] if dspec.intercept else [])
                         + [(f"{scale}:b:{col}", p) for col, p in sub.coefficient_priors(dspec).items()])
            dpar_sds = [(f"{scale}:sd:{g}", p) for (g, c), p in sub.sd_priors(dspec).items()]   # D08: a shared ID's terms
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
        scales += dpar_sds
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

    @staticmethod
    def _validate_extra(fam, extra, prior) -> None:
        """D01: the prior of one of the family's extra parameters (``FamilySpec.extra_parameters``): given, one ``Prior``, and
        one its support takes (``EXTRA_PRIORS``)."""
        if prior is None:
            example = {"real": "ab.Normal(0, 4)", "positive": "ab.Gamma(1, 0.1)", "below_min_y": 'ab.Uniform(0, "min_Y")',
                       "unit": "ab.Beta(1, 1)"}
            raise PriorError(f"the {fam.name} family needs a prior on its {extra.meaning} {extra.name}: pass {extra.name}=..., "
                             f"such as {example[extra.support]} (brms's default is {brms_prior(extra.default_prior())})")
        if not isinstance(prior, Prior):
            raise PriorError(f"prior for {extra.name!r} must be a Prior, got {type(prior).__name__} (the {extra.name} of the "
                             f"{fam.name} family takes no formula of its own, divergence {extra.divergence})")
        allowed = EXTRA_PRIORS[extra.support]
        if prior.name not in allowed:
            what = {"real": "a location prior (brms's real " + extra.name + ")",
                    "positive": "a positive-support prior (brms's real<lower=0> " + extra.name + ")",
                    "below_min_y": "a prior truncated to (0, min(y)) and renormalised there (brms's real<lower=0,upper=min_Y> "
                                   + extra.name + "; HalfStudentT(nu, s) is StudentT(nu, 0, s) there)",
                    "unit": "a prior on (0, 1) (brms's real<lower=0,upper=1> " + extra.name + ")"}[extra.support]
            if extra.derived:                                     # E02: brms's prior of class xi is on its unbounded tmp_xi
                what = f"a location prior (brms puts the prior of class {extra.name} on its real {extra.variable})"
            raise PriorError(f"prior for {extra.name!r} must be {what}, one of {sorted(allowed)}; got {prior}")

    def to_bambi(self, spec, family="gaussian") -> dict:
        """Bambi's prior dict: one prior per term (array-valued parameters for the levels of a
        categorical term). Each group-level coefficient (``1|g``, and ``x|g`` for a varying slope) gets
        Normal(0, sd) with its own ``sd`` as hyperprior; without a group term no group prior is emitted
        (Bambi would accept a stray ``1|g`` entry silently). ``family`` as for ``validate``. An auxiliary parameter with its
        own formula (B01) gets the dict of its formula's term priors, which Bambi reads for that distributional component.
        A family's extra parameters (D01, ``alpha``, ``beta``, ``ndt``) are not in it: the model adds them
        (``ExtraParameter.bambi_prior``), because ``ndt``'s bound is the data's smallest response."""
        import bambi as bmb
        from autobayes.families import FamilySpec, get_family
        from autobayes.formula import ModelSpec, bambi_group_name
        from autobayes.shrinkage import is_shrinkage

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
                                              **sub._bambi_terms(dspec),
                                              # D08: a shared correlation ID's terms, Normal(0, sd) with their own sd prior
                                              **{bambi_group_name(g, c): bmb.Prior("Normal", mu=0.0, sigma=p.to_bambi())
                                                 for (g, c), p in sub.sd_priors(dspec).items()}}
        elif fam.scale_parameter is not None:              # under Bambi's name: "sigma", or "alpha" for brms's "shape"
            out[fam.bambi_scale_parameter] = getattr(self, fam.scale_parameter).to_bambi()
        if fam.zero_parameter is not None:                  # the zero-inflation / hurdle probability, under brms's name (B06)
            out[fam.zero_parameter] = getattr(self, fam.zero_parameter).to_bambi()
        if self.nu is not None:                             # a learned Student-t nu (A13): its prior truncated at the bound
            out["nu"] = self.nu.to_bambi_truncated(fam.learnable_parameters["nu"])
        if is_shrinkage(self.b):                            # D07: each term's coefficients are its slice of b = zb * sdb
            from autobayes.shrinkage import bambi_priors

            out.update(bambi_priors(self.b, spec, fam))
        else:
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
        top = [name for name in ("intercept", "b", "sd", "sigma", "shape", "phi", "cor", "nu", "zi", "hu", "simo", "meanme", "sdme",
                                 "alpha", "beta", "ndt", "zoi", "coi", "kappa", "quantile", "xi")
               if getattr(self, name) is not None]
        if top:
            raise PriorError(f"a mixture's priors are per component, {form}; {top} at the top level do not apply")
        if self.ar is not None or self.ma is not None:              # C04: as brms, a mixture takes no autocorrelation term
            raise PriorError("ar and ma are the priors of an autocorrelation term ar(), ma(), arma() (docs/autocorrelation.md); a "
                             "mixture takes neither, as in brms")
        if self.sdgp is not None or self.lscale is not None:          # C03: gp() in a mixture is refused (D55)
            raise PriorError("sdgp and lscale are the priors of a Gaussian process gp() (docs/gaussian-processes.md); a mixture takes "
                             "no Gaussian process in this slice (divergence D55)")
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
