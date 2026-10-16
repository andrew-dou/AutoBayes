"""What one response family contributes to the library: a ``FamilySpec``.

Everything the library once decided by comparing a family name against a string is a field of
the spec. A family is one module in this package that builds its spec (``gaussian.py``,
``student.py``, ``poisson.py``, ``negbinomial.py``, ``gamma.py``, ``lognormal.py``, ``beta.py``, ``bernoulli.py``,
``binomial.py``, ``zero_inflated_poisson.py``, ``hurdle_poisson.py``, ``cumulative.py``, ``categorical.py``, ``mixture.py``,
``weibull.py``, ``exponential.py``, ``skew_normal.py``, ``shifted_lognormal.py``, ``exgaussian.py``, ``beta_binomial.py``,
``geometric.py``, ``negbinomial2.py``);
``autobayes.families.FAMILIES`` is the registry and ``get_family(name)`` the lookup. Callable fields
take the built ``Model`` (or the frame being validated) because that is what the call sites have in
hand; none of them is called before ``Model.__init__`` has resolved the spec.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

NO_FIXED_PARAMETERS: Mapping[str, Callable[[object], float]] = MappingProxyType({})
NO_OTHER_LINKS: Mapping[str, Callable[[], "FamilySpec"]] = MappingProxyType({})
NO_PARKED_LINKS: Mapping[str, str] = MappingProxyType({})
NO_LEARNABLE_PARAMETERS: Mapping[str, float] = MappingProxyType({})
NO_LEARNED_DEFAULTS: Mapping[str, Callable[[], object]] = MappingProxyType({})
STANDARD_PREDICTION_KINDS = frozenset({"mean", "response", "linear"})


def exp_mean(eta, what: str):
    """``exp(eta)`` as a log-link family's expected response, refused rather than clipped when it overflows."""
    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.exp(eta)
    if not np.isfinite(expected).all():
        raise ValueError(f"{what} overflow/nonfinite prediction; inspect the log-mean draws")
    return expected


# -- distributional parameters (B01, docs/distributional.md) ------------------------------------------------------------
def _exp_parameter(eta, name: str):
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.exp(eta)
    if not (np.isfinite(value).all() and (value > 0).all()):
        raise ValueError(f"{name} = exp(linear predictor) overflows or underflows to 0 for some draws and rows; inspect the "
                         f"draws of {name}'s formula")
    return value


# The inverse link of a distributional parameter's formula (NumPy): (its linear predictor, its name) -> the parameter, one
# value per draw and row, refused rather than clipped where it is not a positive finite number.
DISTRIBUTIONAL_INVERSE_LINKS: Mapping[str, Callable] = MappingProxyType({"log": _exp_parameter})


def auxiliary(model, post, parameter: str):
    """The posterior draws of the auxiliary parameter ``parameter`` (``sigma``, ``shape``, ``phi``, by brms name) as an
    array that broadcasts against the linear predictor's ``(chain, draw, row)``: a parameter without a formula has
    ``(chain, draw)`` draws, which gain a row axis; a distributional parameter's (B01) are one per row already
    (``predict()`` puts them into the posterior under the parameter's PyMC name, as Bambi does its response parameters)."""
    values = post[model.names[parameter]].to_numpy()
    return values if values.ndim == 3 else values[..., None]


def is_distributional(model, parameter: str | None) -> bool:
    """Whether ``parameter`` has its own formula in the model (B01)."""
    return parameter is not None and parameter in model.spec.dpar_names


def scale_symbol(model) -> str:
    """How ``describe()`` writes the family's auxiliary parameter in its likelihood lines: ``sigma``, or ``sigma_i`` when
    it has its own formula (B01), one value per row."""
    name = model.family_spec.scale_parameter
    return f"{name}_i" if is_distributional(model, name) else name


def scale_line(model) -> str:
    """``describe()``'s prior line of the family's auxiliary parameter (lower bound 0); for a distributional one (B01), a
    pointer to the paragraph that writes its formula out instead."""
    from autobayes.priors import describe_prior

    name = model.family_spec.scale_parameter
    if is_distributional(model, name):
        return (f"  {name}_i = exp(eta_{name},i), one per row: the formula '{model.spec.dpar(name).formula}' "
                f"(Distributional parameter {name}, above)")
    return f"  {name} ~ {describe_prior(getattr(model.priors, name), lb=0)}"


# -- a family's auxiliary parameters beside its scale parameter (D01, docs/continuous-families.md) ----------------------
# How brms bounds such a parameter decides how it is sampled and which priors it takes (``priors.EXTRA_PRIORS``):
#   "real"         unbounded, brms's ``real alpha`` (the skew-normal skewness): a location prior, sampled as it is;
#   "positive"     ``real<lower=0>`` (the ex-Gaussian ``beta``): a positive-support prior, sampled as its log;
#   "below_min_y"  ``real<lower=0,upper=min_Y>`` with brms's ``min_Y = min(Y)`` (the shifted lognormal's ``ndt``): a prior
#                  truncated to (0, min y) and renormalised there, as brms does, sampled on Stan's lower-upper coordinate
#                  ``log(ndt / (min_Y - ndt))``, PyMC's interval transform.
EXTRA_SUPPORTS = ("real", "positive", "below_min_y")


@dataclass(frozen=True)
class ExtraParameter:
    """An auxiliary parameter a family has beside its ``scale_parameter`` (D01): brms's ``alpha`` (skew_normal), ``beta``
    (exgaussian) and ``ndt`` (shifted_lognormal). Its brms name is its ``Priors`` field, its ``Model.value_point`` keyword,
    its ``model.names`` key and its PyMC variable's name; it takes one prior (no distributional formula, D79)."""

    name: str                                    # brms's name of the parameter and its prior class
    meaning: str                                 # what it is, for describe() and the messages: "skewness", ...
    support: str                                 # one of EXTRA_SUPPORTS
    default_prior: Callable[[], object]          # brms's default prior (A07): normal(0, 4), gamma(1, 0.1), uniform(0, min_Y)

    def __post_init__(self):
        if self.support not in EXTRA_SUPPORTS:
            raise ValueError(f"support must be one of {EXTRA_SUPPORTS}, got {self.support!r}")

    @property
    def coordinate(self) -> str:
        """The PyMC value variable it is sampled as: ``alpha``, ``beta_log__``, ``ndt_interval__``."""
        return {"real": self.name, "positive": f"{self.name}_log__", "below_min_y": f"{self.name}_interval__"}[self.support]

    @property
    def bounds(self) -> tuple[str, str]:
        """brms's ``lb`` and ``ub`` of its class in the prior table."""
        return {"real": ("", ""), "positive": ("0", ""), "below_min_y": ("0", "min_Y")}[self.support]

    @property
    def coordinate_meaning(self) -> str:
        """``describe()``'s words for its value variable."""
        return {"real": f"{self.name}, the {self.meaning} (unconstrained)",
                "positive": f"log({self.name}), the {self.meaning}",
                "below_min_y": f"log({self.name} / (min(y) - {self.name})), the {self.meaning} on Stan's lower-upper coordinate"
                }[self.support]

    def upper(self, response) -> float | None:
        """The upper bound the data give it: ``min(y)`` over the fitted rows for ``"below_min_y"`` (brms's ``min_Y``), else
        None."""
        return float(np.min(np.asarray(response, dtype=float))) if self.support == "below_min_y" else None

    def bambi_prior(self, prior, response):
        """Bambi's prior of the parameter: the prior with float64 parameters (PyTensor would make a Python 4.0 a float32
        constant), or for ``"below_min_y"`` the prior truncated to (0, min y) and renormalised (``Prior.to_bambi_bounded``)."""
        if self.support == "below_min_y":
            return prior.to_bambi_bounded(0.0, self.upper(response))
        import bambi as bmb

        return bmb.Prior(prior.name, **{k: np.float64(v) for k, v in prior.params.items()})

    def value(self, value, response, family: str) -> np.ndarray:
        """The value of the parameter on its PyMC coordinate (``Model.value_point``), refused outside its support."""
        arr = np.asarray(value, dtype=float)
        if arr.shape != () or not np.isfinite(arr):
            raise ValueError(f"{self.name} (the {self.meaning} of the {family} family) must be a finite scalar")
        if self.support == "real":
            return np.asarray(arr)
        if self.support == "positive":
            if not arr > 0:
                raise ValueError(f"{self.name} (the {self.meaning} of the {family} family) must be positive")
            return np.asarray(np.log(arr))
        top = self.upper(response)
        if not 0 < arr < top:
            raise ValueError(f"{self.name} (the {self.meaning} of the {family} family) must lie strictly between 0 and min(y) = "
                             f"{top!r}")
        return np.asarray(np.log(arr) - np.log(top - arr))

    def prior_line(self, model) -> str:
        """``describe()``'s prior line of the parameter, with its bounds."""
        from autobayes.priors import describe_prior

        prior = getattr(model.priors, self.name)
        if self.support == "real":
            return f"  {self.name} ~ {describe_prior(prior)}   ({self.meaning})"
        if self.support == "positive":
            return f"  {self.name} ~ {describe_prior(prior, lb=0)}   ({self.meaning}, > 0)"
        top = self.upper(model.design.y)
        if prior.name == "Uniform":                       # brms's uniform(0, min_Y): the parameter's own support
            return f"  {self.name} ~ {describe_prior(prior)}, uniform on (0, min(y) = {top:g})   ({self.meaning})"
        return (f"  {self.name} ~ {describe_prior(prior)}, truncated to (0, min(y) = {top:g}) and renormalised there, as brms's "
                f"real<lower=0,upper=min_Y> {self.name}   ({self.meaning})")


NO_EXTRA_PARAMETERS: tuple[ExtraParameter, ...] = ()


def extra_draws(model, post, name: str):
    """The posterior draws of the family's extra parameter ``name`` (``alpha``, ``beta``, ``ndt``) with a row axis, to
    broadcast against the linear predictor's ``(chain, draw, row)``."""
    return post[model.names[name]].to_numpy()[..., None]


@dataclass(frozen=True, eq=False)
class FamilySpec:
    """One response family. Fields in the order the model consults them."""

    name: str                                    # the registry key and ``model.family``
    tag: str                                     # the ``describe()`` heading: "M0", "M1", "M2", "Student-t", "Poisson"
    parent: str                                  # Bambi's parent parameter, the linear predictor's name: "mu", "logit_p", "probit_p"
    link: str                                    # "identity", "logit", "log" or "probit"
    link_scale: str | None                       # the scale the location priors live on when it is not the response ("logit", "log", "probit")
    mean_name: str                               # the name ``predict()`` gives the expected response: "mu" or "p"
    response_kind: str                           # "continuous", "binary", "count" or "binomial" (successes with trials)
    response_dtype: object | None                # the response column and its draws are cast to this (int64 for the discrete families)
    scale_parameter: str | None                  # the auxiliary parameter with its own prior, by its brms name ("sigma", "shape"), or None
    bambi_scale_parameter: str | None            # Bambi's (and so PyMC's) name for it: "sigma", or "alpha" for brms's "shape"
    fixed_parameters: Mapping[str, Callable[[object], float]]   # user-supplied fixed scalars by name, each with its check ("nu")
    bambi_family: Callable[[], object]           # the Bambi family object or name handed to ``bmb.Model``
    validate_response: Callable[[object, object], None]         # (ModelSpec, DataFrame): refuse a response outside the family's support
    response_warnings: Callable[[object, object], list[str]]    # (ModelSpec, fitted frame): data warnings the model records
    response_upper_bound: Callable[[object], object]            # (model): upper bound of prior-predictive counts, or None
    inverse_link: Callable[[object], object]     # linear predictor -> the parent parameter on its natural scale (the expected response unless ``expected_response`` is set), with the family's own checks
    expected_response: Callable[[object, object, object], object] | None   # (model, posterior, eta) -> E[y] when it also needs an auxiliary parameter (lognormal: exp(mu + sigma^2 / 2)); None: inverse_link(eta)
    draw_response: Callable[[object, object, object, object, object, object], object]   # (model, posterior, eta, mean, frame, rng) -> new observations
    manual_prediction: bool                      # ``predict()`` always assembles this family's predictions by hand
    prediction_kinds: frozenset[str]             # the ``kind`` values ``predict()`` accepts
    kind_error: str                              # the message for any other ``kind``
    scale_rule: str                              # ``value_point``'s message when ``sigma`` or ``shape`` is wrongly given or omitted
    scale_missing_error: str | None              # ``Priors.validate``'s message when the auxiliary prior is missing (families with one)
    linpred_symbol: str                          # "mu" or "eta" in the written-out linear predictor
    scale_phrase: str                            # "response", "log-odds" or "log mean": the scale coefficients act on
    intercept_meaning: str                       # what the centred intercept is ("expected response", "log-odds of y = 1", ...)
    mean_lines: Callable[[object], list[str]]    # (model): ``describe()`` lines relating the mean to the linear predictor
    likelihood_lines: Callable[[object], list[str]]   # (model): the scale prior, fixed parameters and likelihood paragraph
    centred_intercept_note: tuple[str, ...]      # ``describe()`` lines after a centred intercept with categorical predictors
    deterministic_note: str                      # the tail of ``describe()``'s "Deterministic:" line
    # brms's default priors (A07, ``ab.default_priors``, docs/default-priors.md):
    default_prior_link: str | None               # how brms reads the response for its data-dependent defaults (def_scale_prior): "identity", "log" (zeros read as 0.1; also the lognormal's log scale), or None for a link it does not read the response through (logit, probit), whose defaults are student_t(3, 0, 2.5)
    default_scale_prior: Callable[[float], object] | None   # (brms's data-dependent scale) -> brms's default prior of the auxiliary parameter (sigma: HalfStudentT(3, scale); shape, phi: the family's constant); None without one
    links: Mapping[str, Callable[[], "FamilySpec"]] = field(default_factory=lambda: NO_OTHER_LINKS)   # the family's other links by name, each building this family's spec under that link (Bernoulli, binomial: probit); empty for a one-link family
    parked_links: Mapping[str, str] = field(default_factory=lambda: NO_PARKED_LINKS)   # links brms offers for this family that this library withholds, each with the reason ``with_link`` reports (none since P6 lifted the cloglog link's; docs/links.md)
    learnable_parameters: Mapping[str, float] = field(default_factory=lambda: NO_LEARNABLE_PARAMETERS)   # fixed parameters a model learns instead when none is given, each under its own prior truncated at the lower bound given here (Student-t: nu > 1, brms's real<lower=1> nu; docs/student-learned-nu.md)
    default_learned_priors: Mapping[str, Callable[[], object]] = field(default_factory=lambda: NO_LEARNED_DEFAULTS)   # brms's default prior of each learnable parameter (Student-t nu: gamma(2, 0.1), Gamma(2, 0.1)); A07
    # The probability of the zero part (B06, docs/zero-inflated-hurdle.md): brms's "zi" (zero-inflation: a structural
    # zero with probability zi, the count part otherwise) or "hu" (a hurdle: zero with probability hu, the zero-truncated
    # count part otherwise), real<lower=0,upper=1> with its own prior on (0, 1) (``Priors(zi=...)``, ``Priors(hu=...)``),
    # sampled as logit(zi) (PyMC's ``zi_logodds__``); the PyMC variable keeps brms's name. It sits beside
    # ``scale_parameter``, which the zero-inflated or hurdle form of a count family with its own auxiliary parameter (the
    # negative binomial's shape) would keep. None for every other family.
    zero_parameter: str | None = None
    default_zero_prior: Callable[[], object] | None = None   # brms's default prior of the zero_parameter: beta(1, 1), ``Beta(1, 1)`` (A07)
    # The family's auxiliary parameters beside scale_parameter and zero_parameter (D01, docs/continuous-families.md): the
    # skew normal's alpha, the ex-Gaussian's beta, the shifted lognormal's ndt, each an ``ExtraParameter`` with its own prior
    # (``Priors(alpha=...)``), required exactly for its family and refused for every other. Empty for every other family.
    extra_parameters: tuple[ExtraParameter, ...] = NO_EXTRA_PARAMETERS
    # brms's response addition terms (B02, docs/addition-terms.md):
    addition_terms: frozenset[str] = frozenset()   # the addition terms besides trials() the family's likelihood takes: "weights" where a frozen brms case verifies it; "cens" and "trunc" where the family has tails (Gaussian; accepted against two references since P6); "mi" where the response's missing values may be parameters (Gaussian, C06)
    tails: Callable[[], object] | None = None      # () -> likelihoods.Tails: the log-CDF, log-CCDF and log mass between bounds (PyTensor, finite with a finite gradient) and the truncated mean and draws (NumPy) that cens() and trunc() need; None without them
    # Ordinal and categorical responses (B05, docs/ordinal-categorical.md). ``response_categories`` turns the response
    # column into brms's ``Y``: (ModelSpec, column) -> (the categories in order as labels, each row's 1-based category,
    # warnings); the model keeps the categories as ``model.categories`` and fits the codes. ``thresholds``: the intercept
    # is brms's vector of ordered thresholds ``Intercept`` (cumulative), which is not part of the linear predictor.
    # ``per_category``: one linear predictor per non-reference category (categorical), every coefficient with a trailing
    # category axis. ``bambi_setup``: (model, Bambi prior dict) -> (Bambi family, prior dict, whether Bambi centres the
    # design), for a family whose likelihood is built from the model (its categories, terms and column means).
    # ``check_formula`` refuses a formula the family does not take; ``intercept_lines`` is ``describe()``'s paragraph on
    # the intercept in place of the generic one. None / False for every other family.
    response_categories: Callable[[object, object], tuple] | None = None
    thresholds: bool = False
    per_category: bool = False
    bambi_setup: Callable[[object, dict], tuple] | None = None
    check_formula: Callable[[object], None] | None = None
    intercept_lines: Callable[[object], list[str]] | None = None
    # brms's distributional formulas (B01, docs/distributional.md): the auxiliary parameter ``scale_parameter`` may have
    # its own formula, ``["y ~ x", "sigma ~ z"]``, under this link (brms's default for sigma, shape and phi: "log", a key
    # of DISTRIBUTIONAL_INVERSE_LINKS); None: it takes no formula. Bambi builds it as a distributional component named
    # ``bambi_scale_parameter``.
    distributional_link: str | None = None
    default_distributional_intercept: Callable[[float], object] | None = None   # (brms's scale of the response read on the log scale) -> brms's default prior on that formula's intercept (def_dpar_prior: sigma student_t(3, 0, scale); shape and phi student_t(3, 0, 2.5)); A07
    # brms's monotonic effects mo(x) (B03, docs/monotonic.md): whether the family's linear predictor takes them, set where a
    # frozen brms case verifies it. The term is added to the parent's linear predictor before the link (it is not centred).
    monotonic: bool = False
    # brms's finite mixtures (C05, docs/mixtures.md): the model is ``mixture()`` of the component families the model holds
    # (``model.components``), one linear predictor per component (brms's ``mu<k>``) with ordered intercepts, a scale per
    # component (``sigma<k>``) and the mixing proportions ``theta``; ``Model``, ``Priors``, ``predict()``, the target identity
    # and the fit record take the paths of ``autobayes/families/mixture.py``. False for every other family.
    mixture: bool = False
    # brms's log_lik() (C07, docs/post-processing.md): (model, posterior, eta) -> the log density of each fitted row's response
    # per draw, (chain, draw, row), from the linear predictor ``eta`` (``predict(kind="linear")``'s, with a trailing category
    # axis for a per-category family) and the posterior's auxiliary parameters (each distributional parameter's per-row values
    # under its PyMC name, as ``predict()`` puts them there); ``log_lik()`` multiplies it by the weights. The NumPy forms are in
    # ``families.pointwise``. None: ``log_lik()``, and so ``loo()``, are not offered for the family.
    log_likelihood: Callable[[object, object, object], object] | None = None
    # brms's multivariate models (C01, docs/multivariate.md): whether ``mvbind(y, y2) ~ ...`` builds one model per response of
    # this family over the same data, with or without a residual correlation (brms's set_rescor()); set where a frozen case
    # verifies it.
    multivariate: bool = False
    # brms's non-linear formulas (C02, docs/nonlinear.md): (model, mu, {auxiliary parameter: its PyMC variable}) -> the
    # likelihood of a non-linear model, registered in the model being built, with ``mu`` the non-linear predictor on the link
    # scale (the deterministic ``model.names["linpred"]``); set where a frozen brms case verifies it. None: the family is not
    # offered with ``nl=True``.
    nonlinear_likelihood: Callable[[object, object, dict], object] | None = None
    # brms's ARMA terms ar(), ma(), arma() in the regression formulation (C04, docs/autocorrelation.md): whether the family's
    # linear predictor takes them, set where a frozen brms case verifies it. They are added last, on the residuals y - eta.
    autocorrelation: bool = False
    # brms's measurement-error term me(x, sdx) and missing-value term mi(col) with the response's mi() (C06,
    # docs/missing-and-measurement-error.md): whether the family's linear predictor takes them, set where a frozen case verifies
    # it (the response's own mi() is an addition term, "mi" in ``addition_terms``).
    measurement_error: bool = False
    missing_values: bool = False
    # brms's shrinkage priors horseshoe() and R2D2() on the whole class b (D07, docs/shrinkage-priors.md): whether the family
    # takes them, set where a frozen brms case verifies it. brms scales them by the residual SD sigma (autoscale = TRUE), which
    # the family's ``bambi_scale_parameter`` names.
    shrinkage: bool = False

    def with_link(self, link: str | None) -> "FamilySpec":
        """This family's spec under ``link``: this spec for ``None`` or its own link, the one a ``links`` entry builds
        for another of the family's links; otherwise a ``ValueError`` naming what the family accepts, or, for a link
        in ``parked_links``, why it is withheld."""
        if link is None or link == self.link:
            return self
        if isinstance(link, str) and link in self.links:
            return self.links[link]()
        if isinstance(link, str) and link in self.parked_links:
            raise ValueError(f"the {link} link of the {self.name} family is withheld: {self.parked_links[link]}")
        if self.links:
            raise ValueError(f"link must be one of {sorted([self.link, *self.links])} for the {self.name} family, got {link!r}")
        raise ValueError(f"the {self.name} family has the {self.link} link only; link={link!r} is not available")

    @property
    def continuous(self) -> bool:
        return self.response_kind == "continuous"

    @property
    def discrete(self) -> bool:
        return not self.continuous

    @property
    def has_trials(self) -> bool:
        """A ``successes | trials(n)`` response."""
        return self.response_kind == "binomial"
