"""What one response family contributes to the library: a ``FamilySpec``.

Everything the library once decided by comparing a family name against a string is a field of
the spec. A family is one module in this package that builds its spec (``gaussian.py``,
``student.py``, ``poisson.py``, ``negbinomial.py``, ``gamma.py``, ``lognormal.py``, ``beta.py``, ``bernoulli.py``,
``binomial.py``, ``zero_inflated_poisson.py``, ``hurdle_poisson.py``, ``cumulative.py``, ``categorical.py``, ``mixture.py``);
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
    # brms's response addition terms (B02, docs/addition-terms.md):
    addition_terms: frozenset[str] = frozenset()   # the addition terms besides trials() the family's likelihood takes: "weights" where a frozen brms case verifies it; "cens" and "trunc" where the family has tails (Gaussian; accepted against two references since P6)
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
