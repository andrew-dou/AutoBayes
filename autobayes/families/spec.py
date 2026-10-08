"""What one response family contributes to the library: a ``FamilySpec``.

Everything the library once decided by comparing a family name against a string is a field of
the spec. A family is one module in this package that builds its spec (``gaussian.py``,
``student.py``, ``poisson.py``, ``negbinomial.py``, ``gamma.py``, ``lognormal.py``, ``beta.py``, ``bernoulli.py``,
``binomial.py``);
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
    parked_links: Mapping[str, str] = field(default_factory=lambda: NO_PARKED_LINKS)   # links brms offers for this family that this library withholds, each with the reason ``with_link`` reports (Bernoulli, binomial: cloglog; docs/links.md)
    learnable_parameters: Mapping[str, float] = field(default_factory=lambda: NO_LEARNABLE_PARAMETERS)   # fixed parameters a model learns instead when none is given, each under its own prior truncated at the lower bound given here (Student-t: nu > 1, brms's real<lower=1> nu; docs/student-learned-nu.md)
    default_learned_priors: Mapping[str, Callable[[], object]] = field(default_factory=lambda: NO_LEARNED_DEFAULTS)   # brms's default prior of each learnable parameter (Student-t nu: gamma(2, 0.1), Gamma(2, 0.1)); A07

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
