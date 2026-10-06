"""What one response family contributes to the library: a ``FamilySpec``.

Everything the library once decided by comparing a family name against a string is a field of
the spec. A family is one module in this package that builds its spec (``gaussian.py``,
``student.py``, ``poisson.py``, ``bernoulli.py``, ``binomial.py``); ``autobayes.families.FAMILIES``
is the registry and ``get_family(name)`` the lookup. Callable fields take the built ``Model`` (or
the frame being validated) because that is what the call sites have in hand; none of them is
called before ``Model.__init__`` has resolved the spec.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

NO_FIXED_PARAMETERS: Mapping[str, Callable[[object], float]] = MappingProxyType({})
STANDARD_PREDICTION_KINDS = frozenset({"mean", "response", "linear"})
LOGIT_INTERCEPT_NOTE = (                     # shared by the logit-link families
    "  expit(a_c) is the probability at that proportion-weighted predictor; it is NOT the",
    "  average probability across the levels (the link is nonlinear).",
)


@dataclass(frozen=True, eq=False)
class FamilySpec:
    """One response family. Fields in the order the model consults them."""

    name: str                                    # the registry key and ``model.family``
    tag: str                                     # the ``describe()`` heading: "M0", "M1", "M2", "Student-t", "Poisson"
    parent: str                                  # Bambi's parent parameter, the linear predictor's name: "mu" or "logit_p"
    link: str                                    # "identity", "logit" or "log"
    link_scale: str | None                       # the scale the location priors live on when it is not the response ("logit", "log")
    mean_name: str                               # the name ``predict()`` gives the expected response: "mu" or "p"
    response_kind: str                           # "continuous", "binary", "count" or "binomial" (successes with trials)
    response_dtype: object | None                # the response column and its draws are cast to this (int64 for the discrete families)
    scale_parameter: str | None                  # the auxiliary scale with its own prior ("sigma"), or None
    fixed_parameters: Mapping[str, Callable[[object], float]]   # user-supplied fixed scalars by name, each with its check ("nu")
    bambi_family: Callable[[], object]           # the Bambi family object or name handed to ``bmb.Model``
    validate_response: Callable[[object, object], None]         # (ModelSpec, DataFrame): refuse a response outside the family's support
    response_warnings: Callable[[object, object], list[str]]    # (ModelSpec, fitted frame): data warnings the model records
    response_upper_bound: Callable[[object], object]            # (model): upper bound of prior-predictive counts, or None
    inverse_link: Callable[[object], object]     # linear predictor -> expected response, with the family's own checks
    draw_response: Callable[[object, object, object, object, object], object]   # (model, posterior, mean, frame, rng) -> new observations
    manual_prediction: bool                      # ``predict()`` always assembles this family's predictions by hand
    prediction_kinds: frozenset[str]             # the ``kind`` values ``predict()`` accepts
    kind_error: str                              # the message for any other ``kind``
    sigma_rule: str                              # ``value_point``'s message when ``sigma`` is wrongly given or omitted
    sigma_missing_error: str | None              # ``Priors.validate``'s message when the scale prior is missing (families with one)
    linpred_symbol: str                          # "mu" or "eta" in the written-out linear predictor
    scale_phrase: str                            # "response", "log-odds" or "log mean": the scale coefficients act on
    intercept_meaning: str                       # what the centred intercept is ("expected response", "log-odds of y = 1", ...)
    mean_lines: Callable[[object], list[str]]    # (model): ``describe()`` lines relating the mean to the linear predictor
    likelihood_lines: Callable[[object], list[str]]   # (model): the scale prior, fixed parameters and likelihood paragraph
    centred_intercept_note: tuple[str, ...]      # ``describe()`` lines after a centred intercept with categorical predictors
    deterministic_note: str                      # the tail of ``describe()``'s "Deterministic:" line

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
