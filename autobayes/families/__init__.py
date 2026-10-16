"""The family registry: one ``FamilySpec`` per response family, looked up by name (and link).

Adding a family is one module in this package that builds its spec, plus its entry in ``FAMILIES``;
see docs/families.md. A family with several links (Bernoulli, binomial: logit, probit and cloglog) registers its
default link's spec and reaches the others through ``FamilySpec.links`` (docs/links.md).
Nothing outside this package compares a family name against a string (``tests/test_family_registry.py`` enforces that).
"""
from __future__ import annotations

from types import MappingProxyType

from autobayes.families.bernoulli import BERNOULLI
from autobayes.families.beta import BETA
from autobayes.families.binomial import BINOMIAL
from autobayes.families.categorical import CATEGORICAL
from autobayes.families.cumulative import CUMULATIVE
from autobayes.families.exgaussian import EXGAUSSIAN
from autobayes.families.exponential import EXPONENTIAL
from autobayes.families.gamma import GAMMA
from autobayes.families.gaussian import GAUSSIAN
from autobayes.families.hurdle_poisson import HURDLE_POISSON
from autobayes.families.lognormal import LOGNORMAL
from autobayes.families.mixture import MIXTURE
from autobayes.families.negbinomial import NEGBINOMIAL
from autobayes.families.poisson import POISSON
from autobayes.families.shifted_lognormal import SHIFTED_LOGNORMAL
from autobayes.families.skew_normal import SKEW_NORMAL
from autobayes.families.spec import FamilySpec
from autobayes.families.student import STUDENT
from autobayes.families.weibull import WEIBULL
from autobayes.families.zero_inflated_poisson import ZERO_INFLATED_POISSON

FAMILIES = MappingProxyType({spec.name: spec for spec in (GAUSSIAN, STUDENT, POISSON, NEGBINOMIAL, GAMMA, LOGNORMAL, BETA, BERNOULLI, BINOMIAL,
                                                          ZERO_INFLATED_POISSON, HURDLE_POISSON, CUMULATIVE, CATEGORICAL, MIXTURE,
                                                          WEIBULL, EXPONENTIAL, SKEW_NORMAL, SHIFTED_LOGNORMAL, EXGAUSSIAN)})   # D01


def get_family(name: str, link: str | None = None) -> FamilySpec:
    """The registered specification of ``name`` (the error lists every family), under ``link`` when one is given
    (``FamilySpec.with_link``: the registered spec for ``None`` or the family's default link)."""
    if name not in FAMILIES:
        raise ValueError(f"family must be one of {sorted(FAMILIES)}, got {name!r}")
    return FAMILIES[name].with_link(link)


def family_of(model) -> FamilySpec:
    """The spec of a built model, or of any object exposing ``family`` and (optionally) ``link``, looked up by name."""
    return get_family(model.family, getattr(model, "link", None))


__all__ = ["FAMILIES", "FamilySpec", "family_of", "get_family"]
