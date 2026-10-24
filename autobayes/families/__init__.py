"""The family registry: one ``FamilySpec`` per response family, looked up by name (and link).

Adding a family is one module in this package that builds its spec, plus its entry in ``FAMILIES``;
see docs/families.md. A family with several links (Bernoulli, binomial: logit, probit and cloglog) registers its
default link's spec and reaches the others through ``FamilySpec.links`` (docs/links.md); so does the cumulative family
for its probit link (D04), and the inverse Gaussian family for its log link (E02). The count-matrix families (F01a:
``multinomial``, ``dirichlet_multinomial``; ``FamilySpec.matrix_response``) are registered here and built by
``autobayes.multinomial``.
Nothing outside this package compares a family name against a string (``tests/test_family_registry.py`` enforces that).
"""
from __future__ import annotations

from types import MappingProxyType

from autobayes.families.asym_laplace import ASYM_LAPLACE
from autobayes.families.bernoulli import BERNOULLI
from autobayes.families.beta import BETA
from autobayes.families.beta_binomial import BETA_BINOMIAL
from autobayes.families.binomial import BINOMIAL
from autobayes.families.categorical import CATEGORICAL
from autobayes.families.cumulative import CUMULATIVE
from autobayes.families.exgaussian import EXGAUSSIAN
from autobayes.families.exponential import EXPONENTIAL
from autobayes.families.frechet import FRECHET
from autobayes.families.gamma import GAMMA
from autobayes.families.gaussian import GAUSSIAN
from autobayes.families.gen_extreme_value import GEN_EXTREME_VALUE
from autobayes.families.geometric import GEOMETRIC
from autobayes.families.hurdle_cumulative import HURDLE_CUMULATIVE
from autobayes.families.hurdle_gamma import HURDLE_GAMMA
from autobayes.families.hurdle_lognormal import HURDLE_LOGNORMAL
from autobayes.families.hurdle_negbinomial import HURDLE_NEGBINOMIAL
from autobayes.families.hurdle_poisson import HURDLE_POISSON
from autobayes.families.inverse_gaussian import INVERSE_GAUSSIAN
from autobayes.families.lognormal import LOGNORMAL
from autobayes.families.mixture import MIXTURE
from autobayes.families.multinomial import DIRICHLET_MULTINOMIAL, MULTINOMIAL
from autobayes.families.negbinomial import NEGBINOMIAL
from autobayes.families.negbinomial2 import NEGBINOMIAL2
from autobayes.families.ordinal import ACAT, CRATIO, SRATIO
from autobayes.families.poisson import POISSON
from autobayes.families.shifted_lognormal import SHIFTED_LOGNORMAL
from autobayes.families.skew_normal import SKEW_NORMAL
from autobayes.families.spec import FamilySpec
from autobayes.families.student import STUDENT
from autobayes.families.von_mises import VON_MISES
from autobayes.families.weibull import WEIBULL
from autobayes.families.wiener import WIENER
from autobayes.families.xbeta import XBETA
from autobayes.families.zero_inflated_beta import ZERO_INFLATED_BETA
from autobayes.families.zero_inflated_beta_binomial import ZERO_INFLATED_BETA_BINOMIAL
from autobayes.families.zero_inflated_binomial import ZERO_INFLATED_BINOMIAL
from autobayes.families.zero_inflated_negbinomial import ZERO_INFLATED_NEGBINOMIAL
from autobayes.families.zero_inflated_poisson import ZERO_INFLATED_POISSON
from autobayes.families.zero_one_inflated_beta import ZERO_ONE_INFLATED_BETA

FAMILIES = MappingProxyType({spec.name: spec for spec in (GAUSSIAN, STUDENT, POISSON, NEGBINOMIAL, GAMMA, LOGNORMAL, BETA, BERNOULLI, BINOMIAL,
                                                          ZERO_INFLATED_POISSON, HURDLE_POISSON, CUMULATIVE, CATEGORICAL, MIXTURE,
                                                          WEIBULL, EXPONENTIAL, SKEW_NORMAL, SHIFTED_LOGNORMAL, EXGAUSSIAN,   # D01
                                                          BETA_BINOMIAL, GEOMETRIC, NEGBINOMIAL2,                             # D02
                                                          SRATIO, CRATIO, ACAT,                                               # D04
                                                          ZERO_INFLATED_NEGBINOMIAL, HURDLE_NEGBINOMIAL, ZERO_INFLATED_BINOMIAL,   # D03
                                                          ZERO_INFLATED_BETA, HURDLE_GAMMA, HURDLE_LOGNORMAL,
                                                          ZERO_ONE_INFLATED_BETA, ZERO_INFLATED_BETA_BINOMIAL, HURDLE_CUMULATIVE,   # E01
                                                          XBETA,
                                                          INVERSE_GAUSSIAN, VON_MISES, ASYM_LAPLACE, FRECHET, GEN_EXTREME_VALUE,   # E02
                                                          WIENER,   # F04
                                                          MULTINOMIAL, DIRICHLET_MULTINOMIAL)})                                # F01a


def get_family(name: str, link: str | None = None) -> FamilySpec:
    """The registered specification of ``name`` (the error lists every family), under ``link`` when one is given
    (``FamilySpec.with_link``: the registered spec for ``None`` or the family's default link)."""
    if name not in FAMILIES:
        raise ValueError(f"family must be one of {sorted(FAMILIES)}, got {name!r}")
    return FAMILIES[name].with_link(link)


def family_of(model) -> FamilySpec:
    """The spec of a built model, or of any object exposing ``family`` and (optionally) ``link``, looked up by name. A model's
    own ``family_spec`` of that family is returned as it is: the registered spec under the model's link, or (E02) the family's
    spec with an extra parameter fixed (``FamilySpec.with_fixed``: the asymmetric Laplace quantile), which no lookup by name
    and link reaches."""
    own = getattr(model, "family_spec", None)
    if isinstance(own, FamilySpec) and own.name == model.family:
        return own
    return get_family(model.family, getattr(model, "link", None))


__all__ = ["FAMILIES", "FamilySpec", "family_of", "get_family"]
