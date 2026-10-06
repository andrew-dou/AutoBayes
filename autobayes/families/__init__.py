"""The family registry: one ``FamilySpec`` per response family, looked up by name.

Adding a family is one module in this package that builds its spec, plus its entry in ``FAMILIES``;
see docs/families.md. Nothing outside this package compares a family name against a string
(``tests/test_family_registry.py`` enforces that).
"""
from __future__ import annotations

from types import MappingProxyType

from autobayes.families.bernoulli import BERNOULLI
from autobayes.families.binomial import BINOMIAL
from autobayes.families.gaussian import GAUSSIAN
from autobayes.families.poisson import POISSON
from autobayes.families.spec import FamilySpec
from autobayes.families.student import STUDENT

FAMILIES = MappingProxyType({spec.name: spec for spec in (GAUSSIAN, STUDENT, POISSON, BERNOULLI, BINOMIAL)})


def get_family(name: str) -> FamilySpec:
    """The registered specification of ``name``; the error lists every family."""
    if name not in FAMILIES:
        raise ValueError(f"family must be one of {sorted(FAMILIES)}, got {name!r}")
    return FAMILIES[name]


__all__ = ["FAMILIES", "FamilySpec", "get_family"]
