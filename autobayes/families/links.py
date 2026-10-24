"""The links of the Bernoulli and binomial families (brms-parity A12): what each link contributes to a spec.

brms offers ``bernoulli(link = "logit" | "probit" | "cloglog" | "cauchit")`` and the same for ``binomial``; this
library builds models with the first three (``LINKS``; the cloglog link since P6, accepted against two references where
brms's own gradient is inexact, divergence D18) and does not offer the cauchit link. ``PARKED_LINKS`` is the place for a
link brms offers that this library withholds, with the reason ``FamilySpec.with_link`` reports (none at present;
docs/links.md). The likelihoods live in ``autobayes.likelihoods`` (the logit's softplus form, the probit's log-CDF, the
cloglog's ``log1mexp(-exp(eta))`` and ``-exp(eta)``, each finite with a finite gradient for every finite ``eta`` on both
the PyTensor and the JAX path); here are the inverse links ``predict()`` applies and the words ``describe()`` uses.
Under every link the graph keeps the linear predictor as the parent ``<link>_p`` (``logit_p``, ``probit_p``,
``cloglog_p``) under Bambi's identity link, so the deterministic the oracle compares with brms's ``posterior_linpred`` is
``eta`` itself.

E06 (docs/links.md) adds links of other Tier A families, each spec built in its family's module: ``poisson("sqrt")``,
``negbinomial("identity")``, ``Gamma("inverse")`` and ``Gamma("identity")``, ``Beta("probit")`` and ``Beta("cloglog")``, and
``cumulative("cloglog")``. Here are what they share: the inverse links of the mean (``square_mean``, ``positive_mean``), the
complement ``1 - mu`` of a probability link computed directly (``COMPLEMENTS``), and ``check_link_formula``, the refusal of
the constructs no frozen case verifies under such a link (divergence D133).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.special import expit, ndtr


def inverse_cloglog(eta):
    """``1 - exp(-exp(eta))`` as ``-expm1(-exp(eta))``: exact where the subtraction returns 0 (``eta`` below about -37).
    The NumPy inverse of the cloglog link (its likelihood is ``likelihoods.cloglog_log_p1`` / ``cloglog_log_p0``)."""
    return -np.expm1(-np.exp(eta))


INVERSE_LINKS = MappingProxyType({"logit": expit, "probit": ndtr, "cloglog": inverse_cloglog})   # eta -> P(y = 1), NumPy


@dataclass(frozen=True)
class Link:
    """One link of a probability family. ``parent`` is Bambi's parent parameter, the deterministic that holds ``eta``;
    ``inverse_link`` maps ``eta`` to the probability (NumPy); the strings are the words ``describe()`` prints."""

    name: str
    parent: str
    inverse_link: Callable[[object], object]
    probability: str              # p_i as a formula in eta_i: "expit(eta_i)", "Phi(eta_i)"
    probability_gloss: str        # what follows it the first time: " = 1 / (1 + exp(-eta_i))", ", the standard Normal CDF"
    scale: str                    # the scale coefficients act on: "log-odds", "probit"
    of_probability: str           # what eta is of P(y = 1): "log-odds of", "probit (Phi^-1) of the probability of"
    trial_success: str            # the binomial intercept's meaning: "per-trial success log-odds", ...
    inverse_of_intercept: str     # the probability at the centred intercept: "expit(a_c)", "Phi(a_c)"

    @property
    def centred_intercept_note(self) -> tuple[str, str]:
        """``describe()``'s note after a centred intercept with categorical predictors."""
        return (f"  {self.inverse_of_intercept} is the probability at that proportion-weighted predictor; it is NOT the",
                "  average probability across the levels (the link is nonlinear).")


LINKS = MappingProxyType({link.name: link for link in (
    Link("logit", "logit_p", expit, "expit(eta_i)", " = 1 / (1 + exp(-eta_i))", "log-odds", "log-odds of",
         "per-trial success log-odds", "expit(a_c)"),
    Link("probit", "probit_p", ndtr, "Phi(eta_i)", ", the standard Normal CDF", "probit",
         "probit (Phi^-1) of the probability of", "per-trial success probability on the probit scale", "Phi(a_c)"),
    Link("cloglog", "cloglog_p", inverse_cloglog, "1 - exp(-exp(eta_i))", ", the inverse complementary log-log",
         "complementary log-log", "complementary log-log (log(-log(1 - p))) of the probability of",
         "per-trial success probability on the complementary log-log scale", "1 - exp(-exp(a_c))"),
)})

# Links brms offers for these families that this library does not build a model with, and why (docs/links.md): none since
# P6, which lifted the cloglog link's park (divergence D18).
PARKED_LINKS = MappingProxyType({})


# -- E06: the links of the Poisson, negative binomial, Gamma, Beta and cumulative families -------------------------------------
COMPLEMENTS = MappingProxyType({                           # eta -> 1 - P(y = 1) (or 1 - mu), NumPy, never by subtraction
    "logit": lambda eta: expit(-np.asarray(eta, dtype=float)),
    "probit": lambda eta: ndtr(-np.asarray(eta, dtype=float)),
    "cloglog": lambda eta: np.exp(-np.exp(np.asarray(eta, dtype=float))),
})


def square_mean(eta, what: str):
    """``eta^2``, the mean under the sqrt link (brms squares the linear predictor; every real ``eta`` gives a valid mean),
    refused rather than clipped when it overflows."""
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.square(np.asarray(eta, dtype=float))
    if not np.isfinite(mean).all():
        raise ValueError(f"{what} overflow/nonfinite prediction; inspect the square-root-mean draws")
    return mean


def positive_mean(eta, link: str, what: str):
    """The mean under the identity (``eta``) or the inverse (``1 / eta``) link, refused where it is not a positive finite
    number: brms's ``posterior_epred`` returns such a mean unchecked and its ``posterior_predict`` draws NaN there, while the
    model's log density is ``-inf`` (brms's Stan rejects it); here ``predict()`` names the rows (divergence D134)."""
    eta = np.asarray(eta, dtype=float)
    bad = ~(eta > 0) | ~np.isfinite(eta)
    if bad.any():
        rows = np.unique(np.nonzero(bad)[-1])
        raise ValueError(f"{what}: the {link} link gives a non-positive or nonfinite mean (linear predictor <= 0) for some draws at "
                         f"{rows.size} row(s), the first {rows[:5].tolist()}; the {link} link needs a positive linear predictor "
                         "(brms's posterior_epred returns that mean unchecked and posterior_predict draws NaN there; divergence D134)")
    with np.errstate(divide="ignore", over="ignore"):
        mean = 1.0 / eta if link == "inverse" else eta
    if not (np.isfinite(mean).all() and (mean > 0).all()):
        raise ValueError(f"{what}: the {link} link's mean underflows or overflows for some draws; inspect the linear predictor")
    return mean


# The constructs a formula may hold under an E06 link, and what is refused: the term model's population- and group-level terms
# (numeric and categorical predictors, interactions, transforms, varying intercepts and slopes, correlated or not, nested) and
# offset() enter the linear predictor as under the family's own link, and the frozen cases check (1 | g) and offset(); the
# special terms, the addition terms and a distributional formula are refused, since no frozen case verifies them under the
# link (divergence D133).
def check_link_formula(spec, fam) -> None:
    """Refuse, naming the construct and the link, what no frozen brms case verifies under the E06 link of ``fam``: ``mo()``,
    ``gp()``, ``mm()``, ``gr(by = )``, ``me()``, ``mi()``, ``cs()``, an autocorrelation term, an addition term (``weights()``,
    ``cens()``, ``trunc()``) and a distributional formula (divergence D133). ``spec`` is the parsed ``ModelSpec``."""
    from autobayes.families import FAMILIES
    from autobayes.formula import FormulaError

    found = []
    found += [f"the monotonic term {t.name!r}" for t in getattr(spec, "mo_terms", ())]
    found += [f"the Gaussian process {g.term!r}" for g in getattr(spec, "gp_terms", ())]
    found += [f"the group term '{t.canonical}'" for t in getattr(spec, "custom_group_terms", ())]
    found += [f"the measurement-error term {t.name!r}" for t in getattr(spec, "me_terms", ())]
    found += [f"the missing-value term {t.name!r}" for t in getattr(spec, "mi_terms", ())]
    found += [f"the category-specific term {t.name!r}" for t in getattr(spec, "cs_terms", ())]
    found += [f"the autocorrelation term {spec.autocor.term!r}"] if getattr(spec, "autocor", None) is not None else []
    found += [f"the CAR term {spec.car.term!r}"] if getattr(spec, "car", None) is not None else []   # F02a
    found += [f"the addition term {a.term!r}" for a in getattr(spec, "additions", ())]
    found += [f"the distributional formula '{d.formula}'" for d in getattr(spec, "dpars", ())]
    if found:
        raise FormulaError(
            f"{spec.formula!r}: {found[0]} is not available under the {fam.link} link of the {fam.name} family: no frozen brms case "
            f"verifies it with that link (the link's formula takes population- and group-level terms and offset(); its "
            f"{FAMILIES[fam.name].link} link takes more; docs/links.md, divergence D133)")
