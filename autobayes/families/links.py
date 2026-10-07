"""The links of the Bernoulli and binomial families (brms-parity A12): what each link contributes to a spec.

brms offers ``bernoulli(link = "logit" | "probit" | "cloglog" | "cauchit")`` and the same for ``binomial``; this
library builds models with the first two (``LINKS``) and withholds the cloglog link (``PARKED_LINKS``, with the
reason ``FamilySpec.with_link`` reports; docs/links.md). The likelihoods live in ``autobayes.likelihoods`` (the
logit's softplus form, the probit's log-CDF, each finite with a finite gradient for every finite ``eta`` on both the
PyTensor and the JAX path); here are the inverse links ``predict()`` applies and the words ``describe()`` uses. Under
every link the graph keeps the linear predictor as the parent ``<link>_p`` (``logit_p``, ``probit_p``) under Bambi's
identity link, so the deterministic the oracle compares with brms's ``posterior_linpred`` is ``eta`` itself.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.special import expit, ndtr


def inverse_cloglog(eta):
    """``1 - exp(-exp(eta))`` as ``-expm1(-exp(eta))``: exact where the subtraction returns 0 (``eta`` below about -37).
    The NumPy inverse of the withheld cloglog link, kept with its likelihood (``likelihoods.cloglog_log_p1``)."""
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
)})

# Links brms offers for these families that this library does not build a model with, and why (docs/links.md).
PARKED_LINKS = MappingProxyType({
    "cloglog": ("brms's own gradient of bernoulli_lpmf(y | inv_cloglog(eta)) is inaccurate where a y = 0 row has eta above "
                "about 2.7 (Stan forms 1 - theta by subtraction, losing eps * exp(exp(eta)) relative), so the frozen brms "
                "fixture cannot certify the link to the oracle's tolerance; the stable likelihood "
                "(autobayes.likelihoods.cloglog_log_p1 / cloglog_log_p0) is kept, tested and withheld until a fixture "
                "can (docs/links.md, divergence D18)"),
})
