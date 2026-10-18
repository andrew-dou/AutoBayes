"""Bayesian hierarchical models built through Bambi/PyMC, with JAX/NumPyro sampling.

Response families: Gaussian and Student-t (nu fixed or learned) with identity links,
Bernoulli and grouped binomial with a logit or probit link, Poisson and negative binomial
counts and Gamma responses with a log link, lognormal responses with the identity
link on the log-scale location, Beta proportions with a logit link and precision ``phi``, and zero-inflated and hurdle
Poisson counts with the zero-inflation or hurdle probability ``zi`` / ``hu`` under a ``Beta`` prior, ordinal responses with
brms's cumulative logit and probit models, its sratio, cratio and acat models (with category-specific effects ``cs(x)``; D04)
and unordered categorical responses with one linear predictor per non-reference category,
and finite mixtures of Gaussian components with ordered intercepts and Dirichlet mixing proportions; Weibull and
exponential responses with a log link on the mean, skew-normal and ex-Gaussian responses parameterised by their mean, and
shifted lognormal responses with a non-decision time below the smallest response (D01); beta-binomial responses with
trials, a logit link on the mean success probability and a precision ``phi``, and geometric and dispersion-parameterised
negative binomial counts (``negbinomial2``, ``sigma = 1 / shape``) with a log link (D02); zero-inflated and hurdle
negative binomial counts, zero-inflated binomial successes and Beta proportions, and hurdle Gamma and lognormal
responses (D03).
Supported formulas combine numeric and declared categorical predictors,
numeric-by-categorical interactions, intercept controls, and optional multiple
independent varying-intercept factors with independent or correlated varying slopes, and a formula of its own for the
auxiliary parameter (``["y ~ x", "sigma ~ z"]``: sigma, shape or phi under a log link), and brms's monotonic effects
``mo(x)`` of ordinal predictors with a Dirichlet prior on their simplex, brms's Gaussian processes ``gp(x)`` (exact) and
``gp(x, k = 10, c = 5/4)`` (Hilbert-space approximation) of one numeric column, and brms's measurement-error term ``me(x, sdx)``
and its missing-value imputation ``mi()`` in a list of formulas, one per response. Explicit priors are required;
brms's own defaults are available explicitly with ``default_priors(formula, data, ...)``.

See docs/current-capabilities.md for supported composition, test entry points,
and the distinction between deterministic checks, CPU examples and GPU evidence.
Unsupported formulas and family options are rejected explicitly.
"""
import os as _os

# Two JAX settings that must precede anything else, done at import because that is the
# one place that reliably precedes a researcher's first jax.devices() call:
# - float64. JAX defaults to float32, which is not acceptable for a log density you
#   intend to trust (reference/failure-atlas.md, "Silent float32"). fit() re-asserts it.
# - the host device count, so CPU chains on the NumPyro path run in parallel. This is
#   a no-op if JAX is already initialised (fit() then reports sequential chains and
#   says why) and is skipped when AUTOBAYES_HOST_DEVICES=0.
try:
    import numpyro as _numpyro
    import jax as _jax

    _n = int(_os.environ.get("AUTOBAYES_HOST_DEVICES", min(_os.cpu_count() or 1, 8)))
    if _n > 0:
        _numpyro.set_host_device_count(_n)
    _jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

from autobayes.formula import Categorical, FormulaError, GroupTerm, ModelSpec, PopTerm, parse_formula
from autobayes.priors import (
    Beta,
    Dirichlet,
    Exponential,
    Flat,
    Gamma,
    HalfCauchy,
    HalfNormal,
    HalfStudentT,
    InverseGamma,
    LKJ,
    Normal,
    Prior,
    PriorError,
    Priors,
    StudentT,
    Uniform,
)
from autobayes.default_priors import default_priors
from autobayes.model import (Model, bernoulli, beta, binomial, categorical, cumulative, gamma, gaussian, hurdle_poisson, lognormal,
                             mixture, negbinomial, student, poisson, zero_inflated_poisson)
from autobayes.multivariate import MultivariateModel
from autobayes.inference import DeviceError, FitResult, fit
from autobayes.predict import predict
from autobayes.prior_predictive import prior_predictive
from autobayes.post_processing import Loo, bayes_R2, log_lik, loo
from autobayes.post_processing import Waic, loo_compare, model_weights, waic   # D05, docs/post-processing.md
from autobayes.hypothesis import Hypothesis, hypothesis                       # D06, docs/post-processing.md
from autobayes.conditional_effects import conditional_effects                # D06
from autobayes.prior_draws import prior_draws                                # E05: brms's sample_prior draws, docs/post-processing.md
from autobayes.diagnostics import panel
from autobayes.session import compare_records, environment, fit_record, format_comparison, format_environment
from autobayes.nonlinear import NonlinearModel, NonlinearSpec        # non-linear formulas, nl=True (C02, docs/nonlinear.md)
from autobayes.model import exgaussian, exponential, shifted_lognormal, skew_normal, weibull   # D01, docs/continuous-families.md
from autobayes.model import beta_binomial, geometric, negbinomial2   # D02, docs/discrete-families.md
from autobayes.shrinkage import Horseshoe, R2D2   # D07: brms's horseshoe() and R2D2() on b, docs/shrinkage-priors.md
from autobayes.model import acat, cratio, sratio                    # D04, docs/ordinal-categorical.md
from autobayes.model import (hurdle_gamma, hurdle_lognormal, hurdle_negbinomial, zero_inflated_beta,   # D03, docs/zero-inflated-hurdle.md
                             zero_inflated_binomial, zero_inflated_negbinomial)

__all__ = [
    "Beta", "Categorical", "DeviceError", "Dirichlet", "Exponential", "FitResult", "Flat", "FormulaError", "Gamma", "GroupTerm", "HalfCauchy",
    "HalfNormal", "HalfStudentT", "InverseGamma", "LKJ", "Model", "ModelSpec", "MultivariateModel", "Normal", "PopTerm", "Prior",
    "PriorError", "Priors", "StudentT", "bernoulli", "beta", "binomial", "compare_records", "default_priors", "environment", "fit", "fit_record",
    "gamma", "lognormal", "negbinomial", "student", "poisson", "format_comparison", "format_environment", "gaussian", "panel",
    "parse_formula", "predict", "prior_predictive", "zero_inflated_poisson", "hurdle_poisson", "cumulative", "categorical",
    "mixture",
    "Loo", "bayes_R2", "log_lik", "loo",
]
__all__ += ["NonlinearModel", "NonlinearSpec"]         # C02
__all__ += ["Waic", "loo_compare", "model_weights", "waic"]   # D05
__all__ += ["Uniform", "exgaussian", "exponential", "shifted_lognormal", "skew_normal", "weibull"]   # D01
__all__ += ["Hypothesis", "conditional_effects", "hypothesis"]   # D06
__all__ += ["beta_binomial", "geometric", "negbinomial2"]   # D02
__all__ += ["Horseshoe", "R2D2"]                       # D07
__all__ += ["acat", "cratio", "sratio"]                             # D04
__all__ += ["hurdle_gamma", "hurdle_lognormal", "hurdle_negbinomial", "zero_inflated_beta", "zero_inflated_binomial",
            "zero_inflated_negbinomial"]   # D03
__all__ += ["prior_draws"]                             # E05
__version__ = "0.8.0"
