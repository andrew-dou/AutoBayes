"""Bayesian hierarchical models built through Bambi/PyMC, with JAX/NumPyro sampling.

Response families: Gaussian and Student-t (nu fixed or learned) with identity links,
Bernoulli and grouped binomial with a logit or probit link, Poisson and negative binomial
counts and Gamma responses with a log link, lognormal responses with the identity
link on the log-scale location, Beta proportions with a logit link and precision ``phi``, and zero-inflated and hurdle
Poisson counts with the zero-inflation or hurdle probability ``zi`` / ``hu`` under a ``Beta`` prior, ordinal responses with
brms's cumulative logit model and unordered categorical responses with one linear predictor per non-reference category.
Supported formulas combine numeric and declared categorical predictors,
numeric-by-categorical interactions, intercept controls, and optional multiple
independent varying-intercept factors with independent or correlated varying slopes, and a formula of its own for the
auxiliary parameter (``["y ~ x", "sigma ~ z"]``: sigma, shape or phi under a log link), and brms's monotonic effects
``mo(x)`` of ordinal predictors with a Dirichlet prior on their simplex. Explicit priors are required;
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
)
from autobayes.default_priors import default_priors
from autobayes.model import (Model, bernoulli, beta, binomial, categorical, cumulative, gamma, gaussian, hurdle_poisson, lognormal,
                             negbinomial, student, poisson, zero_inflated_poisson)
from autobayes.inference import DeviceError, FitResult, fit
from autobayes.predict import predict
from autobayes.prior_predictive import prior_predictive
from autobayes.diagnostics import panel
from autobayes.session import compare_records, environment, fit_record, format_comparison, format_environment

__all__ = [
    "Beta", "Categorical", "DeviceError", "Dirichlet", "Exponential", "FitResult", "Flat", "FormulaError", "Gamma", "GroupTerm", "HalfCauchy",
    "HalfNormal", "HalfStudentT", "InverseGamma", "LKJ", "Model", "ModelSpec", "Normal", "PopTerm", "Prior",
    "PriorError", "Priors", "StudentT", "bernoulli", "beta", "binomial", "compare_records", "default_priors", "environment", "fit", "fit_record",
    "gamma", "lognormal", "negbinomial", "student", "poisson", "format_comparison", "format_environment", "gaussian", "panel",
    "parse_formula", "predict", "prior_predictive", "zero_inflated_poisson", "hurdle_poisson", "cumulative", "categorical",
]
__version__ = "0.8.0"
