"""AutoBayes: brms-like Bayesian hierarchical models on PyMC, with a JAX/NumPyro path.

Three model families on one right-hand side ``x + z [+ (1 | group)]`` with numeric predictors,
treatment-coded categorical predictors declared with ``categorical={column: reference}``
(M3a-1), numeric-by-categorical interactions ``x:z`` with both main effects (M3a-3) and
independent varying intercepts across distinct factors (M3c-3), optional since M3c-1 (a formula without it is a population-only
model, as in brms): Gaussian with identity link (milestone M0), Bernoulli with
logit link for a strictly 0/1 response (M1), and binomial counts ``successes | trials(n)`` with
logit link (M2). Everything else is refused with an error that names the unsupported
construct. See ``docs/m0-decision.md``, ``docs/m1-decision.md``, ``docs/m2-decision.md`` and
``docs/m3a1-decision.md``, ``docs/m3a2-decision.md`` and ``docs/m3c1-decision.md`` for what Bambi supplies and
what this layer adds.
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

from autobayes.formula import Categorical, FormulaError, ModelSpec, parse_formula
from autobayes.priors import (
    Exponential,
    HalfCauchy,
    HalfNormal,
    HalfStudentT,
    Normal,
    Prior,
    PriorError,
    Priors,
    StudentT,
)
from autobayes.model import Model, bernoulli, binomial, gaussian, student
from autobayes.inference import DeviceError, FitResult, fit
from autobayes.predict import predict
from autobayes.diagnostics import panel
from autobayes.session import compare_records, environment, fit_record, format_comparison, format_environment

__all__ = [
    "Categorical", "DeviceError", "Exponential", "FitResult", "FormulaError", "HalfCauchy",
    "HalfNormal", "HalfStudentT", "Model", "ModelSpec", "Normal", "Prior",
    "PriorError", "Priors", "StudentT", "bernoulli", "binomial", "compare_records", "environment", "fit", "fit_record",
    "student", "format_comparison", "format_environment", "gaussian", "panel", "parse_formula", "predict",
]
__version__ = "0.8.0"
