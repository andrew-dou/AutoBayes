"""AutoBayes: brms-like Bayesian hierarchical models on PyMC, with a JAX/NumPyro path.

Milestone M0 supports one model family: a Gaussian regression with numeric predictors
and one varying intercept, ``y ~ x + (1 | group)``. Everything else is refused with an
error that names the unsupported construct. See ``docs/m0-decision.md`` for what Bambi
supplies and what this layer adds.
"""
from autobayes.formula import FormulaError, ModelSpec, parse_formula
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
from autobayes.model import Model, gaussian
from autobayes.inference import DeviceError, FitResult, fit
from autobayes.predict import predict
from autobayes.diagnostics import panel

__all__ = [
    "DeviceError", "Exponential", "FitResult", "FormulaError", "HalfCauchy",
    "HalfNormal", "HalfStudentT", "Model", "ModelSpec", "Normal", "Prior",
    "PriorError", "Priors", "StudentT", "fit", "gaussian", "panel",
    "parse_formula", "predict",
]
__version__ = "0.0.1"
