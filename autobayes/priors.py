"""Explicit priors, by parameter class, in brms terms.

brms classes: ``Intercept``, ``b`` (slopes), ``sd`` (group SD) and, for the Gaussian family
only, ``sigma`` (residual SD). Every class the family has must be given; there are no
silent defaults here (Bambi would supply some, and that is exactly what this layer refuses
to let happen). For the Bernoulli family the intercept and slope priors are on the logit
scale and ``sigma`` must be omitted.

The parameterisation follows PyMC/Bambi: ``Normal(mu, sigma)`` is mean and standard
deviation, ``StudentT(nu, mu, sigma)`` has a scale ``sigma``, ``Exponential(lam)`` is a
rate, ``HalfCauchy(beta)`` a scale. This is also brms's convention.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

_LOCATION = {"Normal", "StudentT"}
_SCALE = {"HalfNormal", "HalfStudentT", "Exponential", "HalfCauchy"}


class PriorError(ValueError):
    """A missing, extra, or ill-typed prior."""


@dataclass(frozen=True)
class Prior:
    name: str
    params: Mapping[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        inside = ", ".join(f"{k}={v:g}" for k, v in self.params.items())
        return f"{self.name}({inside})"

    def to_bambi(self):
        import bambi as bmb

        return bmb.Prior(self.name, **dict(self.params))


def _positive(name: str, value: float) -> float:
    if not value > 0:
        raise PriorError(f"{name} must be positive, got {value!r}")
    return float(value)


def Normal(mu: float, sigma: float) -> Prior:
    return Prior("Normal", {"mu": float(mu), "sigma": _positive("sigma", sigma)})


def StudentT(nu: float, mu: float, sigma: float) -> Prior:
    return Prior("StudentT", {"nu": _positive("nu", nu), "mu": float(mu), "sigma": _positive("sigma", sigma)})


def HalfNormal(sigma: float) -> Prior:
    return Prior("HalfNormal", {"sigma": _positive("sigma", sigma)})


def HalfStudentT(nu: float, sigma: float) -> Prior:
    return Prior("HalfStudentT", {"nu": _positive("nu", nu), "sigma": _positive("sigma", sigma)})


def Exponential(lam: float) -> Prior:
    return Prior("Exponential", {"lam": _positive("lam", lam)})


def HalfCauchy(beta: float) -> Prior:
    return Prior("HalfCauchy", {"beta": _positive("beta", beta)})


@dataclass(frozen=True)
class Priors:
    """Priors for the four M0 parameter classes.

    ``b`` is either one prior applied to every slope or a mapping from predictor name to
    prior, which must then cover every predictor exactly.
    """

    intercept: Prior
    b: Prior | Mapping[str, Prior]
    sd: Prior
    sigma: Prior | None = None

    def slope_priors(self, predictors: tuple[str, ...]) -> dict[str, Prior]:
        if isinstance(self.b, Prior):
            return {p: self.b for p in predictors}
        given = dict(self.b)
        missing = [p for p in predictors if p not in given]
        extra = [k for k in given if k not in predictors]
        if missing or extra:
            raise PriorError(
                f"slope priors must cover exactly the predictors {list(predictors)}; "
                f"missing {missing}, unexpected {extra}"
            )
        return {p: given[p] for p in predictors}

    def validate(self, predictors: tuple[str, ...], family: str = "gaussian") -> None:
        if family == "gaussian" and self.sigma is None:
            raise PriorError("the Gaussian family needs a prior on the residual SD: pass sigma=...")
        if family != "gaussian" and self.sigma is not None:
            raise PriorError(
                f"the {family} family has no residual SD; do not pass sigma. Its intercept and "
                "slope priors are on the logit scale."
            )
        for label, prior in [("intercept", self.intercept), *self.slope_priors(predictors).items()]:
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for {label!r} must be a Prior, got {type(prior).__name__}")
            if prior.name not in _LOCATION:
                raise PriorError(
                    f"prior for {label!r} must be a location prior ({sorted(_LOCATION)}), got {prior}"
                )
        scales = [("sd", self.sd)] + ([("sigma", self.sigma)] if self.sigma is not None else [])
        for label, prior in scales:
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for {label!r} must be a Prior, got {type(prior).__name__}")
            if prior.name not in _SCALE:
                raise PriorError(
                    f"prior for {label!r} must be a positive-support prior ({sorted(_SCALE)}), got {prior}"
                )

    def to_bambi(self, predictors: tuple[str, ...], group: str, family: str = "gaussian") -> dict:
        """Bambi's prior dict. The group term is Normal(0, sd) with ``sd`` as hyperprior."""
        import bambi as bmb

        self.validate(predictors, family)
        out = {"Intercept": self.intercept.to_bambi()}
        if self.sigma is not None:
            out["sigma"] = self.sigma.to_bambi()
        for p, prior in self.slope_priors(predictors).items():
            out[p] = prior.to_bambi()
        out[f"1|{group}"] = bmb.Prior("Normal", mu=0.0, sigma=self.sd.to_bambi())
        return out
