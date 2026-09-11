"""Explicit priors, by parameter class, in brms terms.

brms classes: ``Intercept``, ``b`` (slopes and dummy coefficients), ``sd`` (group SDs: one prior, or one per varying effect) and, for the Gaussian family
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
    """Priors for the four parameter classes.

    ``b`` is either one prior applied to every coefficient (numeric slopes and the dummy
    coefficients of categorical predictors) or a mapping from formula term to prior, which
    must cover every predictor term exactly. For a categorical term the entry is one prior
    (the same for every non-reference level) or a mapping ``{level: prior}`` covering exactly
    the non-reference levels (docs/m3a1-decision.md); nothing is filled in silently.
    """

    intercept: Prior
    b: Prior | Mapping[str, Prior | Mapping[str, Prior]]
    sd: Prior | Mapping[str, Prior]
    sigma: Prior | None = None

    # -- group-effect SD priors ---------------------------------------------------------------
    def sd_priors(self, spec) -> dict[str, Prior]:
        """One SD prior per varying effect, keyed ``"Intercept"`` and, with ``(1 + x || g)``, the slope
        name. A single prior is the varying-intercept-only form; a varying slope requires the mapping
        to name both effects exactly (docs/m3b1-decision.md)."""
        from autobayes.formula import ModelSpec

        slope = spec.varying_slope if isinstance(spec, ModelSpec) else None
        wanted = ["Intercept"] + ([slope] if slope else [])
        if isinstance(self.sd, Prior):
            if slope:
                raise PriorError(
                    f"a model with the varying slope {slope!r} needs one SD prior per varying effect: "
                    f"sd={{'Intercept': <prior>, {slope!r}: <prior>}} (nothing is filled in)"
                )
            return {"Intercept": self.sd}
        if not isinstance(self.sd, Mapping):
            raise PriorError(f"sd must be a Prior or a mapping {{'Intercept': prior[, 'x': prior]}}, got {type(self.sd).__name__}")
        given = dict(self.sd)
        missing = [k for k in wanted if k not in given]
        extra = [k for k in given if k not in wanted]
        if missing or extra:
            raise PriorError(
                f"sd priors must cover exactly the varying effects {wanted}; missing {missing}, unexpected {extra}"
            )
        return {k: given[k] for k in wanted}

    # -- coefficient priors -----------------------------------------------------------------
    def coefficient_priors(self, spec) -> dict[str, Prior]:
        """One prior per design column, keyed by column label, in design order. ``spec`` is a
        ``ModelSpec`` (or a tuple of numeric predictor names); categorical levels must be
        known (data seen) for per-level mappings to be checked."""
        from autobayes.formula import ModelSpec

        if not isinstance(spec, ModelSpec):
            spec = ModelSpec(formula="", response="", predictors=tuple(spec), group="")
        cat_names = spec.categorical_names
        if isinstance(self.b, Prior):
            return {col: self.b for col in spec.columns}
        if not isinstance(self.b, Mapping):
            raise PriorError(f"b must be a Prior or a mapping from term to prior, got {type(self.b).__name__}")
        given = dict(self.b)
        missing = [t for t in spec.terms if t not in given]
        extra = [k for k in given if k not in spec.terms]
        if missing or extra:
            hint = ""
            for k in extra:
                if isinstance(k, str) and "[" in k and k.split("[", 1)[0] in (*cat_names, *spec.interaction_names):
                    hint = (f"; {k!r} looks like a design-column label: give per-level priors as "
                            f"b={{{k.split('[', 1)[0]!r}: {{level: prior}}}}")
                    break
                if isinstance(k, str) and ":" in k and ":".join(reversed(k.split(":", 1))) in spec.interaction_names:
                    hint = f"; write the interaction key in its canonical spelling {':'.join(reversed(k.split(':', 1)))!r} (numeric first)"
                    break
            raise PriorError(
                f"coefficient priors must cover exactly the predictor terms {list(spec.terms)}; "
                f"missing {missing}, unexpected {extra}{hint}"
            )
        out: dict[str, Prior] = {}
        for term in spec.terms:
            entry = given[term]
            if not spec.is_vector_term(term):
                if not isinstance(entry, Prior):
                    raise PriorError(
                        f"predictor {term!r} takes one prior, got {type(entry).__name__}; a mapping {{level: prior}} "
                        f"is only for a categorical term, declared with categorical={{{term!r}: <reference level>}}"
                    )
                out[term] = entry
                continue
            cat = spec.categorical(term if term in cat_names else spec.interaction_of(term)[1])
            if isinstance(entry, Prior):
                for col in spec.term_columns(term):
                    out[col] = entry
                continue
            if not isinstance(entry, Mapping):
                raise PriorError(
                    f"categorical predictor {term!r} takes one prior or a mapping {{level: prior}}, "
                    f"got {type(entry).__name__}"
                )
            keyed: dict[str, Prior] = {}
            for k, v in entry.items():
                try:
                    from autobayes.formula import FormulaError, _label
                    lab = _label(k)               # the same scalar rule as the data (no float intermediate)
                except FormulaError as e:         # bool / complex / fractional / large-float keys
                    raise PriorError(f"prior key {k!r} for {term!r} is not a level label: {e}") from None
                if lab in keyed:
                    raise PriorError(f"ambiguous prior keys for {term!r}: more than one key maps to level {lab!r}")
                keyed[lab] = v
            if not cat.levels:                     # structure only; coverage needs the data
                for lab, v in keyed.items():
                    if not isinstance(v, Prior):
                        raise PriorError(f"prior for {term!r} level {lab!r} must be a Prior, got {type(v).__name__}")
                continue
            wanted = list(cat.coefficients)
            missing_l = [l for l in wanted if l not in keyed]
            extra_l = [l for l in keyed if l not in wanted]
            if missing_l or extra_l:
                ref_note = (f" ({cat.reference!r} is the reference level and has no coefficient)"
                            if cat.reference in extra_l else "")
                raise PriorError(
                    f"per-level priors for {term!r} must cover exactly the non-reference levels {wanted}; "
                    f"missing {missing_l}, unexpected {extra_l}{ref_note}"
                )
            names = {keyed[l].name for l in wanted if isinstance(keyed[l], Prior)}
            if len(names) > 1:
                raise PriorError(
                    f"per-level priors of {term!r} must share one distribution family (Bambi takes one "
                    f"prior per term, with per-level parameters); got {sorted(names)}"
                )
            for l in wanted:
                out[f"{term}[{l}]"] = keyed[l]
        return out

    def slope_priors(self, predictors: tuple[str, ...]) -> dict[str, Prior]:
        """Numeric-only compatibility form of ``coefficient_priors``."""
        return self.coefficient_priors(tuple(predictors))

    def validate(self, spec, family: str = "gaussian") -> None:
        if family == "gaussian" and self.sigma is None:
            raise PriorError("the Gaussian family needs a prior on the residual SD: pass sigma=...")
        if family != "gaussian" and self.sigma is not None:
            raise PriorError(
                f"the {family} family has no residual SD; do not pass sigma. Its intercept and "
                "slope priors are on the logit scale."
            )
        for label, prior in [("intercept", self.intercept), *self.coefficient_priors(spec).items()]:
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for {label!r} must be a Prior, got {type(prior).__name__}")
            if prior.name not in _LOCATION:
                raise PriorError(
                    f"prior for {label!r} must be a location prior ({sorted(_LOCATION)}), got {prior}"
                )
        scales = [(f"sd:{k}", v) for k, v in self.sd_priors(spec).items()] + ([("sigma", self.sigma)] if self.sigma is not None else [])
        for label, prior in scales:
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for {label!r} must be a Prior, got {type(prior).__name__}")
            if prior.name not in _SCALE:
                raise PriorError(
                    f"prior for {label!r} must be a positive-support prior ({sorted(_SCALE)}), got {prior}"
                )

    def to_bambi(self, spec, group: str, family: str = "gaussian") -> dict:
        """Bambi's prior dict: one prior per term (array-valued parameters for the levels of a
        categorical term). The group term is Normal(0, sd) with ``sd`` as hyperprior."""
        import bambi as bmb
        import numpy as np
        from autobayes.formula import ModelSpec

        self.validate(spec, family)
        if not isinstance(spec, ModelSpec):
            spec = ModelSpec(formula="", response="", predictors=tuple(spec), group="")
        per_column = self.coefficient_priors(spec)
        out = {"Intercept": self.intercept.to_bambi()}
        if self.sigma is not None:
            out["sigma"] = self.sigma.to_bambi()
        for term in spec.terms:
            if not spec.is_vector_term(term):
                out[term] = per_column[term].to_bambi()
                continue
            priors = [per_column[col] for col in spec.term_columns(term)]
            name = priors[0].name
            params = {k: np.array([float(pr.params[k]) for pr in priors]) for k in priors[0].params}
            out[term] = bmb.Prior(name, **params)
        sds = self.sd_priors(spec)
        out[f"1|{group}"] = bmb.Prior("Normal", mu=0.0, sigma=sds["Intercept"].to_bambi())
        if spec.varying_slope:
            out[f"{spec.varying_slope}|{group}"] = bmb.Prior("Normal", mu=0.0, sigma=sds[spec.varying_slope].to_bambi())
        return out
