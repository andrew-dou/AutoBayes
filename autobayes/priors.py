"""Explicit priors, by parameter class, in brms terms.

brms classes: ``Intercept``, ``b`` (slopes and dummy coefficients), ``sd`` (one group SD per group-level
coefficient: the varying intercept of ``(1 | g)``, the intercept and each slope of ``(1 + x || g)``) and, for
Gaussian/Student-t, ``sigma`` (residual SD/scale respectively). Every class the model has
must be given, and a class the model does not have is refused (M3c-1: ``sd`` without a group term); there are no
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
    the actual coefficient levels (all levels for the first categorical main effect
    without an intercept; non-reference levels otherwise). Omit intercept exactly
    when the formula removes it. ``sd`` is keyed by grouping factor (an exact-key mapping; a single
    intercept-only factor also accepts a bare Prior): an intercept-only factor takes one Prior (or the
    one-entry mapping ``{"Intercept": prior}``), a factor with varying slopes a mapping over exactly its
    coefficients, ``{"g": {"Intercept": HalfNormal(1), "x": HalfNormal(0.5)}}`` (brms ``class = sd, group =
    g, coef = x``). No automatic priors are filled in silently and nothing is broadcast.
    """

    intercept: Prior | None = None
    b: Prior | Mapping[str, Prior | Mapping[str, Prior]] | None = None
    sd: Prior | Mapping[str, Prior | Mapping[str, Prior]] | None = None   # required with a group term, refused without one (M3c-1)
    sigma: Prior | None = None

    # -- coefficient priors -----------------------------------------------------------------
    def coefficient_priors(self, spec) -> dict[str, Prior]:
        """One prior per design column, keyed by column label, in design order. ``spec`` is a
        ``ModelSpec`` (or a tuple of numeric predictor names); categorical levels must be
        known (data seen) for per-level mappings to be checked."""
        from autobayes.formula import ModelSpec

        if not isinstance(spec, ModelSpec):
            spec = ModelSpec(formula="", response="", predictors=tuple(spec), group=None)
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
        for t in spec.pop_terms:
            term, entry = t.name, given[t.name]
            if not t.is_vector:
                if not isinstance(entry, Prior):
                    raise PriorError(
                        f"predictor {term!r} takes one prior, got {type(entry).__name__}; a mapping {{level: prior}} "
                        f"is only for a categorical term, declared with categorical={{{term!r}: <reference level>}}"
                    )
                out[term] = entry
                continue
            cat = spec.categorical(t.operands[-1])            # the categorical side of a categorical or interaction term
            if isinstance(entry, Prior):
                for col in t.columns:
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
            wanted = list(t.levels)
            missing_l = [l for l in wanted if l not in keyed]
            extra_l = [l for l in keyed if l not in wanted]
            if missing_l or extra_l:
                ref_note = (f" ({cat.reference!r} is the reference level and has no coefficient)"
                            if cat.reference in extra_l and cat.reference not in wanted else "")
                raise PriorError(
                    f"per-level priors for {term!r} must cover exactly the {'coefficient levels' if cat.reference in wanted else 'non-reference levels'} {wanted}; "
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

    def sd_priors(self, spec) -> dict[tuple[str, str], Prior]:
        """One SD prior per group-level coefficient, keyed ``(factor, coefficient)`` in formula order (the order
        of ``Design.group_blocks``). Coverage is checked exactly, by factor and then by coefficient; the
        values are checked to be positive-support priors by ``validate``. No scalar is broadcast."""
        group_terms = getattr(spec, "group_terms", ())
        groups = [t.factor for t in group_terms]
        if not group_terms:
            if self.sd is not None:
                raise PriorError("the formula has no group-level term '(1 | group)'; do not pass sd")
            return {}
        if self.sd is None:
            if len(groups) == 1:
                t = group_terms[0]
                if t.slopes:
                    raise PriorError(
                        f"the group term '{t.canonical}' needs a prior on the SD of each of its coefficients "
                        f"{list(t.coefs)}: pass sd={{{t.factor!r}: {{{', '.join(f'{c!r}: <prior>' for c in t.coefs)}}}}}"
                    )
                raise PriorError(f"the varying intercept '(1 | {groups[0]})' needs a prior on the SD of its varying intercept: pass sd=...")
            raise PriorError(f"each varying intercept needs a prior on the SD: pass sd for {list(groups)}")
        if isinstance(self.sd, Mapping):
            if set(self.sd) != set(groups):
                hint = ""
                if len(group_terms) == 1 and set(self.sd) <= set(group_terms[0].coefs):
                    t = group_terms[0]
                    hint = (f"; {list(self.sd)} look like the coefficients of {t.factor!r}: nest them under the factor, "
                            f"sd={{{t.factor!r}: {{{', '.join(f'{c!r}: <prior>' for c in t.coefs)}}}}}")
                raise PriorError(f"sd must cover exactly the grouping factors {list(groups)}; got {list(self.sd)}{hint}")
            by_factor = {g: self.sd[g] for g in groups}
        else:
            if len(groups) != 1:
                raise PriorError("multiple grouping factors require an exact-key sd mapping; scalar sd is not broadcast")
            by_factor = {groups[0]: self.sd}
        out: dict[tuple[str, str], Prior] = {}
        for t in group_terms:
            entry = by_factor[t.factor]
            if isinstance(entry, Mapping):
                missing = [c for c in t.coefs if c not in entry]
                extra = [k for k in entry if k not in t.coefs]
                if missing or extra:
                    hint = ""
                    for k in extra:
                        if isinstance(k, str) and k in getattr(spec, "predictors", ()):
                            hint = (f" ({k!r} is a population-level term, but '{t.canonical}' has no varying slope on it; "
                                    f"write '(1 + {k} || {t.factor})' to add one)")
                            break
                    raise PriorError(
                        f"SD priors for {t.factor!r} must cover exactly its coefficients {list(t.coefs)}; "
                        f"missing {missing}, unexpected {extra}{hint}"
                    )
                for c in t.coefs:
                    out[(t.factor, c)] = entry[c]
            elif t.slopes:
                raise PriorError(
                    f"sd for {t.factor!r} must be a mapping over its coefficients {list(t.coefs)} ('{t.canonical}' has "
                    f"varying slopes), got a single {type(entry).__name__}; the same prior for every coefficient must be spelled out"
                )
            else:
                out[(t.factor, "Intercept")] = entry
        return out

    def group_priors(self, spec) -> dict[str, Prior]:
        """The SD prior of each factor's varying intercept, in formula order (the per-factor view that
        ``describe()``, fit records and the target identity have always used); ``sd_priors`` has every
        group-level coefficient, slopes included."""
        return {g: prior for (g, coef), prior in self.sd_priors(spec).items() if coef == "Intercept"}

    def validate(self, spec, family: str = "gaussian") -> None:
        from autobayes.families import get_family

        fam = get_family(family)
        has_intercept = getattr(spec, "intercept", True)
        if has_intercept and self.intercept is None:
            raise PriorError("the formula has a population intercept; an intercept prior is required")
        if not has_intercept and self.intercept is not None:
            raise PriorError("the formula has no population intercept; omit the intercept prior (or pass None)")
        sd_priors = self.sd_priors(spec)
        if fam.scale_parameter is not None and self.sigma is None:
            raise PriorError(fam.sigma_missing_error)
        if fam.scale_parameter is None and self.sigma is not None:
            raise PriorError(
                f"the {fam.name} family has no residual SD; do not pass sigma. Its intercept and "
                f"slope priors are on the {fam.link_scale} scale."
            )
        for label, prior in ([("intercept", self.intercept)] if has_intercept else []) + list(self.coefficient_priors(spec).items()):
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for {label!r} must be a Prior, got {type(prior).__name__}")
            if prior.name not in _LOCATION:
                raise PriorError(
                    f"prior for {label!r} must be a location prior ({sorted(_LOCATION)}), got {prior}"
                )
        from autobayes.formula import sd_tag

        scales = [(f"sd:{sd_tag(g, c)}", prior) for (g, c), prior in sd_priors.items()] + ([("sigma", self.sigma)] if self.sigma is not None else [])
        for label, prior in scales:
            if not isinstance(prior, Prior):
                raise PriorError(f"prior for {label!r} must be a Prior, got {type(prior).__name__}")
            if prior.name not in _SCALE:
                raise PriorError(
                    f"prior for {label!r} must be a positive-support prior ({sorted(_SCALE)}), got {prior}"
                )

    def to_bambi(self, spec, family: str = "gaussian") -> dict:
        """Bambi's prior dict: one prior per term (array-valued parameters for the levels of a
        categorical term). Each group-level coefficient (``1|g``, and ``x|g`` for a varying slope) gets
        Normal(0, sd) with its own ``sd`` as hyperprior; without a group term no group prior is emitted
        (Bambi would accept a stray ``1|g`` entry silently)."""
        import bambi as bmb
        import numpy as np
        from autobayes.formula import ModelSpec, bambi_group_name

        self.validate(spec, family)
        if not isinstance(spec, ModelSpec):
            spec = ModelSpec(formula="", response="", predictors=tuple(spec), group=None)
        per_column = self.coefficient_priors(spec)
        out = {"Intercept": self.intercept.to_bambi()} if spec.intercept else {}
        if self.sigma is not None:
            out["sigma"] = self.sigma.to_bambi()
        for term in spec.pop_terms:
            if not term.is_vector:
                out[term.name] = per_column[term.name].to_bambi()
                continue
            priors = [per_column[col] for col in term.columns]
            name = priors[0].name
            params = {k: np.array([float(pr.params[k]) for pr in priors]) for k in priors[0].params}
            out[term.name] = bmb.Prior(name, **params)
        for (group, coef), prior in self.sd_priors(spec).items():
            out[bambi_group_name(group, coef)] = bmb.Prior("Normal", mu=0.0, sigma=prior.to_bambi())
        return out
