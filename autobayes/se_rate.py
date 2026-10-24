"""brms's response addition terms ``se()`` and ``rate()`` (F06a, docs/se-rate.md).

``se(col)``                known standard errors, a meta-analysis (Gaussian and Student-t families): brms's Stan data holds
                           ``se`` (the column) and ``sigma = 0``. There is no sigma parameter, no sigma prior and no sigma row
                           in brms's ``default_prior()``, and row ``i`` has the scale ``se_i``.
``se(col, sigma = TRUE)``  ``sigma`` is a parameter as usual and row ``i`` has the scale ``sqrt(sigma^2 + se_i^2)``; brms's Stan
                           data holds ``se`` and no ``sigma``. ``se(col, TRUE)`` is the same term.
``rate(col)``              the denominator of a rate (Poisson and negative binomial families, log link): brms's Stan data holds
                           ``denom``. The mean of row ``i`` is ``exp(eta_i) denom_i``, which brms writes on the log scale as
                           ``eta_i + log(denom_i)``, and the negative binomial's shape is ``shape * denom_i``.
                           ``posterior_linpred`` is ``eta_i``, without the denominator, and ``posterior_epred`` is
                           ``exp(eta_i) denom_i``, on new rows with their own denominators.

The term model holds them as ``formula.AdditionTerm("se", column, sigma=...)`` and ``AdditionTerm("rate", column)``.
``formula.addition_data`` gives brms's ``se`` and ``denom`` on a frame, validated here (``se_values``, ``rate_values``), and
``Design.se`` and ``Design.denom`` hold them on the fitted rows. The likelihood is the family's own distribution with these
parameters (``likelihood_parameters``, which ``likelihoods.addition_family`` applies to the family's parameters by Bambi's
names, ``FamilySpec.se_parameter`` and ``FamilySpec.rate_parameters``). Under ``rate()`` the graph's linear predictor gains
``log(denom_i)`` before the link (``log_denominator_hook``), so that Bambi's parent ``mu`` is the expected count, brms's
posterior_epred. Without ``sigma = TRUE`` the model's family spec is ``se_only_spec(fam)``: the family without its scale
parameter, whose Bambi likelihood has no ``sigma`` (``family_for`` chooses it from the parsed formula).

Refused, naming the construct (divergence D157): ``se()`` or ``rate()`` next to another addition term (``weights()``,
``cens()``, ``trunc()``, ``mi()``), with a distributional formula, an expression inside the call (compute the column), and
``rate()`` under the Poisson's sqrt and the negative binomial's identity links (the link's own refusal, D133); the families that
do not take them (``FamilySpec.addition_terms``). Multivariate, non-linear, mixture, autocorrelation, SAR and shrinkage models
refuse every addition term already (D56, D61, D55, D70, D143, D102). A zero standard error without ``sigma = TRUE`` is refused
(D156: brms accepts it, and its likelihood is undefined there).
"""
from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pandas as pd

from autobayes.formula import FormulaError, _is_numeric, reject_complex

KINDS = ("se", "rate")              # the addition terms of this module, formula.AdditionTerm's kinds


# -- the data -------------------------------------------------------------------------------------------------------------
def _numeric(column: str, s: pd.Series, where: str, term: str, what: str) -> np.ndarray:
    reject_complex(f"{where} {column!r}", s)
    if not _is_numeric(s):
        raise FormulaError(f"{where} {column!r} has dtype {s.dtype}; the {what} of {term!r} must be numeric")
    v = s.to_numpy(dtype=float)
    if not np.all(np.isfinite(v)):
        raise FormulaError(f"{where} {column!r} of {term!r} has missing or non-finite values")
    return v


def se_values(column: str, s: pd.Series, where: str, *, sigma: bool) -> np.ndarray:
    """brms's ``se`` of ``se(col)`` or ``se(col, sigma = TRUE)``: finite and non-negative, as brms requires ('Standard errors
    must be non-negative'); without ``sigma = TRUE`` also positive, since the row's scale is then ``se_i`` itself (brms accepts
    a zero there and its likelihood ``normal_lpdf(y | mu, 0)`` is undefined; divergence D156)."""
    term = f"se({column}, sigma = TRUE)" if sigma else f"se({column})"
    v = _numeric(column, s, where, term, "standard errors")
    bad = np.flatnonzero(v < 0) if sigma else np.flatnonzero(v <= 0)
    if bad.size:
        i = int(bad[0])
        if v[i] < 0:
            raise FormulaError(f"{where} {column!r}: the standard errors of {term!r} must be non-negative (brms: 'Standard errors must "
                               f"be non-negative'); row {i} has {float(v[i])!r}")
        raise FormulaError(f"{where} {column!r}: the standard errors of {term!r} must be positive: without sigma = TRUE row {i}'s scale "
                           f"is its standard error, 0 here, where the likelihood is undefined (brms accepts the row; divergence D156); "
                           f"write se({column}, sigma = TRUE) or drop the row")
    return v


def rate_values(column: str, s: pd.Series, where: str) -> np.ndarray:
    """brms's ``denom`` of ``rate(col)``: finite and positive, as brms requires (it refuses a zero or negative denominator)."""
    v = _numeric(column, s, where, f"rate({column})", "denominators")
    bad = np.flatnonzero(v <= 0)
    if bad.size:
        i = int(bad[0])
        raise FormulaError(f"{where} {column!r}: the denominators of 'rate({column})' must be positive (brms refuses a zero or negative "
                           f"one too); row {i} has {float(v[i])!r}")
    return v


# -- the grammar ----------------------------------------------------------------------------------------------------------
_TRUE, _FALSE = ("TRUE", "T"), ("FALSE", "F")


def parse_se(args: list[str], piece: str, formula: str) -> bool:
    """Whether ``se(col, ...)``'s arguments after the column ask for ``sigma = TRUE``: none, ``sigma = TRUE`` / ``FALSE`` or
    brms's positional ``TRUE`` / ``FALSE`` (R's ``T`` and ``F`` too); anything else is refused."""
    if len(args) == 1:
        return False
    m = re.fullmatch(r"(?:sigma\s*=\s*)?(\w+)", args[1]) if len(args) == 2 else None
    if m is None or m.group(1) not in (*_TRUE, *_FALSE):
        raise FormulaError(f"{formula!r}: {piece!r}: se() takes the column of the standard errors and, to learn sigma as well, "
                           "sigma = TRUE: se(col) or se(col, sigma = TRUE) (docs/se-rate.md)")
    return m.group(1) in _TRUE


def check_additions(additions, formula: str) -> None:
    """Refuse ``se()`` or ``rate()`` next to another addition term: no frozen brms case verifies the pair (brms accepts
    ``se(s) + weights(w)``, for one; divergence D157)."""
    ours = [a for a in additions if a.kind in KINDS]
    if ours and len(additions) > 1:
        other = next(a for a in additions if a is not ours[0])
        raise FormulaError(f"{formula!r}: {ours[0].term!r} together with {other.term!r} is not supported in this slice: no frozen brms "
                           "case verifies the pair; se() and rate() are offered alone on the left-hand side (docs/se-rate.md, "
                           "divergence D157)")


def check_distributional(additions, dpar_formula: str, formula: str) -> None:
    """Refuse ``se()`` or ``rate()`` (among ``additions``) in a model with the distributional formula ``dpar_formula`` (brms refuses
    ``sigma ~ ...`` next to ``se(col)`` without ``sigma = TRUE`` too, 'Cannot predict or fix 'sigma' in this model'; divergence
    D157)."""
    ours = [a for a in additions if a.kind in KINDS]
    if ours:
        raise FormulaError(f"{formula!r}: {ours[0].term!r} with the distributional formula {dpar_formula!r} is not supported in this "
                           "slice: no frozen brms case verifies a formula on a parameter next to it (docs/se-rate.md, divergence D157)")


# -- the family -----------------------------------------------------------------------------------------------------------
def family_for(spec, fam):
    """The family spec of a model of ``spec``: ``se_only_spec(fam)`` under ``se(col)`` without ``sigma = TRUE`` (brms fixes
    sigma at 0), ``fam`` otherwise."""
    se = spec.addition("se")
    return se_only_spec(fam) if se is not None and not se.sigma else fam


def check_priors(spec, priors) -> None:
    """Refuse a sigma prior under ``se(col)`` without ``sigma = TRUE``: brms fixes sigma at 0 there, so the model has none (and
    brms's default_prior() no sigma row)."""
    from autobayes.priors import PriorError

    se = spec.addition("se")
    if se is not None and not se.sigma and getattr(priors, "sigma", None) is not None:
        raise PriorError(f"{se.term!r} without sigma = TRUE fixes sigma at 0 (brms's Stan data sigma = 0; each row's scale is its standard "
                         f"error): do not pass sigma; write se({se.column}, sigma = TRUE) to learn sigma as well (docs/se-rate.md)")


def _without_scale_family(base: str, scale: str):
    """Bambi's built-in family ``base`` (the Gaussian's ``"gaussian"``, the Student-t's ``"t"``) without its parameter ``scale``:
    Bambi's likelihood then has no such parameter, and ``likelihood_parameters`` gives the distribution its value per row."""
    import bambi as bmb
    from bambi.backend.utils import get_distribution_from_likelihood
    from bambi.defaults import get_builtin_family

    family = get_builtin_family(base)
    lk = family.likelihood
    params = [p for p in lk.params if p != scale]
    likelihood = bmb.Likelihood(f"{lk.name}KnownSE", params=params, parent=lk.parent, dist=get_distribution_from_likelihood(lk))
    return bmb.Family(f"{family.name}_se", likelihood, {p: family.link[p].name for p in params})


@lru_cache(maxsize=None)
def se_only_spec(fam):
    """``fam`` (the Gaussian or Student-t family) under ``se(col)`` without ``sigma = TRUE``: brms fixes sigma at 0, so the
    model has no scale parameter (no prior, no value, no draws, no row of the default-prior table) and each row's scale is its
    standard error. The likelihood, its draws and its pointwise log density are the family's own (each reads the row's scale
    through ``row_scale``)."""
    scale = fam.bambi_scale_parameter
    return replace(
        fam, scale_parameter=None, bambi_scale_parameter=None,
        bambi_family=lambda: _without_scale_family(fam.bambi_family(), scale),
        scale_rule=(f"sigma does not apply: se(col) without sigma = TRUE fixes the {fam.name} family's {scale} at 0 (brms's Stan data "
                    "sigma = 0; each row's scale is its standard error); write se(col, sigma = TRUE) to learn sigma as well"),
        scale_missing_error=None, default_scale_prior=None,
        distributional_link=None, default_distributional_intercept=None,   # no sigma ~ ... (refused, D157)
    )


def has_scale(model) -> bool:
    """Whether the model has the family's scale parameter (False under ``se(col)`` without ``sigma = TRUE``)."""
    return model.family_spec.scale_parameter is not None


def row_scale(model, sigma, se):
    """The scale of each row: ``sigma`` without ``se()``, ``se`` under ``se(col)`` and ``sqrt(sigma^2 + se^2)`` under
    ``se(col, sigma = TRUE)``; ``sigma`` broadcasts against the rows (NumPy, for draws and ``log_lik()``)."""
    if se is None:
        return sigma
    if not has_scale(model):
        return se
    return np.sqrt(np.square(sigma) + np.square(se))


def scale_text(model, sigma: str) -> str:
    """How ``describe()`` writes each row's scale: ``sigma`` (as given), ``se_i`` or ``sqrt(sigma^2 + se_i^2)``."""
    se = model.spec.addition("se")
    if se is None:
        return sigma
    return f"{se.column}_i" if not has_scale(model) else f"sqrt({sigma}^2 + {se.column}_i^2)"


def likelihood_parameters(kw: dict, data, fam) -> dict:
    """The parameters of the family's distribution under ``se()`` and ``rate()``, from Bambi's (``kw``, by Bambi's names) and the
    terms' data on the fitted rows (``formula.addition_data``): ``se`` replaces the scale ``FamilySpec.se_parameter`` (no such
    parameter under ``se(col)``) or adds to it in quadrature (``se(col, sigma = TRUE)``); ``denom`` multiplies each of
    ``FamilySpec.rate_parameters`` (the negative binomial's shape; the mean has the denominator in the linear predictor
    already, ``log_denominator_hook``). Other kinds' data change nothing here."""
    import pytensor.tensor as pt

    se, denom = data.get("se"), data.get("denom")
    if se is None and denom is None:
        return kw
    out = dict(kw)
    if se is not None:
        name = fam.se_parameter
        out[name] = pt.as_tensor_variable(se) if name not in kw else pt.sqrt(pt.sqr(kw[name]) + np.square(se))
    for name in (fam.rate_parameters if denom is not None else ()):
        out[name] = kw[name] * denom
    return out


def log_denominator_hook(log_denom: np.ndarray):
    """``rate()``'s part of the linear predictor, for ``likelihoods.monotonic_family``: ``eta + log(denom)`` (brms's
    ``mu + log_denom``)."""
    def add(eta):
        return eta + log_denom

    return add


# -- predictions ----------------------------------------------------------------------------------------------------------
def newdata_columns(model, kind: str) -> list[str]:
    """The columns ``predict()`` needs on new rows for these terms: ``rate()``'s for ``kind="mean"`` and ``"response"`` (the
    expected count is ``exp(eta) denom``), ``se()``'s for ``kind="response"`` (each new row's scale); none for ``"linear"``."""
    out = []
    rate, se = model.spec.addition("rate"), model.spec.addition("se")
    if rate is not None and kind in ("mean", "response"):
        out.append(rate.column)
    if se is not None and kind == "response":
        out.append(se.column)
    return out


def check_newdata(model, newdata: pd.DataFrame, kind: str) -> None:
    """Refuse new rows without the columns ``newdata_columns`` names, or with values ``se_values`` or ``rate_values`` refuses."""
    need = newdata_columns(model, kind)
    missing = [c for c in need if c not in newdata.columns]
    if missing:
        term = next(a.term for a in model.spec.additions if a.column == missing[0])
        why = "each new row's expected count is exp(eta) times its denominator" if term.startswith("rate") else \
            "each new row's response is drawn with its own standard error"
        raise FormulaError(f"newdata lacks the column {missing[0]!r} of {term!r}, which kind={kind!r} needs: {why} (kind='linear' "
                           "needs neither)")
    for a in model.spec.additions:
        if a.column in need:
            if a.kind == "rate":
                rate_values(a.column, newdata[a.column], "newdata column")
            else:
                se_values(a.column, newdata[a.column], "newdata column", sigma=a.sigma)


def denominators(model, df: pd.DataFrame | None):
    """Each row's denominator under ``rate()`` (the fitted rows' for ``df`` None), or None without ``rate()``."""
    rate = model.spec.addition("rate")
    if rate is None:
        return None
    if df is None:
        return model.design.denom
    return rate_values(rate.column, df[rate.column], "newdata column")


def standard_errors(model, df: pd.DataFrame | None):
    """Each row's standard error under ``se()`` (the fitted rows' for ``df`` None), or None without ``se()``."""
    se = model.spec.addition("se")
    if se is None:
        return None
    if df is None:
        return model.design.se
    return se_values(se.column, df[se.column], "newdata column", sigma=se.sigma)


def rate_mean(model, mean, df: pd.DataFrame | None):
    """``predict(kind="mean")``'s expected response under ``rate()``: the family's mean times each row's denominator (brms's
    posterior_epred); the mean itself otherwise."""
    d = denominators(model, df)
    return mean if d is None else mean * d


# -- describe() -----------------------------------------------------------------------------------------------------------
def data_lines(model, a) -> list[str]:
    """``describe()``'s data line of ``se()`` or ``rate()``."""
    d = model.design
    if a.kind == "rate":
        return [f"  rate      : {a.term}: each row's denominator {a.column}_i, {d.denom.min():.4f} to {d.denom.max():.4f} (brms denom): "
                f"the mean is exp(eta_i) * {a.column}_i, eta_i + log({a.column}_i) on the log scale (the graph's "
                f"{model.names['mean']} holds it)"
                + (f", and the shape is shape * {a.column}_i" if model.family_spec.rate_parameters else "")
                + "; predict(kind='linear') is eta_i without the denominator (brms's posterior_linpred), kind='mean' "
                f"exp(eta_i) * {a.column}_i (brms's posterior_epred), new rows with their own {a.column}"]
    scale = (f"sqrt(sigma^2 + {a.column}_i^2)" if a.sigma else f"{a.column}_i itself, and the model has no sigma (brms fixes it at 0)")
    return [f"  se        : {a.term}: each row's known standard error {a.column}_i, {d.se.min():.4f} to {d.se.max():.4f} (brms se); "
            f"row i's scale is {scale}; predict(kind='mean') is mu_i (brms's posterior_epred), and kind='response' draws with each "
            f"row's scale, new rows with their own {a.column}"]


def likelihood_lines(model, a) -> list[str]:
    """The likelihood paragraph's line of ``se()`` or ``rate()``."""
    if a.kind == "rate":
        return [f"  rate(): the log likelihood is the family's at the mean exp(eta_i) * {a.column}_i (brms's mu + log_denom)"
                + (f" and the shape shape * {a.column}_i" if model.family_spec.rate_parameters else "")]
    if a.sigma:
        return [f"  se(): row i's scale is sigma and its known standard error {a.column}_i in quadrature, "
                f"{scale_text(model, 'sigma')} (brms's sqrt(square(sigma) + se2))"]
    return [f"  se(): row i's scale is its known standard error {a.column}_i; the model has no sigma (brms's Stan data sigma = 0)"]
