"""brms's simultaneous autoregressive term ``sar(M, type = "lag")`` and ``sar(M, type = "error")`` (F02b, docs/spatial-sar.md).

``M`` is an ``N x N`` spatial weight matrix over the fitted rows, in data order, handed to the constructor in ``data2`` under
the name the formula gives it (brms's ``data2 = list(M = ...)``). With ``mu`` the linear predictor (brms's
``posterior_linpred``), ``A = I - rho M`` and ``rho`` brms's ``lagsar`` or ``errorsar``:

    lag:    A y = mu + e,        e ~ Normal(0, sigma^2 I),   so y ~ Normal(A^-1 mu, sigma^2 A^-1 A^-T)
    error:  A (y - mu) = e,      e ~ Normal(0, sigma^2 I),   so y ~ Normal(mu, sigma^2 A^-1 A^-T)

and the joint log density is ``log |A| + sum_i log Normal(r_i | 0, sigma)`` with the residual ``r = A y - mu`` (lag) or
``r = A (y - mu)`` (error). ``log |A| = sum_k log(1 - rho lambda_k)`` over the eigenvalues ``lambda_k`` of ``M`` (brms's
``eigenMsar``), and ``rho`` is bounded by the smallest and largest of them (brms's ``real<lower=min_eigenMsar,
upper=max_eigenMsar>``), flat by default. brms 2.23 leaves the Gaussian constant ``-N/2 log(2 pi)`` out of both densities
(``normal_lagsar`` and ``normal_errorsar``: measured on the frozen cases, divergence D142); the model's log density leaves it
out as brms's does, while ``log_lik()`` keeps it, as brms's does.

``log_lik()`` of these non-factorised models is each row's density given every other row (brms's ``log_lik()``): with the
precision ``Q = A^T A / sigma^2`` and ``z = Q (y - E[y]) = A^T r / sigma^2``, ``y_i | y_-i ~ Normal(y_i - z_i / Q_ii,
1 / Q_ii)``, so ``log_lik_i = -log(2 pi) / 2 + log(Q_ii) / 2 - z_i^2 / (2 Q_ii)`` (Buerkner, Gabry and Vehtari 2021,
"Efficient leave-one-out cross-validation for Bayesian non-factorized normal and Student-t models", Computational
Statistics 36). The expected response is ``A^-1 mu`` for the lag model and ``mu`` for the error model (brms's
``posterior_epred`` fails for the latter, divergence D142).
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autobayes.formula import _IDENT, FormulaError, _split_top_level, data2_mapping

SAR_PARAMETERS = {"lag": "lagsar", "error": "errorsar"}       # brms's parameter (and prior class) of each type
BOUNDS = ("min_eigenMsar", "max_eigenMsar")                   # brms's names of the bounds, as its prior table writes them
_CALL_RE = re.compile(r"^sar\s*\((.*)\)$", re.S)
_STRING_RE = re.compile(r'"([^"]*)"|\'([^\']*)\'')
_FORMALS = ("M", "type")
REAL_TOLERANCE = 1e-10           # the largest |imaginary part| of an eigenvalue of a non-symmetric M taken as rounding, relative
SPECTRUM_TOLERANCE = 1e-8        # how far past 1 the largest |eigenvalue| of M may lie (rounding of a normalised matrix)


@dataclass(frozen=True)
class Sar:
    """A formula's SAR term, brms's ``sar(M, type)`` (F02b): ``matrix`` the name of the weight matrix in ``data2`` and ``type``
    ``"lag"`` (an autoregression of the responses) or ``"error"`` (of the residuals). It is not a population-level term: it
    has no design column and no ``b`` coefficient, but one parameter, brms's ``lagsar`` or ``errorsar``."""

    matrix: str
    type: str

    @property
    def term(self) -> str:
        """The term as the canonical formula writes it: ``sar(M, type = "lag")``."""
        return f'sar({self.matrix}, type = "{self.type}")'

    @property
    def parameter(self) -> str:
        """brms's name of the term's parameter and of its prior class: ``lagsar`` or ``errorsar``."""
        return SAR_PARAMETERS[self.type]


def parse_sar(piece: str, formula: str) -> Sar | None:
    """brms's ``sar(M, type = "lag")`` as written in ``formula``: a ``Sar``, or None when ``piece`` is not one. R's argument
    rules: named arguments first, then the positional ones in order; ``type`` a string that R's ``match.arg`` reads, a
    non-empty prefix of ``"lag"`` or ``"error"`` (brms: ``'arg' should be one of "lag", "error"`` otherwise); ``M`` the name
    of a matrix in ``data2``."""
    m = _CALL_RE.match(piece)
    if not m:
        return None
    inside = m.group(1).strip()
    named, positional = {}, []
    for arg in (_split_top_level(inside, ",") if inside else []):
        arg = arg.strip()
        km = re.fullmatch(rf"({_IDENT})\s*=(?!=)\s*(.+)", arg, re.S)
        if km is None:
            positional.append(arg)
            continue
        name, value = km.group(1), km.group(2).strip()
        if name not in _FORMALS:
            raise FormulaError(f"{piece!r}: sar() has no argument {name!r}; brms's sar() takes M, type")
        if name in named:
            raise FormulaError(f"{piece!r}: the argument {name!r} is given twice")
        named[name] = value
    rest = [f for f in _FORMALS if f not in named]
    if len(positional) > len(rest):
        raise FormulaError(f"{piece!r}: too many arguments; brms's sar() takes M, type")
    named.update(zip(rest, positional))
    if "M" not in named:
        raise FormulaError(f"{piece!r}: sar() needs its weight matrix, sar(M, type = \"lag\"), M the name of a matrix in data2")
    matrix = named["M"]
    if not re.fullmatch(_IDENT, matrix):
        raise FormulaError(f"{piece!r}: M = {matrix}: M must be the name of a matrix handed in data2 (data2={{'M': matrix}}); "
                           "docs/spatial-sar.md")
    sm = _STRING_RE.fullmatch(named.get("type", '"lag"'))
    text = None if sm is None else (sm.group(1) if sm.group(1) is not None else sm.group(2))
    types = [t for t in SAR_PARAMETERS if text and t.startswith(text)]
    if len(types) != 1:
        raise FormulaError(f"{piece!r}: type = {named['type']}: type must be one of \"lag\", \"error\" (brms: 'arg' should be one "
                           "of \"lag\", \"error\")")
    return Sar(matrix=matrix, type=types[0])


def check_sar(spec, fam, formula: str) -> None:
    """The refusals of a SAR term that depend on the rest of the formula (divergence D143): the family must take it
    (``FamilySpec.sar``: the Gaussian, verified against frozen brms cases; brms also takes the Student-t), and the formula may
    hold population-level terms only (numeric and categorical predictors, interactions, transforms, with or without an
    intercept): group-level terms, an offset, response addition terms, ``mo()``, ``gp()``, ``me()``, ``mi()``, ``cs()`` and an
    autocorrelation term next to it are not verified by a frozen case."""
    from autobayes.families import FAMILIES

    term = spec.sar.term
    if not fam.sar:
        offering = [name for name, other in FAMILIES.items() if other.sar]
        raise FormulaError(
            f"{formula!r}: the SAR term {term!r} is not available for the {fam.name} family; brms's sar() is verified here for the "
            f"{', '.join(offering) or 'no'} famil{'y' if len(offering) == 1 else 'ies'} (brms also takes the student family; "
            "docs/spatial-sar.md, divergence D143)"
        )
    others = ([f"the group-level term {spec.group_terms[0].canonical!r}"] if spec.group_terms else [])
    others += [f"the offset {spec.offset.term!r}"] if spec.offset is not None else []
    others += [f"the response addition term {a.term!r}" for a in spec.additions]
    others += [f"the term {t.name!r}" for t in (*spec.mo_terms, *spec.me_terms, *spec.mi_terms, *spec.cs_terms)]
    others += [f"the Gaussian process {g.term!r}" for g in spec.gp_terms]
    others += [f"the autocorrelation term {spec.autocor.term!r}"] if spec.autocor is not None else []
    if others:
        raise FormulaError(
            f"{formula!r}: the SAR term {term!r} next to {others[0]} is not supported: no frozen brms case verifies "
            "the pair (brms takes it); a SAR model takes population-level terms here (docs/spatial-sar.md, divergence D143)"
        )


# -- the weight matrix of the fitted rows -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SarDesign:
    """The SAR term on the fitted rows (F02b): ``M`` the ``(n, n)`` weight matrix in data order (brms's ``Msar``), its real
    eigenvalues in decreasing order (brms's ``eigenMsar``, R's ``eigen()`` order), and whether ``M`` is symmetric (the
    eigenvalues then from LAPACK's symmetric solver). ``lower`` and ``upper``, the smallest and largest eigenvalue, bound the
    parameter, as brms's ``min_eigenMsar`` and ``max_eigenMsar`` do."""

    term: str
    type: str
    parameter: str
    matrix: str
    M: np.ndarray
    eigenvalues: np.ndarray
    symmetric: bool

    @property
    def n(self) -> int:
        return int(self.M.shape[0])

    @property
    def lower(self) -> float:
        return float(self.eigenvalues.min())

    @property
    def upper(self) -> float:
        return float(self.eigenvalues.max())


def sar_design(term: Sar, data2: Mapping | None, n: int) -> SarDesign:
    """The weight matrix of ``term`` from ``data2`` on the ``n`` fitted rows, checked: present, real, finite and ``n x n`` (brms:
    "Dimensions of 'M' for SAR terms must be equal to the number of observations"), rows and columns read in data order (labels
    are not matched). Its eigenvalues must be real (a non-symmetric M's within ``REAL_TOLERANCE``
    relative, then their real parts: brms hands R's complex values to Stan, which refuses them, divergence D143) and lie in
    [-1, 1] (to ``SPECTRUM_TOLERANCE``): brms bounds the parameter by the smallest and largest eigenvalue, and ``1 - rho
    lambda`` stays positive over that interval only then (D143)."""
    name = term.matrix
    data2_mapping(data2)
    if data2 is None or name not in data2:
        given = "no data2" if data2 is None else f"data2 with {sorted(map(str, data2))}"
        raise FormulaError(f"the SAR term {term.term!r} reads the matrix {name!r} from data2, and the model was given {given}: pass "
                           f"data2={{{name!r}: matrix}}, an {n} x {n} matrix over the fitted rows in data order (docs/spatial-sar.md)")
    value = data2[name]
    raw = value.to_numpy() if isinstance(value, pd.DataFrame) else np.asarray(value)
    if np.issubdtype(raw.dtype, np.complexfloating) or not np.issubdtype(raw.dtype, np.number):
        raise FormulaError(f"data2[{name!r}] of {term.term!r} has dtype {raw.dtype}; the SAR weight matrix must be real numbers")
    M = np.array(raw, dtype=np.float64)
    if M.shape != (n, n):
        raise FormulaError(f"data2[{name!r}] of {term.term!r} has shape {M.shape}; the model has {n} fitted rows, so M must be "
                           f"{n} x {n} (brms: 'Dimensions of 'M' for SAR terms must be equal to the number of observations')")
    if not np.isfinite(M).all():
        raise FormulaError(f"data2[{name!r}] of {term.term!r} has missing or infinite entries (brms refuses them too)")
    symmetric = bool(np.array_equal(M, M.T))
    if symmetric:
        eigenvalues = np.linalg.eigvalsh(M)[::-1]
    else:
        values = np.linalg.eigvals(M)
        scale = max(1.0, float(np.abs(values).max()))
        if float(np.abs(values.imag).max()) > REAL_TOLERANCE * scale:
            raise FormulaError(
                f"data2[{name!r}] of {term.term!r} has complex eigenvalues (largest imaginary part "
                f"{float(np.abs(values.imag).max()):.3g}); the SAR log determinant sum(log(1 - rho * lambda)) and brms's bounds of "
                "rho need real ones (brms passes complex ones to Stan, which refuses them; docs/spatial-sar.md, divergence D143)")
        eigenvalues = np.sort(values.real)[::-1]
    eigenvalues = np.ascontiguousarray(eigenvalues, dtype=np.float64)
    if not float(np.abs(eigenvalues).max()) <= 1.0 + SPECTRUM_TOLERANCE:
        raise FormulaError(
            f"the eigenvalues of data2[{name!r}] of {term.term!r} lie in [{eigenvalues.min():.6g}, {eigenvalues.max():.6g}], outside "
            "[-1, 1]: brms bounds the SAR parameter by the smallest and largest eigenvalue, and 1 - rho * lambda is then not "
            "positive over all of that interval, where the log determinant is undefined; normalise M (row-standardised, or "
            "D^-1/2 W D^-1/2 for an adjacency W with degrees D) (docs/spatial-sar.md, divergence D143)")
    if not eigenvalues.min() < eigenvalues.max():
        raise FormulaError(f"the eigenvalues of data2[{name!r}] of {term.term!r} are all {eigenvalues[0]:g}: they bound no interval "
                           "for the SAR parameter")
    return SarDesign(term=term.term, type=term.type, parameter=term.parameter, matrix=name, M=M, eigenvalues=eigenvalues,
                     symmetric=symmetric)


# -- the model ---------------------------------------------------------------------------------------------------------
def sar_family(base, design: SarDesign):
    """The Bambi family ``base`` (the Gaussian's name or a family object) with the SAR likelihood: Bambi builds the linear
    predictor ``mu`` (its ``mu`` deterministic, brms's ``posterior_linpred``) and ``sigma`` as for the Gaussian family, and
    hands them to this likelihood, which creates the parameter ``rho`` (``design.parameter``) under brms's flat prior on
    ``(design.lower, design.upper)`` (a ``CustomDist`` of log density 0 on the interval, sampled on PyMC's interval coordinate
    ``log((rho - lower) / (upper - rho))``, Stan's lower-upper transform, whose log Jacobian is the whole of rho's contribution,
    as in brms; its support point, PyMC's initial value, the interval's midpoint) and evaluates the joint density of the
    responses, one vector (signature ``(n),(),()->(n)``): ``-n log(sigma) - sum_i r_i^2 / (2 sigma^2) + log|A|``, brms's
    ``normal_lagsar_lpdf`` / ``normal_errorsar_lpdf`` (without the Gaussian constant, as brms's). Draws are the joint model's
    (``sar_response``)."""
    import copy

    import bambi as bmb
    import pymc as pm
    import pytensor.tensor as pt
    from bambi.defaults import get_builtin_family
    from pymc.distributions.transforms import Interval

    family = get_builtin_family(base) if isinstance(base, str) else copy.copy(base)
    lk = family.likelihood
    M, eigenvalues = np.asarray(design.M, dtype=np.float64), np.asarray(design.eigenvalues, dtype=np.float64)
    lower, upper, n = np.float64(design.lower), np.float64(design.upper), np.float64(design.n)
    half = np.float64(0.5)
    lag = design.type == "lag"

    def logp(value, mu, sigma, rho):
        if lag:
            r = value - rho * pt.dot(M, value) - mu
        else:
            e = value - mu
            r = e - rho * pt.dot(M, e)
        log_det = pt.sum(pt.log1p(-rho * eigenvalues))
        return -n * pt.log(sigma) - half * pt.sum(pt.sqr(r)) / pt.sqr(sigma) + log_det

    def flat(value):
        return pt.switch(pt.and_(pt.gt(value, lower), pt.lt(value, upper)), pt.zeros_like(value), -np.inf)

    def middle(rv, size):                                     # a constant: full_like(rv, ...) would draw rv to read its shape
        return pt.as_tensor_variable(half * (lower + upper))

    def random(mu, sigma, rho, rng=None, size=None):
        mu, sigma, rho = (np.asarray(v, dtype=np.float64) for v in (mu, sigma, rho))
        batch = tuple(size) if size is not None else np.broadcast_shapes(mu.shape[:-1], sigma.shape, rho.shape)
        eta = np.broadcast_to(mu, batch + mu.shape[-1:])
        eps = np.broadcast_to(sigma, batch)[..., None] * rng.standard_normal(eta.shape)
        return sar_response(eta, eps, np.broadcast_to(rho, batch), design)

    class SarDist:
        """``y ~`` the SAR model's joint density, given Bambi's ``mu`` and ``sigma``; creates ``rho``."""

        def __new__(cls, name, *, observed=None, dims=None, **params):
            rho = pm.CustomDist(design.parameter, logp=flat, support_point=middle, transform=Interval(lower, upper), dtype="float64")
            return pm.CustomDist(name, params["mu"], params["sigma"], rho, logp=logp, random=random, signature="(n),(),()->(n)",
                                 observed=observed, dims=dims, dtype="float64")

    family.likelihood = bmb.Likelihood(lk.name, params=lk.params, parent=lk.parent, dist=SarDist)
    return family


def _solve(rho, values, design: SarDesign) -> np.ndarray:
    """``(I - rho M)^-1 values`` for each draw: ``rho`` of shape ``lead`` and ``values`` of ``lead + (n,)``, one linear solve
    per draw (memory ``n x n``, not a matrix per draw)."""
    rho = np.asarray(rho, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    lead, n = values.shape[:-1], values.shape[-1]
    rho = np.broadcast_to(rho, lead).reshape(-1)
    flat = values.reshape(-1, n)
    out = np.empty_like(flat)
    identity = np.eye(n)
    for k in range(flat.shape[0]):
        out[k] = np.linalg.solve(identity - rho[k] * design.M, flat[k])
    return out.reshape(values.shape)


def sar_mean(eta, rho, design: SarDesign) -> np.ndarray:
    """The expected response: ``(I - rho M)^-1 eta`` of the lag model (brms's ``posterior_epred``), ``eta`` itself for the
    error model, whose residuals have mean zero (brms's ``posterior_epred`` fails there; divergence D142). ``eta`` is
    ``lead + (n,)``, ``rho`` ``lead``."""
    if design.type == "lag":
        return _solve(rho, eta, design)
    return np.array(eta, dtype=np.float64)


def sar_response(eta, eps, rho, design: SarDesign) -> np.ndarray:
    """Responses from innovations ``eps`` (``sigma`` times standard normal draws): ``(I - rho M)^-1 (eta + eps)`` under the
    lag model, ``eta + (I - rho M)^-1 eps`` under the error model."""
    if design.type == "lag":
        return _solve(rho, np.asarray(eta) + np.asarray(eps), design)
    return np.asarray(eta, dtype=np.float64) + _solve(rho, eps, design)


def sar_log_lik(y, eta, sigma, rho, design: SarDesign) -> np.ndarray:
    """brms's ``log_lik()`` of the SAR model: each row's log density given every other row (module docstring), for each draw.
    ``y`` the ``(n,)`` responses, ``eta`` ``lead + (n,)``, ``sigma`` and ``rho`` ``lead``; returns ``lead + (n,)``."""
    eta = np.asarray(eta, dtype=np.float64)
    lead, n = eta.shape[:-1], eta.shape[-1]
    y = np.asarray(y, dtype=np.float64)
    flat = eta.reshape(-1, n)
    rho = np.broadcast_to(np.asarray(rho, dtype=np.float64), lead).reshape(-1)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), lead).reshape(-1)
    out = np.empty_like(flat)
    identity = np.eye(n)
    for k in range(flat.shape[0]):
        A = identity - rho[k] * design.M
        r = A @ y - flat[k] if design.type == "lag" else A @ (y - flat[k])
        s2 = sigma[k] ** 2
        z = A.T @ r / s2                                      # Q (y - E[y]), Q = A^T A / sigma^2
        q = np.einsum("ji,ji->i", A, A) / s2                  # Q's diagonal
        out[k] = -0.5 * math.log(2.0 * math.pi) + 0.5 * np.log(q) - 0.5 * z ** 2 / q
    return out.reshape(eta.shape)


# -- describe() ----------------------------------------------------------------------------------------------------------
def data_lines(model) -> list[str]:
    """``describe()``'s data line of the SAR term (F02b): its weight matrix and the interval its eigenvalues bound."""
    d = model.design.sar
    shape = "symmetric" if d.symmetric else "non-symmetric, with real eigenvalues"
    return [f"  sar       : {d.term}: data2[{d.matrix!r}], {d.n} x {d.n} ({shape}) over the fitted rows in data order (brms's Msar); "
            f"its eigenvalues lie in [{d.lower:.6g}, {d.upper:.6g}] (brms's eigenMsar), which bound {d.parameter}"]


def likelihood_lines(model) -> list[str]:
    """``describe()``'s prior lines of sigma and the SAR parameter and its likelihood paragraph (F02b), in place of the Gaussian
    family's."""
    from autobayes.families.spec import scale_line
    from autobayes.priors import describe_prior

    d = model.design.sar
    rho, prior = d.parameter, model.priors.sar_priors(model.spec)[d.parameter]
    residual = "r = A y - mu" if d.type == "lag" else "r = A (y - mu)"
    model_line = ("A y = mu + e: y ~ MultiNormal(A^-1 mu, sigma^2 A^-1 A^-T), an autoregression of the responses" if d.type == "lag"
                  else "A (y - mu) = e: y ~ MultiNormal(mu, sigma^2 A^-1 A^-T), an autoregression of the residuals")
    return [scale_line(model),
            f"  {rho} ~ {describe_prior(prior)} on ({d.lower:.6g}, {d.upper:.6g})   (brms class {rho}, bounded by min_eigenMsar and "
            "max_eigenMsar; flat: only the coordinate's Jacobian enters the log density)",
            "", "Likelihood",
            f"  {model_line},",
            f"  with A = I - {rho} * {d.matrix}, e ~ Normal(0, sigma^2 I) and mu the linear predictor;",
            f"  log p(y) = log|A| + sum_i [-log(sigma) - r_i^2 / (2 sigma^2)] with {residual} and log|A| = sum_k log(1 - {rho} *",
            f"  lambda_k) over the eigenvalues of {d.matrix}: brms's normal_{rho}_lpdf, which leaves out the constant -n/2 log(2 pi)",
            "  (divergence D142); log_lik() keeps it. The rows are not independent: log_lik() is each row's density given every",
            "  other row, and predict(kind=\"mean\") is " + ("A^-1 mu (brms's posterior_epred)." if d.type == "lag" else
                                                            "mu (brms's posterior_epred fails here; divergence D142).")]


def sar_draws(model, post, mu, rng) -> np.ndarray:
    """``predict(kind="response")``'s draws of the SAR model (F02b): per posterior draw, innovations ``sigma * z`` with ``z``
    standard normal (``rng``), turned into responses by ``sar_response``. ``mu`` is the ``(chain, draw, n)`` linear predictor."""
    d = model.design.sar
    sigma = post[model.names["sigma"]].to_numpy()[..., None]
    return sar_response(mu, sigma * rng.standard_normal(np.shape(mu)), post[d.parameter].to_numpy(), d)
