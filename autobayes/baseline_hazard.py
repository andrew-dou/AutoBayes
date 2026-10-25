"""brms's ``cox()`` proportional hazards model (brms-parity F05, docs/cox.md): ``time | cens(c) ~ x + (1 | g)``.

Row ``i`` has the hazard ``h0(t) exp(eta_i)`` at time ``t``, with ``eta_i`` the linear predictor (brms's log link: brms's ``mu``
is ``exp(eta)``, its ``posterior_linpred`` ``eta``) and the baseline hazard ``h0(t) = sum_k sbhaz[k] M_k(t)``, whose integral
from the lower boundary knot is ``H0(t) = sum_k sbhaz[k] I_k(t)``. ``M_k`` are the cubic M-splines of the time with an intercept
(every basis function kept) and ``I_k`` their integrals, the I-splines (Ramsay 1988, "Monotone regression splines in action",
Statistical Science 3); ``sbhaz`` is a simplex under a Dirichlet prior (brms's class ``sbhaz``), so that ``H0`` rises from 0 at
the lower boundary knot to 1 at the upper one. A row with an event contributes ``log h0(t_i) + eta_i - H0(t_i) exp(eta_i)``, a
right-censored row ``-H0(t_i) exp(eta_i)``, the log survival; the time enters only through the bases.

The bases (brms's Stan data ``Zbhaz``, ``Zcbhaz``; ``Kbhaz`` of them) are built here from their definitions. With the knot
sequence ``t``, each boundary knot repeated ``k = degree + 1 = 4`` times around the interior knots, the B-splines of order ``k``
follow from Cox and de Boor's recursion (de Boor 1972, "On calculating with B-splines", J. Approx. Theory 6)

    B_{i,1}(x) = 1 if t_i <= x < t_{i+1} (the last non-empty interval closed on the right), else 0;
    B_{i,j}(x) = (x - t_i) / (t_{i+j-1} - t_i) B_{i,j-1}(x) + (t_{i+j} - x) / (t_{i+j} - t_{i+1}) B_{i+1,j-1}(x), 0 / 0 = 0,

the M-splines are ``M_i = k B_{i,k} / (t_{i+k} - t_i)``, each integrating to 1 (Ramsay's normalisation), and, since
``d/dx B'_{j,k+1} = M_{j-1} - M_j`` for the order-``k + 1`` B-splines ``B'`` on the same knots with each boundary knot repeated
once more, the I-splines are ``I_i = sum_{j > i} B'_{j,k+1}``, 0 at the lower boundary knot and 1 at the upper one. Against
the frozen fixtures' ``Zbhaz`` and ``Zcbhaz`` (splines2's, which brms calls) they agree to 7e-16 (tests/test_cox.py).

brms's knots (``bhaz()``'s defaults, probed by the fixture step): the boundary knots are the range of the times of every row,
censored rows too, widened by 2% of its length at each end; ``df - 4`` interior knots (``df`` 5 by default: the median alone)
at the times' quantiles (R's type 7) at the ``df - 2`` equally spaced probabilities from 0 to 1 less the two ends;
``bhaz(knots = ...)`` and ``bhaz(Boundary.knots = ...)`` give either. Divergences D152-D155 (docs/brms-parity/divergences.md).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from autobayes.formula import AdditionTerm, FormulaError

DEGREE = 3                         # brms's cubic M-splines (bhaz()'s degree, which is not offered as an argument: D155)
DEFAULT_DF = 5                     # bhaz()'s default number of bases
MIN_DF = DEGREE + 1                # the bases of a cubic spline with its intercept and no interior knot (brms refuses df = 3)
WIDENING = 0.02                    # brms widens the range of the times by 2% of its length at each end
SBHAZ = "sbhaz"                    # brms's simplex[Kbhaz] sbhaz, the PyMC variable of the same name
SBHAZ_DIM = "sbhaz_dim"            # its coordinate, "1".."Kbhaz" (brms's column names of Zbhaz)
_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_BHAZ_ARGUMENTS = ("df", "knots", "Boundary.knots")

INTERVAL_REFUSED = ("brms 2.23 evaluates both ends of an interval-censored cox row with the bases at the row's own time (it builds "
                    "none at the upper bound), so the row's log likelihood, the log difference of two equal survival "
                    "probabilities, is -inf (the fixture step's probe, #101); refused rather than copied (docs/cox.md, "
                    "divergence D153)")
TRUNCATION_REFUSED = ("brms 2.23 normalises a truncated cox row by the survival at its own time instead of at the truncation "
                      "bound, which cancels the row's survival term: on cox_varint's data under trunc(lb = 0.5) brms's normalising "
                      "terms sum to 17.13382 (the sum of H0(y) exp(eta)) where -sum log S(0.5) is 0.3908019 (the fixture step's "
                      "probe, #101); refused rather than copied (docs/cox.md, divergence D153)")
WEIGHTS_REFUSED = ("weights() of a cox model is not verified by a frozen brms case (docs/cox.md, divergence D155)")
DRAWS_REFUSED = ("draws of a cox model's response are not offered: brms refuses posterior_predict for cox (\"Cannot sample from "
                 "the posterior predictive distribution for family 'cox'.\"), and no sampler of the survival time is "
                 "implemented here (docs/cox.md, divergence D154)")


# -- bhaz() ----------------------------------------------------------------------------------------------------------
def _numbers(value: str, what: str, piece: str, formula: str) -> tuple[float, ...]:
    """The finite numbers of ``c(a, b, ...)`` or of one number."""
    text = value.strip()
    m = re.fullmatch(r"c\s*\((.*)\)", text, re.S)
    parts = [p.strip() for p in (m.group(1).split(",") if m else [text])]
    if not parts or not all(re.fullmatch(_NUMBER, p) for p in parts):
        raise FormulaError(f"{formula!r}: {piece!r}: {what} must be a number or c() of numbers, got {value.strip()!r}")
    out = tuple(float(p) for p in parts)
    if not all(math.isfinite(v) for v in out):
        raise FormulaError(f"{formula!r}: {piece!r}: {what} must be finite numbers, got {value.strip()!r}")
    return out


def parse_bhaz(args: list[str], piece: str, formula: str) -> AdditionTerm:
    """brms's ``bhaz()`` of a cox model, the settings of its baseline hazard: ``df = K`` (the number of bases, at least 4;
    brms refuses 3), ``knots = c(...)`` (the interior knots, strictly increasing; the number of bases is then their number
    plus 4) and ``Boundary.knots = c(lo, hi)``, each at most once; ``df`` and ``knots`` not together (brms ignores ``df``
    then, D152). ``gr = `` (a stratified baseline hazard), ``degree``, ``intercept`` and splines2's other arguments are
    refused (D155)."""
    seen: dict[str, str] = {}
    for arg in args:
        km = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.+)", arg, re.S)
        if km is None:
            raise FormulaError(f"{formula!r}: {piece!r}: bhaz() takes named arguments, df = , knots = and Boundary.knots = ; "
                               f"got {arg!r}")
        name, value = km.group(1), km.group(2)
        if name == "gr":
            raise FormulaError(f"{formula!r}: {piece!r}: a stratified baseline hazard, bhaz(gr = ), one baseline hazard per "
                               "level, is not offered: no frozen brms case verifies it (docs/cox.md, divergence D155)")
        if name not in _BHAZ_ARGUMENTS:
            raise FormulaError(f"{formula!r}: {piece!r}: bhaz({name} = ) is not offered: the baseline hazard is brms's cubic "
                               "M-spline basis with its intercept, set by df, knots and Boundary.knots only (docs/cox.md, "
                               "divergence D155)")
        if name in seen:
            raise FormulaError(f"{formula!r}: {piece!r}: bhaz() names {name} twice")
        seen[name] = value
    df = knots = boundary = None
    if "df" in seen:
        value = seen["df"].strip()
        if not re.fullmatch(r"\d+(?:\.0*)?", value):
            raise FormulaError(f"{formula!r}: {piece!r}: bhaz(df = ) is a whole number of bases, got {value!r}")
        df = int(float(value))
        if df < MIN_DF:
            raise FormulaError(f"{formula!r}: {piece!r}: bhaz(df = {df}) gives too few bases: a cubic baseline hazard with its "
                               f"intercept has at least {MIN_DF} (brms: 'The specified df was too small.')")
    if "knots" in seen:
        knots = _numbers(seen["knots"], "knots", piece, formula)
        if not all(b > a for a, b in zip(knots, knots[1:])):
            raise FormulaError(f"{formula!r}: {piece!r}: the interior knots must be strictly increasing, got {list(knots)}")
        if df is not None:
            raise FormulaError(f"{formula!r}: {piece!r}: bhaz() takes df or knots, not both: brms ignores df when knots are "
                               f"given (bhaz(df = {df}, knots = ...) has {len(knots) + MIN_DF} bases; docs/cox.md, divergence D152)")
    if "Boundary.knots" in seen:
        boundary = _numbers(seen["Boundary.knots"], "Boundary.knots", piece, formula)
        if len(boundary) != 2 or not boundary[0] < boundary[1]:
            raise FormulaError(f"{formula!r}: {piece!r}: Boundary.knots are two numbers, the lower one first, got {list(boundary)}")
        if knots is not None and not (boundary[0] < knots[0] and knots[-1] < boundary[1]):
            raise FormulaError(f"{formula!r}: {piece!r}: the interior knots {list(knots)} must lie strictly inside the boundary "
                               f"knots {list(boundary)}")
    return AdditionTerm("bhaz", df=df, knots=knots, boundary=boundary)


DEFAULT_BHAZ = AdditionTerm("bhaz")      # a cox formula without bhaz(): brms's defaults


def baseline_term(spec) -> AdditionTerm:
    """The formula's ``bhaz()`` term, or brms's defaults without one."""
    return spec.addition("bhaz") or DEFAULT_BHAZ


def n_bases(spec) -> int:
    """brms's ``Kbhaz``: the number of interior knots plus 4, or ``df`` (5 by default)."""
    term = baseline_term(spec)
    return len(term.knots) + MIN_DF if term.knots is not None else (term.df if term.df is not None else DEFAULT_DF)


# -- the bases -----------------------------------------------------------------------------------------------------
def bsplines(x, t, order: int) -> np.ndarray:
    """The ``len(t) - order`` B-splines of ``order`` on the knot sequence ``t`` at ``x``, ``(len(x), len(t) - order)``, by
    Cox and de Boor's recursion (module docstring), the last non-empty knot interval closed on the right so that the upper
    boundary knot belongs to it."""
    x, t = np.asarray(x, dtype=float).reshape(-1), np.asarray(t, dtype=float)
    last = int(np.flatnonzero(t[:-1] < t[1:])[-1])
    B = ((t[:-1] <= x[:, None]) & (x[:, None] < t[1:])).astype(float)
    B[x == t[-1], last] = 1.0
    for j in range(2, order + 1):
        n = len(t) - j
        left, right = t[j - 1:j - 1 + n] - t[:n], t[j:j + n] - t[1:1 + n]
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(left > 0, (x[:, None] - t[:n]) / left, 0.0)
            b = np.where(right > 0, (t[j:j + n] - x[:, None]) / right, 0.0)
        B = a * B[:, :n] + b * B[:, 1:n + 1]
    return B


def spline_bases(x, interior, boundary, degree: int = DEGREE) -> tuple[np.ndarray, np.ndarray]:
    """The M-spline basis of ``degree`` with its intercept on the interior and boundary knots at ``x`` and its I-spline basis
    (module docstring): ``(M, I)``, each ``(len(x), len(interior) + degree + 1)``."""
    k = degree + 1
    lo, hi = float(boundary[0]), float(boundary[1])
    inner = [float(v) for v in interior]
    t = np.array([lo] * k + inner + [hi] * k)
    B = bsplines(x, t, k)
    n = B.shape[1]
    M = k * B / (t[k:k + n] - t[:n])
    B1 = bsplines(x, np.array([lo] * (k + 1) + inner + [hi] * (k + 1)), k + 1)   # n + 1 functions of order k + 1
    I = np.cumsum(B1[:, ::-1], axis=1)[:, ::-1][:, 1:]                           # I_i = sum_{j > i} B1_j
    return M, I


def baseline_knots(y, term: AdditionTerm) -> tuple[np.ndarray, tuple[float, float]]:
    """brms's interior and boundary knots of ``term`` (``bhaz()``) for the times ``y`` (module docstring)."""
    y = np.asarray(y, dtype=float)
    lo, hi = float(y.min()), float(y.max())
    if term.boundary is not None:
        boundary = (float(term.boundary[0]), float(term.boundary[1]))
    else:
        if not hi > lo:
            raise FormulaError(f"every time is {lo!r}: brms's boundary knots, the range of the times widened by 2%, need two "
                               "different times")
        boundary = (lo - WIDENING * (hi - lo), hi + WIDENING * (hi - lo))
    if term.knots is not None:
        interior = np.asarray(term.knots, dtype=float)
    else:
        df = DEFAULT_DF if term.df is None else term.df
        interior = np.quantile(y, np.linspace(0.0, 1.0, df - DEGREE + 1)[1:-1], method="linear")   # R's type 7
    if interior.size and not (np.all(np.diff(interior) > 0) and boundary[0] < interior[0] and interior[-1] < boundary[1]):
        raise FormulaError(f"the interior knots {interior.tolist()} of the baseline hazard are not strictly increasing inside the "
                           f"boundary knots {list(boundary)} (tied times give equal quantiles; pass bhaz(knots = c(...)) or fewer "
                           "df; docs/cox.md, divergence D152)")
    return interior, boundary


@dataclass(frozen=True)
class BaselineHazard:
    """The baseline hazard of a cox model on its fitted rows (``Design.bhaz``): the ``bhaz()`` term (brms's defaults without
    one), its interior and boundary knots, brms's ``Zbhaz`` (the M-spline bases at each row's time) and ``Zcbhaz`` (the
    I-spline bases), and which rows have an event (cens 0, or every row without ``cens()``)."""

    term: AdditionTerm
    knots: np.ndarray
    boundary: tuple[float, float]
    Zbhaz: np.ndarray
    Zcbhaz: np.ndarray
    event: np.ndarray

    @property
    def K(self) -> int:
        """brms's ``Kbhaz``."""
        return self.Zbhaz.shape[1]

    @property
    def labels(self) -> list[str]:
        """The coordinates of ``sbhaz``: brms's column names of ``Zbhaz``, ``"1"``..``"K"``."""
        return [str(k) for k in range(1, self.K + 1)]


def baseline_design(spec, df) -> BaselineHazard:
    """The ``BaselineHazard`` of the fitted rows ``df``: the knots from every row's time, the bases at each, and the events
    from ``cens()`` (0 an event, 1 right-censored; left censoring, -1, is refused: D155). A time outside given
    ``Boundary.knots`` is refused (D152)."""
    from autobayes.formula import addition_data

    term = baseline_term(spec)
    y = df[spec.response].to_numpy(dtype=float)
    cens = addition_data(spec, df).get("cens")
    if cens is not None and (cens == -1).any():
        i = int(np.flatnonzero(cens == -1)[0])
        raise FormulaError(f"row {i} of {spec.addition('cens').term!r} is left-censored (-1); the cox family takes right censoring "
                           "only here, 0 an event and 1 right-censored (no frozen brms case verifies brms's left-censored cox "
                           "rows; docs/cox.md, divergence D155)")
    interior, boundary = baseline_knots(y, term)
    outside = np.flatnonzero((y < boundary[0]) | (y > boundary[1]))
    if outside.size:
        i = int(outside[0])
        raise FormulaError(f"the time {float(y[i])!r} of row {i} lies outside the boundary knots {list(boundary)} of {term.term!r}; "
                           "every fitted row's time must lie within them (docs/cox.md, divergence D152)")
    M, I = spline_bases(y, interior, boundary)
    event = np.ones(len(y), dtype=bool) if cens is None else cens == 0
    return BaselineHazard(term=term, knots=interior, boundary=boundary, Zbhaz=M, Zcbhaz=I, event=event)


# -- the likelihood ------------------------------------------------------------------------------------------------
def cox_logp(eta, Zbhaz, Zcbhaz, sbhaz, event):
    """Each row's log likelihood (PyTensor): ``log h0 + eta - H0 exp(eta)`` at an event, ``-H0 exp(eta)`` at a right-censored
    row, with ``h0 = Zbhaz sbhaz`` and ``H0 = Zcbhaz sbhaz``; the logarithm is taken of the events' ``h0`` alone."""
    import pytensor.tensor as pt

    event = np.asarray(event, dtype=bool)
    h0 = (np.asarray(Zbhaz, dtype=np.float64) * sbhaz).sum(axis=-1)       # sbhaz (K,), or (1, K) as the likelihood's batch hands it
    H0 = (np.asarray(Zcbhaz, dtype=np.float64) * sbhaz).sum(axis=-1)
    cumulative = H0 * pt.exp(eta)
    return pt.switch(event, pt.log(pt.switch(event, h0, np.float64(1.0))) + eta - cumulative, -cumulative)


def cox_log_lik(eta, Zbhaz, Zcbhaz, sbhaz, event) -> np.ndarray:
    """``cox_logp`` in NumPy for draws: ``eta`` ``(..., n)``, ``sbhaz`` ``(..., K)``; the result is ``(..., n)``."""
    eta, s = np.asarray(eta, dtype=float), np.asarray(sbhaz, dtype=float)
    h0, H0 = s @ np.asarray(Zbhaz, dtype=float).T, s @ np.asarray(Zcbhaz, dtype=float).T
    with np.errstate(over="ignore"):
        cumulative = H0 * np.exp(eta)
    return np.where(event, np.log(np.where(event, h0, 1.0)) + eta - cumulative, -cumulative)


def _refuse_draws(*args, **kwargs):
    raise ValueError(DRAWS_REFUSED)


def cox_family(block: BaselineHazard, concentration):
    """Bambi family of brms's ``cox()``: the linear predictor ``log_mu`` under Bambi's identity link, and the likelihood
    ``cox_logp`` of the fitted rows' bases; building it adds the simplex ``sbhaz`` (a Dirichlet of ``concentration``, sampled
    on Stan's stick-breaking coordinates; ``monotonic.simplex_rv``) on the coordinate ``sbhaz_dim``. It draws nothing."""
    import bambi as bmb
    import pymc as pm

    from autobayes.monotonic import simplex_rv

    def logp(value, eta, sbhaz):
        return cox_logp(eta, block.Zbhaz, block.Zcbhaz, sbhaz, block.event)

    class Cox:
        def __new__(cls, name_, **kwargs):
            eta = kwargs.pop("log_mu")
            pm.modelcontext(None).add_coord(SBHAZ_DIM, block.labels)
            sbhaz = simplex_rv(SBHAZ, concentration, SBHAZ_DIM)
            return pm.CustomDist(name_, eta, sbhaz, logp=logp, random=_refuse_draws, signature="(),(k)->()", **kwargs)

        @staticmethod
        def dist(**kwargs):
            raise ValueError(DRAWS_REFUSED)

    likelihood = bmb.Likelihood("Cox", params=["log_mu"], parent="log_mu", dist=Cox)
    return bmb.Family("cox", likelihood, link={"log_mu": "identity"})


# -- priors, points and the graph ------------------------------------------------------------------------------------
def check_priors(priors, spec, fam) -> None:
    """``sbhaz``: the ``Dirichlet`` prior of the baseline hazard's simplex, required for the cox family (one concentration for
    every basis, brms's default ``dirichlet(1)``, or one per basis) and refused for every other family."""
    from autobayes.priors import Prior, PriorError, dirichlet_concentration

    prior = getattr(priors, "sbhaz", None)
    if not fam.baseline_hazard:
        if prior is not None:
            raise PriorError(f"sbhaz is the prior of the cox family's baseline hazard simplex (brms class sbhaz); the {fam.name} "
                             "family has none")
        return
    if prior is None:
        raise PriorError("the cox family needs a prior on its baseline hazard simplex sbhaz: pass sbhaz=ab.Dirichlet(1) (brms's "
                         "default dirichlet(1)), one concentration for every basis or one per basis")
    if not isinstance(prior, Prior) or prior.name != "Dirichlet":
        raise PriorError(f"prior for 'sbhaz' must be a Dirichlet prior on the simplex of the baseline hazard's weights (brms's "
                         f"simplex sbhaz), got {prior}")
    dirichlet_concentration(prior, n_bases(spec))


def value_entries(model, sbhaz) -> dict[str, np.ndarray]:
    """``Model.value_point``'s entry of the baseline hazard's simplex: ``sbhaz`` (``Kbhaz`` positive weights summing to 1) on
    Stan's stick-breaking coordinates, required for a cox model and refused for every other."""
    from autobayes.monotonic import check_simplex, stick_breaking_free

    block = getattr(model.design, "bhaz", None)
    if block is None:
        if sbhaz is not None:
            raise ValueError(f"sbhaz is the cox family's baseline hazard simplex; the {model.family} family has none")
        return {}
    if sbhaz is None:
        raise ValueError(f"sbhaz (the baseline hazard's {block.K} weights, a simplex) is required for the cox family")
    return {f"{SBHAZ}_stickbreaking__": stick_breaking_free(check_simplex(sbhaz, block.K, "sbhaz"))}


def check_graph(model) -> None:
    """The graph's ``sbhaz``: a free simplex of ``Kbhaz`` entries on the coordinate ``sbhaz_dim`` (``"1"``..``"K"``), sampled
    on Stan's ``K - 1`` stick-breaking coordinates."""
    from autobayes.monotonic import StickBreakingTransform

    pmm, K = model.pymc_model, model.design.bhaz.K
    rv = pmm[SBHAZ]
    if rv not in pmm.free_RVs or tuple(rv.type.shape) != (K,):
        raise RuntimeError(f"{SBHAZ!r} is not a free simplex of {K} entries in the graph")
    value = pmm.rvs_to_values[rv]
    if (value.name != f"{SBHAZ}_stickbreaking__" or tuple(value.type.shape) != (K - 1,)
            or not isinstance(pmm.rvs_to_transforms[rv], StickBreakingTransform)):
        raise RuntimeError(f"{SBHAZ!r} is not sampled on Stan's {K - 1} stick-breaking coordinates")
    if (tuple(pmm.named_vars_to_dims.get(SBHAZ, ())) != (SBHAZ_DIM,)
            or [str(v) for v in pmm.coords[SBHAZ_DIM]] != model.design.bhaz.labels):
        raise RuntimeError(f"PyMC dims of {SBHAZ!r} are not ({SBHAZ_DIM!r},) with coordinates 1..{K}")


# -- the formula -----------------------------------------------------------------------------------------------------
def check_formula(spec) -> None:
    """What a cox formula takes besides the generic refusals (``FamilySpec.check_formula``): ``cens()`` without an interval's
    upper bounds (D153: brms's interval-censored cox rows are -inf)."""
    cens = spec.addition("cens")
    if cens is not None and cens.y2 is not None:
        raise FormulaError(f"{spec.formula!r}: {cens.term!r}: interval censoring of a cox model is not offered: {INTERVAL_REFUSED}")


def describe_lines(model) -> list[str]:
    """``describe()``'s lines of the baseline hazard and the likelihood (``FamilySpec.likelihood_lines``)."""
    from autobayes.priors import describe_prior, dirichlet_concentration

    block, prior = model.design.bhaz, model.priors.sbhaz
    con = ", ".join(f"{v:g}" for v in dirichlet_concentration(prior, block.K))
    knots = ", ".join(f"{v:g}" for v in block.knots) or "none"
    events = int(block.event.sum())
    term = block.term.term if block.term != DEFAULT_BHAZ else f"bhaz() (brms's defaults, df = {DEFAULT_DF})"
    return [f"  {SBHAZ} ~ {describe_prior(prior)}   (brms class sbhaz; concentrations {con}: the weights of the {block.K} baseline "
            "hazard bases, a simplex)",
            "", "Likelihood",
            "  The hazard of row i at time t is h0(t) exp(eta_i) (brms's cox(), log link). The baseline hazard h0(t) = sum_k "
            f"sbhaz[k] M_k(t),",
            f"  M_k the {block.K} cubic M-spline bases with their intercept (Ramsay 1988; brms's Zbhaz) of {term}: boundary "
            f"knots {block.boundary[0]:g} and {block.boundary[1]:g},",
            f"  interior knot(s) {knots}; its integral H0(t) = sum_k sbhaz[k] I_k(t), I_k the I-spline bases (brms's Zcbhaz), each "
            "rising from 0 to 1.",
            "  An event contributes log h0(t_i) + eta_i - H0(t_i) exp(eta_i), a right-censored row -H0(t_i) exp(eta_i), the log "
            f"survival: {events} event(s),",
            f"  {model.design.n - events} right-censored row(s). The times enter only through the bases; they are strictly positive."]


def value_roles(model) -> dict[str, str]:
    """``describe()``'s value variable of the baseline hazard."""
    K = model.design.bhaz.K
    return {f"{SBHAZ}_stickbreaking__": f"{SBHAZ}[1..{K}] on Stan's stick-breaking simplex coordinates ({K - 1} entries; brms's "
                                        f"sbhaz); dims {SBHAZ_DIM}"}
