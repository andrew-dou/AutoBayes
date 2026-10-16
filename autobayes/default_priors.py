"""brms's default priors, opt-in (brms-parity A07, docs/default-priors.md).

``default_priors(formula, data, ...)`` returns the ``Priors`` brms 2.23 uses when it is given no prior: the table of
``brms::default_prior()`` for the same formula, data and family, as priors the model constructors accept. Nothing here
is filled in behind the user's back: the constructors still require every prior (D1), and these are passed to them
explicitly, ``ab.gaussian(f, df, ab.default_priors(f, df))``. Each prior it returns is marked ``source="default"``, which
``describe()`` and fit records report.

brms's rules (``default_prior()``, ``def_dpar_prior``, ``def_scale_prior``, ``prior_re``, ``prior_fe`` in brms 2.23):

- ``Intercept`` (the centred intercept): ``student_t(3, location, scale)``. When brms reads the response through its
  family's link (identity; log, where zeros count as 0.1; the lognormal's log scale), ``location`` is
  ``round(median(y_link), 1)`` minus the mean of the offset, and ``scale`` is ``max(2.5, round(mad(y_link), 1))``; for
  any other link (logit, probit) they are 0 and 2.5. ``FamilySpec.default_prior_link`` says which.
- ``b``: flat (an empty prior string), ``Flat()``. Without centring (``center_predictors=False``) the intercept is one
  of brms's ``b`` coefficients and flat as well.
- ``sd`` and ``sigma``: ``student_t(3, 0, scale)`` bounded below by 0, ``HalfStudentT(3, scale)``, with the scale above.
- ``cor``: ``lkj(1)``. The family's auxiliary parameter and a learned ``nu``: the family's own default
  (``FamilySpec.default_scale_prior``, ``FamilySpec.default_learned_priors``): negative binomial ``shape``
  ``inv_gamma(0.4, 0.3)``, Gamma ``shape`` and Beta ``phi`` ``gamma(0.01, 0.01)``, Student-t ``nu`` ``gamma(2, 0.1)``.
  The zero-inflation or hurdle probability of the zero-inflated and hurdle families (B06): ``beta(1, 1)`` with bounds 0
  and 1 (``FamilySpec.default_zero_prior``).
- A monotonic term ``mo(x)`` (B03): its coefficient is flat like every ``b``; its simplex ``dirichlet(1)`` (brms's
  ``eval_dirichlet`` default, the uniform distribution on the simplex), ``Dirichlet(1)``, on the class ``simo`` row of
  coefficient ``mox1``.
- A Gaussian process ``gp(x)`` (C03): its marginal SD ``sdgp`` takes the default of a group-level SD (brms's documentation
  says so), ``student_t(3, 0, scale)`` with ``lb`` 0 on the class row; its length-scale ``lscale`` an inverse gamma tuned to
  the process's inputs on its coefficient row (``gpx``), the class row empty with ``lb`` 0: ``default_lscale``.
- An autocorrelation term (C04, ``ar()``, ``ma()``, ``arma()``): its ``ar`` and ``ma`` flat (brms's classes ``ar`` and
  ``ma``, no bounds in the regression formulation), ``Flat()``, one class row each.
- An auxiliary parameter with its own formula (B01, ``sigma ~ z``; brms's ``def_dpar_prior`` under the formula's log link):
  its intercept ``student_t(3, 0, scale)`` with brms's scale of the response read on the log scale for ``sigma``
  (``max(2.5, round(mad(log y), 1))``, zeros read as 0.1, 2.5 when a response is negative) and ``student_t(3, 0, 2.5)``
  for ``shape`` and ``phi`` (``FamilySpec.default_distributional_intercept``); its coefficients flat; no class row of
  the parameter itself.

The data-dependent numbers are R's arithmetic, transcribed here and checked against R itself: ``median``, ``mad``
(constant 1.4826) and ``mean`` (R's two-pass mean on doubles), ``round(x, 1)`` (R 4's algorithm, which picks the
nearer of the two one-decimal candidates in double arithmetic) and ``as.character`` (``r_number``: 15 significant
digits chosen as R's ``formatReal`` does, so that ``-0.978004218445523`` prints as brms prints it). Logs use the C
library's ``log``, as R does.

``prior_table`` renders any ``Priors`` in brms's table form (the rows of ``default_prior()``: prior, class, coef, group,
resp, dpar, nlpar, lb, ub, tag, source), sorted as brms sorts them; the oracle gate G6 compares it with the table every
frozen fixture records.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from autobayes.priors import (LKJ, Dirichlet, Flat, HalfNormal, HalfStudentT, InverseGamma, Prior, PriorError, Priors, StudentT,
                              brms_prior)

BRMS_VERSION = "2.23.0"          # the brms whose default_prior() this reproduces (the oracle fixtures' meta.brms)
DF = 3.0                         # the degrees of freedom of brms's student_t(3, ., .) defaults
SCALE = 2.5                      # brms's smallest default scale
COLUMNS = ("prior", "class", "coef", "group", "resp", "dpar", "nlpar", "lb", "ub", "tag", "source")   # brms's table


# -- R's arithmetic ---------------------------------------------------------------------------------------------------
def r_round(x: float, digits: int = 1) -> float:
    """R 4's ``round(x, digits)`` for ``digits >= 0``: of the two candidates ``floor(x 10^d) / 10^d`` and
    ``ceiling(x 10^d) / 10^d`` the nearer to ``x`` in double arithmetic, the even one on a tie; values with more than
    15 significant digits before the cut are returned unchanged."""
    x = float(x)
    if math.isnan(x) or math.isinf(x) or x == 0.0:
        return x
    sgn = 1.0
    if x < 0.0:
        sgn, x = -1.0, -x
    if digits > 0 and math.log10(x) + digits > 15:
        return sgn * x
    pow10 = 1.0
    for _ in range(digits):
        pow10 *= 10.0
    x10 = pow10 * x
    i10 = math.floor(x10)
    xd, xu = i10 / pow10, math.ceil(x10) / pow10
    du, dd = xu - x, x - xd
    return sgn * (xu if (du < dd or (du == dd and math.fmod(i10, 2.0) == 1.0)) else xd)


def r_mean(values: Sequence[float]) -> float:
    """R's ``mean()`` of a double vector: the sum in order, divided by n, then corrected by the mean of the
    residuals (R's second pass), in double arithmetic (this R has no extended long double)."""
    xs = [float(v) for v in values]
    s = 0.0
    for v in xs:
        s += v
    s /= len(xs)
    if math.isfinite(s):
        t = 0.0
        for v in xs:
            t += v - s
        s += t / len(xs)
    return s


def r_median(values: Sequence[float]) -> float:
    """R's ``median()``: the middle value, or R's mean of the two middle values; NaN when any value is NaN."""
    xs = [float(v) for v in values]
    if any(math.isnan(v) for v in xs):
        return math.nan
    xs.sort()
    n = len(xs)
    half = (n + 1) // 2
    return xs[half - 1] if n % 2 == 1 else r_mean(xs[half - 1:half + 1])


def r_mad(values: Sequence[float]) -> float:
    """R's ``mad()``: ``1.4826 * median(|x - median(x)|)``."""
    xs = [float(v) for v in values]
    center = r_median(xs)
    return 1.4826 * r_median([abs(v - center) for v in xs])


_POW10 = [10.0 ** k for k in range(23)]                  # exact powers of ten up to 1e22


def _r_scientific(x: float) -> tuple[bool, int, int, bool]:
    """R's ``scientific()`` (format.c) at 15 digits: sign, decimal exponent, significant digits, and whether rounding
    widens the number, with R's scaling of ``|x|`` to a 15-digit integer."""
    neg = x < 0.0
    r = -x if neg else x
    kp = int(math.floor(math.log10(r))) - 15 + 1
    r_prec = r
    if abs(kp) <= 22:
        if kp > 0:
            r_prec /= _POW10[kp]
        elif kp < 0:
            r_prec *= _POW10[-kp]
    elif kp <= -308:
        r_prec = (r * 1e303) / math.pow(10.0, kp + 303)
    else:
        r_prec /= math.pow(10.0, kp)
    if r_prec < _POW10[14]:
        r_prec *= 10.0
        kp -= 1
    alpha = float(round(r_prec))                           # nearbyint: half to even
    nsig = 15
    for _ in range(15):
        alpha /= 10.0
        if alpha == math.floor(alpha):
            nsig -= 1
        else:
            break
    if nsig == 0:
        nsig, kp = 1, kp + 1
    kpower = kp + 15 - 1
    rgt = min(max(15 - kpower, 0), 22)
    widens = 0 < kpower <= 22 and r < _POW10[kpower] - 0.5 / _POW10[rgt]
    return neg, kpower, nsig, widens


def r_number(x: float) -> str:
    """``as.character(x)`` for one double, as R prints it in brms's prior strings (``sargs``): 15 significant digits,
    fixed or scientific notation by R's width rule, trailing zeros dropped (``3``, ``0.9``, ``-0.978004218445523``,
    ``1e+05``)."""
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == 0.0:
        return "0"
    neg, kpower, nsig, widens = _r_scientific(x)
    left = kpower + 1 - (1 if widens else 0)
    rgt = max(0, nsig - left)
    width_fixed = int(neg) + (1 if left <= 0 else left) + rgt + (1 if rgt else 0)
    d = nsig - 1 if nsig > 1 else 0
    width_sci = int(neg) + (1 if d > 0 else 0) + d + 4 + (2 if (kpower >= 100 or kpower <= -99) else 1)
    if width_fixed <= width_sci:
        s = f"{x:.{rgt}f}"
        return s.rstrip("0").rstrip(".") if "." in s else s
    mant, exp = f"{x:.{d}e}".split("e")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    return f"{mant}e{exp[0]}{exp[1:].lstrip('0').rjust(2, '0')}"


# -- brms's data-dependent location and scale -------------------------------------------------------------------------
def _on_link(y: Sequence[float], link: str) -> list[float]:
    """The response as brms's ``def_scale_prior`` reads it: as given under ``"identity"``; under ``"log"`` with zeros
    read as 0.1 (``ifelse(y == 0, y + 0.1, y)``) and logged (a negative value is NaN, as in R)."""
    if link == "identity":
        return [float(v) for v in y]
    if link == "log":
        out = []
        for v in y:
            v = 0.1 if float(v) == 0.0 else float(v)
            out.append(math.log(v) if v > 0 else math.nan)
        return out
    raise ValueError(f"brms reads the response for its default priors through 'identity' or 'log', not {link!r}")


def location_scale(y: Sequence[float], link: str | None, offset: Sequence[float] | None = None) -> tuple[float, float]:
    """brms's ``def_scale_prior`` arithmetic: the location and scale of the default Intercept prior
    ``student_t(3, location, scale)`` (the scale is also that of the default ``sd`` and ``sigma`` priors). ``link`` is
    the family's ``default_prior_link``; ``None`` gives brms's constants 0 and 2.5. ``offset`` is the offset term's
    values, whose mean brms subtracts from the location (not rounded)."""
    location, scale = 0.0, SCALE
    if link is None:
        return location, scale
    v = _on_link(y, link)
    scale_y = r_round(r_mad(v), 1)
    if math.isfinite(scale_y):
        scale = max(scale, scale_y)
    location_y = r_round(r_median(v), 1)
    if math.isfinite(location_y):
        location = location_y
    if offset is not None and len(offset):
        mean_offset = r_mean(offset)
        if math.isfinite(mean_offset):
            location = location - mean_offset
    return (0.0 if location == 0 else location), scale          # R's -0 (round(-0.04, 1)) prints as 0: the same prior


def _offset_values(spec, data: pd.DataFrame) -> list[float] | None:
    """The offset term's values per row as brms computes them (R's ``log`` of the column for ``offset(log(col))``)."""
    if spec.offset is None:
        return None
    col = [float(v) for v in data[spec.offset.column].to_numpy(dtype=float)]
    return [math.log(v) for v in col] if spec.offset.log else col


def _default(prior: Prior) -> Prior:
    return replace(prior, source="default")


def default_lscale(gp) -> Prior:
    """brms's default prior of a Gaussian process's length-scale (C03, docs/gaussian-processes.md), on its scaled inputs
    (``gp`` a ``formula.GaussianProcess`` with its data seen): the inverse gamma with 1% of its mass below the smallest
    distance between distinct scaled inputs, floored at 0.01, and 1% above the largest (1), solved exactly
    (``gaussian_process.inverse_gamma_tails``) and each parameter rounded to 6 decimals as brms prints it; with two distinct
    inputs the two distances coincide, no inverse gamma has those tails, and brms's documented fallback, the half-normal
    ``normal(0, 0.5)``, is returned. Where brms's own iterative solution sits just below a rounding boundary that the exact
    one crosses, the sixth decimal differs by one (divergence D67)."""
    from autobayes.gaussian_process import DIGITS, FALLBACK_SCALE, FLOOR, TAIL, inverse_gamma_tails, smallest_distance

    x = gp.scaled_inputs
    if x.size == 2:
        return HalfNormal(FALLBACK_SCALE)
    a, b = inverse_gamma_tails(max(smallest_distance(x), FLOOR), float(x.max() - x.min()), TAIL)
    return InverseGamma(r_round(a, DIGITS), r_round(b, DIGITS))


def default_priors(formula: str | list[str], data: pd.DataFrame, *, family: str = "gaussian", categorical: Mapping[str, object] | None = None,
                   nu: float | None = None, link: str | None = None, center_predictors: bool = True,
                   families: list[str] | None = None, order: str | bool | None = None,
                   rescor: bool | None = None, nl: bool = False, imputed: tuple[str, ...] = ()) -> Priors:
    """brms's default priors for ``formula`` on ``data`` under ``family`` (brms 2.23's ``default_prior()``), as a
    ``Priors`` the constructors accept: ``ab.gaussian(f, df, ab.default_priors(f, df))``. ``formula`` may be a list with a
    distributional formula (B01, ``["y ~ x", "sigma ~ z"]``), whose priors are its intercept's and coefficients'.

    ``categorical``, ``link`` and ``center_predictors`` are the constructor's own and say which model the defaults are
    for (without centring, brms's intercept is a flat ``b`` coefficient). ``nu`` is the Student-t degrees of freedom: left
    ``None`` the model learns them and they get brms's ``gamma(2, 0.1)``; a fixed value gets no prior. The data-dependent
    location and scale are computed from the response as brms computes them (module docstring). Every prior is marked
    ``source="default"``. Refused: a family or link the library does not have, a formula outside the supported subset
    (the same errors as the constructors), and a family whose parameter has no recorded brms default.

    A mixture (C05, ``family="mixture"`` with ``families=[...]`` and ``order`` as ``ab.mixture`` takes them): one ``Priors``
    per component, the component family's own defaults for the same formula and data (brms's ``def_dpar_prior`` of
    ``mu<k>`` and ``sigma<k>`` reads the response through the component's link: a Gaussian component's Intercept
    ``student_t(3, location, scale)``, flat ``b`` and ``sigma<k>`` ``student_t(3, 0, scale)``), and ``dirichlet(1)`` on
    ``theta``.

    A multivariate formula
    (C01, ``mvbind(y, y2) ~ ...`` with ``rescor=True`` or ``False``, docs/multivariate.md) gets each response's defaults under
    ``resp``, ``lkj(1)`` on a correlated term once for all responses and on the residual correlation (``rescor=True``). A list of formulas
    with ``mi()`` (C06, ``["y ~ mi(xm) + w", "xm | mi() ~ w"]`` with ``rescor=False``) is a multivariate model too: each
    response's defaults, a response with missing values read as brms reads it (its median and MAD are NA there, so location 0 and
    scale 2.5), and an ``mi()`` term's coefficient flat like every ``b``; ``imputed`` is set for those responses' formulas (the
    responses with ``| mi()``, ``formula.check_imputation``). An ``me(x, sdx)`` term (C06) gets brms's flat ``meanme`` and
    ``sdme`` (``Flat()``, on the positive half line for ``sdme``).

    ``nl=True`` gives the defaults of a non-linear model (C02, ``nonlinear.nonlinear_default_priors``, docs/nonlinear.md),
    which takes neither a mixture's ``families``/``order`` nor a multivariate formula or ``rescor`` (divergence D62)."""
    from autobayes.families import FAMILIES, get_family
    from autobayes.formula import check_imputation, parse_formula
    from autobayes.multivariate import is_multivariate_formula, multivariate_default_priors

    if nl:                                                            # a non-linear model (C02): its parameters' defaults; it refuses
        from autobayes.nonlinear import nonlinear_default_priors     # a mixture's and a multivariate model's arguments itself

        return nonlinear_default_priors(formula, data, family=family, categorical=categorical, nu=nu, link=link,
                                        families=families, order=order, rescor=rescor)
    if rescor is not None or is_multivariate_formula(formula):       # C01: each response's defaults, and rescor's
        return multivariate_default_priors(formula, data, family=family, categorical=categorical, nu=nu, link=link,
                                           center_predictors=center_predictors, rescor=rescor)

    fam = get_family(family, link)                                    # an unknown family, or a withheld link, is refused here
    learned = {}
    if "nu" in fam.fixed_parameters:
        if nu is None and "nu" in fam.learnable_parameters:           # learned under brms's default prior
            learned["nu"] = fam.default_learned_priors.get("nu")
        else:
            fam.fixed_parameters["nu"](nu)                            # a fixed nu, checked as the constructor checks it: no prior
    elif nu is not None:
        accepting = [name for name, other in FAMILIES.items() if "nu" in other.fixed_parameters]
        raise ValueError(f"nu applies only to the {' or '.join(accepting)} family")
    if fam.mixture:                                                   # C05: each component's defaults, and dirichlet(1) on theta
        from autobayes.families.mixture import check_mixture

        components = check_mixture(formula, families, order, None, center_predictors)
        spec = parse_formula(formula, data, family, categorical)      # the mixture's own refusals (group terms, offsets, ...)
        priors = Priors(components=[default_priors(formula, data, family=name, categorical=categorical) for name in components],
                        theta=_default(Dirichlet(1)))
        priors.validate(spec, fam)
        return priors
    if families is not None or order is not None:
        raise ValueError(f"families= and order= are a mixture's (family='mixture'); the {fam.name} family takes neither")
    spec = parse_formula(formula, data, family, categorical)
    check_imputation(spec, imputed)                                   # C06: mi() belongs to a multivariate model
    # B05: brms never reads an ordinal or categorical response for its defaults (no identity or log link: 0 and 2.5), and
    # its labels need not be numbers
    y = np.asarray(data[spec.response], dtype=float) if fam.response_categories is None else np.zeros(len(data))
    location, scale = location_scale(y, fam.default_prior_link, _offset_values(spec, data))
    intercept = None
    if spec.intercept:                                                # the cumulative thresholds keep brms's Intercept class (B05)
        intercept = StudentT(DF, location, scale) if center_predictors or fam.thresholds else Flat()
    sd = {}
    for t in spec.group_terms:
        sd[t.factor] = HalfStudentT(DF, scale) if not t.slopes else {c: HalfStudentT(DF, scale) for c in t.coefs}
    cor = {t.factor: LKJ(1) for t in spec.correlated_terms}
    cor.update({s.factor: LKJ(1) for s in spec.shared_ids})          # D08: a correlation ID shared with a distributional formula
    auxiliary = {}
    if fam.scale_parameter in spec.dpar_names:                       # its own formula (B01): brms's def_dpar_prior, log link
        ds = spec.dpar(fam.scale_parameter)
        if fam.default_distributional_intercept is None:
            raise PriorError(f"no brms default prior is recorded for the intercept of {fam.scale_parameter}'s formula in the "
                             f"{fam.name} family; give the priors explicitly")
        _, log_scale = location_scale(y, "log")                      # brms's def_scale_prior with the formula's log link
        auxiliary[fam.scale_parameter] = {**({"intercept": fam.default_distributional_intercept(log_scale)} if ds.intercept else {}),
                                          **({"b": Flat()} if ds.pop_terms else {}),
                                          # D08: a shared ID's terms: brms's def_scale_prior of the parameter, its log link
                                          **({"sd": {t.factor: HalfStudentT(DF, log_scale) for t in ds.group_terms}} if ds.group_terms else {})}
    elif fam.scale_parameter is not None:
        if fam.default_scale_prior is None:
            raise PriorError(f"no brms default prior is recorded for the {fam.name} family's {fam.scale_parameter}; "
                             f"give the priors explicitly (ab.Priors(..., {fam.scale_parameter}=...))")
        auxiliary[fam.scale_parameter] = fam.default_scale_prior(scale)
    if fam.zero_parameter is not None:                                # zi or hu (B06): beta(1, 1)
        if fam.default_zero_prior is None:
            raise PriorError(f"no brms default prior is recorded for the {fam.name} family's {fam.zero_parameter}; "
                             f"give the priors explicitly (ab.Priors(..., {fam.zero_parameter}=...))")
        auxiliary[fam.zero_parameter] = fam.default_zero_prior()
    for name, make in learned.items():
        if make is None:
            raise PriorError(f"no brms default prior is recorded for the {fam.name} family's learned {name}; "
                             f"give the priors explicitly (ab.Priors(..., {name}=...))")
        learned[name] = make()

    def mark(value):
        if value is None:
            return None
        if isinstance(value, Mapping):
            return {k: mark(v) for k, v in value.items()}
        return _default(value)

    simo = {t.operands[0]: Dirichlet(1) for t in spec.mo_terms}         # B03: brms's dirichlet(1) on each monotonic simplex
    sdgp = {g.column: HalfStudentT(DF, scale) for g in spec.gp_terms}   # C03: a group-level SD's default
    lscale = {g.column: default_lscale(g) for g in spec.gp_terms}      # C03: tuned to the process's inputs
    autocor = {part: Flat() for part, _ in spec.autocor.parts} if spec.autocor is not None else {}   # C04: flat ar and ma
    me = {t.operands[0]: Flat() for t in spec.me_terms}                  # C06: brms's flat meanme and sdme (sdme on (0, inf))
    priors = Priors(intercept=mark(intercept), b=mark(Flat()) if spec.pop_terms else None, sd=mark(sd) or None,
                    cor=mark(cor) or None, nu=mark(learned.get("nu")), simo=mark(simo) or None,
                    sdgp=mark(sdgp) or None, lscale=mark(lscale) or None, meanme=mark(me) or None, sdme=mark(me) or None,
                    **{k: mark(v) for k, v in auxiliary.items()}, **{k: mark(v) for k, v in autocor.items()})
    priors.validate(spec, fam)
    return priors


# -- brms's table ---------------------------------------------------------------------------------------------------
_RENAME = ((" ", ""), ("(", ""), (")", ""), ("[", ""), ("]", ""), (",", ""), ('"', ""), ("'", ""), ("?", ""),
           ("+", "P"), ("-", "M"), ("*", "MU"), ("/", "D"), ("^", "E"), ("=", "EQ"), ("$", "USD"))


def brms_rename(name: str) -> str:
    """brms's ``rename()`` of a model-matrix column name into a coefficient name (``log(xpos)`` -> ``logxpos``,
    ``I(x^2)`` -> ``IxE2``, ``(Intercept)`` -> ``Intercept``)."""
    for old, new in _RENAME:
        name = name.replace(old, new)
    return name


def _r_variable_order(spec) -> dict[str, int]:
    """The position of each main effect in the formula: the order in which R lists the variables of brms's
    fixed-effects formula (main effects before interactions, each in the order written) and so writes the operands of
    an interaction column, ``x:zq`` for ``x * z`` and ``zq:x`` for ``z * x`` or ``x:z + z + x`` (checked against
    brms 2.23's ``default_prior()``). Every interaction operand is a main effect here (D14)."""
    return {t.name: i for i, t in enumerate(spec.pop_terms) if t.kind in ("numeric", "categorical")}


def brms_coefficient_names(spec) -> dict[str, str]:
    """brms's name of each design column of ``spec`` (its ``b`` coefficient labels, ``rename()`` of R's model-matrix
    column names): ``x``; ``zq`` for level ``q`` of ``z``; ``x:zq``, ``x:w`` and ``zq:hh2`` with the operands in the order
    R writes them; ``logxpos`` and ``IxE2`` for transforms; and of each monotonic term (B03), keyed by its name, its
    coefficient's ``moord`` (brms's ``bsp_moord``). ``spec`` must have seen the data (the levels name columns)."""
    from autobayes.formula import interaction_cells

    if any(not c.levels for c in spec.categoricals):
        raise ValueError("brms's coefficient names need the categorical levels: parse the formula with the data")
    order = _r_variable_order(spec)
    out: dict[str, str] = {}
    for t in spec.pop_terms:
        if t.kind == "numeric":
            out[t.columns[0]] = brms_rename(t.name)
        elif t.is_special:                                            # mo(ord) -> moord, me(x, sdx) -> mexsdx, mi(x) -> mix (C06)
            out[t.name] = brms_rename(t.name)
        elif t.kind == "transform":
            out[t.columns[0]] = brms_rename(t.transform.term)
        elif t.kind == "categorical":
            for level, col in zip(t.levels, t.columns):
                out[col] = brms_rename(f"{t.name}{level}")
        else:
            ops = sorted(t.operands, key=lambda c: order.get(c, len(order)))
            if not t.categorical:                                     # x:w, one product column
                out[t.columns[0]] = brms_rename(":".join(ops))
            elif len(t.categorical) == 1:                             # x:z, one column per non-reference level of z
                z = t.categorical[0]
                for level, col in zip(t.levels, t.columns):
                    out[col] = brms_rename(":".join(f"{z}{level}" if op == z else op for op in ops))
            else:                                                     # z:h, one column per cell
                zc, hc = (spec.categorical(c) for c in t.categorical)
                for (lz, lh), col in zip(interaction_cells(zc, hc), t.columns):
                    piece = {zc.name: f"{zc.name}{lz}", hc.name: f"{hc.name}{lh}"}
                    out[col] = brms_rename(":".join(piece[op] for op in ops))
    return out


def brms_dpar_names(categories) -> list[str]:
    """brms's names of the linear predictors of the categorical family's categories (B05): ``mu`` followed by the
    category's label with every character but letters and digits removed (brms's ``make_stan_names(paste0("mu", cats))``:
    R's ``make.names``, then its dots and underscores dropped), so ``c2`` is ``muc2`` and ``a b`` is ``muab``. Two categories
    that would get the same name are refused (brms makes the names unique first and refuses only what still collides)."""
    from autobayes.formula import FormulaError

    out = ["mu" + "".join(ch for ch in str(c) if ch.isalnum()) for c in categories]
    clash = sorted({n for n in out if out.count(n) > 1})
    if clash:
        which = [str(c) for c, n in zip(categories, out) if n in clash]
        raise FormulaError(f"the response categories {which} would share brms's parameter name {clash[0]!r} (brms keeps only the "
                           "letters and digits of a category in the name of its linear predictor); relabel them")
    return out


def prior_table(priors: Priors, spec, family="gaussian", *, center_predictors: bool = True,
                categories=None) -> list[dict[str, str]]:
    """``priors`` in brms's table form: one row per class, group and coefficient as brms's ``default_prior()`` lists
    them, with the columns ``COLUMNS`` (all strings), sorted as brms sorts them (by resp, dpar, nlpar, class, group, coef;
    C-locale order). A prior shared by every member of a class sits on the class row and the member rows are empty
    (they inherit it), as brms writes its defaults; otherwise the most general level shared by the members carries it
    (an ``sd`` prior per factor on the group row, a coefficient's own prior on its row). Priors are written in brms's
    syntax (``brms_prior``: ``HalfStudentT(3, 2.5)`` on ``sd`` is ``student_t(3, 0, 2.5)`` with ``lb`` 0; ``Flat()`` is
    the empty string); ``source`` is ``default`` for a prior from ``default_priors`` and ``user`` otherwise.
    ``spec`` is the model's parsed formula with its data seen (``model.spec``); without centring the intercept is
    brms's ``b`` coefficient ``Intercept``. The ordinal and categorical families (B05) need the response's ``categories``
    (``Model.prior_table()`` passes them): the cumulative thresholds are brms's class ``Intercept`` with one row per
    threshold (coef ``1``, ..., ``K - 1``), centred or not; the categorical family repeats the ``Intercept`` and ``b`` rows
    for each non-reference category under its ``dpar`` (``muc2``, ...), with the one prior the categories share."""
    from autobayes.families import FamilySpec, get_family

    if getattr(spec, "responses", None):                              # C01: a multivariate model, one block of rows per response
        from autobayes.multivariate import multivariate_prior_table

        return multivariate_prior_table(priors, spec, family, center_predictors=center_predictors)
    fam = family if isinstance(family, FamilySpec) else get_family(family)
    if getattr(spec, "is_nonlinear", False):                          # a non-linear model (C02): its rows carry nlpar
        from autobayes.nonlinear import nonlinear_prior_table

        return nonlinear_prior_table(priors, spec, fam)
    priors.validate(spec, fam)
    if (fam.thresholds or fam.per_category) and not categories:
        raise PriorError(f"the {fam.name} family's prior table needs the response categories (Model.prior_table() passes them)")
    rows: list[dict[str, str]] = []

    def add(prior, cls, *, coef="", group="", lb="", ub="", source=None, dpar=""):
        text = brms_prior(prior) if isinstance(prior, Prior) else ""
        src = source if source is not None else (prior.source if isinstance(prior, Prior) else "default")
        rows.append(dict(zip(COLUMNS, (text, cls, coef, group, "", dpar, "", lb, ub, "", src))))

    def same(ps):
        return all(p == ps[0] for p in ps)

    def add_coefficients(coefs, dpar=""):
        if coefs:
            ps = [p for _, p in coefs]
            if same(ps):
                add(ps[0], "b", dpar=dpar)
                for name, p in coefs:
                    add(None, "b", coef=name, source=p.source, dpar=dpar)
            else:
                add(None, "b", dpar=dpar)
                for name, p in coefs:
                    add(p, "b", coef=name, dpar=dpar)

    names = brms_coefficient_names(spec)
    if fam.mixture:                                                   # C05: each component's dpar mu<k> rows and class sigma<k>,
        for k, c in enumerate(priors.components, start=1):            # then theta's row (brms's default_prior() of a mixture)
            add(c.intercept, "Intercept", dpar=f"mu{k}")
            add_coefficients([(names[col], p) for col, p in c.coefficient_priors(spec).items()], dpar=f"mu{k}")
            add(c.sigma, f"sigma{k}", lb="0")
        add(priors.theta, "theta")
        rows.sort(key=lambda r: (r["resp"], r["dpar"], r["nlpar"], r["class"], r["group"], r["coef"], r["tag"]))
        return rows
    coefficient_priors = priors.coefficient_priors(spec)
    for dpar in (brms_dpar_names(categories[1:]) if fam.per_category else [""]):   # brms's mu<k>: one block per category (B05)
        coefs = [(names[col], p) for col, p in coefficient_priors.items()]
        if spec.intercept and (center_predictors or fam.thresholds):
            add(priors.intercept, "Intercept", dpar=dpar)
            if fam.thresholds:                                        # brms lists every threshold under the class row (B05)
                for k in range(1, len(categories)):
                    add(None, "Intercept", coef=str(k), source=priors.intercept.source)
        elif spec.intercept:
            coefs.insert(0, ("Intercept", priors.intercept))          # brms's uncentred intercept is a b coefficient
        add_coefficients(coefs, dpar=dpar)
    for ds in spec.dpars:                                             # an auxiliary parameter's own formula (B01): its dpar rows
        sub, dnames = priors.dpar_priors(spec, ds.response), brms_coefficient_names(ds)
        if ds.intercept:                                              # brms's centred Intercept_<dpar>, class Intercept
            add(sub.intercept, "Intercept", dpar=ds.response)
        add_coefficients([(dnames[col], p) for col, p in sub.coefficient_priors(ds).items()], dpar=ds.response)
        dsd = sub.sd_priors(ds) if ds.group_terms else {}             # D08: a shared ID's terms, class sd with dpar = <parameter>
        if dsd:
            ps = list(dsd.values())
            add(ps[0] if same(ps) else None, "sd", lb="0", dpar=ds.response, source=None if same(ps) else "default")
            for t in ds.group_terms:
                mine = [dsd[(t.factor, c)] for c in t.coefs]
                top = same(ps) or same(mine)
                add(None if same(ps) or not same(mine) else mine[0], "sd", group=t.factor, dpar=ds.response,
                    source=mine[0].source if same(ps) else None)
                for c, p in zip(t.coefs, mine):
                    add(None if top else p, "sd", coef=brms_rename(c), group=t.factor, dpar=ds.response, source=p.source if top else None)
    for column, prior in priors.simo_priors(spec).items():           # a monotonic term's simplex (B03): brms's class simo,
        add(prior, "simo", coef=f"{brms_rename(spec.monotonic(column).term)}1")   # coefficient mox1, no class row
    for cls, by_column in (("sdgp", priors.sdgp_priors(spec)), ("lscale", priors.lscale_priors(spec))):   # C03, coefficient gpx
        ps = list(by_column.values())
        if ps and cls == "sdgp" and same(ps):                         # one prior for every process: the class row carries it
            add(ps[0], cls, lb="0")
            for column, p in by_column.items():
                add(None, cls, coef=brms_rename(f"gp({column})"), source=p.source)
        elif ps:                                                      # lscale's default is tuned per process: its coefficient
            add(None, cls, lb="0")                                    # rows carry the priors, the class row is empty
            for column, p in by_column.items():
                add(p, cls, coef=brms_rename(f"gp({column})"))
    for part, prior in priors.autocor_priors(spec).items():          # an ARMA term's ar / ma (C04): a class row each, unbounded
        add(prior, part)
    for cls, lb in (("meanme", ""), ("sdme", "0")):                    # C06: an me() term's latent values, brms's class rows
        me = priors.me_priors(spec, cls)                               # (lb 0 on sdme's) and one coefficient row me<x> each
        if me:
            ps = list(me.values())
            add(ps[0] if same(ps) else None, cls, lb=lb, source=ps[0].source if same(ps) else "default")
            for x, p in me.items():
                if same(ps):
                    add(None, cls, coef=f"me{brms_rename(x)}", source=p.source)
                else:
                    add(p, cls, coef=f"me{brms_rename(x)}")
    if fam.scale_parameter is not None and fam.scale_parameter not in spec.dpar_names:
        add(getattr(priors, fam.scale_parameter), fam.scale_parameter, lb="0")
    if fam.zero_parameter is not None:                                # brms's real<lower=0,upper=1> zi / hu (B06)
        add(getattr(priors, fam.zero_parameter), fam.zero_parameter, lb="0", ub="1")
    if priors.nu is not None:
        add(priors.nu, "nu", lb=r_number(fam.learnable_parameters["nu"]))
    sd = priors.sd_priors(spec)
    if sd:
        ps = list(sd.values())
        if same(ps):
            add(ps[0], "sd", lb="0")
        else:
            add(None, "sd", lb="0")
        for t in spec.group_terms:
            mine = [sd[(t.factor, c)] for c in t.coefs]
            if same(ps):
                add(None, "sd", group=t.factor, source=mine[0].source)
                for c, p in zip(t.coefs, mine):
                    add(None, "sd", coef=brms_rename(c), group=t.factor, source=p.source)
            elif same(mine):
                add(mine[0], "sd", group=t.factor)
                for c, p in zip(t.coefs, mine):
                    add(None, "sd", coef=brms_rename(c), group=t.factor, source=p.source)
            else:
                add(None, "sd", group=t.factor)
                for c, p in zip(t.coefs, mine):
                    add(p, "sd", coef=brms_rename(c), group=t.factor)
    cor = priors.cor_priors(spec)
    if cor:
        ps = list(cor.values())
        add(ps[0] if same(ps) else None, "cor")
        for g, p in cor.items():
            if same(ps):
                add(None, "cor", group=g, source=p.source)
            else:
                add(p, "cor", group=g)
    rows.sort(key=lambda r: (r["resp"], r["dpar"], r["nlpar"], r["class"], r["group"], r["coef"], r["tag"]))
    return rows
