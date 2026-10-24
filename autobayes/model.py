"""The model: a thin, inspectable layer over a Bambi-built PyMC model.

The families of ``autobayes.families``, including Poisson and negative binomial counts, Gamma, lognormal, Beta and
Student-t responses (``nu`` fixed or learned). ``"gaussian"`` (M0): identity link, residual SD ``sigma``. ``"bernoulli"``
(M1): logit link, ``p = expit(eta)``, no residual scale. ``"binomial"`` (M2): logit link on
``successes | trials(n)``, counts. The Bernoulli and binomial families also take ``link="probit"`` (A12) and
``link="cloglog"`` (P6; docs/links.md), and the Poisson (``"sqrt"``), negative binomial (``"identity"``), Gamma (``"inverse"``,
``"identity"``) and Beta (``"probit"``, ``"cloglog"``) families brms's further links (E06). All share the right-hand side ``[1 +] x1 [+ x2 ...] [+ offset(log(col))]
[+ (1 | factor) | (1 + x || factor) | (1 + x | factor) ...]`` with numeric terms and, since M3a-1, treatment-coded categorical
terms declared with ``categorical={column: reference}``; an optional offset term (A06, docs/offset.md), added to
the linear predictor with no coefficient; centring; non-centred group effects (varying
intercepts and, with ``||``, independent varying slopes; with ``|``, slopes correlated with the intercept under an
LKJ prior, A02, docs/correlated-slopes.md) when a group term is present, and a population-only
model without one (M3c-1). The formula is a term model (``PopTerm``, ``GroupTerm`` in ``autobayes.formula``;
docs/term-model.md) and ``Design`` holds one ``GroupBlock`` per group-level coefficient. Bambi builds every
group-level coefficient as an independent non-centred term; for a correlated term ``Model`` then rewrites the
slopes' deterministics in the PyMC graph to brms's ``r = (diag(sd) L z)^T`` with a Cholesky factor ``L`` sampled on
Stan's ``cholesky_corr`` coordinates (``autobayes.correlation``). The left-hand side may carry brms's ``weights(col)``
next to the binomial's ``trials(n)`` (B02, docs/addition-terms.md): the family's likelihood is then wrapped so that each
row's log density is multiplied by its weight (``likelihoods.addition_family``); ``cens()`` and ``trunc()`` are
offered for the Gaussian family (accepted against two references since P6), and since E04 (docs/cens-trunc-families.md)
``cens(c, y2)``, ``cens()`` for the Weibull and lognormal families and ``trunc(ub = b)`` for the Poisson and negative binomial
families (``FamilySpec.tails``, ``autobayes.tails``). The family's auxiliary parameter
(``sigma``, ``shape``, ``phi``) may have its own formula (B01, docs/distributional.md): ``formula`` is then a list,
``["y ~ x + (1 | g)", "sigma ~ z"]``, and Bambi builds the parameter as a distributional component under a log link
(``Design.dpars`` holds its design, ``Model.dpar_names`` its PyMC names). A finite mixture of Gaussian components (C05,
docs/mixtures.md; ``ab.mixture(..., families=[...])``) has one linear predictor per component with ordered intercepts, a
``sigma<k>`` per component and the mixing proportions ``theta`` (``autobayes/families/mixture.py``). A Gaussian process ``gp(x)`` or ``gp(x, k = 10, c =
5/4)`` (C03, docs/gaussian-processes.md) is added to the parent's linear predictor while Bambi builds the model, as a
monotonic term is (``Design.gp`` holds its inputs, ``Model.gp_names`` its PyMC names).
brms's multi-membership term ``(1 | mm(g, g2, ...))`` and ``(1 | gr(g, by = gby))`` (D08, docs/multi-membership-and-by.md) are
added to the parent's linear predictor the same way, under Bambi's names of a varying intercept (``1|mmgg2_sigma``, ...); a
correlation ID shared with sigma's formula is Bambi's two components' terms correlated by the post-build edit of A02.
brms's autocorrelation terms ``ar()``, ``ma()`` and ``arma()`` in the regression formulation (C04, docs/autocorrelation.md)
are added to the Gaussian linear predictor last, on the residuals of every other term (``autobayes.autocor``;
``Design.autocor`` holds the series of the fitted rows). brms's ``sar(M, type = "lag" | "error")`` (F02b, docs/spatial-sar.md)
replaces the Gaussian likelihood with the joint SAR density of the responses (``autobayes.sar``; ``M`` comes in ``data2``,
``Design.sar`` holds it and its eigenvalues).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd

from collections.abc import Mapping

from autobayes.correlation import check_correlation_cholesky, cholesky_corr_free, cholesky_corr_rv, n_free
from autobayes.families import FAMILIES, FamilySpec, get_family
from autobayes.formula import (Categorical, FormulaError, ModelSpec, addition_data, bambi_group_name, binomial_counts, by_levels_of,
                               categorical_labels, check_imputation, factor_column, group_labels, interaction_cells,
                               measurement_values, missing_values, mm_weights, parse_formula, pop_term_design, sd_tag)
from autobayes.priors import AUXILIARY, PriorError, Priors, describe_prior

_bambi_group_name = bambi_group_name          # ``1|g`` for the varying intercept of ``g``, ``x|g`` for the slope of ``x``
_SYMMETRIC_HALF = frozenset({"HalfNormal", "HalfStudentT", "HalfCauchy"})   # D08: half priors of a distribution symmetric about 0
_AUTOCOR_MEANING = {"ar": "autoregressive", "ma": "moving-average"}   # an autocorrelation term's parts (C04)
_DPAR_INVERSE = {"log": "exp(eta_{})", "logit": "inv_logit(eta_{})", "identity": "eta_{}"}   # describe()'s value of a dpar (B01, E03)


def _prior_rv(name: str, prior, *, positive: bool = False):
    """A scalar PyMC variable named ``name`` under ``prior`` (its parameters as float64, which PyTensor keeps; C06): a
    location prior, a positive-support prior, or ``Flat()``, flat on the real line, or with ``positive`` on the positive half
    line (PyMC's ``HalfFlat``, brms's flat prior on a ``real<lower=0>``). Must be called inside a PyMC model context."""
    import pymc as pm

    dist = pm.HalfFlat if positive and prior.name == "Flat" else getattr(pm, prior.name)
    return dist(name, **{k: np.float64(v) for k, v in prior.params.items()})


@dataclass(frozen=True)
class GroupBlock:
    """The design block of one group-level coefficient ``c`` of grouping factor ``m`` (docs/term-model.md).

    ``levels`` are the factor's sorted fitted labels (brms's level order; ``N_m`` of them). ``Z`` is the
    ``(n, N_m)`` block: row ``i`` holds the coefficient's covariate value (one for the intercept, so the block
    is the one-hot indicator) in the column of row ``i``'s level, which is brms's ``Z_m_c[i]`` spread over
    its ``J_m[i]``. ``group_index`` is that level's 0-based index per row (brms ``J_m - 1``) and
    ``column_slice`` where the block sits in the concatenated ``Design.Z``.
    """
    levels: tuple[str, ...]
    Z: np.ndarray
    group_index: np.ndarray
    column_slice: slice
    # D08, a multi-membership term (1 | mm(g, g2, ...)): each member's 0-based level index per row (brms's J_m_k - 1) and its
    # weight per row (brms's W_m_k, scaled to sum to 1 on each row); Z is their weighted sum of indicators and group_index
    # the first member's index. Empty for every other term.
    memberships: tuple[np.ndarray, ...] = ()
    weights: tuple[np.ndarray, ...] = ()
    # D08, a term with gr(g, by = gby): the levels of gby, sorted (brms's Nby_m of them), and the 0-based by-level of each
    # level of g (brms's Jby_m - 1); the coefficient's SD is a vector over by_levels. Empty / None for every other term.
    by_levels: tuple[str, ...] = ()
    by_index: np.ndarray | None = None

    @property
    def J(self) -> int:
        """The number of levels (brms ``N_m``; this library has always called it ``J``)."""
        return len(self.levels)


@dataclass(frozen=True)
class DparDesign:
    """The design of one distributional parameter's own formula (B01, docs/distributional.md): brms's ``X_<dpar>`` without
    its intercept column. ``name`` is the parameter's brms name (``sigma``, ``shape``, ``phi``), ``columns`` its design
    labels (``ModelSpec.dpar(name).columns``), ``X_raw`` the columns as built, ``X_model`` what multiplies its coefficients
    (centred at ``x_means`` when ``centered``: the formula has an intercept, and brms and Bambi then centre its design on
    its own means, brms's ``Xc_<dpar>``), and ``link`` its link (``log``): the parameter is the inverse link of its linear
    predictor, one value per row."""
    name: str
    X_raw: np.ndarray
    X_model: np.ndarray
    x_means: np.ndarray
    columns: tuple[str, ...]
    centered: bool
    link: str
    # D08: the group blocks of the formula's group-level terms, which share a correlation ID with the response formula's
    # (``sigma ~ (1 |p| g)``), keyed (factor, coefficient) as ``Design.group_blocks``; empty for every other formula
    group_blocks: Mapping[tuple[str, str], GroupBlock] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class MonotonicBlock:
    """The block of one monotonic term ``mo(col)`` (B03, docs/monotonic.md): ``codes`` are brms's ``Xmo``, each fitted row's
    value coded ``0..D`` (``formula.Monotonic.codes``), ``D`` brms's ``Jmo``, the number of steps and the length of the
    simplex; the term adds ``b * D * sum(simplex[:codes_i])`` to row ``i``'s linear predictor."""
    term: str
    column: str
    codes: np.ndarray
    D: int


@dataclass(frozen=True)
class GPBlock:
    """The block of one Gaussian process ``gp(x)`` (C03, docs/gaussian-processes.md): ``index`` is each fitted row's 0-based
    position among the distinct inputs (brms's ``Jgp`` less 1), ``inputs`` the distinct inputs scaled by ``dmax``, in the order
    they first appear (brms's ``Xgp_prior``, and the exact GP's ``Xgp``), ``basis`` the approximate GP's ``(N, k)`` basis at
    them (brms's ``Xgp``; None for the exact GP) and ``slambda`` its frequencies (brms's ``slambda``). The term adds
    ``f[index]`` to the linear predictor, ``f`` the process's values at the distinct inputs."""
    term: str
    column: str
    index: np.ndarray
    inputs: np.ndarray
    basis: np.ndarray | None = None
    slambda: np.ndarray | None = None


@dataclass(frozen=True)
class MeasurementBlock:
    """The block of a measurement-error term ``me(x, sdx)`` (C06, docs/missing-and-measurement-error.md): ``Xn`` the noisy
    values of ``column`` and ``noise`` their known standard deviations (``noise_column``) on the fitted rows, brms's ``Xn_k``
    and ``noise_k``; the term adds ``b * Xme_i`` to row ``i``'s linear predictor, ``Xme = meanme + sdme * zme`` the latent
    values, and the model has ``Xn_i ~ Normal(Xme_i, noise_i)``."""
    term: str
    column: str
    noise_column: str
    Xn: np.ndarray
    noise: np.ndarray


@dataclass(frozen=True)
class MissingBlock:
    """A column whose missing values are parameters of the model (C06): the response of ``col | mi() ~ ...``, or the column
    of an ``mi(col)`` term. ``values`` holds the fitted rows with NaN at the missing ones (brms's ``Y_col`` holds Inf there),
    ``index`` the missing rows, 0-based (brms's ``Jmi_col - 1``; ``Nmi_col`` of them), whose values are the free variable
    ``Ymi_col`` of the model."""
    column: str
    values: np.ndarray
    index: np.ndarray

    @property
    def filled(self) -> np.ndarray:
        """``values`` with 0.0 at the missing rows: the placeholder the graph replaces with the missing values."""
        return np.where(np.isnan(self.values), 0.0, self.values)


@dataclass(frozen=True)
class Design:
    """What the model was actually fit to. All arrays are dense NumPy."""

    y: np.ndarray                 # (n,)
    X_raw: np.ndarray             # (n, P) design columns as given (numeric values, 0/1 dummies), no intercept
    X_model: np.ndarray           # (n, P) what multiplies the coefficients: centred if center_predictors
    x_means: np.ndarray           # (P,) column means of X_raw (zeros if not centred); a dummy's mean is its level's proportion
    predictors: tuple[str, ...]   # numeric predictor terms
    Z: np.ndarray                 # the group blocks side by side in formula order; (n, 0) without groups
    group_blocks: Mapping[tuple[str, str], GroupBlock]   # (factor, coefficient) -> block, in formula order
    centered: bool
    trials: np.ndarray | None = None   # binomial only: (n,) trials per fitted row; y holds the successes
    columns: tuple[str, ...] = ()      # design-column labels in order (``x``, ``z[b]``, ...); == predictors when numeric only
    categoricals: tuple[Categorical, ...] = ()
    offset: np.ndarray | None = None   # (n,) the offset term's values per fitted row (brms ``offsets``), or None
    weights: np.ndarray | None = None  # (n,) brms ``weights`` of a ``weights(col)`` term (B02), or None
    cens: np.ndarray | None = None     # (n,) brms ``cens`` of a ``cens(col)`` term: -1 left-, 0 un-, 1 right-censored (B02, parked); 2 interval-censored under ``cens(col, y2)`` (E04)
    rcens: np.ndarray | None = None    # (n,) brms ``rcens`` of ``cens(col, y2)``: the upper bound y2 on the interval-censored rows, 0 elsewhere (E04)
    lb: np.ndarray | None = None       # (n,) brms ``lb`` and ``ub`` of a ``trunc()`` term, -inf / inf on an open side (B02, parked)
    ub: np.ndarray | None = None
    dpars: Mapping[str, DparDesign] = field(default_factory=lambda: MappingProxyType({}))   # each distributional parameter's own design, by brms name (B01)
    monotonic: Mapping[str, MonotonicBlock] = field(default_factory=lambda: MappingProxyType({}))   # each monotonic term's block, by column (B03)
    gp: Mapping[str, GPBlock] = field(default_factory=lambda: MappingProxyType({}))   # each Gaussian process's block, by column (C03)
    autocor: object | None = None      # autocor.AutocorDesign: the series of an autocorrelation term on the fitted rows (C04), or None
    sar: object | None = None          # sar.SarDesign: a SAR term's weight matrix and its eigenvalues on the fitted rows (F02b), or None
    measurement_error: Mapping[str, MeasurementBlock] = field(default_factory=lambda: MappingProxyType({}))   # each me() term's block, by noisy column (C06)
    missing: Mapping[str, MissingBlock] = field(default_factory=lambda: MappingProxyType({}))   # each column whose missing values are parameters (C06)
    category_specific: Mapping[str, np.ndarray] = field(default_factory=lambda: MappingProxyType({}))   # each cs() term's column on the fitted rows (D04; brms's Xcs), by column

    def __post_init__(self):
        if not self.columns:
            object.__setattr__(self, "columns", tuple(self.predictors))

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def group_designs(self) -> dict[str, GroupBlock]:
        """The varying-intercept block of each factor: the ``(factor, "Intercept")`` entries of ``group_blocks``."""
        return {factor: block for (factor, coef), block in self.group_blocks.items() if coef == "Intercept"}

    def _single_group(self):
        if len(self.group_designs) > 1:
            raise ValueError("singular group design access is ambiguous with multiple grouping factors; use group_designs")
        return next(iter(self.group_designs.values()), None)

    @property
    def levels(self) -> tuple[str, ...]:
        block = self._single_group()
        return block.levels if block is not None else ()

    @property
    def group_index(self) -> np.ndarray:
        block = self._single_group()
        return block.group_index if block is not None else np.zeros(0, dtype=int)

    @property
    def J(self) -> int:
        return len(self.levels)

    @property
    def group_slices(self) -> dict[str, slice]:
        return {g: block.column_slice for g, block in self.group_designs.items()}


def _cells(spec: ModelSpec, t) -> list[tuple[str, str, str]]:
    """``describe()``'s cells of a categorical-by-categorical interaction: (cell label, level of the first operand, level of
    the second)."""
    return [(lvl, *pair) for lvl, pair in zip(t.levels, interaction_cells(*(spec.categorical(c) for c in t.categorical)))]


def _linear_terms(spec: ModelSpec, centered: bool, b: str = "b") -> list[str]:
    """``describe()``'s population-level terms of one formula's linear predictor, each column times its coefficient
    ``<b>_<term>``: the response formula's (``b``) or a distributional parameter's (``b_sigma``, B01), centred columns
    written with their means (``p_z[l]``, ``q_x:z[l]``, ``mean(x)``) when the formula is centred."""
    parts = []
    for t in spec.pop_terms:
        term = t.name
        if t.kind == "categorical":
            for lvl in t.levels:
                ind = f"1[{term}_i = {lvl}]"
                parts.append(f"({ind} - p_{term}[{lvl}]) * {b}_{term}[{lvl}]" if centered else f"{ind} * {b}_{term}[{lvl}]")
        elif t.kind == "interaction" and len(t.categorical) == 1:
            zn = t.categorical[0]
            xn = next(c for c in t.operands if c != zn)
            for lvl in t.levels:
                prod = f"x_{xn} * 1[{zn}_i = {lvl}]"
                parts.append(f"({prod} - q_{term}[{lvl}]) * {b}_{term}[{lvl}]" if centered else f"{prod} * {b}_{term}[{lvl}]")
        elif t.kind == "interaction" and not t.categorical:
            xn, wn = t.operands
            prod = f"x_{xn} * x_{wn}"
            parts.append(f"({prod} - q_{term}) * {b}_{term}" if centered else f"{prod} * {b}_{term}")
        elif t.kind == "interaction":
            zn, hn = t.operands
            for lvl, lz, lh in _cells(spec, t):
                prod = f"1[{zn}_i = {lz}] * 1[{hn}_i = {lh}]"
                parts.append(f"({prod} - p_{term}[{lvl}]) * {b}_{term}[{lvl}]" if centered else f"{prod} * {b}_{term}[{lvl}]")
        elif t.kind == "transform":
            value = t.transform.expression(f"{t.transform.column}_i")
            parts.append(f"({value} - mean({term})) * {b}_{term}" if centered else f"{value} * {b}_{term}")
        elif t.is_monotonic:                                 # B03: never centred, brms's bsp * mo(simo, Xmo)
            parts.append(f"{b}_{term} * {term}_i")
        elif t.kind == "measurement_error":                  # C06: never centred, brms's bsp * Xme
            parts.append(f"{b}_{term} * Xme_{t.operands[0]},i")
        elif t.kind == "missing":                            # C06: never centred, brms's bsp * Yl (observed or Ymi)
            parts.append(f"{b}_{term} * {t.operands[0]}_i")
        elif t.kind == "category_specific":                  # D04: one coefficient per threshold, in mu_i[k] (the family's lines)
            continue
        else:
            parts.append(f"(x_{term} - mean(x_{term})) * {b}_{term}" if centered else f"x_{term} * {b}_{term}")
    return parts


class Model:
    """``y ~ x [+ (1 | factor) ...]`` with explicit priors; Gaussian, Student-t, Poisson, Bernoulli or binomial likelihood.

    Attributes
    ----------
    family : str               a registered family name: "gaussian", "student", "poisson", "negbinomial", "gamma", "lognormal", "beta", "bernoulli", "binomial",
                               "zero_inflated_poisson", "hurdle_poisson", "cumulative", "categorical", "mixture", and (D01) "weibull",
                               "exponential", "skew_normal", "shifted_lognormal", "exgaussian", and (D02) "beta_binomial", "geometric",
                               "negbinomial2", and (E02) "inverse_gaussian", "von_mises", "asym_laplace", "frechet", "gen_extreme_value"
    categories : tuple[str]    the response's categories in order (the cumulative and categorical families, B05); () otherwise
    hurdle_category : str      hurdle_cumulative's hurdle category, brms's Y = 0 (E01); None otherwise
    family_spec : FamilySpec   the specification of that family under the model's link (autobayes/families/; the registered spec unless ``link`` names another of the family's links, or, E02, ``quantile`` fixes the asymmetric Laplace quantile)
    quantile : float | None    E02: the asymmetric Laplace family's fixed quantile, or None when it is learned (that family only)
    link : str                 the link: the family's own, or the ``link`` given ("probit" for the Bernoulli and binomial families)
    spec : ModelSpec
    priors : Priors
    bambi_model : bambi.Model  (already built)
    pymc_model : pymc.Model    the graph that is sampled; inspect it directly
    design : Design            design matrices and group coding

    ``formula`` is one formula, or a list of them (B01, docs/distributional.md): the response formula, then a formula of
    the family's auxiliary parameter (``["y ~ x + (1 | g)", "sigma ~ z"]``, brms's ``bf(y ~ x + (1 | g), sigma ~ z)``),
    whose priors are ``Priors(sigma={"intercept": ..., "b": ...})``.
    """

    def __init__(self, formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
                 family: str = "gaussian", center_predictors: bool = True,
                 categorical: Mapping[str, object] | None = None, nu: float | None = None, link: str | None = None,
                 families: list[str] | None = None, order: str | bool | None = None, imputed: tuple[str, ...] = (),
                 quantile: float | None = None, data2: Mapping[str, object] | None = None):
        """``data2`` (F02b) holds the matrices the formula's terms read by name, brms's ``data2``: ``sar(M, ...)``'s weight
        matrix over the fitted rows, ``data2={"M": matrix}`` (a NumPy array or a DataFrame, in data order); an entry no term
        reads is refused. ``imputed`` (C06) is set by ``MultivariateModel`` for the models of its responses: the responses of its
        ``col | mi() ~ ...`` formulas, whose missing values are parameters (``formula.check_imputation``); empty otherwise.
        ``quantile`` (E02) fixes the asymmetric Laplace family's quantile (brms's ``bf(y ~ ..., quantile = p)``): the model's
        spec is then the family's with that parameter fixed (``FamilySpec.with_fixed``) and ``self.quantile`` its value."""
        import bambi as bmb

        fam = get_family(family)
        if link is not None:                                   # the family's spec under another of its links (A12)
            try:
                fam = fam.with_link(link)
            except ValueError as e:
                offering = [name for name, other in FAMILIES.items() if other.links]
                hint = (" (link= applies to the families with more than one link: "      # each with its own links (A12, D88)
                        + ", ".join(f"{name} {sorted([FAMILIES[name].link, *FAMILIES[name].links])}" for name in offering)
                        + ")") if not fam.links else ""
                raise ValueError(f"{e}{hint}") from None
        if quantile is not None or fam.fixable:                # E02: a parameter brms's formula may fix (bf(quantile = p))
            fam = fam.with_fixed("quantile", quantile)
            self.quantile = fam.fixed_parameters["quantile"](quantile) if quantile is not None else None
        if "nu" in fam.fixed_parameters:
            if nu is not None and priors.nu is not None:
                raise ValueError(f"nu={nu!r} fixes the degrees of freedom and Priors(nu={priors.nu}) learns them: pass one "
                                 "or the other (nu=None with the prior learns nu, as brms's student() does)")
            if nu is None and "nu" in fam.learnable_parameters:   # learned under its own prior (A13)
                if priors.nu is None:
                    raise PriorError(f"the {family} family needs nu: a fixed nu > {fam.learnable_parameters['nu']:g} (nu=4), or "
                                     "nu=None with a positive-support prior Priors(nu=...) to learn it")
                self.nu = None
            else:
                self.nu = fam.fixed_parameters["nu"](nu)
        elif nu is not None:
            accepting = [name for name, other in FAMILIES.items() if "nu" in other.fixed_parameters]
            raise ValueError(f"nu applies only to the {' or '.join(accepting)} family")
        self.family = family
        self.family_spec: FamilySpec = fam
        self.priors = priors
        self.center_predictors = bool(center_predictors)
        self.dropped_rows: tuple[int, ...] = ()     # binomial: positional indices of zero-trial rows
        self.warnings: list[str] = []
        self.components: tuple[str, ...] = ()       # a mixture's component families, in order (C05, docs/mixtures.md)
        if fam.mixture:
            from autobayes.families.mixture import check_mixture
            self.components = check_mixture(formula, families, order, priors, self.center_predictors)
        elif families is not None or order is not None:
            raise ValueError(f"families= and order= are a mixture's (ab.mixture, docs/mixtures.md); the {family} family takes neither")

        spec0 = parse_formula(formula, None, family, categorical)
        if fam is not get_family(family) and fam.check_formula is not None:   # a link's own refusals (E06, divergence D133)
            fam.check_formula(spec0)
        self.imputed: tuple[str, ...] = tuple(imputed)        # C06: the columns a multivariate model imputes by their own formula
        check_imputation(spec0, self.imputed)
        self.effective_center_predictors = self.center_predictors and spec0.intercept
        centred = [d for d in spec0.dpars if d.intercept and d.pop_terms]   # a distributional formula brms centres (B01)
        if centred and not self.center_predictors:
            raise FormulaError(
                f"center_predictors=False with the formula '{centred[0].formula}' is not supported: brms centres a "
                "distributional parameter's own formula on its means even under bf(..., center = FALSE), and Bambi centres "
                "every formula that has an intercept or none; keep center_predictors=True, or write that formula without an "
                "intercept, '0 + ...' (docs/distributional.md, divergence D34)"
            )
        priors.validate(spec0, fam)                     # structure now; per-level coverage once levels are known
        if self.shrinkage_names and not self.center_predictors:   # D07: brms's b of an uncentred design holds the intercept
            raise PriorError(f"{priors.b.name}() on b with center_predictors=False is not offered: no frozen case verifies it "
                             "(docs/shrinkage-priors.md, divergence D102)")
        operands = [c for t in spec0.group_terms for c in t.data_columns]   # the columns the grouping factors are read from
        cols = ([spec0.response, *spec0.pop_columns] + operands
                + ([spec0.trials] if spec0.trials else []) + ([spec0.offset.column] if spec0.offset is not None else [])
                + [c for a in spec0.additions for c in a.columns]          # weights() and cens() columns (B02)
                + [c for d in spec0.dpars for c in d.pop_columns]          # the columns distributional formulas read (B01)
                + [c for d in spec0.dpars for t in d.group_terms for c in t.data_columns])   # and their shared-ID terms (D08)
        cols += list(spec0.autocor.columns) if spec0.autocor is not None else []   # an autocorrelation term's time and series (C04)
        cols = list(dict.fromkeys(cols))                        # the offset column may also be a predictor
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise FormulaError(f"columns not in data: {missing}")
        df = data[cols].copy().reset_index(drop=True)
        if fam.has_trials:
            k, n = binomial_counts(spec0, df)               # validated before any cast, all rows
            zero = np.flatnonzero(n == 0)
            self.dropped_rows = tuple(int(i) for i in zero)
            df[spec0.response] = k
            df[spec0.trials] = n
            if zero.size:
                for cat in spec0.categoricals:              # levels are those of the fitted rows
                    labels, _ = categorical_labels(f"categorical predictor {cat.name!r}", df[cat.name])
                    kept = set(labels[n > 0].tolist())
                    only_dropped = sorted(set(labels[zero].tolist()) - kept)
                    if cat.reference in only_dropped:
                        raise FormulaError(
                            f"reference level {cat.reference!r} of {cat.name!r} occurs only in zero-trial rows, "
                            f"which are dropped before the design is built; observed levels among the fitted "
                            f"rows: {sorted(kept)}"
                        )
                    if only_dropped:
                        self.warnings.append(
                            f"level(s) {only_dropped} of {cat.name!r} occur only in the {zero.size} dropped "
                            "zero-trial row(s) and are not levels of the fitted model; predictions for them are refused."
                        )
                df = df.drop(index=zero).reset_index(drop=True)
        self.spec: ModelSpec = parse_formula(formula, df, family, categorical)   # levels over the fitted rows
        for s in (self.spec, *self.spec.dpars):              # a distributional formula's terms too (B01)
            for t in s.pop_terms:                            # a cell of z:h with no fitted rows has a zero column (brms keeps it too)
                if t.kind == "interaction" and len(t.categorical) == 2:
                    empty = [lvl for lvl, col in zip(t.levels, pop_term_design(t, df, s.categoricals).T) if not col.any()]
                    if empty:
                        self.warnings.append(
                            f"cell(s) {empty} of {t.name!r}{'' if s is self.spec else ' in the formula of ' + s.response} have no "
                            "fitted rows: their coefficients are informed by the prior alone (the design column is zero, as in brms)."
                        )
        for m in self.spec.monotonics:                       # a monotonic predictor's values no fitted row takes (B03)
            self.warnings.extend(m.notes(df[m.column]))
        for t in self.spec.group_terms:                      # labels as text; a derived factor g:h gets its own column g_h
            if t.is_mm:                                      # D08: a multi-membership term's labels are its member columns
                for c in t.members:
                    df[c] = group_labels(c, df)
                continue
            df[t.column] = group_labels(t.factor, df)
            if t.by is not None:                             # D08: and gr(by = )'s by column, as text
                df[t.by] = group_labels(t.by, df)
        for t in (*self.spec.transform_terms, *(t for d in self.spec.dpars for t in d.transform_terms)):
            df[t.name] = t.transform.values(df)              # the evaluated column Bambi is handed, under the term's own name (A05)
        self.categories: tuple[str, ...] = ()               # the response's categories, in order (B05: cumulative, categorical)
        self.hurdle_category: str | None = None              # E01: hurdle_cumulative's hurdle category, brms's Y = 0
        if fam.hurdle_category is not None:                  # read before the column becomes brms's codes
            self.hurdle_category = fam.hurdle_category(self.spec, df[self.spec.response])
        if fam.response_categories is not None:              # the response is fitted as its 1-based category, brms's Y
            self.categories, codes, notes = fam.response_categories(self.spec, df[self.spec.response])
            df[self.spec.response] = codes
            self.warnings.extend(notes)
        if fam.response_dtype is not None:
            df[self.spec.response] = df[self.spec.response].astype(fam.response_dtype)
        self.warnings.extend(fam.response_warnings(self.spec, df))   # separation, all-zero counts
        self.original_labels: dict[str, np.ndarray] = {}     # labels of the SUPPLIED column, fitted rows, before any recoding
        for cat in self.spec.all_categoricals:               # every formula's, each column once (B01)
            # An ORDERED Categorical with the reference first: formulae keeps this order (it sorts
            # anything else) and takes the first category as the treatment reference; built from
            # the observed labels only, so an unused declared category never becomes a zero column.
            labels, _ = categorical_labels(f"categorical predictor {cat.name!r}", df[cat.name])
            self.original_labels[cat.name] = labels
            df[cat.name] = pd.Categorical(labels, categories=list(cat.categories), ordered=True)
            if cat.unused:
                self.warnings.append(
                    f"declared categories {list(cat.unused)} of {cat.name!r} do not occur in the fitted rows: "
                    "they get no coefficient, and predictions for them are refused."
                )
        self.data = df
        self._autocor_design = None                          # C04: the series of the fitted rows, brms's sorted Stan rows
        if self.spec.autocor is not None:
            from autobayes.autocor import series_layout
            self._autocor_design = series_layout(self.spec.autocor, self.data)
            longest = self._autocor_design.T                 # lag k needs a series of more than k rows
            unread = [f"{part}[{k}]" for part, order in self.spec.autocor.parts for k in range(max(longest, 1), order + 1)]
            if unread:
                self.warnings.append(
                    f"no series of {self.spec.autocor.term} has more than {longest} row(s): {unread} multiply no lagged residual and "
                    "are informed by their prior alone (brms keeps them too)"
                )
        from autobayes.sar import check_data2, sar_design

        self._sar_design = sar_design(self.spec.sar, data2, len(self.data)) if self.spec.sar is not None else None
        check_data2(self.spec, data2)                         # F02b: data2 holds what sar() reads, and nothing else

        bambi_family = fam.bambi_family()
        if any(a.kind != "mi" for a in self.spec.additions):   # brms's response addition terms act on the likelihood (B02)
            from autobayes.likelihoods import addition_family
            bambi_family = addition_family(bambi_family, fam, addition_data(self.spec, self.data))
        bambi_data = self.data
        own = self.missing_names.get(self.spec.response)      # C06: the response's missing values are parameters (| mi())
        if own is not None:
            from autobayes.likelihoods import missing_response_family
            block = self._missing_block(self.spec.response)
            bambi_data = self.data.copy()                     # Bambi refuses a missing response: a placeholder it never reads
            bambi_data[self.spec.response] = block.filled
            bambi_family = missing_response_family(bambi_family, block.index, own["values"], own["coordinate"])
        self._check_generated_names()
        bambi_priors = priors.to_bambi(self.spec, fam)
        for name in fam.fixed_parameters:                    # Bambi takes a fixed nu through the prior dict; a learned
            if name not in self.learned_parameters:          # one's truncated prior came from Priors.to_bambi (A13)
                bambi_priors[name] = np.float64(getattr(self, name))   # PyTensor makes a Python 4.0 a float32 constant
        for extra in fam.extra_parameters:                   # D01: alpha, beta, ndt; ndt truncated to (0, min y) of these rows;
            if extra.name not in self.spec.dpar_names:       # E03: an alpha with its own formula has its formula's priors there
                bambi_priors[extra.variable] = extra.bambi_prior(getattr(priors, extra.name), self.data[self.spec.response])   # E02: tmp_xi
        bambi_formula = self.spec.bambi_formula
        # Bambi centres every formula that has an intercept, or none. Without a distributional formula that is the
        # response formula's centring as it has always been; with one (B01), brms centres each formula that has an
        # intercept on its own means, and the refusal above leaves exactly that case. A family that builds what Bambi is
        # handed from the model (B05: cumulative, categorical; neither takes a distributional formula) decides it itself.
        self._bambi_centring = self.effective_center_predictors
        if self.spec.dpars:                                  # brms's bf(y ~ ..., sigma ~ ...): Bambi's distributional components (B01)
            bambi_formula = bmb.Formula(bambi_formula, *(f"{self._bambi_dpar(d.response)} ~ {d.bambi_rhs}" for d in self.spec.dpars))
            self._bambi_centring = self.center_predictors
        if fam.bambi_setup is not None:                      # B05: the family builds what Bambi is handed from the model
            bambi_family, bambi_priors, self._bambi_centring = fam.bambi_setup(self, bambi_priors)
        if self.spec.mo_terms:                               # brms's mo() (B03): added to the parent's linear predictor at build
            from autobayes.likelihoods import monotonic_family
            bambi_family = monotonic_family(bambi_family, self._add_monotonic_effects)
        if self.spec.gp_terms:                               # brms's gp() (C03): added the same way, through the same hook
            from autobayes.likelihoods import monotonic_family as linear_predictor_family
            bambi_family = linear_predictor_family(bambi_family, self._add_gp_effects)
        if self.spec.me_terms or self.spec.mi_terms:         # brms's me() and mi() (C06): added the same way
            from autobayes.likelihoods import monotonic_family
            bambi_family = monotonic_family(bambi_family, self._add_measurement_effects)
        if self.spec.custom_group_terms:                     # brms's mm() and gr(by = ) (D08): added the same way
            from autobayes.likelihoods import monotonic_family
            bambi_family = monotonic_family(bambi_family, self._add_custom_group_effects)
        if self.spec.autocor is not None:                    # brms's ar(), ma(), arma() (C04): added last, on the residuals
            from autobayes.autocor import autocor_family
            bambi_family = autocor_family(bambi_family, self._add_autocor_effects)
        if self.spec.sar is not None:                        # brms's sar(M, type) (F02b): the joint SAR likelihood
            from autobayes.sar import sar_family
            bambi_family = sar_family(bambi_family, self._sar_design)
        self.bambi_model = bmb.Model(
            bambi_formula, bambi_data, family=bambi_family,
            priors=bambi_priors,
            noncentered=True, center_predictors=self._bambi_centring,
        )
        self.bambi_model.build()
        self.pymc_model = self.bambi_model.backend.model
        if self.spec.correlated_terms or self.spec.shared_ids:   # brms's correlated slopes (A02) and shared IDs (D08): a post-build
            self.pymc_model = self._correlate_group_effects(self.pymc_model)   # edit of Bambi's graph
            self.bambi_model.backend.model = self.pymc_model  # what Bambi's fit() samples and cleans
        self.design = self._extract_design()
        if self.shrinkage_names:                             # D07: b = zb * sdb under horseshoe() / R2D2(), checked
            from autobayes.shrinkage import check_graph
            check_graph(self)
        if self.spec.me_terms or self.spec.mi_terms:         # C06: each term's contribution to the linear predictor, checked
            self._check_measurement_effects()
        if fam.mixture:                                      # C05: the graph built beside Bambi's components, checked
            from autobayes.families.mixture import check_graph
            check_graph(self)
        if self.spec.autocor is not None:                    # C04: the graph's ARMA terms against brms's recursion
            self._check_autocor_effects()
        if fam.discrete:
            rv = self.pymc_model[self.names["y"]]
            obs = np.asarray(self.pymc_model.rvs_to_values[rv].eval())
            if not np.array_equal(obs, self.design.y.astype(int)):
                raise RuntimeError("Bambi's observed response does not equal the response column")
            if fam.has_trials:
                consts = [np.asarray(i.data) for i in rv.owner.inputs if getattr(i, "data", None) is not None]
                if not any(c.shape == self.design.trials.shape and np.array_equal(c, self.design.trials) for c in consts):
                    raise RuntimeError("the trials attached to the likelihood do not equal the trials column")

        from autobayes.identity import freeze_target
        self._target_identity = MappingProxyType(freeze_target(self))

    @property
    def target_identity(self) -> dict[str, str]:
        """Frozen built-target specification digest; returns a defensive copy."""
        return dict(self._target_identity)

    def _check_generated_names(self):
        """Refuse names that would alias a coefficient, effect, likelihood or coordinate."""
        variable_names = (["Intercept"] if self.spec.intercept else []) + list(self.spec.terms)
        variable_names += [self.names["y"], self.names["linpred"]]
        if self.family_spec.scale_parameter is not None:
            variable_names += [self.family_spec.bambi_scale_parameter]   # the PyMC variable: "sigma", or "alpha" for "shape"
        if self.family_spec.zero_parameter is not None:
            variable_names += [self.family_spec.zero_parameter]          # the zero-inflation / hurdle probability, "zi" or "hu" (B06)
        variable_names += [e.name for e in self.family_spec.extra_parameters]   # alpha, beta or ndt (D01)
        variable_names += [e.variable for e in self.family_spec.extra_parameters if e.derived]   # E02: tmp_xi beside xi
        variable_names += list(self.family_spec.fixed_parameters)   # Bambi uses "nu" as an auxiliary-prior key even when fixed.
        if self.family_spec.thresholds:                     # B05: the raw thresholds' deterministic and their coordinate
            variable_names += [self.names["thresholds"], self.names["threshold_dim"]]
        if self.spec.cs_terms:                              # D04: Bambi's parent mu beside the thresholds' predictor mu_cs
            variable_names += [self.family_spec.parent]
        if self.family_spec.per_category:                   # B05: the category coordinates Bambi adds
            variable_names += [self.names["category_dim"], f"{self.spec.response}_dim"]
        if self.family_spec.mixture:                        # C05: each component's sigma, theta and the component coordinates
            roles = self.mixture_names
            variable_names += [*roles["sigma"], roles["theta"], roles["coordinate"], f"{self.spec.response}_dim"]
        for roles in (*self.group_coefficient_names.values(), *self.dpar_group_names.values()):   # D08: a shared ID's dpar terms too
            variable_names.extend(roles[k] for k in ("sd", "offset", "group_effect", "by_normalisation") if k in roles)
        for roles in self.correlation_names.values():
            variable_names.extend(roles[k] for k in ("L", "cor"))
        dpar_coefficients = [v for roles in self.dpar_names.values() for k, v in roles.items() if k != "value"]   # B01
        variable_names += dpar_coefficients
        variable_names += [roles["simplex"] for roles in self.mo_names.values()]   # each monotonic term's simplex (B03)
        variable_names += [roles[k] for roles in self.gp_names.values() for k in ("sdgp", "lscale", "zgp")]   # each GP's (C03)
        variable_names += [v for k, v in self.autocor_names.items() if not k.endswith("_coordinate")]   # ar and ma (C04)
        variable_names += list(self.sar_names.values())      # lagsar or errorsar (F02b)
        variable_names += [roles[k] for roles in self.me_names.values() for k in ("mean", "sd", "z", "latent", "noisy")]   # C06
        variable_names += [roles["values"] for roles in self.missing_names.values()]   # C06: the missing values Ymi_<col>
        shrinkage = self.shrinkage_names                    # D07: zb, sdb and the horseshoe's or R2D2's own variables
        variable_names += [v for k, v in shrinkage.items() if k not in ("prior", "coordinate", "lprior")]
        # Bambi's default result cleaning drops every posterior variable ending in _offset.
        retained = list(self.spec.terms) + [roles["group_effect"] for roles in self.group_coefficient_names.values()] + dpar_coefficients
        if any(name.endswith("_offset") for name in retained):
            raise FormulaError("generated name collision: coefficient/effect suffix '_offset' is reserved by Bambi; rename the column")
        coords = [roles["coordinate"] for roles in self.group_names.values()]
        coords += [roles["by_coordinate"] for roles in self.group_names.values() if "by_coordinate" in roles]   # D08
        coords += [roles[k] for roles in self.correlation_names.values() for k in ("coordinate", "coordinate_T")]
        coords += [f"{t.name}_dim" for t in self.spec.pop_terms if t.is_vector]
        coords += [f"{self.dpar_names[d.response][f'b:{t.name}']}_dim" for d in self.spec.dpars for t in d.pop_terms if t.is_vector]
        coords += [roles["coordinate"] for roles in self.mo_names.values()]
        coords += [roles["coordinate"] for roles in self.gp_names.values()]       # C03
        coords += [v for k, v in self.autocor_names.items() if k.endswith("_coordinate")]   # ar_dim and ma_dim (C04)
        coords += [roles["coordinate"] for roles in self.missing_names.values()]   # C06: the missing rows of each column
        coords += [shrinkage["coordinate"]] if shrinkage else []                    # D07: zb_dim
        coords += ["__obs__"]
        names = variable_names + coords
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise FormulaError(f"generated variable/coordinate name collision: {duplicates}; rename the conflicting column")

    # -- correlated group effects (A02) -------------------------------------------------
    def _correlate_group_effects(self, model):
        """brms's correlated construction for every ``(1 + x | g)`` term, as an edit of the PyMC graph Bambi built.

        Bambi built each coefficient ``c`` of the factor as the independent non-centred term ``r_c = sd_c * z_c``
        (``x|g = x|g_sigma * x|g_offset``). brms's ``scale_r_cor(z, sd, L)`` is ``r = (diag(sd) L z)^T``, i.e.
        ``r_c = sd_c * sum_{k <= c} L[c, k] z_k``; its first row is the intercept's own ``sd_0 z_0`` (``L[0, 0] = 1``),
        so only the slopes' deterministics change. The graph is converted to a function graph
        (``pymc.model.fgraph``), a free variable ``L|g`` (brms ``L_m``, ``lkj_corr_cholesky(eta)``, sampled on Stan's
        ``cholesky_corr`` coordinates) and the deterministic ``cor|g = L L^T`` (brms ``Cor_m``) are added, each slope's
        deterministic is replaced by the correlated expression everywhere it is used (the linear predictor included),
        and a new ``pymc.Model`` is rebuilt; ``_extract_design`` then verifies names, dims and coordinates."""
        import pytensor
        from pymc.model.fgraph import fgraph_from_model, model_deterministic, model_free_rv, model_from_fgraph
        from pymc.pytensorf import toposort_replace

        cor_priors = self.priors.cor_priors(self.spec)
        fgraph, memo = fgraph_from_model(model, inlined_views=True)
        replacements, added = [], []
        # D08: a correlation ID shared with a distributional formula is one more correlated group, its coefficients the
        # response formula's term's and then the distributional formula's (Bambi's sigma_1|g, under the component's prefix)
        groups = [(t.factor, t.coefs, [self._group_roles(t.factor, c) for c in t.coefs]) for t in self.spec.correlated_terms]
        groups += [(s.factor, s.labels, [self._shared_roles(s.factor, dpar, c) for dpar, c in s.coefs]) for s in self.spec.shared_ids]
        for g, labels, roles in groups:
            M = len(labels)
            names = self._correlation_roles(g)
            rv, transform, value = cholesky_corr_rv(names["L"], float(cor_priors[g].params["eta"]), M)
            for dim in (names["coordinate"], names["coordinate_T"]):
                fgraph._coords[dim] = tuple(labels)
                fgraph._dim_lengths[dim] = pytensor.shared(M, name=dim)
            L = model_free_rv(rv, value, transform, names["coordinate"], names["coordinate_T"])
            L.name = names["L"]
            cor = L @ L.T
            cor.name = names["cor"]
            cor_det = model_deterministic(cor, names["coordinate"], names["coordinate_T"])
            cor_det.name = names["cor"]
            added += [L, cor_det]
            zs = [memo[model[r["offset"]]] for r in roles]
            for c in range(1, M):
                sd = memo[model[roles[c]["sd"]]]
                expr = sd * sum(L[c, k] * zs[k] for k in range(c + 1))
                expr.name = roles[c]["group_effect"]
                det = model_deterministic(expr, roles[c]["coordinate"])
                det.name = roles[c]["group_effect"]
                replacements.append((memo[model[roles[c]["group_effect"]]], det))
        toposort_replace(fgraph, tuple(replacements))
        for var in added:
            fgraph.add_output(var)
        return model_from_fgraph(fgraph, mutate_fgraph=True)

    # -- design ------------------------------------------------------------------------
    def _population_design(self, comp, spec: ModelSpec, Xfull: np.ndarray, prefix: str = "") -> np.ndarray:
        """Bambi's common design ``Xfull`` of one formula (the response formula's, or a distributional parameter's, B01,
        whose PyMC names carry the component's ``prefix``, ``sigma_``), without an offset column, checked term by term
        against the data; returns the columns the coefficients multiply (``X_raw``)."""
        common = comp.design.common
        bambi_terms = [t.name for t in spec.bambi_terms]                # a monotonic term is not Bambi's (B03)
        if list(comp.common_terms) != bambi_terms:
            raise RuntimeError(f"Bambi's coefficient terms {list(comp.common_terms)} differ from the formula's {bambi_terms}")
        if spec.intercept and not np.allclose(Xfull[:, 0], 1.0):
            raise RuntimeError("Bambi's first common column is not an intercept column")
        # Verify Bambi's columns term by term against each term's own design block on the data
        # (``pop_term_design``): a numeric column as given, a numeric-by-numeric product as one column, a
        # categorical or other interaction term as one column per coefficient level in the recorded order.
        col = int(spec.intercept)
        for term in spec.bambi_terms:
            expected = pop_term_design(term, self.data, spec.categoricals)
            bterm = common.terms[term.name]
            is_inter = term.kind == "interaction"
            if not term.is_vector:                                     # x, or the product x:w: one column, no levels
                expected_kind = "interaction" if is_inter else "numeric"
                if getattr(bterm, "kind", None) != expected_kind or list(bterm.labels) != [term.name] or getattr(bterm, "levels", None):
                    raise RuntimeError(
                        f"Bambi coded {term.name!r} as {getattr(bterm, 'kind', None)} with levels {getattr(bterm, 'levels', None)} "
                        f"and labels {getattr(bterm, 'labels', None)}; expected one {expected_kind} column {term.name!r}"
                    )
                if not np.array_equal(Xfull[:, col], expected[:, 0]):
                    what = f"the product {' * '.join(term.operands)}" if is_inter else "the data column"
                    raise RuntimeError(f"Bambi's column for {term.name!r} does not equal {what}")
                col += 1
                continue
            cats = [spec.categorical(c) for c in term.categorical]
            expected_kind = "interaction" if is_inter else "categoric"
            if getattr(bterm, "kind", None) != expected_kind or list(bterm.levels) != list(term.levels) \
                    or list(bterm.labels) != list(term.columns):
                raise RuntimeError(
                    f"Bambi coded {term.name!r} as {getattr(bterm, 'kind', None)} with levels {getattr(bterm, 'levels', None)} "
                    f"and labels {getattr(bterm, 'labels', None)}; expected {expected_kind} coding with reference "
                    f"{', '.join(repr(c.reference) for c in cats)}, levels {list(term.levels)} and labels {list(term.columns)}"
                )
            for cat in cats:
                labels = self.original_labels[cat.name]                # the supplied values' labels, not the recoded column
                if not np.array_equal(labels, self.data[cat.name].astype(str).to_numpy()):
                    raise RuntimeError(f"the recoded column {cat.name!r} does not reproduce the supplied labels")
            for j, lvl in enumerate(term.levels):
                if not np.array_equal(Xfull[:, col + j], expected[:, j]):
                    what = (f"the indicator of level {lvl!r}" if not is_inter else
                            f"the indicator product of cell {lvl!r}" if len(cats) == 2 else f"x * indicator of level {lvl!r}")
                    raise RuntimeError(f"Bambi's column {term.columns[j]!r} is not {what}")
            if not is_inter:
                cat, labels = cats[0], self.original_labels[cats[0].name]
                if not np.array_equal(Xfull[:, col:col + len(term.levels)].sum(1) + (0 if cat.full_rank else (labels == cat.reference)), np.ones(len(labels))):
                    raise RuntimeError(f"the dummies of {term.name!r} and the reference indicator do not partition the rows")
            coords = [str(c) for c in self.pymc_model.coords.get(f"{prefix}{term.name}_dim", ())]
            if coords != list(term.levels):
                raise RuntimeError(f"PyMC coordinates {prefix}{term.name}_dim = {coords} differ from the coefficient levels {list(term.levels)}")
            col += len(term.levels)
        if col != Xfull.shape[1]:
            raise RuntimeError(f"Bambi's common design has {Xfull.shape[1]} columns, expected {col}")
        return Xfull[:, int(spec.intercept):]

    def _extract_design(self) -> Design:
        comp = self.bambi_model.components[self.family_spec.parent]
        common = comp.design.common
        group = comp.design.group
        names = list(common.terms) if common is not None else []    # none at all: '0 +' and only monotonic terms (B03)
        offset_terms = [self.spec.offset.term] if self.spec.offset is not None else []
        if names != (["Intercept"] if self.spec.intercept else []) + [t.name for t in self.spec.bambi_terms] + offset_terms:
            raise RuntimeError(f"unexpected common terms from Bambi: {names}")
        Xfull = np.asarray(common.design_matrix, dtype=float) if common is not None else np.zeros((len(self.data), 0))
        # The offset term (brms offset(), formulae's kind "offset"): Bambi adds its column to the linear predictor as
        # given, outside the coefficient design and never centred. Its values are recomputed here from the data and
        # must equal what Bambi evaluated; the column is then removed from the design that the coefficients multiply.
        offset = None
        if self.spec.offset is not None:
            term = self.spec.offset.term
            if list(comp.offset_terms) != [term] or getattr(common.terms[term], "kind", None) != "offset":
                raise RuntimeError(f"Bambi did not build {term!r} as the single offset term (offset terms: {list(comp.offset_terms)})")
            offset = self.spec.offset.values(self.data)
            built = np.asarray(comp.offset_terms[term].data, dtype=float).ravel()
            if built.shape != offset.shape or not np.array_equal(built, offset):
                raise RuntimeError(f"Bambi's offset {term!r} does not equal {self.spec.offset.expression} computed from the data")
            if not np.array_equal(Xfull[:, common.slices[term]].ravel(), offset):
                raise RuntimeError(f"the offset column of Bambi's common design does not equal {self.spec.offset.expression}")
            Xfull = np.delete(Xfull, common.slices[term], axis=1)
        X_raw = self._population_design(comp, self.spec, Xfull)
        # One block per (factor, coefficient), in formula order; Bambi's concatenated group design is their
        # side-by-side concatenation. The varying intercept's block is the one-hot indicator of the level; a
        # varying slope's is that indicator times the slope's (raw, never centred) column, which is what brms's
        # Z_m_c spread over J_m is. Bambi was handed '(1 + x | g)' for '(1 + x || g)': it builds '1|g' and 'x|g'
        # as independent terms, and each block is checked here against the data, so a different coding is a
        # hard error rather than a silently different model.
        group_blocks: dict[tuple[str, str], GroupBlock] = {}
        if not self.spec.has_group:
            if group is not None:
                raise RuntimeError("Bambi built a group design for a formula without a group-level term")
            Z = np.zeros((len(X_raw), 0))
        else:
            # D08: the multi-membership and by terms are the model's own (_custom_group_blocks); Bambi's group design holds
            # the others, side by side, and Design.Z every block in formula order
            expected_terms = [_bambi_group_name(t.factor, coef) for t in self.spec.bambi_group_terms for coef in t.coefs]
            if (group is None) != (not expected_terms) or (group is not None and list(group.terms) != expected_terms):
                raise RuntimeError("Bambi group terms differ from formula order")
            bambi_Z = group.design_matrix if group is not None else np.zeros((len(X_raw), 0))
            bambi_Z = np.asarray(bambi_Z.toarray() if hasattr(bambi_Z, "toarray") else bambi_Z, dtype=float)
            custom = self._custom_group_blocks() if self.spec.custom_group_terms else {}
            start, bambi_start, parts = 0, 0, []
            for t in self.spec.group_terms:
                g = t.factor
                levels = self.spec.levels_by_group[g]
                if t.custom:                                 # D08: the block built and checked against the graph here
                    block = custom[g]
                    group_blocks[(g, "Intercept")] = replace(block, column_slice=slice(start, start + len(levels)))
                    parts.append(block.Z)
                    start += len(levels)
                    continue
                labels = self.data[t.column].to_numpy()
                indicator = np.column_stack([labels == level for level in levels]).astype(float)
                for coef in t.coefs:
                    name = _bambi_group_name(g, coef)
                    bterm = group.terms[name]
                    if tuple(str(v) for v in bterm.groups) != levels:
                        raise RuntimeError(f"Bambi level order for {g!r} differs from sorted fitted labels")
                    if group.slices[name] != slice(bambi_start, bambi_start + len(levels)):
                        raise RuntimeError(f"Bambi group slice for {g!r} differs from formula order")
                    block_slice = slice(start, start + len(levels))
                    block = bambi_Z[:, group.slices[name]]
                    parts.append(block)
                    bambi_start += len(levels)
                    if coef == "Intercept":
                        if not np.array_equal(block, indicator) or not np.array_equal(block.sum(1), np.ones(len(X_raw))):
                            raise RuntimeError(f"group design for {g!r} does not encode original rows one-hot")
                    else:
                        kind = getattr(getattr(bterm, "expr", None), "kind", None)
                        if kind != "numeric":
                            raise RuntimeError(f"Bambi coded the varying slope {name!r} as {kind!r}; expected a numeric slope")
                        covariate = self.data[coef].to_numpy(dtype=float)
                        if not np.array_equal(block, indicator * covariate[:, None]):
                            raise RuntimeError(
                                f"Bambi's block for {name!r} is not the level indicator of {g!r} times the {coef!r} column"
                            )
                    group_blocks[(g, coef)] = GroupBlock(levels, block, indicator.argmax(1), block_slice)
                    start += len(levels)
                coord = self.group_names[g]["coordinate"]
                if tuple(str(v) for v in self.pymc_model.coords[coord]) != levels:
                    raise RuntimeError(f"PyMC coordinate {coord!r} differs from fitted labels")
                for coef in t.coefs:                       # every coefficient's effects carry the factor's level dimension
                    roles = self._group_roles(g, coef)
                    for key in ("offset", "group_effect"):
                        dims = tuple(self.pymc_model.named_vars_to_dims.get(roles[key], ()))
                        if dims != (coord,):
                            raise RuntimeError(f"PyMC dims of {roles[key]!r} are {dims}; expected ({coord!r},)")
            Z = np.concatenate(parts, axis=1) if parts else np.zeros((len(X_raw), 0))
            if Z.shape != (len(X_raw), start) or bambi_Z.shape != (len(X_raw), bambi_start):
                raise RuntimeError("unexpected concatenated group design shape")
            for g in self.correlation_names:                 # the correlation factor (A02; a shared ID's, D08): a free variable on
                coefs = self.correlation_coefs(g)            # Stan's coordinates
                t = SimpleNamespace(factor=g, coefs=coefs)
                names, M = self._correlation_roles(t.factor), len(t.coefs)
                L = self.pymc_model[names["L"]]
                if L not in self.pymc_model.free_RVs or tuple(L.type.shape) != (M, M):
                    raise RuntimeError(f"{names['L']!r} is not a free {M} x {M} variable of the graph")
                value = self.pymc_model.rvs_to_values[L]
                if value.name != f"{names['L']}_cholesky_corr__" or tuple(value.type.shape) != (n_free(M),):
                    raise RuntimeError(f"{names['L']!r} is not sampled on the {n_free(M)} cholesky_corr coordinates")
                if names["cor"] not in self.pymc_model.named_vars or self.pymc_model[names["cor"]] not in self.pymc_model.deterministics:
                    raise RuntimeError(f"{names['cor']!r} is not a deterministic of the graph")
                for key in ("L", "cor"):
                    dims = tuple(self.pymc_model.named_vars_to_dims.get(names[key], ()))
                    if dims != (names["coordinate"], names["coordinate_T"]):
                        raise RuntimeError(f"PyMC dims of {names[key]!r} are {dims}; expected the coefficient dimensions")
                for dim in (names["coordinate"], names["coordinate_T"]):
                    if tuple(str(v) for v in self.pymc_model.coords[dim]) != t.coefs:
                        raise RuntimeError(f"PyMC coordinate {dim!r} differs from the coefficients {list(t.coefs)}")
        if self.effective_center_predictors:
            x_means = X_raw.mean(0)
            X_model = X_raw - x_means
        else:
            x_means = np.zeros(X_raw.shape[1])
            X_model = X_raw
        y = np.asarray(self.data[self.spec.response], dtype=float)
        trials = np.asarray(self.data[self.spec.trials], dtype=np.int64) if self.spec.trials else None
        additions = addition_data(self.spec, self.data)      # brms's weights, cens, lb and ub on the fitted rows (B02)
        return Design(y=y, X_raw=X_raw, X_model=X_model, x_means=x_means,
                      predictors=self.spec.predictors, Z=Z, group_blocks=group_blocks,
                      centered=self.effective_center_predictors, trials=trials,
                      columns=self.spec.columns, categoricals=self.spec.categoricals, offset=offset,
                      weights=additions.get("weights"), cens=additions.get("cens"), lb=additions.get("lb"), ub=additions.get("ub"),
                      rcens=additions.get("rcens"),                                         # E04: cens(col, y2)'s upper bounds
                      dpars=MappingProxyType({d.response: self._dpar_design(d) for d in self.spec.dpars}),
                      monotonic=MappingProxyType(self._monotonic_design()), gp=MappingProxyType(self._gp_design()),
                      autocor=self._autocor_design, sar=self._sar_design,                    # F02b
                      measurement_error=MappingProxyType(self._measurement_design()),   # C06
                      missing=MappingProxyType(self._missing_design()),
                      category_specific=MappingProxyType(self._category_specific_design()))   # D04

    def _dpar_design(self, d: ModelSpec) -> DparDesign:
        """The design of distributional parameter ``d.response``'s formula (B01), as Bambi built its distributional
        component (named after the parameter, Bambi's ``alpha`` for brms's ``shape``): the intercept and population-level
        terms checked term by term against the data as the response formula's are, no group-level or offset term, the
        parameter itself a per-row deterministic, its coefficients free variables named ``<component>_<term>``. Centred on
        its own means when it has an intercept (brms's ``Xc_<dpar>``; Bambi centres every formula with an intercept)."""
        bname = self._bambi_dpar(d.response)
        comp = self.bambi_model.components.get(bname)
        if comp is None or comp.design.common is None:
            raise RuntimeError(f"Bambi built no distributional component {bname!r} for the formula '{d.formula}'")
        group_blocks = self._dpar_group_blocks(d, comp) if d.group_terms else {}   # D08: a shared correlation ID's terms
        if comp.design.group is not None and not d.group_terms:
            raise RuntimeError(f"Bambi built group-level terms for the formula '{d.formula}', which has none")
        names = list(comp.design.common.terms)
        if names != (["Intercept"] if d.intercept else []) + list(d.terms):
            raise RuntimeError(f"unexpected terms from Bambi for the formula '{d.formula}': {names}")
        X_raw = self._population_design(comp, d, np.asarray(comp.design.common.design_matrix, dtype=float), prefix=f"{bname}_")
        roles = self.dpar_names[d.response]
        value = self.pymc_model[roles["value"]]
        if value not in self.pymc_model.deterministics or tuple(self.pymc_model.named_vars_to_dims.get(roles["value"], ())) != ("__obs__",):
            raise RuntimeError(f"{roles['value']!r} is not the per-row deterministic of {d.response}'s formula")
        for key, name in roles.items():
            if key != "value" and self.pymc_model[name] not in self.pymc_model.free_RVs:
                raise RuntimeError(f"{name!r} is not a free variable of the graph")
        link = self.family_spec.distributional_parameter_link(d.response)   # log (B01); E03: logit for zi / hu, identity for alpha
        if self.bambi_model.family.link[bname].name != link:
            raise RuntimeError(f"Bambi's link of {bname!r} is {self.bambi_model.family.link[bname].name}, expected {link}")
        centered = bool(self._bambi_centring and d.intercept and d.pop_terms)   # an intercept-only formula has nothing to centre
        x_means = X_raw.mean(0) if centered else np.zeros(X_raw.shape[1])
        return DparDesign(name=d.response, X_raw=X_raw, X_model=X_raw - x_means if centered else X_raw, x_means=x_means,
                          columns=d.columns, centered=centered, link=link,
                          group_blocks=MappingProxyType(group_blocks))

    def _dpar_group_blocks(self, d: ModelSpec, comp) -> dict[tuple[str, str], GroupBlock]:
        """D08: the blocks of a distributional formula's group-level terms (each shares a correlation ID with the response
        formula's term of its factor; the varying intercept only), as Bambi built its distributional component's group
        design: every term in formula order, the level order the sorted fitted labels, each block the one-hot indicator of
        the row's level, and the effects ``<component>_1|g`` and offsets on the factor's level dimension."""
        group = comp.design.group
        expected = [_bambi_group_name(t.factor, c) for t in d.group_terms for c in t.coefs]
        if group is None or list(group.terms) != expected:
            raise RuntimeError(f"Bambi's group terms of the formula '{d.formula}' are {None if group is None else list(group.terms)}; "
                               f"expected {expected}")
        Z = np.asarray(group.design_matrix.toarray() if hasattr(group.design_matrix, "toarray") else group.design_matrix, dtype=float)
        blocks, start = {}, 0
        for t in d.group_terms:
            levels = self.spec.levels_by_group[t.factor]
            if d.levels_by_group[t.factor] != levels or tuple(str(v) for v in group.terms[_bambi_group_name(t.factor, "Intercept")].groups) != levels:
                raise RuntimeError(f"the levels of {t.factor!r} in the formula '{d.formula}' differ from the response formula's")
            labels = self.data[t.column].to_numpy()
            indicator = np.column_stack([labels == level for level in levels]).astype(float)
            block_slice = slice(start, start + len(levels))
            if group.slices[_bambi_group_name(t.factor, "Intercept")] != block_slice or not np.array_equal(Z[:, block_slice], indicator):
                raise RuntimeError(f"Bambi's group design of {t.factor!r} in the formula '{d.formula}' does not encode the rows one-hot")
            roles = self._shared_roles(t.factor, d.response, "Intercept")
            for key in ("offset", "group_effect"):
                if tuple(self.pymc_model.named_vars_to_dims.get(roles[key], ())) != (roles["coordinate"],):
                    raise RuntimeError(f"PyMC dims of {roles[key]!r} are not ({roles['coordinate']!r},)")
            if self.pymc_model[roles["sd"]] not in self.pymc_model.free_RVs:
                raise RuntimeError(f"{roles['sd']!r} is not a free variable of the graph")
            blocks[(t.factor, "Intercept")] = GroupBlock(levels, Z[:, block_slice], indicator.argmax(1), block_slice)
            start += len(levels)
        return blocks

    def _bambi_dpar(self, name: str) -> str:
        """Bambi's (and so PyMC's) name of distributional parameter ``name`` (B01): the family's ``bambi_scale_parameter``
        (``alpha`` for brms's ``shape``), the only parameter that takes a formula here."""
        return self.family_spec.bambi_scale_parameter if name == self.family_spec.scale_parameter else name

    # -- monotonic effects (B03) ---------------------------------------------------------
    def _add_monotonic_effects(self, eta):
        """brms's monotonic terms, built inside the model Bambi is building (``likelihoods.monotonic_family``, which Bambi
        calls on the parent's linear predictor before the link): for each term its simplex (brms's ``simo``, a Dirichlet
        under its ``simo`` prior on Stan's stick-breaking coordinates) and its coefficient (brms's ``bsp``, under its ``b``
        prior), and ``eta`` plus ``bsp * mo(simo, Xmo_i)`` per row, ``mo`` the term's share of the simplex times ``D``."""
        import pymc as pm

        from autobayes.monotonic import coefficient_rv, monotonic_effect, simplex_rv
        from autobayes.priors import dirichlet_concentration

        model = pm.modelcontext(None)
        coefficients, simplexes = self.priors.coefficient_priors(self.spec), self.priors.simo_priors(self.spec)
        for t in self.spec.mo_terms:
            m, roles = self.spec.monotonic(t.operands[0]), self.mo_names[t.operands[0]]
            model.add_coord(roles["coordinate"], [str(k) for k in range(1, m.D + 1)])
            simplex = simplex_rv(roles["simplex"], dirichlet_concentration(simplexes[m.column], m.D), roles["coordinate"])
            coefficient = coefficient_rv(roles["b"], coefficients[t.name])
            eta = eta + coefficient * monotonic_effect(simplex, m.codes(self.data[m.column]), m.D)
        return eta

    def _monotonic_design(self) -> dict[str, MonotonicBlock]:
        """The block of each monotonic term (B03), checked against the graph: the simplex a free Dirichlet of ``D`` entries
        with the coordinate ``"1"``..``"D"``, sampled on its ``D - 1`` stick-breaking coordinates; the coefficient a free
        scalar; and the term's contribution to the parent's linear predictor recomputed from the data
        (``_check_monotonic_effects``). Empty without a monotonic term."""
        from autobayes.monotonic import StickBreakingTransform

        pmm, blocks = self.pymc_model, {}
        for t in self.spec.mo_terms:
            m, roles = self.spec.monotonic(t.operands[0]), self.mo_names[t.operands[0]]
            simplex, coefficient = pmm[roles["simplex"]], pmm[roles["b"]]
            if simplex not in pmm.free_RVs or tuple(simplex.type.shape) != (m.D,):
                raise RuntimeError(f"{roles['simplex']!r} is not a free simplex of {m.D} entries in the graph")
            value = pmm.rvs_to_values[simplex]
            if (value.name != f"{roles['simplex']}_stickbreaking__" or tuple(value.type.shape) != (m.D - 1,)
                    or not isinstance(pmm.rvs_to_transforms[simplex], StickBreakingTransform)):
                raise RuntimeError(f"{roles['simplex']!r} is not sampled on Stan's {m.D - 1} stick-breaking coordinates")
            if (tuple(pmm.named_vars_to_dims.get(roles["simplex"], ())) != (roles["coordinate"],)
                    or tuple(str(v) for v in pmm.coords[roles["coordinate"]]) != tuple(str(k) for k in range(1, m.D + 1))):
                raise RuntimeError(f"PyMC dims of {roles['simplex']!r} are not ({roles['coordinate']!r},) with coordinates 1..{m.D}")
            if coefficient not in pmm.free_RVs or coefficient.type.ndim != 0:
                raise RuntimeError(f"{roles['b']!r} is not a free scalar of the graph")
            blocks[m.column] = MonotonicBlock(t.name, m.column, m.codes(self.data[m.column]), m.D)
        if blocks:
            self._check_monotonic_effects(blocks)
        return blocks

    def _check_monotonic_effects(self, blocks) -> None:
        """Each monotonic term's contribution, read off the graph: at a simplex with unequal steps and every other variable at
        the graph's initial point, the parent's linear predictor (on Bambi's link scale: the parent itself, or the log of
        ``mu`` under a log link) must move, row by row, by exactly ``c * D * sum(simplex[:Xmo_i])`` (recomputed from the data's
        codes, ``monotonic.monotonic_values``) when the term's coefficient goes from 0 to ``c = 1 / D`` (a move of at most 1,
        which no link overflows)."""
        from autobayes.monotonic import monotonic_values, stick_breaking_free

        pmm, parent = self.pymc_model, self.family_spec.parent
        link = self.bambi_model.family.link[parent].name
        if link not in ("identity", "log"):
            raise RuntimeError(f"Bambi's link of {parent!r} is {link}; the monotonic terms are checked under the identity and log links")
        fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[parent]]), inputs=pmm.value_vars, point_fn=True,
                            on_unused_input="ignore", mode="FAST_COMPILE")
        point = pmm.initial_point()
        point.update({roles["b"]: np.asarray(0.0) for roles in self.mo_names.values()})
        for column, block in blocks.items():
            roles = self.mo_names[column]
            test, c = np.arange(1.0, block.D + 1.0) / (block.D * (block.D + 1) / 2.0), 1.0 / block.D
            at0 = {**point, f"{roles['simplex']}_stickbreaking__": stick_breaking_free(test)}
            lo, hi = (np.asarray(fn(p)[0], dtype=float).ravel() for p in (at0, {**at0, roles["b"]: np.asarray(c)}))
            moved = hi - lo if link == "identity" else np.log(hi) - np.log(lo)
            expected = c * monotonic_values(test, block.codes)
            if moved.shape != expected.shape or not np.allclose(moved, expected, rtol=1e-10, atol=1e-10):
                raise RuntimeError(f"the graph's {parent!r} does not move by D * sum(simplex[:Xmo]) per row with the coefficient of "
                                   f"{block.term!r}; largest difference {np.max(np.abs(moved - expected)) if moved.shape == expected.shape else moved.shape}")

    # -- category-specific effects (D04) -------------------------------------------------
    def _add_category_specific_effects(self, mu):
        """brms's category-specific effects ``cs(x)`` (D04, docs/ordinal-categorical.md), built inside the model Bambi is
        building (the ordinal likelihood calls this with the parent ``mu``, ``likelihoods.ordinal_family``): for each term its
        coefficients, one per threshold (brms's row of ``bcs`` for the column; a free vector named after the term, ``cs(x)``, on
        the thresholds' dimension, under its ``b`` prior, as brms puts ``bcs`` under class ``b``), and the predictor of every
        threshold, ``mu_i + sum_c x_c,i bcs_c[k]`` (brms's ``mu`` and ``Xcs bcs``, the column never centred), registered as the
        deterministic ``names["linpred"]`` (``mu_cs``; dims ``__obs__`` and the thresholds'), which the likelihood reads."""
        import pymc as pm

        coefficients = self.priors.coefficient_priors(self.spec)
        dim = self.names["threshold_dim"]
        eta = mu[:, None]
        for t in self.spec.cs_terms:
            prior = coefficients[t.name]
            coef = getattr(pm, prior.name)(t.name, **{k: np.float64(v) for k, v in prior.params.items()}, dims=dim)
            eta = eta + np.asarray(self.data[t.operands[0]], dtype=np.float64)[:, None] * coef[None, :]
        return pm.Deterministic(self.names["linpred"], eta, dims=("__obs__", dim))

    def _category_specific_design(self) -> dict[str, np.ndarray]:
        """Each category-specific term's column on the fitted rows (brms's ``Xcs``, by column, in term order), checked against
        the graph: its coefficients a free vector over the thresholds' dimension, and the thresholds' predictor ``mu_cs`` moving,
        row by row and threshold by threshold, by exactly ``x_i * c[k]`` from ``mu`` when the coefficients go from 0 to a test
        vector ``c``. Empty without a ``cs()`` term."""
        terms = self.spec.cs_terms
        if not terms:
            return {}
        pmm, dim = self.pymc_model, self.names["threshold_dim"]
        K1 = len(self.categories) - 1
        blocks = {t.operands[0]: np.asarray(self.data[t.operands[0]], dtype=float) for t in terms}
        for t in terms:
            rv = pmm[t.name]
            if rv not in pmm.free_RVs or tuple(pmm.named_vars_to_dims.get(t.name, ())) != (dim,) or pmm.rvs_to_values[rv].name != t.name:
                raise RuntimeError(f"{t.name!r} is not a free vector of the graph over {dim!r}, sampled as it is")
        lin = self.names["linpred"]
        if pmm[lin] not in pmm.deterministics or tuple(pmm.named_vars_to_dims.get(lin, ())) != ("__obs__", dim):
            raise RuntimeError(f"{lin!r} is not the per-row and per-threshold deterministic of the graph")
        fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[self.family_spec.parent], pmm[lin]]), inputs=pmm.value_vars,
                            point_fn=True, on_unused_input="ignore", mode="FAST_COMPILE")
        tests = {t.name: np.cos(np.arange(1.0, K1 + 1.0) * (j + 1.0)) for j, t in enumerate(terms)}
        point = pmm.initial_point()
        point.update(tests)
        mu, mu_cs = (np.asarray(v, dtype=float) for v in fn(point))
        expected = mu[:, None] + sum(blocks[t.operands[0]][:, None] * tests[t.name][None, :] for t in terms)
        if mu_cs.shape != expected.shape or not np.allclose(mu_cs, expected, rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"the graph's {lin!r} is not mu plus the cs() columns times their coefficients")
        return blocks

    # -- Gaussian processes (C03) --------------------------------------------------------
    def _add_gp_effects(self, eta):
        """brms's Gaussian processes, built inside the model Bambi is building (through the same ``transform_backend_eta`` hook
        as the monotonic terms, before the link): for each process its marginal SD ``sdgp`` and length-scale ``lscale`` under
        their priors (sampled on their logs) and its standardised values ``zgp``, and ``eta`` plus the process's values at
        each row's input, ``f[index]`` (``autobayes.gaussian_process``: the exact GP's ``chol(K + 1e-12 I) zgp``, or the
        Hilbert-space basis times ``sqrt(S) * zgp``)."""
        import pymc as pm

        from autobayes.gaussian_process import approximate_effect, exact_effect, positive_rv

        model = pm.modelcontext(None)
        sdgp, lscale = self.priors.sdgp_priors(self.spec), self.priors.lscale_priors(self.spec)
        for g in self.spec.gp_terms:
            roles = self.gp_names[g.column]
            model.add_coord(roles["coordinate"], [str(j) for j in range(1, g.size + 1)])
            s, l = positive_rv(roles["sdgp"], sdgp[g.column]), positive_rv(roles["lscale"], lscale[g.column])
            z = pm.Normal(roles["zgp"], mu=np.float64(0.0), sigma=np.float64(1.0), shape=(g.size,), dims=roles["coordinate"])
            if g.approximate:
                f = approximate_effect(g.basis(pd.Series(g.inputs)), g.slambda, s, l, z)
            else:
                f = exact_effect(g.scaled_inputs, s, l, z)
            eta = eta + f[g.index(self.data[g.column])]
        return eta

    def _gp_design(self) -> dict[str, GPBlock]:
        """The block of each Gaussian process (C03), checked against the graph: ``sdgp`` and ``lscale`` free scalars sampled on
        their logs, ``zgp`` a free vector of the process's size with the coordinate ``"1"``..``"N"``, and the process's
        contribution to the parent's linear predictor recomputed from the data (``_check_gp_effects``). Empty without one."""
        pmm, blocks = self.pymc_model, {}
        for g in self.spec.gp_terms:
            roles = self.gp_names[g.column]
            for key in ("sdgp", "lscale"):
                rv = pmm[roles[key]]
                if rv not in pmm.free_RVs or rv.type.ndim != 0 or pmm.rvs_to_values[rv].name != f"{roles[key]}_log__":
                    raise RuntimeError(f"{roles[key]!r} is not a free positive scalar of the graph sampled on its log")
            z = pmm[roles["zgp"]]
            if (z not in pmm.free_RVs or tuple(z.type.shape) != (g.size,) or pmm.rvs_to_values[z].name != roles["zgp"]
                    or tuple(pmm.named_vars_to_dims.get(roles["zgp"], ())) != (roles["coordinate"],)
                    or tuple(str(v) for v in pmm.coords[roles["coordinate"]]) != tuple(str(j) for j in range(1, g.size + 1))):
                raise RuntimeError(f"{roles['zgp']!r} is not a free vector of {g.size} entries with the coordinate {roles['coordinate']!r}")
            blocks[g.column] = GPBlock(g.term, g.column, g.index(self.data[g.column]), g.scaled_inputs,
                                       g.basis(pd.Series(g.inputs)) if g.approximate else None, g.slambda if g.approximate else None)
        if blocks:
            self._check_gp_effects(blocks)
        return blocks

    def _check_gp_effects(self, blocks) -> None:
        """Each Gaussian process's contribution, read off the graph: with every process at ``sdgp = 1``, ``lscale`` at the
        smallest distance between its scaled inputs (exact; a well-conditioned covariance) or 0.3 (approximate) and ``zgp = 0``,
        and every other variable at the graph's initial point, the parent's linear predictor (on Bambi's link scale) must move,
        row by row, by exactly the process's values at a fixed ``zgp``, recomputed from the data
        (``gaussian_process.exact_values`` / ``approximate_values``), when its ``zgp`` goes from 0 to that vector."""
        from autobayes.gaussian_process import approximate_values, exact_values, smallest_distance

        pmm, parent = self.pymc_model, self.family_spec.parent
        link = self.bambi_model.family.link[parent].name
        if link not in ("identity", "log"):
            raise RuntimeError(f"Bambi's link of {parent!r} is {link}; the Gaussian processes are checked under the identity and log links")
        fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[parent]]), inputs=pmm.value_vars, point_fn=True,
                            on_unused_input="ignore", mode="FAST_COMPILE")
        at0, scales = pmm.initial_point(), {}
        for column, block in blocks.items():
            g, roles = self.spec.gp(column), self.gp_names[column]
            scales[column] = 0.3 if g.approximate else smallest_distance(block.inputs)
            at0.update({f"{roles['sdgp']}_log__": np.asarray(0.0), f"{roles['lscale']}_log__": np.asarray(np.log(scales[column])),
                        roles["zgp"]: np.zeros(g.size)})
        for column, block in blocks.items():
            g, roles, lscale = self.spec.gp(column), self.gp_names[column], scales[column]
            test = np.cos(np.arange(1.0, g.size + 1.0))
            lo, hi = (np.asarray(fn(p)[0], dtype=float).ravel() for p in (at0, {**at0, roles["zgp"]: test}))
            moved = hi - lo if link == "identity" else np.log(hi) - np.log(lo)
            f = (approximate_values(block.basis, block.slambda, 1.0, lscale, test) if g.approximate
                 else exact_values(block.inputs, 1.0, lscale, test))
            expected = f[block.index]
            if moved.shape != expected.shape or not np.allclose(moved, expected, rtol=1e-10, atol=1e-10):
                raise RuntimeError(f"the graph's {parent!r} does not move by the values of {block.term!r} at its rows' inputs; largest "
                                   f"difference {np.max(np.abs(moved - expected)) if moved.shape == expected.shape else moved.shape}")

    # -- multi-membership and by terms (D08) ---------------------------------------------
    def _custom_term_layout(self, t) -> dict:
        """D08: what one multi-membership or ``by`` term reads off the fitted rows: its sorted levels; each member's 0-based
        level index per row and its weights (brms's ``J_m_k - 1`` and ``W_m_k``; one member, weight 1, for a ``by`` term);
        for a ``by`` term the sorted levels of the by column and the 0-based by-level of each level (brms's ``Jby_m - 1``)."""
        levels = self.spec.levels_by_group[t.factor]
        position = {level: j for j, level in enumerate(levels)}
        if t.is_mm:
            index = tuple(np.array([position[str(v)] for v in self.data[c]], dtype=np.int64) for c in t.members)
            weights = mm_weights(t, self.data)
            return {"levels": levels, "index": index, "weights": tuple(weights[:, k] for k in range(len(t.members))),
                    "by_levels": (), "by_index": None}
        index = (np.array([position[str(v)] for v in self.data[t.column]], dtype=np.int64),)
        by_of = by_levels_of(t, self.data)
        by_levels = tuple(sorted(set(by_of.values())))
        return {"levels": levels, "index": index, "weights": (np.ones(len(self.data)),), "by_levels": by_levels,
                "by_index": np.array([by_levels.index(by_of[level]) for level in levels], dtype=np.int64)}

    def _add_custom_group_effects(self, eta):
        """brms's multi-membership terms ``(1 | mm(g, g2, weights = cbind(w1, w2)))`` and ``(1 | gr(g, by = gby))`` (D08), built
        inside the model Bambi is building (``likelihoods.monotonic_family``'s hook on the parent's linear predictor, before the
        link), named as Bambi names a varying intercept: the SD ``1|g_sigma`` under the term's ``sd`` prior (sampled on its log;
        with ``by``, one SD per level of the by column on ``g__by_dim``), the standardised effects ``1|g_offset ~ Normal(0, 1)``
        and the effects ``1|g = sd[by of the level] * 1|g_offset`` (a deterministic on ``g__factor_dim``); each row adds
        ``sum_k W_k * 1|g[J_k]`` (brms's ``W_m_k r_m_1[J_m_k]``), its one level's effect for a ``by`` term."""
        import pymc as pm
        import pytensor.tensor as pt

        model = pm.modelcontext(None)
        sds = self.priors.sd_priors(self.spec)
        for t in self.spec.custom_group_terms:
            roles, layout = self.group_coefficient_names[(t.factor, "Intercept")], self._custom_term_layout(t)
            model.add_coord(roles["coordinate"], list(layout["levels"]))
            prior = sds[(t.factor, "Intercept")]
            dist = getattr(pm, prior.name)
            params = {k: np.float64(v) for k, v in prior.params.items()}
            if t.by is not None:
                model.add_coord(roles["by_coordinate"], list(layout["by_levels"]))
                sd = dist(roles["sd"], **params, dims=roles["by_coordinate"])
                scale = sd[layout["by_index"]]
                # brms's lprior of sd_m (M_m x Nby_m, M_m = 1 here) adds the untruncated density of every entry and subtracts
                # the lower bound's log CCDF once for the term, not once per by level (gauss_gr_by's lprior): for a prior
                # symmetric about 0 (normal, student_t, cauchy) brms's density is log 2 below the half prior's for each by
                # level after the first. The constant is reproduced so that the log density is brms's; it moves no parameter.
                missing = (len(layout["by_levels"]) - 1) * (np.log(2.0) if prior.name in _SYMMETRIC_HALF else 0.0)
                if missing:
                    pm.Potential(roles["by_normalisation"], pt.as_tensor_variable(np.float64(-missing)))
            else:
                scale = dist(roles["sd"], **params)
            z = pm.Normal(roles["offset"], mu=np.float64(0.0), sigma=np.float64(1.0), dims=roles["coordinate"])
            r = pm.Deterministic(roles["group_effect"], scale * z, dims=roles["coordinate"])
            for index, weight in zip(layout["index"], layout["weights"]):
                eta = eta + (r[index] if t.by is not None else np.asarray(weight, dtype=np.float64) * r[index])
        return eta

    def _custom_group_blocks(self) -> dict[str, GroupBlock]:
        """The block of each multi-membership and ``by`` term (D08), by factor, checked against the graph: the SD a free
        positive variable sampled on its log (a vector over the by levels with ``by``), the standardised effects a free vector
        and the effects a deterministic on the level dimension, the coordinates the sorted levels; and the term's contribution
        to the parent's linear predictor recomputed from the data (``_check_custom_group_effects``). ``Z`` is the weighted sum
        of the members' indicators (the indicator of the level for a ``by`` term)."""
        pmm, blocks = self.pymc_model, {}
        start = pmm.initial_point()
        for t in self.spec.custom_group_terms:
            roles, layout = self.group_coefficient_names[(t.factor, "Intercept")], self._custom_term_layout(t)
            levels, n = layout["levels"], len(self.data)
            sd = pmm[roles["sd"]]
            shape = (len(layout["by_levels"]),) if t.by is not None else ()
            if (sd not in pmm.free_RVs or pmm.rvs_to_values[sd].name != f"{roles['sd']}_log__"
                    or np.shape(start.get(f"{roles['sd']}_log__")) != shape):
                raise RuntimeError(f"{roles['sd']!r} is not a free positive variable of shape {shape} sampled on its log")
            if t.by is not None and (tuple(pmm.named_vars_to_dims.get(roles["sd"], ())) != (roles["by_coordinate"],)
                                     or tuple(str(v) for v in pmm.coords[roles["by_coordinate"]]) != layout["by_levels"]):
                raise RuntimeError(f"PyMC dims of {roles['sd']!r} are not ({roles['by_coordinate']!r},) with the levels of {t.by!r}")
            if tuple(str(v) for v in pmm.coords[roles["coordinate"]]) != levels:
                raise RuntimeError(f"PyMC coordinate {roles['coordinate']!r} differs from the fitted labels")
            if pmm[roles["offset"]] not in pmm.free_RVs or pmm[roles["group_effect"]] not in pmm.deterministics:
                raise RuntimeError(f"{roles['offset']!r} is not a free variable or {roles['group_effect']!r} not a deterministic")
            for key in ("offset", "group_effect"):
                if tuple(pmm.named_vars_to_dims.get(roles[key], ())) != (roles["coordinate"],):
                    raise RuntimeError(f"PyMC dims of {roles[key]!r} are not ({roles['coordinate']!r},)")
            Z = np.zeros((n, len(levels)))
            for index, weight in zip(layout["index"], layout["weights"]):
                np.add.at(Z, (np.arange(n), index), weight)
            blocks[t.factor] = GroupBlock(levels, Z, layout["index"][0], slice(0, len(levels)),
                                          memberships=layout["index"] if t.is_mm else (),
                                          weights=layout["weights"] if t.is_mm else (),
                                          by_levels=layout["by_levels"], by_index=layout["by_index"])
        self._check_custom_group_effects(blocks)
        return blocks

    def _check_custom_group_effects(self, blocks) -> None:
        """Each multi-membership and ``by`` term's contribution (D08), read off the graph: with every such term's SD at 1 and
        standardised effects at 0, and every other variable at the graph's initial point, the parent's linear predictor (on
        Bambi's link scale: the parent itself, or the log of ``mu`` under a log link) must move, row by row, by exactly
        ``sum_k W_k * test[J_k]`` (``Z @ test``, recomputed from the data) when the term's standardised effects go from 0 to
        a fixed vector ``test``."""
        pmm, parent = self.pymc_model, self.family_spec.parent
        link = self.bambi_model.family.link[parent].name
        if link not in ("identity", "log"):
            raise RuntimeError(f"Bambi's link of {parent!r} is {link}; the mm() and gr(by = ) terms are checked under the identity "
                               "and log links")
        fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[parent]]), inputs=pmm.value_vars, point_fn=True,
                            on_unused_input="ignore", mode="FAST_COMPILE")
        at0 = pmm.initial_point()
        for factor in blocks:
            roles = self.group_coefficient_names[(factor, "Intercept")]
            at0[f"{roles['sd']}_log__"] = np.zeros(np.shape(at0[f"{roles['sd']}_log__"]))
            at0[roles["offset"]] = np.zeros(blocks[factor].J)
        for factor, block in blocks.items():
            roles = self.group_coefficient_names[(factor, "Intercept")]
            test = np.cos(np.arange(1.0, block.J + 1.0))
            lo, hi = (np.asarray(fn(p)[0], dtype=float).ravel() for p in (at0, {**at0, roles["offset"]: test}))
            moved = hi - lo if link == "identity" else np.log(hi) - np.log(lo)
            expected = block.Z @ test
            if moved.shape != expected.shape or not np.allclose(moved, expected, rtol=1e-10, atol=1e-10):
                raise RuntimeError(f"the graph's {parent!r} does not move by the weighted effects of {factor!r} at its rows' levels; "
                                   f"largest difference {np.max(np.abs(moved - expected)) if moved.shape == expected.shape else moved.shape}")

    # -- autocorrelation terms (C04) -----------------------------------------------------
    def _add_autocor_effects(self, eta):
        """brms's ARMA term, built inside the model Bambi is building (``autocor.autocor_family``, which Bambi calls on the
        parent's linear predictor before the link, after every other term): ``ar`` and ``ma`` under their priors, vectors on
        the coordinates ``"1"``..``"p"`` (``"1"``..``"q"``), and ``eta`` plus the term on the residuals of the fitted
        responses along each series (``autocor.arma_effect``)."""
        import pymc as pm

        from autobayes.autocor import arma_effect, coefficient_vector

        model = pm.modelcontext(None)
        roles, priors, coefficients = self.autocor_names, self.priors.autocor_priors(self.spec), {}
        for part, order in self.spec.autocor.parts:
            model.add_coord(roles[f"{part}_coordinate"], [str(k) for k in range(1, order + 1)])
            coefficients[part] = coefficient_vector(roles[part], priors[part], roles[f"{part}_coordinate"])
        y = np.asarray(self.data[self.spec.response], dtype=float)
        return arma_effect(eta, y, coefficients.get("ar"), coefficients.get("ma"), self._autocor_design)

    def _check_autocor_effects(self) -> None:
        """The autocorrelation term read off the graph (C04): ``ar`` and ``ma`` free, unconstrained vectors of ``p`` and ``q``
        entries on the coordinates ``"1"``.., and the parent's linear predictor, at coefficients with distinct alternating
        entries, equal row by row to brms's recursion (``autocor.arma_values``: a loop along brms's sorted rows with
        ``J_lag``) applied to the predictor at ``ar = ma = 0``, where the term vanishes; every other variable sits a small,
        distinct step from the graph's initial point, so that the predictor varies by row."""
        from autobayes.autocor import arma_values, probe_coefficients

        pmm, roles, parent = self.pymc_model, self.autocor_names, self.family_spec.parent
        if self.bambi_model.family.link[parent].name != "identity":
            raise RuntimeError(f"Bambi's link of {parent!r} is {self.bambi_model.family.link[parent].name}; the autocorrelation "
                               "term is checked under the identity link")
        start = pmm.initial_point()
        for part, order in self.spec.autocor.parts:
            rv = pmm[roles[part]]
            if rv not in pmm.free_RVs or pmm.rvs_to_values[rv].name != roles[part] or np.shape(start[roles[part]]) != (order,):
                raise RuntimeError(f"{roles[part]!r} is not a free, unconstrained vector of {order} entries in the graph")
            coordinate = roles[f"{part}_coordinate"]
            if (tuple(pmm.named_vars_to_dims.get(roles[part], ())) != (coordinate,)
                    or tuple(str(v) for v in pmm.coords[coordinate]) != tuple(str(k) for k in range(1, order + 1))):
                raise RuntimeError(f"PyMC dims of {roles[part]!r} are not ({coordinate!r},) with coordinates 1..{order}")
        fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[parent]]), inputs=pmm.value_vars, point_fn=True,
                            on_unused_input="ignore", mode="FAST_COMPILE")
        coefficient_vars = {roles[part] for part, _ in self.spec.autocor.parts}
        point = {name: np.asarray(v, dtype=float) + (0.0 if name in coefficient_vars else
                                                     0.05 * (1.0 + np.arange(np.size(v))).reshape(np.shape(v)) / max(np.size(v), 1))
                 for name, v in start.items()}
        probe = {part: probe_coefficients(order, 0.6 if part == "ar" else -0.45) for part, order in self.spec.autocor.parts}
        eta = np.asarray(fn({**point, **{roles[part]: np.zeros(order) for part, order in self.spec.autocor.parts}})[0], dtype=float).ravel()
        got = np.asarray(fn({**point, **{roles[part]: value for part, value in probe.items()}})[0], dtype=float).ravel()
        expected = arma_values(eta, self.design.y, probe.get("ar"), probe.get("ma"), self.design.autocor)
        if got.shape != expected.shape or not np.allclose(got, expected, rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"the graph's {parent!r} does not add brms's ARMA term {self.spec.autocor.term!r} on the residuals; "
                               f"largest difference {np.max(np.abs(got - expected)) if got.shape == expected.shape else got.shape}")

    # -- measurement-error and missing-value terms (C06) ---------------------------------
    def _add_measurement_effects(self, eta):
        """brms's ``me()`` and ``mi()`` terms (C06, docs/missing-and-measurement-error.md), built inside the model Bambi is
        building (``likelihoods.monotonic_family``'s hook on the parent's linear predictor, before the link). ``me(x, sdx)``:
        the location ``meanme_x`` and scale ``sdme_x`` of the latent values under their priors, the standardised latent values
        ``zme_x ~ Normal(0, 1)``, one per row, the latent values ``Xme_x = meanme_x + sdme_x * zme_x`` (a deterministic, brms's
        ``Xme``), the noisy values observed as ``Xn_x ~ Normal(Xme_x, sdx)`` (brms's ``Xn`` and ``noise``) and the coefficient
        (brms's ``bsp``) under its ``b`` prior, adding ``coefficient * Xme_x``. ``mi(col)``: the column's missing values
        ``Ymi_col``, a free variable under brms's flat prior on the missing rows, completing the observed ones (brms's ``Yl``),
        and the coefficient, adding ``coefficient * Yl``. Neither term is centred, as brms centres neither."""
        import pymc as pm
        import pytensor.tensor as pt

        from autobayes.monotonic import coefficient_rv

        model = pm.modelcontext(None)
        coefficients = self.priors.coefficient_priors(self.spec)
        means, sds = self.priors.me_priors(self.spec, "meanme"), self.priors.me_priors(self.spec, "sdme")
        for t in self.spec.me_terms:
            x, sdx = t.operands
            roles = self.me_names[x]
            noisy, noise = measurement_values(t.name, self.data[x], self.data[sdx])
            mean = _prior_rv(roles["mean"], means[x])
            sd = _prior_rv(roles["sd"], sds[x], positive=True)
            z = pm.Normal(roles["z"], mu=np.float64(0.0), sigma=np.float64(1.0), dims="__obs__")
            latent = pm.Deterministic(roles["latent"], mean + sd * z, dims="__obs__")
            pm.Normal(roles["noisy"], mu=latent, sigma=np.asarray(noise, dtype=np.float64), observed=noisy, dims="__obs__")
            eta = eta + coefficient_rv(roles["b"], coefficients[t.name]) * latent
        for t in self.spec.mi_terms:
            col = t.operands[0]
            block, roles = self._missing_block(col), self.missing_names.get(col)
            values = pt.as_tensor(block.filled)
            if roles is not None:                            # brms's Yl: the observed values, Ymi at the missing rows
                model.add_coord(roles["coordinate"], [int(i) for i in block.index])
                values = pt.set_subtensor(values[block.index], pm.Flat(roles["values"], dims=roles["coordinate"]))
            eta = eta + coefficient_rv(t.name, coefficients[t.name]) * values
        return eta

    def _missing_block(self, column: str) -> MissingBlock:
        """The fitted rows of a column whose missing values are parameters (C06), with its missing rows."""
        values = missing_values(f"column {column!r}", self.data[column], column == self.spec.response)
        return MissingBlock(column, values, np.flatnonzero(np.isnan(values)))

    def _measurement_design(self) -> dict[str, MeasurementBlock]:
        """The block of each ``me(x, sdx)`` term (C06), checked against the graph: the location a free scalar, the scale a
        free scalar sampled on the log scale, the standardised latent values a free vector on ``__obs__``, the latent values a
        deterministic on ``__obs__`` and the noisy values an observed variable holding the data's; the coefficient a free
        scalar. Empty without an ``me()`` term."""
        pmm, blocks = self.pymc_model, {}
        for t in self.spec.me_terms:
            x, sdx = t.operands
            roles = self.me_names[x]
            noisy, noise = measurement_values(t.name, self.data[x], self.data[sdx])
            for key in ("mean", "sd", "b"):
                if pmm[roles[key]] not in pmm.free_RVs or pmm[roles[key]].type.ndim != 0:
                    raise RuntimeError(f"{roles[key]!r} is not a free scalar of the graph")
            if pmm.rvs_to_values[pmm[roles["sd"]]].name != f"{roles['sd']}_log__":
                raise RuntimeError(f"{roles['sd']!r} is not sampled on the log scale")
            if pmm[roles["z"]] not in pmm.free_RVs or pmm[roles["latent"]] not in pmm.deterministics:
                raise RuntimeError(f"{roles['z']!r} is not a free variable or {roles['latent']!r} not a deterministic of the graph")
            for key in ("z", "latent", "noisy"):
                if tuple(pmm.named_vars_to_dims.get(roles[key], ())) != ("__obs__",):
                    raise RuntimeError(f"PyMC dims of {roles[key]!r} are not ('__obs__',)")
            observed = pmm[roles["noisy"]]
            held = np.asarray(getattr(pmm.rvs_to_values.get(observed), "data", np.nan), dtype=float)
            if observed not in pmm.observed_RVs or held.shape != noisy.shape or not np.array_equal(held, noisy):
                raise RuntimeError(f"{roles['noisy']!r} is not the observed noisy values of {t.name!r}")
            blocks[x] = MeasurementBlock(t.name, x, sdx, noisy, noise)
        return blocks

    def _missing_design(self) -> dict[str, MissingBlock]:
        """The block of each column whose missing values are parameters (C06: the response's under ``| mi()``, each ``mi()``
        term's column), checked against the graph: the missing values a free variable with brms's flat prior on the
        coordinate of the missing rows. Empty without one."""
        pmm, blocks = self.pymc_model, {}
        for col in self.spec.missing_columns:
            block = self._missing_block(col)
            roles = self.missing_names.get(col)
            if roles is not None:
                rv = pmm[roles["values"]]
                if (rv not in pmm.free_RVs or type(rv.owner.op).__name__ != "FlatRV" or rv.type.ndim != 1
                        or pmm.rvs_to_values[rv].name != roles["values"]):
                    raise RuntimeError(f"{roles['values']!r} is not a free flat vector of the {block.index.size} missing values of {col!r}")
                if (tuple(pmm.named_vars_to_dims.get(roles["values"], ())) != (roles["coordinate"],)
                        or [int(v) for v in pmm.coords[roles["coordinate"]]] != block.index.tolist()):
                    raise RuntimeError(f"PyMC dims of {roles['values']!r} are not the missing rows of {col!r}")
            blocks[col] = block
        return blocks

    def _check_measurement_effects(self) -> None:
        """Each ``me()`` and ``mi()`` term's contribution, read off the graph (C06): at the graph's initial point with test values
        of the latent values (``meanme = 0.3``, ``sdme = 0.7``, ``zme`` from -1 to 1) and of the missing values (from -1 to 1),
        the parent's linear predictor (on Bambi's link scale) must move, row by row, by exactly ``c`` times the term's values
        (``Xme``, or the observed values completed by the missing ones) when the term's coefficient goes from 0 to ``c``."""
        pmm, parent = self.pymc_model, self.family_spec.parent
        link = self.bambi_model.family.link[parent].name
        if link not in ("identity", "log"):
            raise RuntimeError(f"Bambi's link of {parent!r} is {link}; the me() and mi() terms are checked under the identity and log links")
        fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[parent]]), inputs=pmm.value_vars, point_fn=True,
                            on_unused_input="ignore", mode="FAST_COMPILE")
        point, terms = pmm.initial_point(), []
        for t in self.spec.me_terms:
            roles = self.me_names[t.operands[0]]
            z = np.linspace(-1.0, 1.0, self.design.n)
            point.update({roles["mean"]: np.asarray(0.3), f"{roles['sd']}_log__": np.asarray(np.log(0.7)), roles["z"]: z})
            terms.append((t.name, roles["b"], 0.3 + 0.7 * z))
        for t in self.spec.mi_terms:
            block, roles = self.design.missing[t.operands[0]], self.missing_names.get(t.operands[0])
            values = block.filled.copy()
            if roles is not None:
                values[block.index] = np.linspace(-1.0, 1.0, block.index.size)
                point[roles["values"]] = values[block.index]
            terms.append((t.name, t.name, values))
        point.update({b: np.asarray(0.0) for _, b, _ in terms})
        for name, b, values in terms:
            c = 1.0 / max(1.0, float(np.max(np.abs(values))))
            lo, hi = (np.asarray(fn(p)[0], dtype=float).ravel() for p in (point, {**point, b: np.asarray(c)}))
            moved = hi - lo if link == "identity" else np.log(hi) - np.log(lo)
            if moved.shape != values.shape or not np.allclose(moved, c * values, rtol=1e-10, atol=1e-10):
                raise RuntimeError(f"the graph's {parent!r} does not move by the values of {name!r} times its coefficient")

    @property
    def me_names(self) -> dict[str, dict[str, str]]:
        """PyMC names of each measurement-error term ``me(x, sdx)`` (C06), by its noisy column: ``"b"`` its coefficient (brms's
        ``bsp``; the term's own name, ``me(x, sdx)``, so ``names["b:me(x, sdx)"]`` too), ``"mean"`` and ``"sd"`` the location and
        scale of its latent values (brms's ``meanme`` and ``sdme``; ``meanme_x``, ``sdme_x``, the latter sampled as
        ``sdme_x_log__``), ``"z"`` the standardised latent values (brms's ``zme``, ``zme_x``, one per row), ``"latent"`` the latent
        values (brms's ``Xme``, the deterministic ``Xme_x``) and ``"noisy"`` the observed noisy values (brms's ``Xn``, ``Xn_x``).
        Empty without an ``me()`` term."""
        return {t.operands[0]: {"b": t.name, "mean": f"meanme_{t.operands[0]}", "sd": f"sdme_{t.operands[0]}",
                                "z": f"zme_{t.operands[0]}", "latent": f"Xme_{t.operands[0]}", "noisy": f"Xn_{t.operands[0]}"}
                for t in self.spec.me_terms}

    @property
    def missing_names(self) -> dict[str, dict[str, str]]:
        """PyMC names of the missing values this model holds as parameters (C06), by column, for each column of
        ``spec.missing_columns`` with a missing fitted row: ``"values"`` the free variable ``Ymi_<col>`` (brms's ``Ymi_<col>``,
        flat) and ``"coordinate"`` its dimension ``Ymi_<col>_dim``, whose coordinates are the missing rows, 0-based (brms's
        ``Jmi`` is 1-based). In a multivariate model the response that imputes the column and every response with an
        ``mi(col)`` term share it. Empty without one."""
        out = {}
        for col in self.spec.missing_columns:
            if self.data[col].isna().any():
                out[col] = {"values": f"Ymi_{col}", "coordinate": f"Ymi_{col}_dim"}
        return out

    # -- naming ------------------------------------------------------------------------
    @property
    def link(self) -> str:
        """The link the model was built with: the family's own ("identity", "log", "logit") or the one given to
        ``link=`` ("probit" for the Bernoulli and binomial families; docs/links.md)."""
        return self.family_spec.link

    @property
    def learned_parameters(self) -> dict[str, float]:
        """The family's fixed parameters this model learns instead, each under its own prior, by name with the lower
        bound it is truncated at: ``{"nu": 1.0}`` for a Student-t model built with ``nu=None`` and ``Priors(nu=...)``
        (A13, docs/student-learned-nu.md); empty otherwise."""
        return {name: lower for name, lower in self.family_spec.learnable_parameters.items() if getattr(self, name) is None}

    @property
    def names(self) -> dict[str, str]:
        """PyMC variable names for each role in the model. ``"linpred"`` is the deterministic
        Bambi builds for the likelihood's parent parameter (``mu`` for Gaussian, ``logit_p``
        for Bernoulli); ``"mean"`` is the name ``predict`` gives the expected response (``mu``
        or ``p``; ``mean`` for the lognormal, whose ``mu`` is the log-scale location). ``"sigma"`` exists for the
        Gaussian, Student-t and lognormal families, ``"shape"`` for the negative binomial and Gamma families (its
        value is Bambi's variable name, ``alpha``) and ``"phi"`` for the Beta family (the precision); ``"nu"`` for a Student-t
        model that learns its degrees of freedom (``learned_parameters``); ``"zi"`` or ``"hu"`` for the zero-inflated and
        hurdle families (B06), whose linear predictor is ``"log_mu"`` and whose ``"mean"`` is the expected count; ``"alpha"``,
        ``"beta"`` or ``"ndt"`` for the skew-normal, ex-Gaussian and shifted lognormal families (D01); ``"phi"`` for the
        beta-binomial (its precision) and ``"sigma"`` for ``negbinomial2`` (its dispersion, brms's ``1 / shape``; D02). ``"b:<term>"`` is the
        coefficient variable of each population-level term (scalar for a numeric term, a vector
        over the coefficient levels otherwise); with exactly one group-level term its ``sd``,
        ``offset`` and ``group_effect`` roles are included here as well, and ``sd:x``, ``offset:x``,
        ``group_effect:x`` for each varying slope ``x`` of that term (see ``group_names``). An auxiliary parameter with its own
        formula (B01) keeps its entry, now a per-row deterministic; its formula's variables are in ``dpar_names``; so do a
        zero parameter and an extra parameter with a formula of their own (E03: ``zi``, ``hu``, ``alpha``)."""
        single = self.spec.group_terms[0] if len(self.spec.group_terms) == 1 else None
        fam = self.family_spec
        out = {
            **({"intercept": "Intercept"} if self.spec.intercept else {}),
            **({k: v for k, v in self.group_names[single.factor].items() if k != "coordinate"} if single is not None else {}),
            "linpred": fam.parent,
            "mean": fam.mean_name,
            "y": self.spec.response if self.spec.trials is None else f"p({self.spec.response}, {self.spec.trials})",
            **{f"b:{t.name}": t.name for t in self.spec.pop_terms},
            **{f"simo:{column}": roles["simplex"] for column, roles in self.mo_names.items()},   # B03
            **{f"{key}:{column}": roles[key] for column, roles in self.gp_names.items() for key in ("sdgp", "lscale", "zgp")},   # C03
            **{f"{label}:{x}": roles[key] for x, roles in self.me_names.items()                   # C06: me()'s latent values
               for key, label in (("mean", "meanme"), ("sd", "sdme"), ("z", "zme"), ("latent", "Xme"), ("noisy", "Xn"))},
            **{f"Ymi:{col}": roles["values"] for col, roles in self.missing_names.items()},       # C06: the missing values
        }
        if fam.scale_parameter is not None:
            out[fam.scale_parameter] = fam.bambi_scale_parameter
        if fam.zero_parameter is not None:                    # "zi" or "hu" (B06): the PyMC variable keeps brms's name
            out[fam.zero_parameter] = fam.zero_parameter
        for extra in fam.extra_parameters:                    # "alpha", "beta" or "ndt" (D01): brms's name
            out[extra.name] = extra.name
            if extra.derived:                                 # E02: "tmp_xi", the sampled variable beside the derived "xi"
                out[extra.variable] = extra.variable
        for name in self.learned_parameters:                  # a learned nu (A13): the PyMC variable of that name
            out[name] = name
        if fam.thresholds:                                    # B05: brms's b_Intercept, the thresholds of the raw design, and
            out["thresholds"] = "b_Intercept"                 # the coordinate of the thresholds ("1", ..., "K - 1")
            out["threshold_dim"] = "Intercept_dim"
        if self.spec.cs_terms:                                # D04: each threshold's predictor, mu plus Xcs bcs (brms's
            out["linpred"] = "mu_cs"                          # posterior_linpred of a model with cs() terms)
        if fam.per_category:                                  # B05: the coordinate of the non-reference categories
            out["category_dim"] = f"{self.spec.response}_reduced_dim"
        if fam.mixture:                                       # C05: each component's sigma<k>, theta and the component coordinate
            roles = self.mixture_names
            out.update({name: name for name in roles["sigma"]}, theta=roles["theta"], component_dim=roles["coordinate"])
        if fam.continuous and fam.parent == "mu":
            out["mu"] = "mu"                                  # the continuous families whose parent is the location mu expose it by name (the Beta's parent is logit_mu)
        out.update({k: v for k, v in self.autocor_names.items() if not k.endswith("_coordinate")})   # "ar", "ma" (C04)
        out.update(self.sar_names)                            # "lagsar" or "errorsar" (F02b)
        return out

    @property
    def mixture_names(self) -> dict:
        """PyMC names of a mixture's own parameters (C05, docs/mixtures.md; ``families.mixture.roles``): ``"dpars"`` brms's
        names of the components' linear predictors (``mu1``, ...), ``"sigma"`` each component's residual SD (``sigma1``, ...),
        ``"theta"`` the mixing proportions, ``"coordinate"`` the component dimension (``mixture_dim``) of the intercepts,
        coefficients and ``theta``, and ``"labels"`` its coordinates ``"1"``..``"K"``. Empty for any other model."""
        from autobayes.families.mixture import roles

        components = getattr(self, "components", ())
        return roles(len(components)) if components else {}

    @property
    def mo_names(self) -> dict[str, dict[str, str]]:
        """PyMC names of each monotonic term ``mo(x)`` (B03), by its column in formula order: ``"b"`` its coefficient (brms's
        ``bsp``; the term's own name, ``mo(x)``, so ``names["b:mo(x)"]`` too), ``"simplex"`` its simplex (brms's ``simo``,
        ``simo_x``, sampled on Stan's stick-breaking coordinates ``simo_x_stickbreaking__``; ``names["simo:x"]``) and
        ``"coordinate"`` the simplex's dimension (``simo_x_dim``, coordinates ``"1"``..``"D"``, one per step). Empty without
        a monotonic term."""
        return {t.operands[0]: {"b": t.name, "simplex": f"simo_{t.operands[0]}", "coordinate": f"simo_{t.operands[0]}_dim"}
                for t in self.spec.mo_terms}

    @property
    def gp_names(self) -> dict[str, dict[str, str]]:
        """PyMC names of each Gaussian process ``gp(x)`` (C03), by its column in formula order: ``"sdgp"`` its marginal SD
        (brms's ``sdgp``; ``sdgp_x``, sampled as ``sdgp_x_log__``), ``"lscale"`` its length-scale on the scaled input (brms's
        ``lscale``; ``lscale_x``, sampled as ``lscale_x_log__``), ``"zgp"`` its standardised values (brms's ``zgp``; ``zgp_x``,
        one per distinct input of the exact GP or per basis function of the approximate one) and ``"coordinate"`` their
        dimension (``zgp_x_dim``, coordinates ``"1"``..``"N"``); also ``names["sdgp:x"]`` and so on. Empty without one."""
        return {g.column: {"sdgp": f"sdgp_{g.column}", "lscale": f"lscale_{g.column}", "zgp": f"zgp_{g.column}",
                           "coordinate": f"zgp_{g.column}_dim"} for g in self.spec.gp_terms}

    @property
    def autocor_names(self) -> dict[str, str]:
        """PyMC names of an autocorrelation term's coefficients (C04): ``"ar"`` and ``"ma"`` (brms's names; vectors of ``p``
        and ``q`` entries, also ``names["ar"]`` and ``names["ma"]``) and their dimensions ``"ar_coordinate"`` (``ar_dim``,
        coordinates ``"1"``..``"p"``) and ``"ma_coordinate"`` (``ma_dim``), for the parts the term has. Empty without one."""
        term = getattr(self.spec, "autocor", None)            # a non-linear model's spec has none (C02)
        out: dict[str, str] = {}
        for part, _ in (term.parts if term is not None else ()):
            out[part], out[f"{part}_coordinate"] = part, f"{part}_dim"
        return out

    @property
    def sar_names(self) -> dict[str, str]:
        """PyMC name of a SAR term's parameter (F02b): ``{"lagsar": "lagsar"}`` or ``{"errorsar": "errorsar"}`` (brms's name; a
        scalar sampled as ``<name>_interval__``, also ``names["lagsar"]``). Empty without one."""
        term = getattr(self.spec, "sar", None)                # a non-linear model's spec has none (C02)
        return {term.parameter: term.parameter} if term is not None else {}

    @property
    def shrinkage_names(self) -> dict:
        """PyMC names of a shrinkage prior on b (D07, docs/shrinkage-priors.md; ``shrinkage.names``): ``"prior"`` its name
        (``Horseshoe`` or ``R2D2``), ``"zb"`` the standardised coefficients, ``"sdb"`` their scales, ``"coordinate"`` the
        dimension of both (``zb_dim``, the design columns), and the prior's own variables under brms's names (``hs_global``,
        ``hs_slab``, ``hs_local``; ``R2D2_R2``, ``R2D2_phi``, ``R2D2_tau2``), with ``"lprior"`` those whose densities brms
        sums into ``lprior``. Empty without one."""
        from autobayes.shrinkage import names

        return names(getattr(self.priors, "b", None))

    @property
    def dpar_names(self) -> dict[str, dict[str, str]]:
        """PyMC names of each distributional parameter's own formula (B01), by brms name in formula order: ``"value"`` the
        parameter itself, a deterministic with one value per row (``sigma``; Bambi's ``alpha`` for brms's ``shape``;
        the same variable as ``names[<parameter>]``), ``"intercept"`` its formula's intercept (``sigma_Intercept``; the
        sampled value is centred, brms's ``Intercept_sigma``, and a posterior from ``fit()`` holds Bambi's raw-scale one,
        brms's ``b_sigma_Intercept``) when it has one, and ``"b:<term>"`` the coefficient variable of each of its
        population-level terms (``sigma_z``, a vector over ``sigma_z_dim`` for a vector term; brms's ``b_sigma``). Empty
        without a distributional formula."""
        out = {}
        for d in self.spec.dpars:
            b = self._bambi_dpar(d.response)
            out[d.response] = {"value": b, **({"intercept": f"{b}_Intercept"} if d.intercept else {}),
                               **{f"b:{t.name}": f"{b}_{t.name}" for t in d.pop_terms}}
        return out

    @staticmethod
    def _group_roles(factor: str, coef: str = "Intercept") -> dict[str, str]:
        """PyMC names of one group-level coefficient of ``factor`` (its varying intercept by default): ``sd`` is
        the coefficient's scale (brms ``sd_m[c]``), ``offset`` its standardised effects (brms ``z_m[c]``),
        ``group_effect`` its effects per level (brms ``r_m_c``), ``coordinate`` the level dimension, shared by
        every coefficient of the factor. Bambi names all of them after the factor's column (``g_h`` for ``g:h``)."""
        name = _bambi_group_name(factor, coef)
        return {"sd": f"{name}_sigma", "offset": f"{name}_offset", "group_effect": name, "coordinate": f"{factor_column(factor)}__factor_dim"}

    @staticmethod
    def _correlation_roles(factor: str) -> dict[str, str]:
        """PyMC names of a correlated term's correlation structure (A02): ``L`` the Cholesky factor of the correlation
        matrix (brms ``L_m``; its value variable is ``L|g_cholesky_corr__``), ``cor`` the correlation matrix (brms
        ``Cor_m``), ``coordinate`` / ``coordinate_T`` the two coefficient dimensions of those matrices."""
        column = factor_column(factor)
        return {"L": f"L|{column}", "cor": f"cor|{column}", "coordinate": f"{column}__cor_dim", "coordinate_T": f"{column}__cor_dim_T"}

    @property
    def correlation_names(self) -> dict[str, dict[str, str]]:
        """``_correlation_roles`` of every correlated group term, by factor in formula order, then of every correlation ID
        shared with a distributional formula (D08; empty without one)."""
        out = {t.factor: self._correlation_roles(t.factor) for t in self.spec.correlated_terms}
        out.update({s.factor: self._correlation_roles(s.factor) for s in getattr(self.spec, "shared_ids", ())})
        return out

    def correlation_coefs(self, factor: str) -> tuple[str, ...]:
        """The coefficients of the correlation matrix of ``factor``, its coordinates: the correlated term's (``Intercept``,
        ``x``), or a shared correlation ID's, brms's names across formulas (``Intercept``, ``x``, ``sigma_Intercept``; D08)."""
        shared = self.spec.shared_id_of(factor) if hasattr(self.spec, "shared_id_of") else None
        return shared.labels if shared is not None else self.spec.group_term_of(factor).coefs

    def _shared_roles(self, factor: str, dpar: str, coef: str) -> dict[str, str]:
        """PyMC names of one coefficient of a shared correlation ID (D08): the response formula's (``dpar`` empty) are
        ``_group_roles``; a distributional formula's carry Bambi's prefix of its component (``sigma_1|g``, ``sigma_1|g_sigma``,
        ``sigma_1|g_offset``), on the factor's own level dimension."""
        roles = self._group_roles(factor, coef)
        if not dpar:
            return roles
        prefix = f"{self._bambi_dpar(dpar)}_"
        return {k: (v if k == "coordinate" else prefix + v) for k, v in roles.items()}

    @property
    def dpar_group_names(self) -> dict[tuple[str, str, str], dict[str, str]]:
        """PyMC roles (``sd``, ``offset``, ``group_effect``, ``coordinate``) of every group-level coefficient of a distributional
        formula (D08: the terms sharing a correlation ID with the response formula's), keyed ``(parameter, factor,
        coefficient)``; empty without one."""
        return {(d.response, t.factor, c): self._shared_roles(t.factor, d.response, c)
                for d in self.spec.dpars for t in d.group_terms for c in t.coefs}

    @property
    def group_names(self) -> dict[str, dict[str, str]]:
        """PyMC roles and coordinate of each group-level term's varying intercept, by factor in formula order; a
        term with varying slopes also lists each slope's roles as ``sd:x``, ``offset:x`` and ``group_effect:x``
        (``group_coefficient_names`` has the same names keyed by coefficient), and a correlated term its ``L`` and
        ``cor`` (``correlation_names``)."""
        out = {}
        for t in self.spec.group_terms:
            roles = self._group_roles(t.factor)
            for coef in t.slopes:
                roles.update({f"{k}:{coef}": v for k, v in self._group_roles(t.factor, coef).items() if k != "coordinate"})
            if t.has_correlation or t.id is not None:          # a correlated term, or a shared correlation ID (D08)
                roles.update({k: v for k, v in self._correlation_roles(t.factor).items() if k in ("L", "cor")})
            if t.by is not None:                               # D08: the dimension of gr(by = )'s SD
                roles["by_coordinate"] = f"{factor_column(t.factor)}__by_dim"
            out[t.factor] = roles
        return out

    @property
    def group_coefficient_names(self) -> dict[tuple[str, str], dict[str, str]]:
        """PyMC roles (``sd``, ``offset``, ``group_effect``, ``coordinate``) of every group-level coefficient, keyed
        ``(factor, coefficient)`` in formula order like ``Design.group_blocks``. A term with ``gr(g, by = gby)`` (D08) also has
        ``by_coordinate``, the dimension of its SD (``g__by_dim``, one entry per level of ``gby``)."""
        by = {t.factor: t.by for t in self.spec.group_terms}
        return {(g, coef): {**self._group_roles(g, coef),
                            **({"by_coordinate": f"{factor_column(g)}__by_dim",
                                "by_normalisation": f"{_bambi_group_name(g, coef)}_sigma_by_normalisation"} if by[g] else {})}
                for g, coef in self.spec.group_coefficients}

    @property
    def coefficients(self) -> tuple[tuple[str, str, int | None], ...]:
        """``(design column, PyMC variable, index into that variable or None)`` in design order:
        ``("x", "x", None)`` for a numeric slope, ``("z[b]", "z", 0)`` for a dummy coefficient
        (the variable ``z`` has dims ``z_dim`` with the coefficient levels as coordinates); one entry
        per column of each ``PopTerm``."""
        return self.spec.column_terms

    def value_point(self, *, intercept: float | None = None, b=None, sd=None, z=None, L=None, sigma=None,
                    shape=None, phi=None, nu: float | None = None,
                    zi: float | None = None, hu: float | None = None, bsp=None, simo=None, theta=None,
                    sdgp=None, lscale=None, zgp=None, ar=None, ma=None, meanme=None, sdme=None, zme=None,
                    Ymi=None, alpha=None, beta=None, ndt=None, shrinkage=None, bcs=None, zoi=None, coi=None,
                    kappa=None, quantile=None, tmp_xi=None, lagsar=None, errorsar=None) -> dict[str, np.ndarray]:
        """A point in PyMC's sampled (value) space, from natural-scale quantities.

        ``intercept`` is the *model* intercept (at the predictor means when centred),
        ``b`` the coefficients in design-column order (``model.coefficients``: numeric slopes
        and dummy coefficients), ``sd`` and ``z`` exact-key factor mappings of positive
        scales and standard Normal offset vectors. An intercept-only factor's entry is the
        scale / the vector itself (or the one-entry mapping ``{"Intercept": ...}``); a factor with
        varying slopes takes a mapping over exactly its coefficients, ``sd={"g": {"Intercept": 0.8,
        "x": 0.3}}``, ``z={"g": {"Intercept": array(J), "x": array(J)}}``. One factor also accepts the
        entry without the factor key; neither applies without groups. ``L`` is the Cholesky factor of the
        correlation matrix of each correlated term, ``L={"g": array(M, M)}`` (brms ``L_m``: lower triangular, unit-norm
        rows), required exactly for the correlated factors (a single one also accepts the bare matrix) and stored on
        Stan's ``cholesky_corr`` coordinates. ``sigma`` is the residual
        SD (Gaussian and Student-t; the SD of ``log y`` for the lognormal), ``shape`` the negative binomial / Gamma
        shape, ``phi`` the Beta precision; exactly the family's own is given. Scales are stored on the log scale because that is what PyMC samples.
        ``nu`` is the degrees of freedom of a Student-t model that learns them (``learned_parameters``), given exactly then
        and stored as ``log(nu - 1)``, PyMC's interval coordinate. ``zi`` / ``hu`` is the zero-inflation / hurdle probability
        of the zero-inflated / hurdle families (B06), given exactly for them, in (0, 1), and stored as ``logit(zi)``, PyMC's
        log-odds coordinate (Stan's for ``real<lower=0,upper=1>``). For the cumulative family (B05) ``intercept`` is the
        K - 1 thresholds (brms's ``Intercept``: of the centred design when centred), strictly increasing, stored on Stan's
        ordered coordinates ``(tau_1, log(tau_2 - tau_1), ...)``; for the categorical family it has one entry per
        non-reference category and ``b`` is ``(columns, K - 1)``, one column per non-reference category. An auxiliary
        parameter with its own formula (B01) is given as that formula's ``{"intercept": ..., "b": ...}`` instead: the
        intercept as sampled (centred when the formula is, brms's ``Intercept_sigma``) and the coefficients in its
        design-column order (brms's ``b_sigma``), each stored under its own variable (``Model.dpar_names``), unconstrained; so is
        a zero parameter or extra parameter with its own formula (E03), ``zi={"intercept": ..., "b": ...}``, ``alpha={...}``.
        A monotonic term ``mo(x)`` (B03) has no entry in ``b``: ``bsp`` is its coefficient (brms's ``bsp``) and ``simo`` its
        simplex (brms's ``simo``: ``D`` positive shares summing to 1, stored on Stan's stick-breaking coordinates), each keyed
        by the term's column, ``bsp={"x": 0.3}``, ``simo={"x": [0.2, 0.5, 0.3]}`` (a single monotonic term also accepts the
        bare value), required exactly when the model has one. A mixture (C05, docs/mixtures.md) takes ``intercept`` (the K
        components' centred intercepts, strictly increasing, on Stan's ordered coordinates), ``b`` of shape ``(columns, K)``,
        ``sigma`` (K residual SDs) and ``theta`` (the mixing proportions, a simplex of K, on Stan's stick-breaking
        coordinates); ``theta`` applies to a mixture only. A Gaussian process ``gp(x)`` (C03) takes ``sdgp`` and ``lscale``
        (positive scalars, stored as their logs) and ``zgp`` (its standardised values, one per distinct input of the exact GP
        or per basis function of the approximate one, in brms's order), each keyed by the process's column,
        ``sdgp={"x": 0.8}``, ``lscale={"x": 0.3}``, ``zgp={"x": array(N)}`` (a single Gaussian process also accepts bare values),
        required exactly when the model has one. ``ar`` and ``ma`` are an autocorrelation term's coefficients (C04, brms's ``ar``
        and ``ma``): ``p`` and ``q`` finite numbers (a scalar for one), stored as given (unconstrained), required
        exactly for the parts the term has. A measurement-error term ``me(x, sdx)`` and a missing-value term ``mi(col)`` (C06,
        docs/missing-and-measurement-error.md) have no entry in ``b`` either: ``bsp`` is the coefficient keyed by the term's
        first column (``bsp={"x": 0.8}``, or the bare value for the model's one such term); ``me()`` takes ``meanme`` and
        ``sdme`` (the location and the positive scale of the latent values, ``sdme`` stored on the log scale) and ``zme`` (the
        standardised latent values, one per fitted row), each keyed by the noisy column or bare; ``Ymi`` gives the missing
        values of each column this model holds (``missing_names``: the response's under ``| mi()`` and each ``mi()`` term's
        column), ``Ymi={"xm": array(Nmi)}`` in the order of the missing rows. ``alpha``, ``beta`` and ``ndt`` are the extra
        parameters of the skew-normal, ex-Gaussian and shifted lognormal families (D01, ``FamilySpec.extra_parameters``), given
        exactly for them: the skewness ``alpha`` (any finite number, stored as given), the exponential component's mean ``beta``
        (positive, stored as its log) and the non-decision time ``ndt`` (strictly between 0 and the smallest response, stored
        as ``log(ndt / (min(y) - ndt))``, PyMC's interval coordinate and Stan's). ``zoi`` and ``coi`` (E01, the zero-one-inflated
        Beta's probabilities, strictly between 0 and 1, stored as their logits, as ``zi`` is) and ``kappa`` (E01, the extended-support
        Beta's, positive, stored as its log) are given exactly for their families. Under a shrinkage prior on b (D07,
        docs/shrinkage-priors.md) ``b`` is a transformed parameter and is omitted: ``shrinkage`` gives brms's ``zb`` and the
        prior's scales instead (``shrinkage.value_entries``), ``{"zb": ..., "hs_global": ..., "hs_slab": ..., "hs_local": ...}``
        or ``{"zb": ..., "R2D2_R2": ..., "R2D2_phi": ...}``, each vector in design-column order.
        D04: the sratio, cratio and acat families'
        ``intercept`` is the K - 1 thresholds in any order (brms's unordered ``Intercept``), stored as given; ``bcs`` gives the
        coefficients of each category-specific term ``cs(x)``, K - 1 per term (brms's ``bcs`` row of the column), keyed by
        column (``bcs={"x": [0.2, -0.1, 0.4]}``, or the bare vector for the model's one such term), required exactly when the
        model has one. E02: ``kappa`` (the von Mises precision, positive, stored as its log), ``quantile`` (the asymmetric
        Laplace quantile, in (0, 1), stored as its logit; omitted when the model fixes it) and ``tmp_xi`` (brms's unscaled
        generalized extreme value shape, any finite number, stored as given; the graph derives ``xi`` from it) likewise.
        F02b: ``lagsar`` or ``errorsar``, a SAR term's parameter (brms's), strictly between the smallest and largest eigenvalue of
        its ``M``, stored on PyMC's interval coordinate ``log((rho - lower) / (upper - rho))`` (Stan's), given exactly with the term.
        """
        extras = {"alpha": alpha, "beta": beta, "ndt": ndt, "zoi": zoi, "coi": coi,   # D01, E01 (zero_one_inflated_beta)
                  "kappa": kappa, "quantile": quantile, "tmp_xi": tmp_xi}             # E01 (xbeta), E02
        if self.family_spec.mixture:                          # C05: the components' parameters and theta
            if any(v is not None for v in extras.values()):   # D01, E01, E02: a mixture's Gaussian components have none
                raise ValueError("alpha, beta, ndt, zoi, coi, kappa, quantile and tmp_xi are the skew_normal, exgaussian, "
                                 "shifted_lognormal, zero_one_inflated_beta, xbeta, von_mises, asym_laplace and gen_extreme_value "
                                 "families' parameters; a mixture has none")
            if ar is not None or ma is not None:              # C04: a mixture has no autocorrelation term
                raise ValueError("ar and ma are an autocorrelation term's coefficients; a mixture has none")
            if shrinkage is not None:                         # D07: a mixture takes no shrinkage prior
                raise ValueError("shrinkage holds the variables of a shrinkage prior on b; a mixture has none")
            if sdgp is not None or lscale is not None or zgp is not None:   # C03: a mixture has no Gaussian process
                raise ValueError("sdgp, lscale and zgp are a Gaussian process's values; a mixture has none")
            if lagsar is not None or errorsar is not None:    # F02b: a mixture has no SAR term
                raise ValueError("lagsar and errorsar are a SAR term's parameters; a mixture has none")
            from autobayes.families.mixture import value_point
            return value_point(self, intercept=intercept, b=b, sigma=sigma, theta=theta, sd=sd, z=z, L=L, shape=shape, phi=phi,
                               nu=nu, zi=zi, hu=hu, bsp=bsp, simo=simo,
                               meanme=meanme, sdme=sdme, zme=zme, Ymi=Ymi)   # C06: a mixture has no me() / mi() term, refused there
        if theta is not None:
            raise ValueError("theta (the mixing proportions) is a mixture's (ab.mixture); this model has none")
        from autobayes.shrinkage import value_entries
        shrunk = value_entries(self, shrinkage)               # D07: zb and the shrinkage prior's scales; {} without one
        if shrunk and b is not None:
            raise ValueError(f"b is a transformed parameter under the {self.priors.b.name} prior (b = zb * sdb): omit it and give "
                             "zb and the prior's scales in shrinkage=...")
        if not shrunk and b is None:
            raise ValueError(f"b is required: {len(self.spec.columns)} coefficients, one per design column {list(self.spec.columns)}")
        cs_point = self._category_specific_point(bcs)          # D04: each cs() term's coefficients, one per threshold
        b = np.zeros(len(self.spec.columns)) if shrunk else np.atleast_1d(np.asarray(b, dtype=float))
        if self.family_spec.per_category:                     # B05: one coefficient per design column and non-reference category
            others = len(self.categories) - 1
            if b.shape != (len(self.spec.columns), others):
                raise ValueError(f"b must be an array of shape ({len(self.spec.columns)}, {others}): one row per design column "
                                 f"{list(self.spec.columns)}, one column per non-reference category {list(self.categories[1:])}")
        elif b.shape != (len(self.spec.columns),):
            raise ValueError(f"b must have {len(self.spec.columns)} entries, one per design column {list(self.spec.columns)}")
        def keyed(value, label):
            if isinstance(value, Mapping):
                if set(value) != set(self.spec.groups):
                    hint = ""
                    if len(self.spec.group_terms) == 1 and set(value) <= set(self.spec.group_terms[0].coefs):
                        t = self.spec.group_terms[0]
                        hint = (f" ({list(value)} look like the coefficients of {t.factor!r}: nest them under the factor, "
                                f"{label}={{{t.factor!r}: {{{', '.join(f'{c!r}: ...' for c in t.coefs)}}}}})")
                    raise ValueError(f"{label} must cover exactly the grouping factors {list(self.spec.groups)}{hint}")
                return {g: value[g] for g in self.spec.groups}
            if len(self.spec.groups) != 1:
                raise ValueError(f"{label} requires an exact-key mapping for multiple grouping factors")
            return {self.spec.groups[0]: value}

        def by_coefficient(value, term, label):
            """One entry per coefficient of ``term``; a bare entry is the varying intercept's (intercept-only factors)."""
            if isinstance(value, Mapping):
                if set(value) != set(term.coefs):
                    raise ValueError(f"{label} for {term.factor!r} must cover exactly its coefficients {list(term.coefs)}; got {list(value)}")
                return {c: value[c] for c in term.coefs}
            if term.slopes:
                raise ValueError(
                    f"{label} for {term.factor!r} must be a mapping over its coefficients {list(term.coefs)} "
                    f"('{term.canonical}' has varying slopes)"
                )
            return {"Intercept": value}

        sd_by_coef, z_by_coef = {}, {}
        if not self.spec.has_group:
            if sd is not None or z is not None:
                raise ValueError("this model has no group-level term; sd and z do not apply to a population-only model")
        else:
            if sd is None or z is None:
                raise ValueError("sd and z are required for every group-level coefficient (varying intercepts and slopes)")
            sd_by_group, z_by_group = keyed(sd, "sd"), keyed(z, "z")
            for t in self.spec.group_terms:
                g = t.factor
                sds, zs = by_coefficient(sd_by_group[g], t, "sd"), by_coefficient(z_by_group[g], t, "z")
                for coef in t.coefs:
                    block = self.design.group_blocks[(g, coef)]
                    arr = np.asarray(zs[coef], dtype=float)
                    if arr.shape != (block.J,) or not np.isfinite(arr).all():
                        what = repr(g) if coef == "Intercept" else f"the slope {coef!r} of {g!r}"
                        raise ValueError(f"z for {what} must have {block.J} finite entries")
                    sd_by_coef[(g, coef)], z_by_coef[(g, coef)] = sds[coef], arr
        L_by_group = {}
        correlated = list(self.correlation_names)             # the correlated terms' factors, and a shared ID's (D08)
        if not correlated:
            if L is not None:
                raise ValueError("this model has no correlated group term '(1 + x | g)'; L does not apply")
        else:
            if L is None:
                raise ValueError(f"L (the Cholesky factor of the correlation matrix) is required for the correlated factors {correlated}")
            if isinstance(L, Mapping):
                if set(L) != set(correlated):
                    raise ValueError(f"L must cover exactly the correlated grouping factors {correlated}; got {list(L)}")
                by_group = {g: L[g] for g in correlated}
            elif len(correlated) == 1:
                by_group = {correlated[0]: L}
            else:
                raise ValueError(f"L requires an exact-key mapping for multiple correlated grouping factors {correlated}")
            for g in correlated:
                L_by_group[g] = check_correlation_cholesky(by_group[g], len(self.correlation_coefs(g)), f"L for {g!r}")
        scale = self.family_spec.scale_parameter                 # "sigma", "shape", "phi" or None
        auxiliary = {"sigma": sigma, "shape": shape, "phi": phi}
        if {k for k, v in auxiliary.items() if v is not None} != ({scale} if scale is not None else set()):
            raise ValueError(self.family_spec.scale_rule)
        zero = self.family_spec.zero_parameter                   # "zi", "hu" or None (B06)
        probabilities = {"zi": zi, "hu": hu}
        if {k for k, v in probabilities.items() if v is not None} != ({zero} if zero is not None else set()):
            if zero is None:
                raise ValueError(f"the {self.family} family has no zero-inflation or hurdle probability; omit zi and hu")
            raise ValueError(f"{zero} (the {AUXILIARY[zero]}) is required for the {self.family} family; "
                             f"{' and '.join(k for k in probabilities if k != zero)} does not apply")
        own = {e.variable: e for e in self.family_spec.extra_parameters}     # D01: alpha, beta or ndt, exactly the family's own
        for name, value in extras.items():                                 # (E02: kappa, quantile, tmp_xi)
            if name in own and value is None:
                raise ValueError(f"{name} (the {own[name].meaning}) is required for the {self.family} family")
            if name not in own and value is not None:
                if name in self.family_spec.fixed_parameters:              # E02: a quantile the model fixes
                    raise ValueError(f"{name} is fixed at {getattr(self, name):g} in this model, not sampled; omit it")
                what = AUXILIARY.get(name) or f"unscaled {AUXILIARY[name.removeprefix('tmp_')]} {name}"
                raise ValueError(f"the {self.family} family has no {what}; omit {name}"
                                 + "".join(f" (its {e.meaning} is {e.variable})" for e in own.values()))
        n = self.names
        if self.spec.intercept and intercept is None:
            raise ValueError("intercept is required for a model with a population intercept")
        if not self.spec.intercept and intercept is not None:
            raise ValueError("no population intercept exists; omit intercept (or pass None)")
        point = {n["intercept"]: np.asarray(float(intercept))} if self.spec.intercept and not self.categories else {}
        if self.spec.intercept and self.categories:          # B05: K - 1 thresholds, or one intercept per non-reference category
            arr = np.atleast_1d(np.asarray(intercept, dtype=float))
            K = len(self.categories)
            if self.family_spec.thresholds and self.family_spec.ordered_thresholds:
                if arr.shape != (K - 1,) or not np.isfinite(arr).all() or not (np.diff(arr) > 0).all():
                    raise ValueError(f"intercept must be the {K - 1} thresholds, finite and strictly increasing (brms's Intercept)")
                point[f"{n['intercept']}_ordered__"] = np.r_[arr[0], np.log(np.diff(arr))]     # Stan's ordered coordinates
            elif self.family_spec.thresholds:                  # D04: brms's unordered vector[nthres] Intercept, as it is
                if arr.shape != (K - 1,) or not np.isfinite(arr).all():
                    raise ValueError(f"intercept must be the {K - 1} thresholds, finite (brms's unordered Intercept of the "
                                     f"{self.family} family)")
                point[n["intercept"]] = arr
            else:
                if arr.shape != (K - 1,) or not np.isfinite(arr).all():
                    raise ValueError(f"intercept must have {K - 1} finite entries, one per non-reference category "
                                     f"{list(self.categories[1:])}")
                point[n["intercept"]] = arr
        def log_scale(value, label):
            arr = np.asarray(value, dtype=float)
            if arr.shape != () or not np.isfinite(arr) or arr <= 0:
                raise ValueError(f"{label} must be a finite positive scalar")
            return np.asarray(np.log(arr))

        for (g, coef), roles in self.group_coefficient_names.items():
            if "by_coordinate" in roles:                       # D08: gr(by = ): one SD per level of the by column
                block = self.design.group_blocks[(g, coef)]
                value = sd_by_coef[(g, coef)]
                if isinstance(value, Mapping):
                    if set(value) != set(block.by_levels):
                        raise ValueError(f"sd for {g!r} must cover exactly the levels {list(block.by_levels)} of its by column")
                    value = [value[b] for b in block.by_levels]
                arr = np.asarray(value, dtype=float)
                if arr.shape != (len(block.by_levels),) or not np.isfinite(arr).all() or (arr <= 0).any():
                    raise ValueError(f"sd for {g!r} must be {len(block.by_levels)} finite positive SDs, one per level "
                                     f"{list(block.by_levels)} of its by column (or a mapping over them)")
                point[f"{roles['sd']}_log__"] = np.log(arr)
            else:
                point[f"{roles['sd']}_log__"] = log_scale(sd_by_coef[(g, coef)], f"sd:{sd_tag(g, coef)}")
            point[roles["offset"]] = z_by_coef[(g, coef)]
        for g, roles in self.correlation_names.items():              # Stan's cholesky_corr coordinates of L (A02)
            point[f"{roles['L']}_cholesky_corr__"] = cholesky_corr_free(L_by_group[g])
        if scale in self.spec.dpar_names:                      # its own formula (B01): the intercept and coefficients as sampled
            point.update(self._dpar_point(scale, auxiliary[scale]))
        elif scale is not None:
            point[f"{n[scale]}_log__"] = log_scale(auxiliary[scale], scale)
        if zero is not None and zero in self.spec.dpar_names:  # E03: its own formula's intercept and coefficients as sampled
            point.update(self._dpar_point(zero, probabilities[zero]))
        elif zero is not None:                                 # B06: a probability, sampled as its log-odds
            p = np.asarray(probabilities[zero], dtype=float)
            if p.shape != () or not np.isfinite(p) or not 0 < p < 1:
                raise ValueError(f"{zero} must be a finite scalar strictly between 0 and 1")
            point[f"{n[zero]}_logodds__"] = np.asarray(np.log(p) - np.log1p(-p))
        for name, extra in own.items():                        # D01: on its coordinate (as is, log, Stan's lower-upper)
            if name in self.spec.dpar_names:                   # E03: alpha's own formula, as sampled
                point.update(self._dpar_point(name, extras[name]))
                continue
            point[extra.coordinate] = extra.value(extras[name], self.design.y, self.family)
        learned = self.learned_parameters
        if "nu" in learned:                                    # learned (A13): sampled as log(nu - lower), the interval coordinate
            if nu is None:
                raise ValueError("nu is required: this model learns nu (built with nu=None and Priors(nu=...))")
            arr = np.asarray(nu, dtype=float)
            if arr.shape != () or not np.isfinite(arr) or arr <= learned["nu"]:
                raise ValueError(f"nu must be a finite scalar > {learned['nu']:g}")
            point[f"{n['nu']}_interval__"] = np.asarray(np.log(arr - learned["nu"]))
        elif nu is not None:
            raise ValueError(f"nu is fixed at {self.nu:g} in this model, not sampled; omit it" if getattr(self, "nu", None) is not None
                             else f"the {self.family} family has no nu; omit it")
        i = 0
        for term in ([] if shrunk else self.spec.bambi_terms):   # b in design-column order: one entry per term column
            k = len(term.columns)
            point[term.name] = np.array(b[i:i + k]) if term.levels else np.asarray(b[i])
            i += k
        point.update(shrunk)                                  # D07: zb, the scales on their coordinates
        mo = bool(self.spec.mo_terms)                        # bsp is a monotonic term's (B03) or an me() / mi() term's (C06)
        point.update(self._monotonic_point(bsp if mo or not (self.spec.me_terms or self.spec.mi_terms) else None, simo))
        point.update(self._gp_point(sdgp, lscale, zgp))      # C03: each Gaussian process's sdgp, lscale and zgp
        point.update(self._autocor_point(ar, ma))            # C04: an autocorrelation term's ar and ma
        point.update(self._measurement_point(None if mo else bsp, meanme, sdme, zme, Ymi))   # C06: me() and mi()
        point.update(cs_point)                                # D04
        point.update(self._sar_point(lagsar, errorsar))       # F02b: a SAR term's lagsar or errorsar
        expected = {v.name for v in self.pymc_model.value_vars}
        if set(point) != expected:
            raise RuntimeError(f"value names {sorted(point)} do not match PyMC's {sorted(expected)}")
        return point

    def _sar_point(self, lagsar, errorsar) -> dict[str, np.ndarray]:
        """``value_point``'s entry of a SAR term (F02b): its parameter as given, strictly between the bounds, on PyMC's interval
        coordinate."""
        d, point = self.design.sar, {}
        for name, value in (("lagsar", lagsar), ("errorsar", errorsar)):
            if d is None or d.parameter != name:
                if value is not None:
                    raise ValueError(f"this model has no {name}" + (f" ('{d.term}' has {d.parameter})" if d is not None else
                                     " (no SAR term sar(M, type = ...))") + f"; {name} does not apply")
                continue
            if value is None:
                raise ValueError(f"{name} is required: the term '{d.term}' has it")
            rho = np.asarray(value, dtype=float)
            if rho.shape != () or not d.lower < float(rho) < d.upper:
                raise ValueError(f"{name} must be a scalar strictly between {d.lower!r} and {d.upper!r} (the smallest and largest "
                                 f"eigenvalue of {d.matrix}), got {value!r}")
            point[f"{name}_interval__"] = np.asarray(np.log(rho - d.lower) - np.log(d.upper - rho))
        return point

    def _category_specific_point(self, bcs) -> dict[str, np.ndarray]:
        """``value_point``'s entries of the category-specific terms (D04): each term's K - 1 coefficients as given, from ``bcs``
        keyed by column (a single term also takes the bare vector)."""
        terms = self.spec.cs_terms
        if not terms:
            if bcs is not None:
                raise ValueError("bcs gives the coefficients of category-specific terms cs(x); this model has none")
            return {}
        columns = [t.operands[0] for t in terms]
        if bcs is None:
            raise ValueError(f"bcs is required: the coefficients of {[t.name for t in terms]}, one per threshold (brms's bcs)")
        if isinstance(bcs, Mapping):
            if set(bcs) != set(columns):
                raise ValueError(f"bcs must cover exactly the category-specific columns {columns}; got {list(bcs)}")
            by_column = {c: bcs[c] for c in columns}
        elif len(columns) == 1:
            by_column = {columns[0]: bcs}
        else:
            raise ValueError(f"bcs requires a mapping by column for the category-specific columns {columns}")
        K1, out = len(self.categories) - 1, {}
        for t in terms:
            arr = np.asarray(by_column[t.operands[0]], dtype=float)
            if arr.shape != (K1,) or not np.isfinite(arr).all():
                raise ValueError(f"bcs for {t.operands[0]!r} must be {K1} finite numbers, one per threshold")
            out[t.name] = arr
        return out

    def _monotonic_point(self, bsp, simo) -> dict[str, np.ndarray]:
        """``value_point``'s entries of the monotonic terms (B03): each coefficient as given and each simplex on Stan's
        stick-breaking coordinates, from ``bsp`` and ``simo`` keyed by column (a single term also takes bare values)."""
        from autobayes.monotonic import check_simplex, stick_breaking_free

        roles_by = self.mo_names
        if not roles_by:
            if bsp is not None or simo is not None:
                raise ValueError("this model has no monotonic term mo(x); bsp and simo do not apply")
            return {}
        columns = list(roles_by)
        if bsp is None or simo is None:
            raise ValueError(f"bsp (the coefficient) and simo (the simplex) are required for the monotonic terms "
                             f"{[r['b'] for r in roles_by.values()]}, keyed by column {columns}")

        def by_column(value, label):
            if isinstance(value, Mapping):
                if set(value) != set(columns):
                    raise ValueError(f"{label} must cover exactly the columns of the monotonic terms {columns}; got {list(value)}")
                return {c: value[c] for c in columns}
            if len(columns) != 1:
                raise ValueError(f"{label} requires an exact-key mapping over the monotonic terms' columns {columns}")
            return {columns[0]: value}

        coefficients, simplexes, point = by_column(bsp, "bsp"), by_column(simo, "simo"), {}
        for c, roles in roles_by.items():
            m = self.spec.monotonic(c)
            value = np.asarray(coefficients[c], dtype=float)
            if value.shape != () or not np.isfinite(value):
                raise ValueError(f"bsp for {m.term!r} must be a finite scalar")
            point[roles["b"]] = value
            point[f"{roles['simplex']}_stickbreaking__"] = stick_breaking_free(check_simplex(simplexes[c], m.D, f"simo for {m.term!r}"))
        return point

    def _gp_point(self, sdgp, lscale, zgp) -> dict[str, np.ndarray]:
        """``value_point``'s entries of the Gaussian processes (C03): ``sdgp`` and ``lscale`` on their logs and ``zgp`` as given,
        each keyed by column (a single Gaussian process also takes bare values)."""
        roles_by = self.gp_names
        if not roles_by:
            if sdgp is not None or lscale is not None or zgp is not None:
                raise ValueError("this model has no Gaussian process gp(x); sdgp, lscale and zgp do not apply")
            return {}
        columns = list(roles_by)
        if sdgp is None or lscale is None or zgp is None:
            raise ValueError(f"sdgp, lscale and zgp are required for the Gaussian processes {[self.spec.gp(c).term for c in columns]}, "
                             f"keyed by column {columns}")

        def by_column(value, label):
            if isinstance(value, Mapping):
                if set(value) != set(columns):
                    raise ValueError(f"{label} must cover exactly the columns of the Gaussian processes {columns}; got {list(value)}")
                return {c: value[c] for c in columns}
            if len(columns) != 1:
                raise ValueError(f"{label} requires an exact-key mapping over the Gaussian processes' columns {columns}")
            return {columns[0]: value}

        sds, scales, zs, point = by_column(sdgp, "sdgp"), by_column(lscale, "lscale"), by_column(zgp, "zgp"), {}
        for c, roles in roles_by.items():
            g = self.spec.gp(c)
            for label, given in (("sdgp", sds[c]), ("lscale", scales[c])):
                value = np.asarray(given, dtype=float)
                if value.shape != () or not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{label} for {g.term!r} must be a finite positive scalar")
                point[f"{roles[label]}_log__"] = np.asarray(np.log(value))
            z = np.asarray(zs[c], dtype=float)
            if z.shape != (g.size,) or not np.isfinite(z).all():
                what = "distinct input" if not g.approximate else "basis function"
                raise ValueError(f"zgp for {g.term!r} must have {g.size} finite entries, one per {what}")
            point[roles["zgp"]] = z
        return point

    def _autocor_point(self, ar, ma) -> dict[str, np.ndarray]:
        """``value_point``'s entries of an autocorrelation term (C04): ``ar`` and ``ma`` as given, ``p`` and ``q`` finite
        entries, for exactly the parts the term has."""
        term, roles, point = self.spec.autocor, self.autocor_names, {}
        orders = dict(term.parts) if term is not None else {}
        for part, value in (("ar", ar), ("ma", ma)):
            if part not in orders:
                if value is not None:
                    raise ValueError(f"this model has no {part} coefficients" + (f" ('{term.term}')" if term is not None else
                                     " (no autocorrelation term ar(), ma() or arma())") + f"; {part} does not apply")
                continue
            if value is None:
                raise ValueError(f"{part} is required: the term '{term.term}' has {orders[part]} {part} coefficient(s)")
            arr = np.atleast_1d(np.asarray(value, dtype=float))
            if arr.shape != (orders[part],) or not np.isfinite(arr).all():
                raise ValueError(f"{part} must have {orders[part]} finite entries ('{term.term}'), got shape {arr.shape}")
            point[roles[part]] = arr
        return point

    def _measurement_point(self, bsp, meanme, sdme, zme, Ymi) -> dict[str, np.ndarray]:
        """``value_point``'s entries of the ``me()`` and ``mi()`` terms (C06): each coefficient (``bsp``) as given, keyed by the
        term's first column; an ``me()`` term's location (``meanme``) as given, its scale (``sdme``) on the log scale and its
        standardised latent values (``zme``, one per fitted row); the missing values (``Ymi``) of each column of
        ``missing_names``, one per missing row. A single term (or column) also takes bare values."""
        terms = {t.operands[0]: t for t in (*self.spec.me_terms, *self.spec.mi_terms)}

        def keyed(value, label, keys):
            if value is None:
                raise ValueError(f"{label} is required, keyed by {keys}")
            if isinstance(value, Mapping):
                if set(value) != set(keys):
                    raise ValueError(f"{label} must cover exactly {keys}; got {list(value)}")
                return {k: value[k] for k in keys}
            if len(keys) != 1:
                raise ValueError(f"{label} requires an exact-key mapping over {keys}")
            return {keys[0]: value}

        def scalar(value, label, positive=False):
            arr = np.asarray(value, dtype=float)
            if arr.shape != () or not np.isfinite(arr) or (positive and arr <= 0):
                raise ValueError(f"{label} must be a finite{' positive' if positive else ''} scalar")
            return arr

        point = {}
        if terms:                                            # (without one, a bsp was refused as a monotonic term's)
            for col, value in keyed(bsp, "bsp (the coefficient of each me() / mi() term)", list(terms)).items():
                point[terms[col].name] = scalar(value, f"bsp for {terms[col].name!r}")
        me = list(self.me_names)
        if me:
            means, sds, zs = keyed(meanme, "meanme", me), keyed(sdme, "sdme", me), keyed(zme, "zme", me)
            for x, roles in self.me_names.items():
                point[roles["mean"]] = scalar(means[x], f"meanme for {x!r}")
                point[f"{roles['sd']}_log__"] = np.asarray(np.log(scalar(sds[x], f"sdme for {x!r}", positive=True)))
                z = np.asarray(zs[x], dtype=float)
                if z.shape != (self.design.n,) or not np.isfinite(z).all():
                    raise ValueError(f"zme for {x!r} must have {self.design.n} finite entries, one per fitted row")
                point[roles["z"]] = z
        elif any(v is not None for v in (meanme, sdme, zme)):
            raise ValueError("this model has no measurement-error term me(x, sdx); meanme, sdme and zme do not apply")
        missing = list(self.missing_names)
        if missing:
            for col, value in keyed(Ymi, "Ymi (the missing values)", missing).items():
                block, arr = self.design.missing[col], np.asarray(value, dtype=float)
                if arr.shape != (block.index.size,) or not np.isfinite(arr).all():
                    raise ValueError(f"Ymi for {col!r} must have {block.index.size} finite entries, one per missing row "
                                     f"{block.index.tolist()}")
                point[self.missing_names[col]["values"]] = arr
        elif Ymi is not None:
            raise ValueError("this model holds no missing values; Ymi does not apply")
        return point

    def _dpar_point(self, name: str, value) -> dict[str, np.ndarray]:
        """``value_point``'s entries of distributional parameter ``name``'s formula (B01) from ``{"intercept": float,
        "b": array}``: the intercept (as sampled) and the coefficients term by term in design-column order."""
        d, roles = self.spec.dpar(name), self.dpar_names[name]
        keys = {"intercept", "b", "sd", "z"} if d.group_terms else {"intercept", "b"}   # D08: a shared ID's terms' sd and z
        if not isinstance(value, Mapping) or set(value) - keys:
            raise ValueError(f"{name} has its own formula '{d.formula}': give {name}={{'intercept': float, 'b': array"
                             + (", 'sd': {factor: float}, 'z': {factor: array}" if d.group_terms else "") + "} "
                             "(the formula's intercept as sampled and its coefficients in design-column order)")
        point = self._dpar_group_point(name, value.get("sd"), value.get("z")) if d.group_terms else {}
        if d.intercept:
            if value.get("intercept") is None:
                raise ValueError(f"{name}['intercept'] is required: '{d.formula}' has an intercept")
            a = np.asarray(value["intercept"], dtype=float)
            if a.shape != () or not np.isfinite(a):
                raise ValueError(f"{name}['intercept'] must be a finite scalar")
            point[roles["intercept"]] = a
        elif value.get("intercept") is not None:
            raise ValueError(f"'{d.formula}' has no intercept; omit {name}['intercept']")
        b = np.atleast_1d(np.asarray(value.get("b", ()), dtype=float))
        if b.shape != (len(d.columns),) or not np.isfinite(b).all():
            raise ValueError(f"{name}['b'] must have {len(d.columns)} finite entries, one per design column {list(d.columns)} "
                             f"of '{d.formula}'")
        i = 0
        for term in d.pop_terms:
            k = len(term.columns)
            point[roles[f"b:{term.name}"]] = np.array(b[i:i + k]) if term.levels else np.asarray(b[i])
            i += k
        return point

    def _dpar_group_point(self, name: str, sd, z) -> dict[str, np.ndarray]:
        """D08: ``value_point``'s entries of a distributional formula's group-level terms (a shared correlation ID's): each
        term's SD on its log and its standardised effects, from ``sd`` and ``z`` keyed by factor (a single term also takes the
        bare values)."""
        d = self.spec.dpar(name)
        factors = [t.factor for t in d.group_terms]

        def keyed(value, label):
            if value is None:
                raise ValueError(f"{name}[{label!r}] is required: '{d.formula}' has the group-level term(s) {factors}")
            if isinstance(value, Mapping):
                if set(value) != set(factors):
                    raise ValueError(f"{name}[{label!r}] must cover exactly the factors {factors}; got {list(value)}")
                return value
            if len(factors) != 1:
                raise ValueError(f"{name}[{label!r}] requires an exact-key mapping over the factors {factors}")
            return {factors[0]: value}

        sds, zs, point = keyed(sd, "sd"), keyed(z, "z"), {}
        for t in d.group_terms:
            roles, block = self._shared_roles(t.factor, name, "Intercept"), self.design.dpars[name].group_blocks[(t.factor, "Intercept")]
            value = np.asarray(sds[t.factor], dtype=float)
            if value.shape != () or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name}['sd'] for {t.factor!r} must be a finite positive scalar")
            arr = np.asarray(zs[t.factor], dtype=float)
            if arr.shape != (block.J,) or not np.isfinite(arr).all():
                raise ValueError(f"{name}['z'] for {t.factor!r} must have {block.J} finite entries")
            point[f"{roles['sd']}_log__"], point[roles["offset"]] = np.log(value), arr
        return point

    # -- description -------------------------------------------------------------------
    def _described_value_names(self) -> list[str]:
        """PyMC's value variables in the order ``describe()`` lists them: PyMC's order, except that the family's auxiliary
        parameters without a formula of their own take the places PyMC gives them in one fixed order: the scale parameter
        (``sigma_log__``, ...), then the zero parameter (``zi_logodds__``, ``hu_logodds__``), then the learned parameters
        (``nu_interval__``), so ``sigma_log__`` before ``nu_interval__``, the order of brms's ``real<lower=0> sigma`` and
        ``real<lower=1> nu``. Bambi creates these variables in the iteration order of a set of their names
        (``Family.auxiliary_parameters``), which depends on ``PYTHONHASHSEED``: with two of them (a Student-t model that
        learns nu) PyMC's order of the two varies from process to process, while the model, its density and its target
        identity do not."""
        names = [v.name for v in self.pymc_model.value_vars]
        fam, n = self.family_spec, self.names
        fixed = []
        if fam.scale_parameter is not None and fam.scale_parameter not in self.spec.dpar_names:
            fixed.append(f"{n[fam.scale_parameter]}_log__")
        if fam.zero_parameter is not None:
            fixed.append(f"{n[fam.zero_parameter]}_logodds__")
        fixed += [e.coordinate for e in fam.extra_parameters]   # D01: brms's sigma before alpha, beta or ndt
        fixed += [f"{n[name]}_interval__" for name in self.learned_parameters]
        present = [name for name in fixed if name in names]
        for i, name in zip([i for i, v in enumerate(names) if v in present], present):
            names[i] = name
        return names

    def describe(self) -> str:
        if self.family_spec.mixture:                          # C05: the components, theta and their priors
            from autobayes.families.mixture import describe
            return describe(self)
        s, d, pr = self.spec, self.design, self.priors
        from autobayes.shrinkage import brms_spelling          # D07: brms's name of a shrinkage prior on b
        def effect(g):
            return "u" if len(s.groups) == 1 else f"u_{g}"
        def slope_effect(g, c):
            return f"v_{c}|{g}"
        coef = pr.coefficient_priors(s)
        # the interaction terms by shape: numeric by categorical (the (x, z) pairs the wording below has always
        # described), numeric by numeric (x:w, one product column) and categorical by categorical (z:h, one cell each)
        num_cat = [t for t in s.pop_terms if t.kind == "interaction" and len(t.categorical) == 1]
        num_num = [t for t in s.pop_terms if t.kind == "interaction" and not t.categorical]
        cat_cat = [t for t in s.pop_terms if t.kind == "interaction" and len(t.categorical) == 2]
        pairs = [t.operands for t in num_cat]
        def cells(t):                                        # (cell label, level of the first operand, level of the second)
            return _cells(s, t)
        fam = self.family_spec
        centred_x = d.centered and not fam.thresholds        # B05: the cumulative mu is on the raw design
        xs = " + ".join(_linear_terms(s, centred_x))
        lp = fam.linpred_symbol
        a = f"a{'_c' if d.centered else ''}" if s.intercept else "0"
        grouping = ("; ".join(f"J = {block.J} levels of {g}" for g, block in d.group_designs.items())
                    if s.has_group else "no group-level term (population-level model)")
        data_line = f"  data      : n = {d.n} observations, {grouping}"
        if fam.has_trials:
            data_line = (f"  data      : {d.n} fitted rows, {int(d.trials.sum())} trials in total, "
                         f"{grouping}; {len(self.dropped_rows)} zero-trial row(s) dropped"
                         + (f" (positions {list(self.dropped_rows)[:10]}{'...' if len(self.dropped_rows) > 10 else ''})"
                            if self.dropped_rows else ""))
        features = ((["categorical predictor(s)"] if s.categoricals else []) + (["interaction(s)"] if s.interactions else [])
                    + (["in-formula transform(s)"] if s.transform_terms else []) + (["monotonic effect(s)"] if s.mo_terms else [])
                    + (["Gaussian process(es)"] if s.gp_terms else [])
                    + (["independent varying slope(s)"] if any(t.slopes and not t.has_correlation and t.id is None for t in s.group_terms) else [])
                    + (["correlated varying slope(s)"] if s.correlated_terms else []) + (["offset"] if s.offset is not None else [])
                    + (["derived grouping factor(s)"] if any(t.derived for t in s.group_terms) else [])
                    + (["multi-membership term(s)"] if any(t.is_mm for t in s.group_terms) else [])           # D08
                    + (["group SDs by level of a factor (gr(by = ))"] if any(t.by for t in s.group_terms) else [])
                    + (["correlation ID(s) shared with a distributional formula"] if s.shared_ids else [])
                    + ([f"response addition term(s) {', '.join(a.term for a in s.additions)}"] if s.additions else [])
                    + ([f"distributional formula(s) for {', '.join(s.dpar_names)}"] if s.dpars else [])
                    + ([f"autocorrelation term {s.autocor.term}"] if s.autocor is not None else [])
                    + ([f"simultaneous autoregressive term {s.sar.term}"] if s.sar is not None else [])          # F02b
                    + (["measurement-error term(s)"] if s.me_terms else []) + (["missing-value term(s)"] if s.mi_terms else [])
                    + ([f"a shrinkage prior on b ({brms_spelling(pr.b).split('(')[0]})"] if self.shrinkage_names else [])   # D07
                    + (["category-specific effect(s) cs()"] if s.cs_terms else []))                      # D04
        lines = [
            f"AutoBayes {fam.tag} model" + (" with " + " and ".join(features) if features else "")
            + (" (population-level only)" if not s.has_group else ""),
            f"  formula   : {s.formula}",
        ]
        if s.canonical_formula != s.formula:
            lines += [f"  expanded  : {s.canonical_formula}   (canonical form handed to Bambi)"]
        for ds in s.dpars:                                   # a distributional parameter's own formula (B01)
            lines += [f"  {ds.response:<10}: {ds.formula}   (distributional formula: {ds.response} has its own linear predictor, "
                      f"{d.dpars[ds.response].link} link)"]
            if ds.canonical_formula != ds.formula:
                lines += [f"  expanded  : {ds.canonical_formula}"]
        lines += [f"  family    : {self.family}, {self.link} link", data_line]
        if self.categories:                                   # B05: the response's categories, fitted as brms's 1-based Y
            hurdle = self.hurdle_category                     # E01: hurdle_cumulative's hurdle category, coded 0
            lines += [f"  {s.response:<10}: response categories " + (f"0={hurdle} (the hurdle), " if hurdle is not None else "")
                      + ", ".join(f"{k}={c}" for k, c in enumerate(self.categories, start=1))
                      + (" (ordered)" if fam.thresholds else f" (the reference is {self.categories[0]!r})")
                      + ("; fitted as brms's Y (0 the hurdle, k the k-th ordered category)" if hurdle is not None
                         else "; fitted as the 1-based category (brms's Y)")]
        if s.offset is not None:
            lines += [f"  offset    : {s.offset.expression}, added to the linear predictor as written (no coefficient, no prior, "
                      f"never centred; brms offset()); range {d.offset.min():.4f} to {d.offset.max():.4f} over the fitted rows"]
        lines += self._addition_lines()
        lines += self._autocor_data_lines()                  # C04: the series of an autocorrelation term
        if s.sar is not None:                                # F02b: the SAR term's weight matrix
            from autobayes.sar import data_lines
            lines += data_lines(self)
        for t in s.transform_terms:
            tr, col = t.transform, d.X_raw[:, s.columns.index(t.name)]
            lines += [f"  {t.name:<10}: in-formula transform of {tr.column!r}, {tr.expression(f'{tr.column}_i')}, range {col.min():.4f} to "
                      f"{col.max():.4f} over the fitted rows; recomputed from {tr.column!r} on new rows; centred like any column; "
                      f"brms names its coefficient {tr.brms_name}"]
        for t in s.mo_terms:                                 # B03: the coding of each monotonic predictor
            m = s.monotonic(t.operands[0])
            coding = (f"ordered categories {' < '.join(m.categories)} coded 0..{m.D}" if m.ordered else
                      f"whole numbers {m.minimum}..{m.maximum} coded as the value minus {m.minimum}, 0..{m.D}")
            lines += [f"  {t.name:<10}: monotonic effect of {m.column!r} (brms mo()): {coding} (brms's Xmo), D = {m.D} steps (brms's "
                      f"Jmo); recomputed from {m.column!r} on new rows, where a value outside the fitted coding is refused"]
        for g in s.gp_terms:                                 # C03: each Gaussian process's inputs and scaling
            kind = "approximate (Hilbert-space) Gaussian process" if g.approximate else "exact Gaussian process"
            basis = (f"; k = {g.k} basis functions on [-L, L], L = c = {g.L:g}, of the scaled input less the mean of its distinct values, "
                     f"{g.cmean:.6g}" if g.approximate else "")
            lines += [f"  {f'gp({g.column})':<10}: {kind} of {g.column!r} (brms {g.term}), squared-exponential kernel; {len(g.inputs)} "
                      f"distinct inputs from {min(g.inputs):g} to {max(g.inputs):g}, scaled by dmax = {g.dmax:g}, their largest "
                      f"distance, so that it is 1 (brms's scale = TRUE){basis}"]
        lines += self._measurement_data_lines()              # C06: me() and mi() terms, and the response's missing values
        for g, block in d.group_designs.items():
            lines += [f"  {'levels    ' if len(s.groups) == 1 else 'levels ' + g}: {', '.join(f'{j}={l}' for j, l in enumerate(block.levels))}"]
        for t in s.group_terms:
            block = d.group_blocks.get((t.factor, "Intercept"))
            if t.is_mm:                                      # D08: brms's mm(): memberships and weights
                w = ("the weight columns " + ", ".join(t.weights) + " divided by their sum on each row (brms's scale = TRUE)"
                     if t.weights else f"equal weights 1/{len(t.members)}")
                lines += [f"  {t.factor:<10}: multi-membership grouping {t.grouping} (brms's mm(); group name {t.factor}): each row belongs to its "
                          f"levels of {', '.join(repr(c) for c in t.members)} with {w}; the levels are the union of the members' labels "
                          f"(brms's J_m_k and W_m_k)"]
            if t.by is not None:                             # D08: brms's gr(by = ): the SD of each level's by level
                groups = "; ".join(f"{b}: " + ", ".join(l for j, l in enumerate(block.levels) if block.by_index[j] == k)
                                   for k, b in enumerate(block.by_levels))
                lines += [f"  {t.factor:<10}: gr({t.factor}, by = {t.by}): one SD per level of {t.by!r} ({groups}); each level of "
                          f"{t.factor!r} lies within one level of {t.by!r} (brms's Nby_m and Jby_m)"]
            if t.derived:
                ga, gb = t.operands
                name = _bambi_group_name(t.factor, "Intercept")
                lines += [f"  {t.factor:<10}: derived grouping factor, one level per observed ({ga}, {gb}) combination, labelled {ga}_{gb} as brms "
                          f"does (paste({ga}, {gb}, sep = \"_\")); unobserved combinations have no level; handed to Bambi as the column "
                          f"{t.column!r}, so its PyMC variables are {name}, {name}_sigma, {name}_offset"]
        for cat in s.all_categoricals:                       # a distributional formula's too (B01), b_sigma_z
            owners = [(b, f.categorical(cat.name)) for b, f in [("b", s), *((f"b_{ds.response}", ds) for ds in s.dpars)]
                      if cat.name in f.categorical_names]
            labels = self.data[cat.name].astype(str).to_numpy()
            props = ", ".join(f"{l} {np.mean(labels == l):.3f}" for l in cat.levels)
            lines += [
                f"  {cat.name:<10}: categorical, {'full-level coding' if cat.full_rank else 'treatment coding'}, reference {cat.reference!r}; levels (sorted) "
                f"{', '.join(cat.levels)}; coefficients " + "; ".join(f"{b}_{cat.name}[{', '.join(c.coefficients)}]" for b, c in owners)
                + (f"; unused declared categories {list(cat.unused)}" if cat.unused else ""),
                f"              proportions among fitted rows: {props}",
            ]
        rest = "".join(f" + gp({g.column})_i" for g in s.gp_terms) + (" + offset_i" if s.offset is not None else "") + "".join(
            ("".join(f" + W_{c},i * {effect(t.factor)}[{c}_i]" for c in t.members) if t.is_mm else   # D08: brms's W_m_k r_m_1[J_m_k]
             f" + {effect(t.factor)}[{t.factor}_i]") + "".join(f" + x_{c},i * {slope_effect(t.factor, c)}[{t.factor}_i]" for c in t.slopes)
            for t in s.group_terms)
        rest += " + arma_i" if s.autocor is not None else ""   # C04: the autocorrelation term, last
        if fam.thresholds:                                    # B05: no intercept in mu, the thresholds take its place
            body = (xs + rest) if xs else (rest[3:] or "0")
        else:
            body = a + (f" + {xs}" if xs else "") + rest
        lines += [
            "",
            "Linear predictor",
            f"  {lp}_i = {body}",
        ]
        if s.offset is not None:
            per_row = f"log({s.offset.column}_i)" if s.offset.log else f"{s.offset.column}_i"
            lines += [f"  offset_i = {per_row}: the offset term, a known quantity added to every row's {lp}_i; "
                      "it has no coefficient and no prior and is not centred."]
        lines += self._monotonic_lines(lp)
        lines += self._gp_lines(lp)
        lines += self._autocor_lines(lp)                     # C04
        lines += self._measurement_lines(lp)                 # C06
        scale = fam.scale_phrase
        same_group = " (same group)" if len(s.groups) == 1 else (" (same group effects)" if s.groups else "")
        for cat in s.categoricals:
            partners = [xn for xn, zn in pairs if zn == cat.name]
            changes = "".join(f" + {xn} * b_{xn}:{cat.name}[l]" for xn in partners)
            if cat.full_rank:
                lines += [f"  b_{cat.name}[l] is the level-specific {scale} baseline at numeric predictors zero, other categorical references, and group effect zero; the reference also has a coefficient."]
                if partners:
                    lines += [f"  The difference between level l and reference {cat.reference!r} on the {scale} scale{same_group}, holding other predictors fixed, is "
                              f"b_{cat.name}[l] - b_{cat.name}[{cat.reference}]" + changes + "."]
                continue
            if partners:
                zero = " and ".join(f"{xn} = 0" for xn in partners)
                where = f"at another value of {partners[0]}" if len(partners) == 1 else "at other values of those numeric predictors"
                lines += [f"  b_{cat.name}[l] is the difference between level l and the reference {cat.reference!r} on the "
                          f"{scale} scale at {zero}{same_group}; {where} the difference is "
                          f"b_{cat.name}[l]" + changes + ", holding other predictors fixed, because the levels have different slopes. "
                          "The reference has no coefficient."]
            else:
                lines += [f"  b_{cat.name}[l] is the difference between level l and the reference {cat.reference!r} "
                          f"on the {scale} scale at the same predictors{' and group' if s.has_group else ''}; the reference has no coefficient."]
        for xn in s.predictors:
            partners = [zn for x, zn in pairs if x == xn]
            if not partners:
                continue
            if len(partners) == 1:
                zn = partners[0]
                ref = s.categorical(zn).reference
                if s.intercept:
                    lines += [f"  {xn}:{zn}: the reference level {ref!r} has the raw intercept a and slope b_{xn}; level l has intercept "
                              f"difference b_{zn}[l] and slope difference b_{xn}:{zn}[l], so its slope in {xn} is b_{xn} + b_{xn}:{zn}[l]. "
                              "These baseline statements set other numeric predictors to zero, other categorical predictors to their references, and group effects to zero."]
                else:
                    lines += [f"  {xn}:{zn}: the reference level {ref!r} has slope b_{xn}; non-reference level l has slope difference b_{xn}:{zn}[l], so its slope in {xn} is b_{xn} + b_{xn}:{zn}[l]."]
            else:
                slope = f"b_{xn}" + "".join(f" + b_{xn}:{zn}[{zn}_i]" for zn in partners)
                lines += [f"  The slope in {xn} on the {scale} scale is {slope}, with each categorical reference contribution zero; other predictors and group effects are held fixed."]
                for zn in partners:
                    lines += [f"  b_{xn}:{zn}[l] is the change in that slope from reference {s.categorical(zn).reference!r} to level l of {zn}, holding the other categorical levels fixed."]
        for t in s.transform_terms:
            lines += [f"  {t.name}: b_{t.name} is the slope in {t.name}, not in {t.transform.column}; a unit change of {t.name} changes {lp} by "
                      f"b_{t.name}, other predictors and group effects fixed."]
        for t in num_num:
            xn, wn = t.operands
            lines += [f"  {t.name}: the slope in {xn} on the {scale} scale is b_{xn} + {wn} * b_{t.name} and the slope in {wn} is b_{wn} + {xn} * b_{t.name}; "
                      f"b_{xn} and b_{wn} are the slopes at {wn} = 0 and at {xn} = 0, and b_{t.name} is the change in either slope per unit of the other "
                      "predictor. Other predictors and group effects are held fixed."]
        for t in cat_cat:
            zn, hn = t.operands
            rz, rh = s.categorical(zn).reference, s.categorical(hn).reference
            lines += [f"  {t.name}: a row in level l of {zn} and level m of {hn} adds the cell's own b_{t.name}[l, m] to the main-effect "
                      f"coefficients of {zn} and {hn} on the {scale} scale; the cell coefficient is zero when either level is the reference "
                      f"({rz!r} for {zn}, {rh!r} for {hn}), so the main-effect coefficient of one factor is the level difference at the other "
                      "factor's reference, and the cell coefficient is how much that difference changes in cell (l, m). Cells with no "
                      "fitted rows keep only their prior."]
        lines += fam.mean_lines(self)
        if fam.intercept_lines is not None:                   # B05: the cumulative thresholds
            lines += fam.intercept_lines(self)
        elif d.centered:
            what = fam.intercept_meaning
            where = "at the predictor means" if not s.categoricals else \
                "at the numeric predictor means and at the sample proportions of the categorical levels"
            if s.interactions:
                expanded = ((["x"] if s.predictors else []) + (["1[z = l]"] if s.categoricals else []) + (["x * 1[z = l]"] if num_cat else [])
                            + (["x * w"] if num_num else []) + (["1[z = l] * 1[h = m]"] if cat_cat else []))
                where = f"at the fitted-row means of every expanded design column ({', '.join(expanded)})"
            lines += [
                "  Predictors are centred at their sample means (center_predictors=True, as in",
                f"  brms and Bambi)" + ("; a dummy column's mean is its level's proportion p_z[l]." if s.categoricals else "."),
            ]
            if num_cat:
                qs = []
                for xn, zn in pairs:
                    lab = self.original_labels[zn]; xv = self.data[xn].to_numpy(dtype=float)
                    for lvl in s.levels_of(f"{xn}:{zn}"):
                        m = lab == lvl
                        qs.append(f"q_{xn}:{zn}[{lvl}] = {xv[m].sum() / len(xv):.4f} = p_{zn}[{lvl}] {m.mean():.3f} * mean({xn} | {zn} = {lvl}) {xv[m].mean():.4f}"
                                  f" (not mean({xn}) {xv.mean():.4f} * p)")
                lines += ["  Each product column x * 1[z = l] is centred at its OWN mean q, which is the level's proportion"] + \
                         [f"  times the mean of x within that level: {q}" for q in qs] + \
                         ["  That point of the expanded design is not a realisable covariate row (a row has x*1[z=l] equal to x or 0)."]
            for t in num_num:
                xn, wn = t.operands
                xv, wv = self.data[xn].to_numpy(dtype=float), self.data[wn].to_numpy(dtype=float)
                lines += [f"  The product column {xn} * {wn} is centred at its OWN mean q_{t.name} = {np.mean(xv * wv):.4f}, "
                          f"not at mean({xn}) * mean({wn}) = {xv.mean() * wv.mean():.4f}; the difference is the covariance of {xn} and {wn}."]
            for t in cat_cat:
                zn, hn = t.operands
                lz_all, lh_all = self.original_labels[zn], self.original_labels[hn]
                ps = ", ".join(f"p_{t.name}[{lvl}] = {np.mean((lz_all == lz) & (lh_all == lh)):.3f}" for lvl, lz, lh in cells(t))
                lines += [f"  Each cell column 1[{zn} = l] * 1[{hn} = m] is centred at the cell's proportion of the fitted rows: {ps}."]
            correction = (" - sum_l p_z[l] * b_z[l]" if s.categoricals else "")
            correction += (" - sum_(x,z),l q_x:z[l] * b_x:z[l]" if len(num_cat) > 1 else (" - sum_l q_x:z[l] * b_x:z[l]" if num_cat else ""))
            correction += "".join(f" - q_{t.name} * b_{t.name}" for t in num_num)
            correction += "".join(f" - sum_(l,m) p_{t.name}[l, m] * b_{t.name}[l, m]" for t in cat_cat)
            lines += [
                f"  a_c is the {what} {where}{' (group effect zero)' if s.has_group else ''}; the",
                "  prior on the intercept applies to a_c. The posterior reports the raw-scale",
                "  intercept a = a_c - sum_p mean(x_p) * b_p" + correction + ", computed after sampling"
                + (": the linear predictor at x = 0 and at the reference level." if s.categoricals else "."),
            ]
            if s.categoricals:
                lines += list(fam.centred_intercept_note)
        elif not s.intercept:
            lines += ["  No population intercept or intercept prior. The raw design is used without back-transformation;",
                      f"  center_predictors requested={self.center_predictors}, effective=False (centering requires a free intercept)."]
            if len(s.categoricals) > 1:
                lines += ["  Only the first categorical main effect is full-level coded. Reordering factors changes independent prior meaning."]
        else:
            lines += ["  Predictors are not centred (center_predictors=False); a is the raw-scale intercept"
                      + ((" (x = 0, reference level" + (", group effect zero)." if s.has_group else ").")) if s.categoricals else ".")]
        for ds in s.dpars:                                   # each distributional parameter's own linear predictor (B01)
            lines += self._dpar_lines(ds)
        if s.has_group:
            lines += ["", "Group effects (non-centred)"]
            for t in s.group_terms:
                g, block = t.factor, d.group_designs[t.factor]
                shared = s.shared_id_of(g)
                if t.by is not None:                         # D08: the SD of the level's by level
                    lines += [f"  u_{g}[j] = sd_{g}[{t.by}[j]] * z_{g}[j], z_{g}[j] ~ Normal(0, 1), j = 0..{block.J - 1}: one SD per level "
                              f"of {t.by!r} ({', '.join(block.by_levels)}), the level's own (brms's sd_m[1, Jby_m[j]])"]
                else:
                    lines += [f"  u_{g}[j] = sd_{g} * z_{g}[j], z_{g}[j] ~ Normal(0, 1), j = 0..{block.J - 1}"]
                if t.is_mm:
                    lines += [f"  Row i adds sum_k W_k,i * u_{g}[level k of row i] over its members {', '.join(t.members)}, the weights summing to 1."]
                if shared is not None:                       # D08: one correlation matrix across the response and sigma formulas
                    effects = [f"u_{g}[j]"] + [f"{slope_effect(g, c)}[j]" for c in t.slopes] + [f"u_{dp},{g}[j]" for dp, _ in shared.coefs if dp]
                    sds = [f"sd_{g}"] + [f"sd_{c}|{g}" for c in t.slopes] + [f"sd_{dp},{g}" for dp, _ in shared.coefs if dp]
                    lines += [f"  ({', '.join(effects)}) = diag({', '.join(sds)}) L_{g} z[j], z[j] ~ Normal(0, I): the correlation ID {shared.id!r}",
                              f"  shares one correlation matrix Cor_{g} = L_{g} L_{g}^T over ({', '.join(shared.labels)}), brms's one group across",
                              f"  the response formula's '{t.canonical}' and the distributional formula's term(s) (brms's scale_r_cor;",
                              f"  PyMC {self._correlation_roles(g)['cor']}); each slope multiplies the raw predictor, never centred."]
                    continue
                if t.has_correlation:
                    zs = [f"z_{g}[j]"] + [f"z_{c}|{g}[j]" for c in t.slopes]
                    for i, c in enumerate(t.slopes, start=1):
                        terms = " + ".join(f"L_{g}[{i},{k}] * {zs[k]}" for k in range(i + 1))
                        lines += [f"  {slope_effect(g, c)}[j] = sd_{c}|{g} * ({terms}), z_{c}|{g}[j] ~ Normal(0, 1), j = 0..{block.J - 1}: "
                                  f"the slope of {c} varies by {g} ('|': correlated with u_{g})"]
                    vec = ", ".join([f"u_{g}[j]"] + [f"{slope_effect(g, c)}[j]" for c in t.slopes])
                    sds = ", ".join([f"sd_{g}"] + [f"sd_{c}|{g}" for c in t.slopes])
                    lines += [f"  ({vec}) = diag({sds}) L_{g} ({', '.join(zs)}): L_{g} is the Cholesky factor of the correlation matrix",
                              f"  Cor_{g} = L_{g} L_{g}^T of the coefficients ({', '.join(t.coefs)}) (brms's scale_r_cor; PyMC {self._correlation_roles(g)['cor']}),",
                              f"  so Cor_{g}[Intercept, {t.slopes[0]}] is the correlation between a level's intercept deviation and its slope deviation.",
                              f"  Each varying slope multiplies the raw predictor (never centred, as brms's Z does); the correlation is a statement",
                              f"  at x = 0: shifting a predictor by c turns u_{g} into u_{g} + c * v and changes it."]
                    continue
                for c in t.slopes:
                    lines += [f"  {slope_effect(g, c)}[j] = sd_{c}|{g} * z_{c}|{g}[j], z_{c}|{g}[j] ~ Normal(0, 1), j = 0..{block.J - 1}: "
                              f"the slope of {c} varies by {g} ('||': no correlation with u_{g})"]
                if t.slopes:
                    lines += [f"  Each varying slope multiplies the raw predictor (never centred, as brms's Z does) and is independent of u_{g}",
                              f"  by construction; that independence is a statement at x = 0: shifting a predictor by c turns u_{g} into",
                              f"  u_{g} + c * v, which is correlated with v."]
            lines += ["  Independent priors across factors; group effects are not mean-centered."]
        else:
            lines += ["", "Group effects", "  none: a population-only model (no '(1 | group)' term)"]
        lines += ["", "Priors" + ("" if fam.link_scale is None else f" (location coefficients on the {fam.link_scale} scale)")]
        each = "[k]" if fam.per_category else ""              # B05: the categorical family's priors hold for every category k
        if s.intercept and fam.thresholds:
            lines += [f"  Intercept[k] ~ {describe_prior(pr.intercept)}, for each of the {len(self.categories) - 1} thresholds "
                      "k (brms's class Intercept)"]
        elif s.intercept:
            lines += [f"  {a}{each} ~ {describe_prior(pr.intercept)}"]
        if self.shrinkage_names:                             # D07: one joint prior on b, b = zb * sdb
            from autobayes.shrinkage import describe_lines
            lines += describe_lines(self)
        else:
            lines += [f"  b_{col}{each} ~ {describe_prior(coef[col])}" for col in s.columns]
        for t in s.cs_terms:                                 # D04: one coefficient per threshold (brms's bcs, class b)
            lines += [f"  bcs_{t.operands[0]}[k] ~ {describe_prior(coef[t.name])}, for each of the {len(self.categories) - 1} thresholds k "
                      f"(brms's bcs, class b, coefficient {t.operands[0]})"]
        if s.mo_terms:                                       # a monotonic term's coefficient (brms bsp, class b) and simplex (B03)
            from autobayes.default_priors import brms_rename
            from autobayes.priors import dirichlet_concentration

            simplexes = pr.simo_priors(s)
            for t in s.mo_terms:
                m, brms = s.monotonic(t.operands[0]), brms_rename(t.name)
                con = ", ".join(f"{v:g}" for v in dirichlet_concentration(simplexes[m.column], m.D))
                lines += [f"  b_{t.name} ~ {describe_prior(coef[t.name])}   (brms's bsp_{brms}: class b, coefficient {brms})",
                          f"  {self.mo_names[m.column]['simplex']} ~ {describe_prior(simplexes[m.column])}   (brms class simo, "
                          f"coefficient {brms}1; concentrations {con})"]
        if s.gp_terms:                                       # each Gaussian process's marginal SD and length-scale (C03)
            from autobayes.default_priors import brms_rename

            sdgp, lscale = pr.sdgp_priors(s), pr.lscale_priors(s)
            for g in s.gp_terms:
                roles, brms = self.gp_names[g.column], brms_rename(f"gp({g.column})")
                lines += [f"  {roles['sdgp']} ~ {describe_prior(sdgp[g.column], lb=0)}   (brms class sdgp, coefficient {brms})",
                          f"  {roles['lscale']} ~ {describe_prior(lscale[g.column], lb=0)}   (brms class lscale, coefficient {brms}; "
                          "on the scaled input)",
                          f"  {roles['zgp']}[j] ~ Normal(0, 1)   (brms's zgp)"]
        lines += self._autocor_prior_lines()                 # C04: ar and ma
        if s.me_terms or s.mi_terms:                         # C06: an me() / mi() term's coefficient (brms bsp, class b)
            from autobayes.default_priors import brms_rename

            means, sds = pr.me_priors(s, "meanme"), pr.me_priors(s, "sdme")
            for t in (*s.me_terms, *s.mi_terms):
                lines += [f"  b_{t.name} ~ {describe_prior(coef[t.name])}   (brms's bsp_{brms_rename(t.name)}: class b, coefficient "
                          f"{brms_rename(t.name)})"]
            for x, roles in self.me_names.items():
                lines += [f"  {roles['mean']} ~ {describe_prior(means[x])}   (brms class meanme, coefficient me{x})",
                          f"  {roles['sd']} ~ {describe_prior(sds[x], lb=0)}   (brms class sdme, coefficient me{x}"
                          + ("; Flat() is flat on the positive half line" if sds[x].name == "Flat" else "") + ")",
                          f"  {roles['z']}_i ~ Normal(0, 1) for each row (brms's zme)"]
            for col, roles in self.missing_names.items():
                lines += [f"  {roles['values']}: flat, no prior of its own (brms's Ymi); informed by the formula of {col} and by every "
                          "formula that reads it"]
        if fam.per_category:
            lines += [f"  (one prior per class, the same for every non-reference category k = {', '.join(self.categories[1:])})"]
        for (g, c), prior in pr.sd_priors(s).items():
            lines += [f"  sd_{sd_tag(g, c)} ~ {describe_prior(prior, lb=0)}"]
        for g, prior in pr.cor_priors(s).items():
            lines += [f"  Cor_{g} ~ {prior} (brms lkj({prior.params['eta']:g}) on the correlation matrix, placed on L_{g} as lkj_corr_cholesky)"
                      + ("   (brms default)" if prior.source == "default" else "")]
        for ds in s.dpars:                                   # a distributional formula's intercept and coefficients (B01)
            name, sub, dd = ds.response, pr.dpar_priors(s, ds.response), d.dpars[ds.response]
            if ds.intercept:
                lines += [f"  a_{name}{'_c' if dd.centered else ''} ~ {describe_prior(sub.intercept)}   (on the {dd.link} scale of {name})"]
            lines += [f"  b_{name}_{col} ~ {describe_prior(prior)}" for col, prior in sub.coefficient_priors(ds).items()]
            lines += [f"  sd_{name},{g} ~ {describe_prior(prior, lb=0)}   (brms class sd with dpar = {name})"   # D08: a shared ID's terms
                      for (g, c), prior in (sub.sd_priors(ds).items() if ds.group_terms else ())]
        if s.sar is None:
            lines += fam.likelihood_lines(self)
        else:                                                # F02b: the joint SAR density in place of the family's
            from autobayes.sar import likelihood_lines
            lines += likelihood_lines(self)
        lines += self._addition_likelihood_lines()
        if self.default_prior_labels:                         # priors filled in by ab.default_priors() (A07)
            from autobayes.default_priors import BRMS_VERSION
            lines += ["", "Default priors (ab.default_priors(); docs/default-priors.md)",
                      f"  {', '.join(self.default_prior_labels)}: brms {BRMS_VERSION}'s default_prior() for this formula, data and family,",
                      "  marked (brms default) above with brms's spelling. The Intercept's student_t(3, location, scale) has the median of",
                      "  the response on the link scale rounded to one decimal as its location (minus the mean offset; 0 under a logit or",
                      "  probit link) and max(2.5, its MAD rounded to one decimal) as its scale, which sd and sigma share; a flat prior is",
                      "  improper: it adds nothing to the log density, and prior_predictive() is refused."]
            if any(label.startswith(("sdgp:", "lscale:")) for label in self.default_prior_labels):
                lines += ["  A Gaussian process's sdgp shares that scale too; its lscale is the inverse gamma with 1% of its mass below the",
                          "  smallest distance between distinct scaled inputs (at least 0.01) and 1% above the largest (1), its parameters",
                          "  rounded to 6 decimals, or brms's half-normal normal(0, 0.5) when the process has two distinct inputs."]
        for w in self.warnings:
            lines += ["", "WARNING: " + w]
        lines += ["", "PyMC variables sampled (value space)"]
        n = self.names
        roles = {n["intercept"]: "a_c" if d.centered else "a"} if s.intercept else {}
        if s.intercept and fam.thresholds and fam.ordered_thresholds:   # B05: Stan's ordered coordinates of the thresholds
            roles = {f"{n['intercept']}_ordered__": f"Intercept[1..{len(self.categories) - 1}], the thresholds"
                     + (" of the centred design" if d.centered else "") + ", on Stan's ordered coordinates "
                     "(Intercept[1], log(Intercept[k] - Intercept[k - 1]))"}
        elif s.intercept and fam.thresholds:                  # D04: brms's unordered thresholds, sampled as they are
            roles = {n["intercept"]: f"Intercept[1..{len(self.categories) - 1}], the thresholds"
                     + (" of the centred design" if d.centered else "") + f" (vector; dims {n['threshold_dim']}, unordered, as brms's)"}
        for t in s.cs_terms:                                  # D04: each cs() term's coefficients, one per threshold
            roles[t.name] = (f"bcs_{t.operands[0]}[1..{len(self.categories) - 1}], the coefficients of {t.name}, one per threshold "
                             f"(brms's bcs; vector, dims {n['threshold_dim']})")
        if fam.per_category:                                  # B05: one entry per non-reference category
            others = ", ".join(self.categories[1:])
            roles = {n["intercept"]: f"{'a_c' if d.centered else 'a'}[{others}] (vector; dims {n['category_dim']})"} if s.intercept else {}
            roles.update({t.name: f"b_{t.name}[{others}] (vector; dims {n['category_dim']})" for t in s.pop_terms if not t.is_vector})
        for (g, c), names in self.group_coefficient_names.items():
            roles[f"{names['sd']}_log__"] = f"log(sd_{sd_tag(g, c)})"
            roles[names["offset"]] = ("z" if len(s.groups) == 1 else f"z_{g}") if c == "Intercept" else f"z_{c}|{g}"
        for (g, c), names in self.group_coefficient_names.items():   # D08: gr(by = )'s SD, a vector over the by levels
            if "by_coordinate" in names:
                roles[f"{names['sd']}_log__"] = f"log(sd_{g}[{', '.join(self.design.group_blocks[(g, c)].by_levels)}]) (vector; dims {names['by_coordinate']})"
        for (name, g, c), names in self.dpar_group_names.items():   # D08: a shared ID's distributional coefficient
            roles[f"{names['sd']}_log__"] = f"log(sd_{name},{g})"
            roles[names["offset"]] = f"z_{name},{g}"
        for g, names in self.correlation_names.items():
            roles[f"{names['L']}_cholesky_corr__"] = f"L_{g} on Stan's cholesky_corr coordinates (canonical partial correlations, arctanh scale)"
        if fam.scale_parameter is not None and fam.scale_parameter not in s.dpar_names:
            roles[f"{n[fam.scale_parameter]}_log__"] = f"log({fam.scale_parameter})"
        for ds in s.dpars:                                   # a distributional formula's variables (B01); scalar ones read b_sigma_x
            dn = self.dpar_names[ds.response]
            if ds.intercept:
                roles[dn["intercept"]] = f"a_{ds.response}{'_c' if d.dpars[ds.response].centered else ''} (the intercept of {ds.response}'s formula)"
            for t in ds.pop_terms:
                if t.is_vector:
                    var = dn[f"b:{t.name}"]
                    shown = ", ".join(f"({l})" for l in t.levels) if len(t.categorical) == 2 else ", ".join(t.levels)
                    roles[var] = f"b_{var}[{shown}] (vector; dims {var}_dim, posterior rows {', '.join(f'{var}[{l}]' for l in t.levels)})"
        if fam.zero_parameter is not None:
            roles[f"{n[fam.zero_parameter]}_logodds__"] = f"logit({fam.zero_parameter})"
        for extra in fam.extra_parameters:                   # D01: alpha, log(beta), ndt on Stan's lower-upper coordinate
            roles[extra.coordinate] = extra.coordinate_meaning
        for name, lower in self.learned_parameters.items():
            roles[f"{n[name]}_interval__"] = f"log({name} - {lower:g})"
        for t in s.mo_terms:                                 # B03: the coefficient and the simplex of each monotonic term
            mo, D = self.mo_names[t.operands[0]], s.monotonic(t.operands[0]).D
            roles[mo["b"]] = f"b_{t.name}, the monotonic term's coefficient (brms's bsp)"
            roles[f"{mo['simplex']}_stickbreaking__"] = (f"{mo['simplex']}[1..{D}] on Stan's stick-breaking simplex coordinates ({D - 1} "
                                                         f"entries; brms's simo); dims {mo['coordinate']}")
        for g in s.gp_terms:                                 # C03: each Gaussian process's SD, length-scale and standardised values
            gn = self.gp_names[g.column]
            roles[f"{gn['sdgp']}_log__"] = f"log({gn['sdgp']}), the marginal SD of gp({g.column}) (brms's sdgp)"
            roles[f"{gn['lscale']}_log__"] = f"log({gn['lscale']}), the length-scale of gp({g.column}) on the scaled input (brms's lscale)"
            roles[gn["zgp"]] = (f"{gn['zgp']}[1..{g.size}], the standardised values of gp({g.column}), one per "
                                f"{'basis function' if g.approximate else 'distinct input'} (brms's zgp); dims {gn['coordinate']}")
        for t in (*s.me_terms, *s.mi_terms):                  # C06: the coefficient of each me() / mi() term, me()'s latent values
            roles[t.name] = f"b_{t.name}, the {'measurement-error' if t.kind == 'measurement_error' else 'missing-value'} term's coefficient (brms's bsp)"
        for x, me in self.me_names.items():
            roles[me["mean"]] = f"meanme_{x}, the location of the latent values of {x} (brms's meanme)"
            roles[f"{me['sd']}_log__"] = f"log(sdme_{x}), the scale of the latent values of {x} (brms's sdme)"
            roles[me["z"]] = f"zme_{x}[0..{d.n - 1}], the standardised latent values of {x} (brms's zme); dims __obs__"
        for col, mi in self.missing_names.items():
            rows = d.missing[col].index.tolist()
            roles[mi["values"]] = (f"Ymi_{col}[{', '.join(map(str, rows[:6]))}{', ...' if len(rows) > 6 else ''}], the missing values of {col} "
                                   f"(brms's Ymi; flat); dims {mi['coordinate']}, coordinates the missing rows (0-based)")
        for t in s.pop_terms:
            if t.is_vector:
                shown = ", ".join(f"({l})" for l in t.levels) if len(t.categorical) == 2 else ", ".join(t.levels)   # cells read (l, m)
                rows = ", ".join(f"{t.name}[{l}]" for l in t.levels)      # ArviZ names a row variable[coordinate]
                roles[t.name] = f"b_{t.name}[{shown}] (vector; dims {t.name}_dim, posterior rows {rows})"
                if fam.per_category:                          # B05: and one column per non-reference category
                    roles[t.name] = f"b_{t.name}[{shown}] (matrix; dims {t.name}_dim x {n['category_dim']})"
        for part, order in (s.autocor.parts if s.autocor is not None else ()):   # C04: ar and ma, unconstrained vectors
            acn = self.autocor_names
            roles[acn[part]] = (f"{part}[1..{order}], the {_AUTOCOR_MEANING[part]} coefficients (brms's {part}; vector, dims "
                                f"{acn[part + '_coordinate']}, unconstrained)")
        if s.sar is not None:                                # F02b: lagsar / errorsar on its interval coordinate
            sd_ = d.sar
            roles[f"{sd_.parameter}_interval__"] = (f"log(({sd_.parameter} - lower) / (upper - {sd_.parameter})), lower and upper the "
                                                    f"smallest and largest eigenvalue of {sd_.matrix} (Stan's lower-upper coordinate)")
        shrinkage_note = ""
        if self.shrinkage_names:                             # D07: zb and the shrinkage prior's scales; b, sdb deterministic
            from autobayes.shrinkage import deterministic_note, value_roles
            roles.update(value_roles(self))
            shrinkage_note = deterministic_note(self)
        for name in self._described_value_names():
            lines.append(f"  {name:<24} {roles.get(name, f'b_{name}')}")
        lines.append("Deterministic: " + "".join(f"{names['group_effect']} = {effect(g) if c == 'Intercept' else slope_effect(g, c)}, "
                                                 for (g, c), names in self.group_coefficient_names.items())
                     + "".join(f"{names['group_effect']} = u_{name},{g}, " for (name, g, _), names in self.dpar_group_names.items())   # D08
                     + "".join(f"{names['cor']} = Cor_{g} = L_{g} L_{g}^T, " for g, names in self.correlation_names.items()) + fam.deterministic_note
                     + "".join(f"; {dn['value']} = {_DPAR_INVERSE[self.design.dpars[name].link].format(name)}, one {name} per row (B01)"
                               for name, dn in self.dpar_names.items())
                     + "".join(f"; {me['latent']} = meanme_{x} + sdme_{x} * zme_{x}, one per row (brms's Xme)" for x, me in self.me_names.items())
                     + shrinkage_note
                     + (f"; {n['linpred']} = mu_i[k], one per row and threshold (D04)" if s.cs_terms else ""))
        return "\n".join(lines)

    def _measurement_data_lines(self) -> list[str]:
        """``describe()``'s data lines of the ``me()`` and ``mi()`` terms and of the response's own missing values (C06)."""
        s, d, out = self.spec, self.design, []
        for x, block in d.measurement_error.items():
            out += [f"  {block.term:<10}: the predictor {x!r} measured with known noise (brms me()): noisy values {block.Xn.min():.4f} to "
                    f"{block.Xn.max():.4f}, known standard deviations {block.noise_column!r} {block.noise.min():.4f} to "
                    f"{block.noise.max():.4f} over the {d.n} fitted rows (brms's Xn and noise)"]
        for col, block in d.missing.items():
            rows = block.index.tolist()
            where = (f"{len(rows)} missing row(s) (0-based {rows[:10]}{'...' if len(rows) > 10 else ''}; brms's Jmi is 1-based)"
                     if rows else "no missing row")
            if col == s.response:
                out += [f"  {col} | mi() : the response's {where}: their values are the parameters Ymi_{col} (brms's Ymi), and the "
                        f"likelihood is evaluated at the observed and imputed values together (brms's Yl_{col})"]
            else:
                out += [f"  mi({col})   : the column {col!r}, {where}: there its values are the parameters Ymi_{col}, which the formula "
                        f"of {col} imputes in the same multivariate model; elsewhere its observed values"]
        return out

    def _measurement_lines(self, lp: str) -> list[str]:
        """``describe()``'s paragraph of the ``me()`` and ``mi()`` terms (C06): the latent values and the measurement model, what
        the coefficient means, and that neither term is centred."""
        out = []
        for t in self.spec.me_terms:
            x, sdx = t.operands
            me = self.me_names[x]
            out += [f"  Xme_{x},i = meanme_{x} + sdme_{x} * zme_{x},i, zme_{x},i ~ Normal(0, 1): the latent (true) value of {x} in row i "
                    f"(brms's Xme), and {x}_i ~ Normal(Xme_{x},i, {sdx}_i), the noisy value observed with the known standard deviation "
                    f"{sdx}_i (PyMC's observed {me['noisy']}).",
                    f"  b_{t.name} (brms's bsp) is the change of {lp} per unit of the latent {x}, not of the noisy one; {t.name} is not "
                    f"centred (brms does not centre it either): the intercept refers to rows whose latent {x} is 0."]
        for t in self.spec.mi_terms:
            col = t.operands[0]
            out += [f"  {col}_i is the observed value of {col}, or at a missing row the parameter Ymi_{col} (brms's Yl_{col}): b_{t.name} "
                    f"(brms's bsp) is the change of {lp} per unit of {col}; {t.name} is not centred (brms does not centre it either)."]
        return out

    def _monotonic_lines(self, lp: str) -> list[str]:
        """``describe()``'s paragraph of the monotonic terms (B03): ``mo()`` written out, what the coefficient and the simplex
        mean, and where the term is zero."""
        out = []
        for t in self.spec.mo_terms:
            m, roles = self.spec.monotonic(t.operands[0]), self.mo_names[t.operands[0]]
            D, c, simplex, steps = m.D, m.column, roles["simplex"], m.step_labels()
            shown = ", ".join(steps) if D <= 6 else f"{', '.join(steps[:2])}, ..., {steps[-1]}"
            code = f"the position of {c}_i among the ordered categories" if m.ordered else f"{c}_i - {m.minimum}"
            out += [f"  {t.name}_i = {D} * ({simplex}[1] + ... + {simplex}[Xmo_i]) with Xmo_i = {code} in 0..{D} ({t.name}_i = 0 "
                    f"at {m.lowest}, {D} at {m.highest}): brms's mo(simo, Xmo).",
                    f"  {simplex} (brms's simo) is a simplex of {D} shares summing to 1, one per step {shown}: b_{t.name} (brms's bsp) "
                    f"is the average change of {lp} per step, {D} * b_{t.name} the change from {m.lowest} to {m.highest}, and {lp} is "
                    f"monotonic in {c} whatever the sign of b_{t.name}.",
                    f"  {t.name} is not centred (brms does not centre it either): the intercept refers to rows at {c} = {m.lowest}."]
        return out

    def _gp_lines(self, lp: str) -> list[str]:
        """``describe()``'s paragraph of the Gaussian processes (C03): each process written out (the exact GP's covariance and
        Cholesky factor, or the approximate GP's basis and spectral density), what ``sdgp`` and ``lscale`` mean, and what new
        rows take."""
        out = []
        for g in self.spec.gp_terms:
            c, roles = g.column, self.gp_names[g.column]
            sd, ls, z = roles["sdgp"], roles["lscale"], roles["zgp"]
            if g.approximate:
                out += [f"  gp({c})_i = sum_j phi_j(u_i) sqrt(S(w_j)) {z}[j], j = 1..{g.k}, with u_i = {c}_i / {g.dmax:g} - {g.cmean:.6g} the "
                        f"scaled, centred input, phi_j(u) = sin(pi j (u + L) / (2 L)) / sqrt(L) the Laplacian's eigenfunctions on [-L, L], "
                        f"L = {g.L:g}, w_j = pi j / (2 L) (brms's slambda), S(w) = {sd}^2 sqrt(2 pi) {ls} exp(-{ls}^2 w^2 / 2) the "
                        f"kernel's spectral density and {z}[j] ~ Normal(0, 1): the Hilbert-space approximation of the Gaussian process "
                        "(Solin and Sarkka 2020; Riutort-Mayol et al. 2023), brms's gp(x, k, c)."]
                lo, hi = (g.cmean - g.L) * g.dmax, (g.cmean + g.L) * g.dmax
                new = (f"  New rows: the basis at their inputs, which must lie inside the boundary, {c} in ({lo:.6g}, {hi:.6g}); "
                       "beyond it the approximation does not hold and the input is refused (divergence D66).")
            else:
                out += [f"  gp({c})_i = f(u_i), u_i = {c}_i / {g.dmax:g} the scaled input: at the {len(g.inputs)} distinct scaled inputs, in "
                        f"the order they first appear (brms's order), f = chol(K + 1e-12 I) {z}, K[j, m] = {sd}^2 exp(-(u_j - u_m)^2 / "
                        f"(2 {ls}^2)), {z}[j] ~ Normal(0, 1): brms's exact Gaussian process, non-centred, with 1e-12 added to the "
                        "diagonal of its covariance."]
                new = (f"  New rows take the process's value at a fitted input of {c}; any other input is refused (brms draws one from "
                       "the conditional normal, divergence D65; the approximate GP gp(x, k = ...) predicts anywhere inside its boundary).")
            out += [f"  {sd} is the process's marginal SD (the SD of gp({c})_i, on the scale of {lp}); {ls} its length-scale on the "
                    f"scaled input: inputs {ls} * {g.dmax:g} apart in {c} are correlated exp(-1/2) = 0.61. gp({c}) is not centred: the "
                    "intercept is not its mean over the rows.", new]
        return out

    def _autocor_data_lines(self) -> list[str]:
        """``describe()``'s data line of an autocorrelation term (C04): its series and their order."""
        term, d = self.spec.autocor, self.design.autocor
        if term is None:
            return []
        lengths = d.lengths
        series = f"{len(d.series)} series (the levels of {term.gr})" if term.gr is not None else "one series (no gr)"
        ordered = f"ordered by {term.time}" if term.time is not None else "in the data order (no time)"
        return [f"  autocor   : {term.term}: {series} of {lengths.min()} to {lengths.max()} rows, each {ordered}; brms's Stan data "
                "list the rows sorted by series and then time, predictions keep the data order"]

    def _autocor_lines(self, lp: str) -> list[str]:
        """``describe()``'s paragraph of an autocorrelation term (C04): brms's ARMA term in its regression formulation, the
        residual it reads and its recursion."""
        term = self.spec.autocor
        if term is None:
            return []
        lagged = [f"ar[{k}] * E_i[{k}]" for k in range(1, term.p + 1)] + [f"ma[{k}] * E_i[{k}]" for k in range(1, term.q + 1)]
        ma = " + ".join(f"ma[{k}] * E_i[{k}]" for k in range(1, term.q + 1))
        residual = (f"  e_i = y_i - eta_i, row i's residual, with eta_i = {lp}_i - arma_i the linear predictor without the term"
                    if not term.q else
                    f"  e_i = y_i - eta_i - ({ma}): row i's residual after the MA part (brms's err), with eta_i = {lp}_i - arma_i "
                    "the linear predictor without the term; it is a recursion along each series")
        return [f"  arma_i = {' + '.join(lagged)}: brms's ARMA term {term.term} (the regression formulation, cov = FALSE),",
                "  added after every other term. E_i[k] is e of the row k places before row i in its own series, 0 when the series",
                "  has no such row (brms's Err and J_lag): lags count rows, not units of time, so a gap in time is one step, as in brms.",
                residual,
                f"  {' and '.join(part for part, _ in term.parts)} {'are' if len(term.parts) > 1 else 'is'} unbounded (brms's vector[K] "
                f"{term.parts[0][0]})" + (": |ar| < 1 is not imposed, so the AR part need not be stationary." if term.p else ".")]

    def _autocor_prior_lines(self) -> list[str]:
        """``describe()``'s prior lines of an autocorrelation term's ``ar`` and ``ma`` (C04)."""
        term = self.spec.autocor
        if term is None:
            return []
        priors = self.priors.autocor_priors(self.spec)
        return [f"  {part}[k] ~ {describe_prior(priors[part])}, k = 1..{order}   (brms class {part}: each {_AUTOCOR_MEANING[part]} "
                "coefficient, unbounded)" for part, order in term.parts]

    def _dpar_lines(self, ds: ModelSpec) -> list[str]:
        """``describe()``'s paragraph of distributional parameter ``ds.response``'s own formula (B01): its linear
        predictor written out (coefficients ``b_<parameter>_<term>``), its link, and its centring."""
        name, dd = ds.response, self.design.dpars[ds.response]
        a = f"a_{name}{'_c' if dd.centered else ''}" if ds.intercept else "0"
        xs = " + ".join(_linear_terms(ds, dd.centered, b=f"b_{name}"))
        groups = "".join(f" + u_{name},{t.factor}[{t.factor}_i]" for t in ds.group_terms)   # D08: a shared ID's varying intercept
        # how the link reads (B01's log; E03: zi and hu under the logit, alpha under the identity): the left-hand side, row i's
        # value from its linear predictor, and the scale the coefficients act on
        left, value, on = {"log": (f"log({name}_i)", f"exp(eta_{name},i) > 0", f"log {name}"),
                           "logit": (f"logit({name}_i)", f"inv_logit(eta_{name},i), in (0, 1),", f"logit {name}"),
                           "identity": (f"{name}_i", f"eta_{name},i", f"{name} itself")}[dd.link]
        lines = ["", f"Distributional parameter {name} (its own formula '{ds.formula}', {dd.link} link; brms's bf(..., {ds.formula}))",
                 f"  {left} = eta_{name},i = {a}" + (f" + {xs}" if xs else "") + groups,
                 f"  {name}_i = {value} is row i's {name} in the likelihood below; the coefficients act on {on}."]
        if dd.centered:
            lines += [f"  Its columns are centred at their own fitted-row means (brms's Xc_{name}), so a_{name}_c is {on} there and",
                      f"  its prior applies to a_{name}_c; the posterior reports the raw-scale intercept",
                      f"  a_{name} = a_{name}_c - sum_p mean(x_p) * b_{name}_p (brms's b_{name}_Intercept), computed after sampling."]
        elif not ds.intercept:
            lines += [f"  No intercept: {on} is the raw design times its coefficients."]
        for t in ds.group_terms:                             # D08: correlated with the response formula's term through its ID
            roles = self.dpar_group_names[(name, t.factor, "Intercept")]
            lines += [f"  u_{name},{t.factor}[j] (PyMC {roles['group_effect']}) is the varying intercept of {on} by {t.factor}, '{t.canonical}': "
                      f"it shares the correlation ID {t.id!r} with the response formula's term (Group effects, below)."]
        return lines

    def _addition_lines(self) -> list[str]:
        """``describe()``'s data lines for the response addition terms (B02, docs/addition-terms.md)."""
        d, out = self.design, []
        for a in self.spec.additions:
            if a.kind == "mi":                               # C06: the response's missing rows, _measurement_data_lines
                continue
            if a.kind == "weights":
                out += [f"  weights   : {a.column}, each fitted row's log likelihood multiplied by its weight (brms weights()); weights "
                        f"{d.weights.min():.4f} to {d.weights.max():.4f}, sum {d.weights.sum():.4f} over {d.n} rows; the priors are not "
                        "weighted, and predict() ignores the weights (brms's posterior_epred and posterior_predict do too)"]
            elif a.kind == "cens" and a.y2 is not None:      # E04: brms's interval censoring cens(c, y2)
                left, right, inside = int((d.cens == -1).sum()), int((d.cens == 1).sum()), int((d.cens == 2).sum())
                out += [f"  censoring : {a.term}: {left} left-censored row(s) (-1: y_i is an upper bound, log P(Y_i <= y_i)), {right} "
                        f"right-censored (1: a lower bound, log P(Y_i > y_i)), {inside} interval-censored (2: y_i < Y_i <= {a.y2}_i, "
                        f"log P(y_i < Y_i <= {a.y2}_i)), {d.n - left - right - inside} observed (0); brms cens(); predict() ignores the "
                        "censoring, as brms does"]
            elif a.kind == "cens":
                left, right = int((d.cens == -1).sum()), int((d.cens == 1).sum())
                out += [f"  censoring : {a.column}: {left} left-censored row(s) (-1: y_i is an upper bound, log P(Y_i <= y_i)), {right} "
                        f"right-censored (1: a lower bound, log P(Y_i > y_i)), {d.n - left - right} observed (0); brms cens(); predict() "
                        "ignores the censoring, as brms does"]
            elif self.family_spec.discrete:                  # E04: a count response's bounds are integers and inclusive
                lb, ub = ("0" if a.lb is None else f"{a.lb:g}"), ("inf" if a.ub is None else f"{a.ub:g}")
                out += [f"  truncation: {a.term}: the counts lie in {{{lb}, ..., {ub}}}, both bounds included; each row's mass is "
                        "divided by the probability of that range (brms trunc()); predict(kind='mean') is the truncated mean and "
                        "kind='response' draws from the truncated distribution, as brms's posterior_predict does"]
            else:
                lb, ub = ("-inf" if a.lb is None else f"{a.lb:g}"), ("inf" if a.ub is None else f"{a.ub:g}")
                out += [f"  truncation: {a.term}: the response lies in [{lb}, {ub}]; each row's density is divided by the probability of "
                        "that interval (brms trunc()); predict(kind='mean') is the truncated mean and kind='response' draws from the "
                        "truncated distribution, as brms's posterior_epred and posterior_predict do"]
        return out

    def _addition_likelihood_lines(self) -> list[str]:
        """The likelihood paragraph's lines for the response addition terms (B02)."""
        out = []
        for a in self.spec.additions:
            if a.kind == "mi":                               # C06: the likelihood at the observed and imputed values
                out += [f"  mi(): the likelihood is evaluated at {self.spec.response}'s observed values and, at its missing rows, the parameters "
                        f"{self.missing_names[self.spec.response]['values'] if self.spec.response in self.missing_names else '(none missing)'} "
                        "(brms's Yl)"]
                continue
            if a.kind == "weights":
                out += [f"  weights(): the log likelihood is sum_i w_i log p(y_i | ...), w_i the {a.column} of row i (brms's weights[n] *)"]
            elif a.kind == "cens":
                out += ["  cens(): a left-censored row contributes log P(Y_i <= y_i), a right-censored row log P(Y_i > y_i), any other row its "
                        "log density (Stan's _lcdf / _lccdf / _lpdf as brms writes them)"]
                if a.y2 is not None:                         # E04
                    out += [f"  cens(..., {a.y2}): an interval-censored row contributes log P(y_i < Y_i <= {a.y2}_i), the log of the CDF "
                            "difference (brms's log_diff_exp of two _lcdf), formed from the tail the interval lies in"]
            elif self.family_spec.discrete:                  # E04
                out += [f"  trunc(): every row's log mass minus log P(lb <= Y_i <= ub), the log of the sum of the masses {a.term} allows"]
            else:
                out += [f"  trunc(): every row's log density minus log P(lb < Y_i <= ub), the log mass of {a.term}"]
        return out

    # -- priors ------------------------------------------------------------------------
    @property
    def default_prior_labels(self) -> list[str]:
        """The parameters whose prior came from ``ab.default_priors`` (brms's defaults; A07), labelled as the prior
        messages label them (``intercept``, ``b:x``, ``sd:g``, ``sd:x|g``, ``cor:g``, ``simo:x``, ``sigma``, ``nu``), in that order; empty
        when the user gave every prior (``zi`` or ``hu`` after the auxiliary parameter, B06; a distributional formula's
        ``sigma:intercept``, ``sigma:b:z[q]`` in place of ``sigma``, B01, and E03's ``zi:intercept`` in place of ``zi``, ``hu`` or
        ``alpha``; a mixture's ``mu1:intercept``, ``mu1:b:x``, ``sigma1``,
        ..., ``theta``, C05). ``fit_record`` stores them under ``"default_priors"``."""
        if self.family_spec.mixture:
            from autobayes.families.mixture import default_prior_labels
            return default_prior_labels(self)
        s, pr, fam = self.spec, self.priors, self.family_spec
        found = ([("intercept", pr.intercept)] if s.intercept else []) + [(f"b:{k}", v) for k, v in pr.coefficient_priors(s).items()]
        found += [(f"sd:{sd_tag(g, c)}", v) for (g, c), v in pr.sd_priors(s).items()]
        found += [(f"cor:{g}", v) for g, v in pr.cor_priors(s).items()]
        found += [(f"simo:{c}", v) for c, v in pr.simo_priors(s).items()]          # a monotonic term's simplex (B03)
        found += [(f"sdgp:{c}", v) for c, v in pr.sdgp_priors(s).items()]          # a Gaussian process's SD and length-scale (C03)
        found += [(f"lscale:{c}", v) for c, v in pr.lscale_priors(s).items()]
        found += list(pr.autocor_priors(s).items())                                  # an ARMA term's ar and ma (C04)
        found += [(f"{cls}:{x}", v) for cls in ("meanme", "sdme") for x, v in pr.me_priors(s, cls).items()]   # me() (C06)
        def formula_priors(name):                            # a parameter's own formula's priors (B01; E03: zi, hu, alpha)
            ds, sub = s.dpar(name), pr.dpar_priors(s, name)
            return (([(f"{ds.response}:intercept", sub.intercept)] if ds.intercept else [])
                    + [(f"{ds.response}:b:{k}", v) for k, v in sub.coefficient_priors(ds).items()]
                    + [(f"{ds.response}:sd:{sd_tag(g, c)}", v) for (g, c), v in (sub.sd_priors(ds).items() if ds.group_terms else ())])   # D08
        if fam.scale_parameter in s.dpar_names:              # its own formula's priors (B01)
            found += formula_priors(fam.scale_parameter)
        elif fam.scale_parameter is not None:
            found.append((fam.scale_parameter, getattr(pr, fam.scale_parameter)))
        if fam.zero_parameter is not None:
            found += (formula_priors(fam.zero_parameter) if fam.zero_parameter in s.dpar_names
                      else [(fam.zero_parameter, getattr(pr, fam.zero_parameter))])
        for e in fam.extra_parameters:                       # alpha, beta, ndt (D01)
            found += formula_priors(e.name) if e.name in s.dpar_names else [(e.name, getattr(pr, e.name))]
        if pr.nu is not None:
            found.append(("nu", pr.nu))
        return [label for label, prior in found if getattr(prior, "source", "user") == "default"]

    def prior_table(self) -> list[dict[str, str]]:
        """This model's priors in brms's table form (``Priors.table``): the rows brms's ``default_prior()`` prints for the
        same model, with this model's priors in brms's syntax."""
        return self.priors.table(self.spec, self.family_spec, center_predictors=self.effective_center_predictors,
                                 categories=self.categories or None)

    # -- convenience -------------------------------------------------------------------
    def prior_predictive(self, *, draws=500, seed=None):
        """Forward prior draws on modeled rows, in native graph parameterization."""
        from autobayes.prior_predictive import prior_predictive
        return prior_predictive(self, draws=draws, seed=seed)

    def fit(self, **kwargs):
        from autobayes.inference import fit

        return fit(self, **kwargs)

    def predict(self, fit_result, **kwargs):
        from autobayes.predict import predict

        return predict(self, fit_result, **kwargs)


def gaussian(formula: str | list[str], data: pd.DataFrame, priors: Priors, *, center_predictors: bool = True,
             categorical: Mapping[str, object] | None = None, rescor: bool | None = None, nl: bool = False,
             loop: bool = True, data2: Mapping[str, object] | None = None) -> "Model | MultivariateModel":
    """Build the Gaussian model (identity link, residual SD ``sigma``), with a varying intercept when the
    formula has ``(1 | group)`` and population-only otherwise (M3c-1).
    ``categorical={column: reference_level}`` declares treatment-coded categorical predictors. ``sigma`` may have its own
    formula (B01, docs/distributional.md): ``["y ~ x + (1 | g)", "sigma ~ z"]`` with
    ``Priors(..., sigma={"intercept": ab.Normal(0, 1), "b": ab.Normal(0, 0.5)})``, a log link, brms's ``bf(y ~ ..., sigma ~ z)``.
    A multivariate formula (C01, docs/multivariate.md), brms's ``mvbind(y, y2) ~ x + (1 | g)``, builds one model per response
    and returns a ``MultivariateModel``: ``rescor=True`` correlates the responses' residuals (``Priors(resp={"y": {...},
    "y2": {...}}, rescor=ab.LKJ(eta))``, brms's ``set_rescor(TRUE)``), ``rescor=False`` keeps their likelihoods independent.
    ``nl=True`` (C02, docs/nonlinear.md) builds a non-linear model, ``ab.NonlinearModel``: ``["y ~ a * exp(b * x)",
    "a ~ 1 + (1 | g)", "b ~ 1"]`` with ``Priors(nlpars={"a": {"b": ..., "sd": ...}, "b": {"b": ...}}, sigma=...)``, brms's
    ``bf(y ~ a * exp(b * x), a ~ 1 + (1 | g), b ~ 1, nl = TRUE)``; ``loop`` is brms's option of that name (only its default,
    True, is offered: divergence D63), and a non-linear model takes no multivariate formula or ``rescor`` (divergence D62).
    ``data2`` (F02b, docs/spatial-sar.md) holds the matrices the formula reads by name, brms's ``data2``: ``y ~ x + sar(M, type =
    "lag")`` with ``data2={"M": matrix}``, the weight matrix over the rows in data order, and ``Priors(..., lagsar=ab.Flat())``;
    a non-linear or multivariate model takes none (divergence D143)."""
    from autobayes.multivariate import MultivariateModel, is_multivariate_formula     # C01

    if data2 is not None and (nl or rescor is not None or is_multivariate_formula(formula)):   # F02b
        raise FormulaError("data2 is read by a SAR term sar(M, type = ...), which a non-linear or multivariate model does not take "
                           "in this slice (docs/spatial-sar.md, divergence D143)")
    if nl:                                                             # C02: a non-linear model
        from autobayes.nonlinear import NonlinearModel

        if rescor is not None:
            raise FormulaError("rescor= belongs to a multivariate model, 'mvbind(y, y2) ~ ...'; a non-linear model in a multivariate "
                               "formula is not supported (docs/nonlinear.md, divergence D62)")
        return NonlinearModel(formula, data, priors, family="gaussian", center_predictors=center_predictors,
                              categorical=categorical, loop=loop)
    if loop is not True:
        raise ValueError("loop is brms's option of a non-linear formula; it applies with nl=True only (docs/nonlinear.md)")
    if rescor is not None or is_multivariate_formula(formula):
        return MultivariateModel(formula, data, priors, family="gaussian", rescor=rescor, center_predictors=center_predictors,
                                 categorical=categorical)
    return Model(formula, data, priors, family="gaussian", center_predictors=center_predictors, categorical=categorical, data2=data2)


def bernoulli(formula: str, data: pd.DataFrame, priors: Priors, *, link: str = "logit", center_predictors: bool = True,
              categorical: Mapping[str, object] | None = None) -> Model:
    """Build the Bernoulli model (response strictly 0/1), with or without a varying intercept.

    ``link`` is ``"logit"`` (``p = expit(eta)``, the default), ``"probit"`` (``p = Phi(eta)``) or ``"cloglog"``
    (``p = 1 - exp(-exp(eta))``), brms's ``bernoulli(link = ...)``; the intercept and slope priors are on that link's
    scale. brms's ``cauchit`` link is not offered (docs/links.md).
    """
    return Model(formula, data, priors, family="bernoulli", link=link, center_predictors=center_predictors, categorical=categorical)


def binomial(formula: str, data: pd.DataFrame, priors: Priors, *, link: str = "logit", center_predictors: bool = True,
             categorical: Mapping[str, object] | None = None) -> Model:
    """Build the binomial model ``successes | trials(n) ~ x [+ (1 | group)]``, with or without a varying intercept;
    ``link`` as for ``bernoulli`` (``"logit"``, ``"probit"`` or ``"cloglog"``)."""
    return Model(formula, data, priors, family="binomial", link=link, center_predictors=center_predictors, categorical=categorical)


def student(formula: str | list[str], data: pd.DataFrame, priors: Priors, *, nu: float | None = None,
            center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Student-t response with an explicit sigma scale prior and degrees of freedom ``nu``, fixed or learned.

    ``nu=4`` (any finite ``nu > 1``) fixes them. ``nu=None``, the default, learns them under ``Priors(nu=...)``: a
    ``Gamma``, ``Exponential``, ``HalfNormal`` or ``HalfCauchy`` prior truncated to ``nu > 1`` and renormalised there, as
    brms's ``real<lower=1> nu`` (brms's default prior is ``gamma(2, 0.1)``, ``ab.Gamma(2, 0.1)``, which is not filled in
    silently; ``ab.default_priors(..., family="student")`` returns it with brms's other defaults; A13,
    docs/student-learned-nu.md). The expected response is mu. Sigma is not the residual standard deviation. ``sigma`` may
    have its own formula under a log link (B01, ``["y ~ x", "sigma ~ z"]``, docs/distributional.md); ``nu`` takes none.
    """
    return Model(formula, data, priors, family="student", nu=nu,
                 center_predictors=center_predictors, categorical=categorical)


def poisson(formula: str, data: pd.DataFrame, priors: Priors, *, link: str = "log",
            center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Poisson count response with log link and explicit coefficient/group priors.

    Retains zero rows. Input counts must be integers in [0, 1e6]. An exposure enters as
    ``offset(log(exposure))`` (docs/offset.md). No sigma, nu or overdispersion parameter
    is supported in this slice. ``link="sqrt"`` is brms's ``poisson("sqrt")``, ``mu = eta^2`` (E06, docs/links.md).
    """
    return Model(formula, data, priors, family="poisson", link=link,
                 center_predictors=center_predictors, categorical=categorical)


def negbinomial(formula: str | list[str], data: pd.DataFrame, priors: Priors, *, link: str = "log",
                center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Negative binomial count response (brms ``negbinomial``): log link, ``Var[y] = mu + mu^2 / shape``.

    Counts as for Poisson (integers in [0, 1e6], zero rows retained). The overdispersion parameter ``shape``
    takes its own positive-support prior, ``Priors(shape=ab.Gamma(0.01, 0.01))``, or its own formula under a log link (B01,
    ``["y ~ x", "shape ~ z"]``, docs/distributional.md); no sigma, nu, offset/exposure or zero-inflation.
    ``link="identity"`` is brms's ``negbinomial("identity")``, ``mu = eta``, which needs a positive linear predictor (E06,
    docs/links.md).
    """
    return Model(formula, data, priors, family="negbinomial", link=link,
                 center_predictors=center_predictors, categorical=categorical)


def gamma(formula: str | list[str], data: pd.DataFrame, priors: Priors, *, link: str = "log",
          center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Gamma response with a log link (brms ``Gamma(link = "log")``): ``E[y] = mu = exp(eta)``, ``Var[y] = mu^2 / shape``.

    Responses are strictly positive finite reals. The shape parameter takes its own positive-support prior,
    ``Priors(shape=ab.Gamma(0.01, 0.01))``, or its own formula under a log link (B01, ``["y ~ x", "shape ~ z"]``,
    docs/distributional.md); no sigma or nu. ``link`` is ``"log"`` by default (brms's default Gamma link is inverse;
    docs/brms-parity/divergences.md, D9); ``"inverse"`` (``mu = 1 / eta``) and ``"identity"`` (``mu = eta``) are brms's
    ``Gamma("inverse")`` and ``Gamma("identity")``, which need a positive linear predictor (E06, docs/links.md).
    """
    return Model(formula, data, priors, family="gamma", link=link,
                 center_predictors=center_predictors, categorical=categorical)


def lognormal(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
              center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Lognormal response (brms ``lognormal()``): ``log(y) ~ Normal(mu, sigma)`` with the identity link on the log-scale
    location ``mu``; ``E[y] = exp(mu + sigma^2 / 2)``.

    Responses are strictly positive finite reals. ``sigma``, the SD of ``log y``, takes its own positive-support prior,
    ``Priors(sigma=ab.HalfNormal(1))``; the intercept and slope priors act on ``log y``. ``predict(kind="mean")`` returns
    brms's ``posterior_epred`` ``exp(mu + sigma^2 / 2)`` as ``mean``; ``kind="linear"`` returns ``mu``. ``sigma`` may have its
    own formula under a log link (B01, ``["y ~ x", "sigma ~ z"]``, docs/distributional.md), and the mean then uses each
    row's ``sigma``. No shape, nu or offset.
    """
    return Model(formula, data, priors, family="lognormal",
                 center_predictors=center_predictors, categorical=categorical)


def beta(formula: str | list[str], data: pd.DataFrame, priors: Priors, *, link: str = "logit",
         center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Beta response with a logit link (brms ``Beta()``): proportions strictly inside (0, 1), ``E[y] = mu = expit(eta)``,
    ``Var[y] = mu (1 - mu) / (1 + phi)``.

    The precision ``phi`` takes its own positive-support prior, ``Priors(phi=ab.Gamma(0.01, 0.01))`` (brms class
    ``phi``), or its own formula under a log link (B01, ``["y ~ x", "phi ~ z"]``, docs/distributional.md); the intercept and
    slope priors are on the logit scale. ``predict(kind="mean")`` returns ``mu``; no sigma, shape, nu or zero/one inflation.
    ``link="probit"`` (``mu = Phi(eta)``) and ``link="cloglog"`` (``mu = 1 - exp(-exp(eta))``) are brms's ``Beta("probit")`` and
    ``Beta("cloglog")``, the priors then on that link's scale (E06, docs/links.md).
    """
    return Model(formula, data, priors, family="beta", link=link,
                 center_predictors=center_predictors, categorical=categorical)


def weibull(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
            center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Weibull response (brms ``weibull()``, D01, docs/continuous-families.md): log link on the mean, ``E[y] = mu = exp(eta)``,
    ``y ~ Weibull(shape, scale = mu / Gamma(1 + 1 / shape))``.

    Responses are strictly positive finite reals. The shape takes its own positive-support prior,
    ``Priors(shape=ab.Gamma(0.01, 0.01))`` (brms class ``shape``; brms's default, on request through ``ab.default_priors``).
    ``predict(kind="mean")`` returns ``mu``. E03 (docs/contexts-newer-families.md): ``shape`` may have its own formula under a
    log link, ``["ypos ~ x", "shape ~ z"]`` with ``Priors(shape={"intercept": ..., "b": ...})``, and ``weights()`` is taken.
    E04 (docs/cens-trunc-families.md): ``cens(c)`` and ``cens(c, y2)`` (survival data); no ``trunc()`` (D122).
    """
    return Model(formula, data, priors, family="weibull", center_predictors=center_predictors, categorical=categorical)


def exponential(formula: str, data: pd.DataFrame, priors: Priors, *,
                center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Exponential response (brms ``exponential()``, D01, docs/continuous-families.md): log link on the mean,
    ``E[y] = mu = exp(eta)``, ``y ~ Exponential(rate = 1 / mu)``; no auxiliary parameter.

    Responses are strictly positive finite reals. ``predict(kind="mean")`` returns ``mu``. No ``weights()``, ``cens()`` or
    ``trunc()`` (D79).
    """
    return Model(formula, data, priors, family="exponential", center_predictors=center_predictors, categorical=categorical)


def skew_normal(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
                center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Skew-normal response (brms ``skew_normal()``, D01, docs/continuous-families.md): identity link on the mean ``mu``,
    standard deviation ``sigma`` and skewness ``alpha`` (Azzalini's skew normal in brms's mean parameterisation).

    ``sigma`` takes a positive-support prior, ``Priors(sigma=ab.HalfNormal(1))``, and ``alpha`` a location prior,
    ``Priors(alpha=ab.Normal(0, 4))`` (brms class ``alpha``; its default, on request through ``ab.default_priors``).
    ``predict(kind="mean")`` returns ``mu``. E03 (docs/contexts-newer-families.md): ``alpha`` may have its own formula under the
    identity link, ``["y ~ x", "alpha ~ z"]`` with ``Priors(alpha={"intercept": ..., "b": ...})``; no formula on ``sigma``, no
    ``weights()``, ``cens()`` or ``trunc()`` (D79).
    """
    return Model(formula, data, priors, family="skew_normal", center_predictors=center_predictors, categorical=categorical)


def shifted_lognormal(formula: str, data: pd.DataFrame, priors: Priors, *,
                      center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Shifted lognormal response (brms ``shifted_lognormal()``, D01, docs/continuous-families.md): ``log(y - ndt) ~
    Normal(mu, sigma)`` with the identity link on ``mu`` and the non-decision time ``0 < ndt < min(y)``;
    ``E[y] = exp(mu + sigma^2 / 2) + ndt``.

    Responses are strictly positive finite reals. ``sigma`` takes a positive-support prior; ``ndt`` a prior truncated to
    (0, min(y)) and renormalised there, as brms does (``Priors(ndt=ab.Normal(0, 0.3))``), or brms's default
    ``uniform(0, min_Y)``, ``Priors(ndt=ab.Uniform(0, "min_Y"))``. ``predict(kind="mean")`` returns brms's ``posterior_epred``
    as ``mean``; ``kind="linear"`` returns ``mu``. No distributional formula, ``weights()``, ``cens()`` or ``trunc()`` (D79).
    """
    return Model(formula, data, priors, family="shifted_lognormal", center_predictors=center_predictors, categorical=categorical)


def exgaussian(formula: str, data: pd.DataFrame, priors: Priors, *,
               center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Ex-Gaussian response (brms ``exgaussian()``, D01, docs/continuous-families.md): ``Normal(mu - beta, sigma)`` plus an
    exponential of mean ``beta``, with the identity link on the mean ``mu``.

    ``sigma`` and ``beta`` take positive-support priors, ``Priors(sigma=ab.HalfNormal(1), beta=ab.Gamma(1, 0.1))`` (brms
    classes ``sigma`` and ``beta``; ``gamma(1, 0.1)`` is brms's default for ``beta``). ``predict(kind="mean")`` returns
    ``mu``. No distributional formula, ``weights()``, ``cens()`` or ``trunc()`` (D79).
    """
    return Model(formula, data, priors, family="exgaussian", center_predictors=center_predictors, categorical=categorical)


def beta_binomial(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
                  center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Beta-binomial response (brms ``beta_binomial()``, D02, docs/discrete-families.md): ``successes | trials(n) ~ ...`` with the
    logit link on the mean success probability ``mu`` and the precision ``phi``, ``k ~ BetaBinomial(n, mu phi, (1 - mu) phi)``.

    Successes and trials as for the binomial family (whole numbers up to 1e6, zero-trial rows dropped). ``phi`` takes a
    positive-support prior, ``Priors(phi=ab.Gamma(0.01, 0.01))`` (brms class ``phi``; brms's default, on request through
    ``ab.default_priors``). ``predict(kind="mean")`` returns ``mu`` and ``expected = n mu`` (brms's ``posterior_epred``),
    ``kind="probability"`` ``mu`` alone. E03 (docs/contexts-newer-families.md): ``phi`` may have its own formula under a log link,
    ``["k | trials(size) ~ x", "phi ~ z"]`` with ``Priors(phi={"intercept": ..., "b": ...})``, and ``weights()`` is taken after
    ``trials()``; no other link (D80).
    """
    return Model(formula, data, priors, family="beta_binomial", center_predictors=center_predictors, categorical=categorical)


def geometric(formula: str, data: pd.DataFrame, priors: Priors, *,
              center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Geometric count response (brms ``geometric()``, D02, docs/discrete-families.md): log link on the mean,
    ``y ~ NegativeBinomial(mu, 1)``, ``Var[y] = mu + mu^2``; no auxiliary parameter.

    Counts as for Poisson (integers in [0, 1e6], zero rows retained). ``predict(kind="mean")`` returns ``mu``. No
    ``weights()`` or other link (D80, D81).
    """
    return Model(formula, data, priors, family="geometric", center_predictors=center_predictors, categorical=categorical)


def negbinomial2(formula: str, data: pd.DataFrame, priors: Priors, *,
                 center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Negative binomial count response parameterised by its dispersion (brms ``negbinomial2()``, D02,
    docs/discrete-families.md): log link on the mean, ``y ~ NegativeBinomial(mu, shape = 1 / sigma)``,
    ``Var[y] = mu + sigma mu^2``.

    Counts as for Poisson. The dispersion ``sigma`` (brms's ``1 / shape`` of ``negbinomial``) takes a positive-support prior,
    ``Priors(sigma=ab.Gamma(0.4, 0.3))`` (brms class ``sigma``; brms's default, on request through ``ab.default_priors``).
    ``predict(kind="mean")`` returns ``mu``. No distributional formula, ``weights()`` or other link (D80, D81).
    """
    return Model(formula, data, priors, family="negbinomial2", center_predictors=center_predictors, categorical=categorical)


def inverse_gaussian(formula: str, data: pd.DataFrame, priors: Priors, *, link: str = "1/mu^2",
                     center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Inverse Gaussian response (brms ``inverse.gaussian()``, E02, docs/continuous-families.md): mean ``mu`` and shape
    ``shape``, ``Var[y] = mu^3 / shape``, under brms's default link ``"1/mu^2"`` (``mu = 1 / sqrt(eta)``, which needs a positive
    linear predictor: the density is -inf where it is not) or ``link="log"`` (``mu = exp(eta)``).

    Responses are strictly positive finite reals. The shape takes a positive-support prior, ``Priors(shape=ab.Gamma(0.01,
    0.01))`` (brms class ``shape``; brms's default). ``predict(kind="mean")`` returns ``mu``. No ``inverse`` or ``identity`` link,
    distributional formula or addition term (D112).
    """
    return Model(formula, data, priors, family="inverse_gaussian", link=link, center_predictors=center_predictors,
                 categorical=categorical)


def von_mises(formula: str, data: pd.DataFrame, priors: Priors, *,
              center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """von Mises response (brms ``von_mises()``, E02, docs/continuous-families.md): angles in [-pi, pi], mean direction ``mu =
    2 atan(eta)`` (brms's ``tan_half`` link) and precision ``kappa``.

    ``kappa`` takes a positive-support prior, ``Priors(kappa=ab.Gamma(2, 0.01))`` (brms class ``kappa``; brms's default).
    ``predict(kind="mean")`` returns ``mu``. No ``identity`` link, distributional formula or addition term (D112).
    """
    return Model(formula, data, priors, family="von_mises", center_predictors=center_predictors, categorical=categorical)


def asym_laplace(formula: str, data: pd.DataFrame, priors: Priors, *, quantile: float | None = None,
                 center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Asymmetric Laplace response (brms ``asym_laplace()``, E02, docs/continuous-families.md): quantile regression, the
    location ``mu`` (identity link) being the ``p``-quantile of ``y``, with scale ``sigma``.

    ``quantile=None`` learns the quantile under ``Priors(quantile=ab.Beta(...))`` (brms class ``quantile``, default ``beta(1,
    1)``); ``quantile=0.25`` fixes it (brms's ``bf(y ~ ..., quantile = 0.25)``), with no prior. ``sigma`` takes a
    positive-support prior. ``predict(kind="mean")`` returns brms's ``posterior_epred``, ``mu + sigma (1 - 2 p) / (p (1 - p))``,
    as ``mean``; ``kind="linear"`` returns ``mu``. No distributional formula or addition term (D112).
    """
    return Model(formula, data, priors, family="asym_laplace", quantile=quantile, center_predictors=center_predictors,
                 categorical=categorical)


def frechet(formula: str, data: pd.DataFrame, priors: Priors, *, nu: float | None = None,
            center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Frechet response (brms ``frechet()``, E02, docs/continuous-families.md): log link on the mean, ``E[y] = mu = exp(eta)``,
    scale ``mu / Gamma(1 - 1 / nu)`` and shape ``nu > 1``.

    Responses are strictly positive finite reals. ``nu=None`` learns the shape under ``Priors(nu=...)`` truncated to ``nu > 1``
    (brms's ``real<lower=1> nu``, default ``gamma(2, 0.1)``), as the Student-t family's ``nu``; ``nu=3`` fixes it.
    ``predict(kind="mean")`` returns ``mu``. No other link, distributional formula or addition term (D112).
    """
    return Model(formula, data, priors, family="frechet", nu=nu, center_predictors=center_predictors, categorical=categorical)


def gen_extreme_value(formula: str, data: pd.DataFrame, priors: Priors, *,
                      center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Generalized extreme value response (brms ``gen_extreme_value()``, E02, docs/continuous-families.md): location ``mu``
    (identity link), scale ``sigma`` and shape ``xi``, derived as brms derives it from its unscaled ``tmp_xi`` so that every
    fitted row lies inside the support.

    ``sigma`` takes a positive-support prior, ``xi`` a location prior on ``tmp_xi``, ``Priors(xi=ab.Normal(0, 2.5))`` (brms class
    ``xi``; brms's default). ``predict(kind="mean")`` returns ``mu + sigma (Gamma(1 - xi) - 1) / xi`` as ``mean`` (infinite for
    ``xi >= 1``, D114); ``kind="linear"`` returns ``mu``. No other link, distributional formula or addition term (D112).
    """
    return Model(formula, data, priors, family="gen_extreme_value", center_predictors=center_predictors, categorical=categorical)


def zero_inflated_poisson(formula: str, data: pd.DataFrame, priors: Priors, *,
                          center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Zero-inflated Poisson counts (brms ``zero_inflated_poisson``): a structural zero with probability ``zi``, a
    ``Poisson(mu)`` count otherwise, ``mu = exp(eta)``; ``E[y] = (1 - zi) mu``.

    Counts as for Poisson (integers in [0, 1e6], zero rows retained). The zero-inflation probability takes its own prior on
    (0, 1), ``Priors(zi=ab.Beta(1, 1))`` (brms class ``zi``; brms's default ``beta(1, 1)`` is not filled in silently,
    ``ab.default_priors(..., family="zero_inflated_poisson")`` returns it); the intercept and slope priors act on the
    log of the Poisson part's mean. ``predict(kind="mean")`` returns ``(1 - zi) mu`` as ``mean``; ``kind="linear"``
    returns ``eta``. ``weights()`` is taken (D03). No overdispersion or distributional (``zi ~ ...``) formula
    (docs/zero-inflated-hurdle.md).
    """
    return Model(formula, data, priors, family="zero_inflated_poisson",
                 center_predictors=center_predictors, categorical=categorical)


def hurdle_poisson(formula: str, data: pd.DataFrame, priors: Priors, *,
                   center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Hurdle Poisson counts (brms ``hurdle_poisson``): zero with probability ``hu``, a zero-truncated ``Poisson(mu)``
    count otherwise, ``mu = exp(eta)``; ``E[y] = (1 - hu) mu / (1 - exp(-mu))``.

    Counts as for Poisson. The hurdle probability takes its own prior on (0, 1), ``Priors(hu=ab.Beta(1, 1))`` (brms class
    ``hu``, brms's default ``beta(1, 1)`` on request through ``ab.default_priors``); the intercept and slope priors act on
    the log of the untruncated Poisson part's mean. ``predict(kind="mean")`` returns brms's expected count as ``mean``;
    ``kind="linear"`` returns ``eta``. ``weights()`` is taken (D03). No overdispersion or distributional (``hu ~ ...``)
    formula (docs/zero-inflated-hurdle.md).
    """
    return Model(formula, data, priors, family="hurdle_poisson",
                 center_predictors=center_predictors, categorical=categorical)


# -- D03: more zero-inflated and hurdle families (docs/zero-inflated-hurdle.md) ---------------------------------------------
_D03_CONTEXTS = ("No distributional formula, cens(), trunc(), mo(), me(), mi(), mm(), gr(by = ), autocorrelation term, multivariate "
                 "or non-linear formula (divergence D84).")
_E03_CONTEXTS = _D03_CONTEXTS.replace("No distributional formula,", "No other distributional formula,")   # zi / hu formulas (E03)


def zero_inflated_negbinomial(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
                              center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Zero-inflated negative binomial counts (brms ``zero_inflated_negbinomial``, D03): a structural zero with probability
    ``zi``, a ``NB(mu, shape)`` count otherwise (mean ``mu = exp(eta)``, variance ``mu + mu^2 / shape``); ``E[y] = (1 - zi) mu``.

    Counts as for Poisson. ``Priors(shape=..., zi=ab.Beta(...))`` (brms classes ``shape`` and ``zi``; brms's defaults
    ``inv_gamma(0.4, 0.3)`` and ``beta(1, 1)`` on request through ``ab.default_priors``); the intercept and slope priors act on
    the log of the negative binomial part's mean. ``predict(kind="mean")`` returns ``(1 - zi) mu`` as ``mean``;
    ``kind="linear"`` returns ``eta``. ``weights()`` is taken. E03 (docs/contexts-newer-families.md): ``zi`` may have its own
    formula under the logit link, ``["zcnt ~ x", "zi ~ z"]`` with ``Priors(zi={"intercept": ..., "b": ...})``. """ + _E03_CONTEXTS
    return Model(formula, data, priors, family="zero_inflated_negbinomial",
                 center_predictors=center_predictors, categorical=categorical)


def hurdle_negbinomial(formula: str, data: pd.DataFrame, priors: Priors, *,
                       center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Hurdle negative binomial counts (brms ``hurdle_negbinomial``, D03): zero with probability ``hu``, a zero-truncated
    ``NB(mu, shape)`` count otherwise, ``mu = exp(eta)``; ``E[y] = (1 - hu) mu / (1 - (shape / (mu + shape))^shape)``.

    Counts as for Poisson. ``Priors(shape=..., hu=ab.Beta(...))``; the intercept and slope priors act on the log of the
    untruncated negative binomial part's mean. ``predict(kind="mean")`` returns brms's expected count as ``mean``;
    ``kind="response"`` draws exact zero-truncated negative binomial counts (D86). ``weights()`` is taken. """ + _D03_CONTEXTS
    return Model(formula, data, priors, family="hurdle_negbinomial",
                 center_predictors=center_predictors, categorical=categorical)


def zero_inflated_binomial(formula: str, data: pd.DataFrame, priors: Priors, *,
                           center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Zero-inflated binomial successes (brms ``zero_inflated_binomial``, D03): ``successes | trials(n) ~ ...``, a structural
    zero with probability ``zi``, a ``Binomial(n, p)`` count otherwise, ``p = expit(eta)``; ``E[y] = (1 - zi) n p``.

    Trials as for the binomial family (zero-trial rows dropped, D4). ``Priors(zi=ab.Beta(...))``; the intercept and slope
    priors are on the logit scale. ``predict(kind="mean")`` returns brms's expected count ``(1 - zi) n p`` as ``expected``
    (new rows need the trials column); ``kind="linear"`` returns ``eta``. The logit link only. ``weights()`` is taken (after
    ``trials()``: ``k | trials(n) + weights(w) ~ ...``). """ + _D03_CONTEXTS
    return Model(formula, data, priors, family="zero_inflated_binomial",
                 center_predictors=center_predictors, categorical=categorical)


def zero_inflated_beta(formula: str, data: pd.DataFrame, priors: Priors, *,
                       center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Zero-inflated Beta proportions (brms ``zero_inflated_beta``, D03): responses in [0, 1), an exact zero with probability
    ``zi``, a ``Beta(mu phi, (1 - mu) phi)`` proportion otherwise, ``mu = expit(eta)``; ``E[y] = (1 - zi) mu``.

    ``Priors(phi=..., zi=ab.Beta(...))`` (brms's defaults ``gamma(0.01, 0.01)`` and ``beta(1, 1)`` on request); the intercept
    and slope priors are on the logit scale. ``predict(kind="mean")`` returns ``(1 - zi) mu`` as ``mean``; ``kind="linear"``
    returns ``eta``. An exact 1 is refused; the logit link only. ``weights()`` is taken. """ + _D03_CONTEXTS
    return Model(formula, data, priors, family="zero_inflated_beta",
                 center_predictors=center_predictors, categorical=categorical)


def hurdle_gamma(formula: str, data: pd.DataFrame, priors: Priors, *,
                 center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Hurdle Gamma responses (brms ``hurdle_gamma``, D03): non-negative reals, an exact zero with probability ``hu``, a
    ``Gamma(shape, shape / mu)`` value otherwise, ``mu = exp(eta)``; ``E[y] = (1 - hu) mu``.

    ``Priors(shape=..., hu=ab.Beta(...))`` (brms's defaults ``gamma(0.01, 0.01)`` and ``beta(1, 1)`` on request); the intercept
    and slope priors act on the log of the Gamma part's mean. ``predict(kind="mean")`` returns ``(1 - hu) mu`` as ``mean``;
    ``kind="linear"`` returns ``eta``. ``weights()`` is taken. """ + _D03_CONTEXTS
    return Model(formula, data, priors, family="hurdle_gamma",
                 center_predictors=center_predictors, categorical=categorical)


def hurdle_lognormal(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
                     center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Hurdle lognormal responses (brms ``hurdle_lognormal``, D03): non-negative reals, an exact zero with probability ``hu``,
    ``LogNormal(mu, sigma)`` otherwise with the identity link on the log-scale location ``mu``; ``E[y] = (1 - hu) exp(mu +
    sigma^2 / 2)``.

    ``Priors(sigma=..., hu=ab.Beta(...))``; the intercept and slope priors act on log y in the lognormal part.
    ``predict(kind="mean")`` returns brms's ``posterior_epred`` as ``mean``; ``kind="linear"`` returns ``mu``. ``weights()`` is
    taken. E03 (docs/contexts-newer-families.md): ``hu`` may have its own formula under the logit link, ``["zpos ~ x", "hu ~ z"]``
    with ``Priors(hu={"intercept": ..., "b": ...})``. """ + _E03_CONTEXTS
    return Model(formula, data, priors, family="hurdle_lognormal",
                 center_predictors=center_predictors, categorical=categorical)


# -- E01: bounded families (docs/bounded-families.md) ------------------------------------------------------------------------
_E01_CONTEXTS = ("No distributional formula, cens(), trunc(), mo(), me(), mi(), mm(), gr(by = ), autocorrelation term, multivariate "
                 "or non-linear formula, mixture component or other link (divergence D108).")


def zero_one_inflated_beta(formula: str, data: pd.DataFrame, priors: Priors, *,
                           center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Zero-one-inflated Beta proportions (brms ``zero_one_inflated_beta``, E01): responses in [0, 1], an exact 0 or 1 with
    probability ``zoi`` (a 1 with probability ``coi`` given that), a ``Beta(mu phi, (1 - mu) phi)`` proportion otherwise, ``mu =
    expit(eta)``; ``E[y] = zoi coi + (1 - zoi) mu``.

    ``Priors(phi=..., zoi=ab.Beta(...), coi=ab.Beta(...))`` (brms classes ``phi``, ``zoi`` and ``coi``; brms's defaults
    ``gamma(0.01, 0.01)``, ``beta(1, 1)`` and ``beta(1, 1)`` on request through ``ab.default_priors``); the intercept and slope
    priors are on the logit scale. ``predict(kind="mean")`` returns brms's ``posterior_epred`` as ``mean``; ``kind="linear"``
    returns ``eta``. ``weights()`` is taken. """ + _E01_CONTEXTS
    return Model(formula, data, priors, family="zero_one_inflated_beta",
                 center_predictors=center_predictors, categorical=categorical)


def zero_inflated_beta_binomial(formula: str, data: pd.DataFrame, priors: Priors, *,
                                center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Zero-inflated beta-binomial successes (brms ``zero_inflated_beta_binomial``, E01): ``successes | trials(n) ~ ...``, a
    structural zero with probability ``zi``, a ``BetaBinomial(n, mu phi, (1 - mu) phi)`` count otherwise, ``mu = expit(eta)``;
    ``E[y] = (1 - zi) n mu``.

    Trials as for the binomial family (zero-trial rows dropped, D4). ``Priors(phi=..., zi=ab.Beta(...))`` (brms's defaults
    ``gamma(0.01, 0.01)`` and ``beta(1, 1)`` on request); the intercept and slope priors are on the logit scale.
    ``predict(kind="mean")`` returns brms's expected count ``(1 - zi) n mu`` as ``expected`` (new rows need the trials column);
    ``kind="linear"`` returns ``eta``. ``weights()`` is taken (after ``trials()``). """ + _E01_CONTEXTS
    return Model(formula, data, priors, family="zero_inflated_beta_binomial",
                 center_predictors=center_predictors, categorical=categorical)


def xbeta(formula: str, data: pd.DataFrame, priors: Priors, *,
          center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Extended-support Beta proportions (brms ``xbeta``, Kosmidis and Zeileis 2024; E01): responses in [0, 1] from one process,
    ``B ~ Beta(mu phi, (1 - mu) phi)`` with ``mu = expit(eta)`` stretched to ``(1 + 2 kappa) B - kappa`` and censored to [0, 1],
    so that exact 0s and 1s carry the mass of the stretched Beta beyond the ends.

    ``Priors(phi=..., kappa=...)`` (brms classes ``phi`` and ``kappa``, both ``real<lower=0>``; brms's defaults
    ``gamma(0.01, 0.01)`` for both on request through ``ab.default_priors``); the intercept and slope priors are on the logit
    scale of ``mu``. ``predict(kind="mean")`` returns brms's ``posterior_epred``, the censored variable's mean (not ``mu``), as
    ``mean``; ``kind="linear"`` returns ``eta``. ``weights()`` is taken. """ + _E01_CONTEXTS
    return Model(formula, data, priors, family="xbeta", center_predictors=center_predictors, categorical=categorical)


def hurdle_cumulative(formula: str, data: pd.DataFrame, priors: Priors, *,
                      center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Hurdle ordinal response (brms ``hurdle_cumulative("logit")``, E01): the hurdle category with probability ``hu``, otherwise
    one of K ordered categories of a cumulative logit model, ``P(y <= k | y > 0) = expit(tau_k - mu)``, with K - 1 increasing
    thresholds ``tau`` and the linear predictor ``mu`` without an intercept.

    The response is brms's: whole numbers ``0..K`` (0 the hurdle) or an ordered pandas Categorical (its first declared category
    the hurdle), fitted as brms's ``Y`` (codes 0..K); ``model.categories`` holds the K ordered categories and
    ``model.hurdle_category`` the hurdle's label. ``Priors(intercept=..., hu=ab.Beta(...))``: the intercept prior applies to every
    threshold (brms's class ``Intercept``), sampled for the centred design on Stan's ordered coordinates as for
    ``ab.cumulative``; brms's defaults on request. The intercept cannot be removed. ``predict(kind="mean")`` returns the K + 1
    probabilities ``p`` with a ``category`` dimension, the hurdle first; ``kind="linear"`` returns ``mu``, ``kind="response"``
    codes 0..K. The logit link only, ``disc = 1`` and brms's flexible thresholds (D109); no ``weights()``, ``cs()``, and """ + \
        _E01_CONTEXTS[3:]
    return Model(formula, data, priors, family="hurdle_cumulative", center_predictors=center_predictors, categorical=categorical)


def cumulative(formula: str, data: pd.DataFrame, priors: Priors, *, link: str = "logit",
               center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Ordinal response (brms ``cumulative("logit")``): K ordered categories, ``P(y <= k) = expit(tau_k - mu)`` with K - 1
    increasing thresholds ``tau`` and the linear predictor ``mu`` without an intercept (docs/ordinal-categorical.md).

    The response is positive integers ``1..K`` (K = the largest; brms's rule) or an ordered pandas Categorical (its declared
    categories, in order); it is fitted as its 1-based category, ``model.categories`` holding the labels. The intercept prior
    ``Priors(intercept=...)`` applies to every threshold, as brms's class ``Intercept`` does; the thresholds are sampled for
    the centred design (brms's ``Intercept``), on Stan's ordered coordinates, and the graph's ``b_Intercept`` holds them for
    the raw design. The intercept cannot be removed (brms refuses it too). ``predict(kind="mean")`` returns the category
    probabilities ``p`` with a ``category`` dimension, ``kind="linear"`` returns ``mu``, ``kind="response"`` 1-based
    categories. ``link`` is ``"logit"`` or ``"probit"`` (D04, accepted against P6's reference, divergence D88); brms's
    ``cumulative("cloglog")`` is implemented but withheld (E06: brms's own log density of its frozen case is inexact, divergence
    D132; docs/links.md) and its other links are not offered. No ``disc``, category-specific effects or threshold structures
    (divergences D38, D89).
    """
    return Model(formula, data, priors, family="cumulative", link=link, center_predictors=center_predictors, categorical=categorical)


def _sequential(family: str, meaning: str):
    def build(formula: str, data: pd.DataFrame, priors: Priors, *, center_predictors: bool = True,
              categorical: Mapping[str, object] | None = None) -> Model:
        return Model(formula, data, priors, family=family, center_predictors=center_predictors, categorical=categorical)

    build.__name__ = build.__qualname__ = family
    build.__doc__ = f"""Ordinal response under brms's ``{family}("logit")`` (D04, docs/ordinal-categorical.md): {meaning}, with K - 1
    thresholds ``tau`` (brms's ``b_Intercept``; unordered, as brms's ``vector[nthres] Intercept``) and the linear predictor ``mu``
    without an intercept.

    The response is coded as for ``ab.cumulative`` (positive integers ``1..K`` or an ordered pandas Categorical, fitted as its
    1-based category). ``Priors(intercept=...)`` applies to every threshold (brms's class ``Intercept``); the thresholds are
    sampled for the centred design (brms's ``Intercept``) as they are, and the graph's ``b_Intercept`` holds them for the raw
    design. ``cs(x)`` terms (brms's category-specific effects) give the numeric column ``x`` one coefficient per threshold
    (brms's ``bcs``, under the ``b`` prior); ``predict(kind="linear")`` then returns every threshold's predictor (a
    ``threshold`` dimension). ``kind="mean"`` returns the category probabilities ``p`` with a ``category`` dimension,
    ``kind="response"`` 1-based categories. The logit link only, ``disc = 1``, brms's flexible thresholds (divergence D38).
    """
    return build


sratio = _sequential("sratio", "the stopping ratio, ``P(y = k | y >= k) = expit(tau_k - mu)``")
cratio = _sequential("cratio", "the continuation ratio, ``P(y > k | y >= k) = expit(mu - tau_k)``")
acat = _sequential("acat", "adjacent categories, ``log(P(y = k + 1) / P(y = k)) = mu - tau_k``")


def categorical(formula: str, data: pd.DataFrame, priors: Priors, *,
                center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Unordered categorical response (brms ``categorical()``): one linear predictor ``mu_k`` per non-reference category,
    ``P(y = k) = softmax(0, mu_2, ..., mu_K)_k`` (docs/ordinal-categorical.md).

    The categories are the response's distinct labels sorted as brms's ``as.factor`` sorts them (text in C-locale order,
    integer codes numerically); the first is the reference and ``model.categories`` holds them all. Each non-reference
    category has its own intercept and coefficients on the same design, all under the one ``Priors(intercept=...)`` and
    ``b`` (one prior per class shared by the categories, divergence D37). ``predict(kind="linear")`` returns ``mu`` with a
    ``category`` dimension over the non-reference categories, ``kind="mean"`` the probabilities ``p`` over all categories,
    ``kind="response"`` 1-based categories. Population-level terms only: group-level terms and the offset are refused in
    this slice (D37). ``categorical={column: reference}`` declares categorical predictors as for every family.
    """
    return Model(formula, data, priors, family="categorical", center_predictors=center_predictors, categorical=categorical)


def mixture(formula: str, data: pd.DataFrame, priors: Priors, *, families: list[str], order: str | bool | None = None,
            center_predictors: bool = True, categorical: Mapping[str, object] | None = None, nl: bool = False) -> Model:
    """A finite mixture (brms ``mixture(gaussian, gaussian)``, C05, docs/mixtures.md): ``y_i ~ sum_k theta_k
    Normal(mu_k,i, sigma_k)`` with one linear predictor per component on the formula's design (brms's ``mu<k>``, each with
    its own centred intercept and coefficients), one residual SD per component (``sigma<k>``) and the mixing proportions
    ``theta``.

    ``families`` names the components, one entry each: Gaussian components, at least two (``["gaussian", "gaussian"]``).
    ``order`` is brms's identification, ``"mu"`` (its default for components of one family; ``None`` means it): the
    components' centred intercepts are ordered, ``a_c[1] < ... < a_c[K]`` (brms's ``ordered_Intercept``). ``priors`` is
    ``ab.Priors(components=[ab.Priors(intercept=..., b=..., sigma=...), ...], theta=ab.Dirichlet(...))``, one Priors per
    component in the order of ``families`` (brms's ``dpar = mu<k>`` and ``class = sigma<k>``) and the Dirichlet of the
    mixing proportions; ``ab.default_priors(formula, data, family="mixture", families=...)`` returns brms's defaults.
    ``predict(kind="linear")`` returns every component's ``mu`` with a ``component`` dimension (brms's
    ``posterior_linpred``), ``kind="mean"`` the mixture's expected response ``sum_k theta_k mu_k`` as ``mean`` and
    ``kind="response"`` draws a component by theta, then its Normal. Population-level terms only; one formula for every
    component; other component families, ``order="none"``, group-level terms and offsets are refused (D52-D55), and so is
    ``nl=True``, a non-linear formula in a mixture (C02, D62).
    """
    if nl:                                                    # C02: brms's bf(..., nl = TRUE) with mixture(...), not verified
        raise FormulaError("nl=True with a mixture is not supported (docs/nonlinear.md, divergence D62): brms puts the non-linear "
                           "expression in every component's mean (mu1, mu2) with shared non-linear parameters, which is not "
                           "verified here; a non-linear model is ab.gaussian([...], data, priors, nl=True)")
    return Model(formula, data, priors, family="mixture", families=families, order=order, center_predictors=center_predictors,
                 categorical=categorical)
