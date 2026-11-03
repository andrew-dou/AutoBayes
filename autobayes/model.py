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
``Design.sar`` holds it and its eigenvalues). brms's fixed residual covariance ``fcor(M)`` (F03a, docs/fcor.md) makes the
Gaussian likelihood the rows' joint normal with covariance ``sigma^2 M``, ``M`` from ``data2`` (``autobayes.fcor``; ``Design.fcor``
holds ``M`` and its Cholesky factor). brms's spatial CAR term ``car(M, gr, type)`` (F02a, docs/spatial-car.md) adds one effect per
location to the Gaussian or Poisson linear predictor through the same hook as ``mo()``, its adjacency matrix read from ``data2``
(``autobayes.car``; ``Design.car`` holds the locations and the graph). brms's residual correlation structures ``unstr(time, gr)`` and
``cosy(time, gr)`` (F03b, docs/unstr-cosy.md) replace the Gaussian or Student-t likelihood by the joint density of each
group's rows (``autobayes.timecor``; ``Design.timecor`` holds the groups of the fitted rows).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd

from collections.abc import Mapping

from autobayes.correlation import check_correlation_cholesky, cholesky_corr_free, cholesky_corr_rv, n_free
from autobayes import expressions
from autobayes.families import FAMILIES, FamilySpec, get_family
from autobayes.formula import (Categorical, FormulaError, ModelSpec, addition_data, bambi_group_name, binomial_counts, by_levels_of,
                               categorical_labels, check_imputation, cs_label, cs_values, factor_column, group_labels, interaction_cells,
                               measurement_values, missing_values, mm_weights, parse_formula, poly_contrasts, pop_term_design, sd_tag,
                               with_trials)
from autobayes.formula import DATA2_UNIVARIATE, check_data2   # Tier F: brms's data2 (F02b, F03a)
from autobayes.formula import derived_group_columns, group_covariate   # G05a: the covariates of group-level slopes
from autobayes.group_index import level_codes   # G11: group-level terms from level codes (docs/scaling.md)
from autobayes.priors import AUXILIARY, PriorError, Priors, class_default, describe_prior

_bambi_group_name = bambi_group_name          # ``1|g`` for the varying intercept of ``g``, ``x|g`` for the slope of ``x``
_SYMMETRIC_HALF = frozenset({"HalfNormal", "HalfStudentT", "HalfCauchy"})   # D08: half priors of a distribution symmetric about 0
_AUTOCOR_MEANING = {"ar": "autoregressive", "ma": "moving-average"}   # an autocorrelation term's parts (C04)
_DPAR_INVERSE = {"log": "exp(eta_{})", "logit": "inv_logit(eta_{})", "identity": "eta_{}",   # describe()'s value of a dpar (B01, E03)
                 "logm1": "1 + exp(eta_{})"}                                               # G06: the Student-t nu


def _dpar_effect(name: str, factor: str, coef: str) -> str:
    """G06: how ``describe()`` writes a distributional formula's group-level effect: ``u_sigma,g`` for the varying intercept of
    ``g`` in sigma's formula (D08's symbol), ``v_sigma,x,g`` for the varying slope of ``x``."""
    return f"u_{name},{factor}" if coef == "Intercept" else f"v_{name},{coef},{factor}"


def _prior_rv(name: str, prior, *, positive: bool = False):
    """A scalar PyMC variable named ``name`` under ``prior`` (its parameters as float64, which PyTensor keeps; C06): a
    location prior, a positive-support prior, or ``Flat()``, flat on the real line, or with ``positive`` on the positive half
    line (PyMC's ``HalfFlat``, brms's flat prior on a ``real<lower=0>``). Must be called inside a PyMC model context."""
    import pymc as pm

    dist = pm.HalfFlat if positive and prior.name == "Flat" else getattr(pm, prior.name)
    return dist(name, **{k: np.float64(v) for k, v in prior.params.items()})


@dataclass(frozen=True, init=False)
class GroupBlock:
    """The design block of one group-level coefficient ``c`` of grouping factor ``m`` (docs/term-model.md).

    ``levels`` are the factor's sorted fitted labels (brms's level order; ``N_m`` of them). ``Z`` is the
    ``(n, N_m)`` block: row ``i`` holds the coefficient's covariate value (one for the intercept, so the block
    is the one-hot indicator) in the column of row ``i``'s level, which is brms's ``Z_m_c[i]`` spread over
    its ``J_m[i]``. ``group_index`` is that level's 0-based index per row (brms ``J_m - 1``) and
    ``column_slice`` where the block sits in the concatenated ``Design.Z``.

    G11 (docs/scaling.md): the block keeps the level codes and the covariate, not ``Z``: ``Z`` is built from them on access
    (the zeros plus each row's value at its level, the array it always was) and refused above
    ``group_index.DENSE_Z_MAX_ENTRIES`` entries. ``covariate`` is the slope's column per row (None: ones, the varying
    intercept, or a monotonic slope's level indicator); a block given a dense ``Z`` (``GroupBlock(levels, Z, ...)``, which
    the families that build their own designs use) keeps it as ``dense``.
    """
    levels: tuple[str, ...]
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
    # G05a, a monotonic varying slope (1 + mo(ord) | g): the column of the monotonic term the level's effect multiplies (brms's
    # r[J] * mo(simo, Xmo), no Z column); Z is then the level indicator. None for every other coefficient.
    monotonic: str | None = None
    # G05b, a varying slope of a multi-membership term (1 + x | mm(g, g2)), mmc(x1, x2): each member's covariate per row (brms's
    # Z_m_c_k: the slope's column for every member, member k's own column for mmc()); Z is their weighted sum of indicators times
    # them. Empty for the varying intercept (ones) and every other term.
    covariates: tuple[np.ndarray, ...] = ()
    # G11: the covariate of a single-member block's coefficient per row (None for ones), and a dense Z given as such
    covariate: np.ndarray | None = None
    dense: np.ndarray | None = field(default=None, repr=False)

    def __init__(self, levels, Z=None, group_index=None, column_slice=None, memberships=(), weights=(), by_levels=(),
                 by_index=None, monotonic=None, covariates=(), *, covariate=None, dense=None):
        """``GroupBlock(levels, Z, group_index, column_slice, ...)`` as before G11 (``Z`` dense), or ``Z=None`` with the codes
        and ``covariate`` (the block's ``Z`` built on access); ``dataclasses.replace(block, Z=...)`` gives it a dense ``Z``."""
        if group_index is None or column_slice is None:
            raise TypeError("GroupBlock needs group_index and column_slice")
        values = dict(levels=levels, group_index=group_index, column_slice=column_slice, memberships=memberships,
                      weights=weights, by_levels=by_levels, by_index=by_index, monotonic=monotonic, covariates=covariates,
                      covariate=covariate, dense=Z if Z is not None else dense)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def J(self) -> int:
        """The number of levels (brms ``N_m``; this library has always called it ``J``)."""
        return len(self.levels)

    @property
    def n(self) -> int:
        """The number of rows."""
        return len(self.group_index)

    @property
    def Z(self) -> np.ndarray:
        """The dense ``(n, J)`` block, built on access (G11): zeros, then each row's value at its level (a multi-membership
        term's weights summed member by member, ``np.add.at``); refused above ``group_index.DENSE_Z_MAX_ENTRIES`` entries."""
        if self.dense is not None:
            return self.dense
        from autobayes.group_index import check_dense_size

        check_dense_size(self.n, self.J, "GroupBlock.Z")
        Z = np.zeros((self.n, self.J))
        self.fill(Z)
        return Z

    def fill(self, Z: np.ndarray) -> None:
        """Write the block into ``Z``, an ``(n, J)`` array of zeros (or a column slice of one): as ``Z`` is built."""
        if self.dense is not None:
            Z[...] = self.dense
            return
        rows = np.arange(self.n)
        if self.memberships:                                 # D08, G05b: member by member, as the model's own blocks always were
            for k, (index, weight) in enumerate(zip(self.memberships, self.weights)):
                np.add.at(Z, (rows, index), weight if not self.covariates else weight * self.covariates[k])
        else:
            Z[rows, self.group_index] += 1.0 if self.covariate is None else self.covariate

    def row_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Each row's entries of ``Z``, ``(columns, values)`` of shape ``(n, k)`` (k the members, 1 for every term but a
        multi-membership one), columns ascending within a row and a member's repeat of a level summed into its first
        place as ``Z`` sums it (a repeated place keeps column -1). Without forming ``Z``."""
        if self.dense is not None:                           # a dense block: its nonzero entries, at most one row's width
            raise ValueError("a dense GroupBlock has no row values; read its Z")
        if not self.memberships:
            values = 0.0 + (np.ones(self.n) if self.covariate is None else np.asarray(self.covariate, dtype=float))
            return np.asarray(self.group_index, dtype=np.int64)[:, None], values[:, None]
        k = len(self.memberships)
        cols = np.stack([np.asarray(m, dtype=np.int64) for m in self.memberships], axis=1)
        vals = np.zeros((self.n, k))
        for j, weight in enumerate(self.weights):
            v = np.asarray(weight if not self.covariates else weight * self.covariates[j], dtype=float)
            placed = np.zeros(self.n, dtype=bool)
            for i in range(j):                               # a level an earlier member holds: summed there, as np.add.at does
                hit = ~placed & (cols[:, i] == cols[:, j])
                vals[hit, i] = vals[hit, i] + v[hit]
                placed |= hit
            vals[~placed, j] = 0.0 + v[~placed]
            cols[placed, j] = -1
        order = np.argsort(np.where(cols < 0, np.iinfo(np.int64).max, cols), axis=1, kind="stable")
        return np.take_along_axis(cols, order, axis=1), np.take_along_axis(vals, order, axis=1)

    def dot(self, v: np.ndarray) -> np.ndarray:
        """``Z @ v`` (one value per row) without forming ``Z``: each row's value times ``v`` at its level, summed over the
        members of a multi-membership term."""
        v = np.asarray(v)
        if self.dense is not None:
            return self.dense @ v
        if not self.memberships:
            return v[self.group_index] if self.covariate is None else self.covariate * v[self.group_index]
        out = 0.0
        for k, (index, weight) in enumerate(zip(self.memberships, self.weights)):
            out = out + (weight if not self.covariates else weight * self.covariates[k]) * v[index]
        return out


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
    # (``sigma ~ (1 |p| g)``), keyed (factor, coefficient) as ``Design.group_blocks``; empty for every other formula. G06: also
    # the formula's own terms (``sigma ~ z + (1 | g)``, brms's group of the parameter) and a shared ID's varying slopes
    group_blocks: Mapping[tuple[str, str], GroupBlock] = field(default_factory=lambda: MappingProxyType({}))
    # G06: the formula's offset term's values on the fitted rows (brms's ``offsets_<dpar>``), added to its linear predictor
    # with no coefficient; None without one
    offset: np.ndarray | None = None
    # G06: whether the intercept the graph samples and a fit's posterior holds is the centred one (brms's ``Intercept_<dpar>``)
    # rather than Bambi's raw-scale one: True where the family's likelihood centres the formula (``DistributionalFormula
    # .centred_by_likelihood``, the cumulative disc), whose posterior Bambi does not uncentre
    centred_draws: bool = False
    # G10: the formula's special products (special_terms.SpecialBlock, by term name; docs/special-term-ext.md)
    special: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


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
    levels: tuple = ()                 # G10: me(x, sdx, gr = g): the sorted levels of g, whose values Xn and noise hold, and
    index: np.ndarray | None = None    # each fitted row's level, 0-based (brms's Jme_k - 1); () and None without gr


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
    """What the model was actually fit to. All arrays are dense NumPy; the group-level design ``Z`` is built on access from
    the blocks' level codes (G11, docs/scaling.md)."""

    y: np.ndarray                 # (n,)
    X_raw: np.ndarray             # (n, P) design columns as given (numeric values, 0/1 dummies), no intercept
    X_model: np.ndarray           # (n, P) what multiplies the coefficients: centred if center_predictors
    x_means: np.ndarray           # (P,) column means of X_raw (zeros if not centred); a dummy's mean is its level's proportion
    predictors: tuple[str, ...]   # numeric predictor terms
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
    fcor: object | None = None         # fcor.FcorDesign: the fixed residual covariance of an fcor(M) term on the fitted rows (F03a), or None
    car: object | None = None          # car.CarDesign: a CAR term's locations and graph on the fitted rows (F02a), or None
    timecor: object | None = None      # timecor.TimeCorDesign: the groups of a residual correlation term on the fitted rows (F03b), or None
    measurement_error: Mapping[str, MeasurementBlock] = field(default_factory=lambda: MappingProxyType({}))   # each me() term's block, by noisy column (C06)
    missing: Mapping[str, MissingBlock] = field(default_factory=lambda: MappingProxyType({}))   # each column whose missing values are parameters (C06)
    category_specific: Mapping[str, np.ndarray] = field(default_factory=lambda: MappingProxyType({}))   # each cs() term's column on the fitted rows (D04; brms's Xcs), by column
    dec: np.ndarray | None = None      # (n,) brms ``dec`` of a ``dec(col)`` term: each wiener row's decision, 1 upper, 0 lower (F04), or None
    thresholds: object | None = None   # grouped_thresholds.ThresholdGroups: thres(gr = z)'s levels on the fitted rows (F06b), or None
    se: np.ndarray | None = None       # (n,) brms ``se`` of an ``se(col)`` term: each row's known standard error (F06a), or None
    denom: np.ndarray | None = None    # (n,) brms ``denom`` of a ``rate(col)`` term: each row's denominator (F06a), or None
    bhaz: object | None = None         # baseline_hazard.BaselineHazard: a cox model's knots, bases (brms's Zbhaz, Zcbhaz) and events (F05), or None
    group_cov: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))   # G05b: each gr(g, cov = A) term's group_cov.GroupCovariance, by factor
    special: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))     # G10: each special product's special_terms.SpecialBlock, by name

    def __post_init__(self):
        if not self.columns:
            object.__setattr__(self, "columns", tuple(self.predictors))

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def Z(self) -> np.ndarray:
        """The group blocks side by side in formula order, ``(n, sum of the blocks' J)``; ``(n, 0)`` without groups. Built on
        access from the blocks (G11, docs/scaling.md) and refused above ``group_index.DENSE_Z_MAX_ENTRIES`` entries."""
        from autobayes.group_index import check_dense_size

        blocks = list(self.group_blocks.values())
        width = max((b.column_slice.stop for b in blocks), default=0)
        check_dense_size(self.n, width, "Design.Z")
        Z = np.zeros((self.n, width))
        for b in blocks:
            b.fill(Z[:, b.column_slice])
        return Z

    @property
    def group_designs(self) -> dict[str, GroupBlock]:
        """The varying-intercept block of each factor: the ``(factor, "Intercept")`` entries of ``group_blocks``."""
        return {factor: block for (factor, coef), block in self.group_blocks.items() if coef == "Intercept"}

    @property
    def factor_blocks(self) -> dict[str, GroupBlock]:
        """G05a: the first block of each factor, in formula order: its varying intercept's, or a slope's for a factor whose
        intercept does not vary (``(0 + x | g)``). Every block of a factor has the same levels and group index."""
        out: dict[str, GroupBlock] = {}
        for (factor, _), block in self.group_blocks.items():
            out.setdefault(factor, block)
        return out

    def _single_group(self):
        if len(self.factor_blocks) > 1:
            raise ValueError("singular group design access is ambiguous with multiple grouping factors; use group_designs")
        return next(iter(self.factor_blocks.values()), None)

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


def _has_dpar_formula(formula, name: str) -> bool:
    """G06: whether a list of formulas has a distributional formula of ``name`` (``"nu ~ z"``), read before the formula is
    parsed (``Model`` checks its keyword arguments first)."""
    if isinstance(formula, str) or not isinstance(formula, (list, tuple)):
        return False
    return any(isinstance(f, str) and f.split("~", 1)[0].strip() == name for f in formula[1:])


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
            value = (t.transform.row_expression() if t.basis is not None          # G04: x_i * w_i, P_1(w_i), 1[ord_i = 2]
                     else t.transform.expression(f"{t.transform.column}_i"))
            parts.append(f"({value} - mean({term})) * {b}_{term}" if centered else f"{value} * {b}_{term}")
        elif t.is_monotonic:                                 # B03: never centred, brms's bsp * mo(simo, Xmo)
            parts.append(f"{b}_{term} * {term}_i")
        elif t.kind == "measurement_error":                  # C06: never centred, brms's bsp * Xme
            parts.append(f"{b}_{term} * Xme_{t.operands[0]},i")
        elif t.kind == "missing":                            # C06: never centred, brms's bsp * Yl (observed or Ymi)
            parts.append(f"{b}_{term} * {t.operands[0]}_i")
        elif t.kind == "category_specific":                  # D04: one coefficient per threshold, in mu_i[k] (the family's lines)
            continue
        elif t.kind == "special_product":                    # G10: brms's bsp * prod(parts) * Csp, never centred
            from autobayes.special_terms import row_expression
            parts.append(f"{b}_{term} * {row_expression(t.special, '' if b == 'b' else b[2:] + '_')}")
        elif t.kind == "contrast":                           # F08: a polynomial contrast of an ordered factor, at row i's level
            value = f"{term}[{t.factor}_i]"
            parts.append(f"({value} - mean({term})) * {b}_{term}" if centered else f"{value} * {b}_{term}")
        elif t.kind == "product":                            # G03: an interaction's column coded by R's rule
            value = " * ".join(f"x_{v}" if lvl is None else f"1[{v}_i = {lvl}]"
                               for v, lvl in zip(t.product.operands, t.product.cell or (None,) * len(t.product.operands)))
            parts.append(f"({value} - q_{term}) * {b}_{term}" if centered else f"{value} * {b}_{term}")
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
                               "negbinomial2", and (E02) "inverse_gaussian", "von_mises", "asym_laplace", "frechet", "gen_extreme_value",
                               and (F04) "wiener", and (F01b) "dirichlet", "logistic_normal", and (F05) "cox"
    categories : tuple[str]    the response's categories in order (the cumulative and categorical families, B05; F01b: a cbind()
                               response's columns, the reference first); () otherwise
    response_matrix : ndarray  F01b: a cbind() response on the simplex, (n, K) in the columns' order (brms's Y); None otherwise
    category_order : ndarray   F01b: the column of each category of ``categories`` in that matrix; None otherwise
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

    def __new__(cls, *args, **kwargs):
        """``Model(..., family=...)`` of a count-matrix family (F01a, ``FamilySpec.matrix_response``: ``multinomial``,
        ``dirichlet_multinomial``) is ``autobayes.multinomial.MultinomialModel``, which builds its own graph from the same
        arguments; every other family is this class."""
        if cls is Model and getattr(FAMILIES.get(kwargs.get("family")), "matrix_response", False):
            from autobayes.multinomial import MultinomialModel

            return super().__new__(MultinomialModel)
        return super().__new__(cls)

    def __init__(self, formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
                 family: str = "gaussian", center_predictors: bool = True,
                 categorical: Mapping[str, object] | None = None, nu: float | None = None, link: str | None = None,
                 families: list[str] | None = None, order: str | bool | None = None, imputed: tuple[str, ...] = (),
                 quantile: float | None = None, data2: Mapping[str, object] | None = None, refcat: str | None = None,
                 dropna: bool = False):
        """``data2`` (Tier F) holds the matrices the formula's terms read by name, brms's ``data2``: ``sar(M, ...)``'s weight
        matrix (F02b) and ``fcor(M)``'s covariance (F03a, docs/fcor.md) over the fitted rows, ``data2={"M": matrix}`` (a NumPy array
        or a DataFrame, in data order), and ``car(W, gr = site)``'s adjacency matrix (F02a, docs/spatial-car.md), a DataFrame labelled
        by location; an entry no term reads is refused (``formula.check_data2``). ``imputed`` (C06) is set by
        ``MultivariateModel`` for the models of its responses: the responses of its
        ``col | mi() ~ ...`` formulas, whose missing values are parameters (``formula.check_imputation``); empty otherwise.
        ``quantile`` (E02) fixes the asymmetric Laplace family's quantile (brms's ``bf(y ~ ..., quantile = p)``): the model's
        spec is then the family's with that parameter fixed (``FamilySpec.with_fixed``) and ``self.quantile`` its value.
        ``refcat`` (F01b) is the reference category of a ``cbind(y1, ..., yK)`` response on the simplex (brms's ``refcat``): one
        of its columns, the first by default (``families.simplex.categories_of``); refused for every other family.
        ``dropna=True`` (D174, docs/missing-rows.md) drops the rows with a missing value in a column the model reads, warns, and
        records their positions in the data as ``dropped_na_rows``; by default such data are refused. A pandas Categorical or
        boolean predictor needs no ``categorical=`` entry (D172, docs/column-types.md)."""
        import bambi as bmb
        from autobayes.column_types import typed_categorical
        from autobayes.missing_rows import check_dropna

        self.dropna = check_dropna(dropna)                    # D174: the opt-in to drop rows with missing values
        self.dropped_na_rows: tuple[int, ...] = ()           # their positions in the data as given
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
            if nu is not None and _has_dpar_formula(formula, "nu"):   # G06: 'nu ~ z' gives nu a formula; nu= would fix it
                raise ValueError(f"nu={nu!r} fixes the degrees of freedom and the formula 'nu ~ ...' gives them a linear predictor: "
                                 "pass one or the other (nu=None with Priors(nu={'intercept': ..., 'b': ...}) for the formula; "
                                 "docs/distributional-groups.md)")
            if nu is not None and priors.nu is not None:
                raise ValueError(f"nu={nu!r} fixes the degrees of freedom and Priors(nu={priors.nu}) learns them: pass one "
                                 "or the other (nu=None with the prior learns nu, as brms's student() does)")
            if nu is None and "nu" in fam.learnable_parameters:   # learned under its own prior (A13), or its formula's (G06)
                if priors.nu is None and not _has_dpar_formula(formula, "nu"):
                    raise PriorError(f"the {family} family needs nu: a fixed nu > {fam.learnable_parameters['nu']:g} (nu=4), or "
                                     "nu=None with a positive-support prior Priors(nu=...) to learn it")
                self.nu = None
            else:
                self.nu = fam.fixed_parameters["nu"](nu)
        elif nu is not None:
            accepting = [name for name, other in FAMILIES.items() if "nu" in other.fixed_parameters]
            raise ValueError(f"nu applies only to the {' or '.join(accepting)} family")
        from autobayes.families.simplex import check_refcat
        check_refcat(fam, refcat)                              # F01b: the simplex families' reference category
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

        # D172: an undeclared pandas Categorical or boolean predictor is declared by its dtype
        categorical = typed_categorical(data, categorical, lambda cat: parse_formula(formula, None, family, cat))
        spec0 = parse_formula(formula, None, family, categorical)
        if fam is not get_family(family) and fam.check_formula is not None:   # a link's own refusals (E06, divergence D133)
            fam.check_formula(spec0)
        self.imputed: tuple[str, ...] = tuple(imputed)        # C06: the columns a multivariate model imputes by their own formula
        check_imputation(spec0, self.imputed)
        if spec0.addition("se") is not None:                 # F06a: se(col) without sigma = TRUE fixes sigma at 0 (no sigma)
            from autobayes.se_rate import check_priors, family_for

            check_priors(spec0, priors)
            fam = self.family_spec = family_for(spec0, fam)
        self.effective_center_predictors = self.center_predictors and spec0.intercept
        if spec0.addition("thres") is not None:              # F06b: grouped thresholds (docs/grouped-thresholds.md): the family's
            fam = self.family_spec = fam.grouped_thresholds()   # spec with them; brms does not centre the design then
            self.effective_center_predictors = False
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
        cols = ([*(spec0.response_columns or (spec0.response,)), *spec0.pop_columns] + operands   # F01b: cbind()'s columns
                + ([spec0.trials] if spec0.trials else []) + (list(spec0.offset.columns) if spec0.offset is not None else [])
                + [c for a in spec0.additions for c in a.columns]          # weights() and cens() columns (B02)
                + [c for d in spec0.dpars for c in d.pop_columns]          # the columns distributional formulas read (B01)
                + [c for d in spec0.dpars for t in d.group_terms for c in t.data_columns]   # and their shared-ID terms (D08)
                + [d.offset.column for d in spec0.dpars if d.offset is not None])   # G06: and their offsets
        cols += list(spec0.autocor.columns) if spec0.autocor is not None else []   # an autocorrelation term's time and series (C04)
        cols += list(spec0.car.columns) if spec0.car is not None else []   # a CAR term's locations (F02a)
        cols += list(spec0.timecor.columns) if spec0.timecor is not None else []   # a residual correlation term's (F03b)
        cols = list(dict.fromkeys(cols))                        # the offset column may also be a predictor
        data = with_trials(spec0, data)                         # F08: trials(15)'s number, under its name
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise FormulaError(f"columns not in data: {missing}")
        df = data[cols].copy().reset_index(drop=True)
        positions = np.arange(len(df))                       # each row's position in the data as given
        if self.dropna:                                      # D174: the rows with a missing value in a column read, but mi()'s
            import warnings

            from autobayes.missing_rows import drop_missing, dropped_warning, row_terms

            df, self.dropped_na_rows, where = drop_missing(df, [c for c in cols if c not in spec0.missing_columns], row_terms(spec0))
            if self.dropped_na_rows:
                positions = np.delete(positions, self.dropped_na_rows)
                self.warnings.append(dropped_warning(self.dropped_na_rows, len(data), where))
                warnings.warn(self.warnings[-1], UserWarning, stacklevel=3)
        if fam.has_trials:
            k, n = binomial_counts(spec0, df)               # validated before any cast, all rows
            zero = np.flatnonzero(n == 0)
            self.dropped_rows = tuple(int(i) for i in positions[zero])
            df[spec0.response] = k
            df[spec0.trials] = n
            if zero.size:
                for cat in spec0.categoricals:              # levels are those of the fitted rows
                    labels, _ = categorical_labels(f"categorical predictor {cat.name!r}", df[cat.name], logical=True)
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
        for t in (t for d in self.spec.dpars for t in d.group_terms):   # G06: a distributional formula's grouping factors, as text
            df[t.column] = group_labels(t.factor, df)
        for t in (*self.spec.transform_terms, *(t for d in self.spec.dpars for t in d.transform_terms)):
            df[t.name] = t.transform.values(df)              # the evaluated column Bambi is handed, under the term's own name (A05)
        if self.spec.offset is not None and self.spec.offset.parts:   # G04: the summed offsets, Bambi's offset(`<expression>`)
            df[self.spec.offset.expression] = self.spec.offset.values(df)
        self.categories: tuple[str, ...] = ()               # the response's categories, in order (B05: cumulative, categorical)
        # F01b: a cbind(y1, ..., yK) response on the simplex, (n, K) in the columns' order (brms's Y), its categories the reference
        # first (self.categories) and each category's column (category_order); Bambi is handed the reference's column under the
        # synthetic response name, which the likelihood does not read
        self.response_matrix: np.ndarray | None = None
        self.category_order: np.ndarray | None = None
        if fam.simplex is not None:
            from autobayes.families.simplex import categories_of, response_matrix
            self.response_matrix = response_matrix(self.spec, df, fam.name)
            self.categories, self.category_order = categories_of(self.spec, refcat, fam)
            df[self.spec.response] = self.response_matrix[:, self.category_order[0]]
        self.hurdle_category: str | None = None              # E01: hurdle_cumulative's hurdle category, brms's Y = 0
        if fam.hurdle_category is not None:                  # read before the column becomes brms's codes
            self.hurdle_category = fam.hurdle_category(self.spec, df[self.spec.response])
        if fam.response_categories is not None:              # the response is fitted as its 1-based category, brms's Y
            self.categories, codes, notes = fam.response_categories(self.spec, df[self.spec.response])
            df[self.spec.response] = codes
            self.warnings.extend(notes)
            cat_term = self.spec.addition("cat")             # F08: brms's cat(K), taken where K is the data's number of categories
            if cat_term is not None and cat_term.count != len(self.categories):
                raise FormulaError(
                    f"{self.spec.formula!r}: {cat_term.term!r}: the response has {len(self.categories)} categories here "
                    f"({', '.join(self.categories)}); cat() with another number (brms then has thresholds no row reaches, or refuses) "
                    "is not offered (docs/syntax-fixes.md, divergence D169)")
        if fam.response_dtype is not None:
            df[self.spec.response] = df[self.spec.response].astype(fam.response_dtype)
        self.warnings.extend(fam.response_warnings(self.spec, df))   # separation, all-zero counts
        self.original_labels: dict[str, np.ndarray] = {}     # labels of the SUPPLIED column, fitted rows, before any recoding
        for cat in self.spec.all_categoricals:               # every formula's, each column once (B01)
            # An ORDERED Categorical with the reference first: formulae keeps this order (it sorts
            # anything else) and takes the first category as the treatment reference; built from
            # the observed labels only, so an unused declared category never becomes a zero column.
            labels, _ = categorical_labels(f"categorical predictor {cat.name!r}", df[cat.name], logical=True)
            self.original_labels[cat.name] = labels
            df[cat.name] = pd.Categorical(labels, categories=list(cat.categories), ordered=True)
            if cat.unused:
                self.warnings.append(
                    f"declared categories {list(cat.unused)} of {cat.name!r} do not occur in the fitted rows: "
                    "they get no coefficient, and predictions for them are refused."
                )
        for t in (t for s in (self.spec, *self.spec.dpars) for t in s.pop_terms if t.kind in ("contrast", "product")):
            df[t.name] = pop_term_design(t, df, self.spec.all_categoricals)[:, 0]   # F08, G03: the column Bambi is handed
        for c in derived_group_columns(self.spec):           # G05a: a product or factor-level slope's covariate, which Bambi reads
            df[c] = group_covariate(self.spec, c, df)        # from the fitted frame under its own name
        self.data = df
        self._threshold_groups = None                        # F06b: thres(gr = z)'s levels on the fitted rows
        if self.spec.addition("thres") is not None:
            from autobayes.grouped_thresholds import ThresholdGroups
            self._threshold_groups = ThresholdGroups.of(self.spec.addition("thres"), self.data)
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
        from autobayes.fcor import fcor_design
        from autobayes.sar import sar_design

        self._sar_design = sar_design(self.spec.sar, data2, len(self.data)) if self.spec.sar is not None else None
        self._fcor_design = fcor_design(self.spec.fcor, data2, len(self.data)) if self.spec.fcor is not None else None   # F03a
        check_data2(self.spec, data2)                         # Tier F: data2 holds what sar(), fcor() and car() read, nothing else
        self._group_cov = {}                                  # G05b: each gr(g, cov = A) term's known covariance on its levels
        for t in self.spec.group_terms:
            if t.cov is not None:
                from autobayes.group_cov import check_dropped_levels, known_covariance
                if self.dropped_na_rows:                      # D174: the levels A covers are those of the rows given
                    check_dropped_levels(t, data[t.factor], self.spec.levels_by_group[t.factor])
                self._group_cov[t.factor], notes = known_covariance(t, self.spec.levels_by_group[t.factor], data2)
                self.warnings.extend(notes)

        self._car_design = None                              # F02a: a CAR term's locations and graph on the fitted rows
        if self.spec.car is not None:
            from autobayes.car import car_layout
            self._car_design, notes = car_layout(self.spec.car, self.data, data2)
            self.warnings.extend(notes)
        self._timecor_design = None                          # F03b: the groups of the fitted rows, brms's sorted Stan rows
        if self.spec.timecor is not None:
            from autobayes.timecor import time_layout
            self._timecor_design = time_layout(self.spec.timecor, self.data)
        self._bhaz_design = None                             # F05: a cox model's knots, bases and events on the fitted rows
        if fam.baseline_hazard:
            from autobayes.baseline_hazard import baseline_design
            self._bhaz_design = baseline_design(self.spec, self.data)

        bambi_family = fam.bambi_family()
        if any(a.kind not in ("mi", "dec", "thres", "cat") and a.kind not in fam.own_additions   # brms's response addition terms act on the
               for a in self.spec.additions):                     # likelihood (B02; F04: the wiener likelihood reads dec() itself; F06b:
                                                                  # thres(); F05: the cox likelihood its cens() and bhaz())
            from autobayes.likelihoods import addition_family
            additions = addition_data(self.spec, self.data)
            bambi_family = addition_family(bambi_family, fam, additions)
            if "denom" in additions:                          # F06a: rate(col), brms's mu + log_denom in the linear predictor
                from autobayes.likelihoods import monotonic_family
                from autobayes.se_rate import log_denominator_hook
                bambi_family = monotonic_family(bambi_family, log_denominator_hook(np.log(additions["denom"])))
        if self._fcor_design is not None:                    # F03a: the rows' joint normal with covariance sigma^2 M
            from autobayes.fcor import fcor_family
            bambi_family = fcor_family(bambi_family, self._fcor_design)
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
            if name not in self.learned_parameters and name not in self.spec.dpar_names:   # one's truncated prior came from
                bambi_priors[name] = np.float64(getattr(self, name))   # Priors.to_bambi (A13), a formula's priors too (G06)
        for extra in fam.extra_parameters:                   # D01: alpha, beta, ndt; ndt truncated to (0, min y) of these rows;
            if extra.name not in self.spec.dpar_names:       # E03: an alpha with its own formula has its formula's priors there
                bambi_priors[extra.variable] = extra.bambi_prior(getattr(priors, extra.name), self.data[self.spec.response])   # E02: tmp_xi
        # G11 (docs/scaling.md): Bambi is handed the population-level formula; each group-level coefficient it builds is added to
        # its component from the level codes before build() (_add_index_group_terms), in the order formulae would give them
        bambi_formula = self.spec.bambi_population_formula
        # Bambi centres every formula that has an intercept, or none. Without a distributional formula that is the
        # response formula's centring as it has always been; with one (B01), brms centres each formula that has an
        # intercept on its own means, and the refusal above leaves exactly that case. A family that builds what Bambi is
        # handed from the model (B05: cumulative, categorical; neither takes a distributional formula) decides it itself.
        self._bambi_centring = self.effective_center_predictors
        if self.spec.dpars:                                  # brms's bf(y ~ ..., sigma ~ ...): Bambi's distributional components (B01)
            bambi_formula = bmb.Formula(bambi_formula, *(f"{self._bambi_dpar(d.response)} ~ {d.bambi_population_rhs}"
                                                         for d in self.spec.dpars))
            self._bambi_centring = self.center_predictors
        if fam.bambi_setup is not None:                      # B05: the family builds what Bambi is handed from the model
            bambi_family, bambi_priors, self._bambi_centring = fam.bambi_setup(self, bambi_priors)
        if self.spec.sp_terms:                               # G10: brms's special products, added after every hook below (the
            from autobayes.likelihoods import monotonic_family   # latent values of me() among them)
            bambi_family = monotonic_family(bambi_family, self._add_special_products)
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
        if self.spec.car is not None:                        # brms's car() (F02a): added the same way
            from autobayes.likelihoods import monotonic_family
            bambi_family = monotonic_family(bambi_family, self._add_car_effects)
        if self.spec.autocor is not None:                    # brms's ar(), ma(), arma() (C04): added last, on the residuals
            from autobayes.autocor import autocor_family
            bambi_family = autocor_family(bambi_family, self._add_autocor_effects)
        if self.spec.sar is not None:                        # brms's sar(M, type) (F02b): the joint SAR likelihood
            from autobayes.sar import sar_family
            bambi_family = sar_family(bambi_family, self._sar_design)
        if self.spec.timecor is not None:                    # brms's unstr(), cosy() (F03b): each group's rows jointly
            from autobayes.timecor import timecor_family
            bambi_family = timecor_family(bambi_family, self)
        from autobayes.likelihoods import ordered_family     # F09: the auxiliary parameters' variables in a fixed order
        bambi_family = ordered_family(bambi_family)
        for d in self.spec.dpars:                            # G06: brms's logm1 link of the Student-t nu, which Bambi does not have
            if fam.distributional_parameter_link(d.response) == "logm1":
                from autobayes.likelihoods import logm1_link
                bambi_family.link = {**bambi_family.link, self._bambi_dpar(d.response): logm1_link()}
            if d.sp_terms:                                   # G10: a distributional formula's special products, before its link
                from autobayes.special_terms import check_dpar_link, special_link
                link = fam.distributional_parameter_link(d.response)
                check_dpar_link(d.response, link, d)
                bambi_family.link = {**bambi_family.link, self._bambi_dpar(d.response): special_link(self, d)}
        self.bambi_model = bmb.Model(
            bambi_formula, bambi_data, family=bambi_family,
            priors=bambi_priors,
            noncentered=True, center_predictors=self._bambi_centring,
        )
        self._add_index_group_terms(bambi_priors)
        self.bambi_model.build()
        self.pymc_model = self.bambi_model.backend.model
        if any(not t.custom for t in self.spec.correlated_terms) or self.spec.shared_ids:   # brms's correlated slopes (A02) and shared IDs (D08): a post-build
            self.pymc_model = self._correlate_group_effects(self.pymc_model)   # edit of Bambi's graph
            self.bambi_model.backend.model = self.pymc_model  # what Bambi's fit() samples and cleans
        self.design = self._extract_design()
        from autobayes.term_algebra import product_notes    # G03: a rank-deficient design of a formula with R's interactions
        self.warnings.extend(product_notes(self.spec, self.design.X_raw, self.spec.intercept))
        for ds in self.spec.dpars:
            self.warnings.extend(product_notes(ds, self.design.dpars[ds.response].X_raw, ds.intercept, f" of {ds.response}"))
        if self.shrinkage_names:                             # D07: b = zb * sdb under horseshoe() / R2D2(), checked
            from autobayes.shrinkage import check_graph
            check_graph(self)
        if self.spec.me_terms or self.spec.mi_terms:         # C06: each term's contribution to the linear predictor, checked
            self._check_measurement_effects()
        if self.spec.sp_terms or any(d.sp_terms for d in self.spec.dpars):   # G10: each special product's, read off the graph
            self._special_design()
        if fam.mixture:                                      # C05: the graph built beside Bambi's components, checked
            from autobayes.families.mixture import check_graph
            check_graph(self)
        if self.spec.autocor is not None:                    # C04: the graph's ARMA terms against brms's recursion
            self._check_autocor_effects()
        if self._fcor_design is not None:                    # F03a: the graph's likelihood against the dense multivariate normal
            self._check_fcor_likelihood()
        if self.spec.car is not None:                        # F02a: the graph's CAR term against car.car_values
            from autobayes.car import check_graph
            check_graph(self)
        if self._bhaz_design is not None:                    # F05: the graph's sbhaz, a free simplex on Stan's coordinates
            from autobayes.baseline_hazard import check_graph
            check_graph(self)
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
        variable_names += [v for v in (self._bambi_dpar(d.response) for d in self.spec.dpars) if v not in variable_names]   # G06: disc
        if self.family_spec.thresholds:                     # B05: the raw thresholds' deterministic and their coordinate
            variable_names += [self.names["thresholds"], self.names["threshold_dim"]]
        if self.spec.has_category_specific:                 # D04: Bambi's parent mu beside the thresholds' predictor mu_cs
            variable_names += [self.family_spec.parent]
        if self.family_spec.per_category:                   # B05: the category coordinates Bambi adds
            variable_names += [self.names["category_dim"], f"{self.spec.response}_dim"]
        if self.family_spec.mixture:                        # C05: each component's sigma, theta and the component coordinates
            roles = self.mixture_names
            variable_names += [*roles["sigma"], roles["theta"], roles["coordinate"], f"{self.spec.response}_dim"]
        simplex = self.simplex_names                        # F01b: the observed categories' coordinate, sigma<k>, Llncor
        variable_names += [simplex[k] for k in ("response_dim", "L", "coordinate", "coordinate_T") if k in simplex]
        variable_names += list(simplex.get("sigma", ()))
        for roles in (*self.group_coefficient_names.values(), *self.dpar_group_names.values()):   # D08: a shared ID's dpar terms too
            variable_names.extend(roles[k] for k in ("sd", "offset", "group_effect", "by_normalisation") if k in roles)
        variable_names += [roles[k] for roles in self.group_dist_names.values() for k in ("df", "udf", "dfm")]   # G05b
        for roles in self.correlation_names.values():
            variable_names.extend(roles[k] for k in ("L", "cor"))
        dpar_coefficients = [v for roles in self.dpar_names.values() for k, v in roles.items() if k != "value"]   # B01
        variable_names += dpar_coefficients
        variable_names += [roles["simplex"] for roles in self.mo_names.values()]   # each monotonic term's simplex (B03)
        variable_names += list(self.sp_names["simplex"].values()) if self.spec.sp_terms else []   # G10: the special products'
        variable_names += [roles[k] for roles in self.gp_names.values() for k in ("sdgp", "lscale", "zgp")]   # each GP's (C03)
        variable_names += [v for k, v in self.autocor_names.items() if not k.endswith("_coordinate")]   # ar and ma (C04)
        variable_names += list(self.sar_names.values())      # lagsar or errorsar (F02b)
        variable_names += [v for k, v in self.car_names.items() if not k.endswith("coordinate")]   # a CAR term's (F02a)
        variable_names += [v for k, v in self.timecor_names.items() if not k.startswith("coordinate")]   # Lcortime, Cortime, cosy (F03b)
        variable_names += [roles[k] for roles in self.me_names.values() for k in ("mean", "sd", "z", "latent", "noisy")]   # C06
        variable_names += [roles["values"] for roles in self.missing_names.values()]   # C06: the missing values Ymi_<col>
        shrinkage = self.shrinkage_names                    # D07: zb, sdb and the horseshoe's or R2D2's own variables
        variable_names += [v for k, v in shrinkage.items() if k not in ("prior", "coordinate", "lprior")]
        variable_names += ["sbhaz"] if self.family_spec.baseline_hazard else []   # F05: the cox baseline hazard's simplex
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
        coords += list(self.sp_names["coordinate"].values()) if self.spec.sp_terms else []   # G10
        coords += [f"{v}_dim" for roles in self.dpar_names.values() for k, v in roles.items() if k.startswith("simo:")]   # G10
        coords += [roles["coordinate"] for roles in self.gp_names.values()]       # C03
        coords += [v for k, v in self.autocor_names.items() if k.endswith("_coordinate")]   # ar_dim and ma_dim (C04)
        coords += [v for k, v in self.car_names.items() if k.endswith("coordinate")]   # rcar_dim, zcar_dim (F02a)
        coords += [v for k, v in self.timecor_names.items() if k.startswith("coordinate")]   # cortime_dim, cortime_dim_T (F03b)
        coords += [roles["coordinate"] for roles in self.missing_names.values()]   # C06: the missing rows of each column
        coords += [shrinkage["coordinate"]] if shrinkage else []                    # D07: zb_dim
        coords += ["sbhaz_dim"] if self.family_spec.baseline_hazard else []          # F05: its coordinate
        coords += ["__obs__"]
        names = variable_names + coords
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise FormulaError(f"generated variable/coordinate name collision: {duplicates}; rename the conflicting column")

    # -- correlated group effects (A02) -------------------------------------------------
    def _add_index_group_terms(self, bambi_priors: dict) -> None:
        """G11 (docs/scaling.md): add every group-level coefficient Bambi builds to its Bambi component before ``build()``, from
        level codes: one ``group_index.IndexGroupTerm`` per coefficient of each term Bambi builds (the response formula's
        ``bambi_group_terms`` and each distributional formula's group-level terms), in formulae's order (term by term, each
        term's coefficients in order), holding the factor's sorted fitted levels, each row's level code, the column formulae
        would read from Bambi's frame (int64 ones for a varying intercept) and the prior Bambi would have given it. Bambi's
        backend then builds the graph it builds from formulae's dense design; ``_extract_design`` checks every term."""
        from autobayes.group_index import bambi_formula_value, bambi_prior, index_group_term, level_codes

        bm = self.bambi_model
        parent = bm.family.likelihood.parent
        owners = [(bm.components[self.family_spec.parent], bambi_priors[parent] if parent in bambi_priors else bambi_priors,
                   self.spec.levels_by_group, [(t, t.bambi_coefs) for t in self.spec.bambi_group_terms])]
        for d in self.spec.dpars:                            # G06, D08: a distributional formula's own and shared-ID terms
            if d.group_terms:
                name = self._bambi_dpar(d.response)
                owners.append((bm.components[name], bambi_priors.get(name, {}), d.levels_by_group, [(t, t.coefs) for t in d.group_terms]))
        codes: dict[str, np.ndarray] = {}                    # one code array per factor, shared by its coefficients
        for comp, priors, levels_by_group, terms in owners:
            for t, coefs in terms:
                levels = levels_by_group[t.factor]
                if t.column not in codes:
                    codes[t.column] = level_codes(self.data[t.column], levels)
                for coef in coefs:
                    label = _bambi_group_name(t.factor, coef)
                    comp.terms[label] = index_group_term(
                        label, "intercept" if coef == "Intercept" else "numeric", levels, codes[t.column],
                        bambi_formula_value(bm.data, None if coef == "Intercept" else coef), bambi_prior(priors, label), comp.prefix)

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
        groups = [(t.factor, t.coefs, [self._group_roles(t.factor, c) for c in t.coefs]) for t in self.spec.correlated_terms
                  if not t.custom]                       # G05b: a term the model builds itself is correlated as it is built
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
        offset_terms = [self.spec.offset.bambi_name] if self.spec.offset is not None else []
        if names != (["Intercept"] if self.spec.intercept else []) + [t.name for t in self.spec.bambi_terms] + offset_terms:
            raise RuntimeError(f"unexpected common terms from Bambi: {names}")
        Xfull = np.asarray(common.design_matrix, dtype=float) if common is not None else np.zeros((len(self.data), 0))
        # The offset term (brms offset(), formulae's kind "offset"): Bambi adds its column to the linear predictor as
        # given, outside the coefficient design and never centred. Its values are recomputed here from the data and
        # must equal what Bambi evaluated; the column is then removed from the design that the coefficients multiply.
        offset = None
        if self.spec.offset is not None:
            term = self.spec.offset.bambi_name
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
        # One block per (factor, coefficient), in formula order; Design.Z is their side-by-side concatenation. The varying
        # intercept's block is the one-hot indicator of the level; a varying slope's is that indicator times the slope's (raw,
        # never centred) column, which is what brms's Z_m_c spread over J_m is. G11 (docs/scaling.md): a block keeps the level
        # codes and the covariate, and Bambi builds each coefficient from the IndexGroupTerm _add_index_group_terms gave it
        # (Bambi builds '(1 + x || g)' as the independent terms '1|g' and 'x|g'); each is checked here against the data, so a
        # different coding is a hard error rather than a silently different model.
        group_blocks: dict[tuple[str, str], GroupBlock] = {}
        bterms = comp.group_specific_terms
        if group is not None:
            raise RuntimeError("Bambi built a group design from the formula; the group-level terms are added from level codes")
        if not self.spec.has_group:
            if bterms:
                raise RuntimeError("Bambi built a group design for a formula without a group-level term")
        else:
            # D08: the multi-membership and by terms are the model's own (_custom_group_blocks); Bambi builds the others
            expected_terms = [_bambi_group_name(t.factor, coef) for t in self.spec.bambi_group_terms for coef in t.bambi_coefs]
            if list(bterms) != expected_terms:
                raise RuntimeError("Bambi group terms differ from formula order")
            custom = self._custom_group_blocks() if self.spec.custom_group_terms else {}
            start, n = 0, len(X_raw)
            for t in self.spec.group_terms:
                g = t.factor
                levels = self.spec.levels_by_group[g]
                if t.custom:                                 # D08: the block built and checked against the graph here (G05b:
                    for coef in t.coefs:                     # one per coefficient)
                        block = custom[(g, coef)]
                        group_blocks[(g, coef)] = replace(block, column_slice=slice(start, start + len(levels)))
                        start += len(levels)
                    continue
                codes = level_codes(self.data[t.column], levels)
                for coef in t.coefs:
                    if coef in t.monotonic_slopes:           # G05a: brms's r[J] * mo(simo, Xmo), built by the model, no Z column
                        group_blocks[(g, coef)] = GroupBlock(levels, None, codes, slice(start, start + len(levels)),
                                                             monotonic=self.spec.pop_term(coef).operands[0])
                        start += len(levels)
                        continue
                    name = _bambi_group_name(g, coef)
                    bterm = bterms[name]
                    if tuple(str(v) for v in bterm.groups) != levels:
                        raise RuntimeError(f"Bambi level order for {g!r} differs from sorted fitted labels")
                    if tuple(bterm.shape) != (n, len(levels)) or not np.array_equal(bterm.group_index, codes):
                        raise RuntimeError(f"group design for {g!r} does not encode original rows one-hot")
                    if coef == "Intercept":
                        covariate = None
                        if bterm.kind != "intercept" or not np.array_equal(bterm.predictor, np.ones(n)):
                            raise RuntimeError(f"group design for {g!r} does not encode original rows one-hot")
                    else:
                        if bterm.kind != "numeric":
                            raise RuntimeError(f"Bambi coded the varying slope {name!r} as {bterm.kind!r}; expected a numeric slope")
                        covariate = np.asarray(group_covariate(self.spec, coef, self.data), dtype=float)   # G05a: a transform's,
                        if not np.array_equal(np.asarray(bterm.predictor, dtype=float), covariate):           # product's or level's
                            raise RuntimeError(
                                f"Bambi's block for {name!r} is not the level indicator of {g!r} times the {coef!r} column"
                            )
                    group_blocks[(g, coef)] = GroupBlock(levels, None, codes, slice(start, start + len(levels)), covariate=covariate)
                    start += len(levels)
                coord = self.group_names[g]["coordinate"]
                if tuple(str(v) for v in self.pymc_model.coords[coord]) != levels:
                    raise RuntimeError(f"PyMC coordinate {coord!r} differs from fitted labels")
                # every coefficient's effects carry the factor's level dimension; a per-category family's (F01b) also the
                # category dimension, its SD that one alone
                category = (self.names["category_dim"],) if self.family_spec.per_category else ()
                for coef in t.coefs:
                    roles = self._group_roles(g, coef)
                    for key in ("offset", "group_effect"):
                        dims = tuple(self.pymc_model.named_vars_to_dims.get(roles[key], ()))
                        if dims != (coord, *category):
                            raise RuntimeError(f"PyMC dims of {roles[key]!r} are {dims}; expected {(coord, *category)}")
                    if category and tuple(self.pymc_model.named_vars_to_dims.get(roles["sd"], ())) != category:
                        raise RuntimeError(f"PyMC dims of {roles['sd']!r} are not {category}: one SD per non-reference category")
            if any(t.monotonic_slopes for t in self.spec.group_terms):   # G05a: each monotonic slope's part of the graph, checked
                self._check_monotonic_slopes(group_blocks)
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
        y = (np.asarray(self.data[self.spec.response], dtype=float) if self.response_matrix is None
             else self.response_matrix.copy())               # F01b: brms's Y, (n, K) in the columns' order
        trials = np.asarray(self.data[self.spec.trials], dtype=np.int64) if self.spec.trials else None
        additions = addition_data(self.spec, self.data)      # brms's weights, cens, lb and ub on the fitted rows (B02)
        return Design(y=y, X_raw=X_raw, X_model=X_model, x_means=x_means,
                      predictors=self.spec.predictors, group_blocks=group_blocks,
                      centered=self.effective_center_predictors, trials=trials,
                      columns=self.spec.columns, categoricals=self.spec.categoricals, offset=offset,
                      weights=additions.get("weights"), cens=additions.get("cens"), lb=additions.get("lb"), ub=additions.get("ub"),
                      rcens=additions.get("rcens"),                                         # E04: cens(col, y2)'s upper bounds
                      dec=additions.get("dec"),                                             # F04: the wiener decisions
                      se=additions.get("se"), denom=additions.get("denom"),                 # F06a: se() and rate()
                      dpars=MappingProxyType({d.response: self._dpar_design(d) for d in self.spec.dpars}),
                      monotonic=MappingProxyType(self._monotonic_design()), gp=MappingProxyType(self._gp_design()),
                      autocor=self._autocor_design, sar=self._sar_design, fcor=self._fcor_design,   # F02b, F03a
                      car=self._car_design,                                                          # F02a
                      timecor=self._timecor_design,                                               # F03b
                      measurement_error=MappingProxyType(self._measurement_design()),   # C06
                      missing=MappingProxyType(self._missing_design()),
                      category_specific=MappingProxyType(self._category_specific_design()),   # D04
                      thresholds=self._threshold_groups,                                     # F06b
                      bhaz=self._bhaz_design,                                                # F05
                      group_cov=MappingProxyType(dict(self._group_cov)))                     # G05b

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
        group_blocks = self._dpar_group_blocks(d, comp) if d.group_terms else {}   # D08: a shared correlation ID's terms; G06: own
        if (comp.design.group is not None or comp.group_specific_terms) and not d.group_terms:
            raise RuntimeError(f"Bambi built group-level terms for the formula '{d.formula}', which has none")
        names = list(comp.design.common.terms)
        offset_terms = [d.offset.bambi_name] if d.offset is not None else []   # G06: an offset of the formula, Bambi's last common term
        if names != (["Intercept"] if d.intercept else []) + [t.name for t in d.bambi_terms] + offset_terms:   # G10: no special term
            raise RuntimeError(f"unexpected terms from Bambi for the formula '{d.formula}': {names}")
        Xfull = np.asarray(comp.design.common.design_matrix, dtype=float)
        offset = None
        if d.offset is not None:                              # G06: checked against the data, then out of the coefficients' design
            offset, Xfull = self._dpar_offset(d, comp, Xfull)
        X_raw = self._population_design(comp, d, Xfull, prefix=f"{bname}_")
        roles = self.dpar_names[d.response]
        value = self.pymc_model[roles["value"]]
        if value not in self.pymc_model.deterministics or tuple(self.pymc_model.named_vars_to_dims.get(roles["value"], ())) != ("__obs__",):
            raise RuntimeError(f"{roles['value']!r} is not the per-row deterministic of {d.response}'s formula")
        for key, name in roles.items():
            if key != "value" and self.pymc_model[name] not in self.pymc_model.free_RVs:
                raise RuntimeError(f"{name!r} is not a free variable of the graph")
        link = self.family_spec.distributional_parameter_link(d.response)   # log (B01); E03: logit for zi / hu, identity for alpha
        formula = self.family_spec.distributional_formula(d.response)        # G06: logm1 for nu; the cumulative disc's centred log
        by_likelihood = formula is not None and formula.centred_by_likelihood
        expected = f"{link}_centred" if by_likelihood else link
        expected = f"{link}_special" if d.sp_terms else expected   # G10: the link that adds the formula's special products
        if self.bambi_model.family.link[bname].name != expected:
            raise RuntimeError(f"Bambi's link of {bname!r} is {self.bambi_model.family.link[bname].name}, expected {expected}")
        # an intercept-only formula has nothing to centre; G06: a family's likelihood may centre it where Bambi does not
        centered = bool((self._bambi_centring or by_likelihood) and d.intercept and d.pop_terms)
        x_means = X_raw.mean(0) if centered else np.zeros(X_raw.shape[1])
        return DparDesign(name=d.response, X_raw=X_raw, X_model=X_raw - x_means if centered else X_raw, x_means=x_means,
                          columns=d.columns, centered=centered, link=link,
                          group_blocks=MappingProxyType(group_blocks), offset=offset,
                          centred_draws=bool(centered and by_likelihood))

    def _dpar_offset(self, d: ModelSpec, comp, Xfull: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """G06: the offset term of distributional formula ``d`` (brms's ``offsets_<dpar>``) as Bambi built it in the parameter's
        component, checked against its values recomputed from the data, and Bambi's common design without its column."""
        term, common = d.offset.bambi_name, comp.design.common
        if list(comp.offset_terms) != [term] or getattr(common.terms[term], "kind", None) != "offset":
            raise RuntimeError(f"Bambi did not build {term!r} as the single offset term of the formula '{d.formula}' "
                               f"(offset terms: {list(comp.offset_terms)})")
        offset = d.offset.values(self.data)
        built = np.asarray(comp.offset_terms[term].data, dtype=float).ravel()
        if built.shape != offset.shape or not np.array_equal(built, offset) or not np.array_equal(Xfull[:, common.slices[term]].ravel(), offset):
            raise RuntimeError(f"Bambi's offset {term!r} in the formula '{d.formula}' does not equal {d.offset.expression} computed "
                               "from the data")
        return offset, np.delete(Xfull, common.slices[term], axis=1)

    def _dpar_group_blocks(self, d: ModelSpec, comp) -> dict[tuple[str, str], GroupBlock]:
        """The blocks of a distributional formula's group-level terms, as Bambi built its distributional component's group
        design: every coefficient of every term in formula order, the level order the sorted fitted labels (the response
        formula's for a factor both formulas read), each block the one-hot indicator of the row's level (times the slope's raw
        column for a varying slope), and the effects ``<component>_1|g`` (``<component>_x|g``) and offsets on the factor's
        level dimension. D08: terms sharing a correlation ID with the response formula's; G06: the formula's own terms (brms's
        group of the parameter) and a shared ID's varying slopes."""
        bterms = comp.group_specific_terms                    # G11: the IndexGroupTerms _add_index_group_terms gave Bambi
        expected = [_bambi_group_name(t.factor, c) for t in d.group_terms for c in t.coefs]
        if comp.design.group is not None or list(bterms) != expected:
            raise RuntimeError(f"Bambi's group terms of the formula '{d.formula}' are {list(bterms)}; expected {expected}")
        blocks, start = {}, 0
        for t in d.group_terms:
            levels = d.levels_by_group[t.factor]
            if self.spec.levels_by_group.get(t.factor, levels) != levels:
                raise RuntimeError(f"the levels of {t.factor!r} in the formula '{d.formula}' differ from the response formula's")
            codes = level_codes(self.data[t.column], levels)
            for coef in t.coefs:
                name = _bambi_group_name(t.factor, coef)
                bterm = bterms[name]
                if tuple(str(v) for v in bterm.groups) != levels:
                    raise RuntimeError(f"Bambi's level order of {t.factor!r} in the formula '{d.formula}' differs from the sorted fitted labels")
                covariate = None if coef == "Intercept" else self.data[coef].to_numpy(dtype=float)
                block_slice = slice(start, start + len(levels))
                if (not np.array_equal(bterm.group_index, codes)
                        or not np.array_equal(np.asarray(bterm.predictor, dtype=float), np.ones(len(codes)) if covariate is None else covariate)):
                    raise RuntimeError(f"Bambi's group design of {name!r} in the formula '{d.formula}' is not the level indicator of "
                                       f"{t.factor!r}" + ("" if coef == "Intercept" else f" times the {coef!r} column"))
                roles = self._shared_roles(t.factor, d.response, coef)
                for key in ("offset", "group_effect"):
                    if tuple(self.pymc_model.named_vars_to_dims.get(roles[key], ())) != (roles["coordinate"],):
                        raise RuntimeError(f"PyMC dims of {roles[key]!r} are not ({roles['coordinate']!r},)")
                if self.pymc_model[roles["sd"]] not in self.pymc_model.free_RVs:
                    raise RuntimeError(f"{roles['sd']!r} is not a free variable of the graph")
                blocks[(t.factor, coef)] = GroupBlock(levels, None, codes, block_slice, covariate=covariate)
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
            effect = monotonic_effect(simplex, m.codes(self.data[m.column]), m.D)
            eta = eta + coefficient * effect + self._monotonic_slope_effects(t.name, effect)
        return eta

    def _monotonic_slope_effects(self, term: str, effect):
        """G05a: the varying slopes of the monotonic term ``term`` (``(1 + mo(ord) | g)``), built inside the monotonic hook with
        the term's ``mo(simo, Xmo)`` per row (``effect``): brms's ``r_m_c[J_m] * mo(simo, Xmo)``, one simplex shared with the
        population-level term and no Z column. Each is named as Bambi names a varying slope, ``mo(ord)|g_sigma`` under its ``sd``
        prior, ``mo(ord)|g_offset ~ Normal(0, 1)`` and the deterministic ``mo(ord)|g = sd * offset`` on the factor's level
        dimension (Bambi's, from the factor's varying intercept), which ``_correlate_group_effects`` correlates with the
        intercept as for any slope. 0 when no term has one."""
        import pymc as pm

        out = 0.0
        for t in self.spec.group_terms:
            if term not in t.monotonic_slopes:
                continue
            roles = self.group_coefficient_names[(t.factor, term)]
            index = level_codes(self.data[t.column], self.spec.levels_by_group[t.factor])   # G11: each row's level, int64
            sd = _prior_rv(roles["sd"], self.priors.sd_priors(self.spec)[(t.factor, term)], positive=True)
            z = pm.Normal(roles["offset"], mu=np.float64(0.0), sigma=np.float64(1.0), dims=roles["coordinate"])
            r = pm.Deterministic(roles["group_effect"], sd * z, dims=roles["coordinate"])
            out = out + r[index] * effect
        return out

    def _check_monotonic_slopes(self, blocks) -> None:
        """G05a: each monotonic varying slope's contribution, read off the graph: at a simplex with unequal steps, the slope's SD at
        1 and every other variable at the graph's initial point (the standardised effects 0, the correlation factor the
        identity), the parent's linear predictor (on Bambi's link scale: the parent itself, or the log of ``mu`` under a log
        link) must move, row by row, by exactly ``test[J_i] * D * sum(simplex[:Xmo_i])`` (recomputed from the data's codes) when
        the slope's standardised effects go from 0 to a fixed vector ``test``."""
        from autobayes.monotonic import monotonic_values, stick_breaking_free

        pmm, parent = self.pymc_model, self.family_spec.parent
        link = self.bambi_model.family.link[parent].name
        fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[parent]]), inputs=pmm.value_vars, point_fn=True,
                            on_unused_input="ignore", mode="FAST_COMPILE")
        for (g, coef), block in blocks.items():
            if block.monotonic is None:
                continue
            roles, mroles, m = self.group_coefficient_names[(g, coef)], self.mo_names[block.monotonic], self.spec.monotonic(block.monotonic)
            simplex, test = np.arange(1.0, m.D + 1.0) / (m.D * (m.D + 1) / 2.0), np.cos(np.arange(1.0, block.J + 1.0))
            at0 = {**pmm.initial_point(), f"{mroles['simplex']}_stickbreaking__": stick_breaking_free(simplex),
                   pmm.rvs_to_values[pmm[roles["sd"]]].name: np.asarray(0.0)}
            lo, hi = (np.asarray(fn(p)[0], dtype=float).ravel() for p in (at0, {**at0, roles["offset"]: test}))
            moved = hi - lo if link == "identity" else np.log(hi) - np.log(lo)
            expected = test[block.group_index] * monotonic_values(simplex, m.codes(self.data[block.monotonic]))
            if moved.shape != expected.shape or not np.allclose(moved, expected, rtol=1e-10, atol=1e-10):
                raise RuntimeError(f"the graph's {parent!r} does not move by r[{g}] * mo({block.monotonic}) per row with the varying slope "
                                   f"{coef!r} of {g!r}")

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

    # -- special products (G10, docs/special-term-ext.md) ---------------------------------
    def _add_special_products(self, eta):
        """brms's special products of the response formula (``mo(ord):x``, ``mo(ord, id = "a")``, ``me(x, sdx):w``), built inside
        the model Bambi is building (``likelihoods.monotonic_family``'s hook, after the other terms' hooks): ``special_terms.add_effects``."""
        from autobayes.special_terms import add_effects

        return add_effects(self, eta, self.spec, self.priors.coefficient_priors(self.spec), self.priors.special_simo_priors(self.spec),
                           self.data)

    def _special_design(self) -> None:
        """Each special product's block, read off the graph (``special_terms.check_effects``): the response formula's on the
        parent under Bambi's link, a distributional formula's on its parameter's per-row deterministic under the log link."""
        from dataclasses import replace

        from autobayes.special_terms import check_effects

        special = {}
        if self.spec.sp_terms:
            parent = self.family_spec.parent
            special = check_effects(self, self.spec, parent, self.bambi_model.family.link[parent].name, self.data)
        dpars = dict(self.design.dpars)
        for d in self.spec.dpars:
            if d.sp_terms:
                bname = self._bambi_dpar(d.response)
                dpars[d.response] = replace(dpars[d.response], special=MappingProxyType(
                    check_effects(self, d, bname, self.family_spec.distributional_parameter_link(d.response), self.data, f"{bname}_")))
        self.design = replace(self.design, special=MappingProxyType(special), dpars=MappingProxyType(dpars))

    def _measurement_rows(self, x: str):
        """G10: each fitted row's level of ``me(x, sdx, gr = g)`` (0-based, the index of its latent value), or None for an
        ``me()`` term without ``gr`` (one latent value per row)."""
        from autobayes.formula import measurement_error_group

        t = next(t for t in self.spec.me_terms if t.operands[0] == x)
        if measurement_error_group(t.name) is None:
            return None
        from autobayes.special_terms import measurement_levels

        return measurement_levels(t.name, self.data)[1]

    @property
    def sp_names(self) -> dict:
        """G10: PyMC names of the response formula's special products (``special_terms.names``): ``"b"`` by term name, and
        ``"simplex"`` and ``"coordinate"`` by flat simplex key (``simo_<key>``). Empty mappings without one."""
        from autobayes.special_terms import names

        return names(self.spec)

    # -- category-specific effects (D04) -------------------------------------------------
    def _add_category_specific_effects(self, mu):
        """brms's category-specific effects ``cs(x)`` (D04, docs/ordinal-categorical.md), built inside the model Bambi is
        building (the ordinal likelihood calls this with the parent ``mu``, ``likelihoods.ordinal_family``): for each term its
        coefficients, one per threshold (brms's row of ``bcs`` for the column; a free vector named after the term, ``cs(x)``, on
        the thresholds' dimension, under its ``b`` prior, as brms puts ``bcs`` under class ``b``), and the predictor of every
        threshold, ``mu_i + sum_c x_c,i bcs_c[k]`` (brms's ``mu`` and ``Xcs bcs``, the column never centred), registered as the
        deterministic ``names["linpred"]`` (``mu_cs``; dims ``__obs__`` and the thresholds'), which the likelihood reads."""
        import pymc as pm
        import pytensor.tensor as pt

        coefficients = self.priors.coefficient_priors(self.spec)
        dim = self.names["threshold_dim"]
        eta = mu[:, None]
        for factor, (rs, index) in getattr(self, "_threshold_effects", {}).items():   # G10: (cs(1) | g), r_k[J] on threshold k
            eta = eta + pt.stack([r[index] for r in rs], axis=1)
        for t in self.spec.cs_terms:
            prior = coefficients[t.name]
            coef = getattr(pm, prior.name)(t.name, **{k: np.float64(v) for k, v in prior.params.items()}, dims=dim)
            eta = eta + cs_values(t, self.data)[:, None] * coef[None, :]   # G10: a factor's contrast column too
        return pm.Deterministic(self.names["linpred"], eta, dims=("__obs__", dim))

    def _category_specific_design(self) -> dict[str, np.ndarray]:
        """Each category-specific term's column on the fitted rows (brms's ``Xcs``, by column, in term order), checked against
        the graph: its coefficients a free vector over the thresholds' dimension, and the thresholds' predictor ``mu_cs`` moving,
        row by row and threshold by threshold, by exactly ``x_i * c[k]`` from ``mu`` when the coefficients go from 0 to a test
        vector ``c``. Empty without a ``cs()`` term. G10: a ``(cs(1) | g)`` term's per-threshold effects move it too, by
        ``test_k[J]`` on threshold k when the term's SDs are 1, its correlation factor the identity and its standardised effects
        ``test_k``."""
        terms = self.spec.cs_terms
        thresholds = [t for t in self.spec.group_terms if t.thresholds]
        if not terms and not thresholds:
            return {}
        pmm, dim = self.pymc_model, self.names["threshold_dim"]
        K1 = len(self.categories) - 1
        blocks = {cs_label(t): cs_values(t, self.data) for t in terms}   # G10: by brms's Xcs column name (x, zq)
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
        group_part = 0.0
        for t in thresholds:                                 # G10: (cs(1) | g) at unit SDs, the identity and test effects
            index, effects = level_codes(self.data[t.column].astype(str), self.spec.levels_by_group[t.factor]), []
            for k, c in enumerate(t.coefs):
                roles = self.group_coefficient_names[(t.factor, c)]
                point[f"{roles['sd']}_log__"] = np.asarray(0.0)
                point[roles["offset"]] = np.sin(np.arange(1.0, len(self.spec.levels_by_group[t.factor]) + 1.0) * (k + 1.0))
                effects.append(point[roles["offset"]][index])
            value = f"{self.correlation_names[t.factor]['L']}_cholesky_corr__"
            point[value] = np.zeros(np.shape(point[value]))
            group_part = group_part + np.stack(effects, axis=1)
        mu, mu_cs = (np.asarray(v, dtype=float) for v in fn(point))
        expected = mu[:, None] + sum(blocks[cs_label(t)][:, None] * tests[t.name][None, :] for t in terms) + group_part
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
        for a ``by`` term the sorted levels of the by column and the 0-based by-level of each level (brms's ``Jby_m - 1``).
        G05b: the same for a ``gr(g, cov = A)`` or ``gr(g, dist = "student")`` term (one member, weight 1, no by levels), and
        ``covariates``, each coefficient's covariate per member and row (brms's ``Z_m_c_k``: ones for the varying intercept, the
        slope's column for every member, member ``k``'s own column for ``mmc()``)."""
        levels = self.spec.levels_by_group[t.factor]
        n = len(self.data)
        covariates = {c: tuple(np.ones(n) if col is None else self.data[col].to_numpy(dtype=float) for col in t.covariate_columns(c))
                      for c in t.coefs}
        if t.is_mm:
            index = tuple(level_codes(self.data[c].astype(str), levels) for c in t.members)   # G11: vectorised
            weights = mm_weights(t, self.data)
            return {"levels": levels, "index": index, "weights": tuple(weights[:, k] for k in range(len(t.members))),
                    "by_levels": (), "by_index": None, "covariates": covariates}
        index = (level_codes(self.data[t.column].astype(str), levels),)
        if t.by is None:                                     # G05b: gr(g, cov = A), gr(g, dist = "student")
            return {"levels": levels, "index": index, "weights": (np.ones(n),), "by_levels": (), "by_index": None,
                    "covariates": covariates}
        by_of = by_levels_of(t, self.data)
        by_levels = tuple(sorted(set(by_of.values())))
        return {"levels": levels, "index": index, "weights": (np.ones(n),), "by_levels": by_levels,
                "by_index": np.array([by_levels.index(by_of[level]) for level in levels], dtype=np.int64), "covariates": covariates}

    def _add_custom_group_effects(self, eta):
        """brms's multi-membership terms ``(1 | mm(g, g2, weights = cbind(w1, w2)))`` and ``(1 | gr(g, by = gby))`` (D08), built
        inside the model Bambi is building (``likelihoods.monotonic_family``'s hook on the parent's linear predictor, before the
        link), named as Bambi names a varying intercept: the SD ``1|g_sigma`` under the term's ``sd`` prior (sampled on its log;
        with ``by``, one SD per level of the by column on ``g__by_dim``), the standardised effects ``1|g_offset ~ Normal(0, 1)``
        and the effects ``1|g = sd[by of the level] * 1|g_offset`` (a deterministic on ``g__factor_dim``); each row adds
        ``sum_k W_k * 1|g[J_k]`` (brms's ``W_m_k r_m_1[J_m_k]``), its one level's effect for a ``by`` term.

        G05b (docs/gr-cov-dist.md): every coefficient of such a term (varying slopes in ``mm()``, ``mmc()``, and the terms with
        ``gr(g, cov = A)`` or ``gr(g, dist = "student")``) has its own SD and standardised effects, as Bambi names a slope's
        (``x|g_sigma``, ``x|g_offset``); with ``|`` they are correlated through ``L|g`` (brms's ``L_m``, ``lkj_corr_cholesky``,
        sampled on Stan's ``cholesky_corr`` coordinates; ``cor|g`` its matrix), brms's ``scale_r_cor``: coefficient ``c``'s effects
        are ``sd_c sum_k L[c, k] z_k``. A known covariance multiplies each coefficient's effects over the levels by ``Lcov`` (brms's
        ``Lcov_m``, so that ``vec(r)`` has covariance ``A kron Sigma``); Student-t effects multiply each level's by ``dfm|g =
        sqrt(df|g * udf|g)``, ``df|g`` under its prior truncated to ``df > 1`` (brms's ``real<lower=1> df_m``) and ``udf|g[j] ~
        InverseGamma(df / 2, 1 / 2)``, Stan's ``inv_chi_square(df)``. Each row adds ``sum_k W_k sum_c r_c[J_k] Z_c,k``."""
        import pymc as pm
        import pytensor.tensor as pt

        model = pm.modelcontext(None)
        sds = self.priors.sd_priors(self.spec)
        for t in self.spec.custom_group_terms:
            layout = self._custom_term_layout(t)
            roles = [self.group_coefficient_names[(t.factor, c)] for c in t.coefs]
            model.add_coord(roles[0]["coordinate"], list(layout["levels"]))
            scales, zs = [], []
            for c, r in zip(t.coefs, roles):
                prior = sds[(t.factor, c)]
                dist = getattr(pm, prior.name)
                params = {k: np.float64(v) for k, v in prior.params.items()}
                if t.by is not None:
                    model.add_coord(r["by_coordinate"], list(layout["by_levels"]))
                    sd = dist(r["sd"], **params, dims=r["by_coordinate"])
                    scale = sd[layout["by_index"]]
                    # brms's lprior of sd_m (M_m x Nby_m, M_m = 1 here) adds the untruncated density of every entry and subtracts
                    # the lower bound's log CCDF once for the term, not once per by level (gauss_gr_by's lprior): for a prior
                    # symmetric about 0 (normal, student_t, cauchy) brms's density is log 2 below the half prior's for each by
                    # level after the first. The constant is reproduced so that the log density is brms's; it moves no parameter.
                    missing = (len(layout["by_levels"]) - 1) * (np.log(2.0) if prior.name in _SYMMETRIC_HALF else 0.0)
                    if missing:
                        pm.Potential(r["by_normalisation"], pt.as_tensor_variable(np.float64(-missing)))
                else:
                    scale = dist(r["sd"], **params)
                scales.append(scale)
                zs.append(pm.Normal(r["offset"], mu=np.float64(0.0), sigma=np.float64(1.0), dims=r["coordinate"]))
            effects = [scales[0] * zs[0]]
            if t.has_correlation:                            # G05b: brms's scale_r_cor, (diag(sd) L z)^T
                L = self._custom_correlation(model, t)
                effects += [scales[c] * sum(L[c, k] * zs[k] for k in range(c + 1)) for c in range(1, len(t.coefs))]
            else:
                effects += [s * z for s, z in zip(scales[1:], zs[1:])]
            if t.cov is not None:                            # G05b: brms's Lcov_m over the levels, every coefficient (A kron Sigma)
                Lcov = pt.as_tensor_variable(np.asarray(self._group_cov[t.factor].L, dtype=np.float64))
                effects = [pt.dot(Lcov, e) for e in effects]
            if t.is_student:                                 # G05b: each level's effects times brms's dfm_m = sqrt(df udf)
                dfm = self._student_scale(model, t)
                effects = [dfm * e for e in effects]
            rs = [pm.Deterministic(r["group_effect"], e, dims=r["coordinate"]) for r, e in zip(roles, effects)]
            if t.thresholds:                                 # G10: (cs(1) | g): each threshold's own, added to mu_cs instead
                self._threshold_effects = {**getattr(self, "_threshold_effects", {}), t.factor: (rs, layout["index"][0])}
                continue
            for k, (index, weight) in enumerate(zip(layout["index"], layout["weights"])):
                for c, r in zip(t.coefs, rs):
                    part = r[index] if c == "Intercept" else r[index] * np.asarray(layout["covariates"][c][k], dtype=np.float64)
                    eta = eta + (np.asarray(weight, dtype=np.float64) * part if t.is_mm else part)
        return eta

    def _custom_correlation(self, model, t):
        """G05b: the correlation factor of a correlated term the model builds itself (``(1 + x | mm(g, g2))``, ``(1 + x | gr(g,
        cov = A))``): the free variable ``L|g`` (brms's ``L_m``) under ``lkj_corr_cholesky(eta)``, sampled on Stan's
        ``cholesky_corr`` coordinates ``L|g_cholesky_corr__``, and the deterministic ``cor|g = L L^T`` (brms's ``Cor_m``), on the
        coefficient dimensions, as ``_correlate_group_effects`` adds them to the terms Bambi builds."""
        import pymc as pm

        names, M = self._correlation_roles(t.factor), len(t.coefs)
        for dim in (names["coordinate"], names["coordinate_T"]):
            model.add_coord(dim, list(t.coefs))
        rv, transform, _ = cholesky_corr_rv(names["L"], float(self.priors.cor_priors(self.spec)[t.factor].params["eta"]), M)
        L = model.register_rv(rv, names["L"], transform=transform, dims=(names["coordinate"], names["coordinate_T"]))
        pm.Deterministic(names["cor"], L @ L.T, dims=(names["coordinate"], names["coordinate_T"]))
        return L

    def _student_scale(self, model, t):
        """G05b, brms's ``gr(g, dist = "student")``: the degrees of freedom ``df|g`` (brms's ``real<lower=1> df_m``) under the
        term's ``df`` prior truncated to ``df > 1`` and renormalised there (brms's lprior subtracts the prior's log CCDF at 1),
        sampled as ``log(df - 1)``; the per-level scales ``udf|g[j] ~ InverseGamma(df / 2, 1 / 2)`` (Stan's ``inv_chi_square(df)``,
        brms's ``udf_m``, sampled on their logs); and the deterministic ``dfm|g = sqrt(df * udf)`` (brms's ``dfm_m``) on the level
        dimension, which multiplies each level's effects."""
        import pymc as pm
        import pytensor.tensor as pt

        names = self.group_dist_names[t.factor]
        prior = self.priors.df_priors(self.spec)[t.factor]
        base = getattr(pm, prior.name).dist(**{k: np.float64(v) for k, v in prior.params.items()})
        df = pm.Truncated(names["df"], base, lower=np.float64(1.0))
        udf = pm.InverseGamma(names["udf"], alpha=df / np.float64(2.0), beta=np.float64(0.5), dims=names["coordinate"])
        return pm.Deterministic(names["dfm"], pt.sqrt(df * udf), dims=names["coordinate"])

    def _custom_group_blocks(self) -> dict[tuple[str, str], GroupBlock]:
        """The block of each coefficient of each multi-membership and ``by`` term (D08) and ``gr(g, cov = A)`` and ``gr(g, dist =
        "student")`` term (G05b), by ``(factor, coefficient)``, checked against the graph: each SD a free positive variable sampled
        on its log (a vector over the by levels with ``by``), the standardised effects a free vector and the effects a
        deterministic on the level dimension, the coordinates the sorted levels; a Student-t term's ``df|g`` free on its interval
        coordinate, ``udf|g`` free on its log and ``dfm|g`` a deterministic, on the level dimension; and the term's contribution to
        the parent's linear predictor recomputed from the data (``_check_custom_group_effects``). ``Z`` is the weighted sum of the
        members' indicators times the coefficient's covariate (the indicator of the level times it for one member); a
        multi-membership slope's block also keeps each member's covariate (``covariates``)."""
        pmm, blocks = self.pymc_model, {}
        start = pmm.initial_point()
        for t in self.spec.custom_group_terms:
            layout = self._custom_term_layout(t)
            levels = layout["levels"]
            for c in t.coefs:
                roles = self.group_coefficient_names[(t.factor, c)]
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
                # G11: the block keeps each member's codes and weights (and a slope's covariates), its Z built on access as
                # the weighted sum of the members' indicators times the covariate; one member, weight 1, for every other term
                blocks[(t.factor, c)] = GroupBlock(levels, None, layout["index"][0], slice(0, len(levels)),
                                                   memberships=layout["index"] if t.is_mm else (),
                                                   weights=layout["weights"] if t.is_mm else (),
                                                   by_levels=layout["by_levels"], by_index=layout["by_index"],
                                                   covariates=layout["covariates"][c] if t.is_mm and c != "Intercept" else (),
                                                   covariate=None if t.is_mm or c == "Intercept" else layout["covariates"][c][0])
            if t.is_student:                                 # G05b: df on (1, inf), udf positive, dfm their deterministic
                names = self.group_dist_names[t.factor]
                if (pmm[names["df"]] not in pmm.free_RVs or pmm.rvs_to_values[pmm[names["df"]]].name != f"{names['df']}_interval__"
                        or pmm[names["udf"]] not in pmm.free_RVs or pmm.rvs_to_values[pmm[names["udf"]]].name != f"{names['udf']}_log__"
                        or pmm[names["dfm"]] not in pmm.deterministics):
                    raise RuntimeError(f"{names['df']!r} and {names['udf']!r} are not free variables on their interval and log "
                                       f"coordinates, or {names['dfm']!r} is not a deterministic")
                for key in ("udf", "dfm"):
                    if tuple(pmm.named_vars_to_dims.get(names[key], ())) != (names["coordinate"],):
                        raise RuntimeError(f"PyMC dims of {names[key]!r} are not ({names['coordinate']!r},)")
        self._check_custom_group_effects(blocks)
        return blocks

    def _check_custom_group_effects(self, blocks) -> None:
        """Each multi-membership and ``by`` term's contribution (D08), read off the graph: with every such term's SD at 1 and
        standardised effects at 0, and every other variable at the graph's initial point, the parent's linear predictor (on
        Bambi's link scale: the parent itself, or the log of ``mu`` under a log link) must move, row by row, by exactly
        ``sum_k W_k * test[J_k]`` (``Z @ test``, recomputed from the data) when the term's standardised effects go from 0 to
        a fixed vector ``test``. G05b: so must every coefficient's of every term the model builds (the correlation factor at its
        initial point, the identity, so that coefficient ``c``'s ``z`` moves its own effects alone), by ``Z_c @ (dfm * (Lcov @
        test))``: ``Lcov`` the known covariance's Cholesky factor (the identity without one) and ``dfm`` the graph's
        Student-t scale at the point (ones without one)."""
        pmm, parent = self.pymc_model, self.family_spec.parent
        link = self.bambi_model.family.link[parent].name
        if link not in ("identity", "log"):
            raise RuntimeError(f"Bambi's link of {parent!r} is {link}; the mm() and gr(by = ) terms are checked under the identity "
                               "and log links")
        students = [self.group_dist_names[t.factor]["dfm"] for t in self.spec.custom_group_terms if t.is_student]
        fn = pmm.compile_fn(pmm.replace_rvs_by_values([pmm[parent], *(pmm[v] for v in students)]), inputs=pmm.value_vars,
                            point_fn=True, on_unused_input="ignore", mode="FAST_COMPILE")
        at0 = pmm.initial_point()
        for (factor, c), block in blocks.items():
            roles = self.group_coefficient_names[(factor, c)]
            at0[f"{roles['sd']}_log__"] = np.zeros(np.shape(at0[f"{roles['sd']}_log__"]))
            at0[roles["offset"]] = np.zeros(block.J)
        for t in self.spec.custom_group_terms:               # G05b: the correlation factor at the identity
            if t.has_correlation:
                value = f"{self.correlation_names[t.factor]['L']}_cholesky_corr__"
                at0[value] = np.zeros(np.shape(at0[value]))
        base = fn(at0)
        dfm = dict(zip(students, (np.asarray(v, dtype=float) for v in base[1:])))
        for (factor, c), block in blocks.items():
            roles, t = self.group_coefficient_names[(factor, c)], self.spec.group_term_of(factor)
            if t.thresholds:                                 # G10: checked on mu_cs (_category_specific_design)
                continue
            test = np.cos(np.arange(1.0, block.J + 1.0))
            lo, hi = (np.asarray(fn(p)[0], dtype=float).ravel() for p in (at0, {**at0, roles["offset"]: test}))
            moved = hi - lo if link == "identity" else np.log(hi) - np.log(lo)
            effect = self._group_cov[factor].L @ test if t.cov is not None else test
            if t.is_student:
                effect = dfm[self.group_dist_names[factor]["dfm"]] * effect
            expected = block.dot(effect)                     # G11: Z @ effect without forming Z
            if moved.shape != expected.shape or not np.allclose(moved, expected, rtol=1e-10, atol=1e-10):
                raise RuntimeError(f"the graph's {parent!r} does not move by the weighted effects of {factor!r} at its rows' levels; "
                                   f"largest difference {np.max(np.abs(moved - expected)) if moved.shape == expected.shape else moved.shape}")

    # -- a spatial CAR term (F02a) -------------------------------------------------------
    def _add_car_effects(self, eta):
        """brms's ``car()`` (F02a, docs/spatial-car.md), built inside the model Bambi is building: ``car.car_effect``."""
        from autobayes.car import car_effect

        return car_effect(eta, self.spec.car, self._car_design, self.priors.car_priors(self.spec), self.car_names)

    @property
    def car_names(self) -> dict[str, str]:
        """PyMC names of a CAR term (F02a, ``car.roles``): ``sdcar``, ``car`` or ``rhocar``, ``rcar``, ``zcar``, ``nszcar`` (those
        the type has, brms's names), ``"coordinate"`` (``rcar_dim``), esicar's ``"zcar_coordinate"`` and ``"lpdf"`` (the
        potential of its log density). Empty without one."""
        from autobayes.car import roles

        return roles(getattr(self.spec, "car", None))

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

    # -- the fixed residual covariance (F03a) -------------------------------------------
    def _check_fcor_likelihood(self) -> None:
        """The likelihood of an ``fcor(M)`` model read off the graph (F03a): at a point a small, distinct step from the graph's
        initial point (so that the predictor varies by row), the sum of the observed variable's elementwise log density equals
        ``log MultiNormal(y | mu, sigma^2 M)`` computed from the dense covariance (``fcor.fcor_values``), with the graph's own
        ``mu`` and ``sigma`` at that point."""
        import pymc as pm

        from autobayes.fcor import fcor_values

        pmm, n = self.pymc_model, self.names
        rv = pmm[n["y"]]
        start = pmm.initial_point()
        point = {name: np.asarray(v, dtype=float) + 0.05 * (1.0 + np.arange(np.size(v))).reshape(np.shape(v)) / max(np.size(v), 1)
                 for name, v in start.items()}
        outputs = pmm.replace_rvs_by_values([pm.logp(rv, pmm.rvs_to_values[rv]).sum(), pmm[self.family_spec.parent], pmm[n["sigma"]]])
        fn = pmm.compile_fn(outputs, inputs=pmm.value_vars, point_fn=True, on_unused_input="ignore", mode="FAST_COMPILE")
        got, mu, sigma = (np.asarray(v, dtype=float) for v in fn(point))
        expected = fcor_values(self.design.y, mu.ravel(), sigma, self.design.fcor)
        if not np.isclose(got, expected, rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"the graph's likelihood of {self.spec.fcor.term} is {float(got)!r}; MultiNormal(y | mu, sigma^2 "
                               f"{self.spec.fcor.matrix}) is {float(expected)!r}")

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
            x, sdx = t.operands[:2]
            roles = self.me_names[x]
            rows, dims = None, "__obs__"
            if "coordinate" in roles:                        # G10: me(x, sdx, gr = g), one latent value per level of g (brms's Jme)
                from autobayes.special_terms import measurement_levels
                levels, rows, noisy, noise = measurement_levels(t.name, self.data)
                dims = roles["coordinate"]
                model.add_coord(dims, list(levels))
            else:
                noisy, noise = measurement_values(t.name, self.data[x], self.data[sdx])
            mean = _prior_rv(roles["mean"], means[x])
            sd = _prior_rv(roles["sd"], sds[x], positive=True)
            z = pm.Normal(roles["z"], mu=np.float64(0.0), sigma=np.float64(1.0), dims=dims)
            latent = pm.Deterministic(roles["latent"], mean + sd * z, dims=dims)
            pm.Normal(roles["noisy"], mu=latent, sigma=np.asarray(noise, dtype=np.float64), observed=noisy, dims=dims)
            eta = eta + coefficient_rv(roles["b"], coefficients[t.name]) * (latent if rows is None else latent[rows])
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
            x, sdx = t.operands[:2]
            roles = self.me_names[x]
            levels, index, dims = (), None, "__obs__"
            if "coordinate" in roles:                        # G10: me(x, sdx, gr = g): the values of each level of g
                from autobayes.special_terms import measurement_levels
                levels, index, noisy, noise = measurement_levels(t.name, self.data)
                dims = roles["coordinate"]
                if tuple(str(v) for v in pmm.coords[dims]) != levels:
                    raise RuntimeError(f"PyMC coordinate {dims!r} is not the sorted levels of the gr column of {t.name!r}")
            else:
                noisy, noise = measurement_values(t.name, self.data[x], self.data[sdx])
            for key in ("mean", "sd", "b"):
                if pmm[roles[key]] not in pmm.free_RVs or pmm[roles[key]].type.ndim != 0:
                    raise RuntimeError(f"{roles[key]!r} is not a free scalar of the graph")
            if pmm.rvs_to_values[pmm[roles["sd"]]].name != f"{roles['sd']}_log__":
                raise RuntimeError(f"{roles['sd']!r} is not sampled on the log scale")
            if pmm[roles["z"]] not in pmm.free_RVs or pmm[roles["latent"]] not in pmm.deterministics:
                raise RuntimeError(f"{roles['z']!r} is not a free variable or {roles['latent']!r} not a deterministic of the graph")
            for key in ("z", "latent", "noisy"):
                if tuple(pmm.named_vars_to_dims.get(roles[key], ())) != (dims,):
                    raise RuntimeError(f"PyMC dims of {roles[key]!r} are not ({dims!r},)")
            observed = pmm[roles["noisy"]]
            held = np.asarray(getattr(pmm.rvs_to_values.get(observed), "data", np.nan), dtype=float)
            if observed not in pmm.observed_RVs or held.shape != noisy.shape or not np.array_equal(held, noisy):
                raise RuntimeError(f"{roles['noisy']!r} is not the observed noisy values of {t.name!r}")
            blocks[x] = MeasurementBlock(t.name, x, sdx, noisy, noise, levels, index)
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
            roles, block = self.me_names[t.operands[0]], self.design.measurement_error[t.operands[0]]
            z = np.linspace(-1.0, 1.0, self.design.n if block.index is None else len(block.levels))   # G10: one per level of gr
            point.update({roles["mean"]: np.asarray(0.3), f"{roles['sd']}_log__": np.asarray(np.log(0.7)), roles["z"]: z})
            terms.append((t.name, roles["b"], (0.3 + 0.7 * z) if block.index is None else (0.3 + 0.7 * z)[block.index]))
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
        from autobayes.formula import measurement_error_group

        return {t.operands[0]: {"b": t.name, "mean": f"meanme_{t.operands[0]}", "sd": f"sdme_{t.operands[0]}",
                                "z": f"zme_{t.operands[0]}", "latent": f"Xme_{t.operands[0]}", "noisy": f"Xn_{t.operands[0]}",
                                # G10: me(x, sdx, gr = g)'s latent values, standardised values and noisy values are per level of g
                                **({"coordinate": f"Xme_{t.operands[0]}_dim"} if measurement_error_group(t.name) else {})}
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
        dpars = getattr(getattr(self, "spec", None), "dpar_names", ())   # G06: a nu with a formula of its own is not learned so
        return {name: lower for name, lower in self.family_spec.learnable_parameters.items()
                if getattr(self, name) is None and name not in dpars}

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
            **{f"simo:{key}": name for key, name in (self.sp_names["simplex"].items() if self.spec.sp_terms else ())},   # G10
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
        if self.spec.has_category_specific:                   # D04: each threshold's predictor, mu plus Xcs bcs (brms's
            out["linpred"] = "mu_cs"                          # posterior_linpred of a model with cs() terms)
        if fam.per_category:                                  # B05: the coordinate of the non-reference categories
            out["category_dim"] = f"{self.spec.response}_reduced_dim"
        if fam.simplex is not None:                           # F01b: the observed K categories' coordinate; the logistic_normal
            roles = self.simplex_names                        # family's sigma<k> and Llncor under brms's names
            out["response_dim"] = roles["response_dim"]
            if "L" in roles:
                out.update({name: name for name in roles["sigma"]}, Llncor=roles["L"])
        if fam.mixture:                                       # C05: each component's sigma<k>, theta and the component coordinate
            roles = self.mixture_names
            out.update({name: name for name in roles["sigma"]}, theta=roles["theta"], component_dim=roles["coordinate"])
        if fam.continuous and fam.parent == "mu":
            out["mu"] = "mu"                                  # the continuous families whose parent is the location mu expose it by name (the Beta's parent is logit_mu)
        out.update({k: v for k, v in self.autocor_names.items() if not k.endswith("_coordinate")})   # "ar", "ma" (C04)
        out.update(self.sar_names)                            # "lagsar" or "errorsar" (F02b)
        out.update({k: v for k, v in self.car_names.items() if not k.endswith("coordinate") and k != "lpdf"})   # F02a: sdcar, rcar, ...
        out.update({v: v for k, v in self.timecor_names.items() if not k.startswith("coordinate")})   # F03b: "Lcortime", "Cortime", "cosy"
        if fam.baseline_hazard:                               # F05: the cox baseline hazard's simplex, brms's sbhaz
            out["sbhaz"] = "sbhaz"
        for d in self.spec.dpars:                             # G06: a parameter with a formula of its own by brms name (the
            out.setdefault(d.response, self._bambi_dpar(d.response))   # Student-t nu, the cumulative disc): its per-row values
        return out

    @property
    def simplex_names(self) -> dict:
        """PyMC names of a simplex model's own parameters and coordinates (F01b, docs/dirichlet-logistic-normal.md;
        ``families.simplex.names``): ``"response_dim"`` the coordinate of the observed response's K categories, and the
        logistic_normal family's ``"sigma"`` (brms's ``sigma<k>``), ``"L"`` (``Llncor``) and its two coordinates. Empty for any
        other model."""
        if self.family_spec.simplex is None:
            return {}
        from autobayes.families.simplex import names

        return names(self)

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
    def timecor_names(self) -> dict[str, str]:
        """PyMC names of a residual correlation term (F03b, docs/unstr-cosy.md): ``unstr()``'s ``"L"`` (``Lcortime``, brms's
        Cholesky factor of the correlation matrix over the times, sampled on Stan's ``cholesky_corr`` coordinates
        ``Lcortime_cholesky_corr__``), ``"cor"`` (``Cortime``, the matrix, a deterministic) and their dimensions
        ``"coordinate"`` / ``"coordinate_T"`` (``cortime_dim``, ``cortime_dim_T``, coordinates the times); ``cosy()``'s
        ``"cosy"`` (brms's ``cosy``). ``names`` holds them under their own names (``names["Lcortime"]``, ``names["Cortime"]``,
        ``names["cosy"]``; ``"L"`` and ``"cor"`` there are a correlated group term's). Empty without one."""
        term = getattr(self.spec, "timecor", None)            # a non-linear model's spec has none (C02)
        if term is None:
            return {}
        if term.call == "cosy":
            return {"cosy": "cosy"}
        return {"L": "Lcortime", "cor": "Cortime", "coordinate": "cortime_dim", "coordinate_T": "cortime_dim_T"}

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
            if d.sp_terms:                                    # G10: the formula's own simplexes, <component>_simo_<key>
                from autobayes.special_terms import names as special_names
                out[d.response].update({f"simo:{k}": v for k, v in special_names(d, f"{b}_")["simplex"].items()})
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
            if not t.has_intercept:                            # G05a: (0 + x | g), the slopes' roles and the level dimension alone
                roles = {"coordinate": roles["coordinate"]}
            for coef in t.slopes:
                roles.update({f"{k}:{coef}": v for k, v in self._group_roles(t.factor, coef).items() if k != "coordinate"})
            if t.has_correlation or t.id is not None:          # a correlated term, or a shared correlation ID (D08)
                roles.update({k: v for k, v in self._correlation_roles(t.factor).items() if k in ("L", "cor")})
            if t.by is not None:                               # D08: the dimension of gr(by = )'s SD
                roles["by_coordinate"] = f"{factor_column(t.factor)}__by_dim"
            if t.is_student:                                   # G05b: gr(g, dist = "student")'s df, udf and dfm
                roles.update({k: v for k, v in self.group_dist_names[t.factor].items() if k != "coordinate"})
            out[t.factor] = roles
        return out

    @property
    def group_dist_names(self) -> dict[str, dict[str, str]]:
        """G05b, brms's ``gr(g, dist = "student")`` (docs/gr-cov-dist.md): PyMC names of each such term's degrees of freedom
        ``df`` (``df|g``, brms's ``df_m``, sampled as ``df|g_interval__``, ``log(df - 1)``), its per-level scales ``udf`` (``udf|g``,
        brms's ``udf_m``, sampled on their logs) and their deterministic ``dfm = sqrt(df * udf)`` (``dfm|g``, brms's ``dfm_m``), on
        the level dimension ``coordinate``, by factor in formula order; also in ``group_names``. Empty without one."""
        out = {}
        for t in self.spec.group_terms:
            if t.is_student:
                column = factor_column(t.factor)
                out[t.factor] = {"df": f"df|{column}", "udf": f"udf|{column}", "dfm": f"dfm|{column}",
                                 "coordinate": f"{column}__factor_dim"}
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
                    kappa=None, quantile=None, tmp_xi=None, lagsar=None, errorsar=None, bs=None, bias=None,
                    car=None, Llncor=None, Lcortime=None, cosy=None, sbhaz=None, df=None, udf=None, disc=None) -> dict[str, np.ndarray]:
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
        F04:
        the wiener family's ``bs`` (the boundary separation, positive, stored as its log; or its formula's ``{"intercept": ...,
        "b": ...}``), ``ndt`` (as the shifted lognormal's) and ``bias`` (the relative starting point, in (0, 1), stored as its logit).
        F02a: ``car`` maps a CAR term's parameters and latent vectors, by brms's names, to their values (``car.value_entries``):
        ``{"sdcar": 0.8, "car": 0.4, "rcar": array(Nloc)}`` (escar), ``{"sdcar": ..., "zcar": array(Nloc - 1)}`` (esicar),
        ``{"sdcar": ..., "zcar": array(Nloc)}`` (icar), ``{"sdcar": ..., "rhocar": ..., "zcar": ..., "nszcar": ...}`` (bym2).
        F01b (docs/dirichlet-logistic-normal.md): the group-level terms of a per-category family (the dirichlet and logistic_normal
        families; the categorical family takes none, D37) give each coefficient one SD per non-reference category and one
        standardised effect per level and category: ``sd={"g": array(K - 1)}``, ``z={"g": array(J, K - 1)}`` (brms's ``sd_m`` and
        ``z_m`` of each category's group term in column k). The dirichlet family's precision is ``phi``; the logistic_normal
        family's SDs are ``sigma``, K - 1 positive numbers in category order (brms's ``sigma<k>``, stored as their logs), and
        ``Llncor`` the Cholesky factor of their correlation (brms's ``Llncor``, stored on Stan's ``cholesky_corr`` coordinates).
        F03b (docs/unstr-cosy.md): ``Lcortime`` is ``unstr()``'s Cholesky factor of the correlation matrix over the times (brms's
        ``Lcortime``: lower triangular, unit-norm rows), stored on Stan's ``cholesky_corr`` coordinates, and ``cosy`` is
        ``cosy()``'s correlation, strictly between 0 and 1, stored as its logit; each given exactly for its term.
        F05 (docs/cox.md): ``sbhaz`` is a cox model's baseline hazard simplex (brms's ``sbhaz``: ``Kbhaz`` positive weights summing
        to 1), stored on Stan's stick-breaking coordinates, required for the cox family and refused for every other.
        G05b (docs/gr-cov-dist.md): a ``gr(g, dist = "student")`` term takes ``df`` (its degrees of freedom, above 1) and ``udf``
        (its per-level scales, brms's ``udf_m``), each keyed by factor or bare for one such term (``_student_group_point``); the
        coefficients of a term the model builds itself (``mm()`` slopes, ``mmc()``, ``gr(cov = )``) take ``sd``, ``z`` and ``L``
        as every term does.
        G06 (docs/distributional-groups.md): a Student-t ``nu`` with its own formula is given as that formula's ``nu={"intercept":
        ..., "b": ...}``, and the cumulative family's ``disc`` with its formula likewise (``disc={...}``, its intercept centred as
        brms's ``Intercept_disc``), required exactly then; a distributional formula's group-level terms (an ID of its own, or one
        shared with the response formula) take ``"sd"`` and ``"z"`` keyed by factor, a mapping over the coefficients for a term
        with varying slopes (``sigma={..., "sd": {"g": {"Intercept": 0.4, "x": 0.2}}, "z": {"g": {"Intercept": array(J), "x":
        array(J)}}}``).
        """
        if disc is not None and "disc" not in self.spec.dpar_names:   # G06: the cumulative family's disc with its formula only
            raise ValueError("disc is the discrimination's formula ('disc ~ ...') of a cumulative model; this model gives disc none")
        from autobayes.car import value_entries
        car_point = value_entries(self, car)                  # F02a: a CAR term's values; refused for a model without one
        extras = {"alpha": alpha, "beta": beta, "ndt": ndt, "zoi": zoi, "coi": coi,   # D01, E01 (zero_one_inflated_beta)
                  "kappa": kappa, "quantile": quantile, "tmp_xi": tmp_xi,             # E01 (xbeta), E02
                  "bs": bs, "bias": bias}                                              # F04 (wiener)
        if self.family_spec.mixture:                          # C05: the components' parameters and theta
            if any(v is not None for v in extras.values()):   # D01, E01, E02: a mixture's Gaussian components have none
                raise ValueError("alpha, beta, ndt, zoi, coi, kappa, quantile, tmp_xi, bs and bias are the skew_normal, exgaussian, "
                                 "shifted_lognormal, zero_one_inflated_beta, xbeta, von_mises, asym_laplace, gen_extreme_value "
                                 "and wiener families' parameters; a mixture has none")
            if ar is not None or ma is not None:              # C04: a mixture has no autocorrelation term
                raise ValueError("ar and ma are an autocorrelation term's coefficients; a mixture has none")
            if Lcortime is not None or cosy is not None:      # F03b: nor a residual correlation term
                raise ValueError("Lcortime and cosy are a residual correlation term's parameters; a mixture has none")
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
                    if self.family_spec.per_category:         # F01b: one column per non-reference category
                        want = (block.J, len(self.categories) - 1)
                        if arr.shape != want or not np.isfinite(arr).all():
                            raise ValueError(f"z for {g!r} must be a finite array of shape {want}: one row per level, one column per "
                                             f"non-reference category {list(self.categories[1:])}")
                    elif arr.shape != (block.J,) or not np.isfinite(arr).all():
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
        simplex = self.family_spec.simplex                       # F01b: the logistic_normal family's own sigma and Llncor
        simplex_own = simplex.own_priors if simplex is not None else ()
        if Llncor is not None and "lncor" not in simplex_own:
            raise ValueError(f"Llncor is the Cholesky factor of the logistic_normal family's correlation of its log-ratios; the "
                             f"{self.family} family has none")
        if {k for k, v in auxiliary.items() if v is not None and k not in simplex_own} != ({scale} if scale is not None else set()):
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
        if self.spec.intercept and self.design.thresholds is not None:   # F06b: each level's thresholds, by level (brms's Intercept_k)
            from autobayes.grouped_thresholds import value_point as threshold_point
            point[f"{n['intercept']}_ordered__"] = threshold_point(self, intercept)
        elif self.spec.intercept and self.categories:        # B05: K - 1 thresholds, or one intercept per non-reference category
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
            elif self.family_spec.per_category:                # F01b: one SD per non-reference category
                arr = np.asarray(sd_by_coef[(g, coef)], dtype=float)
                if arr.shape != (len(self.categories) - 1,) or not np.isfinite(arr).all() or (arr <= 0).any():
                    raise ValueError(f"sd for {g!r} must be {len(self.categories) - 1} finite positive SDs, one per non-reference "
                                     f"category {list(self.categories[1:])}")
                point[f"{roles['sd']}_log__"] = np.log(arr)
            else:
                point[f"{roles['sd']}_log__"] = log_scale(sd_by_coef[(g, coef)], f"sd:{sd_tag(g, coef)}")
            point[roles["offset"]] = z_by_coef[(g, coef)]
        for g, roles in self.correlation_names.items():              # Stan's cholesky_corr coordinates of L (A02)
            point[f"{roles['L']}_cholesky_corr__"] = cholesky_corr_free(L_by_group[g])
        if simplex_own:                                        # F01b: the logistic_normal family's sigma<k> and Llncor
            from autobayes.families.simplex import own_point
            point.update(own_point(self, sigma=sigma, Llncor=Llncor))
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
        if "nu" in self.spec.dpar_names:                       # G06: nu's own formula's intercept and coefficients, as sampled
            point.update(self._dpar_point("nu", nu))
        elif "nu" in learned:                                  # learned (A13): sampled as log(nu - lower), the interval coordinate
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
        sp_point, bsp, simo = self._special_point(self.spec, "", bsp, simo)   # G10: the special products' entries of bsp and simo
        point.update(sp_point)
        mo = bool(self.spec.mo_terms)                        # bsp is a monotonic term's (B03) or an me() / mi() term's (C06)
        point.update(self._monotonic_point(bsp if mo or not (self.spec.me_terms or self.spec.mi_terms) else None, simo))
        point.update(self._gp_point(sdgp, lscale, zgp))      # C03: each Gaussian process's sdgp, lscale and zgp
        point.update(self._autocor_point(ar, ma))            # C04: an autocorrelation term's ar and ma
        point.update(self._timecor_point(Lcortime, cosy))    # F03b: a residual correlation term's Lcortime or cosy
        point.update(self._measurement_point(None if mo else bsp, meanme, sdme, zme, Ymi))   # C06: me() and mi()
        point.update(cs_point)                                # D04
        point.update(self._sar_point(lagsar, errorsar))       # F02b: a SAR term's lagsar or errorsar
        point.update(car_point)                               # F02a
        from autobayes.baseline_hazard import value_entries as baseline_hazard_point
        point.update(baseline_hazard_point(self, sbhaz))      # F05: the cox baseline hazard's simplex
        point.update(self._student_group_point(df, udf))      # G05b: gr(g, dist = "student")'s df and udf
        if "disc" in self.spec.dpar_names:                     # G06: the cumulative family's disc, its formula as sampled
            point.update(self._dpar_point("disc", disc))
        expected = {v.name for v in self.pymc_model.value_vars}
        if set(point) != expected:
            raise RuntimeError(f"value names {sorted(point)} do not match PyMC's {sorted(expected)}")
        return point

    def _student_group_point(self, df, udf) -> dict[str, np.ndarray]:
        """G05b: ``value_point``'s entries of the ``gr(g, dist = "student")`` terms: ``df`` the degrees of freedom (a finite number
        above 1, stored as ``log(df - 1)``, Stan's and PyMC's coordinate) and ``udf`` the per-level scales (``J`` positive numbers in
        the level order, stored as their logs), each keyed by factor, or bare for the model's one such term; given exactly with
        such terms."""
        names = self.group_dist_names
        if not names:
            if df is not None or udf is not None:
                raise ValueError("df and udf are the degrees of freedom and scales of a gr(g, dist = \"student\") term; this model has none")
            return {}
        factors, point = list(names), {}

        def keyed(value, label):
            if value is None:
                raise ValueError(f"{label} is required: the model has the Student-t group-level term(s) of {factors}")
            if isinstance(value, Mapping):
                if set(value) != set(factors):
                    raise ValueError(f"{label} must cover exactly the factors {factors}; got {list(value)}")
                return value
            if len(factors) != 1:
                raise ValueError(f"{label} requires an exact-key mapping over the factors {factors}")
            return {factors[0]: value}

        dfs, udfs = keyed(df, "df"), keyed(udf, "udf")
        for g, roles in names.items():
            v = np.asarray(dfs[g], dtype=float)
            if v.shape != () or not np.isfinite(v) or v <= 1:
                raise ValueError(f"df for {g!r} must be a finite scalar > 1")
            u = np.asarray(udfs[g], dtype=float)
            J = len(self.design.group_designs[g].levels)
            if u.shape != (J,) or not np.isfinite(u).all() or (u <= 0).any():
                raise ValueError(f"udf for {g!r} must be {J} finite positive numbers, one per level")
            point[f"{roles['df']}_interval__"], point[f"{roles['udf']}_log__"] = np.log(v - 1.0), np.log(u)
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
        columns = [cs_label(t) for t in terms]                # G10: zq for cs(z)[q]
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
            arr = np.asarray(by_column[cs_label(t)], dtype=float)
            if arr.shape != (K1,) or not np.isfinite(arr).all():
                raise ValueError(f"bcs for {cs_label(t)!r} must be {K1} finite numbers, one per threshold")
            out[t.name] = arr
        return out

    def _special_point(self, spec, prefix: str, bsp, simo):
        """G10: ``value_point``'s entries of the special products of ``spec`` (the response formula, or a distributional formula
        with ``prefix``): ``bsp`` keyed by term name (``"mo(ord):x"``) beside the plain terms' columns, ``simo`` by id, term name or
        ``{term: {column: ...}}`` beside the plain monotonic terms' columns (``special_terms.split_simo``); each simplex stored on
        Stan's stick-breaking coordinates. Returns the entries and what is left of ``bsp`` and ``simo`` for the plain terms
        (None when nothing is)."""
        from autobayes import special_terms
        from autobayes.monotonic import check_simplex, stick_breaking_free

        if not spec.sp_terms:
            return {}, bsp, simo
        roles, terms = special_terms.names(spec, prefix), [t.name for t in spec.sp_terms]
        others = [*(t.operands[0] for t in spec.mo_terms), *(t.operands[0] for t in (*spec.me_terms, *spec.mi_terms))]
        if not isinstance(bsp, Mapping) and not (len(terms) == 1 and not others and bsp is not None):
            raise ValueError(f"bsp must map each special term {terms} (and the columns of the plain terms {others}) to its coefficient")
        coefficients = bsp if isinstance(bsp, Mapping) else {terms[0]: bsp}
        missing = [k for k in terms if k not in coefficients]
        if missing:
            raise ValueError(f"bsp has no coefficient for the special term(s) {missing}")
        point = {}
        for name in terms:
            value = np.asarray(coefficients[name], dtype=float)
            if value.shape != () or not np.isfinite(value):
                raise ValueError(f"bsp for {name!r} must be a finite scalar")
            point[roles["b"][name]] = value
        keys = special_terms.simplexes(spec)
        rest_simo = simo
        if keys:
            if simo is None:
                raise ValueError(f"simo is required for the simplexes {list(keys)} of the special terms")
            rest_simo, given = special_terms.split_simo(simo, spec, [t.operands[0] for t in spec.mo_terms])
            for key, (m, _, _, _) in keys.items():
                point[f"{roles['simplex'][key]}_stickbreaking__"] = stick_breaking_free(check_simplex(given[key], m.D, f"simo for {key!r}"))
        rest = {k: v for k, v in coefficients.items() if k not in terms}
        return point, (rest or None), rest_simo

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

    def _timecor_point(self, Lcortime, cosy) -> dict[str, np.ndarray]:
        """``value_point``'s entries of a residual correlation term (F03b): ``unstr()``'s ``Lcortime`` on Stan's
        ``cholesky_corr`` coordinates, or ``cosy()``'s ``cosy`` as its logit (the coordinate of PyMC's ``Uniform(0, 1)`` and
        ``Beta`` alike), exactly for the term the model has."""
        roles, point = self.timecor_names, {}
        given = {"L": Lcortime, "cosy": cosy}
        for key, label in (("L", "Lcortime"), ("cosy", "cosy")):
            if key not in roles:
                if given[key] is not None:
                    raise ValueError(f"this model has no {label}" + (f" ('{self.spec.timecor.term}')" if self.spec.timecor is not None
                                     else " (no residual correlation term unstr() or cosy())") + f"; {label} does not apply")
                continue
            if given[key] is None:
                raise ValueError(f"{label} is required: the model has the term '{self.spec.timecor.term}'")
            value = self.pymc_model.rvs_to_values[self.pymc_model[roles[key]]].name
            if key == "L":
                point[value] = cholesky_corr_free(check_correlation_cholesky(Lcortime, self.design.timecor.dim, "Lcortime"))
                continue
            p = np.asarray(cosy, dtype=float)
            if p.shape != () or not np.isfinite(p) or not 0 < p < 1:
                raise ValueError("cosy must be a finite scalar strictly between 0 and 1")
            point[value] = np.asarray(np.log(p) - np.log1p(-p))
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
                z, block = np.asarray(zs[x], dtype=float), self.design.measurement_error[x]
                size = self.design.n if block.index is None else len(block.levels)   # G10: one per level of me()'s gr
                if z.shape != (size,) or not np.isfinite(z).all():
                    raise ValueError(f"zme for {x!r} must have {size} finite entries, one per "
                                     + ("fitted row" if block.index is None else f"level of the gr column {list(block.levels)}"))
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
        keys |= {"bsp", "simo"} if d.sp_terms else set()     # G10: the formula's special products
        if not isinstance(value, Mapping) or set(value) - keys:
            raise ValueError(f"{name} has its own formula '{d.formula}': give {name}={{'intercept': float, 'b': array"
                             + (", 'sd': {factor: float}, 'z': {factor: array}" if d.group_terms else "") + "} "
                             "(the formula's intercept as sampled and its coefficients in design-column order"
                             + ("; a term with varying slopes takes a mapping over its coefficients" if any(t.slopes for t in d.group_terms)
                                else "") + ")")
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
        for term in d.bambi_terms:                            # G10: a special product's coefficient is bsp's, below
            k = len(term.columns)
            point[roles[f"b:{term.name}"]] = np.array(b[i:i + k]) if term.levels else np.asarray(b[i])
            i += k
        if d.sp_terms:
            point.update(self._special_point(d, f"{self._bambi_dpar(name)}_", value.get("bsp"), value.get("simo"))[0])
        return point

    def _dpar_group_point(self, name: str, sd, z) -> dict[str, np.ndarray]:
        """D08: ``value_point``'s entries of a distributional formula's group-level terms (a shared correlation ID's; G06: or the
        formula's own): each coefficient's SD on its log and its standardised effects, from ``sd`` and ``z`` keyed by factor (a
        single term also takes the bare values), each entry a mapping over the coefficients for a term with varying slopes (G06)
        and the varying intercept's value otherwise."""
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

        def by_coefficient(value, t, label):                  # G06: one entry per coefficient of a term with varying slopes
            if isinstance(value, Mapping):
                if set(value) != set(t.coefs):
                    raise ValueError(f"{name}[{label!r}] for {t.factor!r} must cover exactly its coefficients {list(t.coefs)}; "
                                     f"got {list(value)}")
                return value
            if t.slopes:
                raise ValueError(f"{name}[{label!r}] for {t.factor!r} must be a mapping over its coefficients {list(t.coefs)} "
                                 f"('{t.canonical}' has varying slopes)")
            return {"Intercept": value}

        sds, zs, point = keyed(sd, "sd"), keyed(z, "z"), {}
        for t in d.group_terms:
            per_sd, per_z = by_coefficient(sds[t.factor], t, "sd"), by_coefficient(zs[t.factor], t, "z")
            for c in t.coefs:
                roles, block = self._shared_roles(t.factor, name, c), self.design.dpars[name].group_blocks[(t.factor, c)]
                what = repr(t.factor) if c == "Intercept" else f"the slope {c!r} of {t.factor!r}"
                value = np.asarray(per_sd[c], dtype=float)
                if value.shape != () or not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{name}['sd'] for {what} must be a finite positive scalar")
                arr = np.asarray(per_z[c], dtype=float)
                if arr.shape != (block.J,) or not np.isfinite(arr).all():
                    raise ValueError(f"{name}['z'] for {what} must have {block.J} finite entries")
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
        from autobayes.car import describe as car_describe     # F02a: a CAR term's lines, [] without one
        from autobayes.column_types import LEVEL_ORDER          # D172: the order of a factor's levels
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
        grouping = ("; ".join(f"J = {block.J} levels of {g}" for g, block in d.factor_blocks.items())
                    if s.has_group else "no group-level term (population-level model)")
        data_line = f"  data      : n = {d.n} observations, {grouping}"
        if fam.has_trials:
            data_line = (f"  data      : {d.n} fitted rows, {int(d.trials.sum())} trials in total, "
                         f"{grouping}; {len(self.dropped_rows)} zero-trial row(s) dropped"
                         + (f" (positions {list(self.dropped_rows)[:10]}{'...' if len(self.dropped_rows) > 10 else ''})"
                            if self.dropped_rows else ""))
        if self.dropped_na_rows:                             # D174: the rows dropna=True dropped
            data_line += (f"; {len(self.dropped_na_rows)} row(s) with missing values dropped (dropna=True, positions "
                          f"{list(self.dropped_na_rows)[:10]}{'...' if len(self.dropped_na_rows) > 10 else ''})")
        # F08: the treatment-coded main effects; an ordered factor's are contrasts; G03: a factor in interactions only has none
        treat = tuple(c for c in s.categoricals if not c.is_poly and c.name in s.categorical_names)
        products = [t for t in s.pop_terms if t.kind == "product"]   # G03: interactions coded by R's rule (term_algebra)
        features = ((["categorical predictor(s)"] if treat else []) + (["interaction(s)"] if s.interactions or products else [])
                    + (["ordered factor(s) with polynomial contrasts"] if any(c.is_poly for c in s.categoricals) else [])
                    + (["in-formula transform(s)"] if s.transform_terms else []) + (["monotonic effect(s)"] if s.mo_terms else [])
                    + (["Gaussian process(es)"] if s.gp_terms else [])
                    + (["independent varying slope(s)"] if any(t.slopes and not t.has_correlation and t.id is None for t in s.group_terms) else [])
                    + (["correlated varying slope(s)"] if s.correlated_terms else []) + (["offset"] if s.offset is not None else [])
                    + (["derived grouping factor(s)"] if any(t.derived for t in s.group_terms) else [])
                    + (["multi-membership term(s)"] if any(t.is_mm for t in s.group_terms) else [])           # D08
                    + (["group SDs by level of a factor (gr(by = ))"] if any(t.by for t in s.group_terms) else [])
                    + (["a known covariance of a factor's levels (gr(cov = ))"] if any(t.cov for t in s.group_terms) else [])   # G05b
                    + (["Student-t group effects (gr(dist = \"student\"))"] if any(t.is_student for t in s.group_terms) else [])
                    + (["correlation ID(s) shared with a distributional formula"] if s.shared_ids else [])
                    + ([f"response addition term(s) {', '.join(a.term for a in s.additions)}"] if s.additions else [])
                    + ([f"distributional formula(s) for {', '.join(s.dpar_names)}"] if s.dpars else [])
                    + ([f"autocorrelation term {s.autocor.term}"] if s.autocor is not None else [])
                    + ([f"simultaneous autoregressive term {s.sar.term}"] if s.sar is not None else [])          # F02b
                    + ([f"fixed residual covariance {s.fcor.term}"] if s.fcor is not None else [])    # F03a
                    + ([f"spatial CAR term {s.car.term}"] if s.car is not None else [])                   # F02a
                    + ([f"residual correlation term {s.timecor.term}"] if s.timecor is not None else [])   # F03b
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
        if fam.simplex is not None:                           # F01b: the cbind() response's columns, categories and reference
            from autobayes.families.simplex import response_lines
            lines += response_lines(self)
        elif self.categories:                                 # B05: the response's categories, fitted as brms's 1-based Y
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
        if d.thresholds is not None:                         # F06b: thres(gr = z)'s levels
            from autobayes.grouped_thresholds import data_lines as threshold_data_lines
            lines += threshold_data_lines(self)
        lines += self._autocor_data_lines()                  # C04: the series of an autocorrelation term
        if s.sar is not None:                                # F02b: the SAR term's weight matrix
            from autobayes.sar import data_lines
            lines += data_lines(self)
        if s.fcor is not None:                               # F03a: the matrix from data2
            M = d.fcor.M
            lines += [f"  fcor      : {s.fcor.term}: data2[{s.fcor.matrix!r}], the {d.n} x {d.n} covariance of the residuals before the "
                      f"factor sigma^2 (brms's Mfcor), its rows and columns the fitted rows in data order; diagonal {np.diag(M).min():.4g} "
                      f"to {np.diag(M).max():.4g}, log det {np.linalg.slogdet(M)[1]:.6g}"]
        lines += car_describe(self, "data")                  # F02a: a CAR term's locations and graph
        lines += self._timecor_data_lines()                  # F03b: the groups of a residual correlation term
        for t in s.transform_terms:
            tr, col = t.transform, d.X_raw[:, s.columns.index(t.name)]
            if t.basis is not None:                          # G04: an in-formula expression's column (docs/expressions.md)
                lines.append(expressions.describe_data(tr, col))
                continue
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
        from autobayes.term_algebra import product_lines
        lines += product_lines(s)                            # G03: each interaction coded by R's rule
        for g, block in d.factor_blocks.items():
            lines += [f"  {'levels    ' if len(s.groups) == 1 else 'levels ' + g}: {', '.join(f'{j}={l}' for j, l in enumerate(block.levels))}"]
        for t in s.group_terms:
            block = d.group_blocks.get((t.factor, "Intercept"))
            if t.is_mm:                                      # D08: brms's mm(): memberships and weights
                w = ("the weight columns " + ", ".join(t.weights) + " divided by their sum on each row (brms's scale = TRUE)"
                     if t.weights else f"equal weights 1/{len(t.members)}")
                if t.unscaled and t.weights:                 # G05b: brms's scale = FALSE
                    w = "the weight columns " + ", ".join(t.weights) + " as given (brms's scale = FALSE)"
                lines += [f"  {t.factor:<10}: multi-membership grouping {t.grouping} (brms's mm(); group name {t.factor}): each row belongs to its "
                          f"levels of {', '.join(repr(c) for c in t.members)} with {w}; the levels are the union of the members' labels "
                          f"(brms's J_m_k and W_m_k)"]
            if t.by is not None:                             # D08: brms's gr(by = ): the SD of each level's by level
                groups = "; ".join(f"{b}: " + ", ".join(l for j, l in enumerate(block.levels) if block.by_index[j] == k)
                                   for k, b in enumerate(block.by_levels))
                lines += [f"  {t.factor:<10}: gr({t.factor}, by = {t.by}): one SD per level of {t.by!r} ({groups}); each level of "
                          f"{t.factor!r} lies within one level of {t.by!r} (brms's Nby_m and Jby_m)"]
            lines += self._custom_data_lines(t)              # G05b: gr(cov = ), gr(dist = "student"), mm() slopes
            if t.derived:
                ops = ", ".join(t.operands)                  # G05a: three columns too, g:h:f2
                name = _bambi_group_name(t.factor, t.coefs[0])
                lines += [f"  {t.factor:<10}: derived grouping factor, one level per observed ({ops}) combination, labelled {'_'.join(t.operands)} as brms "
                          f"does (paste({ops}, sep = \"_\")); unobserved combinations have no level; handed to Bambi as the column "
                          f"{t.column!r}, so its PyMC variables are {name}, {name}_sigma, {name}_offset"]
            lines += self._group_syntax_lines(t)            # G05a: slopes alone, several terms, factor/transform/product/mo slopes
        for cat in s.all_categoricals:                       # a distributional formula's too (B01), b_sigma_z
            owners = [(b, f.categorical(cat.name)) for b, f in [("b", s), *((f"b_{ds.response}", ds) for ds in s.dpars)]
                      if cat.name in f.categorical_names]
            labels = self.data[cat.name].astype(str).to_numpy()
            props = ", ".join(f"{l} {np.mean(labels == l):.3f}" for l in cat.levels)
            if cat.is_poly:                                  # F08: an ordered factor, R's polynomial contrasts (docs/ordered-factors.md)
                C = poly_contrasts(len(cat.levels))
                values = "; ".join(f"{l}: " + ", ".join(f"{v:.4f}" for v in C[i]) for i, l in enumerate(cat.levels))
                lines += [
                    f"  {cat.term_name:<10}: ordered factor, polynomial contrasts (R's contr.poly over its levels {' < '.join(cat.levels)}, as "
                    f"brms codes an ordered factor): columns {', '.join(cat.contrast_columns)}, the orthonormal polynomials of degree 1 to "
                    f"{len(cat.levels) - 1} in the level's rank; their values by level ({values})"
                    + (f"; unused declared categories {list(cat.unused)}" if cat.unused else ""),
                    f"              proportions among fitted rows: {props}",
                ]
                continue
            if not owners:                                   # G03: a factor in interactions only, coded there by R's rule
                lines += [f"  {cat.name:<10}: categorical, in interactions only (no main effect; each interaction's line gives its "
                          f"coding), reference {cat.reference!r}; levels (sorted) {', '.join(cat.levels)}",
                          f"              proportions among fitted rows: {props}"]
                continue
            lines += [
                f"  {cat.name:<10}: categorical, {'full-level coding' if cat.full_rank else 'treatment coding'}, reference {cat.reference!r}; "
                f"levels ({LEVEL_ORDER[cat.from_dtype]}) "                    # D172: a pandas Categorical's or a logical's order
                f"{', '.join(cat.levels)}; coefficients " + "; ".join(f"{b}_{cat.name}[{', '.join(c.coefficients)}]" for b, c in owners)
                + (f"; unused declared categories {list(cat.unused)}" if cat.unused else ""),
                f"              proportions among fitted rows: {props}",
            ]
        def mm_member(t, k, c):                             # G05b: member k's share, its varying slopes' covariates its own
            slopes = "".join(f" + x_{t.covariate_columns(coef)[k]},i * {slope_effect(t.factor, coef)}[{c}_i]" for coef in t.slopes)
            return f" + W_{c},i * ({effect(t.factor)}[{c}_i]{slopes})" if slopes else f" + W_{c},i * {effect(t.factor)}[{c}_i]"
        rest = "".join(f" + gp({g.column})_i" for g in s.gp_terms) + (" + offset_i" if s.offset is not None else "") + "".join(
            ("".join(mm_member(t, k, c) for k, c in enumerate(t.members)) if t.is_mm else   # D08: brms's W_m_k r_m_1[J_m_k] (G05b: slopes)
             (f" + {effect(t.factor)}[{t.factor}_i]" if t.has_intercept else "")   # G05a: (0 + x | g) has no varying intercept
             + "".join(f" + {self._slope_value(t, c)} * {slope_effect(t.factor, c)}[{t.factor}_i]" for c in t.slopes))
            for t in s.group_terms)
        rest += " + arma_i" if s.autocor is not None else ""   # C04: the autocorrelation term, last
        rest += f" + rcar[{s.car.gr}_i]" if s.car is not None else ""   # F02a: the CAR term
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
            per_row = s.offset.row_expression()
            lines += [f"  offset_i = {per_row}: the offset term, a known quantity added to every row's {lp}_i; "
                      "it has no coefficient and no prior and is not centred."]
        lines += self._monotonic_lines(lp)
        if s.sp_terms:                                       # G10: the special products written out (docs/special-term-ext.md)
            from autobayes.special_terms import describe_lines
            lines += describe_lines(s)
        lines += self._gp_lines(lp)
        lines += self._autocor_lines(lp)                     # C04
        lines += car_describe(self, "term", lp)              # F02a
        lines += self._measurement_lines(lp)                 # C06
        scale = fam.scale_phrase
        same_group = " (same group)" if len(s.groups) == 1 else (" (same group effects)" if s.groups else "")
        for cat in (c for c in s.categoricals if c.is_poly or c.name in s.categorical_names):   # G03: main effects only
            if cat.is_poly:                                   # F08
                from autobayes.default_priors import brms_rename
                lines += [f"  {', '.join(f'b_{c}' for c in cat.contrast_columns)}: the coefficients of {cat.term_name}'s polynomial contrasts "
                          f"(brms's b_{', b_'.join(brms_rename(c) for c in cat.contrast_columns)}), the linear, quadratic, ... trend across its "
                          f"ordered levels on the {fam.scale_phrase} scale; level l adds sum_d C[l, d] b_d, C the contrasts above."]
                continue
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
            if t.basis is not None:                          # G04
                lines.append(expressions.describe_meaning(t.transform, lp))
                continue
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
            where = "at the predictor means" if not treat else \
                "at the numeric predictor means and at the sample proportions of the categorical levels"
            if s.interactions or products:
                expanded = ((["x"] if s.predictors else []) + (["1[z = l]"] if treat else []) + (["x * 1[z = l]"] if num_cat else [])
                            + (["x * w"] if num_num else []) + (["1[z = l] * 1[h = m]"] if cat_cat else [])
                            + (["the interaction columns of R's coding"] if products else []))
                where = f"at the fitted-row means of every expanded design column ({', '.join(expanded)})"
            lines += [
                "  Predictors are centred at their sample means (center_predictors=True, as in",
                f"  brms and Bambi)" + ("; a dummy column's mean is its level's proportion p_z[l]." if treat else "."),
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
            correction = (" - sum_l p_z[l] * b_z[l]" if treat else "")
            correction += (" - sum_(x,z),l q_x:z[l] * b_x:z[l]" if len(num_cat) > 1 else (" - sum_l q_x:z[l] * b_x:z[l]" if num_cat else ""))
            correction += "".join(f" - q_{t.name} * b_{t.name}" for t in num_num)
            correction += "".join(f" - sum_(l,m) p_{t.name}[l, m] * b_{t.name}[l, m]" for t in cat_cat)
            if products:                                     # G03: each column of R's coding at its own mean
                cols = [d.X_raw[:, s.columns.index(t.name)] for t in products]
                lines += [f"  Each interaction column of R's coding is centred at its own mean over the fitted rows: "
                          + ", ".join(f"q_{t.name} = {c.mean():.4f}" for t, c in zip(products, cols)) + "."]
                correction += " - sum_c q_c * b_c (the interaction columns c)"
            lines += [
                f"  a_c is the {what} {where}{' (group effect zero)' if s.has_group else ''}; the",
                "  prior on the intercept applies to a_c. The posterior reports the raw-scale",
                "  intercept a = a_c - sum_p mean(x_p) * b_p" + correction + ", computed after sampling"
                + (": the linear predictor at x = 0 and at the reference level." if treat else "."),
            ]
            if treat:
                lines += list(fam.centred_intercept_note)
        elif not s.intercept:
            lines += ["  No population intercept or intercept prior. The raw design is used without back-transformation;",
                      f"  center_predictors requested={self.center_predictors}, effective=False (centering requires a free intercept)."]
            if len(s.categoricals) > 1:
                lines += ["  Only the first categorical main effect is full-level coded. Reordering factors changes independent prior meaning."]
        else:
            lines += ["  Predictors are not centred (center_predictors=False); a is the raw-scale intercept"
                      + ((" (x = 0, reference level" + (", group effect zero)." if s.has_group else ").")) if treat else ".")]
        for ds in s.dpars:                                   # each distributional parameter's own linear predictor (B01)
            lines += self._dpar_lines(ds)
        if s.has_group:
            lines += ["", "Group effects (non-centred)"]
            for t in s.group_terms:
                g, block = t.factor, d.factor_blocks[t.factor]
                shared = s.shared_id_of(g)
                if t.by is not None:                         # D08: the SD of the level's by level
                    lines += [f"  u_{g}[j] = sd_{g}[{t.by}[j]] * z_{g}[j], z_{g}[j] ~ Normal(0, 1), j = 0..{block.J - 1}: one SD per level "
                              f"of {t.by!r} ({', '.join(block.by_levels)}), the level's own (brms's sd_m[1, Jby_m[j]])"]
                elif fam.per_category:                       # F01b: one group term per non-reference category, as brms's
                    lines += [f"  u_{g}[j, k] = sd_{g}[k] * z_{g}[j, k], z_{g}[j, k] ~ Normal(0, 1), j = 0..{block.J - 1}, for each "
                              f"non-reference category k = {', '.join(self.categories[1:])}: brms's own group term of {g} in each mu<k>, "
                              "independent across categories; mu_i[k] adds u_" + f"{g}[{g}_i, k]"]
                elif t.has_intercept:
                    lines += [f"  u_{g}[j] = sd_{g} * z_{g}[j], z_{g}[j] ~ Normal(0, 1), j = 0..{block.J - 1}"]
                else:                                        # G05a: (0 + x | g), the slopes alone
                    lines += self._slope_only_lines(t, block)
                    continue
                if t.is_mm and not t.slopes and not t.unscaled:
                    lines += [f"  Row i adds sum_k W_k,i * u_{g}[level k of row i] over its members {', '.join(t.members)}, the weights summing to 1."]
                lines += self._custom_effect_lines(t, block.J)  # G05b: mm() slopes and scale = FALSE, Lcov, Student-t scales
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
        if s.intercept and d.thresholds is not None:         # F06b: every threshold of every level
            from autobayes.grouped_thresholds import prior_line
            lines += [prior_line(self)]
        elif s.intercept and fam.thresholds:
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
        if s.sp_terms:                                       # G10: a special product's coefficient (class b) and simplexes
            from autobayes.special_terms import prior_lines
            lines += prior_lines(s, coef, pr.special_simo_priors(s), describe_prior)
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
        lines += car_describe(self, "priors")                # F02a: sdcar, car, rhocar
        lines += self._timecor_prior_lines()                 # F03b: cortime or cosy
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
            lines += [f"  sd_{sd_tag(g, c)}{each} ~ {describe_prior(prior, lb=0)}"]
        for g, prior in pr.df_priors(s).items():             # G05b: gr(g, dist = "student")'s degrees of freedom
            lines += [f"  df_{g} ~ {describe_prior(prior, lb=1)}, truncated to df_{g} > 1 and renormalised there (brms class df, group {g})"]
        for g, prior in pr.cor_priors(s).items():
            lines += [f"  Cor_{g} ~ {prior} (brms lkj({prior.params['eta']:g}) on the correlation matrix, placed on L_{g} as lkj_corr_cholesky)"
                      + ("   (brms default)" if prior.source == "default" else "")]
        for cls, what in (("b", "coefficient"), ("sd", "group-level SD"), ("cor", "correlated factor")):   # D173
            default = class_default(getattr(pr, cls), cls)
            if default is not None:
                lines += [f"  ({cls}['*'] = {default}: the class-wide prior of every {what} the mapping does not name, brms's class "
                          f"{cls} row, which the named entries override; docs/class-priors.md)"]
        for ds in s.dpars:                                   # a distributional formula's intercept and coefficients (B01)
            name, sub, dd = ds.response, pr.dpar_priors(s, ds.response), d.dpars[ds.response]
            if ds.intercept:
                lines += [f"  a_{name}{'_c' if dd.centered else ''} ~ {describe_prior(sub.intercept)}   (on the {dd.link} scale of {name})"]
            lines += [f"  b_{name}_{col} ~ {describe_prior(prior)}" for col, prior in sub.coefficient_priors(ds).items()]
            if ds.sp_terms:                                  # G10: the formula's own simplexes
                from autobayes.special_terms import prior_lines
                lines += prior_lines(ds, sub.coefficient_priors(ds), sub.special_simo_priors(ds), describe_prior,
                                     f"{self._bambi_dpar(name)}_")[len(ds.sp_terms):]
            lines += [f"  sd_{name},{sd_tag(g, c)} ~ {describe_prior(prior, lb=0)}   (brms class sd with dpar = {name})"   # D08, G06
                      for (g, c), prior in (sub.sd_priors(ds).items() if ds.group_terms else ())]
        if s.sar is None:
            lines += fam.likelihood_lines(self)
        else:                                                # F02b: the joint SAR density in place of the family's
            from autobayes.sar import likelihood_lines
            lines += likelihood_lines(self)
        lines += self._timecor_lines()                       # F03b: each group's rows jointly
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
        if s.intercept and d.thresholds is not None:         # F06b: each level's block on Stan's ordered coordinates
            from autobayes.grouped_thresholds import value_line
            roles = dict([value_line(self)])
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
            if fam.per_category:                              # F01b: one SD per non-reference category, z by level and category
                others = ", ".join(self.categories[1:])
                roles[f"{names['sd']}_log__"] = f"log(sd_{sd_tag(g, c)}[{others}]) (vector; dims {n['category_dim']})"
                roles[names["offset"]] = f"z_{g}[j, k] (matrix; dims {names['coordinate']} x {n['category_dim']})"
        simplex_roles = self.simplex_names                    # F01b: the logistic_normal family's sigma<k> and Llncor
        for name, cat in zip(simplex_roles.get("sigma", ()), self.categories[1:]):
            roles[f"{name}_log__"] = f"log({name}), the SD of log(y[{cat}] / y[{self.categories[0]}]) (brms's {name})"
        if "L" in simplex_roles:
            roles[f"{simplex_roles['L']}_cholesky_corr__"] = (f"{simplex_roles['L']}, the Cholesky factor of the log-ratios' correlation, on "
                                                              "Stan's cholesky_corr coordinates (brms's Llncor)")
        for (g, c), names in self.group_coefficient_names.items():   # D08: gr(by = )'s SD, a vector over the by levels
            if "by_coordinate" in names:
                roles[f"{names['sd']}_log__"] = f"log(sd_{g}[{', '.join(self.design.group_blocks[(g, c)].by_levels)}]) (vector; dims {names['by_coordinate']})"
        for (name, g, c), names in self.dpar_group_names.items():   # D08: a shared ID's distributional coefficient; G06: own, slopes
            roles[f"{names['sd']}_log__"] = f"log(sd_{name},{sd_tag(g, c)})"
            roles[names["offset"]] = f"z_{name},{sd_tag(g, c)}"
        for g, names in self.correlation_names.items():
            roles[f"{names['L']}_cholesky_corr__"] = f"L_{g} on Stan's cholesky_corr coordinates (canonical partial correlations, arctanh scale)"
        for g, names in self.group_dist_names.items():       # G05b: gr(g, dist = "student")'s df and udf
            roles[f"{names['df']}_interval__"] = f"log(df_{g} - 1), the degrees of freedom of the Student-t effects of {g} (brms's df)"
            roles[f"{names['udf']}_log__"] = f"log(udf_{g}[j]), the per-level scales (brms's udf; vector, dims {names['coordinate']})"
        if fam.scale_parameter is not None and fam.scale_parameter not in s.dpar_names:
            roles[f"{n[fam.scale_parameter]}_log__"] = f"log({fam.scale_parameter})"
        for ds in s.dpars:                                   # a distributional formula's variables (B01); scalar ones read b_sigma_x
            dn = self.dpar_names[ds.response]
            if ds.intercept:
                roles[dn["intercept"]] = f"a_{ds.response}{'_c' if d.dpars[ds.response].centered else ''} (the intercept of {ds.response}'s formula)"
            for k, v in dn.items():                         # G10: the formula's own simplexes
                if k.startswith("simo:"):
                    roles[f"{v}_stickbreaking__"] = f"{v} on Stan's stick-breaking simplex coordinates (brms's simo_{ds.response}); dims {v}_dim"
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
        if s.sp_terms:                                       # G10: each special product's coefficient and simplexes
            from autobayes.special_terms import simplexes as sp_simplexes
            sn = self.sp_names
            roles.update({v: f"b_{k}, the special term's coefficient (brms's bsp)" for k, v in sn["b"].items()})
            roles.update({f"{sn['simplex'][k]}_stickbreaking__": f"{sn['simplex'][k]}[1..{m.D}] on Stan's stick-breaking simplex "
                          f"coordinates (brms's simo); dims {sn['coordinate'][k]}" for k, (m, _, _, _) in sp_simplexes(s).items()})
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
        roles.update(car_describe(self, "roles"))            # F02a: a CAR term's value variables
        for part, order in (s.autocor.parts if s.autocor is not None else ()):   # C04: ar and ma, unconstrained vectors
            acn = self.autocor_names
            roles[acn[part]] = (f"{part}[1..{order}], the {_AUTOCOR_MEANING[part]} coefficients (brms's {part}; vector, dims "
                                f"{acn[part + '_coordinate']}, unconstrained)")
        if s.sar is not None:                                # F02b: lagsar / errorsar on its interval coordinate
            sd_ = d.sar
            roles[f"{sd_.parameter}_interval__"] = (f"log(({sd_.parameter} - lower) / (upper - {sd_.parameter})), lower and upper the "
                                                    f"smallest and largest eigenvalue of {sd_.matrix} (Stan's lower-upper coordinate)")
        roles.update(self._timecor_roles())                  # F03b: Lcortime's cholesky_corr coordinates, or logit(cosy)
        if d.bhaz is not None:                               # F05: the cox baseline hazard's simplex
            from autobayes.baseline_hazard import value_roles
            roles.update(value_roles(self))
        shrinkage_note = ""
        if self.shrinkage_names:                             # D07: zb and the shrinkage prior's scales; b, sdb deterministic
            from autobayes.shrinkage import deterministic_note, value_roles
            roles.update(value_roles(self))
            shrinkage_note = deterministic_note(self)
        for name in self._described_value_names():
            lines.append(f"  {name:<24} {roles.get(name, f'b_{name}')}")
        lines.append("Deterministic: " + "".join(f"{names['group_effect']} = {effect(g) if c == 'Intercept' else slope_effect(g, c)}, "
                                                 for (g, c), names in self.group_coefficient_names.items())
                     + "".join(f"{names['group_effect']} = {_dpar_effect(name, g, c)}, " for (name, g, c), names in self.dpar_group_names.items())   # D08
                     + "".join(f"{names['cor']} = Cor_{g} = L_{g} L_{g}^T, " for g, names in self.correlation_names.items())
                     + "".join(f"{names['dfm']} = sqrt(df_{g} * udf_{g}), " for g, names in self.group_dist_names.items())   # G05b
                     + fam.deterministic_note
                     + "".join(f"; {dn['value']} = {_DPAR_INVERSE[self.design.dpars[name].link].format(name)}, one {name} per row"
                               for name, dn in self.dpar_names.items())
                     + "".join(f"; {me['latent']} = meanme_{x} + sdme_{x} * zme_{x}, one per row (brms's Xme)" for x, me in self.me_names.items())
                     + (f"; {self.timecor_names['cor']} = Lcortime Lcortime^T (brms's Cortime)" if "cor" in self.timecor_names else "")
                     + shrinkage_note
                     + (f"; {n['linpred']} = mu_i[k], one per row and threshold" if s.has_category_specific else ""))
        return "\n".join(lines)

    def _slope_value(self, t, c: str) -> str:
        """``describe()``'s covariate of the varying slope ``c`` of ``t`` at row ``i``: ``x_c,i`` for a column, as it has always
        read; G05a: a transform's value (``I(x^2)_i``), a product (``x_x,i * x_w,i``), a factor level's indicator (``1[z_i = q]``)
        and a monotonic term's ``mo(ord)_i``."""
        if c in t.monotonic_slopes:
            return f"{c}_i"
        term = next((pt for pt in self.spec.pop_terms if c in pt.columns and pt.kind != "numeric"), None)
        if term is None:
            return f"x_{c},i"
        if term.kind == "transform":
            return f"{c}_i"
        if term.kind == "interaction":
            return " * ".join(f"x_{o},i" for o in term.operands)
        return f"1[{term.name}_i = {c[len(term.name) + 1:-1]}]"

    def _group_syntax_lines(self, t) -> list[str]:
        """G05a: ``describe()``'s data lines of a group-level term written in several terms, without its varying intercept, or
        with a slope that is no population-level term, a factor, a transform, a product or a monotonic term
        (docs/group-term-syntax.md); none for any other term."""
        g, out = t.factor, []
        if t.blocks:
            out.append(f"  {g:<10}: written as {len(t.blocks)} terms, {t.canonical}: brms's {len(t.blocks)} group IDs of {g}, each with its "
                       f"own SD(s) and no correlation with the other(s), so the coefficients {', '.join(t.coefs)} vary independently")
        if not t.has_intercept:
            out.append(f"  {g:<10}: {t.canonical} lets the slope(s) alone vary: the intercept does not vary by {g} (brms's term without "
                       "its varying intercept)")
        for c in t.slope_columns:
            out.append(f"  {c:<10}: varies by {g} without a population-level term: no b_{c} (its population-level slope is 0); each "
                       f"level's effect multiplies the raw column {c} (brms's Z)")
        for w in (w for w in t.written if w in self.spec.categorical_names):
            cat, coefs = self.spec.categorical(w), [c for c in t.coefs if t.term_of(c) == w]
            out.append(f"  {w:<10}: a factor slope of {g}: one varying coefficient per non-reference level, {', '.join(coefs)}, each "
                       f"multiplying the population design's indicator 1[{w}_i = l] (treatment contrasts against {cat.reference!r})")
        for c in t.slopes:
            if any(c in pt.columns and pt.kind in ("transform", "interaction") for pt in self.spec.pop_terms):
                out.append(f"  {c:<10}: the slope of the population-level term {c} varies by {g}, multiplying its design column "
                           f"{self._slope_value(t, c)}")
        for c in t.monotonic_slopes:
            column = self.spec.pop_term(c).operands[0]
            out.append(f"  {c:<10}: the monotonic term's slope varies by {g}: row i adds (b_{c} + v_{c}|{g}[{g}_i]) * mo(simo_{column}, "
                       f"{column}_i), one simplex shared with the population-level term and no design column (brms's (bsp + r) * "
                       "mo(simo, Xmo))")
        return out

    def _slope_only_lines(self, t, block) -> list[str]:
        """G05a: ``describe()``'s group-effect lines of a term whose intercept does not vary, ``(0 + x | g)``."""
        g = t.factor
        if not t.has_correlation:
            return [f"  v_{c}|{g}[j] = sd_{c}|{g} * z_{c}|{g}[j], z_{c}|{g}[j] ~ Normal(0, 1), j = 0..{block.J - 1}: the slope of {c} "
                    f"varies by {g}{' independently of the other slopes' if len(t.slopes) > 1 else ''}; the intercept does not"
                    for c in t.slopes]
        vec, sds = ", ".join(f"v_{c}|{g}[j]" for c in t.slopes), ", ".join(f"sd_{c}|{g}" for c in t.slopes)
        return [f"  ({vec}) = diag({sds}) L_{g} ({', '.join(f'z_{c}|{g}[j]' for c in t.slopes)}), z ~ Normal(0, 1), j = 0..{block.J - 1}: "
                f"the slopes vary by {g} with the correlation matrix Cor_{g} = L_{g} L_{g}^T of ({', '.join(t.slopes)}) (brms's "
                f"scale_r_cor; PyMC {self._correlation_roles(g)['cor']}); the intercept does not vary by {g}"]
    def _custom_data_lines(self, t) -> list[str]:
        """G05b, ``describe()``'s data lines of a term the model builds itself: a known covariance's matrix and levels, a Student-t
        term's construction, and each varying slope of a multi-membership term with its covariate per member."""
        lines = []
        if t.cov is not None:
            cov = self.design.group_cov[t.factor]
            lines += [f"  {t.factor:<10}: gr({t.factor}, cov = {t.cov}): the levels' known covariance data2[{t.cov!r}] on the "
                      f"{len(cov.levels)} fitted levels, Lcov_{t.factor} its lower Cholesky factor (brms's Lcov_m)"
                      + (f"; its labels {list(cov.unused)} are no fitted level and are dropped" if cov.unused else "")]
        if t.is_student:
            lines += [f"  {t.factor:<10}: gr({t.factor}, dist = \"student\"): each level's effects are Student-t, with the degrees of "
                      f"freedom df_{t.factor} (brms's df_m and udf_m)"]
        for coef in (t.slopes if t.is_mm else ()):
            cols = t.covariate_columns(coef)
            lines += [f"  {coef:<10}: the varying slope of {coef} in {t.grouping} multiplies, for each member, "
                      + ", ".join(f"{col!r} for {m!r}" for col, m in zip(cols, t.members)) + " (brms's Z_m_c_k)"]
        return lines

    def _custom_effect_lines(self, t, J: int) -> list[str]:
        """G05b, ``describe()``'s group-effect lines of a term the model builds itself: a multi-membership term's row sum with its
        slopes or raw weights, a known covariance's ``Lcov`` and a Student-t term's scales."""
        g, lines = t.factor, []
        if t.is_mm and (t.slopes or t.unscaled):
            coefs = "".join(f" + Z_{c},k,i * v_{c}|{g}[level k of row i]" for c in t.slopes)
            lines += [f"  Row i adds sum_k W_k,i * (u_{g}[level k of row i]{coefs}) over its members {', '.join(t.members)}"
                      + (", the weights as given (brms's scale = FALSE)" if t.unscaled else ", the weights summing to 1")
                      + (", Z_c,k,i coefficient c's covariate for member k (brms's Z_m_c_k)" if t.slopes else "") + "."]
        if t.cov is not None:
            effects = ", ".join([f"u_{g}"] + [f"v_{c}|{g}" for c in t.slopes])
            lines += [f"  Known covariance: each coefficient's effects over the {J} levels ({effects}) are Lcov_{g} times the effects "
                      f"written here without it (brms's Lcov_m), so u_{g} has covariance sd_{g}^2 A"
                      + (f" and vec(r) = (Lcov_{g} kron diag(sd) L_{g}) vec(z) has covariance A kron Sigma, Sigma the coefficients' "
                         "covariance (brms 2.23.0 drops the negative entries of Lcov here; divergence D185)" if t.slopes else "") + "."]
        if t.is_student:
            lines += [f"  Student-t effects: each level's effects are multiplied by dfm_{g}[j] = sqrt(df_{g} * udf_{g}[j]), udf_{g}[j] ~ "
                      f"InverseGamma(df_{g} / 2, 1 / 2) (Stan's inv_chi_square(df_{g}), brms's udf_m), so u_{g}[j] ~ Student-t(df_{g}, 0, "
                      f"sd_{g}), j = 0..{J - 1} (PyMC dfm|{factor_column(g)})"
                      + ("; each level's vector of effects is multivariate Student-t" if t.slopes else "") + "."]
        return lines

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
            x, sdx = t.operands[:2]
            me = self.me_names[x]
            if "coordinate" in me:                           # G10: me(x, sdx, gr = g), docs/special-term-ext.md
                g = t.operands[2]
                out += [f"  Xme_{x},j = meanme_{x} + sdme_{x} * zme_{x},j, zme_{x},j ~ Normal(0, 1): the latent (true) value of {x} in "
                        f"level j of {g} (brms's Xme with gr = {g}), which every row of the level shares, and {x}_j ~ Normal(Xme_{x},j, "
                        f"{sdx}_j), the level's noisy value observed with its known standard deviation (PyMC's observed {me['noisy']}).",
                        f"  b_{t.name} (brms's bsp) is the change of {lp} per unit of the latent {x}; row i reads its level's Xme_{x},j "
                        "(brms's Jme); not centred."]
                continue
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

    def _timecor_data_lines(self) -> list[str]:
        """``describe()``'s data line of a residual correlation term (F03b): its groups, their sizes and the times."""
        term, d = self.spec.timecor, self.design.timecor
        if term is None:
            return []
        rows = (f"its times {', '.join(d.labels)} (brms's n_unique_t = {d.dim}: the rows and columns of Cortime)" if term.call == "unstr"
                else f"its positions 1..{d.dim} in time order (the largest group)")
        return [f"  timecor   : {term.term}: {len(d.groups)} group(s) of {term.gr!r} of {d.nobs.min()} to {d.nobs.max()} rows, each "
                f"ordered by {term.time!r}; each row is {rows}; {len(d.patterns)} distinct set(s) of rows of the correlation "
                "matrix; brms's Stan data list the rows sorted by group and then time (N_tg, begin_tg, nobs_tg)"]

    def _timecor_lines(self) -> list[str]:
        """``describe()``'s paragraph of a residual correlation term (F03b): the joint density of each group's rows."""
        term = self.spec.timecor
        if term is None:
            return []
        kernel = self.family_spec.time_correlation()
        nu = "nu, " if kernel.parameters else ""
        cor = ("Cortime[J_g, J_g], the rows and columns of the group's times J_g in Cortime (unstr: one correlation per pair of "
               "times)" if term.call == "unstr" else
               "the compound-symmetric matrix of the group's size, 1 on the diagonal and cosy everywhere else (cosy)")
        return ["", f"Residual correlation ({term.term}; brms {term.call}())",
                f"  The rows of different groups of {term.gr!r} are independent; the residuals e_g = y_g - mu_g of group g are jointly",
                f"  e_g ~ {kernel.name}({nu}0, sigma^2 Cor_g), Cor_g = {cor},",
                "  so each row is marginally the likelihood above. log_lik() is each row's density given its group's other rows,",
                "  as brms's log_lik() of this non-factorised model is."]

    def _timecor_prior_lines(self) -> list[str]:
        """``describe()``'s prior line of a residual correlation term's parameter (F03b)."""
        term = self.spec.timecor
        if term is None:
            return []
        prior = self.priors.timecor_priors(self.spec)[term.prior_class]
        if term.call == "unstr":
            return [f"  Cortime ~ {prior} (brms lkj({prior.params['eta']:g}) on the correlation matrix, class cortime, placed on "
                    "Lcortime as lkj_corr_cholesky)" + ("   (brms default)" if prior.source == "default" else "")]
        text = ("Flat(): uniform on (0, 1), proper (brms's flat prior on its real<lower=0,upper=1>)" if prior.name == "Flat"
                else str(prior))
        return [f"  cosy ~ {text}   (brms class cosy, lb 0, ub 1)" + ("   (brms default)" if prior.source == "default" else "")]

    def _timecor_roles(self) -> dict[str, str]:
        """``describe()``'s value variables of a residual correlation term (F03b)."""
        roles = self.timecor_names
        if not roles:
            return {}
        value = self.pymc_model.rvs_to_values[self.pymc_model[roles.get("L", roles.get("cosy"))]].name
        if "L" in roles:
            return {value: f"Lcortime, the Cholesky factor of Cortime, on Stan's cholesky_corr coordinates (canonical partial "
                           f"correlations, arctanh scale); dims {roles['coordinate']} x {roles['coordinate_T']}"}
        return {value: "logit(cosy), the compound-symmetric correlation (brms's cosy)"}

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
        groups = "".join(f" + {_dpar_effect(name, t.factor, c)}[{t.factor}_i]" + ("" if c == "Intercept" else f" * {c}_i")
                         for t in ds.group_terms for c in t.coefs)   # D08: a shared ID's varying intercept; G06: own terms, slopes
        groups += f" + offset_i ({ds.offset.term})" if ds.offset is not None else ""   # G06: the formula's offset
        # how the link reads (B01's log; E03: zi and hu under the logit, alpha under the identity): the left-hand side, row i's
        # value from its linear predictor, and the scale the coefficients act on
        left, value, on = {"log": (f"log({name}_i)", f"exp(eta_{name},i) > 0", f"log {name}"),
                           "logit": (f"logit({name}_i)", f"inv_logit(eta_{name},i), in (0, 1),", f"logit {name}"),
                           "identity": (f"{name}_i", f"eta_{name},i", f"{name} itself"),
                           "logm1": (f"log({name}_i - 1)", f"1 + exp(eta_{name},i) > 1", f"log({name} - 1)")}[dd.link]   # G06: nu
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
            for c in t.coefs:
                roles = self.dpar_group_names[(name, t.factor, c)]
                what = "varying intercept" if c == "Intercept" else f"varying slope of {c}"
                if t.id is not None:
                    tail = f"it shares the correlation ID {t.id!r} with the response formula's term (Group effects, below)."
                else:                                        # G06: brms's group of the parameter, uncorrelated with every other
                    tail = (f"{name}'s own group (brms's r_<id>_{name}_1), uncorrelated with every other: "
                            f"{_dpar_effect(name, t.factor, c)}[j] = sd_{name},{t.factor} * z_{name},{t.factor}[j].")
                lines += [f"  {_dpar_effect(name, t.factor, c)}[j] (PyMC {roles['group_effect']}) is the {what} of {on} by {t.factor}, "
                          f"'{t.canonical}': {tail}"]
        if ds.offset is not None:                            # G06: brms's offsets_<dpar>, no coefficient
            lines += [f"  offset_i is {ds.offset.expression} at row i (brms's offsets_{name}), added with no coefficient."]
        return lines

    def _addition_lines(self) -> list[str]:
        """``describe()``'s data lines for the response addition terms (B02, docs/addition-terms.md)."""
        d, out = self.design, []
        for a in self.spec.additions:
            if a.kind in ("mi", "thres", "bhaz"):            # C06: the response's missing rows, _measurement_data_lines; F06b: the
                continue                                     # levels of thres(gr = z), grouped_thresholds.data_lines; F05: the cox
                                                             # family's lines say what bhaz() set
            if a.kind == "cat":                              # F08: brms's cat(K), the data's own number of categories
                out += [f"  categories: {a.term}: the response's {a.count} categories, the data's own (brms's deprecated cat(K), its "
                        f"thres({a.count - 1})); it changes nothing"]
                continue
            if a.kind == "dec":                              # F04: the wiener decisions (brms's dec)
                upper = int((d.dec == 1).sum())
                out += [f"  decisions : {a.term}: {upper} row(s) at the upper boundary (1), {d.n - upper} at the lower one (0); "
                        "predict() needs no decisions (the mean response time is over both boundaries)"]
                continue
            if a.kind in ("se", "rate"):                     # F06a: brms's se and denom
                from autobayes.se_rate import data_lines
                out += data_lines(self, a)
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
                lb, ub = ("0" if a.lb is None else f"{max(a.lb, 0):g}"), ("inf" if a.ub is None else f"{a.ub:g}")
                mean = "the truncated mean"
                if a.lb is not None and a.lb >= 0:           # D121: brms's posterior_epred moves a lower bound up by one
                    ours, theirs = ((f"{lb} <= Y <= {ub}", f"{a.lb + 1:g} <= Y <= {ub}") if a.ub is not None
                                    else (f"Y >= {lb}", f"Y >= {a.lb + 1:g}"))
                    mean += f" E[Y | {ours}] (brms's posterior_epred is E[Y | {theirs}] instead: divergence D121)"
                out += [f"  truncation: {a.term}: the counts lie in {{{lb}, ..., {ub}}}, both bounds included; each row's mass is "
                        f"divided by the probability of that range (brms trunc()); predict(kind='mean') is {mean} and "
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
            if a.kind in ("dec", "thres", "cat") or a.kind in self.family_spec.own_additions:   # F04: the wiener likelihood's own lines
                continue                                     # say how it reads dec; F06b: the grouped cumulative likelihood's how it
                                                             # reads thres(); F05: the cox likelihood's its cens() and bhaz()
            if a.kind in ("se", "rate"):                     # F06a
                from autobayes.se_rate import likelihood_lines
                out += likelihood_lines(self, a)
                continue
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
        found += [(f"df:{g}", v) for g, v in pr.df_priors(s).items()]              # gr(g, dist = "student")'s df
        found += [(f"simo:{c}", v) for c, v in pr.simo_priors(s).items()]          # a monotonic term's simplex (B03)
        found += [(f"sdgp:{c}", v) for c, v in pr.sdgp_priors(s).items()]          # a Gaussian process's SD and length-scale (C03)
        found += [(f"lscale:{c}", v) for c, v in pr.lscale_priors(s).items()]
        found += list(pr.autocor_priors(s).items())                                  # an ARMA term's ar and ma (C04)
        found += list(pr.car_priors(s).items())                                      # a CAR term's sdcar, car, rhocar (F02a)
        found += list(pr.timecor_priors(s).items())                                  # cortime or cosy (F03b)
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
        if fam.simplex is not None and fam.simplex.own_priors:   # F01b: the logistic_normal family's sigma<k> and lncor
            found += [("sigma", pr.sigma), ("lncor", pr.lncor)]
        if fam.baseline_hazard:                              # F05: the cox baseline hazard's simplex
            found.append(("sbhaz", pr.sbhaz))
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
             loop: bool = True, data2: Mapping[str, object] | None = None, dropna: bool = False) -> "Model | MultivariateModel":
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
    ``data2`` (Tier F) holds the matrices the formula reads by name, brms's ``data2``: ``y ~ x + sar(M, type = "lag")`` with
    ``data2={"M": matrix}``, the weight matrix over the rows in data order, and ``Priors(..., lagsar=ab.Flat())`` (F02b,
    docs/spatial-sar.md), ``y ~ x + fcor(M)``, the residual covariance (F03a, docs/fcor.md), or ``y ~ x + car(W, gr = site)``, the
    adjacency of the locations (F02a, docs/spatial-car.md); a non-linear or multivariate model takes none (divergences D141,
    D143, D144)."""
    from autobayes.multivariate import MultivariateModel, is_multivariate_formula     # C01

    if data2 is not None and (nl or rescor is not None or is_multivariate_formula(formula)):   # F02b, F03a
        raise FormulaError(DATA2_UNIVARIATE)
    if nl:                                                             # C02: a non-linear model
        from autobayes.nonlinear import NonlinearModel

        if rescor is not None:
            raise FormulaError("rescor= belongs to a multivariate model, 'mvbind(y, y2) ~ ...'; a non-linear model in a multivariate "
                               "formula is not supported (docs/nonlinear.md, divergence D62)")
        if dropna:                                                     # D174: no frozen case verifies it (D177)
            raise ValueError("dropna=True is not offered for a non-linear model (docs/missing-rows.md, divergence D177): drop the rows "
                             "with missing values from the data first")
        return NonlinearModel(formula, data, priors, family="gaussian", center_predictors=center_predictors,
                              categorical=categorical, loop=loop)
    if loop is not True:
        raise ValueError("loop is brms's option of a non-linear formula; it applies with nl=True only (docs/nonlinear.md)")
    if rescor is not None or is_multivariate_formula(formula):
        return MultivariateModel(formula, data, priors, family="gaussian", rescor=rescor, center_predictors=center_predictors,
                                 categorical=categorical, dropna=dropna)
    return Model(formula, data, priors, family="gaussian", center_predictors=center_predictors, categorical=categorical, data2=data2,
                 dropna=dropna)


def bernoulli(formula: str, data: pd.DataFrame, priors: Priors, *, link: str = "logit", center_predictors: bool = True,
              categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Build the Bernoulli model (response strictly 0/1), with or without a varying intercept.

    ``link`` is ``"logit"`` (``p = expit(eta)``, the default), ``"probit"`` (``p = Phi(eta)``) or ``"cloglog"``
    (``p = 1 - exp(-exp(eta))``), brms's ``bernoulli(link = ...)``; the intercept and slope priors are on that link's
    scale. brms's ``cauchit`` link is not offered (docs/links.md).
    """
    return Model(formula, data, priors, family="bernoulli", link=link, center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def binomial(formula: str, data: pd.DataFrame, priors: Priors, *, link: str = "logit", center_predictors: bool = True,
             categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Build the binomial model ``successes | trials(n) ~ x [+ (1 | group)]``, with or without a varying intercept;
    ``link`` as for ``bernoulli`` (``"logit"``, ``"probit"`` or ``"cloglog"``)."""
    return Model(formula, data, priors, family="binomial", link=link, center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def student(formula: str | list[str], data: pd.DataFrame, priors: Priors, *, nu: float | None = None,
            center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Student-t response with an explicit sigma scale prior and degrees of freedom ``nu``, fixed or learned.

    ``nu=4`` (any finite ``nu > 1``) fixes them. ``nu=None``, the default, learns them under ``Priors(nu=...)``: a
    ``Gamma``, ``Exponential``, ``HalfNormal`` or ``HalfCauchy`` prior truncated to ``nu > 1`` and renormalised there, as
    brms's ``real<lower=1> nu`` (brms's default prior is ``gamma(2, 0.1)``, ``ab.Gamma(2, 0.1)``, which is not filled in
    silently; ``ab.default_priors(..., family="student")`` returns it with brms's other defaults; A13,
    docs/student-learned-nu.md). The expected response is mu. Sigma is not the residual standard deviation. ``sigma`` may
    have its own formula under a log link (B01, ``["y ~ x", "sigma ~ z"]``, docs/distributional.md); ``nu`` takes none.
    """
    return Model(formula, data, priors, family="student", nu=nu,
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def poisson(formula: str, data: pd.DataFrame, priors: Priors, *, link: str = "log",
            center_predictors: bool = True, categorical: Mapping[str, object] | None = None,
            data2: Mapping | None = None, dropna: bool = False) -> Model:
    """Poisson count response with log link and explicit coefficient/group priors.

    Retains zero rows. Input counts must be integers in [0, 1e6]. An exposure enters as
    ``offset(log(exposure))`` (docs/offset.md). The Poisson family has no sigma, nu or overdispersion parameter
    (``ab.negbinomial`` has one). ``link="sqrt"`` is brms's ``poisson("sqrt")``, ``mu = eta^2`` (E06, docs/links.md).
    ``data2`` (F02a) holds ``car(W, gr = site)``'s adjacency matrix, ``data2={"W": W}`` (docs/spatial-car.md).
    """
    return Model(formula, data, priors, family="poisson", link=link,
                 center_predictors=center_predictors, categorical=categorical, data2=data2, dropna=dropna)


def negbinomial(formula: str | list[str], data: pd.DataFrame, priors: Priors, *, link: str = "log",
                center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Negative binomial count response (brms ``negbinomial``): log link, ``Var[y] = mu + mu^2 / shape``.

    Counts as for Poisson (integers in [0, 1e6], zero rows retained). The overdispersion parameter ``shape``
    takes its own positive-support prior, ``Priors(shape=ab.Gamma(0.01, 0.01))``, or its own formula under a log link (B01,
    ``["y ~ x", "shape ~ z"]``, docs/distributional.md); no sigma, nu, offset/exposure or zero-inflation.
    ``link="identity"`` is brms's ``negbinomial("identity")``, ``mu = eta``, which needs a positive linear predictor (E06,
    docs/links.md).
    """
    return Model(formula, data, priors, family="negbinomial", link=link,
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def gamma(formula: str | list[str], data: pd.DataFrame, priors: Priors, *, link: str = "log",
          center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Gamma response with a log link (brms ``Gamma(link = "log")``): ``E[y] = mu = exp(eta)``, ``Var[y] = mu^2 / shape``.

    Responses are strictly positive finite reals. The shape parameter takes its own positive-support prior,
    ``Priors(shape=ab.Gamma(0.01, 0.01))``, or its own formula under a log link (B01, ``["y ~ x", "shape ~ z"]``,
    docs/distributional.md); no sigma or nu. ``link`` is ``"log"`` by default, as brms's ``brmsfamily("Gamma")``; R's
    ``stats::Gamma()``, which brms takes as ``family = Gamma()``, defaults to the inverse link, so a model ported from it needs
    ``link="inverse"`` (docs/brms-parity/divergences.md, D9). ``"inverse"`` (``mu = 1 / eta``) and ``"identity"`` (``mu = eta``)
    are brms's ``Gamma("inverse")`` and ``Gamma("identity")``, which need a positive linear predictor (E06, docs/links.md).
    """
    return Model(formula, data, priors, family="gamma", link=link,
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def lognormal(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
              center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Lognormal response (brms ``lognormal()``): ``log(y) ~ Normal(mu, sigma)`` with the identity link on the log-scale
    location ``mu``; ``E[y] = exp(mu + sigma^2 / 2)``.

    Responses are strictly positive finite reals. ``sigma``, the SD of ``log y``, takes its own positive-support prior,
    ``Priors(sigma=ab.HalfNormal(1))``; the intercept and slope priors act on ``log y``. ``predict(kind="mean")`` returns
    brms's ``posterior_epred`` ``exp(mu + sigma^2 / 2)`` as ``mean``; ``kind="linear"`` returns ``mu``. ``sigma`` may have its
    own formula under a log link (B01, ``["y ~ x", "sigma ~ z"]``, docs/distributional.md), and the mean then uses each
    row's ``sigma``. No shape, nu or offset.
    """
    return Model(formula, data, priors, family="lognormal",
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def beta(formula: str | list[str], data: pd.DataFrame, priors: Priors, *, link: str = "logit",
         center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Beta response with a logit link (brms ``Beta()``): proportions strictly inside (0, 1), ``E[y] = mu = expit(eta)``,
    ``Var[y] = mu (1 - mu) / (1 + phi)``.

    The precision ``phi`` takes its own positive-support prior, ``Priors(phi=ab.Gamma(0.01, 0.01))`` (brms class
    ``phi``), or its own formula under a log link (B01, ``["y ~ x", "phi ~ z"]``, docs/distributional.md); the intercept and
    slope priors are on the logit scale. ``predict(kind="mean")`` returns ``mu``; no sigma, shape, nu or zero/one inflation.
    ``link="probit"`` (``mu = Phi(eta)``) and ``link="cloglog"`` (``mu = 1 - exp(-exp(eta))``) are brms's ``Beta("probit")`` and
    ``Beta("cloglog")``, the priors then on that link's scale (E06, docs/links.md).
    """
    return Model(formula, data, priors, family="beta", link=link,
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def weibull(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
            center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Weibull response (brms ``weibull()``, D01, docs/continuous-families.md): log link on the mean, ``E[y] = mu = exp(eta)``,
    ``y ~ Weibull(shape, scale = mu / Gamma(1 + 1 / shape))``.

    Responses are strictly positive finite reals. The shape takes its own positive-support prior,
    ``Priors(shape=ab.Gamma(0.01, 0.01))`` (brms class ``shape``; brms's default, on request through ``ab.default_priors``).
    ``predict(kind="mean")`` returns ``mu``. E03 (docs/contexts-newer-families.md): ``shape`` may have its own formula under a
    log link, ``["ypos ~ x", "shape ~ z"]`` with ``Priors(shape={"intercept": ..., "b": ...})``, and ``weights()`` is taken.
    E04 (docs/cens-trunc-families.md): ``cens(c)`` and ``cens(c, y2)`` (survival data); no ``trunc()`` (D122).
    """
    return Model(formula, data, priors, family="weibull", center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def exponential(formula: str, data: pd.DataFrame, priors: Priors, *,
                center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Exponential response (brms ``exponential()``, D01, docs/continuous-families.md): log link on the mean,
    ``E[y] = mu = exp(eta)``, ``y ~ Exponential(rate = 1 / mu)``; no auxiliary parameter.

    Responses are strictly positive finite reals. ``predict(kind="mean")`` returns ``mu``. No ``weights()``, ``cens()`` or
    ``trunc()`` (D79).
    """
    return Model(formula, data, priors, family="exponential", center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def skew_normal(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
                center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Skew-normal response (brms ``skew_normal()``, D01, docs/continuous-families.md): identity link on the mean ``mu``,
    standard deviation ``sigma`` and skewness ``alpha`` (Azzalini's skew normal in brms's mean parameterisation).

    ``sigma`` takes a positive-support prior, ``Priors(sigma=ab.HalfNormal(1))``, and ``alpha`` a location prior,
    ``Priors(alpha=ab.Normal(0, 4))`` (brms class ``alpha``; its default, on request through ``ab.default_priors``).
    ``predict(kind="mean")`` returns ``mu``. E03 (docs/contexts-newer-families.md): ``alpha`` may have its own formula under the
    identity link, ``["y ~ x", "alpha ~ z"]`` with ``Priors(alpha={"intercept": ..., "b": ...})``; no formula on ``sigma``, no
    ``weights()``, ``cens()`` or ``trunc()`` (D79).
    """
    return Model(formula, data, priors, family="skew_normal", center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def shifted_lognormal(formula: str, data: pd.DataFrame, priors: Priors, *,
                      center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Shifted lognormal response (brms ``shifted_lognormal()``, D01, docs/continuous-families.md): ``log(y - ndt) ~
    Normal(mu, sigma)`` with the identity link on ``mu`` and the non-decision time ``0 < ndt < min(y)``;
    ``E[y] = exp(mu + sigma^2 / 2) + ndt``.

    Responses are strictly positive finite reals. ``sigma`` takes a positive-support prior; ``ndt`` a prior truncated to
    (0, min(y)) and renormalised there, as brms does (``Priors(ndt=ab.Normal(0, 0.3))``), or brms's default
    ``uniform(0, min_Y)``, ``Priors(ndt=ab.Uniform(0, "min_Y"))``. ``predict(kind="mean")`` returns brms's ``posterior_epred``
    as ``mean``; ``kind="linear"`` returns ``mu``. No distributional formula, ``weights()``, ``cens()`` or ``trunc()`` (D79).
    """
    return Model(formula, data, priors, family="shifted_lognormal", center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def exgaussian(formula: str, data: pd.DataFrame, priors: Priors, *,
               center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Ex-Gaussian response (brms ``exgaussian()``, D01, docs/continuous-families.md): ``Normal(mu - beta, sigma)`` plus an
    exponential of mean ``beta``, with the identity link on the mean ``mu``.

    ``sigma`` and ``beta`` take positive-support priors, ``Priors(sigma=ab.HalfNormal(1), beta=ab.Gamma(1, 0.1))`` (brms
    classes ``sigma`` and ``beta``; ``gamma(1, 0.1)`` is brms's default for ``beta``). ``predict(kind="mean")`` returns
    ``mu``. No distributional formula, ``weights()``, ``cens()`` or ``trunc()`` (D79).
    """
    return Model(formula, data, priors, family="exgaussian", center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def beta_binomial(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
                  center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Beta-binomial response (brms ``beta_binomial()``, D02, docs/discrete-families.md): ``successes | trials(n) ~ ...`` with the
    logit link on the mean success probability ``mu`` and the precision ``phi``, ``k ~ BetaBinomial(n, mu phi, (1 - mu) phi)``.

    Successes and trials as for the binomial family (whole numbers up to 1e6, zero-trial rows dropped). ``phi`` takes a
    positive-support prior, ``Priors(phi=ab.Gamma(0.01, 0.01))`` (brms class ``phi``; brms's default, on request through
    ``ab.default_priors``). ``predict(kind="mean")`` returns ``mu`` and ``expected = n mu`` (brms's ``posterior_epred``),
    ``kind="probability"`` ``mu`` alone. E03 (docs/contexts-newer-families.md): ``phi`` may have its own formula under a log link,
    ``["k | trials(size) ~ x", "phi ~ z"]`` with ``Priors(phi={"intercept": ..., "b": ...})``, and ``weights()`` is taken after
    ``trials()``; no other link (D80).
    """
    return Model(formula, data, priors, family="beta_binomial", center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def geometric(formula: str, data: pd.DataFrame, priors: Priors, *,
              center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Geometric count response (brms ``geometric()``, D02, docs/discrete-families.md): log link on the mean,
    ``y ~ NegativeBinomial(mu, 1)``, ``Var[y] = mu + mu^2``; no auxiliary parameter.

    Counts as for Poisson (integers in [0, 1e6], zero rows retained). ``predict(kind="mean")`` returns ``mu``. No
    ``weights()`` or other link (D80, D81).
    """
    return Model(formula, data, priors, family="geometric", center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def negbinomial2(formula: str, data: pd.DataFrame, priors: Priors, *,
                 center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Negative binomial count response parameterised by its dispersion (brms ``negbinomial2()``, D02,
    docs/discrete-families.md): log link on the mean, ``y ~ NegativeBinomial(mu, shape = 1 / sigma)``,
    ``Var[y] = mu + sigma mu^2``.

    Counts as for Poisson. The dispersion ``sigma`` (brms's ``1 / shape`` of ``negbinomial``) takes a positive-support prior,
    ``Priors(sigma=ab.Gamma(0.4, 0.3))`` (brms class ``sigma``; brms's default, on request through ``ab.default_priors``).
    ``predict(kind="mean")`` returns ``mu``. No distributional formula, ``weights()`` or other link (D80, D81).
    """
    return Model(formula, data, priors, family="negbinomial2", center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def inverse_gaussian(formula: str, data: pd.DataFrame, priors: Priors, *, link: str = "1/mu^2",
                     center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Inverse Gaussian response (brms ``inverse.gaussian()``, E02, docs/continuous-families.md): mean ``mu`` and shape
    ``shape``, ``Var[y] = mu^3 / shape``, under brms's default link ``"1/mu^2"`` (``mu = 1 / sqrt(eta)``, which needs a positive
    linear predictor: the density is -inf where it is not) or ``link="log"`` (``mu = exp(eta)``).

    Responses are strictly positive finite reals. The shape takes a positive-support prior, ``Priors(shape=ab.Gamma(0.01,
    0.01))`` (brms class ``shape``; brms's default). ``predict(kind="mean")`` returns ``mu``. No ``inverse`` or ``identity`` link,
    distributional formula or addition term (D112).
    """
    return Model(formula, data, priors, family="inverse_gaussian", link=link, center_predictors=center_predictors,
                 categorical=categorical, dropna=dropna)


def von_mises(formula: str, data: pd.DataFrame, priors: Priors, *,
              center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """von Mises response (brms ``von_mises()``, E02, docs/continuous-families.md): angles in [-pi, pi], mean direction ``mu =
    2 atan(eta)`` (brms's ``tan_half`` link) and precision ``kappa``.

    ``kappa`` takes a positive-support prior, ``Priors(kappa=ab.Gamma(2, 0.01))`` (brms class ``kappa``; brms's default).
    ``predict(kind="mean")`` returns ``mu``. No ``identity`` link, distributional formula or addition term (D112).
    """
    return Model(formula, data, priors, family="von_mises", center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def asym_laplace(formula: str, data: pd.DataFrame, priors: Priors, *, quantile: float | None = None,
                 center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Asymmetric Laplace response (brms ``asym_laplace()``, E02, docs/continuous-families.md): quantile regression, the
    location ``mu`` (identity link) being the ``p``-quantile of ``y``, with scale ``sigma``.

    ``quantile=None`` learns the quantile under ``Priors(quantile=ab.Beta(...))`` (brms class ``quantile``, default ``beta(1,
    1)``); ``quantile=0.25`` fixes it (brms's ``bf(y ~ ..., quantile = 0.25)``), with no prior. ``sigma`` takes a
    positive-support prior. ``predict(kind="mean")`` returns brms's ``posterior_epred``, ``mu + sigma (1 - 2 p) / (p (1 - p))``,
    as ``mean``; ``kind="linear"`` returns ``mu``. No distributional formula or addition term (D112).
    """
    return Model(formula, data, priors, family="asym_laplace", quantile=quantile, center_predictors=center_predictors,
                 categorical=categorical, dropna=dropna)


def frechet(formula: str, data: pd.DataFrame, priors: Priors, *, nu: float | None = None,
            center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Frechet response (brms ``frechet()``, E02, docs/continuous-families.md): log link on the mean, ``E[y] = mu = exp(eta)``,
    scale ``mu / Gamma(1 - 1 / nu)`` and shape ``nu > 1``.

    Responses are strictly positive finite reals. ``nu=None`` learns the shape under ``Priors(nu=...)`` truncated to ``nu > 1``
    (brms's ``real<lower=1> nu``, default ``gamma(2, 0.1)``), as the Student-t family's ``nu``; ``nu=3`` fixes it.
    ``predict(kind="mean")`` returns ``mu``. No other link, distributional formula or addition term (D112).
    """
    return Model(formula, data, priors, family="frechet", nu=nu, center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def gen_extreme_value(formula: str, data: pd.DataFrame, priors: Priors, *,
                      center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Generalized extreme value response (brms ``gen_extreme_value()``, E02, docs/continuous-families.md): location ``mu``
    (identity link), scale ``sigma`` and shape ``xi``, derived as brms derives it from its unscaled ``tmp_xi`` so that every
    fitted row lies inside the support.

    ``sigma`` takes a positive-support prior, ``xi`` a location prior on ``tmp_xi``, ``Priors(xi=ab.Normal(0, 2.5))`` (brms class
    ``xi``; brms's default). ``predict(kind="mean")`` returns ``mu + sigma (Gamma(1 - xi) - 1) / xi`` as ``mean`` (infinite for
    ``xi >= 1``, D114); ``kind="linear"`` returns ``mu``. No other link, distributional formula or addition term (D112).
    """
    return Model(formula, data, priors, family="gen_extreme_value", center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def wiener(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
           center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Wiener diffusion model of response times and decisions (brms ``wiener()``, F04, docs/wiener.md): ``rt | dec(dec) ~ ...``
    with ``dec`` 1 for the upper boundary and 0 for the lower one, the identity link on the drift rate ``mu``, the boundary
    separation ``bs``, the non-decision time ``0 < ndt < min(rt)`` and the relative starting point ``0 < bias < 1``.

    ``bs`` takes a positive-support prior (brms's default ``gamma(1, 1)``), ``ndt`` a prior truncated to (0, min(rt)) and
    renormalised there, as for the shifted lognormal (``Uniform(0, "min_Y")`` is brms's default), and ``bias`` a prior on (0, 1)
    (brms's default ``beta(1, 1)``): ``Priors(bs=ab.Gamma(1, 1), ndt=ab.Normal(0, 0.3), bias=ab.Beta(2, 2))``. ``bs`` may have
    its own formula under the log link, ``["rt | dec(dec) ~ x", "bs ~ z"]`` with ``Priors(bs={"intercept": ..., "b": ...})``.
    ``predict(kind="mean")`` returns the mean response time ``ndt + E[T]`` as ``mean`` (not brms's posterior_epred, D148);
    ``kind="linear"`` returns ``mu``; response draws are refused (D149). No formula on ``ndt`` or ``bias``, other link or other
    addition term (D151).
    """
    return Model(formula, data, priors, family="wiener", center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def cox(formula: str, data: pd.DataFrame, priors: Priors, *,
        center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """brms's ``cox()`` proportional hazards model of survival times (F05, docs/cox.md): ``time | cens(c) ~ ...`` with ``c`` 0 for
    an event and 1 for a right-censored row (without ``cens()`` every row is an event), the hazard ``h0(t) exp(eta)`` (brms's
    log link) and the baseline hazard ``h0`` a cubic M-spline basis of the time with the simplex ``sbhaz`` of its weights, set by
    ``bhaz(df = 5)`` (brms's default), ``bhaz(knots = c(...))`` and ``bhaz(Boundary.knots = c(lo, hi))``.

    ``sbhaz`` takes a Dirichlet prior (brms's class ``sbhaz``, default ``dirichlet(1)``): ``Priors(intercept=..., b=...,
    sbhaz=ab.Dirichlet(1))``. ``predict(kind="linear")`` returns ``eta`` and ``kind="relative_hazard"`` ``mu = exp(eta)``; brms
    refuses its expected values and draws, and so does this model (D154); ``ab.log_lik`` and ``ab.loo`` are brms's. Interval
    censoring and ``trunc()`` (which brms computes wrongly, D153), left censoring, ``weights()``, ``bhaz(gr = )`` and other links
    are refused (D155).
    """
    return Model(formula, data, priors, family="cox", center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def zero_inflated_poisson(formula: str, data: pd.DataFrame, priors: Priors, *,
                          center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
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
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def hurdle_poisson(formula: str, data: pd.DataFrame, priors: Priors, *,
                   center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Hurdle Poisson counts (brms ``hurdle_poisson``): zero with probability ``hu``, a zero-truncated ``Poisson(mu)``
    count otherwise, ``mu = exp(eta)``; ``E[y] = (1 - hu) mu / (1 - exp(-mu))``.

    Counts as for Poisson. The hurdle probability takes its own prior on (0, 1), ``Priors(hu=ab.Beta(1, 1))`` (brms class
    ``hu``, brms's default ``beta(1, 1)`` on request through ``ab.default_priors``); the intercept and slope priors act on
    the log of the untruncated Poisson part's mean. ``predict(kind="mean")`` returns brms's expected count as ``mean``;
    ``kind="linear"`` returns ``eta``. ``weights()`` is taken (D03). No overdispersion or distributional (``hu ~ ...``)
    formula (docs/zero-inflated-hurdle.md).
    """
    return Model(formula, data, priors, family="hurdle_poisson",
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


# -- D03: more zero-inflated and hurdle families (docs/zero-inflated-hurdle.md) ---------------------------------------------
_D03_CONTEXTS = ("No distributional formula, cens(), trunc(), mo(), me(), mi(), mm(), gr(by = ), autocorrelation term, multivariate "
                 "or non-linear formula (divergence D84).")
_E03_CONTEXTS = _D03_CONTEXTS.replace("No distributional formula,", "No other distributional formula,")   # zi / hu formulas (E03)


def zero_inflated_negbinomial(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
                              center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Zero-inflated negative binomial counts (brms ``zero_inflated_negbinomial``, D03): a structural zero with probability
    ``zi``, a ``NB(mu, shape)`` count otherwise (mean ``mu = exp(eta)``, variance ``mu + mu^2 / shape``); ``E[y] = (1 - zi) mu``.

    Counts as for Poisson. ``Priors(shape=..., zi=ab.Beta(...))`` (brms classes ``shape`` and ``zi``; brms's defaults
    ``inv_gamma(0.4, 0.3)`` and ``beta(1, 1)`` on request through ``ab.default_priors``); the intercept and slope priors act on
    the log of the negative binomial part's mean. ``predict(kind="mean")`` returns ``(1 - zi) mu`` as ``mean``;
    ``kind="linear"`` returns ``eta``. ``weights()`` is taken. E03 (docs/contexts-newer-families.md): ``zi`` may have its own
    formula under the logit link, ``["zcnt ~ x", "zi ~ z"]`` with ``Priors(zi={"intercept": ..., "b": ...})``. """ + _E03_CONTEXTS
    return Model(formula, data, priors, family="zero_inflated_negbinomial",
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def hurdle_negbinomial(formula: str, data: pd.DataFrame, priors: Priors, *,
                       center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Hurdle negative binomial counts (brms ``hurdle_negbinomial``, D03): zero with probability ``hu``, a zero-truncated
    ``NB(mu, shape)`` count otherwise, ``mu = exp(eta)``; ``E[y] = (1 - hu) mu / (1 - (shape / (mu + shape))^shape)``.

    Counts as for Poisson. ``Priors(shape=..., hu=ab.Beta(...))``; the intercept and slope priors act on the log of the
    untruncated negative binomial part's mean. ``predict(kind="mean")`` returns brms's expected count as ``mean``;
    ``kind="response"`` draws exact zero-truncated negative binomial counts (D86). ``weights()`` is taken. """ + _D03_CONTEXTS
    return Model(formula, data, priors, family="hurdle_negbinomial",
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def zero_inflated_binomial(formula: str, data: pd.DataFrame, priors: Priors, *,
                           center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Zero-inflated binomial successes (brms ``zero_inflated_binomial``, D03): ``successes | trials(n) ~ ...``, a structural
    zero with probability ``zi``, a ``Binomial(n, p)`` count otherwise, ``p = expit(eta)``; ``E[y] = (1 - zi) n p``.

    Trials as for the binomial family (zero-trial rows dropped, D4). ``Priors(zi=ab.Beta(...))``; the intercept and slope
    priors are on the logit scale. ``predict(kind="mean")`` returns brms's expected count ``(1 - zi) n p`` as ``expected``
    (new rows need the trials column); ``kind="linear"`` returns ``eta``. The logit link only. ``weights()`` is taken (after
    ``trials()``: ``k | trials(n) + weights(w) ~ ...``). """ + _D03_CONTEXTS
    return Model(formula, data, priors, family="zero_inflated_binomial",
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def zero_inflated_beta(formula: str, data: pd.DataFrame, priors: Priors, *,
                       center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Zero-inflated Beta proportions (brms ``zero_inflated_beta``, D03): responses in [0, 1), an exact zero with probability
    ``zi``, a ``Beta(mu phi, (1 - mu) phi)`` proportion otherwise, ``mu = expit(eta)``; ``E[y] = (1 - zi) mu``.

    ``Priors(phi=..., zi=ab.Beta(...))`` (brms's defaults ``gamma(0.01, 0.01)`` and ``beta(1, 1)`` on request); the intercept
    and slope priors are on the logit scale. ``predict(kind="mean")`` returns ``(1 - zi) mu`` as ``mean``; ``kind="linear"``
    returns ``eta``. An exact 1 is refused; the logit link only. ``weights()`` is taken. """ + _D03_CONTEXTS
    return Model(formula, data, priors, family="zero_inflated_beta",
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def hurdle_gamma(formula: str, data: pd.DataFrame, priors: Priors, *,
                 center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Hurdle Gamma responses (brms ``hurdle_gamma``, D03): non-negative reals, an exact zero with probability ``hu``, a
    ``Gamma(shape, shape / mu)`` value otherwise, ``mu = exp(eta)``; ``E[y] = (1 - hu) mu``.

    ``Priors(shape=..., hu=ab.Beta(...))`` (brms's defaults ``gamma(0.01, 0.01)`` and ``beta(1, 1)`` on request); the intercept
    and slope priors act on the log of the Gamma part's mean. ``predict(kind="mean")`` returns ``(1 - hu) mu`` as ``mean``;
    ``kind="linear"`` returns ``eta``. ``weights()`` is taken. """ + _D03_CONTEXTS
    return Model(formula, data, priors, family="hurdle_gamma",
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def hurdle_lognormal(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
                     center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Hurdle lognormal responses (brms ``hurdle_lognormal``, D03): non-negative reals, an exact zero with probability ``hu``,
    ``LogNormal(mu, sigma)`` otherwise with the identity link on the log-scale location ``mu``; ``E[y] = (1 - hu) exp(mu +
    sigma^2 / 2)``.

    ``Priors(sigma=..., hu=ab.Beta(...))``; the intercept and slope priors act on log y in the lognormal part.
    ``predict(kind="mean")`` returns brms's ``posterior_epred`` as ``mean``; ``kind="linear"`` returns ``mu``. ``weights()`` is
    taken. E03 (docs/contexts-newer-families.md): ``hu`` may have its own formula under the logit link, ``["zpos ~ x", "hu ~ z"]``
    with ``Priors(hu={"intercept": ..., "b": ...})``. """ + _E03_CONTEXTS
    return Model(formula, data, priors, family="hurdle_lognormal",
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


# -- E01: bounded families (docs/bounded-families.md) ------------------------------------------------------------------------
_E01_CONTEXTS = ("No distributional formula, cens(), trunc(), mo(), me(), mi(), mm(), gr(by = ), autocorrelation term, multivariate "
                 "or non-linear formula, mixture component or other link (divergence D108).")


def zero_one_inflated_beta(formula: str, data: pd.DataFrame, priors: Priors, *,
                           center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Zero-one-inflated Beta proportions (brms ``zero_one_inflated_beta``, E01): responses in [0, 1], an exact 0 or 1 with
    probability ``zoi`` (a 1 with probability ``coi`` given that), a ``Beta(mu phi, (1 - mu) phi)`` proportion otherwise, ``mu =
    expit(eta)``; ``E[y] = zoi coi + (1 - zoi) mu``.

    ``Priors(phi=..., zoi=ab.Beta(...), coi=ab.Beta(...))`` (brms classes ``phi``, ``zoi`` and ``coi``; brms's defaults
    ``gamma(0.01, 0.01)``, ``beta(1, 1)`` and ``beta(1, 1)`` on request through ``ab.default_priors``); the intercept and slope
    priors are on the logit scale. ``predict(kind="mean")`` returns brms's ``posterior_epred`` as ``mean``; ``kind="linear"``
    returns ``eta``. ``weights()`` is taken. """ + _E01_CONTEXTS
    return Model(formula, data, priors, family="zero_one_inflated_beta",
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def zero_inflated_beta_binomial(formula: str, data: pd.DataFrame, priors: Priors, *,
                                center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Zero-inflated beta-binomial successes (brms ``zero_inflated_beta_binomial``, E01): ``successes | trials(n) ~ ...``, a
    structural zero with probability ``zi``, a ``BetaBinomial(n, mu phi, (1 - mu) phi)`` count otherwise, ``mu = expit(eta)``;
    ``E[y] = (1 - zi) n mu``.

    Trials as for the binomial family (zero-trial rows dropped, D4). ``Priors(phi=..., zi=ab.Beta(...))`` (brms's defaults
    ``gamma(0.01, 0.01)`` and ``beta(1, 1)`` on request); the intercept and slope priors are on the logit scale.
    ``predict(kind="mean")`` returns brms's expected count ``(1 - zi) n mu`` as ``expected`` (new rows need the trials column);
    ``kind="linear"`` returns ``eta``. ``weights()`` is taken (after ``trials()``). """ + _E01_CONTEXTS
    return Model(formula, data, priors, family="zero_inflated_beta_binomial",
                 center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def xbeta(formula: str, data: pd.DataFrame, priors: Priors, *,
          center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Extended-support Beta proportions (brms ``xbeta``, Kosmidis and Zeileis 2024; E01): responses in [0, 1] from one process,
    ``B ~ Beta(mu phi, (1 - mu) phi)`` with ``mu = expit(eta)`` stretched to ``(1 + 2 kappa) B - kappa`` and censored to [0, 1],
    so that exact 0s and 1s carry the mass of the stretched Beta beyond the ends.

    ``Priors(phi=..., kappa=...)`` (brms classes ``phi`` and ``kappa``, both ``real<lower=0>``; brms's defaults
    ``gamma(0.01, 0.01)`` for both on request through ``ab.default_priors``); the intercept and slope priors are on the logit
    scale of ``mu``. ``predict(kind="mean")`` returns brms's ``posterior_epred``, the censored variable's mean (not ``mu``), as
    ``mean``; ``kind="linear"`` returns ``eta``. ``weights()`` is taken. """ + _E01_CONTEXTS
    return Model(formula, data, priors, family="xbeta", center_predictors=center_predictors, categorical=categorical, dropna=dropna)


def hurdle_cumulative(formula: str, data: pd.DataFrame, priors: Priors, *,
                      center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
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
    return Model(formula, data, priors, family="hurdle_cumulative", center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def cumulative(formula: str, data: pd.DataFrame, priors: Priors, *, link: str = "logit",
               center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
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
    (divergences D38, D89), except F06b's grouped thresholds under the logit link: ``ord | thres(gr = z) ~ ...`` and
    ``ord | thres(K, gr = z) ~ ...`` give each level of ``z`` its own ordered thresholds (``intercept`` of ``value_point`` a
    mapping by level), and the design is then never centred, as brms leaves it (docs/grouped-thresholds.md, D158, D159).
    """
    return Model(formula, data, priors, family="cumulative", link=link, center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def _sequential(family: str, meaning: str):
    def build(formula: str, data: pd.DataFrame, priors: Priors, *, center_predictors: bool = True,
              categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
        return Model(formula, data, priors, family=family, center_predictors=center_predictors, categorical=categorical, dropna=dropna)

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
                center_predictors: bool = True, categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """Unordered categorical response (brms ``categorical()``): one linear predictor ``mu_k`` per non-reference category,
    ``P(y = k) = softmax(0, mu_2, ..., mu_K)_k`` (docs/ordinal-categorical.md).

    The categories are the response's distinct labels sorted as brms's ``as.factor`` sorts them (text in C-locale order,
    integer codes numerically); the first is the reference and ``model.categories`` holds them all. Each non-reference
    category has its own intercept and coefficients on the same design, all under the one ``Priors(intercept=...)`` and
    ``b`` (one prior per class shared by the categories, divergence D37). ``predict(kind="linear")`` returns ``mu`` with a
    ``category`` dimension over the non-reference categories, ``kind="mean"`` the probabilities ``p`` over all categories,
    ``kind="response"`` 1-based categories. Population-level terms only: group-level terms and the offset are refused
    (divergence D37). ``categorical={column: reference}`` declares categorical predictors as for every family.
    """
    return Model(formula, data, priors, family="categorical", center_predictors=center_predictors, categorical=categorical,
                 dropna=dropna)


def dirichlet(formula: str, data: pd.DataFrame, priors: Priors, *, refcat: str | None = None, center_predictors: bool = True,
              categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """A response on the simplex under brms's ``dirichlet()`` (F01b, docs/dirichlet-logistic-normal.md): ``cbind(y1, ..., yK) ~
    <terms>``, K proportion columns whose rows sum to 1, ``y_i ~ Dirichlet(phi * softmax(0, mu_i))`` with one linear predictor
    per non-reference category (brms's ``mu<k>``: its own centred intercept, coefficients and varying intercepts with their own
    SD) and the precision ``phi`` (``Priors(phi=...)``, brms's default ``gamma(0.01, 0.01)``).

    The categories are the columns, the first the reference (``refcat`` may name only it: divergence D138). ``Priors(intercept=,
    b=, sd=)`` hold for every non-reference category (one prior per class). ``predict(kind="linear")`` returns ``mu`` with a
    ``category`` dimension over the non-reference categories (brms's ``posterior_linpred``), ``kind="mean"`` the expected
    proportions ``softmax(0, mu)`` over all K (brms's ``posterior_epred``), ``kind="response"`` draws on the simplex.
    Population-level terms of every kind and varying intercepts ``(1 | g)``; group-level slopes, offsets and addition terms are
    refused (D138)."""
    return Model(formula, data, priors, family="dirichlet", center_predictors=center_predictors, categorical=categorical,
                 refcat=refcat, dropna=dropna)


def logistic_normal(formula: str, data: pd.DataFrame, priors: Priors, *, refcat: str | None = None, center_predictors: bool = True,
                    categorical: Mapping[str, object] | None = None, dropna: bool = False) -> Model:
    """A response on the simplex under brms's ``logistic_normal()`` (F01b, docs/dirichlet-logistic-normal.md): ``cbind(y1, ...,
    yK) ~ <terms>``, K >= 3 proportion columns whose rows sum to 1; the log-ratios ``log(y_i[k] / y_i[ref])`` of the
    non-reference categories are ``MultiNormal(mu_i, diag(sigma) Lncor diag(sigma))``, the density on the simplex with its
    Jacobian ``-sum_k log y_i[k]``. One linear predictor per non-reference category (brms's ``mu<k>``, as for ``ab.dirichlet``),
    one SD per non-reference category (brms's ``sigma<k>``, ``Priors(sigma=...)``, one prior for all; brms's default
    ``student_t(3, 0, 2.5)``) and the correlation of the log-ratios (``Priors(lncor=ab.LKJ(eta))``, brms's class ``lncor``,
    default ``lkj(1)``).

    ``refcat`` names the reference category, one of the columns (the first by default; brms's ``refcat``). ``predict(kind=
    "linear")`` returns ``mu`` (brms's ``posterior_linpred``) and ``kind="response"`` draws on the simplex; ``kind="mean"`` is
    refused, as brms refuses ``posterior_epred`` for the family. Population-level terms and varying intercepts ``(1 | g)``;
    group-level slopes, offsets, addition terms and two-column responses are refused (D139)."""
    return Model(formula, data, priors, family="logistic_normal", center_predictors=center_predictors, categorical=categorical,
                 refcat=refcat, dropna=dropna)


def mixture(formula: str, data: pd.DataFrame, priors: Priors, *, families: list[str], order: str | bool | None = None,
            center_predictors: bool = True, categorical: Mapping[str, object] | None = None, nl: bool = False, dropna: bool = False) -> Model:
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
                 categorical=categorical, dropna=dropna)
