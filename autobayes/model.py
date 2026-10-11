"""The model: a thin, inspectable layer over a Bambi-built PyMC model.

The families of ``autobayes.families``, including Poisson and negative binomial counts, Gamma, lognormal, Beta and
Student-t responses (``nu`` fixed or learned). ``"gaussian"`` (M0): identity link, residual SD ``sigma``. ``"bernoulli"``
(M1): logit link, ``p = expit(eta)``, no residual scale. ``"binomial"`` (M2): logit link on
``successes | trials(n)``, counts. The Bernoulli and binomial families also take ``link="probit"`` (A12) and
``link="cloglog"`` (P6; docs/links.md). All share the right-hand side ``[1 +] x1 [+ x2 ...] [+ offset(log(col))]
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
offered for the Gaussian family (accepted against two references since P6). The family's auxiliary parameter
(``sigma``, ``shape``, ``phi``) may have its own formula (B01, docs/distributional.md): ``formula`` is then a list,
``["y ~ x + (1 | g)", "sigma ~ z"]``, and Bambi builds the parameter as a distributional component under a log link
(``Design.dpars`` holds its design, ``Model.dpar_names`` its PyMC names). A finite mixture of Gaussian components (C05,
docs/mixtures.md; ``ab.mixture(..., families=[...])``) has one linear predictor per component with ordered intercepts, a
``sigma<k>`` per component and the mixing proportions ``theta`` (``autobayes/families/mixture.py``). A Gaussian process ``gp(x)`` or ``gp(x, k = 10, c =
5/4)`` (C03, docs/gaussian-processes.md) is added to the parent's linear predictor while Bambi builds the model, as a
monotonic term is (``Design.gp`` holds its inputs, ``Model.gp_names`` its PyMC names).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

from collections.abc import Mapping

from autobayes.correlation import check_correlation_cholesky, cholesky_corr_free, cholesky_corr_rv, n_free
from autobayes.families import FAMILIES, FamilySpec, get_family
from autobayes.formula import (Categorical, FormulaError, ModelSpec, addition_data, bambi_group_name, binomial_counts,
                               categorical_labels, factor_column, group_labels, interaction_cells, parse_formula, pop_term_design,
                               sd_tag)
from autobayes.priors import AUXILIARY, PriorError, Priors, describe_prior

_bambi_group_name = bambi_group_name          # ``1|g`` for the varying intercept of ``g``, ``x|g`` for the slope of ``x``


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
    cens: np.ndarray | None = None     # (n,) brms ``cens`` of a ``cens(col)`` term: -1 left-, 0 un-, 1 right-censored (B02, parked)
    lb: np.ndarray | None = None       # (n,) brms ``lb`` and ``ub`` of a ``trunc()`` term, -inf / inf on an open side (B02, parked)
    ub: np.ndarray | None = None
    dpars: Mapping[str, DparDesign] = field(default_factory=lambda: MappingProxyType({}))   # each distributional parameter's own design, by brms name (B01)
    monotonic: Mapping[str, MonotonicBlock] = field(default_factory=lambda: MappingProxyType({}))   # each monotonic term's block, by column (B03)
    gp: Mapping[str, GPBlock] = field(default_factory=lambda: MappingProxyType({}))   # each Gaussian process's block, by column (C03)

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
        else:
            parts.append(f"(x_{term} - mean(x_{term})) * {b}_{term}" if centered else f"x_{term} * {b}_{term}")
    return parts


class Model:
    """``y ~ x [+ (1 | factor) ...]`` with explicit priors; Gaussian, Student-t, Poisson, Bernoulli or binomial likelihood.

    Attributes
    ----------
    family : str               a registered family name: "gaussian", "student", "poisson", "negbinomial", "gamma", "lognormal", "beta", "bernoulli", "binomial",
                               "zero_inflated_poisson", "hurdle_poisson", "cumulative", "categorical"
    categories : tuple[str]    the response's categories in order (the cumulative and categorical families, B05); () otherwise
    family_spec : FamilySpec   the specification of that family under the model's link (autobayes/families/; the registered spec unless ``link`` names another of the family's links)
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
                 families: list[str] | None = None, order: str | bool | None = None):
        import bambi as bmb

        fam = get_family(family)
        if link is not None:                                   # the family's spec under another of its links (A12)
            try:
                fam = fam.with_link(link)
            except ValueError as e:
                offering = [name for name, other in FAMILIES.items() if other.links]
                hint = (f" (link= applies to the {' and '.join(offering)} families, which take "
                        f"{sorted({l for name in offering for l in (FAMILIES[name].link, *FAMILIES[name].links)})})") if not fam.links else ""
                raise ValueError(f"{e}{hint}") from None
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
        operands = [c for t in spec0.group_terms for c in t.operands]   # the columns the grouping factors are read from
        cols = ([spec0.response, *spec0.pop_columns] + operands
                + ([spec0.trials] if spec0.trials else []) + ([spec0.offset.column] if spec0.offset is not None else [])
                + [c for a in spec0.additions for c in a.columns]          # weights() and cens() columns (B02)
                + [c for d in spec0.dpars for c in d.pop_columns])         # the columns distributional formulas read (B01)
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
            df[t.column] = group_labels(t.factor, df)
        for t in (*self.spec.transform_terms, *(t for d in self.spec.dpars for t in d.transform_terms)):
            df[t.name] = t.transform.values(df)              # the evaluated column Bambi is handed, under the term's own name (A05)
        self.categories: tuple[str, ...] = ()               # the response's categories, in order (B05: cumulative, categorical)
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

        bambi_family = fam.bambi_family()
        if self.spec.additions:                              # brms's response addition terms act on the likelihood (B02)
            from autobayes.likelihoods import addition_family
            bambi_family = addition_family(bambi_family, fam, addition_data(self.spec, self.data))
        self._check_generated_names()
        bambi_priors = priors.to_bambi(self.spec, fam)
        for name in fam.fixed_parameters:                    # Bambi takes a fixed nu through the prior dict; a learned
            if name not in self.learned_parameters:          # one's truncated prior came from Priors.to_bambi (A13)
                bambi_priors[name] = np.float64(getattr(self, name))   # PyTensor makes a Python 4.0 a float32 constant
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
        self.bambi_model = bmb.Model(
            bambi_formula, self.data, family=bambi_family,
            priors=bambi_priors,
            noncentered=True, center_predictors=self._bambi_centring,
        )
        self.bambi_model.build()
        self.pymc_model = self.bambi_model.backend.model
        if self.spec.correlated_terms:                       # brms's correlated slopes (A02): a post-build edit of Bambi's graph
            self.pymc_model = self._correlate_group_effects(self.pymc_model)
            self.bambi_model.backend.model = self.pymc_model  # what Bambi's fit() samples and cleans
        self.design = self._extract_design()
        if fam.mixture:                                      # C05: the graph built beside Bambi's components, checked
            from autobayes.families.mixture import check_graph
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
        variable_names += list(self.family_spec.fixed_parameters)   # Bambi uses "nu" as an auxiliary-prior key even when fixed.
        if self.family_spec.thresholds:                     # B05: the raw thresholds' deterministic and their coordinate
            variable_names += [self.names["thresholds"], self.names["threshold_dim"]]
        if self.family_spec.per_category:                   # B05: the category coordinates Bambi adds
            variable_names += [self.names["category_dim"], f"{self.spec.response}_dim"]
        if self.family_spec.mixture:                        # C05: each component's sigma, theta and the component coordinates
            roles = self.mixture_names
            variable_names += [*roles["sigma"], roles["theta"], roles["coordinate"], f"{self.spec.response}_dim"]
        for roles in self.group_coefficient_names.values():
            variable_names.extend(roles[k] for k in ("sd", "offset", "group_effect"))
        for roles in self.correlation_names.values():
            variable_names.extend(roles[k] for k in ("L", "cor"))
        dpar_coefficients = [v for roles in self.dpar_names.values() for k, v in roles.items() if k != "value"]   # B01
        variable_names += dpar_coefficients
        variable_names += [roles["simplex"] for roles in self.mo_names.values()]   # each monotonic term's simplex (B03)
        variable_names += [roles[k] for roles in self.gp_names.values() for k in ("sdgp", "lscale", "zgp")]   # each GP's (C03)
        # Bambi's default result cleaning drops every posterior variable ending in _offset.
        retained = list(self.spec.terms) + [roles["group_effect"] for roles in self.group_coefficient_names.values()] + dpar_coefficients
        if any(name.endswith("_offset") for name in retained):
            raise FormulaError("generated name collision: coefficient/effect suffix '_offset' is reserved by Bambi; rename the column")
        coords = [roles["coordinate"] for roles in self.group_names.values()]
        coords += [roles[k] for roles in self.correlation_names.values() for k in ("coordinate", "coordinate_T")]
        coords += [f"{t.name}_dim" for t in self.spec.pop_terms if t.is_vector]
        coords += [f"{self.dpar_names[d.response][f'b:{t.name}']}_dim" for d in self.spec.dpars for t in d.pop_terms if t.is_vector]
        coords += [roles["coordinate"] for roles in self.mo_names.values()]
        coords += [roles["coordinate"] for roles in self.gp_names.values()]       # C03
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
        for t in self.spec.correlated_terms:
            g, M = t.factor, len(t.coefs)
            names = self._correlation_roles(g)
            rv, transform, value = cholesky_corr_rv(names["L"], float(cor_priors[g].params["eta"]), M)
            for dim in (names["coordinate"], names["coordinate_T"]):
                fgraph._coords[dim] = tuple(t.coefs)
                fgraph._dim_lengths[dim] = pytensor.shared(M, name=dim)
            L = model_free_rv(rv, value, transform, names["coordinate"], names["coordinate_T"])
            L.name = names["L"]
            cor = L @ L.T
            cor.name = names["cor"]
            cor_det = model_deterministic(cor, names["coordinate"], names["coordinate_T"])
            cor_det.name = names["cor"]
            added += [L, cor_det]
            roles = [self._group_roles(g, c) for c in t.coefs]
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
            expected_terms = [_bambi_group_name(t.factor, coef) for t in self.spec.group_terms for coef in t.coefs]
            if group is None or list(group.terms) != expected_terms:
                raise RuntimeError("Bambi group terms differ from formula order")
            Z = group.design_matrix
            Z = np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, dtype=float)
            start = 0
            for t in self.spec.group_terms:
                g = t.factor
                levels = self.spec.levels_by_group[g]
                labels = self.data[t.column].to_numpy()
                indicator = np.column_stack([labels == level for level in levels]).astype(float)
                for coef in t.coefs:
                    name = _bambi_group_name(g, coef)
                    bterm = group.terms[name]
                    if tuple(str(v) for v in bterm.groups) != levels:
                        raise RuntimeError(f"Bambi level order for {g!r} differs from sorted fitted labels")
                    block_slice = slice(start, start + len(levels))
                    if group.slices[name] != block_slice:
                        raise RuntimeError(f"Bambi group slice for {g!r} differs from formula order")
                    block = Z[:, block_slice]
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
            if Z.shape != (len(X_raw), start):
                raise RuntimeError("unexpected concatenated group design shape")
            for t in self.spec.correlated_terms:             # the correlation factor (A02): a free variable on Stan's coordinates
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
                      dpars=MappingProxyType({d.response: self._dpar_design(d) for d in self.spec.dpars}),
                      monotonic=MappingProxyType(self._monotonic_design()), gp=MappingProxyType(self._gp_design()))

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
        if comp.design.group is not None:
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
        if self.bambi_model.family.link[bname].name != self.family_spec.distributional_link:
            raise RuntimeError(f"Bambi's link of {bname!r} is {self.bambi_model.family.link[bname].name}, expected "
                               f"{self.family_spec.distributional_link}")
        centered = bool(self._bambi_centring and d.intercept and d.pop_terms)   # an intercept-only formula has nothing to centre
        x_means = X_raw.mean(0) if centered else np.zeros(X_raw.shape[1])
        return DparDesign(name=d.response, X_raw=X_raw, X_model=X_raw - x_means if centered else X_raw, x_means=x_means,
                          columns=d.columns, centered=centered, link=self.family_spec.distributional_link)

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
        hurdle families (B06), whose linear predictor is ``"log_mu"`` and whose ``"mean"`` is the expected count. ``"b:<term>"`` is the
        coefficient variable of each population-level term (scalar for a numeric term, a vector
        over the coefficient levels otherwise); with exactly one group-level term its ``sd``,
        ``offset`` and ``group_effect`` roles are included here as well, and ``sd:x``, ``offset:x``,
        ``group_effect:x`` for each varying slope ``x`` of that term (see ``group_names``). An auxiliary parameter with its own
        formula (B01) keeps its entry, now a per-row deterministic; its formula's variables are in ``dpar_names``."""
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
        }
        if fam.scale_parameter is not None:
            out[fam.scale_parameter] = fam.bambi_scale_parameter
        if fam.zero_parameter is not None:                    # "zi" or "hu" (B06): the PyMC variable keeps brms's name
            out[fam.zero_parameter] = fam.zero_parameter
        for name in self.learned_parameters:                  # a learned nu (A13): the PyMC variable of that name
            out[name] = name
        if fam.thresholds:                                    # B05: brms's b_Intercept, the thresholds of the raw design, and
            out["thresholds"] = "b_Intercept"                 # the coordinate of the thresholds ("1", ..., "K - 1")
            out["threshold_dim"] = "Intercept_dim"
        if fam.per_category:                                  # B05: the coordinate of the non-reference categories
            out["category_dim"] = f"{self.spec.response}_reduced_dim"
        if fam.mixture:                                       # C05: each component's sigma<k>, theta and the component coordinate
            roles = self.mixture_names
            out.update({name: name for name in roles["sigma"]}, theta=roles["theta"], component_dim=roles["coordinate"])
        if fam.continuous and fam.parent == "mu":
            out["mu"] = "mu"                                  # the continuous families whose parent is the location mu expose it by name (the Beta's parent is logit_mu)
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
        """``_correlation_roles`` of every correlated group term, by factor in formula order (empty without one)."""
        return {t.factor: self._correlation_roles(t.factor) for t in self.spec.correlated_terms}

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
            if t.has_correlation:
                roles.update({k: v for k, v in self._correlation_roles(t.factor).items() if k in ("L", "cor")})
            out[t.factor] = roles
        return out

    @property
    def group_coefficient_names(self) -> dict[tuple[str, str], dict[str, str]]:
        """PyMC roles (``sd``, ``offset``, ``group_effect``, ``coordinate``) of every group-level coefficient, keyed
        ``(factor, coefficient)`` in formula order like ``Design.group_blocks``."""
        return {(g, coef): self._group_roles(g, coef) for g, coef in self.spec.group_coefficients}

    @property
    def coefficients(self) -> tuple[tuple[str, str, int | None], ...]:
        """``(design column, PyMC variable, index into that variable or None)`` in design order:
        ``("x", "x", None)`` for a numeric slope, ``("z[b]", "z", 0)`` for a dummy coefficient
        (the variable ``z`` has dims ``z_dim`` with the coefficient levels as coordinates); one entry
        per column of each ``PopTerm``."""
        return self.spec.column_terms

    def value_point(self, *, intercept: float | None = None, b, sd=None, z=None, L=None, sigma=None,
                    shape=None, phi=None, nu: float | None = None,
                    zi: float | None = None, hu: float | None = None, bsp=None, simo=None, theta=None,
                    sdgp=None, lscale=None, zgp=None) -> dict[str, np.ndarray]:
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
        design-column order (brms's ``b_sigma``), each stored under its own variable (``Model.dpar_names``), unconstrained.
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
        required exactly when the model has one.
        """
        if self.family_spec.mixture:                          # C05: the components' parameters and theta
            from autobayes.families.mixture import value_point
            return value_point(self, intercept=intercept, b=b, sigma=sigma, theta=theta, sd=sd, z=z, L=L, shape=shape, phi=phi,
                               nu=nu, zi=zi, hu=hu, bsp=bsp, simo=simo)
        if theta is not None:
            raise ValueError("theta (the mixing proportions) is a mixture's (ab.mixture); this model has none")
        b = np.atleast_1d(np.asarray(b, dtype=float))
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
        correlated = [t.factor for t in self.spec.correlated_terms]
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
            for t in self.spec.correlated_terms:
                L_by_group[t.factor] = check_correlation_cholesky(by_group[t.factor], len(t.coefs), f"L for {t.factor!r}")
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
        n = self.names
        if self.spec.intercept and intercept is None:
            raise ValueError("intercept is required for a model with a population intercept")
        if not self.spec.intercept and intercept is not None:
            raise ValueError("no population intercept exists; omit intercept (or pass None)")
        point = {n["intercept"]: np.asarray(float(intercept))} if self.spec.intercept and not self.categories else {}
        if self.spec.intercept and self.categories:          # B05: K - 1 thresholds, or one intercept per non-reference category
            arr = np.atleast_1d(np.asarray(intercept, dtype=float))
            K = len(self.categories)
            if self.family_spec.thresholds:
                if arr.shape != (K - 1,) or not np.isfinite(arr).all() or not (np.diff(arr) > 0).all():
                    raise ValueError(f"intercept must be the {K - 1} thresholds, finite and strictly increasing (brms's Intercept)")
                point[f"{n['intercept']}_ordered__"] = np.r_[arr[0], np.log(np.diff(arr))]     # Stan's ordered coordinates
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
            point[f"{roles['sd']}_log__"] = log_scale(sd_by_coef[(g, coef)], f"sd:{sd_tag(g, coef)}")
            point[roles["offset"]] = z_by_coef[(g, coef)]
        for g, roles in self.correlation_names.items():              # Stan's cholesky_corr coordinates of L (A02)
            point[f"{roles['L']}_cholesky_corr__"] = cholesky_corr_free(L_by_group[g])
        if scale in self.spec.dpar_names:                      # its own formula (B01): the intercept and coefficients as sampled
            point.update(self._dpar_point(scale, auxiliary[scale]))
        elif scale is not None:
            point[f"{n[scale]}_log__"] = log_scale(auxiliary[scale], scale)
        if zero is not None:                                   # B06: a probability, sampled as its log-odds
            p = np.asarray(probabilities[zero], dtype=float)
            if p.shape != () or not np.isfinite(p) or not 0 < p < 1:
                raise ValueError(f"{zero} must be a finite scalar strictly between 0 and 1")
            point[f"{n[zero]}_logodds__"] = np.asarray(np.log(p) - np.log1p(-p))
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
        for term in self.spec.bambi_terms:                    # b in design-column order: one entry per term column
            k = len(term.columns)
            point[term.name] = np.array(b[i:i + k]) if term.levels else np.asarray(b[i])
            i += k
        point.update(self._monotonic_point(bsp, simo))       # B03: each monotonic term's coefficient and simplex
        point.update(self._gp_point(sdgp, lscale, zgp))      # C03: each Gaussian process's sdgp, lscale and zgp
        expected = {v.name for v in self.pymc_model.value_vars}
        if set(point) != expected:
            raise RuntimeError(f"value names {sorted(point)} do not match PyMC's {sorted(expected)}")
        return point

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

    def _dpar_point(self, name: str, value) -> dict[str, np.ndarray]:
        """``value_point``'s entries of distributional parameter ``name``'s formula (B01) from ``{"intercept": float,
        "b": array}``: the intercept (as sampled) and the coefficients term by term in design-column order."""
        d, roles = self.spec.dpar(name), self.dpar_names[name]
        if not isinstance(value, Mapping) or set(value) - {"intercept", "b"}:
            raise ValueError(f"{name} has its own formula '{d.formula}': give {name}={{'intercept': float, 'b': array}} "
                             "(the formula's intercept as sampled and its coefficients in design-column order)")
        point = {}
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

    # -- description -------------------------------------------------------------------
    def describe(self) -> str:
        if self.family_spec.mixture:                          # C05: the components, theta and their priors
            from autobayes.families.mixture import describe
            return describe(self)
        s, d, pr = self.spec, self.design, self.priors
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
                    + (["independent varying slope(s)"] if any(t.slopes and not t.has_correlation for t in s.group_terms) else [])
                    + (["correlated varying slope(s)"] if s.correlated_terms else []) + (["offset"] if s.offset is not None else [])
                    + (["derived grouping factor(s)"] if any(t.derived for t in s.group_terms) else [])
                    + ([f"response addition term(s) {', '.join(a.term for a in s.additions)}"] if s.additions else [])
                    + ([f"distributional formula(s) for {', '.join(s.dpar_names)}"] if s.dpars else []))
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
            lines += [f"  {s.response:<10}: response categories " + ", ".join(f"{k}={c}" for k, c in enumerate(self.categories, start=1))
                      + (" (ordered)" if fam.thresholds else f" (the reference is {self.categories[0]!r})")
                      + "; fitted as the 1-based category (brms's Y)"]
        if s.offset is not None:
            lines += [f"  offset    : {s.offset.expression}, added to the linear predictor as written (no coefficient, no prior, "
                      f"never centred; brms offset()); range {d.offset.min():.4f} to {d.offset.max():.4f} over the fitted rows"]
        lines += self._addition_lines()
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
        for g, block in d.group_designs.items():
            lines += [f"  {'levels    ' if len(s.groups) == 1 else 'levels ' + g}: {', '.join(f'{j}={l}' for j, l in enumerate(block.levels))}"]
        for t in s.group_terms:
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
            f" + {effect(t.factor)}[{t.factor}_i]" + "".join(f" + x_{c},i * {slope_effect(t.factor, c)}[{t.factor}_i]" for c in t.slopes)
            for t in s.group_terms)
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
                lines += [f"  u_{g}[j] = sd_{g} * z_{g}[j], z_{g}[j] ~ Normal(0, 1), j = 0..{block.J - 1}"]
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
        lines += [f"  b_{col}{each} ~ {describe_prior(coef[col])}" for col in s.columns]
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
        lines += fam.likelihood_lines(self)
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
        if s.intercept and fam.thresholds:                    # B05: Stan's ordered coordinates of the thresholds
            roles = {f"{n['intercept']}_ordered__": f"Intercept[1..{len(self.categories) - 1}], the thresholds"
                     + (" of the centred design" if d.centered else "") + ", on Stan's ordered coordinates "
                     "(Intercept[1], log(Intercept[k] - Intercept[k - 1]))"}
        if fam.per_category:                                  # B05: one entry per non-reference category
            others = ", ".join(self.categories[1:])
            roles = {n["intercept"]: f"{'a_c' if d.centered else 'a'}[{others}] (vector; dims {n['category_dim']})"} if s.intercept else {}
            roles.update({t.name: f"b_{t.name}[{others}] (vector; dims {n['category_dim']})" for t in s.pop_terms if not t.is_vector})
        for (g, c), names in self.group_coefficient_names.items():
            roles[f"{names['sd']}_log__"] = f"log(sd_{sd_tag(g, c)})"
            roles[names["offset"]] = ("z" if len(s.groups) == 1 else f"z_{g}") if c == "Intercept" else f"z_{c}|{g}"
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
        for t in s.pop_terms:
            if t.is_vector:
                shown = ", ".join(f"({l})" for l in t.levels) if len(t.categorical) == 2 else ", ".join(t.levels)   # cells read (l, m)
                rows = ", ".join(f"{t.name}[{l}]" for l in t.levels)      # ArviZ names a row variable[coordinate]
                roles[t.name] = f"b_{t.name}[{shown}] (vector; dims {t.name}_dim, posterior rows {rows})"
                if fam.per_category:                          # B05: and one column per non-reference category
                    roles[t.name] = f"b_{t.name}[{shown}] (matrix; dims {t.name}_dim x {n['category_dim']})"
        for v in self.pymc_model.value_vars:
            lines.append(f"  {v.name:<24} {roles.get(v.name, f'b_{v.name}')}")
        lines.append("Deterministic: " + "".join(f"{names['group_effect']} = {effect(g) if c == 'Intercept' else slope_effect(g, c)}, "
                                                 for (g, c), names in self.group_coefficient_names.items())
                     + "".join(f"{names['cor']} = Cor_{g} = L_{g} L_{g}^T, " for g, names in self.correlation_names.items()) + fam.deterministic_note
                     + "".join(f"; {dn['value']} = exp(eta_{name}), one {name} per row (B01)" for name, dn in self.dpar_names.items()))
        return "\n".join(lines)

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

    def _dpar_lines(self, ds: ModelSpec) -> list[str]:
        """``describe()``'s paragraph of distributional parameter ``ds.response``'s own formula (B01): its linear
        predictor written out (coefficients ``b_<parameter>_<term>``), its link, and its centring."""
        name, dd = ds.response, self.design.dpars[ds.response]
        a = f"a_{name}{'_c' if dd.centered else ''}" if ds.intercept else "0"
        xs = " + ".join(_linear_terms(ds, dd.centered, b=f"b_{name}"))
        lines = ["", f"Distributional parameter {name} (its own formula '{ds.formula}', {dd.link} link; brms's bf(..., {ds.formula}))",
                 f"  {dd.link}({name}_i) = eta_{name},i = {a}" + (f" + {xs}" if xs else ""),
                 f"  {name}_i = exp(eta_{name},i) > 0 is row i's {name} in the likelihood below; the coefficients act on log {name}."]
        if dd.centered:
            lines += [f"  Its columns are centred at their own fitted-row means (brms's Xc_{name}), so a_{name}_c is log {name} there and",
                      f"  its prior applies to a_{name}_c; the posterior reports the raw-scale intercept",
                      f"  a_{name} = a_{name}_c - sum_p mean(x_p) * b_{name}_p (brms's b_{name}_Intercept), computed after sampling."]
        elif not ds.intercept:
            lines += [f"  No intercept: log {name} is the raw design times its coefficients."]
        return lines

    def _addition_lines(self) -> list[str]:
        """``describe()``'s data lines for the response addition terms (B02, docs/addition-terms.md)."""
        d, out = self.design, []
        for a in self.spec.additions:
            if a.kind == "weights":
                out += [f"  weights   : {a.column}, each fitted row's log likelihood multiplied by its weight (brms weights()); weights "
                        f"{d.weights.min():.4f} to {d.weights.max():.4f}, sum {d.weights.sum():.4f} over {d.n} rows; the priors are not "
                        "weighted, and predict() ignores the weights (brms's posterior_epred and posterior_predict do too)"]
            elif a.kind == "cens":
                left, right = int((d.cens == -1).sum()), int((d.cens == 1).sum())
                out += [f"  censoring : {a.column}: {left} left-censored row(s) (-1: y_i is an upper bound, log P(Y_i <= y_i)), {right} "
                        f"right-censored (1: a lower bound, log P(Y_i > y_i)), {d.n - left - right} observed (0); brms cens(); predict() "
                        "ignores the censoring, as brms does"]
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
            if a.kind == "weights":
                out += [f"  weights(): the log likelihood is sum_i w_i log p(y_i | ...), w_i the {a.column} of row i (brms's weights[n] *)"]
            elif a.kind == "cens":
                out += ["  cens(): a left-censored row contributes log P(Y_i <= y_i), a right-censored row log P(Y_i > y_i), any other row its "
                        "log density (Stan's _lcdf / _lccdf / _lpdf as brms writes them)"]
            else:
                out += [f"  trunc(): every row's log density minus log P(lb < Y_i <= ub), the log mass of {a.term}"]
        return out

    # -- priors ------------------------------------------------------------------------
    @property
    def default_prior_labels(self) -> list[str]:
        """The parameters whose prior came from ``ab.default_priors`` (brms's defaults; A07), labelled as the prior
        messages label them (``intercept``, ``b:x``, ``sd:g``, ``sd:x|g``, ``cor:g``, ``simo:x``, ``sigma``, ``nu``), in that order; empty
        when the user gave every prior (``zi`` or ``hu`` after the auxiliary parameter, B06; a distributional formula's
        ``sigma:intercept``, ``sigma:b:z[q]`` in place of ``sigma``, B01; a mixture's ``mu1:intercept``, ``mu1:b:x``, ``sigma1``,
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
        if fam.scale_parameter in s.dpar_names:              # its own formula's priors (B01)
            ds, sub = s.dpar(fam.scale_parameter), pr.dpar_priors(s, fam.scale_parameter)
            found += [(f"{ds.response}:intercept", sub.intercept)] if ds.intercept else []
            found += [(f"{ds.response}:b:{k}", v) for k, v in sub.coefficient_priors(ds).items()]
        elif fam.scale_parameter is not None:
            found.append((fam.scale_parameter, getattr(pr, fam.scale_parameter)))
        if fam.zero_parameter is not None:
            found.append((fam.zero_parameter, getattr(pr, fam.zero_parameter)))
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
             loop: bool = True) -> "Model | MultivariateModel":
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
    True, is offered: divergence D63), and a non-linear model takes no multivariate formula or ``rescor`` (divergence D62)."""
    from autobayes.multivariate import MultivariateModel, is_multivariate_formula     # C01

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
    return Model(formula, data, priors, family="gaussian", center_predictors=center_predictors, categorical=categorical)


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


def poisson(formula: str, data: pd.DataFrame, priors: Priors, *,
            center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Poisson count response with log link and explicit coefficient/group priors.

    Retains zero rows. Input counts must be integers in [0, 1e6]. An exposure enters as
    ``offset(log(exposure))`` (docs/offset.md). No sigma, nu or overdispersion parameter
    is supported in this slice.
    """
    return Model(formula, data, priors, family="poisson",
                 center_predictors=center_predictors, categorical=categorical)


def negbinomial(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
                center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Negative binomial count response (brms ``negbinomial``): log link, ``Var[y] = mu + mu^2 / shape``.

    Counts as for Poisson (integers in [0, 1e6], zero rows retained). The overdispersion parameter ``shape``
    takes its own positive-support prior, ``Priors(shape=ab.Gamma(0.01, 0.01))``, or its own formula under a log link (B01,
    ``["y ~ x", "shape ~ z"]``, docs/distributional.md); no sigma, nu, offset/exposure or zero-inflation.
    """
    return Model(formula, data, priors, family="negbinomial",
                 center_predictors=center_predictors, categorical=categorical)


def gamma(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
          center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Gamma response with a log link (brms ``Gamma(link = "log")``): ``E[y] = mu = exp(eta)``, ``Var[y] = mu^2 / shape``.

    Responses are strictly positive finite reals. The shape parameter takes its own positive-support prior,
    ``Priors(shape=ab.Gamma(0.01, 0.01))``, or its own formula under a log link (B01, ``["y ~ x", "shape ~ z"]``,
    docs/distributional.md); no sigma, nu, offset or other link (brms's default Gamma link is inverse;
    docs/brms-parity/divergences.md, D9).
    """
    return Model(formula, data, priors, family="gamma",
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


def beta(formula: str | list[str], data: pd.DataFrame, priors: Priors, *,
         center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Beta response with a logit link (brms ``Beta()``): proportions strictly inside (0, 1), ``E[y] = mu = expit(eta)``,
    ``Var[y] = mu (1 - mu) / (1 + phi)``.

    The precision ``phi`` takes its own positive-support prior, ``Priors(phi=ab.Gamma(0.01, 0.01))`` (brms class
    ``phi``), or its own formula under a log link (B01, ``["y ~ x", "phi ~ z"]``, docs/distributional.md); the intercept and
    slope priors are on the logit scale. ``predict(kind="mean")`` returns ``mu``; no sigma, shape, nu, other link or
    zero/one inflation.
    """
    return Model(formula, data, priors, family="beta",
                 center_predictors=center_predictors, categorical=categorical)


def zero_inflated_poisson(formula: str, data: pd.DataFrame, priors: Priors, *,
                          center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Zero-inflated Poisson counts (brms ``zero_inflated_poisson``): a structural zero with probability ``zi``, a
    ``Poisson(mu)`` count otherwise, ``mu = exp(eta)``; ``E[y] = (1 - zi) mu``.

    Counts as for Poisson (integers in [0, 1e6], zero rows retained). The zero-inflation probability takes its own prior on
    (0, 1), ``Priors(zi=ab.Beta(1, 1))`` (brms class ``zi``; brms's default ``beta(1, 1)`` is not filled in silently,
    ``ab.default_priors(..., family="zero_inflated_poisson")`` returns it); the intercept and slope priors act on the
    log of the Poisson part's mean. ``predict(kind="mean")`` returns ``(1 - zi) mu`` as ``mean``; ``kind="linear"``
    returns ``eta``. No overdispersion or distributional (``zi ~ ...``) formula (docs/zero-inflated-hurdle.md).
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
    ``kind="linear"`` returns ``eta``. No overdispersion or distributional (``hu ~ ...``) formula
    (docs/zero-inflated-hurdle.md).
    """
    return Model(formula, data, priors, family="hurdle_poisson",
                 center_predictors=center_predictors, categorical=categorical)


def cumulative(formula: str, data: pd.DataFrame, priors: Priors, *,
               center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Ordinal response (brms ``cumulative("logit")``): K ordered categories, ``P(y <= k) = expit(tau_k - mu)`` with K - 1
    increasing thresholds ``tau`` and the linear predictor ``mu`` without an intercept (docs/ordinal-categorical.md).

    The response is positive integers ``1..K`` (K = the largest; brms's rule) or an ordered pandas Categorical (its declared
    categories, in order); it is fitted as its 1-based category, ``model.categories`` holding the labels. The intercept prior
    ``Priors(intercept=...)`` applies to every threshold, as brms's class ``Intercept`` does; the thresholds are sampled for
    the centred design (brms's ``Intercept``), on Stan's ordered coordinates, and the graph's ``b_Intercept`` holds them for
    the raw design. The intercept cannot be removed (brms refuses it too). ``predict(kind="mean")`` returns the category
    probabilities ``p`` with a ``category`` dimension, ``kind="linear"`` returns ``mu``, ``kind="response"`` 1-based
    categories. No other link, no ``disc``, category-specific effects or threshold structures (divergence D38).
    """
    return Model(formula, data, priors, family="cumulative", center_predictors=center_predictors, categorical=categorical)


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
            center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
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
    component; other component families, ``order="none"``, group-level terms and offsets are refused (D52-D55).
    """
    return Model(formula, data, priors, family="mixture", families=families, order=order, center_predictors=center_predictors,
                 categorical=categorical)
