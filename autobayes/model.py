"""The model: a thin, inspectable layer over a Bambi-built PyMC model.

The families of ``autobayes.families``, including Poisson and negative binomial counts, Gamma, lognormal, Beta and
Student-t responses (``nu`` fixed or learned). ``"gaussian"`` (M0): identity link, residual SD ``sigma``. ``"bernoulli"``
(M1): logit link, ``p = expit(eta)``, no residual scale. ``"binomial"`` (M2): logit link on
``successes | trials(n)``, counts. The Bernoulli and binomial families also take ``link="probit"`` (A12,
docs/links.md; brms's cloglog link is withheld there). All share the right-hand side ``[1 +] x1 [+ x2 ...] [+ offset(log(col))]
[+ (1 | factor) | (1 + x || factor) | (1 + x | factor) ...]`` with numeric terms and, since M3a-1, treatment-coded categorical
terms declared with ``categorical={column: reference}``; an optional offset term (A06, docs/offset.md), added to
the linear predictor with no coefficient; centring; non-centred group effects (varying
intercepts and, with ``||``, independent varying slopes; with ``|``, slopes correlated with the intercept under an
LKJ prior, A02, docs/correlated-slopes.md) when a group term is present, and a population-only
model without one (M3c-1). The formula is a term model (``PopTerm``, ``GroupTerm`` in ``autobayes.formula``;
docs/term-model.md) and ``Design`` holds one ``GroupBlock`` per group-level coefficient. Bambi builds every
group-level coefficient as an independent non-centred term; for a correlated term ``Model`` then rewrites the
slopes' deterministics in the PyMC graph to brms's ``r = (diag(sd) L z)^T`` with a Cholesky factor ``L`` sampled on
Stan's ``cholesky_corr`` coordinates (``autobayes.correlation``).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from collections.abc import Mapping

from autobayes.correlation import check_correlation_cholesky, cholesky_corr_free, cholesky_corr_rv, n_free
from autobayes.families import FAMILIES, FamilySpec, get_family
from autobayes.formula import (Categorical, FormulaError, ModelSpec, bambi_group_name, binomial_counts, categorical_labels,
                               factor_column, group_labels, interaction_cells, parse_formula, pop_term_design, sd_tag)
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


class Model:
    """``y ~ x [+ (1 | factor) ...]`` with explicit priors; Gaussian, Student-t, Poisson, Bernoulli or binomial likelihood.

    Attributes
    ----------
    family : str               a registered family name: "gaussian", "student", "poisson", "negbinomial", "gamma", "lognormal", "beta", "bernoulli", "binomial",
                               "zero_inflated_poisson", "hurdle_poisson"
    family_spec : FamilySpec   the specification of that family under the model's link (autobayes/families/; the registered spec unless ``link`` names another of the family's links)
    link : str                 the link: the family's own, or the ``link`` given ("probit" for the Bernoulli and binomial families)
    spec : ModelSpec
    priors : Priors
    bambi_model : bambi.Model  (already built)
    pymc_model : pymc.Model    the graph that is sampled; inspect it directly
    design : Design            design matrices and group coding
    """

    def __init__(self, formula: str, data: pd.DataFrame, priors: Priors, *,
                 family: str = "gaussian", center_predictors: bool = True,
                 categorical: Mapping[str, object] | None = None, nu: float | None = None, link: str | None = None):
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

        spec0 = parse_formula(formula, None, family, categorical)
        self.effective_center_predictors = self.center_predictors and spec0.intercept
        priors.validate(spec0, fam)                     # structure now; per-level coverage once levels are known
        operands = [c for t in spec0.group_terms for c in t.operands]   # the columns the grouping factors are read from
        cols = ([spec0.response, *spec0.pop_columns] + operands
                + ([spec0.trials] if spec0.trials else []) + ([spec0.offset.column] if spec0.offset is not None else []))
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
        for t in self.spec.pop_terms:                        # a cell of z:h with no fitted rows has a zero column (brms keeps it too)
            if t.kind == "interaction" and len(t.categorical) == 2:
                empty = [lvl for lvl, col in zip(t.levels, pop_term_design(t, df, self.spec.categoricals).T) if not col.any()]
                if empty:
                    self.warnings.append(
                        f"cell(s) {empty} of {t.name!r} have no fitted rows: their coefficients are informed by the prior alone "
                        "(the design column is zero, as in brms)."
                    )
        for t in self.spec.group_terms:                      # labels as text; a derived factor g:h gets its own column g_h
            df[t.column] = group_labels(t.factor, df)
        for t in self.spec.transform_terms:                  # the evaluated column Bambi is handed, under the term's own name (A05)
            df[t.name] = t.transform.values(df)
        if fam.response_dtype is not None:
            df[self.spec.response] = df[self.spec.response].astype(fam.response_dtype)
        self.warnings.extend(fam.response_warnings(self.spec, df))   # separation, all-zero counts
        self.original_labels: dict[str, np.ndarray] = {}     # labels of the SUPPLIED column, fitted rows, before any recoding
        for cat in self.spec.categoricals:
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
        self._check_generated_names()
        bambi_priors = priors.to_bambi(self.spec, fam)
        for name in fam.fixed_parameters:                    # Bambi takes a fixed nu through the prior dict; a learned
            if name not in self.learned_parameters:          # one's truncated prior came from Priors.to_bambi (A13)
                bambi_priors[name] = getattr(self, name)
        self.bambi_model = bmb.Model(
            self.spec.bambi_formula, self.data, family=bambi_family,
            priors=bambi_priors,
            noncentered=True, center_predictors=self.effective_center_predictors,
        )
        self.bambi_model.build()
        self.pymc_model = self.bambi_model.backend.model
        if self.spec.correlated_terms:                       # brms's correlated slopes (A02): a post-build edit of Bambi's graph
            self.pymc_model = self._correlate_group_effects(self.pymc_model)
            self.bambi_model.backend.model = self.pymc_model  # what Bambi's fit() samples and cleans
        self.design = self._extract_design()
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
        for roles in self.group_coefficient_names.values():
            variable_names.extend(roles[k] for k in ("sd", "offset", "group_effect"))
        for roles in self.correlation_names.values():
            variable_names.extend(roles[k] for k in ("L", "cor"))
        # Bambi's default result cleaning drops every posterior variable ending in _offset.
        retained = list(self.spec.terms) + [roles["group_effect"] for roles in self.group_coefficient_names.values()]
        if any(name.endswith("_offset") for name in retained):
            raise FormulaError("generated name collision: coefficient/effect suffix '_offset' is reserved by Bambi; rename the column")
        coords = [roles["coordinate"] for roles in self.group_names.values()]
        coords += [roles[k] for roles in self.correlation_names.values() for k in ("coordinate", "coordinate_T")]
        coords += [f"{t.name}_dim" for t in self.spec.pop_terms if t.is_vector]
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
    def _extract_design(self) -> Design:
        comp = self.bambi_model.components[self.family_spec.parent]
        common = comp.design.common
        group = comp.design.group
        names = list(common.terms)
        offset_terms = [self.spec.offset.term] if self.spec.offset is not None else []
        if names != (["Intercept"] if self.spec.intercept else []) + list(self.spec.terms) + offset_terms:
            raise RuntimeError(f"unexpected common terms from Bambi: {names}")
        Xfull = np.asarray(common.design_matrix, dtype=float)
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
        if list(comp.common_terms) != list(self.spec.terms):
            raise RuntimeError(f"Bambi's coefficient terms {list(comp.common_terms)} differ from the formula's {list(self.spec.terms)}")
        if self.spec.intercept and not np.allclose(Xfull[:, 0], 1.0):
            raise RuntimeError("Bambi's first common column is not an intercept column")
        # Verify Bambi's columns term by term against each term's own design block on the data
        # (``pop_term_design``): a numeric column as given, a numeric-by-numeric product as one column, a
        # categorical or other interaction term as one column per coefficient level in the recorded order.
        col = int(self.spec.intercept)
        for term in self.spec.pop_terms:
            expected = pop_term_design(term, self.data, self.spec.categoricals)
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
            cats = [self.spec.categorical(c) for c in term.categorical]
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
            coords = [str(c) for c in self.pymc_model.coords.get(f"{term.name}_dim", ())]
            if coords != list(term.levels):
                raise RuntimeError(f"PyMC coordinates {term.name}_dim = {coords} differ from the coefficient levels {list(term.levels)}")
            col += len(term.levels)
        if col != Xfull.shape[1]:
            raise RuntimeError(f"Bambi's common design has {Xfull.shape[1]} columns, expected {col}")
        X_raw = Xfull[:, int(self.spec.intercept):]
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
        return Design(y=y, X_raw=X_raw, X_model=X_model, x_means=x_means,
                      predictors=self.spec.predictors, Z=Z, group_blocks=group_blocks,
                      centered=self.effective_center_predictors, trials=trials,
                      columns=self.spec.columns, categoricals=self.spec.categoricals, offset=offset)

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
        ``group_effect:x`` for each varying slope ``x`` of that term (see ``group_names``)."""
        single = self.spec.group_terms[0] if len(self.spec.group_terms) == 1 else None
        fam = self.family_spec
        out = {
            **({"intercept": "Intercept"} if self.spec.intercept else {}),
            **({k: v for k, v in self.group_names[single.factor].items() if k != "coordinate"} if single is not None else {}),
            "linpred": fam.parent,
            "mean": fam.mean_name,
            "y": self.spec.response if self.spec.trials is None else f"p({self.spec.response}, {self.spec.trials})",
            **{f"b:{t.name}": t.name for t in self.spec.pop_terms},
        }
        if fam.scale_parameter is not None:
            out[fam.scale_parameter] = fam.bambi_scale_parameter
        if fam.zero_parameter is not None:                    # "zi" or "hu" (B06): the PyMC variable keeps brms's name
            out[fam.zero_parameter] = fam.zero_parameter
        for name in self.learned_parameters:                  # a learned nu (A13): the PyMC variable of that name
            out[name] = name
        if fam.continuous and fam.parent == "mu":
            out["mu"] = "mu"                                  # the continuous families whose parent is the location mu expose it by name (the Beta's parent is logit_mu)
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

    def value_point(self, *, intercept: float | None = None, b, sd=None, z=None, L=None, sigma: float | None = None,
                    shape: float | None = None, phi: float | None = None, nu: float | None = None,
                    zi: float | None = None, hu: float | None = None) -> dict[str, np.ndarray]:
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
        log-odds coordinate (Stan's for ``real<lower=0,upper=1>``).
        """
        b = np.atleast_1d(np.asarray(b, dtype=float))
        if b.shape != (len(self.spec.columns),):
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
        point = {n["intercept"]: np.asarray(float(intercept))} if self.spec.intercept else {}
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
        if scale is not None:
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
        for term in self.spec.pop_terms:                      # b in design-column order: one entry per term column
            k = len(term.columns)
            point[term.name] = np.array(b[i:i + k]) if term.levels else np.asarray(b[i])
            i += k
        expected = {v.name for v in self.pymc_model.value_vars}
        if set(point) != expected:
            raise RuntimeError(f"value names {sorted(point)} do not match PyMC's {sorted(expected)}")
        return point

    # -- description -------------------------------------------------------------------
    def describe(self) -> str:
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
            return [(lvl, *pair) for lvl, pair in zip(t.levels, interaction_cells(*(s.categorical(c) for c in t.categorical)))]
        parts = []
        for t in s.pop_terms:
            term = t.name
            if t.kind == "categorical":
                for lvl in t.levels:
                    ind = f"1[{term}_i = {lvl}]"
                    parts.append(f"({ind} - p_{term}[{lvl}]) * b_{term}[{lvl}]" if d.centered else f"{ind} * b_{term}[{lvl}]")
            elif t.kind == "interaction" and len(t.categorical) == 1:
                zn = t.categorical[0]
                xn = next(c for c in t.operands if c != zn)
                for lvl in t.levels:
                    prod = f"x_{xn} * 1[{zn}_i = {lvl}]"
                    parts.append(f"({prod} - q_{term}[{lvl}]) * b_{term}[{lvl}]" if d.centered else f"{prod} * b_{term}[{lvl}]")
            elif t.kind == "interaction" and not t.categorical:
                xn, wn = t.operands
                prod = f"x_{xn} * x_{wn}"
                parts.append(f"({prod} - q_{term}) * b_{term}" if d.centered else f"{prod} * b_{term}")
            elif t.kind == "interaction":
                zn, hn = t.operands
                for lvl, lz, lh in cells(t):
                    prod = f"1[{zn}_i = {lz}] * 1[{hn}_i = {lh}]"
                    parts.append(f"({prod} - p_{term}[{lvl}]) * b_{term}[{lvl}]" if d.centered else f"{prod} * b_{term}[{lvl}]")
            elif t.kind == "transform":
                value = t.transform.expression(f"{t.transform.column}_i")
                parts.append(f"({value} - mean({term})) * b_{term}" if d.centered else f"{value} * b_{term}")
            else:
                parts.append(f"(x_{term} - mean(x_{term})) * b_{term}" if d.centered else f"x_{term} * b_{term}")
        xs = " + ".join(parts)
        fam = self.family_spec
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
                    + (["in-formula transform(s)"] if s.transform_terms else [])
                    + (["independent varying slope(s)"] if any(t.slopes and not t.has_correlation for t in s.group_terms) else [])
                    + (["correlated varying slope(s)"] if s.correlated_terms else []) + (["offset"] if s.offset is not None else [])
                    + (["derived grouping factor(s)"] if any(t.derived for t in s.group_terms) else []))
        lines = [
            f"AutoBayes {fam.tag} model" + (" with " + " and ".join(features) if features else "")
            + (" (population-level only)" if not s.has_group else ""),
            f"  formula   : {s.formula}",
        ]
        if s.canonical_formula != s.formula:
            lines += [f"  expanded  : {s.canonical_formula}   (canonical form handed to Bambi)"]
        lines += [f"  family    : {self.family}, {self.link} link", data_line]
        if s.offset is not None:
            lines += [f"  offset    : {s.offset.expression}, added to the linear predictor as written (no coefficient, no prior, "
                      f"never centred; brms offset()); range {d.offset.min():.4f} to {d.offset.max():.4f} over the fitted rows"]
        for t in s.transform_terms:
            tr, col = t.transform, d.X_raw[:, s.columns.index(t.name)]
            lines += [f"  {t.name:<10}: in-formula transform of {tr.column!r}, {tr.expression(f'{tr.column}_i')}, range {col.min():.4f} to "
                      f"{col.max():.4f} over the fitted rows; recomputed from {tr.column!r} on new rows; centred like any column; "
                      f"brms names its coefficient {tr.brms_name}"]
        for g, block in d.group_designs.items():
            lines += [f"  {'levels    ' if len(s.groups) == 1 else 'levels ' + g}: {', '.join(f'{j}={l}' for j, l in enumerate(block.levels))}"]
        for t in s.group_terms:
            if t.derived:
                ga, gb = t.operands
                name = _bambi_group_name(t.factor, "Intercept")
                lines += [f"  {t.factor:<10}: derived grouping factor, one level per observed ({ga}, {gb}) combination, labelled {ga}_{gb} as brms "
                          f"does (paste({ga}, {gb}, sep = \"_\")); unobserved combinations have no level; handed to Bambi as the column "
                          f"{t.column!r}, so its PyMC variables are {name}, {name}_sigma, {name}_offset"]
        for cat in s.categoricals:
            labels = self.data[cat.name].astype(str).to_numpy()
            props = ", ".join(f"{l} {np.mean(labels == l):.3f}" for l in cat.levels)
            lines += [
                f"  {cat.name:<10}: categorical, {'full-level coding' if cat.full_rank else 'treatment coding'}, reference {cat.reference!r}; levels (sorted) "
                f"{', '.join(cat.levels)}; coefficients b_{cat.name}[{', '.join(cat.coefficients)}]"
                + (f"; unused declared categories {list(cat.unused)}" if cat.unused else ""),
                f"              proportions among fitted rows: {props}",
            ]
        lines += [
            "",
            "Linear predictor",
            f"  {lp}_i = {a}" + (f" + {xs}" if xs else "") + (" + offset_i" if s.offset is not None else "")
            + "".join(f" + {effect(t.factor)}[{t.factor}_i]" + "".join(f" + x_{c},i * {slope_effect(t.factor, c)}[{t.factor}_i]" for c in t.slopes)
                      for t in s.group_terms),
        ]
        if s.offset is not None:
            per_row = f"log({s.offset.column}_i)" if s.offset.log else f"{s.offset.column}_i"
            lines += [f"  offset_i = {per_row}: the offset term, a known quantity added to every row's {lp}_i; "
                      "it has no coefficient and no prior and is not centred."]
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
        if d.centered:
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
        if s.intercept:
            lines += [f"  {a} ~ {describe_prior(pr.intercept)}"]
        lines += [f"  b_{col} ~ {describe_prior(coef[col])}" for col in s.columns]
        for (g, c), prior in pr.sd_priors(s).items():
            lines += [f"  sd_{sd_tag(g, c)} ~ {describe_prior(prior, lb=0)}"]
        for g, prior in pr.cor_priors(s).items():
            lines += [f"  Cor_{g} ~ {prior} (brms lkj({prior.params['eta']:g}) on the correlation matrix, placed on L_{g} as lkj_corr_cholesky)"
                      + ("   (brms default)" if prior.source == "default" else "")]
        lines += fam.likelihood_lines(self)
        if self.default_prior_labels:                         # priors filled in by ab.default_priors() (A07)
            from autobayes.default_priors import BRMS_VERSION
            lines += ["", "Default priors (ab.default_priors(); docs/default-priors.md)",
                      f"  {', '.join(self.default_prior_labels)}: brms {BRMS_VERSION}'s default_prior() for this formula, data and family,",
                      "  marked (brms default) above with brms's spelling. The Intercept's student_t(3, location, scale) has the median of",
                      "  the response on the link scale rounded to one decimal as its location (minus the mean offset; 0 under a logit or",
                      "  probit link) and max(2.5, its MAD rounded to one decimal) as its scale, which sd and sigma share; a flat prior is",
                      "  improper: it adds nothing to the log density, and prior_predictive() is refused."]
        for w in self.warnings:
            lines += ["", "WARNING: " + w]
        lines += ["", "PyMC variables sampled (value space)"]
        n = self.names
        roles = {n["intercept"]: "a_c" if d.centered else "a"} if s.intercept else {}
        for (g, c), names in self.group_coefficient_names.items():
            roles[f"{names['sd']}_log__"] = f"log(sd_{sd_tag(g, c)})"
            roles[names["offset"]] = ("z" if len(s.groups) == 1 else f"z_{g}") if c == "Intercept" else f"z_{c}|{g}"
        for g, names in self.correlation_names.items():
            roles[f"{names['L']}_cholesky_corr__"] = f"L_{g} on Stan's cholesky_corr coordinates (canonical partial correlations, arctanh scale)"
        if fam.scale_parameter is not None:
            roles[f"{n[fam.scale_parameter]}_log__"] = f"log({fam.scale_parameter})"
        if fam.zero_parameter is not None:
            roles[f"{n[fam.zero_parameter]}_logodds__"] = f"logit({fam.zero_parameter})"
        for name, lower in self.learned_parameters.items():
            roles[f"{n[name]}_interval__"] = f"log({name} - {lower:g})"
        for t in s.pop_terms:
            if t.is_vector:
                shown = ", ".join(f"({l})" for l in t.levels) if len(t.categorical) == 2 else ", ".join(t.levels)   # cells read (l, m)
                rows = ", ".join(f"{t.name}[{l}]" for l in t.levels)      # ArviZ names a row variable[coordinate]
                roles[t.name] = f"b_{t.name}[{shown}] (vector; dims {t.name}_dim, posterior rows {rows})"
        for v in self.pymc_model.value_vars:
            lines.append(f"  {v.name:<24} {roles.get(v.name, f'b_{v.name}')}")
        lines.append("Deterministic: " + "".join(f"{names['group_effect']} = {effect(g) if c == 'Intercept' else slope_effect(g, c)}, "
                                                 for (g, c), names in self.group_coefficient_names.items())
                     + "".join(f"{names['cor']} = Cor_{g} = L_{g} L_{g}^T, " for g, names in self.correlation_names.items()) + fam.deterministic_note)
        return "\n".join(lines)

    # -- priors ------------------------------------------------------------------------
    @property
    def default_prior_labels(self) -> list[str]:
        """The parameters whose prior came from ``ab.default_priors`` (brms's defaults; A07), labelled as the prior
        messages label them (``intercept``, ``b:x``, ``sd:g``, ``sd:x|g``, ``cor:g``, ``sigma``, ``nu``), in that order; empty
        when the user gave every prior (``zi`` or ``hu`` after the auxiliary parameter, B06). ``fit_record`` stores them
        under ``"default_priors"``."""
        s, pr, fam = self.spec, self.priors, self.family_spec
        found = ([("intercept", pr.intercept)] if s.intercept else []) + [(f"b:{k}", v) for k, v in pr.coefficient_priors(s).items()]
        found += [(f"sd:{sd_tag(g, c)}", v) for (g, c), v in pr.sd_priors(s).items()]
        found += [(f"cor:{g}", v) for g, v in pr.cor_priors(s).items()]
        if fam.scale_parameter is not None:
            found.append((fam.scale_parameter, getattr(pr, fam.scale_parameter)))
        if fam.zero_parameter is not None:
            found.append((fam.zero_parameter, getattr(pr, fam.zero_parameter)))
        if pr.nu is not None:
            found.append(("nu", pr.nu))
        return [label for label, prior in found if getattr(prior, "source", "user") == "default"]

    def prior_table(self) -> list[dict[str, str]]:
        """This model's priors in brms's table form (``Priors.table``): the rows brms's ``default_prior()`` prints for the
        same model, with this model's priors in brms's syntax."""
        return self.priors.table(self.spec, self.family_spec, center_predictors=self.effective_center_predictors)

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


def gaussian(formula: str, data: pd.DataFrame, priors: Priors, *, center_predictors: bool = True,
             categorical: Mapping[str, object] | None = None) -> Model:
    """Build the Gaussian model (identity link, residual SD ``sigma``), with a varying intercept when the
    formula has ``(1 | group)`` and population-only otherwise (M3c-1).
    ``categorical={column: reference_level}`` declares treatment-coded categorical predictors."""
    return Model(formula, data, priors, family="gaussian", center_predictors=center_predictors, categorical=categorical)


def bernoulli(formula: str, data: pd.DataFrame, priors: Priors, *, link: str = "logit", center_predictors: bool = True,
              categorical: Mapping[str, object] | None = None) -> Model:
    """Build the Bernoulli model (response strictly 0/1), with or without a varying intercept.

    ``link`` is ``"logit"`` (``p = expit(eta)``, the default) or ``"probit"`` (``p = Phi(eta)``), brms's
    ``bernoulli(link = ...)``; the intercept and slope priors are on that link's scale. brms's ``cloglog`` link is
    withheld and its ``cauchit`` link is not offered (docs/links.md).
    """
    return Model(formula, data, priors, family="bernoulli", link=link, center_predictors=center_predictors, categorical=categorical)


def binomial(formula: str, data: pd.DataFrame, priors: Priors, *, link: str = "logit", center_predictors: bool = True,
             categorical: Mapping[str, object] | None = None) -> Model:
    """Build the binomial model ``successes | trials(n) ~ x [+ (1 | group)]``, with or without a varying intercept;
    ``link`` as for ``bernoulli`` (``"logit"`` or ``"probit"``)."""
    return Model(formula, data, priors, family="binomial", link=link, center_predictors=center_predictors, categorical=categorical)


def student(formula: str, data: pd.DataFrame, priors: Priors, *, nu: float | None = None,
            center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Student-t response with an explicit sigma scale prior and degrees of freedom ``nu``, fixed or learned.

    ``nu=4`` (any finite ``nu > 1``) fixes them. ``nu=None``, the default, learns them under ``Priors(nu=...)``: a
    ``Gamma``, ``Exponential``, ``HalfNormal`` or ``HalfCauchy`` prior truncated to ``nu > 1`` and renormalised there, as
    brms's ``real<lower=1> nu`` (brms's default prior is ``gamma(2, 0.1)``, ``ab.Gamma(2, 0.1)``, which is not filled in
    silently; ``ab.default_priors(..., family="student")`` returns it with brms's other defaults; A13,
    docs/student-learned-nu.md). The expected response is mu. Sigma is not the residual standard deviation.
    Distributional auxiliary formulas are not supported.
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


def negbinomial(formula: str, data: pd.DataFrame, priors: Priors, *,
                center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Negative binomial count response (brms ``negbinomial``): log link, ``Var[y] = mu + mu^2 / shape``.

    Counts as for Poisson (integers in [0, 1e6], zero rows retained). The overdispersion parameter ``shape``
    takes its own positive-support prior, ``Priors(shape=ab.Gamma(0.01, 0.01))``; no sigma, nu, offset/exposure
    or zero-inflation.
    """
    return Model(formula, data, priors, family="negbinomial",
                 center_predictors=center_predictors, categorical=categorical)


def gamma(formula: str, data: pd.DataFrame, priors: Priors, *,
          center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Gamma response with a log link (brms ``Gamma(link = "log")``): ``E[y] = mu = exp(eta)``, ``Var[y] = mu^2 / shape``.

    Responses are strictly positive finite reals. The shape parameter takes its own positive-support prior,
    ``Priors(shape=ab.Gamma(0.01, 0.01))``; no sigma, nu, offset or other link (brms's default Gamma link is
    inverse; docs/brms-parity/divergences.md, D9).
    """
    return Model(formula, data, priors, family="gamma",
                 center_predictors=center_predictors, categorical=categorical)


def lognormal(formula: str, data: pd.DataFrame, priors: Priors, *,
              center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Lognormal response (brms ``lognormal()``): ``log(y) ~ Normal(mu, sigma)`` with the identity link on the log-scale
    location ``mu``; ``E[y] = exp(mu + sigma^2 / 2)``.

    Responses are strictly positive finite reals. ``sigma``, the SD of ``log y``, takes its own positive-support prior,
    ``Priors(sigma=ab.HalfNormal(1))``; the intercept and slope priors act on ``log y``. ``predict(kind="mean")`` returns
    brms's ``posterior_epred`` ``exp(mu + sigma^2 / 2)`` as ``mean``; ``kind="linear"`` returns ``mu``. No shape, nu,
    offset or distributional (``sigma ~ ...``) formula.
    """
    return Model(formula, data, priors, family="lognormal",
                 center_predictors=center_predictors, categorical=categorical)


def beta(formula: str, data: pd.DataFrame, priors: Priors, *,
         center_predictors: bool = True, categorical: Mapping[str, object] | None = None) -> Model:
    """Beta response with a logit link (brms ``Beta()``): proportions strictly inside (0, 1), ``E[y] = mu = expit(eta)``,
    ``Var[y] = mu (1 - mu) / (1 + phi)``.

    The precision ``phi`` takes its own positive-support prior, ``Priors(phi=ab.Gamma(0.01, 0.01))`` (brms class
    ``phi``); the intercept and slope priors are on the logit scale. ``predict(kind="mean")`` returns ``mu``; no sigma,
    shape, nu, other link, zero/one inflation or distributional (``phi ~ ...``) formula.
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
