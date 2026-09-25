"""The model: a thin, inspectable layer over a Bambi-built PyMC model.

Three families. ``"gaussian"`` (M0): identity link, residual SD ``sigma``. ``"bernoulli"``
(M1): logit link, ``p = expit(eta)``, no residual scale. ``"binomial"`` (M2): logit link on
``successes | trials(n)``, counts. All share the right-hand side ``[1 +] x1 [+ x2 ...]
[+ (1 | factor) ...]`` with numeric terms and, since M3a-1, treatment-coded categorical terms declared
with ``categorical={column: reference}``; centring; non-centred group effects when a group term is
present, and a population-only model without one (M3c-1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from collections.abc import Mapping

from autobayes.formula import Categorical, FormulaError, ModelSpec, binomial_counts, categorical_labels, parse_formula
from autobayes.priors import Priors


@dataclass(frozen=True)
class GroupDesign:
    """One factor's sorted levels, original-row coding and concatenated Z slice."""
    levels: tuple[str, ...]
    Z: np.ndarray
    group_index: np.ndarray
    column_slice: slice

    @property
    def J(self) -> int:
        return len(self.levels)


@dataclass(frozen=True)
class Design:
    """What the model was actually fit to. All arrays are dense NumPy."""

    y: np.ndarray                 # (n,)
    X_raw: np.ndarray             # (n, P) design columns as given (numeric values, 0/1 dummies), no intercept
    X_model: np.ndarray           # (n, P) what multiplies the coefficients: centred if center_predictors
    x_means: np.ndarray           # (P,) column means of X_raw (zeros if not centred); a dummy's mean is its level's proportion
    predictors: tuple[str, ...]   # numeric predictor terms
    Z: np.ndarray                 # concatenated one-hot blocks in formula order; (n, 0) without groups
    group_designs: Mapping[str, GroupDesign]
    centered: bool
    trials: np.ndarray | None = None   # binomial only: (n,) trials per fitted row; y holds the successes
    columns: tuple[str, ...] = ()      # design-column labels in order (``x``, ``z[b]``, ...); == predictors when numeric only
    categoricals: tuple[Categorical, ...] = ()

    def __post_init__(self):
        if not self.columns:
            object.__setattr__(self, "columns", tuple(self.predictors))

    @property
    def n(self) -> int:
        return len(self.y)

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


_FAMILIES = {
    # family: (Bambi parent parameter = the deterministic in the graph, link name)
    "gaussian": ("mu", "identity"),
    "bernoulli": ("logit_p", "logit"),
    "binomial": ("logit_p", "logit"),
}


class Model:
    """``y ~ x [+ (1 | factor) ...]`` with explicit priors; Gaussian, Bernoulli or binomial likelihood.

    Attributes
    ----------
    family : str               "gaussian" or "bernoulli"
    spec : ModelSpec
    priors : Priors
    bambi_model : bambi.Model  (already built)
    pymc_model : pymc.Model    the graph that is sampled; inspect it directly
    design : Design            design matrices and group coding
    """

    def __init__(self, formula: str, data: pd.DataFrame, priors: Priors, *,
                 family: str = "gaussian", center_predictors: bool = True,
                 categorical: Mapping[str, object] | None = None):
        import bambi as bmb

        if family not in _FAMILIES:
            raise ValueError(f"family must be one of {sorted(_FAMILIES)}, got {family!r}")
        self.family = family
        self.priors = priors
        self.center_predictors = bool(center_predictors)
        self.dropped_rows: tuple[int, ...] = ()     # binomial: positional indices of zero-trial rows
        self.warnings: list[str] = []

        spec0 = parse_formula(formula, None, family, categorical)
        self.effective_center_predictors = self.center_predictors and spec0.intercept
        priors.validate(spec0, family)                  # structure now; per-level coverage once levels are known
        cols = ([spec0.response, *spec0.predictors, *spec0.categorical_names] + list(spec0.groups)
                + ([spec0.trials] if spec0.trials else []))
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise FormulaError(f"columns not in data: {missing}")
        df = data[cols].copy().reset_index(drop=True)
        if family == "binomial":
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
            k_kept = df[spec0.response].to_numpy()
            n_kept = df[spec0.trials].to_numpy()
            if np.all(k_kept == 0) or np.all(k_kept == n_kept):
                which = "zero successes" if np.all(k_kept == 0) else "all successes"
                self.warnings.append(
                    (f"every fitted row has {which}. There is no population intercept; any separating direction "
                     "depends on the design. Check coefficient-prior sensitivity, without assuming an intercept direction.")
                    if not spec0.intercept else
                    f"every fitted row has {which}: the data are separated along the intercept "
                    "direction, the likelihood is monotone there, and the posterior in that direction "
                    "is set by the intercept prior's tail and the data's extent. Check prior sensitivity "
                    "(refit with a wider and a narrower intercept prior and report the movement)."
                )
        self.spec: ModelSpec = parse_formula(formula, df, family, categorical)   # levels over the fitted rows
        for group in self.spec.groups:
            df[group] = df[group].astype(str)
        if family == "bernoulli":
            df[self.spec.response] = df[self.spec.response].astype(int)
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

        if family == "bernoulli":
            from autobayes.likelihoods import bernoulli_logit_family

            bambi_family = bernoulli_logit_family()   # stable likelihood; see likelihoods.py
        elif family == "binomial":
            from autobayes.likelihoods import binomial_logit_family

            bambi_family = binomial_logit_family()
        else:
            bambi_family = family
        self._check_generated_names()
        self.bambi_model = bmb.Model(
            self.spec.bambi_formula, self.data, family=bambi_family,
            priors=priors.to_bambi(self.spec, family),
            noncentered=True, center_predictors=self.effective_center_predictors,
        )
        self.bambi_model.build()
        self.pymc_model = self.bambi_model.backend.model
        self.design = self._extract_design()
        if family in ("bernoulli", "binomial"):
            rv = self.pymc_model[self.names["y"]]
            obs = np.asarray(self.pymc_model.rvs_to_values[rv].eval())
            if not np.array_equal(obs, self.design.y.astype(int)):
                raise RuntimeError("Bambi's observed response does not equal the response column")
            if family == "binomial":
                consts = [np.asarray(i.data) for i in rv.owner.inputs if getattr(i, "data", None) is not None]
                if not any(c.shape == self.design.trials.shape and np.array_equal(c, self.design.trials) for c in consts):
                    raise RuntimeError("the trials attached to the likelihood do not equal the trials column")

    def _check_generated_names(self):
        """Refuse names that would alias a coefficient, effect, likelihood or coordinate."""
        variable_names = (["Intercept"] if self.spec.intercept else []) + list(self.spec.terms)
        variable_names += [self.names["y"], self.names["linpred"]]
        if self.family == "gaussian":
            variable_names += ["sigma"]
        for roles in self.group_names.values():
            variable_names.extend(roles[k] for k in ("sd", "offset", "group_effect"))
        # Bambi's default result cleaning drops every posterior variable ending in _offset.
        retained = list(self.spec.terms) + [roles["group_effect"] for roles in self.group_names.values()]
        if any(name.endswith("_offset") for name in retained):
            raise FormulaError("generated name collision: coefficient/effect suffix '_offset' is reserved by Bambi; rename the column")
        coords = [f"{g}__factor_dim" for g in self.spec.groups]
        coords += [f"{t}_dim" for t in self.spec.terms if self.spec.is_vector_term(t)]
        coords += ["__obs__"]
        names = variable_names + coords
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise FormulaError(f"generated variable/coordinate name collision: {duplicates}; rename the conflicting column")

    # -- design ------------------------------------------------------------------------
    def _extract_design(self) -> Design:
        comp = self.bambi_model.components[_FAMILIES[self.family][0]]
        common = comp.design.common
        group = comp.design.group
        names = list(common.terms)
        if names != (["Intercept"] if self.spec.intercept else []) + list(self.spec.terms):
            raise RuntimeError(f"unexpected common terms from Bambi: {names}")
        Xfull = np.asarray(common.design_matrix, dtype=float)
        if self.spec.intercept and not np.allclose(Xfull[:, 0], 1.0):
            raise RuntimeError("Bambi's first common column is not an intercept column")
        # Verify Bambi's columns one by one against the data: numeric columns as given, a
        # categorical term as one dummy per non-reference level in the recorded order.
        col = int(self.spec.intercept)
        for term in self.spec.terms:
            if term in self.spec.predictors:
                if not np.array_equal(Xfull[:, col], self.data[term].to_numpy(dtype=float)):
                    raise RuntimeError(f"Bambi's column for {term!r} does not equal the data column")
                col += 1
                continue
            is_inter = term in self.spec.interaction_names
            xname, zname = self.spec.interaction_of(term) if is_inter else (None, term)
            cat = self.spec.categorical(zname)
            bterm = common.terms[term]
            expected_kind = "interaction" if is_inter else "categoric"
            cols = self.spec.term_columns(term)
            coefficient_levels = self.spec.levels_of(term)
            if getattr(bterm, "kind", None) != expected_kind or list(bterm.levels) != list(coefficient_levels) \
                    or list(bterm.labels) != list(cols):
                raise RuntimeError(
                    f"Bambi coded {term!r} as {getattr(bterm, 'kind', None)} with levels {getattr(bterm, 'levels', None)} "
                    f"and labels {getattr(bterm, 'labels', None)}; expected {expected_kind} coding with reference "
                    f"{cat.reference!r}, levels {list(coefficient_levels)} and labels {list(cols)}"
                )
            labels = self.original_labels[zname]                       # the supplied values' labels, not the recoded column
            if not np.array_equal(labels, self.data[zname].astype(str).to_numpy()):
                raise RuntimeError(f"the recoded column {zname!r} does not reproduce the supplied labels")
            xval = self.data[xname].to_numpy(dtype=float) if is_inter else np.ones(len(labels))
            for j, lvl in enumerate(coefficient_levels):
                if not np.array_equal(Xfull[:, col + j], xval * (labels == lvl)):
                    what = f"x * indicator of level {lvl!r}" if is_inter else f"the indicator of level {lvl!r}"
                    raise RuntimeError(f"Bambi's column {cols[j]!r} is not {what}")
            if not is_inter and not np.array_equal(Xfull[:, col:col + len(coefficient_levels)].sum(1) + (0 if cat.full_rank else (labels == cat.reference)), np.ones(len(labels))):
                raise RuntimeError(f"the dummies of {term!r} and the reference indicator do not partition the rows")
            coords = [str(c) for c in self.pymc_model.coords.get(f"{term}_dim", ())]
            if coords != list(coefficient_levels):
                raise RuntimeError(f"PyMC coordinates {term}_dim = {coords} differ from the coefficient levels {list(coefficient_levels)}")
            col += len(coefficient_levels)
        if col != Xfull.shape[1]:
            raise RuntimeError(f"Bambi's common design has {Xfull.shape[1]} columns, expected {col}")
        X_raw = Xfull[:, int(self.spec.intercept):]
        group_designs = {}
        if not self.spec.has_group:
            if group is not None:
                raise RuntimeError("Bambi built a group design for a formula without a group-level term")
            Z = np.zeros((len(X_raw), 0))
        else:
            expected_terms = [f"1|{g}" for g in self.spec.groups]
            if group is None or list(group.terms) != expected_terms:
                raise RuntimeError("Bambi group terms differ from formula order")
            Z = group.design_matrix
            Z = np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, dtype=float)
            start = 0
            for g in self.spec.groups:
                name = f"1|{g}"
                levels = self.spec.levels_by_group[g]
                if tuple(str(v) for v in group.terms[name].groups) != levels:
                    raise RuntimeError(f"Bambi level order for {g!r} differs from sorted fitted labels")
                block_slice = slice(start, start + len(levels))
                if group.slices[name] != block_slice:
                    raise RuntimeError(f"Bambi group slice for {g!r} differs from formula order")
                labels = self.data[g].to_numpy()
                expected = np.column_stack([labels == level for level in levels]).astype(float)
                block = Z[:, block_slice]
                if not np.array_equal(block, expected) or not np.array_equal(block.sum(1), np.ones(len(X_raw))):
                    raise RuntimeError(f"group design for {g!r} does not encode original rows one-hot")
                coord = self.group_names[g]["coordinate"]
                if tuple(str(v) for v in self.pymc_model.coords[coord]) != levels:
                    raise RuntimeError(f"PyMC coordinate {coord!r} differs from fitted labels")
                group_designs[g] = GroupDesign(levels, block, block.argmax(1), block_slice)
                start += len(levels)
            if Z.shape != (len(X_raw), start):
                raise RuntimeError("unexpected concatenated group design shape")
        if self.effective_center_predictors:
            x_means = X_raw.mean(0)
            X_model = X_raw - x_means
        else:
            x_means = np.zeros(X_raw.shape[1])
            X_model = X_raw
        y = np.asarray(self.data[self.spec.response], dtype=float)
        trials = np.asarray(self.data[self.spec.trials], dtype=np.int64) if self.spec.trials else None
        return Design(y=y, X_raw=X_raw, X_model=X_model, x_means=x_means,
                      predictors=self.spec.predictors, Z=Z, group_designs=group_designs,
                      centered=self.effective_center_predictors, trials=trials,
                      columns=self.spec.columns, categoricals=self.spec.categoricals)

    # -- naming ------------------------------------------------------------------------
    @property
    def link(self) -> str:
        return _FAMILIES[self.family][1]

    @property
    def names(self) -> dict[str, str]:
        """PyMC variable names for each role in the model. ``"linpred"`` is the deterministic
        Bambi builds for the likelihood's parent parameter (``mu`` for Gaussian, ``logit_p``
        for Bernoulli); ``"mean"`` is the name ``predict`` gives the expected response (``mu``
        or ``p``). ``"sigma"`` exists only for the Gaussian family."""
        g = self.spec.groups[0] if len(self.spec.groups) == 1 else None
        out = {
            **({"intercept": "Intercept"} if self.spec.intercept else {}),
            **({"sd": f"1|{g}_sigma", "offset": f"1|{g}_offset", "group_effect": f"1|{g}"} if g is not None else {}),
            "linpred": _FAMILIES[self.family][0],
            "mean": "mu" if self.family == "gaussian" else "p",
            "y": self.spec.response if self.spec.trials is None else f"p({self.spec.response}, {self.spec.trials})",
            **{f"b:{p}": p for p in self.spec.terms},
        }
        if self.family == "gaussian":
            out["sigma"] = "sigma"
            out["mu"] = "mu"
        return out

    @property
    def group_names(self) -> dict[str, dict[str, str]]:
        """PyMC roles and coordinate for each factor, in formula order."""
        return {g: {"sd": f"1|{g}_sigma", "offset": f"1|{g}_offset",
                    "group_effect": f"1|{g}", "coordinate": f"{g}__factor_dim"}
                for g in self.spec.groups}

    @property
    def coefficients(self) -> tuple[tuple[str, str, int | None], ...]:
        """``(design column, PyMC variable, index into that variable or None)`` in design order:
        ``("x", "x", None)`` for a numeric slope, ``("z[b]", "z", 0)`` for a dummy coefficient
        (the variable ``z`` has dims ``z_dim`` with the coefficient levels as coordinates)."""
        return self.spec.column_terms

    def value_point(self, *, intercept: float | None = None, b, sd: float | Mapping[str, float] | None = None, z=None, sigma: float | None = None) -> dict[str, np.ndarray]:
        """A point in PyMC's sampled (value) space, from natural-scale quantities.

        ``intercept`` is the *model* intercept (at the predictor means when centred),
        ``b`` the coefficients in design-column order (``model.coefficients``: numeric slopes
        and dummy coefficients), ``sd`` and ``z`` exact-key factor mappings of positive
        scales and standard Normal offset vectors. One factor also accepts scalar/vector
        arguments; neither applies without groups. ``sigma`` is the residual
        SD (Gaussian only). Scales are stored on the log scale because that is what PyMC samples.
        """
        b = np.atleast_1d(np.asarray(b, dtype=float))
        if b.shape != (len(self.spec.columns),):
            raise ValueError(f"b must have {len(self.spec.columns)} entries, one per design column {list(self.spec.columns)}")
        def keyed(value, label):
            if isinstance(value, Mapping):
                if set(value) != set(self.spec.groups):
                    raise ValueError(f"{label} must cover exactly the grouping factors {list(self.spec.groups)}")
                return {g: value[g] for g in self.spec.groups}
            if len(self.spec.groups) != 1:
                raise ValueError(f"{label} requires an exact-key mapping for multiple grouping factors")
            return {self.spec.groups[0]: value}

        sd_by_group, z_by_group = {}, {}
        if not self.spec.has_group:
            if sd is not None or z is not None:
                raise ValueError("this model has no group-level term; sd and z do not apply to a population-only model")
        else:
            if sd is None or z is None:
                raise ValueError("sd and z are required for every varying intercept")
            sd_by_group, z_by_group = keyed(sd, "sd"), keyed(z, "z")
            for g, block in self.design.group_designs.items():
                z_by_group[g] = np.asarray(z_by_group[g], dtype=float)
                if z_by_group[g].shape != (block.J,) or not np.isfinite(z_by_group[g]).all():
                    raise ValueError(f"z for {g!r} must have {block.J} finite entries")
        if (sigma is None) != (self.family != "gaussian"):
            raise ValueError("sigma is required for the Gaussian family and must be omitted for Bernoulli and binomial")
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

        for g, roles in self.group_names.items():
            point[f"{roles['sd']}_log__"] = log_scale(sd_by_group[g], f"sd:{g}")
            point[roles["offset"]] = z_by_group[g]
        if sigma is not None:
            point[f"{n['sigma']}_log__"] = log_scale(sigma, "sigma")
        for i, (col, term, j) in enumerate(self.coefficients):
            if j is None:
                point[term] = np.asarray(b[i])
            else:
                point.setdefault(term, np.zeros(len(self.spec.levels_of(term))))
                point[term][j] = b[i]
        expected = {v.name for v in self.pymc_model.value_vars}
        if set(point) != expected:
            raise RuntimeError(f"value names {sorted(point)} do not match PyMC's {sorted(expected)}")
        return point

    # -- description -------------------------------------------------------------------
    def describe(self) -> str:
        s, d, pr = self.spec, self.design, self.priors
        def effect(g):
            return "u" if len(s.groups) == 1 else f"u_{g}"
        coef = pr.coefficient_priors(s)
        parts = []
        for term in s.terms:
            if term in s.categorical_names:
                cat = s.categorical(term)
                for lvl in cat.coefficients:
                    ind = f"1[{term}_i = {lvl}]"
                    parts.append(f"({ind} - p_{term}[{lvl}]) * b_{term}[{lvl}]" if d.centered else f"{ind} * b_{term}[{lvl}]")
            elif term in s.interaction_names:
                xn, zn = s.interaction_of(term)
                for lvl in s.levels_of(term):
                    prod = f"x_{xn} * 1[{zn}_i = {lvl}]"
                    parts.append(f"({prod} - q_{term}[{lvl}]) * b_{term}[{lvl}]" if d.centered else f"{prod} * b_{term}[{lvl}]")
            else:
                parts.append(f"(x_{term} - mean(x_{term})) * b_{term}" if d.centered else f"x_{term} * b_{term}")
        xs = " + ".join(parts)
        gaussian = self.family == "gaussian"
        binomial = self.family == "binomial"
        lp = "mu" if gaussian else "eta"
        a = f"a{'_c' if d.centered else ''}" if s.intercept else "0"
        tag = {"gaussian": "M0", "bernoulli": "M1", "binomial": "M2"}[self.family]
        grouping = ("; ".join(f"J = {block.J} levels of {g}" for g, block in d.group_designs.items())
                    if s.has_group else "no group-level term (population-level model)")
        data_line = f"  data      : n = {d.n} observations, {grouping}"
        if binomial:
            data_line = (f"  data      : {d.n} fitted rows, {int(d.trials.sum())} trials in total, "
                         f"{grouping}; {len(self.dropped_rows)} zero-trial row(s) dropped"
                         + (f" (positions {list(self.dropped_rows)[:10]}{'...' if len(self.dropped_rows) > 10 else ''})"
                            if self.dropped_rows else ""))
        lines = [
            f"AutoBayes {tag} model" + (" with categorical predictor(s)" if s.categoricals else "")
            + (" and interaction(s)" if s.interactions else "") + (" (population-level only)" if not s.has_group else ""),
            f"  formula   : {s.formula}",
        ]
        if s.canonical_formula != s.formula:
            lines += [f"  expanded  : {s.canonical_formula}   (canonical form handed to Bambi)"]
        lines += [f"  family    : {self.family}, {self.link} link", data_line]
        for g, block in d.group_designs.items():
            lines += [f"  {'levels    ' if len(s.groups) == 1 else 'levels ' + g}: {', '.join(f'{j}={l}' for j, l in enumerate(block.levels))}"]
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
            f"  {lp}_i = {a}" + (f" + {xs}" if xs else "") + "".join(f" + {effect(g)}[{g}_i]" for g in s.groups),
        ]
        scale = "response" if gaussian else "log-odds"
        same_group = " (same group)" if len(s.groups) == 1 else (" (same group effects)" if s.groups else "")
        for cat in s.categoricals:
            partners = [xn for xn, zn in s.interactions if zn == cat.name]
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
            partners = [zn for x, zn in s.interactions if x == xn]
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
        if binomial:
            lines += [f"  P(success) = p_i = expit(eta_i); {s.response}_i ~ Binomial({s.trials}_i, p_i); eta is on the log-odds scale."]
        elif not gaussian:
            lines += ["  P(y_i = 1) = p_i = expit(eta_i) = 1 / (1 + exp(-eta_i)); eta is on the log-odds scale."]
        if d.centered:
            what = ("expected response" if gaussian else
                    "per-trial success log-odds" if binomial else "log-odds of y = 1")
            where = "at the predictor means" if not s.categoricals else \
                "at the numeric predictor means and at the sample proportions of the categorical levels"
            if s.interactions:
                where = "at the fitted-row means of every expanded design column (x, 1[z = l], x * 1[z = l])"
            lines += [
                "  Predictors are centred at their sample means (center_predictors=True, as in",
                f"  brms and Bambi)" + ("; a dummy column's mean is its level's proportion p_z[l]." if s.categoricals else "."),
            ]
            if s.interactions:
                qs = []
                for xn, zn in s.interactions:
                    lab = self.original_labels[zn]; xv = self.data[xn].to_numpy(dtype=float)
                    for lvl in s.levels_of(f"{xn}:{zn}"):
                        m = lab == lvl
                        qs.append(f"q_{xn}:{zn}[{lvl}] = {xv[m].sum() / len(xv):.4f} = p_{zn}[{lvl}] {m.mean():.3f} * mean({xn} | {zn} = {lvl}) {xv[m].mean():.4f}"
                                  f" (not mean({xn}) {xv.mean():.4f} * p)")
                lines += ["  Each product column x * 1[z = l] is centred at its OWN mean q, which is the level's proportion"] + \
                         [f"  times the mean of x within that level: {q}" for q in qs] + \
                         ["  That point of the expanded design is not a realisable covariate row (a row has x*1[z=l] equal to x or 0)."]
            lines += [
                f"  a_c is the {what} {where}{' (group effect zero)' if s.has_group else ''}; the",
                "  prior on the intercept applies to a_c. The posterior reports the raw-scale",
                "  intercept a = a_c - sum_p mean(x_p) * b_p" + (" - sum_l p_z[l] * b_z[l]" if s.categoricals else "")
                + (" - sum_(x,z),l q_x:z[l] * b_x:z[l]" if len(s.interactions) > 1 else (" - sum_l q_x:z[l] * b_x:z[l]" if s.interactions else "")) + ", computed after sampling"
                + (": the linear predictor at x = 0 and at the reference level." if s.categoricals else "."),
            ]
            if s.categoricals and not gaussian:
                lines += ["  expit(a_c) is the probability at that proportion-weighted predictor; it is NOT the",
                          "  average probability across the levels (the link is nonlinear)."]
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
            for g, block in d.group_designs.items():
                lines += [f"  u_{g}[j] = sd_{g} * z_{g}[j], z_{g}[j] ~ Normal(0, 1), j = 0..{block.J - 1}"]
            lines += ["  Independent priors across factors; group effects are not mean-centered."]
        else:
            lines += ["", "Group effects", "  none: a population-only model (no '(1 | group)' term)"]
        lines += ["", "Priors" + ("" if gaussian else " (location coefficients on the logit scale)")]
        if s.intercept:
            lines += [f"  {a} ~ {pr.intercept}"]
        lines += [f"  b_{col} ~ {coef[col]}" for col in s.columns]
        for g, prior in pr.group_priors(s).items():
            lines += [f"  sd_{g} ~ {prior}"]
        if gaussian:
            lines += [f"  sigma ~ {pr.sigma}", "", "Likelihood", "  y_i ~ Normal(mu_i, sigma)"]
        elif binomial:
            lines += [
                "", "Likelihood",
                f"  log p({s.response}_i | {s.trials}_i, eta_i) = log C({s.trials}_i, {s.response}_i)"
                f" - {s.response}_i softplus(-eta_i) - ({s.trials}_i - {s.response}_i) softplus(eta_i)",
                "  with log C computed inside the graph from the evaluated value and a softplus whose gradient",
                "  is finite on both the PyTensor and the JAX path. Validated (docs/m2-decision.md) for counts",
                "  up to 1e6 per row at logits to 1e6 against a 60-digit reference; larger counts are refused.",
                "  Zero-trial rows are dropped before design construction; centering, when effective, uses only fitted rows.",
            ]
        else:
            lines += [
                "", "Likelihood", "  y_i ~ Bernoulli(p_i), sampled as log p(y_i | eta_i) = y_i * eta_i - softplus(eta_i)",
                "  with a softplus whose value and gradient are exact for every finite eta on both the",
                "  PyTensor and the JAX path (autobayes/likelihoods.py; Bambi's built-in bernoulli family",
                "  is not used because its likelihood is floored at machine epsilon for |eta| > ~37).",
            ]
        for w in self.warnings:
            lines += ["", "WARNING: " + w]
        lines += ["", "PyMC variables sampled (value space)"]
        n = self.names
        roles = {n["intercept"]: "a_c" if d.centered else "a"} if s.intercept else {}
        for g, names in self.group_names.items():
            roles[f"{names['sd']}_log__"] = f"log(sd_{g})"
            roles[names["offset"]] = "z" if len(s.groups) == 1 else f"z_{g}"
        if gaussian:
            roles[f"{n['sigma']}_log__"] = "log(sigma)"
        for term in (*s.categorical_names, *s.interaction_names):
            roles[term] = f"b_{term}[{', '.join(s.levels_of(term))}] (vector; dims {term}_dim, posterior rows {', '.join(s.term_columns(term))})"
        for v in self.pymc_model.value_vars:
            lines.append(f"  {v.name:<24} {roles.get(v.name, f'b_{v.name}')}")
        lines.append("Deterministic: " + "".join(f"{names['group_effect']} = {effect(g)}, " for g, names in self.group_names.items()) + f"{n['linpred']} = linear predictor"
                     + ("" if gaussian else "; p = expit(logit_p)" + (", expected = n p and count draws" if binomial else "")
                        + " are computed by predict()"))
        return "\n".join(lines)

    # -- convenience -------------------------------------------------------------------
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


def bernoulli(formula: str, data: pd.DataFrame, priors: Priors, *, center_predictors: bool = True,
              categorical: Mapping[str, object] | None = None) -> Model:
    """Build the Bernoulli model (logit link, response strictly 0/1), with or without a varying intercept."""
    return Model(formula, data, priors, family="bernoulli", center_predictors=center_predictors, categorical=categorical)


def binomial(formula: str, data: pd.DataFrame, priors: Priors, *, center_predictors: bool = True,
             categorical: Mapping[str, object] | None = None) -> Model:
    """Build the binomial model ``successes | trials(n) ~ x [+ (1 | group)]``, with or without a varying intercept."""
    return Model(formula, data, priors, family="binomial", center_predictors=center_predictors, categorical=categorical)
