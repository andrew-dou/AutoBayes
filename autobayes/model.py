"""The model: a thin, inspectable layer over a Bambi-built PyMC model.

Three families. ``"gaussian"`` (M0): identity link, residual SD ``sigma``. ``"bernoulli"``
(M1): logit link, ``p = expit(eta)``, no residual scale. ``"binomial"`` (M2): logit link on
``successes | trials(n)``, counts. All share the right-hand side ``[1 +] x1 [+ x2 ...] +
(1 | group)`` with numeric terms and, since M3a-1, treatment-coded categorical terms declared
with ``categorical={column: reference}``; centring; non-centred group effects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from collections.abc import Mapping

from autobayes.formula import Categorical, FormulaError, ModelSpec, binomial_counts, categorical_labels, parse_formula
from autobayes.priors import Priors


@dataclass(frozen=True)
class Design:
    """What the model was actually fit to. All arrays are dense NumPy."""

    y: np.ndarray                 # (n,)
    X_raw: np.ndarray             # (n, P) design columns as given (numeric values, 0/1 dummies), no intercept
    X_model: np.ndarray           # (n, P) what multiplies the coefficients: centred if center_predictors
    x_means: np.ndarray           # (P,) column means of X_raw (zeros if not centred); a dummy's mean is its level's proportion
    predictors: tuple[str, ...]   # numeric predictor terms
    Z: np.ndarray                 # (n, J) one-hot group indicators
    group_index: np.ndarray       # (n,) integer index into levels
    levels: tuple[str, ...]       # sorted level labels; column j of Z is levels[j]
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

    @property
    def J(self) -> int:
        return len(self.levels)


_FAMILIES = {
    # family: (Bambi parent parameter = the deterministic in the graph, link name)
    "gaussian": ("mu", "identity"),
    "bernoulli": ("logit_p", "logit"),
    "binomial": ("logit_p", "logit"),
}


class Model:
    """``y ~ x + (1 | group)`` with explicit priors, Gaussian or Bernoulli likelihood.

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
        priors.validate(spec0, family)                  # structure now; per-level coverage once levels are known
        cols = [spec0.response, *spec0.predictors, *spec0.categorical_names, spec0.group] + ([spec0.trials] if spec0.trials else [])
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
                    f"every fitted row has {which}: the data are separated along the intercept "
                    "direction, the likelihood is monotone there, and the posterior in that direction "
                    "is set by the intercept prior's tail and the data's extent. Check prior sensitivity "
                    "(refit with a wider and a narrower intercept prior and report the movement)."
                )
        self.spec: ModelSpec = parse_formula(formula, df, family, categorical)   # levels over the fitted rows
        df[self.spec.group] = df[self.spec.group].astype(str)
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
        self.bambi_model = bmb.Model(
            self.spec.bambi_formula, self.data, family=bambi_family,
            priors=priors.to_bambi(self.spec, self.spec.group, family),
            noncentered=True, center_predictors=self.center_predictors,
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

    # -- design ------------------------------------------------------------------------
    def _extract_design(self) -> Design:
        comp = self.bambi_model.components[_FAMILIES[self.family][0]]
        common = comp.design.common
        group = comp.design.group
        names = list(common.terms)
        if names[0] != "Intercept" or names[1:] != list(self.spec.terms):
            raise RuntimeError(f"unexpected common terms from Bambi: {names}")
        Xfull = np.asarray(common.design_matrix, dtype=float)
        if not np.allclose(Xfull[:, 0], 1.0):
            raise RuntimeError("Bambi's first common column is not an intercept column")
        # Verify Bambi's columns one by one against the data: numeric columns as given, a
        # categorical term as one dummy per non-reference level in the recorded order.
        col = 1
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
            if getattr(bterm, "kind", None) != expected_kind or list(bterm.levels) != list(cat.coefficients) \
                    or list(bterm.labels) != list(cols):
                raise RuntimeError(
                    f"Bambi coded {term!r} as {getattr(bterm, 'kind', None)} with levels {getattr(bterm, 'levels', None)} "
                    f"and labels {getattr(bterm, 'labels', None)}; expected {expected_kind} coding with reference "
                    f"{cat.reference!r}, levels {list(cat.coefficients)} and labels {list(cols)}"
                )
            labels = self.original_labels[zname]                       # the supplied values' labels, not the recoded column
            if not np.array_equal(labels, self.data[zname].astype(str).to_numpy()):
                raise RuntimeError(f"the recoded column {zname!r} does not reproduce the supplied labels")
            xval = self.data[xname].to_numpy(dtype=float) if is_inter else np.ones(len(labels))
            for j, lvl in enumerate(cat.coefficients):
                if not np.array_equal(Xfull[:, col + j], xval * (labels == lvl)):
                    what = f"x * indicator of level {lvl!r}" if is_inter else f"the indicator of level {lvl!r}"
                    raise RuntimeError(f"Bambi's column {cols[j]!r} is not {what}")
            if not is_inter and not np.array_equal(Xfull[:, col:col + len(cat.coefficients)].sum(1) + (labels == cat.reference), np.ones(len(labels))):
                raise RuntimeError(f"the dummies of {term!r} and the reference indicator do not partition the rows")
            coords = [str(c) for c in self.pymc_model.coords.get(f"{term}_dim", ())]
            if coords != list(cat.coefficients):
                raise RuntimeError(f"PyMC coordinates {term}_dim = {coords} differ from the coefficient levels {list(cat.coefficients)}")
            col += len(cat.coefficients)
        if col != Xfull.shape[1]:
            raise RuntimeError(f"Bambi's common design has {Xfull.shape[1] - 1} columns, expected {col - 1}")
        X_raw = Xfull[:, 1:]
        term = group.terms[self.spec.group_term]
        levels = tuple(str(v) for v in term.groups)
        if levels != self.spec.levels:
            raise RuntimeError(f"Bambi level order {levels} differs from sorted order {self.spec.levels}")
        Z = group.design_matrix
        Z = np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, dtype=float)
        if Z.shape != (len(X_raw), len(levels)) or not np.array_equal(Z.sum(1), np.ones(len(X_raw))):
            raise RuntimeError("group design matrix is not one-hot")
        group_index = Z.argmax(1)
        if self.center_predictors:
            x_means = X_raw.mean(0)
            X_model = X_raw - x_means
        else:
            x_means = np.zeros(X_raw.shape[1])
            X_model = X_raw
        y = np.asarray(self.data[self.spec.response], dtype=float)
        trials = np.asarray(self.data[self.spec.trials], dtype=np.int64) if self.spec.trials else None
        return Design(y=y, X_raw=X_raw, X_model=X_model, x_means=x_means,
                      predictors=self.spec.predictors, Z=Z, group_index=group_index,
                      levels=levels, centered=self.center_predictors, trials=trials,
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
        g = self.spec.group
        out = {
            "intercept": "Intercept",
            "sd": f"1|{g}_sigma",
            "offset": f"1|{g}_offset",
            "group_effect": f"1|{g}",
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
    def coefficients(self) -> tuple[tuple[str, str, int | None], ...]:
        """``(design column, PyMC variable, index into that variable or None)`` in design order:
        ``("x", "x", None)`` for a numeric slope, ``("z[b]", "z", 0)`` for a dummy coefficient
        (the variable ``z`` has dims ``z_dim`` with the coefficient levels as coordinates)."""
        return self.spec.column_terms

    def value_point(self, *, intercept: float, b, sd: float, z, sigma: float | None = None) -> dict[str, np.ndarray]:
        """A point in PyMC's sampled (value) space, from natural-scale quantities.

        ``intercept`` is the *model* intercept (at the predictor means when centred),
        ``b`` the coefficients in design-column order (``model.coefficients``: numeric slopes
        and dummy coefficients), ``sd`` the group SD, ``z`` the J standard normal offsets,
        ``sigma`` the residual SD (Gaussian only). Scales are stored on the log scale because
        that is what PyMC samples.
        """
        b = np.atleast_1d(np.asarray(b, dtype=float))
        if b.shape != (len(self.spec.columns),):
            raise ValueError(f"b must have {len(self.spec.columns)} entries, one per design column {list(self.spec.columns)}")
        z = np.asarray(z, dtype=float)
        if z.shape != (self.design.J,):
            raise ValueError(f"z must have {self.design.J} entries")
        if (sigma is None) != (self.family != "gaussian"):
            raise ValueError("sigma is required for the Gaussian family and must be omitted for Bernoulli and binomial")
        n = self.names
        point = {n["intercept"]: np.asarray(float(intercept)),
                 f"{n['sd']}_log__": np.asarray(np.log(float(sd))),
                 n["offset"]: z}
        if sigma is not None:
            point[f"{n['sigma']}_log__"] = np.asarray(np.log(float(sigma)))
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
        a = f"a{'_c' if d.centered else ''}"
        tag = {"gaussian": "M0", "bernoulli": "M1", "binomial": "M2"}[self.family]
        data_line = f"  data      : n = {d.n} observations, J = {d.J} levels of {s.group}"
        if binomial:
            data_line = (f"  data      : {d.n} fitted rows, {int(d.trials.sum())} trials in total, "
                         f"J = {d.J} levels of {s.group}; {len(self.dropped_rows)} zero-trial row(s) dropped"
                         + (f" (positions {list(self.dropped_rows)[:10]}{'...' if len(self.dropped_rows) > 10 else ''})"
                            if self.dropped_rows else ""))
        lines = [
            f"AutoBayes {tag} model" + (" with categorical predictor(s)" if s.categoricals else "")
            + (" and interaction(s)" if s.interactions else ""),
            f"  formula   : {s.formula}",
        ]
        if s.canonical_formula != s.formula:
            lines += [f"  expanded  : {s.canonical_formula}   (canonical form handed to Bambi)"]
        lines += [
            f"  family    : {self.family}, {self.link} link",
            data_line,
            f"  levels    : {', '.join(f'{j}={l}' for j, l in enumerate(d.levels))}",
        ]
        for cat in s.categoricals:
            labels = self.data[cat.name].astype(str).to_numpy()
            props = ", ".join(f"{l} {np.mean(labels == l):.3f}" for l in cat.levels)
            lines += [
                f"  {cat.name:<10}: categorical, treatment coding, reference {cat.reference!r}; levels (sorted) "
                f"{', '.join(cat.levels)}; coefficients b_{cat.name}[{', '.join(cat.coefficients)}]"
                + (f"; unused declared categories {list(cat.unused)}" if cat.unused else ""),
                f"              proportions among fitted rows: {props}",
            ]
        lines += [
            "",
            "Linear predictor",
            f"  {lp}_i = {a}" + (f" + {xs}" if xs else "") + f" + u[{s.group}_i]",
        ]
        scale = "response" if gaussian else "log-odds"
        for cat in s.categoricals:
            partners = [xn for xn, zn in s.interactions if zn == cat.name]
            if partners:
                xn = partners[0]
                lines += [f"  b_{cat.name}[l] is the difference between level l and the reference {cat.reference!r} on the "
                          f"{scale} scale at {xn} = 0 (same group); at another value of {xn} the difference is "
                          f"b_{cat.name}[l] + {xn} * b_{xn}:{cat.name}[l], because the levels have different slopes. "
                          f"The reference has no coefficient."]
            else:
                lines += [f"  b_{cat.name}[l] is the difference between level l and the reference {cat.reference!r} "
                          f"on the {scale} scale at the same predictors and group; the reference has no coefficient."]
        for xn, zn in s.interactions:
            ref = s.categorical(zn).reference
            lines += [f"  {xn}:{zn}: the reference level {ref!r} has the raw intercept a and slope b_{xn}; level l has intercept "
                      f"difference b_{zn}[l] and slope difference b_{xn}:{zn}[l], so its slope in {xn} is b_{xn} + b_{xn}:{zn}[l]."]
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
                f"  a_c is the {what} {where} (group effect zero); the",
                "  prior on the intercept applies to a_c. The posterior reports the raw-scale",
                "  intercept a = a_c - sum_p mean(x_p) * b_p" + (" - sum_l p_z[l] * b_z[l]" if s.categoricals else "")
                + (" - sum_l q_x:z[l] * b_x:z[l]" if s.interactions else "") + ", computed after sampling"
                + (": the linear predictor at x = 0 and at the reference level." if s.categoricals else "."),
            ]
            if s.categoricals and not gaussian:
                lines += ["  expit(a_c) is the probability at that proportion-weighted predictor; it is NOT the",
                          "  average probability across the levels (the link is nonlinear)."]
        else:
            lines += ["  Predictors are not centred (center_predictors=False); a is the raw-scale intercept"
                      + (" (x = 0, reference level, group effect zero)." if s.categoricals else ".")]
        lines += [
            "",
            "Group effects (non-centred)",
            f"  u[j] = sd_{s.group} * z[j],  z[j] ~ Normal(0, 1),  j = 0..{d.J - 1}",
            "",
            "Priors" + ("" if gaussian else " (a_c and b on the logit scale)"),
            f"  {a} ~ {pr.intercept}",
        ]
        lines += [f"  b_{col} ~ {coef[col]}" for col in s.columns]
        lines += [f"  sd_{s.group} ~ {pr.sd}"]
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
                "  Zero-trial rows are dropped before centring; predictors are centred at the means of the fitted rows.",
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
        roles = {n["intercept"]: "a_c" if d.centered else "a",
                 f"{n['sd']}_log__": f"log(sd_{s.group})",
                 n["offset"]: "z"}
        if gaussian:
            roles[f"{n['sigma']}_log__"] = "log(sigma)"
        for term in (*s.categorical_names, *s.interaction_names):
            roles[term] = f"b_{term}[{', '.join(s.levels_of(term))}] (vector; dims {term}_dim, posterior rows {', '.join(s.term_columns(term))})"
        for v in self.pymc_model.value_vars:
            lines.append(f"  {v.name:<24} {roles.get(v.name, f'b_{v.name}')}")
        lines.append(f"Deterministic: {n['group_effect']} = u, {n['linpred']} = linear predictor"
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
    """Build the M0 Gaussian varying-intercept model (identity link, residual SD ``sigma``).
    ``categorical={column: reference_level}`` declares treatment-coded categorical predictors."""
    return Model(formula, data, priors, family="gaussian", center_predictors=center_predictors, categorical=categorical)


def bernoulli(formula: str, data: pd.DataFrame, priors: Priors, *, center_predictors: bool = True,
              categorical: Mapping[str, object] | None = None) -> Model:
    """Build the M1 Bernoulli varying-intercept model (logit link, response strictly 0/1)."""
    return Model(formula, data, priors, family="bernoulli", center_predictors=center_predictors, categorical=categorical)


def binomial(formula: str, data: pd.DataFrame, priors: Priors, *, center_predictors: bool = True,
             categorical: Mapping[str, object] | None = None) -> Model:
    """Build the M2 binomial varying-intercept model: ``successes | trials(n) ~ x + (1 | group)``."""
    return Model(formula, data, priors, family="binomial", center_predictors=center_predictors, categorical=categorical)
