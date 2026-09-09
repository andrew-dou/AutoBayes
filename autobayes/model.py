"""The model: a thin, inspectable layer over a Bambi-built PyMC model.

Two families. ``"gaussian"`` (M0): identity link, residual SD ``sigma``. ``"bernoulli"``
(M1): logit link, ``p = expit(eta)``, no residual scale. Both share the formula subset
``y ~ [1 +] x1 [+ x2 ...] + (1 | group)``, centring, and non-centred group effects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from autobayes.formula import ModelSpec, parse_formula
from autobayes.priors import Priors


@dataclass(frozen=True)
class Design:
    """What the model was actually fit to. All arrays are dense NumPy."""

    y: np.ndarray                 # (n,)
    X_raw: np.ndarray             # (n, p) predictors as given, no intercept column
    X_model: np.ndarray           # (n, p) what multiplies the slopes: centred if center_predictors
    x_means: np.ndarray           # (p,) column means of X_raw (zeros if not centred)
    predictors: tuple[str, ...]
    Z: np.ndarray                 # (n, J) one-hot group indicators
    group_index: np.ndarray       # (n,) integer index into levels
    levels: tuple[str, ...]       # sorted level labels; column j of Z is levels[j]
    centered: bool

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
                 family: str = "gaussian", center_predictors: bool = True):
        import bambi as bmb

        if family not in _FAMILIES:
            raise ValueError(f"family must be one of {sorted(_FAMILIES)}, got {family!r}")
        self.family = family
        self.spec: ModelSpec = parse_formula(formula, data, family)
        self.priors = priors
        priors.validate(self.spec.predictors, family)
        self.center_predictors = bool(center_predictors)

        cols = [self.spec.response, *self.spec.predictors, self.spec.group]
        df = data[cols].copy()
        df[self.spec.group] = df[self.spec.group].astype(str)
        if family == "bernoulli":
            df[self.spec.response] = df[self.spec.response].astype(int)
        self.data = df.reset_index(drop=True)

        if family == "bernoulli":
            from autobayes.likelihoods import bernoulli_logit_family

            bambi_family = bernoulli_logit_family()   # stable likelihood; see likelihoods.py
        else:
            bambi_family = family
        self.bambi_model = bmb.Model(
            self.spec.formula, self.data, family=bambi_family,
            priors=priors.to_bambi(self.spec.predictors, self.spec.group, family),
            noncentered=True, center_predictors=self.center_predictors,
        )
        self.bambi_model.build()
        self.pymc_model = self.bambi_model.backend.model
        self.design = self._extract_design()
        if family == "bernoulli":
            rv = self.pymc_model[self.spec.response]
            obs = np.asarray(self.pymc_model.rvs_to_values[rv].eval())
            if not np.array_equal(obs, self.design.y.astype(int)):
                raise RuntimeError("Bambi's observed response does not equal the 0/1 column (success coding)")

    # -- design ------------------------------------------------------------------------
    def _extract_design(self) -> Design:
        comp = self.bambi_model.components[_FAMILIES[self.family][0]]
        common = comp.design.common
        group = comp.design.group
        names = list(common.terms)
        if names[0] != "Intercept" or names[1:] != list(self.spec.predictors):
            raise RuntimeError(f"unexpected common terms from Bambi: {names}")
        Xfull = np.asarray(common.design_matrix, dtype=float)
        if not np.allclose(Xfull[:, 0], 1.0):
            raise RuntimeError("Bambi's first common column is not an intercept column")
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
        return Design(y=y, X_raw=X_raw, X_model=X_model, x_means=x_means,
                      predictors=self.spec.predictors, Z=Z, group_index=group_index,
                      levels=levels, centered=self.center_predictors)

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
            "y": self.spec.response,
            **{f"b:{p}": p for p in self.spec.predictors},
        }
        if self.family == "gaussian":
            out["sigma"] = "sigma"
            out["mu"] = "mu"
        return out

    def value_point(self, *, intercept: float, b, sd: float, z, sigma: float | None = None) -> dict[str, np.ndarray]:
        """A point in PyMC's sampled (value) space, from natural-scale quantities.

        ``intercept`` is the *model* intercept (at the predictor means when centred),
        ``b`` the slopes in predictor order, ``sd`` the group SD, ``z`` the J standard
        normal offsets, ``sigma`` the residual SD (Gaussian only). Scales are stored on the
        log scale because that is what PyMC samples.
        """
        b = np.atleast_1d(np.asarray(b, dtype=float))
        if b.shape != (len(self.spec.predictors),):
            raise ValueError(f"b must have {len(self.spec.predictors)} entries")
        z = np.asarray(z, dtype=float)
        if z.shape != (self.design.J,):
            raise ValueError(f"z must have {self.design.J} entries")
        if (sigma is None) != (self.family == "bernoulli"):
            raise ValueError("sigma is required for the Gaussian family and must be omitted for Bernoulli")
        n = self.names
        point = {n["intercept"]: np.asarray(float(intercept)),
                 f"{n['sd']}_log__": np.asarray(np.log(float(sd))),
                 n["offset"]: z}
        if sigma is not None:
            point[f"{n['sigma']}_log__"] = np.asarray(np.log(float(sigma)))
        for i, p in enumerate(self.spec.predictors):
            point[p] = np.asarray(b[i])
        expected = {v.name for v in self.pymc_model.value_vars}
        if set(point) != expected:
            raise RuntimeError(f"value names {sorted(point)} do not match PyMC's {sorted(expected)}")
        return point

    # -- description -------------------------------------------------------------------
    def describe(self) -> str:
        s, d, pr = self.spec, self.design, self.priors
        slopes = pr.slope_priors(s.predictors)
        xs = " + ".join(
            (f"(x_{p} - mean(x_{p})) * b_{p}" if d.centered else f"x_{p} * b_{p}") for p in s.predictors
        )
        gaussian = self.family == "gaussian"
        lp = "mu" if gaussian else "eta"
        a = f"a{'_c' if d.centered else ''}"
        lines = [
            f"AutoBayes {'M0' if gaussian else 'M1'} model",
            f"  formula   : {s.formula}",
            f"  family    : {self.family}, {self.link} link",
            f"  data      : n = {d.n} observations, J = {d.J} levels of {s.group}",
            f"  levels    : {', '.join(f'{j}={l}' for j, l in enumerate(d.levels))}",
            "",
            "Linear predictor",
            f"  {lp}_i = {a}" + (f" + {xs}" if xs else "") + f" + u[{s.group}_i]",
        ]
        if not gaussian:
            lines += ["  P(y_i = 1) = p_i = expit(eta_i) = 1 / (1 + exp(-eta_i)); eta is on the log-odds scale."]
        if d.centered:
            what = "expected response" if gaussian else "log-odds of y = 1"
            lines += [
                "  Predictors are centred at their sample means (center_predictors=True, as in",
                f"  brms and Bambi). a_c is the {what} at the predictor means; the",
                "  prior on the intercept applies to a_c. The posterior reports the raw-scale",
                "  intercept a = a_c - sum_p mean(x_p) * b_p, computed after sampling.",
            ]
        else:
            lines += ["  Predictors are not centred (center_predictors=False); a is the raw-scale intercept."]
        lines += [
            "",
            "Group effects (non-centred)",
            f"  u[j] = sd_{s.group} * z[j],  z[j] ~ Normal(0, 1),  j = 0..{d.J - 1}",
            "",
            "Priors" + ("" if gaussian else " (a_c and b on the logit scale)"),
            f"  {a} ~ {pr.intercept}",
        ]
        lines += [f"  b_{p} ~ {slopes[p]}" for p in s.predictors]
        lines += [f"  sd_{s.group} ~ {pr.sd}"]
        if gaussian:
            lines += [f"  sigma ~ {pr.sigma}", "", "Likelihood", "  y_i ~ Normal(mu_i, sigma)"]
        else:
            lines += [
                "", "Likelihood", "  y_i ~ Bernoulli(p_i), sampled as log p(y_i | eta_i) = y_i * eta_i - softplus(eta_i)",
                "  with a softplus whose value and gradient are exact for every finite eta on both the",
                "  PyTensor and the JAX path (autobayes/likelihoods.py; Bambi's built-in bernoulli family",
                "  is not used because its likelihood is floored at machine epsilon for |eta| > ~37).",
            ]
        lines += ["", "PyMC variables sampled (value space)"]
        n = self.names
        roles = {n["intercept"]: "a_c" if d.centered else "a",
                 f"{n['sd']}_log__": f"log(sd_{s.group})",
                 n["offset"]: "z"}
        if gaussian:
            roles[f"{n['sigma']}_log__"] = "log(sigma)"
        for v in self.pymc_model.value_vars:
            lines.append(f"  {v.name:<24} {roles.get(v.name, f'b_{v.name}')}")
        lines.append(f"Deterministic: {n['group_effect']} = u, {n['linpred']} = linear predictor"
                     + ("" if gaussian else "; p = expit(logit_p) is computed by predict()"))
        return "\n".join(lines)

    # -- convenience -------------------------------------------------------------------
    def fit(self, **kwargs):
        from autobayes.inference import fit

        return fit(self, **kwargs)

    def predict(self, fit_result, **kwargs):
        from autobayes.predict import predict

        return predict(self, fit_result, **kwargs)


def gaussian(formula: str, data: pd.DataFrame, priors: Priors, *, center_predictors: bool = True) -> Model:
    """Build the M0 Gaussian varying-intercept model (identity link, residual SD ``sigma``)."""
    return Model(formula, data, priors, family="gaussian", center_predictors=center_predictors)


def bernoulli(formula: str, data: pd.DataFrame, priors: Priors, *, center_predictors: bool = True) -> Model:
    """Build the M1 Bernoulli varying-intercept model (logit link, response strictly 0/1)."""
    return Model(formula, data, priors, family="bernoulli", center_predictors=center_predictors)
