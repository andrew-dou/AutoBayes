"""Draws from a model's priors as brms's ``sample_prior = "yes"`` records them (brms-parity E05, docs/post-processing.md):
``ab.prior_draws(model)`` is brms's ``prior_draws()`` of such a fit, one column per prior brms samples, named as brms names
it, for the Savage-Dickey evidence ratios of ``ab.hypothesis(..., prior=...)``.

Names and layout (settled on brms 2.23's outputs, ``tests/hypothesis_probes/brms-2.23.0-prior-draws.json``):

- ``Intercept``: the prior of the intercept of the centred design (brms's class ``Intercept``; the raw-scale intercept has
  no prior draws, so ``hypothesis()`` gives a point hypothesis on ``Intercept`` no evidence ratio);
- ``b``: one draw per row shared by every coefficient when ``b`` is one prior (brms's class-wide prior); one column per
  coefficient, ``b_x``, ``b_zq``, ``b_x:zq``, when ``b`` is a mapping by term (brms gives every coefficient its own column as
  soon as one has a prior of its own);
- the family's parameters with a prior of their own, in brms's order: ``sigma``, ``shape`` or ``phi``, then ``nu``, then
  ``zi`` or ``hu``;
- for each grouping factor in formula order: ``sd_g`` when its coefficients' SDs share one prior (an intercept-only factor,
  or a mapping whose priors are all equal: brms's group-wide prior), else ``sd_g__Intercept``, ``sd_g__x``, ...; then
  ``cor_g`` for a correlated term, one correlation, distributed as one off-diagonal entry of an LKJ(eta) correlation
  matrix of the term's dimension d: ``2 B - 1`` with ``B ~ Beta(eta - 1 + d / 2, eta - 1 + d / 2)`` (Lewandowski, Kurowicka
  and Joe 2009);
- a flat prior (``ab.Flat()``) is improper and is not sampled, as brms does not sample it: its column is left out.

The draws are the model's own prior distributions (its PyMC variables, so a truncated ``nu`` and half-normal SDs are drawn
as the model's density has them), independent across columns and rows. They are random: brms's are drawn by Stan's RNG
and cannot be reproduced; what is compared with brms is computed from given draws (the evidence ratios) and the
distributions (tests/test_prior_draws.py). Not offered, each refused naming the construct (divergence D124): multivariate,
non-linear, mixture, ordinal and categorical models, distributional formulas, shrinkage priors, ``mo()``, ``gp()``,
``me()``, ``mi()``, ``cs()``, autocorrelation terms, ``fcor()``, ``mm()`` and ``gr(by = )`` group terms, the D01 families' extra
parameters (``alpha``, ``beta``, ``ndt``) and ``center_predictors=False``.
"""
from __future__ import annotations

import json
import re
from numbers import Integral

import numpy as np
import pandas as pd

FAMILY_ORDER = ("sigma", "shape", "phi", "nu", "zi", "hu")      # brms's order of a family's parameters (its dpars)


def _refuse_unsupported(model) -> None:
    def refuse(construct: str):
        raise ValueError(f"prior_draws() of a model with {construct} is not offered: no probe of brms's sample_prior verifies "
                         "its prior draws (docs/post-processing.md, divergence D124)")

    if getattr(model, "responses", None):
        refuse("several responses (a multivariate model)")
    spec = getattr(model, "spec", None)
    if getattr(spec, "is_nonlinear", False):
        refuse("a non-linear formula (nl=True)")
    fam = model.family_spec
    if fam.mixture:
        refuse("a mixture")
    if fam.thresholds or fam.per_category:
        refuse(f"the {model.family} family's {'thresholds' if fam.thresholds else 'categories'}")
    if fam.extra_parameters:
        refuse(f"the {model.family} family's {', '.join(e.name for e in fam.extra_parameters)}")
    if not model.center_predictors and spec.intercept:
        refuse("center_predictors=False (brms's center = FALSE makes the intercept a coefficient of class b)")
    if getattr(model, "shrinkage_names", None):
        refuse(f"a {model.priors.b.name}() prior on b")
    constructs = [
        (bool(spec.dpars), lambda: f"a distributional formula for {', '.join(spec.dpar_names)}"),
        (bool(spec.mo_terms), lambda: f"the monotonic term {spec.mo_terms[0].name!r}"),
        (bool(spec.gp_terms), lambda: f"the Gaussian process {spec.gp_terms[0].term!r}"),
        (bool(getattr(spec, "me_terms", ())), lambda: f"the measurement-error term {spec.me_terms[0].name!r}"),
        (bool(getattr(spec, "mi_terms", ())) or bool(getattr(spec, "missing_columns", ())), lambda: "missing values modelled by mi()"),
        (bool(getattr(spec, "cs_terms", ())), lambda: f"the category-specific term {spec.cs_terms[0].name!r}"),
        (spec.autocor is not None, lambda: f"the autocorrelation term {spec.autocor.term!r}"),
        (getattr(spec, "sar", None) is not None, lambda: f"the SAR term {spec.sar.term!r}"),                     # F02b (D143)
        (getattr(spec, "fcor", None) is not None, lambda: f"the fixed residual covariance {spec.fcor.term!r}"),   # F03a
        (getattr(spec, "car", None) is not None, lambda: f"the CAR term {spec.car.term!r}"),     # F02a
        (getattr(spec, "timecor", None) is not None, lambda: f"the residual correlation term {spec.timecor.term!r}"),   # F03b
    ]
    for present, construct in constructs:
        if present:
            refuse(construct())
    for t in spec.group_terms:
        if t.members or t.by is not None or t.id is not None:
            refuse(f"the group term {t.canonical!r} (mm(), gr(by = ) or a shared ID)")


def _brms_group_coef(coef: str) -> str:
    """brms's name of a group-level coefficient: ``Intercept``, ``x``, and ``zq`` for a level column ``z[q]``."""
    return re.sub(r"\[([^\]]*)\]", r"\1", coef)


def _columns(model):
    """``[(brms column, PyMC variable name, element index or None), ...]`` for the priors brms samples, in brms's order,
    plus the correlated terms ``[(brms column, eta, dimension), ...]``."""
    from autobayes.default_priors import brms_coefficient_names
    from autobayes.priors import Prior

    spec, priors, names = model.spec, model.priors, model.names
    out = []
    if spec.intercept and not (isinstance(priors.intercept, Prior) and priors.intercept.name == "Flat"):
        out.append(("Intercept", names["intercept"], None))
    coefficient_priors = priors.coefficient_priors(spec) if model.coefficients else {}
    proper = [(column, variable, j) for column, variable, j in model.coefficients if coefficient_priors[column].name != "Flat"]
    if isinstance(priors.b, Prior):
        if proper:
            out.append(("b", proper[0][1], proper[0][2]))
    else:
        brms = brms_coefficient_names(spec)
        out += [(f"b_{brms[column]}", variable, j) for column, variable, j in proper]
    out += [(par, names[par], None) for par in FAMILY_ORDER if par in names]
    sd_priors = priors.sd_priors(spec)
    correlated = []
    for t in spec.group_terms:
        coefs = list(t.coefs)
        sds = [model.group_coefficient_names[(t.factor, c)]["sd"] for c in coefs]
        shared = len({repr(sd_priors[(t.factor, c)]) for c in coefs}) == 1
        if shared:
            out.append((f"sd_{t.factor}", sds[0], None))
        else:
            out += [(f"sd_{t.factor}__{_brms_group_coef(c)}", sd, None) for c, sd in zip(coefs, sds)]
        if t.factor in model.correlation_names and len(coefs) > 1:
            cor = (priors.cor if isinstance(priors.cor, Prior) else priors.cor[t.factor])
            out.append((f"cor_{t.factor}", None, None))
            correlated.append((f"cor_{t.factor}", float(cor.params["eta"]), len(coefs)))
    return out, correlated


def prior_draws(model, *, draws: int = 4000, seed: int | None = None) -> pd.DataFrame:
    """brms's ``prior_draws()`` of a fit with ``sample_prior = "yes"`` (module docstring): ``draws`` independent draws from
    the model's priors (4000, a brms fit's default number of draws), one column per prior brms samples, named as brms names
    it. ``seed`` (a nonnegative integer) makes them reproducible. Returns a ``pandas.DataFrame``, one row per draw; its
    ``attrs`` record the model's target identity, the seed and the number of draws."""
    import pymc as pm

    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, Integral) or draws < 2:
        raise ValueError(f"draws must be a whole number of at least 2 (a density at a point needs two), got {draws!r}")
    if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0):
        raise ValueError(f"seed must be a nonnegative integer or None, got {seed!r}")
    _refuse_unsupported(model)
    columns, correlated = _columns(model)
    rng = np.random.default_rng(seed)
    variables = list(dict.fromkeys(v for _, v, _ in columns if v is not None))
    values = {}
    if variables:
        drawn = pm.draw([model.pymc_model[v] for v in variables], draws=int(draws), random_seed=rng)
        values = dict(zip(variables, drawn if len(variables) > 1 else [drawn]))
    eta_dim = {name: (eta, d) for name, eta, d in correlated}
    out = {}
    for name, variable, j in columns:
        if variable is None:                                     # a correlation: one off-diagonal entry of an LKJ matrix
            eta, d = eta_dim[name]
            a = eta - 1.0 + d / 2.0
            out[name] = 2.0 * rng.beta(a, a, size=int(draws)) - 1.0
            continue
        v = np.asarray(values[variable], dtype=float).reshape(int(draws), -1)
        out[name] = v[:, 0 if j is None else j]
    frame = pd.DataFrame(out)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("prior draws contain non-finite values; inspect the priors")
    frame.attrs.update(autobayes_generation_kind="prior_draws",
                       autobayes_built_target_identity_json=json.dumps(model.target_identity, sort_keys=True),
                       autobayes_seed_json=json.dumps(None if seed is None else int(seed)), autobayes_requested_draws=int(draws))
    return frame
