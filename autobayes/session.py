"""What a fresh session needs to pick a fit back up: environment, a fit record, agreement.

Three plain functions, no state:

- ``environment()``: versions of the stack, the JAX devices, float64 status. The first
  thing to print in a new session and the first thing to paste into a handoff.
- ``fit_record(model, fit)``: one JSON-serialisable dict that says what was fit, on
  what, with which seed, how it sampled, and the population-level posterior summary.
  This is the unit of evidence a status file or wiki page should carry; a number that
  is not in it was not recorded.
- ``compare_records(a, b)``: do two records target the same posterior? Means of the
  population parameters within ``k`` combined Monte Carlo standard errors. This is the
  CPU-vs-GPU (or backend-vs-backend) agreement check; it says nothing about speed.
"""
from __future__ import annotations

import platform
import sys
from importlib import metadata

import numpy as np


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def environment() -> dict:
    """Versions, JAX devices and float64 status.

    Calling this initialises the JAX backend. That is safe after ``import autobayes``
    (which sets the host device count first); it is only a problem if JAX was touched
    before autobayes was imported, and ``fit()`` reports that case in its notes.
    """
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {p: _version(p) for p in ("pymc", "pytensor", "bambi", "formulae",
                                               "arviz", "numpyro", "jax", "jaxlib", "numpy")},
        "jax_devices": None,
        "jax_x64": None,
    }
    try:
        import jax

        env["jax_devices"] = [f"{d.platform}:{getattr(d, 'device_kind', '?')}(id={d.id})"
                              for d in jax.devices()]
        env["jax_x64"] = bool(jax.config.jax_enable_x64)
    except Exception as e:  # jax absent or broken: say so, do not guess
        env["jax_error"] = f"{type(e).__name__}: {e}"
    return env


def format_environment(env: dict) -> str:
    v = env["versions"]
    lines = ["AutoBayes environment",
             f"  python    : {env['python']} ({env['platform']})",
             "  versions  : " + ", ".join(f"{k} {val or 'absent'}" for k, val in v.items()),
             f"  jax       : {', '.join(env['jax_devices']) if env['jax_devices'] else env.get('jax_error', 'absent')}",
             f"  jax x64   : {env['jax_x64']}"]
    return "\n".join(lines)


def _summary_rows(model, idata) -> dict:
    import arviz as az

    from autobayes.families import family_of

    n = model.names
    scale = family_of(model).scale_parameter                    # "sigma", "shape", "phi" or None
    if scale in model.spec.dpar_names:                          # its own formula (B01): its intercept and coefficients
        scale_names = [v for k, v in model.dpar_names[scale].items() if k != "value"]
    else:
        scale_names = [n[scale]] if scale is not None else []
    names = ([n["intercept"]] if model.spec.intercept else []) + list(model.spec.terms) + [roles["sd"] for roles in model.group_coefficient_names.values()] + scale_names
    names += [n[name] for name in model.learned_parameters]     # a learned Student-t nu (A13)
    names += [roles["simplex"] for roles in getattr(model, "mo_names", {}).values()]   # each monotonic term's simplex (B03)
    names += [roles[k] for roles in getattr(model, "me_names", {}).values() for k in ("mean", "sd")]   # an me() term's (C06)
    mixture = getattr(model, "mixture_names", {})               # a mixture's sigma<k> and theta (C05)
    names += [*mixture["sigma"], mixture["theta"]] if mixture else []
    names += [roles[k] for roles in getattr(model, "gp_names", {}).values() for k in ("sdgp", "lscale")]   # each GP's (C03)
    names += [v for k, v in getattr(model, "autocor_names", {}).items() if not k.endswith("_coordinate")]   # ar and ma (C04)
    names += list(getattr(model, "sar_names", {}).values())     # lagsar or errorsar (F02b)
    names += [v for k, v in getattr(model, "car_names", {}).items() if k in ("sdcar", "car", "rhocar")]   # a CAR term's (F02a)
    zero = family_of(model).zero_parameter                      # the zero-inflation / hurdle probability (B06)
    own = model.spec.dpar_names                                 # E03: zi, hu or alpha with a formula: its intercept and coefficients
    for name in ([zero] if zero is not None else []) + [e.name for e in family_of(model).extra_parameters]:   # alpha, beta, ndt (D01)
        names += [v for k, v in model.dpar_names[name].items() if k != "value"] if name in own else [n[name]]
    s = az.summary(idata, var_names=names, hdi_prob=0.94, round_to="none")
    rows = {}
    for name, r in s.iterrows():
        rows[name] = {k: float(r[k]) for k in ("mean", "sd", "hdi_3%", "hdi_97%", "mcse_mean",
                                                "ess_bulk", "ess_tail", "r_hat")}
    return rows


def _scale_priors(model, scale: str | None = None) -> dict:
    """The fit record's entries of the auxiliary parameter's prior: ``{"sigma": "HalfNormal(sigma=1)"}``, or with its own
    formula (B01) its intercept's and coefficients' priors, ``{"sigma:intercept": ..., "sigma:b:z[q]": ...}``; ``scale`` names
    another family parameter (E03: ``zi``, ``hu``, ``alpha``, with or without a formula of its own), by default the family's
    auxiliary parameter."""
    from autobayes.families import family_of

    scale = family_of(model).scale_parameter if scale is None else scale
    s, pr = model.spec, model.priors
    if scale not in s.dpar_names:
        return {scale: str(getattr(pr, scale))}
    ds, sub = s.dpar(scale), pr.dpar_priors(s, scale)
    return {**({f"{scale}:intercept": str(sub.intercept)} if ds.intercept else {}),
            **{f"{scale}:b:{col}": str(p) for col, p in sub.coefficient_priors(ds).items()}}


def fit_record(model, fit, *, panel: dict | None = None, max_treedepth: int = 10, allow_unverified: bool = False) -> dict:
    """A JSON-serialisable record of one fit: specification, execution, diagnostics, posterior."""
    from autobayes import __version__
    from autobayes.diagnostics import panel as _panel
    from autobayes.families import family_of

    from autobayes.families.mixture import ORDER, prior_labels
    from autobayes.formula import sd_tag
    from autobayes.identity import check_fit_binding
    if getattr(model, "responses", None):                       # C01: not part of the multivariate slice
        raise ValueError("fit_record() of a multivariate model is not part of this slice (docs/multivariate.md); record each "
                         "response's parameters from fit.idata by their names, model.response_names")
    binding = check_fit_binding(model, fit, allow_unverified=allow_unverified)

    p = panel if panel is not None else _panel(fit.idata, max_treedepth=max_treedepth)
    if getattr(model.spec, "is_nonlinear", False):             # a non-linear model (C02): its own model section and summary
        from autobayes.nonlinear import nonlinear_fit_record

        return nonlinear_fit_record(model, fit, p, binding)
    if getattr(model.spec, "is_multinomial", False):           # F01a: the count-matrix families' own model section and summary
        from autobayes.multinomial import multinomial_fit_record

        return multinomial_fit_record(model, fit, p, binding)
    fam = family_of(model)                                     # the family under the model's link
    s, d, pr = model.spec, model.design, model.priors
    mixture = fam.mixture                                      # C05: a mixture's priors are its components' and theta's
    coef = {} if mixture else pr.coefficient_priors(s)
    group_priors = pr.group_priors(s)
    sd_priors = pr.sd_priors(s)
    fixed = {name: getattr(model, name) for name in fam.fixed_parameters if getattr(model, name) is not None}   # not a learned nu (A13)
    cor_priors = pr.cor_priors(s)
    from autobayes.default_priors import BRMS_VERSION
    defaults = getattr(model, "default_prior_labels", [])
    return {
        "autobayes_version": __version__,
        "model": {
            "formula": s.formula,
            "family": model.family,
            **({"target_identity": model.target_identity} if binding == "matched" else
               {"requested_target_identity": model.target_identity}),
            "fit_binding_status": binding,
            **({"family_parameters": fixed} if fixed else {}),
            "link": model.link,
            "n": int(d.n),
            **({"J": int(d.J), "group": s.group} if len(s.groups) <= 1 else {}),
            "groups": [{"factor": g, "levels": list(block.levels), "J": block.J,
                        "term": f"1|{g}", "names": model.group_names[g],
                        "coordinate": model.group_names[g]["coordinate"],
                        "sd_prior": str(group_priors[g]), "noncentered": True,
                        "slopes": [{"coef": c, "term": f"{c}|{g}", "names": model.group_coefficient_names[(g, c)],
                                    "sd_prior": str(sd_priors[(g, c)])} for c in s.group_term_of(g).slopes],
                        # a correlated term (A02): its coefficients share a correlation matrix under an LKJ prior
                        **({"correlated": True, "cor_prior": str(cor_priors[g]), "correlation_names": model.correlation_names[g]}
                           if g in cor_priors else {})}
                       for g, block in d.group_designs.items()],
            "intercept": s.intercept,
            "center_predictors": bool(d.centered),
            "center_predictors_requested": model.center_predictors,
            "noncentered": True if s.has_group else None,
            "priors": {**({"intercept": str(pr.intercept)} if s.intercept and not mixture else {}), **{f"b:{k}": str(v) for k, v in coef.items()},
                       **{("sd" if len(s.groups) == 1 else f"sd:{g}"): str(prior) for g, prior in group_priors.items()},
                       **{f"sd:{sd_tag(g, c)}": str(prior) for (g, c), prior in sd_priors.items() if c != "Intercept"},
                       **{f"cor:{g}": str(prior) for g, prior in cor_priors.items()},
                       **{f"simo:{c}": str(prior) for c, prior in pr.simo_priors(s).items()},
                       **{f"sdgp:{c}": str(prior) for c, prior in pr.sdgp_priors(s).items()},       # C03
                       **{f"lscale:{c}": str(prior) for c, prior in pr.lscale_priors(s).items()},
                       **{f"{cls}:{x}": str(prior) for cls in ("meanme", "sdme") for x, prior in pr.me_priors(s, cls).items()},   # C06
                       **(_scale_priors(model) if fam.scale_parameter is not None else {}),
                       **(_scale_priors(model, fam.zero_parameter) if fam.zero_parameter is not None else {}),   # E03: zi:intercept
                       **{k: v for e in fam.extra_parameters for k, v in _scale_priors(model, e.name).items()},   # alpha, beta, ndt (D01)
                       **({"nu": str(pr.nu)} if pr.nu is not None else {}),
                       **({label: str(prior) for label, prior in prior_labels(model).items()} if mixture else {}),
                       **{part: str(prior) for part, prior in pr.autocor_priors(s).items()},   # an ARMA term's ar and ma (C04)
                       **{name: str(prior) for name, prior in pr.sar_priors(s).items()},       # a SAR term's lagsar / errorsar (F02b)
                       **{cls: str(prior) for cls, prior in pr.car_priors(s).items()}},   # a CAR term's sdcar, car, rhocar (F02a)
            # a mixture (C05): its component families, identification and PyMC names
            **({"mixture": {"families": list(model.components), "order": ORDER,
                            "names": {k: list(v) if isinstance(v, tuple) else v for k, v in model.mixture_names.items()}}}
               if mixture else {}),
            # which of those priors ab.default_priors() filled in (brms's defaults, A07); absent when the user gave them all
            **({"default_priors": {"brms": BRMS_VERSION, "parameters": list(defaults)}} if defaults else {}),
            "terms": list(s.terms),
            "design_columns": list(s.columns),
            **({"offset": s.offset.term} if s.offset is not None else {}),
            **({"addition_terms": [a.term for a in s.additions]} if s.additions else {}),   # weights() (B02)
            # each distributional parameter's own formula (B01): as written, canonical, link, terms, columns and PyMC names
            **({"distributional": {ds.response: {"formula": ds.formula, "canonical_formula": ds.canonical_formula,
                                                 "link": d.dpars[ds.response].link, "intercept": ds.intercept,
                                                 "centered": d.dpars[ds.response].centered, "terms": list(ds.terms),
                                                 "design_columns": list(ds.columns), "names": model.dpar_names[ds.response]}
                                   for ds in s.dpars}} if s.dpars else {}),
            **({"transforms": [t.name for t in s.transform_terms]} if s.transform_terms else {}),
            # each monotonic term (B03): its column's coding, number of steps and PyMC names
            **({"monotonic": {t.name: {"column": t.operands[0], "coding": "ordered" if s.monotonic(t.operands[0]).ordered else "integer",
                                       "values": [s.monotonic(t.operands[0]).lowest, s.monotonic(t.operands[0]).highest],
                                       "D": s.monotonic(t.operands[0]).D, "names": model.mo_names[t.operands[0]]}
                               for t in s.mo_terms}} if s.mo_terms else {}),
            # each Gaussian process (C03): its column, exact or approximate, its scaling and PyMC names
            **({"gaussian_processes": {g.term: {"column": g.column, "approximate": g.approximate, "k": g.k, "c": g.c,
                                                "distinct_inputs": len(g.inputs), "dmax": g.dmax, "cmean": g.cmean,
                                                "names": model.gp_names[g.column]}
                                       for g in s.gp_terms}} if getattr(s, "gp_terms", ()) else {}),
            # an autocorrelation term (C04): the term, its series and the PyMC names of ar and ma
            **({"autocorrelation": {"term": s.autocor.term, "time": s.autocor.time, "gr": s.autocor.gr, "p": s.autocor.p,
                                    "q": s.autocor.q, "series": len(d.autocor.series), "names": model.autocor_names}}
               if getattr(s, "autocor", None) is not None else {}),
            # a SAR term (F02b): the term, its matrix's name and size, the parameter's bounds and its PyMC name
            **({"sar": {"term": s.sar.term, "type": s.sar.type, "matrix": s.sar.matrix, "n": d.sar.n,
                        "bounds": [d.sar.lower, d.sar.upper], "names": model.sar_names}}
               if getattr(s, "sar", None) is not None else {}),
            # a fixed residual covariance (F03a): the term, the data2 name of its matrix and the matrix's size
            **({"fcor": {"term": s.fcor.term, "matrix": s.fcor.matrix, "rows": d.fcor.n}} if getattr(s, "fcor", None) is not None else {}),
            # a CAR term (F02a): the term, its locations and edges, and its PyMC names
            **({"car": {"term": s.car.term, "type": s.car.type, "gr": s.car.gr, "matrix": s.car.matrix, "locations": d.car.Nloc,
                        "edges": len(d.car.edges), "names": model.car_names}} if getattr(s, "car", None) is not None else {}),
            # each measurement-error term (C06): its columns and PyMC names
            **({"measurement_error": {t.name: {"column": t.operands[0], "noise_column": t.operands[1], "names": model.me_names[t.operands[0]]}
                                      for t in s.me_terms}} if s.me_terms else {}),
            "interactions": [list(pair) for pair in s.interactions],
            "canonical_formula": s.canonical_formula,
            "categorical": {c.name: {"reference": c.reference, "levels": list(c.levels),
                                     "coefficients": list(c.coefficients), "unused_levels": list(c.unused), "coding": "full_level" if c.full_rank else "treatment"}
                            for c in s.categoricals},
            "intercept_prior_applies_to": None if not s.intercept else ("centred intercept (at the numeric predictor means"
                                           + (" and the sample proportions of the categorical levels)" if s.categoricals else ")")
                                           if d.centered else "raw intercept")
                                          + (f", {fam.link_scale} scale" if fam.link_scale else ""),
            **({"trials": s.trials, "trials_total": int(d.trials.sum()),
                "zero_trial_rows_dropped": len(model.dropped_rows),
                "zero_trial_row_positions": list(model.dropped_rows)} if fam.has_trials else {}),
            "warnings": list(model.warnings),
        },
        "execution": {
            "backend": fit.backend, "device_requested": fit.device_requested,
            "device_actual": fit.device_actual, "platform": fit.platform, "dtype": fit.dtype,
            "chain_method": fit.chain_method, "chains": fit.chains, "draws": fit.draws,
            "tune": fit.tune, "target_accept": fit.target_accept, "seed": fit.seed,
            "wall_seconds": round(float(fit.wall_seconds), 3), "notes": list(fit.notes),
        },
        "environment": environment(),
        "diagnostics": {"values": {k: (float(v) if v is not None else None) for k, v in p["values"].items()},
                        "status": dict(p["status"]), "overall": p["overall"]},
        "posterior": _summary_rows(model, fit.idata),
    }


def compare_records(a: dict, b: dict, *, k: float = 4.0, allow_legacy: bool = False) -> dict:
    """Population-parameter means within ``k`` combined MCSE. Returns per-parameter rows and ``agree``."""
    from autobayes.identity import comparison_identity

    identity_status = comparison_identity(a, b, allow_legacy=allow_legacy)
    same = all(a["model"].get(k) == b["model"].get(k) for k in ("formula", "priors", "family", "distributional", "mixture"))
    same = same and a["model"].get("family_parameters", {}) == b["model"].get("family_parameters", {})
    if not same:
        raise ValueError("records fit different models; agreement is only defined for the same family, family parameters, formula "
                         "(distributional formulas included) and priors")
    if set(a["posterior"]) != set(b["posterior"]):
        raise ValueError("records must contain identical posterior parameter sets")
    rows, agree = {}, True
    for name in a["posterior"]:
        if name not in b["posterior"]:
            raise ValueError(f"parameter {name!r} missing from second record")
        ra, rb = a["posterior"][name], b["posterior"][name]
        diff = abs(ra["mean"] - rb["mean"])
        tol = k * float(np.hypot(ra["mcse_mean"], rb["mcse_mean"]))
        ok = bool(diff <= tol)
        agree &= ok
        rows[name] = {"mean_a": ra["mean"], "mean_b": rb["mean"], "diff": diff, "tol": tol, "agree": ok}
    return {"k": k, "rows": rows, "agree": bool(agree), "identity_status": identity_status,
            "a": a["execution"]["device_actual"], "b": b["execution"]["device_actual"]}


def format_comparison(c: dict) -> str:
    out = [f"Agreement of population-parameter means ({c['k']:g} combined MCSE): {c['a']} vs {c['b']}"]
    if c.get("identity_status") != "matched":
        out.append("WARNING: target identity is unverified; agreement does not establish a common target.")
    for name, r in c["rows"].items():
        out.append(f"  {name:<16} {r['mean_a']:.4f} vs {r['mean_b']:.4f}  |diff| {r['diff']:.4f}  tol {r['tol']:.4f}  {'ok' if r['agree'] else 'DISAGREE'}")
    out.append(f"  agree: {c['agree']}  (a statement about the target, not about speed)")
    return "\n".join(out)
