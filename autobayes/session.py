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

    from autobayes.families import get_family

    n = model.names
    scale = get_family(model.family).scale_parameter            # "sigma", "shape", "phi" or None
    names = ([n["intercept"]] if model.spec.intercept else []) + list(model.spec.terms) + [roles["sd"] for roles in model.group_coefficient_names.values()] + ([n[scale]] if scale is not None else [])
    s = az.summary(idata, var_names=names, hdi_prob=0.94, round_to="none")
    rows = {}
    for name, r in s.iterrows():
        rows[name] = {k: float(r[k]) for k in ("mean", "sd", "hdi_3%", "hdi_97%", "mcse_mean",
                                                "ess_bulk", "ess_tail", "r_hat")}
    return rows


def fit_record(model, fit, *, panel: dict | None = None, max_treedepth: int = 10, allow_unverified: bool = False) -> dict:
    """A JSON-serialisable record of one fit: specification, execution, diagnostics, posterior."""
    from autobayes import __version__
    from autobayes.diagnostics import panel as _panel
    from autobayes.families import get_family

    from autobayes.formula import sd_tag
    from autobayes.identity import check_fit_binding
    binding = check_fit_binding(model, fit, allow_unverified=allow_unverified)

    p = panel if panel is not None else _panel(fit.idata, max_treedepth=max_treedepth)
    fam = get_family(model.family)
    s, d, pr = model.spec, model.design, model.priors
    coef = pr.coefficient_priors(s)
    group_priors = pr.group_priors(s)
    sd_priors = pr.sd_priors(s)
    return {
        "autobayes_version": __version__,
        "model": {
            "formula": s.formula,
            "family": model.family,
            **({"target_identity": model.target_identity} if binding == "matched" else
               {"requested_target_identity": model.target_identity}),
            "fit_binding_status": binding,
            **({"family_parameters": {name: getattr(model, name) for name in fam.fixed_parameters}} if fam.fixed_parameters else {}),
            "link": model.link,
            "n": int(d.n),
            **({"J": int(d.J), "group": s.group} if len(s.groups) <= 1 else {}),
            "groups": [{"factor": g, "levels": list(block.levels), "J": block.J,
                        "term": f"1|{g}", "names": model.group_names[g],
                        "coordinate": model.group_names[g]["coordinate"],
                        "sd_prior": str(group_priors[g]), "noncentered": True,
                        "slopes": [{"coef": c, "term": f"{c}|{g}", "names": model.group_coefficient_names[(g, c)],
                                    "sd_prior": str(sd_priors[(g, c)])} for c in s.group_term_of(g).slopes]}
                       for g, block in d.group_designs.items()],
            "intercept": s.intercept,
            "center_predictors": bool(d.centered),
            "center_predictors_requested": model.center_predictors,
            "noncentered": True if s.has_group else None,
            "priors": {**({"intercept": str(pr.intercept)} if s.intercept else {}), **{f"b:{k}": str(v) for k, v in coef.items()},
                       **{("sd" if len(s.groups) == 1 else f"sd:{g}"): str(prior) for g, prior in group_priors.items()},
                       **{f"sd:{sd_tag(g, c)}": str(prior) for (g, c), prior in sd_priors.items() if c != "Intercept"},
                       **({fam.scale_parameter: str(getattr(pr, fam.scale_parameter))} if fam.scale_parameter is not None else {})},
            "terms": list(s.terms),
            "design_columns": list(s.columns),
            **({"offset": s.offset.term} if s.offset is not None else {}),
            **({"transforms": [t.name for t in s.transform_terms]} if s.transform_terms else {}),
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
    same = all(a["model"].get(k) == b["model"].get(k) for k in ("formula", "priors", "family"))
    same = same and a["model"].get("family_parameters", {}) == b["model"].get("family_parameters", {})
    if not same:
        raise ValueError("records fit different models; agreement is only defined for the same family, family parameters, formula and priors")
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
