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

    n = model.names
    names = [n["intercept"], *model.spec.terms] + ([n["sd"]] if "sd" in n else []) + ([n["sigma"]] if "sigma" in n else [])
    s = az.summary(idata, var_names=names, hdi_prob=0.94, round_to="none")
    rows = {}
    for name, r in s.iterrows():
        rows[name] = {k: float(r[k]) for k in ("mean", "sd", "hdi_3%", "hdi_97%", "mcse_mean",
                                                "ess_bulk", "ess_tail", "r_hat")}
    return rows


def fit_record(model, fit, *, panel: dict | None = None, max_treedepth: int = 10) -> dict:
    """A JSON-serialisable record of one fit: specification, execution, diagnostics, posterior."""
    from autobayes.diagnostics import panel as _panel

    p = panel if panel is not None else _panel(fit.idata, max_treedepth=max_treedepth)
    s, d, pr = model.spec, model.design, model.priors
    coef = pr.coefficient_priors(s)
    return {
        "autobayes_version": _version("autobayes") or "0.5.0",
        "model": {
            "formula": s.formula,
            "family": model.family,
            "link": model.link,
            "n": int(d.n), "J": int(d.J), "group": s.group,                 # group None and J 0: population-only (M3c-1)
            "center_predictors": bool(d.centered),
            "noncentered": True if s.group is not None else None,
            "priors": {"intercept": str(pr.intercept), **{f"b:{k}": str(v) for k, v in coef.items()},
                       **({"sd": str(pr.sd)} if pr.sd is not None else {}),
                       **({"sigma": str(pr.sigma)} if pr.sigma is not None else {})},
            "terms": list(s.terms),
            "design_columns": list(s.columns),
            "interactions": [list(pair) for pair in s.interactions],
            "canonical_formula": s.canonical_formula,
            "categorical": {c.name: {"reference": c.reference, "levels": list(c.levels),
                                     "coefficients": list(c.coefficients), "unused_levels": list(c.unused)}
                            for c in s.categoricals},
            "intercept_prior_applies_to": ("centred intercept (at the numeric predictor means"
                                           + (" and the sample proportions of the categorical levels)" if s.categoricals else ")")
                                           if d.centered else "raw intercept")
                                          + (", logit scale" if model.family != "gaussian" else ""),
            **({"trials": s.trials, "trials_total": int(d.trials.sum()),
                "zero_trial_rows_dropped": len(model.dropped_rows),
                "zero_trial_row_positions": list(model.dropped_rows)} if model.family == "binomial" else {}),
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


def compare_records(a: dict, b: dict, *, k: float = 4.0) -> dict:
    """Population-parameter means within ``k`` combined MCSE. Returns per-parameter rows and ``agree``."""
    same = all(a["model"].get(k) == b["model"].get(k) for k in ("formula", "priors", "family"))
    if not same:
        raise ValueError("records fit different models; agreement is only defined for the same family, formula and priors")
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
    return {"k": k, "rows": rows, "agree": bool(agree),
            "a": a["execution"]["device_actual"], "b": b["execution"]["device_actual"]}


def format_comparison(c: dict) -> str:
    out = [f"Agreement of population-parameter means ({c['k']:g} combined MCSE): {c['a']} vs {c['b']}"]
    for name, r in c["rows"].items():
        out.append(f"  {name:<16} {r['mean_a']:.4f} vs {r['mean_b']:.4f}  |diff| {r['diff']:.4f}  tol {r['tol']:.4f}  {'ok' if r['agree'] else 'DISAGREE'}")
    out.append(f"  agree: {c['agree']}  (a statement about the target, not about speed)")
    return "\n".join(out)
