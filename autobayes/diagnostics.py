"""The diagnostic panel from ``reference/diagnostics.md``, with its thresholds."""
from __future__ import annotations

import numpy as np
import arviz as az

THRESHOLDS = {
    # name: (pass predicate, investigate predicate) on the value
    "divergence_fraction": (lambda v: v == 0, lambda v: v < 1e-3),
    "max_rhat": (lambda v: v <= 1.01, lambda v: v <= 1.05),
    "min_ess_bulk": (lambda v: v > 400, lambda v: v >= 100),
    "min_ess_tail": (lambda v: v > 400, lambda v: v >= 100),
    "min_bfmi": (lambda v: v > 0.3, lambda v: v >= 0.2),
    "treedepth_saturation": (lambda v: v < 0.01, lambda v: v <= 0.20),
}


def _status(name: str, value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    ok, investigate = THRESHOLDS[name]
    return "pass" if ok(value) else ("investigate" if investigate(value) else "fail")


def panel(idata, *, max_treedepth: int = 10, var_names=None) -> dict:
    """Compute the panel. Returns ``{"values": {...}, "status": {...}, "overall": str}``."""
    post = idata.posterior if var_names is None else idata.posterior[var_names]
    ss = idata.sample_stats
    n_draws = int(ss.sizes["chain"] * ss.sizes["draw"])
    div = float(ss["diverging"].sum()) if "diverging" in ss else np.nan
    rhat = az.rhat(post)
    ess_b = az.ess(post, method="bulk")
    ess_t = az.ess(post, method="tail")
    vals = {
        "divergences": div,
        "divergence_fraction": div / n_draws if np.isfinite(div) else np.nan,
        "max_rhat": float(max(np.nanmax(np.asarray(v)) for v in rhat.data_vars.values())),
        "min_ess_bulk": float(min(np.nanmin(np.asarray(v)) for v in ess_b.data_vars.values())),
        "min_ess_tail": float(min(np.nanmin(np.asarray(v)) for v in ess_t.data_vars.values())),
        "min_bfmi": float(np.min(az.bfmi(idata))) if "energy" in ss else np.nan,
        "treedepth_saturation": float((ss["tree_depth"] >= max_treedepth).mean()) if "tree_depth" in ss else np.nan,
        "n_draws": n_draws,
    }
    status = {k: _status(k, vals[k]) for k in THRESHOLDS}
    order = {"pass": 0, "n/a": 0, "investigate": 1, "fail": 2}
    overall = max(status.values(), key=lambda s: order[s])
    return {"values": vals, "status": status, "overall": overall}


def format_panel(p: dict) -> str:
    v, s = p["values"], p["status"]
    rows = [
        ("divergences", f"{v['divergences']:.0f} of {v['n_draws']} ({100 * v['divergence_fraction']:.3f}%)", s["divergence_fraction"]),
        ("max R-hat", f"{v['max_rhat']:.4f}", s["max_rhat"]),
        ("min bulk ESS", f"{v['min_ess_bulk']:.0f}", s["min_ess_bulk"]),
        ("min tail ESS", f"{v['min_ess_tail']:.0f}", s["min_ess_tail"]),
        ("min E-BFMI", f"{v['min_bfmi']:.3f}", s["min_bfmi"]),
        ("tree-depth saturation", f"{100 * v['treedepth_saturation']:.2f}%", s["treedepth_saturation"]),
    ]
    out = ["Diagnostic panel (thresholds from reference/diagnostics.md)"]
    out += [f"  {name:<22} {val:<28} {st}" for name, val, st in rows]
    out.append(f"  overall: {p['overall']}")
    return "\n".join(out)
