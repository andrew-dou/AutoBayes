"""brms's ``gp()`` beyond one input (G09, docs/gaussian-processes.md, "Several inputs, by, other kernels and scale = FALSE"):
what ``describe()``, predictions, the target identity and fit records say about a Gaussian process with several inputs
(``gp(o2, b2)``, ``iso = FALSE``), a ``by`` variable (a factor: one process per level; numeric: the process times the column),
the Matern 3/2, Matern 5/2 or exponential kernel (exact, or approximate with ``k``), or ``scale = FALSE``. C03's form, one input
with the squared-exponential kernel, keeps its own text in ``model.py``; the mathematics is ``autobayes.gaussian_process``'s
and the term ``formula.GaussianProcess``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KERNEL_NAMES = {"exp_quad": "squared-exponential", "matern32": "Matern 3/2", "matern52": "Matern 5/2",
                "exponential": "exponential"}
_KERNEL_OF_R = {"exp_quad": "exp(-r^2 / 2)", "matern32": "(1 + sqrt(3) r) exp(-sqrt(3) r)",
                "matern52": "(1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)", "exponential": "exp(-r)"}
_SPECTRAL = {"exp_quad": "sqrt(2 pi) {l} exp(-{l}^2 w^2 / 2)", "matern32": "12 sqrt(3) {l} (3 + {l}^2 w^2)^-2",
             "matern52": "(400 sqrt(5) / 3) {l} (5 + {l}^2 w^2)^-3", "exponential": "2 {l} (1 + {l}^2 w^2)^-1"}


def _inputs_text(g) -> str:
    return ", ".join(repr(c) for c in g.columns)


def _scaling(p) -> str:
    """How the process's inputs are scaled, for one process (a level's, for a factor by)."""
    if not p.scale:
        return "unscaled (brms's scale = FALSE: lscale is on the inputs' own scale)"
    return f"scaled by dmax = {p.dmax:g}, their largest {'Euclidean ' if p.ndim > 1 else ''}distance, so that it is 1 (brms's scale = TRUE)"


def _spread(p) -> str:
    x = np.asarray(p.inputs, dtype=float)
    if p.ndim == 1:
        return f"{len(p.inputs)} distinct inputs from {x.min():g} to {x.max():g}"
    return f"{len(p.inputs)} distinct input rows"


def gp_data_lines(g) -> list[str]:
    """``describe()``'s data lines of the Gaussian process ``g``: its inputs, kernel, by variable and scaling."""
    kind = "approximate (Hilbert-space) Gaussian process" if g.approximate else "exact Gaussian process"
    kernel = f"{KERNEL_NAMES[g.cov]} kernel" + (" of the Euclidean distance" if g.ndim > 1 and g.iso else "")
    kernel += "" if g.iso else ", one length-scale per input (iso = FALSE: the product of one-dimensional kernels)"
    head = f"  {g.term}: {kind} of {_inputs_text(g)}, {kernel}"
    if g.is_by_factor:
        out = [head + f"; one process per level of the factor {g.by!r}, on that level's rows (brms's by = {g.by}, cmc = TRUE)"]
        return out + [f"    level {p.level!r}: {_spread(p)}, {_scaling(p)}" for p in g.parts]
    by = f"; multiplied row by row by {g.by!r} (brms's by = {g.by}, Cgp)" if g.by is not None else ""
    basis = (f"; k = {g.k} basis functions on [-L, L], L = c = {g.L:g}, of the scaled input less the mean of its distinct values, "
             f"{g.cmean:.6g}" if g.approximate else "")
    return [head + by + f"; {_spread(g)}, {_scaling(g)}{basis}"]


def _brms_coef(key: str) -> str:
    from autobayes.default_priors import brms_rename

    return brms_rename(f"gp{key}")


def gp_prior_lines(model, g, sdgp, lscale) -> list[str]:
    """``describe()``'s prior lines of ``g``: each process's ``sdgp``, each length-scale and ``zgp``."""
    from autobayes.priors import describe_prior

    out = []
    for p in g.subprocesses:
        roles = model.gp_names[p.key]
        out.append(f"  {roles['sdgp']} ~ {describe_prior(sdgp[p.key], lb=0)}   (brms class sdgp, coefficient {_brms_coef(p.key)})")
        on = "on the scaled input" if p.scale else "on the unscaled input"
        for lk, name in model.gp_lscale_names(p.key).items():
            out.append(f"  {name} ~ {describe_prior(lscale[lk], lb=0)}   (brms class lscale, coefficient {_brms_coef(lk)}; {on})")
        out.append(f"  {roles['zgp']}[j] ~ Normal(0, 1)   (brms's zgp)")
    return out


def gp_variable_roles(model, g) -> dict[str, str]:
    """``describe()``'s sampled variables of ``g``: each process's log SD, log length-scales and standardised values."""
    out = {}
    for p in g.subprocesses:
        roles = model.gp_names[p.key]
        of = g.term + ("" if p.level is None else f" at level {p.level!r} of {g.by}")
        out[f"{roles['sdgp']}_log__"] = f"log({roles['sdgp']}), the marginal SD of {of} (brms's sdgp)"
        for lk, name in model.gp_lscale_names(p.key).items():
            which = "" if p.iso else f" in {lk[len(p.key):]!r}"
            out[f"{name}_log__"] = f"log({name}), the length-scale{which} of {of} (brms's lscale)"
        out[roles["zgp"]] = (f"{roles['zgp']}[1..{p.size}], the standardised values of {of}, one per "
                             f"{'basis function' if p.approximate else 'distinct input'} (brms's zgp); dims {roles['coordinate']}")
    return out


def gp_paragraph(model, g, lp: str) -> list[str]:
    """``describe()``'s paragraph of ``g``: the process written out, what ``sdgp`` and ``lscale`` mean, and what new rows take."""
    out = []
    for p in g.subprocesses:
        roles = model.gp_names[p.key]
        sd, z = roles["sdgp"], roles["zgp"]
        ls = list(model.gp_lscale_names(p.key).values())
        row = " row" if p.ndim > 1 else ""
        u = f"u_i, the scaled input{row}" if p.scale else f"u_i, the input{row} as it is (scale = FALSE)"
        where = f" on the rows where {g.by} = {p.level!r}" if p.level is not None else ""
        if p.approximate:
            body = (f"sum_j phi_j(u_i) sqrt(S(w_j)) {z}[j], j = 1..{p.k}, phi_j(u) = sin(pi j (u - {p.cmean:.6g} + L) / (2 L)) / sqrt(L) "
                    f"on [-L, L], L = {p.L:g}, w_j = pi j / (2 L), S(w) = {sd}^2 {_SPECTRAL[p.cov].format(l=ls[0])} the "
                    f"{KERNEL_NAMES[p.cov]} kernel's spectral density, {z}[j] ~ Normal(0, 1): brms's approximate Gaussian process")
        else:
            if p.iso:
                r = f"r = |u_j - u_m| / {ls[0]}" + (" (Euclidean)" if p.ndim > 1 else "")
                K = f"K[j, m] = {sd}^2 k(r), {r}, k(r) = {_KERNEL_OF_R[p.cov]}"
            else:
                factors = " * ".join(f"exp(-(u_j[{c}] - u_m[{c}])^2 / (2 {name}^2))" for c, name in zip(p.columns, ls))
                K = f"K[j, m] = {sd}^2 {factors}"
            body = (f"f(u_i) at the {len(p.inputs)} distinct inputs in the order they first appear (brms's), f = chol(K + 1e-12 I) "
                    f"{z}, {K}, {z}[j] ~ Normal(0, 1): brms's exact Gaussian process, non-centred")
        times = f" times {g.by}_i" if g.by is not None and not g.is_by_factor else ""
        out.append(f"  {g.term}_i{where} = {body}{times}; {u}.")
        apart = ("" if not p.iso else f": inputs {ls[0]}{f' * {p.dmax:g}' if p.scale else ''} apart are correlated "
                 f"{'exp(-1/2) = 0.61' if p.cov == 'exp_quad' else 'as the kernel at r = 1'}")
        out.append(f"  {sd} is the process's marginal SD (on the scale of {lp}); {', '.join(ls)} its length-scale"
                   f"{'s, one per input' if not p.iso else ''}{apart}. It is not centred: the intercept is not its mean over the rows.")
    if g.approximate:
        lo, hi = (g.cmean - g.L) * g.dmax, (g.cmean + g.L) * g.dmax
        out.append(f"  New rows: the basis at their inputs, which must lie inside the boundary, {g.column} in ({lo:.6g}, {hi:.6g}); "
                   "beyond it the approximation does not hold and the input is refused (divergence D66).")
    else:
        level = "; a row of a level the fit did not see is refused" if g.is_by_factor else ""
        out.append(f"  New rows take the process's value at a fitted input{' of their level' if g.is_by_factor else ''}{level}; any "
                   "other input is refused (brms draws one from the conditional normal, divergence D65).")
    return out


def gp_values(model, g, post, df: pd.DataFrame) -> np.ndarray:
    """(chain, draw, row) draws of the Gaussian process ``g``'s part of the linear predictor on the rows of ``df`` (the fitted
    frame, or checked new rows): each process's values from its draws, at each row's input (a fitted input of an exact GP,
    D65; inside an approximate GP's boundary, D66), on its level's rows for a factor by, times the column for a numeric by."""
    from autobayes.gaussian_process import approximate_values, exact_values

    rows_by = g.level_rows(df, where="newdata column") if g.is_by_factor else {}
    out = 0.0
    for p in g.subprocesses:
        roles = model.gp_names[p.key]
        sdgp = post[roles["sdgp"]].to_numpy()
        ls = [post[name].to_numpy() for name in model.gp_lscale_names(p.key).values()]
        lscale = ls[0] if p.iso else ls
        z = post[roles["zgp"]].transpose("chain", "draw", roles["coordinate"]).to_numpy()
        out = out + np.zeros(sdgp.shape + (len(df),))
        rows = rows_by.get(p.level)
        frame = df if rows is None else df.iloc[rows]
        if p.approximate:
            f = approximate_values(p.basis(frame, where="newdata column"), p.slambda, sdgp, lscale, z, p.cov)
        else:
            f = exact_values(p.scaled_inputs, sdgp, lscale, z, p.cov)[..., p.index(frame, where="newdata column")]
        if rows is not None:
            out[..., rows] += f
        elif g.by is not None:
            out = out + g.multiplier(df, where="newdata column") * f
        else:
            out = out + f
    return out


def gp_identity(model, g, sdgp, lscale, prior) -> dict:
    """The target identity's entry of ``g``: the term and, per process, its inputs, scaling, row index (and rows or by
    values), an approximate process's basis and frequencies, and its priors."""
    parts = []
    for p in g.subprocesses:
        block = model.design.gp[p.key]
        parts.append({'key': p.key, 'level': p.level, 'inputs': [list(r) if isinstance(r, tuple) else r for r in p.inputs],
                      'dmax': p.dmax, 'cmean': p.cmean, 'index': block.index,
                      **({'rows': block.rows} if block.rows is not None else {}),
                      **({'multiplier': block.multiplier} if block.multiplier is not None else {}),
                      **({'basis': block.basis, 'slambda': block.slambda} if p.approximate else {}),
                      'priors': {'sdgp': prior(sdgp[p.key]),
                                 'lscale': {lk: prior(lscale[lk]) for lk in p.lscale_keys}}})
    return {'term': g.term, 'columns': list(g.columns), 'by': g.by, 'cov': g.cov, 'iso': g.iso, 'scale': g.scale, 'k': g.k,
            'c': g.c, 'processes': parts}


def gp_record(model, g) -> dict:
    """The fit record's entry of ``g``: its inputs, kind, kernel, by variable, scaling and each process's PyMC names."""
    return {"columns": list(g.columns), "approximate": g.approximate, "k": g.k, "c": g.c, "cov": g.cov, "iso": g.iso,
            "scale": g.scale, "by": g.by,
            "processes": {p.key: {"level": p.level, "distinct_inputs": len(p.inputs), "dmax": p.dmax, "cmean": p.cmean,
                                  "names": model.gp_names[p.key]} for p in g.subprocesses}}
