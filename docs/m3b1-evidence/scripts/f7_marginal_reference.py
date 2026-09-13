"""F7 marginal reference (docs/m3b1-decision.md, "Marginal-reference protocol"): deterministic quadrature of the
analytically integrated Gaussian varying-slope posterior of (sd0, sd1, sigma) on the frozen seed-17 data.
Usage: python f7_marginal_reference.py OUT.json OUT.txt   (from the repository root; counts evaluations and time)"""
import json, sys, time
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy.special import logsumexp

sys.path.insert(0, ".")
from tests.reference_marginal import GaussianSlopeMarginal, f7_data

CAP_EVALS, CAP_SECONDS = 2_000_000, 15 * 60
NAMES = ["1|g_sigma", "x|g_sigma", "sigma"]
out_json, out_txt = Path(sys.argv[1]), Path(sys.argv[2])
y, x, g = f7_data()
ref = GaussianSlopeMarginal(y, x, g, prior_mean=(0.0, 0.0), prior_sd=(5.0, 2.0), center=True)
counter = {"evals": 0}; t_start = time.perf_counter()


def ell(theta):
    theta = np.atleast_2d(theta); out = np.empty(len(theta))
    for i in range(0, len(theta), 32768):
        out[i:i + 32768] = ref.log_posterior_theta(theta[i:i + 32768])
    counter["evals"] += len(theta)
    if counter["evals"] > CAP_EVALS or time.perf_counter() - t_start > CAP_SECONDS:
        raise RuntimeError(f"cap exceeded: {counter['evals']} evaluations, {time.perf_counter() - t_start:.0f} s")
    return out


# -- mode, curvature, box (analytic target only) --------------------------------------------------------------
res = optimize.minimize(lambda th: -ell(th)[0], x0=np.log([0.4, 0.2, 0.4]), method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 4000})
res = optimize.minimize(lambda th: -ell(th)[0], x0=res.x, method="BFGS", options={"gtol": 1e-9})
theta_star, ell_star = res.x, -res.fun
h = 1e-4; H = np.zeros((3, 3))
for a in range(3):
    for b in range(3):
        ea, eb = np.eye(3)[a] * h, np.eye(3)[b] * h
        H[a, b] = -(ell(theta_star + ea + eb)[0] - ell(theta_star + ea - eb)[0] - ell(theta_star - ea + eb)[0] + ell(theta_star - ea - eb)[0]) / (4 * h * h)
scales = np.sqrt(np.diag(np.linalg.inv(H)))
dense_at_mode = ref.dense_log_marginal(*np.exp(theta_star))
lowrank_at_mode = float(ref.log_marginal(*[np.array([v]) for v in np.exp(theta_star)])[0])


def box(c, extra=None):
    extra = np.zeros(3) if extra is None else extra
    return np.column_stack([theta_star - c * scales - extra, theta_star + c * scales + extra])


def gl(n, lo, hi):
    tnodes, w = np.polynomial.legendre.leggauss(n)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    return mid + half * tnodes, w * half


def faces(B, n=48):
    """max(ell - ell*) over each of the six faces of the box."""
    out = {}
    for k in range(3):
        others = [i for i in range(3) if i != k]
        nodes = [gl(n, B[i, 0], B[i, 1])[0] for i in others]
        m = np.meshgrid(*nodes, indexing="ij")
        for side, val in (("lo", B[k, 0]), ("hi", B[k, 1])):
            th = np.zeros((n * n, 3)); th[:, others[0]] = m[0].ravel(); th[:, others[1]] = m[1].ravel(); th[:, k] = val
            out[f"{NAMES[k]}:{side}"] = float((ell(th) - ell_star).max())
    return out


extra = np.zeros(3); tail = None
for attempt in range(4):
    B8 = box(8.0, extra); tail = faces(B8)
    bad = [k for k in range(3) if max(tail[f"{NAMES[k]}:lo"], tail[f"{NAMES[k]}:hi"]) > -25]
    if not bad:
        break
    for k in bad:
        extra[k] += 2 * scales[k]
B10 = box(10.0, extra)


def quad3d(B, n):
    nodes, weights = zip(*[gl(n, B[k, 0], B[k, 1]) for k in range(3)])
    m = np.meshgrid(*nodes, indexing="ij"); wm = np.meshgrid(*weights, indexing="ij")
    th = np.column_stack([a.ravel() for a in m]); logw = np.log(wm[0] * wm[1] * wm[2]).ravel()
    lp = ell(th) + logw
    logZ = logsumexp(lp); p = np.exp(lp - logZ)
    means = [float((np.exp(th[:, k]) * p).sum()) for k in range(3)]
    sds = [float(np.sqrt(((np.exp(th[:, k]) - means[k]) ** 2 * p).sum())) for k in range(3)]
    return {"n": n, "logZ": float(logZ), "means": means, "posterior_sds": sds, "evaluations": n ** 3}


levels = [quad3d(B8, n) for n in (24, 32, 48, 64)]
expanded = quad3d(B10, 64)
refine_err = {NAMES[k]: abs(levels[-1]["means"][k] - levels[-2]["means"][k]) for k in range(3)}
domain_err = {NAMES[k]: abs(expanded["means"][k] - levels[-1]["means"][k]) for k in range(3)}
logZ_refine, logZ_domain = abs(levels[-1]["logZ"] - levels[-2]["logZ"]), abs(expanded["logZ"] - levels[-1]["logZ"])


def marginal(k, B, n_fine=141, n_other=48):
    others = [i for i in range(3) if i != k]
    fine = np.linspace(B[k, 0], B[k, 1], n_fine)
    nodes, weights = zip(*[gl(n_other, B[i, 0], B[i, 1]) for i in others])
    m = np.meshgrid(*nodes, indexing="ij"); wm = np.meshgrid(*weights, indexing="ij")
    logw = np.log(wm[0] * wm[1]).ravel()
    th = np.zeros((n_fine * n_other * n_other, 3))
    th[:, others[0]] = np.tile(m[0].ravel(), n_fine); th[:, others[1]] = np.tile(m[1].ravel(), n_fine); th[:, k] = np.repeat(fine, n_other * n_other)
    lp = (ell(th) + np.tile(logw, n_fine)).reshape(n_fine, -1)
    logm = logsumexp(lp, axis=1)                                            # unnormalised log marginal on the fine grid
    dens = np.exp(logm - logm.max())
    mass = np.trapezoid(dens, fine); dens_n = dens / mass
    log_mass_vs_Z = float(np.log(mass) + logm.max() - levels[-1]["logZ"])   # should be ~0 if the 1-D and 3-D normalisations agree
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens_n[1:] + dens_n[:-1]) * np.diff(fine))]); cdf /= cdf[-1]
    mean = float(np.trapezoid(np.exp(fine) * dens_n, fine))
    def F(value):                                                            # CDF at a natural-scale value
        return float(np.interp(np.log(value), fine, cdf))
    def Q(prob):
        return float(np.exp(np.interp(prob, cdf, fine)))
    return {"theta_grid": fine.tolist(), "density_theta": dens_n.tolist(), "cdf": cdf.tolist(), "mean": mean,
            "log_mass_minus_logZ": log_mass_vs_Z, "F": F, "Q": Q, "evaluations": n_fine * n_other * n_other}


margs = {NAMES[k]: marginal(k, B8) for k in range(3)}
consistency = {NAMES[k]: abs(margs[NAMES[k]]["mean"] - levels[-1]["means"][k]) for k in range(3)}
records = json.load(open("docs/m3b1-evidence/F7_varying_slope_recovery_records.json"))
targets_met = (all(v <= 1e-4 for v in refine_err.values()) and all(v <= 1e-4 for v in domain_err.values())
               and logZ_refine <= 1e-4 and logZ_domain <= 1e-4 and all(v <= 1e-4 for v in consistency.values()))
cdf_points = {"x|g_sigma@0.30": margs["x|g_sigma"]["F"](0.30)}
for be, rec in records.items():
    for k in NAMES:
        cdf_points[f"{k}@mcmc_mean_{be}"] = margs[k]["F"](rec["posterior"][k]["mean"])
# CDF stability at the reference points: recompute the sd1 marginal on B(10) and with 64 nodes
alt = marginal(1, B10, n_fine=141, n_other=64)
cdf_stability = {"x|g_sigma@0.30_B8_48": cdf_points["x|g_sigma@0.30"], "x|g_sigma@0.30_B10_64": alt["F"](0.30),
                 "abs_change": abs(alt["F"](0.30) - cdf_points["x|g_sigma@0.30"])}
targets_met = targets_met and cdf_stability["abs_change"] <= 1e-3
comparison = {}
if targets_met:
    for be, rec in records.items():
        comparison[be] = {}
        for k in NAMES:
            mc, mcse = rec["posterior"][k]["mean"], rec["posterior"][k]["mcse_mean"]
            err_q = max(refine_err[k], domain_err[k], consistency[k])
            diff = abs(levels[-1]["means"][NAMES.index(k)] - mc)
            comparison[be][k] = {"quadrature_mean": levels[-1]["means"][NAMES.index(k)], "mcmc_mean": mc, "abs_diff": diff, "mcse_stored": mcse,
                                 "tolerance_4mcse_plus_err": 4 * mcse + err_q, "within": bool(diff <= 4 * mcse + err_q), "quadrature_error_used": err_q}
intervals = {k: {"central_99": [margs[k]["Q"](0.005), margs[k]["Q"](0.995)], "central_90": [margs[k]["Q"](0.05), margs[k]["Q"](0.95)],
                 "median": margs[k]["Q"](0.5), "mean": margs[k]["mean"]} for k in NAMES}
elapsed = time.perf_counter() - t_start
result = {"protocol": "docs/m3b1-decision.md, Marginal-reference protocol (2026-09-12)", "data": "f7_data(seed=17): 600 rows, 30 groups of 20",
          "mode_theta": theta_star.tolist(), "mode_scales": np.exp(theta_star).tolist(), "log_posterior_at_mode": float(ell_star),
          "curvature_scales_theta": scales.tolist(), "lowrank_vs_dense_marginal_at_mode": lowrank_at_mode - dense_at_mode,
          "box8": B8.tolist(), "box10": B10.tolist(), "face_max_ell_minus_ellstar": tail, "levels": levels, "expanded_B10_n64": expanded,
          "refinement_error_means": refine_err, "domain_error_means": domain_err, "logZ_refinement_error": logZ_refine, "logZ_domain_error": logZ_domain,
          "marginal_vs_3d_mean_consistency": consistency, "marginal_mass_minus_logZ": {k: margs[k]["log_mass_minus_logZ"] for k in NAMES},
          "cdf_points": cdf_points, "cdf_stability": cdf_stability, "targets_met": bool(targets_met), "comparison": comparison, "intervals": intervals,
          "marginals": {k: {kk: margs[k][kk] for kk in ("theta_grid", "density_theta", "cdf")} for k in NAMES},
          "evaluations": counter["evals"], "seconds": elapsed, "caps": {"evaluations": CAP_EVALS, "seconds": CAP_SECONDS}}
out_json.write_text(json.dumps(result, indent=1) + "\n")
lines = [f"F7 marginal reference: {counter['evals']} evaluations in {elapsed:.0f} s (caps {CAP_EVALS}, {CAP_SECONDS} s)",
         f"mode (sd0, sd1, sigma) = {np.round(np.exp(theta_star), 5).tolist()}, log posterior at mode {ell_star:.6f}; low-rank minus dense at mode {lowrank_at_mode - dense_at_mode:.2e}",
         f"curvature scales (theta) {np.round(scales, 4).tolist()}; face maxima of ell - ell*: " + ", ".join(f"{k} {v:.1f}" for k, v in tail.items()),
         "3-D Gauss-Legendre levels (n, logZ, means):"] + [f"  n={l['n']:>2}: logZ {l['logZ']:.8f}  means {np.round(l['means'], 7).tolist()}" for l in levels] + [
         f"  B(10) n=64: logZ {expanded['logZ']:.8f}  means {np.round(expanded['means'], 7).tolist()}",
         f"refinement error of means {dict((k, f'{v:.2e}') for k, v in refine_err.items())}; domain error {dict((k, f'{v:.2e}') for k, v in domain_err.items())}",
         f"logZ refinement error {logZ_refine:.2e}, domain error {logZ_domain:.2e}; marginal-vs-3D mean consistency {dict((k, f'{v:.2e}') for k, v in consistency.items())}",
         f"marginal mass minus logZ {dict((k, f'{v:.2e}') for k, v in result['marginal_mass_minus_logZ'].items())}",
         f"CDF P(x|g_sigma <= 0.30 | y) = {cdf_points['x|g_sigma@0.30']:.6f} (B8/48) vs {cdf_stability['x|g_sigma@0.30_B10_64']:.6f} (B10/64), change {cdf_stability['abs_change']:.2e}",
         f"targets met: {targets_met}"]
for k in NAMES:
    lines.append(f"{k}: quadrature mean {intervals[k]['mean']:.6f}, median {intervals[k]['median']:.6f}, central 90% {np.round(intervals[k]['central_90'], 5).tolist()}, central 99% {np.round(intervals[k]['central_99'], 5).tolist()}")
for be in comparison:
    for k, c in comparison[be].items():
        lines.append(f"compare {be} {k}: quad {c['quadrature_mean']:.6f} vs mcmc {c['mcmc_mean']:.6f} |diff| {c['abs_diff']:.2e} tol(4 mcse + err) {c['tolerance_4mcse_plus_err']:.2e} within {c['within']}")
out_txt.write_text("\n".join(lines) + "\n"); print("\n".join(lines))
