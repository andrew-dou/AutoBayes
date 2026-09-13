"""Addendum to f7_marginal_reference.py (run once on 2026-09-13 WITHOUT any budget check, 97,600 evaluations, after the
unit cap had already been exceeded; see F7_marginal_reference_ACCOUNTING.txt): the mass of the x|g_sigma (log sd1)
marginal below the lower face of B(8). Kept for the record; now routed through the budget controller and NOT to be
run again under the exhausted cap. Usage: IN.json OUT.txt LEDGER.json ATTEMPT_NAME"""
import json, sys, time
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0, ".")
from tests.reference_marginal import GaussianSlopeMarginal, f7_data
from tests.quadrature_budget import BudgetedEvaluator

res = json.load(open(sys.argv[1]))
y, x, g = f7_data(); ref = GaussianSlopeMarginal(y, x, g, prior_mean=(0.0, 0.0), prior_sd=(5.0, 2.0), center=True)
B8 = np.array(res["box8"]); logZ = res["levels"][-1]["logZ"]; ell_star = res["log_posterior_at_mode"]
lo = B8[1, 0]
t0 = time.perf_counter(); evals = 0
evaluator = BudgetedEvaluator(lambda th: ref.log_posterior_theta(th), cap_evaluations=2_000_000, unit_deadline_seconds=900, ledger_path=sys.argv[3], attempt=sys.argv[4])


def gl(n, a, b):
    t, w = np.polynomial.legendre.leggauss(n); mid, half = (a + b) / 2, (b - a) / 2
    return mid + half * t, w * half


n0, w0 = gl(40, B8[0, 0], B8[0, 1]); n2, w2 = gl(40, B8[2, 0], B8[2, 1])
m0, m2 = np.meshgrid(n0, n2, indexing="ij"); logw = np.log(np.outer(w0, w2)).ravel()
ext = np.linspace(lo - 6.0, lo, 61)                                          # sd1 from lo*e^-6 (~3e-5) up to the face
logm = []
for th1 in ext:
    th = np.column_stack([m0.ravel(), np.full(m0.size, th1), m2.ravel()])
    logm.append(logsumexp(evaluator(th) + logw)); evals += m0.size
logm = np.array(logm)
mass_below = np.exp(logsumexp(np.log(np.gradient(ext)) + logm)) / np.exp(logZ)   # trapezoid-like on the extension
slope = np.gradient(logm, ext)
p = evaluator.progress("done")
lines = [f"tail addendum: {evals} evaluations this run in {time.perf_counter() - t0:.1f} s; ledger cumulative requested {p['requested']}, reserved-uncertain {p['reserved_uncertain']}, confirmed {p['confirmed']} (unit cap {evaluator.cap})",
         f"lower face of log x|g_sigma at theta = {lo:.4f} (sd1 = {np.exp(lo):.5f}); face value ell - ell* = {res['face_max_ell_minus_ellstar']['x|g_sigma:lo']:.2f}",
         f"marginal log density on the extension: at the face {logm[-1] - logZ:.3f}, at theta = lo - 3 {logm[30] - logZ:.3f}, at lo - 6 {logm[0] - logZ:.3f}; d/dtheta near the face {slope[-3:].mean():.3f} (Jacobian-dominated, ~1)",
         f"relative posterior mass of x|g_sigma below the face: {mass_below:.3e}  (error targets were 1e-4 for means, 1e-3 for CDFs)"]
open(sys.argv[2], "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
