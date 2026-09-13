"""F7 marginal reference (docs/m3b1-decision.md, "Marginal-reference protocol"): deterministic quadrature of the
analytically integrated Gaussian varying-slope posterior of (sd0, sd1, sigma) on the frozen seed-17 data.

Usage: python f7_marginal_reference.py OUT.json OUT.txt LEDGER.json ATTEMPT_NAME   (repository root; NOT to be run
under the exhausted 2,000,000-evaluation unit ledger; a new cap needs a new written authorisation and ledger)

Structure (second budget-guard review, 2026-09-13): the numerical stages run through ``BudgetedEvaluator``
(tests/quadrature_budget.py: reservation before work, cumulative unit cap and deadline, ownership); every real
evaluation is counted, including the two dense/low-rank diagnostic calls at the mode (charged as one each);
``assemble_output`` is pure bookkeeping and is unit-tested with stubs: the output always carries the COMPLETE
prespecified condition inventory (six faces, refinement, domain, logZ, marginal consistency, and the CDF stability
of every reference point under the resolution and the domain variant separately), with conditions that were not
computed reported as unchecked, and acceptance decided fail-closed by ``acceptance_report``.
History: the first driver checked its counter after computing each batch, reset it per process and shadowed the
attempt name with an integer loop index; attempts 1 and 2 and a tail addendum spent 4,125,558 evaluations
(F7_marginal_reference_ACCOUNTING.txt)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy.special import logsumexp

sys.path.insert(0, ".")
from tests.quadrature_budget import BudgetExhausted, BudgetedEvaluator, acceptance_report, run_is_valid   # noqa: E402

CAP_EVALS, CAP_SECONDS = 2_000_000, 15 * 60
NAMES = ["1|g_sigma", "x|g_sigma", "sigma"]
FACES = [f"{k}:{side}" for k in NAMES for side in ("lo", "hi")]
LEVELS = (24, 32, 48, 64)
CDF_VARIANTS = {"resolution_B8_48": ("B8", 48), "domain_B10_40": ("B10", 40)}
TARGETS = {"mean": 1e-4, "logZ": 1e-4, "consistency": 1e-4, "cdf": 1e-3, "face": -25.0}


# -- inventory -----------------------------------------------------------------------------------------------
def reference_points(records: dict) -> dict:
    """sd1 = 0.30 (the population slope SD) and every stored MCMC mean of every scale, per scale."""
    pts = {k: [] for k in NAMES}
    pts["x|g_sigma"].append(("population_0.30", 0.30))
    for be, rec in records.items():
        for k in NAMES:
            pts[k].append((f"mcmc_mean_{be}", float(rec["posterior"][k]["mean"])))
    return pts


def condition_inventory(ref_points: dict) -> list:
    """The complete prespecified list of acceptance conditions, fixed before any stage runs."""
    names = [f"face:{f}" for f in FACES]
    names += [f"refinement_mean:{k}" for k in NAMES] + [f"domain_mean:{k}" for k in NAMES]
    names += ["logZ_refinement", "logZ_domain"] + [f"marginal_consistency:{k}" for k in NAMES]
    names += [f"cdf_stability:{k}@{lab}:{variant}" for k in NAMES for lab, _ in ref_points[k] for variant in CDF_VARIANTS]
    return names


# -- numerical stages (each may stop with BudgetExhausted; partial results are kept) -------------------------
def gl(n, lo, hi):
    t, w = np.polynomial.legendre.leggauss(n)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    return mid + half * t, w * half


def compute_stages(ref, evaluator, records, stages: dict) -> dict:
    """Runs the protocol's stages in order, filling ``stages`` as it goes; returns the stage at which it stopped."""
    def ell(theta):
        return evaluator(np.atleast_2d(theta))
    try:
        # mode, curvature, diagnostic low-rank/dense agreement (charged as real work)
        res = optimize.minimize(lambda th: -ell(th)[0], x0=np.log([0.4, 0.2, 0.4]), method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 4000})
        res = optimize.minimize(lambda th: -ell(th)[0], x0=res.x, method="BFGS", options={"gtol": 1e-9})
        theta_star, ell_star = res.x, -res.fun
        h = 1e-4; H = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                ea, eb = np.eye(3)[a] * h, np.eye(3)[b] * h
                H[a, b] = -(ell(theta_star + ea + eb)[0] - ell(theta_star + ea - eb)[0] - ell(theta_star - ea + eb)[0] + ell(theta_star - ea - eb)[0]) / (4 * h * h)
        scales = np.sqrt(np.diag(np.linalg.inv(H)))
        rid = evaluator.charge(2, "dense and low-rank diagnostic marginal at the mode"); t0 = evaluator.clock()
        try:
            dense_at_mode = ref.dense_log_marginal(*np.exp(theta_star))
            lowrank_at_mode = float(ref.log_marginal(*[np.array([v]) for v in np.exp(theta_star)])[0])
        except BaseException as e:
            evaluator.fail_charge(rid, f"{type(e).__name__}: {e}", evaluator.clock() - t0)
            raise
        evaluator.confirm_charge(rid, 2, evaluator.clock() - t0)
        stages["mode"] = dict(theta_star=theta_star.tolist(), ell_star=float(ell_star), scales=scales.tolist(), lowrank_minus_dense=lowrank_at_mode - dense_at_mode)

        def box(c, extra):
            return np.column_stack([theta_star - c * scales - extra, theta_star + c * scales + extra])

        def faces(B, n=48):
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
        for expansion in range(4):                                  # distinct name: never shadows the attempt label
            B8 = box(8.0, extra); tail = faces(B8)
            bad = [k for k in range(3) if max(tail[f"{NAMES[k]}:lo"], tail[f"{NAMES[k]}:hi"]) > TARGETS["face"]]
            if not bad:
                break
            for k in bad:
                extra[k] += 2 * scales[k]
        B10 = box(10.0, extra)
        stages["faces"] = tail; stages["box8"] = B8.tolist(); stages["box10"] = B10.tolist(); stages["expansions"] = extra.tolist()

        def quad3d(B, n):
            nodes, weights = zip(*[gl(n, B[k, 0], B[k, 1]) for k in range(3)])
            m = np.meshgrid(*nodes, indexing="ij"); wm = np.meshgrid(*weights, indexing="ij")
            th = np.column_stack([a.ravel() for a in m]); logw = np.log(wm[0] * wm[1] * wm[2]).ravel()
            lp = ell(th) + logw
            logZ = logsumexp(lp); p = np.exp(lp - logZ)
            return {"n": n, "logZ": float(logZ), "means": [float((np.exp(th[:, k]) * p).sum()) for k in range(3)], "evaluations": n ** 3}

        stages["levels"] = []
        for n in LEVELS:
            stages["levels"].append(quad3d(B8, n))
        stages["expanded_B10_n64"] = quad3d(B10, 64)

        def marginal(k, B, n_fine, n_other):
            others = [i for i in range(3) if i != k]
            fine = np.linspace(B[k, 0], B[k, 1], n_fine)
            nodes, weights = zip(*[gl(n_other, B[i, 0], B[i, 1]) for i in others])
            m = np.meshgrid(*nodes, indexing="ij"); wm = np.meshgrid(*weights, indexing="ij"); logw = np.log(wm[0] * wm[1]).ravel()
            th = np.zeros((n_fine * n_other * n_other, 3))
            th[:, others[0]] = np.tile(m[0].ravel(), n_fine); th[:, others[1]] = np.tile(m[1].ravel(), n_fine); th[:, k] = np.repeat(fine, n_other * n_other)
            logm = logsumexp((ell(th) + np.tile(logw, n_fine)).reshape(n_fine, -1), axis=1)
            dens = np.exp(logm - logm.max()); mass = np.trapezoid(dens, fine); dens_n = dens / mass
            cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens_n[1:] + dens_n[:-1]) * np.diff(fine))]); cdf /= cdf[-1]
            return {"theta_grid": fine.tolist(), "density_theta": dens_n.tolist(), "cdf": cdf.tolist(),
                    "mean": float(np.trapezoid(np.exp(fine) * dens_n, fine)), "log_mass_minus_logZ": float(np.log(mass) + logm.max() - stages["levels"][-1]["logZ"])}

        stages["marginals"] = {}
        for k in range(3):
            stages["marginals"][NAMES[k]] = marginal(k, B8, 141, 40)
        stages["cdf_variants"] = {}
        for k in range(3):
            for variant, (which, n_other) in CDF_VARIANTS.items():
                stages["cdf_variants"][f"{NAMES[k]}:{variant}"] = marginal(k, B8 if which == "B8" else B10, 141, n_other)
        return "complete"
    except BudgetExhausted as e:
        stages["stopped"] = str(e)
        return "stopped"


# -- pure bookkeeping: always the full inventory, fail-closed acceptance ---------------------------------------
def cdf_at(marg: dict, value: float) -> float:
    return float(np.interp(np.log(value), marg["theta_grid"], marg["cdf"]))


def quantile(marg: dict, prob: float) -> float:
    return float(np.exp(np.interp(prob, marg["cdf"], marg["theta_grid"])))


def assemble_output(stages: dict, ledger_view: dict, records: dict, attempt: str, caps: dict) -> dict:
    ref_points = reference_points(records)
    required = condition_inventory(ref_points)
    conditions = {name: None for name in required}                  # unchecked until computed
    faces = stages.get("faces") or {}
    for f in FACES:
        if f in faces:
            conditions[f"face:{f}"] = bool(faces[f] <= TARGETS["face"])
    levels = stages.get("levels") or []; expanded = stages.get("expanded_B10_n64")
    errors = {}
    if len(levels) >= 2:
        for i, k in enumerate(NAMES):
            errors[f"refinement_mean:{k}"] = abs(levels[-1]["means"][i] - levels[-2]["means"][i])
            conditions[f"refinement_mean:{k}"] = bool(errors[f"refinement_mean:{k}"] <= TARGETS["mean"])
        errors["logZ_refinement"] = abs(levels[-1]["logZ"] - levels[-2]["logZ"]); conditions["logZ_refinement"] = bool(errors["logZ_refinement"] <= TARGETS["logZ"])
    if levels and expanded:
        for i, k in enumerate(NAMES):
            errors[f"domain_mean:{k}"] = abs(expanded["means"][i] - levels[-1]["means"][i])
            conditions[f"domain_mean:{k}"] = bool(errors[f"domain_mean:{k}"] <= TARGETS["mean"])
        errors["logZ_domain"] = abs(expanded["logZ"] - levels[-1]["logZ"]); conditions["logZ_domain"] = bool(errors["logZ_domain"] <= TARGETS["logZ"])
    margs = stages.get("marginals") or {}
    if levels:
        for i, k in enumerate(NAMES):
            if k in margs:
                errors[f"marginal_consistency:{k}"] = abs(margs[k]["mean"] - levels[-1]["means"][i])
                conditions[f"marginal_consistency:{k}"] = bool(errors[f"marginal_consistency:{k}"] <= TARGETS["consistency"])
    variants = stages.get("cdf_variants") or {}
    cdf_points = {}
    for k in NAMES:
        if k not in margs:
            continue
        for lab, val in ref_points[k]:
            cdf_points[f"{k}@{lab}"] = cdf_at(margs[k], val)
            for variant in CDF_VARIANTS:
                alt = variants.get(f"{k}:{variant}")
                if alt is not None:
                    change = abs(cdf_at(alt, val) - cdf_points[f"{k}@{lab}"])
                    errors[f"cdf_stability:{k}@{lab}:{variant}"] = change
                    conditions[f"cdf_stability:{k}@{lab}:{variant}"] = bool(change <= TARGETS["cdf"])
    protocol = acceptance_report(conditions, required)
    run = run_is_valid(ledger_view, stages.get("stopped"))
    acceptance = {**protocol, "protocol_accepted": protocol["accepted"], "run_valid": run["valid"], "run_problems": run["problems"],
                  "accepted": bool(protocol["accepted"] and run["valid"])}
    comparison = {}
    if acceptance["accepted"]:
        for be, rec in records.items():
            comparison[be] = {}
            for i, k in enumerate(NAMES):
                mc, mcse = float(rec["posterior"][k]["mean"]), float(rec["posterior"][k]["mcse_mean"])
                err_q = max(errors[f"refinement_mean:{k}"], errors[f"domain_mean:{k}"], errors[f"marginal_consistency:{k}"])
                diff = abs(levels[-1]["means"][i] - mc)
                comparison[be][k] = {"quadrature_mean": levels[-1]["means"][i], "mcmc_mean": mc, "abs_diff": diff, "mcse_stored": mcse,
                                     "tolerance_4mcse_plus_err": 4 * mcse + err_q, "within": bool(diff <= 4 * mcse + err_q), "quadrature_error_used": err_q}
    intervals = {k: {"central_99": [quantile(margs[k], 0.005), quantile(margs[k], 0.995)], "central_90": [quantile(margs[k], 0.05), quantile(margs[k], 0.95)],
                     "median": quantile(margs[k], 0.5), "mean": margs[k]["mean"]} for k in NAMES if k in margs}
    return {"protocol": "docs/m3b1-decision.md, Marginal-reference protocol (2026-09-12)", "attempt": attempt, "caps": caps,
            "budget": ledger_view, "stopped": stages.get("stopped"), "mode": stages.get("mode"), "box8": stages.get("box8"), "box10": stages.get("box10"),
            "expansions": stages.get("expansions"), "face_max_ell_minus_ellstar": faces, "levels": levels, "expanded_B10_n64": expanded,
            "errors": errors, "cdf_points": cdf_points, "conditions": conditions, "required_conditions": required, "acceptance": acceptance,
            "status": "complete" if acceptance["accepted"] else "incomplete: " + "; ".join(
                [f"{k}={len(v)}" for k, v in protocol.items() if isinstance(v, list) and v and k != "met"] + list(run["problems"])),
            "exit_code": exit_code(acceptance),
            "comparison": comparison, "intervals": intervals,
            "marginals": {k: {kk: margs[k][kk] for kk in ("theta_grid", "density_theta", "cdf")} for k in margs}}


def exit_code(acceptance: dict) -> int:
    """0 only for a valid run whose every protocol condition is met; 3 stopped/over-budget/invalid accounting;
    4 protocol conditions failed, unchecked, omitted, malformed or unexpected."""
    if acceptance["accepted"]:
        return 0
    if not acceptance["run_valid"]:
        return 3
    return 4


def write_summary(result: dict, out_txt: Path) -> None:
    b = result["budget"]
    lines = [f"F7 marginal reference ({result['attempt']}): requested {b['requested']}, reserved-uncertain {b['reserved_uncertain']}, confirmed {b['confirmed']} (cumulative, unit cap {result['caps']['evaluations']}); status {result['status']}",
             f"stopped: {result['stopped']}", f"acceptance: {result['acceptance']}"]
    for k, v in result["intervals"].items():
        lines.append(f"{k}: mean {v['mean']:.6f}, median {v['median']:.6f}, central 90% {np.round(v['central_90'], 5).tolist()}, central 99% {np.round(v['central_99'], 5).tolist()}")
    for be in result["comparison"]:
        for k, c in result["comparison"][be].items():
            lines.append(f"compare {be} {k}: quad {c['quadrature_mean']:.6f} vs mcmc {c['mcmc_mean']:.6f} |diff| {c['abs_diff']:.2e} tol {c['tolerance_4mcse_plus_err']:.2e} within {c['within']}")
    out_txt.write_text("\n".join(lines) + "\n")


def main(argv):
    from tests.reference_marginal import GaussianSlopeMarginal, f7_data
    out_json, out_txt, ledger_path, attempt = Path(argv[1]), Path(argv[2]), Path(argv[3]), str(argv[4])
    y, x, g = f7_data()
    ref = GaussianSlopeMarginal(y, x, g, prior_mean=(0.0, 0.0), prior_sd=(5.0, 2.0), center=True)
    records = json.load(open("docs/m3b1-evidence/F7_varying_slope_recovery_records.json"))
    stages = {}
    caps = {"evaluations": CAP_EVALS, "seconds": CAP_SECONDS}

    def checkpoint(info):
        out_json.with_suffix(".partial.json").write_text(json.dumps({"attempt": attempt, "budget": info, "stages_done": sorted(stages)}, indent=1) + "\n")

    evaluator = BudgetedEvaluator(lambda th: ref.log_posterior_theta(th), cap_evaluations=CAP_EVALS, unit_deadline_seconds=CAP_SECONDS,
                                  ledger_path=ledger_path, attempt=attempt, batch_size=32768, checkpoint=checkpoint)
    outcome = compute_stages(ref, evaluator, records, stages)
    view = evaluator.progress(outcome)
    if outcome == "complete":
        evaluator.release()
    result = assemble_output(stages, view, records, attempt, caps)
    out_json.write_text(json.dumps(result, indent=1) + "\n"); write_summary(result, out_txt)
    print(f"[f7_marginal_reference] {result['status']}; budget {result['budget']}")
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main(sys.argv))
