#!/usr/bin/env python
"""Compare the two benchmark runs: wall times by stage, where the sampling time goes, effective sample size per second,
and posterior agreement.

    python examples/brms_parity/compare.py <out_dir> [--date YYYY-MM-DD]

Reads <out_dir>/autobayes_<model>.json and brms_<model>.json and writes <out_dir>/comparison-<date>.json (the date of
the runs, from the records, unless given); copy it to examples/brms_parity/results/ to keep it. Both libraries' draws go
through the same ArviZ estimators (rank-normalised bulk and tail ESS, R-hat, MCSE of the mean), so the numbers are
comparable. A parameter's posterior means agree when |mean_AutoBayes - mean_brms| is within 4 combined MCSE. Also carries
each side's divergent transitions and the post-processing timings (median of three calls after the first).

The decomposition (README.md): for each side, the sampler's execution time T (warm-up and sampling of every chain,
compilation excluded), the leapfrog steps S summed over chains and warm-up and sampling, the cost g of one log density
and gradient inside the sampler, and the CPU seconds C spent over T. g is AutoBayes's cost per evaluation inside one
compiled loop and rstan's direct call to the model instance less the cost of a call that does no work: the host-call
medians the benchmark also records (``g_host_us``) include a call overhead the samplers do not pay, a large one for a
jitted JAX function on the CPU. Then exactly

    T = S * g * O / U,   O = C / (S * g)  (CPU per leapfrog step, in gradients),   U = C / T  (cores kept busy),

so brms's T over AutoBayes's is the product of four factors: steps S_b / S_a, gradient g_b / g_a, overhead O_b / O_a and
parallelism U_a / U_b. The table also prints the additive form T = S_max * g + residual, with S_max the steps of the
chain that took the most.
"""
import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np


def _arr(values, chains):
    a = np.asarray(values, dtype=float)
    return a.reshape(chains, -1)


def _side(steps_w, steps_s, T, g_us, cpu) -> dict:
    """One side's decomposition (module docstring)."""
    steps_w, steps_s = np.atleast_1d(steps_w).tolist(), np.atleast_1d(steps_s).tolist()      # one chain: maybe a number
    per_chain = [w + s for w, s in zip(steps_w, steps_s)]
    S, g = float(sum(per_chain)), g_us * 1e-6
    out = dict(T=T, S=S, S_warmup=float(sum(steps_w)), S_sampling=float(sum(steps_s)), S_max_chain=float(max(per_chain)),
               g_us=g_us, cpu=cpu, S_g=S * g, S_max_g=max(per_chain) * g, residual=T - max(per_chain) * g)
    out.update(O=cpu / (S * g) if S * g > 0 else None, U=cpu / T if T > 0 else None)
    return out


def stages_autobayes(rec: dict) -> dict | None:
    st = rec.get("stages")
    if not st:
        return None
    g = st["gradient"]
    dec = _side(st["steps_warmup"], st["steps_sampling"], st["exec"], g["in_loop_us"], st["cpu_exec"])
    dec["g_host_us"], dec["g_dispatch_us"] = g["host"]["median_us"], g["dispatch"]["median_us"]
    return dict(build=rec["build_seconds"], compile=st["sampler_compile"], fit_pre=st["fit_pre"], exec=st["exec"],
                fit_post=st["fit_post"], fit=rec["fit_seconds"], total=rec["build_seconds"] + rec["fit_seconds"],
                compile_events_sampler=st["compile_events_sampler"], compile_events_fit=st["compile_events_fit"],
                nuts_compile_xla=st["nuts_compile_xla"], grad_lower=g["lower_seconds"], grad_compile=g["compile_seconds"],
                n_unconstrained=g["n_unconstrained"],
                tree_depth_warmup=st["tree_depth_warmup"], tree_depth_sampling=st["tree_depth_sampling"],
                step_size_final=st["step_size_final"], divergent_warmup=st["divergent_warmup"], decomposition=dec)


def stages_brms(rec: dict) -> dict | None:
    st = rec.get("stages")
    if not st:
        return None
    g = st["gradient"]
    net = g["direct"]["median_us"] - g["dispatch"]["median_us"]           # the direct call less a call that does no work
    dec = _side(st["steps_warmup"], st["steps_sampling"], st["exec"], net if net > 0 else g["direct"]["median_us"],
                st["cpu_exec"])
    dec["g_host_us"], dec["g_direct_us"], dec["g_dispatch_us"] = (g["host"]["median_us"], g["direct"]["median_us"],
                                                                  g["dispatch"]["median_us"])
    return dict(codegen=st["codegen"], compile=rec["compile_seconds"], sample=rec["sample_seconds"], exec=st["exec"],
                sample_overhead=rec["sample_seconds"] - st["exec"], total=rec["compile_seconds"] + rec["sample_seconds"],
                chain_warmup_seconds=st["chain_warmup_seconds"], chain_sampling_seconds=st["chain_sampling_seconds"],
                cpu_self=st["cpu_self"], cpu_child=st["cpu_child"], grad_lp_check=g["lp_check"],
                n_unconstrained=g["n_unconstrained"], grad_point_seconds=g["point_seconds"],
                tree_depth_warmup=st["tree_depth_warmup"], tree_depth_sampling=st["tree_depth_sampling"],
                step_size_final=st["step_size_final"], divergent_warmup=st["divergent_warmup"], decomposition=dec)


def factors(a: dict, b: dict) -> dict:
    """brms's sampler time over AutoBayes's, as the product of the four factors of the module docstring."""
    f = dict(speedup=b["T"] / a["T"], steps=b["S"] / a["S"], gradient=b["g_us"] / a["g_us"])
    f["overhead"] = b["O"] / a["O"] if a["O"] and b["O"] else None
    f["parallelism"] = a["U"] / b["U"] if a["U"] and b["U"] else None
    named = {k: v for k, v in f.items() if k != "speedup" and v}
    f["largest"] = max(named, key=lambda k: abs(np.log(named[k]))) if named else None
    return f


def main(out_dir: str, date: str | None = None) -> None:
    out = Path(out_dir)
    rows = []
    for ab_path in sorted(out.glob("autobayes_*.json"), key=lambda p: int(p.stem.split("_")[1][1:])):
        name = ab_path.stem.removeprefix("autobayes_")
        br_path = out / f"brms_{name}.json"
        if not br_path.exists():
            print(f"{name}: no brms record, skipped")
            continue
        ab_rec, br_rec = json.loads(ab_path.read_text()), json.loads(br_path.read_text())
        ca, cb = ab_rec.get("chains", 4), br_rec.get("chains", 4)
        date = date or ab_rec.get("date")
        params = [p for p in br_rec["draws"] if p in ab_rec["draws"]]
        per = {}
        for p in params:
            a, b = _arr(ab_rec["draws"][p], ca), _arr(br_rec["draws"][p], cb)
            ess_a, ess_b = float(az.ess(a, method="bulk")), float(az.ess(b, method="bulk"))
            tail_a, tail_b = float(az.ess(a, method="tail")), float(az.ess(b, method="tail"))
            mcse_a, mcse_b = float(np.ravel(az.mcse(a, method="mean"))[0]), float(np.ravel(az.mcse(b, method="mean"))[0])
            z = (a.mean() - b.mean()) / np.hypot(mcse_a, mcse_b)
            per[p] = dict(mean_autobayes=a.mean(), mean_brms=b.mean(), sd_autobayes=a.std(), sd_brms=b.std(), z=z,
                          ess_autobayes=ess_a, ess_brms=ess_b, ess_tail_autobayes=tail_a, ess_tail_brms=tail_b,
                          rhat_autobayes=float(az.rhat(a)) if ca > 1 else None,
                          rhat_brms=float(az.rhat(b)) if cb > 1 else None)
        only = dict(autobayes=sorted(set(ab_rec["draws"]) - set(params)), brms=sorted(set(br_rec["draws"]) - set(params)))
        min_ess_a = min(v["ess_autobayes"] for v in per.values())
        min_ess_b = min(v["ess_brms"] for v in per.values())
        min_tail_a = min(v["ess_tail_autobayes"] for v in per.values())
        min_tail_b = min(v["ess_tail_brms"] for v in per.values())
        sample_a = ab_rec["fit_seconds"] - ab_rec["compile_estimate_seconds"]          # the 17 October definition
        total_a = ab_rec["build_seconds"] + ab_rec["fit_seconds"]
        total_b = br_rec["compile_seconds"] + br_rec["sample_seconds"]
        sa, sb = stages_autobayes(ab_rec), stages_brms(br_rec)
        row = dict(model=name, n=ab_rec["n"], params=per, params_one_side=only,
                   autobayes=dict(build=ab_rec["build_seconds"], compile=ab_rec["compile_estimate_seconds"],
                                  fit=ab_rec["fit_seconds"], sample=sample_a, total=total_a, min_ess=min_ess_a,
                                  min_ess_tail=min_tail_a, ess_per_s_sampling=min_ess_a / sample_a,
                                  ess_tail_per_s_sampling=min_tail_a / sample_a, ess_per_s_total=min_ess_a / total_a),
                   brms=dict(compile=br_rec["compile_seconds"], sample=br_rec["sample_seconds"], total=total_b,
                             min_ess=min_ess_b, min_ess_tail=min_tail_b,
                             ess_per_s_sampling=min_ess_b / br_rec["sample_seconds"],
                             ess_tail_per_s_sampling=min_tail_b / br_rec["sample_seconds"],
                             ess_per_s_total=min_ess_b / total_b),
                   max_abs_z=max(abs(v["z"]) for v in per.values()),
                   divergent=dict(autobayes=ab_rec.get("divergent"), brms=br_rec.get("divergent")),
                   post={fn: dict(autobayes=ab_rec.get("post", {}).get(fn), brms=br_rec.get("post", {}).get(fn))
                         for fn in ab_rec.get("post", {})},
                   settings=dict(autobayes=ab_rec.get("settings"), brms=br_rec.get("settings"),
                                 smoke=bool(ab_rec.get("smoke") or br_rec.get("smoke"))),
                   versions=dict(autobayes=ab_rec.get("versions"), brms=br_rec.get("versions")))
        if sa and sb:
            row["stages"] = dict(autobayes=sa, brms=sb)
            row["factors"] = factors(sa["decomposition"], sb["decomposition"])
        rows.append(row)
        print(f"{name}: AutoBayes total {total_a:.1f} s (sampling ~{sample_a:.1f}), brms total {total_b:.1f} s "
              f"(compile {br_rec['compile_seconds']:.1f}, sampling {br_rec['sample_seconds']:.1f}); min ESS bulk "
              f"{min_ess_a:.0f} vs {min_ess_b:.0f}, tail {min_tail_a:.0f} vs {min_tail_b:.0f}; max |z| {row['max_abs_z']:.2f}"
              + (f"; compared on one side only: {only}" if only["autobayes"] or only["brms"] else ""))
    print_decomposition(rows)
    path = out / f"comparison-{date}.json"
    path.write_text(json.dumps(rows, indent=1))
    print(f"wrote {path}")


def print_decomposition(rows) -> None:
    """The decomposition table: per model and side T, S, g, S_max * g, residual, O, U, then the four factors."""
    head = (f"{'model':<26}{'side':<10}{'T s':>8}{'steps':>10}{'warm-up':>9}{'g host':>9}{'g us':>9}{'Smax*g s':>10}"
            f"{'resid s':>9}{'O':>6}{'U':>6}")
    print("\nWhere the sampling time goes (T = sampler execution, compilation excluded; steps summed over chains; g host = "
          "the host-call median, g = in the sampler)")
    print(head)
    print("-" * len(head))
    for r in rows:
        if "stages" not in r:
            continue
        for side in ("brms", "autobayes"):
            d = r["stages"][side]["decomposition"]
            print(f"{r['model'] if side == 'brms' else '':<26}{side:<10}{d['T']:>8.2f}{d['S']:>10.0f}"
                  f"{d['S_warmup'] / d['S']:>9.0%}{d['g_host_us']:>9.1f}{d['g_us']:>9.1f}{d['S_max_g']:>10.2f}"
                  f"{d['residual']:>9.2f}"
                  f"{d['O'] or float('nan'):>6.2f}{d['U'] or float('nan'):>6.2f}")
        f = r["factors"]
        fmt = lambda v: f"{v:.2f}" if v else "n/a"                                # noqa: E731
        print(f"{'':<26}brms/AutoBayes {fmt(f['speedup'])}x = steps {fmt(f['steps'])} x gradient {fmt(f['gradient'])} "
              f"x overhead {fmt(f['overhead'])} x parallelism {fmt(f['parallelism'])}  (largest: {f['largest']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--date", help="the date in the output name (default: the AutoBayes records' date)")
    a = ap.parse_args()
    main(a.out_dir, a.date)
