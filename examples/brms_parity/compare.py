#!/usr/bin/env python
"""Compare the two benchmark runs: wall times by stage, where the sampling time goes, effective sample size per second,
and posterior agreement.

    python examples/brms_parity/compare.py <out_dir> [--date YYYY-MM-DD]
    python examples/brms_parity/compare.py <results_root> --scaling [--date YYYY-MM-DD]

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

``--scaling`` (examples/brms_parity/scaling/README.md) reads every record of a scaling model (S1-S5) under
<results_root>, at any depth, one per (model, N, library, configuration, seed), and writes <results_root>/scaling-<date>.json
without the draws: per record the smallest bulk and tail ESS over the parameters every record of that model reports, and
those per second of the samplers' execution T (warm-up and sampling, compilation excluded) and end to end (build or
compile plus the fit), the CPU seconds and cores used (U = C / T against the cores of the configuration), and the time
per leapfrog step of the slowest chain (T / S_max, which needs no separate gradient and so is there for cmdstanr too);
per configuration the median and range over the seeds; and per (model, N) each library's best configuration (the
highest median bulk ESS per second of T) against the other's.
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
    if "skipped" in g:                                    # cmdstanr: no gradient measured; no decomposition
        return None
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


def config_label(rec: dict) -> str:
    """The configuration a record was run with, from its own options: brms-<backend>[-t<threads>][-native][-unnorm],
    autobayes[-c<cores>]."""
    o = rec.get("options") or {}
    if rec["library"] == "autobayes":
        return f"autobayes-c{rec['cores']}" if rec.get("cores") else "autobayes"
    label = f"brms-{rec.get('backend', 'rstan')}" + (f"-t{o['threads']}" if o.get("threads") else "")
    if o.get("grainsize_requested"):
        label += f"-g{o['grainsize_requested']}"
    return label + ("-native" if o.get("march_native") else "") + ("" if o.get("normalize", True) else "-unnorm")


def scaling_record(rec: dict, params: list[str]) -> dict:
    """One scaling record's summary (module docstring)."""
    chains, st = rec.get("chains", 4), rec["stages"]
    ess = {p: (float(az.ess(_arr(rec["draws"][p], chains), method="bulk")),
               float(az.ess(_arr(rec["draws"][p], chains), method="tail"))) for p in params}
    rhat = [float(az.rhat(_arr(rec["draws"][p], chains))) for p in params] if chains > 1 else []
    steps = [w + s for w, s in zip(np.atleast_1d(st["steps_warmup"]).tolist(), np.atleast_1d(st["steps_sampling"]).tolist())]
    T, cpu = st["exec"], st["cpu_exec"]
    if rec["library"] == "autobayes":
        compile_s, total = st["sampler_compile"] + rec["build_seconds"], rec["build_seconds"] + rec["fit_seconds"]
        cores = rec.get("cores") or (rec.get("machine") or {}).get("n_allowed")
    else:
        compile_s, total = rec["compile_seconds"], rec["compile_seconds"] + rec["sample_seconds"]
        cores = rec.get("cores") or rec["settings"]["cores"]
    bulk, tail = min(v[0] for v in ess.values()), min(v[1] for v in ess.values())
    m = rec.get("machine") or {}
    return dict(model=rec["model"], n=rec["n"], library=rec["library"], config=config_label(rec),
                seed=(rec.get("options") or {}).get("seed", rec["settings"].get("seed")), smoke=bool(rec.get("smoke")),
                T=T, compile=compile_s, total=total, min_ess_bulk=bulk, min_ess_tail=tail,
                worst_bulk=min(ess, key=lambda p: ess[p][0]), ess_bulk_per_s=bulk / T, ess_tail_per_s=tail / T,
                ess_bulk_per_s_total=bulk / total, cpu=cpu, cores=cores, U=cpu / T if T > 0 else None,
                over_cap=bool(cores and cpu / T > 1.05 * cores), step_ms=T / max(steps) * 1e3 if max(steps) else None,
                steps=float(sum(steps)), max_rhat=max(rhat) if rhat else None,
                divergent=rec.get("divergent"), cpu_model=m.get("cpu_model"), host=m.get("host"),
                versions=rec.get("versions"), options=rec.get("options"))


def _summary(vals: list) -> dict:
    vals = [v for v in vals if v is not None]
    return dict(median=float(np.median(vals)), lo=float(min(vals)), hi=float(max(vals))) if vals else None


def scaling(root: str, date: str | None = None) -> Path:
    root_p = Path(root)
    recs = []
    for path in sorted(root_p.rglob("*.json")):
        if not path.name.startswith(("autobayes_S", "brms_S")):
            continue
        rec = json.loads(path.read_text())
        if rec.get("stages"):
            recs.append(rec)
    if not recs:
        raise SystemExit(f"no scaling records (autobayes_S*.json, brms_S*.json) under {root}")
    rows = []
    for model in sorted({r["model"] for r in recs}):
        mine = [r for r in recs if r["model"] == model]
        params = sorted(set.intersection(*(set(r["draws"]) for r in mine)))
        dropped = sorted(set.union(*(set(r["draws"]) for r in mine)) - set(params))
        if dropped:
            print(f"{model}: compared on {params}; not in every record: {dropped}")
        rows += [scaling_record(r, params) | dict(params=params) for r in mine]
    date = date or max(r.get("date", "") for r in recs)
    configs = []
    for key in sorted({(r["model"], r["n"], r["library"], r["config"]) for r in rows}):
        rs = [r for r in rows if (r["model"], r["n"], r["library"], r["config"]) == key]
        configs.append(dict(model=key[0], n=key[1], library=key[2], config=key[3], seeds=sorted(r["seed"] for r in rs),
                            cores=rs[0]["cores"], **{k: _summary([r[k] for r in rs]) for k in (
                                "ess_bulk_per_s", "ess_tail_per_s", "ess_bulk_per_s_total", "T", "compile", "total", "cpu",
                                "U", "step_ms", "min_ess_bulk", "min_ess_tail", "max_rhat")},
                            divergent=[r["divergent"] for r in rs], over_cap=any(r["over_cap"] for r in rs)))
    best = []
    for model, n in sorted({(c["model"], c["n"]) for c in configs}):
        side = {}
        for lib in ("brms", "autobayes"):
            cs = [c for c in configs if (c["model"], c["n"], c["library"]) == (model, n, lib) and c["ess_bulk_per_s"]]
            side[lib] = max(cs, key=lambda c: c["ess_bulk_per_s"]["median"]) if cs else None
        b, a = side["brms"], side["autobayes"]
        ratio = (a["ess_bulk_per_s"]["median"] / b["ess_bulk_per_s"]["median"]) if a and b else None
        best.append(dict(model=model, n=n, brms=b and b["config"], autobayes=a and a["config"], ratio_bulk=ratio,
                         ratio_tail=(a["ess_tail_per_s"]["median"] / b["ess_tail_per_s"]["median"]) if a and b else None,
                         ratio_total=(a["ess_bulk_per_s_total"]["median"] / b["ess_bulk_per_s_total"]["median"])
                         if a and b else None))
    print(f"{'model':<20}{'N':>9}  {'configuration':<24}{'bulk/s':>9}{'tail/s':>9}{'T s':>9}{'total s':>9}{'cores':>6}"
          f"{'U':>6}{'ms/step':>9}")
    for c in configs:
        f = lambda k, d=1: f"{c[k]['median']:.{d}f}" if c[k] else "--"       # noqa: E731
        print(f"{c['model']:<20}{c['n']:>9}  {c['config']:<24}{f('ess_bulk_per_s'):>9}{f('ess_tail_per_s'):>9}{f('T'):>9}"
              f"{f('total'):>9}{str(c['cores']):>6}{f('U', 2):>6}{f('step_ms', 3):>9}" + ("  CPU over the cap" if c["over_cap"] else ""))
    print("\nbest configuration of each library (median bulk ESS per second of the samplers' time over the seeds)")
    for b in best:
        r = f"{b['ratio_bulk']:.2f}" if b["ratio_bulk"] else "--"
        print(f"{b['model']:<20}{b['n']:>9}  brms {b['brms']}, AutoBayes {b['autobayes']}: AutoBayes / brms bulk ESS/s {r}")
    path = root_p / f"scaling-{date}.json"
    path.write_text(json.dumps(dict(date=date, smoke=any(r["smoke"] for r in rows), records=rows, configs=configs,
                                    best=best), indent=1))
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--date", help="the date in the output name (default: the AutoBayes records' date)")
    ap.add_argument("--scaling", action="store_true", help="summarise the scaling records under out_dir (module docstring)")
    a = ap.parse_args()
    if a.scaling:
        scaling(a.out_dir, a.date)
    else:
        main(a.out_dir, a.date)
