#!/usr/bin/env python
"""Compare the two benchmark runs: wall times, effective sample size per second, and posterior agreement.

    python examples/brms_parity/compare.py <out_dir>

Reads <out_dir>/autobayes_<model>.json and brms_<model>.json and writes <out_dir>/comparison.json. Both libraries'
draws go through the same ArviZ estimators (rank-normalised bulk ESS, R-hat, MCSE of the mean), so the numbers are
comparable. A parameter's posterior means agree when |mean_AutoBayes - mean_brms| is within 4 combined MCSE.
"""
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np


def _arr(values, chains=4):
    a = np.asarray(values, dtype=float)
    return a.reshape(chains, -1)


def main(out_dir: str) -> None:
    out = Path(out_dir)
    rows = []
    for ab_path in sorted(out.glob("autobayes_*.json")):
        name = ab_path.stem.removeprefix("autobayes_")
        ab_rec = json.loads(ab_path.read_text())
        br_rec = json.loads((out / f"brms_{name}.json").read_text())
        params = [p for p in br_rec["draws"] if p in ab_rec["draws"]]
        per = {}
        for p in params:
            a, b = _arr(ab_rec["draws"][p]), _arr(br_rec["draws"][p])
            ess_a, ess_b = float(az.ess(a, method="bulk")), float(az.ess(b, method="bulk"))
            mcse_a, mcse_b = float(np.ravel(az.mcse(a, method="mean"))[0]), float(np.ravel(az.mcse(b, method="mean"))[0])
            z = (a.mean() - b.mean()) / np.hypot(mcse_a, mcse_b)
            per[p] = dict(mean_autobayes=a.mean(), mean_brms=b.mean(), sd_autobayes=a.std(), sd_brms=b.std(), z=z,
                          ess_autobayes=ess_a, ess_brms=ess_b, rhat_autobayes=float(az.rhat(a)), rhat_brms=float(az.rhat(b)))
        min_ess_a = min(v["ess_autobayes"] for v in per.values())
        min_ess_b = min(v["ess_brms"] for v in per.values())
        sample_a = ab_rec["fit_seconds"] - ab_rec["compile_estimate_seconds"]
        total_a = ab_rec["build_seconds"] + ab_rec["fit_seconds"]
        total_b = br_rec["compile_seconds"] + br_rec["sample_seconds"]
        rows.append(dict(model=name, n=ab_rec["n"], params=per,
                         autobayes=dict(build=ab_rec["build_seconds"], compile=ab_rec["compile_estimate_seconds"],
                                        fit=ab_rec["fit_seconds"], sample=sample_a, total=total_a, min_ess=min_ess_a,
                                        ess_per_s_sampling=min_ess_a / sample_a, ess_per_s_total=min_ess_a / total_a),
                         brms=dict(compile=br_rec["compile_seconds"], sample=br_rec["sample_seconds"], total=total_b,
                                   min_ess=min_ess_b, ess_per_s_sampling=min_ess_b / br_rec["sample_seconds"],
                                   ess_per_s_total=min_ess_b / total_b),
                         max_abs_z=max(abs(v["z"]) for v in per.values())))
        r = rows[-1]
        print(f"{name}: AutoBayes total {total_a:.1f} s (sampling ~{sample_a:.1f}), brms total {total_b:.1f} s "
              f"(compile {br_rec['compile_seconds']:.1f}, sampling {br_rec['sample_seconds']:.1f}); min ESS {min_ess_a:.0f} vs "
              f"{min_ess_b:.0f}; max |z| {r['max_abs_z']:.2f}")
    (out / "comparison.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
