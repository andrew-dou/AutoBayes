#!/usr/bin/env python
"""Researcher example for milestone M0: y ~ x + (1 | group), Gaussian, explicit priors.

    .venv/bin/python examples/m0_varying_intercept.py                       # PyMC NUTS on CPU
    .venv/bin/python examples/m0_varying_intercept.py --backend numpyro     # JAX/NumPyro on CPU
    .venv/bin/python examples/m0_varying_intercept.py --backend numpyro --device gpu
    .venv/bin/python examples/m0_varying_intercept.py --backend numpyro --device gpu \
        --record docs/m0-evidence/gpu_record.json --compare docs/m0-evidence/B2_numpyro_cpu_record.json

The GPU form is the reproducible command for the GPU validation. It errors clearly on
a machine without a visible GPU rather than running on the CPU. ``--record`` writes the
fit record (``ab.fit_record``) as JSON; ``--compare`` checks the population-parameter
means against an earlier record within four combined MCSE (``ab.compare_records``).
"""
import argparse
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import autobayes as ab
from autobayes.diagnostics import format_panel, panel


def simulate(n=200, J=20, seed=20):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, J, n)
    x = rng.normal(size=n)
    u = rng.normal(0, 0.5, J)
    y = 1.0 + 0.5 * x + u[g] + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "x": x, "school": [f"s{k:02d}" for k in g]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="pymc", choices=["pymc", "numpyro"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--draws", type=int, default=2500)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--record", type=Path, default=None, help="write the fit record (JSON) here")
    ap.add_argument("--compare", type=Path, default=None, help="an earlier fit record to agree with")
    args = ap.parse_args()

    print(ab.format_environment(ab.environment()))
    print()
    df = simulate()
    priors = ab.Priors(
        intercept=ab.Normal(0, 5),        # on the intercept at the predictor means (centred)
        b=ab.Normal(0, 2),                # every slope
        sd=ab.HalfNormal(1),              # SD of the varying intercepts
        sigma=ab.HalfNormal(1),           # residual SD
    )
    model = ab.gaussian("y ~ x + (1 | school)", df, priors)
    print(model.describe())
    print()
    print("PyMC model:", model.pymc_model)   # inspect further with pytensor.dprint(model.pymc_model["mu"])

    fit = model.fit(backend=args.backend, device=args.device, chains=args.chains,
                    draws=args.draws, tune=args.tune, seed=args.seed)
    print()
    print(fit.report())
    print()
    print(format_panel(panel(fit.idata)))
    print()
    print(az.summary(fit.idata, var_names=["Intercept", "x", "1|school_sigma", "sigma"]).to_string())
    print("truth: Intercept 1.0, x 0.5, sd 0.5, sigma 0.3")

    ppc = model.predict(fit, kind="response", seed=1)
    print()
    print(f"posterior predictive: mean of y draws {float(ppc['y'].mean()):.3f} vs observed mean {df.y.mean():.3f}")
    new = pd.DataFrame({"x": [0.0, 0.0], "school": ["s00", "brand_new_school"]})
    pop = model.predict(fit, newdata=new, kind="response", new_groups="population", seed=1)
    sd_obs, sd_new = float(pop["y"][..., 0].std()), float(pop["y"][..., 1].std())
    print(f"predictive SD for an observed school {sd_obs:.3f}; for a new school (population draw) {sd_new:.3f}")

    record = ab.fit_record(model, fit)
    if args.record is not None:
        if args.record.exists():
            sys.exit(f"refusing to overwrite {args.record}")
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=1))
        print(f"fit record written to {args.record}")
    if args.compare is not None:
        earlier = json.loads(args.compare.read_text())
        print()
        print(ab.format_comparison(ab.compare_records(record, earlier)))


if __name__ == "__main__":
    main()
