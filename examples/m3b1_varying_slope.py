#!/usr/bin/env python
"""Researcher example for milestone M3b-1: an independent varying intercept and varying slope.

    y ~ dose + (1 + dose || site), Gaussian: each site has its own intercept and its own dose slope,
    drawn independently (brms's '||'); the correlated '(1 + dose | site)' form is refused in this slice.

    uv run --no-sync python examples/m3b1_varying_slope.py                    # PyMC NUTS on CPU
    uv run --no-sync python examples/m3b1_varying_slope.py --backend numpyro  # JAX/NumPyro on CPU
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

TRUTH = dict(a=2.0, b=0.5, sd_intercept=0.5, sd_slope=0.3, sigma=0.4)


def simulate(J=24, per=15, seed=31):
    rng = np.random.default_rng(seed)
    site = np.repeat([f"s{j:02d}" for j in range(J)], per)
    dose = rng.normal(size=J * per)                                  # raw dose, centred near 0 by design of the study
    u = rng.normal(0, TRUTH["sd_intercept"], J); v = rng.normal(0, TRUTH["sd_slope"], J)
    gi = np.repeat(np.arange(J), per)
    y = TRUTH["a"] + TRUTH["b"] * dose + u[gi] + dose * v[gi] + rng.normal(0, TRUTH["sigma"], J * per)
    return pd.DataFrame({"y": y, "dose": dose, "site": site})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="pymc", choices=["pymc", "numpyro"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--compare", type=Path, default=None)
    args = ap.parse_args()

    print(ab.format_environment(ab.environment())); print()
    df = simulate()
    print(f"{len(df)} rows, {df.site.nunique()} sites, dose varies within every site (min within-site SD {df.groupby('site').dose.std().min():.2f})")
    priors = ab.Priors(
        intercept=ab.Normal(0, 5),                         # a_c: expected y at the mean dose, site effects zero
        b=ab.Normal(0, 2),                                 # population dose slope
        sd={"Intercept": ab.HalfNormal(1),                 # SD of the site intercepts (at dose = 0, raw scale)
            "dose": ab.HalfNormal(1)},                     # SD of the site dose slopes; both required, no defaults
        sigma=ab.HalfNormal(1),
    )
    try:
        ab.gaussian("y ~ dose + (1 + dose | site)", df, priors)
    except ab.FormulaError as e:
        print("the single-bar (correlated) form is refused:", str(e).split(";")[0]); print()
    model = ab.gaussian("y ~ dose + (1 + dose || site)", df, priors)
    print(model.describe()); print()
    print("group design check: Z_slope equals raw dose times the one-hot site matrix ->",
          bool(np.array_equal(model.design.Z_slope, df.dose.to_numpy()[:, None] * model.design.Z)))
    print()
    fit = model.fit(backend=args.backend, device=args.device, chains=args.chains, draws=args.draws, tune=args.tune, seed=args.seed)
    print(fit.report()); print()
    print(format_panel(panel(fit.idata))); print()
    print(az.summary(fit.idata, var_names=["Intercept", "dose", "1|site_sigma", "dose|site_sigma", "sigma"]).to_string())
    print(f"truth: Intercept {TRUTH['a']}, dose {TRUTH['b']}, sd of site intercepts {TRUTH['sd_intercept']}, sd of site slopes "
          f"{TRUTH['sd_slope']}, sigma {TRUTH['sigma']}   (one recovery, not calibration)")
    post = fit.idata.posterior
    corr = np.corrcoef(post["1|site"].mean(("chain", "draw")).to_numpy(), post["dose|site"].mean(("chain", "draw")).to_numpy())[0, 1]
    print(f"correlation of the posterior-mean site intercepts and slopes: {corr:+.2f} (no correlation parameter exists in this model;"
          " a strong value here is a reason to consider the correlated form in a later milestone)")
    print()
    new = pd.DataFrame({"dose": [0.0, 1.0, 0.0, 1.0, 1.0], "site": ["s00", "s00", "new_site", "new_site", "another_new_site"]})
    out = model.predict(fit, newdata=new, kind="mean", new_groups="population", seed=1)
    mu = out["mu"].mean(("chain", "draw")).to_numpy(); sd = out["mu"].std(("chain", "draw")).to_numpy()
    print(f"expected y: site s00 dose 0 -> {mu[0]:.2f} (SD {sd[0]:.2f}), dose 1 -> {mu[1]:.2f} (SD {sd[1]:.2f}); "
          f"a new site (population mode: a fresh intercept AND slope drawn per site per draw): dose 0 -> {mu[2]:.2f} (SD {sd[2]:.2f}), "
          f"dose 1 -> {mu[3]:.2f} (SD {sd[3]:.2f}); another new site at dose 1 -> {mu[4]:.2f}")
    out2 = model.predict(fit, newdata=new, kind="mean", new_groups="resample_observed", seed=1)
    print(f"resample_observed mode (one observed site's intercept-and-slope pair per new site per draw): new site dose 1 -> "
          f"{float(out2['mu'][..., 3].mean()):.2f} (SD {float(out2['mu'][..., 3].std()):.2f})")

    record = ab.fit_record(model, fit)
    if args.record is not None:
        if args.record.exists():
            sys.exit(f"refusing to overwrite {args.record}")
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=1)); print(f"fit record written to {args.record}")
    if args.compare is not None:
        earlier = json.loads(args.compare.read_text()); print(); print(ab.format_comparison(ab.compare_records(record, earlier)))


if __name__ == "__main__":
    main()
