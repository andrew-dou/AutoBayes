#!/usr/bin/env python
"""Researcher example for milestone M3a-2: a numeric-by-categorical interaction.

    y ~ dose + arm + dose:arm + (1 | site), Gaussian; arm in {control, low, high}, reference "control".

    uv run --no-sync python examples/m3a2_interaction.py                    # PyMC NUTS on CPU
    uv run --no-sync python examples/m3a2_interaction.py --backend numpyro  # JAX/NumPyro on CPU

Each arm has its own intercept and its own dose slope: the control arm's are ``Intercept`` (raw) and
``dose``; the others are expressed as differences ``arm[l]`` and ``dose:arm[l]``. The arms have
different mean doses on purpose, so the centred product columns' means are not the product of the
means; ``describe()`` prints the actual means used.
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
from autobayes.session import compare_records, environment, fit_record, format_comparison, format_environment

TRUTH = dict(a=2.0, b=0.5, low=-0.6, high=0.9, dlow=0.3, dhigh=-0.8, sd=0.5, sigma=0.4)
DOSE_MEAN = {"control": 0.0, "low": 1.0, "high": -0.5}


def simulate(rows=300, J=12, seed=23):
    rng = np.random.default_rng(seed)
    g = np.array([i % J for i in range(rows)]); rng.shuffle(g)
    arm = rng.choice(["control", "low", "high"], size=rows, p=[0.5, 0.3, 0.2])
    dose = rng.normal(size=rows) + np.array([DOSE_MEAN[a] for a in arm])          # different mean dose per arm
    u = rng.normal(0, TRUTH["sd"], J)
    inter = np.select([arm == "low", arm == "high"], [TRUTH["low"], TRUTH["high"]], 0.0)
    slope = TRUTH["b"] + np.select([arm == "low", arm == "high"], [TRUTH["dlow"], TRUTH["dhigh"]], 0.0)
    y = TRUTH["a"] + inter + slope * dose + u[g] + rng.normal(0, TRUTH["sigma"], rows)
    df = pd.DataFrame({"y": y, "dose": dose, "arm": arm, "site": [f"s{k:02d}" for k in g]})
    df.attrs["mean_site_effect"] = float(u.mean())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="pymc", choices=["pymc", "numpyro"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--compare", type=Path, default=None)
    args = ap.parse_args()

    print(format_environment(environment()))
    print()
    df = simulate()
    print(f"{len(df)} rows, {df.site.nunique()} sites; arms {df.arm.value_counts().to_dict()}; "
          f"mean dose by arm {df.groupby('arm').dose.mean().round(2).to_dict()}")
    priors = ab.Priors(
        intercept=ab.Normal(0, 5),                  # a_c: expected y at the means of every expanded design column
        b={"dose": ab.Normal(0, 2),                 # control arm's dose slope
           "arm": ab.Normal(0, 2),                  # arm minus control at dose = 0 (one prior for both differences)
           "dose:arm": {"low": ab.Normal(0, 1),     # slope differences, per level ...
                        "high": ab.Normal(0, 1)}},  # ... or "dose:arm": ab.Normal(0, 1) for both
        sd=ab.HalfNormal(1), sigma=ab.HalfNormal(1),
    )
    model = ab.gaussian("y ~ dose + arm + dose:arm + (1 | site)", df, priors, categorical={"arm": "control"})
    print(model.describe())
    print()
    print("design columns:", model.spec.columns)
    print("coefficients   :", model.coefficients)
    print("column means used for centring:", dict(zip(model.spec.columns, np.round(model.design.x_means, 4))))
    print()

    fit = model.fit(backend=args.backend, device=args.device, chains=args.chains, draws=args.draws, tune=args.tune, seed=args.seed)
    print(fit.report()); print()
    print(format_panel(panel(fit.idata))); print()
    print(az.summary(fit.idata, var_names=["Intercept", "dose", "arm", "dose:arm", "1|site_sigma", "sigma"]).to_string())
    ubar = df.attrs["mean_site_effect"]
    print(f"truth: Intercept {TRUTH['a']} (raw: dose = 0, control), dose {TRUTH['b']}, arm[low] {TRUTH['low']}, arm[high] {TRUTH['high']}, "
          f"dose:arm[low] {TRUTH['dlow']}, dose:arm[high] {TRUTH['dhigh']}, sd {TRUTH['sd']}, sigma {TRUTH['sigma']}   (one recovery, not calibration)")
    print(f"the realised mean of the site effects is {ubar:+.3f}; compare Intercept with {TRUTH['a'] + ubar:.3f}")
    post = fit.idata.posterior
    for arm in ["control", "low", "high"]:
        slope = post["dose"] + (post["dose:arm"].sel({"dose:arm_dim": arm}) if arm != "control" else 0)
        print(f"  dose slope in arm {arm:<7}: mean {float(slope.mean()):+.3f}, 94% HDI [{float(slope.quantile(0.03)):+.3f}, {float(slope.quantile(0.97)):+.3f}]")
    print()
    new = pd.DataFrame({"dose": [0.0, 1.0, 0.0, 1.0, 0.0], "arm": ["high", "high", "control", "control", "low"], "site": ["s00", "s00", "s00", "s00", "new_site"]})
    out = model.predict(fit, newdata=new, kind="mean", new_groups="population", seed=1)
    mu = out["mu"].mean(("chain", "draw")).to_numpy()
    print(f"expected y at site s00: high arm dose 0 -> {mu[0]:.2f}, dose 1 -> {mu[1]:.2f}; control dose 0 -> {mu[2]:.2f}, dose 1 -> {mu[3]:.2f}; "
          f"a new site, low arm, dose 0 -> {mu[4]:.2f} (SD across draws {float(out['mu'][..., 4].std()):.2f})")
    try:
        model.predict(fit, newdata=new.assign(arm=["placebo", "high", "control", "control", "low"]), kind="mean", new_groups="population")
    except ab.FormulaError as e:
        print("an arm that was not fitted is refused:", str(e).split(";")[0])

    record = fit_record(model, fit)
    if args.record is not None:
        if args.record.exists():
            sys.exit(f"refusing to overwrite {args.record}")
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=1))
        print(f"fit record written to {args.record}")
    if args.compare is not None:
        earlier = json.loads(args.compare.read_text())
        print(); print(format_comparison(compare_records(record, earlier)))


if __name__ == "__main__":
    main()
