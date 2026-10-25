#!/usr/bin/env python
"""Researcher example for milestone M3a-1: one categorical predictor with a declared reference.

    y ~ x + treatment + (1 | site), Gaussian, treatment in {control, low, high}, reference "control".

    uv run --no-sync python examples/m3a1_categorical.py                    # PyMC NUTS on CPU
    uv run --no-sync python examples/m3a1_categorical.py --backend numpyro  # JAX/NumPyro on CPU

The scenario: sites (rows nested in sites) receive one of three treatment arms; ``x`` is a
standardised site-level covariate. The arm is declared categorical with an explicit reference,
so each coefficient is "this arm minus control" on the response scale at the same x and site.
``--record`` writes the fit record; ``--compare`` checks agreement with an earlier record.
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

TRUTH = dict(a=2.0, b=0.5, low=-0.6, high=0.9, sd=0.5, sigma=0.4)


def simulate(rows=240, J=12, seed=21):
    rng = np.random.default_rng(seed)
    g = np.array([i % J for i in range(rows)]); rng.shuffle(g)                    # every site present
    arm = rng.choice(["control", "low", "high"], size=rows, p=[0.5, 0.3, 0.2])    # unequal arm sizes
    x = rng.normal(size=rows)
    u = rng.normal(0, TRUTH["sd"], J)
    effect = np.select([arm == "low", arm == "high"], [TRUTH["low"], TRUTH["high"]], 0.0)
    y = TRUTH["a"] + TRUTH["b"] * x + effect + u[g] + rng.normal(0, TRUTH["sigma"], rows)
    df = pd.DataFrame({"y": y, "x": x, "treatment": arm, "site": [f"s{k:02d}" for k in g]})
    df.attrs["mean_site_effect"] = float(u.mean())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="pymc", choices=["pymc", "numpyro"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--record", type=Path, default=None, help="write the fit record (JSON) here")
    ap.add_argument("--compare", type=Path, default=None, help="an earlier fit record to agree with")
    args = ap.parse_args()

    print(format_environment(environment()))
    print()
    df = simulate()
    print(f"{len(df)} rows, {df.site.nunique()} sites; arms: {df.treatment.value_counts().to_dict()}")
    priors = ab.Priors(
        intercept=ab.Normal(0, 5),                       # a_c: expected y at mean x and the sample arm proportions
        b={"x": ab.Normal(0, 2),                         # per unit of x
           "treatment": {"low": ab.Normal(0, 2),         # low minus control, response scale (per-level form) ...
                         "high": ab.Normal(0, 2)}},      # ... or one prior for the whole term: "treatment": ab.Normal(0, 2)
        sd=ab.HalfNormal(1),
        sigma=ab.HalfNormal(1),
    )
    model = ab.gaussian("y ~ x + treatment + (1 | site)", df, priors, categorical={"treatment": "control"})
    print(model.describe())                              # reference, levels, proportions, equation, priors, naming
    print()
    print("design columns:", model.spec.columns, "| coefficients:", model.coefficients)
    print("first rows of the raw design (x, low, high):")
    print(np.column_stack([df.treatment[:5], np.round(model.design.X_raw[:5], 3)]))
    print()

    fit = model.fit(backend=args.backend, device=args.device, chains=args.chains,
                    draws=args.draws, tune=args.tune, seed=args.seed)
    print(fit.report())
    print()
    print(format_panel(panel(fit.idata)))
    print()
    print(az.summary(fit.idata, var_names=["Intercept", "x", "treatment", "1|site_sigma", "sigma"]).to_string())
    ubar = df.attrs["mean_site_effect"]
    print(f"truth: Intercept {TRUTH['a']} (raw: x = 0, control), x {TRUTH['b']}, treatment[low] {TRUTH['low']}, "
          f"treatment[high] {TRUTH['high']}, sd {TRUTH['sd']}, sigma {TRUTH['sigma']}   (one recovery, not calibration)")
    print(f"the realised mean of the simulated site effects is {ubar:+.3f}; in one data set the intercept is "
          f"identified relative to it, so compare Intercept with {TRUTH['a'] + ubar:.3f}")
    print()

    # Predictions per arm at x = 0 for one observed site and for a new site; an unseen arm is refused.
    new = pd.DataFrame({"x": [0.0] * 4, "treatment": ["high", "control", "low", "control"], "site": ["s00", "s00", "s00", "new_site"]})
    out = model.predict(fit, newdata=new, kind="mean", new_groups="population", seed=1)
    mu = out["mu"].mean(("chain", "draw")).to_numpy()
    print(f"expected y at x = 0, site s00: high {mu[0]:.2f}, control {mu[1]:.2f}, low {mu[2]:.2f}; "
          f"a new site, control: {mu[3]:.2f} (SD across draws {float(out['mu'][..., 3].std()):.2f})")
    try:
        model.predict(fit, newdata=new.assign(treatment=["placebo", "control", "low", "control"]), kind="mean", new_groups="population")
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
        print()
        print(format_comparison(compare_records(record, earlier)))


if __name__ == "__main__":
    main()
