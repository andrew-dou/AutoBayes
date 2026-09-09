#!/usr/bin/env python
"""Researcher example for milestone M1: a binary outcome nested in groups.

    y ~ x + (1 | clinic), Bernoulli likelihood, logit link, explicit logit-scale priors.

    .venv/bin/python examples/m1_bernoulli_varying_intercept.py                    # PyMC NUTS on CPU
    .venv/bin/python examples/m1_bernoulli_varying_intercept.py --backend numpyro  # JAX/NumPyro on CPU
    .venv/bin/python examples/m1_bernoulli_varying_intercept.py --backend numpyro --device gpu   # not yet run on a GPU

The scenario: patients (rows) nested in clinics (groups); ``x`` a standardised risk score;
``y`` whether the outcome occurred. The clinic intercepts absorb between-clinic differences
in the baseline log-odds. ``--record`` writes the fit record; ``--compare`` checks agreement
with an earlier record (same model, same priors) within four combined MCSE.
"""
import argparse
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import autobayes as ab
from autobayes.diagnostics import format_panel, panel

TRUTH = dict(a=-0.5, b=0.8, sd=0.7)


def simulate(n=600, J=30, seed=21):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, J, n)
    x = rng.normal(size=n)
    u = rng.normal(0, TRUTH["sd"], J)
    y = rng.binomial(1, expit(TRUTH["a"] + TRUTH["b"] * x + u[g]))
    return pd.DataFrame({"y": y, "x": x, "clinic": [f"c{k:02d}" for k in g]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="pymc", choices=["pymc", "numpyro"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--record", type=Path, default=None, help="write the fit record (JSON) here")
    ap.add_argument("--compare", type=Path, default=None, help="an earlier fit record to agree with")
    args = ap.parse_args()

    print(ab.format_environment(ab.environment()))
    print()
    df = simulate()
    print(f"outcome rate {df.y.mean():.3f} over n = {len(df)} in {df.clinic.nunique()} clinics")
    priors = ab.Priors(
        intercept=ab.Normal(0, 1.5),      # log-odds of y = 1 at the mean risk score (centred intercept)
        b=ab.Normal(0, 1),                # log-odds ratio per unit of x
        sd=ab.HalfNormal(1),              # SD of the clinic intercepts, logit scale
    )                                     # no sigma: the Bernoulli family has no residual scale
    model = ab.bernoulli("y ~ x + (1 | clinic)", df, priors)
    print(model.describe())
    print()

    fit = model.fit(backend=args.backend, device=args.device, chains=args.chains,
                    draws=args.draws, tune=args.tune, seed=args.seed)
    print(fit.report())
    print()
    print(format_panel(panel(fit.idata)))
    print()
    print(az.summary(fit.idata, var_names=["Intercept", "x", "1|clinic_sigma"]).to_string())
    print(f"truth: Intercept {TRUTH['a']}, x {TRUTH['b']}, sd {TRUTH['sd']}   (one recovery, not calibration)")

    # Probabilities versus outcomes: kind="mean" is P(y = 1); kind="response" adds 0/1 draws.
    ppc = model.predict(fit, kind="response", seed=1)
    print()
    print(f"in-sample: mean predicted probability {float(ppc['p'].mean()):.3f}, "
          f"mean of 0/1 predictive draws {float(ppc['y'].mean()):.3f}, observed rate {df.y.mean():.3f}")
    new = pd.DataFrame({"x": [0.0, 0.0], "clinic": ["c00", "new_clinic"]})
    pop = model.predict(fit, newdata=new, kind="response", new_groups="population", seed=1)
    p0, p1 = pop["p"][..., 0].to_numpy(), pop["p"][..., 1].to_numpy()
    print(f"P(y = 1) at x = 0: observed clinic c00 mean {p0.mean():.3f} (SD {p0.std():.3f}); "
          f"a new clinic (population draw) mean {p1.mean():.3f} (SD {p1.std():.3f})")
    print("the wider SD for the new clinic is the between-clinic variance on the probability scale")

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
