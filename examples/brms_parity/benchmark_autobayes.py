#!/usr/bin/env python
"""The AutoBayes side of the brms benchmark: fit each data set with NumPyro on the CPU and record times, stages and draws.

    python examples/brms_parity/benchmark_autobayes.py <data_dir> <out_dir> [models ...] [--smoke] [--cores c] [--seed s]
        [--no-post]

Same models, priors and sampler settings as benchmark_brms.R: four chains run in parallel, 1000 warm-up and 1000
kept draws each, target acceptance 0.9 (``--smoke``: one chain, 50 and 50, to check the scripts only). Writes
<out_dir>/autobayes_<model>.json with the timings, the stages below, the number of divergent transitions, the draws of
the parameters both libraries report (named as brms names them), and the times of the post-processing functions on the
fit (POST below, each the first call and the median of three more), for compare.py.

Stages, and how each is measured (README.md has the limits):

- build: constructing the model (``build_seconds``, Bambi/PyMC graph), as before.
- compile_estimate: a one-draw fit before the real one (``compile_estimate_seconds``), kept so that the numbers stay
  comparable with the 17 October run, whose "sampling" was the fit minus this.
- The real fit is ``ab.fit`` unchanged, instrumented from outside (``StageProbe``):
  - a ``jax.monitoring`` listener records every interval JAX spends tracing, lowering to MLIR or compiling with XLA;
  - a wrapper around PyMC's ``_sample_numpyro_nuts`` marks the NumPyro run inside the fit and keeps the potential
    function it was handed and the unconstrained draws it returned;
  - ``numpyro.infer.MCMC.run`` is wrapped so that the run also collects the warm-up iterations' extra fields (leapfrog
    steps, step size, divergences); the warm-up part is removed again before PyMC reads the run, so PyMC sees exactly
    the kept draws it always sees. The chains, seeds and compiled sampler are the ones ``ab.fit`` uses.
  From these: ``fit_pre`` (start of the fit to the NumPyro run: PyTensor's graph turned into JAX and the jittered
  initial points), ``sampler_compile`` (start of the run to the end of the largest XLA compilation in it, which is the
  NUTS program: tracing, lowering and compiling it, with ``compile_events_sampler`` the part the listener saw as JAX
  compilation), ``exec`` (the end of that compilation to the end of the run, every chain's warm-up and sampling, with
  ``cpu_exec`` the process's CPU seconds over the same span) and ``fit_post`` (the end of the run to the end of the fit:
  the draws moved to the constrained scale, the InferenceData and its checks).
- gradient: the potential the sampler used (PyMC's jaxified ``-logp`` with the Jacobian), as ``jax.jit(jax.value_and_grad
  (...))``, at the last kept draw of the first chain (unconstrained): its tracing-and-lowering and its XLA compile timed
  on their own (``jit(...).lower(x)`` and ``.compile()``), then the median of GRAD_CALLS host calls after a first one, each
  until the result is ready. Also: the median host call of a jitted function of the same arguments that does no work
  (``dispatch``), and the cost per evaluation inside one compiled loop of GRAD_CALLS evaluations (``in_loop``), each
  point moved by 1e-300 times the previous gradient so that XLA cannot hoist the evaluation out of the loop.
- sampler work: leapfrog steps (NumPyro's ``num_steps``) per chain for warm-up and for sampling, the mean tree depth
  (PyMC's ``floor(log2(num_steps)) + 1``) and the final step size per chain.

Options (the defaults are the 25 October run's): ``--seed s`` the sampler's seed (default 2026); ``--no-post`` skips the
post-processing timings (the scaling runs); ``--cores c`` caps the CPU cores of the whole process, applied in
``_cap_cores`` before numpy, JAX or autobayes is imported (examples/brms_parity/scaling/README.md, "The core cap"):

- on Linux the process's CPU affinity is set to c of the CPUs it is allowed (under SLURM, of its allocation). Every
  thread created afterwards, XLA's thread pools among them, inherits it, so the process cannot use more than c cores at
  once. This is the one enforced part. macOS has no CPU affinity: there the cap is not enforced (recorded as such).
- the thread-pool sizes are set to c: NPROC (read by XLA's CPU client when it sizes its pool, in the XLA versions we
  know of; otherwise the pool is sized by the affinity, also c), OMP_NUM_THREADS, MKL_NUM_THREADS and
  OPENBLAS_NUM_THREADS (numpy's and scipy's BLAS).
- the XLA host device count is the number of chains (AUTOBAYES_HOST_DEVICES, which autobayes passes to NumPyro's
  set_host_device_count, i.e. XLA_FLAGS=--xla_force_host_platform_device_count): one CPU device per chain, the chains
  run in parallel (NumPyro's chain_method="parallel") and share the c cores.

What it cannot control: how XLA spreads the c cores over the chains and within one chain (whether a gradient is split
over threads is XLA's decision per operation; there is no reduce_sum here), and threads that do not belong to this process.
"""
import os
import sys


def _cap_cores(argv: list[str]) -> dict:
    """``--cores c`` (module docstring), applied to this process before any library that starts threads is imported."""
    cores = None
    for i, a in enumerate(argv):
        if a == "--cores" and i + 1 < len(argv):
            cores = int(argv[i + 1])
        elif a.startswith("--cores="):
            cores = int(a.split("=", 1)[1])
    if cores is None:
        return dict(cores=None)
    if cores < 1:
        raise SystemExit("--cores takes a positive integer")
    chains = 1 if "--smoke" in argv else 4                     # SMOKE's and SETTINGS's chains below
    info = dict(cores=cores, host_devices=chains)
    if hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) < cores:
            raise SystemExit(f"--cores {cores}: this process may use only {len(allowed)} CPUs ({allowed})")
        os.sched_setaffinity(0, allowed[:cores])
        info.update(enforced=True, affinity=allowed[:cores], allowed_before=len(allowed))
    else:
        info.update(enforced=False, affinity=None, allowed_before=None)
    for var in ("NPROC", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(cores)
    os.environ["AUTOBAYES_HOST_DEVICES"] = str(chains)
    return info


CORE_CAP = _cap_cores(sys.argv[1:])

import argparse  # noqa: E402  (everything below is imported after the core cap)
import datetime
import json
import platform
import resource
import statistics
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import autobayes as ab  # noqa: E402
import jax  # noqa: E402  (after autobayes, which sets the host device count and float64)

SETTINGS = dict(chains=4, draws=1000, tune=1000, target_accept=0.9, seed=2026)
SMOKE = dict(chains=1, draws=50, tune=50, target_accept=0.9, seed=2026)
GRAD_CALLS = 2000
N01, H1 = ab.Normal(0, 1), ab.HalfNormal(1)


def b1(df):
    pr = ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd={"g": {"Intercept": H1, "x": H1}}, cor={"g": ab.LKJ(2)})
    return ab.bernoulli("y ~ x + (1 + x | g)", df, pr)


def b2(df):
    pr = ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd={"g": {"Intercept": H1, "x": H1}}, cor={"g": ab.LKJ(2)}, sigma=H1)
    return ab.gaussian("y ~ x + z + (1 + x | g)", df, pr)


def b3(df):
    return ab.cumulative("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1))


def b4(df):
    return ab.zero_inflated_poisson("y ~ x + (1 | g)", df,
                                    ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1, zi=ab.Beta(2, 2)))


def b5(df):
    return ab.weibull("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1, shape=ab.Gamma(2, 0.5)))


def b6(df):
    return ab.zero_inflated_negbinomial("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1,
                                                                         shape=ab.Gamma(2, 0.5), zi=ab.Beta(2, 2)))


def b7(df):
    return ab.sratio("y ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1))


def b8(df):
    formula = "y ~ " + " + ".join(f"x{j}" for j in range(1, 21))
    return ab.gaussian(formula, df, ab.Priors(intercept=ab.Normal(0, 2.5), b=ab.Horseshoe(), sigma=H1))


def b9(df):
    return ab.beta_binomial("k | trials(size) ~ x + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1,
                                                                             phi=ab.Gamma(2, 0.1)))


def b10(df):
    return ab.gaussian("y ~ x + (1 | mm(g1, g2))", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1, sigma=H1))


def b11(df):
    comp = [ab.Priors(intercept=ab.Normal(-1, 2), b=N01, sigma=H1), ab.Priors(intercept=ab.Normal(1, 2), b=N01, sigma=H1)]
    return ab.mixture("y ~ x", df, ab.Priors(components=comp, theta=ab.Dirichlet(2)), families=["gaussian", "gaussian"])


def b12(df):
    return ab.gaussian("y ~ gp(x)", df, ab.Priors(intercept=ab.Normal(0, 2.5), sigma=H1, sdgp=H1, lscale=ab.InverseGamma(3, 1)))


def b13(df):
    return ab.gaussian("y ~ x + ar(time = t, gr = g, p = 1)", df,
                       ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sigma=H1, ar=ab.Normal(0, 0.5)))


def b14(df):
    pr = ab.Priors(nlpars={"a": {"b": ab.Normal(2, 1), "sd": H1}, "b": {"b": N01}}, sigma=H1)
    return ab.gaussian(["y ~ a * exp(b * x)", "a ~ 1 + (1 | g)", "b ~ 1"], df, pr, nl=True)


def b15(df):
    resp = {"intercept": ab.Normal(0, 2.5), "b": N01, "sd": H1, "sigma": H1}
    return ab.gaussian("mvbind(y1, y2) ~ x + (1 | g)", df, ab.Priors(resp={"y1": resp, "y2": resp}, rescor=ab.LKJ(2)),
                       rescor=True)


def b16(df):
    return ab.gaussian("y ~ x + mo(edu) + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1, sigma=H1,
                                                                   simo={"edu": ab.Dirichlet(2, 2, 2, 2)}))


def b17(df):
    return ab.weibull("y | cens(cens_r) ~ x + (1 | g)", df,
                      ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1, shape=ab.Gamma(2, 0.5)))


def b18(df):
    pr = ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1, sigma={"intercept": N01, "b": ab.Normal(0, 0.5)})
    return ab.gaussian(["y ~ x + (1 | g)", "sigma ~ z"], df, pr)


BUILD = {"B1_logistic_glmm": b1, "B2_gaussian_glmm": b2, "B3_ordinal_glmm": b3, "B4_zip_glmm": b4,
         "B5_weibull_glmm": b5, "B6_zinb_glmm": b6, "B7_sratio_glmm": b7, "B8_horseshoe_regression": b8,
         "B9_betabinomial_glmm": b9, "B10_multimembership": b10,
         "B11_gauss_mixture": b11, "B12_gp_exact": b12, "B13_ar1_panel": b13, "B14_nonlinear_growth": b14,
         "B15_mv_rescor": b15, "B16_monotonic": b16, "B17_weibull_censored": b17, "B18_distributional_sigma": b18}


# The scaling models (benchmark_data.py --scaling N), run only when named; priors as benchmark_brms.R's.
def s1(df):
    pr = ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd={"g": {"Intercept": H1, "x1": H1}}, cor={"g": ab.LKJ(2)})
    return ab.bernoulli("y ~ x1 + x2 + (1 + x1 | g)", df, pr)


def s2(df):
    pr = ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd={"g": {"Intercept": H1, "x1": H1, "x2": H1}}, cor={"g": ab.LKJ(2)},
                   sigma=H1)
    return ab.gaussian("y ~ x1 + x2 + (1 + x1 + x2 | g)", df, pr)


def s3(df):
    sd = {"g1": {"Intercept": H1}, "g2": {"Intercept": H1}}
    return ab.poisson("y ~ x1 + (1 | g1) + (1 | g2)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=sd))


def s4(df):
    return ab.cumulative("y ~ x1 + x2 + (1 | g)", df, ab.Priors(intercept=ab.Normal(0, 2.5), b=N01, sd=H1))


def s5(df):
    return ab.gaussian("y ~ gp(x, k = 30)", df, ab.Priors(intercept=ab.Normal(0, 2.5), sigma=H1, sdgp=H1,
                                                          lscale=ab.InverseGamma(3, 1)))


SCALING = {"S1_logistic_glmm": s1, "S2_gaussian_slopes": s2, "S3_poisson_crossed": s3, "S4_ordinal_glmm": s4,
           "S5_gp_approx": s5}
# The post-processing functions timed on each fit, as brms's: posterior_epred(), log_lik(), loo(), waic(), bayes_R2(),
# hypothesis(fit, "x > 0") and conditional_effects(fit, "x"), with the hypothesis and the effect below where a model has
# no coefficient x (the same in benchmark_brms.R). A function that refuses the model is recorded as such.
HYPOTHESIS = {"B8_horseshoe_regression": "x1 > 0", "B11_gauss_mixture": "mu1_x > 0", "B12_gp_exact": "Intercept > 0",
              "B14_nonlinear_growth": "a_Intercept > 0", "B15_mv_rescor": "y1_x > 0"}
EFFECT = {"B8_horseshoe_regression": "x1"}
POST = {
    "posterior_epred": lambda m, f, name: m.predict(f, kind="mean"),
    "log_lik": lambda m, f, name: ab.log_lik(m, f),
    "loo": lambda m, f, name: ab.loo(m, f),
    "waic": lambda m, f, name: ab.waic(m, f),
    "bayes_R2": lambda m, f, name: ab.bayes_R2(m, f),
    "hypothesis": lambda m, f, name: ab.hypothesis(m, f, HYPOTHESIS.get(name, "x > 0")),
    "conditional_effects": lambda m, f, name: ab.conditional_effects(m, f, [EFFECT.get(name, "x")]),
}


def time_post(model, fit, model_name) -> dict:
    out = {}
    for name, call in POST.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.perf_counter()
                call(model, fit, model_name)
                first = time.perf_counter() - t0
                again = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    call(model, fit, model_name)
                    again.append(time.perf_counter() - t0)
            out[name] = dict(first=first, median=statistics.median(again))
        except Exception as e:                                 # a refusal: recorded, not timed
            out[name] = dict(refused=f"{type(e).__name__}: {str(e)[:160]}")
    return out


def draws_by_brms_name(model, post) -> dict[str, list[float]]:
    """The draws (chain-major, flattened) of the parameters brms also reports, under brms's names (B1-B10)."""
    out = {}

    def put(name, arr):
        out[name] = np.asarray(arr, dtype=float).reshape(-1).tolist()

    names = model.names
    if "intercept" in names and names["intercept"] in post:
        v = post[names["intercept"]]
        if v.ndim == 2:
            put("b_Intercept", v)
        else:                                                  # ordered thresholds: b_Intercept[1], ...
            for j in range(v.shape[-1]):
                put(f"b_Intercept[{j + 1}]", v.isel({v.dims[-1]: j}))
    for key, var in names.items():                             # every population-level coefficient, b_<column>
        if key.startswith("b:") and var in post and post[var].ndim == 2:
            put(f"b_{key[2:]}", post[var])
    for var in post.data_vars:                                 # group-level SDs: 1|g_sigma -> sd_g__Intercept
        if var.endswith("_sigma") and "|" in var:
            coef, group = var[:-len("_sigma")].split("|", 1)
            put(f"sd_{group}__{'Intercept' if coef == '1' else coef}", post[var])
    if "cor|g" in post:
        c = post["cor|g"]
        put("cor_g__Intercept__x", c.isel({c.dims[2]: 0, c.dims[3]: 1}))
    for par in ("sigma", "zi", "shape", "phi"):
        if isinstance(names.get(par), str) and names[par] in post:
            put(par, post[names[par]])
    return out


# B11-B18: (brms's name, the AutoBayes posterior variable, the index into its axes after chain and draw, or None)
RENAME = {
    "B11_gauss_mixture": [("b_mu1_Intercept", "Intercept", (0,)), ("b_mu2_Intercept", "Intercept", (1,)),
                          ("b_mu1_x", "x", (0,)), ("b_mu2_x", "x", (1,)), ("sigma1", "sigma1", None),
                          ("sigma2", "sigma2", None), ("theta1", "theta", (0,)), ("theta2", "theta", (1,))],
    "B12_gp_exact": [("b_Intercept", "Intercept", None), ("sdgp_gpx", "sdgp_x", None), ("lscale_gpx", "lscale_x", None),
                     ("sigma", "sigma", None)],
    "B13_ar1_panel": [("b_Intercept", "Intercept", None), ("b_x", "x", None), ("ar[1]", "ar", (0,)), ("sigma", "sigma", None)],
    "B14_nonlinear_growth": [("b_a_Intercept", "a_Intercept", None), ("b_b_Intercept", "b_Intercept", None),
                             ("sd_g__a_Intercept", "a_1|g_sigma", None), ("sigma", "sigma", None)],
    "B15_mv_rescor": [(f"b_{r}_Intercept", f"{r}_Intercept", None) for r in ("y1", "y2")]
                     + [(f"b_{r}_x", f"{r}_x", None) for r in ("y1", "y2")]
                     + [(f"sd_g__{r}_Intercept", f"{r}_1|g_sigma", None) for r in ("y1", "y2")]
                     + [(f"sigma_{r}", f"{r}_sigma", None) for r in ("y1", "y2")]
                     + [("rescor__y1__y2", "rescor", (0, 1))],
    "B16_monotonic": [("b_Intercept", "Intercept", None), ("b_x", "x", None), ("bsp_moedu", "mo(edu)", None)]
                     + [(f"simo_moedu1[{j + 1}]", "simo_edu", (j,)) for j in range(4)]
                     + [("sd_g__Intercept", "1|g_sigma", None), ("sigma", "sigma", None)],
    "B17_weibull_censored": [("b_Intercept", "Intercept", None), ("b_x", "x", None), ("sd_g__Intercept", "1|g_sigma", None),
                             ("shape", "alpha", None)],
    "B18_distributional_sigma": [("b_Intercept", "Intercept", None), ("b_x", "x", None),
                                 ("sd_g__Intercept", "1|g_sigma", None), ("b_sigma_Intercept", "sigma_Intercept", None),
                                 ("b_sigma_z", "sigma_z", None)],
    "S5_gp_approx": [("b_Intercept", "Intercept", None), ("sdgp_gpx", "sdgp_x", None), ("lscale_gpx", "lscale_x", None),
                     ("sigma", "sigma", None)],
}


def correlations_by_brms_name(post) -> dict[str, list[float]]:
    """Every correlation of every correlated group term (the scaling models), brms's cor_<g>__<c1>__<c2>: the
    correlation matrix ``cor|<g>``'s upper triangle, its coefficients named by its coordinate labels."""
    out = {}
    for var in post.data_vars:
        if not var.startswith("cor|"):
            continue
        c = post[var]
        labels = ["Intercept" if str(v) in ("1", "Intercept") else str(v) for v in c.coords[c.dims[2]].values]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                out[f"cor_{var[4:]}__{labels[i]}__{labels[j]}"] = (
                    np.asarray(c.isel({c.dims[2]: i, c.dims[3]: j}), dtype=float).reshape(-1).tolist())
    return out


def draws_renamed(name, post) -> tuple[dict[str, list[float]], list[str]]:
    """The draws of a B11-B18 model under brms's names, and the AutoBayes variables that were not in the posterior."""
    out, missing = {}, []
    for brms_name, var, index in RENAME[name]:
        if var not in post:
            missing.append(var)
            continue
        v = np.asarray(post[var])
        if index is not None:
            v = v[(slice(None), slice(None)) + index]
        if v.ndim != 2:
            missing.append(f"{var} (not a scalar per draw)")
            continue
        out[brms_name] = v.astype(float).reshape(-1).tolist()
    return out, missing


def _cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _union(intervals, lo=-np.inf, hi=np.inf) -> float:
    """Total length of the union of (start, end) intervals clipped to [lo, hi] (nested JAX events count once)."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted((max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0.0)


class StageProbe:
    """Instruments ab.fit() from outside without changing what it computes (see the module docstring)."""

    COMPILE_EVENTS = ("/jax/core/compile/jaxpr_trace_duration", "/jax/core/compile/jaxpr_to_mlir_module_duration",
                      "/jax/core/compile/backend_compile_duration")
    BACKEND_COMPILE = "/jax/core/compile/backend_compile_duration"

    def __init__(self):
        import pymc.sampling.jax as pymc_jax
        from numpyro.infer import MCMC

        self.events: list[tuple[float, float, float, str]] = []      # (start, end, process CPU at end, event)
        self.runs: list[dict] = []
        jax.monitoring.register_event_duration_secs_listener(self._on_event)
        orig_sample, orig_run = pymc_jax._sample_numpyro_nuts, MCMC.run
        probe = self

        def sample(*args, **kwargs):                           # PyMC's NumPyro run, marked and kept
            run = dict(t0=time.perf_counter(), cpu0=_cpu(), open=True)
            probe.runs.append(run)
            try:
                out = orig_sample(*args, **kwargs)
                jax.block_until_ready(out[:2])
            finally:
                run.update(t1=time.perf_counter(), cpu1=_cpu(), open=False)
            # the last kept draw of the first chain, copied now: PyMC donates the draws' buffers when it transforms them
            run.update(logp_fn=kwargs.get("logp_fn"), point=[np.array(a[0, -1]) for a in out[0]])
            return out

        def mcmc_run(mcmc, rng_key, *args, **kwargs):          # also collect the warm-up iterations, then drop them
            run = probe.runs[-1] if probe.runs and probe.runs[-1]["open"] else None
            if run is None or mcmc._warmup_state is not None or mcmc.thinning != 1 or mcmc.num_warmup == 0:
                return orig_run(mcmc, rng_key, *args, **kwargs)
            w = mcmc.num_warmup
            total = w + mcmc.num_samples
            mcmc._set_collection_params(0, total, total)
            run["mcmc_t0"] = time.perf_counter()
            orig_run(mcmc, rng_key, *args, **kwargs)
            jax.block_until_ready(mcmc._states)
            run["exec_t1"], run["exec_cpu1"] = time.perf_counter(), _cpu()
            fields = {k: np.asarray(v) for k, v in mcmc._states.items() if k != mcmc._sample_field}
            run["warmup_fields"] = {k: v[:, :w] for k, v in fields.items()}
            run["sampling_fields"] = {k: v[:, w:] for k, v in fields.items()}
            mcmc._states = jax.tree.map(lambda x: x[:, w:], mcmc._states)
            mcmc._states_flat = None
            return None

        pymc_jax._sample_numpyro_nuts = sample
        MCMC.run = mcmc_run

    def _on_event(self, event, duration, **kwargs):
        if event in self.COMPILE_EVENTS:
            end = time.perf_counter()
            self.events.append((end - duration, end, _cpu(), event))

    def reset(self):
        self.events.clear()
        self.runs.clear()

    def stages(self, fit_t0: float, fit_t1: float) -> dict:
        """The stages of the one probed fit between fit_t0 and fit_t1."""
        if len(self.runs) != 1 or "exec_t1" not in self.runs[0]:
            raise RuntimeError(f"expected one probed NumPyro run in the fit, found {len(self.runs)}")
        run = self.runs[0]
        compiles = [e for e in self.events if e[3] == self.BACKEND_COMPILE and run["t0"] <= e[1] <= run["exec_t1"]]
        main = max(compiles, key=lambda e: e[1] - e[0])        # the NUTS program: the largest XLA compilation in the run
        exec_t0, exec_cpu0 = main[1], main[2]
        spans = [(s, e) for s, e, _, _ in self.events]
        warm, samp = run["warmup_fields"], run["sampling_fields"]
        depth = lambda n: np.floor(np.log2(np.maximum(n, 1))) + 1      # noqa: E731  (PyMC's tree depth from num_steps)
        return dict(
            fit_pre=run["t0"] - fit_t0, sampler_compile=exec_t0 - run["t0"], exec=run["exec_t1"] - exec_t0,
            fit_post=fit_t1 - run["exec_t1"], cpu_exec=run["exec_cpu1"] - exec_cpu0,
            nuts_compile_xla=main[1] - main[0],
            compile_events_sampler=_union(spans, run["t0"], exec_t0), compile_events_fit=_union(spans, fit_t0, fit_t1),
            steps_warmup=warm["num_steps"].sum(axis=1).astype(int).tolist(),
            steps_sampling=samp["num_steps"].sum(axis=1).astype(int).tolist(),
            tree_depth_warmup=depth(warm["num_steps"]).mean(axis=1).tolist(),
            tree_depth_sampling=depth(samp["num_steps"]).mean(axis=1).tolist(),
            step_size_final=samp["adapt_state.step_size"][:, -1].astype(float).tolist(),
            divergent_warmup=warm["diverging"].sum(axis=1).astype(int).tolist(),
            divergent_sampling=samp["diverging"].sum(axis=1).astype(int).tolist())


def _median_us(times) -> dict:
    t = np.asarray(times) * 1e6
    return dict(median_us=float(np.median(t)), p10_us=float(np.quantile(t, 0.1)), p90_us=float(np.quantile(t, 0.9)))


def gradient_cost(logp_fn, point, calls: int = GRAD_CALLS) -> dict:
    """The cost of one potential-and-gradient evaluation at the last kept draw of the first chain (module docstring)."""
    import jax.numpy as jnp

    point = [jnp.asarray(a) for a in point]
    vg = jax.value_and_grad(logp_fn)
    t0 = time.perf_counter()
    lowered = jax.jit(vg).lower(point)
    t1 = time.perf_counter()
    lowered.compile()
    t2 = time.perf_counter()

    def host_calls(f):
        jax.block_until_ready(f(point))                        # the first call (and the jit cache's own compile)
        out = []
        for _ in range(calls):
            s = time.perf_counter()
            jax.block_until_ready(f(point))
            out.append(time.perf_counter() - s)
        return out

    f = jax.jit(vg)
    value = float(f(point)[0])
    grad = host_calls(f)
    dispatch = host_calls(jax.jit(lambda xs: xs[0].ravel()[:1]))

    def loop(xs):
        def body(i, carry):
            xs, acc = carry
            v, g = vg(xs)
            return jax.tree.map(lambda x, gx: x + 1e-300 * gx, xs, g), acc + v
        return jax.lax.fori_loop(0, calls, body, (xs, jnp.zeros(())))[1]

    looped = jax.jit(loop)
    jax.block_until_ready(looped(point))
    reps = []
    for _ in range(3):
        s = time.perf_counter()
        jax.block_until_ready(looped(point))
        reps.append(time.perf_counter() - s)
    return dict(point="last kept draw of chain 1, unconstrained", calls=calls, potential=value,
                n_unconstrained=int(sum(np.size(p) for p in point)), lower_seconds=t1 - t0, compile_seconds=t2 - t1,
                host=_median_us(grad), dispatch=_median_us(dispatch), in_loop_us=float(np.median(reps)) / calls * 1e6)


def versions() -> dict:
    import subprocess

    import numpyro
    import pymc
    try:                                                       # the commit the benchmark ran on
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        rev = "?"
    return dict(autobayes=getattr(ab, "__version__", "?"), commit=rev, jax=jax.__version__, numpyro=numpyro.__version__,
                pymc=pymc.__version__, python=platform.python_version(), machine=platform.platform(),
                cpu_count=__import__("os").cpu_count())


def machine() -> dict:
    """Where this process may run, for the record (Linux: the CPUs it is allowed, after the core cap; SLURM's job)."""
    def first(path, key):
        try:
            return next(line.split(":", 1)[1].strip() for line in open(path) if line.startswith(key))
        except (OSError, StopIteration):
            return None
    return dict(host=platform.node(), cpus_allowed=first("/proc/self/status", "Cpus_allowed_list"),
                n_allowed=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                cpu_model=first("/proc/cpuinfo", "model name"), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                slurm_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"), xla_flags=os.environ.get("XLA_FLAGS"),
                jax_cpu_devices=len(jax.devices("cpu")))


def run_model(name, build, data_dir, settings, smoke, probe, post=True) -> dict:
    """Build, fit and time one model; its record for compare.py."""
    df = pd.read_csv(Path(data_dir) / f"{name}.csv", dtype={"g": str, "g1": str, "g2": str})
    t0 = time.perf_counter()
    model = build(df)
    t_build = time.perf_counter() - t0
    warm = ab.fit(model, backend="numpyro", chains=settings["chains"], draws=1, tune=1, seed=1)   # as on 17 October
    probe.reset()
    fit_t0 = time.perf_counter()
    fit = ab.fit(model, backend="numpyro", **settings)
    fit_t1 = time.perf_counter()
    stages = probe.stages(fit_t0, fit_t1)
    stages["gradient"] = gradient_cost(probe.runs[0]["logp_fn"], probe.runs[0]["point"])
    divergent = int(np.asarray(fit.idata.sample_stats["diverging"]).sum()) if "diverging" in fit.idata.sample_stats else None
    if name in RENAME:
        draws, missing = draws_renamed(name, fit.idata.posterior)
    else:
        draws, missing = draws_by_brms_name(model, fit.idata.posterior), []
    if name in SCALING:                                        # every correlation, not only B1's and B2's one
        draws.pop("cor_g__Intercept__x", None)
        draws.update(correlations_by_brms_name(fit.idata.posterior))
    return dict(model=name, n=len(df), library="autobayes", backend="numpyro", device=fit.device_actual,
                chain_method=fit.chain_method, date=datetime.date.today().isoformat(), smoke=smoke, settings=settings,
                chains=settings["chains"], options=dict(seed=settings["seed"], post=post, cores=CORE_CAP["cores"]),
                cores=CORE_CAP["cores"], core_cap=CORE_CAP, machine=machine(), versions=versions(), build_seconds=t_build,
                compile_estimate_seconds=warm.wall_seconds, fit_seconds=fit.wall_seconds, stages=stages, divergent=divergent,
                post=time_post(model, fit, name) if post else {}, draws=draws, missing_draws=missing)


def main(data_dir: str, out_dir: str, only=None, smoke: bool = False, seed: int = 2026, post: bool = True) -> None:
    settings = dict(SMOKE if smoke else SETTINGS, seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    unknown = set(only or ()) - set(BUILD) - set(SCALING)
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")
    probe = StageProbe()
    models = {k: v for k, v in {**BUILD, **SCALING}.items() if k in only} if only else BUILD
    for name, build in models.items():
        try:
            rec = run_model(name, build, data_dir, settings, smoke, probe, post)
        except Exception:                                      # one failing model does not stop the run
            traceback.print_exc()
            print(f"{name}: FAILED (traceback above)", flush=True)
            continue
        (out / f"autobayes_{name}.json").write_text(json.dumps(rec))
        st, g = rec["stages"], rec["stages"]["gradient"]
        print(f"{name}: build {rec['build_seconds']:.1f} s, compile ~{rec['compile_estimate_seconds']:.1f} s, fit "
              f"{rec['fit_seconds']:.1f} s (sampler compile {st['sampler_compile']:.1f}, exec {st['exec']:.1f}), steps "
              f"{sum(st['steps_warmup'])}+{sum(st['steps_sampling'])}, gradient {g['host']['median_us']:.1f} us (in loop "
              f"{g['in_loop_us']:.1f}), {rec['divergent']} divergent"
              + (f"; missing draws {rec['missing_draws']}" if rec["missing_draws"] else ""), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("models", nargs="*")
    ap.add_argument("--smoke", action="store_true", help="one chain, 50 warm-up and 50 kept draws: checks the scripts only")
    ap.add_argument("--cores", type=int, help="cap the process's CPU cores (applied before the imports; module docstring)")
    ap.add_argument("--seed", type=int, default=2026, help="the sampler's seed (default 2026)")
    ap.add_argument("--no-post", dest="post", action="store_false", help="skip the post-processing timings")
    a = ap.parse_args()
    main(a.data_dir, a.out_dir, set(a.models) or None, a.smoke, a.seed, a.post)
