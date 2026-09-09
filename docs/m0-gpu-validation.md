# M0 GPU validation: pending

Status (2026-09-08): **not validated on a GPU.** The development machine (Apple M1 Pro)
has no CUDA device; JAX sees one CPU device. What exists is *code support*: the
NumPyro path resolves `device="gpu"` against `jax.devices()`, refuses with a clear error
when no GPU is visible (`docs/m0-evidence/example_numpyro_gpu_attempt.txt` is that
refusal, captured), and records the actual device, dtype and chain method in
`FitResult.report()`.

What has been executed: the same NumPyro path on the CPU (`example_numpyro_cpu.txt`,
`B2_recovery.json`), with the JAX-compiled log density checked against the PyTensor
path at fixed points (test A4). The GPU run exercises the same compiled graph on a
different device; it has not been observed.

## Reproducible command

On any machine with an NVIDIA GPU and a CUDA-enabled `jaxlib`:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[dev]" "jax[cuda12]" numpyro "matplotlib<3.11"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python examples/m0_varying_intercept.py --backend numpyro --device gpu --chains 4
```

Expected evidence: the fit report line `device actual : gpu:<name>(id=0)` (or `cuda:`),
`dtype : float64`, `chain method : vectorized`, a diagnostic panel with every row at
`pass`, and posterior means for `Intercept`, `x`, `1|school_sigma`, `sigma` within four
combined MCSE of the CPU values in `docs/m0-evidence/B2_recovery.json`. Save the output
as `docs/m0-evidence/example_numpyro_gpu.txt`. No speed claim follows from this run: the
targets are matched but the hardware is not, and the reference pack's rule applies (no
speedup without matched statistical targets and comparable inference quality).

## On FAS RC

`cluster/m0_gpu_smoke.sbatch` is the batch form. It needs an environment with
`bambi`, `numpyro` and a CUDA `jaxlib`; the existing `electit_env2` environment does not
contain Bambi, so build a new one with `scaffold/cluster/env/install_gpu_env.sh` plus
`pip install -e .` from a checkout of this repository, and set `ENV_PREFIX` and
`PROJECT_DIR` in the script. Partition `gpu` carries A100-80GB nodes; `gpu_requeue`
carries H100/H200. Submitting is outside the M0 handoff (no new compute jobs); the
script is here so the next session can run it without rediscovering the traps.
