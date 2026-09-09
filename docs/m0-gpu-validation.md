# M0 GPU validation

Status (2026-09-09): **executed once, on a rented H100**, matched to the CPU record.
Before that (2026-09-08) the GPU path was supported in code only; the refusal on a machine
without a GPU is still captured in `docs/m0-evidence/example_numpyro_gpu_attempt.txt`.

## What was run and what it showed

- Node: GiveMeANode `autobayes-m0-smoke`, chip `h100-1` (NVIDIA H100 80GB HBM3, driver
  595.71.05), image `cuda-12.9`, region europe-north1, mission `autobayes-m0-gpu-smoke`.
  Rate $0.0666/min; 10.83 billed minutes, **$0.72**. Node stopped and deleted afterwards.
- Environment: Python 3.11.15, pymc 5.28.5, pytensor 2.38.3, bambi 0.17.2, formulae 0.6.2,
  arviz 0.23.4, numpyro 0.21.0, `jax[cuda12]==0.10.2` (same versions as the laptop).
- Command (20-minute `timeout`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `PYTENSOR_FLAGS=floatX=float64`):

```bash
python -u examples/m0_varying_intercept.py --backend numpyro --device gpu \
    --chains 4 --draws 2500 --tune 1000 \
    --record gpu_record.json --compare docs/m0-evidence/B2_numpyro_cpu_record.json
```

- Fit report: `device actual : gpu:NVIDIA H100 80GB HBM3(id=0)`, `dtype : float64`,
  `chain method : vectorized`, seed 7, wall 39.2 s including XLA compilation.
- Panel: 0 divergences, max R-hat 1.0060, min bulk ESS 798, min tail ESS 1404, min E-BFMI
  0.674, 0 % tree-depth saturation. All rows `pass`.
- Agreement with the CPU NumPyro record (same code, same seed, 4 parallel CPU chains on the
  laptop): population-parameter means within 4 combined MCSE on every parameter
  (`agree: True`; the largest ratio of difference to tolerance is 0.24, on `Intercept`).
- A4 on the GPU (`gpu_a4_check.py`, reproduced in `example_numpyro_gpu.txt`): the
  JAX-compiled log density that NumPyro samples equals PyTensor's C path at five random
  points to 9e-13, with JAX's default device the GPU.
- No speed claim. 39 s on the GPU versus 4 s on the laptop CPU is what 200 rows and 4
  chains look like on an H100 with compilation inside both numbers. The reference pack's
  rule stands: no speedup without matched targets and comparable inference quality, and
  this run was designed to match targets, not to measure throughput.

Evidence files: `docs/m0-evidence/example_numpyro_gpu.txt` (full output),
`docs/m0-evidence/gpu_record.json` (fit record), `docs/m0-evidence/B2_numpyro_cpu_record.json`
(the CPU record it was compared with).

## Reproducing it

On any machine with an NVIDIA GPU and CUDA 12 drivers:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python "pymc==5.28.5" "bambi==0.17.2" "arviz==0.23.4" \
    "numpyro==0.21.0" "jax[cuda12]==0.10.2" "matplotlib<3.11" pytest pandas scipy xarray
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTENSOR_FLAGS=floatX=float64 timeout 20m \
.venv/bin/python -u examples/m0_varying_intercept.py --backend numpyro --device gpu --chains 4 \
    --record docs/m0-evidence/gpu_record_<host>.json --compare docs/m0-evidence/B2_numpyro_cpu_record.json
```

What counts as evidence: `device actual : gpu:...` (or `cuda:`), `dtype : float64`,
`chain method : vectorized`, every panel row at `pass`, and `agree: True` in the comparison
block. Keep the whole output under `docs/m0-evidence/` with a new file name; never write
over an earlier record.

On a rented node the source has to get there without a repository credential: this run
shipped `autobayes/`, the example and the CPU record as files (a shell archive with quoted
heredocs; make sure every file ends in a newline or the terminator is swallowed), checked
SHA-256 against the local copies, ran the command detached with a `timeout`, read the
output back, then stopped and deleted the node.

## On FAS RC

`cluster/m0_gpu_smoke.sbatch` is the batch form. It needs an environment with `bambi`,
`numpyro` and a CUDA `jaxlib`; the existing `electit_env2` environment does not contain
Bambi, so build a new one with `scaffold/cluster/env/install_gpu_env.sh` plus
`pip install -e .` from a checkout of this repository, and set `ENV_PREFIX` and
`PROJECT_DIR` in the script. Partition `gpu` carries A100-80GB nodes; `gpu_requeue`
carries H100/H200. Not yet run there.
