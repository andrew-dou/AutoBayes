# Cluster

Templates and preflight checks for GPU sampling runs under SLURM.

## Files

| File | Purpose |
| --- | --- |
| `slurm_gpu_fit.sbatch` | Copy per run. Carries the traps below, already handled. |
| `preflight.py` | Cheap checks that catch expensive failures. Run before queueing. |
| `env/install_gpu_env.sh` | Reference environment build, with the version pins that matter. |

## The traps

These are not hypothetical. Each one cost real time on the project this came from.

**`source activate` does not export `LD_LIBRARY_PATH`.** On a compute node the job
dies about seven seconds in with `ImportError: libffi.so.8`. It reads like a broken
install and is not one — the library is in the environment, the loader just cannot
see it. Every script that activates an environment needs, immediately after:

```bash
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

**A wheel that imports is not a wheel that runs.** A build linked against a newer
CUDA than the driver imports cleanly on a login node and dies on the first kernel
launch. `preflight.py` does an actual matmul rather than just importing. Pin the
accelerator wheel to the driver's CUDA version and verify with `ldd`, not with a
successful import.

**JAX preallocates the device.** Left at the default it takes nearly all of the
GPU at import, and anything else sharing the node fails with an out-of-memory error
naming the wrong process:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
```

**Extension dispatch.** If any part of your tooling dispatches on file extension, a
batch script named `.sh` can be executed on the login node instead of submitted.
It then looks exactly like a hung SSH session for as long as you let it run. Keep
`.sbatch` on anything meant for `sbatch`.

**Fresh output directory, always.** The template refuses to start if the output
directory already exists. A fitted trace is large, is somebody's comparison
baseline, and is not recoverable once overwritten.

## Workflow

1. Copy `slurm_gpu_fit.sbatch` to `cluster/fit_<name>.sbatch`. Do not edit the
   template in place.
2. Fill in every `CHANGEME`.
3. Append the `DECISIONS.md` entry — `python3 scripts/new_decision.py "<title>"`.
4. `python3 cluster/preflight.py --outdir fits/<tag>` on a login node.
5. `sbatch cluster/fit_<name>.sbatch`. Record the job ID immediately.
6. Monitor with accounting and log tails. Do not poll faster than the job can
   change state.
7. On completion the template registers the run automatically. Fill in the
   **Result** section of the decision entry and the `wiki/models/` page.

## A note on accounting retention

Scheduler accounting is purged, often after a few weeks, and a query spanning too
wide a date range fails rather than truncating. Do not rely on it for anything you
will want later — that is what `scripts/run_registry.py` is for. Wall clock
recovered from log timestamps is the fallback, and it includes setup and
serialization, so it will exceed the sampler's own reported time.
