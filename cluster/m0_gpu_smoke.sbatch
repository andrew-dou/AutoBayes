#!/bin/bash
# m0_gpu_smoke.sbatch — one GPU run of the M0 example, to turn "GPU supported in code"
# into "GPU validated". Copy per run; set ENV_PREFIX and PROJECT_DIR. See
# docs/m0-gpu-validation.md for what counts as evidence.

#SBATCH --job-name=autobayes_m0_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

ENV_PREFIX="/CHANGEME/path/to/env"        # must contain bambi, numpyro, jax[cuda]
PROJECT_DIR="/CHANGEME/path/to/AutoBayes"

cd "$PROJECT_DIR"
mkdir -p logs docs/m0-evidence

# libffi trap: activation does not export LD_LIBRARY_PATH on compute nodes.
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export PYTENSOR_FLAGS="device=cpu,floatX=float64,cxx="

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$ENV_PREFIX/bin/python" scaffold/cluster/preflight.py --skip-model

OUT="docs/m0-evidence/example_numpyro_gpu_${SLURM_JOB_ID:-local}.txt"
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT" >&2; exit 1; fi
"$ENV_PREFIX/bin/python" -u examples/m0_varying_intercept.py \
  --backend numpyro --device gpu --chains 4 --draws 2500 --tune 1000 | tee "$OUT"
