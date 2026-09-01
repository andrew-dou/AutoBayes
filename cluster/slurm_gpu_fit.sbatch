#!/bin/bash
# =============================================================================
# slurm_gpu_fit.sbatch — template for a single GPU sampling run.
#
# Copy this per run; do not edit it in place. Name the copy after the run
# (fit_m47_h200.sbatch), not after the template.
#
# Usage:
#   sbatch cluster/fit_<name>.sbatch
#
# The comments marked TRAP are failure modes that have actually cost time.
# Leave them in the copies.
# =============================================================================

#SBATCH --job-name=fit
#SBATCH --partition=CHANGEME_gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=15:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

# --- TRAP: extension dispatch -----------------------------------------------
# If anything in your pipeline dispatches by file extension, a batch script
# named .sh may be executed directly on the LOGIN node instead of submitted.
# It then looks exactly like a hung SSH session for as long as you let it.
# Keep the .sbatch extension on anything meant for sbatch.

ENV_PREFIX="/CHANGEME/path/to/conda/env"
PROJECT_DIR="/CHANGEME/path/to/project"

cd "$PROJECT_DIR"
mkdir -p logs

module load python 2>/dev/null || true
source activate "$ENV_PREFIX"

# --- TRAP: libffi -----------------------------------------------------------
# `source activate` does NOT export LD_LIBRARY_PATH on a compute node. Without
# the next line the job dies about seven seconds in with
#   ImportError: libffi.so.8: cannot open shared object file
# which reads like a broken install and is not one. Verify the library is
# actually in the env with:  find "$ENV_PREFIX" -name 'libffi.so*'
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# --- TRAP: JAX preallocation ------------------------------------------------
# By default JAX grabs almost the whole device at import. Anything else sharing
# the GPU then fails with an out-of-memory error that names the wrong process.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

# Keep the graph-building backend on CPU in double precision; only the sampler
# runs on the accelerator.
export PYTENSOR_FLAGS="device=cpu,floatX=float64,cxx="

# --- sampler configuration --------------------------------------------------
SEED=1234
CHAINS=40
DRAWS=2500
WARMUP=2500
MAX_TREEDEPTH=11
TARGET_ACCEPT=0.9
JITTER_SCALE=0.3

# --- TRAP: output directory -------------------------------------------------
# A fresh directory per run, always. Never point this at an existing fit
# directory: the trace already there is somebody's comparison baseline, it is
# large, and overwriting it is not recoverable from anywhere.
RUN_TAG="CHANGEME_run_tag"
OUTDIR="fits/${RUN_TAG}"
if [ -e "$OUTDIR" ]; then
  echo "ERROR: $OUTDIR already exists. Choose a new RUN_TAG." >&2
  exit 1
fi
mkdir -p "$OUTDIR"

# --- preflight --------------------------------------------------------------
# Cheap and worth it: build the model, take one log-density and one gradient,
# assert both finite. Catches specification errors before the sampler starts.
python3 cluster/preflight.py --config "$OUTDIR/../../config.yaml" || {
  echo "ERROR: preflight failed; not sampling." >&2
  exit 1
}

# --- record what this run is ------------------------------------------------
{
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "node=$(hostname)"
  echo "started=$(date -Is)"
  echo "chains=$CHAINS draws=$DRAWS warmup=$WARMUP"
  echo "max_treedepth=$MAX_TREEDEPTH target_accept=$TARGET_ACCEPT seed=$SEED"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee "$OUTDIR/run_meta.txt"

# --- sample -----------------------------------------------------------------
START=$(date +%s)

python3 -u CHANGEME_fit_entrypoint.py \
  --chains "$CHAINS" \
  --draws "$DRAWS" \
  --warmup "$WARMUP" \
  --max-treedepth "$MAX_TREEDEPTH" \
  --target-accept "$TARGET_ACCEPT" \
  --jitter-scale "$JITTER_SCALE" \
  --seed "$SEED" \
  --outdir "$OUTDIR"

END=$(date +%s)
ELAPSED=$(( END - START ))
echo "elapsed_seconds=$ELAPSED" | tee -a "$OUTDIR/run_meta.txt"
printf 'elapsed=%dh%02dm\n' $(( ELAPSED / 3600 )) $(( (ELAPSED % 3600) / 60 )) \
  | tee -a "$OUTDIR/run_meta.txt"

# Register the run so the wall clock survives log rotation.
python3 scripts/run_registry.py add \
  --tag "$RUN_TAG" \
  --outdir "$OUTDIR" \
  --job-id "${SLURM_JOB_ID:-none}" \
  --chains "$CHAINS" --draws "$DRAWS" --warmup "$WARMUP" \
  --max-treedepth "$MAX_TREEDEPTH" \
  --seconds "$ELAPSED" || true

echo "done. outputs in $OUTDIR"
