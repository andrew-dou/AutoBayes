# Shared by setup.sbatch, run.sbatch and submit.sh: the settings and two checks.
SCALING_DIR=${SCALING_DIR:?SCALING_DIR must name examples/brms_parity/scaling}
source "$SCALING_DIR/cluster.env"

require_filled() {                     # refuse to run with a placeholder left in cluster.env
  local left
  left=$(grep -o '__FILL_[A-Z_]*__' "$SCALING_DIR/cluster.env" | sort -u | tr '\n' ' ' || true)
  if [[ -n "$left" ]]; then
    echo "cluster.env still has placeholders: $left" >&2
    exit 2
  fi
}

require_job() {                        # never build or sample on a login node
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "$(basename "$0") must run inside a SLURM job (submit it with submit.sh), not on a login node" >&2
    exit 2
  fi
}

load_modules() {
  module purge
  module load "$R_MODULE"
  [[ "$GCC_MODULE" == none ]] || module load "$GCC_MODULE"
  [[ "$PYTHON_MODULE" == none ]] || module load "$PYTHON_MODULE"
}

sbatch_common() {                      # the options every job shares
  local opts=(--partition="$PARTITION")
  [[ "$ACCOUNT" == none ]] || opts+=(--account="$ACCOUNT")
  [[ "$CONSTRAINT" == none ]] || opts+=(--constraint="$CONSTRAINT")
  printf '%s\n' "${opts[@]}"
}
