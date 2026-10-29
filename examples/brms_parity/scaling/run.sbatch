#!/bin/bash
#SBATCH --job-name=ab-bench-run
# One task of the scaling grid. submit.sh submits one array per resource class (cpus, memory, time), with
# --cpus-per-task, --mem, --time and the partition set there, and exports SCALING_DIR, RESULTS and TASK_FILE; line
# $SLURM_ARRAY_TASK_ID of TASK_FILE is this task:
#   <model> <rows> <library> <config id> <seed> <cpus> <arguments of the benchmark script ...>
# It writes $RESULTS/raw/<model>/N<rows>/<config id>/seed<seed>/: the benchmark's JSON, task.json (where and how it ran)
# and time.txt (/usr/bin/time -v of the benchmark, peak memory among it). It refuses a directory that is not empty.
set -euo pipefail
source "$SCALING_DIR/common.sh"
require_filled
require_job

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")
read -r model n library config seed cpus args <<<"$line"
if [[ -z "${cpus:-}" || "$cpus" != "${SLURM_CPUS_PER_TASK:-}" ]]; then
  echo "task $SLURM_ARRAY_TASK_ID ($line): needs $cpus CPUs, the allocation has ${SLURM_CPUS_PER_TASK:-none}" >&2
  exit 2
fi
out="$RESULTS/raw/$model/N$n/$config/seed$seed"
if [[ -d "$out" && -n "$(ls -A "$out")" ]]; then
  echo "$out is not empty: refusing to overwrite it" >&2
  exit 3
fi
mkdir -p "$out/tmp"
export TMPDIR="$out/tmp"                 # cmdstanr's model, data and CSV files; removed at the end

load_modules
export R_LIBS_USER="$ENV_DIR/Rlib"
export CMDSTAN
CMDSTAN=$(cat "$ENV_DIR/cmdstan_path")
bench="$SRC_DIR/examples/brms_parity"

cat > "$out/task.json" <<EOF
{"model": "$model", "n": $n, "library": "$library", "config": "$config", "seed": $seed, "cpus": $cpus,
 "args": "$args", "job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}", "node": "$(hostname)",
 "cpus_allowed": "$(grep Cpus_allowed_list /proc/self/status | cut -f2)",
 "cpu_model": "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')",
 "threads_per_core": "$(lscpu | awk -F: '/Thread\(s\) per core/ {gsub(/ /, "", $2); print $2}')",
 "src_commit": "$(cat "$ENV_DIR/src_commit")", "start": "$(date -Is)"}
EOF

timer=(command)
[[ -x /usr/bin/time ]] && timer=(/usr/bin/time -v -o "$out/time.txt")
# $args is split into words on purpose: it holds the benchmark's options
# shellcheck disable=SC2086
case $library in
  brms) "${timer[@]}" Rscript "$bench/benchmark_brms.R" "$DATA_DIR/N$n" "$out" "$model" $args --seed "$seed" --no-post ;;
  autobayes) "${timer[@]}" "$ENV_DIR/venv/bin/python" "$bench/benchmark_autobayes.py" "$DATA_DIR/N$n" "$out" "$model" \
               $args --seed "$seed" --no-post ;;
  *) echo "unknown library $library" >&2; exit 2 ;;
esac
rm -rf "$out/tmp"
ls "$out"/*_"$model".json >/dev/null 2>&1 || { echo "no record written for $model (see the log)" >&2; exit 1; }
echo "done: $out"
