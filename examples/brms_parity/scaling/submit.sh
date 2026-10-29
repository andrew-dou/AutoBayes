#!/bin/bash
# The scaling benchmark's entry point on the cluster (README.md). Run on a login node; it only submits jobs.
#
#   submit.sh setup                      build the environment and the data (setup.sbatch)
#   submit.sh run [--dry-run] [--tag T] [--models "S1_logistic_glmm ..."] [--sizes "10000 ..."] [--seeds "1 ..."]
#                 [--configs "brms-rstan ..."]
#                                        expand grid.sh (restricted by the options) into one task per (model, size,
#                                        configuration, seed) and submit one array per resource class into a new
#                                        $BENCH_ROOT/results/<date>-<tag>/; --dry-run prints the tasks and the estimate only
#   submit.sh collect <results dir>      a small job: compare.py --scaling over the run, <results dir>/scaling-<date>.json
set -euo pipefail
SCALING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SCALING_DIR
source "$SCALING_DIR/common.sh"
source "$SCALING_DIR/grid.sh"
cmd=${1:-}
shift || true
common=()
while IFS= read -r o; do common+=("$o"); done < <(sbatch_common)

case $cmd in
setup)
  require_filled
  mkdir -p "$BENCH_ROOT/logs"
  sbatch "${common[@]}" --output="$BENCH_ROOT/logs/setup-%j.out" --export=ALL,SCALING_DIR="$SCALING_DIR" \
    "$SCALING_DIR/setup.sbatch"
  ;;
run)
  dry=0 tag=grid models=("${MODELS[@]}") sizes=("${SIZES[@]}") seeds=("${SEEDS[@]}") only_configs=()
  while [[ $# -gt 0 ]]; do
    case $1 in
      --dry-run) dry=1; shift ;;
      --tag) tag=$2; shift 2 ;;
      --models) read -ra models <<<"$2"; shift 2 ;;
      --sizes) read -ra sizes <<<"$2"; shift 2 ;;
      --seeds) read -ra seeds <<<"$2"; shift 2 ;;
      --configs) read -ra only_configs <<<"$2"; shift 2 ;;
      *) echo "unknown option $1" >&2; exit 2 ;;
    esac
  done
  [[ $dry == 1 ]] || require_filled
  results="$BENCH_ROOT/results/$(date +%F)-$tag"
  if [[ $dry == 1 ]]; then
    work=$(mktemp -d)
  else
    [[ -e "$results" ]] && { echo "$results exists: pick another --tag (nothing is written into an old run)" >&2; exit 3; }
    [[ -f "$ENV_DIR/src_commit" ]] || { echo "no finished setup in $ENV_DIR (submit.sh setup)" >&2; exit 3; }
    mkdir -p "$results/logs" "$results/tasks"
    work="$results/tasks"
  fi
  total=0 core_minutes=0
  for model in "${models[@]}"; do
    for n in "${sizes[@]}"; do
      for c in "${CONFIGS[@]}"; do
        IFS='|' read -r id library cpus args <<<"$c"
        if [[ ${#only_configs[@]} -gt 0 && " ${only_configs[*]} " != *" $id "* ]]; then continue; fi
        for seed in "${seeds[@]}"; do
          mem=$(mem_for "$n" "$library") time=$(time_for "$n")
          echo "$model $n $library $id $seed $cpus $args" >> "$work/tasks-${cpus}c-${mem}-${time//:/}.txt"
          total=$((total + 1))
          core_minutes=$((core_minutes + cpus * $(est_minutes "$n" "$cpus")))
        done
      done
    done
  done
  for f in "$work"/tasks-*.txt; do
    k=$(wc -l < "$f")
    name=$(basename "$f" .txt)
    IFS=- read -r _ cpus mem time <<<"$name"
    cpus=${cpus%c} time="${time:0:2}:${time:2:2}:${time:4:2}"
    echo "$name: $k tasks, --cpus-per-task=$cpus --mem=$mem --time=$time"
    if [[ $dry == 0 ]]; then
      sbatch "${common[@]}" --array=1-"$k" --cpus-per-task="$cpus" --mem="$mem" --time="$time" \
        --output="$results/logs/%x-%A_%a.out" \
        --export=ALL,SCALING_DIR="$SCALING_DIR",RESULTS="$results",TASK_FILE="$f" "$SCALING_DIR/run.sbatch"
    fi
  done
  echo "$total tasks; rough estimate $((core_minutes / 60)) core-hours ($((core_minutes / 60 / 48)) node-hours at 48 cores)"
  if [[ $dry == 1 ]]; then rm -rf "$work"; else echo "results: $results"; fi
  ;;
collect)
  require_filled
  results=${1:?usage: submit.sh collect <results dir>}
  sbatch "${common[@]}" --cpus-per-task=1 --mem=8G --time=01:00:00 --job-name=ab-bench-collect \
    --output="$results/logs/collect-%j.out" \
    --wrap="'$ENV_DIR/venv/bin/python' '$SRC_DIR/examples/brms_parity/compare.py' '$results' --scaling"
  ;;
*)
  sed -n '2,12p' "$0"
  exit 2
  ;;
esac
