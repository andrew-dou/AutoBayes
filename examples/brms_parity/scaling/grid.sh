# The scaling grid, sourced by submit.sh: one SLURM array task per (model, size, configuration, seed).

MODELS=(S1_logistic_glmm S2_gaussian_slopes S3_poisson_crossed S4_ordinal_glmm S5_gp_approx)
SIZES=(10000 100000 1000000)
SEEDS=(1 2 3)                          # the samplers' seeds; the data are fixed per (model, size)

# id|library|cpus|arguments of the benchmark script (run.sbatch adds --seed and --no-post). cpus is the SLURM
# --cpus-per-task: brms 4 chains x t threads; AutoBayes --cores the same number.
CONFIGS=(
  "brms-rstan|brms|4|--backend rstan"
  "brms-cmdstanr|brms|4|--backend cmdstanr"
  "brms-cmdstanr-t1|brms|4|--backend cmdstanr --threads 1"
  "brms-cmdstanr-t2|brms|8|--backend cmdstanr --threads 2"
  "brms-cmdstanr-t4|brms|16|--backend cmdstanr --threads 4"
  "autobayes-c4|autobayes|4|--cores 4"
  "autobayes-c8|autobayes|8|--cores 8"
  "autobayes-c16|autobayes|16|--cores 16"
  # off by default; uncomment to add brms's other speed options to the sweep:
  # "brms-cmdstanr-t4-native|brms|16|--backend cmdstanr --threads 4 --march-native"
  # "brms-cmdstanr-t4-unnorm|brms|16|--backend cmdstanr --threads 4 --normalize false"
)

mem_for() {                            # mem_for <rows> <library>: --mem; AutoBayes holds formulae's N x J indicators
  case $1 in
    10000) echo 8G ;;
    100000) [[ $2 == autobayes ]] && echo 24G || echo 12G ;;
    *) [[ $2 == autobayes ]] && echo 96G || echo 32G ;;
  esac
}

time_for() {                           # time_for <rows>: --time (a guess; a TIMEOUT task is resubmitted with more)
  case $1 in
    10000) echo 02:00:00 ;;
    100000) echo 08:00:00 ;;
    *) echo 24:00:00 ;;
  esac
}

est_minutes() {                        # est_minutes <rows> <cpus>: a rough wall time, for submit.sh --dry-run's total only:
  local fixed sample                   # compile, data, gradient measurement; plus sampling on 4 cores, perfectly scaled
  case $1 in
    10000) fixed=4; sample=4 ;;
    100000) fixed=6; sample=30 ;;
    *) fixed=15; sample=240 ;;
  esac
  echo $(( fixed + sample * 4 / $2 ))
}
