# The scaling benchmark on FASRC

brms (rstan, and cmdstanr with Stan's within-chain threading) against AutoBayes (NumPyro on the CPU, with a matched
core cap) on five larger models at 10k, 100k and 1M rows, scored by bulk and tail ESS per second. It reuses the
benchmark one directory up unchanged; the options it adds default to the 25 October run's behaviour.

| file | what it is |
| --- | --- |
| `cluster.env` | paths, modules, partition: fill every `__FILL_...__` on the cluster (the scripts refuse to run until then) |
| `grid.sh` | models, sizes, seeds, the configurations, and memory and time per size |
| `submit.sh` | `setup`, `run [--dry-run] [--tag ...] [--models/--sizes/--seeds/--configs "..."]`, `collect <results dir>` |
| `setup.sbatch` | builds `env/` (R library, cmdstanr, CmdStan, the Python venv) and `data/N<rows>/` |
| `run.sbatch` | one array task: one (model, rows, configuration, seed); refuses to run outside a SLURM job |
| `common.sh` | the checks and module loading the three share |

## Steps

1. On a login node, clone the code into the fresh benchmark directory (a download from GitHub; the repository is
   public): `git clone --branch bench/scaling https://github.com/andrew-dou/AutoBayes.git
   /n/netscratch/king_lab/Everyone/adou/autobayes-bench/src`. Fill `src/examples/brms_parity/scaling/cluster.env`.
2. `submit.sh setup`. It loads the modules and prints their versions, then: brms 2.23.0 into `env/Rlib` only if the
   cluster's R lacks that version (its dependencies, rstan among them, are compiled from source: most of the job's
   time); cmdstanr from https://stan-dev.r-universe.dev; CmdStan (latest release) via `cmdstanr::install_cmdstan()`
   into `env/cmdstan`; uv, then a venv `env/venv` with exactly `uv.lock`'s versions (`uv sync --frozen --extra jax`,
   CPU jaxlib); the data. Every download is marked `DOWNLOAD` in `setup.sbatch`. It also compiles a tiny model with
   threading and `--march-native`'s flags and says whether `-march=native` reached the compile line. Read its log
   (`logs/setup-<job>.out`) before the first run.
3. `submit.sh run --dry-run` prints the task classes and the estimate; `submit.sh run --sizes 10000 --seeds 1 --tag
   pilot` is a cheap first pass (40 tasks); then `submit.sh run`. Each call writes a new `results/<date>-<tag>/`
   (refused if it exists): `raw/<model>/N<rows>/<config>/seed<s>/` holds the benchmark's JSON, `task.json` (node,
   CPU model, allowed CPUs, threads per core, commit) and `time.txt` (`/usr/bin/time -v`: peak memory).
4. `submit.sh collect <results dir>`: a one-core job running `compare.py --scaling`, which writes
   `<results dir>/scaling-<date>.json` (no draws, a few hundred kB). Only that file comes back
   (`scp fasrc:<results dir>/scaling-<date>.json examples/brms_parity/results/`); the raw records, with their
   draws, stay on the cluster. `report.py examples/brms_parity/results/scaling-<date>.json --pdf` writes the report.

## The grid

5 models x 3 sizes x 8 configurations x 3 seeds = 360 tasks. `--dry-run`'s rough estimate is 2,600 core-hours
(about 55 node-hours of 48 cores), dominated by the 1M rows; it assumes perfect threading, so the real figure may be
up to twice that.

- Models (`benchmark_data.py --scaling N`): S1 `y ~ x1 + x2 + (1 + x1 | g)` Bernoulli; S2 `y ~ x1 + x2 + (1 + x1 + x2 |
  g)` Gaussian; S3 `y ~ x1 + (1 | g1) + (1 | g2)` Poisson, J and J/2 crossed levels; S4 `y ~ x1 + x2 + (1 | g)`
  cumulative logit; S5 `y ~ gp(x, k = 30)` Gaussian. Explicit priors as B1-B18: intercept normal(0, 2.5), `b`
  normal(0, 1), `sd`, `sigma` and `sdgp` half-normal(1), `cor` lkj(2), `lscale` inv_gamma(3, 1).
- J = N/100 up to 100k rows, capped at 1,000 at 1M: AutoBayes builds the design through Bambi and formulae, which keep a
  dense int64 N x J indicator per group term (80 GB a term at N = 1M, J = 10,000). Memory requests follow (`mem_for`).
- Configurations: brms rstan (4 chains, 4 cores); cmdstanr without threading; cmdstanr `threading(t)`, t = 1, 2, 4
  (4t cores); AutoBayes `--cores` 4, 8, 16. Commented-out lines add cmdstanr with `--march-native` and with
  `--normalize false`.
- Seeds 1-3 are the samplers'; the data are fixed per model and size.
- The comparison: per (model, size) each library's configuration with the highest median bulk ESS per second of the
  samplers' time, against the other's, plus the full sweep (ESS per second, end-to-end ESS per second, CPU seconds,
  cores kept busy U, time per leapfrog step, R-hat).

## The core cap

brms: SLURM gives the task 4t CPUs (`--cpus-per-task`, enforced by the cgroup); cmdstanr runs the four chains as four
processes (`parallel_chains = 4`), each with t TBB threads (`threads_per_chain`). rstan's four chains are four
forked processes of one thread.

AutoBayes, `--cores c` (with the same `--cpus-per-task`), applied before numpy, JAX or autobayes is imported:

- Enforced: the process's CPU affinity is set to c of the CPUs it is allowed (`os.sched_setaffinity`). Every thread
  started afterwards inherits it, XLA's included, so the process never runs on more than c cores at once. Under
  SLURM the cgroup enforces the same c.
- Sized to c: `NPROC` (XLA's CPU client sizes its thread pool from it, as far as we know; otherwise from the affinity,
  which is also c), `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`.
- One XLA CPU device per chain (`AUTOBAYES_HOST_DEVICES=4`, that is `--xla_force_host_platform_device_count=4`):
  NumPyro runs the four chains in parallel on the c cores.
- Not controlled: how XLA divides the c cores among the chains and inside one chain. AutoBayes has nothing like
  `reduce_sum`: whether a gradient is split over threads is XLA's choice, operation by operation. At c = 16 it may
  still keep only about four cores busy; U in the records shows what it did. macOS has no CPU affinity, so a local
  run is not capped (`core_cap.enforced = false` in its record).

compare.py flags any record whose CPU seconds over the samplers' time exceed its cores by 5%.

## What these runs do not show

- cmdstanr's gradient is not measured on its own: the brms fit has no rstan model to call. The scaling comparison
  uses the time per leapfrog step instead (slowest chain's time over its steps), which needs no separate gradient.
- Under cmdstanr, `brm(chains = 0)` runs one chain of two iterations after compiling; the compile stage includes it.
- Tasks share nodes with other jobs unless `--exclusive` is added in `submit.sh`; memory bandwidth is shared. Pin
  one CPU model with `CONSTRAINT`, or the libraries may land on different hardware.
- Post-processing is not timed (`--no-post`): at 1M rows `posterior_epred` alone is 4,000 x 1M doubles.

## Not checked before the first cluster run

Only rstan, the data at 10k rows and AutoBayes's one-chain smoke run of S1 were exercised locally (macOS, no
cmdstanr). Unchecked: everything under `--backend cmdstanr` (compile, `save_warmup`, `$cpp_options()`,
`$exe_info()`); whether `STAN_CPP_OPTIMS=true` and `CXXFLAGS_OPTIM=-O3 -march=native` reach the compiler (setup
checks); whether XLA honours `NPROC`; the four-chain AutoBayes path under the affinity cap; the S2-S5 fits; the 1M
memory requests and time limits (guesses); `setup.sbatch`, `run.sbatch` and `submit.sh` themselves, except
`submit.sh run --dry-run`.
