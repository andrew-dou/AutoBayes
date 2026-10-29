#!/bin/bash
#SBATCH --job-name=ab-bench-setup
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# Builds the benchmark's environment under $ENV_DIR (cluster.env) and the scaling data under $DATA_DIR. Submitted by
# `submit.sh setup` (which adds the partition, account and log path). Every step that downloads is marked DOWNLOAD.
# Each step is skipped when its result is already there, so a failed setup can be resubmitted; it writes nowhere else.
set -euo pipefail
source "$SCALING_DIR/common.sh"
require_filled
require_job

marker="$ENV_DIR/.autobayes-bench-env"
if [[ -d "$ENV_DIR" && ! -f "$marker" ]]; then
  echo "$ENV_DIR exists and was not made by this setup: refusing to write into it" >&2
  exit 3
fi
mkdir -p "$ENV_DIR"
touch "$marker"
export MAKEFLAGS="-j${SLURM_CPUS_PER_TASK:-8}"

load_modules
echo "== versions on $(hostname) =="
R --version | head -1
g++ --version | head -1
make --version | head -1
python3 --version || true
module list 2>&1

# --- R: brms 2.23.0 (only if the cluster's R lacks it), cmdstanr, CmdStan --------------------------------------------
export R_LIBS_USER="$ENV_DIR/Rlib"
mkdir -p "$R_LIBS_USER"
export CMDSTAN_DIR="$ENV_DIR/cmdstan"
Rscript - <<'EOF'
options(Ncpus = as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "8")), repos = c(CRAN = "https://cloud.r-project.org"))
lib <- Sys.getenv("R_LIBS_USER")
.libPaths(c(lib, .libPaths()))
have <- function(p, v = NULL) {
  ok <- requireNamespace(p, quietly = TRUE)
  ok && (is.null(v) || as.character(packageVersion(p)) == v)
}
if (!have("brms", "2.23.0")) {
  # DOWNLOAD: brms's dependencies from CRAN (rstan among them, compiled from source: the slow part), then brms 2.23.0
  # itself from CRAN's current or archived source tarball.
  deps <- tools::package_dependencies("brms", db = available.packages(), which = c("Depends", "Imports", "LinkingTo"),
                                      recursive = TRUE)[[1]]
  base <- rownames(installed.packages(priority = "base"))
  missing <- setdiff(deps, c(base, rownames(installed.packages())))
  if (length(missing)) install.packages(missing, lib = lib)
  tarball <- file.path(tempdir(), "brms_2.23.0.tar.gz")
  ok <- tryCatch(download.file("https://cloud.r-project.org/src/contrib/brms_2.23.0.tar.gz", tarball) == 0,
                 error = function(e) FALSE, warning = function(e) FALSE)
  if (!ok) download.file("https://cloud.r-project.org/src/contrib/Archive/brms/brms_2.23.0.tar.gz", tarball)
  install.packages(tarball, repos = NULL, type = "source", lib = lib)
}
if (!have("cmdstanr")) {
  # DOWNLOAD: cmdstanr from the Stan team's r-universe (not on CRAN), its dependencies from CRAN.
  install.packages("cmdstanr", lib = lib, repos = c("https://stan-dev.r-universe.dev", getOption("repos")))
}
library(cmdstanr)
dir <- Sys.getenv("CMDSTAN_DIR")
dir.create(dir, showWarnings = FALSE)
if (!length(list.dirs(dir, recursive = FALSE))) {
  # DOWNLOAD: the latest CmdStan release from GitHub (stan-dev/cmdstan), built here with the loaded compiler.
  install_cmdstan(dir = dir, cores = getOption("Ncpus"))
}
set_cmdstan_path(tail(sort(list.dirs(dir, recursive = FALSE)), 1))
writeLines(cmdstan_path(), file.path(dirname(dir), "cmdstan_path"))
cat("brms", as.character(packageVersion("brms")), "| rstan", as.character(packageVersion("rstan")), "| cmdstanr",
    as.character(packageVersion("cmdstanr")), "| CmdStan", cmdstan_version(), "at", cmdstan_path(), "\n")

# Check that threading and --march-native's flags reach the compiler (benchmark_brms.R's cpp_options).
f <- write_stan_file("data { int N; } parameters { real mu; } model { mu ~ normal(0, 1); }", dir = tempdir())
opts <- list(stan_threads = TRUE, STAN_CPP_OPTIMS = "true", CXXFLAGS_OPTIM = "-O3 -march=native")
log <- capture.output(m <- cmdstan_model(f, cpp_options = opts, quiet = FALSE, force_recompile = TRUE), type = "output")
hits <- grep("march=native", log, value = TRUE)
cat(if (length(hits)) "march-native check: -march=native is on the compile line\n" else
  "march-native check: NOT FOUND in the captured compile output; read it below before using --march-native\n")
if (!length(hits)) writeLines(tail(log, 40))
cat("threading check: exe_info stan_threads =", tryCatch(format(m$exe_info()$stan_threads), error = function(e) "?"), "\n")
EOF

# --- Python: a venv with AutoBayes's pinned dependencies (uv.lock), CPU JAX ------------------------------------------
export UV_CACHE_DIR="$ENV_DIR/uv-cache" UV_PYTHON_INSTALL_DIR="$ENV_DIR/uv-python"
if [[ ! -x "$ENV_DIR/uv/bin/uv" ]]; then
  # DOWNLOAD: uv (the lock file's installer) from PyPI into its own small venv.
  python3 -m venv "$ENV_DIR/uv"
  "$ENV_DIR/uv/bin/pip" install --quiet uv==0.11.24
fi
if [[ "$PYTHON_MODULE" == none ]]; then
  py=3.11            # DOWNLOAD (if no Python 3.11 is found): uv fetches a standalone CPython 3.11 into $UV_PYTHON_INSTALL_DIR
else
  py=$(command -v python3.11 || command -v python3)
  "$py" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' \
    || { echo "PYTHON_MODULE gives $("$py" --version), not 3.11 (pyproject.toml requires 3.11)" >&2; exit 4; }
fi
if [[ ! -x "$ENV_DIR/venv/bin/python" ]]; then
  # DOWNLOAD: the exact versions in uv.lock (pymc, bambi, jax/jaxlib CPU wheels, numpyro, ...) from PyPI.
  UV_PROJECT_ENVIRONMENT="$ENV_DIR/venv" "$ENV_DIR/uv/bin/uv" sync --frozen --extra jax --project "$SRC_DIR" --python "$py"
fi
"$ENV_DIR/venv/bin/python" - "$SRC_DIR" <<'EOF'
import sys
from pathlib import Path
import jax, numpyro, pymc, autobayes
src = Path(sys.argv[1]).resolve()
assert Path(autobayes.__file__).resolve().is_relative_to(src), f"autobayes imported from {autobayes.__file__}, not {src}"
print(f"python {sys.version.split()[0]} | autobayes {autobayes.__file__} | jax {jax.__version__} | numpyro "
      f"{numpyro.__version__} | pymc {pymc.__version__} | devices {jax.devices()}")
EOF

# --- the scaling data: data/N<rows>/<model>.csv, fixed seeds (benchmark_data.py) -------------------------------------
source "$SCALING_DIR/grid.sh"
for n in "${SIZES[@]}"; do
  d="$DATA_DIR/N$n"
  if [[ -d "$d" ]]; then
    echo "data N=$n: $d exists, left as it is"
    continue
  fi
  "$ENV_DIR/venv/bin/python" "$SRC_DIR/examples/brms_parity/benchmark_data.py" "$d" --scaling "$n"
  echo "data N=$n: $(ls "$d" | wc -l) files, $(du -sh "$d" | cut -f1)"
done
git -C "$SRC_DIR" rev-parse HEAD > "$ENV_DIR/src_commit"
echo "setup done: commit $(cat "$ENV_DIR/src_commit")"
