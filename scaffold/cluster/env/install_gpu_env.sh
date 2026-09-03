#!/bin/bash
# Reference environment build for GPU sampling. Adapt the pins; keep the checks.
#
#   bash cluster/env/install_gpu_env.sh /path/to/env
set -euo pipefail

ENV_PREFIX="${1:?usage: install_gpu_env.sh <env-prefix>}"

# Pin the accelerator wheel to the DRIVER's CUDA version, not to the newest
# available. A wheel built against a newer CUDA imports fine and dies on the
# first kernel launch. Check the driver first:
nvidia-smi --query-gpu=driver_version --format=csv,noheader || true
echo "Pin the CUDA wheel to the CUDA version this driver supports."

conda create -y -p "$ENV_PREFIX" python=3.11
source activate "$ENV_PREFIX"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

pip install --upgrade pip

# EDIT: match the local CUDA. cuda12 shown; use the matching extra for your driver.
pip install "jax[cuda12]"
pip install numpyro pymc arviz numpy scipy pandas matplotlib

# --- verification, not optional ---------------------------------------------
python3 - <<'PY'
import jax, jax.numpy as jnp
print("jax", jax.__version__)
print("devices", jax.devices())
x = jnp.ones((512, 512))
print("kernel ok, matmul sum =", float((x @ x).sum()))
PY

# Confirm the shared library the loader will actually need is present.
find "$ENV_PREFIX" -name 'libffi.so*' -print -quit

echo
echo "Environment built at $ENV_PREFIX"
echo "In every sbatch script, after 'source activate', add:"
echo "  export LD_LIBRARY_PATH=\"${ENV_PREFIX}/lib:\${LD_LIBRARY_PATH:-}\""
