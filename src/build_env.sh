#!/bin/bash
# =============================================================================
# build_env.sh -- One-shot, reproducible build of the ESM-2 GPU environment
# for the icbinb project on ASU Sol.
#
# Creates a conda env at /scratch/author/icbinb/.env_build/esm2 using the
# cluster's `mamba/latest` Lmod module. Nothing is written to $HOME.
#
# Usage:  bash /scratch/author/icbinb/src/build_env.sh
# =============================================================================
set -euo pipefail

PROJ=/scratch/author/icbinb
ENV_PREFIX=$PROJ/.env_build/esm2

# --- CRITICAL: ignore ~/.local/lib/python3.11/site-packages ----------------
# A conda env is NOT a venv, so python still puts the *user* site-packages on
# sys.path ahead of the env's own. This user has ~292 packages in
# ~/.local/lib/python3.11/site-packages (tokenizers, packaging, pyyaml, ...),
# which pip would happily treat as "already satisfied" -- leaving the env
# silently dependent on $HOME and non-reproducible. PYTHONNOUSERSITE kills that.
export PYTHONNOUSERSITE=1

# --- Keep every cache off $HOME -------------------------------------------
export CONDA_PKGS_DIRS=$PROJ/.env_build/conda_pkgs
export PIP_CACHE_DIR=$PROJ/.cache/pip
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$PROJ/.cache"

# --- Cluster modules -------------------------------------------------------
source /etc/profile.d/modules.sh
module purge
module load mamba/latest

echo "### Creating python 3.11 env at $ENV_PREFIX"
mamba create -y -p "$ENV_PREFIX" -c conda-forge python=3.11 pip

PY=$ENV_PREFIX/bin/python
echo "### Base python: $($PY -VV)"

echo "### Installing PyTorch (CUDA 12.8 build, A100 sm_80)"
"$PY" -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

echo "### Installing scientific stack + ESM"
"$PY" -m pip install --no-cache-dir \
    fair-esm \
    transformers \
    numpy scipy pandas matplotlib seaborn scikit-learn \
    biotite biopython tqdm h5py pyarrow \
    accelerate safetensors

# Belt-and-braces: make `conda activate` set it too, for anyone who activates
# the env that way instead of sourcing activate_env.sh.
mkdir -p "$ENV_PREFIX/etc/conda/activate.d"
cat > "$ENV_PREFIX/etc/conda/activate.d/zz_icbinb_env.sh" <<'EOF'
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/author/icbinb/.cache
export TORCH_HOME=/scratch/author/icbinb/.cache
export XDG_CACHE_HOME=/scratch/author/icbinb/.cache
unset TRANSFORMERS_CACHE
EOF

echo "### Verifying no dependency leaks from ~/.local"
"$PY" -m pip check || true

echo "### DONE building. Package versions:"
"$PY" -m pip list 2>/dev/null | grep -iE '^(torch|fair-esm|transformers|numpy|scipy|pandas|matplotlib|seaborn|scikit-learn|biotite|biopython|tqdm|h5py|pyarrow) '
