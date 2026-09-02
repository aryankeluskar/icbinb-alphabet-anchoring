# =============================================================================
# activate_env.sh -- activate the icbinb ESM-2 GPU environment on ASU Sol.
#
#   source /scratch/author/icbinb/src/activate_env.sh
#
# (SOURCE it, don't execute it -- it has to modify your current shell.)
#
# Gives you: python 3.11 / torch 2.11.0+cu128 / fair-esm 2.0.0 / transformers 5.x
# Nothing is read from or written to $HOME.
# =============================================================================

ICBINB_PROJ=/scratch/author/icbinb
ICBINB_ENV=$ICBINB_PROJ/.env_build/esm2

# --- 1. Lmod: mamba provides the conda runtime -------------------------------
if [ -z "${LMOD_CMD:-}" ] && [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
fi
module load mamba/latest >/dev/null 2>&1

# --- 2. Activate the env -----------------------------------------------------
# `conda activate` needs the shell hook; fall back to a plain PATH prepend.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate "$ICBINB_ENV" 2>/dev/null
fi
if [ "${CONDA_PREFIX:-}" != "$ICBINB_ENV" ]; then
    export PATH="$ICBINB_ENV/bin:$PATH"
    export CONDA_PREFIX="$ICBINB_ENV"
fi

# --- 3. CRITICAL: ignore ~/.local/lib/python3.11/site-packages ----------------
# A conda env is not a venv, so python would otherwise put the *user* site-dir
# on sys.path AHEAD of the env's own. This account has ~292 packages there
# (scipy, pandas, matplotlib, sklearn, tqdm, tokenizers -- and even pip), which
# would shadow the pinned versions installed here and silently break
# reproducibility. Without this the env is NOT self-contained.
export PYTHONNOUSERSITE=1

# --- 4. Keep every cache on /scratch, never $HOME ----------------------------
export HF_HOME=$ICBINB_PROJ/.cache/huggingface
export TORCH_HOME=$ICBINB_PROJ/.cache/torch
export XDG_CACHE_HOME=$ICBINB_PROJ/.cache
export PIP_CACHE_DIR=$ICBINB_PROJ/.cache/pip
export CONDA_PKGS_DIRS=$ICBINB_PROJ/.env_build/conda_pkgs
export MPLCONFIGDIR=$ICBINB_PROJ/.cache/matplotlib

# ~/.bashrc lines 54-55 export HF_HOME and HUGGINGFACE_HUB_CACHE pointing at
# /scratch/author/deceptive-llms/models/ (a DIFFERENT project). The legacy
# HUGGINGFACE_HUB_CACHE / TRANSFORMERS_CACHE vars take PRECEDENCE over HF_HOME
# in huggingface_hub, so setting HF_HOME alone is not enough -- downloads still
# land in the other project's directory. Kill them and pin HF_HUB_CACHE.
unset TRANSFORMERS_CACHE
unset HUGGINGFACE_HUB_CACHE
export HF_HUB_CACHE=$ICBINB_PROJ/.cache/huggingface/hub

# --- 5. Cluster-specific fixes ----------------------------------------------
# marks.hms.harvard.edu (ProteinGym) omits an intermediate CA; use the patched
# bundle if it is present (see memory/env-ssl-and-paths.md).
if [ -f "$ICBINB_PROJ/.cache/ca-bundle-plus.crt" ]; then
    export CURL_CA_BUNDLE=$ICBINB_PROJ/.cache/ca-bundle-plus.crt
    export REQUESTS_CA_BUNDLE=$ICBINB_PROJ/.cache/ca-bundle-plus.crt
    export SSL_CERT_FILE=$ICBINB_PROJ/.cache/ca-bundle-plus.crt
fi

# Silence the ASU AI-model-load tracker's import hook (it wraps
# transformers.from_pretrained via PYTHONPATH=/etc/python/sitecustomize.py and
# shows up in tracebacks). Comment out if you want to keep the tracking.
export NO_AI_TRACKING=true

mkdir -p "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" 2>/dev/null

echo "icbinb env active: $(python -c 'import sys;print(sys.version.split()[0])') @ $ICBINB_ENV"
