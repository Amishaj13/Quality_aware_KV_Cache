#!/usr/bin/env bash
# One-time environment setup for the FP16 vs KIVI baseline on SageMaker.
# Works on a SageMaker Studio JupyterLab space or a classic notebook instance.
#
# Usage (from a JupyterLab terminal):
#   bash setup_env.sh
#
# Creates a conda env at $WORKDIR/envs/kivi, clones KIVI, builds its CUDA kernel,
# and installs flash-attn from a prebuilt wheel (no multi-hour compile).
set -euo pipefail

# Notebook instances only persist /home/ec2-user/SageMaker; Studio persists $HOME.
if [ -d "$HOME/SageMaker" ]; then WORKDIR="${WORKDIR:-$HOME/SageMaker/kivi-work}"; else WORKDIR="${WORKDIR:-$HOME/kivi-work}"; fi
ENV_PREFIX="$WORKDIR/envs/kivi"
KIVI_DIR="$WORKDIR/KIVI"
mkdir -p "$WORKDIR"
echo ">> WORKDIR=$WORKDIR"

echo ">> GPU check"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
case "$GPU_CC" in
  8.*|9.*) echo "   compute capability $GPU_CC OK (flash-attn + KIVI need Ampere or newer)";;
  *) echo "!! compute capability $GPU_CC is too old (T4/V100). Use ml.g5 / ml.g6 / ml.g6e."; exit 1;;
esac

# conda or micromamba (Studio's SageMaker Distribution ships micromamba)
if command -v conda >/dev/null 2>&1; then CONDA=conda
elif command -v micromamba >/dev/null 2>&1; then CONDA=micromamba
else echo "!! neither conda nor micromamba found"; exit 1; fi
echo ">> using $CONDA"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  $CONDA create -y -p "$ENV_PREFIX" -c conda-forge python=3.10
fi
# CUDA 12.1 toolkit (nvcc + headers) to build KIVI's kernel against torch 2.4.1+cu121
if [ ! -x "$ENV_PREFIX/bin/nvcc" ]; then
  $CONDA install -y -p "$ENV_PREFIX" -c "nvidia/label/cuda-12.1.1" cuda-toolkit
fi

# nvcc 12.1 rejects gcc > 12 (SageMaker Distribution ships a newer gcc), so use conda-forge gcc 12
if [ ! -x "$ENV_PREFIX/bin/g++" ]; then
  $CONDA install -y -p "$ENV_PREFIX" -c conda-forge "gcc=12" "gxx=12"
fi

PY="$ENV_PREFIX/bin/python"
PIP="$PY -m pip"
$PIP install --upgrade pip wheel setuptools ninja packaging

# Versions pinned by KIVI's pyproject (its Llama 3 / GQA support targets transformers 4.43)
$PIP install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
$PIP install "transformers==4.43.1" "accelerate>=0.33,<1.0" "tokenizers>=0.19,<0.20" \
  sentencepiece protobuf "numpy<2" pandas matplotlib tabulate huggingface_hub

# Prebuilt flash-attn wheel for torch 2.4 / cu12 / py3.10 (building from source takes hours)
$PIP install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

if [ ! -d "$KIVI_DIR" ]; then
  git clone https://github.com/jy-yuan/KIVI.git "$KIVI_DIR"
fi

echo ">> Building KIVI CUDA kernel (kivi_gemv), ~2-5 min"
export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
export CPATH="$ENV_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib:${LIBRARY_PATH:-}"
export CC="$ENV_PREFIX/bin/gcc" CXX="$ENV_PREFIX/bin/g++" CUDAHOSTCXX="$ENV_PREFIX/bin/g++"
export TORCH_CUDA_ARCH_LIST="$GPU_CC"
export MAX_JOBS="${MAX_JOBS:-4}"
(cd "$KIVI_DIR/quant" && $PIP install -e . --no-build-isolation)

echo ">> Sanity check"
(cd "$KIVI_DIR" && $PY - <<'EOF'
import torch, transformers, flash_attn, kivi_gemv
from models.llama_kivi import LlamaForCausalLM_KIVI  # noqa: F401
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "flash_attn", flash_attn.__version__)
print("KIVI import OK")
EOF
)

cat > "$WORKDIR/env.sh" <<EOF
# source this before running the benchmark
export PATH="$ENV_PREFIX/bin:\$PATH"
export KIVI_DIR="$KIVI_DIR"
EOF
echo
echo ">> Done. Next:"
echo "   source $WORKDIR/env.sh"
echo "   huggingface-cli login     # token with access to meta-llama/Llama-3.1-8B-Instruct"
echo "   bash run_all.sh"
