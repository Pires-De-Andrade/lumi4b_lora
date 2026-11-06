#!/usr/bin/env bash
# Compila causal-conv1d e mamba-ssm contra o PyTorch do .venv, só com nvcc do conda-forge (sem sudo).
# Pré-requisitos: source env.sh; nvcc 13.0 em .micromamba/envs/nvcc; torch instalado no .venv.
# Uso recomendado (desacoplado): scripts/detach.sh build-kernels scripts/build_kernels.sh
#
# mamba-ssm usa uma cópia local patchada (scripts/patches/mamba-ssm-sm89-cxx20.patch):
#   - compila só para sm_89 (RTX 4090); em outra GPU, ajustar o patch e recompilar;
#   - troca -std=c++17 por -std=c++20, exigido pelos headers do PyTorch 2.14.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

CC_VER=1.7.0
MAMBA_VER=2.3.2.post1
SRC=.cache/src/mamba-ssm

command -v nvcc >/dev/null || { echo "nvcc não encontrado: veja env.sh e .micromamba/envs/nvcc" >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available()' || { echo "PyTorch sem CUDA" >&2; exit 1; }

export MAX_JOBS="${MAX_JOBS:-12}" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE

echo "[1/2] causal-conv1d $CC_VER"
uv pip install -v --no-build-isolation --no-deps "causal-conv1d==$CC_VER"

echo "[2/2] mamba-ssm $MAMBA_VER (patchado)"
rm -rf "$SRC"; mkdir -p "$SRC"
url=$(curl -fsS "https://pypi.org/pypi/mamba-ssm/$MAMBA_VER/json" \
  | python -I -c "import json,sys;print([x['url'] for x in json.load(sys.stdin)['urls'] if x['packagetype']=='sdist'][0])")
curl -fsSL "$url" -o "$SRC/src.tar.gz"
tar -xzf "$SRC/src.tar.gz" -C "$SRC" --no-same-owner
dir="$SRC/mamba_ssm-$MAMBA_VER"
patch -p1 -d "$dir" < scripts/patches/mamba-ssm-sm89-cxx20.patch
uv pip install -v --no-build-isolation --no-deps "$PWD/$dir"

python -c 'import mamba_ssm, causal_conv1d; print("import ok", mamba_ssm.__version__, causal_conv1d.__version__)'
