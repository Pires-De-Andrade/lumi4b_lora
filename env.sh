# Carregar com: source env.sh  (mantém tudo dentro do projeto)
export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XDG_CACHE_HOME="$PROJ/.cache"
export HF_HOME="$PROJ/.cache/huggingface"
export UV_CACHE_DIR="$PROJ/.cache/uv"
export PIP_CACHE_DIR="$PROJ/.cache/pip"
export TORCH_HOME="$PROJ/.cache/torch"
export TRITON_CACHE_DIR="$PROJ/.cache/triton"
export MAMBA_ROOT_PREFIX="$PROJ/.micromamba"
export PATH="$PROJ/tools/mm/bin:$PATH"
# nvcc 13.0 (conda-forge), usado só para compilar mamba-ssm e causal-conv1d
export CUDA_HOME="$MAMBA_ROOT_PREFIX/envs/nvcc"
if [ -d "$CUDA_HOME/bin" ]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# headers e libs do toolkit ficam em targets/ no pacote conda
if [ -d "$CUDA_HOME/targets/x86_64-linux" ]; then
  export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
if [ -d "$PROJ/.venv" ]; then source "$PROJ/.venv/bin/activate"; fi
# evita que "uv run" sincronize e remova mamba-ssm/causal-conv1d, que ficam fora do lockfile
export UV_NO_SYNC=1
