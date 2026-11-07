#!/usr/bin/env bash
# Instala o vLLM num ambiente separado (.venv-vllm), sem tocar no .venv do projeto.
# Motivo: o vLLM 0.33.0 fixa torch==2.14.0 e transformers<5.21, e o .venv tem torch 2.14.1, transformers 5.22.2
# e kernels (mamba-ssm, causal-conv1d) compilados contra ele. O torch 2.14.0 do PyPI é CUDA 13.0.3, que o driver 580 aceita.
# Uso: scripts/detach.sh install-vllm scripts/install_vllm.sh
# Rodar o vLLM:  VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/run_gsm8k_vllm.py ...
#   (VLLM_USE_FLASHINFER_SAMPLER=0: o kernel de amostragem do FlashInfer é compilado na hora com o nvcc do conda,
#    que não tem curand.h; o amostrador do PyTorch serve para greedy e para temperatura/top-p.)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
uv venv .venv-vllm --python 3.12
uv pip install --python .venv-vllm/bin/python vllm==0.33.0 datasets
.venv-vllm/bin/python -c 'import torch, vllm, transformers; print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__, "| transformers", transformers.__version__, "| gpu:", torch.cuda.is_available())'
