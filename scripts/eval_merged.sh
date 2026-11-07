#!/usr/bin/env bash
# Avalia modelos já fundidos (runs/merged/NOME) na validação, um depois do outro. Retomável (eval_vllm.py pula o que já fez).
# Uso (desacoplado): scripts/detach.sh --gpu eval-x scripts/eval_merged.sh NOME [NOME...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
for name in "$@"; do
  VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/eval_vllm.py --name "$name" --model "runs/merged/$name" --tasks validation
done
echo "AVALIACAO COMPLETA: $*"
