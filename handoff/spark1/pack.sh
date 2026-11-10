#!/usr/bin/env bash
# Monta o pacote da Spark (rodar na 4090, na máquina que tem o modelo fundido): um arquivo só, handoff/spark1_pacote.tar.gz (~2 GB) + .sha256.
# Dentro: código, provas novas, scripts, os arquivos pequenos do T6-Lumi e handoff/spark1/t6_delta.bin, a diferença exata entre os pesos do T6-Lumi e os do
# Lumi (72 dos 263 tensores, comprimida; `delta.py`). handoff/spark1/PESOS.sha256 é o sha256 do model.safetensors do T6-Lumi (7.947.142.672 bytes), que o
# `rodar.sh` confere depois de reconstruir. Só o T6-Lumi vai; as variantes (H69), 8.192 tokens (H67) e amostragem (H68) ficam para a 4090.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
M=thor_pt_antidoom_of
source env.sh
LUMI=$(ls .cache/huggingface/hub/models--ceia-br--lumi-4b/snapshots/*/model.safetensors | head -1)
[ -s "runs/merged/$M/model.safetensors" ] || { echo "falta runs/merged/$M/model.safetensors" >&2; exit 1; }
sha256sum "runs/merged/$M/model.safetensors" > handoff/spark1/PESOS.sha256
[ handoff/spark1/t6_delta.bin -nt "runs/merged/$M/model.safetensors" ] || python handoff/spark1/delta.py pack --base "$LUMI" --model "runs/merged/$M/model.safetensors" --out handoff/spark1/t6_delta.bin
tar -I 'gzip -1' -cf handoff/spark1_pacote.tar.gz --exclude=__pycache__ --exclude='*.pyc' --exclude=model.safetensors \
  src scripts/eval_vllm.py data/novas_provas "runs/merged/$M" \
  handoff/spark1/README.md handoff/spark1/rodar.sh handoff/spark1/status.sh handoff/spark1/run_spark1.sh handoff/spark1/ambiente.sh handoff/spark1/compilador.sh \
  handoff/spark1/delta.py handoff/spark1/t6_delta.bin handoff/spark1/PESOS.sha256
(cd handoff && sha256sum spark1_pacote.tar.gz > spark1_pacote.sha256)
echo "gerado: handoff/spark1_pacote.tar.gz ($(du -h handoff/spark1_pacote.tar.gz | cut -f1))"; cat handoff/spark1_pacote.sha256
