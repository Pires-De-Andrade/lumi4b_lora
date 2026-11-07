#!/usr/bin/env bash
# Treina, funde e avalia na validação, em sequência. Uso (desacoplado):
#   scripts/detach.sh --gpu exp-NOME scripts/run_experiment.sh NOME --arm lora --data data/x.parquet --epochs 1 --lr 1e-4
# Saídas: runs/train/NOME, runs/merged/NOME, runs/eval/NOME (summary.json). Retomável: treino e avaliação retomam sozinhos.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
name="$1"; shift
PYTHONPATH=src python scripts/train_sft.py --out "runs/train/$name" "$@"
[ -f "runs/merged/$name/merge_report.json" ] || PYTHONPATH=src python scripts/merge_adapter.py --adapter "runs/train/$name/adapter" --out "runs/merged/$name"
VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/eval_vllm.py --name "$name" --model "runs/merged/$name" --tasks validation
echo "EXPERIMENTO $name COMPLETO"
