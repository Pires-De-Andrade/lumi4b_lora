#!/usr/bin/env bash
# Traduz o raciocínio das questões de ciência da NVIDIA (português do Brasil), monta a mistura com os dados do adalberto
# (matemática) e treina/avalia `nemo_mix`. Uso (desacoplado): scripts/detach.sh --gpu nemo-mix scripts/run_nemo_mix.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
VL="VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python"
[ -f data/nemotron_pt_mcq_8k.jsonl ] || PYTHONPATH=src python scripts/prepare_nemotron_pt.py --n 8000 --out data/nemotron_pt_mcq_8k.jsonl
env $VL scripts/translate_traces.py --src data/nemotron_pt_mcq_8k.jsonl --n 8000 --out data/nemo_pt
PYTHONPATH=src python scripts/filter_translated.py data/nemo_pt
PYTHONPATH=src python scripts/build_teacher_dataset.py --think data/adalberto_think.jsonl data/nemo_pt/think_clean.jsonl \
  --nothink data/adalberto_nothink.jsonl data/nemo_pt/nothink_clean.jsonl --split-sources adalberto \
  --max-think 11500 --max-nothink 7000 --max-len 2048 --out data/train_nemo_mix.parquet
scripts/run_experiment.sh nemo_mix --arm lora --data data/train_nemo_mix.parquet --epochs 1 --lr 1e-4
echo "NEMO_MIX COMPLETO"
