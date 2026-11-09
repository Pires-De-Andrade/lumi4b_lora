#!/usr/bin/env bash
# Médias de pesos (sopas) dos adaptadores de RL da receita do `gspo1`, todos a partir do mesmo SFT (`adalberto`), no ciclo 4 (it003). Só CPU; idempotente.
#   gspo1_soup2: gspo1 (semente 0) + gspo1_s1 (semente 1)
#   gspo1_soup3: gspo1_soup2 + gspo1_T06 (semente 0, logits do treinador / 0,6)
# Cada sopa vira um adaptador em runs/train/<nome>/adapter (com soup_report.json: distância de cada entrada à média e erro da aproximação
# média(B)·média(A) contra a média dos produtos B·A) e um modelo fundido em runs/merged/<nome>, pronto para a avaliação (eval_vllm.py).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

A0=runs/grpo/gspo1/it003/adapter; A1=runs/grpo/gspo1_s1/it003/adapter; A2=runs/grpo/gspo1_T06/it003/adapter
for a in "$A0" "$A1" "$A2"; do [ -f "$a/adapter_model.safetensors" ] || { echo "FALHA: falta $a"; exit 1; }; done

soup() {  # $1 = nome, resto = adaptadores
  local nome="$1"; shift
  if [ ! -f "runs/train/$nome/adapter/adapter_model.safetensors" ]; then
    PYTHONPATH=src python scripts/average_adapters.py --adapters "$@" --out "runs/train/$nome/adapter" || exit 1
  fi
  if [ ! -f "runs/merged/$nome/merge_report.json" ]; then
    PYTHONPATH=src python scripts/merge_adapter.py --adapter "runs/train/$nome/adapter" --out "runs/merged/$nome" || exit 1
  fi
}
soup gspo1_soup2 "$A0" "$A1"
soup gspo1_soup3 "$A0" "$A1" "$A2"
echo "sopas prontas: runs/merged/gspo1_soup2, runs/merged/gspo1_soup3"
