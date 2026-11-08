#!/usr/bin/env bash
# Cadeia de 08/11 (versão 4, 18:05): RL com gabarito oficial e limite de geração de 4.096 tokens (o mesmo da avaliação).
# O `gspo_gold` com 2.048 foi interrompido no ciclo 2 (pasta runs/grpo/gspo_gold): 36% das respostas de múltipla escolha
# truncavam (52% na WebInstruct) e respostas corretas e longas valiam 0. Receita igual à do `gspo1` (ponto de partida `adalberto`, 8 ciclos,
# T 0,6, top-p 0,95, --ratio seq, lr 2e-5); muda o conjunto de questões E o limite (--max-new 4096). Nome novo: `gspo_gold4k`.
# A suíte do gspo1 NÃO está aqui: roda em outra sessão (suite-gspo1), que espera esta terminar (assim dá para mudá-la sem derrubar o RL).
# O teste rápido do registro (smoke_diag) passou em 08/11 17:20, e o de memória com 4.096 (smoke_4k) roda antes de lançar esta cadeia.
# Estado: runs/jobs/gold-chain.status. Uso: scripts/detach.sh gold-chain scripts/run_gold_chain.sh   (NÃO editar com ele rodando)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
status=runs/jobs/gold-chain.status
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }

while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
  say "GPU ocupada, aguardando"; sleep 120
done

say "INICIO RL gspo_gold4k (prompts data/grpo_prompts_gold.jsonl, --max-new 4096)"
scripts/run_grpo.sh gspo_gold4k --init runs/train/adalberto/adapter --iters 8 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
  --max-new 4096 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq --prompts data/grpo_prompts_gold.jsonl
say "FIM RL gspo_gold4k (código $?)"
PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true
say "FIM da cadeia"
