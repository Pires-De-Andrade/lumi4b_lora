#!/usr/bin/env bash
# Cadeia de 08/11 (pedido do usuário, versão 2): a suíte antiga (scripts/run_suite.sh) termina o Lumi e para (suite.stop, pedido na hora de
# lançar a cadeia); aqui: (1) espera ela acabar; (2) suíte de teste do gspo1; (3) RL com gabarito oficial (receita do gspo1, só os prompts mudam).
# A suíte do gspo_thor ficou de fora por pedido do usuário (pode ser feita depois: scripts/run_suite_models.sh "gspo_thor|runs/grpo/gspo_thor/merged").
# Estado: runs/jobs/gold-chain.status. Uso: scripts/detach.sh gold-chain scripts/run_gold_chain.sh   (NÃO editar com ele rodando)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
status=runs/jobs/gold-chain.status
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }

# 1. espera a suíte antiga acabar (ela para depois do Lumi por causa do suite.stop) e a GPU ficar livre
while tmux has-session -t suite-queue 2>/dev/null; do sleep 20; done
say "suite-queue terminou"
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
  say "GPU ocupada, aguardando"; sleep 120
done

# 2. suíte de teste do gspo1 (o scripts/run_suite_models.sh apaga o suite.stop antigo ao começar)
say "INICIO suíte do gspo1"
scripts/run_suite_models.sh "gspo1|runs/grpo/gspo1/merged"
say "FIM suíte do gspo1 (código $?)"

# 3. RL com gabarito oficial: receita do gspo1 (mesmo ponto de partida e parâmetros), com os prompts de gabarito oficial; avaliação do ciclo 4
say "INICIO RL gspo_gold (prompts data/grpo_prompts_gold.jsonl)"
scripts/run_grpo.sh gspo_gold --init runs/train/adalberto/adapter --iters 8 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
  --max-new 2048 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq --prompts data/grpo_prompts_gold.jsonl
say "FIM RL gspo_gold (código $?)"
PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true
say "FIM da cadeia"
