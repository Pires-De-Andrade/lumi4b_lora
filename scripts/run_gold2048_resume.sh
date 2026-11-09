#!/usr/bin/env bash
# Replanejamento de 08/11 (~19:55), depois da regra de parada do ciclo 4 do `gspo_gold4k` (múlt. escolha 63,1%, fecha 83,9%):
# retoma o `gspo_gold` (limite de 2.048, mesmas questões do gold; ciclos 0 e 1 prontos desde 17:20-17:59) até o ciclo 4 e avalia. Compara direto com o
# `gspo_gold4k` no ciclo 4 (só muda o limite de geração) e com o `gspo1_c4` (só mudam as questões). Receita idêntica à original, com --iters 4: o
# final do script funde o adaptador depois de 4 atualizações e avalia a validação (equivale ao "modelo que entra na iteração 4").
# Espera o `bench-rollout` acabar. Estado: runs/jobs/gold2048.status. Uso: scripts/detach.sh gold2048 scripts/run_gold2048_resume.sh
# (NÃO editar com ele rodando; cancelar enquanto espera: tmux kill-session -t gold2048)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
status=runs/jobs/gold2048.status
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }

while tmux has-session -t bench-rollout 2>/dev/null; do sleep 30; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
  say "GPU ocupada, aguardando"; sleep 120
done

say "INICIO RL gspo_gold retomado (2.048, ciclos 2 e 3, depois avaliação do ciclo 4)"
scripts/run_grpo.sh gspo_gold --init runs/train/adalberto/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
  --max-new 2048 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq --prompts data/grpo_prompts_gold.jsonl
say "FIM RL gspo_gold retomado (código $?)"
PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true
say "FIM da cadeia gold2048"
