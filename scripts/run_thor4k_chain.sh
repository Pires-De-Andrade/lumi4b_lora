#!/usr/bin/env bash
# Triagem de 08/11 (~19:20): RL com gabarito oficial e limite de 4.096 a partir do `thor_pt` (SFT do raciocínio do Nemotron 30B
# traduzido), 4 ciclos. Mesma receita e mesmas questões do `gspo_gold4k`, só muda o ponto de partida (--init), então o ciclo 4 compara direto com o
# `gspo_gold4k` no ciclo 4 (e com o `gspo_thor` antigo, de 2.048 e questões antigas). A avaliação do final (modelo depois de 4 atualizações) equivale à
# "avaliação do modelo que entra na iteração 4". Motivo extra: o dataset do `adalberto` não tem licença declarada; se este ponto de partida empatar,
# é o caminho mais limpo para entregar. Roda com a configuração padrão do vLLM (igual à do `gspo_gold4k`), sem as mudanças do benchmark de geração.
# Espera a `gold-chain` (RL do gspo_gold4k) e o `bench-rollout` (benchmark de geração) acabarem. Estado: runs/jobs/thor4k.status.
# Uso: scripts/detach.sh thor4k scripts/run_thor4k_chain.sh   (NÃO editar com ele rodando; para cancelar: tmux kill-session -t thor4k, enquanto espera)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
status=runs/jobs/thor4k.status
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }

while tmux has-session -t gold-chain 2>/dev/null || tmux has-session -t bench-rollout 2>/dev/null; do sleep 30; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
  say "GPU ocupada, aguardando"; sleep 120
done

say "INICIO RL gspo_gold4k_thor (4 ciclos, --init thor_pt, --max-new 4096)"
scripts/run_grpo.sh gspo_gold4k_thor --init runs/train/thor_pt/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
  --max-new 4096 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq --prompts data/grpo_prompts_gold.jsonl
say "FIM RL gspo_gold4k_thor (código $?)"
PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true
say "FIM da cadeia thor4k"
