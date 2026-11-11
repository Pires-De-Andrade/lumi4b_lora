#!/usr/bin/env bash
# Retoma a fila `noite18` (H69, H68 e o que falta do Nemotron) depois da cadeia do RL (H75). Espera as sessões `noite18` e `rl_t6` acabarem e roda o mesmo script da
# fila: toda avaliação que já está completa é pulada ("já feita: pulada"), então só a H69 em diante roda. Se a ordem original não for alterada, nada muda (a fila
# termina e, depois do RL, isto só confere e pula). Cancelar: tmux kill-session -t noite18b (e conferir os processos do vLLM).
# Uso: scripts/detach.sh noite18b scripts/run_noite18_resto.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
while tmux has-session -t noite18 2> /dev/null || tmux has-session -t rl_t6 2> /dev/null; do sleep 60; done
echo "$(date +%d/%m\ %H:%M) retomada da fila depois da cadeia do RL (H75): H69, H68 e o Nemotron que falta" | tee -a runs/jobs/noite18.status
exec bash scripts/run_noite18_chain.sh
