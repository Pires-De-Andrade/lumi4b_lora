#!/usr/bin/env bash
# Passa a cadeia do RL (H75, `rl_t6`) na frente da H69 na fila `noite18`, sem editar o script da fila (o bash lê o .sh aos poucos; editar com ele rodando quebra).
# A fila faz H67 (Lumi e T6-Lumi a 8.192 tokens) e depois começa a H69. Este vigia espera o status da fila registrar o INÍCIO da primeira avaliação da H69 e, nesse
# instante, encerra a sessão `noite18` e os processos dela (o vLLM da H69 ainda está subindo; a avaliação é retomável). Como a cadeia `rl_t6` espera a sessão
# `noite18` acabar, ela dispara sozinha em seguida. A H69, a H68 e o que falta do Nemotron voltam por `scripts/run_noite18_resto.sh`, depois do RL.
# Só mata processos da própria sessão tmux da fila (a árvore de processos dela); nada de outros usuários.
# Variáveis (para teste): SESSAO (padrão noite18), STATUS (padrão runs/jobs/noite18.status), PADRAO (padrão: início da H69).
# Uso: scripts/detach.sh reordena scripts/reordenar_h75_antes_h69.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SESSAO="${SESSAO:-noite18}"
STATUS="${STATUS:-runs/jobs/noite18.status}"
PADRAO="${PADRAO:-INICIO avaliação thor_pt_antidoom (}"   # texto literal (grep -F): "thor_pt_antidoom (" e não "thor_pt_antidoom_of (": só a H69
conta() { grep -cF -- "$PADRAO" "$STATUS" 2> /dev/null || true; }
arvore() { local p; for p in $(ps -o pid= --ppid "$1" 2> /dev/null); do echo "$p"; arvore "$p"; done; }

base=$(conta)
echo "$(date +%d/%m\ %H:%M) vigia ligado: espero o início da H69 em $STATUS (ocorrências agora: $base)"
while [ "$(conta)" -le "$base" ]; do
  tmux has-session -t "$SESSAO" 2> /dev/null || { echo "$(date +%d/%m\ %H:%M) a sessão $SESSAO acabou antes da H69; nada a fazer"; exit 0; }
  sleep 2
done
raiz=$(tmux list-panes -t "$SESSAO" -F '#{pane_pid}' 2> /dev/null | head -1)
pids="$raiz $(arvore "$raiz")"
echo "$(date +%d/%m\ %H:%M) a H69 começou: encerrando a sessão $SESSAO (processos: $pids)"
tmux kill-session -t "$SESSAO"
sleep 2
kill -TERM $pids 2> /dev/null
sleep 15
kill -KILL $pids 2> /dev/null
echo "$(date +%d/%m\ %H:%M) reordenação: a fila $SESSAO foi encerrada ao começar a H69; a cadeia do RL (H75) dispara agora e a H69, a H68 e o Nemotron voltam depois dela (scripts/run_noite18_resto.sh)" | tee -a "$STATUS"
for _ in $(seq 1 12); do
  usado=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' ')
  [ "${usado:-0}" -le 1024 ] && { echo "$(date +%d/%m\ %H:%M) GPU livre (${usado} MiB)"; exit 0; }
  sleep 5
done
echo "$(date +%d/%m\ %H:%M) AVISO: a GPU segue com ${usado:-?} MiB em uso; conferir os processos do vLLM (nvidia-smi); a cadeia do RL espera a GPU ficar livre"
