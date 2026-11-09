#!/usr/bin/env bash
# Fila da noite de 09/11 (versão 5). Substitui o run_noite4_chain.sh a partir do fim do `gspo1_s1`. Cada etapa é retomável e pulada se já pronta.
# Motivo: a entrega 2 do pacote 4 da Thor (traços de múltipla escolha do Qwen) é usada num teste parcial antes dos diagnósticos de RL.
# Ordem (todas na 4090, uma de cada vez; a suíte de teste do Lumi é a ÚLTIMA de todas):
#   0. Espera o `gspo1_s1` (em curso na fila noite4, congelada com SIGSTOP) terminar e encerra a noite4.
#   1. Teste parcial `qwen_mix_p`: SFT com a entrega parcial do pacote 4 (data/thor4/parcial/thor4_parcial.tar.gz) + a matemática do `adalberto`,
#      avaliação na validação e RL `gspo_qwen_mix_p` (scripts/run_qwen_mix.sh com NOME=qwen_mix_p).
#   2. RL `gspo1_eps3x`, `gspo1_same`, `gspo1_4k` (diagnósticos da receita do `gspo1`, como na noite4). Antes de cada um, se o pacote 4 final
#      (data/thor4/thor4_out.tar.gz, parado há mais de 60 s) já chegou, o trabalho com o pacote final (scripts/run_qwen_mix.sh) roda primeiro.
#   3. Se o pacote final ainda não chegou: suíte de teste do Lumi como preenchimento, interrompida quando ele chegar (retomável); depois o trabalho
#      com o pacote final.
#   4. Suíte de teste do Lumi até o fim (retomável; parar: touch runs/jobs/suite.stop).
# Só o pacote final vai para data/thor4/thor4_out.tar.gz; entregas parciais ficam em data/thor4/parcial/.
# Uso: scripts/detach.sh noite5 scripts/run_noite5_chain.sh. Cancelar: tmux kill-session -t noite5 (e conferir os processos do vLLM).
# Estado: runs/jobs/noite5.status. NÃO editar com ele rodando.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

status=runs/jobs/noite5.status
PACOTE4=data/thor4/thor4_out.tar.gz
PARCIAL=data/thor4/parcial/thor4_parcial.tar.gz
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }
wait_gpu() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
    say "GPU ocupada, aguardando"; sleep 120
  done
}
atualizar_tabela() { PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true; }
nota_env() { mkdir -p "runs/grpo/$1"; echo "$2" >> "runs/grpo/$1/ENV.txt"; }  # registra as variáveis de ambiente da rodada
pacote4_pronto() {  # o arquivo existe e não muda há mais de 60 s (scp terminado)
  [ -f "$PACOTE4" ] && [ $(( $(date +%s) - $(stat -c %Y "$PACOTE4") )) -gt 60 ]
}
rl_gspo1() {  # $1 = nome; as variáveis de ambiente da rodada vêm do chamador
  wait_gpu
  say "INICIO RL $1 (receita do gspo1, 2.048, 4 ciclos)"
  scripts/run_grpo.sh "$1" --init runs/train/adalberto/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
    --max-new 2048 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq
  say "FIM RL $1 (código $?)"
  atualizar_tabela
}
FINAL_FEITO=0
pacote_final() {  # roda o trabalho com o pacote final, uma vez, se ele já chegou
  [ "$FINAL_FEITO" = 1 ] && return
  pacote4_pronto || return
  FINAL_FEITO=1
  wait_gpu
  say "INICIO trabalho com o pacote 4 final (scripts/run_qwen_mix.sh)"
  scripts/run_qwen_mix.sh
  say "FIM trabalho com o pacote 4 final (código $?)"
  atualizar_tabela
}

# 0. espera o gspo1_s1 da noite4 (congelada) e encerra a noite4
while pgrep -f "scripts/run_grpo.sh gspo1_s1 --init" > /dev/null; do sleep 20; done
say "gspo1_s1 terminou: encerrando a fila noite4 (congelada)"
pkill -KILL -f "scripts/run_noite4_chain.sh" 2>/dev/null; sleep 2
tmux kill-session -t =noite4 2>/dev/null
atualizar_tabela

# 1. teste parcial com a entrega do pacote 4
say "INICIO teste parcial com o pacote 4 (qwen_mix_p)"
NOME=qwen_mix_p PKG="$PARCIAL" EX=data/thor4/parcial/e1 PREP=data/thor4/parcial/prep MIN_KEPT=1500 scripts/run_qwen_mix.sh
say "FIM teste parcial com o pacote 4 (código $?)"
atualizar_tabela

# 2. diagnósticos de RL (o pacote final entra na frente assim que chegar)
pacote_final
nota_env gspo1_eps3x "GRPO_EPS_LOW=1e-3 GRPO_EPS_HIGH=1.3e-3"; GRPO_EPS_LOW=1e-3 GRPO_EPS_HIGH=1.3e-3 rl_gspo1 gspo1_eps3x
pacote_final
nota_env gspo1_same "GRPO_OLD_SAME_BATCHES=1";                 GRPO_OLD_SAME_BATCHES=1 rl_gspo1 gspo1_same
pacote_final
wait_gpu
say "INICIO RL gspo1_4k (receita do gspo1, questões antigas, 4.096, 4 ciclos)"
scripts/run_grpo.sh gspo1_4k --init runs/train/adalberto/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
  --max-new 4096 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq
say "FIM RL gspo1_4k (código $?)"
atualizar_tabela

# 3. intervalo até o pacote final: a suíte do Lumi enche a GPU e sai da frente quando o pacote chega
if [ "$FINAL_FEITO" = 0 ] && ! pacote4_pronto; then
  wait_gpu
  say "INICIO suíte de teste do Lumi como preenchimento (sai da frente quando $PACOTE4 chegar)"
  setsid scripts/run_suite_models.sh "base|-" > runs/jobs/noite5.suite.log 2>&1 &
  SUITE_PID=$!
  while ! pacote4_pronto; do
    if ! kill -0 "$SUITE_PID" 2>/dev/null; then say "a suíte do Lumi terminou antes do pacote 4; aguardando o pacote"; break; fi
    sleep 30
  done
  if kill -0 "$SUITE_PID" 2>/dev/null; then
    say "pacote 4 chegou: interrompendo a suíte do Lumi (retomável)"
    kill -TERM -- "-$SUITE_PID" 2>/dev/null; sleep 5; kill -KILL -- "-$SUITE_PID" 2>/dev/null
    pkill -f "scripts/eval_vllm.py --name base" 2>/dev/null; sleep 10
  fi
  while ! pacote4_pronto; do sleep 60; done
fi
pacote_final

# 4. suíte de teste do Lumi, até o fim
wait_gpu
say "INICIO suíte de teste do Lumi (o que falta)"
scripts/run_suite_models.sh "base|-"
say "FIM da fila da noite (versão 5)"
