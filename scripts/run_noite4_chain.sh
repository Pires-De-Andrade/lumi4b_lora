#!/usr/bin/env bash
# Fila da noite de 08/11 (versão 4, ~23:25). Substitui o run_noite3_chain.sh (não pode ser editado em execução). Cada etapa é retomável e pulada se já pronta.
# Ordem (todas na 4090, uma de cada vez). A suíte de teste do Lumi é a ÚLTIMA de todas: ela só ocupa a GPU nos intervalos e sai da frente quando o
# pacote 4 da Thor (traços de múltipla escolha do Qwen) chega, voltando depois que o trabalho com esse pacote acabar.
#   1. SFT `adalberto_dft` (perda DFT; receita do `adalberto`)                          [já em curso na versão 3: pulado se avaliado]
#   2. RL `gspo1_T06`: receita do `gspo1`, logits do treinador / 0,6 (GRPO_LOGIT_TEMP=0.6), semente 0, 4 ciclos
#   3. RL `gspo1_s1`: receita do `gspo1`, semente 1 (GRPO_SEED=1): variação entre rodadas
#   4. RL `gspo1_eps3x`: receita do `gspo1` com o corte do GSPO 3x maior (GRPO_EPS_LOW=1e-3, GRPO_EPS_HIGH=1.3e-3)
#   5. RL `gspo1_same`: receita do `gspo1` com as log-probabilidades antigas nos mesmos micro-batches do treino (GRPO_OLD_SAME_BATCHES=1)
#   6. RL `gspo1_4k`: receita do `gspo1` com limite 4.096 (ramo B da regra de estudos/08)
#   7. Intervalo até o pacote 4 chegar (data/thor4/thor4_out.tar.gz): suíte de teste do Lumi em segundo plano; quando o arquivo aparece e para de
#      mudar, a suíte é interrompida (é retomável por (prova, modo)) e roda scripts/run_qwen_mix.sh (se existir)
#   8. Suíte de teste do Lumi até o fim (retomável; parar: touch runs/jobs/suite.stop)
# Todas as rodadas de RL das etapas 2 a 6 usam a mesma receita e os mesmos prompts do `gspo1_c4`; só muda o que está na descrição.
# Uso: scripts/detach.sh noite4 scripts/run_noite4_chain.sh. Cancelar: tmux kill-session -t noite4 (e conferir os processos do vLLM).
# Estado: runs/jobs/noite4.status. NÃO editar com ele rodando.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

status=runs/jobs/noite4.status
PACOTE4=data/thor4/thor4_out.tar.gz
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }
wait_gpu() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
    say "GPU ocupada, aguardando"; sleep 120
  done
}
avaliacao_completa() {  # $1 = nome em runs/eval: as 4 avaliações da validação estão no summary.json?
  python3 - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(f"runs/eval/{sys.argv[1]}/summary.json"))
    assert all(f"{t}|{m}" in d for t in ("gsm8k_val", "mcq_val") for m in ("thinking", "nothinking"))
except Exception:
    sys.exit(1)
PY
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

# Uma sessão `noite3` antiga pode estar em curso: espera acabar (nome exato).
while tmux has-session -t =noite3 2>/dev/null; do sleep 30; done

# 1. SFT com a perda DFT
if avaliacao_completa adalberto_dft; then say "SFT adalberto_dft já treinado e avaliado: etapa pulada"; else
  wait_gpu
  say "INICIO SFT adalberto_dft (receita do adalberto, --loss dft)"
  scripts/run_experiment.sh adalberto_dft --arm lora --data data/train_adalberto.parquet --epochs 1 --lr 1e-4 --loss dft
  say "FIM SFT adalberto_dft (código $?)"
fi
atualizar_tabela

# 2 a 5. RLs da receita do gspo1 com uma mudança de cada vez
nota_env gspo1_T06 "GRPO_LOGIT_TEMP=0.6";                         GRPO_LOGIT_TEMP=0.6 rl_gspo1 gspo1_T06
nota_env gspo1_s1 "GRPO_SEED=1";                                   GRPO_SEED=1 rl_gspo1 gspo1_s1
nota_env gspo1_eps3x "GRPO_EPS_LOW=1e-3 GRPO_EPS_HIGH=1.3e-3";     GRPO_EPS_LOW=1e-3 GRPO_EPS_HIGH=1.3e-3 rl_gspo1 gspo1_eps3x
nota_env gspo1_same "GRPO_OLD_SAME_BATCHES=1";                     GRPO_OLD_SAME_BATCHES=1 rl_gspo1 gspo1_same

# 6. ramo B: limite de 4.096 nas questões antigas
wait_gpu
say "INICIO RL gspo1_4k (receita do gspo1, questões antigas, 4.096, 4 ciclos)"
scripts/run_grpo.sh gspo1_4k --init runs/train/adalberto/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
  --max-new 4096 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq
say "FIM RL gspo1_4k (código $?)"
atualizar_tabela

# 7. intervalo até o pacote 4: a suíte do Lumi enche a GPU e sai da frente quando o pacote chega
if ! pacote4_pronto; then
  wait_gpu
  say "INICIO suíte de teste do Lumi como preenchimento (sai da frente quando $PACOTE4 chegar)"
  setsid scripts/run_suite_models.sh "base|-" > runs/jobs/noite4.suite.log 2>&1 &
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
say "pacote 4 pronto: $PACOTE4"
if [ -x scripts/run_qwen_mix.sh ]; then
  wait_gpu
  say "INICIO trabalho com o pacote 4 (scripts/run_qwen_mix.sh)"
  scripts/run_qwen_mix.sh
  say "FIM trabalho com o pacote 4 (código $?)"
  atualizar_tabela
else
  say "scripts/run_qwen_mix.sh não existe ou não é executável: sem trabalho com o pacote 4"
fi

# 8. suíte de teste do Lumi, até o fim
wait_gpu
say "INICIO suíte de teste do Lumi (o que falta)"
scripts/run_suite_models.sh "base|-"
say "FIM da fila da noite (versão 4)"
