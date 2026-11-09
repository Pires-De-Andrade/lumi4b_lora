#!/usr/bin/env bash
# Fila de 09/11 (versão 6, depois da queda da máquina entre 02:45 e 12:48). Substitui o run_noite5_chain.sh. Cada etapa é retomável e pulada se já pronta.
# A 4090 é compartilhada: cada etapa espera a GPU ficar livre (menos de 1 GiB em uso) antes de começar.
# Ordem:
#   1. Avaliação do SFT parcial `qwen_mix_p` (já treinado e mesclado; a queda veio quando a avaliação começava). Sem RL: com o pacote completo
#      em mãos, o RL do parcial perde o valor.
#   2. Pacote 4 completo (data/thor4/thor4_out.tar.gz, 6.261 traços mantidos): SFT `qwen_mix` (dados do `adalberto` + MCQ do pacote), avaliação e RL
#      `gspo_qwen_mix` (scripts/run_qwen_mix.sh).
#   3. RL `gspo1_eps3x`, `gspo1_same`, `gspo1_4k` (diagnósticos da receita do `gspo1`).
#   4. Suíte de teste do Lumi até o fim (a ÚLTIMA de todas; retomável; parar: touch runs/jobs/suite.stop).
# Uso: scripts/detach.sh noite6 scripts/run_noite6_chain.sh. Cancelar: tmux kill-session -t noite6 (e conferir os processos do vLLM).
# Estado: runs/jobs/noite6.status. NÃO editar com ele rodando.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

status=runs/jobs/noite6.status
PARCIAL=data/thor4/parcial/thor4_parcial.tar.gz
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }
wait_gpu() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
    sleep 120
  done
}
atualizar_tabela() { PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true; }
nota_env() { mkdir -p "runs/grpo/$1"; echo "$2" >> "runs/grpo/$1/ENV.txt"; }  # registra as variáveis de ambiente da rodada
rl_gspo1() {  # $1 = nome; as variáveis de ambiente da rodada vêm do chamador
  wait_gpu
  say "INICIO RL $1 (receita do gspo1, 2.048, 4 ciclos)"
  scripts/run_grpo.sh "$1" --init runs/train/adalberto/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
    --max-new 2048 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq
  say "FIM RL $1 (código $?)"
  atualizar_tabela
}

# 1. avaliação do SFT parcial, sem RL
say "INICIO avaliação do SFT parcial qwen_mix_p (sem RL)"
NOME=qwen_mix_p PKG="$PARCIAL" EX=data/thor4/parcial/e1 PREP=data/thor4/parcial/prep MIN_KEPT=1500 SEM_RL=1 scripts/run_qwen_mix.sh
say "FIM avaliação do SFT parcial qwen_mix_p (código $?)"
atualizar_tabela

# 2. pacote 4 completo
say "INICIO trabalho com o pacote 4 completo (scripts/run_qwen_mix.sh)"
scripts/run_qwen_mix.sh
say "FIM trabalho com o pacote 4 completo (código $?)"
atualizar_tabela

# 3. diagnósticos de RL
nota_env gspo1_eps3x "GRPO_EPS_LOW=1e-3 GRPO_EPS_HIGH=1.3e-3"; GRPO_EPS_LOW=1e-3 GRPO_EPS_HIGH=1.3e-3 rl_gspo1 gspo1_eps3x
nota_env gspo1_same "GRPO_OLD_SAME_BATCHES=1";                 GRPO_OLD_SAME_BATCHES=1 rl_gspo1 gspo1_same
wait_gpu
say "INICIO RL gspo1_4k (receita do gspo1, questões antigas, 4.096, 4 ciclos)"
scripts/run_grpo.sh gspo1_4k --init runs/train/adalberto/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
  --max-new 4096 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq
say "FIM RL gspo1_4k (código $?)"
atualizar_tabela

# 4. suíte de teste do Lumi, até o fim
wait_gpu
say "INICIO suíte de teste do Lumi (o que falta)"
scripts/run_suite_models.sh "base|-"
say "FIM da fila (versão 6)"
