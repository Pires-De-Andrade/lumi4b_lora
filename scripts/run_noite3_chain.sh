#!/usr/bin/env bash
# Fila da noite de 08/11 (versão 3, ~22:30). Substitui o run_noite_chain.sh, que não pode ser editado em execução. Cada etapa é retomável.
# Ordem (todas na 4090, uma de cada vez; a suíte do Lumi, só referência, fica por último):
#   1. RL `gspo_thor_e1` (a partir do `thor_pt_e1`, 4 ciclos): continua de onde parou, se parou.
#   2. Diagnóstico de log-probabilidades (scripts/diag_logprob.py): descompasso vLLM x treinador, ruído do corte do GSPO (eps 3e-4 e 4e-4)
#      e cabeça FP32, sobre as respostas do ciclo 0 do `gspo_thor_e1`; ~15 min.
#   3. SFT `adalberto_dft`: a receita do `adalberto` (data/train_adalberto.parquet, lr 1e-4, 1 época) trocando só a perda por `--loss dft`
#      (Dynamic Fine-Tuning); compara com o `adalberto` (62,3% em múltipla escolha com thinking); ~40 min com a avaliação.
#   4. RL `gspo1_T06`: a receita do `gspo1` (questões antigas, 2.048, T 0,6, `--ratio seq`, 4 ciclos, semente 0) com os logits divididos por 0,6
#      no treinador (GRPO_LOGIT_TEMP=0.6); compara com o `gspo1_c4` (67,8%), com os mesmos prompts; ~1,5 h.
#   5. RL `gspo1_s1`: a mesma receita do `gspo1` (T de logits 1,0, como antes) com a semente 1 (GRPO_SEED=1): mede a variação entre rodadas; ~1,5 h.
#   6. RL `gspo1_4k` (ramo B da regra de estudos/08, leitura em runs/jobs/noite.ramo): receita do `gspo1` com limite 4.096; ~1,6 h.
#   7. Suíte de teste do Lumi (o que falta; retomável; parar: touch runs/jobs/suite.stop).
# Uso: scripts/detach.sh noite3 scripts/run_noite3_chain.sh. Cancelar: tmux kill-session -t noite3 (e conferir os processos do vLLM).
# Estado: runs/jobs/noite3.status. NÃO editar com ele rodando.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

status=runs/jobs/noite3.status
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
nota_env() { mkdir -p "runs/grpo/$1"; echo "$2" >> "runs/grpo/$1/ENV.txt"; }  # registra variáveis de ambiente da rodada
RECEITA_GSPO1=(--init runs/train/adalberto/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5
               --max-new 2048 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq)

# Uma sessão `noite` antiga ou uma avaliação manual podem estar em curso: espera acabar.
while tmux has-session -t =noite 2>/dev/null || tmux has-session -t =eval-e1 2>/dev/null; do sleep 30; done  # '=' = nome exato (sem isso 'noite' casa com 'noite3')

# 1. RL a partir do thor_pt_e1
wait_gpu
say "INICIO RL gspo_thor_e1 (a partir do thor_pt_e1, questões antigas, 2.048, 4 ciclos)"
scripts/run_grpo.sh gspo_thor_e1 --init runs/train/thor_pt_e1/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
  --max-new 2048 --temperature 0.6 --top-p 0.95 --max-noclose 0.85 --ratio seq
say "FIM RL gspo_thor_e1 (código $?)"
atualizar_tabela

# 2. diagnóstico de log-probabilidades
if [ -s runs/diag/lp_report.json ]; then say "diagnóstico de log-probabilidades já feito: etapa pulada"; else
  wait_gpu
  say "INICIO diagnóstico de log-probabilidades (vLLM x treinador, ruído do corte, cabeça FP32)"
  mkdir -p runs/diag
  VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/diag_logprob.py vllm --model runs/merged/thor_pt_e1 \
    --rollouts runs/grpo/gspo_thor_e1/it000/rollouts.jsonl --out runs/diag/lp_vllm.json
  say "FIM diagnóstico, estágio vLLM (código $?)"
  wait_gpu
  PYTHONPATH=src python scripts/diag_logprob.py hf --adapter runs/train/thor_pt_e1/adapter \
    --rollouts runs/grpo/gspo_thor_e1/it000/rollouts.jsonl --vllm-json runs/diag/lp_vllm.json --out runs/diag/lp_report.json
  say "FIM diagnóstico, estágio HF (código $?)"
fi

# 3. SFT com a perda DFT
if avaliacao_completa adalberto_dft; then say "SFT adalberto_dft já treinado e avaliado: etapa pulada"; else
  wait_gpu
  say "INICIO SFT adalberto_dft (receita do adalberto, --loss dft)"
  scripts/run_experiment.sh adalberto_dft --arm lora --data data/train_adalberto.parquet --epochs 1 --lr 1e-4 --loss dft
  say "FIM SFT adalberto_dft (código $?)"
fi
atualizar_tabela

# 4. RL com a temperatura nos logits do treinador
wait_gpu
say "INICIO RL gspo1_T06 (receita do gspo1, logits do treinador / 0,6, semente 0, 4 ciclos)"
nota_env gspo1_T06 "GRPO_LOGIT_TEMP=0.6"
GRPO_LOGIT_TEMP=0.6 scripts/run_grpo.sh gspo1_T06 "${RECEITA_GSPO1[@]}"
say "FIM RL gspo1_T06 (código $?)"
atualizar_tabela

# 5. segunda semente da receita do gspo1
wait_gpu
say "INICIO RL gspo1_s1 (receita do gspo1, semente 1, 4 ciclos)"
nota_env gspo1_s1 "GRPO_SEED=1"
GRPO_SEED=1 scripts/run_grpo.sh gspo1_s1 "${RECEITA_GSPO1[@]}"
say "FIM RL gspo1_s1 (código $?)"
atualizar_tabela

# 6. ramo B: limite de 4.096 nas questões antigas
wait_gpu
say "INICIO RL gspo1_4k (receita do gspo1, questões antigas, 4.096, 4 ciclos)"
scripts/run_grpo.sh gspo1_4k --init runs/train/adalberto/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
  --max-new 4096 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq
say "FIM RL gspo1_4k (código $?)"
atualizar_tabela

# 7. suíte de teste do Lumi
wait_gpu
say "INICIO suíte de teste do Lumi (o que falta)"
scripts/run_suite_models.sh "base|-"
say "FIM da fila da noite (versão 3)"
