#!/usr/bin/env bash
# Suíte de teste (a Nota de raciocínio do PRD: GSM8K-PT, MMLU-ProX PT, ENEM, BLUEX, OAB), nos dois modos, um modelo depois do outro.
# Retomável: eval_vllm.py pula as (tarefa, modo) que já estão no summary.json (o `base` já tem o gsm8k_test).
# Parar entre modelos: touch runs/jobs/suite.stop      Estado: runs/jobs/suite.status
# Uso: scripts/detach.sh suite-queue 'while tmux has-session -t cap-queue 2>/dev/null; do sleep 30; done; scripts/run_suite.sh'
# (NÃO editar este arquivo com a fila rodando.) Cada linha de MODELOS é NOME|modelo ("-" = o Lumi original do Hub).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
stop=runs/jobs/suite.stop; status=runs/jobs/suite.status
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }
MODELOS=(
  "base|-"
  "gspo_thor|runs/grpo/gspo_thor/merged"
  "gspo1|runs/grpo/gspo1/merged"
)
rm -f "$stop"
for item in "${MODELOS[@]}"; do
  name="${item%%|*}"; model="${item#*|}"
  [ -f "$stop" ] && { say "PARADA pedida ($stop) antes de $name"; break; }
  # a GPU é compartilhada: espera ficar livre (menos de 1 GiB em uso) em vez de estourar a memória
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
    [ -f "$stop" ] && break
    say "GPU ocupada, aguardando ($name)"; sleep 120
  done
  [ -f "$stop" ] && continue
  say "INICIO suíte $name | modelo $model"
  args=(--name "$name" --tasks suite)
  [ "$model" != "-" ] && args+=(--model "$model")
  VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/eval_vllm.py "${args[@]}"
  say "FIM suíte $name (código $?)"
  PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true
done
say "FIM da fila da suíte"
