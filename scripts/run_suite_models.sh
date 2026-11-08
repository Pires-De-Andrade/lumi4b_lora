#!/usr/bin/env bash
# Suíte de teste (a Nota de raciocínio do PRD: GSM8K-PT, MMLU-ProX PT, ENEM, BLUEX, OAB), nos dois modos, para os modelos dados nos
# argumentos, um depois do outro. Cada argumento é NOME|modelo ("-" = o Lumi original do Hub). Retomável: eval_vllm.py pula as
# (tarefa, modo) que já estão em runs/eval/NOME/summary.json. Parar entre modelos: touch runs/jobs/suite.stop. Estado: runs/jobs/suite.status.
# Uso: scripts/run_suite_models.sh "gspo1|runs/grpo/gspo1/merged" "gspo_thor|runs/grpo/gspo_thor/merged"
# (Substitui, para novos pedidos, o scripts/run_suite.sh, que tem a lista fixa; NÃO editar este arquivo com ele rodando.)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
stop=runs/jobs/suite.stop; status=runs/jobs/suite.status
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }
rm -f "$stop"
for item in "$@"; do
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
