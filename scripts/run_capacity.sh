#!/usr/bin/env bash
# Fila do estudo de capacidade, tipo de bloco e posição (estudos/03). Lê scripts/capacity_plan.txt UMA LINHA POR VEZ
# (NOME|argumentos do train_sft.py), então o plano pode ser reordenado ou estendido sem editar este script.
# Parar entre rodadas: touch runs/jobs/cap.stop    Estado: runs/jobs/cap.status
# Uso: scripts/detach.sh --gpu cap-queue scripts/run_capacity.sh        (NÃO editar este arquivo com a fila rodando)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
plan=scripts/capacity_plan.txt; stop=runs/jobs/cap.stop; status=runs/jobs/cap.status
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }
complete() {  # a rodada já tem os quatro blocos da validação?
  python - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(f"runs/eval/{sys.argv[1]}/summary.json"))
except Exception:
    sys.exit(1)
sys.exit(0 if all(k in d for k in ("gsm8k_val|thinking", "gsm8k_val|nothinking", "mcq_val|thinking", "mcq_val|nothinking")) else 1)
PY
}
i=0
while true; do
  i=$((i + 1))
  [ -f "$stop" ] && { say "PARADA pedida ($stop)"; break; }
  [ "$i" -gt "$(wc -l < "$plan")" ] && { say "FIM do plano"; break; }
  line="$(sed -n "${i}p" "$plan")"
  case "$line" in ''|'#'*) continue ;; esac
  name="${line%%|*}"; args="${line#*|}"
  if complete "$name"; then say "PULO $name (já completa)"; continue; fi
  # a GPU é compartilhada: espera ficar livre (menos de 1 GiB em uso) em vez de estourar a memória
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
    [ -f "$stop" ] && break
    say "GPU ocupada, aguardando ($name)"; sleep 120
  done
  [ -f "$stop" ] && continue
  say "INICIO $name | $args"
  eval "scripts/run_experiment.sh '$name' --arm lora --data data/train_adalberto.parquet --epochs 1 --lr 1e-4 $args"
  say "FIM $name (código $?)"
  PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true
done
