#!/usr/bin/env bash
# Vigia: espera a (prova, modo) aparecer no summary.json de uma avaliação em curso (scripts/eval_vllm.py) e então encerra o processo dela.
# O eval_vllm.py grava o summary.json por último (depois do .jsonl e do _scores.json), então o resultado dessa prova já está inteiro em disco;
# o que vier depois (o modo seguinte) é perdido, e a retomada pula o que está no summary.json.
# Uso: scripts/stop_eval_after_task.sh <pid do eval_vllm.py> <summary.json> <prova> <modo>
# Exemplo (08/11 17:05, pedido do usuário: liberar a GPU para o gspo_gold):
#   scripts/detach.sh lumi-stop scripts/stop_eval_after_task.sh 440452 runs/eval/base/summary.json mmlu_prox_test thinking
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
pid="$1"; summ="$2"; key="$3|$4"
say() { echo "$(date +%d/%m\ %H:%M:%S) $*"; }
grep -q "eval_vllm.py" "/proc/$pid/cmdline" 2>/dev/null || { say "PID $pid não é um eval_vllm.py em execução: nada a vigiar"; exit 1; }
say "vigiando $summ pela chave '$key' (pid $pid)"
while kill -0 "$pid" 2>/dev/null; do
  if python3 -c "import json,sys; sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)" "$summ" "$key" 2>/dev/null; then
    sleep 5  # folga para o resto do processo (o summary já está completo)
    say "'$key' gravado: encerrando o pid $pid"
    # só os descendentes DESTE processo (a GPU é compartilhada: nada de pkill por nome)
    kids="$(pstree -p "$pid" | grep -o '([0-9]\+)' | tr -d '()' | grep -vx "$pid" | tr '\n' ' ')"
    kill -TERM "$pid"
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { say "pid $pid não saiu em 60 s: SIGKILL"; kill -KILL "$pid"; }
    sleep 5
    for k in $kids; do kill -0 "$k" 2>/dev/null && { say "descendente $k ainda vivo: SIGKILL"; kill -KILL "$k" 2>/dev/null; }; done
    say "FIM: processo encerrado (VRAM em uso agora: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1))"
    exit 0
  fi
  sleep 10
done
say "o pid $pid terminou sozinho antes da chave '$key' aparecer"
