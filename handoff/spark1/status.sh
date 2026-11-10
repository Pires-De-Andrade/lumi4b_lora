#!/usr/bin/env bash
# Mostra o andamento da fila da Spark: últimas linhas do status, avaliações prontas, GPU e o arquivo de devolução.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# a Spark pode não ter pgrep, ps nem curl: a fila é procurada em /proc e o Hugging Face é testado pelo python
fila_rodando() {
  local f
  for f in /proc/[0-9]*/cmdline; do
    tr '\0' ' ' < "$f" 2> /dev/null | grep -q "[h]andoff/spark1/run_spark1.sh" && return 0
  done
  return 1
}
echo "== fila: $(fila_rodando && echo rodando || echo parada ou terminada)"
tail -n 8 runs/jobs/spark1.status 2>/dev/null || echo "(sem status ainda)"
echo "== avaliações prontas (tarefa|modo)"
python3 - <<'PY'
import glob, json
for f in sorted(glob.glob("runs/eval*/*/summary.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    print(f.split("/")[-2], ":", ", ".join(sorted(k for k in d if "|" in k)) or "-")
PY
echo "== GPU"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1 | head -1
echo "== devolução"; ls -l spark1_saida.tar.gz 2>/dev/null && cat spark1_saida.sha256 || echo "(ainda não há spark1_saida.tar.gz)"
