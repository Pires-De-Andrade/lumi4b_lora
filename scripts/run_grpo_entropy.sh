#!/usr/bin/env bash
# Mede a entropia do modelo nas respostas de cada ciclo de um GRPO já rodado (para ver se ela sobe).
# Para o ciclo k, o modelo que gerou as respostas é o adaptador do ciclo k-1 (o do SFT no ciclo 0).
# Uso (desacoplado): scripts/detach.sh --gpu grpo-ent scripts/run_grpo_entropy.sh NOME --init runs/train/adalberto/adapter [--max-samples 512]
# Saída: runs/grpo/NOME/entropy/itNNN/entropy.json e a tabela no fim.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
name="$1"; shift
init=""; maxs=512
while [ $# -gt 0 ]; do
  case "$1" in --init) init="$2"; shift 2;; --max-samples) maxs="$2"; shift 2;; *) echo "argumento desconhecido: $1" >&2; exit 2;; esac
done
[ -n "$init" ] || { echo "faltou --init" >&2; exit 2; }
root="runs/grpo/$name"; cur="$init"
for d in "$root"/it[0-9][0-9][0-9]; do
  [ -f "$d/rollouts.jsonl" ] || continue
  o="$root/entropy/$(basename "$d")"
  if [ ! -f "$o/entropy.json" ]; then
    echo "=== $(basename "$d"): entropia com $cur"
    PYTHONPATH=src python scripts/grpo_train.py --rollouts "$d/rollouts.jsonl" --adapter-in "$cur" --out "$o" --entropy-only --max-samples "$maxs" | grep -E "entropy|EXIT_OK"
  fi
  [ -f "$d/adapter/adapter_config.json" ] && cur="$d/adapter"
done
PYTHONPATH=src python - "$root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print("ciclo | entropia (nats/token) todas / mcq / math | logp média | amostras")
for f in sorted((root / "entropy").glob("it*/entropy.json")):
    j = json.loads(f.read_text()); e = j["entropy"]
    print(f"{f.parent.name} | {e['all']} / {e.get('mcq')} / {e.get('math')} | {e['mean_logp']} | {j.get('samples_used', j['samples'])}")
PY
