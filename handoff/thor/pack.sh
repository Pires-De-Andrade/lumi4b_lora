#!/usr/bin/env bash
# Filtra os traços e empacota a entrega em thor_out.tar.gz. Pode rodar com a geração em andamento (entrega parcial).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -f out/traces.jsonl ] || { echo "out/traces.jsonl não existe: rode o passo 3 antes" >&2; exit 1; }
python3 filter_traces.py --traces out/traces.jsonl --prompts data/prompts.jsonl --outdir out >/dev/null
files="out/kept.jsonl out/traces.jsonl out/report.json out/samples.md"
for f in out/env_info.txt out/NOTAS.md; do [ -f "$f" ] && files="$files $f"; done
tar czf thor_out.tar.gz $files
echo "gerado: $(pwd)/thor_out.tar.gz ($(du -h thor_out.tar.gz | cut -f1)); mantidos: $(wc -l < out/kept.jsonl); gerados: $(wc -l < out/traces.jsonl)"
sha256sum thor_out.tar.gz
