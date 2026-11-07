#!/usr/bin/env bash
# Empacota tudo o que já existir em thor3_out.tar.gz: traduções (pacotes 2 e 3) e a geração do pacote 3.
# Pode rodar a qualquer hora, com a geração ou a tradução em andamento (entrega parcial).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
files=""
if [ -f out/traces.jsonl ]; then
  python3 filter_traces_en.py --traces out/traces.jsonl --prompts data/prompts.jsonl --outdir out >/dev/null
  files="out/kept.jsonl out/traces.jsonl out/report.json out/samples.md"
fi
for f in out/translated.jsonl out/translated.jsonl.report.json out/smoke_tr.jsonl out/smoke_tr.jsonl.report.json out/env_info.txt out/NOTAS.md; do
  if [ -f "$f" ]; then files="$files $f"; fi
done
if [ -z "$files" ]; then echo "nada para empacotar ainda" >&2; exit 1; fi
tar czf thor3_out.tar.gz $files
echo "gerado: $(pwd)/thor3_out.tar.gz ($(du -h thor3_out.tar.gz | cut -f1))"
if [ -f out/translated.jsonl ]; then echo "traduzidos: $(wc -l < out/translated.jsonl) (ok: $(grep -c '"verdict": "ok"' out/translated.jsonl || true))"; fi
if [ -f out/kept.jsonl ]; then echo "pacote 3: gerados $(wc -l < out/traces.jsonl), mantidos $(wc -l < out/kept.jsonl)"; fi
sha256sum thor3_out.tar.gz
