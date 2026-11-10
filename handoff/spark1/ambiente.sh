#!/usr/bin/env bash
# Confere a Spark antes de gastar horas: GPU, memória, disco, vLLM, acesso ao Hugging Face, provas novas e pesos. Grava out/env_info.txt (devolvido junto).
# Uso (na raiz do projeto): bash handoff/spark1/ambiente.sh [--sha]     (--sha confere o sha256 dos pesos, ~1 min por modelo)
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p out
PY="${PY:-.venv-eval/bin/python}"
{
echo "== data: $(date -Is)   host: $(hostname)   $(uname -srm)"
echo "== GPU"; nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv 2>&1 | head -3
echo "== processos na GPU"; nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>&1 | head -5
echo "== memória (kB)"; grep -E "MemTotal|MemAvailable" /proc/meminfo
echo "== disco livre aqui"; df -h . | tail -1
echo "== nvcc"; (nvcc --version 2>&1 | tail -1) || echo "sem nvcc"
echo "== python do sistema: $(python3 --version 2>&1)"
echo "== docker: $(command -v docker || echo não)   tmux: $(command -v tmux || echo não)   sha256sum: $(command -v sha256sum || echo não)"
echo "== PY=$PY"
"$PY" - <<'EOF' 2>&1 | head -12
import importlib.metadata as m, sys
print("python", sys.version.split()[0])
for p in ["vllm", "torch", "transformers", "tokenizers", "datasets", "flashinfer-python", "numpy"]:
    try:
        print(p, m.version(p))
    except Exception:
        print(p, "NÃO INSTALADO")
import torch
print("cuda disponível:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF
echo "(a 4090 roda vllm 0.33.0, torch 2.14.0, transformers 5.20.2, flashinfer-python 0.7.2.post1: usar versões iguais ou próximas e anotar as usadas)"
echo "== Hugging Face (esperado 200): $("$PY" -c "
import urllib.request
try:
    print(urllib.request.urlopen('https://huggingface.co/api/models/ceia-br/lumi-4b', timeout=15).status)
except Exception as e:
    print(getattr(e, 'code', type(e).__name__))
" 2>&1 | tail -1)"
echo "== provas novas (linhas; esperado 1093, 179, 1000, 1000, 1000)"
for t in oab_novas enem_2024 mmlu_pt_exatas mmlu_pt_enade mmlu_pt_concursos; do
  printf "%s: %s\n" "$t" "$(wc -l < data/novas_provas/$t.jsonl 2>/dev/null || echo FALTA)"
done
echo "== pesos (esperado 7947142672 bytes cada)"
for d in runs/merged/*/; do
  n=$(basename "$d"); f="$d/model.safetensors"
  if [ -e "$f" ]; then printf "%s: %s bytes\n" "$n" "$(stat -c %s "$f")"; else printf "%s: sem model.safetensors\n" "$n"; fi
done
if [ "${1:-}" = "--sha" ] && [ -s handoff/spark1/PESOS.sha256 ]; then
  echo "== sha256 dos pesos presentes"; sha256sum -c --ignore-missing handoff/spark1/PESOS.sha256 2>&1
fi
} 2>&1 | tee out/env_info.txt
