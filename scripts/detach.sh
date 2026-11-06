#!/usr/bin/env bash
# Uso: scripts/detach.sh [--gpu] <nome> <comando...>
# --gpu: recusa lançar se a GPU já estiver em uso (processo ou mais de 1 GiB de VRAM ocupada).
# Roda o comando numa sessão tmux própria, independente do Claude Code e do SSH.
# Log:      runs/jobs/<nome>.log     Código de saída: runs/jobs/<nome>.exit
# Acompanhar: tmux attach -t <nome>   |   tail -f runs/jobs/<nome>.log
set -euo pipefail
gpu=0
if [ "${1:-}" = "--gpu" ]; then gpu=1; shift; fi
name="$1"; shift
if [ "$gpu" = 1 ]; then
  apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 || true)"
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [ -n "$apps" ] || [ "${used:-0}" -gt 1024 ]; then
    echo "GPU ocupada (${used:-?} MiB em uso). Não lancei '$name'." >&2
    [ -n "$apps" ] && echo "$apps" >&2
    exit 3
  fi
fi
proj="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$proj/runs/jobs"
log="$proj/runs/jobs/$name.log"; ex="$proj/runs/jobs/$name.exit"
tmux has-session -t "$name" 2>/dev/null && { echo "sessão '$name' já existe" >&2; exit 1; }
rm -f "$ex"
runner="$proj/runs/jobs/$name.run.sh"
cat > "$runner" <<EOF
#!/usr/bin/env bash
cd '$proj' && source env.sh
{ $*; } 2>&1 | tee -a '$log'
echo \${PIPESTATUS[0]} > '$ex'
EOF
tmux new-session -d -s "$name" "bash '$runner'"
echo "job '$name' iniciado | log: runs/jobs/$name.log | tmux attach -t $name"
