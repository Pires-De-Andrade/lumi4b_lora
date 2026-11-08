#!/usr/bin/env bash
# Benchmark A/B da velocidade da geração do RL (08/11 18:50): roda scripts/bench_rollout.py para várias configurações do
# vLLM, uma por processo (motor novo cada vez), com o mesmo modelo, prompts e amostragem. Não mexe no RL (só arquivos novos).
# Configurações (NOME|prefix caching|backend Mamba|memória): a base repete no fim para medir o ruído entre execuções.
# Cada execução tem tempo máximo (CFG_TIMEOUT, 1.500 s). Uma que falhe (ex.: o FlashInfer não compilar) não derruba as outras.
# Estado: runs/jobs/bench-rollout.status; resultados: runs/bench_rollout/results.jsonl (tabela: scripts/bench_rollout.py --report).
# Uso: scripts/detach.sh bench-rollout 'while tmux has-session -t gold-chain 2>/dev/null; do sleep 30; done; scripts/run_bench_rollout.sh'
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
MODEL="${BENCH_MODEL:-runs/grpo/gspo1/merged}"   # modelo fixo e já pronto (o mesmo que iria à suíte)
CFG_TIMEOUT="${CFG_TIMEOUT:-1500}"
dir=runs/bench_rollout; status=runs/jobs/bench-rollout.status
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }
mkdir -p "$dir"
[ -f "$dir/results.jsonl" ] && mv "$dir/results.jsonl" "$dir/results.$(date +%Y%m%d-%H%M%S).jsonl"
CONFIGS=(
  "base|1|triton|0.88"
  "prefix_off|0|triton|0.88"
  "prefix_off_mem93|0|triton|0.93"
  "flashinfer_prefix_off|0|flashinfer|0.88"
  "base_repete|1|triton|0.88"
)
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
  say "GPU ocupada, aguardando"; sleep 120
done
[ -f "$MODEL/config.json" ] || { say "FALHOU: modelo $MODEL não existe"; exit 1; }
for item in "${CONFIGS[@]}"; do
  IFS='|' read -r name pc backend mem <<< "$item"
  say "INICIO bench $name (prefix_caching=$pc, backend=$backend, mem=$mem)"
  VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src timeout "$CFG_TIMEOUT" .venv-vllm/bin/python scripts/bench_rollout.py --model "$MODEL" --label "$name" \
    --prefix-caching "$pc" --mamba-backend "$backend" --mem "$mem" > "$dir/$name.log" 2>&1
  rc=$?
  [ "$rc" -eq 0 ] && say "FIM bench $name (ok)" || say "FALHOU bench $name (código $rc; veja $dir/$name.log)"
  PYTHONPATH=src python scripts/bench_rollout.py --parse-log "$dir/$name.log" --label "$name" > /dev/null 2>&1 || true
  sleep 10   # deixa a GPU liberar a memória antes do próximo motor
done
say "FIM do benchmark"
PYTHONPATH=src python scripts/bench_rollout.py --report | tee "$dir/tabela.md" | sed "s/^/  /"
