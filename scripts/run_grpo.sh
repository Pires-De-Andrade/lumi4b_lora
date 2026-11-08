#!/usr/bin/env bash
# GRPO em iterações: funde o adaptador, gera respostas (vLLM), treina; repete. Retomável: pula as iterações já completas.
# Uso (desacoplado):
#   scripts/detach.sh --gpu grpo scripts/run_grpo.sh NOME --init runs/train/adalberto/adapter --iters 10 [--eval-every 3]
#     [--n-prompts 256] [--group 8] [--lr 2e-5] [--frac-mcq 0.5] [--temperature 0.8] [--top-p 1.0] [--max-new 2048] [--adv std|mean]
#     [--mask-truncated 0|1] [--max-noclose 0.05] [--ratio token|seq] [--positives-only 0|1] [--final-eval 0|1]
#     [--prompts data/grpo_prompts.jsonl]  (padrão do grpo_rollout.py; use data/grpo_prompts_gold.jsonl para o gabarito oficial)
# --mask-truncated 1: as respostas truncadas não entram na perda (filtro de truncadas do DAPO), mas contam na vantagem do grupo.
# --max-noclose X (0 desliga): antes de treinar cada iteração, para se, em algum tipo, mais que X das respostas terminarem sem um único
#   `</think>` (degeneração na amostragem); grava runs/grpo/NOME/STOPPED e avalia o último adaptador bom. --ratio seq: variante GSPO.
# Saídas em runs/grpo/NOME: itNNN/{rollouts.jsonl,adapter,train_log.jsonl,ckpt.pt}, merged/ (último fundido), ckpt.pt (AdamW).
# Avaliação (validação, modo thinking) do modelo que entra na iteração N: runs/eval/NOME_itN. No fim, avaliação completa em runs/eval/NOME.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
name="$1"; shift
init=""; iters=10; eval_every=3; n_prompts=256; group=8; lr=2e-5; frac_mcq=0.5; temp=0.8; max_new=2048; adv=std; final_eval=1
top_p=1.0; mask_trunc=0; max_noclose=0; ratio=token; pos_only=0; prompts=""
while [ $# -gt 0 ]; do
  case "$1" in
    --init) init="$2"; shift 2;; --iters) iters="$2"; shift 2;; --eval-every) eval_every="$2"; shift 2;;
    --n-prompts) n_prompts="$2"; shift 2;; --group) group="$2"; shift 2;; --lr) lr="$2"; shift 2;;
    --frac-mcq) frac_mcq="$2"; shift 2;; --temperature) temp="$2"; shift 2;; --max-new) max_new="$2"; shift 2;;
    --adv) adv="$2"; shift 2;; --final-eval) final_eval="$2"; shift 2;; --top-p) top_p="$2"; shift 2;;
    --mask-truncated) mask_trunc="$2"; shift 2;; --max-noclose) max_noclose="$2"; shift 2;; --ratio) ratio="$2"; shift 2;;
    --positives-only) pos_only="$2"; shift 2;; --prompts) prompts="$2"; shift 2;; *) echo "argumento desconhecido: $1" >&2; exit 2;;
  esac
done
[ -n "$init" ] || { echo "faltou --init (adaptador de partida)" >&2; exit 2; }
root="runs/grpo/$name"; mkdir -p "$root"
VL="VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python"
cur="$init"
for ((it=0; it<iters; it++)); do
  d="$root/it$(printf %03d $it)"
  if [ -f "$d/adapter/adapter_config.json" ] && [ -f "$d/ckpt.pt" -o -f "$d/train_log.jsonl" ]; then cur="$d/adapter"; echo "it$it já feita"; continue; fi
  echo "=== iteração $it: fundir"
  fast=""; [ "$it" -gt 0 ] && fast="--fast"
  PYTHONPATH=src python scripts/merge_adapter.py --adapter "$cur" --out "$root/merged" $fast | tail -n 2
  if [ "$it" -gt 0 ] && [ $((it % eval_every)) -eq 0 ]; then
    echo "=== avaliação do modelo que entra na iteração $it"
    env $VL scripts/eval_vllm.py --name "${name}_it$it" --model "$root/merged" --tasks validation --modes thinking
  fi
  echo "=== iteração $it: gerar"
  [ -f "$d/rollouts.jsonl" ] || env $VL scripts/grpo_rollout.py --model "$root/merged" --iter "$it" --n-prompts "$n_prompts" --group "$group" \
      --frac-mcq "$frac_mcq" --temperature "$temp" --top-p "$top_p" --max-new "$max_new" --adv "$adv" --out "$d/rollouts.jsonl" ${prompts:+--prompts "$prompts"}
  if [ "$max_noclose" != 0 ]; then
    if ! PYTHONPATH=src python scripts/grpo_guard.py "$d/rollouts.stats.json" --max-noclose "$max_noclose"; then
      echo "PARADA na iteração $it: respostas sem </think> acima de $max_noclose; o último adaptador bom é $cur" | tee "$root/STOPPED"
      break
    fi
  fi
  echo "=== iteração $it: treinar"
  ck=""; [ -f "$root/ckpt.pt" ] && ck="--ckpt-in $root/ckpt.pt"
  mt=""; [ "$mask_trunc" = 1 ] && mt="--mask-truncated"; [ "$pos_only" = 1 ] && mt="$mt --positives-only"
  PYTHONPATH=src python scripts/grpo_train.py --rollouts "$d/rollouts.jsonl" --adapter-in "$cur" $ck --out "$d" --lr "$lr" --ratio "$ratio" $mt
  [ -f "$d/ckpt.pt" ] && cp "$d/ckpt.pt" "$root/ckpt.pt"
  cur="$d/adapter"
done
if [ "$final_eval" = 1 ]; then
  echo "=== final: fundir e avaliar"
  PYTHONPATH=src python scripts/merge_adapter.py --adapter "$cur" --out "$root/merged"
  env $VL scripts/eval_vllm.py --name "$name" --model "$root/merged" --tasks validation
fi
echo "GRPO $name COMPLETO"
