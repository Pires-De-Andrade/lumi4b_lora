#!/usr/bin/env bash
# Fila da noite de 08/11 (versão 2, 21:00). Idempotente: cada etapa é retomável (run_grpo.sh, run_experiment.sh e eval_vllm.py pulam o que já existe).
# Ordem: (1) SFT `thor_pt_e1`; (2) RL `gspo_thor_e1` a partir dele; (3) o RL do ramo escolhido pela regra de estudos/08; (4) suíte de teste do Lumi.
#  (1) SFT `thor_pt_e1`: entrega 1 do pacote 3 (9.850 exemplos com raciocínio, limite 2.048, receita do `thor_pt`), avaliação na validação.
#  (2) RL `gspo_thor_e1`: receita do `gspo_thor` (questões antigas, 2.048, T 0,6, top-p 0,95, --ratio seq, --max-noclose 0.85) com 4 ciclos, a partir
#      do `thor_pt_e1`. Roda sempre que o SFT tiver avaliação (a triagem de 4 ciclos é barata e o RL foi o que tirou o `thor_pt` dos laços).
#  (3) Ramo, pela avaliação do `gspo_gold` (2.048) no ciclo 4 (X = múlt. escolha com thinking, C = fecha `</think>` e responde = 1 - no_close - empty_answer):
#      A: X >= 0,67 e C >= 0,90 -> `gspo_gold` até o ciclo 8 e `gspo_gold_thor` (a partir do `thor_pt`, 2.048, questões do gold)
#      B: X <= 0,64 ou C < 0,87  -> `gspo1_4k` (receita do `gspo1`, questões antigas, limite 4.096, 4 ciclos)
#      C: entre os dois          -> `gspo_gold` até o ciclo 8
#      A leitura de 20:51 foi `B X=0,6390 C=0,8935` e está em runs/jobs/noite.ramo (se o arquivo existe, é usado sem reler).
#  (4) Suíte de teste do Lumi (o que falta; retomável; parar: touch runs/jobs/suite.stop). A suíte é só checagem e referência.
# Uso: scripts/detach.sh noite scripts/run_noite_chain.sh        Teste da regra: scripts/run_noite_chain.sh --decide NOME_DA_AVALIACAO
# Cancelar: tmux kill-session -t noite (e conferir os processos do vLLM). Estado: runs/jobs/noite.status. NÃO editar com ele rodando.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

decide() {  # $1 = nome em runs/eval; imprime "RAMO X=... C=..." ou SEM_RESULTADO
  python3 - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(f"runs/eval/{sys.argv[1]}/summary.json"))["mcq_val|thinking"]
    x = d["strict_acc"]; r = d["status_rates"]; c = 1 - r["no_close"] - r["empty_answer"]
except Exception:
    print("SEM_RESULTADO"); sys.exit(0)
ramo = "A" if (x >= 0.67 and c >= 0.90) else "B" if (x <= 0.64 or c < 0.87) else "C"
print(f"{ramo} X={x:.4f} C={c:.4f}")
PY
}
if [ "${1:-}" = "--decide" ]; then decide "$2"; exit 0; fi

status=runs/jobs/noite.status
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }
wait_gpu() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
    say "GPU ocupada, aguardando"; sleep 120
  done
}
tem_avaliacao() {  # $1 = nome em runs/eval: tem a múltipla escolha com thinking?
  python3 - "$1" <<'PY'
import json, sys
try:
    json.load(open(f"runs/eval/{sys.argv[1]}/summary.json"))["mcq_val|thinking"]
except Exception:
    sys.exit(1)
PY
}

avaliacao_completa() {  # $1 = nome em runs/eval: as 4 avaliações da validação (GSM8K e múlt. escolha, com e sem thinking) estão no summary.json?
  python3 - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(f"runs/eval/{sys.argv[1]}/summary.json"))
    assert all(f"{t}|{m}" in d for t in ("gsm8k_val", "mcq_val") for m in ("thinking", "nothinking"))
except Exception:
    sys.exit(1)
PY
}

sft_entrega1() {
  if avaliacao_completa thor_pt_e1; then say "SFT thor_pt_e1 já treinado e avaliado: etapa pulada"; return; fi
  wait_gpu
  say "INICIO SFT thor_pt_e1 (data/thor_pt_e1.parquet: 9.850 exemplos com raciocínio + 5.544 sem, limite 2.048)"
  scripts/run_experiment.sh thor_pt_e1 --arm lora --data data/thor_pt_e1.parquet --epochs 1 --lr 1e-4
  say "FIM SFT thor_pt_e1 (código $?)"
}
rl_a_partir_do_e1() {
  if ! tem_avaliacao thor_pt_e1; then say "sem avaliação do thor_pt_e1: RL gspo_thor_e1 não roda"; return; fi
  wait_gpu
  say "INICIO RL gspo_thor_e1 (a partir do thor_pt_e1, questões antigas, 2.048, 4 ciclos)"
  scripts/run_grpo.sh gspo_thor_e1 --init runs/train/thor_pt_e1/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
    --max-new 2048 --temperature 0.6 --top-p 0.95 --max-noclose 0.85 --ratio seq
  say "FIM RL gspo_thor_e1 (código $?)"
}
gold_ate_ciclo8() {
  # O laço avalia o modelo que entra na iteração 4 como `gspo_gold_it4` e pula o que já existe; essa avaliação (4 ciclos) já foi feita com esse nome.
  wait_gpu
  say "INICIO RL gspo_gold até o ciclo 8 (2.048, questões do gold)"
  scripts/run_grpo.sh gspo_gold --init runs/train/adalberto/adapter --iters 8 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
    --max-new 2048 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq --prompts data/grpo_prompts_gold.jsonl
  say "FIM RL gspo_gold (código $?)"
}
triagem_thor_2048() {
  wait_gpu
  say "INICIO RL gspo_gold_thor (a partir do thor_pt, 2.048, questões do gold, 4 ciclos)"
  scripts/run_grpo.sh gspo_gold_thor --init runs/train/thor_pt/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
    --max-new 2048 --temperature 0.6 --top-p 0.95 --max-noclose 0.85 --ratio seq --prompts data/grpo_prompts_gold.jsonl
  say "FIM RL gspo_gold_thor (código $?)"
}
gspo1_com_4096() {
  wait_gpu
  say "INICIO RL gspo1_4k (receita do gspo1, questões antigas, 4.096, 4 ciclos)"
  scripts/run_grpo.sh gspo1_4k --init runs/train/adalberto/adapter --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
    --max-new 4096 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq
  say "FIM RL gspo1_4k (código $?)"
}
atualizar_tabela() { PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true; }

# Uma avaliação manual (sessão eval-e1) pode estar em curso: espera acabar antes de decidir o que pular.
while tmux has-session -t eval-e1 2>/dev/null; do sleep 30; done

# (1) e (2): SFT com a entrega 1 e o RL a partir dele
sft_entrega1
atualizar_tabela
rl_a_partir_do_e1
atualizar_tabela

# (3): ramo da regra de estudos/08
ramo_arq=runs/jobs/noite.ramo
if [ -s "$ramo_arq" ]; then
  res="$(cat "$ramo_arq")"
else
  while tmux has-session -t gold2048 2>/dev/null; do sleep 30; done
  res="$(decide gspo_gold_it4)"; [ "$res" = "SEM_RESULTADO" ] && res="$(decide gspo_gold)"
  echo "$res" > "$ramo_arq"
fi
ramo="${res%% *}"
say "leitura do gspo_gold (2.048, ciclo 4): $res"
case "$ramo" in
  A) say "ramo A: o limite de 4.096 é o culpado"; gold_ate_ciclo8; triagem_thor_2048;;
  B) say "ramo B: as questões são o problema"; gspo1_com_4096;;
  C) say "ramo C: ambíguo"; gold_ate_ciclo8;;
  *) say "sem resultado do gspo_gold (avaliação ausente): sem RL do ramo";;
esac
atualizar_tabela

# (4): suíte de teste do Lumi
wait_gpu
say "INICIO suíte de teste do Lumi (o que falta)"
scripts/run_suite_models.sh "base|-"
say "FIM da fila da noite"
