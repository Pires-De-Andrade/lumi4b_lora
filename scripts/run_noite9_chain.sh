#!/usr/bin/env bash
# Fila de 09/11 (versão 9). Substitui o run_noite8_chain.sh (congelada com SIGSTOP; esta fila espera os processos filhos dela acabarem). Cada etapa é
# retomável e pulada se já pronta. A 4090 é compartilhada: cada etapa espera a GPU ficar livre (menos de 1 GiB em uso) antes de começar.
# Ordem:
#   0. Espera o que a noite8 tinha em curso acabar e encerra a noite8.
#   1. Avaliação das sopas de pesos (H55) e do Lumi com o começo forçado em português (H56), como na versão 8 (puladas se já avaliadas).
#   2. Fase anti-laço (H57) sobre o `gspo1` de 8 ciclos (runs/grpo/gspo1/it007/adapter, fundido em runs/merged/gspo1_c8):
#        a. taxa de laço de referência nos prompts de `heldout` (750, decodificação gulosa, 3.072 tokens);
#        b. mineração de laços nos prompts de `mine` (8.000 de múltipla escolha e 2.000 de matemática);
#        c. teste de fumaça do treino (6 eventos, 2 passos) e treino com parada antecipada (chosen_win >= 0,35 na validação dos eventos);
#        d. avaliação da validação do modelo novo (`gspo1_antidoom_r1`) e taxa de laço nos mesmos prompts de `heldout`.
#      Leitura prevista (H57): apoia se o não terminar em `heldout` cair pelo menos à metade e, na validação, a múltipla escolha com thinking ficar em
#      >= 65,5% (o `gspo1` tem 67,5%; queda de até 2 pontos é tolerada), fechar >= 95% e o traço em português ficar >= 90%.
#   3. SFTs de teste `qwen_mix_short` (H53), `selfdist_min` e `selfdist_300` (H54).
#   4. Suíte de teste do Lumi até o fim, como preenchimento (a ÚLTIMA de todas; retomável; parar: touch runs/jobs/suite.stop).
# Uso: scripts/detach.sh noite9 scripts/run_noite9_chain.sh. Cancelar: tmux kill-session -t noite9 (e conferir os processos do vLLM).
# Estado: runs/jobs/noite9.status. NÃO editar com ele rodando.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

status=runs/jobs/noite9.status
ANT=runs/antidoom/r1
ADAPTER=runs/grpo/gspo1/it007/adapter
BASE_MODEL=runs/merged/gspo1_c8
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }
wait_gpu() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
    sleep 120
  done
}
avaliacao_completa() {  # $1 = nome em runs/eval: as 4 avaliações da validação estão no summary.json?
  python3 - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(f"runs/eval/{sys.argv[1]}/summary.json"))
    assert all(f"{t}|{m}" in d for t in ("gsm8k_val", "mcq_val") for m in ("thinking", "nothinking"))
except Exception:
    sys.exit(1)
PY
}
atualizar_tabela() { PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true; }
avaliar() {  # $1 = nome, $2 = modelo (pasta ou id do HF), resto = argumentos extras do eval_vllm.py; até 3 tentativas (segfault intermitente do vLLM)
  local nome="$1" modelo="$2"; shift 2
  if avaliacao_completa "$nome"; then say "avaliação de $nome já feita: etapa pulada"; return; fi
  for t in 1 2 3; do
    avaliacao_completa "$nome" && break
    wait_gpu
    say "INICIO avaliação do $nome, tentativa $t"
    VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/eval_vllm.py --name "$nome" --model "$modelo" --tasks validation "$@"
  done
  say "FIM avaliação do $nome"
}
sft_e_avaliar() {  # $1 = nome, $2 = parquet; receita do adalberto (LoRA r=16, lr 1e-4, 1 época)
  if avaliacao_completa "$1"; then say "SFT $1 já treinado e avaliado: etapa pulada"; return; fi
  wait_gpu
  say "INICIO SFT $1 ($2)"
  if [ ! -f "runs/train/$1/adapter/adapter_config.json" ]; then
    scripts/run_experiment.sh "$1" --arm lora --data "$2" --epochs 1 --lr 1e-4
  fi
  [ -f "runs/merged/$1/merge_report.json" ] || PYTHONPATH=src python scripts/merge_adapter.py --adapter "runs/train/$1/adapter" --out "runs/merged/$1"
  avaliar "$1" "runs/merged/$1"
  say "FIM SFT $1"
}
minerar() {  # $1 = rótulo, $2 = modelo, $3 = split (mine ou heldout), $4 = pasta de saída; até 2 tentativas
  if [ -s "$4/stats.json" ]; then say "mineração $1 já feita: etapa pulada"; return; fi
  for t in 1 2; do
    [ -s "$4/stats.json" ] && break
    wait_gpu
    say "INICIO mineração $1 ($3), tentativa $t"
    VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/antidoom_mine.py --model "$2" --split "$3" --out "$4"
  done
  say "FIM mineração $1: $(python3 -c "import json; d = json.load(open('$4/stats.json')); print(d['taxas'], d['contagens'])" 2>&1 | tail -n 1)"
}

# 0. espera os filhos da noite8 (congelada) acabarem e encerra a noite8
P8=$(pgrep -f "^bash scripts/run_noite8_chain.sh" | head -1)
if [ -n "$P8" ]; then
  while pgrep -P "$P8" > /dev/null; do sleep 20; done
  say "noite8 sem processos em curso: encerrando a fila (congelada)"
  pkill -KILL -f "scripts/run_noite8_chain.sh" 2>/dev/null; sleep 2
  tmux kill-session -t =noite8 2>/dev/null
fi
atualizar_tabela

# dados e modelo (CPU, pulados se prontos)
scripts/prepare_soups.sh
scripts/prepare_sft_curtos.sh
[ -f data/antidoom_prompts.jsonl ] || python scripts/make_antidoom_prompts.py
[ -f "$BASE_MODEL/merge_report.json" ] || PYTHONPATH=src python scripts/merge_adapter.py --adapter "$ADAPTER" --out "$BASE_MODEL"

# 1. sopas de pesos e Lumi com o começo forçado em português
avaliar gspo1_soup2 runs/merged/gspo1_soup2
avaliar gspo1_soup3 runs/merged/gspo1_soup3
atualizar_tabela
avaliar base_ptprefill ceia-br/lumi-4b --prefill "Vou raciocinar em português."
atualizar_tabela

# 2. fase anti-laço
minerar "referência do gspo1_c8" "$BASE_MODEL" heldout "$ANT/base_heldout"
minerar "do gspo1_c8" "$BASE_MODEL" mine "$ANT/mine"
if [ -s "$ANT/mine/events.jsonl" ] && [ ! -f "$ANT/train/adapter/adapter_model.safetensors" ]; then
  wait_gpu
  say "INICIO teste de fumaça do treino anti-laço"
  if PYTHONPATH=src python scripts/antidoom_train.py --smoke --events "$ANT/mine/events.jsonl" --adapter-in "$ADAPTER" --out "$ANT/smoke"; then
    say "FIM teste de fumaça: ok"
    wait_gpu
    say "INICIO treino anti-laço"
    PYTHONPATH=src python scripts/antidoom_train.py --events "$ANT/mine/events.jsonl" --adapter-in "$ADAPTER" --out "$ANT/train"
    say "FIM treino anti-laço (código $?)"
  else
    say "FALHA no teste de fumaça do treino anti-laço: treino não roda"
  fi
fi
if [ -f "$ANT/train/adapter/adapter_model.safetensors" ]; then
  mkdir -p runs/train/gspo1_antidoom_r1 && cp -r "$ANT/train/adapter" runs/train/gspo1_antidoom_r1/
  [ -f runs/merged/gspo1_antidoom_r1/merge_report.json ] || PYTHONPATH=src python scripts/merge_adapter.py --adapter runs/train/gspo1_antidoom_r1/adapter --out runs/merged/gspo1_antidoom_r1
  avaliar gspo1_antidoom_r1 runs/merged/gspo1_antidoom_r1
  minerar "do gspo1_antidoom_r1" runs/merged/gspo1_antidoom_r1 heldout "$ANT/after_heldout"
  atualizar_tabela
fi

# 3. SFTs de teste
sft_e_avaliar qwen_mix_short data/qwen_mix_short.parquet
atualizar_tabela
sft_e_avaliar selfdist_min data/selfdist_min.parquet
atualizar_tabela
sft_e_avaliar selfdist_300 data/selfdist_300.parquet
atualizar_tabela

# 4. suíte de teste do Lumi, até o fim
wait_gpu
say "INICIO suíte de teste do Lumi (o que falta; preenchimento, interrompível)"
scripts/run_suite_models.sh "base|-"
say "FIM da fila (versão 9)"
