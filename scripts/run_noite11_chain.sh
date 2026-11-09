#!/usr/bin/env bash
# Fila de 09/11 (versão 11). Substitui o run_noite10_chain.sh (congelada com SIGSTOP; esta fila espera os processos filhos dela acabarem). Cada etapa é
# retomável e pulada se já pronta. A 4090 é compartilhada: cada etapa espera a GPU ficar livre (menos de 1 GiB em uso) antes de começar.
# Motivo da mudança: nas variantes da Thor quase toda resposta que não fecha é laço exato (gspo_thor: 44 de 47; thor_pt: 129 de 131), enquanto no `gspo1`
# são 4 de 26 (discovery §20.59). A fase anti-laço passa a rodar primeiro no `gspo_thor` (RL de 8 ciclos a partir do `thor_pt`, validação 67,0%, fecha 87,8%).
# Ordem:
#   0. Espera o que a noite10 tinha em curso (a mineração do `gspo1`) acabar e encerra a noite10.
#   1. Teste de fumaça do treino anti-laço (usa os eventos do `gspo1`; o código ainda não rodou na GPU).
#   2. Fase anti-laço no `gspo_thor` (H58): taxa de laço de referência em `heldout` (750 prompts), mineração em 5.000 prompts de múltipla escolha,
#      treino com parada antecipada, avaliação da validação e taxa de laço depois, nos mesmos prompts de `heldout`.
#   3. SFTs de teste `qwen_mix_short` (H53), `selfdist_min` e `selfdist_300` (H54).
#   4. Fase anti-laço no `gspo1` de 8 ciclos (H57, teto de ~+0,6 ponto): treino com os eventos já minerados, avaliação e taxa de laço depois.
#   5. Suíte de teste do `gspo1` (8 ciclos, runs/grpo/gspo1/merged), a ÚLTIMA de todas (retomável; parar: touch runs/jobs/suite.stop). A suíte do Lumi roda
#      em outra máquina (Jetson Thor).
# Uso: scripts/detach.sh noite11 scripts/run_noite11_chain.sh. Cancelar: tmux kill-session -t noite11 (e conferir os processos do vLLM).
# Estado: runs/jobs/noite11.status. NÃO editar com ele rodando.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

status=runs/jobs/noite11.status
ANT=runs/antidoom/r1            # fase anti-laço sobre o gspo1
ADAPTER=runs/grpo/gspo1/it007/adapter
BASE_MODEL=runs/merged/gspo1_c8
ANT2=runs/antidoom/r2           # fase anti-laço sobre o gspo_thor
THOR_ADAPTER=runs/grpo/gspo_thor/it007/adapter
THOR_MODEL=runs/grpo/gspo_thor/merged
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
minerar() {  # $1 = rótulo, $2 = modelo, $3 = split (mine ou heldout), $4 = pasta de saída, resto = argumentos extras; até 2 tentativas
  local rot="$1" modelo="$2" split="$3" saida="$4"; shift 4
  if [ -s "$saida/stats.json" ]; then say "mineração $rot já feita: etapa pulada"; return; fi
  for t in 1 2; do
    [ -s "$saida/stats.json" ] && break
    wait_gpu
    say "INICIO mineração $rot ($split), tentativa $t"
    VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/antidoom_mine.py --model "$modelo" --split "$split" --out "$saida" "$@"
  done
  say "FIM mineração $rot: $(python3 -c "import json; d = json.load(open('$saida/stats.json')); print(d['taxas'], d['contagens'])" 2>&1 | tail -n 1)"
}
treinar_antilaco() {  # $1 = rótulo, $2 = eventos, $3 = adaptador de partida, $4 = pasta de saída
  if [ -f "$4/adapter/adapter_model.safetensors" ]; then say "treino anti-laço $1 já feito: etapa pulada"; return; fi
  wait_gpu
  say "INICIO treino anti-laço $1"
  PYTHONPATH=src python scripts/antidoom_train.py --events "$2" --adapter-in "$3" --out "$4"
  say "FIM treino anti-laço $1 (código $?)"
}
avaliar_antilaco() {  # $1 = nome do modelo novo, $2 = pasta do treino, $3 = pasta da taxa de laço depois
  [ -f "$2/adapter/adapter_model.safetensors" ] || { say "sem adaptador em $2: avaliação de $1 pulada"; return; }
  mkdir -p "runs/train/$1" && cp -r "$2/adapter" "runs/train/$1/"
  [ -f "runs/merged/$1/merge_report.json" ] || PYTHONPATH=src python scripts/merge_adapter.py --adapter "runs/train/$1/adapter" --out "runs/merged/$1"
  avaliar "$1" "runs/merged/$1"
  minerar "do $1" "runs/merged/$1" heldout "$3"
  atualizar_tabela
}

# 0. espera os filhos da noite10 (congelada) acabarem e encerra a noite10
filhos_vivos() {  # processos filhos de $1 que ainda não terminaram (zumbis de um pai parado não contam)
  local c st
  for c in $(pgrep -P "$1"); do
    st=$(ps -o stat= -p "$c" 2>/dev/null | tr -d ' ')
    [ -n "$st" ] && [[ "$st" != Z* ]] && echo "$c"
  done
}
P10=$(pgrep -f "^bash scripts/run_noite10_chain.sh" | head -1)
if [ -n "$P10" ]; then
  while [ -n "$(filhos_vivos "$P10")" ]; do sleep 20; done
  say "noite10 sem processos em curso: encerrando a fila (congelada)"
  pkill -KILL -f "scripts/run_noite10_chain.sh" 2>/dev/null; sleep 2
  tmux kill-session -t =noite10 2>/dev/null
fi
atualizar_tabela

# 1. teste de fumaça do treino anti-laço (eventos do gspo1; se a mineração não deixou eventos, os da referência em `heldout`)
EV="$ANT/mine/events.jsonl"; [ -s "$EV" ] || EV="$ANT/base_heldout/events.jsonl"
SMOKE_OK=0
if [ -f "$ANT/smoke/summary.json" ]; then
  SMOKE_OK=1
elif [ -s "$EV" ]; then
  wait_gpu
  say "INICIO teste de fumaça do treino anti-laço ($EV)"
  if PYTHONPATH=src python scripts/antidoom_train.py --smoke --events "$EV" --adapter-in "$ADAPTER" --out "$ANT/smoke"; then
    say "FIM teste de fumaça: ok"; SMOKE_OK=1
  else
    say "FALHA no teste de fumaça do treino anti-laço: treinos anti-laço não rodam"
  fi
fi

# 2. fase anti-laço no gspo_thor (H58)
if [ "$SMOKE_OK" = 1 ]; then
  minerar "referência do gspo_thor" "$THOR_MODEL" heldout "$ANT2/base_heldout"
  minerar "do gspo_thor" "$THOR_MODEL" mine "$ANT2/mine" --limit 5000
  [ -s "$ANT2/mine/events.jsonl" ] && treinar_antilaco "do gspo_thor" "$ANT2/mine/events.jsonl" "$THOR_ADAPTER" "$ANT2/train"
  avaliar_antilaco gspo_thor_antidoom_r2 "$ANT2/train" "$ANT2/after_heldout"
fi

# 3. SFTs de teste
sft_e_avaliar qwen_mix_short data/qwen_mix_short.parquet
atualizar_tabela
sft_e_avaliar selfdist_min data/selfdist_min.parquet
atualizar_tabela
sft_e_avaliar selfdist_300 data/selfdist_300.parquet
atualizar_tabela

# 4. fase anti-laço no gspo1 (H57)
if [ "$SMOKE_OK" = 1 ] && [ -s "$ANT/mine/events.jsonl" ]; then
  treinar_antilaco "do gspo1" "$ANT/mine/events.jsonl" "$ADAPTER" "$ANT/train"
  avaliar_antilaco gspo1_antidoom_r1 "$ANT/train" "$ANT/after_heldout"
fi

# 5. suíte de teste do gspo1 (8 ciclos), a última de todas
wait_gpu
say "INICIO suíte de teste do gspo1 (última etapa da fila)"
scripts/run_suite_models.sh "gspo1|runs/grpo/gspo1/merged"
say "FIM da fila (versão 11)"
