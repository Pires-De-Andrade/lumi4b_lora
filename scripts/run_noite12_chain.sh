#!/usr/bin/env bash
# Fila de 09/11 (versão 12). Substitui o run_noite11_chain.sh (encerrada antes de começar) e a run_noite10_chain.sh (congelada com SIGSTOP; esta fila
# espera os processos filhos dela acabarem). Cada etapa é retomável e pulada se já pronta. A 4090 é compartilhada: cada etapa espera a GPU ficar livre.
# Ordem, pelo ganho esperado (discovery §20.59 a §20.61):
#   0. Espera a mineração do `gspo1` (noite10) acabar e encerra a noite10.
#   1. Enquanto o código da receita oficial do Antidoom não fica pronto (arquivo runs/jobs/antidoom_v2.pronto, criado depois dos testes), a GPU roda os
#      SFTs de teste `qwen_mix_short` (H53), `selfdist_min` e `selfdist_300` (H54), um por vez, conferindo o arquivo entre eles.
#   2. Teste de fumaça do treino na receita oficial.
#   3. Fase anti-laço na receita oficial sobre o `thor_pt` (H59: 33% de laço exato na validação, teto grosseiro de 71,7%): referência de laço em `heldout`
#      (gulosa), mineração em 10.000 prompts (temperatura 0,1, top-k 50, min-p 0,01, 4.000 tokens, detector oficial), treino (LoRA novo r=256 com
#      `lm_head`; se falhar, r=128), fusão sobre o `thor_pt`, avaliação da validação e taxa de laço depois.
#   4. O mesmo sobre o `gspo_thor` (H58: 11% de laço, teto grosseiro de 70,6%), mineração em 8.000 prompts de múltipla escolha.
#   5. Braço de conferência com o treinador do repositório oficial nos eventos do `thor_pt` (só se runs/jobs/antidoom_trainer_oficial.pronto existir).
#   6. SFTs de teste que não couberam no passo 1.
#   7. Fase anti-laço na receita oficial sobre o `gspo1` (H57, teto de ~+0,6 ponto), com os eventos já minerados.
#   8. Suíte de teste de UM candidato, a última etapa: o que passar a regra pré-registrada (scripts/pick_suite_candidate.py); senão o `gspo1`.
# Uso: scripts/detach.sh noite12 scripts/run_noite12_chain.sh. Cancelar: tmux kill-session -t noite12 (e conferir os processos do vLLM).
# Estado: runs/jobs/noite12.status. NÃO editar com ele rodando.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

status=runs/jobs/noite12.status
FLAG=runs/jobs/antidoom_v2.pronto
FLAG_OF=runs/jobs/antidoom_trainer_oficial.pronto
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
avaliar() {  # $1 = nome, $2 = modelo, resto = argumentos extras do eval_vllm.py; até 3 tentativas (segfault intermitente do vLLM)
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
  if avaliacao_completa "$1"; then return; fi
  wait_gpu
  say "INICIO SFT $1 ($2)"
  if [ ! -f "runs/train/$1/adapter/adapter_config.json" ]; then
    scripts/run_experiment.sh "$1" --arm lora --data "$2" --epochs 1 --lr 1e-4
  fi
  [ -f "runs/merged/$1/merge_report.json" ] || PYTHONPATH=src python scripts/merge_adapter.py --adapter "runs/train/$1/adapter" --out "runs/merged/$1"
  avaliar "$1" "runs/merged/$1"
  say "FIM SFT $1"
  atualizar_tabela
}
minerar() {  # $1 = rótulo, $2 = modelo, $3 = split, $4 = pasta de saída, resto = argumentos extras; até 2 tentativas
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
SFTS=("qwen_mix_short data/qwen_mix_short.parquet" "selfdist_min data/selfdist_min.parquet" "selfdist_300 data/selfdist_300.parquet")
sfts_pendentes() {  # $1 = "ate_flag" (para quando o código ficar pronto) ou "todos"
  local s
  for s in "${SFTS[@]}"; do
    [ "$1" = "ate_flag" ] && [ -f "$FLAG" ] && return
    sft_e_avaliar ${s}
  done
}
SMOKE_OK=0
fumaca_v2() {
  if [ -f runs/antidoom/smoke_v2/summary.json ]; then SMOKE_OK=1; return; fi
  wait_gpu
  say "INICIO teste de fumaça do treino anti-laço (receita oficial)"
  if PYTHONPATH=src python scripts/antidoom_train.py --recipe oficial --smoke --events runs/antidoom/r1/mine/events.jsonl \
       --base-model runs/merged/gspo1_c8 --out runs/antidoom/smoke_v2; then
    say "FIM teste de fumaça (receita oficial): ok"; SMOKE_OK=1
  else
    say "FALHA no teste de fumaça (receita oficial): as minerações seguem, os treinos não"
  fi
}
antilaco_v2() {  # $1 = nome do modelo novo, $2 = modelo de partida (pasta fundida), $3 = pasta da fase, resto = argumentos extras da mineração
  local nome="$1" base="$2" dir="$3"; shift 3
  minerar "referência para $nome" "$base" heldout "$dir/base_heldout"
  minerar "para $nome" "$base" mine "$dir/mine" --temperature 0.1 --top-k 50 --min-p 0.01 --max-new 4000 --detector oficial "$@"
  [ "$SMOKE_OK" = 1 ] || { say "treino de $nome não roda (teste de fumaça falhou)"; return; }
  [ -s "$dir/mine/events.jsonl" ] || { say "sem eventos em $dir/mine: $nome pulado"; return; }
  if [ ! -f "$dir/train/adapter/adapter_model.safetensors" ]; then
    wait_gpu
    say "INICIO treino anti-laço (receita oficial) de $nome"
    PYTHONPATH=src python scripts/antidoom_train.py --recipe oficial --events "$dir/mine/events.jsonl" --base-model "$base" --out "$dir/train"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
      say "treino de $nome falhou (código $rc): nova tentativa com LoRA r=128"
      wait_gpu
      PYTHONPATH=src python scripts/antidoom_train.py --recipe oficial --lora-r 128 --lora-alpha 64 --events "$dir/mine/events.jsonl" --base-model "$base" --out "$dir/train"
    fi
    say "FIM treino anti-laço de $nome"
  fi
  [ -f "$dir/train/adapter/adapter_model.safetensors" ] || { say "FALHA: sem adaptador de $nome"; return; }
  [ -f "runs/merged/$nome/merge_report.json" ] || PYTHONPATH=src python scripts/merge_on_base.py --base "$base" --adapter "$dir/train/adapter" --out "runs/merged/$nome"
  avaliar "$nome" "runs/merged/$nome"
  minerar "depois, $nome" "runs/merged/$nome" heldout "$dir/after_heldout"
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

# 1. SFTs de teste enquanto o código da receita oficial não fica pronto
sfts_pendentes ate_flag
if [ ! -f "$FLAG" ]; then
  say "aguardando o código da receita oficial ($FLAG)"
  while [ ! -f "$FLAG" ]; do sleep 60; done
fi
say "código da receita oficial pronto: fase anti-laço"

# 2. teste de fumaça
fumaca_v2

# 3 e 4. fase anti-laço na receita oficial: thor_pt (mais laço) e gspo_thor
antilaco_v2 thor_pt_antidoom runs/merged/thor_pt runs/antidoom/thor_pt --limit 10000
antilaco_v2 gspo_thor_antidoom runs/grpo/gspo_thor/merged runs/antidoom/gspo_thor --limit 8000

# 5. braço de conferência com o treinador do repositório oficial
if [ -f "$FLAG_OF" ] && [ "$SMOKE_OK" = 1 ] && [ -s runs/antidoom/thor_pt/mine/events.jsonl ]; then
  if [ ! -f runs/antidoom/thor_pt_of/adapter/adapter_model.safetensors ]; then
    wait_gpu
    say "INICIO treino com o treinador do repositório oficial (eventos do thor_pt)"
    PYTHONPATH=src python scripts/antidoom_oficial_train.py --events runs/antidoom/thor_pt/mine/events.jsonl --base-model runs/merged/thor_pt \
      --out runs/antidoom/thor_pt_of
    say "FIM treino com o treinador oficial (código $?)"
  fi
  if [ -f runs/antidoom/thor_pt_of/adapter/adapter_model.safetensors ]; then
    [ -f runs/merged/thor_pt_antidoom_of/merge_report.json ] || PYTHONPATH=src python scripts/merge_on_base.py --base runs/merged/thor_pt \
      --adapter runs/antidoom/thor_pt_of/adapter --out runs/merged/thor_pt_antidoom_of
    avaliar thor_pt_antidoom_of runs/merged/thor_pt_antidoom_of
    minerar "depois, thor_pt_antidoom_of" runs/merged/thor_pt_antidoom_of heldout runs/antidoom/thor_pt_of/after_heldout
    atualizar_tabela
  fi
else
  say "braço do treinador oficial não roda (falta $FLAG_OF, o teste de fumaça ou os eventos do thor_pt)"
fi

# 6. SFTs de teste restantes
sfts_pendentes todos

# 7. fase anti-laço no gspo1, com os eventos já minerados (receita oficial no treino)
if [ "$SMOKE_OK" = 1 ] && [ -s runs/antidoom/r1/mine/events.jsonl ]; then
  if [ ! -f runs/antidoom/r1/train_v2/adapter/adapter_model.safetensors ]; then
    wait_gpu
    say "INICIO treino anti-laço (receita oficial) do gspo1"
    PYTHONPATH=src python scripts/antidoom_train.py --recipe oficial --events runs/antidoom/r1/mine/events.jsonl --base-model runs/merged/gspo1_c8 \
      --out runs/antidoom/r1/train_v2
    say "FIM treino anti-laço do gspo1 (código $?)"
  fi
  if [ -f runs/antidoom/r1/train_v2/adapter/adapter_model.safetensors ]; then
    [ -f runs/merged/gspo1_antidoom/merge_report.json ] || PYTHONPATH=src python scripts/merge_on_base.py --base runs/merged/gspo1_c8 \
      --adapter runs/antidoom/r1/train_v2/adapter --out runs/merged/gspo1_antidoom
    avaliar gspo1_antidoom runs/merged/gspo1_antidoom
    minerar "depois, gspo1_antidoom" runs/merged/gspo1_antidoom heldout runs/antidoom/r1/after_heldout
    atualizar_tabela
  fi
fi

# 8. suíte de teste de um candidato, a última de todas
CAND=$(PYTHONPATH=src python scripts/pick_suite_candidate.py 2>> "$status" | tail -n 1)
[ -n "$CAND" ] || CAND="gspo1|runs/grpo/gspo1/merged"
wait_gpu
say "INICIO suíte de teste de ${CAND%%|*} (última etapa da fila; regra em estudos/07)"
scripts/run_suite_models.sh "$CAND"
say "FIM da fila (versão 12)"
