#!/usr/bin/env bash
# Fila de 10/11 (versão 14): segunda rodada do Antidoom, depois da suíte de teste da noite13. Espera a sessão tmux `noite13` acabar (o último passo dela é a
# suíte do `thor_pt_antidoom_of`) e NÃO roda nenhuma suíte de teste: um segundo candidato só vai à suíte com autorização (regra em estudos/07, H64 e H65).
# Motivo do desenho (discovery §20.65): depois da primeira rodada o laço quase some (em `heldout`, o laço oficial do `thor_pt_antidoom_of` é 0,4%; na
# validação, 1 das 17 respostas que não fecham), então uma segunda mineração direta tem material de menos (~30 eventos em 8.000 prompts, contra 3.394 da
# primeira rodada). Por isso a segunda rodada é:
#   1. Sonda (H64): minera 3.000 prompts de múltipla escolha no `thor_pt_antidoom_of` com as configurações oficiais (~25 min). Se achar >= 100 eventos
#      (>= 3,3%, bem acima do esperado), escala: mineração de 8.000 prompts, treino de um LoRA novo sobre o `thor_pt_antidoom_of`, fusão, validação e
#      laço depois. Se achar menos, registra "sem material" e segue.
#   2. Sementes (H65): refaz o treino da primeira rodada com as mesmas 3.394 eventos e duas sementes novas (1 e 2), para medir a variação do resultado
#      (cada uma: ~5 min de treino, fusão, validação e laço depois, ~25 min).
# Uso: scripts/detach.sh noite14 scripts/run_noite14_chain.sh. Cancelar: tmux kill-session -t noite14 (e conferir os processos do vLLM).
# Estado: runs/jobs/noite14.status. NÃO editar com ele rodando.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

status=runs/jobs/noite14.status
A=runs/merged/thor_pt_antidoom_of
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
avaliar() {  # $1 = nome, $2 = modelo; até 3 tentativas (segfault intermitente do vLLM)
  local nome="$1" modelo="$2"
  if avaliacao_completa "$nome"; then say "avaliação de $nome já feita: etapa pulada"; return; fi
  for t in 1 2 3; do
    avaliacao_completa "$nome" && break
    wait_gpu
    say "INICIO avaliação do $nome, tentativa $t"
    VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/eval_vllm.py --name "$nome" --model "$modelo" --tasks validation
  done
  say "FIM avaliação do $nome"
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
  say "FIM mineração $rot: $(python3 -c "import json; d = json.load(open('$saida/stats.json')); print('eventos', d['eventos'], d['taxas'], d['contagens'])" 2>&1 | tail -n 1)"
}
treinar_e_avaliar() {  # $1 = nome do modelo novo, $2 = modelo de partida (fundido), $3 = eventos, $4 = pasta do treino, $5 = pasta do laço depois, resto = args do treino
  local nome="$1" base="$2" eventos="$3" dir="$4" depois="$5"; shift 5
  if [ ! -f "$dir/adapter/adapter_model.safetensors" ]; then
    wait_gpu
    say "INICIO treino anti-laço de $nome"
    PYTHONPATH=src python scripts/antidoom_train.py --recipe oficial --events "$eventos" --base-model "$base" --out "$dir" "$@"
    say "FIM treino anti-laço de $nome (código $?)"
  fi
  [ -f "$dir/adapter/adapter_model.safetensors" ] || { say "FALHA: sem adaptador de $nome"; return; }
  [ -f "runs/merged/$nome/merge_report.json" ] || PYTHONPATH=src python scripts/merge_on_base.py --base "$base" --adapter "$dir/adapter" --out "runs/merged/$nome"
  avaliar "$nome" "runs/merged/$nome"
  minerar "depois, $nome" "runs/merged/$nome" heldout "$depois"
  atualizar_tabela
}

# 0. espera a noite13 (a suíte de teste) acabar
while tmux has-session -t =noite13 2>/dev/null; do sleep 60; done
say "noite13 terminou: segunda rodada do Antidoom"
atualizar_tabela

# 1. sonda da segunda rodada (H64)
minerar "sonda da 2ª rodada em thor_pt_antidoom_of" "$A" mine runs/antidoom/r2_probe --temperature 0.1 --top-k 50 --min-p 0.01 --max-new 4000 --detector oficial --limit 3000
N=$(python3 -c "import json; print(json.load(open('runs/antidoom/r2_probe/stats.json'))['eventos'])" 2>/dev/null || echo 0)
say "sonda: $N eventos em 3.000 prompts (limiar para escalar: 100)"
if [ "${N:-0}" -ge 100 ]; then
  say "material suficiente: escalando a segunda rodada (8.000 prompts, treino e validação)"
  minerar "da 2ª rodada em thor_pt_antidoom_of" "$A" mine runs/antidoom/r2/mine --temperature 0.1 --top-k 50 --min-p 0.01 --max-new 4000 --detector oficial --limit 8000
  [ -s runs/antidoom/r2/mine/events.jsonl ] && treinar_e_avaliar thor_pt_antidoom_r2 "$A" runs/antidoom/r2/mine/events.jsonl runs/antidoom/r2/train runs/antidoom/r2/after_heldout
else
  say "SEM MATERIAL: a segunda mineração achou $N eventos (< 100); a segunda rodada de laço não treina"
fi

# 2. sementes da primeira rodada (H65): os mesmos eventos do thor_pt, sementes 1 e 2
for s in 1 2; do
  treinar_e_avaliar "thor_pt_antidoom_s$s" runs/merged/thor_pt runs/antidoom/thor_pt/mine/events.jsonl "runs/antidoom/thor_pt_s$s/train" "runs/antidoom/thor_pt_s$s/after_heldout" --seed "$s"
done
say "FIM da fila (versão 14); nenhuma suíte de teste foi rodada"
