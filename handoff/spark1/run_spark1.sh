#!/usr/bin/env bash
# Fila da DGX Spark (pacote 6, `spark1`), 10/11: o T6-Lumi (thor_pt_antidoom_of) e o Lumi, na mesma máquina, nas cinco provas novas do plano de 10/11.
# Mesmo protocolo da suíte: vLLM, gulosa, 4.096 tokens, `scripts/eval_vllm.py` sem alteração. Estudos: estudos/07 (H74). Normalmente é chamada por
# `bash handoff/spark1/rodar.sh`, que prepara tudo e a lança em segundo plano.
# Ordem:
#   0. teste de fumaça (Lumi, 20 itens por prova nova)                                     (~5 min; a fila para se falhar)
#   1. oab_novas + enem_2024: T6-Lumi (thinking), depois Lumi (os dois modos)               (~2,5 h)
#   2. as três fatias do MMLU-PT: T6-Lumi (thinking), depois Lumi (os dois modos)           (~2,5 h)
# As estimativas vêm dos tempos da 4090 vezes 3 (a razão da Thor no MMLU-ProX); a Spark não foi medida.
# Variáveis: PY (python do vLLM, padrão .venv-eval/bin/python), MEM (fração da memória, padrão 0.5), SEQS (sequências simultâneas, padrão 256),
#            DRY=1 (só mostra os comandos). Parar depois do passo em andamento: `touch runs/jobs/spark1.stop`.
# O fim de cada par grava spark1_saida.tar.gz (+ .sha256) na raiz: pode ser baixado a qualquer hora (entrega parcial).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-.venv-eval/bin/python}"
MEM="${MEM:-0.5}"
SEQS="${SEQS:-256}"
DRY="${DRY:-0}"
mkdir -p runs/jobs out
status=runs/jobs/spark1.status

NOVAS=oab_novas,enem_2024,mmlu_pt_exatas,mmlu_pt_enade,mmlu_pt_concursos
P1=oab_novas,enem_2024
MM=mmlu_pt_exatas,mmlu_pt_enade,mmlu_pt_concursos
T6=runs/merged/thor_pt_antidoom_of
say() { if [ "$DRY" = 1 ]; then echo "$(date +%d/%m\ %H:%M) $*"; else echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; fi; }
completo() {  # $1 = nome em runs/eval, $2 = tarefas (vírgulas), $3 = modos (vírgulas)
  python3 - "$1" "$2" "$3" <<'PY'
import json, sys
try:
    d = json.load(open(f"runs/eval/{sys.argv[1]}/summary.json"))
    assert all(f"{t}|{m}" in d for t in sys.argv[2].split(",") for m in sys.argv[3].split(","))
except Exception:
    sys.exit(1)
PY
}
avaliar() {  # $1 = nome, $2 = modelo ("-" = Lumi original), $3 = tarefas, $4 = modos, resto = argumentos do eval_vllm.py; até 3 tentativas (segfault do vLLM)
  local nome="$1" modelo="$2" tarefas="$3" modos="$4"; shift 4
  local margs=()
  [ "$modelo" != "-" ] && margs=(--model "$modelo")
  if [ -e runs/jobs/spark1.stop ]; then say "PARADA pedida (runs/jobs/spark1.stop): fila encerrada antes de $nome"; empacotar; exit 0; fi
  if completo "$nome" "$tarefas" "$modos"; then say "avaliação $nome ($tarefas; $modos) já feita: pulada"; return; fi
  if [ "$DRY" = 1 ]; then echo "[DRY] VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src $PY scripts/eval_vllm.py --name $nome ${margs[*]} --tasks $tarefas --modes $modos --mem $MEM --max-seqs $SEQS $*"; return; fi
  for t in 1 2 3; do
    completo "$nome" "$tarefas" "$modos" && break
    say "INICIO avaliação $nome ($tarefas; $modos), tentativa $t"
    VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src "$PY" scripts/eval_vllm.py --name "$nome" "${margs[@]}" --tasks "$tarefas" --modes "$modos" \
      --mem "$MEM" --max-seqs "$SEQS" "$@"
  done
  if completo "$nome" "$tarefas" "$modos"; then say "FIM avaliação $nome ($tarefas)"; else say "FALHA: avaliação $nome incompleta ($tarefas)"; fi
}
empacotar() {  # devolução parcial: tudo o que já saiu, num arquivo só (escrito ao lado e renomeado: nunca fica pela metade)
  [ "$DRY" = 1 ] && return
  cp -f "$status" out/spark1.status 2>/dev/null || true
  cp -f runs/jobs/spark1.log out/spark1.log 2>/dev/null || true
  local itens=()
  for d in runs/eval/*_spark runs/eval_smoke out; do [ -e "$d" ] && itens+=("$d"); done
  [ "${#itens[@]}" -gt 0 ] || return 0
  tar czf spark1_saida.tar.gz.tmp "${itens[@]}" 2>/dev/null
  if [ "$?" -le 1 ]; then
    mv -f spark1_saida.tar.gz.tmp spark1_saida.tar.gz && sha256sum spark1_saida.tar.gz > spark1_saida.sha256
    say "devolução atualizada: spark1_saida.tar.gz ($(du -h spark1_saida.tar.gz | cut -f1)), sha256 em spark1_saida.sha256"
  else
    say "AVISO: não consegui gerar spark1_saida.tar.gz"
  fi
}

say "INICIO da fila spark1 (PY=$PY MEM=$MEM SEQS=$SEQS DRY=$DRY)"
if [ "$DRY" != 1 ]; then
  [ -s "$T6/model.safetensors" ] || { say "FALHA: faltam os pesos do T6-Lumi em $T6/model.safetensors (ver README, passo 2)"; exit 1; }
  [ -s data/novas_provas/oab_novas.jsonl ] || { say "FALHA: faltam as provas novas em data/novas_provas/"; exit 1; }
fi

# 0. teste de fumaça: o Lumi, 20 itens de cada prova nova (confere vLLM, Hub e provas); se falhar, a fila para
SM=runs/eval_smoke/smoke_novas/summary.json
ok_fumaca() { python3 -c "import json,sys; d=json.load(open('$SM')); sys.exit(0 if all(t+'|thinking' in d for t in '$NOVAS'.split(',')) else 1)" 2>/dev/null; }
if [ "$DRY" = 1 ]; then
  echo "[DRY] fumaça: $PY scripts/eval_vllm.py --name smoke_novas --tasks $NOVAS --modes thinking --limit 20 --out runs/eval_smoke"
else
  ok_fumaca || { say "INICIO teste de fumaça (Lumi, 20 itens por prova nova)"
    VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src "$PY" scripts/eval_vllm.py --name smoke_novas --tasks "$NOVAS" --modes thinking --limit 20 \
      --out runs/eval_smoke --mem "$MEM" --max-seqs "$SEQS"; }
  if ok_fumaca; then
    say "FIM teste de fumaça: $(python3 -c "import json; d=json.load(open('$SM')); print({k.split('|')[0]: (v['n'], v['strict_acc'], v['truncated_rate']) for k, v in d.items() if '|' in k})")"
  else
    say "FALHA: teste de fumaça das provas novas; a fila para (ver runs/jobs/spark1.log)"; empacotar; exit 1
  fi
fi

# 1. provas novas: T6-Lumi (thinking) e Lumi (os dois modos), em dois blocos
for grupo in "$P1" "$MM"; do
  avaliar thor_pt_antidoom_of_spark "$T6" "$grupo" thinking
  avaliar base_spark - "$grupo" thinking,nothinking
  empacotar
done

say "FIM da fila spark1"
