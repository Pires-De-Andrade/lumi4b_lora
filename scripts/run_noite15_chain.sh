#!/usr/bin/env bash
# Fila de 10/11 (versão 15): o Lumi original na 4090 em MMLU-ProX PT (só o modo sem thinking), ENEM, BLUEX e OAB (os dois modos), depois que a `noite14`
# acabar (que espera a suíte de teste da `noite13`). Motivo (discovery §20.66, H66): a suíte do Lumi que existe para essas provas foi rodada na Jetson Thor,
# e a dos nossos modelos roda na 4090. O Lumi rodado nas duas máquinas concorda na média (GSM8K-PT de teste: 88,73% e 88,96%; MMLU-ProX PT: 53,57% e 53,54%),
# mas discorda em 57 dos 1.295 itens do GSM8K (4,4%): a numérica diferente muda a decodificação gulosa de alguns itens. O IC pareado não inclui esse ruído.
# Com o Lumi também na 4090, a comparação do relatório sai na mesma máquina, nas cinco provas.
# Os DOIS modos do Lumi, porque o PRD compara o nosso modelo (com thinking) com o Lumi nos dois: a Nota de raciocínio tem de ter o IC da diferença acima de 0
# contra o Lumi com thinking e contra o Lumi sem thinking, e o GSM8K-PT é medido contra o Lumi sem thinking (+3,0 com IC acima de 0). Na 4090 o `base` já tem
# gsm8k_test nos dois modos e mmlu_prox_test com thinking; faltam o resto (eval_vllm.py pula o que já está no summary.json).
# NÃO roda candidato nenhum: é a referência (o `base`), não uma segunda suíte de teste de modelo nosso. Os resultados entram em runs/eval/base/summary.json.
# Estimativa pela contagem de tokens da rodada da Thor: com thinking (ENEM, BLUEX, OAB) ~9,0 milhões de tokens e sem thinking (as quatro provas) ~6,1 milhões;
# a ~2.900 tok/s são ~52 min + ~35 min; reserve ~1,5 a 2 h com a carga do modelo.
# Ao fim, se a suíte do `thor_pt_antidoom_of` estiver completa, grava em runs/jobs/noite15_comparacao.md duas tabelas (IC pareado por prova e na nota média das
# cinco): a do nosso modelo contra o Lumi COM thinking e a contra o Lumi SEM thinking, ambos na 4090, com o Lumi da Thor ao lado. Senão, só registra que falta.
# Uso: scripts/detach.sh noite15 scripts/run_noite15_chain.sh. Cancelar: tmux kill-session -t noite15 (e conferir os processos do vLLM).
# Estado: runs/jobs/noite15.status. NÃO editar com ele rodando.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

status=runs/jobs/noite15.status
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }
wait_gpu() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
    sleep 120
  done
}
completo() {  # $1 = nome em runs/eval, resto = tarefas: todas estão no summary.json nos dois modos?
  local nome="$1"; shift
  python3 - "$nome" "$@" <<'PY'
import json, sys
try:
    d = json.load(open(f"runs/eval/{sys.argv[1]}/summary.json"))
    assert all(f"{t}|{m}" in d for t in sys.argv[2:] for m in ("thinking", "nothinking"))
except Exception:
    sys.exit(1)
PY
}
TAREFAS="gsm8k_test mmlu_prox_test enem bluex oab"

# 0. espera a noite14 acabar (e, por ela, a suíte da noite13)
while tmux has-session -t =noite14 2>/dev/null; do sleep 60; done
say "noite14 terminou: Lumi na 4090 em MMLU-ProX PT (sem thinking), ENEM, BLUEX e OAB (os dois modos)"

# 1. o Lumi original, o que falta nas cinco provas, nos dois modos; até 3 tentativas (segfault intermitente do vLLM)
for t in 1 2 3; do
  completo base $TAREFAS && break
  wait_gpu
  say "INICIO Lumi (base) na 4090: o que falta de $TAREFAS, os dois modos; tentativa $t"
  VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/eval_vllm.py --name base --tasks mmlu_prox_test,enem,bluex,oab --modes thinking,nothinking
done
completo base $TAREFAS || { say "FALHA: o Lumi na 4090 não completou as cinco provas nos dois modos"; exit 1; }
say "FIM Lumi (base) na 4090: as cinco provas nos dois modos"
PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true

# 2. tabelas de comparação, se a suíte do thor_pt_antidoom_of estiver completa
if completo thor_pt_antidoom_of $TAREFAS; then
  cols=(--col "Lumi com thinking (4090)=base:thinking" --col "Lumi sem thinking (4090)=base:nothinking" --col "Lumi com thinking (Thor)=base_thor:thinking"
        --col "Lumi sem thinking (Thor)=base_thor:nothinking" --col "thor_pt_antidoom_of (com thinking)=thor_pt_antidoom_of:thinking")
  : > runs/jobs/noite15_comparacao.md
  for ref in "Lumi com thinking (4090)" "Lumi sem thinking (4090)"; do
    python scripts/report_compare.py --ref "$ref" "${cols[@]}" --tasks ${TAREFAS// /,} --mean >> runs/jobs/noite15_comparacao.md
    echo >> runs/jobs/noite15_comparacao.md
  done
  say "tabelas de comparação em runs/jobs/noite15_comparacao.md"
else
  say "a suíte do thor_pt_antidoom_of não está completa: sem tabelas de comparação"
fi
say "FIM da fila (versão 15); nenhum modelo nosso foi avaliado"
