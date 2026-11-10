#!/usr/bin/env bash
# Fila de 10/11 (versão 18): o plano de provas novas do usuário (plano_benchmarks_finais.md), prioridade máxima; substitui a versão 17, que nunca usou a GPU.
# Provas novas = `oab_novas` (OAB 2018 a 2025), `enem_2024` e três fatias do MMLU-PT (`mmlu_pt_exatas`, `mmlu_pt_enade`, `mmlu_pt_concursos`), montadas
# por scripts/build_novas_provas.py (data/novas_provas/). São checagem externa: não trocam o candidato nem a Nota. Regras de leitura em estudos/07 (H74).
# Reordenada às 09:05 de 10/11 (antes de a fila usar a GPU): o usuário colocou o pacote handoff/thor5 na Thor, que roda o Lumi e o Nemotron nas provas novas e o
# Nemotron na suíte do PRD (resultados voltam como runs/eval/base_thor e runs/eval/nemotron4b_thor, ver scripts/import_thor_eval.py). Então a 4090 faz primeiro o
# que só ela pode fazer (o T6-Lumi) e depois a confirmação na mesma máquina; o que a Thor já devolveu é pulado na 4090.
# Ordem:
#   0. teste de fumaça (Lumi, 20 itens de cada prova nova, ~3 min)
#   1. T6-Lumi (thinking) nas cinco provas novas                                                    (~45 min)
#   2. Lumi nas provas novas, os dois modos (comparação na mesma máquina, a principal pela regra H66) (~1 h)
#   3. H67: orçamento de 8.192 tokens, Lumi e T6-Lumi, em oab_s800, BLUEX, ENEM e mmlu_prox_diag_pt   (~2 h)
#   4. H69: as três outras execuções da receita (thor_pt_antidoom, _s1, _s2) nas provas novas        (~2 h)
#   5. H68: amostragem T 0,6 (top-p 0,95) do Lumi e do T6-Lumi na `oab_novas`                         (~0,5 h)
#   6. Nemotron nas provas novas e na suíte do PRD (GSM8K-PT, ENEM, BLUEX, OAB, MMLU-ProX), dois modos: só o que a Thor ainda não devolveu (~1 a 5 h)
# A GPU é compartilhada: cada passo espera ela ficar com <= 1 GiB em uso. Se existir runs/jobs/gpu_corun.flag (autorização do usuário para dividir a
# GPU), com o conteúdo "<limite de MiB em uso> <fração --mem>", o limite e a fração passam a valer na hora (sem editar este arquivo).
# Tabelas em runs/jobs/noite18_comparacao.md (reescritas depois de cada etapa). Uso: scripts/detach.sh noite18 scripts/run_noite18_chain.sh.
# Cancelar: tmux kill-session -t noite18 (e conferir os processos do vLLM). Estado: runs/jobs/noite18.status. NÃO editar com ele rodando.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

status=runs/jobs/noite18.status
NOVAS=oab_novas,enem_2024,mmlu_pt_exatas,mmlu_pt_enade,mmlu_pt_concursos
P1=oab_novas,enem_2024
MM=mmlu_pt_exatas,mmlu_pt_enade,mmlu_pt_concursos
SUITE_NEM=gsm8k_test,enem,bluex,oab
H8K=oab_s800,bluex,enem,mmlu_prox_diag_pt
T6=runs/merged/thor_pt_antidoom_of
NEMOTRON=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
FLAG=runs/jobs/gpu_corun.flag
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }
limite_mib() { if [ -s "$FLAG" ]; then awk '{print $1}' "$FLAG"; else echo 1024; fi; }
fracao_mem() { if [ -s "$FLAG" ]; then awk '{print $2}' "$FLAG"; else echo 0.88; fi; }
wait_gpu() {
  local avisou=0
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt "$(limite_mib)" ]; do
    [ "$avisou" = 0 ] && { say "GPU ocupada (outro processo), aguardando"; avisou=1; }
    sleep 60
  done
}
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
  if completo "$nome" "$tarefas" "$modos"; then say "avaliação $nome ($tarefas; $modos) já feita: pulada"; return; fi
  for t in 1 2 3; do
    completo "$nome" "$tarefas" "$modos" && break
    wait_gpu
    say "INICIO avaliação $nome ($tarefas; $modos) $*, tentativa $t"
    VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/eval_vllm.py --name "$nome" "${margs[@]}" --tasks "$tarefas" --modes "$modos" --mem "$(fracao_mem)" "$@"
  done
  if completo "$nome" "$tarefas" "$modos"; then say "FIM avaliação $nome ($tarefas)"; else say "FALHA: avaliação $nome incompleta ($tarefas)"; fi
}
tabelas() {
  local R=runs/jobs/noite18_comparacao.md; : > "$R"
  rc() { local t="$1"; shift; python scripts/report_compare.py "$@" --tasks "$t" --mean >> "$R" 2>&1; echo >> "$R"; }
  local COLS=(--col "Lumi com thinking=base:thinking" --col "Lumi sem thinking=base:nothinking" --col "Nemotron com thinking=nemotron4b:thinking"
              --col "Nemotron sem thinking=nemotron4b:nothinking" --col "T6-Lumi=thor_pt_antidoom_of:thinking"
              --col "Lumi com thinking (Thor)=base_thor:thinking" --col "Lumi sem thinking (Thor)=base_thor:nothinking"
              --col "Nemotron com thinking (Thor)=nemotron4b_thor:thinking" --col "Nemotron sem thinking (Thor)=nemotron4b_thor:nothinking")
  local EST="oab_novas=data/novas_provas/oab_novas_estrito_ids.json,enem_2024=data/novas_provas/enem_2024_estrito_ids.json,mmlu_pt_exatas=data/novas_provas/mmlu_pt_exatas_estrito_ids.json,mmlu_pt_enade=data/novas_provas/mmlu_pt_enade_estrito_ids.json,mmlu_pt_concursos=data/novas_provas/mmlu_pt_concursos_estrito_ids.json"
  echo "# Fila 18 (reescrita em $(date +%d/%m\ %H:%M)); 4.096 tokens, gulosa, 4090. Provas novas = $NOVAS" >> "$R"
  for ref in "Lumi com thinking" "Lumi sem thinking" "Nemotron com thinking" "Lumi com thinking (Thor)" "Lumi sem thinking (Thor)" "Nemotron com thinking (Thor)"; do
    echo "## Provas novas, referência: $ref" >> "$R"; rc "$NOVAS" --ref "$ref" "${COLS[@]}"
  done
  echo "## Provas novas, subconjunto estrito (nenhum 8-grama em comum com a suíte, a validação e o treino), referência: Lumi com thinking" >> "$R"
  rc "$NOVAS" --ref "Lumi com thinking" "${COLS[@]}" --subset-ids "$EST"
  echo "## Suíte do PRD com o Nemotron (a coluna do Nemotron só fecha com o MMLU-ProX), referência: Lumi com thinking" >> "$R"
  rc "gsm8k_test,mmlu_prox_test,enem,bluex,oab" --ref "Lumi com thinking" "${COLS[@]}"
  echo "## H69: variantes da receita nas provas novas, referência: Lumi com thinking" >> "$R"
  rc "$NOVAS" --ref "Lumi com thinking" --col "Lumi com thinking=base:thinking" --col "T6-Lumi=thor_pt_antidoom_of:thinking" \
     --col "portagem, semente 0=thor_pt_antidoom:thinking" --col "portagem, semente 1=thor_pt_antidoom_s1:thinking" --col "portagem, semente 2=thor_pt_antidoom_s2:thinking"
  echo "## H68: amostragem T 0,6 na oab_novas" >> "$R"
  rc "oab_novas" --ref "Lumi T 0,6" --col "Lumi T 0,6=base_t06:thinking" --col "T6-Lumi T 0,6=thor_pt_antidoom_of_t06:thinking"
  echo "## H67: 8.192 tokens ($H8K)" >> "$R"
  rc "$H8K" --ref "Lumi 8.192" --col "Lumi 8.192=base_8k:thinking" --col "T6-Lumi 8.192=thor_pt_antidoom_of_8k:thinking"
  echo "## Decomposição pareada por quadrante de fechamento (contra o Lumi com thinking e contra o Nemotron com thinking)" >> "$R"
  for t in ${NOVAS//,/ }; do
    for ref in base nemotron4b; do
      [ -f "runs/eval/$ref/${t}_thinking_scores.json" ] && [ -f "runs/eval/thor_pt_antidoom_of/${t}_thinking_scores.json" ] \
        && python scripts/decompose_paired.py --task "$t" --ref "$ref" --names thor_pt_antidoom_of >> "$R" 2>&1
    done
  done
  PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true
}

say "INICIO da fila (versão 18)"

# 0. teste de fumaça: o Lumi, 20 itens de cada prova nova; se falhar, a fila para
SM=runs/eval_smoke/smoke_novas/summary.json
if ! python3 -c "import json,sys; d=json.load(open('$SM')); sys.exit(0 if all(t+'|thinking' in d for t in '$NOVAS'.split(',')) else 1)" 2>/dev/null; then
  wait_gpu
  say "INICIO teste de fumaça (Lumi, 20 itens por prova nova)"
  VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/eval_vllm.py --name smoke_novas --tasks "$NOVAS" --modes thinking --limit 20 --out runs/eval_smoke --mem "$(fracao_mem)"
fi
if python3 -c "import json,sys; d=json.load(open('$SM')); sys.exit(0 if all(t+'|thinking' in d for t in '$NOVAS'.split(',')) else 1)" 2>/dev/null; then
  say "FIM teste de fumaça: $(python3 -c "import json; d=json.load(open('$SM')); print({k.split('|')[0]: (v['n'], v['strict_acc'], v['truncated_rate']) for k, v in d.items() if '|' in k})")"
else
  say "FALHA: teste de fumaça das provas novas; a fila para"; exit 1
fi

# 1. T6-Lumi nas provas novas
avaliar thor_pt_antidoom_of "$T6" "$NOVAS" thinking
tabelas
# 2. Lumi nas provas novas, na 4090
avaliar base - "$NOVAS" thinking,nothinking
tabelas
# 3. H67: orçamento de 8.192 tokens
avaliar base_8k - "$H8K" thinking --max-new 8192 --max-len 10240
avaliar thor_pt_antidoom_of_8k "$T6" "$H8K" thinking --max-new 8192 --max-len 10240
tabelas
# 4. H69: as outras execuções da receita nas provas novas
for n in thor_pt_antidoom thor_pt_antidoom_s1 thor_pt_antidoom_s2; do
  avaliar "$n" "runs/merged/$n" "$NOVAS" thinking
done
tabelas
# 5. H68: amostragem na oab_novas
avaliar base_t06 - oab_novas thinking --temperature 0.6 --top-p 0.95 --sample-seed 0
avaliar thor_pt_antidoom_of_t06 "$T6" oab_novas thinking --temperature 0.6 --top-p 0.95 --sample-seed 0
tabelas
# 6. Nemotron: só o que a Thor ainda não devolveu (nemotron4b_thor)
nemotron_se_falta() {  # $1 = tarefas
  if completo nemotron4b_thor "$1" thinking,nothinking; then say "Nemotron ($1) já veio da Thor: pulado na 4090"; else avaliar nemotron4b "$NEMOTRON" "$1" thinking,nothinking; fi
}
nemotron_se_falta "$P1"
nemotron_se_falta "$MM"
nemotron_se_falta "$SUITE_NEM"
nemotron_se_falta mmlu_prox_test
tabelas
say "FIM da fila (versão 18); nenhum candidato foi trocado"
