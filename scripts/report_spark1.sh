#!/usr/bin/env bash
# Tabelas do pacote handoff/spark1 (depois de `python scripts/import_thor_eval.py --origem spark --tar spark1_saida.tar.gz`): T6-Lumi e Lumi medidos na
# mesma máquina (a Spark) nas provas novas, mais a calibração contra o Lumi da Thor e, quando existirem, os da 4090. Escreve runs/jobs/spark1_comparacao.md.
# Colunas sem resultado aparecem como "(pendente)". Regras de leitura: estudos/07 (H73, H74). Uso, na raiz: bash scripts/report_spark1.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
R=runs/jobs/spark1_comparacao.md; : > "$R"
NOVAS=oab_novas,enem_2024,mmlu_pt_exatas,mmlu_pt_enade,mmlu_pt_concursos
rc() { local t="$1"; shift; python scripts/report_compare.py "$@" --tasks "$t" --mean >> "$R" 2>&1; echo >> "$R"; }
COLS=(--col "Lumi com thinking (Spark)=base_spark:thinking" --col "Lumi sem thinking (Spark)=base_spark:nothinking" --col "T6-Lumi (Spark)=thor_pt_antidoom_of_spark:thinking"
      --col "Lumi com thinking (Thor)=base_thor:thinking" --col "Lumi sem thinking (Thor)=base_thor:nothinking"
      --col "Lumi com thinking (4090)=base:thinking" --col "Lumi sem thinking (4090)=base:nothinking" --col "T6-Lumi (4090)=thor_pt_antidoom_of:thinking")
EST="oab_novas=data/novas_provas/oab_novas_estrito_ids.json,enem_2024=data/novas_provas/enem_2024_estrito_ids.json,mmlu_pt_exatas=data/novas_provas/mmlu_pt_exatas_estrito_ids.json,mmlu_pt_enade=data/novas_provas/mmlu_pt_enade_estrito_ids.json,mmlu_pt_concursos=data/novas_provas/mmlu_pt_concursos_estrito_ids.json"
echo "# Spark (pacote 6), escrito em $(date +%d/%m\ %H:%M); 4.096 tokens e gulosa, salvo onde o título diz outra coisa. Provas novas = $NOVAS" >> "$R"
for ref in "Lumi com thinking (Spark)" "Lumi sem thinking (Spark)"; do
  echo "## Provas novas, mesma máquina (Spark), referência: $ref" >> "$R"; rc "$NOVAS" --ref "$ref" "${COLS[@]}"
done
echo "## Calibração entre máquinas: o Lumi da Spark contra o Lumi da Thor (diferença > 1,0 ponto em alguma prova e modo = máquinas não equivalentes)" >> "$R"
rc "$NOVAS" --ref "Lumi com thinking (Thor)" "${COLS[@]}"
rc "$NOVAS" --ref "Lumi sem thinking (Thor)" "${COLS[@]}"
echo "## Provas novas, subconjunto estrito (nenhum 8-grama em comum com a suíte, a validação e o treino), referência: Lumi com thinking (Spark)" >> "$R"
rc "$NOVAS" --ref "Lumi com thinking (Spark)" "${COLS[@]}" --subset-ids "$EST"
echo "## Decomposição pareada por quadrante de fechamento, T6-Lumi contra o Lumi com thinking, na Spark" >> "$R"
for t in ${NOVAS//,/ }; do
  [ -f "runs/eval/base_spark/${t}_thinking_scores.json" ] && [ -f "runs/eval/thor_pt_antidoom_of_spark/${t}_thinking_scores.json" ] \
    && python scripts/decompose_paired.py --task "$t" --ref base_spark --names thor_pt_antidoom_of_spark >> "$R" 2>&1
done
echo "escrito: $R"
