#!/usr/bin/env bash
# Cadeia do RL a partir do T6-Lumi (H75, estudos/07). Espera a fila `noite18` terminar e a GPU ficar livre; depois, em ordem:
#   1. pool de prompts (data/grpo_prompts_t6.jsonl, já feito por scripts/make_rl_pool_t6.py) e adaptador LoRA em branco sobre o T6-Lumi (a fusão devolve o T6-Lumi bit a bit)
#   2. seleção dos prompts que dão sinal: N_SEL prompts do pool (80% múltipla escolha) x 8 respostas do T6-Lumi (T 0,6, top-p 0,95, 4.096 tokens); ficam os com 1 a 7 acertos
#      de 8 (scripts/select_hard_prompts.py). Estimativa: ~8 a 16 min a cada 256 prompts (rodadas anteriores), então ~40 a 80 min para 1.280. SELECIONAR=0 pula (usa o pool
#      inteiro; o RL descarta sozinho os grupos sem sinal, mas gasta geração neles)
#   3. RL (scripts/run_grpo.sh, base = T6-Lumi por GRPO_BASE): 4 iterações de 256 prompts x 8 respostas, GSPO (--ratio seq), lr 2e-5, T 0,6, top-p 0,95, 4.096 tokens, 80% de
#      múltipla escolha, como o `gspo_gold4k`, mas com --max-noclose 0,5 (o 0,05 pararia na iteração 0: nos prompts difíceis o T6-Lumi trunca muito). ~1,6 h.
#   4. validação do modelo final (feita pelo run_grpo.sh, runs/eval/rl_t6) e portão da H75 (scripts/rl_t6_gate.py)
#   5. só se o portão passar: suíte de teste e provas novas, com thinking, em runs/eval/rl_t6_teste (~3 a 4 h); tabelas em runs/jobs/rl_t6_comparacao.md
# Nenhum resultado troca o candidato: a regra de leitura está na H75. Estado: runs/jobs/rl_t6.status. Cancelar: tmux kill-session -t rl_t6 (e conferir os processos do vLLM).
# Uso: scripts/detach.sh rl_t6 scripts/run_rl_t6_chain.sh   (NÃO editar com ele rodando). DRY=1 mostra os comandos sem rodar nem esperar.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

status=runs/jobs/rl_t6.status
DRY="${DRY:-0}"
N_SEL="${N_SEL:-1280}"
SELECIONAR="${SELECIONAR:-1}"
NAME=rl_t6
T6=runs/merged/thor_pt_antidoom_of
POOL=data/grpo_prompts_t6.jsonl
HARD=data/grpo_prompts_t6_hard.jsonl
SEL=runs/rl_t6/select/rollouts.jsonl
VL=(env VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python)
SUITE=gsm8k_test,mmlu_prox_test,enem,bluex,oab
NOVAS=oab_novas,enem_2024,mmlu_pt_exatas,mmlu_pt_enade,mmlu_pt_concursos
say() { if [ "$DRY" = 1 ]; then echo "$(date +%d/%m\ %H:%M) $*"; else echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "[DRY] $*"; else "$@"; fi; }
wait_gpu() {
  [ "$DRY" = 1 ] && return 0
  local avisou=0
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
    [ "$avisou" = 0 ] && { say "GPU ocupada (outro processo), aguardando"; avisou=1; }
    sleep 60
  done
}

say "INICIO da cadeia do RL a partir do T6-Lumi (N_SEL=$N_SEL SELECIONAR=$SELECIONAR DRY=$DRY)"
if [ "$DRY" != 1 ]; then
  while tmux has-session -t noite18 2> /dev/null; do sleep 60; done
  say "a fila noite18 terminou"
fi
wait_gpu

# 1. pool e adaptador em branco
[ -s "$POOL" ] || { say "refazendo o pool"; run env PYTHONPATH=src python scripts/make_rl_pool_t6.py; }
[ -s "$POOL" ] || [ "$DRY" = 1 ] || { say "FALHA: sem $POOL"; exit 1; }
if [ ! -f runs/rl_t6/init_adapter/adapter_config.json ]; then
  say "criando o adaptador em branco sobre o T6-Lumi"
  run env PYTHONPATH=src python scripts/make_blank_adapter.py --base "$T6" --out runs/rl_t6/init_adapter || { say "FALHA: adaptador em branco"; exit 1; }
fi

# 2. seleção dos prompts que dão sinal
PROMPTS="$POOL"
if [ "$SELECIONAR" = 1 ]; then
  if [ ! -s "$HARD" ]; then
    if [ ! -s "$SEL" ]; then
      wait_gpu
      say "INICIO seleção: $N_SEL prompts x 8 respostas do T6-Lumi"
      run "${VL[@]}" scripts/grpo_rollout.py --model "$T6" --prompts "$POOL" --iter 0 --n-prompts "$N_SEL" --group 8 --frac-mcq 0.8 --temperature 0.6 --top-p 0.95 \
        --max-new 4096 --out "$SEL" || { say "FALHA: geração da seleção"; exit 1; }
    fi
    run env PYTHONPATH=src python scripts/select_hard_prompts.py --rollouts "$SEL" --pool "$POOL" --out "$HARD" || { say "FALHA: seleção"; exit 1; }
    say "FIM seleção: $(python3 -c "import json; r=json.load(open('${HARD%.jsonl}.report.json')); print(r['selecionados'], 'prompts com sinal de', r['prompts_gerados'], r['por_tipo'])" 2> /dev/null || echo "(ver ${HARD%.jsonl}.report.json)")"
  fi
  if [ "$DRY" != 1 ] && [ "$(wc -l < "$HARD")" -lt 300 ]; then
    say "AVISO: menos de 300 prompts com sinal; usando o pool inteiro"
  else
    PROMPTS="$HARD"
  fi
fi

# 3. RL
wait_gpu
say "INICIO RL $NAME (prompts $PROMPTS, base $T6)"
run env GRPO_BASE="$T6" scripts/run_grpo.sh "$NAME" --init runs/rl_t6/init_adapter --iters 4 --eval-every 2 --n-prompts 256 --group 8 --lr 2e-5 \
  --frac-mcq 0.8 --max-new 4096 --temperature 0.6 --top-p 0.95 --max-noclose 0.5 --ratio seq --prompts "$PROMPTS"
say "FIM RL $NAME (código $?)"

# 4. portão da H75
if [ "$DRY" = 1 ]; then echo "[DRY] python scripts/rl_t6_gate.py --name $NAME"; PASSA=1
elif python scripts/rl_t6_gate.py --name "$NAME" | tee -a "$status"; then PASSA=1; else PASSA=0; fi
if [ "$PASSA" != 1 ]; then
  say "o portão da H75 não passou: o T6-Lumi continua o candidato; a suíte de teste não roda (validação em runs/eval/$NAME)"
  say "FIM da cadeia"
  exit 0
fi

# 5. suíte de teste e provas novas do modelo final, com thinking
wait_gpu
say "INICIO suíte de teste e provas novas do $NAME (thinking)"
run "${VL[@]}" scripts/eval_vllm.py --name "${NAME}_teste" --model "runs/grpo/$NAME/merged" --tasks "$SUITE,$NOVAS" --modes thinking
R=runs/jobs/rl_t6_comparacao.md
if [ "$DRY" != 1 ]; then
  : > "$R"
  echo "# RL a partir do T6-Lumi (H75), escrito em $(date +%d/%m\ %H:%M); 4.096 tokens, gulosa, 4090" >> "$R"
  COLS=(--col "Lumi com thinking=base:thinking" --col "Lumi sem thinking=base:nothinking" --col "T6-Lumi=thor_pt_antidoom_of:thinking" --col "RL do T6-Lumi=${NAME}_teste:thinking")
  for ref in "T6-Lumi" "Lumi com thinking" "Lumi sem thinking"; do
    echo "## Suíte de teste, referência: $ref" >> "$R"; python scripts/report_compare.py --ref "$ref" "${COLS[@]}" --tasks "$SUITE" --mean >> "$R" 2>&1; echo >> "$R"
    echo "## Provas novas, referência: $ref" >> "$R"; python scripts/report_compare.py --ref "$ref" "${COLS[@]}" --tasks "$NOVAS" --mean >> "$R" 2>&1; echo >> "$R"
  done
  PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true
fi
say "FIM da cadeia (tabelas em $R); nenhum candidato foi trocado"
