#!/usr/bin/env bash
# Trabalho com o pacote 4 da Thor (traços de múltipla escolha do Qwen3-30B-A3B-Thinking em português). Chamado pela fila
# quando o pacote chega; também roda à mão. Cada passo é retomável e pulado se já pronto.
# Escolhas padrão (a revisar com o resultado do pacote; mudar só nos pontos marcados, antes de lançar):
#   - Mistura do SFT: os dados do `adalberto` como estão (BASE_SFT = data/train_adalberto.parquet: 5.842 traços de matemática do Qwen de até 1.024 tokens,
#     3.000 linhas sem raciocínio do `adalberto` e 5.544 do GigaVerbo) + todos os traços de múltipla escolha do pacote 4 que passam no filtro de
#     contaminação, de até 2.048 tokens (a mediana dos traços é 1.100: com 1.024 sobrariam 43%). Assim o único acréscimo ao `adalberto` é a
#     múltipla escolha, e a comparação com ele isola esse efeito. Só professor Qwen (fecha o `</think>`); os dados do Nemotron ficam de fora porque
#     os três SFTs com eles (`thor_pt`, `thor_pt_short`, `thor_pt_e1`) não passaram do `adalberto`.
#   - SFT: LoRA r=16, alpha 32, lr 1e-4, 1 época (a receita do `adalberto`), avaliação na validação (com até 3 tentativas: o vLLM já deu segfault
#     intermitente ao carregar o tokenizador).
#   - RL `gspo_<NOME>`: receita do `gspo1` (questões antigas, 2.048, T 0,6, top-p 0,95, --ratio seq, --max-noclose 0.05), 4 ciclos, a partir do SFT.
#   - Para se o pacote vier com menos de MIN_KEPT = 2.000 exemplos mantidos (o filtro da Thor já deixa só traços em português) ou depois do filtro de contaminação.
# Variáveis de ambiente (padrão = pacote final):
#   NOME  nome do SFT em runs/train, runs/merged e runs/eval, e da mistura data/<NOME>.parquet (padrão qwen_mix; o teste com a entrega parcial usa qwen_mix_p)
#   PKG   arquivo do pacote (padrão data/thor4/thor4_out.tar.gz)   EX  pasta de extração (data/thor4/e1)   PREP  pasta dos dados preparados (data/thor4/prep)
#   MIN_KEPT  mínimo de exemplos (2000)   BASE_SFT  parquet de base (data/train_adalberto.parquet)
#   SO_MISTURA=1  para depois de montar a mistura (confere os passos 1 a 3 sem usar a GPU)   SEM_RL=1  para depois da avaliação do SFT
# Estado: runs/jobs/<NOME>.status.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
NOME=${NOME:-qwen_mix}
PKG=${PKG:-data/thor4/thor4_out.tar.gz}
EX=${EX:-data/thor4/e1}
PREP=${PREP:-data/thor4/prep}
MIX=data/$NOME.parquet
BASE_SFT=${BASE_SFT:-data/train_adalberto.parquet}
MIN_KEPT=${MIN_KEPT:-2000}
status=runs/jobs/$NOME.status
say() { echo "$(date +%d/%m\ %H:%M) $*" | tee -a "$status"; }

wait_gpu() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -gt 1024 ]; do
    say "GPU ocupada, aguardando"; sleep 120
  done
}
avaliacao_completa() {
  python3 - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(f"runs/eval/{sys.argv[1]}/summary.json"))
    assert all(f"{t}|{m}" in d for t in ("gsm8k_val", "mcq_val") for m in ("thinking", "nothinking"))
except Exception:
    sys.exit(1)
PY
}

[ -f "$PKG" ] || { say "FALHA: $PKG não existe"; exit 1; }
say "pacote 4 ($NOME): $PKG, sha256 $(sha256sum "$PKG" | cut -c1-64)"

# 1. extrair em pasta nova (conferindo os caminhos do arquivo)
if [ ! -s "$EX/out/kept.jsonl" ]; then
  if tar -tzf "$PKG" | grep -E '(^/|\.\./)' > /dev/null; then say "FALHA: caminhos suspeitos no pacote"; exit 1; fi
  mkdir -p "$EX" && tar -xzf "$PKG" --no-same-owner -C "$EX" || { say "FALHA ao extrair"; exit 1; }
fi
[ -s "$EX/out/kept.jsonl" ] || { say "FALHA: $EX/out/kept.jsonl não existe ou está vazio"; exit 1; }

# 2. conferir o relatório da Thor
python3 - "$EX/out/report.json" "$MIN_KEPT" <<'PY' || { say "FALHA: relatório do pacote 4 abaixo do mínimo"; exit 1; }
import json, sys
r = json.load(open(sys.argv[1])); minimo = int(sys.argv[2])
kept = r["funil"]["mantidos"]
lang = r.get("idioma_do_traço_entre_os_que_fecham", {})  # só para registrar: o filtro da Thor já deixa só os traços em português
print(f"mantidos {kept}, rendimento {r.get('rendimento')}, idioma entre os que fecham {lang}")
sys.exit(0 if kept >= minimo else 1)
PY

# 3. preparar, filtrar contaminação e montar a mistura
if [ ! -s "$MIX" ]; then
  PYTHONPATH=src python scripts/prepare_thor4.py --kept "$EX/out/kept.jsonl" --out-dir "$PREP" || { say "FALHA em prepare_thor4.py"; exit 1; }
  PYTHONPATH=src python scripts/filter_translated.py "$PREP" || { say "FALHA no filtro de contaminação"; exit 1; }
  n_ok=$(wc -l < "$PREP/think_clean.jsonl"); say "pacote 4: $n_ok exemplos depois do filtro de contaminação"
  [ "$n_ok" -ge "$MIN_KEPT" ] || { say "FALHA: menos de $MIN_KEPT exemplos depois do filtro"; exit 1; }
  [ -s "$BASE_SFT" ] || { say "FALHA: $BASE_SFT não existe"; exit 1; }
  PYTHONPATH=src python scripts/build_teacher_dataset.py --think "$PREP/think_clean.jsonl" --disjoint-inputs --no-base \
    --out "$PREP/mcq_only.parquet" --max-think 40000 --max-len 2048 --per-problem 1 || { say "FALHA ao montar $PREP/mcq_only.parquet"; exit 1; }
  PYTHONPATH=src python scripts/concat_parquets.py --inputs "$BASE_SFT" "$PREP/mcq_only.parquet" --out "$MIX" || { say "FALHA ao juntar em $MIX"; exit 1; }
fi
say "mistura pronta: $MIX (sha256 $(sha256sum "$MIX" | cut -c1-16)…)"
if [ "${SO_MISTURA:-0}" = 1 ]; then say "SO_MISTURA=1: parando antes do SFT"; exit 0; fi

# 4. SFT e avaliação
if ! avaliacao_completa "$NOME"; then
  wait_gpu
  say "INICIO SFT $NOME"
  if [ ! -f "runs/train/$NOME/adapter/adapter_config.json" ]; then
    scripts/run_experiment.sh "$NOME" --arm lora --data "$MIX" --epochs 1 --lr 1e-4
  fi
  for t in 1 2 3; do
    avaliacao_completa "$NOME" && break
    [ -f "runs/merged/$NOME/merge_report.json" ] || PYTHONPATH=src python scripts/merge_adapter.py --adapter "runs/train/$NOME/adapter" --out "runs/merged/$NOME"
    wait_gpu
    say "avaliação do $NOME, tentativa $t"
    VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/eval_vllm.py --name "$NOME" --model "runs/merged/$NOME" --tasks validation
  done
  say "FIM SFT $NOME"
fi
avaliacao_completa "$NOME" || { say "FALHA: sem avaliação do $NOME; o RL não roda"; exit 1; }
if [ "${SEM_RL:-0}" = 1 ]; then say "SEM_RL=1: parando depois da avaliação do SFT"; exit 0; fi

# 5. RL a partir do SFT
wait_gpu
say "INICIO RL gspo_$NOME (receita do gspo1, a partir do $NOME, 4 ciclos)"
scripts/run_grpo.sh "gspo_$NOME" --init "runs/train/$NOME/adapter" --iters 4 --eval-every 4 --n-prompts 256 --group 8 --lr 2e-5 \
  --max-new 2048 --temperature 0.6 --top-p 0.95 --max-noclose 0.05 --ratio seq
say "FIM RL gspo_$NOME (código $?)"
PYTHONPATH=src python scripts/gen_estudos_resultados.py > /dev/null 2>&1 || true
