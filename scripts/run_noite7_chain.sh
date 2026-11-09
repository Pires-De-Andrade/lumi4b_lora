#!/usr/bin/env bash
# Fila de 09/11 (versão 7). Substitui o run_noite6_chain.sh a partir do fim do RL `gspo_qwen_mix`. Cada etapa é retomável e pulada se já pronta.
# A 4090 é compartilhada: cada etapa espera a GPU ficar livre (menos de 1 GiB em uso) antes de começar.
# Ordem:
#   0. Espera o RL `gspo_qwen_mix` (em curso na fila noite6, congelada com SIGSTOP) terminar e encerra a noite6.
#   1. Dados dos dois SFTs de teste (scripts/prepare_sft_curtos.sh, só CPU; pulado se já pronto).
#   2. Teste 2, SFT `qwen_mix_short` (adalberto + traços de MCQ de até 1.024 tokens): H53, traços curtos reduzem os laços. Leitura prevista:
#      apoia H53 se fechar >= 87% e a múltipla escolha com thinking for >= 62%; compara com o `adalberto` (62,3%; fecha 86,8%) e o `qwen_mix` (62,1%; 83,9%).
#   3. Teste 1, SFT `selfdist_min` (3.000 traços do próprio Lumi em português, ~94 passos): H54, o custo de acerto vem da quantidade de SFT, não do
#      idioma. Leitura prevista: acerto entre os que fecham >= 78% (Lumi 80,7%) indica custo de quantidade; <= 75% indica custo do idioma; traço em
#      português >= 90% é condição para valer como candidato.
#   4. Suíte de teste do Lumi até o fim, como preenchimento (a ÚLTIMA de todas; retomável por (prova, modo); parar: touch runs/jobs/suite.stop). Os RLs
#      a partir de cada SFT de teste ficam fora da fila: dependem do resultado e entram interrompendo a suíte.
# Fora desta fila, adiados: RL `gspo1_eps3x`, `gspo1_same`, `gspo1_4k` (rodadas únicas; o ruído de 3 a 4 pontos entre sementes só deixa ler efeitos grandes).
# Uso: scripts/detach.sh noite7 scripts/run_noite7_chain.sh. Cancelar: tmux kill-session -t noite7 (e conferir os processos do vLLM).
# Estado: runs/jobs/noite7.status. NÃO editar com ele rodando.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

status=runs/jobs/noite7.status
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
sft_e_avaliar() {  # $1 = nome, $2 = parquet; receita do adalberto (LoRA r=16, lr 1e-4, 1 época); até 3 tentativas de avaliação (segfault intermitente do vLLM)
  if avaliacao_completa "$1"; then say "SFT $1 já treinado e avaliado: etapa pulada"; return; fi
  wait_gpu
  say "INICIO SFT $1 ($2)"
  if [ ! -f "runs/train/$1/adapter/adapter_config.json" ]; then
    scripts/run_experiment.sh "$1" --arm lora --data "$2" --epochs 1 --lr 1e-4
  fi
  for t in 1 2 3; do
    avaliacao_completa "$1" && break
    [ -f "runs/merged/$1/merge_report.json" ] || PYTHONPATH=src python scripts/merge_adapter.py --adapter "runs/train/$1/adapter" --out "runs/merged/$1"
    wait_gpu
    say "avaliação do $1, tentativa $t"
    VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src .venv-vllm/bin/python scripts/eval_vllm.py --name "$1" --model "runs/merged/$1" --tasks validation
  done
  say "FIM SFT $1"
}

# 0. espera o RL gspo_qwen_mix da noite6 (congelada) e encerra a noite6
while pgrep -f "bash scripts/run_qwen_mix.sh" > /dev/null; do sleep 20; done
say "RL gspo_qwen_mix terminou: encerrando a fila noite6 (congelada)"
pkill -KILL -f "scripts/run_noite6_chain.sh" 2>/dev/null; sleep 2
tmux kill-session -t =noite6 2>/dev/null
atualizar_tabela

# 1. dados (CPU)
say "INICIO dados dos SFTs de teste"
scripts/prepare_sft_curtos.sh
say "FIM dados dos SFTs de teste (código $?)"

# 2 e 3. os dois SFTs de teste, o curto primeiro
sft_e_avaliar qwen_mix_short data/qwen_mix_short.parquet
atualizar_tabela
sft_e_avaliar selfdist_min data/selfdist_min.parquet
atualizar_tabela

# 4. suíte de teste do Lumi, até o fim
wait_gpu
say "INICIO suíte de teste do Lumi (o que falta; preenchimento, interrompível)"
scripts/run_suite_models.sh "base|-"
say "FIM da fila (versão 7)"
