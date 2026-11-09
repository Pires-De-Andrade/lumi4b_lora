#!/usr/bin/env bash
# Fila de 09/11 (versão 8). Substitui o run_noite7_chain.sh (que só esperava o RL terminar). Cada etapa é retomável e pulada se já pronta.
# A 4090 é compartilhada: cada etapa espera a GPU ficar livre (menos de 1 GiB em uso) antes de começar.
# Ordem:
#   0. Espera o RL `gspo_qwen_mix` (em curso na fila noite6, congelada com SIGSTOP) terminar e encerra a noite6.
#   1. Avaliação das sopas de pesos dos RLs da receita do `gspo1` (scripts/prepare_soups.sh, já montadas na CPU): `gspo1_soup2` (sementes 0 e 1) e
#      `gspo1_soup3` (mais o `gspo1_T06`). H55: a média reduz o ruído entre sementes. As faixas das rodadas: 64,4 a 67,8% (múltipla escolha).
#   2. Diagnóstico `base_ptprefill`: o Lumi sem adaptador, com o começo forçado em português (`--prefill`). H56: mede o custo de pensar em português
#      sem treino nenhum (o §20.11 do discovery só mediu o idioma da questão).
#   3. Teste 2, SFT `qwen_mix_short` (adalberto + traços de MCQ de até 1.024 tokens, H53).
#   4. Teste 1, SFT `selfdist_min` (3.000 traços do próprio Lumi, ~94 passos, H54) e teste 1b, `selfdist_300` (300 traços, ~10 passos).
#   5. Suíte de teste do Lumi até o fim, como preenchimento (a ÚLTIMA de todas; retomável por (prova, modo); parar: touch runs/jobs/suite.stop).
#      Os RLs a partir dos SFTs de teste ficam fora da fila: dependem do resultado e entram interrompendo a suíte.
# Adiados: RL `gspo1_eps3x`, `gspo1_same`, `gspo1_4k` (rodadas únicas; o ruído de 3 a 4 pontos entre sementes só deixa ler efeitos grandes).
# Uso: scripts/detach.sh noite8 scripts/run_noite8_chain.sh. Cancelar: tmux kill-session -t noite8 (e conferir os processos do vLLM).
# Estado: runs/jobs/noite8.status. NÃO editar com ele rodando.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

status=runs/jobs/noite8.status
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

# 0. espera o RL gspo_qwen_mix da noite6 (congelada) e encerra a noite6
while pgrep -f "bash scripts/run_qwen_mix.sh" > /dev/null; do sleep 20; done
say "RL gspo_qwen_mix terminou: encerrando a fila noite6 (congelada)"
pkill -KILL -f "scripts/run_noite6_chain.sh" 2>/dev/null; sleep 2
tmux kill-session -t =noite6 2>/dev/null
atualizar_tabela

# dados (CPU, pulados se prontos)
scripts/prepare_soups.sh
scripts/prepare_sft_curtos.sh

# 1. sopas de pesos
avaliar gspo1_soup2 runs/merged/gspo1_soup2
atualizar_tabela
avaliar gspo1_soup3 runs/merged/gspo1_soup3
atualizar_tabela

# 2. Lumi sem adaptador, com o começo forçado em português
avaliar base_ptprefill ceia-br/lumi-4b --prefill "Vou raciocinar em português."
atualizar_tabela

# 3 e 4. SFTs de teste, o curto primeiro
sft_e_avaliar qwen_mix_short data/qwen_mix_short.parquet
atualizar_tabela
sft_e_avaliar selfdist_min data/selfdist_min.parquet
atualizar_tabela
sft_e_avaliar selfdist_300 data/selfdist_300.parquet
atualizar_tabela

# 5. suíte de teste do Lumi, até o fim
wait_gpu
say "INICIO suíte de teste do Lumi (o que falta; preenchimento, interrompível)"
scripts/run_suite_models.sh "base|-"
say "FIM da fila (versão 8)"
