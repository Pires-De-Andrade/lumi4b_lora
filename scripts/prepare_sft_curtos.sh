#!/usr/bin/env bash
# Dados dos dois SFTs de teste de 09/11 (só CPU; idempotente: cada parquet é pulado se já existe).
#   Teste 2, `qwen_mix_short`: os dados do `adalberto` como estão (data/train_adalberto.parquet) + os traços de múltipla escolha do pacote 4 de até
#     1.024 tokens (o `qwen_mix` usa até 2.048). Hipótese H53: traços curtos reduzem os laços (o `qwen_mix` fecha 83,9% contra 86,8% do `adalberto`).
#   Teste 1, `selfdist_min`: 3.000 traços de matemática do próprio Lumi, em português com o começo forçado (os aceitos do `selfdist`, um por problema,
#     até 1.024 tokens), sem linhas sem raciocínio, ~94 passos (1/5 do `adalberto`). Hipótese H54: o custo de acerto vem da quantidade de SFT e não do idioma.
# Entradas: data/thor4/prep/think_clean.jsonl (script run_qwen_mix.sh), data/selfdist/think.jsonl, data/train_adalberto.parquet.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh

if [ ! -s data/qwen_mix_short.parquet ]; then
  [ -s data/thor4/prep/think_clean.jsonl ] || { echo "FALHA: falta data/thor4/prep/think_clean.jsonl (rode scripts/run_qwen_mix.sh antes)"; exit 1; }
  PYTHONPATH=src python scripts/build_teacher_dataset.py --think data/thor4/prep/think_clean.jsonl --disjoint-inputs --no-base \
    --out data/thor4/prep/mcq_short.parquet --max-think 40000 --max-len 1024 --per-problem 1 || exit 1
  PYTHONPATH=src python scripts/concat_parquets.py --inputs data/train_adalberto.parquet data/thor4/prep/mcq_short.parquet --out data/qwen_mix_short.parquet || exit 1
fi
if [ ! -s data/selfdist_min.parquet ]; then
  [ -s data/selfdist/think.jsonl ] || { echo "FALHA: falta data/selfdist/think.jsonl"; exit 1; }
  PYTHONPATH=src python scripts/build_teacher_dataset.py --think data/selfdist/think.jsonl --disjoint-inputs --no-base \
    --out data/selfdist_min.parquet --max-think 3000 --max-len 1024 --per-problem 1 || exit 1
fi
echo "dados prontos: data/qwen_mix_short.parquet, data/selfdist_min.parquet"
