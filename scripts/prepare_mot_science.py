"""Prepara as questões de ciência (múltipla escolha) do `open-r1/Mixture-of-Thoughts` para tradução.

Lê os 4 arquivos `science/`, agrupa por questão, mantém só traços que fecham `</think>`, terminam em `\\boxed{letra}` igual à
letra majoritária (>= 75%, ao menos 3 traços) e têm até --max-tokens tokens (contados pelo dataset). Grava
data/mot_science_en.jsonl. Uso: PYTHONPATH=src python scripts/prepare_mot_science.py
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from lumi_ft.data import mot_science as M

REPO = "open-r1/Mixture-of-Thoughts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--per-question", type=int, default=2)
    ap.add_argument("--out", default="data/mot_science_en.jsonl")
    a = ap.parse_args()
    rows = []
    for i in range(4):
        f = hf_hub_download(REPO, f"science/train-0000{i}-of-00004.parquet", repo_type="dataset")
        rows += pq.read_table(f).to_pylist()
    out, st = M.select(rows, max_tokens=a.max_tokens, per_question=a.per_question)
    Path(a.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    st["linhas_lidas"] = len(rows)
    st["tokens_dos_registros"] = sum(r["num_tokens"] for r in out)
    print(json.dumps(st, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
