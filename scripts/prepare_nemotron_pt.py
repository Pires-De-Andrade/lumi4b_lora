"""Separa as questões de múltipla escolha de nvidia/Nemotron-SFT-Multilingual-v2 (ciência, português do Brasil) para o SFT.

Baixa (ou usa o cache de) `ultra-v3_stem_pt_translated_postedit_final.jsonl`, fica com as de múltipla escolha com gabarito válido
e raciocínio curto e grava data/nemotron_pt_mcq.jsonl (registros `ready_pt`: só o raciocínio, em inglês, falta traduzir).
Uso: PYTHONPATH=src python scripts/prepare_nemotron_pt.py [--max-words 1000]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from lumi_ft.data import nemotron_pt as N

REPO, FILE = "nvidia/Nemotron-SFT-Multilingual-v2", "ultra-v3_stem_pt_translated_postedit_final.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-words", type=int, default=1000)
    ap.add_argument("--n", type=int, default=0, help="0 = todas; senão, ao todo N: todas as GPQA (variadas, 10 alternativas) e o resto de RQA (química), sorteado")
    ap.add_argument("--out", default="data/nemotron_pt_mcq.jsonl")
    a = ap.parse_args()
    path = hf_hub_download(REPO, FILE, repo_type="dataset")
    rows = (json.loads(line) for line in open(path))
    out, st = N.select(rows, max_words=a.max_words)
    if a.n and a.n < len(out):
        import random

        gpqa = [r for r in out if r["category"] != "RQA synthetics"]
        rqa = [r for r in out if r["category"] == "RQA synthetics"]
        random.Random(0).shuffle(rqa)
        out = gpqa + rqa[: max(0, a.n - len(gpqa))]
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    st["por_categoria"] = dict(Counter(r["category"] for r in out))
    st["opcoes"] = dict(Counter(len(r["options"]) for r in out))
    print(json.dumps(st, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
