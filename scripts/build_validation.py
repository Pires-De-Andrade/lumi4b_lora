"""Conjuntos de validação de A-10: 1.000 itens do GSM8K-PT `train` sem 8-gramas em comum com o treino final,
e 385 questões de múltipla escolha (Global-MMLU PT `dev` 285, MMLU-ProX PT `validation` 70, INCLUDE PT `validation` 30).
Os de múltipla escolha já estão na lista de descontaminação do build do treino.

Saída: data/val_gsm8k.jsonl, data/val_mcq.jsonl e reports/val_manifest.json (hashes, D-11).
Uso: scripts/detach.sh build-val PYTHONPATH=src python scripts/build_validation.py
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset

from lumi_ft.data.decontaminate import NgramIndex
from lumi_ft.eval import mcq as M
from lumi_ft.modeling.load import load_tokenizer


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    t0 = time.time()
    tok = load_tokenizer()
    train = pq.read_table("data/train.parquet", columns=["id", "input_ids"]).to_pylist()
    idx = NgramIndex(8)
    for i in range(0, len(train), 2000):
        for text in tok.batch_decode([r["input_ids"] for r in train[i: i + 2000]], skip_special_tokens=False):
            idx.add(text)
    print(f"índice do treino: {len(idx):,} 8-gramas de {idx.n_texts:,} exemplos ({time.time() - t0:.0f} s)", flush=True)

    gsm = load_dataset("Polygl0t/gsm8k-pt", split="train")
    clean = [i for i, r in enumerate(gsm) if not idx.hits(r["question"] + "\n" + r["answer"])]
    print(f"GSM8K-PT train: {len(clean)} de {len(gsm)} itens sem 8-gramas em comum com o treino ({100 * len(clean) / len(gsm):.1f}%)", flush=True)
    assert len(clean) >= 1000, "poucos itens limpos para a validação"
    chosen = sorted(random.Random(0).sample(clean, 1000))
    Path("data").mkdir(exist_ok=True)
    p1 = Path("data/val_gsm8k.jsonl")
    p1.write_text("\n".join(json.dumps({"id": f"gsm8k_train:{i}", "question": gsm[i]["question"], "answer": gsm[i]["answer"]},
                                       ensure_ascii=False) for i in chosen) + "\n")

    mcq = M.LOADERS["global_mmlu_dev"]() + M.LOADERS["mmlu_prox_val"]() + M.LOADERS["include_val"]()
    p2 = Path("data/val_mcq.jsonl")
    p2.write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in mcq) + "\n")
    # nenhuma questão de validação de múltipla escolha pode ter 8-grama no treino
    leaked = [it["id"] for it in mcq if idx.hits(it["question"] + "\n" + "\n".join(it["options"]))]
    manifest = {"gsm8k_val": {"items": len(chosen), "clean_pool": len(clean), "train_items": len(gsm), "sha256": sha(p1)},
                "mcq_val": {"items": len(mcq), "sources": {"global_mmlu_dev": 285, "mmlu_prox_val": 70, "include_val": 30},
                            "overlap_with_training": leaked, "sha256": sha(p2)},
                "train_sha256": sha(Path("data/train.parquet")), "seconds": round(time.time() - t0)}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/val_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    print("EXIT_OK")


if __name__ == "__main__":
    sys.exit(main())
