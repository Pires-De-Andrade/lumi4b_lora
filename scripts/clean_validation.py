"""Subconjunto limpo da validação do GSM8K: tira os itens que são duplicatas (outra tradução) de problemas de treino.

Critério: mesma assinatura de números (>= 2 números) E sobreposição de palavras (Jaccard) >= 0.3 com algum problema das
fontes de treino (GigaVerbo math_reasoning do treino e os problemas do adalberto). Só o 8-grama não pegava traduções
diferentes: ~1/3 da validação estava no treino. O TESTE do GSM8K-PT não tem esse problema (nenhum item com Jaccard >= 0.3).
Saída: data/val_gsm8k_clean_ids.json e reports/val_clean_report.json.  Uso: PYTHONPATH=src python scripts/clean_validation.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from lumi_ft.data import teacher as T
from lumi_ft.modeling.load import load_tokenizer

J_MIN = 0.3


def words(t: str) -> set[str]:
    return set(re.findall(r"[a-zà-ÿ]{4,}", t.lower()))


def main() -> None:
    tok = load_tokenizer()
    by_sig: dict[tuple, list[set]] = defaultdict(list)
    n_src = {}

    def add(q: str, src: str) -> None:
        s = T.number_signature(q)
        if len(s) >= 2:
            by_sig[s].append(words(q))
        n_src[src] = n_src.get(src, 0) + 1

    for r in pq.read_table("data/train.parquet", columns=["mode", "task_type", "n_prompt", "input_ids"]).to_pylist():
        if r["mode"] == "think" and r["task_type"] == "math_reasoning":
            p = tok.decode(r["input_ids"][: r["n_prompt"]], skip_special_tokens=False)
            add(p.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>")[0], "gv_math")
    seen = set()
    p = Path("data/adalberto_think.jsonl")
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["problem_id"] not in seen:
                seen.add(r["problem_id"])
                add(r["question"], "adalberto")
    val = [json.loads(line) for line in Path("data/val_gsm8k.jsonl").read_text().splitlines()]
    clean, dup = [], []
    for v in val:
        s = T.number_signature(v["question"])
        w = words(v["question"])
        best = max((len(w & c) / max(1, len(w | c)) for c in by_sig.get(s, [])), default=0.0) if len(s) >= 2 else 0.0
        (dup if best >= J_MIN else clean).append(v["id"])
    Path("data/val_gsm8k_clean_ids.json").write_text(json.dumps(clean))
    rep = {"val_total": len(val), "clean": len(clean), "duplicates_removed": len(dup), "j_min": J_MIN, "sources_n": n_src}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/val_clean_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    sys.exit(main())
