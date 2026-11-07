"""Converte o kept.jsonl devolvido pela Thor em registros de treino (com e sem raciocínio), sem duplicatas da validação.

Entrada: data/thor/kept.jsonl ({id, source, question, thought, answer, ref}). O pacote da Thor foi gerado antes de
descobrirmos que ~1/3 da validação do GSM8K estava no treino em outra tradução; este passo tira esses problemas.
Saída: data/thor_think.jsonl e data/thor_nothink.jsonl (para build_teacher_dataset.py).
Uso: PYTHONPATH=src python scripts/prepare_thor.py --kept data/thor/kept.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from lumi_ft.data import teacher as T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", default="data/thor/kept.jsonl")
    ap.add_argument("--out-prefix", default="data/thor")
    a = ap.parse_args()
    val = T.load_val_signatures()
    st, think, nothink = Counter(), [], []
    for line in Path(a.kept).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        st["lidos"] += 1
        if T.is_val_duplicate(r["question"], val):
            st["duplicata_da_validacao"] += 1
            continue
        st["usados"] += 1
        base = {"problem_id": f"thor:{r['id']}", "question": r["question"], "source": "thor"}
        think.append({**base, "id": f"thor:{r['id']}", "thought": r["thought"], "answer": r["answer"]})
        nothink.append({**base, "id": f"thor_nt:{r['id']}", "answer": r["answer"]})
    Path(f"{a.out_prefix}_think.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in think) + "\n")
    Path(f"{a.out_prefix}_nothink.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in nothink) + "\n")
    print(json.dumps(dict(st)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
