"""Monta data/grpo_prompts.jsonl: prompts de matemática (GSM8K-PT train, com gabarito) e de múltipla escolha (questões de
ciência traduzidas e descontaminadas). Cada linha: {id, kind: math|mcq, prompt, ref | (gold, n_options)}.

Matemática: só `gsm8k_train` de handoff/thor/data/prompts.jsonl (os do GigaVerbo têm referência fraca), sem os duplicados da
validação (assinatura de números + Jaccard). Uso: PYTHONPATH=src python scripts/make_grpo_prompts.py --mcq data/mot_questions/questions_clean.jsonl
"""
import argparse
import json
from pathlib import Path

from lumi_ft.data import teacher as T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--math", default="handoff/thor/data/prompts.jsonl")
    ap.add_argument("--mcq", default="")
    ap.add_argument("--out", default="data/grpo_prompts.jsonl")
    a = ap.parse_args()
    val = T.load_val_signatures()
    rows, dropped = [], 0
    for line in Path(a.math).read_text().splitlines():
        r = json.loads(line)
        if r["source"] != "gsm8k_train":
            continue
        if T.is_val_duplicate(r["question"], val):
            dropped += 1
            continue
        rows.append({"id": r["id"], "kind": "math", "prompt": r["prompt"], "ref": r["ref"]})
    n_math = len(rows)
    if a.mcq and Path(a.mcq).exists():
        for line in Path(a.mcq).read_text().splitlines():
            q = json.loads(line)
            rows.append({"id": f"mcq:{q['id']}", "kind": "mcq", "prompt": q["prompt"], "gold": q["gold"], "n_options": q["n_options"]})
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"math": n_math, "math_removidos_duplicados_da_validacao": dropped, "mcq": len(rows) - n_math, "out": a.out}))


if __name__ == "__main__":
    main()
