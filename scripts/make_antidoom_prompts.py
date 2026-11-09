"""Prompts para minerar laços e medir a taxa de laço na fase anti-laço (data/antidoom_prompts.jsonl).

Só prompts de treino (descontaminados contra as provas e a validação): todas as de múltipla escolha de `data/grpo_prompts.jsonl` (letra majoritária do R1),
mais múltipla escolha do conjunto com gabarito oficial (`data/grpo_prompts_gold.jsonl`) e matemática do GSM8K `train`. O campo `split` separa o que
alimenta o treino ("mine") do que só mede a taxa de laço depois ("heldout", sem repetir prompt do outro).
Uso: python scripts/make_antidoom_prompts.py [--mcq-mine 8000 --math-mine 2000 --mcq-heldout 500 --math-heldout 250 --seed 0]
"""
import argparse
import json
import random
import sys
from pathlib import Path


def build(old: list[dict], gold: list[dict], *, mcq_mine: int = 8000, math_mine: int = 2000, mcq_heldout: int = 500, math_heldout: int = 250,
          seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    seen: set[str] = set()

    def fresh(rows: list[dict]) -> list[dict]:
        out = []
        for r in sorted(rows, key=lambda r: r["id"]):
            if r["prompt"] not in seen:
                seen.add(r["prompt"])
                out.append(r)
        return out

    old_mcq = fresh([r for r in old if r["kind"] == "mcq"])
    old_math = fresh([r for r in old if r["kind"] == "math"])
    gold_mcq = fresh([r for r in gold if r["kind"] == "mcq"])
    rng.shuffle(old_mcq)
    rng.shuffle(old_math)
    rng.shuffle(gold_mcq)
    pool = old_mcq + gold_mcq  # as do conjunto antigo primeiro, depois as do gabarito oficial
    mine_mcq, held_mcq = pool[:mcq_mine], pool[mcq_mine:mcq_mine + mcq_heldout]
    mine_math, held_math = old_math[:math_mine], old_math[math_mine:math_mine + math_heldout]
    out = []
    for split, rows in (("mine", mine_mcq + mine_math), ("heldout", held_mcq + held_math)):
        for r in rows:
            out.append({"id": r["id"], "kind": r["kind"], "prompt": r["prompt"], "split": split})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", default="data/grpo_prompts.jsonl")
    ap.add_argument("--gold", default="data/grpo_prompts_gold.jsonl")
    ap.add_argument("--out", default="data/antidoom_prompts.jsonl")
    ap.add_argument("--mcq-mine", type=int, default=8000)
    ap.add_argument("--math-mine", type=int, default=2000)
    ap.add_argument("--mcq-heldout", type=int, default=500)
    ap.add_argument("--math-heldout", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    load = lambda p: [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    rows = build(load(a.old), load(a.gold), mcq_mine=a.mcq_mine, math_mine=a.math_mine, mcq_heldout=a.mcq_heldout, math_heldout=a.math_heldout, seed=a.seed)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    from collections import Counter
    print(dict(Counter((r["split"], r["kind"]) for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
