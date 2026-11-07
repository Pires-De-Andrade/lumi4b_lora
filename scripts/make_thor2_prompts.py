"""Prompts para o Nemotron 3 Nano 30B-A3B gerar traços (pacote handoff/thor2): matemática em português + múltipla escolha em inglês.

Matemática: os 5.612 problemas do GSM8K-PT `train` com gabarito e 4.000 do GigaVerbo (referência fraca) de handoff/thor/data/prompts.jsonl,
com o prompt em português (`#### N`). Múltipla escolha: 12.000 questões distintas de ciência do Mixture-of-Thoughts (inglês), fora das
que usamos como prompts do GRPO, com a letra majoritária dos traços do R1 como referência (indicador, não verdade) e o prompt em inglês
terminando em `Resposta: <letter>`. Ordem embaralhada (semente 0): qualquer prefixo é representativo.
Saída: handoff/thor2/data/prompts.jsonl {id, source, kind, question, options (mcq), prompt, ref}
Uso: PYTHONPATH=src python scripts/make_thor2_prompts.py
"""
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from translate_traces import pick  # noqa: E402

from lumi_ft.eval import mcq as M  # noqa: E402


def main() -> None:
    math_all = [json.loads(line) for line in Path("handoff/thor/data/prompts.jsonl").read_text().splitlines() if line.strip()]
    gsm = [r for r in math_all if r["source"] == "gsm8k_train"]
    gv = [r for r in math_all if r["source"] == "gv_math"][:4000]
    items = [{"id": r["id"], "source": r["source"], "kind": "math", "question": r["question"], "prompt": r["prompt"], "ref": r["ref"]} for r in gsm + gv]
    n_math = len(items)

    records = [json.loads(line) for line in Path("data/mot_science_en.jsonl").read_text().splitlines() if line.strip()]
    grpo_ids = {r["problem_id"] for r in pick(records, 4000, 1, 1)}  # as 4.000 questões do GRPO (translate_questions.py --seed 1)
    rest = [r for r in records if r["problem_id"] not in grpo_ids]
    for r in pick(rest, 12_000, 1, 0):
        items.append({"id": r["problem_id"], "source": "mot_science", "kind": "mcq", "question": r["question"], "options": r["options"],
                      "prompt": M.build_prompt_en({"question": r["question"], "options": r["options"]}), "ref": r["letter"]})
    random.Random(0).shuffle(items)
    out = Path("handoff/thor2/data")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "prompts.jsonl"
    path.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "prompts.sha256").write_text(f"{sha}  prompts.jsonl\n")
    print(json.dumps({"total": len(items), "math": n_math, "mcq": len(items) - n_math, "bytes": path.stat().st_size, "sha256": sha[:16]}))


if __name__ == "__main__":
    main()
