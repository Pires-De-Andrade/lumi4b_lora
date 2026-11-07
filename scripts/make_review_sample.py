"""D-10: amostra para revisão humana de traços (100 de cada task_type de raciocínio) -> reports/review_d10.md.
Uso: PYTHONPATH=src python scripts/make_review_sample.py"""
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

from lumi_ft.modeling.load import load_tokenizer


def main() -> int:
    tok = load_tokenizer()
    rows = [r for r in pq.read_table("data/train.parquet").to_pylist() if r["mode"] == "think"]
    out = ["# Revisão humana de traços (D-10)\n",
           "Para cada exemplo: o traço é coerente? leva à resposta? é útil (não só repete o enunciado)? está em português? "
           "Anote problemas por número. 100 de `math_reasoning` e 100 de `reasoning`, sorteados com semente 0.\n"]
    for tt in ("math_reasoning", "reasoning"):
        pool = sorted((r for r in rows if r["task_type"] == tt), key=lambda r: r["id"])
        for k, r in enumerate(random.Random(0).sample(pool, 100), 1):
            ids = r["input_ids"]
            prompt = tok.decode(ids[: r["n_prompt"]], skip_special_tokens=False)
            resp = tok.decode(ids[r["n_prompt"]: r["loss_end"] - 1], skip_special_tokens=False)
            user = prompt.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>")[0]
            out += [f"\n---\n\n## {tt} #{k} ({r['id']}, {r['n_tokens']} tokens)\n", f"**Pergunta:** {user.strip()}\n", "**Resposta do dataset (traço + resposta):**\n", "```text", resp.strip(), "```"]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/review_d10.md").write_text("\n".join(out) + "\n")
    print("gravado reports/review_d10.md", len(out), "blocos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
