"""Lista de problemas de matemática para um professor maior gerar traços em português (pacote da Thor).

Fontes: (1) os problemas `math_reasoning` do GigaVerbo já filtrados no treino (referência = último número da resposta do
dataset, ou seja, concordância com outro modelo, não gabarito); (2) o GSM8K-PT `train` fora da validação (referência =
gabarito `#### N`). Dedupe por enunciado (fica o GSM8K, que tem gabarito), sem 8-gramas das provas de teste nem da
validação, só referências numéricas. Ordem embaralhada (semente 0), então qualquer prefixo é representativo.

Saída: handoff/thor/data/prompts.jsonl  {id, source, question, prompt, ref}
Uso: PYTHONPATH=src python scripts/make_thor_prompts.py
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset

sys.path.insert(0, "scripts")
import build_dataset as bd  # noqa: E402  (reaproveita a lista de provas da descontaminação)

from lumi_ft.data import gigaverbo as GV  # noqa: E402
from lumi_ft.data.decontaminate import NgramIndex  # noqa: E402
from lumi_ft.eval import gsm8k as K  # noqa: E402
from lumi_ft.eval import parse as P  # noqa: E402
from lumi_ft.modeling.load import load_tokenizer  # noqa: E402

NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")


def main() -> None:
    tok = load_tokenizer()
    test_idx, _ = bd.build_decontam_index(8)
    val_idx = NgramIndex(8)
    for line in Path("data/val_gsm8k.jsonl").read_text().splitlines():
        r = json.loads(line)
        val_idx.add(r["question"] + "\n" + r["answer"])
    for line in Path("data/val_mcq.jsonl").read_text().splitlines():
        r = json.loads(line)
        val_idx.add(r["question"] + "\n" + "\n".join(r["options"]))
    val_ids = {json.loads(line)["id"] for line in Path("data/val_gsm8k.jsonl").read_text().splitlines()}

    # (2) GSM8K-PT train, fora da validação, com gabarito
    gsm = load_dataset("Polygl0t/gsm8k-pt", split="train")
    items, seen, drops = [], set(), {"gsm_val": 0, "gsm_overlap": 0, "gsm_ref": 0}
    for i, r in enumerate(gsm):
        if f"gsm8k_train:{i}" in val_ids:
            drops["gsm_val"] += 1
            continue
        text = r["question"] + "\n" + r["answer"]
        if val_idx.hits(text) or test_idx.hits(text):
            drops["gsm_overlap"] += 1
            continue
        ref = P.gsm8k_target(r["answer"])
        if not NUMERIC.match(ref):
            drops["gsm_ref"] += 1
            continue
        seen.add(GV.norm_prompt(r["question"]))
        items.append({"id": f"gsm8k_train:{i}", "source": "gsm8k_train", "question": r["question"], "ref": ref})

    # (1) GigaVerbo math_reasoning que entrou no treino (já sem 8-gramas das provas de teste)
    n_gsm = len(items)
    rows = pq.read_table("data/train.parquet", columns=["id", "mode", "task_type", "n_prompt", "loss_end", "input_ids"]).to_pylist()
    gdrops = {"dup": 0, "val_overlap": 0, "ref": 0}
    for r in rows:
        if r["mode"] != "think" or r["task_type"] != "math_reasoning":
            continue
        ids = r["input_ids"]
        prompt = tok.decode(ids[: r["n_prompt"]], skip_special_tokens=False)
        user = prompt.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>")[0].strip()
        resp = tok.decode(ids[r["n_prompt"]: r["loss_end"] - 1], skip_special_tokens=False)
        answer = resp.split("</think>", 1)[-1]
        if GV.norm_prompt(user) in seen:
            gdrops["dup"] += 1
            continue
        if val_idx.hits(user):
            gdrops["val_overlap"] += 1
            continue
        ln = P.last_number(answer)
        ref = P.normalize_number(ln) if ln else ""
        if not NUMERIC.match(ref):
            gdrops["ref"] += 1
            continue
        seen.add(GV.norm_prompt(user))
        items.append({"id": r["id"], "source": "gv_math", "question": user, "ref": ref})

    for it in items:
        it["prompt"] = K.build_prompt(it["question"])
    random.Random(0).shuffle(items)
    out = Path("handoff/thor/data")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "prompts.jsonl"
    path.write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "prompts.sha256").write_text(f"{sha}  prompts.jsonl\n")
    print(json.dumps({"total": len(items), "gsm8k_train": n_gsm, "gv_math": len(items) - n_gsm, "drops_gsm": drops,
                      "drops_gv": gdrops, "sha256": sha[:16], "bytes": path.stat().st_size}, indent=1))


if __name__ == "__main__":
    main()
