"""Prompts do pacote 3 da Thor: questões de múltipla escolha com gabarito, em inglês, de domínios além da ciência do pacote 2.

7.000 do WebInstruct-verified (negócios, economia, matemática, física, finanças, química, biologia, psicologia, história,
filosofia, direito...; nenhuma área com mais de 20%) e 5.000 do MedMCQA (21 áreas médicas, nenhuma com mais de 8%). Fora:
enunciado repetido e 8-grama em comum com o MMLU-ProX (teste e validação) ou o MMLU (teste) em inglês, que são a origem de
provas nossas. A descontaminação contra as provas em português é feita depois da tradução (scripts/filter_translated.py).
Prompt em inglês terminando em `Resposta: <letter>` (mcq.build_prompt_en), como no pacote 2. Ordem embaralhada (semente 0).
Saída: handoff/thor3/data/prompts.jsonl {id, source, kind, category, question, options, prompt, ref} e prompts.sha256
Uso: PYTHONPATH=src python scripts/make_thor3_prompts.py
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from lumi_ft.data import thor3 as T
from lumi_ft.data.decontaminate import NgramIndex
from lumi_ft.eval import mcq as M

WEBINSTRUCT = ("TIGER-Lab/WebInstruct-verified", "data/train-00000-of-00001.parquet", "3e8a350b3a935d68fe70bdf379692500abc9ff51")
MEDMCQA = ("openlifescienceai/medmcqa", "data/train-00000-of-00001.parquet", "91c6572c454088bf71b679ad90aa8dffcd0d5868")
MMLU_TEST = ("cais/mmlu", "all/test-00000-of-00001.parquet", None)


def rows(spec) -> list[dict]:
    import pandas as pd

    repo, path, rev = spec
    return pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset", revision=rev)).to_dict("records")


def norm(q: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", q.lower()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-webinstruct", type=int, default=7000)
    ap.add_argument("--n-medmcqa", type=int, default=5000)
    ap.add_argument("--out", default="handoff/thor3/data")
    a = ap.parse_args()

    idx = NgramIndex(8)
    n_ref = 0
    for split in ("test", "validation"):
        for it in M.load_mmlu_prox(split, "en"):
            idx.add(it["question"] + "\n" + "\n".join(it["options"]))
            n_ref += 1
    for r in rows(MMLU_TEST):
        idx.add(r["question"] + "\n" + "\n".join(map(str, r["choices"])))
        n_ref += 1

    stats = Counter()
    pools = {}
    for name, spec, fn in (("webinstruct", WEBINSTRUCT, T.webinstruct_item), ("medmcqa", MEDMCQA, T.medmcqa_item)):
        raw = rows(spec)
        items = [x for x in map(fn, raw) if x]
        stats[f"{name}:linhas"], stats[f"{name}:separadas"] = len(raw), len(items)
        seen, keep = set(), []
        for it in sorted(items, key=lambda x: x["id"]):
            k = norm(it["question"])
            if k in seen:
                stats[f"{name}:repetidas"] += 1
                continue
            seen.add(k)
            if idx.hits(it["question"] + "\n" + "\n".join(it["options"])):
                stats[f"{name}:8grama_mmlu"] += 1
                continue
            keep.append(it)
        pools[name] = keep
        stats[f"{name}:candidatas"] = len(keep)

    chosen = T.balanced_sample(pools["webinstruct"], a.n_webinstruct, 0.20, seed=0) + T.balanced_sample(pools["medmcqa"], a.n_medmcqa, 0.08, seed=0)
    for it in chosen:
        it["kind"] = "mcq"
        it["prompt"] = M.build_prompt_en(it)
    random.Random(0).shuffle(chosen)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "prompts.jsonl"
    keys = ("id", "source", "kind", "category", "question", "options", "prompt", "ref")
    path.write_text("".join(json.dumps({k: it[k] for k in keys}, ensure_ascii=False) + "\n" for it in chosen))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "prompts.sha256").write_text(f"{sha}  prompts.jsonl\n")
    rep = {"referencia_8grama_itens": n_ref, **dict(stats), "escolhidas": len(chosen),
           "por_fonte": dict(Counter(it["source"] for it in chosen)),
           "por_categoria": {s: dict(Counter(it["category"] for it in chosen if it["source"] == s).most_common()) for s in ("webinstruct", "medmcqa")},
           "alternativas": dict(Counter(len(it["options"]) for it in chosen)), "gabarito": dict(Counter(it["ref"] for it in chosen)), "sha256": sha}
    (out / "prompts_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
