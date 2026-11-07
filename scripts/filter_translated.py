"""Descontamina os registros traduzidos (8-grama contra as nossas provas e a validação) e escreve <dir>/{think,nothink,questions}_clean.jsonl.

Uso: PYTHONPATH=src python scripts/filter_translated.py data/mot_pilot
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts")
import build_dataset as bd  # noqa: E402

from lumi_ft.data.decontaminate import NgramIndex  # noqa: E402


def main() -> None:
    d = Path(sys.argv[1])
    idx, _ = bd.build_decontam_index(8)
    val = NgramIndex(8)
    for line in Path("data/val_mcq.jsonl").read_text().splitlines():
        m = json.loads(line)
        val.add(m["question"] + "\n" + "\n".join(m["options"]))
    for line in Path("data/val_gsm8k.jsonl").read_text().splitlines():
        v = json.loads(line)
        val.add(v["question"] + "\n" + v["answer"])
    st = Counter()
    for mode in ("think", "nothink", "questions"):
        f = d / f"{mode}.jsonl"
        recs = [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []
        keep = []
        for r in recs:
            text = r["question"] + "\n" + ("\n".join(r["options"]) if "options" in r else r["prompt"].split("\n\n", 2)[-1])
            if idx.hits(text) or val.hits(text):
                st[f"{mode}:removido_8grama"] += 1
            else:
                keep.append(r)
        (d / f"{mode}_clean.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
        st[f"{mode}:entrada"], st[f"{mode}:saida"] = len(recs), len(keep)
    print(json.dumps(dict(st), indent=1))


if __name__ == "__main__":
    main()
