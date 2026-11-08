"""Descontamina os registros traduzidos (8-grama contra as nossas provas e a validação) e escreve <dir>/{think,nothink,questions,think_en}_clean.jsonl.

Uso: PYTHONPATH=src python scripts/filter_translated.py data/mot_pilot
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts")
import build_dataset as bd  # noqa: E402

from lumi_ft.data import teacher as T  # noqa: E402
from lumi_ft.data.decontaminate import NgramIndex  # noqa: E402


def record_text(r: dict) -> str:
    """Texto do registro para o filtro de 8-gramas: enunciado + alternativas (múltipla escolha), ou o resto do prompt
    quando há `prompt`; a matemática vem sem `prompt` (usa o da avaliação), então entra com a resposta, como a validação foi indexada."""
    if "options" in r:
        tail = "\n".join(r["options"])
    elif "prompt" in r:
        tail = r["prompt"].split("\n\n", 2)[-1]
    else:
        tail = r["answer"]
    return r["question"] + "\n" + tail


def drop_reason(r: dict, idx, val, val_sig) -> str | None:
    """Por que o registro sai (None = fica): 8-grama contra as provas ou a validação; ou, na matemática, duplicata da validação
    em outra tradução (mesma assinatura de números e Jaccard >= 0,3 do enunciado, como em prepare_thor.py)."""
    text = record_text(r)
    if idx.hits(text) or val.hits(text):
        return "removido_8grama"
    if "options" not in r and T.is_val_duplicate(r["question"], val_sig):
        return "removido_duplicata_validacao"
    return None


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
    val_sig = T.load_val_signatures()
    st = Counter()
    for mode in ("think", "nothink", "questions", "think_en"):
        f = d / f"{mode}.jsonl"
        recs = [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []
        keep = []
        for r in recs:
            why = drop_reason(r, idx, val, val_sig)
            if why:
                st[f"{mode}:{why}"] += 1
            else:
                keep.append(r)
        (d / f"{mode}_clean.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
        st[f"{mode}:entrada"], st[f"{mode}:saida"] = len(recs), len(keep)
    print(json.dumps(dict(st), indent=1))


if __name__ == "__main__":
    main()
