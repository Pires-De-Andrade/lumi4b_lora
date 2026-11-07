"""Converte as traduções da Thor (handoff/thor3/translate_thor.py) nos registros do treino (build_teacher_dataset.py).

Escreve em --out:
- think.jsonl: exemplos com o raciocínio traduzido (os 70% de problemas com raciocínio, pelo hash do id);
- nothink.jsonl: os outros 30%, só com a resposta final (disjunto do think: não precisa de --split-sources);
- think_en.jsonl: o par de think.jsonl com o raciocínio original em inglês (mesmo prompt e mesma resposta em português),
  para o experimento "raciocínio em inglês" com os mesmos problemas;
- report.json.
Prompt de múltipla escolha = mcq.build_prompt (o da avaliação); resposta = texto final + `Resposta: X`. Matemática: sem `prompt`
(build_rows usa o da avaliação do GSM8K) e resposta terminando em `#### N`. Depois: scripts/filter_translated.py <out>.
Uso: PYTHONPATH=src python scripts/prepare_thor_translated.py data/thor3/out/translated.jsonl --out data/thor_tr
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from lumi_ft.data import gigaverbo as GV
from lumi_ft.eval import mcq


def convert(rec: dict) -> tuple[dict, dict | None]:
    """(registro sem raciocínio, par com raciocínio {pt, en} ou None)."""
    base = {"id": rec["id"], "problem_id": rec["id"], "question": rec["question_pt"], "answer": rec["answer_pt"], "source": f"thor_{rec['source']}"}
    if rec["kind"] == "mcq":
        base["options"] = rec["options_pt"]
        base["letter"] = rec["ref"]
        base["prompt"] = mcq.build_prompt({"question": rec["question_pt"], "options": rec["options_pt"]})
    if not rec.get("with_thought"):
        return base, None
    return base, {"pt": {**base, "thought": rec["thought_pt"]}, "en": {**base, "thought": rec["thought_en"]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("translated", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    think, think_en, nothink, st, seen = [], [], [], Counter(), set()
    for path in a.translated:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["id"] in seen:
                st["repetido"] += 1
                continue
            seen.add(rec["id"])
            st[f"{rec['kind']}:{rec['verdict'].split(':')[-1]}"] += 1
            if rec["verdict"] != "ok":
                continue
            base, pair = convert(rec)
            if pair:
                assert GV.bucket(rec["id"]) < 7000, rec["id"]  # a mesma divisão do translate_thor.py
                think.append(pair["pt"])
                think_en.append(pair["en"])
            elif GV.bucket(rec["id"]) >= 7000:
                nothink.append(base)
            else:
                st["sem_raciocinio_no_lado_think"] += 1  # só se a tradução tiver rodado com --think-frac diferente de 0,7
    for name, rows in (("think", think), ("think_en", think_en), ("nothink", nothink)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        st[f"saida:{name}"] = len(rows)
    st_src = Counter(r["source"] for r in think + nothink)
    (out / "report.json").write_text(json.dumps({**dict(st), "por_fonte": dict(st_src)}, ensure_ascii=False, indent=1))
    print(json.dumps({**dict(st), "por_fonte": dict(st_src)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
