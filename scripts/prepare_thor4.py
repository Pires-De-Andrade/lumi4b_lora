"""Converte o `kept.jsonl` do pacote 4 da Thor (traços de múltipla escolha do Qwen em português) em registros de treino.

Entrada: {id, source, category, prompt, thought, answer, ref, lang, completion_tokens} (saída de handoff/thor4/filter_traces_pt_mcq.py; o
`prompt` já é o da avaliação e a `answer` termina em `Resposta: <letra>`).
Saída em --out-dir: think.jsonl, no esquema que `filter_translated.py` e `build_teacher_dataset.py` leem
({id, problem_id, question, answer, source, prompt, thought, letter}), e prepare_report.json.
Depois: scripts/filter_translated.py <out-dir>  (8-grama contra as provas e a validação)  ->  think_clean.jsonl
Uso: PYTHONPATH=src python scripts/prepare_thor4.py --kept data/thor4/e1/out/kept.jsonl --out-dir data/thor4/prep
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def convert(k: dict) -> dict:
    """Registro de treino de uma linha do kept.jsonl. `question` recebe o prompt inteiro (é o texto que o filtro de 8-gramas confere)."""
    rid = f"qwen_mcq:{k['id']}"
    return {"id": rid, "problem_id": rid, "question": k["prompt"], "answer": k["answer"], "source": f"qwen_{k['source']}",
            "prompt": k["prompt"], "thought": k["thought"], "letter": k["ref"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kept", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs, st = [], Counter()
    for line in Path(a.kept).read_text().splitlines():
        if not line.strip():
            continue
        k = json.loads(line)
        st["lidos"] += 1
        if not k.get("thought") or not k.get("answer") or not k["answer"].rstrip().splitlines()[-1].lstrip("*> `").startswith("Resposta:"):
            st["descartado_sem_resposta_final"] += 1
            continue
        recs.append(convert(k))
        st[f"fonte:{recs[-1]['source']}"] += 1
    (out / "think.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    st["usados"] = len(recs)
    (out / "prepare_report.json").write_text(json.dumps(dict(st), indent=1))
    print(json.dumps(dict(st), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
