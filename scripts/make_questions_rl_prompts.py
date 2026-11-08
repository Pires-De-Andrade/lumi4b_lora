"""Monta os prompts de RL de múltipla escolha com GABARITO OFICIAL a partir de out/questions_pt.jsonl (a tradução das 12.000 questões
do pacote 3, feita na Thor por handoff/thor3/translate_questions.py) e os junta à matemática de data/grpo_prompts.jsonl.

Cada questão vira {id: "mcq:<id>", kind: "mcq", prompt, gold, n_options, source, category}. O prompt é o MESMO da avaliação
(`mcq.build_prompt`: instrução + "Questão: ..." + alternativas A), B), ...). Ficam só as questões que passam em tudo:
  1. veredito "ok" da tradução (as recusadas já foram conferidas na Thor);
  2. esquema válido (2 a 10 alternativas, nenhuma vazia, gabarito é uma letra dentro do intervalo, id único) e sem marcador de tradução
     suspeita (`question_quality`: negação do inglês sem correspondente no português, enunciado não traduzido, alternativas repetidas);
  3. sem 8-grama em comum com as nossas provas e a validação (texto em português) nem com o MMLU-ProX em inglês (o original das questões);
  4. prompt de até --max-prompt-tokens tokens.
Lê com tolerância: uma linha quebrada (cópia parcial tirada com a tradução rodando) é contada e pulada.

Uso: PYTHONPATH=src python scripts/make_questions_rl_prompts.py --questions data/thor3/qpartial/out/questions_pt.jsonl --out data/grpo_prompts_gold.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from lumi_ft.data.question_quality import excluded_reason

LETTERS = "ABCDEFGHIJ"


def read_jsonl(path: str) -> tuple[list[dict], int]:
    rows, bad = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return rows, bad


def convert(rec: dict, seen: set, build_prompt, contaminated_pt, contaminated_en, n_tokens, max_prompt_tokens: int):
    """(linha de RL, None) se a questão fica; (None, motivo) se sai. `seen` guarda os ids já aceitos."""
    if rec.get("verdict") != "ok":
        return None, "recusada_na_traducao"
    opts = [str(o).strip() for o in (rec.get("options_pt") or [])]
    q = str(rec.get("question_pt") or "").strip()
    gold = str(rec.get("ref") or "").strip().upper()
    if not q or not 2 <= len(opts) <= len(LETTERS) or any(not o for o in opts) or len(gold) != 1 or LETTERS.find(gold) not in range(len(opts)):
        return None, "esquema_invalido"
    if rec["id"] in seen:
        return None, "id_repetido"
    bad = excluded_reason(rec)  # negação do inglês que sumiu, enunciado sem tradução, alternativas repetidas (o gabarito pode ter ficado errado)
    if bad:
        return None, f"qualidade_{bad}"
    if contaminated_pt(q + "\n" + "\n".join(opts)):
        return None, "8grama_provas_pt"
    en = str(rec.get("question_en") or "") + "\n" + "\n".join(str(o) for o in (rec.get("options_en") or []))
    if contaminated_en(en):
        return None, "8grama_mmlu_prox_en"
    prompt = build_prompt({"question": q, "options": opts})
    if n_tokens(prompt) > max_prompt_tokens:
        return None, "prompt_longo"
    seen.add(rec["id"])
    return {"id": f"mcq:{rec['id']}", "kind": "mcq", "prompt": prompt, "gold": gold, "n_options": len(opts),
            "source": rec.get("source"), "category": rec.get("category")}, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--base", default="data/grpo_prompts.jsonl", help="de onde vem a matemática (as linhas kind=math); a múltipla escolha antiga, de gabarito fraco, não entra")
    ap.add_argument("--out", default="data/grpo_prompts_gold.jsonl")
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    a = ap.parse_args()

    sys.path.insert(0, "scripts")
    import build_dataset as bd

    from lumi_ft.data.decontaminate import NgramIndex
    from lumi_ft.eval import mcq as M
    from lumi_ft.modeling.load import load_tokenizer

    idx, _ = bd.build_decontam_index(8)
    val = NgramIndex(8)
    for line in Path("data/val_mcq.jsonl").read_text().splitlines():
        m = json.loads(line)
        val.add(m["question"] + "\n" + "\n".join(m["options"]))
    for line in Path("data/val_gsm8k.jsonl").read_text().splitlines():
        v = json.loads(line)
        val.add(v["question"] + "\n" + v["answer"])
    # o original em inglês também não pode estar no MMLU-ProX (teste e validação), que tem a mesma coleção das provas em português
    from datasets import load_dataset

    en_idx = NgramIndex(8)
    for split in ("test", "validation"):
        for r in load_dataset("li-lab/MMLU-ProX", "en", split=split):
            opts = [r[f"option_{i}"] for i in range(10) if r.get(f"option_{i}") not in (None, "", "N/A")]
            en_idx.add(str(r["question"]) + "\n" + "\n".join(str(o) for o in opts))
    tok = load_tokenizer()

    recs, bad = read_jsonl(a.questions)
    seen, rows, why = set(), [], Counter()
    for r in recs:
        row, reason = convert(r, seen, M.build_prompt, lambda t: idx.hits(t) or val.hits(t), en_idx.hits,
                              lambda t: len(tok(t, add_special_tokens=False)["input_ids"]), a.max_prompt_tokens)
        if row:
            rows.append(row)
        else:
            why[reason] += 1
    base, _ = read_jsonl(a.base)
    math = [r for r in base if r["kind"] == "math"]
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in math + rows))
    rep = {"linhas_lidas": len(recs), "linhas_quebradas": bad, "mcq_aceitas": len(rows), "math": len(math), "saidas_por_motivo": dict(why),
           "mcq_por_fonte": dict(Counter(r["source"] for r in rows)), "mcq_por_categoria": dict(Counter(r["category"] for r in rows).most_common()),
           "mcq_gabarito": dict(sorted(Counter(r["gold"] for r in rows).items())),
           "mcq_n_alternativas": dict(sorted(Counter(r["n_options"] for r in rows).items())), "out": a.out}
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
