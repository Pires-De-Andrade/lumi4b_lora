#!/usr/bin/env python3
"""Traduz para o português do Brasil só o enunciado e as alternativas das questões de data/prompts.jsonl (sem traços de raciocínio).

Serve para montar prompts de aprendizado por reforço com **gabarito oficial** (campo `ref`), de todas as questões, e não só das que
o professor acertou. Reaproveita sem alterar o `translate_thor.py` (mesmo prompt de tradução, T=0, sem raciocínio) e as mesmas
conferências por pedaço (terminou, tamanho, números preservados, não respondeu a questão, as letras das alternativas voltaram em ordem).

Saída: out/questions_pt.jsonl, uma linha por questão: id, source, category, kind, ref, question_en, options_en, question_pt,
options_pt, verdict ("ok" ou o motivo da recusa), mais out/questions_pt.jsonl.report.json. Retomável: ids já gravados são pulados.

Uso (na pasta do pacote, com o servidor `teacher` de pé; as mesmas flags do translate_thor.py):
  python3 translate_questions.py --prompts data/prompts.jsonl --out out/questions_pt.jsonl \\
      --tokenizer $TOK --model-name teacher --concurrency 96
"""
import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import translate_thor as T


def translate_question(rec: dict, tok, a, post_fn=T.post) -> dict:
    """Traduz enunciado e alternativas de uma questão. `verdict` = "ok" ou "q:<motivo>"."""
    out = {k: rec.get(k) for k in ("id", "source", "category", "kind", "ref")}
    out.update(question_en=rec["question"], options_en=rec["options"])
    src = T.question_block(rec["question"], rec["options"])
    text, finished, n_tokens, err = T.translate_piece(src, tok, a, post_fn)
    out["completion_tokens"] = n_tokens
    why = "erro_de_requisicao" if err else T.check_piece(src, text, finished)
    if why:
        return {**out, "verdict": f"q:{why}", "piece": text[:400]}
    parsed = T.split_question_block(text, len(rec["options"]))
    if parsed is None:
        return {**out, "verdict": "q:alternativas_nao_voltaram", "piece": text[:400]}
    out["question_pt"], out["options_pt"] = parsed
    out["verdict"] = "ok"
    return out


def run(a, tok, post_fn=T.post) -> dict:
    recs, seen = [], set()
    for line in Path(a.prompts).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["id"] not in seen and r.get("kind") == "mcq" and r.get("options") and r.get("ref"):
                seen.add(r["id"])
                recs.append(r)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    todo = [r for r in recs if r["id"] not in done]
    if a.limit:
        todo = todo[a.start: a.start + a.limit] if a.start else todo[: a.limit]
    elif a.start:
        todo = todo[a.start:]
    print(f"{len(recs)} questões, {len(done)} já feitas, {len(todo)} a fazer", flush=True)
    t0, ntok, stats = time.time(), 0, Counter()
    deadline = t0 + a.time_budget_min * 60 if a.time_budget_min else None
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex, out.open("a") as fout:
        for i in range(0, len(todo), a.block):
            if deadline and time.time() > deadline:
                print("tempo esgotado; o que já foi gravado fica em", out, flush=True)
                break
            block = todo[i: i + a.block]
            for rec in ex.map(lambda r: translate_question(r, tok, a, post_fn), block):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                ntok += rec.get("completion_tokens") or 0
                stats[rec["verdict"].split(":")[-1]] += 1
            el = time.time() - t0
            print(f"[{el / 60:5.1f} min] {i + len(block)}/{len(todo)} | {ntok / max(el, 1e-9):.0f} tokens/s | {dict(stats)}", flush=True)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    rep = {"questoes": len(rows), "ok": sum(r["verdict"] == "ok" for r in rows),
           "aproveitamento": round(sum(r["verdict"] == "ok" for r in rows) / max(1, len(rows)), 4),
           "por_veredito": dict(Counter(r["verdict"].split(":")[-1] for r in rows)),
           "ok_por_fonte": dict(Counter(r["source"] for r in rows if r["verdict"] == "ok")),
           "ok_por_categoria": dict(Counter(r["category"] for r in rows if r["verdict"] == "ok")),
           "esta_execucao": {"minutos": round((time.time() - t0) / 60, 1), "tokens_gerados": ntok}}
    Path(str(out) + ".report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--out", default="out/questions_pt.jsonl")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1/completions")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--block", type=int, default=256, help="questões por bloco (grava ao fim de cada bloco)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--time-budget-min", type=float, default=0)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--api-key", default="")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    rep = run(a, tok)
    return 0 if rep["questoes"] else 1


if __name__ == "__main__":
    sys.exit(main())
