#!/usr/bin/env python3
"""Filtra e resume os traços gerados: fica só o que fecha `</think>`, em português, com a resposta certa.

Saídas (em --outdir): kept.jsonl (para treino), report.json e samples.md (12 exemplos para ler).
Uso: python filter_traces.py --traces out/traces.jsonl --prompts data/prompts.jsonl --outdir out
"""
import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

# Cópia de lumi_ft/eval/shape.py (heurística de idioma por palavras funcionais), para o pacote ser independente.
_PT = frozenset("de que não uma para com os as na no por mais mas foi ao ele das tem seu sua ou ser quando muito nos já "
                "está também só pelo pela até isso ela entre era depois sem mesmo ter quem nas esse eles você então "
                "cada porque são vezes temos".split())
_EN = frozenset("the of and to is that it was for are with his they be at have this from or had by but what there we can "
                "were your when which their if will about then them would so these her has he you i each first".split())
_WORD = re.compile(r"[a-zà-ÿ]+")
_STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")


def trace_language(text: str, min_hits: int = 3, ratio: float = 2.0) -> str:
    words = _WORD.findall(text.lower())
    pt, en = sum(w in _PT for w in words), sum(w in _EN for w in words)
    if pt + en < min_hits:
        return "unk"
    if pt >= ratio * en:
        return "pt"
    if en >= ratio * pt:
        return "en"
    return "mixed"


def to_dec(s: str):
    """Número em formato brasileiro ou inglês: `70.000` = 70000, `3,5` = 3,5, `1.234,56` = 1234,56."""
    s = re.sub(r"\.$", "", s.replace("$", "").strip())
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def split(text: str):
    """(traço, resposta) se houver exatamente um `</think>`, senão None."""
    if text.count("</think>") != 1:
        return None
    head, _, tail = text.partition("</think>")
    return head.removeprefix("<think>").strip(), tail.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="out/traces.jsonl")
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--max-tokens-keep", type=int, default=3000, help="descarta traços com mais tokens que isso")
    a = ap.parse_args()
    questions = {}
    for line in Path(a.prompts).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            questions[r["id"]] = r["question"]
    rows = [json.loads(line) for line in Path(a.traces).read_text().splitlines() if line.strip()]
    funnel = Counter()
    kept, by_source = [], {}
    lang, fin = Counter(), Counter()
    for r in rows:
        src = r["source"]
        by_source.setdefault(src, Counter())["total"] += 1
        if "error" in r:
            funnel["erro_de_requisição"] += 1
            continue
        funnel["gerados"] += 1
        fin[r.get("finish_reason")] += 1
        if r.get("finish_reason") != "stop":
            funnel["truncado"] += 1
            continue
        sp = split(r["text"])
        if sp is None:
            funnel["sem_um_unico_</think>"] += 1
            continue
        thought, answer = sp
        if not thought or not answer:
            funnel["traço_ou_resposta_vazios"] += 1
            continue
        lang[trace_language(thought)] += 1
        m = _STRICT.search(answer)
        if not m:
            funnel["sem_####_na_resposta"] += 1
            continue
        got, want = to_dec(m.group(1)), to_dec(r["ref"])
        if got is None or want is None or got != want:
            funnel["resposta_diferente_da_referência"] += 1
            continue
        if trace_language(thought) != "pt":
            funnel["traço_não_está_em_português"] += 1
            continue
        if (r.get("completion_tokens") or 0) > a.max_tokens_keep:
            funnel["longo_demais"] += 1
            continue
        funnel["mantidos"] += 1
        by_source[src]["mantidos"] += 1
        kept.append({"id": r["id"], "source": src, "question": questions[r["id"]], "thought": thought, "answer": answer,
                     "ref": r["ref"], "completion_tokens": r.get("completion_tokens")})
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kept.jsonl").write_text("\n".join(json.dumps(k, ensure_ascii=False) for k in kept) + ("\n" if kept else ""))
    toks = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
    report = {"linhas": len(rows), "funil": dict(funnel), "por_fonte": {k: dict(v) for k, v in by_source.items()},
              "idioma_do_traço_entre_os_que_fecham": dict(lang), "finish_reason": dict(fin),
              "rendimento": round(funnel["mantidos"] / max(1, funnel["gerados"]), 4),
              "tokens_gerados": {"média": round(statistics.fmean(toks), 1) if toks else None,
                                 "mediana": statistics.median(toks) if toks else None, "total": sum(toks)},
              "tokens_dos_mantidos": {"mediana": statistics.median([k["completion_tokens"] or 0 for k in kept]) if kept else None}}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    random.seed(0)
    md = ["# Amostra dos traços mantidos\n"]
    for k in random.sample(kept, min(12, len(kept))):
        md += [f"\n---\n\n## {k['id']} ({k['source']}, ref {k['ref']}, {k['completion_tokens']} tokens)\n", f"**Problema:** {k['question']}\n",
               "**Traço:**\n", "```text", k["thought"][:2500], "```", f"\n**Resposta:**\n\n{k['answer']}\n"]
    (out / "samples.md").write_text("\n".join(md))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
