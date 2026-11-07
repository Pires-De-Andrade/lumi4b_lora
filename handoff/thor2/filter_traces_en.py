#!/usr/bin/env python3
"""Filtra os traços do Nemotron: fica só o que fecha `</think>` uma vez, termina sozinho e acerta a referência.

Matemática: a resposta tem `#### N` igual à referência (leitura numérica brasileira ou inglesa). Múltipla escolha: a última linha da
resposta é `Resposta: <letra>` e a letra é a da referência. O idioma do traço é só registrado (campo `lang`): o esperado é inglês, que
será traduzido do outro lado. Saídas em --outdir: kept.jsonl, report.json e samples.md (12 exemplos para ler).
Uso: python filter_traces_en.py --traces out/traces.jsonl --prompts data/prompts.jsonl --outdir out
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

_PT = frozenset("de que não uma para com os as na no por mais mas foi ao ele das tem seu sua ou ser quando muito nos já "
                "está também só pelo pela até isso ela entre era depois sem mesmo ter quem nas esse eles você então "
                "cada porque são vezes temos".split())
_EN = frozenset("the of and to is that it was for are with his they be at have this from or had by but what there we can "
                "were your when which their if will about then them would so these her has he you i each first".split())
_WORD = re.compile(r"[a-zà-ÿ]+")
_STRICT_NUM = re.compile(r"#### (\-?[0-9\.\,]+)")
_STRICT_LETTER = re.compile(r"[\s\*>`]*Resposta:\s*\**\s*([A-J])\s*[\.\*`]*\s*")


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


def check_answer(kind: str, answer: str, ref: str) -> str | None:
    """None se a resposta serve; senão o motivo."""
    if kind == "math":
        m = _STRICT_NUM.search(answer)
        if not m:
            return "sem_####_na_resposta"
        got, want = to_dec(m.group(1)), to_dec(ref)
        return None if got is not None and want is not None and got == want else "resposta_diferente_da_referência"
    last = answer.strip().splitlines()[-1] if answer.strip() else ""
    m = _STRICT_LETTER.fullmatch(last)
    if not m:
        return "sem_Resposta_letra_na_ultima_linha"
    return None if m.group(1) == ref else "resposta_diferente_da_referência"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="out/traces.jsonl")
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--max-tokens-keep", type=int, default=2500, help="descarta traços com mais tokens que isso")
    a = ap.parse_args()
    prompts = {}
    for line in Path(a.prompts).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            prompts[r["id"]] = r
    rows = [json.loads(line) for line in Path(a.traces).read_text().splitlines() if line.strip()]
    funnel, by_kind = Counter(), {}
    lang, fin, kept = Counter(), Counter(), []
    for r in rows:
        p = prompts[r["id"]]
        kind = p["kind"]
        by_kind.setdefault(kind, Counter())["total"] += 1
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
        lg = trace_language(thought)
        lang[lg] += 1
        why = check_answer(kind, answer, r["ref"])
        if why:
            funnel[why] += 1
            continue
        if (r.get("completion_tokens") or 0) > a.max_tokens_keep:
            funnel["longo_demais"] += 1
            continue
        funnel["mantidos"] += 1
        by_kind[kind]["mantidos"] += 1
        kept.append({"id": r["id"], "kind": kind, "source": p["source"], "question": p["question"], "options": p.get("options"),
                     "thought": thought, "answer": answer, "ref": r["ref"], "lang": lg, "completion_tokens": r.get("completion_tokens")})
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kept.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept))
    toks = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
    report = {"linhas": len(rows), "funil": dict(funnel), "por_tipo": {k: dict(v) for k, v in by_kind.items()},
              "idioma_do_traço_entre_os_que_fecham": dict(lang), "idioma_dos_mantidos": dict(Counter(k["lang"] for k in kept)),
              "finish_reason": dict(fin), "rendimento": round(funnel["mantidos"] / max(1, funnel["gerados"]), 4),
              "tokens_gerados": {"média": round(statistics.fmean(toks), 1) if toks else None,
                                 "mediana": statistics.median(toks) if toks else None, "total": sum(toks)},
              "tokens_dos_mantidos": {"mediana": statistics.median([k["completion_tokens"] or 0 for k in kept]) if kept else None}}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    random.seed(0)
    pick = [k for k in kept if k["kind"] == "math"][:600]
    pick = random.sample(pick, min(6, len(pick))) + random.sample([k for k in kept if k["kind"] == "mcq"], min(6, sum(k["kind"] == "mcq" for k in kept)))
    md = ["# Amostra dos traços mantidos\n"]
    for k in pick:
        opts = "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(k["options"] or []))
        md += [f"\n---\n\n## {k['id']} ({k['kind']}, ref {k['ref']}, idioma do traço {k['lang']}, {k['completion_tokens']} tokens)\n",
               f"**Problema:** {k['question']}\n", opts, "\n**Traço:**\n", "```text", k["thought"][:2500], "```", f"\n**Resposta:**\n\n{k['answer']}\n"]
    (out / "samples.md").write_text("\n".join(md))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
