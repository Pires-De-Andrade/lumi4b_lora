"""Questões de múltipla escolha de nvidia/Nemotron-SFT-Multilingual-v2 (português do Brasil) para treino.

Cada linha traz: prompt do usuário em português (instrução + enunciado + alternativas `A. ...`), resposta final em português
terminando em `Resposta: \\boxed{X}` e o raciocínio (`reasoning_content`) em inglês, no estilo do GPT-OSS ("We need to ..."),
que é o do Nemotron 3 e, pelo que medimos, o do Lumi. Aqui só se separa enunciado e alternativas e se confere o gabarito; o
raciocínio é traduzido depois (`scripts/translate_traces.py`, registros `ready_pt`).
"""

from __future__ import annotations

import re
from collections import Counter

LETTERS = "ABCDEFGHIJ"
BOXED = re.compile(r"\\boxed\{\s*([A-J])\s*\}")
FINAL_LINE = re.compile(r"\s*Resposta:\s*\\boxed\{\s*[A-J]\s*\}\s*$")


def parse_prompt(u: str) -> tuple[str, list[str]] | None:
    """(enunciado, alternativas) de um prompt `instrução\\n\\nenunciado\\n\\nA. ...\\nB. ...`; None se não achar ao menos 4
    alternativas em ordem (`A.` ou `A)` no começo da linha). Linhas que não começam uma alternativa continuam a anterior."""
    head, sep, body = u.partition("\n\n")
    if not sep or not re.search(r"(?i)múltipla escolha", head):
        return None
    lines = body.split("\n")
    for i, ln in enumerate(lines):
        m = re.match(r"^A([\.\)])\s+(.*)$", ln)
        if not m:
            continue
        mark = re.escape(m.group(1))
        opts = [m.group(2)]
        for nxt in lines[i + 1:]:
            mm = re.match(rf"^([A-J]){mark}\s+(.*)$", nxt)
            if mm and len(opts) < len(LETTERS) and mm.group(1) == LETTERS[len(opts)]:
                opts.append(mm.group(2))
            else:
                opts[-1] += "\n" + nxt
        opts = [o.strip() for o in opts]
        stem = "\n".join(lines[:i]).strip()
        if len(opts) >= 4 and stem and all(opts):
            return stem, opts
    return None


def select(rows, *, max_words: int = 1000) -> tuple[list[dict], dict]:
    """`rows`: linhas do jsonl ({messages, uuid, metadata}). Fica com as de múltipla escolha com gabarito válido, raciocínio
    de até `max_words` palavras e enunciado ainda não visto."""
    st, out, seen = Counter(), [], set()
    for r in rows:
        st["lidas"] += 1
        msgs = r["messages"]
        a = msgs[-1]
        parsed = parse_prompt(msgs[0]["content"])
        if parsed is None:
            st["sem_alternativas_ou_nao_mcq"] += 1
            continue
        stem, opts = parsed
        letters = BOXED.findall(a["content"])
        if not letters or LETTERS.index(letters[-1]) >= len(opts):
            st["sem_boxed_valido"] += 1
            continue
        thought = (a.get("reasoning_content") or "").strip()
        words = len(thought.split())
        if not thought or words > max_words:
            st["traco_vazio_ou_longo"] += 1
            continue
        if stem in seen:
            st["enunciado_repetido"] += 1
            continue
        seen.add(stem)
        final = BOXED.sub(r"\1", FINAL_LINE.sub("", a["content"].strip()).strip())
        meta = r.get("metadata")
        cat = meta.get("source category") if isinstance(meta, dict) else (re.search(r"'source category': '([^']*)'", str(meta)) or [None, "?"])[1]
        out.append({"id": r["uuid"], "problem_id": f"nemotron_pt:{r['uuid']}", "question": stem, "options": opts, "letter": letters[-1],
                    "thought": thought, "final": final, "words": words, "category": cat, "source": "nemotron_pt", "ready_pt": True})
        st["mantidas"] += 1
    return out, dict(st)
